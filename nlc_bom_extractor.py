#!/usr/bin/env python3
"""
NLC BOM Extractor
-----------------
Extracts Bill of Materials data from Nonlinearcircuits PDF files using Tabula.

Usage:
    python nlc_bom_extractor.py /path/to/bom/directory
    python nlc_bom_extractor.py /path/to/bom/directory --output json
    python nlc_bom_extractor.py /path/to/bom/directory --output excel --combine
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import tabula


def extract_tables_from_pdf(pdf_path: Path, pages: str = "all") -> list[pd.DataFrame]:
    """
    Extract all tables from a PDF using Tabula.
    
    Args:
        pdf_path: Path to the PDF file
        pages: Which pages to extract ("all", "1", "1-3", etc.)
    
    Returns:
        List of DataFrames, one per table found
    """
    try:
        # Try lattice mode first (for tables with visible borders)
        tables = tabula.read_pdf(
            str(pdf_path),
            pages=pages,
            lattice=True,
            pandas_options={"header": None}
        )
        
        # If no tables found, try stream mode (for borderless tables)
        if not tables or all(df.empty for df in tables):
            tables = tabula.read_pdf(
                str(pdf_path),
                pages=pages,
                stream=True,
                pandas_options={"header": None}
            )
        
        return tables if tables else []
    
    except Exception as e:
        print(f"  Warning: Error extracting from {pdf_path.name}: {e}")
        return []


def clean_bom_dataframe(df: pd.DataFrame, min_cols: int = 2, min_rows: int = 3) -> pd.DataFrame:
    """
    Clean and normalize a BOM DataFrame.
    
    - Attempts to identify header row
    - Removes empty rows/columns
    - Standardizes column names
    - Filters out noise tables
    - Cleans cell content (newlines, whitespace)
    """
    if df.empty:
        return df
    
    # Remove completely empty rows and columns
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)
    
    if df.empty:
        return df
    
    # Filter out tables that are too small (likely noise)
    if len(df.columns) < min_cols or len(df) < min_rows:
        return pd.DataFrame()
    
    # Try to identify header row (look for common BOM headers)
    header_keywords = [
        "qty", "quantity", "value", "part", "component", "ref", "reference",
        "designator", "description", "footprint", "package", "mouser", "digikey"
    ]
    
    header_row_idx = None
    for idx, row in df.iterrows():
        row_str = " ".join(str(v).lower() for v in row.values if pd.notna(v))
        if any(kw in row_str for kw in header_keywords):
            # Validate this looks like a real header row
            # Reject if any cell is too long (likely data, not header)
            max_cell_len = max(len(str(v)) for v in row.values if pd.notna(v))
            # Reject if cells contain newlines (multi-line content = data)
            has_newlines = any("\r" in str(v) or "\n" in str(v) for v in row.values if pd.notna(v))
            # Reject if cells contain decimal numbers (likely quantities)
            has_decimals = any("." in str(v) and str(v).replace(".", "").isdigit() 
                             for v in row.values if pd.notna(v))
            
            if max_cell_len < 50 and not has_newlines and not has_decimals:
                header_row_idx = idx
                break
    
    # If we found a header row, use it
    if header_row_idx is not None:
        new_headers = df.loc[header_row_idx].fillna("").astype(str).tolist()
        # Make headers unique
        seen = {}
        unique_headers = []
        for h in new_headers:
            h = h.strip()
            if h in seen:
                seen[h] += 1
                unique_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique_headers.append(h)
        
        df.columns = unique_headers
        df = df.loc[header_row_idx + 1:].reset_index(drop=True)
    else:
        # No BOM header found - try to infer if it's a continuation table
        # Check if columns look like VALUE/QTY/DETAILS pattern
        if len(df.columns) >= 2:
            # Look at the data to guess the structure
            first_col_samples = df.iloc[:, 0].astype(str).tolist()[:5]
            
            # If first column has component-like values, treat as BOM continuation
            component_patterns = ["pF", "nF", "uF", "µF", "ohm", "Ω", "k", "M", 
                                 "pot", "socket", "jack", "connector", "LED", 
                                 "TL0", "LM", "BC", "diode", "cap", "pin"]
            
            looks_like_bom = any(
                any(pat.lower() in str(val).lower() for pat in component_patterns)
                for val in first_col_samples
            )
            
            if looks_like_bom:
                # Assign generic column names based on count
                if len(df.columns) == 2:
                    df.columns = ["VALUE", "DETAILS"]
                elif len(df.columns) == 3:
                    df.columns = ["VALUE", "QUANTITY", "DETAILS"]
                else:
                    df.columns = ["VALUE", "QUANTITY", "DETAILS"] + [f"col_{i}" for i in range(3, len(df.columns))]
            else:
                # Not a BOM table
                return pd.DataFrame()
    
    # Remove empty-named columns
    df = df.loc[:, df.columns.astype(str).str.strip() != ""]
    
    # Remove rows that look like repeated headers
    if len(df) > 0:
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.lower().isin(header_keywords)]
    
    # Clean cell contents
    for col in df.columns:
        df[col] = df[col].astype(str).apply(clean_cell_value)
    
    # Remove rows where all values are empty or "nan"
    df = df[~df.apply(lambda row: all(v in ["", "nan"] for v in row), axis=1)]
    
    return df.reset_index(drop=True)


def clean_cell_value(value: str) -> str:
    """Clean individual cell values."""
    if pd.isna(value) or value == "nan":
        return ""
    
    # Replace various newline characters with space
    value = value.replace("\r", " ").replace("\n", " ")
    
    # Collapse multiple spaces
    while "  " in value:
        value = value.replace("  ", " ")
    
    return value.strip()


def extract_part_numbers(details: str) -> dict:
    """Extract supplier part numbers from the details field."""
    parts = {}
    
    # Common patterns in NLC BOMs
    patterns = [
        (r"[Mm]ouser\s*(?:[Pp]art\s*)?(?:[Nn]o)?:?\s*([A-Z0-9\-]+)", "mouser"),
        (r"[Tt]ayda:?\s*([A-Z0-9\-]+)", "tayda"),
        (r"[Dd]igi-?[Kk]ey:?\s*([A-Z0-9\-]+)", "digikey"),
    ]
    
    import re
    for pattern, supplier in patterns:
        match = re.search(pattern, details)
        if match:
            parts[supplier] = match.group(1)
    
    return parts


def expand_designators(designator_str: str) -> list[str]:
    """
    Expand a designator string into individual designators.
    e.g., "R1, R2, R3" -> ["R1", "R2", "R3"]
    e.g., "C1-C5" -> ["C1", "C2", "C3", "C4", "C5"]
    """
    import re
    
    designators = []
    # Split on comma and clean
    parts = [p.strip() for p in designator_str.split(",") if p.strip()]
    
    for part in parts:
        # Check for range pattern like "C1-C5" or "R10-R15"
        range_match = re.match(r"([A-Z]+)(\d+)\s*[-–]\s*([A-Z]+)?(\d+)", part, re.IGNORECASE)
        if range_match:
            prefix = range_match.group(1)
            start = int(range_match.group(2))
            end = int(range_match.group(4))
            for i in range(start, end + 1):
                designators.append(f"{prefix}{i}")
        else:
            # Single designator, remove any extra notes in parentheses
            clean = re.sub(r"\s*\([^)]*\)\s*", "", part).strip()
            if clean:
                designators.append(clean)
    
    return designators


def count_designators(designator_str: str) -> int:
    """Count the number of components from a designator string."""
    return len(expand_designators(designator_str))


def parse_embedded_quantity(component_str: str) -> tuple[str, str]:
    """
    Parse quantity from component strings like "TL072 (5)" or "1k (10)".
    Returns (component_name, quantity).
    """
    import re
    
    # Match patterns like "TL072 (5)" or "100nF/104 (7)"
    match = re.match(r"^(.+?)\s*\((\d+)\)\s*$", component_str.strip())
    if match:
        return match.group(1).strip(), match.group(2)
    
    return component_str, ""


def unpack_side_by_side_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle tables with side-by-side columns like:
    component | quantity | notes | component_1 | quantity_1 | notes_1
    
    Unpacks them into a single normalized table.
    """
    # Check if we have duplicate column patterns (component_1, quantity_1, etc.)
    cols = df.columns.tolist()
    
    # Look for patterns like col, col_1 or component, component_1
    base_cols = []
    suffix_groups = {}
    
    for col in cols:
        # Check for _N suffix
        import re
        match = re.match(r"^(.+?)_(\d+)$", col)
        if match:
            base = match.group(1)
            suffix = int(match.group(2))
            if suffix not in suffix_groups:
                suffix_groups[suffix] = []
            suffix_groups[suffix].append((base, col))
        else:
            base_cols.append(col)
    
    # If we found side-by-side columns, unpack them
    if suffix_groups:
        all_rows = []
        
        for _, row in df.iterrows():
            # First, add the base columns as a row
            base_row = {col: row[col] for col in base_cols}
            if any(str(v).strip() and str(v) != "nan" for v in base_row.values()):
                all_rows.append(base_row)
            
            # Then add each suffix group as additional rows
            for suffix in sorted(suffix_groups.keys()):
                suffix_row = {}
                for base, full_col in suffix_groups[suffix]:
                    # Map back to base column name
                    suffix_row[base] = row[full_col]
                
                if any(str(v).strip() and str(v) != "nan" for v in suffix_row.values()):
                    all_rows.append(suffix_row)
        
        if all_rows:
            return pd.DataFrame(all_rows)
    
    return df


def unpack_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle variant-style BOMs where columns represent different build options.
    E.g., columns like: component | torpor | apathy | inertia
    
    These get unpacked so each variant becomes rows with VALUE/QUANTITY/DETAILS.
    """
    cols = df.columns.tolist()
    
    # Detect variant tables - first column is designator (C1, R1, etc.), 
    # other columns are variant names with component values
    first_col = cols[0].lower() if cols else ""
    
    # Check if first column looks like a designator column
    if len(df) > 0:
        first_col_vals = df.iloc[:, 0].astype(str).tolist()
        import re
        designator_pattern = re.compile(r'^[A-Z]+\d+$', re.I)
        designator_count = sum(1 for v in first_col_vals if designator_pattern.match(v.strip()))
        
        # If most values in first column look like designators (C1, R1, U1, etc.)
        if designator_count > len(first_col_vals) * 0.5 and len(cols) > 2:
            # This is likely a variant table - unpack it
            all_rows = []
            variant_cols = cols[1:]  # All columns except the designator column
            
            for _, row in df.iterrows():
                designator = str(row.iloc[0])
                
                for variant_col in variant_cols:
                    value = str(row[variant_col])
                    if value and value.lower() not in ["nan", "", "nothing!"]:
                        all_rows.append({
                            "VALUE": value,
                            "QUANTITY": "1",
                            "DETAILS": f"Designator: {designator}, Variant: {variant_col}"
                        })
            
            if all_rows:
                return pd.DataFrame(all_rows)
    
    return df


def normalize_bom_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize various BOM formats to a standard structure.
    Handles:
    - Embedded quantities like "TL072 (5)"
    - Side-by-side column layouts
    - Variant tables (torpor/apathy/inertia style)
    - Various column naming conventions
    """
    # First, unpack side-by-side tables
    df = unpack_side_by_side_table(df)
    
    # Check for and unpack variant tables
    df = unpack_variant_table(df)
    
    # Check for "COMPONENT (QUANTITY)" style columns
    cols_lower = {c: c.lower() for c in df.columns}
    
    for col in df.columns:
        col_lower = cols_lower[col]
        if "component" in col_lower and "quantity" in col_lower:
            # This column has embedded quantities - parse them out
            components = []
            quantities = []
            
            for val in df[col]:
                comp, qty = parse_embedded_quantity(str(val))
                components.append(comp)
                quantities.append(qty)
            
            # Replace the column with just component, add quantity column
            df[col] = components
            df.rename(columns={col: "VALUE"}, inplace=True)
            
            # Insert quantity column if it doesn't exist
            if "QUANTITY" not in df.columns and "quantity" not in [c.lower() for c in df.columns]:
                # Find the position after the component column
                col_idx = df.columns.get_loc("VALUE")
                df.insert(col_idx + 1, "QUANTITY", quantities)
    
    # Normalize column names to standard: VALUE, QUANTITY, DETAILS
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Map to VALUE
        if col_lower in ["value", "component", "part", "item"]:
            column_mapping[col] = "VALUE"
        # Map to QUANTITY  
        elif col_lower in ["quantity", "qty", "count", "amount"]:
            column_mapping[col] = "QUANTITY"
        # Map to DETAILS
        elif col_lower in ["details", "notes", "description", "info", "comments"]:
            column_mapping[col] = "DETAILS"
    
    # Apply the mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Ensure we have the standard columns (add empty ones if missing)
    for std_col in ["VALUE", "QUANTITY", "DETAILS"]:
        if std_col not in df.columns:
            df[std_col] = ""
    
    # Reorder to put standard columns first
    std_cols = ["VALUE", "QUANTITY", "DETAILS"]
    other_cols = [c for c in df.columns if c not in std_cols]
    df = df[std_cols + other_cols]
    
    # If VALUE column is empty but we have other columns with data, 
    # try to use the first non-standard column as VALUE
    if df["VALUE"].astype(str).str.strip().eq("").all() or df["VALUE"].isna().all():
        for col in other_cols:
            if not df[col].astype(str).str.strip().eq("").all():
                df["VALUE"] = df[col]
                df = df.drop(columns=[col])
                break
    
    # Clean up: remove rows where VALUE is empty/nan
    df = df[df["VALUE"].astype(str).str.strip().ne("")]
    df = df[df["VALUE"].astype(str).ne("nan")]
    
    return df


def process_single_pdf(pdf_path: Path) -> dict:
    """
    Process a single BOM PDF and return extracted data.
    
    Returns:
        Dictionary with module name, tables, and metadata
    """
    module_name = pdf_path.stem
    
    # Handle common NLC naming patterns
    # e.g., "NLC - 4seq BOM.pdf" -> "4seq"
    clean_name = module_name
    for prefix in ["NLC - ", "NLC-", "NLC ", "nlc - ", "nlc-", "nlc "]:
        if clean_name.lower().startswith(prefix.lower()):
            clean_name = clean_name[len(prefix):]
    for suffix in [" BOM", " bom", "_BOM", "_bom", "-BOM", "-bom"]:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
    
    print(f"Processing: {module_name}")
    
    tables = extract_tables_from_pdf(pdf_path)
    cleaned_tables = []
    total_rows = 0
    
    for i, df in enumerate(tables):
        cleaned = clean_bom_dataframe(df)
        if not cleaned.empty:
            # Normalize the table format
            cleaned = normalize_bom_table(cleaned)
            cleaned_tables.append(cleaned)
            total_rows += len(cleaned)
    
    print(f"  Found {len(cleaned_tables)} table(s), {total_rows} total rows")
    
    return {
        "filename": pdf_path.name,
        "module_name": clean_name,
        "tables": cleaned_tables,
        "table_count": len(cleaned_tables),
        "total_rows": total_rows
    }


def scan_directory(directory: Path, recursive: bool = False) -> list[Path]:
    """Find all PDF files in a directory."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = list(directory.glob(pattern))
    
    # Also check for .PDF extension
    pdfs.extend(directory.glob(pattern.replace(".pdf", ".PDF")))
    
    return sorted(set(pdfs))


def export_to_csv(results: list[dict], output_dir: Path, combine: bool = False):
    """Export results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if combine:
        # Combine all tables into one CSV with standardized columns
        all_dfs = []
        for result in results:
            for df in result["tables"]:
                df = df.copy()
                
                # Ensure standard columns exist
                for col in ["VALUE", "QUANTITY", "DETAILS"]:
                    if col not in df.columns:
                        df[col] = ""
                
                # Keep only standard columns plus module info
                df = df[["VALUE", "QUANTITY", "DETAILS"]].copy()
                df.insert(0, "_module", result["module_name"])
                df.insert(1, "_source_file", result["filename"])
                all_dfs.append(df)
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            output_path = output_dir / "all_boms_combined.csv"
            combined.to_csv(output_path, index=False)
            print(f"Saved combined CSV: {output_path}")
    else:
        # One CSV per module
        for result in results:
            if result["tables"]:
                # Combine tables from same PDF
                combined = pd.concat(result["tables"], ignore_index=True)
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in result["module_name"])
                output_path = output_dir / f"{safe_name}.csv"
                combined.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")


def export_to_json(results: list[dict], output_dir: Path, combine: bool = False):
    """Export results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def df_to_records(tables):
        """Convert list of DataFrames to list of record lists."""
        return [df.fillna("").to_dict(orient="records") for df in tables]
    
    if combine:
        output_data = {
            "extracted_at": datetime.now().isoformat(),
            "total_modules": len(results),
            "modules": []
        }
        
        for result in results:
            output_data["modules"].append({
                "module_name": result["module_name"],
                "filename": result["filename"],
                "table_count": result["table_count"],
                "total_rows": result["total_rows"],
                "tables": df_to_records(result["tables"])
            })
        
        output_path = output_dir / "all_boms.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved combined JSON: {output_path}")
    else:
        for result in results:
            if result["tables"]:
                output_data = {
                    "module_name": result["module_name"],
                    "filename": result["filename"],
                    "extracted_at": datetime.now().isoformat(),
                    "tables": df_to_records(result["tables"])
                }
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in result["module_name"])
                output_path = output_dir / f"{safe_name}.json"
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"Saved: {output_path}")


def export_to_excel(results: list[dict], output_dir: Path, combine: bool = False):
    """Export results to Excel files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if combine:
        output_path = output_dir / "all_boms.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = [{
                "Module": r["module_name"],
                "Source File": r["filename"],
                "Tables": r["table_count"],
                "Total Rows": r["total_rows"]
            } for r in results]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            # One sheet per module
            for result in results:
                if result["tables"]:
                    combined = pd.concat(result["tables"], ignore_index=True)
                    # Excel sheet names limited to 31 chars
                    sheet_name = result["module_name"][:31]
                    # Remove invalid characters
                    sheet_name = "".join(c if c not in "[]:*?/\\" else "_" for c in sheet_name)
                    combined.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Saved combined Excel: {output_path}")
    else:
        for result in results:
            if result["tables"]:
                combined = pd.concat(result["tables"], ignore_index=True)
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in result["module_name"])
                output_path = output_dir / f"{safe_name}.xlsx"
                combined.to_excel(output_path, index=False)
                print(f"Saved: {output_path}")


def export_to_mouser(results: list[dict], output_dir: Path, combine: bool = False):
    """
    Export results in Mouser-compatible BOM format.
    Extracts Mouser part numbers and generates a format suitable for upload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_parts = []
    
    for result in results:
        for df in result["tables"]:
            for _, row in df.iterrows():
                # Find the value/part column
                value = ""
                quantity = ""
                details = ""
                designators = ""
                
                for col in df.columns:
                    col_lower = col.lower()
                    cell_value = str(row[col]) if pd.notna(row[col]) else ""
                    
                    if col_lower in ["value", "component", "part"]:
                        value = cell_value
                    elif "value" in col_lower or "component" in col_lower or "part" in col_lower:
                        if not value:  # Don't override if already set
                            value = cell_value
                    elif "designator" in col_lower:
                        # "DESIGNATOR/QUANTITY" column - this is designators, not quantities
                        designators = cell_value
                    elif col_lower in ["qty", "quantity"]:
                        quantity = cell_value
                    elif "qty" in col_lower or "quantity" in col_lower:
                        if not quantity:
                            quantity = cell_value
                    elif "ref" in col_lower:
                        designators = cell_value
                    elif col_lower in ["details", "notes", "description"]:
                        details = cell_value
                    elif "detail" in col_lower or "note" in col_lower or "description" in col_lower:
                        if not details:
                            details = cell_value
                
                # If quantity column doesn't exist or is empty, count designators
                if (not quantity or quantity in ["", "nan"]) and designators:
                    quantity = str(count_designators(designators))
                
                # Extract part numbers
                part_nums = extract_part_numbers(details)
                
                part_entry = {
                    "Module": result["module_name"],
                    "Value": value,
                    "Quantity": quantity,
                    "Designators": designators,
                    "Footprint/Package": "",
                    "Mouser_PN": part_nums.get("mouser", ""),
                    "Tayda_PN": part_nums.get("tayda", ""),
                    "DigiKey_PN": part_nums.get("digikey", ""),
                    "Details": details
                }
                
                # Try to extract footprint from details
                import re
                footprint_match = re.search(r"\b(0805|0603|0402|1206|SOT-23|SOIC|SOD-\d+|sod-\d+)\b", details, re.I)
                if footprint_match:
                    part_entry["Footprint/Package"] = footprint_match.group(1).upper()
                
                all_parts.append(part_entry)
    
    if all_parts:
        mouser_df = pd.DataFrame(all_parts)
        
        if combine:
            output_path = output_dir / "mouser_bom.csv"
            mouser_df.to_csv(output_path, index=False)
            print(f"Saved Mouser BOM: {output_path}")
        else:
            for module_name in mouser_df["Module"].unique():
                module_df = mouser_df[mouser_df["Module"] == module_name].drop(columns=["Module"])
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in module_name)
                output_path = output_dir / f"{safe_name}_mouser.csv"
                module_df.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract BOM data from Nonlinearcircuits PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s ./boms                          # Extract to CSV (default)
    %(prog)s ./boms -o json                  # Extract to JSON
    %(prog)s ./boms -o excel --combine       # Combine all into one Excel file
    %(prog)s ./boms -r                       # Recursive directory scan
        """
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing BOM PDF files"
    )
    
    parser.add_argument(
        "-o", "--output",
        choices=["csv", "json", "excel"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "-d", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./extracted_boms)"
    )
    
    parser.add_argument(
        "-c", "--combine",
        action="store_true",
        help="Combine all results into a single file"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively scan subdirectories"
    )
    
    parser.add_argument(
        "-e", "--expand",
        action="store_true",
        help="Expand designator lists into individual component rows"
    )
    
    parser.add_argument(
        "--mouser",
        action="store_true",
        help="Generate Mouser-compatible BOM format (CSV with part numbers)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    if not args.directory.is_dir():
        print(f"Error: Not a directory: {args.directory}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir or Path("./extracted_boms")
    
    # Find PDFs
    print(f"Scanning: {args.directory}")
    pdfs = scan_directory(args.directory, args.recursive)
    
    if not pdfs:
        print("No PDF files found.")
        sys.exit(0)
    
    print(f"Found {len(pdfs)} PDF file(s)\n")
    
    # Process each PDF
    results = []
    for pdf_path in pdfs:
        result = process_single_pdf(pdf_path)
        results.append(result)
    
    # Export results
    print(f"\nExporting to {args.output.upper()}...")
    
    if args.mouser:
        export_to_mouser(results, output_dir, args.combine)
    elif args.output == "csv":
        export_to_csv(results, output_dir, args.combine)
    elif args.output == "json":
        export_to_json(results, output_dir, args.combine)
    elif args.output == "excel":
        export_to_excel(results, output_dir, args.combine)
    
    # Summary
    total_tables = sum(r["table_count"] for r in results)
    total_rows = sum(r["total_rows"] for r in results)
    print(f"\nDone! Processed {len(results)} files, extracted {total_tables} tables, {total_rows} total rows")


if __name__ == "__main__":
    main()
