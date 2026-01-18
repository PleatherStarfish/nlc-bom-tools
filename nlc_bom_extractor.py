#!/usr/bin/env python3
"""
NLC BOM Extractor
-----------------
Extracts Bill of Materials data from Nonlinearcircuits PDF files using Tabula.

Usage:
    python nlc_bom_extractor.py /path/to/bom/directory
    python nlc_bom_extractor.py /path/to/bom/directory --output json
    python nlc_bom_extractor.py /path/to/bom/directory --output excel --combine
    python nlc_bom_extractor.py /path/to/bom/directory --filename-priority  # recommended
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib

import pandas as pd
import tabula

# Import the module name resolver
from module_name_resolver import ModuleNameResolver


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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
            max_cell_len = max(len(str(v)) for v in row.values if pd.notna(v))
            has_newlines = any("\r" in str(v) or "\n" in str(v) for v in row.values if pd.notna(v))
            has_decimals = any("." in str(v) and str(v).replace(".", "").isdigit() 
                             for v in row.values if pd.notna(v))
            
            if max_cell_len < 50 and not has_newlines and not has_decimals:
                header_row_idx = idx
                break
    
    # If we found a header row, use it
    if header_row_idx is not None:
        new_headers = df.loc[header_row_idx].fillna("").astype(str).tolist()
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
        # No BOM header found - try to infer structure
        if len(df.columns) >= 2:
            first_col_samples = df.iloc[:, 0].astype(str).tolist()[:10]
            second_col_samples = df.iloc[:, 1].astype(str).tolist()[:10] if len(df.columns) > 1 else []
            
            designator_pattern = re.compile(r'^[A-Z]+\d+[A-Z]?$', re.I)
            designator_count = sum(1 for val in first_col_samples if designator_pattern.match(val.strip()))
            looks_like_placement_list = designator_count > len(first_col_samples) * 0.5
            
            component_patterns = ["pF", "nF", "uF", "µF", "ohm", "Ω", "k", "M", 
                                 "pot", "socket", "jack", "connector", "LED", 
                                 "TL0", "LM", "BC", "diode", "cap", "pin"]
            
            looks_like_bom = any(
                any(pat.lower() in str(val).lower() for pat in component_patterns)
                for val in first_col_samples
            )
            
            if looks_like_placement_list:
                if len(df.columns) == 2:
                    df.columns = ["DESIGNATOR", "VALUE"]
                elif len(df.columns) == 3:
                    df.columns = ["DESIGNATOR", "VALUE", "NOTES"]
                else:
                    df.columns = ["DESIGNATOR", "VALUE", "NOTES"] + [f"col_{i}" for i in range(3, len(df.columns))]
            elif looks_like_bom:
                if len(df.columns) == 2:
                    df.columns = ["VALUE", "DETAILS"]
                elif len(df.columns) == 3:
                    df.columns = ["VALUE", "QUANTITY", "DETAILS"]
                else:
                    df.columns = ["VALUE", "QUANTITY", "DETAILS"] + [f"col_{i}" for i in range(3, len(df.columns))]
            else:
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
    
    value = value.replace("\r", " ").replace("\n", " ")
    while "  " in value:
        value = value.replace("  ", " ")
    
    return value.strip()


def extract_part_numbers(details: str) -> dict:
    """Extract supplier part numbers from the details field."""
    parts = {}
    
    patterns = [
        (r"[Mm]ouser\s*(?:[Pp]art\s*)?(?:[Nn]o)?:?\s*([A-Z0-9\-]+)", "mouser"),
        (r"[Tt]ayda:?\s*([A-Z0-9\-]+)", "tayda"),
        (r"[Dd]igi-?[Kk]ey:?\s*([A-Z0-9\-]+)", "digikey"),
    ]
    
    for pattern, supplier in patterns:
        match = re.search(pattern, details)
        if match:
            parts[supplier] = match.group(1)
    
    return parts


def expand_designators(designator_str: str) -> list[str]:
    """Expand a designator string into individual designators."""
    designators = []
    parts = [p.strip() for p in designator_str.split(",") if p.strip()]
    
    for part in parts:
        range_match = re.match(r"([A-Z]+)(\d+)\s*[-–]\s*([A-Z]+)?(\d+)", part, re.IGNORECASE)
        if range_match:
            prefix = range_match.group(1)
            start = int(range_match.group(2))
            end = int(range_match.group(4))
            for i in range(start, end + 1):
                designators.append(f"{prefix}{i}")
        else:
            clean = re.sub(r"\s*\([^)]*\)\s*", "", part).strip()
            if clean:
                designators.append(clean)
    
    return designators


def count_designators(designator_str: str) -> int:
    """Count the number of components from a designator string."""
    return len(expand_designators(designator_str))


def parse_embedded_quantity(component_str: str) -> tuple[str, str]:
    """Parse quantity from component strings like "TL072 (5)"."""
    match = re.match(r"^(.+?)\s*\((\d+)\)\s*$", component_str.strip())
    if match:
        return match.group(1).strip(), match.group(2)
    return component_str, ""


def unpack_side_by_side_table(df: pd.DataFrame) -> pd.DataFrame:
    """Handle tables with side-by-side columns."""
    cols = df.columns.tolist()
    
    # Pattern 1: Look for _N suffix patterns
    base_cols = []
    suffix_groups = {}
    
    for col in cols:
        match = re.match(r"^(.+?)_(\d+)$", col)
        if match:
            base = match.group(1)
            suffix = int(match.group(2))
            if suffix not in suffix_groups:
                suffix_groups[suffix] = []
            suffix_groups[suffix].append((base, col))
        else:
            base_cols.append(col)
    
    if suffix_groups:
        all_rows = []
        
        for _, row in df.iterrows():
            base_row = {col: row[col] for col in base_cols}
            if any(str(v).strip() and str(v) != "nan" for v in base_row.values()):
                all_rows.append(base_row)
            
            for suffix in sorted(suffix_groups.keys()):
                suffix_row = {}
                for base, full_col in suffix_groups[suffix]:
                    suffix_row[base] = row[full_col]
                
                if any(str(v).strip() and str(v) != "nan" for v in suffix_row.values()):
                    all_rows.append(suffix_row)
        
        if all_rows:
            return pd.DataFrame(all_rows)
    
    # Pattern 2: Side-by-side placement lists
    if len(cols) >= 4:
        designator_pattern = re.compile(r'^[A-Z]+\d+[A-Z]?$', re.I)
        
        designator_cols = []
        for i, col in enumerate(cols):
            col_vals = df.iloc[:, i].dropna().astype(str).tolist()
            designator_count = sum(1 for v in col_vals if designator_pattern.match(v.strip()))
            if designator_count > len(col_vals) * 0.3 and len(col_vals) > 0:
                designator_cols.append(i)
        
        if len(designator_cols) >= 2:
            all_rows = []
            
            segments = []
            for desig_col in designator_cols:
                segment_end = min(desig_col + 3, len(cols))
                next_desig = next((d for d in designator_cols if d > desig_col), len(cols))
                segment_end = min(segment_end, next_desig)
                
                if segment_end - desig_col >= 2:
                    segments.append((desig_col, segment_end))
            
            if segments:
                for _, row in df.iterrows():
                    for seg_start, seg_end in segments:
                        designator = str(row.iloc[seg_start]).strip()
                        value = str(row.iloc[seg_start + 1]).strip() if seg_start + 1 < len(row) else ""
                        notes = str(row.iloc[seg_start + 2]).strip() if seg_start + 2 < seg_end else ""
                        
                        if not designator_pattern.match(designator):
                            continue
                        if not value or value.lower() == "nan":
                            continue
                        
                        row_data = {
                            "DESIGNATOR": designator,
                            "VALUE": value,
                            "NOTES": notes if notes and notes.lower() != "nan" else ""
                        }
                        all_rows.append(row_data)
                
                if all_rows:
                    return pd.DataFrame(all_rows)
    
    # Pattern 3: Simple paired columns (for TRUE side-by-side tables only)
    # FIXED: Only apply to 4 or 6 column tables, NOT 3-column tables
    # 3-column tables like [component, quantity, notes] are standard BOM format,
    # not side-by-side paired columns. The third column contains important info
    # like Tayda/Mouser part numbers that must be preserved.
    if len(cols) in [4, 6] and not suffix_groups:
        odd_cols_numeric = True
        even_cols_values = True
        
        for i, col in enumerate(cols):
            col_vals = df.iloc[:, i].astype(str).tolist()
            numeric_count = sum(1 for v in col_vals if re.match(r'^\d+$', v.strip()))
            
            if i % 2 == 1:
                if numeric_count < len(col_vals) * 0.4:
                    odd_cols_numeric = False
            else:
                value_patterns = [r'\d+[pnuμ]?[FfHh]?', r'\d+[kKMmRrΩ]', r'[A-Z]{2}\d', r'LL', r'BC', r'S\d']
                value_count = sum(1 for v in col_vals 
                                if any(re.search(pat, v) for pat in value_patterns))
                if value_count < len(col_vals) * 0.3:
                    even_cols_values = False
        
        if odd_cols_numeric and even_cols_values:
            all_rows = []
            num_pairs = len(cols) // 2
            
            for _, row in df.iterrows():
                for pair_idx in range(num_pairs):
                    value_col = pair_idx * 2
                    qty_col = pair_idx * 2 + 1
                    
                    value = str(row.iloc[value_col]).strip()
                    qty = str(row.iloc[qty_col]).strip()
                    
                    if value and value.lower() != "nan" and qty and qty.lower() != "nan":
                        all_rows.append({
                            "VALUE": value,
                            "QUANTITY": qty,
                            "DETAILS": ""
                        })
            
            if all_rows:
                return pd.DataFrame(all_rows)
    
    return df


def is_placement_list(df: pd.DataFrame) -> bool:
    """Detect if a table is a placement list format."""
    if df.empty or len(df.columns) < 2 or len(df.columns) > 3:
        return False
    
    designator_col = 0
    value_col = 1
    
    first_col_vals = df.iloc[:, designator_col].astype(str).tolist()
    designator_pattern = re.compile(r'^[A-Z]+\d+[A-Z]?$', re.I)
    designator_count = sum(1 for v in first_col_vals 
                          if designator_pattern.match(v.strip()) or 
                             v.strip().lower() in ['part number', 'part', 'designator', 'ref'])
    
    second_col_vals = df.iloc[:, value_col].astype(str).tolist()
    value_patterns = [
        r'^\d+[pnuμ]?[FfHh]?$',
        r'^\d+[kKMmRrΩ]?\d*$',
        r'^TL0\d+',
        r'^LM\d+',
        r'^BC\d+',
        r'^BCM\d+',
        r'^PT\d+',
        r'^\d+n$',
        r'^\d+u$',
        r'^\d+p$',
        r'^[12][kKMm]T?$',
        r'^\d+[kK]\d*$',
        r'^value$',
        r'^\d+R$',
        r'^\d+[KKMM]\d*$',
    ]
    value_count = sum(1 for v in second_col_vals 
                      if any(re.match(pat, v.strip(), re.I) for pat in value_patterns))
    
    total_rows = len(first_col_vals)
    return (
        designator_count > total_rows * 0.5 and
        value_count > total_rows * 0.4 and
        len(df.columns) <= 3
    )


def aggregate_placement_list(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a placement list into an aggregated BOM."""
    from collections import defaultdict
    
    aggregated = defaultdict(lambda: {"qty": 0, "designators": [], "notes": set()})
    
    col_names = [str(c).lower() for c in df.columns]
    
    designator_col_idx = 0
    for i, name in enumerate(col_names):
        if 'designator' in name or 'part' in name or name == 'value':
            designator_col_idx = i
            break
    
    if col_names[0] == 'value' and len(col_names) >= 2 and 'quantity' in col_names[1]:
        first_col_vals = df.iloc[:, 0].astype(str).tolist()[:5]
        designator_pattern = re.compile(r'^[A-Z]+\d+[A-Z]?$', re.I)
        if any(designator_pattern.match(v.strip()) for v in first_col_vals):
            designator_col_idx = 0
            value_col_idx = 1
        else:
            designator_col_idx = 0
            value_col_idx = 1
    else:
        value_col_idx = 1 if len(df.columns) > 1 else 0
    
    notes_col_idx = 2 if len(df.columns) > 2 else None
    
    for _, row in df.iterrows():
        designator = str(row.iloc[designator_col_idx]).strip()
        value = str(row.iloc[value_col_idx]).strip() if value_col_idx < len(row) else ""
        notes = str(row.iloc[notes_col_idx]).strip() if notes_col_idx is not None and notes_col_idx < len(row) and pd.notna(row.iloc[notes_col_idx]) else ""
        
        if not value or value.lower() == "nan":
            continue
            
        if value.lower() in ["value", "component", "part", "comments", "quantity", "notes"]:
            continue
        
        if not re.match(r'^[A-Z]+\d+', designator, re.I):
            if re.match(r'^[A-Z]+\d+', value, re.I):
                designator, value = value, designator
            else:
                continue
        
        aggregated[value]["qty"] += 1
        aggregated[value]["designators"].append(designator)
        if notes and notes.lower() != "nan":
            aggregated[value]["notes"].add(notes)
    
    rows = []
    for value, data in aggregated.items():
        def designator_sort_key(d):
            match = re.match(r'^([A-Z]+)(\d+)([A-Z]?)$', d, re.I)
            if match:
                return (match.group(1).upper(), int(match.group(2)), match.group(3))
            return (d, 0, '')
        
        sorted_designators = sorted(data["designators"], key=designator_sort_key)
        designator_str = ", ".join(sorted_designators)
        notes_str = "; ".join(data["notes"]) if data["notes"] else ""
        details = f"Designators: {designator_str}"
        if notes_str:
            details += f" | {notes_str}"
        
        rows.append({
            "VALUE": value,
            "QUANTITY": str(data["qty"]),
            "DETAILS": details
        })
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def unpack_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    """Handle variant-style BOMs (torpor/apathy/inertia style)."""
    cols = df.columns.tolist()
    
    if is_placement_list(df):
        return aggregate_placement_list(df)
    
    first_col = cols[0].lower() if cols else ""
    
    if len(df) > 0:
        first_col_vals = df.iloc[:, 0].astype(str).tolist()
        designator_pattern = re.compile(r'^[A-Z]+\d+$', re.I)
        designator_count = sum(1 for v in first_col_vals if designator_pattern.match(v.strip()))
        
        if designator_count > len(first_col_vals) * 0.5 and len(cols) > 3:
            all_rows = []
            variant_cols = cols[1:]
            
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
    """Normalize various BOM formats to standard structure."""
    df = unpack_side_by_side_table(df)
    
    if is_placement_list(df):
        return aggregate_placement_list(df)
    
    df = unpack_variant_table(df)
    
    # Handle embedded quantities FIRST (before 2-column handler)
    # This handles tables like MBD where column is "COMPONENT (QUANTITY)" with values like "TL072 (5)"
    cols_lower = {c: c.lower() for c in df.columns}
    embedded_qty_found = False
    
    for col in list(df.columns):  # Use list() to avoid modification during iteration
        col_lower = cols_lower.get(col, col.lower())
        if "component" in col_lower and "quantity" in col_lower:
            embedded_qty_found = True
            components = []
            quantities = []
            
            for val in df[col]:
                comp, qty = parse_embedded_quantity(str(val))
                components.append(comp)
                quantities.append(qty)
            
            df[col] = components
            df = df.rename(columns={col: "VALUE"})
            
            if "QUANTITY" not in df.columns and "quantity" not in [c.lower() for c in df.columns]:
                col_idx = df.columns.get_loc("VALUE")
                df.insert(col_idx + 1, "QUANTITY", quantities)
            break  # Only process one embedded quantity column
    
    # Handle simple 2-column tables (only if no embedded qty was found)
    if not embedded_qty_found and len(df.columns) == 2:
        col1_vals = df.iloc[:, 0].astype(str).tolist()
        col2_vals = df.iloc[:, 1].astype(str).tolist()
        
        qty_count = sum(1 for v in col2_vals if re.match(r'^\d+$', v.strip()))
        value_patterns = [r'\d+[pnuμ]?[FfHh]?', r'\d+[kKMmRrΩ]', r'[A-Z]{2}\d', r'LL\d', r'BC\d', r'S\dJL']
        value_count = sum(1 for v in col1_vals 
                        if any(re.search(pat, v) for pat in value_patterns))
        
        if qty_count > len(col2_vals) * 0.5 and value_count > len(col1_vals) * 0.3:
            df.columns = ["VALUE", "QUANTITY"]
            df["DETAILS"] = ""
            return df
    
    # Normalize column names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if col_lower in ["value", "component", "part", "item"]:
            column_mapping[col] = "VALUE"
        elif col_lower in ["quantity", "qty", "count", "amount"]:
            column_mapping[col] = "QUANTITY"
        elif col_lower in ["details", "notes", "description", "info", "comments"]:
            column_mapping[col] = "DETAILS"
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    for std_col in ["VALUE", "QUANTITY", "DETAILS"]:
        if std_col not in df.columns:
            df[std_col] = ""
    
    std_cols = ["VALUE", "QUANTITY", "DETAILS"]
    other_cols = [c for c in df.columns if c not in std_cols]
    df = df[std_cols + other_cols]
    
    if df["VALUE"].astype(str).str.strip().eq("").all() or df["VALUE"].isna().all():
        for col in other_cols:
            if not df[col].astype(str).str.strip().eq("").all():
                df["VALUE"] = df[col]
                df = df.drop(columns=[col])
                break
    
    df = df[df["VALUE"].astype(str).str.strip().ne("")]
    df = df[df["VALUE"].astype(str).ne("nan")]
    
    return df


def process_single_pdf(pdf_path: Path, resolver: ModuleNameResolver, strategy: str = "filename") -> dict:
    """
    Process a single BOM PDF and return extracted data.
    
    Args:
        pdf_path: Path to the PDF file
        resolver: ModuleNameResolver instance
        strategy: "filename" (recommended) or "content"
    
    Returns:
        Dictionary with module name, tables, and metadata
    """
    filename = pdf_path.stem
    
    # Get module name using the resolver
    clean_name, detection_method = resolver.resolve(pdf_path, strategy=strategy)
    
    print(f"Processing: {pdf_path.name}")
    if clean_name != filename:
        method_indicator = {
            "filename_override": "✓",
            "filename_db_match": "◆",
            "filename_cleaned": "○",
            "content_exact_match": "📄",
            "content_alias_match": "📄",
            "content_title_extraction": "📄",
            "content_frequency_analysis": "📄",
            "filename_fallback": "📁",
        }.get(detection_method, "?")
        print(f"  → Module: {clean_name} [{method_indicator} {detection_method}]")
    
    tables = extract_tables_from_pdf(pdf_path)
    cleaned_tables = []
    placement_list_tables = []
    total_rows = 0
    total_raw_rows = 0
    filtered_rows = 0
    
    candidate_tables = 0
    noise_tables = 0
    tables_not_bom = 0
    
    if not tables:
        print(f"  ⚠ NO TABLES FOUND - Tabula could not detect any table structures")
    else:
        for i, df in enumerate(tables):
            if df.empty:
                noise_tables += 1
                continue
                
            raw_rows = len(df)
            raw_cols = len(df.columns)
            
            if raw_cols < 2 or raw_rows < 2:
                noise_tables += 1
                continue
            
            candidate_tables += 1
            total_raw_rows += raw_rows
            
            cleaned = clean_bom_dataframe(df)
            if not cleaned.empty:
                if is_placement_list(cleaned):
                    aggregated = aggregate_placement_list(cleaned)
                    if not aggregated.empty:
                        placement_list_tables.append(aggregated)
                        filtered_rows += raw_rows - len(aggregated)
                else:
                    cleaned = normalize_bom_table(cleaned)
                    if not cleaned.empty:
                        cleaned_tables.append(cleaned)
                        cleaned_rows = len(cleaned)
                        total_rows += cleaned_rows
                        
                        rows_lost = raw_rows - cleaned_rows
                        if rows_lost > 0:
                            filtered_rows += rows_lost
                    else:
                        tables_not_bom += 1
            else:
                tables_not_bom += 1
    
    # Merge placement list tables
    if placement_list_tables:
        from collections import defaultdict
        
        merged_aggregation = defaultdict(lambda: {"qty": 0, "designators": [], "notes": set()})
        
        for pl_df in placement_list_tables:
            for _, row in pl_df.iterrows():
                value = row.get("VALUE", "")
                qty = int(row.get("QUANTITY", 0)) if row.get("QUANTITY", "").isdigit() else 0
                details = row.get("DETAILS", "")
                
                designator_match = re.search(r"Designators?: ([^|]+)", details)
                if designator_match:
                    designators = [d.strip() for d in designator_match.group(1).split(",")]
                else:
                    designators = []
                
                notes_match = re.search(r"\| (.+)$", details)
                notes = notes_match.group(1) if notes_match else ""
                
                merged_aggregation[value]["qty"] += qty
                merged_aggregation[value]["designators"].extend(designators)
                if notes:
                    merged_aggregation[value]["notes"].add(notes)
        
        merged_rows = []
        for value, data in merged_aggregation.items():
            def designator_sort_key(d):
                match = re.match(r'^([A-Z]+)(\d+)([A-Z]?)$', d, re.I)
                if match:
                    return (match.group(1).upper(), int(match.group(2)), match.group(3))
                return (d, 0, '')
            
            sorted_designators = sorted(set(data["designators"]), key=designator_sort_key)
            designator_str = ", ".join(sorted_designators)
            notes_str = "; ".join(data["notes"]) if data["notes"] else ""
            details = f"Designators: {designator_str}"
            if notes_str:
                details += f" | {notes_str}"
            
            merged_rows.append({
                "VALUE": value,
                "QUANTITY": str(data["qty"]),
                "DETAILS": details
            })
        
        if merged_rows:
            merged_df = pd.DataFrame(merged_rows)
            cleaned_tables.insert(0, merged_df)
            total_rows += len(merged_rows)
    
    # Summary output
    if cleaned_tables:
        print(f"  Found {len(cleaned_tables)} BOM table(s), {total_rows} total component rows")
        if placement_list_tables:
            print(f"  ℹ Merged {len(placement_list_tables)} placement list segment(s)")
        if filtered_rows > 0:
            print(f"  ℹ {filtered_rows} row(s) filtered during cleaning")
        if tables_not_bom > 0:
            print(f"  ℹ {tables_not_bom} candidate table(s) not recognized as BOM")
    else:
        if candidate_tables > 0:
            print(f"  ⚠ TABLE(S) NOT CONSUMED - {candidate_tables} candidate(s) found but none recognized as BOM")
        elif tables:
            print(f"  ⚠ NO BOM TABLES FOUND - {len(tables)} small fragment(s) detected")
    
    return {
        "filename": pdf_path.name,
        "module_name": clean_name,
        "name_detection_method": detection_method,
        "tables": cleaned_tables,
        "table_count": len(cleaned_tables),
        "total_rows": total_rows,
        "raw_tables_found": len(tables) if tables else 0,
        "candidate_tables": candidate_tables,
        "raw_rows_found": total_raw_rows,
        "rows_filtered": filtered_rows,
        "tables_not_bom": tables_not_bom,
        "placement_lists_merged": len(placement_list_tables) if placement_list_tables else 0
    }


def scan_directory(directory: Path, recursive: bool = False) -> list[Path]:
    """Find all PDF files in a directory."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = list(directory.glob(pattern))
    pdfs.extend(directory.glob(pattern.replace(".pdf", ".PDF")))
    return sorted(set(pdfs))


def export_to_csv(results: list[dict], output_dir: Path, combine: bool = False):
    """Export results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if combine:
        all_dfs = []
        for result in results:
            for df in result["tables"]:
                df = df.copy()
                
                for col in ["VALUE", "QUANTITY", "DETAILS"]:
                    if col not in df.columns:
                        df[col] = ""
                
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
        for result in results:
            if result["tables"]:
                combined = pd.concat(result["tables"], ignore_index=True)
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in result["module_name"])
                output_path = output_dir / f"{safe_name}.csv"
                combined.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")


def export_to_json(results: list[dict], output_dir: Path, combine: bool = False):
    """Export results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def df_to_records(tables):
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
                "name_detection_method": result.get("name_detection_method", "unknown"),
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
                    "name_detection_method": result.get("name_detection_method", "unknown"),
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
            summary_data = [{
                "Module": r["module_name"],
                "Detection Method": r.get("name_detection_method", "unknown"),
                "Source File": r["filename"],
                "Tables": r["table_count"],
                "Total Rows": r["total_rows"]
            } for r in results]
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            for result in results:
                if result["tables"]:
                    combined = pd.concat(result["tables"], ignore_index=True)
                    sheet_name = result["module_name"][:31]
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
    """Export results in Mouser-compatible BOM format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_parts = []
    
    for result in results:
        for df in result["tables"]:
            for _, row in df.iterrows():
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
                        if not value:
                            value = cell_value
                    elif "designator" in col_lower:
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
                
                if (not quantity or quantity in ["", "nan"]) and designators:
                    quantity = str(count_designators(designators))
                
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
    %(prog)s ./boms                             # Extract using filename-priority (default)
    %(prog)s ./boms --content-priority          # Extract using content-based detection
    %(prog)s ./boms -o json --combine           # Combine all into one JSON file
    %(prog)s ./boms -r                          # Recursive directory scan
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
        help="Output directory (default: ./output)"
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
        "--mouser",
        action="store_true",
        help="Generate Mouser-compatible BOM format"
    )
    
    # Module name resolution options
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument(
        "--filename-priority",
        action="store_true",
        default=True,
        help="Use filename for module names with overrides (default, recommended)"
    )
    name_group.add_argument(
        "--content-priority",
        action="store_true",
        help="Use PDF content for module names (may misidentify some modules)"
    )
    
    parser.add_argument(
        "--modules-db",
        type=Path,
        default=None,
        help="Path to nlc_modules.json canonical names database"
    )
    
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="Path to module_overrides.json for filename→name mappings"
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
    output_dir = args.output_dir or Path("./output")
    
    # Determine naming strategy
    strategy = "content" if args.content_priority else "filename"
    
    # Initialize the resolver
    resolver = ModuleNameResolver(
        modules_db_path=args.modules_db,
        overrides_path=args.overrides
    )
    
    # Report configuration
    if resolver.db.get("canonical_names"):
        print(f"Loaded {len(resolver.db['canonical_names'])} canonical module names")
    if resolver.overrides:
        print(f"Loaded {len(resolver.overrides)} filename overrides")
    print(f"Module naming strategy: {strategy}")
    
    # Find PDFs
    print(f"\nScanning: {args.directory}")
    pdfs = scan_directory(args.directory, args.recursive)
    
    if not pdfs:
        print("No PDF files found.")
        sys.exit(0)
    
    print(f"Found {len(pdfs)} PDF file(s)\n")

    seen_hashes = set()
    duplicates = []
    
    # Process each PDF
    results = []
    for pdf_path in pdfs:
        h = file_sha256(pdf_path)
        if h in seen_hashes:
            duplicates.append(pdf_path)
            print(f"Skipping duplicate (same bytes): {pdf_path.name}")
            continue
        seen_hashes.add(h)
        result = process_single_pdf(pdf_path, resolver, strategy=strategy)
        results.append(result)
    
    if duplicates:
        print(f"\nSkipped {len(duplicates)} duplicate PDF(s) by content hash.")
    
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
    
    files_no_tables = sum(1 for r in results if r["raw_tables_found"] == 0)
    files_not_consumed = sum(1 for r in results if r["candidate_tables"] > 0 and r["table_count"] == 0)
    total_filtered = sum(r.get("rows_filtered", 0) for r in results)
    total_not_bom = sum(r.get("tables_not_bom", 0) for r in results)
    
    method_counts = Counter(r.get("name_detection_method", "unknown") for r in results)
    
    print(f"\nDone! Processed {len(results)} files, extracted {total_tables} BOM tables, {total_rows} total rows")
    
    print(f"\nModule name detection breakdown:")
    for method, count in method_counts.most_common():
        indicator = {
            "filename_override": "✓",
            "filename_db_match": "◆",
            "filename_cleaned": "○",
            "content_exact_match": "📄",
            "content_alias_match": "📄",
            "content_title_extraction": "📄",
            "content_frequency_analysis": "📄",
            "filename_fallback": "📁",
        }.get(method, "?")
        print(f"  {indicator} {method}: {count}")
    
    if files_no_tables > 0:
        print(f"\n  ⚠ {files_no_tables} file(s) had NO TABLES detected by Tabula")
    if files_not_consumed > 0:
        print(f"  ⚠ {files_not_consumed} file(s) had candidate tables NOT CONSUMED")
    if total_filtered > 0:
        print(f"  ℹ {total_filtered} total row(s) filtered during cleaning")
    if total_not_bom > 0:
        print(f"  ℹ {total_not_bom} candidate table(s) not recognized as BOM")


if __name__ == "__main__":
    main()
