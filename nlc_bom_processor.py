#!/usr/bin/env python3
"""
NLC BOM Post-Processor
Cleans, normalizes, deduplicates and analyzes extracted BOM data.
Extracts Tayda/Mouser part numbers and generates clean shopping lists.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
from urllib.parse import unquote

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Number of part number columns to generate per supplier
MAX_PN_COLUMNS = 3


# =============================================================================
# Part Number Extraction
# =============================================================================

def extract_tayda_pn(text: str) -> list[str]:
    """Extract all Tayda part numbers from text.
    
    Tayda format: A-XXXX where X is digits, sometimes with suffix like A-1234-RED
    Examples: A-1234, A-01234, A-5476-OG, A-3500
    """
    if not text or pd.isna(text):
        return []
    text = str(text)
    
    found = set()
    
    # Pattern 1: Explicit "Tayda: A-1234" or "Tayda A-1234"
    for m in re.finditer(r'[Tt]ayda[:\s]+([A-Z]-\d+(?:-[A-Z0-9]+)?)', text, re.I):
        found.add(m.group(1).upper())
    
    # Pattern 2: Standalone A-XXXX pattern (Tayda's standard format)
    # Must start with A- followed by 3-5 digits, optionally followed by -suffix
    for m in re.finditer(r'\b(A-\d{3,5}(?:-[A-Z0-9]+)?)\b', text, re.I):
        found.add(m.group(1).upper())
    
    return sorted(found)


def extract_mouser_pn(text: str) -> list[str]:
    """Extract all Mouser part numbers from text.
    
    Mouser format: NNN-XXXXX where NNN is 3-4 digit manufacturer code
    Examples: 595-TL072CP, 512-LM13700N/NOPB, 78-1N4148, 652-BC547BTA
    """
    if not text or pd.isna(text):
        return []
    text = str(text)
    
    found = set()
    
    # Pattern 1: Explicit "Mouser: 595-TL072" or "Mouser Part No 512-LM13700"
    for m in re.finditer(r'[Mm]ouser[:\s]+(?:Part\s*No\.?\s*)?(\d{2,4}-[A-Za-z0-9\-/]+)', text):
        found.add(m.group(1))
    
    # Pattern 2: Standalone Mouser PN format (3-4 digit prefix, dash, alphanumeric)
    # Be careful not to match other things like dates or random numbers
    for m in re.finditer(r'\b(\d{2,4}-[A-Z][A-Z0-9\-/]{3,})\b', text, re.I):
        pn = m.group(1)
        # Validate it looks like a real Mouser PN (not a date, not too short)
        if len(pn) >= 8 and not re.match(r'^\d+-\d+$', pn):
            found.add(pn)
    
    return sorted(found)


def extract_digikey_pn(text: str) -> list[str]:
    """Extract all DigiKey part numbers from text."""
    if not text or pd.isna(text):
        return []
    text = str(text)
    
    found = set()
    for m in re.finditer(r'[Dd]igi-?[Kk]ey[:\s]+([A-Za-z0-9-]+)', text):
        found.add(m.group(1))
    
    return sorted(found)


def extract_package(text: str) -> str:
    """Extract package/footprint info from text."""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    
    # Common packages
    packages = []
    if re.search(r'\b0805\b', text):
        packages.append("0805")
    if re.search(r'\b0603\b', text):
        packages.append("0603")
    if re.search(r'\b1206\b', text):
        packages.append("1206")
    if re.search(r'\bSOIC\b', text, re.I):
        packages.append("SOIC")
    if re.search(r'\bSOT-?23\b', text, re.I):
        packages.append("SOT-23")
    if re.search(r'\bSOD-?80\b', text, re.I):
        packages.append("SOD-80")
    if re.search(r'\bDIP\b', text, re.I):
        packages.append("DIP")
    if re.search(r'\bthru[- ]?hole\b', text, re.I):
        packages.append("THT")
    if re.search(r'\belectro', text, re.I):
        packages.append("Electrolytic")
    
    return ", ".join(packages) if packages else ""


# =============================================================================
# Component Normalization
# =============================================================================

def is_valid_component(val: str) -> bool:
    """Check if a value looks like a valid component."""
    if not val or pd.isna(val):
        return False
    
    val = str(val).strip()
    
    # Reject empty or too short
    if len(val) < 2:
        return False
    
    # Reject pure numbers (like "2", "10", "25")
    if re.match(r'^\d+\.?\d*$', val):
        return False
    
    # Reject designators (C1, R1, U1, D1, etc.)
    if re.match(r'^[CRUDQLJ]\d+$', val, re.I):
        return False
    
    # Reject special characters only
    if re.match(r'^[\$\#\@\!\*\&\^\%\-\_]+$', val):
        return False
    
    # Reject common noise patterns
    noise_patterns = [
        r'^[\(\)\[\]\{\}]+$',  # Just brackets
        r'^\d+[vV]$',  # Just voltage like "25V"
        r'^optional',  # "optional" text
        r'^\$',  # Starts with $
        r'^#\d+$',  # Just "#1", "#2"
        r'^see\s+note',  # "see notes"
        r'^n/?a$',  # "n/a", "na"
        r'^-+$',  # Just dashes
        r'^\*+$',  # Just asterisks
        r'^install',  # "install on..."
        r'^leave\s+',  # "leave empty"
        r'^do\s+not',  # "do not..."
        r'^LEAVE',  # "LEAVE EMPTY"
        r'^cut\s+to',  # "cut to size"
        r'^[CRUDQL]\d+-[CRUDQL]?\d+$',  # Designator ranges like "C1-C6", "R1-R10"
        r'^CAPS$',  # Generic "CAPS"
        r'^RESISTORS?$',  # Generic "RESISTOR"
        r'^\d+\s*x\s*\d+\s*pins?$',  # "2x3 pins"
        r'^thru-?hole\s+resistors?$',  # "thru-hole resistors"
        r'^\d+\s+(and|or)\s+\d+\s*pin',  # "8 and/or 14 pin..."
        r'^\d+[kKmM]?\s+INSTALL',  # "33k INSTALL 10K"
        r'^\d+[kKmM]?\d*\s*[–\-]\s*\d+',  # "4M7 – 10M" (range, not value)
        r'^\d+\s+or\s+\d+\s+resistor',  # "1206 or 0805 resistors"
    ]
    
    for pattern in noise_patterns:
        if re.match(pattern, val, re.I):
            return False
    
    return True


def normalize_value(val: str) -> tuple[str, str]:
    """
    Normalize a component value to standard format.
    Returns (normalized_value, component_type)
    """
    if not val or pd.isna(val):
        return "", "unknown"
    
    val = str(val).strip()
    
    # Skip placeholders
    if val.lower() in ["-", "nan", "n/a", "", "nothing!"]:
        return "", "skip"
    
    # Validate component value
    if not is_valid_component(val):
        return "", "skip"
    
    # Clean up common formatting issues
    val = val.replace('\r', ' ').replace('\n', ' ')
    val = re.sub(r'\s+', ' ', val).strip()
    
    # --- RESISTORS ---
    # 4k7, 4K7, 1M2 style (European notation)
    m = re.match(r'^(\d+)\s*([kKmM])\s*(\d+)\s*(?:\(.*\))?\s*$', val)
    if m:
        mult = m.group(2).upper()
        return f"{m.group(1)}{mult}{m.group(3)}", "resistor"
    
    # 100k, 100K, 1M, 1m (with optional suffix like "100k (c)")
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([kKmM])\s*(?:ohm|Ω)?\s*(?:\(.*\))?\s*$', val)
    if m:
        mult = m.group(2).upper()
        num = m.group(1)
        if '.' in num and num.endswith('0'):
            num = num.rstrip('0').rstrip('.')
        return f"{num}{mult}", "resistor"
    
    # 100R, 470R, 10R (with optional "resistor" suffix)
    m = re.match(r'^(\d+(?:\.\d+)?)\s*[rRΩ]\s*(?:resistor)?\s*$', val)
    if m:
        return f"{m.group(1)}R", "resistor"
    
    # Plain number + "ohm" 
    m = re.match(r'^(\d+(?:\.\d+)?)\s*ohm\s*$', val, re.I)
    if m:
        return f"{m.group(1)}R", "resistor"
    
    # RL (LED resistor placeholder)
    if val.upper() == "RL" or val.lower().startswith("rl "):
        return "RL (LED resistor)", "resistor"
    
    # --- CAPACITORS ---
    # Value with SMD code: 100nF (104), 22nF/223
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([pPnNuUμµ])[fF]?\s*[/\(]?\s*\d*\s*\)?\s*$', val)
    if m:
        unit = m.group(2).lower()
        if unit in ['μ', 'µ']:
            unit = 'u'
        return f"{m.group(1)}{unit}F", "capacitor"
    
    # Plain capacitor: 100n, 1u, 47p
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([pPnNuUμµ])\s*$', val)
    if m:
        unit = m.group(2).lower()
        if unit in ['μ', 'µ']:
            unit = 'u'
        return f"{m.group(1)}{unit}F", "capacitor"
    
    # SMD code only: (104), (103)
    m = re.match(r'^\(?\s*(10[0-6]|22[0-5]|33[0-4]|47[0-4]|68[0-4])\s*\)?$', val)
    if m:
        code = m.group(1)
        decoded = {
            "100": "10pF", "101": "100pF", "102": "1nF", "103": "10nF", "104": "100nF", "105": "1uF", "106": "10uF",
            "220": "22pF", "221": "220pF", "222": "2.2nF", "223": "22nF", "224": "220nF", "225": "2.2uF",
            "330": "33pF", "331": "330pF", "332": "3.3nF", "333": "33nF", "334": "330nF",
            "470": "47pF", "471": "470pF", "472": "4.7nF", "473": "47nF", "474": "470nF",
            "680": "68pF", "681": "680pF", "682": "6.8nF", "683": "68nF", "684": "680nF",
        }.get(code, f"{code}")
        return decoded, "capacitor"
    
    # Electrolytic: 10μF 25V, 100uF electro
    m = re.match(r'^(\d+)\s*[uUμµ][fF]?\s*(?:\d+[vV])?\s*(?:electro.*)?$', val, re.I)
    if m:
        return f"{m.group(1)}uF", "capacitor"
    
    # --- ICs / CHIPS ---
    ic_patterns = [
        (r'^TL07[24].*', "TL072/074"),
        (r'^TL08[24].*', "TL082/084"),
        (r'^LM13700.*', "LM13700"),
        (r'^LM358.*', "LM358"),
        (r'^NE555.*|^LM555.*', "555 Timer"),
        (r'^CD40\d+.*', lambda m: m.group(0).split()[0].upper()),
        (r'^74[HhLl][CcSs]\d+.*', lambda m: m.group(0).split()[0].upper()),
        (r'^PT2399.*', "PT2399"),
        (r'^4046.*', "CD4046"),
        (r'^V13700.*', "V13700"),
    ]
    for pattern, repl in ic_patterns:
        m = re.match(pattern, val, re.I)
        if m:
            if callable(repl):
                return repl(m), "ic"
            return repl, "ic"
    
    # Generic IC with "or" alternatives: "TL072 or TL082"
    m = re.match(r'^(TL\d+|LM\d+|NE\d+)\s+or\s+(TL\d+|LM\d+|NE\d+)', val, re.I)
    if m:
        return f"{m.group(1).upper()}/{m.group(2).upper()}", "ic"
    
    # --- TRANSISTORS ---
    transistor_patterns = [
        (r'^BC[58][0-9]{2}.*', lambda m: m.group(0)[:5].upper()),
        (r'^2N\d{4}.*', lambda m: m.group(0)[:6].upper()),
        (r'^MMBF.*', lambda m: m.group(0).split()[0].upper()),
        (r'^BCM847.*', "BCM847DS"),
        (r'^J\d{3}.*', lambda m: m.group(0).split()[0].upper()),
    ]
    for pattern, repl in transistor_patterns:
        m = re.match(pattern, val, re.I)
        if m:
            if callable(repl):
                return repl(m), "transistor"
            return repl, "transistor"
    
    # FET patterns
    if re.match(r'^[NP]?FET\s*J?\d+', val, re.I) or "J270" in val.upper() or "J309" in val.upper():
        m = re.search(r'J\d+', val, re.I)
        if m:
            return m.group(0).upper(), "transistor"
    
    # --- DIODES ---
    diode_patterns = [
        (r'1N4148|LL4148', "1N4148"),
        (r'1N400[1-7]', "1N400x"),
        (r'1N5819|B5819', "1N5819"),
        (r'BAT54', "BAT54"),
        (r'S1JL', "S1JL"),
    ]
    for pattern, repl in diode_patterns:
        if re.search(pattern, val, re.I):
            return repl, "diode"
    
    if "schottky" in val.lower():
        return "Schottky", "diode"
    if "zener" in val.lower():
        m = re.search(r'(\d+[vV]\d*)', val)
        if m:
            return f"Zener {m.group(1).upper()}", "diode"
        return "Zener", "diode"
    
    # --- CONNECTORS ---
    if "3.5mm" in val.lower() or "3.5 mm" in val.lower() or "kobiconn" in val.lower() or "thonkiconn" in val.lower():
        if "stereo" in val.lower():
            return "3.5mm Jack Stereo", "connector"
        return "3.5mm Jack Mono", "connector"
    
    if "6.35mm" in val.lower() or "1/4" in val or "¼" in val:
        return "6.35mm Jack", "connector"
    
    if "eurorack" in val.lower() and "power" in val.lower():
        return "Eurorack Power Header", "connector"
    
    if "pin header" in val.lower():
        return "Pin Header", "connector"
    
    # --- POTS ---
    if "pot" in val.lower():
        m = re.search(r'(\d+)[kK]', val)
        if m:
            taper = "B" if re.search(r'\d+[kK]?[bB]', val) else ""
            return f"{m.group(1)}k{taper} Pot", "pot"
        return "Pot", "pot"
    
    if "trimpot" in val.lower() or "trimmer" in val.lower():
        m = re.search(r'(\d+)[kK]', val)
        if m:
            return f"{m.group(1)}k Trimpot", "pot"
        return "Trimpot", "pot"
    
    # --- LEDS ---
    if "led" in val.lower():
        if "bipolar" in val.lower():
            return "Bipolar LED", "led"
        m = re.search(r'(\d+)\s*mm', val.lower())
        if m:
            return f"{m.group(1)}mm LED", "led"
        return "LED", "led"
    
    # --- MISC ---
    if "vactrol" in val.lower():
        return "Vactrol", "vactrol"
    
    if "switch" in val.lower():
        if "dpdt" in val.lower():
            return "DPDT Switch", "switch"
        if "spdt" in val.lower():
            return "SPDT Switch", "switch"
        return "Switch", "switch"
    
    if "78l05" in val.lower() or "7805" in val.lower():
        return "78L05", "regulator"
    
    # --- Additional patterns ---
    
    # Capacitors with "or" and SMD codes
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([pPnNuUμµ])[fF]?\s*(?:or|\/)?\s*\d*\s*$', val)
    if m:
        unit = m.group(2).lower()
        if unit in ['μ', 'µ']:
            unit = 'u'
        return f"{m.group(1)}{unit}F", "capacitor"
    
    # Resistors with asterisks or question marks
    m = re.match(r'^(\d+)([kKmM])(\d*)\s*[\*\?]?\s*$', val)
    if m:
        mult = m.group(2).upper()
        suffix = m.group(3) if m.group(3) else ""
        return f"{m.group(1)}{mult}{suffix}", "resistor"
    
    # Plain resistor value with asterisk
    m = re.match(r'^(\d+)\s*[rRΩ]\s*[\*\?]+\s*$', val)
    if m:
        return f"{m.group(1)}R", "resistor"
    
    # LED resistors
    if re.match(r'^R[Ldv]+$', val, re.I):
        return "RL (LED resistor)", "resistor"
    
    # LDR
    if val.upper() == "LDR":
        return "LDR", "sensor"
    
    # Jacks mentioned generically
    if val.lower() == "jacks" or val.lower() == "jack":
        return "3.5mm Jack Mono", "connector"
    
    # Toggle switches
    if "toggle" in val.lower():
        if "spdt" in val.lower():
            return "SPDT Toggle Switch", "switch"
        if "dpdt" in val.lower():
            return "DPDT Toggle Switch", "switch"
        return "Toggle Switch", "switch"
    
    # MN3xxx / V3xxx BBD chips
    if re.search(r'[MV]N?3[012]\d{2}', val, re.I):
        m = re.search(r'([MV]N?3[012]\d{2})', val, re.I)
        return m.group(1).upper(), "ic"
    
    # DG4xx / DG5xx analog switches
    if re.search(r'DG[45]\d{2}', val, re.I):
        m = re.search(r'(DG[45]\d{2})', val, re.I)
        return m.group(1).upper(), "ic"
    
    # More ICs
    ic_generic_patterns = [
        (r'^LM\d{3,4}', lambda m: m.group(0).upper()),
        (r'^4013\b', "CD4013"),
        (r'^4060\b', "CD4060"),
        (r'^555\s*(?:or|/)\s*7555', "555/7555"),
        (r'^SSI\d+', lambda m: m.group(0).upper()),
        (r'^LTC\d+', lambda m: m.group(0).split()[0].upper()),
        (r'^SA571', "SA571"),
        (r'^79L05', "79L05"),
    ]
    for pattern, repl in ic_generic_patterns:
        m = re.match(pattern, val, re.I)
        if m:
            if callable(repl):
                return repl(m), "ic"
            return repl, "ic"
    
    # 79L05 is a regulator
    if "79l05" in val.lower():
        return "79L05", "regulator"
    
    # Capacitor with description
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([pPnNuUμµ])[fF]?\s*\(', val)
    if m:
        unit = m.group(2).lower()
        if unit in ['μ', 'µ']:
            unit = 'u'
        return f"{m.group(1)}{unit}F", "capacitor"
    
    # Capacitor with European notation
    m = re.match(r'^(\d+)([nNpPuU])(\d+)\s*\(', val)
    if m:
        unit = m.group(2).lower()
        return f"{m.group(1)}.{m.group(3)}{unit}F", "capacitor"
    
    # "100nF capacitor" style
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([pPnNuUμµ])[fF]?\s+cap', val, re.I)
    if m:
        unit = m.group(2).lower()
        if unit in ['μ', 'µ']:
            unit = 'u'
        return f"{m.group(1)}{unit}F", "capacitor"
    
    # "10μF 25V" style electrolytic
    m = re.match(r'^(\d+)\s*[uUμµ][fF]?\s+\d+[vV]', val)
    if m:
        return f"{m.group(1)}uF", "capacitor"
    
    # "Cx 100n" style
    m = re.match(r'^C\d+\s+(\d+(?:\.\d+)?)\s*([pPnNuUμµ])', val, re.I)
    if m:
        unit = m.group(2).lower()
        if unit in ['μ', 'µ']:
            unit = 'u'
        return f"{m.group(1)}{unit}F", "capacitor"
    
    # Power connector
    if "power connector" in val.lower() or "10 pin power" in val.lower():
        return "Eurorack Power Header", "connector"
    
    # Crystal
    if "crystal" in val.lower() or "xtal" in val.lower():
        m = re.search(r'(\d+(?:\.\d+)?)\s*[kKmM]?[hH]z', val)
        if m:
            return f"{m.group(0)} Crystal", "other"
        return "Crystal", "other"
    
    # IC sockets
    if "socket" in val.lower() and ("pin" in val.lower() or "ic" in val.lower()):
        return "IC Socket", "connector"
    
    # Return cleaned original if no match
    return val, "other"


def clean_module_name(name: str) -> str:
    """Clean up module name."""
    if not name:
        return ""
    
    # URL decode (+ to space, %XX codes)
    name = unquote(name.replace('+', ' '))
    
    # Remove common suffixes
    for suffix in ['_build_and_bom', '_build_and', '_BOM', '_bom', ' BOM', ' bom', 
                   '_Build_and_BOM', ' Build and BOM', ' build and BOM']:
        if name.lower().endswith(suffix.lower()):
            name = name[:-len(suffix)]
    
    # Clean up underscores and extra spaces
    name = name.replace('_', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def clean_quantity(qty) -> int:
    """Parse quantity to integer."""
    if pd.isna(qty) or qty == "":
        return 1
    
    qty = str(qty).strip()
    
    if qty.endswith('.0'):
        qty = qty[:-2]
    
    m = re.match(r'^(\d+)\s*x\s*(\d+)', qty, re.I)
    if m:
        return int(m.group(1)) * int(m.group(2))
    
    m = re.match(r'^(\d+)', qty)
    if m:
        return int(m.group(1))
    
    return 1


# =============================================================================
# Main Processing
# =============================================================================

def process_bom(input_path: Path) -> pd.DataFrame:
    """Load and process a BOM CSV file."""
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    
    value_cols = ['Component', 'VALUE', 'component', 'Value']
    qty_cols = ['quantity', 'QUANTITY', 'Quantity']
    detail_cols = ['notes', 'DETAILS', 'Notes', 'NOTES', 'Details']
    
    rows = []
    for _, row in df.iterrows():
        module = clean_module_name(str(row.get('_module', '')))
        
        value = ""
        for col in value_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                value = str(row[col]).strip()
                break
        
        variant_cols = ['torpor', 'apathy', 'inertia']
        has_variants = any(col in row and pd.notna(row.get(col)) for col in variant_cols)
        
        if has_variants and not value:
            designator = str(row.get('Component', row.get('component', ''))).strip()
            for variant in variant_cols:
                if variant in row and pd.notna(row[variant]):
                    var_val = str(row[variant]).strip()
                    if var_val and var_val.lower() not in ['nan', 'nothing!', '']:
                        norm_val, comp_type = normalize_value(var_val)
                        if norm_val and comp_type != "skip":
                            rows.append({
                                'Module': module,
                                'Value': norm_val,
                                'Quantity': 1,
                                'Type': comp_type,
                                'Package': '',
                                'Tayda_PNs': [],
                                'Mouser_PNs': [],
                                'Original': var_val,
                                'Details': f"{designator} ({variant})",
                            })
            continue
        
        qty = 1
        for col in qty_cols:
            if col in row and pd.notna(row[col]):
                qty = clean_quantity(row[col])
                break
        
        details = ""
        for col in detail_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                details = str(row[col]).strip()
                break
        
        all_text = " ".join(str(x) for x in row.values if pd.notna(x))
        
        tayda_pns = extract_tayda_pn(all_text)
        mouser_pns = extract_mouser_pn(all_text)
        package = extract_package(all_text)
        
        if value:
            norm_val, comp_type = normalize_value(value)
            if norm_val and comp_type != "skip":
                rows.append({
                    'Module': module,
                    'Value': norm_val,
                    'Quantity': qty,
                    'Type': comp_type,
                    'Package': package,
                    'Tayda_PNs': tayda_pns,
                    'Mouser_PNs': mouser_pns,
                    'Original': value,
                    'Details': details,
                })
    
    result = pd.DataFrame(rows)
    print(f"Processed to {len(result)} clean rows")
    return result


def generate_stats(df: pd.DataFrame) -> dict:
    """Generate statistics from processed BOM."""
    
    stats = {
        'total_rows': len(df),
        'total_components': int(df['Quantity'].sum()),
        'modules': df['Module'].nunique(),
        'unique_values': df['Value'].nunique(),
    }
    
    type_counts = df.groupby('Type')['Quantity'].sum().sort_values(ascending=False)
    stats['by_type'] = type_counts.to_dict()
    
    value_counts = df.groupby('Value')['Quantity'].sum().sort_values(ascending=False)
    stats['most_used'] = value_counts.head(30).to_dict()
    
    stats['most_used_by_type'] = {}
    for comp_type in ['resistor', 'capacitor', 'ic', 'connector', 'transistor', 'diode', 'pot']:
        type_df = df[df['Type'] == comp_type]
        if len(type_df) > 0:
            counts = type_df.groupby('Value')['Quantity'].sum().sort_values(ascending=False)
            stats['most_used_by_type'][comp_type] = counts.head(10).to_dict()
    
    module_counts = df.groupby('Module')['Quantity'].sum().sort_values(ascending=False)
    stats['module_complexity'] = module_counts.to_dict()
    
    return stats


def print_stats(stats: dict):
    """Print formatted statistics."""
    
    print("\n" + "=" * 70)
    print("NLC BOM STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal line items:     {stats['total_rows']:,}")
    print(f"Total components:     {stats['total_components']:,}")
    print(f"Unique modules:       {stats['modules']}")
    print(f"Unique part values:   {stats['unique_values']}")
    
    print("\n" + "-" * 70)
    print("COMPONENTS BY TYPE")
    print("-" * 70)
    for comp_type, count in stats['by_type'].items():
        print(f"  {comp_type:20} {count:6,}")
    
    print("\n" + "-" * 70)
    print("TOP 25 MOST USED COMPONENTS")
    print("-" * 70)
    for i, (value, count) in enumerate(list(stats['most_used'].items())[:25], 1):
        print(f"  {i:2}. {value:30} {count:6,}")
    
    for comp_type in ['resistor', 'capacitor', 'ic']:
        if comp_type in stats['most_used_by_type']:
            print(f"\n--- Top {comp_type.upper()}S ---")
            for value, count in list(stats['most_used_by_type'][comp_type].items())[:8]:
                print(f"  {value:30} {count:6,}")
    
    print("\n" + "-" * 70)
    print("LARGEST MODULES (by component count)")
    print("-" * 70)
    for i, (module, count) in enumerate(list(stats['module_complexity'].items())[:15], 1):
        print(f"  {i:2}. {module:45} {count:5,}")


def generate_shopping_list(df: pd.DataFrame, by_type: bool = True) -> pd.DataFrame:
    """Generate consolidated shopping list with part numbers in separate columns."""
    
    agg_data = defaultdict(lambda: {
        'total_qty': 0,
        'modules': set(),
        'type': '',
        'packages': set(),
        'tayda_pns': set(),
        'mouser_pns': set(),
    })
    
    for _, row in df.iterrows():
        key = row['Value']
        if not key or key == "":
            continue
            
        agg_data[key]['total_qty'] += row['Quantity']
        agg_data[key]['type'] = row['Type']
        agg_data[key]['modules'].add(row['Module'])
        
        if row.get('Package'):
            agg_data[key]['packages'].add(row['Package'])
        
        tayda_list = row.get('Tayda_PNs', [])
        mouser_list = row.get('Mouser_PNs', [])
        
        if isinstance(tayda_list, str) and tayda_list:
            agg_data[key]['tayda_pns'].add(tayda_list)
        elif isinstance(tayda_list, list):
            agg_data[key]['tayda_pns'].update(tayda_list)
            
        if isinstance(mouser_list, str) and mouser_list:
            agg_data[key]['mouser_pns'].add(mouser_list)
        elif isinstance(mouser_list, list):
            agg_data[key]['mouser_pns'].update(mouser_list)
    
    rows = []
    for component, data in agg_data.items():
        if not component or not component.strip():
            continue
            
        tayda_list = sorted(data['tayda_pns']) if data['tayda_pns'] else []
        mouser_list = sorted(data['mouser_pns']) if data['mouser_pns'] else []
        
        row_data = {
            'Component': component,
            'Type': data['type'],
            'Qty': data['total_qty'],
            'Package': '; '.join(sorted(data['packages'])) if data['packages'] else '',
            'Modules': len(data['modules']),
        }
        
        for i in range(MAX_PN_COLUMNS):
            col_name = f'Tayda_PN_{i+1}'
            row_data[col_name] = tayda_list[i] if i < len(tayda_list) else ''
        
        for i in range(MAX_PN_COLUMNS):
            col_name = f'Mouser_PN_{i+1}'
            row_data[col_name] = mouser_list[i] if i < len(mouser_list) else ''
        
        rows.append(row_data)
    
    shopping_df = pd.DataFrame(rows)
    
    type_order = ['resistor', 'capacitor', 'ic', 'transistor', 'diode', 'connector', 'pot', 'led', 'sensor', 'vactrol', 'regulator', 'switch', 'other']
    shopping_df['_type_order'] = shopping_df['Type'].apply(lambda x: type_order.index(x) if x in type_order else 99)
    shopping_df = shopping_df.sort_values(['_type_order', 'Qty'], ascending=[True, False])
    shopping_df = shopping_df.drop(columns=['_type_order'])
    
    base_cols = ['Component', 'Type', 'Qty', 'Package']
    tayda_cols = [f'Tayda_PN_{i+1}' for i in range(MAX_PN_COLUMNS)]
    mouser_cols = [f'Mouser_PN_{i+1}' for i in range(MAX_PN_COLUMNS)]
    final_cols = base_cols + tayda_cols + mouser_cols + ['Modules']
    
    final_cols = [c for c in final_cols if c in shopping_df.columns]
    shopping_df = shopping_df[final_cols]
    
    return shopping_df


def generate_type_sheets(df: pd.DataFrame) -> dict:
    """Generate separate DataFrames for each component type."""
    sheets = {}
    
    for comp_type in df['Type'].unique():
        type_df = df[df['Type'] == comp_type].copy()
        type_df = type_df.sort_values('Qty', ascending=False)
        cols_to_keep = [c for c in type_df.columns if c != 'Type']
        type_df = type_df[cols_to_keep]
        sheets[comp_type] = type_df
    
    return sheets


def save_shopping_excel(shopping_df: pd.DataFrame, output_path: Path):
    """Save shopping list as Excel with separate sheets per type."""
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        shopping_df.to_excel(writer, sheet_name='All Components', index=False)
        
        type_sheets = generate_type_sheets(shopping_df)
        
        for comp_type, type_df in type_sheets.items():
            sheet_name = comp_type.capitalize()[:31]
            type_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Saved Excel shopping list to {output_path}")


# =============================================================================
# Main
# =============================================================================

def ensure_output_dir(path: Path) -> Path:
    """Ensure the output directory exists and return the full path."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    if path.parent == Path(".") or path.parent == Path(""):
        return output_dir / path
    
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_module_list(modules_str: str, available_modules: list[str]) -> list[str]:
    """Parse comma-separated module list and match against available modules.
    
    Supports case-insensitive partial matching.
    """
    if not modules_str:
        return []
    
    requested = [m.strip().lower() for m in modules_str.split(',')]
    matched = []
    
    for req in requested:
        # Try exact match first (case-insensitive)
        exact = [m for m in available_modules if m.lower() == req]
        if exact:
            matched.extend(exact)
            continue
        
        # Try partial match (module name contains the search term)
        partial = [m for m in available_modules if req in m.lower()]
        if partial:
            matched.extend(partial)
        else:
            print(f"Warning: No module matching '{req}' found")
    
    return list(set(matched))  # Deduplicate


def main():
    parser = argparse.ArgumentParser(description="Process NLC BOM data")
    parser.add_argument("input", type=Path, help="Input CSV file")
    parser.add_argument("-o", "--output", type=Path, help="Output cleaned CSV")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--shopping", type=Path, help="Output shopping list (CSV or XLSX)")
    parser.add_argument("--type", type=str, help="Filter to specific type (resistor, capacitor, ic, etc.)")
    parser.add_argument("--modules", type=str, help="Comma-separated list of module names to include (supports partial matching)")
    parser.add_argument("--list-modules", action="store_true", help="List all available module names and exit")
    parser.add_argument("--max-pn", type=int, default=3, help="Max part number columns per supplier (default: 3)")
    parser.add_argument("--no-output-dir", action="store_true", help="Save files in current directory instead of output/")
    
    args = parser.parse_args()
    
    global MAX_PN_COLUMNS
    MAX_PN_COLUMNS = args.max_pn
    
    df = process_bom(args.input)
    
    # List modules and exit if requested
    if args.list_modules:
        modules = sorted(df['Module'].unique())
        print(f"\nAvailable modules ({len(modules)}):")
        for m in modules:
            count = df[df['Module'] == m]['Quantity'].sum()
            print(f"  {m} ({count} components)")
        return
    
    # Filter by modules if specified
    if args.modules:
        available = df['Module'].unique().tolist()
        selected_modules = parse_module_list(args.modules, available)
        if selected_modules:
            print(f"\nFiltering to {len(selected_modules)} module(s): {', '.join(selected_modules)}")
            df = df[df['Module'].isin(selected_modules)].copy()
            print(f"  {len(df)} rows, {df['Quantity'].sum()} components")
        else:
            print("Error: No matching modules found")
            return
    
    # Output cleaned data
    if args.output:
        output_path = args.output if args.no_output_dir else ensure_output_dir(args.output)
        output_df = df[df['Type'] == args.type.lower()].copy() if args.type else df
        for col in ['Tayda_PNs', 'Mouser_PNs']:
            if col in output_df.columns:
                output_df[col] = output_df[col].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
        output_df.to_csv(output_path, index=False)
        print(f"\nSaved cleaned data to {output_path}")
    
    # Statistics
    if args.stats or not (args.output or args.shopping):
        stats = generate_stats(df)
        print_stats(stats)
    
    # Shopping list
    if args.shopping:
        shopping_path = args.shopping if args.no_output_dir else ensure_output_dir(args.shopping)
        shopping_input = df[df['Type'] == args.type.lower()].copy() if args.type else df
        if args.type:
            print(f"Filtered to {len(shopping_input)} rows of type '{args.type}'")
        shopping = generate_shopping_list(shopping_input)
        
        if shopping_path.suffix.lower() == '.xlsx':
            save_shopping_excel(shopping, shopping_path)
        else:
            shopping.to_csv(shopping_path, index=False)
            print(f"\nSaved shopping list to {shopping_path}")
        
        print(f"\n--- Shopping List Summary ---")
        print(f"Total unique components: {len(shopping)}")
        print(f"Total quantity: {shopping['Qty'].sum():,}")
        
        tayda_cols = [c for c in shopping.columns if c.startswith('Tayda_PN_')]
        has_tayda = shopping[tayda_cols].apply(lambda row: any(row != ''), axis=1).sum()
        print(f"Components with Tayda PN: {has_tayda}")
        
        mouser_cols = [c for c in shopping.columns if c.startswith('Mouser_PN_')]
        has_mouser = shopping[mouser_cols].apply(lambda row: any(row != ''), axis=1).sum()
        print(f"Components with Mouser PN: {has_mouser}")


if __name__ == "__main__":
    main()