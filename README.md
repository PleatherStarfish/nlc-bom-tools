# NLC BOM Tools

Extract, normalize, and analyze Bills of Materials from [Nonlinearcircuits](https://www.nonlinearcircuits.com/) PDF build guides.

## Features

- **Extract BOMs from PDFs** - Handles multiple table formats (side-by-side, embedded quantities, variant tables)
- **Normalize component values** - `100k`, `100K`, `100k (c)` → `100K`
- **Extract supplier part numbers** - Tayda (A-1234) and Mouser (595-TL072CP) with multiple PNs per component
- **Generate shopping lists** - CSV or Excel with separate sheets per component type
- **Filter by module** - Generate shopping lists for specific modules only
- **Filter by type** - Get just ICs, resistors, capacitors, etc.
- **Statistics** - Most used components, module complexity, component counts by type

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install tabula-py pandas openpyxl

# Note: tabula-py requires Java
# macOS: brew install java
# Ubuntu: sudo apt install default-jre
```

## Quick Start

### Step 1: Extract BOMs from PDFs

```bash
# Extract all PDFs in a directory to a combined CSV
python3 nlc_bom_extractor.py ~/nlc-pdfs/ -o csv --combine

# Output: ./output/all_boms_combined.csv
```

### Step 2: Process and generate shopping list

```bash
# Generate Excel shopping list with sheets per component type
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.xlsx

# Generate CSV shopping list
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv

# Show statistics
python3 nlc_bom_processor.py output/all_boms_combined.csv --stats
```

### Step 3: Filter to specific modules

```bash
# List all available modules
python3 nlc_bom_processor.py output/all_boms_combined.csv --list-modules

# Generate shopping list for specific modules only
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv \
  --modules "Sloth Chaos,Hyperchaos Deluxe,Divine CMOS"
```

## Usage

### nlc_bom_extractor.py

Extracts tables from NLC PDF build guides.

```bash
nlc_bom_extractor.py <input_path> [options]

Arguments:
  input_path          PDF file or directory containing PDFs

Options:
  -o, --output        Output format: csv, json, excel (default: csv)
  -d, --output-dir    Output directory (default: ./output)
  -c, --combine       Combine all BOMs into single file
  -r, --recursive     Search subdirectories for PDFs
  --mouser            Export in Mouser-compatible format
```

**Examples:**

```bash
# Single PDF
python3 nlc_bom_extractor.py "Sloth_build_and_BOM.pdf" -o csv

# Directory of PDFs, combined output
python3 nlc_bom_extractor.py ~/nlc-boms/ -o csv --combine

# Recursive search, Excel output
python3 nlc_bom_extractor.py ~/nlc-boms/ -o excel --combine -r

# Mouser-compatible CSV
python3 nlc_bom_extractor.py ~/nlc-boms/ --mouser --combine
```

### nlc_bom_processor.py

Normalizes, deduplicates, and analyzes extracted BOM data.

```bash
nlc_bom_processor.py <input_csv> [options]

Arguments:
  input_csv           CSV file from nlc_bom_extractor.py

Options:
  -o, --output        Output cleaned/normalized CSV
  --stats             Show statistics
  --shopping FILE     Generate shopping list (CSV or XLSX)
  --type TYPE         Filter to component type
  --modules LIST      Comma-separated module names to include (supports partial matching)
  --list-modules      List all available module names and exit
  --max-pn N          Max part number columns per supplier (default: 3)
  --no-output-dir     Save files in current directory instead of output/
```

**Examples:**

```bash
# Show statistics
python3 nlc_bom_processor.py output/all_boms_combined.csv --stats

# Generate shopping list as Excel (with per-type sheets)
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.xlsx

# Generate shopping list as CSV
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv

# List all modules in the BOM data
python3 nlc_bom_processor.py output/all_boms_combined.csv --list-modules

# Filter to specific modules (exact or partial match)
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv \
  --modules "Sloth,Hyperchaos,Divine CMOS"

# Filter to ICs only
python3 nlc_bom_processor.py output/all_boms_combined.csv --type ic --shopping ic_parts.csv

# Combine module and type filters
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv \
  --modules "Sloth Chaos,4SEQ" --type ic

# Export cleaned/normalized data
python3 nlc_bom_processor.py output/all_boms_combined.csv -o cleaned.csv

# More part number columns (default is 3)
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv --max-pn 5

# Save in current directory instead of output/
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv --no-output-dir

# Combine options
python3 nlc_bom_processor.py output/all_boms_combined.csv -o cleaned.csv --shopping shopping.xlsx --stats
```

**Valid `--type` values:**
`resistor`, `capacitor`, `ic`, `transistor`, `diode`, `connector`, `pot`, `led`, `sensor`, `vactrol`, `regulator`, `switch`, `other`

**Module filtering:**
- Case-insensitive matching
- Supports partial matching: `--modules "sloth"` matches "1u Sloth Chaos", "Super Sloth", etc.
- Comma-separated for multiple modules
- Use `--list-modules` to see available module names

## Output Formats

### Shopping List Columns

| Column | Description |
|--------|-------------|
| Component | Normalized name (100K, TL072/074, 3.5mm Jack Mono) |
| Type | resistor, capacitor, ic, transistor, etc. |
| Qty | Total quantity across all modules |
| Package | 0805, SOIC, SOT-23, DIP, THT, etc. |
| Tayda_PN_1 | First Tayda part number (A-1234 format) |
| Tayda_PN_2 | Second Tayda part number |
| Tayda_PN_3 | Third Tayda part number |
| Mouser_PN_1 | First Mouser part number |
| Mouser_PN_2 | Second Mouser part number |
| Mouser_PN_3 | Third Mouser part number |
| Modules | Number of modules using this component |

Note: Part number columns are configurable with `--max-pn` (default: 3).

### Excel Shopping List

The `.xlsx` output includes separate sheets:
- **All Components** - Complete list sorted by type then quantity
- **Resistor**, **Capacitor**, **Ic**, **Transistor**, etc. - Per-type sheets

### Cleaned BOM Columns

| Column | Description |
|--------|-------------|
| Module | Module name (cleaned) |
| Value | Normalized component value |
| Quantity | Count |
| Type | Component type |
| Package | Footprint/package |
| Tayda_PNs | Tayda part numbers (semicolon-separated) |
| Mouser_PNs | Mouser part numbers (semicolon-separated) |
| Original | Original value from PDF |
| Details | Notes/details from PDF |

## Output Directory

By default, all output files are saved to an `output/` subdirectory:

```bash
python3 nlc_bom_processor.py input.csv --shopping shopping.csv
# Creates: output/shopping.csv

python3 nlc_bom_processor.py input.csv -o cleaned.csv --shopping shopping.xlsx
# Creates: output/cleaned.csv and output/shopping.xlsx
```

Use `--no-output-dir` to save in the current directory instead.

## Component Normalization

The processor normalizes component values for deduplication:

| Original | Normalized | Type |
|----------|------------|------|
| `100k`, `100K`, `100k (c)` | `100K` | resistor |
| `4k7`, `4K7` | `4K7` | resistor |
| `100nF`, `100n`, `(104)` | `100nF` | capacitor |
| `TL072 or TL082`, `TL074` | `TL072/074` | ic |
| `LL4148`, `1N4148 diode` | `1N4148` | diode |
| `3.5MM SOCKET Kobiconn` | `3.5mm Jack Mono` | connector |
| `BC547`, `BC847` | `BC547`, `BC847` | transistor |

## Sample Output

### Statistics
```
======================================================================
NLC BOM STATISTICS
======================================================================

Total line items:     1,040
Total components:     5,320
Unique modules:       45
Unique part values:   162

----------------------------------------------------------------------
COMPONENTS BY TYPE
----------------------------------------------------------------------
  resistor              2,653
  capacitor               787
  connector               565
  ic                      251
  ...

----------------------------------------------------------------------
TOP 25 MOST USED COMPONENTS
----------------------------------------------------------------------
   1. 100K                              592
   2. 10K                               491
   3. 3.5mm Jack Mono                   380
   4. 1K                                291
   5. 100nF                             205
   ...
```

### Module List
```
Available modules (45):
  1u Difference Rectifier (32 components)
  1u Sloth Chaos (28 components)
  4SEQ (45 components)
  ARSEq (52 components)
  ...
```

### IC Shopping List
```
Component    Type  Qty  Package    Tayda_PN_1  Tayda_PN_2  Mouser_PN_1            Modules
TL072/074    ic    180  SOIC       A-1136      A-1137                             42
PT2399       ic     19  DIP; SOIC  A-1526      A-5781                             8
LM13700      ic     14  SOIC                               926-LM13700MX/NOPB     6
CD40106      ic      4  SOIC                               595-CD40106BM96        3
```

## Supported PDF Formats

The extractor handles various NLC BOM table formats:

1. **Standard tables** - Component | Quantity | Notes
2. **Side-by-side tables** - Two component columns per row
3. **Embedded quantities** - `TL072 (5)` format
4. **Variant tables** - Torpor/Apathy/Inertia style columns
5. **Continuation tables** - Tables spanning multiple pages

## Troubleshooting

### Java/PDFBox font warnings
```
WARNING: Could not load font file: /Library/Fonts/...
```
These are harmless - extraction still works. To suppress:
```bash
python3 nlc_bom_extractor.py ~/boms/ --combine 2>/dev/null
```

### Missing tabula-py
```bash
pip install tabula-py
# Requires Java: brew install java (macOS) or apt install default-jre (Ubuntu)
```

### Wrong Python version
```bash
# Use python3 explicitly
python3 nlc_bom_extractor.py ...

# Or check version
python --version  # Should be 3.8+
```

### No matching modules found
```bash
# Use --list-modules to see available module names
python3 nlc_bom_processor.py output/all_boms_combined.csv --list-modules

# Module names are case-insensitive and support partial matching
--modules "sloth"  # Matches "1u Sloth Chaos", "Super Sloth", etc.
```

## License

MIT License - feel free to use and modify.

## Credits

- [Nonlinearcircuits](https://www.nonlinearcircuits.com/) - Andrew Fitch's amazing DIY synth modules
- [tabula-py](https://github.com/chezou/tabula-py) - PDF table extraction