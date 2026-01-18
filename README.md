# NLC BOM Tools

Extract and normalize Bills of Materials from [Nonlinearcircuits](https://www.nonlinearcircuits.com/) PDF build guides. Generates shopping lists with Tayda/Mouser part numbers.

## Features

- **Extract BOMs from PDFs** - Handles multiple table formats (side-by-side, embedded quantities, variant tables)
- **Normalize component values** - `100k`, `100K`, `100k (c)` → `100K`
- **Extract supplier part numbers** - Tayda (A-1234) and Mouser (595-TL072CP)
- **Generate shopping lists** - CSV or Excel with separate sheets per component type
- **Filter by type** - Get just ICs, resistors, capacitors, etc.
- **Statistics** - Most used components, module complexity, component counts by type

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Drop your NLC PDF build guides into boms/

# 3. Run everything
make all

# Output: output/shopping.xlsx
```

## Folder Structure

```
nlc-bom-tools/
├── boms/                 ← Put your NLC PDFs here
├── output/               ← Generated files appear here
│   ├── all_boms_combined.csv
│   └── shopping.xlsx
├── nlc_bom_extractor.py  # PDF → CSV
├── nlc_bom_processor.py  # CSV → shopping list
├── Makefile
└── requirements.txt
```

## Installation

```bash
# Install Python dependencies
make install

# Note: tabula-py requires Java
# macOS: brew install java
# Ubuntu: sudo apt install default-jre
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Make Commands

```bash
make install    # Install dependencies
make extract    # Extract BOMs from PDFs in boms/
make process    # Generate shopping list from extracted data
make all        # Run extract + process
make clean      # Remove output files
```

### Manual Usage

#### Step 1: Extract BOMs from PDFs

```bash
python3 nlc_bom_extractor.py boms/ -o csv --combine -d output
```

#### Step 2: Generate shopping list

```bash
# Excel (with separate sheets per component type)
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping output/shopping.xlsx

# CSV
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping output/shopping.csv

# Filter to specific type
python3 nlc_bom_processor.py output/all_boms_combined.csv --type ic --shopping output/ic_parts.csv

# Show statistics
python3 nlc_bom_processor.py output/all_boms_combined.csv --stats
```

**Valid `--type` values:**
`resistor`, `capacitor`, `ic`, `transistor`, `diode`, `connector`, `pot`, `led`, `sensor`, `vactrol`, `regulator`, `switch`, `other`

## Output Format

### Shopping List Columns

| Column | Description |
|--------|-------------|
| Component | Normalized name (100K, TL072/074, 3.5mm Jack Mono) |
| Type | resistor, capacitor, ic, transistor, etc. |
| Qty | Total quantity across all modules |
| Package | 0805, SOIC, SOT-23, DIP, THT, etc. |
| Part_Number | Best available PN (prefers Tayda) |
| Tayda_PN | Tayda part numbers (A-1234 format) |
| Mouser_PN | Mouser part numbers |
| Modules | Number of modules using this component |

### Sample Output

```
Component     Type        Qty  Package  Part_Number  Tayda_PN
100K          resistor    592  0805
TL072/074     ic          180  SOIC     A-1136       A-1136; A-1137; A-1139
3.5mm Jack    connector   380           A-2563       A-2563; A-865
PT2399        ic           19  DIP      A-1526       A-1526; A-5781
BC847         transistor   68  SOT-23   A-1339       A-1339
```

## Component Normalization

| Original | Normalized | Type |
|----------|------------|------|
| `100k`, `100K`, `100k (c)` | `100K` | resistor |
| `4k7`, `4K7` | `4K7` | resistor |
| `100nF`, `100n`, `(104)` | `100nF` | capacitor |
| `TL072 or TL082` | `TL072/074` | ic |
| `LL4148`, `1N4148 diode` | `1N4148` | diode |
| `3.5MM SOCKET Kobiconn` | `3.5mm Jack Mono` | connector |

## Troubleshooting

### Java/PDFBox font warnings
```
WARNING: Could not load font file: /Library/Fonts/...
```
Harmless - extraction still works. Suppress with `2>/dev/null`.

### Missing Java
```bash
# macOS
brew install java

# Ubuntu
sudo apt install default-jre
```

## License

MIT

## Credits

- [Nonlinearcircuits](https://www.nonlinearcircuits.com/) - Andrew Fitch's DIY synth modules
- [tabula-py](https://github.com/chezou/tabula-py) - PDF table extraction
