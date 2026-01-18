# NLC BOM Tools

**Generate shopping lists from Nonlinearcircuits PDF build guides.**

If you've ordered a bunch of [NLC](https://www.nonlinearcircuits.com/) PCBs and want to know exactly what components to buy, this tool extracts the Bills of Materials from the PDF build guides and creates organized shopping lists.

## What This Tool Does

1. **Reads NLC PDF build guides** and extracts the component lists
2. **Combines and organizes** components across multiple modules
3. **Generates shopping lists** showing exactly what to order (with quantities!)
4. **Includes part numbers** for Tayda and Mouser where available

**Example:** You ordered 5 NLC modules. Instead of manually going through 5 PDFs and adding up how many 100K resistors you need total, this tool does it for you.

---

## Before You Start

You'll need to install a few things first. Don't worry—this only takes a few minutes!

### What You Need

| Requirement | What it is | How to check if you have it |
|-------------|-----------|----------------------------|
| **Python 3.8+** | Programming language that runs the tool | Open Terminal and type `python3 --version` |
| **Java** | Required for reading PDFs | Type `java -version` in Terminal |
| **The NLC PDF build guides** | Download from your NLC order or the [NLC website](https://www.nonlinearcircuits.com/) | — |

### Installing Python (if needed)

**Mac:**
```bash
# Install Homebrew first (if you don't have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install Python
brew install python
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and run the installer. **Important:** Check the box that says "Add Python to PATH" during installation.

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Installing Java (if needed)

**Mac:**
```bash
brew install java
```

**Windows:**
Download from [java.com](https://www.java.com/download/) and run the installer.

**Linux:**
```bash
sudo apt install default-jre
```

---

## Installation

### Step 1: Download this tool

Download and unzip this repository, or clone it:
```bash
git clone https://github.com/yourusername/nlc-bom-tools.git
cd nlc-bom-tools
```

### Step 2: Set up the Python environment

Open Terminal (Mac/Linux) or Command Prompt (Windows), navigate to the folder, and run:

```bash
# Create a virtual environment (keeps things tidy)
python3 -m venv venv

# Activate it
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows

# Install the required packages
pip install tabula-py pandas openpyxl
```

**Note:** You'll need to run `source venv/bin/activate` (or `venv\Scripts\activate` on Windows) each time you open a new Terminal window to use this tool.

---

## How to Use

### The Basic Workflow

1. **Put your NLC PDF build guides in a folder** (e.g., `~/nlc-pdfs/`)
2. **Extract the BOMs** from the PDFs into a combined file
3. **Generate a shopping list** from that file

### Step 1: Extract BOMs from your PDFs

```bash
python3 nlc_bom_extractor.py ~/nlc-pdfs/ -o csv --combine
```

This creates `output/all_boms_combined.csv` containing all the component data.

**What the options mean:**
- `~/nlc-pdfs/` — The folder containing your PDF files
- `-o csv` — Output as a CSV file
- `--combine` — Combine all PDFs into one file

### Step 2: Generate your shopping list

```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.xlsx
```

This creates `output/shopping.xlsx` — an Excel file with your complete shopping list!

---

## Common Tasks

### "I only want a shopping list for specific modules"

First, see what modules are available:
```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --list-modules
```

Then generate a list for just those modules:
```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.xlsx \
  --modules "Sloth Chaos,Divine CMOS,4SEQ"
```

**Tip:** Module matching is flexible—`"sloth"` will match "1u Sloth Chaos", "Super Sloth", etc.

### "I just want to see the ICs I need"

```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping ics.csv --type ic
```

Other types you can filter: `resistor`, `capacitor`, `transistor`, `diode`, `connector`, `pot`, `led`

### "I want to see statistics about my build"

```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --stats
```

This shows you things like:
- Total component count
- Most-used components
- Which modules have the most parts

### "I want a CSV instead of Excel"

Just change the file extension:
```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.csv
```

---

## Understanding the Output

### Shopping List Columns

| Column | What it means |
|--------|--------------|
| **Component** | What the part is (e.g., "100K", "TL072", "3.5mm Jack") |
| **Type** | Category (resistor, capacitor, ic, etc.) |
| **Qty** | How many you need total |
| **Package** | Size/footprint (0805, DIP, etc.) |
| **Tayda_PN_1/2/3** | Tayda Electronics part numbers |
| **Mouser_PN_1/2/3** | Mouser part numbers |
| **Modules** | How many of your modules use this part |

### Excel Output

The Excel file has multiple sheets:
- **All Components** — Everything in one list
- **Resistor** — Just resistors
- **Capacitor** — Just capacitors
- **Ic** — Just ICs
- etc.

---

## Complete Command Reference

### nlc_bom_extractor.py

Extracts component lists from PDF build guides.

```bash
python3 nlc_bom_extractor.py <pdf_folder> [options]
```

| Option | What it does |
|--------|-------------|
| `-o csv` | Output as CSV (default) |
| `-o excel` | Output as Excel |
| `--combine` | Combine all PDFs into one file |
| `-r` | Search subfolders too |
| `-d folder` | Save output to a specific folder |

### nlc_bom_processor.py

Processes extracted data and generates shopping lists.

```bash
python3 nlc_bom_processor.py <input_file> [options]
```

| Option | What it does |
|--------|-------------|
| `--shopping file.xlsx` | Generate shopping list (Excel) |
| `--shopping file.csv` | Generate shopping list (CSV) |
| `--modules "A,B,C"` | Only include specific modules |
| `--list-modules` | Show all available module names |
| `--type resistor` | Only include one component type |
| `--stats` | Show statistics |
| `-o file.csv` | Export cleaned/normalized data |
| `--max-pn 5` | More part number columns (default: 3) |
| `--no-output-dir` | Save in current folder instead of output/ |

---

## Troubleshooting

### "command not found: python3"

Python isn't installed or isn't in your PATH. See the installation instructions above.

**Quick fix for Mac:**
```bash
brew install python
```

### "No module named 'tabula'" or "No module named 'pandas'"

You need to activate the virtual environment and/or install dependencies:
```bash
source venv/bin/activate    # Mac/Linux
pip install tabula-py pandas openpyxl
```

### "Error: No Java runtime present"

Java isn't installed. See the installation instructions above.

**Quick fix for Mac:**
```bash
brew install java
```

### Lots of "WARNING: Could not load font file" messages

These are harmless—the extraction still works. You can hide them:
```bash
python3 nlc_bom_extractor.py ~/pdfs/ --combine 2>/dev/null
```

### "No matching modules found"

The module name might be slightly different. Check what's available:
```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --list-modules
```

Module matching is case-insensitive and partial, so try shorter names like `"sloth"` instead of `"Sloth Chaos"`.

### "FileNotFoundError: No such file or directory"

Make sure you're specifying the correct path. The input file is usually in the `output/` folder:
```bash
python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping shopping.xlsx
```

---

## Tips

- **Start small:** Test with one or two PDFs first to make sure everything works
- **Check the output:** Open the Excel file and verify it looks reasonable before ordering parts
- **Part numbers aren't perfect:** Not every component has a Tayda/Mouser part number in the PDF—you may need to look some up manually
- **Back up your PDFs:** Keep your original NLC build guide PDFs somewhere safe

---

## Getting Help

If you run into problems:

1. Make sure Python and Java are installed correctly
2. Make sure you've activated the virtual environment (`source venv/bin/activate`)
3. Check the troubleshooting section above
4. Look at the exact error message—it usually tells you what's wrong

---

## Credits

- [Nonlinearcircuits](https://www.nonlinearcircuits.com/) — Andrew Fitch's amazing DIY synth modules
- [tabula-py](https://github.com/chezou/tabula-py) — PDF table extraction

## License

MIT License — feel free to use and modify.