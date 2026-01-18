.PHONY: help install extract extract-content process all example clean test-names

help:
	@echo "NLC BOM Tools"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install dependencies"
	@echo "  make extract         Extract BOMs (filename-based naming, default)"
	@echo "  make extract-content Extract BOMs (content-based naming)"
	@echo "  make process         Process extracted data into shopping list"
	@echo "  make all             Run extract + process"
	@echo "  make test-names      Test module name resolution"
	@echo "  make example         Run example with sample data"
	@echo "  make clean           Remove output files"

install:
	pip install -r requirements.txt

# Default: use filename-based naming with overrides (more reliable)
extract:
	python3 nlc_bom_extractor.py boms/ -o csv --combine -d output --filename-priority --overrides module_overrides.json

# Alternative: content-based detection (may misidentify some modules)
extract-content:
	python3 nlc_bom_extractor.py boms/ -o csv --combine -d output --content-priority

process:
	python3 nlc_bom_processor.py output/all_boms_combined.csv --shopping output/shopping.xlsx --stats

all: extract process

# Test module name resolution without running full extraction
test-names:
	@python3 module_name_resolver.py

example:
	@echo "Processing example input..."
	python3 nlc_bom_processor.py examples/sample_input.csv --stats --shopping output/example_shopping.csv
	@echo ""
	@echo "Output saved to output/example_shopping.csv"

clean:
	rm -rf output/*.csv output/*.xlsx output/*.json
	rm -rf __pycache__