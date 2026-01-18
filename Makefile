.PHONY: help install test example clean

help:
	@echo "NLC BOM Tools"
	@echo ""
	@echo "Usage:"
	@echo "  make install    Install dependencies"
	@echo "  make example    Run example with sample data"
	@echo "  make clean      Remove output files"
	@echo ""
	@echo "Manual usage:"
	@echo "  python3 nlc_bom_extractor.py <pdf_dir> -o csv --combine"
	@echo "  python3 nlc_bom_processor.py all_boms_combined.csv --shopping shopping.xlsx"

install:
	pip install -r requirements.txt

example:
	@echo "Processing example input..."
	python3 nlc_bom_processor.py examples/sample_input.csv --stats --shopping output/example_shopping.csv
	@echo ""
	@echo "Output saved to output/example_shopping.csv"

clean:
	rm -rf output/*.csv output/*.xlsx output/*.json
	rm -rf __pycache__
