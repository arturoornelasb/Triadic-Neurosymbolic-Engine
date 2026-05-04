# Makefile for Triadic Neurosymbolic Engine & Paper Compilation

.PHONY: all experiments paper clean

PAPER_DIR = ./paper
PYTHON = python
SRC_DIR = $(PAPER_DIR)/src
PAPER_NAME = PrimeFactorization_NeurosymbolicBridge_OrnelasBrand_2026

all: experiments paper

experiments:
	@echo "Running massive experiments and generating LaTeX tables..."
	$(PYTHON) $(PAPER_DIR)/scripts/run_experiments.py
	@echo "Experiments complete. Tables saved to $(PAPER_DIR)/tables/"

paper:
	@echo "Compiling the academic paper..."
	cd $(SRC_DIR) && pdflatex -interaction=nonstopmode main.tex
	cd $(SRC_DIR) && bibtex main || true
	cd $(SRC_DIR) && pdflatex -interaction=nonstopmode main.tex
	cd $(SRC_DIR) && pdflatex -interaction=nonstopmode main.tex
	cp $(SRC_DIR)/main.pdf $(PAPER_DIR)/$(PAPER_NAME).pdf
	@echo "Compilation successful. PDF generated at $(PAPER_DIR)/$(PAPER_NAME).pdf"

clean:
	@echo "Cleaning up LaTeX auxiliary files..."
	cd $(SRC_DIR) && rm -f main.aux main.log main.out main.bbl main.blg main.toc
	cd $(PAPER_DIR) && rm -f $(PAPER_NAME).aux $(PAPER_NAME).log $(PAPER_NAME).out $(PAPER_NAME).bbl $(PAPER_NAME).blg $(PAPER_NAME).toc
