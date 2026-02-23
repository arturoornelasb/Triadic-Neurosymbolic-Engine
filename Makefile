# Makefile for Triadic Neurosymbolic Engine & Paper Compilation

.PHONY: all experiments paper clean

PAPER_DIR = ./paper
PYTHON = python
SRC_DIR = $(PAPER_DIR)/src

all: experiments paper

experiments:
	@echo "Running massive experiments and generating LaTeX tables..."
	$(PYTHON) $(PAPER_DIR)/scripts/run_experiments.py
	@echo "Experiments complete. Tables saved to $(PAPER_DIR)/tables/"

paper:
	@echo "Compiling the academic paper..."
	cd $(SRC_DIR) && pdflatex main.tex
	cd $(SRC_DIR) && bibtex main || true
	cd $(SRC_DIR) && pdflatex main.tex
	cd $(SRC_DIR) && pdflatex main.tex
	@echo "Compilation successful. PDF generated at $(SRC_DIR)/main.pdf"

clean:
	@echo "Cleaning up LaTeX auxiliary files..."
	cd $(SRC_DIR) && rm -f *.aux *.log *.out *.bbl *.blg *.toc
