.PHONY: all clean data figures tables

# Default target
all: experiments pdf

# Run Python experiments to generate data and tables
experiments:
	@echo "Running experiments to generate tables and data..."
	python3 scripts/run_experiments.py

# Compile the LaTeX document
pdf:
	@echo "Compiling LaTeX document..."
	cd src && pdflatex main.tex
	cd src && bibtex main || true
	cd src && pdflatex main.tex
	cd src && pdflatex main.tex

# Clean generated files (excluding the ones tracked in git ideally, but for now just cleanup latex aux files)
clean:
	@echo "Cleaning up LaTeX auxiliary files..."
	cd src && rm -f *.aux *.log *.out *.bbl *.blg *.toc *.bcf *.xml *.snm *.nav
