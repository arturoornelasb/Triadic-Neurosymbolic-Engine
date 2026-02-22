# Prime Factorization as a Neurosymbolic Bridge

Paper: *"Prime Factorization as a Neurosymbolic Bridge: Projecting Continuous Embeddings into Discrete Algebraic Space for Deterministic Verification"*

**Author:** Arturo Ornelas

## Repository Structure

```
├── src/                    # LaTeX source
│   ├── main.tex            # Main paper
│   └── references.bib      # Bibliography
├── scripts/                # Experiment scripts
│   └── run_experiments.py  # Full experimental suite & LaTeX table generation
├── figures/                # Generated figures
├── tables/                 # Generated LaTeX tables
├── data/                   # Raw experiment data (CSV)
└── README.md
```

## Reproducing Experiments

```bash
# Install dependencies
pip install sentence-transformers numpy pandas

# Fast Build: Run all experiments AND compile the paper automatically
make all

# Or run steps individually:
make experiments   # Runs scripts/run_experiments.py
make pdf           # Compiles src/main.tex into PDF
```

## License

MIT
