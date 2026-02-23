# Triadic Neurosymbolic Engine

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18748671.svg)](https://doi.org/10.5281/zenodo.18748671)

**A deterministic algebraic framework for neurosymbolic validation, semantic projection, and AI model auditing.**

Cosine similarity tells you *"King and Queen are 0.87 similar"* — a black-box number.

The Triadic Engine tells you *"King = 2×3×5 and Queen = 2×5×7. They share {2,5} (Royalty). King has {3} (Male) that Queen lacks. Queen has {7} (Female) that King lacks."* — fully transparent, deterministic decomposition.

---

## Quickstart

```bash
# Install
pip install -e .

# Python API
from neurosym import ContinuousEncoder, DiscreteMapper, DiscreteValidator

encoder = ContinuousEncoder("all-MiniLM-L6-v2")

# Choose a projection mode:
mapper = DiscreteMapper(n_bits=8, projection="pca")       # Deterministic, corpus-adapted
# mapper = DiscreteMapper(n_bits=8, projection="random")   # Classic LSH
# mapper = DiscreteMapper(n_bits=8, projection="consensus") # Multi-seed noise filtering
# mapper = DiscreteMapper(n_bits=8, projection="contrastive",  # Supervised
#                         hypernym_pairs=[("Animal","Dog"), ("Vehicle","Car")])

embeddings = encoder.encode(["King", "Queen", "Man", "Woman"])
prime_map = mapper.fit_transform(["King", "Queen", "Man", "Woman"], embeddings)

validator = DiscreteValidator()
print(validator.subsumes(prime_map["King"], prime_map["Queen"]))  # Subsumption check
print(validator.explain_gap(prime_map["King"], prime_map["Queen"]))  # Gap analysis
```

## How It Works

```
Text → Neural Embedding → LSH Hyperplanes → Composite Prime Integer
         (R^384)            (k projections)      (Φ(x) = ∏ pᵢ)
```

Each concept becomes a single integer whose **prime factors are its semantic features**. This enables three operations **impossible** under cosine similarity:

| Operation | Math | What it answers |
|-----------|------|----------------|
| **Subsumption** | `Φ(A) mod Φ(B) == 0` | "Does A contain every feature of B?" |
| **Composition** | `lcm(Φ(A), Φ(B))` | "What concept has all features of both A and B?" |
| **Gap Analysis** | `gcd(Φ(A), Φ(B))` + quotients | "Which features do they share? Which are unique?" |

## Projection Modes

| Mode | Deterministic | Requires Labels | Best For |
|------|:---:|:---:|------------|
| `random` | ✗ (seed-dependent) | ✗ | Baseline, exploration |
| `pca` | ✓ | ✗ | Production, reproducibility |
| `consensus` | ✓ | ✗ | Noise filtering, stability analysis |
| `contrastive` | ✓ | ✓ (hypernym pairs) | Maximum accuracy (100% TP at k=6) |

## Core Modules

| Module | Description |
|--------|-------------|
| `neurosym.encoder` | Multi-backend embedding encoder (HuggingFace, OpenAI, Cohere) + 4-mode LSH→Prime projection |
| `neurosym.triadic` | Algebraic validation: subsumption, composition, abductive gap analysis |
| `neurosym.graph` | Scalable graph builder with inverted prime index (avoids O(N²)) |
| `neurosym.storage` | SQLite persistence for prime indices and audit results |
| `neurosym.reports` | Exportable reports in HTML, JSON, and CSV formats |
| `neurosym.ingest` | Database ingestion pipeline with batch processing |
| `neurosym.anomaly` | Multiplicative anomaly detection for tabular data |

## Interactive Dashboard

```bash
streamlit run app.py
```

Five tabs: **Ingestion**, **Semantic Graph**, **Logic & Search**, **AI Auditor**, **Benchmarks**

The AI Auditor compares how different embedding models structure the same concepts using topological shortest-path differencing — finding exact structural discrepancies between models.

## CLI Tools

```bash
# Massive topological audit (model vs model)
python scripts/triadic_auditor.py --input examples/data/wordnet_2k.csv --output reports/audit.csv

# PCA vs Random vs Consensus vs Contrastive benchmark
python scripts/benchmark_pca.py
```

## Benchmarks

- **28.4× faster** pairwise verification than cosine similarity (50K operations)
- **100% composition guarantee** verified across 5,671 word pairs
- **100% hypernym detection** with contrastive projection at k=6
- **108,694 discrepancies** found auditing 2M semantic chains across 2 models

## Academic Paper

The full paper with 9 experiments is in [`paper/`](paper/), compilable with:

```bash
cd paper
pdflatex -output-directory=. -jobname=PrimeFactorization_NeurosymbolicBridge_OrnelasBrand_2026 src/main.tex
bibtex PrimeFactorization_NeurosymbolicBridge_OrnelasBrand_2026
pdflatex -output-directory=. -jobname=PrimeFactorization_NeurosymbolicBridge_OrnelasBrand_2026 src/main.tex
pdflatex -output-directory=. -jobname=PrimeFactorization_NeurosymbolicBridge_OrnelasBrand_2026 src/main.tex
# Or simply run `make paper` from the root directory
```

## Citation

```bibtex
@software{ornelas2026triadic,
  author       = {Ornelas Brand, J. Arturo},
  title        = {Triadic Neurosymbolic Engine: Prime Factorization as a
                  Neurosymbolic Bridge for Deterministic Verification},
  year         = 2026,
  url          = {https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine}
}
```

## Project Structure

```
├── src/neurosym/          ← Core Python package (pip installable)
├── paper/                 ← Academic paper (LaTeX, 11 pages)
├── app.py                 ← Streamlit interactive dashboard
├── scripts/               ← CLI auditing & benchmark tools
├── tests/                 ← Test suite
├── examples/              ← Usage examples & sample data
└── pyproject.toml         ← Package metadata & dependencies
```

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You may share and adapt this work for non-commercial purposes with attribution. For commercial licensing inquiries, contact: arturoornelas62@gmail.com

© 2026 José Arturo Ornelas Brand
