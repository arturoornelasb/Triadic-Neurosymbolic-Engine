# Triadic Neurosymbolic Engine

[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](https://mariadb.com/bsl11/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/triadic-engine.svg)](https://pypi.org/project/triadic-engine/)
[![CI](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/actions)
[![DOI Software](https://zenodo.org/badge/DOI/10.5281/zenodo.18748671.svg)](https://doi.org/10.5281/zenodo.18748671)
[![DOI Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.19205805.svg)](https://doi.org/10.5281/zenodo.19205805)

**A deterministic algebraic framework for neurosymbolic validation, semantic projection, and AI model auditing.**

Cosine similarity tells you *"King and Queen are 0.87 similar"* — a black-box number.

The Triadic Engine tells you *"King = 2×3×5 and Queen = 2×5×7. They share {2,5} (Royalty). King has {3} (Male) that Queen lacks. Queen has {7} (Female) that King lacks."* — fully transparent, deterministic decomposition.

---

## Why not cosine similarity?

| | Cosine Similarity | **Triadic Engine** |
|---|:---:|:---:|
| Speed (50K pairs) | baseline | **28.4× faster** |
| Explainability | Black box | ✅ Prime factor proof |
| Subsumption (`A ⊆ B`?) | ❌ Approximation | ✅ Exact (`Φ(A) mod Φ(B) == 0`) |
| Composition (`A ∪ B`) | ❌ Geometric average | ✅ `lcm(Φ(A), Φ(B))` |
| Gap analysis | ❌ Not possible | ✅ `gcd` + quotient decomposition |
| Determinism | ❌ Seed-dependent | ✅ PCA / contrastive modes |
| AI model audit | ❌ Not supported | ✅ Topological discrepancy |

---

## Install

```bash
pip install triadic-engine

# With optional extras
pip install "triadic-engine[dashboard]"  # Streamlit dashboard
pip install "triadic-engine[api]"        # FastAPI server
```

---

## Quickstart

```python
from neurosym import ContinuousEncoder, DiscreteMapper, DiscreteValidator

encoder = ContinuousEncoder("all-MiniLM-L6-v2")

# Choose a projection mode:
mapper = DiscreteMapper(n_bits=8, projection="pca")        # Deterministic, corpus-adapted
# mapper = DiscreteMapper(n_bits=8, projection="random")    # Classic LSH
# mapper = DiscreteMapper(n_bits=8, projection="consensus") # Multi-seed noise filtering
# mapper = DiscreteMapper(n_bits=8, projection="contrastive",   # Supervised
#                         hypernym_pairs=[("Animal","Dog"), ("Vehicle","Car")])

concepts = ["King", "Queen", "Man", "Woman"]
embeddings = encoder.encode(concepts)
prime_map = mapper.fit_transform(concepts, embeddings)

validator = DiscreteValidator()

print(validator.subsumes(prime_map["King"], prime_map["Queen"]))
# → False (King does not contain ALL features of Queen)

print(validator.explain_gap(prime_map["King"], prime_map["Queen"]))
# → {"shared": 10, "only_in_a": 3, "only_in_b": 7, "a_contains_b": False, "b_contains_a": False}

print(validator.compose(prime_map["King"], prime_map["Queen"]))
# → LCM of both — a new integer containing all features of King AND Queen

# Analogy: King:Man :: Queen:?
result = validator.analogy_prediction(prime_map["King"], prime_map["Man"], prime_map["Queen"])
print(result.output_value)  # → predicted integer for "Woman"
```

---

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

---

## Projection Modes

| Mode | Deterministic | Requires Labels | Best For |
|------|:---:|:---:|------------|
| `random` | ✗ (seed-dependent) | ✗ | Baseline, exploration |
| `pca` | ✓ | ✗ | Production, reproducibility |
| `consensus` | ✓ | ✗ | Noise filtering, stability analysis |
| `contrastive` | ✓ | ✓ (hypernym pairs) | Maximum accuracy (100% TP at k=6) |

---

## Core Modules

| Module | Description |
|--------|-------------|
| `neurosym.encoder` | Multi-backend embedding encoder (HuggingFace, OpenAI, Cohere) + 4-mode LSH→Prime projection |
| `neurosym.triadic` | Algebraic validation: subsumption, composition, abductive gap analysis |
| `neurosym.graph` | Scalable graph builder with inverted prime index (avoids O(N²)) |
| `neurosym.storage` | SQLite persistence for prime indices and audit results |
| `neurosym.reports` | Exportable reports in HTML, JSON, and CSV formats |
| `neurosym.ingest` | DataFrame ingestion with inverted prime index and semantic search |
| `neurosym.anomaly` | Multiplicative anomaly detection for tabular data |

---

## Use Cases

**Explainable RAG** — Instead of returning top-k by cosine score, return documents whose prime signatures *subsume* the query signature. Every result is provably relevant.

**AI Model Auditing** — Detect when two LLMs structure the same concept differently. The engine found 108,694 discrepancies auditing 2M semantic chains across two embedding models.

**Semantic Deduplication** — Two records are semantically duplicate if `Φ(A) mod Φ(B) == 0`. Exact, not probabilistic.

**Compliance Validation** — Verify that "GDPR" subsumes "consent" and "data-subject-rights" in your ontology. Machine-checkable, not fuzzy.

**Anomaly Detection** — Tabular rows that break the multiplicative patterns of their peers are flagged as anomalies — with a proof, not just a score.

---

## Interactive Dashboard

```bash
pip install "triadic-engine[dashboard]"
triadic-dashboard
```

Six tabs: **Ingestion & Encoding**, **Semantic Graph**, **Logic & Search**, **AI Auditor**, **Anomaly Detection**, **Benchmarks**

The AI Auditor compares how different embedding models structure the same concepts using topological shortest-path differencing — finding exact structural discrepancies between models.

---

## REST API

```bash
pip install "triadic-engine[api]"
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Engine status and loaded concepts count |
| `/encode` | POST | Encode concepts into composite prime integers |
| `/audit` | POST | Compare two embedding models topologically |
| `/search` | POST | GCD-based semantic search over indexed concepts |
| `/report` | GET | Export engine state as HTML, JSON, or CSV |

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## CLI Tools

```bash
# Massive topological audit (model vs model)
python scripts/triadic_auditor.py --input examples/data/wordnet_2k.csv --col concept --output reports/audit.csv

# PCA vs Random vs Consensus vs Contrastive benchmark
python scripts/benchmark_pca.py
```

---

## Benchmarks

| Metric | Result |
|--------|--------|
| Pairwise verification speed | **28.4× faster** than cosine (50K operations) |
| Composition guarantee | **100%** verified across 5,671 word pairs |
| Hypernym detection accuracy | **100% TP** with contrastive projection at k=6 |
| Model audit scale | **108,694 discrepancies** in 2M semantic chains (2 models) |

---

## Academic Paper

Full paper with 9 experiments: [`paper/`](paper/)

```bash
make paper   # requires pdflatex + bibtex
```

---

## Citation

### Paper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19205805.svg)](https://doi.org/10.5281/zenodo.19205805)

Ornelas Brand, J. A. (2026). Triadic Neurosymbolic Engine: Prime Factorization as a Neurosymbolic Bridge: Projecting Continuous Embeddings into Discrete Algebraic Space for Deterministic Verification. Zenodo. https://doi.org/10.5281/zenodo.19205805

```bibtex
@article{ornelas2026prime,
  author       = {Ornelas Brand, J. Arturo},
  title        = {Triadic Neurosymbolic Engine: Prime Factorization as a
                  Neurosymbolic Bridge: Projecting Continuous Embeddings
                  into Discrete Algebraic Space for Deterministic Verification},
  year         = 2026,
  month        = mar,
  doi          = {10.5281/zenodo.19205805},
  url          = {https://doi.org/10.5281/zenodo.19205805}
}
```

### Repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18748671.svg)](https://doi.org/10.5281/zenodo.18748671)

Ornelas Brand, J. A. (2026). Prime Factorization as a Neurosymbolic Bridge: Projecting Continuous Embeddings into Discrete Algebraic Space for Deterministic Verification (Repository) (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.18748671

```bibtex
@software{ornelas2026triadic,
  author       = {Ornelas Brand, J. Arturo},
  title        = {Prime Factorization as a Neurosymbolic Bridge: Projecting
                  Continuous Embeddings into Discrete Algebraic Space
                  for Deterministic Verification (Repository)},
  year         = 2026,
  month        = feb,
  version      = {0.1.0},
  doi          = {10.5281/zenodo.18748671},
  url          = {https://doi.org/10.5281/zenodo.18748671}
}
```

---

## Project Structure

```
├── src/neurosym/          ← Core Python package (pip installable)
├── api/                   ← FastAPI REST server
├── app.py                 ← Streamlit interactive dashboard
├── paper/                 ← Academic paper (LaTeX, 12 pages)
├── scripts/               ← CLI auditing & benchmark tools
├── tests/                 ← Test suite
├── notebooks/             ← Reproducibility demo (Jupyter)
├── examples/              ← Sample datasets (WordNet, e-commerce)
└── pyproject.toml         ← Package metadata & dependencies
```

---

## License

**Business Source License 1.1 (BUSL-1.1)**

| | Allowed |
|---|---|
| Individuals / personal projects / freelancing | ✅ Free |
| Academic / research institutions | ✅ Free |
| Non-profit organizations | ✅ Free |
| For-profit companies (production use) | ❌ Requires participation agreement |

All users must contribute improvements back. See [TERMS.md](TERMS.md).
Companies: see [COMMERCIAL.md](COMMERCIAL.md) for the consortium participation model.

**Change Date:** 2030-03-21 — auto-converts to AGPL-3.0.

Contact: arturoornelas62@gmail.com

© 2026 J. Arturo Ornelas Brand
