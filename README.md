# Triadic Neurosymbolic Engine

[![License: BUSL-1.1](https://img.shields.io/badge/License-BUSL--1.1-blue.svg)](https://mariadb.com/bsl11/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/neurosym.svg)](https://pypi.org/project/neurosym/)
[![CI](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18748671.svg)](https://doi.org/10.5281/zenodo.18748671)

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
# Local research / open-source (CC BY-NC 4.0)
pip install neurosym

# Optional extras
pip install "neurosym[dashboard]"  # Streamlit dashboard
pip install "neurosym[api]"        # FastAPI server
pip install "neurosym[dev]"        # Development tools
```

> **Hosted Cloud API (no GPU needed, commercial use):**
> ```bash
> pip install neurosym-cloud
> ```
> See [Triadic Cloud API →](https://fuaflow.com/triadic/)

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
# → {"subsumes": False, "shared": [2, 5], "a_only": [3], "b_only": [7]}

print(validator.explain_gap(prime_map["King"], prime_map["Queen"]))
# → "King and Queen share {Royalty}. King has {Male}. Queen has {Female}."
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
| `neurosym.ingest` | Database ingestion pipeline with batch processing |
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
pip install "neurosym[dashboard]"
streamlit run app.py
```

Five tabs: **Ingestion**, **Semantic Graph**, **Logic & Search**, **AI Auditor**, **Benchmarks**

The AI Auditor compares how different embedding models structure the same concepts using topological shortest-path differencing — finding exact structural discrepancies between models.

---

## CLI Tools

```bash
# Massive topological audit (model vs model)
python scripts/triadic_auditor.py --input examples/data/wordnet_2k.csv --output reports/audit.csv

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

## Triadic Cloud API

The open-source engine runs locally. For production workloads without GPU setup, the **[Triadic Cloud API](https://fuaflow.com/triadic/)** is the hosted version:

```python
# pip install neurosym-cloud
from neurosym_cloud import TriadicClient

client = TriadicClient(api_key="tne-...")
result = client.encode(["King", "Queen"])
print(client.subsumes("King", "Queen"))
```

| Tier | Price | Requests/day |
|------|-------|-------------|
| Community | Free | 100 |
| Pro | $29/mo | 5,000 |
| Enterprise | $299/mo | Unlimited + SLA |

---

## Academic Paper

Full paper with 9 experiments: [`paper/`](paper/)

```bash
cd paper
make paper   # requires pdflatex + bibtex
```

---

## Citation

```bibtex
@software{ornelas2026triadic,
  author       = {Ornelas Brand, J. Arturo},
  title        = {Triadic Neurosymbolic Engine: Prime Factorization as a
                  Neurosymbolic Bridge for Deterministic Verification},
  year         = 2026,
  doi          = {10.5281/zenodo.18748671},
  url          = {https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine}
}
```

---

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

---

## License

**Business Source License 1.1 (BUSL-1.1)**

| | Permitido |
|---|---|
| Uso académico / investigación / proyectos personales | ✅ Sin restricción |
| Self-hosted / tooling interno | ✅ Sin restricción |
| Producción propia (no competidora) | ✅ Sin restricción |
| Ofrecer una API de búsqueda semántica neurosimbólica a terceros | ❌ Requiere licencia comercial |

**Change Date:** 2040-01-01 → convierte automáticamente a Apache 2.0.

Para uso comercial como servicio de API, usa el [Triadic Cloud API](https://fuaflow.com/triadic/) o contacta: arturoornelas62@gmail.com

© 2026 José Arturo Ornelas Brand
