# Triadic Neurosymbolic Engine (`neurosym`)

**PROPRIETARY AND CONFIDENTIAL**
*This software and its underlying mathematical framework are proprietary intellectual property. It is not open-source and is strictly intended for commercial monetization and private research. Unauthorized copying, distribution, or use is strictly prohibited.*

A Deterministic Algebraic Framework for Neurosymbolic Validation, Semantic Projection, and AI Model Auditing.

This library shifts the paradigm from probabilistic floating-point "black boxes" to deterministic, discrete mathematics using modular arithmetic, homological type theory invariants, and entropic graph theory. By projecting continuous vector embeddings into $O(1)$ integer Prime Factor spaces, it enables **perfect mathematical interpretability** of Large Language Models and embedding spaces.

## 🚀 Novel Key Feature: Topological Semantic Auditing

With the `Triadic AI Auditor`, we introduce a mathematically rigorous way to compute the "Diff" (discrepancy) between two different opaque AI neural networks using **Topological Shortest-Path differencing**.

Instead of merely comparing raw cosine similarity vectors, the Triadic Engine maps two different AI models (e.g., `all-MiniLM` vs `paraphrase-MiniLM`) onto discrete deterministic graphs. It then calculates the exact Chains of Thought (shortest paths via `NetworkX`) that an AI makes to connect Concept A with Concept B. 

**This allows data scientists and enterprise users to:**
- Identify exactly when an Enterprise RAG model mathematically severs critical semantic links (e.g., Doctor -> Medicine -> Patient).
- Quantify Cognitive Biases using $O(1)$ Integer Divisibility without falling back to fuzzy T-SNE plots.

## 📦 Core Architecture (`src/neurosym`)

The engine is composed of updated modules designed to bridge the gap between continuous latent semantic spaces and rigorous discrete logic:

1. **`neurosym.encoder` (Continuous to Discrete Projection)** 
   Extracts embeddings from PyTorch sentence-transformers and hashes them into orthogonal Locality-Sensitive Hashing hyperplanes, ultimately discretizing them into Prime Factors.
   
2. **`neurosym.triadic` (Algebraic Validation Module)**
   Projects continuous data into a domain of integers (factors/primes). Evaluates relationships using Homotopy Type Theory invariants and elementary modular arithmetic (`math.gcd`). Discovers "hidden variables" or structurally missing factors through topological obstruction detection.
   
3. **`neurosym.ingest` (Database Ingestion)**
   Hooks the mathematical engine into standard Postgres SQL databases and SQLite datalakes.
   
4. **`neurosym.anomaly` (Symbolic Anomaly Detection)**
   Runs deterministic semantic gap analysis to find hallucinations that break mathematical transitiveness.

## 🖥️ Interactive Web Dashboard

To run the full Interactive Graph and Triadic Auditor UI:

```bash
pip install -e .
streamlit run app.py
```

Features 4 modules:
1. **Ingestion & Encoding:** Upload CSV dictionaries.
2. **Holographic Graph:** `streamlit-agraph` physical force-directed semantic graphs.
3. **Logic & Search:** Run Arithmetic Abduction (Subsumption).
4. **AI Auditor:** The Model-vs-Model Relational Diff feature.

## 🛠️ CLI Tools & Auditing

### Massive Topological AI Auditor
To audit large databases (e.g., WordNet 2,000 concepts) and generate a CSV diff:

```bash
python scripts/triadic_auditor.py --input examples/data/wordnet_2k.csv --output reports/wordnet_2k_topological_audit.csv
```

### Academic Experiments Generator
To reproduce the $O(1)$ vs Cosine Similarity benchmarks from the scientific paper:

```bash
python scripts/run_experiments.py
```
*Note: This generates LaTeX ready `tables/` for inclusion in the formal paper.*
