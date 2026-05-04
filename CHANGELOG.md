# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Paper Table 3 (Subsumption Accuracy): false-positive rates now reflect the latest 10-seed run (commit `20e407c` regenerated CSVs but had not re-flowed into `main.tex`).
- Paper Table 5 (Analogy Resolution): `k=3` and `k=6` corrected from `1/50 (2.0%)` to `0/50 (0.0%)`.
- Paper Experiment 7: clarified that the `28.4×` benchmark measures `analogy_prediction` (a richer operation) on `N=20{,}000`, distinct from Experiment 1's `30.9×` single-pair GCD baseline.
- Paper Experiment 8: corrected scale from `20{,}000-concept` to `2{,}000-concept` WordNet audit (matches the actual `wordnet_2k.csv` used; `1{,}999{,}000` chains, `108{,}694` discrepancies).
- Paper Table 9: added note about Contrastive variability (100% TP at `k=6` is single-run; deterministic sorted-vocab gives 96.2%, range 92–100%).
- Paper abstract: pairwise verification speedup updated from `28.4×` (Exp 7) to `30.9×` (Exp 1) to match the paper's primary timing table and conclusion.
- Makefile `paper` target: now compiles from `paper/src/` so relative paths (`../figures/`, `references.bib`) resolve correctly; previous behavior produced PDFs without the pipeline figure or citations.
- Makefile `clean` target: no longer deletes committed PDF artifacts.
- BibTeX: `kotlerman2010directional` re-typed from `@inproceedings` to `@article` (it appears in *Natural Language Engineering* journal); resolves the `volume`/`number` warning.
- `pyproject.toml` description: speedup figure aligned with Exp 1 (`28.4x` → `30.9x`).
- `.zenodo.json` description: same speedup correction.
- `src/neurosym/reports.py`: `datetime.utcnow()` (deprecated in 3.13) replaced with `datetime.now(timezone.utc)`.
- README REST API table: added the four endpoints already implemented in `api/server.py` (`/subsumes`, `/compose`, `/gap`, `/analogy`).
- README Benchmarks table: speedup figures now distinguish naive (`30.9×`) and pre-normalized (`5.4×`) baselines; Contrastive 100% TP qualified with the deterministic 96.2% reference; audit scale specified as WordNet 2K.
- ROADMAP: marked the four REST endpoints as implemented; corrected test-coverage status for `PrimeIndexDB`, `ScalableGraphBuilder`, and `ReportGenerator` (all three have full coverage, not zero).

### Changed
- `pyproject.toml`: `[tool.setuptools.packages.find]` now excludes `neurosym._archived*` so deprecated modules (`buss.py`, `uhrt.py`) no longer ship in the wheel/sdist.
- Internal `__version__` strings synchronized to `0.3.0` across `src/neurosym/__init__.py`, `src/neurosym/reports.py`, `api/server.py`, and `api/models.py` (had drifted to `0.2.0` after the `c100836` bump).

## [0.3.0] — 2026-03-25

### Added
- Bundled Streamlit dashboard inside the package; `triadic-dashboard` CLI entry point.
- REST API endpoints `/subsumes`, `/compose`, `/gap`, `/analogy` exposing the algebraic primitives over HTTP.
- Held-out subsumption evaluation (Experiment 10) on 67 unseen hypernym pairs across 14 categories.
- 4-mode projection benchmark: random, PCA, consensus, contrastive (Experiment 9).
- Companion-work DOIs (`triadic-microgpt`, `reptimeline`, `triadic-emergent-duality`) referenced from paper and README.

### Changed
- Paper expanded to 10 experiments, including industrial-scale benchmark (`N=20{,}000`) and topological bias auditing.
- Author name standardized to "J. Arturo Ornelas Brand" everywhere.
- License unified to BUSL-1.1 (auto-converts to AGPL-3.0 on 2030-03-21); consortium participation model documented in `COMMERCIAL.md`.
- CORS wildcard replaced with env-based origins (`CORS_ORIGINS`) for security.

### Fixed
- Figure and bibliography paths in paper resolve correctly from `paper/src/`.
- `tqdm` progress bars suppressed during model loading (Streamlit compatibility).
- `__version__` synced to `0.2.0` to match `pyproject.toml` (later drifted again — see Unreleased).

## [0.2.0] — 2026-03-21

Initial PyPI release as `triadic-engine`.

### Added
- Public release with reproducible Jupyter notebook (`notebooks/Reproducibility_Demo.ipynb`).
- Academic LaTeX paper merged into the codebase under `paper/`.
- Multi-backend embedding encoder (HuggingFace/OpenAI/Cohere) with 4-mode LSH→Prime projection.
- Scalable graph builder with inverted prime index (avoids O(N²)).
- HTML/JSON/CSV report generator and `/report` endpoint.
- SQLite persistence for prime indices and audit results (`PrimeIndexDB`).
- DataFrame ingestion + tabular anomaly detection.
- CI workflow: lint (`ruff`), tests (`pytest`), build (wheel + sdist).

[Unreleased]: https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/releases/tag/v0.3.0
[0.2.0]: https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/releases/tag/v0.2.0
