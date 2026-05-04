# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] — 2026-05-03

### Added
- REST API endpoints `/subsumes`, `/compose`, `/gap`, `/analogy` exposing the algebraic primitives over HTTP (commit `682d2a3`).
- Held-out subsumption evaluation (Experiment 10) on 67 unseen hypernym pairs across 14 categories (commit `20e407c`).
- `CHANGELOG.md` — bootstrapped version history (this file).
- Companion-work DOIs (`triadic-microgpt`, `reptimeline`, `triadic-emergent-duality`) referenced from paper and README.

### Changed
- Paper Table 3 (Subsumption Accuracy): false-positive rates re-aligned with the 10-seed run saved in `subsumption_accuracy.tex` (`main.tex` had hardcoded earlier values).
- Paper Experiment 8 scale corrected from `20,000-concept` to `2,000-concept` WordNet audit; matches `wordnet_2k.csv` (1,999,000 chains, 108,694 discrepancies).
- Paper Table 9: added footnote on Contrastive variability — the 100% TP at `k=6` is a single-run figure; deterministic sorted-vocabulary order yields 96.2%, range 92–100% across orderings.
- Paper abstract speedup figure: `28.4×` → `30.9×` (matches Experiment 1, the primary timing benchmark).
- `pyproject.toml` description: speedup figure `28.4x` → `30.9x`.
- `.zenodo.json` description: same correction; version field bumped to `0.4.0` (commit `2b98585`).
- `pyproject.toml`: `[tool.setuptools.packages.find]` now excludes `neurosym._archived*` so deprecated modules (`buss.py`, `uhrt.py`) no longer ship in the wheel/sdist.
- README + ROADMAP refreshed with actual implementation status: documented 4 new REST endpoints, corrected test-coverage status (`PrimeIndexDB`, `ScalableGraphBuilder`, `ReportGenerator` all have full coverage).
- Internal `__version__` strings synchronized to `0.4.0` across `src/neurosym/__init__.py`, `src/neurosym/reports.py` (fallback), `api/server.py`, `api/models.py`.
- CI workflow: bumped `actions/checkout@v4` → `@v5`, `setup-python@v5` → `@v6`, `upload-artifact@v4` → `@v5`, `download-artifact@v4` → `@v5` (Node.js 24-compatible).
- BibTeX: `kotlerman2010directional` re-typed `@inproceedings` → `@article` (it appears in *Natural Language Engineering* journal); resolves the `volume`/`number` warning.

### Fixed
- Paper Table 5 (Analogy Resolution): `k=3` and `k=6` corrected from `1/50 (2.0%)` to `0/50 (0.0%)` (matched `experiment5_analogy.csv`).
- Paper Experiment 7 prose: clarified that the `28.4×` measures `analogy_prediction` (a heavier operation) at `N=20,000`, distinct from Experiment 1's `30.9×` single-pair GCD baseline.
- Makefile `paper` target: now compiles from `paper/src/` so relative paths (`../figures/pipeline.png`, `\bibliography{references}`) resolve correctly; previous behavior produced PDFs without the figure or citations.
- Makefile `clean` target: no longer deletes committed PDF artifacts.
- `src/neurosym/reports.py`: `datetime.utcnow()` (deprecated in Python 3.12, scheduled for removal) replaced with `datetime.now(timezone.utc)`.
- `tests/test_storage.py`: removed unused `os` import that was failing every CI run via `ruff F401`.

## [0.3.0] — 2026-03-25

### Added
- Bundled Streamlit dashboard inside the package; `triadic-dashboard` CLI entry point (commit `c100836`).
- 4-mode projection benchmark (random, PCA, consensus, contrastive) — Experiment 9.
- PCA-directed projection mode: deterministic, corpus-adapted alternative to random hyperplanes.
- Multi-seed consensus encoding mode (filters projection noise via voting threshold).
- Contrastive hyperplane learning mode (gradient-free optimization on hypernym pairs).
- Topological audit table with prime-to-word interpretation (Experiment 8 expansion).
- Bitwise–prime equivalence section in paper (1,000 exhaustive equivalence tests).

### Changed
- Paper expanded from 8 to 9 experiments (industrial-scale benchmark, topological bias auditing).
- License unified to BUSL-1.1 (auto-converts to AGPL-3.0 on 2030-03-21); consortium participation model documented in `COMMERCIAL.md`.
- Author name standardized to "J. Arturo Ornelas Brand" everywhere.
- CORS wildcard replaced with env-based origins (`CORS_ORIGINS`) for security.

### Fixed
- Figure and bibliography paths in paper resolve correctly from `paper/src/`.
- `tqdm` progress bars suppressed during model loading (Streamlit/Jupyter compatibility).
- `__version__` synced to `0.2.0` to match `pyproject.toml` (drifted again post-bump — fixed in 0.4.0).

## [0.2.0] — 2026-03-21

Initial PyPI release as `triadic-engine`.

### Added
- Public release with reproducible Jupyter notebook (`notebooks/Reproducibility_Demo.ipynb`).
- Academic LaTeX paper merged into the codebase under `paper/`.
- Multi-backend embedding encoder (HuggingFace/OpenAI/Cohere) with LSH→Prime projection.
- `DiscreteValidator` with `subsumes`, `compose`, `explain_gap`, `analogy_prediction`.
- `ScalableGraphBuilder` with inverted prime index (avoids O(N²)).
- `PrimeIndexDB` SQLite persistence layer.
- HTML/JSON/CSV `ReportGenerator` and `/report` endpoint.
- DataFrame `DatabaseIngestor` + tabular `AnomalyDetector`.
- CI workflow: lint (`ruff`), tests (`pytest`), build (wheel + sdist), Trusted Publishing to PyPI.

[Unreleased]: https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/releases/tag/v0.4.0
[0.3.0]: https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/releases/tag/v0.3.0
[0.2.0]: https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/releases/tag/v0.2.0
