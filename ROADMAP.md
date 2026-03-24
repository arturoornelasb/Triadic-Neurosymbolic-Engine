# Roadmap

What remains for the Triadic Neurosymbolic Engine to be production-ready and commercially viable.

---

## Tecnico

### Test coverage

- [ ] `PrimeIndexDB` — 8 public methods, 0 tests (save/load/delete/export/list)
- [ ] `ScalableGraphBuilder` — 5 public methods, 0 tests (build_index, find_edges, find_neighbors, get_stats)
- [ ] `ReportGenerator` — 8 public methods, 0 tests (to_html, to_json, to_csv, save)
- [ ] `OpenAIEncoder` / `CohereEncoder` — 0 tests (require API keys; add with `@pytest.mark.skipif`)
- [ ] `DiscreteMapper.get_factor()` — exported but never tested
- [ ] `create_encoder()` factory — exported but never tested
- [ ] Edge cases: empty inputs, zero values, overflow at high k

### REST API completeness

The API server (`api/server.py`) only exposes 5 of the engine's core operations:

- [x] `/health`, `/encode`, `/audit`, `/search`, `/report`
- [ ] `/subsumes` — `DiscreteValidator.subsumes()`
- [ ] `/compose` — `DiscreteValidator.compose()`
- [ ] `/gap` — `DiscreteValidator.explain_gap()`
- [ ] `/analogy` — `DiscreteValidator.analogy_prediction()`
- [ ] `/save-index`, `/load-index` — `PrimeIndexDB` persistence

### API hardening

- [ ] Authentication (API keys or JWT)
- [ ] Rate limiting per client
- [ ] Request validation beyond Pydantic (payload size limits, concept deduplication)
- [ ] Structured logging (replace print statements)
- [ ] Health check should verify encoder is loaded, not just return 200
- [ ] OpenAPI schema versioning (`/v1/encode`)

### Containerization

- [ ] Dockerfile (multi-stage: build + runtime)
- [ ] docker-compose.yml (API + optional dashboard)
- [ ] Pre-download `all-MiniLM-L6-v2` model in image build (avoid cold-start download)

### CI/CD

- [ ] Add `plotly` import test to CI (new dashboard dependency)
- [ ] Add API integration tests (`httpx` is already in dev deps)
- [ ] Publish wheel alongside sdist (only `.tar.gz` in `dist/` currently)
- [ ] Add `ruff format --check` to CI (currently only `ruff check`)

### Code quality

- [ ] `playground/engine.py` — `TriadicEngine` class defined twice (lines 53 and 132); second overrides first
- [ ] `playground/performance_benchmark.py` — calls `engine.get_embedding()` which does not exist
- [ ] `playground/` is in `.gitignore` but still tracked; decide: clean up or fully exclude
- [ ] Add `py.typed` marker to `src/neurosym/` for downstream type checking

### Observability

- [ ] Structured JSON logging across all modules (currently uses `logging` with no handler config)
- [ ] Timing metrics for encode/map/search pipeline stages
- [ ] Error tracking integration (Sentry or equivalent)

---

## Comercial

### Cloud API (triadic-cloud)

The engine library is BUSL-1.1 (source-available, not open-source), and a hosted API is the monetization path:

- [ ] Deploy API behind gateway (e.g., AWS API Gateway, Cloudflare Workers)
- [ ] Tiered API key system (free/pro/enterprise)
- [ ] Usage metering and billing integration (Stripe)
- [ ] Dashboard for API key management and usage stats
- [ ] SLA documentation (uptime, latency guarantees)

### Client SDK (neurosym-client)

- [ ] Publish Python client that wraps the Cloud API
- [ ] `pip install neurosym-client` with `TriadicClient(api_key="...")`
- [ ] Thin wrapper: encode, subsumes, compose, gap, search, audit

### Documentation

- [ ] Landing page / docs site (GitHub Pages or similar)
- [ ] API reference with request/response examples for every endpoint
- [ ] Integration guides: Python, SQL (PostgreSQL), Prolog (appendix already in paper)
- [ ] Jupyter notebooks for each use case (RAG, auditing, deduplication, anomaly detection)
- [ ] CHANGELOG.md — no version history exists

### Licensing clarity

- [ ] FAQ page explaining BUSL-1.1 in plain language for potential customers
- [ ] Clear boundary: what counts as "competing neurosymbolic validation API"
- [ ] Commercial license template ready for enterprise deals
- [ ] Dual license option: BUSL-1.1 (open) + commercial (paid)

### Validation

- [ ] Contrastive projection evaluates on training pairs (paper Section 4.10 notes this); need held-out evaluation
- [ ] Benchmark on established hypernym datasets (HyperLex, BLESS) for third-party credibility
- [ ] Case study with real customer data (RAG pipeline or audit workflow)

---

## Bloqueos

### Criticos (bloquean produccion)

1. **No auth on API** — anyone can call `/encode` and `/audit` without credentials. Must add before any public deployment.
2. **No rate limiting** — a single client can exhaust server resources. Blocks cloud offering.
3. **Cold-start model download** — `all-MiniLM-L6-v2` (~80MB) downloads on first use. In containers, this must be baked into the image.

### Importantes (bloquean comercializacion)

4. **No client SDK** — customers need `neurosym-client` to integrate without self-hosting.
5. **No billing infrastructure** — can't charge for API usage without metering + Stripe.
6. **Test coverage gaps** — storage, graph, and reports modules have zero tests. Risky to guarantee correctness to paying customers.
7. **Missing REST endpoints** — core operations (subsumes, compose, gap) are only available via Python import, not HTTP.

### Limitaciones conocidas (del paper, Section 5)

8. **Hash coincidence != semantic containment** — subsumption reflects LSH bucket overlap, not genuine semantic containment. Changing the seed can reverse relationships.
9. **Lossy projection** — R^384 -> Z is inherently lossy. "Happy" and "elated" may get identical encodings.
10. **Useful k range is narrow** — k=6-12 is the practical regime; no principled selection method exists.
11. **Analogy accuracy is low** — 2-10% (paper Experiment 5). The method is for *verification*, not *discovery*.
