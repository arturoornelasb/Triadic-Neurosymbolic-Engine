import streamlit as st
import pandas as pd
import time
import math

from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.triadic import DiscreteValidator
from neurosym.ingest import DatabaseIngestor
from neurosym.storage import PrimeIndexDB
from neurosym.anomaly import AnomalyDetector, RelationalRule
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Triadic Neurosymbolic Engine",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] { color: #e94560; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #a8a8b3; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    [data-testid="stSidebar"] hr { border-color: #30363d; }
    .severity-critical { color: #ef4444; font-weight: 700; }
    .severity-warning  { color: #f59e0b; font-weight: 600; }
    .severity-info     { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# State Management
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_encoder():
    return ContinuousEncoder("all-MiniLM-L6-v2")

encoder    = load_encoder()
validator  = DiscreteValidator()
db_index   = PrimeIndexDB()

DEFAULT_WORDS = [
    "Doctor", "Hospital", "Medicine", "Nurse", "Surgery", "Patient",
    "Computer", "Algorithm", "Software", "Hardware", "Internet", "Programmer",
    "Dog", "Cat", "Animal", "Pet", "Veterinarian",
]

for _k, _v in {
    "mapper":           DiscreteMapper(n_bits=8, seed=42),
    "prime_map":        {},
    "word_list":        DEFAULT_WORDS[:],
    "embeddings_cache": {},
    "anomaly_rules":    [],
    "graph_nodes":      [],
    "graph_edges":      [],
    "projection_mode":  "random",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "db_ingestor" not in st.session_state:
    st.session_state.db_ingestor = DatabaseIngestor(encoder, st.session_state.mapper)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def get_cached_embeddings(word_list):
    import numpy as np
    missing = [w for w in word_list if w not in st.session_state.embeddings_cache]
    if missing:
        for w, emb in zip(missing, encoder.encode(missing)):
            st.session_state.embeddings_cache[w] = emb
    return np.array([st.session_state.embeddings_cache[w] for w in word_list])

def recompute_mapping():
    if not st.session_state.word_list:
        return
    mode = st.session_state.get("projection_mode", "random")
    st.session_state.mapper = DiscreteMapper(
        n_bits=st.session_state.lsh_bits,
        seed=st.session_state.lsh_seed,
        projection=mode,
    )
    embeddings = get_cached_embeddings(st.session_state.word_list)
    st.session_state.prime_map = st.session_state.mapper.fit_transform(
        st.session_state.word_list, embeddings
    )
    st.session_state.db_ingestor.mapper = st.session_state.mapper

CLUSTER_PALETTE = [
    "#e94560","#0f3460","#00bcd4","#ff6f61","#6b5b95",
    "#88b04b","#f7cac9","#92a8d1","#955251","#b565a7",
    "#009b77","#dd4124","#d65076","#45b8ac","#efc050",
    "#5b5ea6","#9b2335","#dfcfbe","#55b4b0","#e15d44",
]

def node_color(prime_value):
    factors = validator._prime_factors(prime_value)
    return CLUSTER_PALETTE[factors[0] % len(CLUSTER_PALETTE)] if factors else CLUSTER_PALETTE[0]

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ Triadic Engine")
    st.caption("Deterministic Neurosymbolic Framework")
    st.markdown("---")
    st.markdown("#### ⚙️ LSH Parameters")
    st.slider("Resolution Bits (k)", 2, 32, 8, 1,
              key="lsh_bits", on_change=recompute_mapping,
              help="Higher bits = stricter semantic definitions.")
    st.number_input("Random Seed", value=42, key="lsh_seed", on_change=recompute_mapping)
    st.selectbox(
        "Projection Mode",
        options=["random", "pca", "consensus", "contrastive"],
        key="projection_mode",
        on_change=recompute_mapping,
        help=(
            "random — fastest, seed-based.\n"
            "pca — corpus-adapted, deterministic.\n"
            "consensus — noise-robust multi-seed voting.\n"
            "contrastive — supervised (requires hypernym pairs)."
        ),
    )
    if st.session_state.get("projection_mode") == "contrastive":
        st.caption("⚠️ Contrastive mode uses random planes when no hypernym pairs are provided.")
    st.markdown("---")
    st.markdown("#### 📊 Engine Status")
    st.metric("Active Concepts",    f"{len(st.session_state.word_list)}")
    unique_primes = len(set(st.session_state.prime_map.values())) if st.session_state.prime_map else 0
    st.metric("Unique Prime Clusters", f"{unique_primes}")
    saved_count = len(db_index.list_indexes())
    st.metric("Saved Indices", f"{saved_count}")
    st.markdown("---")
    st.markdown("#### 🔗 Links")
    st.markdown("[📄 GitHub Repository](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine)")
    st.markdown("[📝 Academic Paper](https://github.com/arturoornelasb/Triadic-Neurosymbolic-Engine/tree/main/paper)")
    st.caption("© 2026 J. Arturo Ornelas Brand")

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
if not st.session_state.prime_map:
    recompute_mapping()

st.title("⚛️ Triadic Neurosymbolic Engine")
st.subheader("Deterministic LLM Interpretability & Verification")
st.markdown("Convert opaque $\\mathbb{R}^n$ embeddings into transparent Prime Factor integers $\\mathbb{Z}$ for $O(1)$ logical verification.")

if st.session_state.prime_map:
    n_c = len(st.session_state.word_list)
    n_cl = len(set(st.session_state.prime_map.values()))
    avg_f = sum(len(validator._prime_factors(p)) for p in st.session_state.prime_map.values()) / max(1, n_c)
    m1, m2, m3 = st.columns(3)
    m1.metric("📚 Active Concepts",     f"{n_c}")
    m2.metric("🧬 Prime Clusters",      f"{n_cl}")
    m3.metric("🔗 Avg. Factors/Concept",f"{avg_f:.1f}")

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧬 1. Ingestion & Encoding",
    "🌌 2. Semantic Graph",
    "🧠 3. Logic & Search",
    "🤖 4. AI Auditor",
    "🔍 5. Anomaly Detection",
    "📊 6. Benchmarks",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — INGESTION & INDEX MANAGEMENT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Step 1: Map Concepts to Prime Integers")
    st.markdown("Encode natural language concepts into composite prime integers for deterministic verification.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Option A: Manual Entry")
        new_words = st.text_input("Add words (comma separated):", placeholder="e.g. Finance, Bank, Money")
        if st.button("Encode New Words", type="primary"):
            if new_words:
                cleaned = [w.strip() for w in new_words.split(",") if w.strip()]
                added = [w for w in cleaned if w not in st.session_state.word_list]
                st.session_state.word_list.extend(added)
                recompute_mapping()
                st.success(f"Added {len(added)} new concepts into Prime Memory.")

        if st.button("🗑️ Clear Dictionary", type="secondary"):
            st.session_state.word_list = []
            st.session_state.prime_map = {}
            st.rerun()

    with col2:
        st.markdown("#### Option B: Upload CSV")
        uploaded_file = st.file_uploader("Upload CSV (catalog, wordnet, etc.)", type="csv")
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df_upload)} rows. Select the text column to index.")
            text_cols = [c for c in df_upload.columns if df_upload[c].dtype == "object"]
            if text_cols:
                target_col = st.selectbox("Text Column:", text_cols)
                if st.button("🚀 Mass Ingest & Encode", type="primary"):
                    with st.spinner(f"Vectorizing {len(df_upload)} rows..."):
                        t0 = time.time()
                        words_to_add = df_upload[target_col].dropna().astype(str).tolist()
                        new_w = [w for w in words_to_add if w not in st.session_state.word_list]
                        st.session_state.word_list.extend(new_w)
                        recompute_mapping()
                        st.success(f"Ingested {len(new_w)} new concepts in {time.time()-t0:.2f}s.")
                        st.rerun()

    # ── Saved Indices ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Named Index Management")
    st.caption("Save your current encoding as a named snapshot. Load it later without re-encoding.")

    idx_col1, idx_col2 = st.columns(2)

    with idx_col1:
        st.markdown("**Save Current Index**")
        index_name = st.text_input("Index name:", placeholder="e.g. Medical Vocabulary v1", key="save_index_name")
        if st.button("💾 Save Index"):
            if not index_name.strip():
                st.error("Please enter a name for the index.")
            elif not st.session_state.prime_map:
                st.error("No concepts encoded yet. Add words first.")
            else:
                db_index.save_index(
                    st.session_state.prime_map,
                    model=index_name.strip(),
                    lsh_bits=st.session_state.lsh_bits,
                    seed=st.session_state.lsh_seed,
                )
                st.success(f"Saved '{index_name}' — {len(st.session_state.prime_map)} concepts.")
                st.rerun()

    with idx_col2:
        st.markdown("**Load / Delete Saved Index**")
        saved = db_index.list_indexes()
        if saved:
            idx_options = {
                f"{r['model']}  ({r['concept_count']} concepts, bits={r['lsh_bits']})": r
                for r in saved
            }
            selected_label = st.selectbox("Select index:", list(idx_options.keys()))
            selected_row = idx_options[selected_label]

            load_col, del_col = st.columns(2)
            with load_col:
                if st.button("📂 Load Index", use_container_width=True):
                    loaded_map = db_index.load_index(
                        model=selected_row["model"],
                        lsh_bits=selected_row["lsh_bits"],
                        seed=selected_row["seed"],
                    )
                    if loaded_map:
                        st.session_state.prime_map = loaded_map
                        st.session_state.word_list = list(loaded_map.keys())
                        st.success(f"Loaded '{selected_row['model']}' — {len(loaded_map)} concepts.")
                        st.rerun()
                    else:
                        st.error("Index not found in database.")
            with del_col:
                if st.button("🗑️ Delete Index", use_container_width=True, type="secondary"):
                    deleted = db_index.delete_index(
                        model=selected_row["model"],
                        lsh_bits=selected_row["lsh_bits"],
                        seed=selected_row["seed"],
                    )
                    st.success(f"Deleted '{selected_row['model']}' ({deleted} concepts removed).")
                    st.rerun()
        else:
            st.info("No saved indices yet. Save one above.")

    # ── Dictionary Table ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Current Mathematical Dictionary")
    if st.session_state.prime_map:
        dict_data = [
            {
                "Concept": w,
                "Integer ID": p,
                "Prime Factors": str(validator._prime_factors(p)),
                "Factor Count": len(validator._prime_factors(p)),
            }
            for w, p in st.session_state.prime_map.items()
        ]
        df_dict = pd.DataFrame(dict_data).sort_values("Factor Count", ascending=False)
        st.dataframe(df_dict, use_container_width=True, hide_index=True)

        # ── Export CSV ───────────────────────────────────────
        import io as _io
        import csv as _csv
        _buf = _io.StringIO()
        _writer = _csv.writer(_buf)
        _writer.writerow(["concept", "prime_factor", "prime_factors", "factor_count",
                          "lsh_bits", "seed", "projection_mode"])
        for row in dict_data:
            _writer.writerow([
                row["Concept"],
                row["Integer ID"],
                row["Prime Factors"],
                row["Factor Count"],
                st.session_state.lsh_bits,
                st.session_state.lsh_seed,
                st.session_state.get("projection_mode", "random"),
            ])
        st.download_button(
            label="📥 Export Index CSV",
            data=_buf.getvalue(),
            file_name=f"triadic_index_{st.session_state.get('projection_mode','random')}_k{st.session_state.lsh_bits}.csv",
            mime="text/csv",
            help="Download the current in-memory index as a CSV file.",
        )


# ══════════════════════════════════════════════════════════════
# TAB 2 — SEMANTIC GRAPH (improved)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Step 2: Semantic Holographic Graph")
    st.markdown("Concepts linked by **GCD > 1** share hidden semantic traits. Node color = dominant prime family.")

    if len(st.session_state.word_list) < 2:
        st.warning("Add words in Step 1 first.")
    else:
        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            min_weight = st.slider("Min. Shared Prime Factors", 1, 8, 2,
                                   help="Increase to hide weak connections.")
        with gcol2:
            max_nodes = st.slider("Max Nodes to Render", 50, 300, 150, 25,
                                  help="Limit for browser performance.")
        with gcol3:
            layout_physics = st.toggle("Physics Simulation", value=True)

        if st.button("🕸️ Render Semantic Graph", use_container_width=True, type="primary"):
            render_list = st.session_state.word_list[:max_nodes]
            if len(st.session_state.word_list) > max_nodes:
                st.warning(f"Showing first {max_nodes} of {len(st.session_state.word_list)} concepts.")

            progress = st.progress(0, text="Building semantic connections...")
            nodes_out, edges_out = [], []
            added_nodes: set = set()
            edge_records = []

            total_pairs = len(render_list) * (len(render_list) - 1) // 2
            pair_count = 0

            for i in range(len(render_list)):
                for j in range(i + 1, len(render_list)):
                    pair_count += 1
                    if pair_count % 500 == 0:
                        progress.progress(min(pair_count / max(1, total_pairs), 1.0),
                                          text=f"Pair {pair_count}/{total_pairs}...")
                    w_a = render_list[i]
                    w_b = render_list[j]
                    p_a = st.session_state.prime_map[w_a]
                    p_b = st.session_state.prime_map[w_b]
                    gap = validator.explain_gap(p_a, p_b)
                    if gap["shared"] > 1:
                        shared_factors = validator._prime_factors(gap["shared"])
                        weight = len(shared_factors)
                        if weight >= min_weight:
                            if w_a not in added_nodes:
                                nodes_out.append(Node(id=w_a, label=w_a,
                                                      size=15 + weight,
                                                      color=node_color(p_a)))
                                added_nodes.add(w_a)
                            if w_b not in added_nodes:
                                nodes_out.append(Node(id=w_b, label=w_b,
                                                      size=15 + weight,
                                                      color=node_color(p_b)))
                                added_nodes.add(w_b)
                            edges_out.append(Edge(source=w_a, target=w_b,
                                                  title=f"Shared: {shared_factors}",
                                                  width=weight))
                            edge_records.append({"From": w_a, "To": w_b,
                                                 "Shared Factors": str(shared_factors),
                                                 "Weight": weight})

            progress.progress(1.0, text="Complete!")
            st.session_state.graph_nodes = nodes_out
            st.session_state.graph_edges = edges_out
            st.session_state.graph_edge_records = edge_records

        if st.session_state.graph_nodes:
            nodes_out = st.session_state.graph_nodes
            edges_out = st.session_state.graph_edges
            edge_records = st.session_state.get("graph_edge_records", [])

            st.success(f"Graph: {len(nodes_out)} nodes · {len(edges_out)} edges")

            cfg = Config(
                width=1000, height=680, directed=False,
                physics=layout_physics, hierarchical=False,
                nodeHighlightBehavior=True, highlightColor="#F7A7A6",
                collapsible=True,
                link={"distance": 150},
                physics_layout={"barnesHut": {
                    "gravitationalConstant": -8000,
                    "springConstant": 0.04,
                    "springLength": 250,
                }},
            )
            agraph(nodes=nodes_out, edges=edges_out, config=cfg)

            # ── Graph Statistics ─────────────────────────────
            st.markdown("---")
            stat1, stat2, stat3 = st.columns(3)
            stat1.metric("Nodes rendered", len(nodes_out))
            stat2.metric("Semantic edges",  len(edges_out))
            stat3.metric("Avg. degree", f"{2*len(edges_out)/max(1,len(nodes_out)):.1f}")

            # ── Cluster Legend ───────────────────────────────
            cluster_map: dict = {}
            for n in nodes_out:
                p = st.session_state.prime_map.get(n.id, 1)
                factors = validator._prime_factors(p)
                dominant = factors[0] if factors else 0
                cluster_map.setdefault(dominant, []).append(n.id)

            if PLOTLY_AVAILABLE:
                cluster_df = pd.DataFrame([
                    {"Prime Family": f"p={k}", "Count": len(v), "Concepts": ", ".join(v[:5]) + ("…" if len(v) > 5 else "")}
                    for k, v in sorted(cluster_map.items())
                ])
                fig_bar = px.bar(cluster_df, x="Prime Family", y="Count",
                                 title="Nodes per Semantic Cluster",
                                 color="Count", color_continuous_scale="viridis",
                                 hover_data=["Concepts"])
                fig_bar.update_layout(height=280, margin=dict(t=40, b=20),
                                      plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                                      font_color="#e0e0e0")
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Export Edge List ─────────────────────────────
            if edge_records:
                csv_edges = pd.DataFrame(edge_records).to_csv(index=False)
                st.download_button("⬇️ Export Edge List CSV", csv_edges,
                                   "triadic_edges.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# TAB 3 — LOGIC & SEARCH
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Step 3: Arithmetic Reasoning")
    st.markdown("Exact algebraic operations: subsumption via `%`, composition via LCM, gap analysis via GCD.")

    if len(st.session_state.word_list) < 2:
        st.info("Add words in Step 1 to enable logic.")
    else:
        col_log1, col_log2 = st.columns(2)

        with col_log1:
            st.markdown("#### Gap Analysis (Semantic Diff)")
            st.markdown("Uses `GCD` to decompose exactly what differs between two concepts.")
            gap_a = st.selectbox("Concept A", st.session_state.word_list, index=0, key="gap_a")
            gap_b = st.selectbox("Concept B", st.session_state.word_list, index=1, key="gap_b")

            if st.button("Calculate Vector Difference", type="primary"):
                p_a = st.session_state.prime_map[gap_a]
                p_b = st.session_state.prime_map[gap_b]
                gap = validator.explain_gap(p_a, p_b)
                st.table(pd.DataFrame({
                    "Analysis": [
                        "Common Semantic Core (GCD)",
                        f"Features only in '{gap_a}'",
                        f"Features only in '{gap_b}'",
                    ],
                    "Operation": ["GCD(A, B)", "A / GCD", "B / GCD"],
                    "Prime Factors": [
                        str(validator._prime_factors(gap["shared"])),
                        str(validator._prime_factors(gap["only_in_a"])),
                        str(validator._prime_factors(gap["only_in_b"])),
                    ],
                }))

                subsumes_ab = p_a % p_b == 0
                subsumes_ba = p_b % p_a == 0
                if subsumes_ab:
                    st.success(f"✅ '{gap_a}' subsumes '{gap_b}' — A contains ALL features of B.")
                elif subsumes_ba:
                    st.success(f"✅ '{gap_b}' subsumes '{gap_a}' — B contains ALL features of A.")
                else:
                    st.info("Neither concept fully subsumes the other — they overlap but diverge.")

        with col_log2:
            st.markdown("#### Triadic Search (GCD Similarity)")
            st.markdown("Search by integer divisibility — deterministic, no cosine approximation.")
            search_query = st.text_input("Search Query:", placeholder="e.g. Medical professional")
            top_k = st.slider("Top-K results", 3, 20, 5)

            if st.button("Search Database", type="primary"):
                if search_query:
                    q_emb = encoder.encode([search_query])
                    q_map = st.session_state.mapper.transform([search_query], q_emb)
                    q_prime = q_map[search_query]
                    st.write(f"Query prime: `{q_prime}` — factors: `{validator._prime_factors(q_prime)}`")
                    results = []
                    for w in st.session_state.word_list:
                        w_prime = st.session_state.prime_map[w]
                        shared = math.gcd(q_prime, w_prime)
                        if shared > 1:
                            score = len(validator._prime_factors(shared))
                            results.append({"Concept": w, "Shared Factors": score,
                                            "Shared Primes": str(validator._prime_factors(shared))})
                    if results:
                        res_df = pd.DataFrame(results).sort_values("Shared Factors", ascending=False).head(top_k)
                        st.dataframe(res_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No shared factors found. Try different words or lower LSH bits.")

        # ── Analogy Solver ────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Analogy Solver — A : B :: C : ?")
        st.markdown("Finds D such that the semantic transformation A→B applies to C→D.")
        an1, an2, an3 = st.columns(3)
        with an1:
            analogy_a = st.selectbox("A", st.session_state.word_list, key="an_a")
        with an2:
            analogy_b = st.selectbox("B", st.session_state.word_list, index=1, key="an_b")
        with an3:
            analogy_c = st.selectbox("C", st.session_state.word_list, index=2, key="an_c")

        if st.button("Solve Analogy", type="primary"):
            p_a = st.session_state.prime_map[analogy_a]
            p_b = st.session_state.prime_map[analogy_b]
            p_c = st.session_state.prime_map[analogy_c]
            try:
                result = validator.analogy_prediction(p_a, p_b, p_c)
                p_d = int(result.output_value)
                candidates = []
                for w, p in st.session_state.prime_map.items():
                    if w not in (analogy_a, analogy_b, analogy_c):
                        closeness = math.gcd(p, p_d)
                        if closeness > 1:
                            candidates.append((len(validator._prime_factors(closeness)), w, p))
                candidates.sort(reverse=True)
                if candidates:
                    st.info(f"**{analogy_a} : {analogy_b} :: {analogy_c} : ?**")
                    top_candidates = candidates[:5]
                    st.dataframe(pd.DataFrame([
                        {"Candidate": w, "Score": s, "Prime": p}
                        for s, w, p in top_candidates
                    ]), use_container_width=True, hide_index=True)
                else:
                    st.warning("No strong analogy candidate found in the current dictionary.")
            except Exception as e:
                st.error(f"Analogy computation failed: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 4 — AI AUDITOR (improved with heatmap + severity)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Step 4: AI Auditor — Model vs Model")
    st.markdown("Compare the **topological structure** of two embedding models on the same concepts. "
                "A discrepancy means the models disagree on how two concepts relate.")

    if len(st.session_state.word_list) < 2:
        st.warning("Add words in Step 1 first.")
    else:
        st.markdown(f"**Auditing {len(st.session_state.word_list)} concepts**")
        aud_col1, aud_col2 = st.columns(2)
        with aud_col1:
            model_a_name = st.selectbox("Model A", [
                "all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "all-mpnet-base-v2",
                "all-MiniLM-L12-v2", "paraphrase-MiniLM-L6-v2",
            ], index=0, key="aud_model_a")
        with aud_col2:
            model_b_name = st.selectbox("Model B", [
                "paraphrase-MiniLM-L3-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2",
                "all-MiniLM-L12-v2", "paraphrase-MiniLM-L6-v2",
            ], index=0, key="aud_model_b")

        show_heatmap = st.toggle("Show Discrepancy Heatmap", value=PLOTLY_AVAILABLE,
                                  disabled=not PLOTLY_AVAILABLE)

        if st.button("🚀 Run Triadic Audit", use_container_width=True, type="primary"):
            if model_a_name == model_b_name:
                st.error("Select two different models.")
            else:
                with st.spinner(f"Loading {model_a_name} and {model_b_name}..."):
                    enc_A = ContinuousEncoder(model_a_name)
                    enc_B = ContinuousEncoder(model_b_name)
                    words = st.session_state.word_list

                    emb_A = enc_A.encode(words)
                    emb_B = enc_B.encode(words)

                    mapper_A = DiscreteMapper(n_bits=st.session_state.lsh_bits, seed=st.session_state.lsh_seed)
                    mapper_B = DiscreteMapper(n_bits=st.session_state.lsh_bits, seed=st.session_state.lsh_seed)
                    primes_A = mapper_A.fit_transform(words, emb_A)
                    primes_B = mapper_B.fit_transform(words, emb_B)

                    graph_A, graph_B = nx.Graph(), nx.Graph()
                    graph_A.add_nodes_from(words)
                    graph_B.add_nodes_from(words)

                    for i in range(len(words)):
                        for j in range(i + 1, len(words)):
                            if math.gcd(primes_A[words[i]], primes_A[words[j]]) > 1:
                                graph_A.add_edge(words[i], words[j])
                            if math.gcd(primes_B[words[i]], primes_B[words[j]]) > 1:
                                graph_B.add_edge(words[i], words[j])

                    paths_A = dict(nx.all_pairs_shortest_path_length(graph_A))
                    paths_B = dict(nx.all_pairs_shortest_path_length(graph_B))

                    results = []
                    total_pairs = discrepancies = 0

                    # Build distance matrix for heatmap
                    n = len(words)
                    diff_matrix = [[0.0] * n for _ in range(n)]

                    for i in range(n):
                        for j in range(i + 1, n):
                            total_pairs += 1
                            w1, w2 = words[i], words[j]
                            dist_A = paths_A.get(w1, {}).get(w2, float("inf"))
                            dist_B = paths_B.get(w1, {}).get(w2, float("inf"))

                            d_a_fin = dist_A if dist_A != float("inf") else 99
                            d_b_fin = dist_B if dist_B != float("inf") else 99
                            diff = abs(d_a_fin - d_b_fin)
                            diff_matrix[i][j] = diff
                            diff_matrix[j][i] = diff

                            if dist_A != dist_B:
                                discrepancies += 1
                                # Severity
                                if diff >= 3:
                                    sev = "🔴 CRITICAL"
                                elif diff == 2:
                                    sev = "🟠 HIGH"
                                else:
                                    sev = "🟡 MEDIUM"

                                path_ex = ""
                                if dist_A != float("inf") and dist_A <= 3:
                                    path_ex = " → ".join(nx.shortest_path(graph_A, w1, w2))
                                elif dist_B != float("inf") and dist_B <= 3:
                                    path_ex = " → ".join(nx.shortest_path(graph_B, w1, w2))

                                results.append({
                                    "Severity": sev,
                                    "Concept A": w1,
                                    "Concept B": w2,
                                    f"{model_a_name[:14]}": str(dist_A) if dist_A != float("inf") else "INF",
                                    f"{model_b_name[:14]}": str(dist_B) if dist_B != float("inf") else "INF",
                                    "Δ Hops": diff,
                                    "Chain Example": path_ex,
                                })

                # Results summary
                rate = discrepancies / max(1, total_pairs)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Pairs",       f"{total_pairs:,}")
                s2.metric("Discrepancies",      f"{discrepancies:,}")
                s3.metric("Agreement Rate",     f"{(1-rate)*100:.1f}%")
                s4.metric("Discrepancy Rate",   f"{rate*100:.1f}%")

                if discrepancies == 0:
                    st.success("✅ Full topological agreement between both models.")
                else:
                    st.error(f"🚨 {discrepancies} relational differences found.")

                    # Sort by severity
                    results_df = pd.DataFrame(results).sort_values("Δ Hops", ascending=False)

                    # Heatmap
                    if show_heatmap and PLOTLY_AVAILABLE and len(words) <= 80:
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=diff_matrix,
                            x=words, y=words,
                            colorscale="RdYlGn_r",
                            zmin=0, zmax=4,
                            colorbar=dict(title="Δ Hops"),
                        ))
                        fig_heat.update_layout(
                            title=f"Discrepancy Heatmap: {model_a_name[:20]} vs {model_b_name[:20]}",
                            height=500,
                            xaxis_tickangle=-45,
                            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                            font_color="#e0e0e0",
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                    elif show_heatmap and len(words) > 80:
                        st.info("Heatmap limited to ≤80 concepts for readability.")

                    # Table
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    # Export
                    csv_audit = results_df.to_csv(index=False)
                    st.download_button("⬇️ Export Audit CSV", csv_audit,
                                       "triadic_audit.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# TAB 5 — ANOMALY DETECTION (new)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Step 5: Multiplicative Anomaly Detection")
    st.markdown(
        "Verify algebraic relationships in tabular data. "
        "Define rules like **Total = Qty × Unit Price × Tax**, "
        "and the engine finds every row that violates them — with exact deviation factors."
    )

    uploaded_anom = st.file_uploader("Upload CSV to scan:", type="csv", key="anom_csv")

    if uploaded_anom is not None:
        df_anom = pd.read_csv(uploaded_anom)
        all_cols = df_anom.columns.tolist()
        numeric_cols = df_anom.select_dtypes(include="number").columns.tolist()

        anp1, anp2 = st.columns([2, 1])
        with anp1:
            st.markdown(f"**{len(df_anom)} rows · {len(all_cols)} columns**")
            st.dataframe(df_anom.head(5), use_container_width=True, hide_index=True)
        with anp2:
            st.markdown("**Column Types**")
            st.dataframe(pd.DataFrame({
                "Column": all_cols,
                "Type": [str(df_anom[c].dtype) for c in all_cols],
            }), use_container_width=True, hide_index=True)

        # ── Rule Builder ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📐 Rule Builder")
        st.caption("Define multiplicative relationships between columns. Click + to add a rule.")

        header_col, add_col = st.columns([5, 1])
        with header_col:
            st.markdown("**Configured Rules**")
        with add_col:
            if st.button("➕ Add Rule", type="primary"):
                st.session_state.anomaly_rules.append({
                    "name": f"Rule {len(st.session_state.anomaly_rules) + 1}",
                    "factor_columns": [],
                    "result_column": numeric_cols[0] if numeric_cols else (all_cols[0] if all_cols else ""),
                    "tolerance": 0.01,
                })
                st.rerun()

        if not st.session_state.anomaly_rules:
            st.info("No rules defined. Click ➕ Add Rule to begin.")
        else:
            rules_to_delete = []
            for i, rule in enumerate(st.session_state.anomaly_rules):
                with st.expander(f"**Rule {i+1}:** {rule['name']}", expanded=True):
                    rc1, rc2, rc3 = st.columns([3, 3, 1])

                    with rc1:
                        rule["name"] = st.text_input(
                            "Rule Name",
                            value=rule["name"],
                            key=f"rname_{i}",
                            placeholder="e.g. Invoice Total"
                        )
                        rule["factor_columns"] = st.multiselect(
                            "Factor Columns (their product = result)",
                            options=all_cols,
                            default=[c for c in rule["factor_columns"] if c in all_cols],
                            key=f"rfactors_{i}",
                        )

                    with rc2:
                        result_default = rule["result_column"] if rule["result_column"] in all_cols else all_cols[0]
                        rule["result_column"] = st.selectbox(
                            "Result Column (expected product)",
                            options=all_cols,
                            index=all_cols.index(result_default),
                            key=f"rresult_{i}",
                        )
                        rule["tolerance"] = st.slider(
                            "Tolerance",
                            0.0, 0.2,
                            float(rule["tolerance"]),
                            0.005,
                            format="%.3f",
                            key=f"rtol_{i}",
                            help="0.01 = 1% tolerance for floating-point rounding"
                        )
                        if rule["factor_columns"] and rule["result_column"]:
                            st.caption(
                                f"Checks: `{rule['result_column']} ≈ "
                                f"{' × '.join(rule['factor_columns'])}`"
                                f"  (±{rule['tolerance']*100:.1f}%)"
                            )

                    with rc3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🗑️", key=f"del_rule_{i}", help="Remove this rule"):
                            rules_to_delete.append(i)

            for idx in reversed(rules_to_delete):
                st.session_state.anomaly_rules.pop(idx)
            if rules_to_delete:
                st.rerun()

        # ── Run Scan ─────────────────────────────────────────
        st.markdown("---")
        valid_rules = [
            r for r in st.session_state.anomaly_rules
            if r["factor_columns"] and r["result_column"]
        ]

        if st.button("🔍 Scan for Anomalies", type="primary",
                     use_container_width=True,
                     disabled=len(valid_rules) == 0):
            detector = AnomalyDetector()
            for rule in valid_rules:
                detector.add_rule(RelationalRule(
                    name=rule["name"],
                    factor_columns=rule["factor_columns"],
                    result_column=rule["result_column"],
                    tolerance=rule["tolerance"],
                ))

            with st.spinner(f"Scanning {len(df_anom)} rows against {len(valid_rules)} rules..."):
                try:
                    anomalies = detector.scan(df_anom)
                except ValueError as e:
                    st.error(str(e))
                    anomalies = []

            # Summary
            n_crit = sum(1 for a in anomalies if a.severity == "CRITICAL")
            n_warn = sum(1 for a in anomalies if a.severity == "WARNING")
            n_info = sum(1 for a in anomalies if a.severity == "INFO")

            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Rows Scanned",  f"{len(df_anom):,}")
            sc2.metric("Anomalies Found", f"{len(anomalies):,}")
            sc3.metric("🔴 Critical",   f"{n_crit}")
            sc4.metric("🟠 Warning",    f"{n_warn}")
            sc5.metric("🔵 Info",       f"{n_info}")

            if not anomalies:
                st.success("✅ All rows pass the defined rules. Data is clean.")
            else:
                severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
                anomaly_rows = sorted([
                    {
                        "Severity": a.severity,
                        "Row": a.row_index,
                        "Rule": a.rule_name,
                        "Expected": round(a.expected, 4),
                        "Actual": round(a.actual, 4),
                        "Ratio": round(a.ratio, 4),
                        "Missing Factor": round(a.missing_factor, 4),
                        "Explanation": a.explanation,
                    }
                    for a in anomalies
                ], key=lambda x: severity_order.get(x["Severity"], 3))

                anom_df = pd.DataFrame(anomaly_rows)
                st.dataframe(anom_df, use_container_width=True, hide_index=True)

                # Severity distribution chart
                if PLOTLY_AVAILABLE:
                    sev_df = pd.DataFrame({
                        "Severity": ["CRITICAL", "WARNING", "INFO"],
                        "Count": [n_crit, n_warn, n_info],
                        "Color": ["#ef4444", "#f59e0b", "#3b82f6"],
                    })
                    fig_sev = px.bar(sev_df[sev_df.Count > 0], x="Severity", y="Count",
                                     color="Severity",
                                     color_discrete_map={
                                         "CRITICAL": "#ef4444",
                                         "WARNING":  "#f59e0b",
                                         "INFO":     "#3b82f6",
                                     },
                                     title="Anomalies by Severity")
                    fig_sev.update_layout(
                        height=260, showlegend=False,
                        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                        font_color="#e0e0e0", margin=dict(t=40, b=10),
                    )
                    st.plotly_chart(fig_sev, use_container_width=True)

                csv_anom = anom_df.to_csv(index=False)
                st.download_button("⬇️ Export Anomaly Report CSV", csv_anom,
                                   "anomaly_report.csv", "text/csv")
    else:
        st.info("Upload a CSV file to begin. The engine will verify any multiplicative relationship you define.")
        st.markdown("**Example CSV format:**")
        example_df = pd.DataFrame({
            "product":    ["Widget A", "Widget B", "Widget C"],
            "qty":        [10, 5, 8],
            "unit_price": [9.99, 24.99, 14.99],
            "tax_rate":   [1.10, 1.10, 1.08],
            "total":      [109.89, 137.45, 130.0],  # row 2 is wrong
        })
        st.dataframe(example_df, use_container_width=True, hide_index=True)
        st.caption("Rule example: **total = qty × unit_price × tax_rate** (row 3 will be flagged as anomaly)")


# ══════════════════════════════════════════════════════════════
# TAB 6 — BENCHMARKS
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Performance Benchmarks")
    st.markdown("Results from Phase 4 validation test — 20,000 concepts from WordNet.")

    st.table(pd.DataFrame({
        "Benchmark":        ["Memory Storage", "Validation Metric", "Compute Operation", "Time (20K rows)"],
        "Traditional RAG":  ["Float32 Matrices (Heavy)", "Cosine Similarity (≈)", "Dense Dot Product", "11.21s"],
        "Triadic Engine":   ["Integer DB (Ultra Light)", "Exact Subsumption (=)", "Integer % Modulo", "0.93s"],
    }))
    st.success("**12× faster** than traditional vector search — with mathematically guaranteed deterministic results.")

    st.markdown("---")
    st.markdown("### Projection Mode Comparison")
    st.table(pd.DataFrame({
        "Mode":        ["Random", "PCA", "Consensus", "Contrastive"],
        "Determinism": ["Seed-dependent", "Full (corpus-adapted)", "High (voting)", "Full (supervised)"],
        "Speed":       ["⚡ Fastest", "⚡ Fast", "🐢 Slower (×seeds)", "🐢 Slow (200 iter)"],
        "Accuracy":    ["Baseline", "Good", "Better (noise-robust)", "100% hypernym @ k=6"],
        "Best For":    ["Prototyping", "Production default", "Stable features", "Supervised tasks"],
    }))

    # ── Live Benchmark ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Live Benchmark — Your Hardware")
    st.caption("Measures real encoding + prime-mapping time on your machine.")

    _BENCH_WORDS = [
        "Doctor", "Hospital", "Medicine", "Nurse", "Surgery", "Patient",
        "Computer", "Algorithm", "Software", "Hardware", "Internet", "Programmer",
        "Dog", "Cat", "Animal", "Pet", "Veterinarian", "Biology",
        "Finance", "Bank", "Loan", "Credit", "Debt", "Investment", "Market",
        "Law", "Justice", "Court", "Judge", "Lawyer", "Contract", "Crime",
        "Atom", "Electron", "Proton", "Neutron", "Molecule", "Chemistry",
        "River", "Mountain", "Ocean", "Forest", "Desert", "Volcano", "Earthquake",
        "Music", "Guitar", "Piano", "Melody", "Rhythm", "Harmony", "Orchestra",
        "Art", "Painting", "Sculpture", "Canvas", "Gallery", "Museum",
        "Engine", "Turbine", "Gear", "Piston", "Motor", "Robot", "Circuit",
        "Philosophy", "Ethics", "Logic", "Reason", "Epistemology", "Ontology",
        "Language", "Grammar", "Syntax", "Semantics", "Morphology", "Phonetics",
        "History", "Culture", "Civilization", "Empire", "Revolution", "Democracy",
        "Star", "Planet", "Galaxy", "Comet", "Asteroid", "Telescope", "Cosmos",
        "Democracy", "Constitution", "President", "Parliament", "Senate", "Election",
        "Bread", "Water", "Salt", "Sugar", "Protein", "Carbohydrate", "Vitamin",
        "Happiness", "Sadness", "Anger", "Fear", "Surprise", "Love", "Grief",
        "Teacher", "Student", "Classroom", "Lecture", "Exam", "Degree", "Research",
        "Book", "Library", "Author", "Publisher", "Chapter", "Paragraph", "Sentence",
    ]

    _bench_n = st.selectbox(
        "Concept count to encode",
        options=[100, 200, 500, len(_BENCH_WORDS)],
        index=1,
        format_func=lambda x: f"{x} concepts" if x != len(_BENCH_WORDS) else f"{x} concepts (full set)",
        key="bench_n",
    )
    _bench_mode = st.selectbox(
        "Projection mode for benchmark",
        options=["random", "pca", "consensus"],
        key="bench_mode",
        help="Contrastive excluded (requires labelled pairs).",
    )

    if st.button("▶ Run Live Benchmark", type="primary"):
        _words = (_BENCH_WORDS * ((_bench_n // len(_BENCH_WORDS)) + 1))[:_bench_n]
        _words = list(dict.fromkeys(_words))          # deduplicate, preserve order
        _words = (_words * ((_bench_n // max(len(_words), 1)) + 1))[:_bench_n]

        with st.spinner(f"Encoding {_bench_n} concepts in '{_bench_mode}' mode…"):
            # ── Step 1: embedding ──────────────────────────────
            _t0 = time.perf_counter()
            _embs = encoder.encode(_words)
            _t_encode = time.perf_counter() - _t0

            # ── Step 2: prime mapping ──────────────────────────
            _mapper_bench = DiscreteMapper(
                n_bits=st.session_state.lsh_bits,
                seed=st.session_state.lsh_seed,
                projection=_bench_mode,
            )
            _t1 = time.perf_counter()
            _pmap = _mapper_bench.fit_transform(_words, _embs)
            _t_map = time.perf_counter() - _t1

            # ── Step 3: GCD search (all-pairs) ─────────────────
            _primes = list(_pmap.values())
            _t2 = time.perf_counter()
            _hits = sum(1 for i in range(len(_primes)) for j in range(i + 1, len(_primes))
                        if math.gcd(_primes[i], _primes[j]) > 1)
            _t_gcd = time.perf_counter() - _t2

        _total = _t_encode + _t_map + _t_gcd

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Encoding", f"{_t_encode*1000:.1f} ms",
                   help="SentenceTransformer → float32 embeddings")
        bc2.metric("Prime Mapping", f"{_t_map*1000:.1f} ms",
                   help="LSH hyperplane projection → prime integers")
        bc3.metric("GCD Search", f"{_t_gcd*1000:.1f} ms",
                   help=f"All-pairs GCD on {len(_primes)} concepts ({_hits} connected pairs)")
        bc4.metric("Total", f"{_total*1000:.1f} ms")

        _n_clusters = len(set(_pmap.values()))
        st.info(
            f"**{len(_primes)} concepts** → **{_n_clusters} prime clusters** "
            f"| {_hits} semantic connections found "
            f"| mode: `{_bench_mode}` | k={st.session_state.lsh_bits}"
        )

        st.markdown("**Throughput**")
        st.table(pd.DataFrame({
            "Phase":          ["Embedding", "Prime Mapping", "GCD Search", "Total"],
            "Time (ms)":      [f"{_t_encode*1000:.1f}", f"{_t_map*1000:.1f}",
                               f"{_t_gcd*1000:.1f}", f"{_total*1000:.1f}"],
            "concepts/sec":   [
                f"{len(_primes)/_t_encode:,.0f}",
                f"{len(_primes)/_t_map:,.0f}",
                "—",
                f"{len(_primes)/_total:,.0f}",
            ],
        }))
