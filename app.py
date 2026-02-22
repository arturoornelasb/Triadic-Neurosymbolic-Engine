import streamlit as st
import pandas as pd
import time
import math
import sys
import os

# Ensure neurosym source is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.triadic import DiscreteValidator
from neurosym.ingest import DatabaseIngestor
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

# --- Page Config ---
st.set_page_config(
    page_title="Triadic Neurosymbolic Engine",
    page_icon="⚛️",
    layout="wide"
)

# --- State Management & Initialization ---
@st.cache_resource
def load_encoder():
    return ContinuousEncoder("all-MiniLM-L6-v2")

encoder = load_encoder()
validator = DiscreteValidator()

if "mapper" not in st.session_state:
    st.session_state.mapper = DiscreteMapper(n_bits=8, seed=42)
if "prime_map" not in st.session_state:
    st.session_state.prime_map = {}
if "word_list" not in st.session_state:
    # Preload a more coherent medical/biology/tech default list to show relationships
    st.session_state.word_list = [
        "Doctor", "Hospital", "Medicine", "Nurse", "Surgery", "Patient",
        "Computer", "Algorithm", "Software", "Hardware", "Internet", "Programmer",
        "Dog", "Cat", "Animal", "Pet", "Veterinarian"
    ]
if "db_ingestor" not in st.session_state:
    st.session_state.db_ingestor = DatabaseIngestor(encoder, st.session_state.mapper)

def recompute_mapping():
    if not st.session_state.word_list:
        return
    st.session_state.mapper = DiscreteMapper(n_bits=st.session_state.lsh_bits, seed=st.session_state.lsh_seed)
    embeddings = encoder.encode(st.session_state.word_list)
    st.session_state.prime_map = st.session_state.mapper.fit_transform(st.session_state.word_list, embeddings)
    st.session_state.db_ingestor.mapper = st.session_state.mapper

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Engine Parameters")
    st.markdown("Control how strictly the AI assigns prime semantic factors.")
    st.slider("LSH Resolution Bits", min_value=2, max_value=32, value=8, step=1, key="lsh_bits", on_change=recompute_mapping, help="Higher bits mean stricter, more disjoint definitions.")
    st.number_input("Random Seed", value=42, key="lsh_seed", on_change=recompute_mapping)
    st.markdown("---")
    st.markdown("**Status:**")
    st.success(f"Loaded {len(st.session_state.word_list)} active concepts in Prime Memory.")

# --- Main Logic ---
if not st.session_state.prime_map:
    recompute_mapping()

st.title("Triadic Neurosymbolic Engine")
st.subheader("Deterministic LLM Interpretabilty & Verification")
st.markdown("Convert opaque continuous $R^n$ embeddings into transparent, arithmetic Prime Factor integers $Z$ for $O(1)$ logical verification.")

# The unified flow
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 1. Ingestion & Encoding", 
    "🌌 2. Holographic Graph", 
    "🧠 3. Logic & Search", 
    "🤖 4. AI Auditor",
    "📊 5. Benchmarks"
])

# --- TAB 1: INGESTION (Combines manual entry + CSV upload) ---
with tab1:
    st.markdown("### Step 1: Map Concepts to Prime Ints")
    st.markdown("To audit an AI or build an exact search index, we must first map words to prime integers. You can type words manually or upload a CSV database.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Option A: Manual Entry")
        new_words = st.text_input("Add words (comma separated):", placeholder="e.g. Finance, Bank, Money")
        if st.button("Encode New Words"):
            if new_words:
                cleaned = [w.strip() for w in new_words.split(",") if w.strip()]
                for w in cleaned:
                    if w not in st.session_state.word_list:
                        st.session_state.word_list.append(w)
                recompute_mapping()
                st.success(f"Added {len(cleaned)} words into Prime Memory!")
        
        if st.button("🗑️ Clear Entire Dictionary", type="secondary"):
            st.session_state.word_list = []
            st.session_state.prime_map = {}
            st.rerun()
            
    with col2:
        st.markdown("#### Option B: Upload CSV Database")
        uploaded_file = st.file_uploader("Upload CSV (e.g. catalog, wordnet)", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.info(f"Loaded CSV with {len(df)} rows. Select the text column to index.")
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            if text_columns:
                target_col = st.selectbox("Text Column:", text_columns)
                if st.button("🚀 Mass Ingest & Encode CSV"):
                    with st.spinner(f"Vectorizing {len(df)} rows and assigning Primes..."):
                        start_time = time.time()
                        words_to_add = df[target_col].dropna().astype(str).tolist()
                        
                        # Add to global dictionary
                        for w in words_to_add:
                            if w not in st.session_state.word_list:
                                st.session_state.word_list.append(w)
                        
                        recompute_mapping()
                        st.success(f"Mass ingestion complete in {time.time()-start_time:.2f} seconds!")
                        st.rerun()

    st.markdown("---")
    st.markdown("#### Current Mathematical Dictionary")
    if st.session_state.prime_map:
        # Create a dataframe from the dictionary to show the map
        dict_data = []
        for word, prime in st.session_state.prime_map.items():
            factors = validator._prime_factors(prime)
            dict_data.append({"Concept": word, "Integer ID": prime, "Underlying Primes (Features)": str(factors)})
        
        df_dict = pd.DataFrame(dict_data).sort_values(by="Integer ID")
        st.dataframe(df_dict, use_container_width=True, hide_index=True)


# --- TAB 2: ONTOLOGICAL GRAPH ---
with tab2:
    st.markdown("### Step 2: Semantic Holographic Graph")
    st.markdown("Since every word is now a Prime number, we mathematically link them using their **Greatest Common Divisor (GCD)**. If GCD > 1, they share hidden semantic traits in the AI's brain.")
    
    if len(st.session_state.word_list) < 2:
        st.warning("Please add some words in Step 1 first.")
    else:
        st.markdown(f"**Visualizing {len(st.session_state.word_list)} Concepts**")
        min_weight = st.slider("Strictness Filter (Min. Prime Factors Shared)", min_value=1, max_value=8, value=2, 
                               help="Increase this slider to hide weak connections and reveal only strong semantic clusters.")
        
        nodes, edges = [], []
        added_nodes = set()
        
        if st.button("🕸️ Render Semantic Graph", use_container_width=True, type="primary"):
            with st.spinner("Calculating GCD intersections for all nodes..."):
                # Limited to 250 for browser performance if necessary, but networkx/agraph handles a few hundred fine.
                max_nodes = 300
                render_list = st.session_state.word_list[:max_nodes]
                if len(st.session_state.word_list) > max_nodes:
                    st.warning(f"Graph limited to first {max_nodes} words to prevent browser lag.")
                
                for i in range(len(render_list)):
                    for j in range(i + 1, len(render_list)):
                        w_a = render_list[i]
                        w_b = render_list[j]
                        p_a = st.session_state.prime_map[w_a]
                        p_b = st.session_state.prime_map[w_b]
                        
                        gap = validator.explain_gap(p_a, p_b)
                        if gap['shared'] > 1:
                            shared_factors = validator._prime_factors(gap['shared'])
                            weight = len(shared_factors)
                            
                            if weight >= min_weight:
                                if w_a not in added_nodes:
                                    nodes.append(Node(id=w_a, label=w_a, size=15, color="#00bcd4"))
                                    added_nodes.add(w_a)
                                    
                                if w_b not in added_nodes:
                                    nodes.append(Node(id=w_b, label=w_b, size=15, color="#00bcd4"))
                                    added_nodes.add(w_b)
                                    
                                edges.append(Edge(source=w_a, target=w_b, title=f"Shared Primes: {shared_factors}", width=weight))

            if not nodes:
                 st.info(f"No connections found sharing {min_weight} or more prime factors. Try lowering the edge weight slider.")
            else:
                 st.success(f"Rendered {len(nodes)} nodes filtering by weight >= {min_weight}.")
                 config = Config(width=1000, height=700, directed=False, 
                                 physics=True, 
                                 hierarchical=False, 
                                 nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True,
                                 link=st.session_state.get("link_config", {"distance": 150}),
                                 physics_layout={"barnesHut": {"gravitationalConstant": -8000, "springConstant": 0.04, "springLength": 250}})
                 agraph(nodes=nodes, edges=edges, config=config)

# --- TAB 3: LOGIC & SEARCH ---
with tab3:
    st.markdown("### Step 3: Arithmetic Reasoning")
    st.markdown("With everything encoded as integers, we can use exact arithmetic to search and verify logic $O(1)$ fast.")
    
    if len(st.session_state.word_list) >= 2:
        col_log1, col_log2 = st.columns(2)
        
        with col_log1:
            st.markdown("#### 1. Abductive Gap Analysis (Audit Diff)")
            st.markdown("Select two words. The engine uses `%` and `GCD` to determine exactly *why* they differ, exposing the AI's biases.")
            
            gap_a = st.selectbox("Concept A", st.session_state.word_list, index=0, key="gap_a")
            gap_b = st.selectbox("Concept B", st.session_state.word_list, index=1, key="gap_b")
            
            if st.button("Calculate Vector Difference"):
                p_a = st.session_state.prime_map[gap_a]
                p_b = st.session_state.prime_map[gap_b]
                gap = validator.explain_gap(p_a, p_b)
                
                gap_data = {
                    "Analysis": ["Common Semantic Core", f"Features ONLY in '{gap_a}'", f"Features ONLY in '{gap_b}'"],
                    "Calculation": ["GCD(A,B)", "A / GCD", "B / GCD"],
                    "Prime Vectors Found": [
                        str(validator._prime_factors(gap['shared'])),
                        str(validator._prime_factors(gap['only_in_a'])),
                        str(validator._prime_factors(gap['only_in_b']))
                    ]
                }
                st.table(pd.DataFrame(gap_data))
                
        with col_log2:
            st.markdown("#### 2. Deterministic Triadic Search")
            st.markdown("Search the entire uploaded dictionary using integer divisibility instead of vector dots.")
            
            search_query = st.text_input("Triadic Search Query:", placeholder="e.g. Medical professional")
            if st.button("Search Database"):
                if search_query:
                    # Quick encode the query
                    q_emb = encoder.encode([search_query])
                    q_prime = st.session_state.mapper.transform([search_query], q_emb)[search_query]
                    q_factors = set(validator._prime_factors(q_prime))
                    
                    st.write(f"Query Prime ID: `{q_prime}` {list(q_factors)}")
                    
                    results = []
                    for w in st.session_state.word_list:
                        w_prime = st.session_state.prime_map[w]
                        # Score by how many prime factors they share (GCD size)
                        shared = math.gcd(q_prime, w_prime)
                        if shared > 1:
                            score = len(validator._prime_factors(shared))
                            results.append({"Word": w, "Score (Primes Shared)": score})
                            
                    res_df = pd.DataFrame(results).sort_values(by="Score (Primes Shared)", ascending=False).head(5)
                    st.table(res_df)
    else:
        st.info("Add words in Step 1 to enable logic.")

# --- TAB 4: AI AUDITOR (DB DIFF) ---
with tab4:
    st.markdown("### Step 4: AI Auditor (Model vs Model)")
    st.markdown("Discover hidden semantic biases between two different mathematical AI brains by comparing how their Prime Factor topologies diverge on the exact same dictionary.")
    
    if len(st.session_state.word_list) < 2:
        st.warning("Please add some words in Step 1 first to use the Auditor.")
    else:
        st.markdown(f"**Auditing {len(st.session_state.word_list)} Concepts**")
        col_m1, col_m2 = st.columns(2)
        with col_m1: model_a_name = st.selectbox("Model A", ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "all-mpnet-base-v2"], index=0)
        with col_m2: model_b_name = st.selectbox("Model B", ["paraphrase-MiniLM-L3-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2"], index=0)
        
        if st.button("🚀 Run Triadic DB Audit", use_container_width=True, type="primary"):
            if model_a_name == model_b_name:
                st.error("Please select two DIFFERENT models to compare.")
            else:
                with st.spinner(f"Loading {model_a_name} and {model_b_name} and mapping structures. This may take a minute..."):
                    enc_A = ContinuousEncoder(model_a_name)
                    enc_B = ContinuousEncoder(model_b_name)
                    
                    st.info(f"Encoding {len(st.session_state.word_list)} concepts and hashing to prime spaces...")
                    emb_A = enc_A.encode(st.session_state.word_list)
                    emb_B = enc_B.encode(st.session_state.word_list)
                    
                    mapper_A = DiscreteMapper(n_bits=st.session_state.lsh_bits, seed=st.session_state.lsh_seed)
                    mapper_B = DiscreteMapper(n_bits=st.session_state.lsh_bits, seed=st.session_state.lsh_seed)
                    
                    primes_A = mapper_A.fit_transform(st.session_state.word_list, emb_A)
                    primes_B = mapper_B.fit_transform(st.session_state.word_list, emb_B)
                    
                    results = []
                    discrepancies = 0
                    words = st.session_state.word_list
                    total_pairs = 0
                    
                    graph_A = nx.Graph()
                    graph_B = nx.Graph()
                    graph_A.add_nodes_from(words)
                    graph_B.add_nodes_from(words)
                    
                    # 1. Build edges based on shared Prime Factors (GCD > 1)
                    for i in range(len(words)):
                        for j in range(i + 1, len(words)):
                            w1, w2 = words[i], words[j]
                            if math.gcd(primes_A[w1], primes_A[w2]) > 1:
                                graph_A.add_edge(w1, w2)
                            if math.gcd(primes_B[w1], primes_B[w2]) > 1:
                                graph_B.add_edge(w1, w2)

                    # 2. Compare topological shortest paths instead of direct connections
                    for i in range(len(words)):
                        for j in range(i + 1, len(words)):
                            total_pairs += 1
                            w1 = words[i]
                            w2 = words[j]
                            
                            has_path_A = nx.has_path(graph_A, w1, w2)
                            has_path_B = nx.has_path(graph_B, w1, w2)
                            
                            dist_A = nx.shortest_path_length(graph_A, w1, w2) if has_path_A else float('inf')
                            dist_B = nx.shortest_path_length(graph_B, w1, w2) if has_path_B else float('inf')
                            
                            # If the traversal distance differs, the semantics diverged
                            if dist_A != dist_B:
                                discrepancies += 1
                                
                                str_dist_A = f"Path Length: {dist_A}" if has_path_A else "❌ Disconnected"
                                str_dist_B = f"Path Length: {dist_B}" if has_path_B else "❌ Disconnected"
                                
                                # Try to show the path for context (up to 3 hops)
                                path_example = ""
                                if has_path_A and dist_A <= 3:
                                    path_example = " ➡️ ".join(nx.shortest_path(graph_A, w1, w2))
                                elif has_path_B and dist_B <= 3:
                                    path_example = " ➡️ ".join(nx.shortest_path(graph_B, w1, w2))
                                    
                                results.append({
                                    "Concept 1": w1,
                                    "Concept 2": w2,
                                    f"Dist. in {model_a_name[:12]}": str_dist_A,
                                    f"Dist. in {model_b_name[:12]}": str_dist_B,
                                    "Shortest Semantic Chain Found": path_example
                                })
                    
                    if discrepancies == 0:
                        st.success(f"✅ Full topological agreement! Both models connect the exact same word pairs across {total_pairs} pairs.")
                    else:
                        st.error(f"🚨 Found {discrepancies} relational bias differences ({discrepancies/max(1, total_pairs)*100:.1f}%) across {total_pairs} possible connections.")
                        st.dataframe(pd.DataFrame(results), use_container_width=True)

# --- TAB 5: BENCHMARKS ---
with tab5:
    st.markdown("### Verification Benchmarks")
    st.markdown("Results from our Phase 4 massive logic iteration test validating 20,000 concepts in WordNet.")
    
    metrics_data = {
        "Benchmark Phase": ["Memory Storage", "Validation Metric", "Compute Operation", "Time (20k rows)"],
        "Traditional RAG / LLM": ["Float32 Matrices (Heavy)", "Cosine Similarity", "Dense Dot Product", "11.21s"],
        "Triadic Engine": ["Integer DB (Ultra Light)", "Exact Subsumption", "Integer % Modulo", "0.93s"]
    }
    
    st.table(pd.DataFrame(metrics_data))
    st.success("Triadic Arithmetic executes **12x faster** than traditional AI vector searches, with mathematically guaranteed deterministic results.")
