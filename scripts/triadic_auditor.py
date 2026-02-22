import os
import sys
import argparse
import pandas as pd
import time
from typing import List

# Ensure neurosym source is in path
ENGINE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_PATH)

from src.neurosym.encoder import ContinuousEncoder, DiscreteMapper
from src.neurosym.triadic import DiscreteValidator

def run_audit(input_csv: str, target_column: str, model_a: str, model_b: str, output_csv: str, lsh_bits: int = 8, sample_size: int = None):
    print("=========================================================")
    print("🤖 TRIADIC RAG-SHIELD: DATABASE AUDITOR")
    print("=========================================================")
    
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file '{input_csv}' not found.")
        return

    # 1. Load Data
    print(f"\n1. Ingesting Database from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if target_column not in df.columns:
        print(f"❌ Error: Target column '{target_column}' not found in the CSV. Available columns: {list(df.columns)}")
        return
        
    # Drop NaNs and ensure string type
    concepts = df[target_column].dropna().astype(str).tolist()
    
    if sample_size and sample_size < len(concepts):
        print(f"   -> Sampling {sample_size} concepts from a total of {len(concepts)}.")
        import random
        random.seed(42)
        concepts = random.sample(concepts, sample_size)
    else:
        print(f"   -> Found {len(concepts)} valid concepts to audit.")

    if len(concepts) < 2:
        print("❌ Error: Need at least 2 concepts to perform meaningful analysis.")
        return

    # 2. Load Models
    print(f"\n2. Loading AI Neural Matrices (Vectorizers)...")
    t0 = time.time()
    print(f"   -> Loading Brain A: {model_a}")
    encoder_A = ContinuousEncoder(model_name=model_a)
    
    print(f"   -> Loading Brain B: {model_b}")
    encoder_B = ContinuousEncoder(model_name=model_b)
    t1 = time.time()
    print(f"   [Models loaded in {t1-t0:.2f} seconds]")

    validator = DiscreteValidator()

    # 3. Vectorization
    print(f"\n3. Vectorizing {len(concepts)} concepts across both Brains...")
    t0 = time.time()
    embeddings_A = encoder_A.encode(concepts)
    embeddings_B = encoder_B.encode(concepts)
    t1 = time.time()
    print(f"   [Vectorization complete in {t1-t0:.2f} seconds]")

    # 4. Triadic Mapping (Continuous to Discrete)
    print(f"\n4. Triadic Hashing: Projecting vectors to Prime Factor Space (LSH k={lsh_bits})...")
    # CRITICAL: Same random seed ensures LSH hyperplanes are geometrically aligned
    mapper_A = DiscreteMapper(n_bits=lsh_bits, seed=123)
    mapper_B = DiscreteMapper(n_bits=lsh_bits, seed=123)

    prime_map_A = mapper_A.fit_transform(concepts, embeddings_A)
    prime_map_B = mapper_B.fit_transform(concepts, embeddings_B)

    # 5. Semantic Gap Analysis via Topological Graphs
    print("\n5. Executing Topological Shortest-Path Gap Analysis (Auditing Biases)...")
    
    import networkx as nx
    import math

    graph_A = nx.Graph()
    graph_B = nx.Graph()
    graph_A.add_nodes_from(concepts)
    graph_B.add_nodes_from(concepts)
    
    print("   -> Building O(N^2) Semantic Edges from Primes...")
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            w1, w2 = concepts[i], concepts[j]
            if math.gcd(prime_map_A[w1], prime_map_A[w2]) > 1:
                graph_A.add_edge(w1, w2)
            if math.gcd(prime_map_B[w1], prime_map_B[w2]) > 1:
                graph_B.add_edge(w1, w2)

    print("   -> Computing Shortest Semantic Paths...")
    # Precompute all pairs shortest paths for speed
    paths_A = dict(nx.all_pairs_shortest_path_length(graph_A))
    paths_B = dict(nx.all_pairs_shortest_path_length(graph_B))
    
    results = []
    discrepancy_count = 0
    total_pairs = 0
    
    # We will sample if it's too massive, otherwise do all. 2000x2000/2 is 2 million pairs.
    # To keep the CSV reasonable, let's only save pairs that actually diverged!
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            total_pairs += 1
            w1 = concepts[i]
            w2 = concepts[j]
            
            # Use dict.get() for fast lookup. If it's not in the dict, they are disconnected (inf)
            dist_topological_A = paths_A.get(w1, {}).get(w2, float('inf'))
            dist_topological_B = paths_B.get(w1, {}).get(w2, float('inf'))
            
            if dist_topological_A != dist_topological_B:
                discrepancy_count += 1
                
                # We only save discrepancies to avoid a 2 million row CSV mostly full of non-connections
                # Extract paths if they exist
                path_A = " ➡️ ".join(nx.shortest_path(graph_A, w1, w2)) if dist_topological_A != float('inf') else "❌ Disconnected"
                path_B = " ➡️ ".join(nx.shortest_path(graph_B, w1, w2)) if dist_topological_B != float('inf') else "❌ Disconnected"
                
                results.append({
                    "Concept 1": w1,
                    "Concept 2": w2,
                    f"Dist. in {model_a}": dist_topological_A if dist_topological_A != float('inf') else "INF",
                    f"Dist. in {model_b}": dist_topological_B if dist_topological_B != float('inf') else "INF",
                    f"Chain in {model_a}": path_A,
                    f"Chain in {model_b}": path_B
                })

    # 6. Save Report
    print(f"\n6. Generating Triadic Audit Report (Saving only Semantic Biases)...")
    if results:
        report_df = pd.DataFrame(results)
        
        # Sort by distance in Model A
        report_df = report_df.sort_values(by=["Concept 1", f"Dist. in {model_a}"])
        report_df.to_csv(output_csv, index=False)
    else:
        # Create empty CSV with headers
        report_df = pd.DataFrame(columns=["Concept 1", "Concept 2", f"Dist. in {model_a}", f"Dist. in {model_b}", f"Chain in {model_a}", f"Chain in {model_b}"])
        report_df.to_csv(output_csv, index=False)
    
    print("\n=========================================================")
    print("📈 AUDIT SUMMARY (Topological Shortest-Path)")
    print("=========================================================")
    print(f"- Total Concepts Evaluated: {len(concepts)}")
    print(f"- Total Possible Chains (Pairs): {total_pairs}")
    print(f"- Relational Biases Discovered: {discrepancy_count} ({(discrepancy_count/total_pairs)*100:.1f}%)")
    print(f"\n✅ Audit complete. Saved {discrepancy_count} divergent chains to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triadic RAG-Shield DB Auditor - Compare LLM Semantic Structuring via Prime Factors")
    
    # We provide default values so it's easy to run out of the box as a prototype
    parser.add_argument("--input", "-i", type=str, default="examples/data/sample_audit.csv", help="Path to input CSV database")
    parser.add_argument("--col", "-c", type=str, default="word", help="The text column to audit")
    parser.add_argument("--model-a", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer Model A (Base)")
    parser.add_argument("--model-b", type=str, default="paraphrase-MiniLM-L3-v2", help="Sentence Transformer Model B (Target)")
    parser.add_argument("--output", "-o", type=str, default="audit_report.csv", help="Path to save the output discrepancy CSV report")
    parser.add_argument("--bits", "-b", type=int, default=8, help="LSH Hash Bits (Resolution dimension)")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Optionally limit audit to N random rows")

    args = parser.parse_args()
    
    # Auto-create sample data if requested and doesn't exist
    if args.input == "examples/data/sample_audit.csv" and not os.path.exists("examples/data"):
        os.makedirs("examples/data", exist_ok=True)
        print("Creating default sample dictionary for audit...")
        sample_data = pd.DataFrame({
            "word": [
                "Liberty", "Freedom", "Justice", "Equality", "Democracy",
                "Engineer", "Nurse", "Doctor", "Teacher", "CEO", "Assistant",
                "Fire", "Water", "Earth", "Air",
                "Happy", "Sad", "Angry", "Fear"
            ]
        })
        sample_data.to_csv("examples/data/sample_audit.csv", index=False)
        print(f"Generated sample at {args.input}")

    run_audit(
        input_csv=args.input,
        target_column=args.col,
        model_a=args.model_a,
        model_b=args.model_b,
        output_csv=args.output,
        lsh_bits=args.bits,
        sample_size=args.sample
    )
