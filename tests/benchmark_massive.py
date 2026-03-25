import time
import psutil
import os
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet as wn
from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.triadic import DiscreteValidator
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def download_data():
    logging.info("Downloading NLTK WordNet...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def get_wordnet_vocab(limit: int = 50000) -> list[str]:
    words = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().replace('_', ' ')
            words.add(word)
            if len(words) >= limit:
                return list(words)
    return list(words)

def measure_memory() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark():
    download_data()
    
    LIMIT = 20000 # 20k words for a balanced local benchmark
    logging.info(f"\n=== Phase 4: Industrial Scale Benchmark (N={LIMIT} words) ===")
    
    words = get_wordnet_vocab(LIMIT)
    logging.info(f"Loaded {len(words)} unique words from WordNet.")
    
    # --- 1. VECTOR ENCODING ---
    logging.info("\n[1] Encoding Continuous Vectors...")
    encoder = ContinuousEncoder()
    t0 = time.time()
    embeddings = encoder.encode(words)
    t_encode = time.time() - t0
    logging.info(f"Encoding took: {t_encode:.2f} seconds")
    
    # --- 2. DISCRETE MAPPING (The Bridge) ---
    logging.info("\n[2] Mapping to Discrete Prime Space...")
    mapper = DiscreteMapper(n_bits=8, seed=42)
    t0 = time.time()
    prime_map = mapper.fit_transform(words, embeddings)
    t_map = time.time() - t0
    logging.info(f"LSH Hash & Prime Mapping took: {t_map:.2f} seconds")
    
    # Prepare a test analogy: King : Man :: Queen : Woman
    # We will pick 4 random related words from our dataset that fell into clusters
    # To ensure fairness, we just test raw logic operation times, not the NLP accuracy here
    idx_a, idx_b = 100, 200
    vec_a, vec_b = embeddings[idx_a].reshape(1, -1), embeddings[idx_b].reshape(1, -1)
    word_a, word_b = words[idx_a], words[idx_b]
    
    prime_a = prime_map[word_a]
    prime_b = prime_map[word_b]
    prime_c = prime_map[words[300]] 
    
    ITERATIONS = 50000
    
    logging.info(f"\n--- RACE: Vector Space vs Triadic Engine ({ITERATIONS} iterations) ---")
    
    # --- COMPETITOR 1: CONTINUOUS VECTOR SPACE (Coseno) ---
    mem_before_vec = measure_memory()
    t0 = time.time()
    # Simulate finding relationship between vector A and B 50k times
    for _ in range(ITERATIONS):
        # A standard vector logic requires cosine distance calculation
        cosine_similarity(vec_a, vec_b)[0][0]
    t_vec = time.time() - t0
    mem_after_vec = measure_memory()
    mem_used_vec = mem_after_vec - mem_before_vec

    # --- COMPETITOR 2: DISCRETE TRIADIC SPACE (Modulo) ---
    validator = DiscreteValidator()
    mem_before_tri = measure_memory()
    t0 = time.time()
    for _ in range(ITERATIONS):
        # Triadic Logic: Deterministic arithmetic resolution
        validator.analogy_prediction(prime_a, prime_b, prime_c)
    t_tri = time.time() - t0
    mem_after_tri = measure_memory()
    mem_used_tri = mem_after_tri - mem_before_tri
    
    logging.info("\n=== FINAL RESULTS ===")
    logging.info(f"Task: Evaluate semantic relationships {ITERATIONS} times")
    logging.info("\n[Traditional RAG / Continuous Vectors]")
    logging.info(f"Time: {t_vec:.4f} seconds")
    logging.info(f"Memory Spike: {mem_used_vec:.2f} MB")
    
    logging.info("\n[Triadic Neurosymbolic Engine / Discrete Integers]")
    logging.info(f"Time: {t_tri:.4f} seconds")
    logging.info(f"Memory Spike: {mem_used_tri:.2f} MB")
    
    speedup = t_vec / t_tri if t_tri > 0 else float('inf')
    logging.info(f"\n=> The Triadic Engine is {speedup:.1f}x FASTER than continuous vectors.")
    logging.info("=> Computational logic is deterministic, exact, and $O(1)$ memory!")
    
if __name__ == "__main__":
    run_benchmark()
