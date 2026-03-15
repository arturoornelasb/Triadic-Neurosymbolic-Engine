import numpy as np
import logging
from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.triadic import DiscreteValidator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_quantum_bridge():
    print("=== Phase 3: The Quantum Bridge (Encoder Test) ===")
    
    # 1. Real World Dictionary Words
    words = [
        "King", "Ruler", "Monarch",
        "Queen", "Empress",
        "Man", "Male", "Boy",
        "Woman", "Female", "Girl",
        "Apple", "Banana", "Fruit"
    ]
    
    # 2. Extract Continuous Embeddings (using sentence-transformers)
    print("\n[Continuous Space $\\mathbb{R}^n$]")
    encoder = ContinuousEncoder()
    embeddings = encoder.encode(words)
    assert embeddings is not None, "encode() should return embeddings"
    assert embeddings.shape[0] == len(words), f"Expected {len(words)} embeddings, got {embeddings.shape[0]}"
    print(f"Extracted dense embeddings. Shape: {embeddings.shape}")
    
    # 3. Discretize via LSH and Prime Factorization
    print("\n[Discrete Space $\\mathbb{Z}$]")
    # We use a small number of bits (e.g. 4) so that embeddings are forced into fewer semantic buckets
    mapper = DiscreteMapper(n_bits=4, seed=42)
    prime_map = mapper.fit_transform(words, embeddings)
    assert len(prime_map) == len(words), f"Expected {len(words)} prime mappings, got {len(prime_map)}"

    print("\nAssigned Prime Factors by Semantic Clusters:")
    for word, p_factor in prime_map.items():
        print(f" - {word}: {p_factor}")
        
    print("\n[Discrete Algebraic Validation]")
    validator = DiscreteValidator()
    
    c_king = prime_map["King"]
    c_man = prime_map["Man"]
    c_queen = prime_map["Queen"]
    c_woman = prime_map["Woman"]
    
    # Normally, King/Queen is not 1:1, but the LSH bucket groups them
    print(f"Analogy Check: King ({c_king}) : Man ({c_man}) :: Queen ({c_queen}) : Woman ({c_woman})")
    
    result = validator.analogy_prediction(c_man, c_king, c_woman)
    
    if result.is_valid:
        print(f"Success! Arithmetically consistent. Result: {result.output_value}")
    else:
        print(f"Obstruction Detected. Missing Factor: {result.missing_factor}")
        print(f"Trace: {result.trace}")
        
if __name__ == "__main__":
    test_quantum_bridge()
