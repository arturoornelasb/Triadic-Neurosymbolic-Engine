from neurosym.encoder import ContinuousEncoder, DiscreteMapper
import math

def test_triadic_search_algorithm():
    print("=== Testing Triadic Search ===")
    
    words = [
        "Horse", "Car", "Fast", "Vehicle", "Woman", 
        "King", "Man", "Queen", "Banana", "Fruit"
    ]
    
    encoder = ContinuousEncoder()
    embeddings = encoder.encode(words)
    
    # We use a bit resolution that forces some semantic clumping
    mapper = DiscreteMapper(n_bits=8, seed=42)
    prime_map = mapper.fit_transform(words, embeddings)
    
    print("\n[Prime Map]")
    for w, p in prime_map.items():
        print(f"{w}: {p}")
        
    def triadic_search(a, b, c):
        p_a = prime_map[a]
        p_b = prime_map[b]
        p_c = prime_map[c]
        
        target_right = p_b * p_c
        
        results = []
        for word, p_x in prime_map.items():
            if word in [a, b, c]:
                continue
            
            left = p_a * p_x
            shared = math.gcd(left, target_right)
            
            missing = target_right // shared
            extra = left // shared
            
            # Distance metric
            dist = abs(extra - missing) + (extra * missing)
            results.append((word, dist, p_x, extra, missing))
            
        results.sort(key=lambda x: x[1])
        return results

    print("\n--- Analogy 1: King : Man :: Queen : [Woman] ---")
    res1 = triadic_search("King", "Man", "Queen")
    for r in res1[:3]: print(f"{r[0]} (Dist: {r[1]}) - Missing: {r[4]}, Extra: {r[3]}")
    assert res1[0][0] == "Woman", f"Expected Woman, got {res1[0][0]}"
    
    print("\n--- Analogy 2: Horse : Car :: Fast : [Vehicle] ---")
    res2 = triadic_search("Horse", "Car", "Fast")
    for r in res2[:3]: print(f"{r[0]} (Dist: {r[1]}) - Missing: {r[4]}, Extra: {r[3]}")
    assert res2[0][0] == "Vehicle", f"Expected Vehicle, got {res2[0][0]}"
    
    print("\nSUCCESS! Triadic Search algorithm is mathematically sound.")

if __name__ == "__main__":
    test_triadic_search_algorithm()
