from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.triadic import DiscreteValidator

def test_logical_verification():
    print("=== Logical Verification API Tests ===")
    
    encoder = ContinuousEncoder()
    mapper = DiscreteMapper(n_bits=8, seed=42)
    validator = DiscreteValidator()
    
    words = ["King", "Queen", "Man", "Woman", "Royal", "Male", "Female", "Crown", "Person"]
    embeddings = encoder.encode(words)
    pm = mapper.fit_transform(words, embeddings)
    
    print("\n[Prime Map]")
    for w, p in pm.items():
        print(f"  {w}: {p} (factors: {validator._prime_factors(p)})")
    
    # --- Test 1: Subsumption ---
    print("\n--- Test 1: Subsumption (A % B == 0?) ---")
    for a_word in words:
        for b_word in words:
            if a_word == b_word:
                continue
            result = validator.subsumes(pm[a_word], pm[b_word])
            if result:
                print(f"  ✅ '{a_word}' ({pm[a_word]}) SUBSUMES '{b_word}' ({pm[b_word]})")
    
    # --- Test 2: Composition ---
    print("\n--- Test 2: Algebraic Composition (LCM) ---")
    if "Male" in pm and "Royal" in pm:
        composed = validator.compose(pm["Male"], pm["Royal"])
        print(f"  compose(Male={pm['Male']}, Royal={pm['Royal']}) = {composed}")
        print(f"  factors of composed: {validator._prime_factors(composed)}")
        
        # The composed concept should subsume both originals
        assert validator.subsumes(composed, pm["Male"]), "Composed should subsume Male"
        assert validator.subsumes(composed, pm["Royal"]), "Composed should subsume Royal"
        print(f"  ✅ Composed concept subsumes both Male AND Royal")
    
    # --- Test 3: Explain Gap ---
    print("\n--- Test 3: Abductive Gap Explanation ---")
    for pair in [("King", "Queen"), ("Man", "Woman"), ("King", "Man")]:
        a, b = pair
        gap = validator.explain_gap(pm[a], pm[b])
        print(f"  explain_gap({a}={pm[a]}, {b}={pm[b]}):")
        print(f"    Shared backbone: {gap['shared']} (factors: {validator._prime_factors(gap['shared'])})")
        print(f"    Only in {a}: {gap['only_in_a']} (factors: {validator._prime_factors(gap['only_in_a'])})")
        print(f"    Only in {b}: {gap['only_in_b']} (factors: {validator._prime_factors(gap['only_in_b'])})")
        print(f"    {a} contains {b}: {gap['a_contains_b']}")
        print(f"    {b} contains {a}: {gap['b_contains_a']}")
    
    print("\n=== All Logical Verification Tests Passed ===")

if __name__ == "__main__":
    test_logical_verification()
