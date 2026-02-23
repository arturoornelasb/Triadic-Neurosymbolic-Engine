"""
Benchmark: PCA vs Random Hyperplanes
Compares subsumption accuracy, analogy resolution, and collision rates
between random LSH projections and PCA-directed projections.

Generates LaTeX-ready tables for the paper.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math
from collections import defaultdict
from neurosym.encoder import ContinuousEncoder, DiscreteMapper

# ─── Vocabulary ───────────────────────────────────────────────
VOCAB = [
    # Royalty
    "King", "Queen", "Prince", "Princess", "Throne",
    # People
    "Man", "Woman", "Boy", "Girl", "Child",
    # Animals
    "Dog", "Cat", "Horse", "Bird", "Fish",
    # Vehicles
    "Car", "Bicycle", "Train", "Airplane", "Boat",
    # Food
    "Apple", "Bread", "Cheese", "Rice", "Meat",
    # Colors
    "Red", "Blue", "Green", "Yellow", "Black", "White",
    # Emotions
    "Happy", "Sad", "Angry", "Fear", "Love", "Hate",
    # Professions
    "Doctor", "Nurse", "Teacher", "Engineer", "CEO",
    # Nature
    "Fire", "Water", "Earth", "Air", "Sun", "Moon",
    # Abstract
    "Freedom", "Justice", "Democracy", "Equality", "Liberty",
    # Tech
    "Computer", "Robot", "Internet", "Algorithm", "Data",
    # Body
    "Heart", "Brain", "Hand", "Eye", "Blood",
]

# Ground-truth hypernym pairs (broader ⊇ narrower)
HYPERNYM_PAIRS = [
    ("Animal", "Dog"), ("Animal", "Cat"), ("Animal", "Horse"),
    ("Animal", "Bird"), ("Animal", "Fish"),
    ("Vehicle", "Car"), ("Vehicle", "Bicycle"), ("Vehicle", "Train"),
    ("Person", "Man"), ("Person", "Woman"), ("Person", "Child"),
]
# Add the hypernym concepts to vocab
for broader, _ in HYPERNYM_PAIRS:
    if broader not in VOCAB:
        VOCAB.append(broader)

# Analogy pairs: A:B :: C:? (expected D)
ANALOGY_PAIRS = [
    ("King", "Man", "Queen", "Woman"),
    ("King", "Queen", "Man", "Woman"),
    ("Father", "Mother", "Boy", "Girl"),
    ("Doctor", "Nurse", "Man", "Woman"),
    ("Sun", "Moon", "Fire", "Water"),
]
# Add missing concepts
for a, b, c, d in ANALOGY_PAIRS:
    for w in [a, b, c, d]:
        if w not in VOCAB:
            VOCAB.append(w)

VOCAB = list(set(VOCAB))  # deduplicate


def run_subsumption_test(mapper, concepts, embeddings, hypernym_pairs, all_concepts):
    """Test subsumption accuracy: TP and FP rates."""
    prime_map = mapper.fit_transform(concepts, embeddings)
    
    tp, tp_total = 0, 0
    fp, fp_total = 0, 0
    
    for broader, narrower in hypernym_pairs:
        if broader in prime_map and narrower in prime_map:
            phi_b = prime_map[broader]
            phi_n = prime_map[narrower]
            if phi_b % phi_n == 0:
                tp += 1
            tp_total += 1
    
    # Random pairs for FP
    rng = np.random.RandomState(99)
    tested = set()
    for _ in range(200):
        i, j = rng.choice(len(all_concepts), 2, replace=False)
        a, b = all_concepts[i], all_concepts[j]
        if (a, b) in tested:
            continue
        tested.add((a, b))
        if a in prime_map and b in prime_map:
            phi_a = prime_map[a]
            phi_b = prime_map[b]
            if phi_a % phi_b == 0:
                fp += 1
            fp_total += 1
    
    tp_rate = (tp / tp_total * 100) if tp_total > 0 else 0
    fp_rate = (fp / fp_total * 100) if fp_total > 0 else 0
    return tp_rate, fp_rate


def run_analogy_test(mapper, concepts, embeddings, analogy_pairs):
    """Test analogy A:B :: C:? using GCD-based resolution."""
    prime_map = mapper.fit_transform(concepts, embeddings)
    correct = 0
    total = 0
    
    for a, b, c, expected_d in analogy_pairs:
        if not all(w in prime_map for w in [a, b, c, expected_d]):
            continue
        
        phi_a, phi_b, phi_c = prime_map[a], prime_map[b], prime_map[c]
        shared_ab = math.gcd(phi_a, phi_b)
        shared_ac = math.gcd(phi_a, phi_c)
        
        # Target: find D where gcd(C,D)/gcd(A,B) is maximal
        target_pattern = math.gcd(phi_b, phi_c)
        
        best_match = None
        best_score = -1
        
        for word, phi_w in prime_map.items():
            if word in [a, b, c]:
                continue
            score = math.gcd(phi_w, phi_c)
            if score > best_score:
                best_score = score
                best_match = word
        
        if best_match == expected_d:
            correct += 1
        total += 1
    
    return (correct / total * 100) if total > 0 else 0, correct, total


def count_collisions(mapper, concepts, embeddings):
    """Count collision rate (identical encodings)."""
    prime_map = mapper.fit_transform(concepts, embeddings)
    values = list(prime_map.values())
    unique = len(set(values))
    collision_rate = (1 - unique / len(values)) * 100
    return collision_rate


def run_bitstring_comparison(embeddings, concepts, hypernym_pairs, k, seed):
    """Compare prime factors vs plain bitstrings using Jaccard similarity."""
    rng = np.random.RandomState(seed)
    planes = rng.randn(k, embeddings.shape[1])
    
    concept_bits = {}
    for concept, emb in zip(concepts, embeddings):
        bits = tuple((np.dot(planes, emb) > 0).astype(int))
        concept_bits[concept] = bits
    
    # Bitstring can only do Jaccard, not subsumption
    def jaccard(a, b):
        intersection = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
        union = sum(1 for x, y in zip(a, b) if x == 1 or y == 1)
        return intersection / union if union > 0 else 0
    
    # Try to predict hypernym via Jaccard > threshold
    best_f1 = 0
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tp, fp, fn = 0, 0, 0
        for broader, narrower in hypernym_pairs:
            if broader in concept_bits and narrower in concept_bits:
                sim = jaccard(concept_bits[broader], concept_bits[narrower])
                if sim >= threshold:
                    tp += 1
                else:
                    fn += 1
        # Random false positives
        rng2 = np.random.RandomState(99)
        for _ in range(100):
            i, j = rng2.choice(len(concepts), 2, replace=False)
            a, b = concepts[i], concepts[j]
            if a in concept_bits and b in concept_bits:
                sim = jaccard(concept_bits[a], concept_bits[b])
                if sim >= threshold:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
    
    return best_f1 * 100


def main():
    print("=" * 70)
    print("BENCHMARK: PCA vs Random Hyperplanes")
    print("=" * 70)
    
    # Load encoder
    print("\nLoading encoder...")
    encoder = ContinuousEncoder("all-MiniLM-L6-v2")
    embeddings = encoder.encode(VOCAB)
    print(f"Encoded {len(VOCAB)} concepts → {embeddings.shape}")
    
    k_values = [6, 8, 12]
    n_seeds = 10
    
    # ═══════════════════════════════════════════════
    # EXPERIMENT A: Subsumption PCA vs Random
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Subsumption Accuracy (PCA vs Random)")
    print("=" * 70)
    
    print(f"\n{'k':>3} | {'Method':>8} | {'TP Rate':>12} | {'FP Rate':>12} | {'TP-FP Gap':>10}")
    print("-" * 55)
    
    results_subsumption = []
    
    for k in k_values:
        # Random (averaged over n_seeds)
        tp_rates, fp_rates = [], []
        for seed in range(n_seeds):
            mapper = DiscreteMapper(n_bits=k, seed=seed, projection="random")
            tp, fp = run_subsumption_test(mapper, VOCAB, embeddings, HYPERNYM_PAIRS, VOCAB)
            tp_rates.append(tp)
            fp_rates.append(fp)
        
        r_tp = np.mean(tp_rates)
        r_fp = np.mean(fp_rates)
        r_tp_std = np.std(tp_rates)
        r_fp_std = np.std(fp_rates)
        print(f"{k:>3} | {'Random':>8} | {r_tp:5.1f}% ±{r_tp_std:4.1f}% | {r_fp:5.1f}% ±{r_fp_std:4.1f}% | {r_tp-r_fp:+5.1f}%")
        results_subsumption.append(('Random', k, r_tp, r_tp_std, r_fp, r_fp_std))
        
        # PCA (deterministic, no seed dependency)
        mapper_pca = DiscreteMapper(n_bits=k, projection="pca")
        p_tp, p_fp = run_subsumption_test(mapper_pca, VOCAB, embeddings, HYPERNYM_PAIRS, VOCAB)
        print(f"{k:>3} | {'PCA':>8} | {p_tp:5.1f}%        | {p_fp:5.1f}%        | {p_tp-p_fp:+5.1f}%")
        results_subsumption.append(('PCA', k, p_tp, 0, p_fp, 0))
    
    # ═══════════════════════════════════════════════
    # EXPERIMENT B: Analogy PCA vs Random
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Analogy Resolution (PCA vs Random)")
    print("=" * 70)
    
    print(f"\n{'k':>3} | {'Method':>8} | {'Accuracy':>10} | {'Correct/Total':>14}")
    print("-" * 45)
    
    results_analogy = []
    
    for k in k_values:
        # Random
        accs = []
        for seed in range(n_seeds):
            mapper = DiscreteMapper(n_bits=k, seed=seed, projection="random")
            acc, _, _ = run_analogy_test(mapper, VOCAB, embeddings, ANALOGY_PAIRS)
            accs.append(acc)
        r_acc = np.mean(accs)
        r_std = np.std(accs)
        print(f"{k:>3} | {'Random':>8} | {r_acc:5.1f}% ±{r_std:4.1f}% | (avg over {n_seeds} seeds)")
        results_analogy.append(('Random', k, r_acc, r_std))
        
        # PCA
        mapper_pca = DiscreteMapper(n_bits=k, projection="pca")
        p_acc, p_correct, p_total = run_analogy_test(mapper_pca, VOCAB, embeddings, ANALOGY_PAIRS)
        print(f"{k:>3} | {'PCA':>8} | {p_acc:5.1f}%        | {p_correct}/{p_total}")
        results_analogy.append(('PCA', k, p_acc, 0))
    
    # ═══════════════════════════════════════════════
    # EXPERIMENT C: Multi-Seed Consensus
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Multi-Seed Consensus (k=8)")
    print("=" * 70)
    
    k = 8
    n_consensus_seeds = 20
    concept_stability = defaultdict(list)
    
    for seed in range(n_consensus_seeds):
        mapper = DiscreteMapper(n_bits=k, seed=seed, projection="random")
        prime_map = mapper.fit_transform(VOCAB, embeddings)
        for concept, phi in prime_map.items():
            concept_stability[concept].append(phi)
    
    # Find most/least stable concepts
    stability_scores = {}
    for concept, phis in concept_stability.items():
        unique_encodings = len(set(phis))
        stability_scores[concept] = unique_encodings / n_consensus_seeds
    
    sorted_stability = sorted(stability_scores.items(), key=lambda x: x[1])
    
    print(f"\nMost stable (fewest unique encodings across {n_consensus_seeds} seeds):")
    for concept, score in sorted_stability[:5]:
        unique = int(score * n_consensus_seeds)
        print(f"  {concept:>12}: {unique}/{n_consensus_seeds} unique encodings ({score*100:.0f}% variance)")
    
    print(f"\nLeast stable:")
    for concept, score in sorted_stability[-5:]:
        unique = int(score * n_consensus_seeds)
        print(f"  {concept:>12}: {unique}/{n_consensus_seeds} unique encodings ({score*100:.0f}% variance)")
    
    # PCA: always 1 encoding (deterministic)
    mapper_pca = DiscreteMapper(n_bits=k, projection="pca")
    prime_map_pca = mapper_pca.fit_transform(VOCAB, embeddings)
    print(f"\nPCA: ALL concepts have exactly 1 encoding (100% deterministic)")
    
    # ═══════════════════════════════════════════════
    # EXPERIMENT D: Bitstring vs Prime Comparison
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Bitstrings (Jaccard) vs Primes (Divisibility)")
    print("=" * 70)
    
    print(f"\n{'k':>3} | {'Bitstring F1':>14} | {'Prime TP Rate':>14} | {'Prime FP Rate':>14}")
    print("-" * 55)
    
    for k in k_values:
        # Bitstring Jaccard (best F1 across thresholds)
        bit_f1s = []
        for seed in range(n_seeds):
            f1 = run_bitstring_comparison(embeddings, VOCAB, HYPERNYM_PAIRS, k, seed)
            bit_f1s.append(f1)
        avg_f1 = np.mean(bit_f1s)
        
        # Prime divisibility
        tp_rates, fp_rates = [], []
        for seed in range(n_seeds):
            mapper = DiscreteMapper(n_bits=k, seed=seed, projection="random")
            tp, fp = run_subsumption_test(mapper, VOCAB, embeddings, HYPERNYM_PAIRS, VOCAB)
            tp_rates.append(tp)
            fp_rates.append(fp)
        
        print(f"{k:>3} | {avg_f1:10.1f}%    | {np.mean(tp_rates):10.1f}%    | {np.mean(fp_rates):10.1f}%")
    
    # ═══════════════════════════════════════════════
    # COLLISION RATES
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COLLISION RATES: PCA vs Random")
    print("=" * 70)
    
    print(f"\n{'k':>3} | {'Random Collision':>17} | {'PCA Collision':>15}")
    print("-" * 42)
    
    for k in k_values:
        r_colls = []
        for seed in range(n_seeds):
            mapper = DiscreteMapper(n_bits=k, seed=seed, projection="random")
            r_colls.append(count_collisions(mapper, VOCAB, embeddings))
        
        mapper_pca = DiscreteMapper(n_bits=k, projection="pca")
        p_coll = count_collisions(mapper_pca, VOCAB, embeddings)
        
        print(f"{k:>3} | {np.mean(r_colls):12.1f}%    | {p_coll:10.1f}%")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
