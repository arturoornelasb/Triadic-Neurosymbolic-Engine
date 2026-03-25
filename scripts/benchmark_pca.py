"""
Benchmark: All 4 Projection Modes
Compares Random, PCA, Consensus, and Contrastive hyperplane methods
on subsumption accuracy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math
from neurosym.encoder import ContinuousEncoder, DiscreteMapper

# ─── Vocabulary ───────────────────────────────────────────────
VOCAB = [
    "King", "Queen", "Prince", "Princess", "Throne",
    "Man", "Woman", "Boy", "Girl", "Child",
    "Dog", "Cat", "Horse", "Bird", "Fish",
    "Car", "Bicycle", "Train", "Airplane", "Boat",
    "Apple", "Bread", "Cheese", "Rice", "Meat",
    "Red", "Blue", "Green", "Yellow", "Black", "White",
    "Happy", "Sad", "Angry", "Fear", "Love", "Hate",
    "Doctor", "Nurse", "Teacher", "Engineer", "CEO",
    "Fire", "Water", "Earth", "Air", "Sun", "Moon",
    "Freedom", "Justice", "Democracy", "Equality", "Liberty",
    "Computer", "Robot", "Internet", "Algorithm", "Data",
    "Heart", "Brain", "Hand", "Eye", "Blood",
    # Hypernym categories
    "Animal", "Vehicle", "Person", "Food", "Color",
    "Emotion", "Profession", "Element", "Father", "Mother",
]
# sorted() ensures deterministic vocab order across Python runs (PYTHONHASHSEED).
# Note: Contrastive TP at k=6 ranges 92-100% depending on vocab order because
# the gradient-free optimizer converges to different local optima. With sorted
# order the result is deterministic (96.2%). The paper reports 100% from one run.
VOCAB = sorted(set(VOCAB))

# Ground-truth hypernym pairs
HYPERNYM_PAIRS = [
    ("Animal", "Dog"), ("Animal", "Cat"), ("Animal", "Horse"),
    ("Animal", "Bird"), ("Animal", "Fish"),
    ("Vehicle", "Car"), ("Vehicle", "Bicycle"), ("Vehicle", "Train"),
    ("Vehicle", "Airplane"), ("Vehicle", "Boat"),
    ("Person", "Man"), ("Person", "Woman"), ("Person", "Child"),
    ("Person", "Boy"), ("Person", "Girl"),
    ("Food", "Apple"), ("Food", "Bread"), ("Food", "Cheese"),
    ("Color", "Red"), ("Color", "Blue"), ("Color", "Green"),
    ("Emotion", "Happy"), ("Emotion", "Sad"), ("Emotion", "Angry"),
    ("Emotion", "Fear"), ("Emotion", "Love"),
]


def subsumption_test(prime_map, hypernym_pairs, all_concepts):
    """Returns (TP rate, FP rate)."""
    tp, tp_total = 0, 0
    for broader, narrower in hypernym_pairs:
        if broader in prime_map and narrower in prime_map:
            if prime_map[broader] % prime_map[narrower] == 0:
                tp += 1
            tp_total += 1

    rng = np.random.RandomState(99)
    fp, fp_total = 0, 0
    tested = set()
    for _ in range(200):
        i, j = rng.choice(len(all_concepts), 2, replace=False)
        a, b = all_concepts[i], all_concepts[j]
        if (a, b) in tested:
            continue
        tested.add((a, b))
        if a in prime_map and b in prime_map:
            if prime_map[a] % prime_map[b] == 0:
                fp += 1
            fp_total += 1

    tp_rate = (tp / tp_total * 100) if tp_total > 0 else 0
    fp_rate = (fp / fp_total * 100) if fp_total > 0 else 0
    return tp_rate, fp_rate


def main():
    print("=" * 70)
    print("FULL BENCHMARK: Random vs PCA vs Consensus vs Contrastive")
    print("=" * 70)

    encoder = ContinuousEncoder("all-MiniLM-L6-v2")
    embeddings = encoder.encode(VOCAB)
    print(f"Encoded {len(VOCAB)} concepts → {embeddings.shape}\n")

    k_values = [6, 8, 12]
    n_seeds = 10

    print(f"{'k':>3} | {'Method':>14} | {'TP Rate':>10} | {'FP Rate':>10} | {'TP-FP Gap':>10} | {'Note':>20}")
    print("-" * 80)

    for k in k_values:
        # ── Random (avg over seeds) ──
        tp_rates, fp_rates = [], []
        for seed in range(n_seeds):
            m = DiscreteMapper(n_bits=k, seed=seed, projection="random")
            pm = m.fit_transform(VOCAB, embeddings)
            tp, fp = subsumption_test(pm, HYPERNYM_PAIRS, VOCAB)
            tp_rates.append(tp)
            fp_rates.append(fp)
        r_tp, r_fp = np.mean(tp_rates), np.mean(fp_rates)
        print(f"{k:>3} | {'Random':>14} | {r_tp:7.1f}%  | {r_fp:7.1f}%  | {r_tp-r_fp:+7.1f}%  | {'avg 10 seeds':>20}")

        # ── PCA ──
        m = DiscreteMapper(n_bits=k, projection="pca")
        pm = m.fit_transform(VOCAB, embeddings)
        tp, fp = subsumption_test(pm, HYPERNYM_PAIRS, VOCAB)
        print(f"{k:>3} | {'PCA':>14} | {tp:7.1f}%  | {fp:7.1f}%  | {tp-fp:+7.1f}%  | {'deterministic':>20}")

        # ── Consensus ──
        for threshold in [0.5, 0.7]:
            m = DiscreteMapper(n_bits=k, projection="consensus",
                               consensus_seeds=20, consensus_threshold=threshold)
            pm = m.fit_transform(VOCAB, embeddings)
            tp, fp = subsumption_test(pm, HYPERNYM_PAIRS, VOCAB)
            label = f"Consensus({threshold})"
            print(f"{k:>3} | {label:>14} | {tp:7.1f}%  | {fp:7.1f}%  | {tp-fp:+7.1f}%  | {'20 seeds':>20}")

        # ── Contrastive ──
        m = DiscreteMapper(n_bits=k, projection="contrastive",
                           hypernym_pairs=HYPERNYM_PAIRS)
        pm = m.fit_transform(VOCAB, embeddings)
        tp, fp = subsumption_test(pm, HYPERNYM_PAIRS, VOCAB)
        print(f"{k:>3} | {'Contrastive':>14} | {tp:7.1f}%  | {fp:7.1f}%  | {tp-fp:+7.1f}%  | {'trained on pairs':>20}")

        print("-" * 80)

    print("\n✅ Benchmark complete.")


if __name__ == "__main__":
    main()
