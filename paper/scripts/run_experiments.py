"""
Experiment Suite for the Prime Factorization Neurosymbolic Bridge Paper.

Runs all experiments described in Section 4 and generates CSV data files.
Experiments:
  1. Timing Benchmark: Cosine vs Prime GCD (50,000 validations)
  2. k-Value Sweep: Resolution-Collision Tradeoff (k=3,4,6,8,12,16)
  3. Subsumption Analysis: Precision of divisibility vs semantic containment
  4. Composition Verification: LCM subsumption guarantee
"""
import sys
import os
import time
import math
import csv
import random
import numpy as np

# Add the engine's src to path
ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Triadic-Neurosymbolic-Engine", "src")
sys.path.insert(0, ENGINE_PATH)

from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.triadic import DiscreteValidator

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# --- Shared Setup ---
print("Loading encoder...")
encoder = ContinuousEncoder()
validator = DiscreteValidator()

# Use a curated vocabulary for experiments
VOCAB = [
    # Royalty
    "King", "Queen", "Prince", "Princess", "Monarch", "Crown", "Throne", "Royal",
    # Gender
    "Man", "Woman", "Boy", "Girl", "Male", "Female", "Father", "Mother",
    # Animals
    "Dog", "Cat", "Horse", "Bird", "Fish", "Lion", "Tiger", "Eagle",
    # Vehicles
    "Car", "Truck", "Bicycle", "Motorcycle", "Bus", "Train", "Airplane", "Ship",
    # Food
    "Apple", "Banana", "Orange", "Bread", "Cheese", "Meat", "Rice", "Pasta",
    # Colors
    "Red", "Blue", "Green", "Yellow", "Black", "White", "Purple", "Gold",
    # Actions
    "Run", "Walk", "Swim", "Fly", "Jump", "Climb", "Drive", "Ride",
    # Abstract
    "Love", "Hate", "Fear", "Joy", "Anger", "Peace", "War", "Hope",
    # Nature
    "Tree", "Flower", "Mountain", "River", "Ocean", "Forest", "Desert", "Sky",
    # Objects
    "Book", "Pen", "Table", "Chair", "Door", "Window", "Lamp", "Clock",
    # Professions
    "Doctor", "Teacher", "Engineer", "Artist", "Soldier", "Farmer", "Judge", "Chef",
    # Body
    "Hand", "Eye", "Heart", "Head", "Foot", "Brain", "Blood", "Bone",
    # Size
    "Big", "Small", "Tall", "Short", "Wide", "Narrow", "Heavy", "Light",
]

# Known semantic groups for subsumption analysis
HYPERNYM_PAIRS = [
    # (broader, narrower) — broader should subsume narrower if encoding is good
    ("Animal", "Dog"), ("Animal", "Cat"), ("Animal", "Horse"), ("Animal", "Bird"),
    ("Vehicle", "Car"), ("Vehicle", "Truck"), ("Vehicle", "Bus"), ("Vehicle", "Train"),
    ("Fruit", "Apple"), ("Fruit", "Banana"), ("Fruit", "Orange"),
]

# We need "Animal", "Vehicle", "Fruit" in our vocab
# sorted() ensures deterministic vocab order across Python runs (PYTHONHASHSEED)
VOCAB_FULL = sorted(set(VOCAB + ["Animal", "Vehicle", "Fruit"]))

print(f"Vocabulary size: {len(VOCAB_FULL)}")
print("Encoding vocabulary...")
embeddings = encoder.encode(VOCAB_FULL)
emb_dict = {w: embeddings[i] for i, w in enumerate(VOCAB_FULL)}


def experiment_1_timing():
    """Experiment 1: Timing Benchmark — Cosine vs Prime GCD"""
    print("\n=== Experiment 1: Timing Benchmark ===")
    
    mapper = DiscreteMapper(n_bits=8, seed=42)
    prime_map = mapper.fit_transform(VOCAB_FULL, embeddings)
    
    N_VALIDATIONS = 50000
    pairs = [(random.choice(VOCAB_FULL), random.choice(VOCAB_FULL)) for _ in range(N_VALIDATIONS)]
    
    # --- Cosine similarity ---
    t0 = time.perf_counter()
    for w1, w2 in pairs:
        v1, v2 = emb_dict[w1], emb_dict[w2]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    t_cosine = time.perf_counter() - t0
    
    # --- Prime GCD ---
    t0 = time.perf_counter()
    for w1, w2 in pairs:
        p1, p2 = prime_map[w1], prime_map[w2]
        g = math.gcd(p1, p2)
        only_1 = p1 // g
        only_2 = p2 // g
    t_gcd = time.perf_counter() - t0
    
    speedup = t_cosine / t_gcd
    
    print(f"  Cosine: {t_cosine:.4f}s for {N_VALIDATIONS} ops")
    print(f"  GCD:    {t_gcd:.4f}s for {N_VALIDATIONS} ops")
    print(f"  Speedup: {speedup:.1f}x")
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, "experiment1_timing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "time_seconds", "n_operations", "ops_per_second"])
        w.writerow(["cosine_similarity", f"{t_cosine:.6f}", N_VALIDATIONS, f"{N_VALIDATIONS/t_cosine:.0f}"])
        w.writerow(["prime_gcd", f"{t_gcd:.6f}", N_VALIDATIONS, f"{N_VALIDATIONS/t_gcd:.0f}"])
    
    # Generate LaTeX table
    with open(os.path.join(TABLES_DIR, "timing_benchmark.tex"), "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Computational efficiency: Cosine similarity vs.\\ Prime GCD for pairwise relationship verification (%d operations).}\n" % N_VALIDATIONS)
        f.write("\\label{tab:timing}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Method & Time (s) & Ops/sec & Output Type \\\\\n\\midrule\n")
        f.write("Cosine Similarity & %.4f & %s & Probabilistic \\\\\n" % (t_cosine, f"{N_VALIDATIONS/t_cosine:,.0f}"))
        f.write("Prime GCD (ours) & %.4f & %s & Deterministic \\\\\n" % (t_gcd, f"{N_VALIDATIONS/t_gcd:,.0f}"))
        f.write("\\midrule\n")
        f.write("\\textbf{Speedup} & \\multicolumn{3}{c}{\\textbf{%.1f$\\times$}} \\\\\n" % speedup)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    
    return t_cosine, t_gcd, speedup


def experiment_2_k_sweep():
    """Experiment 2: k-Value Sweep (Resolution-Collision Tradeoff)"""
    print("\n=== Experiment 2: k-Value Sweep ===")
    
    k_values = [3, 4, 6, 8, 12, 16]
    seeds = [42, 123, 456, 789, 1010]
    
    rows = []
    
    for k in k_values:
        collision_rates = []
        avg_factors = []
        max_primes = []
        subsumption_counts = []
        
        for seed in seeds:
            mapper = DiscreteMapper(n_bits=k, seed=seed)
            pm = mapper.fit_transform(VOCAB_FULL, embeddings)
            
            primes = list(pm.values())
            unique_primes = len(set(primes))
            collision_rate = 1.0 - (unique_primes / len(primes))
            collision_rates.append(collision_rate)
            
            factors = [len(validator._prime_factors(p)) for p in primes]
            avg_factors.append(np.mean(factors))
            max_primes.append(max(primes))
            
            # Count subsumption relationships
            sub_count = 0
            for i, w1 in enumerate(VOCAB_FULL):
                for j, w2 in enumerate(VOCAB_FULL):
                    if i != j and pm[w1] % pm[w2] == 0:
                        sub_count += 1
            subsumption_counts.append(sub_count)
        
        row = {
            "k": k,
            "collision_rate_mean": np.mean(collision_rates),
            "collision_rate_std": np.std(collision_rates),
            "avg_prime_factors": np.mean(avg_factors),
            "max_prime_value": int(np.mean(max_primes)),
            "subsumption_pairs_mean": np.mean(subsumption_counts),
            "subsumption_pairs_std": np.std(subsumption_counts),
        }
        rows.append(row)
        print(f"  k={k:2d}: collision={row['collision_rate_mean']:.2%}, avg_factors={row['avg_prime_factors']:.1f}, subsumptions={row['subsumption_pairs_mean']:.0f}")
    
    # Save CSV
    with open(os.path.join(OUTPUT_DIR, "experiment2_k_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    
    # Generate LaTeX table
    with open(os.path.join(TABLES_DIR, "k_sweep.tex"), "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Effect of $k$ (number of LSH hyperplanes) on encoding properties. Averaged over 5 random seeds.}\n")
        f.write("\\label{tab:ksweep}\n")
        f.write("\\begin{tabular}{ccccc}\n\\toprule\n")
        f.write("$k$ & Collision Rate & Avg. Factors & Max Prime & Subsumption Pairs \\\\\n\\midrule\n")
        for r in rows:
            f.write("%d & %.1f\\%% & %.1f & $%.1e$ & %.0f $\\pm$ %.0f \\\\\n" % (
                r['k'], r['collision_rate_mean']*100, r['avg_prime_factors'],
                r['max_prime_value'], r['subsumption_pairs_mean'], r['subsumption_pairs_std']
            ))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    
    return rows


def experiment_3_subsumption():
    """Experiment 3: Subsumption Accuracy against known hypernym pairs"""
    print("\n=== Experiment 3: Subsumption Accuracy ===")
    
    k_values = [3, 4, 6, 8, 12, 16]
    seeds = list(range(10))  # 10 seeds for statistical significance
    
    rows = []
    
    for k in k_values:
        true_positives_all = []
        false_positives_all = []
        
        for seed in seeds:
            mapper = DiscreteMapper(n_bits=k, seed=seed)
            pm = mapper.fit_transform(VOCAB_FULL, embeddings)
            
            tp = 0
            tested = 0
            for broader, narrower in HYPERNYM_PAIRS:
                if broader in pm and narrower in pm:
                    tested += 1
                    if pm[broader] % pm[narrower] == 0:
                        tp += 1
            
            true_positives_all.append(tp / tested if tested > 0 else 0)
            
            # False positive: random non-hypernym pairs that accidentally subsume
            fp = 0
            random_tested = 0
            for _ in range(100):
                w1, w2 = random.choice(VOCAB_FULL), random.choice(VOCAB_FULL)
                if w1 != w2:
                    random_tested += 1
                    if pm[w1] % pm[w2] == 0:
                        fp += 1
            
            false_positives_all.append(fp / random_tested if random_tested > 0 else 0)
        
        row = {
            "k": k,
            "true_positive_rate": np.mean(true_positives_all),
            "tp_std": np.std(true_positives_all),
            "false_positive_rate": np.mean(false_positives_all),
            "fp_std": np.std(false_positives_all),
        }
        rows.append(row)
        print(f"  k={k:2d}: TP rate={row['true_positive_rate']:.2%} ± {row['tp_std']:.2%}, FP rate={row['false_positive_rate']:.2%} ± {row['fp_std']:.2%}")
    
    # Save CSV
    with open(os.path.join(OUTPUT_DIR, "experiment3_subsumption.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    
    # Generate LaTeX table
    with open(os.path.join(TABLES_DIR, "subsumption_accuracy.tex"), "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Subsumption accuracy: True positive rate (known hypernym pairs) vs.\\ false positive rate (random pairs). Averaged over 10 seeds.}\n")
        f.write("\\label{tab:subsumption}\n")
        f.write("\\begin{tabular}{ccc}\n\\toprule\n")
        f.write("$k$ & True Positive Rate & False Positive Rate \\\\\n\\midrule\n")
        for r in rows:
            f.write("%d & %.1f\\%% $\\pm$ %.1f\\%% & %.1f\\%% $\\pm$ %.1f\\%% \\\\\n" % (
                r['k'], r['true_positive_rate']*100, r['tp_std']*100,
                r['false_positive_rate']*100, r['fp_std']*100
            ))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    
    return rows


def experiment_4_composition():
    """Experiment 4: Composition Guarantee Verification"""
    print("\n=== Experiment 4: Composition Guarantee ===")
    
    mapper = DiscreteMapper(n_bits=8, seed=42)
    pm = mapper.fit_transform(VOCAB_FULL, embeddings)
    
    total_pairs = 0
    guarantee_holds = 0
    
    words = list(pm.keys())
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            w1, w2 = words[i], words[j]
            composed = validator.compose(pm[w1], pm[w2])
            
            sub_a = composed % pm[w1] == 0
            sub_b = composed % pm[w2] == 0
            
            total_pairs += 1
            if sub_a and sub_b:
                guarantee_holds += 1
    
    pct = guarantee_holds / total_pairs * 100
    print(f"  Tested {total_pairs} pairs. Guarantee holds: {guarantee_holds}/{total_pairs} ({pct:.1f}%)")
    
    with open(os.path.join(OUTPUT_DIR, "experiment4_composition.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_pairs", "guarantee_holds", "percentage"])
        w.writerow([total_pairs, guarantee_holds, f"{pct:.2f}"])
    
    return total_pairs, guarantee_holds


def experiment_5_analogy():
    """Experiment 5: Analogy Resolution — A:B::C:?"""
    print("\n=== Experiment 5: Analogy Resolution ===")
    
    # Known analogies: (A, B, C, expected_D)
    analogies = [
        ("King", "Man", "Queen", "Woman"),
        ("King", "Queen", "Man", "Woman"),
        ("Father", "Mother", "Boy", "Girl"),
        ("Dog", "Cat", "Lion", "Tiger"),
        ("Car", "Truck", "Bicycle", "Motorcycle"),
    ]
    
    k_values = [3, 6, 8, 12]
    seeds = list(range(10))
    
    rows = []
    for k in k_values:
        correct_total = 0
        tested_total = 0
        
        for seed in seeds:
            mapper = DiscreteMapper(n_bits=k, seed=seed)
            pm = mapper.fit_transform(VOCAB_FULL, embeddings)
            
            for a, b, c, expected_d in analogies:
                if all(w in pm for w in [a, b, c, expected_d]):
                    tested_total += 1
                    # Find D such that GCD distance is minimized
                    target = pm[b] * pm[c]
                    best_word = None
                    best_dist = float('inf')
                    
                    for w, p_x in pm.items():
                        if w in [a, b, c]:
                            continue
                        left = p_x * pm[a]
                        g = math.gcd(left, target)
                        missing = target // g
                        extra = left // g
                        dist = abs(extra - missing) + (extra * missing) if not (extra == 1 and missing == 1) else 0
                        if dist < best_dist:
                            best_dist = dist
                            best_word = w
                    
                    if best_word == expected_d:
                        correct_total += 1
        
        accuracy = correct_total / tested_total * 100 if tested_total > 0 else 0
        row = {"k": k, "correct": correct_total, "total": tested_total, "accuracy": accuracy}
        rows.append(row)
        print(f"  k={k:2d}: {correct_total}/{tested_total} correct ({accuracy:.1f}%)")
    
    with open(os.path.join(OUTPUT_DIR, "experiment5_analogy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    
    with open(os.path.join(TABLES_DIR, "analogy_accuracy.tex"), "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Analogy resolution accuracy ($A:B::C:?$) across $k$ values. 5 analogy pairs tested over 10 random seeds.}\n")
        f.write("\\label{tab:analogy}\n")
        f.write("\\begin{tabular}{ccc}\n\\toprule\n")
        f.write("$k$ & Correct / Total & Accuracy \\\\\n\\midrule\n")
        for r in rows:
            f.write("%d & %d / %d & %.1f\\%% \\\\\n" % (r['k'], r['correct'], r['total'], r['accuracy']))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    
    return rows


def experiment_6_gap_examples():
    """Experiment 6: Worked Gap Analysis Examples for the paper"""
    print("\n=== Experiment 6: Gap Analysis Worked Examples ===")
    
    mapper = DiscreteMapper(n_bits=8, seed=42)
    pm = mapper.fit_transform(VOCAB_FULL, embeddings)
    
    pairs = [
        ("King", "Queen"), ("King", "Man"), ("Man", "Woman"),
        ("Dog", "Cat"), ("Car", "Bicycle"), ("Love", "Hate"),
    ]
    
    rows = []
    for w1, w2 in pairs:
        if w1 in pm and w2 in pm:
            p1, p2 = pm[w1], pm[w2]
            g = math.gcd(p1, p2)
            only1 = p1 // g
            only2 = p2 // g
            f1 = validator._prime_factors(p1)
            f2 = validator._prime_factors(p2)
            fg = validator._prime_factors(g)
            fo1 = validator._prime_factors(only1)
            fo2 = validator._prime_factors(only2)
            sub_ab = p1 % p2 == 0
            sub_ba = p2 % p1 == 0
            
            row = {
                "word_a": w1, "word_b": w2,
                "prime_a": p1, "prime_b": p2,
                "factors_a": str(f1), "factors_b": str(f2),
                "shared_gcd": g, "shared_factors": str(fg),
                "only_a": only1, "only_a_factors": str(fo1),
                "only_b": only2, "only_b_factors": str(fo2),
                "a_subsumes_b": sub_ab, "b_subsumes_a": sub_ba,
            }
            rows.append(row)
            print(f"  {w1}({p1}) vs {w2}({p2}): shared={g}{fg}, only_{w1}={only1}{fo1}, only_{w2}={only2}{fo2}, sub={sub_ab}/{sub_ba}")
    
    with open(os.path.join(OUTPUT_DIR, "experiment6_gap_examples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    
    with open(os.path.join(TABLES_DIR, "gap_examples.tex"), "w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{Worked examples of abductive gap analysis. Each row decomposes the relationship between two concepts into shared backbone (GCD), unique factors, and subsumption status. $k=8$, seed $=42$.}\n")
        f.write("\\label{tab:gap}\n")
        f.write("\\begin{tabular}{llrrlrrlcc}\n\\toprule\n")
        f.write("Word A & Word B & $\\Phi(A)$ & $\\Phi(B)$ & Shared (GCD) & Only A & Only B & $A \\supseteq B$ & $B \\supseteq A$ \\\\\n\\midrule\n")
        for r in rows:
            f.write("%s & %s & %s & %s & %s %s & %s %s & %s %s & %s & %s \\\\\n" % (
                r['word_a'], r['word_b'],
                r['prime_a'], r['prime_b'],
                r['shared_gcd'], r['shared_factors'],
                r['only_a'], r['only_a_factors'],
                r['only_b'], r['only_b_factors'],
                "\\checkmark" if r['a_subsumes_b'] else "$\\times$",
                "\\checkmark" if r['b_subsumes_a'] else "$\\times$",
            ))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    
    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("Running Paper Experiment Suite")
    print("=" * 60)
    
    t_cos, t_gcd, speedup = experiment_1_timing()
    k_results = experiment_2_k_sweep()
    sub_results = experiment_3_subsumption()
    total, holds = experiment_4_composition()
    analogy_results = experiment_5_analogy()
    gap_results = experiment_6_gap_examples()
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"  Data saved to: {OUTPUT_DIR}/")
    print(f"  Tables saved to: {TABLES_DIR}/")
    print("=" * 60)
