"""
Held-out Subsumption Evaluation for the Prime Factorization Paper.

Tests hypernym detection (subsumption via GCD divisibility) on a larger,
held-out set of word pairs from common English taxonomies.

Reports: Precision, Recall, F1, FP rate, coverage.
"""
import sys
import os
import numpy as np
from collections import defaultdict

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Triadic-Neurosymbolic-Engine", "src")
sys.path.insert(0, ENGINE_PATH)

from neurosym.encoder import ContinuousEncoder, DiscreteMapper

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Training set (from paper's Experiment 3) ---
TRAIN_PAIRS = [
    ("Animal", "Dog"), ("Animal", "Cat"), ("Animal", "Horse"), ("Animal", "Bird"),
    ("Vehicle", "Car"), ("Vehicle", "Truck"), ("Vehicle", "Bus"), ("Vehicle", "Train"),
    ("Fruit", "Apple"), ("Fruit", "Banana"), ("Fruit", "Orange"),
]

# --- Held-out hypernym pairs (NOT used in training/tuning) ---
HELD_OUT_HYPERNYMS = [
    # Animals (new hyponyms)
    ("Animal", "Fish"), ("Animal", "Lion"), ("Animal", "Tiger"), ("Animal", "Eagle"),
    ("Animal", "Cow"), ("Animal", "Pig"), ("Animal", "Sheep"), ("Animal", "Deer"),
    # Vehicles (new)
    ("Vehicle", "Bicycle"), ("Vehicle", "Motorcycle"), ("Vehicle", "Airplane"), ("Vehicle", "Ship"),
    # Fruit (new)
    ("Fruit", "Grape"), ("Fruit", "Peach"), ("Fruit", "Cherry"), ("Fruit", "Mango"),
    # New categories entirely
    ("Color", "Red"), ("Color", "Blue"), ("Color", "Green"), ("Color", "Yellow"),
    ("Color", "Black"), ("Color", "White"), ("Color", "Purple"),
    ("Profession", "Doctor"), ("Profession", "Teacher"), ("Profession", "Engineer"),
    ("Profession", "Artist"), ("Profession", "Farmer"), ("Profession", "Chef"),
    ("Emotion", "Love"), ("Emotion", "Hate"), ("Emotion", "Fear"),
    ("Emotion", "Joy"), ("Emotion", "Anger"), ("Emotion", "Hope"),
    ("Body", "Hand"), ("Body", "Eye"), ("Body", "Heart"),
    ("Body", "Head"), ("Body", "Foot"), ("Body", "Brain"),
    ("Weapon", "Sword"), ("Weapon", "Gun"), ("Weapon", "Knife"),
    ("Tool", "Hammer"), ("Tool", "Wrench"), ("Tool", "Saw"),
    ("Furniture", "Table"), ("Furniture", "Chair"), ("Furniture", "Bed"),
    ("Furniture", "Desk"), ("Furniture", "Sofa"),
    ("Instrument", "Piano"), ("Instrument", "Guitar"), ("Instrument", "Violin"),
    ("Instrument", "Drum"), ("Instrument", "Flute"),
    ("Clothing", "Shirt"), ("Clothing", "Pants"), ("Clothing", "Hat"),
    ("Clothing", "Shoes"), ("Clothing", "Jacket"),
    ("Sport", "Football"), ("Sport", "Basketball"), ("Sport", "Tennis"),
    ("Sport", "Swimming"), ("Sport", "Boxing"),
]

# --- Unrelated pairs (no hypernym relation) ---
UNRELATED_PAIRS = [
    ("Dog", "Table"), ("Cat", "Bread"), ("King", "River"), ("Car", "Apple"),
    ("Red", "Doctor"), ("Love", "Bicycle"), ("Tree", "Hammer"), ("Bird", "Chair"),
    ("Lion", "Piano"), ("Fear", "Ship"), ("Hand", "Yellow"), ("Chef", "Tiger"),
    ("Eye", "Truck"), ("Hope", "Grape"), ("Brain", "Sword"), ("Cow", "Guitar"),
    ("Fish", "Hat"), ("Pig", "Violin"), ("Deer", "Pants"), ("Eagle", "Bed"),
    ("Horse", "Drum"), ("Sheep", "Shoes"), ("Bus", "Mango"), ("Train", "Saw"),
    ("Book", "Lion"), ("Pen", "Cow"), ("Door", "Peach"), ("Window", "Football"),
    ("Mountain", "Shirt"), ("River", "Boxing"), ("Ocean", "Knife"), ("Forest", "Flute"),
]


def evaluate(encoder, k_values=(8, 12, 16)):
    """Evaluate subsumption on held-out pairs at multiple k values."""
    # Collect all unique words
    all_words = set()
    for h, c in HELD_OUT_HYPERNYMS + UNRELATED_PAIRS + TRAIN_PAIRS:
        all_words.add(h)
        all_words.add(c)
    all_words = sorted(all_words)

    print(f"  Total unique words: {len(all_words)}")
    print(f"  Held-out hypernym pairs: {len(HELD_OUT_HYPERNYMS)}")
    print(f"  Unrelated pairs: {len(UNRELATED_PAIRS)}")
    print()

    # Encode all
    embeddings = encoder.encode(all_words)

    results = {}
    for k in k_values:
        print(f"  --- k = {k} ---")
        # Test multiple projection modes
        for proj in ("random", "pca", "contrastive"):
            print(f"    [projection={proj}]")
            mapper = DiscreteMapper(
                n_bits=k,
                projection=proj,
                hypernym_pairs=TRAIN_PAIRS if proj == "contrastive" else None,
            )
            prime_dict = mapper.fit_transform(all_words, embeddings)

            # Evaluate hypernym pairs
            tp, fp, fn, tn = 0, 0, 0, 0

            # True positives: hypernym subsumes hyponym
            for hyper, hypo in HELD_OUT_HYPERNYMS:
                p_hyper = prime_dict[hyper]
                p_hypo = prime_dict[hypo]
                if p_hypo % p_hyper == 0:  # hyper divides hypo = subsumption
                    tp += 1
                else:
                    fn += 1

            # False positives: unrelated pairs that accidentally subsume
            for w1, w2 in UNRELATED_PAIRS:
                p1 = prime_dict[w1]
                p2 = prime_dict[w2]
                if p2 % p1 == 0 or p1 % p2 == 0:  # either direction
                    fp += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            # Theoretical FP rate: (3/4)^k for random bit overlap
            fp_theoretical = (3/4) ** k

            print(f"      TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            print(f"      Precision: {precision:.3f}")
            print(f"      Recall:    {recall:.3f}")
            print(f"      F1:        {f1:.3f}")
            print(f"      FPR:       {fpr:.3f} (theoretical: {fp_theoretical:.4f})")
            print(f"      FP ratio (obs/theo): {fpr/fp_theoretical:.2f}" if fp_theoretical > 0 else "")

            # Training pairs sanity check
            train_tp = sum(1 for h, c in TRAIN_PAIRS if prime_dict[c] % prime_dict[h] == 0)
            print(f"      Training recall: {train_tp}/{len(TRAIN_PAIRS)} ({train_tp/len(TRAIN_PAIRS):.1%})")
            print()

            results[f"k{k}_{proj}"] = {
                'k': k, 'projection': proj,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision, 'recall': recall, 'f1': f1,
                'fpr': fpr, 'fp_theoretical': fp_theoretical,
                'train_recall': train_tp / len(TRAIN_PAIRS),
            }

    return results


def main():
    print("=" * 70)
    print("  HELD-OUT SUBSUMPTION EVALUATION")
    print("  Tests hypernym detection on pairs NOT used in development")
    print("=" * 70)
    print()

    encoder = ContinuousEncoder()
    results = evaluate(encoder, k_values=(8, 12, 16))

    # Save results
    import json
    out_path = os.path.join(OUTPUT_DIR, 'held_out_subsumption.json')
    # Convert numpy types
    clean = {}
    for k, v in results.items():
        clean[str(k)] = {kk: (float(vv) if isinstance(vv, (int, float, np.integer, np.floating)) else str(vv))
                         for kk, vv in v.items()}
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
