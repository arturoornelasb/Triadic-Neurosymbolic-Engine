import numpy as np
import networkx as nx
from neurosym._archived.buss import BipolarExtractor
from neurosym.triadic import DiscreteValidator
from neurosym._archived.uhrt import GraphRegularizer
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_real_world_validation():
    print("=== Neurosymbolic Engine: Real-World Pragmative Validation ===")
    
    # 1. LATENT EXTRACTION (BUSS)
    # We simulate a small real-world corpus
    print("\n--- Phase 1: Latent Extraction ---")
    corpus = [
        "The king is a powerful man who rules the kingdom.",
        "The queen is a powerful woman who rules the kingdom.",
        "A man is an adult male human.",
        "A woman is an adult female human.",
        "Water is a liquid necessary for life.",
        "Fire is hot and burns."
    ]
    
    extractor = BipolarExtractor(n_components=2)
    axes = extractor.fit(corpus)
    print(f"Extracted {axes.shape[0]} latent semantic axes with length {axes.shape[1]}.")
    
    # Let's project some words to get their "continuous" values
    # In a real scenario, these would map to large integers. 
    # For this test, we'll assign synthetic discrete values to represent the output of the discretizer
    
    print("\n--- Phase 2: Discrete Algebraic Validation ---")
    validator = DiscreteValidator()
    
    # Let's test the abductive discovery on a noisy real-world semantic relation.
    # Suppose our NLP pipeline mapped concepts to these integers (factors):
    C_King = 21   # 7 * 3 (Royalty * Male)
    C_Man = 3     # 3 (Male)
    C_Woman = 5   # 5 (Female)
    
    print(f"Validating Analogy: King ({C_King}) : Man ({C_Man}) :: Queen (?) : Woman ({C_Woman})")
    
    # D = (C * B) / A -> Queen = (Woman * King) / Man
    prediction_result = validator.analogy_prediction(
        source_a=C_Man, 
        source_b=C_King, 
        target_a=C_Woman
    )
    
    if prediction_result.is_valid:
        print(f"Success! The predicted integer for Queen is: {prediction_result.output_value}")
    else:
        print(f"Obstruction. {prediction_result.trace}")
        
        
    print("\n--- Phase 3: Entropic Graph Regularization (UHRT) ---")
    # Let's build a noisy Knowledge Graph representing scraped relations
    kg = nx.Graph()
    kg.add_edge("King", "Man", weight=0.9)
    kg.add_edge("Queen", "Woman", weight=0.85)
    kg.add_edge("King", "Kingdom", weight=0.95)
    kg.add_edge("Queen", "Kingdom", weight=0.92)
    
    # Add some noise (glitches)
    kg.add_edge("King", "Fire", weight=0.1) 
    kg.add_edge("Man", "Water", weight=0.2)
    
    regularizer = GraphRegularizer()
    initial_entropy = regularizer.calculate_entropy(kg)
    print(f"Initial Graph Entropy: {initial_entropy:.4f}")
    
    # We want a highly organized graph (lower entropy)
    target = 1.0 
    pruned_kg, pruning_frac = regularizer.optimize_entropy(kg, target_entropy=target)
    
    final_entropy = regularizer.calculate_entropy(pruned_kg)
    print(f"Final Graph Entropy: {final_entropy:.4f}")
    print(f"Edges pruned: {kg.number_of_edges() - pruned_kg.number_of_edges()}")
    print("Remaining stable relations:", pruned_kg.edges())

if __name__ == "__main__":
    run_real_world_validation()
