import networkx as nx
import numpy as np
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

class GraphRegularizer:
    """
    Entropic Graph Pruning (formerly Unified Holographic Resonance Theory - UHRT).
    A heuristic regularizer for network topologies (Knowledge Graphs, Neural Networks).
    Monitors arithmetic/structural stability and performs entropic pruning if a 'glitch' 
    (instability or contradiction) is detected.
    """
    
    @staticmethod
    def calculate_entropy(graph: nx.Graph) -> float:
        """
        Calculates the Shannon Entropy of the graph based on its degree distribution.
        """
        degrees = np.array([d for n, d in graph.degree()])
        if len(degrees) == 0 or degrees.sum() == 0:
            return 0.0
        
        probabilities = degrees / degrees.sum()
        return float(entropy(probabilities, base=2))

    @staticmethod
    def prune_graph(graph: nx.Graph, fraction: float) -> nx.Graph:
        """
        Prunes the graph by removing a specified fraction of the lowest-weight edges.
        Assumes edges have a 'weight' attribute. If not, assumes weight=1.
        """
        g_copy = graph.copy()
        if fraction <= 0 or g_copy.number_of_edges() == 0:
            return g_copy
            
        # Sort edges by weight (ascending)
        edges = sorted(g_copy.edges(data=True), key=lambda e: e[2].get('weight', 1.0))
        num_to_remove = int(len(edges) * fraction)
        
        edges_to_remove = edges[:num_to_remove]
        g_copy.remove_edges_from([(u, v) for u, v, _ in edges_to_remove])
        return g_copy

    def optimize_entropy(self, graph: nx.Graph, target_entropy: float, 
                         max_prune_fraction: float = 0.5, eps: float = 0.01) -> tuple[nx.Graph, float]:
        """
        Uses ternary search to find the optimal pruning fraction that brings the graph's 
        entropy closest to the target_entropy.
        
        This mimics the 'Glitch Resolution' process, forcing the network to drop weak 
        connections to regain information-theoretic stability.
        
        Returns:
            (Pruned Graph, Final Pruning Fraction)
        """
        low, high = 0.0, max_prune_fraction
        
        if graph.number_of_edges() == 0:
            return graph, 0.0

        logger.info(f"Optimizing graph towards target entropy: {target_entropy}")
        
        while high - low > eps:
            delta = high - low
            mid1 = low + delta / 3
            mid2 = high - delta / 3

            g1 = self.prune_graph(graph, mid1)
            g2 = self.prune_graph(graph, mid2)
            
            ent1 = self.calculate_entropy(g1)
            ent2 = self.calculate_entropy(g2)
            
            # We want to minimize the difference to the target entropy
            if abs(ent1 - target_entropy) < abs(ent2 - target_entropy):
                high = mid2
            else:
                low = mid1
                
        final_fraction = (low + high) / 2
        pruned_graph = self.prune_graph(graph, final_fraction)
        
        logger.info(f"Optimization complete. Pruned {final_fraction:.1%} of weakest edges.")
        return pruned_graph, final_fraction
