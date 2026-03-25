"""
Scalable Graph Builder for the Triadic Neurosymbolic Engine.

Replaces the naive O(N²) all-pairs GCD scan with an inverted prime index
approach that only compares concepts sharing at least one prime factor.

Complexity: O(N * F * B) where:
    N = number of concepts
    F = average number of prime factors per concept  
    B = average bucket size per prime factor
    
For sparse LSH projections (k=8-16), F ≈ 4-8 and B << N, so this is
dramatically faster than O(N²) for large datasets.
"""
import logging
import sympy
from typing import Dict, List, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class ScalableGraphBuilder:
    """
    Builds semantic graphs using an inverted prime factor index
    instead of brute-force all-pairs comparison.
    """
    
    def __init__(self):
        self.inverted_index: Dict[int, Set[str]] = defaultdict(set)
        self.concept_factors: Dict[str, List[int]] = {}
    
    def _factorize(self, n: int) -> List[int]:
        if n <= 1:
            return []
        return list(sympy.factorint(n).keys())
    
    def build_index(self, prime_map: Dict[str, int]):
        """
        Build the inverted index from a prime map.
        Maps each prime factor → set of concepts containing that factor.
        """
        self.inverted_index.clear()
        self.concept_factors.clear()
        
        for concept, composite in prime_map.items():
            factors = self._factorize(composite)
            self.concept_factors[concept] = factors
            for f in factors:
                self.inverted_index[f].add(concept)
        
        logger.info(
            f"Built inverted index: {len(prime_map)} concepts, "
            f"{len(self.inverted_index)} prime buckets"
        )
    
    def find_edges(self, prime_map: Dict[str, int], min_shared: int = 1) -> List[Tuple[str, str, int, List[int]]]:
        """
        Find all edges (concept pairs with shared prime factors) using
        the inverted index. Returns list of (concept_a, concept_b, weight, shared_factors).
        
        This is O(N * F * B) instead of O(N²).
        """
        if not self.inverted_index:
            self.build_index(prime_map)
        
        # Collect candidate pairs from the inverted index
        pair_shared: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        
        for prime_factor, concepts in self.inverted_index.items():
            concept_list = sorted(concepts)  # Sort for consistent pair ordering
            for i in range(len(concept_list)):
                for j in range(i + 1, len(concept_list)):
                    pair = (concept_list[i], concept_list[j])
                    pair_shared[pair].add(prime_factor)
        
        # Filter by minimum shared weight
        edges = []
        for (a, b), shared_primes in pair_shared.items():
            weight = len(shared_primes)
            if weight >= min_shared:
                edges.append((a, b, weight, sorted(shared_primes)))
        
        logger.info(
            f"Found {len(edges)} edges (min_shared={min_shared}) "
            f"from {len(pair_shared)} candidate pairs"
        )
        return edges
    
    def find_neighbors(self, concept: str, prime_map: Dict[str, int], min_shared: int = 1) -> List[Tuple[str, int, List[int]]]:
        """
        Find all neighbors of a single concept using the inverted index.
        Returns list of (neighbor, weight, shared_factors).
        
        This is O(F * B) — constant relative to total dataset size.
        """
        if not self.inverted_index:
            self.build_index(prime_map)
        
        if concept not in self.concept_factors:
            return []
        
        neighbor_shared: Dict[str, Set[int]] = defaultdict(set)
        
        for factor in self.concept_factors[concept]:
            for other in self.inverted_index[factor]:
                if other != concept:
                    neighbor_shared[other].add(factor)
        
        results = []
        for neighbor, shared_primes in neighbor_shared.items():
            weight = len(shared_primes)
            if weight >= min_shared:
                results.append((neighbor, weight, sorted(shared_primes)))
        
        results.sort(key=lambda x: -x[1])
        return results
    
    def get_stats(self) -> dict:
        """Return index statistics."""
        if not self.inverted_index:
            return {"indexed": False}
        
        bucket_sizes = [len(v) for v in self.inverted_index.values()]
        return {
            "indexed": True,
            "total_concepts": len(self.concept_factors),
            "total_prime_buckets": len(self.inverted_index),
            "avg_bucket_size": sum(bucket_sizes) / max(1, len(bucket_sizes)),
            "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
            "avg_factors_per_concept": sum(len(f) for f in self.concept_factors.values()) / max(1, len(self.concept_factors)),
        }
