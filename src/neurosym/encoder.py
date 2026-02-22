import numpy as np
from typing import List, Dict
import sympy
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ContinuousEncoder:
    """
    A lightweight wrapper around sentence-transformers to generate 
    continuous vector embeddings for natural language concepts.
    Optimized for CPU usage.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def encode(self, concepts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of concepts.
        """
        logger.info(f"Encoding {len(concepts)} concepts...")
        embeddings = self.model.encode(concepts, convert_to_numpy=True)
        return embeddings

class DiscreteMapper:
    """
    Maps continuous dense vectors to discrete integer space 
    using Locality Sensitive Hashing (LSH) and Prime Factorization.
    
    Each LSH hyperplane is assigned a unique prime number. A concept's
    discrete representation is the product of all primes corresponding
    to hyperplanes where its projection is positive.
    """
    def __init__(self, n_bits: int = 16, seed: int = 42):
        self.n_bits = n_bits
        self.random_state = np.random.RandomState(seed)
        self.random_planes = None
        self.concept_to_prime: Dict[str, int] = {}
        
    def _generate_random_planes(self, dim: int):
        self.random_planes = self.random_state.randn(self.n_bits, dim)

    def fit_transform(self, concepts: List[str], embeddings: np.ndarray) -> Dict[str, int]:
        """
        Maps a list of concepts and their continuous embeddings to discrete composite primes.
        Each LSH hyperplane corresponds to a unique prime number.
        """
        logger.info(f"Mapping {len(concepts)} concepts to discrete integer space...")
        
        if self.random_planes is None:
            self._generate_random_planes(embeddings.shape[1])
            
        # Assign a prime to each hyperplane (Semantic Feature)
        plane_primes = [sympy.prime(i + 1) for i in range(self.n_bits)]
        
        for concept, emb in zip(concepts, embeddings):
            projections = np.dot(self.random_planes, emb)
            bits = (projections > 0).astype(int)
            
            # The discrete integer is the product of its active semantic feature primes
            composite_integer = 1
            for bit, prime_factor in zip(bits, plane_primes):
                if bit == 1:
                    composite_integer *= prime_factor
            
            # Fallback for origin vector
            if composite_integer == 1:
                composite_integer = 2 # minimum prime
                
            self.concept_to_prime[concept] = composite_integer
            
        return self.concept_to_prime
    
    def get_factor(self, concept: str) -> int:
        if concept not in self.concept_to_prime:
            raise ValueError(f"Concept '{concept}' not found in the discrete mapping.")
        return self.concept_to_prime[concept]
