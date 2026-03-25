import numpy as np
from typing import List, Dict, Optional, Tuple
import sympy
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================
# Abstract Encoder Interface (Pluggable Backend)
# ============================================================

class BaseEncoder(ABC):
    """
    Abstract interface for all embedding backends.
    Subclass this to add new providers (OpenAI, Cohere, etc.)
    """
    
    @abstractmethod
    def encode(self, concepts: List[str]) -> np.ndarray:
        """Returns a 2D numpy array of shape (len(concepts), dim)."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier."""
        ...


# ============================================================
# Local HuggingFace / Sentence-Transformers Backend
# ============================================================

class ContinuousEncoder(BaseEncoder):
    """
    A lightweight wrapper around sentence-transformers to generate 
    continuous vector embeddings for natural language concepts.
    Optimized for CPU usage. Runs entirely offline.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading local embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._name = model_name
        
    def encode(self, concepts: List[str]) -> np.ndarray:
        logger.info(f"Encoding {len(concepts)} concepts with {self._name}...")
        return self.model.encode(concepts, convert_to_numpy=True)
    
    @property
    def name(self) -> str:
        return self._name


# ============================================================
# OpenAI Embedding Backend (Requires API Key)
# ============================================================

class OpenAIEncoder(BaseEncoder):
    """
    Embedding backend using OpenAI's text-embedding API.
    Requires: pip install openai
    Set OPENAI_API_KEY environment variable.
    """
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self._name = model_name
        import os
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("OpenAI backend requires: pip install openai")
        
    def encode(self, concepts: List[str]) -> np.ndarray:
        logger.info(f"Encoding {len(concepts)} concepts with OpenAI/{self._name}...")
        response = self.client.embeddings.create(input=concepts, model=self._name)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def name(self) -> str:
        return f"openai/{self._name}"


# ============================================================
# Cohere Embedding Backend (Requires API Key)
# ============================================================

class CohereEncoder(BaseEncoder):
    """
    Embedding backend using Cohere's embed API.
    Requires: pip install cohere
    Set COHERE_API_KEY environment variable.
    """
    def __init__(self, model_name: str = "embed-english-v3.0"):
        self._name = model_name
        import os
        try:
            import cohere
            self.client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
        except ImportError:
            raise ImportError("Cohere backend requires: pip install cohere")
    
    def encode(self, concepts: List[str]) -> np.ndarray:
        logger.info(f"Encoding {len(concepts)} concepts with Cohere/{self._name}...")
        response = self.client.embed(
            texts=concepts,
            model=self._name,
            input_type="search_document"
        )
        return np.array(response.embeddings, dtype=np.float32)
    
    @property
    def name(self) -> str:
        return f"cohere/{self._name}"


# ============================================================
# Encoder Factory
# ============================================================

ENCODER_REGISTRY = {
    "local": ContinuousEncoder,
    "openai": OpenAIEncoder,
    "cohere": CohereEncoder,
}

def create_encoder(provider: str = "local", model: str = None) -> BaseEncoder:
    """
    Factory function to create an encoder from a provider name.
    
    Examples:
        create_encoder("local", "all-MiniLM-L6-v2")
        create_encoder("openai", "text-embedding-3-large")
        create_encoder("cohere", "embed-english-v3.0")
    """
    if provider not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(ENCODER_REGISTRY.keys())}")
    
    cls = ENCODER_REGISTRY[provider]
    if model:
        return cls(model_name=model)
    return cls()


# ============================================================
# Discrete Mapper (LSH → Prime Factorization)
# ============================================================

class DiscreteMapper:
    """
    Maps continuous dense vectors to discrete integer space 
    using Locality Sensitive Hashing (LSH) and Prime Factorization.
    
    Each LSH hyperplane is assigned a unique prime number. A concept's
    discrete representation is the product of all primes corresponding
    to hyperplanes where its projection is positive.
    
    Supports four projection modes:
        - 'random' (default): Random hyperplanes from N(0, I_d)
        - 'pca': Principal Component directions from the corpus
        - 'consensus': Multi-seed voting — keeps only stable primes
        - 'contrastive': Trains hyperplanes on known hypernym pairs
    """
    def __init__(self, n_bits: int = 16, seed: int = 42, projection: str = "random",
                 consensus_seeds: int = 20, consensus_threshold: float = 0.7,
                 hypernym_pairs: Optional[List[Tuple[str, str]]] = None):
        self.n_bits = n_bits
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.projection = projection
        self.planes = None
        self.concept_to_prime: Dict[str, int] = {}
        # Consensus parameters
        self.consensus_seeds = consensus_seeds
        self.consensus_threshold = consensus_threshold
        # Contrastive parameters
        self.hypernym_pairs = hypernym_pairs or []
        
    def _generate_random_planes(self, dim: int):
        """Generate random hyperplanes from N(0, I_d)."""
        self.planes = self.random_state.randn(self.n_bits, dim)
    
    def _generate_pca_planes(self, embeddings: np.ndarray):
        """
        Generate hyperplanes from the top-k principal components.
        Deterministic, seed-independent, corpus-adapted.
        """
        from sklearn.decomposition import PCA
        
        k = min(self.n_bits, embeddings.shape[1], embeddings.shape[0])
        pca = PCA(n_components=k)
        pca.fit(embeddings)
        self.planes = pca.components_[:self.n_bits]
        
        if self.planes.shape[0] < self.n_bits:
            extra = self.random_state.randn(
                self.n_bits - self.planes.shape[0], embeddings.shape[1]
            )
            self.planes = np.vstack([self.planes, extra])
        
        logger.info(
            f"PCA projection: {k} components explain "
            f"{sum(pca.explained_variance_ratio_[:k])*100:.1f}% of variance"
        )

    def _generate_consensus_encoding(self, concepts: List[str], embeddings: np.ndarray) -> Dict[str, int]:
        """
        Multi-seed consensus encoding.
        
        Runs N random projections and, for each concept, keeps only the
        prime factors that appear in more than `threshold` fraction of runs.
        This filters out random projection noise and retains only stable
        semantic features.
        """
        from collections import Counter
        
        plane_primes = [sympy.prime(i + 1) for i in range(self.n_bits)]
        concept_prime_votes: Dict[str, Counter] = {c: Counter() for c in concepts}
        
        for seed in range(self.consensus_seeds):
            rng = np.random.RandomState(seed)
            planes = rng.randn(self.n_bits, embeddings.shape[1])
            
            for concept, emb in zip(concepts, embeddings):
                projections = np.dot(planes, emb)
                bits = (projections > 0).astype(int)
                for bit, prime in zip(bits, plane_primes):
                    if bit == 1:
                        concept_prime_votes[concept][prime] += 1
        
        # Keep only primes that appear in > threshold fraction of seeds
        min_votes = int(self.consensus_seeds * self.consensus_threshold)
        result = {}
        for concept in concepts:
            composite = 1
            for prime, votes in concept_prime_votes[concept].items():
                if votes >= min_votes:
                    composite *= prime
            if composite == 1:
                composite = 2
            result[concept] = composite
        
        stable_count = sum(
            1 for c in concepts 
            if len([v for v in concept_prime_votes[c].values() if v >= min_votes]) > 0
        )
        logger.info(
            f"Consensus encoding: {self.consensus_seeds} seeds, "
            f"threshold={self.consensus_threshold}, "
            f"{stable_count}/{len(concepts)} concepts have stable factors"
        )
        return result

    def _generate_contrastive_planes(self, embeddings: np.ndarray, concepts: List[str]):
        """
        Contrastive hyperplane learning.
        
        Given known hypernym pairs (broader ⊇ narrower), optimize hyperplane
        directions to maximize the number of correct subsumption relationships
        (broader % narrower == 0).
        
        Uses gradient-free optimization: start with PCA directions, then
        iteratively perturb each hyperplane and keep changes that improve
        subsumption accuracy on the training pairs.
        """
        if not self.hypernym_pairs:
            logger.warning("No hypernym pairs provided for contrastive learning, falling back to PCA")
            self._generate_pca_planes(embeddings)
            return
        
        concept_idx = {c: i for i, c in enumerate(concepts)}
        valid_pairs = [
            (broader, narrower) 
            for broader, narrower in self.hypernym_pairs
            if broader in concept_idx and narrower in concept_idx
        ]
        
        if not valid_pairs:
            logger.warning("No valid hypernym pairs found in vocabulary, falling back to PCA")
            self._generate_pca_planes(embeddings)
            return
        
        # Start from PCA directions
        self._generate_pca_planes(embeddings)
        best_planes = self.planes.copy()
        best_score = self._score_subsumption(best_planes, embeddings, concepts, valid_pairs)
        
        logger.info(f"Contrastive learning: {len(valid_pairs)} hypernym pairs, initial score={best_score}")
        
        # Iterative perturbation (gradient-free optimization)
        rng = np.random.RandomState(self.seed)
        n_iterations = 200
        temperature = 0.1
        
        for iteration in range(n_iterations):
            plane_idx = rng.randint(self.n_bits)
            perturbation = rng.randn(embeddings.shape[1]) * temperature
            
            candidate = best_planes.copy()
            candidate[plane_idx] += perturbation
            candidate[plane_idx] /= np.linalg.norm(candidate[plane_idx])
            
            score = self._score_subsumption(candidate, embeddings, concepts, valid_pairs)
            
            if score > best_score:
                best_planes = candidate
                best_score = score
            
            if iteration % 50 == 49:
                temperature *= 0.8
        
        self.planes = best_planes
        logger.info(f"Contrastive learning complete: final score={best_score}")
    
    def _score_subsumption(self, planes, embeddings, concepts, hypernym_pairs):
        """Score a set of planes by how many hypernym relationships they capture correctly."""
        plane_primes = [sympy.prime(i + 1) for i in range(self.n_bits)]
        
        prime_map = {}
        for concept, emb in zip(concepts, embeddings):
            projections = np.dot(planes, emb)
            bits = (projections > 0).astype(int)
            composite = 1
            for bit, prime in zip(bits, plane_primes):
                if bit == 1:
                    composite *= prime
            if composite == 1:
                composite = 2
            prime_map[concept] = composite
        
        # Count correct subsumptions
        tp = sum(
            1 for broader, narrower in hypernym_pairs
            if broader in prime_map and narrower in prime_map
            and prime_map[broader] % prime_map[narrower] == 0
        )
        
        # Penalize false positives
        rng = np.random.RandomState(99)
        fp = 0
        for _ in range(50):
            i, j = rng.choice(len(concepts), 2, replace=False)
            a, b = concepts[i], concepts[j]
            if a in prime_map and b in prime_map:
                if prime_map[a] % prime_map[b] == 0:
                    fp += 1
        
        return tp - fp * 0.5

    def fit_transform(self, concepts: List[str], embeddings: np.ndarray) -> Dict[str, int]:
        """
        Maps a list of concepts and their continuous embeddings to discrete composite primes.
        Each LSH hyperplane corresponds to a unique prime number.
        """
        logger.info(f"Mapping {len(concepts)} concepts to discrete integer space (mode={self.projection})...")
        
        # Consensus mode has its own encoding pipeline
        if self.projection == "consensus":
            self.concept_to_prime = self._generate_consensus_encoding(concepts, embeddings)
            return self.concept_to_prime
        
        if self.planes is None:
            if self.projection == "pca":
                self._generate_pca_planes(embeddings)
            elif self.projection == "contrastive":
                self._generate_contrastive_planes(embeddings, concepts)
            else:
                self._generate_random_planes(embeddings.shape[1])
            
        # Assign a prime to each hyperplane (Semantic Feature)
        plane_primes = [sympy.prime(i + 1) for i in range(self.n_bits)]
        
        for concept, emb in zip(concepts, embeddings):
            projections = np.dot(self.planes, emb)
            bits = (projections > 0).astype(int)
            
            composite_integer = 1
            for bit, prime_factor in zip(bits, plane_primes):
                if bit == 1:
                    composite_integer *= prime_factor
            
            if composite_integer == 1:
                composite_integer = 2
                
            self.concept_to_prime[concept] = composite_integer
            
        return self.concept_to_prime
    
    def get_factor(self, concept: str) -> int:
        if concept not in self.concept_to_prime:
            raise ValueError(f"Concept '{concept}' not found in the discrete mapping. Use transform() for new concepts.")
        return self.concept_to_prime[concept]

    def transform(self, concepts: List[str], embeddings: np.ndarray) -> Dict[str, int]:
        """
        Maps a list of new concepts to discrete composite primes using EXISTING planes.
        Does NOT update internal planes.
        """
        if self.planes is None:
            raise ValueError("DiscreteMapper has not been fitted. Call fit_transform first.")
            
        plane_primes = [sympy.prime(i + 1) for i in range(self.n_bits)]
        results = {}
        
        for concept, emb in zip(concepts, embeddings):
            projections = np.dot(self.planes, emb)
            bits = (projections > 0).astype(int)
            
            composite_integer = 1
            for bit, prime_factor in zip(bits, plane_primes):
                if bit == 1:
                    composite_integer *= prime_factor
            
            if composite_integer == 1:
                composite_integer = 2
            
            results[concept] = composite_integer
            # Optionally cache it
            self.concept_to_prime[concept] = composite_integer
            
        return results
