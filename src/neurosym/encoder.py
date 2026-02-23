import numpy as np
from typing import List, Dict, Optional
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
    
    Supports two projection modes:
        - 'random' (default): Random hyperplanes from N(0, I_d)
        - 'pca': Principal Component directions from the corpus
          (deterministic, seed-independent, corpus-adapted)
    """
    def __init__(self, n_bits: int = 16, seed: int = 42, projection: str = "random"):
        self.n_bits = n_bits
        self.random_state = np.random.RandomState(seed)
        self.projection = projection
        self.planes = None
        self.concept_to_prime: Dict[str, int] = {}
        
    def _generate_random_planes(self, dim: int):
        """Generate random hyperplanes from N(0, I_d)."""
        self.planes = self.random_state.randn(self.n_bits, dim)
    
    def _generate_pca_planes(self, embeddings: np.ndarray):
        """
        Generate hyperplanes from the top-k principal components.
        
        This makes the encoding:
          - Deterministic (no seed dependency)
          - Corpus-adapted (captures actual variance directions)
          - More semantically meaningful (each hyperplane splits
            along a direction of maximum variance in the data)
        """
        from sklearn.decomposition import PCA
        
        k = min(self.n_bits, embeddings.shape[1], embeddings.shape[0])
        pca = PCA(n_components=k)
        pca.fit(embeddings)
        
        # Use the principal component vectors as hyperplane normals
        self.planes = pca.components_[:self.n_bits]
        
        # If we need more planes than PCA gives, pad with random ones
        if self.planes.shape[0] < self.n_bits:
            extra = self.random_state.randn(
                self.n_bits - self.planes.shape[0], embeddings.shape[1]
            )
            self.planes = np.vstack([self.planes, extra])
        
        logger.info(
            f"PCA projection: {k} components explain "
            f"{sum(pca.explained_variance_ratio_[:k])*100:.1f}% of variance"
        )

    def fit_transform(self, concepts: List[str], embeddings: np.ndarray) -> Dict[str, int]:
        """
        Maps a list of concepts and their continuous embeddings to discrete composite primes.
        Each LSH hyperplane corresponds to a unique prime number.
        """
        logger.info(f"Mapping {len(concepts)} concepts to discrete integer space (mode={self.projection})...")
        
        if self.planes is None:
            if self.projection == "pca":
                self._generate_pca_planes(embeddings)
            else:
                self._generate_random_planes(embeddings.shape[1])
            
        # Assign a prime to each hyperplane (Semantic Feature)
        plane_primes = [sympy.prime(i + 1) for i in range(self.n_bits)]
        
        for concept, emb in zip(concepts, embeddings):
            projections = np.dot(self.planes, emb)
            bits = (projections > 0).astype(int)
            
            # The discrete integer is the product of its active semantic feature primes
            composite_integer = 1
            for bit, prime_factor in zip(bits, plane_primes):
                if bit == 1:
                    composite_integer *= prime_factor
            
            # Fallback for origin vector
            if composite_integer == 1:
                composite_integer = 2  # minimum prime
                
            self.concept_to_prime[concept] = composite_integer
            
        return self.concept_to_prime
    
    def get_factor(self, concept: str) -> int:
        if concept not in self.concept_to_prime:
            raise ValueError(f"Concept '{concept}' not found in the discrete mapping.")
        return self.concept_to_prime[concept]

