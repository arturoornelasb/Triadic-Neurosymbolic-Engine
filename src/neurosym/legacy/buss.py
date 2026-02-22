import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class BipolarExtractor:
    """
    Extracts orthogonal latent semantic axes from natural language corpora 
    using Centered Singular Value Decomposition (BUSS Methodology).
    """
    def __init__(self, n_components: int = 100):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2, 
            stop_words='english'
        )
        self.scaler = StandardScaler(with_std=False)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        self.is_fitted = False
        self.axes = None

    def fit(self, corpus: list[str]) -> np.ndarray:
        """
        Fits the extractor on a corpus of text.
        1. TF-IDF Vectorization (Bag of Words)
        2. Mean Centering
        3. SVD Dimensionality Reduction
        
        Returns the extracted axes (n_components, vocab_size)
        """
        logger.info(f"Vectorizing corpus of size {len(corpus)}...")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        logger.info("Centering the TF-IDF matrix...")
        # Note: For very large datasets, dense centering can OOM. 
        # Future optimization: Incremental centering or randomized SVD
        dense_matrix = tfidf_matrix.toarray()
        centered_matrix = self.scaler.fit_transform(dense_matrix)
        
        logger.info(f"Extracting {self.n_components} bipolar axes via SVD...")
        self.svd.fit(centered_matrix)
        
        self.axes = self.svd.components_
        self.is_fitted = True
        
        logger.info("Extraction complete.")
        return self.axes

    def transform(self, texts: list[str], target_axis_idx: int = 0) -> np.ndarray:
        """
        Projects new text onto a specific extracted semantic axis.
        """
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transforming text.")
            
        tfidf_vectors = self.vectorizer.transform(texts)
        # Using the standard uncentered vectors for inference projection
        # Projection = Text_TFIDF * Axis_Vector^T
        axis_vector = self.axes[target_axis_idx].reshape(-1, 1)
        projections = tfidf_vectors @ axis_vector
        
        return projections
