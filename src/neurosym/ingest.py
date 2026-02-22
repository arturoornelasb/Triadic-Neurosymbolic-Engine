import pandas as pd
import numpy as np
import logging
import math
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import sympy
from neurosym.encoder import ContinuousEncoder, DiscreteMapper

logger = logging.getLogger(__name__)

class DatabaseIngestor:
    """
    Ingests tabular data (CSV) and builds a Discrete Prime Index for fast
    semantic search using integer arithmetic instead of vector distance.
    
    Uses an inverted index keyed by prime factors for sub-linear lookup.
    """
    def __init__(self, encoder: ContinuousEncoder, mapper: DiscreteMapper):
        self.encoder = encoder
        self.mapper = mapper
        self.discrete_index: Dict[int, Dict] = {}  # {id: {text, prime_factor}}
        self.inverted_index: Dict[int, Set[int]] = defaultdict(set)  # {prime: {record_ids}}
        self.is_indexed = False
        
    def _factorize(self, n: int) -> List[int]:
        """Returns the list of unique prime factors of n."""
        if n <= 1:
            return []
        factors = []
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        return factors
        
    def ingest_dataframe(self, df: pd.DataFrame, text_column: str, id_column: str = None) -> Dict:
        """
        Takes a pandas DataFrame, vectorizes the text column, hashes it 
        into composite prime factors, and builds an inverted index.
        
        Complexity: O(N * E) where N = records and E = embedding time.
        """
        logger.info(f"Ingesting {len(df)} records into the Triadic Engine...")
        
        texts = df[text_column].astype(str).tolist()
        
        # 1. Continuous Vector Batch Encoding
        embeddings = self.encoder.encode(texts)
        
        # 2. Discrete Projection (LSH → Composite Primes)
        prime_map = self.mapper.fit_transform(texts, embeddings)
        
        # 3. Build the forward index AND the inverted index
        self.discrete_index.clear()
        self.inverted_index.clear()
        
        for idx, row in df.iterrows():
            record_id = row[id_column] if id_column and id_column in df.columns else idx
            text_content = row[text_column]
            prime_factor = prime_map[text_content]
            
            self.discrete_index[record_id] = {
                "text": text_content,
                "prime_factor": prime_factor
            }
            
            # Inverted index: map each prime factor to the records that contain it
            for p in self._factorize(prime_factor):
                self.inverted_index[p].add(record_id)
            
        self.is_indexed = True
        logger.info(f"Indexed {len(df)} records. Inverted index has {len(self.inverted_index)} prime buckets.")
        return self.discrete_index

    def triadic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, int, int]]:
        """
        Semantic search using the inverted prime index.
        
        1. Encode the query into a composite prime.
        2. Factorize the query prime to get its active semantic features.
        3. Use the inverted index to find candidate records that share
           at least one prime factor (set union over factor buckets).
        4. Rank candidates by GCD-based distance.
        
        Complexity: O(C * log(C)) where C = number of candidate records
        sharing at least one factor. In the best case C << N.
        Falls back to full scan if no candidates are found.
        """
        if not self.is_indexed:
            raise ValueError("No database loaded. Please ingest data first.")
            
        # 1. Project query into discrete prime space
        query_emb = self.encoder.encode([query])
        self.mapper.fit_transform([query], query_emb)
        query_prime = self.mapper.concept_to_prime[query]
        
        # 2. Find candidates via inverted index (set union)
        query_factors = self._factorize(query_prime)
        candidate_ids: Set[int] = set()
        for p in query_factors:
            if p in self.inverted_index:
                candidate_ids.update(self.inverted_index[p])
        
        # 3. If no candidates share any factor, fall back to full scan
        if not candidate_ids:
            logger.info("No shared factors found. Falling back to full scan.")
            candidate_ids = set(self.discrete_index.keys())
        
        logger.info(f"Query '{query}' (prime={query_prime}): {len(candidate_ids)} candidates from {len(self.discrete_index)} total records.")
        
        # 4. Rank candidates by GCD-based distance
        results = []
        for record_id in candidate_ids:
            data = self.discrete_index[record_id]
            db_prime = data["prime_factor"]
            
            shared = math.gcd(query_prime, db_prime)
            missing = query_prime // shared
            extra = db_prime // shared
            
            if extra == 1 and missing == 1:
                distance = 0
            else:
                distance = abs(extra - missing) + (extra * missing)
            
            results.append((distance, record_id, data["text"], db_prime))
            
        results.sort(key=lambda x: x[0])
        return [(r[1], r[2], r[0], r[3]) for r in results[:top_k]]
