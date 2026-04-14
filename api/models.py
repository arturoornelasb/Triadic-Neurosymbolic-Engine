"""
Pydantic request/response models for the Triadic Neurosymbolic Engine REST API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# --- Request Models ---

class EncodeRequest(BaseModel):
    concepts: List[str] = Field(..., min_length=1, max_length=10000, description="List of concepts to encode into prime integers (max 10,000)")
    model: str = Field(default="all-MiniLM-L6-v2", max_length=100, description="Embedding model name")
    lsh_bits: int = Field(default=8, ge=2, le=64, description="Number of LSH hyperplane bits")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    persist: bool = Field(default=False, description="Whether to persist the index to SQLite")


class AuditRequest(BaseModel):
    concepts: List[str] = Field(..., min_length=2, max_length=1000, description="List of concepts to audit across two models (max 1,000)")
    model_a: str = Field(default="all-MiniLM-L6-v2", max_length=100, description="First embedding model")
    model_b: str = Field(default="paraphrase-MiniLM-L3-v2", max_length=100, description="Second embedding model")
    lsh_bits: int = Field(default=8, ge=2, le=64)
    seed: int = Field(default=42)


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500, description="Natural language query to search for")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")


# --- Response Models ---

class ConceptPrime(BaseModel):
    concept: str
    prime_factor: int
    prime_decomposition: List[int]


class EncodeResponse(BaseModel):
    model: str
    lsh_bits: int
    seed: int
    total_concepts: int
    results: List[ConceptPrime]


class AuditDiscrepancy(BaseModel):
    concept_a: str
    concept_b: str
    distance_model_a: str
    distance_model_b: str
    chain: str = ""


class AuditResponse(BaseModel):
    model_a: str
    model_b: str
    total_concepts: int
    total_pairs: int
    discrepancies_found: int
    discrepancy_rate: float
    results: List[AuditDiscrepancy]


class SearchResult(BaseModel):
    rank: int
    concept: str
    distance: int
    prime_factor: int


class SearchResponse(BaseModel):
    query: str
    query_prime: int
    total_indexed: int
    results: List[SearchResult]


class SubsumesRequest(BaseModel):
    concept_a: str = Field(..., max_length=500, description="First concept (potential supertype)")
    concept_b: str = Field(..., max_length=500, description="Second concept (potential subtype)")


class SubsumesResponse(BaseModel):
    concept_a: str
    concept_b: str
    prime_a: int
    prime_b: int
    a_subsumes_b: bool
    b_subsumes_a: bool


class ComposeRequest(BaseModel):
    concepts: List[str] = Field(..., min_length=2, max_length=100, description="Concepts to compose")


class ComposeResponse(BaseModel):
    concepts: List[str]
    primes: List[int]
    composed_prime: int
    composed_factors: List[int]


class GapRequest(BaseModel):
    concept_a: str = Field(..., max_length=500)
    concept_b: str = Field(..., max_length=500)


class GapResponse(BaseModel):
    concept_a: str
    concept_b: str
    prime_a: int
    prime_b: int
    shared: int
    only_in_a: int
    only_in_b: int
    a_contains_b: bool
    b_contains_a: bool


class AnalogyRequest(BaseModel):
    concept_a: str = Field(..., max_length=500, description="A in A:B :: C:D")
    concept_b: str = Field(..., max_length=500, description="B in A:B :: C:D")
    concept_c: str = Field(..., max_length=500, description="C in A:B :: C:D")


class AnalogyResponse(BaseModel):
    concept_a: str
    concept_b: str
    concept_c: str
    prime_a: int
    prime_b: int
    prime_c: int
    predicted_prime_d: int
    is_valid: bool
    is_hypothetical: bool
    missing_factor: Optional[int] = None
    trace: str


class HealthResponse(BaseModel):
    engine: str = "Triadic Neurosymbolic Engine"
    version: str = "0.2.0"
    status: str = "operational"
    default_model: str
    concepts_loaded: int
