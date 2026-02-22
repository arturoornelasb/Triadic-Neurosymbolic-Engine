"""
Triadic Neurosymbolic Engine — REST API Server

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Interactive docs at: http://localhost:8000/docs
"""
import sys
import os
import math

# Ensure neurosym source is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import networkx as nx

from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.triadic import DiscreteValidator
from neurosym.storage import PrimeIndexDB
from api.models import (
    EncodeRequest, EncodeResponse, ConceptPrime,
    AuditRequest, AuditResponse, AuditDiscrepancy,
    SearchRequest, SearchResponse, SearchResult,
    HealthResponse,
)

# --- Global engine state ---
engine = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the default encoder and validator on startup."""
    engine["encoder"] = ContinuousEncoder("all-MiniLM-L6-v2")
    engine["validator"] = DiscreteValidator()
    engine["db"] = PrimeIndexDB()
    engine["prime_map"] = {}
    engine["model_name"] = "all-MiniLM-L6-v2"
    yield
    engine.clear()


app = FastAPI(
    title="Triadic Neurosymbolic Engine API",
    description="Deterministic algebraic framework for neurosymbolic validation, semantic projection, and AI model auditing.",
    version="0.1.0",
    license_info={"name": "CC BY-NC 4.0", "url": "https://creativecommons.org/licenses/by-nc/4.0/"},
    lifespan=lifespan,
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check engine status."""
    return HealthResponse(
        default_model=engine.get("model_name", "none"),
        concepts_loaded=len(engine.get("prime_map", {})),
    )


@app.post("/encode", response_model=EncodeResponse, tags=["Core"])
async def encode_concepts(req: EncodeRequest):
    """
    Encode a list of natural language concepts into composite prime integers
    via LSH projection and prime factorization.
    """
    enc = engine["encoder"]
    val = engine["validator"]
    db = engine["db"]
    
    mapper = DiscreteMapper(n_bits=req.lsh_bits, seed=req.seed)
    embeddings = enc.encode(req.concepts)
    prime_map = mapper.fit_transform(req.concepts, embeddings)
    
    # Update global state
    engine["prime_map"].update(prime_map)
    
    # Persist if requested
    if req.persist:
        db.save_index(prime_map, req.model, req.lsh_bits, req.seed)
    
    results = [
        ConceptPrime(
            concept=c,
            prime_factor=p,
            prime_decomposition=val._prime_factors(p)
        )
        for c, p in prime_map.items()
    ]
    
    return EncodeResponse(
        model=req.model,
        lsh_bits=req.lsh_bits,
        seed=req.seed,
        total_concepts=len(results),
        results=results,
    )


@app.post("/audit", response_model=AuditResponse, tags=["Core"])
async def audit_models(req: AuditRequest):
    """
    Compare how two different embedding models structure the same concepts
    by analyzing topological shortest-path discrepancies in their prime factor graphs.
    """
    if req.model_a == req.model_b:
        raise HTTPException(status_code=400, detail="model_a and model_b must be different.")
    
    # Load both models
    enc_a = ContinuousEncoder(req.model_a)
    enc_b = ContinuousEncoder(req.model_b)
    
    # Encode
    emb_a = enc_a.encode(req.concepts)
    emb_b = enc_b.encode(req.concepts)
    
    # Map to primes
    mapper_a = DiscreteMapper(n_bits=req.lsh_bits, seed=req.seed)
    mapper_b = DiscreteMapper(n_bits=req.lsh_bits, seed=req.seed)
    primes_a = mapper_a.fit_transform(req.concepts, emb_a)
    primes_b = mapper_b.fit_transform(req.concepts, emb_b)
    
    # Build graphs
    graph_a = nx.Graph()
    graph_b = nx.Graph()
    graph_a.add_nodes_from(req.concepts)
    graph_b.add_nodes_from(req.concepts)
    
    for i in range(len(req.concepts)):
        for j in range(i + 1, len(req.concepts)):
            w1, w2 = req.concepts[i], req.concepts[j]
            if math.gcd(primes_a[w1], primes_a[w2]) > 1:
                graph_a.add_edge(w1, w2)
            if math.gcd(primes_b[w1], primes_b[w2]) > 1:
                graph_b.add_edge(w1, w2)
    
    # Precompute shortest paths
    paths_a = dict(nx.all_pairs_shortest_path_length(graph_a))
    paths_b = dict(nx.all_pairs_shortest_path_length(graph_b))
    
    results = []
    total_pairs = 0
    discrepancies = 0
    
    for i in range(len(req.concepts)):
        for j in range(i + 1, len(req.concepts)):
            total_pairs += 1
            w1, w2 = req.concepts[i], req.concepts[j]
            
            dist_a = paths_a.get(w1, {}).get(w2, float("inf"))
            dist_b = paths_b.get(w1, {}).get(w2, float("inf"))
            
            if dist_a != dist_b:
                discrepancies += 1
                chain = ""
                if dist_a != float("inf") and dist_a <= 3:
                    chain = " → ".join(nx.shortest_path(graph_a, w1, w2))
                elif dist_b != float("inf") and dist_b <= 3:
                    chain = " → ".join(nx.shortest_path(graph_b, w1, w2))
                
                results.append(AuditDiscrepancy(
                    concept_a=w1,
                    concept_b=w2,
                    distance_model_a=str(dist_a) if dist_a != float("inf") else "INF",
                    distance_model_b=str(dist_b) if dist_b != float("inf") else "INF",
                    chain=chain,
                ))
    
    # Persist audit results
    db = engine["db"]
    db.save_audit([r.model_dump() for r in results], req.model_a, req.model_b)
    
    return AuditResponse(
        model_a=req.model_a,
        model_b=req.model_b,
        total_concepts=len(req.concepts),
        total_pairs=total_pairs,
        discrepancies_found=discrepancies,
        discrepancy_rate=round(discrepancies / max(1, total_pairs), 4),
        results=results,
    )


@app.post("/search", response_model=SearchResponse, tags=["Core"])
async def search_concepts(req: SearchRequest):
    """
    Search the in-memory prime index using GCD-based arithmetic similarity.
    """
    prime_map = engine.get("prime_map", {})
    if not prime_map:
        raise HTTPException(status_code=400, detail="No concepts loaded. Call /encode first.")
    
    enc = engine["encoder"]
    val = engine["validator"]
    mapper = DiscreteMapper(n_bits=8, seed=42)
    
    q_emb = enc.encode([req.query])
    q_map = mapper.fit_transform([req.query], q_emb)
    q_prime = q_map[req.query]
    
    scored = []
    for concept, p in prime_map.items():
        shared = math.gcd(q_prime, p)
        if shared > 1:
            score = len(val._prime_factors(shared))
            scored.append((score, concept, p))
    
    scored.sort(key=lambda x: -x[0])
    top = scored[:req.top_k]
    
    results = [
        SearchResult(rank=i + 1, concept=c, distance=s, prime_factor=p)
        for i, (s, c, p) in enumerate(top)
    ]
    
    return SearchResponse(
        query=req.query,
        query_prime=q_prime,
        total_indexed=len(prime_map),
        results=results,
    )
