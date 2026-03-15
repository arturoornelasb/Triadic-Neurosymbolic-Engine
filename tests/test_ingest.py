import pandas as pd
from neurosym.encoder import ContinuousEncoder, DiscreteMapper
from neurosym.ingest import DatabaseIngestor

def test_database_ingestion():
    print("=== Testing Triadic Database Ingestion & Search ===")
    
    # 1. Load the mock database
    df = pd.read_csv("tests/sample_catalog.csv")
    print(f"Loaded {len(df)} records from CSV.")
    
    # 2. Initialize the Engine
    encoder = ContinuousEncoder()
    mapper = DiscreteMapper(n_bits=8, seed=42)
    
    # 3. Ingest and create Discrete Index
    ingestor = DatabaseIngestor(encoder, mapper)
    index = ingestor.ingest_dataframe(df, text_column="name", id_column="id")
    assert index is not None, "ingest_dataframe() should return an index"
    assert len(index) == len(df), f"Expected {len(df)} index entries, got {len(index)}"

    print("\n[Discrete Prime Index Created]")
    for record_id, data in index.items():
        print(f"ID: {record_id} | Text: '{data['text']}' | Prime: {data['prime_factor']}")
        
    # 4. Perform a Triadic Search
    # "Fast Silver Racing Motorcycle" should map closely to "Speed Red Sports Car"
    queries = ["A fast vehicle", "A red fruit to eat", "Something for a King"]
    
    print("\n--- Running Triadic Queries (GCD Arithmetic) ---")
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = ingestor.triadic_search(q, top_k=2)
        assert results is not None, "triadic_search() should return results"
        assert len(results) <= 2, f"Expected at most 2 results, got {len(results)}"
        for i, (record_id, text, distance, prime) in enumerate(results):
            print(f"  {i+1}. Result: '{text}' (Distance: {distance}, Prime: {prime})")

if __name__ == "__main__":
    test_database_ingestion()
