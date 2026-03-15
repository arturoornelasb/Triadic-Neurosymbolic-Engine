import pandas as pd
from neurosym.anomaly import AnomalyDetector, RelationalRule

def test_anomaly_detection():
    print("=== Anomaly Detection Engine Test ===\n")
    
    # 1. Load the invoices
    df = pd.read_csv("tests/sample_invoices.csv")
    print(f"Loaded {len(df)} invoices.\n")
    
    # 2. Define the rule: total = qty × unit_price × tax_rate
    detector = AnomalyDetector()
    detector.add_rule(RelationalRule(
        name="Invoice Total Check",
        factor_columns=["qty", "unit_price", "tax_rate"],
        result_column="total",
        tolerance=0.01  # 1% tolerance for rounding
    ))
    
    # 3. Scan
    anomalies = detector.scan(df)
    assert anomalies is not None, "scan() should return a list"
    assert isinstance(anomalies, list), "scan() should return a list"

    print(f"\n--- Results: {len(anomalies)} anomalies found ---\n")
    
    for a in anomalies:
        invoice_id = df.loc[a.row_index, "invoice_id"]
        print(f"  [{a.severity}] {invoice_id} (row {a.row_index})")
        print(f"    {a.explanation}")
        print(f"    Missing factor: {a.missing_factor:.4f}x")
        print()
    
    # Verify we caught the planted anomalies
    flagged_rows = {a.row_index for a in anomalies}
    assert len(flagged_rows) > 0, "Should detect at least one anomaly in sample data"
    print(f"Flagged row indices: {flagged_rows}")
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_anomaly_detection()
