"""Tests for ReportGenerator — HTML/JSON/CSV export."""
import json
import os
import pytest
from neurosym.reports import ReportGenerator

PRIME_MAP = {"King": 30, "Queen": 10, "Man": 3}
EDGES = [("King", "Queen", 2, [2, 5]), ("King", "Man", 1, [3])]


def _factorize(n):
    if n <= 1:
        return []
    factors = []
    for p in [2, 3, 5, 7, 11, 13]:
        while n % p == 0:
            if p not in factors:
                factors.append(p)
            n //= p
    return factors


@pytest.fixture
def report():
    r = ReportGenerator(title="Test Report")
    r.add_encoding_section(PRIME_MAP, model="test", lsh_bits=8, factorize_fn=_factorize)
    r.add_graph_section(EDGES, node_count=3)
    return r


def test_add_encoding_section_rows_sorted_by_prime(report):
    enc = [s for s in report.sections if s["type"] == "encoding"][0]
    primes = [r["prime_factor"] for r in enc["rows"]]
    assert primes == sorted(primes)


def test_add_graph_section_stats(report):
    graph = [s for s in report.sections if s["type"] == "graph"][0]
    assert graph["total_nodes"] == 3
    assert graph["total_edges"] == 2
    assert graph["avg_edge_weight"] == 1.5


def test_add_audit_section():
    r = ReportGenerator()
    discs = [{"concept_a": "cat", "concept_b": "dog"}]
    r.add_audit_section(discs, "m1", "m2", total_pairs=10, total_concepts=5)
    audit = r.sections[0]
    assert audit["discrepancy_rate"] == "10.0%"


def test_to_json_structure(report):
    j = json.loads(report.to_json())
    assert j["title"] == "Test Report"
    assert "generated_at" in j
    assert len(j["sections"]) == 2


def test_to_csv_contains_encoding_rows(report):
    csv_out = report.to_csv()
    assert "King" in csv_out
    assert "Queen" in csv_out
    assert "Prime Factor" in csv_out


def test_to_html_is_valid_document(report):
    html = report.to_html()
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Test Report</title>" in html
    assert "King" in html
    assert "</html>" in html


def test_html_escapes_special_characters():
    r = ReportGenerator()
    r.add_encoding_section({"<script>alert(1)</script>": 2}, model="t", lsh_bits=8)
    html = r.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_save_writes_file(tmp_path, report):
    for fmt in ["html", "json", "csv"]:
        path = str(tmp_path / f"report.{fmt}")
        report.save(path, format=fmt)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0


def test_save_rejects_unknown_format(tmp_path, report):
    with pytest.raises(ValueError, match="Unsupported format"):
        report.save(str(tmp_path / "report.xml"), format="xml")
