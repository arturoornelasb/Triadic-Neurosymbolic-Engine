"""Tests for PrimeIndexDB — SQLite persistence layer."""
import pytest
from neurosym.storage import PrimeIndexDB


@pytest.fixture
def db(tmp_path):
    return PrimeIndexDB(db_path=str(tmp_path / "test.db"))


def test_save_and_load_index(db):
    pm = {"King": 30, "Queen": 10, "Man": 3}
    db.save_index(pm, model="test-model", lsh_bits=8, seed=42)
    loaded = db.load_index(model="test-model", lsh_bits=8, seed=42)
    assert loaded == pm


def test_load_empty_index(db):
    loaded = db.load_index(model="nonexistent", lsh_bits=8, seed=42)
    assert loaded == {}


def test_upsert_overwrites(db):
    db.save_index({"King": 30}, model="m", lsh_bits=8, seed=42)
    db.save_index({"King": 60}, model="m", lsh_bits=8, seed=42)
    loaded = db.load_index(model="m", lsh_bits=8, seed=42)
    assert loaded["King"] == 60


def test_different_configs_are_isolated(db):
    db.save_index({"A": 10}, model="m1", lsh_bits=8, seed=42)
    db.save_index({"B": 20}, model="m2", lsh_bits=8, seed=42)
    assert db.load_index("m1", 8, 42) == {"A": 10}
    assert db.load_index("m2", 8, 42) == {"B": 20}


def test_delete_index(db):
    db.save_index({"A": 10}, model="m1", lsh_bits=8, seed=42)
    db.save_index({"B": 20}, model="m2", lsh_bits=8, seed=42)
    deleted = db.delete_index("m1", 8, 42)
    assert deleted == 1
    assert db.load_index("m1", 8, 42) == {}
    assert db.load_index("m2", 8, 42) == {"B": 20}


def test_list_indexes(db):
    db.save_index({"A": 10, "B": 20}, model="m1", lsh_bits=8, seed=42)
    db.save_index({"C": 30}, model="m2", lsh_bits=16, seed=7)
    indexes = db.list_indexes()
    assert len(indexes) == 2
    models = {ix["model"] for ix in indexes}
    assert models == {"m1", "m2"}
    for ix in indexes:
        if ix["model"] == "m1":
            assert ix["concept_count"] == 2


def test_concept_count(db):
    assert db.concept_count() == 0
    db.save_index({"A": 10, "B": 20}, model="m", lsh_bits=8, seed=42)
    assert db.concept_count() == 2


def test_save_audit_and_export_csv(db):
    results = [
        {"concept_a": "cat", "concept_b": "dog", "distance_model_a": "1", "distance_model_b": "2"},
    ]
    db.save_audit(results, model_a="m1", model_b="m2")
    csv_out = db.export_csv()
    assert "text" in csv_out  # header row from concepts table (empty but valid)


def test_export_csv_with_data(db):
    db.save_index({"X": 5, "Y": 7}, model="m", lsh_bits=8, seed=42)
    csv_out = db.export_csv(model="m")
    assert "X" in csv_out
    assert "Y" in csv_out
    lines = csv_out.strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
