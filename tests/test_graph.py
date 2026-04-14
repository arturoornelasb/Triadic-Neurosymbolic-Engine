"""Tests for ScalableGraphBuilder — inverted prime index graph construction."""
import pytest
from neurosym.graph import ScalableGraphBuilder

# Known prime map: A=2*3*5, B=2*5, C=3*5, D=7
PRIME_MAP = {"A": 30, "B": 10, "C": 15, "D": 7}


@pytest.fixture
def gb():
    return ScalableGraphBuilder()


def test_build_index_populates_structures(gb):
    gb.build_index(PRIME_MAP)
    assert len(gb.concept_factors) == 4
    assert set(gb.concept_factors["A"]) == {2, 3, 5}
    assert 7 in gb.inverted_index
    assert "D" in gb.inverted_index[7]


def test_find_edges_returns_correct_pairs(gb):
    edges = gb.find_edges(PRIME_MAP)
    pairs = {(a, b) for a, b, _, _ in edges}
    # A(2,3,5) shares with B(2,5) and C(3,5); B(2,5) shares with C(5)
    assert ("A", "B") in pairs
    assert ("A", "C") in pairs
    assert ("B", "C") in pairs
    # D(7) shares nothing with A,B,C
    assert not any("D" in (a, b) for a, b, _, _ in edges)


def test_find_edges_weights(gb):
    edges = gb.find_edges(PRIME_MAP)
    edge_dict = {(a, b): w for a, b, w, _ in edges}
    assert edge_dict[("A", "B")] == 2  # share 2 and 5
    assert edge_dict[("A", "C")] == 2  # share 3 and 5
    assert edge_dict[("B", "C")] == 1  # share only 5


def test_find_edges_min_shared_filters(gb):
    edges = gb.find_edges(PRIME_MAP, min_shared=2)
    pairs = {(a, b) for a, b, _, _ in edges}
    assert ("A", "B") in pairs
    assert ("A", "C") in pairs
    assert ("B", "C") not in pairs  # only 1 shared prime


def test_find_edges_shared_primes_list(gb):
    edges = gb.find_edges(PRIME_MAP)
    for a, b, w, shared in edges:
        if (a, b) == ("A", "B"):
            assert set(shared) == {2, 5}


def test_find_neighbors(gb):
    gb.build_index(PRIME_MAP)
    neighbors = gb.find_neighbors("A", PRIME_MAP)
    neighbor_names = [n for n, _, _ in neighbors]
    assert "B" in neighbor_names
    assert "C" in neighbor_names
    assert "D" not in neighbor_names


def test_find_neighbors_unknown_concept(gb):
    gb.build_index(PRIME_MAP)
    assert gb.find_neighbors("Z", PRIME_MAP) == []


def test_find_neighbors_isolated_node(gb):
    gb.build_index(PRIME_MAP)
    assert gb.find_neighbors("D", PRIME_MAP) == []


def test_get_stats(gb):
    stats_before = gb.get_stats()
    assert stats_before["indexed"] is False
    gb.build_index(PRIME_MAP)
    stats = gb.get_stats()
    assert stats["indexed"] is True
    assert stats["total_concepts"] == 4
    assert stats["total_prime_buckets"] == 4  # primes 2, 3, 5, 7
