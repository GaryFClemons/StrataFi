"""Lineage returns every descendant of a document."""

import pytest

from stratafi_core.lineage import LineageGraph

# A small sweep over two documents: two chunkers, two embedding models, and one index built over
# both documents' vectors (a diamond that joins the documents).
EDGES = [
    ("parsed-a", "doc-a"),
    ("cleaned-a", "parsed-a"),
    ("chunks-a-heading", "cleaned-a"),
    ("chunks-a-fixed", "cleaned-a"),
    ("vectors-a-heading-m1", "chunks-a-heading"),
    ("vectors-a-heading-m2", "chunks-a-heading"),
    ("vectors-a-fixed-m1", "chunks-a-fixed"),
    ("parsed-b", "doc-b"),
    ("chunks-b-heading", "parsed-b"),
    ("vectors-b-heading-m1", "chunks-b-heading"),
    ("index-m1", "vectors-a-heading-m1"),
    ("index-m1", "vectors-b-heading-m1"),
    ("index-m1", "vectors-a-fixed-m1"),
]

DESCENDANTS_OF_A = {
    "parsed-a": 1,
    "cleaned-a": 2,
    "chunks-a-heading": 3,
    "chunks-a-fixed": 3,
    "vectors-a-heading-m1": 4,
    "vectors-a-heading-m2": 4,
    "vectors-a-fixed-m1": 4,
    "index-m1": 5,
}


@pytest.fixture
def graph() -> LineageGraph:
    return LineageGraph.from_edges(EDGES)


def test_every_descendant_of_a_document_with_its_depth(graph: LineageGraph) -> None:
    assert graph.descendants("doc-a") == DESCENDANTS_OF_A


def test_a_shared_artifact_is_a_descendant_of_both_documents(graph: LineageGraph) -> None:
    assert set(graph.descendants("doc-b")) == {
        "parsed-b",
        "chunks-b-heading",
        "vectors-b-heading-m1",
        "index-m1",
    }


def test_ancestors(graph: LineageGraph) -> None:
    assert graph.ancestors("vectors-a-heading-m2") == {
        "chunks-a-heading": 1,
        "cleaned-a": 2,
        "parsed-a": 3,
        "doc-a": 4,
    }
    assert set(graph.ancestors("index-m1")) >= {"doc-a", "doc-b"}


def test_roots_leaves_and_inputs(graph: LineageGraph) -> None:
    assert graph.roots() == {"doc-a", "doc-b"}
    assert graph.leaves() == {"vectors-a-heading-m2", "index-m1"}
    assert graph.inputs("index-m1") == {
        "vectors-a-heading-m1",
        "vectors-b-heading-m1",
        "vectors-a-fixed-m1",
    }
    assert "doc-a" in graph
    assert "unknown" not in graph


def test_an_artifact_with_no_descendants(graph: LineageGraph) -> None:
    assert graph.descendants("index-m1") == {}
    assert graph.descendants("unknown") == {}


def test_a_cycle_still_ends() -> None:
    graph = LineageGraph.from_edges([("b", "a"), ("c", "b"), ("a", "c")])
    assert graph.descendants("a") == {"b": 1, "c": 2}


def test_an_artifact_cannot_be_its_own_input() -> None:
    with pytest.raises(ValueError, match="own input"):
        LineageGraph().add("a", ["a"])
