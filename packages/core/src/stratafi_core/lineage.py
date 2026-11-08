"""Artifact lineage as a graph: which artifacts were made from which.

Each artifact records its input artifacts. A raw document is an artifact with no inputs, so its
descendants are everything derived from it: the parsed document, the chunks, the vectors. Purging
a document follows this to every derived artifact.

``LineageGraph`` is the in-memory form. ``stratafi_store.lineage`` answers the same questions in
SQL, and the integration tests check that both agree.

Content addressing makes cycles impossible (an artifact's key depends on its inputs' keys), but
the walks don't rely on that: each node is visited once.
"""

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping


class LineageGraph:
    def __init__(self) -> None:
        self._inputs: dict[str, set[str]] = defaultdict(set)
        self._outputs: dict[str, set[str]] = defaultdict(set)
        self._nodes: set[str] = set()

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, str]]) -> "LineageGraph":
        """Build from ``(artifact, input)`` pairs."""
        graph = cls()
        for artifact, source in edges:
            graph.add(artifact, [source])
        return graph

    def add(self, artifact: str, inputs: Iterable[str] = ()) -> None:
        self._nodes.add(artifact)
        for source in inputs:
            if source == artifact:
                raise ValueError("an artifact can't be its own input")
            self._nodes.add(source)
            self._inputs[artifact].add(source)
            self._outputs[source].add(artifact)

    def __contains__(self, artifact: object) -> bool:
        return artifact in self._nodes

    def inputs(self, artifact: str) -> frozenset[str]:
        return frozenset(self._inputs.get(artifact, ()))

    def descendants(self, artifact: str) -> dict[str, int]:
        """Every artifact derived from ``artifact``, directly or not, with its shortest depth."""
        return _walk(self._outputs, artifact)

    def ancestors(self, artifact: str) -> dict[str, int]:
        """Every artifact ``artifact`` was derived from, with its shortest depth."""
        return _walk(self._inputs, artifact)

    def roots(self) -> set[str]:
        """Artifacts with no inputs: raw documents, imported files."""
        return {node for node in self._nodes if not self._inputs.get(node)}

    def leaves(self) -> set[str]:
        """Artifacts nothing was derived from."""
        return {node for node in self._nodes if not self._outputs.get(node)}


def _walk(neighbours: Mapping[str, set[str]], start: str) -> dict[str, int]:
    depths: dict[str, int] = {}
    queue = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        for nxt in neighbours.get(node, ()):
            if nxt != start and nxt not in depths:
                depths[nxt] = depth + 1
                queue.append((nxt, depth + 1))
    return depths
