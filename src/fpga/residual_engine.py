from __future__ import annotations
import networkx as nx
from compiler.residual_graph import ResidualGraph


class ResidualEliminationEngine:
    """Greedy minimum-weight matching on each connected component of the residual graph."""

    def __init__(self, residual_graph: ResidualGraph, ns_per_node: float):
        self._graph = residual_graph
        self._ns_per_node = ns_per_node

    def solve(self, active_nodes: frozenset | None = None) -> tuple[frozenset, float]:
        """
        Solve the residual subgraph.

        active_nodes: if provided, restrict to this subset of node IDs (per-shot residual).
        Returns (correction, latency_ns).
        """
        g = self._graph._g
        if active_nodes is not None:
            g = g.subgraph(active_nodes)

        n = g.number_of_nodes()
        if n == 0:
            return frozenset(), 0.0

        correction: set[int] = set()
        for component in nx.connected_components(g):
            sub = g.subgraph(component)
            correction.update(self._greedy_match(sub))

        latency = n * self._ns_per_node
        return frozenset(correction), latency

    def _greedy_match(self, g: nx.Graph) -> set[int]:
        """Greedy matching: repeatedly pick the highest-weight edge, emit its data errors."""
        matched: set[int] = set()
        correction: set[int] = set()
        edges = sorted(g.edges(data=True), key=lambda e: e[2].get('weight', 0.0), reverse=True)
        for u, v, data in edges:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                correction.add(u)
        return correction
