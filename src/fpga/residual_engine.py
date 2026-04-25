from compiler.residual_graph import ResidualGraph


class ResidualEliminationEngine:
    def __init__(self, residual_graph: ResidualGraph, ns_per_node: float):
        self._graph = residual_graph
        self._ns_per_node = ns_per_node

    def solve(self) -> tuple:
        n = self._graph.num_nodes
        latency = n * self._ns_per_node
        correction = frozenset()
        return correction, latency
