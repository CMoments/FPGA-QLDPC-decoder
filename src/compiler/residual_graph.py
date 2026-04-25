from dataclasses import dataclass
import networkx as nx


@dataclass
class ResidualNode:
    node_id: int
    tile: tuple
    time_slice: int


@dataclass
class ResidualEdge:
    u: int
    v: int
    weight: float
    cross_tile: bool


class ResidualGraph:
    def __init__(self):
        self._g = nx.Graph()
        self._edges = []

    def add_node(self, node: ResidualNode) -> None:
        self._g.add_node(node.node_id, tile=node.tile, time_slice=node.time_slice)

    def add_edge(self, edge: ResidualEdge) -> None:
        self._g.add_edge(edge.u, edge.v, weight=edge.weight, cross_tile=edge.cross_tile)
        self._edges.append(edge)

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def component_sizes(self) -> list:
        return [len(c) for c in nx.connected_components(self._g)]

    def cross_tile_edge_count(self) -> int:
        return sum(1 for e in self._edges if e.cross_tile)
