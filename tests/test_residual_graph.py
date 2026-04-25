# tests/test_residual_graph.py
import pytest
from compiler.residual_graph import ResidualGraph, ResidualNode, ResidualEdge

def test_residual_graph_empty():
    g = ResidualGraph()
    assert g.num_nodes == 0
    assert g.num_edges == 0

def test_residual_graph_add_node_and_edge():
    g = ResidualGraph()
    g.add_node(ResidualNode(node_id=0, tile=(0,0), time_slice=1))
    g.add_node(ResidualNode(node_id=1, tile=(0,1), time_slice=1))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.3, cross_tile=True))
    assert g.num_nodes == 2
    assert g.num_edges == 1

def test_residual_graph_component_sizes():
    g = ResidualGraph()
    for i in range(4):
        g.add_node(ResidualNode(node_id=i, tile=(0,0), time_slice=0))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.5, cross_tile=False))
    g.add_edge(ResidualEdge(u=2, v=3, weight=0.5, cross_tile=False))
    sizes = g.component_sizes()
    assert sorted(sizes) == [2, 2]

def test_residual_graph_cross_tile_edge_count():
    g = ResidualGraph()
    for i in range(3):
        g.add_node(ResidualNode(node_id=i, tile=(i,0), time_slice=0))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.4, cross_tile=True))
    g.add_edge(ResidualEdge(u=1, v=2, weight=0.4, cross_tile=False))
    assert g.cross_tile_edge_count() == 1
