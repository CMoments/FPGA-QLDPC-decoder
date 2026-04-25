# tests/test_geometry.py
import pytest
from qldpc.geometry import RoutedGeometry, TileLayout, MovementConstraint

def test_tile_layout_qubit_count():
    layout = TileLayout(rows=4, cols=4, data_per_tile=4, ancilla_per_tile=2)
    assert layout.total_data_qubits == 64
    assert layout.total_ancilla_qubits == 32
    assert layout.num_tiles == 16

def test_routed_geometry_locality():
    layout = TileLayout(rows=2, cols=2, data_per_tile=4, ancilla_per_tile=2)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    assert geom.is_local(tile_a=(0,0), tile_b=(0,1)) is True
    assert geom.is_local(tile_a=(0,0), tile_b=(1,1)) is True

def test_movement_constraint_rejects_long_range():
    layout = TileLayout(rows=4, cols=4, data_per_tile=4, ancilla_per_tile=2)
    geom = RoutedGeometry(layout=layout, max_move_distance=1)
    assert geom.is_local(tile_a=(0,0), tile_b=(3,3)) is False

def test_cross_tile_edges():
    layout = TileLayout(rows=2, cols=2, data_per_tile=4, ancilla_per_tile=2)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    edges = geom.cross_tile_edge_count()
    assert isinstance(edges, int)
    assert edges >= 0
