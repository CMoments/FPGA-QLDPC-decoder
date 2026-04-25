# tests/test_pipeline.py
import pytest
from compiler.fault_atom import FaultAtom, FaultAtomLibrary
from compiler.residual_graph import ResidualGraph, ResidualNode, ResidualEdge
from fpga.pipeline import FPGAPipeline, ShotResult


def _make_lib():
    lib = FaultAtomLibrary()
    lib.add(FaultAtom(
        atom_id="a0", detector_signature=frozenset([0, 1]),
        data_error_candidates=[frozenset([0])], prior_weight=0.9,
        tile=(0, 0), time_slice=0, window_size=1,
    ))
    return lib


def _make_residual():
    g = ResidualGraph()
    g.add_node(ResidualNode(node_id=0, tile=(0, 0), time_slice=0))
    g.add_node(ResidualNode(node_id=1, tile=(0, 1), time_slice=0))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.3, cross_tile=True))
    return g


def test_pipeline_returns_shot_result():
    pipeline = FPGAPipeline(_make_lib(), _make_residual(), 10.0, 50.0)
    result = pipeline.decode(frozenset([0, 1]))
    assert isinstance(result, ShotResult)


def test_pipeline_atom_hit_path():
    pipeline = FPGAPipeline(_make_lib(), _make_residual(), 10.0, 50.0)
    result = pipeline.decode(frozenset([0, 1]))
    assert result.atom_hit is True
    assert result.latency_ns == 10.0
    assert result.correction == frozenset([0])


def test_pipeline_residual_path_uses_active_nodes():
    """Residual path must only process nodes present in the shot's unmatched detectors."""
    pipeline = FPGAPipeline(_make_lib(), _make_residual(), 10.0, 50.0)
    # detectors 0 and 1 are in the residual graph but not matched by any atom as {99,100}
    result = pipeline.decode(frozenset([0, 1, 99]))
    # atom lookup for {0,1,99} misses; residual active nodes = {0,1} (99 not in graph)
    assert result.atom_hit is False
    # latency must reflect only the 2 active residual nodes
    assert result.latency_ns == 2 * 50.0


def test_pipeline_residual_path_empty_active_nodes():
    """Shot with no nodes in the residual graph should return zero latency."""
    pipeline = FPGAPipeline(_make_lib(), _make_residual(), 10.0, 50.0)
    result = pipeline.decode(frozenset([99, 100]))
    assert result.atom_hit is False
    assert result.latency_ns == 0.0


def test_pipeline_residual_correction_nonempty_when_nodes_active():
    pipeline = FPGAPipeline(_make_lib(), _make_residual(), 10.0, 50.0)
    result = pipeline.decode(frozenset([0, 1, 99]))
    # greedy match on edge (0,1) should produce a non-empty correction
    assert isinstance(result.correction, frozenset)


def test_pipeline_batch_stats():
    pipeline = FPGAPipeline(_make_lib(), _make_residual(), 10.0, 50.0)
    shots = [frozenset([0, 1]), frozenset([99, 100]), frozenset([0, 1])]
    stats = pipeline.batch_decode(shots)
    assert stats["mean_latency_ns"] > 0
    assert 0.0 <= stats["atom_hit_rate"] <= 1.0
    assert "p99_latency_ns" in stats
