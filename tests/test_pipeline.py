# tests/test_pipeline.py
import pytest
from compiler.fault_atom import FaultAtom, FaultAtomLibrary
from compiler.residual_graph import ResidualGraph, ResidualNode, ResidualEdge
from fpga.pipeline import FPGAPipeline, ShotResult

def _make_lib():
    lib = FaultAtomLibrary()
    lib.add(FaultAtom(
        atom_id="a0", detector_signature=frozenset([0,1]),
        data_error_candidates=[frozenset([0])], prior_weight=0.9,
        tile=(0,0), time_slice=0, window_size=1,
    ))
    return lib

def _make_residual():
    g = ResidualGraph()
    g.add_node(ResidualNode(node_id=0, tile=(0,0), time_slice=0))
    g.add_node(ResidualNode(node_id=1, tile=(0,1), time_slice=0))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.3, cross_tile=True))
    return g

def test_pipeline_returns_shot_result():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    result = pipeline.decode(detector_events=frozenset([0, 1]))
    assert isinstance(result, ShotResult)

def test_pipeline_atom_hit_path():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    result = pipeline.decode(detector_events=frozenset([0, 1]))
    assert result.atom_hit is True
    assert result.latency_ns < 100.0

def test_pipeline_residual_path():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    result = pipeline.decode(detector_events=frozenset([99, 100]))
    assert result.atom_hit is False
    assert result.latency_ns >= 50.0

def test_pipeline_batch_stats():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    shots = [frozenset([0,1]), frozenset([99,100]), frozenset([0,1])]
    stats = pipeline.batch_decode(shots)
    assert stats["mean_latency_ns"] > 0
    assert 0.0 <= stats["atom_hit_rate"] <= 1.0
    assert "p99_latency_ns" in stats
