# tests/test_extraction_compiler.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from compiler.extraction_compiler import ExtractionCompiler, CompilerOutput
from compiler.fault_atom import FaultAtomLibrary
from compiler.residual_graph import ResidualGraph
from compiler.hardware_cost import HardwareCostIR

def _make_simple_inputs():
    code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
    layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    steps = [ScheduleStep(time=t, ancilla_group=t%2, detectors=list(range(4)))
             for t in range(6)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    return code, geom, sched

def test_compiler_returns_output():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out, CompilerOutput)

def test_compiler_output_has_atom_library():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out.atom_library, FaultAtomLibrary)

def test_compiler_output_has_residual_graph():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out.residual_graph, ResidualGraph)

def test_compiler_output_has_cost_ir():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out.cost, HardwareCostIR)
    assert 0.0 <= out.cost.predicted_atom_hit_ratio <= 1.0
