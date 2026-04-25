# tests/test_integration.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from search.candidate import Candidate
from search.scorer import CandidateScorer, ScoreWeights
from search.search_loop import SearchLoop
from compiler.extraction_compiler import ExtractionCompiler
from fpga.pipeline import FPGAPipeline

def test_end_to_end_search_and_decode():
    candidates = []
    for move_dist in [1, 2]:
        code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
        layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
        geom = RoutedGeometry(layout=layout, max_move_distance=move_dist)
        steps = [ScheduleStep(time=t, ancilla_group=t%2, detectors=list(range(4)))
                 for t in range(6)]
        sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
        candidates.append(Candidate(code=code, geometry=geom, schedule=sched))

    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.1,
                           residual_solve_rate=1.0, cross_tile_traffic=0.1)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                             max_residual_component_size=100)
    loop = SearchLoop(scorer=scorer)
    result = loop.run(candidates=candidates)
    assert result.best_score < float("inf")

    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(
        code=result.best_candidate.code,
        geometry=result.best_candidate.geometry,
        schedule=result.best_candidate.schedule,
    )
    pipeline = FPGAPipeline(
        atom_library=out.atom_library,
        residual_graph=out.residual_graph,
        ns_per_atom_match=10.0,
        ns_per_residual_node=50.0,
    )
    shots = [frozenset([0, 1]), frozenset([99, 100]), frozenset([0, 1])]
    stats = pipeline.batch_decode(shots)
    assert stats["mean_latency_ns"] > 0
    assert 0.0 <= stats["atom_hit_rate"] <= 1.0
