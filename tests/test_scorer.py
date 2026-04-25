# tests/test_scorer.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from search.candidate import Candidate
from search.scorer import CandidateScorer, ScoreWeights

def _make_candidate():
    code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
    layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    steps = [ScheduleStep(time=t, ancilla_group=t%2, detectors=list(range(4)))
             for t in range(6)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    return Candidate(code=code, geometry=geom, schedule=sched)

def test_candidate_fields():
    c = _make_candidate()
    assert c.code is not None
    assert c.geometry is not None
    assert c.schedule is not None

def test_scorer_returns_float():
    c = _make_candidate()
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.5,
                           residual_solve_rate=1.0, cross_tile_traffic=0.5)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.80, max_bank_conflict_rate=0.10,
                             max_residual_component_size=8)
    score = scorer.score(c)
    assert isinstance(score, float)

def test_scorer_rejects_bad_candidate():
    c = _make_candidate()
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.5,
                           residual_solve_rate=1.0, cross_tile_traffic=0.5)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.9999, max_bank_conflict_rate=0.001,
                             max_residual_component_size=1)
    score = scorer.score(c)
    assert score == float("inf")
