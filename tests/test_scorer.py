# tests/test_scorer.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from search.candidate import Candidate
from search.scorer import CandidateScorer, ScoreWeights


def _make_candidate():
    code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
    layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    steps = [ScheduleStep(time=t, ancilla_group=t % 2, detectors=list(range(4)))
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
                             min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                             max_residual_component_size=1000)
    score = scorer.score(c)
    assert isinstance(score, float)


def test_scorer_rejects_bad_hardware_thresholds():
    c = _make_candidate()
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.5,
                           residual_solve_rate=1.0, cross_tile_traffic=0.5)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.9999, max_bank_conflict_rate=0.001,
                             max_residual_component_size=1)
    assert scorer.score(c) == float("inf")


def test_scorer_rejects_on_ler_constraint():
    c = _make_candidate()
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.5,
                           residual_solve_rate=1.0, cross_tile_traffic=0.5)
    # max_logical_error_rate=0.0 forces rejection of any candidate with LER > 0
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.05,
                             min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                             max_residual_component_size=1000,
                             max_logical_error_rate=0.0, ler_num_shots=100)
    assert scorer.score(c) == float("inf")


def test_scorer_score_has_no_magic_scale():
    """Score must not blow up due to a hidden 1e6 multiplier on residual_solve_rate."""
    c = _make_candidate()
    weights = ScoreWeights(mean_latency=0.0, p99_latency=0.0, memory_traffic=0.0,
                           residual_solve_rate=1.0, cross_tile_traffic=0.0)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                             max_residual_component_size=1000)
    score = scorer.score(c)
    # residual_solve_rate is in [0,1]; with weight=1 the score must be <= 1.0
    assert score <= 1.0
