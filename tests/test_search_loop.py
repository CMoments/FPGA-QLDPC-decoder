# tests/test_search_loop.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from search.candidate import Candidate
from search.scorer import CandidateScorer, ScoreWeights
from search.search_loop import SearchLoop, SearchResult


def _make_candidates(n: int) -> list:
    candidates = []
    for move_dist in range(1, n + 1):
        code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
        layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
        geom = RoutedGeometry(layout=layout, max_move_distance=move_dist)
        steps = [ScheduleStep(time=t, ancilla_group=t % 2, detectors=list(range(4)))
                 for t in range(4 + move_dist)]
        sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
        candidates.append(Candidate(code=code, geometry=geom, schedule=sched))
    return candidates


def _make_scorer():
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.1,
                           residual_solve_rate=1.0, cross_tile_traffic=0.1)
    return CandidateScorer(weights=weights, physical_error_rate=0.001,
                           min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                           max_residual_component_size=100)


def test_search_returns_result():
    result = SearchLoop(_make_scorer()).run(_make_candidates(3))
    assert isinstance(result, SearchResult)


def test_search_selects_best_candidate():
    candidates = _make_candidates(3)
    result = SearchLoop(_make_scorer()).run(candidates)
    assert result.best_candidate in candidates
    assert result.best_score < float("inf")


def test_search_all_scores_sorted():
    result = SearchLoop(_make_scorer()).run(_make_candidates(3))
    scores = [s for _, s in result.all_scores]
    assert scores == sorted(scores)


def test_search_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        SearchLoop(_make_scorer()).run([])


def test_search_all_rejected_returns_inf():
    candidates = _make_candidates(2)
    weights = ScoreWeights(mean_latency=1.0, p99_latency=1.0, memory_traffic=1.0,
                           residual_solve_rate=1.0, cross_tile_traffic=1.0)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.9999, max_bank_conflict_rate=0.0001,
                             max_residual_component_size=1)
    result = SearchLoop(scorer).run(candidates)
    assert result.best_score == float("inf")
