from __future__ import annotations
from dataclasses import dataclass
from search.candidate import Candidate
from compiler.extraction_compiler import ExtractionCompiler
from baselines.bp_decoder import FixedExtractionBaseline


@dataclass
class ScoreWeights:
    mean_latency: float
    p99_latency: float
    memory_traffic: float
    residual_solve_rate: float
    cross_tile_traffic: float


class CandidateScorer:
    def __init__(self, weights: ScoreWeights, physical_error_rate: float,
                 min_atom_hit_ratio: float, max_bank_conflict_rate: float,
                 max_residual_component_size: int,
                 max_logical_error_rate: float = 1.0,
                 ler_num_shots: int = 500):
        self._weights = weights
        self._compiler = ExtractionCompiler(physical_error_rate)
        self._min_atom_hit = min_atom_hit_ratio
        self._max_bank_conflict = max_bank_conflict_rate
        self._max_res_size = max_residual_component_size
        self._max_ler = max_logical_error_rate
        self._baseline = FixedExtractionBaseline(physical_error_rate, ler_num_shots)

    def score(self, candidate: Candidate) -> float:
        out = self._compiler.compile(code=candidate.code, geometry=candidate.geometry,
                                     schedule=candidate.schedule)
        cost = out.cost
        if not cost.passes_thresholds(self._min_atom_hit, self._max_bank_conflict,
                                      self._max_res_size):
            return float("inf")
        if self._max_ler < 1.0:
            ler = self._baseline.logical_error_rate(candidate.code)
            if ler > self._max_ler:
                return float("inf")
        w = self._weights
        return (w.mean_latency * cost.predicted_mean_latency_ns
                + w.p99_latency * cost.predicted_p99_latency_ns
                + w.memory_traffic * cost.predicted_bram_traffic
                + w.residual_solve_rate * cost.predicted_residual_solve_rate
                + w.cross_tile_traffic * cost.predicted_cross_tile_traffic)
