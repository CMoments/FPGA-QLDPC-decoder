from dataclasses import dataclass
from search.candidate import Candidate
from compiler.extraction_compiler import ExtractionCompiler

@dataclass
class ScoreWeights:
    mean_latency: float
    p99_latency: float
    memory_traffic: float
    residual_solve_rate: float
    cross_tile_traffic: float

class CandidateScorer:
    def __init__(self, weights, physical_error_rate, min_atom_hit_ratio,
                 max_bank_conflict_rate, max_residual_component_size):
        self._weights = weights
        self._compiler = ExtractionCompiler(physical_error_rate)
        self._min_atom_hit = min_atom_hit_ratio
        self._max_bank_conflict = max_bank_conflict_rate
        self._max_res_size = max_residual_component_size

    def score(self, candidate) -> float:
        out = self._compiler.compile(code=candidate.code, geometry=candidate.geometry,
                                     schedule=candidate.schedule)
        cost = out.cost
        if not cost.passes_thresholds(self._min_atom_hit, self._max_bank_conflict,
                                      self._max_res_size):
            return float("inf")
        w = self._weights
        return (w.mean_latency * cost.predicted_mean_latency_ns
                + w.p99_latency * cost.predicted_p99_latency_ns
                + w.memory_traffic * cost.predicted_bram_traffic
                + w.residual_solve_rate * cost.predicted_residual_solve_rate * 1e6
                + w.cross_tile_traffic * cost.predicted_cross_tile_traffic)
