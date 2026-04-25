from dataclasses import dataclass


@dataclass
class TileCost:
    tile: tuple
    memory_pressure: float
    atom_library_size: int
    local_conflict_density: float


@dataclass
class HardwareCostIR:
    tile_costs: list
    predicted_mean_latency_ns: float
    predicted_p99_latency_ns: float
    predicted_bram_traffic: int
    predicted_bank_conflict_rate: float
    predicted_cross_tile_traffic: int
    predicted_atom_hit_ratio: float
    predicted_residual_solve_rate: float
    residual_component_histogram: dict

    def passes_thresholds(
        self,
        min_atom_hit_ratio: float,
        max_bank_conflict_rate: float,
        max_residual_component_size: int,
    ) -> bool:
        if self.predicted_atom_hit_ratio < min_atom_hit_ratio:
            return False
        if self.predicted_bank_conflict_rate > max_bank_conflict_rate:
            return False
        if self.residual_component_histogram:
            if max(self.residual_component_histogram.keys()) > max_residual_component_size:
                return False
        return True
