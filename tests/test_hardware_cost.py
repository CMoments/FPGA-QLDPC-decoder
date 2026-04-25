# tests/test_hardware_cost.py
import pytest
from compiler.hardware_cost import HardwareCostIR, TileCost

def test_hardware_cost_ir_fields():
    tile_costs = [TileCost(tile=(0,0), memory_pressure=0.4, atom_library_size=12,
                           local_conflict_density=0.1)]
    cost = HardwareCostIR(
        tile_costs=tile_costs,
        predicted_mean_latency_ns=200.0,
        predicted_p99_latency_ns=800.0,
        predicted_bram_traffic=1024,
        predicted_bank_conflict_rate=0.05,
        predicted_cross_tile_traffic=256,
        predicted_atom_hit_ratio=0.85,
        predicted_residual_solve_rate=0.15,
        residual_component_histogram={1: 10, 2: 3, 3: 1},
    )
    assert cost.predicted_atom_hit_ratio == 0.85
    assert cost.tile_costs[0].tile == (0, 0)

def test_hardware_cost_passes_thresholds():
    cost = HardwareCostIR(
        tile_costs=[],
        predicted_mean_latency_ns=200.0,
        predicted_p99_latency_ns=800.0,
        predicted_bram_traffic=1024,
        predicted_bank_conflict_rate=0.05,
        predicted_cross_tile_traffic=256,
        predicted_atom_hit_ratio=0.85,
        predicted_residual_solve_rate=0.15,
        residual_component_histogram={1: 10},
    )
    assert cost.passes_thresholds(
        min_atom_hit_ratio=0.80,
        max_bank_conflict_rate=0.10,
        max_residual_component_size=4,
    ) is True

def test_hardware_cost_fails_atom_hit_threshold():
    cost = HardwareCostIR(
        tile_costs=[],
        predicted_mean_latency_ns=200.0,
        predicted_p99_latency_ns=800.0,
        predicted_bram_traffic=1024,
        predicted_bank_conflict_rate=0.05,
        predicted_cross_tile_traffic=256,
        predicted_atom_hit_ratio=0.60,
        predicted_residual_solve_rate=0.40,
        residual_component_histogram={1: 5},
    )
    assert cost.passes_thresholds(
        min_atom_hit_ratio=0.80,
        max_bank_conflict_rate=0.10,
        max_residual_component_size=4,
    ) is False
