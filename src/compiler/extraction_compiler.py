from dataclasses import dataclass
import numpy as np
from qldpc.codes import QLDPCCode
from qldpc.geometry import RoutedGeometry
from qldpc.schedule import ExtractionSchedule
from compiler.fault_atom import FaultAtom, FaultAtomLibrary
from compiler.residual_graph import ResidualGraph, ResidualNode, ResidualEdge
from compiler.hardware_cost import HardwareCostIR, TileCost


@dataclass
class CompilerOutput:
    atom_library: FaultAtomLibrary
    residual_graph: ResidualGraph
    cost: HardwareCostIR


class ExtractionCompiler:
    def __init__(self, physical_error_rate: float):
        self.p = physical_error_rate

    def compile(self, code: QLDPCCode, geometry: RoutedGeometry,
                schedule: ExtractionSchedule) -> CompilerOutput:
        lib = self._build_atom_library(code, geometry, schedule)
        residual = self._build_residual_graph(code, geometry, schedule, lib)
        cost = self._estimate_cost(lib, residual, schedule)
        return CompilerOutput(atom_library=lib, residual_graph=residual, cost=cost)

    def _build_atom_library(self, code, geometry, schedule) -> FaultAtomLibrary:
        lib = FaultAtomLibrary()
        rows, cols = geometry.layout.rows, geometry.layout.cols
        atom_idx = 0
        for step in schedule.steps:
            tile_r = (atom_idx // cols) % rows
            tile_c = atom_idx % cols
            for det in step.detectors:
                sig = frozenset([det, (det + 1) % code.Hx.shape[0]])
                atom = FaultAtom(
                    atom_id=f"atom_{atom_idx}",
                    detector_signature=sig,
                    data_error_candidates=[frozenset([det % code.n])],
                    prior_weight=1.0 - self.p,
                    tile=(tile_r, tile_c),
                    time_slice=step.time,
                    window_size=step.ancilla_group + 1,
                )
                lib.add(atom)
                atom_idx += 1
        return lib

    def _build_residual_graph(self, code, geometry, schedule, lib) -> ResidualGraph:
        g = ResidualGraph()
        n_residual = max(1, code.Hx.shape[0] // 4)
        for i in range(n_residual):
            tile = (i % geometry.layout.rows, i % geometry.layout.cols)
            g.add_node(ResidualNode(node_id=i, tile=tile,
                                    time_slice=i % schedule.depth))
        for i in range(n_residual - 1):
            cross = (i % geometry.layout.cols == geometry.layout.cols - 1)
            g.add_edge(ResidualEdge(u=i, v=i+1, weight=self.p, cross_tile=cross))
        return g

    def _estimate_cost(self, lib, residual, schedule) -> HardwareCostIR:
        n_atoms = len(lib)
        n_res = residual.num_nodes
        total = n_atoms + n_res if (n_atoms + n_res) > 0 else 1
        atom_hit_ratio = n_atoms / total
        comp_sizes = residual.component_sizes() or [0]
        histogram = {}
        for s in comp_sizes:
            histogram[s] = histogram.get(s, 0) + 1
        return HardwareCostIR(
            tile_costs=[],
            predicted_mean_latency_ns=50.0 * n_res + 10.0,
            predicted_p99_latency_ns=200.0 * n_res + 50.0,
            predicted_bram_traffic=n_atoms * 64,
            predicted_bank_conflict_rate=min(0.5, self.p * 10),
            predicted_cross_tile_traffic=residual.cross_tile_edge_count() * 8,
            predicted_atom_hit_ratio=atom_hit_ratio,
            predicted_residual_solve_rate=1.0 - atom_hit_ratio,
            residual_component_histogram=histogram,
        )
