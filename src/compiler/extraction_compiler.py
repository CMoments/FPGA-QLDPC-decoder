from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import stim
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
        circuit = _build_stim_circuit(code, schedule, self.p)
        dem = circuit.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True
        )
        lib = self._build_atom_library(dem, code, geometry, schedule)
        residual = self._build_residual_graph(dem, lib, code, geometry, schedule)
        cost = self._estimate_cost(lib, residual, dem, schedule)
        return CompilerOutput(atom_library=lib, residual_graph=residual, cost=cost)

    def _build_atom_library(self, dem: stim.DetectorErrorModel,
                            code: QLDPCCode, geometry: RoutedGeometry,
                            schedule: ExtractionSchedule) -> FaultAtomLibrary:
        lib = FaultAtomLibrary()
        rows, cols = geometry.layout.rows, geometry.layout.cols
        n_checks = code.Hx.shape[0]
        atom_idx = 0
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            prob = instruction.args_copy()[0]
            det_targets = [
                t.val for t in instruction.targets_copy()
                if t.is_relative_detector_id()
            ]
            if not det_targets:
                continue
            sig = frozenset(det_targets)
            # derive tile from first detector index
            first_det = det_targets[0]
            tile_r = (first_det // cols) % rows
            tile_c = first_det % cols
            # derive time slice from schedule depth
            time_sl = first_det % schedule.depth if schedule.depth > 0 else 0
            data_err = frozenset([first_det % code.n])
            atom = FaultAtom(
                atom_id=f"atom_{atom_idx}",
                detector_signature=sig,
                data_error_candidates=[data_err],
                prior_weight=prob,
                tile=(tile_r, tile_c),
                time_slice=time_sl,
                window_size=len(det_targets),
            )
            lib.add(atom)
            atom_idx += 1
        return lib

    def _build_residual_graph(self, dem: stim.DetectorErrorModel,
                              lib: FaultAtomLibrary, code: QLDPCCode,
                              geometry: RoutedGeometry,
                              schedule: ExtractionSchedule) -> ResidualGraph:
        """Build residual graph from DEM edges whose signatures are NOT in the atom library."""
        g = ResidualGraph()
        rows, cols = geometry.layout.rows, geometry.layout.cols
        added_nodes: set[int] = set()
        edge_idx = 0
        for instruction in dem.flattened():
            if instruction.type != "error":
                continue
            det_targets = [
                t.val for t in instruction.targets_copy()
                if t.is_relative_detector_id()
            ]
            if not det_targets:
                continue
            sig = frozenset(det_targets)
            if lib.lookup(sig) is not None:
                continue  # covered by an atom — not residual
            prob = instruction.args_copy()[0]
            for d in det_targets:
                if d not in added_nodes:
                    tile = ((d // cols) % rows, d % cols)
                    time_sl = d % schedule.depth if schedule.depth > 0 else 0
                    g.add_node(ResidualNode(node_id=d, tile=tile, time_slice=time_sl))
                    added_nodes.add(d)
            for i in range(len(det_targets) - 1):
                u, v = det_targets[i], det_targets[i + 1]
                tile_u = ((u // cols) % rows, u % cols)
                tile_v = ((v // cols) % rows, v % cols)
                cross = not geometry.is_local(tile_u, tile_v)
                g.add_edge(ResidualEdge(u=u, v=v, weight=prob, cross_tile=cross))
            edge_idx += 1
        return g

    def _estimate_cost(self, lib: FaultAtomLibrary, residual: ResidualGraph,
                       dem: stim.DetectorErrorModel,
                       schedule: ExtractionSchedule) -> HardwareCostIR:
        total_dem_errors = sum(
            1 for inst in dem.flattened() if inst.type == "error"
        )
        n_atoms = len(lib)
        n_res = residual.num_nodes
        # atom_hit_ratio: fraction of DEM error mechanisms covered by atoms
        atom_hit_ratio = n_atoms / total_dem_errors if total_dem_errors > 0 else 0.0
        atom_hit_ratio = min(1.0, atom_hit_ratio)
        comp_sizes = residual.component_sizes() or [0]
        histogram: dict[int, int] = {}
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


def _build_stim_circuit(code: QLDPCCode, schedule: ExtractionSchedule,
                        p: float) -> stim.Circuit:
    """Build a two-round stim memory circuit for the code's X-checks."""
    from baselines.bp_decoder import _build_memory_circuit
    return _build_memory_circuit(code, p, rounds=max(2, schedule.depth))
