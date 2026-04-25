from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from compiler.fault_atom import FaultAtomLibrary
from compiler.residual_graph import ResidualGraph
from fpga.atom_matcher import AtomMatcherArray
from fpga.residual_engine import ResidualEliminationEngine


@dataclass
class ShotResult:
    correction: frozenset
    latency_ns: float
    atom_hit: bool


class FPGAPipeline:
    def __init__(self, atom_library: FaultAtomLibrary, residual_graph: ResidualGraph,
                 ns_per_atom_match: float, ns_per_residual_node: float):
        self._matcher = AtomMatcherArray(atom_library)
        self._engine = ResidualEliminationEngine(residual_graph, ns_per_residual_node)
        self._ns_per_atom = ns_per_atom_match
        self._residual_graph = residual_graph

    def decode(self, detector_events: frozenset) -> ShotResult:
        atom = self._matcher.match(detector_events)
        if atom is not None:
            candidates = atom.data_error_candidates
            correction = candidates[0] if candidates else frozenset()
            return ShotResult(correction=correction, latency_ns=self._ns_per_atom,
                              atom_hit=True)
        # per-shot residual: only nodes present in this shot's unmatched detectors
        active = frozenset(
            d for d in detector_events
            if d in self._residual_graph._g.nodes
        )
        correction, latency = self._engine.solve(active_nodes=active)
        return ShotResult(correction=correction, latency_ns=latency, atom_hit=False)

    def batch_decode(self, shots: list) -> dict:
        results = [self.decode(s) for s in shots]
        latencies = [r.latency_ns for r in results]
        hits = sum(1 for r in results if r.atom_hit)
        return {
            "mean_latency_ns": float(np.mean(latencies)),
            "p99_latency_ns": float(np.percentile(latencies, 99)),
            "atom_hit_rate": hits / len(results) if results else 0.0,
        }
