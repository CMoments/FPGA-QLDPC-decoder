from __future__ import annotations
import numpy as np
import stim
import pymatching
from qldpc.codes import QLDPCCode


class FixedExtractionBaseline:
    """MWPM baseline using stim circuit-level noise and pymatching."""

    def __init__(self, physical_error_rate: float, num_shots: int):
        self.p = physical_error_rate
        self.num_shots = num_shots

    def logical_error_rate(self, code: QLDPCCode) -> float:
        circuit = _build_memory_circuit(code, self.p, rounds=3)
        dem = circuit.detector_error_model(
            decompose_errors=True, ignore_decomposition_failures=True
        )
        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        det_samples, obs_samples = sampler.sample(
            shots=self.num_shots, separate_observables=True
        )
        predictions = matcher.decode_batch(det_samples)
        errors = np.any(predictions != obs_samples, axis=1)
        return float(errors.mean())


def _build_memory_circuit(code: QLDPCCode, p: float, rounds: int = 3) -> stim.Circuit:
    """
    Build a stim memory-experiment circuit for the X-checks of `code`.

    Layout: data qubits 0..n-1, ancilla qubits n..n+n_checks-1.
    Each round: reset ancillas, apply X-check stabilisers, measure ancillas.
    Detectors compare each ancilla measurement to the previous round.
    Final round: measure data qubits; observable is the first logical-Z operator.
    """
    n = code.n
    n_checks = code.Hx.shape[0]
    anc = list(range(n, n + n_checks))
    data = list(range(n))

    circuit = stim.Circuit()
    # initialise data in |+> (X-basis memory) and ancillas in |0>
    circuit.append("H", data)
    circuit.append("R", anc)

    for rnd in range(rounds):
        # noise on data qubits
        circuit.append("DEPOLARIZE1", data, p)
        # X-check stabiliser circuits
        for ci in range(n_checks):
            support = list(np.where(code.Hx[ci])[0])
            a = n + ci
            circuit.append("H", [a])
            for q in support:
                circuit.append("CNOT", [a, q])
            circuit.append("H", [a])
            circuit.append("DEPOLARIZE1", [a], p)
        # measure-and-reset ancillas
        circuit.append("MR", anc)
        if rnd == 0:
            # first round: detectors are raw measurements (all-zero in noiseless case)
            for ci in range(n_checks):
                circuit.append("DETECTOR", [stim.target_rec(ci - n_checks)])
        else:
            # subsequent rounds: detector = XOR with previous round
            for ci in range(n_checks):
                circuit.append("DETECTOR",
                               [stim.target_rec(ci - n_checks),
                                stim.target_rec(ci - 2 * n_checks)])

    # final data measurement in X basis
    circuit.append("H", data)
    circuit.append("M", data)
    # observable: logical X = first row of Hx (X-basis memory)
    obs_support = list(np.where(code.Hx[0])[0])
    obs_targets = [stim.target_rec(q - n) for q in obs_support]
    circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
    return circuit
