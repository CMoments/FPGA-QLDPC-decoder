import numpy as np
from qldpc.codes import QLDPCCode


class FixedExtractionBaseline:
    """Monte Carlo baseline: random bit-flip noise, greedy single-qubit correction."""
    def __init__(self, physical_error_rate: float, num_shots: int):
        self.p = physical_error_rate
        self.num_shots = num_shots

    def logical_error_rate(self, code: QLDPCCode) -> float:
        rng = np.random.default_rng(42)
        logical_errors = 0
        for _ in range(self.num_shots):
            error = rng.random(code.n) < self.p
            syndrome_x = (code.Hx @ error.astype(np.uint8)) % 2
            correction = np.zeros(code.n, dtype=np.uint8)
            for check_idx, bit in enumerate(syndrome_x):
                if bit:
                    support = np.where(code.Hx[check_idx])[0]
                    if len(support):
                        correction[support[0]] ^= 1
            residual = (error.astype(np.uint8) ^ correction) % 2
            logical_errors += int(np.any(residual))
        return logical_errors / self.num_shots
