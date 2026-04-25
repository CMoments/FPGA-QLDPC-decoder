from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class QLDPCCode:
    Hx: NDArray[np.uint8]
    Hz: NDArray[np.uint8]

    @property
    def n(self) -> int:
        return self.Hx.shape[1]

    @property
    def k(self) -> int:
        rx = int(np.linalg.matrix_rank(self.Hx.astype(float)))
        rz = int(np.linalg.matrix_rank(self.Hz.astype(float)))
        return self.n - rx - rz

    @property
    def weight_x(self) -> int:
        return int(self.Hx.sum(axis=1).max())

    @property
    def weight_z(self) -> int:
        return int(self.Hz.sum(axis=1).max())


def _cyclic_shift(n: int, shift: int) -> NDArray[np.uint8]:
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + shift) % n] = 1
    return S


def _bb_poly(l: int, m: int, exponents: list) -> NDArray[np.uint8]:
    result = np.zeros((l * m, l * m), dtype=np.uint8)
    for e in exponents:
        el, em = e % l, e % m
        result = (result + np.kron(_cyclic_shift(l, el), _cyclic_shift(m, em))) % 2
    return result


class BivariateBicycleCode(QLDPCCode):
    def __init__(self, l: int, m: int, a: list, b: list):
        A = _bb_poly(l, m, a)
        B = _bb_poly(l, m, b)
        Hx = np.hstack([A, B])
        Hz = np.hstack([B.T, A.T])
        super().__init__(
            Hx=(Hx % 2).astype(np.uint8),
            Hz=(Hz % 2).astype(np.uint8),
        )


class HypergraphProductCode(QLDPCCode):
    def __init__(self, H1: NDArray[np.uint8], H2: NDArray[np.uint8]):
        r1, n1 = H1.shape
        r2, n2 = H2.shape
        In1 = np.eye(n1, dtype=np.uint8)
        In2 = np.eye(n2, dtype=np.uint8)
        Ir1 = np.eye(r1, dtype=np.uint8)
        Ir2 = np.eye(r2, dtype=np.uint8)
        Hx = np.hstack([np.kron(H1, In2), np.kron(Ir1, H2.T)])
        Hz = np.hstack([np.kron(In1, H2), np.kron(H1.T, Ir2)])
        super().__init__(
            Hx=(Hx % 2).astype(np.uint8),
            Hz=(Hz % 2).astype(np.uint8),
        )
