from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu


@dataclass
class QLDPCCode:
    Hx: NDArray[np.uint8]
    Hz: NDArray[np.uint8]

    @property
    def n(self) -> int:
        return self.Hx.shape[1]

    @property
    def k(self) -> int:
        rx = _gf2_rank(self.Hx)
        rz = _gf2_rank(self.Hz)
        return self.n - rx - rz

    @property
    def weight_x(self) -> int:
        return int(self.Hx.sum(axis=1).max())

    @property
    def weight_z(self) -> int:
        return int(self.Hz.sum(axis=1).max())


def _gf2_rank(H: NDArray[np.uint8]) -> int:
    """Gaussian elimination over GF(2) to compute matrix rank."""
    M = H.copy().astype(np.uint8)
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if M[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(rows):
            if row != rank and M[row, col]:
                M[row] ^= M[rank]
        rank += 1
    return rank


def _cyclic_shift(n: int, shift: int) -> NDArray[np.uint8]:
    return np.roll(np.eye(n, dtype=np.uint8), shift, axis=1)


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
