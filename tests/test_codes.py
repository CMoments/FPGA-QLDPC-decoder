# tests/test_codes.py
import numpy as np
import pytest
from qldpc.codes import BivariateBicycleCode, HypergraphProductCode, _gf2_rank


def test_bb_code_parity_check_dimensions():
    code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
    assert code.Hx.shape[1] == code.n
    assert code.Hz.shape[1] == code.n
    assert code.Hx.shape[0] == code.n // 2
    assert code.Hz.shape[0] == code.n // 2


def test_bb_code_commutativity():
    code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
    product = (code.Hx @ code.Hz.T) % 2
    assert np.all(product == 0), "Hx Hz^T must be zero mod 2"


def test_hgp_code_dimensions():
    H1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    H2 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    code = HypergraphProductCode(H1, H2)
    assert code.Hx.shape[1] == code.n
    assert code.Hz.shape[1] == code.n


def test_hgp_commutativity():
    H1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    H2 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    code = HypergraphProductCode(H1, H2)
    product = (code.Hx @ code.Hz.T) % 2
    assert np.all(product == 0)


def test_gf2_rank_identity():
    I = np.eye(4, dtype=np.uint8)
    assert _gf2_rank(I) == 4


def test_gf2_rank_singular():
    M = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    assert _gf2_rank(M) == 2


def test_k_is_nonnegative():
    code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
    assert code.k >= 0
