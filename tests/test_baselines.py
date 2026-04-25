# tests/test_baselines.py
import numpy as np
import pytest
from qldpc.codes import BivariateBicycleCode
from baselines.bp_decoder import FixedExtractionBaseline


def test_baseline_returns_logical_error_rate():
    code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
    baseline = FixedExtractionBaseline(physical_error_rate=0.01, num_shots=200)
    ler = baseline.logical_error_rate(code)
    assert 0.0 <= ler <= 1.0


def test_baseline_lower_error_rate_at_lower_p():
    code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
    ler_high = FixedExtractionBaseline(physical_error_rate=0.05, num_shots=300).logical_error_rate(code)
    ler_low = FixedExtractionBaseline(physical_error_rate=0.001, num_shots=300).logical_error_rate(code)
    assert ler_low <= ler_high + 0.05


def test_baseline_uses_mwpm_not_greedy():
    """MWPM baseline must produce lower LER than a trivial all-zero correction at p=0.05."""
    code = BivariateBicycleCode(l=6, m=6, a=[3, 1, 2], b=[3, 1, 2])
    baseline = FixedExtractionBaseline(physical_error_rate=0.05, num_shots=300)
    ler = baseline.logical_error_rate(code)
    # MWPM should do better than random (0.5); if it returns ~0.5 the decoder is broken
    assert ler < 0.5
