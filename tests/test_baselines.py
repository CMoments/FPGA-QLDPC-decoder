# tests/test_baselines.py
import numpy as np
import pytest
from qldpc.codes import BivariateBicycleCode
from baselines.bp_decoder import FixedExtractionBaseline

def test_baseline_returns_logical_error_rate():
    code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
    baseline = FixedExtractionBaseline(physical_error_rate=0.01, num_shots=200)
    ler = baseline.logical_error_rate(code)
    assert 0.0 <= ler <= 1.0

def test_baseline_lower_error_rate_at_lower_p():
    code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
    b_high = FixedExtractionBaseline(physical_error_rate=0.05, num_shots=200)
    b_low  = FixedExtractionBaseline(physical_error_rate=0.001, num_shots=200)
    ler_high = b_high.logical_error_rate(code)
    ler_low  = b_low.logical_error_rate(code)
    assert ler_low <= ler_high + 0.1
