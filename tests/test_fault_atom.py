# tests/test_fault_atom.py
import pytest
from compiler.fault_atom import FaultAtom, FaultAtomLibrary

def test_fault_atom_fields():
    atom = FaultAtom(
        atom_id="ancilla_reuse_0",
        detector_signature=frozenset([0, 1]),
        data_error_candidates=[frozenset([2]), frozenset([3])],
        prior_weight=0.9,
        tile=(0, 0),
        time_slice=1,
        window_size=2,
    )
    assert atom.atom_id == "ancilla_reuse_0"
    assert 0 in atom.detector_signature
    assert atom.prior_weight == 0.9

def test_fault_atom_library_add_and_lookup():
    lib = FaultAtomLibrary()
    atom = FaultAtom(
        atom_id="hook_0",
        detector_signature=frozenset([4, 5]),
        data_error_candidates=[frozenset([6])],
        prior_weight=0.8,
        tile=(1, 0),
        time_slice=0,
        window_size=1,
    )
    lib.add(atom)
    result = lib.lookup(frozenset([4, 5]))
    assert result is not None
    assert result.atom_id == "hook_0"

def test_fault_atom_library_miss_returns_none():
    lib = FaultAtomLibrary()
    assert lib.lookup(frozenset([99, 100])) is None

def test_fault_atom_library_coverage():
    lib = FaultAtomLibrary()
    for i in range(5):
        lib.add(FaultAtom(
            atom_id=f"a{i}",
            detector_signature=frozenset([i]),
            data_error_candidates=[frozenset()],
            prior_weight=1.0,
            tile=(0,0), time_slice=i, window_size=1,
        ))
    total = 10
    hits = 5
    assert lib.coverage(total_events=total, hit_events=hits) == 0.5
