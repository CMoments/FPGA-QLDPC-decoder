from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FaultAtom:
    atom_id: str
    detector_signature: frozenset
    data_error_candidates: tuple
    prior_weight: float
    tile: tuple
    time_slice: int
    window_size: int

    def __post_init__(self):
        object.__setattr__(self, 'detector_signature', frozenset(self.detector_signature))
        object.__setattr__(self, 'data_error_candidates', tuple(self.data_error_candidates))
        object.__setattr__(self, 'tile', tuple(self.tile))


class FaultAtomLibrary:
    def __init__(self):
        self._by_signature: dict[frozenset, list[FaultAtom]] = {}

    def add(self, atom: FaultAtom) -> None:
        sig = atom.detector_signature
        if sig not in self._by_signature:
            self._by_signature[sig] = []
        self._by_signature[sig].append(atom)

    def lookup(self, signature) -> FaultAtom | None:
        """Return the highest-weight atom for this signature, or None."""
        candidates = self._by_signature.get(frozenset(signature))
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.prior_weight)

    def coverage(self, total_events: int, hit_events: int) -> float:
        if total_events == 0:
            return 0.0
        return hit_events / total_events

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_signature.values())
