from dataclasses import dataclass


@dataclass(frozen=True)
class FaultAtom:
    atom_id: str
    detector_signature: frozenset
    data_error_candidates: tuple
    prior_weight: float
    tile: tuple
    time_slice: int
    window_size: int

    def __init__(self, atom_id, detector_signature, data_error_candidates,
                 prior_weight, tile, time_slice, window_size):
        object.__setattr__(self, 'atom_id', atom_id)
        object.__setattr__(self, 'detector_signature', frozenset(detector_signature))
        object.__setattr__(self, 'data_error_candidates', tuple(data_error_candidates))
        object.__setattr__(self, 'prior_weight', prior_weight)
        object.__setattr__(self, 'tile', tuple(tile))
        object.__setattr__(self, 'time_slice', time_slice)
        object.__setattr__(self, 'window_size', window_size)


class FaultAtomLibrary:
    def __init__(self):
        self._by_signature = {}

    def add(self, atom: FaultAtom) -> None:
        self._by_signature[atom.detector_signature] = atom

    def lookup(self, signature) -> FaultAtom | None:
        return self._by_signature.get(frozenset(signature))

    def coverage(self, total_events: int, hit_events: int) -> float:
        if total_events == 0:
            return 0.0
        return hit_events / total_events

    def __len__(self) -> int:
        return len(self._by_signature)
