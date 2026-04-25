from compiler.fault_atom import FaultAtom, FaultAtomLibrary


class AtomMatcherArray:
    def __init__(self, library: FaultAtomLibrary):
        self._lib = library

    def match(self, detector_events: frozenset) -> FaultAtom | None:
        return self._lib.lookup(detector_events)
