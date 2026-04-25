from dataclasses import dataclass
from enum import Enum


class AncillaReusePolicy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class ScheduleStep:
    time: int
    ancilla_group: int
    detectors: list


@dataclass
class ExtractionSchedule:
    steps: list
    ancilla_reuse: AncillaReusePolicy

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def num_ancilla_groups(self) -> int:
        return len({s.ancilla_group for s in self.steps})

    def is_feasible(self, max_depth: int) -> bool:
        return self.depth <= max_depth
