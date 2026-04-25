from dataclasses import dataclass
from qldpc.codes import QLDPCCode
from qldpc.geometry import RoutedGeometry
from qldpc.schedule import ExtractionSchedule

@dataclass
class Candidate:
    code: QLDPCCode
    geometry: RoutedGeometry
    schedule: ExtractionSchedule
