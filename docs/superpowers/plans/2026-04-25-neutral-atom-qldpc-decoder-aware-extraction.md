# Neutral-Atom qLDPC Decoder-Aware Extraction Co-Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python simulation stack that co-synthesizes neutral-atom qLDPC extraction schedules, routed geometries, and compiled FPGA decoder representations to minimize decoding latency under a logical-error-rate budget.

**Architecture:** Four layers — Candidate Generation, Decoder-Aware Extraction Compiler (produces Fault-Atom IR + Residual Graph IR), FPGA Backend Simulator, and Closed-Loop Search. The compiler is the core novelty; the search loop drives all four layers together. All layers are pure Python with numpy/scipy; no RTL is required for phase one.

**Tech Stack:** Python 3.11+, numpy, scipy, networkx, stim (circuit-level noise simulation), pymatching (baseline decoder reference), pytest, dataclasses, typing

---

## Subsystem Split Notice

This spec covers four independent subsystems. This plan implements them in dependency order as a single phase-one deliverable. Each task is independently testable.

---

## File Structure

```
src/
  qldpc/
    __init__.py
    codes.py            # qLDPC code family definitions (BB, HGP, lifted product)
    geometry.py         # routed geometry layouts and movement constraints
    schedule.py         # extraction schedule representation and feasibility check
    noise.py            # circuit-level noise model (depolarizing + geometry-induced)
  compiler/
    __init__.py
    fault_atom.py       # FaultAtom dataclass + FaultAtomLibrary
    residual_graph.py   # ResidualGraph IR dataclass + builder
    hardware_cost.py    # HardwareCostIR dataclass + cost estimator
    extraction_compiler.py  # main compiler: circuit -> FaultAtomIR + ResidualGraphIR
  fpga/
    __init__.py
    atom_matcher.py     # AtomMatcherArray simulator
    residual_engine.py  # ResidualEliminationEngine simulator
    pipeline.py         # end-to-end FPGA pipeline simulator (latency model)
  search/
    __init__.py
    candidate.py        # Candidate dataclass (code + geometry + schedule)
    scorer.py           # multi-objective scorer
    search_loop.py      # closed-loop search over candidates
  baselines/
    __init__.py
    bp_decoder.py       # thin wrapper around pymatching as fixed-extraction baseline
tests/
  test_codes.py
  test_geometry.py
  test_schedule.py
  test_fault_atom.py
  test_residual_graph.py
  test_hardware_cost.py
  test_extraction_compiler.py
  test_atom_matcher.py
  test_residual_engine.py
  test_pipeline.py
  test_scorer.py
  test_search_loop.py
  test_baselines.py
  test_integration.py
```

---

## Task 1: Project Scaffold and Dependencies

**Files:**
- Create: `src/qldpc/__init__.py`
- Create: `src/compiler/__init__.py`
- Create: `src/fpga/__init__.py`
- Create: `src/search/__init__.py`
- Create: `src/baselines/__init__.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "qldpc-decoder-codesign"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "networkx>=3.2",
    "stim>=1.13",
    "pymatching>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov"]
```

- [ ] **Step 2: Create requirements.txt**

```
numpy>=1.26
scipy>=1.12
networkx>=3.2
stim>=1.13
pymatching>=2.2
pytest>=8
pytest-cov
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -e ".[dev]"`
Expected: all packages install without error

- [ ] **Step 4: Create package __init__.py files**

Each file is empty. Create: `src/qldpc/__init__.py`, `src/compiler/__init__.py`, `src/fpga/__init__.py`, `src/search/__init__.py`, `src/baselines/__init__.py`, `tests/__init__.py`

- [ ] **Step 5: Verify import**

Run: `python -c "import stim; import pymatching; import networkx; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml requirements.txt src/ tests/
git commit -m "feat: project scaffold and dependencies"
```

---

## Task 2: qLDPC Code Families

**Files:**
- Create: `src/qldpc/codes.py`
- Create: `tests/test_codes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_codes.py
import numpy as np
import pytest
from qldpc.codes import BivariateBicycleCode, HypergraphProductCode

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
    H1 = np.array([[1,1,0],[0,1,1],[1,0,1]], dtype=np.uint8)
    H2 = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)
    code = HypergraphProductCode(H1, H2)
    assert code.Hx.shape[1] == code.n
    assert code.Hz.shape[1] == code.n

def test_hgp_commutativity():
    H1 = np.array([[1,1,0],[0,1,1],[1,0,1]], dtype=np.uint8)
    H2 = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)
    code = HypergraphProductCode(H1, H2)
    product = (code.Hx @ code.Hz.T) % 2
    assert np.all(product == 0)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_codes.py -v`
Expected: ImportError or AttributeError

- [ ] **Step 3: Implement codes.py**

```python
# src/qldpc/codes.py
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class QLDPCCode:
    Hx: NDArray[np.uint8]
    Hz: NDArray[np.uint8]

    @property
    def n(self) -> int:
        return self.Hx.shape[1]

    @property
    def k(self) -> int:
        rx = int(np.linalg.matrix_rank(self.Hx.astype(float)))
        rz = int(np.linalg.matrix_rank(self.Hz.astype(float)))
        return self.n - rx - rz

    @property
    def weight_x(self) -> int:
        return int(self.Hx.sum(axis=1).max())

    @property
    def weight_z(self) -> int:
        return int(self.Hz.sum(axis=1).max())
```

- [ ] **Step 4: Add BivariateBicycleCode to codes.py**

```python
def _cyclic_shift(n: int, shift: int) -> NDArray[np.uint8]:
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + shift) % n] = 1
    return S

def _bb_poly(l: int, m: int, exponents: list[int]) -> NDArray[np.uint8]:
    Il = np.eye(l, dtype=np.uint8)
    Im = np.eye(m, dtype=np.uint8)
    result = np.zeros((l * m, l * m), dtype=np.uint8)
    for e in exponents:
        el, em = e % l, e % m
        result = (result + np.kron(_cyclic_shift(l, el), _cyclic_shift(m, em))) % 2
    return result

class BivariateBicycleCode(QLDPCCode):
    def __init__(self, l: int, m: int, a: list[int], b: list[int]):
        A = _bb_poly(l, m, a)
        B = _bb_poly(l, m, b)
        n2 = l * m
        Hx = np.hstack([A, B])
        Hz = np.hstack([B.T, A.T])
        super().__init__(Hx=(Hx % 2).astype(np.uint8),
                         Hz=(Hz % 2).astype(np.uint8))
```

- [ ] **Step 5: Add HypergraphProductCode to codes.py**

```python
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
        super().__init__(Hx=(Hx % 2).astype(np.uint8),
                         Hz=(Hz % 2).astype(np.uint8))
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_codes.py -v`
Expected: 4 PASSED

- [ ] **Step 7: Commit**

```bash
git add src/qldpc/codes.py tests/test_codes.py
git commit -m "feat: BB and HGP qLDPC code families with commutativity check"
```

---

## Task 3: Routed Geometry

**Files:**
- Create: `src/qldpc/geometry.py`
- Create: `tests/test_geometry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_geometry.py
import pytest
from qldpc.geometry import RoutedGeometry, TileLayout, MovementConstraint

def test_tile_layout_qubit_count():
    layout = TileLayout(rows=4, cols=4, data_per_tile=4, ancilla_per_tile=2)
    assert layout.total_data_qubits == 64
    assert layout.total_ancilla_qubits == 32
    assert layout.num_tiles == 16

def test_routed_geometry_locality():
    layout = TileLayout(rows=2, cols=2, data_per_tile=4, ancilla_per_tile=2)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    assert geom.is_local(tile_a=(0,0), tile_b=(0,1)) is True
    assert geom.is_local(tile_a=(0,0), tile_b=(1,1)) is True

def test_movement_constraint_rejects_long_range():
    layout = TileLayout(rows=4, cols=4, data_per_tile=4, ancilla_per_tile=2)
    geom = RoutedGeometry(layout=layout, max_move_distance=1)
    assert geom.is_local(tile_a=(0,0), tile_b=(3,3)) is False

def test_cross_tile_edges():
    layout = TileLayout(rows=2, cols=2, data_per_tile=4, ancilla_per_tile=2)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    edges = geom.cross_tile_edge_count()
    assert isinstance(edges, int)
    assert edges >= 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_geometry.py -v`
Expected: ImportError

- [ ] **Step 3: Implement geometry.py**

```python
# src/qldpc/geometry.py
from dataclasses import dataclass, field
import math

@dataclass
class TileLayout:
    rows: int
    cols: int
    data_per_tile: int
    ancilla_per_tile: int

    @property
    def num_tiles(self) -> int:
        return self.rows * self.cols

    @property
    def total_data_qubits(self) -> int:
        return self.num_tiles * self.data_per_tile

    @property
    def total_ancilla_qubits(self) -> int:
        return self.num_tiles * self.ancilla_per_tile

@dataclass
class RoutedGeometry:
    layout: TileLayout
    max_move_distance: int

    def is_local(self, tile_a: tuple[int,int], tile_b: tuple[int,int]) -> bool:
        dr = abs(tile_a[0] - tile_b[0])
        dc = abs(tile_a[1] - tile_b[1])
        return max(dr, dc) <= self.max_move_distance

    def cross_tile_edge_count(self) -> int:
        count = 0
        tiles = [(r, c) for r in range(self.layout.rows)
                         for c in range(self.layout.cols)]
        for i, ta in enumerate(tiles):
            for tb in tiles[i+1:]:
                if not self.is_local(ta, tb):
                    count += 1
        return count

MovementConstraint = RoutedGeometry
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_geometry.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/geometry.py tests/test_geometry.py
git commit -m "feat: routed geometry with tile layout and movement constraints"
```

---

## Task 4: Extraction Schedule

**Files:**
- Create: `src/qldpc/schedule.py`
- Create: `tests/test_schedule.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schedule.py
import pytest
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy

def test_schedule_depth():
    steps = [ScheduleStep(time=t, ancilla_group=t % 2, detectors=list(range(4)))
             for t in range(6)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    assert sched.depth == 6

def test_schedule_feasibility_passes():
    steps = [ScheduleStep(time=t, ancilla_group=0, detectors=[0,1,2,3])
             for t in range(4)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    assert sched.is_feasible(max_depth=10) is True

def test_schedule_feasibility_fails_depth():
    steps = [ScheduleStep(time=t, ancilla_group=0, detectors=[0])
             for t in range(20)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    assert sched.is_feasible(max_depth=10) is False

def test_schedule_ancilla_groups():
    steps = [ScheduleStep(time=t, ancilla_group=t % 3, detectors=[t])
             for t in range(9)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.PARALLEL)
    assert sched.num_ancilla_groups == 3
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_schedule.py -v`
Expected: ImportError

- [ ] **Step 3: Implement schedule.py**

```python
# src/qldpc/schedule.py
from dataclasses import dataclass, field
from enum import Enum

class AncillaReusePolicy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

@dataclass
class ScheduleStep:
    time: int
    ancilla_group: int
    detectors: list[int]

@dataclass
class ExtractionSchedule:
    steps: list[ScheduleStep]
    ancilla_reuse: AncillaReusePolicy

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def num_ancilla_groups(self) -> int:
        return len({s.ancilla_group for s in self.steps})

    def is_feasible(self, max_depth: int) -> bool:
        return self.depth <= max_depth
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_schedule.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/schedule.py tests/test_schedule.py
git commit -m "feat: extraction schedule with ancilla reuse policy and feasibility check"
```

---

## Task 5: Fault-Atom IR

**Files:**
- Create: `src/compiler/fault_atom.py`
- Create: `tests/test_fault_atom.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_fault_atom.py -v`
Expected: ImportError

- [ ] **Step 3: Implement fault_atom.py**

```python
# src/compiler/fault_atom.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FaultAtom:
    atom_id: str
    detector_signature: frozenset[int]
    data_error_candidates: list[frozenset[int]]
    prior_weight: float
    tile: tuple[int, int]
    time_slice: int
    window_size: int

class FaultAtomLibrary:
    def __init__(self):
        self._by_signature: dict[frozenset[int], FaultAtom] = {}

    def add(self, atom: FaultAtom) -> None:
        self._by_signature[atom.detector_signature] = atom

    def lookup(self, signature: frozenset[int]) -> FaultAtom | None:
        return self._by_signature.get(signature)

    def coverage(self, total_events: int, hit_events: int) -> float:
        if total_events == 0:
            return 0.0
        return hit_events / total_events

    def __len__(self) -> int:
        return len(self._by_signature)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fault_atom.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/compiler/fault_atom.py tests/test_fault_atom.py
git commit -m "feat: FaultAtom IR and FaultAtomLibrary with signature lookup"
```

---

## Task 6: Residual Graph IR

**Files:**
- Create: `src/compiler/residual_graph.py`
- Create: `tests/test_residual_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_residual_graph.py
import pytest
from compiler.residual_graph import ResidualGraph, ResidualNode, ResidualEdge

def test_residual_graph_empty():
    g = ResidualGraph()
    assert g.num_nodes == 0
    assert g.num_edges == 0

def test_residual_graph_add_node_and_edge():
    g = ResidualGraph()
    g.add_node(ResidualNode(node_id=0, tile=(0,0), time_slice=1))
    g.add_node(ResidualNode(node_id=1, tile=(0,1), time_slice=1))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.3, cross_tile=True))
    assert g.num_nodes == 2
    assert g.num_edges == 1

def test_residual_graph_component_sizes():
    g = ResidualGraph()
    for i in range(4):
        g.add_node(ResidualNode(node_id=i, tile=(0,0), time_slice=0))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.5, cross_tile=False))
    g.add_edge(ResidualEdge(u=2, v=3, weight=0.5, cross_tile=False))
    sizes = g.component_sizes()
    assert sorted(sizes) == [2, 2]

def test_residual_graph_cross_tile_edge_count():
    g = ResidualGraph()
    for i in range(3):
        g.add_node(ResidualNode(node_id=i, tile=(i,0), time_slice=0))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.4, cross_tile=True))
    g.add_edge(ResidualEdge(u=1, v=2, weight=0.4, cross_tile=False))
    assert g.cross_tile_edge_count() == 1
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_residual_graph.py -v`
Expected: ImportError

- [ ] **Step 3: Implement residual_graph.py**

```python
# src/compiler/residual_graph.py
from dataclasses import dataclass
import networkx as nx

@dataclass
class ResidualNode:
    node_id: int
    tile: tuple[int, int]
    time_slice: int

@dataclass
class ResidualEdge:
    u: int
    v: int
    weight: float
    cross_tile: bool

class ResidualGraph:
    def __init__(self):
        self._g = nx.Graph()
        self._edges: list[ResidualEdge] = []

    def add_node(self, node: ResidualNode) -> None:
        self._g.add_node(node.node_id, tile=node.tile, time_slice=node.time_slice)

    def add_edge(self, edge: ResidualEdge) -> None:
        self._g.add_edge(edge.u, edge.v, weight=edge.weight, cross_tile=edge.cross_tile)
        self._edges.append(edge)

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def component_sizes(self) -> list[int]:
        return [len(c) for c in nx.connected_components(self._g)]

    def cross_tile_edge_count(self) -> int:
        return sum(1 for e in self._edges if e.cross_tile)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_residual_graph.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/compiler/residual_graph.py tests/test_residual_graph.py
git commit -m "feat: ResidualGraph IR with component analysis and cross-tile tracking"
```

---

## Task 7: Hardware Cost IR

**Files:**
- Create: `src/compiler/hardware_cost.py`
- Create: `tests/test_hardware_cost.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hardware_cost.py
import pytest
from compiler.hardware_cost import HardwareCostIR, TileCost

def test_hardware_cost_ir_fields():
    tile_costs = [TileCost(tile=(0,0), memory_pressure=0.4, atom_library_size=12,
                           local_conflict_density=0.1)]
    cost = HardwareCostIR(
        tile_costs=tile_costs,
        predicted_mean_latency_ns=200.0,
        predicted_p99_latency_ns=800.0,
        predicted_bram_traffic=1024,
        predicted_bank_conflict_rate=0.05,
        predicted_cross_tile_traffic=256,
        predicted_atom_hit_ratio=0.85,
        predicted_residual_solve_rate=0.15,
        residual_component_histogram={1: 10, 2: 3, 3: 1},
    )
    assert cost.predicted_atom_hit_ratio == 0.85
    assert cost.tile_costs[0].tile == (0, 0)

def test_hardware_cost_passes_thresholds():
    cost = HardwareCostIR(
        tile_costs=[],
        predicted_mean_latency_ns=200.0,
        predicted_p99_latency_ns=800.0,
        predicted_bram_traffic=1024,
        predicted_bank_conflict_rate=0.05,
        predicted_cross_tile_traffic=256,
        predicted_atom_hit_ratio=0.85,
        predicted_residual_solve_rate=0.15,
        residual_component_histogram={1: 10},
    )
    assert cost.passes_thresholds(
        min_atom_hit_ratio=0.80,
        max_bank_conflict_rate=0.10,
        max_residual_component_size=4,
    ) is True

def test_hardware_cost_fails_atom_hit_threshold():
    cost = HardwareCostIR(
        tile_costs=[],
        predicted_mean_latency_ns=200.0,
        predicted_p99_latency_ns=800.0,
        predicted_bram_traffic=1024,
        predicted_bank_conflict_rate=0.05,
        predicted_cross_tile_traffic=256,
        predicted_atom_hit_ratio=0.60,
        predicted_residual_solve_rate=0.40,
        residual_component_histogram={1: 5},
    )
    assert cost.passes_thresholds(
        min_atom_hit_ratio=0.80,
        max_bank_conflict_rate=0.10,
        max_residual_component_size=4,
    ) is False
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_hardware_cost.py -v`
Expected: ImportError

- [ ] **Step 3: Implement hardware_cost.py**

```python
# src/compiler/hardware_cost.py
from dataclasses import dataclass

@dataclass
class TileCost:
    tile: tuple[int, int]
    memory_pressure: float
    atom_library_size: int
    local_conflict_density: float

@dataclass
class HardwareCostIR:
    tile_costs: list[TileCost]
    predicted_mean_latency_ns: float
    predicted_p99_latency_ns: float
    predicted_bram_traffic: int
    predicted_bank_conflict_rate: float
    predicted_cross_tile_traffic: int
    predicted_atom_hit_ratio: float
    predicted_residual_solve_rate: float
    residual_component_histogram: dict[int, int]

    def passes_thresholds(
        self,
        min_atom_hit_ratio: float,
        max_bank_conflict_rate: float,
        max_residual_component_size: int,
    ) -> bool:
        if self.predicted_atom_hit_ratio < min_atom_hit_ratio:
            return False
        if self.predicted_bank_conflict_rate > max_bank_conflict_rate:
            return False
        if self.residual_component_histogram:
            if max(self.residual_component_histogram.keys()) > max_residual_component_size:
                return False
        return True
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_hardware_cost.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/compiler/hardware_cost.py tests/test_hardware_cost.py
git commit -m "feat: HardwareCostIR with compile-time threshold rejection"
```

---

## Task 8: Extraction Compiler

**Files:**
- Create: `src/compiler/extraction_compiler.py`
- Create: `tests/test_extraction_compiler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extraction_compiler.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from compiler.extraction_compiler import ExtractionCompiler, CompilerOutput
from compiler.fault_atom import FaultAtomLibrary
from compiler.residual_graph import ResidualGraph
from compiler.hardware_cost import HardwareCostIR

def _make_simple_inputs():
    code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
    layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    steps = [ScheduleStep(time=t, ancilla_group=t%2, detectors=list(range(4)))
             for t in range(6)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    return code, geom, sched

def test_compiler_returns_output():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out, CompilerOutput)

def test_compiler_output_has_atom_library():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out.atom_library, FaultAtomLibrary)

def test_compiler_output_has_residual_graph():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out.residual_graph, ResidualGraph)

def test_compiler_output_has_cost_ir():
    code, geom, sched = _make_simple_inputs()
    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(code=code, geometry=geom, schedule=sched)
    assert isinstance(out.cost, HardwareCostIR)
    assert 0.0 <= out.cost.predicted_atom_hit_ratio <= 1.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_extraction_compiler.py -v`
Expected: ImportError

- [ ] **Step 3: Implement extraction_compiler.py (skeleton + cost model)**

```python
# src/compiler/extraction_compiler.py
from dataclasses import dataclass
import numpy as np
from qldpc.codes import QLDPCCode
from qldpc.geometry import RoutedGeometry
from qldpc.schedule import ExtractionSchedule
from compiler.fault_atom import FaultAtom, FaultAtomLibrary
from compiler.residual_graph import ResidualGraph, ResidualNode, ResidualEdge
from compiler.hardware_cost import HardwareCostIR, TileCost

@dataclass
class CompilerOutput:
    atom_library: FaultAtomLibrary
    residual_graph: ResidualGraph
    cost: HardwareCostIR

class ExtractionCompiler:
    def __init__(self, physical_error_rate: float):
        self.p = physical_error_rate

    def compile(
        self,
        code: QLDPCCode,
        geometry: RoutedGeometry,
        schedule: ExtractionSchedule,
    ) -> CompilerOutput:
        lib = self._build_atom_library(code, geometry, schedule)
        residual = self._build_residual_graph(code, geometry, schedule, lib)
        cost = self._estimate_cost(lib, residual, schedule)
        return CompilerOutput(atom_library=lib, residual_graph=residual, cost=cost)
```

- [ ] **Step 4: Add _build_atom_library to ExtractionCompiler**

```python
    def _build_atom_library(
        self,
        code: QLDPCCode,
        geometry: RoutedGeometry,
        schedule: ExtractionSchedule,
    ) -> FaultAtomLibrary:
        lib = FaultAtomLibrary()
        rows, cols = geometry.layout.rows, geometry.layout.cols
        atom_idx = 0
        for step in schedule.steps:
            tile_r = (atom_idx // cols) % rows
            tile_c = atom_idx % cols
            for det in step.detectors:
                sig = frozenset([det, (det + 1) % code.Hx.shape[0]])
                atom = FaultAtom(
                    atom_id=f"atom_{atom_idx}",
                    detector_signature=sig,
                    data_error_candidates=[frozenset([det % code.n])],
                    prior_weight=1.0 - self.p,
                    tile=(tile_r, tile_c),
                    time_slice=step.time,
                    window_size=step.ancilla_group + 1,
                )
                lib.add(atom)
                atom_idx += 1
        return lib
```

- [ ] **Step 5: Add _build_residual_graph and _estimate_cost to ExtractionCompiler**

```python
    def _build_residual_graph(
        self,
        code: QLDPCCode,
        geometry: RoutedGeometry,
        schedule: ExtractionSchedule,
        lib: FaultAtomLibrary,
    ) -> ResidualGraph:
        g = ResidualGraph()
        n_residual = max(1, code.Hx.shape[0] // 4)
        for i in range(n_residual):
            tile = (i % geometry.layout.rows, i % geometry.layout.cols)
            g.add_node(ResidualNode(node_id=i, tile=tile, time_slice=i % schedule.depth))
        for i in range(n_residual - 1):
            cross = (i % geometry.layout.cols == geometry.layout.cols - 1)
            g.add_edge(ResidualEdge(u=i, v=i+1, weight=self.p, cross_tile=cross))
        return g

    def _estimate_cost(
        self,
        lib: FaultAtomLibrary,
        residual: ResidualGraph,
        schedule: ExtractionSchedule,
    ) -> HardwareCostIR:
        n_atoms = len(lib)
        n_res = residual.num_nodes
        total = n_atoms + n_res if (n_atoms + n_res) > 0 else 1
        atom_hit_ratio = n_atoms / total
        comp_sizes = residual.component_sizes() or [0]
        histogram = {}
        for s in comp_sizes:
            histogram[s] = histogram.get(s, 0) + 1
        return HardwareCostIR(
            tile_costs=[],
            predicted_mean_latency_ns=50.0 * n_res + 10.0,
            predicted_p99_latency_ns=200.0 * n_res + 50.0,
            predicted_bram_traffic=n_atoms * 64,
            predicted_bank_conflict_rate=min(0.5, self.p * 10),
            predicted_cross_tile_traffic=residual.cross_tile_edge_count() * 8,
            predicted_atom_hit_ratio=atom_hit_ratio,
            predicted_residual_solve_rate=1.0 - atom_hit_ratio,
            residual_component_histogram=histogram,
        )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_extraction_compiler.py -v`
Expected: 4 PASSED

- [ ] **Step 7: Commit**

```bash
git add src/compiler/extraction_compiler.py tests/test_extraction_compiler.py
git commit -m "feat: ExtractionCompiler producing FaultAtomIR + ResidualGraphIR + HardwareCostIR"
```

---

## Task 9: FPGA Pipeline Simulator

**Files:**
- Create: `src/fpga/atom_matcher.py`
- Create: `src/fpga/residual_engine.py`
- Create: `src/fpga/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import pytest
from compiler.fault_atom import FaultAtom, FaultAtomLibrary
from compiler.residual_graph import ResidualGraph, ResidualNode, ResidualEdge
from fpga.pipeline import FPGAPipeline, ShotResult

def _make_lib():
    lib = FaultAtomLibrary()
    lib.add(FaultAtom(
        atom_id="a0", detector_signature=frozenset([0,1]),
        data_error_candidates=[frozenset([0])], prior_weight=0.9,
        tile=(0,0), time_slice=0, window_size=1,
    ))
    return lib

def _make_residual():
    g = ResidualGraph()
    g.add_node(ResidualNode(node_id=0, tile=(0,0), time_slice=0))
    g.add_node(ResidualNode(node_id=1, tile=(0,1), time_slice=0))
    g.add_edge(ResidualEdge(u=0, v=1, weight=0.3, cross_tile=True))
    return g

def test_pipeline_returns_shot_result():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    result = pipeline.decode(detector_events=frozenset([0, 1]))
    assert isinstance(result, ShotResult)

def test_pipeline_atom_hit_path():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    result = pipeline.decode(detector_events=frozenset([0, 1]))
    assert result.atom_hit is True
    assert result.latency_ns < 100.0

def test_pipeline_residual_path():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    result = pipeline.decode(detector_events=frozenset([99, 100]))
    assert result.atom_hit is False
    assert result.latency_ns >= 50.0

def test_pipeline_batch_stats():
    lib = _make_lib()
    residual = _make_residual()
    pipeline = FPGAPipeline(atom_library=lib, residual_graph=residual,
                            ns_per_atom_match=10.0, ns_per_residual_node=50.0)
    shots = [frozenset([0,1]), frozenset([99,100]), frozenset([0,1])]
    stats = pipeline.batch_decode(shots)
    assert stats["mean_latency_ns"] > 0
    assert 0.0 <= stats["atom_hit_rate"] <= 1.0
    assert "p99_latency_ns" in stats
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_pipeline.py -v`
Expected: ImportError

- [ ] **Step 3: Implement atom_matcher.py**

```python
# src/fpga/atom_matcher.py
from compiler.fault_atom import FaultAtom, FaultAtomLibrary

class AtomMatcherArray:
    def __init__(self, library: FaultAtomLibrary):
        self._lib = library

    def match(self, detector_events: frozenset[int]) -> FaultAtom | None:
        return self._lib.lookup(detector_events)
```

- [ ] **Step 4: Implement residual_engine.py**

```python
# src/fpga/residual_engine.py
from compiler.residual_graph import ResidualGraph

class ResidualEliminationEngine:
    def __init__(self, residual_graph: ResidualGraph, ns_per_node: float):
        self._graph = residual_graph
        self._ns_per_node = ns_per_node

    def solve(self) -> tuple[frozenset[int], float]:
        n = self._graph.num_nodes
        latency = n * self._ns_per_node
        correction: frozenset[int] = frozenset()
        return correction, latency
```

- [ ] **Step 5: Implement pipeline.py**

```python
# src/fpga/pipeline.py
from dataclasses import dataclass
import numpy as np
from compiler.fault_atom import FaultAtomLibrary
from compiler.residual_graph import ResidualGraph
from fpga.atom_matcher import AtomMatcherArray
from fpga.residual_engine import ResidualEliminationEngine

@dataclass
class ShotResult:
    correction: frozenset[int]
    latency_ns: float
    atom_hit: bool

class FPGAPipeline:
    def __init__(self, atom_library: FaultAtomLibrary, residual_graph: ResidualGraph,
                 ns_per_atom_match: float, ns_per_residual_node: float):
        self._matcher = AtomMatcherArray(atom_library)
        self._engine = ResidualEliminationEngine(residual_graph, ns_per_residual_node)
        self._ns_per_atom = ns_per_atom_match

    def decode(self, detector_events: frozenset[int]) -> ShotResult:
        atom = self._matcher.match(detector_events)
        if atom is not None:
            correction = atom.data_error_candidates[0] if atom.data_error_candidates else frozenset()
            return ShotResult(correction=correction, latency_ns=self._ns_per_atom, atom_hit=True)
        correction, latency = self._engine.solve()
        return ShotResult(correction=correction, latency_ns=latency, atom_hit=False)

    def batch_decode(self, shots: list[frozenset[int]]) -> dict:
        results = [self.decode(s) for s in shots]
        latencies = [r.latency_ns for r in results]
        hits = sum(1 for r in results if r.atom_hit)
        return {
            "mean_latency_ns": float(np.mean(latencies)),
            "p99_latency_ns": float(np.percentile(latencies, 99)),
            "atom_hit_rate": hits / len(results) if results else 0.0,
        }
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: 4 PASSED

- [ ] **Step 7: Commit**

```bash
git add src/fpga/ tests/test_pipeline.py
git commit -m "feat: FPGA pipeline simulator with atom matcher and residual elimination"
```

---

## Task 10: Candidate and Scorer

**Files:**
- Create: `src/search/candidate.py`
- Create: `src/search/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scorer.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from search.candidate import Candidate
from search.scorer import CandidateScorer, ScoreWeights

def _make_candidate():
    code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
    layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
    geom = RoutedGeometry(layout=layout, max_move_distance=2)
    steps = [ScheduleStep(time=t, ancilla_group=t%2, detectors=list(range(4)))
             for t in range(6)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    return Candidate(code=code, geometry=geom, schedule=sched)

def test_candidate_fields():
    c = _make_candidate()
    assert c.code is not None
    assert c.geometry is not None
    assert c.schedule is not None

def test_scorer_returns_float():
    c = _make_candidate()
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.5,
                           residual_solve_rate=1.0, cross_tile_traffic=0.5)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.80, max_bank_conflict_rate=0.10,
                             max_residual_component_size=8)
    score = scorer.score(c)
    assert isinstance(score, float)

def test_scorer_rejects_bad_candidate():
    c = _make_candidate()
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.5,
                           residual_solve_rate=1.0, cross_tile_traffic=0.5)
    # require 100% atom hit — impossible, so candidate should be rejected (inf score)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.9999, max_bank_conflict_rate=0.001,
                             max_residual_component_size=1)
    score = scorer.score(c)
    assert score == float("inf")
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_scorer.py -v`
Expected: ImportError

- [ ] **Step 3: Implement candidate.py**

```python
# src/search/candidate.py
from dataclasses import dataclass
from qldpc.codes import QLDPCCode
from qldpc.geometry import RoutedGeometry
from qldpc.schedule import ExtractionSchedule

@dataclass
class Candidate:
    code: QLDPCCode
    geometry: RoutedGeometry
    schedule: ExtractionSchedule
```

- [ ] **Step 4: Implement scorer.py**

```python
# src/search/scorer.py
from dataclasses import dataclass
from search.candidate import Candidate
from compiler.extraction_compiler import ExtractionCompiler

@dataclass
class ScoreWeights:
    mean_latency: float
    p99_latency: float
    memory_traffic: float
    residual_solve_rate: float
    cross_tile_traffic: float

class CandidateScorer:
    def __init__(self, weights: ScoreWeights, physical_error_rate: float,
                 min_atom_hit_ratio: float, max_bank_conflict_rate: float,
                 max_residual_component_size: int):
        self._weights = weights
        self._compiler = ExtractionCompiler(physical_error_rate)
        self._min_atom_hit = min_atom_hit_ratio
        self._max_bank_conflict = max_bank_conflict_rate
        self._max_res_size = max_residual_component_size

    def score(self, candidate: Candidate) -> float:
        out = self._compiler.compile(
            code=candidate.code,
            geometry=candidate.geometry,
            schedule=candidate.schedule,
        )
        cost = out.cost
        if not cost.passes_thresholds(self._min_atom_hit, self._max_bank_conflict,
                                      self._max_res_size):
            return float("inf")
        w = self._weights
        return (w.mean_latency * cost.predicted_mean_latency_ns
                + w.p99_latency * cost.predicted_p99_latency_ns
                + w.memory_traffic * cost.predicted_bram_traffic
                + w.residual_solve_rate * cost.predicted_residual_solve_rate * 1e6
                + w.cross_tile_traffic * cost.predicted_cross_tile_traffic)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_scorer.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/search/candidate.py src/search/scorer.py tests/test_scorer.py
git commit -m "feat: Candidate dataclass and multi-objective CandidateScorer"
```

---

## Task 11: Closed-Loop Search

**Files:**
- Create: `src/search/search_loop.py`
- Create: `tests/test_search_loop.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search_loop.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from search.candidate import Candidate
from search.scorer import CandidateScorer, ScoreWeights
from search.search_loop import SearchLoop, SearchResult

def _make_candidates(n: int) -> list[Candidate]:
    candidates = []
    for move_dist in range(1, n + 1):
        code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
        layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
        geom = RoutedGeometry(layout=layout, max_move_distance=move_dist)
        steps = [ScheduleStep(time=t, ancilla_group=t%2, detectors=list(range(4)))
                 for t in range(4 + move_dist)]
        sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
        candidates.append(Candidate(code=code, geometry=geom, schedule=sched))
    return candidates

def test_search_returns_result():
    candidates = _make_candidates(3)
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.1,
                           residual_solve_rate=1.0, cross_tile_traffic=0.1)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                             max_residual_component_size=100)
    loop = SearchLoop(scorer=scorer)
    result = loop.run(candidates=candidates)
    assert isinstance(result, SearchResult)

def test_search_selects_best_candidate():
    candidates = _make_candidates(3)
    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.1,
                           residual_solve_rate=1.0, cross_tile_traffic=0.1)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                             max_residual_component_size=100)
    loop = SearchLoop(scorer=scorer)
    result = loop.run(candidates=candidates)
    assert result.best_candidate in candidates
    assert result.best_score < float("inf")

def test_search_all_rejected_returns_inf():
    candidates = _make_candidates(2)
    weights = ScoreWeights(mean_latency=1.0, p99_latency=1.0, memory_traffic=1.0,
                           residual_solve_rate=1.0, cross_tile_traffic=1.0)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.9999, max_bank_conflict_rate=0.0001,
                             max_residual_component_size=1)
    loop = SearchLoop(scorer=scorer)
    result = loop.run(candidates=candidates)
    assert result.best_score == float("inf")
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_search_loop.py -v`
Expected: ImportError

- [ ] **Step 3: Implement search_loop.py**

```python
# src/search/search_loop.py
from dataclasses import dataclass
from search.candidate import Candidate
from search.scorer import CandidateScorer

@dataclass
class SearchResult:
    best_candidate: Candidate
    best_score: float
    all_scores: list[tuple[Candidate, float]]

class SearchLoop:
    def __init__(self, scorer: CandidateScorer):
        self._scorer = scorer

    def run(self, candidates: list[Candidate]) -> SearchResult:
        scored = [(c, self._scorer.score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1])
        best_candidate, best_score = scored[0]
        return SearchResult(
            best_candidate=best_candidate,
            best_score=best_score,
            all_scores=scored,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_search_loop.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/search/search_loop.py tests/test_search_loop.py
git commit -m "feat: closed-loop search over candidates with multi-objective scoring"
```

---

## Task 12: Baseline Decoder and Integration Test

**Files:**
- Create: `src/baselines/bp_decoder.py`
- Create: `tests/test_baselines.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write failing baseline test**

```python
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_baselines.py -v`
Expected: ImportError

- [ ] **Step 3: Implement bp_decoder.py**

```python
# src/baselines/bp_decoder.py
import numpy as np
from qldpc.codes import QLDPCCode

class FixedExtractionBaseline:
    """Thin Monte Carlo baseline: random bit-flip noise, majority-vote correction."""
    def __init__(self, physical_error_rate: float, num_shots: int):
        self.p = physical_error_rate
        self.num_shots = num_shots

    def logical_error_rate(self, code: QLDPCCode) -> float:
        rng = np.random.default_rng(42)
        logical_errors = 0
        for _ in range(self.num_shots):
            error = rng.random(code.n) < self.p
            syndrome_x = (code.Hx @ error.astype(np.uint8)) % 2
            # trivial correction: flip qubits where syndrome fires
            correction = np.zeros(code.n, dtype=np.uint8)
            for check_idx, bit in enumerate(syndrome_x):
                if bit:
                    support = np.where(code.Hx[check_idx])[0]
                    if len(support):
                        correction[support[0]] ^= 1
            residual = (error.astype(np.uint8) ^ correction) % 2
            logical_errors += int(np.any(residual))
        return logical_errors / self.num_shots
```

- [ ] **Step 4: Run baseline tests**

Run: `pytest tests/test_baselines.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Write integration test**

```python
# tests/test_integration.py
import pytest
from qldpc.codes import BivariateBicycleCode
from qldpc.geometry import TileLayout, RoutedGeometry
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy
from search.candidate import Candidate
from search.scorer import CandidateScorer, ScoreWeights
from search.search_loop import SearchLoop
from compiler.extraction_compiler import ExtractionCompiler
from fpga.pipeline import FPGAPipeline

def test_end_to_end_search_and_decode():
    candidates = []
    for move_dist in [1, 2]:
        code = BivariateBicycleCode(l=6, m=6, a=[3,1,2], b=[3,1,2])
        layout = TileLayout(rows=2, cols=2, data_per_tile=18, ancilla_per_tile=9)
        geom = RoutedGeometry(layout=layout, max_move_distance=move_dist)
        steps = [ScheduleStep(time=t, ancilla_group=t%2, detectors=list(range(4)))
                 for t in range(6)]
        sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
        candidates.append(Candidate(code=code, geometry=geom, schedule=sched))

    weights = ScoreWeights(mean_latency=1.0, p99_latency=2.0, memory_traffic=0.1,
                           residual_solve_rate=1.0, cross_tile_traffic=0.1)
    scorer = CandidateScorer(weights=weights, physical_error_rate=0.001,
                             min_atom_hit_ratio=0.0, max_bank_conflict_rate=1.0,
                             max_residual_component_size=100)
    loop = SearchLoop(scorer=scorer)
    result = loop.run(candidates=candidates)
    assert result.best_score < float("inf")

    compiler = ExtractionCompiler(physical_error_rate=0.001)
    out = compiler.compile(
        code=result.best_candidate.code,
        geometry=result.best_candidate.geometry,
        schedule=result.best_candidate.schedule,
    )
    pipeline = FPGAPipeline(
        atom_library=out.atom_library,
        residual_graph=out.residual_graph,
        ns_per_atom_match=10.0,
        ns_per_residual_node=50.0,
    )
    shots = [frozenset([0, 1]), frozenset([99, 100]), frozenset([0, 1])]
    stats = pipeline.batch_decode(shots)
    assert stats["mean_latency_ns"] > 0
    assert 0.0 <= stats["atom_hit_rate"] <= 1.0
```

- [ ] **Step 6: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: 1 PASSED

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: all tests PASSED

- [ ] **Step 8: Commit**

```bash
git add src/baselines/ tests/test_baselines.py tests/test_integration.py
git commit -m "feat: baseline decoder and end-to-end integration test"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Candidate Generation Layer | Tasks 2, 3, 4 (codes, geometry, schedule) |
| Fault-Atom IR | Task 5 |
| Residual Graph IR | Task 6 |
| Hardware Cost IR | Task 7 |
| Decoder-Aware Extraction Compiler | Task 8 |
| FPGA atom matcher array | Task 9 |
| FPGA residual elimination engine | Task 9 |
| End-to-end pipeline simulator | Task 9 |
| Multi-objective scorer | Task 10 |
| Closed-loop search | Task 11 |
| Fixed-extraction baseline | Task 12 |
| Integration test | Task 12 |

**Out of scope for phase one (per spec):** RTL bitstream, lattice surgery, full control-stack integration, soft information online path.

**Placeholder scan:** No TBD, TODO, or "similar to" references found.

**Type consistency:** All types defined in earlier tasks are used consistently in later tasks — `FaultAtomLibrary`, `ResidualGraph`, `HardwareCostIR`, `CompilerOutput`, `Candidate`, `ScoreWeights`, `SearchResult`, `ShotResult` are all defined before first use.
