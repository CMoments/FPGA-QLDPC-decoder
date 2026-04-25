# qLDPC Decoder-Aware Extraction

A research system for co-designing neutral-atom quantum LDPC decoders. The core idea: rather than accepting syndrome extraction circuits as fixed inputs, this system jointly optimizes the extraction schedule, routed geometry, decoder IR, and FPGA microarchitecture to minimize decoding latency while preserving logical error rates.

## What it does

Given a qLDPC code family, the system searches over candidate (code, geometry, schedule) triples, compiles each into a hardware-friendly decoding representation, scores them on a multi-objective cost model, and returns the best candidate for FPGA deployment.

```
Code + Geometry + Schedule
         │
         ▼
  ExtractionCompiler
         │
    ┌────┴────┐
    │         │
FaultAtomIR  ResidualGraphIR  HardwareCostIR
    │         │
    └────┬────┘
         ▼
   FPGAPipeline
   (atom matcher + residual engine)
```

## Project structure

```
src/
  qldpc/      # Code abstractions: QLDPCCode, TileLayout, ExtractionSchedule
  compiler/   # ExtractionCompiler → FaultAtomIR, ResidualGraphIR, HardwareCostIR
  fpga/       # FPGAPipeline, atom matcher, residual elimination engine
  search/     # SearchLoop, CandidateScorer, Candidate dataclass
  baselines/  # Belief-propagation baseline decoder
tests/        # 13 test files covering all modules + integration
docs/         # Per-module documentation
```

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

Key runtime dependencies: `numpy`, `scipy`, `networkx`, `stim`, `pymatching`.

## Running tests

```bash
pytest
pytest --cov   # with coverage
```

## Design

See [`2026-04-25-neutral-atom-qldpc-decoder-aware-extraction-design.md`](2026-04-25-neutral-atom-qldpc-decoder-aware-extraction-design.md) for the full design spec covering the problem statement, IR definitions, search objectives, FPGA microarchitecture, and success criteria.
