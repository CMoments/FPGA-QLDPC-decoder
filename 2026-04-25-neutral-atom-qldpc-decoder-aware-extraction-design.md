# Neutral-Atom qLDPC Decoder-Aware Extraction Co-Design

## Goal

Define a research and system design for a neutral-atom qLDPC decoding stack that co-synthesizes:

- syndrome extraction schedule
- routed geometry
- decoder intermediate representation
- FPGA microarchitecture

The central objective is to minimize decoding latency under a logical-error-rate budget, instead of minimizing logical error under a fixed extraction circuit.

## Design Thesis

Existing FPGA qLDPC decoders are increasingly strong on the decoder side, but they still mostly accept the extraction circuit and routed geometry as fixed inputs. This design changes the optimization boundary.

The proposed system makes the decoding problem hardware-easy before online decoding starts. It searches over neutral-atom-friendly qLDPC code families, extraction schedules, and routed geometries, then compiles the resulting circuit-level detector structure into:

- a high-hit-rate fault-atom front end
- a bounded residual graph
- an FPGA dataflow dominated by local template matching instead of global sparse-graph iteration

The intended outcome is substantially lower mean and tail latency than generic BP-family FPGA decoders at similar logical error rates.

## Scope

This design covers:

- neutral-atom memory-style qLDPC decoding workloads
- circuit-level noise with geometry-induced correlations
- code-family, extraction-schedule, geometry, and decoder co-design
- a compiled FPGA architecture with atom-first streaming and residual elimination
- benchmarking methodology, ablations, and success criteria

This design does not cover:

- a universal decoder for all qLDPC codes and all hardware platforms
- full quantum control-stack integration
- a board-specific bitstream or timing-closed RTL implementation as a design requirement
- non-memory logical operations such as lattice surgery as a phase-one target

## Why This Is Not Incremental

The design is intentionally not "a better BP variant" or "a stronger post-processor." It shifts the problem from:

- fixed extraction plus better decoding

to:

- decoder-aware extraction synthesis plus compiled decoding

This separates it from work that improves message passing, graph rewiring, speculative post-processing, or hardware mapping while keeping the circuit fixed. The novelty is in jointly shaping the circuit-level fault structure and the hardware-consumable decoding representation.

## Problem Statement

For neutral-atom qLDPC memories, routed geometry and extraction scheduling strongly influence the effective detector graph, correlation structure, and online hardware cost. A decoder that ignores those degrees of freedom inherits:

- dense short cycles
- irregular memory traffic
- long convergence tails
- large residual subgraphs
- high cross-tile communication

The design problem is therefore:

Given a family of neutral-atom-compatible qLDPC constructions and feasible extraction schedules, find the schedule, geometry, and compiled decoder representation that minimize online decode latency while preserving logical performance close to a strong fixed-circuit baseline.

## Primary Success Criteria

The primary target is to demonstrate all of the following against the strongest fixed-extraction baseline available for the same code family and noise model:

- at least 3x lower mean decode latency
- at least 5x lower P99 decode latency
- no worse than 1.3x logical error rate at matched physical error rate and schedule budget
- materially lower on-chip memory traffic and cross-tile traffic

Secondary targets:

- atom front-end resolves at least 80% of shots without invoking a heavy residual solve
- residual graph size is reduced by at least 4x on average relative to the uncompiled detector problem
- deadline miss rate is significantly reduced under neutral-atom timing budgets

## System Overview

The system has four layers.

### 1. Candidate Generation Layer

This layer enumerates feasible combinations of:

- qLDPC family candidates suited to neutral-atom routing
- routed geometry layouts
- extraction schedules
- ancilla reuse policies
- movement and locality constraints

The search space is restricted to candidates that can be physically scheduled and routed on a neutral-atom platform.

### 2. Decoder-Aware Extraction Compiler

This compiler takes a candidate circuit and produces two runtime artifacts:

- Fault-Atom IR
- Residual Graph IR

It also produces compile-time cost estimates tied to FPGA execution:

- predicted mean latency
- predicted tail latency
- predicted BRAM traffic
- predicted bank conflicts
- predicted cross-tile traffic
- predicted atom-hit ratio
- predicted residual solve rate

### 3. FPGA Decoder Backend

The backend maps the compiled artifacts into an FPGA microarchitecture composed of:

- event ingest and time-slice buffering
- atom matcher arrays
- local aggregation and conflict resolution
- residual graph construction
- bounded residual elimination

The dominant execution path must remain local and streaming.

### 4. Closed-Loop Search Layer

The search loop scores each candidate using:

- logical error rate under circuit-level noise
- hardware cost model from the compiled representation
- timing-budget feasibility

The selected design is the one that best satisfies latency and tail-latency objectives under the logical-performance constraint.

## Core Intermediate Representations

### Fault-Atom IR

Fault-Atom IR is the central abstraction of the design. It replaces generic detector-graph processing with a finite dictionary of recurring circuit-level fault motifs.

Each fault atom stores:

- an atom identifier
- the local extraction pattern that can trigger it
- the detector signature it produces
- one or more candidate data-error outcomes
- a prior weight or score
- the local state window needed for online matching
- the tile and time-slice locality metadata required for hardware placement

The design goal is to maximize the fraction of online events that can be explained by this IR.

### Residual Graph IR

Residual Graph IR represents the unresolved remainder after atom matching and local conflict resolution.

It is explicitly designed to have:

- small connected components
- low fill-in under the chosen elimination order
- limited cross-tile edges
- bounded solve depth
- a fixed or narrowly distributed resource footprint

This IR is not a copy of the original detector graph. It is a reduced problem designed for fast, regular, hardware-friendly cleanup.

### Hardware Cost IR

The compiler also emits a cost-oriented representation used only for search and backend placement.

It includes:

- per-tile memory pressure
- per-tile atom library size
- predicted local conflict density
- predicted residual component histogram
- communication volume across tile boundaries
- schedule depth and atom evaluation count

This representation prevents the search from choosing a candidate that looks good abstractly but compiles into an irregular FPGA workload.

## Search Objective

The search objective is:

minimize

- weighted mean latency
- weighted P99 latency
- weighted memory traffic
- weighted residual solve rate
- weighted cross-tile traffic

subject to:

- logical error rate within the allowed budget
- schedule depth within the experiment budget
- geometry and movement constraints
- FPGA resource constraints

The optimizer should reject candidates early if they compile into:

- low atom coverage
- large residual components
- severe bank conflicts
- excessive cross-tile communication
- deadline miss predictions above the acceptable threshold

## FPGA Microarchitecture

### 1. Event Ingest and Time-Slice Buffer

The input stream is partitioned by time slice and spatial tile instead of flattening all detector information into a global irregular graph.

Each event record carries:

- detector bits
- tile identifier
- time-slice identifier
- route or movement tag
- ancilla group identifier
- optional soft information if available

This preserves locality and enables deterministic streaming.

### 2. Fault-Atom Matcher Array

This is the dominant execution path.

Each matcher bank handles one atom family, such as:

- ancilla reuse motifs
- same-tick correlated crosstalk motifs
- route-induced correlated motifs
- local hook-style measurement motifs

Each bank performs:

- local window fetch
- template match
- score evaluation
- candidate correction emission
- unresolved detector tagging

The matcher array is fixed-depth and non-iterative in the common case.

### 3. Local Aggregation and Conflict Resolution

Multiple atom candidates may overlap. A lightweight local resolver merges them using:

- local confidence comparison
- short-range voting
- mutually exclusive candidate rejection
- unresolved detector marking

This block must remain tile-local and must not become a hidden global decoder.

### 4. Residual Graph Builder

Only unresolved events are promoted into the residual problem.

The builder constructs:

- residual detector nodes
- residual correlation edges
- tile-boundary links
- the compile-time chosen elimination order

The result is a much smaller graph than the original detector problem.

### 5. Residual Elimination Engine

The residual engine performs bounded cleanup using a fixed or narrowly variable pipeline.

Allowed strategies include:

- blockwise elimination
- tree-decomposition-style local message passing
- bounded-depth local search
- a rare-case fallback lane for extremely stubborn residuals

The design rule is that iteration is moved from the common case to the rare case.

## Hardware Mapping Strategy

The mapping strategy follows the routed geometry.

- one FPGA tile corresponds to one local routed region
- each tile owns its local atom templates in BRAM
- event windows are consumed locally whenever possible
- only unresolved residual information crosses tile boundaries

This is expected to reduce:

- random memory traffic
- crossbar pressure
- unpredictable iteration tails

The decoder's dominant path should therefore be local template matching, not global sparse-graph traversal.

## Runtime Data Flow

The end-to-end online flow is:

1. Neutral-atom extraction events arrive as time-sliced local records.
2. Records are routed into tile-local buffers.
3. Atom matcher arrays evaluate the incoming events against compiled atom libraries.
4. Local aggregation resolves consistent candidates and tags unresolved patterns.
5. Residual graph builder constructs a reduced cleanup problem.
6. Residual elimination engine solves the cleanup problem.
7. The final correction is emitted within the timing budget.
8. Runtime counters log atom-hit rate, residual size, and deadline behavior for analysis.

## Candidate Rejection and Failure Handling

The compiler and runtime fail fast.

Compile-time rejection triggers:

- atom coverage below the configured threshold
- residual components larger than the bounded hardware envelope
- predicted bank conflict rate above the accepted limit
- predicted deadline miss rate above the accepted limit

Runtime safeguards:

- if a shot exceeds the normal atom-plus-residual path, it enters a rare-case fallback lane
- if fallback activation frequency exceeds the configured threshold, the schedule candidate is considered invalid for deployment
- if calibration drift causes atom hit rate to collapse, the system requires recompile or retuning instead of silently degrading into uncontrolled behavior

## Evaluation Methodology

### Workloads

Evaluation will focus on neutral-atom memory-style qLDPC workloads with:

- circuit-level extraction noise
- geometry-induced correlated errors
- movement and routing constraints
- multiple code-family candidates, not only one fixed BB-style instance

### Baselines

The minimum baseline set is:

- fixed extraction plus FPGA-oriented Relay-BP-style decoder
- fixed extraction plus correlated-error graph-rewriting decoder
- fixed extraction plus hierarchical or sparse non-BP decoder when applicable
- the same candidate family without decoder-aware extraction synthesis

### Required Ablations

The paper must include:

- co-synthesis full system
- fixed extraction with compiled atom-plus-residual backend
- decoder-aware extraction without atom front end
- search optimized for logical error only
- search optimized for latency only

These ablations are necessary to show that the gain comes from joint optimization, not from one isolated heuristic.

### Reported Metrics

The paper must report:

- logical error rate
- mean decode latency
- P95 and P99 decode latency
- deadline miss rate
- atom-hit rate
- residual component size distribution
- BRAM traffic and bank conflict statistics
- cross-tile communication volume
- FPGA resource estimates or measurements

## Verification Plan

### Algorithm Verification

- validate Fault-Atom IR extraction on small hand-checkable circuits
- confirm that atom matches reproduce the intended detector signatures
- confirm that residual graph construction preserves unresolved corrections
- compare compiled decoding outputs against a stronger software reference on small instances

### Search Verification

- verify that hardware-cost predictions correlate with measured simulator or RTL-level metrics
- verify that early-rejected candidates are genuinely poor under measured execution
- verify that optimizing only logical error produces worse runtime behavior than joint optimization

### Hardware Verification

- build a cycle-accurate simulator for the atom-first streaming pipeline
- validate deterministic replay on captured event traces
- test tile-local buffering, conflict resolution, and residual promotion logic independently
- stress-test rare-case fallback frequency under adversarial correlated noise

### Scientific Validation

- sweep physical error rates around the practical operating regime
- test multiple geometry and schedule families
- include at least one code-family transfer test to show the method is not a one-instance artifact

## Key Risks

### Risk 1: Atom Library Coverage Is Too Low

If the compiled atom dictionary does not explain most shots, the design collapses into a residual decoder with extra overhead.

Mitigation:

- constrain the search to candidates that produce repetitive local motifs
- use atom-coverage thresholds as a compile-time rejection criterion
- co-optimize schedule and geometry specifically for motif dominance

### Risk 2: LER Degrades Too Much

Over-optimizing for hardware regularity may distort the extraction schedule enough to damage logical performance.

Mitigation:

- keep logical error rate as a hard search constraint
- include weighted exposure penalties during candidate scoring
- report the Pareto frontier instead of a single chosen point

### Risk 3: Residual Components Remain Too Large

Some candidates may still produce long-range unresolved structures.

Mitigation:

- reject candidates with poor residual histograms
- enforce elimination-order and component-size bounds during compilation
- keep a rare-case lane but require very low activation frequency

### Risk 4: Hardware Cost Model Is Inaccurate

If the search optimizes the wrong proxy, the selected candidates may not be the fastest on real hardware.

Mitigation:

- calibrate the cost model against simulator or prototype measurements
- include iterative retraining of the cost model
- publish both predicted and measured metrics

## Deliverables

Phase-one deliverables are:

- a formal decoder-aware extraction compiler definition
- a Fault-Atom IR and Residual Graph IR specification
- a search framework over code, geometry, and schedule candidates
- an FPGA backend model for atom-first streaming plus residual elimination
- an evaluation against strong fixed-extraction baselines
- a paper-quality set of ablations and Pareto plots

## Recommended Paper Framing

The paper should be framed as a systems-and-architecture contribution, not only as a decoder contribution.

Suggested one-sentence claim:

We introduce a decoder-aware extraction compiler for neutral-atom qLDPC memories that transforms circuit-level syndrome extraction into an atom-dominant, residual-bounded inference problem, enabling substantially lower average and tail FPGA decoding latency at similar logical error rates.

Suggested contribution structure:

- a new co-design objective: minimize decode latency under a logical-error budget
- a compiler that converts extraction circuits into hardware-friendly decoding IRs
- an atom-first FPGA architecture that removes most online global iteration
- a benchmark study showing large latency reductions with similar logical performance

## Open Questions To Resolve Before Implementation Planning

- which qLDPC family subset will anchor the first paper submission
- whether optional soft information is available online or only offline
- how much schedule freedom the targeted neutral-atom experiment can tolerate
- whether the first prototype uses cycle-accurate simulation only or includes RTL for key blocks
- which fallback solver provides the best rare-case tradeoff
