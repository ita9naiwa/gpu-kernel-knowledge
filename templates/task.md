# Kernel task contract

## Requested behavior

- Operation and existing implementation/callers:
- Inputs/outputs, shape distribution, strides/layouts, device placement:
- Input, output and accumulation dtypes:
- Target GPU/compute capability; driver, CUDA, compiler, framework/library versions:
- Objective and boundary: kernel / graph replay / request / throughput; representative weighting:

## Constraints and evidence

State graph-capture requirements, stream behavior, allowed workspace, determinism, aliasing/in-place rules, unusual masks or empty inputs, and supported fallbacks. Mark unknowns explicitly.

| Relevant source or practice | Exact revision/section | Applies because | Does not establish |
| --- | --- | --- | --- |
| | | | |

## Implementation decision

Existing library path considered:

Observed bottleneck and supporting measurement, or unmeasured hypothesis:

Smallest proposed change and preserved semantics:

Representative and difficult correctness/performance cases; fallback:

## Initial testbed and coordination (when needed)

- Resolved facts, material questions still pending, and explicit assumptions:
- Frozen source/config/fixture/evaluator identities; baseline smoke and coarse profile evidence:
- Authorized time/GPU budget; minimum useful gain, complexity budget and stopping rule:
- Coordination/integration owner; shared-source update and final evaluation ownership:

| Hypothesis/mechanism | Owner and private files | Expected whole-boundary impact | Budget/GPU slot | Deliverable and stop condition | State/result |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Record both validated gains and useful scoped rejections. Reserve experiments
before launch; distinguish intentional replication from duplicated search.

## Numerical contract (freeze before search)

- Mode: ordinary numerical variation (default) / explicitly allowed lossy / bitwise same:
- Reference, baseline repeat variability, dtype and precision/accumulation constraints:
- Output/gradient/consumer scope; metrics, thresholds, aggregation and zero-value policies:
- Material question or explicit assumption; evidence supporting the gate:

Use the [numerical contract guide](../recipes/kernel-writing-workflow.md#set-the-numerical-contract-before-search).
