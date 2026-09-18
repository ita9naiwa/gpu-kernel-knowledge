# Kernel task contract

## Requested behavior

- Operation and existing implementation/callers:
- Inputs/outputs, shape distribution, strides/layouts, device placement:
- Input, output, accumulation dtypes; tolerance and reference:
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

Correctness cases and performance cases to execute:

Acceptance criteria and fallback if they fail:
