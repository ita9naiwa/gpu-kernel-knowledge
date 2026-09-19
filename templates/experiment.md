# Kernel experiment

Status: proposed / implemented / CPU-checked / GPU-correctness-checked / measured / rejected

Outcome: accepted in scope / wrong answer / correct but slower / no distinguishable gain / illegal configuration / environment blocked

## Hypothesis and change

If **[one change]**, then **[metric]** improves for **[workload]** because **[mechanism]**. It may regress **[countercase]**.

Baseline revision, candidate revision, diff, upstream evidence:

Minimum useful whole-boundary gain, complexity budget and stopping rule:

Observed runtime precision, fusion, backend, compile, checkpoint and Graph settings;
differences from a precision-matched control:

Measured bottleneck and why this candidate has priority:

KB rule ID / reference read, workload applicability, and decision it changed:

Countercondition and cheapest check that would reject the hypothesis:

Freeze reference and harness identities too. Preserve the first failing case and
raw error output. A harness correction starts a new comparison; retain the old
record and explain why its conclusion is invalid or narrower.

## Reproduction

Record exact commands, working directory, environment, GPU model/count, compute capability, driver/CUDA/compiler/framework versions, launch configuration, seeds, shape/layout/dtype/mask distribution, and workspace. Record competing GPU load and clock/power settings when relevant.

```sh
# Correctness command:
# Sanitizer command, when needed:
# Baseline benchmark command:
# Candidate benchmark command:
# Profiling command, if used to explain the result:
```

Define timing boundary, warm-up, repetitions, synchronization, stream, allocation inclusion, compilation/autotuning inclusion, graph capture/replay mode, and cache policy. Preserve raw outputs. State whether repetitions share inputs and whether measurements are sequential or interleaved. Keep profiler runs separate from timing claims.

## Correctness results

| Case / reference | Max absolute error | Relative L2 | Other required metric | Tolerance | Pass/fail |
| --- | ---: | ---: | --- | --- | --- |
| | | | | | |

Define relative L2 as `norm(candidate - reference) / norm(reference)` and record the zero-reference policy. Include relevant empty/ragged inputs, tails, large values, masks, strides, and accumulation behavior. Record sanitizer findings and unrun checks explicitly.

## Performance results

| Workload / timing boundary | Baseline latency | Candidate latency | Baseline / candidate | Variability / samples |
| --- | ---: | ---: | ---: | --- |
| | | | | |

Report live/peak allocated, reserved and device-used memory with their scopes; do not sum overlapping counters. Use consistent units. State the summary statistic and variability statistic. Keep kernel results separate from application results. Include unfavorable representative cases; weight aggregates using the actual workload distribution, if known.

## Decision

Observed facts:

Interpretation and unresolved alternative explanations:

Accept / reject / keep behind dispatch, with supported workload boundary:

Reusable failure lesson and conditions under which it should be reconsidered:

Next single experiment, only if needed:

Optional evaluator feedback: verified contribution (gain / material question / scoped rejection), evidence, decision affected, and next-budget rationale. No points for counts alone.
