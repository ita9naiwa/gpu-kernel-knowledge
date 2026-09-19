# Kernel experiment

Status: proposed / implemented / CPU-checked / GPU-correctness-checked / measured / rejected

Outcome: accepted in scope / wrong answer / correct but slower / no distinguishable gain / illegal configuration / environment blocked

## Hypothesis and change

If **[one change]**, then **[metric]** improves for **[workload]** because **[mechanism]**. It may regress **[countercase]**.

Frozen [task contract](task.md), baseline/candidate revisions and diff:

Implementation origin: installed API / public upstream adaptation / original /
explicitly authorized task reproduction. Source revision/symbol, license/notices,
adapted parts and local changes; reason for rewriting if applicable:

Record only deviations from the task's environment, runtime settings, accuracy
and stopping rules; do not duplicate an unchanged contract.

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

Use the task’s [numerical contract](../recipes/kernel-writing-workflow.md#set-the-numerical-contract-before-search). Include relevant empty/ragged inputs, tails, large values, masks, strides, and accumulation behavior. Record sanitizer findings and unrun checks explicitly.

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

Optional [evaluator feedback](../recipes/kernel-writing-workflow.md#evidence-based-feedback-experimental): evidence, decision affected and next-budget rationale.
