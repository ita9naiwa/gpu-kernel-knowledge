# Decision rules distilled from failed optimization attempts

These are generalized engineering guidelines distilled from reviewed private
optimization records, published with the owner's authorization. Original code,
logs, identities, workloads, measurements and transcript excerpts are not
published. The lessons are not independently reproducible public benchmark
results, proofs of correctness, or established microarchitectural diagnoses.
The linked public references explain related mechanisms; they are not evidence
that the private experiments were reproduced upstream.

## 1. Preserve the executed numerical contract, not just the formula

Before fusing arithmetic, inspect the caller's actual operand dtypes and
intermediate casts. A test using identical dtypes for every operand can miss
the mixed-dtype path used by the application. A compiler option intended to
preserve precision does not replace output-level validation.

When changing a reduction's tile shape, retest exactness even if the removed
lanes contain mathematical zeros. The compiled reduction layout or evaluation
order can change. Similarly, successful channel/extent pairs do not imply that
the Cartesian product of their coordinates is safe. Keep the tested joint
predicate and an explicit fallback; a sampled pass still does not prove the
entire admitted domain.

**Decision:** retain the first failing input class and its tolerance contract.
If a rounding change restores parity, record the recovery separately from the
claimed compiler cause unless generated-code evidence isolates that cause.
See [fusion and reductions](../practices/fusion-and-reductions.md) and
[tiling and layout](../practices/tiling-and-layout.md).

## 2. Treat execution mode and dispatch as specialization dimensions

A fusion can help an eager fallback while regressing compiled execution. A
backend-specific guard is a new candidate with its own acceptance domain;
neither a broad rejection nor a narrow success transfers automatically.

Trace every consumer of a shape-admission setting. The same setting may govern
both compilation eligibility and a higher-level route. Changing it to admit a
tile can unintentionally admit a whole input. A stable graph-cache entry count
does not prove that the newly admitted call replayed a captured graph. Likewise,
a warmed success does not establish cold-capture behavior.

**Decision:** verify the executed backend and nearby fallback cases, not only
the intended guard. A size threshold must be measured through the real wrapper:
its dispatch and fallback overhead can preserve the very small-input regression
it was meant to remove. See [dispatch and graphs](../practices/dispatch-and-graphs.md).

## 3. Carry the winning configuration into a faithful full-path comparison

A phase sweep may time preallocated output and precomputed statistics. Its
winner is a candidate for the full path, not a full-path result. Check that the
actual selector uses that winner: a policy keeping one warp count for every
shape can omit the best sampled configuration. Also distinguish padded kernel
dimensions from the public input dimensions used to describe the result.

Inspect the baseline as closely as the candidate. An extra helper call in the
baseline can create an apparent gain that disappears against the original
implementation. Correcting that harness starts a new comparison; do not pool
samples from the old and corrected baselines.

**Decision:** retain source/configuration identity for the compared paths and
time the requested public boundary, including necessary preparation and
conversion. See [measurement boundaries](measurement-boundaries.md).

## 4. Separate less work, completed evaluation, and successful optimization

Fewer kernel launches and shorter summed GPU activity can coexist with no
distinguishable application-latency gain. Count actual GPU activity if claiming
a kernel-count reduction: counting one host launch API misses alternative
submission paths. Summed profiler durations are not an elapsed-time decomposition.

Read benchmark parameters at the helper that consumes them. Names suggesting
iteration counts may be passed to time-budget arguments; inspect the returned
sample count rather than inferring it from configuration labels. An evaluator's
"passed" timing status may mean measurement completed, while an independent
performance/adoption gate still rejects the candidate. A baseline-only partial
record after a correctness failure contains no candidate timing result.

**Decision:** apply the task's actual acceptance gate to matched measurements.
Keep noise-limited differences inconclusive. Report-only timings remain
report-only when raw samples or the corresponding harness are unavailable.
See [correctness and benchmarking](../practices/correctness-and-benchmarking.md).

## 5. Evaluate CuTe ports as concrete implementations, not language verdicts

A slower scalar port does not rule out a better vectorized descendant. But
changing the grid, per-thread work, alignment assumptions and launch geometry
together does not isolate which change helped. Scalar code can already access
contiguous addresses across lanes. A vector-copy abstraction alone does not
prove the emitted transaction width or the elimination of indexing overhead.

For mixed-precision tensor-core work, relaxing a top-level dtype check is not
enough. Follow operand types through shared-memory layouts, copy atoms, MMA
selection and descriptor construction. A rejected descriptor identifies a
legality problem in that implementation; it establishes neither hardware-wide
impossibility nor a performance result.

**Decision:** preserve the original losing attempt and the narrower follow-up
as separate observations. Recheck neighboring shapes and actual application
calls, and inspect generated code before assigning a mechanism to the gain.
See [architecture selection](architecture-selection.md) and the
[public debugging postmortems](failure-driven-debugging.md).
