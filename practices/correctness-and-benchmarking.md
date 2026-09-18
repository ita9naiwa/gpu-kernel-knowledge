# Correctness and benchmarking

Keywords: reference, numerical error, relative L2, memcheck, racecheck, synccheck, CUDA events, CUPTI, cold L2, graph replay, profiler.

These practice rules derive from source inspection; the cited upstream kernels were not locally validated for correctness or performance as part of this derivation. Separately recorded experiments have their own workload and evidence scope. CPU documentation checks cannot establish GPU correctness or speed.

## CHECK-01 — Validate the full contract before optimizing its latency

**Applies when:** creating or replacing any GPU operator.

**Do:** build an independent reference and test the supported shape/dtype/layout/semantic matrix. Include outputs, mutations, auxiliary state and gradients when supported. Derive tricky small cases by hand so the reference cannot repeat the implementation's indexing error.

**Source / evidence:** **documented upstream policy**: FlashAttention [`Tests`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/README.md#L552-L556) checks output and gradients across dtype, dimensions, lengths and causal modes; Triton [`matmul tutorial`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/03-matrix-multiplication.py) pairs implementation with reference comparison. Source IDs: `flashattention`, `triton`. The full-contract matrix is a **derived recommendation**.

**Counterconditions:** passing one square contiguous random test is insufficient. Unsupported inputs should fail clearly; testing them does not imply expanding the kernel's scope.

Tutorial reference comparisons are examples, not necessarily failing test gates: the pinned Triton matmul tutorial prints success/failure for its `allclose` check. Turn acceptance conditions into assertions in the target harness rather than treating process exit zero as proof that its printed comparison passed.

**Validate:** tails, zero/one supported sizes, ragged lengths, large offsets, nonzero storage offsets, allowed strides, extreme values and empty masks. Track `max_abs`, `relative_L2 = ||actual-reference||2 / ||reference||2` with an explicit zero-reference policy, and NaN/Inf agreement. If a denominator floor is used, label that metric as stabilized relative L2 and report the floor separately.

## CHECK-02 — Check memory and synchronization separately from numerical outputs

**Applies when:** raw CUDA/Triton pointer arithmetic, shared memory, async transfers, atomics or in-place updates are changed.

**Do:** run the available sanitizer tools appropriate to the code, beginning with memory errors before interpreting race/synchronization findings. Exercise the exact optimized specialization and its tail paths.

**Source / evidence:** **official tooling contract**: NVIDIA [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) provides memory, race, initialization and synchronization checks. Source ID: `compute-sanitizer`. The order and workload selection are **derived recommendations**.

**Counterconditions:** tools have architecture/instruction coverage limits and version-specific behavior. A clean run is evidence for exercised executions, not proof of race freedom for every schedule.

**Validate:** smallest/partial/full/multi-stage-wrap cases and concurrent use where supported. Record tool version, command, exit status and unsupported-check warnings. For automated gates, set an explicit nonzero `--error-exitcode` (for example `--error-exitcode 1`) and retain/inspect the sanitizer summary: the documented default is zero, so a successful application can return zero despite tool findings. If no GPU is available, mark these checks not run.

## BENCH-01 — Name the timed region and the cache/launch regime

**Applies when:** reporting latency or comparing kernels.

**Do:** choose among device-kernel time, graph replay time, full operator time and end-to-end serving time, and label it. Match cache warmth and launch mode to the deployment question. Exclude compilation/warmup from steady-state measurements while reporting setup separately.

**Source / evidence:** **observed benchmarking utilities**: FlashInfer [`testing/utils.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/testing/utils.py) has CUDA-event, CUPTI and graph methods with explicit cold-L2 controls. Triton [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) exposes warmup, repetition and quantiles. Source IDs: `flashinfer`, `triton`.

**Counterconditions:** cold-L2 is a benchmark regime, not automatically the correct serving regime. A graph-only comparison can hide host-side overhead; naive host wall time without synchronization can time enqueueing rather than GPU completion.

**Validate:** use the same timed region and inputs for baseline/candidate, multiple independent repetitions, median plus dispersion and workload coverage. Report kernel launches/reductions included, cache policy, graph mode and synchronization method.

For a source-level audit of fallback timing methods, CUPTI activity spans, rotating buffers and aliasing, read [measurement boundaries](../recipes/measurement-boundaries.md).

## BENCH-02 — Use profiler counters to test a hypothesis, then time without the profiler

**Applies when:** deciding between memory traffic, occupancy, instruction throughput or synchronization optimizations.

**Do:** write one bottleneck hypothesis, collect the minimum relevant counters/trace, and inspect generated resource usage. Run final timing outside replay-heavy profiling. Preserve launch dimensions, registers, shared memory and useful FLOP/byte accounting with the result.

**Source / evidence:** **official tooling behavior**: NVIDIA [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) describes replay, cache and clock controls. Source ID: `nsight-compute`. The hypothesis-driven loop is a **derived recommendation**.

**Counterconditions:** profiling can serialize work, flush caches or alter clocks; metrics can require multiple passes. Peak hardware throughput is neither a latency prediction nor proof of the limiting resource.

**Validate:** corroborate the proposed bottleneck with at least one targeted implementation/configuration change; compare standalone timing before/after, and explain regressions in other shape classes. Record profiler configuration when using its numbers.

## BENCH-03 — Report a decision over the workload distribution

**Applies when:** accepting a specialization or publishing a speedup.

**Do:** compare representative decode and prefill shapes, boundaries and frequent production cases. Keep per-case latency visible; add a weighted aggregate only when the workload weights are stated. Include startup and memory costs when the optimization changes them.

**Source / evidence:** **derived recommendation** grounded in shape-specific tuning in Triton [`matmul autotuning`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/03-matrix-multiplication.py) and FlashInfer's distinct execution regimes in [`decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py). Source IDs: `triton`, `flashinfer`.

**Counterconditions:** a geometric mean can conceal an unacceptable latency regression on a critical case. A throughput gain under high concurrency does not imply a single-request latency gain.

**Validate:** retain exact environment, commits, inputs, baseline and commands. State which regressions are accepted and why; route unsupported or unprofitable cases to the existing path instead of claiming a universal replacement.
