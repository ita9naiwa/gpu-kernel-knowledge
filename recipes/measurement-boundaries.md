# Benchmark-helper behavior and measurement boundaries

Tags: benchmark, CUDA events, CUPTI, CUDA Graph, cold cache, warm cache, aliasing,
Nsight Compute, kernel span, latency, end-to-end.
Evidence: pinned helper implementation and official profiler documentation.
These helper behaviors were inspected in source; no local timing result is claimed here.

## Name the interval first

| Question | Useful interval | What it cannot establish |
| --- | --- | --- |
| Did device execution get faster? | Matched kernel activity span or CUDA-event interval | Request latency improvement |
| Did graph replay get faster? | Replay interval with the real captured sequence | Cold-start compilation or capture cost |
| Is the wrapper faster? | Invocation including required layout conversion, planning and copies | Serving performance under unrelated traffic |
| Is serving better? | Matched request mix, concurrency, latency and throughput | Which individual kernel caused the change |

CUDA events timestamp positions in a stream. For tiny eager launches, an event
interval can include device idle gaps while the host submits work. It is not
simply host wall-clock duration. If other streams contribute work, ensure the
timed stream joins that work before its end event; otherwise the interval can
miss required execution.

## Read the benchmark implementation, not just its function name

In the pinned [FlashInfer timing helpers](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/testing/utils.py),
`bench_gpu_time_with_cupti` can fall back to graph/event timing when CUPTI setup
is unavailable. Record warnings and the effective method, not just the requested
method. Its activity processing uses earliest start to latest end for an
iteration's correlated activities; this is a span, not a sum of kernel durations.
Concurrent or separated kernels make that distinction material.

The graph helper repeats operations in one capture and divides replay time by
the repetition count. Its cold-cache path clones explicit GPU tensor arguments
and rotates the copies. A closure that hides the tensors gives this machinery
nothing to rotate. An argument object containing hidden tensors can have the
same problem. Inspect whether the helper traverses the actual input structure.

Two further source-derived caveats at this revision:

- The number of buffer copies can exceed `num_iters_within_graph`; capture
  references only the first `min(num_rotations, num_iters_within_graph)` copies.
  Allocated bytes therefore do not prove the replay touches enough distinct
  data to exceed L2. Check the addresses and bytes actually accessed.
- Recursive tensor cloning is per tensor occurrence. Distinct views of shared
  storage, or repeated references to one tensor, need not retain their alias
  relationship across cloned arguments. Validate in-place and aliased contracts
  before using this path. Rotating only independent inputs is a simpler case.

These are deductions from source control flow, not measured cache behavior or
a blanket rejection of the helper. The correct cache model depends on production
reuse. Report both warm and cold variants when that ambiguity affects a decision.

## Profiling is a separate experiment

[Nsight Compute's profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)
explains replay and default cache flushing. Its instrumentation, clock policy
and serialization can change the workload being observed. Compare baseline and
candidate under identical profiler settings, then collect decision timings
without profiler overhead. Preserve cache priming with an appropriate replay
mode when the experiment requires it. Do not use a host timer around an `ncu`
run as the kernel's performance result.

## Correctness must survive repeated invocation

Test mutable outputs, scratch buffers and counters after multiple replays,
not just the first call. A benchmark may accidentally time progressively
different inputs or a buffer that was never reset. Keep resets inside the timing
scope when deployment pays them each invocation; otherwise state the amortization.
For graphs, retained addresses and object lifetimes are part of correctness.

The [vLLM microbenchmark guidance](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/.agents/skills/kernel-microbenchmark/SKILL.md)
also separates kernel-only ablations from the full wrapper and requires
matched metadata. Its cold-cache defaults and rough throughput reference points
are project conventions, not universal acceptance thresholds.

## Minimal comparison protocol

1. Freeze input distribution, reference, tolerances, candidate versions, scope,
   cache policy and effective timing backend.
2. Finish compilation/autotuning and run correctness on ordinary and edge cases,
   including repeated calls or replay if those are used in deployment.
3. Alternate baseline/candidate order, such as A/B/B/A, over repeated rounds;
   retain raw samples and report median and spread. This is a local methodology
   recommendation, not a guarantee that thermal drift has been eliminated.
4. Recheck the best candidate on held-out workload shapes and the complete
   wrapper; inspect any speedup exceeding plausible work/traffic bounds.
5. Save failures and untested conditions with the result. Do not replace the
   deployment objective with a more flattering microbenchmark after the fact.
