# Bottleneck triage from application traces

Tags: bottleneck, roofline, bandwidth, occupancy, spills, launch overhead,
small GEMM, split-K, prefill, decode, prioritization, Amdahl.
Evidence: a local decision procedure linked to upstream mechanisms. No measured
performance claims are made here.

## Start with the critical path

Read the application trace before choosing a kernel. Distinguish host gaps,
device work, transfers and synchronization. [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
provides the timeline context; kernel counters alone cannot say whether a kernel
lies on the application's limiting path. Record the actual call frequency and
shape distribution, not only the slowest isolated shape.

## Use a signal to choose a candidate, not to declare the cause

| Observation | Check before editing | First bounded experiment | Reject if |
| --- | --- | --- | --- |
| Tiny kernels separated by host gaps | Real graph/eager mode and wrapper cost | Capture the legal region or fuse one producer/consumer pair | Required semantics or replay lifetimes break; wrapper gets slower |
| Large DRAM traffic around a temporary | Whether another consumer requires it | Keep that intermediate on chip | Spills or additional passes erase the traffic saving |
| Too few output tiles with a long reduction | Grid size, occupancy limits, actual K work | Smaller output tile or bounded split-K/split-KV | Merge/workspace cost exceeds extra parallelism benefit |
| Memory stalls with low useful issue rate | Coalescing, dependencies and stage lifetimes | One layout or pipeline-depth change | More stages reduce residency or short-workload latency regresses |
| Lower kernel time but unchanged request time | Dispatch, conversions, planning and critical path | Time the complete wrapper and confirm the selected kernel | Claimed improvement exists only in the isolated ablation |

These are hypotheses. Stall counters are meaningful in context: NVIDIA's
[Nsight Compute sections guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)
distinguishes scheduler, memory and compute analyses. A high stall percentage is
not automatically the limiting cause, and high occupancy is not the objective.
Use the section names available in the installed profiler rather than assuming
every GPU exposes the same metrics.

## Relate the experiment to an actual implementation

For small-M/N GEMM, [CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/efficient_gemm.md)
provides tile and split-reduction mechanisms. For cache-ordering candidates,
the [Triton GEMM tutorial](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/03-matrix-multiplication.py#L94-L145)
maps program IDs into grouped output tiles. Borrow the mechanism and boundary
handling, not its demonstration's speedup or tuning constants. In particular,
the final group can be shorter; dividing by the nominal group size can map the
tail incorrectly. Launch order is a locality hint, never an inter-block
synchronization guarantee.

For long-context decode, start with the
[attention practices](../practices/attention-and-kv-cache.md). Splitting KV
creates parallel partial states, but the merge belongs in the measured path.
For fusion, read [fusion and reductions](../practices/fusion-and-reductions.md)
before dropping an intermediate used by backward, aliasing or a second consumer.

## Write the experiment so failure is useful

Use the existing [attempt record](../templates/experiment.md) to preserve the
observation, hypothesis, acceptance contract and raw evidence.

Keep an unchanged baseline and a representative held-out shape. If the expected
counter improves but latency does not, revise the bottleneck hypothesis rather
than stacking another optimization on top. Use
[measurement boundaries](measurement-boundaries.md) to rule out a changed
timing method, cache state or warmup as the explanation.
