# Asynchronous pipelines and synchronization

Keywords: cp.async, TMA, WGMMA, producer, consumer, barrier, phase, ring buffer, pipeline stage, warp specialization.

Architecture-specific operations require matching hardware and compiler support. The observations below were read from upstream; none were locally executed on a GPU.

## ASYNC-01 — Write the buffer ownership protocol before overlapping work

**Applies when:** global-to-shared copies overlap computation, shared-memory stages are recycled, or producer/consumer warps differ.

**Do:** for each stage specify who acquires empty storage, which event establishes data readiness, who waits, and which event proves all consumers have finished before reuse. Track stage index and phase/generation independently. Start with the matching CUTLASS pipeline abstraction and its tested participant counts.

**Source / evidence:** **documented contract**: CUTLASS [`pipeline.md`](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/pipeline.md) defines acquire/commit/wait/release; TMA pipeline commit may be a no-op because the transfer signals completion. Source ID: `cutlass`. The ownership worksheet is a **derived recommendation**.

**Counterconditions:** a thread barrier is not automatically an async-engine completion wait; a memory fence is not a rendezvous. The exact protocol depends on the primitive and scope. Do not substitute synchronization primitives by name similarity.

**Concrete SM100 participant boundary:** CUTLASS's pinned
[`02_mma_tma_sm100.cu`](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu#L275-L336)
uses one elected thread to issue both TMA loads and arms their barrier with the
**combined A+B byte count**. The MMA calls instead execute under a warp-level
guard: CuTe internally elects the issuing thread. These are distinct participant
contracts; do not copy the TMA single-thread guard mechanically around the MMA
wrapper. Separate TMA and MMA barriers maintain separate phase bits. The latter
is signaled with `umma_arrive` and waited before the next K tile overwrites A/B
shared memory; TMA readiness alone cannot authorize that reuse. This is the
SM100 `tcgen05` single-stage tutorial, not Hopper WGMMA or a measured overlapped
multistage pipeline.

**Validate:** one tile, fewer tiles than stages, exactly one ring wrap, multiple wraps, ragged tails, and repeated launches. Use race/synchronization checking where supported; successful execution once is insufficient evidence.

## ASYNC-02 — Release a stage after the asynchronous consumer is actually done

**Applies when:** an MMA or copy engine continues reading shared memory after its issue instruction returns.

**Do:** identify the completion primitive for the specific async operation. Place stage release after the last relevant completion, and document whether the stage contains operands, accumulators, or output storage. Audit any refactor that moves release earlier as a correctness change.

**Source / evidence:** **documented protocol**: CUTLASS [`consumer_release` contract](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/pipeline.md) requires completed consumption. **Observed implementation**: FlashInfer [`MergeStatesLargeNumIndexSetsKernel`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/cascade.cuh) waits for async copies and synchronizes before reads and before overwriting stages. Source IDs: `cutlass`, `flashinfer`.

**Counterconditions:** the FlashInfer example uses `cp.async`, not a complete TMA/WGMMA recipe. Copying its barriers into another instruction family does not establish correctness.

**Validate:** stress with changed stage counts and block scheduling pressure; check each producer/consumer dependency against the exact instruction documentation. A slower synchronized reference helps detect missing ordering.

## ASYNC-03 — Increase pipeline depth only to hide a demonstrated latency gap

**Applies when:** the kernel exposes copy/compute bubbles and stage count is tunable.

**Do:** measure a small stage-count sweep and inspect shared-memory/register growth. Account for prologue and drain overhead. Re-evaluate tile shape with depth because stage storage can reduce resident work.

**Source / evidence:** **documented mechanism**: CUTLASS [`Efficient GEMM`, pipelining](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/efficient_gemm.md); **upstream research**: [FlashAttention-3](https://arxiv.org/abs/2407.08608) overlaps data movement and computation using Hopper features. Source IDs: `cutlass`, `flashattention`. The tuning rule is a **derived recommendation**.

**Counterconditions:** short reductions may never reach a steady state; excessive stages consume resources; compute-bound kernels may gain nothing. Hopper results are not measurements on Blackwell or consumer GPUs.

**Validate:** separate short and long reductions; inspect achieved occupancy, waits/stalls, shared memory and spills. Keep changes only if total latency improves on the target workload.

## ASYNC-04 — Make tail predication consistent with synchronization participation

**Applies when:** the final tile is partial, some rows are empty, or persistent workers process tasks of differing sizes.

**Do:** distinguish invalid data from inactive synchronization participants. Predicated loads may skip data movement while the participating threads still owe barriers. Initialize unused data when consumers can read it, or prove the read is masked. Drain outstanding async work before repurposing storage.

**Source / evidence:** **observed implementation**: FlashInfer [`PersistentVariableLengthMergeStatesKernel`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/cascade.cuh) includes a loop-start barrier specifically to avoid hazards for small numbers of index sets, plus copy waits before final reductions. Source ID: `flashinfer`. Generalization to other kernels is a **derived recommendation**.

**Counterconditions:** uniform early return for an entire block can be legal; divergent return across threads participating in a block-wide protocol can deadlock or race. Analyze the actual participant set.

**Validate:** persistent task sequences alternating zero, one, many and partial tiles, not merely a uniform large workload. Use randomized lengths and rerun enough times to expose scheduling-sensitive bugs.
