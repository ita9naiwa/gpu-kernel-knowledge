# Algorithm choice and IO cost

Keywords: arithmetic intensity, roofline, HBM, SRAM, recomputation, online softmax, prefill, decode, launch overhead.

Supporting principles for the [source-specific findings](README.md). These are broad algorithmic deductions; upstream results are not local measurements.

## IO-01 — Count materialized intermediates before counting instructions

**Applies when:** a tensor expression materializes a large temporary used only by the next operation.

**Do:** the cited attention loop replaces a full score/probability matrix with per-row maximum, normalizer and output accumulator. Identify whether the target API exposes that matrix before applying the same elimination.

**Source / evidence:** **upstream algorithm**: [FlashAttention](https://arxiv.org/abs/2205.14135) reduces HBM traffic with tiled exact attention. **Observed implementation**: Triton [`_attn_fwd_inner`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/06-fused-attention.py#L49-L111) updates row maxima, normalization and output accumulators online. Source IDs: `flashattention`, `triton`.

**Counterconditions:** an intermediate required by another consumer or by the public API may need materialization. Excessive register/shared-memory demand can make fusion slower. A byte count is a lower-bound model, not a measured latency.

**Validate:** output and auxiliary state against a small materializing reference; global traffic, spills and peak memory against the complete upstream-library operation.

## IO-02 — Recompute cheap state when storing it costs more

**Applies when:** backward or a multi-pass operator can reconstruct an intermediate from compact saved state.

**Do:** attention backward reconstructs probability tiles from Q/K and saved normalization data. Preserve its units and any dropout/RNG state; see the concrete [backward preparation trace](backward-and-determinism.md).

**Source / evidence:** **upstream algorithm**: [FlashAttention](https://arxiv.org/abs/2205.14135) combines tiling and recomputation. **Observed implementation**: Triton [`fused-attention tutorial` backward](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/06-fused-attention.py) reconstructs probability tiles using stored normalization data. Source IDs: `flashattention`, `triton`. General use is a **derived recommendation**.

**Counterconditions:** expensive or nondeterministic recomputation, mutation of inputs, RNG changes, or a numerically different reconstruction may violate the required result. Do not discard state without tracing backward and other consumers.

**Validate:** forward and gradients separately; dropout seed/offset replay where supported; peak memory and complete forward-plus-backward latency. Saving less memory alone does not establish a faster training step.

## IO-03 — Classify prefill, decode and tiny operators separately

**Applies when:** selecting an attention implementation or interpreting a kernel benchmark.

**Do:** distinguish reuse, KV-read volume, available blocks and launch gaps. FlashInfer's split-KV mechanism addresses insufficient parallelism; it is not evidence that all decode workloads need splitting. See [ATT-03](attention-and-kv-cache.md).

**Source / evidence:** **documented mechanisms**: [FlashInfer recursive attention](https://docs.flashinfer.ai/tutorials/recursive_attention.html) describes insufficient block parallelism in long-context inference; [FlashAttention-3](https://arxiv.org/abs/2407.08608) targets Hopper attention execution overlap. Source IDs: `flashinfer`, `flashattention`. The classification is a **derived recommendation**, not an unconditional statement that every decode is memory-bound.

**Counterconditions:** GQA ratio, batch size, KV precision, head dimensions, cache residency and hardware shift the balance. A roofline based on peak bandwidth does not diagnose poor occupancy or launch gaps.

**Validate:** record `B,Q,K,Hq,Hkv,D`, dtype, layout, masking and replay mode for each row. Compare end-to-end operator and isolated device times to expose host overhead.

## IO-04 — Optimize the path the serving workload actually invokes

**Applies when:** a library contains many backends, shape branches or compiled specializations.

**Do:** follow public API → dispatch predicates → kernel → postprocessing. The [four source traces](source-traces.md) retain actual guards and conversions; use those instead of inferring execution from an available kernel's name.

**Source / evidence:** **observed structure**: FlashInfer [`decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py) separates multiple backend paths and `plan`/`run`; vLLM [`CUDA Graph design`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/docs/design/cuda_graphs.md) distinguishes runtime modes and batch compositions. Source IDs: `flashinfer`, `vllm`. The trace requirement is a **derived recommendation**.

**Counterconditions:** upstream docs can describe an older implementation, and an available kernel may never be selected under the current configuration. Do not report a kernel microbenchmark as serving throughput.

**Validate:** retain the effective kernel/configuration and include per-call repacking, planning, quantization and reductions in the measured boundary.
