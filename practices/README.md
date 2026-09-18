# Practice map

Start with the findings below: they capture implementation details that are easy to lose when transferring an upstream optimization. Links lead to pinned evidence, branch conditions and validation routes. **These documents report source inspection, not local validation of the cited kernels.**

## Source details worth retrieving first

| Integration problem | Finding retained from upstream | Read |
|---|---|---|
| Attention backends agree on output shape but disagree numerically | FlashInfer cascade consumes **base-2 LSE**; internal unnormalized state and normalized merge inputs are different interfaces. FlashAttention's unequal-length causal mask uses bottom-right alignment. FlashInfer FA2 fixed-split determinism is not a graph guarantee; the same pin's `cute-dsl` branch also consumes the option without inheriting that documented determinism contract. | [ATT-01–03](attention-and-kv-cache.md), [backend traces](source-traces.md) |
| A mathematically equivalent diffusion fusion fails exact parity | SGLang's exact modulation path supports **hidden sizes 2048/4096**, preserves intermediate BF16 rounding and a specific reduction tree. A similarly named residual/norm implementation is **XPU-only**. QK/RoPE paths have cache-dtype, rotary-span and `pack_kv` gates. | [DIT-01–05](diffusion-transformer-fusions.md), [FUSE-02](fusion-and-reductions.md) |
| Sparse attention passes forward tests but fails boundaries or backward | “Full” tiles skip `mask_mod`, not every sequence-boundary check. Builders differ on out-of-bounds classification; list order can matter to the consumer. Five-point classification requires a suitable mask family. Deterministic backward needs write-order metadata and a compatible scheduling direction. | [SPARSE-01–04](block-sparse-attention.md), [BWD-01–03](backward-and-determinism.md) |
| A serving optimization silently changes routing or sampling | vLLM can bypass expert alignment under token/expert and quantization predicates; route weights and final reduction have explicit ownership. FlashInfer sequential top-k/top-p is not its joint mode, and a scalar-K fast path has different dispatch requirements from the fallback. | [MOE-01–04](moe-and-irregular-work.md), [SAMPLE-01–04](sampling.md) |
| A benchmark or paper suggests a gain the deployed path does not reproduce | FlashInfer CUPTI can fall back; activity timing is a span, and graph rotation can access fewer copies than allocated while breaking input aliases. FA4 tuning distinguishes SM103; delayed rescaling has a dtype bound; the inspected scheduler rejects `sort=True` despite the author's LPT discussion. | [Measurement helper audit](../recipes/measurement-boundaries.md), [FA evolution](../recipes/flashattention-evolution.md) |

These are snapshot-specific facts, not defaults to impose on another implementation. Read the exact caller and cited guard before adapting them. The broad topics below provide supporting reasoning; they are not additional evidence that a candidate is fast.

## Supporting topic map

| Task | Read | Rule IDs |
|---|---|---|
| Choose the algorithm or bottleneck hypothesis | [Algorithm and IO](algorithm-and-io.md) | IO-01–04 |
| Map tiles, memory, indices and launch geometry | [Tiling and layout](tiling-and-layout.md) | TILE-01–05 |
| Overlap copies/computation safely | [Async pipelines](async-pipelines.md) | ASYNC-01–04 |
| Implement/integrate attention and KV caches | [Attention and KV](attention-and-kv-cache.md) | ATT-01–05 |
| Implement training gradients and reproducibility | [Backward and determinism](backward-and-determinism.md) | BWD-01–03 |
| Build or adapt sparse masks and metadata | [Block-sparse attention](block-sparse-attention.md) | SPARSE-01–04 |
| Fuse pointwise operations or reductions | [Fusion and reductions](fusion-and-reductions.md) | FUSE-01–04 |
| Change numerical precision | [Low precision](low-precision.md) | PREC-01–04 |
| Add specialization, planning or graph replay | [Dispatch and graphs](dispatch-and-graphs.md) | GRAPH-01–04 |
| Validate and measure | [Correctness and benchmarking](correctness-and-benchmarking.md) | CHECK-01–02, BENCH-01–03 |
| Adapt image/video transformer non-attention ops | [Diffusion-transformer fusions](diffusion-transformer-fusions.md) | DIT-01–05 |
| Handle MoE routing and ragged expert work | [MoE and irregular work](moe-and-irregular-work.md) | MOE-01–04 |
| Optimize categorical/top-k/top-p sampling | [Sampling](sampling.md) | SAMPLE-01–04 |
| Inspect concrete caller-to-kernel dispatch chains | [Four source traces](source-traces.md) | TRACE-FA/FI/VLLM/SGLANG |

Rule IDs are stable retrieval handles. Numeric tuning thresholds and backend support described in a source trace belong to that pinned snapshot; reverify them before changing another checkout. The linked upstream files and test routes are the starting point for implementation, not permission to transplant a whole kernel without its caller contract.

For public negative results and debugging postmortems, read [failure evidence](../recipes/failure-driven-debugging.md).
