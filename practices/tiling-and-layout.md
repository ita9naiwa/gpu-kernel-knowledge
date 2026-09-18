# Tiling, layouts and resource limits

Keywords: GEMM, CTA, warp, coalescing, shared memory, bank conflicts, register pressure, occupancy, grouped ordering, tail masks.

These are source-grounded candidate-selection rules, not performance promises. This document reports no local GPU validation of the cited implementations; separate experiments retain their own workload scope.

## TILE-01 — Choose tile shape from reuse and available parallel work together

**Applies when:** GEMM, attention blocks, or another tiled reduction has tunable CTA/warp shapes.

**Do:** estimate output-tile count and resource usage alongside bytes reused. Sweep a small set of legal tiles around the workload's actual dimensions. A large tile reduces repeated input traffic but can waste work on edges or leave too few blocks to occupy the machine.

**Source / evidence:** **documented mechanism**: CUTLASS [`Efficient GEMM`, threadblock-level discussion](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/efficient_gemm.md#L64-L84). Source ID: `cutlass`. The sweep policy is a **derived recommendation**.

**Counterconditions:** small-M/N, large-K workloads can need split reductions; large regular workloads can benefit from much larger tiles. Highest theoretical occupancy is not the objective; useful work per unit time is.

**Validate:** include just-below/at/above tile boundaries and small-grid cases. Record registers, shared memory, spill evidence, active warps and latency. Recheck resource usage after changing pipeline depth or fusion, even when the tile is unchanged.

## TILE-02 — Design a layout for the consumer instruction and the writer

**Applies when:** tensor-core operand preparation, shared-memory staging, transposes or vectorized loads/stores.

**Do:** establish how adjacent lanes map to addresses, how shared-memory banks are touched, and what layout/alignment the chosen MMA/load primitive accepts. A coalesced global load alone does not prove an efficient shared-memory consumer. Reuse upstream layout abstractions for the same instruction family instead of inventing a swizzle from a diagram.

**Source / evidence:** **documented mechanism**: CUTLASS [`Efficient GEMM`, warp-level and epilogue sections](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/efficient_gemm.md) distinguishes compute-friendly accumulator placement from efficient output movement. Source ID: `cutlass`. The address audit is a **derived recommendation**.

**Counterconditions:** a swizzle is architecture/instruction/layout specific. A layout tuned for `mma.sync` is not automatically valid for WGMMA, TMA or Blackwell tensor-memory operations. Overhead can dominate a tiny transpose.

**Validate:** legal and misaligned inputs per API contract; transposed and strided cases if supported; compiler-generated load/store instructions; memory transactions and shared-bank conflicts where available. Reject unsupported layouts early rather than reinterpret them silently.

## TILE-03 — Preserve semantic stride and tail handling before specializing

**Applies when:** translating tensor code into explicit pointer arithmetic or adding contiguous fast paths.

**Do:** express every address using declared strides or enforce an explicit contiguity contract. Give reduction tails their neutral values, and mask stores independently. For sum/GEMM tails the neutral contribution is zero; for a max reduction it is negative infinity. Modulo-based duplicated loads are only safe where invalid output lanes are excluded and reduction operands remain correct.

**Source / evidence:** **observed implementation**: Triton [`matrix multiplication tutorial`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/03-matrix-multiplication.py) uses explicit strides, wraps M/N loads, masks K loads and masks output stores. [`softmax tutorial`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/02-fused-softmax.py#L85-L111) uses negative infinity for padded inputs. Source ID: `triton`.

**Counterconditions:** a supported domain may intentionally require contiguous last dimensions and aligned sizes. Make those guards part of dispatch; do not claim arbitrary-stride support because a stride argument exists.

**Validate:** prime dimensions, dimensions of one, non-unit supported strides, sliced tensors with nonzero storage offsets, and exact tile boundaries. Include invalid inputs to verify guards, not only successful launches.

## TILE-04 — Change program ordering when it creates measurable cache reuse

**Applies when:** neighboring output tiles share input tiles and L2 capacity/locality matters.

**Do:** compare grouped tile traversal with simple row-major traversal while holding arithmetic and tile shape fixed. Evaluate the actual shape distribution; group-size tuning belongs with tile tuning.

**Source / evidence:** **observed implementation / upstream example**: Triton [`matrix multiplication tutorial`, L2 cache ordering](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/03-matrix-multiplication.py#L94-L145) implements grouped program-ID mapping. Source ID: `triton`. It is an example, not a guaranteed speedup on another GPU.

**Counterconditions:** tiny workloads, different operand aspect ratios, competing workloads, or already-persistent schedulers may favor another traversal. Do not introduce inter-block correctness dependencies based on launch order.

**Validate:** same tile configuration, warm and representative cache states, and L2/DRAM traffic if profiling is available. Ensure the last incomplete group covers each output tile exactly once.

## TILE-05 — Widen address operands before arithmetic and bound every grid dimension

**Applies when:** long-context KV-cache addressing, large tensors/strides, flattened token/expert work, or launch grids grow with user-configured sequence length.

**Do:** bound the largest byte/element address intermediate, not only the logical index. Cast to int64 **before** a potentially overflowing multiply or addition. Validate each launch-grid dimension against device limits; for the CUDA Triton case discussed upstream, grid dimensions 1 and 2 must not exceed 65,535. Flatten or tile a growing dimension rather than relying on typical batch/sequence sizes.

**Source / evidence:** **documented project guidance**: vLLM [`triton-kernel-writing`, launch and indexing](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/.agents/skills/triton-kernel-writing/SKILL.md); **observed implementation**: [`fused_moe_kernel`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L402-L421) widens token/expert offsets before stride products. Source ID: `vllm`.

**Counterconditions:** casting the final result cannot repair an earlier overflow. Moving all token work to grid axis zero can be legal yet inefficient for prefill; legality and useful work per program are separate decisions. Device/backend limits should be checked for the actual target.

**Validate:** boundary arithmetic around signed 32-bit range with a host reference, actual large-stride supported GPU cases, and launch sizes on both sides of the permitted grid bounds. Inspect generated index arithmetic where possible and run memory checking. Unsupported giant shapes should reject clearly instead of wrapping addresses.
