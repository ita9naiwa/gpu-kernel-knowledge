# Source audit of the initial practice rules

Audit date: 2026-09-18. Method: inspect the pinned upstream file bodies already fetched during authoring, compare the rule's claim and linked location, and separate observed behavior from recommendations. This is a documentation/source audit, not GPU validation.

| Rule | Source inspected | Result |
|---|---|---|
| ATT-01 | FlashInfer `attention/cascade.cuh` and `state.cuh` | Base-2 LSE, `exp2`/`log2`, normalized merge inputs and empty-state handling support the claim. Small FP64 cases plus a trusted baseline at scale replace an impractical requirement for a huge materialized reference. |
| ATT-02 | FlashAttention README causal examples | Bottom-right alignment and zero output for all-masked rows are explicit. Scope remains this API's semantics. |
| ATT-03 | FlashInfer `decode.py` fixed-split docs and guards | Units are pages and graph compatibility is not guaranteed. **Later consumer-review correction:** the deterministic-reduction documentation is FA2-specific, but the same pin's `cute-dsl` branch also consumes positive `fixed_split_size`; the original backend-exclusivity wording was too broad. ATT-03 now links both branches. |
| TILE-01 | CUTLASS `efficient_gemm.md` threadblock discussion | Reuse versus edge waste/grid parallelism tradeoff is supported. No numeric tile recommendation or occupancy-as-objective claim was added. |
| TILE-03 | Triton matmul and softmax tutorial kernels | Explicit strides, K masks, output masks and negative-infinity softmax padding support the rule. Modulo loads are not recommended for arbitrary reductions. |
| ASYNC-01 | CUTLASS `pipeline.md` | Acquire/commit/wait/release and possible TMA commit no-op are supported. Exact primitive/scope caveat retained. |
| ASYNC-02 | FlashInfer cascade async staging and CUTLASS release contract | Copy waits and barriers occur before consuming/reusing storage. Rule does not present a cp.async example as a TMA/WGMMA protocol. |
| FUSE-02 | SGLang fused-add-RMSNorm CUDA kernel and HF reference test | **Corrected:** replaced weak generic softmax evidence with direct cast-order branches and mutation of both output tensors. |
| PREC-03 | Triton attention scale/exp2 path; FlashInfer state LSE | Base conversion claim supported. Public wrapper conversion remains an explicit item to verify independently. |
| GRAPH-01 | FlashInfer decode custom-op setup and `plan()` contract | Planning is excluded from the captured/compiled run path in this wrapper. No blanket claim about every API called `plan`. |

Source pins and rule links live in the [practice map](README.md) and [catalog](../sources/catalog.json). The table identifies inspected symbols without duplicating those canonical citations.

## Additional reliability corrections

The sanitizer source ID now matches `compute-sanitizer`; relative L2 uses the same unregularized ratio plus explicit zero-reference policy as the experiment template. A denominator floor is separately labeled stabilized relative L2.

The Triton matmul tutorial's `allclose` path prints failure rather than making the process fail. CHECK-01 now says to turn acceptance criteria into assertions in the target harness. A tutorial's zero exit code alone is insufficient evidence.

CHECK-02 now requires an explicit nonzero Compute Sanitizer `--error-exitcode` plus retained findings, since the documented default is zero. These static practice derivations do not establish GPU measurements on a new workload.

The sparse-mask builder emits ordered lists and the inspected consumer applies boundary masking to the first reverse-traversed block. SPARSE-02 now states that arbitrary index permutations need a correctness proof even when they represent the same mathematical set.

## Independent recipe audit

[FlashAttention evolution](../recipes/flashattention-evolution.md) was checked against pinned `softmax.py` lines 363–454, `flash_fwd_sm100.py` lines 78–127 and 2188–2208, `prepare_scheduler.py` lines 35–62, and `interface.py` lines 4176–4204. Native/emulated exponential selection, delayed-rescaling precision guard and the narrow LPT-sort rejection match those bodies. Architecture and path limitations are retained; no correction was needed.

[Measurement boundaries](../recipes/measurement-boundaries.md) was checked against pinned FlashInfer `testing/utils.py`: recursive cloning at lines 137–177, CUPTI fallback at 1042–1094, correlated activity span at 1269–1318, and graph rotation/capture at 1442–1500. Its distinction between allocated and accessed buffers, lost alias relationships, effective timing backend, and activity span versus duration sum is supported. These remain source-derived caveats rather than measured cache or timing results.

## 2026-09-18 targeted guard and consumer review

Re-fetched pinned SGLang diffusion kernels, numerical helpers, `BitExactFusionGate`, fused-add RMSNorm host/kernel/test and `RMSNorm.forward_cuda`; FlashInfer sampling wrappers/state validation; and vLLM assignment, GEMM epilogue and weight/reduce consumers. This sampled review corrected or made explicit:

- DIT-01/05: per-signature compile/capture guards belong to callers; XPU LayerNorm fusion uses an unrounded norm input despite storing a rounded residual, with narrower shape/platform guards and no universal matching-modulation-dtype predicate.
- FUSE-02: upstream HF comparison is tolerance-based; caller width/dtype policy is narrower than C++ launch capability, and post-residual association is part of the contract.
- SAMPLE-03/04: omitting either explicit RNG value replaces both from the generator; tensor length follows output batch size. Compact top-k requires Python-integer K and still performs multiple selection/normalization/sampling/remapping operations.
- MOE-03: routing weight is applied in a GEMM epilogue after bias; contiguous reduction mutates expert outputs, delegated reduction needs an owner, and arithmetic NoOP may still copy to the required output.

Rule-local links retain exact revisions and inspected boundaries. These findings do not establish every backend, invalid-input outcome, RNG distribution, race freedom or end-to-end performance. No GPU execution or generated examples were used.

## 2026-09-18 attention state and split-policy review

Re-fetched pinned FlashInfer `cascade.cuh`, `state.cuh`, `math.cuh`, paged-decode
plan guards and backend calls; and FlashAttention's forward configuration and
completion path. ATT-01 now states that the textbook merge formula needs an
empty-state guard, the pin uses IEEE infinity, and zero weights do not sanitize
invalid output payloads. ATT-03 retains the corrected positive `fixed_split_size`
handling in `cute-dsl`, distinguishes frozen graph-shape checks from split-policy
compatibility, and records that branch's `reduction` argument. TRACE-FA now
records the differing-head-dimension override of requested split choices.
Rechecked FI compact sampling, vLLM low-token assignment and SGLang residual
rounding bodies against the corresponding trace rows; no contradiction found in
those sampled bodies. This is source inspection, not runtime validation.

## Limits

DeepGEMM PREC-02 was additionally traced through the pinned scale API, host layout
helper and layout tests on September 18. The narrower prepacked K-granularity
gate, independently padded groups, aligned-vs-raw K distinction and unsynchronized
upper-bound allocation are source facts. No quantized kernel was executed.

The initial table audits ten rules; the later sections record additional targeted reviews. Neither is exhaustive verification of every upstream branch. Linked commits are snapshots and may differ from the target checkout. Some recommendations deliberately extrapolate from a mechanism; their evidence labels and counterconditions must travel with retrieved excerpts. No GPU sanitizer, numerical parity, performance or model-quality result is implied by this audit.
