# Diffusion-transformer fusion: preserve the actual eager chain

Keywords: image, video, DiT, AdaLN, RMSNorm, LayerNorm, RoPE, SiLU, SwiGLU, residual gate, modulation, BF16 rounding, trajectory parity.

The principles here transfer to image/video transformers. They do not identify the user's model or establish performance on it. SGLang has a dedicated [diffusion kernel collection](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/diffusion/README.md) with explicit numerical contracts. **Evidence is source inspection and upstream-reported checks only; no local image/video quality or GPU validation was run.** Source ID: `sglang`.

## DIT-01 — Fuse modulation without silently deleting eager rounding boundaries

**Applies when:** replacing `norm(x) * (1 + scale) + shift`, possibly preceded by `residual + gate * update`.

**Do:** write the existing chain with explicit dtype casts. If bitwise parity is required, preserve each materialized BF16 rounding boundary even when the intermediate remains in an FP32 register. Decide separately whether the norm's reduction tree must match the selected upstream kernel.

**Source / evidence:** **observed implementation**: SGLang [`rmsnorm_scale_shift_bitexact.py`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/diffusion/norm/rmsnorm_scale_shift_bitexact.py#L98-L146) explicitly rounds gate multiplication, residual addition, normalization output, `1+scale`, modulation product and final addition. Its norm implementation also mirrors a particular reduction order. This is more than algebraic fusion.

**Counterconditions:** the legality predicate is intentionally narrow: CUDA, contiguous BF16 `(B,S,H)` inputs, BF16 weight, per-batch `(B,1,H)` modulation, and `_threads_per_row` implies **H=2048 or H=4096** in this snapshot. Do not generalize the bit-exact claim to arbitrary widths, dtypes, RMSNorm backends or compiler versions. A plain FP32 fusion can be acceptable under an explicitly approximate contract.

**Validate:** compare `torch.equal` only for the promised exact path and compare both returned residual and modulated output. Cover gate on/off, both supported widths, nontrivial scale/shift and changed batch/sequence shapes. For an approximate path, evaluate per-block errors and the complete denoising trajectory under fixed inputs/seeds; an isolated small RMSNorm error does not quantify final image/video differences.

**Integration boundary:** SGLang's [`BitExactFusionGate`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/diffusion/sites/bitexact_gate.py#L63-L170) distinguishes a one-time check from checks per dispatch signature. First-sight comparison runs the reference and synchronizes with the host, so it must be skipped during compile tracing and graph capture. `can_attempt_once()` implements those guards for once-for-all mode; per-signature callers must implement them themselves. The fused RMSNorm wrappers also rely on their caller to run the separate eligibility predicate. A successful first call is a deployment guard, not a substitute for the test matrix; choose the signature granularity to match shape-dependent reference dispatch and warm it before capture.

## DIT-02 — QK normalization plus RoPE needs a layout and rounding contract

**Applies when:** combining Q/K norm, rotary embedding and optional QKV packing before attention.

**Do:** record head dimension, rotary span, pairing convention, position/frequency layout, cache dtype, output layout and whether normalization is rounded before rotation. Pass explicit semantics into specialization rather than inferring them from tensor shapes.

**Source / evidence:** **observed wrapper**: [`qknorm_rope_jit.py`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/diffusion/rope/qknorm_rope_jit.py#L27-L125) specializes `head_dim`, `rope_dim`, `is_neox`, input/cache dtype, `round_norm_before_rope` and cache-width behavior. Its predicate admits head dimensions 64/128/256, requires the rotary span to fit the head and satisfy per-thread lane divisibility, and rejects incompatible packing/cache combinations.

**Counterconditions:** NeoX half-pairing and interleaved even/odd pairing differ. Video/3D positional schemes can allocate dimensions among axes; a 1D LLM RoPE kernel is not interchangeable merely because the final tensor shape matches. A compile success does not validate the frequency ordering.

**Validate:** hand-check a small rotary pair, test nonzero positions and each axis independently for multidimensional RoPE, verify untouched nonrotary tails, then compare the complete Q/K tensors and downstream attention. Test normalization rounding mode on/off as different contracts.

## DIT-03 — A rotate-half fusion can remove concatenation while retaining eager arithmetic

**Applies when:** an eager RoPE path constructs `cat(-x2,x1)` and performs separate products/addition.

**Do:** compute partner addresses directly, retain the eager product-rounding boundaries where exactness is required, and copy columns beyond the rotary span unchanged. Reuse precomputed cosine/sine only while their position, dtype and frequency configuration remain valid.

**Source / evidence:** **observed implementation**: [`rope_rotate_half_bitexact.py`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/diffusion/rope/rope_rotate_half_bitexact.py#L50-L109) uses per-row/head addressing, separately rounded products, tail copies and BF16/contiguity/rotary-span guards.

**Counterconditions:** fusing multiply/add into one FMA can change a two-operation eager result. The upstream bit-exact claim is for a selected BF16 chain with precomputed BF16 trig tensors, not for recomputing trig in another precision. Guard the platform before attempting inline-PTX helpers on non-CUDA backends.

**Validate:** even rotary widths, partial/full rotary spans, multiple heads, distinct cosine/sine values in each half, and tail equality. Compare the cost of trig preparation at its actual reuse frequency; do not claim that cost disappears when each call needs new positions.

## DIT-04 — Fuse separate gate/up activations without adding a packing pass

**Applies when:** separate GEMMs produce the gate and up tensors for `silu(gate) * up`.

**Do:** prefer a kernel accepting the existing two buffers to concatenating them just to reuse a packed-input API. Preserve the SiLU output cast before multiplication if matching an eager BF16 chain is required. If the inputs already share a packed layout, use its direct addressing path.

**Source / evidence:** **observed implementation**: [`silu_mul_bitexact.py`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/diffusion/activation/silu_mul_bitexact.py#L27-L74) provides separate-input and packed-input kernels. The separate-input eligibility predicate requires same-shape/device, contiguous, nonempty CUDA BF16 tensors; the packed wrapper has its own rank/stride guards.

**Counterconditions:** a packed GEMM epilogue may already avoid the intermediate; compare with that existing path first. FP16/FP32 behavior is not established by this BF16 implementation, and an upstream random-value equality comment is not an exhaustive proof over infinities/NaNs.

**Validate:** existing buffer layout, supported tails, zero/large activation values, sign changes, and actual first-call/reference parity. Include any `.contiguous()` or packing cost in the full-chain benchmark.

## DIT-05 — Inspect platform and broadcast guards before copying a fusion

**Applies when:** moving a kernel between NVIDIA, AMD, Intel or model-specific implementations.

**Do:** read `can_use_*`, compile flags and host validation before the compute body. Keep per-token, per-batch and per-frame modulation distinct; they imply different address maps. Derive a new specialization only from the requested target contract.

**Source / evidence:** **observed contrast**: SGLang [`scale_residual_norm_scale_shift_triton.py`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/diffusion/norm/scale_residual_norm_scale_shift_triton.py#L76-L119) has an eligibility predicate for **XPU**, BF16/FP16 contiguous batch-one rank-three input and hidden size ≤8192. Gate is the integer 1, a `(1,1,H)` tensor, or a `(1,F,1,H)` tensor with sequence length divisible by F; scale/shift each contain one or H values, and weight/bias must be supplied together. Residual dtype must match input, but the predicate does not require every modulation/affine tensor to have that dtype. The bit-exact RMSNorm path above explicitly gates CUDA because its numerical helpers use inline PTX.

**Counterconditions:** a file named `triton` is not proof of cross-platform deployability. This XPU example computes LayerNorm mean and centered variance, whereas the earlier RMSNorm example uses mean square; swapping them changes the model. It stores a dtype-rounded residual but computes normalization from the still-unrounded intermediate, and does not reproduce the earlier BF16 boundary chain. Its existence therefore does not establish eager bitwise parity. Calling its launch wrapper directly does not run the separate eligibility predicate.

**Validate:** formula first, then each platform/dtype/broadcast predicate, then benchmark on the intended device. Retain a fallback for unsupported shapes or failed parity. No performance conclusion follows from the existence of a fusion in another backend.
