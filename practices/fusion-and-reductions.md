# Fusion and reductions

Keywords: softmax, RMSNorm, layer normalization, epilogue, residual, reduction, persistent program, numerical stability.

## FUSE-01 — Fuse a producer and consumer when it removes real traffic

**Applies when:** pointwise/reduction chains or GEMM epilogues create avoidable intermediates.

**Do:** the cited softmax tutorial holds a padded row on chip; that footprint is part of its applicability. Compare against the complete unfused chain, not one of its kernels.

**Source / evidence:** **observed teaching implementation**: Triton [`fused softmax`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/02-fused-softmax.py) loads a row, computes its reduction and writes output in one kernel; its intended regime is rows fitting on chip. **Documented mechanism**: CUTLASS [`GEMM epilogue`](https://github.com/NVIDIA/cutlass/blob/f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c/media/docs/cpp/efficient_gemm.md) separates output transformation from the mainloop. Source IDs: `triton`, `cutlass`.

**Counterconditions:** increased live ranges, spills, lost parallelism, extra recomputation, or an unfused intermediate with multiple consumers can defeat the traffic saving.

**Validate:** complete-chain latency and temporary allocation costs; compiled register/shared-memory use and spill traffic.

## FUSE-02 — Preserve normalization and residual update semantics

**Applies when:** fusing residual addition with RMSNorm/LayerNorm or another in-place update.

**Do:** preserve SGLang's two-output mutation and selected cast order. Its HF-style branch retains unrounded residual sums for normalization and casts the normalized activation before multiplying by weight; this is different from merely casting a final algebraically equivalent expression. Trace the selected branch before matching its reference.

**Source / evidence:** **observed implementation**: SGLang [`fused_add_rmsnorm.cuh`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/jit/csrc/elementwise/fused_add_rmsnorm.cuh#L44-L135) has distinct cast-order branches and writes both input and residual. [`Its test`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/test/registered/kernels/ops/layernorm/test_fused_add_rmsnorm.py#L38-L45) spells out the HF-style reference. The test uses tolerances, not `torch.equal`: normalized-output `atol=1e-2`, `rtol=1.5e-2` for HF mode and `1e-2` otherwise; residual uses `1e-2` for both. Cast-order compatibility is not a bitwise guarantee. Source ID: `sglang`. The contract checklist is a **derived recommendation**, not a claim that all norm kernels use this precision policy.

**Dispatch boundary:** the Python [eligibility helper](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/layernorm/norm.py#L69-L70) requires positive hidden size, a multiple of 16, and ≤8192. The C++ launcher's potentially wider ≤12288 regime when its vector width is 32 bytes does not expand that caller policy. The [CUDA HF residual caller](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/srt/layers/layernorm.py#L568-L598) additionally checks FP16/BF16 input, matching weight dtype and matching optional post-residual dtype. C++ validates matching input/residual/weight dtype and device plus contiguous 2D activation layouts; the Python branch is not an independent guarantee that mixed residual dtypes will fall back. Post-residual addition is performed before this kernel as `residual + post_residual_addition`; reassociating it to `(input + residual) + post_residual_addition` changes the arithmetic contract.

**Counterconditions:** changing the rounding policy can be allowed, but needs its own tolerance and model-level acceptance. An in-place optimization is invalid when another consumer needs the original tensor.

**Validate:** compare all mutated outputs, not only normalized output; include near-zero variance, large values, non-power-of-two widths, supported alias cases and rejected alias cases. Test epsilon placement with a hand-computed small input.

## FUSE-03 — Use a multi-stage reduction when one-program storage stops scaling

**Applies when:** wide rows or long reductions exceed an efficient register/shared-memory footprint.

**Do:** use the compiled resource footprint, as the cited tutorial does, before proposing a split. A split softmax requires maxima/normalizers and weighted state, not an average of partial outputs; follow [ATT-01](attention-and-kv-cache.md).

**Source / evidence:** **observed implementation**: Triton [`softmax` occupancy calculation](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/02-fused-softmax.py#L142-L179) uses compiled register/shared-memory usage to size persistent work. FlashInfer [`attention states`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/state.cuh) supplies a stable merge state. Source IDs: `triton`, `flashinfer`. The crossover search is a **derived recommendation**.

**Counterconditions:** extra launches and workspace can make splitting worse for small rows or plentiful independent rows. Tutorial occupancy arithmetic is illustrative and should not replace compiler/device limits for another architecture.

**Validate:** widths around powers of two and resource cliffs; total reduction time and temporary memory; cancellation/extreme-value cases; different split counts and deterministic requirements.

## FUSE-04 — Use explicit neutral states for empty or masked reductions

**Applies when:** padding, sparse blocks, ragged data or all-masked attention rows.

**Do:** define the empty result in the operator contract. Prevent undefined arithmetic such as `-inf - -inf` and `0/0` from leaking into final outputs. Do not replace masks with zero logits, which creates probability mass.

**Source / evidence:** **observed implementation**: FlashInfer [`state_t::merge` and `normalize`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/state.cuh#L53-L84) guards empty-state arithmetic; [`MergeStatesKernel`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/cascade.cuh) writes zero output and negative-infinity LSE for zero states. Source ID: `flashinfer`.

**Counterconditions:** generic softmax of an empty/all-negative-infinity row may have another documented convention. Match the chosen API rather than silently imposing zero.

**Validate:** zero valid entries, one valid entry, masked infinities, and a mixture of empty/nonempty partial states; explicitly check finiteness or the expected infinities before comparing numerical errors.
