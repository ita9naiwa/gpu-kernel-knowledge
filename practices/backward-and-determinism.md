# Backward kernels and deterministic reductions

Keywords: backward, gradients, dQ, dK, dV, dLSE, atomic reduction, scratch initialization, determinism, semaphore, GQA.

These rules describe the pinned FlashAttention CuTe implementation. **Evidence: observed source and upstream tests; no local GPU execution.** Source ID: `flashattention`.

## BWD-01 — Treat preprocessing and scratch initialization as part of backward

**Applies when:** adapting an attention backward mainloop or reusing accumulation workspace.

**Do:** trace saved forward state → preprocessing → accumulation → postprocessing. Specify who clears each accumulation buffer and semaphore before every logical invocation. Retain LSE unit conversion and any auxiliary-output gradient correction.

**Source / evidence:** [`interface.py` backward preparation](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L2293-L2333) creates zeroed deterministic semaphores and invokes preprocessing. [`flash_bwd_preprocess.py`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_bwd_preprocess.py#L393-L440) computes a rowwise `O*dO` reduction, subtracts `dLSE` when supplied, clears dQ accumulation and converts natural LSE to base-2 units. The dedicated head-dimension-256 path has different scratch needs, so this is a path-specific trace.

**Counterconditions:** deleting a zeroing launch can be valid only after proving every subsequently read element is overwritten or initialized elsewhere. A fast isolated backward mainloop may silently rely on setup excluded from its benchmark. Ignoring the gradient of an exposed LSE output changes the backward contract.

**Validate:** start scratch buffers with nonzero/NaN patterns, invoke backward repeatedly with different inputs and ragged lengths, and compare dQ/dK/dV independently. Include nonzero `dLSE` when supported, plus the exact scratch/postprocess time in full backward latency.

## BWD-02 — Determinism is an ordered accumulation protocol, not an accuracy toggle

**Applies when:** multiple CTAs contribute to dQ or multiple Q heads contribute to shared KV-head gradients.

**Do:** distinguish numerical closeness from repeatability. For deterministic operation, preserve the contribution ordering and semaphore release protocol alongside the arithmetic. Treat changes to tile shape, scheduler order and GQA mapping as potential changes to reproducibility.

**Source / evidence:** [`FlashAttentionBackwardSm100`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_bwd_sm100.py#L3652-L3704) computes lock order; its [`dQ accumulation loop`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_bwd_sm100.py#L3888-L3927) waits for the required semaphore value before bulk reduction and controls completion/release. [`interface.py`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L2293-L2303) allocates additional K/V semaphores for deterministic GQA.

**Counterconditions:** an atomic add prevents lost updates but does not by itself fix floating-point summation order. More accurate or deterministic results need not be faster. An ordering scheme that conflicts with available CTA scheduling can hang; do not invent a global spin ordering without a progress argument.

For sparse KV subtiles, a tail block may include contributors that are never
scheduled. Preserve both lock assignment and the non-unit release increment
described in [sparse backward](block-sparse-attention.md); the [release site](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_bwd_sm100.py#L3942-L3956)
uses `dq_sem_release_inc`, not an unconditional increment of one.

**Validate:** repeated identical inputs and upstream-style `torch.equal` checks for deterministic mode, separate closeness-to-reference tests, GQA/MQA and masking cases, and scheduler stress. [`test_flash_attn_race_condition.py`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/tests/cute/test_flash_attn_race_condition.py#L308-L345) repeatedly exercises gradients; choose a bounded local test subset before an expensive full sweep.

## BWD-03 — Forward feature support does not imply backward feature support

**Applies when:** enabling score modifications, custom masks, sparsity, determinism or a new architecture for training.

**Do:** inspect backward guards independently, including derivative callbacks and architecture-specific paths. Test the exact feature combination; provide a fallback or explicit rejection if the backward path is unsupported.

**Source / evidence:** [`interface.py` SM120 backward branch](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L1969-L1993) explicitly rejects block sparsity, score/mask modifications and deterministic backward at this snapshot. Its [`SM100 sparse branch`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L2024-L2042) requires a compatible 2CTA path for head dimension 192.

**Counterconditions:** a feature may work in inference and fail only when gradients are requested. Replacing an unsupported backward with a mathematically incomplete approximation is not a valid fallback.

**Validate:** forward and all required gradients, small high-precision/reference cases, ragged/empty cases and the full architecture-feature dispatch matrix. Report inference-only scope explicitly if backward remains untested or unsupported.
