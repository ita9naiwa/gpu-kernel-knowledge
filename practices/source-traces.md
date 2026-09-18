# Source traces: four production codebases

These traces show how to extract a reusable rule from a real dispatch chain. They describe the pinned snapshots, not every release or the user's selected runtime. **Evidence: source inspection only; no local GPU execution.** Follow the linked symbols and surrounding guards before adapting them.

## TRACE-FA — FlashAttention: GQA packing changes the scheduling problem

**Applies when:** adapting FlashAttention tiles, split-KV heuristics or persistent variable-length scheduling.

| Stage | Observed behavior | Source |
|---|---|---|
| Configuration | `_get_fwd_config` chooses architecture-specific tile defaults. | [`interface.py:305`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L305) |
| Effective query shape | `max_seqlen_q` is multiplied by the Q/KV head ratio when `pack_gqa` is enabled. On the selected SM100/SM110 branch, this effective length determines whether `q_stage` is one or two. | [`interface.py:356`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L356) |
| Split policy | Effective M-block count, loaded KV span, SM count and a split bound feed `num_splits_heuristic`; the SM120 branch explicitly requires one split. | [`interface.py:362`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L362) |
| Runtime scheduler state | Optional metadata provides per-request block counts/splits and a tile-count semaphore; presence of that semaphore changes scheduling selection. | [`interface.py:1055`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L1055) |
| Completion and reuse | Split results are combined, then a reused tile-count semaphore is reset for the next invocation. | [`interface.py:1602`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L1602) |

**Do:** tune scheduling on the effective packed shape, include metadata preparation/reset in the timed boundary where it occurs, and preserve the target architecture's legality guards. Scheduler metadata is mutable execution state, not only cached shape information. Source ID: `flashattention`.

**Counterconditions / bad transfer:** copying a split count from an unpacked MHA benchmark into packed GQA can oversplit a workload that already has ample parallel work. Copying an SM100 split path into SM120 bypasses an explicit source restriction. Reusing a semaphore without reset can change the next call's work distribution. These are risks inferred from the observed chain, not reproduced failures.

The same [`_get_fwd_config` body](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L390-L401)
can override a split choice on SM100/SM110 when Q/K and V head dimensions differ:
it either shrinks N tiles and recomputes splits or forces one split to respect
shared-memory needs. Record the resolved configuration, not only the requested
split count.

**Validate:** packed/unpacked GQA parity, different head ratios, one/two-stage boundary, local windows with an explicit zero right bound, repeated metadata reuse, empty-Q inputs, and each supported architecture independently. Trace the actual selected kernel and combine launch. The source's heuristic comment says it needs revisiting for persistent split-KV: do not promote its numeric constants into permanent rules.

## TRACE-FI — FlashInfer: a fused sampling API may select a multi-kernel path

**Applies when:** optimizing top-k/top-p sampling or claiming one public call equals one GPU launch.

| Stage | Observed behavior | Source |
|---|---|---|
| Public semantics | `filter_apply_order` chooses sequential `top_k_first` or joint filtering; explicit tensor RNG state is documented for graph-compatible use. | [`sampling.py:1421`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L1421) |
| Fast-path gate | Compact top-k requires no row `indices`, a scalar integer K within its bound, a sufficiently large vocabulary, and K smaller than vocabulary size. | [`sampling.py:1345`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L1345) |
| Compact execution | `_top_k_first_fast_path` selects sorted top-k values/indices, normalizes the retained values, samples locally, then maps local IDs back to vocabulary IDs. | [`sampling.py:1359`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L1359) |
| Other branches | Sequential fallback masks logits, applies softmax, then top-p sampling. Joint filtering takes a separate module path after softmax. | [`sampling.py:1534`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L1534) |
| Validation | Upstream tests include sample frequency/support, logits/probabilities alignment, per-request K, RNG reproducibility and NaN validity behavior. | [`test_sampling.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/tests/utils/test_sampling.py) |

**Do:** benchmark the dispatch branch used by the caller, preserve filter order, and test the returned token IDs after index remapping. The docstring's broad single-kernel description does not describe every branch in this snapshot. Source ID: `flashinfer`.

**Counterconditions / bad transfer:** replacing sequential filtering by a joint mask changes the probability distribution. Distribution-equivalent implementations need not return the same token for every seed if RNG consumption/order differs. The upstream fast-path comment mentions an empirical TV result; it is not a general correctness threshold or a measurement made here.

**Validate:** tied logits at the K boundary, per-row thresholds, K near fast-path bounds, repeated row indices, zero-probability support, invalid distributions and graph replays with updated seed/offset buffers. Check deterministic repeatability and empirical distribution separately; time selection, normalization, sampling and remapping together.

## TRACE-VLLM — vLLM: MoE padding and expert ownership control valid output

**Applies when:** adapting grouped expert GEMM, low-batch MoE paths or expert-parallel reduction.

| Stage | Observed behavior | Source |
|---|---|---|
| Input contract | `fused_experts_impl` checks hidden/weight dimensions, matching top-k shapes, contiguity/last-dimension strides and activation dtype. | [`fused_moe.py:1652`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L1652) |
| Assignment selection | `_prepare_expert_assignment` bypasses token alignment for a narrow sparse low-token regime, unless expert mapping or particular quantization grouping prevents it. | [`fused_moe.py:1541`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L1541) |
| Padded representation | `moe_align_block_size` groups flattened token/expert assignments, pads per expert, and supplies expert IDs plus a device-side actual padded count. | [`moe_align_block_size.py`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/moe_align_block_size.py) |
| Kernel guards | `fused_moe_kernel` skips blocks beyond actual padded count, masks sentinel token IDs, promotes offsets before stride multiplication, and zeroes output for an expert ID of `-1`. | [`fused_moe.py:402`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L402) |
| Final state | The functional chain applies expert GEMM → activation → second GEMM → expert sum. It zero-initializes the final intermediate for expert-mapped paths and applies routing weights on the configured side. | [`fused_moe.py:1774`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L1774) |

**Do:** preserve the distinction between buffer capacity, actual padded work and real token count. Specify who writes zeros for nonlocal/invalid routes and who applies routing weights. Source ID: `vllm`.

**Counterconditions / bad transfer:** replacing the `-1` expert path with a return can expose old/uninitialized values if final reduction reads those slots. Moving the routing-weight multiply across a nonlinear activation changes the function. A kernel tuned for evenly populated experts can waste most of its tile on sparse decode. These are derived risks; current call sites may use different modular backends.

**Validate:** all tokens to one expert, uniform and skewed routing, empty experts, nonlocal experts, one-token decode, many experts with small K, partial tiles, and large stride products. Start with upstream [`test_moe_align_block_size.py`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/tests/kernels/moe/test_moe_align_block_size.py) and [`test_moe.py`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/tests/kernels/moe/test_moe.py), then test the selected serving integration.

## TRACE-SGLANG — SGLang: RMSNorm fusion preserves two rounding contracts

**Applies when:** fusing residual addition/normalization in LLM or diffusion-transformer blocks.

| Stage | Observed behavior | Source |
|---|---|---|
| Model-level dispatch | `RMSNorm.forward_cuda` guards empty input, variance override, batch invariance and quantization fusion before choosing the residual/HF-cast branches. | [`layernorm.py:491`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/srt/layers/layernorm.py#L491) |
| JIT eligibility | The HF-style fused residual path checks input/weight dtype, post-residual dtype and supported hidden size before invoking the JIT op; otherwise it uses the native path. | [`layernorm.py:568`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/srt/layers/layernorm.py#L568) |
| Specialization | `norm.py` includes `cast_x_before_out_mul` in the JIT C++ arguments; it is a semantic specialization. | [`norm.py:73`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/ops/layernorm/norm.py#L73) |
| Numerical implementation | The fused CUDA kernel accumulates squared FP32 residual sums. Its HF-style branch retains those unrounded sums for normalization and rounds normalized values before the weight multiply; the standard branch consumes the dtype-rounded residual values. | [`fused_add_rmsnorm.cuh:44`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/python/sglang/kernels/jit/csrc/elementwise/fused_add_rmsnorm.cuh#L44) |
| Mutation/reference | Both input and residual are updated. Upstream tests compare both tensors and select a separate HF reference when the cast flag is enabled. | [`test_fused_add_rmsnorm.py:38`](https://github.com/sgl-project/sglang/blob/2dee23a876a02d3221b252c7978453ed08bed147/test/registered/kernels/ops/layernorm/test_fused_add_rmsnorm.py#L38) |

**Do:** treat cast placement, residual precision and the returned residual as part of equivalence. Read guards and kernel host validation together: model-level supported shapes can be narrower than a low-level kernel's theoretical capacity. Source ID: `sglang`.

**Counterconditions / bad transfer:** deleting the FP32 cached sum in the HF-style branch to save registers changes its rounding behavior. The actual standalone RMSNorm support predicate is more restrictive above 8192 than its broad error message suggests; use executable guards rather than prose to select legal shapes. The model-level code explicitly notes a changed association when a third residual term is folded into the existing two-input kernel.

**Validate:** both output tensors, BF16/FP16 supported paths, cast flag on/off, extra residual term, higher-rank reshape path, small and large hidden sizes, and fallback dispatch. Upstream test tolerances are examples for their distributions; establish tolerances for the target model and do not copy the comments' approximate ULP arithmetic as a numerical proof.
