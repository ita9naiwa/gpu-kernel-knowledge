# Attention and KV-cache contracts

Keywords: prefill, decode, GQA, MQA, split-KV, cascade, logsumexp, LSE, causal alignment, paged attention.

Evidence labels: **observed implementation** means source was read at the linked commit; **documented contract** means upstream documentation states the behavior; **derived recommendation** means our proposed application, not a measured speedup. No rule below has been GPU-validated in this knowledge base.

## ATT-01 — Treat attention state as a typed numerical interface

**Applies when:** combining partial attention results, split-KV, shared-prefix cascade, distributed attention, or swapping producer and merge backends.

**Do:** record whether a state holds normalized output or an unnormalized weighted sum, its log base, scale placement, dtype, and empty-state convention. For normalized outputs `oA,oB` and natural-log normalizers `sA,sB`, use `m=max(sA,sB)`, `a=exp(sA-m)`, `b=exp(sB-m)`, then `(a*oA+b*oB)/(a+b)`. A base-2 normalizer requires `exp2`; convert natural LSE with `s2=s_e/log(2)` before a base-2 consumer. Never infer the log base from the variable name `lse`.

**Source / evidence:** **observed implementation**: FlashInfer [`MergeStateKernel`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/cascade.cuh#L34-L82) explicitly consumes base-2 LSE and uses `ptx_exp2`/`ptx_log2`; [`state_t`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/state.cuh#L30-L84) tracks an unnormalized accumulator until `normalize()`. Source ID: `flashinfer`.

**Counterconditions:** an output-only API may not expose enough state to merge exactly. Overlapping KV partitions double-count tokens unless the overlap is deliberately corrected. Floating-point merges are not bitwise associative.

**Empty-state boundary:** the formula above assumes at least one nonempty state.
At this pin, [`math::inf`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/math.cuh#L34-L37)
is IEEE infinity, not the historical finite masked-logit sentinel. With two
empty states, subtracting their `-inf` normalizers produces NaN. The merge body
clamps those exponent arguments and guards zero total weight; `state_t.normalize()`
also guards zero `d`. Preserve these operations and a finite zero output for
empty producers. Zero merge weights do not sanitize NaN/Inf output payloads.

**Validate:** compare small representative/adversarial merged cases with one-shot FP64 attention, then compare large cases with a trusted deployed baseline without materializing a huge attention matrix. Include unequal partition lengths, very different maxima, one empty partition, two empty partitions, reversed merge order, and explicit natural/base-2 conversions. Test output and LSE independently: output can look plausible while the normalizer is wrong.

## ATT-02 — Make unequal-length causal alignment explicit

**Applies when:** `seqlen_q != seqlen_k`, KV-cache decoding, chunked prefill, cross attention, or replacing one attention API with another.

**Do:** write the allowed `(query_position,key_position)` relation before coding. FlashAttention's documented bottom-right convention keeps `j <= i + seqlen_k - seqlen_q`; it differs from an unshifted triangular mask. Keep masking, positional encodings, sliding-window bounds, and cache offsets in the same position coordinate system.

**Source / evidence:** **documented contract**: FlashAttention [`README` causal examples and v2.1 change](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/README.md#L325-L335). All-masked rows produce zero output there. Source ID: `flashattention`. The coordinate-system rule is a **derived recommendation**.

**Counterconditions:** another API or model can intentionally use top-left or explicit absolute positions. Do not change its semantics merely to match FlashAttention.

**Validate:** materialize small masks for `(Q,K)=(2,5),(5,2),(1,5)` and compare row by row; test all-masked rows and chunk boundaries. Square self-attention alone cannot expose this bug.

## ATT-03 — Split KV only when the extra parallelism pays for the merge

**Applies when:** long KV contexts with too few independent query/head blocks to fill the GPU, especially decode and small batches.

**Do:** compare an unsplit path against a small split-count sweep, including temporary-buffer traffic and the merge launch. Preserve split policy in reproducibility records. Treat split count as a shape/workload decision, not a universal decode optimization.

**Source / evidence:** **documented mechanism**: FlashInfer [recursive attention](https://docs.flashinfer.ai/tutorials/recursive_attention.html) partitions KV work and merges attention states; **observed implementation / contract**: [`decode.py` fixed split options](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py#L1497-L1512). `fixed_split_size` is measured in **pages**; the documented deterministic-reduction contract applies to FA2 tensor-core decode and is not a general CUDA Graph compatibility guarantee. The same pin's [`cute-dsl` branch](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py#L1814-L1828) also consumes a positive value, deriving `kv_splits` from the maximum KV length and `fixed_split_size * page_size`; this does not establish the FA2 determinism contract for that backend. Source ID: `flashinfer`.

**Counterconditions:** large batch/head parallelism, short contexts, or merge traffic can erase the gain. Fixed splits can improve reduction-order invariance but do not promise bitwise agreement with an unsplit kernel.

**Graph/branch boundary:** the shared [`plan` guards](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py#L1627-L1649)
freeze batch size and `q_len_per_req` and enforce preallocated index capacity in
graph mode. Passing those checks does not establish compatibility with changing
split counts. The [`cute-dsl` call](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py#L1840-L1859)
passes `disable_split_kv` as `reduction="none"` versus `"auto"`; inspect that
backend's selected plan rather than transferring an FA2 planner interpretation.

**Validate:** vary batch, Q-head/KV-head ratio, context length and raggedness; report total operator time plus split and merge components. Change batch composition while keeping one request fixed to test batch invariance. Include graph replay with changing KV lengths separately.

## ATT-04 — Cache layout is an end-to-end contract, not a reshape

**Applies when:** paged KV, prefix sharing, quantized KV, GQA/MQA, or backend integration.

**Do:** trace the cache writer, page allocator, metadata builder, and attention reader. Record physical layout, strides, page size, head mapping, last-page validity, index dtype and capacity. Use the backend's own layout terms precisely: FlashInfer `NHD` and `HND` change the meaning of axes. Avoid repacking the entire cache on every decode step to satisfy a kernel microbenchmark.

**Source / evidence:** **documented contract / observed wrapper checks**: FlashInfer [`BatchDecodeWithPagedKVCacheWrapper`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py) exposes page metadata and validates cached Q/KV dtypes and layout. Source ID: `flashinfer`. The four-component trace and repacking warning are **derived recommendations**.

**Counterconditions:** a one-time layout conversion can be worthwhile for a long-lived cache or many subsequent operations; measure its amortization. A historical PagedAttention diagram is not proof of a current serving engine's selected backend.

**Validate:** use nonconsecutive page indices, mixed sequence lengths, partial final pages, shared-prefix pages, multiple KV heads and GQA ratios. Compare against a contiguous reference assembled from the same logical tokens; include cache append followed by attention, not only prefilled synthetic storage.

## ATT-05 — Distinguish exact attention implementation changes from model changes

**Applies when:** sparse masks, token pruning, reduced context, low precision, soft caps, or approximate attention are proposed as optimizations.

**Do:** classify the candidate as same mathematical operator with changed reduction order, changed precision, or changed attended token set. Keep accuracy criteria appropriate to that category. Name masks and soft-cap order explicitly; do not assume mathematically similar rewrites commute.

**Source / evidence:** **documented scope**: [FlashAttention paper](https://arxiv.org/abs/2205.14135) develops IO-aware exact attention; [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608) separately evaluates low-precision techniques. Source ID: `flashattention`. The classification is a **derived recommendation**.

**Counterconditions:** model-level approximation may be an acceptable task requirement, but cannot pass under a kernel-equivalence claim.

**Validate:** exact paths against a higher-precision operator reference; quantized paths against both that reference and an agreed model metric; sparse/token-reduced paths against the full task quality evaluation. State which validation has not been performed.
