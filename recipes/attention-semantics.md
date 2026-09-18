# Attention interface contracts found in upstream implementations

Tags: attention, LSE, log2, natural log, split-KV, causal, unequal lengths, masked rows.
Evidence: pinned implementation and upstream issue/PR inspection. No local GPU validation of these attention paths.

## The optimization candidate

Splitting a long KV sequence creates more parallel work when batch/head parallelism
is insufficient. The partial output is not a complete interface: retain its softmax
normalizer too. Merge disjoint partitions using normalizer-derived weights, with
max subtraction for stability. The [FlashInfer recursive-attention tutorial](https://docs.flashinfer.ai/tutorials/recursive_attention.html)
describes the state algebra. Real floating-point merge orders need not be bitwise equal.

For natural-log normalizers `s_a, s_b`, let `m=max(s_a,s_b)` and
`w_a=exp(s_a-m), w_b=exp(s_b-m)`. Then
`o=(w_a*o_a+w_b*o_b)/(w_a+w_b)` and `s=m+log(w_a+w_b)`.
Partitions must cover the intended keys without double counting. An empty state
needs explicit handling: subtracting two negative infinities produces NaN.

## An interface trap an agent must check

A producer using base-2 LSE and a consumer using natural-log LSE disagree even if
both tensors have identical shapes and dtypes. Convert a true base-2 log partition
to natural log with `s_e = s_2 * ln(2)`, or use a merge implementation with matching
conventions. Do not convert blindly: inspect the exact producer, backend option,
and consumer at the installed revision. Logit scaling for `exp2` is a separate
part of this contract; changing only the final logarithm cannot fix incorrectly
scaled logits.

The pinned [FlashInfer merge kernel](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/cascade.cuh#L30-L74)
explicitly requires base-2 inputs and uses `exp2`/`log2`; it also guards the
normalizer when both states are empty. This concrete interface is more precise
than inferring the implementation's log base from mathematical notation.

The open [SGLang PR #35045](https://github.com/sgl-project/sglang/pull/35045)
reports this type of mismatch in a particular chunked-prefix MLA path. As inspected
on 2026-09-18, it was an open proposal, not proof that the fix was merged or that
all SGLang/FlashInfer paths are affected. Its reported GPU numbers are not used as
local validation here.

## Another trap: rectangular causal attention

For bottom-right causal alignment the allowed key index is
`j <= i + key_length - query_length`. For `Q=2, K=5` the first row can see four
keys, not one. For `Q=5, K=2` the first three rows are fully masked. The
[FlashAttention API README](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/README.md)
documents this alignment and zero output for fully masked rows in the KV-cache
API. Match the API you are replacing; not every library's `causal=True` means this.

## Transfer to a real kernel

1. Record each backend's mask alignment, LSE base, scaling, empty-row contract,
   and supported partition scheme before editing code.
2. Compare split and unsplit GPU paths over multiple split boundaries, uneven
   partitions, equal/unequal Q and K lengths, and fully masked rows when supported.
3. Check actual dtype error, all output elements, LSE, and gradients if required;
   then run memory/race/synchronization checks appropriate to the implementation.
4. Benchmark the complete split-plus-merge path, including workspace and planning
   costs at their actual reuse frequency. Extra parallelism must repay this cost.

Use a nonsplit fallback when the workload is too small to amortize merging or
the required semantics are not supported by the proposed split implementation.
