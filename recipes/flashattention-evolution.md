# FlashAttention mechanisms and architecture-specific implementation limits

Tags: FlashAttention-3, FlashAttention-4, Hopper, Blackwell, SM103, exp2, SFU,
MUFU, FMA, conditional rescaling, tensor memory, scheduler, causal, varlen.
Evidence: author explanations plus source inspected at FA commit
`1bda8f9290cd48d030f1516f0e680cd464ef3554`. No local GPU measurement.

## What changes across generations

The [FA3 author explanation](https://tridao.me/blog/2024/flash3/) separates
copy/compute overlap from overlap between matrix multiplication and softmax.
Its Hopper design uses asynchronous execution to keep different execution units
busy, at the cost of more live state and register pressure. The useful lesson is
to schedule a dependency graph against actual execution resources; a low-FLOP
operation can still occupy a slow functional unit for a substantial time.

The [FA4 author explanation](https://tridao.me/blog/2026/flash4/) describes a
different balance on B200: forward softmax work and backward shared-memory
traffic matter as tensor-core throughput grows. Its mechanisms include
conditional rescaling, a mixture of native and software-emulated exponential,
TMEM-based intermediates, and different scheduling for uneven attention work.
These are reported mechanisms, not a guarantee that every current code path or
every Blackwell SKU uses all of them.

## Case 1: exponential emulation is architecture-specific

In [the SM100 forward tuning table](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_fwd_sm100.py#L78-L127),
the key includes `is_sm103`, causal mode, padded head dimension and CTA mode.
Several SM103 entries set `ex2_emu_freq=0`, selecting native exponential.
[SoftmaxSm100.apply_exp2_convert](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/softmax.py#L413-L454)
contains the native/emulated selection. An agent porting a B200 optimization to
B300 should inspect that branch rather than automatically add polynomial math.

**Candidate:** offload some exponential work only if the target's native path is
a demonstrated bottleneck and the alternative has spare execution capacity.
**Countercase:** added instructions and registers can lose on another architecture,
causal mode or head size. Test numerical approximation and total kernel latency;
an isolated exponential microbenchmark cannot validate the attention pipeline.

## Case 2: delayed rescaling has a precision bound

[SoftmaxSm100.update_row_max](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/softmax.py#L363-L390)
can retain the previous row maximum and use scale one when the increase is
within the configured threshold. This removes some rescaling work while keeping
the numerator and denominator in a consistent normalization frame. It does not
mean the probability tile remains bounded by one before the final normalization.

The [forward setup guard](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_fwd_sm100.py#L2188-L2208)
requires `max_offset + rescale_threshold < log2(dtype_max)` to avoid saturating
the probability tile when it is cast before `P @ V`. The inspected defaults use
threshold zero for 8-bit input paths and a positive threshold for 16-bit paths.
The FP32 denominator and lower-precision probability tile must still represent
compatible weights; saturation in only the numerator path can shrink outputs.

**Candidate:** tune rescaling only as part of the complete softmax-state contract.
**Countercase:** transferring a threshold to a narrower format can introduce
systematic error without NaNs. Validate large jumps in score maxima, small
probability tails, all-masked rows, output and LSE across multiple tiles.

## Case 3: an author-described feature may not be enabled in the API

The author blog describes longest-processing-time-first ordering for variable
lengths. At this pinned revision,
[FlashPrepareScheduler](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/prepare_scheduler.py#L35-L62)
rejects `sort=True`, and the public
[scheduler metadata preparation path](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L4176-L4204)
also guards it. This is a narrow statement about that path, not a claim that
all scheduling or causal swizzling is absent.

**Candidate:** investigate workload reordering only after identifying the selected
scheduler and preserving the virtual-to-actual batch mapping.
**Countercase:** preprocessing repeated for short calls can cost more than the
tail it saves. Do not count cached metadata as free if requests invalidate it.
Test empty sequences, skewed lengths and output restoration to original order.

## Conditions for transferring these source decisions

1. Read the matching architecture branch, dtype guard and API entry point.
2. State the mechanism and the resource it trades for the expected saving.
3. Keep the numerical state and stage-lifetime invariants explicit.
4. Test the hardest boundary before sweeping tuning constants.
5. Report the result only for the measured workload and effective dispatch path.

Related: [architecture selection](architecture-selection.md),
[async pipeline rules](../practices/async-pipelines.md), and
[low precision rules](../practices/low-precision.md).
