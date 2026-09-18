# Failure evidence from upstream postmortems

Tags: failure, regression, numerical mismatch, NaN, varlen, padded offset,
sanitizer, codegen, compile cache, hypothesis, negative result.
Evidence: public upstream author postmortems.
An upstream narrative is not a locally reproduced failure.

## Case: valid addresses can still be the wrong addresses

The pinned [FlashAttention varlen preprocess postmortem](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/AI/VARLEN_PREPROCESS_TILE_BUG.md)
reports a mismatch between the tile size used by preprocessing and backward.
Both computed offsets inside valid scratch storage, but they disagreed about
where each sequence's padded region began. A memory checker could therefore
miss the semantic error. The author reports that globally zeroed allocation
masked the issue, while reused memory exposed NaNs.

The transferable rule is to trace allocation, initialization, producer and
consumer with the same layout parameters. A scratch-buffer initialization pass
is part of the algorithm, not interchangeable boilerplate. Test multiple
sequences and nondefault tile sizes; a first-batch-only test misses the offset
term involving batch index. Repeated calls under allocator reuse add a different
test dimension from repeating the same input on freshly zeroed storage.

Do not infer that `torch.empty` is zero-initialized or that every NaN is caused
by this defect. This source describes one failure mechanism with its own
historical circumstances.

## Case: a successful perturbation is not an established root cause

FlashAttention's [debugging methodology note](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/AI/DEBUG_METHODOLOGY.md)
distinguishes observed recovery from a verified causal explanation. Changes to
instrumentation or source can perturb scheduling and generated code. For an
intermittent failure, compare fixed and original builds repeatedly under the
same toolchain, inspect the generated code, and test a prediction that can
disprove the proposed cause. A debug print making a hang disappear establishes
sensitivity, not which synchronization edge was wrong.

Record compile-cache identity or use an isolated experiment cache to ensure the
binary matches the source. Preserve other users' work and shared caches; the
useful invariant is source/binary correspondence, not indiscriminate deletion.
The upstream note's exact repetition count and workflow are project policy,
not a statistical proof or a mandatory global threshold.

## Case: fewer mask instructions did not improve causal-attention latency

The pinned [SM90 R2P masking investigation](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/AI/SM90_R2P_MASKING_SASS.md)
reports replacing repeated predicate generation with register-to-predicate mask
conversion. Its causal-attention timing rows show no useful gain: head dimension
64 at sequence length 8192 was 2.463 → 2.473 ms, and dimension 128 was
1.937 → 1.944 ms. Local-window rows improved 0.394 → 0.346 ms and
0.237 → 0.222 ms respectively. These are upstream-reported measurements;
the note supplies neither raw repetitions nor a complete timing protocol.

The static SASS-count comparison uses a **different configuration**: head
dimension 128, sequence length 113 and tile N=128. It reports 3% fewer total
instructions for causal masking and 15% fewer for local masking. Do not pair
those percentages directly with the long-sequence timing rows. Also, each
`R2P` with mask `0x7f` covers seven predicates: the note's detailed breakdown
accounts for four byte-high bits separately, correcting its introductory
shorthand that four R2Ps cover all 32 elements by themselves.

The failed generalization is that reducing mask instruction count must improve
the complete attention kernel. The author's explanation is that WGMMA dominates
causal attention while local windows expose more masking cost; the retained
note provides no profiler attribution sufficient to establish that cause
independently. This is not a reported compiler correctness defect, and the note
does not document a correctness failure or subsequent bug fix. Preserve the
workload-dependent benefit rather than rejecting the technique globally.

## A negative result needs a denominator

| Outcome | Preserve | Reusable conclusion |
| --- | --- | --- |
| Wrong answer | First failing case, output/error, contract, diff and reference | Which semantic invariant the change violated |
| Correct but slower | Matched samples, scope, workload, resource evidence | Where the mechanism's extra cost outweighed its benefit |
| No distinguishable gain | Raw spread and repeated comparisons | Evidence is insufficient to promote the change |
| Illegal configuration | Compiler/launch error and exact target/configuration | A legality boundary, not a speed result |
| Environment blocked | Allocation/toolchain/connectivity failure | Experiment was not evaluated; no kernel conclusion |

Avoid recording only “didn't work.” Conversely, one losing configuration does
not disprove a technique across all shapes or GPUs. Preserve the proposed
mechanism separately from what the evidence actually ruled out.

## The smallest useful attempt record

```text
Hypothesis and source/practice:
Baseline and candidate identity:
Workload and exact command:
Correctness result and first counterexample:
Timing samples and scope, if correctness passed:
Decision and why:
Reusable boundary; what remains untested:
```

Keep the original harness stable across candidates. A corrected harness starts
a new comparison; do not combine its measurements with the old ones as if they
shared a baseline. Promote only the narrow lesson supported by the evidence,
and link the raw artifact so another agent can check the conclusion.
