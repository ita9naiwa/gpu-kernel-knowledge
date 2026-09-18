# Pair implementation with validation

Resolve the entry IDs below in [catalog.json](catalog.json) to find pinned links. These are inspected **upstream test designs**, not tests executed by this knowledge base. A filename containing “test” or a process exiting zero does not establish numerical correctness.

## Attention and serving

| Implementation / contract | Correctness or negative-case source | Benchmark / boundary |
| --- | --- | --- |
| FlashInfer paged decode, wrapper plan/run, stride | `flashinfer-11` decode matrix; `flashinfer-13` noncontiguous views | `flashinfer-16` times run after plan. Fixed heads/page/dimension; reported input-byte bandwidth is an estimate. |
| FlashInfer ragged prefill, mask ranges, state/LSE | `flashinfer-12` prefill; `flashinfer-14` targeted decode/prefill LSE comparison; `flashinfer-15` invalid masks, range boundaries, padded queries | Read `flashinfer-10` timing helper defaults. A decode sweep is not a prefill workload proxy. |
| FlashAttention FA2 / FA3 / FA4 | `flashattention-9` / `flashattention-10` / `flashattention-11`, respectively; `flashattention-12` repeated race stress | `flashattention-13` CuTe SM90 sweep reports forward maximum error without asserting tolerance; backward has no gradient oracle in that benchmark. |
| vLLM FlashInfer adapter and backend eligibility | `vllm-9` full/fast plan; `vllm-10` integrated attention; `vllm-11` capability rejection; `vllm-12` probe-error propagation | `vllm-8` harness and `vllm-13` decode YAML; head counts are after tensor-parallel sharding. Selection tests do not establish GPU numerical parity. |
| SGLang fused dispatch, attention integration, graphs | `sglang-6` GPU parity; `sglang-7` selection contracts; `sglang-8` FlashInfer cases plus `sglang-9` actual dense oracle | `sglang-10` measures CPU no-op wrapper overhead only. `sglang-11` mocks fence calls; it does not prove GPU race freedom. |

For feature coverage, inspect the actual parameter values, skip/dependency gates and selected backend. An entire test suite can appear successful while a GPU-specific case is skipped. Do not transfer Hopper coverage to Blackwell or one attention family to another.

## Layouts, pipelines and GEMM

| Mechanism | Direct validation source | Limit |
| --- | --- | --- |
| CUTLASS CuTe swizzle/slice address algebra | `cutlass-9` host layout tests | No runtime bank-conflict/performance measurement. |
| CUTLASS SM90 producer/consumer lifecycle | `cutlass-10`, `cutlass-11`, harness `cutlass-12` | Simulates TMA commits and GMMA consumers. Random iteration counts plus successful launches are synchronization stress, not a GEMM numerical oracle. |
| CUTLASS Hopper teaching GEMM | `cutlass-3` main contains timing and host output copy | No numerical reference comparison in this example. Architecture mismatch is waived with a success return. Add an independent oracle. |
| CUTLASS Blackwell teaching GEMM | `cutlass-4` main has host reference comparison; `cutlass-6` and `cutlass-7` have torch reference/assert_close | Preserve the teaching example's shape/type assumptions and extend tails before accepting a general kernel. |
| Triton/Gluon tutorials | `triton-1`–`triton-9` include checks in the linked files | See distinctions below; not every “validate” routine fails the process. |

Triton fused softmax checks an irregular width, layer norm checks forward and gradients, fused attention checks outputs and gradients, and Gluon async/TMA/warp-specialization examples use explicit assertions. Block-scaled matmul reconstructs the scaled reference. Match these to the exact datatype and hardware gates.

The ordinary matmul tutorial (`triton-2`) prints whether `torch.allclose` passed. Persistent matmul (`triton-5`) prints icons from `run_test`, uses `atol=1.0`, and does not assert failure there. Reusing those scripts in an automated acceptance gate requires converting the result into an explicit failure and selecting a justified numerical tolerance.

## Transfer procedure

1. Choose the test that exercises the **same public contract**, including return values, aliasing, mutation, masking and backward behavior.
2. Retain negative and boundary cases; add target workload shapes rather than replacing the upstream matrix with one representative shape.
3. Make every skip, unsupported route and numerical failure visible to the runner; never interpret a benchmark number as a correctness assertion.
4. Run correctness separately from timing, then preserve the benchmark's inclusion boundary: preparation, packing, planning, allocation, graph capture and replay.
5. Record the selected implementation and effective configuration in the result so a dispatch fallback cannot silently masquerade as validation of the candidate.
