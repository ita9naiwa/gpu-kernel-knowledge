# Low precision and numerical contracts

Keywords: FP16, BF16, FP8, FP4, quantization scale, accumulation, tensor core, exp2, error budget.

## PREC-01 — Separate storage, multiplication, accumulation and output precision

**Applies when:** tensor-core kernels, quantized weights/KV, or replacing a framework reference.

**Do:** distinguish dot-product inputs from accumulation and output: the cited Triton attention path casts probabilities before the dot, while GEMM accumulates in FP32 before its output cast. FP32 accumulation does not recover information lost in the input cast.

**Source / evidence:** **observed implementation**: Triton [`matrix multiplication tutorial`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/03-matrix-multiplication.py) accumulates into FP32 before converting outputs; [`attention tutorial`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/06-fused-attention.py) converts probabilities to the dot-product dtype. Source ID: `triton`. The precision table is a **derived recommendation**.

**Counterconditions:** accumulation mode, compiler flags and hardware instructions can change the actual behavior. Inspect the compiled path if precision is material to acceptance.

**Validate:** compare against both the higher-precision operator and the deployed baseline, including small outputs and cancellation. Use the metric definitions in [CHECK-01](correctness-and-benchmarking.md).

## PREC-02 — Treat quantization scales and layout as part of the tensor type

**Applies when:** FP8/FP4 or integer kernels, block-scaled GEMM, and quantized KV caches.

**Do:** preserve format, scale direction/granularity/layout, rounding, saturation and dequantization placement. FA3's FP8 result includes block quantization and additional numerical techniques; a dtype-only rewrite does not reproduce that method.

**Source / evidence:** **upstream research**: [FlashAttention-3](https://arxiv.org/abs/2407.08608) uses block quantization and additional numerical techniques for FP8 rather than a bare dtype conversion. Source ID: `flashattention`. The full contract checklist is a **derived recommendation**.

**Counterconditions:** a backend may support only one operand orientation or scaling mode. A lower-precision compute win can be lost to runtime quantization/repacking. Published FP8 accuracy results do not establish accuracy for FP4 or different distributions.

**Validate:** zero blocks, outliers, values near representable limits, scale broadcasting/layout, tail groups, and round-trip reconstruction. Measure both prequantized steady state and any quantization performed per invocation.

**Concrete source boundary — DeepGEMM K-grouped scales:** at the catalog pin,
the [transform dispatch](https://github.com/deepseek-ai/DeepGEMM/blob/78b69000794d0937b47ae3387eff7663410264d1/csrc/apis/layout.hpp#L89-L115)
accepts prepacked INT32 UE8M0 for `gran_k=32` on architecture major 10; the FP32
packing helper accepts `gran_k=32` or `128`. The generic claim that both are
"packed scales" does not make these API domains interchangeable.

The [layout tests](https://github.com/deepseek-ai/DeepGEMM/blob/78b69000794d0937b47ae3387eff7663410264d1/tests/test_layout.py#L81-L147)
require every group to start a fresh packed row and zero-fill its own final
row's unused byte slots. Psum metadata contains raw K, while the scale tensor
covers aligned K per group. Packing all groups as one uninterrupted byte stream
would violate that layout. Shape and total byte count alone cannot validate it.

The [host helper](https://github.com/deepseek-ai/DeepGEMM/blob/78b69000794d0937b47ae3387eff7663410264d1/csrc/jit_kernels/impls/smxx_layout.hpp#L218-L316)
also distinguishes synchronized CPU group lengths from device-only psum lengths.
Without CPU lengths it allocates an upper bound; the corresponding test compares
only the valid packed prefix. Do not interpret allocation capacity as initialized
logical output. Prepacked validation checks layout/capacity when CPU lengths are
available, but does not prove scale contents are correct. Preserve the producer's
packing and consumer's group metadata together. These are source-observed contracts,
not local FP8/FP4 correctness or performance measurements.

## PREC-03 — Keep exponential base and score scaling consistent

**Applies when:** replacing `exp` with `exp2`, using online softmax, or passing LSE across libraries.

**Do:** apply `log2(e)` consistently when converting natural-exponent logits to base-2 exponentiation. Record whether stored maxima/LSE are in the original or converted units. Keep the mask/soft-cap ordering from the operator contract.

**Source / evidence:** **observed implementation**: Triton [`attention tutorial`](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/06-fused-attention.py) uses `exp2` and converts the softmax scale; FlashInfer [`state_t::get_lse`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/include/flashinfer/attention/state.cuh#L45) returns base-2 state. Source IDs: `triton`, `flashinfer`.

**Counterconditions:** a wrapper may convert LSE on return; verify that public interface independently from internal math. Multiplying an already-converted scale twice silently changes attention temperature.

**Validate:** hand-computed two-logit cases and random high-dynamic-range scores; output plus LSE, not output alone; round-trip conversion between natural and base-2 units.

## PREC-04 — Set tolerances from the operation and baseline, not from one passing seed

**Applies when:** accepting approximate math, reordered reductions or low-precision execution.

**Do:** preserve the distinction in FlashAttention's tests between error against a reference and the PyTorch baseline's error. Reusing that relative comparison requires the same reference and scope; it does not supply a universal tolerance.

**Source / evidence:** **documented upstream testing policy**: FlashAttention [`Tests`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/README.md#L552-L556) compares numerical error against a PyTorch baseline across configurations. Source ID: `flashattention`. It is a useful policy example, not a universal factor-of-two acceptance rule.

**Counterconditions:** bitwise reproducibility, stochastic outputs, gradients and model-quality requirements need different acceptance criteria. A cosine score can hide scale errors.

**Validate:** adversarial values and multiple seeds across the dispatch matrix; maximum error and relative L2 with the exact definition; NaN/Inf counts; repeated runs and different batch composition when deterministic behavior matters.
