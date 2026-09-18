# PyTorch 2.10 RMSNorm as a native baseline

For an environment already running PyTorch 2.10.0, benchmark the public API alongside an eager oracle and candidate kernel:

```python
import torch.nn.functional as F
native = F.rms_norm(x, (x.shape[-1],), weight=w, eps=eps)
```

This is a **source-inspected baseline candidate**, not a measured speed claim. It closes the gap between “faster than an unfused eager expression” and “competitive with an installed fused operator.” A win against this baseline still does not establish state of the art.

Version examined: release tag `v2.10.0`, resolved through the GitHub tag API to **449b1768410104d3ed79d3bcfe4ba1d65c7f22c0**. This is separate from the newer PyTorch research snapshot in [catalog.json](catalog.json). Match the installed build's `torch.version.git_version` before treating the release source as exact binary provenance.

## Numerical contract

The [public functional wrapper](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/nn/functional.py#L2940) delegates to `torch.rms_norm`. In the CUDA vectorized implementation, [weight and input are promoted to accumulator type before multiplication](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L282), then the result is cast when written to the output. The [accumulator mapping](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/AccumulateType.h#L130) uses FP32 for BF16, FP16 and FP32 CUDA inputs.

Thus this path does **not** round normalized BF16 activations before multiplying BF16 weight. It has the same precision class as an oracle that promotes input and weight, normalizes/multiplies in FP32, and casts once at the end. It is not bitwise identical by construction: reduction order, reciprocal-square-root implementation and floating-point reassociation can differ.

The [unvectorized output kernel](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L100) also multiplies in accumulator precision before storage. Its moment computation reconstructs the second moment from variance plus squared mean, which is a different evaluation path from a direct sum of squares. Keep an independent high-precision oracle and actual error measurements.

## Routing and timing traps

| Condition | Source behavior | Comparison consequence |
| --- | --- | --- |
| FP32/FP16/BF16, aligned buffers, N divisible by4 and at most2^24 | [Vectorized dispatch](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L1102) | Typical aligned hidden sizes can use the fused kernel. Confirm actual launch path rather than assuming it from the API. |
| Unsupported vectorization condition | [Two-kernel moments plus output path](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L1115) | Odd hidden dimensions or misalignment can change launch count and numerical evaluation. |
| Input and weight dtypes differ | [Composite fallback with warning](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/layer_norm.cpp#L343) | FP32 weights with BF16 input are not the same dispatch as same-dtype tensors. Record the effective route. |
| `eps=None` | [CUDA default uses accumulator-type epsilon](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L1852) | Pass the task's epsilon explicitly; avoid comparing different normalization equations. |
| Noncontiguous input/weight or output allocation | [Wrapper makes contiguous views/copies and allocates output plus rstd](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L1849) | Match full operator timing or explicitly disclose a candidate's preallocation advantage. |

Compare ordinary API invocations under the same launch/cache regime first. If testing graph replay, capture the same external contract for both operators. If `torch.compile` is available and relevant, a compiled version of the same oracle is another distinct comparator; report compilation/setup separately and confirm the generated path.

Do not substitute an installed FlashInfer, SGLang or Quack version's semantics from this repository's latest research pins. Probe its version and selected operator, then inspect that exact source before interpreting parity or performance.
