# Primary source map

Start from the workload, select one implementation family, then read its wrapper, kernel, test and benchmark together. The machine-readable index is [catalog.json](catalog.json). Checked on **2026-09-18**. Repository links are immutable commit snapshots; vendor documentation links roll forward. Nothing here claims locally measured GPU speedups.

| Need | Read first | Why this source |
| --- | --- | --- |
| Paged/ragged decode or prefill | `flashinfer` | KV metadata contracts, plan/run separation and split-KV scheduling coexist with the kernel. |
| Dense fused attention, training or backward | `flashattention` | FA2, Hopper FA3 and current CuTeDSL FA4 show different hardware tradeoffs; tests expose semantics. |
| Serving integration and backend eligibility | `vllm`, `sglang` | Actual callers establish supported shape/dtype/layout combinations and graph lifetimes. |
| GEMM, tensor-core pipelines or layouts | `cutlass`, `triton` | Progressive examples connect thread/data mapping to asynchronous pipeline implementation. |
| A performance or correctness diagnosis | `nsight-systems`, `nsight-compute`, `compute-sanitizer` | Separate application gaps, kernel bottlenecks and correctness defects. |

## Evidence policy

The [source audit](audit.md) records ownership, pinned-tree/path checks, license scope and reproducible refresh steps.

Use the [validation map](validation-map.md) to pair each core implementation with matching tests, negative cases and benchmarks. It also identifies scripts that print failures without failing the process and pipeline tests that do not verify numerical GEMM results.

`entries[].inspection = content` means the linked content was opened and relevant portions inspected. `tree` means only the exact path was checked in the pinned repository tree: use it as a navigation pointer, not as an audited implementation claim. `evidence_status` summarizes each source; it does not override entry-level inspection. Upstream tests or benchmark claims are not local measurements.

For normative synchronization and instruction questions, prefer `cuda-programming-guide` or `ptx-isa`. Use implementation comments to explain why a design was selected, not to establish universal hardware rules. For a speedup claim, preserve workload, hardware, precision, baseline and timing boundary.

## Read paths

For the installed PyTorch 2.10 comparison case, use the [version-specific native RMSNorm note](pytorch-2.10-rmsnorm.md). It verifies FP32 multiplication before the final cast and identifies dispatch/fallback timing traps.

**FlashInfer:** `docs/tutorials/kv_layout.rst` → `flashinfer/decode.py` or `prefill.py` → `include/flashinfer/attention/scheduler.cuh` → corresponding kernel. Read `recursive_attention.rst` for state merging. The caller owns allocation/page tables; a page layout is a semantic contract, not just a tuning detail.

**FlashAttention:** root README and public interface → implementation family → numerical tests. `flash_attn/cute/README.md` identifies the current CuTeDSL implementation as FlashAttention-4 for Hopper and Blackwell. `hopper/` remains a separate source family. Do not infer equivalent supported options from the shared project name.

**vLLM:** `docs/design/attention_backends.md` → selected `vllm/v1/attention/backends/` adapter → kernel/library call → attention benchmark workload. `docs/design/paged_attention.md` explicitly labels itself historical and says it no longer describes current code. It is useful for concepts, not current routing.

**SGLang:** `python/sglang/kernels/README.md` → public operator → selected implementation and parity test. At this pin the legacy `sglang.jit_kernel` namespace has been removed in favor of `sglang.kernels`. The simple registry selection and `BaseFusedOp` capability-driven selection are distinct mechanisms; read which one the caller uses.

**CUTLASS / Triton:** first read one minimal example that matches the target architecture. Then compare a pipelined or persistent version against it. Copying only the mainloop discards alignment, descriptor, launch and synchronization assumptions that make it correct.

## Refreshing this map

1. Resolve the **deployed version first**: installed package/build, vendored revision, target GPU and compiler. Use a newer official commit to investigate a candidate fix, not to describe an older deployment.
2. Check referenced paths and changed wrappers, capability gates and tests at the chosen SHA. Update the catalog pin and every dependent practice/recipe link atomically; do not mix a new wrapper with an old kernel silently.
3. Prefer versioned vendor docs matching the toolchain. If only a rolling URL is available, preserve its observed edition and check date, then re-open the relevant section before acting. A displayed version is not an immutable URL.
4. For documentation/code contradictions, record both and trace the executable caller and tests. Distinguish an outdated tutorial from a real implementation defect; do not silently generalize either. Promote `tree` to `content` only after inspecting relevant content, and never promote source inspection to runtime validation.
5. Keep old performance results attached to their original versions; a new source snapshot does not refresh a measurement. After edits, check affected local links and confirm that catalog entries still match their recorded repository, revision and path.

No third-party summaries, stars, or benchmark rankings determine authority here. Primary author tutorials and papers can complement code, but executable contracts come from the exact selected implementation.

## Complementary implementations

These are **distinct mechanisms to study**, not a ranked list of experimentally proven winners. Start with the core source matching the production path; use these when that mechanism matters.

| Source ID | Distinct reason to open it | Boundary |
| --- | --- | --- |
| `deepgemm` | Scale layout, grouped MoE and low-precision descriptor construction. | Include conversion/preshuffle cost; SM90 and SM100 contracts differ. |
| `thunderkittens`, `colfax-gemm` | Incremental teaching kernels and explicit producer/consumer scheduling. | Educational shape assumptions and historical performance figures. |
| `quack`, `liger-kernel` | Reduction register pressure, reload decisions and memory-bounded training fusion. | Check backward, AMP, loss reduction and architecture-specific thresholds. |
| `tilelang`, `pytorch-inductor` | Compiler-generated kernel inspection and controlled autotuning. | Generated code and compiler version are evidence; source DSL brevity is not a speed guarantee. |
| `aiter` | AMD-specific dispatch and blocked versus row-parallel reduction. | Portability reference only; no NVIDIA hardware-rule transfer. |

`flashinfer-bench` complements all of them with explicit operator definitions, concrete workload provenance and per-definition evaluation criteria. `flashattention3-author` and `flashattention4-author` explain resource bottlenecks and pipeline choices; their published measurements retain the paper's workload boundary.
