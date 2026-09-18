# Source audit — 2026-09-18

This audit covers the current [catalog](catalog.json): **26 source records, 156 entry links, 14 repositories at 15 pinned snapshots**. Of the entries, **137 are repository files and 19 are web/document links**. PyTorch has separate deployed-release and current-research records. The audit establishes provenance and navigation integrity, not complete-file review, successful builds or GPU correctness.

## What was checked

| Check | Result | Boundary |
| --- | --- | --- |
| Repository identity | GitHub repository metadata returned the exact catalog owner/name and canonical URL for all 14; none were archived when checked. | Ownership is authoritative provenance, not evidence that every implementation is best for a workload. |
| Git revision/path | Every repository entry matched a blob in that repository's recursive tree at its full catalog SHA; all 15 snapshot trees were available without truncation. | A path existing is not a correctness assertion. |
| Content access | All 137 repository entry bodies were fetched; relevant portions were inspected and purpose descriptions tied to those portions. | “content” does not mean every line of every large file was audited. No “whole files verified” claim. |
| Licensing | Pinned top-level license text inspected for all 14 repositories. | Individual/vendored files and datasets can carry separate terms. GitHub's detected license is not the licensing authority. |
| Local navigation | Local relative links and catalog schema checked during packaging. | These checks do not establish remote availability or source correctness. |

| Source ID | Canonical repository | SHA prefix | File entries | Top-level license |
| --- | --- | --- | --- | --- |
| `flashinfer` | `flashinfer-ai/flashinfer` | `0e4c173821a0` | 20 | Apache-2.0 |
| `flashattention` | `Dao-AILab/flash-attention` | `1bda8f9290cd` | 25 | BSD-3-Clause |
| `vllm` | `vllm-project/vllm` | `8da75d61904c` | 16 | Apache-2.0 |
| `sglang` | `sgl-project/sglang` | `2dee23a876a0` | 20 | Apache-2.0 |
| `cutlass` | `NVIDIA/cutlass` | `f614dc40e17f` | 13 | BSD-3-Clause |
| `triton` | `triton-lang/triton` | `92d654e2094d` | 9 | MIT |
| `flashinfer-bench` | `flashinfer-ai/flashinfer-bench` | `40e6ca7844b5` | 4 | Apache-2.0 |
| `deepgemm` | `deepseek-ai/DeepGEMM` | `78b69000794d` | 7 | MIT |
| `thunderkittens` | `HazyResearch/ThunderKittens` | `67845f5fa48d` | 2 | MIT |
| `tilelang` | `tile-ai/tilelang` | `efe9654bf4e1` | 2 | MIT |
| `quack` | `Dao-AILab/quack` | `35266c3298f0` | 4 | Apache-2.0 |
| `liger-kernel` | `linkedin/Liger-Kernel` | `19d92e730965` | 3 | BSD-2-Clause |
| `aiter` | `ROCm/aiter` | `1834cd14b33f` | 4 | MIT |
| `pytorch-inductor` | `pytorch/pytorch` | `8d6ffa599ea1` | 4 | BSD-3-Clause |
| `pytorch-rmsnorm-2.10` | `pytorch/pytorch` | `449b17684101` | 4 | BSD-3-Clause |

GitHub metadata returned `NOASSERTION` for CUTLASS, TileLang and PyTorch. The catalog uses inspected license text instead: CUTLASS BSD-3-Clause, TileLang MIT text with a historical collaboration notice, and PyTorch's top-level BSD-style license plus component notices. This is source metadata for reuse review, not a blanket legal conclusion.

The two DeepGEMM scale-dispatch/layout entries were added in a follow-up pass:
both paths were verified against a fresh untruncated tree at the existing SHA,
and their relevant host guards were read alongside `tests/test_layout.py`.


## Documentation identity and version handling

NVIDIA sources are hosted on `docs.nvidia.com`. Their URLs are rolling; observed CUDA guide/tuning editions and PTX edition are recorded separately from the access date. A rolling URL plus a version label is not an immutable snapshot. The CUDA Programming Guide currently uses `cuda/cuda-programming-guide/`; old `cuda-c-programming-guide/` links should not be silently substituted.

Tri Dao's author site and versioned FA3 arXiv reference provide design rationale. Colfax's tutorial explicitly links its own implemented Stream-K example, making it a primary implementation tutorial. Their measured results remain attached to their workloads and dates.

Previously discovered source drift is retained in the map: the vLLM paged-attention design document labels itself historical; SGLang replaced the legacy JIT namespace; FA2, FA3 and FA4 are distinct implementation paths; FlashInfer cascade headers live under `attention/`. FA4 scheduler metadata currently rejects `sort=True` in the inspected path despite the author's broader algorithm discussion.

## Negative evidence that changes automation

The [validation map](validation-map.md) records numerical checks that only print pass/fail, teaching kernels without reference comparison, simulated pipeline tests and mocked graph tests.

One further automation trap: NVIDIA's [Compute Sanitizer option table](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#command-line-options) gives `error-exitcode` a default of **0**. Set a nonzero value in an automated gate and retain the summary; a successful application exit alone need not mean the tool found no errors. The pinned DeepGEMM sanitizer launcher checks subprocess return codes but does not set that option. It also defaults to memcheck/synccheck, excludes selected kernels and forces blocking launches; those choices limit the claim. The vendor states blocking launches can reduce the number and precision of reported errors.

## Reproduce or refresh

1. Select a source from `catalog.json` and inspect the exact pinned paths.
2. For each repository, GET `https://api.github.com/repos/OWNER/REPO` to compare canonical identity, then GET `https://api.github.com/repos/OWNER/REPO/git/trees/FULL_SHA?recursive=1`; require `truncated=false` and exact entry paths of type `blob`.
3. Fetch `https://raw.githubusercontent.com/OWNER/REPO/FULL_SHA/PATH` for the selected entries and license; inspect the relevant implementation, caller and tests. A successful HTTP response alone cannot refresh the purpose text.
4. Reopen the relevant vendor/author document sections and record the effective edition/date. Preserve historical performance results and distinguish rolling docs from pinned code.
5. Update pins, entry purposes and dependent practice/recipe links together; check affected local links and catalog consistency. Review new hardware or compiler claims against the deployed version before using them.
