# GPU Kernel Knowledge

A reference skill for **Codex and Claude Code**: 55 kernel engineering rules
across 13 topics, grounded in 26 primary sources and 156 file/document references.
Sources include FlashAttention, FlashInfer, vLLM, SGLang, CUTLASS, Triton,
DeepGEMM, and NVIDIA documentation.

The useful unit is a technique **with its applicability, counterconditions,
exact source, and validation boundary**. Public upstream postmortems preserve
failed approaches as well as successful mechanisms. This is a documentation
skill; it needs no server, API key, package installation, or GPU to read.

## Install

The repository root is a complete skill folder. Choose the client you use.

**Codex** — install in its personal skill directory:

```sh
mkdir -p "$HOME/.agents/skills"
git clone https://github.com/ita9naiwa/gpu-kernel-knowledge.git "$HOME/.agents/skills/gpu-kernel-knowledge"
```

Then invoke `$gpu-kernel-knowledge` with your kernel task.

**Claude Code** — install in its personal skill directory:

```sh
mkdir -p "$HOME/.claude/skills"
git clone https://github.com/ita9naiwa/gpu-kernel-knowledge.git "$HOME/.claude/skills/gpu-kernel-knowledge"
```

Then invoke `/gpu-kernel-knowledge` with your kernel task.

These use the documented [Codex skill locations](https://learn.chatgpt.com/docs/build-skills)
and [Claude Code personal skills](https://code.claude.com/docs/en/skills).
They install locally, not into hosted/cloud sessions. If an existing installation
occupies the destination, update that checkout with `git pull --ff-only` instead
of cloning over it. Restart the client if it has not discovered the skill.

## Upstream submodules

Seven repositories are linked at the same commits cited by the source catalog.
A normal clone downloads the reference skill only. From this repository's root,
fetch one source when you need it:

```sh
git submodule update --init --depth 1 -- upstream/quack
```

| Submodule path | Repository | Read for |
| --- | --- | --- |
| `upstream/flashattention` | [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Attention implementation families, backward, debugging postmortems |
| `upstream/flashinfer` | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Inference kernels, paged KV, split/merge and sampling |
| `upstream/cutlass` | [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass) | Layouts, tensor-core instructions and asynchronous pipelines |
| `upstream/triton` | [Triton](https://github.com/triton-lang/triton) | Kernel patterns, compiler, autotuning and tests |
| `upstream/vllm` | [vLLM](https://github.com/vllm-project/vllm) | Serving callers, backend eligibility, graphs and MoE |
| `upstream/sglang` | [SGLang](https://github.com/sgl-project/sglang) | Serving integration and conditional diffusion fusions |
| `upstream/quack` | [Quack](https://github.com/Dao-AILab/quack) | CuTe DSL reductions, normalization, GEMM and matching tests |

To fetch all seven, run `git submodule update --init --depth 1`.
Their own nested submodules are not needed for source reading. Building/running
an upstream project may require its additional dependencies and instructions.
This uses ordinary [Git submodules](https://git-scm.com/docs/git-submodule):
`--depth 1` reduces history, not the checked-out file set. No sparse-checkout
wrapper is bundled. Avoid `--remote` when reproducing the catalog snapshot;
update the gitlink, catalog revision and dependent citations together.

You can also attach this knowledge base to another project:

```sh
git submodule add https://github.com/ita9naiwa/gpu-kernel-knowledge.git docs/gpu-kernel-knowledge
```

Point your coding agent at that checkout's `SKILL.md`, or use the personal skill
installation above for automatic discovery. Adding a documentation submodule
alone does not register a client skill.

## Browse

| Need | Start here |
| --- | --- |
| Agent entrypoint | [SKILL.md](SKILL.md) |
| Choose a technique and understand its limits | [Practice map](practices/README.md) |
| Find an authoritative implementation | [Source map](sources/README.md), [machine-readable catalog](sources/catalog.json) |
| Find matching tests and benchmark boundaries | [Validation map](sources/validation-map.md) |
| Learn from real public failures | [Upstream postmortems](recipes/failure-driven-debugging.md) |

## Evidence and scope

Research snapshot: **2026-09-18**. Repository references pin commits; vendor
documentation can change. Recheck the deployed package, architecture, and actual
caller before transferring a mechanism. Source inspection does not establish
GPU correctness, speed, or an improvement in coding-agent output quality.

This public edition contains public-source research and original synthesis.
It excludes private experiment archives, internal source code, raw agent
transcripts, and machine-specific records. Public source inspection history is
recorded in the [source audit](sources/audit.md) and
[practice audit](practices/audit.md).

Original material in this repository is MIT licensed. Referenced upstream code,
papers, and documentation retain their own licenses; a link does not relicense
them. The catalog records repository-level license information and its limits.
