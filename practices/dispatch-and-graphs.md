# Dispatch, planning and CUDA Graphs

Keywords: specialization, JIT, autotune, graph capture, replay, static buffers, dynamic metadata, workspace, fallback.

## GRAPH-01 — Separate preparation from the repeatedly executed device path

**Applies when:** JIT compilation, workspace planning, metadata conversion or autotuning precedes repeated kernel execution.

**Do:** establish explicit prepare/plan and run phases. Warm up the selected specialization before capture; allocate persistent workspace outside the captured region; report setup cost separately from steady-state latency. Replan whenever the API's plan-dependent inputs change.

**Source / evidence:** **observed implementation / documented contract**: FlashInfer [`decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py#L528-L540) excludes planning from its custom op; its `plan()` documentation explicitly disallows CUDA Graph capture and `torch.compile` for that method. Source ID: `flashinfer`.

**Counterconditions:** setup amortization can be poor for one-off shapes; a faster run path may lose overall if each request triggers a new expensive plan. Not every library's method named `plan` has identical restrictions.

**Validate:** first call, warm calls, shape/metadata changes and repeated layer reuse. Confirm no hidden compilation, allocation or CPU readback enters the timed/captured run path.

## GRAPH-02 — Distinguish stable storage from mutable contents

**Applies when:** replaying a captured kernel while sequence lengths, page tables, positions or input values change.

**Do:** list captured addresses, launch dimensions and scalar parameters separately from device-buffer contents. Update mutable contents in the same persistent allocations using the prescribed ordering. Keep inputs, outputs and workspace alive until pending launches and captured graphs are retired.

**Source / evidence:** **observed wrapper contract**: FlashInfer [`graph-mode checks in decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py#L1637-L1655) enforces fixed batch size and bounded index-buffer capacity. Its prepared SM110 interface explicitly requires retained storage and separate workspace for concurrent calls. Source ID: `flashinfer`. The address/content checklist is a **derived recommendation**.

**Counterconditions:** a CUDA Graph does not automatically become dynamically shaped because a compiler supports dynamic shapes. Some APIs use bounded maximum launches with device-side lengths; others capture concrete geometry or immutable scalar lengths.

**Validate:** replay after changing values, page indices and lengths within the supported bounds; compare each replay against eager execution. Exercise two independent contexts or ordered streams to detect workspace reuse hazards.

## GRAPH-03 — Include all validity predicates in specialization and fallback

**Applies when:** dispatching by GPU architecture, dtype, shape, layout, mask, head grouping or graph mode.

**Do:** distinguish a kernel's legal domain from its profitable domain. Guard legality first, select a tuned path second, and retain a known-correct fallback. Record the effective backend and configuration, not just the requested backend name.

**Source / evidence:** **observed implementation**: Triton [`fused attention` autotune/pruning](https://github.com/triton-lang/triton/blob/92d654e2094db0d390db1f4e2992aa47f48e2556/python/tutorials/06-fused-attention.py#L151-L190) removes invalid block configurations; FlashInfer [`decode.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/decode.py) has backend-specific feature checks. Source IDs: `triton`, `flashinfer`. The legal/profitable split is a **derived recommendation**.

**Counterconditions:** architecture family labels are not sufficient capability checks. Do not reuse a tuning cache across incompatible devices/compiler versions, or let an unsupported optimized path fail deep inside execution when a guard can reject it.

**Validate:** boundary cases on both sides of every dispatch predicate; unsupported dtype/layout cases; eager and graph execution; assert or trace which branch was selected.

## GRAPH-04 — Treat capture bucketing as a latency, memory and semantics tradeoff

**Applies when:** a serving engine pads batches or uses multiple captured graph sizes.

**Do:** measure capture count, startup time, retained memory, padding work and replay latency together. Test mixed prefill/decode and speculative decode separately. A graph mode is part of the operator's execution environment.

**Source / evidence:** **documented architecture**: vLLM [`CUDA Graph design`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/docs/design/cuda_graphs.md) distinguishes full/piecewise execution and uniform versus nonuniform batches. Source ID: `vllm`. This design document contains historical context; inspect current dispatcher code before copying its defaults or backend-support claims. The tradeoff measurement is a **derived recommendation**.

**Counterconditions:** sharing a memory pool does not make graph objects or every allocation free; eager execution can win for rare shapes. Fixed split size does not guarantee a stable CTA count as KV lengths vary.

**Validate:** replay across bucket boundaries and padding cases, correctness of inactive rows, peak allocated/reserved memory after all captures, and workload-weighted latency. Report whether latency excludes setup and whether the graph covers the entire operation.
