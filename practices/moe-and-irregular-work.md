# MoE and irregular grouped work

Keywords: MoE, grouped GEMM, ragged tokens, routing, padding, expert parallel, sentinel, block assignment, weighted reduction.

All observations refer to the pinned vLLM snapshot; not every model selects these functional kernels. **No local GPU measurements.**

## MOE-01 — Distinguish real assignments, padded work and allocated capacity

**Applies when:** grouping token/expert assignments into GEMM blocks.

**Do:** retain separate counters for logical assignments, actual padded assignments and maximum storage. Treat sentinel tokens as invalid for loads/stores; use the actual padded-work count to bound active blocks. Promote offsets before multiplying by large strides.

**Source / evidence:** **observed implementation**: [`moe_align_block_size.py`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/moe_align_block_size.py#L11-L103) allocates an upper bound and returns a separate padded count; [`fused_moe_kernel`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L402-L438) masks sentinels and casts token offsets to int64 before address products. Source ID: `vllm`.

**Counterconditions:** padded capacity is not useful FLOPs. A graph may launch for capacity and read device-side valid counts; eliminating that distinction can produce extra work or invalid accesses. An int64 cast after a 32-bit product has overflowed is too late.

**Validate:** no tokens for an expert, one token, block-size minus/plus one, skewed assignments, maximum capacity and large strides. Use a CPU routing map to verify every valid token/expert pair appears once and sentinels contribute nothing.

## MOE-02 — Avoid expensive grouping when a guarded tiny-work path wins

**Applies when:** decode produces very few token/expert assignments relative to the expert count.

**Do:** compare route sorting/alignment plus efficient GEMM against a simpler assignment path that wastes some lanes but avoids setup. Keep both validity and profitability conditions visible.

**Source / evidence:** **observed implementation**: vLLM [`_prepare_expert_assignment`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L1541-L1587) has a low-token path with no sorted-token array, guarded by expert-map absence, a sparse-work heuristic and quantization constraints. The corresponding kernel places a single real assignment in the block and masks the rest. Source ID: `vllm`.

**Counterconditions:** its threshold is an upstream heuristic, not a portable constant. Bigger batches and heavy expert reuse can make grouping clearly worthwhile; expert-parallel mapping and grouped quantization change eligibility.

**Validate:** total prepare→GEMM→activation→GEMM→reduce time across the crossover, not GEMM alone. Test both sides of each predicate with identical routing and weights; record wasted padded work separately from real math.

## MOE-03 — Assign exactly one owner for routing weights and reduction

**Applies when:** mixing fused-expert implementations with prepare/finalize or communication backends.

**Do:** specify whether expert output is weighted, reduced and restored to token order at every boundary. Ensure each routing weight is applied once at the model-defined location. Keep nonlocal expert contributions zero or absent according to the combine contract.

**Source / evidence:** **observed implementation**: vLLM [`topk_weight_and_reduce.py`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py) distinguishes delegated, already-completed and explicit weight/reduce paths. [`fused_experts_impl`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L1774-L1855) places the routing multiply according to `apply_router_weight_on_input`. Source ID: `vllm`.

**Consumer boundary:** in the [Triton epilogue](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py#L569-L601), the routing multiply is an FP32 **GEMM epilogue after dequantization and bias, before the output cast**. The flag selects first-GEMM output before activation versus second-GEMM output; its name does not prove multiplication of the original hidden-state buffer. `TopKWeightAndReduceContiguous` weights its expert buffer **in place** when input weighting is false, so consuming that same buffer twice can double-weight it. `Delegate.apply()` raises until the prepare/finalize owner selects an implementation. `NoOP` means arithmetic is complete, but still copies into a supplied output unless it is the identical tensor object; it does not perform a general storage-alias test.

**Counterconditions:** multiplying before a nonlinear activation is not generally equivalent to multiplying after the expert network. Removing a zero write because a route is nonlocal can leave undefined data for a later sum. An output alias may make a copy unnecessary, but only when the caller's output contract is satisfied.

**Validate:** use nonuniform routing weights, nonlinear activation, expert-map holes and prefilled nonzero output/workspace buffers. Compare both expert-level intermediates and final token results. Test in-place behavior separately.

## MOE-04 — Benchmark skew and routing cost as part of grouped GEMM

**Applies when:** persistent/grouped scheduling or expert tiles are tuned.

**Do:** include balanced, highly skewed and many-empty-expert distributions. Measure sorting/permutation, quantization, dispatch, GEMMs, unpermutation and reduction at their real frequency. Record per-expert token histogram along with aggregate token count.

**Source / evidence:** **derived recommendation** grounded in vLLM's distinct assignment paths and padded expert blocks in [`fused_moe.py`](https://github.com/vllm-project/vllm/blob/8da75d61904cd0b2c5c322746ea15251773ec51f/vllm/model_executor/layers/fused_moe/fused_moe.py). Source ID: `vllm`.

**Counterconditions:** uniform synthetic tokens can conceal load imbalance, padding waste and setup overhead. A single global tile size may favor large experts while punishing small ones; specialize only if the workload and validation justify the added dispatch.

**Validate:** fixed total assignments with different histograms, changing expert count/top-k, graph/eager modes and communication if distributed. Compare against the current full operator path and make no end-to-end throughput claim from isolated GEMM timing.
