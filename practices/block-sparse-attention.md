# Block-sparse attention and mask metadata

Keywords: block sparse, partial block, full block, mask_mod, video attention, block mask, metadata, sparse backward, five-point sampling.

**Evidence: source inspection at a pinned FlashAttention snapshot and upstream test reading; no local GPU or model-quality validation.** Source ID: `flashattention`.

## SPARSE-01 — A full block skips the logical mask, not sequence-boundary safety

**Applies when:** producing sparse metadata or separating fully valid and partially masked tiles.

**Do:** define full/partial/absent in terms of valid logical token pairs. A block may be full over in-bounds positions even when its physical tile extends beyond Q/K length. Keep sequence-length masks independently from custom logical-mask evaluation.

**Source / evidence:** [`block_sparse_utils.py`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/block_sparse_utils.py#L577-L607) uses `mask_mod=None` for full blocks while selectively retaining sequence-length masking. [`test_block_sparsity.py`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/tests/cute/test_block_sparsity.py#L57-L65) explicitly documents why CuTe and PyTorch can classify boundary blocks differently.

**Counterconditions:** marking a genuinely partial logical block full changes attention. Treating every boundary tile as safely unmasked can read invalid tokens. Raw equality of metadata arrays across libraries is not necessarily the right correctness criterion because their OOB conventions differ.

**Validate:** reconstruct the effective token-level mask on small inputs and compare attention results. Include both Q and K lengths just below/above block boundaries, all-masked rows, full-only and partial-only rows, and mixed lists.

## SPARSE-02 — Metadata block size is a contract with the kernel tile

**Applies when:** reusing a sparse mask across kernels or autotuning tile sizes.

**Do:** record sparse Q/K block sizes separately from kernel tile dimensions and stage/cluster factors. Rebuild or normalize metadata only under a supported mapping; validate counts, index dtype/device, list bounds and batch/head broadcasting.

**Source / evidence:** [`normalize_block_sparse_config`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/block_sparsity.py#L558-L610) requires KV sparse blocks to be multiples of the kernel N tile and only permits subtiles when explicitly allowed. In this snapshot, the varlen branch requires Q sparse blocks to match `q_stage * tile_m` and fixes the Q subtile factor to one. [`normalization helpers`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/block_sparsity.py#L247-L299) check int32 CUDA metadata and shared device.

**Counterconditions:** metadata made for one tile size cannot be passed unchanged to another merely because tensor ranks match. Counts and capacity differ; padded index storage is not a list of valid blocks. Runtime validators may check shape/type without exhaustively proving semantic index validity.

**Validate:** tile/metadata divisibility boundaries, fixed versus varlen layouts, supported broadcasts, invalid counts/indices at the integration boundary, and the dense mask represented after normalization. Include metadata construction in runtime cost when masks change per invocation.

**Ordering boundary:** the upstream [builder](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/compute_block_sparsity.py#L312-L319) appends block IDs in ascending traversal order. The inspected [consumer](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/block_sparse_utils.py#L481-L550) traverses each list from its final entry and applies sequence-boundary masking to the first processed block of each nonempty partial/full list. Preserve the producer's sorted-order invariant; treating an equivalent set of indices in arbitrary order as interchangeable requires a new correctness proof. This conclusion is source-derived, not a reproduced failure.

## SPARSE-03 — Sampled mask classification requires a proof about the mask family

**Applies when:** using a fast sparse-metadata builder that evaluates only selected points in a tile.

**Do:** use exhaustive classification unless the mask structure makes the sampled classifier valid. Document that structural assumption. The `fast_sampling` decorator is a declaration by the caller, not an automatic proof.

**Source / evidence:** [`ComputeBlockSparsity`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/compute_block_sparsity.py#L33-L41) optionally classifies from four corners plus center and restricts suitability to masks where those samples suffice; [`fast_sampling`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/block_sparsity.py#L723-L726) only sets an attribute. Upstream [fast-sampling tests](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/tests/cute/test_block_sparsity.py#L426-L445) cover selected mask families.

**Counterconditions:** a mask with a small unobserved interior hole or isolated allowed island can agree at all five samples yet have a different classification. Dynamic learned or arbitrary video masks need their own argument and tests; this optimization is not safe just because the output usually looks right.

**Validate:** exhaustive small-tile enumeration versus sampled classification, adversarial holes/islands away from samples, boundary tiles and each supported parameter range of the mask. Benchmark the builder separately and together with attention; skip it entirely when a reusable mask already exists.

## SPARSE-04 — Sparse backward needs transposed connectivity and matching write order

**Applies when:** training with sparse attention, especially deterministic backward.

**Do:** derive backward Q-contributor lists from the forward connectivity, retain full/partial classification, and create write-order metadata consistent with scheduler direction. Treat these as one coupled configuration.

**Source / evidence:** [`compute_dq_write_order`](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/block_sparsity.py#L80-L113) assigns each contributing KV block an accumulation rank and documents the scheduling direction required for progress. [`interface.py` deterministic checks](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/interface.py#L2363-L2386) requires partial write order, full-block write order when present, and the `spt` direction flag.

**Counterconditions:** swapping forward indices into a backward kernel does not transpose the graph. Reordering sparse lists without regenerating write-order metadata can violate accumulation ordering or progress. Forward-only sparse support does not establish backward support on a target architecture.

**Tail-block detail:** a sparse KV block can contain physical subtiles beyond the sequence end that never receive a CTA. The SM100 [lock calculation](https://github.com/Dao-AILab/flash-attention/blob/1bda8f9290cd48d030f1516f0e680cd464ef3554/flash_attn/cute/flash_bwd_sm100.py#L3676-L3704) explicitly handles this under `spt` by reversing only scheduled local groups; the corresponding release protocol skips vacant semaphore slots. Replacing that protocol with a simple reversed rank and unit increment can wait forever for an unscheduled contributor. This is a source-documented progress requirement, not a locally reproduced hang.

**Validate:** dense-reference dQ/dK/dV on asymmetric sparse patterns, empty contributors, mixed full/partial blocks and both supported order directions. Test repeated deterministic runs separately from numerical tolerance and inspect architecture guards in [backward rules](backward-and-determinism.md).
