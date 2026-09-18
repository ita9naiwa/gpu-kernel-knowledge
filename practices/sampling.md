# Sampling kernels: preserve distributions and RNG contracts

Keywords: sampling, top-k, top-p, min-p, logits, probabilities, seed, offset, deterministic, categorical, ties, CUDA Graph.

These rules follow FlashInfer's pinned APIs and tests. **No local GPU or statistical sampling validation was run.**

## SAMPLE-01 — Filter order and normalization define the distribution

**Applies when:** fusing top-k and top-p or replacing a sampling implementation.

**Do:** state whether top-p is applied after top-k renormalization or to the original distribution jointly with top-k. Preserve tie handling, support, output index mapping and scalar/per-row thresholds.

**Source / evidence:** **documented contract / observed branches**: FlashInfer [`top_k_top_p_sampling_from_logits`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L1421-L1575) distinguishes `top_k_first` and `joint`. Source ID: `flashinfer`.

**Counterconditions:** sequential and joint filtering are not interchangeable optimizations: top-k renormalization changes the probabilities used by the subsequent top-p stage. Boundary inclusion and ties still depend on the exact API.

**Validate:** small hand-computed distributions away from and at thresholds, tied K-boundary values, per-row K/P, repeated row indices and token-ID remapping. Compare support and normalized probabilities before testing sampled frequencies.

## SAMPLE-02 — Reproducibility and distributional equivalence are different tests

**Applies when:** changing RNG engines, rejection loops, launch geometry or compact-vocabulary sampling.

**Do:** preserve the documented seed/offset contract when exact replay is required. Separately test that outputs obey the intended categorical distribution. A deterministic implementation can be consistently biased; an unbiased implementation can consume random numbers in a different order.

**Source / evidence:** **observed upstream tests**: FlashInfer [`test_sampling.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/tests/utils/test_sampling.py) tests frequency/support and seed/offset reproducibility independently. Source ID: `flashinfer`. Statistical acceptance design is a **derived recommendation**.

**Counterconditions:** equality of sampled token IDs across unrelated algorithms is not necessary for distributional equivalence unless the API requires it. A single seed or a cosine comparison alone does not characterize rare-event errors or full distribution distance.

**Validate:** repeated seed/offset, changed seed/offset, zero-probability tokens never selected, empirical frequencies for small distributions and a stated statistical tolerance/sample count. Test batch composition effects if per-request reproducibility is required.

## SAMPLE-03 — Keep RNG state mutable across graph replay

**Applies when:** capturing sampling in CUDA Graphs or providing explicit RNG state.

**Do:** use the API's supported device seed/offset buffers and update contents with correct stream ordering. Record how many random values the selected operation consumes before advancing offsets; do not assume all sampling methods consume one value per output.

**Source / evidence:** **documented contract / observed implementation**: FlashInfer [`sampling.py`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L1473-L1489) requires tensor seed/offset for graph compatibility and assigns explicit-state updates to the caller. Internal methods request different offset increments. Source ID: `flashinfer`.

**State boundary:** the [internal wrapper](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L108-L128) replaces **both** values from the generator if either seed or offset is omitted. Supplying only a seed does not freeze that seed in this branch. The [validator](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L675-L736) requires both scalars or both tensors; tensors must be on the input device, 1D int64/uint64, each length one or output batch size. With indices, that batch size is the indices length, not the number of source distribution rows. The generator helper rounds requested reservations up to a multiple of four; those reservations are not a measured count of all dynamically consumed draws.

**Counterconditions:** capturing a fixed Python integer seed/offset is not equivalent to updating a device buffer. Shared RNG state across concurrent calls needs an explicit ordering/ownership policy. Runtime `check_nan` paths can perform device-to-host synchronization; inspect the selected branch before capture.

**Validate:** replay with identical state intentionally repeats, then replay with updated state changes draws while remaining in support. Test independent requests/streams with independent state and ensure offsets are not accidentally reused.

## SAMPLE-04 — Verify the actual fast path and invalid-input behavior

**Applies when:** optimizing small-K large-vocabulary sampling or benchmarking an API called fused.

**Do:** trace the gate and include selection, normalization, sample generation and index remapping in timing. Establish whether invalid probability rows raise, return a validity flag or have undefined behavior; keep that contract at the integration boundary.

**Source / evidence:** **observed implementation**: FlashInfer [`_top_k_first_fast_path_applicable`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/flashinfer/sampling.py#L1336-L1417) requires Python-integer K in 1–256, vocabulary ≥65536, K smaller than vocabulary and no row indices; even a zero-dimensional tensor K bypasses it. Tensor top-p remains allowed. The helper performs top-k selection, conversion/normalization, top-p sampling and a final gather back to vocabulary IDs: API fusion does not mean one launch. `deterministic` is forwarded to both selection and sampling, but the comment's sample-alignment statement concerns logits/probability versions of this compact path, not arbitrary fallback paths. [`test_sampling_nan_input`](https://github.com/flashinfer-ai/flashinfer/blob/0e4c173821a0aca29e9eb00c50c3bde8696e6dc6/tests/utils/test_sampling.py#L1159) exercises validity reporting. Source ID: `flashinfer`.

**Counterconditions:** per-row K or indexed distributions can bypass that path. Fewer retained categories do not imply one launch. A fallback with different tie ordering can preserve some metrics while breaking exact sample alignment.

**Validate:** K and vocabulary size around gate thresholds, logits versus probability entry points, NaN/all-zero rows per API contract, scalar versus tensor K and indices on/off. Report the effective path and complete timed region.
