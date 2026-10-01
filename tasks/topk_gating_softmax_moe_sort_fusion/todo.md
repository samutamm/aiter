
# topk_gating_softmax_moe_sort_fusion

## Details
This is an implementation task.
- branch: topk_gating_softmax_moe_sort_fusion
- testing node: gbt350-odcdh2-c05-1.png-odc.dcgpu
- testing docker image:
  - vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516
- example docker run command (just take the docker run part, ignore the rest):
  - docker/run_shape_sweep_bench_test.sh
  - for iterative testing, feel free to modify files inside docker file on the node

## Implementation guide

Implementation guide — fused_topk_softmax_moe_sort (AITER, gfx950)

Fuse topk_gating/topk_softmax and flydsl_moe_sorting_fwd into one FlyDSL kernel (@flyc.kernel), dropped into aiter/ops/flydsl/moe_sorting.py, gated behind AITER_USE_FUSED_TOPK_MOE_SORT=1 with transparent fallback for M > 16 (prefill).

1. Gate on M ≤ 16 (decode only). For this regime every active expert gets ≤16 tokens < tile size 32, so skip dynamic ceil-division — hardcode padded_count(e) = 32 * (count(e) > 0) and assign each active expert a single contiguous 32-token tile directly.
2. Compute activations only for the 16 selected experts, not all 896 — the full-vocabulary softmax denominator cancels exactly under top-k renorm (renorm(softmax_E(Z)[K]) ≡ softmax_K(Z[K]), SymPy-verified), so do top-k selection on raw logits/sigmoid scores first, then normalize only the 16 winners in-register.
3. Keep topk ids/weights entirely in registers (14 logits/lane across a 64-lane wave, one wave per token for M≤16) — never materialize topk_ids/topk_weights to HBM.
4. Use a compact 256-slot LDS table (expert_counts[896], active_expert_list[256], compact_token_ids[256], compact_weights[256] ≈ 9.5 KB total) instead of the original [16×897] sparse mesh (57.4 KB) for the binning/prefix-sum step.
5. Single-block execution + grid-stride zeroing: block 0 does the fused topk+sort; blocks 1..N-1 zero moe_buf concurrently via 128-bit vectorized stores — same pattern as the existing FlyDSL oneshot sort.
6. Pad sentinel slots (C_e to 31 within each expert's tile) with (16<<24)|M id and 0.0 weight, written via Vec4 128-bit stores.
7. No host sync — pass M as a dynamic tensor scalar/dispatch constant, never a CPU .item() call, to preserve CUDA-graph/HIP-graph safety.

Validate against sequential AITER reference across M ∈ {1,2,4,8,16} for both softmax and sigmoid+bias scoring — expect bit-identical expert/token IDs and 0.0 max abs diff (already passed in smoketest). Target footprint: ≤9.5 KB LDS, ≤64 VGPRs, 100% occupancy (4 waves/SIMD).

See /home/stammine/repos/brains/Arete/export/topk_gating_softmax_moe_sort_fusion/ for more info.

## Next step

- Create a detailed plan into tasks/topk_gating_softmax_moe_sort_fusion/plan.md and try to iron-out few details.
- Let me check the plan, then implement.

## Known gap (deferred): correction_bias support in the fast path

`fused_topk_softmax_moe_sort`'s fast path (aiter/ops/flydsl/moe_sorting.py) always
falls back to the sequential `topk_gating` + `flydsl_moe_sorting_fwd` path whenever
a correction `bias` is supplied, regardless of the env gate / M / scoring_func:

```python
use_fast_path = _USE_FUSED_TOPK_MOE_SORT and M <= 16 and not has_bias and scoring_supported
```

Why: bias is added to the score *before* top-k selection (confirmed against
`aiter/ops/topk.py`'s `biased_grouped_topk_torch`: `scores_for_choice = sigmoid(logit)
+ bias`, selection happens on the biased score, but the returned weights are gathered
from the *raw* unbiased score and then renormalized). The fast path's "evaluate only
the K selected experts" shortcut relies on selecting via `torch.topk` on the raw
logits — supporting bias would mean re-deriving/replicating `topk_gating`'s exact
correction-bias selection semantics in PyTorch, which was judged out of scope for
this pass (see tasks/topk_gating_softmax_moe_sort_fusion/plan.md, "Not in scope for
this pass").

Practical consequence: the microbenchmark (op_tests/bench_fused_topk_moe_sort.py)
must be run with `bias=None` to actually exercise the new fused kernel. A benchmark
that passes a non-empty bias (as the reference Arete export's
`kernel/bench_fused_topk_moe_sorting.py` does) ends up comparing the sequential
fallback path against itself for both "sequential" and "fused" candidates — the new
kernel never runs — so such a benchmark would be measuring noise, not a real
comparison.

Follow-up (not yet implemented): add a bias-aware fast path — compute
`scores_for_choice = score(logits) + bias` for selection, then gather raw
(unbiased) scores for the top-k indices and renormalize, matching
`biased_grouped_topk_torch`'s semantics — so bias-routed decode calls (e.g.
Llama4/DeepSeek-style `sigmoid` + `correction_bias` routing) can also use the
fused kernel instead of always falling back.

## Microbenchmark results (op_tests/bench_fused_topk_moe_sort.py, gbt350-odcdh2-c05-1.png-odc.dcgpu, vllm/vllm-openai-rocm:nightly-rocm100-36768d1bfd39094681cdbc8cb37d4b31c0729c89)

Fast path is standalone *slower* than the sequential baseline across every
real model shape tested, not just Kimi-K3 — bigger relative gap on
smaller-E shapes, since the sequential HIP `topk_gating` kernel gets cheaper
at small E (6-12us) while the single-block LDS sort kernel has a roughly
fixed ~20-33us floor. M=32/64/128 rows confirm the M>16 fallback gate adds
no measurable overhead (ratio ~1.0x, same code path both sides).

| mode | model | M | E | topk | unit_size | sequential us | fused us | ratio |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| fused | Kimi-K3 | 1 | 896 | 16 | 32 | 26.81 | 44.92 | 0.597 |
| fused | Kimi-K3 | 2 | 896 | 16 | 32 | 28.11 | 47.14 | 0.596 |
| fused | Kimi-K3 | 4 | 896 | 16 | 32 | 28.55 | 47.94 | 0.596 |
| fused | Kimi-K3 | 8 | 896 | 16 | 32 | 29.54 | 50.90 | 0.580 |
| fused | Kimi-K3 | 16 | 896 | 16 | 32 | 33.40 | 54.42 | 0.614 |
| fused | Kimi-K2 | 1 | 384 | 8 | 32 | 8.62 | 32.50 | 0.265 |
| fused | Kimi-K2 | 2 | 384 | 8 | 32 | 9.69 | 32.03 | 0.303 |
| fused | Kimi-K2 | 4 | 384 | 8 | 32 | 9.93 | 33.88 | 0.293 |
| fused | Kimi-K2 | 8 | 384 | 8 | 32 | 10.80 | 39.77 | 0.271 |
| fused | Kimi-K2 | 16 | 384 | 8 | 32 | 12.25 | 43.66 | 0.281 |
| fused | DeepSeek-V3 | 1 | 256 | 8 | 32 | 7.96 | 22.02 | 0.361 |
| fused | DeepSeek-V3 | 2 | 256 | 8 | 32 | 8.93 | 23.81 | 0.375 |
| fused | DeepSeek-V3 | 4 | 256 | 8 | 32 | 9.46 | 25.37 | 0.373 |
| fused | DeepSeek-V3 | 8 | 256 | 8 | 32 | 10.22 | 29.60 | 0.345 |
| fused | DeepSeek-V3 | 16 | 256 | 8 | 32 | 12.15 | 32.16 | 0.378 |
| fused | GLM-5 | 1 | 256 | 8 | 32 | 7.87 | 22.03 | 0.357 |
| fused | GLM-5 | 2 | 256 | 8 | 32 | 8.86 | 23.33 | 0.380 |
| fused | GLM-5 | 4 | 256 | 8 | 32 | 9.49 | 25.42 | 0.374 |
| fused | GLM-5 | 8 | 256 | 8 | 32 | 10.26 | 29.47 | 0.348 |
| fused | GLM-5 | 16 | 256 | 8 | 32 | 12.10 | 32.15 | 0.376 |
| fused | Qwen3-235B | 1 | 128 | 8 | 32 | 6.89 | 19.94 | 0.345 |
| fused | Qwen3-235B | 2 | 128 | 8 | 32 | 7.53 | 21.41 | 0.352 |
| fused | Qwen3-235B | 4 | 128 | 8 | 32 | 8.26 | 23.11 | 0.357 |
| fused | Qwen3-235B | 8 | 128 | 8 | 32 | 9.02 | 24.80 | 0.364 |
| fused | Qwen3-235B | 16 | 128 | 8 | 32 | 10.12 | 27.46 | 0.368 |
| fused | MiniMax-M3 | 1 | 128 | 4 | 32 | 5.96 | 19.94 | 0.299 |
| fused | MiniMax-M3 | 2 | 128 | 4 | 32 | 6.79 | 21.43 | 0.317 |
| fused | MiniMax-M3 | 4 | 128 | 4 | 32 | 7.46 | 22.92 | 0.325 |
| fused | MiniMax-M3 | 8 | 128 | 4 | 32 | 8.15 | 24.51 | 0.333 |
| fused | MiniMax-M3 | 16 | 128 | 4 | 32 | 9.16 | 27.18 | 0.337 |
| fallback | Kimi-K3 | 32 | 896 | 16 | 32 | 40.77 | 40.70 | 1.002 |
| fallback | Kimi-K3 | 64 | 896 | 16 | 32 | 40.98 | 40.93 | 1.001 |
| fallback | Kimi-K3 | 128 | 896 | 16 | 32 | 41.02 | 41.09 | 0.998 |

## Wiring into aiter/fused_moe.py::moe_sorting() -- test results (gbt350-odcdh2-c05-1.png-odc.dcgpu, vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516, 2026-10-01)

Per tasks/topk_gating_softmax_moe_sort_fusion/wiring_kernel_to_moe.md: split
`fused_topk_softmax_moe_sort` into standalone `fused_topk_gating` (gating-only)
and `flydsl_fused_topk_moe_sort` (sort-only), with the combined function now a
thin composition of the two. Wired `flydsl_fused_topk_moe_sort` into
`aiter/fused_moe.py::moe_sorting()` as a new dispatch branch gated on
`_USE_FUSED_TOPK_MOE_SORT`, `dispatch_policy==0`, `expert_mask is None`,
`M<=16` (checked before the existing `_USE_FLYDSL_MOE_SORTING` branch).

All GPU runs below used `AITER_USE_FUSED_TOPK_MOE_SORT=1` where relevant.

- `op_tests/test_fused_topk_moe_sort.py` (pytest, 11 tests): **all passed**,
  covering the pre-existing combined-function cases plus new gating-only
  (`fused_topk_gating` vs HIP `topk_gating`) and sort-only
  (`flydsl_fused_topk_moe_sort` vs `flydsl_moe_sorting_fwd`) cases -- decode
  shape sweep, softmax-no-renorm fallback, M>16 prefill fallback, and
  gate-off exactness, each tested against both split pieces independently.
- `op_tests/test_moe_sorting.py::test_moe_sorting` (direct run,
  `-m 1 2 4 8 16 -e 896 -t 16 -md 7168 -p 0 -dp 0 -em f -accum t -rc valid`,
  extended with a new `"fused_topk"` candidate/backend alongside
  `opus`/`ck`/`flydsl`): **all passed** (`err=0` for every M in {1,2,4,8,16}
  against the torch reference) -- confirms the new moe_sorting() branch is
  both reachable and correct for in-gate shapes.
- `op_tests/test_moe_sorting.py::test_moe_sorting_flydsl_cuda_graph_capture`
  (same run, included in the sweep above): **passed** -- "moe_sorting FlyDSL
  cuda-graph capture/replay: all passed".
- `op_tests/test_moe_sorting.py::test_moe_sorting_invalid_topk_ids`: extended
  `direct_cases` with `("fused_topk", 8, 0, False)` (decode-sized M=8, in
  gate) alongside `all-empty`/`mixed` routing cases; **all passed**,
  confirming the new branch handles the -1 sentinel / malformed-input paths
  the same as the existing backends.
- `op_tests/bench_fused_topk_moe_sort.py` rerun on the current branch (table
  above, same node/image) to confirm the gating/sort split didn't change the
  combined function's numerics or performance -- numbers match the prior
  recording within noise (e.g. Kimi-K3 M=1: 44.92us -> 44.26us; Kimi-K2 M=1:
  32.50us -> 30.81us), so the known "fast path is standalone slower than
  sequential" finding above still stands; this refactor is a wiring change,
  not a perf change.

Not run this pass: `test_moe_sorting_decode_graph_perf` (perf-only CLI
harness, no correctness assertions) and the vLLM router patch (separate
repo, follow-up task per the wiring plan).