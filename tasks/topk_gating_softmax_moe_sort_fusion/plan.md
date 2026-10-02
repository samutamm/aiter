# fused_topk_softmax_moe_sort (AITER, gfx950 decode)

## Context

KimiMoE's router currently runs `topk_gating` (HIP) then `flydsl_moe_sorting_fwd` (FlyDSL) as two separate kernel launches per MoE layer, 92 layers/decode step (184 launches). A prior AMD "Arete" optimization campaign (exported at `/home/stammine/repos/brains/Arete/export/topk_gating_softmax_moe_sort_fusion/`) identified this gap and produced a working prototype that fuses the two into effectively one GPU-side pipeline for the decode regime (M ≤ 16), yielding a measured +1.66% e2e tps via launch-count reduction, even though the fused kernel is slower standalone than the sequential baseline (96µs vs ~30µs) — the win is from eliminating CPU-GPU launch/sync bubbles across 92 layers, not from a faster microbenchmark.

This plan adapts that prototype into `aiter/ops/flydsl/moe_sorting.py` as a new `fused_topk_softmax_moe_sort` entrypoint, per the task's implementation guide, with two deliberate deviations the user confirmed:

1. **Reuse the export's PyTorch-level selective top-k/softmax**, not a from-scratch in-kernel/in-register top-k. Top-k selection + softmax/sigmoid renorm stay in PyTorch (`torch.topk` + selective softmax, using the exact-denominator-cancellation identity `renorm(softmax_E(Z)[K]) == softmax_K(Z[K])`), falling back to `aiter.ops.topk.topk_gating` whenever a correction bias is supplied. Only the **sort** step becomes the custom single-block FlyDSL kernel, with gating weights passed in directly instead of round-tripping through the generic multi-phase `moe_sorting_kernel.py` path.
2. **Parametrize the kernel over `(num_experts, topk, unit_size)`** via an `lru_cache`-keyed compiler function, following the existing `_compile_moe_sorting_oneshot` convention in `aiter/ops/flydsl/kernels/moe_sorting_kernel.py`, instead of hardcoding `E=896` into the LDS struct as the export does. This makes the op usable beyond Kimi-K3.

Bias semantics (confirmed against `aiter/ops/topk.py:219-260`, `biased_grouped_topk_torch`): bias is added to the **score** before top-k selection (`scores_for_choice = sigmoid(logit) + bias`, selection happens on the biased score), but the returned top-k **weights** are gathered from the raw (unbiased) score and then renormalized. The PyTorch selective-softmax fast path only applies when `bias is None`; any non-empty bias routes straight to `topk_gating` for the gating step (bias changes which experts are selected, so the "evaluate only the 16 winners" shortcut can't trivially apply bias post-hoc without re-deriving the correction-bias kernel semantics in Python — not worth the complexity here).

## Implementation

### 1. New parametrized FlyDSL sort kernel — `aiter/ops/flydsl/kernels/fused_topk_moe_sort_kernel.py` (new file)

Port the export's `_compact_binning_kernel` (`kernel/fused_topk_moe_sorting.py:47-169`) algorithm — LDS counts/tile_offsets arrays, atomic per-expert counting, chunked prefix-sum via the existing `_allwave_inclusive_prefix_sum` (imported from `moe_sorting_kernel.py`), sentinel pre-fill, sequential per-token scatter with `atomic_add_i32` slot allocation — but:
- Replace every hardcoded `896` with a python-level `num_experts` compile-time constant captured by closure.
- Wrap kernel construction in a `_compile_fused_topk_moe_sort(*, num_experts, topk, unit_size, block_size=256)` function, `functools.lru_cache(maxsize=64)`-decorated, mirroring `_compile_moe_sorting_oneshot` (`moe_sorting_kernel.py:223-254`): returns the compiled `@flyc.jit` launcher closed over those shape constants.
- `BLOCK_SIZE=256` stays fixed (sufficient for scatter/count loops of any practical `E`; loops over `range_constexpr(0, E, BLOCK_SIZE)` already generalize correctly once `896` → `E`).
- LDS struct `FusedBinningLDS` becomes a locally-defined `@fx.struct` inside the compiler function (per-shape struct sizes, same pattern as `moe_sorting_oneshot_kernel`'s `SharedStorage`).
- Reuse `atomic_add_i32` from `kernels_common.py` and `_allwave_inclusive_prefix_sum` from `moe_sorting_kernel.py` — no duplication.
- Packed id format stays `(topk_slot << 24) | token_id`; sentinel stays `(topk << 24) | M`, matching `moe_sorting_kernel.py`'s convention so `_compare_moe_sorting_outputs` in the existing test suite works unmodified.

### 2. Host entrypoint — add to `aiter/ops/flydsl/moe_sorting.py`

Add alongside the existing `flydsl_moe_sorting_fwd`:

```python
_USE_FUSED_TOPK_MOE_SORT = os.environ.get("AITER_USE_FUSED_TOPK_MOE_SORT", "0") == "1"

def fused_topk_softmax_moe_sort(
    gating_logits, bias=None, topk=..., unit_size=32,
    scoring_func="softmax", need_renorm=True,
    sorted_ids=None, sorted_weights=None, sorted_expert_ids=None,
    num_valid_ids=None, moe_buf=None, stream=None,
) -> tuple[...]:
```

Logic (ported from the export's `fused_topk_softmax_moe_sort`/`fused_topk_gating`, `dispatch/aiter_wrapper_fused_topk_moe_sorting.py:259-619`):
- Gate: `_USE_FUSED_TOPK_MOE_SORT and M <= 16`. Anything else (prefill, flag off) falls back to the existing sequential `topk_gating(...)` + `flydsl_moe_sorting_fwd(...)` path — no new fallback code needed, just call the two existing functions directly (simpler than the export's separate `fallback_sequential_gating_and_sort`, since both pieces already exist in this file/module).
- Fast path: selective top-k + softmax/sigmoid in PyTorch per the bias semantics above, then call the new parametrized FlyDSL sort launcher (`_compile_fused_topk_moe_sort(num_experts=E, topk=topk, unit_size=unit_size)(...)`) instead of `flydsl_moe_sorting_fwd`.
- Buffer sizing (`max_tokens_padded`, `max_blocks`), `moe_buf.zero_()`, `sentinel` construction: copy verbatim from the export — already correct and consistent with `moe_sorting_kernel.py` conventions.
- No new workspace needed (unlike `flydsl_moe_sorting_fwd`, the compact-binning kernel does everything in LDS, no HBM workspace).

### 3. Tests — new `op_tests/test_fused_topk_moe_sort.py`

- Reuse `run_torch_moe_sorting` and `_compare_moe_sorting_outputs` from `op_tests/test_moe_sorting.py` for the sort-correctness check (bit-identical `sorted_expert_ids`/`num_valid_ids`, sentinel-masked `sorted_ids`/`sorted_weights` comparison).
- Reuse reference gating math patterns from `op_tests/test_moe_topk_gating.py` (softmax and sigmoid+bias) to build the `topk_ids`/`topk_weights` reference independently of the implementation under test.
- Parametrize: `M ∈ {1,2,4,8,16}`, `scoring_func ∈ {"softmax", "sigmoid"}`, with and without `bias`, and at least two `(num_experts, topk, unit_size)` shapes — `(896, 16, 32)` (Kimi-K3) plus one smaller shape (e.g. `(64, 8, 32)`) to prove the parametrization actually generalizes, not just happening to work at the one hardcoded shape.
- Assert the `M > 16` fallback path produces identical results to directly calling `topk_gating` + `flydsl_moe_sorting_fwd` (i.e. the gate correctly no-ops).
- A `set_fused_topk_moe_sort_backend`-style helper (monkeypatching the module-level `_USE_FUSED_TOPK_MOE_SORT` flag), matching `op_tests/test_moe_sorting.py`'s `set_moe_sorting_backend` convention, so tests don't need subprocess env-var tricks.

### Not in scope for this pass

- Wiring into `aiter/fused_moe.py`'s `moe_sorting()` dispatcher or vLLM's router (the export's `dispatch_patch_*.diff` files) — the todo only asks for the op itself, gated and fallback-safe, dropped into `moe_sorting.py`. Model-level integration is a follow-up once this op is validated on hardware.
- On-device dynamic token-count (`local_tokens_tensor`) support for CUDA-graph replay with varying decode batch size — `M` is taken from `gating_logits.shape[0]` (host-known, no `.item()` sync) exactly as the export does; this satisfies "no host sync" without needing per-M kernel recompilation. True dynamic-M-under-one-capture is out of scope unless hardware testing shows it's needed.

## Verification

On `gbt350-odcdh2-c05-1.png-odc.dcgpu`, inside a container built from this
task's own `tasks/topk_gating_softmax_moe_sort_fusion/Dockerfile`
(`vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516`
base, this branch's commit `2ed2f032a` installed as the active `aiter`) --
image tag: `vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516_aiter_2ed2f032a`.
This supersedes the generic `docker/run_shape_sweep_bench_test.sh` base image
for this task, since that script's image doesn't pin a specific aiter
commit -- important here because the base image ships its own pre-installed
`amd-aiter` under `dist-packages/aiter`, which silently shadows a plain
`setup.py develop` install unless it's removed first (the Dockerfile does
this; see its `pip uninstall amd-aiter` step). Verified by importing
`fused_topk_gating`/`flydsl_fused_topk_moe_sort` and confirming
`aiter.__file__` resolves to `/aiter/aiter/__init__.py` at `git rev-parse
HEAD` == `2ed2f032a7c4e06da539ff3a4f85340c5f2dcfd2`, then running both
functions against real GPU gating logits.

1. `python op_tests/test_fused_topk_moe_sort.py` — all shape/scoring_func/bias combinations pass bit-identical comparison against the sequential reference.
2. Confirm `AITER_USE_FUSED_TOPK_MOE_SORT=0` (default) reproduces today's behavior exactly (fallback path untouched).
3. Microbenchmark the fused op vs sequential `topk_gating`+`flydsl_moe_sorting_fwd` at M∈{1,2,4,8,16} for the Kimi-K3 shape, to record the per-call latency delta going into the plan's writeup (expect it may still be slower standalone per the export's finding — document rather than treat as a blocker, since the real win is launch-count reduction at the model level, out of scope here).
