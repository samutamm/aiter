# Wiring the fused kernel into aiter's dispatch (and, later, vLLM)

## Context

`aiter/ops/flydsl/moe_sorting.py::fused_topk_softmax_moe_sort` currently exists
as a standalone function nobody calls: it's not wired into `aiter/fused_moe.py`
or any vLLM code path. It also bundles two things (gating + sorting) into one
call, which doesn't match how vLLM and aiter actually structure a MoE forward
pass today: gating happens at the router (vLLM), sorting happens later at the
expert-dispatch step (aiter's `moe_sorting()`), as two independent calls with
other work (e.g. expert-parallel masking) potentially happening in between.
Wiring in one combined function would mean restructuring that call sequence.
Wiring in two separate pieces, one per existing call site, does not.

A reference for this exact wiring pattern exists at
`/home/stammine/repos/brains/Arete/export/topk_gating_softmax_moe_sort_fusion/`
(see "Reference implementation" section at the end) — it already split gating
and sorting the same way, at the same two call sites. This plan follows that
shape but is written from scratch against this repo's current code, not a copy
of it.

## Plan: split into two independently-callable pieces

### 1. `fused_topk_gating` — gating-only, dropped into vLLM's router

New function in `aiter/ops/flydsl/moe_sorting.py` (or a new
`aiter/ops/flydsl/fused_topk_gating.py` if that reads better once written),
factored out of `fused_topk_softmax_moe_sort`'s current gating block
(lines ~254-268 today):

```python
def fused_topk_gating(
    gating_output,
    bias=None,
    topk=8,
    scoring_func="softmax",
    need_renorm=True,
    routed_scaling_factor=1.0,
):
    """Returns (topk_weights, topk_indices), shape [M, topk] each.

    Fast path (M<=16, no bias, scoring_func in {softmax-with-renorm, sigmoid}):
    selective top-k + in-PyTorch softmax/sigmoid renorm (exact denominator
    cancellation -- see fused_topk_softmax_moe_sort's docstring for why this
    is exact, not approximate).

    Otherwise falls back to aiter.ops.topk.topk_gating (HIP).
    """
```

Gate condition: identical to `fused_topk_softmax_moe_sort`'s current
`use_fast_path` check (`M <= 16 and not has_bias and scoring_supported`),
applied here independently of sorting.

Call site (vLLM): `vllm/model_executor/layers/fused_moe/router/*.py`, wherever
`gating_output.softmax(dim=-1)` / `.sigmoid()` + `torch.topk` currently happens
for the relevant router (e.g. `fused_topk_bias_router.py`). The patch there is
additive — an early return inside a `try/except: pass` before the existing
code, same shape as the reference's `framework_patch` diff.

### 2. Sort-only: expose the existing fast-sort kernel directly

The sort half already exists as the second part of
`fused_topk_softmax_moe_sort` (buffer allocation + the
`_compile_fused_topk_moe_sort` launcher call). Factor it into its own
function, parallel to `flydsl_moe_sorting_fwd`:

```python
def flydsl_fused_topk_moe_sort(
    topk_ids,
    topk_weights,
    sorted_ids=None,
    sorted_weights=None,
    sorted_expert_ids=None,
    num_valid_ids=None,
    moe_buf=None,
    num_experts=None,
    unit_size=32,
    stream=None,
):
    """Sort-only fast path: same inputs/outputs as flydsl_moe_sorting_fwd,
    but uses the single-block compact-binning kernel (fused_topk_moe_sort_kernel.py)
    instead of the generic multi-phase/oneshot sort.

    Gate: M = topk_ids.shape[0] <= 16. Falls back to flydsl_moe_sorting_fwd
    otherwise.
    """
```

Call site (aiter): `aiter/fused_moe.py::moe_sorting()`, as an early-return
branch before the existing `_flydsl_moe_sorting` / `_moe_sorting_impl` calls,
gated the same way `_USE_FLYDSL_MOE_SORTING` already gates the generic FlyDSL
path (module-level flag check first, then a shape/kwarg compatibility check:
`topk_ids.shape[0] <= 16`, `not expert_mask`, `dispatch_policy == 0`, etc. --
mirror the existing `_USE_CK_MOE_SORTING`/`_USE_FLYDSL_MOE_SORTING` guard at
`aiter/fused_moe.py:598-608`).

### 3. `fused_topk_softmax_moe_sort` becomes a thin composition

Once both pieces exist standalone, `fused_topk_softmax_moe_sort` just calls
`fused_topk_gating` then `flydsl_fused_topk_moe_sort` -- kept around for
tests/benchmarks and for any caller that genuinely wants gating+sort in one
call, but no longer the thing that gets wired into dispatch.

### What doesn't change

- The env var gate (`AITER_USE_FUSED_TOPK_MOE_SORT`), the M<=16/no-bias/
  scoring_func gating conditions, and the correction-bias gap documented in
  `todo.md` all stay as-is -- this is purely a factoring/wiring change, not a
  capability change.
- `aiter/ops/flydsl/kernels/fused_topk_moe_sort_kernel.py` (the actual kernel)
  is untouched; only the host-side call site changes.

## Verification

### 1. Split-piece unit tests in `op_tests/test_fused_topk_moe_sort.py`

The existing file only tests the combined `fused_topk_softmax_moe_sort`
against `sequential_topk_softmax_moe_sort` (see `_run_case`, `_compare`,
`set_fused_topk_moe_sort_backend`). Once gating and sorting are split out,
add new, separate test functions rather than folding this into `_run_case` --
the two pieces have different reference baselines and different shapes of
output, so a single shared helper would just grow branchy.

**Gating-only (`fused_topk_gating`)**

- Reference: `topk_weights, topk_indices` computed directly in-test from
  `gating_logits.softmax(dim=-1).topk(...)` (+ renorm) or `.sigmoid().topk(...)`,
  matching whatever `sequential_topk_softmax_moe_sort`'s gating half currently
  does -- do not reuse `sequential_topk_softmax_moe_sort` itself as the
  reference here since that also does the sort; pull just its gating logic
  into a small local reference fn (or factor *that* out too, named e.g.
  `sequential_topk_gating`, so both the new fast path and both test files
  share one baseline implementation).
- Compare `topk_weights` with `checkAllclose` (float tolerance) and
  `topk_indices` with `atol=0` exact match -- same split `_compare` already
  does for `sorted_weights` vs `sorted_ids`/`sorted_expert_ids`.
- Reuse the existing shape matrix: `(E, topk, unit_size)` pairs
  `(896, 16, 32)` and `(64, 8, 32)`, `M in (1, 2, 4, 8, 16)`,
  `scoring_func in ("softmax", "sigmoid")`, `has_bias in (False, True)` --
  same `itertools.product` as `test_fused_topk_moe_sort_decode_shapes`, since
  the gate condition is identical.
- Carry over the two edge-case tests as gating-only variants:
  - `softmax` + `need_renorm=False` must fall back (can't recover the true
    un-renormalized weight from K selected logits alone) -- assert it matches
    the reference bit-for-bit, not just "doesn't crash".
  - `M > 16` (prefill) must fall back regardless of the gate.
  - gate off (`_USE_FUSED_TOPK_MOE_SORT = False`, or whatever flag
    `fused_topk_gating` ends up checking) must still match the reference.
- Bias case: note `todo.md`'s correction-bias gap applies here too, so
  `has_bias=True` should still be included in the matrix (it currently forces
  fallback, i.e. `use_fast_path=False`) rather than skipped -- the test should
  assert fallback-is-exact, same as it implicitly does today via `_run_case`.

**Sort-only (`flydsl_fused_topk_moe_sort`)**

- Reference: `flydsl_moe_sorting_fwd` (the existing generic FlyDSL sort, same
  one `_USE_FLYDSL_MOE_SORTING` selects in `aiter/fused_moe.py`), fed
  `topk_ids`/`topk_weights` produced by `sequential_topk_softmax_moe_sort`'s
  gating half (or a fixed random `torch.topk` over logits -- the gating
  algorithm doesn't matter for a sort-only test, only that both sides see
  the *same* ids/weights).
- Reuse `_compare`'s shape/sentinel logic verbatim (capacity mismatch guard,
  `num_tokens_post_pad` from `num_valid_ids`, sentinel-masking of
  uninitialized `sorted_weights` capacity) -- it's already written generically
  in terms of `(ref, out, topk, num_rows, unit_size, label)` tuples and
  doesn't care which function produced either tuple.
- Shape matrix: same `(E, topk, unit_size)` x `M` sweep. Additionally test the
  `M > 16` and gate-off fallback cases the same way as gating.
- Also test `flydsl_fused_topk_moe_sort`'s own output-buffer-reuse path
  (passing pre-allocated `sorted_ids=`, `sorted_weights=`, etc. instead of
  `None`) if `flydsl_moe_sorting_fwd` supports that today -- check its
  signature; if it does, the fast-sort function should honor it identically
  since `aiter/fused_moe.py::moe_sorting()` likely passes pre-allocated
  buffers on the decode-graph-capture path (see
  `test_moe_sorting_flydsl_cuda_graph_capture` below).

**Composed function regression**

- Keep `test_fused_topk_moe_sort_decode_shapes` and friends in place, now
  exercising `fused_topk_softmax_moe_sort` as a caller of the two split
  functions rather than its own monolithic implementation -- they should
  pass unmodified if the refactor is behavior-preserving. This is the
  cheapest regression signal that the split didn't change either piece's
  numerics.

### 2. Wiring into `aiter/fused_moe.py::moe_sorting()`

- Follow `op_tests/test_moe_sorting.py`'s existing `set_moe_sorting_backend`
  pattern (`fm._USE_CK_MOE_SORTING` / `fm._USE_FLYDSL_MOE_SORTING` toggles) --
  add a parallel `"fused_topk"` branch there (or a dedicated
  `set_fused_topk_moe_sort_backend`-style toggle for
  `fm._USE_FUSED_TOPK_MOE_SORT`, mirroring what
  `test_fused_topk_moe_sort.py` already does at the `moe_sorting_mod` level)
  so `test_moe_sorting`'s existing parametrized cases can run with the new
  branch selected, not just a new standalone test.
- `test_moe_sorting` is parametrized over `has_expert_mask`, `dispatch_policy`,
  `padding_extra` (see its signature + the `dispatch_policy == 0` FlyDSL-only
  guard at the bottom of the shared setup). The new branch's gate from the
  reference impl is `M<=16, num_experts==896, topk==16, block_size==32,
  expert_mask is None` (i.e. `dispatch_policy==0` only, `has_expert_mask=False`
  only) -- so most of `test_moe_sorting`'s matrix will *not* hit the new
  branch and should fall through to the existing FlyDSL/CK path unchanged.
  Add an explicit assertion or log line confirming the new branch was
  actually taken for the in-gate cases (e.g. a counter/monkeypatch on the
  launcher, or just trust the shape gate and add a case specifically sized to
  hit it) -- otherwise a broken gate check could silently no-op the whole
  wiring and the test would still pass via fallback.
- Re-run `test_moe_sorting_invalid_topk_ids` (malformed-input handling) and
  `test_moe_sorting_flydsl_cuda_graph_capture` (CUDA-graph capture/replay)
  with the new backend selected for at least one case each -- the fused path
  reuses `flydsl_moe_sorting_fwd`'s kernel internals per the plan above, but
  the *buffer allocation* is new code and graph capture is exactly where
  stale/reallocated buffers break silently (replay reads old pointers).
- `test_moe_sorting_decode_graph_perf` is a perf harness (CLI-args-driven, not
  a pytest case) -- not required for correctness sign-off, but worth a manual
  run comparing the new branch's latency against `flydsl` to confirm the
  "fused" framing is actually faster for in-gate shapes, since that's the
  entire point of wiring it in.

### 3. vLLM router patch

Only after (1) and (2) pass: prototype the vLLM router patch for
`fused_topk_gating` separately, likely as its own follow-up task since it
touches a different repo. No aiter-side test coverage for this step --
correctness there is vLLM's router test suite's responsibility; this repo's
obligation ends at `fused_topk_gating` matching its documented reference
behavior, which (1) above already covers.

---

## Reference implementation (for context only, not to be copied verbatim)

`/home/stammine/repos/brains/Arete/export/topk_gating_softmax_moe_sort_fusion/`
wired its equivalent functions in with two small patches, each an early-return
inside a `try/except: pass` at an existing call site:

- `dispatch/dispatch_patch_fused_topk_moe_sorting.diff` patches
  `aiter/fused_moe.py::moe_sorting()` to call their sort-only
  `flydsl_compact_moe_sorting(topk_ids, topk_weights, ...)` when
  `AITER_USE_FUSED_TOPK_MOE_SORT=1` and the shape matches
  (`M<=16, num_experts==896, topk==16, block_size==32, expert_mask is None`).
- `dispatch/framework_patch_fused_topk_moe_sorting.diff` patches vLLM's
  `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py` to
  call their gating-only `fused_topk_gating(gating_output, ...)` under the
  same env var + shape gate, before vLLM's own softmax/topk code.

Notably, their combined `fused_topk_softmax_moe_sort` function (gating+sort in
one call) was exported in `aiter/ops/flydsl/__init__.py`'s module registry but
never actually called from either patch -- both patches used the split
gating-only / sort-only pieces instead, matching the two-call-site structure
already present in vLLM/aiter. This plan follows that same shape.
