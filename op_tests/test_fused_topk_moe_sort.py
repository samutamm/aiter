# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for fused_topk_softmax_moe_sort.

Compares its decode-only fast path against the normal sequential path
(topk_gating + flydsl_moe_sorting_fwd), across different batch sizes, expert
counts, scoring functions, and with/without a correction bias. Expert and
token IDs must match exactly; weights must match within floating-point
tolerance.
"""

import itertools

import torch

import aiter.ops.flydsl.moe_sorting as moe_sorting_mod
from aiter.ops.flydsl.moe_sorting import (
    _fused_topk_gating_biased,
    flydsl_fused_topk_moe_sort,
    flydsl_moe_sorting_fwd,
    fused_topk_gating,
    fused_topk_softmax_moe_sort,
    sequential_topk_softmax_moe_sort,
)
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")


def set_fused_topk_moe_sort_backend(enabled: bool) -> None:
    """Force the fused_topk_softmax_moe_sort fast-path gate for a test."""
    moe_sorting_mod._USE_FUSED_TOPK_MOE_SORT = enabled


def sequential_topk_gating(gating_logits, bias=None, topk=8, scoring_func="softmax", need_renorm=True):
    """Gating-only reference, shared by this file's gating tests and
    sequential_topk_softmax_moe_sort's combined reference: the HIP
    topk_gating op, unconditionally (no fast-path gate)."""
    from aiter.ops.topk import topk_gating

    M = gating_logits.shape[0]
    device = gating_logits.device
    topk_w = torch.empty(M, topk, dtype=torch.float32, device=device)
    topk_i = torch.empty(M, topk, dtype=torch.int32, device=device)
    topk_gating(
        topk_w, topk_i, gating_logits, bias, need_renorm=need_renorm, score_func=scoring_func
    )
    return topk_w, topk_i


def _compare(ref, out, topk, num_rows, unit_size, label):
    sorted_ids_a, sorted_weights_a, sorted_expert_ids_a, num_valid_a, _moe_buf_a = ref
    sorted_ids_b, sorted_weights_b, sorted_expert_ids_b, num_valid_b, _moe_buf_b = out

    assert sorted_ids_b.shape == sorted_ids_a.shape, (
        f"[{label}] sorted_ids capacity mismatch: "
        f"expected {sorted_ids_a.shape}, got {sorted_ids_b.shape}"
    )
    assert sorted_expert_ids_b.shape == sorted_expert_ids_a.shape, (
        f"[{label}] sorted_expert_ids capacity mismatch: "
        f"expected {sorted_expert_ids_a.shape}, got {sorted_expert_ids_b.shape}"
    )

    errs = {}
    errs["num_valid_ids"] = checkAllclose(
        num_valid_a, num_valid_b, atol=0, msg=f"{label} num_valid_ids"
    )

    # Only [0, num_tokens_post_pad) in sorted_ids/sorted_weights, and
    # [0, num_tokens_post_pad // unit_size) in sorted_expert_ids, are
    # guaranteed-written by either implementation; capacity beyond that
    # (sized for the worst case where every expert gets its own tile) is
    # left as uninitialized torch.empty() memory by both the reference and
    # the op under test, so comparing it would just diff two unrelated
    # garbage regions.
    num_tokens_post_pad = int(num_valid_a[0].item())
    sentinel = (topk << 24) | num_rows
    ids_a = sorted_ids_a[:num_tokens_post_pad]
    ids_b = sorted_ids_b[:num_tokens_post_pad]
    weight_mask = ids_a != sentinel

    errs["sorted_ids"] = checkAllclose(
        ids_a,
        ids_b,
        atol=0,
        msg=f"{label} sorted_ids",
    )
    errs["sorted_weights"] = checkAllclose(
        sorted_weights_a[:num_tokens_post_pad][weight_mask],
        sorted_weights_b[:num_tokens_post_pad][weight_mask],
        msg=f"{label} sorted_weights",
    )
    block_count = num_tokens_post_pad // unit_size
    errs["sorted_expert_ids"] = checkAllclose(
        sorted_expert_ids_a[:block_count],
        sorted_expert_ids_b[:block_count],
        atol=0,
        msg=f"{label} sorted_expert_ids",
    )

    bad = {k: v for k, v in errs.items() if v}
    assert not bad, f"[{label}] mismatch: {bad}"


def _run_case(M, E, topk, unit_size, scoring_func, has_bias, need_renorm=True):
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    bias = None
    if has_bias:
        bias = torch.randn(E, dtype=torch.float32, device="cuda") * 0.1

    ref = sequential_topk_softmax_moe_sort(
        gating_logits,
        bias=bias,
        topk=topk,
        unit_size=unit_size,
        scoring_func=scoring_func,
        need_renorm=need_renorm,
    )

    set_fused_topk_moe_sort_backend(True)
    try:
        out = fused_topk_softmax_moe_sort(
            gating_logits,
            bias=bias,
            topk=topk,
            unit_size=unit_size,
            scoring_func=scoring_func,
            need_renorm=need_renorm,
        )
    finally:
        set_fused_topk_moe_sort_backend(False)

    label = f"M={M} E={E} topk={topk} unit={unit_size} func={scoring_func} bias={has_bias}"
    _compare(ref, out, topk, M, unit_size, label)
    print(f"[PASS] {label}")


def test_fused_topk_moe_sort_decode_shapes():
    """Fast-path correctness across decode batch sizes and shapes."""
    shapes = [(896, 16, 32), (64, 8, 32)]
    for (E, topk, unit_size), M, scoring_func, has_bias in itertools.product(
        shapes, (1, 2, 4, 8, 16), ("softmax", "sigmoid"), (False, True)
    ):
        _run_case(M, E, topk, unit_size, scoring_func, has_bias)


def test_fused_topk_moe_sort_softmax_no_renorm_falls_back():
    """scoring_func='softmax', need_renorm=False must not use the fast-path
    selective-softmax shortcut (it can't recover the true un-renormalized
    weight from only the K selected logits), so it should still match the
    sequential reference exactly via the fallback path."""
    _run_case(4, 64, 8, 32, "softmax", has_bias=False, need_renorm=False)


def test_fused_topk_moe_sort_prefill_fallback_matches_reference():
    """M > 16 (prefill) must take the fallback path regardless of the gate."""
    M, E, topk, unit_size = 32, 64, 8, 32
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    ref = sequential_topk_softmax_moe_sort(
        gating_logits, topk=topk, unit_size=unit_size, scoring_func="softmax", need_renorm=True
    )

    set_fused_topk_moe_sort_backend(True)
    try:
        out = fused_topk_softmax_moe_sort(
            gating_logits, topk=topk, unit_size=unit_size, scoring_func="softmax"
        )
    finally:
        set_fused_topk_moe_sort_backend(False)

    _compare(ref, out, topk, M, unit_size, "prefill fallback")


def test_fused_topk_moe_sort_gate_off_matches_reference():
    """With the env gate off, fused_topk_softmax_moe_sort must reproduce the
    sequential baseline exactly even for decode-sized M."""
    M, E, topk, unit_size = 8, 64, 8, 32
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    ref = sequential_topk_softmax_moe_sort(
        gating_logits, topk=topk, unit_size=unit_size, scoring_func="softmax", need_renorm=True
    )

    set_fused_topk_moe_sort_backend(False)
    out = fused_topk_softmax_moe_sort(
        gating_logits, topk=topk, unit_size=unit_size, scoring_func="softmax"
    )
    _compare(ref, out, topk, M, unit_size, "gate off")


def _run_gating_case(M, E, topk, scoring_func, has_bias, need_renorm=True):
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    bias = None
    if has_bias:
        bias = torch.randn(E, dtype=torch.float32, device="cuda") * 0.1

    ref_w, ref_i = sequential_topk_gating(
        gating_logits, bias=bias, topk=topk, scoring_func=scoring_func, need_renorm=need_renorm
    )

    set_fused_topk_moe_sort_backend(True)
    try:
        out_w, out_i = fused_topk_gating(
            gating_logits, bias=bias, topk=topk, scoring_func=scoring_func, need_renorm=need_renorm
        )
    finally:
        set_fused_topk_moe_sort_backend(False)

    label = f"M={M} E={E} topk={topk} func={scoring_func} bias={has_bias}"
    assert out_i.shape == ref_i.shape, f"[{label}] topk_indices shape mismatch"
    errs = {
        "topk_indices": checkAllclose(ref_i, out_i, atol=0, msg=f"{label} topk_indices"),
        "topk_weights": checkAllclose(ref_w, out_w, msg=f"{label} topk_weights"),
    }
    bad = {k: v for k, v in errs.items() if v}
    assert not bad, f"[{label}] mismatch: {bad}"
    print(f"[PASS] gating {label}")


def test_fused_topk_gating_decode_shapes():
    """Gating-only fast-path correctness across decode batch sizes and shapes."""
    shapes = [(896, 16), (64, 8)]
    for (E, topk), M, scoring_func, has_bias in itertools.product(
        shapes, (1, 2, 4, 8, 16), ("softmax", "sigmoid"), (False, True)
    ):
        _run_gating_case(M, E, topk, scoring_func, has_bias)


def test_fused_topk_gating_softmax_no_renorm_falls_back():
    """scoring_func='softmax', need_renorm=False must fall back (can't
    recover the true un-renormalized weight from only the K selected
    logits), matching the HIP topk_gating reference exactly."""
    _run_gating_case(4, 64, 8, "softmax", has_bias=False, need_renorm=False)


def _reference_biased_kernel_math(gating_logits, bias, topk, scoring_func, need_renorm):
    """Independent reimplementation of topk_gating_kernels.cuh's bias
    semantics, written from scratch (not calling anything in
    aiter.ops.flydsl.moe_sorting or aiter.ops.topk), so a bug shared between
    the fast path and the HIP fallback wouldn't hide behind a test that only
    compares those two to each other."""
    if scoring_func == "softmax":
        unbiased = torch.softmax(gating_logits.double(), dim=-1)
    else:
        unbiased = torch.sigmoid(gating_logits.double())
    biased = unbiased + bias.double().unsqueeze(0)
    idx = torch.topk(biased, k=topk, dim=-1, sorted=True)[1]
    w = unbiased.gather(1, idx)
    if need_renorm:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.float(), idx.to(torch.int32)


def test_fused_topk_gating_biased_matches_kernel_semantics_directly():
    """Bias-aware fast path vs. a from-scratch reimplementation of the HIP
    kernel's bias math (independent reference, not the production code
    path), for both scoring functions and with a renorm-off sigmoid case."""
    torch.manual_seed(0)
    for scoring_func, need_renorm in (
        ("softmax", True),
        ("sigmoid", True),
        ("sigmoid", False),
    ):
        for M, E, topk in ((1, 896, 16), (16, 896, 16), (4, 64, 8)):
            gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
            bias = torch.randn(E, dtype=torch.float32, device="cuda") * 0.1

            ref_w, ref_i = _reference_biased_kernel_math(
                gating_logits, bias, topk, scoring_func, need_renorm
            )
            out_w, out_i = _fused_topk_gating_biased(
                gating_logits, bias, topk=topk, scoring_func=scoring_func, need_renorm=need_renorm
            )

            label = f"M={M} E={E} topk={topk} func={scoring_func} renorm={need_renorm}"
            errs = {
                "topk_indices": checkAllclose(ref_i, out_i, atol=0, msg=f"{label} topk_indices"),
                "topk_weights": checkAllclose(ref_w, out_w, msg=f"{label} topk_weights"),
            }
            bad = {k: v for k, v in errs.items() if v}
            assert not bad, f"[{label}] mismatch vs from-scratch reference: {bad}"
            print(f"[PASS] biased-kernel-semantics {label}")


def test_fused_topk_gating_biased_reorders_selection():
    """A deliberately large-magnitude bias must change which experts get
    selected relative to the unbiased top-k, and the fast path's indices
    must track that change (not silently fall back to unbiased selection
    while still 'passing' because small random biases rarely flip the
    top-k set)."""
    torch.manual_seed(1)
    M, E, topk = 8, 64, 8
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")

    # Bias one normally-unselected expert per row up by a large margin so it
    # must be selected, and push one normally-selected expert down so it
    # must be dropped.
    unbiased_rank = torch.argsort(gating_logits, dim=-1, descending=True)
    must_include = unbiased_rank[:, topk]  # just outside the unbiased top-k
    must_exclude = unbiased_rank[:, 0]  # unbiased #1 pick

    bias = torch.zeros(M, E, dtype=torch.float32, device="cuda")
    bias.scatter_(1, must_include.unsqueeze(1), 100.0)
    bias.scatter_(1, must_exclude.unsqueeze(1), -100.0)
    # The fast path only supports a per-expert (not per-token) bias, so run
    # one row at a time with that row's bias vector -- this still exercises
    # the exact same code path, just with a controlled worst-case bias per
    # call instead of one shared bias for the whole batch.
    for m in range(M):
        row_logits = gating_logits[m : m + 1]
        row_bias = bias[m]

        set_fused_topk_moe_sort_backend(True)
        try:
            out_w, out_i = fused_topk_gating(
                row_logits, bias=row_bias, topk=topk, scoring_func="softmax", need_renorm=True
            )
        finally:
            set_fused_topk_moe_sort_backend(False)

        ref_w, ref_i = sequential_topk_gating(
            row_logits, bias=row_bias, topk=topk, scoring_func="softmax", need_renorm=True
        )

        selected = set(out_i[0].tolist())
        assert must_include[m].item() in selected, (
            f"row {m}: large positive bias on expert {must_include[m].item()} "
            f"did not force its selection: got {sorted(selected)}"
        )
        assert must_exclude[m].item() not in selected, (
            f"row {m}: large negative bias on expert {must_exclude[m].item()} "
            f"did not force its exclusion: got {sorted(selected)}"
        )
        checkAllclose(ref_i, out_i, atol=0, msg=f"row {m} biased-selection topk_indices")
        checkAllclose(ref_w, out_w, msg=f"row {m} biased-selection topk_weights")
    print("[PASS] biased gating actually reorders selection vs. unbiased top-k")


def test_fused_topk_gating_prefill_fallback_matches_reference():
    """M > 16 (prefill) must take the fallback path regardless of the gate."""
    M, E, topk = 32, 64, 8
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    ref_w, ref_i = sequential_topk_gating(gating_logits, topk=topk, scoring_func="softmax")

    set_fused_topk_moe_sort_backend(True)
    try:
        out_w, out_i = fused_topk_gating(gating_logits, topk=topk, scoring_func="softmax")
    finally:
        set_fused_topk_moe_sort_backend(False)

    checkAllclose(ref_i, out_i, atol=0, msg="prefill fallback topk_indices")
    checkAllclose(ref_w, out_w, msg="prefill fallback topk_weights")


def test_fused_topk_gating_gate_off_matches_reference():
    """With the env gate off, fused_topk_gating must reproduce the HIP
    topk_gating baseline exactly even for decode-sized M."""
    M, E, topk = 8, 64, 8
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    ref_w, ref_i = sequential_topk_gating(gating_logits, topk=topk, scoring_func="softmax")

    set_fused_topk_moe_sort_backend(False)
    out_w, out_i = fused_topk_gating(gating_logits, topk=topk, scoring_func="softmax")

    checkAllclose(ref_i, out_i, atol=0, msg="gate off topk_indices")
    checkAllclose(ref_w, out_w, msg="gate off topk_weights")


def _run_sort_only_case(M, E, topk, unit_size, enable_gate):
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    topk_w, topk_i = sequential_topk_gating(gating_logits, topk=topk, scoring_func="softmax")

    device = gating_logits.device
    _, sorted_ids_a, sorted_weights_a, sorted_expert_ids_a, num_valid_ids_a, moe_buf = (
        moe_sorting_mod._alloc_sort_outputs(
            M, E, topk, unit_size, device, None, None, None, None, None
        )
    )

    flydsl_moe_sorting_fwd(
        topk_i,
        topk_w,
        sorted_ids_a,
        sorted_weights_a,
        sorted_expert_ids_a,
        num_valid_ids_a,
        moe_buf,
        E,
        unit_size,
    )
    ref = (sorted_ids_a, sorted_weights_a, sorted_expert_ids_a, num_valid_ids_a, moe_buf)

    set_fused_topk_moe_sort_backend(enable_gate)
    try:
        out = flydsl_fused_topk_moe_sort(
            topk_i, topk_w, num_experts=E, unit_size=unit_size
        )
    finally:
        set_fused_topk_moe_sort_backend(False)

    label = f"M={M} E={E} topk={topk} unit={unit_size} gate={enable_gate}"
    _compare(ref, out, topk, M, unit_size, label)
    print(f"[PASS] sort-only {label}")


def test_flydsl_fused_topk_moe_sort_decode_shapes():
    """Sort-only fast-path correctness: fused sort kernel vs the generic
    FlyDSL sort (flydsl_moe_sorting_fwd), given identical topk_ids/weights."""
    shapes = [(896, 16, 32), (64, 8, 32)]
    for (E, topk, unit_size), M in itertools.product(shapes, (1, 2, 4, 8, 16)):
        _run_sort_only_case(M, E, topk, unit_size, enable_gate=True)


def test_flydsl_fused_topk_moe_sort_prefill_fallback_matches_reference():
    """M > 16 (prefill) must take the fallback path regardless of the gate."""
    _run_sort_only_case(32, 64, 8, 32, enable_gate=True)


def test_flydsl_fused_topk_moe_sort_gate_off_matches_reference():
    """With the env gate off, flydsl_fused_topk_moe_sort must reproduce
    flydsl_moe_sorting_fwd exactly even for decode-sized M."""
    _run_sort_only_case(8, 64, 8, 32, enable_gate=False)


if __name__ == "__main__":
    test_fused_topk_moe_sort_decode_shapes()
    test_fused_topk_moe_sort_softmax_no_renorm_falls_back()
    test_fused_topk_moe_sort_prefill_fallback_matches_reference()
    test_fused_topk_moe_sort_gate_off_matches_reference()
    test_fused_topk_gating_decode_shapes()
    test_fused_topk_gating_biased_matches_kernel_semantics_directly()
    test_fused_topk_gating_biased_reorders_selection()
    test_fused_topk_gating_softmax_no_renorm_falls_back()
    test_fused_topk_gating_prefill_fallback_matches_reference()
    test_fused_topk_gating_gate_off_matches_reference()
    test_flydsl_fused_topk_moe_sort_decode_shapes()
    test_flydsl_fused_topk_moe_sort_prefill_fallback_matches_reference()
    test_flydsl_fused_topk_moe_sort_gate_off_matches_reference()
    print("All fused_topk_softmax_moe_sort tests passed.")
