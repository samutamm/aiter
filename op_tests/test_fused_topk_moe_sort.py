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
    fused_topk_softmax_moe_sort,
    sequential_topk_softmax_moe_sort,
)
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")


def set_fused_topk_moe_sort_backend(enabled: bool) -> None:
    """Force the fused_topk_softmax_moe_sort fast-path gate for a test."""
    moe_sorting_mod._USE_FUSED_TOPK_MOE_SORT = enabled


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


if __name__ == "__main__":
    test_fused_topk_moe_sort_decode_shapes()
    test_fused_topk_moe_sort_softmax_no_renorm_falls_back()
    test_fused_topk_moe_sort_prefill_fallback_matches_reference()
    test_fused_topk_moe_sort_gate_off_matches_reference()
    print("All fused_topk_softmax_moe_sort tests passed.")
