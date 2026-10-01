# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MoE sorting kernel — drop-in replacement for CK/Opus moe_sorting_fwd.

Provides `flydsl_moe_sorting_fwd()` with the same signature as
`aiter.moe_sorting_fwd()` so it can be used as a direct dispatch target
in `_moe_sorting_impl()`.

Workspace is pre-allocated here (not inside the kernel) so that CUDA graph
capture sees deterministic allocations.
"""

import os

import torch

_workspace_cache = {}

# Gate for fused_topk_softmax_moe_sort's fast path (decode-only, M<=16, no
# correction bias). Off by default; flip on once validated on target hardware.
_USE_FUSED_TOPK_MOE_SORT = os.environ.get("AITER_USE_FUSED_TOPK_MOE_SORT", "0") == "1"


def flydsl_moe_sorting_fwd(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    num_experts,
    unit_size,
    expert_mask=None,
    num_local_tokens=None,
):
    from .kernels.moe_sorting_kernel import (
        moe_sorting_flydsl,
        moe_sorting_get_workspace_size,
    )

    M = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device

    # Pre-allocate workspace (cached per device for CUDA graph compatibility).
    # A larger workspace can satisfy smaller requests, so we keep the largest seen.
    ws_size = moe_sorting_get_workspace_size(M, num_experts, topk, unit_size)
    workspace = None
    if ws_size > 0:
        workspace = _workspace_cache.get(device)
        if workspace is None or workspace.numel() < ws_size:
            workspace = torch.empty(ws_size, dtype=torch.int32, device=device)
            _workspace_cache[device] = workspace

    moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        unit_size,
        expert_mask,
        num_local_tokens,
        workspace,
    )


def _alloc_sort_outputs(
    M,
    E,
    topk,
    unit_size,
    device,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
):
    """Allocate any output buffers the caller didn't pre-provide."""
    max_tokens_padded = int(M * topk + E * unit_size - topk)
    max_blocks = int((max_tokens_padded + unit_size - 1) // unit_size)
    max_tokens_padded = max_blocks * unit_size

    if sorted_ids is None:
        sorted_ids = torch.empty(max_tokens_padded, dtype=torch.int32, device=device)
    if sorted_weights is None:
        sorted_weights = torch.empty(
            max_tokens_padded, dtype=torch.float32, device=device
        )
    if sorted_expert_ids is None:
        sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    if num_valid_ids is None:
        num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if moe_buf is None:
        moe_buf = torch.empty((0, 0), dtype=torch.bfloat16, device=device)

    return (
        max_blocks,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
    )


def sequential_topk_softmax_moe_sort(
    gating_logits,
    bias=None,
    topk=8,
    unit_size=32,
    scoring_func="softmax",
    need_renorm=True,
    sorted_ids=None,
    sorted_weights=None,
    sorted_expert_ids=None,
    num_valid_ids=None,
    moe_buf=None,
):
    """Sequential topk_gating + flydsl_moe_sorting_fwd.

    This is what fused_topk_softmax_moe_sort's fallback path calls; it's
    exposed directly so tests/benchmarks can use the exact same code as the
    reference instead of re-implementing it.
    """
    M, E = gating_logits.shape
    device = gating_logits.device
    topk_w = torch.empty(M, topk, dtype=torch.float32, device=device)
    topk_i = torch.empty(M, topk, dtype=torch.int32, device=device)

    from ..topk import topk_gating

    topk_gating(
        topk_w,
        topk_i,
        gating_logits,
        bias,
        need_renorm=need_renorm,
        score_func=scoring_func,
    )

    _, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        _alloc_sort_outputs(
            M,
            E,
            topk,
            unit_size,
            device,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
        )
    )

    flydsl_moe_sorting_fwd(
        topk_i,
        topk_w,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        E,
        unit_size,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def fused_topk_softmax_moe_sort(
    gating_logits,
    bias=None,
    topk=8,
    unit_size=32,
    scoring_func="softmax",
    need_renorm=True,
    sorted_ids=None,
    sorted_weights=None,
    sorted_expert_ids=None,
    num_valid_ids=None,
    moe_buf=None,
    stream=None,
):
    """Top-k gating + MoE token sorting, fused into one fast path for decode.

    For small batches (M <= 16), this picks the top-k experts and normalizes
    their weights in PyTorch, then sorts tokens by expert with one FlyDSL
    kernel call -- skipping the usual round-trip of topk_ids/topk_weights
    through a separate sort kernel.

    It falls back to the normal sequential path
    (:func:`sequential_topk_softmax_moe_sort`) when: the env-var gate is off,
    M > 16 (prefill, since the sort kernel assumes each expert fits in one
    tile), a correction bias is supplied (bias can change which experts get
    picked, so this isn't supported in the fast path yet), or scoring_func is
    "softmax" without renormalization (the un-renormalized weight needs the
    full set of experts, not just the selected top-k).

    Args:
        gating_logits: [M, E] float32 router logits.
        bias: optional [E] correction bias (routes to the fallback path).
        topk: experts routed per token.
        unit_size: GEMM tile-M for padding alignment.
        scoring_func: "softmax" or "sigmoid" (anything else routes to the
            fallback path, which also accepts "sqrtsoftplus" via topk_gating).
        need_renorm: renormalize the top-k weights to sum to 1.
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf:
            optional pre-allocated output buffers (see flydsl_moe_sorting_fwd).
        stream: optional torch.cuda.Stream for kernel launch.

    Returns:
        (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    """
    assert gating_logits.dim() == 2, (
        f"gating_logits must be 2D [M, E], got shape {tuple(gating_logits.shape)}"
    )
    M, E = gating_logits.shape
    device = gating_logits.device
    has_bias = bias is not None and bias.numel() > 0
    scoring_supported = scoring_func == "sigmoid" or (
        scoring_func == "softmax" and need_renorm
    )
    use_fast_path = (
        _USE_FUSED_TOPK_MOE_SORT and M <= 16 and not has_bias and scoring_supported
    )

    if not use_fast_path:
        return sequential_topk_softmax_moe_sort(
            gating_logits,
            bias=bias,
            topk=topk,
            unit_size=unit_size,
            scoring_func=scoring_func,
            need_renorm=need_renorm,
            sorted_ids=sorted_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            moe_buf=moe_buf,
        )

    # --- Fast path: selective top-k + in-PyTorch softmax/sigmoid renorm ---
    if scoring_func == "softmax":
        topk_indices = torch.topk(gating_logits, k=topk, dim=-1, sorted=True)[1].to(
            torch.int32
        )
        selected_z = gating_logits.gather(1, topk_indices.to(torch.int64))
        topk_w = torch.softmax(selected_z, dim=-1)
    else:  # scoring_func == "sigmoid"
        choice_scores = torch.sigmoid(gating_logits)
        topk_indices = torch.topk(choice_scores, k=topk, dim=-1, sorted=True)[1].to(
            torch.int32
        )
        selected_z = gating_logits.gather(1, topk_indices.to(torch.int64))
        raw_w = torch.sigmoid(selected_z)
        topk_w = raw_w / raw_w.sum(dim=-1, keepdim=True) if need_renorm else raw_w

    max_blocks, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        _alloc_sort_outputs(
            M,
            E,
            topk,
            unit_size,
            device,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
        )
    )
    if moe_buf.numel() > 0:
        moe_buf.zero_()

    sentinel = (topk << 24) | M

    import flydsl.expr as fx

    from .kernels.fused_topk_moe_sort_kernel import _compile_fused_topk_moe_sort

    cu_stream = stream if stream is not None else torch.cuda.current_stream(device)
    fly_stream = fx.Stream(cu_stream)

    launcher = _compile_fused_topk_moe_sort(
        num_experts=E, topk=topk, unit_size=unit_size
    )
    launcher(
        topk_indices.flatten(),
        topk_w.flatten(),
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        int(M),
        int(max_blocks),
        int(sentinel),
        fly_stream,
    )

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf
