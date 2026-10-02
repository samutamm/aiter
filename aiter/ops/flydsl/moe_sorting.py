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

# Gate for fused_topk_softmax_moe_sort's fast path (decode-only, M<=16,
# correction bias supported). Off by default; flip on once validated on
# target hardware.
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


def _fused_topk_gating_biased(gating_output, bias, *, topk, scoring_func, need_renorm):
    """Bias-aware fast-path gating, matching topk_gating_kernels.cuh's HIP
    kernel semantics exactly: bias shifts selection only, not the reported
    weight.

        unbiased_score = softmax(logits) or sigmoid(logits)   # full E
        biased_score   = unbiased_score + bias
        indices        = topk(biased_score)
        weight         = gather(unbiased_score, indices)
        weight        /= weight.sum(-1) if need_renorm else weight

    Requires a full-E softmax/sigmoid up front (unlike the no-bias fast
    path) since bias can change the ranking across all E experts, not just
    perturb the already-selected top-k.
    """
    bias_row = bias.to(dtype=torch.float32).unsqueeze(0)
    if scoring_func == "softmax":
        unbiased_score = torch.softmax(gating_output.float(), dim=-1)
    else:  # scoring_func == "sigmoid"
        unbiased_score = torch.sigmoid(gating_output.float())

    biased_score = unbiased_score + bias_row
    topk_indices = torch.topk(biased_score, k=topk, dim=-1, sorted=True)[1].to(
        torch.int32
    )
    topk_w = unbiased_score.gather(1, topk_indices.to(torch.int64))
    if need_renorm:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    return topk_w, topk_indices


def fused_topk_gating(
    gating_output,
    bias=None,
    topk=8,
    scoring_func="softmax",
    need_renorm=True,
):
    """Top-k gating: selects experts and their weights for each token.

    Returns (topk_weights, topk_indices), shape [M, topk] each.

    Fast path (M<=16, scoring_func in {softmax-with-renorm, sigmoid}):
    in-PyTorch top-k + softmax/sigmoid, with two shapes depending on whether a
    correction bias is supplied:

    - No bias: selective top-k on the raw logits/sigmoid scores, normalizing
      only the selected experts. This is exact, not approximate:
      softmax(selected_z) only needs the selected logits because the excluded
      experts' exp() terms cancel out of both numerator and denominator once
      normalized over just the top-k set; sigmoid weights are independent per
      expert, so renormalizing over the top-k subset is exact by
      construction.
    - With bias: matches aiter.ops.topk.topk_gating's HIP kernel semantics
      (see csrc/include/topk_gating_kernels.cuh) -- bias only shifts which
      experts get *selected*, not the reported weight. Concretely: compute
      the unbiased score over all E experts (softmax(logits) or
      sigmoid(logits)), add bias to get the selection score, top-k on the
      biased score, then gather the *unbiased* score at the selected indices
      as the weight (renormalizing over just those K values if need_renorm).
      This requires a full-E softmax/sigmoid (bias can reorder the full
      ranking, not just perturb the already-selected top-k), so it doesn't
      get the no-bias path's "only touch the K winners" compute saving --
      the benefit here is purely fewer kernel launches into the fused sort,
      not less overall math.

    Falls back to aiter.ops.topk.topk_gating (HIP) when: the env-var gate is
    off, M > 16 (prefill, where the fast path offers no benefit), or
    scoring_func is "softmax" without renormalization (the un-renormalized
    weight needs the full set of experts, not just the selected top-k, even
    in the bias-free case).

    Args:
        gating_output: [M, E] float32 router logits.
        bias: optional [E] correction bias.
        topk: experts routed per token.
        scoring_func: "softmax" or "sigmoid" (anything else routes to the
            fallback path, which also accepts "sqrtsoftplus" via topk_gating).
        need_renorm: renormalize the top-k weights to sum to 1.

    Returns:
        (topk_weights, topk_indices), each [M, topk].
    """
    assert gating_output.dim() == 2, (
        f"gating_output must be 2D [M, E], got shape {tuple(gating_output.shape)}"
    )
    M, E = gating_output.shape
    device = gating_output.device
    has_bias = bias is not None and bias.numel() > 0
    scoring_supported = scoring_func == "sigmoid" or (
        scoring_func == "softmax" and need_renorm
    )
    use_fast_path = _USE_FUSED_TOPK_MOE_SORT and M <= 16 and scoring_supported

    if not use_fast_path:
        topk_w = torch.empty(M, topk, dtype=torch.float32, device=device)
        topk_i = torch.empty(M, topk, dtype=torch.int32, device=device)

        from ..topk import topk_gating

        topk_gating(
            topk_w,
            topk_i,
            gating_output,
            bias,
            need_renorm=need_renorm,
            score_func=scoring_func,
        )
        return topk_w, topk_i

    if has_bias:
        return _fused_topk_gating_biased(
            gating_output, bias, topk=topk, scoring_func=scoring_func, need_renorm=need_renorm
        )

    # --- Fast path: selective top-k + in-PyTorch softmax/sigmoid renorm ---
    if scoring_func == "softmax":
        topk_indices = torch.topk(gating_output, k=topk, dim=-1, sorted=True)[1].to(
            torch.int32
        )
        selected_z = gating_output.gather(1, topk_indices.to(torch.int64))
        topk_w = torch.softmax(selected_z, dim=-1)
    else:  # scoring_func == "sigmoid"
        choice_scores = torch.sigmoid(gating_output)
        topk_indices = torch.topk(choice_scores, k=topk, dim=-1, sorted=True)[1].to(
            torch.int32
        )
        selected_z = gating_output.gather(1, topk_indices.to(torch.int64))
        raw_w = torch.sigmoid(selected_z)
        topk_w = raw_w / raw_w.sum(dim=-1, keepdim=True) if need_renorm else raw_w

    return topk_w, topk_indices


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
    """Sort-only fast path: same inputs/outputs as flydsl_moe_sorting_fwd, but
    uses the single-block compact-binning kernel
    (kernels/fused_topk_moe_sort_kernel.py) instead of the generic
    multi-phase/oneshot sort.

    Gate: the env-var gate is on and M = topk_ids.shape[0] <= 16. Falls back
    to flydsl_moe_sorting_fwd otherwise.

    Args:
        topk_ids: [M, topk] int32 selected expert indices.
        topk_weights: [M, topk] float32 selected expert weights.
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf:
            optional pre-allocated output buffers (see flydsl_moe_sorting_fwd).
        num_experts: total number of experts E.
        unit_size: GEMM tile-M for padding alignment.
        stream: optional torch.cuda.Stream for kernel launch.

    Returns:
        (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    """
    M, topk = topk_ids.shape
    E = num_experts
    device = topk_ids.device

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

    if not (_USE_FUSED_TOPK_MOE_SORT and M <= 16):
        flydsl_moe_sorting_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            E,
            unit_size,
        )
        return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf

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
        topk_ids.flatten(),
        topk_weights.flatten(),
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
    """Top-k gating + MoE token sorting, composed from the standalone
    :func:`fused_topk_gating` and :func:`flydsl_fused_topk_moe_sort` pieces.

    Kept around for tests/benchmarks and for callers that want gating+sort in
    one call; the dispatch call sites (vLLM's router, aiter's
    fused_moe.py::moe_sorting()) call the two pieces independently instead,
    since that's how a real MoE forward pass is structured (gating at the
    router, sorting later at expert-dispatch, with other work such as
    expert-parallel masking potentially happening in between).

    Args:
        gating_logits: [M, E] float32 router logits.
        bias: optional [E] correction bias (see fused_topk_gating for how
            the fast path handles it).
        topk: experts routed per token.
        unit_size: GEMM tile-M for padding alignment.
        scoring_func: "softmax" or "sigmoid" (anything else routes gating to
            the fallback path, which also accepts "sqrtsoftplus" via
            topk_gating).
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
    E = gating_logits.shape[1]

    topk_w, topk_i = fused_topk_gating(
        gating_logits,
        bias=bias,
        topk=topk,
        scoring_func=scoring_func,
        need_renorm=need_renorm,
    )
    return flydsl_fused_topk_moe_sort(
        topk_i,
        topk_w,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        moe_buf=moe_buf,
        num_experts=E,
        unit_size=unit_size,
        stream=stream,
    )
