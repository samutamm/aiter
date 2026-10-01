# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MoE token sorting kernel for pre-computed top-k gating (FlyDSL).

Companion to ``moe_sorting_kernel.py``'s generic multi-phase/oneshot mesh-sort,
specialized for the small-batch decode regime (M small relative to unit_size,
one 32-token-wide tile per active expert is always enough). Given
(topk_ids, topk_weights) already selected on the host side (see
``aiter.ops.flydsl.moe_sorting.fused_topk_softmax_moe_sort``), this performs
the counting-sort entirely in a single block's LDS:

  1. Count tokens routed to each expert (atomic histogram in LDS).
  2. Chunked prefix-sum over experts to compactly assign one tile per *active*
     expert (inactive experts get no tile at all -- no wasted GEMM blocks).
  3. Pre-fill every allocated tile with the sentinel packed id.
  4. Scatter each token's (topk-slot, weight) pair into its expert's tile.

Packed token ID format: (topk_position << 24) | token_id (same convention as
moe_sorting_kernel.py). Padding sentinel: (topk << 24) | M.

Parametrized over (num_experts, topk, unit_size) via an lru_cache-keyed
compiler, mirroring ``moe_sorting_kernel.py``'s ``_compile_moe_sorting_oneshot``
-- the kernel is JIT-recompiled (and cached) per distinct shape rather than
hardcoding one model's expert count.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from .kernels_common import atomic_add_i32, get_warp_size
from .moe_sorting_kernel import _allwave_inclusive_prefix_sum

BLOCK_SIZE = 256


@functools.lru_cache(maxsize=64)
def _compile_fused_topk_moe_sort(
    *,
    num_experts: int,
    topk: int,
    unit_size: int,
    block_size: int = BLOCK_SIZE,
):
    """Compile+cache the fused top-k sort launcher for one (E, topk, unit_size) shape."""
    E = num_experts
    BS = block_size
    WARP_SIZE = get_warp_size()
    NUM_WAVES = BS // WARP_SIZE

    @fx.struct
    class FusedBinningLDS:
        counts: fx.Array[fx.Int32, E, 16]
        tile_offsets: fx.Array[fx.Int32, E, 16]
        active_count: fx.Array[fx.Int32, 1, 16]
        scratch: fx.Array[fx.Int32, NUM_WAVES, 16]

    @flyc.kernel(known_block_size=[BS, 1, 1])
    def _compact_binning_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        num_tokens: fx.Int32,
        max_blocks: fx.Int32,
        sentinel: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(WARP_SIZE)
        wave = tid // fx.Int32(WARP_SIZE)

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_E = fx.Int32(E)
        c_topk = fx.Int32(topk)
        c_unit = fx.Int32(unit_size)

        storage = fx.SharedAllocator().allocate(FusedBinningLDS).peek()
        counts = storage.counts.view(fx.make_layout(E, 1))
        tile_offsets = storage.tile_offsets.view(fx.make_layout(E, 1))
        ac = storage.active_count.view(fx.make_layout(1, 1))
        scratch_mr = storage.scratch.ptr

        # 1. Clear LDS counts
        for i in range_constexpr(0, E, BS):
            idx = fx.Int32(i) + tid
            if idx < c_E:
                counts[idx] = c_zero
        if tid == c_zero:
            ac[c_zero] = c_zero
        gpu.barrier()

        # 2. Count tokens per expert
        total_pairs = num_tokens * c_topk
        if tid < total_pairs:
            eid = topk_ids[tid]
            if (eid >= c_zero) & (eid < c_E):
                atomic_add_i32(counts, c_one, eid, "workgroup")
        gpu.barrier()

        # 3. Chunked prefix scan across experts: assign one compact tile to
        # each *active* expert (count > 0), in ascending expert-id order.
        running_active = c_zero
        for chunk_base in range_constexpr(0, E, BS):
            idx = fx.Int32(chunk_base) + tid
            is_valid = idx < c_E
            safe_idx = is_valid.select(idx, c_zero)
            c_val = is_valid.select(counts[safe_idx], c_zero)
            is_active = is_valid & (c_val > c_zero)
            flag = is_active.select(c_one, c_zero)

            _, inclusive = _allwave_inclusive_prefix_sum(
                flag, lane, wave, scratch_mr, NUM_WAVES, WARP_SIZE
            )

            if is_active:
                my_tile = running_active + inclusive - c_one
                tile_offsets[idx] = my_tile
                sorted_expert_ids[my_tile] = idx

            if tid == fx.Int32(BS - 1):
                storage.scratch[c_zero] = inclusive
            gpu.barrier()
            chunk_total = storage.scratch[c_zero]
            running_active = running_active + chunk_total
            gpu.barrier()

        if tid == c_zero:
            ac[c_zero] = running_active
            num_valid[c_zero] = running_active * c_unit
            num_valid[c_one] = num_tokens
        gpu.barrier()

        # Reuse counts[] as per-expert scatter cursors.
        for i in range_constexpr(0, E, BS):
            idx = fx.Int32(i) + tid
            if idx < c_E:
                counts[idx] = c_zero
        gpu.barrier()

        total_active = ac[c_zero]

        # Zero remaining sorted_expert_ids beyond the active tiles.
        for i in range(tid, max_blocks - total_active, fx.Int32(BS)):
            sorted_expert_ids[total_active + i] = c_zero

        # 4. Pre-fill sentinel into all active tiles.
        total_slots = total_active * c_unit
        c_zero_f32 = fx.Float32(0.0)
        for s in range(tid, total_slots, fx.Int32(BS)):
            sorted_ids[s] = sentinel
            sorted_weights[s] = c_zero_f32
        gpu.barrier()

        # 5. Scatter valid tokens into allocated tiles, sequentially by token
        # index m, so tokens sharing an expert land in ascending token order.
        for m in range(fx.Int32(0), num_tokens, fx.Int32(1)):
            if tid < c_topk:
                pair_idx = m * c_topk + tid
                eid = topk_ids[pair_idx]
                if (eid >= c_zero) & (eid < c_E):
                    slot = atomic_add_i32(counts, c_one, eid, "workgroup")
                    tile = tile_offsets[eid]
                    out_idx = tile * c_unit + slot
                    packed_id = (tid << fx.Int32(24)) | m
                    sorted_ids[out_idx] = packed_id
                    sorted_weights[out_idx] = topk_weights[pair_idx]
            gpu.barrier()

    @flyc.jit
    def _launch(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        num_tokens: fx.Int32,
        max_blocks: fx.Int32,
        sentinel: fx.Int32,
        stream: fx.Stream,
    ):
        _compact_binning_kernel(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid,
            num_tokens,
            max_blocks,
            sentinel,
        ).launch(grid=(1, 1, 1), block=(BS, 1, 1), stream=stream)

    return _launch
