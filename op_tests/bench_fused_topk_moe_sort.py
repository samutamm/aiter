# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Microbenchmark: fused_topk_softmax_moe_sort fast path vs the sequential
aiter.ops.topk.topk_gating + flydsl_moe_sorting_fwd baseline.

Two parts:
  1. Fused-vs-sequential comparison at decode batch sizes M in {1,2,4,8,16}
     across several real MoE model shapes (all use the fast path).
  2. A fallback-path sanity check at M in {32,64,128} for the Kimi-K3 shape:
     the fast path's M<=16 gate exists because the kernel assigns exactly one
     unit_size-wide tile per active expert, which only stays correct if no
     expert can receive >= unit_size tokens in one call -- guaranteed at
     M<=16 (worst case one expert gets all M tokens, still < 32), but not at
     M>=unit_size. So at M>16 fused_topk_softmax_moe_sort always takes the
     same sequential fallback as the baseline; this part just confirms that
     routing adds no measurable overhead, it is not a fused-kernel speed test.
"""

import pandas as pd
import torch

import aiter.ops.flydsl.moe_sorting as moe_sorting_mod
from aiter.ops.flydsl.moe_sorting import (
    fused_topk_softmax_moe_sort,
    sequential_topk_softmax_moe_sort,
)
from aiter.test_common import run_perftest

torch.set_default_device("cuda")

# (name, num_experts, topk, unit_size), all unit_size=32 as used in production.
MODEL_SHAPES = [
    ("Kimi-K3", 896, 16, 32),
    ("Kimi-K2", 384, 8, 32),
    ("DeepSeek-V3", 256, 8, 32),
    ("GLM-5", 256, 8, 32),
    ("Qwen3-235B", 128, 8, 32),
    ("MiniMax-M3", 128, 4, 32),
]
DECODE_MS = (1, 2, 4, 8, 16)
FALLBACK_MS = (32, 64, 128)
FALLBACK_SHAPE = ("Kimi-K3", 896, 16, 32)


def _sequential(gating_logits, topk, unit_size):
    return sequential_topk_softmax_moe_sort(
        gating_logits, topk=topk, unit_size=unit_size, scoring_func="softmax"
    )


def _fused(gating_logits, topk, unit_size):
    moe_sorting_mod._USE_FUSED_TOPK_MOE_SORT = True
    try:
        return fused_topk_softmax_moe_sort(
            gating_logits, topk=topk, unit_size=unit_size, scoring_func="softmax",
        )
    finally:
        moe_sorting_mod._USE_FUSED_TOPK_MOE_SORT = False


def _bench_row(name, E, topk, unit_size, M, mode):
    gating_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    _, seq_us = run_perftest(_sequential, gating_logits, topk, unit_size, num_warmup=5, num_iters=50)
    _, fused_us = run_perftest(_fused, gating_logits, topk, unit_size, num_warmup=5, num_iters=50)
    row = {
        "mode": mode,
        "model": name,
        "M": M,
        "E": E,
        "topk": topk,
        "unit_size": unit_size,
        "sequential us": seq_us,
        "fused us": fused_us,
        "ratio": seq_us / fused_us,
    }
    label = "speedup" if mode == "fused" else "ratio (expect ~1x, fallback both sides)"
    print(
        f"[{mode:8s}] {name:12s} M={M:3d} E={E:4d} topk={topk:3d}  "
        f"sequential={seq_us:8.2f}us  fused={fused_us:8.2f}us  {label}={row['ratio']:.3f}x"
    )
    return row


def main():
    rows = []

    print("--- fused fast path vs sequential (M<=16, real model shapes) ---")
    for name, E, topk, unit_size in MODEL_SHAPES:
        for M in DECODE_MS:
            rows.append(_bench_row(name, E, topk, unit_size, M, mode="fused"))

    print()
    print("--- fallback path sanity check (M>16, both sides take the same code path) ---")
    name, E, topk, unit_size = FALLBACK_SHAPE
    for M in FALLBACK_MS:
        rows.append(_bench_row(name, E, topk, unit_size, M, mode="fallback"))

    df = pd.DataFrame(rows)
    print()
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
