#!/usr/bin/env python3
"""Warm-start V3 v10 V2 (5 TFs × 25 multi-TF feats) state_dict from V3 v9 (5 TFs × 17).

V3 v9 baseline lives at:
  models/exit_transformer_v0/EXIT_V9_MULTI_TF_LR5E4_SCALE025_*

Differences V3 v9 → V3 v10 V2:
  - Multi-TF per-TF dim: 17 → 25 (V1 → V2 feature set)
    Per-TF *_proj.weight grows (128, 17) → (128, 25). Pad zeros on new dims.
  - Multi-TF fuse: 5 TFs × 128 = 640 in/out, same. Transfer clean.
  - Base transformer (M1L512 input_dim=91): same on V7 dataset. Transfer clean.
  - All encoder layers: same shape. Transfer clean.
  - All heads (main / profit_protect / family): same shape. Transfer clean.

Net effect: V2 model starts ~equivalent to V3 v9 (new features = 0), then
gradually learns to use the 8 new V2 features per TF.

Usage:
    python -m gx1.scripts.warm_start_v3_v10_v2_from_v9 \\
        --v9-bundle /path/to/EXIT_V9_MULTI_TF_LR5E4_SCALE025_* \\
        --v10-out   /path/to/V3_V10_V2_WARMSTART_FROM_V9.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch


PER_TF_PROJ_PREFIXES = ("m5_proj", "m15_proj", "h1_proj", "h4_proj", "d1_proj")


def _pad_zeros_in_dim(v9_weight: torch.Tensor, target_in_dim: int) -> torch.Tensor:
    """Pad weight (out_dim, in_dim) with zeros to reach target_in_dim columns."""
    if v9_weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(v9_weight.shape)}")
    out_dim, cur_in_dim = v9_weight.shape
    if cur_in_dim >= target_in_dim:
        return v9_weight
    pad = torch.zeros((out_dim, target_in_dim - cur_in_dim), dtype=v9_weight.dtype)
    return torch.cat([v9_weight, pad], dim=1)


def warm_start_state_dict(
    v9_state: Dict[str, torch.Tensor],
    v10_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    transferred = 0
    padded_per_tf = 0
    kept_v10_init = 0

    v9_keys = set(v9_state.keys())
    v10_keys = set(v10_state.keys())

    for k in v10_keys:
        v10_val = v10_state[k]
        if k not in v9_keys:
            out[k] = v10_val
            kept_v10_init += 1
            continue
        v9_val = v9_state[k]
        if v9_val.shape == v10_val.shape:
            out[k] = v9_val
            transferred += 1
            continue
        # Per-TF *_proj.weight: input dim grows 17 → 25
        is_per_tf_proj_w = any(
            k.startswith(prefix) and k.endswith(".weight")
            for prefix in PER_TF_PROJ_PREFIXES
        )
        if is_per_tf_proj_w and v10_val.dim() == 2 and v9_val.dim() == 2 \
                and v10_val.shape[0] == v9_val.shape[0]:
            out[k] = _pad_zeros_in_dim(v9_val, target_in_dim=v10_val.shape[1])
            padded_per_tf += 1
            continue
        print(f"  [WARN] keep V10 init for '{k}': V9 shape {tuple(v9_val.shape)} "
              f"vs V10 shape {tuple(v10_val.shape)}", file=sys.stderr)
        out[k] = v10_val
        kept_v10_init += 1

    skipped_v9_only = len(v9_keys - v10_keys)
    print(f"\n[WARM_START_SUMMARY]")
    print(f"  transferred clean: {transferred}")
    print(f"  padded per-TF proj: {padded_per_tf}")
    print(f"  kept V10 init (new/unknown mismatch): {kept_v10_init}")
    print(f"  skipped V9-only keys: {skipped_v9_only}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v9-bundle", type=Path, required=True,
                   help="Path to V3 v9 bundle dir (must contain exit_transformer_v0.pt)")
    p.add_argument("--v10-out", type=Path, required=True,
                   help="Output path for warm-started V10 state_dict (.pt)")
    p.add_argument("--input-dim", type=int, default=91,
                   help="V3 input dim (default 91 = EXIT_IO_V6_CTX_V3CANONICAL_M1L512)")
    p.add_argument("--per-tf-dim", type=int, default=25,
                   help="V2 per-TF feature dim (default 25 = MULTI_TF_PER_BAR_FEATURES_V2)")
    p.add_argument("--multi-tf-seq-len", type=int, default=96)
    p.add_argument("--multi-tf-scale", type=float, default=0.25,
                   help="V3 v9 baseline used 0.25")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--window-len", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args()

    v9_state_path = args.v9_bundle / "exit_transformer_v0.pt"
    if not v9_state_path.is_file():
        raise FileNotFoundError(f"V9 state_dict not found: {v9_state_path}")
    print(f"[LOAD] V9 state_dict: {v9_state_path}")
    v9_state = torch.load(v9_state_path, map_location="cpu", weights_only=True)
    # Strip _orig_mod. if torch.compile-wrapped
    v9_state = {k.removeprefix("_orig_mod."): v for k, v in v9_state.items()}
    print(f"  → {len(v9_state)} keys")

    sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
    print("[BUILD] reference V10 V2 model for shape inference")
    from gx1.policy.exit_transformer_v0 import ExitTransformerV0
    ref_model = ExitTransformerV0(
        input_dim=int(args.input_dim),
        window_len=int(args.window_len),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        dropout=float(args.dropout),
        enable_multi_tf=True,
        m5_seq_dim=int(args.per_tf_dim), m15_seq_dim=int(args.per_tf_dim),
        h1_seq_dim=int(args.per_tf_dim), h4_seq_dim=int(args.per_tf_dim),
        d1_seq_dim=int(args.per_tf_dim),
        m5_seq_len=int(args.multi_tf_seq_len), m15_seq_len=int(args.multi_tf_seq_len),
        h1_seq_len=int(args.multi_tf_seq_len), h4_seq_len=int(args.multi_tf_seq_len),
        d1_seq_len=int(args.multi_tf_seq_len),
        multi_tf_scale=float(args.multi_tf_scale),
    )
    v10_state = {k: v.clone() for k, v in ref_model.state_dict().items()}
    print(f"  → {len(v10_state)} keys, {sum(p.numel() for p in ref_model.parameters())/1e6:.2f}M params")

    print("[WARM_START] transferring weights...")
    out_state = warm_start_state_dict(v9_state, v10_state)

    args.v10_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, args.v10_out)
    print(f"\n[SAVE] {args.v10_out} ({args.v10_out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
