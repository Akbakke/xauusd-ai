#!/usr/bin/env python3
"""Warm-start V10 V2 (5 TFs × 25 feats) state_dict from V1 (4 TFs × 17 feats).

Strategy:
  - Most layers transfer cleanly (encoder, seq_proj, snap_proj, ctx_*).
  - Per-TF *_proj input dim grows 17 → 25: pad weight with zeros on new dims
    so the new features start as no-op and the model gradually learns them.
  - multi_tf_fuse input dim grows 4*d → 5*d: pad zeros for the new M5 slot.
  - m5_proj + m5_encoder are new: random init (default at model construction).

Net result: V2 model starts ~equivalent to V1 (M5 branch + new features = 0),
then converges faster than scratch-init since the V1-trained sub-modules are
already in a good loss basin.

Usage:
    python -m gx1.scripts.warm_start_v10_v2_from_v1 \\
        --v1-bundle /path/to/V1_BUNDLE \\
        --v2-out    /path/to/V2_WARMSTARTED.pt

The output is just a state_dict .pt file — load it via:
    state = torch.load("V2_WARMSTARTED.pt", map_location="cpu")
    v2_model.load_state_dict(state, strict=False)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch


# Layers whose key matches but input-dim changes from V1 (17) to V2 (25).
# Pad weight with zeros on the in_dim axis, copy bias verbatim.
PER_TF_PROJ_PREFIXES = ("m15_proj", "h1_proj", "h4_proj", "d1_proj")


def _pad_zeros_in_dim(v1_weight: torch.Tensor, target_in_dim: int) -> torch.Tensor:
    """Pad weight (out_dim, in_dim) with zeros to reach target_in_dim columns.

    e.g. V1 m15_proj.weight = (d_model, 17), target=25 → returns (d_model, 25)
    where last 8 cols are zero.
    """
    if v1_weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(v1_weight.shape)}")
    out_dim, cur_in_dim = v1_weight.shape
    if cur_in_dim >= target_in_dim:
        return v1_weight  # already big enough
    pad = torch.zeros((out_dim, target_in_dim - cur_in_dim), dtype=v1_weight.dtype)
    return torch.cat([v1_weight, pad], dim=1)


def warm_start_state_dict(
    v1_state: Dict[str, torch.Tensor],
    v2_state: Dict[str, torch.Tensor],
    *,
    new_per_tf_dim: int = 25,
) -> Dict[str, torch.Tensor]:
    """Build a V2-compatible state_dict by transferring from V1 where possible.

    Strategy per key:
      - same shape: copy V1 → V2
      - per-TF *_proj.weight with in_dim mismatch: pad zeros
      - multi_tf_fuse[0].weight with in_dim mismatch: pad zeros (new M5 slot)
      - anything else mismatched: KEEP V2 init (return v2_state's value)
      - keys only in V2 (m5_*): KEEP V2 init
      - keys only in V1: drop (silently)
    """
    out: Dict[str, torch.Tensor] = {}
    transferred = 0
    padded_per_tf = 0
    padded_fuse = 0
    kept_v2_init = 0
    skipped_v1_only = 0

    v1_keys = set(v1_state.keys())
    v2_keys = set(v2_state.keys())

    for k in v2_keys:
        v2_val = v2_state[k]
        if k not in v1_keys:
            out[k] = v2_val
            kept_v2_init += 1
            continue
        v1_val = v1_state[k]
        if v1_val.shape == v2_val.shape:
            out[k] = v1_val
            transferred += 1
            continue
        # Shape mismatch — handle specific known patterns.
        is_per_tf_proj_w = any(
            k.startswith(prefix) and k.endswith(".weight")
            for prefix in PER_TF_PROJ_PREFIXES
        )
        if is_per_tf_proj_w and v2_val.dim() == 2 and v1_val.dim() == 2 \
                and v2_val.shape[0] == v1_val.shape[0]:
            out[k] = _pad_zeros_in_dim(v1_val, target_in_dim=v2_val.shape[1])
            padded_per_tf += 1
            continue
        # multi_tf_fuse.0.weight: in_dim grows 4*d → 5*d (extra M5 slot at start).
        # Pad zeros at the END (new slot position is arbitrary — model relearns layer-order).
        if k == "multi_tf_fuse.0.weight" and v2_val.dim() == 2 and v1_val.dim() == 2 \
                and v2_val.shape[0] == v1_val.shape[0]:
            out[k] = _pad_zeros_in_dim(v1_val, target_in_dim=v2_val.shape[1])
            padded_fuse += 1
            continue
        # Unknown mismatch: keep V2 init, log.
        print(f"  [WARN] keep V2 init for '{k}': V1 shape {tuple(v1_val.shape)} "
              f"vs V2 shape {tuple(v2_val.shape)}", file=sys.stderr)
        out[k] = v2_val
        kept_v2_init += 1

    skipped_v1_only = len(v1_keys - v2_keys)
    print(f"\n[WARM_START_SUMMARY]")
    print(f"  transferred clean: {transferred}")
    print(f"  padded per-TF proj: {padded_per_tf}")
    print(f"  padded multi_tf_fuse: {padded_fuse}")
    print(f"  kept V2 init (new keys or unknown mismatch): {kept_v2_init}")
    print(f"  skipped V1-only keys: {skipped_v1_only}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1-bundle", type=Path, required=True,
                   help="Path to V1 V10 bundle dir (must contain model_state_dict.pt)")
    p.add_argument("--v2-out", type=Path, required=True,
                   help="Output path for warm-started V2 state_dict (.pt)")
    p.add_argument("--per-tf-dim", type=int, default=25,
                   help="V2 per-TF feature dim (default 25 = MULTI_TF_PER_BAR_FEATURES_V2)")
    args = p.parse_args()

    v1_state_path = args.v1_bundle / "model_state_dict.pt"
    if not v1_state_path.is_file():
        raise FileNotFoundError(f"V1 state_dict not found: {v1_state_path}")
    print(f"[LOAD] V1 state_dict: {v1_state_path}")
    v1_state = torch.load(v1_state_path, map_location="cpu", weights_only=True)
    # Strip _orig_mod. prefix from torch.compile-wrapped bundles.
    v1_state = {k.removeprefix("_orig_mod."): v for k, v in v1_state.items()}
    print(f"  → {len(v1_state)} keys")

    # Build a V2 model with default config to get the canonical V2 state_dict layout.
    # We construct a V2 model with placeholder dims that match common training defaults.
    print("[BUILD] reference V2 model for shape inference")
    sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
    # Use V10 v3+ canonical dims. Per-TF len defaults to 96 — irrelevant for state_dict shape.
    ref_model = EntryV10CtxHybridTransformer(
        seq_input_dim=37, snap_input_dim=37, seq_len=96,
        ctx_cont_dim=45, ctx_cat_dim=6,
        enable_multi_tf=True,
        enable_multi_tf_m5=True,
        m5_seq_dim=int(args.per_tf_dim), m15_seq_dim=int(args.per_tf_dim),
        h1_seq_dim=int(args.per_tf_dim), h4_seq_dim=int(args.per_tf_dim),
        d1_seq_dim=int(args.per_tf_dim),
        m5_seq_len=96, m15_seq_len=96, h1_seq_len=96, h4_seq_len=96, d1_seq_len=96,
    )
    v2_state = {k: v.clone() for k, v in ref_model.state_dict().items()}
    print(f"  → {len(v2_state)} keys, {sum(p.numel() for p in ref_model.parameters())/1e6:.2f}M params")

    print("[WARM_START] transferring weights...")
    out_state = warm_start_state_dict(v1_state, v2_state, new_per_tf_dim=int(args.per_tf_dim))

    args.v2_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_state, args.v2_out)
    print(f"\n[SAVE] {args.v2_out} ({args.v2_out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
