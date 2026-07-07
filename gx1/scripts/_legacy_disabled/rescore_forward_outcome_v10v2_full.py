#!/usr/bin/env python3
"""Full V10 batch inference over canonical_v3 + join into forward-outcome.

Unlike the legacy-disabled rescore_forward_outcome_with_v10_v2.py (which only
covered candidates present in the V10 training dataset, 47.6% match), this script:

  1. Loads canonical_v3 M5 prebuilt (456K bars, 2020-2026)
  2. Augments via augment_canonical_v3 (adds session/interaction features)
  3. Runs XGB → 7-dim bridge matrix
  4. Loads V10 V2 prelim + V2 multi-TF cache
  5. Runs V10 V2 forward on EVERY M5 bar where idx >= seq_len-1
  6. Saves time-indexed V10 output parquet
  7. Joins into forward-outcome (~100% match expected)

Heavy on first invocation (~5-10 min for XGB + V10 inference on full M5 series),
but gives proper V2 coverage for FINAL retrain.

Usage:
    python -m gx1.scripts.rescore_forward_outcome_v10v2_full \\
        --fwd-in   /path/to/CANDIDATE_FORWARD_OUTCOME_..._LOCK \\
        --fwd-out  /path/to/CANDIDATE_FORWARD_OUTCOME_..._LOCK_V10V2_FULL \\
        --v10-bundle /path/to/ENTRY_V10_V2_PRELIM_*
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CANONICAL = Path("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_FULL_PLUS_CTX_2020_2026.parquet")
# --xgb-bundle is REQUIRED (no silent stale-bundle default; rule 8). The old hardcoded literal
# (xgb_v7_base80_20260526T052210Z) is now an INVALIDATED history entry — pass the contract-active
# xgb explicitly (PROJECT_STATE_artifacts.json active.xgb) so a rescore can never bake a stale bundle.
DEFAULT_XGB_CONTRACT = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_base80_v1.json"
DEFAULT_XGB_SANITIZER = REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base80_v1.json"
DEFAULT_V2_CACHE_DIR = Path("/home/andre2/GX1_DATA/data/data/prebuilt/MULTI_TF_V2_CACHE")

# V10 output cols (snapshot at decision bar). Keep the explicit _v2 columns for
# lineage/debugging, but also overwrite the base names Entry-IQL actually reads.
V10_V2_OUT_COLS = [
    "p_long_v2", "p_short_v2", "p_flat_v2", "p_hat_v2",
    "tradable_prob_v2", "mfe_first_n_pred_v2", "path_quality_pred_v2", "bad_path_prob_v2",
    "direction_logit_long_v2", "direction_logit_short_v2", "direction_logit_flat_v2",
    "path_quality_std_v2",
]
V10_V2_TO_BASE = {
    "p_long_v2": "p_long",
    "p_short_v2": "p_short",
    "p_flat_v2": "p_flat",
    "p_hat_v2": "p_hat",
    "tradable_prob_v2": "tradable_prob",
    "mfe_first_n_pred_v2": "mfe_first_n_pred",
    "path_quality_pred_v2": "path_quality_pred",
    "bad_path_prob_v2": "bad_path_prob",
    "direction_logit_long_v2": "direction_logit_long",
    "direction_logit_short_v2": "direction_logit_short",
    "direction_logit_flat_v2": "direction_logit_flat",
    "path_quality_std_v2": "path_quality_std",
}
V10_DERIVED_BASE_COLS = ["margin", "uncertainty_score", "entropy_v1"]
_NEW_HEAD_DIMS = {"dip": 18, "forecast": 4, "timing": 12, "tail_risk": 6, "vol_forecast": 3}
V10_OUT_COLS = V10_V2_OUT_COLS + list(V10_V2_TO_BASE.values()) + V10_DERIVED_BASE_COLS


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.clip(x.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - np.nanmax(logits.astype(np.float64), axis=1, keepdims=True)
    probs = np.exp(z)
    return probs / np.nansum(probs, axis=1, keepdims=True)


def _margin_top1_top2(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs.astype(np.float64), axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def _entropy_natural(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _assert_v10_base_matches_v2(df: pd.DataFrame, *, context: str = "v10_rescore") -> None:
    for v2_col, base_col in V10_V2_TO_BASE.items():
        if v2_col not in df.columns or base_col not in df.columns:
            raise AssertionError(f"[{context}] missing V10 proof column pair: {v2_col}->{base_col}")
        left = pd.to_numeric(df[v2_col], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(df[base_col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.allclose(left, right, rtol=0.0, atol=1e-6, equal_nan=True):
            raise AssertionError(f"[{context}] base column {base_col} does not match {v2_col}")

    prob_cols = ["p_long_v2", "p_short_v2", "p_flat_v2"]
    if all(c in df.columns for c in prob_cols):
        probs = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        checks = {
            "margin": _margin_top1_top2(probs),
            "uncertainty_score": 1.0 - np.nanmax(probs, axis=1),
            "entropy_v1": _entropy_natural(probs),
        }
        for col, expected in checks.items():
            if col not in df.columns:
                raise AssertionError(f"[{context}] missing derived base column: {col}")
            actual = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            if not np.allclose(actual, expected, rtol=0.0, atol=1e-6, equal_nan=True):
                raise AssertionError(f"[{context}] derived base column {col} is inconsistent")


def _add_entry_iql_base_columns(v10_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in V10_V2_OUT_COLS if c not in v10_df.columns]
    if missing:
        raise RuntimeError(f"[BASE_OVERWRITE] missing V10 _v2 columns: {missing}")

    out = v10_df.copy()
    for v2_col, base_col in V10_V2_TO_BASE.items():
        out[base_col] = pd.to_numeric(out[v2_col], errors="coerce").astype("float32")

    probs = out[["p_long_v2", "p_short_v2", "p_flat_v2"]].to_numpy(dtype=np.float64)
    out["margin"] = _margin_top1_top2(probs).astype(np.float32)
    out["uncertainty_score"] = (1.0 - np.nanmax(probs, axis=1)).astype(np.float32)
    out["entropy_v1"] = _entropy_natural(probs).astype(np.float32)

    for head_name, head_dim in _NEW_HEAD_DIMS.items():
        for i in range(head_dim):
            v2_col = f"v10_{head_name}_{i}_v2"
            if v2_col in out.columns:
                out[f"v10_{head_name}_{i}"] = pd.to_numeric(out[v2_col], errors="coerce").astype("float32")

    _assert_v10_base_matches_v2(out, context="BASE_OVERWRITE")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fwd-in", type=Path, required=True)
    p.add_argument("--fwd-out", type=Path, required=True)
    p.add_argument("--v10-bundle", type=Path, required=True)
    # Pre-rebuild fail-close (2026-06-04, rule 4): --canonical-v3 + --v2-cache-dir are now REQUIRED — the old
    # defaults pointed at the stale/DEGENERATE FULL_PLUS_CTX (const trend_regime_id, 05-22) and the frozen
    # 05-22 MULTI_TF_V2_CACHE, now quarantined. A re-score MUST pass the regime-fresh artifacts explicitly.
    p.add_argument("--canonical-v3", type=Path, required=True,
                   help="explicit regime-fresh canonical_v3/FULL_PLUS_CTX (no silent default)")
    p.add_argument("--xgb-bundle", type=Path, required=True,
                   help="explicit xgb bundle (no silent stale default; pass the contract-active xgb)")
    p.add_argument("--xgb-contract", type=Path, default=DEFAULT_XGB_CONTRACT)
    p.add_argument("--xgb-sanitizer", type=Path, default=DEFAULT_XGB_SANITIZER)
    p.add_argument("--v2-cache-dir", type=Path, required=True,
                   help="explicit regime-fresh MULTI_TF_V2_CACHE (no silent default)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--min-match-rate", type=float, default=0.99,
                   help="fail if fewer than this fraction of forward-outcome rows receive exact V10 scores")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[FULL_V10V2] device={device}", flush=True)

    # ── Step 1: load canonical_v3 + augment ───────────────────────────────
    from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
    print(f"[STEP 1] loading canonical_v3: {args.canonical_v3.name}", flush=True)
    cv3 = pd.read_parquet(args.canonical_v3)
    if "time" in cv3.columns and not isinstance(cv3.index, pd.DatetimeIndex):
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()
    print(f"  cv3 shape={cv3.shape} ts=[{cv3.index[0]} → {cv3.index[-1]}]", flush=True)

    pre_aug_cols = set(cv3.columns)
    cv3 = augment_canonical_v3(cv3, cv3)
    print(f"  augmented: +{len(set(cv3.columns) - pre_aug_cols)} cols → {len(cv3.columns)} total", flush=True)

    # ── Step 2: XGB → bridge ──────────────────────────────────────────────
    from gx1.scripts.materialize_inference_batch_candidates_v3_v1 import (
        run_xgb_inference, build_v10_input_matrices,
    )
    print(f"[STEP 2] XGB inference → bridge", flush=True)
    bridge = run_xgb_inference(
        cv3, xgb_bundle=args.xgb_bundle,
        feature_contract=args.xgb_contract, sanitizer_config=args.xgb_sanitizer,
    )
    print(f"  bridge shape: {bridge.shape}", flush=True)

    # ── Step 3: build V10 input matrices ──────────────────────────────────
    print(f"[STEP 3] building V10 input matrices", flush=True)
    per_bar, ctx_cont, ctx_cat = build_v10_input_matrices(cv3, bridge)
    print(f"  per_bar={per_bar.shape} ctx_cont={ctx_cont.shape} ctx_cat={ctx_cat.shape}", flush=True)

    # ── Step 4: load V10 V2 prelim + V2 multi-TF cache ───────────────────
    print(f"[STEP 4] loading V10 V2 bundle: {args.v10_bundle.name}", flush=True)
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
    bundle = load_entry_v10_ctx_bundle(bundle_dir=args.v10_bundle, device=args.device, xgb_models=None)
    model = bundle.transformer_model
    mtf_meta = bundle.metadata.get("multi_tf", {})
    print(f"  multi_tf: v2={mtf_meta.get('v2_mode')} m5_seq_dim={mtf_meta.get('m5_seq_dim')} "
          f"h4_len={mtf_meta.get('h4_seq_len')} d1_len={mtf_meta.get('d1_seq_len')}", flush=True)

    from gx1.features.htf_features import load_multi_tf_v2_cache, MULTI_TF_SHIFT
    multi_tf_feats = load_multi_tf_v2_cache(args.v2_cache_dir)
    print(f"  V2 cache loaded: TFs={list(multi_tf_feats.keys())}", flush=True)

    # ── Step 5: V10 V2 inference on ALL M5 decision bars (custom loop for V2) ──
    print(f"[STEP 5] V10 V2 inference on {len(cv3):,} M5 bars", flush=True)
    from gx1.features.htf_features import get_last_n_at_or_before
    seq_len = int(bundle.metadata["seq_len"])
    decision_indices = np.arange(seq_len - 1, len(cv3))
    n_dec = len(decision_indices)
    per_tf_lens = {
        "M5": int(mtf_meta.get("m5_seq_len", 96)),
        "M15": int(mtf_meta.get("m15_seq_len", 96)),
        "H1": int(mtf_meta.get("h1_seq_len", 96)),
        "H4": int(mtf_meta.get("h4_seq_len", 96)),
        "D1": int(mtf_meta.get("d1_seq_len", 96)),
    }
    v2_mode = bool(mtf_meta.get("v2_mode", False))
    print(f"  per-TF seq_lens: {per_tf_lens} | v2_mode={v2_mode}", flush=True)
    decision_ts_ns = cv3.index.values.astype("datetime64[ns]").astype(np.int64)

    out_logits = np.zeros((n_dec, 3), dtype=np.float32)
    out_path_quality = np.zeros(n_dec, dtype=np.float32)
    out_mfe = np.zeros(n_dec, dtype=np.float32)
    out_tradable_logit = np.zeros(n_dec, dtype=np.float32)
    out_bad_path_logit = np.zeros(n_dec, dtype=np.float32)
    # 2026-05-27: COSTFIX V10 new heads (dip/forecast/timing/tail/vol). NaN-init so
    # a bundle lacking a head emits NaN (Entry-IQL trainer missing-fills). Emitted
    # index-consistent below: out["<head>_pred"][:, i] -> column v10_<head>_{i}_v2.
    out_new_heads = {h: np.full((n_dec, d), np.nan, dtype=np.float32) for h, d in _NEW_HEAD_DIMS.items()}

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for batch_start in range(0, n_dec, args.batch_size):
            batch_idx = decision_indices[batch_start:batch_start + args.batch_size]
            B = len(batch_idx)
            # Build seq_x (B, seq_len, dim)
            seq_x = np.zeros((B, seq_len, per_bar.shape[1]), dtype=np.float32)
            for bi, di in enumerate(batch_idx):
                seq_x[bi] = per_bar[di - seq_len + 1: di + 1]
            snap_x = per_bar[batch_idx]
            ctx_cont_x = ctx_cont[batch_idx]
            ctx_cat_x = ctx_cat[batch_idx]
            # Build multi-TF (per-TF len)
            mtf_kw = {}
            for tf in ("M5", "M15", "H1", "H4", "D1"):
                if tf == "M5" and not v2_mode:
                    continue  # V1 V10 doesn't have M5 branch
                if tf not in multi_tf_feats:
                    continue
                n = per_tf_lens[tf]
                arr = np.zeros((B, n, multi_tf_feats[tf].shape[1]), dtype=np.float32)
                for bi, di in enumerate(batch_idx):
                    arr[bi] = get_last_n_at_or_before(
                        multi_tf_feats[tf], pd.Timestamp(decision_ts_ns[di], tz="UTC", unit="ns"),
                        n=n, tf_shift=MULTI_TF_SHIFT[tf],
                    )
                mtf_kw[f"seq_{tf.lower()}"] = torch.from_numpy(arr).to(device)
            seq_t = torch.from_numpy(seq_x).to(device)
            snap_t = torch.from_numpy(snap_x).to(device)
            cont_t = torch.from_numpy(ctx_cont_x).to(device)
            cat_t = torch.from_numpy(ctx_cat_x).to(device)
            out = model(seq_t, snap_t, ctx_cat=cat_t, ctx_cont=cont_t, **mtf_kw)
            slc = slice(batch_start, batch_start + B)
            out_logits[slc] = out["direction_logits"].cpu().numpy()
            out_path_quality[slc] = out["path_quality"].cpu().numpy().squeeze(-1)
            out_mfe[slc] = out["mfe_first_n"].cpu().numpy().squeeze(-1)
            out_tradable_logit[slc] = out["tradable_logit"].cpu().numpy().squeeze(-1)
            out_bad_path_logit[slc] = out["bad_path_logit"].cpu().numpy().squeeze(-1)
            for _h in _NEW_HEAD_DIMS:
                _k = f"{_h}_pred"
                if _k in out:
                    out_new_heads[_h][slc] = out[_k].detach().cpu().numpy()
            if (batch_start // args.batch_size) % 50 == 0:
                elapsed = time.time() - t0
                rate = (batch_start + B) / max(1e-6, elapsed)
                eta = (n_dec - batch_start - B) / max(1, rate)
                print(f"  {batch_start+B:,}/{n_dec:,} ({100*(batch_start+B)/n_dec:.1f}%) rate={rate:.0f}/s eta={eta:.0f}s", flush=True)
    print(f"  V10 inference done in {time.time()-t0:.1f}s", flush=True)
    out_dict = {
        "direction_logits": out_logits,
        "path_quality": out_path_quality[:, None],
        "mfe_first_n": out_mfe[:, None],
        "tradable_logit": out_tradable_logit[:, None],
        "bad_path_logit": out_bad_path_logit[:, None],
    }

    # ── Step 6: build time-indexed output DataFrame ──────────────────────
    seq_len = int(bundle.metadata["seq_len"])
    valid_idx = np.arange(seq_len - 1, len(cv3))  # decision possible at idx >= seq_len-1
    times = pd.to_datetime(cv3.index.values[valid_idx], utc=True)
    print(f"[STEP 6] building V10 score parquet ({len(times):,} rows)", flush=True)

    dir_logits = out_dict["direction_logits"]  # (n, 3)
    dir_probs = _softmax_np(dir_logits).astype(np.float32)

    _v10_data = {
        "time": times,
        "p_long_v2": dir_probs[:, 0].astype(np.float32),
        "p_short_v2": dir_probs[:, 1].astype(np.float32),
        "p_flat_v2": dir_probs[:, 2].astype(np.float32),
        "p_hat_v2": dir_probs.max(axis=1).astype(np.float32),
        "tradable_prob_v2": _sigmoid_np(out_dict["tradable_logit"].squeeze(-1)).astype(np.float32),
        "mfe_first_n_pred_v2": out_dict["mfe_first_n"].squeeze(-1).astype(np.float32),
        "path_quality_pred_v2": out_dict["path_quality"].squeeze(-1).astype(np.float32),
        "bad_path_prob_v2": _sigmoid_np(out_dict["bad_path_logit"].squeeze(-1)).astype(np.float32),
        "direction_logit_long_v2": dir_logits[:, 0].astype(np.float32),
        "direction_logit_short_v2": dir_logits[:, 1].astype(np.float32),
        "direction_logit_flat_v2": dir_logits[:, 2].astype(np.float32),
        "path_quality_std_v2": np.zeros(len(times), dtype=np.float32),  # log_var disabled in prelim
    }
    # 2026-05-27: COSTFIX V10 new heads (dip 18 / forecast 4 / timing 12 / tail_risk 6
    # / vol_forecast 3 = 43). Index-consistent: out["<head>_pred"][:, i] -> v10_<head>_{i}_v2.
    # NaN where the bundle lacks the head. Entry-IQL reads the base names, so
    # _add_entry_iql_base_columns mirrors v10_<head>_{i}_v2 -> v10_<head>_{i}.
    for _h, _arr in out_new_heads.items():
        for _i in range(_arr.shape[1]):
            _v10_data[f"v10_{_h}_{_i}_v2"] = _arr[:, _i].astype(np.float32)
    v10_df = pd.DataFrame(_v10_data).set_index("time").sort_index()
    v10_df = _add_entry_iql_base_columns(v10_df)
    print(
        f"  v10_df ready: shape={v10_df.shape} "
        f"(+{sum(d for d in _NEW_HEAD_DIMS.values())} _v2 new-head cols, base Entry-IQL overwrite enabled)",
        flush=True,
    )

    # ── Step 7: join into forward-outcome per-week ───────────────────────
    args.fwd_out.mkdir(parents=True, exist_ok=True)
    per_week_in = args.fwd_in / "per_week"
    per_week_out = args.fwd_out / "per_week"
    per_week_out.mkdir(parents=True, exist_ok=True)
    week_paths = sorted(per_week_in.glob("*.parquet"))
    print(f"[STEP 7] join {len(week_paths)} per-week parquets", flush=True)

    v10_idx_int = v10_df.index.values.astype("datetime64[ns]").astype(np.int64)
    v10_arr = v10_df.to_numpy(dtype=np.float32)
    v10_cols = list(v10_df.columns)
    required_join_cols = set(V10_V2_OUT_COLS) | set(V10_V2_TO_BASE.values()) | set(V10_DERIVED_BASE_COLS)
    missing_join_cols = sorted(required_join_cols - set(v10_cols))
    if missing_join_cols:
        raise RuntimeError(f"[JOIN_CONTRACT] missing columns before join: {missing_join_cols}")

    n_matched = 0
    n_total = 0
    t0 = time.time()
    for i, wp in enumerate(week_paths):
        df = pd.read_parquet(wp)
        n_total += len(df)
        ts = pd.to_datetime(df["decision_ts_utc"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
        right = np.searchsorted(v10_idx_int, ts, side="left")
        in_range = right < len(v10_idx_int)
        match_mask = np.zeros(len(ts), dtype=bool)
        match_mask[in_range] = (v10_idx_int[right[in_range]] == ts[in_range])
        n_matched += int(match_mask.sum())
        new_cols = np.full((len(ts), len(v10_cols)), np.nan, dtype=np.float32)
        if match_mask.any():
            matched_idx = right[match_mask]
            new_cols[match_mask] = v10_arr[matched_idx]
        for j, col in enumerate(v10_cols):
            df[col] = new_cols[:, j]
        if match_mask.any():
            proof_cols = sorted(required_join_cols)
            _assert_v10_base_matches_v2(df.loc[match_mask, proof_cols], context=f"JOIN_PROOF:{wp.name}")
        df.to_parquet(per_week_out / wp.name, index=False)
        if (i + 1) % 50 == 0 or i + 1 == len(week_paths):
            elapsed = time.time() - t0
            print(f"  [JOIN] {i+1}/{len(week_paths)} weeks, matched={n_matched:,}/{n_total:,} "
                  f"({100*n_matched/max(1,n_total):.1f}%), elapsed={elapsed:.0f}s", flush=True)
    match_rate = n_matched / max(1, n_total)
    print(f"[DONE] total matched: {n_matched:,}/{n_total:,} ({100*match_rate:.1f}%)", flush=True)
    if match_rate < float(args.min_match_rate):
        raise RuntimeError(
            f"[JOIN_CONTRACT] V10 score match rate {match_rate:.6f} < required {args.min_match_rate:.6f}; "
            "refusing to publish a partial new-eyes forward_outcome"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
