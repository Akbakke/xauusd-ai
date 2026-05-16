"""
Score V3 v8 (exit transformer) outputs onto the EXIT_IQL_PER_BAR_DATASET_V2_M1.

Adds per-bar V3 transformer predictions as new columns:
  - v3_v8_should_exit_prob:    sigmoid(main_head_logit) — primary HOLD/EXIT signal
  - v3_v8_profit_protect_prob: sigmoid(profit_protect_logit)
  - v3_v8_family_argmax:       argmax over family classes (top-1 family)
  - v3_v8_family_logit_max:    max logit across families (confidence)

Why: validation showed V3 transformer (90% test acc) is NOT input to Exit-IQL.
Adding its predictions as features should give Exit-IQL access to the strongest
HOLD/EXIT signal currently in the stack, lifting Q-rangering quality.

Architecture:
  - V3 v8 input: (B, T=512, D=91) — EXIT_IO_V6 contract M1 sequence
  - For each per-bar row:
    - bar_ts_ns_v1 → m1_idx_now via m1_time_ns binary search
    - Window = m1_feature_matrix[m1_idx_now-511 : m1_idx_now+1]
    - For in-trade bars in window: overlay 19 trade-state values from
      this trade's per-bar rows (sorted by bars_in_trade)
    - V3 v8 forward → 3 outputs

Inputs:
  --v3-bundle:      path to V3 v8 bundle (with .pt + transformer_config.json)
  --v3-dataset-dir: path to V3 training dataset (provides m1_feature_matrix + m1_time_ns)
  --per-bar-dir:    path to EXIT_IQL_PER_BAR_DATASET_V2_M1 (input)
  --out-root:       where to write augmented per-bar parquets
  --week:           optional, repeat for specific week names (smoke test)
  --batch-size:     transformer batch size (default 256)

Output:
  <out-root>/per_week/exit_per_bar_m1_<week>.parquet  — augmented with V3 cols
  <out-root>/manifest_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.policy.exit_transformer_v0 import ExitTransformerV0
from gx1.exits.contracts.exit_io_v6_ctx_v3canonical_m1l512 import (
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES as V6_FEATURES,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT as V6_FEATURE_COUNT,
)

ACTION = "SCORE_V3_V8_ON_PER_BAR_V1"
# V12.2 (2026-05-15): default updated to V3 v9 multi-TF. The script
# HARD-REQUIRES a multi-TF bundle; V8 (non-multi-TF) is deprecated.
DEFAULT_V3_BUNDLE = Path(
    "/home/andre2/GX1_DATA/models/exit_transformer_v0/EXIT_V9_MULTI_TF_LR5E4_SCALE025_20260513T223544Z"
)
DEFAULT_V3_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3"
)
DEFAULT_PER_BAR_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/EXIT_IQL_PER_BAR_DATASET_V2_M1"
)
WINDOW_LEN = 512

# V6 trade-state feature names → per-bar parquet column names
TRADE_STATE_FEATURE_NAMES_V6 = [
    "p_long_entry", "p_hat_entry", "uncertainty_entry", "entropy_entry",
    "margin_entry", "pnl_bps_now", "mfe_bps", "mae_bps", "dd_from_mfe_bps",
    "distance_from_peak_mfe_bps", "bars_held", "time_since_mfe_bars",
    "mfe_decay_rate", "pnl_velocity", "pnl_acceleration", "rolling_slope_since_entry",
    "atr_bps_now", "giveback_ratio", "giveback_acceleration",
]
# Map V6 name → per-bar parquet column name. _v1 suffix on most. Some renamed.
PER_BAR_COL_BY_V6 = {
    "p_long_entry": "p_long_entry_v1",
    "p_hat_entry": "p_hat_entry_v1",
    "uncertainty_entry": "uncertainty_entry_v1",
    "entropy_entry": "entropy_entry_v1",
    "margin_entry": "margin_entry_v1",
    "pnl_bps_now": "current_unrealized_pnl_bps_v1",
    "mfe_bps": "current_mfe_bps_v1",
    "mae_bps": "current_mae_bps_v1",
    "dd_from_mfe_bps": "pnl_drawdown_from_peak_v1",        # mfe - pnl_now equivalent
    "distance_from_peak_mfe_bps": "pnl_drawdown_from_peak_v1",  # same metric
    "bars_held": "bars_in_trade_v1",
    "time_since_mfe_bars": "bars_since_mfe_peak_v1",
    "mfe_decay_rate": "mfe_decay_rate_v1",
    "pnl_velocity": "pnl_velocity_v1",
    "pnl_acceleration": "pnl_acceleration_v1",
    "rolling_slope_since_entry": "rolling_slope_since_entry_v1",
    "atr_bps_now": "current_atr_bps_v1",
    "giveback_ratio": "giveback_ratio_v1",
    "giveback_acceleration": "giveback_acceleration_v1",
}
# V6 indices for trade-state features in the 91-feature vector
TRADE_STATE_V6_INDICES = [V6_FEATURES.index(n) for n in TRADE_STATE_FEATURE_NAMES_V6]


def load_v3_v8_model(bundle_dir: Path, device: torch.device) -> ExitTransformerV0:
    cfg_path = bundle_dir / "transformer_config.json"
    state_path = bundle_dir / "exit_transformer_v0.pt"
    cfg = json.loads(cfg_path.read_text())
    expected_dim = V6_FEATURE_COUNT
    if cfg["input_dim"] != expected_dim:
        raise ValueError(f"V3 v8 input_dim={cfg['input_dim']} mismatch V6={expected_dim}")
    if cfg.get("exit_ml_io_version") != "EXIT_IO_V6_CTX_V3CANONICAL_M1L512":
        raise ValueError(f"V3 v8 io_version mismatch: {cfg.get('exit_ml_io_version')}")

    # V12.2 (2026-05-15): multi-TF is now REQUIRED. Reject non-multi-TF bundles.
    mtf_cfg = cfg.get("multi_tf", {}) or {}
    enable_mtf = bool(mtf_cfg.get("enabled", False))
    if not enable_mtf:
        raise RuntimeError(
            f"[V3 LOAD] bundle {bundle_dir.name} is NOT multi-TF "
            f"(multi_tf.enabled={enable_mtf}). V12.2 hard-requires multi-TF. "
            "Use V3 v9+ bundle or update transformer_config.json."
        )
    mtf_kwargs = dict(
        enable_multi_tf=True,
        m5_seq_dim=int(mtf_cfg["m5_seq_dim"]),
        m15_seq_dim=int(mtf_cfg["m15_seq_dim"]),
        h1_seq_dim=int(mtf_cfg["h1_seq_dim"]),
        h4_seq_dim=int(mtf_cfg["h4_seq_dim"]),
        d1_seq_dim=int(mtf_cfg["d1_seq_dim"]),
        m5_seq_len=int(mtf_cfg["m5_seq_len"]),
        m15_seq_len=int(mtf_cfg["m15_seq_len"]),
        h1_seq_len=int(mtf_cfg["h1_seq_len"]),
        h4_seq_len=int(mtf_cfg["h4_seq_len"]),
        d1_seq_len=int(mtf_cfg["d1_seq_len"]),
    )
    print(f"[V3 LOAD] multi-TF bundle detected — model constructed with M5+M15+H1+H4+D1", flush=True)

    model = ExitTransformerV0(
        input_dim=cfg["input_dim"], window_len=cfg["window_len"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        dropout=cfg.get("dropout", 0.1),
        **mtf_kwargs,
    )
    state_dict = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def build_overlay_for_trade(trade_rows: pd.DataFrame) -> np.ndarray:
    """Build (n_bars, 19) overlay matrix from trade's per-bar rows sorted by bar_idx_v1.

    Each row in the overlay is the 19-dim trade-state vector at that bar.
    """
    sorted_rows = trade_rows.sort_values("bar_idx_v1")
    overlay = np.zeros((len(sorted_rows), len(TRADE_STATE_FEATURE_NAMES_V6)), dtype=np.float32)
    for j, v6_name in enumerate(TRADE_STATE_FEATURE_NAMES_V6):
        col = PER_BAR_COL_BY_V6[v6_name]
        if col in sorted_rows.columns:
            overlay[:, j] = pd.to_numeric(sorted_rows[col], errors="coerce").fillna(0.0).to_numpy()
    return overlay, sorted_rows


@torch.no_grad()
def score_week(
    week_parquet: Path, m1_feature_matrix: np.memmap, m1_time_ns: np.ndarray,
    model: ExitTransformerV0, device: torch.device, batch_size: int,
    out_path: Path,
    multi_tf_feats: Optional[Dict[str, Any]] = None,
    multi_tf_shift: Optional[Dict[str, Any]] = None,
    multi_tf_seq_len: int = 96,
) -> Dict[str, int]:
    df = pd.read_parquet(week_parquet)
    n_rows = len(df)
    if n_rows == 0:
        df["v3_v8_should_exit_prob"] = np.array([], dtype=np.float32)
        df["v3_v8_profit_protect_prob"] = np.array([], dtype=np.float32)
        df["v3_v8_family_argmax"] = np.array([], dtype=np.int64)
        df["v3_v8_family_logit_max"] = np.array([], dtype=np.float32)
        df.to_parquet(out_path, index=False)
        return {"week": week_parquet.stem, "n_rows": 0, "n_skipped_oob": 0, "n_trades": 0}

    out_should = np.full(n_rows, np.nan, dtype=np.float32)
    out_profit = np.full(n_rows, np.nan, dtype=np.float32)
    out_family = np.full(n_rows, -1, dtype=np.int64)
    out_family_max = np.full(n_rows, np.nan, dtype=np.float32)

    bar_ts_ns = pd.to_numeric(df["bar_ts_ns_v1"], errors="coerce").fillna(0).astype("int64").to_numpy()
    m1_idx_now = np.searchsorted(m1_time_ns, bar_ts_ns, side="right") - 1
    bars_in_trade = pd.to_numeric(df["bars_in_trade_v1"], errors="coerce").fillna(0).astype("int64").to_numpy()

    n_skipped_oob = 0
    n_trades = 0
    grouped = df.groupby("candidate_uid", sort=False)
    pending_x: List[np.ndarray] = []
    pending_idx: List[int] = []
    # V12.2: per-pending timestamp (only used when multi-TF is enabled).
    # We collect bar_ts_ns alongside io windows, then slice multi-TF windows
    # batched in flush_batch.
    pending_ts_ns: List[int] = []

    enable_mtf = bool(getattr(model, "enable_multi_tf", False))
    if enable_mtf and (multi_tf_feats is None or multi_tf_shift is None):
        raise RuntimeError(
            "[score_v3_v8] V3 bundle is multi-TF but multi_tf_feats/multi_tf_shift "
            "are None. Caller (main) must pre-build features from M5 prebuilt "
            "(via build_multi_tf_per_bar_features) and pass them in."
        )

    # Cache: precompute pandas Timedelta → int64 ns for fast slicing
    _mtf_shift_ns: Dict[str, int] = {}
    if enable_mtf:
        _mtf_shift_ns = {tf: int(td.value) for tf, td in multi_tf_shift.items()}

    def _build_mtf_batch(ts_ns_list: List[int]) -> Dict[str, torch.Tensor]:
        """Vectorized multi-TF batch slicing — uses pre-computed feats.attrs
        (ts_int64 + feats_np) and a vectorized searchsorted per TF, instead of
        the per-item pd.Timestamp + get_last_n_at_or_before path (~5-10x faster
        for B=256). The inner B-loop is unavoidable because the right index
        varies per item, but each iter is just a numpy slice (cheap)."""
        ts_ns_arr = np.asarray(ts_ns_list, dtype=np.int64)
        B = ts_ns_arr.shape[0]
        out: Dict[str, torch.Tensor] = {}
        for tf, feats in multi_tf_feats.items():
            ts_int64 = feats.attrs["ts_int64"]
            feats_np = feats.attrs["feats_np"]
            D = feats_np.shape[1]
            n = multi_tf_seq_len
            shift_ns = _mtf_shift_ns[tf]
            cutoffs = ts_ns_arr - shift_ns
            right_idx = np.searchsorted(ts_int64, cutoffs, side="right")
            stacked = np.zeros((B, n, D), dtype=np.float32)
            for i in range(B):
                r = int(right_idx[i])
                if r <= 0:
                    continue
                left = r - n if r >= n else 0
                tail = feats_np[left:r]
                if tail.shape[0] < n:
                    stacked[i, -tail.shape[0]:] = tail
                else:
                    stacked[i] = tail
            out[f"seq_{tf.lower()}"] = torch.from_numpy(stacked).to(device, non_blocking=True)
        return out

    def flush_batch() -> None:
        if not pending_x:
            return
        x = np.stack(pending_x, axis=0)
        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        mtf_kwargs = _build_mtf_batch(pending_ts_ns) if enable_mtf else {}
        # V12.2 speedup: bfloat16 autocast on the heavy forward.
        # ~1.5-2x throughput on Ampere/Hopper; V3 model is robust to mixed prec.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            h = model._encode(x_t, **mtf_kwargs)   # (B, d_model)
            main_logit = model.head(h).squeeze(-1)
            prof_logit = model.profit_protect_head(h).squeeze(-1)
            fam_logits = model.family_head(h)  # (B, n_family)

        # cast back to fp32 for stable sigmoid + argmax + cpu transfer
        main_prob = torch.sigmoid(main_logit.float()).cpu().numpy()
        prof_prob = torch.sigmoid(prof_logit.float()).cpu().numpy()
        fam_argmax = torch.argmax(fam_logits.float(), dim=-1).cpu().numpy()
        fam_max = fam_logits.float().max(dim=-1).values.cpu().numpy()

        for i, row_idx in enumerate(pending_idx):
            out_should[row_idx] = float(main_prob[i])
            out_profit[row_idx] = float(prof_prob[i])
            out_family[row_idx] = int(fam_argmax[i])
            out_family_max[row_idx] = float(fam_max[i])
        pending_x.clear()
        pending_idx.clear()
        pending_ts_ns.clear()

    for cand_uid, trade_rows in grouped:
        n_trades += 1
        overlay, sorted_rows = build_overlay_for_trade(trade_rows)
        sorted_indices = sorted_rows.index.to_numpy()  # absolute df indices
        sorted_bars_in_trade = pd.to_numeric(
            sorted_rows["bars_in_trade_v1"], errors="coerce"
        ).fillna(0).astype("int64").to_numpy()
        sorted_m1_idx = m1_idx_now[sorted_indices]
        # m1_idx of trade entry = m1_idx_now - bars_in_trade (per row, but constant per trade if data clean)
        s_t_arr = sorted_m1_idx - sorted_bars_in_trade  # candidate s_t per row
        # Use median to robust-est. of s_t for this trade
        s_t = int(np.median(s_t_arr))

        for i_in_trade, abs_idx in enumerate(sorted_indices):
            mi = int(sorted_m1_idx[i_in_trade])
            if mi < WINDOW_LEN - 1 or mi >= len(m1_feature_matrix):
                n_skipped_oob += 1
                continue
            win_start = mi - WINDOW_LEN + 1
            win_end = mi + 1
            io = np.array(m1_feature_matrix[win_start:win_end], dtype=np.float32, copy=True)

            # Apply overlay for in-trade bars in this window
            in_trade_start_in_win = max(0, s_t - win_start)
            in_trade_end_in_win = min(WINDOW_LEN, s_t + i_in_trade + 1 - win_start + 1)  # exclusive
            n_in_trade = max(0, in_trade_end_in_win - in_trade_start_in_win)
            if n_in_trade > 0:
                overlay_start_row = max(0, win_start - s_t)
                slice_end = overlay_start_row + n_in_trade
                slice_end = min(slice_end, len(overlay))
                actual_n = slice_end - overlay_start_row
                if actual_n > 0:
                    io[in_trade_start_in_win: in_trade_start_in_win + actual_n,
                       TRADE_STATE_V6_INDICES] = overlay[overlay_start_row: overlay_start_row + actual_n]

            pending_x.append(io)
            pending_idx.append(int(abs_idx))
            if enable_mtf:
                pending_ts_ns.append(int(bar_ts_ns[abs_idx]))
            if len(pending_x) >= batch_size:
                flush_batch()
    flush_batch()

    df = df.copy()
    df["v3_v8_should_exit_prob"] = out_should
    df["v3_v8_profit_protect_prob"] = out_profit
    df["v3_v8_family_argmax"] = out_family
    df["v3_v8_family_logit_max"] = out_family_max
    df.to_parquet(out_path, index=False)
    return {
        "week": week_parquet.stem, "n_rows": n_rows,
        "n_skipped_oob": int(n_skipped_oob), "n_trades": int(n_trades),
        "out_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ACTION}")
    parser.add_argument("--v3-bundle", type=str, default=str(DEFAULT_V3_BUNDLE))
    parser.add_argument("--v3-dataset-dir", type=str, default=str(DEFAULT_V3_DATASET_DIR))
    parser.add_argument("--per-bar-dir", type=str, default=str(DEFAULT_PER_BAR_DIR))
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--week", action="append", default=None,
                        help="Specific week stem to process (repeat). If unset, process all.")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="V12.2: default 4096 — uses ~5GB GPU mem with bf16 autocast, "
                             "leaves plenty of headroom on 24GB cards.")
    # V12.2: only needed when V3 bundle is multi-TF (V3 v9+).
    parser.add_argument("--m5-prebuilt-path", type=str, default=None,
                        help="V12.2: canonical_v3 M5 prebuilt parquet, required when "
                             "scoring with a multi-TF V3 bundle (auto-detected from cfg.multi_tf).")
    parser.add_argument("--multi-tf-seq-len", type=int, default=96)
    args = parser.parse_args()

    bundle = Path(args.v3_bundle).expanduser().resolve()
    v3_ds = Path(args.v3_dataset_dir).expanduser().resolve()
    pb_dir = Path(args.per_bar_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_per_week = out_root / "per_week"
    out_per_week.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ACTION}] device={device}", flush=True)

    print(f"[{ACTION}] loading V3 v8 from {bundle}", flush=True)
    model = load_v3_v8_model(bundle, device)
    print(f"[{ACTION}] loaded V3 v8: input_dim={model.input_dim} window_len={model.window_len} "
          f"d_model={model.d_model}", flush=True)

    # V12.2 speedup: torch.compile the _encode hot path. Dynamic=True handles
    # the last-batch shorter-than-batch_size case without re-compilation.
    if device.type == "cuda":
        try:
            compiled_encode = torch.compile(model._encode, mode="reduce-overhead", dynamic=True)
            model._encode = compiled_encode  # type: ignore[method-assign]
            print(f"[{ACTION}] torch.compile applied to model._encode (reduce-overhead, dynamic)", flush=True)
        except Exception as e:
            print(f"[{ACTION}] torch.compile FAILED ({e!r}) — falling back to eager", flush=True)

    # V12.2: if multi-TF bundle, build features ONCE before week-loop.
    mtf_feats = None
    mtf_shift = None
    if getattr(model, "enable_multi_tf", False):
        if not args.m5_prebuilt_path:
            raise RuntimeError(
                "[score_v3_v8] V3 bundle is multi-TF — --m5-prebuilt-path is required."
            )
        from gx1.features.htf_features import (
            build_multi_tf_per_bar_features, MULTI_TF_SHIFT,
        )
        print(f"[{ACTION}] V12.2: building multi-TF features from {args.m5_prebuilt_path}", flush=True)
        m5 = pd.read_parquet(args.m5_prebuilt_path,
                              columns=["time", "open", "high", "low", "close"])
        m5["time"] = pd.to_datetime(m5["time"], utc=True)
        m5 = m5.set_index("time").sort_index()
        for c in ("open", "high", "low", "close"):
            m5[c] = m5[c].astype(np.float32)
        mtf_feats = build_multi_tf_per_bar_features(m5)
        mtf_shift = MULTI_TF_SHIFT
        del m5
        import gc; gc.collect()
        for tf, df in mtf_feats.items():
            print(f"[{ACTION}]   {tf}: {len(df):,} bars × {df.shape[1]} feats", flush=True)

    print(f"[{ACTION}] loading m1_feature_matrix + m1_time_ns from {v3_ds}", flush=True)
    m1_feature_matrix = np.load(v3_ds / "m1_feature_matrix.npy", mmap_mode="r")
    m1_time_ns = np.load(v3_ds / "m1_time_ns.npy")
    print(f"[{ACTION}] m1 matrix shape={m1_feature_matrix.shape}, time_ns range="
          f"{m1_time_ns[0]}..{m1_time_ns[-1]}", flush=True)
    if m1_feature_matrix.shape[1] != V6_FEATURE_COUNT:
        raise ValueError(f"matrix dim {m1_feature_matrix.shape[1]} != V6 {V6_FEATURE_COUNT}")

    pb_per_week_dir = pb_dir / "per_week"
    week_files = sorted(pb_per_week_dir.glob("exit_per_bar_m1_*.parquet"))
    if args.week:
        wanted = set(args.week)
        week_files = [w for w in week_files if w.stem in wanted or any(t in w.stem for t in wanted)]
    print(f"[{ACTION}] processing {len(week_files)} week parquets", flush=True)

    total_rows = total_skipped = total_trades = 0
    week_stats: List[Dict] = []
    for w_idx, wp in enumerate(week_files, start=1):
        out_path = out_per_week / wp.name
        print(f"[{ACTION}] [{w_idx}/{len(week_files)}] {wp.stem}", flush=True)
        stats = score_week(
            wp, m1_feature_matrix, m1_time_ns, model, device,
            args.batch_size, out_path,
            multi_tf_feats=mtf_feats,
            multi_tf_shift=mtf_shift,
            multi_tf_seq_len=int(args.multi_tf_seq_len),
        )
        week_stats.append(stats)
        total_rows += stats["n_rows"]
        total_skipped += stats["n_skipped_oob"]
        total_trades += stats["n_trades"]
        print(f"[{ACTION}]   rows={stats['n_rows']} trades={stats['n_trades']} skipped_oob={stats['n_skipped_oob']}",
              flush=True)

    manifest = {
        "action_v1": ACTION,
        "v3_bundle": str(bundle),
        "v3_dataset_dir": str(v3_ds),
        "per_bar_dir": str(pb_dir),
        "out_root": str(out_root),
        "n_weeks": len(week_files),
        "total_rows": int(total_rows),
        "total_skipped_oob": int(total_skipped),
        "total_trades": int(total_trades),
        "v3_v8_input_dim": V6_FEATURE_COUNT,
        "v3_v8_window_len": WINDOW_LEN,
        "added_columns": [
            "v3_v8_should_exit_prob",
            "v3_v8_profit_protect_prob",
            "v3_v8_family_argmax",
            "v3_v8_family_logit_max",
        ],
    }
    (out_root / "manifest_v1.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[{ACTION}] DONE — {total_rows:,} rows, {total_skipped} skipped, {total_trades} trades", flush=True)


if __name__ == "__main__":
    main()
