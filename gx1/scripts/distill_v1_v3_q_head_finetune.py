#!/usr/bin/env python3
"""Phase B Stage 3b — V3 q_head fine-tune via Exit-IQL Q-target distillation.

Twin to distill_v1_v10_q_head_finetune.py for the EXIT side of the cascade.

Strategy: keep V3 v9 multi-TF backbone FROZEN; train only the new `q_head`
layer (zero-init in patched model, already supported by ExitTransformerV0
when enable_q_head=True) to mirror Exit-IQL's per-bar Q-distribution
{q_hold, q_exit}.

Inputs:
  - V3 v9 multi-TF bundle (default: EXIT_V9_MULTI_TF_LR5E4_SCALE025_*)
  - V3 training dataset dir (provides m1_feature_matrix.npy + m1_time_ns.npy)
  - Exit-IQL Q-target dir (per_week/exit_iql_q_targets_<week>.parquet from
    distill_v1_compute_exit_iql_q_targets.py)
  - Exit-IQL PER_BAR dataset dir (per_week/exit_per_bar_m1_<week>.parquet —
    supplies bar_ts_ns_v1, bars_in_trade_v1, candidate_uid, trade-state cols)
  - canonical_v3 M5 prebuilt (for multi-TF cache)

Loss: KL(softmax(q_pred / T), softmax(q_target / T)) * T²
Default T=3.0, batch=256, lr=3e-4, epochs=3.

Output bundle: EXIT_V9_MULTI_TF_DISTILLED_<ts>/
  - exit_transformer_v0.pt    (backbone + new q_head)
  - transformer_config.json   (clone of v9, plus "enable_q_head": true)
  - manifest.json             (clone of v9, plus "distill" subdict with loss curve)

Reuses windowing + multi-TF logic from score_v3_v8_on_per_bar_v1.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path("/home/andre2/src/GX1_ENGINE")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gx1.exits.contracts.exit_io_v6_ctx_v3canonical_m1l512 import (
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES as V6_FEATURES,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT as V6_FEATURE_COUNT,
)
from gx1.policy.exit_transformer_v0 import ExitTransformerV0
from gx1.scripts.score_v3_v8_on_per_bar_v1 import (
    TRADE_STATE_FEATURE_NAMES_V6,
    TRADE_STATE_V6_INDICES,
    build_overlay_for_trade,
)


ACTION = "DISTILL_V1_V3_QHEAD"
WINDOW_LEN = 512

DEFAULT_V3_BUNDLE = Path(
    "/home/andre2/GX1_DATA/models/exit_transformer_v0/"
    "EXIT_V9_MULTI_TF_LR5E4_SCALE025_20260513T223544Z"
)
DEFAULT_V3_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3"
)
DEFAULT_PER_BAR_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_V3PLUS_FULL_20260519T012648Z_LOCK_"
    "V3TRACKED_20260519T022946Z_LOCK"
)
DEFAULT_M5_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
OUTPUT_PARENT = Path("/home/andre2/GX1_DATA/models/exit_transformer_v0")

# Train/val split by week stem date — weeks ending before this cutoff = train.
DEFAULT_VAL_CUTOFF_UTC = "2025-09-01T00:00:00+00:00"  # matches V3 v9 manifest


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    temperature: float
    val_cutoff_utc: str
    max_train_bars: int
    max_val_bars: int
    seed: int


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────


def load_v3_with_q_head(bundle_dir: Path, device: torch.device) -> tuple[ExitTransformerV0, dict[str, Any]]:
    """Load V3 v9 multi-TF bundle and patch in enable_q_head=True.

    Returns (model, cfg). q_head is zero-init; backbone weights warmstart
    from the v9 .pt checkpoint via strict=False (q_head keys expected missing).
    """
    cfg_path = bundle_dir / "transformer_config.json"
    state_path = bundle_dir / "exit_transformer_v0.pt"
    cfg = json.loads(cfg_path.read_text())
    if cfg["input_dim"] != V6_FEATURE_COUNT:
        raise ValueError(f"V3 input_dim={cfg['input_dim']} mismatch V6={V6_FEATURE_COUNT}")
    mtf_cfg = cfg.get("multi_tf", {}) or {}
    if not mtf_cfg.get("enabled", False):
        raise RuntimeError(
            f"[V3 LOAD] {bundle_dir.name} is NOT multi-TF. Distill requires v9+ multi-TF bundle."
        )
    mtf_kwargs = dict(
        enable_multi_tf=True,
        m5_seq_dim=int(mtf_cfg["m5_seq_dim"]), m15_seq_dim=int(mtf_cfg["m15_seq_dim"]),
        h1_seq_dim=int(mtf_cfg["h1_seq_dim"]), h4_seq_dim=int(mtf_cfg["h4_seq_dim"]),
        d1_seq_dim=int(mtf_cfg["d1_seq_dim"]),
        m5_seq_len=int(mtf_cfg["m5_seq_len"]), m15_seq_len=int(mtf_cfg["m15_seq_len"]),
        h1_seq_len=int(mtf_cfg["h1_seq_len"]), h4_seq_len=int(mtf_cfg["h4_seq_len"]),
        d1_seq_len=int(mtf_cfg["d1_seq_len"]),
    )
    model = ExitTransformerV0(
        input_dim=cfg["input_dim"], window_len=cfg["window_len"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        dropout=cfg.get("dropout", 0.1),
        enable_q_head=True,
        **mtf_kwargs,
    )
    state_dict = torch.load(state_path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {incompatible.unexpected_keys}")
    expected_missing = {"q_head.weight", "q_head.bias"}
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing:
        raise RuntimeError(f"Missing keys mismatch: got {actual_missing}, expected {expected_missing}")
    return model.to(device), cfg


# ─────────────────────────────────────────────────────────────────────────────
# Multi-TF batch builder (mirrors score_v3_v8_on_per_bar_v1)
# ─────────────────────────────────────────────────────────────────────────────


def build_mtf_batch(
    ts_ns_list: list[int],
    multi_tf_feats: dict[str, pd.DataFrame],
    multi_tf_shift_ns: dict[str, int],
    multi_tf_seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    ts_ns_arr = np.asarray(ts_ns_list, dtype=np.int64)
    B = ts_ns_arr.shape[0]
    out: dict[str, torch.Tensor] = {}
    for tf, feats in multi_tf_feats.items():
        ts_int64 = feats.attrs["ts_int64"]
        feats_np = feats.attrs["feats_np"]
        D = feats_np.shape[1]
        n = multi_tf_seq_len
        cutoffs = ts_ns_arr - multi_tf_shift_ns[tf]
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-bar sample generator (yields (window, ts_ns, q_target) tuples)
# ─────────────────────────────────────────────────────────────────────────────


def iter_bar_samples(
    per_bar_df: pd.DataFrame,
    q_target_df: pd.DataFrame,
    m1_feature_matrix: np.memmap,
    m1_time_ns: np.ndarray,
) -> Any:
    """Yield (window_91x512, bar_ts_ns, q_target_pair) for each bar with a matching Q-target.

    Joins per_bar_df + q_target_df on (candidate_uid, bar_idx_v1). For each
    joined row, builds the 512-bar M1 window and applies the in-trade overlay
    from this candidate's prior bars.
    """
    # Build (uid, bar_idx) → (q_hold, q_exit) lookup
    q_map: dict[tuple[str, int], tuple[float, float]] = {}
    for uid, bidx, qh, qe in zip(
        q_target_df["candidate_uid"].astype(str).to_numpy(),
        q_target_df["bar_idx_v1"].astype(np.int64).to_numpy(),
        q_target_df["iql_exit_q_hold_v1"].astype(np.float32).to_numpy(),
        q_target_df["iql_exit_q_exit_v1"].astype(np.float32).to_numpy(),
    ):
        q_map[(uid, int(bidx))] = (float(qh), float(qe))
    if not q_map:
        return

    # Pre-compute m1_idx_now per row of per_bar_df
    bar_ts_ns = pd.to_numeric(per_bar_df["bar_ts_ns_v1"], errors="coerce").fillna(0).astype("int64").to_numpy()
    m1_idx_now_all = np.searchsorted(m1_time_ns, bar_ts_ns, side="right") - 1
    bars_in_trade_all = pd.to_numeric(per_bar_df["bars_in_trade_v1"], errors="coerce").fillna(0).astype("int64").to_numpy()
    bar_idx_all = pd.to_numeric(per_bar_df["bar_idx_v1"], errors="coerce").fillna(0).astype("int64").to_numpy()
    uid_all = per_bar_df["candidate_uid"].astype(str).to_numpy()

    grouped = per_bar_df.groupby("candidate_uid", sort=False)
    matrix_len = len(m1_feature_matrix)
    for cand_uid, trade_rows in grouped:
        overlay, sorted_rows = build_overlay_for_trade(trade_rows)
        sorted_indices = sorted_rows.index.to_numpy()
        sorted_m1_idx = m1_idx_now_all[sorted_indices]
        sorted_bars_in_trade = bars_in_trade_all[sorted_indices]
        sorted_bar_idx = bar_idx_all[sorted_indices]
        s_t_arr = sorted_m1_idx - sorted_bars_in_trade
        s_t = int(np.median(s_t_arr))

        for i_in_trade, abs_idx in enumerate(sorted_indices):
            key = (str(cand_uid), int(sorted_bar_idx[i_in_trade]))
            q_target = q_map.get(key)
            if q_target is None:
                continue
            mi = int(sorted_m1_idx[i_in_trade])
            if mi < WINDOW_LEN - 1 or mi >= matrix_len:
                continue
            win_start = mi - WINDOW_LEN + 1
            win_end = mi + 1
            io = np.array(m1_feature_matrix[win_start:win_end], dtype=np.float32, copy=True)

            in_trade_start_in_win = max(0, s_t - win_start)
            in_trade_end_in_win = min(WINDOW_LEN, s_t + i_in_trade + 1 - win_start + 1)
            n_in_trade = max(0, in_trade_end_in_win - in_trade_start_in_win)
            if n_in_trade > 0:
                overlay_start_row = max(0, win_start - s_t)
                slice_end = min(overlay_start_row + n_in_trade, len(overlay))
                actual_n = slice_end - overlay_start_row
                if actual_n > 0:
                    io[in_trade_start_in_win: in_trade_start_in_win + actual_n,
                       TRADE_STATE_V6_INDICES] = overlay[overlay_start_row: overlay_start_row + actual_n]

            yield io, int(bar_ts_ns[abs_idx]), q_target


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def run_training(
    model: ExitTransformerV0,
    train_weeks: list[tuple[Path, Path]],
    val_weeks: list[tuple[Path, Path]],
    m1_feature_matrix: np.memmap,
    m1_time_ns: np.ndarray,
    mtf_feats: dict,
    mtf_shift_ns: dict[str, int],
    mtf_seq_len: int,
    cfg: TrainConfig,
    device: torch.device,
    use_bf16: bool = False,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.q_head.parameters(), lr=cfg.lr, weight_decay=0.0)
    T = float(cfg.temperature)
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_bf16 else (lambda: torch.cuda.amp.autocast(enabled=False))

    def _stream_batches(weeks: list[tuple[Path, Path]], shuffle: bool, max_bars: int):
        rng = random.Random(cfg.seed if shuffle else None)
        weeks_iter = list(weeks)
        if shuffle:
            rng.shuffle(weeks_iter)
        seen = 0
        pending_x: list[np.ndarray] = []
        pending_ts: list[int] = []
        pending_q: list[tuple[float, float]] = []

        def flush() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor] | None:
            if not pending_x:
                return None
            x = np.stack(pending_x, axis=0)
            q = np.asarray(pending_q, dtype=np.float32)
            x_t = torch.from_numpy(x).to(device, non_blocking=True)
            mtf_kwargs = build_mtf_batch(pending_ts, mtf_feats, mtf_shift_ns, mtf_seq_len, device)
            q_t = torch.from_numpy(q).to(device, non_blocking=True)
            pending_x.clear(); pending_ts.clear(); pending_q.clear()
            return x_t, mtf_kwargs, q_t

        for week_pb, week_qt in weeks_iter:
            try:
                pb_df = pd.read_parquet(week_pb)
                qt_df = pd.read_parquet(week_qt)
            except FileNotFoundError:
                continue
            # Drop forced_terminal / never_fire bars (match exit-IQL training filter)
            if "is_never_fire_v1" in pb_df.columns:
                pb_df = pb_df[(pb_df.get("is_never_fire_v1", 0) == 0) &
                              (pb_df.get("forced_terminal_v1", 0) == 0)]
            if pb_df.empty or qt_df.empty:
                continue
            # Reset to positional index so per-row numpy arrays (built below)
            # align with sorted_rows.index from groupby.
            pb_df = pb_df.reset_index(drop=True)
            for io, ts_ns, q_pair in iter_bar_samples(
                pb_df, qt_df, m1_feature_matrix, m1_time_ns
            ):
                pending_x.append(io)
                pending_ts.append(ts_ns)
                pending_q.append(q_pair)
                if len(pending_x) >= cfg.batch_size:
                    yield flush()
                    seen += cfg.batch_size
                    if max_bars and seen >= max_bars:
                        return
            del pb_df, qt_df
            gc.collect()
        flushed = flush()
        if flushed is not None:
            yield flushed

    history: list[dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        model.train()
        # Backbone frozen, q_head trainable (set below)
        train_kl_sum = 0.0
        train_n_batches = 0
        t_epoch = time.time()
        for batch in _stream_batches(train_weeks, shuffle=True, max_bars=cfg.max_train_bars):
            x_t, mtf_kwargs, q_target = batch
            with torch.set_grad_enabled(True), autocast_ctx():
                q_pred = model.forward_q_per_action(x_t, **mtf_kwargs)  # (B, 2)
                # KL in fp32 for numerical stability
                q_pred = q_pred.float()
                kl = F.kl_div(
                    F.log_softmax(q_pred / T, dim=-1),
                    F.softmax(q_target / T, dim=-1),
                    reduction="batchmean",
                ) * (T ** 2)
            optimizer.zero_grad()
            kl.backward()
            optimizer.step()
            train_kl_sum += float(kl.item())
            train_n_batches += 1
        train_kl = train_kl_sum / max(1, train_n_batches)

        model.eval()
        val_kl_sum = 0.0
        val_n_batches = 0
        with torch.no_grad():
            for batch in _stream_batches(val_weeks, shuffle=False, max_bars=cfg.max_val_bars):
                x_t, mtf_kwargs, q_target = batch
                with autocast_ctx():
                    q_pred = model.forward_q_per_action(x_t, **mtf_kwargs).float()
                kl = F.kl_div(
                    F.log_softmax(q_pred / T, dim=-1),
                    F.softmax(q_target / T, dim=-1),
                    reduction="batchmean",
                ) * (T ** 2)
                val_kl_sum += float(kl.item())
                val_n_batches += 1
        val_kl = val_kl_sum / max(1, val_n_batches)
        elapsed = (time.time() - t_epoch) / 60.0
        print(f"  epoch {epoch+1}/{cfg.epochs}  train_kl={train_kl:.4f}  val_kl={val_kl:.4f}  "
              f"train_batches={train_n_batches}  val_batches={val_n_batches}  elapsed={elapsed:.1f}min",
              flush=True)
        history.append({
            "epoch": epoch + 1,
            "train_kl": train_kl, "val_kl": val_kl,
            "train_batches": train_n_batches, "val_batches": val_n_batches,
            "elapsed_min": elapsed,
        })
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _week_stem_to_end_date(stem: str) -> pd.Timestamp:
    """exit_per_bar_m1_TRUTH_MONFRI_WEEK_20240101_20240108 → 2024-01-08 UTC."""
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return pd.Timestamp(f"{parts[-1][:4]}-{parts[-1][4:6]}-{parts[-1][6:8]}", tz="UTC")
    return pd.Timestamp("1970-01-01", tz="UTC")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-bundle",        type=Path, default=DEFAULT_V3_BUNDLE)
    ap.add_argument("--v3-dataset-dir",   type=Path, default=DEFAULT_V3_DATASET_DIR)
    ap.add_argument("--per-bar-dir",      type=Path, default=DEFAULT_PER_BAR_DIR)
    ap.add_argument("--exit-q-targets-dir", type=Path, required=True,
                    help="IQL_EXIT_Q_TARGETS_*/ dir (must contain per_week/*.parquet)")
    ap.add_argument("--m5-prebuilt",      type=Path, default=DEFAULT_M5_PREBUILT)
    ap.add_argument("--out-bundle-name",  type=str, default=None)
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--batch-size", type=int,   default=2048,
                    help="2048 fits comfortably in 24GB VRAM with bf16 autocast.")
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--no-bf16",    action="store_true",
                    help="Disable bf16 autocast (default ON for CUDA).")
    ap.add_argument("--no-compile", action="store_true",
                    help="Disable torch.compile on _encode (default ON for CUDA).")
    ap.add_argument("--val-cutoff-utc", type=str, default=DEFAULT_VAL_CUTOFF_UTC)
    ap.add_argument("--max-train-bars", type=int, default=0,
                    help="Cap bars per epoch for fast experiments (0 = no cap).")
    ap.add_argument("--max-val-bars", type=int, default=200_000,
                    help="Cap bars per val pass (default 200K — full coverage of 39 val weeks "
                         "is 7-8M bars and adds ~30 min/epoch).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"[{ACTION}] device={device}", flush=True)

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        temperature=args.temperature, val_cutoff_utc=args.val_cutoff_utc,
        max_train_bars=args.max_train_bars, max_val_bars=args.max_val_bars,
        seed=args.seed,
    )

    # ── Load V3 v9 + patch q_head ────────────────────────────────────
    print(f"[1/5] Loading V3 v9 + patching q_head from {args.v3_bundle.name}", flush=True)
    model, v3_cfg = load_v3_with_q_head(args.v3_bundle, device)
    # Freeze backbone, train only q_head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.q_head.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params (q_head only): {n_trainable:,} / {n_total:,}", flush=True)

    # bf16 autocast + torch.compile on hot path (CUDA only)
    use_bf16 = (device.type == "cuda") and not args.no_bf16
    if device.type == "cuda" and not args.no_compile:
        try:
            model._encode = torch.compile(model._encode, mode="reduce-overhead", dynamic=True)
            print(f"  torch.compile applied to model._encode (reduce-overhead)", flush=True)
        except Exception as exc:
            print(f"  torch.compile failed ({exc!r}) — falling back to eager", flush=True)
    print(f"  bf16 autocast: {use_bf16}", flush=True)

    # ── Load m1 feature matrix + time index (memmap) ─────────────────
    print(f"[2/5] Loading m1_feature_matrix from {args.v3_dataset_dir}", flush=True)
    m1_feature_matrix = np.load(args.v3_dataset_dir / "m1_feature_matrix.npy", mmap_mode="r")
    m1_time_ns = np.load(args.v3_dataset_dir / "m1_time_ns.npy")
    print(f"  m1 matrix shape={m1_feature_matrix.shape}  time_ns="
          f"{m1_time_ns[0]}..{m1_time_ns[-1]}", flush=True)
    if m1_feature_matrix.shape[1] != V6_FEATURE_COUNT:
        raise ValueError(f"matrix dim {m1_feature_matrix.shape[1]} != V6 {V6_FEATURE_COUNT}")

    # ── Build multi-TF features once ─────────────────────────────────
    print(f"[3/5] Building multi-TF features from {args.m5_prebuilt}", flush=True)
    from gx1.features.htf_features import build_multi_tf_per_bar_features, MULTI_TF_SHIFT
    m5 = pd.read_parquet(args.m5_prebuilt, columns=["time", "open", "high", "low", "close"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.set_index("time").sort_index()
    for c in ("open", "high", "low", "close"):
        m5[c] = m5[c].astype(np.float32)
    mtf_feats = build_multi_tf_per_bar_features(m5)
    mtf_shift_ns = {tf: int(td.value) for tf, td in MULTI_TF_SHIFT.items()}
    mtf_seq_len = int(v3_cfg["multi_tf"]["m5_seq_len"])
    del m5
    gc.collect()
    for tf, df in mtf_feats.items():
        print(f"  {tf}: {len(df):,} bars × {df.shape[1]} feats", flush=True)

    # ── Pair per_bar weeks with q-target weeks; split train/val ──────
    print(f"[4/5] Pairing per_bar + Q-target parquets, splitting at {cfg.val_cutoff_utc}", flush=True)
    pb_dir = args.per_bar_dir / "per_week"
    qt_dir = args.exit_q_targets_dir / "per_week"
    pb_files = sorted(pb_dir.glob("exit_per_bar_m1_*.parquet"))
    val_cutoff_ts = pd.Timestamp(cfg.val_cutoff_utc)

    train_weeks: list[tuple[Path, Path]] = []
    val_weeks:   list[tuple[Path, Path]] = []
    missing_qt = 0
    for pb in pb_files:
        week_id = pb.stem.removeprefix("exit_per_bar_m1_")
        qt = qt_dir / f"exit_iql_q_targets_{week_id}.parquet"
        if not qt.exists():
            missing_qt += 1
            continue
        end_date = _week_stem_to_end_date(week_id)
        (val_weeks if end_date >= val_cutoff_ts else train_weeks).append((pb, qt))
    print(f"  pair counts: train={len(train_weeks)}  val={len(val_weeks)}  missing_qt={missing_qt}",
          flush=True)
    if not train_weeks:
        raise RuntimeError("No training weeks paired — check --exit-q-targets-dir is populated.")

    # ── Train ────────────────────────────────────────────────────────
    print(f"[5/5] Training q_head (epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr} T={cfg.temperature})", flush=True)
    history = run_training(
        model=model, train_weeks=train_weeks, val_weeks=val_weeks,
        m1_feature_matrix=m1_feature_matrix, m1_time_ns=m1_time_ns,
        mtf_feats=mtf_feats, mtf_shift_ns=mtf_shift_ns, mtf_seq_len=mtf_seq_len,
        cfg=cfg, device=device, use_bf16=use_bf16,
    )

    # ── Save distilled bundle ────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_name = args.out_bundle_name or f"EXIT_V9_MULTI_TF_DISTILLED_{ts}"
    out_bundle = OUTPUT_PARENT / out_name
    out_bundle.mkdir(parents=True, exist_ok=True)

    new_cfg = dict(v3_cfg)
    new_cfg["enable_q_head"] = True
    new_cfg["distill"] = {
        "warmstart_bundle": str(args.v3_bundle),
        "exit_q_targets_dir": str(args.exit_q_targets_dir),
        "per_bar_dir": str(args.per_bar_dir),
        "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "lr": cfg.lr, "temperature": cfg.temperature,
        "val_cutoff_utc": cfg.val_cutoff_utc, "max_train_bars": cfg.max_train_bars,
        "seed": cfg.seed,
        "history": history,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_weeks": len(train_weeks), "n_val_weeks": len(val_weeks),
    }
    (out_bundle / "transformer_config.json").write_text(json.dumps(new_cfg, indent=2))
    torch.save(model.state_dict(), out_bundle / "exit_transformer_v0.pt")
    # Copy manifest.json from source if present
    src_manifest = args.v3_bundle / "manifest.json"
    if src_manifest.exists():
        man = json.loads(src_manifest.read_text())
        man["distilled_from_v1"] = str(args.v3_bundle)
        man["distill_history_v1"] = history
        (out_bundle / "manifest.json").write_text(json.dumps(man, indent=2))
    (out_bundle / "distill_summary.json").write_text(json.dumps({
        "history": history,
        "final_val_kl": history[-1]["val_kl"] if history else None,
    }, indent=2))

    print(f"\n[{ACTION}] DONE → {out_bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
