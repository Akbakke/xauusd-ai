"""Selective Entry edge evaluator.

This is a research/audit tool for the post-2026-06-27 Entry plan. It does not
train, pin, or mutate bundles. It answers whether model confidence selects a
small subset with positive spread-aware fixed-horizon EV.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.contracts.signal_bridge_v3 import ORDERED_SEQ_FIELDS_V3
from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset,
    _multi_tf_kwargs_from_batch,
)


SESSION_NAMES = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
CLASS_NAMES = {0: "LONG", 1: "SHORT", 2: "FLAT"}
SIGNAL_BRIDGE_NEUTRAL_VALUES = np.array([
    1.0 / 3.0,  # p_long
    1.0 / 3.0,  # p_short
    1.0 / 3.0,  # p_flat
    1.0 / 3.0,  # p_hat
    2.0 / 3.0,  # uncertainty_score = 1 - p_hat
    0.0,        # margin_top1_top2
    math.log(3.0),  # entropy for a uniform 3-class distribution
], dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    path: Path | None = None


@dataclass(frozen=True)
class SessionPrior:
    session_side: dict[str, int]
    session_score: dict[str, float]
    long_score: dict[str, float]
    short_score: dict[str, float]
    global_long_score: float
    global_short_score: float
    global_side: int
    source_split: str


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _parse_model_specs(values: list[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--bundle must be name=/abs/path, got {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"--bundle name is empty in {raw!r}")
        specs.append(ModelSpec(name=name, kind="entry_bundle", path=Path(path).expanduser().resolve()))
    return specs


def _split_files(dataset_dir: Path, splits: Iterable[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split in splits:
        matches = sorted(dataset_dir.glob(f"*_{split}.parquet"))
        if not matches:
            matches = sorted(dataset_dir.glob(f"*{split}*.parquet"))
        if not matches:
            raise FileNotFoundError(f"no parquet for split={split!r} under {dataset_dir}")
        if len(matches) > 1:
            exact = [p for p in matches if p.name.endswith(f"_{split}.parquet")]
            matches = exact or matches
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous parquet files for split={split}: {[str(p) for p in matches]}")
        out[split] = matches[0]
    return out


def _parse_float_list(raw: str) -> list[float]:
    values = [float(x) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise SystemExit("expected at least one float value")
    return values


def _load_base_frame(parquet_path: Path) -> pd.DataFrame:
    cols = [
        "time",
        "ctx_cat",
        "y_direction",
        "y_tradable",
        "label_horizon_bars",
        "mae_first_n_bps",
        "mfe_first_n_bps",
        "path_quality_bps",
        "y_forecast_ret_K24",
    ]
    pf = pq.ParquetFile(parquet_path)
    missing = [c for c in cols if c not in pf.schema_arrow.names]
    if missing:
        raise RuntimeError(f"{parquet_path} missing required columns: {missing}")
    df = pd.read_parquet(parquet_path, columns=[c for c in cols if c != "ctx_cat"])
    ctx_cat = _stack_list_column(pd.read_parquet(parquet_path, columns=["ctx_cat"])["ctx_cat"], np.int64)
    df["session_id"] = ctx_cat[:, 0].astype(int)
    df["session"] = df["session_id"].map(SESSION_NAMES).fillna("UNKNOWN")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _build_session_prior(
    *,
    dataset_dir: Path,
    source_parquet: Path,
    train_split: str,
) -> tuple[SessionPrior, dict[str, Any]]:
    train_file = _split_files(dataset_dir, [train_split])[train_split]
    train_df = _load_base_frame(train_file)
    train_path = _spread_aware_path(
        source_parquet=source_parquet,
        sample_times=train_df["time"],
        horizons=train_df["label_horizon_bars"].to_numpy(np.int64),
    )
    long_pnl = train_path["long_pnl_bps"]
    short_pnl = train_path["short_pnl_bps"]
    sessions = train_df["session"].astype(str).to_numpy()

    global_long = _safe_mean(long_pnl)
    global_short = _safe_mean(short_pnl)
    global_long_score = float(global_long if global_long is not None else 0.0)
    global_short_score = float(global_short if global_short is not None else 0.0)
    global_side = 0 if global_long_score >= global_short_score else 1

    session_side: dict[str, int] = {}
    session_score: dict[str, float] = {}
    long_score: dict[str, float] = {}
    short_score: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for session in sorted(set(sessions) | set(SESSION_NAMES.values())):
        mask = sessions == session
        l_mean = _safe_mean(long_pnl[mask])
        s_mean = _safe_mean(short_pnl[mask])
        l_score = float(l_mean if l_mean is not None else global_long_score)
        s_score = float(s_mean if s_mean is not None else global_short_score)
        side = 0 if l_score >= s_score else 1
        selected_score = l_score if side == 0 else s_score
        session_side[session] = side
        session_score[session] = selected_score
        long_score[session] = l_score
        short_score[session] = s_score
        rows.append({
            "session": session,
            "n": int(mask.sum()),
            "long_mean_pnl_bps": l_score,
            "short_mean_pnl_bps": s_score,
            "selected_side": CLASS_NAMES[side],
            "selected_mean_pnl_bps": selected_score,
        })

    prior = SessionPrior(
        session_side=session_side,
        session_score=session_score,
        long_score=long_score,
        short_score=short_score,
        global_long_score=global_long_score,
        global_short_score=global_short_score,
        global_side=global_side,
        source_split=train_split,
    )
    metadata = {
        "source_split": train_split,
        "source_file": str(train_file),
        "n_train_rows": int(len(train_df)),
        "global_long_mean_pnl_bps": global_long_score,
        "global_short_mean_pnl_bps": global_short_score,
        "global_side": CLASS_NAMES[global_side],
        "sessions": rows,
    }
    return prior, metadata


def _prior_probs_and_score(df: pd.DataFrame, kind: str, prior: SessionPrior) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sessions = df["session"].astype(str).to_numpy()
    n = len(df)
    chosen_side = np.zeros(n, dtype=np.int64)
    score = np.zeros(n, dtype=np.float64)

    if kind == "prior_always_long":
        chosen_side.fill(0)
        for i, session in enumerate(sessions):
            score[i] = prior.long_score.get(session, prior.global_long_score)
    elif kind == "prior_always_short":
        chosen_side.fill(1)
        for i, session in enumerate(sessions):
            score[i] = prior.short_score.get(session, prior.global_short_score)
    elif kind == "prior_session_side":
        for i, session in enumerate(sessions):
            chosen_side[i] = prior.session_side.get(session, prior.global_side)
            score[i] = prior.session_score.get(
                session,
                prior.global_long_score if prior.global_side == 0 else prior.global_short_score,
            )
    else:
        raise RuntimeError(f"unknown prior kind: {kind}")

    probs = np.zeros((n, 3), dtype=np.float64)
    probs[np.arange(n), chosen_side] = 1.0
    return probs, chosen_side, score


def _load_xgb_signal_probs(parquet_path: Path) -> np.ndarray:
    snap = _stack_list_column(pd.read_parquet(parquet_path, columns=["snap"])["snap"], np.float32)
    idx = {name: i for i, name in enumerate(ORDERED_SEQ_FIELDS_V3)}
    probs = snap[:, [idx["p_long"], idx["p_short"], idx["p_flat"]]].astype(np.float64, copy=False)
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.maximum(row_sum, 1e-12))
    return probs


def _device_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _bundle_dataset_kwargs(meta: dict[str, Any], m5_prebuilt_path: Path) -> dict[str, Any]:
    mtf = meta.get("multi_tf") if isinstance(meta.get("multi_tf"), dict) else {}
    if not bool(mtf.get("enabled", False)):
        return {}
    seq_len = int(mtf.get("m15_seq_len", 96))
    return {
        "enable_multi_tf": True,
        "m5_prebuilt_path": m5_prebuilt_path,
        "multi_tf_seq_len": seq_len,
        "per_tf_seq_lens": {
            "M5": int(mtf.get("m5_seq_len", seq_len)),
            "M15": int(mtf.get("m15_seq_len", seq_len)),
            "H1": int(mtf.get("h1_seq_len", seq_len)),
            "H4": int(mtf.get("h4_seq_len", seq_len)),
            "D1": int(mtf.get("d1_seq_len", seq_len)),
        },
    }


def _neutralize_signal_bridge(seq_x: torch.Tensor, snap_x: torch.Tensor) -> None:
    values = torch.as_tensor(SIGNAL_BRIDGE_NEUTRAL_VALUES, dtype=seq_x.dtype, device=seq_x.device)
    seq_x[..., : len(SIGNAL_BRIDGE_NEUTRAL_VALUES)] = values
    snap_x[..., : len(SIGNAL_BRIDGE_NEUTRAL_VALUES)] = values.to(dtype=snap_x.dtype, device=snap_x.device)


def _predict_entry_bundle(
    *,
    bundle_dir: Path,
    parquet_path: Path,
    m5_prebuilt_path: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    neutralize_signal_bridge: bool = False,
) -> np.ndarray:
    bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device=device, xgb_models=None)
    model = bundle.transformer_model
    meta = bundle.metadata
    seq_len = int(meta.get("seq_len") or 96)
    dataset = EntryV10CtxDataset(
        parquet_path=parquet_path,
        seq_len=seq_len,
        allow_constant_labels=True,
        **_bundle_dataset_kwargs(meta, m5_prebuilt_path),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    dev = torch.device(device)
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            non_blocking = dev.type == "cuda"
            seq_x = batch["seq_x"].to(dev, non_blocking=non_blocking)
            snap_x = batch["snap_x"].to(dev, non_blocking=non_blocking)
            ctx_cont = batch["ctx_cont"].to(dev, non_blocking=non_blocking)
            ctx_cat = batch["ctx_cat"].to(dev, non_blocking=non_blocking)
            if neutralize_signal_bridge:
                _neutralize_signal_bridge(seq_x, snap_x)
            out = model(
                seq_x,
                snap_x,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
                **_multi_tf_kwargs_from_batch(batch, dev),
            )
            logits = out["direction_logits"]
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    out_probs = np.concatenate(probs, axis=0).astype(np.float64, copy=False)
    if len(out_probs) != len(dataset):
        raise RuntimeError(f"prediction length mismatch: probs={len(out_probs)} dataset={len(dataset)}")
    return out_probs


def _spread_aware_path(
    *,
    source_parquet: Path,
    sample_times: pd.Series,
    horizons: np.ndarray,
) -> dict[str, np.ndarray]:
    src_cols = ["time", "bid_close", "ask_close", "bid_high", "bid_low", "ask_high", "ask_low"]
    src = pd.read_parquet(source_parquet, columns=src_cols)
    src["time"] = pd.to_datetime(src["time"], utc=True)
    src = src.sort_values("time").reset_index(drop=True)
    src_index = pd.Index(src["time"])
    idx = src_index.get_indexer(pd.to_datetime(sample_times, utc=True))
    if np.any(idx < 0):
        missing = int((idx < 0).sum())
        raise RuntimeError(f"{missing} sample times missing from source parquet")

    bid_close = src["bid_close"].to_numpy(np.float64)
    ask_close = src["ask_close"].to_numpy(np.float64)
    bid_high = src["bid_high"].to_numpy(np.float64)
    bid_low = src["bid_low"].to_numpy(np.float64)
    ask_high = src["ask_high"].to_numpy(np.float64)
    ask_low = src["ask_low"].to_numpy(np.float64)

    n = len(idx)
    long_pnl = np.full(n, np.nan, dtype=np.float64)
    short_pnl = np.full(n, np.nan, dtype=np.float64)
    long_mfe = np.full(n, np.nan, dtype=np.float64)
    long_mae = np.full(n, np.nan, dtype=np.float64)
    short_mfe = np.full(n, np.nan, dtype=np.float64)
    short_mae = np.full(n, np.nan, dtype=np.float64)
    for i, start in enumerate(idx.astype(int)):
        h = int(horizons[i])
        end = start + h
        if h <= 0 or end >= len(src):
            continue
        entry_ask = ask_close[start]
        entry_bid = bid_close[start]
        if entry_ask <= 0 or entry_bid <= 0:
            continue
        fut = slice(start + 1, end + 1)
        long_pnl[i] = (bid_close[end] - entry_ask) / entry_ask * 1e4
        short_pnl[i] = (entry_bid - ask_close[end]) / entry_bid * 1e4
        long_mfe[i] = (np.nanmax(bid_high[fut]) - entry_ask) / entry_ask * 1e4
        long_mae[i] = (entry_ask - np.nanmin(bid_low[fut])) / entry_ask * 1e4
        short_mfe[i] = (entry_bid - np.nanmin(ask_low[fut])) / entry_bid * 1e4
        short_mae[i] = (np.nanmax(ask_high[fut]) - entry_bid) / entry_bid * 1e4
    return {
        "long_pnl_bps": long_pnl,
        "short_pnl_bps": short_pnl,
        "long_mfe_bps": long_mfe,
        "long_mae_bps": long_mae,
        "short_mfe_bps": short_mfe,
        "short_mae_bps": short_mae,
    }


def _safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.mean())


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _metrics_row(
    *,
    split: str,
    model_name: str,
    scope: str,
    top_frac: float | None,
    extra_cost_bps: float,
    group: str,
    mask: np.ndarray,
    total_n: int,
    labels: np.ndarray,
    chosen_side: np.ndarray,
    score: np.ndarray,
    probs: np.ndarray,
    pnl_bps: np.ndarray,
    mfe_bps: np.ndarray,
    mae_bps: np.ndarray,
    path_quality_bps: np.ndarray,
) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {
            "split": split,
            "model": model_name,
            "scope": scope,
            "top_frac": top_frac,
            "extra_cost_bps": extra_cost_bps,
            "group": group,
            "n": 0,
            "coverage": 0.0,
        }
    y = labels[mask]
    side = chosen_side[mask]
    p = probs[mask]
    pnl = pnl_bps[mask]
    wins = pnl > 0.0
    finite_pnl = np.isfinite(pnl)
    correct = y == side
    return {
        "split": split,
        "model": model_name,
        "scope": scope,
        "top_frac": top_frac,
        "extra_cost_bps": extra_cost_bps,
        "group": group,
        "n": n,
        "coverage": float(n / max(total_n, 1)),
        "label_long_rate": float((y == 0).mean()),
        "label_short_rate": float((y == 1).mean()),
        "label_flat_rate": float((y == 2).mean()),
        "chosen_long_rate": float((side == 0).mean()),
        "chosen_short_rate": float((side == 1).mean()),
        "direction_precision": float(correct.mean()),
        "win_rate": float(wins[finite_pnl].mean()) if finite_pnl.any() else None,
        "mean_pnl_bps": _safe_mean(pnl),
        "median_pnl_bps": _safe_percentile(pnl, 50),
        "p10_pnl_bps": _safe_percentile(pnl, 10),
        "p90_pnl_bps": _safe_percentile(pnl, 90),
        "sum_pnl_bps": float(np.nansum(pnl)),
        "mean_mfe_bps": _safe_mean(mfe_bps[mask]),
        "mean_mae_bps": _safe_mean(mae_bps[mask]),
        "mean_path_quality_bps": _safe_mean(path_quality_bps[mask]),
        "mean_score": _safe_mean(score[mask]),
        "mean_p_long": _safe_mean(p[:, 0]),
        "mean_p_short": _safe_mean(p[:, 1]),
        "mean_p_flat": _safe_mean(p[:, 2]),
    }


def _evaluate_predictions(
    *,
    split: str,
    model_name: str,
    df: pd.DataFrame,
    probs: np.ndarray,
    path: dict[str, np.ndarray],
    top_fracs: list[float],
    extra_cost_bps_values: list[float],
    chosen_side_override: np.ndarray | None = None,
    score_override: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(df) != len(probs):
        raise RuntimeError(f"{model_name}/{split} row mismatch: df={len(df)} probs={len(probs)}")
    labels = df["y_direction"].to_numpy(np.int64)
    argmax_pred = probs.argmax(axis=1).astype(np.int64)
    directional_prob = np.maximum(probs[:, 0], probs[:, 1])
    chosen_side = (
        np.asarray(chosen_side_override, dtype=np.int64)
        if chosen_side_override is not None
        else np.where(probs[:, 0] >= probs[:, 1], 0, 1).astype(np.int64)
    )
    score = (
        np.asarray(score_override, dtype=np.float64)
        if score_override is not None
        else directional_prob - probs[:, 2]
    )
    if len(chosen_side) != len(df) or len(score) != len(df):
        raise RuntimeError(f"{model_name}/{split} override length mismatch")
    base_pnl_bps = np.where(chosen_side == 0, path["long_pnl_bps"], path["short_pnl_bps"])
    mfe_bps = np.where(chosen_side == 0, path["long_mfe_bps"], path["short_mfe_bps"])
    mae_bps = np.where(chosen_side == 0, path["long_mae_bps"], path["short_mae_bps"])
    path_quality = df["path_quality_bps"].to_numpy(np.float64)
    total_n = len(df)

    rows: list[dict[str, Any]] = []
    rows.append({
        "split": split,
        "model": model_name,
        "scope": "full_argmax",
        "top_frac": 1.0,
        "extra_cost_bps": 0.0,
        "group": "ALL",
        "n": total_n,
        "accuracy": float((argmax_pred == labels).mean()),
        "pred_long_rate": float((argmax_pred == 0).mean()),
        "pred_short_rate": float((argmax_pred == 1).mean()),
        "pred_flat_rate": float((argmax_pred == 2).mean()),
        "label_long_rate": float((labels == 0).mean()),
        "label_short_rate": float((labels == 1).mean()),
        "label_flat_rate": float((labels == 2).mean()),
        "mean_p_long": _safe_mean(probs[:, 0]),
        "mean_p_short": _safe_mean(probs[:, 1]),
        "mean_p_flat": _safe_mean(probs[:, 2]),
    })

    order = np.argsort(-score, kind="mergesort")
    sessions = df["session"].astype(str).to_numpy()
    for extra_cost_bps in extra_cost_bps_values:
        pnl_bps = base_pnl_bps - float(extra_cost_bps)
        for frac in top_fracs:
            k = max(1, int(math.ceil(total_n * float(frac))))
            selected = np.zeros(total_n, dtype=bool)
            selected[order[:k]] = True
            groups: list[tuple[str, np.ndarray]] = [("ALL", selected)]
            groups.extend([
                ("side=LONG", selected & (chosen_side == 0)),
                ("side=SHORT", selected & (chosen_side == 1)),
            ])
            for session in ["ASIA", "EU", "OVERLAP", "US"]:
                sm = selected & (sessions == session)
                groups.append((f"session={session}", sm))
                groups.append((f"session={session}|side=LONG", sm & (chosen_side == 0)))
                groups.append((f"session={session}|side=SHORT", sm & (chosen_side == 1)))
            for group_name, mask in groups:
                rows.append(_metrics_row(
                    split=split,
                    model_name=model_name,
                    scope="top_score",
                    top_frac=float(frac),
                    extra_cost_bps=float(extra_cost_bps),
                    group=group_name,
                    mask=mask,
                    total_n=total_n,
                    labels=labels,
                    chosen_side=chosen_side,
                    score=score,
                    probs=probs,
                    pnl_bps=pnl_bps,
                    mfe_bps=mfe_bps,
                    mae_bps=mae_bps,
                    path_quality_bps=path_quality,
                ))

    summary = {
        "split": split,
        "model": model_name,
        "full_argmax_accuracy": float((argmax_pred == labels).mean()),
        "top_score_all": [],
        "top1_all_mean_pnl_bps": None,
        "top2_all_mean_pnl_bps": None,
        "top5_all_mean_pnl_bps": None,
        "top10_all_mean_pnl_bps": None,
        "top20_all_mean_pnl_bps": None,
    }
    frac_keys = {
        0.01: "top1_all_mean_pnl_bps",
        0.02: "top2_all_mean_pnl_bps",
        0.05: "top5_all_mean_pnl_bps",
        0.10: "top10_all_mean_pnl_bps",
        0.20: "top20_all_mean_pnl_bps",
    }
    for frac in top_fracs:
        candidates = [
            r for r in rows
            if r.get("scope") == "top_score"
            and math.isclose(float(r.get("top_frac", -1.0)), float(frac), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(r.get("extra_cost_bps", -1.0)), 0.0, rel_tol=0.0, abs_tol=1e-12)
            and r.get("group") == "ALL"
        ]
        if candidates:
            row = candidates[0]
            summary["top_score_all"].append({
                "top_frac": row.get("top_frac"),
                "extra_cost_bps": row.get("extra_cost_bps"),
                "n": row.get("n"),
                "coverage": row.get("coverage"),
                "direction_precision": row.get("direction_precision"),
                "win_rate": row.get("win_rate"),
                "mean_pnl_bps": row.get("mean_pnl_bps"),
                "median_pnl_bps": row.get("median_pnl_bps"),
                "mean_mfe_bps": row.get("mean_mfe_bps"),
                "mean_mae_bps": row.get("mean_mae_bps"),
            })
            for known_frac, key in frac_keys.items():
                if math.isclose(float(frac), known_frac, rel_tol=0.0, abs_tol=1e-12):
                    summary[key] = row.get("mean_pnl_bps")
    summary["cost_stress_all"] = [
        {
            "top_frac": r.get("top_frac"),
            "extra_cost_bps": r.get("extra_cost_bps"),
            "n": r.get("n"),
            "coverage": r.get("coverage"),
            "direction_precision": r.get("direction_precision"),
            "win_rate": r.get("win_rate"),
            "mean_pnl_bps": r.get("mean_pnl_bps"),
            "median_pnl_bps": r.get("median_pnl_bps"),
        }
        for r in rows
        if r.get("scope") == "top_score" and r.get("group") == "ALL"
    ]
    return rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    top_fracs = _parse_float_list(args.top_fracs)
    extra_cost_bps_values = _parse_float_list(args.extra_cost_bps)
    files = _split_files(dataset_dir, splits)
    device = _device_arg(args.device)
    models: list[ModelSpec] = []
    if args.include_xgb_only:
        models.append(ModelSpec(name="xgb_signal7", kind="xgb_signal7"))
    bundle_specs = _parse_model_specs(args.bundle or [])
    models.extend(bundle_specs)
    if args.include_no_xgb_ablation:
        models.extend([
            ModelSpec(name=f"{spec.name}_no_xgb", kind="entry_bundle_no_xgb", path=spec.path)
            for spec in bundle_specs
        ])
    prior: SessionPrior | None = None
    prior_metadata: dict[str, Any] | None = None
    if args.include_priors:
        prior, prior_metadata = _build_session_prior(
            dataset_dir=dataset_dir,
            source_parquet=source_parquet,
            train_split=str(args.prior_train_split),
        )
        models.extend([
            ModelSpec(name="prior_always_long", kind="prior_always_long"),
            ModelSpec(name="prior_always_short", kind="prior_always_short"),
            ModelSpec(name="prior_session_side", kind="prior_session_side"),
        ])
    if not models:
        raise SystemExit("provide --include-xgb-only, --include-priors, and/or --bundle name=/path")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split, parquet_path in files.items():
        df = _load_base_frame(parquet_path)
        path = _spread_aware_path(
            source_parquet=source_parquet,
            sample_times=df["time"],
            horizons=df["label_horizon_bars"].to_numpy(np.int64),
        )
        for spec in models:
            chosen_side_override = None
            score_override = None
            if spec.kind == "xgb_signal7":
                probs = _load_xgb_signal_probs(parquet_path)
            elif spec.kind in ("entry_bundle", "entry_bundle_no_xgb"):
                assert spec.path is not None
                probs = _predict_entry_bundle(
                    bundle_dir=spec.path,
                    parquet_path=parquet_path,
                    m5_prebuilt_path=Path(args.m5_prebuilt_path).expanduser().resolve(),
                    device=device,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    neutralize_signal_bridge=spec.kind == "entry_bundle_no_xgb",
                )
            elif spec.kind.startswith("prior_"):
                if prior is None:
                    raise RuntimeError(f"{spec.kind} requested without prior metadata")
                probs, chosen_side_override, score_override = _prior_probs_and_score(df, spec.kind, prior)
            else:
                raise RuntimeError(f"unknown model kind: {spec.kind}")
            rows, summary = _evaluate_predictions(
                split=split,
                model_name=spec.name,
                df=df,
                probs=probs,
                path=path,
                top_fracs=top_fracs,
                extra_cost_bps_values=extra_cost_bps_values,
                chosen_side_override=chosen_side_override,
                score_override=score_override,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    metrics = pd.DataFrame(all_rows)
    metrics_path = out_dir / "selective_edge_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = {
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "out_dir": str(out_dir),
        "splits": splits,
        "top_fracs": top_fracs,
        "extra_cost_bps": extra_cost_bps_values,
        "models": [spec.__dict__ for spec in models],
        "no_xgb_ablation": {
            "enabled": bool(args.include_no_xgb_ablation),
            "neutralized_fields": ORDERED_SEQ_FIELDS_V3[:len(SIGNAL_BRIDGE_NEUTRAL_VALUES)],
            "neutral_values": SIGNAL_BRIDGE_NEUTRAL_VALUES.tolist(),
        },
        "prior_metadata": prior_metadata,
        "metrics_csv": str(metrics_path),
        "summaries": summaries,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--source-parquet", required=True, help="FULL_PLUS_CTX parquet with bid/ask OHLC")
    ap.add_argument("--m5-prebuilt-path", required=True, help="Canonical V3 M5 parquet for Entry bundle MTF eval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--top-fracs", default="0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--extra-cost-bps", default="0.0", help="Comma-separated extra per-trade costs after spread")
    ap.add_argument("--include-xgb-only", action="store_true")
    ap.add_argument("--include-priors", action="store_true")
    ap.add_argument("--include-no-xgb-ablation", action="store_true")
    ap.add_argument("--prior-train-split", default="train")
    ap.add_argument("--bundle", action="append", default=[], help="name=/abs/bundle_dir; repeatable")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
