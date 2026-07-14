#!/usr/bin/env python3
"""Row-level forensic audit for XAU smart direction decisions.

This is a research/launch-blocking audit, not a live trading rule. It joins a
smart520 prediction parquet with the emitted training/replay dataset and writes
selected rows with component evidence: labels, side utilities, path quality,
anchor/final probabilities, geometry/support/resistance and MTF fields.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260626_spreadfix/"
    "v10_dataset_6yr_smartctx_xau_direction_repair"
)
DEFAULT_PREDICTIONS = Path(
    "/home/andre2/GX1_DATA/reports/entry_candidate_selective_edge_20260628_v1/"
    "smart_seq520_xau_direction_repair/selective_edge_predictions.parquet"
)
DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/"
    "v10_entry_smart_seq520_xau_direction_repair_candidate"
)
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/smart_direction_contribution_audit_v1")

SIDE_NAME = {0: "LONG", 1: "SHORT", 2: "FLAT"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_parquet(dataset_dir: Path, split: str) -> Path:
    matches = sorted(dataset_dir.glob(f"*_{split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one *_{split}.parquet in {dataset_dir}, got {matches}")
    return matches[0]


def _schema_cols(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def _matrix(series: pd.Series) -> np.ndarray:
    return np.asarray([np.asarray(x, dtype=np.float32) for x in series.to_numpy()], dtype=np.float32)


def _first(
    matrix: np.ndarray,
    names: list[str],
    candidates: list[str],
    default: float = 0.0,
    *,
    required: bool = True,
) -> np.ndarray:
    for name in candidates:
        if name in names:
            idx = int(names.index(name))
            if idx >= matrix.shape[1]:
                raise RuntimeError(f"feature {name!r} index {idx} outside matrix width {matrix.shape[1]}")
            return matrix[:, idx].astype(np.float32)
    if required:
        raise RuntimeError(f"required feature candidates missing: {candidates}")
    return np.full(matrix.shape[0], float(default), dtype=np.float32)


def _safe_num(frame: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), float(default), dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)


def _is_expected_utility_mode(frame: pd.DataFrame) -> bool:
    if "selection_score_mode" not in frame.columns:
        return False
    mode = frame["selection_score_mode"].astype(str).str.lower()
    return bool(mode.eq("expected_utility").any())


def _expected_utility_side(frame: pd.DataFrame) -> np.ndarray:
    missing = [name for name in ("expected_utility_long_bps", "expected_utility_short_bps") if name not in frame.columns]
    if missing:
        raise RuntimeError(f"expected_utility predictions require side utility columns for contribution audit: {missing}")
    expected_long = pd.to_numeric(frame["expected_utility_long_bps"], errors="coerce").to_numpy(dtype=np.float64)
    expected_short = pd.to_numeric(frame["expected_utility_short_bps"], errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(expected_long).all() and np.isfinite(expected_short).all()):
        raise RuntimeError("expected_utility side columns contain non-finite values for contribution audit")
    return np.where(expected_long >= expected_short, 0, 1).astype(np.int8)


def _nanmean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.full(0, np.nan, dtype=np.float64)
    stack = np.vstack([np.asarray(arr, dtype=np.float64) for arr in arrays])
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    sums = np.where(finite, stack, 0.0).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _chosen_side(frame: pd.DataFrame) -> np.ndarray:
    if _is_expected_utility_mode(frame):
        return _expected_utility_side(frame)
    if "trade_side" in frame.columns:
        raw = pd.to_numeric(frame["trade_side"], errors="coerce").fillna(2).to_numpy(dtype=np.int16)
        out = np.where(np.isin(raw, [0, 1]), raw, 2).astype(np.int8)
        return out
    p_long = _safe_num(frame, "p_long", default=0.0)
    p_short = _safe_num(frame, "p_short", default=0.0)
    side = np.full(len(frame), 2, dtype=np.int8)
    side[p_long >= p_short] = 0
    side[p_short > p_long] = 1
    return side


def _selected(frame: pd.DataFrame, edge_threshold: float) -> np.ndarray:
    expected_utility_mode = _is_expected_utility_mode(frame)
    if expected_utility_mode:
        if "selection_score" not in frame.columns:
            raise RuntimeError("expected_utility predictions require selection_score for contribution audit")
        score = pd.to_numeric(frame["selection_score"], errors="coerce").to_numpy(dtype=np.float64)
        if "selection_score_threshold" in frame.columns:
            threshold = pd.to_numeric(frame["selection_score_threshold"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            threshold = np.zeros(len(frame), dtype=np.float64)
        return score >= threshold
    if "selected" in frame.columns:
        return pd.Series(frame["selected"]).fillna(False).astype(bool).to_numpy()
    if "action" in frame.columns:
        action = frame["action"].astype(str).str.upper()
        return action.isin(["TAKE_LONG_NOW", "TAKE_SHORT_NOW", "LONG", "SHORT"]).to_numpy()
    if "selection_score" in frame.columns and ("selection_score_threshold" in frame.columns or expected_utility_mode):
        score = pd.to_numeric(frame["selection_score"], errors="coerce").to_numpy(dtype=np.float64)
        if "selection_score_threshold" in frame.columns:
            threshold = pd.to_numeric(frame["selection_score_threshold"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            threshold = np.zeros(len(frame), dtype=np.float64)
        return score >= threshold
    edge = _safe_num(frame, "edge_score", default=-np.inf)
    return edge >= float(edge_threshold)


def _expected_utility_contract_failures(frame: pd.DataFrame) -> list[str]:
    if not _is_expected_utility_mode(frame):
        return []
    required = (
        "selection_score",
        "selection_score_threshold",
        "trade_side",
        "expected_utility_long_bps",
        "expected_utility_short_bps",
        "expected_utility_side",
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        return [f"expected_utility contribution audit missing prediction columns: {missing}"]
    failures: list[str] = []
    utility_side = _expected_utility_side(frame)
    expected_side = pd.to_numeric(frame["expected_utility_side"], errors="coerce").fillna(2).to_numpy(dtype=np.int16)
    expected_side_mismatches = int(np.sum(expected_side != utility_side))
    if expected_side_mismatches:
        failures.append(
            "expected_utility contribution audit expected_utility_side mismatches long/short utility: "
            f"mismatches={expected_side_mismatches}"
        )
    trade_side = pd.to_numeric(frame["trade_side"], errors="coerce").fillna(2).to_numpy(dtype=np.int16)
    trade_side_mismatches = int(np.sum(trade_side != utility_side))
    if trade_side_mismatches:
        failures.append(
            "expected_utility contribution audit trade_side mismatches long/short utility side: "
            f"mismatches={trade_side_mismatches}"
        )
    score = pd.to_numeric(frame["selection_score"], errors="coerce").to_numpy(dtype=np.float64)
    threshold = pd.to_numeric(frame["selection_score_threshold"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    score_selected = score >= threshold
    if "selected" in frame.columns:
        legacy_selected = pd.Series(frame["selected"]).fillna(False).astype(bool).to_numpy()
        selected_mismatches = int(np.sum(legacy_selected != score_selected))
        if selected_mismatches:
            failures.append(
                "expected_utility contribution audit selected column mismatches selection_score threshold: "
                f"mismatches={selected_mismatches}"
            )
    if "action" in frame.columns:
        action = frame["action"].astype(str).str.upper()
        action_side = np.full(len(frame), 2, dtype=np.int8)
        action_side[action.isin(["TAKE_LONG_NOW", "LONG"])] = 0
        action_side[action.isin(["TAKE_SHORT_NOW", "SHORT"])] = 1
        action_selected = action_side != 2
        selected_mismatches = int(np.sum(action_selected != score_selected))
        if selected_mismatches:
            failures.append(
                "expected_utility contribution audit action column mismatches selection_score threshold: "
                f"mismatches={selected_mismatches}"
            )
        side_mismatches = int(np.sum(action_selected & (action_side != utility_side)))
        if side_mismatches:
            failures.append(
                "expected_utility contribution audit action side mismatches long/short utility side: "
                f"mismatches={side_mismatches}"
            )
    return failures


def _fmt_side(side: int) -> str:
    return SIDE_NAME.get(int(side), f"UNKNOWN_{side}")


def _vector_gap(frame: pd.DataFrame, col: str, left: int = 0, right: int = 1) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float64)
    out = np.full(len(frame), np.nan, dtype=np.float64)
    for i, value in enumerate(frame[col].to_numpy()):
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if len(arr) > max(left, right):
                out[i] = float(arr[left] - arr[right])
        except Exception:
            continue
    return out


def _assert_prediction_provenance(pred_path: Path, bundle_dir: Path, dataset_dir: Path) -> list[str]:
    failures: list[str] = []
    latest = pred_path.parent / "ENTRY_CANDIDATE_SELECTIVE_EDGE_latest.json"
    if not latest.is_file():
        return [f"missing selective-edge provenance JSON next to predictions: {latest}"]
    try:
        meta = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not read selective-edge provenance JSON {latest}: {exc}"]
    bundle_raw = str(meta.get("bundle_dir") or "").strip()
    dataset_raw = str(meta.get("dataset_dir") or "").strip()
    if not bundle_raw:
        failures.append(f"prediction provenance lacks bundle_dir: {latest}")
    else:
        expected_bundle = Path(bundle_raw).expanduser()
        if expected_bundle.resolve() != bundle_dir.resolve():
            failures.append(
                f"prediction bundle_dir mismatch: provenance={expected_bundle} audit={bundle_dir}"
            )
    if not dataset_raw:
        failures.append(f"prediction provenance lacks dataset_dir: {latest}")
    else:
        expected_dataset = Path(dataset_raw).expanduser()
        if expected_dataset.resolve() != dataset_dir.resolve():
            failures.append(
                f"prediction dataset_dir mismatch: provenance={expected_dataset} audit={dataset_dir}"
            )
    return failures


def _why(row: pd.Series) -> str:
    side = _fmt_side(int(row["selected_side"]))
    return (
        f"{side}: p_long={row['p_long']:.4f} p_short={row['p_short']:.4f} p_flat={row['p_flat']:.4f} "
        f"p_trade={row['p_trade']:.4f} pL|T={row['p_long_given_trade']:.4f} pS|T={row['p_short_given_trade']:.4f} "
        f"edge={row['edge_score']:.4f}; anchor_gap={row['anchor_long_minus_short']:.4f} "
        f"delta_gap={row['delta_long_minus_short']:.4f} mtf_gap={row['mtf_long_minus_short']:.4f}; "
        f"trend={row['mtf_trend_evidence']:.3f} support={row['support_evidence']:.3f} "
        f"resistance={row['resistance_evidence']:.3f} channel_edge={row['channel_edge_pressure']:.3f} "
        f"rail_LS={row['trendline_rail_long_minus_short']:.3f}; "
        f"euL={row['expected_utility_long_bps']:.2f} euS={row['expected_utility_short_bps']:.2f}; "
        f"long_util={row['y_long_path_utility_bps']:.2f} short_util={row['y_short_path_utility_bps']:.2f} "
        f"mae={row['mae_first_n_bps']:.2f} mfe={row['mfe_first_n_bps']:.2f}"
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SMART Direction Contribution Audit",
        "",
        f"decision: {report['decision']}",
        f"created_utc: {report['created_utc']}",
        f"bundle_dir: {report['bundle_dir']}",
        f"dataset: {report['dataset_parquet']}",
        f"predictions: {report['predictions_parquet']}",
        f"rows_joined: {report['rows_joined']}",
        f"selected_rows: {report['selected_rows']}",
        "",
        "## Focus",
    ]
    for name, count in report["focus_counts"].items():
        lines.append(f"- {name}: {count}")
    if report.get("failures"):
        lines.extend(["", "## Failures", *[f"- {x}" for x in report["failures"]]])
    lines.extend(
        [
            "",
            "## Outputs",
            f"- row_report_csv: {report['row_report_csv']}",
            f"- row_report_parquet: {report['row_report_parquet']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--predictions-parquet", type=Path, default=DEFAULT_PREDICTIONS)
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--edge-threshold", type=float, default=0.145)
    ap.add_argument("--focus-start", default="2026-07-08T18:00:00Z")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    meta_path = bundle_dir / "bundle_metadata.json"
    if not meta_path.is_file():
        raise RuntimeError(f"missing bundle metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    signal_names = [str(x) for x in (meta.get("ordered_signal_names") or [])]
    ctx_cont_names = [str(x) for x in (meta.get("ordered_ctx_cont_names") or [])]
    if not signal_names:
        raise RuntimeError("bundle metadata lacks ordered_signal_names")

    dataset_parquet = _split_parquet(args.dataset_dir.expanduser().resolve(), args.split)
    dataset_dir = args.dataset_dir.expanduser().resolve()
    pred_path = args.predictions_parquet.expanduser().resolve()
    failures: list[str] = []
    failures.extend(_assert_prediction_provenance(pred_path, bundle_dir, dataset_dir))
    ds_optional = [
        "y_direction_long_score_bps",
        "y_direction_short_score_bps",
        "y_long_path_utility_bps",
        "y_short_path_utility_bps",
        "y_long_bad_path",
        "y_short_bad_path",
        "y_long_expected_mae_bps",
        "y_short_expected_mae_bps",
        "y_rising_channel_support_touch",
        "y_falling_channel_resistance_touch",
        "y_countertrend_short_trap",
        "y_countertrend_long_trap",
        "y_mtf_conflict_m5_vs_higher_side",
    ]
    ds_required = [
        "time",
        "snap",
        "ctx_cont",
        "y_direction",
        "y_tradable",
        "mae_first_n_bps",
        "mfe_first_n_bps",
        "path_quality_bps",
    ]
    ds_cols = _schema_cols(dataset_parquet)
    read_ds_cols = ds_required + [c for c in ds_optional if c in ds_cols]
    ds = pd.read_parquet(dataset_parquet, columns=read_ds_cols)
    ds["time"] = pd.to_datetime(ds["time"], utc=True)

    pred_cols = _schema_cols(pred_path)
    pred_base = ["time", "p_long", "p_short", "p_flat", "edge_score"]
    pred_optional = [
        "split",
        "model",
        "selected",
        "trade_side",
        "action",
        "p_trade",
        "p_flat_hier",
        "p_long_given_trade",
        "p_short_given_trade",
        "expected_utility_long_bps",
        "expected_utility_short_bps",
        "expected_utility_side",
        "long_path_utility_pred_bps",
        "short_path_utility_pred_bps",
        "long_bad_path_prob",
        "short_bad_path_prob",
        "long_expected_mae_bps",
        "short_expected_mae_bps",
        "anchor_logits",
        "delta_logits",
        "mtf_dir_logits",
        "mtf_p_long",
        "mtf_p_short",
        "mtf_p_flat",
        "mtf_long_minus_short",
        "selection_score_mode",
        "selection_score",
        "selection_score_threshold",
    ]
    missing_pred = [c for c in pred_base if c not in pred_cols]
    if missing_pred:
        raise RuntimeError(f"predictions parquet missing required cols: {missing_pred}")
    pred = pd.read_parquet(pred_path, columns=pred_base + [c for c in pred_optional if c in pred_cols])
    pred["time"] = pd.to_datetime(pred["time"], utc=True)
    if "split" in pred.columns:
        pred = pred[pred["split"] == args.split].copy()
    if "model" in pred.columns:
        pred = pred[pred["model"] == args.model_name].copy()

    for name, data in (("predictions", pred), ("dataset", ds)):
        dup = data["time"].duplicated(keep=False)
        if bool(dup.any()):
            raise RuntimeError(f"{name} has duplicate time rows, first={data.loc[dup, 'time'].head(5).tolist()}")
    frame = pred.merge(ds, on="time", how="outer", suffixes=("_pred", ""), indicator=True, validate="one_to_one")
    coverage = frame["_merge"].value_counts().to_dict()
    if coverage.get("left_only", 0) or coverage.get("right_only", 0):
        failures.append(f"prediction/dataset time coverage mismatch: {coverage}")
    frame = frame.loc[frame["_merge"] == "both"].drop(columns=["_merge"]).copy()
    if frame.empty:
        raise RuntimeError("empty prediction/dataset join")

    snap = _matrix(frame["snap"])
    ctx_cont = _matrix(frame["ctx_cont"])
    p_long = _safe_num(frame, "p_long", default=0.0)
    p_short = _safe_num(frame, "p_short", default=0.0)
    p_flat = _safe_num(frame, "p_flat", default=0.0)
    edge = _safe_num(frame, "edge_score", default=np.nan)
    eps = 1e-9
    final_log_long = np.log(np.clip(p_long, eps, 1.0))
    final_log_short = np.log(np.clip(p_short, eps, 1.0))
    final_log_flat = np.log(np.clip(p_flat, eps, 1.0))

    anchor_p_long = _first(snap, signal_names, ["p_long", "xgb.p_long", "signal_bridge.p_long"], default=np.nan)
    anchor_p_short = _first(snap, signal_names, ["p_short", "xgb.p_short", "signal_bridge.p_short"], default=np.nan)
    anchor_p_flat = _first(snap, signal_names, ["p_flat", "xgb.p_flat", "signal_bridge.p_flat"], default=np.nan)
    anchor_log_long = np.log(np.clip(anchor_p_long, eps, 1.0))
    anchor_log_short = np.log(np.clip(anchor_p_short, eps, 1.0))
    anchor_gap = _vector_gap(frame, "anchor_logits")
    anchor_gap = np.where(np.isfinite(anchor_gap), anchor_gap, anchor_log_long - anchor_log_short)
    delta_gap = _vector_gap(frame, "delta_logits")

    support = _first(
        snap,
        signal_names,
        [
            "chart.geometry_support_line_proximity_stack",
            "chart.sr_memory_support_level_proximity_stack",
            "chart.sr_memory_support_respect_pressure_long",
        ],
    )
    resistance = _first(
        snap,
        signal_names,
        [
            "chart.geometry_resistance_line_proximity_stack",
            "chart.sr_memory_resistance_level_proximity_stack",
            "chart.sr_memory_resistance_respect_pressure_short",
        ],
    )
    channel_edge = _first(snap, signal_names, ["chart.geometry_channel_edge_pressure"])
    rising_support_rail_long = _first(
        snap,
        signal_names,
        ["chart.geometry_rising_support_rail_long_pressure"],
        required=False,
        default=np.nan,
    )
    rising_support_rail_short_trap = _first(
        snap,
        signal_names,
        ["chart.geometry_rising_support_rail_short_trap_pressure"],
        required=False,
        default=np.nan,
    )
    falling_resistance_rail_short = _first(
        snap,
        signal_names,
        ["chart.geometry_falling_resistance_rail_short_pressure"],
        required=False,
        default=np.nan,
    )
    falling_resistance_rail_long_trap = _first(
        snap,
        signal_names,
        ["chart.geometry_falling_resistance_rail_long_trap_pressure"],
        required=False,
        default=np.nan,
    )
    trendline_rail_long_evidence = _nanmean_stack([rising_support_rail_long, falling_resistance_rail_long_trap])
    trendline_rail_short_evidence = _nanmean_stack([falling_resistance_rail_short, rising_support_rail_short_trap])
    trendline_rail_long_minus_short = trendline_rail_long_evidence - trendline_rail_short_evidence
    mtf_trend = _first(snap, signal_names, ["trend.mtf_confluence_trend_direction_score", "trend.ema_stack_alignment_score"])
    mtf_conflict = _first(snap, signal_names, ["trend.mtf_confluence_trend_tf_conflict"], required=False, default=np.nan)
    d1_slope = _first(ctx_cont, ctx_cont_names, ["d1_ema_slope_20_canon_v2"])
    mtf_gap = _safe_num(frame, "mtf_long_minus_short", default=np.nan)
    raw_mtf_gap = _vector_gap(frame, "mtf_dir_logits")
    mtf_gap = np.where(np.isfinite(mtf_gap), mtf_gap, raw_mtf_gap)
    if "mtf_p_long" in frame.columns and "mtf_p_short" in frame.columns:
        mtf_p_long = _safe_num(frame, "mtf_p_long", default=np.nan)
        mtf_p_short = _safe_num(frame, "mtf_p_short", default=np.nan)
        prob_mtf_gap = np.log(np.clip(mtf_p_long, eps, 1.0)) - np.log(np.clip(mtf_p_short, eps, 1.0))
        mtf_gap = np.where(np.isfinite(mtf_gap), mtf_gap, prob_mtf_gap)
    if not np.isfinite(anchor_gap).any():
        failures.append("anchor evidence missing: no anchor_logits and no snap signal bridge probabilities")
    if "delta_logits" not in frame.columns:
        failures.append("delta_logits missing from predictions; residual gap uses final-anchor proxy")
        delta_gap = (final_log_long - final_log_short) - anchor_gap
    if not np.isfinite(mtf_gap).any():
        failures.append("MTF evidence missing: no mtf_dir_logits/mtf probs/mtf_long_minus_short")

    selected = _selected(frame, float(args.edge_threshold))
    side = _chosen_side(frame)
    failures.extend(_expected_utility_contract_failures(frame))
    focus_start = pd.Timestamp(args.focus_start)
    focus_start = focus_start.tz_localize("UTC") if focus_start.tzinfo is None else focus_start.tz_convert("UTC")
    times = pd.to_datetime(frame["time"], utc=True)
    time_min = times.min()
    time_max = times.max()
    if focus_start > time_max:
        failures.append(f"focus_start {focus_start} is after artifact time_max {time_max}")
    mae = _safe_num(frame, "mae_first_n_bps", default=np.nan)
    mfe = _safe_num(frame, "mfe_first_n_bps", default=np.nan)
    rising_support = (
        (_safe_num(frame, "y_rising_channel_support_touch", default=0.0) > 0.5)
        | (rising_support_rail_long >= 0.25)
        | (rising_support_rail_short_trap >= 0.25)
        | ((mtf_trend >= 0.0) & (support >= resistance) & (support >= 0.35) & (channel_edge >= 0.15))
    )
    high_mae_low_mfe = selected & ((mae >= 6.0) & (mfe <= 2.0))
    rising_support_short = selected & rising_support & (side == 1)
    focus_window = selected & (times >= focus_start)
    row_mask = selected | focus_window | high_mae_low_mfe | rising_support_short

    long_util = _safe_num(
        frame,
        "y_long_path_utility_bps",
        default=np.nan,
    )
    if np.isnan(long_util).all():
        long_util = _safe_num(frame, "y_direction_long_score_bps", default=np.nan)
    short_util = _safe_num(frame, "y_short_path_utility_bps", default=np.nan)
    if np.isnan(short_util).all():
        short_util = _safe_num(frame, "y_direction_short_score_bps", default=np.nan)

    row_report = pd.DataFrame(
        {
            "time": times,
            "selected": selected,
            "source_selected": _safe_num(frame, "selected", default=np.nan),
            "source_action": frame["action"].astype(str).to_numpy() if "action" in frame.columns else "",
            "source_trade_side": _safe_num(frame, "trade_side", default=np.nan),
            "source_expected_utility_side": _safe_num(frame, "expected_utility_side", default=np.nan),
            "focus_window_from_2026_07_08_18": focus_window,
            "high_mae_low_mfe_loser": high_mae_low_mfe,
            "rising_support_selected_short": rising_support_short,
            "selected_side": side,
            "selected_side_name": [_fmt_side(x) for x in side],
            "y_direction": _safe_num(frame, "y_direction", default=np.nan),
            "y_tradable": _safe_num(frame, "y_tradable", default=np.nan),
            "p_long": p_long,
            "p_short": p_short,
            "p_flat": p_flat,
            "p_trade": _safe_num(frame, "p_trade", default=np.nan),
            "p_flat_hier": _safe_num(frame, "p_flat_hier", default=np.nan),
            "p_long_given_trade": _safe_num(frame, "p_long_given_trade", default=np.nan),
            "p_short_given_trade": _safe_num(frame, "p_short_given_trade", default=np.nan),
            "edge_score": edge,
            "expected_utility_long_bps": _safe_num(frame, "expected_utility_long_bps", default=np.nan),
            "expected_utility_short_bps": _safe_num(frame, "expected_utility_short_bps", default=np.nan),
            "long_path_utility_pred_bps": _safe_num(frame, "long_path_utility_pred_bps", default=np.nan),
            "short_path_utility_pred_bps": _safe_num(frame, "short_path_utility_pred_bps", default=np.nan),
            "long_bad_path_prob": _safe_num(frame, "long_bad_path_prob", default=np.nan),
            "short_bad_path_prob": _safe_num(frame, "short_bad_path_prob", default=np.nan),
            "long_expected_mae_bps": _safe_num(frame, "long_expected_mae_bps", default=np.nan),
            "short_expected_mae_bps": _safe_num(frame, "short_expected_mae_bps", default=np.nan),
            "final_long_minus_short": final_log_long - final_log_short,
            "final_long_minus_flat": final_log_long - final_log_flat,
            "final_short_minus_flat": final_log_short - final_log_flat,
            "anchor_p_long": anchor_p_long,
            "anchor_p_short": anchor_p_short,
            "anchor_p_flat": anchor_p_flat,
            "anchor_long_minus_short": anchor_gap,
            "delta_long_minus_short": delta_gap,
            "mtf_long_minus_short": mtf_gap,
            "support_evidence": support,
            "resistance_evidence": resistance,
            "channel_edge_pressure": channel_edge,
            "rising_support_rail_long_pressure": rising_support_rail_long,
            "rising_support_rail_short_trap_pressure": rising_support_rail_short_trap,
            "falling_resistance_rail_short_pressure": falling_resistance_rail_short,
            "falling_resistance_rail_long_trap_pressure": falling_resistance_rail_long_trap,
            "trendline_rail_long_evidence": trendline_rail_long_evidence,
            "trendline_rail_short_evidence": trendline_rail_short_evidence,
            "trendline_rail_long_minus_short": trendline_rail_long_minus_short,
            "mtf_trend_evidence": mtf_trend,
            "mtf_conflict": mtf_conflict,
            "d1_slope": d1_slope,
            "mae_first_n_bps": mae,
            "mfe_first_n_bps": mfe,
            "path_quality_bps": _safe_num(frame, "path_quality_bps", default=np.nan),
            "y_long_path_utility_bps": long_util,
            "y_short_path_utility_bps": short_util,
            "y_long_bad_path": _safe_num(frame, "y_long_bad_path", default=np.nan),
            "y_short_bad_path": _safe_num(frame, "y_short_bad_path", default=np.nan),
            "y_long_expected_mae_bps": _safe_num(frame, "y_long_expected_mae_bps", default=np.nan),
            "y_short_expected_mae_bps": _safe_num(frame, "y_short_expected_mae_bps", default=np.nan),
            "y_countertrend_short_trap": _safe_num(frame, "y_countertrend_short_trap", default=np.nan),
            "y_countertrend_long_trap": _safe_num(frame, "y_countertrend_long_trap", default=np.nan),
            "y_mtf_conflict_m5_vs_higher_side": _safe_num(frame, "y_mtf_conflict_m5_vs_higher_side", default=np.nan),
        }
    )
    row_report = row_report.loc[row_mask].copy()
    if len(row_report):
        row_report["why_this_side"] = row_report.apply(_why, axis=1)
    if int(focus_window.sum()) == 0:
        failures.append("focus population empty for --focus-start window")
    if int(selected.sum()) == 0:
        failures.append("selected population empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = args.out_dir / f"SMART_DIRECTION_CONTRIBUTION_ROWS_{stamp}.csv"
    parquet_path = args.out_dir / f"SMART_DIRECTION_CONTRIBUTION_ROWS_{stamp}.parquet"
    json_path = args.out_dir / f"SMART_DIRECTION_CONTRIBUTION_AUDIT_{stamp}.json"
    md_path = args.out_dir / f"SMART_DIRECTION_CONTRIBUTION_AUDIT_{stamp}.md"
    row_report.to_csv(csv_path, index=False)
    row_report.to_parquet(parquet_path, index=False)

    report = {
        "schema_version": "smart_direction_contribution_audit_v1",
        "created_utc": _utc_now(),
        "bundle_dir": str(bundle_dir),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "dataset_dir": str(dataset_dir),
        "dataset_parquet": str(dataset_parquet),
        "predictions_parquet": str(pred_path),
        "split": args.split,
        "model_name": args.model_name,
        "edge_threshold": float(args.edge_threshold),
        "rows_joined": int(len(frame)),
        "time_min": str(time_min),
        "time_max": str(time_max),
        "selected_rows": int(selected.sum()),
        "row_report_rows": int(len(row_report)),
        "focus_counts": {
            "focus_window_from_2026_07_08_18": int(focus_window.sum()),
            "high_mae_low_mfe_loser": int(high_mae_low_mfe.sum()),
            "rising_support_selected_short": int(rising_support_short.sum()),
        },
        "row_report_csv": str(csv_path),
        "row_report_parquet": str(parquet_path),
        "component_notes": {
            "anchor_gap": "log(anchor_p_long)-log(anchor_p_short) from snap signal bridge columns when present",
            "delta_gap": "final log-prob long-short gap minus anchor long-short gap; proxy for residual/calibration push",
            "missing_live_logits": "If prediction parquet lacks explicit anchor/delta/MTF logits, this audit uses probability-derived proxy gaps.",
        },
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    (args.out_dir / "SMART_DIRECTION_CONTRIBUTION_AUDIT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (args.out_dir / "SMART_DIRECTION_CONTRIBUTION_AUDIT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(json.dumps({"decision": report["decision"], "failures": failures, "json": str(json_path), "rows": int(len(row_report))}, indent=2))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
