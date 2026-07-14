#!/usr/bin/env python3
"""Audit smart520 direction behavior in live-like conflict pockets.

This is a promotion/audit tool, not a live trading rule. It checks whether a
candidate learned pathological directional behavior in pockets like:

* intraday bullish, higher-TF bearish -> model keeps selecting SHORT
* intraday bearish, higher-TF bullish -> model keeps selecting LONG

The July 2026 XAU issue lives in the first pocket. The inverse pocket is kept
in the same report so a fix does not reintroduce the previous long-in-short
failure mode.
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
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/smart_direction_live_like_pocket_audit_v1")

SIDE_LONG = 0
SIDE_SHORT = 1
SIDE_FLAT = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_parquet(dataset_dir: Path, split: str) -> Path:
    matches = sorted(dataset_dir.glob(f"*_{split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one *_{split}.parquet in {dataset_dir}, got {matches}")
    return matches[0]


def _load_meta(bundle_dir: Path) -> dict[str, Any]:
    meta_path = bundle_dir / "bundle_metadata.json"
    if not meta_path.is_file():
        raise RuntimeError(f"missing bundle metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


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
    elif Path(bundle_raw).expanduser().resolve() != bundle_dir.resolve():
        failures.append(
            f"prediction bundle_dir mismatch: provenance={Path(bundle_raw).expanduser()} audit={bundle_dir}"
        )
    if not dataset_raw:
        failures.append(f"prediction provenance lacks dataset_dir: {latest}")
    elif Path(dataset_raw).expanduser().resolve() != dataset_dir.resolve():
        failures.append(
            f"prediction dataset_dir mismatch: provenance={Path(dataset_raw).expanduser()} audit={dataset_dir}"
        )
    return failures


def _matrix(series: pd.Series) -> np.ndarray:
    return np.asarray([np.asarray(x, dtype=np.float32) for x in series.to_numpy()], dtype=np.float32)


def _named_column(
    matrix: np.ndarray,
    names: list[str],
    name: str,
    default: float = 0.0,
    *,
    required: bool = True,
) -> np.ndarray:
    if name not in names:
        if required:
            raise RuntimeError(f"required feature {name!r} missing from bundle metadata")
        return np.full(matrix.shape[0], float(default), dtype=np.float32)
    idx = int(names.index(name))
    if idx >= matrix.shape[1]:
        raise RuntimeError(f"feature {name!r} index {idx} outside matrix width {matrix.shape[1]}")
    return matrix[:, idx].astype(np.float32)


def _first_named_column(
    matrix: np.ndarray,
    names: list[str],
    candidates: list[str],
    default: float = 0.0,
    *,
    required: bool = True,
) -> np.ndarray:
    for name in candidates:
        if name in names:
            return _named_column(matrix, names, name, default=default)
    if required:
        raise RuntimeError(f"required feature candidates missing: {candidates}")
    return np.full(matrix.shape[0], float(default), dtype=np.float32)


def _max_named_column(
    matrix: np.ndarray,
    names: list[str],
    candidates: list[str],
    default: float = 0.0,
    *,
    required: bool = True,
) -> np.ndarray:
    values = [
        _named_column(matrix, names, name, default=default)
        for name in candidates
        if name in names
    ]
    if not values:
        if required:
            raise RuntimeError(f"required feature candidates missing: {candidates}")
        return np.full(matrix.shape[0], float(default), dtype=np.float32)
    return np.maximum.reduce(values).astype(np.float32)


def _rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def _safe_mean(values: pd.Series | np.ndarray) -> float | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(arr) == 0:
        return None
    return float(np.mean(arr))


def _is_expected_utility_mode(frame: pd.DataFrame) -> bool:
    if "selection_score_mode" not in frame.columns:
        return False
    mode = frame["selection_score_mode"].astype(str).str.lower()
    return bool(mode.eq("expected_utility").any())


def _expected_utility_side_from_predictions(frame: pd.DataFrame) -> np.ndarray:
    required = ("expected_utility_long_bps", "expected_utility_short_bps")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"expected_utility predictions require side utility columns: {missing}")
    expected_long = pd.to_numeric(frame["expected_utility_long_bps"], errors="coerce").to_numpy(dtype=np.float64)
    expected_short = pd.to_numeric(frame["expected_utility_short_bps"], errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(expected_long).all() and np.isfinite(expected_short).all()):
        raise RuntimeError("expected_utility side columns contain non-finite values")
    return np.where(expected_long >= expected_short, SIDE_LONG, SIDE_SHORT).astype(np.int8)


def _side_from_predictions(frame: pd.DataFrame) -> np.ndarray:
    if _is_expected_utility_mode(frame):
        return _expected_utility_side_from_predictions(frame)
    if "trade_side" in frame.columns:
        raw = pd.to_numeric(frame["trade_side"], errors="coerce").fillna(SIDE_FLAT).to_numpy(dtype=np.int16)
        side = np.full(len(frame), SIDE_FLAT, dtype=np.int8)
        side[raw == SIDE_LONG] = SIDE_LONG
        side[raw == SIDE_SHORT] = SIDE_SHORT
        return side
    if "action" in frame.columns:
        action = frame["action"].astype(str).str.upper()
        side = np.full(len(frame), SIDE_FLAT, dtype=np.int8)
        side[action.isin(["TAKE_LONG_NOW", "LONG"])] = SIDE_LONG
        side[action.isin(["TAKE_SHORT_NOW", "SHORT"])] = SIDE_SHORT
        return side
    p_long = pd.to_numeric(frame["p_long"], errors="coerce").to_numpy(dtype=np.float64)
    p_short = pd.to_numeric(frame["p_short"], errors="coerce").to_numpy(dtype=np.float64)
    side = np.full(len(frame), SIDE_FLAT, dtype=np.int8)
    side[p_long >= p_short] = SIDE_LONG
    side[p_short > p_long] = SIDE_SHORT
    return side


def _selected_from_predictions(
    frame: pd.DataFrame,
    edge_threshold: float,
    selection_score_threshold_override: float | None = None,
) -> np.ndarray:
    expected_utility_mode = _is_expected_utility_mode(frame)
    if expected_utility_mode and "selection_score" not in frame.columns:
        raise RuntimeError("expected_utility predictions require selection_score for pocket audit")
    if "selection_score" in frame.columns and ("selection_score_threshold" in frame.columns or expected_utility_mode):
        score = pd.to_numeric(frame["selection_score"], errors="coerce").to_numpy(dtype=np.float64)
        if selection_score_threshold_override is not None:
            threshold = np.full(len(frame), float(selection_score_threshold_override), dtype=np.float64)
        elif "selection_score_threshold" in frame.columns:
            threshold = pd.to_numeric(frame["selection_score_threshold"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            threshold = np.zeros(len(frame), dtype=np.float64)
        return score >= threshold
    if selection_score_threshold_override is None and "selected" in frame.columns:
        return pd.Series(frame["selected"]).fillna(False).astype(bool).to_numpy()
    if selection_score_threshold_override is None and "action" in frame.columns:
        action = frame["action"].astype(str).str.upper()
        return action.isin(["TAKE_LONG_NOW", "TAKE_SHORT_NOW", "LONG", "SHORT"]).to_numpy()
    edge = pd.to_numeric(frame["edge_score"], errors="coerce").to_numpy(dtype=np.float64)
    return edge >= float(edge_threshold)


def _selection_score_mode_values(frame: pd.DataFrame) -> list[str]:
    if "selection_score_mode" not in frame.columns:
        return []
    return sorted(
        {
            str(value).strip().lower()
            for value in frame["selection_score_mode"].dropna().astype(str).to_numpy()
            if str(value).strip()
        }
    )


def _assert_selection_score_mode(frame: pd.DataFrame, required_mode: str) -> list[str]:
    required = str(required_mode or "").strip().lower()
    if not required:
        return []
    observed = _selection_score_mode_values(frame)
    failures: list[str] = []
    if observed != [required]:
        failures.append(f"selection_score_mode mismatch: required={required} observed={observed or ['<missing>']}")
    if required == "expected_utility":
        required_columns = (
            "selection_score",
            "selection_score_threshold",
            "trade_side",
            "expected_utility_long_bps",
            "expected_utility_short_bps",
            "expected_utility_side",
        )
        missing = [name for name in required_columns if name not in frame.columns]
        if missing:
            failures.append(f"expected_utility pocket audit missing prediction columns: {missing}")
        elif len(frame):
            score = pd.to_numeric(frame["selection_score"], errors="coerce").to_numpy(dtype=np.float64)
            threshold = pd.to_numeric(frame["selection_score_threshold"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            score_selected = score >= threshold
            utility_side = _expected_utility_side_from_predictions(frame)
            expected_side = pd.to_numeric(frame["expected_utility_side"], errors="coerce").fillna(SIDE_FLAT).to_numpy(dtype=np.int16)
            side_mismatches = int(np.sum(expected_side != utility_side))
            if side_mismatches:
                failures.append(
                    "expected_utility pocket audit expected_utility_side mismatches long/short utility: "
                    f"mismatches={side_mismatches}"
                )
            trade_side = pd.to_numeric(frame["trade_side"], errors="coerce").fillna(SIDE_FLAT).to_numpy(dtype=np.int16)
            trade_side_mismatches = int(np.sum(trade_side != utility_side))
            if trade_side_mismatches:
                failures.append(
                    "expected_utility pocket audit trade_side mismatches long/short utility side: "
                    f"mismatches={trade_side_mismatches}"
                )
            if "selected" in frame.columns:
                legacy_selected = pd.Series(frame["selected"]).fillna(False).astype(bool).to_numpy()
                mismatches = int(np.sum(legacy_selected != score_selected))
                if mismatches:
                    failures.append(
                        "expected_utility pocket audit selected column mismatches selection_score threshold: "
                        f"mismatches={mismatches}"
                    )
            if "action" in frame.columns:
                action = frame["action"].astype(str).str.upper()
                action_selected = action.isin(["TAKE_LONG_NOW", "TAKE_SHORT_NOW", "LONG", "SHORT"]).to_numpy()
                mismatches = int(np.sum(action_selected != score_selected))
                if mismatches:
                    failures.append(
                        "expected_utility pocket audit action column mismatches selection_score threshold: "
                        f"mismatches={mismatches}"
                    )
    return failures


def _pnl_proxy_for_side(frame: pd.DataFrame, side: np.ndarray) -> np.ndarray:
    if {"y_long_path_utility_bps", "y_short_path_utility_bps"}.issubset(frame.columns):
        long_score = pd.to_numeric(frame["y_long_path_utility_bps"], errors="coerce").to_numpy(dtype=np.float64)
        short_score = pd.to_numeric(frame["y_short_path_utility_bps"], errors="coerce").to_numpy(dtype=np.float64)
        return np.where(side == SIDE_LONG, long_score, short_score)
    if "y_forecast_ret_K24" in frame.columns:
        ret = pd.to_numeric(frame["y_forecast_ret_K24"], errors="coerce").to_numpy(dtype=np.float64)
        return np.where(side == SIDE_LONG, ret, -ret)
    if {"y_direction_long_score_bps", "y_direction_short_score_bps"}.issubset(frame.columns):
        long_score = pd.to_numeric(frame["y_direction_long_score_bps"], errors="coerce").to_numpy(dtype=np.float64)
        short_score = pd.to_numeric(frame["y_direction_short_score_bps"], errors="coerce").to_numpy(dtype=np.float64)
        return np.where(side == SIDE_LONG, long_score, short_score)
    return np.full(len(frame), np.nan, dtype=np.float64)


def _summarize(frame: pd.DataFrame, mask: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    sub = frame.loc[mask].copy()
    sel = mask & selected
    sub_sel = frame.loc[sel].copy()
    side = _side_from_predictions(sub_sel) if len(sub_sel) else np.asarray([], dtype=np.int8)
    label = pd.to_numeric(sub_sel.get("y_direction", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=np.float64)
    pnl = _pnl_proxy_for_side(sub_sel, side) if len(sub_sel) else np.asarray([], dtype=np.float64)
    mae = pd.to_numeric(sub_sel.get("mae_first_n_bps", pd.Series(dtype=float)), errors="coerce")
    mfe = pd.to_numeric(sub_sel.get("mfe_first_n_bps", pd.Series(dtype=float)), errors="coerce")
    path = pd.to_numeric(sub_sel.get("path_quality_bps", pd.Series(dtype=float)), errors="coerce")
    return {
        "rows": int(len(sub)),
        "selected_rows": int(len(sub_sel)),
        "selected_rate": (int(len(sub_sel)) / int(len(sub)) if len(sub) else 0.0),
        "selected_side_long_count": int(np.sum(side == SIDE_LONG)),
        "selected_side_short_count": int(np.sum(side == SIDE_SHORT)),
        "selected_side_long_rate": _rate(side == SIDE_LONG),
        "selected_side_short_rate": _rate(side == SIDE_SHORT),
        "selected_label_long_rate": _rate(label == SIDE_LONG),
        "selected_label_short_rate": _rate(label == SIDE_SHORT),
        "selected_label_flat_rate": _rate(label == SIDE_FLAT),
        "selected_mean_edge_score": _safe_mean(sub_sel.get("edge_score", pd.Series(dtype=float))),
        "selected_mean_p_long": _safe_mean(sub_sel.get("p_long", pd.Series(dtype=float))),
        "selected_mean_p_short": _safe_mean(sub_sel.get("p_short", pd.Series(dtype=float))),
        "selected_mean_p_flat": _safe_mean(sub_sel.get("p_flat", pd.Series(dtype=float))),
        "selected_mean_proxy_pnl_bps": _safe_mean(pnl),
        "selected_mean_mfe_first_n_bps": _safe_mean(mfe),
        "selected_mean_mae_first_n_bps": _safe_mean(mae),
        "selected_mean_path_quality_bps": _safe_mean(path),
    }


def _decision(max_bad_rate: float, min_rows: int, summaries: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    failures: list[str] = []
    short_bad_pockets = [
        "intraday_bull",
        "intraday_bull__htf_bull",
        "intraday_bull__htf_bear",
        "rising_channel_support_touch",
        "support_retest_continuation",
        "rising_channel_support_continuation",
        "countertrend_short_trap",
        "short_high_mae_low_mfe_early_failure",
    ]
    long_bad_pockets = [
        "intraday_bear",
        "intraday_bear__htf_bear",
        "intraday_bear__htf_bull",
        "falling_channel_resistance_touch",
        "resistance_retest_continuation",
        "falling_channel_resistance_continuation",
        "countertrend_long_trap",
        "long_high_mae_low_mfe_early_failure",
    ]
    for name in short_bad_pockets:
        row = summaries[name]
        if int(row["rows"]) >= min_rows and int(row["selected_rows"]) < min_rows:
            failures.append(
                f"{name} selected coverage {int(row['selected_rows'])} < {min_rows} "
                f"with pocket rows={int(row['rows'])}"
            )
        selected_rows = int(row["selected_rows"])
        short_count = int(row.get("selected_side_short_count", round(float(row["selected_side_short_rate"]) * selected_rows)))
        if selected_rows < min_rows and short_count > 0:
            failures.append(f"{name} selected SHORT count {short_count} with low support n={selected_rows} < {min_rows}")
        if selected_rows >= min_rows and float(row["selected_side_short_rate"]) > max_bad_rate:
            failures.append(
                f"{name} selected SHORT rate {row['selected_side_short_rate']:.3f} > {max_bad_rate:.3f}"
            )
    for name in long_bad_pockets:
        row = summaries[name]
        if int(row["rows"]) >= min_rows and int(row["selected_rows"]) < min_rows:
            failures.append(
                f"{name} selected coverage {int(row['selected_rows'])} < {min_rows} "
                f"with pocket rows={int(row['rows'])}"
            )
        selected_rows = int(row["selected_rows"])
        long_count = int(row.get("selected_side_long_count", round(float(row["selected_side_long_rate"]) * selected_rows)))
        if selected_rows < min_rows and long_count > 0:
            failures.append(f"{name} selected LONG count {long_count} with low support n={selected_rows} < {min_rows}")
        if selected_rows >= min_rows and float(row["selected_side_long_rate"]) > max_bad_rate:
            failures.append(
                f"{name} selected LONG rate {row['selected_side_long_rate']:.3f} > {max_bad_rate:.3f}"
            )
    utility_pockets = (
        "rising_channel_support_touch",
        "support_retest_continuation",
        "rising_channel_support_continuation",
        "countertrend_short_trap",
        "falling_channel_resistance_touch",
        "resistance_retest_continuation",
        "falling_channel_resistance_continuation",
        "countertrend_long_trap",
    )
    for name in utility_pockets:
        row = summaries[name]
        mean_pnl = row.get("selected_mean_proxy_pnl_bps")
        if int(row["selected_rows"]) >= min_rows and mean_pnl is not None and float(mean_pnl) <= 0.0:
            failures.append(f"{name} selected mean proxy pnl {float(mean_pnl):.2f} <= 0")
    return ("PASS" if not failures else "FAIL", failures)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SMART520 Direction Live-Like Pocket Audit",
        "",
        f"decision: {report['decision']}",
        f"created_utc: {report['created_utc']}",
        f"bundle_dir: {report['bundle_dir']}",
        f"predictions: {report['predictions_parquet']}",
        f"dataset: {report['dataset_parquet']}",
        "",
        "| pocket | rows | selected | side_long | side_short | label_long | label_short | mean_pnl | mean_mfe | mean_mae |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["pockets"].items():
        lines.append(
            "| {name} | {rows} | {sel} | {sl:.3f} | {ss:.3f} | {ll:.3f} | {ls:.3f} | {pnl} | {mfe} | {mae} |".format(
                name=name,
                rows=row["rows"],
                sel=row["selected_rows"],
                sl=row["selected_side_long_rate"],
                ss=row["selected_side_short_rate"],
                ll=row["selected_label_long_rate"],
                ls=row["selected_label_short_rate"],
                pnl=("NA" if row["selected_mean_proxy_pnl_bps"] is None else f"{row['selected_mean_proxy_pnl_bps']:.2f}"),
                mfe=("NA" if row["selected_mean_mfe_first_n_bps"] is None else f"{row['selected_mean_mfe_first_n_bps']:.2f}"),
                mae=("NA" if row["selected_mean_mae_first_n_bps"] is None else f"{row['selected_mean_mae_first_n_bps']:.2f}"),
            )
        )
    if report["failures"]:
        lines.extend(["", "## Failures", *[f"- {x}" for x in report["failures"]]])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--predictions-parquet", type=Path, default=DEFAULT_PREDICTIONS)
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--edge-threshold", type=float, default=0.145)
    ap.add_argument(
        "--selection-score-threshold",
        type=float,
        default=None,
        help="Optional audit-only override for prediction files with selection_score/expected_utility outputs.",
    )
    ap.add_argument(
        "--require-selection-score-mode",
        choices=("expected_utility", "edge_score", "any"),
        default="expected_utility",
        help="Fail unless predictions were materialized on the required selection surface. Use any for research-only legacy audits.",
    )
    ap.add_argument("--max-bad-side-rate", type=float, default=0.35)
    ap.add_argument("--min-selected-rows", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    pred_path = args.predictions_parquet.expanduser().resolve()
    failures: list[str] = []
    failures.extend(_assert_prediction_provenance(pred_path, bundle_dir, dataset_dir))

    meta = _load_meta(bundle_dir)
    signal_names = list(meta.get("ordered_signal_names") or [])
    ctx_cont_names = list(meta.get("ordered_ctx_cont_names") or [])
    ctx_cat_names = list(meta.get("ordered_ctx_cat_names") or [])
    if not signal_names or not ctx_cont_names or not ctx_cat_names:
        raise RuntimeError("bundle metadata lacks ordered feature names needed for contract-driven audit")

    dataset_parquet = _split_parquet(dataset_dir, args.split)
    ds_cols = [
        "time",
        "snap",
        "ctx_cont",
        "ctx_cat",
        "y_direction",
        "y_tradable",
        "mae_first_n_bps",
        "mfe_first_n_bps",
        "path_quality_bps",
    ]
    optional_ds_cols = [
        "y_forecast_ret_K24",
        "y_direction_long_score_bps",
        "y_direction_short_score_bps",
        "y_long_path_utility_bps",
        "y_short_path_utility_bps",
        "y_rising_channel_support_touch",
        "y_falling_channel_resistance_touch",
        "y_support_retest_continuation",
        "y_resistance_retest_continuation",
        "y_countertrend_short_trap",
        "y_countertrend_long_trap",
        "y_mtf_conflict_m5_vs_higher_side",
        "y_long_high_mae_low_mfe_early_failure",
        "y_short_high_mae_low_mfe_early_failure",
    ]
    # pandas does not expose all parquet columns cheaply here; let read_parquet fail
    # only for required columns and add optional columns if present through metadata.
    import pyarrow.parquet as pq

    parquet_cols = set(pq.ParquetFile(dataset_parquet).schema_arrow.names)
    required_pocket_label_cols = [
        "y_rising_channel_support_touch",
        "y_falling_channel_resistance_touch",
        "y_support_retest_continuation",
        "y_resistance_retest_continuation",
        "y_countertrend_short_trap",
        "y_countertrend_long_trap",
        "y_long_high_mae_low_mfe_early_failure",
        "y_short_high_mae_low_mfe_early_failure",
    ]
    missing_pocket_label_cols = [c for c in required_pocket_label_cols if c not in parquet_cols]
    if missing_pocket_label_cols:
        failures.append(
            "dataset lacks direction-repair pocket labels required for launch-blocking audit: "
            + ",".join(missing_pocket_label_cols)
        )
    read_cols = ds_cols + [c for c in optional_ds_cols if c in parquet_cols]
    ds = pd.read_parquet(dataset_parquet, columns=read_cols)
    ds["time"] = pd.to_datetime(ds["time"], utc=True)

    pred_cols = ["time", "split", "model", "p_long", "p_short", "p_flat", "edge_score"]
    pred_schema_cols = set(pq.ParquetFile(pred_path).schema_arrow.names)
    pred_optional = [
        "y_direction",
        "path_quality_bps",
        "mae_first_n_bps",
        "mfe_first_n_bps",
        "selected",
        "trade_side",
        "action",
        "selection_score_mode",
        "selection_score",
        "selection_score_threshold",
        "expected_utility_long_bps",
        "expected_utility_short_bps",
        "expected_utility_side",
    ]
    pred = pd.read_parquet(
        pred_path,
        columns=pred_cols + [c for c in pred_optional if c in pred_schema_cols],
    )
    pred["time"] = pd.to_datetime(pred["time"], utc=True)
    pred = pred[(pred["split"] == args.split) & (pred["model"] == args.model_name)].copy()

    for name, data in (("predictions", pred), ("dataset", ds)):
        dup = data["time"].duplicated(keep=False)
        if bool(dup.any()):
            raise RuntimeError(f"{name} has duplicate time rows, first={data.loc[dup, 'time'].head(5).tolist()}")
    frame = pred.merge(ds, on="time", how="outer", suffixes=("_pred", ""), indicator=True, validate="one_to_one")
    unmatched = frame["_merge"].value_counts().to_dict()
    if unmatched.get("left_only", 0) or unmatched.get("right_only", 0):
        raise RuntimeError(f"prediction/dataset time coverage mismatch: {unmatched}")
    frame = frame.drop(columns=["_merge"])
    if frame.empty:
        raise RuntimeError("empty prediction/dataset join")
    required_selection_mode = "" if args.require_selection_score_mode == "any" else str(args.require_selection_score_mode)
    failures.extend(_assert_selection_score_mode(frame, required_selection_mode))

    snap = _matrix(frame["snap"])
    ctx_cont = _matrix(frame["ctx_cont"])
    ctx_cat = _matrix(frame["ctx_cat"])

    ema_stack = _named_column(snap, signal_names, "trend.ema_stack_alignment_score")
    long_bias = _named_column(snap, signal_names, "trend.mtf_confluence_long_trend_bias")
    short_bias = _named_column(snap, signal_names, "trend.mtf_confluence_short_trend_bias")
    h4d1_bull = _named_column(snap, signal_names, "session_regime.h4_d1_stack_bull_pressure")
    h4d1_bear = _named_column(snap, signal_names, "session_regime.h4_d1_stack_bear_pressure")
    trend_direction = _named_column(snap, signal_names, "trend.mtf_confluence_trend_direction_score")
    support_prox = _max_named_column(
        snap,
        signal_names,
        [
            "chart.geometry_support_line_proximity_stack",
            "chart.sr_memory_support_level_proximity_stack",
            "chart.sr_memory_support_respect_pressure_long",
            "chart.sr_memory_support_reclaim_pressure_long",
            "chart.sr_memory_liquidity_low_level_rejection_long",
            "chart.geometry_fib_support_confluence_long_pressure",
            "chart.geometry_rising_support_rail_long_pressure",
        ],
    )
    resistance_prox = _max_named_column(
        snap,
        signal_names,
        [
            "chart.geometry_resistance_line_proximity_stack",
            "chart.sr_memory_resistance_level_proximity_stack",
            "chart.sr_memory_resistance_respect_pressure_short",
            "chart.sr_memory_resistance_reclaim_pressure_short",
            "chart.sr_memory_liquidity_high_level_rejection_short",
            "chart.geometry_fib_resistance_confluence_short_pressure",
            "chart.geometry_falling_resistance_rail_short_pressure",
        ],
    )
    channel_edge = _first_named_column(snap, signal_names, ["chart.geometry_channel_edge_pressure"])
    channel_pos = _first_named_column(
        snap,
        signal_names,
        ["chart.geometry_channel_position_low_to_high"],
        default=0.5,
        required=False,
    )
    support_respect = _first_named_column(
        snap,
        signal_names,
        [
            "chart.sr_memory_support_respect_pressure_long",
            "chart.sr_memory_liquidity_low_level_rejection_long",
        ],
    )
    resistance_respect = _first_named_column(
        snap,
        signal_names,
        [
            "chart.sr_memory_resistance_respect_pressure_short",
            "chart.sr_memory_liquidity_high_level_rejection_short",
        ],
    )
    d1_slope = _named_column(ctx_cont, ctx_cont_names, "d1_ema_slope_20_canon_v2")
    h4_ema = _named_column(ctx_cont, ctx_cont_names, "_v1h4_ema_diff")
    h4_trend_cat = _named_column(ctx_cat, ctx_cat_names, "H4_trend_sign_cat", default=1.0)

    selected = _selected_from_predictions(
        frame,
        float(args.edge_threshold),
        selection_score_threshold_override=args.selection_score_threshold,
    )
    intraday_bull = (ema_stack > 0.0) & (long_bias > short_bias) & (trend_direction >= 0.0)
    intraday_bear = (ema_stack < 0.0) & (short_bias > long_bias) & (trend_direction <= 0.0)
    h4_trend_cat_sign = np.where(h4_trend_cat == 2, 1.0, np.where(h4_trend_cat == 0, -1.0, 0.0))
    htf_score = (
        (h4d1_bull - h4d1_bear)
        + (0.25 * np.sign(d1_slope))
        + (0.25 * np.sign(h4_ema))
        + (0.25 * h4_trend_cat_sign)
    )
    htf_bull = htf_score >= 0.25
    htf_bear = htf_score <= -0.25
    if "y_rising_channel_support_touch" in frame.columns:
        rising_channel_support = (
            pd.to_numeric(frame["y_rising_channel_support_touch"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            > 0.5
        )
    else:
        rising_channel_support = (
            intraday_bull
            & (support_prox >= 0.35)
            & (support_prox >= resistance_prox)
            & ((channel_edge >= 0.15) | (channel_pos <= 0.42) | (support_respect >= 0.35))
        )
    if "y_falling_channel_resistance_touch" in frame.columns:
        falling_channel_resistance = (
            pd.to_numeric(frame["y_falling_channel_resistance_touch"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            > 0.5
        )
    else:
        falling_channel_resistance = (
            intraday_bear
            & (resistance_prox >= 0.35)
            & (resistance_prox >= support_prox)
            & ((channel_edge >= 0.15) | (channel_pos >= 0.58) | (resistance_respect >= 0.35))
        )

    def label_mask(name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        return (
            pd.to_numeric(frame[name], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
            > 0.5
        )

    support_retest_continuation = label_mask("y_support_retest_continuation")
    resistance_retest_continuation = label_mask("y_resistance_retest_continuation")
    countertrend_short_trap = label_mask("y_countertrend_short_trap")
    countertrend_long_trap = label_mask("y_countertrend_long_trap")
    long_high_mae_low_mfe_early_failure = label_mask("y_long_high_mae_low_mfe_early_failure")
    short_high_mae_low_mfe_early_failure = label_mask("y_short_high_mae_low_mfe_early_failure")
    rising_channel_support_continuation = rising_channel_support & support_retest_continuation
    falling_channel_resistance_continuation = falling_channel_resistance & resistance_retest_continuation

    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "selected_all": selected,
        "intraday_bull__htf_bull": intraday_bull & htf_bull,
        "intraday_bull__htf_bear": intraday_bull & htf_bear,
        "intraday_bear__htf_bear": intraday_bear & htf_bear,
        "intraday_bear__htf_bull": intraday_bear & htf_bull,
        "intraday_bull": intraday_bull,
        "intraday_bear": intraday_bear,
        "rising_channel_support_touch": rising_channel_support,
        "falling_channel_resistance_touch": falling_channel_resistance,
        "support_retest_continuation": support_retest_continuation,
        "resistance_retest_continuation": resistance_retest_continuation,
        "rising_channel_support_continuation": rising_channel_support_continuation,
        "falling_channel_resistance_continuation": falling_channel_resistance_continuation,
        "countertrend_short_trap": countertrend_short_trap,
        "countertrend_long_trap": countertrend_long_trap,
        "short_high_mae_low_mfe_early_failure": short_high_mae_low_mfe_early_failure,
        "long_high_mae_low_mfe_early_failure": long_high_mae_low_mfe_early_failure,
    }
    summaries = {name: _summarize(frame, mask, selected) for name, mask in masks.items()}
    decision, gate_failures = _decision(float(args.max_bad_side_rate), int(args.min_selected_rows), summaries)
    failures.extend(gate_failures)
    decision = "PASS" if not failures else "FAIL"

    report = {
        "schema_version": "smart_direction_live_like_pocket_audit_v1",
        "created_utc": _utc_now(),
        "decision": decision,
        "failures": failures,
        "bundle_dir": str(bundle_dir),
        "predictions_parquet": str(pred_path),
        "dataset_dir": str(dataset_dir),
        "dataset_parquet": str(dataset_parquet),
        "split": args.split,
        "model_name": args.model_name,
        "edge_threshold": float(args.edge_threshold),
        "required_selection_score_mode": args.require_selection_score_mode,
        "observed_selection_score_modes": _selection_score_mode_values(frame),
        "selection_score_threshold_source": (
            "override"
            if args.selection_score_threshold is not None
            else "prediction_column"
            if "selection_score_threshold" in frame.columns
            else "edge_threshold"
        ),
        "selection_score_threshold_override": (
            None if args.selection_score_threshold is None else float(args.selection_score_threshold)
        ),
        "max_bad_side_rate": float(args.max_bad_side_rate),
        "min_selected_rows": int(args.min_selected_rows),
        "features": {
            "intraday": [
                "trend.ema_stack_alignment_score",
                "trend.mtf_confluence_long_trend_bias",
                "trend.mtf_confluence_short_trend_bias",
                "trend.mtf_confluence_trend_direction_score",
            ],
            "higher_tf": [
                "session_regime.h4_d1_stack_bull_pressure",
                "session_regime.h4_d1_stack_bear_pressure",
                "d1_ema_slope_20_canon_v2",
                "_v1h4_ema_diff",
                "H4_trend_sign_cat",
            ],
            "geometry_support_resistance": [
                "chart.geometry_support_line_proximity_stack",
                "chart.geometry_resistance_line_proximity_stack",
                "chart.geometry_channel_edge_pressure",
                "chart.geometry_channel_position_low_to_high",
                "chart.sr_memory_support_respect_pressure_long",
                "chart.sr_memory_resistance_respect_pressure_short",
                "y_rising_channel_support_touch",
                "y_falling_channel_resistance_touch",
                "y_support_retest_continuation",
                "y_resistance_retest_continuation",
                "y_countertrend_short_trap",
                "y_countertrend_long_trap",
                "y_long_high_mae_low_mfe_early_failure",
                "y_short_high_mae_low_mfe_early_failure",
            ],
        },
        "pockets": summaries,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out_dir / f"SMART_DIRECTION_LIVE_LIKE_POCKET_AUDIT_{stamp}.json"
    md_path = args.out_dir / f"SMART_DIRECTION_LIVE_LIKE_POCKET_AUDIT_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    (args.out_dir / "SMART_DIRECTION_LIVE_LIKE_POCKET_AUDIT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (args.out_dir / "SMART_DIRECTION_LIVE_LIKE_POCKET_AUDIT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "failures": failures, "json": str(json_path)}, indent=2))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
