#!/usr/bin/env python3
"""Audit model-native seq513 XAU direction behavior in evidence pockets.

This is a promotion/audit tool, not a live trading rule. It checks whether a
candidate learned pathological directional behavior in pockets like:

* intraday bullish, higher-TF bearish -> model keeps selecting SHORT
* intraday bearish, higher-TF bullish -> model keeps selecting LONG

The July 2026 XAU issue lives in the first pocket. The inverse pocket is kept
in the same immutable event so a fix does not reintroduce the previous
long-in-short failure mode.

Direction/action is the persisted raw ``entry_action_q_bps`` unique argmax.
The pocket thresholds are offline launch-evidence gates; they are not
live direction thresholds.  Utility, trend, session, path, structure, and rail
evidence may define audited slices but never select, suppress, or replace
LONG/SHORT/FLAT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    require_model_direction_decision_contract,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE,
    DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95,
    DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE,
    DIRECTION_POCKET_MIN_SELECTED_ROWS,
    DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS,
    DIRECTION_POCKET_SPREAD_AWARE_PROXY_CONTRACT,
    DIRECTION_POCKET_WILSON_CONFIDENCE_LEVEL,
    MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION,
    MODEL_NATIVE_REQUIRED_MODEL_NAME,
    MODEL_NATIVE_REQUIRED_TEST_SPLIT,
    MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION,
    UTC_TIME_COVERAGE_SCHEMA_VERSION,
    direction_pocket_wilson_upper_95,
    serve_gate_event_contract_failures,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
    resolve_and_validate_prediction_evidence,
    sha256_file,
)

SCHEMA_VERSION = MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION
EVENT_PREFIX = "MODEL_NATIVE_DIRECTION_POCKET_AUDIT"
EXPECTED_CTX_CONT_DIM = MODEL_NATIVE_CTX_CONT_DIM
EXPECTED_CTX_CAT_DIM = MODEL_NATIVE_CTX_CAT_DIM

SIDE_LONG = 0
SIDE_SHORT = 1
SIDE_FLAT = 2
MODEL_DIRECTION_REQUIRED_COLUMNS = (
    "pred_direction",
    "selection_score_mode",
    "entry_action_q_bps",
    "entry_action_q_margin_bps",
)


def _time_coverage_contract(values: object, *, label: str) -> dict[str, object]:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    except Exception as exc:
        raise RuntimeError(f"{label} contains invalid UTC times") from exc
    if index.empty:
        raise RuntimeError(f"{label} time coverage is empty")
    index = index.sort_values()
    if index.has_duplicates:
        raise RuntimeError(f"{label} time coverage contains duplicates")
    utc_ns = np.asarray(index.asi8, dtype="<i8")
    return {
        "schema_version": UTC_TIME_COVERAGE_SCHEMA_VERSION,
        "rows": int(len(index)),
        "first_utc": index[0].isoformat(),
        "last_utc": index[-1].isoformat(),
        "utc_ns_sha256": hashlib.sha256(utc_ns.tobytes()).hexdigest(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_explicit_dataset_parquet(
    dataset_dir: Path,
    requested_path: Path,
    split: str,
) -> Path:
    raw_path = requested_path.expanduser()
    if raw_path.is_symlink():
        raise RuntimeError(f"dataset parquet cannot be a symlink: {raw_path}")
    path = raw_path.resolve()
    if not path.is_file():
        raise RuntimeError(f"explicit dataset parquet is missing: {path}")
    if path.parent != dataset_dir:
        raise RuntimeError(
            "explicit dataset parquet must be a direct child of dataset_dir: "
            f"dataset={path} dataset_dir={dataset_dir}"
        )
    if not path.name.endswith(f"_{split}.parquet"):
        raise RuntimeError(
            f"dataset parquet name does not bind split={split!r}: {path.name}"
        )
    return path


def _load_meta(bundle_dir: Path, requested_path: Path) -> tuple[Path, dict[str, Any]]:
    raw_path = requested_path.expanduser()
    if raw_path.is_symlink():
        raise RuntimeError(f"bundle metadata cannot be a symlink: {raw_path}")
    meta_path = raw_path.resolve()
    expected = (bundle_dir / "bundle_metadata.json").resolve()
    if meta_path != expected:
        raise RuntimeError(
            "explicit bundle metadata path mismatch: "
            f"requested={meta_path} expected={expected}"
        )
    if not meta_path.is_file():
        raise RuntimeError(f"missing bundle metadata: {meta_path}")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid bundle metadata JSON {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"bundle metadata must be a JSON object: {meta_path}")
    return meta_path, payload


def _matrix(series: pd.Series) -> np.ndarray:
    try:
        matrix = np.stack(
            [np.asarray(value, dtype=np.float32) for value in series.to_numpy()]
        )
    except Exception as exc:
        raise RuntimeError(f"{series.name!r} is not a dense numeric matrix") from exc
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError(
            f"{series.name!r} must be a finite rank-2 matrix; got shape={matrix.shape}"
        )
    return matrix


def _named_column(
    matrix: np.ndarray,
    names: list[str],
    name: str,
) -> np.ndarray:
    if name not in names:
        raise RuntimeError(f"required feature {name!r} missing from bundle metadata")
    idx = int(names.index(name))
    if idx >= matrix.shape[1]:
        raise RuntimeError(f"feature {name!r} index {idx} outside matrix width {matrix.shape[1]}")
    values = matrix[:, idx].astype(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(f"required feature {name!r} contains non-finite values")
    return values


def _rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def _safe_mean(values: pd.Series | np.ndarray) -> float | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    if len(arr) == 0:
        return None
    if not np.isfinite(arr).all():
        raise RuntimeError("pocket evidence contains non-finite numeric values")
    return float(np.mean(arr))


def _finite_distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise RuntimeError(
            f"model input evidence must be a non-empty finite vector; shape={array.shape}"
        )
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _strict_integer_column(frame: pd.DataFrame, name: str, allowed: set[int]) -> np.ndarray:
    if name not in frame.columns:
        raise RuntimeError(f"model direction prediction contract missing column: {name}")
    raw = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise RuntimeError(f"model direction prediction column {name} contains non-finite values")
    rounded = np.rint(raw)
    if not np.array_equal(raw, rounded):
        raise RuntimeError(f"model direction prediction column {name} contains non-integer values")
    values = rounded.astype(np.int8)
    invalid = sorted(set(int(value) for value in values) - allowed)
    if invalid:
        raise RuntimeError(
            f"model direction prediction column {name} contains invalid values: {invalid}"
        )
    return values


def _strict_vector_column(frame: pd.DataFrame, name: str, width: int) -> np.ndarray:
    if name not in frame.columns:
        raise RuntimeError(f"model direction prediction contract missing column: {name}")
    try:
        values = np.stack(
            [np.asarray(value, dtype=np.float64) for value in frame[name].to_numpy()]
        )
    except Exception as exc:
        raise RuntimeError(
            f"model direction prediction column {name} is not a dense vector"
        ) from exc
    if values.shape != (len(frame), width):
        raise RuntimeError(
            f"model direction prediction column {name} must have shape "
            f"({len(frame)},{width}); got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"model direction prediction column {name} contains non-finite values"
        )
    return values


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _side_from_predictions(frame: pd.DataFrame) -> np.ndarray:
    """Return the model's final LONG/SHORT/FLAT argmax without substitutes."""

    return _strict_integer_column(
        frame,
        "pred_direction",
        {SIDE_LONG, SIDE_SHORT, SIDE_FLAT},
    )


def _selected_from_predictions(frame: pd.DataFrame) -> np.ndarray:
    """A trade action is exactly a non-FLAT model direction; no threshold exists."""

    return _side_from_predictions(frame) != SIDE_FLAT


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


def _assert_selection_score_mode(frame: pd.DataFrame) -> list[str]:
    observed = _selection_score_mode_values(frame)
    failures: list[str] = []
    if observed != [MODEL_DIRECTION_SELECTION_MODE]:
        failures.append(
            "selection_score_mode mismatch: "
            f"required={MODEL_DIRECTION_SELECTION_MODE} observed={observed or ['<missing>']}"
        )
    elif not bool(
        frame["selection_score_mode"].map(
            lambda value: isinstance(value, str)
            and value == MODEL_DIRECTION_SELECTION_MODE
        ).all()
    ):
        failures.append(
            "selection_score_mode must equal model_direction_argmax exactly on every row"
        )
    return failures


def _model_direction_contract_failures(frame: pd.DataFrame) -> list[str]:
    """Validate unique raw-bps Q argmax as the sole Entry authority."""

    missing = [
        name for name in MODEL_DIRECTION_REQUIRED_COLUMNS if name not in frame.columns
    ]
    if missing:
        return [f"fitted-Q audit missing prediction columns: {missing}"]

    failures = _assert_selection_score_mode(frame)
    try:
        pred_direction = _side_from_predictions(frame)
        entry_q = _strict_vector_column(frame, "entry_action_q_bps", 3)
    except RuntimeError as exc:
        failures.append(str(exc))
        return failures

    winner_counts = np.count_nonzero(
        entry_q == np.max(entry_q, axis=1, keepdims=True),
        axis=1,
    )
    tied_rows = int(np.count_nonzero(winner_counts != 1))
    if tied_rows:
        failures.append(
            f"entry_action_q_bps has no unique top action: rows={tied_rows}"
        )
    q_argmax = np.argmax(entry_q, axis=1).astype(np.int8)
    mismatches = int(np.sum(q_argmax != pred_direction))
    if mismatches:
        failures.append(
            "pred_direction mismatches raw entry_action_q_bps argmax: "
            f"mismatches={mismatches}"
        )
    persisted_margin = pd.to_numeric(
        frame["entry_action_q_margin_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    ordered = np.sort(entry_q, axis=1)
    expected_margin = ordered[:, -1] - ordered[:, -2]
    if (
        not np.isfinite(persisted_margin).all()
        or not np.allclose(
            persisted_margin, expected_margin, rtol=0.0, atol=2e-6
        )
    ):
        failures.append("entry_action_q_margin_bps mismatches raw Q")
    return failures
def _pnl_proxy_for_side(frame: pd.DataFrame, side: np.ndarray) -> np.ndarray:
    """Return diagnostic outcome evidence; never choose a direction."""

    required = ("y_long_path_utility_bps", "y_short_path_utility_bps")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"canonical outcome utility columns missing: {missing}")
    long_score = pd.to_numeric(
        frame["y_long_path_utility_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    short_score = pd.to_numeric(
        frame["y_short_path_utility_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not (np.isfinite(long_score).all() and np.isfinite(short_score).all()):
        raise RuntimeError("canonical outcome utility columns contain non-finite values")
    return np.where(
        side == SIDE_FLAT,
        0.0,
        np.where(side == SIDE_LONG, long_score, short_score),
    )


def _summarize(frame: pd.DataFrame, mask: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    sub = frame.loc[mask].copy()
    sel = mask & selected
    sub_sel = frame.loc[sel].copy()
    all_side = _side_from_predictions(sub) if len(sub) else np.asarray([], dtype=np.int8)
    side = _side_from_predictions(sub_sel) if len(sub_sel) else np.asarray([], dtype=np.int8)
    label = pd.to_numeric(sub_sel["y_direction"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(label).all() or not np.isin(
        label, (SIDE_LONG, SIDE_SHORT, SIDE_FLAT)
    ).all():
        raise RuntimeError(
            "selected y_direction labels must be exact finite LONG/SHORT/FLAT ids"
        )
    pnl = _pnl_proxy_for_side(sub_sel, side) if len(sub_sel) else np.asarray([], dtype=np.float64)
    mae = pd.to_numeric(sub_sel["mae_first_n_bps"], errors="coerce")
    mfe = pd.to_numeric(sub_sel["mfe_first_n_bps"], errors="coerce")
    path = pd.to_numeric(sub_sel["path_quality_bps"], errors="coerce")
    entry_q = (
        _strict_vector_column(sub_sel, "entry_action_q_bps", 3)
        if len(sub_sel)
        else np.empty((0, 3), dtype=np.float64)
    )
    edge_diagnostic = (
        np.maximum(entry_q[:, SIDE_LONG], entry_q[:, SIDE_SHORT])
        - entry_q[:, SIDE_FLAT]
    )
    selected_rows = int(len(sub_sel))
    long_count = int(np.sum(side == SIDE_LONG))
    short_count = int(np.sum(side == SIDE_SHORT))
    label_int = label.astype(np.int8, copy=False)
    label_correct_count = int(np.sum(side == label_int))
    label_error_count = selected_rows - label_correct_count
    label_error_wilson = (
        direction_pocket_wilson_upper_95(
            failures=label_error_count,
            total=selected_rows,
        )
        if selected_rows
        else None
    )
    return {
        "rows": int(len(sub)),
        "selected_rows": selected_rows,
        "selected_rate": (int(len(sub_sel)) / int(len(sub)) if len(sub) else 0.0),
        "selected_side_long_count": long_count,
        "selected_side_short_count": short_count,
        "selected_side_long_rate": _rate(side == SIDE_LONG),
        "selected_side_short_rate": _rate(side == SIDE_SHORT),
        "selected_label_correct_count": label_correct_count,
        "selected_label_error_count": label_error_count,
        "selected_label_correct_rate": _rate(side == label_int),
        "selected_label_error_rate": _rate(side != label_int),
        "selected_label_error_wilson_upper_95": label_error_wilson,
        "model_direction_flat_count": int(np.sum(all_side == SIDE_FLAT)),
        "model_direction_flat_rate": _rate(all_side == SIDE_FLAT),
        "selected_label_long_rate": _rate(label == SIDE_LONG),
        "selected_label_short_rate": _rate(label == SIDE_SHORT),
        "selected_label_flat_rate": _rate(label == SIDE_FLAT),
        "selected_mean_edge_score_diagnostic": _safe_mean(edge_diagnostic),
        "selected_mean_q_long_bps": _safe_mean(entry_q[:, SIDE_LONG]),
        "selected_mean_q_short_bps": _safe_mean(entry_q[:, SIDE_SHORT]),
        "selected_mean_q_flat_bps": _safe_mean(entry_q[:, SIDE_FLAT]),
        "selected_mean_proxy_pnl_bps": _safe_mean(pnl),
        "selected_mean_mfe_first_n_bps": _safe_mean(mfe),
        "selected_mean_mae_first_n_bps": _safe_mean(mae),
        "selected_mean_path_quality_bps": _safe_mean(path),
    }


def _decision(
    summaries: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    max_error_rate = DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE
    min_rows = DIRECTION_POCKET_MIN_SELECTED_ROWS
    min_mean_proxy_pnl_bps = (
        DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE
    )
    failures: list[str] = []
    for name in DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS:
        row = summaries[name]
        pocket_rows = int(row["rows"])
        selected_rows = int(row["selected_rows"])
        if pocket_rows < min_rows:
            failures.append(
                f"{name} pocket support {pocket_rows} < {min_rows}; "
                "direction edge is unproven"
            )
        if selected_rows < min_rows:
            failures.append(
                f"{name} selected coverage {selected_rows} < {min_rows} "
                f"with pocket rows={pocket_rows}"
            )
        if int(row["selected_rows"]) < min_rows:
            continue
        error_rate = row["selected_label_error_rate"]
        if (
            error_rate is None
            or float(error_rate) > max_error_rate
        ):
            failures.append(
                f"{name} selected label error rate {error_rate} > "
                f"{max_error_rate:.3f}"
            )
        error_wilson = row["selected_label_error_wilson_upper_95"]
        if (
            error_wilson is None
            or float(error_wilson)
            > DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95
        ):
            failures.append(
                f"{name} selected label-error Wilson upper 95% "
                f"{error_wilson} > "
                f"{DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95:.3f}"
            )
        mean_pnl = row["selected_mean_proxy_pnl_bps"]
        if mean_pnl is None:
            failures.append(
                f"{name} selected mean proxy pnl is missing; utility edge is unproven"
            )
        elif float(mean_pnl) <= min_mean_proxy_pnl_bps:
            failures.append(
                f"{name} selected mean proxy pnl {float(mean_pnl):.2f} <= "
                f"{min_mean_proxy_pnl_bps:.2f}"
            )
    return ("PASS" if not failures else "FAIL", failures)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument(
        "--dataset-parquet",
        type=Path,
        required=True,
        help="explicit regular *_test.parquet bound into the immutable event",
    )
    ap.add_argument(
        "--predictions-parquet",
        type=Path,
        required=True,
        help="explicit timestamped authoritative selective_edge_predictions_<stamp>.parquet",
    )
    ap.add_argument(
        "--predictions-sha256",
        required=True,
        help="caller-pinned SHA-256 of the exact TEST prediction parquet",
    )
    ap.add_argument(
        "--prediction-report-json",
        type=Path,
        required=True,
        help="exact matching ENTRY_CANDIDATE_SELECTIVE_EDGE_<stamp>.json",
    )
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument(
        "--bundle-metadata-json",
        type=Path,
        required=True,
        help="explicit regular <bundle-dir>/bundle_metadata.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="explicit output directory for one timestamped immutable JSON event",
    )
    args = ap.parse_args()

    raw_out_dir = args.out_dir.expanduser()
    if raw_out_dir.is_symlink():
        raise RuntimeError(f"output directory cannot be a symlink: {raw_out_dir}")

    requested_bundle_dir = args.bundle_dir.expanduser()
    requested_dataset_dir = args.dataset_dir.expanduser()
    requested_pred_path = args.predictions_parquet.expanduser()
    requested_report_path = args.prediction_report_json.expanduser()
    failures: list[str] = []

    pred_path, prediction_report, prediction_evidence = (
        resolve_and_validate_prediction_evidence(
            requested_pred_path,
            expected_sha256=str(args.predictions_sha256),
            prediction_report_path=requested_report_path,
            bundle_dir=requested_bundle_dir,
            dataset_dir=requested_dataset_dir,
            expected_stage=RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
            expected_splits=(MODEL_NATIVE_REQUIRED_TEST_SPLIT,),
            expected_model=MODEL_NATIVE_REQUIRED_MODEL_NAME,
        )
    )
    bundle_dir = requested_bundle_dir.resolve()
    dataset_dir = requested_dataset_dir.resolve()
    requested_report_path = requested_report_path.resolve()
    meta_path, meta = _load_meta(bundle_dir, args.bundle_metadata_json)
    require_model_direction_decision_contract(
        meta,
        context=f"model-native direction pocket audit bundle {bundle_dir}",
    )
    signal_contract = meta["model_native_signal_contract"]
    require_model_native_signal_contract(
        signal_contract,
        context="MODEL_NATIVE_DIRECTION_POCKET_BUNDLE",
    )
    prediction_report_evidence = {
        "json_path": str(requested_report_path),
        "sha256": sha256_file(requested_report_path),
    }
    signal_names = [str(value) for value in meta["ordered_signal_names"]]
    contracted_signal_names = [str(value) for value in signal_contract["fields"]]
    if signal_names != contracted_signal_names or len(signal_names) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "bundle ordered_signal_names do not exactly match the model-native "
            f"{MODEL_NATIVE_SIGNAL_DIM}-field contract"
        )
    ctx_cont_names = [str(value) for value in meta["ordered_ctx_cont_names"]]
    ctx_cat_names = [str(value) for value in meta["ordered_ctx_cat_names"]]
    if len(MODEL_NATIVE_CTX_CONT_FIELDS) != EXPECTED_CTX_CONT_DIM:
        raise RuntimeError(
            "runtime context contract is not the required "
            f"{EXPECTED_CTX_CONT_DIM}-field model-native surface"
        )
    if len(MODEL_NATIVE_CTX_CAT_FIELDS) != EXPECTED_CTX_CAT_DIM:
        raise RuntimeError(
            "runtime categorical context contract is not the required 5-field surface"
        )
    if ctx_cont_names != list(MODEL_NATIVE_CTX_CONT_FIELDS):
        raise RuntimeError(
            "bundle ordered_ctx_cont_names mismatch exact "
            f"{EXPECTED_CTX_CONT_DIM}-field contract"
        )
    if ctx_cat_names != list(MODEL_NATIVE_CTX_CAT_FIELDS):
        raise RuntimeError("bundle ordered_ctx_cat_names mismatch exact 5-field contract")
    exact_dimensions = {
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": EXPECTED_CTX_CONT_DIM,
        "ctx_cat_dim": EXPECTED_CTX_CAT_DIM,
    }
    dimension_mismatches = {
        name: {"observed": meta[name], "expected": expected}
        for name, expected in exact_dimensions.items()
        if meta[name] != expected
    }
    if dimension_mismatches:
        raise RuntimeError(
            f"bundle model-native input dimensions mismatch: {dimension_mismatches}"
        )

    dataset_parquet = _require_explicit_dataset_parquet(
        dataset_dir,
        args.dataset_parquet,
        MODEL_NATIVE_REQUIRED_TEST_SPLIT,
    )
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
        "y_long_path_utility_bps",
        "y_short_path_utility_bps",
        "y_line_support_touch_held",
        "y_line_support_touch_mask",
        "y_line_resistance_touch_held",
        "y_line_resistance_touch_mask",
        "y_support_retest_continuation",
        "y_resistance_retest_continuation",
        "y_countertrend_short_trap",
        "y_countertrend_long_trap",
        "y_mtf_conflict_m5_vs_higher_side",
        "y_long_high_mae_low_mfe_early_failure",
        "y_short_high_mae_low_mfe_early_failure",
    ]
    import pyarrow.parquet as pq

    parquet_cols = set(pq.ParquetFile(dataset_parquet).schema_arrow.names)
    missing_dataset_cols = [name for name in ds_cols if name not in parquet_cols]
    if missing_dataset_cols:
        raise RuntimeError(
            "model-native dataset lacks required pocket/outcome columns: "
            f"{missing_dataset_cols}"
        )
    ds = pd.read_parquet(dataset_parquet, columns=ds_cols)
    ds["time"] = pd.to_datetime(ds["time"], utc=True, errors="raise")

    pred_cols = ["time", "split", "model", *MODEL_DIRECTION_REQUIRED_COLUMNS]
    pred_schema_cols = set(pq.ParquetFile(pred_path).schema_arrow.names)
    missing_pred_cols = [name for name in pred_cols if name not in pred_schema_cols]
    if missing_pred_cols:
        raise RuntimeError(
            "predictions parquet missing model_direction_argmax contract columns: "
            f"{missing_pred_cols}"
        )
    pred = pd.read_parquet(pred_path, columns=pred_cols)
    pred["time"] = pd.to_datetime(pred["time"], utc=True, errors="raise")
    pred = pred[
        (pred["split"] == MODEL_NATIVE_REQUIRED_TEST_SPLIT)
        & (pred["model"] == MODEL_NATIVE_REQUIRED_MODEL_NAME)
    ].copy()
    if pred.empty:
        raise RuntimeError(
            "prediction evidence has no candidate TEST rows"
        )

    for name, data in (("predictions", pred), ("dataset", ds)):
        dup = data["time"].duplicated(keep=False)
        if bool(dup.any()):
            raise RuntimeError(f"{name} has duplicate time rows, first={data.loc[dup, 'time'].head(5).tolist()}")
    dataset_coverage = _time_coverage_contract(ds["time"], label="TEST dataset")
    prediction_coverage = _time_coverage_contract(
        pred["time"], label="candidate TEST predictions"
    )
    if dataset_coverage != prediction_coverage:
        raise RuntimeError(
            "candidate prediction time coverage does not exactly equal the complete "
            f"TEST dataset: dataset={dataset_coverage} predictions={prediction_coverage}"
        )
    frame = pred.merge(
        ds,
        on="time",
        how="outer",
        suffixes=("_pred", ""),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = {
        name: int(np.sum(frame["_merge"].astype(str).to_numpy() == name))
        for name in ("left_only", "right_only", "both")
    }
    if unmatched["left_only"] or unmatched["right_only"]:
        raise RuntimeError(f"prediction/dataset time coverage mismatch: {unmatched}")
    frame = frame.drop(columns=["_merge"])
    if frame.empty:
        raise RuntimeError("empty prediction/dataset join")
    failures.extend(_model_direction_contract_failures(frame))

    snap = _matrix(frame["snap"])
    ctx_cont = _matrix(frame["ctx_cont"])
    ctx_cat = _matrix(frame["ctx_cat"])
    expected_matrix_widths = {
        "snap": (snap.shape[1], MODEL_NATIVE_SIGNAL_DIM),
        "ctx_cont": (ctx_cont.shape[1], EXPECTED_CTX_CONT_DIM),
        "ctx_cat": (ctx_cat.shape[1], EXPECTED_CTX_CAT_DIM),
    }
    bad_matrix_widths = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in expected_matrix_widths.items()
        if observed != expected
    }
    if bad_matrix_widths:
        raise RuntimeError(
            f"model-native dataset matrix widths mismatch: {bad_matrix_widths}"
        )

    # Audit-only market slices are defined by exact signed primitives.  They
    # do not enter inference and do not reproduce the retired hand-weighted
    # trend/session/chart scorebooks.
    #
    # 2026-08-19: the two price-vs-EMA fields were renamed `_bps` -> `_atr`
    # when the producing owner stopped dividing them by `close` and started
    # dividing them by the shared strictly-positive Wilder-14 ATR (they had
    # been carrying the volatility regime).  Every use below is a `> 0.0` /
    # `< 0.0` sign test and both denominators are strictly positive, so
    # sign((close-ema)/atr) === sign((close-ema)/close) and the slice
    # boundaries are bit-identical.  No threshold was introduced or moved.
    local_spread_atr = _named_column(
        snap, signal_names, "chart.local_ema50_200_spread_atr"
    )
    local_price_vs_ema50 = _named_column(
        snap, signal_names, "chart.local_price_vs_ema50_atr"
    )
    local_price_vs_ema200 = _named_column(
        snap, signal_names, "chart.local_price_vs_ema200_atr"
    )
    m15_ema_spread = _named_column(
        ctx_cont, ctx_cont_names, "m15_ema5_20_spread_atr_canon_v2"
    )
    h1_ema = _named_column(ctx_cont, ctx_cont_names, "_v1h1_ema_diff")
    d1_slope = _named_column(ctx_cont, ctx_cont_names, "d1_ema_slope_20_canon_v2")
    h4_ema = _named_column(ctx_cont, ctx_cont_names, "_v1h4_ema_diff")
    h4_mid_ema50_distance = _named_column(
        ctx_cont, ctx_cont_names, "h4_mid_ema50_dist_atr_canon_v2"
    )
    level_below_dist = _named_column(snap, signal_names, "level_below_dist_atr")
    level_above_dist = _named_column(snap, signal_names, "level_above_dist_atr")
    line_below_dist = _named_column(
        snap, signal_names, "chart.geomline_below_dist_atr"
    )
    line_above_dist = _named_column(
        snap, signal_names, "chart.geomline_above_dist_atr"
    )
    selected = _selected_from_predictions(frame)
    intraday_bull = (
        (local_spread_atr > 0.0)
        & (local_price_vs_ema50 > 0.0)
        & (local_price_vs_ema200 > 0.0)
    )
    intraday_bear = (
        (local_spread_atr < 0.0)
        & (local_price_vs_ema50 < 0.0)
        & (local_price_vs_ema200 < 0.0)
    )
    htf_bull = (
        (m15_ema_spread > 0.0)
        & (h1_ema > 0.0)
        & (h4_ema > 0.0)
        & (d1_slope > 0.0)
        & (h4_mid_ema50_distance > 0.0)
    )
    htf_bear = (
        (m15_ema_spread < 0.0)
        & (h1_ema < 0.0)
        & (h4_ema < 0.0)
        & (d1_slope < 0.0)
        & (h4_mid_ema50_distance < 0.0)
    )
    def label_mask(name: str) -> np.ndarray:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"required pocket label {name!r} contains non-finite values")
        if not np.isin(values, (0.0, 1.0)).all():
            invalid = sorted(set(float(value) for value in values) - {0.0, 1.0})
            raise RuntimeError(
                f"required pocket label {name!r} is not binary: {invalid[:10]}"
            )
        return values == 1.0

    # V29 stage 2: the same-bar tautology labels were replaced by
    # forward-realized, touch-event-masked registry line-hold labels.  The
    # pocket masks select the DEFINED (touch-event) rows via the emitted
    # masks; the held outcome remains available per row.
    rising_channel_support = label_mask("y_line_support_touch_mask")
    falling_channel_resistance = label_mask("y_line_resistance_touch_mask")

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
    decision, gate_failures = _decision(summaries)
    failures.extend(gate_failures)
    decision = "PASS" if not failures else "FAIL"

    report = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION,
        "created_utc": _utc_now(),
        "decision": decision,
        "failures": failures,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "entry_decision_mode": MODEL_DIRECTION_SELECTION_MODE,
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": EXPECTED_CTX_CONT_DIM,
        "ctx_cat_dim": EXPECTED_CTX_CAT_DIM,
        "model_native_signal_contract": signal_contract,
        "direction_decision_contract": meta["direction_decision_contract"],
        "live_direction_authority": (
            "unique_argmax(entry_action_q_bps)"
        ),
        "audit_gate_scope": "offline_launch_evidence_only",
        "audit_thresholds_are_live_direction_rules": False,
        "auxiliary_evidence_direction_authority": "none",
        "bundle_dir": str(bundle_dir),
        "bundle_metadata_json": str(meta_path),
        "bundle_metadata_sha256": sha256_file(meta_path),
        "predictions_parquet": str(pred_path),
        "requested_predictions_parquet": str(requested_pred_path),
        "requested_prediction_report_json": str(requested_report_path),
        "prediction_report_json": str(prediction_report["json_path"]),
        "prediction_report_evidence": prediction_report_evidence,
        "prediction_evidence": prediction_evidence,
        "dataset_dir": str(dataset_dir),
        "dataset_parquet": str(dataset_parquet),
        "dataset_parquet_sha256": sha256_file(dataset_parquet),
        "split": MODEL_NATIVE_REQUIRED_TEST_SPLIT,
        "model_name": MODEL_NATIVE_REQUIRED_MODEL_NAME,
        "required_selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "observed_selection_score_modes": _selection_score_mode_values(frame),
        "required_prediction_columns": list(MODEL_DIRECTION_REQUIRED_COLUMNS),
        "selection_policy": "unique_argmax(entry_action_q_bps) != FLAT",
        "max_selected_label_error_rate": (
            DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE
        ),
        "max_selected_label_error_wilson_upper_95": (
            DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95
        ),
        "wilson_confidence_level": DIRECTION_POCKET_WILSON_CONFIDENCE_LEVEL,
        "min_selected_rows": DIRECTION_POCKET_MIN_SELECTED_ROWS,
        "min_mean_proxy_pnl_bps_exclusive": (
            DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE
        ),
        "spread_aware_proxy_pnl_contract": (
            DIRECTION_POCKET_SPREAD_AWARE_PROXY_CONTRACT
        ),
        "test_coverage": {
            "dataset": dataset_coverage,
            "predictions": prediction_coverage,
            "exact_match": True,
        },
        "features": {
            "intraday": [
                "chart.local_ema50_200_spread_atr",
                "chart.local_price_vs_ema50_atr",
                "chart.local_price_vs_ema200_atr",
            ],
            "higher_tf": [
                "m15_ema5_20_spread_atr_canon_v2",
                "_v1h1_ema_diff",
                "d1_ema_slope_20_canon_v2",
                "_v1h4_ema_diff",
                "h4_mid_ema50_dist_atr_canon_v2",
            ],
            "identified_support_resistance_distances": [
                "level_below_dist_atr",
                "level_above_dist_atr",
                "chart.geomline_below_dist_atr",
                "chart.geomline_above_dist_atr",
            ],
            "future_outcome_pocket_labels": [
                "y_line_support_touch_held",
                "y_line_support_touch_mask",
                "y_line_resistance_touch_held",
                "y_line_resistance_touch_mask",
                "y_support_retest_continuation",
                "y_resistance_retest_continuation",
                "y_countertrend_short_trap",
                "y_countertrend_long_trap",
                "y_long_high_mae_low_mfe_early_failure",
                "y_short_high_mae_low_mfe_early_failure",
            ],
        },
        "diagnostic_model_input_evidence": {
            "local_ema50_200_spread_atr": _finite_distribution(local_spread_atr),
            "local_price_vs_ema50_atr": _finite_distribution(local_price_vs_ema50),
            "local_price_vs_ema200_atr": _finite_distribution(local_price_vs_ema200),
            "m15_ema5_20_spread_atr": _finite_distribution(m15_ema_spread),
            "h1_ema_diff": _finite_distribution(h1_ema),
            "h4_ema_diff": _finite_distribution(h4_ema),
            "h4_mid_ema50_dist_atr": _finite_distribution(
                h4_mid_ema50_distance
            ),
            "d1_ema_slope_20": _finite_distribution(d1_slope),
            "level_below_dist_atr": _finite_distribution(level_below_dist),
            "level_above_dist_atr": _finite_distribution(level_above_dist),
            "geomline_below_dist_atr": _finite_distribution(line_below_dist),
            "geomline_above_dist_atr": _finite_distribution(line_above_dist),
        },
        "pockets": summaries,
    }

    if not failures:
        contract_failures = serve_gate_event_contract_failures(
            report,
            evidence_name="model_native_direction_pocket_audit",
        )
        if contract_failures:
            failures.extend(contract_failures)
            report["failures"] = list(failures)
            report["decision"] = "FAIL"
            decision = "FAIL"

    json_path, report = write_immutable_json_event(
        args.out_dir,
        EVENT_PREFIX,
        report,
    )
    print(json.dumps({"decision": decision, "failures": failures, "json": str(json_path)}, indent=2))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
