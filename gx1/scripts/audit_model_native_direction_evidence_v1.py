#!/usr/bin/env python3
"""Audit row-level evidence behind model-native XAU direction decisions.

This launch-blocking research audit joins an immutable seq513 prediction event
to its exact training/replay split.  It exposes final direction logits together
with contracted trend, MTF, geometry, rail, path-quality, and utility evidence.

Only persisted ``pred_direction == argmax(direction_logits)`` has direction
authority.  Every other value is diagnostic; none may select, suppress, or
replace LONG/SHORT/FLAT.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_NAME_BY_INDEX,
    MODEL_DIRECTION_SHORT_INDEX,
    require_model_direction_decision_contract,
)
from gx1.scripts.audit_model_native_direction_pockets_v1 import (
    MODEL_DIRECTION_REQUIRED_COLUMNS,
    MODEL_DIRECTION_SELECTION_MODE,
    _model_direction_contract_failures,
    _selected_from_predictions,
    _side_from_predictions,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
    resolve_and_validate_prediction_evidence,
    sha256_file,
)


DEFAULT_OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/model_native_direction_evidence_audit_v1"
)
EVENT_PREFIX = "MODEL_NATIVE_DIRECTION_EVIDENCE_AUDIT"
ROWS_PREFIX = "MODEL_NATIVE_DIRECTION_EVIDENCE_ROWS"
SIDE_NAME = MODEL_DIRECTION_NAME_BY_INDEX
FORBIDDEN_PREDICTION_COLUMNS = frozenset(
    {
        "anchor_logits",
        "anchor_gate",
        "anchor_logits_long_minus_short",
        "anchor_gate_long_minus_short",
        "delta_logits",
        "delta_logits_long_minus_short",
    }
)


def _prediction_report_split_parquet(
    prediction_report: dict[str, Any],
    dataset_dir: Path,
    split: str,
) -> Path:
    contract = prediction_report.get("dataset_signal_contract")
    rows = contract.get("splits") if isinstance(contract, dict) else None
    row = rows.get(split) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        raise RuntimeError(
            f"prediction report lacks exact {split} dataset artifact binding"
        )
    paths: dict[str, Path] = {}
    for kind, suffix in (
        ("manifest", f"_{split}.manifest.json"),
        ("parquet", f"_{split}.parquet"),
    ):
        path = Path(str(row.get(f"{kind}_path") or "")).expanduser()
        expected_sha = str(row.get(f"{kind}_sha256") or "").strip().lower()
        if (
            not path.is_absolute()
            or path.resolve() != path
            or path.is_symlink()
            or not path.is_file()
            or path.parent != dataset_dir
            or not path.name.endswith(suffix)
            or any("latest" in part.lower() for part in path.parts)
        ):
            raise RuntimeError(
                f"prediction report {split} {kind} identity is invalid: {path}"
            )
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise RuntimeError(
                f"prediction report {split} {kind} lacks SHA-256"
            )
        if sha256_file(path) != expected_sha:
            raise RuntimeError(
                f"prediction report {split} {kind} hash mismatch"
            )
        paths[kind] = path
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if Path(str(manifest.get("output_data_path") or "")).expanduser() != paths[
        "parquet"
    ]:
        raise RuntimeError(
            f"prediction report {split} manifest output_data_path mismatch"
        )
    if sha256_file(paths["manifest"]) != str(
        row["manifest_sha256"]
    ).strip().lower():
        raise RuntimeError(
            f"prediction report {split} manifest changed during validation"
        )
    return paths["parquet"]


def _schema_cols(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def _matrix(series: pd.Series) -> np.ndarray:
    try:
        out = np.stack(
            [np.asarray(value, dtype=np.float32) for value in series.to_numpy()]
        )
    except Exception as exc:
        raise RuntimeError(f"{series.name!r} is not a dense numeric matrix") from exc
    if out.ndim != 2 or not np.isfinite(out).all():
        raise RuntimeError(
            f"{series.name!r} must be a finite rank-2 matrix; got shape={out.shape}"
        )
    return out


def _vector_matrix(frame: pd.DataFrame, col: str, width: int) -> np.ndarray:
    if col not in frame.columns:
        raise RuntimeError(f"predictions parquet missing required vector column: {col}")
    out = _matrix(frame[col])
    if out.shape != (len(frame), width):
        raise RuntimeError(
            f"prediction column {col!r} must have shape ({len(frame)},{width}); "
            f"got {out.shape}"
        )
    return out.astype(np.float64)


def _first(matrix: np.ndarray, names: list[str], candidates: list[str]) -> np.ndarray:
    for name in candidates:
        if name in names:
            index = int(names.index(name))
            if index >= matrix.shape[1]:
                raise RuntimeError(
                    f"feature {name!r} index {index} outside matrix width {matrix.shape[1]}"
                )
            values = matrix[:, index].astype(np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"required model input {name!r} is non-finite")
            return values
    raise RuntimeError(f"required feature candidates missing: {candidates}")


def _finite_num(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        raise RuntimeError(f"required numeric column missing: {col}")
    out = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(out).all():
        bad = np.flatnonzero(~np.isfinite(out))[:10].tolist()
        raise RuntimeError(f"required numeric column {col!r} is non-finite at rows {bad}")
    return out


def _chosen_side(frame: pd.DataFrame) -> np.ndarray:
    return _side_from_predictions(frame)


def _selected(frame: pd.DataFrame) -> np.ndarray:
    return _selected_from_predictions(frame)


def _fmt_side(side: int) -> str:
    return SIDE_NAME.get(int(side), f"UNKNOWN_{side}")


def _why(row: pd.Series) -> str:
    side = _fmt_side(int(row["pred_direction"]))
    return (
        f"{side}: pred_direction={int(row['pred_direction'])} "
        f"public_trade={row['public_trade_probability']:.4f} "
        f"public_flat={row['public_flat_probability']:.4f} "
        f"public_margin={row['public_trade_flat_margin']:.4f} "
        f"public_hard={int(row['public_trade_flat_hard_decision'])}; "
        f"p_long={row['p_long']:.4f} p_short={row['p_short']:.4f} "
        f"p_flat={row['p_flat']:.4f} edge_diag={row['edge_score_diagnostic']:.4f}; "
        f"logit_LS={row['direction_logit_long_minus_short']:.4f} "
        f"logit_LF={row['direction_logit_long_minus_flat']:.4f} "
        f"logit_SF={row['direction_logit_short_minus_flat']:.4f}; "
        f"trend={row['mtf_trend_evidence']:.3f} "
        f"mtf_conflict={row['mtf_conflict']:.3f} "
        f"support={row['support_evidence']:.3f} "
        f"resistance={row['resistance_evidence']:.3f} "
        f"rail_LS={row['trendline_rail_long_minus_short']:.3f}; "
        f"pred_util_L={row['long_path_utility_pred_bps']:.2f} "
        f"pred_util_S={row['short_path_utility_pred_bps']:.2f}; "
        f"label_util_L={row['y_long_path_utility_bps']:.2f} "
        f"label_util_S={row['y_short_path_utility_bps']:.2f} "
        f"mae={row['mae_first_n_bps']:.2f} mfe={row['mfe_first_n_bps']:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--predictions-parquet",
        type=Path,
        required=True,
        help="explicit immutable selective_edge_predictions_<stamp>.parquet",
    )
    parser.add_argument(
        "--prediction-report-json",
        type=Path,
        required=True,
        help="matching newest immutable ENTRY_CANDIDATE_SELECTIVE_EDGE_<stamp>.json",
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="candidate")
    parser.add_argument(
        "--focus-start",
        help="optional explicit UTC start for an additional forensic focus window",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    requested_pred_path = args.predictions_parquet.expanduser().resolve()
    requested_report_path = args.prediction_report_json.expanduser().resolve()
    failures: list[str] = []

    pred_path, prediction_report, prediction_evidence = (
        resolve_and_validate_prediction_evidence(
            requested_pred_path,
            prediction_report_path=requested_report_path,
            bundle_dir=bundle_dir,
            dataset_dir=dataset_dir,
            expected_split=str(args.split),
            expected_model=str(args.model_name),
        )
    )

    meta_path = bundle_dir / "bundle_metadata.json"
    if not meta_path.is_file():
        raise RuntimeError(f"missing bundle metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    require_model_direction_decision_contract(
        meta,
        context=f"model-native direction evidence audit bundle {bundle_dir}",
    )
    signal_contract = meta.get("model_native_signal_contract")
    require_model_native_signal_contract(
        signal_contract,
        context="MODEL_NATIVE_DIRECTION_EVIDENCE_BUNDLE",
    )
    signal_names = [str(value) for value in (meta.get("ordered_signal_names") or [])]
    contracted_signal_names = [str(value) for value in signal_contract["fields"]]
    if signal_names != contracted_signal_names or len(signal_names) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "bundle ordered_signal_names do not exactly match the model-native "
            f"{MODEL_NATIVE_SIGNAL_DIM}-field contract"
        )
    ctx_cont_names = [
        str(value) for value in (meta.get("ordered_ctx_cont_names") or [])
    ]
    if not ctx_cont_names:
        raise RuntimeError("bundle metadata lacks ordered_ctx_cont_names")

    dataset_parquet = _prediction_report_split_parquet(
        prediction_report,
        dataset_dir,
        str(args.split),
    )
    dataset_required = [
        "time",
        "snap",
        "ctx_cont",
        "y_direction",
        "y_tradable",
        "mae_first_n_bps",
        "mfe_first_n_bps",
        "path_quality_bps",
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
    dataset_columns = _schema_cols(dataset_parquet)
    missing_dataset = [
        name for name in dataset_required if name not in dataset_columns
    ]
    if missing_dataset:
        raise RuntimeError(
            "model-native dataset lacks required feature/utility/path labels: "
            f"{missing_dataset}"
        )
    dataset = pd.read_parquet(dataset_parquet, columns=dataset_required)
    dataset["time"] = pd.to_datetime(dataset["time"], utc=True)

    prediction_columns = _schema_cols(pred_path)
    forbidden_present = sorted(
        FORBIDDEN_PREDICTION_COLUMNS & prediction_columns
    )
    if forbidden_present:
        raise RuntimeError(
            "model-native prediction evidence contains retired anchor/residual "
            f"columns: {forbidden_present}"
        )
    prediction_required = list(
        dict.fromkeys(
            [
                "time",
                "split",
                "model",
                *MODEL_DIRECTION_REQUIRED_COLUMNS,
                "direction_logits",
                "path_quality_pred",
                "bad_path_prob",
                "long_path_utility_pred_bps",
                "short_path_utility_pred_bps",
                "long_bad_path_prob",
                "short_bad_path_prob",
                "long_expected_mae_bps",
                "short_expected_mae_bps",
            ]
        )
    )
    missing_predictions = [
        name for name in prediction_required if name not in prediction_columns
    ]
    if missing_predictions:
        raise RuntimeError(
            "model-native predictions lack required final-direction/utility/path "
            f"evidence: {missing_predictions}"
        )
    predictions = pd.read_parquet(pred_path, columns=prediction_required)
    predictions["time"] = pd.to_datetime(predictions["time"], utc=True)
    predictions = predictions[
        (predictions["split"].astype(str) == str(args.split))
        & (predictions["model"].astype(str) == str(args.model_name))
    ].copy()
    if predictions.empty:
        raise RuntimeError(
            f"prediction evidence has no rows for split={args.split!r} "
            f"model={args.model_name!r}"
        )

    for name, data in (("predictions", predictions), ("dataset", dataset)):
        duplicate = data["time"].duplicated(keep=False)
        if bool(duplicate.any()):
            raise RuntimeError(
                f"{name} has duplicate time rows, "
                f"first={data.loc[duplicate, 'time'].head(5).tolist()}"
            )

    frame = predictions.merge(
        dataset,
        on="time",
        how="outer",
        suffixes=("_pred", ""),
        indicator=True,
        validate="one_to_one",
    )
    coverage = {
        str(key): int(value)
        for key, value in frame["_merge"].value_counts().to_dict().items()
    }
    if coverage.get("left_only", 0) or coverage.get("right_only", 0):
        failures.append(f"prediction/dataset time coverage mismatch: {coverage}")
    frame = frame.loc[frame["_merge"] == "both"].drop(columns=["_merge"]).copy()
    if frame.empty:
        raise RuntimeError("empty prediction/dataset join")
    failures.extend(_model_direction_contract_failures(frame))

    snap = _matrix(frame["snap"])
    ctx_cont = _matrix(frame["ctx_cont"])
    if snap.shape[1] != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            f"snap width={snap.shape[1]} expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    if ctx_cont.shape[1] != len(ctx_cont_names):
        raise RuntimeError(
            f"ctx_cont width={ctx_cont.shape[1]} metadata_names={len(ctx_cont_names)}"
        )

    direction_logits = _vector_matrix(frame, "direction_logits", 3)
    p_long = _finite_num(frame, "p_long")
    p_short = _finite_num(frame, "p_short")
    p_flat = _finite_num(frame, "p_flat")
    edge_diagnostic = np.maximum(p_long, p_short) - p_flat

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
    channel_edge = _first(
        snap, signal_names, ["chart.geometry_channel_edge_pressure"]
    )
    rising_support_rail_long = _first(
        snap,
        signal_names,
        ["chart.geometry_rising_support_rail_long_pressure"],
    )
    rising_support_rail_short_trap = _first(
        snap,
        signal_names,
        ["chart.geometry_rising_support_rail_short_trap_pressure"],
    )
    falling_resistance_rail_short = _first(
        snap,
        signal_names,
        ["chart.geometry_falling_resistance_rail_short_pressure"],
    )
    falling_resistance_rail_long_trap = _first(
        snap,
        signal_names,
        ["chart.geometry_falling_resistance_rail_long_trap_pressure"],
    )
    trendline_rail_long_evidence = np.mean(
        np.vstack([rising_support_rail_long, falling_resistance_rail_long_trap]),
        axis=0,
    )
    trendline_rail_short_evidence = np.mean(
        np.vstack([falling_resistance_rail_short, rising_support_rail_short_trap]),
        axis=0,
    )
    trendline_rail_long_minus_short = (
        trendline_rail_long_evidence - trendline_rail_short_evidence
    )
    mtf_trend = _first(
        snap,
        signal_names,
        ["trend.mtf_confluence_trend_direction_score"],
    )
    mtf_conflict = _first(
        snap,
        signal_names,
        ["trend.mtf_confluence_trend_tf_conflict"],
    )
    d1_slope = _first(
        ctx_cont,
        ctx_cont_names,
        ["d1_ema_slope_20_canon_v2"],
    )

    selected = _selected(frame)
    side = _chosen_side(frame)
    times = pd.to_datetime(frame["time"], utc=True)
    time_min = times.min()
    time_max = times.max()
    if args.focus_start:
        focus_start = pd.Timestamp(args.focus_start)
        focus_start = (
            focus_start.tz_localize("UTC")
            if focus_start.tzinfo is None
            else focus_start.tz_convert("UTC")
        )
        focus_window = (times >= focus_start).to_numpy(dtype=bool)
        if focus_start > time_max:
            failures.append(
                f"focus_start {focus_start} is after artifact time_max {time_max}"
            )
    else:
        focus_start = None
        focus_window = np.zeros(len(frame), dtype=bool)

    mae = _finite_num(frame, "mae_first_n_bps")
    mfe = _finite_num(frame, "mfe_first_n_bps")
    rising_support = (
        (_finite_num(frame, "y_rising_channel_support_touch") > 0.5)
        | (rising_support_rail_long >= 0.25)
        | (rising_support_rail_short_trap >= 0.25)
        | (
            (mtf_trend >= 0.0)
            & (support >= resistance)
            & (support >= 0.35)
            & (channel_edge >= 0.15)
        )
    )
    high_mae_low_mfe = selected & ((mae >= 6.0) & (mfe <= 2.0))
    rising_support_short = selected & rising_support & (side == 1)
    row_mask = selected | focus_window

    row_report = pd.DataFrame(
        {
            "time": times,
            "selected": selected,
            "focus_window": focus_window,
            "high_mae_low_mfe_loser": high_mae_low_mfe,
            "rising_support_selected_short": rising_support_short,
            "pred_direction": side,
            "model_action": [_fmt_side(value) for value in side],
            "y_direction": _finite_num(frame, "y_direction"),
            "y_tradable": _finite_num(frame, "y_tradable"),
            "p_long": p_long,
            "p_short": p_short,
            "p_flat": p_flat,
            "public_trade_probability": _finite_num(
                frame, "public_trade_probability"
            ),
            "public_flat_probability": _finite_num(
                frame, "public_flat_probability"
            ),
            "public_trade_flat_margin": _finite_num(
                frame, "public_trade_flat_margin"
            ),
            "public_trade_flat_hard_decision": _finite_num(
                frame, "public_trade_flat_hard_decision"
            ),
            "selection_score_mode": frame["selection_score_mode"].astype(str).to_numpy(),
            "edge_score_diagnostic": edge_diagnostic,
            "direction_logit_long_minus_short": (
                direction_logits[:, MODEL_DIRECTION_LONG_INDEX]
                - direction_logits[:, MODEL_DIRECTION_SHORT_INDEX]
            ),
            "direction_logit_long_minus_flat": (
                direction_logits[:, MODEL_DIRECTION_LONG_INDEX]
                - direction_logits[:, MODEL_DIRECTION_FLAT_INDEX]
            ),
            "direction_logit_short_minus_flat": (
                direction_logits[:, MODEL_DIRECTION_SHORT_INDEX]
                - direction_logits[:, MODEL_DIRECTION_FLAT_INDEX]
            ),
            "path_quality_pred": _finite_num(frame, "path_quality_pred"),
            "bad_path_prob": _finite_num(frame, "bad_path_prob"),
            "long_path_utility_pred_bps": _finite_num(
                frame, "long_path_utility_pred_bps"
            ),
            "short_path_utility_pred_bps": _finite_num(
                frame, "short_path_utility_pred_bps"
            ),
            "long_bad_path_prob": _finite_num(frame, "long_bad_path_prob"),
            "short_bad_path_prob": _finite_num(frame, "short_bad_path_prob"),
            "long_expected_mae_bps": _finite_num(
                frame, "long_expected_mae_bps"
            ),
            "short_expected_mae_bps": _finite_num(
                frame, "short_expected_mae_bps"
            ),
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
            "path_quality_bps": _finite_num(frame, "path_quality_bps"),
            "y_long_path_utility_bps": _finite_num(
                frame, "y_long_path_utility_bps"
            ),
            "y_short_path_utility_bps": _finite_num(
                frame, "y_short_path_utility_bps"
            ),
            "y_long_bad_path": _finite_num(frame, "y_long_bad_path"),
            "y_short_bad_path": _finite_num(frame, "y_short_bad_path"),
            "y_long_expected_mae_bps": _finite_num(
                frame, "y_long_expected_mae_bps"
            ),
            "y_short_expected_mae_bps": _finite_num(
                frame, "y_short_expected_mae_bps"
            ),
            "y_countertrend_short_trap": _finite_num(
                frame, "y_countertrend_short_trap"
            ),
            "y_countertrend_long_trap": _finite_num(
                frame, "y_countertrend_long_trap"
            ),
            "y_mtf_conflict_m5_vs_higher_side": _finite_num(
                frame, "y_mtf_conflict_m5_vs_higher_side"
            ),
        }
    )
    row_report = row_report.loc[row_mask].copy()
    if len(row_report):
        row_report["why_this_side"] = row_report.apply(_why, axis=1)
    if args.focus_start and int(focus_window.sum()) == 0:
        failures.append("focus population empty for explicit --focus-start window")
    if int(selected.sum()) == 0:
        failures.append("selected population empty")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    parquet_path = out_dir / f"{ROWS_PREFIX}_{stamp}.parquet"
    atomic_write_parquet_immutable(row_report, parquet_path)

    report = {
        "schema_version": "model_native_direction_evidence_audit_v1",
        "created_utc": created.isoformat(),
        "bundle_dir": str(bundle_dir),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "dataset_dir": str(dataset_dir),
        "dataset_parquet": str(dataset_parquet),
        "predictions_parquet": str(pred_path),
        "requested_predictions_parquet": str(requested_pred_path),
        "requested_prediction_report_json": str(requested_report_path),
        "prediction_report_json": str(prediction_report.get("json_path") or ""),
        "prediction_report_evidence": {
            "json_path": str(requested_report_path),
            "sha256": sha256_file(requested_report_path),
        },
        "prediction_evidence": prediction_evidence,
        "split": str(args.split),
        "model_name": str(args.model_name),
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "required_prediction_columns": prediction_required,
        "forbidden_prediction_columns": sorted(FORBIDDEN_PREDICTION_COLUMNS),
        "model_native_signal_contract": signal_contract,
        "selection_policy": "pred_direction != FLAT",
        "join_coverage": coverage,
        "rows_joined": int(len(frame)),
        "time_min": str(time_min),
        "time_max": str(time_max),
        "focus_start": str(focus_start) if focus_start is not None else None,
        "selected_rows": int(selected.sum()),
        "model_direction_counts": {
            _fmt_side(direction): int(np.sum(side == direction))
            for direction in (0, 1, 2)
        },
        "row_report_rows": int(len(row_report)),
        "focus_counts": {
            "explicit_focus_window": int(focus_window.sum()),
            "high_mae_low_mfe_loser": int(high_mae_low_mfe.sum()),
            "rising_support_selected_short": int(rising_support_short.sum()),
        },
        "row_report_parquet": str(parquet_path),
        "row_report_parquet_sha256": sha256_file(parquet_path),
        "component_notes": {
            "direction_authority": (
                "Only final calibrated direction_logits argmax selects LONG/SHORT/FLAT."
            ),
            "feature_evidence": (
                "Trend, MTF, geometry and rail values are contracted model inputs; "
                "utility/path values are supervised diagnostics and never overrides."
            ),
            "retired_surfaces": (
                "Anchor, delta and signal-bridge probability columns are rejected."
            ),
        },
    }
    json_path, published = write_immutable_json_event(
        out_dir,
        EVENT_PREFIX,
        report,
    )
    print(
        json.dumps(
            {
                "decision": published["decision"],
                "failures": failures,
                "json": str(json_path),
                "rows": int(len(row_report)),
            },
            indent=2,
        )
    )
    return 0 if published["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
