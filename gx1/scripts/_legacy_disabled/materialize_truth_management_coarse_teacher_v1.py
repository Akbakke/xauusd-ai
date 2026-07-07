from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_SUFFIX = "MANAGEMENT_AUDIT_EXTENSION_V1"
BUILD_SUMMARY_FILE = "shadow_meta_all_trade_review_management_audit_extension_build_summary_v1.json"
DECISION_LOG_FILE = "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet"
OUTCOME_BACKFILL_FILE = "shadow_meta_all_trade_review_management_policy_logging_outcome_backfill_harness_v1.parquet"
QUALITY_AS_OF_FILE = "shadow_meta_all_trade_review_management_outcome_quality_regime_audit_as_of_v1.parquet"
QUALITY_HINDSIGHT_FILE = "shadow_meta_all_trade_review_management_outcome_quality_regime_audit_hindsight_v1.parquet"

VIEW_FILE = "truth_management_coarse_teacher_v1.parquet"
CELL_ROLLUP_FILE = "truth_management_coarse_teacher_cell_rollup_v1.csv"
SUMMARY_FILE = "truth_management_coarse_teacher_summary_v1.json"

JOIN_KEYS = ["candidate_uid_exact_v1", "trade_uid_exact_v1", "trade_id_exact_v1"]
GRID_DEFS = {
    "SESSION_ONLY": ["overlay_session_axis_v1"],
    "SESSION_HOLD_GIVEBACK": [
        "overlay_session_axis_v1",
        "overlay_hold_age_axis_v1",
        "overlay_giveback_axis_v1",
    ],
    "SESSION_VOL_HOLD_GIVEBACK": [
        "overlay_session_axis_v1",
        "overlay_vol_axis_v1",
        "overlay_hold_age_axis_v1",
        "overlay_giveback_axis_v1",
    ],
}


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None = None) -> Path:
    if extension_dir_arg:
        path = Path(extension_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Extension dir does not exist: {path}")
        return path

    candidates = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir()
            and path.name.endswith(EXTENSION_SUFFIX)
            and (path / BUILD_SUMMARY_FILE).exists()
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No extension dirs with {BUILD_SUMMARY_FILE} found under {reports_root}"
        )
    return candidates[0]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{frame_name} missing required columns: {missing}")


def _require_unique_keys(frame: pd.DataFrame, keys: list[str], frame_name: str) -> None:
    duplicated = int(frame.duplicated(subset=keys).sum())
    if duplicated != 0:
        raise RuntimeError(f"{frame_name} has duplicate rows on {keys}: {duplicated}")


def _scalar_string(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    text = str(value).strip()
    return text if text else "NA"


def _safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", "", "na", "nan", "none"}:
        return False
    return False


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_scalar_bool).astype(bool)


def _feedback_payload(row: pd.Series) -> tuple[str, pd.Int64Dtype | Any]:
    action = _scalar_string(row.get("observed_action_v1"))
    strong = _scalar_bool(row.get("quality_band_strong_capture_v1", False))
    weak = _scalar_bool(row.get("quality_band_weak_capture_v1", False))
    good_exit = _scalar_bool(row.get("good_exit", False))
    premature_exit = _scalar_bool(row.get("premature_exit", False))
    late_exit = _scalar_bool(row.get("late_exit", False))

    if action == "HOLD" and strong:
        return "OBSERVED_HOLD_DEFENSIBLE", 1
    if action == "EXIT_NOW" and good_exit:
        return "OBSERVED_EXIT_DEFENSIBLE", 1
    if action == "EXIT_NOW" and premature_exit:
        return "OBSERVED_EXIT_TOO_EARLY", 0
    if action == "HOLD" and weak:
        return "OBSERVED_HOLD_TOO_WEAK", 0
    if action == "EXIT_NOW" and late_exit:
        return "OBSERVED_EXIT_TOO_LATE", pd.NA
    return "AMBIGUOUS_OR_OTHER", pd.NA


def build_management_coarse_teacher_payload(
    reports_root: Path,
    *,
    extension_dir: Path | None = None,
    min_cell_rows: int = 20,
    min_rows_per_action: int = 10,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    extension_dir = (
        extension_dir.expanduser().resolve()
        if extension_dir is not None
        else _resolve_extension_dir(reports_root)
    )

    decision_log_path = extension_dir / DECISION_LOG_FILE
    outcome_backfill_path = extension_dir / OUTCOME_BACKFILL_FILE
    quality_as_of_path = extension_dir / QUALITY_AS_OF_FILE
    quality_hindsight_path = extension_dir / QUALITY_HINDSIGHT_FILE

    for path in [decision_log_path, outcome_backfill_path, quality_as_of_path, quality_hindsight_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    decision_log_df = pd.read_parquet(decision_log_path)
    outcome_backfill_df = pd.read_parquet(outcome_backfill_path)
    quality_as_of_df = pd.read_parquet(quality_as_of_path)
    quality_hindsight_df = pd.read_parquet(quality_hindsight_path)

    if decision_log_df.empty:
        raise RuntimeError("decision_log_df is empty")
    if outcome_backfill_df.empty:
        raise RuntimeError("outcome_backfill_df is empty")
    if quality_as_of_df.empty:
        raise RuntimeError("quality_as_of_df is empty")
    if quality_hindsight_df.empty:
        raise RuntimeError("quality_hindsight_df is empty")

    _require_columns(
        decision_log_df,
        [
            "management_row_key_v1",
            "run_id",
            "as_of_row_uid_v1",
            "decision_ts_utc_v1",
            "decision_anchor_type_v1",
            "split_bucket_v1",
            "observed_action_v1",
            "shadow_score_v1",
            "shadow_bucket_status_v1",
            "shadow_bucket_rank_v1",
            "overlay_session_axis_v1",
            "overlay_vol_axis_v1",
            "overlay_hold_age_axis_v1",
            "overlay_giveback_axis_v1",
            "as_of_management_core_minutes_held_at_anchor_v1",
            "as_of_management_core_giveback_ratio_from_peak_v1",
            "as_of_atr_bps_v1",
            "as_of_hour_utc_v1",
            "as_of_session_v1",
            "as_of_side_v1",
            "as_of_trend_regime_v1",
            "as_of_vol_regime_v1",
        ]
        + JOIN_KEYS,
        "decision_log_df",
    )
    _require_columns(
        outcome_backfill_df,
        [
            "management_row_key_v1",
            "realized_pnl_bps",
            "mfe_bps",
            "mae_bps",
            "trade_outcome_class",
            "exit_reason",
            "good_exit",
            "premature_exit",
            "late_exit",
            "hindsight_peak_mfe_bps_v1",
            "hindsight_peak_to_exit_giveback_bps_v1",
        ],
        "outcome_backfill_df",
    )
    _require_columns(
        quality_as_of_df,
        JOIN_KEYS
        + [
            "walkforward_slice_v1",
            "walkforward_slice_start_utc_v1",
            "walkforward_slice_end_utc_v1",
        ],
        "quality_as_of_df",
    )
    _require_columns(
        quality_hindsight_df,
        JOIN_KEYS
        + [
            "realized_pnl_bps_v1",
            "mfe_bps_v1",
            "mae_bps_v1",
            "giveback_ratio_v1",
            "holding_time_bars_v1",
            "quality_band_strong_capture_v1",
            "quality_band_weak_capture_v1",
            "quality_band_high_giveback_v1",
            "quality_band_low_mae_v1",
            "quality_band_tail_risk_v1",
        ],
        "quality_hindsight_df",
    )

    _require_unique_keys(decision_log_df, ["management_row_key_v1"], "decision_log_df")
    _require_unique_keys(outcome_backfill_df, ["management_row_key_v1"], "outcome_backfill_df")
    _require_unique_keys(quality_as_of_df, JOIN_KEYS, "quality_as_of_df")
    _require_unique_keys(quality_hindsight_df, JOIN_KEYS, "quality_hindsight_df")

    view_df = decision_log_df.copy()
    view_df = view_df.merge(
        outcome_backfill_df[
            [
                "management_row_key_v1",
                "realized_pnl_bps",
                "mfe_bps",
                "mae_bps",
                "trade_outcome_class",
                "exit_reason",
                "good_exit",
                "premature_exit",
                "late_exit",
                "hindsight_peak_mfe_bps_v1",
                "hindsight_peak_to_exit_giveback_bps_v1",
            ]
        ],
        on="management_row_key_v1",
        how="left",
        validate="one_to_one",
    )
    view_df = view_df.merge(
        quality_as_of_df[
            JOIN_KEYS
            + [
                "walkforward_slice_v1",
                "walkforward_slice_start_utc_v1",
                "walkforward_slice_end_utc_v1",
            ]
        ],
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    view_df = view_df.merge(
        quality_hindsight_df[
            JOIN_KEYS
            + [
                "realized_pnl_bps_v1",
                "mfe_bps_v1",
                "mae_bps_v1",
                "giveback_ratio_v1",
                "holding_time_bars_v1",
                "quality_band_strong_capture_v1",
                "quality_band_weak_capture_v1",
                "quality_band_high_giveback_v1",
                "quality_band_low_mae_v1",
                "quality_band_tail_risk_v1",
            ]
        ],
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )

    if int(len(view_df)) != int(len(decision_log_df)):
        raise RuntimeError("Row count changed while building management coarse teacher view")

    for column in [
        "realized_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "hindsight_peak_mfe_bps_v1",
        "hindsight_peak_to_exit_giveback_bps_v1",
        "realized_pnl_bps_v1",
        "mfe_bps_v1",
        "mae_bps_v1",
        "giveback_ratio_v1",
        "holding_time_bars_v1",
        "shadow_score_v1",
        "as_of_management_core_minutes_held_at_anchor_v1",
        "as_of_management_core_giveback_ratio_from_peak_v1",
        "as_of_atr_bps_v1",
        "as_of_hour_utc_v1",
    ]:
        if column in view_df.columns:
            view_df[column] = pd.to_numeric(view_df[column], errors="coerce")

    for column in [
        "good_exit",
        "premature_exit",
        "late_exit",
        "quality_band_strong_capture_v1",
        "quality_band_weak_capture_v1",
        "quality_band_high_giveback_v1",
        "quality_band_low_mae_v1",
        "quality_band_tail_risk_v1",
    ]:
        if column in view_df.columns:
            view_df[column] = _coerce_bool_series(view_df[column])

    feedback_values = view_df.apply(_feedback_payload, axis=1)
    view_df["coarse_teacher_feedback_label_v1"] = feedback_values.map(lambda item: item[0]).astype("string")
    view_df["coarse_teacher_binary_target_v1"] = pd.array(
        [item[1] for item in feedback_values],
        dtype="Int64",
    )
    view_df["coarse_teacher_binary_target_eligible_v1"] = view_df["coarse_teacher_binary_target_v1"].notna()

    hold_longer_extra = pd.to_numeric(view_df["hindsight_peak_to_exit_giveback_bps_v1"], errors="coerce")
    if hold_longer_extra.isna().all():
        hold_longer_extra = (
            pd.to_numeric(view_df["hindsight_peak_mfe_bps_v1"], errors="coerce")
            - pd.to_numeric(view_df["realized_pnl_bps"], errors="coerce")
        )
    view_df["hold_longer_extra_value_bps_v1"] = hold_longer_extra.clip(lower=0.0)
    view_df["hold_longer_pressure_10bps_v1"] = view_df["hold_longer_extra_value_bps_v1"].ge(10.0)
    view_df["hold_longer_pressure_25bps_v1"] = view_df["hold_longer_extra_value_bps_v1"].ge(25.0)

    train_scores = pd.to_numeric(
        view_df.loc[view_df["split_bucket_v1"].astype("string").eq("TRAIN"), "shadow_score_v1"],
        errors="coerce",
    ).dropna()
    if train_scores.empty:
        raise RuntimeError("No TRAIN shadow scores available for coarse raw-score calibration")
    low_q = float(train_scores.quantile(1.0 / 3.0))
    high_q = float(train_scores.quantile(2.0 / 3.0))

    def _score_band(value: Any) -> str:
        try:
            score = float(value)
        except Exception:
            return "UNRESOLVED"
        if np.isnan(score) or np.isinf(score):
            return "UNRESOLVED"
        if score <= low_q:
            return "LOW"
        if score <= high_q:
            return "MID"
        return "HIGH"

    view_df["shadow_score_coarse_band_v1"] = view_df["shadow_score_v1"].map(_score_band).astype("string")
    view_df["shadow_score_sign_v1"] = np.where(
        pd.to_numeric(view_df["shadow_score_v1"], errors="coerce").gt(0.0),
        "POSITIVE",
        "NEGATIVE_OR_ZERO",
    )
    view_df["shadow_score_split_pct_rank_v1"] = (
        view_df.groupby("split_bucket_v1", dropna=False)["shadow_score_v1"]
        .rank(method="first", pct=True)
    )
    view_df["shadow_score_slice_pct_rank_v1"] = (
        view_df.groupby("walkforward_slice_v1", dropna=False)["shadow_score_v1"]
        .rank(method="first", pct=True)
    )

    for grid_name, cols in GRID_DEFS.items():
        view_df[f"coarse_grid_{grid_name.lower()}_v1"] = (
            view_df[cols].apply(lambda row: "|".join(_scalar_string(value) for value in row), axis=1).astype("string")
        )

    cell_rollup_rows: list[dict[str, Any]] = []
    grid_score_rows: list[dict[str, Any]] = []
    total_rows = int(len(view_df))
    for grid_name, cols in GRID_DEFS.items():
        grid_col = f"coarse_grid_{grid_name.lower()}_v1"
        grouped = view_df.groupby(grid_col, dropna=False)
        viable_cell_rows = 0
        eligible_teacher_rows = 0
        viable_cell_count = 0
        for grid_value, part in grouped:
            row_count = int(len(part))
            hold_part = part.loc[part["observed_action_v1"].astype("string").eq("HOLD")].copy()
            exit_part = part.loc[part["observed_action_v1"].astype("string").eq("EXIT_NOW")].copy()
            hold_rows = int(len(hold_part))
            exit_rows = int(len(exit_part))
            viable = row_count >= min_cell_rows and hold_rows >= min_rows_per_action and exit_rows >= min_rows_per_action
            teacher_rows = int(part["coarse_teacher_binary_target_eligible_v1"].sum())
            if viable:
                viable_cell_rows += row_count
                viable_cell_count += 1
            eligible_teacher_rows += teacher_rows
            split_counts = (
                part["split_bucket_v1"].astype("string").value_counts(dropna=False).to_dict()
                if "split_bucket_v1" in part.columns
                else {}
            )
            feedback_counts = part["coarse_teacher_feedback_label_v1"].astype("string").value_counts(dropna=False).to_dict()
            cell_rollup_rows.append(
                {
                    "grid_name_v1": grid_name,
                    "grid_value_v1": _scalar_string(grid_value),
                    "row_count_v1": row_count,
                    "hold_rows_v1": hold_rows,
                    "exit_rows_v1": exit_rows,
                    "coarse_teacher_binary_target_rows_v1": teacher_rows,
                    "positive_teacher_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
                    "negative_teacher_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum()),
                    "hold_positive_teacher_rows_v1": int(
                        hold_part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()
                    ),
                    "hold_negative_teacher_rows_v1": int(
                        hold_part["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum()
                    ),
                    "exit_positive_teacher_rows_v1": int(
                        exit_part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()
                    ),
                    "exit_negative_teacher_rows_v1": int(
                        exit_part["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum()
                    ),
                    "strong_capture_hold_rows_v1": int(hold_part["quality_band_strong_capture_v1"].fillna(False).sum()),
                    "weak_capture_hold_rows_v1": int(hold_part["quality_band_weak_capture_v1"].fillna(False).sum()),
                    "good_exit_exit_rows_v1": int(exit_part["good_exit"].fillna(False).sum()),
                    "premature_exit_exit_rows_v1": int(exit_part["premature_exit"].fillna(False).sum()),
                    "viable_cell_v1": bool(viable),
                    "teacher_positive_rate_v1": _safe_rate(
                        float(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
                        float(teacher_rows),
                    ),
                    "hold_teacher_positive_rate_v1": _safe_rate(
                        float(hold_part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
                        float(int(hold_part["coarse_teacher_binary_target_eligible_v1"].sum())),
                    ),
                    "exit_teacher_positive_rate_v1": _safe_rate(
                        float(exit_part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
                        float(int(exit_part["coarse_teacher_binary_target_eligible_v1"].sum())),
                    ),
                    "mean_shadow_score_v1": float(pd.to_numeric(part["shadow_score_v1"], errors="coerce").mean()),
                    "mean_realized_pnl_bps_v1": float(pd.to_numeric(part["realized_pnl_bps"], errors="coerce").mean()),
                    "mean_hold_longer_extra_value_bps_v1": float(
                        pd.to_numeric(part["hold_longer_extra_value_bps_v1"], errors="coerce").mean()
                    ),
                    "mean_exit_hold_longer_extra_value_bps_v1": float(
                        pd.to_numeric(exit_part["hold_longer_extra_value_bps_v1"], errors="coerce").mean()
                    )
                    if exit_rows > 0
                    else None,
                    "mean_hold_realized_pnl_bps_v1": float(
                        pd.to_numeric(hold_part["realized_pnl_bps"], errors="coerce").mean()
                    )
                    if hold_rows > 0
                    else None,
                    "split_counts_json_v1": json.dumps(
                        {str(key): int(value) for key, value in split_counts.items()},
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    "feedback_counts_json_v1": json.dumps(
                        {str(key): int(value) for key, value in feedback_counts.items()},
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                }
            )
        grid_score_rows.append(
            {
                "grid_name_v1": grid_name,
                "cell_count_v1": int(view_df[grid_col].astype("string").nunique(dropna=False)),
                "viable_cell_count_v1": viable_cell_count,
                "viable_row_share_v1": _safe_rate(float(viable_cell_rows), float(total_rows)),
                "teacher_row_share_v1": _safe_rate(float(eligible_teacher_rows), float(total_rows)),
                "grid_columns_v1": cols,
            }
        )

    cell_rollup_df = pd.DataFrame.from_records(cell_rollup_rows)
    grid_score_df = pd.DataFrame.from_records(grid_score_rows)
    if cell_rollup_df.empty or grid_score_df.empty:
        raise RuntimeError("Failed to build coarse teacher cell rollups")

    multiaxis_grid_scores = grid_score_df.loc[grid_score_df["grid_name_v1"].astype("string") != "SESSION_ONLY"].copy()
    if not multiaxis_grid_scores.empty:
        recommended_grid_row = multiaxis_grid_scores.sort_values(
            ["viable_row_share_v1", "teacher_row_share_v1", "viable_cell_count_v1"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]
    else:
        recommended_grid_row = grid_score_df.sort_values(
            ["viable_row_share_v1", "teacher_row_share_v1", "viable_cell_count_v1"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]
    recommended_grid_name = _scalar_string(recommended_grid_row["grid_name_v1"])
    recommended_grid_col = f"coarse_grid_{recommended_grid_name.lower()}_v1"
    recommended_cells_df = cell_rollup_df.loc[
        cell_rollup_df["grid_name_v1"].astype("string").eq(recommended_grid_name)
    ].copy()
    recommended_status_map = (
        recommended_cells_df[["grid_value_v1", "viable_cell_v1"]]
        .drop_duplicates(subset=["grid_value_v1"])
        .set_index("grid_value_v1")["viable_cell_v1"]
        .to_dict()
    )
    view_df["recommended_coarse_grid_name_v1"] = recommended_grid_name
    view_df["recommended_coarse_grid_value_v1"] = view_df[recommended_grid_col].astype("string")
    view_df["recommended_coarse_grid_viable_cell_v1"] = view_df["recommended_coarse_grid_value_v1"].map(
        lambda value: bool(recommended_status_map.get(_scalar_string(value), False))
    )
    view_df["coarse_teacher_row_status_v1"] = np.select(
        [
            view_df["coarse_teacher_binary_target_eligible_v1"] & view_df["recommended_coarse_grid_viable_cell_v1"],
            view_df["coarse_teacher_binary_target_eligible_v1"] & ~view_df["recommended_coarse_grid_viable_cell_v1"],
        ],
        [
            "ELIGIBLE_BINARY_FEEDBACK_IN_VIABLE_CELL",
            "ELIGIBLE_BINARY_FEEDBACK_IN_THIN_CELL",
        ],
        default="NO_BINARY_FEEDBACK_LABEL",
    )

    exit_pressure_cells = recommended_cells_df.loc[recommended_cells_df["exit_rows_v1"] >= min_rows_per_action].copy()
    exit_pressure_cells = exit_pressure_cells.sort_values(
        ["mean_exit_hold_longer_extra_value_bps_v1", "premature_exit_exit_rows_v1", "exit_rows_v1"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(10)

    hold_cells = recommended_cells_df.loc[recommended_cells_df["hold_rows_v1"] >= min_rows_per_action].copy()
    hold_cells = hold_cells.sort_values(
        ["strong_capture_hold_rows_v1", "hold_teacher_positive_rate_v1", "hold_rows_v1"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(10)

    summary = {
        "reports_root": str(reports_root),
        "extension_dir": str(extension_dir),
        "row_count_v1": int(len(view_df)),
        "join_contract_v1": {
            "decision_log_rows_v1": int(len(decision_log_df)),
            "outcome_backfill_rows_v1": int(len(outcome_backfill_df)),
            "quality_as_of_rows_v1": int(len(quality_as_of_df)),
            "quality_hindsight_rows_v1": int(len(quality_hindsight_df)),
            "view_rows_v1": int(len(view_df)),
        },
        "action_counts_v1": {
            str(key): int(value)
            for key, value in view_df["observed_action_v1"].astype("string").value_counts(dropna=False).to_dict().items()
        },
        "split_counts_v1": {
            str(key): int(value)
            for key, value in view_df["split_bucket_v1"].astype("string").value_counts(dropna=False).to_dict().items()
        },
        "coarse_teacher_feedback_counts_v1": {
            str(key): int(value)
            for key, value in view_df["coarse_teacher_feedback_label_v1"].astype("string").value_counts(dropna=False).to_dict().items()
        },
        "binary_teacher_target_summary_v1": {
            "eligible_rows_v1": int(view_df["coarse_teacher_binary_target_eligible_v1"].sum()),
            "positive_rows_v1": int(view_df["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
            "negative_rows_v1": int(view_df["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum()),
            "eligible_rate_v1": _safe_rate(
                float(view_df["coarse_teacher_binary_target_eligible_v1"].sum()),
                float(len(view_df)),
            ),
            "by_action_v1": {
                action: {
                    "eligible_rows_v1": int(part["coarse_teacher_binary_target_eligible_v1"].sum()),
                    "positive_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
                    "negative_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum()),
                }
                for action, part in view_df.groupby(view_df["observed_action_v1"].astype("string"), dropna=False)
            },
        },
        "feedback_action_balance_status_v1": {
            action: (
                "BALANCED_POSITIVE_AND_NEGATIVE"
                if stats["positive_rows_v1"] > 0 and stats["negative_rows_v1"] > 0
                else (
                    "POSITIVE_ONLY"
                    if stats["positive_rows_v1"] > 0 and stats["negative_rows_v1"] == 0
                    else (
                        "NEGATIVE_ONLY"
                        if stats["negative_rows_v1"] > 0 and stats["positive_rows_v1"] == 0
                        else "NO_ELIGIBLE_ROWS"
                    )
                )
            )
            for action, stats in {
                action: {
                    "positive_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(1).sum()),
                    "negative_rows_v1": int(part["coarse_teacher_binary_target_v1"].fillna(-1).eq(0).sum()),
                }
                for action, part in view_df.groupby(view_df["observed_action_v1"].astype("string"), dropna=False)
            }.items()
        },
        "hold_longer_pressure_v1": {
            "mean_extra_value_bps_v1": float(pd.to_numeric(view_df["hold_longer_extra_value_bps_v1"], errors="coerce").mean()),
            "median_extra_value_bps_v1": float(pd.to_numeric(view_df["hold_longer_extra_value_bps_v1"], errors="coerce").median()),
            "ten_bps_rate_v1": _safe_rate(
                float(view_df["hold_longer_pressure_10bps_v1"].fillna(False).sum()),
                float(len(view_df)),
            ),
            "twenty_five_bps_rate_v1": _safe_rate(
                float(view_df["hold_longer_pressure_25bps_v1"].fillna(False).sum()),
                float(len(view_df)),
            ),
        },
        "grid_scoreboard_v1": grid_score_df.to_dict(orient="records"),
        "recommended_grid_v1": {
            "grid_name_v1": recommended_grid_name,
            "grid_columns_v1": list(GRID_DEFS[recommended_grid_name]),
            "viable_row_share_v1": recommended_grid_row["viable_row_share_v1"],
            "teacher_row_share_v1": recommended_grid_row["teacher_row_share_v1"],
            "viable_cell_count_v1": int(recommended_grid_row["viable_cell_count_v1"]),
        },
        "top_exit_pressure_cells_v1": exit_pressure_cells[
            [
                "grid_value_v1",
                "row_count_v1",
                "exit_rows_v1",
                "premature_exit_exit_rows_v1",
                "mean_exit_hold_longer_extra_value_bps_v1",
                "exit_teacher_positive_rate_v1",
            ]
        ].to_dict(orient="records"),
        "top_hold_strength_cells_v1": hold_cells[
            [
                "grid_value_v1",
                "row_count_v1",
                "hold_rows_v1",
                "strong_capture_hold_rows_v1",
                "hold_teacher_positive_rate_v1",
                "mean_hold_realized_pnl_bps_v1",
            ]
        ].to_dict(orient="records"),
        "recommended_feature_bundle_v1": [
            "shadow_score_v1",
            "shadow_score_split_pct_rank_v1",
            "shadow_score_slice_pct_rank_v1",
            "shadow_score_coarse_band_v1",
            "overlay_session_axis_v1",
            "overlay_hold_age_axis_v1",
            "overlay_giveback_axis_v1",
            "as_of_management_core_minutes_held_at_anchor_v1",
            "as_of_management_core_giveback_ratio_from_peak_v1",
            "as_of_atr_bps_v1",
        ]
        + (["overlay_vol_axis_v1"] if "overlay_vol_axis_v1" in GRID_DEFS[recommended_grid_name] else []),
        "recommended_next_step_v1": (
            "RETRAIN_MANAGEMENT_FEEDBACK_MODEL_ON_COARSE_GRID_PLUS_RAW_SCORE"
            if float(recommended_grid_row["viable_row_share_v1"] or 0.0) >= 0.25
            and int(view_df["coarse_teacher_binary_target_eligible_v1"].sum()) >= 250
            else "KEEP_AS_RESEARCH_SURFACE_UNTIL_MORE_SIGNAL"
        ),
        "contract_note_v1": (
            "This is an observed-action coarse teacher surface built from exact management decision logging, exact outcome backfill, "
            "and hindsight quality overlays. It does not fabricate counterfactual actions or synthetic rewards."
        ),
    }
    return {
        "view_df": view_df,
        "cell_rollup_df": cell_rollup_df.sort_values(
            ["grid_name_v1", "viable_cell_v1", "row_count_v1", "grid_value_v1"],
            ascending=[True, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True),
        "summary": summary,
    }


def write_management_coarse_teacher_artifacts(
    reports_root: Path,
    *,
    extension_dir: Path | None = None,
    min_cell_rows: int = 20,
    min_rows_per_action: int = 10,
) -> Dict[str, str]:
    payload = build_management_coarse_teacher_payload(
        reports_root=reports_root,
        extension_dir=extension_dir,
        min_cell_rows=min_cell_rows,
        min_rows_per_action=min_rows_per_action,
    )
    reports_root = Path(reports_root).expanduser().resolve()
    view_path = reports_root / VIEW_FILE
    cell_rollup_path = reports_root / CELL_ROLLUP_FILE
    summary_path = reports_root / SUMMARY_FILE

    payload["view_df"].to_parquet(view_path, index=False)
    payload["cell_rollup_df"].to_csv(cell_rollup_path, index=False)
    _write_json(summary_path, payload["summary"])

    return {
        "view_path": str(view_path.resolve()),
        "cell_rollup_path": str(cell_rollup_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a coarse-grid plus raw-score management teacher surface from exact truth artifacts."
    )
    parser.add_argument("--reports-root", dest="reports_root", default=None)
    parser.add_argument("--extension-dir", dest="extension_dir", default=None)
    parser.add_argument("--min-cell-rows", dest="min_cell_rows", type=int, default=20)
    parser.add_argument("--min-rows-per-action", dest="min_rows_per_action", type=int, default=10)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    extension_dir = (
        Path(args.extension_dir).expanduser().resolve()
        if args.extension_dir
        else None
    )
    written = write_management_coarse_teacher_artifacts(
        reports_root=reports_root,
        extension_dir=extension_dir,
        min_cell_rows=max(1, int(args.min_cell_rows)),
        min_rows_per_action=max(1, int(args.min_rows_per_action)),
    )
    print(json.dumps(written, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
