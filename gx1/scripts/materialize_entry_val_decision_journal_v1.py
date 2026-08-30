"""Materialize a sealed VAL-only per-decision MAE/MFE journal.

This is a plumbing audit for an already-written prediction evidence parquet.
It never runs a model, trains, reads TEST, chooses a threshold, compounds
overlapping decisions, or claims a backtested strategy.  It joins every
existing VAL Entry-Q choice to the existing executable-label outcomes so that
later candidate evaluation has an honest, row-level journal surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PrefreezeTestSealLineageError,
    require_pretest_or_prefreeze_test_guard_lineage,
)


SCHEMA_VERSION = "gx1_entry_val_decision_journal_v1"
TEST_BOUNDARY_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
_REQUIRED_PREDICTION_COLUMNS = frozenset(
    {
        "split",
        "time",
        "trade_side",
        "side",
        "entry_action_q_bps",
        "entry_action_q_margin_bps",
        "selection_score_mode",
        "selection_score",
        "research_policy_gross_spread_inclusive_pnl_bps",
    }
)
_REQUIRED_LABEL_COLUMNS = frozenset(
    {
        "time",
        "mfe_long_first_n_bps",
        "mae_long_first_n_bps",
        "mfe_short_first_n_bps",
        "mae_short_first_n_bps",
        "y_long_final_pnl_at_direction_horizon_bps",
        "y_short_final_pnl_at_direction_horizon_bps",
        "y_long_expected_mae_bps",
        "y_short_expected_mae_bps",
    }
)


class ValDecisionJournalError(RuntimeError):
    """The pre-candidate VAL journal inputs were not an exact sealed pair."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or any(parent.is_symlink() for parent in candidate.parents)
        or not candidate.is_file()
    ):
        raise ValDecisionJournalError(f"[{label}_PATH_INVALID]")
    return candidate


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_absolute(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValDecisionJournalError(f"[{label}_JSON_INVALID]") from exc
    if not isinstance(value, dict):
        raise ValDecisionJournalError(f"[{label}_JSON_INVALID]")
    return value


def _atomic_write_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink() or not path.is_absolute() or not path.parent.is_dir():
        raise ValDecisionJournalError("[VAL_JOURNAL_OUTPUT_PATH_INVALID]")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        with Path(temporary).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.is_absolute() or not path.parent.is_dir():
        raise ValDecisionJournalError("[VAL_JOURNAL_REPORT_PATH_INVALID]")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _require_val_clock(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if "time" not in frame:
        raise ValDecisionJournalError(f"[{context}_TIME_MISSING]")
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    if (
        result.empty
        or bool(result["time"].isna().any())
        or bool((result["time"] >= TEST_BOUNDARY_UTC).any())
        or bool(result["time"].duplicated().any())
        or not bool(result["time"].is_monotonic_increasing)
    ):
        raise ValDecisionJournalError(f"[{context}_VAL_CLOCK_INVALID]")
    return result


def _select_side_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the evidence's existing side; never tune or replace the policy."""

    result = frame.copy()
    side = result["side"].astype(str)
    if not bool(side.isin(("LONG", "SHORT", "FLAT")).all()):
        raise ValDecisionJournalError("[VAL_JOURNAL_POLICY_SIDE_INVALID]")
    expected_trade_side = np.select(
        [side.to_numpy() == "LONG", side.to_numpy() == "SHORT"],
        [-1, 1],
        default=2,
    )
    trade_side = pd.to_numeric(result["trade_side"], errors="coerce").to_numpy()
    if not np.array_equal(trade_side, expected_trade_side):
        raise ValDecisionJournalError("[VAL_JOURNAL_POLICY_SIDE_ENCODING_INVALID]")
    long_mask = side.to_numpy() == "LONG"
    short_mask = side.to_numpy() == "SHORT"
    trade_mask = long_mask | short_mask
    result["actual_final_executable_pnl_bps"] = np.where(
        long_mask,
        result["y_long_final_pnl_at_direction_horizon_bps"],
        result["y_short_final_pnl_at_direction_horizon_bps"],
    )
    result["actual_mfe_bps"] = np.where(
        long_mask,
        result["mfe_long_first_n_bps"],
        result["mfe_short_first_n_bps"],
    )
    result["actual_mae_bps"] = np.where(
        long_mask,
        result["mae_long_first_n_bps"],
        result["mae_short_first_n_bps"],
    )
    result["label_expected_mae_bps"] = np.where(
        long_mask,
        result["y_long_expected_mae_bps"],
        result["y_short_expected_mae_bps"],
    )
    required_numeric = (
        "actual_final_executable_pnl_bps",
        "actual_mfe_bps",
        "actual_mae_bps",
        "label_expected_mae_bps",
        "research_policy_gross_spread_inclusive_pnl_bps",
    )
    for column in required_numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not bool(
            np.isfinite(
                result.loc[trade_mask, column].to_numpy(dtype=np.float64)
            ).all()
        ):
            raise ValDecisionJournalError(f"[VAL_JOURNAL_NONFINITE] {column}")
    # An MFE can be negative when the complete horizon stayed adverse; MAE is
    # stored as an adverse magnitude and therefore cannot be negative.
    if bool((result.loc[trade_mask, "actual_mae_bps"] < 0.0).any()):
        raise ValDecisionJournalError("[VAL_JOURNAL_EXCURSION_SIGN_INVALID]")
    if not bool(
        np.allclose(
            result.loc[trade_mask, "actual_final_executable_pnl_bps"].to_numpy(dtype=np.float64),
            result.loc[trade_mask, "research_policy_gross_spread_inclusive_pnl_bps"].to_numpy(dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        )
    ):
        raise ValDecisionJournalError("[VAL_JOURNAL_POLICY_PNL_LABEL_MISMATCH]")
    if not bool(
        np.allclose(
            result.loc[~trade_mask, "research_policy_gross_spread_inclusive_pnl_bps"].to_numpy(dtype=np.float64),
            0.0,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise ValDecisionJournalError("[VAL_JOURNAL_FLAT_POLICY_PNL_INVALID]")
    result.loc[~trade_mask, [
        "actual_final_executable_pnl_bps",
        "actual_mfe_bps",
        "actual_mae_bps",
        "label_expected_mae_bps",
    ]] = np.nan
    return result


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    trades = frame.loc[frame["side"].isin(("LONG", "SHORT"))].copy()
    if trades.empty:
        raise ValDecisionJournalError("[VAL_JOURNAL_NO_TRADE_DECISIONS]")
    pnl = trades["actual_final_executable_pnl_bps"].to_numpy(dtype=np.float64)
    mfe = trades["actual_mfe_bps"].to_numpy(dtype=np.float64)
    mae = trades["actual_mae_bps"].to_numpy(dtype=np.float64)
    result: dict[str, Any] = {
        "decision_rows": int(len(frame)),
        "trade_rows": int(len(trades)),
        "flat_rows": int((frame["side"] == "FLAT").sum()),
        "overlap_policy": "all_M5_decisions_overlap__no_compounding_or_equity_curve_claim",
        "mean_executable_pnl_bps": float(np.mean(pnl)),
        "median_executable_pnl_bps": float(np.median(pnl)),
        "win_rate": float(np.mean(pnl > 0.0)),
        "mean_mfe_bps": float(np.mean(mfe)),
        "median_mfe_bps": float(np.median(mfe)),
        "mean_mae_bps": float(np.mean(mae)),
        "median_mae_bps": float(np.median(mae)),
        "mfe_to_mae_ratio_of_means": (
            float(np.mean(mfe) / np.mean(mae)) if float(np.mean(mae)) > 0.0 else None
        ),
        "mae_before_mfe": {
            "status": "NOT_AVAILABLE_FROM_CURRENT_LABEL_SURFACE",
            "reason": "the sealed VAL labels retain extrema but not the ordered intrahorizon path needed to establish whether MAE occurred before MFE",
        },
        "by_side": {},
    }
    for side, group in trades.groupby("side", sort=True):
        values = group["actual_final_executable_pnl_bps"].to_numpy(dtype=np.float64)
        result["by_side"][str(side)] = {
            "rows": int(len(group)),
            "mean_executable_pnl_bps": float(np.mean(values)),
            "win_rate": float(np.mean(values > 0.0)),
            "mean_mfe_bps": float(group["actual_mfe_bps"].mean()),
            "mean_mae_bps": float(group["actual_mae_bps"].mean()),
        }
    return result


def run(
    *,
    val_parquet: Path,
    prediction_parquet: Path,
    prediction_report: Path,
    test_guard_json: Path,
    test_guard_sha256: str,
    dataset_run_id: str,
    out_journal: Path,
    out_report: Path,
) -> dict[str, Any]:
    val_parquet = _regular_absolute(val_parquet, label="VAL_JOURNAL_VAL")
    prediction_parquet = _regular_absolute(prediction_parquet, label="VAL_JOURNAL_PREDICTIONS")
    prediction_report = _regular_absolute(prediction_report, label="VAL_JOURNAL_PREDICTION_REPORT")
    test_guard_json = _regular_absolute(test_guard_json, label="VAL_JOURNAL_TEST_GUARD")
    if not isinstance(dataset_run_id, str) or not dataset_run_id:
        raise ValDecisionJournalError("[VAL_JOURNAL_DATASET_RUN_ID_INVALID]")
    try:
        guard = require_pretest_or_prefreeze_test_guard_lineage(
            test_guard_json,
            test_guard_sha256,
            expected_dataset_run_id=dataset_run_id,
            expected_dataset_dir=val_parquet.parent,
        )
    except (PrefreezeTestSealLineageError, OSError, ValueError) as exc:
        raise ValDecisionJournalError("[VAL_JOURNAL_TEST_GUARD_INVALID]") from exc
    if guard.get("test_accessed") is True or guard.get("access_proof", {}).get("test_dataset_bytes_read") is True:
        raise ValDecisionJournalError("[VAL_JOURNAL_TEST_GUARD_OPEN]")
    prior_report = _read_json(prediction_report, label="VAL_JOURNAL_PREDICTION_REPORT")
    if prior_report.get("test_accessed") is True:
        raise ValDecisionJournalError("[VAL_JOURNAL_PREDICTION_REPORT_TEST_ACCESS]")
    labels = pd.read_parquet(val_parquet, columns=sorted(_REQUIRED_LABEL_COLUMNS))
    predictions = pd.read_parquet(prediction_parquet)
    missing_labels = sorted(_REQUIRED_LABEL_COLUMNS - set(labels.columns))
    missing_predictions = sorted(_REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing_labels or missing_predictions:
        raise ValDecisionJournalError(
            f"[VAL_JOURNAL_COLUMNS_MISSING] labels={missing_labels} predictions={missing_predictions}"
        )
    labels = _require_val_clock(labels, context="VAL_JOURNAL_LABELS")
    predictions = _require_val_clock(predictions, context="VAL_JOURNAL_PREDICTIONS")
    if not bool((predictions["split"].astype(str) == "val").all()):
        raise ValDecisionJournalError("[VAL_JOURNAL_PREDICTION_SPLIT_INVALID]")
    # Prediction evidence retains a few labels for diagnostics.  Keep the
    # explicit sealed-VAL copy as the bare column name; a duplicate prediction
    # copy receives a suffix and is never used for the realised journal field.
    merged = predictions.merge(
        labels,
        on="time",
        how="inner",
        validate="one_to_one",
        suffixes=("_prediction", ""),
    )
    if len(merged) != len(labels) or len(merged) != len(predictions):
        raise ValDecisionJournalError("[VAL_JOURNAL_TIMESTAMP_JOIN_INCOMPLETE]")
    journal = _select_side_outcomes(merged)
    journal.insert(0, "journal_row", np.arange(len(journal), dtype=np.int64))
    journal.insert(1, "journal_schema_version", SCHEMA_VERSION)
    journal.insert(2, "authority", "VAL_ONLY_PLUMBING_NOT_EDGE_OR_BACKTEST")
    keep = [
        "journal_row", "journal_schema_version", "authority", "time", "side", "trade_side",
        "entry_action_q_bps", "entry_action_q_margin_bps", "selection_score_mode", "selection_score",
        "research_policy_gross_spread_inclusive_pnl_bps", "actual_final_executable_pnl_bps",
        "actual_mfe_bps", "actual_mae_bps", "label_expected_mae_bps",
    ]
    _atomic_write_parquet_new(out_journal, journal.loc[:, keep])
    report = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS_VAL_ONLY_PLUMBING_NOT_EDGE_OR_BACKTEST",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "authority": {
            "technical_preflight": True,
            "candidate": False,
            "test": False,
            "backtest": False,
            "paper": False,
            "live": False,
        },
        "inputs": {
            "val_parquet": {"path": str(val_parquet), "sha256": _sha256(val_parquet)},
            "prediction_parquet": {"path": str(prediction_parquet), "sha256": _sha256(prediction_parquet)},
            "prediction_report": {"path": str(prediction_report), "sha256": _sha256(prediction_report)},
            "test_guard": {"path": str(test_guard_json), "sha256": str(test_guard_sha256)},
        },
        "journal": {"path": str(out_journal), "sha256": _sha256(out_journal), "columns": keep},
        "summary": _summary(journal),
    }
    _atomic_write_json_new(out_report, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--prediction-parquet", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--test-guard-json", type=Path, required=True)
    parser.add_argument("--test-guard-sha256", required=True)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--out-journal", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(
        val_parquet=args.val_parquet,
        prediction_parquet=args.prediction_parquet,
        prediction_report=args.prediction_report,
        test_guard_json=args.test_guard_json,
        test_guard_sha256=str(args.test_guard_sha256),
        dataset_run_id=str(args.dataset_run_id),
        out_journal=args.out_journal,
        out_report=args.out_report,
    ), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
