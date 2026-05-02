from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_CANONICAL_TRUTH_V1"
CALENDAR_FILE = "TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json"

OUTPUT_FILES = {
    "trade_truth": "monday_r6_trade_truth_v1.parquet",
    "candidate_surface": "monday_r6_candidate_surface_v1.parquet",
    "xgb_signal_surface": "monday_r6_xgb_signal_surface_v1.parquet",
    "bar_feature_surface": "monday_r6_bar_feature_surface_v1.parquet",
    "exit_eval_trace": "monday_r6_exit_eval_trace_v1.parquet",
    "feature_manifest": "monday_r6_truth_feature_manifest_v1.csv",
    "coverage_summary": "monday_r6_truth_coverage_summary_v1.json",
    "quality_summary": "monday_r6_truth_quality_summary_v1.json",
    "exit_conflict_summary": "monday_r6_exit_conflict_summary_v1.json",
    "lineage": "monday_r6_truth_lineage_v1.csv",
    "run_inventory": "monday_r6_truth_run_inventory_v1.csv",
    "status": "monday_r6_canonical_truth_status_v1.json",
    "summary": "summary_v1.json",
    "manifest": "manifest_v1.json",
    "audit": "consistency_audit_v1.csv",
    "report": "report_v1.md",
}

CORE_OUTCOME_COLUMNS = [
    "run_id",
    "trade_uid",
    "trade_id",
    "candidate_uid",
    "entry_time",
    "exit_time",
    "open_ts_utc",
    "close_ts_utc",
    "pnl_bps",
    "mae_bps",
    "mfe_bps",
    "duration_bars",
    "side",
    "session",
    "exit_reason",
    "entry_spread_bps",
    "exit_spread_bps",
    "post_exit_mfe_bps",
    "early_exit_regret",
    "post_exit_mfe_bps_replay_end_obs",
    "early_exit_regret_replay_end_obs",
]

REJECT_PATTERNS = [
    "EXIT_REPLAY_DETERMINISTIC_ARBITER_REJECT_BYPASS",
    "LOSS_CLOSE_NOT_ALLOWED",
    "MODEL_EXIT_MIN_REALIZED_PNL_BPS",
    "[ARB] reject model exit",
    "EXIT_MODEL_DECIDED_EXIT",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, pd.NaT.__class__):
        return None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_utc_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("int64")


def _coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in columns:
        if column in frame.columns:
            out = out.where(out.notna(), frame[column])
    return out


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def _safe_numeric(series: pd.Series | Any) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.Series(dtype="float64")


def _load_calendar(reports_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    calendar = _read_json(reports_root / CALENDAR_FILE)
    by_run: dict[str, dict[str, Any]] = {}
    for row in calendar.get("full_monday_weeks", []) or []:
        if isinstance(row, dict) and row.get("run_id"):
            by_run[str(row["run_id"])] = row
    return calendar, by_run


def _run_dirs(reports_root: Path, calendar_by_run: dict[str, dict[str, Any]], include_quarantine: bool) -> list[Path]:
    dirs = sorted(path for path in reports_root.glob("TRUTH_MONFRI_WEEK_*") if path.is_dir())
    selected: list[Path] = []
    for run_dir in dirs:
        meta = calendar_by_run.get(run_dir.name, {})
        quarantine_status = str(meta.get("quarantine_status") or "NOT_IN_CALENDAR")
        if not include_quarantine and quarantine_status != "ACTIVE_CANDIDATE":
            continue
        if (run_dir / f"trade_outcomes_{run_dir.name}_MERGED.parquet").exists():
            selected.append(run_dir)
    return selected


def _read_parquet(path: Path, run_id: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "run_id" not in df.columns:
        df.insert(0, "run_id", run_id)
    else:
        df["run_id"] = df["run_id"].fillna(run_id).astype("string")
    return df


def _read_csv(path: Path, run_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "run_id" not in df.columns:
        df.insert(0, "run_id", run_id)
    else:
        df["run_id"] = df["run_id"].fillna(run_id).astype("string")
    return df


def _read_optional_parquets(run_dirs: list[Path], pattern: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / pattern.format(run_id=run_dir.name)
        if path.exists():
            frames.append(_read_parquet(path, run_dir.name))
    return frames


def _calendar_cols(frame: pd.DataFrame, calendar_by_run: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if frame.empty or "run_id" not in frame.columns:
        return frame
    out = frame.copy()
    meta = out["run_id"].astype("string").map(calendar_by_run)
    out["calendar_quarantine_status_v1"] = meta.map(
        lambda row: (row or {}).get("quarantine_status", "NOT_IN_CALENDAR")
    )
    out["calendar_quarantine_reason_v1"] = meta.map(lambda row: (row or {}).get("quarantine_reason"))
    out["calendar_start_utc_v1"] = meta.map(lambda row: (row or {}).get("calendar_start_utc"))
    out["calendar_end_exclusive_utc_v1"] = meta.map(lambda row: (row or {}).get("calendar_end_exclusive_utc"))
    out["friday_flat_cutoff_utc_v1"] = meta.map(lambda row: (row or {}).get("friday_flat_cutoff_utc"))
    return out


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_surfaces(
    run_dirs: list[Path],
    calendar_by_run: dict[str, dict[str, Any]],
    *,
    write_bar_surface: bool,
) -> dict[str, pd.DataFrame]:
    outcome_frames = _read_optional_parquets(run_dirs, "trade_outcomes_{run_id}_MERGED.parquet")
    journal_frames = _read_optional_parquets(run_dirs, "trade_journal_{run_id}_MERGED.parquet")
    candidate_frames = _read_optional_parquets(run_dirs, "shadow_meta_candidates_{run_id}_MERGED.parquet")
    xgb_frames = _read_optional_parquets(run_dirs, "xgb_multi_horizon_predictions_{run_id}.parquet")

    exit_frames: list[pd.DataFrame] = []
    bar_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        trace_path = run_dir / "replay" / "chunk_0" / "EXIT_EVAL_TRACE.csv"
        if trace_path.exists():
            exit_frames.append(_read_csv(trace_path, run_dir.name))
        bar_path = run_dir / "replay" / "chunk_0" / "chunk_0_data.parquet"
        if write_bar_surface and bar_path.exists():
            bar_frames.append(_read_parquet(bar_path, run_dir.name))

    surfaces = {
        "outcomes": _calendar_cols(_concat(outcome_frames), calendar_by_run),
        "journal": _calendar_cols(_concat(journal_frames), calendar_by_run),
        "candidates": _calendar_cols(_concat(candidate_frames), calendar_by_run),
        "xgb": _calendar_cols(_concat(xgb_frames), calendar_by_run),
        "exit_trace": _calendar_cols(_concat(exit_frames), calendar_by_run),
        "bar": _calendar_cols(_concat(bar_frames), calendar_by_run) if write_bar_surface else pd.DataFrame(),
    }
    return surfaces


def _accepted_candidate_map(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    accepted = candidates[_bool_series(candidates, "accepted")].copy()
    if accepted.empty:
        return accepted
    keep = [
        "run_id",
        "trade_id",
        "trade_uid",
        "candidate_uid",
        "decision_ts_utc",
        "side",
        "session",
        "weekday_utc",
        "hour_utc",
        "atr_bps",
        "entry_spread_bps",
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "margin",
        "uncertainty_score",
        "tradable_prob",
        "mfe_first_n_pred",
        "path_quality_pred",
        "vol_regime",
        "trend_regime",
        "decision",
        "decision_reason",
        "policy_hash",
        "entry_bundle_sha256",
        "exit_bundle_sha256",
        "positive_exit",
        "cata",
        "never_mfe",
        "good_mfe_then_rot",
        "good_trade_mfe20_mae5_v1",
        "mfe_mae_ratio_v1",
    ]
    keep = [column for column in keep if column in accepted.columns]
    accepted = accepted[keep].copy()
    accepted = accepted.sort_values(["run_id", "trade_id", "decision_ts_utc"], na_position="last")
    accepted = accepted.drop_duplicates(["run_id", "trade_id"], keep="last")
    rename = {
        column: f"entry_candidate_{column}_v1"
        for column in accepted.columns
        if column not in {"run_id", "trade_id", "trade_uid", "candidate_uid"}
    }
    return accepted.rename(columns=rename)


def _journal_map(journal: pd.DataFrame) -> pd.DataFrame:
    if journal.empty:
        return pd.DataFrame()
    work = journal.copy()
    work = work.sort_values(["run_id", "trade_id", "open_ts_utc"], na_position="last")
    work = work.drop_duplicates(["run_id", "trade_id"], keep="last")
    rename = {column: f"journal_{column}_v1" for column in work.columns if column not in {"run_id", "trade_id", "trade_uid"}}
    return work.rename(columns=rename)


def _xgb_entry_map(xgb: pd.DataFrame) -> pd.DataFrame:
    if xgb.empty:
        return pd.DataFrame()
    work = xgb.copy()
    if "ts" not in work.columns:
        return pd.DataFrame()
    work["_ts_ns"] = _to_utc_ns(work["ts"])
    work = work.sort_values(["run_id", "_ts_ns", "head", "horizon_bars"], na_position="last")
    work = work.drop_duplicates(["run_id", "_ts_ns"], keep="last")
    keep = ["run_id", "_ts_ns", "p_long", "p_short", "p_flat", "p_hat", "pred_side", "has_ctx", "head", "horizon_bars"]
    keep = [column for column in keep if column in work.columns]
    rename = {column: f"entry_xgb_{column}_v1" for column in keep if column not in {"run_id", "_ts_ns"}}
    return work[keep].rename(columns=rename)


def _build_trade_truth(surfaces: dict[str, pd.DataFrame]) -> pd.DataFrame:
    outcomes = surfaces["outcomes"].copy()
    if outcomes.empty:
        return outcomes
    for column in CORE_OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = pd.NA
    outcomes = outcomes.copy()

    accepted = _accepted_candidate_map(surfaces["candidates"])
    if not accepted.empty:
        outcomes = outcomes.merge(
            accepted,
            on=["run_id", "trade_id"],
            how="left",
            suffixes=("", "_accepted_candidate_drop"),
        )
        if "trade_uid_accepted_candidate_drop" in outcomes.columns:
            outcomes["trade_uid"] = outcomes["trade_uid"].where(outcomes["trade_uid"].notna(), outcomes["trade_uid_accepted_candidate_drop"])
            outcomes = outcomes.drop(columns=["trade_uid_accepted_candidate_drop"])
        if "candidate_uid_accepted_candidate_drop" in outcomes.columns:
            outcomes["candidate_uid"] = outcomes["candidate_uid"].where(outcomes["candidate_uid"].notna(), outcomes["candidate_uid_accepted_candidate_drop"])
            outcomes = outcomes.drop(columns=["candidate_uid_accepted_candidate_drop"])

    journal = _journal_map(surfaces["journal"])
    if not journal.empty:
        outcomes = outcomes.merge(journal, on=["run_id", "trade_id"], how="left", suffixes=("", "_journal_drop"))
        if "trade_uid_journal_drop" in outcomes.columns:
            outcomes["trade_uid"] = outcomes["trade_uid"].where(outcomes["trade_uid"].notna(), outcomes["trade_uid_journal_drop"])
            outcomes = outcomes.drop(columns=["trade_uid_journal_drop"])

    outcomes["canonical_entry_ts_utc_v1"] = _coalesce(
        outcomes,
        ["journal_open_ts_utc_v1", "open_ts_utc", "entry_time", "entry_candidate_decision_ts_utc_v1"],
    )
    outcomes["canonical_exit_ts_utc_v1"] = _coalesce(outcomes, ["journal_close_ts_utc_v1", "close_ts_utc", "exit_time"])
    outcomes["decision_timestamp_v1"] = _coalesce(outcomes, ["entry_candidate_decision_ts_utc_v1", "canonical_entry_ts_utc_v1"])
    outcomes["canonical_pnl_bps_v1"] = pd.to_numeric(_coalesce(outcomes, ["pnl_bps", "journal_pnl_bps_v1"]), errors="coerce")
    outcomes["canonical_mfe_bps_v1"] = pd.to_numeric(_coalesce(outcomes, ["mfe_bps", "journal_mfe_bps_v1"]), errors="coerce")
    outcomes["canonical_mae_bps_v1"] = pd.to_numeric(_coalesce(outcomes, ["mae_bps", "journal_mae_bps_v1"]), errors="coerce")
    outcomes["canonical_duration_bars_v1"] = pd.to_numeric(_coalesce(outcomes, ["duration_bars", "journal_bars_in_trade_v1"]), errors="coerce")
    outcomes["canonical_exit_reason_v1"] = _coalesce(outcomes, ["exit_reason", "journal_exit_reason_v1"]).astype("string")
    outcomes["canonical_side_v1"] = _coalesce(outcomes, ["side", "journal_side_v1", "entry_candidate_side_v1"]).astype("string")
    outcomes["canonical_session_v1"] = _coalesce(outcomes, ["session", "journal_session_v1", "entry_candidate_session_v1"]).astype("string")

    xgb = _xgb_entry_map(surfaces["xgb"])
    outcomes["_entry_ts_ns"] = _to_utc_ns(outcomes["canonical_entry_ts_utc_v1"])
    if not xgb.empty:
        outcomes = outcomes.merge(xgb, left_on=["run_id", "_entry_ts_ns"], right_on=["run_id", "_ts_ns"], how="left")
        outcomes = outcomes.drop(columns=["_ts_ns"], errors="ignore")
    outcomes["entry_xgb_exact_available_v1"] = outcomes.get("entry_xgb_p_hat_v1", pd.Series(pd.NA, index=outcomes.index)).notna()

    pnl = outcomes["canonical_pnl_bps_v1"]
    mfe = outcomes["canonical_mfe_bps_v1"]
    mae = outcomes["canonical_mae_bps_v1"]
    outcomes["truth_positive_exit_v1"] = pnl.gt(0)
    outcomes["truth_loss_v1"] = pnl.lt(0)
    outcomes["truth_runner_50_mfe_v1"] = mfe.ge(50)
    outcomes["truth_runner_100_mfe_v1"] = mfe.ge(100)
    outcomes["truth_runner_200_mfe_v1"] = mfe.ge(200)
    outcomes["truth_mae_50_or_worse_v1"] = mae.le(-50)
    outcomes["truth_mae_100_or_worse_v1"] = mae.le(-100)
    outcomes["truth_good_trade_mfe20_mae5_v1"] = mfe.ge(20) & mae.ge(-5)
    outcomes["truth_strongest_winner_v1"] = mfe.ge(200) & pnl.gt(0)
    outcomes["truth_bad_loss_with_low_mfe_v1"] = pnl.lt(0) & mfe.lt(20)
    outcomes["truth_cata_or_friday_flat_damage_v1"] = outcomes["canonical_exit_reason_v1"].isin(
        ["CATASTROPHIC_GUARD", "POLICY_FRIDAY_FLAT"]
    ) & pnl.lt(0)
    outcomes["truth_exit_too_early_regret_primary_v1"] = _bool_series(outcomes, "early_exit_regret")
    outcomes["truth_exit_too_early_regret_replay_end_v1"] = _bool_series(outcomes, "early_exit_regret_replay_end_obs")
    outcomes["truth_post_exit_mfe_bps_v1"] = pd.to_numeric(
        _coalesce(outcomes, ["post_exit_mfe_bps", "journal_post_exit_mfe_bps_v1"]), errors="coerce"
    )
    outcomes["truth_capture_ratio_v1"] = np.where(mfe.gt(0), pnl / mfe.replace(0, np.nan), np.nan)
    outcomes["monday_r6_truth_contract_v1"] = "MONDAY_R6_CANONICAL_TRUTH_V1|REALIZED_REPLAY_TRUTH|NOT_LIVE_GATE|NOT_WEDNESDAY_COPY"
    return outcomes.drop(columns=["_entry_ts_ns"], errors="ignore")


def _bar_coverage(trade_truth: pd.DataFrame, bar: pd.DataFrame) -> dict[str, Any]:
    if trade_truth.empty or bar.empty or "time" not in bar.columns:
        return {
            "entry_bar_exact_coverage_v1": 0,
            "exit_bar_exact_coverage_v1": 0,
            "entry_bar_exact_rate_v1": None,
            "exit_bar_exact_rate_v1": None,
        }
    bar_work = bar[["run_id", "time"]].copy()
    bar_work["_bar_ts_ns"] = _to_utc_ns(bar_work["time"])
    by_run = {
        run_id: set(group["_bar_ts_ns"].dropna().astype("int64"))
        for run_id, group in bar_work.groupby(bar_work["run_id"].astype("string"), dropna=False)
    }
    entry_ns = _to_utc_ns(trade_truth["canonical_entry_ts_utc_v1"])
    exit_ns = _to_utc_ns(trade_truth["canonical_exit_ts_utc_v1"])
    entry_hit = []
    exit_hit = []
    for pos, (_, row) in enumerate(trade_truth.iterrows()):
        run_set = by_run.get(str(row["run_id"]), set())
        entry_hit.append(int(entry_ns.iloc[pos]) in run_set)
        exit_hit.append(int(exit_ns.iloc[pos]) in run_set)
    den = len(trade_truth)
    return {
        "entry_bar_exact_coverage_v1": int(sum(entry_hit)),
        "exit_bar_exact_coverage_v1": int(sum(exit_hit)),
        "entry_bar_exact_rate_v1": float(sum(entry_hit) / den) if den else None,
        "exit_bar_exact_rate_v1": float(sum(exit_hit) / den) if den else None,
    }


def _feature_manifest(surfaces: dict[str, pd.DataFrame], trade_truth: pd.DataFrame, run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_surface(surface: str, frame: pd.DataFrame, role: str) -> None:
        for column in frame.columns:
            null_rate = float(frame[column].isna().mean()) if len(frame) else None
            rows.append(
                {
                    "surface_v1": surface,
                    "feature_name_v1": str(column),
                    "dtype_v1": str(frame[column].dtype),
                    "row_count_v1": int(len(frame)),
                    "null_rate_v1": null_rate,
                    "role_v1": role,
                }
            )

    add_surface("trade_truth", trade_truth, "canonical realized trade truth and derived truth pockets")
    add_surface("candidate_surface", surfaces["candidates"], "entry decision/candidate surface")
    add_surface("xgb_signal_surface", surfaces["xgb"], "xgb multi-horizon prediction signal surface")
    add_surface("exit_eval_trace", surfaces["exit_trace"], "bar-level exit model/eval trace")
    if not surfaces["bar"].empty:
        add_surface("bar_feature_surface", surfaces["bar"], "raw joined M1/M5/prebuilt bar feature surface")

    seen_exit_features: set[str] = set()
    for run_dir in run_dirs:
        proof_path = run_dir / "replay" / "chunk_0" / "EXIT_FEATURE_VECTOR_PROOF.json"
        proof = _read_json(proof_path)
        for item in proof.get("feature_list", []) or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            if name in seen_exit_features:
                continue
            seen_exit_features.add(name)
            rows.append(
                {
                    "surface_v1": "exit_transformer_runtime_input",
                    "feature_name_v1": name,
                    "dtype_v1": "runtime_tensor_float_or_encoded_cat",
                    "row_count_v1": None,
                    "null_rate_v1": None,
                    "role_v1": f"exit transformer input group={item.get('group')}",
                }
            )
    return pd.DataFrame(rows)


def _quality_summary(trade_truth: pd.DataFrame) -> dict[str, Any]:
    if trade_truth.empty:
        return {}
    pnl = pd.to_numeric(trade_truth["canonical_pnl_bps_v1"], errors="coerce")
    mfe = pd.to_numeric(trade_truth["canonical_mfe_bps_v1"], errors="coerce")
    mae = pd.to_numeric(trade_truth["canonical_mae_bps_v1"], errors="coerce")
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross_win = float(winners.sum()) if len(winners) else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) else 0.0
    by_exit = trade_truth["canonical_exit_reason_v1"].astype("string").value_counts(dropna=False).to_dict()
    by_session = trade_truth["canonical_session_v1"].astype("string").value_counts(dropna=False).to_dict()
    return {
        "trade_rows_v1": int(len(trade_truth)),
        "total_pnl_bps_v1": float(pnl.sum()),
        "mean_pnl_bps_v1": float(pnl.mean()),
        "median_pnl_bps_v1": float(pnl.median()),
        "win_rate_v1": float(pnl.gt(0).mean()),
        "loss_count_v1": int(pnl.lt(0).sum()),
        "profit_factor_v1": float(gross_win / gross_loss) if gross_loss else None,
        "mfe_50_count_v1": int(mfe.ge(50).sum()),
        "mfe_100_count_v1": int(mfe.ge(100).sum()),
        "mfe_200_count_v1": int(mfe.ge(200).sum()),
        "mae_50_or_worse_count_v1": int(mae.le(-50).sum()),
        "mae_100_or_worse_count_v1": int(mae.le(-100).sum()),
        "primary_exit_regret_count_v1": int(_bool_series(trade_truth, "truth_exit_too_early_regret_primary_v1").sum()),
        "replay_end_exit_regret_count_v1": int(_bool_series(trade_truth, "truth_exit_too_early_regret_replay_end_v1").sum()),
        "exit_reason_counts_v1": {str(k): int(v) for k, v in by_exit.items()},
        "session_counts_v1": {str(k): int(v) for k, v in by_session.items()},
        "worst_trades_v1": trade_truth.sort_values("canonical_pnl_bps_v1", ascending=True)
        .head(10)[["run_id", "trade_id", "candidate_uid", "canonical_entry_ts_utc_v1", "canonical_exit_ts_utc_v1", "canonical_pnl_bps_v1", "canonical_mfe_bps_v1", "canonical_mae_bps_v1", "canonical_exit_reason_v1"]]
        .to_dict("records"),
    }


def _scan_log_conflicts(run_dirs: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_run: dict[str, Counter[str]] = {}
    for run_dir in run_dirs:
        run_counter: Counter[str] = Counter()
        for log_path in sorted((run_dir / "replay" / "chunk_0" / "logs").glob("*.log")):
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in REJECT_PATTERNS:
                value = text.count(pattern)
                if value:
                    counts[pattern] += value
                    run_counter[pattern] += value
        if run_counter:
            by_run[run_dir.name] = run_counter
    return {
        "log_pattern_counts_v1": {key: int(value) for key, value in counts.items()},
        "log_pattern_counts_by_run_v1": {
            run_id: {key: int(value) for key, value in counter.items()} for run_id, counter in by_run.items()
        },
    }


def _exit_conflict_summary(exit_trace: pd.DataFrame, run_dirs: list[Path]) -> dict[str, Any]:
    if exit_trace.empty:
        return {"exit_trace_rows_v1": 0, **_scan_log_conflicts(run_dirs)}
    trace = exit_trace.copy()
    exit_decision = trace.get("exit_decision", pd.Series("", index=trace.index)).astype("string")
    pnl = pd.to_numeric(trace.get("pnl_bps", pd.Series(np.nan, index=trace.index)), errors="coerce")
    exit_prob = pd.to_numeric(trace.get("exit_prob", pd.Series(np.nan, index=trace.index)), errors="coerce")
    threshold = pd.to_numeric(trace.get("exit_threshold", pd.Series(np.nan, index=trace.index)), errors="coerce")
    model_eval = pd.to_numeric(trace.get("exit_model_evaluated", pd.Series(0, index=trace.index)), errors="coerce").fillna(0).gt(0)
    model_would_exit = model_eval & exit_prob.ge(threshold)
    subfloor = model_would_exit & pnl.lt(1.0)
    hard_decisions = exit_decision.value_counts(dropna=False).to_dict()
    out = {
        "exit_trace_rows_v1": int(len(trace)),
        "model_evaluated_rows_v1": int(model_eval.sum()),
        "model_would_exit_rows_v1": int(model_would_exit.sum()),
        "model_would_exit_but_subfloor_rows_v1": int(subfloor.sum()),
        "exit_decision_counts_v1": {str(k): int(v) for k, v in hard_decisions.items()},
        "threshold_decision_negative_or_subfloor_count_v1": int((exit_decision.eq("threshold") & pnl.lt(1.0)).sum()),
        "guard_exit_count_v1": int(exit_decision.eq("guard").sum()),
        "policy_friday_flat_count_v1": int(exit_decision.eq("policy_friday_flat").sum()),
        "post_edge_separator_count_v1": int(exit_decision.eq("post_edge_separator").sum()),
    }
    out.update(_scan_log_conflicts(run_dirs))
    return out


def _run_inventory(run_dirs: list[Path], calendar_by_run: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        meta = calendar_by_run.get(run_id, {})
        def count_parquet(path: Path) -> int | None:
            if not path.exists():
                return None
            try:
                import pyarrow.parquet as pq

                return int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                return int(pd.read_parquet(path).shape[0])

        outcome_path = run_dir / f"trade_outcomes_{run_id}_MERGED.parquet"
        candidate_path = run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet"
        journal_path = run_dir / f"trade_journal_{run_id}_MERGED.parquet"
        xgb_path = run_dir / f"xgb_multi_horizon_predictions_{run_id}.parquet"
        trace_path = run_dir / "replay" / "chunk_0" / "EXIT_EVAL_TRACE.csv"
        bar_path = run_dir / "replay" / "chunk_0" / "chunk_0_data.parquet"
        rows.append(
            {
                "run_id": run_id,
                "calendar_quarantine_status_v1": meta.get("quarantine_status", "NOT_IN_CALENDAR"),
                "calendar_quarantine_reason_v1": meta.get("quarantine_reason"),
                "calendar_start_utc_v1": meta.get("calendar_start_utc"),
                "calendar_end_exclusive_utc_v1": meta.get("calendar_end_exclusive_utc"),
                "friday_flat_cutoff_utc_v1": meta.get("friday_flat_cutoff_utc"),
                "run_completed_json_exists_v1": (run_dir / "RUN_COMPLETED.json").exists(),
                "postrun_e2e_exists_v1": (run_dir / "POSTRUN_E2E.json").exists(),
                "outcome_rows_v1": count_parquet(outcome_path),
                "candidate_rows_v1": count_parquet(candidate_path),
                "journal_rows_v1": count_parquet(journal_path),
                "xgb_rows_v1": count_parquet(xgb_path),
                "exit_trace_rows_v1": int(sum(1 for _ in trace_path.open(encoding="utf-8"))) - 1 if trace_path.exists() else None,
                "bar_rows_v1": count_parquet(bar_path),
                "outcome_path_v1": str(outcome_path),
                "candidate_path_v1": str(candidate_path),
                "journal_path_v1": str(journal_path),
                "xgb_path_v1": str(xgb_path),
                "exit_trace_path_v1": str(trace_path),
                "bar_path_v1": str(bar_path),
            }
        )
    return pd.DataFrame(rows)


def _coverage_summary(
    run_inventory: pd.DataFrame,
    surfaces: dict[str, pd.DataFrame],
    trade_truth: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    calendar: dict[str, Any],
) -> dict[str, Any]:
    total_runs = int(len(run_inventory))
    active_runs = int(run_inventory["calendar_quarantine_status_v1"].eq("ACTIVE_CANDIDATE").sum()) if total_runs else 0
    quarantine_runs = int(run_inventory["calendar_quarantine_status_v1"].ne("ACTIVE_CANDIDATE").sum()) if total_runs else 0
    zero_trade_runs = int(pd.to_numeric(run_inventory["outcome_rows_v1"], errors="coerce").fillna(0).eq(0).sum()) if total_runs else 0
    accepted_candidates = int(_bool_series(surfaces["candidates"], "accepted").sum()) if not surfaces["candidates"].empty else 0
    bar_cov = _bar_coverage(trade_truth, surfaces["bar"])
    return {
        "calendar_artifact_v1": CALENDAR_FILE if calendar else None,
        "calendar_full_monday_week_count_v1": calendar.get("full_monday_week_count"),
        "included_run_count_v1": total_runs,
        "active_candidate_run_count_v1": active_runs,
        "quarantine_marked_run_count_v1": quarantine_runs,
        "zero_trade_run_count_v1": zero_trade_runs,
        "trade_truth_rows_v1": int(len(trade_truth)),
        "candidate_surface_rows_v1": int(len(surfaces["candidates"])),
        "accepted_candidate_rows_v1": accepted_candidates,
        "xgb_signal_rows_v1": int(len(surfaces["xgb"])),
        "journal_rows_v1": int(len(surfaces["journal"])),
        "exit_eval_trace_rows_v1": int(len(surfaces["exit_trace"])),
        "bar_feature_rows_v1": int(len(surfaces["bar"])),
        "feature_manifest_rows_v1": int(len(feature_manifest)),
        "candidate_uid_coverage_v1": int(trade_truth["candidate_uid"].notna().sum()) if "candidate_uid" in trade_truth.columns else 0,
        "candidate_uid_coverage_rate_v1": float(trade_truth["candidate_uid"].notna().mean()) if len(trade_truth) and "candidate_uid" in trade_truth.columns else None,
        "entry_xgb_exact_coverage_v1": int(_bool_series(trade_truth, "entry_xgb_exact_available_v1").sum()),
        "entry_xgb_exact_rate_v1": float(_bool_series(trade_truth, "entry_xgb_exact_available_v1").mean()) if len(trade_truth) else None,
        **bar_cov,
    }


def _lineage(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roles = {
        "trade_outcomes_{run_id}_MERGED.parquet": "realized trade outcome truth",
        "trade_journal_{run_id}_MERGED.parquet": "journal/exit-state trade truth",
        "shadow_meta_candidates_{run_id}_MERGED.parquet": "entry candidate decision surface",
        "xgb_multi_horizon_predictions_{run_id}.parquet": "xgb prediction feature surface",
        "replay/chunk_0/EXIT_EVAL_TRACE.csv": "bar-level exit evaluation trace",
        "replay/chunk_0/chunk_0_data.parquet": "raw joined bar feature surface",
        "replay/chunk_0/EXIT_FEATURE_VECTOR_PROOF.json": "exit transformer runtime feature contract",
        "replay/chunk_0/MODEL_USED_CAPSULE.json": "entry/xgb/guard model capsule",
        "replay/chunk_0/EXIT_RUNTIME_SOURCE_OF_TRUTH.json": "exit model source-of-truth contract",
    }
    for run_dir in run_dirs:
        for template, role in roles.items():
            rel = template.format(run_id=run_dir.name)
            path = run_dir / rel
            rows.append(
                {
                    "run_id": run_dir.name,
                    "source_role_v1": role,
                    "relative_path_v1": rel,
                    "absolute_path_v1": str(path),
                    "exists_v1": path.exists(),
                    "byte_size_v1": int(path.stat().st_size) if path.exists() else None,
                    "sha256_v1": _sha256(path) if path.exists() and path.stat().st_size < 25_000_000 else None,
                }
            )
    return pd.DataFrame(rows)


def _audit(
    coverage: dict[str, Any],
    trade_truth: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    include_quarantine: bool,
) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("ONE_MONDAY_R6_TRUTH_BUILT", "PASS" if coverage["trade_truth_rows_v1"] > 0 else "FAIL", coverage["trade_truth_rows_v1"]),
            row("USES_ACTUAL_MONDAY_REPLAY_RUNS", "PASS", {"included_run_count": coverage["included_run_count_v1"], "include_quarantine": include_quarantine}),
            row("NOT_1689_EXACT_ONLY", "PASS" if coverage["trade_truth_rows_v1"] != 1689 else "FAIL", coverage["trade_truth_rows_v1"]),
            row("NOT_OLD_1852_ONLY", "PASS" if coverage["trade_truth_rows_v1"] != 1852 else "FAIL", coverage["trade_truth_rows_v1"]),
            row("CANDIDATE_SURFACE_MATERIALIZED", "PASS" if coverage["candidate_surface_rows_v1"] > 0 else "FAIL", coverage["candidate_surface_rows_v1"]),
            row("BAR_FEATURE_SURFACE_MATERIALIZED", "PASS" if coverage["bar_feature_rows_v1"] > 0 else "WARN", coverage["bar_feature_rows_v1"]),
            row("EXIT_TRACE_MATERIALIZED", "PASS" if coverage["exit_eval_trace_rows_v1"] > 0 else "WARN", coverage["exit_eval_trace_rows_v1"]),
            row("FEATURE_MANIFEST_MATERIALIZED", "PASS" if len(feature_manifest) > 0 else "FAIL", len(feature_manifest)),
            row("CANDIDATE_UID_COVERAGE", "PASS" if coverage["candidate_uid_coverage_rate_v1"] == 1.0 else "WARN", coverage["candidate_uid_coverage_rate_v1"]),
            row("ENTRY_XGB_EXACT_COVERAGE", "PASS" if (coverage["entry_xgb_exact_rate_v1"] or 0.0) > 0.5 else "WARN", coverage["entry_xgb_exact_rate_v1"]),
        ]
    )


def _status(audit_df: pd.DataFrame, coverage: dict[str, Any]) -> dict[str, Any]:
    failed = int(audit_df["status_v1"].eq("FAIL").sum())
    warn = int(audit_df["status_v1"].eq("WARN").sum())
    if failed:
        status = "MONDAY_R6_CANONICAL_TRUTH_BUILD_FAILED"
    elif warn:
        status = "MONDAY_R6_CANONICAL_TRUTH_BUILT_WITH_WARNINGS"
    else:
        status = "MONDAY_R6_CANONICAL_TRUTH_BUILT"
    return {
        "layer_name": "MONDAY_R6_CANONICAL_TRUTH_STATUS_V1",
        "status_v1": status,
        "failed_check_count_v1": failed,
        "warning_check_count_v1": warn,
        "not_live_gate_v1": True,
        "not_policy_controller_v1": True,
        "training_started_v1": False,
        "truth_rows_v1": coverage["trade_truth_rows_v1"],
        "blocked_uses_v1": [
            "DO_NOT_TREAT_AS_LIVE_PROMOTION",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_USE_OLD_1852_MONDAY_R6_AS_CANONICAL_TRUTH",
        ],
        "next_action_v1": "REVIEW_MONDAY_R6_CANONICAL_TRUTH_THEN_BUILD_R6_TRAINING_FROM_THIS_TRUTH",
    }


def _report(summary: dict[str, Any], quality: dict[str, Any], coverage: dict[str, Any], status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Canonical Truth V1",
            "",
            f"Materialized at: `{summary['materialized_at_utc_v1']}`",
            "",
            "## Status",
            "",
            f"- Status: `{status['status_v1']}`",
            f"- Trade truth rows: `{coverage['trade_truth_rows_v1']}`",
            f"- Included runs: `{coverage['included_run_count_v1']}`",
            f"- Active/quarantine-marked runs: `{coverage['active_candidate_run_count_v1']}` / `{coverage['quarantine_marked_run_count_v1']}`",
            f"- Candidate surface rows: `{coverage['candidate_surface_rows_v1']}`",
            f"- Bar feature rows: `{coverage['bar_feature_rows_v1']}`",
            f"- Exit trace rows: `{coverage['exit_eval_trace_rows_v1']}`",
            "",
            "## Quality",
            "",
            f"- Total PnL bps: `{quality.get('total_pnl_bps_v1')}`",
            f"- Win rate: `{quality.get('win_rate_v1')}`",
            f"- Profit factor: `{quality.get('profit_factor_v1')}`",
            f"- 50+/100+/200+ MFE counts: `{quality.get('mfe_50_count_v1')}` / `{quality.get('mfe_100_count_v1')}` / `{quality.get('mfe_200_count_v1')}`",
            f"- MAE 50+/100+ adverse counts: `{quality.get('mae_50_or_worse_count_v1')}` / `{quality.get('mae_100_or_worse_count_v1')}`",
            "",
            "## Contract",
            "",
            "This is a Monday replay truth materialization over the actual available Monday run universe. It is not the 1689 exact-only surface, not the old 1852 Monday R6, and not a Wednesday copy.",
            "",
        ]
    )


def materialize(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    *,
    include_quarantine: bool = True,
    write_bar_surface: bool = True,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=False)

    calendar, calendar_by_run = _load_calendar(reports_root)
    run_dirs = _run_dirs(reports_root, calendar_by_run, include_quarantine=include_quarantine)
    run_inventory = _run_inventory(run_dirs, calendar_by_run)
    surfaces = _load_surfaces(run_dirs, calendar_by_run, write_bar_surface=write_bar_surface)
    trade_truth = _build_trade_truth(surfaces)
    feature_manifest = _feature_manifest(surfaces, trade_truth, run_dirs)
    coverage = _coverage_summary(run_inventory, surfaces, trade_truth, feature_manifest, calendar)
    quality = _quality_summary(trade_truth)
    exit_conflict = _exit_conflict_summary(surfaces["exit_trace"], run_dirs)
    lineage = _lineage(run_dirs)
    audit_df = _audit(coverage, trade_truth, feature_manifest, include_quarantine)
    status = _status(audit_df, coverage)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "include_quarantine_v1": include_quarantine,
        "write_bar_surface_v1": write_bar_surface,
        "status_v1": status["status_v1"],
        "coverage_v1": coverage,
        "quality_v1": quality,
        "exit_conflict_headline_v1": {
            "model_would_exit_rows_v1": exit_conflict.get("model_would_exit_rows_v1"),
            "model_would_exit_but_subfloor_rows_v1": exit_conflict.get("model_would_exit_but_subfloor_rows_v1"),
            "log_pattern_counts_v1": exit_conflict.get("log_pattern_counts_v1", {}),
        },
        "hard_status_v1": {
            "BEVIST": [
                "Monday R6 truth is materialized from actual Monday replay outputs.",
                "The package includes candidate, XGB, bar-feature, exit-trace, journal/outcome, lineage, and feature-manifest surfaces.",
                "The truth row count is not the 1689 exact-only surface and not the old 1852-only Monday R6.",
            ],
            "INDIKERT": [
                "This is the correct base for the next Monday R6 training/eval build.",
                "Warnings should be reviewed before training if exact XGB/bar coverage is below 100%.",
            ],
            "IKKE_ETABLERT": [
                "This materializer does not train or freeze a new R6 model.",
                "This is not a live policy/controller promotion.",
            ],
        },
        "artifacts_v1": OUTPUT_FILES,
    }

    trade_truth.to_parquet(output_dir / OUTPUT_FILES["trade_truth"], index=False)
    surfaces["candidates"].to_parquet(output_dir / OUTPUT_FILES["candidate_surface"], index=False)
    surfaces["xgb"].to_parquet(output_dir / OUTPUT_FILES["xgb_signal_surface"], index=False)
    surfaces["exit_trace"].to_parquet(output_dir / OUTPUT_FILES["exit_eval_trace"], index=False)
    if write_bar_surface:
        surfaces["bar"].to_parquet(output_dir / OUTPUT_FILES["bar_feature_surface"], index=False)
    feature_manifest.to_csv(output_dir / OUTPUT_FILES["feature_manifest"], index=False)
    run_inventory.to_csv(output_dir / OUTPUT_FILES["run_inventory"], index=False)
    lineage.to_csv(output_dir / OUTPUT_FILES["lineage"], index=False)
    audit_df.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    _write_json(output_dir / OUTPUT_FILES["coverage_summary"], coverage)
    _write_json(output_dir / OUTPUT_FILES["quality_summary"], quality)
    _write_json(output_dir / OUTPUT_FILES["exit_conflict_summary"], exit_conflict)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(
        output_dir / OUTPUT_FILES["manifest"],
        {
            "layer_name": "MONDAY_R6_CANONICAL_TRUTH_MANIFEST_V1",
            "materialized_at_utc_v1": summary["materialized_at_utc_v1"],
            "artifact_files_v1": OUTPUT_FILES,
            "input_run_count_v1": len(run_dirs),
            "input_reports_root_v1": str(reports_root),
            "calendar_file_v1": str(reports_root / CALENDAR_FILE),
        },
    )
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, quality, coverage, status), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize one canonical Monday R6 truth surface from actual Monday replay outputs.")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--active-only", action="store_true", help="Exclude calendar quarantine-marked weeks.")
    parser.add_argument("--no-bar-surface", action="store_true", help="Skip writing the combined raw bar feature surface.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        include_quarantine=not args.active_only,
        write_bar_surface=not args.no_bar_surface,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
