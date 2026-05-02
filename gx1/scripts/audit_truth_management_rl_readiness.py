from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
LEDGER_NAMESPACE_PREFIX = "ALL_TRADE_REVIEW_LEDGER_"
SUPPORTED_RUN_PREFIXES = ("E2E_SANITY_ORDERFIX_", "TRUTH_MONFRI_WEEK_")
RUN_ID_RE = re.compile(r"^(?:E2E_SANITY_ORDERFIX|TRUTH_MONFRI_WEEK)_(\d{8})_(\d{8})$")
DOWNSTREAM_SENTINELS = [
    "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json",
    "shadow_meta_all_trade_review_management_bandit_status_v1.json",
    "shadow_meta_all_trade_review_management_policy_training_examples_core_v4.parquet",
]
ENTRY_HANDOFF_STATUS_FILE = "shadow_meta_all_trade_review_entry_actualization_status_v1.json"
ENTRY_HANDOFF_SUMMARY_FILE = "shadow_meta_all_trade_review_entry_actual_take_to_management_handoff_summary_v1.json"
TRAINABLE_REQUIRED_COLS = [
    "decision_ts_utc",
    "side",
    "session",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "margin",
    "uncertainty_score",
    "entry_spread_bps",
    "open_ts_utc",
    "close_ts_utc",
    "pnl_bps",
    "mfe_bps",
    "mae_bps",
    "bars_in_trade",
    "exit_reason",
    "mfe_threshold_bps",
    "positive_exit",
    "cata",
    "never_mfe",
    "good_mfe_then_rot",
    "meta_allow_label_v1",
    "mfe_first_n_pred",
    "path_quality_pred",
]


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _read_parquet_with_run_id(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "run_id" not in df.columns:
        df = df.copy()
        df["run_id"] = path.parent.name
    return df


def _runs_root(reports_root: Path) -> Path:
    candidate = reports_root / "runs"
    return candidate if candidate.exists() else reports_root


def _iter_run_dirs(reports_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in _runs_root(reports_root).iterdir()
            if path.is_dir() and RUN_ID_RE.fullmatch(path.name)
        ]
    )


def _resolve_review_dir(reports_root: Path, review_dir_arg: str | None = None) -> Path | None:
    if review_dir_arg:
        review_dir = Path(review_dir_arg).expanduser().resolve()
        if not review_dir.exists():
            raise FileNotFoundError(f"Review dir does not exist: {review_dir}")
        return review_dir

    if all((reports_root / name).exists() for name in DOWNSTREAM_SENTINELS):
        return reports_root

    namespace_dirs = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir() and path.name.startswith(LEDGER_NAMESPACE_PREFIX)
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in namespace_dirs:
        if all((candidate / name).exists() for name in DOWNSTREAM_SENTINELS):
            return candidate
    return namespace_dirs[0] if namespace_dirs else None


def _regret_summary(paths: list[Path]) -> dict[str, Any]:
    overall_rates: list[float] = []
    threshold_rates: list[float] = []
    replay_end_rates: list[float] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        overall = (payload.get("overall") or {}).get("regret_rate")
        threshold = (payload.get("threshold_exits") or {}).get("meaningful_followthrough_rate")
        threshold_count = (payload.get("threshold_exits") or {}).get("count")
        replay_end = ((payload.get("replay_end_observability") or {}).get("overall") or {}).get("regret_rate")
        if overall is not None:
            overall_rates.append(float(overall))
        if threshold is not None and threshold_count not in (None, 0):
            threshold_rates.append(float(threshold))
        if replay_end is not None:
            replay_end_rates.append(float(replay_end))
    return {
        "files": len(paths),
        "mean_regret_rate": (sum(overall_rates) / len(overall_rates)) if overall_rates else None,
        "median_regret_rate": statistics.median(overall_rates) if overall_rates else None,
        "mean_threshold_followthrough_rate": (sum(threshold_rates) / len(threshold_rates)) if threshold_rates else None,
        "mean_replay_end_regret_rate": (sum(replay_end_rates) / len(replay_end_rates)) if replay_end_rates else None,
    }


def _status(pass_condition: bool) -> str:
    return "PASS" if pass_condition else "FAIL"


def build_truth_management_rl_readiness_summary(
    reports_root: Path,
    *,
    review_dir: Path | None = None,
    sample_limit: int = 10,
) -> dict[str, Any]:
    run_dirs = _iter_run_dirs(reports_root)
    completed_markers = sorted([run_dir / "RUN_COMPLETED.json" for run_dir in run_dirs if (run_dir / "RUN_COMPLETED.json").exists()])
    trade_paths = sorted(
        [path for run_dir in run_dirs for path in run_dir.glob(f"trade_outcomes_{run_dir.name}_MERGED.parquet")]
    )
    shadow_paths = sorted(
        [path for run_dir in run_dirs for path in run_dir.glob(f"shadow_meta_candidates_{run_dir.name}_MERGED.parquet")]
    )
    provenance_paths = sorted(
        [path for run_dir in run_dirs for path in run_dir.glob(f"shadow_meta_provenance_{run_dir.name}.json")]
    )
    regret_paths = sorted(
        [path for run_dir in run_dirs for path in run_dir.glob("post_exit_regret_audit_*.json")]
    )

    trade_df = pd.concat([_read_parquet_with_run_id(path) for path in trade_paths], ignore_index=True) if trade_paths else pd.DataFrame()
    shadow_df = pd.concat([_read_parquet_with_run_id(path) for path in shadow_paths], ignore_index=True) if shadow_paths else pd.DataFrame()
    trainable_df = shadow_df.loc[shadow_df.get("trainable_mask_v1").fillna(False).astype(bool)].copy() if not shadow_df.empty else pd.DataFrame()

    trainable_fill_rates = {
        col: float(trainable_df[col].notna().mean())
        for col in TRAINABLE_REQUIRED_COLS
        if col in trainable_df.columns
    }

    per_run_trade_counts = []
    if not trade_df.empty:
        per_run_counts_df = (
            trade_df.groupby("run_id", dropna=False)
            .size()
            .reset_index(name="trade_count")
            .sort_values(["trade_count", "run_id"], ascending=[False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        per_run_trade_counts = per_run_counts_df.to_dict(orient="records")
    else:
        per_run_counts_df = pd.DataFrame(columns=["run_id", "trade_count"])

    zero_trade_runs = sorted(
        [path.parent.name for path in trade_paths if len(pd.read_parquet(path, columns=["trade_id"])) == 0]
    )
    max_trade_week = per_run_trade_counts[0] if per_run_trade_counts else None
    min_nonzero_trade_week = None
    if per_run_trade_counts:
        for row in reversed(per_run_trade_counts):
            if int(row["trade_count"]) > 0:
                min_nonzero_trade_week = row
                break

    missing_from_trainable_df = pd.DataFrame()
    if not trade_df.empty and "trade_uid" in trade_df.columns and "trade_uid" in trainable_df.columns:
        missing_mask = ~trade_df["trade_uid"].astype("string").isin(set(trainable_df["trade_uid"].astype("string")))
        missing_from_trainable_df = trade_df.loc[
            missing_mask,
            ["run_id", "trade_uid", "trade_id", "exit_reason", "pnl_bps", "mfe_bps", "mae_bps"],
        ].copy()

    session_cata_rates = []
    hour_cata_rates = []
    if not trainable_df.empty:
        for session, part in trainable_df.groupby("session", dropna=False):
            cata_rate = float(part.get("cata", pd.Series(dtype="boolean")).fillna(False).astype(bool).mean())
            session_cata_rates.append(
                {
                    "session": str(session),
                    "rows": int(len(part)),
                    "cata_rate": cata_rate,
                    "positive_exit_rate": float(part.get("positive_exit", pd.Series(dtype="boolean")).fillna(False).astype(bool).mean()),
                }
            )
        ts = pd.to_datetime(trainable_df.get("decision_ts_utc"), utc=True, errors="coerce")
        work = trainable_df.copy()
        work["decision_hour_utc_v1"] = ts.dt.hour
        for hour, part in work.groupby("decision_hour_utc_v1", dropna=False):
            hour_cata_rates.append(
                {
                    "hour_utc": None if pd.isna(hour) else int(hour),
                    "rows": int(len(part)),
                    "cata_rate": float(part.get("cata", pd.Series(dtype="boolean")).fillna(False).astype(bool).mean()),
                }
            )
        hour_cata_rates.sort(key=lambda row: (-row["cata_rate"], -row["rows"], row["hour_utc"] if row["hour_utc"] is not None else 99))
        session_cata_rates.sort(key=lambda row: (-row["rows"], row["session"]))

    resolved_review_dir = Path(review_dir).expanduser().resolve() if review_dir is not None else _resolve_review_dir(reports_root)
    downstream_artifacts = {
        name: ((resolved_review_dir / name).exists() if resolved_review_dir is not None else False)
        for name in DOWNSTREAM_SENTINELS
    }
    downstream_status_files = [
        "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json",
        "shadow_meta_all_trade_review_management_rl_sequence_status_v1.json",
        "shadow_meta_all_trade_review_management_bandit_status_v1.json",
        "shadow_meta_all_trade_review_management_exit_local_status_v1.json",
    ]
    downstream_status_payloads: dict[str, Any] = {}
    if resolved_review_dir is not None:
        for name in downstream_status_files:
            path = resolved_review_dir / name
            if path.exists():
                downstream_status_payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    entry_actualization_status_payload: dict[str, Any] = {}
    entry_handoff_summary_payload: dict[str, Any] = {}
    if resolved_review_dir is not None:
        entry_status_path = resolved_review_dir / ENTRY_HANDOFF_STATUS_FILE
        if entry_status_path.exists():
            entry_actualization_status_payload = json.loads(entry_status_path.read_text(encoding="utf-8"))
        entry_handoff_summary_path = resolved_review_dir / ENTRY_HANDOFF_SUMMARY_FILE
        if entry_handoff_summary_path.exists():
            entry_handoff_summary_payload = json.loads(entry_handoff_summary_path.read_text(encoding="utf-8"))
    runtime_recovery_fallback_detected = any(
        (
            isinstance(payload, dict)
            and (
                "runtime_recovery_fallback_reason_v1" in payload
                or "RUNTIME_RECOVERY_EMPTY_FALLBACK" in {str(value) for value in payload.values()}
            )
        )
        for payload in downstream_status_payloads.values()
    )
    downstream_management_ready = bool(all(downstream_artifacts.values()) and not runtime_recovery_fallback_detected)
    entry_to_management_handoff_status = entry_actualization_status_payload.get("ENTRY_TO_MANAGEMENT_HANDOFF_STATUS")
    handoff_status_counts = entry_handoff_summary_payload.get("management_handoff_status_counts_v1") or {}
    actual_take_with_management_diagnostic_only_review_count = handoff_status_counts.get(
        "ACTUAL_TAKE_WITH_MANAGEMENT_DIAGNOSTIC_ONLY_REVIEW"
    )
    actual_take_with_nontrainable_management_review_only_count = handoff_status_counts.get(
        "ACTUAL_TAKE_WITH_NONTRAINABLE_MANAGEMENT_REVIEW_ONLY"
    )
    actual_take_without_provable_management_head_count = handoff_status_counts.get(
        "ACTUAL_TAKE_WITHOUT_PROVABLE_MANAGEMENT_HEAD"
    )
    if handoff_status_counts:
        actual_take_without_provable_management_head_count = int(
            sum(
                int(value)
                for key, value in handoff_status_counts.items()
                if str(key) != "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD"
            )
        )
    actual_take_with_provable_management_head_count = handoff_status_counts.get(
        "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD"
    )
    entry_to_management_handoff_fully_established = (
        False
        if actual_take_without_provable_management_head_count is None
        else int(actual_take_without_provable_management_head_count) == 0
    )

    trade_count = int(len(trade_df))
    clean_good_trade_count = int(((trade_df.get("mfe_bps", pd.Series(dtype=float)) >= 20.0) & (trade_df.get("mae_bps", pd.Series(dtype=float)) > -5.0)).sum()) if not trade_df.empty else 0
    positive_mfe_no_mae_count = int(((trade_df.get("mfe_bps", pd.Series(dtype=float)) > 0.0) & (trade_df.get("mae_bps", pd.Series(dtype=float)) >= 0.0)).sum()) if not trade_df.empty else 0

    summary = {
        "reports_root": str(reports_root),
        "downstream_review_dir": (str(resolved_review_dir) if resolved_review_dir is not None else None),
        "run_dir_count": len(run_dirs),
        "completed_runs": len(completed_markers),
        "trade_outcome_files": len(trade_paths),
        "shadow_meta_candidate_files": len(shadow_paths),
        "shadow_meta_provenance_files": len(provenance_paths),
        "post_exit_regret_files": len(regret_paths),
        "trade_count": trade_count,
        "win_count": int((trade_df.get("pnl_bps", pd.Series(dtype=float)) > 0.0).sum()) if not trade_df.empty else 0,
        "loss_count": int((trade_df.get("pnl_bps", pd.Series(dtype=float)) <= 0.0).sum()) if not trade_df.empty else 0,
        "trainable_shadow_rows": int(len(trainable_df)),
        "accepted_shadow_rows": int((shadow_df.get("decision", pd.Series(dtype="string")).astype("string").str.upper() == "LONG").sum()) if not shadow_df.empty else 0,
        "missing_from_trainable_count": int(len(missing_from_trainable_df)),
        "missing_from_trainable_sample": missing_from_trainable_df.head(sample_limit).to_dict(orient="records"),
        "trainable_fill_rates": trainable_fill_rates,
        "session_cata_rates": session_cata_rates,
        "top_hour_cata_rates": hour_cata_rates[:10],
        "zero_trade_run_ids": zero_trade_runs,
        "zero_trade_run_count": len(zero_trade_runs),
        "max_trade_week": max_trade_week,
        "min_nonzero_trade_week": min_nonzero_trade_week,
        "per_run_trade_count_top10": per_run_trade_counts[:10],
        "positive_mfe_no_mae_count": positive_mfe_no_mae_count,
        "clean_good_trade_mfe20_mae5_count": clean_good_trade_count,
        "clean_good_trade_mfe20_mae5_rate": (float(clean_good_trade_count / trade_count) if trade_count else None),
        "shadow_helper_label_available": bool("good_trade_mfe20_mae5_v1" in shadow_df.columns),
        "shadow_helper_label_true_count": int(shadow_df.get("good_trade_mfe20_mae5_v1", pd.Series(dtype="boolean")).fillna(False).astype(bool).sum()) if not shadow_df.empty else 0,
        "downstream_artifacts_present": downstream_artifacts,
        "downstream_status_payloads_v1": downstream_status_payloads,
        "downstream_runtime_recovery_fallback_detected": runtime_recovery_fallback_detected,
        "downstream_management_ready": downstream_management_ready,
        "entry_actualization_status_v1": entry_actualization_status_payload,
        "entry_to_management_handoff_status_v1": entry_to_management_handoff_status,
        "entry_to_management_handoff_summary_v1": entry_handoff_summary_payload,
        "entry_to_management_handoff_fully_established_v1": entry_to_management_handoff_fully_established,
        "actual_take_without_provable_management_head_count_v1": actual_take_without_provable_management_head_count,
        "actual_take_with_provable_management_head_count_v1": actual_take_with_provable_management_head_count,
        "actual_take_with_management_diagnostic_only_review_count_v1": actual_take_with_management_diagnostic_only_review_count,
        "actual_take_with_nontrainable_management_review_only_count_v1": actual_take_with_nontrainable_management_review_only_count,
        "regret_summary": _regret_summary(regret_paths),
        "verdicts": {
            "completed_shadow_surface_status": _status(len(completed_markers) == len(trade_paths) == len(shadow_paths) == len(provenance_paths)),
            "trainable_backfill_status": _status(len(missing_from_trainable_df) == 0),
            "trainable_fill_status": _status(all(rate >= 0.999 for rate in trainable_fill_rates.values()) if trainable_fill_rates else False),
            "downstream_management_status": _status(downstream_management_ready),
            "entry_to_management_handoff_status": _status(entry_to_management_handoff_fully_established),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a truth replay root for management RL readiness.")
    parser.add_argument("--reports-root", help="Path to the truth replay root. Defaults to ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt.")
    parser.add_argument("--review-dir", help="Optional ALL_TRADE_REVIEW_LEDGER namespace directory to inspect.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--sample-limit", type=int, default=10, help="How many sample rows to keep for trainable-gap examples.")
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    review_dir = _resolve_review_dir(reports_root, args.review_dir)
    summary = build_truth_management_rl_readiness_summary(
        reports_root,
        review_dir=review_dir,
        sample_limit=max(1, int(args.sample_limit)),
    )
    payload = json.dumps(summary, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        Path(args.output).expanduser().resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
