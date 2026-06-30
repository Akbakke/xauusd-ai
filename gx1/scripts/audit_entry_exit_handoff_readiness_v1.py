#!/usr/bin/env python3
"""Audit readiness to hand active Entry/IQL replay evidence to Exit research.

This report is deliberately fail-closed: it does not train Exit, does not build
adapters, does not run replay, and does not open shadow/live/promotion. It
checks whether the current Entry/IQL evidence is strong enough to start an
Exit per-bar reconstruction line, then verifies that an active Exit substrate
exists before allowing any Exit model work.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_handoff_readiness_20260630_v1"
DEFAULT_IQL_COMPARISON_JSON = (
    REPORTS_ROOT / "entry_iql_replay_comparison_20260628_v1/ENTRY_IQL_REPLAY_COMPARISON_latest.json"
)
DEFAULT_IQL_SLICE_AUDIT_JSON = (
    REPORTS_ROOT / "entry_iql_replay_slice_audit_20260628_v1/ENTRY_IQL_REPLAY_SLICE_AUDIT_latest.json"
)
LEGACY_EXIT_TRUTH_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_ACTIVE_EXIT_SUBSTRATE_ROOT = REPORTS_ROOT / "entry_exit_per_bar_handoff_20260630_v1"

REQUIRED_TRADE_FIELDS = (
    "fold",
    "policy_id",
    "session",
    "vol_regime",
    "entry_time",
    "exit_time",
    "side",
    "p_long",
    "p_short",
    "p_flat",
    "score",
    "path_quality_pred",
    "bad_path_prob",
    "net_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "held_bars",
    "exit_reason",
)

REQUIRED_EXIT_SUBSTRATE_FIELDS = (
    "entry_trade_id",
    "bar_ts",
    "bar_index",
    "side",
    "action_set",
    "running_pnl_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "running_giveback_bps",
    "bars_held",
    "session",
    "vol_regime",
    "spread_bps",
    "atr_bps",
    "entry_score",
    "entry_p_long",
    "entry_p_short",
    "entry_p_flat",
    "entry_path_quality_pred",
    "entry_bad_path_prob",
    "entry_candidate_bundle_dir",
    "entry_iql_policy_id",
    "entry_replay_identity_hash",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=0).columns)


def _read_csv_or_empty(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


def _path_from_report(report: dict[str, Any], key: str) -> Path:
    raw = str(report.get(key) or "").strip()
    return Path(raw).expanduser().resolve() if raw else Path("")


def _exit_opportunity_iql_all(slice_audit: dict[str, Any]) -> dict[str, Any]:
    summary = slice_audit.get("exit_opportunity_summary")
    if not isinstance(summary, dict):
        return {}
    rows = summary.get("iql_all")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return {}


def _active_exit_substrate_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    patterns = ("*.parquet", "*.csv", "*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(path for path in files if path.is_file())


def _substrate_field_review(root: Path) -> dict[str, Any]:
    files = _active_exit_substrate_files(root)
    dataset_files = [path for path in files if path.suffix.lower() in {".csv", ".parquet"}]
    observed: list[str] = []
    dataset_path = ""
    if dataset_files:
        dataset_path = str(dataset_files[0])
        if dataset_files[0].suffix.lower() == ".csv":
            observed = _read_csv_header(dataset_files[0])
        else:
            try:
                observed = list(pd.read_parquet(dataset_files[0], engine="pyarrow").head(0).columns)
            except Exception:
                observed = []
    missing = [field for field in REQUIRED_EXIT_SUBSTRATE_FIELDS if field not in set(observed)]
    return {
        "root": str(root),
        "exists": root.exists(),
        "files": [str(path) for path in files],
        "dataset_path": dataset_path,
        "observed_fields": observed,
        "required_fields": list(REQUIRED_EXIT_SUBSTRATE_FIELDS),
        "missing_required_fields": missing,
        "ready": bool(dataset_path and not missing),
    }


def _substrate_report_review(root: Path) -> dict[str, Any]:
    report_path = root / "ENTRY_EXIT_PER_BAR_HANDOFF_latest.json"
    report = _read_json_or_empty(report_path)
    dataset_raw = str(report.get("dataset_csv") or "").strip()
    dataset_path = _path_from_report(report, "dataset_csv")
    dataset_exists = bool(dataset_raw) and dataset_path.exists()
    exclusions_raw = str(report.get("gap_exclusions_csv") or "").strip()
    exclusions_path = _path_from_report(report, "gap_exclusions_csv")
    exclusions_exists = bool(exclusions_raw) and exclusions_path.exists()
    exclusions_hash = str(report.get("gap_exclusions_csv_sha256") or "").strip()
    price_diagnostics = report.get("price_diagnostics") if isinstance(report.get("price_diagnostics"), dict) else {}
    supplemental_rows = int(price_diagnostics.get("supplemental_rows_used") or 0)
    supplemental_used = price_diagnostics.get("supplemental_paths_used")
    supplemental_hashes = price_diagnostics.get("supplemental_input_sha256")
    if not isinstance(supplemental_used, list):
        supplemental_used = []
    if not isinstance(supplemental_hashes, dict):
        supplemental_hashes = {}
    supplemental_hash_ready = supplemental_rows == 0 or (
        bool(supplemental_used)
        and all(str(path) in supplemental_hashes and bool(supplemental_hashes[str(path)]) for path in supplemental_used)
    )
    allowed_decisions = {"PASS", "PASS_WITH_EXPLICIT_GAP_EXCLUSIONS"}
    source_trade_count = int(report.get("source_trade_count") or report.get("trade_count") or 0)
    included_trade_count = int(report.get("included_trade_count") or report.get("complete_trade_count") or 0)
    excluded_trade_count = int(report.get("excluded_trade_count") or max(0, source_trade_count - included_trade_count))
    complete_trade_count = int(report.get("complete_trade_count") or 0)
    gap_exclusions_ready = excluded_trade_count == 0 or (exclusions_exists and bool(exclusions_hash))
    failures = report.get("failures")
    ready = (
        report_path.exists()
        and str(report.get("decision")) in allowed_decisions
        and dataset_exists
        and int(report.get("dataset_rows") or 0) > 0
        and source_trade_count > 0
        and included_trade_count > 0
        and complete_trade_count == included_trade_count
        and excluded_trade_count == source_trade_count - included_trade_count
        and gap_exclusions_ready
        and not failures
        and bool(supplemental_hash_ready)
        and report.get("exit_training_allowed") is False
        and report.get("exit_iql_allowed") is False
        and report.get("trainer_started") is False
        and report.get("replay_started") is False
        and report.get("promotion_shadow_live_allowed") is False
    )
    return {
        "path": str(report_path),
        "exists": report_path.exists(),
        "decision": report.get("decision"),
        "dataset_csv": str(dataset_path) if dataset_raw else "",
        "dataset_csv_exists": dataset_exists,
        "gap_exclusions_csv": str(exclusions_path) if exclusions_raw else "",
        "gap_exclusions_csv_exists": exclusions_exists,
        "gap_exclusions_csv_sha256_present": bool(exclusions_hash),
        "dataset_rows": int(report.get("dataset_rows") or 0),
        "trade_count": source_trade_count,
        "source_trade_count": source_trade_count,
        "included_trade_count": included_trade_count,
        "excluded_trade_count": excluded_trade_count,
        "complete_trade_count": complete_trade_count,
        "failure_count": len(failures) if isinstance(failures, list) else None,
        "covered_trade_ratio": report.get("covered_trade_ratio"),
        "supplemental_rows_used": supplemental_rows,
        "supplemental_paths_used": supplemental_used,
        "supplemental_hash_ready": bool(supplemental_hash_ready),
        "gap_exclusions_ready": bool(gap_exclusions_ready),
        "exit_training_allowed": report.get("exit_training_allowed"),
        "exit_iql_allowed": report.get("exit_iql_allowed"),
        "trainer_started": report.get("trainer_started"),
        "replay_started": report.get("replay_started"),
        "promotion_shadow_live_allowed": report.get("promotion_shadow_live_allowed"),
        "ready": bool(ready),
    }


def _trade_field_review(path: Path) -> dict[str, Any]:
    columns = _read_csv_header(path)
    missing = [field for field in REQUIRED_TRADE_FIELDS if field not in set(columns)]
    row_count = 0
    if path.exists():
        try:
            row_count = int(sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
        except Exception:
            row_count = 0
    return {
        "path": str(path),
        "exists": path.exists(),
        "row_count": max(0, row_count),
        "observed_fields": columns,
        "required_fields": list(REQUIRED_TRADE_FIELDS),
        "missing_required_fields": missing,
        "ready": bool(path.exists() and row_count > 0 and not missing),
    }


def _legacy_exit_review(root: Path) -> dict[str, Any]:
    expected_locks = (
        "EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1_20260429T100845Z_LOCK",
        "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK",
        "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1_20260429T120330Z_LOCK",
        "EXIT_ACTION_SUPPORT_AUGMENT_V1_20260429T130000Z_LOCK",
        "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T141227Z_LOCK",
        "EXIT_OFF_POLICY_EVAL_HARNESS_V1_20260429T150200Z_LOCK",
    )
    observed = sorted(path.name for path in root.glob("EXIT_*") if path.is_dir()) if root.exists() else []
    missing = [name for name in expected_locks if name not in set(observed)]
    return {
        "root": str(root),
        "exists": root.exists(),
        "expected_lock_dirs": list(expected_locks),
        "observed_exit_dirs": observed,
        "missing_expected_lock_dirs": missing,
        "ready": bool(root.exists() and not missing),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Handoff Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Entry evidence ready: `{report['entry_evidence_ready']}`",
        f"- Exit per-bar substrate ready: `{report['exit_per_bar_substrate_ready']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
        f"- Shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        "",
        "## Next",
        "",
        f"- {report['next_required_gate']}",
        "",
        "## Exit Opportunity",
        "",
    ]
    iql_all = report.get("iql_exit_opportunity_all") or {}
    if iql_all:
        lines.extend(
            [
                f"- IQL trades: `{iql_all.get('n_trades')}`",
                f"- Mean MFE capture: `{iql_all.get('mean_mfe_capture_ratio')}`",
                f"- P90 giveback bps: `{iql_all.get('p90_giveback_bps')}`",
                f"- Peak-oracle lift bps: `{iql_all.get('peak_oracle_lift_sum_bps')}`",
            ]
        )
    else:
        lines.append("- Missing")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = Path(args.iql_comparison_json).expanduser().resolve()
    slice_audit_path = Path(args.iql_slice_audit_json).expanduser().resolve()
    legacy_exit_root = Path(args.legacy_exit_truth_root).expanduser().resolve()
    active_exit_root = Path(args.active_exit_substrate_root).expanduser().resolve()

    comparison = _read_json_or_empty(comparison_path)
    slice_audit = _read_json_or_empty(slice_audit_path)
    iql_trade_path = _path_from_report(slice_audit, "iql_trades_path")
    candidate_trade_path = _path_from_report(slice_audit, "candidate_trades_path")
    exit_csv = _path_from_report(slice_audit, "exit_opportunity_csv")
    iql_trade_review = _trade_field_review(iql_trade_path)
    candidate_trade_review = _trade_field_review(candidate_trade_path)
    exit_opportunity = _read_csv_or_empty(exit_csv, nrows=10)
    iql_all = _exit_opportunity_iql_all(slice_audit)
    legacy_review = _legacy_exit_review(legacy_exit_root)
    substrate_report_review = _substrate_report_review(active_exit_root)
    substrate_review = _substrate_field_review(active_exit_root)

    entry_evidence_ready = (
        str(comparison.get("decision")) == "READY_FOR_PROMOTION_REVIEW_VEDTAK"
        and str(slice_audit.get("decision")) == "PASS"
        and bool(iql_all)
        and bool(iql_trade_review["ready"])
        and bool(candidate_trade_review["ready"])
        and exit_csv.exists()
        and not exit_opportunity.empty
    )
    exit_per_bar_ready = bool(substrate_report_review["ready"] and substrate_review["ready"])
    checks = [
        _check(
            "IQL replay comparison is ready",
            str(comparison.get("decision")) == "READY_FOR_PROMOTION_REVIEW_VEDTAK",
            {"path": str(comparison_path), "decision": comparison.get("decision")},
        ),
        _check(
            "IQL replay slice audit is PASS",
            str(slice_audit.get("decision")) == "PASS",
            {"path": str(slice_audit_path), "decision": slice_audit.get("decision")},
        ),
        _check("IQL replay trade log has exit handoff fields", bool(iql_trade_review["ready"]), iql_trade_review),
        _check("candidate replay trade log has exit handoff fields", bool(candidate_trade_review["ready"]), candidate_trade_review),
        _check(
            "exit opportunity diagnostics exist",
            exit_csv.exists() and not exit_opportunity.empty and bool(iql_all),
            {"exit_opportunity_csv": str(exit_csv), "iql_all": iql_all},
        ),
        _check(
            "legacy truth_e2e_sanity exit locks are available or superseded by active substrate",
            bool(legacy_review["ready"]) or exit_per_bar_ready,
            legacy_review,
        ),
        _check(
            "active Entry-bound exit per-bar materializer report is PASS or explicit gap-exclusion PASS",
            bool(substrate_report_review["ready"]),
            substrate_report_review,
        ),
        _check(
            "active Entry-bound exit per-bar substrate is available",
            exit_per_bar_ready,
            substrate_review,
        ),
        _check(
            "handoff audit never trains, replays, builds adapters, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "replay_started": False,
                "adapter_built": False,
                "exit_training_allowed": False,
                "exit_iql_allowed": False,
                "promotion_shadow_live_allowed": False,
            },
        ),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    if not entry_evidence_ready:
        decision = "BLOCKED_BY_ENTRY_EVIDENCE"
        next_required_gate = "repair Entry/IQL replay comparison, slice audit or trade-log provenance before Exit handoff"
    elif not exit_per_bar_ready:
        decision = "BLOCKED_BY_MISSING_EXIT_PER_BAR_SUBSTRATE"
        next_required_gate = "materialize active Entry-bound per-bar Exit substrate with the required handoff fields"
    else:
        decision = "READY_FOR_EXIT_PER_BAR_RECONSTRUCTION_REVIEW"
        next_required_gate = "audit active Exit per-bar reconstruction before any Exit Transformer/IQL training"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_HANDOFF_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_HANDOFF_READINESS_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_handoff_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "entry_evidence_ready": bool(entry_evidence_ready),
        "exit_per_bar_substrate_ready": bool(exit_per_bar_ready),
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "adapter_built": False,
        "promotion_shadow_live_allowed": False,
        "iql_comparison_json": str(comparison_path),
        "iql_slice_audit_json": str(slice_audit_path),
        "iql_replay_trade_log_review": iql_trade_review,
        "candidate_replay_trade_log_review": candidate_trade_review,
        "iql_exit_opportunity_all": iql_all,
        "exit_opportunity_csv": str(exit_csv),
        "legacy_exit_truth_root_review": legacy_review,
        "active_exit_substrate_report_review": substrate_report_review,
        "active_exit_substrate_review": substrate_review,
        "required_exit_substrate_fields": list(REQUIRED_EXIT_SUBSTRATE_FIELDS),
        "checks": checks,
        "failures": failures,
        "next_required_gate": next_required_gate,
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_HANDOFF_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_HANDOFF_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "entry_evidence_ready": report["entry_evidence_ready"],
                    "exit_per_bar_substrate_ready": report["exit_per_bar_substrate_ready"],
                    "failures": failures,
                    "json_path": str(json_path),
                    "next_required_gate": next_required_gate,
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and decision != "READY_FOR_EXIT_PER_BAR_RECONSTRUCTION_REVIEW":
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iql-comparison-json", default=str(DEFAULT_IQL_COMPARISON_JSON))
    ap.add_argument("--iql-slice-audit-json", default=str(DEFAULT_IQL_SLICE_AUDIT_JSON))
    ap.add_argument("--legacy-exit-truth-root", default=str(LEGACY_EXIT_TRUTH_ROOT))
    ap.add_argument("--active-exit-substrate-root", default=str(DEFAULT_ACTIVE_EXIT_SUBSTRATE_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
