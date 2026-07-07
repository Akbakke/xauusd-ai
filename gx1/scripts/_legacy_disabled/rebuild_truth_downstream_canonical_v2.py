#!/usr/bin/env python3
"""
Canonical downstream rebuild for a completed truth replay root.

This orchestrates the V2 validation artifacts, the append-only all-trade review
ledger namespace, and the core truth audits that should travel with the build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from traceback import format_exc
from typing import Any, Callable, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.analysis.shadow_meta_v1 import (
    write_all_trade_review_ledger_closed_trades,
    write_shadow_meta_v2_activation_history_artifacts,
    write_shadow_meta_v2_audit_only_holdout_artifacts,
    write_shadow_meta_v2_contact_cohort_baseline_artifacts,
    write_shadow_meta_v2_contact_week_descriptive_artifacts,
    write_shadow_meta_v2_parallel_test_artifacts,
    write_shadow_meta_v2_prefreeze_pocket_artifacts,
    write_shadow_meta_v2_prefreeze_promote_semantics_artifact,
    write_shadow_meta_v2_prefreeze_shield_artifacts,
    write_shadow_meta_v2_prefreeze_threshold_artifacts,
    write_shadow_meta_v2_split_manifest,
)
from gx1.scripts.audit_truth_entry_skipability_pressure import (
    _resolve_reports_root as _resolve_truth_reports_root,
    build_skipability_pressure_summary,
)
from gx1.scripts.audit_truth_continuous_market_opportunity import (
    build_continuous_market_opportunity_summary,
)
from gx1.scripts.audit_truth_management_rl_readiness import (
    build_truth_management_rl_readiness_summary,
)
from gx1.scripts.audit_truth_trade_foundation_quality import (
    build_trade_foundation_quality_summary,
)


def _normalize_step_result(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {f"item_{idx}": str(item) for idx, item in enumerate(value)}
    return {"value": str(value)}


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return path


def rebuild_truth_downstream_canonical_v2(
    reports_root: Path,
    *,
    ledger_out_dir: Path | None = None,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    steps: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
        ("split_manifest", lambda: write_shadow_meta_v2_split_manifest(reports_root)),
        ("prefreeze_threshold", lambda: write_shadow_meta_v2_prefreeze_threshold_artifacts(reports_root)),
        ("prefreeze_shield", lambda: write_shadow_meta_v2_prefreeze_shield_artifacts(reports_root)),
        ("prefreeze_pocket", lambda: write_shadow_meta_v2_prefreeze_pocket_artifacts(reports_root)),
        ("prefreeze_promote_semantics", lambda: write_shadow_meta_v2_prefreeze_promote_semantics_artifact(reports_root)),
        ("audit_only_holdout", lambda: write_shadow_meta_v2_audit_only_holdout_artifacts(reports_root)),
        ("activation_history", lambda: write_shadow_meta_v2_activation_history_artifacts(reports_root)),
        ("contact_week_descriptive", lambda: write_shadow_meta_v2_contact_week_descriptive_artifacts(reports_root)),
        ("contact_cohort_baseline", lambda: write_shadow_meta_v2_contact_cohort_baseline_artifacts(reports_root)),
        ("parallel_tests", lambda: write_shadow_meta_v2_parallel_test_artifacts(reports_root)),
        (
            "all_trade_review_ledger",
            lambda: write_all_trade_review_ledger_closed_trades(reports_root, out_dir=ledger_out_dir),
        ),
    ]

    step_results: List[Dict[str, Any]] = []
    for step_name, step_fn in steps:
        try:
            result = step_fn()
            step_results.append(
                {
                    "step": step_name,
                    "status": "ok",
                    "result": _normalize_step_result(result),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised in live rebuilds
            step_results.append(
                {
                    "step": step_name,
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": format_exc().strip().splitlines()[-8:],
                }
            )

    ledger_dir = None
    ledger_step = next((row for row in step_results if row["step"] == "all_trade_review_ledger"), None)
    if ledger_step and ledger_step.get("status") == "ok":
        target_dir = ledger_step.get("result", {}).get("out_dir") or ledger_step.get("result", {}).get("target_dir")
        if target_dir:
            ledger_dir = Path(target_dir)

    skipability_summary = build_skipability_pressure_summary(reports_root, sample_limit=sample_limit)
    readiness_summary = build_truth_management_rl_readiness_summary(
        reports_root,
        review_dir=ledger_dir,
        sample_limit=sample_limit,
    )
    foundation_summary = build_trade_foundation_quality_summary(reports_root, sample_limit=sample_limit)
    market_opportunity_summary = build_continuous_market_opportunity_summary(
        reports_root,
        sample_limit=sample_limit,
    )

    skipability_path = _write_json(reports_root / "truth_entry_skipability_pressure_v1.json", skipability_summary)
    readiness_path = _write_json(reports_root / "truth_management_rl_readiness_v1.json", readiness_summary)
    foundation_path = _write_json(reports_root / "truth_trade_foundation_quality_v1.json", foundation_summary)
    market_opportunity_path = _write_json(
        reports_root / "truth_continuous_market_opportunity_v1.json",
        market_opportunity_summary,
    )

    summary_payload = {
        "reports_root": str(reports_root),
        "ledger_dir": str(ledger_dir) if ledger_dir else None,
        "steps": step_results,
        "audit_paths": {
            "truth_entry_skipability_pressure_v1": str(skipability_path),
            "truth_management_rl_readiness_v1": str(readiness_path),
            "truth_trade_foundation_quality_v1": str(foundation_path),
            "truth_continuous_market_opportunity_v1": str(market_opportunity_path),
        },
        "headline": {
            "zero_trade_runs": skipability_summary.get("completed_zero_trade_runs"),
            "candidate_rich_zero_trade_runs": skipability_summary.get("candidate_rich_zero_trade_runs"),
            "opportunity_rich_zero_trade_runs": len(market_opportunity_summary.get("opportunity_rich_zero_trade_runs_anchor", [])),
            "trade_count": foundation_summary.get("trade_count"),
            "outlook_v1": foundation_summary.get("outlook_v1"),
            "downstream_management_ready": readiness_summary.get("downstream_management_ready"),
            "blocked_step_count": sum(1 for row in step_results if row.get("status") == "blocked"),
        },
    }
    summary_path = _write_json(reports_root / "truth_downstream_canonical_rebuild_v1.json", summary_payload)
    return {
        "reports_root": reports_root,
        "ledger_dir": ledger_dir,
        "summary_path": summary_path,
        "skipability_path": skipability_path,
        "readiness_path": readiness_path,
        "foundation_path": foundation_path,
        "market_opportunity_path": market_opportunity_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical downstream rebuild for a completed truth replay root.")
    parser.add_argument("--reports-root", help="Path to the truth replay root. Defaults to ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt.")
    parser.add_argument("--ledger-out-dir", help="Optional append-only target directory for ALL_TRADE_REVIEW_LEDGER artifacts.")
    parser.add_argument("--sample-limit", type=int, default=10, help="Sample size for audit snippets.")
    args = parser.parse_args()

    reports_root = _resolve_truth_reports_root(args.reports_root)
    ledger_out_dir = Path(args.ledger_out_dir).expanduser().resolve() if args.ledger_out_dir else None
    result = rebuild_truth_downstream_canonical_v2(
        reports_root=reports_root,
        ledger_out_dir=ledger_out_dir,
        sample_limit=max(1, int(args.sample_limit)),
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
