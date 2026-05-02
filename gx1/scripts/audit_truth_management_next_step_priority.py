from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_SUFFIX = "MANAGEMENT_AUDIT_EXTENSION_V1"
BUILD_SUMMARY_FILE = "shadow_meta_all_trade_review_management_audit_extension_build_summary_v1.json"
ROOT_CAUSE_SUMMARY_FILE = "shadow_meta_all_trade_review_management_root_cause_parallel_audit_summary_v1.json"
TRIAGE_SUMMARY_FILE = "shadow_meta_all_trade_review_management_parallel_next_step_triage_summary_v1.json"
WALKFORWARD_SUMMARY_FILE = "shadow_meta_all_trade_review_management_walkforward_regime_scoreboard_summary_v1.json"
REGIME_INCREMENTAL_SUMMARY_FILE = "shadow_meta_all_trade_review_management_regime_incremental_signal_audit_summary_v1.json"
FOUNDATION_SUMMARY_FILE = "truth_trade_foundation_quality_v1.json"
SKIPABILITY_SUMMARY_FILE = "truth_entry_skipability_pressure_v1.json"
MARKET_OPPORTUNITY_SUMMARY_FILE = "truth_continuous_market_opportunity_v1.json"


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _priority_row(
    rank: int,
    track: str,
    why_now: str,
    action: str,
    impact: str,
    gate: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "priority_rank_v1": rank,
        "track_v1": track,
        "why_now_v1": why_now,
        "recommended_action_v1": action,
        "expected_impact_v1": impact,
        "gate_v1": gate,
        "evidence_v1": evidence,
    }


def build_management_next_step_priority_summary(
    reports_root: Path,
    *,
    extension_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_root = reports_root.expanduser().resolve()
    resolved_extension = (
        extension_dir.expanduser().resolve()
        if extension_dir is not None
        else _resolve_extension_dir(resolved_root)
    )

    build_summary = _read_json(resolved_extension / BUILD_SUMMARY_FILE)
    root_cause_summary = _read_json(resolved_extension / ROOT_CAUSE_SUMMARY_FILE)
    triage_summary = _read_json(resolved_extension / TRIAGE_SUMMARY_FILE)
    walkforward_summary = _read_json(resolved_extension / WALKFORWARD_SUMMARY_FILE)
    regime_incremental_summary = _read_json(resolved_extension / REGIME_INCREMENTAL_SUMMARY_FILE)
    foundation_summary = _read_json(resolved_root / FOUNDATION_SUMMARY_FILE)
    skipability_summary = _read_json(resolved_root / SKIPABILITY_SUMMARY_FILE)
    market_opportunity_summary = _read_json(resolved_root / MARKET_OPPORTUNITY_SUMMARY_FILE)

    policy_summary = build_summary.get("policy_logging_summary_v1", {})
    regime_summary = build_summary.get("regime_overlay_summary_v1", {})
    path_summary = build_summary.get("path_dynamics_source_binding_summary_v1", {})
    reference_summary = build_summary.get("reference_compare_summary_v1", {})
    local_reference_summary = build_summary.get("local_reference_diff_summary_v1", {})
    manual_summary = build_summary.get("manual_deep_dive_summary_v1", {})
    hold_longer = (foundation_summary.get("hold_longer_pressure") or {})
    profitability = (foundation_summary.get("profitability") or {})
    triage_winner = ((triage_summary.get("triage_summary_v1") or {}).get("winner_v1") or {})

    missing_derivations = [
        row.get("field_name_v1")
        for row in path_summary.get("derivation_rows_v1", [])
        if row.get("derivation_status_v1") == "NOT_ESTABLISHED"
    ]

    priorities = [
        _priority_row(
            1,
            "COARSE_CONDITIONING_GRID_AND_RAW_SCORE",
            "Current bucket taxonomy does not create usable contrasts, while hold-longer pressure is high.",
            "Replace the current management bucket-first view with a coarse conditioning grid and/or raw-score teacher for HOLD vs EXIT, then retrain management before any new replay.",
            "Directly targets under-capture and should improve 'hold longer' behavior without inventing synthetic micro-pockets.",
            "BLOCKS_MANAGEMENT_RETRAIN",
            [
                f"Triaged next-step winner is {triage_winner.get('track_v1', 'UNKNOWN')}.",
                f"Root-cause primary is {root_cause_summary.get('root_cause_primary_v1', 'UNKNOWN')} with recommendation {root_cause_summary.get('recommended_next_step_v1', 'UNKNOWN')}.",
                f"Hold-longer pressure shows mean extra value {((hold_longer.get('extra_value_bps') or {}).get('mean'))} bps and >=10 bps extra on {hold_longer.get('meaningful_extra_value_10bps_rate')} of trades.",
                f"Current average pnl is {profitability.get('avg_pnl_bps')} bps/trade with max drawdown {profitability.get('max_drawdown_bps')} bps.",
            ],
        ),
        _priority_row(
            2,
            "BEHAVIOR_POLICY_AND_PROPENSITY_LOGGING",
            "The data is real and action-observed, but the policy surface is still not propensity-aware.",
            "Establish true behavior-policy logging and propensity capture in the management decision log so supervised management training and later RL/off-policy evaluation rest on observed policy, not inference.",
            "Turns the management ledger into a real training/evaluation substrate for RL-related work instead of a descriptive audit only.",
            "BLOCKS_RL",
            [
                f"Instrumentation status is {policy_summary.get('instrumentation_status_v1', 'UNKNOWN')}.",
                f"Behavior-policy readiness is {policy_summary.get('behavior_policy_readiness_v1', 'UNKNOWN')}.",
                f"Propensity readiness is {policy_summary.get('propensity_readiness_v1', 'UNKNOWN')}.",
                f"Observed management sample rows: {policy_summary.get('observed_sample_rows_v1')}, decision-log rows: {policy_summary.get('decision_log_rows_v1')}, manual-review attachments: {policy_summary.get('manual_review_rows_attached_v1')}.",
            ],
        ),
        _priority_row(
            3,
            "COARSE_REGIME_AND_SESSION_CONDITIONING",
            "The current overlay taxonomy is too thin and too fragmented to trust as a controller.",
            "Collapse the regime overlay to a coarser, walk-forward-safe session/regime conditioning layer and use it as a feature family, not as a brittle pocket controller.",
            "Improves generalization and reduces the risk of overfitting a few pockets or weeks.",
            "BLOCKS_ROBUST_GENERALIZATION",
            [
                f"Root-cause secondary is {root_cause_summary.get('root_cause_secondary_v1', 'UNKNOWN')}.",
                f"Regime consistency status is {regime_summary.get('regime_consistency_status_v1', 'UNKNOWN')}, while outcome advantage is {regime_summary.get('outcome_advantage_status_v1', 'UNKNOWN')}.",
                f"Walk-forward thin groups (<10 rows) count is {walkforward_summary.get('thin_groups_lt_10_v1')}, with stability counts {walkforward_summary.get('stability_status_counts_v1')}.",
                f"Regime incremental verdict counts are {regime_incremental_summary.get('verdict_counts_v1')}.",
            ],
        ),
    ]

    if missing_derivations:
        priorities.append(
            _priority_row(
                4,
                "PATH_DYNAMICS_DERIVABLE_MINUTES_FIELDS",
                "The upstream path-dynamics fields are source-bound, but two timestamp derivations are not independently reconstructible yet.",
                "Log or derive the missing `minutes_since_last_peak` and `minutes_since_last_mfe` fields in the management decision harness so the as-of state is fully auditable and portable.",
                "Strengthens explainability and reduces future ambiguity around management state reconstruction.",
                "DO_AFTER_TOP3",
                [
                    f"Missing derivations are {missing_derivations}.",
                    "Replay coverage for the five path-dynamics fields is already 100%, so this is an auditability upgrade rather than a data-rescue task.",
                ],
            )
        )

    retrain_replay_gate = "WAIT_FOR_MANAGEMENT_RETRAIN_THEN_REPLAY"
    if any(row.get("gate_v1") == "BLOCKS_RL" for row in priorities):
        rl_gate = "DO_NOT_START_RL_TRAINING_YET"
    else:
        rl_gate = "RL_CAN_PROGRESS"

    report = {
        "reports_root": str(resolved_root),
        "extension_dir": str(resolved_extension),
        "overall_verdict_v1": "REAL_AUDIT_SURFACE_CLEAR_PRIORITIES_NOT_RL_READY",
        "foundation_snapshot_v1": {
            "trade_count": foundation_summary.get("trade_count"),
            "outlook_v1": foundation_summary.get("outlook_v1"),
            "avg_pnl_bps": profitability.get("avg_pnl_bps"),
            "profit_factor": profitability.get("profit_factor"),
            "max_drawdown_bps": profitability.get("max_drawdown_bps"),
            "hold_longer_mean_extra_value_bps": ((hold_longer.get("extra_value_bps") or {}).get("mean")),
            "hold_longer_10bps_rate": hold_longer.get("meaningful_extra_value_10bps_rate"),
            "completed_zero_trade_runs": skipability_summary.get("completed_zero_trade_runs"),
            "candidate_rich_zero_trade_runs": skipability_summary.get("candidate_rich_zero_trade_runs"),
            "opportunity_rich_zero_trade_runs_anchor": market_opportunity_summary.get(
                "opportunity_rich_zero_trade_runs_anchor", []
            ),
        },
        "source_invariants_v1": build_summary.get("source_invariants_v1", {}),
        "priority_tracks_v1": priorities,
        "do_not_optimize_toward_v1": [
            {
                "topic_v1": "LOCAL_TIGHT_REFERENCE_EXIT_EXPLANATIONS",
                "reason_v1": "Current line honestly has no local tight references; this is a finding, not a bug to patch with synthetic neighbors.",
                "evidence_v1": [
                    f"Reference compare summary says local tight reference pairs = {reference_summary.get('local_tight_reference_pairs_v1')}.",
                    f"Local reference diff status is {local_reference_summary.get('status_v1', 'UNKNOWN')}.",
                ],
            },
            {
                "topic_v1": "MANUAL_DEEP_DIVE_CASE_STORIES_AS_POLICY_TRUTH",
                "reason_v1": "The manual deep-dive packet is descriptive and all five target cases remain IKKE_ETABLERT.",
                "evidence_v1": [
                    f"Manual deep-dive conclusion counts are {manual_summary.get('overall_conclusion_status_counts_v1')}.",
                ],
            },
            {
                "topic_v1": "ZERO-TRADE_WEEK_PATCHING_INSIDE_MANAGEMENT",
                "reason_v1": "The zero-trade issue is real but belongs to entry/gating calibration, not to management exit control.",
                "evidence_v1": [
                    f"Completed zero-trade runs = {skipability_summary.get('completed_zero_trade_runs')}.",
                    f"Candidate-rich zero-trade runs = {skipability_summary.get('candidate_rich_zero_trade_runs')}.",
                ],
            },
        ],
        "recommended_execution_order_v1": [
            "coarse_conditioning_grid_and_raw_score",
            "behavior_policy_and_propensity_logging",
            "coarse_regime_and_session_conditioning",
            "path_dynamics_derivable_minutes_fields",
            "retrain_management_teacher",
            "run_full_replay_as_evaluation",
            "only_then_consider_rl_training",
        ],
        "gates_v1": {
            "management_retrain_gate_v1": retrain_replay_gate,
            "rl_training_gate_v1": rl_gate,
            "replay_gate_v1": "RUN_REPLAY_AFTER_NEW_MANAGEMENT_WEIGHTS_ONLY",
        },
    }
    return report


def write_management_next_step_priority_artifacts(
    reports_root: Path,
    *,
    extension_dir: Path | None = None,
) -> dict[str, str]:
    summary = build_management_next_step_priority_summary(
        reports_root=reports_root,
        extension_dir=extension_dir,
    )
    json_path = reports_root / "truth_management_next_step_priority_v1.json"
    md_path = reports_root / "truth_management_next_step_priority_v1.md"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        "# Truth Management Next-Step Priority V1",
        "",
        f"- reports_root: `{summary['reports_root']}`",
        f"- extension_dir: `{summary['extension_dir']}`",
        f"- overall_verdict_v1: `{summary['overall_verdict_v1']}`",
        "",
        "## Foundation Snapshot",
        "",
    ]
    for key, value in summary["foundation_snapshot_v1"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Priority Tracks", ""])
    for row in summary["priority_tracks_v1"]:
        lines.append(f"### P{row['priority_rank_v1']} {row['track_v1']}")
        lines.append("")
        lines.append(f"- why_now_v1: {row['why_now_v1']}")
        lines.append(f"- recommended_action_v1: {row['recommended_action_v1']}")
        lines.append(f"- expected_impact_v1: {row['expected_impact_v1']}")
        lines.append(f"- gate_v1: `{row['gate_v1']}`")
        lines.append("- evidence_v1:")
        for item in row["evidence_v1"]:
            lines.append(f"  - {item}")
        lines.append("")

    lines.extend(["## Do Not Optimize Toward", ""])
    for row in summary["do_not_optimize_toward_v1"]:
        lines.append(f"- {row['topic_v1']}: {row['reason_v1']}")

    lines.extend(["", "## Recommended Execution Order", ""])
    for item in summary["recommended_execution_order_v1"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Gates", ""])
    for key, value in summary["gates_v1"].items():
        lines.append(f"- {key}: `{value}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a prioritized management next-step report from truth audits.")
    parser.add_argument("--reports-root", dest="reports_root", default=None)
    parser.add_argument("--extension-dir", dest="extension_dir", default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    extension_dir = (
        Path(args.extension_dir).expanduser().resolve()
        if args.extension_dir
        else None
    )
    written = write_management_next_step_priority_artifacts(
        reports_root=reports_root,
        extension_dir=extension_dir,
    )
    print(json.dumps(written, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
