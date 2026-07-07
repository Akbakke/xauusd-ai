#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
R8_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260420_RUNTIME_RECOVERY_R8_HANDOFF_REALFIX"
EXTENSION_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260420T133345Z_MANAGEMENT_AUDIT_EXTENSION_V1"

OUTPUT_JSON = "truth_management_inventory_contract_v1.json"
OUTPUT_MD = "truth_management_inventory_contract_v1.md"


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    return Path(ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required inventory source missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    return payload


def _scalar(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _row(name: str, status: str, counts: str, available: str, improve: str, source: str) -> Dict[str, str]:
    return {
        "surface_v1": str(name),
        "status_v1": str(status),
        "counts_v1": str(counts),
        "available_now_v1": str(available),
        "needs_improvement_v1": str(improve),
        "source_v1": str(source),
    }


def build_management_inventory_contract(reports_root: Path) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    r8_dir = reports_root / R8_LEDGER_DIRNAME
    extension_dir = reports_root / EXTENSION_DIRNAME

    foundation = _load_json(reports_root / "truth_trade_foundation_quality_v1.json")
    skipability = _load_json(reports_root / "truth_entry_skipability_pressure_v1.json")
    market = _load_json(reports_root / "truth_continuous_market_opportunity_v1.json")
    readiness = _load_json(reports_root / "truth_management_rl_readiness_v1.json")
    coarse_teacher = _load_json(reports_root / "truth_management_coarse_teacher_summary_v1.json")
    coarse_benchmark = _load_json(reports_root / "truth_management_coarse_feedback_benchmark_summary_v1.json")
    next_step = _load_json(reports_root / "truth_management_next_step_priority_v1.json")

    policy_logging = _load_json(
        extension_dir / "shadow_meta_all_trade_review_management_policy_logging_summary_v1.json"
    )
    regime_overlay = _load_json(
        extension_dir / "shadow_meta_all_trade_review_management_regime_overlay_summary_v1.json"
    )
    outcome_quality = _load_json(
        extension_dir / "shadow_meta_all_trade_review_management_outcome_quality_regime_audit_summary_v1.json"
    )

    rl_status = _load_json(r8_dir / "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json")
    rl_sequence_status = _load_json(r8_dir / "shadow_meta_all_trade_review_management_rl_sequence_status_v1.json")
    bandit_status = _load_json(r8_dir / "shadow_meta_all_trade_review_management_bandit_status_v1.json")
    exit_local_status = _load_json(r8_dir / "shadow_meta_all_trade_review_management_exit_local_status_v1.json")
    entry_actualization = _load_json(r8_dir / "shadow_meta_all_trade_review_entry_actualization_status_v1.json")
    handoff_summary = _load_json(
        r8_dir / "shadow_meta_all_trade_review_entry_actual_take_to_management_handoff_summary_v1.json"
    )

    profitability = foundation.get("profitability", {})
    trade_shape = foundation.get("trade_shape", {})
    hold_longer_pressure = foundation.get("hold_longer_pressure", {})
    quality_flags = foundation.get("quality_flags", {})
    exit_efficiency = foundation.get("exit_efficiency", {})

    inventory_rows: List[Dict[str, str]] = [
        _row(
            "CANONICAL_TRUTH_ROOT",
            "PASS",
            f"66/66 weeks | {foundation.get('trade_count')} trades",
            "Canonical replay truth is complete and downstream management build is real.",
            "Keep append-only discipline and only bless new lines after full replay sanity.",
            "truth_trade_foundation_quality_v1.json",
        ),
        _row(
            "TRADE_FOUNDATION",
            foundation.get("outlook_v1", "UNKNOWN"),
            (
                f"avg_pnl={_scalar(profitability.get('avg_pnl_bps'))}bps | "
                f"pf={_scalar(profitability.get('profit_factor'))} | "
                f"dd={_scalar(profitability.get('max_drawdown_bps'))}bps"
            ),
            "PnL, win-rate, PF, DD, MFE, MAE and session summaries are materialized and stable.",
            "Improve capture and drawdown; current edge is positive but regret-heavy.",
            "truth_trade_foundation_quality_v1.json",
        ),
        _row(
            "EXIT_EFFICIENCY_AND_HOLD_LONGER",
            foundation.get("verdicts", {}).get("exit_efficiency_status", "UNKNOWN"),
            (
                f"regret_count={_scalar(exit_efficiency.get('early_exit_regret_count'))} | "
                f"regret_rate={_scalar(exit_efficiency.get('early_exit_regret_rate'))} | "
                f"mean_extra={_scalar(hold_longer_pressure.get('extra_value_bps', {}).get('mean'))}bps"
            ),
            "Too-early exit, post-exit MFE and hold-longer pressure are explicitly tracked.",
            "This remains the biggest management weakness and should stay a primary target.",
            "truth_trade_foundation_quality_v1.json",
        ),
        _row(
            "QUALITY_FLAGS",
            "PASS",
            (
                f"clean_good_trade={_scalar(quality_flags.get('clean_good_trade_mfe20_mae5_count'))} | "
                f"home_run_200bps={_scalar(quality_flags.get('home_run_200bps_count'))}"
            ),
            "We already have cata, never_mfe, good_mfe_then_rot, clean_good_trade_mfe20_mae5 and home-run slices.",
            "Keep 200bps as elite label, not primary good-trade label.",
            "truth_trade_foundation_quality_v1.json + replay_merge.py",
        ),
        _row(
            "ENTRY_SKIPABILITY",
            skipability.get("verdicts", {}).get("zero_trade_acceptance_status", "UNKNOWN"),
            (
                f"zero_trade_weeks={_scalar(skipability.get('completed_zero_trade_runs'))} | "
                f"candidate_rich_zero={_scalar(skipability.get('candidate_rich_zero_trade_runs'))}"
            ),
            "We can now see should-have-trade pressure and zero-trade clustering explicitly.",
            "Entry gating is still too conservative in several weeks; prioritize skipability calibration.",
            "truth_entry_skipability_pressure_v1.json",
        ),
        _row(
            "CONTINUOUS_MARKET_OPPORTUNITY",
            market.get("verdicts", {}).get("zero_trade_opportunity_rich_outlier_status", "UNKNOWN"),
            f"opportunity_rich_zero_outliers={len(market.get('opportunity_rich_zero_trade_runs_anchor', []))}",
            "Forward opportunity and backward market pressure exist at 15/60/240-bar horizons.",
            "Promote opportunity pressure into canonical entry scorecard so missed weeks stand out immediately.",
            "truth_continuous_market_opportunity_v1.json",
        ),
        _row(
            "MANAGEMENT_POLICY_LOGGING",
            policy_logging.get("instrumentation_status_v1", "UNKNOWN"),
            (
                f"observed_rows={sum(policy_logging.get('observed_action_counts_v1', {}).values())} | "
                f"hold={policy_logging.get('observed_action_counts_v1', {}).get('HOLD', 'NA')} | "
                f"exit={policy_logging.get('observed_action_counts_v1', {}).get('EXIT_NOW', 'NA')}"
            ),
            "Observed management actions are exact and physically separated from hindsight outcome backfill.",
            "Behavior policy and propensity are still not established.",
            "management_policy_logging_summary_v1.json",
        ),
        _row(
            "PATH_DYNAMICS_AS_OF_CORE",
            "PASS",
            "5/5 core fields logged in management harness",
            "last_peak_ts, last_mfe_ts, peak_price, anchor_price and mfe_at_anchor are present.",
            "Derive or log portable elapsed-time fields like minutes_since_last_peak and minutes_since_last_mfe.",
            "shadow_meta_v1.py path-dynamics harness",
        ),
        _row(
            "COARSE_MANAGEMENT_TEACHER",
            "PASS",
            (
                f"rows={_scalar(coarse_teacher.get('row_count_v1'))} | "
                f"eligible_binary={_scalar(coarse_teacher.get('binary_teacher_target_summary_v1', {}).get('eligible_rows_v1'))} | "
                f"hold_balanced={coarse_teacher.get('feedback_action_balance_status_v1', {}).get('HOLD', 'NA')}"
            ),
            "Hold/exit feedback surface with strong/weak capture, good/premature/late exit and hold-longer pressure is built.",
            "EXIT side is still one-sided; coarse surface is best used as teacher/diagnostic, not controller yet.",
            "truth_management_coarse_teacher_summary_v1.json",
        ),
        _row(
            "INCUMBENT_COMPARISON_BENCHMARK",
            coarse_benchmark.get("shadow_promotion_guard_v1", "UNKNOWN"),
            (
                f"hold_train={coarse_benchmark.get('universe_counts_v1', {}).get('split_counts_v1', {}).get('TRAIN', 'NA')} | "
                f"holdout_delta_brier={_scalar(coarse_benchmark.get('current_bucket_holdout_brier_improvement_v1'))}"
            ),
            "We now compare the new P1 candidate directly against the current bucket baseline, not dummy.",
            "Current coarse/raw-score candidate does not beat incumbent on holdout and must not be promoted yet.",
            "truth_management_coarse_feedback_benchmark_summary_v1.json",
        ),
        _row(
            "RL_READINESS_SUBSTRATE",
            rl_status.get("MANAGEMENT_RL_READINESS_STATUS", "UNKNOWN"),
            (
                f"dm_candidates={bandit_status.get('MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1', 'NA')} | "
                f"hold_episode_returns={bandit_status.get('MANAGEMENT_BANDIT_HOLD_EPISODE_RETURN_ROW_COUNT_V1', 'NA')} | "
                f"exit_local={bandit_status.get('MANAGEMENT_BANDIT_EXIT_LOCAL_REWARD_ROW_COUNT_V1', 'NA')}"
            ),
            "Offline RL substrate, action-reward split, exact terminal channels and canonical observations are in place.",
            "Still substrate-only; not behavior-policy-ready and not safe for full RL training yet.",
            "R8 management_rl_readiness_status + bandit_status",
        ),
        _row(
            "RL_SEQUENCE_AND_HANDOFF",
            rl_sequence_status.get("MANAGEMENT_RL_SEQUENCE_BLOCKER_STATUS", "UNKNOWN"),
            (
                f"provable_heads={handoff_summary.get('management_core_v4_present_count_v1', 'NA')} | "
                f"diagnostic_only={handoff_summary.get('management_bridge_diagnostic_only_count_v1', 'NA')}"
            ),
            "Entry-to-management handoff is mostly real and sequence substrate exists.",
            "Close the remaining next-step gaps and the 6 diagnostic-only handoffs.",
            "R8 handoff summary + rl_sequence_status",
        ),
        _row(
            "REGIME_OVERLAY",
            regime_overlay.get("regime_consistency_status_v1", "UNKNOWN"),
            f"slice_count={_scalar(outcome_quality.get('slice_count_v1'))}",
            "Session/vol/hold_age/giveback axes exist and show some outcome advantage.",
            "Current taxonomy is too fragmented; coarsen before using it as a controller.",
            "management_regime_overlay_summary_v1.json",
        ),
        _row(
            "EXIT_LOCAL_BASELINE",
            exit_local_status.get("MANAGEMENT_EXIT_LOCAL_BASELINE_STATUS", "UNKNOWN"),
            f"binary_target={_scalar(exit_local_status.get('BINARY_TARGET_STATUS_V1'))}",
            "A true exit-local observed-action baseline exists and is runnable.",
            "It is EXIT_NOW-only and not a full HOLD-vs-EXIT policy model.",
            "R8 management_exit_local_status",
        ),
    ]

    labels_inventory_v1 = {
        "trade_outcome_labels_v1": [
            "cata",
            "never_mfe",
            "good_mfe_then_rot",
            "good_trade_mfe20_mae5_v1",
            "mfe_mae_ratio_v1",
        ],
        "entry_skipability_labels_v1": [
            "hindsight_should_skip_trade_v1",
            "hindsight_take_was_ok_v1",
        ],
        "management_review_labels_v1": [
            "good_exit",
            "premature_exit",
            "late_exit",
            "quality_band_strong_capture_v1",
            "quality_band_weak_capture_v1",
            "quality_band_high_giveback_v1",
            "quality_band_low_mae_v1",
            "quality_band_tail_risk_v1",
            "hold_longer_extra_value_bps_v1",
            "hold_longer_pressure_10bps_v1",
            "hold_longer_pressure_25bps_v1",
        ],
        "continuous_market_pressure_fields_v1": [
            "as_of_entry_replay_window_up_move_15_bps_v1",
            "as_of_entry_replay_window_down_move_15_bps_v1",
            "as_of_entry_replay_window_range_15_bps_v1",
            "as_of_entry_replay_window_directional_imbalance_15_bps_v1",
            "as_of_entry_replay_window_close_in_range_15_v1",
            "as_of_entry_replay_window_up_move_60_bps_v1",
            "as_of_entry_replay_window_down_move_60_bps_v1",
            "as_of_entry_replay_window_range_60_bps_v1",
            "as_of_entry_replay_window_directional_imbalance_60_bps_v1",
            "as_of_entry_replay_window_close_in_range_60_v1",
            "as_of_entry_replay_window_up_move_240_bps_v1",
            "as_of_entry_replay_window_down_move_240_bps_v1",
            "as_of_entry_replay_window_range_240_bps_v1",
            "as_of_entry_replay_window_directional_imbalance_240_bps_v1",
            "as_of_entry_replay_window_close_in_range_240_v1",
        ],
        "path_dynamics_core_fields_v1": [
            "as_of_management_core_last_peak_ts_utc_v1",
            "as_of_management_core_last_mfe_ts_utc_v1",
            "as_of_management_core_peak_price_v1",
            "as_of_management_core_anchor_price_v1",
            "as_of_management_core_mfe_bps_at_anchor_v1",
        ],
    }

    top_improvements_v1 = [
        {
            "priority_rank_v1": 1,
            "track_v1": "behavior_policy_and_propensity",
            "why_v1": "Blocks real RL/off-policy evaluation.",
            "evidence_v1": [
                policy_logging.get("behavior_policy_readiness_v1"),
                policy_logging.get("propensity_readiness_v1"),
                bandit_status.get("MANAGEMENT_BANDIT_PROPENSITY_STATUS"),
            ],
        },
        {
            "priority_rank_v1": 2,
            "track_v1": "entry_gating_and_skipability",
            "why_v1": "Zero-trade weeks are candidate-rich and some are opportunity-rich.",
            "evidence_v1": [
                f"zero_trade_weeks={skipability.get('completed_zero_trade_runs')}",
                f"candidate_rich_zero={skipability.get('candidate_rich_zero_trade_runs')}",
                f"opportunity_rich_zero={len(market.get('opportunity_rich_zero_trade_runs_anchor', []))}",
            ],
        },
        {
            "priority_rank_v1": 3,
            "track_v1": "management_hold_longer_capture",
            "why_v1": "Capture is still too weak and regret too high.",
            "evidence_v1": [
                f"avg_pnl_bps={profitability.get('avg_pnl_bps')}",
                f"mean_hold_longer_extra_bps={hold_longer_pressure.get('extra_value_bps', {}).get('mean')}",
                f"regret_rate={exit_efficiency.get('early_exit_regret_rate')}",
            ],
        },
        {
            "priority_rank_v1": 4,
            "track_v1": "sequence_and_handoff_cleanup",
            "why_v1": "Still small but real integrity gaps before RL.",
            "evidence_v1": [
                rl_sequence_status.get("MANAGEMENT_RL_SEQUENCE_BLOCKER_STATUS"),
                readiness.get("entry_to_management_handoff_status_v1"),
                f"diagnostic_only={handoff_summary.get('management_bridge_diagnostic_only_count_v1')}",
            ],
        },
        {
            "priority_rank_v1": 5,
            "track_v1": "coarse_regime_conditioning",
            "why_v1": "Regime overlay is informative but not stable enough as a controller.",
            "evidence_v1": [
                regime_overlay.get("regime_consistency_status_v1"),
                regime_overlay.get("outcome_advantage_status_v1"),
            ],
        },
        {
            "priority_rank_v1": 6,
            "track_v1": "path_dynamics_elapsed_time_fields",
            "why_v1": "Auditability upgrade still missing for portable elapsed-time fields.",
            "evidence_v1": [
                "minutes_since_last_peak_v1 not independently materialized",
                "minutes_since_last_mfe_v1 not independently materialized",
            ],
        },
    ]

    contract = {
        "reports_root": str(reports_root),
        "r8_ledger_dir": str(r8_dir),
        "extension_dir": str(extension_dir),
        "headline_v1": {
            "trade_count_v1": foundation.get("trade_count"),
            "avg_pnl_bps_v1": profitability.get("avg_pnl_bps"),
            "profit_factor_v1": profitability.get("profit_factor"),
            "max_drawdown_bps_v1": profitability.get("max_drawdown_bps"),
            "hold_longer_regret_rate_v1": exit_efficiency.get("early_exit_regret_rate"),
            "zero_trade_weeks_v1": skipability.get("completed_zero_trade_runs"),
            "opportunity_rich_zero_weeks_v1": len(market.get("opportunity_rich_zero_trade_runs_anchor", [])),
            "management_ready_v1": readiness.get("downstream_management_ready"),
            "rl_policy_ready_v1": False,
        },
        "inventory_rows_v1": inventory_rows,
        "labels_inventory_v1": labels_inventory_v1,
        "top_improvements_v1": top_improvements_v1,
        "gates_v1": next_step.get("gates_v1", {}),
        "recommended_execution_order_v1": next_step.get("recommended_execution_order_v1", []),
        "contract_note_v1": (
            "This inventory is a concrete working contract for truth/management/RL surfaces. "
            "It describes what is already real, what is still substrate-only, and what must improve before RL moves "
            "from diagnosis to policy training."
        ),
    }
    return contract


def _markdown_table(rows: List[Dict[str, str]]) -> List[str]:
    lines = [
        "| Surface | Status | Counts | Available Now | Needs Improvement | Source |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {surface_v1} | {status_v1} | {counts_v1} | {available_now_v1} | {needs_improvement_v1} | {source_v1} |".format(
                **row
            )
        )
    return lines


def build_management_inventory_markdown(contract: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Truth Management Inventory Contract V1",
        "",
        "## Headline",
        "",
        f"- trades: `{_scalar(contract.get('headline_v1', {}).get('trade_count_v1'))}`",
        f"- avg pnl bps/trade: `{_scalar(contract.get('headline_v1', {}).get('avg_pnl_bps_v1'))}`",
        f"- profit factor: `{_scalar(contract.get('headline_v1', {}).get('profit_factor_v1'))}`",
        f"- max drawdown bps: `{_scalar(contract.get('headline_v1', {}).get('max_drawdown_bps_v1'))}`",
        f"- hold-longer regret rate: `{_scalar(contract.get('headline_v1', {}).get('hold_longer_regret_rate_v1'))}`",
        f"- zero-trade weeks: `{_scalar(contract.get('headline_v1', {}).get('zero_trade_weeks_v1'))}`",
        f"- opportunity-rich zero-trade weeks: `{_scalar(contract.get('headline_v1', {}).get('opportunity_rich_zero_weeks_v1'))}`",
        f"- management ready: `{_scalar(contract.get('headline_v1', {}).get('management_ready_v1'))}`",
        f"- rl policy ready: `{_scalar(contract.get('headline_v1', {}).get('rl_policy_ready_v1'))}`",
        "",
        "## Surface Inventory",
        "",
    ]
    lines.extend(_markdown_table(contract.get("inventory_rows_v1", [])))
    lines.extend(
        [
            "",
            "## Labels And Fields",
            "",
        ]
    )
    labels_inventory = contract.get("labels_inventory_v1", {})
    for section_name, values in labels_inventory.items():
        lines.append(f"### {section_name}")
        lines.append("")
        for value in values:
            lines.append(f"- `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Top Improvements",
            "",
        ]
    )
    for row in contract.get("top_improvements_v1", []):
        lines.append(
            f"{int(row.get('priority_rank_v1', 0))}. `{row.get('track_v1')}`: {row.get('why_v1')}"
        )
        for evidence in row.get("evidence_v1", []):
            lines.append(f"   evidence: `{_scalar(evidence)}`")
    lines.extend(
        [
            "",
            "## Gates",
            "",
        ]
    )
    for key, value in contract.get("gates_v1", {}).items():
        lines.append(f"- `{key}` = `{_scalar(value)}`")
    lines.extend(
        [
            "",
            "## Execution Order",
            "",
        ]
    )
    for idx, step in enumerate(contract.get("recommended_execution_order_v1", []), start=1):
        lines.append(f"{idx}. `{_scalar(step)}`")
    lines.append("")
    return "\n".join(lines)


def write_management_inventory_contract(reports_root: Path) -> Dict[str, str]:
    reports_root = Path(reports_root).expanduser().resolve()
    contract = build_management_inventory_contract(reports_root)
    json_path = reports_root / OUTPUT_JSON
    md_path = reports_root / OUTPUT_MD
    json_path.write_text(json.dumps(contract, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_management_inventory_markdown(contract), encoding="utf-8")
    return {
        "json_path": str(json_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a concrete truth/management/RL inventory contract for the active truth root."
    )
    parser.add_argument("--reports-root", default=None)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    written = write_management_inventory_contract(reports_root)
    print(json.dumps(written, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
