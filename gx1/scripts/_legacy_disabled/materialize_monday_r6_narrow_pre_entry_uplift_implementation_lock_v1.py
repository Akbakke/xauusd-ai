#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_PREFIX = "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_V1"
PRIOR_SPEC_PREFIX = "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_AND_RETRAIN_PREREQS_LOCK_V1_"
PRIOR_DIAG_PREFIX = "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_"

CANONICAL_LEDGER_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260411"

ENTRY_RAW_STATE = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
ENTRY_RAW_CONTRACT = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv"
ENTRY_RAW_CONTRACT_SUMMARY = "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json"
ENTRY_RL_VIEW = "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet"
ENTRY_RL_CONTRACT = "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json"
ENTRY_RL_SUMMARY = "shadow_meta_all_trade_review_entry_rl_observability_summary_v1.json"

PRIOR_ENTRY_LEGALITY = "entry_feature_legality_boundary_lock_v1.csv"
PRIOR_LEGAL_CANDIDATES = "legal_pre_entry_path_context_candidates_v1.csv"
PRIOR_PROTECTION_LOCK = "repaired_165_and_runner_pocket_protection_lock_v1.json"
PRIOR_RETRAIN_PREREQS = "retrain_prerequisites_lock_v1.json"
PRIOR_SUMMARY = "summary_v1.json"

PRIOR_REPAIRED_FORENSIC = "repaired_165_damage_forensic_v1.json"
PRIOR_GAP_MAP = "failure_backlog_gap_map_v1.csv"

CONTRACT = "contract_v1.json"
NARROW_PLAN = "narrow_feature_implementation_plan_v1.csv"
PROXY_CONTRACTS = "pre_entry_proxy_contracts_v1.csv"
RUNNER_GUARD_SPEC = "runner_protection_guard_spec_v1.json"
WIRING_PLAN = "feature_wiring_plan_v1.csv"
LEGALITY_TEST_PLAN = "leakage_and_legality_test_plan_v1.csv"
FAILURE_HOOKS = "failure_pocket_eval_hooks_v1.csv"
READINESS_GATE = "implementation_readiness_gate_v1.json"
POST_IMPLEMENTATION_PLAN = "post_implementation_not_retrain_yet_plan_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

SELECTED_CANDIDATES = [
    "pre_entry_volatility_expansion_compression_stack_v1",
    "pre_entry_directional_asymmetry_proxy_v1",
    "pre_entry_swing_retracement_alignment_v1",
    "pre_entry_tail_leakage_pocket_proxy_v1",
    "runner_protection_guard_score_v1",
]

DEFERRED_CANDIDATES = [
    "pre_entry_session_pocket_runner_expectancy_v1",
    "pre_entry_adverse_first_risk_proxy_v1",
    "spread_cost_pressure_hardening_v1",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    matches = sorted([path for path in reports_root.iterdir() if path.is_dir() and path.name.startswith(prefix)], key=lambda path: path.name)
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_extension_dir(reports_root: Path, extension_dir_arg: str | None) -> Path:
    if extension_dir_arg:
        return Path(extension_dir_arg).expanduser().resolve()
    return reports_root / f"{EXTENSION_PREFIX}_{_utc_compact()}"


def _resolve_ledger_dir(reports_root: Path) -> Path:
    canonical = reports_root / CANONICAL_LEDGER_DIRNAME
    if canonical.exists():
        return canonical
    matches = sorted(
        [
            path
            for path in reports_root.iterdir()
            if path.is_dir()
            and path.name.startswith("ALL_TRADE_REVIEW_LEDGER_")
            and (path / ENTRY_RAW_STATE).exists()
            and (path / ENTRY_RL_VIEW).exists()
        ],
        key=lambda path: (len(path.name), path.name),
    )
    if not matches:
        raise FileNotFoundError(f"Could not find canonical ledger dir under {reports_root}")
    return matches[0]


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _load_inputs(reports_root: Path) -> Dict[str, Any]:
    prior_spec_dir = _latest_dir(reports_root, PRIOR_SPEC_PREFIX)
    prior_diag_dir = _latest_dir(reports_root, PRIOR_DIAG_PREFIX)
    ledger_dir = _resolve_ledger_dir(reports_root)
    prior_legality_df = pd.read_csv(prior_spec_dir / PRIOR_ENTRY_LEGALITY)
    prior_candidates_df = pd.read_csv(prior_spec_dir / PRIOR_LEGAL_CANDIDATES)
    prior_protection = _load_json(prior_spec_dir / PRIOR_PROTECTION_LOCK)
    prior_prereqs = _load_json(prior_spec_dir / PRIOR_RETRAIN_PREREQS)
    prior_summary = _load_json(prior_spec_dir / PRIOR_SUMMARY)
    repaired_forensic = _load_json(prior_diag_dir / PRIOR_REPAIRED_FORENSIC)
    gap_map_df = pd.read_csv(prior_diag_dir / PRIOR_GAP_MAP)
    raw_cols = pd.read_parquet(ledger_dir / ENTRY_RAW_STATE, columns=["candidate_uid"]).columns.tolist()
    raw_cols = pd.read_parquet(ledger_dir / ENTRY_RAW_STATE).columns.tolist()
    entry_view_cols = pd.read_parquet(ledger_dir / ENTRY_RL_VIEW, columns=["candidate_uid"]).columns.tolist()
    entry_view_cols = pd.read_parquet(ledger_dir / ENTRY_RL_VIEW).columns.tolist()
    return {
        "prior_spec_dir": prior_spec_dir,
        "prior_diag_dir": prior_diag_dir,
        "ledger_dir": ledger_dir,
        "prior_legality_df": prior_legality_df,
        "prior_candidates_df": prior_candidates_df,
        "prior_protection": prior_protection,
        "prior_prereqs": prior_prereqs,
        "prior_summary": prior_summary,
        "repaired_forensic": repaired_forensic,
        "gap_map_df": gap_map_df,
        "raw_cols": set(raw_cols),
        "entry_view_cols": set(entry_view_cols),
    }


def _candidate_map(prior_candidates_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prior_candidates_df.to_dict(orient="records"):
        out[str(row["candidate_name_v1"])] = row
    return out


def _narrow_plan(candidate_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {
            "implementation_decision_v1": "SELECT_NOW",
            "candidate_name_v1": "pre_entry_volatility_expansion_compression_stack_v1",
            "why_now_v1": "Best low-leakage first slice for separating 10-50 tail pocket, should-not-take, and noisy risky allows using already-materialized legal replay context.",
            "primary_bucket_v1": "missed 10-50 tail-control",
            "secondary_buckets_v1": "missed should-not-take,risky allows",
            "legality_status_v1": candidate_map["pre_entry_volatility_expansion_compression_stack_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["pre_entry_volatility_expansion_compression_stack_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "gx1/analysis/shadow_meta_v1.py::_build_entry_replay_bar_state_expansion_and_reprobe",
            "dependencies_v1": "existing as_of_skip_replay compression/volatility window fields only",
            "main_risk_v1": "If overcollapsed into one aggressive score, it can become hidden blocker hardening without protector balancing.",
            "why_prioritized_over_others_v1": "Uses existing legal fields with low complexity and directly addresses the tail-pocket diagnosis.",
        },
        {
            "implementation_decision_v1": "SELECT_NOW",
            "candidate_name_v1": "pre_entry_directional_asymmetry_proxy_v1",
            "why_now_v1": "Gives the cleanest legal prior-path proxy for distinguishing noisy trap from continuation seed without touching same-trade path truth.",
            "primary_bucket_v1": "runner near-misses",
            "secondary_buckets_v1": "missed should-not-take,risky allows",
            "legality_status_v1": candidate_map["pre_entry_directional_asymmetry_proxy_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["pre_entry_directional_asymmetry_proxy_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "gx1/analysis/shadow_meta_v1.py::_build_entry_replay_bar_state_expansion_and_reprobe",
            "dependencies_v1": "existing as_of_skip_replay window up/down move, imbalance, close-in-range, micro-momentum fields",
            "main_risk_v1": "Directional imbalance can become brittle if not bounded across 15/60/240 windows.",
            "why_prioritized_over_others_v1": "Most direct legal translation of path-style intuition for runner-vs-trap separation.",
        },
        {
            "implementation_decision_v1": "SELECT_NOW",
            "candidate_name_v1": "pre_entry_swing_retracement_alignment_v1",
            "why_now_v1": "Most targeted feature for repaired-165 and moderate-MAE runner-seed protection using existing swing/retracement context.",
            "primary_bucket_v1": "repaired-165 protection",
            "secondary_buckets_v1": "runner near-misses,missed should-not-take",
            "legality_status_v1": candidate_map["pre_entry_swing_retracement_alignment_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["pre_entry_swing_retracement_alignment_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "gx1/analysis/shadow_meta_v1.py::_build_entry_replay_bar_state_expansion_and_reprobe",
            "dependencies_v1": "existing as_of_skip_replay swing distance, bars-since-swing, retracement, EMA distance fields",
            "main_risk_v1": "Can become regime-specific if not normalized and clipped carefully.",
            "why_prioritized_over_others_v1": "Directly targets the repaired-pocket failure that broke the Monday R6 freeze path.",
        },
        {
            "implementation_decision_v1": "SELECT_NOW",
            "candidate_name_v1": "pre_entry_tail_leakage_pocket_proxy_v1",
            "why_now_v1": "Provides a narrow tail-pocket score instead of trying to solve 10-50 leakage indirectly through broader blocker expansion.",
            "primary_bucket_v1": "missed 10-50 tail-control",
            "secondary_buckets_v1": "runner near-misses",
            "legality_status_v1": candidate_map["pre_entry_tail_leakage_pocket_proxy_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["pre_entry_tail_leakage_pocket_proxy_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "gx1/analysis/shadow_meta_v1.py::_build_entry_replay_bar_state_expansion_and_reprobe",
            "dependencies_v1": "selected volatility/directional/swing proxies plus existing bar geometry and session-boundary fields",
            "main_risk_v1": "If derived too close to hindsight tail labels, it becomes illegal; derivation must stay entirely pre-entry.",
            "why_prioritized_over_others_v1": "Directly addresses the 198 missed tail-control cases without opening broad new model complexity.",
        },
        {
            "implementation_decision_v1": "SELECT_NOW",
            "candidate_name_v1": "runner_protection_guard_score_v1",
            "why_now_v1": "Needed to stop repaired-165 and runner near-miss damage before any future blocker-side uplift is allowed.",
            "primary_bucket_v1": "repaired-165 protection",
            "secondary_buckets_v1": "runner near-misses,50+ MFE protection",
            "legality_status_v1": candidate_map["runner_protection_guard_score_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["runner_protection_guard_score_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "gx1/analysis/shadow_meta_v1.py::_build_entry_replay_bar_state_expansion_and_reprobe; future eval consumption only, not live policy",
            "dependencies_v1": "selected legal proxies plus existing candidate snapshot pre-entry fields",
            "main_risk_v1": "Can become a disguised hindsight protector if management/exit truth leaks into inputs.",
            "why_prioritized_over_others_v1": "Protection must be built before any new retrain, otherwise the same repaired-pocket mistake can recur.",
        },
        {
            "implementation_decision_v1": "DEFER_NOT_NOW",
            "candidate_name_v1": "pre_entry_session_pocket_runner_expectancy_v1",
            "why_now_v1": "Wait until the first narrow proxies are in place; session pocket alone is too easy to overfit as a standalone signal.",
            "primary_bucket_v1": "runner near-misses",
            "secondary_buckets_v1": "tail-control assistance",
            "legality_status_v1": candidate_map["pre_entry_session_pocket_runner_expectancy_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["pre_entry_session_pocket_runner_expectancy_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "later wave only",
            "dependencies_v1": "needs stability check after first narrow proxies",
            "main_risk_v1": "Pocket-specific expectancy can become unstable across slices.",
            "why_prioritized_over_others_v1": "Deferred because it is useful, but less foundational than the first four chosen proxies.",
        },
        {
            "implementation_decision_v1": "DEFER_NOT_NOW",
            "candidate_name_v1": "pre_entry_adverse_first_risk_proxy_v1",
            "why_now_v1": "Do not add blocker-side aggression before runner-protection and repaired-pocket guards are implemented and verified.",
            "primary_bucket_v1": "risky allows",
            "secondary_buckets_v1": "missed should-not-take",
            "legality_status_v1": candidate_map["pre_entry_adverse_first_risk_proxy_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["pre_entry_adverse_first_risk_proxy_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "later wave only",
            "dependencies_v1": "requires protector-first wiring and legal test lock already green",
            "main_risk_v1": "Most likely candidate to improve recall while damaging runners if added too early.",
            "why_prioritized_over_others_v1": "Deferred intentionally because the current diagnosis says protection-first, not blocker-first.",
        },
        {
            "implementation_decision_v1": "DEFER_NOT_NOW",
            "candidate_name_v1": "spread_cost_pressure_hardening_v1",
            "why_now_v1": "Useful hardening, but lower value than direct runner/tail/separation proxies.",
            "primary_bucket_v1": "risky allows",
            "secondary_buckets_v1": "missed should-not-take",
            "legality_status_v1": candidate_map["spread_cost_pressure_hardening_v1"]["legality_v1"],
            "expected_value_v1": candidate_map["spread_cost_pressure_hardening_v1"]["expected_value_v1"],
            "implementation_codepaths_v1": "later wave only",
            "dependencies_v1": "none beyond existing raw-state",
            "main_risk_v1": "Can consume time without solving the core repaired-runner weakness.",
            "why_prioritized_over_others_v1": "Deferred because it does not directly answer the repaired-165 / runner near-miss diagnosis.",
        },
    ]
    return pd.DataFrame(rows)


def _proxy_contracts() -> pd.DataFrame:
    rows = [
        {
            "feature_name_v1": "as_of_pre_entry_vol_exp_comp_score_v1",
            "source_candidate_v1": "pre_entry_volatility_expansion_compression_stack_v1",
            "precise_definition_v1": "A clipped composite score from pre-entry compression, range expansion, and realized-volatility context at the exact entry anchor.",
            "input_sources_v1": _json_dumps(
                {
                    "artifact": ENTRY_RAW_STATE,
                    "columns": [
                        "as_of_skip_replay_h1_range_compression_ratio_v1",
                        "as_of_skip_replay_m15_range_compression_ratio_v1",
                        "as_of_skip_replay_bb_squeeze_20_2_v1",
                        "as_of_skip_replay_bb_bandwidth_delta_10_v1",
                        "as_of_skip_replay_window_range_ratio_mean_5_v1",
                        "as_of_skip_replay_window_realized_vol_3_bps_v1",
                        "as_of_skip_replay_window_realized_vol_5_bps_v1",
                        "as_of_skip_replay_d1_atr_percentile_252_v1",
                    ],
                }
            ),
            "legal_input_v1": "Only as_of_skip_replay_* anchor-bar and prior-window fields from entry raw-state.",
            "illegal_input_v1": "Any last_peak/last_mfe/max_mfe_without_mae/sequence-order field; any policy-log score; any realized pnl/MFE/MAE/giveback field.",
            "temporal_legality_v1": "All inputs must be known at or before the entry anchor bar; no same-trade future path allowed.",
            "null_default_handling_v1": "If fewer than 5 legal source fields are non-null, emit null score and explicit availability flag false.",
            "expected_range_v1": "Finite clipped float in [-10, 10] after normalization; null only when contract says unavailable.",
            "leakage_guard_v1": "Source columns must come only from ENTRY_SKIPABILITY_RAW_STATE replay/candidate families.",
            "testable_invariants_v1": "Changing downstream management/exit fields must not change the score; all-missing sources must yield null+unavailable.",
        },
        {
            "feature_name_v1": "as_of_pre_entry_directional_asymmetry_score_v1",
            "source_candidate_v1": "pre_entry_directional_asymmetry_proxy_v1",
            "precise_definition_v1": "A pre-entry path-style score summarizing directional imbalance, move asymmetry, close-in-range, and short-horizon momentum before entry.",
            "input_sources_v1": _json_dumps(
                {
                    "artifact": ENTRY_RAW_STATE,
                    "columns": [
                        "as_of_skip_replay_window_up_move_15_bps_v1",
                        "as_of_skip_replay_window_up_move_60_bps_v1",
                        "as_of_skip_replay_window_up_move_240_bps_v1",
                        "as_of_skip_replay_window_down_move_15_bps_v1",
                        "as_of_skip_replay_window_down_move_60_bps_v1",
                        "as_of_skip_replay_window_down_move_240_bps_v1",
                        "as_of_skip_replay_window_directional_imbalance_15_bps_v1",
                        "as_of_skip_replay_window_directional_imbalance_60_bps_v1",
                        "as_of_skip_replay_window_directional_imbalance_240_bps_v1",
                        "as_of_skip_replay_window_close_in_range_15_v1",
                        "as_of_skip_replay_window_close_in_range_60_v1",
                        "as_of_skip_replay_window_close_in_range_240_v1",
                        "as_of_skip_replay_micro_momentum_3_v1",
                        "as_of_skip_replay_micro_momentum_5_v1",
                        "as_of_skip_replay_micro_acceleration_v1",
                    ],
                }
            ),
            "legal_input_v1": "Only entry-anchor replay-window fields and momentum fields already materialized in raw-state.",
            "illegal_input_v1": "Any future path label, management/exit anchor field, tail label, or decision-log score.",
            "temporal_legality_v1": "Rolling windows must terminate at the entry anchor and never cross into post-entry bars.",
            "null_default_handling_v1": "If no valid horizon is available, emit null score and availability false; partial horizons are allowed with explicit count check.",
            "expected_range_v1": "Finite clipped float in [-10, 10]; monotone with stronger directional asymmetry after normalization.",
            "leakage_guard_v1": "No column outside as_of_skip_replay_* may enter except deterministic availability metadata.",
            "testable_invariants_v1": "Future-only changes must not affect the score; horizon subsets must stay finite and reproducible.",
        },
        {
            "feature_name_v1": "as_of_pre_entry_swing_retracement_alignment_score_v1",
            "source_candidate_v1": "pre_entry_swing_retracement_alignment_v1",
            "precise_definition_v1": "A structural alignment score describing whether the setup is continuation-friendly or exhaustion-like relative to recent swings and retracement state.",
            "input_sources_v1": _json_dumps(
                {
                    "artifact": ENTRY_RAW_STATE,
                    "columns": [
                        "as_of_skip_replay_dist_last_swing_high_atr_v1",
                        "as_of_skip_replay_dist_last_swing_low_atr_v1",
                        "as_of_skip_replay_bars_since_swing_high_v1",
                        "as_of_skip_replay_bars_since_swing_low_v1",
                        "as_of_skip_replay_retracement_from_last_impulse_v1",
                        "as_of_skip_replay_distance_ema_fast_v1",
                        "as_of_skip_replay_d1_dist_from_ema200_atr_v1",
                    ],
                }
            ),
            "legal_input_v1": "Only swing/structure fields known at entry anchor.",
            "illegal_input_v1": "Any post-entry swing resolution, management trace, or realized winner label.",
            "temporal_legality_v1": "All structure fields must originate from replay state at the exact anchor timestamp.",
            "null_default_handling_v1": "If core swing distance and retracement signals are missing, emit null+unavailable; otherwise allow partial composition with bounded output.",
            "expected_range_v1": "Finite clipped float in [-10, 10].",
            "leakage_guard_v1": "Disallow any field whose lineage is management/exit or hindsight review.",
            "testable_invariants_v1": "Output must be deterministic from current swing state only; repaired/trade outcome tags must not affect it.",
        },
        {
            "feature_name_v1": "as_of_pre_entry_tail_leakage_pocket_score_v1",
            "source_candidate_v1": "pre_entry_tail_leakage_pocket_proxy_v1",
            "precise_definition_v1": "A legal pre-entry risk score for the 10-50 MFE leakage pocket built from already-legal proxy scores plus anchor-bar geometry and session-boundary context.",
            "input_sources_v1": _json_dumps(
                {
                    "derived_dependencies": [
                        "as_of_pre_entry_vol_exp_comp_score_v1",
                        "as_of_pre_entry_directional_asymmetry_score_v1",
                        "as_of_pre_entry_swing_retracement_alignment_score_v1",
                    ],
                    "raw_artifact": ENTRY_RAW_STATE,
                    "raw_columns": [
                        "as_of_skip_replay_close_in_bar_v1",
                        "as_of_skip_replay_body_share_v1",
                        "as_of_skip_replay_upper_wick_share_v1",
                        "as_of_skip_replay_lower_wick_share_v1",
                        "as_of_skip_replay_minutes_to_next_session_boundary_v1",
                        "as_of_skip_replay_session_change_flag_v1",
                        "as_of_skip_replay_spread_bps_v1",
                    ],
                }
            ),
            "legal_input_v1": "Only legal derived proxies and raw entry-anchor geometry/session fields.",
            "illegal_input_v1": "Any realized tail label, peak_mfe, giveback, exit timing, or management policy output.",
            "temporal_legality_v1": "All inputs must be frozen at entry time; no same-trade tail truth may enter.",
            "null_default_handling_v1": "If fewer than 2 core proxy dependencies exist, emit null+unavailable instead of inventing a fallback score.",
            "expected_range_v1": "Finite clipped float in [0, 1] interpreted as pre-entry tail-pocket risk.",
            "leakage_guard_v1": "Guard explicitly bans all hindsight tail labels and management path fields.",
            "testable_invariants_v1": "Replacing same-trade outcome labels must leave the score unchanged; dependency availability must be explicit.",
        },
        {
            "feature_name_v1": "as_of_pre_entry_runner_protection_guard_score_v1",
            "source_candidate_v1": "runner_protection_guard_score_v1",
            "precise_definition_v1": "A pre-entry protection score that estimates whether a setup belongs to a legitimate runner pocket that should resist aggressive blocker expansion in shadow evaluation.",
            "input_sources_v1": _json_dumps(
                {
                    "derived_dependencies": [
                        "as_of_pre_entry_vol_exp_comp_score_v1",
                        "as_of_pre_entry_directional_asymmetry_score_v1",
                        "as_of_pre_entry_swing_retracement_alignment_score_v1",
                        "as_of_pre_entry_tail_leakage_pocket_score_v1",
                    ],
                    "raw_artifact": ENTRY_RAW_STATE,
                    "raw_columns": [
                        "as_of_skip_candidate_p_hat_v1",
                        "as_of_skip_candidate_margin_v1",
                        "as_of_skip_candidate_path_quality_pred_v1",
                        "as_of_skip_replay_spread_bps_v1",
                    ],
                }
            ),
            "legal_input_v1": "Only selected legal proxies and candidate snapshot fields available at decision time.",
            "illegal_input_v1": "Management policy scores, decision-log fields, exit truth, repaired tags as model inputs, and any realized trade-quality/hindsight label.",
            "temporal_legality_v1": "Must be computable at entry anchor from raw-state and candidate snapshot only.",
            "null_default_handling_v1": "Emit null+unavailable if fewer than 3 selected proxy inputs exist; never coalesce from hindsight sources.",
            "expected_range_v1": "Finite clipped float in [0, 1], higher means stronger runner-protection need.",
            "leakage_guard_v1": "Explicit deny-list for management/exit-anchor fields and hindsight labels; repaired pocket stays eval-only.",
            "testable_invariants_v1": "Forensic trade and repaired pocket must be evaluable through this score without any management-truth dependency.",
        },
    ]
    return pd.DataFrame(rows)


def _runner_guard_spec(repaired_forensic: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "layer_name_v1": "RUNNER_PROTECTION_GUARD_SPEC_V1",
        "research_shadow_only_v1": True,
        "not_live_guard_v1": True,
        "guard_score_name_v1": "as_of_pre_entry_runner_protection_guard_score_v1",
        "guard_intent_v1": "Express pre-entry probability-like need to protect a setup from over-aggressive blocker expansion because it resembles a legitimate runner seed.",
        "legal_signal_inputs_v1": [
            "as_of_pre_entry_vol_exp_comp_score_v1",
            "as_of_pre_entry_directional_asymmetry_score_v1",
            "as_of_pre_entry_swing_retracement_alignment_score_v1",
            "as_of_pre_entry_tail_leakage_pocket_score_v1",
            "as_of_skip_candidate_p_hat_v1",
            "as_of_skip_candidate_margin_v1",
            "as_of_skip_candidate_path_quality_pred_v1",
            "as_of_skip_replay_spread_bps_v1",
        ],
        "protected_failure_pockets_v1": [
            "repaired_165_pocket",
            "forensic_trade_TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
            "runner_near_miss_pocket",
            "50_plus_mfe_runner_pocket",
        ],
        "interaction_rules_v1": [
            "The guard is evaluated before any future blocker-side recall expansion.",
            "High guard score must suppress newly aggressive block decisions unless bad-risk is extreme under the future eval contract.",
            "Tail-risk scores may not override the guard on repaired-165-like or strongest-winner protection cases without explicit audit failure.",
            "The guard remains shadow/research only until a later readiness recheck approves model consumption.",
        ],
        "how_it_avoids_repaired_165_repeat_v1": [
            "It explicitly treats repaired-pocket-like structure/continuation context as something to protect, not just something to ignore.",
            "It forbids using management/exit truth as a shortcut; only pre-entry proxies are allowed.",
            "It requires forensic trade coverage in every next eval contract.",
        ],
        "explicit_monitor_cases_v1": [
            "repaired_165_damage == 0",
            "forensic trade remains unblocked",
            "50+ MFE blocked <= 1",
            "100+ MFE blocked == 0",
            "200+ MFE blocked == 0",
            "strongest_winner_path_damage == 0",
        ],
    }


def _feature_wiring_plan() -> pd.DataFrame:
    rows = [
        {
            "object_v1": "selected_pre_entry_proxies",
            "implementation_scope_v1": "RAW_STATE_AND_AUDIT_ONLY_NOW",
            "code_files_v1": "gx1/analysis/shadow_meta_v1.py",
            "functions_or_contracts_v1": "_build_entry_replay_bar_state_expansion_and_reprobe; _ALL_TRADE_REVIEW_ENTRY_SKIPABILITY_RAW_STATE_V1 contract rows",
            "tests_to_add_v1": "tests/test_entry_pre_entry_proxy_legality_v1.py; tests/test_entry_pre_entry_proxy_contracts_v1.py",
            "artifacts_to_extend_v1": f"{ENTRY_RAW_STATE}; {ENTRY_RAW_CONTRACT}; {ENTRY_RAW_CONTRACT_SUMMARY}",
            "also_in_audit_surface_v1": True,
            "why_scope_is_narrow_v1": "Only extend canonical entry raw-state and its contract summary; do not touch training views or policy stacks yet.",
        },
        {
            "object_v1": "runner_protection_guard_score_v1",
            "implementation_scope_v1": "RAW_STATE_AND_EVAL_HOOKS_ONLY_NOW",
            "code_files_v1": "gx1/analysis/shadow_meta_v1.py; future readiness/eval script only",
            "functions_or_contracts_v1": "entry raw-state derivation helper; new audit hook read-model, not live policy",
            "tests_to_add_v1": "tests/test_entry_runner_protection_guard_legality_v1.py",
            "artifacts_to_extend_v1": f"{ENTRY_RAW_STATE}; {ENTRY_RAW_CONTRACT}; {ENTRY_RAW_CONTRACT_SUMMARY}",
            "also_in_audit_surface_v1": True,
            "why_scope_is_narrow_v1": "Guard is materialized as research signal and audit field only; not wired into controller or retrain yet.",
        },
        {
            "object_v1": "future_model_consumption_placeholder",
            "implementation_scope_v1": "DO_NOT_ACTIVATE_NOW",
            "code_files_v1": "gx1/scripts/train_r6_entry_runner_first_retrain_v1.py",
            "functions_or_contracts_v1": "_feature_names; future AS_OF feature table wiring after readiness recheck",
            "tests_to_add_v1": "none in this phase",
            "artifacts_to_extend_v1": "none in this phase",
            "also_in_audit_surface_v1": False,
            "why_scope_is_narrow_v1": "Explicitly postponed until legality/coverage/readiness recheck completes.",
        },
        {
            "object_v1": "entry_rl_observability_view",
            "implementation_scope_v1": "NO_SCHEMA_CHANGE_UNLESS_STRICTLY_NEEDED",
            "code_files_v1": "gx1/analysis/shadow_meta_v1.py",
            "functions_or_contracts_v1": f"{ENTRY_RL_VIEW}; {ENTRY_RL_CONTRACT}; {ENTRY_RL_SUMMARY}",
            "tests_to_add_v1": "only if a new legal proxy truly needs core-view fields",
            "artifacts_to_extend_v1": "defer by default",
            "also_in_audit_surface_v1": False,
            "why_scope_is_narrow_v1": "Avoid broad schema spread; raw-state is enough for the first uplift slice.",
        },
    ]
    return pd.DataFrame(rows)


def _legality_test_plan() -> pd.DataFrame:
    rows = [
        {
            "test_id_v1": "NEGATIVE_DIRECT_MANAGEMENT_EXIT_FIELDS_REJECTED",
            "level_v1": "UNIT_NEGATIVE",
            "target_v1": "all selected proxies",
            "what_it_proves_v1": "Direct management/exit-anchor truth cannot be registered as legal proxy inputs.",
            "expected_result_v1": "Hard fail if any deny-listed field appears in proxy input schema.",
        },
        {
            "test_id_v1": "POSITIVE_VOL_EXP_COMP_LEGAL_INPUTS_ONLY",
            "level_v1": "UNIT_CONTRACT",
            "target_v1": "as_of_pre_entry_vol_exp_comp_score_v1",
            "what_it_proves_v1": "Only raw-state anchor-bar and prior-window replay fields are used.",
            "expected_result_v1": "Pass only when all inputs belong to approved as_of_skip_replay families.",
        },
        {
            "test_id_v1": "POSITIVE_DIRECTIONAL_ASYMMETRY_WINDOW_BOUNDARY",
            "level_v1": "UNIT_PROPERTY",
            "target_v1": "as_of_pre_entry_directional_asymmetry_score_v1",
            "what_it_proves_v1": "Directional windows terminate at the anchor and do not cross into post-entry bars.",
            "expected_result_v1": "Property holds for synthetic anchor/future perturbation fixture.",
        },
        {
            "test_id_v1": "POSITIVE_SWING_ALIGNMENT_NULL_HANDLING",
            "level_v1": "UNIT_CONTRACT",
            "target_v1": "as_of_pre_entry_swing_retracement_alignment_score_v1",
            "what_it_proves_v1": "Missing swing inputs produce explicit null+availability=false instead of fallback fabrication.",
            "expected_result_v1": "Pass on partial-null and all-null fixtures.",
        },
        {
            "test_id_v1": "NEGATIVE_TAIL_PROXY_CANNOT_READ_HINDSIGHT_TAIL_LABELS",
            "level_v1": "UNIT_NEGATIVE",
            "target_v1": "as_of_pre_entry_tail_leakage_pocket_score_v1",
            "what_it_proves_v1": "Tail proxy does not read realized peak/giveback/tail labels.",
            "expected_result_v1": "Hard fail if any hindsight tail field is referenced.",
        },
        {
            "test_id_v1": "NEGATIVE_RUNNER_GUARD_CANNOT_READ_POLICY_LOG_OR_EXIT_TRUTH",
            "level_v1": "UNIT_NEGATIVE",
            "target_v1": "as_of_pre_entry_runner_protection_guard_score_v1",
            "what_it_proves_v1": "Runner guard stays pre-entry legal and ignores management/exit scores and decision-log fields.",
            "expected_result_v1": "Hard fail on any policy-log or exit-truth lineage in inputs.",
        },
        {
            "test_id_v1": "SCHEMA_RAW_STATE_CONTRACT_EXTENDS_WITH_AS_OF_ONLY_FIELDS",
            "level_v1": "CONTRACT_SCHEMA",
            "target_v1": "entry raw-state contract",
            "what_it_proves_v1": "New proxy fields are documented with AS_OF-only semantics and no hindsight claims.",
            "expected_result_v1": "Contract csv/json summary shows legal source lineage and null semantics.",
        },
        {
            "test_id_v1": "COVERAGE_NULL_DEFAULT_SANITY",
            "level_v1": "READONLY_AUDIT",
            "target_v1": "all selected proxies",
            "what_it_proves_v1": "Availability flags and null rules behave deterministically on real ledger rows.",
            "expected_result_v1": "No hidden default fills from forbidden sources.",
        },
    ]
    return pd.DataFrame(rows)


def _failure_hooks(repaired_forensic: Dict[str, Any]) -> pd.DataFrame:
    trade_key = str(repaired_forensic["deterministic_trade_key_v1"])
    rows = [
        {
            "pocket_id_v1": "REPAIRED_165_POCKET",
            "tracking_filter_v1": "is_repaired_165_v1 == true",
            "not_worse_than_requirement_v1": "blocked_count must remain 0",
            "uplift_signals_expected_v1": "swing_alignment + runner_guard",
            "regression_guard_v1": "hard fail if repaired_165_damage > 0",
        },
        {
            "pocket_id_v1": "FORENSIC_REPAIRED_TRADE",
            "tracking_filter_v1": f"candidate_uid == {trade_key}",
            "not_worse_than_requirement_v1": "must remain explicitly unblocked; guard score must be materialized",
            "uplift_signals_expected_v1": "runner_guard + swing_alignment",
            "regression_guard_v1": "hard fail if trade is blocked or lacks proxy coverage",
        },
        {
            "pocket_id_v1": "RUNNER_NEAR_MISS",
            "tracking_filter_v1": "r6_label_runner_near_miss_v1 == true or backlog bucket RUNNER_NEAR_MISS",
            "not_worse_than_requirement_v1": "count and protection quality must be no worse than Monday R6 failure-miner baseline",
            "uplift_signals_expected_v1": "directional_asymmetry + runner_guard",
            "regression_guard_v1": "watch 50+/100+/200+ blocked counts separately",
        },
        {
            "pocket_id_v1": "MISSED_10_50_TAIL_CONTROL",
            "tracking_filter_v1": "tail_10_50_mfe_v1 == true or backlog bucket MISSED_10_50_TAIL_CONTROL",
            "not_worse_than_requirement_v1": "tail proxy coverage must be present before any retrain is reopened",
            "uplift_signals_expected_v1": "vol_exp_comp + tail_leakage_pocket",
            "regression_guard_v1": "true runner pockets must not worsen while tail-control improves",
        },
        {
            "pocket_id_v1": "MISSED_SHOULD_NOT_TAKE",
            "tracking_filter_v1": "label_should_not_take_v1 == true and currently unblocked",
            "not_worse_than_requirement_v1": "no new blocker work allowed unless runner protection remains green",
            "uplift_signals_expected_v1": "vol_exp_comp + directional_asymmetry",
            "regression_guard_v1": "do not reopen blocker expansion until repaired pocket remains safe",
        },
        {
            "pocket_id_v1": "RISKY_ALLOW",
            "tracking_filter_v1": "r6_label_risky_allow_v1 == true or backlog bucket RISKY_ALLOW",
            "not_worse_than_requirement_v1": "monitor only in this phase; do not optimize before guard-first uplift is verified",
            "uplift_signals_expected_v1": "future adverse_first proxy later; not part of first implementation slice",
            "regression_guard_v1": "no blocker-side aggression increase in this phase",
        },
    ]
    return pd.DataFrame(rows)


def _implementation_readiness_gate() -> Dict[str, Any]:
    return {
        "layer_name_v1": "IMPLEMENTATION_READINESS_GATE_V1",
        "decision_v1": "READY_TO_IMPLEMENT_NARROW_FEATURES",
        "retrain_now_v1": False,
        "why_ready_v1": [
            "Selected proxies are few, high-value, and tied directly to diagnosed pockets.",
            "Proxy contracts are explicit enough to implement without legal ambiguity.",
            "Legality/leakage tests are defined before any model work resumes.",
            "Wiring is narrow and stays inside canonical entry raw-state plus audit surfaces.",
        ],
        "what_is_not_ready_v1": [
            "No retrain is open yet.",
            "No blocker-side feature expansion beyond the selected narrow slice is approved.",
            "Management/exit-anchor fields remain banned for entry use.",
        ],
    }


def _post_implementation_plan() -> Dict[str, Any]:
    return {
        "layer_name_v1": "POST_IMPLEMENTATION_NOT_RETRAIN_YET_PLAN_V1",
        "steps_v1": [
            "1. Run code/test verification for new proxy builders and guard logic.",
            "2. Recheck raw-state schema and contract summaries for the added AS_OF-only fields.",
            "3. Run legality/leakage recheck against explicit deny-lists and future-perturbation tests.",
            "4. Run feature coverage and null-rate audit over the full canonical Monday ledger.",
            "5. Run failure-pocket coverage audit for repaired-165, forensic trade, runner near-misses, and 10-50 tail pocket.",
            "6. Materialize a new readiness recheck package deciding whether narrow retrain can open.",
            "7. Only then evaluate whether a narrow retrain phase is justified.",
        ],
        "hard_rule_v1": "Do not jump directly from implementation to retrain.",
    }


def _next_action_lock() -> Dict[str, Any]:
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "IMPLEMENT_SELECTED_LEGAL_PROXIES_NOW",
        "supporting_actions_v1": [
            "IMPLEMENT_RUNNER_PROTECTION_GUARD_NOW",
            "LOCK_LEGALITY_TESTS_BEFORE_ANY_MODEL_WORK",
            "DO_NOT_RETRAIN_YET",
            "ONLY_AFTER_IMPLEMENTATION_RUN_READINESS_RECHECK",
            "KEEP_MONDAY_R6_AS_FAILURE_MINER",
        ],
    }


def _status_block() -> Dict[str, Any]:
    return {
        "layer_name_v1": "STATUS_DISCIPLINE_V1",
        "BEVIST": [
            "The next phase should implement a few legal pre-entry proxies and a narrow runner-protection guard, not reopen retrain.",
            "Selected proxies can be built from existing entry raw-state and candidate snapshot fields without direct management/exit truth.",
            "Management/exit-anchor fields remain forbidden as direct entry inputs.",
            "Repaired-165 and runner pockets are explicit hard eval gates for the next phase.",
        ],
        "INDIKERT": [
            "This narrow proxy set is the best first slice for improving runner-protection and pocket separation without uncontrolled complexity.",
            "Guard-first uplift is more promising than blocker-first expansion.",
            "A raw-state-first implementation path is safer than widening the whole entry schema immediately.",
        ],
        "IKKE_ETABLERT": [
            "That this narrow uplift alone will beat frozen Wednesday R6.",
            "That no second iteration will be needed after the first implementation slice.",
            "That the deferred adverse-first and session-pocket candidates will remain unnecessary later.",
        ],
    }


def _render_report(
    plan_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    readiness_gate: Dict[str, Any],
    next_action: Dict[str, Any],
    status_block: Dict[str, Any],
) -> str:
    lines = [
        "# Monday R6 Narrow Pre-Entry Uplift Implementation Lock V1",
        "",
        "Read-only implementation spec. No retrain or replay was started.",
        "",
        "## Headline",
        "",
        f"- Implementation readiness: `{readiness_gate['decision_v1']}`",
        f"- Primary next action: `{next_action['primary_action_v1']}`",
        "",
        "## Selected Now",
        "",
    ]
    for row in plan_df[plan_df["implementation_decision_v1"].astype("string").eq("SELECT_NOW")].to_dict(orient="records"):
        lines.append(f"- `{row['candidate_name_v1']}`: {row['why_now_v1']}")
    lines += [
        "",
        "## Deferred",
        "",
    ]
    for row in plan_df[plan_df["implementation_decision_v1"].astype("string").eq("DEFER_NOT_NOW")].to_dict(orient="records"):
        lines.append(f"- `{row['candidate_name_v1']}`: {row['why_now_v1']}")
    lines += [
        "",
        "## Proxy Contracts",
        "",
    ]
    for row in proxy_df.to_dict(orient="records"):
        lines.append(f"- `{row['feature_name_v1']}`: {row['precise_definition_v1']}")
    lines += [
        "",
        "## Hard Status",
        "",
    ]
    for key in ["BEVIST", "INDIKERT", "IKKE_ETABLERT"]:
        lines.append(f"### {key}")
        lines.append("")
        for item in status_block[key]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_payload(reports_root: Path, extension_dir: Path) -> Dict[str, Any]:
    inputs = _load_inputs(reports_root)
    candidate_map = _candidate_map(inputs["prior_candidates_df"])
    plan_df = _narrow_plan(candidate_map)
    proxy_df = _proxy_contracts()
    guard_spec = _runner_guard_spec(inputs["repaired_forensic"])
    wiring_df = _feature_wiring_plan()
    legality_test_df = _legality_test_plan()
    failure_hooks_df = _failure_hooks(inputs["repaired_forensic"])
    readiness_gate = _implementation_readiness_gate()
    post_plan = _post_implementation_plan()
    next_action = _next_action_lock()
    status_block = _status_block()

    required_raw_cols = {
        "as_of_skip_replay_h1_range_compression_ratio_v1",
        "as_of_skip_replay_m15_range_compression_ratio_v1",
        "as_of_skip_replay_bb_squeeze_20_2_v1",
        "as_of_skip_replay_bb_bandwidth_delta_10_v1",
        "as_of_skip_replay_window_range_ratio_mean_5_v1",
        "as_of_skip_replay_window_realized_vol_3_bps_v1",
        "as_of_skip_replay_window_realized_vol_5_bps_v1",
        "as_of_skip_replay_d1_atr_percentile_252_v1",
        "as_of_skip_replay_window_up_move_15_bps_v1",
        "as_of_skip_replay_window_up_move_60_bps_v1",
        "as_of_skip_replay_window_up_move_240_bps_v1",
        "as_of_skip_replay_window_down_move_15_bps_v1",
        "as_of_skip_replay_window_down_move_60_bps_v1",
        "as_of_skip_replay_window_down_move_240_bps_v1",
        "as_of_skip_replay_window_directional_imbalance_15_bps_v1",
        "as_of_skip_replay_window_directional_imbalance_60_bps_v1",
        "as_of_skip_replay_window_directional_imbalance_240_bps_v1",
        "as_of_skip_replay_window_close_in_range_15_v1",
        "as_of_skip_replay_window_close_in_range_60_v1",
        "as_of_skip_replay_window_close_in_range_240_v1",
        "as_of_skip_replay_micro_momentum_3_v1",
        "as_of_skip_replay_micro_momentum_5_v1",
        "as_of_skip_replay_micro_acceleration_v1",
        "as_of_skip_replay_dist_last_swing_high_atr_v1",
        "as_of_skip_replay_dist_last_swing_low_atr_v1",
        "as_of_skip_replay_bars_since_swing_high_v1",
        "as_of_skip_replay_bars_since_swing_low_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        "as_of_skip_replay_distance_ema_fast_v1",
        "as_of_skip_replay_d1_dist_from_ema200_atr_v1",
        "as_of_skip_replay_close_in_bar_v1",
        "as_of_skip_replay_body_share_v1",
        "as_of_skip_replay_upper_wick_share_v1",
        "as_of_skip_replay_lower_wick_share_v1",
        "as_of_skip_replay_minutes_to_next_session_boundary_v1",
        "as_of_skip_replay_session_change_flag_v1",
        "as_of_skip_replay_spread_bps_v1",
        "as_of_skip_candidate_p_hat_v1",
        "as_of_skip_candidate_margin_v1",
        "as_of_skip_candidate_path_quality_pred_v1",
    }
    deny_list = {
        "last_peak_ts",
        "last_mfe_ts",
        "last_peak_mfe",
        "max_mfe_without_mae",
        "mfe_mae_sequence_order",
        "management_policy_scores_or_decision_log_fields",
    }
    legality_lookup = inputs["prior_legality_df"].set_index("feature_or_family_v1")["classification_v1"].astype("string").to_dict()
    consistency_df = pd.DataFrame(
        [
            _audit_record("PRIOR_SPEC_PRESENT", "PASS", {"dir": str(inputs["prior_spec_dir"])}),
            _audit_record("PRIOR_DIAG_PRESENT", "PASS", {"dir": str(inputs["prior_diag_dir"])}),
            _audit_record("RAW_STATE_ARTIFACT_PRESENT", "PASS", {"path": str(inputs["ledger_dir"] / ENTRY_RAW_STATE)}),
            _audit_record("ENTRY_VIEW_ARTIFACT_PRESENT", "PASS", {"path": str(inputs["ledger_dir"] / ENTRY_RL_VIEW)}),
            _audit_record(
                "SELECTED_CANDIDATES_EXIST_AND_HIGH_PRIORITY",
                "PASS" if all(candidate in candidate_map and str(candidate_map[candidate]["priority_v1"]) == "HIGH" for candidate in SELECTED_CANDIDATES) else "FAIL",
                {"selected": SELECTED_CANDIDATES},
            ),
            _audit_record(
                "DEFERRED_CANDIDATES_EXIST",
                "PASS" if all(candidate in candidate_map for candidate in DEFERRED_CANDIDATES) else "FAIL",
                {"deferred": DEFERRED_CANDIDATES},
            ),
            _audit_record(
                "SELECTED_INPUT_COLUMNS_PRESENT_IN_RAW_STATE",
                "PASS" if required_raw_cols.issubset(inputs["raw_cols"]) else "FAIL",
                {"missing": sorted(required_raw_cols - inputs["raw_cols"])},
            ),
            _audit_record(
                "DENY_LIST_STILL_NOT_ENTRY_LEGAL",
                "PASS" if all(str(legality_lookup.get(field)) == "NOT_LEGAL_FOR_ENTRY" for field in deny_list) else "FAIL",
                {"checked_fields": sorted(deny_list)},
            ),
            _audit_record(
                "RETRAIN_STILL_CLOSED",
                "PASS" if inputs["prior_prereqs"]["retrain_now_v1"] is False else "FAIL",
                {"decision": inputs["prior_prereqs"]["decision_v1"]},
            ),
            _audit_record(
                "FORENSIC_TRADE_LOCK_PRESENT",
                "PASS" if "d2e2d6b7fb03" in str(inputs["repaired_forensic"]["deterministic_trade_key_v1"]) else "FAIL",
                {"trade_key": inputs["repaired_forensic"]["deterministic_trade_key_v1"]},
            ),
        ]
    )

    contract = {
        "layer_name_v1": "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_CONTRACT_V1",
        "mode_v1": "READ_ONLY_IMPLEMENTATION_SPEC_ONLY",
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_promotion_v1": True,
        "inputs_v1": {
            "prior_spec_dir_v1": str(inputs["prior_spec_dir"]),
            "prior_diag_dir_v1": str(inputs["prior_diag_dir"]),
            "ledger_dir_v1": str(inputs["ledger_dir"]),
            "frozen_benchmark_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "monday_safety_reference_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
            "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        },
    }

    status = {
        "layer_name_v1": "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_STATUS_V1",
        "SPEC_STATUS": "MATERIALIZED_READ_ONLY",
        "failed_check_count_v1": int(consistency_df["status_v1"].eq("FAIL").sum()),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_promotion_v1": True,
    }
    summary = {
        "layer_name_v1": "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "extension_dir_v1": str(extension_dir),
        "benchmark_lock_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
        "safety_reference_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
        "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        "selected_candidates_v1": SELECTED_CANDIDATES,
        "deferred_candidates_v1": DEFERRED_CANDIDATES,
        "implementation_readiness_v1": readiness_gate["decision_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "status_v1": status,
        "hard_status_division_v1": status_block,
    }
    manifest = {
        "layer_name_v1": "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "narrow_feature_implementation_plan": NARROW_PLAN,
            "pre_entry_proxy_contracts": PROXY_CONTRACTS,
            "runner_protection_guard_spec": RUNNER_GUARD_SPEC,
            "feature_wiring_plan": WIRING_PLAN,
            "leakage_and_legality_test_plan": LEGALITY_TEST_PLAN,
            "failure_pocket_eval_hooks": FAILURE_HOOKS,
            "implementation_readiness_gate": READINESS_GATE,
            "post_implementation_not_retrain_yet_plan": POST_IMPLEMENTATION_PLAN,
            "next_agent_action_lock": NEXT_ACTION,
            "summary": SUMMARY,
            "report": REPORT,
            "manifest": MANIFEST,
            "status": STATUS,
            "consistency_audit": CONSISTENCY_AUDIT,
        }
    }
    report = _render_report(plan_df, proxy_df, readiness_gate, next_action, status_block)
    return {
        "contract": contract,
        "plan_df": plan_df,
        "proxy_df": proxy_df,
        "guard_spec": guard_spec,
        "wiring_df": wiring_df,
        "legality_test_df": legality_test_df,
        "failure_hooks_df": failure_hooks_df,
        "readiness_gate": readiness_gate,
        "post_plan": post_plan,
        "next_action": next_action,
        "summary": summary,
        "manifest": manifest,
        "status": status,
        "consistency_df": consistency_df,
        "report": report,
    }


def materialize(reports_root: Path, *, extension_dir: Path | None = None) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    extension_dir = _resolve_extension_dir(reports_root, str(extension_dir) if extension_dir else None)
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(reports_root, extension_dir)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    payload["plan_df"].to_csv(extension_dir / NARROW_PLAN, index=False)
    payload["proxy_df"].to_csv(extension_dir / PROXY_CONTRACTS, index=False)
    _write_json(extension_dir / RUNNER_GUARD_SPEC, payload["guard_spec"])
    payload["wiring_df"].to_csv(extension_dir / WIRING_PLAN, index=False)
    payload["legality_test_df"].to_csv(extension_dir / LEGALITY_TEST_PLAN, index=False)
    payload["failure_hooks_df"].to_csv(extension_dir / FAILURE_HOOKS, index=False)
    _write_json(extension_dir / READINESS_GATE, payload["readiness_gate"])
    _write_json(extension_dir / POST_IMPLEMENTATION_PLAN, payload["post_plan"])
    _write_json(extension_dir / NEXT_ACTION, payload["next_action"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    _write_json(extension_dir / STATUS, payload["status"])
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    return {"extension_dir": str(extension_dir), "status": payload["status"], "summary": payload["summary"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a narrow legal pre-entry uplift implementation lock after Monday-native R6 diagnosis.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--extension-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(reports_root, extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
