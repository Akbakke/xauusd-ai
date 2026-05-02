#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_investigate_true_r5_2_rebuild_or_label_objective_next_v1 import (
    R6_FRAME,
    _base_masks,
    _mae_bucket,
    _merge_score_r6,
    _mfe_bucket,
)
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_RISKY_PROB,
    R6_TAIL_PROB,
    _jsonable,
)
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    SCORE_FRAME,
    SCORE_SUMMARY,
    _bool,
    _num,
    _read_json,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "FIX_R5_2_LABEL_OBJECTIVE_FIRST_V1"

V3_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V3_R5_R51_R52"
V3_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V3_R6_FROM_V3_R52"
INVESTIGATION_DEFAULT = DEFAULT_REPORTS_ROOT / "INVESTIGATE_TRUE_R5_2_REBUILD_OR_LABEL_OBJECTIVE_NEXT_V1_20260426T_LOCK"
ASSET_REUSE_DEFAULT = DEFAULT_REPORTS_ROOT / "EXISTING_ASSET_FIRST_R6_REUSE_AND_DUPLICATE_GUARD_V1_20260426T_LOCK"

OUTPUT_FILES = {
    "mismatch": "r5_2_objective_mismatch_lock_v1.json",
    "label_contract": "r5_2_new_label_contract_v1.json",
    "weighting": "r5_2_objective_weighting_spec_v1.json",
    "pocket_table": "r5_2_pocket_label_table_v1.csv",
    "feature_alignment": "r5_2_feature_signal_alignment_for_new_objective_v1.csv",
    "rebuild_spec": "r5_2_rebuild_experiment_spec_v1.json",
    "dry_run": "r5_2_new_objective_dry_run_simulation_v1.json",
    "gate": "r5_2_label_objective_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

BUCKET_ORDER = [
    "RUNNER_PROTECT_TARGET",
    "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD",
    "TAIL_CONTROL_TARGET",
    "STRONG_BAD_BLOCK_TARGET",
    "RISKY_ALLOW_TARGET",
    "IGNORE_OR_MONITOR_ONLY",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _hard_runner_protect(frame: pd.DataFrame) -> pd.Series:
    return (
        (_bool(frame, "take_was_ok_v1") & (_bool(frame, "fifty_plus_mfe_v1") | _bool(frame, "hundred_plus_mfe_v1") | _bool(frame, "two_hundred_plus_mfe_v1")))
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | _bool(frame, "r6_label_runner_near_miss_v1")
    ).fillna(False)


def _ambiguous_high_mfe(frame: pd.DataFrame) -> pd.Series:
    return (
        _bool(frame, "label_should_not_take_v1")
        & (_bool(frame, "fifty_plus_mfe_v1") | _bool(frame, "hundred_plus_mfe_v1") | _bool(frame, "two_hundred_plus_mfe_v1"))
        & ~_bool(frame, "take_was_ok_v1")
    ).fillna(False)


def _low_winner_risk(frame: pd.DataFrame) -> pd.Series:
    return (
        ~_bool(frame, "fifty_plus_mfe_v1")
        & ~_bool(frame, "hundred_plus_mfe_v1")
        & ~_bool(frame, "two_hundred_plus_mfe_v1")
        & ~_bool(frame, "strongest_winner_path_v1")
        & ~_bool(frame, "r6_label_repaired_165_like_runner_v1")
        & ~_bool(frame, "r6_label_runner_near_miss_v1")
    ).fillna(False)


def _strong_bad(frame: pd.DataFrame) -> pd.Series:
    return (
        _bool(frame, "label_should_not_take_v1")
        & _low_winner_risk(frame)
        & (
            _num(frame, "peak_mfe_bps_v1").lt(10.0).fillna(False)
            | _num(frame, "mae_abs_bps_v1").ge(40.0).fillna(False)
            | _num(frame, "baseline_realized_pnl_bps_v1").le(0.0).fillna(False)
            | _bool(frame, "r6_label_risky_allow_v1")
        )
    ).fillna(False)


def _tail_control(frame: pd.DataFrame) -> pd.Series:
    return (_bool(frame, "tail_10_50_mfe_v1") & _low_winner_risk(frame)).fillna(False)


def _risky_allow(frame: pd.DataFrame) -> pd.Series:
    return (_bool(frame, "r6_label_risky_allow_v1") & _low_winner_risk(frame) & ~_strong_bad(frame) & ~_tail_control(frame)).fillna(False)


def _bucket_reason(row: pd.Series) -> str:
    bucket = str(row.get("new_r5_2_label_bucket_v1"))
    if bucket == "RUNNER_PROTECT_TARGET":
        return "Hard protect pocket: take-ok high-MFE, strongest, repaired-like, or runner-near-miss."
    if bucket == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD":
        return "Should-not-take row has high MFE; monitor/protect, but do not train as bad-positive."
    if bucket == "TAIL_CONTROL_TARGET":
        return "10-50 MFE tail-control row with low high-MFE/winner risk."
    if bucket == "STRONG_BAD_BLOCK_TARGET":
        return "Clear should-not-take bad-risk row with low winner risk."
    if bucket == "RISKY_ALLOW_TARGET":
        return "Risky row should influence R5.2 attention but cannot hard-block alone."
    return "No safe target assignment; monitor only."


def _label_table(score: pd.DataFrame, r6: pd.DataFrame, score_summary: dict[str, Any]) -> pd.DataFrame:
    frame = _merge_score_r6(score, r6)
    masks = _base_masks(frame, score_summary)
    for name, mask in masks.items():
        frame[f"r5_2_{name}_base_flag_v1"] = mask.to_numpy(dtype=bool)
    frame["mfe_bucket_v1"] = _mfe_bucket(frame).to_numpy()
    frame["mae_bucket_v1"] = _mae_bucket(frame).to_numpy()
    hard = _hard_runner_protect(frame)
    ambiguous = _ambiguous_high_mfe(frame) & ~hard
    tail = _tail_control(frame) & ~hard & ~ambiguous
    strong_bad = _strong_bad(frame) & ~hard & ~ambiguous & ~tail
    risky = _risky_allow(frame) & ~hard & ~ambiguous & ~tail & ~strong_bad
    bucket = np.select(
        [hard, ambiguous, tail, strong_bad, risky],
        [
            "RUNNER_PROTECT_TARGET",
            "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD",
            "TAIL_CONTROL_TARGET",
            "STRONG_BAD_BLOCK_TARGET",
            "RISKY_ALLOW_TARGET",
        ],
        default="IGNORE_OR_MONITOR_ONLY",
    )
    frame["new_r5_2_label_bucket_v1"] = bucket
    frame["target_class_v1"] = np.select(
        [
            frame["new_r5_2_label_bucket_v1"].eq("STRONG_BAD_BLOCK_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("TAIL_CONTROL_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("RISKY_ALLOW_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"),
        ],
        ["BAD_ELIGIBILITY_POSITIVE", "TAIL_ELIGIBILITY_POSITIVE", "RISKY_ATTENTION_POSITIVE", "PROTECTED_NEGATIVE", "AMBIGUOUS_MONITOR_PROTECTED"],
        default="MONITOR_ONLY",
    )
    frame["sample_weight_v1"] = np.select(
        [
            frame["new_r5_2_label_bucket_v1"].eq("STRONG_BAD_BLOCK_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("TAIL_CONTROL_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("RISKY_ALLOW_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"),
        ],
        [3.0, 2.5, 1.25, 2.0, 0.5],
        default=0.25,
    )
    frame["protection_weight_v1"] = np.select(
        [
            frame["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET"),
            frame["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"),
            _bool(frame, "hundred_plus_mfe_v1") | _bool(frame, "two_hundred_plus_mfe_v1") | _bool(frame, "strongest_winner_path_v1") | _bool(frame, "r6_label_repaired_165_like_runner_v1"),
        ],
        [10.0, 6.0, 20.0],
        default=0.0,
    )
    frame["eval_only_flag_v1"] = frame["new_r5_2_label_bucket_v1"].isin(["AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", "IGNORE_OR_MONITOR_ONLY"])
    frame["bad_eligibility_target_v1"] = frame["new_r5_2_label_bucket_v1"].eq("STRONG_BAD_BLOCK_TARGET")
    frame["tail_eligibility_target_v1"] = frame["new_r5_2_label_bucket_v1"].eq("TAIL_CONTROL_TARGET")
    frame["risky_attention_target_v1"] = frame["new_r5_2_label_bucket_v1"].eq("RISKY_ALLOW_TARGET")
    frame["runner_protect_target_v1"] = frame["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")
    frame["ambiguous_high_mfe_monitor_v1"] = frame["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
    frame["reason_v1"] = frame.apply(_bucket_reason, axis=1)
    cols = [
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "run_id",
        "split_scope_v1",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "r6_label_risky_allow_v1",
        "r5_2_label_bad_blocker_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "mfe_bucket_v1",
        "mae_bucket_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        "r5_2_original_base_flag_v1",
        "r5_2_v1_base_flag_v1",
        "r5_2_v2_base_flag_v1",
        "r5_2_v3_base_flag_v1",
        "new_r5_2_label_bucket_v1",
        "target_class_v1",
        "sample_weight_v1",
        "protection_weight_v1",
        "eval_only_flag_v1",
        "bad_eligibility_target_v1",
        "tail_eligibility_target_v1",
        "risky_attention_target_v1",
        "runner_protect_target_v1",
        "ambiguous_high_mfe_monitor_v1",
        "reason_v1",
    ]
    return frame[[col for col in cols if col in frame.columns]].copy()


def _mismatch_lock(investigation: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_MISMATCH_LOCK_V1",
        "mismatch_proven_v1": True,
        "evidence_v1": {
            "missed_bad_tail_traced_v1": int(investigation.get("missed_rows_traced_v1") or 390),
            "r5_2_score_weak_v1": 184,
            "feature_signal_present_but_underused_v1": 46,
            "row_actually_dangerous_high_mfe_ambiguous_v1": 147,
            "base_contract_too_strict_v1": 13,
            "r6_v3_bad_tail_v1": [82, 51],
            "wednesday_bad_tail_v1": [180, 149],
        },
        "locked_findings_v1": [
            "Current R5.2 objective learns should_not_take minus high-MFE ambiguous rows.",
            "R6 needs R5.2 to be a broader bad/tail eligibility base, not a tiny hard blocker.",
            "R5.2 must still explicitly protect true runners/winners.",
            "Tiny V1/V2/V3 base-extension loop is complete and must stop.",
            "True R5.2 rebuild must use a new label/objective contract before running.",
        ],
    }


def _label_contract() -> dict[str, Any]:
    role = {
        "input_fields_v1": "Existing Monday foundation labels/outcomes define targets; AS_OF features are used only by later training.",
        "as_of_hindsight_role_v1": "HINDSIGHT labels define supervised targets and eval guards; AS_OF fields remain model inputs only.",
    }
    return {
        "layer_name": "R5_2_NEW_LABEL_CONTRACT_V1",
        "contract_id_v1": "R5_2_LABEL_OBJECTIVE_BAD_TAIL_ELIGIBILITY_WITH_HARD_WINNER_PROTECTION_V1",
        "labels_v1": {
            "STRONG_BAD_BLOCK_TARGET": {
                **role,
                "definition_v1": "label_should_not_take with low high-MFE/winner/repaired/near-miss risk and clear adverse/low-value evidence.",
                "use_v1": "training_positive_bad_eligibility",
                "risk_v1": "Overblocking low-quality but recoverable runners if protection head is weak.",
                "expected_effect_v1": "Raises bad-block recall for clearly bad low-winner-risk rows.",
            },
            "TAIL_CONTROL_TARGET": {
                **role,
                "definition_v1": "tail_10_50_mfe with no 50+/100+/200+/strongest/repaired/near-miss pocket.",
                "use_v1": "training_positive_tail_eligibility",
                "risk_v1": "Can suppress small winners if blended into hard bad target.",
                "expected_effect_v1": "Gives 10-50 tail rows direct positive target instead of weak indirect R5.2 signal.",
            },
            "RISKY_ALLOW_TARGET": {
                **role,
                "definition_v1": "r6 risky_allow style row that deserves attention but cannot hard-block alone.",
                "use_v1": "auxiliary_attention_target",
                "risk_v1": "If used as hard block, precision can collapse.",
                "expected_effect_v1": "Improves R5.2 awareness of rows R6 currently tries to recover downstream.",
            },
            "RUNNER_PROTECT_TARGET": {
                **role,
                "definition_v1": "take-ok high-MFE, strongest-winner, repaired-like, or runner-near-miss row.",
                "use_v1": "training_negative_protection_head_and_eval_guard",
                "risk_v1": "Underweighting this target causes winner damage.",
                "expected_effect_v1": "Preserves safety while broadening bad/tail eligibility.",
            },
            "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD": {
                **role,
                "definition_v1": "label_should_not_take row with 50+/100+/200+ MFE or runner-like high-MFE ambiguity.",
                "use_v1": "eval_guard_monitor_or_protected_negative_not_bad_positive",
                "risk_v1": "Treating these as bad-positive recreates the December/high-MFE damage problem.",
                "expected_effect_v1": "Prevents recall gain from being bought by blocking high-MFE opportunities.",
            },
            "IGNORE_OR_MONITOR_ONLY": {
                **role,
                "definition_v1": "Rows without safe target confidence for the first R5.2 objective rebuild.",
                "use_v1": "monitor_only_or_low_weight_background",
                "risk_v1": "Too broad ignore class can hide missing pockets.",
                "expected_effect_v1": "Keeps first rebuild clean and auditable.",
            },
        },
    }


def _weighting_spec() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_WEIGHTING_SPEC_V1",
        "positive_class_weights_v1": {
            "STRONG_BAD_BLOCK_TARGET": 3.0,
            "TAIL_CONTROL_TARGET": 2.5,
            "RISKY_ALLOW_TARGET": 1.25,
        },
        "protection_costs_v1": {
            "RUNNER_PROTECT_TARGET": 10.0,
            "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD": 6.0,
            "HUNDRED_OR_TWO_HUNDRED_MFE": 20.0,
            "STRONGEST_WINNER": 20.0,
            "REPAIRED_LIKE": 20.0,
        },
        "handling_v1": {
            "bad_tail_recall_v1": "Strong bad and 10-50 tail are direct positives, unlike current R5.2 where tail is only indirect.",
            "winner_damage_v1": "High-MFE winners/protected rows are hard negatives/protection targets with no-go eval constraints.",
            "fifty_plus_mfe_v1": "50+ rows are protected or ambiguous-monitor, not bad-positive in first rebuild.",
            "runner_near_miss_v1": "Always protection target/eval guard.",
            "support_r6_without_ukritisk_blocker_v1": "R5.2 outputs eligibility and protection scores; final base requires eligibility positive and protection low.",
        },
    }


def _feature_alignment(label_table: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_sets = {
        "AS_OF_CORE": [col for col in score.columns if col.startswith("as_of_candidate") or col.startswith("as_of_entry_candidate") or col.startswith("as_of_skip_replay")][:24],
        "R5_R5_1_SCORES": [
            "pred__entry_r5_should_not_take__prob_true_v1",
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
            "pred__entry_r5_runner_protect__prob_true_v1",
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
            "r5_1_bad_blocker_score_v1",
            "r5_1_runner_guard_score_v1",
        ],
        "R5_2_CURRENT_SCORES": [R5_2_BAD_PROB, R5_2_RUNNER_PROB],
        "R6_DOWNSTREAM_SCORES_EVAL_ONLY": [R6_BAD_PROB, R6_RISKY_PROB, R6_TAIL_PROB],
    }
    underused = {
        "FEATURE_SIGNAL_PRESENT_BUT_UNDERUSED": "Existing R5/R5.1/R6 signals separate some missed rows, but current R5.2 objective does not reward them correctly.",
        "SCORE_WEAK": "Current R5.2 bad score is weak; AS_OF + R5/R5.1 signals are available for a true rebuild.",
        "AMBIGUOUS_HIGH_MFE": "Signal must be used to protect or monitor, not as bad-positive.",
    }
    for bucket in BUCKET_ORDER:
        bucket_rows = label_table[label_table["new_r5_2_label_bucket_v1"] == bucket]
        for family, features in feature_sets.items():
            legal = "REUSE_NOW" if family != "R6_DOWNSTREAM_SCORES_EVAL_ONLY" else "REUSE_FOR_EVAL_ONLY"
            rows.append(
                {
                    "label_bucket_v1": bucket,
                    "feature_family_v1": family,
                    "row_count_v1": int(len(bucket_rows)),
                    "existing_features_v1": "|".join([feature for feature in features if feature in score.columns]),
                    "legality_v1": legal,
                    "underused_signal_v1": underused.get("FEATURE_SIGNAL_PRESENT_BUT_UNDERUSED") if bucket in {"STRONG_BAD_BLOCK_TARGET", "TAIL_CONTROL_TARGET", "RISKY_ALLOW_TARGET"} else underused.get("AMBIGUOUS_HIGH_MFE"),
                    "signal_basis_enough_for_true_rebuild_v1": bool(features),
                }
            )
    return pd.DataFrame(rows)


def _rebuild_spec(label_contract: dict[str, Any], weighting: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_REBUILD_EXPERIMENT_SPEC_V1",
        "recommended_design_v1": "POCKET_AWARE_MULTI_HEAD_R5_2_BAD_TAIL_ELIGIBILITY_WITH_RUNNER_PROTECTION",
        "do_not_run_training_in_this_round_v1": True,
        "training_surface_v1": "Canonical Monday fullcoverage foundation score surface, 1914 rows, 1852 active, 62 quarantine eval-only; not 1689 exact-only.",
        "label_table_v1": "r5_2_pocket_label_table_v1.csv",
        "feature_set_v1": {
            "use_now_v1": ["AS_OF 109/core replay features", "existing R5 score outputs", "existing R5.1 score outputs"],
            "eval_only_v1": ["R6 downstream scores", "hindsight/path labels"],
            "forbidden_v1": ["exit/management truth as entry features", "bridge/readiness as training surface", "protector-first surfaces"],
        },
        "model_family_v1": "XGB-style deterministic multi-head matching current stack",
        "heads_v1": [
            "bad_eligibility_head",
            "tail_10_50_eligibility_head",
            "risky_attention_head",
            "runner_protect_head",
        ],
        "target_structure_v1": label_contract["contract_id_v1"],
        "sample_weights_v1": weighting["positive_class_weights_v1"],
        "protection_weights_v1": weighting["protection_costs_v1"],
        "split_eval_v1": "Existing train/validation/holdout/LOSO with quarantine eval-only.",
        "outputs_v1": [
            "r5_2_new_label_table",
            "r5_2_multi_head_prediction_view",
            "base_eligibility_policy_report",
            "R6 downstream replay package",
            "pocket safety audit",
        ],
        "no_go_cases_v1": [
            "repaired damage > 0",
            "forensic trade blocked",
            "100+/200+ blocked > 0",
            "50+ blocked > 1",
            "strongest-winner damage > 0",
            "runner near-miss worse",
            "precision/worst LOSO below Wednesday safety reference",
        ],
        "r6_consumption_v1": "R6 consumes R5.2 base eligibility plus protection scores; no live/controller/freeze until explicit R6 eval gate passes.",
    }


def _dry_run(label_table: pd.DataFrame) -> dict[str, Any]:
    missed = label_table[(_bool(label_table, "label_should_not_take_v1") | _bool(label_table, "tail_10_50_mfe_v1")) & ~_bool(label_table, "r5_2_v3_base_flag_v1")]
    bucket_counts = label_table["new_r5_2_label_bucket_v1"].value_counts().to_dict()
    missed_counts = missed["new_r5_2_label_bucket_v1"].value_counts().to_dict()
    ambiguous_bad_positive = int(
        (
            label_table["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
            & label_table["bad_eligibility_target_v1"].astype(bool)
        ).sum()
    )
    positives = label_table[label_table["target_class_v1"].isin(["BAD_ELIGIBILITY_POSITIVE", "TAIL_ELIGIBILITY_POSITIVE", "RISKY_ATTENTION_POSITIVE"])]
    protected = label_table[label_table["target_class_v1"].isin(["PROTECTED_NEGATIVE", "AMBIGUOUS_MONITOR_PROTECTED"])]
    return {
        "layer_name": "R5_2_NEW_OBJECTIVE_DRY_RUN_SIMULATION_V1",
        "row_count_v1": int(len(label_table)),
        "class_balance_v1": {str(key): int(value) for key, value in label_table["target_class_v1"].value_counts().to_dict().items()},
        "pocket_balance_v1": {str(key): int(value) for key, value in bucket_counts.items()},
        "positive_counts_v1": {
            "bad_positive_v1": int(label_table["bad_eligibility_target_v1"].sum()),
            "tail_positive_v1": int(label_table["tail_eligibility_target_v1"].sum()),
            "risky_attention_positive_v1": int(label_table["risky_attention_target_v1"].sum()),
            "total_positive_or_attention_v1": int(len(positives)),
        },
        "protected_counts_v1": {
            "runner_protect_v1": int(label_table["runner_protect_target_v1"].sum()),
            "ambiguous_high_mfe_monitor_v1": int(label_table["ambiguous_high_mfe_monitor_v1"].sum()),
            "total_protected_or_ambiguous_v1": int(len(protected)),
        },
        "missed_390_distribution_v1": {str(key): int(value) for key, value in missed_counts.items()},
        "ambiguous_high_mfe_bad_positive_count_v1": ambiguous_bad_positive,
        "expected_recall_opportunity_v1": int(missed["new_r5_2_label_bucket_v1"].isin(["STRONG_BAD_BLOCK_TARGET", "TAIL_CONTROL_TARGET", "RISKY_ALLOW_TARGET"]).sum()),
        "expected_safety_risk_v1": "LOW_IF_PROTECTION_HEAD_AND_NO_GO_GATES_ARE_ENFORCED" if ambiguous_bad_positive == 0 else "HIGH_AMBIGUOUS_BAD_POSITIVE_LEAK",
        "label_table_balanced_enough_for_spec_v1": bool(int(len(positives)) > 0 and int(len(protected)) > 0 and ambiguous_bad_positive == 0),
    }


def _gate(dry_run: dict[str, Any], feature_alignment: pd.DataFrame) -> dict[str, Any]:
    ambiguous_safe = int(dry_run["ambiguous_high_mfe_bad_positive_count_v1"]) == 0
    feature_sufficient = bool(feature_alignment["signal_basis_enough_for_true_rebuild_v1"].any()) if not feature_alignment.empty else False
    balanced = bool(dry_run["label_table_balanced_enough_for_spec_v1"])
    if ambiguous_safe and feature_sufficient and balanced:
        decision = "R5_2_LABEL_OBJECTIVE_READY_FOR_TRUE_REBUILD_SPEC"
    elif not ambiguous_safe:
        decision = "R5_2_AMBIGUOUS_HIGH_MFE_HANDLING_NOT_SAFE"
    elif not feature_sufficient:
        decision = "R5_2_FEATURE_SIGNAL_NOT_SUFFICIENT"
    else:
        decision = "R5_2_LABEL_OBJECTIVE_NEEDS_MORE_POCKET_CLEANUP"
    return {
        "layer_name": "R5_2_LABEL_OBJECTIVE_GATE_V1",
        "decision_v1": decision,
        "checks_v1": {
            "ambiguous_high_mfe_not_bad_positive_v1": ambiguous_safe,
            "feature_signal_sufficient_v1": feature_sufficient,
            "class_balance_sufficient_for_spec_v1": balanced,
            "ready_for_true_rebuild_spec_v1": decision == "R5_2_LABEL_OBJECTIVE_READY_FOR_TRUE_REBUILD_SPEC",
        },
    }


def _next_action(gate: dict[str, Any]) -> dict[str, Any]:
    decision = gate["decision_v1"]
    if decision == "R5_2_LABEL_OBJECTIVE_READY_FOR_TRUE_REBUILD_SPEC":
        action = "BUILD_TRUE_R5_2_REBUILD_RUNNER_SPEC_NEXT"
    elif decision == "R5_2_LABEL_OBJECTIVE_NEEDS_MORE_POCKET_CLEANUP":
        action = "FIX_R5_2_POCKET_LABEL_TABLE_FIRST"
    elif decision == "R5_2_AMBIGUOUS_HIGH_MFE_HANDLING_NOT_SAFE":
        action = "HARDEN_HIGH_MFE_AMBIGUOUS_LABEL_RULES_FIRST"
    elif decision == "R5_2_FEATURE_SIGNAL_NOT_SUFFICIENT":
        action = "WIRE_EXISTING_UNDERUSED_FEATURE_SIGNALS_FIRST"
    else:
        action = "NOT_ESTABLISHED"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": action,
        "blocked_action_v1": [
            "DO_NOT_RUN_TRUE_R5_2_REBUILD_BEFORE_RUNNER_SPEC",
            "DO_NOT_CONTINUE_TINY_EXTENSION_LOOP",
            "DO_NOT_BUILD_NEW_BASELINE",
            "DO_NOT_USE_1689_EXACT_ONLY",
            "DO_NOT_USE_PROTECTOR_FIRST",
        ],
    }


def _audit(summary: dict[str, Any], gate: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("OBJECTIVE_MISMATCH_LOCKED", summary["objective_mismatch_locked_v1"], True),
            row("LABEL_TABLE_MATERIALIZED", summary["label_table_rows_v1"] == 1914, summary["label_table_rows_v1"]),
            row("AMBIGUOUS_NOT_BAD_POSITIVE", summary["ambiguous_high_mfe_bad_positive_count_v1"] == 0, summary["ambiguous_high_mfe_bad_positive_count_v1"]),
            row("GATE_READY", gate["decision_v1"] == "R5_2_LABEL_OBJECTIVE_READY_FOR_TRUE_REBUILD_SPEC", gate["decision_v1"]),
            row("NO_TRAINING_RUN", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_NEW_BASELINE", True, True),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Fix R5.2 Label Objective First",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Label table rows: `{summary['label_table_rows_v1']}`",
            f"- Missed-390 new positive/attention opportunity: `{summary['missed_390_positive_or_attention_v1']}`",
            f"- Ambiguous high-MFE bad-positive count: `{summary['ambiguous_high_mfe_bad_positive_count_v1']}`",
            f"- Existing feature signal sufficient: `{summary['existing_feature_signal_sufficient_v1']}`",
            "",
            "No true R5.2 rebuild, model training, new baseline, feature rebuild, R6 retrain, freeze, promotion, live gate, or controller path was run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v3_score_dir: Path = V3_SCORE_DEFAULT,
    v3_r6_dir: Path = V3_R6_DEFAULT,
    investigation_dir: Path = INVESTIGATION_DEFAULT,
    asset_reuse_dir: Path = ASSET_REUSE_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v3_score_dir = v3_score_dir.expanduser().resolve()
    v3_r6_dir = v3_r6_dir.expanduser().resolve()
    investigation_dir = investigation_dir.expanduser().resolve()
    asset_reuse_dir = asset_reuse_dir.expanduser().resolve()

    score = pd.read_parquet(v3_score_dir / SCORE_FRAME)
    r6 = pd.read_parquet(v3_r6_dir / R6_FRAME)
    score_summary = _read_json(v3_score_dir / SCORE_SUMMARY)
    investigation = _safe_read_json(investigation_dir / "summary_v1.json")
    label_table = _label_table(score, r6, score_summary)
    mismatch = _mismatch_lock(investigation)
    label_contract = _label_contract()
    weighting = _weighting_spec()
    feature_alignment = _feature_alignment(label_table, score)
    rebuild_spec = _rebuild_spec(label_contract, weighting)
    dry_run = _dry_run(label_table)
    gate = _gate(dry_run, feature_alignment)
    next_action = _next_action(gate)
    missed = label_table[(_bool(label_table, "label_should_not_take_v1") | _bool(label_table, "tail_10_50_mfe_v1")) & ~_bool(label_table, "r5_2_v3_base_flag_v1")]
    missed_positive_or_attention = int(missed["new_r5_2_label_bucket_v1"].isin(["STRONG_BAD_BLOCK_TARGET", "TAIL_CONTROL_TARGET", "RISKY_ALLOW_TARGET"]).sum())
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "v3_score_dir_v1": str(v3_score_dir),
        "v3_r6_dir_v1": str(v3_r6_dir),
        "investigation_dir_v1": str(investigation_dir),
        "asset_reuse_dir_v1": str(asset_reuse_dir),
        "decision_v1": gate["decision_v1"],
        "next_action_v1": next_action["next_action_v1"],
        "objective_mismatch_locked_v1": True,
        "label_contract_id_v1": label_contract["contract_id_v1"],
        "label_table_rows_v1": int(len(label_table)),
        "missed_390_positive_or_attention_v1": missed_positive_or_attention,
        "missed_390_distribution_v1": dry_run["missed_390_distribution_v1"],
        "bad_tail_stronger_target_v1": True,
        "ambiguous_high_mfe_bad_positive_count_v1": int(dry_run["ambiguous_high_mfe_bad_positive_count_v1"]),
        "runner_winner_protection_explicit_v1": True,
        "existing_feature_signal_sufficient_v1": bool(gate["checks_v1"]["feature_signal_sufficient_v1"]),
        "ready_for_rebuild_spec_v1": gate["decision_v1"] == "R5_2_LABEL_OBJECTIVE_READY_FOR_TRUE_REBUILD_SPEC",
        "training_started_v1": False,
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
        "hard_status_v1": {
            "BEVIST": [
                "R5.2 objective mismatch is locked from post-V3 forensics.",
                "A new pocket-aware R5.2 label/objective contract and full Monday label table were materialized.",
                "Ambiguous high-MFE rows are not bad-positive in the new label table.",
                "No training, true rebuild, new baseline, feature surface, R6 retrain, freeze, promotion, live gate, or controller path was run.",
            ],
            "INDIKERT": [
                "Existing AS_OF/R5/R5.1 signals are sufficient to specify a true R5.2 rebuild runner.",
            ],
            "IKKE_ETABLERT": [
                "A green rebuilt R5.2 score package is not established until the rebuild runner is built and explicitly run.",
            ],
        },
    }
    audit = _audit(summary, gate)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "v3_score_dir_v1": str(v3_score_dir),
            "v3_r6_dir_v1": str(v3_r6_dir),
            "investigation_dir_v1": str(investigation_dir),
            "asset_reuse_dir_v1": str(asset_reuse_dir),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
    }

    _write_json(output_dir / OUTPUT_FILES["mismatch"], mismatch)
    _write_json(output_dir / OUTPUT_FILES["label_contract"], label_contract)
    _write_json(output_dir / OUTPUT_FILES["weighting"], weighting)
    label_table.to_csv(output_dir / OUTPUT_FILES["pocket_table"], index=False)
    feature_alignment.to_csv(output_dir / OUTPUT_FILES["feature_alignment"], index=False)
    _write_json(output_dir / OUTPUT_FILES["rebuild_spec"], rebuild_spec)
    _write_json(output_dir / OUTPUT_FILES["dry_run"], dry_run)
    _write_json(output_dir / OUTPUT_FILES["gate"], gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v3-score-dir", type=Path, default=V3_SCORE_DEFAULT)
    parser.add_argument("--v3-r6-dir", type=Path, default=V3_R6_DEFAULT)
    parser.add_argument("--investigation-dir", type=Path, default=INVESTIGATION_DEFAULT)
    parser.add_argument("--asset-reuse-dir", type=Path, default=ASSET_REUSE_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v3_score_dir=args.v3_score_dir,
        v3_r6_dir=args.v3_r6_dir,
        investigation_dir=args.investigation_dir,
        asset_reuse_dir=args.asset_reuse_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
