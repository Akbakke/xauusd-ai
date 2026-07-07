"""Tests for materialize_run_per_bar_xgb_replay_for_transformer_signal_family_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_run_per_bar_xgb_replay_for_transformer_signal_family_v1 as gate,
)


def test_session_int_to_name_mapping_is_canonical() -> None:
    assert gate.SESSION_ID_INT_TO_NAME == {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}


def test_session_head_vocab_matches_xgb_bundle_heads() -> None:
    assert set(gate.SESSION_HEAD_VOCAB) == set(gate.SESSION_ID_INT_TO_NAME.values())


def test_normalize_session_maps_integers_to_head_names() -> None:
    s = pd.Series([0.0, 1.0, 2.0, 3.0, np.nan])
    out = gate._normalize_session(s)
    assert out.iloc[0] == "ASIA"
    assert out.iloc[1] == "EU"
    assert out.iloc[2] == "OVERLAP"
    assert out.iloc[3] == "US"
    assert pd.isna(out.iloc[4])


def test_normalize_session_uppercases_string_input() -> None:
    s = pd.Series(["asia", "eu", "overlap", "us"])
    out = gate._normalize_session(s)
    assert list(out) == ["ASIA", "EU", "OVERLAP", "US"]


def test_per_bar_xgb_output_columns_match_v2_contract_names() -> None:
    expected = {
        "per_bar_xgb_p_long_v2",
        "per_bar_xgb_p_short_v2",
        "per_bar_xgb_p_flat_v2",
        "per_bar_xgb_p_hat_v2",
        "per_bar_xgb_uncertainty_score_v2",
        "per_bar_xgb_margin_top1_top2_v2",
        "per_bar_xgb_entropy_v2",
    }
    assert set(gate.PER_BAR_XGB_OUTPUT_COLUMNS) == expected


def test_audit_feature_alignment_passes_when_all_features_present() -> None:
    bundle_features = ["a", "b", "c"]
    base34_cols = {"a", "b", "c", "extra"}
    audit = gate.audit_feature_alignment(bundle_features, base34_cols)
    assert audit["status_v1"] == "PASS"
    assert audit["bundle_features_missing_in_base34_v1"] == []
    assert "extra" in audit["base34_columns_not_used_by_bundle_v1"]


def test_audit_feature_alignment_fails_when_missing() -> None:
    bundle_features = ["a", "b", "missing_feature"]
    base34_cols = {"a", "b"}
    audit = gate.audit_feature_alignment(bundle_features, base34_cols)
    assert audit["status_v1"] == "FAIL"
    assert "missing_feature" in audit["bundle_features_missing_in_base34_v1"]


def test_audit_temporal_correctness_documents_policy() -> None:
    audit = gate.audit_temporal_correctness(pd.DataFrame())
    assert audit["status_v1"] == "PASS"
    assert "merge_asof" in audit["policy_v1"]
    assert "backward" in audit["policy_v1"]


def test_audit_session_coverage_groups_by_head() -> None:
    df = pd.DataFrame(
        {
            "xgb_head_used_v1": ["ASIA", "ASIA", "EU", "OVERLAP", None, None, "US"],
            "replay_status_v1": ["REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1"] * 7,
        }
    )
    audit = gate.audit_session_coverage(df)
    assert audit["status_v1"] == "PASS"
    counts = audit["head_counts_v1"]
    assert counts["ASIA"] == 2
    assert counts["EU"] == 1
    assert counts["OVERLAP"] == 1
    assert counts["US"] == 1
    assert counts["__NULL__"] == 2


def test_audit_replay_status_distribution_computes_replay_rate() -> None:
    df = pd.DataFrame(
        {
            "replay_status_v1": [
                "REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1",
                "REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1",
                "REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1",
                "NOT_REPLAYED_BASE34_NAN",
            ],
        }
    )
    audit = gate.audit_replay_status_distribution(df)
    assert audit["row_count_v1"] == 4
    assert audit["replayed_count_v1"] == 3
    assert audit["replay_rate_v1"] == 0.75


def test_audit_signal7_invariants_passes_on_clean_replay() -> None:
    df = pd.DataFrame(
        {
            "replay_status_v1": ["REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1"] * 2,
            "per_bar_xgb_p_long_v2": [0.5, 0.6],
            "per_bar_xgb_p_short_v2": [0.3, 0.2],
            "per_bar_xgb_p_flat_v2": [0.2, 0.2],
            "per_bar_xgb_p_hat_v2": [0.5, 0.6],
            "per_bar_xgb_uncertainty_score_v2": [0.5, 0.4],
            "per_bar_xgb_margin_top1_top2_v2": [0.2, 0.4],
            "per_bar_xgb_entropy_v2": [1.0, 0.9],
        }
    )
    audit = gate.audit_signal7_invariants(df)
    assert audit["status_v1"] == "PASS"
    assert audit["row_count_v1"] == 2


def test_audit_signal7_invariants_fails_when_uncertainty_breaks_invariant() -> None:
    df = pd.DataFrame(
        {
            "replay_status_v1": ["REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1"],
            "per_bar_xgb_p_long_v2": [0.5],
            "per_bar_xgb_p_short_v2": [0.3],
            "per_bar_xgb_p_flat_v2": [0.2],
            "per_bar_xgb_p_hat_v2": [0.5],
            # bug: uncertainty should equal 1 - p_hat = 0.5, but it's 0.9
            "per_bar_xgb_uncertainty_score_v2": [0.9],
            "per_bar_xgb_margin_top1_top2_v2": [0.2],
            "per_bar_xgb_entropy_v2": [1.0],
        }
    )
    audit = gate.audit_signal7_invariants(df)
    assert audit["status_v1"] == "FAIL"
    assert "UNCERTAINTY_NOT_EQUAL_1_MINUS_P_HAT" in audit["failures_v1"]


def test_audit_signal7_invariants_fails_when_probs_dont_sum_to_one() -> None:
    df = pd.DataFrame(
        {
            "replay_status_v1": ["REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1"],
            "per_bar_xgb_p_long_v2": [0.5],
            "per_bar_xgb_p_short_v2": [0.3],
            "per_bar_xgb_p_flat_v2": [0.5],  # sum = 1.3
            "per_bar_xgb_p_hat_v2": [0.5],
            "per_bar_xgb_uncertainty_score_v2": [0.5],
            "per_bar_xgb_margin_top1_top2_v2": [0.2],
            "per_bar_xgb_entropy_v2": [1.0],
        }
    )
    audit = gate.audit_signal7_invariants(df)
    assert audit["status_v1"] == "FAIL"
    assert "PROB_SUM_NOT_ONE" in audit["failures_v1"]


def test_go_no_go_pass_full_coverage() -> None:
    feature = {"status_v1": "PASS"}
    replay = {
        "row_count_v1": 1000,
        "replayed_count_v1": 1000,
        "replay_rate_v1": 1.0,
        "status_counts_v1": {"REPLAYED_FROM_BASE34_M5_AT_BAR_T_MINUS_1": 1000},
    }
    sig = {"status_v1": "PASS"}
    status, action, _, headline = gate._go_no_go(feature, replay, sig)
    assert status == "RUN_PER_BAR_XGB_REPLAY_PASS_FULL_COVERAGE_V1"
    assert action == "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1"
    assert headline["replay_rate_v1"] == 1.0


def test_go_no_go_partial_coverage_above_threshold() -> None:
    feature = {"status_v1": "PASS"}
    replay = {
        "row_count_v1": 1000,
        "replayed_count_v1": 980,
        "replay_rate_v1": 0.98,
        "status_counts_v1": {},
    }
    sig = {"status_v1": "PASS"}
    status, action, _, _ = gate._go_no_go(feature, replay, sig)
    assert status == "RUN_PER_BAR_XGB_REPLAY_PARTIAL_COVERAGE_V1"
    assert action == "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1"


def test_go_no_go_blocks_low_coverage() -> None:
    feature = {"status_v1": "PASS"}
    replay = {
        "row_count_v1": 1000,
        "replayed_count_v1": 500,
        "replay_rate_v1": 0.50,
        "status_counts_v1": {},
    }
    sig = {"status_v1": "PASS"}
    status, action, _, _ = gate._go_no_go(feature, replay, sig)
    assert status == "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_LOW_COVERAGE_V1"
    assert action == "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1"


def test_go_no_go_blocks_feature_mismatch() -> None:
    feature = {"status_v1": "FAIL"}
    replay = {
        "row_count_v1": 0,
        "replayed_count_v1": 0,
        "replay_rate_v1": None,
        "status_counts_v1": {},
    }
    sig = {"status_v1": "PASS"}
    status, action, _, _ = gate._go_no_go(feature, replay, sig)
    assert status == "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_FEATURE_MISMATCH_V1"
    assert action == "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1"


def test_go_no_go_blocks_when_signal7_invariants_fail() -> None:
    feature = {"status_v1": "PASS"}
    replay = {
        "row_count_v1": 1000,
        "replayed_count_v1": 1000,
        "replay_rate_v1": 1.0,
        "status_counts_v1": {},
    }
    sig = {"status_v1": "FAIL", "failures_v1": ["P_HAT_OUT_OF_RANGE"]}
    status, action, _, _ = gate._go_no_go(feature, replay, sig)
    assert status == "RUN_PER_BAR_XGB_REPLAY_BLOCKED_BY_LOW_COVERAGE_V1"
    assert action == "REPAIR_PER_BAR_XGB_REPLAY_BEFORE_PROMOTION_V1"


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "MADE_UP", "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "RUN_PER_BAR_XGB_REPLAY_PASS_FULL_COVERAGE_V1", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))
