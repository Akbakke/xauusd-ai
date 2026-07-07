from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_rebuild_iql_state_contract_with_more_as_of_features_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path(
                "/tmp/REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1_20260429T000000Z_LOCK"
            )
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_no_forbidden_actions_default_pass_and_explicit_block_fail() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        adapter=True,
        r6=True,
        iql_production=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
        broad_sweep=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]


def test_validate_final_status_only_accepts_allowlist() -> None:
    assert gate.validate_final_status(
        "REBUILD_STATE_CONTRACT_PASS_V2_READY_REWARD_VARIANTS_LOCKED_TIMING_AUDIT_AVAILABLE",
        "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "ARBITRARY_STATUS_NOT_ALLOWED",
            "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
        )
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "REBUILD_STATE_CONTRACT_PASS_V2_READY_REWARD_VARIANTS_LOCKED_TIMING_AUDIT_AVAILABLE",
            "OPEN_R6_NOW",
        )


def test_validate_no_deprecated_revival_blocks_quarantine_imports(tmp_path: Path) -> None:
    bad = tmp_path / "with_quarantine.py"
    bad.write_text(
        "from gx1.quarantine._DEPRECATED_SCRIPTS_20260219 import run_shadow_counterfactual\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        gate.validate_no_deprecated_revival(bad)
    good = tmp_path / "without_quarantine.py"
    good.write_text("import pandas as pd\n", encoding="utf-8")
    assert gate.validate_no_deprecated_revival(good)


def test_classify_family_picks_first_matching_pattern() -> None:
    assert gate._classify_family("session_overlap_v1") == "REGIME"
    assert gate._classify_family("score_pctile_train_only_v1") == "UNCERTAINTY"
    assert gate._classify_family("threshold_policy_v1") == "MARGIN"
    assert gate._classify_family("source_evidence_count_v1") == "SOURCE_QUALITY"
    assert gate._classify_family("totally_random_field") is None


def test_state_v2_discovery_rejects_blocklisted_and_low_coverage() -> None:
    rng = np.random.default_rng(42)
    n = 1914
    frame = pd.DataFrame(
        {
            "candidate_score_v1": rng.normal(size=n),
            "signal_r5_1_bad_score_v1": rng.integers(0, 2, size=n).astype(bool),
            # New REGIME candidate — well-covered, high variance, unique values
            "session_overlap_v1": pd.Categorical(
                rng.choice(["ASIA", "EU", "US", "OVERLAP"], size=n)
            ),
            # New UNCERTAINTY candidate — well-covered numeric
            "score_pctile_train_only_v1": rng.uniform(0, 1, size=n),
            # Should be REJECTED by name blocklist (mfe pattern)
            "regime_mfe_proxy_v1": rng.normal(size=n),
            # Should be REJECTED by low coverage
            "regime_sparse_field_v1": pd.array(
                [rng.normal() if i < 100 else None for i in range(n)],
                dtype="Float64",
            ),
            # Should be REJECTED by no family pattern match
            "totally_unclassified_v1": rng.normal(size=n),
            # Should be REJECTED by degenerate variance
            "session_constant_v1": np.zeros(n),
            # Should be ACCEPTED but as SOURCE_QUALITY (count keyword)
            "source_evidence_count_v1": rng.integers(0, 10, size=n).astype(float),
        }
    )
    discovery = gate._state_v2_discovery(frame)
    accepted_names = {r["field_name_v1"] for r in discovery["accepted_v1"]}
    assert "session_overlap_v1" in accepted_names
    assert "score_pctile_train_only_v1" in accepted_names
    assert "source_evidence_count_v1" in accepted_names
    assert "regime_mfe_proxy_v1" not in accepted_names
    assert "regime_sparse_field_v1" not in accepted_names
    assert "totally_unclassified_v1" not in accepted_names
    assert "session_constant_v1" not in accepted_names
    family_status = discovery["summary_v1"]["family_status_v1"]
    assert family_status["REGIME"] == "QUALIFYING_AS_OF_CANDIDATE_ACCEPTED"
    assert family_status["UNCERTAINTY"] == "QUALIFYING_AS_OF_CANDIDATE_ACCEPTED"
    assert family_status["SOURCE_QUALITY"] == "QUALIFYING_AS_OF_CANDIDATE_ACCEPTED"
    assert family_status["MARGIN"] == "NOT_ESTABLISHED_NO_QUALIFYING_AS_OF_CANDIDATE"


def test_state_no_shortcut_audit_v2_blocks_denied_and_reward_inputs() -> None:
    safe_rows = [
        {"field_name_v1": "candidate_score_v1", "allowed_as_state_v1": True},
        {"field_name_v1": "session_overlap_v1", "allowed_as_state_v1": True},
        {"field_name_v1": "bad_label_v1", "allowed_as_state_v1": False},
        {"field_name_v1": "mfe_bps", "allowed_as_state_v1": False},
    ]
    audit = gate._state_no_shortcut_audit_v2(safe_rows)
    assert audit["no_shortcut_status_v1"] == "PASS"

    leaky_rows = list(safe_rows)
    leaky_rows.append({"field_name_v1": "bad_label_v1", "allowed_as_state_v1": True})
    with pytest.raises(RuntimeError, match="STATE_V2_NO_SHORTCUT_AUDIT_FAILED"):
        gate._state_no_shortcut_audit_v2(leaky_rows)

    leaky_rows_2 = list(safe_rows)
    leaky_rows_2.append({"field_name_v1": "mfe_bps", "allowed_as_state_v1": True})
    with pytest.raises(RuntimeError, match="STATE_V2_NO_SHORTCUT_AUDIT_FAILED"):
        gate._state_no_shortcut_audit_v2(leaky_rows_2)


def test_reward_formulas_match_specs() -> None:
    iql_view = pd.DataFrame(
        {
            "candidate_uid_v1": ["a", "b", "c"],
            "is_inside_78_shield_v1": [True, True, True],
            "is_safe_core_89_v1": [True, True, True],
            "take_trade_action_v1": [True, True, True],
            "pnl_bps": [50.0, -20.0, 80.0],
            "mae_bps": [-10.0, -50.0, -5.0],
            "mfe_bps": [60.0, 10.0, 100.0],
        }
    )
    table, distributions = gate._compute_reward_variants(iql_view)
    pnl_col = "ENTRY_REALIZED_PNL_REWARD_V2_value_v1"
    mfe_col = "ENTRY_MFE_CAPTURE_REWARD_V2_value_v1"
    mae_col = "ENTRY_MAE_BURDEN_REWARD_V2_value_v1"
    combined_col = "ENTRY_TRANSPARENT_COMBINED_REWARD_V2_value_v1"
    assert list(table[pnl_col].values) == [50.0, -20.0, 80.0]
    assert list(table[mfe_col].values) == pytest.approx([50 / 60, -20 / 10, 80 / 100])
    assert list(table[mae_col].values) == [50 - 5, -20 - 25, 80 - 2.5]
    assert list(table[combined_col].values) == pytest.approx(
        [
            50 - 0.25 * 10 - 0.25 * max(60 - 50, 0),
            -20 - 0.25 * 50 - 0.25 * max(10 - (-20), 0),
            80 - 0.25 * 5 - 0.25 * max(100 - 80, 0),
        ]
    )
    assert distributions["ENTRY_REALIZED_PNL_REWARD_V2"]["count_v1"] == 3
    assert distributions["ENTRY_MFE_CAPTURE_REWARD_V2"]["clip_low_count_v1"] == 1


def test_reward_class_audit_blocks_mae_mfe_pnl_in_state() -> None:
    safe_rows = [
        {"field_name_v1": "candidate_score_v1", "allowed_as_state_v1": True},
        {"field_name_v1": "session_overlap_v1", "allowed_as_state_v1": True},
    ]
    audit = gate._reward_class_audit(safe_rows)
    assert audit["leakage_status_v1"] == "PASS"
    leaky_rows = list(safe_rows) + [
        {"field_name_v1": "mfe_bps", "allowed_as_state_v1": True}
    ]
    with pytest.raises(RuntimeError, match="REWARD_INPUT_LEAK_INTO_STATE"):
        gate._reward_class_audit(leaky_rows)


def test_go_no_go_partitions_outcomes_correctly() -> None:
    state_pass = {"accepted_count_v1": 3, "rejected_count_v1": 0}
    state_partial = {"accepted_count_v1": 1, "rejected_count_v1": 0}
    reward_locked = {
        "join_status_v1": "REWARD_JOIN_LOCKED",
        "iql_dataset_row_count_v1": 1914,
        "take_trade_match_rate_v1": 1.0,
        "overall_match_rate_v1": 1.0,
    }
    reward_blocked = {
        "join_status_v1": "REWARD_JOIN_NOT_ESTABLISHED",
        "iql_dataset_row_count_v1": 1914,
        "take_trade_match_rate_v1": 0.5,
        "overall_match_rate_v1": 0.5,
    }
    timing_ok = {"timing_status_v1": "TIMING_AUDIT_AVAILABLE"}
    timing_bad = {"timing_status_v1": "TIMING_AUDIT_NOT_ESTABLISHED"}

    status, action, _ = gate._go_no_go(state_pass, reward_locked, timing_ok)
    assert status.startswith("REBUILD_STATE_CONTRACT_PASS_V2_READY")

    status, action, _ = gate._go_no_go(state_partial, reward_locked, timing_ok)
    assert status == "REBUILD_STATE_PARTIAL_REWARD_VARIANTS_LOCKED_STATE_INSUFFICIENT"
    assert action == "DEEPEN_IQL_STATE_FAMILY_DISCOVERY_V1"

    status, action, _ = gate._go_no_go(state_pass, reward_blocked, timing_ok)
    assert status == "REBUILD_STATE_PARTIAL_STATE_OK_REWARD_JOIN_NOT_ESTABLISHED"
    assert action == "REPAIR_REWARD_JOIN_LINEAGE_V1"

    status, action, _ = gate._go_no_go(state_pass, reward_locked, timing_bad)
    assert status == "REBUILD_STATE_PARTIAL_TIMING_NOT_ESTABLISHED"
    assert action == "DEEPEN_TIMING_AUDIT_ALT_PATH_V1"

    status, action, _ = gate._go_no_go(state_partial, reward_blocked, timing_bad)
    assert status == "REBUILD_STATE_BLOCKED_NO_NEW_AS_OF_FIELDS_AND_REWARD_JOIN_FAILED"


def test_materialize_self_passes_no_deprecated_revival_check() -> None:
    script_path = (
        Path(gate.__file__).resolve()
    )
    assert gate.validate_no_deprecated_revival(script_path)


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in gate.REQUIRED_OUTPUTS_TOPLEVEL:
        assert (artifact_root / required).exists(), f"missing {required}"
    state_dir = artifact_root / "STATE_EXPANSION_V2"
    reward_dir = artifact_root / "REWARD_VARIANTS_V2"
    timing_dir = artifact_root / "TIMING_AUDIT_V1"
    for sub in (state_dir, reward_dir, timing_dir):
        assert sub.is_dir(), f"missing sub-dir {sub}"
    assert (state_dir / "iql_offline_state_contract_v2.json").exists()
    assert (state_dir / "iql_offline_state_contract_v2_diff_vs_v1.csv").exists()
    assert (state_dir / "state_no_shortcut_audit_v2.json").exists()
    assert (reward_dir / "iql_entry_iql_reward_variants_contract_v2.json").exists()
    assert (reward_dir / "reward_variant_class_audit_v1.json").exists()
    assert (reward_dir / "entry_iql_post_trade_outcome_join_audit_v1.json").exists()
    assert (timing_dir / "entry_timing_audit_recommendation_v1.json").exists()

    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["row_count_invariant_v1"] is True
    assert summary["seventy_eight_shield_invariant_v1"] is True
    assert summary["reward_variant_count_v1"] == 4
    assert summary["state_v1_field_count_v1"] >= 9
    assert summary["state_v2_field_count_v1"] >= summary["state_v1_field_count_v1"]
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS

    no_shortcut = json.loads((state_dir / "state_no_shortcut_audit_v2.json").read_text())
    assert no_shortcut["no_shortcut_status_v1"] == "PASS"
    class_audit = json.loads(
        (reward_dir / "reward_variant_class_audit_v1.json").read_text()
    )
    assert class_audit["leakage_status_v1"] == "PASS"
