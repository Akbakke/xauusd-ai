from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_exit_per_bar_state_feature_contract_v1 as gate


def test_explicit_artifact_roots_reject_latest() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1_20260429T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_LOCKED_AVAILABILITY_AUDIT_PASSED",
        "EXIT_ACTION_SUPPORT_AUGMENT_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("ARBITRARY", "EXIT_ACTION_SUPPORT_AUGMENT_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_LOCKED_AVAILABILITY_AUDIT_PASSED",
            "TRAIN_NOW_V1",
        )


def test_validate_no_deprecated_revival(tmp_path: Path) -> None:
    bad = tmp_path / "imports_quarantine.py"
    bad.write_text(
        "from gx1.quarantine._DEPRECATED_SCRIPTS_20260219 import x\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        gate.validate_no_deprecated_revival(bad)
    good = tmp_path / "clean.py"
    good.write_text("import pandas\n", encoding="utf-8")
    assert gate.validate_no_deprecated_revival(good)
    assert gate.validate_no_deprecated_revival(Path(gate.__file__))


def test_no_shortcut_blocks_forbidden_field_in_schema() -> None:
    safe = [{"field_name_v1": "running_pnl_at_close_bps_v1"}]
    audit = gate.validate_no_shortcut(safe)
    assert audit["status_v1"] == "PASS"
    leaky = [{"field_name_v1": "bar_count_v1"}]
    with pytest.raises(RuntimeError, match="FORBIDDEN_STATE_FIELD_IN_PROPOSED_SCHEMA"):
        gate.validate_no_shortcut(leaky)


def test_no_shortcut_blocks_forbidden_token_pattern() -> None:
    leaky_post_exit = [{"field_name_v1": "post_exit_drift_v1"}]
    with pytest.raises(RuntimeError, match="FORBIDDEN_TOKEN_IN_FIELD_NAME"):
        gate.validate_no_shortcut(leaky_post_exit)
    leaky_replay = [{"field_name_v1": "mfe_replay_end_obs_v1"}]
    with pytest.raises(RuntimeError):
        gate.validate_no_shortcut(leaky_replay)


def test_no_shortcut_blocks_identity_tokens() -> None:
    leaky = [{"field_name_v1": "trade_uid_z_v1"}]
    with pytest.raises(RuntimeError, match="IDENTITY_TOKEN_IN_FIELD_NAME"):
        gate.validate_no_shortcut(leaky)


def test_no_shortcut_allows_exit_prob_despite_exit_substring() -> None:
    safe = [{"field_name_v1": "exit_prob_v1"}]
    audit = gate.validate_no_shortcut(safe)
    assert audit["status_v1"] == "PASS"


def test_proposed_state_features_pass_no_shortcut() -> None:
    audit = gate.validate_no_shortcut(gate.PROPOSED_STATE_FEATURES)
    assert audit["status_v1"] == "PASS"
    assert audit["forbidden_field_intersection_v1"] == []
    assert audit["forbidden_token_pattern_hits_v1"] == []
    assert audit["identity_token_hits_v1"] == []


def test_proposed_state_features_have_required_categories() -> None:
    cats = {f["category_v1"] for f in gate.PROPOSED_STATE_FEATURES}
    assert cats == {
        "TRADE_STATE_RUNNING",
        "MARKET_STATE_AT_BAR",
        "ENTRY_CONTEXT_SNAPSHOT",
        "TRANSFORMER_SIGNAL_AT_BAR",
    }


def test_proposed_state_features_include_exit_prob_for_samstemte_alignment() -> None:
    field_names = {f["field_name_v1"] for f in gate.PROPOSED_STATE_FEATURES}
    assert "exit_prob_v1" in field_names
    exit_prob = next(
        f for f in gate.PROPOSED_STATE_FEATURES if f["field_name_v1"] == "exit_prob_v1"
    )
    assert exit_prob["category_v1"] == "TRANSFORMER_SIGNAL_AT_BAR"
    assert exit_prob["source_v1"] == "EXIT_EVAL_TRACE"
    assert exit_prob["availability_v1"] == "HAVE"


def test_proposed_state_features_mark_entry_transformer_outputs_not_established() -> None:
    not_estab = [
        f for f in gate.PROPOSED_STATE_FEATURES if f["availability_v1"] == "NOT_ESTABLISHED"
    ]
    not_estab_names = {f["field_name_v1"] for f in not_estab}
    for required_blocker in [
        "p_long_entry_v1",
        "p_hat_entry_v1",
        "uncertainty_entry_v1",
        "margin_entry_v1",
    ]:
        assert required_blocker in not_estab_names
    for f in not_estab:
        assert f["source_v1"] == "NOT_ESTABLISHED"


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in [
        "manifest_v1.json",
        "summary_v1.json",
        "status_v1.json",
        "report_v1.md",
        "exit_per_bar_state_feature_contract_go_no_go_v1.json",
        "input_manifest_v1.json",
        "state_feature_contract_v1.json",
        "availability_audit_v1.json",
        "feature_availability_table_v1.csv",
        "no_shortcut_audit_v1.json",
        "sample_state_vector_validation_v1.json",
        "reproducibility_audit_v1.json",
    ]:
        assert (artifact_root / required).exists(), f"missing {required}"
    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert summary["training_blocked_v1"] is True
    assert summary["no_shortcut_audit_status_v1"] == "PASS"
    contract = json.loads((artifact_root / "state_feature_contract_v1.json").read_text())
    assert contract["samstemte_alignment_field_v1"] == "exit_prob_v1"
    cat_counts = contract["category_counts_v1"]
    assert cat_counts["TRADE_STATE_RUNNING"] >= 5
    assert cat_counts["MARKET_STATE_AT_BAR"] >= 5
    assert cat_counts["TRANSFORMER_SIGNAL_AT_BAR"] >= 1
