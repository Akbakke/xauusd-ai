from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1 import (
    ARTIFACT_HASH_TABLE,
    CONSISTENCY_AUDIT,
    FAILURE_CLUSTER_TABLE,
    FREEZE_MANIFEST,
    GO_NO_GO_MATRIX,
    POLICY_LOGGING_LOCK,
    R6_OPPORTUNITY_AUDIT,
    R6_TRAINING_TARGET_SPEC,
    SUMMARY,
    materialize,
)
from gx1.scripts.materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import materialize as materialize_phase_gate
from tests.test_materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import _build_fixture


def _write_model_fixture(r5_2_dir: Path) -> None:
    for label in ["r5_2_label_bad_blocker_v1", "r5_2_label_runner_protect_v1"]:
        model_dir = r5_2_dir / "models" / "global_r5_2_runner_aware" / label
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.joblib").write_bytes(f"{label}:model".encode("utf-8"))
        (model_dir / "feature_preprocessor.joblib").write_bytes(f"{label}:preprocessor".encode("utf-8"))
        (model_dir / "metadata.json").write_text(
            json.dumps({"label_col_v1": label, "model_tag_v1": "fixture"}, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_r5_2_shadow_freeze_and_r6_backlog_materializes(tmp_path: Path) -> None:
    reports_root, r5_2_dir, harvest_dir, rl_dir, unified_dir = _build_fixture(tmp_path)
    _write_model_fixture(r5_2_dir)
    phase_dir = reports_root / "phase_gate"
    materialize_phase_gate(
        reports_root,
        r5_2_dir=r5_2_dir,
        harvest_dir=harvest_dir,
        rl_recommendation_dir=rl_dir,
        rl_unified_dir=unified_dir,
        extension_dir=phase_dir,
        batch_weeks=1,
        expected_ledger_count=50,
    )
    extension_dir = reports_root / "freeze"
    result = materialize(
        reports_root,
        phase_dir=phase_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        expected_ledger_count=50,
        test_status="PYTEST_FIXTURE_PASS",
    )
    assert result["status"]["not_live_gate"] is True
    for artifact in [
        FREEZE_MANIFEST,
        POLICY_LOGGING_LOCK,
        FAILURE_CLUSTER_TABLE,
        R6_OPPORTUNITY_AUDIT,
        R6_TRAINING_TARGET_SPEC,
        GO_NO_GO_MATRIX,
        ARTIFACT_HASH_TABLE,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()
    manifest = json.loads((extension_dir / FREEZE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["freeze_status_v1"] == "FROZEN_SHADOW_FALLBACK_CANDIDATE_NOT_LIVE_GATE"
    assert manifest["model_version_id_v1"]
    assert manifest["selected_policy_stack_v1"] == "R5_2_CANDIDATE_FIXTURE_SELECTED"
    policy = pd.read_parquet(extension_dir / POLICY_LOGGING_LOCK)
    assert len(policy) == 50
    assert policy["candidate_uid_exact_v1"].notna().all()
    assert policy["threshold_version_id_v1"].astype("string").eq(manifest["threshold_version_id_v1"]).all()
    hashes = pd.read_csv(extension_dir / ARTIFACT_HASH_TABLE)
    assert int(hashes["artifact_role_v1"].eq("R5_2_MODEL_ARTIFACT").sum()) >= 6
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
