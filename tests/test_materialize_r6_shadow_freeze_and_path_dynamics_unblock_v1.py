from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1 import materialize as materialize_r5_2_freeze
from gx1.scripts.materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import materialize as materialize_phase_gate
from gx1.scripts.materialize_r6_shadow_freeze_and_path_dynamics_unblock_v1 import (
    BATCH05_MARGIN_MONITOR,
    CONSISTENCY_AUDIT,
    CONTRACT_LOCK_TABLE,
    FREEZE_MANIFEST,
    HINDSIGHT_BACKFILL_LOCK,
    NEXT_STEP_DECISION_MATRIX,
    PATH_DYNAMICS_BLOCKER_AUDIT,
    PATH_DYNAMICS_INSTRUMENTATION_SPEC,
    POLICY_LOGGING_LOCK,
    R7_BACKLOG_TABLE,
    SUMMARY,
    materialize,
)
from gx1.scripts.train_r6_entry_runner_first_retrain_v1 import materialize as materialize_r6
from tests.test_materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1 import _write_model_fixture
from tests.test_materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import _build_fixture


def test_r6_shadow_freeze_and_path_dynamics_unblock_materializes(tmp_path: Path) -> None:
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
    r5_2_freeze_dir = reports_root / "r5_2_freeze"
    materialize_r5_2_freeze(
        reports_root,
        phase_dir=phase_dir,
        extension_dir=r5_2_freeze_dir,
        batch_weeks=1,
        expected_ledger_count=50,
        test_status="PYTEST_FIXTURE_PASS",
    )
    r6_dir = reports_root / "r6"
    materialize_r6(
        reports_root,
        freeze_dir=r5_2_freeze_dir,
        extension_dir=r6_dir,
        batch_weeks=1,
        expected_ledger_count=50,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        compact_grid=True,
    )
    extension_dir = reports_root / "r6_freeze"
    result = materialize(
        reports_root,
        r6_dir=r6_dir,
        r5_2_freeze_dir=r5_2_freeze_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        expected_ledger_count=50,
        test_status="PYTEST_FIXTURE_PASS",
    )
    assert result["status"]["not_live_gate"] is True
    for artifact in [
        FREEZE_MANIFEST,
        CONTRACT_LOCK_TABLE,
        POLICY_LOGGING_LOCK,
        HINDSIGHT_BACKFILL_LOCK,
        BATCH05_MARGIN_MONITOR,
        PATH_DYNAMICS_BLOCKER_AUDIT,
        PATH_DYNAMICS_INSTRUMENTATION_SPEC,
        R7_BACKLOG_TABLE,
        NEXT_STEP_DECISION_MATRIX,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["status_v1"]["not_live_gate"] is True
    assert summary["path_dynamics_v1"]["blocked_field_count_v1"] == 5
    policy = pd.read_parquet(extension_dir / POLICY_LOGGING_LOCK)
    backfill = pd.read_parquet(extension_dir / HINDSIGHT_BACKFILL_LOCK)
    assert len(policy) == 50
    assert len(backfill) == 50
    assert policy["policy_mask_matches_materialized_v1"].all()
    blocker = pd.read_csv(extension_dir / PATH_DYNAMICS_BLOCKER_AUDIT)
    assert len(blocker) == 5
    assert blocker["blocker_status_v1"].eq("LOGGING_BLOCKED_FOR_R6_ENTRY_AS_OF").all()
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
