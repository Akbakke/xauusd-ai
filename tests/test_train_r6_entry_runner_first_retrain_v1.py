from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1 import materialize as materialize_freeze
from gx1.scripts.materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import materialize as materialize_phase_gate
from gx1.scripts.train_r6_entry_runner_first_retrain_v1 import (
    AS_OF_FEATURE_TABLE,
    BAD_RISK_LABEL_AUDIT,
    CONSISTENCY_AUDIT,
    CONTRACT,
    DECISION_MATRIX,
    FEATURE_PATH_DYNAMICS_AUDIT,
    HINDSIGHT_LABEL_OUTCOME_TABLE,
    LOSO_METRICS,
    MODEL_FAMILY_BAKEOFF,
    POLICY_PREDICTION_VIEW,
    ROLLING_WINDOW_METRICS,
    RUNNER_LABEL_AUDIT,
    SUMMARY,
    TAIL_CONTROL_AUDIT,
    WALKFORWARD_METRICS,
    materialize,
)
from tests.test_materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1 import _write_model_fixture
from tests.test_materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import _build_fixture


def test_r6_entry_runner_first_retrain_materializes(tmp_path: Path) -> None:
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
    freeze_dir = reports_root / "freeze"
    materialize_freeze(
        reports_root,
        phase_dir=phase_dir,
        extension_dir=freeze_dir,
        batch_weeks=1,
        expected_ledger_count=50,
        test_status="PYTEST_FIXTURE_PASS",
    )
    extension_dir = reports_root / "r6"
    result = materialize(
        reports_root,
        freeze_dir=freeze_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        expected_ledger_count=50,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        compact_grid=True,
    )
    assert result["status"]["not_live_gate"] is True
    for artifact in [
        CONTRACT,
        AS_OF_FEATURE_TABLE,
        HINDSIGHT_LABEL_OUTCOME_TABLE,
        RUNNER_LABEL_AUDIT,
        BAD_RISK_LABEL_AUDIT,
        TAIL_CONTROL_AUDIT,
        FEATURE_PATH_DYNAMICS_AUDIT,
        MODEL_FAMILY_BAKEOFF,
        WALKFORWARD_METRICS,
        LOSO_METRICS,
        ROLLING_WINDOW_METRICS,
        POLICY_PREDICTION_VIEW,
        DECISION_MATRIX,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["coverage_v1"]["ledger_trade_count_v1"] == 50
    assert summary["coverage_v1"]["synthetic_count_v1"] == 0
    assert summary["status_v1"]["not_live_gate"] is True
    asof = pd.read_parquet(extension_dir / AS_OF_FEATURE_TABLE)
    hindsight = pd.read_parquet(extension_dir / HINDSIGHT_LABEL_OUTCOME_TABLE)
    prediction = pd.read_parquet(extension_dir / POLICY_PREDICTION_VIEW)
    assert len(asof) == 50
    assert len(hindsight) == 50
    assert len(prediction) == 50
    assert "pred__entry_r5_2_bad_blocker__prob_true_v1" in asof.columns
    assert "r6_label_bad_risk_v1" not in asof.columns
    assert "r6_label_bad_risk_v1" in hindsight.columns
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()


def test_r6_batch05_absent_is_not_reported_as_fail(tmp_path: Path) -> None:
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
        batch_weeks=2,
        expected_ledger_count=50,
    )
    freeze_dir = reports_root / "freeze"
    materialize_freeze(
        reports_root,
        phase_dir=phase_dir,
        extension_dir=freeze_dir,
        batch_weeks=2,
        expected_ledger_count=50,
        test_status="PYTEST_FIXTURE_PASS",
    )
    extension_dir = reports_root / "r6_compact"
    materialize(
        reports_root,
        freeze_dir=freeze_dir,
        extension_dir=extension_dir,
        batch_weeks=2,
        expected_ledger_count=50,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        compact_grid=True,
    )
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["selected_candidate_v1"]["batch05_loso_pass_v1"] is None
