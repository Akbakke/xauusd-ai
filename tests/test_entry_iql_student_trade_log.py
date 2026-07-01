import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_iql_distillation_contract_v1 import (
    IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS,
    _sha256_file,
)
from gx1.scripts.materialize_entry_iql_student_trade_log_v1 import (
    _apply_student_risk_filters,
    _row_selection_tuple,
    run,
)


def _write_ready_distillation_contract(path: Path) -> None:
    root = path.parent
    artifact_paths: dict[str, str] = {}
    for key in IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS:
        artifact = root / f"{key}.txt"
        artifact.write_text(f"{key}\n", encoding="utf-8")
        artifact_paths[key] = str(artifact.resolve())
    payload = {
        "decision": "ENTRY_IQL_DISTILLATION_CONTRACT_READY",
        "iql_research_distillation_allowed": True,
        "promotion_shadow_live_allowed": False,
        "artifact_paths": artifact_paths,
        "artifact_sha256": {key: _sha256_file(Path(value)) for key, value in artifact_paths.items()},
        "evidence_identity": {
            "candidate_bundle_dir": "/tmp/candidate_bundle",
            "selective_edge_bundle_dir": "/tmp/candidate_bundle",
            "replay_identity_candidate_bundle_dir": "/tmp/candidate_bundle",
            "replay_identity_ready": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_iql_student_trade_log_selects_validation_policy_and_writes_2026_trades(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    val_times = pd.date_range("2025-12-31T23:00:00Z", periods=6, freq="5min")
    test_times = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="5min")
    pd.DataFrame({"time": val_times, "label_horizon_bars": [1] * len(val_times)}).to_parquet(
        dataset_dir / "tiny_val.parquet",
        index=False,
    )
    pd.DataFrame({"time": test_times, "label_horizon_bars": [1] * len(test_times)}).to_parquet(
        dataset_dir / "tiny_test.parquet",
        index=False,
    )
    predictions = pd.DataFrame(
        {
            "split": ["val"] * len(val_times) + ["test"] * len(test_times),
            "model": ["candidate"] * (len(val_times) + len(test_times)),
            "time": list(val_times) + list(test_times),
            "y_direction": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "pred_direction": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "trade_side": [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "side": ["LONG", "SHORT"] * 7,
            "session": ["EU"] * 14,
            "vol_regime": ["1"] * 14,
            "edge_score": [0.90, 0.88, 0.86, 0.20, 0.10, 0.05, 0.95, 0.93, 0.91, 0.89, 0.30, 0.20, 0.10, 0.05],
            "p_long": [0.85, 0.10, 0.82, 0.20, 0.55, 0.10, 0.90, 0.88, 0.10, 0.12, 0.60, 0.58, 0.10, 0.12],
            "p_short": [0.10, 0.84, 0.12, 0.60, 0.20, 0.55, 0.06, 0.08, 0.86, 0.84, 0.20, 0.22, 0.70, 0.68],
            "p_flat": [0.05, 0.06, 0.06, 0.20, 0.25, 0.35, 0.04, 0.04, 0.04, 0.04, 0.20, 0.20, 0.20, 0.20],
            "path_quality_pred": [0.8] * 14,
            "bad_path_prob": [0.2] * 14,
        }
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    source_times = list(val_times) + list(test_times) + [test_times[-1] + pd.Timedelta(minutes=5)]
    source = pd.DataFrame(
        {
            "time": source_times,
            "bid_close": [100.00, 99.90, 100.20, 100.00, 100.10, 99.90, 100.00, 100.30, 100.60, 100.40, 100.70, 100.90, 100.60, 100.30, 100.10],
            "ask_close": [100.02, 99.92, 100.22, 100.02, 100.12, 99.92, 100.02, 100.32, 100.62, 100.42, 100.72, 100.92, 100.62, 100.32, 100.12],
            "bid_high": [100.05, 99.95, 100.25, 100.05, 100.15, 99.95, 100.05, 100.35, 100.65, 100.45, 100.75, 100.95, 100.65, 100.35, 100.15],
            "bid_low": [99.95, 99.85, 100.15, 99.95, 100.05, 99.85, 99.95, 100.25, 100.55, 100.35, 100.65, 100.85, 100.55, 100.25, 100.05],
            "ask_high": [100.07, 99.97, 100.27, 100.07, 100.17, 99.97, 100.07, 100.37, 100.67, 100.47, 100.77, 100.97, 100.67, 100.37, 100.17],
            "ask_low": [99.97, 99.87, 100.17, 99.97, 100.07, 99.87, 99.97, 100.27, 100.57, 100.37, 100.67, 100.87, 100.57, 100.27, 100.07],
        }
    )
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)
    contract = tmp_path / "distill.json"
    _write_ready_distillation_contract(contract)

    report = run(
        argparse.Namespace(
            vedtak="ENTRY_FOUNDATION_IQL_DISTILL_TEST_V1",
            distillation_contract_json=str(contract),
            selective_edge_predictions=str(predictions_path),
            dataset_dir=str(dataset_dir),
            source_parquet=str(source_path),
            out_dir=str(tmp_path / "out"),
            model_name="candidate",
            policy_id="entry_iql_student",
            threshold_top_fracs="0.50",
            cost_stress_bps="0.0",
            exit_mode="stop_tp_mfe_protect",
            stop_loss_bps=80.0,
            take_profit_bps_grid="80,120",
            daily_loss_limit_bps_grid="0",
            same_bar_policy="stop_first",
            mfe_protect_activation_bps=20.0,
            mfe_protect_breakeven_offset_bps=0.0,
            mfe_protect_trailing_capture_ratio=0.0,
            mfe_protect_trailing_floor_bps=0.0,
            cooldown_bars=0,
            max_trades_per_day=0,
            min_direction_prob=0.0,
            min_score_floor=0.0,
            slippage_bps=0.0,
            size_multiplier=1.0,
            selection_objective="mean",
            max_bad_path_prob_grid="none",
            min_path_quality_pred_grid="none",
            test_grid_diagnostics_limit=2,
            min_validation_trades=1,
            min_validation_profit_factor=0.0,
            max_validation_drawdown_bps=1000.0,
            max_abs_loss_bps=80.0,
            require_validation_positive_months=False,
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    trades = pd.read_csv(tmp_path / "out" / "entry_iql_student_trade_log.csv")
    assert report["decision"] == "PASS"
    assert report["student_policy_fit_started"] is True
    assert report["runtime_trainer_started"] is False
    assert report["adapter_built"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert report["test_grid_diagnostics"]["enabled"] is True
    assert report["test_grid_diagnostics"]["diagnostic_only_not_selection_criterion"] is True
    assert report["selected_policy"]["exit_mode"] == "stop_tp_mfe_protect"
    assert report["exit_policy_contract"]["offline_only"] is True
    assert report["exit_policy_contract"]["promotion_shadow_live_allowed"] is False
    assert set(trades["policy_id"]) == {"entry_iql_student"}
    assert set(trades["exit_mode"]) == {"stop_tp_mfe_protect"}
    assert trades["exit_policy_config_hash"].notna().all()
    assert trades["student_selected_exit_policy_config_hash"].notna().all()
    assert set(pd.to_datetime(trades["entry_time"], utc=True).dt.year) == {2026}
    assert "teacher_score" in trades.columns
    assert "state_session" in trades.columns
    diagnostics = json.loads((tmp_path / "out" / "entry_iql_student_test_grid_diagnostics.json").read_text())
    assert len(diagnostics) == 2
    assert {row["diagnostic_only_not_selection_criterion"] for row in diagnostics} == {True}


def test_iql_student_risk_filters_apply_bad_path_and_path_quality_veto() -> None:
    frame = pd.DataFrame(
        {
            "edge_score": [0.9, 0.8, 0.7, 0.6],
            "bad_path_prob": [0.10, 0.35, 0.20, 0.05],
            "path_quality_pred": [0.55, 0.80, 0.30, 0.95],
        }
    )

    filtered = _apply_student_risk_filters(
        frame,
        max_bad_path_prob=0.20,
        min_path_quality_pred=0.50,
    )

    assert filtered["edge_score"].tolist() == [0.9, 0.6]


def test_iql_student_selection_tiebreak_prefers_less_restrictive_daily_loss() -> None:
    args = argparse.Namespace(
        min_validation_trades=10,
        min_validation_profit_factor=1.05,
        max_validation_drawdown_bps=650.0,
        max_abs_loss_bps=80.0,
        require_validation_positive_months=True,
        selection_objective="net",
    )
    metrics = {
        "n_trades": 231,
        "net_sum_bps": 4161.731,
        "net_mean_bps": 18.016,
        "profit_factor": 11.95,
        "max_drawdown_bps": 45.0,
        "max_loss_bps": -45.0,
        "all_months_positive": True,
    }
    restrictive = {
        "validation_metrics": metrics,
        "daily_loss_limit_bps": 80.0,
        "validation_eligible_rows": 17_646,
    }
    less_restrictive = {
        "validation_metrics": metrics,
        "daily_loss_limit_bps": 150.0,
        "validation_eligible_rows": 17_646,
    }

    assert _row_selection_tuple(less_restrictive, args) > _row_selection_tuple(restrictive, args)
