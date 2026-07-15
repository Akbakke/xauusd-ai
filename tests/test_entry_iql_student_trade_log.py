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
    source_times = list(pd.date_range(val_times[0], test_times[-1] + pd.Timedelta(minutes=5), freq="5min"))
    bid_close = [100.0 + ((idx % 6) - 2) * 0.05 for idx in range(len(source_times))]
    bid_close[4] = 99.95
    ask_close = [value + 0.02 for value in bid_close]
    source = pd.DataFrame(
        {
            "time": source_times,
            "bid_open": bid_close,
            "ask_open": ask_close,
            "bid_close": bid_close,
            "ask_close": ask_close,
            "bid_high": [value + 0.05 for value in bid_close],
            "bid_low": [value - 0.05 for value in bid_close],
            "ask_high": [value + 0.05 for value in ask_close],
            "ask_low": [value - 0.05 for value in ask_close],
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
    validation_trades = pd.read_csv(tmp_path / "out" / "entry_iql_student_validation_trade_log.csv")
    assert report["decision"] == "PASS"
    assert report["student_policy_fit_started"] is True
    assert report["runtime_trainer_started"] is False
    assert report["adapter_built"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert report["validation_trade_log_diagnostic_only_not_replay_evidence"] is True
    assert report["validation_trade_log_replay_evidence_allowed"] is False
    assert Path(report["validation_trades_path"]).is_file()
    assert report["selected_validation_trade_count"] == len(validation_trades)
    assert report["selected_validation_counts"]["trades"] == len(validation_trades)
    assert report["selected_validation_trade_metrics"]["n_trades"] == len(validation_trades)
    assert report["test_grid_diagnostics"]["enabled"] is True
    assert report["test_grid_diagnostics"]["diagnostic_only_not_selection_criterion"] is True
    assert report["selected_policy"]["exit_mode"] == "stop_tp_mfe_protect"
    assert report["exit_policy_contract"]["offline_only"] is True
    assert report["exit_policy_contract"]["promotion_shadow_live_allowed"] is False
    assert set(trades["policy_id"]) == {"entry_iql_student"}
    assert set(validation_trades["policy_id"]) == {"entry_iql_student"}
    assert set(trades["exit_mode"]) == {"stop_tp_mfe_protect"}
    assert set(validation_trades["exit_mode"]) == {"stop_tp_mfe_protect"}
    assert set(trades["student_trade_log_split"]) == {"test"}
    assert set(validation_trades["student_trade_log_split"]) == {"validation"}
    assert set(trades["diagnostic_only_not_replay_evidence"]) == {False}
    assert set(validation_trades["diagnostic_only_not_replay_evidence"]) == {True}
    assert trades["exit_policy_config_hash"].notna().all()
    assert validation_trades["exit_policy_config_hash"].notna().all()
    assert trades["student_selected_exit_policy_config_hash"].notna().all()
    assert validation_trades["student_selected_exit_policy_config_hash"].notna().all()
    assert set(pd.to_datetime(trades["entry_time"], utc=True).dt.year) == {2026}
    assert set(pd.to_datetime(validation_trades["entry_time"], utc=True).dt.year) == {2025}
    assert "teacher_score" in trades.columns
    assert "teacher_score" in validation_trades.columns
    assert "state_session" in trades.columns
    assert "state_session" in validation_trades.columns
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
