import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.features.entry_specialist_feature_groups_v1 import required_training_specialists_for_mode
from gx1.scripts.materialize_entry_candidate_replay_evidence_v1 import (
    _identity_contract,
    audit_iql_transition_trades,
    build_replay_slices,
    build_replay_tables,
    normalize_trades,
    run,
)
from gx1.scripts.verify_entry_replay_readiness_v1 import _replay_checks


def _specialists(mode: str = "foundation_seq146") -> list[str]:
    return sorted(required_training_specialists_for_mode(mode))


def _specialist_snapshot(mode: str = "foundation_seq146") -> dict:
    expected = _specialists(mode)
    dim = 215 if mode == "challenger_seq215" else 146
    return {
        "requested_contract_mode": mode,
        "observed_contract_mode": mode,
        "expected_signal_dim": dim,
        "bundle_seq_input_dim": dim,
        "bundle_snap_input_dim": dim,
        "specialist_fusion_enabled": True,
        "expected_specialists": expected,
        "observed_specialists": expected,
        "required_specialists_exact": True,
        "chart_geometry_present": "chart_geometry_encoder" in expected,
        "price_action_candle_present": "price_action_candle_encoder" in expected,
        "specialist_model_contract_valid": True,
        "specialist_model_contract_set_exact": True,
        "specialist_model_contract_owned_objectives_match": True,
        "specialist_model_contract_signal_families_match": True,
        "specialist_model_contract_support_heads_match": True,
        "specialist_model_contract_model_roles_match": True,
        "failures": [],
    }


def _candidate_audit(bundle_dir: str = "/tmp/candidate_bundle", mode: str = "foundation_seq146") -> dict:
    specialists = _specialists(mode)
    dim = 215 if mode == "challenger_seq215" else 146
    return {
        "decision": "PASS",
        "bundle_dir": bundle_dir,
        "specialist_contract_mode": mode,
        "required_training_specialists": specialists,
        "bundle_summary": {
            "contract_mode": mode,
            "seq_input_dim": dim,
            "snap_input_dim": dim,
            "specialist_fusion_enabled": True,
            "specialist_groups": specialists,
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "specialist_model_contract_support_heads_match": True,
            "specialist_model_contract_signal_families_match": True,
            "specialist_model_contract_model_roles_match": True,
        },
        "bundle_specialist_model_contract": {
            "valid": True,
            "set_exact": True,
            "owned_objectives_match": True,
            "support_heads_match": True,
            "signal_families_match": True,
            "model_roles_match": True,
            "failures": [],
        },
    }


def _selective_summary(bundle_dir: str = "/tmp/candidate_bundle", mode: str = "foundation_seq146") -> dict:
    dim = 215 if mode == "challenger_seq215" else 146
    return {
        "decision": "PASS",
        "contract_mode": mode,
        "bundle_dir": bundle_dir,
        "bundle_seq_input_dim": dim,
        "bundle_snap_input_dim": dim,
        "no_xgb_bundle_dir": "/tmp/candidate_no_xgb",
        "bundle_specialist_contract": _specialist_snapshot(mode),
        "no_xgb_bundle_specialist_contract": _specialist_snapshot(mode),
    }


def _raw_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": [
                "2026-01-03T08:00:00Z",
                "2026-01-04T09:00:00Z",
                "2026-02-03T10:00:00Z",
                "2026-02-04T11:00:00Z",
            ],
            "policy_id": ["candidate_top5"] * 4,
            "session": ["EU", "EU", "US", "US"],
            "vol_regime": ["2", "2", "3", "3"],
            "side": ["LONG", "SHORT", "LONG", "SHORT"],
            "label": ["LONG", "SHORT", "LONG", "LONG"],
            "score": [0.8, 0.7, 0.75, 0.6],
            "p_long": [0.82, 0.10, 0.78, 0.25],
            "p_short": [0.10, 0.78, 0.12, 0.62],
            "p_flat": [0.08, 0.12, 0.10, 0.13],
            "net_pnl_bps": [120.0, -20.0, 90.0, 30.0],
            "gross_pnl_bps": [125.0, -15.0, 95.0, 35.0],
            "mfe_bps": [140.0, 10.0, 110.0, 50.0],
            "mae_bps": [10.0, 35.0, 12.0, 20.0],
            "held_bars": [12, 8, 10, 6],
            "bad_path_prob": [0.10, 0.70, 0.20, 0.35],
            "path_quality_pred": [0.90, 0.30, 0.80, 0.60],
            "foundation_bos_age_long": [3, 12, 4, 8],
            "specialist_structure_gate": [0.21, 0.18, 0.25, 0.17],
        }
    )


def _write_identity_artifacts(tmp_path: Path, *, bundle_dir: str = "/tmp/candidate_bundle") -> tuple[Path, Path]:
    candidate_audit = tmp_path / "candidate_audit.json"
    selective_summary = tmp_path / "selective_summary.json"
    candidate_audit.write_text(json.dumps(_candidate_audit(bundle_dir=bundle_dir)), encoding="utf-8")
    selective_summary.write_text(json.dumps(_selective_summary(bundle_dir=bundle_dir)), encoding="utf-8")
    return candidate_audit, selective_summary


def test_normalize_trades_requires_2026_and_derives_fields() -> None:
    trades, failures = normalize_trades(_raw_trades(), policy_id="candidate_top5", require_year=2026, allow_non_2026=False)

    assert failures == []
    assert list(trades["entry_month"].unique()) == ["2026-01", "2026-02"]
    assert "direction_correct" in trades.columns
    assert "vol_regime" in trades.columns
    assert "tail_bucket" in trades.columns
    assert "bad_path_bucket" in trades.columns
    assert "foundation_bos_age_long" in trades.columns
    assert "specialist_structure_gate" in trades.columns
    assert int(trades["direction_correct"].sum()) == 3


def test_iql_transition_audit_requires_teacher_state_action_reward_columns() -> None:
    trades, _ = normalize_trades(_raw_trades(), policy_id="candidate_top5", require_year=2026, allow_non_2026=False)

    audit = audit_iql_transition_trades(trades)

    assert audit["ready"] is True
    assert audit["missing_columns"] == []
    assert audit["probability_sum_max_abs_error"] < 1e-9


def test_iql_transition_audit_fails_when_probabilities_are_missing() -> None:
    raw = _raw_trades().drop(columns=["p_long", "p_short", "p_flat"])
    trades, _ = normalize_trades(raw, policy_id="candidate_top5", require_year=2026, allow_non_2026=False)

    audit = audit_iql_transition_trades(trades)

    assert audit["ready"] is False
    assert set(audit["missing_columns"]) == {"p_long", "p_short", "p_flat"}


def test_build_replay_tables_matches_replay_readiness_contract(tmp_path: Path) -> None:
    trades, _ = normalize_trades(_raw_trades(), policy_id="candidate_top5", require_year=2026, allow_non_2026=False)
    metrics, _daily, monthly = build_replay_tables(trades)
    slices = build_replay_slices(trades)

    assert {"policy_id", "n_trades", "net_sum_bps", "win_rate", "profit_factor", "max_drawdown_bps", "max_loss_bps"}.issubset(metrics.columns)
    assert "month" in monthly.columns
    assert {"session", "regime", "direction", "tail", "bad_path"}.issubset(
        set(slices["slice_dimension"])
    )
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    candidate_audit, selective_summary = _write_identity_artifacts(tmp_path)
    manifest = {
        "decision": "PASS",
        "failures": [],
        "replay_identity_contract": _identity_contract(
            candidate_bundle_audit_path=candidate_audit,
            selective_edge_summary_path=selective_summary,
            require_identity_artifacts=True,
        ),
    }
    checks = _replay_checks(
        replay_dir,
        manifest,
        metrics,
        monthly,
        trades,
        min_net_sum_bps=0.0,
        min_profit_factor=1.05,
        max_drawdown_bps=650.0,
        expected_candidate_bundle_dir="/tmp/candidate_bundle",
    )
    assert all(check["ok"] for check in checks)


def test_identity_contract_rejects_mismatched_selective_edge_bundle(tmp_path: Path) -> None:
    candidate_audit, selective_summary = _write_identity_artifacts(tmp_path, bundle_dir="/tmp/candidate_bundle")
    selective_summary.write_text(
        json.dumps({"decision": "PASS", "bundle_dir": "/tmp/other_bundle", "no_xgb_bundle_dir": "/tmp/no_xgb"}),
        encoding="utf-8",
    )

    contract = _identity_contract(
        candidate_bundle_audit_path=candidate_audit,
        selective_edge_summary_path=selective_summary,
        require_identity_artifacts=True,
    )

    assert contract["ready"] is False
    assert any("does not match" in failure for failure in contract["failures"])


def test_replay_evidence_run_writes_readiness_files(tmp_path: Path) -> None:
    trades_path = tmp_path / "trades.csv"
    _raw_trades().to_csv(trades_path, index=False)
    out_dir = tmp_path / "out"
    candidate_audit, selective_summary = _write_identity_artifacts(tmp_path)

    report = run(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(out_dir),
            candidate_bundle_audit_json=str(candidate_audit),
            selective_edge_summary_json=str(selective_summary),
            policy_id="candidate_top5",
            require_year=2026,
            allow_non_2026=False,
            require_iql_transition_fields=True,
            require_identity_artifacts=True,
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    assert report["decision"] == "PASS"
    assert report["replay_started"] is False
    assert report["iql_transition_dataset_ready"] is True
    assert report["replay_identity_contract"]["ready"] is True
    assert report["replay_identity_contract"]["candidate_bundle_audit_sha256"]
    assert report["replay_identity_contract"]["selective_edge_summary_sha256"]
    assert report["promotion_shadow_live_allowed"] is False
    assert (out_dir / "replay_policy_metrics.csv").exists()
    assert (out_dir / "replay_policy_monthly.csv").exists()
    assert (out_dir / "replay_policy_slices.csv").exists()
    assert (out_dir / "replay_policy_slices.csv").stat().st_size > 0
    assert "replay_policy_slices.csv" in report["artifact_hashes"]
    assert json.loads((out_dir / "summary.json").read_text())["decision"] == "PASS"


def test_replay_evidence_fails_on_non_2026_rows(tmp_path: Path) -> None:
    bad = _raw_trades()
    bad.loc[0, "entry_time"] = "2025-12-31T08:00:00Z"
    trades_path = tmp_path / "trades.csv"
    bad.to_csv(trades_path, index=False)

    report = run(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "out"),
            candidate_bundle_audit_json=str(tmp_path / "missing_candidate_audit.json"),
            selective_edge_summary_json=str(tmp_path / "missing_selective_summary.json"),
            policy_id="candidate_top5",
            require_year=2026,
            allow_non_2026=False,
            require_iql_transition_fields=True,
            require_identity_artifacts=False,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("outside 2026" in failure for failure in report["failures"])
