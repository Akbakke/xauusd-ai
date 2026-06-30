import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_entry_candidate_replay_evidence_v1 import run as run_candidate_replay_evidence
from gx1.scripts.materialize_entry_iql_distillation_contract_v1 import (
    IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS,
    _sha256_file,
)
from gx1.scripts.materialize_entry_iql_replay_evidence_v1 import run as run_iql_replay_evidence
from gx1.scripts.verify_entry_iql_replay_comparison_v1 import run as run_iql_compare


CANDIDATE_BUNDLE_DIR = "/tmp/candidate_bundle"
BASE_SPECIALIST_GROUPS = [
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
]


def _raw_trades(net_values: list[float], *, policy_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": [
                "2026-01-03T08:00:00Z",
                "2026-01-04T09:00:00Z",
                "2026-02-03T10:00:00Z",
                "2026-02-04T11:00:00Z",
            ],
            "policy_id": [policy_id] * 4,
            "session": ["EU", "EU", "US", "US"],
            "side": ["LONG", "SHORT", "LONG", "SHORT"],
            "score": [0.8, 0.7, 0.75, 0.6],
            "p_long": [0.82, 0.10, 0.78, 0.25],
            "p_short": [0.10, 0.78, 0.12, 0.62],
            "p_flat": [0.08, 0.12, 0.10, 0.13],
            "net_pnl_bps": net_values,
            "gross_pnl_bps": net_values,
            "mfe_bps": [160.0, 20.0, 130.0, 55.0],
            "mae_bps": [10.0, 35.0, 12.0, 20.0],
            "held_bars": [12, 8, 10, 6],
        }
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_json_if_missing(path: Path, payload: dict) -> None:
    if not path.exists():
        _write_json(path, payload)


def _write_text_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _candidate_specialist_identity(contract_mode: str = "foundation_seq146") -> dict:
    groups = list(BASE_SPECIALIST_GROUPS)
    if contract_mode == "challenger_seq215":
        groups.extend(["chart_geometry_encoder", "price_action_candle_encoder"])
    return {
        "ready": True,
        "contract_mode": contract_mode,
        "bundle_specialist_groups": groups,
        "failures": [],
    }


def _selective_specialist_identity(contract_mode: str = "foundation_seq146") -> dict:
    groups = _candidate_specialist_identity(contract_mode)["bundle_specialist_groups"]
    return {
        "ready": True,
        "contract_mode": contract_mode,
        "candidate_bundle_specialist_contract": {"observed_specialists": groups},
        "failures": [],
    }


def _replay_specialist_identity_contract(contract_mode: str = "foundation_seq146") -> dict:
    return {
        "ok": True,
        "contract_mode": contract_mode,
        "candidate_specialist_contract": _candidate_specialist_identity(contract_mode),
        "selective_edge_specialist_contract": _selective_specialist_identity(contract_mode),
        "failures": [],
    }


def _replay_identity_contract(
    *,
    bundle_dir: str = CANDIDATE_BUNDLE_DIR,
    contract_mode: str = "foundation_seq146",
) -> dict:
    return {
        "ready": True,
        "contract_mode": contract_mode,
        "candidate_bundle_dir": bundle_dir,
        "candidate_specialist_contract": _candidate_specialist_identity(contract_mode),
        "selective_edge_specialist_contract": _selective_specialist_identity(contract_mode),
    }


def _specialist_snapshot(contract_mode: str = "foundation_seq146") -> dict:
    groups = _candidate_specialist_identity(contract_mode)["bundle_specialist_groups"]
    dim = 215 if contract_mode == "challenger_seq215" else 146
    return {
        "requested_contract_mode": contract_mode,
        "expected_signal_dim": dim,
        "bundle_seq_input_dim": dim,
        "bundle_snap_input_dim": dim,
        "expected_specialists": groups,
        "observed_specialists": groups,
        "specialist_fusion_enabled": True,
        "required_specialists_exact": True,
        "specialist_model_contract_valid": True,
        "specialist_model_contract_set_exact": True,
        "specialist_model_contract_owned_objectives_match": True,
        "specialist_model_contract_signal_families_match": True,
        "specialist_model_contract_support_heads_match": True,
        "specialist_model_contract_model_roles_match": True,
        "failures": [],
    }


def _candidate_audit_payload(contract_mode: str = "foundation_seq146") -> dict:
    groups = _candidate_specialist_identity(contract_mode)["bundle_specialist_groups"]
    dim = 215 if contract_mode == "challenger_seq215" else 146
    return {
        "decision": "PASS",
        "contract_mode": contract_mode,
        "specialist_contract_mode": contract_mode,
        "bundle_dir": CANDIDATE_BUNDLE_DIR,
        "required_training_specialists": groups,
        "bundle_summary": {
            "contract_mode": contract_mode,
            "specialist_contract_mode": contract_mode,
            "seq_input_dim": dim,
            "snap_input_dim": dim,
            "specialist_groups": groups,
            "specialist_fusion_enabled": True,
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


def _selective_summary_payload(contract_mode: str = "foundation_seq146") -> dict:
    dim = 215 if contract_mode == "challenger_seq215" else 146
    snapshot = _specialist_snapshot(contract_mode)
    return {
        "decision": "PASS",
        "contract_mode": contract_mode,
        "bundle_dir": CANDIDATE_BUNDLE_DIR,
        "no_xgb_bundle_dir": "/tmp/candidate_no_xgb",
        "bundle_seq_input_dim": dim,
        "bundle_snap_input_dim": dim,
        "bundle_specialist_contract": snapshot,
        "no_xgb_bundle_specialist_contract": snapshot,
    }


def _write_candidate_identity_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    candidate_audit = tmp_path / "candidate_audit.json"
    selective_summary = tmp_path / "selective_summary.json"
    _write_json(candidate_audit, _candidate_audit_payload())
    _write_json(selective_summary, _selective_summary_payload())
    return candidate_audit, selective_summary


def _write_distillation_contract(path: Path, *, ready: bool = True) -> None:
    root = path.parent
    candidate_replay_dir = root / "candidate"
    candidate_replay_manifest = candidate_replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    _write_json_if_missing(root / "candidate_readiness.json", {"decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK"})
    _write_json_if_missing(root / "candidate_audit.json", _candidate_audit_payload())
    _write_json_if_missing(root / "selective_summary.json", _selective_summary_payload())
    _write_text_if_missing(root / "selective_metrics.csv", "split,model\nval,candidate\n")
    _write_json_if_missing(
        candidate_replay_manifest,
        {
            "decision": "PASS",
            "replay_identity_contract": _replay_identity_contract(),
        },
    )
    _write_text_if_missing(
        candidate_replay_dir / "replay_policy_metrics.csv",
        "scope,policy_id,n_trades,net_sum_bps,win_rate,profit_factor,max_drawdown_bps,max_loss_bps\n"
        "aggregate,candidate_replay,4,100,0.75,1.5,-40,-20\n",
    )
    _write_text_if_missing(
        candidate_replay_dir / "replay_policy_monthly.csv",
        "policy_id,month,net_sum_bps\ncandidate_replay,2026-01,50\ncandidate_replay,2026-02,50\n",
    )
    _write_text_if_missing(
        candidate_replay_dir / "replay_policy_trades.csv",
        "entry_time,policy_id,session,side,score,p_long,p_short,p_flat,net_pnl_bps,mfe_bps,mae_bps,held_bars\n"
        "2026-01-03T08:00:00Z,candidate_replay,EU,LONG,0.8,0.8,0.1,0.1,50,80,10,8\n",
    )
    replay_readiness = root / "replay_readiness.json"
    _write_json_if_missing(
        replay_readiness,
        {
            "decision": "READY_FOR_IQL_DISTILLATION_VEDTAK",
            "iql_distillation_allowed_with_explicit_vedtak": True,
            "promotion_shadow_live_allowed": False,
        },
    )
    artifact_paths = {
        "replay_readiness": str(replay_readiness.resolve()),
        "candidate_readiness": str((root / "candidate_readiness.json").resolve()),
        "candidate_bundle_audit": str((root / "candidate_audit.json").resolve()),
        "selective_edge_summary": str((root / "selective_summary.json").resolve()),
        "selective_edge_metrics": str((root / "selective_metrics.csv").resolve()),
        "candidate_replay_manifest": str(candidate_replay_manifest.resolve()),
        "candidate_replay_metrics": str((candidate_replay_dir / "replay_policy_metrics.csv").resolve()),
        "candidate_replay_monthly": str((candidate_replay_dir / "replay_policy_monthly.csv").resolve()),
        "candidate_replay_trades": str((candidate_replay_dir / "replay_policy_trades.csv").resolve()),
    }
    _write_json(
        path,
        {
            "schema_version": "entry_iql_distillation_contract_v1",
            "decision": "ENTRY_IQL_DISTILLATION_CONTRACT_READY" if ready else "ENTRY_IQL_DISTILLATION_CONTRACT_NOT_READY",
            "promotion_shadow_live_allowed": False,
            "candidate_pretrain_provenance_contract": {
                "found": True,
                "ok": True,
                "check": {
                    "name": "candidate bundle audit validated pre-train manifest provenance",
                    "ok": True,
                },
            },
            "smoke_dataset_provenance_contract": {
                "ok": True,
                "smoke_dataset_audit_provenance_all_artifacts_present": True,
                "smoke_dataset_audit_provenance_all_artifact_hashes_present": True,
                "smoke_edge_worktree_critical_gate_review_ok": True,
                "failures": [],
            },
            "specialist_set_provenance_contract": {
                "ok": True,
                "specialist_required_training_set_exact": True,
                "specialist_trainable_set_exact": True,
                "smoke_edge_required_specialists_exact": True,
                "smoke_edge_specialist_groups_exact": True,
                "failures": [],
            },
            "specialist_model_provenance_contract": {
                "ok": True,
                "specialist_model_contract_valid": True,
                "specialist_model_contract_set_exact": True,
                "specialist_model_contract_owned_objectives_match": True,
                "smoke_edge_specialist_model_contract_valid": True,
                "smoke_edge_specialist_model_contract_set_exact": True,
                "smoke_edge_specialist_model_contract_owned_objectives_match": True,
                "failures": [],
            },
            "bundle_specialist_model_provenance_contract": {
                "ok": True,
                "bundle_summary": {
                    "specialist_model_contract_declared_valid": True,
                    "specialist_model_contract_valid": True,
                    "specialist_model_contract_set_exact": True,
                    "specialist_model_contract_owned_objectives_match": True,
                    "specialist_model_contract_support_heads_match": True,
                    "specialist_model_contract_signal_families_match": True,
                    "specialist_model_contract_model_roles_match": True,
                },
                "bundle_specialist_model_contract": {
                    "decision": "PASS",
                    "valid": True,
                    "set_exact": True,
                    "owned_objectives_match": True,
                    "support_heads_match": True,
                    "signal_families_match": True,
                    "model_roles_match": True,
                    "failures": [],
                },
                "failures": [],
            },
            "replay_artifact_provenance_contract": {
                "ok": True,
                "gate_decision": "PASS",
                "failures": [],
            },
            "replay_specialist_identity_contract": _replay_specialist_identity_contract(),
            "artifact_paths": artifact_paths,
            "artifact_sha256": {
                key: _sha256_file(Path(artifact_paths[key])) for key in IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS
            },
            "evidence_identity": {
                "candidate_bundle_audit_json": str(path.parent / "candidate_audit.json"),
                "selective_edge_summary_json": str(path.parent / "selective_summary.json"),
                "replay_evidence_manifest_json": str(candidate_replay_manifest),
                "candidate_bundle_dir": CANDIDATE_BUNDLE_DIR,
                "selective_edge_bundle_dir": CANDIDATE_BUNDLE_DIR,
                "replay_identity_candidate_bundle_dir": CANDIDATE_BUNDLE_DIR,
                "no_xgb_bundle_dir": "/tmp/candidate_no_xgb",
                "replay_identity_ready": True,
                "candidate_specialist_contract": _candidate_specialist_identity(),
                "selective_edge_specialist_contract": _selective_specialist_identity(),
                "candidate_specialist_contract_ready": True,
                "selective_edge_specialist_contract_ready": True,
            },
        },
    )


def _write_candidate_replay_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "candidate" / "REPLAY_EVIDENCE_MANIFEST.json"
    _write_json(
        manifest,
        {
            "decision": "PASS",
            "replay_identity_contract": _replay_identity_contract(),
        },
    )
    return manifest


def test_iql_replay_evidence_run_writes_manifest_and_identity(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    _write_candidate_replay_manifest(tmp_path)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    manifest = json.loads((tmp_path / "iql" / "REPLAY_EVIDENCE_MANIFEST.json").read_text())
    assert report["decision"] == "PASS"
    assert report["trainer_started"] is False
    assert report["replay_started"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert manifest["replay_identity_contract"]["candidate_bundle_dir"] == CANDIDATE_BUNDLE_DIR
    assert manifest["replay_identity_contract"]["ready"] is True
    assert manifest["replay_identity_contract"]["distillation_artifact_hash_contract"]["ready"] is True


def test_iql_replay_evidence_fails_on_wrong_policy_id(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="candidate_replay").to_csv(trades_path, index=False)
    _write_candidate_replay_manifest(tmp_path)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("policy_id must match" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_distillation_contract_not_ready(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract, ready=False)

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("decision is not ready" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_distillation_pretrain_provenance_missing(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["candidate_pretrain_provenance_contract"] = {"found": True, "ok": False}
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("pretrain provenance" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_smoke_dataset_provenance_missing(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["smoke_dataset_provenance_contract"] = {"ok": False, "failures": ["missing smoke dataset provenance"]}
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("smoke dataset audit provenance" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_specialist_set_provenance_missing(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["specialist_set_provenance_contract"] = {
        "ok": False,
        "failures": ["missing exact specialist set provenance"],
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("exact specialist set provenance" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_specialist_model_provenance_missing(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["specialist_model_provenance_contract"] = {
        "ok": False,
        "failures": ["missing specialist model contract provenance"],
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("specialist model contract provenance" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_bundle_specialist_model_provenance_missing(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["bundle_specialist_model_provenance_contract"] = {
        "ok": False,
        "failures": ["missing candidate bundle specialist model contract provenance"],
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("candidate bundle specialist model contract provenance" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_replay_artifact_provenance_missing(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["replay_artifact_provenance_contract"] = {"ok": False, "failures": ["missing replay artifact hash"]}
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("replay artifact provenance" in failure for failure in report["failures"])


def test_iql_replay_evidence_fails_when_distillation_hash_mismatches(tmp_path: Path) -> None:
    trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(trades_path, index=False)
    contract = tmp_path / "distill.json"
    _write_distillation_contract(contract)
    (tmp_path / "candidate_readiness.json").write_text('{"decision":"MUTATED"}', encoding="utf-8")

    report = run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(trades_path),
            out_dir=str(tmp_path / "iql"),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=False,
            quiet=True,
        )
    )

    assert report["decision"] == "FAIL"
    assert any("artifact hash mismatch" in failure for failure in report["failures"])


def test_iql_replay_evidence_output_can_feed_iql_compare(tmp_path: Path) -> None:
    candidate_audit, selective_summary = _write_candidate_identity_artifacts(tmp_path)
    candidate_trades_path = tmp_path / "candidate_trades.csv"
    iql_trades_path = tmp_path / "iql_trades.csv"
    _raw_trades([60.0, -20.0, 50.0, 10.0], policy_id="candidate_replay").to_csv(candidate_trades_path, index=False)
    _raw_trades([150.0, -10.0, 100.0, 40.0], policy_id="entry_iql_student").to_csv(iql_trades_path, index=False)
    candidate_dir = tmp_path / "candidate"
    iql_dir = tmp_path / "iql"
    contract = tmp_path / "distill.json"

    run_candidate_replay_evidence(
        argparse.Namespace(
            trades_path=str(candidate_trades_path),
            out_dir=str(candidate_dir),
            candidate_bundle_audit_json=str(candidate_audit),
            selective_edge_summary_json=str(selective_summary),
            policy_id="candidate_replay",
            require_year=2026,
            allow_non_2026=False,
            require_iql_transition_fields=True,
            require_identity_artifacts=True,
            fail_on_audit_fail=True,
            quiet=True,
        )
    )
    _write_distillation_contract(contract)
    run_iql_replay_evidence(
        argparse.Namespace(
            trades_path=str(iql_trades_path),
            out_dir=str(iql_dir),
            distillation_contract_json=str(contract),
            policy_id="entry_iql_student",
            require_year=2026,
            allow_non_2026=False,
            require_policy_id=True,
            require_iql_transition_fields=True,
            fail_on_audit_fail=True,
            quiet=True,
        )
    )

    report = run_iql_compare(
        argparse.Namespace(
            distillation_contract_json=str(contract),
            candidate_replay_dir=str(candidate_dir),
            iql_replay_dir=str(iql_dir),
            out_dir=str(tmp_path / "comparison"),
            min_net_lift_bps=0.0,
            min_iql_profit_factor=1.05,
            min_profit_factor_lift=0.0,
            max_drawdown_worsening_bps=0.0,
            max_loss_worsening_bps=10.0,
            fail_on_not_ready=True,
            quiet=True,
        )
    )

    assert report["decision"] == "READY_FOR_PROMOTION_REVIEW_VEDTAK"
    assert report["evidence_identity"]["candidate_bundle_dir"] == CANDIDATE_BUNDLE_DIR
