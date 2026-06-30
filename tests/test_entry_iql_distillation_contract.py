import argparse
import hashlib
import json
from pathlib import Path

from gx1.scripts.materialize_entry_iql_distillation_contract_v1 import IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS, run


PRETRAIN_PROVENANCE_GATE = {
    "name": "candidate_bundle_audit",
    "decision": "PASS",
    "checks": [
        {
            "name": "candidate bundle audit validated pre-train manifest provenance",
            "ok": True,
            "details": {
                "pretrain_manifest_contract": {
                    "decision": "PASS",
                    "feature_source_field_liveness_all_live": True,
                    "specialist_active_heads_match_target": True,
                    "specialist_blocked_heads_match_target": True,
                    "specialist_required_training_set_exact": True,
                    "specialist_trainable_set_exact": True,
                    "specialist_model_contract_valid": True,
                    "specialist_model_contract_set_exact": True,
                    "specialist_model_contract_owned_objectives_match": True,
                    "smoke_edge_required_specialists_exact": True,
                    "smoke_edge_specialist_groups_exact": True,
                    "smoke_edge_specialist_model_contract_valid": True,
                    "smoke_edge_specialist_model_contract_set_exact": True,
                    "smoke_edge_specialist_model_contract_owned_objectives_match": True,
                    "smoke_dataset_audit_provenance_all_artifacts_present": True,
                    "smoke_dataset_audit_provenance_all_artifact_hashes_present": True,
                    "smoke_edge_worktree_critical_gate_review_ok": True,
                }
            },
        }
    ],
}

BUNDLE_SPECIALIST_MODEL_GATE = {
    "name": "candidate_bundle_audit",
    "decision": "PASS",
    "checks": [
        {
            "name": "candidate bundle specialist model contract is preserved in bundle metadata",
            "ok": True,
            "details": {
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
            },
        }
    ],
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": digest,
    }


BASE_SPECIALIST_GROUPS = [
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
]


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


def _replay_identity_contract(
    *,
    bundle_dir: str = "/tmp/candidate_bundle",
    contract_mode: str = "foundation_seq146",
) -> dict:
    return {
        "ready": True,
        "contract_mode": contract_mode,
        "candidate_bundle_dir": bundle_dir,
        "candidate_specialist_contract": _candidate_specialist_identity(contract_mode),
        "selective_edge_specialist_contract": _selective_specialist_identity(contract_mode),
    }


def _replay_artifact_fingerprints(
    *,
    candidate: Path,
    candidate_audit: Path,
    selective_summary: Path,
    selective_metrics: Path,
    replay_manifest: Path,
    replay_dir: Path,
) -> dict:
    return {
        "candidate_readiness": _fingerprint(candidate),
        "candidate_bundle_audit": _fingerprint(candidate_audit),
        "selective_edge_summary": _fingerprint(selective_summary),
        "selective_edge_metrics": _fingerprint(selective_metrics),
        "candidate_replay_manifest": _fingerprint(replay_manifest),
        "candidate_replay_metrics": _fingerprint(replay_dir / "replay_policy_metrics.csv"),
        "candidate_replay_monthly": _fingerprint(replay_dir / "replay_policy_monthly.csv"),
        "candidate_replay_trades": _fingerprint(replay_dir / "replay_policy_trades.csv"),
    }


def test_iql_distillation_contract_blocks_current_not_ready_artifact(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            vedtak="PYTEST",
            replay_readiness_json="/home/andre2/GX1_DATA/reports/entry_replay_readiness_20260628_v1/ENTRY_REPLAY_READINESS_latest.json",
            out_dir=str(tmp_path),
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "ENTRY_IQL_DISTILLATION_CONTRACT_NOT_READY"
    assert report["iql_research_distillation_allowed"] is False
    assert report["trainer_started"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert Path(report["json_path"]).exists()


def test_iql_distillation_contract_opens_on_replay_ready_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate_audit = tmp_path / "candidate_audit.json"
    selective_summary = tmp_path / "selective" / "summary.json"
    selective_metrics = tmp_path / "selective" / "selective_edge_metrics.csv"
    replay_dir = tmp_path / "replay"
    replay_manifest = replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    _write_json(candidate, {"decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK"})
    _write_json(candidate_audit, {"decision": "PASS", "bundle_dir": "/tmp/candidate_bundle"})
    _write_json(selective_summary, {"splits": ["val", "test"]})
    selective_metrics.parent.mkdir(parents=True, exist_ok=True)
    selective_metrics.write_text("split,model\nval,candidate\n", encoding="utf-8")
    replay_dir.mkdir(parents=True)
    (replay_dir / "replay_policy_metrics.csv").write_text(
        "scope,policy_id,n_trades,net_sum_bps,win_rate,profit_factor,max_drawdown_bps,max_loss_bps\n"
        "aggregate,candidate_replay,4,200,0.75,1.5,-50,-20\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_monthly.csv").write_text(
        "policy_id,month,net_sum_bps\ncandidate_replay,2026-01,100\ncandidate_replay,2026-02,100\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_trades.csv").write_text(
        "entry_time,policy_id,session,side,score,p_long,p_short,p_flat,net_pnl_bps,mfe_bps,mae_bps,held_bars\n"
        "2026-01-03T08:00:00Z,candidate_replay,EU,LONG,0.8,0.8,0.1,0.1,100,120,10,8\n",
        encoding="utf-8",
    )
    _write_json(
        replay_manifest,
        {
            "decision": "PASS",
            "replay_identity_contract": _replay_identity_contract(),
        },
    )
    ready = tmp_path / "ENTRY_REPLAY_READINESS_latest.json"
    replay_fingerprints = _replay_artifact_fingerprints(
        candidate=candidate,
        candidate_audit=candidate_audit,
        selective_summary=selective_summary,
        selective_metrics=selective_metrics,
        replay_manifest=replay_manifest,
        replay_dir=replay_dir,
    )
    _write_json(
        ready,
        {
            "decision": "READY_FOR_IQL_DISTILLATION_VEDTAK",
            "iql_distillation_allowed_with_explicit_vedtak": True,
            "promotion_shadow_live_allowed": False,
            "candidate_readiness_json": str(candidate),
            "candidate_bundle_audit_json": str(candidate_audit),
            "selective_edge_summary_json": str(selective_summary),
            "selective_edge_metrics_csv": str(selective_metrics),
            "replay_dir": str(replay_dir),
            "evidence_identity": {
                "candidate_bundle_audit_json": str(candidate_audit),
                "selective_edge_summary_json": str(selective_summary),
                "replay_evidence_manifest_json": str(replay_manifest),
                "candidate_bundle_dir": "/tmp/candidate_bundle",
                "selective_edge_bundle_dir": "/tmp/candidate_bundle",
                "replay_identity_candidate_bundle_dir": "/tmp/candidate_bundle",
                "no_xgb_bundle_dir": "/tmp/no_xgb_bundle",
                "replay_identity_ready": True,
                "candidate_specialist_contract": _candidate_specialist_identity(),
                "selective_edge_specialist_contract": _selective_specialist_identity(),
                "candidate_specialist_contract_ready": True,
                "selective_edge_specialist_contract_ready": True,
            },
            "artifact_fingerprints": replay_fingerprints,
            "gates": [
                PRETRAIN_PROVENANCE_GATE,
                BUNDLE_SPECIALIST_MODEL_GATE,
                {"name": "artifact_provenance", "decision": "PASS", "checks": []},
            ],
        },
    )

    report = run(
        argparse.Namespace(
            vedtak="PYTEST_READY",
            replay_readiness_json=str(ready),
            out_dir=str(tmp_path / "out"),
            fail_on_not_ready=True,
            quiet=True,
        )
    )

    assert report["decision"] == "ENTRY_IQL_DISTILLATION_CONTRACT_READY"
    assert report["iql_research_distillation_allowed"] is True
    assert report["adapter_built"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert report["evidence_identity"]["candidate_bundle_dir"] == "/tmp/candidate_bundle"
    assert set(IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS).issubset(report["artifact_sha256"])
    assert all(row["ok"] for row in report["artifact_hash_checks"].values())
    assert report["replay_artifact_provenance_contract"]["ok"] is True
    assert report["replay_specialist_identity_contract"]["ok"] is True
    assert report["smoke_dataset_provenance_contract"]["ok"] is True
    assert report["specialist_set_provenance_contract"]["ok"] is True
    assert report["specialist_model_provenance_contract"]["ok"] is True
    assert report["bundle_specialist_model_provenance_contract"]["ok"] is True
    assert {task["id"] for task in report["distillation_tasks"]} == {
        "entry_transformer_teacher",
        "replay_reward_critic",
        "entry_iql_student",
        "post_distillation_replay_compare",
    }


def test_iql_distillation_contract_rejects_missing_pretrain_provenance_gate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate_audit = tmp_path / "candidate_audit.json"
    selective_summary = tmp_path / "selective" / "summary.json"
    selective_metrics = tmp_path / "selective" / "selective_edge_metrics.csv"
    replay_dir = tmp_path / "replay"
    replay_manifest = replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    _write_json(candidate, {"decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK"})
    _write_json(candidate_audit, {"decision": "PASS", "bundle_dir": "/tmp/candidate_bundle"})
    _write_json(selective_summary, {"splits": ["val", "test"]})
    selective_metrics.parent.mkdir(parents=True, exist_ok=True)
    selective_metrics.write_text("split,model\nval,candidate\n", encoding="utf-8")
    replay_dir.mkdir(parents=True)
    (replay_dir / "replay_policy_metrics.csv").write_text(
        "scope,policy_id,n_trades,net_sum_bps,win_rate,profit_factor,max_drawdown_bps,max_loss_bps\n"
        "aggregate,candidate_replay,4,200,0.75,1.5,-50,-20\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_monthly.csv").write_text(
        "policy_id,month,net_sum_bps\ncandidate_replay,2026-01,100\ncandidate_replay,2026-02,100\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_trades.csv").write_text(
        "entry_time,policy_id,session,side,score,p_long,p_short,p_flat,net_pnl_bps,mfe_bps,mae_bps,held_bars\n"
        "2026-01-03T08:00:00Z,candidate_replay,EU,LONG,0.8,0.8,0.1,0.1,100,120,10,8\n",
        encoding="utf-8",
    )
    _write_json(
        replay_manifest,
        {
            "decision": "PASS",
            "replay_identity_contract": _replay_identity_contract(),
        },
    )
    ready = tmp_path / "ENTRY_REPLAY_READINESS_latest.json"
    replay_fingerprints = _replay_artifact_fingerprints(
        candidate=candidate,
        candidate_audit=candidate_audit,
        selective_summary=selective_summary,
        selective_metrics=selective_metrics,
        replay_manifest=replay_manifest,
        replay_dir=replay_dir,
    )
    bad_gate = {
        "name": "candidate_bundle_audit",
        "decision": "FAIL",
        "checks": [
            {
                "name": "candidate bundle audit validated pre-train manifest provenance",
                "ok": False,
                "details": {"pretrain_manifest_contract": {}},
            }
        ],
    }
    _write_json(
        ready,
        {
            "decision": "READY_FOR_IQL_DISTILLATION_VEDTAK",
            "iql_distillation_allowed_with_explicit_vedtak": True,
            "promotion_shadow_live_allowed": False,
            "candidate_readiness_json": str(candidate),
            "candidate_bundle_audit_json": str(candidate_audit),
            "selective_edge_summary_json": str(selective_summary),
            "selective_edge_metrics_csv": str(selective_metrics),
            "replay_dir": str(replay_dir),
            "evidence_identity": {
                "candidate_bundle_audit_json": str(candidate_audit),
                "selective_edge_summary_json": str(selective_summary),
                "replay_evidence_manifest_json": str(replay_manifest),
                "candidate_bundle_dir": "/tmp/candidate_bundle",
                "selective_edge_bundle_dir": "/tmp/candidate_bundle",
                "replay_identity_candidate_bundle_dir": "/tmp/candidate_bundle",
                "no_xgb_bundle_dir": "/tmp/no_xgb_bundle",
                "replay_identity_ready": True,
                "candidate_specialist_contract": _candidate_specialist_identity(),
                "selective_edge_specialist_contract": _selective_specialist_identity(),
                "candidate_specialist_contract_ready": True,
                "selective_edge_specialist_contract_ready": True,
            },
            "artifact_fingerprints": replay_fingerprints,
            "gates": [
                bad_gate,
                BUNDLE_SPECIALIST_MODEL_GATE,
                {"name": "artifact_provenance", "decision": "PASS", "checks": []},
            ],
        },
    )

    report = run(
        argparse.Namespace(
            vedtak="PYTEST_BAD_PROVENANCE",
            replay_readiness_json=str(ready),
            out_dir=str(tmp_path / "out"),
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "ENTRY_IQL_DISTILLATION_CONTRACT_NOT_READY"
    assert report["iql_research_distillation_allowed"] is False
    assert any(failure["check"] == "replay-readiness preserved candidate pretrain provenance" for failure in report["failures"])
    assert any(failure["check"] == "replay-readiness preserved smoke dataset audit provenance" for failure in report["failures"])
    assert any(failure["check"] == "replay-readiness preserved exact specialist set provenance" for failure in report["failures"])
    assert any(failure["check"] == "replay-readiness preserved specialist model contract provenance" for failure in report["failures"])


def test_iql_distillation_contract_rejects_missing_specialist_model_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate_audit = tmp_path / "candidate_audit.json"
    selective_summary = tmp_path / "selective" / "summary.json"
    selective_metrics = tmp_path / "selective" / "selective_edge_metrics.csv"
    replay_dir = tmp_path / "replay"
    replay_manifest = replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    _write_json(candidate, {"decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK"})
    _write_json(candidate_audit, {"decision": "PASS", "bundle_dir": "/tmp/candidate_bundle"})
    _write_json(selective_summary, {"splits": ["val", "test"]})
    selective_metrics.parent.mkdir(parents=True, exist_ok=True)
    selective_metrics.write_text("split,model\nval,candidate\n", encoding="utf-8")
    replay_dir.mkdir(parents=True)
    (replay_dir / "replay_policy_metrics.csv").write_text(
        "scope,policy_id,n_trades,net_sum_bps,win_rate,profit_factor,max_drawdown_bps,max_loss_bps\n"
        "aggregate,candidate_replay,4,200,0.75,1.5,-50,-20\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_monthly.csv").write_text(
        "policy_id,month,net_sum_bps\ncandidate_replay,2026-01,100\ncandidate_replay,2026-02,100\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_trades.csv").write_text(
        "entry_time,policy_id,session,side,score,p_long,p_short,p_flat,net_pnl_bps,mfe_bps,mae_bps,held_bars\n"
        "2026-01-03T08:00:00Z,candidate_replay,EU,LONG,0.8,0.8,0.1,0.1,100,120,10,8\n",
        encoding="utf-8",
    )
    _write_json(
        replay_manifest,
        {
            "decision": "PASS",
            "replay_identity_contract": _replay_identity_contract(),
        },
    )
    ready = tmp_path / "ENTRY_REPLAY_READINESS_latest.json"
    replay_fingerprints = _replay_artifact_fingerprints(
        candidate=candidate,
        candidate_audit=candidate_audit,
        selective_summary=selective_summary,
        selective_metrics=selective_metrics,
        replay_manifest=replay_manifest,
        replay_dir=replay_dir,
    )
    bad_gate = json.loads(json.dumps(PRETRAIN_PROVENANCE_GATE))
    contract = bad_gate["checks"][0]["details"]["pretrain_manifest_contract"]
    contract["specialist_model_contract_valid"] = False
    contract["specialist_model_contract_set_exact"] = False
    contract["specialist_model_contract_owned_objectives_match"] = False
    contract["smoke_edge_specialist_model_contract_valid"] = False
    contract["smoke_edge_specialist_model_contract_set_exact"] = False
    contract["smoke_edge_specialist_model_contract_owned_objectives_match"] = False
    _write_json(
        ready,
        {
            "decision": "READY_FOR_IQL_DISTILLATION_VEDTAK",
            "iql_distillation_allowed_with_explicit_vedtak": True,
            "promotion_shadow_live_allowed": False,
            "candidate_readiness_json": str(candidate),
            "candidate_bundle_audit_json": str(candidate_audit),
            "selective_edge_summary_json": str(selective_summary),
            "selective_edge_metrics_csv": str(selective_metrics),
            "replay_dir": str(replay_dir),
            "evidence_identity": {
                "candidate_bundle_audit_json": str(candidate_audit),
                "selective_edge_summary_json": str(selective_summary),
                "replay_evidence_manifest_json": str(replay_manifest),
                "candidate_bundle_dir": "/tmp/candidate_bundle",
                "selective_edge_bundle_dir": "/tmp/candidate_bundle",
                "replay_identity_candidate_bundle_dir": "/tmp/candidate_bundle",
                "no_xgb_bundle_dir": "/tmp/no_xgb_bundle",
                "replay_identity_ready": True,
                "candidate_specialist_contract": _candidate_specialist_identity(),
                "selective_edge_specialist_contract": _selective_specialist_identity(),
                "candidate_specialist_contract_ready": True,
                "selective_edge_specialist_contract_ready": True,
            },
            "artifact_fingerprints": replay_fingerprints,
            "gates": [
                bad_gate,
                BUNDLE_SPECIALIST_MODEL_GATE,
                {"name": "artifact_provenance", "decision": "PASS", "checks": []},
            ],
        },
    )

    report = run(
        argparse.Namespace(
            vedtak="PYTEST_BAD_SPECIALIST_MODEL",
            replay_readiness_json=str(ready),
            out_dir=str(tmp_path / "out"),
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "ENTRY_IQL_DISTILLATION_CONTRACT_NOT_READY"
    assert report["specialist_model_provenance_contract"]["ok"] is False
    assert any(
        failure["check"] == "replay-readiness preserved specialist model contract provenance"
        for failure in report["failures"]
    )


def test_iql_distillation_contract_rejects_missing_bundle_specialist_model_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate_audit = tmp_path / "candidate_audit.json"
    selective_summary = tmp_path / "selective" / "summary.json"
    selective_metrics = tmp_path / "selective" / "selective_edge_metrics.csv"
    replay_dir = tmp_path / "replay"
    replay_manifest = replay_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    _write_json(candidate, {"decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK"})
    _write_json(candidate_audit, {"decision": "PASS", "bundle_dir": "/tmp/candidate_bundle"})
    _write_json(selective_summary, {"splits": ["val", "test"]})
    selective_metrics.parent.mkdir(parents=True, exist_ok=True)
    selective_metrics.write_text("split,model\nval,candidate\n", encoding="utf-8")
    replay_dir.mkdir(parents=True)
    (replay_dir / "replay_policy_metrics.csv").write_text(
        "scope,policy_id,n_trades,net_sum_bps,win_rate,profit_factor,max_drawdown_bps,max_loss_bps\n"
        "aggregate,candidate_replay,4,200,0.75,1.5,-50,-20\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_monthly.csv").write_text(
        "policy_id,month,net_sum_bps\ncandidate_replay,2026-01,100\ncandidate_replay,2026-02,100\n",
        encoding="utf-8",
    )
    (replay_dir / "replay_policy_trades.csv").write_text(
        "entry_time,policy_id,session,side,score,p_long,p_short,p_flat,net_pnl_bps,mfe_bps,mae_bps,held_bars\n"
        "2026-01-03T08:00:00Z,candidate_replay,EU,LONG,0.8,0.8,0.1,0.1,100,120,10,8\n",
        encoding="utf-8",
    )
    _write_json(
        replay_manifest,
        {"decision": "PASS", "replay_identity_contract": _replay_identity_contract()},
    )
    ready = tmp_path / "ENTRY_REPLAY_READINESS_latest.json"
    replay_fingerprints = _replay_artifact_fingerprints(
        candidate=candidate,
        candidate_audit=candidate_audit,
        selective_summary=selective_summary,
        selective_metrics=selective_metrics,
        replay_manifest=replay_manifest,
        replay_dir=replay_dir,
    )
    bad_gate = json.loads(json.dumps(BUNDLE_SPECIALIST_MODEL_GATE))
    bad_gate["checks"][0]["ok"] = False
    bad_gate["checks"][0]["details"]["bundle_summary"]["specialist_model_contract_valid"] = False
    bad_gate["checks"][0]["details"]["bundle_specialist_model_contract"]["valid"] = False
    bad_gate["checks"][0]["details"]["bundle_specialist_model_contract"]["failures"] = ["forced invalid bundle contract"]
    _write_json(
        ready,
        {
            "decision": "READY_FOR_IQL_DISTILLATION_VEDTAK",
            "iql_distillation_allowed_with_explicit_vedtak": True,
            "promotion_shadow_live_allowed": False,
            "candidate_readiness_json": str(candidate),
            "candidate_bundle_audit_json": str(candidate_audit),
            "selective_edge_summary_json": str(selective_summary),
            "selective_edge_metrics_csv": str(selective_metrics),
            "replay_dir": str(replay_dir),
            "evidence_identity": {
                "candidate_bundle_audit_json": str(candidate_audit),
                "selective_edge_summary_json": str(selective_summary),
                "replay_evidence_manifest_json": str(replay_manifest),
                "candidate_bundle_dir": "/tmp/candidate_bundle",
                "selective_edge_bundle_dir": "/tmp/candidate_bundle",
                "replay_identity_candidate_bundle_dir": "/tmp/candidate_bundle",
                "no_xgb_bundle_dir": "/tmp/no_xgb_bundle",
                "replay_identity_ready": True,
                "candidate_specialist_contract": _candidate_specialist_identity(),
                "selective_edge_specialist_contract": _selective_specialist_identity(),
                "candidate_specialist_contract_ready": True,
                "selective_edge_specialist_contract_ready": True,
            },
            "artifact_fingerprints": replay_fingerprints,
            "gates": [
                PRETRAIN_PROVENANCE_GATE,
                bad_gate,
                {"name": "artifact_provenance", "decision": "PASS", "checks": []},
            ],
        },
    )

    report = run(
        argparse.Namespace(
            vedtak="PYTEST_MISSING_BUNDLE_CONTRACT",
            replay_readiness_json=str(ready),
            out_dir=str(tmp_path / "out"),
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "ENTRY_IQL_DISTILLATION_CONTRACT_NOT_READY"
    assert report["bundle_specialist_model_provenance_contract"]["ok"] is False
    assert any(
        failure["check"] == "replay-readiness preserved candidate bundle specialist model contract provenance"
        for failure in report["failures"]
    )
