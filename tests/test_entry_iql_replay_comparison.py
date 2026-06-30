import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.verify_entry_iql_replay_comparison_v1 import build_comparison_checks, run


CANDIDATE_BUNDLE_DIR = "/tmp/candidate_bundle"
CANDIDATE_REPLAY_MANIFEST_JSON = "/tmp/candidate/REPLAY_EVIDENCE_MANIFEST.json"
IQL_REPLAY_MANIFEST_JSON = "/tmp/iql/REPLAY_EVIDENCE_MANIFEST.json"
DISTILLATION_CONTRACT_JSON = "/tmp/distill.json"
BASE_SPECIALIST_GROUPS = [
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
]


def _metrics(net: float, pf: float, dd: float, loss: float, trades: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["aggregate"],
            "policy_id": ["policy"],
            "n_trades": [trades],
            "net_sum_bps": [net],
            "win_rate": [0.60],
            "profit_factor": [pf],
            "max_drawdown_bps": [dd],
            "max_loss_bps": [loss],
        }
    )


def _monthly(a: float = 80.0, b: float = 90.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_id": ["policy", "policy"],
            "month": ["2026-01", "2026-02"],
            "net_sum_bps": [a, b],
        }
    )


def _write_replay(path: Path, metrics: pd.DataFrame, monthly: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path / "replay_policy_metrics.csv", index=False)
    monthly.to_csv(path / "replay_policy_monthly.csv", index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def _evidence_identity(
    bundle_dir: str = CANDIDATE_BUNDLE_DIR,
    candidate_replay_manifest_json: str = CANDIDATE_REPLAY_MANIFEST_JSON,
    contract_mode: str = "foundation_seq146",
) -> dict:
    return {
        "contract_mode": contract_mode,
        "replay_evidence_manifest_json": candidate_replay_manifest_json,
        "candidate_bundle_dir": bundle_dir,
        "selective_edge_bundle_dir": bundle_dir,
        "replay_identity_candidate_bundle_dir": bundle_dir,
        "no_xgb_bundle_dir": "/tmp/no_xgb_bundle",
        "replay_identity_ready": True,
        "candidate_specialist_contract": _candidate_specialist_identity(contract_mode),
        "selective_edge_specialist_contract": _selective_specialist_identity(contract_mode),
        "candidate_specialist_contract_ready": True,
        "selective_edge_specialist_contract_ready": True,
    }


def _distill_contract(
    bundle_dir: str = CANDIDATE_BUNDLE_DIR,
    candidate_replay_manifest_json: str = CANDIDATE_REPLAY_MANIFEST_JSON,
) -> dict:
    return {
        "decision": "ENTRY_IQL_DISTILLATION_CONTRACT_READY",
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
        "evidence_identity": _evidence_identity(bundle_dir, candidate_replay_manifest_json),
    }


def _replay_manifest(
    bundle_dir: str = CANDIDATE_BUNDLE_DIR,
    *,
    distillation_contract_json: str = DISTILLATION_CONTRACT_JSON,
    candidate_replay_manifest_json: str = CANDIDATE_REPLAY_MANIFEST_JSON,
) -> dict:
    return {
        "decision": "PASS",
        "distillation_contract_json": distillation_contract_json,
        "replay_identity_contract": {
            "ready": True,
            "contract_mode": "foundation_seq146",
            "candidate_bundle_dir": bundle_dir,
            "candidate_replay_evidence_manifest_json": candidate_replay_manifest_json,
            "distillation_contract_json": distillation_contract_json,
            "distillation_artifact_hash_contract": {"ready": True, "failures": []},
            "candidate_specialist_contract": _candidate_specialist_identity(),
            "selective_edge_specialist_contract": _selective_specialist_identity(),
            "replay_specialist_identity_contract": _replay_specialist_identity_contract(),
        },
        "replay_specialist_identity_contract": _replay_specialist_identity_contract(),
    }


def test_comparison_checks_pass_when_iql_beats_candidate() -> None:
    checks, comparison = build_comparison_checks(
        distill_contract=_distill_contract(),
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    assert all(check["ok"] for check in checks)
    assert comparison["net_lift_bps"] == 60.0


def test_comparison_checks_fail_on_drawdown_degradation() -> None:
    checks, _ = build_comparison_checks(
        distill_contract=_distill_contract(),
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 140.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL replay drawdown does not worsen beyond bound" in failed


def test_comparison_checks_fail_on_iql_identity_mismatch() -> None:
    checks, _ = build_comparison_checks(
        distill_contract=_distill_contract(),
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest("/tmp/wrong_candidate_bundle"),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL replay manifest evidence identity matches distillation contract" in failed


def test_comparison_checks_fail_on_contract_mode_identity_mismatch() -> None:
    contract = _distill_contract()
    contract["contract_mode"] = "challenger_seq215"
    contract["evidence_identity"]["contract_mode"] = "challenger_seq215"
    candidate_manifest = _replay_manifest()
    iql_manifest = _replay_manifest()
    candidate_manifest["replay_identity_contract"]["contract_mode"] = "foundation_seq146"
    iql_manifest["replay_identity_contract"]["contract_mode"] = "challenger_seq215"

    checks, _ = build_comparison_checks(
        distill_contract=contract,
        candidate_replay_manifest=candidate_manifest,
        iql_replay_manifest=iql_manifest,
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "IQL replay comparison contract mode identity is aligned" in failed


def test_comparison_checks_fail_when_distillation_pretrain_provenance_missing() -> None:
    bad_contract = _distill_contract()
    bad_contract["candidate_pretrain_provenance_contract"] = {"found": True, "ok": False}
    checks, _ = build_comparison_checks(
        distill_contract=bad_contract,
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL distillation contract preserved candidate pretrain provenance" in failed


def test_comparison_checks_fail_when_smoke_dataset_provenance_missing() -> None:
    bad_contract = _distill_contract()
    bad_contract["smoke_dataset_provenance_contract"] = {"ok": False, "failures": ["missing smoke dataset provenance"]}
    checks, _ = build_comparison_checks(
        distill_contract=bad_contract,
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL distillation contract preserved smoke dataset audit provenance" in failed


def test_comparison_checks_fail_when_specialist_set_provenance_missing() -> None:
    bad_contract = _distill_contract()
    bad_contract["specialist_set_provenance_contract"] = {
        "ok": False,
        "failures": ["missing exact specialist set provenance"],
    }
    checks, _ = build_comparison_checks(
        distill_contract=bad_contract,
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL distillation contract preserved exact specialist set provenance" in failed


def test_comparison_checks_fail_when_specialist_model_contract_provenance_missing() -> None:
    bad_contract = _distill_contract()
    bad_contract["specialist_model_provenance_contract"] = {
        "ok": False,
        "failures": ["missing specialist model contract provenance"],
    }
    checks, _ = build_comparison_checks(
        distill_contract=bad_contract,
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL distillation contract preserved specialist model contract provenance" in failed


def test_comparison_checks_fail_when_bundle_specialist_model_contract_provenance_missing() -> None:
    bad_contract = _distill_contract()
    bad_contract["bundle_specialist_model_provenance_contract"] = {
        "ok": False,
        "failures": ["missing candidate bundle specialist model contract provenance"],
    }

    checks, _ = build_comparison_checks(
        candidate_metrics=_metrics(100.0, 1.4, -40.0, -20.0),
        iql_metrics=_metrics(170.0, 1.7, -35.0, -15.0),
        candidate_monthly=_monthly(),
        iql_monthly=_monthly(),
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distill_contract=bad_contract,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "IQL distillation contract preserved candidate bundle specialist model contract provenance" in failed


def test_comparison_checks_fail_when_replay_artifact_provenance_missing() -> None:
    bad_contract = _distill_contract()
    bad_contract["replay_artifact_provenance_contract"] = {"ok": False, "failures": ["missing replay artifact hash"]}
    checks, _ = build_comparison_checks(
        distill_contract=bad_contract,
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL distillation contract preserved replay artifact provenance" in failed


def test_comparison_checks_fail_when_iql_replay_hash_contract_not_validated() -> None:
    bad_iql_manifest = _replay_manifest()
    bad_iql_manifest["replay_identity_contract"]["distillation_artifact_hash_contract"] = {
        "ready": False,
        "failures": ["hash mismatch"],
    }
    checks, _ = build_comparison_checks(
        distill_contract=_distill_contract(),
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=bad_iql_manifest,
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL replay manifest validated distillation artifact hashes" in failed


def test_comparison_checks_fail_on_iql_distillation_contract_path_mismatch() -> None:
    checks, _ = build_comparison_checks(
        distill_contract=_distill_contract(),
        candidate_replay_manifest=_replay_manifest(),
        iql_replay_manifest=_replay_manifest(distillation_contract_json="/tmp/other_distill.json"),
        distillation_contract_json=DISTILLATION_CONTRACT_JSON,
        candidate_replay_manifest_json=CANDIDATE_REPLAY_MANIFEST_JSON,
        iql_replay_manifest_json=IQL_REPLAY_MANIFEST_JSON,
        candidate_metrics=_metrics(200.0, 1.20, 100.0, -25.0),
        candidate_monthly=_monthly(),
        iql_metrics=_metrics(260.0, 1.35, 90.0, -20.0),
        iql_monthly=_monthly(120.0, 140.0),
        min_net_lift_bps=0.0,
        min_iql_profit_factor=1.05,
        min_profit_factor_lift=0.0,
        max_drawdown_worsening_bps=0.0,
        max_loss_worsening_bps=0.0,
    )

    failed = {check["name"] for check in checks if not check["ok"]}
    assert "IQL replay manifest distillation contract matches comparison input" in failed


def test_iql_replay_comparison_run_passes_on_ready_fixture(tmp_path: Path) -> None:
    contract = tmp_path / "distill.json"
    candidate = tmp_path / "candidate"
    iql = tmp_path / "iql"
    candidate_manifest = candidate / "REPLAY_EVIDENCE_MANIFEST.json"
    iql_manifest = iql / "REPLAY_EVIDENCE_MANIFEST.json"
    _write_json(contract, _distill_contract(candidate_replay_manifest_json=str(candidate_manifest)))
    _write_replay(candidate, _metrics(200.0, 1.20, 100.0, -25.0), _monthly())
    _write_replay(iql, _metrics(260.0, 1.35, 90.0, -20.0), _monthly(120.0, 140.0))
    _write_json(
        candidate_manifest,
        _replay_manifest(
            distillation_contract_json=str(contract),
            candidate_replay_manifest_json=str(candidate_manifest),
        ),
    )
    _write_json(
        iql_manifest,
        _replay_manifest(
            distillation_contract_json=str(contract),
            candidate_replay_manifest_json=str(candidate_manifest),
        ),
    )

    report = run(
        argparse.Namespace(
            distillation_contract_json=str(contract),
            candidate_replay_dir=str(candidate),
            iql_replay_dir=str(iql),
            out_dir=str(tmp_path / "out"),
            min_net_lift_bps=0.0,
            min_iql_profit_factor=1.05,
            min_profit_factor_lift=0.0,
            max_drawdown_worsening_bps=0.0,
            max_loss_worsening_bps=0.0,
            fail_on_not_ready=True,
            quiet=True,
        )
    )

    assert report["decision"] == "READY_FOR_PROMOTION_REVIEW_VEDTAK"
    assert report["promotion_review_allowed_with_explicit_vedtak"] is True
    assert report["promotion_shadow_live_allowed"] is False
    assert report["evidence_identity"]["candidate_bundle_dir"] == CANDIDATE_BUNDLE_DIR
    assert Path(report["json_path"]).exists()


def test_iql_replay_comparison_current_artifacts_not_ready(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            distillation_contract_json="/home/andre2/GX1_DATA/reports/entry_iql_distillation_contract_20260628_v1/ENTRY_IQL_DISTILLATION_CONTRACT_latest.json",
            candidate_replay_dir="/home/andre2/GX1_DATA/reports/entry_candidate_replay_20260628_v1",
            iql_replay_dir="/home/andre2/GX1_DATA/reports/entry_iql_distillation_replay_20260628_v1",
            out_dir=str(tmp_path),
            min_net_lift_bps=0.0,
            min_iql_profit_factor=1.05,
            min_profit_factor_lift=0.0,
            max_drawdown_worsening_bps=0.0,
            max_loss_worsening_bps=0.0,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["promotion_shadow_live_allowed"] is False
    assert report["decision"] in {"NOT_READY_FOR_PROMOTION_REVIEW", "READY_FOR_PROMOTION_REVIEW_VEDTAK"}
    if report["decision"] == "NOT_READY_FOR_PROMOTION_REVIEW":
        failed = {failure["check"] for failure in report["failures"]}
        assert failed
        assert {
            "IQL replay metrics have rows",
            "IQL replay net sum beats candidate",
            "IQL replay manifest is PASS",
            "IQL distillation contract preserved replay specialist identity",
            "candidate replay manifest preserves specialist identity",
            "IQL replay manifest preserves replay specialist identity",
        } & failed
    else:
        assert report["failures"] == []
        assert report["promotion_review_allowed_with_explicit_vedtak"] is True
