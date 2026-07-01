#!/usr/bin/env python3
"""Materialize the Entry foundation IQL-distillation contract.

This is a research handoff contract, not a trainer. It can only open after the
Entry candidate has green replay-readiness evidence. It never promotes, starts
shadow/live, or writes adapter/pin artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT
from gx1.scripts.verify_entry_replay_readiness_v1 import DEFAULT_OUT_DIR as REPLAY_READINESS_OUT_DIR
from gx1.scripts.verify_entry_training_readiness_v1 import _check


DEFAULT_REPLAY_READINESS_JSON = REPLAY_READINESS_OUT_DIR / "ENTRY_REPLAY_READINESS_latest.json"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_iql_distillation_contract_20260628_v1"

REQUIRED_DECISION = "READY_FOR_IQL_DISTILLATION_VEDTAK"
CONTRACT_INPUT_DIMS = {
    "foundation_seq146": 146,
    "challenger_seq215": 215,
    "smart_seq520_candidate": 520,
}
IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS = (
    "replay_readiness",
    "candidate_readiness",
    "candidate_bundle_audit",
    "selective_edge_summary",
    "selective_edge_metrics",
    "candidate_replay_manifest",
    "candidate_replay_metrics",
    "candidate_replay_monthly",
    "candidate_replay_trades",
)
REPLAY_PRETRAIN_PROVENANCE_GATE = "candidate_bundle_audit"
REPLAY_PRETRAIN_PROVENANCE_CHECK = "candidate bundle audit validated pre-train manifest provenance"
REPLAY_BUNDLE_SPECIALIST_MODEL_GATE = "candidate_bundle_audit"
REPLAY_BUNDLE_SPECIALIST_MODEL_CHECK = "candidate bundle specialist model contract is preserved in bundle metadata"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        out = float(obj)
        return out if np.isfinite(out) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_exists_from_report(report: dict[str, Any], key: str, *, file: bool) -> bool:
    raw = str(report.get(key) or "")
    if not raw:
        return False
    path = Path(raw).expanduser()
    return path.is_file() if file else path.is_dir()


def _normal_artifact_path(value: Any) -> str:
    raw = str(value or "").strip()
    return str(Path(raw).expanduser().resolve()) if raw else ""


def _artifact_paths_from_replay_readiness(
    replay_readiness: dict[str, Any],
    replay_readiness_path: Path,
) -> dict[str, str]:
    identity = (
        replay_readiness.get("evidence_identity")
        if isinstance(replay_readiness.get("evidence_identity"), dict)
        else {}
    )
    replay_dir_raw = str(replay_readiness.get("replay_dir") or "").strip()
    replay_dir = Path(replay_dir_raw).expanduser().resolve() if replay_dir_raw else None
    replay_manifest = str(identity.get("replay_evidence_manifest_json") or "").strip()
    if not replay_manifest and replay_dir is not None:
        replay_manifest = str(replay_dir / "REPLAY_EVIDENCE_MANIFEST.json")
    return {
        "replay_readiness": str(replay_readiness_path),
        "candidate_readiness": _normal_artifact_path(replay_readiness.get("candidate_readiness_json")),
        "candidate_bundle_audit": _normal_artifact_path(replay_readiness.get("candidate_bundle_audit_json")),
        "selective_edge_summary": _normal_artifact_path(replay_readiness.get("selective_edge_summary_json")),
        "selective_edge_metrics": _normal_artifact_path(replay_readiness.get("selective_edge_metrics_csv")),
        "candidate_replay_manifest": _normal_artifact_path(replay_manifest),
        "candidate_replay_metrics": str(replay_dir / "replay_policy_metrics.csv") if replay_dir is not None else "",
        "candidate_replay_monthly": str(replay_dir / "replay_policy_monthly.csv") if replay_dir is not None else "",
        "candidate_replay_trades": str(replay_dir / "replay_policy_trades.csv") if replay_dir is not None else "",
    }


def _artifact_hash_contract(artifact_paths: dict[str, str]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    hashes: dict[str, str] = {}
    checks: dict[str, dict[str, Any]] = {}
    for key in IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS:
        raw = str(artifact_paths.get(key) or "")
        path = Path(raw).expanduser().resolve() if raw else None
        exists = bool(path and path.is_file())
        observed = _sha256_file(path) if exists and path is not None else ""
        if observed:
            hashes[key] = observed
        checks[key] = {
            "path": str(path) if path is not None else "",
            "exists": exists,
            "sha256": observed,
            "ok": bool(exists and observed),
        }
    return hashes, checks


def _identity_checks(replay_readiness: dict[str, Any]) -> list[dict[str, Any]]:
    identity = replay_readiness.get("evidence_identity") if isinstance(replay_readiness.get("evidence_identity"), dict) else {}
    candidate_bundle_dir = str(identity.get("candidate_bundle_dir") or "")
    selective_bundle_dir = str(identity.get("selective_edge_bundle_dir") or "")
    replay_bundle_dir = str(identity.get("replay_identity_candidate_bundle_dir") or "")
    contract_mode = str(replay_readiness.get("contract_mode") or identity.get("contract_mode") or "foundation_seq146")
    identity_contract_modes = {
        "evidence_identity": str(identity.get("contract_mode") or contract_mode),
        "candidate_bundle": str(identity.get("candidate_bundle_contract_mode") or contract_mode),
        "selective_edge": str(identity.get("selective_edge_contract_mode") or contract_mode),
        "replay_identity": str(identity.get("replay_identity_contract_mode") or contract_mode),
    }
    return [
        _check("replay-readiness carries evidence identity", bool(identity), {"evidence_identity": identity}),
        _check(
            "replay-readiness carries supported contract mode",
            contract_mode in CONTRACT_INPUT_DIMS,
            {"contract_mode": contract_mode, "supported_contract_modes": sorted(CONTRACT_INPUT_DIMS)},
        ),
        _check(
            "replay-readiness contract mode identity is aligned",
            all(mode == contract_mode for mode in identity_contract_modes.values()),
            {"contract_mode": contract_mode, "identity_contract_modes": identity_contract_modes},
        ),
        _check("evidence identity has candidate bundle dir", bool(candidate_bundle_dir), {"evidence_identity": identity}),
        _check(
            "evidence identity matches selective-edge bundle",
            bool(candidate_bundle_dir) and selective_bundle_dir == candidate_bundle_dir,
            {"candidate_bundle_dir": candidate_bundle_dir, "selective_edge_bundle_dir": selective_bundle_dir},
        ),
        _check(
            "evidence identity matches replay manifest bundle",
            bool(candidate_bundle_dir) and replay_bundle_dir == candidate_bundle_dir,
            {"candidate_bundle_dir": candidate_bundle_dir, "replay_identity_candidate_bundle_dir": replay_bundle_dir},
        ),
        _check("evidence identity replay contract is ready", bool(identity.get("replay_identity_ready")), {"evidence_identity": identity}),
        _check(
            "replay evidence manifest artifact exists",
            bool(identity.get("replay_evidence_manifest_json"))
            and Path(str(identity.get("replay_evidence_manifest_json"))).expanduser().is_file(),
            {"replay_evidence_manifest_json": identity.get("replay_evidence_manifest_json")},
        ),
    ]


def _replay_specialist_identity_contract(
    replay_readiness: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    evidence_identity = (
        replay_readiness.get("evidence_identity")
        if isinstance(replay_readiness.get("evidence_identity"), dict)
        else {}
    )
    replay_manifest_path = Path(str(artifact_paths.get("candidate_replay_manifest") or "")).expanduser()
    replay_manifest = _read_json(replay_manifest_path) if replay_manifest_path.is_file() else {}
    replay_identity = (
        replay_manifest.get("replay_identity_contract")
        if isinstance(replay_manifest.get("replay_identity_contract"), dict)
        else {}
    )
    candidate_specialist = (
        evidence_identity.get("candidate_specialist_contract")
        if isinstance(evidence_identity.get("candidate_specialist_contract"), dict)
        else replay_identity.get("candidate_specialist_contract")
        if isinstance(replay_identity.get("candidate_specialist_contract"), dict)
        else {}
    )
    selective_specialist = (
        evidence_identity.get("selective_edge_specialist_contract")
        if isinstance(evidence_identity.get("selective_edge_specialist_contract"), dict)
        else replay_identity.get("selective_edge_specialist_contract")
        if isinstance(replay_identity.get("selective_edge_specialist_contract"), dict)
        else {}
    )
    contract_mode = str(replay_readiness.get("contract_mode") or evidence_identity.get("contract_mode") or "foundation_seq146")
    failures: list[str] = []

    if not bool(candidate_specialist.get("ready")):
        failures.append("candidate replay specialist identity is not ready")
    if not bool(selective_specialist.get("ready")):
        failures.append("selective-edge replay specialist identity is not ready")
    for label, payload in (
        ("candidate", candidate_specialist),
        ("selective-edge", selective_specialist),
    ):
        observed_mode = str(payload.get("contract_mode") or "").strip()
        if observed_mode and observed_mode != contract_mode:
            failures.append(f"{label} specialist identity contract mode mismatch: {observed_mode} != {contract_mode}")
        if payload.get("failures"):
            failures.append(f"{label} specialist identity has failures: {payload.get('failures')}")
    if contract_mode in {"challenger_seq215", "smart_seq520_candidate"}:
        candidate_groups = set(str(x) for x in candidate_specialist.get("bundle_specialist_groups", []) if str(x))
        selective_candidate = (
            selective_specialist.get("candidate_bundle_specialist_contract")
            if isinstance(selective_specialist.get("candidate_bundle_specialist_contract"), dict)
            else {}
        )
        selective_observed = set(str(x) for x in selective_candidate.get("observed_specialists", []) if str(x))
        for required in ("chart_geometry_encoder", "price_action_candle_encoder"):
            if required not in candidate_groups:
                failures.append(f"candidate replay specialist identity missing {required}")
            if required not in selective_observed:
                failures.append(f"selective-edge replay specialist identity missing {required}")

    return {
        "ok": not failures,
        "contract_mode": contract_mode,
        "candidate_replay_manifest_json": str(replay_manifest_path),
        "candidate_specialist_contract": candidate_specialist,
        "selective_edge_specialist_contract": selective_specialist,
        "failures": failures,
    }


def _replay_readiness_check(
    replay_readiness: dict[str, Any],
    *,
    gate_name: str,
    check_name: str,
) -> dict[str, Any]:
    observed_gate_decision = None
    for gate in replay_readiness.get("gates") or []:
        if not isinstance(gate, dict) or str(gate.get("name")) != str(gate_name):
            continue
        observed_gate_decision = gate.get("decision")
        for check in gate.get("checks") or []:
            if isinstance(check, dict) and str(check.get("name")) == str(check_name):
                return {
                    "found": True,
                    "gate_decision": gate.get("decision"),
                    "check": check,
                    "ok": bool(check.get("ok")),
                }
    return {
        "found": False,
        "gate_decision": observed_gate_decision,
        "check": {},
        "ok": False,
    }


def _pretrain_manifest_contract_from_check(pretrain_provenance: dict[str, Any]) -> dict[str, Any]:
    check = pretrain_provenance.get("check") if isinstance(pretrain_provenance.get("check"), dict) else {}
    details = check.get("details") if isinstance(check.get("details"), dict) else {}
    contract = (
        details.get("pretrain_manifest_contract")
        if isinstance(details.get("pretrain_manifest_contract"), dict)
        else {}
    )
    return contract


def _smoke_dataset_provenance_contract(pretrain_provenance: dict[str, Any]) -> dict[str, Any]:
    contract = _pretrain_manifest_contract_from_check(pretrain_provenance)
    failures: list[str] = []
    if not bool(pretrain_provenance.get("ok")):
        failures.append("candidate pretrain provenance gate is not ok")
    if not bool(contract.get("smoke_dataset_audit_provenance_all_artifacts_present")):
        failures.append("candidate pretrain contract did not preserve smoke-dataset audit artifacts")
    if not bool(contract.get("smoke_dataset_audit_provenance_all_artifact_hashes_present")):
        failures.append("candidate pretrain contract did not preserve smoke-dataset audit artifact hashes")
    if not bool(contract.get("smoke_edge_worktree_critical_gate_review_ok")):
        failures.append("candidate pretrain contract did not preserve worktree critical-gate proof")
    return {
        "ok": not failures,
        "smoke_dataset_audit_provenance_all_artifacts_present": bool(
            contract.get("smoke_dataset_audit_provenance_all_artifacts_present")
        ),
        "smoke_dataset_audit_provenance_all_artifact_hashes_present": bool(
            contract.get("smoke_dataset_audit_provenance_all_artifact_hashes_present")
        ),
        "smoke_edge_worktree_critical_gate_review_ok": bool(
            contract.get("smoke_edge_worktree_critical_gate_review_ok")
        ),
        "failures": failures,
    }


def _specialist_set_provenance_contract(pretrain_provenance: dict[str, Any]) -> dict[str, Any]:
    contract = _pretrain_manifest_contract_from_check(pretrain_provenance)
    failures: list[str] = []
    if not bool(pretrain_provenance.get("ok")):
        failures.append("candidate pretrain provenance gate is not ok")
    if not bool(contract.get("specialist_required_training_set_exact")):
        failures.append("candidate pretrain contract did not preserve exact candidate required specialists")
    if not bool(contract.get("specialist_trainable_set_exact")):
        failures.append("candidate pretrain contract did not preserve exact candidate trainable specialists")
    if not bool(contract.get("smoke_edge_required_specialists_exact")):
        failures.append("candidate pretrain contract did not preserve exact required specialists")
    if not bool(contract.get("smoke_edge_specialist_groups_exact")):
        failures.append("candidate pretrain contract did not preserve exact smoke specialist groups")
    return {
        "ok": not failures,
        "specialist_required_training_set_exact": bool(contract.get("specialist_required_training_set_exact")),
        "specialist_trainable_set_exact": bool(contract.get("specialist_trainable_set_exact")),
        "smoke_edge_required_specialists_exact": bool(contract.get("smoke_edge_required_specialists_exact")),
        "smoke_edge_specialist_groups_exact": bool(contract.get("smoke_edge_specialist_groups_exact")),
        "failures": failures,
    }


def _specialist_model_provenance_contract(pretrain_provenance: dict[str, Any]) -> dict[str, Any]:
    contract = _pretrain_manifest_contract_from_check(pretrain_provenance)
    failures: list[str] = []
    if not bool(pretrain_provenance.get("ok")):
        failures.append("candidate pretrain provenance gate is not ok")
    if not bool(contract.get("specialist_model_contract_valid")):
        failures.append("candidate pretrain contract did not preserve a valid specialist model contract")
    if not bool(contract.get("specialist_model_contract_set_exact")):
        failures.append("candidate pretrain contract did not preserve the exact specialist model set")
    if not bool(contract.get("specialist_model_contract_owned_objectives_match")):
        failures.append("candidate pretrain contract did not preserve exact specialist owned objectives")
    if not bool(contract.get("smoke_edge_specialist_model_contract_valid")):
        failures.append("smoke edge contract did not preserve a valid specialist model contract")
    if not bool(contract.get("smoke_edge_specialist_model_contract_set_exact")):
        failures.append("smoke edge contract did not preserve the exact specialist model set")
    if not bool(contract.get("smoke_edge_specialist_model_contract_owned_objectives_match")):
        failures.append("smoke edge contract did not preserve exact specialist owned objectives")
    return {
        "ok": not failures,
        "specialist_model_contract_valid": bool(contract.get("specialist_model_contract_valid")),
        "specialist_model_contract_set_exact": bool(contract.get("specialist_model_contract_set_exact")),
        "specialist_model_contract_owned_objectives_match": bool(
            contract.get("specialist_model_contract_owned_objectives_match")
        ),
        "smoke_edge_specialist_model_contract_valid": bool(
            contract.get("smoke_edge_specialist_model_contract_valid")
        ),
        "smoke_edge_specialist_model_contract_set_exact": bool(
            contract.get("smoke_edge_specialist_model_contract_set_exact")
        ),
        "smoke_edge_specialist_model_contract_owned_objectives_match": bool(
            contract.get("smoke_edge_specialist_model_contract_owned_objectives_match")
        ),
        "failures": failures,
    }


def _bundle_specialist_model_provenance_contract(bundle_provenance: dict[str, Any]) -> dict[str, Any]:
    check = bundle_provenance.get("check") if isinstance(bundle_provenance.get("check"), dict) else {}
    details = check.get("details") if isinstance(check.get("details"), dict) else {}
    bundle_summary = details.get("bundle_summary") if isinstance(details.get("bundle_summary"), dict) else {}
    bundle_contract = (
        details.get("bundle_specialist_model_contract")
        if isinstance(details.get("bundle_specialist_model_contract"), dict)
        else {}
    )
    failures: list[str] = []
    if not bool(bundle_provenance.get("ok")):
        failures.append("candidate bundle specialist model contract gate is not ok")
    if not bool(bundle_summary.get("specialist_model_contract_declared_valid")):
        failures.append("candidate bundle did not declare specialist model contract valid")
    if not bool(bundle_summary.get("specialist_model_contract_valid")):
        failures.append("candidate bundle did not preserve a valid specialist model contract")
    if not bool(bundle_summary.get("specialist_model_contract_set_exact")):
        failures.append("candidate bundle did not preserve exact specialist model set")
    if not bool(bundle_summary.get("specialist_model_contract_owned_objectives_match")):
        failures.append("candidate bundle did not preserve exact specialist owned objectives")
    if not bool(bundle_summary.get("specialist_model_contract_support_heads_match")):
        failures.append("candidate bundle did not preserve specialist support heads")
    if not bool(bundle_summary.get("specialist_model_contract_signal_families_match")):
        failures.append("candidate bundle did not preserve specialist signal families")
    if not bool(bundle_summary.get("specialist_model_contract_model_roles_match")):
        failures.append("candidate bundle did not preserve specialist model roles")
    if str(bundle_contract.get("decision")) != "PASS" or not bool(bundle_contract.get("valid")):
        failures.append("candidate bundle specialist model contract report is not PASS")
    for field in ("set_exact", "owned_objectives_match", "support_heads_match", "signal_families_match", "model_roles_match"):
        if not bool(bundle_contract.get(field)):
            failures.append(f"candidate bundle specialist model contract report missing {field}")
    if bundle_contract.get("failures"):
        failures.append(f"candidate bundle specialist model contract report has failures: {bundle_contract.get('failures')}")
    return {
        "ok": not failures,
        "found": bool(bundle_provenance.get("found")),
        "gate_decision": bundle_provenance.get("gate_decision"),
        "check_name": check.get("name"),
        "bundle_summary": bundle_summary,
        "bundle_specialist_model_contract": bundle_contract,
        "failures": failures,
    }


def _replay_artifact_provenance_contract(
    replay_readiness: dict[str, Any],
    artifact_paths: dict[str, str],
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    gate = None
    for row in replay_readiness.get("gates") or []:
        if isinstance(row, dict) and str(row.get("name")) == "artifact_provenance":
            gate = row
            break
    fingerprints = (
        replay_readiness.get("artifact_fingerprints")
        if isinstance(replay_readiness.get("artifact_fingerprints"), dict)
        else {}
    )
    failures: list[str] = []
    if not gate or str(gate.get("decision")) != "PASS":
        failures.append("replay-readiness artifact_provenance gate is not PASS")
    if not fingerprints:
        failures.append("replay-readiness artifact fingerprints are missing")
    for key in IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS:
        if key == "replay_readiness":
            continue
        fingerprint = fingerprints.get(key)
        if not isinstance(fingerprint, dict):
            failures.append(f"replay-readiness missing artifact fingerprint: {key}")
            continue
        expected_path = _normal_artifact_path(artifact_paths.get(key))
        observed_path = _normal_artifact_path(fingerprint.get("path"))
        if expected_path != observed_path:
            failures.append(f"replay-readiness artifact fingerprint path mismatch: {key}")
        if str(fingerprint.get("sha256") or "") != str(artifact_hashes.get(key) or ""):
            failures.append(f"replay-readiness artifact fingerprint hash mismatch: {key}")
    return {
        "ok": not failures,
        "gate_decision": gate.get("decision") if isinstance(gate, dict) else None,
        "fingerprint_keys": sorted(fingerprints),
        "failures": failures,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry IQL Distillation Contract",
        "",
        f"- Decision: `{report['decision']}`",
        f"- IQL research distillation allowed: `{report['iql_research_distillation_allowed']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        f"- Vedtak: `{report['vedtak']}`",
        "",
        "## Tasks",
        "",
    ]
    for task in report["distillation_tasks"]:
        lines.append(f"- `{task['id']}`: {task['purpose']}")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_readiness_path = Path(args.replay_readiness_json).expanduser().resolve()
    replay_readiness = _read_json(replay_readiness_path)
    evidence_identity = replay_readiness.get("evidence_identity") if isinstance(replay_readiness.get("evidence_identity"), dict) else {}
    contract_mode = str(replay_readiness.get("contract_mode") or evidence_identity.get("contract_mode") or "foundation_seq146")
    artifact_paths = _artifact_paths_from_replay_readiness(replay_readiness, replay_readiness_path)
    artifact_sha256, artifact_hash_checks = _artifact_hash_contract(artifact_paths)
    pretrain_provenance = _replay_readiness_check(
        replay_readiness,
        gate_name=REPLAY_PRETRAIN_PROVENANCE_GATE,
        check_name=REPLAY_PRETRAIN_PROVENANCE_CHECK,
    )
    bundle_specialist_model_gate = _replay_readiness_check(
        replay_readiness,
        gate_name=REPLAY_BUNDLE_SPECIALIST_MODEL_GATE,
        check_name=REPLAY_BUNDLE_SPECIALIST_MODEL_CHECK,
    )
    smoke_dataset_provenance = _smoke_dataset_provenance_contract(pretrain_provenance)
    specialist_set_provenance = _specialist_set_provenance_contract(pretrain_provenance)
    specialist_model_provenance = _specialist_model_provenance_contract(pretrain_provenance)
    bundle_specialist_model_provenance = _bundle_specialist_model_provenance_contract(bundle_specialist_model_gate)
    replay_artifact_provenance = _replay_artifact_provenance_contract(
        replay_readiness,
        artifact_paths,
        artifact_sha256,
    )
    replay_specialist_identity = _replay_specialist_identity_contract(replay_readiness, artifact_paths)

    checks = [
        _check(
            "replay-readiness decision opens IQL distillation",
            str(replay_readiness.get("decision")) == REQUIRED_DECISION,
            {"decision": replay_readiness.get("decision")},
        ),
        _check(
            "replay-readiness requires explicit vedtak",
            bool(replay_readiness.get("iql_distillation_allowed_with_explicit_vedtak")) is True,
        ),
        _check(
            "replay-readiness still blocks promotion/shadow/live",
            bool(replay_readiness.get("promotion_shadow_live_allowed")) is False,
        ),
        _check(
            "candidate-readiness artifact exists",
            _artifact_exists_from_report(replay_readiness, "candidate_readiness_json", file=True),
            {"candidate_readiness_json": replay_readiness.get("candidate_readiness_json")},
        ),
        _check(
            "selective-edge summary artifact exists",
            _artifact_exists_from_report(replay_readiness, "selective_edge_summary_json", file=True),
            {"selective_edge_summary_json": replay_readiness.get("selective_edge_summary_json")},
        ),
        _check(
            "selective-edge metrics artifact exists",
            _artifact_exists_from_report(replay_readiness, "selective_edge_metrics_csv", file=True),
            {"selective_edge_metrics_csv": replay_readiness.get("selective_edge_metrics_csv")},
        ),
        _check(
            "offline replay artifact dir exists",
            _artifact_exists_from_report(replay_readiness, "replay_dir", file=False),
            {"replay_dir": replay_readiness.get("replay_dir")},
        ),
        _check(
            "IQL distillation input artifacts exist and are hashed",
            all(bool((artifact_hash_checks.get(key) or {}).get("ok")) for key in IQL_DISTILLATION_REQUIRED_ARTIFACT_KEYS),
            {"artifact_hash_checks": artifact_hash_checks},
        ),
        _check(
            "replay-readiness preserved candidate pretrain provenance",
            bool(pretrain_provenance.get("ok")),
            pretrain_provenance,
        ),
        _check(
            "replay-readiness preserved smoke dataset audit provenance",
            bool(smoke_dataset_provenance.get("ok")),
            smoke_dataset_provenance,
        ),
        _check(
            "replay-readiness preserved exact specialist set provenance",
            bool(specialist_set_provenance.get("ok")),
            specialist_set_provenance,
        ),
        _check(
            "replay-readiness preserved specialist model contract provenance",
            bool(specialist_model_provenance.get("ok")),
            specialist_model_provenance,
        ),
        _check(
            "replay-readiness preserved candidate bundle specialist model contract provenance",
            bool(bundle_specialist_model_provenance.get("ok")),
            bundle_specialist_model_provenance,
        ),
        _check(
            "replay-readiness artifact provenance is preserved",
            bool(replay_artifact_provenance.get("ok")),
            replay_artifact_provenance,
        ),
        _check(
            "replay-readiness preserved replay specialist identity",
            bool(replay_specialist_identity.get("ok")),
            replay_specialist_identity,
        ),
        *_identity_checks(replay_readiness),
        _check("contract never trains a model", True),
        _check("contract never writes production adapter", True),
        _check("contract never promotes, shadows, or starts live", True),
    ]
    failures = [check for check in checks if not check["ok"]]
    ready = not failures
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    distillation_tasks = [
        {
            "id": "entry_transformer_teacher",
            "purpose": "Use the replay-proven specialist-fusion candidate as teacher scores and direction priors.",
            "inputs": ["candidate bundle", "selective-edge metrics", "offline replay trade paths"],
        },
        {
            "id": "replay_reward_critic",
            "purpose": "Train/evaluate reward views for realized PnL, drawdown, MAE tail, path quality, duration, and missed opportunity.",
            "inputs": ["replay_policy_metrics.csv", "replay_policy_monthly.csv", "trade-level replay journal"],
        },
        {
            "id": "entry_iql_student",
            "purpose": "Fit an offline IQL entry policy against replay rewards, then compare against the candidate before any adapter work.",
            "inputs": [f"{contract_mode} state", "teacher labels", "replay rewards"],
        },
        {
            "id": "post_distillation_replay_compare",
            "purpose": "Require offline replay lift versus candidate and no-XGB ablation before any promotion review.",
            "inputs": ["IQL policy artifact", "candidate replay evidence", "slice reports"],
        },
    ]
    report = {
        "schema_version": "entry_iql_distillation_contract_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_mode": contract_mode,
        "decision": "ENTRY_IQL_DISTILLATION_CONTRACT_READY" if ready else "ENTRY_IQL_DISTILLATION_CONTRACT_NOT_READY",
        "vedtak": str(args.vedtak),
        "replay_readiness_json": str(replay_readiness_path),
        "replay_readiness_decision": replay_readiness.get("decision"),
        "evidence_identity": evidence_identity,
        "candidate_pretrain_provenance_contract": pretrain_provenance,
        "smoke_dataset_provenance_contract": smoke_dataset_provenance,
        "specialist_set_provenance_contract": specialist_set_provenance,
        "specialist_model_provenance_contract": specialist_model_provenance,
        "bundle_specialist_model_provenance_contract": bundle_specialist_model_provenance,
        "replay_artifact_provenance_contract": replay_artifact_provenance,
        "replay_specialist_identity_contract": replay_specialist_identity,
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
        "artifact_hash_checks": artifact_hash_checks,
        "iql_research_distillation_allowed": bool(ready),
        "trainer_started": False,
        "adapter_built": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "run offline IQL research distillation, then post-distillation replay comparison"
            if ready
            else "satisfy replay-readiness before IQL distillation contract"
        ),
        "distillation_tasks": distillation_tasks,
        "checks": checks,
        "failures": [{"check": check["name"], "details": check.get("details") or {}} for check in failures],
    }
    json_path = out_dir / f"ENTRY_IQL_DISTILLATION_CONTRACT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_IQL_DISTILLATION_CONTRACT_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_IQL_DISTILLATION_CONTRACT_latest.json"
    latest_md = out_dir / "ENTRY_IQL_DISTILLATION_CONTRACT_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                    "next_required_gate": report["next_required_gate"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vedtak", required=True)
    ap.add_argument("--replay-readiness-json", default=str(DEFAULT_REPLAY_READINESS_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
