#!/usr/bin/env python3
"""Verify Entry foundation readiness for the next vedtak-gated smoke train.

This is a readiness gate, not a training launcher. It proves that the feature,
target, specialist, smoke-dataset, wrapper and post-smoke audit contracts are in
place before the user gives an explicit smoke-train vedtak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_MODEL_CONTRACT,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
)
from gx1.scripts.audit_entry_foundation_features_v1 import REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
from gx1.scripts.audit_entry_foundation_worktree_hygiene_v1 import (
    DEFAULT_OUT_DIR as WORKTREE_HYGIENE_OUT_DIR,
)
from gx1.scripts.audit_entry_foundation_worktree_hygiene_v1 import run as run_worktree_hygiene
from gx1.scripts.verify_entry_foundation_state_v1 import (
    FEATURE_AUDIT_LATEST,
    FOUNDATION_DATASET_DIR,
    FOUNDATION_SMOKE_DATASET_DIR,
    REPO,
    REPORTS_ROOT,
    SPECIALIST_AUDIT_LATEST,
    TARGET_AUDIT_LATEST,
)
from gx1.scripts.verify_entry_foundation_state_v1 import run as run_foundation_verify


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_training_readiness_20260628_v1"
FOUNDATION_GUARDRAILS_LATEST = (
    REPORTS_ROOT / "entry_foundation_guardrails_20260628_v1/ENTRY_FOUNDATION_GUARDRAILS_latest.json"
)
SMOKE_BUNDLE_AUDIT_LATEST = (
    REPORTS_ROOT
    / "entry_foundation_smoke_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
SMOKE_DATASET_MANIFEST = FOUNDATION_SMOKE_DATASET_DIR / "SMOKE_DATASET_MANIFEST.json"
WORKTREE_HYGIENE_LATEST = WORKTREE_HYGIENE_OUT_DIR / "ENTRY_FOUNDATION_WORKTREE_HYGIENE_latest.json"
ADOPTION_CANDIDATE_ROOT = REPORTS_ROOT / "entry_foundation_adoption_candidate_20260629_v1"
ACTIVATION_PLAN_ROOT = REPORTS_ROOT / "entry_foundation_activation_plan_20260629_v1"
ACTIVATION_APPLY_ROOT = REPORTS_ROOT / "entry_foundation_activation_apply_20260629_v1"
ACTIVATION_POST_APPLY_ROOT = REPORTS_ROOT / "entry_foundation_activation_post_apply_20260629_v1"
REQUIRED_SPECIALISTS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
)
REQUIRED_FOUNDATION_REQUIREMENTS = (
    "hh_hl_lh_ll",
    "bos_choch_age",
    "sweep_reclaim",
    "compression_expansion",
    "impulse_pullback_phase",
    "session_x_structure",
)
EXPECTED_ACTIVE_TRAINING_HEADS = SPECIALIST_FUSION_ACTIVE_HEADS
EXPECTED_BLOCKED_HEADS = SPECIALIST_FUSION_BLOCKED_HEADS
EXPECTED_SMOKE_HEAD_FLAGS = (
    "--enable-tf-agreement-head",
    "--enable-path-quality-variance-head",
    "--enable-position-size-head",
    "--enable-dip-head",
    "--enable-forecast-head",
    "--enable-timing-head",
    "--enable-tail-risk-head",
    "--enable-vol-forecast-head",
    "--enable-mtf-direction-head",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _all_ok(checks: list[dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in checks)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "sha256": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _artifact_fingerprints(artifacts: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {name: _artifact_fingerprint(Path(path)) for name, path in artifacts.items()}


def _artifact_fingerprint_checks(fingerprints: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _check(
            "readiness records sha256 fingerprints for all artifacts",
            bool(fingerprints)
            and all(
                bool(row.get("exists"))
                and int(row.get("size_bytes") or 0) > 0
                and isinstance(row.get("mtime_ns"), int)
                and isinstance(row.get("sha256"), str)
                and len(str(row.get("sha256"))) == 64
                for row in fingerprints.values()
            ),
            {"artifact_fingerprints": fingerprints},
        )
    ]


def _latest_report(root: Path, filename: str) -> Path | None:
    candidates = (
        sorted(
            root.glob(f"*/{filename}"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if root.exists()
        else []
    )
    root_latest = root / filename
    if root_latest.exists():
        candidates.append(root_latest)
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return candidates[0] if candidates else None


def _read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_path_matches(reported: Any, expected: Path) -> bool:
    reported_text = str(reported or "")
    if reported_text == str(expected):
        return True
    if not reported_text:
        return False
    try:
        return Path(reported_text).expanduser().resolve() == expected.expanduser().resolve()
    except OSError:
        return False


def _activation_apply_command(plan_path: Path | None) -> list[str]:
    if plan_path is None:
        return []
    return [
        "scripts/entry_next_edge_control.sh",
        "foundation-activation-apply",
        "--plan-json",
        str(plan_path),
        "--apply",
        "--vedtak",
        "<id>",
    ]


def _activation_post_apply_command(apply_path: Path | None) -> list[str]:
    if apply_path is None:
        return []
    return [
        "scripts/entry_next_edge_control.sh",
        "foundation-activation-post-apply",
        "--activation-apply-json",
        str(apply_path),
        "--apply",
        "--vedtak",
        "<id>",
    ]


def _activation_transition(
    *,
    foundation_contract_ready: bool,
    foundation_activation: dict[str, Any],
) -> dict[str, Any]:
    activation_apply_required = bool(
        not foundation_contract_ready
        and foundation_activation.get("adoption_candidate_ready")
        and foundation_activation.get("activation_plan_ready")
        and foundation_activation.get("activation_apply_ready")
        and not foundation_activation.get("activation_apply_applied")
    )
    activation_post_apply_required = bool(
        not foundation_contract_ready
        and foundation_activation.get("activation_apply_applied")
        and not foundation_activation.get("activation_post_apply_completed")
        and foundation_activation.get("activation_post_apply_command")
    )
    next_allowed_command = ""
    if activation_apply_required:
        next_allowed_command = " ".join(
            str(part) for part in foundation_activation.get("activation_apply_command") or []
        )
    elif activation_post_apply_required:
        next_allowed_command = " ".join(
            str(part) for part in foundation_activation.get("activation_post_apply_command") or []
        )
    return {
        "activation_required_before_smoke": bool(activation_apply_required or activation_post_apply_required),
        "activation_apply_required_before_smoke": activation_apply_required,
        "activation_post_apply_required_before_smoke": activation_post_apply_required,
        "next_allowed_command": next_allowed_command,
    }


def _foundation_activation_summary() -> dict[str, Any]:
    adoption_path = _latest_report(ADOPTION_CANDIDATE_ROOT, "ENTRY_FOUNDATION_ADOPTION_CANDIDATE_latest.json")
    plan_path = _latest_report(ACTIVATION_PLAN_ROOT, "ENTRY_FOUNDATION_ACTIVATION_PLAN_latest.json")
    apply_path = _latest_report(ACTIVATION_APPLY_ROOT, "ENTRY_FOUNDATION_ACTIVATION_APPLY_latest.json")
    post_apply_path = _latest_report(
        ACTIVATION_POST_APPLY_ROOT,
        "ENTRY_FOUNDATION_ACTIVATION_POST_APPLY_latest.json",
    )
    adoption = _read_optional_json(adoption_path)
    plan = _read_optional_json(plan_path)
    apply = _read_optional_json(apply_path)
    post_apply = _read_optional_json(post_apply_path)
    post_apply_commands = (
        apply.get("post_apply_commands")
        if isinstance(apply.get("post_apply_commands"), list)
        else []
    )
    adoption_artifacts = adoption.get("artifacts") if isinstance(adoption.get("artifacts"), dict) else {}
    candidate_paths = plan.get("candidate_paths") if isinstance(plan.get("candidate_paths"), dict) else {}
    active_paths = plan.get("active_paths") if isinstance(plan.get("active_paths"), dict) else {}
    activation_apply_decision = str(apply.get("decision") or "")
    activation_apply_mutation_performed = bool(apply.get("mutation_performed"))
    activation_apply_applied = bool(
        activation_apply_decision == "APPLIED_ALIAS_SWITCH" and activation_apply_mutation_performed
    )
    activation_post_apply_decision = str(post_apply.get("decision") or "")
    activation_post_apply_mutations_performed = bool(post_apply.get("post_apply_mutations_performed"))
    activation_post_apply_completed = bool(
        activation_post_apply_decision == "POST_APPLY_REFRESH_COMPLETED"
        and activation_post_apply_mutations_performed
    )
    return {
        "adoption_candidate_ready": bool(adoption.get("candidate_ready_for_activation")),
        "adoption_candidate_report": str(adoption_path) if adoption_path else None,
        "activation_plan_ready": str(plan.get("decision")) == "READY_FOR_VEDTAK_ACTIVATION",
        "activation_plan_report": str(plan_path) if plan_path else None,
        "activation_plan_strategy": plan.get("recommended_strategy"),
        "activation_apply_decision": activation_apply_decision or None,
        "activation_apply_ready": activation_apply_decision == "READY_FOR_VEDTAK_APPLY",
        "activation_apply_report": str(apply_path) if apply_path else None,
        "activation_apply_mutation_performed": activation_apply_mutation_performed,
        "activation_apply_applied": activation_apply_applied,
        "activation_allowed_without_vedtak": bool(
            adoption.get("activation_allowed_without_vedtak")
            or plan.get("activation_allowed_without_vedtak")
        ),
        "candidate_dataset_dir": candidate_paths.get("candidate_dataset_dir")
        or adoption_artifacts.get("candidate_dataset_dir"),
        "active_dataset_dir": active_paths.get("foundation_dataset_dir") or str(FOUNDATION_DATASET_DIR),
        "post_apply_command_count": len(post_apply_commands),
        "activation_apply_command": _activation_apply_command(plan_path),
        "activation_post_apply_report": str(post_apply_path) if post_apply_path else None,
        "activation_post_apply_decision": activation_post_apply_decision or None,
        "activation_post_apply_waiting_for_activation": (
            activation_post_apply_decision == "WAITING_FOR_ACTIVATION_APPLY"
        ),
        "activation_post_apply_ready": activation_post_apply_decision == "READY_FOR_POST_APPLY_REFRESH",
        "activation_post_apply_completed": activation_post_apply_completed,
        "activation_post_apply_mutations_performed": activation_post_apply_mutations_performed,
        "activation_post_apply_next_required_action": post_apply.get("next_required_action"),
        "activation_post_apply_command": _activation_post_apply_command(apply_path),
        "training_allowed": False,
    }


def _artifact_self_reference_check(name: str, report: dict[str, Any], latest_path: Path) -> dict[str, Any]:
    json_path = Path(str(report.get("json_path") or ""))
    md_path = Path(str(report.get("md_path") or ""))
    return _check(
        f"{name} latest records timestamped artifact paths",
        bool(report.get("json_path"))
        and bool(report.get("md_path"))
        and json_path.exists()
        and md_path.exists()
        and json_path.parent == latest_path.parent
        and md_path.parent == latest_path.parent
        and json_path.name != latest_path.name
        and json_path.name.endswith(".json")
        and md_path.name.endswith(".md"),
        {
            "latest_path": str(latest_path),
            "json_path": report.get("json_path"),
            "md_path": report.get("md_path"),
        },
    )


def _balanced_label_counts(counts: dict[str, Any], *, required_classes: int = 3) -> bool:
    if len(counts) != required_classes:
        return False
    values = [int(v) for v in counts.values()]
    return bool(values) and min(values) > 0 and len(set(values)) == 1


def _run_control_command(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [str(REPO / "scripts/entry_next_edge_control.sh"), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": ["scripts/entry_next_edge_control.sh", *args],
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _git_status_short() -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "status", "--short"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return [f"git status failed: {proc.stderr.strip()}"]
    return proc.stdout.splitlines()


def _feature_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    objective_rows = report.get("foundation_objective_coverage")
    objective_rows = objective_rows if isinstance(objective_rows, list) else []
    objective_by_name = {
        str(row.get("objective")): row
        for row in objective_rows
        if isinstance(row, dict)
    }
    expected_objectives = set(REQUIRED_FOUNDATION_OBJECTIVE_FEATURES)
    objective_liveness_rows = report.get("foundation_objective_liveness")
    objective_liveness_rows = objective_liveness_rows if isinstance(objective_liveness_rows, list) else []
    objective_liveness_by_key = {
        (str(row.get("split")), str(row.get("objective"))): row
        for row in objective_liveness_rows
        if isinstance(row, dict)
    }
    source_liveness_rows = report.get("foundation_source_field_liveness")
    source_liveness_rows = source_liveness_rows if isinstance(source_liveness_rows, list) else []
    source_liveness_by_key = {
        (str(row.get("split")), str(row.get("source_field"))): row
        for row in source_liveness_rows
        if isinstance(row, dict)
    }
    emitted_contracts = report.get("emitted_contracts") if isinstance(report.get("emitted_contracts"), dict) else {}
    source_contract_details = {
        str(split): {
            "source_field_count": int((contract or {}).get("foundation_structure_source_field_count") or 0),
            "source_missing_count": int((contract or {}).get("foundation_structure_source_missing_count") or 0),
            "source_missing": list((contract or {}).get("foundation_structure_source_missing") or []),
        }
        for split, contract in emitted_contracts.items()
        if isinstance(contract, dict)
    }
    return [
        _artifact_self_reference_check("feature audit", report, FEATURE_AUDIT_LATEST),
        _check("feature audit PASS", str(report.get("decision")) == "PASS", {"path": str(FEATURE_AUDIT_LATEST)}),
        _check("feature audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check(
            "feature audit points at active foundation dataset",
            _dataset_path_matches(report.get("dataset_dir"), FOUNDATION_DATASET_DIR),
            {"dataset_dir": report.get("dataset_dir")},
        ),
        _check(
            "feature audit foundation structure version matches code",
            str(report.get("foundation_structure_feature_version")) == FOUNDATION_STRUCTURE_FEATURE_VERSION,
            {
                "audit_foundation_structure_feature_version": report.get("foundation_structure_feature_version"),
                "code_foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
                "required_action": (
                    "re-materialize sequence structure manifest, rebuild the foundation seq146 dataset, "
                    "then rerun feature/target/specialist audits"
                ),
            },
        ),
        _check(
            "all required foundation features selected",
            int(report.get("foundation_missing_from_manifest_count") or 0) == 0
            and bool(report.get("manifest_foundation_all_required_selected")),
            {
                "foundation_required_feature_count": report.get("foundation_required_feature_count"),
                "selected_feature_count": report.get("selected_feature_count"),
            },
        ),
        _check(
            "feature audit covers exact foundation objective features",
            bool(report.get("foundation_objective_coverage_all_present"))
            and expected_objectives.issubset(set(objective_by_name))
            and all(
                int((objective_by_name.get(name) or {}).get("missing_count") or 0) == 0
                for name in expected_objectives
            ),
            {
                "expected_objectives": sorted(expected_objectives),
                "observed_objectives": sorted(objective_by_name),
                "coverage": objective_rows,
            },
        ),
        _check(
            "feature audit validates exact foundation objective liveness per split",
            bool(report.get("foundation_objective_liveness_all_live"))
            and all(
                (split, objective) in objective_liveness_by_key
                and int((objective_liveness_by_key.get((split, objective)) or {}).get("observed_count") or 0)
                == len(required_features)
                and int((objective_liveness_by_key.get((split, objective)) or {}).get("missing_count") or 0) == 0
                and int((objective_liveness_by_key.get((split, objective)) or {}).get("nonfinite_count") or 0) == 0
                and int((objective_liveness_by_key.get((split, objective)) or {}).get("near_constant_count") or 0) == 0
                and float((objective_liveness_by_key.get((split, objective)) or {}).get("mean_active_rate") or 0.0)
                >= float(report.get("min_required_objective_active_rate") or 0.0)
                for split in ("train", "val", "test")
                for objective, required_features in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.items()
            ),
            {
                "expected_objectives": sorted(expected_objectives),
                "min_required_objective_active_rate": report.get("min_required_objective_active_rate"),
                "objective_liveness": objective_liveness_rows,
            },
        ),
        _check(
            "feature audit validates foundation source fields per split",
            bool(source_contract_details)
            and all(
                row["source_field_count"] == len(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
                and row["source_missing_count"] == 0
                and not row["source_missing"]
                for row in source_contract_details.values()
            ),
            {
                "expected_source_field_count": len(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
                "splits": source_contract_details,
            },
        ),
        _check(
            "feature audit validates foundation source-field liveness per split",
            bool(report.get("foundation_source_field_liveness_all_live"))
            and all(
                (split, source_field) in source_liveness_by_key
                and bool((source_liveness_by_key.get((split, source_field)) or {}).get("observed"))
                and int((source_liveness_by_key.get((split, source_field)) or {}).get("nonfinite_count") or 0) == 0
                and not bool((source_liveness_by_key.get((split, source_field)) or {}).get("near_constant"))
                and int((source_liveness_by_key.get((split, source_field)) or {}).get("active_count") or 0)
                >= int(report.get("min_required_source_active_count") or 0)
                and float((source_liveness_by_key.get((split, source_field)) or {}).get("active_rate") or 0.0)
                >= float(report.get("min_required_source_active_rate") or 0.0)
                for split in ("train", "val", "test")
                for source_field in FOUNDATION_STRUCTURE_SOURCE_FIELDS
            ),
            {
                "expected_source_field_count": len(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
                "min_required_source_active_rate": report.get("min_required_source_active_rate"),
                "min_required_source_active_count": report.get("min_required_source_active_count"),
                "source_field_liveness": source_liveness_rows,
            },
        ),
    ]


def _target_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    head_contract = report.get("target_head_contract") if isinstance(report.get("target_head_contract"), dict) else {}
    active_heads = set(str(x) for x in head_contract.get("active_training_heads", []))
    blocked_heads = set(str(x) for x in head_contract.get("blocked_heads", []))
    liveness = head_contract.get("head_target_liveness") if isinstance(head_contract.get("head_target_liveness"), dict) else {}
    return [
        _artifact_self_reference_check("target audit", report, TARGET_AUDIT_LATEST),
        _check("target audit PASS", str(report.get("decision")) == "PASS", {"path": str(TARGET_AUDIT_LATEST)}),
        _check("target audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check(
            "target audit points at active foundation dataset",
            _dataset_path_matches(report.get("dataset_dir"), FOUNDATION_DATASET_DIR),
            {"dataset_dir": report.get("dataset_dir")},
        ),
        _check("target contract present", isinstance(report.get("target_contract"), dict), {"target_contract": report.get("target_contract")}),
        _check(
            "target head contract present",
            bool(head_contract),
            {"target_head_contract_keys": sorted(head_contract.keys()) if head_contract else []},
        ),
        _check(
            "target head contract activates expected training heads",
            set(EXPECTED_ACTIVE_TRAINING_HEADS).issubset(active_heads),
            {"expected": list(EXPECTED_ACTIVE_TRAINING_HEADS), "actual": sorted(active_heads)},
        ),
        _check(
            "target head contract has exact active training head set",
            active_heads == set(EXPECTED_ACTIVE_TRAINING_HEADS),
            {"expected": list(EXPECTED_ACTIVE_TRAINING_HEADS), "actual": sorted(active_heads)},
        ),
        _check(
            "target head contract blocks hold-horizon until target is live",
            set(EXPECTED_BLOCKED_HEADS).issubset(blocked_heads),
            {"expected_blocked": list(EXPECTED_BLOCKED_HEADS), "actual_blocked": sorted(blocked_heads)},
        ),
        _check(
            "target head contract has exact blocked head set",
            blocked_heads == set(EXPECTED_BLOCKED_HEADS),
            {"expected_blocked": list(EXPECTED_BLOCKED_HEADS), "actual_blocked": sorted(blocked_heads)},
        ),
        _check(
            "active optional head targets are live",
            all(bool((liveness.get(head) or {}).get("live_all_splits")) for head in EXPECTED_ACTIVE_TRAINING_HEADS if head not in {"direction", "tradable", "path_quality", "mfe_first_n", "bad_path", "clean_edge", "survival"}),
            {"head_liveness": {head: (liveness.get(head) or {}).get("live_all_splits") for head in EXPECTED_ACTIVE_TRAINING_HEADS}},
        ),
    ]


def _specialist_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    required_training_specialists = {
        str(name) for name in report.get("required_training_specialists", []) if str(name)
    }
    counts = {
        str(row.get("specialist")): int(row.get("signal_feature_count") or 0)
        for row in report.get("specialist_counts", [])
    }
    requirement_rows = {
        str(row.get("requirement")): bool(row.get("all_mapped_to_expected"))
        for row in report.get("foundation_requirements", [])
    }
    objective_rows = {
        str(row.get("objective")): row
        for row in report.get("foundation_objective_routing", [])
        if isinstance(row, dict)
    }
    model_contract = report.get("specialist_model_contract")
    model_contract = model_contract if isinstance(model_contract, dict) else {}
    expected_model_contract = SPECIALIST_MODEL_CONTRACT
    model_contract_keys = {str(name) for name in model_contract}
    expected_model_contract_keys = {str(name) for name in expected_model_contract}
    model_owned_objectives = {
        str(specialist): tuple(str(x) for x in (spec or {}).get("owned_objectives") or ())
        for specialist, spec in model_contract.items()
        if isinstance(spec, dict)
    }
    expected_owned_objectives = {
        str(specialist): tuple(str(x) for x in spec.get("owned_objectives") or ())
        for specialist, spec in expected_model_contract.items()
    }
    model_objective_owners: dict[str, list[str]] = {}
    model_support_heads: dict[str, tuple[str, ...]] = {}
    model_signal_families: dict[str, tuple[str, ...]] = {}
    for specialist, spec in model_contract.items():
        if not isinstance(spec, dict):
            continue
        specialist_name = str(specialist)
        model_support_heads[specialist_name] = tuple(str(x) for x in spec.get("supports_heads") or ())
        model_signal_families[specialist_name] = tuple(str(x) for x in spec.get("primary_signal_families") or ())
        for objective in spec.get("owned_objectives") or ():
            model_objective_owners.setdefault(str(objective), []).append(specialist_name)
    liveness_rows = report.get("specialist_input_liveness")
    liveness_rows = liveness_rows if isinstance(liveness_rows, list) else []
    liveness_by_key = {
        (str(row.get("split")), str(row.get("specialist"))): row
        for row in liveness_rows
        if isinstance(row, dict)
    }
    arch = report.get("architecture_contract") if isinstance(report.get("architecture_contract"), dict) else {}
    recommended_fusion = arch.get("recommended_fusion") if isinstance(arch.get("recommended_fusion"), dict) else {}
    architecture_active_heads = set(
        str(head) for head in (recommended_fusion.get("active_heads") or recommended_fusion.get("heads") or []) if str(head)
    )
    architecture_blocked_heads = set(str(head) for head in (recommended_fusion.get("blocked_heads") or []) if str(head))
    trainer_probe = _trainer_specialist_contract_probe()
    checks = [
        _artifact_self_reference_check("specialist audit", report, SPECIALIST_AUDIT_LATEST),
        _check("specialist audit PASS", str(report.get("decision")) == "PASS", {"path": str(SPECIALIST_AUDIT_LATEST)}),
        _check("specialist audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check(
            "specialist audit points at active foundation dataset",
            _dataset_path_matches(report.get("dataset_dir"), FOUNDATION_DATASET_DIR),
            {"dataset_dir": report.get("dataset_dir")},
        ),
        _check("specialist signal dim is 146", int(report.get("signal_field_count") or 0) == 146),
        _check("specialist selected extension count is 105", int(report.get("selected_feature_count") or 0) == 105),
        _check("specialist architecture input dim is 146", int(arch.get("input_dim") or 0) == 146),
        _check(
            "specialist architecture active heads match target training contract",
            architecture_active_heads == set(EXPECTED_ACTIVE_TRAINING_HEADS),
            {
                "expected_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "architecture_active_heads": sorted(architecture_active_heads),
            },
        ),
        _check(
            "specialist architecture blocked heads match target training contract",
            architecture_blocked_heads == set(EXPECTED_BLOCKED_HEADS)
            and not (architecture_active_heads & architecture_blocked_heads),
            {
                "expected_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "architecture_blocked_heads": sorted(architecture_blocked_heads),
                "active_blocked_overlap": sorted(architecture_active_heads & architecture_blocked_heads),
            },
        ),
        _check(
            "all required specialists have signal fields",
            all(counts.get(name, 0) > 0 for name in REQUIRED_SPECIALISTS),
            {"specialist_signal_counts": counts},
        ),
        _check(
            "specialist audit has exact required training specialist set",
            required_training_specialists == set(REQUIRED_SPECIALISTS),
            {
                "expected_required_training_specialists": list(REQUIRED_SPECIALISTS),
                "actual_required_training_specialists": sorted(required_training_specialists),
            },
        ),
        _check(
            "all required specialists have live input features per split",
            bool(report.get("specialist_input_liveness_all_live"))
            and all(
                (split, specialist) in liveness_by_key
                and int((liveness_by_key.get((split, specialist)) or {}).get("live_feature_count") or 0)
                >= int((liveness_by_key.get((split, specialist)) or {}).get("min_required_live_feature_count") or 1)
                and int((liveness_by_key.get((split, specialist)) or {}).get("nonfinite_count") or 0) == 0
                and float((liveness_by_key.get((split, specialist)) or {}).get("mean_active_rate") or 0.0) > 0.0
                for split in ("train", "val", "test")
                for specialist in REQUIRED_SPECIALISTS
            ),
            {
                "required_specialists": list(REQUIRED_SPECIALISTS),
                "specialist_input_liveness": liveness_rows,
            },
        ),
        _check(
            "all foundation requirements are mapped to specialist encoders",
            all(requirement_rows.get(name, False) for name in REQUIRED_FOUNDATION_REQUIREMENTS),
            {"foundation_requirements": requirement_rows},
        ),
        _check(
            "all exact foundation objective features are routed to expected specialists",
            bool(report.get("foundation_objective_routing_all_present_and_expected"))
            and all(
                bool((objective_rows.get(name) or {}).get("all_present_and_routed_to_expected"))
                for name in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
            ),
            {
                "foundation_objective_routing": {
                    name: {
                        "expected_specialist": (row or {}).get("expected_specialist"),
                        "required_count": (row or {}).get("required_count"),
                        "routed_to_expected_count": (row or {}).get("routed_to_expected_count"),
                        "missing_count": (row or {}).get("missing_count"),
                        "misrouted_count": (row or {}).get("misrouted_count"),
                    }
                    for name, row in objective_rows.items()
                }
            },
        ),
        _check(
            "specialist audit has valid specialist model contract",
            bool(report.get("specialist_model_contract_valid"))
            and not report.get("specialist_model_contract_failures"),
            {
                "specialist_model_contract_valid": report.get("specialist_model_contract_valid"),
                "specialist_model_contract_failures": report.get("specialist_model_contract_failures"),
            },
        ),
        _check(
            "specialist model contract has exact trainable specialist set",
            model_contract_keys == expected_model_contract_keys == set(REQUIRED_SPECIALISTS),
            {
                "expected_specialists": sorted(expected_model_contract_keys),
                "contract_specialists": sorted(model_contract_keys),
            },
        ),
        _check(
            "specialist model contract owns exact roadmap objectives",
            {
                objective: sorted(owners)
                for objective, owners in model_objective_owners.items()
                if objective in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
            }
            == {
                objective: [specialist]
                for specialist, objectives in expected_owned_objectives.items()
                for objective in objectives
            },
            {
                "model_objective_owners": {
                    objective: sorted(owners)
                    for objective, owners in model_objective_owners.items()
                },
                "expected_owned_objectives": expected_owned_objectives,
            },
        ),
        _check(
            "specialist model contract matches registry owned objectives",
            model_owned_objectives == expected_owned_objectives,
            {
                "model_owned_objectives": model_owned_objectives,
                "expected_owned_objectives": expected_owned_objectives,
            },
        ),
        _check(
            "specialist model contract support heads are active target heads",
            all(
                bool(model_support_heads.get(specialist))
                and set(model_support_heads.get(specialist) or ()).issubset(set(EXPECTED_ACTIVE_TRAINING_HEADS))
                for specialist in REQUIRED_SPECIALISTS
            ),
            {
                "model_support_heads": model_support_heads,
                "expected_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
            },
        ),
        _check(
            "specialist model contract declares signal families for every specialist",
            all(bool(model_signal_families.get(specialist)) for specialist in REQUIRED_SPECIALISTS),
            {"model_signal_families": model_signal_families},
        ),
        _check(
            "trainer specialist-fusion loader accepts current audit contract",
            bool(trainer_probe.get("ok")),
            trainer_probe,
        ),
        _check(
            "trainer specialist-fusion loader returns required trainable specialists",
            set(REQUIRED_SPECIALISTS).issubset(set(trainer_probe.get("loaded_specialists") or [])),
            trainer_probe,
        ),
        _check(
            "trainer specialist-fusion loader returns exact trainable specialist set",
            set(trainer_probe.get("loaded_specialists") or []) == set(REQUIRED_SPECIALISTS),
            trainer_probe,
        ),
        _check(
            "trainer specialist-fusion loader excludes non-required specialist groups",
            not (
                {"neutral_bridge_anchor", "unmapped", "price_action_candle_encoder"}
                & set(trainer_probe.get("loaded_specialists") or [])
            ),
            trainer_probe,
        ),
    ]
    return checks


def _trainer_specialist_contract_probe() -> dict[str, Any]:
    details: dict[str, Any] = {
        "audit_json": str(SPECIALIST_AUDIT_LATEST),
        "expected_signal_dim": 146,
    }
    try:
        from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _load_specialist_fusion_contract

        indices, meta = _load_specialist_fusion_contract(
            SPECIALIST_AUDIT_LATEST,
            expected_signal_dim=146,
        )
        loaded = sorted(str(name) for name in indices)
        group_counts = {str(name): int(len(vals)) for name, vals in indices.items()}
        details.update(
            {
                "ok": True,
                "loaded_specialists": loaded,
                "group_feature_counts": group_counts,
                "meta_signal_field_count": int(meta.get("signal_field_count") or 0),
                "meta_selected_feature_count": int(meta.get("selected_feature_count") or 0),
                "meta_audit_created_utc": str(meta.get("audit_created_utc") or ""),
                "trainable_specialists": list(meta.get("trainable_specialists") or []),
                "excluded_specialist_groups": dict(meta.get("excluded_specialist_groups") or {}),
            }
        )
    except Exception as exc:
        details.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "loaded_specialists": [],
                "group_feature_counts": {},
            }
        )
    return details


def _execution_hygiene_checks() -> list[dict[str, Any]]:
    hygiene = run_worktree_hygiene(
        argparse.Namespace(out_dir=str(WORKTREE_HYGIENE_OUT_DIR), fail_on_dirty=False, quiet=True)
    )
    status_short = [f"{entry.get('status')} {entry.get('path')}" for entry in hygiene.get("status_entries", [])]
    return [
        _check(
            "git worktree clean before real smoke train",
            bool(hygiene.get("real_smoke_train_allowed")),
            {
                "dirty_count": len(status_short),
                "status_short_first_80": status_short[:80],
                "worktree_hygiene_json": str(WORKTREE_HYGIENE_LATEST),
                "foundation_cleanup_dirty_count": hygiene.get("foundation_cleanup_dirty_count"),
                "review_before_stage_dirty_count": hygiene.get("review_before_stage_dirty_count"),
                "foundation_change_set_isolated": hygiene.get("foundation_change_set_isolated"),
                "clean_git_resolution": hygiene.get("clean_git_resolution"),
                "required_before_real_train": "commit/stash/remove dirty worktree changes, then rerun train-readiness",
            },
        )
    ]


def _foundation_guardrail_checks(report: dict[str, Any], command: dict[str, Any]) -> list[dict[str, Any]]:
    cases = {
        str(row.get("name")): row
        for row in report.get("cases", [])
        if isinstance(row, dict)
    }
    readiness_policy_checks = {
        str(row.get("name")): row
        for row in report.get("readiness_policy_checks", [])
        if isinstance(row, dict)
    }
    required_cases = (
        "control_preview_shadow_blocked",
        "control_start_shadow_blocked",
        "control_verify_shadow_blocked",
        "direct_no_xgb_shadow_launcher_blocked",
        "legacy_plan_verifier_closed",
        "generic_train_blocked",
    )
    required_policy_checks = (
        "readiness_policy_snapshot_report_only",
        "readiness_policy_command_set_exact",
        "readiness_policy_command_schema_complete",
        "readiness_policy_allowed_now_has_no_vedtak_placeholders",
        "readiness_policy_adoption_candidate_does_not_activate_without_vedtak",
        "readiness_policy_safe_now_verify",
        "readiness_policy_safe_now_foundation_guardrails",
        "readiness_policy_safe_now_foundation_activation_plan",
        "readiness_policy_safe_now_foundation_activation_apply_dry_run",
        "readiness_policy_safe_now_foundation_activation_post_apply_dry_run",
        "readiness_policy_safe_now_candidate_readiness_report",
        "readiness_policy_safe_now_replay_readiness_report",
        "readiness_policy_safe_now_stage_foundation_cleanup_dry_run",
        "readiness_policy_blocks_smoke_train",
        "readiness_policy_blocks_foundation_activation_apply",
        "readiness_policy_blocks_foundation_activation_post_apply",
        "readiness_policy_blocks_candidate_train",
        "readiness_policy_blocks_selective_edge",
        "readiness_policy_blocks_replay_evidence",
        "readiness_policy_blocks_iql_distill",
        "readiness_policy_blocks_iql_replay_evidence",
        "readiness_policy_blocks_iql_compare",
        "readiness_policy_blocks_preview_shadow",
        "readiness_policy_blocks_start_shadow",
        "readiness_policy_blocks_live",
        "readiness_policy_candidate_train_declares_trainer",
        "readiness_policy_iql_distill_declares_iql_side_effect",
        "readiness_policy_shadow_live_declares_live_touch",
    )
    return [
        _check(
            "foundation guardrails command exits cleanly",
            command.get("returncode") == 0,
            {"stderr": command.get("stderr"), "cmd": command.get("cmd")},
        ),
        _check(
            "foundation guardrails latest artifact exists",
            FOUNDATION_GUARDRAILS_LATEST.exists(),
            {"path": str(FOUNDATION_GUARDRAILS_LATEST)},
        ),
        _check(
            "foundation guardrails PASS",
            str(report.get("decision")) == "PASS",
            {"path": str(FOUNDATION_GUARDRAILS_LATEST), "decision": report.get("decision")},
        ),
        _check(
            "foundation guardrails keep promotion/shadow/live closed",
            bool(report.get("promotion_shadow_live_allowed")) is False,
            {"promotion_shadow_live_allowed": report.get("promotion_shadow_live_allowed")},
        ),
        _check(
            "foundation guardrails cover closed legacy shadow/train surfaces",
            all(name in cases and str((cases.get(name) or {}).get("status")) == "PASS" for name in required_cases),
            {"required_cases": list(required_cases), "observed_cases": sorted(cases)},
        ),
        _check(
            "foundation guardrails validate readiness command policy",
            all(
                name in readiness_policy_checks and bool((readiness_policy_checks.get(name) or {}).get("ok"))
                for name in required_policy_checks
            ),
            {
                "required_policy_checks": list(required_policy_checks),
                "observed_policy_checks": sorted(readiness_policy_checks),
            },
        ),
    ]



def _smoke_dataset_checks(
    report: dict[str, Any],
    artifact_fingerprints: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    splits = report.get("splits") if isinstance(report.get("splits"), dict) else {}
    expected_rows = {"train": 4095, "val": 1536, "test": 1536}
    provenance = report.get("audit_provenance") if isinstance(report.get("audit_provenance"), dict) else {}
    provenance_artifacts = provenance.get("artifacts") if isinstance(provenance.get("artifacts"), dict) else {}
    required_audit_artifacts = ("feature_audit", "target_audit", "specialist_audit")

    def audit_artifact_matches(name: str) -> bool:
        manifest_row = provenance_artifacts.get(name) if isinstance(provenance_artifacts.get(name), dict) else {}
        active_row = artifact_fingerprints.get(name) if isinstance(artifact_fingerprints.get(name), dict) else {}
        return (
            bool(manifest_row.get("exists"))
            and bool(active_row.get("exists"))
            and str(manifest_row.get("path")) == str(active_row.get("path"))
            and isinstance(manifest_row.get("sha256"), str)
            and str(manifest_row.get("sha256")) == str(active_row.get("sha256"))
            and len(str(manifest_row.get("sha256"))) == 64
        )

    def source_manifest_hash_matches(split: str) -> bool:
        row = splits.get(split) if isinstance(splits.get(split), dict) else {}
        path = Path(str(row.get("source_manifest") or ""))
        expected_sha = row.get("source_manifest_sha256")
        return (
            path.exists()
            and isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and _sha256_file(path) == expected_sha
        )

    def output_hashes_match(split: str) -> bool:
        row = splits.get(split) if isinstance(splits.get(split), dict) else {}
        out_path = Path(str(row.get("out_path") or ""))
        out_manifest = Path(str(row.get("out_manifest") or ""))
        expected_parquet_sha = row.get("out_parquet_sha256")
        expected_manifest_sha = row.get("out_manifest_sha256")
        return (
            out_path.exists()
            and out_manifest.exists()
            and isinstance(expected_parquet_sha, str)
            and len(expected_parquet_sha) == 64
            and isinstance(expected_manifest_sha, str)
            and len(expected_manifest_sha) == 64
            and _sha256_file(out_path) == expected_parquet_sha
            and _sha256_file(out_manifest) == expected_manifest_sha
        )

    def file_hash_for(raw_path: Any) -> str | None:
        raw = str(raw_path or "")
        if not raw:
            return None
        path = Path(raw)
        return _sha256_file(path) if path.is_file() else None

    return [
        _check("smoke dataset manifest exists", SMOKE_DATASET_MANIFEST.exists(), {"path": str(SMOKE_DATASET_MANIFEST)}),
        _check("smoke dataset schema is foundation smoke v1", report.get("schema_version") == "entry_foundation_seq146_smoke_dataset_v1"),
        _check(
            "smoke dataset points at active foundation dataset",
            _dataset_path_matches(report.get("source_dir"), FOUNDATION_DATASET_DIR),
            {"source_dir": report.get("source_dir")},
        ),
        _check(
            "smoke split row counts match current readiness contract",
            all(int((splits.get(split) or {}).get("rows") or 0) == rows for split, rows in expected_rows.items()),
            {"expected_rows": expected_rows, "actual_rows": {split: (splits.get(split) or {}).get("rows") for split in expected_rows}},
        ),
        _check(
            "smoke split labels are class-balanced",
            all(_balanced_label_counts((splits.get(split) or {}).get("label_counts") or {}) for split in expected_rows),
            {"label_counts": {split: (splits.get(split) or {}).get("label_counts") for split in expected_rows}},
        ),
        _check(
            "smoke dataset records active audit artifact provenance",
            str(provenance.get("schema_version")) == "entry_foundation_smoke_dataset_audit_provenance_v1"
            and bool(provenance.get("all_artifacts_present"))
            and bool(provenance.get("all_artifact_hashes_present"))
            and all(audit_artifact_matches(name) for name in required_audit_artifacts),
            {
                "required_audit_artifacts": list(required_audit_artifacts),
                "manifest_artifacts": provenance_artifacts,
                "active_artifact_fingerprints": {
                    name: artifact_fingerprints.get(name)
                    for name in required_audit_artifacts
                },
            },
        ),
        _check(
            "smoke dataset records source split manifest hashes",
            all(source_manifest_hash_matches(split) for split in expected_rows),
            {
                "source_manifest_hashes": {
                    split: {
                        "source_manifest": (splits.get(split) or {}).get("source_manifest"),
                        "source_manifest_sha256": (splits.get(split) or {}).get("source_manifest_sha256"),
                    }
                    for split in expected_rows
                }
            },
        ),
        _check(
            "smoke dataset output parquet and manifest hashes match files",
            all(output_hashes_match(split) for split in expected_rows),
            {
                "output_hashes": {
                    split: {
                        "out_path": (splits.get(split) or {}).get("out_path"),
                        "out_parquet_sha256": (splits.get(split) or {}).get("out_parquet_sha256"),
                        "out_parquet_sha256_observed": file_hash_for((splits.get(split) or {}).get("out_path")),
                        "out_manifest": (splits.get(split) or {}).get("out_manifest"),
                        "out_manifest_sha256": (splits.get(split) or {}).get("out_manifest_sha256"),
                        "out_manifest_sha256_observed": file_hash_for((splits.get(split) or {}).get("out_manifest")),
                    }
                    for split in expected_rows
                }
            },
        ),
    ]


def _smoke_bundle_audit_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    split_rows = {
        split: int((row or {}).get("rows") or 0)
        for split, row in (report.get("splits") or {}).items()
        if isinstance(row, dict)
    }
    gate_errors = {
        split: (((row or {}).get("specialist_gate") or {}).get("row_sum_max_abs_error"))
        for split, row in (report.get("splits") or {}).items()
        if isinstance(row, dict)
    }
    return [
        _check("reference smoke bundle audit PASS", str(report.get("decision")) == "PASS", {"path": str(SMOKE_BUNDLE_AUDIT_LATEST)}),
        _check("reference smoke bundle audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check("reference smoke bundle audit used smoke dataset", str(report.get("dataset_dir")) == str(FOUNDATION_SMOKE_DATASET_DIR)),
        _check("reference smoke bundle audit strict-loaded seq146 bundle", int(bundle.get("seq_input_dim") or 0) == 146),
        _check("reference smoke bundle audit has specialist fusion", bool(bundle.get("specialist_fusion_enabled"))),
        _check(
            "reference smoke bundle audit covered val/test rows",
            split_rows.get("val", 0) == 1536 and split_rows.get("test", 0) == 1536,
            {"split_rows": split_rows},
        ),
        _check(
            "reference smoke bundle audit specialist gates sum to one",
            all(value is not None and float(value) <= 1e-4 for value in gate_errors.values()),
            {"gate_row_sum_max_abs_error": gate_errors},
        ),
    ]


def _wrapper_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dry = _run_control_command(["smoke-train", "--vedtak", "READINESS_DRY_RUN_ONLY", "--dry-run", "--require-edge-audit"])
    blocked_train = _run_control_command(["train"])
    blocked_live = _run_control_command(["live"])
    stdout = str(dry.get("stdout") or "")
    blocked_text = str(blocked_train.get("stderr") or "") + str(blocked_live.get("stderr") or "")
    wrapper_text = (REPO / "scripts/run_entry_foundation_seq146_smoke_train.sh").read_text(encoding="utf-8")
    checks = [
        _check("smoke train dry-run exits cleanly", int(dry["returncode"]) == 0, {"stderr": dry.get("stderr")}),
        _check("smoke train dry-run documents foundation verify preflight", "verify --quiet" in stdout),
        _check("smoke train dry-run documents foundation guardrails preflight", "foundation-guardrails --quiet" in stdout),
        _check("smoke train dry-run documents train-readiness preflight", "train-readiness --quiet" in stdout),
        _check("smoke train dry-run documents pre-train run manifest path", "Pre-train run manifest path:" in stdout),
        _check(
            "smoke train manifest records audit artifact hashes",
            "artifact_sha256" in wrapper_text and "sha256_file" in wrapper_text,
        ),
        _check(
            "smoke train manifest records feature and specialist contracts",
            "preflight_contracts" in wrapper_text
            and "feature_contract_summary" in wrapper_text
            and "specialist_contract_summary" in wrapper_text
            and "foundation_objective_coverage_all_present" in wrapper_text
            and "foundation_objective_liveness_all_live" in wrapper_text
            and "foundation_source_field_liveness_all_live" in wrapper_text
            and "architecture_active_heads" in wrapper_text
            and "architecture_blocked_heads" in wrapper_text
            and "foundation_objective_routing_all_present_and_expected" in wrapper_text
            and "specialist_input_liveness_all_live" in wrapper_text
            and "specialist_model_contract_valid" in wrapper_text
            and "specialist_model_contract" in wrapper_text,
        ),
        _check(
            "smoke train command records explicit edge recipe env",
            "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1" in stdout
            and "ENTRY_AUX_BAD_PATH_WEIGHT=" in stdout
            and "ENTRY_PRED_BALANCE_ALPHA=" in stdout
            and "GX1_V10_CKPT_MONITOR=dir_acc" in stdout
            and "ENTRY_SYMMETRIC_NEGATIVES=1" in stdout
            and "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT=" in stdout
            and "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT=" in stdout
            and "ENTRY_SPECIALIST_GATE_MIN_MEAN=" in stdout
            and "smoke_recipe_env" in wrapper_text,
        ),
        _check("smoke train dry-run includes specialist fusion", "--enable-specialist-fusion" in stdout),
        _check(
            "smoke train dry-run enables expected live auxiliary heads",
            all(flag in stdout for flag in EXPECTED_SMOKE_HEAD_FLAGS) and "--enable-hold-horizon-head" not in stdout,
            {"expected_flags": list(EXPECTED_SMOKE_HEAD_FLAGS)},
        ),
        _check("smoke train dry-run includes post-smoke audit", "Post-smoke audit command:" in stdout and "audit-smoke-bundle" in stdout),
        _check("post-smoke audit dry-run requires active head contract", "--require-head-contract" in stdout),
        _check("post-smoke audit dry-run receives pre-train manifest", "--pretrain-manifest-json" in stdout),
        _check("post-smoke audit dry-run requires edge diagnostics", "--require-edge" in stdout),
        _check("generic train command is blocked", int(blocked_train["returncode"]) == 2, {"stderr": blocked_train.get("stderr")}),
        _check("live command is blocked", int(blocked_live["returncode"]) == 2, {"stderr": blocked_live.get("stderr")}),
        _check("blocked commands point back to foundation verify", "entry_next_edge_control.sh verify" in blocked_text),
        _check("blocked commands point at worktree hygiene", "entry_next_edge_control.sh worktree-hygiene" in blocked_text),
        _check(
            "blocked commands point at foundation cleanup dry-run",
            "entry_next_edge_control.sh stage-foundation-cleanup --dry-run" in blocked_text,
        ),
    ]
    return checks, {
        "smoke_train_dry_run": dry,
        "blocked_train": blocked_train,
        "blocked_live": blocked_live,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    activation = report.get("foundation_activation") if isinstance(report.get("foundation_activation"), dict) else {}
    lines = [
        "# Entry Training Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Next command: `{report['next_allowed_command']}`",
        f"- Foundation contract ready for smoke: `{report['foundation_contract_ready_for_smoke']}`",
        f"- Smoke training allowed with explicit vedtak: `{report['smoke_training_allowed_with_explicit_vedtak']}`",
        f"- Candidate training allowed: `{report['candidate_training_allowed']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        f"- Foundation activation required before smoke: `{report.get('foundation_activation_required_before_smoke')}`",
        f"- Foundation activation apply required before smoke: `{report.get('foundation_activation_apply_required_before_smoke')}`",
        f"- Foundation post-apply refresh required before smoke: `{report.get('foundation_activation_post_apply_required_before_smoke')}`",
        f"- Activation apply ready: `{activation.get('activation_apply_ready')}`",
        f"- Activation apply mutation performed: `{activation.get('activation_apply_mutation_performed')}`",
        f"- Activation post-apply completed: `{activation.get('activation_post_apply_completed')}`",
        f"- Execution blockers: `{len(report.get('execution_blockers') or [])}`",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        lines.append(f"- `{gate['name']}`: {gate['decision']} ({gate['passed']}/{gate['total']} checks)")
    lines.extend(["", "## Failures", ""])
    failures = [
        f"{gate['name']}: {check['name']}"
        for gate in report["gates"]
        for check in gate["checks"]
        if not check["ok"]
    ]
    if failures:
        lines.extend([f"- {failure}" for failure in failures])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    foundation_verify_error: dict[str, Any] | None = None
    try:
        foundation_report = run_foundation_verify(
            argparse.Namespace(audit_doc=str(args.audit_doc), out="", quiet=True, selftest=True)
        )
    except Exception as exc:
        foundation_verify_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        foundation_report = {}
    foundation_guardrail_command = _run_control_command(["foundation-guardrails", "--quiet"])
    foundation_guardrails = (
        _read_json(FOUNDATION_GUARDRAILS_LATEST)
        if FOUNDATION_GUARDRAILS_LATEST.exists()
        else {}
    )
    feature = _read_json(FEATURE_AUDIT_LATEST)
    target = _read_json(TARGET_AUDIT_LATEST)
    specialist = _read_json(SPECIALIST_AUDIT_LATEST)
    smoke_dataset = _read_json(SMOKE_DATASET_MANIFEST)
    smoke_bundle_audit = _read_json(SMOKE_BUNDLE_AUDIT_LATEST)
    wrapper, command_proofs = _wrapper_checks()

    artifacts = {
        "feature_audit": str(FEATURE_AUDIT_LATEST),
        "target_audit": str(TARGET_AUDIT_LATEST),
        "specialist_audit": str(SPECIALIST_AUDIT_LATEST),
        "foundation_guardrails": str(FOUNDATION_GUARDRAILS_LATEST),
        "worktree_hygiene": str(WORKTREE_HYGIENE_LATEST),
        "smoke_dataset_manifest": str(SMOKE_DATASET_MANIFEST),
        "reference_smoke_bundle_audit": str(SMOKE_BUNDLE_AUDIT_LATEST),
    }
    execution_hygiene_checks = _execution_hygiene_checks()
    artifact_fingerprints = _artifact_fingerprints(artifacts)
    gate_checks = {
        "foundation_state": [
            _check(
                "foundation verifier completed",
                foundation_verify_error is None,
                {"foundation_verify_error": foundation_verify_error},
            ),
            _check(
                "foundation verifier selftest passed",
                foundation_verify_error is None and int(foundation_report.get("checks_passed") or 0) >= 80,
                {
                    "checks_passed": foundation_report.get("checks_passed"),
                    "foundation_verify_error": foundation_verify_error,
                },
            ),
            _check("foundation state keeps generic training blocked", bool(foundation_report.get("training_allowed")) is False),
            _check("foundation state marks smoke gate ready", bool(foundation_report.get("smoke_training_gate_ready")) is True),
        ],
        "feature_foundation": _feature_checks(feature),
        "target_foundation": _target_checks(target),
        "specialist_contract": _specialist_checks(specialist),
        "foundation_guardrails": _foundation_guardrail_checks(foundation_guardrails, foundation_guardrail_command),
        "smoke_dataset": _smoke_dataset_checks(smoke_dataset, artifact_fingerprints),
        "reference_smoke_bundle_audit": _smoke_bundle_audit_checks(smoke_bundle_audit),
        "control_surface": wrapper,
        "execution_hygiene": execution_hygiene_checks,
    }
    gate_checks["artifact_provenance"] = _artifact_fingerprint_checks(artifact_fingerprints)
    gates = []
    for name, checks in gate_checks.items():
        passed = sum(1 for check in checks if check["ok"])
        gates.append(
            {
                "name": name,
                "decision": "PASS" if _all_ok(checks) else "FAIL",
                "passed": int(passed),
                "total": int(len(checks)),
                "checks": checks,
            }
        )
    all_failures = [
        {"gate": gate["name"], "check": check["name"], "details": check.get("details") or {}}
        for gate in gates
        for check in gate["checks"]
        if not check["ok"]
    ]
    non_blocking_diagnostic_gate_names = {"reference_smoke_bundle_audit"}
    readiness_blocking_gate_names = set(gate_checks) - non_blocking_diagnostic_gate_names
    failures = [
        failure
        for failure in all_failures
        if str(failure.get("gate")) in readiness_blocking_gate_names
    ]
    diagnostic_failures = [
        failure
        for failure in all_failures
        if str(failure.get("gate")) in non_blocking_diagnostic_gate_names
    ]
    contract_gate_names = set(gate_checks) - {"execution_hygiene"} - non_blocking_diagnostic_gate_names
    foundation_contract_ready = all(
        gate["decision"] == "PASS"
        for gate in gates
        if str(gate.get("name")) in contract_gate_names
    )
    execution_blockers = [
        failure
        for failure in failures
        if failure.get("gate") == "execution_hygiene"
    ]
    foundation_activation = _foundation_activation_summary()
    activation_transition = _activation_transition(
        foundation_contract_ready=bool(foundation_contract_ready),
        foundation_activation=foundation_activation,
    )
    activation_required_before_smoke = bool(activation_transition["activation_required_before_smoke"])
    ready_for_smoke = not failures
    if ready_for_smoke:
        decision = "READY_FOR_VEDTAK_SMOKE_TRAIN"
        next_allowed_command = "scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit"
    elif foundation_contract_ready and execution_blockers:
        decision = "READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN"
        next_allowed_command = (
            "clean git worktree, then scripts/entry_next_edge_control.sh "
            "smoke-train --vedtak <id> --require-edge-audit"
        )
    else:
        decision = "NOT_READY"
        if activation_required_before_smoke and activation_transition.get("next_allowed_command"):
            next_allowed_command = str(activation_transition["next_allowed_command"])
        else:
            next_allowed_command = "fix failing readiness gates, then rerun scripts/entry_next_edge_control.sh train-readiness"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_training_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "foundation_contract_ready_for_smoke": bool(foundation_contract_ready),
        "foundation_activation_required_before_smoke": activation_required_before_smoke,
        "foundation_activation_apply_required_before_smoke": bool(
            activation_transition["activation_apply_required_before_smoke"]
        ),
        "foundation_activation_post_apply_required_before_smoke": bool(
            activation_transition["activation_post_apply_required_before_smoke"]
        ),
        "foundation_activation": foundation_activation,
        "execution_blockers": execution_blockers,
        "smoke_training_allowed_with_explicit_vedtak": bool(ready_for_smoke),
        "candidate_training_allowed": False,
        "promotion_shadow_live_allowed": False,
        "reason_candidate_training_not_allowed": "requires actual smoke-train bundle plus --require-edge audit and offline replay gates",
        "diagnostic_failures": diagnostic_failures,
        "non_blocking_diagnostic_gates": sorted(non_blocking_diagnostic_gate_names),
        "readiness_blocking_gates": sorted(readiness_blocking_gate_names),
        "next_allowed_command": next_allowed_command,
        "foundation_dataset_dir": str(FOUNDATION_DATASET_DIR),
        "smoke_dataset_dir": str(FOUNDATION_SMOKE_DATASET_DIR),
        "artifacts": artifacts,
        "artifact_fingerprints": artifact_fingerprints,
        "gates": gates,
        "failures": failures,
        "command_proofs": command_proofs,
    }
    report["command_proofs"]["foundation_guardrails"] = foundation_guardrail_command
    json_path = out_dir / f"ENTRY_TRAINING_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_TRAINING_READINESS_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_TRAINING_READINESS_latest.json"
    latest_md = out_dir / "ENTRY_TRAINING_READINESS_latest.md"
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
                    "next_allowed_command": report["next_allowed_command"],
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
    ap.add_argument("--audit-doc", default=str(REPO / "docs/ENTRY_FOUNDATION_AUDIT_20260628.md"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
