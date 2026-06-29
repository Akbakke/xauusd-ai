#!/usr/bin/env python3
"""Apply the dataset-alias part of a reviewed Entry foundation activation plan.

Default mode is dry-run. Real apply requires an explicit vedtak and only performs
the filesystem alias switch described by the plan; audit refresh, smoke refresh,
and train-readiness remain separate required follow-up gates.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_foundation_activation_apply_20260629_v1"
REQUIRED_VEDTAK_PREFIX = "ENTRY_FOUNDATION_ACTIVATE_"
DEFAULT_SOURCE_PARQUET = (
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260626_spreadfix/FULL_PLUS_CTX_v3src.parquet"
)
DEFAULT_SEQ_STRUCTURE_MANIFEST = (
    "/home/andre2/GX1_DATA/reports/sequence_structure_feature_layer_20260628_v1/"
    "sequence_structure_feature_layer_manifest.json"
)
DEFAULT_SMOKE_STEM = "v10_foundation_seq146_smoke__HOLD_03B"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _archive_path(active_dataset: Path, suffix: str) -> Path:
    return active_dataset.with_name(f"{active_dataset.name}_{suffix}")


def _activation_plan_has_active_verify_before_train_readiness(plan: dict[str, Any]) -> bool:
    steps = plan.get("activation_steps") if isinstance(plan.get("activation_steps"), list) else []
    commands: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        commands.append(str(step.get("command") or step.get("command_shape") or ""))
    try:
        verify_index = commands.index("scripts/entry_next_edge_control.sh verify --quiet")
        train_index = commands.index("scripts/entry_next_edge_control.sh train-readiness --quiet --no-fail-on-not-ready")
    except ValueError:
        return False
    return verify_index < train_index


def _activation_plan_records_current_control_sources(plan: dict[str, Any]) -> bool:
    checks = plan.get("checks") if isinstance(plan.get("checks"), list) else []
    names = {
        str(check.get("name") or "")
        for check in checks
        if isinstance(check, dict) and check.get("ok") is True
    }
    required = {
        "source pointer contract present: gx1/scripts/apply_entry_foundation_activation_v1.py",
        "source pointer contract present: gx1/scripts/run_entry_foundation_activation_post_apply_v1.py",
        "source pointer contract present: scripts/entry_next_edge_control.sh",
    }
    return required.issubset(names)


def _activation_plan_records_current_adoption_contract(plan: dict[str, Any]) -> bool:
    checks = plan.get("checks") if isinstance(plan.get("checks"), list) else []
    names = {
        str(check.get("name") or "")
        for check in checks
        if isinstance(check, dict) and check.get("ok") is True
    }
    required = {
        "adoption report has expected schema",
        "adoption report has zero failures",
        "adoption candidate gates are all PASS",
        "adoption artifact fingerprints match current files",
    }
    return required.issubset(names) and isinstance(plan.get("adoption_contract"), dict)


def _validate_plan(plan: dict[str, Any], plan_json: Path, *, apply: bool, vedtak: str, archive_suffix: str) -> dict[str, Any]:
    active = plan.get("active_paths") if isinstance(plan.get("active_paths"), dict) else {}
    candidate = plan.get("candidate_paths") if isinstance(plan.get("candidate_paths"), dict) else {}
    active_dataset = Path(str(active.get("foundation_dataset_dir") or ""))
    candidate_dataset = Path(str(candidate.get("candidate_dataset_dir") or ""))
    archive = _archive_path(active_dataset, archive_suffix) if str(active_dataset) else Path("")
    checks = [
        _check("activation plan JSON exists", plan_json.exists(), {"plan_json": str(plan_json)}),
        _check("activation plan is ready", plan.get("decision") == "READY_FOR_VEDTAK_ACTIVATION", {"decision": plan.get("decision")}),
        _check(
            "activation plan includes active verify before train-readiness",
            _activation_plan_has_active_verify_before_train_readiness(plan),
            {"activation_steps": plan.get("activation_steps")},
        ),
        _check(
            "activation plan records current apply/post-apply/control source checks",
            _activation_plan_records_current_control_sources(plan),
            {"checks": plan.get("checks")},
        ),
        _check(
            "activation plan records current adoption artifact contract",
            _activation_plan_records_current_adoption_contract(plan),
            {"checks": plan.get("checks"), "adoption_contract": plan.get("adoption_contract")},
        ),
        _check(
            "activation plan does not allow activation without vedtak",
            plan.get("activation_allowed_without_vedtak") is False,
            {"activation_allowed_without_vedtak": plan.get("activation_allowed_without_vedtak")},
        ),
        _check("candidate dataset exists", candidate_dataset.exists() and candidate_dataset.is_dir(), {"candidate_dataset": str(candidate_dataset)}),
        _check("active dataset path is non-empty", bool(str(active_dataset)), {"active_dataset": str(active_dataset)}),
        _check(
            "candidate and active dataset paths differ",
            str(candidate_dataset) != str(active_dataset),
            {"candidate_dataset": str(candidate_dataset), "active_dataset": str(active_dataset)},
        ),
        _check("archive path is free", bool(str(archive)) and not archive.exists(), {"archive_path": str(archive)}),
        _check(
            "apply requires explicit activation vedtak",
            (not apply) or (bool(vedtak) and vedtak.startswith(REQUIRED_VEDTAK_PREFIX)),
            {"apply": apply, "vedtak": vedtak, "required_prefix": REQUIRED_VEDTAK_PREFIX},
        ),
    ]
    return {
        "active": active,
        "candidate": candidate,
        "active_dataset": active_dataset,
        "candidate_dataset": candidate_dataset,
        "archive_dataset": archive,
        "checks": checks,
    }


def _apply_alias(active_dataset: Path, candidate_dataset: Path, archive_dataset: Path) -> None:
    if not active_dataset.exists():
        raise RuntimeError(f"active dataset path missing: {active_dataset}")
    if active_dataset.is_symlink():
        raise RuntimeError(f"active dataset path is already a symlink: {active_dataset}")
    if archive_dataset.exists():
        raise RuntimeError(f"archive path already exists: {archive_dataset}")
    active_dataset.rename(archive_dataset)
    try:
        os.symlink(candidate_dataset, active_dataset, target_is_directory=True)
    except Exception:
        if not active_dataset.exists() and archive_dataset.exists():
            archive_dataset.rename(active_dataset)
        raise


def _post_apply_commands(active: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = str(active.get("foundation_dataset_dir") or "")
    smoke_dataset = str(active.get("foundation_smoke_dataset_dir") or "")
    feature_audit = Path(str(active.get("feature_audit") or ""))
    target_audit = Path(str(active.get("target_audit") or ""))
    specialist_audit = Path(str(active.get("specialist_audit") or ""))
    return [
        {
            "name": "refresh_canonical_feature_audit",
            "mutates": "canonical_feature_audit_latest",
            "argv": [
                ".venv/bin/python",
                "-m",
                "gx1.scripts.audit_entry_foundation_features_v1",
                "--dataset-dir",
                dataset,
                "--source-parquet",
                DEFAULT_SOURCE_PARQUET,
                "--seq-structure-manifest",
                DEFAULT_SEQ_STRUCTURE_MANIFEST,
                "--out-dir",
                str(feature_audit.parent),
                "--quiet",
            ],
        },
        {
            "name": "refresh_canonical_target_audit",
            "mutates": "canonical_target_audit_latest",
            "argv": [
                ".venv/bin/python",
                "-m",
                "gx1.scripts.audit_entry_foundation_targets_v1",
                "--dataset-dir",
                dataset,
                "--out-dir",
                str(target_audit.parent),
                "--quiet",
            ],
        },
        {
            "name": "refresh_canonical_specialist_audit",
            "mutates": "canonical_specialist_audit_latest",
            "argv": [
                ".venv/bin/python",
                "-m",
                "gx1.scripts.audit_entry_specialist_feature_groups_v1",
                "--dataset-dir",
                dataset,
                "--seq-structure-manifest",
                DEFAULT_SEQ_STRUCTURE_MANIFEST,
                "--out-dir",
                str(specialist_audit.parent),
                "--quiet",
            ],
        },
        {
            "name": "refresh_canonical_smoke_dataset",
            "mutates": "canonical_smoke_dataset",
            "argv": [
                ".venv/bin/python",
                "-m",
                "gx1.scripts.materialize_entry_foundation_smoke_dataset_v1",
                "--source-dir",
                dataset,
                "--out-dir",
                smoke_dataset,
                "--stem",
                DEFAULT_SMOKE_STEM,
                "--feature-audit-json",
                str(feature_audit),
                "--target-audit-json",
                str(target_audit),
                "--specialist-audit-json",
                str(specialist_audit),
                "--quiet",
            ],
        },
        {
            "name": "verify_active_foundation_state",
            "mutates": "none",
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "verify",
                "--quiet",
            ],
        },
        {
            "name": "verify_train_readiness",
            "mutates": "readiness_report_only",
            "argv": [
                "scripts/entry_next_edge_control.sh",
                "train-readiness",
                "--quiet",
                "--no-fail-on-not-ready",
            ],
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_json = Path(args.plan_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dry_run_requested = bool(getattr(args, "dry_run", False))
    apply = bool(args.apply)
    vedtak = str(args.vedtak or "")
    archive_suffix = str(args.archive_suffix or "STALE_PRE_DIRECTIONAL_SMC_20260629")
    plan = _read_json(plan_json)
    validation = _validate_plan(plan, plan_json, apply=apply, vedtak=vedtak, archive_suffix=archive_suffix)
    validation["checks"].append(
        _check(
            "dry-run and apply flags are mutually exclusive",
            not (dry_run_requested and apply),
            {"dry_run": dry_run_requested, "apply": apply},
        )
    )
    failures = [check for check in validation["checks"] if not check["ok"]]
    mutation_performed = False
    mutation_error: str | None = None
    if apply and not failures:
        try:
            _apply_alias(
                validation["active_dataset"],
                validation["candidate_dataset"],
                validation["archive_dataset"],
            )
            mutation_performed = True
        except Exception as exc:
            mutation_error = f"{type(exc).__name__}: {exc}"
            failures.append(_check("activation alias switch completed", False, {"error": mutation_error}))
    elif not apply:
        validation["checks"].append(
            _check(
                "dry-run did not mutate filesystem",
                True,
                {
                    "active_dataset": str(validation["active_dataset"]),
                    "candidate_dataset": str(validation["candidate_dataset"]),
                    "archive_dataset": str(validation["archive_dataset"]),
                },
            )
        )

    decision = "APPLIED_ALIAS_SWITCH" if mutation_performed else ("READY_FOR_VEDTAK_APPLY" if not failures else "NOT_READY")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_foundation_activation_apply_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "apply_requested": apply,
        "dry_run_requested": dry_run_requested,
        "vedtak": vedtak or None,
        "mutation_performed": mutation_performed,
        "training_allowed": False,
        "plan_json": str(plan_json),
        "active_dataset": str(validation["active_dataset"]),
        "candidate_dataset": str(validation["candidate_dataset"]),
        "archive_dataset": str(validation["archive_dataset"]),
        "post_apply_commands": _post_apply_commands(validation["active"]),
        "checks": validation["checks"],
        "failures": failures,
        "next_required_action": (
            "rerun feature/target/specialist audits into canonical active paths, materialize canonical smoke dataset, then rerun train-readiness"
            if mutation_performed
            else (
                f"rerun with --apply --vedtak {REQUIRED_VEDTAK_PREFIX}<id> to perform alias switch"
                if not failures
                else "fix activation apply failures, then rerun dry-run"
            )
        ),
    }
    json_path = out_dir / f"ENTRY_FOUNDATION_ACTIVATION_APPLY_{timestamp}.json"
    latest_json = out_dir / "ENTRY_FOUNDATION_ACTIVATION_APPLY_latest.json"
    report["json_path"] = str(json_path)
    report["latest_json_path"] = str(latest_json)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "mutation_performed": report["mutation_performed"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "next_required_action": report["next_required_action"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-json", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--dry-run", action="store_true", help="Accepted for clarity; dry-run is the default unless --apply is set.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--vedtak", default="")
    ap.add_argument("--archive-suffix", default="STALE_PRE_DIRECTIONAL_SMC_20260629")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    report = run(build_parser().parse_args())
    if report["decision"] == "NOT_READY":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
