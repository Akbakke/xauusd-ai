#!/usr/bin/env python3
"""Build a reviewed activation plan for a green Entry foundation adoption candidate.

This script is report-only. It does not move datasets, rewrite latest audit
artifacts, edit source files, start training, or touch shadow/live paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.verify_entry_foundation_state_v1 import (
    FEATURE_AUDIT_LATEST,
    FOUNDATION_DATASET_DIR,
    FOUNDATION_SMOKE_DATASET_DIR,
    REPO,
    REPORTS_ROOT,
    SPECIALIST_AUDIT_LATEST,
    TARGET_AUDIT_LATEST,
)


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_foundation_activation_plan_20260629_v1"
DEFAULT_ADOPTION_ROOT = REPORTS_ROOT / "entry_foundation_adoption_candidate_20260629_v1"
REQUIRED_ADOPTION_GATES = (
    "candidate_dataset",
    "feature_audit",
    "target_audit",
    "specialist_audit",
    "smoke_dataset",
    "artifact_fingerprints",
)


SOURCE_POINTER_CONTRACT = {
    "gx1/scripts/verify_entry_foundation_state_v1.py": (
        "FOUNDATION_DATASET_DIR",
        "FOUNDATION_SMOKE_DATASET_DIR",
        "FEATURE_AUDIT_LATEST",
        "TARGET_AUDIT_LATEST",
        "SPECIALIST_AUDIT_LATEST",
    ),
    "gx1/scripts/audit_entry_foundation_features_v1.py": (
        "DEFAULT_FOUNDATION_DATASET_DIR",
        "DEFAULT_OUT_DIR",
    ),
    "gx1/scripts/audit_entry_foundation_targets_v1.py": (
        "DEFAULT_FOUNDATION_DATASET_DIR",
        "DEFAULT_OUT_DIR",
    ),
    "gx1/scripts/audit_entry_specialist_feature_groups_v1.py": (
        "FOUNDATION_DATASET_DIR",
        "DEFAULT_OUT_DIR",
    ),
    "gx1/scripts/apply_entry_foundation_activation_v1.py": (
        "REQUIRED_VEDTAK_PREFIX",
        "_apply_alias",
        "_post_apply_commands",
        "verify_active_foundation_state",
    ),
    "gx1/scripts/run_entry_foundation_activation_post_apply_v1.py": (
        "EXPECTED_ORDER",
        "FORBIDDEN_TOKENS",
        "verify_active_foundation_state",
        "POST_APPLY_REFRESH_COMPLETED",
    ),
    "scripts/entry_next_edge_control.sh": (
        "foundation-activation-apply)",
        "foundation-activation-post-apply)",
        "verify --quiet",
        "readiness-report)",
    ),
    "scripts/run_entry_foundation_seq146_smoke_train.sh": (
        "SOURCE_DATASET=",
        "SMOKE_DATASET=",
        "SPECIALIST_AUDIT=",
    ),
    "scripts/run_entry_foundation_seq146_candidate_train.sh": (
        "FOUNDATION_DATASET=",
        "SPECIALIST_AUDIT=",
    ),
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_adoption_report(root: Path) -> Path:
    candidates = (
        sorted(
            root.glob("*/ENTRY_FOUNDATION_ADOPTION_CANDIDATE_latest.json"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        if root.exists()
        else []
    )
    if not candidates:
        raise RuntimeError(f"no adoption candidate latest report under {root}")
    return candidates[0]


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _source_pointer_checks(repo_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rel, required_tokens in SOURCE_POINTER_CONTRACT.items():
        path = repo_root / rel
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        missing = [token for token in required_tokens if token not in text]
        checks.append(
            _check(
                f"source pointer contract present: {rel}",
                path.exists() and not missing,
                {
                    "path": str(path),
                    "required_tokens": list(required_tokens),
                    "missing_tokens": missing,
                },
            )
        )
    return checks


def _adoption_gate_summary(adoption: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gates = adoption.get("gates") if isinstance(adoption.get("gates"), list) else []
    summary: dict[str, dict[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        name = str(gate.get("name") or "")
        if not name:
            continue
        summary[name] = {
            "decision": gate.get("decision"),
            "passed": gate.get("passed"),
            "total": gate.get("total"),
        }
    return summary


def _adoption_artifact_fingerprint_report(adoption: dict[str, Any], candidate: dict[str, str]) -> dict[str, dict[str, Any]]:
    fingerprints = adoption.get("artifact_fingerprints") if isinstance(adoption.get("artifact_fingerprints"), dict) else {}
    expected_paths = {
        "feature_audit": candidate.get("feature_audit") or "",
        "target_audit": candidate.get("target_audit") or "",
        "specialist_audit": candidate.get("specialist_audit") or "",
        "smoke_dataset_manifest": str(Path(candidate.get("candidate_smoke_dataset_dir") or "") / "SMOKE_DATASET_MANIFEST.json")
        if candidate.get("candidate_smoke_dataset_dir")
        else "",
    }
    report: dict[str, dict[str, Any]] = {}
    for name, expected_path in expected_paths.items():
        row = fingerprints.get(name) if isinstance(fingerprints.get(name), dict) else {}
        path = Path(str(row.get("path") or expected_path or ""))
        current_sha = _sha256_file(path)
        report[name] = {
            "reported_path": row.get("path"),
            "expected_path": expected_path,
            "exists": path.exists() and path.is_file(),
            "reported_exists": row.get("exists"),
            "reported_sha256": row.get("sha256"),
            "current_sha256": current_sha,
            "path_matches": bool(expected_path) and str(row.get("path")) == expected_path,
            "sha256_matches": isinstance(row.get("sha256"), str) and row.get("sha256") == current_sha,
        }
    return report


def _adoption_contract_checks(adoption: dict[str, Any], candidate: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gate_summary = _adoption_gate_summary(adoption)
    fingerprint_report = _adoption_artifact_fingerprint_report(adoption, candidate)
    checks = [
        _check(
            "adoption report has expected schema",
            adoption.get("schema_version") == "entry_foundation_adoption_candidate_v1",
            {"schema_version": adoption.get("schema_version")},
        ),
        _check(
            "adoption report has zero failures",
            not adoption.get("failures"),
            {"failures": adoption.get("failures")},
        ),
        _check(
            "adoption candidate gates are all PASS",
            all(
                gate_summary.get(name, {}).get("decision") == "PASS"
                and int(gate_summary.get(name, {}).get("passed") or -1) == int(gate_summary.get(name, {}).get("total") or -2)
                for name in REQUIRED_ADOPTION_GATES
            ),
            {"required_gates": list(REQUIRED_ADOPTION_GATES), "gate_summary": gate_summary},
        ),
        _check(
            "adoption artifact fingerprints match current files",
            all(
                row["exists"]
                and row["reported_exists"] is True
                and row["path_matches"]
                and row["sha256_matches"]
                for row in fingerprint_report.values()
            ),
            {"fingerprints": fingerprint_report},
        ),
    ]
    return checks, {
        "required_gates": list(REQUIRED_ADOPTION_GATES),
        "gate_summary": gate_summary,
        "artifact_fingerprints": fingerprint_report,
    }


def _activation_steps(active: dict[str, str], candidate: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "step": "require explicit activation vedtak",
            "mutates": False,
            "command_shape": (
                "scripts/entry_next_edge_control.sh foundation-activation-plan --adoption-report <report> "
                "# report-only; a separate apply step must require --vedtak <id>"
            ),
        },
        {
            "step": "archive stale canonical active dataset before aliasing",
            "mutates": True,
            "source": active["foundation_dataset_dir"],
            "suggested_archive_name": f"{active['foundation_dataset_dir']}_STALE_PRE_DIRECTIONAL_SMC_20260629",
        },
        {
            "step": "point canonical active dataset path at green candidate dataset",
            "mutates": True,
            "canonical_active_path": active["foundation_dataset_dir"],
            "candidate_dataset": candidate["candidate_dataset_dir"],
            "strategy": "symlink_or_reviewed_atomic_directory_switch",
        },
        {
            "step": "rerun feature, target, and specialist audits into canonical active report paths",
            "mutates": True,
            "feature_audit_out_dir": str(Path(active["feature_audit"]).parent),
            "target_audit_out_dir": str(Path(active["target_audit"]).parent),
            "specialist_audit_out_dir": str(Path(active["specialist_audit"]).parent),
            "must_produce": {
                "feature_audit": active["feature_audit"],
                "target_audit": active["target_audit"],
                "specialist_audit": active["specialist_audit"],
            },
        },
        {
            "step": "materialize canonical active smoke dataset from canonical active dataset and canonical audits",
            "mutates": True,
            "canonical_smoke_dataset_dir": active["foundation_smoke_dataset_dir"],
            "candidate_smoke_dataset_dir_for_reference": candidate["candidate_smoke_dataset_dir"],
        },
        {
            "step": "verify active foundation state after canonical refresh",
            "mutates": False,
            "command": "scripts/entry_next_edge_control.sh verify --quiet",
        },
        {
            "step": "rerun train-readiness and keep smoke/candidate/replay/IQL gates closed until they pass",
            "mutates": False,
            "command": "scripts/entry_next_edge_control.sh train-readiness --quiet --no-fail-on-not-ready",
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(getattr(args, "repo_root", REPO)).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    adoption_report_path = (
        Path(args.adoption_report).expanduser().resolve()
        if str(args.adoption_report or "").strip()
        else _latest_adoption_report(Path(args.adoption_root).expanduser().resolve())
    )
    adoption = _read_json(adoption_report_path)
    artifacts = adoption.get("artifacts") if isinstance(adoption.get("artifacts"), dict) else {}

    active = {
        "foundation_dataset_dir": str(Path(getattr(args, "active_dataset_dir", FOUNDATION_DATASET_DIR)).expanduser()),
        "foundation_smoke_dataset_dir": str(
            Path(getattr(args, "active_smoke_dataset_dir", FOUNDATION_SMOKE_DATASET_DIR)).expanduser()
        ),
        "feature_audit": str(Path(getattr(args, "active_feature_audit_json", FEATURE_AUDIT_LATEST)).expanduser()),
        "target_audit": str(Path(getattr(args, "active_target_audit_json", TARGET_AUDIT_LATEST)).expanduser()),
        "specialist_audit": str(Path(getattr(args, "active_specialist_audit_json", SPECIALIST_AUDIT_LATEST)).expanduser()),
    }
    candidate = {
        "candidate_dataset_dir": str(artifacts.get("candidate_dataset_dir") or ""),
        "candidate_smoke_dataset_dir": str(artifacts.get("candidate_smoke_dataset_dir") or ""),
        "feature_audit": str(artifacts.get("feature_audit") or ""),
        "target_audit": str(artifacts.get("target_audit") or ""),
        "specialist_audit": str(artifacts.get("specialist_audit") or ""),
    }
    candidate_paths = {
        name: Path(path)
        for name, path in candidate.items()
        if path
    }
    active_paths = {
        name: Path(path)
        for name, path in active.items()
        if path
    }

    checks = [
        _check("adoption report exists", adoption_report_path.exists(), {"path": str(adoption_report_path)}),
        _check("adoption report PASS", adoption.get("decision") == "PASS", {"decision": adoption.get("decision")}),
        _check(
            "adoption candidate ready for activation",
            adoption.get("candidate_ready_for_activation") is True,
            {"candidate_ready_for_activation": adoption.get("candidate_ready_for_activation")},
        ),
        _check(
            "adoption report does not allow activation without vedtak",
            adoption.get("activation_allowed_without_vedtak") is False,
            {"activation_allowed_without_vedtak": adoption.get("activation_allowed_without_vedtak")},
        ),
        _check(
            "candidate paths all exist",
            bool(candidate_paths) and all(path.exists() for path in candidate_paths.values()),
            {"candidate_paths": {name: str(path) for name, path in candidate_paths.items()}},
        ),
        _check(
            "active canonical paths are known",
            bool(active_paths) and all(str(path) for path in active_paths.values()),
            {"active_paths": {name: str(path) for name, path in active_paths.items()}},
        ),
        _check(
            "candidate differs from active canonical dataset path",
            candidate.get("candidate_dataset_dir") != active.get("foundation_dataset_dir"),
            {
                "candidate_dataset_dir": candidate.get("candidate_dataset_dir"),
                "active_foundation_dataset_dir": active.get("foundation_dataset_dir"),
            },
        ),
    ]
    adoption_contract_checks, adoption_contract = _adoption_contract_checks(adoption, candidate)
    checks.extend(adoption_contract_checks)
    checks.extend(_source_pointer_checks(repo_root))

    failures = [check for check in checks if not check["ok"]]
    decision = "READY_FOR_VEDTAK_ACTIVATION" if not failures else "NOT_READY"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_foundation_activation_plan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "activation_allowed_without_vedtak": False,
        "training_allowed": False,
        "recommended_strategy": "canonical_active_alias_then_canonical_audit_refresh",
        "adoption_report": str(adoption_report_path),
        "adoption_contract": adoption_contract,
        "active_paths": active,
        "candidate_paths": candidate,
        "activation_steps": _activation_steps(active, candidate),
        "checks": checks,
        "failures": failures,
        "next_required_action": (
            "explicit activation vedtak, then perform the reviewed canonical alias/audit-refresh steps"
            if decision == "READY_FOR_VEDTAK_ACTIVATION"
            else "fix activation-plan failures, then rerun this report"
        ),
    }
    json_path = out_dir / f"ENTRY_FOUNDATION_ACTIVATION_PLAN_{timestamp}.json"
    latest_json = out_dir / "ENTRY_FOUNDATION_ACTIVATION_PLAN_latest.json"
    report["json_path"] = str(json_path)
    report["latest_json_path"] = str(latest_json)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": report["failures"],
                    "json_path": str(json_path),
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
    ap.add_argument("--adoption-report", default="")
    ap.add_argument("--adoption-root", default=str(DEFAULT_ADOPTION_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    report = run(build_parser().parse_args())
    return 0 if report["decision"] == "READY_FOR_VEDTAK_ACTIVATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
