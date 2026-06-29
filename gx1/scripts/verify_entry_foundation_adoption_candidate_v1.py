#!/usr/bin/env python3
"""Verify a staged Entry foundation dataset before making it active.

This is an adoption proof, not an activation step. It does not change active
latest audit paths, start training, promote a bundle, or touch shadow/live.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT
from gx1.scripts.verify_entry_training_readiness_v1 import REQUIRED_SPECIALISTS


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_foundation_adoption_candidate_20260629_v1"
DEFAULT_EXPECTED_SMOKE_ROWS = {"train": 4095, "val": 1536, "test": 1536}


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


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": None, "mtime_ns": None, "sha256": None}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _split_file(dataset_dir: Path, split: str, suffix: str) -> Path | None:
    matches = sorted(dataset_dir.glob(f"*_{split}{suffix}"))
    return matches[0] if len(matches) == 1 else None


def _balanced_label_counts(counts: dict[str, Any]) -> bool:
    if set(str(k) for k in counts) != {"0", "1", "2"}:
        return False
    values = [int(counts[str(k)]) for k in ("0", "1", "2")]
    return min(values) > 0 and len(set(values)) == 1


def _dataset_checks(dataset_dir: Path) -> list[dict[str, Any]]:
    checks = [_check("candidate dataset dir exists", dataset_dir.exists() and dataset_dir.is_dir(), {"dataset_dir": str(dataset_dir)})]
    for split in ("train", "val", "test"):
        parquet = _split_file(dataset_dir, split, ".parquet")
        manifest = _split_file(dataset_dir, split, ".manifest.json")
        checks.append(
            _check(
                f"{split} split parquet exists exactly once",
                parquet is not None and parquet.exists(),
                {"matches": [str(p) for p in sorted(dataset_dir.glob(f'*_{split}.parquet'))]},
            )
        )
        checks.append(
            _check(
                f"{split} split manifest exists exactly once",
                manifest is not None and manifest.exists(),
                {"matches": [str(p) for p in sorted(dataset_dir.glob(f'*_{split}.manifest.json'))]},
            )
        )
        if manifest is None or not manifest.exists():
            continue
        data = _read_json(manifest)
        signal_bridge = ((data.get("extra") or {}).get("signal_bridge") or {})
        extension = signal_bridge.get("seq_structure_extension_v1") or {}
        fields = [str(x) for x in signal_bridge.get("fields", [])]
        checks.extend(
            [
                _check(f"{split} signal fields are seq146", len(fields) == 146, {"field_count": len(fields)}),
                _check(
                    f"{split} seq/snap input dims are 146",
                    int(signal_bridge.get("seq_input_dim") or 0) == 146
                    and int(signal_bridge.get("snap_input_dim") or 0) == 146,
                    {
                        "seq_input_dim": signal_bridge.get("seq_input_dim"),
                        "snap_input_dim": signal_bridge.get("snap_input_dim"),
                    },
                ),
                _check(
                    f"{split} seq structure extension dim is 105",
                    int(signal_bridge.get("seq_structure_extension_dim") or 0) == 105,
                    {"seq_structure_extension_dim": signal_bridge.get("seq_structure_extension_dim")},
                ),
                _check(
                    f"{split} foundation structure version matches code",
                    str(extension.get("foundation_structure_feature_version")) == FOUNDATION_STRUCTURE_FEATURE_VERSION,
                    {
                        "emitted": extension.get("foundation_structure_feature_version"),
                        "code": FOUNDATION_STRUCTURE_FEATURE_VERSION,
                    },
                ),
                _check(
                    f"{split} foundation structure feature count matches code",
                    int(extension.get("foundation_structure_feature_count") or 0)
                    == len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
                    {
                        "emitted": extension.get("foundation_structure_feature_count"),
                        "expected": len(FOUNDATION_STRUCTURE_FEATURE_NAMES),
                    },
                ),
                _check(
                    f"{split} all foundation structure features selected",
                    bool(extension.get("foundation_structure_all_required_selected"))
                    and int(extension.get("foundation_structure_missing_feature_count") or 0) == 0,
                    {
                        "all_required": extension.get("foundation_structure_all_required_selected"),
                        "missing_count": extension.get("foundation_structure_missing_feature_count"),
                    },
                ),
            ]
        )
    return checks


def _feature_audit_checks(report: dict[str, Any], audit_path: Path, dataset_dir: Path) -> list[dict[str, Any]]:
    return [
        _check("feature audit artifact exists", audit_path.exists(), {"path": str(audit_path)}),
        _check("feature audit PASS", str(report.get("decision")) == "PASS", {"decision": report.get("decision")}),
        _check("feature audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check(
            "feature audit points at candidate dataset",
            str(report.get("dataset_dir")) == str(dataset_dir),
            {"dataset_dir": report.get("dataset_dir"), "candidate_dataset_dir": str(dataset_dir)},
        ),
        _check(
            "feature audit foundation structure version matches code",
            str(report.get("foundation_structure_feature_version")) == FOUNDATION_STRUCTURE_FEATURE_VERSION,
            {
                "audit_foundation_structure_feature_version": report.get("foundation_structure_feature_version"),
                "code_foundation_structure_feature_version": FOUNDATION_STRUCTURE_FEATURE_VERSION,
            },
        ),
        _check(
            "feature audit preserved all required foundation selection",
            int(report.get("foundation_missing_from_manifest_count") or 0) == 0
            and bool(report.get("manifest_foundation_all_required_selected")),
            {
                "foundation_missing_from_manifest_count": report.get("foundation_missing_from_manifest_count"),
                "manifest_foundation_all_required_selected": report.get("manifest_foundation_all_required_selected"),
            },
        ),
        _check(
            "feature audit objective and source liveness all live",
            bool(report.get("foundation_objective_coverage_all_present"))
            and bool(report.get("foundation_objective_liveness_all_live"))
            and bool(report.get("foundation_source_field_liveness_all_live")),
            {
                "foundation_objective_coverage_all_present": report.get("foundation_objective_coverage_all_present"),
                "foundation_objective_liveness_all_live": report.get("foundation_objective_liveness_all_live"),
                "foundation_source_field_liveness_all_live": report.get("foundation_source_field_liveness_all_live"),
            },
        ),
    ]


def _target_audit_checks(report: dict[str, Any], audit_path: Path, dataset_dir: Path) -> list[dict[str, Any]]:
    return [
        _check("target audit artifact exists", audit_path.exists(), {"path": str(audit_path)}),
        _check("target audit PASS", str(report.get("decision")) == "PASS", {"decision": report.get("decision")}),
        _check("target audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check(
            "target audit points at candidate dataset",
            str(report.get("dataset_dir")) == str(dataset_dir),
            {"dataset_dir": report.get("dataset_dir"), "candidate_dataset_dir": str(dataset_dir)},
        ),
        _check("target audit has head contract", isinstance(report.get("target_head_contract"), dict)),
    ]


def _specialist_audit_checks(report: dict[str, Any], audit_path: Path, dataset_dir: Path) -> list[dict[str, Any]]:
    required = {str(x) for x in report.get("required_training_specialists", []) if str(x)}
    return [
        _check("specialist audit artifact exists", audit_path.exists(), {"path": str(audit_path)}),
        _check("specialist audit PASS", str(report.get("decision")) == "PASS", {"decision": report.get("decision")}),
        _check("specialist audit has zero failures", not report.get("failures"), {"failures": report.get("failures")}),
        _check(
            "specialist audit points at candidate dataset",
            str(report.get("dataset_dir")) == str(dataset_dir),
            {"dataset_dir": report.get("dataset_dir"), "candidate_dataset_dir": str(dataset_dir)},
        ),
        _check("specialist signal dim is 146", int(report.get("signal_field_count") or 0) == 146),
        _check("specialist selected extension count is 105", int(report.get("selected_feature_count") or 0) == 105),
        _check(
            "specialist required training set is exact",
            required == set(REQUIRED_SPECIALISTS),
            {"expected": list(REQUIRED_SPECIALISTS), "actual": sorted(required)},
        ),
        _check(
            "specialist feature routing and liveness are live",
            bool(report.get("specialist_input_liveness_all_live"))
            and bool(report.get("foundation_objective_routing_all_present_and_expected")),
            {
                "specialist_input_liveness_all_live": report.get("specialist_input_liveness_all_live"),
                "foundation_objective_routing_all_present_and_expected": report.get(
                    "foundation_objective_routing_all_present_and_expected"
                ),
            },
        ),
    ]


def _smoke_dataset_checks(
    smoke_dir: Path,
    dataset_dir: Path,
    audit_paths: dict[str, Path],
    *,
    expected_rows: dict[str, int],
) -> list[dict[str, Any]]:
    manifest_path = smoke_dir / "SMOKE_DATASET_MANIFEST.json"
    checks = [
        _check("candidate smoke dataset manifest exists", manifest_path.exists(), {"path": str(manifest_path)}),
    ]
    if not manifest_path.exists():
        return checks
    report = _read_json(manifest_path)
    splits = report.get("splits") if isinstance(report.get("splits"), dict) else {}
    provenance = report.get("audit_provenance") if isinstance(report.get("audit_provenance"), dict) else {}
    provenance_artifacts = provenance.get("artifacts") if isinstance(provenance.get("artifacts"), dict) else {}

    def artifact_matches(name: str) -> bool:
        row = provenance_artifacts.get(name) if isinstance(provenance_artifacts.get(name), dict) else {}
        path = audit_paths[name]
        return (
            bool(row.get("exists"))
            and str(row.get("path")) == str(path)
            and isinstance(row.get("sha256"), str)
            and row.get("sha256") == _sha256_file(path)
        )

    def source_manifest_hash_matches(split: str) -> bool:
        row = splits.get(split) if isinstance(splits.get(split), dict) else {}
        path = Path(str(row.get("source_manifest") or ""))
        return path.exists() and row.get("source_manifest_sha256") == _sha256_file(path)

    def output_hashes_match(split: str) -> bool:
        row = splits.get(split) if isinstance(splits.get(split), dict) else {}
        parquet = Path(str(row.get("out_path") or ""))
        manifest = Path(str(row.get("out_manifest") or ""))
        return (
            parquet.exists()
            and manifest.exists()
            and row.get("out_parquet_sha256") == _sha256_file(parquet)
            and row.get("out_manifest_sha256") == _sha256_file(manifest)
        )

    checks.extend(
        [
            _check(
                "candidate smoke dataset schema is foundation smoke v1",
                report.get("schema_version") == "entry_foundation_seq146_smoke_dataset_v1",
                {"schema_version": report.get("schema_version")},
            ),
            _check(
                "candidate smoke dataset points at candidate source",
                str(report.get("source_dir")) == str(dataset_dir),
                {"source_dir": report.get("source_dir"), "candidate_dataset_dir": str(dataset_dir)},
            ),
            _check(
                "candidate smoke row counts match readiness contract",
                all(int((splits.get(split) or {}).get("rows") or 0) == rows for split, rows in expected_rows.items()),
                {
                    "expected_rows": expected_rows,
                    "actual_rows": {split: (splits.get(split) or {}).get("rows") for split in expected_rows},
                },
            ),
            _check(
                "candidate smoke labels are class-balanced",
                all(_balanced_label_counts((splits.get(split) or {}).get("label_counts") or {}) for split in expected_rows),
                {"label_counts": {split: (splits.get(split) or {}).get("label_counts") for split in expected_rows}},
            ),
            _check(
                "candidate smoke records exact audit artifact provenance",
                str(provenance.get("schema_version")) == "entry_foundation_smoke_dataset_audit_provenance_v1"
                and all(artifact_matches(name) for name in audit_paths),
                {"audit_paths": {name: str(path) for name, path in audit_paths.items()}, "provenance": provenance_artifacts},
            ),
            _check(
                "candidate smoke records source manifest hashes",
                all(source_manifest_hash_matches(split) for split in expected_rows),
                {"splits": {split: (splits.get(split) or {}) for split in expected_rows}},
            ),
            _check(
                "candidate smoke output hashes match files",
                all(output_hashes_match(split) for split in expected_rows),
                {"splits": {split: (splits.get(split) or {}) for split in expected_rows}},
            ),
        ]
    )
    return checks


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Foundation Adoption Candidate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Candidate ready for activation: `{report['candidate_ready_for_activation']}`",
        f"- Training allowed: `{report['training_allowed']}`",
        f"- Next required action: `{report['next_required_action']}`",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        lines.append(f"- `{gate['name']}`: {gate['decision']} ({gate['passed']}/{gate['total']} checks)")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        lines.extend([f"- {row['gate']}: {row['check']}" for row in report["failures"]])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    feature_audit = Path(args.feature_audit_json).expanduser().resolve()
    target_audit = Path(args.target_audit_json).expanduser().resolve()
    specialist_audit = Path(args.specialist_audit_json).expanduser().resolve()
    smoke_dir = Path(args.smoke_dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_smoke_rows = {
        "train": int(args.expected_smoke_train_rows),
        "val": int(args.expected_smoke_val_rows),
        "test": int(args.expected_smoke_test_rows),
    }
    audit_paths = {
        "feature_audit": feature_audit,
        "target_audit": target_audit,
        "specialist_audit": specialist_audit,
    }
    artifacts = {
        "candidate_dataset_dir": str(dataset_dir),
        "candidate_smoke_dataset_dir": str(smoke_dir),
        **{name: str(path) for name, path in audit_paths.items()},
    }
    artifact_fingerprints = {
        name: _fingerprint(path)
        for name, path in {
            "feature_audit": feature_audit,
            "target_audit": target_audit,
            "specialist_audit": specialist_audit,
            "smoke_dataset_manifest": smoke_dir / "SMOKE_DATASET_MANIFEST.json",
        }.items()
    }
    feature_report = _read_json(feature_audit)
    target_report = _read_json(target_audit)
    specialist_report = _read_json(specialist_audit)
    gate_checks = {
        "candidate_dataset": _dataset_checks(dataset_dir),
        "feature_audit": _feature_audit_checks(feature_report, feature_audit, dataset_dir),
        "target_audit": _target_audit_checks(target_report, target_audit, dataset_dir),
        "specialist_audit": _specialist_audit_checks(specialist_report, specialist_audit, dataset_dir),
        "smoke_dataset": _smoke_dataset_checks(
            smoke_dir,
            dataset_dir,
            audit_paths,
            expected_rows=expected_smoke_rows,
        ),
        "artifact_fingerprints": [
            _check(
                "all adoption artifacts have sha256 fingerprints",
                all(
                    bool(row.get("exists"))
                    and isinstance(row.get("sha256"), str)
                    and len(str(row.get("sha256"))) == 64
                    for row in artifact_fingerprints.values()
                ),
                {"artifact_fingerprints": artifact_fingerprints},
            )
        ],
    }
    gates = []
    for name, checks in gate_checks.items():
        passed = sum(1 for check in checks if check["ok"])
        gates.append(
            {
                "name": name,
                "decision": "PASS" if passed == len(checks) else "FAIL",
                "passed": int(passed),
                "total": int(len(checks)),
                "checks": checks,
            }
        )
    failures = [
        {"gate": gate["name"], "check": check["name"], "details": check.get("details") or {}}
        for gate in gates
        for check in gate["checks"]
        if not check["ok"]
    ]
    decision = "PASS" if not failures else "NOT_READY"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_foundation_adoption_candidate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "candidate_ready_for_activation": decision == "PASS",
        "training_allowed": False,
        "activation_allowed_without_vedtak": False,
        "next_required_action": (
            "explicit vedtak to switch active foundation dataset/audit paths, "
            "then rerun scripts/entry_next_edge_control.sh train-readiness"
            if decision == "PASS"
            else "fix failing adoption-candidate gates, then rerun this verifier"
        ),
        "artifacts": artifacts,
        "artifact_fingerprints": artifact_fingerprints,
        "expected_smoke_rows": expected_smoke_rows,
        "gates": gates,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_FOUNDATION_ADOPTION_CANDIDATE_{timestamp}.json"
    md_path = out_dir / f"ENTRY_FOUNDATION_ADOPTION_CANDIDATE_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_FOUNDATION_ADOPTION_CANDIDATE_latest.json"
    latest_md = out_dir / "ENTRY_FOUNDATION_ADOPTION_CANDIDATE_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
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
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--feature-audit-json", required=True)
    ap.add_argument("--target-audit-json", required=True)
    ap.add_argument("--specialist-audit-json", required=True)
    ap.add_argument("--smoke-dataset-dir", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--expected-smoke-train-rows", type=int, default=DEFAULT_EXPECTED_SMOKE_ROWS["train"])
    ap.add_argument("--expected-smoke-val-rows", type=int, default=DEFAULT_EXPECTED_SMOKE_ROWS["val"])
    ap.add_argument("--expected-smoke-test-rows", type=int, default=DEFAULT_EXPECTED_SMOKE_ROWS["test"])
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    if args.fail_on_not_ready and report["decision"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
