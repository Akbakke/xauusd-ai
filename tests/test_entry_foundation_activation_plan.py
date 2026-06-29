import argparse
import hashlib
import json
from pathlib import Path

from gx1.scripts.plan_entry_foundation_activation_v1 import SOURCE_POINTER_CONTRACT, run


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_source_contract(repo: Path) -> None:
    for rel, tokens in SOURCE_POINTER_CONTRACT.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def _write_artifacts(root: Path, *, decision: str = "PASS") -> tuple[Path, dict[str, Path]]:
    dataset = root / "candidate_dataset"
    smoke = root / "candidate_smoke"
    dataset.mkdir()
    smoke.mkdir()
    smoke_manifest = smoke / "SMOKE_DATASET_MANIFEST.json"
    smoke_manifest.write_text(json.dumps({"schema_version": "entry_foundation_seq146_smoke_dataset_v1"}), encoding="utf-8")
    feature = root / "feature.json"
    target = root / "target.json"
    specialist = root / "specialist.json"
    for path in (feature, target, specialist):
        path.write_text("{}", encoding="utf-8")
    adoption = root / "adoption.json"
    adoption.write_text(
        json.dumps(
            {
                "schema_version": "entry_foundation_adoption_candidate_v1",
                "decision": decision,
                "candidate_ready_for_activation": decision == "PASS",
                "activation_allowed_without_vedtak": False,
                "failures": [] if decision == "PASS" else [{"gate": "candidate_dataset", "check": "failed by test"}],
                "gates": [
                    {
                        "name": name,
                        "decision": "PASS" if decision == "PASS" else "FAIL",
                        "passed": 1 if decision == "PASS" else 0,
                        "total": 1,
                        "checks": [],
                    }
                    for name in (
                        "candidate_dataset",
                        "feature_audit",
                        "target_audit",
                        "specialist_audit",
                        "smoke_dataset",
                        "artifact_fingerprints",
                    )
                ],
                "artifacts": {
                    "candidate_dataset_dir": str(dataset),
                    "candidate_smoke_dataset_dir": str(smoke),
                    "feature_audit": str(feature),
                    "target_audit": str(target),
                    "specialist_audit": str(specialist),
                },
                "artifact_fingerprints": {
                    "feature_audit": {"path": str(feature), "exists": True, "sha256": _sha(feature)},
                    "target_audit": {"path": str(target), "exists": True, "sha256": _sha(target)},
                    "specialist_audit": {"path": str(specialist), "exists": True, "sha256": _sha(specialist)},
                    "smoke_dataset_manifest": {
                        "path": str(smoke_manifest),
                        "exists": True,
                        "sha256": _sha(smoke_manifest),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return adoption, {
        "dataset": dataset,
        "smoke": smoke,
        "feature": feature,
        "target": target,
        "specialist": specialist,
    }


def _args(tmp_path: Path, repo: Path, adoption: Path) -> argparse.Namespace:
    return argparse.Namespace(
        adoption_report=str(adoption),
        adoption_root=str(tmp_path / "unused_adoption_root"),
        out_dir=str(tmp_path / "reports"),
        quiet=True,
        repo_root=str(repo),
        active_dataset_dir=str(tmp_path / "active_dataset"),
        active_smoke_dataset_dir=str(tmp_path / "active_smoke"),
        active_feature_audit_json=str(tmp_path / "active_feature.json"),
        active_target_audit_json=str(tmp_path / "active_target.json"),
        active_specialist_audit_json=str(tmp_path / "active_specialist.json"),
    )


def test_entry_foundation_activation_plan_is_report_only_and_vedtak_gated(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_source_contract(repo)
    adoption, _artifacts = _write_artifacts(tmp_path)

    report = run(_args(tmp_path, repo, adoption))

    assert report["decision"] == "READY_FOR_VEDTAK_ACTIVATION"
    assert report["report_only"] is True
    assert report["activation_allowed_without_vedtak"] is False
    assert report["training_allowed"] is False
    assert report["recommended_strategy"] == "canonical_active_alias_then_canonical_audit_refresh"
    assert report["adoption_contract"]["gate_summary"]["candidate_dataset"]["decision"] == "PASS"
    assert report["adoption_contract"]["artifact_fingerprints"]["feature_audit"]["sha256_matches"] is True
    assert report["failures"] == []
    assert any(step["mutates"] is True for step in report["activation_steps"])
    step_commands = [str(step.get("command") or step.get("command_shape") or "") for step in report["activation_steps"]]
    assert "scripts/entry_next_edge_control.sh verify --quiet" in step_commands
    assert step_commands.index("scripts/entry_next_edge_control.sh verify --quiet") < step_commands.index(
        "scripts/entry_next_edge_control.sh train-readiness --quiet --no-fail-on-not-ready"
    )
    assert "explicit activation vedtak" in report["next_required_action"]
    assert Path(report["json_path"]).exists()
    persisted = json.loads(Path(report["latest_json_path"]).read_text(encoding="utf-8"))
    assert persisted["json_path"] == report["json_path"]
    check_names = {check["name"] for check in report["checks"]}
    assert "source pointer contract present: gx1/scripts/apply_entry_foundation_activation_v1.py" in check_names
    assert "source pointer contract present: gx1/scripts/run_entry_foundation_activation_post_apply_v1.py" in check_names
    assert "source pointer contract present: scripts/entry_next_edge_control.sh" in check_names
    assert "adoption candidate gates are all PASS" in check_names
    assert "adoption artifact fingerprints match current files" in check_names


def test_entry_foundation_activation_plan_fails_when_adoption_not_pass(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_source_contract(repo)
    adoption, _artifacts = _write_artifacts(tmp_path, decision="NOT_READY")

    report = run(_args(tmp_path, repo, adoption))

    assert report["decision"] == "NOT_READY"
    assert any(check["name"] == "adoption report PASS" for check in report["failures"])


def test_entry_foundation_activation_plan_fails_when_adoption_fingerprint_is_stale(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write_source_contract(repo)
    adoption, artifacts = _write_artifacts(tmp_path)
    artifacts["feature"].write_text('{"changed": true}', encoding="utf-8")

    report = run(_args(tmp_path, repo, adoption))

    assert report["decision"] == "NOT_READY"
    assert any(
        check["name"] == "adoption artifact fingerprints match current files"
        for check in report["failures"]
    )
