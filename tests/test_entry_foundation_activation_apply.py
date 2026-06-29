import argparse
import json
import os
from pathlib import Path

from gx1.scripts.apply_entry_foundation_activation_v1 import REQUIRED_VEDTAK_PREFIX, run


def _write_plan(tmp_path: Path) -> tuple[Path, Path, Path]:
    active = tmp_path / "active_dataset"
    candidate = tmp_path / "candidate_dataset"
    active.mkdir()
    candidate.mkdir()
    (active / "old.txt").write_text("old", encoding="utf-8")
    (candidate / "new.txt").write_text("new", encoding="utf-8")
    plan = {
        "schema_version": "entry_foundation_activation_plan_v1",
        "decision": "READY_FOR_VEDTAK_ACTIVATION",
        "activation_allowed_without_vedtak": False,
        "activation_steps": [
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
        ],
        "checks": [
            {
                "name": "source pointer contract present: gx1/scripts/apply_entry_foundation_activation_v1.py",
                "ok": True,
            },
            {
                "name": "source pointer contract present: gx1/scripts/run_entry_foundation_activation_post_apply_v1.py",
                "ok": True,
            },
            {
                "name": "source pointer contract present: scripts/entry_next_edge_control.sh",
                "ok": True,
            },
            {
                "name": "adoption report has expected schema",
                "ok": True,
            },
            {
                "name": "adoption report has zero failures",
                "ok": True,
            },
            {
                "name": "adoption candidate gates are all PASS",
                "ok": True,
            },
            {
                "name": "adoption artifact fingerprints match current files",
                "ok": True,
            },
        ],
        "adoption_contract": {
            "required_gates": [
                "candidate_dataset",
                "feature_audit",
                "target_audit",
                "specialist_audit",
                "smoke_dataset",
                "artifact_fingerprints",
            ],
            "gate_summary": {},
            "artifact_fingerprints": {},
        },
        "active_paths": {
            "foundation_dataset_dir": str(active),
            "foundation_smoke_dataset_dir": str(tmp_path / "active_smoke"),
            "feature_audit": str(tmp_path / "active_feature.json"),
            "target_audit": str(tmp_path / "active_target.json"),
            "specialist_audit": str(tmp_path / "active_specialist.json"),
        },
        "candidate_paths": {
            "candidate_dataset_dir": str(candidate),
            "candidate_smoke_dataset_dir": str(tmp_path / "candidate_smoke"),
            "feature_audit": str(tmp_path / "feature.json"),
            "target_audit": str(tmp_path / "target.json"),
            "specialist_audit": str(tmp_path / "specialist.json"),
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, active, candidate


def _args(tmp_path: Path, plan_path: Path, *, apply: bool, vedtak: str = "", dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        plan_json=str(plan_path),
        out_dir=str(tmp_path / "reports"),
        dry_run=dry_run,
        apply=apply,
        vedtak=vedtak,
        archive_suffix="ARCHIVED_BY_TEST",
        quiet=True,
    )


def test_activation_apply_dry_run_is_side_effect_free(tmp_path: Path) -> None:
    plan_path, active, candidate = _write_plan(tmp_path)

    report = run(_args(tmp_path, plan_path, apply=False))

    assert report["decision"] == "READY_FOR_VEDTAK_APPLY"
    assert report["mutation_performed"] is False
    assert active.is_dir()
    assert not active.is_symlink()
    assert (active / "old.txt").exists()
    assert candidate.is_dir()
    assert Path(report["archive_dataset"]).exists() is False
    command_names = {row["name"] for row in report["post_apply_commands"]}
    assert command_names == {
        "refresh_canonical_feature_audit",
        "refresh_canonical_target_audit",
        "refresh_canonical_specialist_audit",
        "refresh_canonical_smoke_dataset",
        "verify_active_foundation_state",
        "verify_train_readiness",
    }
    feature_cmd = next(row for row in report["post_apply_commands"] if row["name"] == "refresh_canonical_feature_audit")
    assert "--dataset-dir" in feature_cmd["argv"]
    assert str(active) in feature_cmd["argv"]
    smoke_cmd = next(row for row in report["post_apply_commands"] if row["name"] == "refresh_canonical_smoke_dataset")
    assert "--source-dir" in smoke_cmd["argv"]
    assert str(active) in smoke_cmd["argv"]
    verify_cmd = next(row for row in report["post_apply_commands"] if row["name"] == "verify_active_foundation_state")
    assert verify_cmd["argv"] == ["scripts/entry_next_edge_control.sh", "verify", "--quiet"]


def test_activation_apply_requires_vedtak(tmp_path: Path) -> None:
    plan_path, active, _candidate = _write_plan(tmp_path)

    report = run(_args(tmp_path, plan_path, apply=True, vedtak=""))

    assert report["decision"] == "NOT_READY"
    assert report["mutation_performed"] is False
    assert active.is_dir()
    assert any(check["name"] == "apply requires explicit activation vedtak" for check in report["failures"])


def test_activation_apply_rejects_stale_plan_without_active_verify(tmp_path: Path) -> None:
    plan_path, active, _candidate = _write_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["activation_steps"] = [
        {
            "step": "rerun train-readiness and keep smoke/candidate/replay/IQL gates closed until they pass",
            "mutates": False,
            "command": "scripts/entry_next_edge_control.sh train-readiness --quiet --no-fail-on-not-ready",
        }
    ]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = run(_args(tmp_path, plan_path, apply=False))

    assert report["decision"] == "NOT_READY"
    assert report["mutation_performed"] is False
    assert active.is_dir()
    assert any(
        check["name"] == "activation plan includes active verify before train-readiness"
        for check in report["failures"]
    )


def test_activation_apply_rejects_stale_plan_without_control_source_checks(tmp_path: Path) -> None:
    plan_path, active, _candidate = _write_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["checks"] = []
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = run(_args(tmp_path, plan_path, apply=False))

    assert report["decision"] == "NOT_READY"
    assert report["mutation_performed"] is False
    assert active.is_dir()
    assert any(
        check["name"] == "activation plan records current apply/post-apply/control source checks"
        for check in report["failures"]
    )


def test_activation_apply_rejects_stale_plan_without_adoption_artifact_contract(tmp_path: Path) -> None:
    plan_path, active, _candidate = _write_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["checks"] = [
        check
        for check in plan["checks"]
        if check["name"]
        not in {
            "adoption report has expected schema",
            "adoption report has zero failures",
            "adoption candidate gates are all PASS",
            "adoption artifact fingerprints match current files",
        }
    ]
    plan.pop("adoption_contract", None)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = run(_args(tmp_path, plan_path, apply=False))

    assert report["decision"] == "NOT_READY"
    assert report["mutation_performed"] is False
    assert active.is_dir()
    assert any(
        check["name"] == "activation plan records current adoption artifact contract"
        for check in report["failures"]
    )


def test_activation_apply_rejects_dry_run_and_apply_together(tmp_path: Path) -> None:
    plan_path, active, _candidate = _write_plan(tmp_path)

    report = run(
        _args(
            tmp_path,
            plan_path,
            dry_run=True,
            apply=True,
            vedtak=f"{REQUIRED_VEDTAK_PREFIX}UNIT_TEST",
        )
    )

    assert report["decision"] == "NOT_READY"
    assert report["mutation_performed"] is False
    assert active.is_dir()
    assert any(check["name"] == "dry-run and apply flags are mutually exclusive" for check in report["failures"])


def test_activation_apply_archives_active_and_symlinks_candidate_with_vedtak(tmp_path: Path) -> None:
    plan_path, active, candidate = _write_plan(tmp_path)

    report = run(
        _args(
            tmp_path,
            plan_path,
            apply=True,
            vedtak=f"{REQUIRED_VEDTAK_PREFIX}UNIT_TEST",
        )
    )

    assert report["decision"] == "APPLIED_ALIAS_SWITCH"
    assert report["mutation_performed"] is True
    archive = Path(report["archive_dataset"])
    assert archive.is_dir()
    assert (archive / "old.txt").read_text(encoding="utf-8") == "old"
    assert active.is_symlink()
    assert active.resolve() == candidate.resolve()
    assert (active / "new.txt").read_text(encoding="utf-8") == "new"
    assert report["training_allowed"] is False


def test_activation_apply_rolls_back_if_symlink_creation_fails(tmp_path: Path, monkeypatch) -> None:
    plan_path, active, _candidate = _write_plan(tmp_path)

    def fail_symlink(*_args, **_kwargs):
        raise OSError("simulated symlink failure")

    monkeypatch.setattr(os, "symlink", fail_symlink)
    report = run(
        _args(
            tmp_path,
            plan_path,
            apply=True,
            vedtak=f"{REQUIRED_VEDTAK_PREFIX}UNIT_TEST",
        )
    )

    assert report["decision"] == "NOT_READY"
    assert report["mutation_performed"] is False
    assert active.is_dir()
    assert not active.is_symlink()
    assert (active / "old.txt").read_text(encoding="utf-8") == "old"
    assert not Path(report["archive_dataset"]).exists()
    assert any(check["name"] == "activation alias switch completed" for check in report["failures"])
