import argparse
import json
import subprocess
from pathlib import Path

import gx1.scripts.run_entry_foundation_activation_post_apply_v1 as post_apply
from gx1.scripts.apply_entry_foundation_activation_v1 import _post_apply_commands
from gx1.scripts.run_entry_foundation_activation_post_apply_v1 import REQUIRED_VEDTAK_PREFIX, run


def _write_activation_report(tmp_path: Path, *, mutation_performed: bool = False) -> Path:
    active = {
        "foundation_dataset_dir": str(tmp_path / "active_dataset"),
        "foundation_smoke_dataset_dir": str(tmp_path / "active_smoke"),
        "feature_audit": str(tmp_path / "reports" / "feature" / "ENTRY_FEATURE_FOUNDATION_AUDIT_latest.json"),
        "target_audit": str(tmp_path / "reports" / "target" / "ENTRY_TARGET_FOUNDATION_AUDIT_latest.json"),
        "specialist_audit": str(tmp_path / "reports" / "specialist" / "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"),
    }
    report = {
        "schema_version": "entry_foundation_activation_apply_v1",
        "decision": "APPLIED_ALIAS_SWITCH" if mutation_performed else "READY_FOR_VEDTAK_APPLY",
        "mutation_performed": mutation_performed,
        "training_allowed": False,
        "post_apply_commands": _post_apply_commands(active),
    }
    path = tmp_path / "activation_apply.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _args(tmp_path: Path, report: Path, *, apply: bool = False, vedtak: str = "", dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        activation_apply_json=str(report),
        out_dir=str(tmp_path / "post_apply_reports"),
        dry_run=dry_run,
        apply=apply,
        vedtak=vedtak,
        timeout_seconds=15,
        quiet=True,
    )


def test_post_apply_dry_run_waits_for_activation_apply(tmp_path: Path) -> None:
    activation = _write_activation_report(tmp_path, mutation_performed=False)

    report = run(_args(tmp_path, activation))

    assert report["decision"] == "WAITING_FOR_ACTIVATION_APPLY"
    assert report["activation_alias_mutation_performed"] is False
    assert report["post_apply_mutations_performed"] is False
    assert report["training_allowed"] is False
    assert report["command_results"] == []
    assert Path(report["latest_json_path"]).exists()


def test_post_apply_dry_run_ready_after_activation_apply(tmp_path: Path) -> None:
    activation = _write_activation_report(tmp_path, mutation_performed=True)

    report = run(_args(tmp_path, activation, dry_run=True))

    assert report["decision"] == "READY_FOR_POST_APPLY_REFRESH"
    assert report["activation_alias_mutation_performed"] is True
    assert report["post_apply_mutations_performed"] is False
    assert [row["name"] for row in report["commands"]] == [
        "refresh_canonical_feature_audit",
        "refresh_canonical_target_audit",
        "refresh_canonical_specialist_audit",
        "refresh_canonical_smoke_dataset",
        "verify_active_foundation_state",
        "verify_train_readiness",
    ]


def test_post_apply_requires_explicit_vedtak_for_apply(tmp_path: Path) -> None:
    activation = _write_activation_report(tmp_path, mutation_performed=True)

    report = run(_args(tmp_path, activation, apply=True))

    assert report["decision"] == "NOT_READY"
    assert report["post_apply_mutations_performed"] is False
    assert any(check["name"] == "apply requires explicit post-apply vedtak" for check in report["failures"])


def test_post_apply_rejects_forbidden_command(tmp_path: Path) -> None:
    activation = _write_activation_report(tmp_path, mutation_performed=True)
    data = json.loads(activation.read_text(encoding="utf-8"))
    data["post_apply_commands"][0]["argv"] = ["scripts/entry_next_edge_control.sh", "smoke-train"]
    activation.write_text(json.dumps(data), encoding="utf-8")

    report = run(_args(tmp_path, activation))

    assert report["decision"] == "NOT_READY"
    assert any(
        check["name"] == "post-apply command is safe: refresh_canonical_feature_audit"
        for check in report["failures"]
    )


def test_post_apply_runs_commands_with_vedtak_after_activation(tmp_path: Path, monkeypatch) -> None:
    activation = _write_activation_report(tmp_path, mutation_performed=True)
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(post_apply.subprocess, "run", fake_run)

    report = run(
        _args(
            tmp_path,
            activation,
            apply=True,
            vedtak=f"{REQUIRED_VEDTAK_PREFIX}UNIT_TEST",
        )
    )

    assert report["decision"] == "POST_APPLY_REFRESH_COMPLETED"
    assert report["post_apply_mutations_performed"] is True
    assert len(calls) == 6
    assert calls[-2] == [
        "scripts/entry_next_edge_control.sh",
        "verify",
        "--quiet",
    ]
    assert calls[-1] == [
        "scripts/entry_next_edge_control.sh",
        "train-readiness",
        "--quiet",
        "--no-fail-on-not-ready",
    ]
