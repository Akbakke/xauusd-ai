from __future__ import annotations

import pytest

from gx1.scripts import run_unified_exit_random_access_val_v1 as cli
from gx1.scripts.run_unified_exit_random_access_val_v1 import main


def test_val_cli_help_imports_without_loading_checkpoint(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "Strict weight-EMA full-cohort VAL executor" in output
    assert "--final-train-checkpoint-authority" in output
    assert "--rollout-progress-path" in output


def test_resumable_window_is_successful_guard_exit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "run",
        lambda **_kwargs: {
            "decision": "PAUSED_RESUMABLE",
            "pause_sha256": "a" * 64,
            "next_state_index": 7,
            "model_forward_count": 11,
        },
    )
    common = [
        "--launch-manifest",
        str(tmp_path / "launch.json"),
        "--final-train-checkpoint-authority",
        str(tmp_path / "authority.json"),
        "--final-train-checkpoint-authority-file-sha256",
        "b" * 64,
        "--checkpoint-pointer",
        str(tmp_path / "pointer.json"),
        "--progress-path",
        str(tmp_path / "campaign-progress.json"),
        "--rollout-progress-path",
        str(tmp_path / "rollout-progress.json"),
        "--result-path",
        str(tmp_path / "result.json"),
        "--device",
        "cpu",
        "--max-forwards-this-invocation",
        "1",
        "--compute-guard-max-model-forwards",
        "100",
        "--compute-guard-max-materialized-state-views",
        "1000",
        "--compute-guard-max-wall-seconds",
        "60",
    ]
    assert main(common) == 0


def test_campaign_context_separates_file_and_internal_sha(monkeypatch, tmp_path) -> None:
    plan_path = (tmp_path / "CAMPAIGN_PLAN.json").resolve()
    plan_path.write_text("{}")
    progress_path = (tmp_path / "PROGRESS.json").resolve()
    authority_path = (tmp_path / "AUTHORITY.json").resolve()
    invocation = {
        "invocation_sha256": "i" * 64,
        "kind": "full_val_window",
        "progress_path": str(progress_path),
        "checkpoint": {"pointer_path": str(tmp_path / "POINTER.json")},
    }
    authority = {
        "selected_batch_size": 16,
        "source_commit": "c" * 40,
        "final_checkpoint_pointer": {"path": str(tmp_path / "POINTER.json")},
    }
    plan = {
        "plan_sha256": "p" * 64,
        "phase": "full_val",
        "final_train_checkpoint_authority": {
            "path": str(authority_path),
            "sha256": "a" * 64,
        },
        "selected_batch_size": 16,
        "source_commit": "c" * 40,
        "checked_invocations": [invocation],
    }
    observed = {}

    def fake_read(path, expected_sha256):
        observed["path"] = path
        observed["expected_sha256"] = expected_sha256
        return {}

    monkeypatch.setattr(cli, "read_bound_json", fake_read)
    monkeypatch.setattr(cli, "require_plan", lambda _value, verify_files=True: plan)
    monkeypatch.setenv("GX1_CAMPAIGN_PLAN_PATH", str(plan_path))
    monkeypatch.setenv("GX1_CAMPAIGN_PLAN_FILE_SHA256", "f" * 64)
    monkeypatch.setenv("GX1_CAMPAIGN_PLAN_SHA256", "p" * 64)
    monkeypatch.setenv("GX1_CAMPAIGN_INVOCATION_SHA256", "i" * 64)
    checked, selected = cli._campaign_context(
        authority,
        authority_path=authority_path,
        authority_file_sha256="a" * 64,
        progress_path=progress_path,
    )
    assert checked is plan
    assert selected is invocation
    assert observed == {
        "path": plan_path,
        "expected_sha256": "f" * 64,
    }

    monkeypatch.setenv("GX1_CAMPAIGN_PLAN_SHA256", "x" * 64)
    with pytest.raises(RuntimeError, match="CAMPAIGN_SHA_INVALID"):
        cli._campaign_context(
            authority,
            authority_path=authority_path,
            authority_file_sha256="a" * 64,
            progress_path=progress_path,
        )
