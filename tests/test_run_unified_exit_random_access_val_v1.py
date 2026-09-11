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
