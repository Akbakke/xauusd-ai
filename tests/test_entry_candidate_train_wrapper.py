from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_train.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), "--profile", "candidate", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_legacy_wrapper_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_wrapper_rejects_candidate_before_any_recipe_or_dataset_input() -> None:
    result = _run("--dry-run")

    assert result.returncode == 2
    assert "legacy candidate profile is retired" in result.stderr
    assert "Capped candidate train command:" not in result.stdout


def test_legacy_wrapper_source_has_no_candidate_launch_surface() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "Usage: run_entry_model_native_seq513_train.sh --profile smoke" in source
    assert "legacy candidate profile is retired" in source
    assert "smoke|candidate" not in source
    assert source.index("  candidate)\n    die") < source.index(
        "PROFILE_VALIDATOR_ARGS=()"
    )
    assert '"$REPO/scripts/gx1_handover.sh" --check' in source


def test_control_surface_rejects_the_retired_candidate_route() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "model-native-candidate-train"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "model-native-candidate-train is retired" in result.stderr
    assert "immutable candidate launch gate" in result.stderr
