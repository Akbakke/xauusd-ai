from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from tests.entry_model_native_train_wrapper_support import (
    DATASET_RUN_ID,
    RUN_ID,
    build_wrapper_contract,
)


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_train.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), "--profile", "candidate", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _capped_command_tokens(stdout: str) -> list[str]:
    line = next(
        row
        for row in stdout.splitlines()
        if row.startswith("Capped candidate train command:")
    )
    return shlex.split(line.split(":", 1)[1].strip())


def test_candidate_wrapper_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_candidate_wrapper_requires_complete_explicit_contract() -> None:
    result = _run("--run-id", RUN_ID, "--dry-run")

    assert result.returncode == 2
    assert "missing required argument" in result.stderr
    assert "Capped candidate train command:" not in result.stdout


def test_candidate_wrapper_rejects_unknown_retired_lane_argument() -> None:
    result = _run("--challenger-seq215", "--dry-run")

    assert result.returncode == 2
    assert "unknown argument" in result.stderr
    assert "Capped candidate train command:" not in result.stdout


def test_candidate_wrapper_validates_exact_contract_without_writes(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="candidate", wrapper=WRAPPER)
    env = os.environ.copy()
    env["ENTRY_STALE_WRAPPER_TEST"] = "must_be_scrubbed"
    env["GX1_STALE_WRAPPER_TEST"] = "must_be_scrubbed"
    result = subprocess.run(
        ["bash", str(WRAPPER), "--profile", "candidate", *args, "--dry-run"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated model-native seq513 candidate contract" in result.stdout
    assert "Capped candidate train command:" in result.stdout
    assert "gx1_capped_run.sh" in result.stdout
    assert f"--dataset-run-id {DATASET_RUN_ID}" in result.stdout
    assert "GX1_ENTRY_M5_PREBUILT_SHA256=" in result.stdout
    assert "--multi-tf-seq-len" not in result.stdout
    assert "--specialist-audit-json" in result.stdout
    assert "--mtf-dir-scale-init" not in result.stdout
    assert "--enable-" not in result.stdout
    assert not paths["out_bundle_dir"].exists()
    command = _capped_command_tokens(result.stdout)
    runner_index = command.index(str(REPO / "scripts/gx1_capped_run.sh"))
    separator_index = command.index("--", runner_index)
    assert command[0] == "/usr/bin/env"
    assert command[runner_index + 1 : runner_index + 4] == [
        "--class",
        "trainer",
        "--mem",
    ]
    assert command[separator_index + 1 : separator_index + 4] == [
        str(REPO / ".venv/bin/python"),
        "-m",
        "gx1.models.entry_v10.entry_v10_ctx_train_v3",
    ]
    assert command[separator_index + 4] == "--train"
    for stale_key in ("ENTRY_STALE_WRAPPER_TEST", "GX1_STALE_WRAPPER_TEST"):
        stale_index = command.index(stale_key)
        assert command[stale_index - 1] == "-u"
        assert stale_index < runner_index
        assert stale_key not in command[separator_index + 1 :]


def test_candidate_wrapper_rejects_zero_mandatory_recipe_value(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="candidate", wrapper=WRAPPER)
    recipe_path = paths["recipe_audit_json"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["trainer_env"]["ENTRY_MTF_DIR_AUX_WEIGHT"] = "0"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "MODEL_NATIVE_RECIPE_ENV_MISMATCH" in result.stderr
    assert "ENTRY_MTF_DIR_AUX_WEIGHT" in result.stderr


def test_candidate_wrapper_rejects_mutated_readiness_binding(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="candidate", wrapper=WRAPPER)
    readiness = paths["candidate_readiness_json"]
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["decision"] = "NOT_READY_FOR_CANDIDATE_TRAINING"
    readiness.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "candidate readiness decision" in result.stderr
    assert "Capped candidate train command:" not in result.stdout


def test_candidate_wrapper_source_is_exact_model_native_and_has_no_stale_launch_paths() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "MODEL_NATIVE_CONTRACT_MODE=xau_seq513_model_native_direction_v4" in text
    assert "MODEL_NATIVE_DIRECTION_LOGIT_MODE=model_native" in text
    assert "MODEL_NATIVE_SIGNAL_DIM=513" in text
    assert "PROFILE=smoke" not in text
    assert "PROFILE=candidate" not in text
    assert "--profile" in text
    assert "--recipe-audit-json" in text
    assert "--pretrain-audit-json" in text
    assert "--prefreeze-test-seal-json" in text
    assert "--prefreeze-test-seal-sha256" in text
    assert "--candidate-readiness-json" in text
    assert "--smoke-bundle-audit-json" in text
    assert "--execute" in text and "--run-id" in text
    assert 'TRAIN_CMD=(\n  "$PY" -m gx1.models.entry_v10.entry_v10_ctx_train_v3' in text
    assert 'RUN_CMD=(\n  "${ENV_COMMAND[@]}"\n  "$CAPPED_RUNNER" --class trainer' in text
    assert '-- "${TRAIN_CMD[@]}"' in text
    assert 'exec "${RUN_CMD[@]}"' in text
    for flag in (
        "--enable-pos-enc",
        "--enable-regime-film",
        "--enable-cross-tf-attn",
        "--enable-tf-agreement-head",
        "--enable-path-quality-variance-head",
        "--enable-position-size-head",
        "--enable-mtf-direction-head",
        "--enable-specialist-fusion",
        "--enable-hierarchical-entry-heads",
        "--enable-side-validity-head",
        "--enable-trendline-rail-head",
    ):
        assert flag not in text
    for stale in (
        "seq520",
        "tombstone",
        "run_manifest",
        "event_ledger",
        "neutral_external_tree_sidecar",
        "anchor_gate",
        "gx1_allow_legacy",
    ):
        assert stale not in lowered
