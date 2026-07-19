from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tests.entry_model_native_train_wrapper_support import (
    VEDTAK,
    build_wrapper_contract,
)


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts/run_entry_model_native_seq513_candidate_train.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


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
    result = _run("--vedtak", VEDTAK, "--dry-run")

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

    result = _run(*args, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Validated model-native seq513 candidate contract" in result.stdout
    assert "Capped candidate train command:" in result.stdout
    assert "gx1_capped_run.sh" in result.stdout
    assert "--multi-tf-seq-len 96" in result.stdout
    assert "--specialist-audit-json" in result.stdout
    assert "--mtf-dir-scale-init" not in result.stdout
    assert "--enable-" not in result.stdout
    assert not paths["out_bundle_dir"].exists()


def test_candidate_wrapper_rejects_zero_mandatory_recipe_value(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="candidate", wrapper=WRAPPER)
    recipe_path = paths["recipe_audit_json"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["trainer_env"]["ENTRY_MTF_DIR_AUX_WEIGHT"] = "0"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "must be finite and > 0" in result.stderr


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

    assert "MODEL_NATIVE_CONTRACT_MODE=xau_seq513_model_native_direction_v1" in text
    assert "MODEL_NATIVE_DIRECTION_LOGIT_MODE=model_native" in text
    assert "MODEL_NATIVE_SIGNAL_DIM=513" in text
    assert "--recipe-audit-json" in text
    assert "--pretrain-audit-json" in text
    assert "--candidate-readiness-json" in text
    assert "--smoke-bundle-audit-json" in text
    assert "--execute" in text and "--vedtak" in text
    assert 'RUN_CMD=("$CAPPED_RUNNER" --mem "$MEMORY_CAP" --swap "$SWAP_CAP" --' in text
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
        "neutral_xgb",
        "anchor_gate",
        "gx1_allow_legacy",
    ):
        assert stale not in lowered
