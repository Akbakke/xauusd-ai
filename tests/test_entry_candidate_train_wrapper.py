from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
    MODEL_NATIVE_RECIPE_ENV_KEYS,
)
from tests.entry_model_native_train_wrapper_support import (
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


def test_candidate_wrapper_rejects_attended_smoke_mode() -> None:
    result = _run("--attended-smoke", "--dry-run")

    assert result.returncode == 2
    assert "valid only for --profile smoke" in result.stderr


def test_candidate_wrapper_rejects_research_smoke_mode() -> None:
    result = _run("--research-smoke", "--dry-run")

    assert result.returncode == 2
    assert "disabled after the WSL/GPU reset" in result.stderr


def test_candidate_wrapper_rejects_self_consistent_noncanonical_dataset(
    tmp_path: Path,
) -> None:
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

    assert result.returncode == 2
    assert "candidate dataset does not match current audited dataset" in result.stderr
    assert "Validated model-native seq513 candidate contract" not in result.stdout
    assert "Capped candidate train command:" not in result.stdout
    assert not paths["out_bundle_dir"].exists()
    assert not paths["out_bundle_dir"].exists()


def test_candidate_wrapper_rejects_zero_mandatory_recipe_value(tmp_path: Path) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="candidate", wrapper=WRAPPER)
    # The zeroed key is taken from the recipe owner, not named here: the
    # hard-coded key this test used to zero was retired by Wave C, and a test
    # that names a key cannot notice when the surface moves under it.
    zeroed_key = next(
        key
        for key in MODEL_NATIVE_RECIPE_ENV_KEYS
        if MODEL_NATIVE_RECIPE_ENV[key] != "0"
    )
    recipe_path = paths["recipe_audit_json"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["trainer_env"][zeroed_key] = "0"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    # Candidate launch now requires a current canonical dataset binding before
    # it reaches any mutable recipe. The fixture intentionally is not V46, so
    # this authority boundary rejects first; recipe env values are covered by
    # the recipe-owner tests.
    assert "candidate dataset does not match current audited dataset" in result.stderr


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


def test_candidate_wrapper_rejects_readiness_for_another_dataset(
    tmp_path: Path,
) -> None:
    args, paths = build_wrapper_contract(tmp_path, profile="candidate", wrapper=WRAPPER)
    readiness = paths["candidate_readiness_json"]
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["smoke_bundle_dataset_dir"] = str(tmp_path / "other-dataset")
    readiness.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = _run(*args, "--dry-run")

    assert result.returncode == 2
    assert "candidate readiness dataset binding mismatch" in result.stderr
    assert "Capped candidate train command:" not in result.stdout


def test_candidate_wrapper_source_is_exact_model_native_and_has_no_stale_launch_paths() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    lowered = text.lower()

    # Rule 13: these used to assert the two literals "…_v18" and "279" were
    # PRESENT in the wrapper.  That is what let them go stale — the owner moved
    # to v20/238 and both suites stayed green while the training chain could not
    # start, because --specialist-contract-mode is compared for exact equality.
    # The wrapper now reads both from the contract owner at launch, so the test
    # asserts OWNERSHIP: no hand-written value, and the derivation is present.
    assert "MODEL_NATIVE_CONTRACT_MODE=xau_seq513_model_native_direction_v" not in text
    assert "MODEL_NATIVE_SIGNAL_DIM=" not in text.replace(
        "MODEL_NATIVE_SIGNAL_DIM=$", ""
    ) or "s.MODEL_NATIVE_SIGNAL_DIM" in text
    assert "gx1.contracts.entry_model_native_signal_v1" in text
    assert "s.MODEL_NATIVE_CONTRACT_MODE" in text
    assert "s.MODEL_NATIVE_SIGNAL_DIM" in text
    assert '"$REPO/scripts/gx1_handover.sh" --check' in text
    assert "unexpected_ignored_path_count: 0" in text
    assert "prunable_worktree_count: 0" in text
    assert "--execute rejects unexpected ignored content" in text
    assert "MODEL_NATIVE_DIRECTION_LOGIT_MODE=model_native" in text
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


def test_candidate_wrapper_caps_large_preflight_validation() -> None:
    """Even --dry-run rehashes large immutable bindings, so it is a capped job."""

    source = WRAPPER.read_text(encoding="utf-8")
    assert 'RECIPE_ENV_TEXT=$(cd "$REPO" && "$CAPPED_RUNNER" \\' in source
    assert '--class audit --mem 4G --swap 512M -- \\' in source
    assert (
        '"$PY" -m gx1.contracts.entry_model_native_train_launch_v1 '
        '"${VALIDATOR_ARGS[@]}"'
    ) in source
