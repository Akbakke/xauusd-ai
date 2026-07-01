import subprocess
from pathlib import Path


REPO = Path("/home/andre2/src/GX1_ENGINE")
WRAPPER = REPO / "scripts/run_entry_foundation_seq146_candidate_train.sh"
EXPECTED_AUX_FLAGS = (
    "--enable-tf-agreement-head",
    "--enable-path-quality-variance-head",
    "--enable-position-size-head",
    "--enable-dip-head",
    "--enable-forecast-head",
    "--enable-timing-head",
    "--enable-tail-risk-head",
    "--enable-vol-forecast-head",
    "--enable-mtf-direction-head",
)


def _run_wrapper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_candidate_train_requires_explicit_vedtak() -> None:
    result = _run_wrapper("--dry-run")

    assert result.returncode == 2
    assert "--vedtak is required" in result.stderr
    assert "Candidate train command:" not in result.stdout


def test_candidate_train_blocks_when_candidate_readiness_is_not_ready() -> None:
    result = _run_wrapper("--vedtak", "PYTEST_DRY_RUN", "--dry-run")

    assert result.returncode == 2
    assert "candidate-readiness is NOT_READY" in result.stderr
    assert "smoke-train --vedtak <id> --require-edge-audit" in result.stderr
    assert "Candidate train command:" not in result.stdout


def test_seq215_candidate_train_blocks_with_seq215_next_gate_when_not_ready() -> None:
    result = _run_wrapper(
        "--challenger-seq215",
        "--vedtak",
        "ENTRY_FOUNDATION_CANDIDATE_TRAIN_20260630_SEQ215_V1",
        "--dry-run",
    )

    assert result.returncode == 2
    assert "candidate-readiness is NOT_READY" in result.stderr
    assert "smoke-train-seq215 --vedtak <id> --require-edge-audit" in result.stderr
    assert "candidate-readiness-seq215" in result.stderr
    assert "Candidate train command:" not in result.stdout


def test_seq215_candidate_train_dry_run_requires_seq215_vedtak_before_readiness() -> None:
    result = _run_wrapper(
        "--challenger-seq215",
        "--vedtak",
        "ENTRY_FOUNDATION_CANDIDATE_TRAIN_20260630_SEQ146_V1",
        "--dry-run",
    )

    assert result.returncode == 2
    assert "requires an explicit SEQ215 vedtak id" in result.stderr
    assert "candidate-readiness is NOT_READY" not in result.stderr
    assert "Candidate train command:" not in result.stdout


def test_candidate_train_wrapper_declares_live_aux_heads_without_hold_horizon() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    for flag in EXPECTED_AUX_FLAGS:
        assert flag in text
    assert "--enable-hold-horizon-head" not in text


def test_candidate_train_wrapper_declares_post_candidate_head_contract_audit() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "AUDIT_CMD" in text
    assert "audit-smoke-bundle" in text
    assert "--require-head-contract" in text
    assert "--pretrain-manifest-json" in text
    assert "ENTRY_FOUNDATION_CANDIDATE_TRAIN_RUN_MANIFEST" in text
    assert "entry_foundation_candidate_train_run_manifest_v1" in text
    assert 'CANDIDATE_MEM_CAP="${ENTRY_FOUNDATION_CANDIDATE_MEM_CAP:-32G}"' in text
    assert 'CANDIDATE_SWAP_CAP="${ENTRY_FOUNDATION_CANDIDATE_SWAP_CAP:-2G}"' in text
    assert "CANDIDATE_CAPPED_RUNNER=scripts/gx1_capped_run.sh" in text
    assert 'CANDIDATE_BAD_PATH_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_WEIGHT:-1.00}"' in text
    assert (
        'CANDIDATE_BAD_PATH_QUALITY_RANK_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_QUALITY_RANK_WEIGHT:-2.00}"'
        in text
    )
    assert 'CANDIDATE_PRED_BALANCE_ALPHA="${ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA:-0.05}"' in text
    assert 'CANDIDATE_PRED_BALANCE_TARGET="${ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_TARGET:-label}"' in text
    assert 'CANDIDATE_DIRECTION_CE_SCALE="${ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE:-1.30}"' in text
    assert 'CANDIDATE_CKPT_MONITOR="${ENTRY_FOUNDATION_CANDIDATE_CKPT_MONITOR:-dir_acc}"' in text
    assert (
        'CANDIDATE_SPECIALIST_GATE_BALANCE_WEIGHT="${ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_GATE_BALANCE_WEIGHT:-0.25}"'
        in text
    )
    assert '--mem "$CANDIDATE_MEM_CAP" --swap "$CANDIDATE_SWAP_CAP"' in text
    assert (
        "Candidate resource cap: mem=$CANDIDATE_MEM_CAP swap=$CANDIDATE_SWAP_CAP "
        "runner=$CANDIDATE_CAPPED_RUNNER num_workers=0"
        in text
    )
    assert "Capped candidate train command:" in text
    assert '"memory_cap": sys.argv[12]' in text
    assert '"swap_cap": sys.argv[13]' in text
    assert '"cgroup_runner": "scripts/gx1_capped_run.sh"' in text
    assert '"uses_gx1_capped_run": True' in text
    assert '"num_workers": int(command_arg_value(train_cmd, "--num-workers") or -1)' in text
    assert "candidate_recipe_env" in text
    assert "command_env_value" in text
    assert "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1" in text
    assert "ENTRY_AUX_BAD_PATH_WEIGHT=" in text
    assert "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT=" in text
    assert "ENTRY_PRED_BALANCE_ALPHA=" in text
    assert "ENTRY_PRED_BALANCE_TARGET=" in text
    assert "ENTRY_DIRECTION_CE_SCALE=" in text
    assert "GX1_V10_CKPT_MONITOR=" in text
    assert "ENTRY_SYMMETRIC_NEGATIVES=" in text
    assert "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT=" in text
    assert "artifact_sha256" in text
    assert "artifact_provenance_decision" in text
    assert "artifact_fingerprints" in text
    assert "report_json_path" in text
    assert "candidate_readiness_json_path" in text
    assert '"candidate_readiness_json": candidate_readiness_json_path' in text
    assert '"candidate_readiness": sha256_file(candidate_readiness_json_path)' in text
    assert 'gate_decision(candidate_readiness, "artifact_provenance")' in text
    assert "candidate_readiness_json" in text
    assert "smoke_bundle_audit_json" in text
    assert "required_training_specialists" in text
    assert "specialist_groups" in text
    assert "trainable_specialists" in text
    assert "excluded_specialist_groups" in text
    assert "specialist_model_contract_valid" in text
    assert "specialist_model_contract_set_exact" in text
    assert "specialist_model_contract_owned_objectives_match" in text
    assert "specialist_model_contract" in text
    assert "_load_specialist_fusion_contract" in text
    assert "--specialist-contract-mode \"$SPECIALIST_CONTRACT_MODE\"" in text
    assert 'SPECIALIST_CONTRACT_MODE=foundation_seq146' in text
    assert 'SPECIALIST_CONTRACT_MODE=challenger_seq215' in text
    assert 'EXPECTED_SIGNAL_DIM=215' in text
    assert '--contract-mode "$SPECIALIST_CONTRACT_MODE"' in text
    assert "--challenger-seq215" in text
    assert "challenger_seq215_20260630_contract8" in text
    assert "entry_foundation_smoke_bundle_audit_20260628_v1/challenger_seq215_20260630" in text
    assert "v10_dataset_challenger_seq215_neutral_20260630" in text
    assert 'contract_mode=sys.argv[15]' in text
    assert "expected_signal_dim=int(sys.argv[16])" in text
    assert "feature_objective_liveness_all_live" in text
    assert "feature_source_field_liveness_all_live" in text
    assert "specialist_active_heads_match_target" in text
    assert "specialist_blocked_heads_match_target" in text
    assert "smoke_dataset_audit_provenance_all_artifacts_present" in text
    assert "smoke_dataset_audit_provenance_all_artifact_hashes_present" in text
    assert "worktree_critical_gate_review_ok" in text
    assert "architecture_active_heads" in text
    assert "architecture_blocked_heads" in text
    assert "specialist_input_liveness_all_live" in text
    assert "entry_candidate_bundle_audit_20260628_v1" in text
    assert "--skip-candidate-audit" in text
    assert "require_clean_git_for_real_candidate_train" in text
    assert "git status --short" in text
    assert "real foundation candidate train requires clean git worktree" in text
    assert 'if [[ "$DRY_RUN" = "1" ]]' in text
    dry_run_block = text[text.index('if [[ "$DRY_RUN" = "1" ]]') : text.index('mkdir -p "$CANDIDATE_TRAIN_MANIFEST_DIR"')]
    assert dry_run_block.index("require_clean_git_for_real_candidate_train") > dry_run_block.index("exit 0")
