import subprocess
from pathlib import Path


REPO = Path("/home/andre2/src/GX1_ENGINE")
WRAPPER = REPO / "scripts/run_entry_foundation_seq146_smoke_train.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"
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
        check=True,
    )


def test_smoke_train_dry_run_prints_post_smoke_audit_command() -> None:
    result = _run_wrapper("--vedtak", "PYTEST_DRY_RUN", "--dry-run")

    assert "Smoke train command:" in result.stdout
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh verify --quiet" in result.stdout
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh foundation-guardrails --quiet" in result.stdout
    assert "Real-train preflight command: scripts/entry_next_edge_control.sh train-readiness --quiet" in result.stdout
    assert "Pre-train run manifest path:" in result.stdout
    assert "Smoke resource cap: mem=22G swap=2G runner=scripts/gx1_capped_run.sh num_workers=0" in result.stdout
    assert "Capped smoke train command:" in result.stdout
    assert "scripts/gx1_capped_run.sh --mem 22G --swap 2G --" in result.stdout
    assert "ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST" in result.stdout
    assert "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1" in result.stdout
    assert "ENTRY_AUX_BAD_PATH_WEIGHT=1.00" in result.stdout
    assert "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT=2.00" in result.stdout
    assert "ENTRY_BAD_PATH_QUALITY_RANK_MARGIN=0.25" in result.stdout
    assert "ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE=0.25" in result.stdout
    assert "ENTRY_PRED_BALANCE_ALPHA=0.05" in result.stdout
    assert "GX1_V10_CKPT_MONITOR=dir_acc" in result.stdout
    assert "ENTRY_SYMMETRIC_NEGATIVES=1" in result.stdout
    assert "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT=0.05" in result.stdout
    assert "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT=0.25" in result.stdout
    assert "ENTRY_SPECIALIST_GATE_MIN_MEAN=0.02" in result.stdout
    assert "--enable-specialist-fusion" in result.stdout
    for flag in EXPECTED_AUX_FLAGS:
        assert flag in result.stdout
    assert "--enable-hold-horizon-head" not in result.stdout
    assert "Post-smoke audit command:" in result.stdout
    assert "audit-smoke-bundle" in result.stdout
    assert "--bundle-dir" in result.stdout
    assert "--dataset-dir" in result.stdout
    assert "--out-dir" in result.stdout
    assert "entry_foundation_smoke_bundle_audit_20260628_v1" in result.stdout
    assert "--require-head-contract" in result.stdout
    assert "--pretrain-manifest-json" in result.stdout
    assert "--require-edge" in result.stdout


def test_smoke_train_dry_run_can_explicitly_skip_post_smoke_audit() -> None:
    result = _run_wrapper(
        "--vedtak",
        "PYTEST_DRY_RUN",
        "--dry-run",
        "--no-require-edge-audit",
        "--skip-smoke-audit",
    )

    assert "Smoke train command:" in result.stdout
    assert "Post-smoke audit command: skipped by --skip-smoke-audit" in result.stdout


def test_smoke_train_rejects_require_edge_without_audit() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--vedtak",
            "PYTEST_DRY_RUN",
            "--dry-run",
            "--skip-smoke-audit",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "--require-edge-audit cannot be combined with --skip-smoke-audit" in result.stderr


def test_seq215_manifest_requires_seq215_vedtak() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--challenger-seq215",
            "--vedtak",
            "ENTRY_FOUNDATION_SMOKE_TRAIN_20260629_SEQ146_V1",
            "--manifest-only",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "requires an explicit SEQ215 vedtak id" in result.stderr


def test_seq215_smoke_dry_run_requires_seq215_vedtak() -> None:
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--challenger-seq215",
            "--vedtak",
            "ENTRY_FOUNDATION_SMOKE_TRAIN_20260630_SEQ146_V1",
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "requires an explicit SEQ215 vedtak id" in result.stderr
    assert "Smoke train command:" not in result.stdout


def test_smoke_train_wrapper_enforces_train_readiness_for_real_train() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "entry_next_edge_control.sh verify --quiet" in text
    assert "entry_next_edge_control.sh foundation-guardrails --quiet" in text
    assert "entry_next_edge_control.sh train-readiness --quiet" in text
    assert "ENTRY_FOUNDATION_SMOKE_TRAIN_RUN_MANIFEST" in text
    assert "trainer_started_by_manifest_writer" in text
    assert "smoke_recipe_env" in text
    assert "command_env_value" in text
    assert "ENTRY_AUX_BAD_PATH_WEIGHT" in text
    assert "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT" in text
    assert "ENTRY_BAD_PATH_QUALITY_RANK_MARGIN" in text
    assert "ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE" in text
    assert "ENTRY_PRED_BALANCE_ALPHA" in text
    assert "GX1_V10_CKPT_MONITOR" in text
    assert "ENTRY_SYMMETRIC_NEGATIVES" in text
    assert "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT" in text
    assert "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT" in text
    assert "ENTRY_SPECIALIST_GATE_MIN_MEAN" in text
    assert "artifact_sha256" in text
    assert "artifact_provenance_decision" in text
    assert "artifact_fingerprints" in text
    assert "def artifact_fingerprint" in text
    assert "def run_artifact_fingerprints" in text
    assert 'if os.environ.get("RUN_FLAVOR", "foundation_seq146") != "foundation_seq146"' in text
    assert "FATAL: non-foundation manifest artifact fingerprint missing" in text
    assert "FATAL: non-foundation manifest artifact fingerprint hash mismatch" in text
    assert '"artifact_fingerprints": readiness_artifact_fingerprints' in text
    assert 'gate_decision(readiness, "artifact_provenance")' in text
    assert "--pretrain-manifest-json" in text
    assert "preflight_contracts" in text
    assert "feature_contract_summary" in text
    assert "foundation_objective_coverage_all_present" in text
    assert "foundation_objective_liveness_all_live" in text
    assert "foundation_source_field_liveness_all_live" in text
    assert "foundation_source_fields_by_split" in text
    assert "specialist_contract_summary" in text
    assert "required_training_specialists" in text
    assert "trainable_specialists" in text
    assert "excluded_specialist_groups" in text
    assert "specialist_model_contract_valid" in text
    assert "specialist_model_contract_failures" in text
    assert "specialist_model_contract" in text
    assert "_load_specialist_fusion_contract" in text
    assert "SPECIALIST_CONTRACT_MODE=foundation_seq146" in text
    assert "--specialist-contract-mode \"$SPECIALIST_CONTRACT_MODE\"" in text
    assert 'contract_mode = os.environ.get("SPECIALIST_CONTRACT_MODE", "foundation_seq146")' in text
    assert "--challenger-seq215" in text
    assert "SPECIALIST_CONTRACT_MODE=challenger_seq215" in text
    assert "EXPECTED_SIGNAL_DIM=215" in text
    assert "entry_foundation_seq215_smoke_dataset_v1" in text
    assert "SMOKE_BUNDLE_AUDIT_OUT" in text
    assert "entry_foundation_smoke_bundle_audit_20260628_v1/challenger_seq215_20260630" in text
    assert '--out-dir "$SMOKE_BUNDLE_AUDIT_OUT"' in text
    assert "architecture_active_heads" in text
    assert "architecture_blocked_heads" in text
    assert "foundation_objective_routing_all_present_and_expected" in text
    assert "specialist_input_liveness_all_live" in text
    assert "target_contract_summary" in text
    assert "smoke_dataset_contract_summary" in text
    assert "audit_provenance_all_artifact_hashes_present" in text
    assert "worktree_contract_summary" in text
    assert "foundation_cleanup_critical_gate_review" in text
    assert "critical_gate_path_count" in text
    assert "--manifest-only" in text
    assert "Manifest-only stop before training" in text
    assert "require_clean_git_for_real_train" in text
    assert "require_foundation_contract_ready_for_manifest_only" in text
    assert "--no-fail-on-not-ready" in text
    assert "foundation_contract_ready_for_smoke" in text
    assert "git status --short" in text
    assert "real foundation smoke train requires clean git worktree" in text
    assert 'if [[ "$DRY_RUN" != "1" ]]' in text
    preflight_block = text[text.index('if [[ "$DRY_RUN" != "1" ]]') : text.index("STAMP=")]
    real_train_branch = preflight_block.split("else", 1)[1]
    assert real_train_branch.index("require_clean_git_for_real_train") < real_train_branch.index(
        "entry_next_edge_control.sh train-readiness --quiet"
    )
    assert "REQUIRE_EDGE_AUDIT=1" in text
    assert "--no-require-edge-audit" in text
    assert 'SMOKE_CAPPED_RUNNER=scripts/gx1_capped_run.sh' in text
    assert 'Smoke resource cap: mem=$SMOKE_RUN_MEM swap=$SMOKE_RUN_SWAP runner=$SMOKE_CAPPED_RUNNER num_workers=0' in text
    assert 'Capped smoke train command:' in text
    assert '"memory_cap": os.environ.get("SMOKE_RUN_MEM")' in text
    assert '"swap_cap": os.environ.get("SMOKE_RUN_SWAP")' in text
    assert '"cgroup_runner": "scripts/gx1_capped_run.sh"' in text
    assert '"uses_gx1_capped_run": True' in text
    assert '"num_workers": int(command_arg_value(train_cmd, "--num-workers") or -1)' in text


def test_control_surface_exposes_manifest_only_smoke_proof() -> None:
    text = CONTROL.read_text(encoding="utf-8")

    assert "scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>" in text
    assert "scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>" in text
    assert "scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit" in text
    assert "smoke-manifest)" in text
    assert "smoke-manifest-seq215)" in text
    assert "smoke-train-seq215)" in text
    assert 'run_entry_foundation_seq146_smoke_train.sh" --manifest-only' in text
    assert 'run_entry_foundation_seq146_smoke_train.sh" --challenger-seq215 --manifest-only' in text
