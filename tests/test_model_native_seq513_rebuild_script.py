from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rebuild_entry_model_native_seq513_dataset.sh"
CAPPED_RUNNER = REPO / "scripts" / "gx1_capped_run.sh"
PRE_COMMIT = REPO / ".claude" / "git-hooks" / "pre-commit"


def test_seq513_rebuild_is_explicit_model_native_and_never_trains() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for required in (
        "--run-id",
        "--source-parquet",
        "--canonical-v2-parquet",
        "--signal-manifest",
        "--feature-ranking-json",
        "--mtf-cache-dir",
        "--tape-root",
        "--m1-lifecycle-pair-manifest-json",
        "--m1-lifecycle-pair-generation-root",
        "--m1-feature-base-parquet",
        "--m5-feature-base-parquet",
        "--exit-lifecycle-dir",
        "--output",
        "--audit-out-dir",
        "--rebuild-terminal-json",
        "--prefreeze-test-seal-json",
        "--history-start",
        "validate_signal_manifest_training_lineage",
        "expected_run_id",
        "expected_source_sha256",
        "expected_train_start_utc",
        "expected_train_end_utc",
        "materialize_entry_full_input_liveness_v1",
        "ENTRY_FULL_INPUT_LIVENESS_CONTRACT_",
        "validate_full_input_liveness_artifact",
        "audit_xau_direction_repair_pretrain_v1",
    ):
        assert required in source

    assert "entry_v10_ctx_train_v3" not in source
    assert "--exit-target-lookahead-m1-steps" not in source
    assert "--base28-manifest" not in source
    assert "--base28_manifest" not in source
    assert "--epochs" not in source
    assert "RUN_TRAIN" not in source
    assert "--neutral-external_tree_sidecar-bridge" not in source
    assert "--allow-missing-hold-map" not in source
    assert "--hold-map-source" not in source
    assert "y_hold_horizon_target" not in source
    assert "smart_seq520" not in source.lower()
    assert "neutral_uniform_proba" not in source
    assert "v10_6yr_rebuild_20260626" not in source
    # The fixed top-133 TRAIN rank subsystem is retired end to end: the
    # rebuild script may not name its producer, artifact or CLI surface.
    assert "materialize_model_native_rank_reference_v1" not in source
    assert "materialize_model_native_train_rank_reference" not in source
    assert "validate_train_rank_reference_lineage" not in source
    assert "rank-reference" not in source
    assert "RANK_NPZ" not in source
    assert "--seq-structure-features-parquet" not in source
    assert "--seq-structure-compute-inline" not in source
    assert "--fail-on-audit-fail" not in source
    assert "--require-rail-features" not in source
    assert "--require-inline-seq-structure" not in source
    assert "--require-xau-provenance" not in source
    assert 'ls ' not in source
    assert 'tail -1' not in source
    assert "EARLY_MOVE_THRESHOLD_BPS=" not in source
    assert "--early_move_threshold_bps" not in source


def test_seq513_rebuild_does_not_hide_target_or_sequence_defaults() -> None:
    builder = (REPO / "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py").read_text(
        encoding="utf-8"
    )

    import ast

    tree = ast.parse(builder)
    observed: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--seq_len"
        ):
            continue
        observed[str(node.args[0].value)] = node

    assert set(observed) == {"--seq_len"}
    for flag, node in observed.items():
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        assert isinstance(keywords.get("required"), ast.Constant)
        assert keywords["required"].value is True, flag
        assert "default" not in keywords, flag
    assert "--early_move_threshold_bps" not in builder
    assert 'direction_target_policy["early_move_threshold_bps"]' in builder
    assert "fit_entry_direction_target_policy(" in builder


def test_seq513_rebuild_rejects_legacy_environment_and_existing_outputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    retired_bridge_env = "GX1_" + "X" + "GB" + "_BUNDLE_DIR"
    assert retired_bridge_env not in source
    assert "GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH" not in source
    assert "GX1_MTF_CACHE_ALLOW_STALE" in source
    assert "GX1_REGIME_V4 GX1_TREND_REGIME_FROM_D1" in source
    assert "export GX1_REGIME_V4" not in source
    assert "export GX1_TREND_REGIME_FROM_D1" not in source
    assert "output split already exists" in source
    assert "dataset build proof already exists" in source
    assert "audit output directory already exists" in source
    assert source.count('--run-id "$RUN_ID"') == 2
    assert source.count('--feature-ranking-json "$FEATURE_RANKING_JSON"') == 1
    assert "SOURCE_PARQUET CANONICAL_V2_PARQUET SIGNAL_MANIFEST FEATURE_RANKING_JSON" in source
    assert "--run-id has invalid format" in source
    assert "--resume-exact-checkpoints" in source
    # The signal/ranking lineage gate binds the explicit chain run id and the
    # freshly hashed source bytes; it now receives the run id positionally.
    assert "expected_run_id=sys.argv[3]" in source
    assert '"$SIGNAL_MANIFEST" "$FEATURE_RANKING_JSON" "$RUN_ID"' in source
    assert "expected_source_sha256=digest.hexdigest()" in source


def test_seq513_rebuild_full_input_liveness_precedes_target_preflight() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    materialize = source.index("materialize_entry_full_input_liveness_v1")
    validate = source.index("validate_full_input_liveness_artifact")
    target_audit = source.index("audit_xau_direction_repair_pretrain_v1")

    assert materialize < validate < target_audit


def test_seq513_rebuild_seals_test_before_prefreeze_audits() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    producer = source.index("--prefreeze-test-seal-json")
    seal_gate = source.index("exact pre-freeze TEST authority")
    liveness = source.index("materialize_entry_full_input_liveness_v1")
    pretrain = source.index("audit_xau_direction_repair_pretrain_v1")

    assert producer < seal_gate < liveness < pretrain
    assert "--test-manifest-json" not in source
    assert "--test-parquet" not in source


def test_capped_runner_serializes_every_heavy_job() -> None:
    source = CAPPED_RUNNER.read_text(encoding="utf-8")

    assert "gx1-heavy-job.lock" in source
    assert "flock -n 9" in source
    assert "another GX1 heavy job owns the exclusive lock" in source
    assert "exit 75" in source
    assert "systemd-run --user --scope --quiet" in source
    assert "exec systemd-run" not in source
    assert "SAFE_JOB_MEMORY_KIB=$((20 * 1024 * 1024))" in source
    assert "SAFE_AUDIT_MEMORY_KIB=$((4 * 1024 * 1024))" in source
    assert "SAFE_JOB_SWAP_KIB=$((512 * 1024))" in source
    assert "MIN_AVAILABLE_MEMORY_KIB=$((20 * 1024 * 1024))" in source
    assert "WSL_GUARD_MIN_REQUEST_KIB=$((4 * 1024 * 1024))" in source
    assert "active WSL MemTotal exceeds configured memory cap" in source
    assert "capped_run_scope_verified" in source
    assert "--class)" in source
    assert 'JOB_CLASS="$2"; shift 2' in source
    assert "trainer class is reserved for the canonical trainer module" in source
    assert "audit jobs may request at most 4G" in source
    assert 'verified_cpu_affinity="$GX1_CPU_AFFINITY"' in source
    assert (
        "unset GX1_EXPECTED_MEMORY_BYTES GX1_EXPECTED_SWAP_BYTES "
        "GX1_EXPECTED_TASKS GX1_CPU_AFFINITY"
    ) in source
    assert '/usr/bin/taskset -c "$verified_cpu_affinity"' in source
    assert "/usr/bin/taskset -c" in source
    assert "--setenv=OMP_NUM_THREADS=1" in source


def test_pre_commit_model_contracts_use_capped_runner() -> None:
    source = PRE_COMMIT.read_text(encoding="utf-8")

    assert (
        'CAP=("$REPO/scripts/gx1_capped_run.sh" --class audit '
        '--mem 4G --swap 512M --)'
    ) in source
    assert '"${CAP[@]}" "$PY" -m pytest -q' in source


def test_seq513_rebuild_caps_every_heavy_stage() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    # The dataset rebuild is a heavy producer over the complete population, not
    # an audit. The pre-wave chain gave it 10G; capping it at 4G is what killed
    # the ranker and the M1 enriched lane.
    assert (
        'CAP=("$ENG/scripts/gx1_capped_run.sh" --class producer '
        '--mem 10G --swap 512M --)'
    ) in source
    assert '"${CAP[@]}" "$PY" -m gx1.scripts.audit_xau_direction_repair_pretrain_v1' in source
