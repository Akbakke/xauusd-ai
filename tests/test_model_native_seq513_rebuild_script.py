from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rebuild_entry_model_native_seq513_dataset.sh"


def test_seq513_rebuild_is_explicit_model_native_and_never_trains() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for required in (
        "--run-id",
        "--source-parquet",
        "--canonical-v2-parquet",
        "--signal-manifest",
        "--feature-ranking-json",
        "--rank-reference-npz",
        "--mtf-cache-dir",
        "--tape-root",
        "--output",
        "--audit-out-dir",
        "--history-start",
        "materialize_model_native_train_rank_reference_v2",
        "--fit-start",
        "--fit-end",
        "validate_signal_manifest_training_lineage",
        "expected_run_id",
        "expected_source_sha256",
        "expected_train_start_utc",
        "expected_train_end_utc",
        "--model-native-rank-reference-npz",
        "materialize_entry_full_input_liveness_v1",
        "ENTRY_FULL_INPUT_LIVENESS_CONTRACT_",
        "validate_full_input_liveness_artifact",
        "audit_xau_direction_repair_pretrain_v1",
    ):
        assert required in source

    assert "entry_v10_ctx_train_v3" not in source
    assert "--base28-manifest" not in source
    assert "--base28_manifest" not in source
    assert "--epochs" not in source
    assert "RUN_TRAIN" not in source
    assert "--neutral-xgb-bridge" not in source
    assert "--allow-missing-hold-map" not in source
    assert "--hold-map-source" not in source
    assert "y_hold_horizon_target" not in source
    assert "smart_seq520" not in source.lower()
    assert "neutral_uniform_proba" not in source
    assert "v10_6yr_rebuild_20260626" not in source
    assert "materialize_model_native_rank_reference_v1" not in source
    assert "--seq-structure-features-parquet" not in source
    assert "--seq-structure-compute-inline" not in source
    assert "--fail-on-audit-fail" not in source
    assert "--require-rail-features" not in source
    assert "--require-inline-seq-structure" not in source
    assert "--require-xau-provenance" not in source
    assert 'ls ' not in source
    assert 'tail -1' not in source


def test_seq513_rebuild_rejects_legacy_environment_and_existing_outputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "GX1_XGB_BUNDLE_DIR" in source
    assert "GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH" not in source
    assert "GX1_MTF_CACHE_ALLOW_STALE" in source
    assert "GX1_REGIME_V4 GX1_TREND_REGIME_FROM_D1" in source
    assert "export GX1_REGIME_V4" not in source
    assert "export GX1_TREND_REGIME_FROM_D1" not in source
    assert "output split already exists" in source
    assert "dataset build proof already exists" in source
    assert "rank reference already exists" in source
    assert "audit output directory already exists" in source
    assert source.count('--run-id "$RUN_ID"') == 3
    assert source.count('--feature-ranking-json "$FEATURE_RANKING_JSON"') == 1
    assert "SOURCE_PARQUET CANONICAL_V2_PARQUET SIGNAL_MANIFEST FEATURE_RANKING_JSON" in source
    assert "--run-id has invalid format" in source


def test_seq513_rebuild_full_input_liveness_precedes_target_preflight() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    materialize = source.index("materialize_entry_full_input_liveness_v1")
    validate = source.index("validate_full_input_liveness_artifact")
    target_audit = source.index("audit_xau_direction_repair_pretrain_v1")

    assert materialize < validate < target_audit
