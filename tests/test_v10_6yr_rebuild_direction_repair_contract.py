from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v10_6yr_rebuild_uses_smart_seq520_inline_direction_repair_surface() -> None:
    text = (REPO / "scripts" / "v10_6yr_rebuild_20260626.sh").read_text(encoding="utf-8")

    assert "SMART_SEQ_STRUCTURE_MANIFEST" in text
    assert "ENTRY_SPECIALIST_CHALLENGER_SMART_EXTENSION_MANIFEST_latest.json" in text
    assert "--seq-structure-manifest" in text
    assert "--seq-structure-compute-inline" in text
    assert "--neutral-xgb-bridge" in text
    assert "--allow-missing-hold-map" in text
    assert "materialize_smart520_rank_reference_v1" in text
    assert "--smart520-rank-reference-npz" in text
    assert "smart520_state_contract" in text
    assert "rank_reference_npz_sha256" in text
    assert "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT=${ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT:-4.00}" in text
    assert "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS=${ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS:-15.0}" in text
    assert "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN=${ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN:-0.10}" in text
    assert "ENTRY_CKPT_DIRECTION_SLICE_GUARD=${ENTRY_CKPT_DIRECTION_SLICE_GUARD:-1}" in text
    assert (
        "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT="
        "${ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN="
        "${ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN:-0.02}" in text
    )
    assert (
        "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT="
        "${ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN="
        "${ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN:-0.02}" in text
    )
    assert "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION=${ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION:-mean_max}" in text
    assert (
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT="
        "${ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT:-6.00}" in text
    )
    assert (
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS="
        "${ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS:-15.0}" in text
    )
    assert (
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN="
        "${ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN:-0.10}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT="
        "${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT:-8.00}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS="
        "${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS:-15.0}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS="
        "${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS:-0.0}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH="
        "${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH:-0.50}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN="
        "${ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN:-0.10}" in text
    )
    assert (
        "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT="
        "${ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE="
        "${ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE:-0.02}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT="
        "${ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS="
        "${ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS:-8}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT="
        "${ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN="
        "${ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN:-0.02}" in text
    )
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT=${ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT:-8.00}" in text
    assert "ENTRY_HIER_FLAT_LOGIT_MARGIN=${ENTRY_HIER_FLAT_LOGIT_MARGIN:-0.10}" in text
    assert (
        "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE="
        "${ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE:-0.10}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT="
        "${ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT:-8.00}" in text
    )
    assert "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN=${ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN:-0.10}" in text
    assert (
        "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE="
        "${ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE:-0.10}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS="
        "${ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS:-8}" in text
    )
    assert (
        "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT="
        "${ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE="
        "${ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE:-0.10}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT="
        "${ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE="
        "${ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE:-0.10}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS="
        "${ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS:-8}" in text
    )
    assert (
        "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT="
        "${ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE="
        "${ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE:-0.02}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT="
        "${ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN="
        "${ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN:-0.02}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT="
        "${ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT:-4.00}" in text
    )
    assert (
        "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS="
        "${ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS:-8}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT="
        "${ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT:-8.00}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS="
        "${ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS:-15.0}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS="
        "${ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS:-0.0}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH="
        "${ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH:-0.50}" in text
    )
    assert (
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP="
        "${ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP:-4.0}" in text
    )
    assert (
        "ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION="
        "${ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION:-1}" in text
    )
    assert "ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP=${ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP:-0.18}" in text
    assert "ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL=${ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL:-1}" in text
    assert (
        "ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE="
        "${ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE:-1}" in text
    )
    assert "ENTRY_HIER_PUBLIC_TRADE_HEAD=${ENTRY_HIER_PUBLIC_TRADE_HEAD:-1}" in text
    assert (
        "ENTRY_HIER_PUBLIC_TRADE_DIR_MARGIN_BRIDGE="
        "${ENTRY_HIER_PUBLIC_TRADE_DIR_MARGIN_BRIDGE:-1}" in text
    )
    assert (
        "ENTRY_HIER_PUBLIC_TRADE_DIR_MARGIN_BRIDGE_SCALE="
        "${ENTRY_HIER_PUBLIC_TRADE_DIR_MARGIN_BRIDGE_SCALE:-0.50}" in text
    )
    assert (
        "ENTRY_HIER_PUBLIC_TRADE_DIR_MARGIN_BRIDGE_CAP="
        "${ENTRY_HIER_PUBLIC_TRADE_DIR_MARGIN_BRIDGE_CAP:-0.25}" in text
    )
    assert "ENTRY_HIER_PUBLIC_SIDE_HEAD=${ENTRY_HIER_PUBLIC_SIDE_HEAD:-1}" in text
    assert "ENTRY_HIER_CTX_PRIOR_ADAPTER=${ENTRY_HIER_CTX_PRIOR_ADAPTER:-1}" in text
    assert "ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE=${ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE:-0.50}" in text
    assert "ENTRY_HIER_CTX_DIRECTION_CALIBRATION=${ENTRY_HIER_CTX_DIRECTION_CALIBRATION:-1}" in text
    assert (
        "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE="
        "${ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE:-0.50}" in text
    )
    assert (
        "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP="
        "${ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP:-0.35}" in text
    )
    assert "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT=${ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT:-8.00}" in text
    assert (
        "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=${ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE:-0.10}"
        in text
    )
    assert "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS=${ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS:-8}" in text
    assert (
        "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION=${ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION:-0.50}"
        in text
    )
    assert "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR=${ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR:-0.10}" in text
    assert (
        "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=${ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN:-0.10}"
        in text
    )


def test_v10_6yr_rebuild_blocks_stale_mtf_and_runs_xau_pretrain_audit() -> None:
    text = (REPO / "scripts" / "v10_6yr_rebuild_20260626.sh").read_text(encoding="utf-8")

    assert "GX1_PERTF_CLOSED_BAR" in text
    assert "GX1_MTF_CACHE_ALLOW_STALE must stay off" in text
    assert "[MTF_CACHE_GATE_FAIL]" in text
    assert "m5_prebuilt_source mismatch" in text
    assert "m5_prebuilt_source_sha256 mismatch" in text
    assert "feature_names mismatch" in text
    assert "shift_contract mismatch" in text
    assert "cache does not cover TEST_END" in text
    assert "audit_xau_direction_repair_pretrain_v1" in text
    assert "--require-rail-features" in text
    assert "v10_6yr_dataset__HOLD_03B" in text
    assert "xgb_bridge_source is not neutral_uniform_proba" in text
    assert "tape_root is not XAUUSD-only" in text
    assert "smart520 rank reference sha mismatch" in text
    assert 'for split in ("train", "val", "test")' in text


def test_v10_6yr_rebuild_blocks_stale_predataset_artifacts() -> None:
    text = (REPO / "scripts" / "v10_6yr_rebuild_20260626.sh").read_text(encoding="utf-8")

    assert "[CANONICAL_V2_GATE_FAIL]" in text
    assert "canonical_features_v2_no_lookahead_close_time_20260713" in text
    assert "[CANONICAL_V3_GATE_FAIL]" in text
    assert "source_v2_parquet_sha256" in text
    assert "source_v2_no_lookahead" in text
    assert "FULL_PLUS_CTX_v3src.proof.json" in text
    assert "[FULL_PLUS_CTX_GATE_FAIL]" in text


def test_mtf_cache_manifest_records_source_sha() -> None:
    text = (REPO / "gx1/scripts/prebuild_multi_tf_cache_v2.py").read_text(encoding="utf-8")

    assert "m5_prebuilt_source_sha256" in text
    assert "feature_names" in text
    assert "shift_contract" in text
    assert "hashlib.sha256" in text


def test_run_replay_defaults_to_xau_price_data_guard() -> None:
    text = (REPO / "scripts/run_replay.sh").read_text(encoding="utf-8")

    assert 'GX1_REPLAY_INSTRUMENT:-XAUUSD' in text
    assert "XAUUSD replay requires XAU price data" in text


def test_xau_direction_repair_builder_requires_smart_geometry_fields() -> None:
    text = (REPO / "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py").read_text(
        encoding="utf-8"
    )

    assert "XAU_DIRECTION_REPAIR_SIGNAL_FIELDS_MISSING" in text
    assert "XAU_DIRECTION_REPAIR_REQUIRES_INLINE_SEQ_STRUCTURE" in text
    assert "chart.geometry_rising_support_rail_short_trap_pressure" in text
    assert "chart.geometry_falling_resistance_rail_long_trap_pressure" in text
    assert "chart.geometry_channel_position_low_to_high" in text


def test_seq_structure_external_join_is_fail_closed_against_leakage() -> None:
    text = (REPO / "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py").read_text(
        encoding="utf-8"
    )

    assert "SEQ_STRUCTURE_EXPLICIT_SELECTED_FEATURES_REQUIRED" in text
    assert "SEQ_STRUCTURE_DUPLICATE_TIME_ROWS" in text
    assert "SEQ_STRUCTURE_FORBIDDEN_FEATURE_NAMES" in text
    assert '"future"' in text
    assert '"target"' in text
    assert '"pnl"' in text
