from __future__ import annotations

from pathlib import Path

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS,
)
from gx1.features.htf_features import (
    HTF_V4_CACHE_SCHEMA_VERSION,
    HTF_V4_MATRIX_CONTRACT,
    V29_REGISTRY_CONSTANTS_SCHEMA_VERSION,
    V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
)


REPO = Path(__file__).resolve().parents[1]
ACTIVE_SOURCES = (
    "gx1/contracts/registry_hyperparameter_fit_v1.py",
    "gx1/contracts/entry_model_native_train_recipe_v1.py",
    "gx1/features/level_registry_v1.py",
    "gx1/features/trendline_registry_v1.py",
    "gx1/features/htf_features.py",
    "gx1/features/entry_model_native_feature_layers_v1.py",
    "gx1/scripts/build_entry_exit_m1_enriched_frame_v1.py",
    "gx1/scripts/prebuild_multi_tf_cache_v4.py",
    "gx1/scripts/materialize_entry_exit_m1_feature_base_v1.py",
    "scripts/run_seq513_rebuild_chain_v1.sh",
    "scripts/entry_next_edge_control.sh",
)
RETIRED_TOKENS = (
    "LEVEL_REGISTRY_REACTION_WINDOW_BARS",
    "LEVEL_REGISTRY_RETEST_WINDOW_BARS",
    "LEVEL_REGISTRY_AGE_CAP_BARS",
    "LEVEL_REGISTRY_FIT_AGE_CAP_BARS",
    "LEVEL_REGISTRY_TOL_QUANTILE_RECIPE_KEY",
    "TRENDLINE_RETEST_WINDOW_BARS",
    "fit_level_registry_tolerance",
    "fit_trendline_tolerance",
    "level_tol_quantile_q",
    "--level-tol-quantile-q",
    "Jeffreys",
    "LEVEL_ROUND_GRID_",
    "round_50",
    "round_100",
    "LEVEL_REGISTRY_ACTIVE_ANCHOR_CAPACITY_PER_SIDE",
    "active_anchor_capacity",
    "level_tol_atr",
    "tol_level_atr",
    "LEVEL_KIND_PIVOT_CLUSTER",
    "selected_zone_test_count",
    "LEVEL_REGISTRY_DIST_SATURATION_ATR",
    "LEVEL_REGISTRY_COUNT_AGE_CAP",
    "TRENDLINE_COUNT_AGE_CAP_V1",
    "member_pivot_count",
    "selected_recurrence_count",
)


def test_active_registry_sources_have_no_retired_static_operator() -> None:
    for relative in ACTIVE_SOURCES:
        source = (REPO / relative).read_text(encoding="utf-8")
        for token in RETIRED_TOKENS:
            assert token not in source, f"{relative}: stale {token}"


# Identities that were live before the retirement wave. A cache or matrix
# still carrying one of these would let a pre-retirement artifact be reused.
SUPERSEDED_CACHE_IDENTITIES = (
    "htf_v4_disk_cache_manifest_v18",
    "htf_v4_disk_cache_manifest_v19",
)
SUPERSEDED_MATRIX_IDENTITIES = (
    "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V9",
    "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V10",
)


def test_registry_artifact_cache_and_full_mtf_matrix_identity_are_bumped() -> None:
    """The registry retirement must move both reuse keys past their old values.

    The exact version numbers are deliberately NOT restated: they bump again on
    every later surface change and a pinned literal here has gone stale within
    days (rule 13). What must hold is that neither identity is any superseded
    value, and that both are still the identities the cache producer and the
    matrix guards actually compare against.
    """

    assert MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS == ()
    assert V29_REGISTRY_CONSTANTS_SCHEMA_VERSION.endswith("_v6")
    assert V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION.endswith("_v6")
    assert HTF_V4_CACHE_SCHEMA_VERSION not in SUPERSEDED_CACHE_IDENTITIES
    assert HTF_V4_MATRIX_CONTRACT not in SUPERSEDED_MATRIX_IDENTITIES
    assert HTF_V4_CACHE_SCHEMA_VERSION.startswith("htf_v4_disk_cache_manifest_v")
    assert HTF_V4_MATRIX_CONTRACT.startswith(
        "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V"
    )
    # Both identities must remain the one the cache producer imports, so a
    # bump cannot be declared in the owner and ignored at the reuse boundary.
    producer = (
        REPO / "gx1/scripts/prebuild_multi_tf_cache_v4.py"
    ).read_text(encoding="utf-8")
    assert "SCHEMA_VERSION = HTF_V4_CACHE_SCHEMA_VERSION" in producer
    assert "MATRIX_CONTRACT = HTF_V4_MATRIX_CONTRACT" in producer
    # No superseded identity may survive anywhere in the active sources.
    for relative in ACTIVE_SOURCES:
        source = (REPO / relative).read_text(encoding="utf-8")
        for stale in (
            *SUPERSEDED_CACHE_IDENTITIES,
            *SUPERSEDED_MATRIX_IDENTITIES,
        ):
            assert stale not in source, f"{relative}: stale {stale}"
