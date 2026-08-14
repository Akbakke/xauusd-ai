"""Fail-closed signal contract for the model-native XAU entry candidate.

The retired Smart520 surface prepended seven externally derived values to 34
genuine per-bar price-state fields.  Fresh XAU builds filled those seven values
with constants, while the Transformer still interpreted three of them as
direction-anchor probabilities.  This contract removes that dead bridge from
the input surface and makes the Transformer direction logits model-native.

Of the selected 479 specialist fields, all 346 registered causal full-stack
layer outputs are code-owned and mandatory.  Only the remaining 133 positions
are ranking-owned.  The emitted manifest still owns the audited exact order,
while this module owns the immutable base order, mandatory registry identity,
dimensions, forbidden legacy fields, and validation of the combined surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gx1.contracts.entry_structural_aux_label_signal_v1 import (
    structural_aux_label_signal_contract_metadata,
)
from gx1.contracts.entry_pretrain_polarity_signal_v1 import (
    pretrain_polarity_signal_contract_metadata,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
)
from gx1.features.entry_smart_context import ENTRY_SMART_CTX_FEATURE_NAMES
from gx1.features.micro_structure_v1 import (
    MICRO_FEATURE_NAMES_V1,
    SPREAD_DYNAMICS_FEATURE_NAMES_V1,
)
from gx1.features.regime_v4_features import REGIME_V4_FEATURE_NAMES
from gx1.features.swing_structure_v1 import (
    SWING_FEATURE_NAMES_V1,
    SWING_V29_ADDITION_NAMES_V1,
)


# v8: V29 Phase-A stage-2 event families extended the mandatory registry
# (level/trendline registries, swing/momentum/regime events); dims derive.
MODEL_NATIVE_SIGNAL_SCHEMA_VERSION = "entry_model_native_signal_v8"
MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION = (
    "entry_model_native_seq513_split_manifest_v4"
)
MODEL_NATIVE_CONTRACT_MODE = "xau_seq513_model_native_direction_v4"
RETIRED_NEUTRAL_BRIDGE_CONTRACT_MODE = "smart_seq520_candidate"
MODEL_NATIVE_DIRECTION_LOGIT_MODE = "model_native"
# The model-native contract owns these fields directly. No decision authority
# may import its input order from a compatibility surface.
FORBIDDEN_LEGACY_BRIDGE_FIELDS = (
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
)

MODEL_NATIVE_BASE_FIELDS = (
    "_v1_atr14",
    "atr_z",
    "ret_1",
    "ret_5",
    "ret_20",
    "rvol_20",
    "body_pct",
    "wick_asym",
    "ema20_slope",
    "pos_vs_ema200",
    "_v1_pk_sigma20",
    "_v1_ema_diff",
    "_v1_close_ema_slope_3",
    "_v1_clv",
    "_v1_range_z",
    "_v1_kama_slope_30",
    "_v1_tema_slope_20",
    "_v1_bb_squeeze_20_2",
    "_v1_bb_bandwidth_delta_10",
    "_v1_body_share_1",
    "_v1_kurt_r",
    "smc_swing_state",
    "smc_bos_up",
    "smc_bos_down",
    "smc_choch",
    "smc_sweep_up",
    "smc_sweep_down",
    "smc_sweep_size_atr",
    "smc_bars_since_sweep",
    "smc_premium_discount",
    "vol_z_20",
    "vol_ratio_5_20",
    "vol_pct_96",
    "signed_vol_z_20",
)

MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS = (
    "atr_bps",
    "spread_bps",
    "D1_dist_from_ema200_atr",
    "H1_range_compression_ratio",
    "D1_atr_percentile_252",
    "M15_range_compression_ratio",
    # V30 (2026-08-13): the H4 sibling of the H1/M15 compression ratio,
    # emitted by the one native-M5 scalar owner
    # (htf_features.MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4; the ctx_cont
    # relative order must preserve that tuple's order per the single-owner
    # contract test).
    "H4_range_compression_ratio",
)
MODEL_NATIVE_CTX_CONT_MICRO_FIELDS = tuple(MICRO_FEATURE_NAMES_V1)
# V30 package 4 (2026-08-13): the quote/spread-dynamics block.  Declared as its
# own tuple rather than folded into MODEL_NATIVE_CTX_CONT_MICRO_FIELDS because
# the two producers take different sources: the micro five are computed from
# (high, low, close) ARRAYS, these three from the quote FRAME (bid/ask closes
# and extremes).  Every ctx producer emits both blocks at the same rebuild
# boundary (rule 6).  Purpose: abstention / execution-regime evidence, never a
# direction signal — see the producer's own contract comment.
MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS = tuple(
    SPREAD_DYNAMICS_FEATURE_NAMES_V1
)
# V30 Phase-A completion (2026-08-13): the swing owner's own declaration is
# performed here.  ``swing_structure_v1`` declared its V29/V30 additions
# "DECLARED SEPARATELY from SWING_FEATURE_NAMES_V1: the accepted contracts bind
# the 5-name V1 surface; the stage-2 V29 wiring adopts these names into the
# ctx/111-surface contracts together with the V29 rebuild (rule 6: train==serve
# moves at one boundary)."  This is that adoption, taken at the V30 rebuild
# boundary together with the per-TF flip; all four ctx producers switch to
# ``include_v29_additions=True`` in the same change.
MODEL_NATIVE_CTX_CONT_SWING_FIELDS = tuple(
    SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1
)
MODEL_NATIVE_CTX_CONT_SESSION_FIELDS = (
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
)
MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS = (
    MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS
    + MODEL_NATIVE_CTX_CONT_MICRO_FIELDS
    + MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS
    + MODEL_NATIVE_CTX_CONT_SWING_FIELDS
)
MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS = (
    MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS + MODEL_NATIVE_CTX_CONT_SESSION_FIELDS
)

MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS = (
    "_v1h1_ema_diff",
    "_v1h1_atr",
    "_v1h1_rsi14_z",
    "_v1h1_slope3",
    "_v1h1_slope5",
    "_v1h4_ema_diff",
    "_v1h4_atr",
    "_v1h4_rsi14_z",
    "_v1h4_slope3",
    "_v1h4_slope5",
    "d1_atr14_canon_v2",
    "d1_rsi14_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "d1_range_z_20_canon_v2",
    "d1_close_pct_in_20day_range_canon_v2",
    "d1_pct_change_5_canon_v2",
    "m15_rsi14_canon_v2",
    "m15_range_z_20_canon_v2",
    "m15_trend_sign_canon_v2",
    # V30 Phase-A completion (2026-08-13): momentum G3 raw-RSI ctx scalars.
    # Verbatim siblings of m15_rsi14_canon_v2 / d1_rsi14_canon_v2 — one
    # `_rsi(close, 14)` owner, raw 0-100 unit, each TF's own closed bars,
    # projected by the one native-M5 scalar owner
    # (htf_features.MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4, whose tuple order
    # this block must preserve).  The `_v1h{1,4}_rsi14_z` z-scores are kept
    # (design doc §3 momentum G3: "z-fields kept").
    "m5_rsi14_canon_v2",
    "h1_rsi14_canon_v2",
    "h4_rsi14_canon_v2",
)

MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "smc_premium_state",
)

MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS = (
    "is_asia_eu_overlap",
    "is_eu_us_overlap",
    "is_eu_only",
    "is_us_only",
    "atr_ratio_m5_h4",
    "atr_ratio_m15_d1",
    "atr_ratio_h1_d1",
    "atr_ratio_m5_m15",
    "vol_pct_m5_1yr",
    "vol_pct_h1_1yr",
    "dist_to_R1_atr",
    "dist_to_R2_atr",
    "dist_to_S1_atr",
    "dist_to_S2_atr",
    "dist_to_m5_hi_atr",
    "dist_to_m5_lo_atr",
    "dist_to_m15_hi_atr",
    "dist_to_m15_lo_atr",
    "dist_to_h1_hi_atr",
    "dist_to_h1_lo_atr",
    "dist_to_h4_hi_atr",
    "dist_to_h4_lo_atr",
    "dist_to_d1_hi_atr",
    "dist_to_d1_lo_atr",
)

MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS = (
    "dip_confirmed_m5_v3",
    "dip_confirmed_m15_v3",
    "dip_proximity_h1_v3",
    "dip_confirmed_h1_v3",
    "dip_proximity_h4_v3",
    "dip_confirmed_h4_v3",
    "dip_proximity_d1_v3",
    "dip_confirmed_d1_v3",
    "struct_continuation_up_m5_v3",
    "struct_pullback_in_uptrend_m5_v3",
    "struct_continuation_down_m5_v3",
    "struct_bounce_in_downtrend_m5_v3",
    "struct_pullback_depth_m5_v3",
    "struct_continuation_up_m15_v3",
    "struct_pullback_in_uptrend_m15_v3",
    "struct_continuation_down_m15_v3",
    "struct_bounce_in_downtrend_m15_v3",
    "struct_pullback_depth_m15_v3",
    "struct_continuation_up_h1_v3",
    "struct_pullback_in_uptrend_h1_v3",
    "struct_continuation_down_h1_v3",
    "struct_bounce_in_downtrend_h1_v3",
    "struct_pullback_depth_h1_v3",
    "struct_continuation_up_h4_v3",
    "struct_pullback_in_uptrend_h4_v3",
    "struct_continuation_down_h4_v3",
    "struct_bounce_in_downtrend_h4_v3",
    "struct_pullback_depth_h4_v3",
    "struct_continuation_up_d1_v3",
    "struct_pullback_in_uptrend_d1_v3",
    "struct_continuation_down_d1_v3",
    "struct_bounce_in_downtrend_d1_v3",
    "struct_pullback_depth_d1_v3",
    "struct_tf_agree_count_v3",
    "struct_dip_x_uptrend_v3",
    "struct_smc_swing_x_dip_v3",
)

MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS = tuple(ENTRY_SMART_CTX_FEATURE_NAMES)
MODEL_NATIVE_CTX_CONT_REGIME_FIELDS = tuple(REGIME_V4_FEATURE_NAMES)
MODEL_NATIVE_CTX_CONT_FIELDS = (
    MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS
    + MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS
    + MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS
    + MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS
    + MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS
    + MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS
    + MODEL_NATIVE_CTX_CONT_REGIME_FIELDS
)
MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME = {
    name: index for index, name in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS)
}
MODEL_NATIVE_CTX_CAT_FIELDS = (
    "session_id",
    "vol_regime_id",
    "atr_bucket",
    "spread_bucket",
    "H4_trend_sign_cat",
)
MODEL_NATIVE_CTX_CAT_DOMAINS = {
    "session_id": (0, 1, 2, 3),
    "vol_regime_id": (0, 1, 2, 3, 4),
    "atr_bucket": (0, 1, 2, 3, 4),
    "spread_bucket": (0, 1, 2, 3, 4),
    "H4_trend_sign_cat": (0, 1, 2),
}
if tuple(MODEL_NATIVE_CTX_CAT_DOMAINS) != MODEL_NATIVE_CTX_CAT_FIELDS:
    raise RuntimeError("MODEL_NATIVE_CTX_CAT_DOMAIN_ORDER_INVALID")
MODEL_NATIVE_VOL_REGIME_NAMES = (
    "VERY_LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "EXTREME",
)
MODEL_NATIVE_TREND_REGIME_NAMES = (
    "TREND_DOWN",
    "TREND_NEUTRAL",
    "TREND_UP",
)
if len(MODEL_NATIVE_VOL_REGIME_NAMES) != len(
    MODEL_NATIVE_CTX_CAT_DOMAINS["vol_regime_id"]
):
    raise RuntimeError("MODEL_NATIVE_VOL_REGIME_NAME_DOMAIN_MISMATCH")
if len(set(MODEL_NATIVE_VOL_REGIME_NAMES)) != len(
    MODEL_NATIVE_VOL_REGIME_NAMES
):
    raise RuntimeError("MODEL_NATIVE_VOL_REGIME_NAMES_NOT_UNIQUE")
if len(MODEL_NATIVE_TREND_REGIME_NAMES) != len(
    MODEL_NATIVE_CTX_CAT_DOMAINS["H4_trend_sign_cat"]
):
    raise RuntimeError("MODEL_NATIVE_TREND_REGIME_NAME_DOMAIN_MISMATCH")
MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME = {
    name: index for index, name in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS)
}
MODEL_NATIVE_CTX_CAT_MIN_MAX = {
    name: (min(domain), max(domain))
    for name, domain in MODEL_NATIVE_CTX_CAT_DOMAINS.items()
}

# Dimensions are DERIVED from the declared field owners (rule 13).  The
# TRAIN-ranked remainder is held constant at 133 and re-ranked TRAIN-only on
# the V29 substrate (design doc §5.1); the mandatory causal count grew with
# the V29 Phase-A stage-2 event families, so the selected and total signal
# dims follow from the registry, never from a restated literal.
MODEL_NATIVE_BASE_SIGNAL_DIM = len(MODEL_NATIVE_BASE_FIELDS)
MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT = 133
MODEL_NATIVE_SELECTED_FEATURE_COUNT = (
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
    + MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
)
MODEL_NATIVE_SIGNAL_DIM = (
    MODEL_NATIVE_BASE_SIGNAL_DIM + MODEL_NATIVE_SELECTED_FEATURE_COUNT
)
MODEL_NATIVE_SEQ_LEN = 96
# 158 = 143 (V30 package 1: + H4_range_compression_ratio) + the V30 package-2
# Phase-A completion: 9 swing V29 event fields adopted into
# MODEL_NATIVE_CTX_CONT_SWING_FIELDS + 3 momentum-G3 raw-RSI canon scalars
# (-> 155) + the V30 package-4 quote/spread-dynamics block
# (MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS, 3 fields, 2026-08-13).
# 164 = 158 + the V30 package-8A emission-only swing additions (2026-08-13):
# the two MISSING run counters, the two "last confirmed pivot not yet closed
# through" intact flags and the two normalized siblings of the raw V1 swing
# ages — all six inside MODEL_NATIVE_CTX_CONT_SWING_FIELDS, which reads
# swing_structure_v1's own tuple.  The field lists above are the owner, this
# literal is the cross-check.
MODEL_NATIVE_CTX_CONT_DIM = 164
MODEL_NATIVE_CTX_CAT_DIM = 5

if len(MODEL_NATIVE_BASE_FIELDS) != MODEL_NATIVE_BASE_SIGNAL_DIM:
    raise RuntimeError(
        "MODEL_NATIVE_BASE_SIGNAL_DIM_MISMATCH: "
        f"fields={len(MODEL_NATIVE_BASE_FIELDS)} expected={MODEL_NATIVE_BASE_SIGNAL_DIM}"
    )
if len(MODEL_NATIVE_CTX_CONT_FIELDS) != MODEL_NATIVE_CTX_CONT_DIM:
    raise RuntimeError(
        "MODEL_NATIVE_CTX_CONT_DIM_MISMATCH: "
        f"fields={len(MODEL_NATIVE_CTX_CONT_FIELDS)} expected={MODEL_NATIVE_CTX_CONT_DIM}"
    )
if len(MODEL_NATIVE_CTX_CAT_FIELDS) != MODEL_NATIVE_CTX_CAT_DIM:
    raise RuntimeError(
        "MODEL_NATIVE_CTX_CAT_DIM_MISMATCH: "
        f"fields={len(MODEL_NATIVE_CTX_CAT_FIELDS)} expected={MODEL_NATIVE_CTX_CAT_DIM}"
    )
if len(set(MODEL_NATIVE_CTX_CONT_FIELDS)) != len(MODEL_NATIVE_CTX_CONT_FIELDS):
    raise RuntimeError("MODEL_NATIVE_CTX_CONT_FIELDS_DUPLICATE")
if len(set(MODEL_NATIVE_CTX_CAT_FIELDS)) != len(MODEL_NATIVE_CTX_CAT_FIELDS):
    raise RuntimeError("MODEL_NATIVE_CTX_CAT_FIELDS_DUPLICATE")
if "trend_regime_id" in MODEL_NATIVE_CTX_CAT_FIELDS:
    raise RuntimeError("MODEL_NATIVE_CTX_CAT_FIELDS_CONTAIN_RETIRED_TREND_BUCKET")
if set(MODEL_NATIVE_BASE_FIELDS) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_BASE_FIELDS_CONTAIN_FORBIDDEN_BRIDGE_FIELDS")
if MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT != 133:
    raise RuntimeError(
        "MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT_MISMATCH: "
        f"observed={MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT} expected=133"
    )
if set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS) & set(MODEL_NATIVE_BASE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FIELDS_OVERLAP_BASE_FIELDS")
if set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FIELDS_CONTAIN_FORBIDDEN_BRIDGE_FIELDS")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


MODEL_NATIVE_BASE_FIELDS_SHA256 = _sha256_json(MODEL_NATIVE_BASE_FIELDS)
MODEL_NATIVE_CTX_CONT_FIELDS_SHA256 = _sha256_json(MODEL_NATIVE_CTX_CONT_FIELDS)
MODEL_NATIVE_CTX_CAT_FIELDS_SHA256 = _sha256_json(MODEL_NATIVE_CTX_CAT_FIELDS)
MODEL_NATIVE_CONTEXT_SCHEMA_VERSION = "entry_model_native_context_v2"
MODEL_NATIVE_CONTEXT_TAG = (
    f"CTX{MODEL_NATIVE_CTX_CONT_DIM}CAT{MODEL_NATIVE_CTX_CAT_DIM}"
)


def model_native_context_contract_metadata() -> dict[str, Any]:
    """Return the exact continuous/categorical Entry context contract.

    Dims derive from the declared field tuples (158 continuous / 5 categorical
    after the V30 wave: package 1 added ``H4_range_compression_ratio``,
    package 2 adopted the swing V29 event fields plus the three
    momentum-G3 raw-RSI canon scalars, and package 4 added the three
    quote/spread-dynamics fields).
    """

    return {
        "schema_version": MODEL_NATIVE_CONTEXT_SCHEMA_VERSION,
        "tag": MODEL_NATIVE_CONTEXT_TAG,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "ctx_cont_fields_sha256": MODEL_NATIVE_CTX_CONT_FIELDS_SHA256,
        "ctx_cat_fields_sha256": MODEL_NATIVE_CTX_CAT_FIELDS_SHA256,
        "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        "ctx_cont_source_prefix_names": list(
            MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS
        ),
        "ctx_cont_micro_features": list(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS),
        "ctx_cont_spread_dynamics_features": list(
            MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS
        ),
        "ctx_cont_swing_features": list(MODEL_NATIVE_CTX_CONT_SWING_FIELDS),
        "ctx_cont_session_features": list(MODEL_NATIVE_CTX_CONT_SESSION_FIELDS),
    }


MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION = (
    "entry_model_native_mandatory_full_stack_v6"
)
MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256 = _sha256_json(
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
)


def model_native_mandatory_full_stack_metadata() -> dict[str, Any]:
    """Return the immutable exact family/name registry embedded in artifacts."""

    return {
        "schema_version": MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION,
        "ordered_family_fields_sha256": MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256,
        "family_count": len(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES),
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "family_order": [
            family for family, _features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        ],
        "family_feature_counts": {
            family: len(features)
            for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        },
        "family_features": {
            family: list(features)
            for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        },
        "ordered_mandatory_fields": list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS),
    }


MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT = (
    structural_aux_label_signal_contract_metadata(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
    )
)

MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT = (
    pretrain_polarity_signal_contract_metadata(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
    )
)


MODEL_NATIVE_STATIC_CONTRACT_SHA256 = _sha256_json(
    {
        "schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "base_fields": MODEL_NATIVE_BASE_FIELDS,
        "base_fields_sha256": MODEL_NATIVE_BASE_FIELDS_SHA256,
        "forbidden_legacy_bridge_fields": FORBIDDEN_LEGACY_BRIDGE_FIELDS,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
        "structural_aux_label_signal_contract": (
            MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT
        ),
        "pretrain_polarity_signal_contract": (
            MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT
        ),
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "context_contract": model_native_context_contract_metadata(),
    }
)


def ordered_model_native_signal_fields(
    selected_fields: Sequence[str],
) -> tuple[str, ...]:
    """Return the exact 513-field surface or fail on any soft compatibility."""

    selected = tuple(str(name).strip() for name in selected_fields)
    failures: list[str] = []
    if len(selected) != MODEL_NATIVE_SELECTED_FEATURE_COUNT:
        failures.append(
            "selected_feature_count="
            f"{len(selected)} expected={MODEL_NATIVE_SELECTED_FEATURE_COUNT}"
        )
    blank = [index for index, name in enumerate(selected) if not name]
    if blank:
        failures.append(f"blank_selected_field_indices={blank[:10]}")
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        failures.append(f"duplicate_selected_fields={duplicates[:20]}")
    forbidden = sorted(set(selected) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    if forbidden:
        failures.append(f"forbidden_legacy_bridge_fields={forbidden}")
    base_overlap = sorted(set(selected) & set(MODEL_NATIVE_BASE_FIELDS))
    if base_overlap:
        failures.append(f"selected_fields_duplicate_base_fields={base_overlap[:20]}")
    selected_set = set(selected)
    missing_mandatory = [
        name
        for name in MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
        if name not in selected_set
    ]
    if missing_mandatory:
        failures.append(
            "missing_mandatory_full_stack_fields="
            f"{missing_mandatory[:20]} total={len(missing_mandatory)}"
        )
    else:
        # The mandatory prefix must exactly equal the immutable causal-layer
        # registry order; membership alone is not the documented contract.
        mandatory_prefix = tuple(
            selected[:MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT]
        )
        if mandatory_prefix != tuple(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS):
            wrong_positions = [
                index
                for index, (got, want) in enumerate(
                    zip(mandatory_prefix, MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
                )
                if got != want
            ]
            failures.append(
                "mandatory_registry_prefix_order_violation="
                f"positions{wrong_positions[:10]} total={len(wrong_positions)}"
            )
    fields = MODEL_NATIVE_BASE_FIELDS + selected
    if len(fields) != MODEL_NATIVE_SIGNAL_DIM:
        failures.append(f"signal_dim={len(fields)} expected={MODEL_NATIVE_SIGNAL_DIM}")
    if failures:
        raise RuntimeError(
            "MODEL_NATIVE_SIGNAL_FIELDS_INVALID: " + " | ".join(failures)
        )
    return fields


def model_native_signal_contract_metadata(
    selected_fields: Sequence[str],
) -> dict[str, Any]:
    selected = tuple(str(name).strip() for name in selected_fields)
    fields = ordered_model_native_signal_fields(selected)
    return {
        "schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "static_contract_sha256": MODEL_NATIVE_STATIC_CONTRACT_SHA256,
        "ordered_fields_sha256": _sha256_json(fields),
        "base_fields": list(MODEL_NATIVE_BASE_FIELDS),
        "selected_fields": list(selected),
        "fields": list(fields),
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
        "structural_aux_label_signal_contract": (
            MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT
        ),
        "pretrain_polarity_signal_contract": (
            MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT
        ),
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        "ctx_cont_fields_sha256": MODEL_NATIVE_CTX_CONT_FIELDS_SHA256,
        "ctx_cat_fields_sha256": MODEL_NATIVE_CTX_CAT_FIELDS_SHA256,
        "forbidden_legacy_bridge_fields": list(FORBIDDEN_LEGACY_BRIDGE_FIELDS),
        "bridge_dim": 0,
        "bridge_source": None,
        "anchor_source": None,
    }


def model_native_signal_contract_failures(contract: Mapping[str, Any]) -> list[str]:
    """Validate a serialized dataset/bundle contract without filling defaults."""

    failures: list[str] = []
    if not isinstance(contract, Mapping) or not contract:
        return ["model_native_signal_contract missing"]

    exact_scalars = {
        "schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "static_contract_sha256": MODEL_NATIVE_STATIC_CONTRACT_SHA256,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "ctx_cont_fields_sha256": MODEL_NATIVE_CTX_CONT_FIELDS_SHA256,
        "ctx_cat_fields_sha256": MODEL_NATIVE_CTX_CAT_FIELDS_SHA256,
        "bridge_dim": 0,
    }
    for key, expected in exact_scalars.items():
        if contract.get(key) != expected:
            failures.append(f"{key}={contract.get(key)!r} expected={expected!r}")

    base_fields = tuple(str(value) for value in (contract.get("base_fields") or ()))
    selected_fields = tuple(
        str(value) for value in (contract.get("selected_fields") or ())
    )
    fields = tuple(str(value) for value in (contract.get("fields") or ()))
    forbidden_declared = tuple(
        str(value) for value in (contract.get("forbidden_legacy_bridge_fields") or ())
    )
    if base_fields != MODEL_NATIVE_BASE_FIELDS:
        failures.append("base_fields order mismatch")
    ctx_cont_fields = tuple(
        str(value) for value in (contract.get("ctx_cont_names") or ())
    )
    ctx_cat_fields = tuple(
        str(value) for value in (contract.get("ctx_cat_names") or ())
    )
    if ctx_cont_fields != MODEL_NATIVE_CTX_CONT_FIELDS:
        failures.append("ctx_cont_names order mismatch")
    if ctx_cat_fields != MODEL_NATIVE_CTX_CAT_FIELDS:
        failures.append("ctx_cat_names order mismatch")
    if forbidden_declared != FORBIDDEN_LEGACY_BRIDGE_FIELDS:
        failures.append("forbidden_legacy_bridge_fields order mismatch")
    mandatory_declared = contract.get("mandatory_full_stack")
    mandatory_expected = model_native_mandatory_full_stack_metadata()
    if not isinstance(mandatory_declared, Mapping):
        failures.append("mandatory_full_stack missing")
    elif dict(mandatory_declared) != mandatory_expected:
        failures.append("mandatory_full_stack metadata mismatch")
    aux_declared = contract.get("structural_aux_label_signal_contract")
    if not isinstance(aux_declared, Mapping):
        failures.append("structural_aux_label_signal_contract missing")
    elif dict(aux_declared) != MODEL_NATIVE_STRUCTURAL_AUX_LABEL_SIGNAL_CONTRACT:
        failures.append("structural_aux_label_signal_contract metadata mismatch")
    polarity_declared = contract.get("pretrain_polarity_signal_contract")
    if not isinstance(polarity_declared, Mapping):
        failures.append("pretrain_polarity_signal_contract missing")
    elif dict(polarity_declared) != MODEL_NATIVE_PRETRAIN_POLARITY_SIGNAL_CONTRACT:
        failures.append("pretrain_polarity_signal_contract metadata mismatch")
    try:
        expected_fields = ordered_model_native_signal_fields(selected_fields)
    except RuntimeError as exc:
        failures.append(str(exc))
        expected_fields = ()
    if fields != expected_fields:
        failures.append("fields order mismatch")
    forbidden_present = sorted(set(fields) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    if forbidden_present:
        failures.append(
            f"fields contain forbidden legacy bridge inputs: {forbidden_present}"
        )
    expected_hash = _sha256_json(fields)
    if str(contract.get("ordered_fields_sha256") or "") != expected_hash:
        failures.append(
            "ordered_fields_sha256 mismatch: "
            f"declared={contract.get('ordered_fields_sha256')!r} actual={expected_hash}"
        )
    if contract.get("bridge_source") is not None:
        failures.append(
            f"bridge_source must be null, got {contract.get('bridge_source')!r}"
        )
    if contract.get("anchor_source") is not None:
        failures.append(
            f"anchor_source must be null, got {contract.get('anchor_source')!r}"
        )
    return failures


def require_model_native_signal_contract(
    contract: Mapping[str, Any], *, context: str
) -> None:
    failures = model_native_signal_contract_failures(contract)
    if failures:
        raise RuntimeError(
            f"[{context}_MODEL_NATIVE_SIGNAL_CONTRACT_INVALID] " + " | ".join(failures)
        )


def require_model_native_manifest(
    manifest: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Validate the feature-selection manifest and return its exact contract."""

    mode = str(manifest.get("manifest_variant") or "").strip()
    if mode == RETIRED_NEUTRAL_BRIDGE_CONTRACT_MODE:
        raise RuntimeError(
            f"[{context}_RETIRED_SMART520_CONTRACT] {mode} contains the retired "
            "seven-field external bridge; materialize a fresh "
            f"{MODEL_NATIVE_CONTRACT_MODE} manifest"
        )
    if mode != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            f"[{context}_CONTRACT_MODE_INVALID] manifest_variant={mode!r} "
            f"expected={MODEL_NATIVE_CONTRACT_MODE!r}"
        )
    if (
        int(manifest.get("base_signal_feature_count") or -1)
        != MODEL_NATIVE_BASE_SIGNAL_DIM
    ):
        raise RuntimeError(
            f"[{context}_BASE_SIGNAL_DIM_INVALID] got={manifest.get('base_signal_feature_count')!r} "
            f"expected={MODEL_NATIVE_BASE_SIGNAL_DIM}"
        )
    if int(manifest.get("expected_seq_snap_width") or -1) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            f"[{context}_SIGNAL_DIM_INVALID] got={manifest.get('expected_seq_snap_width')!r} "
            f"expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    selected = manifest.get("selected_features")
    if not isinstance(selected, list):
        raise RuntimeError(f"[{context}_SELECTED_FIELDS_MISSING]")
    declared_mandatory = manifest.get("mandatory_full_stack")
    expected_mandatory = model_native_mandatory_full_stack_metadata()
    if not isinstance(declared_mandatory, Mapping):
        raise RuntimeError(f"[{context}_MANDATORY_FULL_STACK_METADATA_MISSING]")
    if dict(declared_mandatory) != expected_mandatory:
        raise RuntimeError(f"[{context}_MANDATORY_FULL_STACK_METADATA_STALE]")
    contract = model_native_signal_contract_metadata(selected)
    declared = manifest.get("model_native_signal_contract")
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"[{context}_MODEL_NATIVE_SIGNAL_CONTRACT_MISSING]")
    require_model_native_signal_contract(declared, context=context)
    if dict(declared) != contract:
        raise RuntimeError(f"[{context}_MODEL_NATIVE_SIGNAL_CONTRACT_NOT_EXACT]")
    return contract
