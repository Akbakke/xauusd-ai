"""Fail-closed signal contract for the model-native XAU entry candidate.

The retired Smart520 surface prepended seven externally derived values to 34
genuine per-bar price-state fields.  Fresh XAU builds filled those seven values
with constants, while the Transformer still interpreted three of them as
direction-anchor probabilities.  This contract removes that dead bridge from
the input surface and makes the Transformer direction logits model-native.

The selected specialist surface contains every code-owned causal primitive and
continuous-context candidate. Handwritten volatility, trend, momentum, structure,
SMC-quality, support/resistance, foundation and session/regime scorebooks are
not part of the surface:
their exact raw evidence is already routed
to the shared model, which learns the cross-feature/MTF interaction. The emitted manifest owns exact
order, while this module owns base order, mandatory registry identity,
dimensions, forbidden legacy fields, and combined-surface validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gx1.features.basic_v1 import (
    BASIC_V1_FEATURES_SHA256,
    BASIC_V1_FORMULA_SHA256,
    BASIC_V1_SCHEMA_VERSION,
)
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
    price_derived_contract_metadata,
)
from gx1.features.entry_candle_primitives_v1 import (
    candle_primitive_contract_metadata,
)
from gx1.features.entry_foundation_structure_v1 import (
    foundation_structure_contract_metadata,
)
from gx1.features.micro_structure_v1 import (
    MICRO_FEATURE_NAMES_V1,
    SPREAD_DYNAMICS_FEATURE_NAMES_V1,
    micro_structure_contract_metadata,
)
from gx1.features.htf_features import MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES
from gx1.features.swing_structure_v1 import (
    SWING_FEATURE_NAMES_V1,
    SWING_V29_ADDITION_NAMES_V1,
    swing_structure_contract_metadata,
)
from gx1.features.smc_v1 import smc_primitive_contract_metadata


# v14: retire the remaining 15 chart votes and 53 fixed-threshold candle
# patterns.  Eleven raw candle primitives replace them.  Three base aliases
# (wick_asym, shifted CLV and shifted body share) are removed because the raw
# current-bar sequence contains their complete source information.
# v15: retire five handwritten regime votes/threshold aliases while preserving
# the raw per-TF regime class, trend age, EMA-stack evidence and D1 distance.
# v16: retire signed_vol_z_20.  It hand-composed direction with activity even
# though the sequence already carries raw ret_1 and vol_z_20 for learned
# interaction.  The three unsigned tick-count primitives remain code-owned.
# v17: replace the eleven provisional candle measurements with the exact
# 26-field causal geometry/relation/carry owner. Its ordered names, version
# and hash are embedded below so a v16 artifact cannot silently reinterpret
# the wider local candle surface.
# v18: adopt the five native-clock TRAIN-fitted volatility-squeeze state
# primitives on both local and MTF surfaces. Their six-clock manifest is an
# external immutable input; older signal/split identities cannot omit it.
# v19: retire the fixed Spearman top-k. Every code-owned context
# candidate is jointly available; TRAIN liveness is evidence, never a silent
# dataset-specific removal authority.
# v20: retire two operator-declared XAUUSD round-grid distances. Neither 50 nor
# 100 USD was owned by a TRAIN-fit artifact. The registry now exposes immutable
# per-pivot anchors and explicitly named TRAIN-fitted recurrence evidence;
# learned lifetime expires event/slot eligibility without deleting identity.
# v21: replace every registry cap/sentinel with raw measurements plus explicit
# current-slot presence on both local and per-TF surfaces.
# v22: exact no-alias names for four basic-v1 formulas that were previously
# mislabeled as slopes or omitted the actual change horizon/normalization.
# v23: retire 36 hand-composed dip/structure fields and 19 smart-context
# score fields. Their raw local/MTF momentum, slope, level, distance and SMC
# sources remain model inputs; learned layers own every interaction.
# v24: raw technical primitives replace partial VWAP warmup, ewm-first EMA,
# ATR floors and clipped ADX/momentum/distance outputs. The local price owner
# metadata below binds the exact shared formula version and hashes.
# v25: retire fourteen operator-composed context values whose raw sources are
# already available: four session votes, four ATR ratios, two trailing
# volatility percentiles and four HTF compression/percentile scalars.  Only
# raw context and distance measurements remain; learned layers own relations.
# v26: replace SMC sentinel/normalized/composite aliases with raw sweep age and
# raw four-pivot-envelope position on every native clock.
# v27: remove the bar-zero swing pseudo-pivot, partial ATR, age/count caps and
# normalized aliases. Raw swing and foundation event memory is uncapped and
# honestly unavailable before the first event; current-structure presence
# masks remain where they describe an actually current object.
# v28: make zero-range microstructure explicit. The range fraction is observed
# only when high > low; a separate binary mask distinguishes its storage zero
# from an actual close-at-high observation.
# v29: replace every capped/log age with raw native-clock state/event age,
# retire global ever-seen fields and the saturated SMC-envelope availability
# mask, and bind the narrower registry/MTF surfaces.
# v30: retire deterministic regime enums, class-flip aliases and volatility /
# spread buckets. Preserve raw native-clock trend ages and D1 distance change;
# replace the lossy M15/H4 signs with their exact continuous pre-sign sources.
# v31: retire the trendline-registry presence masks
# chart.geomline_{above,below}_active. Their owner emitted them from the same
# branch as geomline_{above,below}_active_count, so each was exactly the
# ">= 1" indicator of the count beside it — no evidence is lost, and the
# guaranteed exact_duplicate on any lane whose per-side ACTIVE population never
# exceeds 1 is gone. The graded counts stay.
# v32: retire candle.raw_zero_range_flag from the mandatory candle family (and
# mtf_candle_raw_zero_range_flag from every per-TF lane). It is constant 0.0
# post-warmup on H4 and D1 — a market fact about gold, not dead wiring — so it
# can never reach a liveness verdict there, and a declared-constant exemption
# only moves the failure to [ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]. Rule 4
# holds with room to spare: the flag is an EXACT algebraic function of three
# shares that stay on the surface, since those three partition the bar range,
# so `high == low` iff body_signed_range, upper_wick_share and
# lower_wick_share are all zero. See the CANDLE_PRIMITIVE_FEATURE_VERSION note
# in gx1.features.entry_candle_primitives_v1 for the proof and the counts.
MODEL_NATIVE_SIGNAL_SCHEMA_VERSION = "entry_model_native_signal_v32"
MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION = (
    "entry_model_native_seq513_split_manifest_v19"
)
MODEL_NATIVE_CONTRACT_MODE = "xau_seq513_model_native_direction_v19"
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
RETIRED_MODEL_NATIVE_SIGNAL_FIELDS = (
    "signed_vol_z_20",
    "smc_sweep_size_atr",
    "smc_bars_since_sweep",
    "smc_premium_discount",
    "smc_bars_since_sweep_norm",
)
RETIRED_STATIC_REGIME_BUCKET_FIELDS = (
    "session_tradable",
    "m15_trend_sign_canon_v2",
    "vol_regime_id",
    "atr_bucket",
    "spread_bucket",
    "H4_trend_sign_cat",
    *(f"{tf}_regime_class_id_v2" for tf in MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES),
    "d1_regime_changed_flag_v3",
    "d1_regime_flip_age_bars_v4",
    *(f"{tf}_regime_changed_flag_v3" for tf in ("m5", "m15", "h1", "h4")),
    *(f"{tf}_regime_flip_age_bars" for tf in ("m5", "m15", "h1", "h4")),
)

MODEL_NATIVE_BASE_FIELDS = (
    "_v1_atr14",
    "atr_z",
    "ret_1",
    "ret_5",
    "ret_20",
    "rvol_20",
    "body_pct",
    "ema20_slope",
    "pos_vs_ema200",
    "_v1_pk_sigma20",
    "_v1_ema_diff",
    "_v1_ema3_ema6_spread_frac",
    "_v1_range_z",
    "_v1_kama30_change_5_atr",
    "_v1_tema20_change_3_atr",
    "_v1_bb_squeeze_20_2",
    "_v1_bb10_bandwidth_change_3",
    "_v1_kurt_r",
    "smc_swing_state",
    "smc_bos_up",
    "smc_bos_down",
    "smc_choch",
    "smc_sweep_up",
    "smc_sweep_down",
    "smc_sweep_event_age_bars",
    "smc_pivot_envelope_position",
    "vol_z_20",
    "vol_ratio_5_20",
    "vol_pct_96",
)

MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS = (
    "atr_bps",
    "spread_bps",
    "D1_dist_from_ema200_atr",
)
MODEL_NATIVE_CTX_CONT_MICRO_FIELDS = tuple(MICRO_FEATURE_NAMES_V1)
# V30 package 4 (2026-08-13): the quote/spread-dynamics block.  Declared as its
# own tuple rather than folded into MODEL_NATIVE_CTX_CONT_MICRO_FIELDS because
# the two producers take different sources: the six micro fields are computed from
# (high, low, close) ARRAYS, these three from the quote FRAME (bid/ask closes
# and extremes).  Every ctx producer emits both blocks at the same rebuild
# boundary (rule 6).  Purpose: abstention / execution-regime evidence, never a
# direction signal — see the producer's own contract comment.
MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS = tuple(
    SPREAD_DYNAMICS_FEATURE_NAMES_V1
)
# The complete swing-v2 owner is present in local context and on every MTF
# clock. Prefix-unavailable pivot measurements are trimmed, while persistent
# event memory uses raw ages after its honest full-history event floor.
MODEL_NATIVE_CTX_CONT_SWING_FIELDS = tuple(
    SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1
)
MODEL_NATIVE_CTX_CONT_SESSION_FIELDS = (
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
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
    "m15_ema5_20_spread_atr_canon_v2",
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
    "h4_mid_ema50_dist_atr_canon_v2",
)

MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS = (
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

RETIRED_HANDCRAFTED_CTX_CONT_FIELDS = (
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
    "smc_choch_recent_tau12",
    "smc_choch_recent_tau24",
    "smc_bos_pressure_last12",
    "smc_bos_pressure_last48",
    "smc_sweep_bull_pressure_last12",
    "smc_sweep_bull_pressure_last48",
    "smc_sweep_size_recent_tau12",
    "smc_sweep_recency_tau24",
    "smc_premium_extreme_snap",
    "sr_nearest_pivot_abs_atr",
    "sr_support_proximity_exp",
    "sr_resistance_proximity_exp",
    "sr_support_minus_resistance_prox",
    "liquidity_hi_nearest_abs_atr",
    "liquidity_lo_nearest_abs_atr",
    "liquidity_lo_minus_hi_prox",
    "dip_confirmed_mean_5tf",
    "dip_confirmed_max_5tf",
    "dip_proximity_mean_h1h4d1",
)
if len(RETIRED_HANDCRAFTED_CTX_CONT_FIELDS) != 55 or len(
    set(RETIRED_HANDCRAFTED_CTX_CONT_FIELDS)
) != len(RETIRED_HANDCRAFTED_CTX_CONT_FIELDS):
    raise RuntimeError("RETIRED_HANDCRAFTED_CTX_CONT_FIELDS_INVALID")
RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS = (
    "is_asia_eu_overlap",
    "is_eu_us_overlap",
    "is_eu_only",
    "is_us_only",
    "atr_ratio_m5_m15",
    "atr_ratio_m5_h4",
    "atr_ratio_m15_d1",
    "atr_ratio_h1_d1",
    "vol_pct_m5_1yr",
    "vol_pct_h1_1yr",
    "H1_range_compression_ratio",
    "M15_range_compression_ratio",
    "H4_range_compression_ratio",
    "D1_atr_percentile_252",
)
if len(RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS) != 14 or len(
    set(RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS)
) != len(RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS):
    raise RuntimeError("RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS_INVALID")
RETIRED_SMC_CTX_COMPOSITE_FIELDS = ("smc_premium_state",)
MODEL_NATIVE_CTX_CONT_REGIME_FIELDS = (
    *(f"{tf}_trend_state_age_bars_v2" for tf in MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES),
    "d1_dist_change_1bar_atr_v4",
)
MODEL_NATIVE_CTX_CONT_FIELDS = (
    MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS
    + MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS
    + MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS
    + MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS
    + MODEL_NATIVE_CTX_CONT_REGIME_FIELDS
)
MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME = {
    name: index for index, name in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS)
}
MODEL_NATIVE_CTX_CAT_FIELDS = (
    "session_id",
)
MODEL_NATIVE_CTX_CAT_DOMAINS = {
    "session_id": (0, 1, 2, 3),
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
MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME = {
    name: index for index, name in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS)
}
MODEL_NATIVE_CTX_CAT_MIN_MAX = {
    name: (min(domain), max(domain))
    for name, domain in MODEL_NATIVE_CTX_CAT_DOMAINS.items()
}

# Dimensions are DERIVED from declared field owners (rule 13). Every
# continuous-context field not already present in the mandatory causal prefix
# is exposed to the local sequence/snapshot path. No fixed count, score cutoff
# or family quota can remove one.
MODEL_NATIVE_BASE_SIGNAL_DIM = len(MODEL_NATIVE_BASE_FIELDS)
MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS = tuple(
    f"ctx_cont.{name}"
    for name in MODEL_NATIVE_CTX_CONT_FIELDS
    if f"ctx_cont.{name}" not in set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
)
MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT = len(
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
)
MODEL_NATIVE_SELECTED_FEATURE_COUNT = (
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
    + MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
)
MODEL_NATIVE_SIGNAL_DIM = (
    MODEL_NATIVE_BASE_SIGNAL_DIM + MODEL_NATIVE_SELECTED_FEATURE_COUNT
)
MODEL_NATIVE_SEQ_LEN = 96
MODEL_NATIVE_CTX_CONT_DIM = len(MODEL_NATIVE_CTX_CONT_FIELDS)
MODEL_NATIVE_CTX_CAT_DIM = len(MODEL_NATIVE_CTX_CAT_FIELDS)

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
if set(MODEL_NATIVE_CTX_CONT_FIELDS) & set(RETIRED_HANDCRAFTED_CTX_CONT_FIELDS):
    raise RuntimeError("MODEL_NATIVE_CTX_CONT_FIELDS_CONTAIN_RETIRED_HANDCRAFTED_FIELDS")
if set(MODEL_NATIVE_CTX_CONT_FIELDS) & set(
    RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS
):
    raise RuntimeError("MODEL_NATIVE_CTX_CONT_FIELDS_CONTAIN_RETIRED_OPERATOR_COMPOSITES")
if set(MODEL_NATIVE_CTX_CONT_FIELDS) & set(RETIRED_SMC_CTX_COMPOSITE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_CTX_CONT_FIELDS_CONTAIN_RETIRED_SMC_COMPOSITES")
if set(MODEL_NATIVE_CTX_CONT_FIELDS) & set(RETIRED_STATIC_REGIME_BUCKET_FIELDS):
    raise RuntimeError("MODEL_NATIVE_CTX_CONT_FIELDS_CONTAIN_RETIRED_STATIC_REGIME_BUCKETS")
if len(set(MODEL_NATIVE_CTX_CAT_FIELDS)) != len(MODEL_NATIVE_CTX_CAT_FIELDS):
    raise RuntimeError("MODEL_NATIVE_CTX_CAT_FIELDS_DUPLICATE")
if "trend_regime_id" in MODEL_NATIVE_CTX_CAT_FIELDS:
    raise RuntimeError("MODEL_NATIVE_CTX_CAT_FIELDS_CONTAIN_RETIRED_TREND_BUCKET")
if set(MODEL_NATIVE_CTX_CAT_FIELDS) & set(RETIRED_STATIC_REGIME_BUCKET_FIELDS):
    raise RuntimeError("MODEL_NATIVE_CTX_CAT_FIELDS_CONTAIN_RETIRED_STATIC_REGIME_BUCKETS")
if set(MODEL_NATIVE_BASE_FIELDS) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_BASE_FIELDS_CONTAIN_FORBIDDEN_BRIDGE_FIELDS")
if set(MODEL_NATIVE_BASE_FIELDS) & set(RETIRED_MODEL_NATIVE_SIGNAL_FIELDS):
    raise RuntimeError("MODEL_NATIVE_BASE_FIELDS_CONTAIN_RETIRED_SIGNAL_FIELDS")
if (
    not MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
    or len(set(MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS))
    != MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
):
    raise RuntimeError("MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS_INVALID")
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
MODEL_NATIVE_BASIC_V1_CONTRACT = {
    "schema_version": BASIC_V1_SCHEMA_VERSION,
    "features_sha256": BASIC_V1_FEATURES_SHA256,
    "formula_sha256": BASIC_V1_FORMULA_SHA256,
}
MODEL_NATIVE_CTX_CONT_FIELDS_SHA256 = _sha256_json(MODEL_NATIVE_CTX_CONT_FIELDS)
MODEL_NATIVE_CTX_CAT_FIELDS_SHA256 = _sha256_json(MODEL_NATIVE_CTX_CAT_FIELDS)
MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS_SHA256 = _sha256_json(
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
)
MODEL_NATIVE_CONTEXT_SCHEMA_VERSION = "entry_model_native_context_v10"
MODEL_NATIVE_CONTEXT_TAG = (
    f"CTX{MODEL_NATIVE_CTX_CONT_DIM}CAT{MODEL_NATIVE_CTX_CAT_DIM}"
)


def model_native_context_contract_metadata() -> dict[str, Any]:
    """Return the exact continuous/categorical Entry context contract.

    Dims derive from the declared field tuples. The raw per-TF evidence remains
    available after the hand-composed dip/structure and smart-context score
    fields are retired.
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
        "micro_structure_owner": micro_structure_contract_metadata(),
        "ctx_cont_spread_dynamics_features": list(
            MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS
        ),
        "ctx_cont_swing_features": list(MODEL_NATIVE_CTX_CONT_SWING_FIELDS),
        "ctx_cont_session_features": list(MODEL_NATIVE_CTX_CONT_SESSION_FIELDS),
        "retired_handcrafted_ctx_cont_fields": list(
            RETIRED_HANDCRAFTED_CTX_CONT_FIELDS
        ),
        "retired_operator_ctx_cont_composite_fields": list(
            RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS
        ),
        "retired_smc_ctx_composite_fields": list(
            RETIRED_SMC_CTX_COMPOSITE_FIELDS
        ),
    }


# v22 (2026-08-15): the price_action_candle_raw_layer family loses
# candle.raw_zero_range_flag, so the ordered family registry and its hash
# change. Widths are never restated here; they derive from the owner tuples.
MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION = (
    "entry_model_native_mandatory_full_stack_v22"
)
MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256 = _sha256_json(
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
)


def model_native_mandatory_full_stack_metadata() -> dict[str, Any]:
    """Return the immutable exact family/name registry embedded in artifacts."""

    return {
        "schema_version": MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION,
        "ordered_family_fields_sha256": MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256,
        "foundation_structure_owner": foundation_structure_contract_metadata(),
        "swing_structure_owner": swing_structure_contract_metadata(),
        "candle_primitive_owner": candle_primitive_contract_metadata(),
        "price_derived_owner": price_derived_contract_metadata(),
        "smc_primitive_owner": smc_primitive_contract_metadata(),
        "family_count": len(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES),
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "available_candidate_feature_count": (
            MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
        ),
        "available_candidate_fields_sha256": (
            MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS_SHA256
        ),
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
        "basic_v1_contract": MODEL_NATIVE_BASIC_V1_CONTRACT,
        "forbidden_legacy_bridge_fields": FORBIDDEN_LEGACY_BRIDGE_FIELDS,
        "retired_signal_fields": RETIRED_MODEL_NATIVE_SIGNAL_FIELDS,
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
    """Return the exact model-native surface or fail on soft compatibility."""

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
    retired = sorted(set(selected) & set(RETIRED_MODEL_NATIVE_SIGNAL_FIELDS))
    if retired:
        failures.append(f"retired_model_native_signal_fields={retired}")
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
    candidate_suffix = tuple(
        selected[MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:]
    )
    if candidate_suffix != MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS:
        failures.append("available_candidate_suffix_order_mismatch")
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
        "basic_v1_contract": dict(MODEL_NATIVE_BASIC_V1_CONTRACT),
        "selected_fields": list(selected),
        "fields": list(fields),
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "available_candidate_feature_count": (
            MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
        ),
        "available_candidate_fields_sha256": (
            MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS_SHA256
        ),
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
        "retired_handcrafted_ctx_cont_fields": list(
            RETIRED_HANDCRAFTED_CTX_CONT_FIELDS
        ),
        "retired_operator_ctx_cont_composite_fields": list(
            RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS
        ),
        "retired_smc_ctx_composite_fields": list(
            RETIRED_SMC_CTX_COMPOSITE_FIELDS
        ),
        "forbidden_legacy_bridge_fields": list(FORBIDDEN_LEGACY_BRIDGE_FIELDS),
        "retired_signal_fields": list(RETIRED_MODEL_NATIVE_SIGNAL_FIELDS),
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
        "available_candidate_feature_count": (
            MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
        ),
        "available_candidate_fields_sha256": (
            MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS_SHA256
        ),
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
    retired_declared = tuple(
        str(value) for value in (contract.get("retired_signal_fields") or ())
    )
    retired_ctx_declared = tuple(
        str(value)
        for value in (contract.get("retired_handcrafted_ctx_cont_fields") or ())
    )
    retired_operator_ctx_declared = tuple(
        str(value)
        for value in (
            contract.get("retired_operator_ctx_cont_composite_fields") or ()
        )
    )
    retired_smc_ctx_declared = tuple(
        str(value)
        for value in (contract.get("retired_smc_ctx_composite_fields") or ())
    )
    if base_fields != MODEL_NATIVE_BASE_FIELDS:
        failures.append("base_fields order mismatch")
    basic_v1_declared = contract.get("basic_v1_contract")
    if not isinstance(basic_v1_declared, Mapping):
        failures.append("basic_v1_contract missing")
    elif dict(basic_v1_declared) != MODEL_NATIVE_BASIC_V1_CONTRACT:
        failures.append("basic_v1_contract metadata mismatch")
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
    if retired_declared != RETIRED_MODEL_NATIVE_SIGNAL_FIELDS:
        failures.append("retired_signal_fields order mismatch")
    if retired_ctx_declared != RETIRED_HANDCRAFTED_CTX_CONT_FIELDS:
        failures.append("retired_handcrafted_ctx_cont_fields order mismatch")
    if retired_operator_ctx_declared != RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS:
        failures.append("retired_operator_ctx_cont_composite_fields order mismatch")
    if retired_smc_ctx_declared != RETIRED_SMC_CTX_COMPOSITE_FIELDS:
        failures.append("retired_smc_ctx_composite_fields order mismatch")
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
