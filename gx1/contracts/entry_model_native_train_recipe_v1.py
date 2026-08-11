"""Exact source-owned environment for model-native seq513 training.

Readiness used to advertise direction-balance, path-quality and tail-direction
settings that were not all admitted by the launch recipe.  The wrappers clear
every ambient ``ENTRY_*``/``GX1_*`` variable, so an omitted key silently fell
back to a trainer default.  This contract is now the one complete owner of the
values that readiness advertises and the trainer actually receives.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
)

SCHEMA_VERSION = "entry_model_native_seq513_train_recipe_env_v1"

# ENTRY_DIRECTION_LOGIT_ADJUST_TAU origin (adopted 2026-08-11): tau=1.0 is the
# standard value of the published method (Menon et al. 2021, "Long-tail
# learning via logit adjustment"); the class priors it scales are the physical
# TRAIN label rates computed in-trainer at dataset load, never a stored guess.
# The adjustment is a training-loss device only; emitted/serving logits stay
# unadjusted. tau=0.0 is the exact-compatibility switch (raw CE, sqrt-softened
# class weights); tau>0 sets the direction CE class weights to the neutral 1.0
# because adjustment replaces reweighting (combining both double-corrects).
_RECIPE_ENV_TEXT = """
ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP=20.0
ENTRY_AUX_BAD_PATH_WEIGHT=1.25
ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP=16.0
ENTRY_AUX_CLEAN_EDGE_WEIGHT=0.45
ENTRY_AUX_MFE_SCALE_BPS=20.0
ENTRY_AUX_MFE_WEIGHT=0.25
ENTRY_AUX_PATH_SCALE_BPS=50.0
ENTRY_AUX_PATH_WEIGHT=0.90
ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP=10.0
ENTRY_AUX_SURVIVAL_WEIGHT=0.10
ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP=12.0
ENTRY_AUX_TRADABLE_WEIGHT=1.15
ENTRY_BAD_PATH_CE_MULTIPLIER=1.50
ENTRY_BAD_PATH_PROB_PENALTY=0.24
ENTRY_BAD_PATH_QUALITY_RANK_MARGIN=0.25
ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE=0.25
ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT=2.00
ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT=0.50
ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE=0.05
ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.35
ENTRY_CKPT_DIRECTION_SLICE_GUARD=1
ENTRY_CLEAN_EDGE_RANKING_MARGIN=0.12
ENTRY_CLEAN_EDGE_RANKING_WEIGHT=0.25
ENTRY_COST_FLAT_TO_LONG=1.60
ENTRY_COST_FLAT_TO_SHORT=1.60
ENTRY_COST_LONG_TO_FLAT=0.45
ENTRY_COST_LONG_TO_SHORT=2.00
ENTRY_COST_SENSITIVE_LOSS=1
ENTRY_COST_SENSITIVE_SCALE=0.25
ENTRY_COST_SHORT_TO_FLAT=0.45
ENTRY_COST_SHORT_TO_LONG=2.00
ENTRY_DEAD_LONG_CE_MULTIPLIER=1.80
ENTRY_DEAD_LONG_PROB_PENALTY=0.40
ENTRY_DIRECTION_CE_SCALE=12.00
ENTRY_DIRECTION_CLASS_WEIGHT_CAP=8.0
ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=0.10
ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS=8
ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR=0.10
ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION=0.50
ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT=8.00
ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02
ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=8.00
ENTRY_DIRECTION_LOGIT_ADJUST_TAU=1.0
ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR=0.05
ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION=0.50
ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=12.00
ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=0.05
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN=0.10
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS=15.0
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT=2.00
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN=0.02
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS=8
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT=4.00
ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS=8
ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT=2.00
ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER=0
ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS=8
ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN=0.02
ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT=4.00
ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES=0,1,2,3,4
ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6
ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=3
ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION=mean_max
ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR=0.05
ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION=0.50
ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=8.00
ENTRY_DIRECTION_SLICE_MIN_ROWS=8
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS=8
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE=0.02
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT=3.00
ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT=4.00
ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS=8
ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR=0.30
ENTRY_DIRECTION_SLICE_TRUE_MARGIN=0.10
ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE=0.10
ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS=8
ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT=2.00
ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN=0.10
ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT=4.00
ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS=15.0
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN=0.10
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH=0.50
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS=15.0
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS=0.0
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT=2.00
ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP=4.0
ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH=0.50
ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS=15.0
ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS=0.0
ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT=2.00
ENTRY_DIRECTION_VS_FLAT_MARGIN=0.10
ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT=4.00
ENTRY_FLAT_CLASS_WEIGHT_FLOOR=1.0
ENTRY_HARD_NEG_LONG_CE_MULTIPLIER=1.35
ENTRY_HARD_NEG_LONG_PROB_PENALTY=0.20
ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP=20.0
ENTRY_HIER_BAD_PATH_WEIGHT=1.25
ENTRY_HIER_FLAT_LOGIT_MARGIN=0.10
ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE=0.10
ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT=8.00
ENTRY_HIER_MAE_WEIGHT=0.35
ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10
ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02
ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT=4.00
ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS=15.0
ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP=8.0
ENTRY_HIER_SIDE_VALIDITY_WEIGHT=1.50
ENTRY_HIER_SIDE_WEIGHT=1.75
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN=0.10
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE=0.10
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS=8
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT=8.00
ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN=0.02
ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT=4.00
ENTRY_HIER_SLICE_SIDE_CE_WEIGHT=4.00
ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE=0.10
ENTRY_HIER_SLICE_SIDE_MIN_ROWS=8
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE=0.10
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS=8
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE=0.02
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT=4.00
ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN=0.10
ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT=3.00
ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN=0.02
ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT=4.00
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE=0.10
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS=8
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE=0.02
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT=4.00
ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10
ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02
ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT=4.00
ENTRY_HIER_TRADE_WEIGHT=2.00
ENTRY_HIER_UTILITY_WEIGHT=1.00
ENTRY_MTF_DIR_AUX_WEIGHT=0.30
ENTRY_OFFLINE_RL_Q_WEIGHT=0.50
ENTRY_OFFLINE_RL_RANK_WEIGHT=0.05
ENTRY_OFFLINE_RL_V_WEIGHT=0.20
ENTRY_PATH_QUALITY_RANK_MARGIN=0.25
ENTRY_PATH_QUALITY_RANK_QUANTILE=0.25
ENTRY_PATH_QUALITY_RANK_WEIGHT=2.00
ENTRY_PRED_BALANCE_ALPHA=0.50
ENTRY_PRED_BALANCE_CLASS_WEIGHTS=1.0,1.0,1.0
ENTRY_PRED_BALANCE_TARGET=label
ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT=0.50
ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT=0.05
ENTRY_SPECIALIST_GATE_MIN_MEAN=0.01
ENTRY_SYMMETRIC_NEGATIVES=1
ENTRY_TAIL_DIRECTION_CE_WEIGHT=0.35
ENTRY_TAIL_DIRECTION_MIN_BATCH=8
ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE=0.70
ENTRY_TEASER_LONG_CE_MULTIPLIER=1.35
ENTRY_TEASER_LONG_PROB_PENALTY=0.16
ENTRY_TRENDLINE_RAIL_AUX_WEIGHT=1.00
ENTRY_UNIFIED_EXIT_ACTION_WEIGHT=1.00
GX1_CTX_CONTRACT=V_NEXT
GX1_V10_CKPT_MONITOR=dir_acc
ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q=REQUIRED
ENTRY_LEVEL_REGISTRY_REACTION_WINDOW_BARS=12
ENTRY_LEVEL_REGISTRY_RETEST_WINDOW_BARS=24
ENTRY_TRENDLINE_REGISTRY_RETEST_WINDOW_BARS=7
""".strip()

# ── V29 registry recipe keys (docs/V29_EVENT_SURFACE_DESIGN_20260811.md §1.4,
# §8 item 4; stage-1 owners: level_registry_v1 / trendline_registry_v1) ──────
# ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q carries the marker value REQUIRED: the
# design doc declares NO default for `q` (operator decision at the immutable
# recipe decision, §8 item 4), so the dataset rebuild must receive it as an
# explicit CLI input (`--level-tol-quantile-q` on the cache prebuild) and the
# fitted tolerances are frozen with provenance in the rebuild authority
# artifacts (rule 18).  A run that tries to consume the marker as a number
# fails closed.  The three window keys restate the stage-1 owners' named
# convention constants and are equality-checked against them below — the
# module constants stay the single numerical truth (rule 2a/13).
MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS = (
    "ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q",
    "ENTRY_LEVEL_REGISTRY_REACTION_WINDOW_BARS",
    "ENTRY_LEVEL_REGISTRY_RETEST_WINDOW_BARS",
    "ENTRY_TRENDLINE_REGISTRY_RETEST_WINDOW_BARS",
)
MODEL_NATIVE_V29_REGISTRY_TOL_QUANTILE_REQUIRED_MARKER = "REQUIRED"


def _parse_recipe_env(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in text.splitlines():
        key, separator, value = row.partition("=")
        if separator != "=" or not key or not value or key in result:
            raise RuntimeError(f"MODEL_NATIVE_RECIPE_ENV_ROW_INVALID: {row!r}")
        if not (key.startswith("ENTRY_") or key.startswith("GX1_")):
            raise RuntimeError(f"MODEL_NATIVE_RECIPE_ENV_KEY_INVALID: {key}")
        result[key] = value
    return result


MODEL_NATIVE_RECIPE_ENV = _parse_recipe_env(_RECIPE_ENV_TEXT)
MODEL_NATIVE_RECIPE_ENV_KEYS = tuple(sorted(MODEL_NATIVE_RECIPE_ENV))
MODEL_NATIVE_RECIPE_ENV_SHA256 = hashlib.sha256(
    json.dumps(
        MODEL_NATIVE_RECIPE_ENV,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()

PATH_CALIBRATION_ENV_KEYS = tuple(
    key
    for key in MODEL_NATIVE_RECIPE_ENV_KEYS
    if key.startswith("ENTRY_PATH_QUALITY_RANK_")
    or key.startswith("ENTRY_BAD_PATH_QUALITY_RANK_")
)
DIRECTION_BALANCE_ENV_KEYS = tuple(
    key
    for key in MODEL_NATIVE_RECIPE_ENV_KEYS
    if key.startswith(
        (
            "ENTRY_PRED_BALANCE_",
            "ENTRY_DIRECTION_",
            "ENTRY_CKPT_",
            "ENTRY_HIER_",
            "ENTRY_TRENDLINE_RAIL_",
            "ENTRY_OFFLINE_RL_",
        )
    )
    or key == "GX1_V10_CKPT_MONITOR"
)
TAIL_DIRECTION_ENV_KEYS = tuple(
    key
    for key in MODEL_NATIVE_RECIPE_ENV_KEYS
    if key.startswith("ENTRY_TAIL_DIRECTION_")
)

PATH_CALIBRATION_ENV_TEMPLATE = {
    key: MODEL_NATIVE_RECIPE_ENV[key] for key in PATH_CALIBRATION_ENV_KEYS
}
DIRECTION_BALANCE_ENV_TEMPLATE = {
    key: MODEL_NATIVE_RECIPE_ENV[key] for key in DIRECTION_BALANCE_ENV_KEYS
}
TAIL_DIRECTION_ENV_TEMPLATE = {
    key: MODEL_NATIVE_RECIPE_ENV[key] for key in TAIL_DIRECTION_ENV_KEYS
}

_RECIPE_LIST_FLOAT_KEYS = frozenset({"ENTRY_PRED_BALANCE_CLASS_WEIGHTS"})
_RECIPE_BOOLEAN_KEYS = frozenset(
    {
        "ENTRY_CKPT_DIRECTION_SLICE_GUARD",
        "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER",
    }
)
_RECIPE_STRING_KEYS = frozenset(
    {
        "ENTRY_PRED_BALANCE_TARGET",
        "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES",
        "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION",
        "GX1_V10_CKPT_MONITOR",
    }
)


def _recipe_public_name(env_key: str) -> str:
    if env_key == "GX1_V10_CKPT_MONITOR":
        return "ckpt_monitor"
    if not env_key.startswith("ENTRY_"):
        raise RuntimeError(f"MODEL_NATIVE_RECIPE_PUBLIC_KEY_INVALID: {env_key}")
    return env_key.removeprefix("ENTRY_").lower()


def _recipe_public_value(env_key: str) -> object:
    raw = MODEL_NATIVE_RECIPE_ENV[env_key]
    if env_key in _RECIPE_LIST_FLOAT_KEYS:
        values = [float(item) for item in raw.split(",")]
        if not values:
            raise RuntimeError(f"MODEL_NATIVE_RECIPE_LIST_EMPTY: {env_key}")
        return values
    if env_key in _RECIPE_BOOLEAN_KEYS:
        if raw not in {"0", "1"}:
            raise RuntimeError(f"MODEL_NATIVE_RECIPE_BOOLEAN_INVALID: {env_key}")
        return raw == "1"
    if env_key in _RECIPE_STRING_KEYS:
        return raw
    try:
        return int(raw) if raw.lstrip("-").isdigit() else float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"MODEL_NATIVE_RECIPE_NUMERIC_INVALID: {env_key}={raw!r}"
        ) from exc


def _recipe_projection(keys: tuple[str, ...]) -> dict[str, object]:
    projected = {
        _recipe_public_name(key): _recipe_public_value(key) for key in keys
    }
    if len(projected) != len(keys):
        raise RuntimeError("MODEL_NATIVE_RECIPE_PUBLIC_KEY_COLLISION")
    return projected


PATH_CALIBRATION_RECIPE_CONTRACT = {
    "path_quality_rank_full_batch": True,
    **_recipe_projection(PATH_CALIBRATION_ENV_KEYS),
}
DIRECTION_BALANCE_RECIPE_CONTRACT = {
    **_recipe_projection(DIRECTION_BALANCE_ENV_KEYS),
    "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    "hierarchical_entry_heads_enabled": True,
    "side_validity_head_enabled": True,
    "trendline_rail_head_enabled": True,
    "anchor_gate_enabled": False,
}
TAIL_DIRECTION_RECIPE_CONTRACT = {
    **_recipe_projection(TAIL_DIRECTION_ENV_KEYS),
    "tail_direction_mask": "directional_tradable_clean_path_top_quality",
}
DIRECTION_CONTEXT_SLICE_CONTRACT = {
    "source": "post_smoke_audit.direction_slice_contract",
    "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
    "min_rows": 64,
    "requires_majority_baseline": True,
    "requires_class_distribution_coverage": True,
    "skips_low_label_diversity": True,
}

# The audited pre-V29 recipe surface held exactly 164 keys; the total is now
# DERIVED as that audited base plus the declared V29 registry key tuple, so a
# key can be neither silently dropped nor silently invented.
_PRE_V29_RECIPE_ENV_KEY_COUNT = 164
_EXPECTED_RECIPE_ENV_KEY_COUNT = _PRE_V29_RECIPE_ENV_KEY_COUNT + len(
    MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS
)
if len(MODEL_NATIVE_RECIPE_ENV) != _EXPECTED_RECIPE_ENV_KEY_COUNT:
    raise RuntimeError(
        "MODEL_NATIVE_RECIPE_ENV_COUNT_INVALID: "
        f"observed={len(MODEL_NATIVE_RECIPE_ENV)} "
        f"expected={_EXPECTED_RECIPE_ENV_KEY_COUNT}"
    )
for _v29_key in MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS:
    if _v29_key not in MODEL_NATIVE_RECIPE_ENV:
        raise RuntimeError(
            f"MODEL_NATIVE_RECIPE_ENV_V29_KEY_MISSING: {_v29_key}"
        )
if MODEL_NATIVE_RECIPE_ENV["ENTRY_LEVEL_REGISTRY_TOL_QUANTILE_Q"] != (
    MODEL_NATIVE_V29_REGISTRY_TOL_QUANTILE_REQUIRED_MARKER
):
    raise RuntimeError(
        "MODEL_NATIVE_RECIPE_ENV_V29_TOL_QUANTILE_INVALID: the design doc "
        "declares no default; the value stays REQUIRED until the immutable "
        "recipe decision supplies it"
    )


def _require_v29_registry_recipe_window_consistency() -> None:
    """The window keys must equal the stage-1 owners' named constants."""

    from gx1.features.level_registry_v1 import (
        LEVEL_REGISTRY_REACTION_WINDOW_BARS,
        LEVEL_REGISTRY_RETEST_WINDOW_BARS,
    )
    from gx1.features.trendline_registry_v1 import (
        TRENDLINE_RETEST_WINDOW_BARS_V1,
    )

    expected = {
        "ENTRY_LEVEL_REGISTRY_REACTION_WINDOW_BARS": (
            LEVEL_REGISTRY_REACTION_WINDOW_BARS
        ),
        "ENTRY_LEVEL_REGISTRY_RETEST_WINDOW_BARS": (
            LEVEL_REGISTRY_RETEST_WINDOW_BARS
        ),
        "ENTRY_TRENDLINE_REGISTRY_RETEST_WINDOW_BARS": (
            TRENDLINE_RETEST_WINDOW_BARS_V1
        ),
    }
    for key, owner_value in expected.items():
        if int(MODEL_NATIVE_RECIPE_ENV[key]) != int(owner_value):
            raise RuntimeError(
                "MODEL_NATIVE_RECIPE_ENV_V29_WINDOW_MISMATCH: "
                f"{key}={MODEL_NATIVE_RECIPE_ENV[key]!r} "
                f"owner={owner_value!r}"
            )


_require_v29_registry_recipe_window_consistency()


def require_model_native_recipe_env(value: Mapping[str, Any]) -> dict[str, str]:
    """Require the exact source-owned key/value surface without pass-through."""

    if not isinstance(value, Mapping):
        raise RuntimeError("MODEL_NATIVE_RECIPE_ENV_MISSING")
    normalized = {str(key): str(item) for key, item in value.items()}
    if normalized != MODEL_NATIVE_RECIPE_ENV:
        missing = sorted(set(MODEL_NATIVE_RECIPE_ENV) - set(normalized))
        extra = sorted(set(normalized) - set(MODEL_NATIVE_RECIPE_ENV))
        changed = sorted(
            key
            for key in set(normalized).intersection(MODEL_NATIVE_RECIPE_ENV)
            if normalized[key] != MODEL_NATIVE_RECIPE_ENV[key]
        )
        raise RuntimeError(
            "MODEL_NATIVE_RECIPE_ENV_MISMATCH: "
            f"missing={missing} extra={extra} changed={changed}"
        )
    return dict(MODEL_NATIVE_RECIPE_ENV)


def model_native_recipe_env_contract_metadata() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "count": len(MODEL_NATIVE_RECIPE_ENV),
        "sha256": MODEL_NATIVE_RECIPE_ENV_SHA256,
        "keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
    }
