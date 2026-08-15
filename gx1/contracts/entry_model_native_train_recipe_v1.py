"""Exact source-owned environment for model-native seq513 training.

Readiness advertises the exact supervised, diagnostic and base-task settings
admitted by the launch recipe.  The wrappers clear
every ambient ``ENTRY_*``/``GX1_*`` variable, so an omitted key silently fell
back to a trainer default.  This contract is now the one complete owner of the
values that readiness advertises and the trainer actually receives.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from gx1.contracts.entry_fitted_q_v1 import entry_fitted_q_contract
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    joint_task_weighting_objective_contract,
)

SCHEMA_VERSION = "entry_model_native_seq513_train_recipe_env_v9"

# ── V30 package 5 (2026-08-13): two optimizer stability dampers ──────────────
# These switches own optimizer scheduling only.  They do not alter direction
# labels, class sampling, class weights, logits, losses, or checkpoint scores.
#
# ENTRY_TRAIN_LR_COSINE_DECAY origin: a SWITCH, not a magnitude.  "1" selects
# torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=the declared
# --epochs budget, eta_min=0.0) - the library's own standard cosine anneal
# (Loshchilov & Hutter 2017, SGDR, without restarts and without warmup), whose
# two parameters are the declared epoch budget and the library default 0.0.
# No number is chosen here.  A step size that decays to zero over the declared
# budget shrinks the oscillation amplitude of exactly this limit-cycle class in
# the late epochs where the leans occur.  "0" reproduces the fixed-LR behaviour
# bit-identically: no scheduler object is constructed and no param_group is
# ever written.
#
# ENTRY_TRAIN_WEIGHT_EMA_DECAY origin (operator decision, 2026-08-13): the key
# declares a HORIZON, not a magnitude.  ``epoch`` selects
# ``derive_weight_ema_decay`` below — the EMA averaging horizon is exactly ONE
# declared epoch of optimizer steps, so the decay is computed from the recipe's
# own declared budget:
#     steps_per_epoch = ceil(train_rows / (batch_size * grad_accum_steps))
#     decay           = 1 - 1 / steps_per_epoch
# Why a derivation and not the textbook 0.999: 0.999 is a PER-STEP constant
# whose averaging horizon is ~1000 optimizer steps, and this profile's ENTIRE
# run is shorter than that (25,000 rows / 640 effective batch = 40 steps per
# epoch x 8 epochs = 320 steps), so a 0.999 shadow would still be dominated by
# its initialization at the last checkpoint — it would damp nothing and would
# also be an invented magnitude here (a repository-wide search for 0.999 /
# ema_decay / EMA_DECAY / AveragedModel / polyak / swa found no named constant
# to adopt, 2026-08-13).  Deriving from the declared budget keeps the horizon
# meaningful and follows batch/accumulation/row-budget changes automatically,
# which a pinned float cannot.  0.0 remains the exact-compatibility OFF
# sentinel (no shadow weights allocated; validation and checkpoint selection
# see the raw training weights, byte-identical to the pre-package-5 trainer).
# The recipe owner declares the rule and owns the formula, while the quantity
# that depends on the run's declared data/budget is resolved at train time from
# this owner's function — never from an ambient default (rule 14).
_RECIPE_ENV_TEXT = """
ENTRY_TRAIN_LR_COSINE_DECAY=1
ENTRY_TRAIN_WEIGHT_EMA_DECAY=epoch
GX1_CTX_CONTRACT=V_NEXT
GX1_V10_CKPT_MONITOR=entry_policy_pnl
""".strip()

# Registry tolerance and lifecycle decisions are immutable TRAIN artifacts.
# No q/reaction/retest operator key remains on the trainer recipe surface.
MODEL_NATIVE_V29_REGISTRY_RECIPE_ENV_KEYS: tuple[str, ...] = ()


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

DIRECTION_DIAGNOSTIC_ENV_KEYS = tuple(
    key
    for key in MODEL_NATIVE_RECIPE_ENV_KEYS
    if key == "GX1_V10_CKPT_MONITOR"
)

DIRECTION_DIAGNOSTIC_ENV_TEMPLATE = {
    key: MODEL_NATIVE_RECIPE_ENV[key]
    for key in DIRECTION_DIAGNOSTIC_ENV_KEYS
}

_RECIPE_LIST_FLOAT_KEYS: frozenset[str] = frozenset()
_RECIPE_BOOLEAN_KEYS = frozenset(
    {
        "ENTRY_CKPT_DIRECTION_SLICE_GUARD",
        # V30 package 5: a schedule ON/OFF switch, never a magnitude.
        "ENTRY_TRAIN_LR_COSINE_DECAY",
    }
)
_RECIPE_STRING_KEYS = frozenset(
    {
        "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES",
        "GX1_V10_CKPT_MONITOR",
        # V30 package 6: a declared horizon token ("epoch") or the "0.0" OFF
        # sentinel — never a bare magnitude, so it is a string key.
        "ENTRY_TRAIN_WEIGHT_EMA_DECAY",
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


DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT = {
    **_recipe_projection(DIRECTION_DIAGNOSTIC_ENV_KEYS),
    "entry_action_q_loss": "masked_raw_bps_mean_squared_error",
    "entry_fitted_q": entry_fitted_q_contract(),
    "fixed_relative_task_weights": False,
    "handwritten_rank_losses": False,
    "handwritten_composite_weights": False,
    "handwritten_gate_regularization": False,
    "fixed_target_normalization_scales": False,
    "joint_task_weighting": joint_task_weighting_objective_contract(),
    "checkpoint_monitor": (
        "val_realized_gross_spread_inclusive_entry_policy_pnl_bps_research_only"
    ),
    "production_authority_ready": False,
}
DIRECTION_CONTEXT_SLICE_CONTRACT = {
    "retired": True,
    "entry_authority": False,
    "replacement": "raw_entry_fitted_q_and_net_oos_policy_replay",
}

# Waves A-C retire every handcrafted loss magnitude, rank margin/quantile,
# target-normalization scale and gate-gradient control from the prior 170-key
# surface. The remaining pre-V29 keys are diagnostics/admission, label-mode and
# optimizer switches; none changes relative task influence.
_PRE_V29_RECIPE_ENV_KEY_COUNT = 4
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

# ── V30 package 5 stability dampers (2026-08-13) ─────────────────────────────
# ``ENTRY_TRAIN_WEIGHT_EMA_DECAY`` is REQUIRED-BY-OPERATOR-DECISION: the
# recipe owner declares the key and pins the value it currently carries, and
# only a new recipe decision may move it. Exactly two values are declared — the DISABLED sentinel and
# the epoch-horizon token that selects :func:`derive_weight_ema_decay`.  A bare
# float is deliberately NOT accepted: a pinned decay could not follow the
# declared batch/accumulation/row budget, which is the whole content of the
# 2026-08-13 operator decision recorded in the origin comment above.
MODEL_NATIVE_WEIGHT_EMA_DECAY_DISABLED_VALUE = "0.0"
MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE = "epoch"
MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES = (
    MODEL_NATIVE_WEIGHT_EMA_DECAY_DISABLED_VALUE,
    MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE,
)
# The horizon the operator declared: ONE epoch of optimizer steps.  It is the
# unit of the trainer's own declared budget (--epochs counts these), not a
# tuned number.
MODEL_NATIVE_WEIGHT_EMA_HORIZON_EPOCHS = 1
MODEL_NATIVE_STABILITY_DAMPER_RECIPE_ENV_KEYS = (
    "ENTRY_TRAIN_LR_COSINE_DECAY",
    "ENTRY_TRAIN_WEIGHT_EMA_DECAY",
)
for _damper_key in MODEL_NATIVE_STABILITY_DAMPER_RECIPE_ENV_KEYS:
    if _damper_key not in MODEL_NATIVE_RECIPE_ENV:
        raise RuntimeError(
            f"MODEL_NATIVE_RECIPE_ENV_STABILITY_DAMPER_KEY_MISSING: {_damper_key}"
        )
if MODEL_NATIVE_RECIPE_ENV["ENTRY_TRAIN_WEIGHT_EMA_DECAY"] not in (
    MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES
):
    raise RuntimeError(
        "MODEL_NATIVE_RECIPE_ENV_WEIGHT_EMA_DECAY_UNDECLARED: the weight-EMA "
        "key carries a horizon declaration, not a magnitude, so it must be "
        f"exactly one of {MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES}; any "
        "other value is an operator recipe decision that must be recorded in "
        "the origin comment above before it is declared here"
    )


def derive_weight_ema_decay(
    *,
    train_rows: int,
    batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    """Derive the weight-EMA decay from the run's declared training budget.

    The operator decision of 2026-08-13 (origin comment at the top of this
    module): the EMA averaging horizon is ONE declared epoch of optimizer
    steps.  A convex EMA ``shadow <- d*shadow + (1-d)*current`` has mean
    averaging horizon ``1/(1-d)`` steps, so a horizon of ``steps_per_epoch``
    steps is exactly ``d = 1 - 1/steps_per_epoch``.  Nothing here is chosen:
    the three inputs are the declared row budget of one epoch and the two
    declared batch-geometry CLI values, and
    ``MODEL_NATIVE_WEIGHT_EMA_HORIZON_EPOCHS`` is the declared horizon.

    Returns the derivation payload (decay + every input and intermediate) so
    the trainer can log and record it instead of recomputing it — one owner
    for the formula (rule 2a origin class 1 for the horizon, class 3 for the
    budget inputs; rule 14: training reads this function, never an ambient
    default).  Fails closed on any non-usable budget; in particular a budget
    that yields a single optimizer step per epoch cannot express an average at
    all (it would produce ``d = 0``, colliding with the OFF sentinel), so it is
    an error rather than a silent disable.
    """

    values = {
        "train_rows": train_rows,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"MODEL_NATIVE_WEIGHT_EMA_BUDGET_INVALID: {name}={value!r} "
                "must be an int"
            )
        if value <= 0:
            raise RuntimeError(
                f"MODEL_NATIVE_WEIGHT_EMA_BUDGET_INVALID: {name}={value!r} "
                "must be > 0"
            )
    effective_batch = int(batch_size) * int(grad_accum_steps)
    steps_per_epoch = -(-int(train_rows) // effective_batch)  # ceil division
    horizon_steps = steps_per_epoch * MODEL_NATIVE_WEIGHT_EMA_HORIZON_EPOCHS
    if horizon_steps < 2:
        raise RuntimeError(
            "MODEL_NATIVE_WEIGHT_EMA_HORIZON_DEGENERATE: the declared budget "
            f"gives {horizon_steps} optimizer step(s) per epoch "
            f"(train_rows={train_rows}, batch_size={batch_size}, "
            f"grad_accum_steps={grad_accum_steps}); an averaging horizon needs "
            "at least two steps"
        )
    decay = 1.0 - 1.0 / float(horizon_steps)
    if not (0.0 < decay < 1.0):
        raise RuntimeError(
            f"MODEL_NATIVE_WEIGHT_EMA_DECAY_DERIVED_INVALID: {decay!r}"
        )
    return {
        "recipe_key": "ENTRY_TRAIN_WEIGHT_EMA_DECAY",
        "declared_value": MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE,
        "horizon_epochs": int(MODEL_NATIVE_WEIGHT_EMA_HORIZON_EPOCHS),
        "train_rows": int(train_rows),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "effective_batch_rows": int(effective_batch),
        "steps_per_epoch": int(steps_per_epoch),
        "horizon_optimizer_steps": int(horizon_steps),
        "weight_ema_decay": float(decay),
    }


def resolve_weight_ema_decay(
    declared_value: str,
    *,
    train_rows: int,
    batch_size: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    """Resolve the declared recipe value into the decay the trainer applies.

    ``0.0`` resolves to ``0.0`` (OFF) without touching the budget at all, so
    the disabled path stays exactly disable-able and bit-identical.  The
    epoch-horizon token resolves through :func:`derive_weight_ema_decay`.  Any
    other string fails closed — the trainer must never invent a decay.
    """

    value = str(declared_value)
    if value == MODEL_NATIVE_WEIGHT_EMA_DECAY_DISABLED_VALUE:
        return {
            "recipe_key": "ENTRY_TRAIN_WEIGHT_EMA_DECAY",
            "declared_value": value,
            "weight_ema_decay": 0.0,
        }
    if value == MODEL_NATIVE_WEIGHT_EMA_DECAY_EPOCH_HORIZON_VALUE:
        return derive_weight_ema_decay(
            train_rows=train_rows,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
        )
    raise RuntimeError(
        "MODEL_NATIVE_WEIGHT_EMA_DECAY_UNDECLARED: "
        f"{value!r} is not one of "
        f"{MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES}"
    )


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
