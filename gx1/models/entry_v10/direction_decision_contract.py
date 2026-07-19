"""Canonical model-native LONG/SHORT/FLAT decision contract.

This module describes the public decision surface written into every new
Entry-V10 bundle.  Consumers must validate the bundle metadata before using
``direction_logits``; auxiliary heads never have authority to rewrite the
model's final argmax.
"""

from __future__ import annotations

from typing import Any, Mapping

DIRECTION_DECISION_CONTRACT_SCHEMA_VERSION = "gx1_model_direction_decision_v3"
MODEL_DIRECTION_SELECTION_MODE = "model_direction_argmax"
DIRECTION_LOGITS_KEY = "direction_logits"
PUBLIC_TRADE_FLAT_LOGITS_KEY = "public_trade_flat_decision_logits"
MODEL_DIRECTION_OPERATING_POINT_KEYS = frozenset({"selection_score", "max_trades"})


def model_direction_decision_contract_metadata() -> dict[str, Any]:
    """Return a fresh, JSON-serializable copy of the exact public contract."""

    return {
        "schema_version": DIRECTION_DECISION_CONTRACT_SCHEMA_VERSION,
        "selection_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_logits_key": DIRECTION_LOGITS_KEY,
        "direction_class_order": ["LONG", "SHORT", "FLAT"],
        "direction_decision": "argmax(direction_logits)",
        "public_trade_flat_logits_key": PUBLIC_TRADE_FLAT_LOGITS_KEY,
        "public_trade_flat_class_order": ["TRADE", "FLAT"],
        "public_trade_flat_formula": (
            "[max(direction_logits[LONG],direction_logits[SHORT]),"
            "direction_logits[FLAT]]"
        ),
        "output_stage": (
            "final_model_forward_after_learned_evidence_fusion_and_calibration"
        ),
        "auxiliary_heads_direction_authority": "none",
        "runtime_direction_overrides_allowed": False,
        "runtime_direction_thresholds_allowed": False,
        "sizing_authority": "separate_top_level_bundle_contract",
    }


def require_model_direction_decision_contract(
    metadata: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate and return the canonical contract, or fail closed.

    Exact equality is intentional: a partially populated or softly compatible
    declaration cannot prove that training, audit, replay, and live all use the
    same public decision path.
    """

    raw = metadata.get("direction_decision_contract")
    expected = model_direction_decision_contract_metadata()
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"{context} missing required direction_decision_contract metadata"
        )
    observed = dict(raw)
    mismatches = [
        key
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    ]
    unexpected = sorted(set(observed) - set(expected))
    missing = sorted(set(expected) - set(observed))
    if mismatches or unexpected or missing:
        raise RuntimeError(
            f"{context} direction_decision_contract mismatch: "
            f"mismatched={sorted(set(mismatches))} missing={missing} "
            f"unexpected={unexpected}"
        )
    return observed


def require_model_direction_operating_point(
    operating_point: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Require the exact rule-free live operating-point surface.

    ``max_trades`` is an execution exposure limit, not a direction selector.
    Everything else that historically rewrote model direction (edge/utility
    thresholds, session allowlists, side overlays, and similar pass-through
    keys) is rejected by exact-key equality rather than silently ignored.
    """

    if not isinstance(operating_point, Mapping):
        raise RuntimeError(f"{context} missing required operating_point mapping")
    observed = dict(operating_point)
    missing = sorted(MODEL_DIRECTION_OPERATING_POINT_KEYS - set(observed))
    unexpected = sorted(set(observed) - MODEL_DIRECTION_OPERATING_POINT_KEYS)
    if missing or unexpected:
        raise RuntimeError(
            f"{context} operating_point contract mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    if observed.get("selection_score") != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError(
            f"{context} operating_point.selection_score must be exactly "
            f"{MODEL_DIRECTION_SELECTION_MODE!r}"
        )
    max_trades = observed.get("max_trades")
    if isinstance(max_trades, bool) or not isinstance(max_trades, int) or max_trades <= 0:
        raise RuntimeError(
            f"{context} operating_point.max_trades must be a positive integer; "
            f"got {max_trades!r}"
        )
    return observed
