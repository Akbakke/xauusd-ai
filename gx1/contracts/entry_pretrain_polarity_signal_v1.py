"""Exact signal prerequisites for the XAU pretrain polarity audit.

The pretrain audit uses these current-bar geometry values to prove that
support/resistance and channel position have coherent polarity.  They are
decision evidence, not optional diagnostics, so deterministic TRAIN ranking
must never be able to remove them from the model-native signal surface.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


PRETRAIN_POLARITY_SIGNAL_SCHEMA_VERSION = "entry_pretrain_polarity_signal_v1"
SUPPORT_STACK_FEATURE = "chart.geometry_support_line_proximity_stack"
RESISTANCE_STACK_FEATURE = "chart.geometry_resistance_line_proximity_stack"
SUPPORT_MINUS_RESISTANCE_FEATURE = (
    "chart.geometry_support_minus_resistance_stack"
)
CHANNEL_POSITION_FEATURE = "chart.geometry_channel_position_low_to_high"

PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS = (
    SUPPORT_STACK_FEATURE,
    RESISTANCE_STACK_FEATURE,
    SUPPORT_MINUS_RESISTANCE_FEATURE,
    CHANNEL_POSITION_FEATURE,
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS_SHA256 = _sha256_json(
    PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS
)


def pretrain_polarity_signal_contract_metadata(
    mandatory_fields: Sequence[str],
) -> dict[str, Any]:
    """Bind every polarity-audit input to the mandatory prefix or fail."""

    mandatory = {str(field) for field in mandatory_fields}
    missing = [
        field
        for field in PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS
        if field not in mandatory
    ]
    if missing:
        raise RuntimeError(
            "PRETRAIN_POLARITY_SIGNAL_REQUIREMENTS_NOT_MANDATORY: "
            + json.dumps(missing)
        )
    return {
        "schema_version": PRETRAIN_POLARITY_SIGNAL_SCHEMA_VERSION,
        "required_fields_sha256": (
            PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS_SHA256
        ),
        "required_field_count": len(PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS),
        "required_fields": list(PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS),
        "all_requirements_mandatory": True,
    }


__all__ = [
    "CHANNEL_POSITION_FEATURE",
    "PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS",
    "PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS_SHA256",
    "PRETRAIN_POLARITY_SIGNAL_SCHEMA_VERSION",
    "RESISTANCE_STACK_FEATURE",
    "SUPPORT_MINUS_RESISTANCE_FEATURE",
    "SUPPORT_STACK_FEATURE",
    "pretrain_polarity_signal_contract_metadata",
]
