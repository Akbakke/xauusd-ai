"""Exact raw level prerequisites for the XAU pretrain polarity audit.

The audit measures whether both nearest-support and nearest-resistance
distance pockets exist on the actual TRAIN/VAL snapshot.  It consumes the
identified level registry directly; no hand-fused proximity stack is allowed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


PRETRAIN_POLARITY_SIGNAL_SCHEMA_VERSION = "entry_pretrain_polarity_signal_v3"
SUPPORT_DISTANCE_FEATURE = "level_below_dist_atr"
RESISTANCE_DISTANCE_FEATURE = "level_above_dist_atr"

PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS = (
    SUPPORT_DISTANCE_FEATURE,
    RESISTANCE_DISTANCE_FEATURE,
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
    "PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS",
    "PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS_SHA256",
    "PRETRAIN_POLARITY_SIGNAL_SCHEMA_VERSION",
    "RESISTANCE_DISTANCE_FEATURE",
    "SUPPORT_DISTANCE_FEATURE",
    "pretrain_polarity_signal_contract_metadata",
]
