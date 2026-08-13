"""Exact signal prerequisites for the XAU pretrain polarity audit.

The pretrain audit uses these current-bar geometry values to prove that the
sided support/resistance proximity surface reaches the dataset snapshot with
both pockets populated.  They are decision evidence, not optional diagnostics,
so deterministic TRAIN ranking must never be able to remove them from the
model-native signal surface.

V30 package 7 (2026-08-13), schema v2: `chart.geometry_support_minus_resistance_stack`
and `chart.geometry_channel_position_low_to_high` are no longer required,
because they are no longer emitted - both were exact-affine duplicates of the
two stacks below plus one ctx field
(docs/FEATURE_VALUE_REVIEW_20260813.md A.4).  The channel-position polarity
STATISTIC is retired with them; see
`gx1/scripts/audit_xau_direction_repair_pretrain_v1.py`, which now reports only
the pocket-occupancy measurement it can still take (rule 2e: omit a field
rather than emit a placeholder that reads as a result).  This is a REDUCTION in
proof coverage, recorded here rather than left implicit.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


PRETRAIN_POLARITY_SIGNAL_SCHEMA_VERSION = "entry_pretrain_polarity_signal_v2"
SUPPORT_STACK_FEATURE = "chart.geometry_support_line_proximity_stack"
RESISTANCE_STACK_FEATURE = "chart.geometry_resistance_line_proximity_stack"

PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS = (
    SUPPORT_STACK_FEATURE,
    RESISTANCE_STACK_FEATURE,
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
    "RESISTANCE_STACK_FEATURE",
    "SUPPORT_STACK_FEATURE",
    "pretrain_polarity_signal_contract_metadata",
]
