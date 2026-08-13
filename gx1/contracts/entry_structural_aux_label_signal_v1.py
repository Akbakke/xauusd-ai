"""Exact current-bar signal prerequisites for structural auxiliary targets.

These inputs condition representation/slice labels only; they never rewrite
the future-utility direction target.  Every requirement must be retained by
the code-owned mandatory signal prefix so deterministic feature ranking cannot
make dataset construction depend on which optional fields happened to win.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any


# V30 package 7 (2026-08-13): schema v3.  The four geometry requirements
# `geometry_channel_edge`, `geometry_channel_position`,
# `geometry_long_fib_sr_proximity` and `geometry_short_fib_sr_proximity` are
# retired with their producers — `chart.geometry_channel_edge_pressure` and
# `chart.geometry_channel_position_low_to_high` were exact-affine duplicates of
# the two sided proximity stacks (both of which remain mandatory below), and
# the two `fib_*_confluence_*` fields were built on the algebraically
# impossible / mislabelled Fibonacci block that was removed wholesale
# (docs/INDICATOR_FIDELITY_AUDIT_20260813.md §1a).  The structural labels that
# consumed them now rest on the surviving support/resistance proximity and
# respect requirements, which are unchanged.
STRUCTURAL_AUX_LABEL_SIGNAL_SCHEMA_VERSION = (
    "entry_structural_aux_label_signal_v3"
)
STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS = OrderedDict(
    [
        (
            "trend_score",
            (
                "trend.ema_stack_alignment_score",
            ),
        ),
        (
            "trend_conflict",
            ("trend.ema_mtf_divergence_pressure",),
        ),
        (
            "long_trend_bias",
            ("trend.ema_stack_bull_pressure",),
        ),
        (
            "short_trend_bias",
            ("trend.ema_stack_bear_pressure",),
        ),
        (
            "structure_direction",
            ("chart.structure_swing_market_structure_regime_state",),
        ),
        (
            "geometry_support_line_proximity",
            ("chart.geometry_support_line_proximity_stack",),
        ),
        (
            "support_level_proximity",
            ("chart.sr_memory_support_level_proximity_stack",),
        ),
        (
            "support_respect",
            ("chart.sr_memory_support_respect_pressure_long",),
        ),
        (
            "support_reclaim",
            ("chart.sr_memory_support_reclaim_pressure_long",),
        ),
        (
            "geometry_resistance_line_proximity",
            ("chart.geometry_resistance_line_proximity_stack",),
        ),
        (
            "resistance_level_proximity",
            ("chart.sr_memory_resistance_level_proximity_stack",),
        ),
        (
            "resistance_respect",
            ("chart.sr_memory_resistance_respect_pressure_short",),
        ),
        (
            "resistance_reclaim",
            ("chart.sr_memory_resistance_reclaim_pressure_short",),
        ),
        (
            "support_liquidity_rejection",
            ("chart.sr_memory_liquidity_low_level_rejection_long",),
        ),
        (
            "resistance_liquidity_rejection",
            ("chart.sr_memory_liquidity_high_level_rejection_short",),
        ),
    ]
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS_SHA256 = _sha256_json(
    STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS
)


def structural_aux_label_signal_contract_metadata(
    mandatory_fields: Sequence[str],
) -> dict[str, Any]:
    """Bind every requirement to at least one mandatory signal or fail."""

    mandatory = tuple(str(field) for field in mandatory_fields)
    mandatory_set = set(mandatory)
    resolved: OrderedDict[str, str] = OrderedDict()
    missing: OrderedDict[str, list[str]] = OrderedDict()
    for label, candidates in STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS.items():
        selected = next(
            (field for field in candidates if field in mandatory_set),
            None,
        )
        if selected is None:
            missing[label] = list(candidates)
        else:
            resolved[label] = selected
    if missing:
        raise RuntimeError(
            "STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS_NOT_MANDATORY: "
            + json.dumps(missing, sort_keys=True)
        )
    return {
        "schema_version": STRUCTURAL_AUX_LABEL_SIGNAL_SCHEMA_VERSION,
        "requirements_sha256": (
            STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS_SHA256
        ),
        "requirement_count": len(STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS),
        "requirements": {
            label: list(candidates)
            for label, candidates in STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS.items()
        },
        "resolved_mandatory_fields": dict(resolved),
        "all_requirements_mandatory": True,
    }


__all__ = [
    "STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS",
    "STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS_SHA256",
    "STRUCTURAL_AUX_LABEL_SIGNAL_SCHEMA_VERSION",
    "structural_aux_label_signal_contract_metadata",
]
