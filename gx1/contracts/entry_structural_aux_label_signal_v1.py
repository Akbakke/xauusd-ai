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


# V30 clock/source repair (2026-08-14): schema v5 retires the five
# hand-weighted trend/structure score prerequisites. The label builder now
# consumes one genuine signed source per closed timeframe. Support/resistance
# auxiliaries use forward-realized trendline-registry touch/hold labels, not
# same-bar hand-fused S/R score fields, so they have no signal prerequisite.
STRUCTURAL_AUX_LABEL_SIGNAL_SCHEMA_VERSION = (
    "entry_structural_aux_label_signal_v7"
)
STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS = OrderedDict(
    [
        (
            # 2026-09-21 (F-9): ``chart.local_ema50_200_spread_atr`` is
            # retired as an exact affine duplicate; the label needs only the
            # SIGN of the 50/200 spread, which the declared difference pair
            # reproduces bit-exactly ((close-ema200)/atr - (close-ema50)/atr
            # = (ema50-ema200)/atr on every row — same denominator, so the
            # sign is preserved exactly, including exact zeros).  Schema v7
            # introduces the explicit difference form ("difference",
            # minuend, subtrahend); a bare string remains a single column.
            "trend_m5",
            (
                (
                    "difference",
                    "chart.local_price_vs_ema200_atr",
                    "chart.local_price_vs_ema50_atr",
                ),
            ),
        ),
        (
            "trend_m15",
            ("ctx_cont.m15_ema5_20_spread_atr_canon_v2",),
        ),
        (
            "trend_h1",
            ("ctx_cont._v1h1_ema_diff",),
        ),
        (
            "trend_h4",
            ("ctx_cont._v1h4_ema_diff",),
        ),
        (
            "trend_d1",
            ("ctx_cont.d1_ema_slope_20_canon_v2",),
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
    resolved: OrderedDict[str, Any] = OrderedDict()
    missing: OrderedDict[str, list[Any]] = OrderedDict()

    def _candidate_fields(candidate: Any) -> tuple[str, ...]:
        if isinstance(candidate, str):
            return (candidate,)
        if (
            isinstance(candidate, (tuple, list))
            and len(candidate) == 3
            and candidate[0] == "difference"
            and all(isinstance(part, str) for part in candidate[1:])
        ):
            return (str(candidate[1]), str(candidate[2]))
        raise RuntimeError(
            "STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENT_FORM_INVALID: "
            + repr(candidate)
        )

    for label, candidates in STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS.items():
        selected = next(
            (
                candidate
                for candidate in candidates
                if all(
                    field in mandatory_set
                    for field in _candidate_fields(candidate)
                )
            ),
            None,
        )
        if selected is None:
            missing[label] = [list(c) if not isinstance(c, str) else c for c in candidates]
        else:
            resolved[label] = (
                selected if isinstance(selected, str) else list(selected)
            )
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
            label: [
                list(candidate) if not isinstance(candidate, str) else candidate
                for candidate in candidates
            ]
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
