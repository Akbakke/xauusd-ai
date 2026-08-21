"""Exact non-executable readiness contract for model-native XAU Entry.

This module owns only immutable metadata and pure validation/fingerprint
helpers.  It never discovers ``latest`` artifacts, starts a trainer, selects a
direction, mutates control state, or writes reports.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_fitted_q_v1 import (
    entry_fitted_q_contract,
    entry_fitted_q_production_economics_readiness,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
)


MODEL_NATIVE_READINESS_SCHEMA_VERSION = "entry_model_native_readiness_v5"
MODEL_NATIVE_REQUIRED_SPECIALISTS = tuple(MODEL_NATIVE_TRAINING_SPECIALISTS)
MODEL_NATIVE_BASE_ACTIVE_HEADS = (
    "entry_action_q",
    "position_size",
    "dip",
    "forecast",
    "timing",
    "tail_risk",
    "vol_forecast",
)
MODEL_NATIVE_EXTRA_ACTIVE_HEADS = (
    "side_mae",
    "trendline_event",
)
MODEL_NATIVE_ACTIVE_HEADS = (
    *MODEL_NATIVE_BASE_ACTIVE_HEADS,
    *MODEL_NATIVE_EXTRA_ACTIVE_HEADS,
)
# A head is blocked because it is RETIRED, never because it appears in the
# pre-V30 fusion list: `SPECIALIST_FUSION_ACTIVE_HEADS` predates the V30
# surface and still names heads this contract declares active. Subtracting the
# current active tuple keeps the two sets disjoint by construction, so serve
# parity can never both require and forbid the same column.
_MODEL_NATIVE_CANDIDATE_BLOCKED_HEADS = (
    *SPECIALIST_FUSION_BLOCKED_HEADS,
    *SPECIALIST_FUSION_ACTIVE_HEADS,
    "trade_side_hierarchy",
    "side_validity",
    "path_quality",
    "mfe_first_n",
    "bad_path",
    "clean_edge",
    "survival",
    "path_quality_log_var",
)
MODEL_NATIVE_BLOCKED_HEADS = tuple(
    dict.fromkeys(
        name
        for name in _MODEL_NATIVE_CANDIDATE_BLOCKED_HEADS
        if name not in set(MODEL_NATIVE_ACTIVE_HEADS)
    )
)
if SPECIALIST_FUSION_ACTIVE_HEADS != MODEL_NATIVE_ACTIVE_HEADS:
    raise RuntimeError("MODEL_NATIVE_SPECIALIST_ACTIVE_HEAD_SPLIT_BRAIN")
if SPECIALIST_FUSION_BLOCKED_HEADS != MODEL_NATIVE_BLOCKED_HEADS:
    raise RuntimeError("MODEL_NATIVE_SPECIALIST_BLOCKED_HEAD_SPLIT_BRAIN")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_json(MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT).encode("utf-8")
).hexdigest()


def model_native_readiness_contract_metadata() -> dict[str, Any]:
    """Return a fresh exact seq513/head/specialist readiness declaration."""

    return {
        "schema_version": MODEL_NATIVE_READINESS_SCHEMA_VERSION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": "retired_entry_fitted_q_raw_bps",
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "required_specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "base_active_heads": list(MODEL_NATIVE_BASE_ACTIVE_HEADS),
        "extra_active_heads": list(MODEL_NATIVE_EXTRA_ACTIVE_HEADS),
        "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
        "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
        "specialist_model_contract_sha256": (
            MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT_SHA256
        ),
        "entry_fitted_q": entry_fitted_q_contract(),
        "entry_fitted_q_production_economics": (
            entry_fitted_q_production_economics_readiness()
        ),
        "candidate_ready_allowed": False,
        "bundle_serving_admission_allowed": False,
        "secondary_direction_authority_allowed": False,
        "mutable_latest_evidence_allowed": False,
    }


def require_model_native_readiness_contract(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Require exact equality; partial or forward-filled contracts are invalid."""

    expected = model_native_readiness_contract_metadata()
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_MODEL_NATIVE_READINESS_CONTRACT_MISSING]")
    observed = dict(value)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected).intersection(observed)
            if observed[key] != expected[key]
        )
        raise RuntimeError(
            f"[{context}_MODEL_NATIVE_READINESS_CONTRACT_INVALID] "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )
    return observed


def readiness_check(
    name: str,
    condition: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"name": str(name), "ok": bool(condition), "details": details or {}}


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint a regular non-symlink file without trusting filesystem mtime."""

    path = Path(path).expanduser().absolute()
    exists = path.exists()
    regular_file = path.is_file() and not path.is_symlink()
    size_bytes = int(path.stat().st_size) if regular_file else None
    return {
        "path": str(path),
        "exists": bool(exists),
        "regular_file": bool(regular_file),
        "size_bytes": size_bytes,
        "sha256": sha256_file(path) if regular_file else None,
    }


def artifact_fingerprints(
    artifacts: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    return {
        str(name): artifact_fingerprint(Path(path))
        for name, path in artifacts.items()
    }


def artifact_fingerprint_checks(
    fingerprints: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        readiness_check(
            "readiness records content-addressed regular-file evidence",
            bool(fingerprints)
            and all(
                row.get("exists") is True
                and row.get("regular_file") is True
                and int(row.get("size_bytes") or 0) > 0
                and isinstance(row.get("sha256"), str)
                and len(str(row.get("sha256"))) == 64
                for row in fingerprints.values()
            ),
            {"artifact_fingerprints": dict(fingerprints)},
        )
    ]
