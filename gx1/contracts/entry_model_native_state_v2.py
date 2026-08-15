"""Immutable common-history state contract for model-native Entry and Exit.

Every split and replay starts feature construction at one declared causal
history boundary. No percentile artifact or row-level regime category is part
of this contract; raw ATR, spread and trend evidence remain continuous.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


MODEL_NATIVE_STATE_SCHEMA_VERSION = "model_native_state_contract_v7"
MODEL_NATIVE_HISTORY_MODE = "common_causal_history_no_split_reset_v1"

RETIRED_RANK_STATE_FIELDS = frozenset(
    {
        "rank_fit_start_utc",
        "rank_fit_end_utc",
        "rank_reference_npz",
        "rank_reference_npz_sha256",
        "rank_reference_sidecar_sha256",
        "rank_reference_schema_version",
        "rank_reference_sidecar_json",
        "rank_reference_fit_row_count",
        "rank_reference_fit_time_min",
        "rank_reference_fit_time_max",
        "rank_reference_source_parquet",
        "rank_reference_source_parquet_sha256",
        "rank_reference_model_source_market_identity",
        "rank_reference_fit_scope",
        "rank_transform",
        "post_fit_rows_in_rank_reference",
    }
)
STALE_STATE_CONTRACT_FIELDS = frozenset(
    {
        "frame_anchor_utc",
        "model_range_start_utc",
        "rank_reference_end_utc",
        "time_split_reference_split",
    }
)


def parse_utc(value: Any, *, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.to_datetime(value, utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError(
            f"MODEL_NATIVE_STATE_TIMESTAMP_INVALID: {field}={value!r}"
        ) from exc
    if pd.isna(timestamp):
        raise RuntimeError(
            f"MODEL_NATIVE_STATE_TIMESTAMP_INVALID: {field}={value!r}"
        )
    return pd.Timestamp(timestamp)


def validate_state_contract_metadata_v2(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("MODEL_NATIVE_STATE_CONTRACT_MISSING")
    data = dict(raw)
    if data.get("schema_version") != MODEL_NATIVE_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            "MODEL_NATIVE_STATE_SCHEMA_INVALID: "
            f"got={data.get('schema_version')!r} "
            f"expected={MODEL_NATIVE_STATE_SCHEMA_VERSION!r}"
        )
    retired = sorted(
        (RETIRED_RANK_STATE_FIELDS | STALE_STATE_CONTRACT_FIELDS) & set(data)
    )
    if retired:
        raise RuntimeError(
            f"MODEL_NATIVE_STATE_RETIRED_FIELDS_FORBIDDEN: {retired}"
        )
    required = {
        "feature_history_start_utc",
        "feature_history_mode",
        "split_reset_allowed",
        "runtime_rule_free",
        "entry_run_id",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"MODEL_NATIVE_STATE_FIELDS_MISSING: {missing}")
    if data["feature_history_mode"] != MODEL_NATIVE_HISTORY_MODE:
        raise RuntimeError("MODEL_NATIVE_STATE_HISTORY_MODE_INVALID")
    if data["split_reset_allowed"] is not False:
        raise RuntimeError("MODEL_NATIVE_STATE_SPLIT_RESET_FORBIDDEN")
    if data["runtime_rule_free"] is not True:
        raise RuntimeError("MODEL_NATIVE_STATE_RUNTIME_RULE_FREE_REQUIRED")
    if not str(data["entry_run_id"] or "").strip():
        raise RuntimeError("MODEL_NATIVE_STATE_EXPLICIT_RUN_ID_MISSING")
    parse_utc(
        data["feature_history_start_utc"],
        field="feature_history_start_utc",
    )
    return data


__all__ = [
    "MODEL_NATIVE_HISTORY_MODE",
    "MODEL_NATIVE_STATE_SCHEMA_VERSION",
    "RETIRED_RANK_STATE_FIELDS",
    "parse_utc",
    "validate_state_contract_metadata_v2",
]
