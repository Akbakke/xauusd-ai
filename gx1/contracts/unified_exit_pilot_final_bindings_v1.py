"""Final immutable normalization and first-state bindings for the v2 pilot."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_input_normalization_v1 import (
    require_input_normalization_contract,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    canonical_sha256,
    require_lifetime_summary_normalization,
)


COMPOSITE_NORMALIZATION_SCHEMA_VERSION = (
    "gx1_unified_exit_composite_train_normalization_v1"
)
SPLIT_SEQUENCE_BINDING_SCHEMA_VERSION = (
    "gx1_unified_exit_first_state_split_sequence_binding_v1"
)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_FINAL_{label}_SHA_INVALID")
    return value


def _clock(values: Sequence[Any], label: str) -> pd.DatetimeIndex:
    clock = pd.DatetimeIndex(
        pd.to_datetime(values, utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        clock.empty
        or clock.hasnans
        or not clock.is_unique
        or not clock.is_monotonic_increasing
    ):
        raise RuntimeError(f"UNIFIED_EXIT_FINAL_{label}_CLOCK_INVALID")
    return clock


def _clock_sha(clock: pd.DatetimeIndex) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(clock.asi8, dtype="<i8").tobytes()
    ).hexdigest()


def build_split_sequence_binding(
    *,
    split: str,
    entry_times: Sequence[Any],
    m1_times: Sequence[Any],
    successor_transition_counts: Sequence[int],
    child_admission_file_sha256: str,
    child_admission_witness_sha256: str,
    child_parquet_sha256: str,
    child_manifest_file_sha256: str,
    child_manifest_contract_sha256: str,
    m1_source_sha256: str,
    m1_manifest_file_sha256: str,
    closure_authority_file_sha256: str,
    closure_authority_sha256: str,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise RuntimeError("UNIFIED_EXIT_FINAL_SEQUENCE_SPLIT_INVALID")
    entry = _clock(entry_times, "ENTRY")
    m1 = _clock(m1_times, "M1")
    counts = np.ascontiguousarray(successor_transition_counts, dtype="<i8")
    if counts.shape != (len(entry),) or np.any(counts < 1):
        raise RuntimeError("UNIFIED_EXIT_FINAL_SEQUENCE_COUNTS_INVALID")
    first_state = entry.asi8 + 300_000_000_000
    positions = np.searchsorted(m1.asi8, first_state)
    if (
        np.any(positions >= len(m1))
        or not np.array_equal(m1.asi8[positions], first_state)
        or np.any(positions + counts >= len(m1))
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_SEQUENCE_RANGE_INVALID")
    bindings = {
        key: _sha(value, key.upper())
        for key, value in {
            "child_admission_file": child_admission_file_sha256,
            "child_admission_witness": child_admission_witness_sha256,
            "child_parquet": child_parquet_sha256,
            "child_manifest_file": child_manifest_file_sha256,
            "child_manifest_contract": child_manifest_contract_sha256,
            "m1_source": m1_source_sha256,
            "m1_manifest_file": m1_manifest_file_sha256,
            "closure_authority_file": closure_authority_file_sha256,
            "closure_authority": closure_authority_sha256,
        }.items()
    }
    stream = np.column_stack(
        [
            np.arange(len(entry), dtype="<i8"),
            entry.asi8,
            positions.astype("<i8"),
            first_state,
            counts,
        ]
    ).astype("<i8", copy=False)
    value = {
        "schema_version": SPLIT_SEQUENCE_BINDING_SCHEMA_VERSION,
        "decision": "PASS",
        "split": split,
        "entry_row_count": len(entry),
        "entry_clock_sha256": _clock_sha(entry),
        "m1_row_count": len(m1),
        "m1_clock_sha256": _clock_sha(m1),
        "first_state_clock_sha256": hashlib.sha256(
            np.ascontiguousarray(first_state, dtype="<i8").tobytes()
        ).hexdigest(),
        "first_state_m1_position_sha256": hashlib.sha256(
            np.ascontiguousarray(positions, dtype="<i8").tobytes()
        ).hexdigest(),
        "successor_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "successor_transition_total": int(counts.sum(dtype=np.int64)),
        "sequence_stream_sha256": hashlib.sha256(stream.tobytes()).hexdigest(),
        "split_boundary_rule": "successor_range_stops_before_child_m1_end",
        "bindings": bindings,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["binding_sha256"] = canonical_sha256(value)
    return value


def require_split_sequence_binding(
    value: Mapping[str, Any],
    *,
    expected_split: str,
    expected_entry_rows: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_FINAL_SEQUENCE_BINDING_INVALID")
    data = dict(value)
    claimed = data.pop("binding_sha256", None)
    bindings = value.get("bindings")
    if (
        value.get("schema_version") != SPLIT_SEQUENCE_BINDING_SCHEMA_VERSION
        or value.get("decision") != "PASS"
        or value.get("split") != expected_split
        or value.get("entry_row_count") != expected_entry_rows
        or value.get("successor_transition_total", 0) < expected_entry_rows
        or value.get("split_boundary_rule")
        != "successor_range_stops_before_child_m1_end"
        or value.get("val_fit_rows") != 0
        or value.get("test_fit_rows") != 0
        or value.get("test_accessed") is not False
        or not isinstance(bindings, Mapping)
        or set(bindings)
        != {
            "child_admission_file",
            "child_admission_witness",
            "child_parquet",
            "child_manifest_file",
            "child_manifest_contract",
            "m1_source",
            "m1_manifest_file",
            "closure_authority_file",
            "closure_authority",
        }
        or any(_sha(item, "SEQUENCE_BINDING") != item for item in bindings.values())
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_SEQUENCE_BINDING_INVALID")
    return dict(value)


def build_composite_normalization_binding(
    *,
    base_artifact: Mapping[str, Any],
    base_path: str,
    base_file_sha256: str,
    summary_normalization: Mapping[str, Any],
    summary_manifest_path: str,
    summary_manifest_file_sha256: str,
    summary_manifest_sha256: str,
) -> dict[str, Any]:
    base = dict(base_artifact)
    contract = base.get("contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("UNIFIED_EXIT_FINAL_BASE_NORMALIZATION_INVALID")
    expected_names = {
        str(name): list(surface.get("field_names") or [])
        for name, surface in contract.get("surfaces", {}).items()
    }
    checked_base = require_input_normalization_contract(
        contract,
        expected_field_names=expected_names,
        expected_ctx_cat_names=list(contract.get("ctx_cat", {}).get("field_names") or []),
    )
    if (
        base.get("schema_version")
        != "gx1_unified_exit_pilot_base_normalization_v1"
        or base.get("decision") != "PASS"
        or base.get("contract_sha256") != checked_base["contract_sha256"]
        or base.get("val_fit_rows") != 0
        or base.get("test_fit_rows") != 0
        or base.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_BASE_NORMALIZATION_INVALID")
    summary = require_lifetime_summary_normalization(summary_normalization)
    value = {
        "schema_version": COMPOSITE_NORMALIZATION_SCHEMA_VERSION,
        "decision": "PASS",
        "fit_scope": "train_only_base_plus_outcome_blind_lifetime_summary",
        "base_feature_normalization": {
            "path": str(base_path),
            "file_sha256": _sha(base_file_sha256, "BASE_FILE"),
            "contract_sha256": checked_base["contract_sha256"],
            "artifact": base,
        },
        "lifetime_summary_normalization": summary,
        "summary_fit_manifest": {
            "path": str(summary_manifest_path),
            "file_sha256": _sha(summary_manifest_file_sha256, "SUMMARY_FILE"),
            "manifest_sha256": _sha(summary_manifest_sha256, "SUMMARY_MANIFEST"),
        },
        "val_mode": "apply_frozen_train_transforms_only",
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    value["composite_normalization_sha256"] = canonical_sha256(value)
    return value


def require_composite_normalization_binding(
    value: Mapping[str, Any],
    *,
    expected_base_contract_sha256: str | None = None,
    expected_summary_normalization_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_FINAL_COMPOSITE_NORMALIZATION_INVALID")
    data = dict(value)
    claimed = data.pop("composite_normalization_sha256", None)
    base_binding = value.get("base_feature_normalization")
    if not isinstance(base_binding, Mapping):
        raise RuntimeError("UNIFIED_EXIT_FINAL_COMPOSITE_NORMALIZATION_INVALID")
    rebuilt = build_composite_normalization_binding(
        base_artifact=base_binding.get("artifact", {}),
        base_path=str(base_binding.get("path", "")),
        base_file_sha256=base_binding.get("file_sha256"),
        summary_normalization=value.get("lifetime_summary_normalization", {}),
        summary_manifest_path=str(value.get("summary_fit_manifest", {}).get("path", "")),
        summary_manifest_file_sha256=value.get("summary_fit_manifest", {}).get("file_sha256"),
        summary_manifest_sha256=value.get("summary_fit_manifest", {}).get("manifest_sha256"),
    )
    if (
        value.get("schema_version") != COMPOSITE_NORMALIZATION_SCHEMA_VERSION
        or value.get("decision") != "PASS"
        or value.get("val_fit_rows") != 0
        or value.get("test_fit_rows") != 0
        or value.get("test_accessed") is not False
        or claimed != canonical_sha256(data)
        or rebuilt != dict(value)
        or (
            expected_base_contract_sha256 is not None
            and base_binding.get("contract_sha256")
            != _sha(expected_base_contract_sha256, "EXPECTED_BASE")
        )
        or (
            expected_summary_normalization_sha256 is not None
            and value.get("lifetime_summary_normalization", {}).get(
                "normalization_sha256"
            )
            != _sha(expected_summary_normalization_sha256, "EXPECTED_SUMMARY")
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_FINAL_COMPOSITE_NORMALIZATION_INVALID")
    return dict(value)


__all__ = [
    "COMPOSITE_NORMALIZATION_SCHEMA_VERSION",
    "SPLIT_SEQUENCE_BINDING_SCHEMA_VERSION",
    "build_composite_normalization_binding",
    "build_split_sequence_binding",
    "require_composite_normalization_binding",
    "require_split_sequence_binding",
]
