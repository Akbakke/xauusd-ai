"""Fail-closed contract for full model-native Entry sequence-integrity proofs.

The emitted train/validation rows can legitimately skip decision timestamps when
the causal M1 fill lifecycle is incomplete.  They therefore cannot always be
reconstructed from their emitted ``snap`` rows alone.  This contract proves the
stronger and correctly scoped property: each emitted sequence belongs to one
physical event chain, without inventing calendar bars across market closures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "entry_model_native_sequence_integrity_audit_v1"
REQUIRED_CHECKS = {
    "all_values_finite_float32": True,
    "every_seq_last_equals_snap_bit_identical": True,
    "timestamps_strictly_increasing": True,
    "timestamps_are_whole_m5_intervals": True,
    "every_emitted_pair_has_exact_physical_overlap": True,
    "physical_event_steps_do_not_exceed_elapsed_m5_intervals": True,
}
AUTHORITY = {
    "data_integrity_only": True,
    "data_reconstruction_only": False,
    "candidate": False,
    "test": False,
    "promotion": False,
    "paper": False,
    "live": False,
}
TRANSITION_SUMMARY_KEYS = {
    "pairs",
    "calendar_one_bar_pairs",
    "calendar_gap_pairs",
    "physical_one_bar_pairs",
    "physical_multi_bar_pairs",
    "calendar_elapsed_bars_total",
    "physical_event_bars_total",
    "nontrading_calendar_bars_total",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"[ENTRY_SEQUENCE_INTEGRITY_AUDIT_INVALID] {message}")


def _sha256(value: Any, *, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field}_sha256",
    )
    return value


def require_sequence_integrity_audit(
    value: Mapping[str, Any] | Any,
    *,
    expected_parquet_path: Path,
    expected_manifest_path: Path,
    expected_parquet_sha256: str,
    expected_manifest_sha256: str,
    expected_rows: int,
    expected_seq_len: int,
    expected_signal_dim: int,
) -> dict[str, Any]:
    """Validate one immutable train/validation sequence-integrity proof."""

    _require(isinstance(value, Mapping), "not_mapping")
    report = dict(value)
    expected_keys = {
        "schema_version",
        "decision",
        "created_utc",
        "parquet_path",
        "parquet_sha256",
        "manifest_path",
        "manifest_sha256",
        "rows",
        "sequence_shape",
        "snapshot_shape",
        "checks",
        "transition_summary",
        "sequence_event_chain_sha256",
        "authority",
    }
    _require(set(report) == expected_keys, "key_set")
    _require(report["schema_version"] == SCHEMA_VERSION, "schema_version")
    _require(report["decision"] == "PASS", "decision")
    _require(isinstance(report["created_utc"], str) and bool(report["created_utc"]), "created_utc")
    _require(report["parquet_path"] == str(expected_parquet_path), "parquet_path")
    _require(report["manifest_path"] == str(expected_manifest_path), "manifest_path")
    _require(_sha256(report["parquet_sha256"], field="parquet") == expected_parquet_sha256, "parquet_binding")
    _require(_sha256(report["manifest_sha256"], field="manifest") == expected_manifest_sha256, "manifest_binding")
    _require(type(report["rows"]) is int and report["rows"] == expected_rows, "rows")
    _require(
        report["sequence_shape"] == [expected_rows, expected_seq_len, expected_signal_dim],
        "sequence_shape",
    )
    _require(
        report["snapshot_shape"] == [expected_rows, expected_signal_dim],
        "snapshot_shape",
    )
    _require(report["checks"] == REQUIRED_CHECKS, "checks")
    _require(report["authority"] == AUTHORITY, "authority")
    _sha256(report["sequence_event_chain_sha256"], field="sequence_event_chain")

    summary = report["transition_summary"]
    _require(isinstance(summary, Mapping) and set(summary) == TRANSITION_SUMMARY_KEYS, "transition_summary_keys")
    for key in TRANSITION_SUMMARY_KEYS:
        _require(type(summary[key]) is int and summary[key] >= 0, f"transition_summary_{key}")
    pairs = expected_rows - 1
    _require(summary["pairs"] == pairs, "transition_pairs")
    _require(
        summary["calendar_one_bar_pairs"] + summary["calendar_gap_pairs"] == pairs,
        "calendar_pair_accounting",
    )
    _require(
        summary["physical_one_bar_pairs"] + summary["physical_multi_bar_pairs"] == pairs,
        "physical_pair_accounting",
    )
    _require(
        summary["calendar_elapsed_bars_total"] >= summary["physical_event_bars_total"] >= pairs,
        "event_step_accounting",
    )
    _require(
        summary["nontrading_calendar_bars_total"]
        == summary["calendar_elapsed_bars_total"] - summary["physical_event_bars_total"],
        "nontrading_calendar_accounting",
    )
    return report
