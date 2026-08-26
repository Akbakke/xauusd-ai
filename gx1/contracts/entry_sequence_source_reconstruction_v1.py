"""Fail-closed authority for source-backed Entry sequence reconstruction.

The supervised Entry parquet intentionally omits timestamps whose causal M1
episode is unavailable.  Its ``seq`` rows are nevertheless materialised from
the immutable M5 feature surface before that label filter is applied.  This
contract permits a trainer to recover each window from that feature surface
only after a full byte-for-byte audit has bound the exact split, split manifest
and feature-surface bytes together.

It is a storage representation authority only.  It neither changes a feature,
target, split, nor grants candidate, TEST, paper or live authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "entry_model_native_sequence_source_reconstruction_audit_v1"
FEATURE_SURFACE_BINDING_KEYS = frozenset(
    {
        "dataset_run_id",
        "inline_split_recomputation",
        "manifest_path",
        "manifest_sha256",
        "pair_generation_id",
        "path",
        "rows",
        "schema_version",
        "sha256",
        "signal_manifest_sha256",
        "time_alignment",
    }
)
REQUIRED_CHECKS = {
    "all_values_finite_float32": True,
    "source_feature_surface_hash_matches_split_manifest": True,
    "source_feature_surface_manifest_hash_matches_split_manifest": True,
    "source_feature_surface_schema_and_dimensions_exact": True,
    "source_timestamps_strictly_increasing": True,
    "every_emitted_timestamp_maps_exactly_once_to_feature_surface": True,
    "every_sequence_equals_exact_source_surface_history_bit_identical": True,
    "every_snapshot_equals_exact_source_surface_current_row_bit_identical": True,
}
AUTHORITY = {
    "data_reconstruction_only": True,
    "candidate": False,
    "test": False,
    "promotion": False,
    "paper": False,
    "live": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_AUDIT_INVALID] {message}"
        )


def _sha256(value: Any, *, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field}_sha256",
    )
    return value


def feature_surface_binding_from_split_manifest(
    manifest: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Return the one M5 surface binding declared by a split manifest."""

    _require(isinstance(manifest, Mapping), "split_manifest_not_mapping")
    # The canonical split writer records the emitted signal bridge as an
    # immutable build ``extra`` (the feature contract itself owns ctx/schema
    # metadata).  Do not accept an alternate nested location: that would make
    # the audit source ambiguous.
    extra = manifest.get("extra")
    signal_bridge = extra.get("signal_bridge") if isinstance(extra, Mapping) else None
    extension = (
        signal_bridge.get("seq_structure_extension_v1")
        if isinstance(signal_bridge, Mapping)
        else None
    )
    binding = extension.get("feature_surface") if isinstance(extension, Mapping) else None
    _require(isinstance(binding, Mapping), "feature_surface_binding_missing")
    binding = dict(binding)
    _require(set(binding) == FEATURE_SURFACE_BINDING_KEYS, "feature_surface_binding_keys")
    _require(binding.get("inline_split_recomputation") is False, "inline_recomputation")
    _require(
        binding.get("time_alignment") == "exact_entry_m5_source_timeline",
        "time_alignment",
    )
    for field in ("path", "manifest_path", "dataset_run_id", "pair_generation_id", "schema_version"):
        _require(isinstance(binding.get(field), str) and bool(binding[field]), field)
    for field in ("sha256", "manifest_sha256", "signal_manifest_sha256"):
        _sha256(binding.get(field), field=field)
    _require(type(binding.get("rows")) is int and int(binding["rows"]) > 0, "rows")
    return binding


def require_sequence_source_reconstruction_audit(
    value: Mapping[str, Any] | Any,
    *,
    expected_parquet_path: Path,
    expected_manifest_path: Path,
    expected_parquet_sha256: str,
    expected_manifest_sha256: str,
    expected_feature_surface: Mapping[str, Any],
    expected_rows: int,
    expected_seq_len: int,
    expected_signal_dim: int,
) -> dict[str, Any]:
    """Validate one immutable source-reconstruction proof for a split."""

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
        "feature_surface_path",
        "feature_surface_sha256",
        "feature_surface_manifest_path",
        "feature_surface_manifest_sha256",
        "feature_surface_rows",
        "rows",
        "sequence_shape",
        "snapshot_shape",
        "checks",
        "sequence_source_chain_sha256",
        "authority",
    }
    _require(set(report) == expected_keys, "key_set")
    _require(report.get("schema_version") == SCHEMA_VERSION, "schema_version")
    _require(report.get("decision") == "PASS", "decision")
    _require(isinstance(report.get("created_utc"), str) and bool(report["created_utc"]), "created_utc")
    _require(report.get("parquet_path") == str(expected_parquet_path), "parquet_path")
    _require(report.get("manifest_path") == str(expected_manifest_path), "manifest_path")
    _require(
        _sha256(report.get("parquet_sha256"), field="parquet")
        == expected_parquet_sha256,
        "parquet_binding",
    )
    _require(
        _sha256(report.get("manifest_sha256"), field="manifest")
        == expected_manifest_sha256,
        "manifest_binding",
    )
    surface = feature_surface_binding_from_split_manifest(expected_feature_surface)
    _require(report.get("feature_surface_path") == surface["path"], "feature_surface_path")
    _require(
        _sha256(report.get("feature_surface_sha256"), field="feature_surface")
        == surface["sha256"],
        "feature_surface_binding",
    )
    _require(
        report.get("feature_surface_manifest_path") == surface["manifest_path"],
        "feature_surface_manifest_path",
    )
    _require(
        _sha256(
            report.get("feature_surface_manifest_sha256"),
            field="feature_surface_manifest",
        )
        == surface["manifest_sha256"],
        "feature_surface_manifest_binding",
    )
    _require(
        type(report.get("feature_surface_rows")) is int
        and report["feature_surface_rows"] == surface["rows"],
        "feature_surface_rows",
    )
    _require(type(report.get("rows")) is int and report["rows"] == expected_rows, "rows")
    _require(
        report.get("sequence_shape")
        == [expected_rows, expected_seq_len, expected_signal_dim],
        "sequence_shape",
    )
    _require(
        report.get("snapshot_shape") == [expected_rows, expected_signal_dim],
        "snapshot_shape",
    )
    _require(report.get("checks") == REQUIRED_CHECKS, "checks")
    _require(report.get("authority") == AUTHORITY, "authority")
    _sha256(report.get("sequence_source_chain_sha256"), field="sequence_source_chain")
    return report
