"""Immutable XAU_USD tape identity and lineage contracts.

Instrument identity is never inferred from a dataset or directory name.  The
base repair binds the canonical M1/M5 manifests and their hashes.  A current
snapshot then binds that base repair, every output-year hash, and the exact
collector snapshots used for its live tail.  Consumers validate the complete
chain before admitting model data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


XAU_INSTRUMENT = "XAU_USD"
BASE_REPAIR_SCHEMA = "m5_tape_dec2024_repair_manifest_v2"
CURRENT_SNAPSHOT_SCHEMA = "m5_tape_current_snapshot_v2"
BASE_REPAIR_METHOD = "recompute_window_from_canonical_m1_drop_unbacked_bars"
CURRENT_SNAPSHOT_METHOD = "immutable_live_collector_snapshot_exact_m5_overlap"
CANONICAL_M5_SOURCE_SCHEMA = "xau_canonical_m5_source_v1"
CANONICAL_M5_PRODUCER_OWNER = (
    "gx1.scripts.backfill_xauusd_m5_from_oanda.fetch_candles_bid_ask"
)
CANONICAL_M5_CLOSURE_CONTRACT = (
    "oanda_complete_true_source_absence_no_synthesis_v1"
)
CANONICAL_M5_REQUIRED_COLUMNS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "volume",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}_MISSING_OR_SYMLINK: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{label}_INVALID_JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}_OBJECT_REQUIRED: {path}")
    return value


def _root(raw: Any, *, label: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise RuntimeError(f"{label}_MISSING")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{label}_NOT_ABSOLUTE: {path}")
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label}_MISSING_OR_SYMLINK: {path}")
    resolved = path.resolve()
    if str(path) != str(resolved):
        raise RuntimeError(f"{label}_NOT_CANONICAL: raw={path} resolved={resolved}")
    return resolved


def _path(raw: Any, *, label: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise RuntimeError(f"{label}_MISSING")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{label}_NOT_ABSOLUTE: {path}")
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}_MISSING_OR_SYMLINK: {path}")
    resolved = path.resolve()
    if str(path) != str(resolved):
        raise RuntimeError(f"{label}_NOT_CANONICAL: raw={path} resolved={resolved}")
    return resolved


def _normalized_instrument(raw: Any) -> str:
    value = str(raw or "").strip().upper().replace("_", "")
    if value != "XAUUSD":
        raise RuntimeError(
            f"XAU_TAPE_INSTRUMENT_MISMATCH: expected={XAU_INSTRUMENT!r} observed={raw!r}"
        )
    return XAU_INSTRUMENT


def _validate_canonical_m5_source_contract(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require one hash-bound native-M5 owner and source-defined closure."""

    exact = {
        "schema_version": CANONICAL_M5_SOURCE_SCHEMA,
        "producer_owner": CANONICAL_M5_PRODUCER_OWNER,
        "source_kind": "oanda_native_mba_candles",
        "source_granularity": "M5",
        "prices": "MBA",
        "timestamp_semantics": "bar_start_utc",
        "bar_duration_seconds": 300,
        "decision_available_offset_seconds": 300,
        "completion_field": "complete",
        "completion_value": True,
        "market_closure_contract": CANONICAL_M5_CLOSURE_CONTRACT,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SOURCE_POLICY_MISMATCH: "
                f"field={key!r} expected={expected!r} "
                f"observed={manifest.get(key)!r}"
            )
    if manifest.get("schema_required_cols") != list(
        CANONICAL_M5_REQUIRED_COLUMNS
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SCHEMA_CONTRACT_MISMATCH")
    if manifest.get("schema_optional_cols") != []:
        raise RuntimeError("XAU_CANONICAL_M5_OPTIONAL_SCHEMA_FORBIDDEN")
    if "slippage_bps" in manifest.get("schema_required_cols", []):
        raise RuntimeError("XAU_CANONICAL_M5_SLIPPAGE_NOT_SOURCE_OBSERVED")

    year_hashes = manifest.get("year_sha256")
    if not isinstance(year_hashes, dict) or not year_hashes:
        raise RuntimeError("XAU_CANONICAL_M5_YEAR_HASHES_MISSING")
    year_rows = manifest.get("year_rows")
    if not isinstance(year_rows, dict) or set(year_rows) != set(year_hashes):
        raise RuntimeError("XAU_CANONICAL_M5_YEAR_ROWS_MISSING")
    actual_parts = sorted(root.glob("year=*/part-000.parquet"))
    actual_keys = [part.parent.name for part in actual_parts]
    if actual_keys != sorted(year_hashes):
        raise RuntimeError(
            "XAU_CANONICAL_M5_YEAR_SET_MISMATCH: "
            f"manifest={sorted(year_hashes)} filesystem={actual_keys}"
        )
    for part in actual_parts:
        expected_hash = year_hashes.get(part.parent.name)
        observed_hash = sha256_file(part)
        if expected_hash != observed_hash:
            raise RuntimeError(
                "XAU_CANONICAL_M5_YEAR_HASH_MISMATCH: "
                f"year={part.parent.name} expected={expected_hash!r} "
                f"observed={observed_hash!r}"
            )
        try:
            metadata = pq.ParquetFile(part)
            observed_columns = metadata.schema_arrow.names
            observed_rows = int(metadata.metadata.num_rows)
        except Exception as exc:
            raise RuntimeError(
                f"XAU_CANONICAL_M5_PARQUET_INVALID: {part}"
            ) from exc
        if observed_columns != list(CANONICAL_M5_REQUIRED_COLUMNS):
            raise RuntimeError(
                "XAU_CANONICAL_M5_PARQUET_SCHEMA_MISMATCH: "
                f"year={part.parent.name} observed={observed_columns}"
            )
        expected_rows = year_rows.get(part.parent.name)
        if (
            isinstance(expected_rows, bool)
            or not isinstance(expected_rows, int)
            or expected_rows <= 0
            or expected_rows != observed_rows
        ):
            raise RuntimeError(
                "XAU_CANONICAL_M5_YEAR_ROWS_MISMATCH: "
                f"year={part.parent.name} expected={expected_rows!r} "
                f"observed={observed_rows}"
            )
    return {
        **exact,
        "schema_required_cols": list(CANONICAL_M5_REQUIRED_COLUMNS),
        "schema_optional_cols": [],
        "year_sha256": dict(sorted(year_hashes.items())),
        "year_rows": dict(sorted(year_rows.items())),
    }


def canonical_xau_source_descriptor_v1(
    root: Path | str,
    *,
    timeframe: str,
) -> dict[str, Any]:
    """Validate one canonical source manifest and return its hash-bound identity."""

    canonical_root = _root(root, label=f"XAU_CANONICAL_{timeframe.upper()}_ROOT")
    manifest_path = _path(
        canonical_root / "MANIFEST.json",
        label=f"XAU_CANONICAL_{timeframe.upper()}_MANIFEST",
    )
    manifest = _json(
        manifest_path,
        label=f"XAU_CANONICAL_{timeframe.upper()}_MANIFEST",
    )
    instrument = _normalized_instrument(manifest.get("instrument"))
    expected_timeframe = str(timeframe).strip().upper()
    observed_timeframe = str(manifest.get("timeframe") or "").strip().upper()
    if observed_timeframe != expected_timeframe:
        raise RuntimeError(
            "XAU_CANONICAL_TIMEFRAME_MISMATCH: "
            f"expected={expected_timeframe!r} observed={manifest.get('timeframe')!r}"
        )
    declared_root = _root(
        manifest.get("out_root"),
        label=f"XAU_CANONICAL_{expected_timeframe}_DECLARED_ROOT",
    )
    if declared_root != canonical_root:
        raise RuntimeError(
            "XAU_CANONICAL_DECLARED_ROOT_MISMATCH: "
            f"manifest={declared_root} input={canonical_root}"
        )
    descriptor = {
        "root": str(canonical_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "instrument": instrument,
        "instrument_observed": str(manifest.get("instrument")),
        "timeframe": expected_timeframe,
    }
    if expected_timeframe == "M5":
        descriptor.update(
            _validate_canonical_m5_source_contract(canonical_root, manifest)
        )
    return descriptor


def _validate_source_descriptor(
    raw: Any,
    *,
    timeframe: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"XAU_TAPE_CANONICAL_{timeframe}_DESCRIPTOR_MISSING")
    observed = canonical_xau_source_descriptor_v1(
        raw.get("root"),
        timeframe=timeframe,
    )
    keys = [
        "root",
        "manifest_path",
        "manifest_sha256",
        "instrument",
        "instrument_observed",
        "timeframe",
    ]
    if timeframe == "M5":
        keys.extend(
            [
                "schema_version",
                "producer_owner",
                "source_kind",
                "source_granularity",
                "prices",
                "timestamp_semantics",
                "bar_duration_seconds",
                "decision_available_offset_seconds",
                "completion_field",
                "completion_value",
                "market_closure_contract",
                "schema_required_cols",
                "schema_optional_cols",
                "year_sha256",
                "year_rows",
            ]
        )
    for key in keys:
        if raw.get(key) != observed[key]:
            raise RuntimeError(
                f"XAU_TAPE_CANONICAL_{timeframe}_{key.upper()}_MISMATCH: "
                f"declared={raw.get(key)!r} observed={observed[key]!r}"
            )
    return observed


def _validate_years(
    tape_root: Path,
    years: Any,
    *,
    label: str,
) -> dict[str, str]:
    if not isinstance(years, dict) or not years:
        raise RuntimeError(f"{label}_YEARS_INVALID")
    parsed: list[int] = []
    for key, metadata in years.items():
        if not str(key).startswith("year=") or not isinstance(metadata, dict):
            raise RuntimeError(f"{label}_YEARS_INVALID")
        try:
            parsed.append(int(str(key).split("=", 1)[1]))
        except ValueError as exc:
            raise RuntimeError(f"{label}_YEARS_INVALID") from exc
    parsed.sort()
    if parsed != list(range(parsed[0], parsed[-1] + 1)):
        raise RuntimeError(f"{label}_YEARS_NOT_CONTIGUOUS: {parsed}")
    hashes: dict[str, str] = {}
    for year in parsed:
        key = f"year={year}"
        part = _path(
            tape_root / key / "part-000.parquet",
            label=f"{label}_{year}_PART",
        )
        digest = sha256_file(part)
        declared = years[key].get("output_sha256")
        if declared != digest:
            raise RuntimeError(
                f"{label}_{year}_HASH_MISMATCH: declared={declared!r} observed={digest!r}"
            )
        hashes[key] = digest
    return hashes


def _validate_base_repair(
    tape_root: Path,
    manifest: dict[str, Any],
    *,
    expected_run_id: str,
) -> dict[str, Any]:
    if manifest.get("schema_version") != BASE_REPAIR_SCHEMA:
        raise RuntimeError(
            "XAU_TAPE_BASE_SCHEMA_MISMATCH: "
            f"expected={BASE_REPAIR_SCHEMA!r} observed={manifest.get('schema_version')!r}"
        )
    if manifest.get("instrument") != XAU_INSTRUMENT:
        raise RuntimeError(
            "XAU_TAPE_BASE_INSTRUMENT_MISMATCH: "
            f"expected={XAU_INSTRUMENT!r} observed={manifest.get('instrument')!r}"
        )
    if manifest.get("explicit_vedtak_id") != expected_run_id:
        raise RuntimeError("XAU_TAPE_BASE_RUN_ID_MISMATCH")
    if manifest.get("method") != BASE_REPAIR_METHOD:
        raise RuntimeError("XAU_TAPE_BASE_METHOD_MISMATCH")
    if manifest.get("geometry_bad_total_after") != 0:
        raise RuntimeError("XAU_TAPE_BASE_GEOMETRY_NOT_PROVEN")
    sources = manifest.get("canonical_sources")
    if not isinstance(sources, dict):
        raise RuntimeError("XAU_TAPE_BASE_CANONICAL_SOURCES_MISSING")
    m5 = _validate_source_descriptor(sources.get("m5"), timeframe="M5")
    m1 = _validate_source_descriptor(sources.get("m1"), timeframe="M1")
    if manifest.get("m5_tape_root") != m5["root"]:
        raise RuntimeError("XAU_TAPE_BASE_M5_ROOT_MISMATCH")
    if manifest.get("m1_tape_root") != m1["root"]:
        raise RuntimeError("XAU_TAPE_BASE_M1_ROOT_MISMATCH")
    year_hashes = _validate_years(tape_root, manifest.get("years"), label="XAU_TAPE_BASE")
    return {
        "schema_version": BASE_REPAIR_SCHEMA,
        "instrument": XAU_INSTRUMENT,
        "entry_run_id": expected_run_id,
        "tape_root": str(tape_root),
        "manifest_path": str(tape_root / "REPAIR_MANIFEST.json"),
        "manifest_sha256": sha256_file(tape_root / "REPAIR_MANIFEST.json"),
        "year_sha256": year_hashes,
        "canonical_sources": {"m5": m5, "m1": m1},
    }


def validate_xau_tape_provenance_v1(
    tape_root: Path | str,
    *,
    expected_run_id: str,
    require_current: bool,
) -> dict[str, Any]:
    """Validate and summarize a complete immutable XAU tape lineage."""

    run_id = str(expected_run_id or "").strip()
    if not run_id:
        raise RuntimeError("XAU_TAPE_EXPECTED_RUN_ID_MISSING")
    root = _root(tape_root, label="XAU_TAPE_ROOT")
    manifest_path = _path(root / "REPAIR_MANIFEST.json", label="XAU_TAPE_MANIFEST")
    manifest = _json(manifest_path, label="XAU_TAPE_MANIFEST")
    schema = manifest.get("schema_version")
    if schema == BASE_REPAIR_SCHEMA:
        if require_current:
            raise RuntimeError(
                f"XAU_TAPE_CURRENT_SNAPSHOT_REQUIRED: observed={BASE_REPAIR_SCHEMA!r}"
            )
        return _validate_base_repair(root, manifest, expected_run_id=run_id)
    if schema != CURRENT_SNAPSHOT_SCHEMA:
        raise RuntimeError(
            "XAU_TAPE_SCHEMA_MISMATCH: "
            f"expected={CURRENT_SNAPSHOT_SCHEMA!r} observed={schema!r}"
        )
    if manifest.get("instrument") != XAU_INSTRUMENT:
        raise RuntimeError(
            "XAU_TAPE_CURRENT_INSTRUMENT_MISMATCH: "
            f"expected={XAU_INSTRUMENT!r} observed={manifest.get('instrument')!r}"
        )
    if manifest.get("entry_run_id") != run_id:
        raise RuntimeError("XAU_TAPE_CURRENT_RUN_ID_MISMATCH")
    if manifest.get("method") != CURRENT_SNAPSHOT_METHOD:
        raise RuntimeError("XAU_TAPE_CURRENT_METHOD_MISMATCH")
    if manifest.get("overlap_exact") is not True:
        raise RuntimeError("XAU_TAPE_CURRENT_EXACT_OVERLAP_NOT_PROVEN")
    if manifest.get("geometry_bad_total_after") != 0:
        raise RuntimeError("XAU_TAPE_CURRENT_GEOMETRY_NOT_PROVEN")

    base_root = _root(manifest.get("base_tape_root"), label="XAU_TAPE_BASE_ROOT")
    base_manifest_path = _path(
        manifest.get("base_manifest_path"),
        label="XAU_TAPE_BASE_MANIFEST",
    )
    if base_manifest_path != base_root / "REPAIR_MANIFEST.json":
        raise RuntimeError("XAU_TAPE_BASE_MANIFEST_PATH_MISMATCH")
    base_manifest_sha = sha256_file(base_manifest_path)
    if manifest.get("base_manifest_sha256") != base_manifest_sha:
        raise RuntimeError("XAU_TAPE_BASE_MANIFEST_HASH_MISMATCH")
    base_manifest = _json(base_manifest_path, label="XAU_TAPE_BASE_MANIFEST")
    base_proof = _validate_base_repair(
        base_root,
        base_manifest,
        expected_run_id=run_id,
    )
    if manifest.get("base_year_sha256") != base_proof["year_sha256"]:
        raise RuntimeError("XAU_TAPE_BASE_YEAR_HASH_SET_MISMATCH")

    year_hashes = _validate_years(root, manifest.get("years"), label="XAU_TAPE_CURRENT")
    if set(year_hashes) != set(base_proof["year_sha256"]):
        raise RuntimeError("XAU_TAPE_CURRENT_YEAR_SET_MISMATCH")

    overlap = manifest.get("overlap_proof")
    if not isinstance(overlap, dict):
        raise RuntimeError("XAU_TAPE_CURRENT_OVERLAP_PROOF_MISSING")
    if int(overlap.get("rows") or 0) < 1 or float(overlap.get("max_abs_diff") or 0.0) != 0.0:
        raise RuntimeError("XAU_TAPE_CURRENT_OVERLAP_PROOF_INVALID")
    if int(overlap.get("new_tail_rows") or 0) < 1:
        raise RuntimeError("XAU_TAPE_CURRENT_NEW_TAIL_NOT_PROVEN")

    snapshots: dict[str, str] = {}
    collector_sources = manifest.get("collector_sources")
    if not isinstance(collector_sources, list) or not collector_sources:
        raise RuntimeError("XAU_TAPE_CURRENT_COLLECTOR_SOURCES_MISSING")
    snapshot_root = root / "collector_snapshot"
    for index, source in enumerate(collector_sources):
        if not isinstance(source, dict):
            raise RuntimeError("XAU_TAPE_CURRENT_COLLECTOR_SOURCE_INVALID")
        snapshot = _path(
            source.get("snapshot_path"),
            label=f"XAU_TAPE_CURRENT_COLLECTOR_SNAPSHOT_{index}",
        )
        if snapshot.parent != snapshot_root:
            raise RuntimeError("XAU_TAPE_CURRENT_COLLECTOR_SNAPSHOT_PATH_MISMATCH")
        digest = sha256_file(snapshot)
        if source.get("sha256") != digest:
            raise RuntimeError("XAU_TAPE_CURRENT_COLLECTOR_SNAPSHOT_HASH_MISMATCH")
        snapshots[snapshot.name] = digest

    return {
        "schema_version": CURRENT_SNAPSHOT_SCHEMA,
        "instrument": XAU_INSTRUMENT,
        "entry_run_id": run_id,
        "tape_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "last_complete_m5_utc": str(manifest.get("last_complete_m5_utc") or ""),
        "year_sha256": year_hashes,
        "collector_snapshot_sha256": snapshots,
        "base": base_proof,
    }
