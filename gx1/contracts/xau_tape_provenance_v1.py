"""Immutable XAU_USD tape identity and lineage contracts.

Instrument identity is never inferred from a dataset or directory name.  The
base repair binds the canonical M1/M5 manifests and their hashes.  A current
snapshot then binds that base repair, every output-year hash, and the exact
collector snapshots used for its live tail.  Consumers validate the complete
chain before admitting model data.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


XAU_INSTRUMENT = "XAU_USD"
BASE_REPAIR_SCHEMA = "m5_tape_dec2024_repair_manifest_v2"
CURRENT_SNAPSHOT_SCHEMA = "m5_tape_current_snapshot_v2"
BASE_REPAIR_METHOD = "recompute_window_from_canonical_m1_drop_unbacked_bars"
CURRENT_SNAPSHOT_METHOD = "immutable_live_collector_snapshot_exact_m5_overlap"
CANONICAL_M5_SOURCE_SCHEMA = "xau_canonical_m5_source_v2"
CANONICAL_M5_PRODUCER_OWNER = (
    "gx1.scripts.backfill_xauusd_m5_from_oanda.materialize_native_m5_snapshot"
)
CANONICAL_M5_CLOSURE_CONTRACT = (
    "oanda_complete_true_source_absence_no_synthesis_v1"
)
CANONICAL_M5_SOURCE_CHUNK_SCHEMA = "xau_oanda_native_m5_source_chunk_v1"
CANONICAL_M5_SOURCE_RESPONSE_ENCODING = "canonical_json_utf8_gzip_mtime0"
CANONICAL_M5_REQUEST_INTERVAL_SEMANTICS = "left_closed_right_open"
CANONICAL_M5_PRODUCER_SOURCE_FILES = (
    "gx1/contracts/entry_model_native_bundle_commit_v1.py",
    "gx1/contracts/xau_tape_provenance_v1.py",
    "gx1/execution/oanda_client.py",
    "gx1/execution/oanda_credentials.py",
    "gx1/scripts/backfill_xauusd_m5_from_oanda.py",
    "gx1/utils/env_loader.py",
    "gx1_guards/gates.py",
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
_CANONICAL_M5_PRICE_COLUMNS = tuple(
    name for name in CANONICAL_M5_REQUIRED_COLUMNS if name not in {"time", "volume"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_VEDTAK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


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


def _utc_m5(raw: Any, *, label: str) -> pd.Timestamp:
    try:
        value = pd.Timestamp(pd.to_datetime(raw, utc=True, errors="raise"))
    except Exception as exc:
        raise RuntimeError(
            f"XAU_CANONICAL_M5_{label}_TIMESTAMP_INVALID: {raw!r}"
        ) from exc
    if pd.isna(value):
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_TIMESTAMP_INVALID: {raw!r}")
    if (
        value.second != 0
        or value.microsecond != 0
        or value.nanosecond != 0
        or value.minute % 5 != 0
    ):
        raise RuntimeError(
            f"XAU_CANONICAL_M5_{label}_TIMESTAMP_NOT_M5_ALIGNED: {value}"
        )
    return value


def _source_float(raw: Any, *, label: str) -> float:
    if isinstance(raw, bool):
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_PRICE_INVALID: {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"XAU_CANONICAL_M5_{label}_PRICE_INVALID: {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_PRICE_INVALID: {raw!r}")
    return value


def _source_volume(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise RuntimeError(f"XAU_CANONICAL_M5_VOLUME_INVALID: {raw!r}")
    return int(raw)


def _empty_canonical_m5_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series([], dtype="datetime64[ns, UTC]"),
            **{
                column: pd.Series([], dtype="float64")
                for column in _CANONICAL_M5_PRICE_COLUMNS
            },
            "volume": pd.Series([], dtype="int64"),
        }
    ).loc[:, list(CANONICAL_M5_REQUIRED_COLUMNS)]


def _validate_canonical_m5_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    allow_empty: bool,
) -> pd.DataFrame:
    observed = list(frame.columns)
    expected = list(CANONICAL_M5_REQUIRED_COLUMNS)
    if (
        observed != expected
        or len(observed) != len(set(observed))
    ):
        raise RuntimeError(
            f"XAU_CANONICAL_M5_{label}_SCHEMA_MISMATCH: "
            f"expected={expected} observed={observed}"
        )
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if out["time"].isna().any():
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_TIME_INVALID")
    if out.empty:
        if allow_empty:
            return _empty_canonical_m5_frame()
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_EMPTY")
    nanos = out["time"].astype("int64").to_numpy()
    if (nanos % 300_000_000_000 != 0).any():
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_TIME_NOT_M5_ALIGNED")
    if out["time"].duplicated().any() or not out["time"].is_monotonic_increasing:
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_TIME_ORDER_INVALID")
    prices = out.loc[:, list(_CANONICAL_M5_PRICE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    price_values = prices.to_numpy(dtype="float64")
    if (
        not math.isfinite(float(price_values.min()))
        or not math.isfinite(float(price_values.max()))
        or (price_values <= 0.0).any()
    ):
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_PRICE_INVALID")
    out.loc[:, list(_CANONICAL_M5_PRICE_COLUMNS)] = prices
    volume_numeric = pd.to_numeric(out["volume"], errors="coerce")
    volume_values = volume_numeric.to_numpy(dtype="float64")
    if (
        not all(math.isfinite(float(value)) for value in volume_values)
        or (volume_values < 0.0).any()
        or (volume_values != volume_values.astype("int64")).any()
    ):
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_VOLUME_INVALID")
    out["volume"] = volume_values.astype("int64")
    for prefix in ("", "bid_", "ask_"):
        opened = out[f"{prefix}open"].to_numpy(dtype="float64")
        high = out[f"{prefix}high"].to_numpy(dtype="float64")
        low = out[f"{prefix}low"].to_numpy(dtype="float64")
        closed = out[f"{prefix}close"].to_numpy(dtype="float64")
        if (
            (high < low).any()
            or (high < opened).any()
            or (high < closed).any()
            or (low > opened).any()
            or (low > closed).any()
        ):
            raise RuntimeError(
                f"XAU_CANONICAL_M5_{label}_{prefix.upper()}OHLC_GEOMETRY_INVALID"
            )
    for suffix in ("open", "high", "low", "close"):
        if (
            out[f"ask_{suffix}"].to_numpy(dtype="float64")
            < out[f"bid_{suffix}"].to_numpy(dtype="float64")
        ).any():
            raise RuntimeError(
                f"XAU_CANONICAL_M5_{label}_BID_ASK_GEOMETRY_INVALID"
            )
    return out.loc[:, expected]


def canonical_m5_rows_bytes(frame: pd.DataFrame) -> bytes:
    """Encode canonical typed rows with the stable v1 digest layout."""

    checked = _validate_canonical_m5_frame(
        frame,
        label="ROW_DIGEST",
        allow_empty=True,
    )
    records = np.empty(
        len(checked),
        dtype=np.dtype(
            [
                ("time", ">i8"),
                ("prices", (">f8", (12,))),
                ("volume", ">i8"),
            ],
            align=False,
        ),
    )
    records["time"] = checked["time"].astype("int64").to_numpy()
    records["prices"] = checked.loc[
        :, list(_CANONICAL_M5_PRICE_COLUMNS)
    ].to_numpy(dtype="float64")
    records["volume"] = checked["volume"].to_numpy(dtype="int64")
    return records.tobytes(order="C")


def canonical_m5_rows_sha256(frame: pd.DataFrame) -> str:
    """Hash canonical typed rows independently of parquet serialization."""

    return hashlib.sha256(canonical_m5_rows_bytes(frame)).hexdigest()


def canonical_m5_frame_from_oanda_response(
    response: Mapping[str, Any],
    *,
    request_start: Any,
    request_end: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one native OANDA MBA response without synthesis or fallback."""

    start = _utc_m5(request_start, label="REQUEST_START")
    end = _utc_m5(request_end, label="REQUEST_END")
    if end <= start:
        raise RuntimeError("XAU_CANONICAL_M5_REQUEST_INTERVAL_INVALID")
    if not isinstance(response, Mapping):
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_RESPONSE_OBJECT_REQUIRED")
    if response.get("instrument") != XAU_INSTRUMENT:
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_RESPONSE_INSTRUMENT_MISMATCH")
    if response.get("granularity") != "M5":
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_RESPONSE_GRANULARITY_MISMATCH")
    candles = response.get("candles")
    if not isinstance(candles, list):
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_RESPONSE_CANDLES_INVALID")

    rows: list[dict[str, Any]] = []
    incomplete = 0
    previous: pd.Timestamp | None = None
    for position, raw_candle in enumerate(candles):
        if not isinstance(raw_candle, Mapping):
            raise RuntimeError(
                f"XAU_CANONICAL_M5_SOURCE_CANDLE_INVALID: position={position}"
            )
        complete = raw_candle.get("complete")
        if not isinstance(complete, bool):
            raise RuntimeError(
                f"XAU_CANONICAL_M5_SOURCE_COMPLETION_INVALID: position={position}"
            )
        if not complete:
            incomplete += 1
            continue
        timestamp = _utc_m5(
            raw_candle.get("time"),
            label=f"SOURCE_CANDLE_{position}",
        )
        if not start <= timestamp < end:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SOURCE_CANDLE_OUTSIDE_REQUEST: "
                f"time={timestamp} start={start} end={end}"
            )
        if previous is not None and timestamp <= previous:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CANDLE_ORDER_INVALID")
        previous = timestamp
        parsed: dict[str, Any] = {"time": timestamp}
        for source_name, prefix in (("mid", ""), ("bid", "bid_"), ("ask", "ask_")):
            source = raw_candle.get(source_name)
            if not isinstance(source, Mapping):
                raise RuntimeError(
                    "XAU_CANONICAL_M5_SOURCE_PRICE_COMPONENT_MISSING: "
                    f"position={position} component={source_name}"
                )
            for source_field, target_field in (
                ("o", "open"),
                ("h", "high"),
                ("l", "low"),
                ("c", "close"),
            ):
                parsed[f"{prefix}{target_field}"] = _source_float(
                    source.get(source_field),
                    label=f"{source_name.upper()}_{source_field.upper()}",
                )
        parsed["volume"] = _source_volume(raw_candle.get("volume"))
        rows.append(parsed)

    frame = (
        pd.DataFrame(rows).loc[:, list(CANONICAL_M5_REQUIRED_COLUMNS)]
        if rows
        else _empty_canonical_m5_frame()
    )
    frame = _validate_canonical_m5_frame(
        frame,
        label="SOURCE_RESPONSE",
        allow_empty=True,
    )
    stats = {
        "response_candles": len(candles),
        "complete_candles": len(frame),
        "incomplete_candles": incomplete,
        "first_complete_utc": (
            None if frame.empty else pd.Timestamp(frame["time"].iloc[0]).isoformat()
        ),
        "last_complete_utc": (
            None if frame.empty else pd.Timestamp(frame["time"].iloc[-1]).isoformat()
        ),
        "canonical_complete_rows_sha256": canonical_m5_rows_sha256(frame),
    }
    return frame, stats


def _safe_relative_path(raw: Any, *, label: str) -> Path:
    value = str(raw or "")
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_PATH_INVALID: {value!r}")
    return path


def _require_sha256(raw: Any, *, label: str) -> str:
    value = str(raw or "").lower()
    if _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"XAU_CANONICAL_M5_{label}_SHA256_INVALID")
    return value


def _decode_source_chunk(raw: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(gzip.decompress(raw).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"XAU_CANONICAL_M5_SOURCE_CHUNK_DECODE_INVALID: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"XAU_CANONICAL_M5_SOURCE_CHUNK_OBJECT_REQUIRED: {path}")
    return value


def _validate_canonical_m5_source_contract(
    root: Path,
    manifest: dict[str, Any],
    *,
    expected_declared_root: Path,
) -> dict[str, Any]:
    """Require one hash-bound native-M5 owner and source-defined closure."""

    exact = {
        "schema_version": CANONICAL_M5_SOURCE_SCHEMA,
        "producer_owner": CANONICAL_M5_PRODUCER_OWNER,
        "source_kind": "oanda_native_mba_candles",
        "source_endpoint": "/instruments/XAU_USD/candles",
        "source_granularity": "M5",
        "prices": "MBA",
        "timestamp_semantics": "bar_start_utc",
        "bar_duration_seconds": 300,
        "decision_available_offset_seconds": 300,
        "completion_field": "complete",
        "completion_value": True,
        "market_closure_contract": CANONICAL_M5_CLOSURE_CONTRACT,
        "request_interval_semantics": CANONICAL_M5_REQUEST_INTERVAL_SEMANTICS,
        "source_response_encoding": CANONICAL_M5_SOURCE_RESPONSE_ENCODING,
        "source_chunk_schema": CANONICAL_M5_SOURCE_CHUNK_SCHEMA,
        "producer_repository_clean": True,
    }
    expected_keys = {
        *exact,
        "instrument",
        "timeframe",
        "out_root",
        "explicit_vedtak_id",
        "created_utc",
        "source_environment",
        "source_base_url",
        "requested_start_utc",
        "requested_end_utc_exclusive",
        "request_chunk_days",
        "source_chunks",
        "source_chunks_sha256",
        "producer_git_commit",
        "producer_source_files",
        "producer_source_inventory_sha256",
        "runtime_versions",
        "schema_required_cols",
        "schema_optional_cols",
        "row_count",
        "time_min_utc",
        "time_max_utc",
        "canonical_rows_sha256",
        "year_sha256",
        "year_rows",
        "year_time_bounds",
        "manifest_payload_sha256",
    }
    if set(manifest) != expected_keys:
        raise RuntimeError(
            "XAU_CANONICAL_M5_MANIFEST_SCHEMA_INVALID: "
            f"missing={sorted(expected_keys - set(manifest))} "
            f"extra={sorted(set(manifest) - expected_keys)}"
        )
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SOURCE_POLICY_MISMATCH: "
                f"field={key!r} expected={expected!r} "
                f"observed={manifest.get(key)!r}"
            )
    if manifest.get("instrument") != XAU_INSTRUMENT or manifest.get("timeframe") != "M5":
        raise RuntimeError("XAU_CANONICAL_M5_IDENTITY_MISMATCH")
    declared = Path(str(manifest.get("out_root") or "")).expanduser()
    if (
        not declared.is_absolute()
        or declared != expected_declared_root
        or str(declared) != str(expected_declared_root)
    ):
        raise RuntimeError(
            "XAU_CANONICAL_M5_DECLARED_ROOT_MISMATCH: "
            f"declared={declared} expected={expected_declared_root}"
        )
    vedtak = str(manifest.get("explicit_vedtak_id") or "")
    if _VEDTAK_RE.fullmatch(vedtak) is None:
        raise RuntimeError("XAU_CANONICAL_M5_VEDTAK_INVALID")
    try:
        created = pd.Timestamp(
            pd.to_datetime(manifest.get("created_utc"), utc=True, errors="raise")
        )
    except Exception as exc:
        raise RuntimeError("XAU_CANONICAL_M5_CREATED_UTC_INVALID") from exc
    if pd.isna(created):
        raise RuntimeError("XAU_CANONICAL_M5_CREATED_UTC_INVALID")
    environment = manifest.get("source_environment")
    base_url_by_environment = {
        "practice": "https://api-fxpractice.oanda.com/v3",
        "live": "https://api-fxtrade.oanda.com/v3",
    }
    if (
        environment not in base_url_by_environment
        or manifest.get("source_base_url") != base_url_by_environment[environment]
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_ENVIRONMENT_INVALID")
    start = _utc_m5(
        manifest.get("requested_start_utc"),
        label="MANIFEST_REQUEST_START",
    )
    end = _utc_m5(
        manifest.get("requested_end_utc_exclusive"),
        label="MANIFEST_REQUEST_END",
    )
    if end <= start:
        raise RuntimeError("XAU_CANONICAL_M5_REQUEST_INTERVAL_INVALID")
    chunk_days = manifest.get("request_chunk_days")
    if isinstance(chunk_days, bool) or not isinstance(chunk_days, int) or not 1 <= chunk_days <= 15:
        raise RuntimeError("XAU_CANONICAL_M5_REQUEST_CHUNK_DAYS_INVALID")
    commit = str(manifest.get("producer_git_commit") or "")
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_GIT_COMMIT_INVALID")
    versions = manifest.get("runtime_versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != {"numpy", "pandas", "pyarrow", "python"}
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise RuntimeError("XAU_CANONICAL_M5_RUNTIME_VERSIONS_INVALID")
    if manifest.get("schema_required_cols") != list(
        CANONICAL_M5_REQUIRED_COLUMNS
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SCHEMA_CONTRACT_MISMATCH")
    if manifest.get("schema_optional_cols") != []:
        raise RuntimeError("XAU_CANONICAL_M5_OPTIONAL_SCHEMA_FORBIDDEN")
    if "slippage_bps" in manifest.get("schema_required_cols", []):
        raise RuntimeError("XAU_CANONICAL_M5_SLIPPAGE_NOT_SOURCE_OBSERVED")

    producer_source_root = root / "producer_source"
    source_chunk_root = root / "source_chunks"
    for label, directory in (
        ("PRODUCER_SOURCE_ROOT", producer_source_root),
        ("SOURCE_CHUNK_ROOT", source_chunk_root),
    ):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"XAU_CANONICAL_M5_{label}_INVALID")
    if any(path.is_symlink() for path in producer_source_root.rglob("*")):
        raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_SYMLINK_FORBIDDEN")
    source_inventory = manifest.get("producer_source_files")
    if not isinstance(source_inventory, list):
        raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_INVENTORY_INVALID")
    expected_source_paths = list(CANONICAL_M5_PRODUCER_SOURCE_FILES)
    observed_source_paths: list[str] = []
    for item in source_inventory:
        if not isinstance(item, dict) or set(item) != {
            "repo_relative_path",
            "snapshot_relative_path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_ITEM_INVALID")
        repo_relative = str(
            _safe_relative_path(
                item.get("repo_relative_path"),
                label="PRODUCER_REPO_SOURCE",
            )
        )
        snapshot_relative = _safe_relative_path(
            item.get("snapshot_relative_path"),
            label="PRODUCER_SNAPSHOT_SOURCE",
        )
        if snapshot_relative != Path("producer_source") / repo_relative:
            raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_PATH_MISMATCH")
        source_path = root / snapshot_relative
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(
                f"XAU_CANONICAL_M5_PRODUCER_SOURCE_MISSING: {source_path}"
            )
        digest = _require_sha256(item.get("sha256"), label="PRODUCER_SOURCE")
        size = item.get("size_bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or source_path.stat().st_size != size
            or sha256_file(source_path) != digest
        ):
            raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_BINDING_MISMATCH")
        observed_source_paths.append(repo_relative)
    if observed_source_paths != expected_source_paths:
        raise RuntimeError(
            "XAU_CANONICAL_M5_PRODUCER_SOURCE_SET_MISMATCH: "
            f"expected={expected_source_paths} observed={observed_source_paths}"
        )
    actual_source_files = sorted(
        str(path.relative_to(root / "producer_source"))
        for path in producer_source_root.rglob("*")
        if path.is_file()
    )
    if actual_source_files != expected_source_paths:
        raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_FILESYSTEM_MISMATCH")
    if (
        _require_sha256(
            manifest.get("producer_source_inventory_sha256"),
            label="PRODUCER_SOURCE_INVENTORY",
        )
        != canonical_json_sha256(source_inventory)
    ):
        raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_INVENTORY_HASH_MISMATCH")

    source_chunks = manifest.get("source_chunks")
    if not isinstance(source_chunks, list) or not source_chunks:
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNKS_MISSING")
    if (
        _require_sha256(
            manifest.get("source_chunks_sha256"),
            label="SOURCE_CHUNKS",
        )
        != canonical_json_sha256(source_chunks)
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNKS_HASH_MISMATCH")
    expected_chunk_keys = {
        "sequence",
        "request_from_utc",
        "request_to_utc_exclusive",
        "relative_path",
        "sha256",
        "size_bytes",
        "response_candles",
        "complete_candles",
        "incomplete_candles",
        "first_complete_utc",
        "last_complete_utc",
        "canonical_complete_rows_sha256",
    }
    source_digest = hashlib.sha256()
    source_row_count = 0
    source_time_min: str | None = None
    source_time_max: str | None = None
    previous_source_time: pd.Timestamp | None = None
    cursor = start
    actual_chunk_files = sorted(
        path
        for path in source_chunk_root.glob("chunk-*.json.gz")
        if path.is_file()
    )
    manifest_chunk_files: list[Path] = []
    for sequence, metadata in enumerate(source_chunks):
        if not isinstance(metadata, dict) or set(metadata) != expected_chunk_keys:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_METADATA_INVALID")
        if metadata.get("sequence") != sequence:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_SEQUENCE_INVALID")
        chunk_start = _utc_m5(
            metadata.get("request_from_utc"),
            label=f"CHUNK_{sequence}_START",
        )
        chunk_end = _utc_m5(
            metadata.get("request_to_utc_exclusive"),
            label=f"CHUNK_{sequence}_END",
        )
        if chunk_start != cursor or chunk_end <= chunk_start or chunk_end > end:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_INTERVAL_INVALID")
        cursor = chunk_end
        relative = _safe_relative_path(
            metadata.get("relative_path"),
            label=f"CHUNK_{sequence}",
        )
        expected_relative = Path("source_chunks") / f"chunk-{sequence:06d}.json.gz"
        if relative != expected_relative:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_PATH_MISMATCH")
        chunk_path = root / relative
        manifest_chunk_files.append(chunk_path)
        digest = _require_sha256(metadata.get("sha256"), label=f"CHUNK_{sequence}")
        size = metadata.get("size_bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or chunk_path.is_symlink()
            or not chunk_path.is_file()
        ):
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_BINDING_MISMATCH")
        compressed = chunk_path.read_bytes()
        if (
            len(compressed) != size
            or hashlib.sha256(compressed).hexdigest() != digest
        ):
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_BINDING_MISMATCH")
        payload = _decode_source_chunk(compressed, path=chunk_path)
        if set(payload) != {"schema_version", "request", "response"}:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_PAYLOAD_SCHEMA_INVALID")
        if payload.get("schema_version") != CANONICAL_M5_SOURCE_CHUNK_SCHEMA:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_SCHEMA_MISMATCH")
        request = payload.get("request")
        expected_request = {
            "instrument": XAU_INSTRUMENT,
            "from": chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "granularity": "M5",
            "price": "MBA",
        }
        if request != expected_request:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_REQUEST_MISMATCH")
        response = payload.get("response")
        frame, stats = canonical_m5_frame_from_oanda_response(
            response,
            request_start=chunk_start,
            request_end=chunk_end,
        )
        for key, expected in stats.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    "XAU_CANONICAL_M5_SOURCE_CHUNK_STATS_MISMATCH: "
                    f"sequence={sequence} field={key}"
                )
        if not frame.empty:
            first = pd.Timestamp(frame["time"].iloc[0])
            last = pd.Timestamp(frame["time"].iloc[-1])
            if previous_source_time is not None and first <= previous_source_time:
                raise RuntimeError(
                    "XAU_CANONICAL_M5_SOURCE_CROSS_CHUNK_ORDER_INVALID"
                )
            if source_time_min is None:
                source_time_min = first.isoformat()
            source_time_max = last.isoformat()
            previous_source_time = last
            source_row_count += len(frame)
            source_digest.update(canonical_m5_rows_bytes(frame))
    if cursor != end or manifest_chunk_files != actual_chunk_files:
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_CLOSURE_INVALID")

    if source_row_count <= 0 or source_time_min is None or source_time_max is None:
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_ALL_EMPTY")
    row_count = manifest.get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count != source_row_count
    ):
        raise RuntimeError("XAU_CANONICAL_M5_ROW_COUNT_MISMATCH")
    source_rows_sha = source_digest.hexdigest()
    if (
        manifest.get("time_min_utc") != source_time_min
        or manifest.get("time_max_utc") != source_time_max
        or _require_sha256(
            manifest.get("canonical_rows_sha256"),
            label="CANONICAL_ROWS",
        )
        != source_rows_sha
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_ROWS_BINDING_MISMATCH")

    year_hashes = manifest.get("year_sha256")
    if not isinstance(year_hashes, dict) or not year_hashes:
        raise RuntimeError("XAU_CANONICAL_M5_YEAR_HASHES_MISSING")
    year_rows = manifest.get("year_rows")
    if not isinstance(year_rows, dict) or set(year_rows) != set(year_hashes):
        raise RuntimeError("XAU_CANONICAL_M5_YEAR_ROWS_MISSING")
    year_time_bounds = manifest.get("year_time_bounds")
    if (
        not isinstance(year_time_bounds, dict)
        or set(year_time_bounds) != set(year_hashes)
    ):
        raise RuntimeError("XAU_CANONICAL_M5_YEAR_TIME_BOUNDS_MISSING")
    actual_parts = sorted(root.glob("year=*/part-000.parquet"))
    actual_keys = [part.parent.name for part in actual_parts]
    if actual_keys != sorted(year_hashes):
        raise RuntimeError(
            "XAU_CANONICAL_M5_YEAR_SET_MISMATCH: "
            f"manifest={sorted(year_hashes)} filesystem={actual_keys}"
        )
    for part in actual_parts:
        if part.is_symlink() or part.parent.is_symlink():
            raise RuntimeError(
                f"XAU_CANONICAL_M5_YEAR_PART_SYMLINK_FORBIDDEN: {part}"
            )
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
    parquet_digest = hashlib.sha256()
    parquet_row_count = 0
    previous_parquet_time: pd.Timestamp | None = None
    for part in actual_parts:
        frame = _validate_canonical_m5_frame(
            pd.read_parquet(part),
            label=f"PARQUET_{part.parent.name}",
            allow_empty=False,
        )
        year = int(part.parent.name.split("=", 1)[1])
        if (frame["time"].dt.year != year).any():
            raise RuntimeError("XAU_CANONICAL_M5_YEAR_PARTITION_CONTENT_MISMATCH")
        bounds = year_time_bounds.get(part.parent.name)
        expected_bounds = {
            "time_min_utc": pd.Timestamp(frame["time"].iloc[0]).isoformat(),
            "time_max_utc": pd.Timestamp(frame["time"].iloc[-1]).isoformat(),
        }
        if bounds != expected_bounds:
            raise RuntimeError("XAU_CANONICAL_M5_YEAR_TIME_BOUNDS_MISMATCH")
        first = pd.Timestamp(frame["time"].iloc[0])
        if previous_parquet_time is not None and first <= previous_parquet_time:
            raise RuntimeError("XAU_CANONICAL_M5_PARQUET_GLOBAL_ORDER_INVALID")
        previous_parquet_time = pd.Timestamp(frame["time"].iloc[-1])
        parquet_row_count += len(frame)
        parquet_digest.update(canonical_m5_rows_bytes(frame))
    if (
        parquet_row_count != source_row_count
        or parquet_digest.hexdigest() != source_rows_sha
    ):
        raise RuntimeError("XAU_CANONICAL_M5_PARQUET_SOURCE_REDERIVATION_MISMATCH")

    allowed_top_level = {
        "MANIFEST.json",
        "producer_source",
        "source_chunks",
        *year_hashes,
    }
    if {path.name for path in root.iterdir()} != allowed_top_level:
        raise RuntimeError("XAU_CANONICAL_M5_FILESYSTEM_SURFACE_INVALID")
    without_manifest_hash = dict(manifest)
    observed_manifest_hash = _require_sha256(
        without_manifest_hash.pop("manifest_payload_sha256", ""),
        label="MANIFEST_PAYLOAD",
    )
    if observed_manifest_hash != canonical_json_sha256(without_manifest_hash):
        raise RuntimeError("XAU_CANONICAL_M5_MANIFEST_PAYLOAD_HASH_MISMATCH")
    return {
        **exact,
        "schema_required_cols": list(CANONICAL_M5_REQUIRED_COLUMNS),
        "schema_optional_cols": [],
        "requested_start_utc": start.isoformat(),
        "requested_end_utc_exclusive": end.isoformat(),
        "row_count": source_row_count,
        "time_min_utc": source_time_min,
        "time_max_utc": source_time_max,
        "canonical_rows_sha256": source_rows_sha,
        "source_chunks_sha256": str(manifest["source_chunks_sha256"]),
        "producer_git_commit": commit,
        "producer_source_inventory_sha256": str(
            manifest["producer_source_inventory_sha256"]
        ),
        "manifest_payload_sha256": observed_manifest_hash,
        "year_sha256": dict(sorted(year_hashes.items())),
        "year_rows": dict(sorted(year_rows.items())),
    }


def validate_canonical_m5_source_bundle_v2(
    physical_root: Path | str,
    *,
    expected_declared_root: Path | str,
) -> dict[str, Any]:
    """Validate a final bundle or its hidden pre-publication staging root."""

    physical_arg = Path(physical_root).expanduser()
    if (
        not physical_arg.is_absolute()
        or physical_arg.is_symlink()
        or not physical_arg.is_dir()
    ):
        raise RuntimeError(
            f"XAU_CANONICAL_M5_PHYSICAL_ROOT_INVALID: {physical_arg}"
        )
    physical = physical_arg.resolve()
    if physical != physical_arg:
        raise RuntimeError(
            "XAU_CANONICAL_M5_PHYSICAL_ROOT_NOT_CANONICAL: "
            f"raw={physical_arg} resolved={physical}"
        )
    declared_arg = Path(expected_declared_root).expanduser()
    if not declared_arg.is_absolute():
        raise RuntimeError(
            f"XAU_CANONICAL_M5_EXPECTED_DECLARED_ROOT_NOT_ABSOLUTE: {declared_arg}"
        )
    declared = declared_arg.resolve(strict=False)
    if declared != declared_arg:
        raise RuntimeError(
            "XAU_CANONICAL_M5_EXPECTED_DECLARED_ROOT_NOT_CANONICAL: "
            f"raw={declared_arg} resolved={declared}"
        )
    manifest = _json(
        physical / "MANIFEST.json",
        label="XAU_CANONICAL_M5_MANIFEST",
    )
    if manifest.get("instrument") != XAU_INSTRUMENT or manifest.get("timeframe") != "M5":
        raise RuntimeError("XAU_CANONICAL_M5_IDENTITY_MISMATCH")
    return _validate_canonical_m5_source_contract(
        physical,
        manifest,
        expected_declared_root=declared,
    )


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
    declared_root = Path(str(manifest.get("out_root") or "")).expanduser()
    if not declared_root.is_absolute():
        raise RuntimeError(
            f"XAU_CANONICAL_{expected_timeframe}_DECLARED_ROOT_NOT_ABSOLUTE: "
            f"{declared_root}"
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
            _validate_canonical_m5_source_contract(
                canonical_root,
                manifest,
                expected_declared_root=canonical_root,
            )
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
                "requested_start_utc",
                "requested_end_utc_exclusive",
                "row_count",
                "time_min_utc",
                "time_max_utc",
                "canonical_rows_sha256",
                "source_chunks_sha256",
                "producer_git_commit",
                "producer_source_inventory_sha256",
                "manifest_payload_sha256",
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
