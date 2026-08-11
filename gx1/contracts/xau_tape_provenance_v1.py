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
CANONICAL_NATIVE_SOURCE_SCHEMA = "xau_canonical_native_source_v3"
CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA = "xau_canonical_native_source_v4"
SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION = (
    "seq513_source_cascade_pair_proof_v2"
)
CANONICAL_NATIVE_SUCCESSOR_MODE = "successor"
CANONICAL_NATIVE_PRODUCER_OWNER = (
    "gx1.scripts.backfill_xauusd_m5_from_oanda.materialize_native_xau_snapshot"
)
CANONICAL_NATIVE_CLOSURE_CONTRACT = (
    "oanda_complete_true_source_absence_no_synthesis_v1"
)
CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA = "xau_oanda_native_source_chunk_v2"
CANONICAL_NATIVE_SOURCE_RESPONSE_ENCODING = "canonical_json_utf8_gzip_mtime0"
CANONICAL_NATIVE_REQUEST_INTERVAL_SEMANTICS = "left_closed_right_open"
CANONICAL_NATIVE_PRODUCER_SOURCE_FILES = (
    "gx1/contracts/entry_model_native_bundle_commit_v1.py",
    "gx1/contracts/xau_tape_provenance_v1.py",
    "gx1/execution/oanda_client.py",
    "gx1/execution/oanda_credentials.py",
    "gx1/scripts/backfill_xauusd_m5_from_oanda.py",
    "gx1/utils/env_loader.py",
    "gx1_guards/gates.py",
)
CANONICAL_NATIVE_REQUIRED_COLUMNS = (
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
_CANONICAL_NATIVE_PRICE_COLUMNS = tuple(
    name
    for name in CANONICAL_NATIVE_REQUIRED_COLUMNS
    if name not in {"time", "volume"}
)
NATIVE_TIMEFRAME_POLICY: dict[str, dict[str, int]] = {
    "M1": {
        "bar_seconds": 60,
        "request_chunk_days": 3,
        "max_theoretical_slots": 4_320,
    },
    "M5": {
        "bar_seconds": 300,
        "request_chunk_days": 15,
        "max_theoretical_slots": 4_320,
    },
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_VEDTAK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_MAX_CANONICAL_NATIVE_ANCESTOR_DEPTH = 1_024
# Retired-ancestor attestation (2026-08-11): a superseded native tape
# generation may be retired through the retention owner
# (gx1.scripts.cleanup_gx1_evidence_v1). Its bytes are gone, but the executed
# DELETE_COMPLETE event chain (plan -> approval -> execution) records a
# hash-bound per-file inventory of exactly what was deleted. A missing
# successor parent root is therefore admitted ONLY when that inventory
# attests the deleted MANIFEST.json with exactly the child's recorded
# parent manifest sha256 — cryptographic identity continuity without the
# bytes. Everything else (absent without attestation, present-but-tampered,
# shape violations) fails closed exactly as before.
_RETIRED_NATIVE_ATTESTATION_SCHEMA = "gx1_retired_native_root_attestation_v1"
_EVIDENCE_CLEANUP_EXECUTION_SCHEMA = "gx1_evidence_cleanup_execution_v1"
_EVIDENCE_CLEANUP_REPORTS_DIRNAME = "gx1_evidence_retention_cleanup_reports"
_CANONICAL_NATIVE_PARENT_BINDING_FIELDS = (
    "schema_version",
    "root",
    "manifest_path",
    "manifest_sha256",
    "instrument",
    "timeframe",
    "explicit_vedtak_id",
    "source_environment",
    "source_base_url",
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
)


def native_timeframe_policy(timeframe: Any) -> tuple[str, dict[str, int]]:
    normalized = str(timeframe or "").strip().upper()
    if normalized not in NATIVE_TIMEFRAME_POLICY:
        raise RuntimeError(
            "XAU_CANONICAL_NATIVE_TIMEFRAME_UNSUPPORTED: "
            f"observed={timeframe!r}"
        )
    return normalized, NATIVE_TIMEFRAME_POLICY[normalized]


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


def canonical_native_parent_binding_v1(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact immutable native identity admitted as a successor parent."""

    if not isinstance(descriptor, Mapping):
        raise RuntimeError("XAU_CANONICAL_NATIVE_PARENT_DESCRIPTOR_INVALID")
    missing = [
        name
        for name in _CANONICAL_NATIVE_PARENT_BINDING_FIELDS
        if name not in descriptor
    ]
    if missing:
        raise RuntimeError(
            "XAU_CANONICAL_NATIVE_PARENT_DESCRIPTOR_FIELDS_MISSING: "
            f"{missing}"
        )
    return json.loads(
        json.dumps(
            {
                name: descriptor[name]
                for name in _CANONICAL_NATIVE_PARENT_BINDING_FIELDS
            },
            sort_keys=True,
            allow_nan=False,
        )
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


def _utc_native_bar(
    raw: Any,
    *,
    timeframe: str,
    label: str,
) -> pd.Timestamp:
    normalized, policy = native_timeframe_policy(timeframe)
    prefix = f"XAU_CANONICAL_{normalized}"
    try:
        # OANDA emits one ISO-8601 value per candle.  ``pd.to_datetime`` on a
        # scalar takes the slow array-dispatch path; this validator is also
        # called over immutable multi-million-row M1 source bundles during
        # lifecycle admission.  Parse the scalar directly while preserving
        # the exact UTC/error semantics of the previous path.
        value = pd.Timestamp(raw)
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        else:
            value = value.tz_convert("UTC")
    except Exception as exc:
        raise RuntimeError(
            f"{prefix}_{label}_TIMESTAMP_INVALID: {raw!r}"
        ) from exc
    if pd.isna(value):
        raise RuntimeError(f"{prefix}_{label}_TIMESTAMP_INVALID: {raw!r}")
    if value.value % (policy["bar_seconds"] * 1_000_000_000) != 0:
        raise RuntimeError(
            f"{prefix}_{label}_TIMESTAMP_NOT_{normalized}_ALIGNED: {value}"
        )
    return value


def _source_float(raw: Any, *, timeframe: str, label: str) -> float:
    normalized, _policy = native_timeframe_policy(timeframe)
    prefix = f"XAU_CANONICAL_{normalized}"
    if isinstance(raw, bool):
        raise RuntimeError(f"{prefix}_{label}_PRICE_INVALID: {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{prefix}_{label}_PRICE_INVALID: {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{prefix}_{label}_PRICE_INVALID: {raw!r}")
    return value


def _source_volume(raw: Any, *, timeframe: str) -> int:
    normalized, _policy = native_timeframe_policy(timeframe)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise RuntimeError(
            f"XAU_CANONICAL_{normalized}_VOLUME_INVALID: {raw!r}"
        )
    return int(raw)


def _empty_canonical_native_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series([], dtype="datetime64[ns, UTC]"),
            **{
                column: pd.Series([], dtype="float64")
                for column in _CANONICAL_NATIVE_PRICE_COLUMNS
            },
            "volume": pd.Series([], dtype="int64"),
        }
    ).loc[:, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)]


def _validate_canonical_native_frame(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    label: str,
    allow_empty: bool,
) -> pd.DataFrame:
    normalized, policy = native_timeframe_policy(timeframe)
    prefix = f"XAU_CANONICAL_{normalized}"
    observed = list(frame.columns)
    expected = list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
    if (
        observed != expected
        or len(observed) != len(set(observed))
    ):
        raise RuntimeError(
            f"{prefix}_{label}_SCHEMA_MISMATCH: "
            f"expected={expected} observed={observed}"
        )
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if out["time"].isna().any():
        raise RuntimeError(f"{prefix}_{label}_TIME_INVALID")
    if out.empty:
        if allow_empty:
            return _empty_canonical_native_frame()
        raise RuntimeError(f"{prefix}_{label}_EMPTY")
    nanos = out["time"].astype("int64").to_numpy()
    if (nanos % (policy["bar_seconds"] * 1_000_000_000) != 0).any():
        raise RuntimeError(
            f"{prefix}_{label}_TIME_NOT_{normalized}_ALIGNED"
        )
    if out["time"].duplicated().any() or not out["time"].is_monotonic_increasing:
        raise RuntimeError(f"{prefix}_{label}_TIME_ORDER_INVALID")
    prices = out.loc[:, list(_CANONICAL_NATIVE_PRICE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    price_values = prices.to_numpy(dtype="float64")
    if (
        not math.isfinite(float(price_values.min()))
        or not math.isfinite(float(price_values.max()))
        or (price_values <= 0.0).any()
    ):
        raise RuntimeError(f"{prefix}_{label}_PRICE_INVALID")
    out.loc[:, list(_CANONICAL_NATIVE_PRICE_COLUMNS)] = prices
    volume_numeric = pd.to_numeric(out["volume"], errors="coerce")
    volume_values = volume_numeric.to_numpy(dtype="float64")
    if (
        not all(math.isfinite(float(value)) for value in volume_values)
        or (volume_values < 0.0).any()
        or (volume_values != volume_values.astype("int64")).any()
    ):
        raise RuntimeError(f"{prefix}_{label}_VOLUME_INVALID")
    out["volume"] = volume_values.astype("int64")
    for price_prefix in ("", "bid_", "ask_"):
        opened = out[f"{price_prefix}open"].to_numpy(dtype="float64")
        high = out[f"{price_prefix}high"].to_numpy(dtype="float64")
        low = out[f"{price_prefix}low"].to_numpy(dtype="float64")
        closed = out[f"{price_prefix}close"].to_numpy(dtype="float64")
        if (
            (high < low).any()
            or (high < opened).any()
            or (high < closed).any()
            or (low > opened).any()
            or (low > closed).any()
        ):
            raise RuntimeError(
                f"{prefix}_{label}_{price_prefix.upper()}OHLC_GEOMETRY_INVALID"
            )
    for suffix in ("open", "high", "low", "close"):
        if (
            out[f"ask_{suffix}"].to_numpy(dtype="float64")
            < out[f"bid_{suffix}"].to_numpy(dtype="float64")
        ).any():
            raise RuntimeError(
                f"{prefix}_{label}_BID_ASK_GEOMETRY_INVALID"
            )
    return out.loc[:, expected]


def validate_canonical_native_frame(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    label: str,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Public SSOT validator for one exact native XAU M1/M5 frame."""

    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(
            f"XAU_CANONICAL_{str(timeframe).upper()}_{label}_FRAME_INVALID"
        )
    return _validate_canonical_native_frame(
        frame,
        timeframe=timeframe,
        label=label,
        allow_empty=allow_empty,
    )


def canonical_native_rows_bytes(
    frame: pd.DataFrame,
    *,
    timeframe: str,
) -> bytes:
    """Encode canonical typed rows with one stable cross-timeframe layout."""

    checked = _validate_canonical_native_frame(
        frame,
        timeframe=timeframe,
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
        :, list(_CANONICAL_NATIVE_PRICE_COLUMNS)
    ].to_numpy(dtype="float64")
    records["volume"] = checked["volume"].to_numpy(dtype="int64")
    return records.tobytes(order="C")


def canonical_native_rows_sha256(
    frame: pd.DataFrame,
    *,
    timeframe: str,
) -> str:
    """Hash canonical typed rows independently of parquet serialization."""

    return hashlib.sha256(
        canonical_native_rows_bytes(frame, timeframe=timeframe)
    ).hexdigest()


def canonical_native_frame_from_oanda_response(
    response: Mapping[str, Any],
    *,
    timeframe: str,
    request_start: Any,
    request_end: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one native OANDA MBA response without synthesis or fallback."""

    normalized, _policy = native_timeframe_policy(timeframe)
    prefix = f"XAU_CANONICAL_{normalized}"
    start = _utc_native_bar(
        request_start,
        timeframe=normalized,
        label="REQUEST_START",
    )
    end = _utc_native_bar(
        request_end,
        timeframe=normalized,
        label="REQUEST_END",
    )
    if end <= start:
        raise RuntimeError(f"{prefix}_REQUEST_INTERVAL_INVALID")
    if not isinstance(response, Mapping):
        raise RuntimeError(f"{prefix}_SOURCE_RESPONSE_OBJECT_REQUIRED")
    if response.get("instrument") != XAU_INSTRUMENT:
        raise RuntimeError(f"{prefix}_SOURCE_RESPONSE_INSTRUMENT_MISMATCH")
    if response.get("granularity") != normalized:
        raise RuntimeError(f"{prefix}_SOURCE_RESPONSE_GRANULARITY_MISMATCH")
    candles = response.get("candles")
    if not isinstance(candles, list):
        raise RuntimeError(f"{prefix}_SOURCE_RESPONSE_CANDLES_INVALID")

    rows: list[dict[str, Any]] = []
    previous: pd.Timestamp | None = None
    for position, raw_candle in enumerate(candles):
        if not isinstance(raw_candle, Mapping):
            raise RuntimeError(
                f"{prefix}_SOURCE_CANDLE_INVALID: position={position}"
            )
        complete = raw_candle.get("complete")
        if not isinstance(complete, bool):
            raise RuntimeError(
                f"{prefix}_SOURCE_COMPLETION_INVALID: position={position}"
            )
        if not complete:
            raise RuntimeError(
                f"{prefix}_SOURCE_INCOMPLETE_CANDLE_FORBIDDEN: "
                f"position={position}"
            )
        timestamp = _utc_native_bar(
            raw_candle.get("time"),
            timeframe=normalized,
            label=f"SOURCE_CANDLE_{position}",
        )
        if not start <= timestamp < end:
            raise RuntimeError(
                f"{prefix}_SOURCE_CANDLE_OUTSIDE_REQUEST: "
                f"time={timestamp} start={start} end={end}"
            )
        if previous is not None and timestamp <= previous:
            raise RuntimeError(f"{prefix}_SOURCE_CANDLE_ORDER_INVALID")
        previous = timestamp
        parsed: dict[str, Any] = {"time": timestamp}
        for source_name, price_prefix in (
            ("mid", ""),
            ("bid", "bid_"),
            ("ask", "ask_"),
        ):
            source = raw_candle.get(source_name)
            if not isinstance(source, Mapping):
                raise RuntimeError(
                    f"{prefix}_SOURCE_PRICE_COMPONENT_MISSING: "
                    f"position={position} component={source_name}"
                )
            for source_field, target_field in (
                ("o", "open"),
                ("h", "high"),
                ("l", "low"),
                ("c", "close"),
            ):
                parsed[f"{price_prefix}{target_field}"] = _source_float(
                    source.get(source_field),
                    timeframe=normalized,
                    label=f"{source_name.upper()}_{source_field.upper()}",
                )
        parsed["volume"] = _source_volume(
            raw_candle.get("volume"),
            timeframe=normalized,
        )
        rows.append(parsed)

    frame = (
        pd.DataFrame(rows).loc[:, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)]
        if rows
        else _empty_canonical_native_frame()
    )
    frame = _validate_canonical_native_frame(
        frame,
        timeframe=normalized,
        label="SOURCE_RESPONSE",
        allow_empty=True,
    )
    stats = {
        "response_candles": len(candles),
        "complete_candles": len(frame),
        "incomplete_candles": 0,
        "first_complete_utc": (
            None if frame.empty else pd.Timestamp(frame["time"].iloc[0]).isoformat()
        ),
        "last_complete_utc": (
            None if frame.empty else pd.Timestamp(frame["time"].iloc[-1]).isoformat()
        ),
        "canonical_complete_rows_sha256": canonical_native_rows_sha256(
            frame,
            timeframe=normalized,
        ),
    }
    return frame, stats


def _safe_relative_path(
    raw: Any,
    *,
    timeframe: str,
    label: str,
) -> Path:
    normalized, _policy = native_timeframe_policy(timeframe)
    value = str(raw or "")
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise RuntimeError(
            f"XAU_CANONICAL_{normalized}_{label}_PATH_INVALID: {value!r}"
        )
    return path


def _require_sha256(
    raw: Any,
    *,
    timeframe: str,
    label: str,
) -> str:
    normalized, _policy = native_timeframe_policy(timeframe)
    value = str(raw or "").lower()
    if _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(
            f"XAU_CANONICAL_{normalized}_{label}_SHA256_INVALID"
        )
    return value


def _decode_source_chunk(
    raw: bytes,
    *,
    timeframe: str,
    path: Path,
) -> dict[str, Any]:
    normalized, _policy = native_timeframe_policy(timeframe)
    try:
        value = json.loads(gzip.decompress(raw).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"XAU_CANONICAL_{normalized}_SOURCE_CHUNK_DECODE_INVALID: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"XAU_CANONICAL_{normalized}_SOURCE_CHUNK_OBJECT_REQUIRED: {path}"
        )
    return value


def _native_parent_binding_from_manifest(
    *,
    parent_root: Path,
    parent_manifest_path: Path,
    parent_manifest: Mapping[str, Any],
    timeframe: str,
) -> dict[str, Any]:
    """Reconstruct the exact successor-parent identity from immutable bytes."""

    normalized, _policy = native_timeframe_policy(timeframe)
    declared_root = Path(str(parent_manifest.get("out_root") or "")).expanduser()
    if declared_root != parent_root:
        raise RuntimeError(
            "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_DECLARED_ROOT_MISMATCH"
        )
    parent_without_hash = dict(parent_manifest)
    parent_payload_sha = _require_sha256(
        parent_without_hash.pop("manifest_payload_sha256", ""),
        timeframe=normalized,
        label="SUCCESSOR_PARENT_PAYLOAD",
    )
    if canonical_json_sha256(parent_without_hash) != parent_payload_sha:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_PARENT_PAYLOAD_INVALID")
    return {
        "schema_version": parent_manifest.get("schema_version"),
        "root": str(parent_root),
        "manifest_path": str(parent_manifest_path),
        "manifest_sha256": sha256_file(parent_manifest_path),
        "instrument": parent_manifest.get("instrument"),
        "timeframe": parent_manifest.get("timeframe"),
        "explicit_vedtak_id": parent_manifest.get("explicit_vedtak_id"),
        "source_environment": parent_manifest.get("source_environment"),
        "source_base_url": parent_manifest.get("source_base_url"),
        "requested_start_utc": _utc_native_bar(
            parent_manifest.get("requested_start_utc"),
            timeframe=normalized,
            label="SUCCESSOR_PARENT_BINDING_START",
        ).isoformat(),
        "requested_end_utc_exclusive": _utc_native_bar(
            parent_manifest.get("requested_end_utc_exclusive"),
            timeframe=normalized,
            label="SUCCESSOR_PARENT_BINDING_END",
        ).isoformat(),
        "row_count": parent_manifest.get("row_count"),
        "time_min_utc": _utc_native_bar(
            parent_manifest.get("time_min_utc"),
            timeframe=normalized,
            label="SUCCESSOR_PARENT_BINDING_TIME_MIN",
        ).isoformat(),
        "time_max_utc": _utc_native_bar(
            parent_manifest.get("time_max_utc"),
            timeframe=normalized,
            label="SUCCESSOR_PARENT_BINDING_TIME_MAX",
        ).isoformat(),
        "canonical_rows_sha256": parent_manifest.get("canonical_rows_sha256"),
        "source_chunks_sha256": parent_manifest.get("source_chunks_sha256"),
        "producer_git_commit": parent_manifest.get("producer_git_commit"),
        "producer_source_inventory_sha256": parent_manifest.get(
            "producer_source_inventory_sha256"
        ),
        "manifest_payload_sha256": parent_payload_sha,
        "year_sha256": parent_manifest.get("year_sha256"),
        "year_rows": parent_manifest.get("year_rows"),
    }


def _retired_native_root_attestation(
    parent_root: Path,
    *,
    expected_manifest_sha256: str,
    reports_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Attest a retired native root through the executed retention chain.

    Returns the attestation payload when an executed DELETE_COMPLETE
    retention event covers exactly ``parent_root`` and its hash-verified
    per-file inventory records the deleted ``MANIFEST.json`` with exactly
    ``expected_manifest_sha256``; returns ``None`` otherwise (the caller
    fails closed). Every referenced artifact is re-verified by content hash
    before it is believed.
    """

    from gx1.contracts.evidence_retention_v1 import GX1_DATA_ROOT

    if reports_dir is None:
        reports_dir = (
            GX1_DATA_ROOT / "reports" / _EVIDENCE_CLEANUP_REPORTS_DIRNAME
        )
    if not reports_dir.is_dir():
        return None
    for execution_path in sorted(
        reports_dir.glob("GX1_EVIDENCE_CLEANUP_EXECUTION_*.json"),
        reverse=True,
    ):
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(execution, dict)
            or execution.get("schema_version")
            != _EVIDENCE_CLEANUP_EXECUTION_SCHEMA
            or execution.get("decision") != "DELETE_COMPLETE"
        ):
            continue
        deleted = execution.get("deleted")
        if not isinstance(deleted, list) or str(parent_root) not in deleted:
            continue
        plan_path = Path(str(execution.get("plan_json") or "")).expanduser()
        plan_sha = execution.get("plan_sha256")
        if (
            not isinstance(plan_sha, str)
            or plan_path.is_symlink()
            or not plan_path.is_file()
            or sha256_file(plan_path) != plan_sha
        ):
            continue
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        plan_body = plan.get("plan") if isinstance(plan, dict) else None
        targets = (
            plan_body.get("targets")
            if isinstance(plan_body, dict)
            else plan.get("targets")
            if isinstance(plan, dict)
            else None
        )
        for target in targets or ():
            if (
                not isinstance(target, dict)
                or target.get("path") != str(parent_root)
                or target.get("kind") != "directory"
            ):
                continue
            inventory_path = Path(
                str(target.get("inventory_jsonl") or "")
            ).expanduser()
            inventory_sha = target.get("inventory_jsonl_sha256")
            if (
                not isinstance(inventory_sha, str)
                or inventory_path.is_symlink()
                or not inventory_path.is_file()
                or sha256_file(inventory_path) != inventory_sha
            ):
                continue
            for line in inventory_path.read_text(
                encoding="utf-8"
            ).splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(entry, dict)
                    and entry.get("relative_path") == "MANIFEST.json"
                    and entry.get("kind") == "file"
                    and entry.get("sha256") == expected_manifest_sha256
                ):
                    return {
                        "schema_version": _RETIRED_NATIVE_ATTESTATION_SCHEMA,
                        "parent_root": str(parent_root),
                        "parent_manifest_sha256": expected_manifest_sha256,
                        "plan_json": str(plan_path),
                        "plan_sha256": plan_sha,
                        "inventory_jsonl": str(inventory_path),
                        "inventory_jsonl_sha256": inventory_sha,
                        "execution_json": str(execution_path),
                        "execution_sha256": sha256_file(execution_path),
                        "vedtak": execution.get("vedtak"),
                    }
    return None


def _validate_native_ancestor_manifest_cas_chain(
    parent_source: Mapping[str, Any],
    *,
    timeframe: str,
) -> None:
    """Walk every ancestor manifest CAS with explicit cycle/depth bounds."""

    normalized, _policy = native_timeframe_policy(timeframe)
    current = canonical_native_parent_binding_v1(parent_source)
    seen_roots: set[Path] = set()
    for _depth in range(_MAX_CANONICAL_NATIVE_ANCESTOR_DEPTH):
        ancestor_root = Path(str(current["root"])).expanduser()
        ancestor_manifest_path = Path(str(current["manifest_path"])).expanduser()
        if (
            not ancestor_root.is_absolute()
            or ancestor_root.is_symlink()
            or not ancestor_root.is_dir()
            or ancestor_root.resolve() != ancestor_root
            or ancestor_root in seen_roots
            or ancestor_manifest_path != ancestor_root / "MANIFEST.json"
            or ancestor_manifest_path.is_symlink()
            or not ancestor_manifest_path.is_file()
        ):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_PATH_OR_CYCLE_INVALID"
            )
        seen_roots.add(ancestor_root)
        expected_manifest_sha = _require_sha256(
            current["manifest_sha256"],
            timeframe=normalized,
            label="SUCCESSOR_ANCESTOR_MANIFEST",
        )
        if sha256_file(ancestor_manifest_path) != expected_manifest_sha:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_MANIFEST_CAS_MISMATCH"
            )
        ancestor_manifest = _json(
            ancestor_manifest_path,
            label="XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_MANIFEST",
        )
        observed = _native_parent_binding_from_manifest(
            parent_root=ancestor_root,
            parent_manifest_path=ancestor_manifest_path,
            parent_manifest=ancestor_manifest,
            timeframe=normalized,
        )
        if observed != current:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_IDENTITY_MISMATCH"
            )
        schema = ancestor_manifest.get("schema_version")
        if schema == CANONICAL_NATIVE_SOURCE_SCHEMA:
            return
        if (
            schema != CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA
            or ancestor_manifest.get("publication_mode")
            != CANONICAL_NATIVE_SUCCESSOR_MODE
        ):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_SCHEMA_INVALID"
            )
        next_parent = ancestor_manifest.get("parent_source")
        if not isinstance(next_parent, Mapping):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_PARENT_MISSING"
            )
        current = canonical_native_parent_binding_v1(next_parent)
        if current != next_parent:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_PARENT_FIELDS_INVALID"
            )
        next_root = Path(str(current["root"])).expanduser()
        if (
            next_root == ancestor_root
            or next_root in ancestor_root.parents
            or ancestor_root in next_root.parents
        ):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_ROOTS_NOT_DISJOINT"
            )
    raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_ANCESTOR_DEPTH_EXCEEDED")


def _validate_native_successor_envelope_retired(
    manifest: dict[str, Any],
    *,
    parent_source: Mapping[str, Any],
    attestation: dict[str, Any],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parent_root: Path,
) -> dict[str, Any]:
    """Successor envelope for a retention-attested retired parent.

    Every proof derivable from the child manifest and the recorded parent
    binding still runs (interval advance, append contract, overlap interval,
    declared row/append counts, time advance). Only the parent-byte
    re-proofs — manifest CAS re-read, ancestor chain walk, parent year CAS
    and the parent-side overlap re-read — are replaced by the executed
    retention attestation; the downstream child-overlap sha comparison still
    binds the publication-time parent-byte proof through
    ``successor_append.overlap_rows_sha256``.
    """

    normalized, _policy = native_timeframe_policy(timeframe)
    binding = canonical_native_parent_binding_v1(parent_source)
    parent_start = _utc_native_bar(
        binding["requested_start_utc"],
        timeframe=normalized,
        label="SUCCESSOR_PARENT_BINDING_START",
    )
    parent_end = _utc_native_bar(
        binding["requested_end_utc_exclusive"],
        timeframe=normalized,
        label="SUCCESSOR_PARENT_BINDING_END",
    )
    parent_time_max = _utc_native_bar(
        binding["time_max_utc"],
        timeframe=normalized,
        label="SUCCESSOR_PARENT_BINDING_TIME_MAX",
    )
    if parent_start != start or parent_end >= end:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_INTERVAL_NOT_ADVANCING")
    parent_row_count = binding.get("row_count")
    if (
        isinstance(parent_row_count, bool)
        or not isinstance(parent_row_count, int)
        or parent_row_count <= 0
    ):
        raise RuntimeError(
            "XAU_CANONICAL_M5_SUCCESSOR_PARENT_FIELDS_INVALID"
        )
    child_chunks = manifest.get("source_chunks")
    if not isinstance(child_chunks, list) or not child_chunks:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_CHUNKS_INVALID")
    append = manifest.get("successor_append")
    expected_append_keys = {
        "overlap_start_utc",
        "parent_end_utc_exclusive",
        "reused_source_chunks",
        "refetched_source_chunks",
        "parent_overlap_rows",
        "appended_rows",
        "overlap_rows_sha256",
    }
    if not isinstance(append, dict) or set(append) != expected_append_keys:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_APPEND_CONTRACT_INVALID")
    reused = append.get("reused_source_chunks")
    refetched = append.get("refetched_source_chunks")
    # The parent-chunk prefix identity (child_chunks[:reused] ==
    # parent_chunks[:reused]) was proven at publication and needs parent
    # bytes; the child-side arithmetic still binds exactly.
    if (
        isinstance(reused, bool)
        or not isinstance(reused, int)
        or reused < 0
        or isinstance(refetched, bool)
        or not isinstance(refetched, int)
        or refetched <= 0
        or refetched != len(child_chunks) - reused
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_CHUNK_LINEAGE_INVALID")
    overlap_start = _utc_native_bar(
        append.get("overlap_start_utc"),
        timeframe=normalized,
        label="SUCCESSOR_OVERLAP_START",
    )
    declared_parent_end = _utc_native_bar(
        append.get("parent_end_utc_exclusive"),
        timeframe=normalized,
        label="SUCCESSOR_DECLARED_PARENT_END",
    )
    if (
        declared_parent_end != parent_end
        or not start <= overlap_start < parent_end < end
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_OVERLAP_INTERVAL_INVALID")
    for name in ("parent_overlap_rows", "appended_rows"):
        value = append.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise RuntimeError(
                f"XAU_CANONICAL_M5_SUCCESSOR_{name.upper()}_INVALID"
            )
    overlap_sha = _require_sha256(
        append.get("overlap_rows_sha256"),
        timeframe=normalized,
        label="SUCCESSOR_OVERLAP_ROWS",
    )
    return {
        "append": append,
        "overlap_rows_sha256": overlap_sha,
        "overlap_start": overlap_start,
        "parent_end": parent_end,
        "parent_time_max": parent_time_max,
        "parent_manifest": None,
        "parent_root": parent_root,
        "parent_row_count": parent_row_count,
        "retired_parent_attestation": attestation,
    }


def _validate_native_successor_envelope(
    root: Path,
    manifest: dict[str, Any],
    *,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Validate the immutable parent CAS and bounded overlap declaration."""

    normalized, _policy = native_timeframe_policy(timeframe)
    if manifest.get("publication_mode") != CANONICAL_NATIVE_SUCCESSOR_MODE:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_MODE_INVALID")
    parent_source = manifest.get("parent_source")
    if not isinstance(parent_source, dict):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_PARENT_MISSING")
    if canonical_native_parent_binding_v1(parent_source) != parent_source:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_PARENT_FIELDS_INVALID")
    parent_root = Path(str(parent_source["root"])).expanduser()
    parent_manifest_path = Path(
        str(parent_source["manifest_path"])
    ).expanduser()
    child_declared_root = Path(str(manifest.get("out_root") or "")).expanduser()
    if (
        not parent_root.is_absolute()
        or parent_root.is_symlink()
        or parent_root.resolve() != parent_root
        or parent_root == root
        or child_declared_root == parent_root
        or child_declared_root in parent_root.parents
        or parent_root in child_declared_root.parents
        or parent_manifest_path != parent_root / "MANIFEST.json"
        or parent_manifest_path.is_symlink()
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_PARENT_PATH_INVALID")
    if not parent_root.is_dir() or not parent_manifest_path.is_file():
        # Absent parent: admitted ONLY through the executed retention
        # attestation (identity continuity via the recorded per-file
        # inventory); anything unattested fails closed as before.
        attestation = _retired_native_root_attestation(
            parent_root,
            expected_manifest_sha256=_require_sha256(
                parent_source["manifest_sha256"],
                timeframe=normalized,
                label="SUCCESSOR_PARENT_MANIFEST",
            ),
        )
        if attestation is None:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_PARENT_PATH_INVALID"
            )
        return _validate_native_successor_envelope_retired(
            manifest,
            parent_source=parent_source,
            attestation=attestation,
            timeframe=normalized,
            start=start,
            end=end,
            parent_root=parent_root,
        )
    if (
        sha256_file(parent_manifest_path)
        != _require_sha256(
            parent_source["manifest_sha256"],
            timeframe=normalized,
            label="SUCCESSOR_PARENT_MANIFEST",
        )
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_PARENT_MANIFEST_CAS_MISMATCH")
    parent_manifest = _json(
        parent_manifest_path,
        label="XAU_CANONICAL_M5_SUCCESSOR_PARENT_MANIFEST",
    )
    observed_parent = _native_parent_binding_from_manifest(
        parent_root=parent_root,
        parent_manifest_path=parent_manifest_path,
        parent_manifest=parent_manifest,
        timeframe=normalized,
    )
    if observed_parent != parent_source:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_PARENT_IDENTITY_MISMATCH")
    _validate_native_ancestor_manifest_cas_chain(
        parent_source,
        timeframe=normalized,
    )
    for field in (
        "instrument",
        "timeframe",
        "explicit_vedtak_id",
        "source_environment",
        "source_base_url",
    ):
        if parent_manifest.get(field) != manifest.get(field):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_PARENT_CONTRACT_MISMATCH: "
                f"field={field}"
            )
    for field in ("requested_start_utc", "time_min_utc"):
        parent_value = _utc_native_bar(
            parent_manifest.get(field),
            timeframe=normalized,
            label=f"SUCCESSOR_PARENT_{field.upper()}",
        )
        child_value = _utc_native_bar(
            manifest.get(field),
            timeframe=normalized,
            label=f"SUCCESSOR_CHILD_{field.upper()}",
        )
        if parent_value != child_value:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_PARENT_CONTRACT_MISMATCH: "
                f"field={field}"
            )
    parent_start = _utc_native_bar(
        parent_manifest.get("requested_start_utc"),
        timeframe=normalized,
        label="SUCCESSOR_PARENT_START",
    )
    parent_end = _utc_native_bar(
        parent_manifest.get("requested_end_utc_exclusive"),
        timeframe=normalized,
        label="SUCCESSOR_PARENT_END",
    )
    parent_time_max = _utc_native_bar(
        parent_manifest.get("time_max_utc"),
        timeframe=normalized,
        label="SUCCESSOR_PARENT_TIME_MAX",
    )
    if parent_start != start or parent_end >= end:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_INTERVAL_NOT_ADVANCING")

    parent_chunks = parent_manifest.get("source_chunks")
    child_chunks = manifest.get("source_chunks")
    if (
        not isinstance(parent_chunks, list)
        or not parent_chunks
        or not isinstance(child_chunks, list)
        or not child_chunks
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_CHUNKS_INVALID")
    append = manifest.get("successor_append")
    expected_append_keys = {
        "overlap_start_utc",
        "parent_end_utc_exclusive",
        "reused_source_chunks",
        "refetched_source_chunks",
        "parent_overlap_rows",
        "appended_rows",
        "overlap_rows_sha256",
    }
    if not isinstance(append, dict) or set(append) != expected_append_keys:
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_APPEND_CONTRACT_INVALID")
    reused = append.get("reused_source_chunks")
    refetched = append.get("refetched_source_chunks")
    if (
        isinstance(reused, bool)
        or not isinstance(reused, int)
        or not 0 <= reused < len(parent_chunks)
        or isinstance(refetched, bool)
        or not isinstance(refetched, int)
        or refetched <= 0
        or refetched != len(child_chunks) - reused
        or child_chunks[:reused] != parent_chunks[:reused]
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_CHUNK_LINEAGE_INVALID")
    overlap_start = _utc_native_bar(
        append.get("overlap_start_utc"),
        timeframe=normalized,
        label="SUCCESSOR_OVERLAP_START",
    )
    last_parent_start = _utc_native_bar(
        parent_chunks[reused].get("request_from_utc"),
        timeframe=normalized,
        label="SUCCESSOR_PARENT_LAST_CHUNK_START",
    )
    declared_parent_end = _utc_native_bar(
        append.get("parent_end_utc_exclusive"),
        timeframe=normalized,
        label="SUCCESSOR_DECLARED_PARENT_END",
    )
    if (
        overlap_start != last_parent_start
        or declared_parent_end != parent_end
        or not start <= overlap_start < parent_end < end
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SUCCESSOR_OVERLAP_INTERVAL_INVALID")
    for name in ("parent_overlap_rows", "appended_rows"):
        value = append.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise RuntimeError(
                f"XAU_CANONICAL_M5_SUCCESSOR_{name.upper()}_INVALID"
            )
    overlap_sha = _require_sha256(
        append.get("overlap_rows_sha256"),
        timeframe=normalized,
        label="SUCCESSOR_OVERLAP_ROWS",
    )
    return {
        "append": append,
        "overlap_rows_sha256": overlap_sha,
        "overlap_start": overlap_start,
        "parent_end": parent_end,
        "parent_time_max": parent_time_max,
        "parent_manifest": parent_manifest,
        "parent_root": parent_root,
    }


def _validate_canonical_native_source_contract_impl(
    root: Path,
    manifest: dict[str, Any],
    *,
    timeframe: str,
    expected_declared_root: Path,
) -> dict[str, Any]:
    """Require one hash-bound native owner and source-defined closure."""

    normalized, policy = native_timeframe_policy(timeframe)
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        CANONICAL_NATIVE_SOURCE_SCHEMA,
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    }:
        raise RuntimeError(
            "XAU_CANONICAL_M5_MANIFEST_SCHEMA_INVALID: "
            f"observed={schema_version!r}"
        )
    exact = {
        "schema_version": schema_version,
        "producer_owner": CANONICAL_NATIVE_PRODUCER_OWNER,
        "source_kind": "oanda_native_mba_candles",
        "source_endpoint": "/instruments/XAU_USD/candles",
        "source_granularity": normalized,
        "prices": "MBA",
        "timestamp_semantics": "bar_start_utc",
        "bar_duration_seconds": policy["bar_seconds"],
        "decision_available_offset_seconds": policy["bar_seconds"],
        "completion_field": "complete",
        "completion_value": True,
        "market_closure_contract": CANONICAL_NATIVE_CLOSURE_CONTRACT,
        "request_interval_semantics": CANONICAL_NATIVE_REQUEST_INTERVAL_SEMANTICS,
        "source_response_encoding": CANONICAL_NATIVE_SOURCE_RESPONSE_ENCODING,
        "source_chunk_schema": CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA,
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
    if schema_version == CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA:
        expected_keys.update(
            {
                "publication_mode",
                "parent_source",
                "successor_append",
            }
        )
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
    if (
        manifest.get("instrument") != XAU_INSTRUMENT
        or manifest.get("timeframe") != normalized
    ):
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
    start = _utc_native_bar(
        manifest.get("requested_start_utc"),
        timeframe=normalized,
        label="MANIFEST_REQUEST_START",
    )
    end = _utc_native_bar(
        manifest.get("requested_end_utc_exclusive"),
        timeframe=normalized,
        label="MANIFEST_REQUEST_END",
    )
    if end <= start:
        raise RuntimeError("XAU_CANONICAL_M5_REQUEST_INTERVAL_INVALID")
    successor_context: dict[str, Any] | None = None
    if schema_version == CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA:
        successor_context = _validate_native_successor_envelope(
            root,
            manifest,
            timeframe=normalized,
            start=start,
            end=end,
        )
    chunk_days = manifest.get("request_chunk_days")
    if chunk_days != policy["request_chunk_days"]:
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
        CANONICAL_NATIVE_REQUIRED_COLUMNS
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
    expected_source_paths = list(CANONICAL_NATIVE_PRODUCER_SOURCE_FILES)
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
                timeframe=normalized,
                label="PRODUCER_REPO_SOURCE",
            )
        )
        snapshot_relative = _safe_relative_path(
            item.get("snapshot_relative_path"),
            timeframe=normalized,
            label="PRODUCER_SNAPSHOT_SOURCE",
        )
        if snapshot_relative != Path("producer_source") / repo_relative:
            raise RuntimeError("XAU_CANONICAL_M5_PRODUCER_SOURCE_PATH_MISMATCH")
        source_path = root / snapshot_relative
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(
                f"XAU_CANONICAL_M5_PRODUCER_SOURCE_MISSING: {source_path}"
            )
        digest = _require_sha256(
            item.get("sha256"),
            timeframe=normalized,
            label="PRODUCER_SOURCE",
        )
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
            timeframe=normalized,
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
            timeframe=normalized,
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
    successor_overlap_frames: list[pd.DataFrame] = []
    successor_appended_rows = 0
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
        chunk_start = _utc_native_bar(
            metadata.get("request_from_utc"),
            timeframe=normalized,
            label=f"CHUNK_{sequence}_START",
        )
        chunk_end = _utc_native_bar(
            metadata.get("request_to_utc_exclusive"),
            timeframe=normalized,
            label=f"CHUNK_{sequence}_END",
        )
        expected_chunk_end = min(
            chunk_start + pd.Timedelta(days=policy["request_chunk_days"]),
            end,
        )
        if chunk_start != cursor or chunk_end != expected_chunk_end:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_INTERVAL_INVALID")
        cursor = chunk_end
        relative = _safe_relative_path(
            metadata.get("relative_path"),
            timeframe=normalized,
            label=f"CHUNK_{sequence}",
        )
        expected_relative = Path("source_chunks") / f"chunk-{sequence:06d}.json.gz"
        if relative != expected_relative:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_PATH_MISMATCH")
        chunk_path = root / relative
        manifest_chunk_files.append(chunk_path)
        digest = _require_sha256(
            metadata.get("sha256"),
            timeframe=normalized,
            label=f"CHUNK_{sequence}",
        )
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
        payload = _decode_source_chunk(
            compressed,
            timeframe=normalized,
            path=chunk_path,
        )
        if set(payload) != {"schema_version", "request", "response"}:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_PAYLOAD_SCHEMA_INVALID")
        if (
            payload.get("schema_version")
            != CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA
        ):
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_SCHEMA_MISMATCH")
        request = payload.get("request")
        expected_request = {
            "instrument": XAU_INSTRUMENT,
            "from": chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "granularity": normalized,
            "price": "MBA",
        }
        if request != expected_request:
            raise RuntimeError("XAU_CANONICAL_M5_SOURCE_CHUNK_REQUEST_MISMATCH")
        response = payload.get("response")
        frame, stats = canonical_native_frame_from_oanda_response(
            response,
            timeframe=normalized,
            request_start=chunk_start,
            request_end=chunk_end,
        )
        if metadata.get("incomplete_candles") != 0:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SOURCE_CHUNK_INCOMPLETE_CANDLES_FORBIDDEN: "
                f"sequence={sequence}"
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
            source_digest.update(
                canonical_native_rows_bytes(
                    frame,
                    timeframe=normalized,
                )
            )
            if successor_context is not None:
                overlap = frame.loc[
                    (frame["time"] >= successor_context["overlap_start"])
                    & (frame["time"] < successor_context["parent_end"])
                ]
                if not overlap.empty:
                    successor_overlap_frames.append(overlap)
                successor_appended_rows += int(
                    (frame["time"] >= successor_context["parent_end"]).sum()
                )
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
            timeframe=normalized,
            label="CANONICAL_ROWS",
        )
        != source_rows_sha
    ):
        raise RuntimeError("XAU_CANONICAL_M5_SOURCE_ROWS_BINDING_MISMATCH")
    if successor_context is not None and successor_context.get(
        "retired_parent_attestation"
    ):
        # Retention-attested retired parent: the parent-side byte re-proof is
        # replaced by the attestation, and the child overlap rows are proven
        # against the publication-time recorded overlap sha (that sha was
        # proven equal to the parent bytes when the successor was published).
        if not successor_overlap_frames:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_OVERLAP_ROWS_MISSING"
            )
        child_overlap = pd.concat(
            successor_overlap_frames,
            ignore_index=True,
        ).loc[:, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)]
        child_overlap_bytes = canonical_native_rows_bytes(
            child_overlap,
            timeframe=normalized,
        )
        child_overlap_sha = hashlib.sha256(child_overlap_bytes).hexdigest()
        append = successor_context["append"]
        if (
            len(child_overlap) != append["parent_overlap_rows"]
            or child_overlap_sha != successor_context["overlap_rows_sha256"]
            or successor_appended_rows != append["appended_rows"]
            or source_row_count
            <= int(successor_context["parent_row_count"])
            or pd.Timestamp(source_time_max)
            <= successor_context["parent_time_max"]
        ):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_PREFIX_OR_APPEND_MISMATCH"
            )
    elif successor_context is not None:
        parent_manifest = successor_context["parent_manifest"]
        parent_root = successor_context["parent_root"]
        overlap_start = successor_context["overlap_start"]
        parent_end = successor_context["parent_end"]
        parent_overlap_frames: list[pd.DataFrame] = []
        parent_year_hashes = parent_manifest.get("year_sha256")
        if not isinstance(parent_year_hashes, dict):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_PARENT_YEAR_HASHES_INVALID"
            )
        overlap_year_keys = [
            key
            for key in sorted(parent_year_hashes)
            if overlap_start.year
            <= int(str(key).split("=", 1)[1])
            <= parent_end.year
        ]
        for key in overlap_year_keys:
            part = parent_root / key / "part-000.parquet"
            expected_part_sha = parent_year_hashes.get(key)
            if (
                not isinstance(expected_part_sha, str)
                or part.is_symlink()
                or not part.is_file()
                or sha256_file(part) != expected_part_sha
            ):
                raise RuntimeError(
                    "XAU_CANONICAL_M5_SUCCESSOR_PARENT_YEAR_CAS_MISMATCH: "
                    f"{key}"
                )
            parent_year = _validate_canonical_native_frame(
                pd.read_parquet(part),
                timeframe=normalized,
                label=f"SUCCESSOR_PARENT_{key}",
                allow_empty=False,
            )
            selected = parent_year.loc[
                (parent_year["time"] >= overlap_start)
                & (parent_year["time"] < parent_end)
            ]
            if not selected.empty:
                parent_overlap_frames.append(selected)
        if not parent_overlap_frames or not successor_overlap_frames:
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_OVERLAP_ROWS_MISSING"
            )
        parent_overlap = pd.concat(
            parent_overlap_frames,
            ignore_index=True,
        ).loc[:, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)]
        child_overlap = pd.concat(
            successor_overlap_frames,
            ignore_index=True,
        ).loc[:, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)]
        parent_overlap_bytes = canonical_native_rows_bytes(
            parent_overlap,
            timeframe=normalized,
        )
        child_overlap_bytes = canonical_native_rows_bytes(
            child_overlap,
            timeframe=normalized,
        )
        overlap_sha = hashlib.sha256(parent_overlap_bytes).hexdigest()
        append = successor_context["append"]
        if (
            len(parent_overlap) != append["parent_overlap_rows"]
            or parent_overlap_bytes != child_overlap_bytes
            or overlap_sha != successor_context["overlap_rows_sha256"]
            or successor_appended_rows != append["appended_rows"]
            or source_row_count
            <= int(parent_manifest.get("row_count") or 0)
            or pd.Timestamp(source_time_max)
            <= successor_context["parent_time_max"]
        ):
            raise RuntimeError(
                "XAU_CANONICAL_M5_SUCCESSOR_PREFIX_OR_APPEND_MISMATCH"
            )

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
        if observed_columns != list(CANONICAL_NATIVE_REQUIRED_COLUMNS):
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
        frame = _validate_canonical_native_frame(
            pd.read_parquet(part),
            timeframe=normalized,
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
        parquet_digest.update(
            canonical_native_rows_bytes(
                frame,
                timeframe=normalized,
            )
        )
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
        timeframe=normalized,
        label="MANIFEST_PAYLOAD",
    )
    if observed_manifest_hash != canonical_json_sha256(without_manifest_hash):
        raise RuntimeError("XAU_CANONICAL_M5_MANIFEST_PAYLOAD_HASH_MISMATCH")
    descriptor = {
        **exact,
        "explicit_vedtak_id": vedtak,
        "source_environment": environment,
        "source_base_url": str(manifest["source_base_url"]),
        "schema_required_cols": list(CANONICAL_NATIVE_REQUIRED_COLUMNS),
        "schema_optional_cols": [],
        "request_chunk_days": policy["request_chunk_days"],
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
    if successor_context is not None:
        descriptor.update(
            {
                "publication_mode": CANONICAL_NATIVE_SUCCESSOR_MODE,
                "parent_source": json.loads(
                    json.dumps(manifest["parent_source"], sort_keys=True)
                ),
                "successor_append": json.loads(
                    json.dumps(manifest["successor_append"], sort_keys=True)
                ),
            }
        )
    return descriptor


def _validate_canonical_native_source_contract(
    root: Path,
    manifest: dict[str, Any],
    *,
    timeframe: str,
    expected_declared_root: Path,
) -> dict[str, Any]:
    """Expose timeframe-exact errors while retaining M5 error compatibility."""

    normalized, _policy = native_timeframe_policy(timeframe)
    try:
        return _validate_canonical_native_source_contract_impl(
            root,
            manifest,
            timeframe=normalized,
            expected_declared_root=expected_declared_root,
        )
    except RuntimeError as exc:
        if normalized == "M5":
            raise
        message = str(exc).replace(
            "XAU_CANONICAL_M5_",
            f"XAU_CANONICAL_{normalized}_",
        )
        raise RuntimeError(message) from exc


def validate_canonical_native_source_bundle(
    physical_root: Path | str,
    *,
    timeframe: str,
    expected_declared_root: Path | str,
) -> dict[str, Any]:
    """Validate a final bundle or its hidden pre-publication staging root."""

    normalized, _policy = native_timeframe_policy(timeframe)
    prefix = f"XAU_CANONICAL_{normalized}"
    physical_arg = Path(physical_root).expanduser()
    if (
        not physical_arg.is_absolute()
        or physical_arg.is_symlink()
        or not physical_arg.is_dir()
    ):
        raise RuntimeError(
            f"{prefix}_PHYSICAL_ROOT_INVALID: {physical_arg}"
        )
    physical = physical_arg.resolve()
    if physical != physical_arg:
        raise RuntimeError(
            f"{prefix}_PHYSICAL_ROOT_NOT_CANONICAL: "
            f"raw={physical_arg} resolved={physical}"
        )
    declared_arg = Path(expected_declared_root).expanduser()
    if not declared_arg.is_absolute():
        raise RuntimeError(
            f"{prefix}_EXPECTED_DECLARED_ROOT_NOT_ABSOLUTE: {declared_arg}"
        )
    declared = declared_arg.resolve(strict=False)
    if declared != declared_arg:
        raise RuntimeError(
            f"{prefix}_EXPECTED_DECLARED_ROOT_NOT_CANONICAL: "
            f"raw={declared_arg} resolved={declared}"
        )
    manifest = _json(
        physical / "MANIFEST.json",
        label=f"{prefix}_MANIFEST",
    )
    if (
        manifest.get("instrument") != XAU_INSTRUMENT
        or manifest.get("timeframe") != normalized
    ):
        raise RuntimeError(f"{prefix}_IDENTITY_MISMATCH")
    return _validate_canonical_native_source_contract(
        physical,
        manifest,
        timeframe=normalized,
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
    expected_timeframe, _policy = native_timeframe_policy(timeframe)
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
    descriptor.update(
        _validate_canonical_native_source_contract(
            canonical_root,
            manifest,
            timeframe=expected_timeframe,
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
    if set(raw) != set(observed):
        raise RuntimeError(
            f"XAU_TAPE_CANONICAL_{timeframe}_DESCRIPTOR_SCHEMA_MISMATCH: "
            f"missing={sorted(set(observed) - set(raw))} "
            f"extra={sorted(set(raw) - set(observed))}"
        )
    for key, observed_value in observed.items():
        if raw.get(key) != observed_value:
            raise RuntimeError(
                f"XAU_TAPE_CANONICAL_{timeframe}_{key.upper()}_MISMATCH: "
                f"declared={raw.get(key)!r} observed={observed_value!r}"
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
    if (
        not (root / "REPAIR_MANIFEST.json").exists()
        and (root / "MANIFEST.json").exists()
    ):
        # A strict native v3 bootstrap or v4 successor source root is complete
        # tape provenance by
        # construction: retained response evidence, source↔parquet identity,
        # year hashes and market-closure semantics are all validated by the
        # canonical descriptor. It has no repair lineage and no collector
        # snapshot; run identity is bound by the consuming event, and
        # currency is enforced downstream against its exact time_max_utc.
        descriptor = canonical_xau_source_descriptor_v1(root, timeframe="M5")
        return {
            "schema_version": descriptor["schema_version"],
            "tape_root": str(root),
            **{
                key: descriptor[key]
                for key in (
                    "manifest_path",
                    "manifest_sha256",
                    "instrument",
                    "timeframe",
                    "explicit_vedtak_id",
                    "requested_start_utc",
                    "requested_end_utc_exclusive",
                    "row_count",
                    "time_min_utc",
                    "time_max_utc",
                    "canonical_rows_sha256",
                    "producer_git_commit",
                    "year_sha256",
                    "year_rows",
                )
            },
        }
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
