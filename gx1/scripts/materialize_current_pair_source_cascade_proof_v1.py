"""Emit compact PASS proof for the current immutable V3 pair source cascade."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.contracts.gx1_scope_v1 import require_offline_scope
from gx1.contracts.xau_tape_provenance_v1 import (
    SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
)
from gx1.features.htf_features import load_multi_tf_v4_cache


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    candidate = path.expanduser().resolve()
    if path.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"CURRENT_PAIR_SOURCE_{label}_MISSING_OR_SYMLINK")
    return candidate


def _json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _regular(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"CURRENT_PAIR_SOURCE_{label}_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"CURRENT_PAIR_SOURCE_{label}_JSON_OBJECT_REQUIRED")
    return payload


def _same(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise RuntimeError(
            f"CURRENT_PAIR_SOURCE_{label}_MISMATCH: "
            f"actual={actual!r} expected={expected!r}"
        )


def _same_path(actual: Any, expected: Path, *, label: str) -> None:
    resolved = Path(str(actual or "")).expanduser().resolve()
    _same(resolved, expected.resolve(), label=label)


def _utc(value: Any, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(f"CURRENT_PAIR_SOURCE_{label}_TIME_INVALID") from exc
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise RuntimeError(f"CURRENT_PAIR_SOURCE_{label}_TIME_NOT_UTC")
    return parsed.tz_convert("UTC")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {temporary}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def emit(
    *,
    run_id: str,
    source_parquet: Path,
    canonical_v2_parquet: Path,
    mtf_cache_dir: Path,
    pair_manifest: Path,
    required_history_start: str,
    out: Path,
) -> dict[str, Any]:
    require_offline_scope("featurebase_build")
    run_id = require_entry_run_id(run_id)
    source = source_parquet.expanduser().resolve()
    canonical = canonical_v2_parquet.expanduser().resolve()
    cache_dir = mtf_cache_dir.expanduser().resolve()
    pair_path = pair_manifest.expanduser().resolve()
    target = out.expanduser().resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or not canonical.is_file()
        or canonical.is_symlink()
        or not cache_dir.is_dir()
        or cache_dir.is_symlink()
        or not pair_path.is_file()
        or pair_path.is_symlink()
        or target.exists()
        or target.is_symlink()
        or target.parent != source.parent
    ):
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_PATH_OR_INPUT_INVALID")
    frame = pd.read_parquet(source, columns=["time", "open", "high", "low", "close", "bid_close", "ask_close"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if frame.empty or frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_TIME_INVALID")
    history = pd.Timestamp(required_history_start)
    if history.tzinfo is None:
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_HISTORY_NOT_UTC")
    history = history.tz_convert("UTC")
    time_min = pd.Timestamp(frame["time"].iloc[0]).tz_convert("UTC")
    time_max = pd.Timestamp(frame["time"].iloc[-1]).tz_convert("UTC")
    if time_min > history:
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_HISTORY_NOT_COVERED")
    cache = load_multi_tf_v4_cache(cache_dir)
    cache_source = Path(str(cache.m5_prebuilt_source)).expanduser().resolve()
    if cache_source.is_symlink() or not cache_source.is_file():
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_MTF_SOURCE_INVALID")
    cache_source_sha256 = _sha256_file(cache_source)
    if cache_source_sha256 != str(cache.m5_prebuilt_source_sha256):
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_MTF_SOURCE_HASH_MISMATCH")
    market_columns = ["time", "open", "high", "low", "close", "volume"]
    source_market = pd.read_parquet(source, columns=market_columns)
    cache_market = pd.read_parquet(cache_source, columns=market_columns)
    source_time = pd.DatetimeIndex(
        pd.to_datetime(source_market.pop("time"), utc=True, errors="raise")
    ).as_unit("ns")
    cache_time = pd.DatetimeIndex(
        pd.to_datetime(cache_market.pop("time"), utc=True, errors="raise")
    ).as_unit("ns")
    source_values = source_market.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    cache_values = cache_market.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not source_time.equals(cache_time)
        or not np.array_equal(source_values, cache_values)
    ):
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_MTF_MARKET_IDENTITY_MISMATCH")
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS",
        "entry_run_id": run_id,
        "event_root": str(source.parent),
        "artifacts": {
            "source_parquet_path": str(source),
            "source_parquet_sha256": _sha256_file(source),
            "canonical_v2_path": str(canonical),
            "canonical_v2_sha256": _sha256_file(canonical),
            "multi_tf_manifest_sha256": _sha256_file(cache_dir / "manifest.json"),
            "multi_tf_cache_identity_sha256": cache.cache_identity_sha256,
            "multi_tf_source_path": str(cache_source),
            "multi_tf_source_sha256": cache_source_sha256,
            "pair_manifest_path": str(pair_path),
            "pair_manifest_sha256": _sha256_file(pair_path),
            "pair_generation_id": str(pair.get("pair_generation_id") or ""),
        },
        "contracts": {
            "required_history_start_utc": history.isoformat(),
            "required_history_start_covered": True,
            "time_min_utc": time_min.isoformat(),
            "time_max_utc": time_max.isoformat(),
            "no_fallback": True,
            "future_rows_used": False,
            "multi_tf_source_market_identity": True,
        },
    }
    _atomic_json(target, payload)
    return payload


def validate_current_pair_source_cascade_proof(
    proof_path: Path,
    *,
    expected_run_id: str,
    expected_source_parquet: Path,
    expected_canonical_v2_parquet: Path,
    expected_mtf_cache_dir: Path,
    expected_history_start_utc: object,
    expected_time_max_utc: object,
) -> dict[str, Any]:
    """Revalidate the sole current-pair source proof at a consumer boundary."""

    run_id = require_entry_run_id(expected_run_id)
    source = _regular(expected_source_parquet, label="BOUND_SOURCE")
    canonical = _regular(
        expected_canonical_v2_parquet,
        label="BOUND_CANONICAL",
    )
    cache_dir = expected_mtf_cache_dir.expanduser().resolve()
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise RuntimeError("CURRENT_PAIR_SOURCE_BOUND_MTF_CACHE_INVALID")
    proof_file = _regular(proof_path, label="PROOF")
    if proof_file.parent != source.parent.resolve():
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_NOT_SOURCE_LOCAL")
    proof = _json(proof_file, label="PROOF")
    required_keys = {
        "schema_version",
        "created_utc",
        "decision",
        "entry_run_id",
        "event_root",
        "artifacts",
        "contracts",
    }
    if set(proof) != required_keys:
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_KEYS_INVALID")
    _same(
        proof.get("schema_version"),
        SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
        label="PROOF_SCHEMA",
    )
    _same(proof.get("decision"), "PASS", label="PROOF_DECISION")
    _same(proof.get("entry_run_id"), run_id, label="PROOF_RUN_ID")
    _same_path(
        proof.get("event_root"),
        source.parent.resolve(),
        label="PROOF_EVENT_ROOT",
    )
    artifacts = proof.get("artifacts")
    contracts = proof.get("contracts")
    if not isinstance(artifacts, dict) or not isinstance(contracts, dict):
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_SECTIONS_INVALID")
    if set(artifacts) != {
        "source_parquet_path",
        "source_parquet_sha256",
        "canonical_v2_path",
        "canonical_v2_sha256",
        "multi_tf_manifest_sha256",
        "multi_tf_cache_identity_sha256",
        "multi_tf_source_path",
        "multi_tf_source_sha256",
        "pair_manifest_path",
        "pair_manifest_sha256",
        "pair_generation_id",
    }:
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_ARTIFACT_KEYS_INVALID")
    if set(contracts) != {
        "required_history_start_utc",
        "required_history_start_covered",
        "time_min_utc",
        "time_max_utc",
        "no_fallback",
        "future_rows_used",
        "multi_tf_source_market_identity",
    }:
        raise RuntimeError("CURRENT_PAIR_SOURCE_PROOF_CONTRACT_KEYS_INVALID")

    _same_path(
        artifacts.get("source_parquet_path"),
        source,
        label="SOURCE_PATH",
    )
    _same(
        artifacts.get("source_parquet_sha256"),
        _sha256_file(source),
        label="SOURCE_HASH",
    )
    _same_path(
        artifacts.get("canonical_v2_path"),
        canonical,
        label="CANONICAL_PATH",
    )
    _same(
        artifacts.get("canonical_v2_sha256"),
        _sha256_file(canonical),
        label="CANONICAL_HASH",
    )
    cache = load_multi_tf_v4_cache(cache_dir)
    _same(
        artifacts.get("multi_tf_manifest_sha256"),
        _sha256_file(cache_dir / "manifest.json"),
        label="MTF_MANIFEST_HASH",
    )
    _same(
        artifacts.get("multi_tf_cache_identity_sha256"),
        cache.cache_identity_sha256,
        label="MTF_IDENTITY",
    )
    cache_source = _regular(
        Path(str(cache.m5_prebuilt_source)),
        label="MTF_SOURCE",
    )
    _same_path(
        artifacts.get("multi_tf_source_path"),
        cache_source,
        label="MTF_SOURCE_PATH",
    )
    cache_source_sha256 = _sha256_file(cache_source)
    _same(
        artifacts.get("multi_tf_source_sha256"),
        cache_source_sha256,
        label="MTF_SOURCE_HASH",
    )
    _same(
        str(cache.m5_prebuilt_source_sha256),
        cache_source_sha256,
        label="MTF_BOUND_SOURCE_HASH",
    )
    pair_manifest = _regular(
        Path(str(artifacts.get("pair_manifest_path") or "")),
        label="PAIR_MANIFEST",
    )
    _same(
        artifacts.get("pair_manifest_sha256"),
        _sha256_file(pair_manifest),
        label="PAIR_MANIFEST_HASH",
    )
    pair = _json(pair_manifest, label="PAIR_MANIFEST")
    _same(
        artifacts.get("pair_generation_id"),
        pair.get("pair_generation_id"),
        label="PAIR_GENERATION_ID",
    )
    expected_history = _utc(
        expected_history_start_utc,
        label="EXPECTED_HISTORY",
    )
    expected_time_max = _utc(
        expected_time_max_utc,
        label="EXPECTED_TIME_MAX",
    )
    _same(
        _utc(contracts.get("required_history_start_utc"), label="HISTORY"),
        expected_history,
        label="HISTORY",
    )
    _same(
        _utc(contracts.get("time_max_utc"), label="TIME_MAX"),
        expected_time_max,
        label="TIME_MAX",
    )
    _same(
        contracts.get("required_history_start_covered"),
        True,
        label="HISTORY_COVERED",
    )
    _same(contracts.get("no_fallback"), True, label="NO_FALLBACK")
    _same(contracts.get("future_rows_used"), False, label="FUTURE_ROWS")
    _same(
        contracts.get("multi_tf_source_market_identity"),
        True,
        label="MTF_MARKET_IDENTITY",
    )
    return {
        "path": str(proof_file),
        "sha256": _sha256_file(proof_file),
        "schema_version": SEQ513_SOURCE_CASCADE_PAIR_PROOF_SCHEMA_VERSION,
        "entry_run_id": run_id,
        "event_root": str(source.parent.resolve()),
        "source_parquet_path": str(source),
        "source_parquet_sha256": str(artifacts["source_parquet_sha256"]),
        "canonical_v2_path": str(canonical),
        "canonical_v2_sha256": str(artifacts["canonical_v2_sha256"]),
        "multi_tf_cache_dir": str(cache_dir),
        "multi_tf_manifest_sha256": str(
            artifacts["multi_tf_manifest_sha256"]
        ),
        "multi_tf_cache_identity_sha256": str(
            artifacts["multi_tf_cache_identity_sha256"]
        ),
        "multi_tf_source_path": str(artifacts["multi_tf_source_path"]),
        "multi_tf_source_sha256": str(artifacts["multi_tf_source_sha256"]),
        "pair_manifest_path": str(pair_manifest),
        "pair_manifest_sha256": str(artifacts["pair_manifest_sha256"]),
        "pair_generation_id": str(artifacts["pair_generation_id"]),
        "history_start_utc": expected_history.isoformat(),
        "time_max_utc": expected_time_max.isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--canonical-v2-parquet", type=Path, required=True)
    parser.add_argument("--mtf-cache-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--required-history-start", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(emit(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
