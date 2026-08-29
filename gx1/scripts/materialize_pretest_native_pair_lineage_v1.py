#!/usr/bin/env python3
"""Publish a minimal immutable lineage binding for direct pre-TEST M1 and M5.

The MTF-cache calibration and the M1 Exit-lifecycle rebuild both require a
single hash-bound statement of the two direct OANDA clock sources.  This is not
the broader canonical feature-generation pair manifest: it contains no TEST
paths, no features, no labels and no derived market data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from gx1.contracts.gx1_scope_v1 import require_offline_scope

TEST_BOUNDARY_UTC = "2026-07-01T00:00:00+00:00"
PAIR_LINEAGE_SCHEMA_VERSION = "gx1_pretest_native_pair_lineage_v1"
_SOURCE_SCHEMAS = frozenset(
    {"gx1_direct_m5_pretest_source_v1", "gx1_direct_native_pretest_source_v2"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_direct_source(path: Path, *, timeframe: str) -> dict[str, Any]:
    artifact = path.expanduser()
    if (
        not artifact.is_absolute()
        or artifact.is_symlink()
        or not artifact.is_file()
        or artifact.resolve(strict=True) != artifact
    ):
        raise RuntimeError("PRETEST_NATIVE_PAIR_SOURCE_MANIFEST_INVALID")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("PRETEST_NATIVE_PAIR_SOURCE_MANIFEST_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("PRETEST_NATIVE_PAIR_SOURCE_MANIFEST_INVALID")
    output_raw = str(payload.get("output_parquet") or "")
    output = Path(output_raw).expanduser()
    expected_sha = str(payload.get("output_parquet_sha256") or "")
    if (
        payload.get("schema_version") not in _SOURCE_SCHEMAS
        or payload.get("timeframe") != timeframe
        or payload.get("test_boundary_utc") != TEST_BOUNDARY_UTC
        or payload.get("test_accessed") is not False
        or payload.get("source_requested_end_utc_exclusive") != TEST_BOUNDARY_UTC
        or not output.is_absolute()
        or output.is_symlink()
        or not output.is_file()
        or output.resolve(strict=True) != output
        or len(expected_sha) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha)
        or _sha256_file(output) != expected_sha
        or not isinstance(payload.get("row_count"), int)
        or int(payload["row_count"]) <= 0
        or not isinstance(payload.get("time_max_utc"), str)
        or payload["time_max_utc"] >= TEST_BOUNDARY_UTC
    ):
        raise RuntimeError("PRETEST_NATIVE_PAIR_SOURCE_BOUNDARY_INVALID")
    return {
        "source_manifest_path": str(artifact),
        "source_manifest_sha256": _sha256_file(artifact),
        "source_manifest_payload_sha256": payload.get("manifest_payload_sha256"),
        "source_parquet": str(output),
        "source_parquet_sha256": expected_sha,
        "row_count": int(payload["row_count"]),
        "time_min_utc": payload.get("time_min_utc"),
        "time_max_utc": payload["time_max_utc"],
    }


def materialize_pretest_native_pair_lineage(
    *,
    m1_source_manifest: Path,
    m5_source_manifest: Path,
    output_json: Path,
) -> dict[str, Any]:
    """Create a no-replace M1/M5 binding used only by pre-TEST producers."""

    require_offline_scope("featurebase_build")
    m1 = _read_direct_source(m1_source_manifest, timeframe="M1")
    m5 = _read_direct_source(m5_source_manifest, timeframe="M5")
    destination = output_json.expanduser()
    if (
        not destination.is_absolute()
        or destination.exists()
        or destination.is_symlink()
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
        or destination.resolve(strict=False) != destination
    ):
        raise RuntimeError("PRETEST_NATIVE_PAIR_OUTPUT_INVALID")
    pair_identity = _canonical_sha256({"m1": m1, "m5": m5})
    payload: dict[str, Any] = {
        "schema_version": PAIR_LINEAGE_SCHEMA_VERSION,
        "pair_generation_id": f"pretest_native_{pair_identity}",
        "pair_symbol": "XAUUSD",
        "test_boundary_utc": TEST_BOUNDARY_UTC,
        "test_accessed": False,
        "m1": m1,
        "m5": m5,
    }
    payload["manifest_payload_sha256"] = _canonical_sha256(payload)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                    "utf-8"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        return payload
    except FileExistsError as exc:
        raise RuntimeError("PRETEST_NATIVE_PAIR_OUTPUT_EXISTS") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-source-manifest", type=Path, required=True)
    parser.add_argument("--m5-source-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_pretest_native_pair_lineage(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
