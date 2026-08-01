"""Emit compact PASS proof for the current immutable V3 pair source cascade."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.contracts.gx1_scope_v1 import require_offline_scope
from gx1.features.htf_features import load_multi_tf_cache
from gx1.scripts.audit_seq513_source_cascade_v1 import CURRENT_PAIR_SCHEMA_VERSION


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    cache = load_multi_tf_cache(cache_dir)
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": CURRENT_PAIR_SCHEMA_VERSION,
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
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


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
