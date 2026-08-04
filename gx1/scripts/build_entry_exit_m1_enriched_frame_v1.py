"""Build causal native M1/M5 frames through the same feature owners.

One implementation runs at an explicit one-minute Exit clock or five-minute
Entry clock.  Each route consumes only its own native closed rows, then builds
closed MTF OHLCV before features.  It never combines M1/M5 values in front of
the owners or resamples computed M1 features upward.  The older
``EXPANDED_BASE34_CTX16CAT6`` artifact is intentionally not an input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.entry_exit_feature_base_v1 import (  # noqa: E402
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
    EXIT_DECISION_BAR_SECONDS,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (  # noqa: E402
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.contracts.entry_model_native_state_v2 import (  # noqa: E402
    load_train_rank_reference_v2,
)
from gx1.contracts.xau_tape_provenance_v1 import (  # noqa: E402
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope  # noqa: E402
from gx1.execution.v12_ctx_augment_live import (  # noqa: E402
    augment_canonical_v3,
)
from gx1.features.htf_features import (  # noqa: E402
    attach_default_regime_v4_v2_scalars,
    build_multi_tf_per_bar_features_v4,
)
from gx1.features.entry_smart_context import (  # noqa: E402
    add_entry_smart_context_features,
)
from gx1.time.session_detector import decision_availability  # noqa: E402
from gx1.scripts.augment_forward_outcome_v2 import (  # noqa: E402
    attach_group_a_dip_struct_ctx_columns_parallel,
)
from gx1.scripts.materialize_build_canonical_features_v2 import (  # noqa: E402
    build_canonical_v2,
)


TIMEFRAME_SPECS = {
    "M1": {
        "duration": pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS),
        "seconds": EXIT_DECISION_BAR_SECONDS,
        "lineage_key": "m1",
        "label": "m1",
    },
    "M5": {
        "duration": pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS),
        "seconds": ENTRY_DECISION_BAR_SECONDS,
        "lineage_key": "m5",
        "label": "m5",
    },
}
RAW_COLUMNS = tuple(CANONICAL_NATIVE_REQUIRED_COLUMNS)
OUTPUT_COLUMNS = tuple(
    dict.fromkeys(
        (
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            *MODEL_NATIVE_BASE_FIELDS,
            *MODEL_NATIVE_CTX_CONT_FIELDS,
            *MODEL_NATIVE_CTX_CAT_FIELDS,
        )
    )
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_regular(path: Path, *, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"M1_ENRICHED_{label}_INVALID: {path}")
    return path


def _source_manifest_identity(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"M1_ENRICHED_SOURCE_ROOT_INVALID: {root}")
    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"M1_ENRICHED_SOURCE_MANIFEST_INVALID: {manifest_path}")
    parts = sorted(root.glob("year=*/part-*.parquet"))
    if not parts:
        raise RuntimeError(f"M1_ENRICHED_SOURCE_PARTS_MISSING: {root}")
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "part_paths": [str(path) for path in parts],
        "part_sha256": {str(path): _sha256_file(path) for path in parts},
    }


def _load_native_frame(
    root: Path,
    *,
    timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    identity = _source_manifest_identity(root)
    frames: list[pd.DataFrame] = []
    for raw_path in identity["part_paths"]:
        frame = pd.read_parquet(raw_path)
        if tuple(frame.columns) != RAW_COLUMNS:
            raise RuntimeError(
                f"{label}_ENRICHED_SOURCE_SCHEMA_INVALID: "
                f"{raw_path} expected={RAW_COLUMNS} observed={tuple(frame.columns)}"
            )
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_EMPTY")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_TIME_INVALID")
    frame = frame.sort_values("time", kind="mergesort").reset_index(drop=True)
    times = pd.DatetimeIndex(frame["time"]).as_unit("ns")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_TIME_ORDER_INVALID")
    if np.any(times.asi8 % int(spec["duration"].value) != 0):
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_OFF_{timeframe}_GRID")
    numeric = frame.loc[:, list(RAW_COLUMNS[1:])].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_NONFINITE")
    high = numeric["high"].to_numpy(dtype=np.float64)
    low = numeric["low"].to_numpy(dtype=np.float64)
    open_ = numeric["open"].to_numpy(dtype=np.float64)
    close = numeric["close"].to_numpy(dtype=np.float64)
    if (
        np.any(open_ <= 0.0)
        or np.any(low <= 0.0)
        or np.any(high < low)
        or np.any(high < open_)
        or np.any(high < close)
        or np.any(low > open_)
        or np.any(low > close)
        or np.any(numeric["volume"].to_numpy(dtype=np.float64) <= 0.0)
    ):
        raise RuntimeError(f"{label}_ENRICHED_SOURCE_OHLCV_GEOMETRY_INVALID")
    return frame, identity


def _checkpoint_key(
    *,
    source_identity: dict[str, Any],
    rank_reference_sha256: str,
    dataset_run_id: str,
    pair_generation_id: str,
    timeframe: str,
) -> str:
    spec = TIMEFRAME_SPECS[timeframe]
    return _canonical_sha256(
        {
            "schema_version": "gx1_entry_exit_enriched_frame_checkpoint_key_v1",
            "timeframe": timeframe,
            "source_manifest_sha256": source_identity["manifest_sha256"],
            "part_sha256": source_identity["part_sha256"],
            "rank_reference_sha256": rank_reference_sha256,
            "dataset_run_id": dataset_run_id,
            "pair_generation_id": pair_generation_id,
            "base_bar_seconds": spec["seconds"],
            "shared_contract": entry_exit_shared_feature_base_contract(),
        }
    )


def _require_pair_binding(
    *,
    pair_manifest_path: Path,
    pair_generation_id: str,
    source_identity: dict[str, Any],
    native_frame: pd.DataFrame | None = None,
    timeframe: str = "M1",
    native_m1: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if native_frame is None:
        native_frame = native_m1
    if native_frame is None:
        raise RuntimeError("ENTRY_EXIT_ENRICHED_NATIVE_FRAME_MISSING")
    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    manifest_path = _require_regular(pair_manifest_path, label="PAIR_MANIFEST")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}_ENRICHED_PAIR_MANIFEST_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}_ENRICHED_PAIR_MANIFEST_INVALID")
    if payload.get("pair_generation_id") != pair_generation_id:
        raise RuntimeError(f"{label}_ENRICHED_PAIR_GENERATION_ID_MANIFEST_MISMATCH")
    lineage = payload.get("lineage")
    native_sources = lineage.get("native_sources") if isinstance(lineage, dict) else None
    bound_source = (
        native_sources.get(spec["lineage_key"])
        if isinstance(native_sources, dict)
        else None
    )
    if not isinstance(bound_source, dict):
        raise RuntimeError(
            f"{label}_ENRICHED_PAIR_NATIVE_{spec['lineage_key'].upper()}_LINEAGE_MISSING"
        )
    expected = {
        "root": source_identity["root"],
        "manifest_path": source_identity["manifest_path"],
        "manifest_sha256": source_identity["manifest_sha256"],
        "row_count": int(len(native_frame)),
        "time_min_utc": pd.Timestamp(native_frame["time"].iloc[0]).isoformat(),
        "time_max_utc": pd.Timestamp(native_frame["time"].iloc[-1]).isoformat(),
    }
    observed = {name: bound_source.get(name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            f"{label}_ENRICHED_PAIR_NATIVE_{spec['lineage_key'].upper()}_BINDING_MISMATCH: "
            f"observed={observed} expected={expected}"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "pair_generation_id": pair_generation_id,
        f"native_{spec['lineage_key']}": expected,
    }


def _build_enriched_frame(
    *,
    native_root: Path,
    timeframe: str,
    rank_reference_npz: Path,
    rank_reference_sha256: str,
    pair_manifest_path: Path,
    output_parquet: Path,
    manifest_path: Path,
    checkpoint_dir: Path,
    dataset_run_id: str,
    pair_generation_id: str,
    workers: int,
    checkpoint_chunk_rows: int = 4096,
) -> dict[str, Any]:
    if timeframe not in TIMEFRAME_SPECS:
        raise RuntimeError(f"ENTRY_EXIT_ENRICHED_TIMEFRAME_INVALID: {timeframe}")
    spec = TIMEFRAME_SPECS[timeframe]
    label = spec["label"].upper()
    require_offline_scope("featurebase_build")
    contract = require_entry_exit_shared_feature_base_contract(
        entry_exit_shared_feature_base_contract(),
        context=f"{label}_ENRICHED_PRODUCER",
    )
    if not isinstance(dataset_run_id, str) or not dataset_run_id:
        raise RuntimeError(f"{label}_ENRICHED_DATASET_RUN_ID_INVALID")
    if (
        not isinstance(pair_generation_id, str)
        or len(pair_generation_id) != 64
        or any(ch not in "0123456789abcdef" for ch in pair_generation_id)
    ):
        raise RuntimeError(f"{label}_ENRICHED_PAIR_GENERATION_ID_INVALID")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers != 1
    ):
        raise RuntimeError(f"{label}_ENRICHED_WORKERS_MUST_EQUAL_ONE")
    if (
        isinstance(checkpoint_chunk_rows, bool)
        or not isinstance(checkpoint_chunk_rows, int)
        or checkpoint_chunk_rows <= 0
    ):
        raise RuntimeError(f"{label}_ENRICHED_CHECKPOINT_CHUNK_ROWS_INVALID")
    if (
        not isinstance(rank_reference_sha256, str)
        or len(rank_reference_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in rank_reference_sha256)
    ):
        raise RuntimeError(f"{label}_ENRICHED_RANK_REFERENCE_SHA256_INVALID")

    output = Path(output_parquet).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if output.exists() or output.is_symlink() or manifest.exists() or manifest.is_symlink():
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_ALREADY_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if checkpoint.is_symlink():
        raise RuntimeError(f"{label}_ENRICHED_CHECKPOINT_SYMLINK")
    checkpoint.mkdir(parents=True, exist_ok=True)

    rank_path = _require_regular(rank_reference_npz, label="RANK_REFERENCE")
    observed_rank_sha256 = _sha256_file(rank_path)
    if observed_rank_sha256 != rank_reference_sha256:
        raise RuntimeError(
            f"{label}_ENRICHED_RANK_REFERENCE_HASH_MISMATCH: "
            f"observed={observed_rank_sha256} expected={rank_reference_sha256}"
        )
    rank_reference = load_train_rank_reference_v2(
        rank_path,
        expected_sha256=rank_reference_sha256,
    )

    raw, source_identity = _load_native_frame(
        native_root,
        timeframe=timeframe,
    )
    pair_binding = _require_pair_binding(
        pair_manifest_path=pair_manifest_path,
        pair_generation_id=pair_generation_id,
        source_identity=source_identity,
        native_frame=raw,
        timeframe=timeframe,
    )
    indexed_raw = raw.set_index("time").sort_index()
    checkpoint_key = _checkpoint_key(
        source_identity=source_identity,
        rank_reference_sha256=rank_reference_sha256,
        dataset_run_id=dataset_run_id,
        pair_generation_id=pair_generation_id,
        timeframe=timeframe,
    )

    canonical = build_canonical_v2(
        raw,
        base_bar_duration=spec["duration"],
    )
    canonical["time"] = pd.to_datetime(canonical["time"], utc=True, errors="raise")
    canonical = canonical.set_index("time").sort_index()
    attach_default_regime_v4_v2_scalars(
        canonical,
        base_bar_duration=spec["duration"],
    )
    canonical = augment_canonical_v3(
        canonical,
        indexed_raw,
        rank_reference=rank_reference,
        base_bar_duration=spec["duration"],
    )

    multi_tf = build_multi_tf_per_bar_features_v4(
        indexed_raw[["open", "high", "low", "close", "volume"]],
        base_bar_duration=spec["duration"],
    )
    enriched = attach_group_a_dip_struct_ctx_columns_parallel(
        canonical,
        multi_tf=multi_tf,
        journal_label=f"{spec['label']}_enriched_frame",
        workers=workers,
        checkpoint_dir=checkpoint,
        checkpoint_key=checkpoint_key,
        checkpoint_chunk_rows=checkpoint_chunk_rows,
        context_m5=indexed_raw[["open", "high", "low", "close"]],
        base_bar_duration=spec["duration"],
    )

    # Complete the same model-native surface independently on the selected
    # native clock. These are owned transforms, not neutral fills: missing
    # source evidence raises below.
    decision_ts = decision_availability(
        enriched.index,
        bar_duration=spec["duration"],
        context=f"{label}_ENRICHED_TIME_FEATURES",
    )
    hour = decision_ts.hour.to_numpy(dtype=np.float32)
    dow = decision_ts.dayofweek.to_numpy(dtype=np.float32)
    enriched["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    enriched["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    enriched["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    enriched["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
    swing_state = pd.to_numeric(
        enriched["smc_swing_state"], errors="raise"
    ).to_numpy(dtype=np.float64)
    premium_discount = pd.to_numeric(
        enriched["smc_premium_discount"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(swing_state).all()
        or not np.isfinite(premium_discount).all()
        or not np.equal(swing_state, np.rint(swing_state)).all()
        or np.any((swing_state < 0.0) | (swing_state > 4.0))
        or np.any((premium_discount < 0.0) | (premium_discount > 1.0))
    ):
        raise RuntimeError(f"{label}_ENRICHED_SMC_PREMIUM_STATE_SOURCE_INVALID")
    enriched["smc_premium_state"] = (
        premium_discount * np.equal(swing_state, 0.0)
    ).astype(np.float32)
    add_entry_smart_context_features(enriched)

    if not enriched.index.is_unique or not enriched.index.is_monotonic_increasing:
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_TIME_ORDER_INVALID")
    missing = [name for name in OUTPUT_COLUMNS if name != "time" and name not in enriched.columns]
    if missing:
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_FIELDS_MISSING: {missing}")
    output_frame = enriched.loc[:, [name for name in OUTPUT_COLUMNS if name != "time"]].copy()
    output_frame.insert(0, "time", enriched.index)
    for name in OUTPUT_COLUMNS[1:]:
        values = pd.to_numeric(output_frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label}_ENRICHED_OUTPUT_NONFINITE: {name}")
    cats = output_frame[list(MODEL_NATIVE_CTX_CAT_FIELDS)].to_numpy(dtype=np.float64)
    if not np.equal(cats, np.rint(cats)).all():
        raise RuntimeError(f"{label}_ENRICHED_OUTPUT_CTX_CAT_NONINTEGER")
    output_frame.to_parquet(output, index=False)

    result = {
        "schema_version": ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
        "decision": "PASS",
        "shared_feature_base_contract": contract,
        "dataset_run_id": dataset_run_id,
        "pair_generation_id": pair_generation_id,
        "timeframe": timeframe,
        "base_bar_seconds": spec["seconds"],
        f"native_{spec['lineage_key']}_source": source_identity,
        "pair_binding": pair_binding,
        "rank_reference_npz": str(rank_path),
        "rank_reference_sha256": rank_reference_sha256,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_key": checkpoint_key,
        "output_parquet": str(output),
        "output_parquet_sha256": _sha256_file(output),
        "rows": int(len(output_frame)),
        "columns": list(output_frame.columns),
        "required_base_fields": list(MODEL_NATIVE_BASE_FIELDS),
        "required_context_cont_fields": list(MODEL_NATIVE_CTX_CONT_FIELDS),
        "required_context_cat_fields": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        "causal_contract": {
            "future_rows_used": False,
            f"decision_uses_closed_{spec['label']}_bar": True,
            "same_feature_owner_as_entry": True,
            "same_specialist_stack_as_entry": True,
            "native_resolution_values": True,
            "cross_resolution_value_copy": False,
            "computed_m1_feature_resampling": False,
            "old_m1_artifacts_consumed": False,
            "missing_field_fill": False,
            "frame_relative_bucket_fallback": False,
        },
    }
    result["manifest_sha256"] = _canonical_sha256(result)
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_m1_enriched_frame(**kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for the existing offline M1 route."""

    if "native_m1_root" in kwargs and "native_root" not in kwargs:
        kwargs["native_root"] = kwargs.pop("native_m1_root")
    return _build_enriched_frame(timeframe="M1", **kwargs)


def build_m5_enriched_frame(**kwargs: Any) -> dict[str, Any]:
    """Build the same owned surface at the Entry M5 decision clock."""

    return _build_enriched_frame(timeframe="M5", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-m1-root", type=Path)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--timeframe", choices=tuple(TIMEFRAME_SPECS), default="M1")
    parser.add_argument("--rank-reference-npz", required=True, type=Path)
    parser.add_argument("--rank-reference-sha256", required=True)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    native_root = args.native_root or args.native_m1_root
    if native_root is None:
        parser.error("one of --native-root or --native-m1-root is required")
    result = _build_enriched_frame(
        native_root=native_root,
        timeframe=args.timeframe,
        rank_reference_npz=args.rank_reference_npz,
        rank_reference_sha256=args.rank_reference_sha256,
        pair_manifest_path=args.pair_manifest,
        output_parquet=args.output_parquet,
        manifest_path=args.manifest_path,
        checkpoint_dir=args.checkpoint_dir,
        dataset_run_id=args.dataset_run_id,
        pair_generation_id=args.pair_generation_id,
        workers=args.workers,
        checkpoint_chunk_rows=args.checkpoint_chunk_rows,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
