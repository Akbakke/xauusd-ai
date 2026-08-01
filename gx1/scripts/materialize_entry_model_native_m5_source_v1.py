"""Materialize the current offline Entry M5 source surface.

The enriched M5 feature frame intentionally contains only the model-native
surface.  TRAIN ranking and spread-aware labels additionally require the
literal native bid/ask closes.  This producer joins those raw fields by exact
timestamp from the same immutable pair generation; it never fills, resamples,
or resolves an alternate source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    entry_exit_shared_feature_base_contract,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
)
from gx1.features.entry_smart_context import ENTRY_SMART_CTX_FEATURE_NAMES
from gx1.features.htf_features import attach_default_regime_v4_v2_scalars
from gx1.features.regime_v4_features import REGIME_V4_SOURCE_COLS
from gx1.scripts.build_entry_exit_m1_enriched_frame_v1 import (
    _load_native_frame,
    _require_pair_binding,
)


RAW_MARKET_COLUMNS = (
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
RANK_COLUMNS = ("time", "high", "low", "close", "bid_close", "ask_close")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def materialize_m5_source(
    *,
    enriched_parquet: Path,
    native_m5_root: Path,
    pair_manifest: Path,
    output_parquet: Path,
    dataset_run_id: str,
    pair_generation_id: str,
) -> dict[str, Any]:
    require_offline_scope("featurebase_build")
    enriched = Path(enriched_parquet).expanduser().resolve()
    output = Path(output_parquet).expanduser().resolve()
    if (
        not enriched.is_file()
        or enriched.is_symlink()
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
    ):
        raise RuntimeError("M5_SOURCE_INPUT_OR_OUTPUT_INVALID")
    if not dataset_run_id or not pair_generation_id:
        raise RuntimeError("M5_SOURCE_LINEAGE_ID_INVALID")

    enriched_frame = pd.read_parquet(enriched)
    if "time" not in enriched_frame.columns:
        raise RuntimeError("M5_SOURCE_ENRICHED_TIME_MISSING")
    enriched_frame["time"] = pd.to_datetime(
        enriched_frame["time"], utc=True, errors="raise"
    )
    enriched_frame = enriched_frame.sort_values("time", kind="mergesort")
    enriched_times = pd.DatetimeIndex(enriched_frame["time"]).as_unit("ns")
    if (
        enriched_frame.empty
        or enriched_times.has_duplicates
        or not enriched_times.is_monotonic_increasing
        or np.any(enriched_times.asi8 % (5 * 60 * 1_000_000_000) != 0)
    ):
        raise RuntimeError("M5_SOURCE_ENRICHED_TIME_GEOMETRY_INVALID")

    native, native_identity = _load_native_frame(native_m5_root, timeframe="M5")
    _require_pair_binding(
        pair_manifest_path=pair_manifest,
        pair_generation_id=pair_generation_id,
        source_identity=native_identity,
        native_frame=native,
        timeframe="M5",
    )
    native["time"] = pd.to_datetime(native["time"], utc=True, errors="raise")
    native = native.sort_values("time", kind="mergesort")
    native_regime = native.set_index("time")[
        ["open", "high", "low", "close", "volume"]
    ].copy()
    attach_default_regime_v4_v2_scalars(
        native_regime,
        base_bar_duration=pd.Timedelta(minutes=5),
    )
    native_regime = native_regime.reset_index()
    native = native.loc[:, list(RAW_MARKET_COLUMNS) + ["time"]]
    native = native.rename(
        columns={name: f"native__{name}" for name in RAW_MARKET_COLUMNS}
    )
    if native["time"].duplicated().any():
        raise RuntimeError("M5_SOURCE_NATIVE_TIME_DUPLICATE")

    merged = enriched_frame.merge(
        native,
        on="time",
        how="left",
        sort=False,
        suffixes=("", "__native"),
        validate="one_to_one",
    )
    native_columns = [f"native__{name}" for name in RAW_MARKET_COLUMNS]
    if merged[native_columns].isna().any().any():
        raise RuntimeError("M5_SOURCE_NATIVE_EXACT_TIME_JOIN_INCOMPLETE")
    for name in RAW_MARKET_COLUMNS:
        native_name = f"native__{name}"
        if name in {"open", "high", "low", "close"}:
            left = pd.to_numeric(merged[name], errors="coerce").to_numpy(
                dtype=np.float64
            )
            right = pd.to_numeric(merged[native_name], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.array_equal(left, right):
                raise RuntimeError(f"M5_SOURCE_ENRICHED_NATIVE_{name.upper()}_MISMATCH")
        if name == "volume":
            merged[name] = merged[native_name].astype(np.int64)
        elif name not in {"open", "high", "low", "close"}:
            merged[name] = merged[native_name]
        merged.drop(columns=[native_name], inplace=True)

    ordered = ["time"] + list(RAW_MARKET_COLUMNS) + [
        name
        for name in merged.columns
        if name not in {"time", *RAW_MARKET_COLUMNS}
    ]
    result_frame = merged.loc[:, ordered]
    source_owned_fields = tuple(
        dict.fromkeys((*REGIME_V4_SOURCE_COLS, "volume"))
    )
    regime_source = result_frame[["time"]].merge(
        native_regime,
        on="time",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    regime_available = set(regime_source.columns)
    regime_missing = [
        name
        for name in source_owned_fields
        if name not in regime_available and name != "D1_dist_from_ema200_atr"
    ]
    if regime_missing:
        raise RuntimeError(f"M5_SOURCE_REGIME_FIELDS_MISSING: {regime_missing}")
    for name in source_owned_fields:
        if name == "D1_dist_from_ema200_atr":
            if name not in result_frame.columns:
                raise RuntimeError("M5_SOURCE_REGIME_D1_DISTANCE_MISSING")
            continue
        incoming = pd.to_numeric(regime_source[name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(incoming).all():
            raise RuntimeError(f"M5_SOURCE_REGIME_NONFINITE: {name}")
        if name in result_frame.columns:
            existing = pd.to_numeric(result_frame[name], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.array_equal(existing, incoming):
                raise RuntimeError(f"M5_SOURCE_REGIME_{name.upper()}_MISMATCH")
        else:
            result_frame[name] = incoming
    ranker_owned_derivations = tuple(
        dict.fromkeys(
            (
                *MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
                *MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
                *ENTRY_SMART_CTX_FEATURE_NAMES,
            )
        )
    )
    result_frame = result_frame.drop(
        columns=[name for name in ranker_owned_derivations if name in result_frame.columns]
    )
    numeric = result_frame.loc[:, list(RAW_MARKET_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("M5_SOURCE_MARKET_NONFINITE")
    if (numeric["ask_close"] < numeric["bid_close"]).any():
        raise RuntimeError("M5_SOURCE_BID_ASK_GEOMETRY_INVALID")
    result_frame.to_parquet(output, index=False)

    manifest = {
        "schema_version": "gx1_entry_model_native_m5_source_surface_v1",
        "decision": "PASS",
        "dataset_run_id": dataset_run_id,
        "pair_generation_id": pair_generation_id,
        "timeframe": "M5",
        "anchor_timeframe": "M5",
        "enriched_source": str(enriched),
        "enriched_source_sha256": _sha256_file(enriched),
        "native_m5_source": native_identity,
        "pair_manifest": str(Path(pair_manifest).expanduser().resolve()),
        "pair_manifest_sha256": _sha256_file(Path(pair_manifest).expanduser().resolve()),
        "output_parquet": str(output),
        "output_parquet_sha256": _sha256_file(output),
        "rows": int(len(result_frame)),
        "columns": list(result_frame.columns),
        "raw_market_columns": list(RAW_MARKET_COLUMNS),
        "rank_source_columns": list(RANK_COLUMNS),
        "source_owned_fields": list(source_owned_fields),
        "regime_source_rehydrated_from_native_m5": True,
        "ranker_owned_derivations_removed": list(ranker_owned_derivations),
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "causal_contract": {
            "exact_timestamp_join": True,
            "future_rows_used": False,
            "native_bid_ask_reused": True,
            "resample_or_fill": False,
            "same_feature_owner_as_entry_exit": True,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    Path(str(output) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-parquet", type=Path, required=True)
    parser.add_argument("--native-m5-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--pair-generation-id", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_m5_source(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
