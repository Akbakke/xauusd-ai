#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_schema_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_prebuilt_sidecars(
    *,
    parquet_path: Path,
    required_all_features: list[str],
    manifest: dict[str, Any],
) -> dict[str, str]:
    schema_names = _parquet_schema_names(parquet_path)
    schema_manifest_path = parquet_path.with_suffix(".schema_manifest.json")
    manifest_path = parquet_path.with_suffix(".manifest.json")
    schema_manifest = {
        "parquet_path": str(parquet_path),
        "schema_names": schema_names,
        "cols_total": int(len(schema_names)),
        "schema_version": manifest.get("schema_version", "truth_monday_week_extension_v1"),
        "ctx_cont_dim": manifest.get("ctx_cont_dim"),
        "ctx_cat_dim": manifest.get("ctx_cat_dim"),
        "required_all_features": list(required_all_features),
        "required_time_column": "time",
        "has_time_column": "time" in schema_names,
        "forbidden_index_artifacts": ["__index_level_0__", "index"],
        "forbidden_found": [c for c in schema_names if c in {"__index_level_0__", "index"}],
    }
    if not schema_manifest["has_time_column"]:
        raise RuntimeError(f"PREBUILT_SCHEMA_TIME_COLUMN_MISSING: {parquet_path}")
    if schema_manifest["forbidden_found"]:
        raise RuntimeError(f"PREBUILT_SCHEMA_FORBIDDEN_INDEX_ARTIFACTS: {schema_manifest['forbidden_found']}")
    _write_json(schema_manifest_path, schema_manifest)
    _write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "schema_manifest": str(schema_manifest_path)}


def _load_partitioned_time(root: Path, years: list[int], *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for year in years:
        p = root / f"year={year}" / "part-000.parquet"
        if not p.exists():
            raise FileNotFoundError(f"partition missing: {p}")
        df = pd.read_parquet(p)
        if "time" not in df.columns:
            raise RuntimeError(f"time column missing: {p}")
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        df = df[(df["time"] >= start) & (df["time"] < end)]
        if len(df):
            frames.append(df)
    if not frames:
        raise RuntimeError("no rows loaded from partitioned tape")
    out = pd.concat(frames, ignore_index=True).sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last")
    out = out.set_index("time").sort_index()
    return out


def _load_contract(engine: Path) -> list[str]:
    path = engine / "gx1" / "xgb" / "contracts" / "xgb_input_features_base28_v1.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    features = obj.get("features")
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise RuntimeError(f"invalid feature contract: {path}")
    return list(features)


def _build_base_features(raw_m5: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    from gx1.features.basic_v1 import build_basic_v1
    from gx1.features.feature_state import FeatureState
    from gx1.utils.feature_context import reset_feature_state, set_feature_state

    df = raw_m5[["open", "high", "low", "close"]].astype(np.float64).copy()
    df["ts"] = pd.DatetimeIndex(raw_m5.index).tz_convert("UTC")
    token = set_feature_state(FeatureState())
    old_env = {
        key: os.environ.get(key)
        for key in [
            "GX1_FEATURE_BUILD_DISABLED",
            "GX1_REPLAY_USE_PREBUILT_FEATURES",
            "GX1_REPLAY",
            "FEATURE_BUILD_TIMEOUT_MS",
            "GX1_FEATURE_USE_NP_ROLLING",
        ]
    }
    try:
        os.environ.pop("GX1_FEATURE_BUILD_DISABLED", None)
        os.environ.pop("GX1_REPLAY_USE_PREBUILT_FEATURES", None)
        os.environ.pop("GX1_REPLAY", None)
        os.environ["FEATURE_BUILD_TIMEOUT_MS"] = "1200000"
        os.environ["GX1_FEATURE_USE_NP_ROLLING"] = "1"
        feature_df, _ = build_basic_v1(df)
    finally:
        reset_feature_state(token)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    missing = [c for c in features if c not in feature_df.columns]
    if missing:
        raise RuntimeError(f"BASE_FEATURES_MISSING: {missing}")
    out = feature_df[features].copy()
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise RuntimeError("BASE_FEATURES_NONFINITE")
    if "ts" not in feature_df.columns:
        raise RuntimeError("BASE_FEATURES_TS_MISSING")
    out_index = pd.DatetimeIndex(pd.to_datetime(feature_df["ts"], utc=True, errors="coerce"))
    if out_index.isna().any():
        raise RuntimeError("BASE_FEATURES_TS_PARSE_FAILED")
    if out_index.tz is None:
        out.index = out_index.tz_localize("UTC")
    else:
        out.index = out_index.tz_convert("UTC")
    mid = ((raw_m5["high"].astype(float) + raw_m5["low"].astype(float)) * 0.5).reindex(out.index)
    if mid.isna().any() or (mid.to_numpy(dtype=float) <= 0).any():
        raise RuntimeError("ATR_BPS_DERIVE_MID_INVALID")
    out["atr_bps"] = (out["_v1_atr14"].to_numpy(dtype=float) / np.maximum(mid.to_numpy(dtype=float), 1e-9)) * 1e4
    if not np.isfinite(out["atr_bps"].to_numpy(dtype=float)).all():
        raise RuntimeError("ATR_BPS_DERIVE_NONFINITE")
    out.index.name = "time"
    return out.sort_index()


def _expand_to_raw_index(model_df: pd.DataFrame, raw_m1: pd.DataFrame) -> pd.DataFrame:
    raw_index = pd.DatetimeIndex(raw_m1.index).sort_values().unique()
    if raw_index.min() < model_df.index.min():
        raise RuntimeError(f"raw index starts before model features: raw_min={raw_index.min()} model_min={model_df.index.min()}")
    expanded = model_df.reindex(raw_index, method="ffill")
    if expanded.isna().any().any():
        bad = [c for c in expanded.columns if expanded[c].isna().any()]
        raise RuntimeError(f"expanded prebuilt has NaNs: {bad[:20]}")
    expanded["is_model_bar"] = expanded.index.isin(model_df.index)
    expanded.index.name = "time"
    return expanded


def materialize(
    *,
    engine: Path,
    data_root: Path,
    start: str,
    end_exclusive: str,
    output_root: Path,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_exclusive)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    years = list(range(int(start_ts.year), int(end_ts.year) + 1))

    m5_root = data_root / "data" / "oanda" / "canonical" / "xauusd_m5_bid_ask__CANONICAL"
    m1_root = data_root / "data" / "oanda" / "canonical" / "xauusd_m1_bid_ask__CANONICAL"
    raw_m5 = _load_partitioned_time(m5_root, years, start=start_ts, end=end_ts)
    raw_m1 = _load_partitioned_time(m1_root, years, start=start_ts, end=end_ts)

    features = _load_contract(engine)
    base_df = _build_base_features(raw_m5, features)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"monday_week_prebuilt_extension_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"xauusd_m5_BASE34_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}_MODEL_BARS.parquet"
    ctx_path = out_dir / f"xauusd_m5_BASE34_CTX16CAT6_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}_MODEL_BARS.parquet"
    expanded_path = out_dir / f"xauusd_m1_EXPANDED_BASE34_CTX16CAT6_{start_ts:%Y%m%d}_{end_ts:%Y%m%d}_RAW_INDEX.parquet"
    base_df.to_parquet(base_path, index=True)
    base_sidecars = _write_prebuilt_sidecars(
        parquet_path=base_path,
        required_all_features=[c for c in base_df.columns if c != "is_model_bar"],
        manifest={
            "kind": "TRUTH_MONDAY_WEEK_PREBUILT_BASE_MODEL_BARS_V1",
            "schema_version": "base_model_bars_v1",
            "parquet_path": str(base_path),
            "sha256": _sha256(base_path),
            "rows": int(len(base_df)),
            "cols": int(len(_parquet_schema_names(base_path))),
            "ctx_cont_dim": 0,
            "ctx_cat_dim": 0,
            "not_promoted": True,
        },
    )

    from gx1.scripts.add_ctx_cont_columns_to_prebuilt import run_add_ctx_cont_columns

    diagnostics_path = ctx_path.with_name(ctx_path.stem + ".ctx_diagnostics.json")
    run_add_ctx_cont_columns(
        prebuilt_path=base_path,
        raw_m5_paths=[m5_root],
        output_parquet=ctx_path,
        diagnostics_path=diagnostics_path,
        ctx_cont_dim=16,
        ctx_cat_dim=6,
        tape_root=m5_root,
    )
    ctx_df = pd.read_parquet(ctx_path)
    # add_ctx_cont now emits `time` as a plain COLUMN with a RangeIndex (one-truth convention
    # change documented in FASE2B_REBUILD_ORDER.md step 3 fix #2). Restore the DatetimeIndex the
    # downstream `_expand_to_raw_index` reindex requires by promoting the `time` column when present.
    if not isinstance(ctx_df.index, pd.DatetimeIndex):
        if "time" in ctx_df.columns:
            ctx_df = ctx_df.set_index(pd.DatetimeIndex(pd.to_datetime(ctx_df["time"], utc=True)))
            ctx_df = ctx_df.drop(columns=["time"])
        else:
            raise RuntimeError("ctx model-bar parquet lost DatetimeIndex")
    if ctx_df.index.tz is None:
        ctx_df.index = ctx_df.index.tz_localize("UTC")
    else:
        ctx_df.index = ctx_df.index.tz_convert("UTC")
    ctx_df.index.name = "time"
    ctx_df.to_parquet(ctx_path, index=True, compression="snappy")
    ctx_required = [c for c in ctx_df.columns if c != "is_model_bar"]
    ctx_sidecars = _write_prebuilt_sidecars(
        parquet_path=ctx_path,
        required_all_features=ctx_required,
        manifest={
            "kind": "TRUTH_MONDAY_WEEK_PREBUILT_CTX_MODEL_BARS_V1",
            "schema_version": "ctx_model_bars_v1",
            "parquet_path": str(ctx_path),
            "sha256": _sha256(ctx_path),
            "rows": int(len(ctx_df)),
            "cols": int(len(_parquet_schema_names(ctx_path))),
            "ctx_cont_dim": 16,
            "ctx_cat_dim": 6,
            "not_promoted": True,
        },
    )
    expanded = _expand_to_raw_index(ctx_df, raw_m1)
    expanded.to_parquet(expanded_path, index=True, compression="snappy")
    expanded_required = [c for c in expanded.columns if c != "is_model_bar"]
    expanded_sidecars = _write_prebuilt_sidecars(
        parquet_path=expanded_path,
        required_all_features=expanded_required,
        manifest={
            "kind": "TRUTH_MONDAY_WEEK_PREBUILT_EXPANDED_RAW_INDEX_V1",
            "schema_version": "expanded_raw_index_v1",
            "parquet_path": str(expanded_path),
            "sha256": _sha256(expanded_path),
            "rows": int(len(expanded)),
            "cols": int(len(_parquet_schema_names(expanded_path))),
            "ctx_cont_dim": 16,
            "ctx_cat_dim": 6,
            "expanded_from_source": str(ctx_path),
            "expanded_to_raw_index": True,
            "raw_source": str(m1_root),
            "raw_index_count": int(len(raw_m1)),
            "model_bar_count": int(expanded["is_model_bar"].sum()),
            "not_promoted": True,
        },
    )

    manifest = {
        "artifact_name": "TRUTH_MONDAY_WEEK_PREBUILT_EXTENSION_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT_NOT_PROMOTED",
        "start": start_ts.isoformat(),
        "end_exclusive": end_ts.isoformat(),
        "m5_root": str(m5_root),
        "m1_root": str(m1_root),
        "base_model_bar_path": str(base_path),
        "ctx_model_bar_path": str(ctx_path),
        "expanded_raw_index_path": str(expanded_path),
        "base_rows": int(len(base_df)),
        "ctx_rows": int(len(ctx_df)),
        "expanded_rows": int(len(expanded)),
        "expanded_model_bar_rows": int(expanded["is_model_bar"].sum()),
        "raw_m1_rows": int(len(raw_m1)),
        "raw_m5_rows": int(len(raw_m5)),
        "expanded_ts_min": expanded.index.min().isoformat(),
        "expanded_ts_max": expanded.index.max().isoformat(),
        "model_bar_ts_min": ctx_df.index.min().isoformat(),
        "model_bar_ts_max": ctx_df.index.max().isoformat(),
        "expanded_sha256": _sha256(expanded_path),
        "ctx_sha256": _sha256(ctx_path),
        "base_sha256": _sha256(base_path),
        "base_sidecars": base_sidecars,
        "ctx_sidecars": ctx_sidecars,
        "expanded_sidecars": expanded_sidecars,
        "not_promoted": True,
        "promotion_note": "Do not update CURRENT_MANIFEST/canonical truth until replay preflight and calendar audit are accepted.",
    }
    manifest_path = out_dir / "TRUTH_MONDAY_WEEK_PREBUILT_EXTENSION_V1.json"
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=Path("/home/andre2/src/GX1_ENGINE"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/andre2/GX1_DATA"))
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-04-20T00:00:00Z")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES"),
    )
    args = parser.parse_args()
    manifest = materialize(
        engine=args.engine.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        start=args.start,
        end_exclusive=args.end_exclusive,
        output_root=args.output_root.expanduser().resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
