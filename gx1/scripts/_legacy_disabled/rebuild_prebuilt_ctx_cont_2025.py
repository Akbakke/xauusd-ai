#!/home/andre2/venvs/gx1/bin/python
"""
Rebuild 2025 prebuilt parquet with ctx_cont + ctx_cat columns (deterministic, TRUTH-style).

Uses non-quarantined builder:
  gx1.scripts.add_ctx_cont_columns_to_prebuilt (no _quarantine dependency)

Output path includes ctx dims + timestamp, e.g.:
  .../xauusd_m5_2025_features_v13_refined3_20260215_150308__CTX_CONT4_CAT5_20260220_120000.parquet
  .../xauusd_m5_2025_features_v13_refined3_20260215_150308__CTX_CONT6_CAT6_20260220_120000.parquet

Usage:
  GX1_ENGINE=/home/andre2/src/GX1_ENGINE GX1_DATA=/home/andre2/GX1_DATA \
  python -m gx1.scripts.rebuild_prebuilt_ctx_cont_2025 --truth-file gx1/configs/canonical_truth_signal_only.json

Notes:
- ctx dims are selected in this order:
  (1) CLI --ctx-cont-dim/--ctx-cat-dim if provided
  (2) lock/bundle metadata if present
  (3) defaults to cont=4, cat=5

- Optional: --prebuilt-input (default: canonical_prebuilt_parquet from truth)
- Optional: --raw-m5 (default: GX1_DATA/data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet)
- In the M1/M5 split, this remains the historical model-feature source used to derive ctx columns;
  the runtime split between raw M1 exit and M5 model bars is handled downstream.

After first successful build: for IOV2 training, set canonical_prebuilt_parquet in
gx1/configs/canonical_truth_signal_only.json to the printed output path (versioned under
GX1_DATA/data/data/prebuilt/ctx{cont}cat{cat}_{timestamp}/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


_ALLOWED_CONT = (2, 4, 6, 11, 16)
_ALLOWED_CAT = (5, 6)


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _infer_ctx_dims_from_lock(engine: Path) -> Optional[Tuple[int, int]]:
    """
    Best-effort inference of ctx dims from lock/bundle metadata.
    Returns (ctx_cont_dim, ctx_cat_dim) or None if unknown.
    """
    # Common candidate locations (adjust if your repo uses different paths)
    candidates = [
        engine / "gx1" / "models" / "MASTER_TRANSFORMER_LOCK.json",
        engine / "gx1" / "models" / "bundle_metadata.json",
    ]

    for p in candidates:
        j = _read_json_if_exists(p)
        if not j:
            continue

        # Try a few key names to be robust
        for k_cont, k_cat in [
            ("ctx_cont_dim", "ctx_cat_dim"),
            ("context_cont_dim", "context_cat_dim"),
            ("ctx_cont", "ctx_cat"),
        ]:
            if k_cont in j and k_cat in j:
                try:
                    return int(j[k_cont]), int(j[k_cat])
                except Exception:
                    pass

    return None


def _resolve_truth_file(engine: Path, truth_file_arg: Path) -> Path:
    """
    Resolve truth file path:
    - If user provided an absolute path, use it.
    - Else resolve relative to engine root.
    - Also allow env GX1_CANONICAL_TRUTH_FILE as fallback.
    """
    # If arg is explicitly provided and exists, prefer it
    tf = truth_file_arg
    if tf is not None:
        tf = Path(tf)
        if tf.is_absolute() and tf.exists():
            return tf
        # Try relative to current working dir first
        if tf.exists():
            return tf
        # Try relative to engine root
        tf2 = (engine / tf).resolve()
        if tf2.exists():
            return tf2

    env_tf = os.environ.get("GX1_CANONICAL_TRUTH_FILE", "").strip()
    if env_tf:
        env_p = Path(env_tf).expanduser().resolve()
        if env_p.exists():
            return env_p

    # Fall back to engine-relative default if it exists
    default = (engine / "gx1" / "configs" / "canonical_truth_signal_only.json").resolve()
    if default.exists():
        return default

    return Path(truth_file_arg)


def _load_prebuilt_from_truth(truth_file: Path) -> Path:
    truth = json.loads(truth_file.read_text(encoding="utf-8"))
    p = Path(truth.get("canonical_prebuilt_parquet", "")).expanduser().resolve()
    return p


def _validate_dims(ctx_cont_dim: int, ctx_cat_dim: int) -> None:
    if ctx_cont_dim not in _ALLOWED_CONT:
        raise ValueError(f"ctx_cont_dim must be one of {_ALLOWED_CONT}, got {ctx_cont_dim}")
    if ctx_cat_dim not in _ALLOWED_CAT:
        raise ValueError(f"ctx_cat_dim must be one of {_ALLOWED_CAT}, got {ctx_cat_dim}")


def main() -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Rebuild prebuilt 2025 with ctx_cont + ctx_cat columns (contracted, versioned output)"
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        default=Path("gx1/configs/canonical_truth_signal_only.json"),
        help="Canonical truth JSON (for canonical_prebuilt_parquet if --prebuilt-input not set). "
             "If not found, will try GX1_CANONICAL_TRUTH_FILE env.",
    )
    parser.add_argument(
        "--prebuilt-input",
        type=Path,
        default=None,
        help="Input prebuilt parquet (default: canonical_prebuilt_parquet from truth).",
    )
    parser.add_argument(
        "--raw-m5",
        type=Path,
        default=None,
        help="Raw model-feature parquet for D1/H1/M15/H4 derived features "
             "(default: GX1_DATA/data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet).",
    )
    parser.add_argument(
        "--tape-root",
        type=Path,
        default=None,
        help="Canonical tape root for micro features (default: GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL).",
    )
    parser.add_argument(
        "--ctx-cont-dim",
        type=int,
        default=None,
        help="Override ctx_cont_dim (allowed: 2,4,6,11,16). Default: inferred from lock/bundle, else 4.",
    )
    parser.add_argument(
        "--ctx-cat-dim",
        type=int,
        default=None,
        help="Override ctx_cat_dim (allowed: 5,6). Default: inferred from lock/bundle, else 5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths/dims only, do not run builder.",
    )
    parser.add_argument(
        "--expand-to-raw-index",
        action="store_true",
        help="Expand the built prebuilt parquet to the raw tape index with forward-fill, preserving the source model-bar timestamps via is_model_bar.",
    )
    args = parser.parse_args()

    engine = Path(os.environ.get("GX1_ENGINE", "/home/andre2/src/GX1_ENGINE")).expanduser().resolve()
    data = Path(os.environ.get("GX1_DATA", "/home/andre2/GX1_DATA")).expanduser().resolve()

    # Resolve truth file
    truth_file = _resolve_truth_file(engine, args.truth_file)
    if not args.prebuilt_input:
        if not truth_file.exists():
            print(f"[rebuild_prebuilt_ctx] truth file not found: {truth_file}", file=sys.stderr)
            print("[rebuild_prebuilt_ctx] Hint: set GX1_CANONICAL_TRUTH_FILE or pass --truth-file", file=sys.stderr)
            return 1
        args.prebuilt_input = _load_prebuilt_from_truth(truth_file)

    prebuilt_input = Path(args.prebuilt_input).expanduser().resolve()
    if not prebuilt_input.exists():
        print(f"[rebuild_prebuilt_ctx] prebuilt not found: {prebuilt_input}", file=sys.stderr)
        return 1

    # Resolve ctx dims
    inferred = _infer_ctx_dims_from_lock(engine)
    ctx_cont_dim = args.ctx_cont_dim if args.ctx_cont_dim is not None else (inferred[0] if inferred else 4)
    ctx_cat_dim = args.ctx_cat_dim if args.ctx_cat_dim is not None else (inferred[1] if inferred else 5)

    try:
        _validate_dims(ctx_cont_dim, ctx_cat_dim)
    except Exception as e:
        print(f"[rebuild_prebuilt_ctx] {e}", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = prebuilt_input.stem
    base_stem = stem.split("__")[0] if "__" in stem else stem

    # Versioned output under GX1_DATA/data/data/prebuilt/ctx{cont}cat{cat}_{timestamp}/
    prebuilt_base = data / "data" / "data" / "prebuilt"
    version_dir = prebuilt_base / f"ctx{ctx_cont_dim}cat{ctx_cat_dim}_{ts}"
    version_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = version_dir / f"{base_stem}__CTX_CONT{ctx_cont_dim}_CAT{ctx_cat_dim}_{ts}.parquet"

    raw_default = data / "data" / "data" / "_staging" / "XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet"
    raw_m5_path = Path(args.raw_m5 or raw_default).expanduser().resolve()
    tape_default = data / "data" / "oanda" / "canonical" / "xauusd_m5_bid_ask__CANONICAL"
    tape_root = Path(args.tape_root or tape_default).expanduser().resolve()

    print(f"[rebuild_prebuilt_ctx] engine: {engine}")
    print(f"[rebuild_prebuilt_ctx] data:   {data}")
    print(f"[rebuild_prebuilt_ctx] truth:  {truth_file if truth_file.exists() else '(missing)'}")
    print(f"[rebuild_prebuilt_ctx] prebuilt input: {prebuilt_input}")
    print(f"[rebuild_prebuilt_ctx] raw_m5: {raw_m5_path}")
    print(f"[rebuild_prebuilt_ctx] tape_root: {tape_root}")
    print(f"[rebuild_prebuilt_ctx] ctx_cont_dim={ctx_cont_dim} ctx_cat_dim={ctx_cat_dim}")
    print(f"[rebuild_prebuilt_ctx] output (versioned): {out_parquet}")

    if args.dry_run:
        return 0

    if not raw_m5_path.exists():
        print(f"[rebuild_prebuilt_ctx] raw_m5 not found: {raw_m5_path}", file=sys.stderr)
        return 1

    if ctx_cont_dim >= 11 and not tape_root.exists():
        print(f"[rebuild_prebuilt_ctx] tape_root not found: {tape_root}", file=sys.stderr)
        return 1

    if args.expand_to_raw_index:
        # Expansion mode: keep the existing canonical prebuilt contents, but
        # reindex onto the raw tape index so replay can run M1 exit while entry/XGB
        # still use the 5-minute model bars via is_model_bar.
        src = pd.read_parquet(prebuilt_input)
        if isinstance(src.index, pd.DatetimeIndex):
            src = src.sort_index()
            if src.index.tz is None:
                src.index = src.index.tz_localize("UTC")
            else:
                src.index = src.index.tz_convert("UTC")
        elif "ts" in src.columns:
            src["ts"] = pd.to_datetime(src["ts"], utc=True, errors="coerce")
            src = src.dropna(subset=["ts"]).set_index("ts").sort_index()
        elif "time" in src.columns:
            src["time"] = pd.to_datetime(src["time"], utc=True, errors="coerce")
            src = src.dropna(subset=["time"]).set_index("time").sort_index()
        else:
            raise RuntimeError("[rebuild_prebuilt_ctx] source prebuilt must have DatetimeIndex/ts/time")

        if not isinstance(src.index, pd.DatetimeIndex) or src.index.tz is None:
            raise RuntimeError("[rebuild_prebuilt_ctx] source prebuilt index must be tz-aware UTC")

        if raw_m5_path.is_dir():
            # Read the raw tape timestamps only; this is the target M1 index.
            raw_parts = []
            for fp in sorted(raw_m5_path.rglob("*.parquet")):
                try:
                    d = pd.read_parquet(fp, columns=["time"])
                except Exception:
                    continue
                if "time" not in d.columns:
                    continue
                d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
                d = d.dropna(subset=["time"])
                if len(d):
                    raw_parts.append(d["time"])
            if not raw_parts:
                raise RuntimeError(f"[rebuild_prebuilt_ctx] no timestamps found under raw source: {raw_m5_path}")
            raw_index = pd.DatetimeIndex(pd.concat(raw_parts, ignore_index=True)).sort_values().unique()
        else:
            raw_df = pd.read_parquet(raw_m5_path, columns=["time"])
            if "time" not in raw_df.columns:
                raise RuntimeError(f"[rebuild_prebuilt_ctx] raw source missing time column: {raw_m5_path}")
            raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True, errors="coerce")
            raw_df = raw_df.dropna(subset=["time"])
            raw_index = pd.DatetimeIndex(raw_df["time"]).sort_values().unique()

        if len(raw_index) == 0:
            raise RuntimeError("[rebuild_prebuilt_ctx] raw index is empty")
        if raw_index.min() < src.index.min():
            raise RuntimeError(
                f"[rebuild_prebuilt_ctx] raw index starts before source prebuilt; raw_min={raw_index.min()} src_min={src.index.min()}"
            )

        expanded = src.reindex(raw_index, method="ffill")
        if expanded.isna().any().any():
            # Don't silently invent history before the source begins.
            bad_cols = [c for c in expanded.columns if expanded[c].isna().any()]
            raise RuntimeError(f"[rebuild_prebuilt_ctx] expanded prebuilt has NaNs after ffill in columns: {bad_cols[:20]}")

        # Mark the source 5-minute model bars explicitly so runtime can skip XGB/entry on raw bars.
        model_bar_index = src.index
        expanded["is_model_bar"] = expanded.index.isin(model_bar_index)
        expanded.index.name = src.index.name or "ts"

        # Preserve a couple of provenance hints.
        expanded.attrs = dict(src.attrs)
        expanded.attrs["expanded_from_model_index_count"] = int(len(model_bar_index))
        expanded.attrs["expanded_to_raw_index_count"] = int(len(expanded))

        # Write output parquet + sidecars directly.
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        expanded.to_parquet(out_parquet, index=True, compression="snappy")

        import hashlib as _hashlib
        h = _hashlib.sha256()
        with open(out_parquet, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        sha = h.hexdigest()

        manifest = {
            "parquet_path": str(out_parquet),
            "sha256": sha,
            "rows": int(len(expanded)),
            "cols": int(len(expanded.columns)),
            "ctx_cont_dim": int(ctx_cont_dim),
            "ctx_cat_dim": int(ctx_cat_dim),
            "expanded_to_raw_index": True,
            "expanded_from_source": str(prebuilt_input),
            "raw_source": str(raw_m5_path),
            "model_bar_count": int(len(model_bar_index)),
            "raw_index_count": int(len(expanded)),
            "schema_version": "expanded_raw_index_v1",
        }
        (out_parquet.parent / f"{out_parquet.stem}.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        schema_manifest = {
            "required_all_features": [c for c in expanded.columns if c != "is_model_bar"],
            "ctx_cont_dim": int(ctx_cont_dim),
            "ctx_cat_dim": int(ctx_cat_dim),
            "schema_version": "expanded_raw_index_v1",
        }
        (out_parquet.with_suffix(".schema_manifest.json")).write_text(
            json.dumps(schema_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        print(f"[rebuild_prebuilt_ctx] Expanded to raw index: {out_parquet}")
    else:
        if ctx_cont_dim >= 11 and not tape_root.exists():
            print(f"[rebuild_prebuilt_ctx] tape_root not found: {tape_root}", file=sys.stderr)
            return 1

        from gx1.scripts.add_ctx_cont_columns_to_prebuilt import run_add_ctx_cont_columns

        diagnostics_path = out_parquet.with_name(out_parquet.stem + ".ctx_diagnostics.json")

        try:
            run_add_ctx_cont_columns(
                prebuilt_path=prebuilt_input,
                raw_m5_paths=[raw_m5_path],
                output_parquet=out_parquet,
                diagnostics_path=diagnostics_path,
                # These kwargs must be supported by the builder; if not, update builder signature accordingly.
                ctx_cont_dim=ctx_cont_dim,
                ctx_cat_dim=ctx_cat_dim,
                tape_root=tape_root,
            )
        except TypeError:
            # Backward-compatible fallback: builder may not yet accept ctx dims explicitly.
            # In that case, it should infer dims internally or default to cont=4/cat=5.
            run_add_ctx_cont_columns(
                prebuilt_path=prebuilt_input,
                raw_m5_paths=[raw_m5_path],
                output_parquet=out_parquet,
                diagnostics_path=diagnostics_path,
            )
        except Exception as e:
            print(f"[rebuild_prebuilt_ctx] {e}", file=sys.stderr)
            return 1

    # Sidecars are written by builder. If missing, warn.
    for ext in (".manifest.json", ".schema_manifest.json"):
        dst = out_parquet.parent / f"{out_parquet.stem}{ext}"
        if not dst.exists():
            print(f"[rebuild_prebuilt_ctx] warning: output {ext} missing ({dst}); E2E gate may fail", file=sys.stderr)

    # Builder should log: "[PREBUILT_CTX_CONTRACT] required cont+cat present; missing: []"
    print(f"[rebuild_prebuilt_ctx] Done. To use as canonical, set canonical_prebuilt_parquet to: {out_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
