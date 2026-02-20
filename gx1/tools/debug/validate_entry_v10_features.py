#!/usr/bin/env python3
"""
ENTRY_V10 Feature Validation Script (ONE UNIVERSE)

Validerer at V10-datasettet følger ONE UNIVERSE:
- XGB: 28 M5-features -> 7 signaler (signal_bridge_v1)
- ENTRY/EXIT Transformers: seq_dim=7, snap_dim=7
- Context: ctx_cont_dim=6, ctx_cat_dim=6

Dette scriptet validerer:
1) V10 parquet har riktige kolonner (seq/snap/ctx_cont/ctx_cat/y_direction)
2) Tensor-shapes er korrekte (seq [T,7], snap [7], ctx_cont [6], ctx_cat [6])
3) (Best-effort) at ingen legacy-dims (16/88 eller session_id/vol_regime_id/trend_regime_id som egne kolonner) finnes i V10
4) (Optional) at V9 feature meta fortsatt eksisterer (kun for sanity; ikke shape-kontrakt)

NB:
- Ingen fallback. Hard fail på mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# ONE UNIVERSE dims
SEQ_DIM = 7
SNAP_DIM = 7
CTX_CONT_DIM = 6
CTX_CAT_DIM = 6


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def find_dataset_files() -> Tuple[Path, Path | None]:
    """
    Find V10 dataset path and (optional) legacy V9 feature meta path (for sanity only).

    NOTE: we do NOT require V9 datasets anymore in ONE UNIVERSE.
    """
    v10_path = project_root / "data/entry_v10/entry_v10_dataset.parquet"
    if not v10_path.exists():
        raise FileNotFoundError(f"V10 dataset not found: {v10_path}")

    # Optional legacy meta (sanity only; do not fail if missing)
    meta_path = project_root / "gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json"
    if not meta_path.exists():
        meta_path = None

    return v10_path, meta_path


def load_feature_meta_best_effort(meta_path: Path | None) -> Dict[str, List[str]]:
    """Load feature metadata from JSON (best effort)."""
    if meta_path is None:
        return {"seq_features": [], "snap_features": []}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_features = meta.get("seq_features", []) or []
    snap_features = meta.get("snap_features", []) or []
    if not isinstance(seq_features, list):
        seq_features = []
    if not isinstance(snap_features, list):
        snap_features = []
    return {"seq_features": [str(x) for x in seq_features], "snap_features": [str(x) for x in snap_features]}


def _to_array(x: Any) -> np.ndarray:
    """
    Convert nested python/list/object arrays from parquet into np.ndarray,
    without guessing legacy dims.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)
    # pandas may store as scalar/None
    return np.array(x)


def _normalize_seq(seq_val: Any) -> np.ndarray:
    """
    Expect seq as:
      - shape (T, 7) OR
      - object array length T where each element length 7
    Does NOT support legacy 16-dim.
    """
    arr = _to_array(seq_val)

    # Case: already 2D numeric
    if arr.ndim == 2:
        return arr

    # Case: object array, e.g. len=T each item length=7
    if arr.ndim == 1 and arr.dtype == object and len(arr) > 0:
        try:
            arr2 = np.array([np.array(v, dtype=np.float32) for v in arr], dtype=np.float32)
            return arr2
        except Exception:
            return arr

    return arr


def _normalize_vec(vec_val: Any) -> np.ndarray:
    """
    Expect snap / ctx_cont / ctx_cat as 1D vectors.
    """
    arr = _to_array(vec_val)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr


def validate_v10_dataset(v10_path: Path) -> Dict[str, Any]:
    """Validate V10 dataset structure and ONE UNIVERSE tensor shapes."""
    print(f"[V10] Loading dataset: {v10_path}")
    df = pd.read_parquet(v10_path)

    # Required columns in ONE UNIVERSE
    required_cols = {"seq", "snap", "ctx_cont", "ctx_cat", "y_direction"}
    missing = required_cols - set(df.columns)
    _require(not missing, f"[V10_SCHEMA_FAIL] Missing required columns: {sorted(missing)}")

    # Forbidden legacy columns (hard fail if present)
    forbidden_legacy_cols = {"session_id", "vol_regime_id", "trend_regime_id"}
    present_legacy_cols = sorted(forbidden_legacy_cols & set(df.columns))
    _require(
        not present_legacy_cols,
        f"[V10_SCHEMA_FAIL] Legacy cols present (forbidden in ONE UNIVERSE): {present_legacy_cols}",
    )

    n_rows = int(len(df))
    _require(n_rows > 0, "[V10_SCHEMA_FAIL] Dataset has 0 rows")

    n_samples = min(25, n_rows)
    seq_shapes: List[Any] = []
    snap_shapes: List[Any] = []
    ctx_cont_shapes: List[Any] = []
    ctx_cat_shapes: List[Any] = []

    # collect some examples of bad shapes (for easier debug)
    bad_examples: List[Dict[str, Any]] = []

    for i in range(n_samples):
        row = df.iloc[i]

        seq = _normalize_seq(row["seq"])
        snap = _normalize_vec(row["snap"])
        ctx_cont = _normalize_vec(row["ctx_cont"])
        ctx_cat = _normalize_vec(row["ctx_cat"])

        seq_shapes.append(getattr(seq, "shape", None))
        snap_shapes.append(getattr(snap, "shape", None))
        ctx_cont_shapes.append(getattr(ctx_cont, "shape", None))
        ctx_cat_shapes.append(getattr(ctx_cat, "shape", None))

        ok = True

        # seq must be 2D with last dim = 7, and T>=1
        if not (isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[1] == SEQ_DIM and seq.shape[0] >= 1):
            ok = False

        # snap must be 1D length 7
        if not (isinstance(snap, np.ndarray) and snap.ndim == 1 and snap.shape[0] == SNAP_DIM):
            ok = False

        # ctx_cont must be 1D length 6
        if not (isinstance(ctx_cont, np.ndarray) and ctx_cont.ndim == 1 and ctx_cont.shape[0] == CTX_CONT_DIM):
            ok = False

        # ctx_cat must be 1D length 6
        if not (isinstance(ctx_cat, np.ndarray) and ctx_cat.ndim == 1 and ctx_cat.shape[0] == CTX_CAT_DIM):
            ok = False

        if not ok and len(bad_examples) < 5:
            bad_examples.append(
                {
                    "row_idx": int(i),
                    "seq_shape": getattr(seq, "shape", None),
                    "snap_shape": getattr(snap, "shape", None),
                    "ctx_cont_shape": getattr(ctx_cont, "shape", None),
                    "ctx_cat_shape": getattr(ctx_cat, "shape", None),
                }
            )

    # Check overall shape ok
    seq_ok = all((s is not None and len(s) == 2 and s[1] == SEQ_DIM and s[0] >= 1) for s in seq_shapes if s is not None)
    snap_ok = all((s is not None and len(s) == 1 and s[0] == SNAP_DIM) for s in snap_shapes if s is not None)
    ctx_cont_ok = all((s is not None and len(s) == 1 and s[0] == CTX_CONT_DIM) for s in ctx_cont_shapes if s is not None)
    ctx_cat_ok = all((s is not None and len(s) == 1 and s[0] == CTX_CAT_DIM) for s in ctx_cat_shapes if s is not None)

    # Extract first row shapes for quick display
    row0 = df.iloc[0]
    seq0 = _normalize_seq(row0["seq"])
    snap0 = _normalize_vec(row0["snap"])
    ctx_cont0 = _normalize_vec(row0["ctx_cont"])
    ctx_cat0 = _normalize_vec(row0["ctx_cat"])

    return {
        "n_rows": n_rows,
        "required_cols": sorted(required_cols),
        "seq_shape_ok": bool(seq_ok),
        "snap_shape_ok": bool(snap_ok),
        "ctx_cont_shape_ok": bool(ctx_cont_ok),
        "ctx_cat_shape_ok": bool(ctx_cat_ok),
        "seq_shapes_sample": seq_shapes[:8],
        "snap_shapes_sample": snap_shapes[:8],
        "ctx_cont_shapes_sample": ctx_cont_shapes[:8],
        "ctx_cat_shapes_sample": ctx_cat_shapes[:8],
        "expected_seq_dim": SEQ_DIM,
        "expected_snap_dim": SNAP_DIM,
        "expected_ctx_cont_dim": CTX_CONT_DIM,
        "expected_ctx_cat_dim": CTX_CAT_DIM,
        "actual_seq_shape_first": getattr(seq0, "shape", None),
        "actual_snap_shape_first": getattr(snap0, "shape", None),
        "actual_ctx_cont_shape_first": getattr(ctx_cont0, "shape", None),
        "actual_ctx_cat_shape_first": getattr(ctx_cat0, "shape", None),
        "bad_examples": bad_examples,
    }


def generate_report(
    v10_results: Dict[str, Any],
    meta_path: Path | None,
    feature_meta: Dict[str, List[str]],
    output_path: Path,
) -> None:
    """Generate markdown validation report."""
    overall_pass = (
        v10_results["seq_shape_ok"]
        and v10_results["snap_shape_ok"]
        and v10_results["ctx_cont_shape_ok"]
        and v10_results["ctx_cat_shape_ok"]
    )

    lines: List[str] = [
        "# ENTRY_V10 Feature Validation Report (ONE UNIVERSE)",
        "",
        "**Date:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "## Summary",
        "",
        f"**Overall Result:** {'✅ PASS' if overall_pass else '❌ FAIL'}",
        "",
        "### ONE UNIVERSE Contract",
        "",
        f"- seq_dim = {SEQ_DIM}",
        f"- snap_dim = {SNAP_DIM}",
        f"- ctx_cont_dim = {CTX_CONT_DIM}",
        f"- ctx_cat_dim = {CTX_CAT_DIM}",
        "",
        "## V10 Dataset Validation",
        "",
        f"- **Dataset:** `{(project_root / 'data/entry_v10/entry_v10_dataset.parquet')}`",
        f"- **Rows:** {v10_results['n_rows']:,}",
        "",
        "### Shape Checks",
        "",
        "| Component | Expected | First Row | Status |",
        "|---|---:|---:|---|",
        f"| seq | (T,{SEQ_DIM}) | {v10_results['actual_seq_shape_first']} | {'✅' if v10_results['seq_shape_ok'] else '❌'} |",
        f"| snap | ({SNAP_DIM},) | {v10_results['actual_snap_shape_first']} | {'✅' if v10_results['snap_shape_ok'] else '❌'} |",
        f"| ctx_cont | ({CTX_CONT_DIM},) | {v10_results['actual_ctx_cont_shape_first']} | {'✅' if v10_results['ctx_cont_shape_ok'] else '❌'} |",
        f"| ctx_cat | ({CTX_CAT_DIM},) | {v10_results['actual_ctx_cat_shape_first']} | {'✅' if v10_results['ctx_cat_shape_ok'] else '❌'} |",
        "",
        "### Samples (first few shapes)",
        "",
        f"- seq shapes: {v10_results['seq_shapes_sample']}",
        f"- snap shapes: {v10_results['snap_shapes_sample']}",
        f"- ctx_cont shapes: {v10_results['ctx_cont_shapes_sample']}",
        f"- ctx_cat shapes: {v10_results['ctx_cat_shapes_sample']}",
        "",
    ]

    if v10_results["bad_examples"]:
        lines.extend(
            [
                "### Bad examples (first 5)",
                "",
                "```json",
                json.dumps(v10_results["bad_examples"], indent=2),
                "```",
                "",
            ]
        )

    # Optional legacy meta info (sanity only)
    lines.extend(
        [
            "## Optional Legacy Meta (sanity only)",
            "",
            f"- **meta_path:** `{meta_path}`" if meta_path else "- **meta_path:** (missing; ignored in ONE UNIVERSE)",
            f"- **seq_features (legacy count):** {len(feature_meta.get('seq_features', []))}",
            f"- **snap_features (legacy count):** {len(feature_meta.get('snap_features', []))}",
            "",
            "## Conclusion",
            "",
            ("✅ **ONE UNIVERSE OK.** V10 dataset matches (7/7 + 6/6)."
             if overall_pass
             else "❌ **ONE UNIVERSE FAIL.** Fix dataset schema/shapes (see above)."),
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[REPORT] Validation report saved to: {output_path}")


def main() -> int:
    print("=" * 80)
    print("ENTRY_V10 DATA VALIDATION (ONE UNIVERSE)")
    print("=" * 80)
    print()

    try:
        v10_path, meta_path = find_dataset_files()
        print(f"[FILES] V10 Dataset: {v10_path}")
        if meta_path:
            print(f"[FILES] (Optional) Legacy Meta: {meta_path}")
        else:
            print("[FILES] (Optional) Legacy Meta: MISSING (ignored)")
        print()

        feature_meta = load_feature_meta_best_effort(meta_path)

        v10_results = validate_v10_dataset(v10_path)

        print("=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print()

        print(f"Rows: {v10_results['n_rows']:,}")
        print(f"Seq OK (T,{SEQ_DIM}): {v10_results['seq_shape_ok']}  first={v10_results['actual_seq_shape_first']}")
        print(f"Snap OK ({SNAP_DIM},): {v10_results['snap_shape_ok']}  first={v10_results['actual_snap_shape_first']}")
        print(f"CtxCont OK ({CTX_CONT_DIM},): {v10_results['ctx_cont_shape_ok']}  first={v10_results['actual_ctx_cont_shape_first']}")
        print(f"CtxCat OK ({CTX_CAT_DIM},): {v10_results['ctx_cat_shape_ok']}  first={v10_results['actual_ctx_cat_shape_first']}")
        if v10_results["bad_examples"]:
            print("\nBad examples:")
            for ex in v10_results["bad_examples"]:
                print(" -", ex)
        print()

        overall_pass = (
            v10_results["seq_shape_ok"]
            and v10_results["snap_shape_ok"]
            and v10_results["ctx_cont_shape_ok"]
            and v10_results["ctx_cat_shape_ok"]
        )

        print("=" * 80)
        print(f"Result: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        print("=" * 80)

        report_path = project_root / "reports/rl/entry_v10/ENTRY_V10_FEATURE_AUDIT.md"
        generate_report(v10_results, meta_path, feature_meta, report_path)

        return 0 if overall_pass else 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())