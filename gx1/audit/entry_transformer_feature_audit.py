"""Full Entry Transformer input audit.

Audits the V10 dataset surfaces that feed the Entry Transformer:
seq/snap, ctx_cont, ctx_cat, and the V2 multi-TF cache. The output is intended
for retrain readiness: it lists every contracted feature and flags static,
near-static, out-of-range, and metadata contract issues.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.signal_bridge_v3 import (
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_SEQ_FIELDS_V3,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V2, load_multi_tf_v2_cache


DEFAULT_REBUILD = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626"
)
DEFAULT_DATASET_DIR = DEFAULT_REBUILD / "v10_dataset_6yr"
DEFAULT_MTF_CACHE = DEFAULT_REBUILD / "MULTI_TF_V2_CACHE"
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/entry_transformer_feature_audit_20260626")

TARGET_COLUMNS = [
    "y_direction",
    "y_early_move",
    "y_quality_score",
    "y_bad_path",
    "y_tradable",
    "y_tf_agreement_score",
    "y_position_size_target",
    "y_hold_horizon_target",
    "mae_first_n_bps",
    "mfe_first_n_bps",
    "path_quality_bps",
    "y_dead_negative_long",
    "y_teaser_negative_long",
    "y_hard_negative_long",
    "y_clean_edge_long",
    "y_survival_long",
    "y_selector_long_mask",
    "y_dead_negative_short",
    "y_teaser_negative_short",
    "y_hard_negative_short",
    "y_clean_edge_short",
    "y_survival_short",
    "y_selector_short_mask",
    "y_clean_edge_bidir",
    "y_survival_bidir",
    "label_horizon_bars",
    "path_quality_horizon_bars",
    "y_forecast_ret_K1",
    "y_forecast_ret_K5",
    "y_forecast_ret_K12",
    "y_forecast_ret_K24",
    "y_vol_fwd_K12",
    "y_vol_fwd_K48",
    "y_vol_fwd_K96",
]

BOUNDS: dict[str, tuple[float, float]] = {
    "p_long": (0.0, 1.0),
    "p_short": (0.0, 1.0),
    "p_flat": (0.0, 1.0),
    "p_hat": (0.0, 1.0),
    "uncertainty_score": (0.0, 1.0),
    "margin_top1_top2": (0.0, 1.0),
    "body_pct": (0.0, 1.0),
    "upper_wick_pct": (0.0, 1.0),
    "lower_wick_pct": (0.0, 1.0),
    "range_pos_20": (0.0, 1.0),
    "rsi14_centered": (-1.0, 1.0),
    "D1_atr_percentile_252": (0.0, 1.0),
    "d1_close_pct_in_20day_range_canon_v2": (0.0, 1.0),
    "vol_pct_m5_1yr": (0.0, 1.0),
    "vol_pct_h1_1yr": (0.0, 1.0),
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _safe_float(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _family(name: str) -> str:
    n = str(name)
    if n.startswith(("p_", "margin", "uncertainty", "entropy")):
        return "xgb_bridge"
    if n.startswith("smc_") or "swing" in n or "structure" in n:
        return "structure_smc_swing"
    if "dip" in n or "mfe" in n or "mae" in n or "tail" in n:
        return "path_dip_tail"
    if "atr" in n or "vol" in n or "range" in n or "rvol" in n or "std" in n:
        return "volatility_range"
    if "ema" in n or "trend" in n or "mom" in n or "slope" in n or "ret_" in n or "roc" in n:
        return "momentum_trend"
    if "session" in n or "hour" in n or "dow" in n or n.startswith("is_"):
        return "session_time"
    if "wick" in n or "body" in n or "clv" in n:
        return "candle_shape"
    if n.startswith(("d1_", "h1_", "h4_", "m15_", "m5_", "_v1h")):
        return "multi_tf_context"
    return "other"


def _stack_list_column(values: Iterable[Any], dtype: np.dtype) -> np.ndarray:
    items = list(values)
    if not items:
        return np.asarray([], dtype=dtype)
    try:
        return np.stack(items).astype(dtype, copy=False)
    except ValueError:
        return np.stack([np.stack(x) for x in items]).astype(dtype, copy=False)


class NumericAccumulator:
    def __init__(self, names: Sequence[str], *, sample_limit: int = 100_000) -> None:
        self.names = list(names)
        self.dim = len(self.names)
        self.n = 0
        self.finite = np.zeros(self.dim, dtype=np.int64)
        self.zero = np.zeros(self.dim, dtype=np.int64)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)
        self.class_counts: dict[int, int] = defaultdict(int)
        self.class_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.dim, dtype=np.float64))
        self.sample_limit = int(sample_limit)
        self.samples: list[np.ndarray] = []
        self.sample_n = 0

    def add(self, values: np.ndarray, labels: np.ndarray | None = None) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise RuntimeError(f"numeric shape mismatch: got={arr.shape} expected=(*,{self.dim})")
        finite = np.isfinite(arr)
        clean = np.where(finite, arr, 0.0)
        self.n += int(arr.shape[0])
        self.finite += finite.sum(axis=0).astype(np.int64)
        self.zero += ((clean == 0.0) & finite).sum(axis=0).astype(np.int64)
        self.sum += clean.sum(axis=0)
        self.sumsq += (clean * clean).sum(axis=0)
        self.min = np.minimum(self.min, np.nanmin(np.where(finite, arr, np.nan), axis=0))
        self.max = np.maximum(self.max, np.nanmax(np.where(finite, arr, np.nan), axis=0))
        if labels is not None:
            y = np.asarray(labels)
            if len(y) != arr.shape[0]:
                raise RuntimeError(f"label length mismatch: {len(y)} vs {arr.shape[0]}")
            for cls in np.unique(y[~pd.isna(y)]):
                cls_i = int(cls)
                mask = y == cls
                self.class_counts[cls_i] += int(mask.sum())
                self.class_sum[cls_i] += clean[mask].sum(axis=0)
        if self.sample_n < self.sample_limit:
            take = min(self.sample_limit - self.sample_n, arr.shape[0])
            if take > 0:
                self.samples.append(clean[:take].astype(np.float32, copy=True))
                self.sample_n += int(take)

    def rows(self, *, split: str, scope: str, group: str | None = None) -> list[dict[str, Any]]:
        mean = self.sum / max(self.n, 1)
        var = np.maximum(self.sumsq / max(self.n, 1) - mean * mean, 0.0)
        std = np.sqrt(var)
        sample = np.vstack(self.samples).astype(np.float64, copy=False) if self.samples else np.empty((0, self.dim))
        if len(sample):
            q05 = np.nanquantile(sample, 0.05, axis=0)
            q50 = np.nanquantile(sample, 0.50, axis=0)
            q95 = np.nanquantile(sample, 0.95, axis=0)
        else:
            q05 = q50 = q95 = np.full(self.dim, np.nan)

        eta = np.full(self.dim, np.nan, dtype=np.float64)
        total_ss = self.sumsq - (self.sum * self.sum / max(self.n, 1))
        if self.class_counts:
            between = np.zeros(self.dim, dtype=np.float64)
            for cls, cnt in self.class_counts.items():
                if cnt <= 0:
                    continue
                cls_mean = self.class_sum[cls] / float(cnt)
                between += float(cnt) * (cls_mean - mean) ** 2
            mask = total_ss > 1e-12
            eta[mask] = between[mask] / total_ss[mask]

        out: list[dict[str, Any]] = []
        for i, name in enumerate(self.names):
            out.append(
                {
                    "scope": scope,
                    "split": split,
                    "feature_index": i,
                    "feature": name,
                    "group": group or _family(name),
                    "n": int(self.n),
                    "finite_rate": float(self.finite[i] / max(self.n, 1)),
                    "mean": _safe_float(mean[i]),
                    "std": _safe_float(std[i]),
                    "min": _safe_float(self.min[i]),
                    "p05_sample": _safe_float(q05[i]),
                    "p50_sample": _safe_float(q50[i]),
                    "p95_sample": _safe_float(q95[i]),
                    "max": _safe_float(self.max[i]),
                    "zero_rate": float(self.zero[i] / max(self.n, 1)),
                    "abs_max": _safe_float(max(abs(self.min[i]), abs(self.max[i]))),
                    "eta2_y_direction": _safe_float(eta[i]),
                    "constant_flag": bool(std[i] <= 1e-12),
                    "near_zero_flag": bool((self.zero[i] / max(self.n, 1)) >= 0.995),
                }
            )
        return out


class CatAccumulator:
    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self.counts = [Counter() for _ in self.names]
        self.n = 0

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values)
        if arr.ndim != 2 or arr.shape[1] != len(self.names):
            raise RuntimeError(f"cat shape mismatch: got={arr.shape} expected=(*,{len(self.names)})")
        self.n += int(arr.shape[0])
        for i in range(arr.shape[1]):
            self.counts[i].update(int(x) for x in arr[:, i])

    def rows(self, *, split: str, scope: str = "ctx_cat") -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, name in enumerate(self.names):
            counts = self.counts[i]
            unique = len(counts)
            top = counts.most_common(1)[0][1] if counts else 0
            out.append(
                {
                    "scope": scope,
                    "split": split,
                    "feature_index": i,
                    "feature": name,
                    "group": "categorical_context",
                    "n": int(self.n),
                    "unique_count": int(unique),
                    "top_rate": float(top / max(self.n, 1)),
                    "counts_json": json.dumps({str(k): int(v) for k, v in sorted(counts.items())}),
                    "constant_flag": bool(unique <= 1),
                    "near_constant_flag": bool(top / max(self.n, 1) >= 0.995),
                }
            )
        return out


def _split_files(dataset_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        matches = sorted(dataset_dir.glob(f"*{split}*.parquet"))
        if matches:
            out[split] = matches[0]
    if not out:
        raise RuntimeError(f"no split parquets found under {dataset_dir}")
    return out


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    vals, counts = np.unique(labels.astype(np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def _target_stats(path: Path, split: str) -> list[dict[str, Any]]:
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in TARGET_COLUMNS if c in schema_names]
    if not cols:
        return []
    df = pd.read_parquet(path, columns=cols)
    rows: list[dict[str, Any]] = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(
            {
                "split": split,
                "target": c,
                "n": int(s.notna().sum()),
                "mean": _safe_float(float(s.mean())),
                "std": _safe_float(float(s.std(ddof=0))),
                "min": _safe_float(float(s.min())),
                "p50": _safe_float(float(s.quantile(0.50))),
                "max": _safe_float(float(s.max())),
                "zero_rate": _safe_float(float((s == 0.0).mean())),
                "unique_count": int(s.nunique(dropna=True)),
            }
        )
    return rows


def audit_split(path: Path, split: str, *, batch_size: int, seq_hist_sample_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pf = pq.ParquetFile(path)
    snap_acc = NumericAccumulator(ORDERED_SEQ_FIELDS_V3)
    seq_last_acc = NumericAccumulator(ORDERED_SEQ_FIELDS_V3)
    seq_hist_acc = NumericAccumulator(ORDERED_SEQ_FIELDS_V3)
    ctx_acc = NumericAccumulator(ORDERED_CTX_CONT_NAMES_V3)
    cat_acc = CatAccumulator(ORDERED_CTX_CAT_NAMES_V3)
    seq_hist_rows_left = int(seq_hist_sample_rows)
    mismatch_rows = 0
    rows = 0
    label_counter: Counter[int] = Counter()
    cols = ["seq", "snap", "ctx_cont", "ctx_cat", "y_direction"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        pdf = batch.to_pandas()
        y = pd.to_numeric(pdf["y_direction"], errors="coerce").to_numpy()
        seq = _stack_list_column(pdf["seq"], np.float32)
        snap = _stack_list_column(pdf["snap"], np.float32)
        ctx = _stack_list_column(pdf["ctx_cont"], np.float32)
        cat = _stack_list_column(pdf["ctx_cat"], np.int64)
        if seq.ndim != 3 or seq.shape[1:] != (96, len(ORDERED_SEQ_FIELDS_V3)):
            raise RuntimeError(f"{split}: seq shape {seq.shape} does not match (B,96,{len(ORDERED_SEQ_FIELDS_V3)})")
        if snap.shape[1] != len(ORDERED_SEQ_FIELDS_V3):
            raise RuntimeError(f"{split}: snap shape {snap.shape}")
        if ctx.shape[1] != len(ORDERED_CTX_CONT_NAMES_V3):
            raise RuntimeError(f"{split}: ctx_cont shape {ctx.shape}")
        if cat.shape[1] != len(ORDERED_CTX_CAT_NAMES_V3):
            raise RuntimeError(f"{split}: ctx_cat shape {cat.shape}")
        rows += int(len(pdf))
        label_counter.update(int(v) for v in y if not pd.isna(v))
        snap_acc.add(snap, y)
        seq_last = seq[:, -1, :]
        seq_last_acc.add(seq_last, y)
        ctx_acc.add(ctx, y)
        cat_acc.add(cat)
        mismatch_rows += int(np.any(np.abs(seq_last - snap) > 1e-6, axis=1).sum())
        if seq_hist_rows_left > 0:
            take = min(seq_hist_rows_left, seq.shape[0])
            seq_hist_acc.add(seq[:take].reshape(-1, seq.shape[-1]), None)
            seq_hist_rows_left -= take
    numeric = []
    numeric.extend(snap_acc.rows(split=split, scope="snap"))
    numeric.extend(seq_last_acc.rows(split=split, scope="seq_last"))
    numeric.extend(seq_hist_acc.rows(split=split, scope="seq_hist_sample"))
    numeric.extend(ctx_acc.rows(split=split, scope="ctx_cont"))
    return (
        numeric,
        cat_acc.rows(split=split),
        _target_stats(path, split),
        {
            "split": split,
            "path": str(path),
            "rows": int(rows),
            "label_counts": {str(k): int(v) for k, v in sorted(label_counter.items())},
            "snap_seq_last_mismatch_rows": int(mismatch_rows),
            "seq_hist_sample_rows": int(seq_hist_sample_rows - seq_hist_rows_left),
        },
    )


def audit_mtf_cache(cache_dir: Path, *, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feats = load_multi_tf_v2_cache(cache_dir)
    names = list(MULTI_TF_PER_BAR_FEATURES_V2)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"path": str(cache_dir), "tfs": {}}
    for tf, df in feats.items():
        arr = np.asarray(df.attrs.get("feats_np"), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != len(names):
            raise RuntimeError(f"MTF {tf}: shape {arr.shape} expected (*,{len(names)})")
        acc = NumericAccumulator(names)
        for start in range(0, arr.shape[0], batch_size):
            acc.add(arr[start : start + batch_size], None)
        rows.extend(acc.rows(split="mtf_cache", scope=f"mtf_{str(tf).lower()}", group="multi_tf_v2"))
        summary["tfs"][str(tf)] = {
            "rows": int(arr.shape[0]),
            "cols": int(arr.shape[1]),
            "start": str(df.index[0]) if len(df.index) else None,
            "end": str(df.index[-1]) if len(df.index) else None,
            "column_names_from_contract": True,
        }
    return rows, summary


def _inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface, names in (
        ("seq_snap", ORDERED_SEQ_FIELDS_V3),
        ("ctx_cont", ORDERED_CTX_CONT_NAMES_V3),
        ("ctx_cat", ORDERED_CTX_CAT_NAMES_V3),
    ):
        for i, name in enumerate(names):
            rows.append({"surface": surface, "feature_index": i, "feature": name, "group": _family(name), "active_in_contract": True})
    for tf in ("M5", "M15", "H1", "H4", "D1"):
        for i, name in enumerate(MULTI_TF_PER_BAR_FEATURES_V2):
            rows.append({"surface": f"multi_tf_{tf}", "feature_index": i, "feature": name, "group": "multi_tf_v2", "active_in_contract": True})
    return rows


def _bundle_checks(bundle_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bd in bundle_dirs:
        for filename in ("bundle_metadata.json", "MASTER_TRANSFORMER_LOCK.json"):
            p = bd / filename
            if not p.exists():
                rows.append({"bundle_dir": str(bd), "file": filename, "ok": False, "issue": "missing"})
                continue
            d = json.loads(p.read_text())
            cont_names = d.get("ordered_ctx_cont_names") or []
            cat_names = d.get("ordered_ctx_cat_names") or []
            cont_dim = int(d.get("ctx_cont_dim") or d.get("expected_ctx_cont_dim") or 0)
            cat_dim = int(d.get("ctx_cat_dim") or d.get("expected_ctx_cat_dim") or 0)
            issue = []
            if cont_dim != len(cont_names):
                issue.append(f"ctx_cont_names_len={len(cont_names)} != ctx_cont_dim={cont_dim}")
            if cat_dim != len(cat_names):
                issue.append(f"ctx_cat_names_len={len(cat_names)} != ctx_cat_dim={cat_dim}")
            if cont_dim == len(ORDERED_CTX_CONT_NAMES_V3) and list(cont_names) != list(ORDERED_CTX_CONT_NAMES_V3):
                issue.append("ctx_cont_names_not_v3_contract")
            if cat_dim == len(ORDERED_CTX_CAT_NAMES_V3) and list(cat_names) != list(ORDERED_CTX_CAT_NAMES_V3):
                issue.append("ctx_cat_names_not_v3_contract")
            rows.append(
                {
                    "bundle_dir": str(bd),
                    "file": filename,
                    "ok": not issue,
                    "ctx_cont_dim": cont_dim,
                    "ctx_cont_names_len": len(cont_names),
                    "ctx_cat_dim": cat_dim,
                    "ctx_cat_names_len": len(cat_names),
                    "issue": "; ".join(issue),
                }
            )
    return rows


def _risk_flags(numeric: pd.DataFrame, categorical: pd.DataFrame, bundle_checks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in numeric.iterrows():
        feature = str(r["feature"])
        flags: list[str] = []
        if bool(r.get("constant_flag", False)):
            flags.append("CONSTANT")
        if bool(r.get("near_zero_flag", False)):
            flags.append("ZERO_GT_99_5PCT")
        abs_max = float(r["abs_max"]) if pd.notna(r.get("abs_max")) else 0.0
        if abs_max > 1e7:
            flags.append("ABS_MAX_GT_1E7")
        if feature in BOUNDS and pd.notna(r.get("min")) and pd.notna(r.get("max")):
            lo, hi = BOUNDS[feature]
            if float(r["min"]) < lo - 1e-6 or float(r["max"]) > hi + 1e-6:
                flags.append(f"BOUNDS_FAIL_{lo}_{hi}")
        eta = r.get("eta2_y_direction")
        if str(r.get("split")) == "train" and str(r.get("scope")) in {"snap", "ctx_cont"} and pd.notna(eta):
            if float(eta) < 1e-7 and not flags:
                flags.append("VERY_LOW_UNIVARIATE_DIR_ASSOC")
        for flag in flags:
            rows.append({**r.to_dict(), "risk_flag": flag})
    for _, r in categorical.iterrows():
        flags: list[str] = []
        if bool(r.get("constant_flag", False)):
            flags.append("CONSTANT_CAT")
        elif bool(r.get("near_constant_flag", False)):
            flags.append("NEAR_CONSTANT_CAT")
        for flag in flags:
            rows.append({**r.to_dict(), "risk_flag": flag})
    for _, r in bundle_checks.iterrows():
        if not bool(r.get("ok")):
            rows.append({**r.to_dict(), "risk_flag": "BUNDLE_METADATA_CONTRACT_FAIL"})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser()
    mtf_cache = Path(args.mtf_cache).expanduser() if args.mtf_cache else None
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_rows: list[dict[str, Any]] = []
    cat_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []
    for split, path in _split_files(dataset_dir).items():
        n, c, t, s = audit_split(path, split, batch_size=int(args.batch_size), seq_hist_sample_rows=int(args.seq_hist_sample_rows))
        numeric_rows.extend(n)
        cat_rows.extend(c)
        target_rows.extend(t)
        split_summaries.append(s)

    mtf_summary: dict[str, Any] | None = None
    if mtf_cache and mtf_cache.exists():
        mtf_rows, mtf_summary = audit_mtf_cache(mtf_cache, batch_size=int(args.batch_size))
        numeric_rows.extend(mtf_rows)

    inventory = pd.DataFrame(_inventory())
    numeric = pd.DataFrame(numeric_rows)
    categorical = pd.DataFrame(cat_rows)
    targets = pd.DataFrame(target_rows)
    bundle_dirs = [Path(p).expanduser() for p in args.bundle_dir]
    bundle_checks = pd.DataFrame(_bundle_checks(bundle_dirs)) if bundle_dirs else pd.DataFrame()
    risks = _risk_flags(numeric, categorical, bundle_checks)

    inventory.to_csv(out_dir / "feature_inventory.csv", index=False)
    numeric.to_csv(out_dir / "feature_stats_by_split.csv", index=False)
    categorical.to_csv(out_dir / "categorical_stats_by_split.csv", index=False)
    targets.to_csv(out_dir / "target_label_stats_by_split.csv", index=False)
    if not bundle_checks.empty:
        bundle_checks.to_csv(out_dir / "bundle_metadata_contract_checks.csv", index=False)
    risks.to_csv(out_dir / "feature_risk_flags.csv", index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "contract_counts": {
            "seq_snap_unique": len(ORDERED_SEQ_FIELDS_V3),
            "ctx_cont": len(ORDERED_CTX_CONT_NAMES_V3),
            "ctx_cat": len(ORDERED_CTX_CAT_NAMES_V3),
            "mtf_features_per_tf": len(MULTI_TF_PER_BAR_FEATURES_V2),
            "mtf_tfs": ["M5", "M15", "H1", "H4", "D1"],
            "total_unique_named_inputs_excluding_time_steps": (
                len(ORDERED_SEQ_FIELDS_V3)
                + len(ORDERED_CTX_CONT_NAMES_V3)
                + len(ORDERED_CTX_CAT_NAMES_V3)
                + len(MULTI_TF_PER_BAR_FEATURES_V2) * 5
            ),
        },
        "splits": split_summaries,
        "mtf_cache_stats": mtf_summary,
        "bundle_checks": bundle_checks.to_dict(orient="records") if not bundle_checks.empty else [],
        "risk_rows": int(len(risks)),
        "risk_counts": risks["risk_flag"].value_counts().to_dict() if not risks.empty else {},
        "outputs": {
            "feature_inventory": str(out_dir / "feature_inventory.csv"),
            "feature_stats_by_split": str(out_dir / "feature_stats_by_split.csv"),
            "categorical_stats_by_split": str(out_dir / "categorical_stats_by_split.csv"),
            "target_label_stats_by_split": str(out_dir / "target_label_stats_by_split.csv"),
            "bundle_metadata_contract_checks": str(out_dir / "bundle_metadata_contract_checks.csv"),
            "feature_risk_flags": str(out_dir / "feature_risk_flags.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--mtf-cache", default=str(DEFAULT_MTF_CACHE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--bundle-dir", action="append", default=[])
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seq-hist-sample-rows", type=int, default=20_000)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
