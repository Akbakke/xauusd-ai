"""Deep feature, tail, and exit diagnostics for Entry chart-structure research.

Offline research only. This script inventories every no-XGB base feature plus
generated chart/deep features, measures feature association with path/tail
labels, joins entry features back onto replay trades, and searches for
diagnostic entry/exit veto candidates that may reduce tail risk. Any veto rule
reported here is in-sample diagnostic and must be retested by replay before use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.evaluate_entry_selective_edge_v1 import _json_default, _split_files
from gx1.scripts.evaluate_entry_tabular_no_xgb_walkforward_v1 import _load_all_data
from gx1.scripts.experiment_entry_chart_structure_ablation_v1 import (
    DEFAULT_DATASET_DIR,
    _build_chart_layer,
    _build_deep_interaction_layer,
)


DEFAULT_REPLAY_DIR = Path("/home/andre2/GX1_DATA/reports/entry_chart_structure_ablation_20260627_focused_v1")
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/entry_feature_tail_exit_deep_audit_20260627_v1")

TARGET_COLUMNS = [
    "time",
    "mae_first_n_bps",
    "mfe_first_n_bps",
    "path_quality_bps",
    "y_bad_path",
    "y_tail_mae_long_K12",
    "y_tail_mae_long_K48",
    "y_tail_mae_long_K96",
    "y_tail_mae_short_K12",
    "y_tail_mae_short_K48",
    "y_tail_mae_short_K96",
    "y_dip_mfe_long_K48",
    "y_dip_mfe_short_K48",
    "y_time_to_mfe_frac_long_K48",
    "y_time_to_mfe_frac_short_K48",
    "y_vol_fwd_K12",
    "y_vol_fwd_K48",
    "y_vol_fwd_K96",
]


def _parse_csv(raw: str) -> list[str]:
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _feature_family(name: str) -> str:
    low = name.lower()
    if low.startswith("chart."):
        if "cross" in low or "ema" in low or "trend_delta" in low:
            return "chart_deep_trend_cross"
        if "tail_pressure" in low or "tail_risk" in low:
            return "chart_deep_tail_pressure"
        if "_x_" in low:
            return "chart_interaction"
        if any(k in low for k in ["hh", "hl", "lh", "ll", "bos", "choch", "sweep", "wick", "pullback"]):
            return "chart_structure"
        if any(k in low for k in ["vol", "atr", "compression", "expansion"]):
            return "chart_volatility"
        return "chart_other"
    if any(k in low for k in ["struct", "smc", "swing", "bos", "choch", "sweep", "wick", "pivot", "liquidity", "pullback"]):
        return "base_structure"
    if any(k in low for k in ["atr", "vol", "range", "squeeze", "bandwidth"]):
        return "base_volatility"
    if any(k in low for k in ["session", "hour", "dow", "is_asia", "is_eu", "is_us", "overlap"]):
        return "base_session_time"
    if any(k in low for k in ["ema", "trend", "slope", "momentum", "ret_", "rsi"]):
        return "base_trend_momentum"
    if any(k in low for k in ["regime", "bucket", "cat"]):
        return "base_regime_cat"
    if any(k in low for k in ["dist", "support", "resistance", "premium", "discount"]):
        return "base_level_location"
    return "other"


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 50:
        return None
    xs = x[mask].astype(np.float64, copy=False)
    ys = y[mask].astype(np.float64, copy=False)
    if float(np.std(xs)) <= 1e-12 or float(np.std(ys)) <= 1e-12:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _z_diff(values: np.ndarray, mask: np.ndarray) -> float | None:
    finite = np.isfinite(values) & np.isfinite(mask.astype(float))
    if int(finite.sum()) < 50:
        return None
    vals = values[finite].astype(np.float64, copy=False)
    m = mask[finite]
    if int(m.sum()) < 10 or int((~m).sum()) < 10:
        return None
    std = float(np.std(vals))
    if std <= 1e-12:
        return None
    return float((vals[m].mean() - vals[~m].mean()) / std)


def _max_drawdown(vals: pd.Series) -> float:
    arr = pd.to_numeric(vals, errors="coerce").fillna(0.0).to_numpy(np.float64)
    if arr.size == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(arr)])
    dd = curve - np.maximum.accumulate(curve)
    return float(abs(np.min(dd)))


def _profit_factor(vals: pd.Series) -> float | None:
    arr = pd.to_numeric(vals, errors="coerce").to_numpy(np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    gains = arr[arr > 0].sum()
    losses = arr[arr < 0].sum()
    if losses == 0:
        return None
    return float(gains / abs(losses))


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["net_pnl_bps"], errors="coerce") if not frame.empty else pd.Series(dtype=float)
    return {
        "n_trades": int(len(frame)),
        "net_sum_bps": float(pnl.sum()) if len(pnl) else 0.0,
        "net_mean_bps": float(pnl.mean()) if len(pnl) else None,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
        "profit_factor": _profit_factor(pnl),
        "max_loss_bps": float(pnl.min()) if len(pnl) else None,
        "max_drawdown_bps": _max_drawdown(pnl),
    }


def _read_existing_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def _load_targets(dataset_dir: Path, splits: list[str]) -> pd.DataFrame:
    files = _split_files(dataset_dir, splits)
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = files[split]
        existing = set(_read_existing_columns(path))
        cols = [c for c in TARGET_COLUMNS if c in existing]
        frame = pd.read_parquet(path, columns=cols)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame["source_split"] = split
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).sort_values("time", kind="mergesort").reset_index(drop=True)
    return out


def _build_all_features(dataset_dir: Path, splits: list[str]) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    x, _y, df, base_names, _cat = _load_all_data(dataset_dir, splits)
    chart_x, chart_names = _build_chart_layer(x, base_names)
    chart_all_x = np.concatenate([x, chart_x], axis=1).astype(np.float32, copy=False)
    chart_all_names = list(base_names) + list(chart_names)
    deep_x, deep_names = _build_deep_interaction_layer(chart_all_x, chart_all_names, df)
    all_x = np.concatenate([x, chart_x, deep_x], axis=1).astype(np.float32, copy=False)
    all_names = list(base_names) + list(chart_names) + list(deep_names)
    out_df = df[["time", "session", "source_split"]].copy()
    return all_x, out_df, all_names


def _feature_inventory(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_index": np.arange(len(names)),
            "feature": names,
            "family": [_feature_family(n) for n in names],
            "is_generated": [str(n).startswith("chart.") for n in names],
            "is_deep": [
                str(n).startswith("chart.")
                and any(k in str(n) for k in ["cross", "delta", "tail_pressure", "tail_risk", "_x_", "regime_", "compression_release"])
                for n in names
            ],
        }
    )


def _dataset_feature_diagnostics(
    x: np.ndarray,
    names: list[str],
    targets: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    n = len(targets)
    if max_rows > 0 and n > max_rows:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(np.arange(n), size=max_rows, replace=False))
    else:
        sample_idx = np.arange(n)
    xs = x[sample_idx].astype(np.float32, copy=False)
    tg = targets.iloc[sample_idx].reset_index(drop=True)
    target_series: dict[str, np.ndarray] = {}
    for col in TARGET_COLUMNS:
        if col == "time" or col not in tg:
            continue
        target_series[col] = pd.to_numeric(tg[col], errors="coerce").to_numpy(np.float64)
    flags: dict[str, np.ndarray] = {}
    if "path_quality_bps" in target_series:
        arr = target_series["path_quality_bps"]
        flags["path_quality_bottom10"] = arr <= np.nanpercentile(arr, 10)
    if "mae_first_n_bps" in target_series:
        arr = target_series["mae_first_n_bps"]
        flags["mae_top10"] = arr >= np.nanpercentile(arr, 90)
    if "y_bad_path" in target_series:
        flags["y_bad_path"] = target_series["y_bad_path"] > 0.5
    for side in ["long", "short"]:
        for k in ["K48", "K96"]:
            col = f"y_tail_mae_{side}_{k}"
            if col in target_series:
                arr = target_series[col]
                flags[f"{col}_top10"] = arr >= np.nanpercentile(arr, 90)

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        vals = xs[:, i].astype(np.float64, copy=False)
        row: dict[str, Any] = {
            "feature_index": i,
            "feature": name,
            "family": _feature_family(name),
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "q05": float(np.nanpercentile(vals, 5)),
            "q50": float(np.nanpercentile(vals, 50)),
            "q95": float(np.nanpercentile(vals, 95)),
        }
        for target_name, target in target_series.items():
            corr = _safe_corr(vals, target)
            if corr is not None:
                row[f"corr__{target_name}"] = corr
        for flag_name, flag in flags.items():
            zd = _z_diff(vals, np.asarray(flag, dtype=bool))
            if zd is not None:
                row[f"zdiff__{flag_name}"] = zd
        rows.append(row)
    return pd.DataFrame(rows)


def _select_policy_ids(leader: pd.DataFrame, explicit: list[str], n: int) -> list[str]:
    if explicit:
        return explicit
    frame = leader[
        (leader["scope"] == "all")
        & (leader["n_trades"] >= 100)
        & (leader["positive_months"] >= 6)
        & (leader["max_drawdown_bps"] <= 180)
        & (leader["net_sum_bps"] > 0)
    ].copy()
    if frame.empty:
        frame = leader[leader["scope"] == "all"].copy()
    frame = frame.sort_values(["risk_adjusted_score", "net_sum_bps"], ascending=[False, False])
    return [str(x) for x in frame["policy_id"].head(n).tolist()]


def _join_trade_features(trades: pd.DataFrame, feature_df: pd.DataFrame, x: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    feature_df = feature_df.reset_index().rename(columns={"index": "feature_row"})
    idx = feature_df.drop_duplicates("time").set_index("time")["feature_row"]
    entry_times = pd.to_datetime(trades["entry_time"], utc=True)
    row_idx = idx.reindex(entry_times).to_numpy()
    ok = pd.notna(row_idx)
    joined = trades.loc[ok].reset_index(drop=True).copy()
    rows = row_idx[ok].astype(np.int64)
    return joined, x[rows]


def _trade_feature_diagnostics(trades: pd.DataFrame, trade_x: np.ndarray, names: list[str], policy_id: str) -> pd.DataFrame:
    pnl = pd.to_numeric(trades["net_pnl_bps"], errors="coerce").to_numpy(np.float64)
    mae = pd.to_numeric(trades["mae_bps"], errors="coerce").to_numpy(np.float64)
    mfe = pd.to_numeric(trades["mfe_bps"], errors="coerce").to_numpy(np.float64)
    flags = {
        "loss_le_minus50": pnl <= -50.0,
        "loss_le_minus80": pnl <= -80.0,
        "mae_ge_80": mae >= 80.0,
        "giveback_mfe60_loss": (mfe >= 60.0) & (pnl < 0.0),
        "winner_ge_50": pnl >= 50.0,
    }
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        vals = trade_x[:, i].astype(np.float64, copy=False)
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "feature_index": i,
            "feature": name,
            "family": _feature_family(name),
            "corr_net_pnl_bps": _safe_corr(vals, pnl),
            "corr_mae_bps": _safe_corr(vals, mae),
            "corr_mfe_bps": _safe_corr(vals, mfe),
        }
        for flag, mask in flags.items():
            zd = _z_diff(vals, mask)
            if zd is not None:
                row[f"zdiff__{flag}"] = zd
        rows.append(row)
    return pd.DataFrame(rows)


def _veto_search(trades: pd.DataFrame, trade_x: np.ndarray, names: list[str], policy_id: str) -> pd.DataFrame:
    base_metrics = _metrics(trades)
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        vals = trade_x[:, i].astype(np.float64, copy=False)
        if np.nanstd(vals) <= 1e-12:
            continue
        for side in ["high", "low"]:
            for q in [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95]:
                threshold = float(np.nanpercentile(vals, q * 100.0))
                skip = vals >= threshold if side == "high" else vals <= threshold
                kept = trades.loc[~skip].copy()
                if len(kept) < 80 or len(kept) > len(trades) * 0.95:
                    continue
                m = _metrics(kept)
                rows.append(
                    {
                        "policy_id": policy_id,
                        "diagnostic_leaky_threshold": True,
                        "feature": name,
                        "family": _feature_family(name),
                        "skip_side": side,
                        "quantile": q,
                        "threshold": threshold,
                        "skipped_trades": int(skip.sum()),
                        "kept_trades": int(len(kept)),
                        "base_net_sum_bps": base_metrics["net_sum_bps"],
                        "base_max_drawdown_bps": base_metrics["max_drawdown_bps"],
                        "base_max_loss_bps": base_metrics["max_loss_bps"],
                        **{f"kept_{k}": v for k, v in m.items()},
                        "delta_net_sum_bps": float(m["net_sum_bps"] - base_metrics["net_sum_bps"]),
                        "delta_max_drawdown_bps": float(m["max_drawdown_bps"] - base_metrics["max_drawdown_bps"]),
                        "delta_max_loss_bps": (
                            float(m["max_loss_bps"] - base_metrics["max_loss_bps"])
                            if m["max_loss_bps"] is not None and base_metrics["max_loss_bps"] is not None
                            else None
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["tail_improvement_score"] = (
        -out["delta_max_drawdown_bps"].fillna(0.0)
        -out["delta_max_loss_bps"].fillna(0.0)
        + 0.25 * out["delta_net_sum_bps"].fillna(0.0)
    )
    return out.sort_values(["tail_improvement_score", "kept_net_sum_bps"], ascending=[False, False])


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    replay_dir = Path(args.replay_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = _parse_csv(args.data_splits)

    x, feature_df, names = _build_all_features(dataset_dir, splits)
    targets = _load_targets(dataset_dir, splits)
    if len(targets) != len(feature_df) or not targets["time"].equals(feature_df["time"]):
        raise RuntimeError("target rows do not align with loaded feature rows")
    inventory = _feature_inventory(names)
    inventory_path = out_dir / "feature_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    feature_diag = _dataset_feature_diagnostics(
        x=x,
        names=names,
        targets=targets,
        max_rows=int(args.max_rows_feature_diag),
        seed=int(args.seed),
    )
    feature_diag_path = out_dir / "dataset_feature_tail_diagnostics.csv"
    feature_diag.to_csv(feature_diag_path, index=False)

    leader = pd.read_csv(replay_dir / "leaderboard.csv")
    trades = pd.read_csv(replay_dir / "ablation_policy_trades.csv")
    policy_ids = _select_policy_ids(leader, _parse_csv(args.policy_ids), int(args.max_policies))

    trade_diag_frames: list[pd.DataFrame] = []
    veto_frames: list[pd.DataFrame] = []
    pattern_rows: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        frame = trades[trades["policy_id"] == policy_id].copy()
        if frame.empty:
            continue
        joined, trade_x = _join_trade_features(frame, feature_df, x, names)
        if joined.empty:
            continue
        diag = _trade_feature_diagnostics(joined, trade_x, names, policy_id)
        trade_diag_frames.append(diag)
        veto_frames.append(_veto_search(joined, trade_x, names, policy_id))

        for keys in [["session"], ["side"], ["fold"], ["entry_month"], ["session", "side"]]:
            grouped = joined.groupby(keys, dropna=False)
            for key, g in grouped:
                m = _metrics(g)
                pattern_rows.append(
                    {
                        "policy_id": policy_id,
                        "group_by": "+".join(keys),
                        "group": str(key),
                        **m,
                    }
                )

    trade_diag = pd.concat(trade_diag_frames, ignore_index=True) if trade_diag_frames else pd.DataFrame()
    trade_diag_path = out_dir / "candidate_trade_feature_tail_diagnostics.csv"
    trade_diag.to_csv(trade_diag_path, index=False)

    veto = pd.concat(veto_frames, ignore_index=True) if veto_frames else pd.DataFrame()
    veto_path = out_dir / "diagnostic_veto_rule_search.csv"
    veto.to_csv(veto_path, index=False)

    patterns = pd.DataFrame(pattern_rows)
    patterns_path = out_dir / "candidate_tail_pattern_groups.csv"
    patterns.to_csv(patterns_path, index=False)

    top_dataset = []
    if not feature_diag.empty:
        zd_cols = [c for c in feature_diag.columns if c.startswith("zdiff__")]
        for col in zd_cols:
            top = feature_diag.reindex(feature_diag[col].abs().sort_values(ascending=False).index).head(10)
            for row in top[["feature", "family", col]].to_dict(orient="records"):
                top_dataset.append({"metric": col, **row})

    summary = {
        "dataset_dir": str(dataset_dir),
        "replay_dir": str(replay_dir),
        "out_dir": str(out_dir),
        "n_features": int(len(names)),
        "feature_family_counts": inventory["family"].value_counts().to_dict(),
        "policy_ids": policy_ids,
        "outputs": {
            "feature_inventory_csv": str(inventory_path),
            "dataset_feature_tail_diagnostics_csv": str(feature_diag_path),
            "candidate_trade_feature_tail_diagnostics_csv": str(trade_diag_path),
            "diagnostic_veto_rule_search_csv": str(veto_path),
            "candidate_tail_pattern_groups_csv": str(patterns_path),
        },
        "top_dataset_tail_features": top_dataset[:80],
        "top_diagnostic_veto_rules": (
            veto.head(30).to_dict(orient="records") if not veto.empty else []
        ),
        "note": "diagnostic_veto_rule_search uses in-sample thresholds; replay retest required before use",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--policy-ids", default="")
    ap.add_argument("--max-policies", type=int, default=6)
    ap.add_argument("--max-rows-feature-diag", type=int, default=180000)
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
