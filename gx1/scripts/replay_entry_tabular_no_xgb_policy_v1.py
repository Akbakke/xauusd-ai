"""Offline replay/policy gate for tabular no-XGB Entry models.

This script is deliberately separate from live execution. It consumes the
walk-forward LightGBM models, calibrates score thresholds only on each
pre-fold validation tail, then replays the evaluated fold chronologically with
basic policy constraints: one position at a time, cooldown, max trades/day,
daily loss limit, sizing, and explicit cost/slippage stress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gx1.scripts.evaluate_entry_selective_edge_v1 import (
    CLASS_NAMES,
    _json_default,
    _parse_float_list,
)
from gx1.scripts.evaluate_entry_tabular_no_xgb_baseline_v1 import (
    _check_no_xgb_feature_names,
    _predict_proba,
)
from gx1.scripts.evaluate_entry_tabular_no_xgb_walkforward_v1 import (
    _default_folds,
    _fold_indices,
    _load_all_data,
    _parse_folds,
)


@dataclass(frozen=True)
class SourceTape:
    times: np.ndarray
    index: pd.Index
    bid_close: np.ndarray
    ask_close: np.ndarray
    bid_high: np.ndarray
    bid_low: np.ndarray
    ask_high: np.ndarray
    ask_low: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "SourceTape":
        cols = ["time", "bid_close", "ask_close", "bid_high", "bid_low", "ask_high", "ask_low"]
        src = pd.read_parquet(path, columns=cols)
        src["time"] = pd.to_datetime(src["time"], utc=True)
        src = src.sort_values("time").reset_index(drop=True)
        return cls(
            times=src["time"].to_numpy(),
            index=pd.Index(src["time"]),
            bid_close=src["bid_close"].to_numpy(np.float64),
            ask_close=src["ask_close"].to_numpy(np.float64),
            bid_high=src["bid_high"].to_numpy(np.float64),
            bid_low=src["bid_low"].to_numpy(np.float64),
            ask_high=src["ask_high"].to_numpy(np.float64),
            ask_low=src["ask_low"].to_numpy(np.float64),
        )

    def indices_for_times(self, sample_times: pd.Series) -> np.ndarray:
        idx = self.index.get_indexer(pd.to_datetime(sample_times, utc=True))
        if np.any(idx < 0):
            raise RuntimeError(f"{int((idx < 0).sum())} sample times missing from source parquet")
        return idx.astype(np.int64, copy=False)

    def simulate_trade(
        self,
        *,
        start_idx: int,
        horizon_bars: int,
        side: int,
        exit_mode: str,
        take_profit_bps: float,
        stop_loss_bps: float,
        same_bar_policy: str,
    ) -> dict[str, Any] | None:
        start = int(start_idx)
        end = start + int(horizon_bars)
        if start < 0 or horizon_bars <= 0 or end >= len(self.times):
            return None

        if side == 0:
            entry_price = float(self.ask_close[start])
        elif side == 1:
            entry_price = float(self.bid_close[start])
        else:
            return None
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None

        exit_idx = end
        exit_reason = "horizon"
        exit_price: float
        gross_pnl_bps: float

        if exit_mode == "stop_tp":
            if take_profit_bps <= 0 or stop_loss_bps <= 0:
                raise RuntimeError("stop_tp exit requires positive take-profit and stop-loss bps")
            hit = self._first_stop_tp_hit(
                start=start,
                end=end,
                side=side,
                entry_price=entry_price,
                take_profit_bps=float(take_profit_bps),
                stop_loss_bps=float(stop_loss_bps),
                same_bar_policy=same_bar_policy,
            )
            if hit is not None:
                exit_idx = int(hit["exit_idx"])
                exit_reason = str(hit["exit_reason"])
                exit_price = float(hit["exit_price"])
                gross_pnl_bps = float(hit["gross_pnl_bps"])
            else:
                exit_price, gross_pnl_bps = self._horizon_exit(start=start, end=end, side=side, entry_price=entry_price)
        elif exit_mode == "horizon":
            exit_price, gross_pnl_bps = self._horizon_exit(start=start, end=end, side=side, entry_price=entry_price)
        else:
            raise RuntimeError(f"unknown exit mode: {exit_mode}")

        if not np.isfinite(exit_price) or not np.isfinite(gross_pnl_bps):
            return None

        mfe_bps, mae_bps = self._mfe_mae(start=start, end=exit_idx, side=side, entry_price=entry_price)
        return {
            "entry_src_idx": start,
            "exit_src_idx": exit_idx,
            "entry_time": pd.Timestamp(self.times[start]),
            "exit_time": pd.Timestamp(self.times[exit_idx]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl_bps": gross_pnl_bps,
            "mfe_bps": mfe_bps,
            "mae_bps": mae_bps,
            "held_bars": int(exit_idx - start),
            "exit_reason": exit_reason,
        }

    def _horizon_exit(self, *, start: int, end: int, side: int, entry_price: float) -> tuple[float, float]:
        if side == 0:
            exit_price = float(self.bid_close[end])
            gross_pnl_bps = (exit_price - entry_price) / entry_price * 1e4
        else:
            exit_price = float(self.ask_close[end])
            gross_pnl_bps = (entry_price - exit_price) / entry_price * 1e4
        return exit_price, float(gross_pnl_bps)

    def _first_stop_tp_hit(
        self,
        *,
        start: int,
        end: int,
        side: int,
        entry_price: float,
        take_profit_bps: float,
        stop_loss_bps: float,
        same_bar_policy: str,
    ) -> dict[str, Any] | None:
        for idx in range(start + 1, end + 1):
            if side == 0:
                tp_price = entry_price * (1.0 + take_profit_bps / 1e4)
                sl_price = entry_price * (1.0 - stop_loss_bps / 1e4)
                hit_tp = bool(self.bid_high[idx] >= tp_price)
                hit_sl = bool(self.bid_low[idx] <= sl_price)
            else:
                tp_price = entry_price * (1.0 - take_profit_bps / 1e4)
                sl_price = entry_price * (1.0 + stop_loss_bps / 1e4)
                hit_tp = bool(self.ask_low[idx] <= tp_price)
                hit_sl = bool(self.ask_high[idx] >= sl_price)

            if not hit_tp and not hit_sl:
                continue
            if hit_tp and hit_sl and same_bar_policy == "target_first":
                hit_sl = False
            elif hit_tp and hit_sl:
                hit_tp = False

            if hit_tp:
                return {
                    "exit_idx": idx,
                    "exit_price": tp_price,
                    "gross_pnl_bps": take_profit_bps,
                    "exit_reason": "take_profit",
                }
            return {
                "exit_idx": idx,
                "exit_price": sl_price,
                "gross_pnl_bps": -stop_loss_bps,
                "exit_reason": "stop_loss",
            }
        return None

    def _mfe_mae(self, *, start: int, end: int, side: int, entry_price: float) -> tuple[float | None, float | None]:
        if end <= start:
            return None, None
        fut = slice(start + 1, end + 1)
        if side == 0:
            mfe = (np.nanmax(self.bid_high[fut]) - entry_price) / entry_price * 1e4
            mae = (entry_price - np.nanmin(self.bid_low[fut])) / entry_price * 1e4
        else:
            mfe = (entry_price - np.nanmin(self.ask_low[fut])) / entry_price * 1e4
            mae = (np.nanmax(self.ask_high[fut]) - entry_price) / entry_price * 1e4
        return (
            float(mfe) if np.isfinite(mfe) else None,
            float(mae) if np.isfinite(mae) else None,
        )


def _decision_arrays(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen_side = np.where(probs[:, 0] >= probs[:, 1], 0, 1).astype(np.int64)
    chosen_prob = np.maximum(probs[:, 0], probs[:, 1]).astype(np.float64, copy=False)
    score = chosen_prob - probs[:, 2]
    return chosen_side, chosen_prob, score.astype(np.float64, copy=False)


def _threshold_from_scores(scores: np.ndarray, top_frac: float, min_score_floor: float) -> float:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("cannot calibrate threshold from empty/non-finite validation scores")
    k = max(1, int(math.ceil(finite.size * float(top_frac))))
    kth = finite.size - k
    threshold = float(np.partition(finite, kth)[kth])
    return max(threshold, float(min_score_floor))


def _frac_label(frac: float) -> str:
    pct = float(frac) * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"top{int(round(pct)):02d}"
    return "top" + str(frac).replace(".", "p")


def _cost_label(cost: float) -> str:
    if abs(float(cost) - round(float(cost))) < 1e-9:
        return f"cost{int(round(float(cost))):02d}"
    return "cost" + str(cost).replace(".", "p")


def _policy_hash(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _run_policy(
    *,
    fold_id: str,
    eval_df: pd.DataFrame,
    probs: np.ndarray,
    source_idx: np.ndarray,
    tape: SourceTape,
    threshold_top_frac: float,
    score_threshold: float,
    cost_stress_bps: float,
    args: argparse.Namespace,
    policy_id: str,
    policy_config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chosen_side, chosen_prob, score = _decision_arrays(probs)
    counts: dict[str, Any] = {
        "fold": fold_id,
        "policy_id": policy_id,
        "policy_config_hash": policy_config_hash,
        "threshold_top_frac": float(threshold_top_frac),
        "score_threshold": float(score_threshold),
        "cost_stress_bps": float(cost_stress_bps),
        "rows": int(len(eval_df)),
        "below_threshold": 0,
        "above_threshold": 0,
        "below_min_direction_prob": 0,
        "skipped_open_or_cooldown": 0,
        "skipped_max_trades_day": 0,
        "skipped_daily_loss_limit": 0,
        "skipped_invalid_path": 0,
        "trades": 0,
    }
    trades: list[dict[str, Any]] = []
    unavailable_until_src_idx = -1
    daily_trade_count: dict[str, int] = {}
    daily_pnl_bps: dict[str, float] = {}
    labels = eval_df["y_direction"].to_numpy(np.int64)

    for i, row in enumerate(eval_df.itertuples(index=False)):
        row_score = float(score[i])
        if not np.isfinite(row_score) or row_score < score_threshold:
            counts["below_threshold"] += 1
            continue
        counts["above_threshold"] += 1

        if float(chosen_prob[i]) < float(args.min_direction_prob):
            counts["below_min_direction_prob"] += 1
            continue

        start_src_idx = int(source_idx[i])
        if start_src_idx <= unavailable_until_src_idx:
            counts["skipped_open_or_cooldown"] += 1
            continue

        entry_time = pd.Timestamp(row.time)
        day = entry_time.date().isoformat()
        if int(args.max_trades_per_day) > 0 and daily_trade_count.get(day, 0) >= int(args.max_trades_per_day):
            counts["skipped_max_trades_day"] += 1
            continue
        if (
            float(args.daily_loss_limit_bps) > 0
            and daily_pnl_bps.get(day, 0.0) <= -float(args.daily_loss_limit_bps)
        ):
            counts["skipped_daily_loss_limit"] += 1
            continue

        sim = tape.simulate_trade(
            start_idx=start_src_idx,
            horizon_bars=int(row.label_horizon_bars),
            side=int(chosen_side[i]),
            exit_mode=str(args.exit_mode),
            take_profit_bps=float(args.take_profit_bps),
            stop_loss_bps=float(args.stop_loss_bps),
            same_bar_policy=str(args.same_bar_policy),
        )
        if sim is None:
            counts["skipped_invalid_path"] += 1
            continue

        net_pnl_bps = (
            float(sim["gross_pnl_bps"])
            - float(cost_stress_bps)
            - float(args.slippage_bps)
        ) * float(args.size_multiplier)
        trade = {
            "fold": fold_id,
            "policy_id": policy_id,
            "policy_config_hash": policy_config_hash,
            "threshold_top_frac": float(threshold_top_frac),
            "score_threshold": float(score_threshold),
            "cost_stress_bps": float(cost_stress_bps),
            "slippage_bps": float(args.slippage_bps),
            "size_multiplier": float(args.size_multiplier),
            "source_split": str(row.source_split),
            "session": str(row.session),
            "entry_day": day,
            "entry_month": entry_time.strftime("%Y-%m"),
            "entry_time": sim["entry_time"],
            "exit_time": sim["exit_time"],
            "side": CLASS_NAMES[int(chosen_side[i])],
            "label": CLASS_NAMES.get(int(labels[i]), str(int(labels[i]))),
            "direction_correct": bool(int(labels[i]) == int(chosen_side[i])),
            "score": row_score,
            "chosen_prob": float(chosen_prob[i]),
            "p_long": float(probs[i, 0]),
            "p_short": float(probs[i, 1]),
            "p_flat": float(probs[i, 2]),
            "entry_price": sim["entry_price"],
            "exit_price": sim["exit_price"],
            "gross_pnl_bps": sim["gross_pnl_bps"],
            "net_pnl_bps": net_pnl_bps,
            "mfe_bps": sim["mfe_bps"],
            "mae_bps": sim["mae_bps"],
            "horizon_bars": int(row.label_horizon_bars),
            "held_bars": sim["held_bars"],
            "exit_reason": sim["exit_reason"],
        }
        trades.append(trade)
        counts["trades"] += 1
        daily_trade_count[day] = daily_trade_count.get(day, 0) + 1
        daily_pnl_bps[day] = daily_pnl_bps.get(day, 0.0) + float(net_pnl_bps)
        unavailable_until_src_idx = int(sim["exit_src_idx"]) + int(args.cooldown_bars)

    return trades, counts


def _safe_mean(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def _safe_percentile(values: pd.Series, q: float) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, q)) if vals.size else None


def _max_drawdown(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(np.float64)
    if vals.size == 0:
        return 0.0, 0.0
    curve = np.concatenate([[0.0], np.cumsum(vals)])
    drawdown = curve - np.maximum.accumulate(curve)
    signed = float(np.min(drawdown))
    return abs(signed), signed


def _profit_factor(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    gains = vals[vals > 0].sum()
    losses = vals[vals < 0].sum()
    if losses == 0:
        return None
    return float(gains / abs(losses))


def _metrics_row(scope: str, fold: str, policy_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "scope": scope,
            "fold": fold,
            "policy_id": policy_id,
            "n_trades": 0,
        }
    dd_abs, dd_signed = _max_drawdown(frame["net_pnl_bps"])
    pnl = pd.to_numeric(frame["net_pnl_bps"], errors="coerce")
    gross = pd.to_numeric(frame["gross_pnl_bps"], errors="coerce")
    return {
        "scope": scope,
        "fold": fold,
        "policy_id": policy_id,
        "policy_config_hash": frame["policy_config_hash"].iloc[0],
        "threshold_top_frac": float(frame["threshold_top_frac"].iloc[0]),
        "cost_stress_bps": float(frame["cost_stress_bps"].iloc[0]),
        "slippage_bps": float(frame["slippage_bps"].iloc[0]),
        "size_multiplier": float(frame["size_multiplier"].iloc[0]),
        "n_trades": int(len(frame)),
        "n_days": int(frame["entry_day"].nunique()),
        "n_months": int(frame["entry_month"].nunique()),
        "net_sum_bps": float(pnl.sum()),
        "net_mean_bps": _safe_mean(pnl),
        "net_median_bps": _safe_percentile(pnl, 50),
        "net_p10_bps": _safe_percentile(pnl, 10),
        "net_p90_bps": _safe_percentile(pnl, 90),
        "gross_mean_bps": _safe_mean(gross),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": _profit_factor(pnl),
        "max_win_bps": float(pnl.max()),
        "max_loss_bps": float(pnl.min()),
        "max_drawdown_bps": dd_abs,
        "max_drawdown_signed_bps": dd_signed,
        "mean_score": _safe_mean(frame["score"]),
        "mean_mfe_bps": _safe_mean(frame["mfe_bps"]),
        "mean_mae_bps": _safe_mean(frame["mae_bps"]),
        "long_rate": float((frame["side"] == "LONG").mean()),
        "short_rate": float((frame["side"] == "SHORT").mean()),
        "direction_precision": float(frame["direction_correct"].mean()),
        "avg_trades_per_day": float(len(frame) / max(frame["entry_day"].nunique(), 1)),
    }


def _aggregate_outputs(trades: pd.DataFrame, decisions: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    trades_path = out_dir / "replay_policy_trades.csv"
    decisions_path = out_dir / "replay_policy_decisions.csv"
    metrics_path = out_dir / "replay_policy_metrics.csv"
    daily_path = out_dir / "replay_policy_daily.csv"
    monthly_path = out_dir / "replay_policy_monthly.csv"

    trades.to_csv(trades_path, index=False)
    decisions.to_csv(decisions_path, index=False)

    metric_rows: list[dict[str, Any]] = []
    if trades.empty:
        for policy_id in sorted(decisions["policy_id"].unique()) if "policy_id" in decisions else []:
            metric_rows.append(_metrics_row("all", "ALL", str(policy_id), trades))
    else:
        for (policy_id, fold), frame in trades.groupby(["policy_id", "fold"], sort=True):
            metric_rows.append(_metrics_row("fold", str(fold), str(policy_id), frame))
        for policy_id, frame in trades.groupby("policy_id", sort=True):
            metric_rows.append(_metrics_row("all", "ALL", str(policy_id), frame))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(metrics_path, index=False)

    if trades.empty:
        pd.DataFrame().to_csv(daily_path, index=False)
        pd.DataFrame().to_csv(monthly_path, index=False)
    else:
        daily = (
            trades.groupby(["policy_id", "entry_day"], as_index=False)
            .agg(
                n_trades=("net_pnl_bps", "size"),
                net_sum_bps=("net_pnl_bps", "sum"),
                net_mean_bps=("net_pnl_bps", "mean"),
                wins=("net_pnl_bps", lambda s: int((s > 0).sum())),
            )
        )
        daily["win_rate"] = daily["wins"] / daily["n_trades"].clip(lower=1)
        daily.to_csv(daily_path, index=False)

        monthly = (
            trades.groupby(["policy_id", "entry_month"], as_index=False)
            .agg(
                n_trades=("net_pnl_bps", "size"),
                net_sum_bps=("net_pnl_bps", "sum"),
                net_mean_bps=("net_pnl_bps", "mean"),
                wins=("net_pnl_bps", lambda s: int((s > 0).sum())),
            )
        )
        monthly["win_rate"] = monthly["wins"] / monthly["n_trades"].clip(lower=1)
        monthly.to_csv(monthly_path, index=False)

    return {
        "trades_csv": str(trades_path),
        "decisions_csv": str(decisions_path),
        "metrics_csv": str(metrics_path),
        "daily_csv": str(daily_path),
        "monthly_csv": str(monthly_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    walkforward_dir = Path(args.walkforward_dir).expanduser().resolve()
    model_dir = walkforward_dir / "models"
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_splits = [s.strip() for s in str(args.data_splits).split(",") if s.strip()]
    folds = _parse_folds(args.folds) if args.folds else _default_folds()
    threshold_top_fracs = _parse_float_list(args.threshold_top_fracs)
    cost_stress_bps_values = _parse_float_list(args.cost_stress_bps)

    x, _y, df, feature_names, _categorical_idx = _load_all_data(dataset_dir, data_splits)
    _check_no_xgb_feature_names(feature_names)
    tape = SourceTape.load(source_parquet)

    all_trades: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    base_policy_config = {
        "model_name": str(args.model_name),
        "data_splits": data_splits,
        "exit_mode": str(args.exit_mode),
        "take_profit_bps": float(args.take_profit_bps),
        "stop_loss_bps": float(args.stop_loss_bps),
        "same_bar_policy": str(args.same_bar_policy),
        "cooldown_bars": int(args.cooldown_bars),
        "max_trades_per_day": int(args.max_trades_per_day),
        "daily_loss_limit_bps": float(args.daily_loss_limit_bps),
        "min_direction_prob": float(args.min_direction_prob),
        "min_score_floor": float(args.min_score_floor),
        "slippage_bps": float(args.slippage_bps),
        "size_multiplier": float(args.size_multiplier),
        "threshold_source": "pre_fold_validation_tail",
    }

    for fold in folds:
        train_idx, val_idx, eval_idx = _fold_indices(
            times=df["time"],
            fold=fold,
            val_tail_days=int(args.val_tail_days),
            min_val_rows=int(args.min_val_rows),
            min_train_rows=int(args.min_train_rows),
        )
        del train_idx

        model_path = model_dir / f"{fold.fold_id}__{args.model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"missing walk-forward model: {model_path}")
        model = joblib.load(model_path)

        val_probs = _predict_proba(model, x[val_idx])
        eval_probs = _predict_proba(model, x[eval_idx])
        _val_side, _val_prob, val_score = _decision_arrays(val_probs)
        eval_df = df.iloc[eval_idx].reset_index(drop=True)
        src_idx = tape.indices_for_times(eval_df["time"])

        for top_frac in threshold_top_fracs:
            threshold = _threshold_from_scores(val_score, top_frac, float(args.min_score_floor))
            threshold_rows.append({
                "fold": fold.fold_id,
                "threshold_top_frac": float(top_frac),
                "score_threshold": threshold,
                "val_rows": int(len(val_idx)),
                "eval_rows": int(len(eval_idx)),
                "val_score_mean": float(np.mean(val_score)),
                "val_score_p50": float(np.percentile(val_score, 50)),
                "val_score_p90": float(np.percentile(val_score, 90)),
                "val_score_p95": float(np.percentile(val_score, 95)),
            })
            for cost_bps in cost_stress_bps_values:
                policy_config = {
                    **base_policy_config,
                    "threshold_top_frac": float(top_frac),
                    "cost_stress_bps": float(cost_bps),
                }
                config_hash = _policy_hash(policy_config)
                policy_id = f"{_frac_label(top_frac)}_{_cost_label(cost_bps)}_{args.exit_mode}_{config_hash}"
                trades, decisions = _run_policy(
                    fold_id=fold.fold_id,
                    eval_df=eval_df,
                    probs=eval_probs,
                    source_idx=src_idx,
                    tape=tape,
                    threshold_top_frac=float(top_frac),
                    score_threshold=threshold,
                    cost_stress_bps=float(cost_bps),
                    args=args,
                    policy_id=policy_id,
                    policy_config_hash=config_hash,
                )
                all_trades.extend(trades)
                decision_rows.append(decisions)

    trades_df = pd.DataFrame(all_trades)
    decisions_df = pd.DataFrame(decision_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    thresholds_path = out_dir / "replay_policy_thresholds.csv"
    thresholds_df.to_csv(thresholds_path, index=False)
    outputs = _aggregate_outputs(trades_df, decisions_df, out_dir)
    outputs["thresholds_csv"] = str(thresholds_path)

    metrics_df = pd.read_csv(outputs["metrics_csv"]) if Path(outputs["metrics_csv"]).stat().st_size > 0 else pd.DataFrame()
    summary = {
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "walkforward_dir": str(walkforward_dir),
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "folds": [{"fold_id": f.fold_id, "start": str(f.start), "end": str(f.end)} for f in folds],
        "feature_policy": {
            "included": ["snap[7:]", "ctx_cont", "ctx_cat"],
            "n_features": len(feature_names),
            "excluded_xgb_derived_snap_fields": [
                "p_long",
                "p_short",
                "p_flat",
                "p_hat",
                "uncertainty_score",
                "margin_top1_top2",
                "entropy",
            ],
        },
        "policy_config": base_policy_config,
        "threshold_top_fracs": threshold_top_fracs,
        "cost_stress_bps": cost_stress_bps_values,
        "n_trades_total": int(len(trades_df)),
        "n_policy_fold_runs": int(len(decisions_df)),
        "outputs": outputs,
        "aggregate_metrics": (
            metrics_df[metrics_df["scope"] == "all"].to_dict(orient="records")
            if not metrics_df.empty and "scope" in metrics_df
            else []
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--source-parquet", required=True)
    ap.add_argument("--walkforward-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--folds", default="", help="Comma-separated id=start:end folds. Empty uses semiyear defaults.")
    ap.add_argument("--model-name", default="lightgbm_tabular_no_xgb_wf")
    ap.add_argument("--threshold-top-fracs", default="0.05,0.10")
    ap.add_argument("--cost-stress-bps", default="0,10")
    ap.add_argument("--val-tail-days", type=int, default=30)
    ap.add_argument("--min-val-rows", type=int, default=2500)
    ap.add_argument("--min-train-rows", type=int, default=50000)
    ap.add_argument("--exit-mode", choices=("horizon", "stop_tp"), default="horizon")
    ap.add_argument("--take-profit-bps", type=float, default=60.0)
    ap.add_argument("--stop-loss-bps", type=float, default=45.0)
    ap.add_argument("--same-bar-policy", choices=("stop_first", "target_first"), default="stop_first")
    ap.add_argument("--cooldown-bars", type=int, default=6)
    ap.add_argument("--max-trades-per-day", type=int, default=8)
    ap.add_argument("--daily-loss-limit-bps", type=float, default=150.0)
    ap.add_argument("--min-direction-prob", type=float, default=0.0)
    ap.add_argument("--min-score-floor", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--size-multiplier", type=float, default=1.0)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
