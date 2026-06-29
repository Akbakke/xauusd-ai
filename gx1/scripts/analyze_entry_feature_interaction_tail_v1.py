"""Mine Entry feature interactions that explain replay tail losses.

Offline research only. This script joins all base + generated chart/deep
features onto replay trades, screens every feature condition for tail-loss
enrichment, then tests pairwise compound conditions such as
``feature_a:high:0.90&feature_b:low:0.10``. Reported rules are diagnostic and
must be retested by fold-calibrated replay before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.evaluate_entry_selective_edge_v1 import _json_default
from gx1.scripts.evaluate_entry_tabular_no_xgb_walkforward_v1 import _load_all_data
from gx1.scripts.experiment_entry_chart_structure_ablation_v1 import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SOURCE_PARQUET,
    _build_chart_layer,
    _build_deep_interaction_layer,
    _build_price_derived_layer,
    _name_index,
)
from gx1.scripts.analyze_entry_feature_tail_exit_deep_v1 import _feature_family


DEFAULT_REPLAY_DIR = Path(
    "/home/andre2/GX1_DATA/reports/entry_chart_structure_multimodel_veto_cost10_20_20260628_v1"
)
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/entry_feature_interaction_tail_20260628_v1")


@dataclass(frozen=True)
class ConditionCandidate:
    feature: str
    family: str
    side: str
    quantile: float
    threshold: float
    mask: np.ndarray
    row: dict[str, Any]


def _parse_csv(raw: str) -> list[str]:
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _q_label(q: float) -> str:
    text = f"{float(q):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _rule_text(feature: str, side: str, quantile: float) -> str:
    return f"{feature}:{side}:{float(quantile):.6g}"


def _stable_rule_set_name(kind: str, policy_id: str, rule: str) -> str:
    raw = f"{kind}|{policy_id}|{rule}".encode("utf-8")
    return f"{kind}_{hashlib.sha256(raw).hexdigest()[:12]}"


def _scope_rule_text(rule: str, *, decision_side: str | None, trade_session: str | None) -> str:
    if not decision_side and not trade_session:
        return rule
    scoped: list[str] = []
    for raw_condition in [p.strip() for p in str(rule).split("&") if p.strip()]:
        pieces = raw_condition.split(":")
        if len(pieces) < 3:
            scoped.append(raw_condition)
            continue
        out = pieces[:]
        if decision_side and len(out) == 3:
            out.append(decision_side)
        if trade_session:
            if len(out) == 3:
                out.append("ANY")
            if len(out) == 4:
                out.append(trade_session)
        scoped.append(":".join(out))
    return "&".join(scoped)


def _filter_trade_scope(
    trades: pd.DataFrame,
    trade_x: np.ndarray,
    *,
    sessions: list[str],
    sides: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = np.ones(len(trades), dtype=bool)
    if sessions:
        allowed_sessions = {s.upper() for s in sessions}
        mask &= trades["session"].astype(str).str.upper().isin(allowed_sessions).to_numpy()
    if sides:
        allowed_sides = {s.upper() for s in sides}
        mask &= trades["side"].astype(str).str.upper().isin(allowed_sides).to_numpy()
    return trades.loc[mask].reset_index(drop=True), trade_x[mask]


def _build_all_features(
    dataset_dir: Path,
    source_parquet: Path,
    splits: list[str],
    *,
    include_price_ema_features: bool,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    x, _y, df, base_names, _cat = _load_all_data(dataset_dir, splits)
    chart_x, chart_names = _build_chart_layer(x, base_names)
    if include_price_ema_features:
        price_x, price_names = _build_price_derived_layer(df, source_parquet)
        if price_x.shape[1]:
            chart_x = (
                np.concatenate([chart_x, price_x], axis=1).astype(np.float32, copy=False)
                if chart_x.shape[1]
                else price_x
            )
            chart_names = list(chart_names) + list(price_names)
    chart_all_x = np.concatenate([x, chart_x], axis=1).astype(np.float32, copy=False)
    chart_all_names = list(base_names) + list(chart_names)
    deep_x, deep_names = _build_deep_interaction_layer(chart_all_x, chart_all_names, df)
    all_x = np.concatenate([x, chart_x, deep_x], axis=1).astype(np.float32, copy=False)
    all_names = list(base_names) + list(chart_names) + list(deep_names)
    feature_df = df[["time", "session", "source_split"]].copy()
    feature_df["time"] = pd.to_datetime(feature_df["time"], utc=True)
    return all_x, feature_df, all_names


def _max_drawdown(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(np.float64)
    if arr.size == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(arr)])
    dd = curve - np.maximum.accumulate(curve)
    return float(abs(np.min(dd)))


def _profit_factor(values: pd.Series | np.ndarray) -> float | None:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    gains = float(arr[arr > 0].sum())
    losses = float(arr[arr < 0].sum())
    if losses == 0.0:
        return None
    return gains / abs(losses)


def _metrics(values: pd.Series | np.ndarray) -> dict[str, Any]:
    pnl = pd.to_numeric(pd.Series(values), errors="coerce")
    if pnl.empty:
        return {
            "n_trades": 0,
            "net_sum_bps": 0.0,
            "net_mean_bps": None,
            "win_rate": None,
            "profit_factor": None,
            "max_loss_bps": None,
            "max_drawdown_bps": 0.0,
        }
    return {
        "n_trades": int(len(pnl)),
        "net_sum_bps": float(pnl.sum()),
        "net_mean_bps": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": _profit_factor(pnl),
        "max_loss_bps": float(pnl.min()),
        "max_drawdown_bps": _max_drawdown(pnl),
    }


def _join_trade_features(trades: pd.DataFrame, feature_df: pd.DataFrame, x: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    row_by_time = feature_df.reset_index().drop_duplicates("time").set_index("time")["index"]
    entry_times = pd.to_datetime(trades["entry_time"], utc=True)
    row_idx = row_by_time.reindex(entry_times).to_numpy()
    ok = pd.notna(row_idx)
    joined = trades.loc[ok].reset_index(drop=True).copy()
    return joined, x[row_idx[ok].astype(np.int64)]


def _condition_mask(vals: np.ndarray, side: str, threshold: float) -> np.ndarray:
    cmp = np.nan_to_num(vals.astype(np.float64, copy=False), nan=threshold, posinf=threshold, neginf=threshold)
    if side == "low":
        return cmp < threshold
    if side == "high":
        return cmp > threshold
    raise RuntimeError(f"unknown condition side {side!r}")


def _candidate_row(
    *,
    policy_id: str,
    kind: str,
    entry_veto_rule: str,
    base: dict[str, Any],
    trades: pd.DataFrame,
    skip: np.ndarray,
    tail_mask: np.ndarray,
    mae_tail_mask: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    skip = np.asarray(skip, dtype=bool)
    n = int(len(trades))
    skipped = int(skip.sum())
    if n == 0 or skipped == 0:
        return None
    kept = trades.loc[~skip].copy()
    if kept.empty:
        return None
    m = _metrics(kept["net_pnl_bps"])
    skipped_pnl = pd.to_numeric(trades.loc[skip, "net_pnl_bps"], errors="coerce")
    skipped_tail = int((skip & tail_mask).sum())
    skipped_mae_tail = int((skip & mae_tail_mask).sum())
    delta_net = float(m["net_sum_bps"] - base["net_sum_bps"])
    delta_dd = float(m["max_drawdown_bps"] - base["max_drawdown_bps"])
    if m["max_loss_bps"] is None or base["max_loss_bps"] is None:
        delta_loss = None
    else:
        delta_loss = float(m["max_loss_bps"] - base["max_loss_bps"])
    tail_precision = float(skipped_tail / skipped)
    mae_tail_precision = float(skipped_mae_tail / skipped)
    skipped_net = float(skipped_pnl.sum())
    score = (
        -2.0 * delta_dd
        + 1.25 * float(delta_loss or 0.0)
        + 25.0 * skipped_tail
        + 12.5 * skipped_mae_tail
        + 0.08 * delta_net
        + 0.04 * max(0.0, -skipped_net)
    )
    row = {
        "policy_id": policy_id,
        "diagnostic_leaky_threshold": True,
        "kind": kind,
        "entry_veto_rule": entry_veto_rule,
        "entry_veto_rule_set": _stable_rule_set_name(kind, policy_id, entry_veto_rule),
        "n_trades": n,
        "skipped_trades": skipped,
        "skip_rate": float(skipped / n),
        "skipped_net_sum_bps": skipped_net,
        "skipped_net_mean_bps": float(skipped_pnl.mean()),
        "skipped_tail_count": skipped_tail,
        "skipped_tail_precision": tail_precision,
        "skipped_mae_tail_count": skipped_mae_tail,
        "skipped_mae_tail_precision": mae_tail_precision,
        "base_net_sum_bps": base["net_sum_bps"],
        "base_max_drawdown_bps": base["max_drawdown_bps"],
        "base_max_loss_bps": base["max_loss_bps"],
        **{f"kept_{k}": v for k, v in m.items()},
        "delta_net_sum_bps": delta_net,
        "delta_max_drawdown_bps": delta_dd,
        "delta_max_loss_bps": delta_loss,
        "tail_improvement_score": float(score),
        **metadata,
    }
    return row


def _screen_single_conditions(
    *,
    policy_id: str,
    trades: pd.DataFrame,
    trade_x: np.ndarray,
    names: list[str],
    quantiles_low: list[float],
    quantiles_high: list[float],
    min_skip_rate: float,
    max_skip_rate: float,
    min_kept_trades: int,
    tail_loss_threshold_bps: float,
    mae_tail_threshold_bps: float,
) -> tuple[pd.DataFrame, list[ConditionCandidate]]:
    pnl = pd.to_numeric(trades["net_pnl_bps"], errors="coerce").to_numpy(np.float64)
    mae = pd.to_numeric(trades["mae_bps"], errors="coerce").to_numpy(np.float64)
    tail_mask = pnl <= float(tail_loss_threshold_bps)
    mae_tail_mask = mae >= float(mae_tail_threshold_bps)
    base = _metrics(pnl)
    rows: list[dict[str, Any]] = []
    candidates: list[ConditionCandidate] = []
    n = int(len(trades))
    for i, feature in enumerate(names):
        vals = trade_x[:, i].astype(np.float64, copy=False)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.nanstd(vals)) <= 1e-12:
            continue
        for side, qs in (("low", quantiles_low), ("high", quantiles_high)):
            for q in qs:
                threshold = float(np.nanquantile(vals, float(q)))
                skip = _condition_mask(vals, side, threshold)
                skipped = int(skip.sum())
                if skipped <= 0:
                    continue
                skip_rate = skipped / max(n, 1)
                if skip_rate < min_skip_rate or skip_rate > max_skip_rate:
                    continue
                if n - skipped < min_kept_trades:
                    continue
                rule = _rule_text(feature, side, float(q))
                metadata = {
                    "feature": feature,
                    "family": _feature_family(feature),
                    "side": side,
                    "quantile": float(q),
                    "threshold": threshold,
                }
                row = _candidate_row(
                    policy_id=policy_id,
                    kind="single",
                    entry_veto_rule=rule,
                    base=base,
                    trades=trades,
                    skip=skip,
                    tail_mask=tail_mask,
                    mae_tail_mask=mae_tail_mask,
                    metadata=metadata,
                )
                if row is None:
                    continue
                rows.append(row)
                candidates.append(
                    ConditionCandidate(
                        feature=feature,
                        family=_feature_family(feature),
                        side=side,
                        quantile=float(q),
                        threshold=threshold,
                        mask=skip,
                        row=row,
                    )
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["tail_improvement_score", "skipped_tail_precision", "kept_net_sum_bps"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out, candidates


def _screen_pairwise_conditions(
    *,
    policy_id: str,
    trades: pd.DataFrame,
    conditions: list[ConditionCandidate],
    max_conditions_for_pairs: int,
    min_skip_rate: float,
    max_skip_rate: float,
    min_kept_trades: int,
    tail_loss_threshold_bps: float,
    mae_tail_threshold_bps: float,
) -> pd.DataFrame:
    pnl = pd.to_numeric(trades["net_pnl_bps"], errors="coerce").to_numpy(np.float64)
    mae = pd.to_numeric(trades["mae_bps"], errors="coerce").to_numpy(np.float64)
    tail_mask = pnl <= float(tail_loss_threshold_bps)
    mae_tail_mask = mae >= float(mae_tail_threshold_bps)
    base = _metrics(pnl)
    ranked = sorted(
        conditions,
        key=lambda c: (
            float(c.row.get("tail_improvement_score") or 0.0),
            float(c.row.get("skipped_tail_precision") or 0.0),
            float(c.row.get("kept_net_sum_bps") or 0.0),
        ),
        reverse=True,
    )[: int(max_conditions_for_pairs)]
    rows: list[dict[str, Any]] = []
    n = int(len(trades))
    seen_rules: set[str] = set()
    for i, left in enumerate(ranked):
        for right in ranked[i + 1 :]:
            if left.feature == right.feature:
                continue
            skip = left.mask & right.mask
            skipped = int(skip.sum())
            if skipped <= 0:
                continue
            skip_rate = skipped / max(n, 1)
            if skip_rate < min_skip_rate or skip_rate > max_skip_rate:
                continue
            if n - skipped < min_kept_trades:
                continue
            left_rule = _rule_text(left.feature, left.side, left.quantile)
            right_rule = _rule_text(right.feature, right.side, right.quantile)
            ordered = sorted([left_rule, right_rule])
            rule = "&".join(ordered)
            if rule in seen_rules:
                continue
            seen_rules.add(rule)
            metadata = {
                "feature_a": left.feature,
                "family_a": left.family,
                "side_a": left.side,
                "quantile_a": float(left.quantile),
                "threshold_a": float(left.threshold),
                "feature_b": right.feature,
                "family_b": right.family,
                "side_b": right.side,
                "quantile_b": float(right.quantile),
                "threshold_b": float(right.threshold),
            }
            row = _candidate_row(
                policy_id=policy_id,
                kind="pairwise",
                entry_veto_rule=rule,
                base=base,
                trades=trades,
                skip=skip,
                tail_mask=tail_mask,
                mae_tail_mask=mae_tail_mask,
                metadata=metadata,
            )
            if row is not None:
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["tail_improvement_score", "skipped_tail_precision", "kept_net_sum_bps"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


def _screen_triple_conditions(
    *,
    policy_id: str,
    trades: pd.DataFrame,
    conditions: list[ConditionCandidate],
    max_conditions_for_triples: int,
    min_skip_rate: float,
    max_skip_rate: float,
    min_kept_trades: int,
    tail_loss_threshold_bps: float,
    mae_tail_threshold_bps: float,
) -> pd.DataFrame:
    pnl = pd.to_numeric(trades["net_pnl_bps"], errors="coerce").to_numpy(np.float64)
    mae = pd.to_numeric(trades["mae_bps"], errors="coerce").to_numpy(np.float64)
    tail_mask = pnl <= float(tail_loss_threshold_bps)
    mae_tail_mask = mae >= float(mae_tail_threshold_bps)
    base = _metrics(pnl)
    ranked = sorted(
        conditions,
        key=lambda c: (
            float(c.row.get("tail_improvement_score") or 0.0),
            float(c.row.get("skipped_tail_precision") or 0.0),
            float(c.row.get("kept_net_sum_bps") or 0.0),
        ),
        reverse=True,
    )[: int(max_conditions_for_triples)]
    rows: list[dict[str, Any]] = []
    n = int(len(trades))
    seen_rules: set[str] = set()
    for i, left in enumerate(ranked):
        for j, middle in enumerate(ranked[i + 1 :], start=i + 1):
            if left.feature == middle.feature:
                continue
            pair_skip = left.mask & middle.mask
            if int(pair_skip.sum()) <= 0:
                continue
            for right in ranked[j + 1 :]:
                if right.feature in {left.feature, middle.feature}:
                    continue
                skip = pair_skip & right.mask
                skipped = int(skip.sum())
                if skipped <= 0:
                    continue
                skip_rate = skipped / max(n, 1)
                if skip_rate < min_skip_rate or skip_rate > max_skip_rate:
                    continue
                if n - skipped < min_kept_trades:
                    continue
                raw_rules = [
                    _rule_text(left.feature, left.side, left.quantile),
                    _rule_text(middle.feature, middle.side, middle.quantile),
                    _rule_text(right.feature, right.side, right.quantile),
                ]
                ordered = sorted(raw_rules)
                rule = "&".join(ordered)
                if rule in seen_rules:
                    continue
                seen_rules.add(rule)
                metadata = {
                    "feature_a": left.feature,
                    "family_a": left.family,
                    "side_a": left.side,
                    "quantile_a": float(left.quantile),
                    "threshold_a": float(left.threshold),
                    "feature_b": middle.feature,
                    "family_b": middle.family,
                    "side_b": middle.side,
                    "quantile_b": float(middle.quantile),
                    "threshold_b": float(middle.threshold),
                    "feature_c": right.feature,
                    "family_c": right.family,
                    "side_c": right.side,
                    "quantile_c": float(right.quantile),
                    "threshold_c": float(right.threshold),
                }
                row = _candidate_row(
                    policy_id=policy_id,
                    kind="triple",
                    entry_veto_rule=rule,
                    base=base,
                    trades=trades,
                    skip=skip,
                    tail_mask=tail_mask,
                    mae_tail_mask=mae_tail_mask,
                    metadata=metadata,
                )
                if row is not None:
                    rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["tail_improvement_score", "skipped_tail_precision", "kept_net_sum_bps"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out


def _tail_group_patterns(trades: pd.DataFrame, policy_id: str, tail_loss_threshold_bps: float) -> pd.DataFrame:
    frame = trades.copy()
    frame["is_tail_loss"] = pd.to_numeric(frame["net_pnl_bps"], errors="coerce") <= float(tail_loss_threshold_bps)
    rows: list[dict[str, Any]] = []
    for group_cols in [
        ["fold"],
        ["entry_month"],
        ["session"],
        ["side"],
        ["session", "side"],
        ["fold", "session"],
        ["entry_month", "session"],
    ]:
        for key, g in frame.groupby(group_cols, dropna=False):
            pnl = pd.to_numeric(g["net_pnl_bps"], errors="coerce")
            rows.append(
                {
                    "policy_id": policy_id,
                    "group_by": "+".join(group_cols),
                    "group": str(key),
                    "n_trades": int(len(g)),
                    "tail_loss_count": int(g["is_tail_loss"].sum()),
                    "tail_loss_rate": float(g["is_tail_loss"].mean()),
                    "net_sum_bps": float(pnl.sum()),
                    "net_mean_bps": float(pnl.mean()),
                    "max_loss_bps": float(pnl.min()),
                    "max_drawdown_bps": _max_drawdown(pnl),
                }
            )
    return pd.DataFrame(rows)


def _select_policy_ids(leader: pd.DataFrame, explicit: list[str], n: int) -> list[str]:
    if explicit:
        return explicit
    frame = leader[
        (leader["scope"] == "all")
        & (leader["n_trades"] >= 80)
        & (leader["positive_months"] >= 6)
        & (leader["net_sum_bps"] > 0)
    ].copy()
    if frame.empty:
        frame = leader[leader["scope"] == "all"].copy()
    frame = frame.sort_values(["risk_adjusted_score", "return_to_dd", "net_sum_bps"], ascending=[False, False, False])
    return [str(x) for x in frame["policy_id"].head(n).tolist()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    replay_dir = Path(args.replay_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = _parse_csv(args.data_splits)
    x, feature_df, names = _build_all_features(
        dataset_dir,
        source_parquet,
        splits,
        include_price_ema_features=bool(args.include_price_ema_features),
    )
    _ = _name_index(names)

    leader = pd.read_csv(replay_dir / "leaderboard.csv")
    trades_all = pd.read_csv(replay_dir / "ablation_policy_trades.csv")
    policy_ids = _select_policy_ids(leader, _parse_csv(args.policy_ids), int(args.max_policies))

    quantiles_low = [float(q) for q in _parse_csv(args.low_quantiles)]
    quantiles_high = [float(q) for q in _parse_csv(args.high_quantiles)]
    single_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    triple_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    policy_summaries: list[dict[str, Any]] = []

    for policy_id in policy_ids:
        trades = trades_all[trades_all["policy_id"] == policy_id].copy()
        if trades.empty:
            continue
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        joined, trade_x = _join_trade_features(trades, feature_df, x)
        if joined.empty:
            continue
        raw_base = _metrics(joined["net_pnl_bps"])
        scoped_joined, scoped_trade_x = _filter_trade_scope(
            joined,
            trade_x,
            sessions=[s.upper() for s in _parse_csv(args.trade_session_filter)],
            sides=[s.upper() for s in _parse_csv(args.trade_side_filter)],
        )
        if scoped_joined.empty:
            continue
        base = _metrics(scoped_joined["net_pnl_bps"])
        policy_summaries.append(
            {
                "policy_id": policy_id,
                "raw_n_trades": raw_base["n_trades"],
                "raw_net_sum_bps": raw_base["net_sum_bps"],
                "raw_max_drawdown_bps": raw_base["max_drawdown_bps"],
                "raw_max_loss_bps": raw_base["max_loss_bps"],
                "scoped_trade_session_filter": args.trade_session_filter,
                "scoped_trade_side_filter": args.trade_side_filter,
                **base,
            }
        )
        single, conditions = _screen_single_conditions(
            policy_id=policy_id,
            trades=scoped_joined,
            trade_x=scoped_trade_x,
            names=names,
            quantiles_low=quantiles_low,
            quantiles_high=quantiles_high,
            min_skip_rate=float(args.min_skip_rate),
            max_skip_rate=float(args.max_skip_rate),
            min_kept_trades=int(args.min_kept_trades),
            tail_loss_threshold_bps=float(args.tail_loss_threshold_bps),
            mae_tail_threshold_bps=float(args.mae_tail_threshold_bps),
        )
        single_frames.append(single)
        pair = _screen_pairwise_conditions(
            policy_id=policy_id,
            trades=scoped_joined,
            conditions=conditions,
            max_conditions_for_pairs=int(args.max_conditions_for_pairs),
            min_skip_rate=float(args.min_pair_skip_rate),
            max_skip_rate=float(args.max_pair_skip_rate),
            min_kept_trades=int(args.min_kept_trades),
            tail_loss_threshold_bps=float(args.tail_loss_threshold_bps),
            mae_tail_threshold_bps=float(args.mae_tail_threshold_bps),
        )
        pair_frames.append(pair)
        if int(args.max_conditions_for_triples) > 0:
            triple = _screen_triple_conditions(
                policy_id=policy_id,
                trades=scoped_joined,
                conditions=conditions,
                max_conditions_for_triples=int(args.max_conditions_for_triples),
                min_skip_rate=float(args.min_triple_skip_rate),
                max_skip_rate=float(args.max_triple_skip_rate),
                min_kept_trades=int(args.min_kept_trades),
                tail_loss_threshold_bps=float(args.tail_loss_threshold_bps),
                mae_tail_threshold_bps=float(args.mae_tail_threshold_bps),
            )
            triple_frames.append(triple)
        group_frames.append(_tail_group_patterns(scoped_joined, policy_id, float(args.tail_loss_threshold_bps)))

    feature_inventory = pd.DataFrame(
        {
            "feature_index": np.arange(len(names)),
            "feature": names,
            "family": [_feature_family(name) for name in names],
            "is_generated": [name.startswith("chart.") for name in names],
        }
    )
    single_out = pd.concat(single_frames, ignore_index=True) if single_frames else pd.DataFrame()
    pair_out = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    triple_out = pd.concat(triple_frames, ignore_index=True) if triple_frames else pd.DataFrame()
    groups_out = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()

    feature_inventory_path = out_dir / "feature_inventory.csv"
    single_path = out_dir / "single_feature_condition_diagnostics.csv"
    pair_path = out_dir / "pairwise_feature_condition_diagnostics.csv"
    triple_path = out_dir / "triple_feature_condition_diagnostics.csv"
    groups_path = out_dir / "tail_pattern_groups.csv"
    rules_path = out_dir / "candidate_entry_veto_rule_sets.txt"

    feature_inventory.to_csv(feature_inventory_path, index=False)
    single_out.to_csv(single_path, index=False)
    pair_out.to_csv(pair_path, index=False)
    triple_out.to_csv(triple_path, index=False)
    groups_out.to_csv(groups_path, index=False)

    best_rules = []
    scope_side_values = _parse_csv(args.rule_scope_side) or _parse_csv(args.trade_side_filter)
    scope_session_values = _parse_csv(args.rule_scope_session) or _parse_csv(args.trade_session_filter)
    scope_side = scope_side_values[0].upper() if len(scope_side_values) == 1 else None
    scope_session = scope_session_values[0].upper() if len(scope_session_values) == 1 else None
    for source, frame in [("triple", triple_out), ("pairwise", pair_out), ("single", single_out)]:
        if frame.empty:
            continue
        filtered = frame[
            (frame["kept_n_trades"] >= int(args.min_kept_trades))
            & (frame["delta_net_sum_bps"] >= float(args.min_delta_net_bps))
        ].copy()
        if filtered.empty:
            filtered = frame.copy()
        filtered = filtered.sort_values(
            ["tail_improvement_score", "delta_max_drawdown_bps", "kept_net_sum_bps"],
            ascending=[False, True, False],
        ).head(int(args.summary_top_n))
        for i, row in enumerate(filtered.itertuples(index=False), start=1):
            rule_name = f"{source}_{i:02d}_{_q_label(float(getattr(row, 'skip_rate')))}"
            scoped_rule = _scope_rule_text(
                str(getattr(row, "entry_veto_rule")),
                decision_side=scope_side,
                trade_session=scope_session,
            )
            best_rules.append(f"{rule_name}={scoped_rule}")
    rules_path.write_text("\n".join(best_rules) + ("\n" if best_rules else ""), encoding="utf-8")

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "replay_dir": str(replay_dir),
        "out_dir": str(out_dir),
        "policy_ids": policy_ids,
        "n_features_screened_single": int(len(names)),
        "pairwise_condition_pool_limit": int(args.max_conditions_for_pairs),
        "triple_condition_pool_limit": int(args.max_conditions_for_triples),
        "tail_loss_threshold_bps": float(args.tail_loss_threshold_bps),
        "mae_tail_threshold_bps": float(args.mae_tail_threshold_bps),
        "trade_session_filter": args.trade_session_filter,
        "trade_side_filter": args.trade_side_filter,
        "rule_scope_side": scope_side,
        "rule_scope_session": scope_session,
        "include_price_ema_features": bool(args.include_price_ema_features),
        "policy_summaries": policy_summaries,
        "outputs": {
            "feature_inventory_csv": str(feature_inventory_path),
            "single_feature_condition_diagnostics_csv": str(single_path),
            "pairwise_feature_condition_diagnostics_csv": str(pair_path),
            "triple_feature_condition_diagnostics_csv": str(triple_path),
            "tail_pattern_groups_csv": str(groups_path),
            "candidate_entry_veto_rule_sets_txt": str(rules_path),
        },
        "top_single_conditions": single_out.head(int(args.summary_top_n)).to_dict(orient="records") if not single_out.empty else [],
        "top_pairwise_conditions": pair_out.head(int(args.summary_top_n)).to_dict(orient="records") if not pair_out.empty else [],
        "top_triple_conditions": triple_out.head(int(args.summary_top_n)).to_dict(orient="records") if not triple_out.empty else [],
        "note": "Thresholds are diagnostic/in-sample. Use candidate_entry_veto_rule_sets.txt only via fold-calibrated replay retest.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    ap.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--include-price-ema-features", action="store_true")
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--policy-ids", default="")
    ap.add_argument("--max-policies", type=int, default=4)
    ap.add_argument("--trade-session-filter", default="")
    ap.add_argument("--trade-side-filter", default="")
    ap.add_argument("--rule-scope-side", default="")
    ap.add_argument("--rule-scope-session", default="")
    ap.add_argument("--low-quantiles", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--high-quantiles", default="0.80,0.85,0.90,0.95")
    ap.add_argument("--min-skip-rate", type=float, default=0.02)
    ap.add_argument("--max-skip-rate", type=float, default=0.35)
    ap.add_argument("--min-pair-skip-rate", type=float, default=0.015)
    ap.add_argument("--max-pair-skip-rate", type=float, default=0.20)
    ap.add_argument("--min-triple-skip-rate", type=float, default=0.01)
    ap.add_argument("--max-triple-skip-rate", type=float, default=0.12)
    ap.add_argument("--min-kept-trades", type=int, default=80)
    ap.add_argument("--max-conditions-for-pairs", type=int, default=180)
    ap.add_argument("--max-conditions-for-triples", type=int, default=0)
    ap.add_argument("--tail-loss-threshold-bps", type=float, default=-50.0)
    ap.add_argument("--mae-tail-threshold-bps", type=float, default=80.0)
    ap.add_argument("--min-delta-net-bps", type=float, default=-1200.0)
    ap.add_argument("--summary-top-n", type=int, default=30)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
