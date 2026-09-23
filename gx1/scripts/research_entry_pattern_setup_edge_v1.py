#!/usr/bin/env python3
"""Rule-based confluence setups measured out of sample (research only, no authority).

The operator's discretionary claim ("wait for confluence: FVG / order-block
retest at a higher-timeframe level in trend direction, sweep-and-reject of
equal highs, PDH/PDL and flag breakouts, momentum agreement") is written
here as explicit, fixed decision rules over the tape-based pattern
primitives (``research_entry_pattern_primitives_v1``) and scored with the
pre-registered selective-edge statistics on every year of TRAIN and on the
untouched confirmation month.  No rule is fitted; the evaluator only counts.

Nulls per setup, horizon and period: exact coin flip on the same rows, the
better of always-LONG / always-SHORT on the same rows ("best constant side",
the drift-neutral null), Newey-West HAC standard error of the advantage
over the coin flip, and the owner's circular-shift null.  ``strict_pass``
requires the pre-registered primary pass AND a positive mean;
``beats_best_constant`` is reported separately.

Why a new module (rule 21): the selective-edge owner evaluates a trained
bundle's predictions; the walk-forward instrument fits learners.  Fixed
rules with no fit are a third bounded research authority; both existing
owners' statistical functions are imported, not duplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SHORT_INDEX,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    CIRCULAR_SHIFT_NULL_DRAWS,
    MIN_PREREGISTERED_TRADE_ROWS,
    _circular_shift_null_means,
    _newey_west_mean_se,
)
from gx1.scripts.research_entry_direction_walkforward_v1 import (
    executable_horizon_targets,
    load_tape,
    tape_positions,
)

SCHEMA_VERSION = "entry_pattern_setup_edge_research_v1"
AUTHORITY = {"research_only": True, "candidate": False, "promotion": False, "test": False, "paper": False, "live": False}


@dataclass(frozen=True)
class Setup:
    name: str
    side: str  # "LONG" | "SHORT"
    description: str
    rule: Callable[[pd.DataFrame], np.ndarray]


def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise RuntimeError(f"SETUP_EDGE_PRIMITIVE_MISSING: {name}")
    return frame[name].to_numpy(np.float64)


def _stack(frame: pd.DataFrame, tf: str, sign: int) -> np.ndarray:
    return _col(frame, f"{tf}:ema_stack") == float(sign)


def declared_setups() -> tuple[Setup, ...]:
    S: list[Setup] = []

    def add(name: str, side: str, description: str, rule: Callable[[pd.DataFrame], np.ndarray]) -> None:
        S.append(Setup(name=name, side=side, description=description, rule=rule))

    for tf, ctx in (("M5", "H1"), ("H1", "H4"), ("H4", "D1")):
        add(f"fvg_bull_retest_{tf}_trend_{ctx}", "LONG", f"bullish FVG retest hold on {tf} with {ctx} EMA stack bullish",
            lambda f, tf=tf, ctx=ctx: (_col(f, f"{tf}:fvg_bull_retest_event") == 1.0) & _stack(f, ctx, 1))
        add(f"fvg_bear_retest_{tf}_trend_{ctx}", "SHORT", f"bearish FVG retest hold on {tf} with {ctx} EMA stack bearish",
            lambda f, tf=tf, ctx=ctx: (_col(f, f"{tf}:fvg_bear_retest_event") == 1.0) & _stack(f, ctx, -1))
        add(f"ob_bull_retest_{tf}_trend_{ctx}", "LONG", f"bullish order-block retest hold on {tf} with {ctx} EMA stack bullish",
            lambda f, tf=tf, ctx=ctx: (_col(f, f"{tf}:ob_bull_retest_event") == 1.0) & _stack(f, ctx, 1))
        add(f"ob_bear_retest_{tf}_trend_{ctx}", "SHORT", f"bearish order-block retest hold on {tf} with {ctx} EMA stack bearish",
            lambda f, tf=tf, ctx=ctx: (_col(f, f"{tf}:ob_bear_retest_event") == 1.0) & _stack(f, ctx, -1))
    for tf in ("M5", "H1"):
        add(f"eqh_sweep_fade_{tf}", "SHORT", f"equal-highs sweep and reject on {tf} (fade)", lambda f, tf=tf: _col(f, f"{tf}:eqh_sweep_event") == 1.0)
        add(f"eql_sweep_fade_{tf}", "LONG", f"equal-lows sweep and reject on {tf} (fade)", lambda f, tf=tf: _col(f, f"{tf}:eql_sweep_event") == 1.0)
        add(f"eqh_sweep_fade_{tf}_trend_H4_bear", "SHORT", f"equal-highs sweep on {tf} with H4 stack bearish",
            lambda f, tf=tf: (_col(f, f"{tf}:eqh_sweep_event") == 1.0) & _stack(f, "H4", -1))
        add(f"eql_sweep_fade_{tf}_trend_H4_bull", "LONG", f"equal-lows sweep on {tf} with H4 stack bullish",
            lambda f, tf=tf: (_col(f, f"{tf}:eql_sweep_event") == 1.0) & _stack(f, "H4", 1))
        add(f"bull_flag_breakout_{tf}", "LONG", f"bull flag breakout on {tf}", lambda f, tf=tf: _col(f, f"{tf}:bull_flag_breakout_event") == 1.0)
        add(f"bear_flag_breakout_{tf}", "SHORT", f"bear flag breakout on {tf}", lambda f, tf=tf: _col(f, f"{tf}:bear_flag_breakout_event") == 1.0)
        add(f"range_break_up_{tf}_trend_H4", "LONG", f"N-bar range breakout up on {tf} with H4 stack bullish",
            lambda f, tf=tf: (_col(f, f"{tf}:range_break_up_event") == 1.0) & _stack(f, "H4", 1))
        add(f"range_break_down_{tf}_trend_H4", "SHORT", f"N-bar range breakout down on {tf} with H4 stack bearish",
            lambda f, tf=tf: (_col(f, f"{tf}:range_break_down_event") == 1.0) & _stack(f, "H4", -1))
    add("pdh_break_trend_H4", "LONG", "first close above previous-day high with H4 stack bullish",
        lambda f: (_col(f, "M5:pdh_break_event") == 1.0) & _stack(f, "H4", 1))
    add("pdl_break_trend_H4", "SHORT", "first close below previous-day low with H4 stack bearish",
        lambda f: (_col(f, "M5:pdl_break_event") == 1.0) & _stack(f, "H4", -1))
    add("asia_hi_break_trend_H1", "LONG", "first close above the completed Asia range high with H1 stack bullish",
        lambda f: (_col(f, "M5:asia_hi_break_event") == 1.0) & _stack(f, "H1", 1))
    add("asia_lo_break_trend_H1", "SHORT", "first close below the completed Asia range low with H1 stack bearish",
        lambda f: (_col(f, "M5:asia_lo_break_event") == 1.0) & _stack(f, "H1", -1))
    add("momentum_confluence_long", "LONG", "H1, H4 and D1 EMA stacks all bullish and M5 range breakout up",
        lambda f: _stack(f, "H1", 1) & _stack(f, "H4", 1) & _stack(f, "D1", 1) & (_col(f, "M5:range_break_up_event") == 1.0))
    add("momentum_confluence_short", "SHORT", "H1, H4 and D1 EMA stacks all bearish and M5 range breakout down",
        lambda f: _stack(f, "H1", -1) & _stack(f, "H4", -1) & _stack(f, "D1", -1) & (_col(f, "M5:range_break_down_event") == 1.0))
    add("trend_all_bull_any_bar", "LONG", "H1, H4 and D1 EMA stacks all bullish (every bar; drift reference)",
        lambda f: _stack(f, "H1", 1) & _stack(f, "H4", 1) & _stack(f, "D1", 1))
    add("trend_all_bear_any_bar", "SHORT", "H1, H4 and D1 EMA stacks all bearish (every bar; drift reference)",
        lambda f: _stack(f, "H1", -1) & _stack(f, "H4", -1) & _stack(f, "D1", -1))
    return tuple(S)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_setup(
    *, name: str, side: str, mask: np.ndarray, long_all: np.ndarray, short_all: np.ndarray, valid: np.ndarray, period: str, horizon: int
) -> dict[str, Any]:
    sel = np.flatnonzero(mask & valid)
    row: dict[str, Any] = {"setup": name, "side": side, "period": period, "horizon_bars": horizon, "n": int(len(sel)), "period_rows": int(valid.sum())}
    if len(sel) < MIN_PREREGISTERED_TRADE_ROWS:
        row["failure_reason"] = "insufficient_setup_rows"
        return row
    side_index = MODEL_DIRECTION_LONG_INDEX if side == "LONG" else MODEL_DIRECTION_SHORT_INDEX
    pnl = long_all[sel] if side == "LONG" else short_all[sel]
    coin = 0.5 * (long_all[sel] + short_all[sel])
    advantage = pnl - coin
    se, lag = _newey_west_mean_se(advantage)
    best_constant = float(max(np.mean(long_all[sel]), np.mean(short_all[sel])))
    mean_pnl = float(np.mean(pnl))
    row.update(
        {
            "mean_pnl_bps": mean_pnl,
            "coin_flip_mean_pnl_bps": float(np.mean(coin)),
            "mean_advantage_over_coin_bps": float(np.mean(advantage)),
            "advantage_standard_error_bps": float(se) if se is not None else None,
            "advantage_hac_lag": int(lag),
            "best_constant_side_mean_pnl_bps": best_constant,
            "excess_over_best_constant_bps": mean_pnl - best_constant,
            "beats_best_constant": bool(mean_pnl > best_constant),
            "win_rate": float(np.mean(pnl > 0)),
            "hit_rate_better_side": float(np.mean((long_all[sel] >= short_all[sel]) == (side == "LONG"))),
            "trades_per_period_row_1000": float(1000.0 * len(sel) / max(1, int(valid.sum()))),
        }
    )
    all_positions = np.flatnonzero(valid)
    if len(all_positions) > CIRCULAR_SHIFT_NULL_DRAWS and se is not None:
        # positions relative to the period population (the owner shifts within the period)
        rel = np.searchsorted(all_positions, sel)
        circ, _ = _circular_shift_null_means(
            all_long=long_all[all_positions], all_short=short_all[all_positions],
            selected_positions=rel, selected_sides=np.full(len(sel), side_index, dtype=np.int64),
        )
        p95 = float(np.percentile(circ, 95.0))
        row["circular_shift_p95_mean_pnl_bps"] = p95
        primary = bool(float(np.mean(advantage)) > 2.0 * float(se) and mean_pnl > p95)
        row["primary_pass"] = primary
        row["strict_pass"] = bool(primary and mean_pnl > 0.0)
    else:
        row["failure_reason"] = "circular_null_unavailable"
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise RuntimeError("SETUP_EDGE_OUT_DIR_EXISTS")
    out_dir.mkdir(parents=True)
    prim_path = Path(args.pattern_primitives_parquet)
    prim = pd.read_parquet(prim_path)
    prim["time"] = pd.to_datetime(prim["time"], utc=True)
    time = pd.DatetimeIndex(prim["time"])
    if not time.is_monotonic_increasing or time.has_duplicates:
        raise RuntimeError("SETUP_EDGE_TIME_INVALID")
    dataset_dir = Path(args.dataset_dir)
    manifest = json.loads((dataset_dir / "entry_dataset__ENTRY_FITTED_Q_train.manifest.json").read_text(encoding="utf-8"))
    train_start = pd.Timestamp(manifest["splits"]["train"]["start"])
    val_start = pd.Timestamp(manifest["splits"]["val"]["start"])
    val_end = pd.Timestamp(manifest["splits"]["val"]["end"])
    has_val = bool((time >= val_start).any())
    tape = load_tape(Path(args.native_m5_root), truncate_before=(val_end if has_val else val_start))
    positions = tape_positions(time, tape)
    boundaries = [pd.Timestamp(v) for v in args.period_boundaries]
    edges = [train_start, *boundaries]
    if any(b <= a for a, b in zip(edges, edges[1:])) or edges[-1] > val_start:
        raise RuntimeError("SETUP_EDGE_PERIOD_BOUNDARIES_INVALID")
    periods: list[tuple[str, np.ndarray, pd.Timestamp]] = []
    for a, b in zip(edges, edges[1:]):
        periods.append((f"{a.date()}..{b.date()}", np.asarray((time >= a) & (time < b), dtype=bool), b))
    if has_val:
        periods.append(("final_val", np.asarray((time >= val_start) & (time < val_end), dtype=bool), val_end))
    setups = declared_setups()
    rows: list[dict[str, Any]] = []
    masks = {s.name: np.asarray(s.rule(prim), dtype=bool) for s in setups}
    tape_time = tape.time
    for h in [int(v) for v in args.horizons]:
        long_bps, short_bps, valid_h = executable_horizon_targets(tape, positions, h)
        for period_name, period_mask, period_end in periods:
            # outcome windows must end before the first tape row at/after the period end
            end_position = int(tape_time.searchsorted(period_end, side="left"))
            valid = period_mask & valid_h & ((positions + h) < end_position)
            for s in setups:
                rows.append(score_setup(name=s.name, side=s.side, mask=masks[s.name], long_all=long_bps, short_all=short_bps, valid=valid, period=period_name, horizon=h))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = summarize(metrics, has_val=has_val)
    report = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instrument_source_sha256": _sha256_file(Path(__file__)),
        "inputs": {
            "pattern_primitives_parquet": str(prim_path), "pattern_primitives_sha256": _sha256_file(prim_path),
            "native_m5_root": tape.root, "native_m5_manifest_sha256": tape.manifest_sha256,
            "dataset_dir": str(dataset_dir), "rows": int(len(prim)), "has_val": has_val,
        },
        "config": {"horizons": [int(v) for v in args.horizons], "period_boundaries": [b.isoformat() for b in boundaries], "min_rows": MIN_PREREGISTERED_TRADE_ROWS},
        "setups": [{"name": s.name, "side": s.side, "description": s.description, "fire_rate": float(np.mean(masks[s.name]))} for s in setups],
        "summary": summary,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def summarize(metrics: pd.DataFrame, *, has_val: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if metrics.empty:
        return out
    scored = metrics[metrics["mean_pnl_bps"].notna()] if "mean_pnl_bps" in metrics else metrics.iloc[0:0]
    for (setup, side, h), g in scored.groupby(["setup", "side", "horizon_bars"], sort=True):
        years = g[g["period"] != "final_val"]
        val = g[g["period"] == "final_val"]
        out.append(
            {
                "setup": setup, "side": side, "horizon_bars": int(h),
                "years_scored": int(len(years)), "years_positive": int((years["mean_pnl_bps"] > 0).sum()),
                "years_strict_pass": int(years["strict_pass"].eq(True).sum()),
                "years_beat_best_constant": int(years["beats_best_constant"].eq(True).sum()),
                "year_mean_pnl_avg": float(years["mean_pnl_bps"].mean()) if len(years) else None,
                "year_mean_pnl_min": float(years["mean_pnl_bps"].min()) if len(years) else None,
                "n_per_year_avg": float(years["n"].mean()) if len(years) else None,
                "val_mean_pnl_bps": float(val["mean_pnl_bps"].iloc[0]) if len(val) else None,
                "val_n": int(val["n"].iloc[0]) if len(val) else None,
                "val_strict_pass": bool(val["strict_pass"].iloc[0]) if len(val) and pd.notna(val["strict_pass"].iloc[0]) else None,
                "val_beats_best_constant": bool(val["beats_best_constant"].iloc[0]) if len(val) else None,
            }
        )
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pattern / confluence setup edge (research, no authority)",
        "",
        f"Created {report['created_utc']}. Horizons {report['config']['horizons']} bars. Periods: yearly TRAIN slices + final VAL month when present.",
        "strict_pass = excess over coin flip > 2 HAC SE AND mean > circular-shift p95 AND mean > 0. beats_best_constant = mean > better of always-LONG / always-SHORT on the same rows.",
        "",
        "| setup | side | h | years | pos | strict | beat-const | mean bps | min bps | n/yr | VAL bps | VAL n | VAL strict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in report["summary"]:
        f = lambda v, nd=2: "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))  # noqa: E731
        lines.append(
            f"| {s['setup']} | {s['side']} | {s['horizon_bars']} | {s['years_scored']} | {s['years_positive']} | {s['years_strict_pass']} | {s['years_beat_best_constant']} | "
            f"{f(s['year_mean_pnl_avg'])} | {f(s['year_mean_pnl_min'])} | {f(s['n_per_year_avg'], 0)} | {f(s['val_mean_pnl_bps'])} | {f(s['val_n'])} | {f(s['val_strict_pass'])} |"
        )
    lines.append("")
    lines.append("Fire rates (share of decision rows): " + ", ".join(f"{s['name']} {s['fire_rate']:.4f}" for s in report["setups"]))
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern-primitives-parquet", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--native-m5-root", required=True)
    p.add_argument("--horizons", nargs="+", required=True)
    p.add_argument("--period-boundaries", nargs="+", required=True, help="UTC timestamps closing each yearly TRAIN period; the last must be the TRAIN end")
    p.add_argument("--out-dir", required=True)
    return p


def main() -> int:
    report = run(build_parser().parse_args())
    passing = [s for s in report["summary"] if s["years_strict_pass"] >= 3]
    print(json.dumps({"setups": len(report["setups"]), "cells": len(report["summary"]), "cells_with_3plus_strict_years": len(passing)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
