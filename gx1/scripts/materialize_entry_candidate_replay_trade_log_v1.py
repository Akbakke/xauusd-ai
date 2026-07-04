#!/usr/bin/env python3
"""Materialize an explicit Entry candidate replay trade log from selective-edge predictions.

This is an offline evidence writer. It does not train, promote, shadow, live, or
choose implicit latest bundles. It converts candidate predictions into an
explicit 2026 trade log with bid/ask fixed-horizon PnL, MFE, MAE, and held bars;
the output is intended for `materialize_entry_candidate_replay_evidence_v1`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.apply_entry_foundation_activation_v1 import DEFAULT_SOURCE_PARQUET
from gx1.scripts.replay_entry_tabular_no_xgb_policy_v1 import (
    EXIT_MODE_CHOICES,
    SourceTape,
    _exit_policy_config_from_args,
    _exit_policy_config_hash,
    _exit_policy_contract_from_args,
    _policy_hash,
    _threshold_from_scores,
    _validate_exit_policy_args,
)
from gx1.scripts.verify_entry_foundation_state_v1 import FOUNDATION_DATASET_DIR, REPORTS_ROOT


DEFAULT_PREDICTIONS = REPORTS_ROOT / "entry_candidate_selective_edge_20260628_v1/selective_edge_predictions.parquet"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_candidate_replay_trade_log_20260628_v1"
SIDE_NAMES = {0: "LONG", 1: "SHORT", 2: "FLAT"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise RuntimeError("expected at least one float value")
    return values


def _split_file(dataset_dir: Path, split: str) -> Path:
    matches = sorted(dataset_dir.glob(f"*_{split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {split} parquet under {dataset_dir}, got {matches}")
    return matches[0]


def _load_horizons(dataset_dir: Path, splits: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = _split_file(dataset_dir, split)
        frame = pd.read_parquet(path, columns=["time", "label_horizon_bars"])
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        if frame["time"].isna().any():
            raise RuntimeError(f"{split} dataset has unparsable time rows: {path}")
        frame["split"] = str(split)
        frame["label_horizon_bars"] = pd.to_numeric(frame["label_horizon_bars"], errors="coerce")
        frames.append(frame[["split", "time", "label_horizon_bars"]])
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["split", "time"]).any():
        dupes = int(out.duplicated(["split", "time"]).sum())
        raise RuntimeError(f"dataset horizon join keys are not unique: duplicate split/time rows={dupes}")
    return out


def _prepare_predictions(predictions_path: Path, dataset_dir: Path, model_name: str) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing selective-edge predictions: {predictions_path}")
    required = {
        "split",
        "model",
        "time",
        "y_direction",
        "trade_side",
        "session",
        "edge_score",
        "p_long",
        "p_short",
        "p_flat",
    }
    predictions = pd.read_parquet(predictions_path)
    missing = sorted(required - set(str(c) for c in predictions.columns))
    if missing:
        raise RuntimeError(f"selective-edge predictions missing columns: {missing}")
    predictions = predictions[predictions["model"].astype(str) == str(model_name)].copy()
    if predictions.empty:
        raise RuntimeError(f"selective-edge predictions contain no rows for model={model_name!r}")
    predictions["time"] = pd.to_datetime(predictions["time"], utc=True, errors="coerce")
    if predictions["time"].isna().any():
        raise RuntimeError("selective-edge predictions have unparsable time rows")
    predictions["split"] = predictions["split"].astype(str)
    horizons = _load_horizons(dataset_dir, sorted(str(x) for x in predictions["split"].dropna().unique()))
    merged = predictions.merge(horizons, on=["split", "time"], how="left", validate="one_to_one")
    if merged["label_horizon_bars"].isna().any():
        missing_rows = int(merged["label_horizon_bars"].isna().sum())
        raise RuntimeError(f"failed to attach label_horizon_bars to selective-edge predictions: rows={missing_rows}")
    merged["label_horizon_bars"] = merged["label_horizon_bars"].astype(np.int64)
    if bool((merged["label_horizon_bars"] <= 0).any()):
        raise RuntimeError("label_horizon_bars must be positive for all replay rows")
    return merged.sort_values(["split", "time"], kind="mergesort").reset_index(drop=True)


def _frac_label(frac: float) -> str:
    pct = float(frac) * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"top{int(round(pct)):02d}"
    return "top" + str(frac).replace(".", "p")


def _cost_label(cost: float) -> str:
    if abs(float(cost) - round(float(cost))) < 1e-9:
        return f"cost{int(round(float(cost))):02d}"
    return "cost" + str(cost).replace(".", "p")


def _run_fixed_horizon_policy(
    *,
    eval_df: pd.DataFrame,
    tape: SourceTape,
    score_threshold: float,
    threshold_top_frac: float,
    cost_stress_bps: float,
    args: argparse.Namespace,
    policy_id: str,
    policy_config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_idx = tape.indices_for_times(eval_df["time"])
    probs = eval_df[["p_long", "p_short", "p_flat"]].astype(float).to_numpy(np.float64)
    chosen_prob = np.maximum(probs[:, 0], probs[:, 1])
    scores = pd.to_numeric(eval_df["edge_score"], errors="coerce").to_numpy(np.float64)
    sides = pd.to_numeric(eval_df["trade_side"], errors="coerce").to_numpy(np.int64)
    labels = pd.to_numeric(eval_df["y_direction"], errors="coerce").to_numpy(np.int64)
    counts: dict[str, Any] = {
        "policy_id": policy_id,
        "threshold_top_frac": float(threshold_top_frac),
        "score_threshold": float(score_threshold),
        "cost_stress_bps": float(cost_stress_bps),
        "exit_mode": str(args.exit_mode),
        "exit_policy_config_hash": _exit_policy_config_hash(_exit_policy_config_from_args(args)),
        "rows": int(len(eval_df)),
        "below_threshold": 0,
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
    for i, row in enumerate(eval_df.itertuples(index=False)):
        row_score = float(scores[i])
        if not np.isfinite(row_score) or row_score < float(score_threshold):
            counts["below_threshold"] += 1
            continue
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
        if float(args.daily_loss_limit_bps) > 0.0 and daily_pnl_bps.get(day, 0.0) <= -float(args.daily_loss_limit_bps):
            counts["skipped_daily_loss_limit"] += 1
            continue
        sim = tape.simulate_trade(
            start_idx=start_src_idx,
            horizon_bars=int(row.label_horizon_bars),
            side=int(sides[i]),
            exit_mode=str(args.exit_mode),
            take_profit_bps=float(args.take_profit_bps),
            stop_loss_bps=float(args.stop_loss_bps),
            same_bar_policy=str(args.same_bar_policy),
            mfe_protect_activation_bps=float(getattr(args, "mfe_protect_activation_bps", 20.0)),
            mfe_protect_breakeven_offset_bps=float(getattr(args, "mfe_protect_breakeven_offset_bps", 0.0)),
            mfe_protect_trailing_capture_ratio=float(getattr(args, "mfe_protect_trailing_capture_ratio", 0.0)),
            mfe_protect_trailing_floor_bps=float(getattr(args, "mfe_protect_trailing_floor_bps", 0.0)),
        )
        if sim is None:
            counts["skipped_invalid_path"] += 1
            continue
        net_pnl_bps = (float(sim["gross_pnl_bps"]) - float(cost_stress_bps) - float(args.slippage_bps)) * float(
            args.size_multiplier
        )
        trade = {
            "fold": "2026_TEST",
            "policy_id": policy_id,
            "policy_config_hash": policy_config_hash,
            "exit_mode": str(args.exit_mode),
            "exit_policy_config_hash": _exit_policy_config_hash(_exit_policy_config_from_args(args)),
            "threshold_source": "validation_top_fraction",
            "threshold_top_frac": float(threshold_top_frac),
            "score_threshold": float(score_threshold),
            "cost_stress_bps": float(cost_stress_bps),
            "slippage_bps": float(args.slippage_bps),
            "size_multiplier": float(args.size_multiplier),
            "source_split": str(row.split),
            "session": str(row.session).upper(),
            "vol_regime": str(getattr(row, "vol_regime", "UNKNOWN")),
            "entry_day": day,
            "entry_month": entry_time.strftime("%Y-%m"),
            "entry_time": sim["entry_time"],
            "exit_time": sim["exit_time"],
            "side": SIDE_NAMES.get(int(sides[i]), str(int(sides[i]))),
            "label": SIDE_NAMES.get(int(labels[i]), str(int(labels[i]))),
            "direction_correct": bool(int(labels[i]) == int(sides[i])),
            "score": row_score,
            "chosen_prob": float(chosen_prob[i]),
            "p_long": float(probs[i, 0]),
            "p_short": float(probs[i, 1]),
            "p_flat": float(probs[i, 2]),
            "path_quality_pred": float(getattr(row, "path_quality_pred", np.nan)),
            "bad_path_prob": float(getattr(row, "bad_path_prob", np.nan)),
            "entry_price": sim["entry_price"],
            "exit_price": sim["exit_price"],
            "gross_pnl_bps": sim["gross_pnl_bps"],
            "net_pnl_bps": net_pnl_bps,
            "mfe_bps": sim["mfe_bps"],
            "mae_bps": sim["mae_bps"],
            "horizon_bars": int(row.label_horizon_bars),
            "held_bars": sim["held_bars"],
            "exit_reason": sim["exit_reason"],
            "mfe_protect_activated": sim["mfe_protect_activated"],
            "mfe_protect_activation_bar": sim["mfe_protect_activation_bar"],
            "mfe_protect_floor_bps_at_exit": sim["mfe_protect_floor_bps_at_exit"],
            "mfe_protect_peak_mfe_bps_at_exit": sim["mfe_protect_peak_mfe_bps_at_exit"],
        }
        trades.append(trade)
        counts["trades"] += 1
        daily_trade_count[day] = daily_trade_count.get(day, 0) + 1
        daily_pnl_bps[day] = daily_pnl_bps.get(day, 0.0) + float(net_pnl_bps)
        unavailable_until_src_idx = int(sim["exit_src_idx"]) + int(args.cooldown_bars)
    return trades, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = Path(args.selective_edge_predictions).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_exit_policy_args(args)

    predictions = _prepare_predictions(predictions_path, dataset_dir, str(args.model_name))
    if "val" not in set(predictions["split"]) or "test" not in set(predictions["split"]):
        raise RuntimeError("candidate replay trade log requires val and test predictions")
    val = predictions[predictions["split"].astype(str) == "val"].reset_index(drop=True)
    test = predictions[predictions["split"].astype(str) == "test"].sort_values("time", kind="mergesort").reset_index(drop=True)
    if not bool((test["time"].dt.year == 2026).all()):
        if not bool(getattr(args, "allow_non_2026_test", False)):
            raise RuntimeError("candidate replay trade log test split must be entirely 2026")
        # Exit-training SUBSTRATE mode (exit-parity wave 2026-07-04): historical
        # coverage is the purpose (goal-doc step 7 — Entry-bound per-bar state
        # from replay trades across history). This log is NOT replay evidence;
        # the replay-evidence gate enforces its own 2026-only guard and will
        # reject historical logs, so the two purposes cannot be confused.
        print("[TRADE_LOG] allow-non-2026-test: exit-training substrate mode (not replay evidence)", flush=True)
    top_fracs = _parse_float_list(str(args.threshold_top_fracs))
    cost_values = _parse_float_list(str(args.cost_stress_bps))
    tape = SourceTape.load(source_parquet)

    base_policy_config = {
        "model_name": str(args.model_name),
        "threshold_source": "validation_top_fraction",
        "eval_split": "test",
        **_exit_policy_config_from_args(args),
        "cooldown_bars": int(args.cooldown_bars),
        "max_trades_per_day": int(args.max_trades_per_day),
        "daily_loss_limit_bps": float(args.daily_loss_limit_bps),
        "min_direction_prob": float(args.min_direction_prob),
        "min_score_floor": float(args.min_score_floor),
        "slippage_bps": float(args.slippage_bps),
        "size_multiplier": float(args.size_multiplier),
    }
    threshold_rows: list[dict[str, Any]] = []
    all_counts: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    val_scores = pd.to_numeric(val["edge_score"], errors="coerce").to_numpy(np.float64)
    for top_frac in top_fracs:
        threshold = _threshold_from_scores(val_scores, float(top_frac), float(args.min_score_floor))
        threshold_rows.append(
            {
                "model": str(args.model_name),
                "threshold_top_frac": float(top_frac),
                "score_threshold": float(threshold),
                "val_rows": int(len(val)),
                "test_rows": int(len(test)),
                "val_score_mean": float(np.nanmean(val_scores)),
                "val_score_p50": float(np.nanpercentile(val_scores, 50)),
                "val_score_p90": float(np.nanpercentile(val_scores, 90)),
                "val_score_p95": float(np.nanpercentile(val_scores, 95)),
            }
        )
        for cost_bps in cost_values:
            policy_config = {**base_policy_config, "threshold_top_frac": float(top_frac), "cost_stress_bps": float(cost_bps)}
            config_hash = _policy_hash(policy_config)
            policy_id = str(args.policy_id).strip() or (
                f"candidate_{_frac_label(top_frac)}_valcal_{args.exit_mode}_{_cost_label(cost_bps)}_{config_hash}"
            )
            trades, counts = _run_fixed_horizon_policy(
                eval_df=test,
                tape=tape,
                score_threshold=float(threshold),
                threshold_top_frac=float(top_frac),
                cost_stress_bps=float(cost_bps),
                args=args,
                policy_id=policy_id,
                policy_config_hash=config_hash,
            )
            all_trades.extend(trades)
            all_counts.append(counts)

    trades_df = pd.DataFrame(all_trades)
    failures: list[str] = []
    if trades_df.empty:
        failures.append("candidate replay trade log produced zero trades")
    else:
        required = {
            "entry_time",
            "policy_id",
            "session",
            "side",
            "score",
            "p_long",
            "p_short",
            "p_flat",
            "net_pnl_bps",
            "mfe_bps",
            "mae_bps",
            "held_bars",
        }
        missing = sorted(required - set(trades_df.columns))
        if missing:
            failures.append(f"candidate replay trade log missing required columns: {missing}")
        years = set(pd.to_datetime(trades_df["entry_time"], utc=True).dt.year.astype(int).unique())
        if years != {2026}:
            failures.append(f"candidate replay trade log contains years outside 2026: {sorted(years)}")
        numeric_cols = ["score", "p_long", "p_short", "p_flat", "net_pnl_bps", "mfe_bps", "mae_bps", "held_bars"]
        for col in numeric_cols:
            values = pd.to_numeric(trades_df[col], errors="coerce")
            if values.isna().any() or not bool(np.isfinite(values.to_numpy(np.float64)).all()):
                failures.append(f"candidate replay trade log numeric column is not fully finite: {col}")

    trades_path = out_dir / "candidate_replay_trade_log.csv"
    thresholds_path = out_dir / "candidate_replay_thresholds.csv"
    counts_path = out_dir / "candidate_replay_policy_counts.csv"
    manifest_path = out_dir / "CANDIDATE_REPLAY_TRADE_LOG_MANIFEST.json"
    report_path = out_dir / f"CANDIDATE_REPLAY_TRADE_LOG_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame(threshold_rows).to_csv(thresholds_path, index=False)
    pd.DataFrame(all_counts).to_csv(counts_path, index=False)
    report = {
        "schema_version": "entry_candidate_replay_trade_log_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "selective_edge_predictions": str(predictions_path),
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "out_dir": str(out_dir),
        "trades_path": str(trades_path),
        "thresholds_path": str(thresholds_path),
        "counts_path": str(counts_path),
        "top_fracs": top_fracs,
        "cost_stress_bps": cost_values,
        "n_trades": int(len(trades_df)),
        "policy_counts": all_counts,
        "policy_config": base_policy_config,
        "exit_policy_contract": _exit_policy_contract_from_args(args),
        "trainer_started": False,
        "replay_started": False,
        "promotion_shadow_live_allowed": False,
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (out_dir / "CANDIDATE_REPLAY_TRADE_LOG_latest.json").write_text(
        report_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selective-edge-predictions", default=str(DEFAULT_PREDICTIONS))
    ap.add_argument("--dataset-dir", default=str(FOUNDATION_DATASET_DIR))
    ap.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--model-name", default="candidate")
    ap.add_argument("--threshold-top-fracs", default="0.05")
    ap.add_argument("--cost-stress-bps", default="0.0")
    ap.add_argument("--policy-id", default="")
    ap.add_argument(
        "--allow-non-2026-test",
        action="store_true",
        help=(
            "Exit-training substrate mode: allow historical (non-2026) test rows. "
            "Mirrors replay-evidence's --allow-non-2026. The output is a training "
            "substrate, NOT replay evidence (the evidence gate enforces its own "
            "2026-only guard)."
        ),
    )
    ap.add_argument("--exit-mode", choices=EXIT_MODE_CHOICES, default="horizon")
    ap.add_argument("--take-profit-bps", type=float, default=60.0)
    ap.add_argument("--stop-loss-bps", type=float, default=45.0)
    ap.add_argument("--same-bar-policy", choices=("stop_first", "target_first"), default="stop_first")
    ap.add_argument("--mfe-protect-activation-bps", type=float, default=20.0)
    ap.add_argument("--mfe-protect-breakeven-offset-bps", type=float, default=0.0)
    ap.add_argument("--mfe-protect-trailing-capture-ratio", type=float, default=0.0)
    ap.add_argument("--mfe-protect-trailing-floor-bps", type=float, default=0.0)
    ap.add_argument("--cooldown-bars", type=int, default=6)
    ap.add_argument("--max-trades-per-day", type=int, default=8)
    ap.add_argument("--daily-loss-limit-bps", type=float, default=150.0)
    ap.add_argument("--min-direction-prob", type=float, default=0.0)
    ap.add_argument("--min-score-floor", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--size-multiplier", type=float, default=1.0)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
