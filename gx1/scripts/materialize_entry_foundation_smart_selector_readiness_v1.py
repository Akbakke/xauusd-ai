#!/usr/bin/env python3
"""Materialize foundation-vs-smart Entry selector readiness evidence.

This report-only gate compares validation and test trade logs for the active
foundation IQL policy and the smart seq520 IQL policy. Selector suggestions are
derived from validation only; test slices are diagnostic and must not be used as
selection criteria.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 import _json_default
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_FOUNDATION_DIR = (
    REPORTS_ROOT
    / "entry_iql_student_trade_log_20260628_v1/foundation_seq146_candidate_20260701_iql_calib_tiebreak_v2"
)
DEFAULT_SMART_DIR = (
    REPORTS_ROOT
    / "entry_iql_student_trade_log_20260628_v1/smart_seq520_candidate_stop_tp_mfe_protect_act1_sl45_broad_net_min190_v2"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_foundation_smart_selector_readiness_20260701_v1"
VALIDATION_LOG_NAME = "entry_iql_student_validation_trade_log.csv"
TEST_LOG_NAME = "entry_iql_student_trade_log.csv"
DEFAULT_CUBES = ("ALL", "side", "session", "vol_regime")


def _profit_factor(pnl: pd.Series) -> float | None:
    values = pd.to_numeric(pnl, errors="coerce").dropna()
    gains = float(values[values > 0.0].sum())
    losses = float(values[values < 0.0].sum())
    if losses == 0.0:
        return None
    return float(gains / abs(losses))


def _max_drawdown(pnl: pd.Series) -> float:
    values = pd.to_numeric(pnl, errors="coerce").fillna(0.0).to_numpy(np.float64)
    if values.size == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(values)])
    drawdown = curve - np.maximum.accumulate(curve)
    return abs(float(np.min(drawdown)))


def _read_trade_log(path: Path, *, model: str, split: str, failures: list[str]) -> pd.DataFrame:
    if not path.is_file():
        failures.append(f"missing {model} {split} trade log: {path}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"entry_time", "net_pnl_bps", "side", "session", "vol_regime", "policy_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        failures.append(f"{model} {split} trade log missing columns: {missing}")
        return pd.DataFrame()
    if "student_trade_log_split" in frame.columns:
        observed = set(frame["student_trade_log_split"].astype(str).unique())
        expected = {"validation" if split == "validation" else "test"}
        if observed != expected:
            failures.append(f"{model} {split} trade log split mismatch: {sorted(observed)} != {sorted(expected)}")
    if "diagnostic_only_not_replay_evidence" in frame.columns:
        flags = set(frame["diagnostic_only_not_replay_evidence"].astype(bool).unique())
        expected_flags = {split == "validation"}
        if flags != expected_flags:
            failures.append(f"{model} {split} diagnostic flag mismatch: {sorted(flags)} != {sorted(expected_flags)}")
    frame = frame.copy()
    frame["model"] = model
    frame["split"] = split
    frame["net_pnl_bps"] = pd.to_numeric(frame["net_pnl_bps"], errors="coerce").fillna(0.0)
    frame["side"] = frame["side"].astype(str)
    frame["session"] = frame["session"].astype(str)
    frame["vol_regime"] = frame["vol_regime"].astype(str)
    return frame


def _metric_row(frame: pd.DataFrame, *, split: str, model: str, cube: str, slice_name: str) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["net_pnl_bps"], errors="coerce").fillna(0.0)
    mae = pd.to_numeric(frame["mae_bps"], errors="coerce") if "mae_bps" in frame.columns else pd.Series(dtype=float)
    return {
        "split": split,
        "model": model,
        "cube": cube,
        "slice": slice_name,
        "n_trades": int(len(frame)),
        "net_sum_bps": float(pnl.sum()),
        "net_mean_bps": float(pnl.mean()) if len(pnl) else None,
        "profit_factor": _profit_factor(pnl),
        "max_drawdown_bps": _max_drawdown(pnl),
        "max_loss_bps": float(pnl.min()) if len(pnl) else None,
        "p90_mae_bps": float(mae.quantile(0.90)) if len(mae.dropna()) else None,
    }


def _slice_metrics(trades: pd.DataFrame, cubes: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, model), base in trades.groupby(["split", "model"], dropna=False):
        for cube in cubes:
            if cube == "ALL":
                rows.append(_metric_row(base, split=str(split), model=str(model), cube="ALL", slice_name="ALL"))
                continue
            if cube not in base.columns:
                continue
            for slice_name, part in base.groupby(cube, dropna=False):
                rows.append(_metric_row(part, split=str(split), model=str(model), cube=cube, slice_name=str(slice_name)))
    return pd.DataFrame(rows)


def _selector_rows(metrics: pd.DataFrame, *, min_slice_trades: int, min_net_lift_bps: float) -> list[dict[str, Any]]:
    validation = metrics[metrics["split"] == "validation"].copy()
    test = metrics[metrics["split"] == "test"].copy()
    foundation = validation[validation["model"] == "foundation"].add_prefix("foundation_")
    smart = validation[validation["model"] == "smart"].add_prefix("smart_")
    joined = foundation.merge(
        smart,
        left_on=["foundation_cube", "foundation_slice"],
        right_on=["smart_cube", "smart_slice"],
        how="outer",
    )
    out: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        cube = str(row.get("foundation_cube") or row.get("smart_cube") or "")
        slice_name = str(row.get("foundation_slice") or row.get("smart_slice") or "")
        f_n = int(row.get("foundation_n_trades") or 0)
        s_n = int(row.get("smart_n_trades") or 0)
        supported = f_n >= min_slice_trades and s_n >= min_slice_trades
        f_net = float(row.get("foundation_net_sum_bps") or 0.0)
        s_net = float(row.get("smart_net_sum_bps") or 0.0)
        validation_lift = s_net - f_net
        selected = "smart" if supported and validation_lift > float(min_net_lift_bps) else "foundation"
        test_part = test[(test["cube"] == cube) & (test["slice"] == slice_name)].copy()
        test_summary = {
            str(r["model"]): {
                "n_trades": int(r["n_trades"]),
                "net_sum_bps": float(r["net_sum_bps"]),
                "profit_factor": r["profit_factor"],
                "max_drawdown_bps": float(r["max_drawdown_bps"]),
            }
            for _, r in test_part.iterrows()
        }
        out.append(
            {
                "cube": cube,
                "slice": slice_name,
                "supported_by_validation": bool(supported),
                "selected_by_validation_only": selected,
                "validation_net_lift_smart_minus_foundation_bps": validation_lift,
                "foundation_validation": {
                    "n_trades": f_n,
                    "net_sum_bps": f_net,
                    "profit_factor": row.get("foundation_profit_factor"),
                    "max_drawdown_bps": row.get("foundation_max_drawdown_bps"),
                },
                "smart_validation": {
                    "n_trades": s_n,
                    "net_sum_bps": s_net,
                    "profit_factor": row.get("smart_profit_factor"),
                    "max_drawdown_bps": row.get("smart_max_drawdown_bps"),
                },
                "test_diagnostic_only_not_selection_criterion": True,
                "test_diagnostic": test_summary,
            }
        )
    return sorted(out, key=lambda item: (item["cube"], item["slice"]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    foundation_dir = Path(args.foundation_dir).expanduser().resolve()
    smart_dir = Path(args.smart_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        _read_trade_log(foundation_dir / VALIDATION_LOG_NAME, model="foundation", split="validation", failures=failures),
        _read_trade_log(foundation_dir / TEST_LOG_NAME, model="foundation", split="test", failures=failures),
        _read_trade_log(smart_dir / VALIDATION_LOG_NAME, model="smart", split="validation", failures=failures),
        _read_trade_log(smart_dir / TEST_LOG_NAME, model="smart", split="test", failures=failures),
    ]
    trades = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if trades.empty:
        failures.append("no selector input trades were loaded")
        metrics = pd.DataFrame()
    else:
        metrics = _slice_metrics(trades, tuple(str(c) for c in args.cubes.split(",") if str(c).strip()))

    selector_rows = _selector_rows(
        metrics,
        min_slice_trades=int(args.min_validation_slice_trades),
        min_net_lift_bps=float(args.min_validation_net_lift_bps),
    ) if not metrics.empty else []
    supported_rows = [row for row in selector_rows if row["supported_by_validation"]]
    smart_supported_wins = [row for row in supported_rows if row["selected_by_validation_only"] == "smart"]
    aggregate = next((row for row in selector_rows if row["cube"] == "ALL" and row["slice"] == "ALL"), {})
    if not aggregate:
        failures.append("missing aggregate foundation-vs-smart selector row")

    metrics_csv = out_dir / "entry_foundation_smart_selector_metrics.csv"
    selector_json = out_dir / "entry_foundation_smart_selector_candidates.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = out_dir / f"ENTRY_FOUNDATION_SMART_SELECTOR_READINESS_{timestamp}.json"
    report_md = out_dir / f"ENTRY_FOUNDATION_SMART_SELECTOR_READINESS_{timestamp}.md"

    metrics.to_csv(metrics_csv, index=False)
    selector_json.write_text(json.dumps(selector_rows, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report = {
        "schema_version": "entry_foundation_smart_selector_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "ENTRY_FOUNDATION_SMART_SELECTOR_READINESS_READY_FOR_REVIEW" if not failures else "FAIL",
        "foundation_dir": str(foundation_dir),
        "smart_dir": str(smart_dir),
        "out_dir": str(out_dir),
        "metrics_csv": str(metrics_csv),
        "selector_candidates_json": str(selector_json),
        "selector_uses_validation_only": True,
        "test_diagnostic_only_not_selection_criterion": True,
        "min_validation_slice_trades": int(args.min_validation_slice_trades),
        "min_validation_net_lift_bps": float(args.min_validation_net_lift_bps),
        "selector_training_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "aggregate_selector_row": aggregate,
        "supported_selector_slice_count": int(len(supported_rows)),
        "smart_supported_validation_win_count": int(len(smart_supported_wins)),
        "smart_supported_validation_wins": smart_supported_wins,
        "failures": failures,
        "json_path": str(report_json),
        "md_path": str(report_md),
    }
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# Entry Foundation Smart Selector Readiness",
                "",
                f"- Decision: `{report['decision']}`",
                f"- Supported selector slices: `{report['supported_selector_slice_count']}`",
                f"- Smart validation wins: `{report['smart_supported_validation_win_count']}`",
                f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_FOUNDATION_SMART_SELECTOR_READINESS_latest.json").write_text(
        report_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_FOUNDATION_SMART_SELECTOR_READINESS_latest.md").write_text(
        report_md.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--foundation-dir", default=str(DEFAULT_FOUNDATION_DIR))
    ap.add_argument("--smart-dir", default=str(DEFAULT_SMART_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cubes", default=",".join(DEFAULT_CUBES))
    ap.add_argument("--min-validation-slice-trades", type=int, default=20)
    ap.add_argument("--min-validation-net-lift-bps", type=float, default=0.0)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
