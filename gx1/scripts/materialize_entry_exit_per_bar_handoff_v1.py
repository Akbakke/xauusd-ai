#!/usr/bin/env python3
"""Materialize active Entry-bound per-bar Exit handoff substrate.

This is a data-substrate materializer only. It reads the active Entry-IQL
student trade log and canonical M5 bid/ask bars, emits per-bar HOLD/EXIT_NOW
state rows, and keeps Exit training/IQL/shadow/live closed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.scripts.audit_entry_exit_handoff_readiness_v1 import REQUIRED_EXIT_SUBSTRATE_FIELDS
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_IQL_TRADE_LOG = (
    REPORTS_ROOT / "entry_iql_student_trade_log_20260628_v1/entry_iql_student_trade_log.csv"
)
DEFAULT_IQL_SLICE_AUDIT_JSON = (
    REPORTS_ROOT / "entry_iql_replay_slice_audit_20260628_v1/ENTRY_IQL_REPLAY_SLICE_AUDIT_latest.json"
)
DEFAULT_IQL_COMPARISON_JSON = (
    REPORTS_ROOT / "entry_iql_replay_comparison_20260628_v1/ENTRY_IQL_REPLAY_COMPARISON_latest.json"
)
DEFAULT_M5_PRICE_PARQUET = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
DEFAULT_SUPPLEMENTAL_M1_GLOB = "/home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_*.parquet"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_per_bar_handoff_20260630_v1"
ATR_PERIOD_BARS = 14

PRICE_COLUMNS = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "atr_bps",
    "spread_bps",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normal_path(value: Any) -> str:
    raw = str(value or "").strip()
    return str(Path(raw).expanduser().resolve()) if raw else ""


def _candidate_bundle_dir(comparison: dict[str, Any]) -> str:
    for source in (
        comparison.get("evidence_identity"),
        (comparison.get("comparison") or {}).get("evidence_identity") if isinstance(comparison.get("comparison"), dict) else {},
    ):
        if isinstance(source, dict):
            raw = source.get("candidate_bundle_dir") or source.get("replay_identity_candidate_bundle_dir")
            if raw:
                return str(raw)
    return ""


def _aggregate_m1_to_m5(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    required = [
        col
        for col in (
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
        )
        if col in available
    ]
    if "time" not in required:
        return pd.DataFrame()
    frame = pd.read_parquet(path, columns=required)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame[(frame["time"] >= start) & (frame["time"] <= end)].copy()
    if frame.empty:
        return frame
    frame["time"] = frame["time"].dt.floor("5min")
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "bid_open": "first",
        "bid_high": "max",
        "bid_low": "min",
        "bid_close": "last",
        "ask_open": "first",
        "ask_high": "max",
        "ask_low": "min",
        "ask_close": "last",
    }
    agg = {key: value for key, value in agg.items() if key in frame.columns}
    out = frame.sort_values("time").groupby("time", as_index=False).agg(agg)
    if {"ask_close", "bid_close", "close"}.issubset(out.columns):
        out["spread_bps"] = ((out["ask_close"] - out["bid_close"]) / out["close"].clip(lower=1e-9)) * 10000.0
    out["atr_bps"] = np.nan
    return out


def _mid_or_fallback(frame: pd.DataFrame, bid_col: str, ask_col: str, fallback_col: str) -> pd.Series:
    values = pd.Series(np.nan, index=frame.index, dtype="float64")
    if bid_col in frame.columns and ask_col in frame.columns:
        bid = pd.to_numeric(frame[bid_col], errors="coerce")
        ask = pd.to_numeric(frame[ask_col], errors="coerce")
        values = (bid + ask) / 2.0
    if fallback_col in frame.columns:
        fallback = pd.to_numeric(frame[fallback_col], errors="coerce")
        values = values.where(values.notna(), fallback)
    return values


def _deterministic_atr_bps(frame: pd.DataFrame, *, period_bars: int = ATR_PERIOD_BARS) -> pd.Series:
    high = _mid_or_fallback(frame, "bid_high", "ask_high", "high")
    low = _mid_or_fallback(frame, "bid_low", "ask_low", "low")
    close = _mid_or_fallback(frame, "bid_close", "ask_close", "close")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    atr = true_range.rolling(int(period_bars), min_periods=1).mean()
    close_abs = close.abs().replace(0.0, np.nan)
    atr_bps = (atr / close_abs) * 10000.0
    return atr_bps.where(np.isfinite(atr_bps) & (atr_bps >= 0.0), np.nan)


def _glob_paths(pattern: str) -> list[Path]:
    if not pattern:
        return []
    return [Path(raw).expanduser().resolve() for raw in sorted(glob.glob(str(Path(pattern).expanduser())))]


def _load_prices(path: Path, supplemental_m1_glob: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [col for col in PRICE_COLUMNS if col in available]
    if not cols:
        cols = list(PRICE_COLUMNS)
    prices = pd.read_parquet(path, columns=cols)
    prices["time"] = pd.to_datetime(prices["time"], utc=True)
    prices = prices[(prices["time"] >= start) & (prices["time"] <= end)].copy()
    prices["price_source"] = "canonical_m5"
    prices["price_source_path"] = str(path)
    supplemental_frames: list[pd.DataFrame] = []
    supplemental_paths = _glob_paths(supplemental_m1_glob)
    for supplemental_path in supplemental_paths:
        frame = _aggregate_m1_to_m5(supplemental_path, start, end)
        if not frame.empty:
            frame["price_source"] = "supplemental_live_m1_to_m5"
            frame["price_source_path"] = str(supplemental_path)
            supplemental_frames.append(frame)
    if supplemental_frames:
        prices = pd.concat([prices, *supplemental_frames], ignore_index=True, sort=False)
        prices["_price_source_priority"] = np.where(prices["price_source"] == "canonical_m5", 0, 1)
        prices = (
            prices.sort_values(["time", "_price_source_priority"])
            .drop_duplicates("time", keep="first")
            .drop(columns=["_price_source_priority"])
        )
    prices = prices.sort_values("time").reset_index(drop=True)
    atr_source_column_present = "atr_bps" in available
    if "atr_bps" not in prices.columns:
        prices["atr_bps"] = np.nan
    prices["atr_bps"] = pd.to_numeric(prices["atr_bps"], errors="coerce")
    atr_missing_before = int(prices["atr_bps"].isna().sum())
    deterministic_atr = _deterministic_atr_bps(prices)
    fill_mask = prices["atr_bps"].isna() & deterministic_atr.notna()
    if bool(fill_mask.any()):
        prices.loc[fill_mask, "atr_bps"] = deterministic_atr.loc[fill_mask]
    atr_missing_after = int(prices["atr_bps"].isna().sum())
    supplemental_rows = (
        prices[prices["price_source"] == "supplemental_live_m1_to_m5"]
        if "price_source" in prices
        else pd.DataFrame()
    )
    supplemental_used_paths = sorted(str(path) for path in supplemental_rows.get("price_source_path", pd.Series(dtype=str)).dropna().unique())
    supplemental_input_sha256 = {used_path: _sha256_file(Path(used_path)) for used_path in supplemental_used_paths}
    diagnostics = {
        "canonical_rows": int((prices["price_source"] == "canonical_m5").sum()) if "price_source" in prices else int(len(prices)),
        "supplemental_rows_used": int((prices["price_source"] == "supplemental_live_m1_to_m5").sum()) if "price_source" in prices else 0,
        "supplemental_paths_considered": [str(candidate) for candidate in supplemental_paths],
        "supplemental_paths_used": supplemental_used_paths,
        "supplemental_input_sha256": supplemental_input_sha256,
        "atr_bps_fill_method": "preserve_source_else_mid_bid_ask_true_range_rolling_14_closed_m5_bars",
        "atr_bps_period_bars": ATR_PERIOD_BARS,
        "atr_bps_source_column_present": bool(atr_source_column_present),
        "atr_bps_null_rows_before_fill": atr_missing_before,
        "atr_bps_filled_rows": int(fill_mask.sum()),
        "atr_bps_null_rows_after_fill": atr_missing_after,
    }
    return prices, diagnostics


def _side_entry_price(trade: pd.Series) -> float:
    return float(trade["entry_price"])


def _side_exit_close(row: pd.Series, side: str) -> float:
    return float(row["bid_close"] if side == "LONG" else row["ask_close"])


def _pnl_bps(price: float, entry: float, side: str) -> float:
    if side == "LONG":
        return ((price - entry) / entry) * 10000.0
    return ((entry - price) / entry) * 10000.0


def _build_trade_rows(
    *,
    trade_idx: int,
    trade: pd.Series,
    bars: pd.DataFrame,
    candidate_bundle_dir: str,
    replay_identity_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    side = str(trade["side"]).upper()
    entry = _side_entry_price(trade)
    expected = int(trade.get("held_bars") or 0) + 1
    entry_trade_id = f"{trade.get('policy_id')}:{trade_idx}:{trade.get('entry_time')}:{side}"
    rows: list[dict[str, Any]] = []
    missing = max(0, expected - len(bars))
    bar_times = pd.to_datetime(bars["time"], utc=True, errors="coerce") if "time" in bars.columns else pd.Series(dtype="datetime64[ns, UTC]")
    invalid_bar_timestamp_count = int(bar_times.isna().sum()) if len(bar_times) else 0
    bar_time_diffs = bar_times.diff().dropna() if len(bar_times) else pd.Series(dtype="timedelta64[ns]")
    non_contiguous_5min_gap_count = int((bar_time_diffs != pd.Timedelta(minutes=5)).sum())
    first_bar_matches_entry = bool(len(bar_times) and bar_times.iloc[0] == pd.Timestamp(trade["entry_time"]))
    last_bar_matches_exit = bool(len(bar_times) and bar_times.iloc[-1] == pd.Timestamp(trade["exit_time"]))
    if side == "LONG":
        favorable = pd.to_numeric(bars["bid_high"], errors="coerce").cummax()
        adverse = pd.to_numeric(bars["bid_low"], errors="coerce").cummin()
        running_mfe = ((favorable - entry) / entry) * 10000.0
        running_mae = ((entry - adverse) / entry) * 10000.0
    else:
        favorable = pd.to_numeric(bars["ask_low"], errors="coerce").cummin()
        adverse = pd.to_numeric(bars["ask_high"], errors="coerce").cummax()
        running_mfe = ((entry - favorable) / entry) * 10000.0
        running_mae = ((adverse - entry) / entry) * 10000.0

    for bar_index, (_, bar) in enumerate(bars.iterrows()):
        exit_close = _side_exit_close(bar, side)
        running_pnl = _pnl_bps(exit_close, entry, side)
        mfe = float(max(0.0, running_mfe.iloc[bar_index]))
        mae = float(max(0.0, running_mae.iloc[bar_index]))
        rows.append(
            {
                "entry_trade_id": entry_trade_id,
                "bar_ts": bar["time"].isoformat(),
                "bar_index": int(bar_index),
                "side": side,
                "action_set": "HOLD,EXIT_NOW",
                "running_pnl_bps": float(running_pnl),
                "running_mfe_bps": mfe,
                "running_mae_bps": mae,
                "running_giveback_bps": float(max(0.0, mfe - running_pnl)),
                "bars_held": int(bar_index),
                "session": str(trade.get("session") or trade.get("state_session") or "UNKNOWN").upper(),
                "vol_regime": str(trade.get("vol_regime") or trade.get("state_vol_regime") or "UNKNOWN"),
                "spread_bps": float(bar.get("spread_bps", np.nan))
                if pd.notna(bar.get("spread_bps", np.nan))
                else float(((bar["ask_close"] - bar["bid_close"]) / max(1e-9, bar["close"])) * 10000.0),
                "atr_bps": float(bar.get("atr_bps", np.nan)) if pd.notna(bar.get("atr_bps", np.nan)) else np.nan,
                "bar_price_source": str(bar.get("price_source") or ""),
                "bar_price_source_path": str(bar.get("price_source_path") or ""),
                "entry_score": float(trade.get("score", np.nan)),
                "entry_p_long": float(trade.get("p_long", np.nan)),
                "entry_p_short": float(trade.get("p_short", np.nan)),
                "entry_p_flat": float(trade.get("p_flat", np.nan)),
                "entry_path_quality_pred": float(trade.get("path_quality_pred", np.nan)),
                "entry_bad_path_prob": float(trade.get("bad_path_prob", np.nan)),
                "entry_candidate_bundle_dir": candidate_bundle_dir,
                "entry_iql_policy_id": str(trade.get("policy_id") or ""),
                "entry_replay_identity_hash": replay_identity_hash,
                "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
                "realized_net_pnl_bps": float(trade.get("net_pnl_bps", np.nan)),
                "realized_gross_pnl_bps": float(trade.get("gross_pnl_bps", np.nan)),
                "realized_mfe_bps": float(trade.get("mfe_bps", np.nan)),
                "realized_mae_bps": float(trade.get("mae_bps", np.nan)),
                "realized_exit_reason": str(trade.get("exit_reason") or ""),
                "is_realized_exit_bar": bool(bar_index == max(0, len(bars) - 1)),
            }
        )
    diag = {
        "entry_trade_id": entry_trade_id,
        "expected_bar_count": expected,
        "observed_bar_count": int(len(bars)),
        "missing_bar_count": int(missing),
        "invalid_bar_timestamp_count": invalid_bar_timestamp_count,
        "non_contiguous_5min_gap_count": non_contiguous_5min_gap_count,
        "first_bar_matches_entry": first_bar_matches_entry,
        "last_bar_matches_exit": last_bar_matches_exit,
        "coverage_ready": bool(
            int(missing) == 0
            and invalid_bar_timestamp_count == 0
            and non_contiguous_5min_gap_count == 0
            and first_bar_matches_entry
            and last_bar_matches_exit
        ),
        "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
        "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
    }
    return rows, diag


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Per-Bar Handoff",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Source trade count: `{report['source_trade_count']}`",
        f"- Included trade count: `{report['included_trade_count']}`",
        f"- Excluded trade count: `{report['excluded_trade_count']}`",
        f"- Complete trades: `{report['complete_trade_count']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
        f"- Dataset: `{report['dataset_csv']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_log_path = Path(args.iql_trade_log).expanduser().resolve()
    price_path = Path(args.m5_price_parquet).expanduser().resolve()
    comparison_path = Path(args.iql_comparison_json).expanduser().resolve()
    slice_audit_path = Path(args.iql_slice_audit_json).expanduser().resolve()
    comparison = _read_json_or_empty(comparison_path)
    slice_audit = _read_json_or_empty(slice_audit_path)
    trades = pd.read_csv(trade_log_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    start = trades["entry_time"].min()
    end = trades["exit_time"].max()
    prices, price_diagnostics = _load_prices(price_path, str(args.supplemental_m1_glob), start, end)
    candidate_bundle_dir = _candidate_bundle_dir(comparison)
    replay_identity_hash = hashlib.sha256(
        (
            _sha256_file(trade_log_path)
            + _sha256_file(comparison_path)
            + _sha256_file(slice_audit_path)
        ).encode("utf-8")
    ).hexdigest()

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    excluded_trades: list[dict[str, Any]] = []
    for trade_idx, trade in trades.iterrows():
        mask = (prices["time"] >= trade["entry_time"]) & (prices["time"] <= trade["exit_time"])
        trade_bars = prices.loc[mask].copy()
        trade_rows, diag = _build_trade_rows(
            trade_idx=int(trade_idx),
            trade=trade,
            bars=trade_bars,
            candidate_bundle_dir=candidate_bundle_dir,
            replay_identity_hash=replay_identity_hash,
        )
        diagnostics.append(diag)
        if bool(diag.get("coverage_ready")):
            rows.extend(trade_rows)
        else:
            reasons: list[str] = []
            if int(diag["missing_bar_count"]) > 0:
                reasons.append("missing_per_bar_price_coverage")
            if int(diag["invalid_bar_timestamp_count"]) > 0:
                reasons.append("invalid_per_bar_price_timestamp")
            if int(diag["non_contiguous_5min_gap_count"]) > 0:
                reasons.append("non_contiguous_5min_per_bar_price_coverage")
            if not bool(diag["first_bar_matches_entry"]):
                reasons.append("first_bar_missing_at_entry_time")
            if not bool(diag["last_bar_matches_exit"]):
                reasons.append("last_bar_missing_at_exit_time")
            excluded_trades.append(
                {
                    **diag,
                    "source_trade_index": int(trade_idx),
                    "side": str(trade.get("side") or ""),
                    "reason": ",".join(reasons) if reasons else "per_bar_price_coverage_not_ready",
                }
            )

    dataset = pd.DataFrame(rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    exclusions_df = pd.DataFrame(excluded_trades)
    missing_fields = [field for field in REQUIRED_EXIT_SUBSTRATE_FIELDS if field not in set(dataset.columns)]
    complete_trade_count = int(diagnostics_df["coverage_ready"].astype(bool).sum()) if not diagnostics_df.empty else 0
    source_trade_count = int(len(trades))
    included_trade_count = complete_trade_count
    excluded_trade_count = int(len(excluded_trades))
    covered_trade_ratio = float(included_trade_count / source_trade_count) if source_trade_count else 0.0
    min_covered_trade_ratio = float(getattr(args, "min_covered_trade_ratio", 0.95))
    min_covered_trades = int(getattr(args, "min_covered_trades", 100))
    dataset_csv = out_dir / "entry_exit_per_bar_handoff.csv"
    diagnostics_csv = out_dir / "entry_exit_per_bar_handoff_trade_coverage.csv"
    exclusions_csv = out_dir / "entry_exit_per_bar_handoff_gap_exclusions.csv"
    dataset.to_csv(dataset_csv, index=False)
    diagnostics_df.to_csv(diagnostics_csv, index=False)
    exclusions_df.to_csv(exclusions_csv, index=False)
    checks = [
        _check("IQL comparison ready", str(comparison.get("decision")) == "READY_FOR_PROMOTION_REVIEW_VEDTAK", {"decision": comparison.get("decision")}),
        _check("IQL slice audit PASS", str(slice_audit.get("decision")) == "PASS", {"decision": slice_audit.get("decision")}),
        _check("IQL trade log rows are present", not trades.empty, {"rows": int(len(trades)), "path": str(trade_log_path)}),
        _check("canonical M5 price rows cover requested date range", not prices.empty, {"rows": int(len(prices)), "start": str(start), "end": str(end), "path": str(price_path)}),
        _check("per-bar dataset rows were produced", not dataset.empty, {"rows": int(len(dataset))}),
        _check(
            "included trades have complete per-bar coverage",
            included_trade_count == complete_trade_count and included_trade_count > 0,
            {
                "included_trade_count": included_trade_count,
                "complete_trade_count": complete_trade_count,
                "source_trade_count": source_trade_count,
            },
        ),
        _check(
            "per-bar coverage ratio meets review floor",
            covered_trade_ratio >= min_covered_trade_ratio and included_trade_count >= min_covered_trades,
            {
                "covered_trade_ratio": covered_trade_ratio,
                "min_covered_trade_ratio": min_covered_trade_ratio,
                "included_trade_count": included_trade_count,
                "min_covered_trades": min_covered_trades,
                "excluded_trade_count": excluded_trade_count,
                "missing": excluded_trades[:20],
            },
        ),
        _check(
            "gap exclusions are explicit and excluded from substrate",
            excluded_trade_count == source_trade_count - included_trade_count and exclusions_csv.exists(),
            {
                "excluded_trade_count": excluded_trade_count,
                "exclusions_csv": str(exclusions_csv),
                "excluded_reasons": sorted(set(exclusions_df["reason"].astype(str))) if not exclusions_df.empty else [],
            },
        ),
        _check("per-bar dataset has required Exit substrate fields", not missing_fields, {"missing_fields": missing_fields, "required_fields": list(REQUIRED_EXIT_SUBSTRATE_FIELDS)}),
        _check("materializer never trains, replays, builds adapters, promotes, shadows, or starts live", True, {"trainer_started": False, "replay_started": False, "adapter_built": False, "exit_training_allowed": False, "exit_iql_allowed": False, "promotion_shadow_live_allowed": False}),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = "PASS_WITH_EXPLICIT_GAP_EXCLUSIONS" if ready and excluded_trade_count else ("PASS" if ready else "FAIL")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_PER_BAR_HANDOFF_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_PER_BAR_HANDOFF_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_per_bar_handoff_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "iql_trade_log": str(trade_log_path),
        "iql_trade_log_sha256": _sha256_file(trade_log_path),
        "m5_price_parquet": str(price_path),
        "m5_price_parquet_sha256": _sha256_file(price_path),
        "supplemental_m1_glob": str(args.supplemental_m1_glob),
        "price_diagnostics": price_diagnostics,
        "iql_comparison_json": str(comparison_path),
        "iql_slice_audit_json": str(slice_audit_path),
        "candidate_bundle_dir": candidate_bundle_dir,
        "entry_replay_identity_hash": replay_identity_hash,
        "dataset_csv": str(dataset_csv),
        "trade_coverage_csv": str(diagnostics_csv),
        "gap_exclusions_csv": str(exclusions_csv),
        "gap_exclusions_csv_sha256": _sha256_file(exclusions_csv),
        "dataset_rows": int(len(dataset)),
        "trade_count": source_trade_count,
        "source_trade_count": source_trade_count,
        "included_trade_count": included_trade_count,
        "excluded_trade_count": excluded_trade_count,
        "complete_trade_count": complete_trade_count,
        "covered_trade_ratio": covered_trade_ratio,
        "min_covered_trade_ratio": min_covered_trade_ratio,
        "min_covered_trades": min_covered_trades,
        "gap_exclusion_policy": "exclude source trades with missing or non-contiguous per-bar price coverage; never synthesize bars",
        "required_exit_substrate_fields": list(REQUIRED_EXIT_SUBSTRATE_FIELDS),
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "adapter_built": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "rerun entry-exit-handoff and audit per-bar reconstruction before any Exit training"
            if ready
            else "repair per-bar handoff coverage before Exit handoff can pass"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_PER_BAR_HANDOFF_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_PER_BAR_HANDOFF_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "dataset_rows": report["dataset_rows"],
                    "trade_count": report["trade_count"],
                    "complete_trade_count": report["complete_trade_count"],
                    "failures": failures,
                    "json_path": str(json_path),
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iql-trade-log", default=str(DEFAULT_IQL_TRADE_LOG))
    ap.add_argument("--m5-price-parquet", default=str(DEFAULT_M5_PRICE_PARQUET))
    ap.add_argument("--supplemental-m1-glob", default=DEFAULT_SUPPLEMENTAL_M1_GLOB)
    ap.add_argument("--iql-comparison-json", default=str(DEFAULT_IQL_COMPARISON_JSON))
    ap.add_argument("--iql-slice-audit-json", default=str(DEFAULT_IQL_SLICE_AUDIT_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-covered-trade-ratio", type=float, default=0.95)
    ap.add_argument("--min-covered-trades", type=int, default=100)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
