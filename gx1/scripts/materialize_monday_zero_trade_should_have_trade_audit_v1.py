#!/usr/bin/env python3
"""
Materialize a conservative Monday zero-trade should-have-trade audit.

This audit does not change policy and does not invent synthetic labels.
It combines:
1. Continuous market opportunity from per-bar eval logs.
2. Entry gate pressure from trade_report_entry_gates.
3. Candidate-surface richness from shadow_meta_candidates.

Because zero-trade Monday weeks currently do not carry populated hindsight
trade outcome labels inside shadow_meta_candidates, the verdict is explicitly
conservative:
- TRUE_NO_TRADE_REGIME -> BEVIST
- OVERFILTERED_SHOULD_HAVE_TRADED -> INDIKERT
- AMBIGUOUS_NEEDS_NO_TRADE_HINDSIGHT -> IKKE_ETABLERT
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


ARTIFACT_NAME = "MONDAY_ZERO_TRADE_SHOULD_HAVE_TRADE_AUDIT_V1"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
RUN_PREFIX = "TRUTH_MONFRI_WEEK_"
QUARANTINE_RUN_IDS = {
    "TRUTH_MONFRI_WEEK_20251201_20251208",
    "TRUTH_MONFRI_WEEK_20251208_20251215",
}
DEFAULT_HORIZONS_BARS = (60, 240)
DEFAULT_THRESHOLD_BPS = (50.0, 100.0)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_reports_root(path_arg: Optional[str]) -> Path:
    root = Path(path_arg).expanduser().resolve() if path_arg else DEFAULT_REPORTS_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Reports root does not exist: {root}")
    return root


def _runs_root(reports_root: Path) -> Path:
    candidate = reports_root / "runs"
    return candidate if candidate.exists() else reports_root


def _parse_run_dates(run_id: str) -> Dict[str, Optional[str]]:
    if not run_id.startswith(RUN_PREFIX):
        return {"start_date": None, "end_date": None}
    payload = run_id[len(RUN_PREFIX) :]
    parts = payload.split("_")
    if len(parts) != 2 or len(parts[0]) != 8 or len(parts[1]) != 8:
        return {"start_date": None, "end_date": None}
    start_ts = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(parts[1], format="%Y%m%d", errors="coerce")
    return {
        "start_date": None if pd.isna(start_ts) else start_ts.date().isoformat(),
        "end_date": None if pd.isna(end_ts) else end_ts.date().isoformat(),
    }


def _stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(values), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": int(len(series)),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def _series_summary(frame: pd.DataFrame, column: str) -> Dict[str, Optional[float]]:
    return _stats(pd.to_numeric(frame.get(column, pd.Series(dtype="float64")), errors="coerce").dropna().tolist())


def _load_json_object(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_eval_price_frame(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(run_dir.glob("replay/chunk_*/logs/eval_log_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                ts_utc = payload.get("ts_utc")
                price = payload.get("price")
                if ts_utc is None or price is None:
                    continue
                rows.append({"ts_utc": ts_utc, "price": price, "session": payload.get("session")})
    if not rows:
        return pd.DataFrame(columns=["ts_utc", "price", "session"])
    frame = pd.DataFrame(rows)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["ts_utc", "price"]).sort_values("ts_utc", kind="mergesort")
    frame = frame.drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)
    return frame


def _forward_best_move_bps(price: pd.Series, horizon_bars: int) -> pd.Series:
    rev = price.iloc[::-1]
    future_max = rev.shift(1).rolling(window=horizon_bars, min_periods=1).max().iloc[::-1]
    future_min = rev.shift(1).rolling(window=horizon_bars, min_periods=1).min().iloc[::-1]
    up_move_bps = ((future_max - price) / price * 1e4).clip(lower=0.0)
    down_move_bps = ((price - future_min) / price * 1e4).clip(lower=0.0)
    return np.maximum(up_move_bps, down_move_bps)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed) or np.isinf(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _metric_against_benchmark(
    value: Optional[float],
    *,
    median_value: Optional[float],
    p25_value: Optional[float],
) -> Dict[str, Any]:
    if value is None or median_value is None or p25_value is None:
        return {
            "value": value,
            "median": median_value,
            "p25": p25_value,
            "ratio_to_median": None,
            "ge_median": False,
            "lt_p25": False,
        }
    ratio = value / median_value if median_value not in {0.0, None} else None
    return {
        "value": value,
        "median": median_value,
        "p25": p25_value,
        "ratio_to_median": ratio,
        "ge_median": bool(value >= median_value),
        "lt_p25": bool(value < p25_value),
    }


def _build_contract(horizons_bars: Sequence[int], thresholds_bps: Sequence[float]) -> Dict[str, Any]:
    return {
        "artifact_name": ARTIFACT_NAME,
        "scope_v1": "MONDAY_ZERO_TRADE_WEEKS_ONLY",
        "run_prefix_v1": RUN_PREFIX,
        "quarantine_excluded_run_ids_v1": sorted(QUARANTINE_RUN_IDS),
        "horizons_bars_v1": [int(item) for item in horizons_bars],
        "thresholds_bps_v1": [float(item) for item in thresholds_bps],
        "truth_contract_v1": {
            "replay_semantics_changed": False,
            "policy_changed": False,
            "thresholds_changed": False,
            "guards_changed": False,
            "locked_truth_changed": False,
        },
        "verdict_contract_v1": {
            "TRUE_NO_TRADE_REGIME": "All anchor opportunity metrics are below the nonzero-week p25 benchmark. This is a BEVIST low-opportunity regime verdict.",
            "OVERFILTERED_SHOULD_HAVE_TRADED": "At least two anchor opportunity metrics meet or exceed nonzero-week medians while the week still has zero accepted entries. This is INDIKERT, not BEVIST, until explicit no-trade hindsight labels exist.",
            "AMBIGUOUS_NEEDS_NO_TRADE_HINDSIGHT": "Neither low-opportunity nor overfiltered criteria are strong enough. This remains IKKE_ETABLERT for should-have-trade.",
        },
        "design_locks_v1": {
            "no_synthetic_labels_v1": True,
            "no_policy_backfill_v1": True,
            "no_as_of_hindsight_mix_v1": True,
            "continuous_market_opportunity_used_as_hindsight_audit_only_v1": True,
            "zero_trade_candidate_hindsight_fields_missing_is_explicit_v1": True,
        },
    }


def build_monday_zero_trade_should_have_trade_audit(
    reports_root: Path,
    *,
    horizons_bars: Sequence[int] = DEFAULT_HORIZONS_BARS,
    thresholds_bps: Sequence[float] = DEFAULT_THRESHOLD_BPS,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    runs_root = _runs_root(reports_root)
    run_dirs = sorted(
        [
            path
            for path in runs_root.iterdir()
            if path.is_dir()
            and path.name.startswith(RUN_PREFIX)
            and "PREFLIGHT" not in path.name
            and path.name not in QUARANTINE_RUN_IDS
        ],
        key=lambda path: path.name,
    )

    run_rows: List[Dict[str, Any]] = []
    missing_eval_runs: List[str] = []

    candidate_columns = [
        "decision",
        "accepted",
        "decision_reason",
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "margin",
        "tradable_prob",
        "mfe_first_n_pred",
        "path_quality_pred",
        "session",
        "vol_regime",
        "trend_regime",
    ]

    for run_dir in run_dirs:
        run_id = run_dir.name
        metrics_paths = list(run_dir.glob("metrics_*_MERGED.json"))
        if not metrics_paths:
            continue

        metrics_payload = _load_json_object(metrics_paths[0])
        run_completed_exists = (run_dir / "RUN_COMPLETED.json").exists()
        if not run_completed_exists:
            continue

        gate_path = run_dir / f"trade_report_entry_gates_{run_id}.json"
        candidate_path = run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet"
        if not gate_path.exists() or not candidate_path.exists():
            continue

        gate_payload = _load_json_object(gate_path)
        candidate_df = pd.read_parquet(candidate_path, columns=candidate_columns)
        price_frame = _load_eval_price_frame(run_dir)
        if price_frame.empty:
            missing_eval_runs.append(run_id)
            continue

        price_series = price_frame["price"].astype("float64")
        row: Dict[str, Any] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run_completed_exists_v1": True,
            "n_trades_v1": _safe_int(metrics_payload.get("n_trades")),
            "n_model_calls_v1": _safe_int(metrics_payload.get("n_model_calls")),
            "candidate_rows_v1": int(len(candidate_df)),
            "eval_price_rows_v1": int(len(price_frame)),
            "accepted_rows_v1": int(candidate_df.get("accepted", pd.Series(dtype="boolean")).fillna(False).astype(bool).sum()),
            "decision_none_rows_v1": int(candidate_df.get("decision", pd.Series(dtype="string")).astype("string").eq("NONE").sum()),
            "decision_long_rows_v1": int(candidate_df.get("decision", pd.Series(dtype="string")).astype("string").eq("LONG").sum()),
            "flat_dominant_rows_v1": int(candidate_df.get("decision_reason", pd.Series(dtype="string")).astype("string").eq("flat_dominant").sum()),
            "flat_veto_rows_v1": int(candidate_df.get("decision_reason", pd.Series(dtype="string")).astype("string").eq("flat_veto").sum()),
            "candidate_signal_p_long_v1": _series_summary(candidate_df, "p_long"),
            "candidate_signal_p_flat_v1": _series_summary(candidate_df, "p_flat"),
            "candidate_signal_p_hat_v1": _series_summary(candidate_df, "p_hat"),
            "candidate_signal_margin_v1": _series_summary(candidate_df, "margin"),
            "candidate_signal_tradable_prob_v1": _series_summary(candidate_df, "tradable_prob"),
            "candidate_signal_mfe_first_n_pred_v1": _series_summary(candidate_df, "mfe_first_n_pred"),
            "candidate_signal_path_quality_pred_v1": _series_summary(candidate_df, "path_quality_pred"),
            "candidate_high_margin_rows_v1": int(pd.to_numeric(candidate_df.get("margin"), errors="coerce").ge(0.90).fillna(False).sum()),
            "candidate_high_tradable_prob_rows_v1": int(pd.to_numeric(candidate_df.get("tradable_prob"), errors="coerce").ge(0.75).fillna(False).sum()),
            "candidate_high_path_quality_rows_v1": int(pd.to_numeric(candidate_df.get("path_quality_pred"), errors="coerce").ge(0.80).fillna(False).sum()),
            "no_trade_hindsight_present_v1": False,
        }

        runner_counters = gate_payload.get("runner_counters", {})
        entry_gate_counters = runner_counters.get("entry_gate_counters", {})
        row.update(
            {
                "pregate_passes_v1": _safe_int(runner_counters.get("pregate_passes")),
                "entry_attempt_long_v1": _safe_int(runner_counters.get("entry_attempt_long")),
                "entry_accept_long_v1": _safe_int(runner_counters.get("entry_accept_long")),
                "candidate_below_threshold_rows_v1": _safe_int(entry_gate_counters.get("candidate_below_threshold")),
                "candidate_flat_veto_rows_v1": _safe_int(entry_gate_counters.get("candidate_flat_veto")),
                "pregate_session_rows_v1": _safe_int(entry_gate_counters.get("pregate_session")),
                "pregate_weekly_entry_window_rows_v1": _safe_int(entry_gate_counters.get("pregate_weekly_entry_window")),
                "pregate_d1_atr_eu_rows_v1": _safe_int(entry_gate_counters.get("pregate_d1_atr_eu")),
                "pregate_regime_filter_rows_v1": _safe_int(entry_gate_counters.get("pregate_regime_filter")),
                "threshold_used_v1": runner_counters.get("threshold_used"),
                "threshold_source_v1": runner_counters.get("threshold_source"),
            }
        )
        candidate_rows = max(1, int(len(candidate_df)))
        row["gate_flat_veto_rate_v1"] = float(row["candidate_flat_veto_rows_v1"] / candidate_rows)
        row["gate_below_threshold_rate_v1"] = float(row["candidate_below_threshold_rows_v1"] / candidate_rows)
        row["decision_flat_reason_share_v1"] = float(
            (row["flat_dominant_rows_v1"] + row["flat_veto_rows_v1"]) / candidate_rows
        )

        for horizon in horizons_bars:
            horizon_int = int(horizon)
            best_move_bps = _forward_best_move_bps(price_series, horizon_int)
            row[f"best_move_mean_h{horizon_int}_bps_v1"] = float(best_move_bps.mean())
            row[f"best_move_median_h{horizon_int}_bps_v1"] = float(pd.Series(best_move_bps).median())
            row[f"best_move_p90_h{horizon_int}_bps_v1"] = float(pd.Series(best_move_bps).quantile(0.90))
            for threshold in thresholds_bps:
                row[f"rate_ge_{int(threshold)}bps_h{horizon_int}_v1"] = float((best_move_bps >= float(threshold)).mean())

        row.update(_parse_run_dates(run_id))
        run_rows.append(row)

    run_df = pd.DataFrame(run_rows)
    if run_df.empty:
        raise RuntimeError("No completed Monday run rows found for zero-trade should-have-trade audit.")

    nonzero_df = run_df.loc[run_df["n_trades_v1"].gt(0)].copy()
    zero_df = run_df.loc[run_df["n_trades_v1"].eq(0)].copy()

    benchmark_columns = [
        "best_move_mean_h60_bps_v1",
        "best_move_median_h60_bps_v1",
        "rate_ge_50bps_h60_v1",
        "best_move_mean_h240_bps_v1",
        "rate_ge_50bps_h240_v1",
    ]
    benchmarks: Dict[str, Dict[str, Optional[float]]] = {
        column: _stats(nonzero_df[column].tolist()) if column in nonzero_df.columns else _stats([])
        for column in benchmark_columns
    }

    verdict_rows: List[Dict[str, Any]] = []
    overfiltered_runs: List[str] = []
    true_no_trade_runs: List[str] = []
    ambiguous_runs: List[str] = []

    for row in zero_df.to_dict(orient="records"):
        metric_checks = {
            "best_move_mean_h60_bps_v1": _metric_against_benchmark(
                _safe_float(row.get("best_move_mean_h60_bps_v1")),
                median_value=benchmarks["best_move_mean_h60_bps_v1"]["median"],
                p25_value=benchmarks["best_move_mean_h60_bps_v1"]["p25"],
            ),
            "best_move_median_h60_bps_v1": _metric_against_benchmark(
                _safe_float(row.get("best_move_median_h60_bps_v1")),
                median_value=benchmarks["best_move_median_h60_bps_v1"]["median"],
                p25_value=benchmarks["best_move_median_h60_bps_v1"]["p25"],
            ),
            "rate_ge_50bps_h60_v1": _metric_against_benchmark(
                _safe_float(row.get("rate_ge_50bps_h60_v1")),
                median_value=benchmarks["rate_ge_50bps_h60_v1"]["median"],
                p25_value=benchmarks["rate_ge_50bps_h60_v1"]["p25"],
            ),
            "best_move_mean_h240_bps_v1": _metric_against_benchmark(
                _safe_float(row.get("best_move_mean_h240_bps_v1")),
                median_value=benchmarks["best_move_mean_h240_bps_v1"]["median"],
                p25_value=benchmarks["best_move_mean_h240_bps_v1"]["p25"],
            ),
            "rate_ge_50bps_h240_v1": _metric_against_benchmark(
                _safe_float(row.get("rate_ge_50bps_h240_v1")),
                median_value=benchmarks["rate_ge_50bps_h240_v1"]["median"],
                p25_value=benchmarks["rate_ge_50bps_h240_v1"]["p25"],
            ),
        }
        high_count = sum(1 for item in metric_checks.values() if item["ge_median"])
        low_count = sum(1 for item in metric_checks.values() if item["lt_p25"])
        gate_wall = bool(
            _safe_float(row.get("gate_below_threshold_rate_v1")) is not None
            and _safe_float(row.get("decision_flat_reason_share_v1")) is not None
            and _safe_float(row.get("gate_below_threshold_rate_v1")) >= 0.95
            and _safe_float(row.get("decision_flat_reason_share_v1")) >= 0.95
        )

        if low_count == len(metric_checks):
            verdict = "TRUE_NO_TRADE_REGIME"
            hard_status = "BEVIST"
            should_have_trade_status = "NO"
            rationale = "All anchor opportunity metrics sit below the nonzero-week p25 benchmark."
            true_no_trade_runs.append(str(row["run_id"]))
        elif high_count >= 2 and gate_wall:
            verdict = "OVERFILTERED_SHOULD_HAVE_TRADED"
            hard_status = "INDIKERT"
            should_have_trade_status = "YES_SIGNAL"
            rationale = "At least two anchor opportunity metrics match or beat nonzero medians while threshold/flat gates block the entire candidate surface."
            overfiltered_runs.append(str(row["run_id"]))
        else:
            verdict = "AMBIGUOUS_NEEDS_NO_TRADE_HINDSIGHT"
            hard_status = "IKKE_ETABLERT"
            should_have_trade_status = "UNKNOWN"
            rationale = "Opportunity pressure is mixed; explicit no-trade hindsight supervision is still missing."
            ambiguous_runs.append(str(row["run_id"]))

        verdict_rows.append(
            {
                **row,
                "opportunity_metric_high_count_v1": int(high_count),
                "opportunity_metric_low_count_v1": int(low_count),
                "gate_wall_v1": bool(gate_wall),
                "verdict_v1": verdict,
                "hard_status_v1": hard_status,
                "should_have_trade_status_v1": should_have_trade_status,
                "verdict_rationale_v1": rationale,
                "metric_checks_json_v1": json.dumps(metric_checks, ensure_ascii=True, sort_keys=True),
            }
        )

    verdict_df = pd.DataFrame(verdict_rows).sort_values(["verdict_v1", "run_id"], kind="mergesort").reset_index(drop=True)

    summary = {
        "artifact_name_v1": ARTIFACT_NAME,
        "reports_root_v1": str(reports_root),
        "runs_root_v1": str(runs_root),
        "created_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "run_dir_count_v1": int(len(run_df)),
        "completed_nonzero_runs_v1": int(len(nonzero_df)),
        "completed_zero_trade_runs_v1": int(len(zero_df)),
        "missing_eval_runs_v1": missing_eval_runs,
        "nonzero_benchmarks_v1": benchmarks,
        "zero_trade_verdict_counts_v1": {
            "TRUE_NO_TRADE_REGIME": int(len(true_no_trade_runs)),
            "OVERFILTERED_SHOULD_HAVE_TRADED": int(len(overfiltered_runs)),
            "AMBIGUOUS_NEEDS_NO_TRADE_HINDSIGHT": int(len(ambiguous_runs)),
        },
        "true_no_trade_run_ids_v1": true_no_trade_runs,
        "overfiltered_should_have_traded_run_ids_v1": overfiltered_runs,
        "ambiguous_zero_trade_run_ids_v1": ambiguous_runs,
        "zero_trade_detail_sample_v1": verdict_df.head(max(1, int(sample_limit))).to_dict(orient="records"),
        "recommendations_v1": [
            "Promote Monday zero-trade weeks to first-class entry review objects instead of treating them as explanatory leftovers.",
            "Build an explicit no-trade hindsight layer before turning OVERFILTERED_SHOULD_HAVE_TRADED from INDIKERT into BEVIST.",
            "Track threshold-plus-flat gate walls per week because several zero-trade weeks still show strong continuous opportunity pressure.",
        ],
        "design_warnings_v1": [
            "shadow_meta_candidates for zero-trade Monday weeks do not currently carry populated hindsight outcome columns.",
            "This audit uses continuous market opportunity as hindsight-only evidence and does not backfill policy labels.",
        ],
        "hard_status_split_v1": {
            "BEVIST": "TRUE_NO_TRADE_REGIME only",
            "INDIKERT": "OVERFILTERED_SHOULD_HAVE_TRADED",
            "IKKE_ETABLERT": "AMBIGUOUS_NEEDS_NO_TRADE_HINDSIGHT",
        },
    }
    return {
        "contract": _build_contract(horizons_bars, thresholds_bps),
        "run_df": run_df.sort_values("run_id", kind="mergesort").reset_index(drop=True),
        "verdict_df": verdict_df,
        "summary": summary,
        "benchmarks": benchmarks,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return path


def _write_markdown(path: Path, *, summary: Dict[str, Any], verdict_df: pd.DataFrame) -> Path:
    lines = [
        f"# {ARTIFACT_NAME}",
        "",
        f"Generated: `{summary['created_at_utc_v1']}`",
        "",
        "## Verdict Counts",
        "",
        f"- `TRUE_NO_TRADE_REGIME`: `{summary['zero_trade_verdict_counts_v1']['TRUE_NO_TRADE_REGIME']}`",
        f"- `OVERFILTERED_SHOULD_HAVE_TRADED`: `{summary['zero_trade_verdict_counts_v1']['OVERFILTERED_SHOULD_HAVE_TRADED']}`",
        f"- `AMBIGUOUS_NEEDS_NO_TRADE_HINDSIGHT`: `{summary['zero_trade_verdict_counts_v1']['AMBIGUOUS_NEEDS_NO_TRADE_HINDSIGHT']}`",
        "",
        "## Hard Status",
        "",
        "- `BEVIST`: only low-opportunity weeks where all anchor metrics are below nonzero p25 benchmarks.",
        "- `INDIKERT`: opportunity-rich zero-trade weeks where threshold/flat gates still block the full surface.",
        "- `IKKE_ETABLERT`: mixed weeks that still need explicit no-trade hindsight labeling.",
        "",
        "## Zero-Trade Weeks",
        "",
        "| run_id | verdict | hard_status | trades | best60_mean | rate50_60 | best240_mean | rate50_240 | flat_veto_rate | below_threshold_rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if verdict_df.empty:
        lines.append("| _none_ | | | | | | | | | |")
    else:
        for row in verdict_df.sort_values(["verdict_v1", "run_id"], kind="mergesort").to_dict(orient="records"):
            lines.append(
                "| {run_id} | {verdict_v1} | {hard_status_v1} | {n_trades_v1} | {best60:.2f} | {rate60:.4f} | {best240:.2f} | {rate240:.4f} | {flat_veto:.4f} | {below_thr:.4f} |".format(
                    run_id=row["run_id"],
                    verdict_v1=row["verdict_v1"],
                    hard_status_v1=row["hard_status_v1"],
                    n_trades_v1=int(row["n_trades_v1"]),
                    best60=float(row["best_move_mean_h60_bps_v1"]),
                    rate60=float(row["rate_ge_50bps_h60_v1"]),
                    best240=float(row["best_move_mean_h240_bps_v1"]),
                    rate240=float(row["rate_ge_50bps_h240_v1"]),
                    flat_veto=float(row["gate_flat_veto_rate_v1"]),
                    below_thr=float(row["gate_below_threshold_rate_v1"]),
                )
            )
        lines.extend(
            [
                "",
                "## Recommendations",
                "",
            ]
        )
        for item in summary.get("recommendations_v1", []):
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def materialize_monday_zero_trade_should_have_trade_audit(
    reports_root: Path,
    *,
    output_root: Optional[Path] = None,
    sample_limit: int = 10,
) -> Dict[str, Path]:
    payload = build_monday_zero_trade_should_have_trade_audit(
        reports_root,
        sample_limit=sample_limit,
    )
    out_dir = output_root or (reports_root / f"{ARTIFACT_NAME}_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=False)

    contract_path = _write_json(out_dir / "contract_v1.json", payload["contract"])
    by_run_path = out_dir / "by_run_v1.csv"
    payload["run_df"].to_csv(by_run_path, index=False)
    verdict_path = out_dir / "zero_trade_verdicts_v1.csv"
    payload["verdict_df"].to_csv(verdict_path, index=False)
    benchmarks_path = _write_json(out_dir / "nonzero_benchmarks_v1.json", payload["benchmarks"])
    summary_path = _write_json(out_dir / "summary_v1.json", payload["summary"])
    report_path = _write_markdown(out_dir / "report_v1.md", summary=payload["summary"], verdict_df=payload["verdict_df"])

    consistency_audit = {
        "status_v1": "PASS",
        "completed_zero_trade_runs_v1": int(payload["summary"]["completed_zero_trade_runs_v1"]),
        "verdict_rows_v1": int(len(payload["verdict_df"])),
        "verdict_count_match_v1": bool(int(len(payload["verdict_df"])) == int(payload["summary"]["completed_zero_trade_runs_v1"])),
        "missing_eval_runs_v1": payload["summary"]["missing_eval_runs_v1"],
        "notes_v1": [
            "Verdict rows must equal the number of completed zero-trade Monday runs.",
            "No-trade hindsight columns are explicitly absent for zero-trade candidate rows; this audit remains conservative by design.",
        ],
    }
    if not consistency_audit["verdict_count_match_v1"]:
        consistency_audit["status_v1"] = "FAIL"
    consistency_path = _write_json(out_dir / "consistency_audit_v1.json", consistency_audit)

    manifest = {
        "artifact_name_v1": ARTIFACT_NAME,
        "created_at_utc_v1": payload["summary"]["created_at_utc_v1"],
        "reports_root_v1": str(reports_root),
        "output_root_v1": str(out_dir),
        "status_v1": consistency_audit["status_v1"],
        "files_v1": {
            "contract_v1": str(contract_path),
            "by_run_v1": str(by_run_path),
            "zero_trade_verdicts_v1": str(verdict_path),
            "nonzero_benchmarks_v1": str(benchmarks_path),
            "summary_v1": str(summary_path),
            "report_v1": str(report_path),
            "consistency_audit_v1": str(consistency_path),
        },
    }
    manifest_path = _write_json(out_dir / "manifest_status_v1.json", manifest)
    return {
        "output_root": out_dir,
        "contract_v1": contract_path,
        "by_run_v1": by_run_path,
        "zero_trade_verdicts_v1": verdict_path,
        "nonzero_benchmarks_v1": benchmarks_path,
        "summary_v1": summary_path,
        "report_v1": report_path,
        "consistency_audit_v1": consistency_path,
        "manifest_status_v1": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Monday zero-trade should-have-trade audit.")
    parser.add_argument("--reports-root", default=None, help="Truth reports root. Defaults to /home/andre2/GX1_DATA/reports/truth_e2e_sanity.")
    parser.add_argument("--output-root", default=None, help="Optional explicit output directory.")
    parser.add_argument("--sample-limit", type=int, default=10, help="Maximum sample rows in summary sections.")
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None
    paths = materialize_monday_zero_trade_should_have_trade_audit(
        reports_root,
        output_root=output_root,
        sample_limit=max(1, int(args.sample_limit)),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
