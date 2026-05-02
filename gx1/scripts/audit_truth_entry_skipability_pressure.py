#!/usr/bin/env python3
"""
Audit truth replay roots for entry/skipability pressure.

This focuses on weekly trade-count dispersion, zero-trade weeks, and whether
those weeks still had rich candidate surfaces that were never accepted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


ACTIVE_ROOT_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
SUPPORTED_RUN_PREFIXES = ("E2E_SANITY_ORDERFIX_", "TRUTH_MONFRI_WEEK_")
RUN_ID_RE = re.compile(r"^(?:E2E_SANITY_ORDERFIX|TRUTH_MONFRI_WEEK)_(\d{8})_(\d{8})$")


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _resolve_reports_root(reports_root: Optional[str]) -> Path:
    if reports_root:
        root = Path(reports_root).expanduser().resolve()
    else:
        if not ACTIVE_ROOT_POINTER.exists():
            raise FileNotFoundError(
                f"Active root pointer not found: {ACTIVE_ROOT_POINTER}"
            )
        root = Path(ACTIVE_ROOT_POINTER.read_text(encoding="utf-8").strip()).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Reports root does not exist: {root}")
    return root


def _runs_root(reports_root: Path) -> Path:
    candidate = reports_root / "runs"
    return candidate if candidate.exists() else reports_root


def _safe_read_parquet(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if columns is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=list(columns))


def _parse_run_dates(run_id: str) -> Dict[str, Optional[str]]:
    match = RUN_ID_RE.fullmatch(run_id)
    if match is None:
        return {"start_date": None, "end_date": None}
    start_ts = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(match.group(2), format="%Y%m%d", errors="coerce")
    return {
        "start_date": None if pd.isna(start_ts) else start_ts.date().isoformat(),
        "end_date": None if pd.isna(end_ts) else end_ts.date().isoformat(),
    }


def _reason_rate(counter: Counter[str], key: str, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(counter.get(key, 0) / total)


def _stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series(list(values), dtype="float64")
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


def _top_counter(counter: Counter[str], limit: int) -> Dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit)}


def _build_zero_trade_clusters(zero_runs_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if zero_runs_df.empty:
        return []
    work = zero_runs_df.copy()
    work["start_ts"] = pd.to_datetime(work["start_date"], errors="coerce")
    work["end_ts"] = pd.to_datetime(work["end_date"], errors="coerce")
    work = work.sort_values(["start_ts", "run_id"], kind="mergesort").reset_index(drop=True)
    clusters: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for row in work.itertuples(index=False):
        if current is None:
            current = {
                "cluster_start_date": row.start_date,
                "cluster_end_date": row.end_date,
                "week_count": 1,
                "run_ids": [row.run_id],
            }
            continue
        prev_end = current["cluster_end_date"]
        if prev_end is not None and row.start_date == prev_end:
            current["cluster_end_date"] = row.end_date
            current["week_count"] += 1
            current["run_ids"].append(row.run_id)
        else:
            clusters.append(current)
            current = {
                "cluster_start_date": row.start_date,
                "cluster_end_date": row.end_date,
                "week_count": 1,
                "run_ids": [row.run_id],
            }
    if current is not None:
        clusters.append(current)
    return clusters


def build_skipability_pressure_summary(reports_root: Path, sample_limit: int = 10) -> Dict[str, Any]:
    runs_root = _runs_root(reports_root)
    run_dirs = sorted(
        [path for path in runs_root.iterdir() if path.is_dir() and RUN_ID_RE.fullmatch(path.name)],
        key=lambda path: path.name,
    )
    run_rows: List[Dict[str, Any]] = []
    reason_counter_zero: Counter[str] = Counter()
    reason_counter_nonzero: Counter[str] = Counter()
    session_counter_zero: Counter[str] = Counter()
    session_counter_nonzero: Counter[str] = Counter()

    for run_dir in run_dirs:
        run_id = run_dir.name
        trade_path = run_dir / f"trade_outcomes_{run_id}_MERGED.parquet"
        candidate_path = run_dir / f"shadow_meta_candidates_{run_id}_MERGED.parquet"
        completed = (run_dir / "RUN_COMPLETED.json").exists()

        trade_df = _safe_read_parquet(trade_path, columns=["trade_id"])
        candidate_df = _safe_read_parquet(
            candidate_path,
            columns=["decision", "accepted", "decision_reason", "session"],
        )

        trade_count = int(len(trade_df))
        candidate_rows = int(len(candidate_df))
        decision_series = candidate_df.get("decision", pd.Series(dtype="string")).astype("string")
        accepted_series = candidate_df.get("accepted", pd.Series(dtype="boolean")).fillna(False).astype(bool)
        reason_series = candidate_df.get("decision_reason", pd.Series(dtype="string")).astype("string")
        session_series = candidate_df.get("session", pd.Series(dtype="string")).astype("string")
        reason_counts = Counter(reason_series.dropna().tolist())
        session_counts = Counter(session_series.dropna().tolist())

        accepted_rows = int(accepted_series.sum())
        none_rows = int(decision_series.eq("NONE").sum())
        long_rows = int(decision_series.eq("LONG").sum())
        acceptance_rate = float(accepted_rows / candidate_rows) if candidate_rows else 0.0
        run_row: Dict[str, Any] = {
            "run_id": run_id,
            "completed": bool(completed),
            "trade_count": trade_count,
            "candidate_rows": candidate_rows,
            "accepted_rows": accepted_rows,
            "accepted_to_trade_delta": int(accepted_rows - trade_count),
            "acceptance_rate": acceptance_rate,
            "none_rows": none_rows,
            "long_rows": long_rows,
            "flat_dominant_rate": _reason_rate(reason_counts, "flat_dominant", candidate_rows),
            "flat_veto_rate": _reason_rate(reason_counts, "flat_veto", candidate_rows),
            "top_decision_reason": reason_counts.most_common(1)[0][0] if reason_counts else None,
            "top_decision_reasons": _top_counter(reason_counts, 5),
            "session_counts": {key: int(value) for key, value in session_counts.items()},
        }
        run_row.update(_parse_run_dates(run_id))
        run_rows.append(run_row)

        target_reason_counter = reason_counter_zero if trade_count == 0 else reason_counter_nonzero
        target_session_counter = session_counter_zero if trade_count == 0 else session_counter_nonzero
        target_reason_counter.update(reason_counts)
        target_session_counter.update(session_counts)

    run_df = pd.DataFrame(run_rows)
    completed_df = run_df.loc[run_df["completed"].fillna(False)].copy() if not run_df.empty else pd.DataFrame()
    zero_trade_df = completed_df.loc[completed_df["trade_count"].fillna(0).eq(0)].copy() if not completed_df.empty else pd.DataFrame()
    nonzero_df = completed_df.loc[completed_df["trade_count"].fillna(0).gt(0)].copy() if not completed_df.empty else pd.DataFrame()
    incomplete_df = run_df.loc[~run_df["completed"].fillna(False)].copy() if not run_df.empty else pd.DataFrame()

    candidate_rich_zero_df = zero_trade_df.loc[zero_trade_df["candidate_rows"].fillna(0).ge(1000)].copy() if not zero_trade_df.empty else pd.DataFrame()
    trade_count_ratio = None
    if not nonzero_df.empty:
        median_nonzero_trade_count = float(nonzero_df["trade_count"].median())
        max_nonzero_trade_count = float(nonzero_df["trade_count"].max())
        if median_nonzero_trade_count > 0:
            trade_count_ratio = float(max_nonzero_trade_count / median_nonzero_trade_count)

    zero_trade_share = float(len(zero_trade_df) / len(completed_df)) if len(completed_df) else 0.0
    zero_trade_clusters = _build_zero_trade_clusters(zero_trade_df)
    zero_trade_reason_top = _top_counter(reason_counter_zero, 8)
    nonzero_reason_top = _top_counter(reason_counter_nonzero, 8)

    recommendations: List[str] = []
    if not zero_trade_df.empty:
        recommendations.append(
            "Promote zero-trade weeks to a first-class entry audit using shadow_meta_candidates, not only trade_outcomes."
        )
    if len(candidate_rich_zero_df) == len(zero_trade_df) and len(zero_trade_df) > 0:
        recommendations.append(
            "Zero-trade weeks are candidate-rich but acceptance-null; prioritize skipability/gating calibration before adding new model families."
        )
    if zero_trade_reason_top.get("flat_dominant", 0) > 0:
        recommendations.append(
            "Track flat_dominant and flat_veto reason mix by week and session to separate genuine no-edge regimes from over-conservative filtering."
        )
    if trade_count_ratio is not None and trade_count_ratio >= 4.0:
        recommendations.append(
            "Add an explicit weekly trade-count dispersion sanity gate before blessing a new canonical entry line."
        )

    summary = {
        "reports_root": str(reports_root),
        "runs_root": str(runs_root),
        "run_dir_count": int(len(run_dirs)),
        "completed_runs": int(len(completed_df)),
        "incomplete_runs": int(len(incomplete_df)),
        "incomplete_run_ids": incomplete_df["run_id"].astype("string").tolist() if not incomplete_df.empty else [],
        "completed_zero_trade_runs": int(len(zero_trade_df)),
        "completed_nonzero_trade_runs": int(len(nonzero_df)),
        "zero_trade_share": zero_trade_share,
        "trade_count_stats_completed_nonzero": _stats(nonzero_df["trade_count"].tolist()) if not nonzero_df.empty else _stats([]),
        "candidate_count_stats_completed_zero": _stats(zero_trade_df["candidate_rows"].tolist()) if not zero_trade_df.empty else _stats([]),
        "candidate_count_stats_completed_nonzero": _stats(nonzero_df["candidate_rows"].tolist()) if not nonzero_df.empty else _stats([]),
        "acceptance_rate_stats_completed_zero": _stats(zero_trade_df["acceptance_rate"].tolist()) if not zero_trade_df.empty else _stats([]),
        "acceptance_rate_stats_completed_nonzero": _stats(nonzero_df["acceptance_rate"].tolist()) if not nonzero_df.empty else _stats([]),
        "max_to_median_nonzero_trade_count_ratio": trade_count_ratio,
        "candidate_rich_zero_trade_runs": int(len(candidate_rich_zero_df)),
        "candidate_rich_zero_trade_run_ids": candidate_rich_zero_df["run_id"].astype("string").tolist() if not candidate_rich_zero_df.empty else [],
        "zero_trade_run_ids": zero_trade_df["run_id"].astype("string").tolist() if not zero_trade_df.empty else [],
        "zero_trade_clusters_v1": zero_trade_clusters,
        "zero_trade_runs_detail_sample": (
            zero_trade_df.sort_values(["candidate_rows", "run_id"], ascending=[False, True], kind="mergesort")
            .head(sample_limit)
            .to_dict(orient="records")
            if not zero_trade_df.empty
            else []
        ),
        "top_nonzero_trade_weeks": (
            nonzero_df.sort_values(["trade_count", "run_id"], ascending=[False, True], kind="mergesort")
            .head(sample_limit)
            .to_dict(orient="records")
            if not nonzero_df.empty
            else []
        ),
        "zero_trade_reason_mix_top8": zero_trade_reason_top,
        "nonzero_reason_mix_top8": nonzero_reason_top,
        "zero_trade_session_mix": {key: int(value) for key, value in session_counter_zero.items()},
        "nonzero_session_mix": {key: int(value) for key, value in session_counter_nonzero.items()},
        "verdicts": {
            "zero_trade_presence_status": _status(len(zero_trade_df) == 0),
            "zero_trade_candidate_surface_status": _status(
                len(zero_trade_df) == 0 or len(candidate_rich_zero_df) != len(zero_trade_df)
            ),
            "zero_trade_acceptance_status": _status(
                len(zero_trade_df) == 0 or float(zero_trade_df["acceptance_rate"].max()) > 0.0
            ),
            "trade_count_dispersion_status": _status(trade_count_ratio is not None and trade_count_ratio < 4.0),
        },
        "recommendations_v1": recommendations,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit truth replay root for entry skipability pressure")
    parser.add_argument("--reports-root", default=None, help="Truth replay root; defaults to ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--sample-limit", type=int, default=10, help="Max rows per sample section")
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    summary = build_skipability_pressure_summary(reports_root=reports_root, sample_limit=max(1, int(args.sample_limit)))
    payload = json.dumps(summary, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
