#!/usr/bin/env python3
"""Materialize Entry candidate offline replay evidence.

This script consumes an explicit trade-level replay log and writes the
`replay_policy_metrics.csv` and `replay_policy_monthly.csv` artifacts required
by Entry replay-readiness. It does not run replay, train, promote, shadow, live,
or select implicit latest/legacy artifacts.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_candidate_replay_20260628_v1"
DEFAULT_CANDIDATE_BUNDLE_AUDIT = (
    REPORTS_ROOT / "entry_candidate_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
)
DEFAULT_SELECTIVE_EDGE_SUMMARY = REPORTS_ROOT / "entry_candidate_selective_edge_20260628_v1/summary.json"

IQL_TRANSITION_REQUIRED_COLUMNS = (
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
)

IQL_TRANSITION_NUMERIC_COLUMNS = (
    "score",
    "p_long",
    "p_short",
    "p_flat",
    "net_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "held_bars",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        out = float(obj)
        return out if np.isfinite(out) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing replay trades file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise RuntimeError(f"unsupported replay trades extension: {path.suffix}")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_contract(
    *,
    candidate_bundle_audit_path: Path,
    selective_edge_summary_path: Path,
    require_identity_artifacts: bool,
) -> dict[str, Any]:
    candidate_audit = _read_json_if_exists(candidate_bundle_audit_path)
    selective_summary = _read_json_if_exists(selective_edge_summary_path)
    failures: list[str] = []
    if require_identity_artifacts and not candidate_bundle_audit_path.exists():
        failures.append(f"missing candidate bundle audit: {candidate_bundle_audit_path}")
    if require_identity_artifacts and not selective_edge_summary_path.exists():
        failures.append(f"missing selective-edge summary: {selective_edge_summary_path}")
    if candidate_audit and str(candidate_audit.get("decision")) != "PASS":
        failures.append(f"candidate bundle audit decision is not PASS: {candidate_audit.get('decision')}")
    if selective_summary and str(selective_summary.get("decision")) != "PASS":
        failures.append(f"selective-edge summary decision is not PASS: {selective_summary.get('decision')}")

    candidate_bundle_dir = str(candidate_audit.get("bundle_dir") or "")
    selective_bundle_dir = str(selective_summary.get("bundle_dir") or "")
    if require_identity_artifacts and not candidate_bundle_dir:
        failures.append("candidate bundle audit does not declare bundle_dir")
    if require_identity_artifacts and not selective_bundle_dir:
        failures.append("selective-edge summary does not declare bundle_dir")
    if candidate_bundle_dir and selective_bundle_dir and candidate_bundle_dir != selective_bundle_dir:
        failures.append(
            "selective-edge bundle_dir does not match candidate bundle audit: "
            f"{selective_bundle_dir} != {candidate_bundle_dir}"
        )

    return {
        "ready": not failures,
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_edge_summary_path),
        "candidate_bundle_dir": candidate_bundle_dir,
        "selective_edge_bundle_dir": selective_bundle_dir,
        "no_xgb_bundle_dir": str(selective_summary.get("no_xgb_bundle_dir") or ""),
        "candidate_audit_decision": str(candidate_audit.get("decision") or ""),
        "selective_edge_decision": str(selective_summary.get("decision") or ""),
        "require_identity_artifacts": bool(require_identity_artifacts),
        "failures": failures,
    }


def _first_present(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _safe_mean(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def _safe_percentile(values: pd.Series, q: float) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, q)) if vals.size else None


def _profit_factor(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    gains = float(vals[vals > 0.0].sum())
    losses = float(vals[vals < 0.0].sum())
    if losses == 0.0:
        return None
    return float(gains / abs(losses))


def _max_drawdown(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(np.float64)
    if vals.size == 0:
        return 0.0, 0.0
    equity = np.concatenate([[0.0], np.cumsum(vals)])
    dd = equity - np.maximum.accumulate(equity)
    signed = float(np.min(dd))
    return abs(signed), signed


def normalize_trades(raw: pd.DataFrame, *, policy_id: str, require_year: int | None, allow_non_2026: bool) -> tuple[pd.DataFrame, list[str]]:
    failures: list[str] = []
    if raw.empty:
        raise RuntimeError("replay trades input is empty")

    time_col = _first_present(raw, ["entry_time", "entry_ts", "time", "open_time", "timestamp", "decision_time"])
    pnl_col = _first_present(raw, ["net_pnl_bps", "realized_pnl_bps", "pnl_bps", "gross_pnl_bps"])
    if time_col is None:
        raise RuntimeError("replay trades input needs an entry time column")
    if pnl_col is None:
        raise RuntimeError("replay trades input needs a PnL bps column")

    out = raw.copy()
    out["entry_time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out["net_pnl_bps"] = pd.to_numeric(out[pnl_col], errors="coerce")
    if "gross_pnl_bps" not in out.columns:
        out["gross_pnl_bps"] = out["net_pnl_bps"]
    else:
        out["gross_pnl_bps"] = pd.to_numeric(out["gross_pnl_bps"], errors="coerce")

    out = out.dropna(subset=["entry_time", "net_pnl_bps"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("replay trades input has no valid entry_time/net_pnl_bps rows")

    if "policy_id" not in out.columns:
        out["policy_id"] = str(policy_id)
    out["policy_id"] = out["policy_id"].fillna(str(policy_id)).astype(str)
    if "fold" not in out.columns:
        out["fold"] = "2026"
    out["fold"] = out["fold"].fillna("2026").astype(str)
    if "side" not in out.columns:
        out["side"] = "UNKNOWN"
    out["side"] = out["side"].fillna("UNKNOWN").astype(str).str.upper()
    if "session" not in out.columns:
        out["session"] = "UNKNOWN"
    out["session"] = out["session"].fillna("UNKNOWN").astype(str).str.upper()
    if "score" not in out.columns:
        out["score"] = np.nan
    if "mfe_bps" not in out.columns:
        out["mfe_bps"] = np.nan
    if "mae_bps" not in out.columns:
        out["mae_bps"] = np.nan
    if "direction_correct" not in out.columns:
        label_col = _first_present(out, ["label", "y_direction", "target_direction"])
        if label_col is not None:
            out["direction_correct"] = out["side"].astype(str).str.upper() == out[label_col].astype(str).str.upper()
        else:
            out["direction_correct"] = np.nan
    out["direction_correct"] = pd.Series(out["direction_correct"]).map(
        lambda x: (str(x).strip().lower() == "true") if str(x).strip().lower() in {"true", "false"} else x
    )
    out["entry_day"] = out["entry_time"].dt.strftime("%Y-%m-%d")
    out["entry_month"] = out["entry_time"].dt.strftime("%Y-%m")

    if require_year is not None:
        years = set(int(x) for x in out["entry_time"].dt.year.dropna().astype(int).unique())
        if int(require_year) not in years:
            failures.append(f"no trades in required replay year {require_year}")
        if not allow_non_2026 and years != {int(require_year)}:
            failures.append(f"replay trades contain years outside {require_year}: {sorted(years)}")

    keep = [
        "fold",
        "policy_id",
        "session",
        "entry_day",
        "entry_month",
        "entry_time",
        "side",
        "direction_correct",
        "score",
        "gross_pnl_bps",
        "net_pnl_bps",
        "mfe_bps",
        "mae_bps",
    ]
    optional_prefixes = (
        "foundation_",
        "specialist_",
        "ctx_",
        "state_",
        "teacher_",
        "candidate_",
    )
    for optional in (
        "candidate_uid",
        "trade_uid",
        "exit_time",
        "exit_reason",
        "held_bars",
        "horizon_bars",
        "p_long",
        "p_short",
        "p_flat",
        "path_quality_pred",
        "bad_path_prob",
        "tradable_prob",
    ):
        if optional in out.columns and optional not in keep:
            keep.append(optional)
    for col in out.columns:
        if any(str(col).startswith(prefix) for prefix in optional_prefixes) and col not in keep:
            keep.append(str(col))
    return out[[c for c in keep if c in out.columns]].copy(), failures


def _safe_value_counts(frame: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in frame.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in frame[col].fillna("UNKNOWN").astype(str).str.upper().value_counts(dropna=False).to_dict().items()
    }


def audit_iql_transition_trades(trades: pd.DataFrame) -> dict[str, Any]:
    missing = [col for col in IQL_TRANSITION_REQUIRED_COLUMNS if col not in trades.columns]
    failures: list[str] = []
    if missing:
        failures.append(f"IQL transition trade log missing required columns: {missing}")

    numeric_status: dict[str, dict[str, Any]] = {}
    for col in IQL_TRANSITION_NUMERIC_COLUMNS:
        if col not in trades.columns:
            continue
        values = pd.to_numeric(trades[col], errors="coerce")
        arr = values.to_numpy(dtype=np.float64)
        finite = bool(np.isfinite(arr).all()) if len(arr) else False
        null_count = int(values.isna().sum())
        numeric_status[col] = {
            "finite": finite,
            "null_count": null_count,
            "min": float(values.min()) if len(values) and values.notna().any() else None,
            "max": float(values.max()) if len(values) and values.notna().any() else None,
        }
        if not finite or null_count > 0:
            failures.append(f"IQL transition numeric column not fully finite: {col}")

    for prob_col in ("p_long", "p_short", "p_flat"):
        if prob_col in trades.columns:
            values = pd.to_numeric(trades[prob_col], errors="coerce")
            if bool(((values < 0.0) | (values > 1.0)).any()):
                failures.append(f"IQL transition probability column outside [0,1]: {prob_col}")

    if {"p_long", "p_short", "p_flat"}.issubset(trades.columns):
        prob_sum = (
            pd.to_numeric(trades["p_long"], errors="coerce")
            + pd.to_numeric(trades["p_short"], errors="coerce")
            + pd.to_numeric(trades["p_flat"], errors="coerce")
        )
        probability_sum_max_abs_error = float((prob_sum - 1.0).abs().max()) if len(prob_sum) else float("inf")
        if not np.isfinite(probability_sum_max_abs_error) or probability_sum_max_abs_error > 0.05:
            failures.append(
                "IQL transition probability sum drifts from 1.0: "
                f"max_abs_error={probability_sum_max_abs_error}"
            )
    else:
        probability_sum_max_abs_error = None

    session_counts = _safe_value_counts(trades, "session")
    if session_counts and set(session_counts) <= {"UNKNOWN"}:
        failures.append("IQL transition session state is all UNKNOWN")

    side_counts = _safe_value_counts(trades, "side")
    valid_side_rows = 0
    if "side" in trades.columns:
        valid_side_rows = int(trades["side"].astype(str).str.upper().isin(["LONG", "SHORT"]).sum())
        if valid_side_rows <= 0:
            failures.append("IQL transition action side has no LONG/SHORT rows")

    return {
        "ready": not failures,
        "required_columns": list(IQL_TRANSITION_REQUIRED_COLUMNS),
        "missing_columns": missing,
        "numeric_status": numeric_status,
        "probability_sum_max_abs_error": probability_sum_max_abs_error,
        "session_counts": session_counts,
        "side_counts": side_counts,
        "valid_side_rows": valid_side_rows,
        "failures": failures,
    }


def _metrics_row(scope: str, fold: str, policy_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["net_pnl_bps"], errors="coerce")
    gross = pd.to_numeric(frame["gross_pnl_bps"], errors="coerce")
    dd_abs, dd_signed = _max_drawdown(pnl)
    direction = pd.to_numeric(frame.get("direction_correct", pd.Series(dtype=float)), errors="coerce")
    return {
        "scope": scope,
        "fold": fold,
        "policy_id": policy_id,
        "n_trades": int(len(frame)),
        "n_days": int(frame["entry_day"].nunique()),
        "n_months": int(frame["entry_month"].nunique()),
        "net_sum_bps": float(pnl.sum()),
        "net_mean_bps": _safe_mean(pnl),
        "net_median_bps": _safe_percentile(pnl, 50),
        "net_p10_bps": _safe_percentile(pnl, 10),
        "net_p90_bps": _safe_percentile(pnl, 90),
        "gross_mean_bps": _safe_mean(gross),
        "win_rate": float((pnl > 0.0).mean()) if len(pnl) else None,
        "profit_factor": _profit_factor(pnl),
        "max_win_bps": float(pnl.max()) if len(pnl) else None,
        "max_loss_bps": float(pnl.min()) if len(pnl) else None,
        "max_drawdown_bps": dd_abs,
        "max_drawdown_signed_bps": dd_signed,
        "mean_score": _safe_mean(frame["score"]),
        "mean_mfe_bps": _safe_mean(frame["mfe_bps"]),
        "mean_mae_bps": _safe_mean(frame["mae_bps"]),
        "long_rate": float((frame["side"].astype(str).str.upper() == "LONG").mean()),
        "short_rate": float((frame["side"].astype(str).str.upper() == "SHORT").mean()),
        "direction_precision": _safe_mean(direction),
        "avg_trades_per_day": float(len(frame) / max(frame["entry_day"].nunique(), 1)),
    }


def build_replay_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for (policy_id, fold), frame in trades.groupby(["policy_id", "fold"], sort=True):
        metric_rows.append(_metrics_row("fold", str(fold), str(policy_id), frame))
    for policy_id, frame in trades.groupby("policy_id", sort=True):
        metric_rows.append(_metrics_row("aggregate", "ALL", str(policy_id), frame))
    metrics = pd.DataFrame(metric_rows)

    daily = (
        trades.groupby(["policy_id", "entry_day"], as_index=False)
        .agg(
            n_trades=("net_pnl_bps", "size"),
            net_sum_bps=("net_pnl_bps", "sum"),
            net_mean_bps=("net_pnl_bps", "mean"),
            wins=("net_pnl_bps", lambda s: int((s > 0.0).sum())),
        )
    )
    daily["win_rate"] = daily["wins"] / daily["n_trades"].clip(lower=1)

    monthly = (
        trades.groupby(["policy_id", "entry_month"], as_index=False)
        .agg(
            n_trades=("net_pnl_bps", "size"),
            net_sum_bps=("net_pnl_bps", "sum"),
            net_mean_bps=("net_pnl_bps", "mean"),
            wins=("net_pnl_bps", lambda s: int((s > 0.0).sum())),
        )
    )
    monthly["month"] = monthly["entry_month"]
    monthly["win_rate"] = monthly["wins"] / monthly["n_trades"].clip(lower=1)
    return metrics, daily, monthly


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Candidate Replay Evidence",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Trades: `{report['n_trades']}`",
        f"- Out dir: `{report['out_dir']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trades_path = Path(args.trades_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    candidate_bundle_audit_path = Path(args.candidate_bundle_audit_json).expanduser().resolve()
    selective_edge_summary_path = Path(args.selective_edge_summary_json).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_table(trades_path)
    identity = _identity_contract(
        candidate_bundle_audit_path=candidate_bundle_audit_path,
        selective_edge_summary_path=selective_edge_summary_path,
        require_identity_artifacts=bool(args.require_identity_artifacts),
    )
    require_year = None if args.require_year <= 0 else int(args.require_year)
    trades, failures = normalize_trades(
        raw,
        policy_id=str(args.policy_id),
        require_year=require_year,
        allow_non_2026=bool(args.allow_non_2026),
    )
    failures.extend(identity["failures"])
    metrics, daily, monthly = build_replay_tables(trades)
    iql_transition_audit = audit_iql_transition_trades(trades)
    if bool(args.require_iql_transition_fields):
        failures.extend(iql_transition_audit["failures"])

    best = metrics[metrics["scope"].astype(str).isin(["aggregate", "all", "ALL"])]
    if best.empty:
        failures.append("no aggregate replay metrics were produced")
    else:
        row = best.sort_values("net_sum_bps", ascending=False).iloc[0]
        if int(row.get("n_trades") or 0) <= 0:
            failures.append("aggregate replay metrics have zero trades")

    trades_out = out_dir / "replay_policy_trades.csv"
    metrics_out = out_dir / "replay_policy_metrics.csv"
    daily_out = out_dir / "replay_policy_daily.csv"
    monthly_out = out_dir / "replay_policy_monthly.csv"
    summary_out = out_dir / "summary.json"
    manifest_out = out_dir / "REPLAY_EVIDENCE_MANIFEST.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = out_dir / f"ENTRY_CANDIDATE_REPLAY_EVIDENCE_{timestamp}.json"
    report_md = out_dir / f"ENTRY_CANDIDATE_REPLAY_EVIDENCE_{timestamp}.md"

    trades.to_csv(trades_out, index=False)
    metrics.to_csv(metrics_out, index=False)
    daily.to_csv(daily_out, index=False)
    monthly.to_csv(monthly_out, index=False)

    best_row = best.sort_values("net_sum_bps", ascending=False).iloc[0].to_dict() if not best.empty else {}
    report = {
        "schema_version": "entry_candidate_replay_evidence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "trades_path": str(trades_path),
        "out_dir": str(out_dir),
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_summary_json": str(selective_edge_summary_path),
        "candidate_bundle_dir": identity["candidate_bundle_dir"],
        "no_xgb_bundle_dir": identity["no_xgb_bundle_dir"],
        "replay_identity_contract": identity,
        "required_year": require_year,
        "n_trades": int(len(trades)),
        "policies": sorted(str(x) for x in trades["policy_id"].unique()),
        "best_aggregate_row": best_row,
        "trades_csv": str(trades_out),
        "metrics_csv": str(metrics_out),
        "daily_csv": str(daily_out),
        "monthly_csv": str(monthly_out),
        "summary_json": str(summary_out),
        "manifest_json": str(manifest_out),
        "iql_transition_dataset_ready": bool(iql_transition_audit["ready"]),
        "iql_transition_contract": iql_transition_audit,
        "json_path": str(report_json),
        "md_path": str(report_md),
        "trainer_started": False,
        "replay_started": False,
        "promotion_shadow_live_allowed": False,
        "failures": failures,
    }
    summary_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    manifest_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    (out_dir / "ENTRY_CANDIDATE_REPLAY_EVIDENCE_latest.json").write_text(
        report_json.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_CANDIDATE_REPLAY_EVIDENCE_latest.md").write_text(
        report_md.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "metrics_csv": str(metrics_out),
                    "monthly_csv": str(monthly_out),
                    "json_path": str(report_json),
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades-path", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--candidate-bundle-audit-json", default=str(DEFAULT_CANDIDATE_BUNDLE_AUDIT))
    ap.add_argument("--selective-edge-summary-json", default=str(DEFAULT_SELECTIVE_EDGE_SUMMARY))
    ap.add_argument("--policy-id", default="candidate_replay")
    ap.add_argument("--require-year", type=int, default=2026)
    ap.add_argument("--allow-non-2026", action="store_true")
    ap.add_argument("--require-iql-transition-fields", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-identity-artifacts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
