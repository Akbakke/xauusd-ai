#!/usr/bin/env python3
"""Audit active Entry-bound Exit per-bar reconstruction.

This gate verifies the reconstructed HOLD/EXIT_NOW per-bar substrate from the
active Entry/IQL replay evidence. It is report-only: it never trains Exit,
never runs replay or IQL, and never opens shadow/live/promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.audit_entry_exit_handoff_readiness_v1 import REQUIRED_EXIT_SUBSTRATE_FIELDS
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_HANDOFF_JSON = (
    REPORTS_ROOT / "entry_exit_per_bar_handoff_20260630_v1/ENTRY_EXIT_PER_BAR_HANDOFF_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_per_bar_reconstruction_audit_20260630_v1"
ALLOWED_HANDOFF_DECISIONS = {"PASS", "PASS_WITH_EXPLICIT_GAP_EXCLUSIONS"}

REQUIRED_RECONSTRUCTION_FIELDS = tuple(
    dict.fromkeys(
        (
            *REQUIRED_EXIT_SUBSTRATE_FIELDS,
            "bar_price_source",
            "bar_price_source_path",
            "entry_time",
            "exit_time",
            "realized_net_pnl_bps",
            "realized_gross_pnl_bps",
            "realized_mfe_bps",
            "realized_mae_bps",
            "realized_exit_reason",
            "is_realized_exit_bar",
        )
    )
)
NUMERIC_FINITE_FIELDS = (
    "bar_index",
    "bars_held",
    "running_pnl_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "running_giveback_bps",
    "spread_bps",
    "atr_bps",
    "entry_score",
    "entry_p_long",
    "entry_p_short",
    "entry_p_flat",
    "entry_path_quality_pred",
    "entry_bad_path_prob",
    "realized_net_pnl_bps",
    "realized_gross_pnl_bps",
    "realized_mfe_bps",
    "realized_mae_bps",
)
PROBABILITY_FIELDS = ("entry_p_long", "entry_p_short", "entry_p_flat")
CATEGORY_FIELDS = (
    "session",
    "vol_regime",
    "side",
    "realized_exit_reason",
    "entry_iql_policy_id",
    "entry_replay_identity_hash",
)
CONSTANT_PER_TRADE_FIELDS = (
    "entry_time",
    "exit_time",
    "realized_net_pnl_bps",
    "realized_gross_pnl_bps",
    "realized_mfe_bps",
    "realized_mae_bps",
    "realized_exit_reason",
    "entry_iql_policy_id",
    "entry_replay_identity_hash",
)
NEXT_CONTRACT_FIELDS = ("exit_now_label", "hold_reward_bps", "exit_now_reward_bps")


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


def _path_from_report(report: dict[str, Any], key: str) -> Path:
    raw = str(report.get(key) or "").strip()
    return Path(raw).expanduser().resolve() if raw else Path("")


def _nonempty_string_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": "", "None": "", "NaT": ""})


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _numeric_review(frame: pd.DataFrame) -> dict[str, Any]:
    field_reviews: dict[str, dict[str, Any]] = {}
    all_finite = True
    for field in NUMERIC_FINITE_FIELDS:
        if field not in frame.columns:
            all_finite = False
            field_reviews[field] = {"present": False, "finite": False}
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
        finite_count = int(finite.sum())
        missing_count = int(len(values) - finite_count)
        field_ok = missing_count == 0
        all_finite = all_finite and field_ok
        field_reviews[field] = {
            "present": True,
            "finite": field_ok,
            "finite_count": finite_count,
            "missing_or_nonfinite_count": missing_count,
            "min": float(values.min()) if finite_count else None,
            "max": float(values.max()) if finite_count else None,
            "unique_count": int(values.nunique(dropna=True)),
        }
    return {"all_finite": bool(all_finite), "fields": field_reviews}


def _category_review(frame: pd.DataFrame) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    all_live = True
    for field in CATEGORY_FIELDS:
        if field not in frame.columns:
            all_live = False
            fields[field] = {"present": False, "nonempty": False}
            continue
        values = _nonempty_string_series(frame[field])
        nonempty_count = int((values != "").sum())
        unknown_count = int(values.str.upper().isin({"UNKNOWN", "NONE", "NAN", ""}).sum())
        unique_values = sorted(value for value in values.unique().tolist() if value)
        field_live = nonempty_count == len(frame) and unknown_count < len(frame)
        all_live = all_live and field_live
        fields[field] = {
            "present": True,
            "nonempty": bool(nonempty_count == len(frame)),
            "nonempty_count": nonempty_count,
            "unknown_count": unknown_count,
            "unique_count": int(len(unique_values)),
            "sample_values": unique_values[:12],
        }
    return {"all_live": bool(all_live), "fields": fields}


def _probability_review(frame: pd.DataFrame, *, max_sum_error: float) -> dict[str, Any]:
    if any(field not in frame.columns for field in PROBABILITY_FIELDS):
        return {"ready": False, "missing_probability_fields": [field for field in PROBABILITY_FIELDS if field not in frame.columns]}
    probs = frame.loc[:, PROBABILITY_FIELDS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(probs.to_numpy(dtype="float64", na_value=np.nan)).all(axis=1)
    bounded = ((probs >= 0.0) & (probs <= 1.0)).all(axis=1)
    sums = probs.sum(axis=1)
    sum_error = (sums - 1.0).abs()
    ready = bool(finite.all() and bounded.all() and (sum_error <= max_sum_error).all())
    return {
        "ready": ready,
        "finite_rows": int(finite.sum()),
        "bounded_rows": int(bounded.sum()),
        "max_sum_error": float(sum_error.max()) if len(sum_error) else None,
        "allowed_max_sum_error": float(max_sum_error),
        "bad_rows": int((~(finite & bounded & (sum_error <= max_sum_error))).sum()),
    }


def _atr_liveness_review(frame: pd.DataFrame) -> dict[str, Any]:
    if "atr_bps" not in frame.columns:
        return {"ready": False, "present": False}
    values = pd.to_numeric(frame["atr_bps"], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
    positive = values > 0.0
    unique_count = int(values.round(8).nunique(dropna=True))
    enough_variation = unique_count >= 2 if len(values) >= 2 else unique_count == 1
    ready = bool(finite.all() and positive.all() and enough_variation)
    return {
        "ready": ready,
        "present": True,
        "finite_count": int(finite.sum()),
        "positive_count": int(positive.sum()),
        "row_count": int(len(values)),
        "unique_count": unique_count,
        "min": float(values.min()) if finite.any() else None,
        "max": float(values.max()) if finite.any() else None,
    }


def _per_trade_review(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "entry_trade_id" not in frame.columns:
        return {"ready": False, "trade_count": 0, "failures": [{"entry_trade_id": "", "reason": "empty_or_missing_entry_trade_id"}]}
    working = frame.copy()
    working["bar_index_num"] = pd.to_numeric(working["bar_index"], errors="coerce")
    working["bars_held_num"] = pd.to_numeric(working["bars_held"], errors="coerce")
    working["bar_ts_dt"] = pd.to_datetime(working["bar_ts"], utc=True, errors="coerce")
    working["entry_time_dt"] = pd.to_datetime(working["entry_time"], utc=True, errors="coerce")
    working["exit_time_dt"] = pd.to_datetime(working["exit_time"], utc=True, errors="coerce")
    working["terminal_bool"] = _bool_series(working["is_realized_exit_bar"])
    failures: list[dict[str, Any]] = []
    five_minutes = pd.Timedelta(minutes=5)
    for entry_trade_id, group in working.groupby("entry_trade_id", sort=False):
        reasons: list[str] = []
        bar_index = group["bar_index_num"].astype("Int64").tolist()
        expected = list(range(len(group)))
        if bar_index != expected:
            reasons.append("bar_index_not_contiguous_from_zero")
        if not group["bars_held_num"].equals(group["bar_index_num"]):
            reasons.append("bars_held_does_not_match_bar_index")
        terminal_count = int(group["terminal_bool"].sum())
        if terminal_count != 1:
            reasons.append("terminal_bar_count_not_one")
        elif not bool(group["terminal_bool"].iloc[-1]):
            reasons.append("terminal_bar_not_last_row")
        if group["bar_ts_dt"].isna().any() or group["entry_time_dt"].isna().any() or group["exit_time_dt"].isna().any():
            reasons.append("invalid_timestamp")
        else:
            if not bool(group["bar_ts_dt"].is_monotonic_increasing):
                reasons.append("bar_ts_not_monotonic")
            diffs = group["bar_ts_dt"].diff().dropna()
            if not bool((diffs == five_minutes).all()):
                reasons.append("bar_ts_not_contiguous_5min")
            if group["bar_ts_dt"].iloc[0] != group["entry_time_dt"].iloc[0]:
                reasons.append("first_bar_ts_not_entry_time")
            if group["bar_ts_dt"].iloc[-1] != group["exit_time_dt"].iloc[-1]:
                reasons.append("last_bar_ts_not_exit_time")
        for field in CONSTANT_PER_TRADE_FIELDS:
            if field in group.columns and group[field].nunique(dropna=False) > 1:
                reasons.append(f"{field}_not_constant_per_trade")
        if reasons:
            failures.append(
                {
                    "entry_trade_id": str(entry_trade_id),
                    "reason": ",".join(sorted(set(reasons))),
                    "row_count": int(len(group)),
                    "first_bar_ts": str(group["bar_ts"].iloc[0]) if "bar_ts" in group.columns else "",
                    "last_bar_ts": str(group["bar_ts"].iloc[-1]) if "bar_ts" in group.columns else "",
                }
            )
    return {
        "ready": not failures,
        "trade_count": int(working["entry_trade_id"].nunique(dropna=True)),
        "failure_count": int(len(failures)),
        "failures": failures[:50],
    }


def _exclusion_review(frame: pd.DataFrame, handoff: dict[str, Any]) -> dict[str, Any]:
    exclusions_raw = str(handoff.get("gap_exclusions_csv") or "").strip()
    exclusions_path = Path(exclusions_raw).expanduser().resolve() if exclusions_raw else Path("")
    excluded_count = int(handoff.get("excluded_trade_count") or 0)
    if excluded_count == 0:
        return {"ready": True, "excluded_trade_count": 0, "gap_exclusions_csv": str(exclusions_path) if exclusions_raw else ""}
    if not exclusions_raw:
        return {"ready": False, "excluded_trade_count": excluded_count, "gap_exclusions_csv": "", "reason": "missing_gap_exclusions_csv_path"}
    if not exclusions_path.exists():
        return {"ready": False, "excluded_trade_count": excluded_count, "gap_exclusions_csv": str(exclusions_path), "reason": "missing_gap_exclusions_csv"}
    exclusions = pd.read_csv(exclusions_path)
    if "entry_trade_id" not in exclusions.columns:
        return {"ready": False, "excluded_trade_count": excluded_count, "gap_exclusions_csv": str(exclusions_path), "reason": "gap_exclusions_missing_entry_trade_id"}
    overlap = sorted(set(frame["entry_trade_id"].astype(str)).intersection(set(exclusions["entry_trade_id"].astype(str))))
    return {
        "ready": len(overlap) == 0,
        "excluded_trade_count": excluded_count,
        "gap_exclusions_csv": str(exclusions_path),
        "overlap_count": int(len(overlap)),
        "overlap_sample": overlap[:20],
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Per-Bar Reconstruction Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Dataset rows: `{report['dataset_rows']}`",
        f"- Included trade count: `{report['included_trade_count']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit IQL allowed: `{report['exit_iql_allowed']}`",
        f"- Dataset: `{report['dataset_csv']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
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
    handoff_json = Path(args.handoff_json).expanduser().resolve()
    handoff = _read_json_or_empty(handoff_json)
    dataset_path = _path_from_report(handoff, "dataset_csv")
    dataset_exists = bool(str(handoff.get("dataset_csv") or "").strip()) and dataset_path.exists()
    dataset = pd.read_csv(dataset_path, low_memory=False) if dataset_exists else pd.DataFrame()
    missing_fields = [field for field in REQUIRED_RECONSTRUCTION_FIELDS if field not in set(dataset.columns)]
    included_trade_count = int(handoff.get("included_trade_count") or handoff.get("complete_trade_count") or 0)
    source_trade_count = int(handoff.get("source_trade_count") or handoff.get("trade_count") or 0)
    covered_trade_ratio = float(handoff.get("covered_trade_ratio") or 0.0)
    min_covered_trade_ratio = float(args.min_covered_trade_ratio)
    min_included_trades = int(args.min_included_trades)
    numeric_review = _numeric_review(dataset) if not dataset.empty else {"all_finite": False, "fields": {}}
    category_review = _category_review(dataset) if not dataset.empty else {"all_live": False, "fields": {}}
    probability_review = (
        _probability_review(dataset, max_sum_error=float(args.max_probability_sum_error))
        if not dataset.empty
        else {"ready": False}
    )
    atr_liveness = _atr_liveness_review(dataset) if not dataset.empty else {"ready": False}
    per_trade_review = _per_trade_review(dataset) if not dataset.empty else {"ready": False, "trade_count": 0, "failures": []}
    exclusion_review = _exclusion_review(dataset, handoff) if not dataset.empty else {"ready": False}
    observed_trade_count = int(dataset["entry_trade_id"].nunique(dropna=True)) if "entry_trade_id" in dataset.columns else 0
    action_values = sorted(_nonempty_string_series(dataset["action_set"]).unique().tolist()) if "action_set" in dataset.columns else []
    side_values = sorted(_nonempty_string_series(dataset["side"]).str.upper().unique().tolist()) if "side" in dataset.columns else []
    next_contract_fields_present = [field for field in NEXT_CONTRACT_FIELDS if field in dataset.columns]
    checks = [
        _check("active per-bar handoff report exists", handoff_json.exists(), {"path": str(handoff_json)}),
        _check(
            "active per-bar handoff decision is PASS or explicit gap-exclusion PASS",
            str(handoff.get("decision")) in ALLOWED_HANDOFF_DECISIONS,
            {"decision": handoff.get("decision"), "allowed": sorted(ALLOWED_HANDOFF_DECISIONS)},
        ),
        _check("active per-bar handoff report has no failures", not handoff.get("failures"), {"failures": handoff.get("failures")}),
        _check("active per-bar dataset exists", dataset_exists, {"dataset_csv": str(dataset_path)}),
        _check("active per-bar dataset has rows", not dataset.empty, {"dataset_rows": int(len(dataset))}),
        _check(
            "active per-bar coverage meets reconstruction floor",
            covered_trade_ratio >= min_covered_trade_ratio and included_trade_count >= min_included_trades,
            {
                "covered_trade_ratio": covered_trade_ratio,
                "min_covered_trade_ratio": min_covered_trade_ratio,
                "included_trade_count": included_trade_count,
                "min_included_trades": min_included_trades,
                "source_trade_count": source_trade_count,
            },
        ),
        _check(
            "dataset trade count matches included handoff trades",
            observed_trade_count == included_trade_count and observed_trade_count > 0,
            {"observed_trade_count": observed_trade_count, "included_trade_count": included_trade_count},
        ),
        _check(
            "per-bar reconstruction fields satisfy active Exit substrate contract",
            not missing_fields,
            {"missing_fields": missing_fields, "required_fields": list(REQUIRED_RECONSTRUCTION_FIELDS)},
        ),
        _check("per-bar numeric state fields are finite", bool(numeric_review.get("all_finite")), numeric_review),
        _check("entry probabilities are bounded and sum to one", bool(probability_review.get("ready")), probability_review),
        _check("atr_bps is positive and live", bool(atr_liveness.get("ready")), atr_liveness),
        _check("categorical provenance fields are live", bool(category_review.get("all_live")), category_review),
        _check(
            "action set is exactly HOLD,EXIT_NOW",
            action_values == ["HOLD,EXIT_NOW"],
            {"observed_action_values": action_values},
        ),
        _check(
            "side values are valid LONG/SHORT states",
            bool(side_values) and set(side_values).issubset({"LONG", "SHORT"}),
            {"observed_side_values": side_values},
        ),
        _check("per-trade timeline reconstruction is contiguous and terminal", bool(per_trade_review.get("ready")), per_trade_review),
        _check("explicit gap exclusions are not present in the reconstructed dataset", bool(exclusion_review.get("ready")), exclusion_review),
        _check(
            "reconstruction audit never trains, replays, distills, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "replay_started": False,
                "iql_distillation_started": False,
                "exit_training_allowed": False,
                "exit_iql_allowed": False,
                "promotion_shadow_live_allowed": False,
            },
        ),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = "READY_FOR_EXIT_STATE_REWARD_CONTRACT_REVIEW" if ready else "BLOCKED_BY_EXIT_RECONSTRUCTION_AUDIT"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_PER_BAR_RECONSTRUCTION_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_PER_BAR_RECONSTRUCTION_AUDIT_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_per_bar_reconstruction_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "handoff_json": str(handoff_json),
        "handoff_json_sha256": _sha256_file(handoff_json) if handoff_json.exists() else "",
        "handoff_decision": handoff.get("decision"),
        "dataset_csv": str(dataset_path) if dataset_exists else "",
        "dataset_csv_sha256": _sha256_file(dataset_path) if dataset_exists else "",
        "dataset_rows": int(len(dataset)),
        "source_trade_count": source_trade_count,
        "included_trade_count": included_trade_count,
        "observed_trade_count": observed_trade_count,
        "covered_trade_ratio": covered_trade_ratio,
        "min_covered_trade_ratio": min_covered_trade_ratio,
        "min_included_trades": min_included_trades,
        "required_reconstruction_fields": list(REQUIRED_RECONSTRUCTION_FIELDS),
        "missing_reconstruction_fields": missing_fields,
        "numeric_review": numeric_review,
        "probability_review": probability_review,
        "atr_liveness": atr_liveness,
        "category_review": category_review,
        "per_trade_review": per_trade_review,
        "gap_exclusion_review": exclusion_review,
        "next_contract_fields_present": next_contract_fields_present,
        "next_contract_fields_required_later": list(NEXT_CONTRACT_FIELDS),
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "materialize active Exit state/reward contract before any Exit Transformer/IQL training"
            if ready
            else "repair active per-bar Exit reconstruction before state/reward contract review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_PER_BAR_RECONSTRUCTION_AUDIT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_PER_BAR_RECONSTRUCTION_AUDIT_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "dataset_rows": int(len(dataset)),
                    "included_trade_count": included_trade_count,
                    "observed_trade_count": observed_trade_count,
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
    ap.add_argument("--handoff-json", default=str(DEFAULT_HANDOFF_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-covered-trade-ratio", type=float, default=0.95)
    ap.add_argument("--min-included-trades", type=int, default=100)
    ap.add_argument("--max-probability-sum-error", type=float, default=0.05)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
