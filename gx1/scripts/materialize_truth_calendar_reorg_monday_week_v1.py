#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class WeekWindow:
    run_id: str
    calendar_start_utc: str
    calendar_end_exclusive_utc: str
    trading_start_utc: str
    friday_flat_cutoff_utc: str
    weekend_no_action: bool
    quarantine_status: str
    quarantine_reason: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_time_range(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    df = pd.read_parquet(path)
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df.index, utc=True, errors="coerce")
    ts = ts.dropna()
    return {
        "exists": True,
        "path": str(path),
        "rows": int(len(df)),
        "ts_min": ts.min().isoformat() if len(ts) else None,
        "ts_max": ts.max().isoformat() if len(ts) else None,
    }


def _prebuilt_model_bar_range(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        df = pd.read_parquet(path, columns=["is_model_bar"])
    except Exception as exc:
        return {"exists": True, "path": str(path), "error": str(exc)}
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    model_mask = df["is_model_bar"].fillna(False).astype(bool).to_numpy()
    model_idx = idx[model_mask]
    return {
        "exists": True,
        "path": str(path),
        "raw_rows": int(len(df)),
        "raw_ts_min": idx.min().isoformat() if len(idx) else None,
        "raw_ts_max": idx.max().isoformat() if len(idx) else None,
        "model_bar_rows": int(model_mask.sum()),
        "model_bar_ts_min": model_idx.min().isoformat() if len(model_idx) else None,
        "model_bar_ts_max": model_idx.max().isoformat() if len(model_idx) else None,
    }


def _latest_candidate_prebuilt(output_root: Path) -> dict[str, Any]:
    manifests = sorted(output_root.glob("monday_week_prebuilt_extension_*/TRUTH_MONDAY_WEEK_PREBUILT_EXTENSION_V1.json"))
    if not manifests:
        return {"exists": False, "search_root": str(output_root)}
    manifest_path = manifests[-1]
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    expanded = Path(obj.get("expanded_raw_index_path", ""))
    return {
        "exists": True,
        "manifest_path": str(manifest_path),
        "status": obj.get("status"),
        "not_promoted": bool(obj.get("not_promoted", True)),
        "expanded_raw_index_path": str(expanded),
        "coverage": _prebuilt_model_bar_range(expanded),
        "raw_m1_rows": obj.get("raw_m1_rows"),
        "raw_m5_rows": obj.get("raw_m5_rows"),
        "expanded_rows": obj.get("expanded_rows"),
        "expanded_model_bar_rows": obj.get("expanded_model_bar_rows"),
        "expanded_sha256": obj.get("expanded_sha256"),
    }


def _completed_eof_summary(root: Path) -> dict[str, Any]:
    runs_root = root / "runs"
    if not runs_root.exists():
        return {"runs_root_exists": False}
    total_runs = 0
    completed = 0
    trades = 0
    eof = 0
    eof_runs: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        total_runs += 1
        if not (run_dir / "RUN_COMPLETED.json").exists():
            continue
        completed += 1
        rid = run_dir.name
        outcomes = run_dir / f"trade_outcomes_{rid}_MERGED.parquet"
        if not outcomes.exists():
            continue
        try:
            df = pd.read_parquet(outcomes, columns=["exit_reason"])
        except Exception:
            continue
        n = int(len(df))
        n_eof = int((df["exit_reason"].astype(str) == "REPLAY_EOF").sum())
        trades += n
        eof += n_eof
        if n_eof:
            eof_runs.append({"run_id": rid, "trades": n, "replay_eof": n_eof})
    return {
        "runs_root_exists": True,
        "total_run_dirs": int(total_runs),
        "completed_runs": int(completed),
        "trade_rows_completed": int(trades),
        "replay_eof_trades": int(eof),
        "replay_eof_runs": eof_runs,
    }


def _week_run_id(prefix: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{prefix}_{start:%Y%m%d}_{end:%Y%m%d}"


def _build_monday_weeks(
    *,
    start: str,
    end_exclusive: str,
    prefix: str,
    quarantined_week_starts: dict[str, str],
) -> list[WeekWindow]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end_exclusive, tz="UTC")
    first_monday = start_ts.normalize()
    while first_monday.weekday() != 0:
        first_monday -= pd.Timedelta(days=1)
    weeks: list[WeekWindow] = []
    cur = first_monday
    while cur < end_ts:
        nxt = cur + pd.Timedelta(days=7)
        friday_flat = cur + pd.Timedelta(days=4, hours=20, minutes=55)
        q_reason = quarantined_week_starts.get(cur.strftime("%Y-%m-%d"))
        weeks.append(
            WeekWindow(
                run_id=_week_run_id(prefix, cur, nxt),
                calendar_start_utc=cur.isoformat(),
                calendar_end_exclusive_utc=nxt.isoformat(),
                trading_start_utc=max(cur, start_ts).isoformat(),
                friday_flat_cutoff_utc=friday_flat.isoformat(),
                weekend_no_action=True,
                quarantine_status="QUARANTINED" if q_reason else "ACTIVE_CANDIDATE",
                quarantine_reason=q_reason,
            )
        )
        cur = nxt
    return weeks


def materialize(
    *,
    reports_root: Path,
    active_v2_root: Path,
    output_name: str,
    data_end_exclusive: str,
) -> dict[str, Any]:
    m1_path = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL/year=2026/part-000.parquet")
    m5_path = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL/year=2026/part-000.parquet")
    current_manifest_path = Path("/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json")
    manifest = json.loads(current_manifest_path.read_text(encoding="utf-8")) if current_manifest_path.exists() else {}
    prebuilt_path = Path(manifest.get("parquet_path", "")) if manifest.get("parquet_path") else Path("")

    quarantined = {
        "2025-12-01": "SPLIT_FROM_PATHOLOGICAL_OLD_WINDOW_20251203_20251210",
        "2025-12-08": "SPLIT_FROM_PATHOLOGICAL_OLD_WINDOW_20251203_20251210",
    }
    full_reorg_weeks = _build_monday_weeks(
        start="2025-01-01T00:00:00Z",
        end_exclusive=data_end_exclusive,
        prefix="TRUTH_MONFRI_WEEK",
        quarantined_week_starts=quarantined,
    )
    oos_weeks = [
        w
        for w in full_reorg_weeks
        if pd.Timestamp(w.calendar_start_utc) >= pd.Timestamp("2026-04-06T00:00:00Z")
    ]

    m1 = _load_time_range(m1_path)
    m5 = _load_time_range(m5_path)
    prebuilt = _prebuilt_model_bar_range(prebuilt_path)
    candidate_prebuilt = _latest_candidate_prebuilt(
        Path("/home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES")
    )
    prebuilt_model_ts_max = (
        pd.Timestamp(prebuilt["model_bar_ts_max"])
        if prebuilt.get("model_bar_ts_max")
        else pd.Timestamp.min.tz_localize("UTC")
    )
    candidate_model_ts_max = (
        pd.Timestamp(candidate_prebuilt.get("coverage", {}).get("model_bar_ts_max"))
        if candidate_prebuilt.get("coverage", {}).get("model_bar_ts_max")
        else pd.Timestamp.min.tz_localize("UTC")
    )
    blocker_reasons: list[str] = []
    candidate_blocker_reasons: list[str] = []
    target_end = pd.Timestamp(data_end_exclusive)
    target_m1_last = target_end - pd.Timedelta(minutes=1)
    target_m5_last = target_end - pd.Timedelta(minutes=5)
    m1_ts_max = pd.Timestamp(m1["ts_max"]) if m1.get("ts_max") else pd.Timestamp.min.tz_localize("UTC")
    m5_ts_max = pd.Timestamp(m5["ts_max"]) if m5.get("ts_max") else pd.Timestamp.min.tz_localize("UTC")
    if m1_ts_max < target_m1_last:
        blocker_reasons.append("M1_DATA_COVERAGE_BEFORE_TARGET_END")
        candidate_blocker_reasons.append("M1_DATA_COVERAGE_BEFORE_TARGET_END")
    if m5_ts_max < target_m5_last:
        blocker_reasons.append("M5_DATA_COVERAGE_BEFORE_TARGET_END")
        candidate_blocker_reasons.append("M5_DATA_COVERAGE_BEFORE_TARGET_END")
    if prebuilt_model_ts_max < target_m5_last:
        blocker_reasons.append("PREBUILT_MODEL_BAR_COVERAGE_BEFORE_TARGET_END")
    if not candidate_prebuilt.get("exists"):
        candidate_blocker_reasons.append("CANDIDATE_PREBUILT_MISSING")
    elif candidate_model_ts_max < target_m5_last:
        candidate_blocker_reasons.append("CANDIDATE_PREBUILT_MODEL_BAR_COVERAGE_BEFORE_TARGET_END")

    if blocker_reasons and not candidate_blocker_reasons:
        status = "CALENDAR_REORG_READY_DATA_READY_CANDIDATE_PREBUILT_READY_CURRENT_MANIFEST_NOT_PROMOTED"
    elif blocker_reasons:
        status = "CALENDAR_REORG_READY_DATA_READY_PREBUILT_BLOCKED"
    else:
        status = "READY_FOR_MONDAY_WEEK_REPLAY"

    payload = {
        "artifact_name": output_name,
        "created_at_utc": _now(),
        "status": status,
        "contract": {
            "calendar_week": "MONDAY_00:00_UTC_TO_NEXT_MONDAY_00:00_UTC_EXCLUSIVE",
            "trading_week": "MONDAY_START_TO_FRIDAY_FLAT_CUTOFF",
            "friday_flat_cutoff_utc": "20:55",
            "weekend_entries": "FORBIDDEN",
            "weekend_management": "NO_ACTION_AFTER_FRIDAY_FLAT_EXPECTED",
            "replay_eof_policy": "REPLAY_EOF_IS_BOUNDARY_ARTIFACT_AND_NOT_EXIT_RL_TRUTH",
            "target_replay_eof_count": 0,
        },
        "active_v2_root": str(active_v2_root),
        "old_wednesday_chunk_summary": _completed_eof_summary(active_v2_root),
        "data_coverage": {"m1": m1, "m5": m5, "target_end_exclusive": data_end_exclusive},
        "prebuilt_coverage": prebuilt,
        "candidate_prebuilt_coverage": candidate_prebuilt,
        "blocker_reasons": blocker_reasons,
        "candidate_blocker_reasons": candidate_blocker_reasons,
        "quarantine_policy": {
            "old_pathological_window": "E2E_SANITY_ORDERFIX_20251203_20251210",
            "new_monday_weeks_quarantined": [
                asdict(w) for w in full_reorg_weeks if w.quarantine_status == "QUARANTINED"
            ],
            "quarantine_verdict": "KEEP_QUARANTINED_DO_NOT_FORCE_REPLAY",
        },
        "oos_extension_weeks": [asdict(w) for w in oos_weeks],
        "full_monday_week_count": int(len(full_reorg_weeks)),
        "full_monday_weeks": [asdict(w) for w in full_reorg_weeks],
        "next_required_steps": [
            "AUDIT_CANDIDATE_PREBUILT_TO_2026_04_19",
            "CREATE_OOS_TRUTH_CONFIG_OR_UPDATE_MANIFEST_ONLY_AFTER_PREBUILT_AUDIT",
            "RUN_MONDAY_WEEK_REPLAY_FOR_2026_04_06_AND_2026_04_13",
            "HARD_FAIL_IF_REPLAY_EOF_GT_0_EXCEPT_EXPLICIT_PARTIAL_WINDOW",
        ],
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    json_path = reports_root / f"{output_name}.json"
    md_path = reports_root / f"{output_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        f"# {output_name}",
        "",
        f"- status: `{payload['status']}`",
        f"- active_v2_root: `{active_v2_root}`",
        f"- M1 max: `{m1.get('ts_max')}`",
        f"- M5 max: `{m5.get('ts_max')}`",
        f"- prebuilt model-bar max: `{prebuilt.get('model_bar_ts_max')}`",
        f"- candidate prebuilt model-bar max: `{candidate_prebuilt.get('coverage', {}).get('model_bar_ts_max')}`",
        f"- blocker_reasons: `{', '.join(blocker_reasons) if blocker_reasons else 'NONE'}`",
        f"- candidate_blocker_reasons: `{', '.join(candidate_blocker_reasons) if candidate_blocker_reasons else 'NONE'}`",
        f"- old replay EOF trades: `{payload['old_wednesday_chunk_summary'].get('replay_eof_trades')}`",
        "",
        "## Contract",
        "",
        "- Calendar window is Monday 00:00 UTC to next Monday 00:00 UTC exclusive.",
        "- Trading/action window is Monday through Friday flat cutoff 20:55 UTC.",
        "- Weekend entries are forbidden; weekend should be no-action after Friday flat.",
        "- REPLAY_EOF is a boundary artifact, not clean exit/RL truth.",
        "",
        "## OOS Weeks",
        "",
    ]
    for w in oos_weeks:
        md_lines.append(
            f"- `{w.run_id}`: `{w.calendar_start_utc}` -> `{w.calendar_end_exclusive_utc}`, "
            f"flat `{w.friday_flat_cutoff_utc}`, status `{w.quarantine_status}`"
        )
    md_lines.extend(
        [
            "",
            "## Quarantine",
            "",
            "- `TRUTH_MONFRI_WEEK_20251201_20251208` quarantined from old pathological December window.",
            "- `TRUTH_MONFRI_WEEK_20251208_20251215` quarantined from old pathological December window.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "payload": payload}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity"),
    )
    parser.add_argument(
        "--active-v2-root",
        type=Path,
        default=Path(
            "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/PATH_DYNAMICS_LOGGING_V2_REPLAY_20260422_1227"
        ),
    )
    parser.add_argument("--output-name", default="TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1")
    parser.add_argument("--data-end-exclusive", default="2026-04-20T00:00:00Z")
    args = parser.parse_args()
    result = materialize(
        reports_root=args.reports_root,
        active_v2_root=args.active_v2_root,
        output_name=args.output_name,
        data_end_exclusive=args.data_end_exclusive,
    )
    print(json.dumps({"json": result["json"], "markdown": result["markdown"], "status": result["payload"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
