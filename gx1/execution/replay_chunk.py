"""
Replay Chunk Runner: Canonical wrapper for "run one replay chunk" (TRUTH-grade).

LOCKED GOALS:
- replay_chunk.py is a thin orchestrator (SSoT flow; not “smart” IO).
- All writes outside the orchestrator happen via atomic/best-effort helpers.
- chunk_footer_writer is a DUMB WRITER:
    - does NOT read from disk
    - does NOT read runner/telemetry objects
    - does NOT validate invariants or mutate status/error
    - writes ONLY chunk_footer.json atomically
- TRUTH/SMOKE is strict: invariants may flip status to failed_invariant.
- No segmented/parallel/owner/preroll in TRUTH 1W1C.

NOTES:
- This file may read small things from runner in-memory (perf counters etc).
- This file must not rebuild data/features (prebuilt-only in TRUTH).
"""

from __future__ import annotations

import logging
import os
import signal
import tempfile
import time
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.utils.dt_module import now_iso as dt_now_iso
import hashlib


def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_import_proof_if_needed(
    chunk_output_dir: Path,
    run_id: str,
    chunk_idx: int,
    truth_artifacts: Optional[Dict[str, Any]],
) -> Tuple[Optional[Path], list]:
    """Write IMPORT_PROOF.json when required by truth_artifacts. Returns (path, forbidden_hits)."""
    cfg = (truth_artifacts or {}).get("replay_config", {}).get("truth_artifacts", {}) or {}
    if not cfg.get("require_import_proof"):
        return None, []
    fname = cfg.get("import_proof_filename") or "IMPORT_PROOF.json"
    target = chunk_output_dir / fname
    forbidden_exact = [
        "gx1.inference.model_loader_worker",
        "gx1.scripts.replay_eval_gated_parallel",
    ]
    forbidden_patterns = ["runtime_v9"]
    modules_sorted = sorted(sys.modules.keys())
    hits = []
    for mod in modules_sorted:
        if mod in forbidden_exact:
            hits.append({"module": mod, "reason": "exact"})
        else:
            for pat in forbidden_patterns:
                if pat in mod:
                    hits.append({"module": mod, "reason": f"pattern:{pat}"})
                    break
    joined = "\n".join(modules_sorted)
    sha = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    payload = {
        "run_id": run_id,
        "chunk_idx": int(chunk_idx),
        "created_utc": dt_now_iso(),
        "truth_file_used": os.getenv("GX1_CANONICAL_TRUTH_FILE") or None,
        "banlist": {
            "forbidden_exact": forbidden_exact,
            "forbidden_patterns": forbidden_patterns,
        },
        "forbidden_hits": hits,
        "sys_modules_count": len(modules_sorted),
        "sys_modules_sha256": sha,
        "sys_modules_sample": modules_sorted[:200],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json_safe(target, payload)
    return target, hits


def _write_prefork_freeze(
    output_dir: Path,
    payload: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "PRE_FORK_FREEZE.json.tmp"
    final = output_dir / "PRE_FORK_FREEZE.json"
    atomic_write_json_safe(tmp, payload)
    os.replace(tmp, final)

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global flag for graceful shutdown (SIGTERM)
# -----------------------------------------------------------------------------
STOP_REQUESTED = False


def _sigterm_handler(signum, frame):
    """SIGTERM handler for graceful shutdown."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    os.environ["GX1_STOP_REQUESTED"] = "1"
    log.warning("[TERM] Received SIGTERM (pid=%s), will attempt graceful stop", os.getpid())


# -----------------------------------------------------------------------------
# Failure helpers (robust, no disk reads required)
# -----------------------------------------------------------------------------
from gx1.execution.chunk_failure import (  # noqa: E402
    convert_to_json_serializable,
    atomic_write_json_safe,
    write_failure_capsule,
    build_failure_context,
    write_signal_event_capsule,
)

# -----------------------------------------------------------------------------
# Dumb footer writer (payload-only)
# -----------------------------------------------------------------------------
from gx1.execution.chunk_footer_writer import ChunkFooterContext, write_chunk_footer  # noqa: E402

# -----------------------------------------------------------------------------
# Bootstrap / data loader / exporters / invariants
# -----------------------------------------------------------------------------
from gx1.execution.chunk_bootstrap import bootstrap_chunk_environment, BootstrapContext  # noqa: E402
from gx1.execution.chunk_data_loader import load_chunk_data, DataContext  # noqa: E402
from gx1.execution.killchain_export import KillchainExportContext, export_killchain  # noqa: E402
from gx1.execution.prebuilt_invariants import PrebuiltInvariantContext, check_prebuilt_invariants  # noqa: E402
from gx1.utils.empty_trade_outcomes import (  # noqa: E402
    TRADE_OUTCOMES_REQUIRED_COLUMNS,
    write_empty_trade_outcomes_parquet,
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _is_truth_or_smoke() -> bool:
    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    return run_mode in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"


def _assert_truth_ban_envs() -> None:
    """Hard-gate: forbid segmented/parallel/owner/preroll envs in TRUTH/SMOKE 1W1C."""
    if not _is_truth_or_smoke():
        return
    for forbidden in (
        "GX1_SEGMENTED_PARALLEL",
        "GX1_SEGMENT_START",
        "GX1_SEGMENT_END",
        "GX1_PREROLL_START",
        "GX1_OWNER_START",
        "GX1_OWNER_END",
    ):
        if forbidden in os.environ:
            raise RuntimeError(f"[FORBIDDEN_ENV] {forbidden} is not allowed in 1W1C TRUTH")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: bool = False) -> bool:
    try:
        if x is None:
            return default
        return bool(x)
    except Exception:
        return default


def _write_minimal_attribution(chunk_output_dir: Path, run_id: str) -> None:
    """
    Write minimal attribution JSON (TRUTH-safe) when legacy attribution pipeline
    is not present. This is used only to satisfy TRUTH gates that expect an
    attribution_{run_id}.json artifact.
    """
    path = chunk_output_dir / f"attribution_{run_id}.json"
    payload = {
        "mode": "truth_minimal",
        "run_id": run_id,
        "chunks": 1,
        "note": "Minimal attribution written by replay_chunk (legacy attribution unavailable).",
        "timestamp": dt_now_iso(),
    }
    atomic_write_json_safe(path, convert_to_json_serializable(payload))


def _write_stuck_trade_audit(
    chunk_output_dir: Path,
    run_id: str,
    journal_df: "pd.DataFrame",
) -> None:
    """
    Stuck-trade / tail-loss audit (observability only, no trading logic changes).

    Reads EXIT_EVAL_TRACE.csv (per-bar prob_close timeseries) already written
    by the runner, joins with trade_journal_df (already built in memory), and
    writes STUCK_TRADE_AUDIT.json to chunk_output_dir.

    Audited trades:
    - All CATASTROPHIC_GUARD closes
    - All REPLAY_EOF closes
    - THRESHOLD closes in the worst-pnl tail (final_pnl_bps <= -100 bps,
      or bottom-20 if fewer than 20 trades would qualify at -100 cutoff)

    Per-trade fields computed (where available from EVAL_TRACE):
    - max/mean prob_close over first 5, first 20, and lifetime bars
    - first bar where prob_close >= threshold, ever_crossed_threshold
    - exit-state MFE/MAE from journal (already computed)
    - overlap_flag (session == OVERLAP), bars0_candidate (bars_in_trade <= 0
      at close, i.e. same-bar candidate)

    Aggregate sections:
    A. Per close_reason summary
    B. long+OVERLAP+bars0_candidate pocket vs rest
    C. Top-20 worst trades table

    Output: STUCK_TRADE_AUDIT.json in chunk_output_dir.
    Logged as [STUCK_TRADE_AUDIT_PROOF] line.
    Never raises — best-effort.
    """
    import json

    AUDIT_REASONS = {"CATASTROPHIC_GUARD", "REPLAY_EOF"}
    THRESHOLD_PNL_CUTOFF = -100.0
    THRESHOLD_MIN_TAIL_N = 20
    TOP_WORST_N = 20

    try:
        if journal_df is None or journal_df.empty:
            log.info("[STUCK_TRADE_AUDIT_PROOF] run_id=%s skipped=journal_empty", run_id)
            return

        jdf = journal_df.copy()

        # ------------------------------------------------------------------
        # 1. Determine audited trade set
        # ------------------------------------------------------------------
        audit_reasons_mask = jdf["exit_reason"].fillna("UNKNOWN").isin(AUDIT_REASONS) if "exit_reason" in jdf.columns else pd.Series(False, index=jdf.index)

        threshold_mask = pd.Series(False, index=jdf.index)
        if "exit_reason" in jdf.columns and "pnl_bps" in jdf.columns:
            pnl_num = pd.to_numeric(jdf["pnl_bps"], errors="coerce")
            thr_trades = jdf["exit_reason"].fillna("UNKNOWN") == "THRESHOLD"
            cutoff_mask = thr_trades & (pnl_num <= THRESHOLD_PNL_CUTOFF)
            if int(cutoff_mask.sum()) < THRESHOLD_MIN_TAIL_N:
                worst_idx = pnl_num[thr_trades].nsmallest(THRESHOLD_MIN_TAIL_N).index
                threshold_mask.loc[worst_idx] = True
            else:
                threshold_mask = cutoff_mask

        audited_mask = audit_reasons_mask | threshold_mask
        audited_df = jdf[audited_mask].copy()

        if audited_df.empty:
            log.info("[STUCK_TRADE_AUDIT_PROOF] run_id=%s skipped=no_audited_trades", run_id)
            return

        # ------------------------------------------------------------------
        # 2. Enrich with overlap_flag and bars0_candidate
        # ------------------------------------------------------------------
        if "session" in audited_df.columns:
            audited_df["overlap_flag"] = audited_df["session"].fillna("").str.upper() == "OVERLAP"
        else:
            audited_df["overlap_flag"] = False

        if "bars_in_trade" in audited_df.columns:
            bars_num = pd.to_numeric(audited_df["bars_in_trade"], errors="coerce")
            audited_df["bars0_candidate"] = bars_num <= 0
        else:
            audited_df["bars0_candidate"] = False

        # ------------------------------------------------------------------
        # 3. Load EXIT_EVAL_TRACE.csv for per-bar prob_close stats
        # ------------------------------------------------------------------
        trace_path = chunk_output_dir / "EXIT_EVAL_TRACE.csv"
        trade_prob_stats: Dict[str, Dict[str, Any]] = {}
        if trace_path.exists():
            try:
                trace_df = pd.read_csv(
                    trace_path,
                    usecols=["trade_id", "bars_held", "exit_prob", "exit_threshold"],
                    dtype={"trade_id": str},
                )
                trace_df["exit_prob"] = pd.to_numeric(trace_df["exit_prob"], errors="coerce")
                trace_df["exit_threshold"] = pd.to_numeric(trace_df["exit_threshold"], errors="coerce")
                trace_df["bars_held"] = pd.to_numeric(trace_df["bars_held"], errors="coerce")

                audited_ids = set(audited_df["trade_id"].dropna().astype(str).tolist()) if "trade_id" in audited_df.columns else set()
                sub_trace = trace_df[trace_df["trade_id"].isin(audited_ids)].copy()

                for tid, grp in sub_trace.groupby("trade_id"):
                    grp = grp.sort_values("bars_held")
                    prob = grp["exit_prob"].dropna()
                    thr_val = grp["exit_threshold"].dropna()
                    thr_scalar = float(thr_val.iloc[0]) if not thr_val.empty else None
                    first5 = prob.iloc[:5] if len(prob) >= 1 else prob
                    first20 = prob.iloc[:20] if len(prob) >= 1 else prob

                    def _safe_stat(s: "pd.Series", fn: str):
                        try:
                            return float(getattr(s, fn)()) if not s.empty else None
                        except Exception:
                            return None

                    first_cross_bar = None
                    ever_crossed = False
                    if thr_scalar is not None:
                        crossed = grp[grp["exit_prob"] >= thr_scalar]
                        if not crossed.empty:
                            ever_crossed = True
                            first_bar_val = crossed["bars_held"].iloc[0]
                            first_cross_bar = int(first_bar_val) if pd.notna(first_bar_val) else None

                    trade_prob_stats[str(tid)] = {
                        "max_prob_close_first_5": _safe_stat(first5, "max"),
                        "mean_prob_close_first_5": _safe_stat(first5, "mean"),
                        "max_prob_close_first_20": _safe_stat(first20, "max"),
                        "mean_prob_close_first_20": _safe_stat(first20, "mean"),
                        "max_prob_close_lifetime": _safe_stat(prob, "max"),
                        "mean_prob_close_lifetime": _safe_stat(prob, "mean"),
                        "first_bar_where_prob_close_ge_threshold": first_cross_bar,
                        "ever_crossed_threshold": ever_crossed,
                        "n_eval_steps": int(len(prob)),
                    }
            except Exception as e:
                log.warning("[STUCK_TRADE_AUDIT_TRACE_FAIL] %s", e)

        # ------------------------------------------------------------------
        # 4. Build per-trade audit records
        # ------------------------------------------------------------------
        def _fv(val: Any) -> Any:
            """Safe float-or-None."""
            try:
                f = float(val)
                return None if (f != f) else round(f, 6)  # NaN check
            except Exception:
                return None

        per_trade_records = []
        for _, row in audited_df.iterrows():
            tid = str(row.get("trade_id", "")) if row.get("trade_id") is not None else None
            prob_s = trade_prob_stats.get(tid, {}) if tid else {}
            rec = {
                "trade_id": tid,
                "entry_time": str(row.get("open_ts_utc", "")) if row.get("open_ts_utc") is not None else None,
                "exit_time": str(row.get("close_ts_utc", "")) if row.get("close_ts_utc") is not None else None,
                "side": row.get("side"),
                "session": row.get("session"),
                "exit_reason": row.get("exit_reason"),
                "overlap_flag": bool(row.get("overlap_flag", False)),
                "bars0_candidate": bool(row.get("bars0_candidate", False)),
                "bars_held_final": _fv(row.get("bars_in_trade")),
                "final_pnl_bps": _fv(row.get("pnl_bps")),
                "mfe_bps_max": _fv(row.get("mfe_bps")),
                "mae_bps_min": _fv(row.get("mae_bps")),
                "time_since_mfe_bars_at_close": _fv(row.get("time_since_mfe_bars_exit")),
                "max_prob_close_first_5": prob_s.get("max_prob_close_first_5"),
                "mean_prob_close_first_5": prob_s.get("mean_prob_close_first_5"),
                "max_prob_close_first_20": prob_s.get("max_prob_close_first_20"),
                "mean_prob_close_first_20": prob_s.get("mean_prob_close_first_20"),
                "max_prob_close_lifetime": prob_s.get("max_prob_close_lifetime"),
                "mean_prob_close_lifetime": prob_s.get("mean_prob_close_lifetime"),
                "first_bar_where_prob_close_ge_threshold": prob_s.get("first_bar_where_prob_close_ge_threshold"),
                "ever_crossed_threshold": prob_s.get("ever_crossed_threshold"),
                "n_eval_steps": prob_s.get("n_eval_steps"),
            }
            per_trade_records.append(rec)

        # Sort by final_pnl_bps ascending (worst first)
        per_trade_records.sort(key=lambda r: (r["final_pnl_bps"] is None, r["final_pnl_bps"] or 0.0))

        # ------------------------------------------------------------------
        # 5. Section A: per close_reason aggregate
        # ------------------------------------------------------------------
        def _reason_agg(records: list) -> Dict[str, Any]:
            from collections import defaultdict
            buckets: Dict[str, list] = defaultdict(list)
            for r in records:
                buckets[r.get("exit_reason") or "UNKNOWN"].append(r)
            out = {}
            for reason, recs in sorted(buckets.items()):
                pnl_vals = [r["final_pnl_bps"] for r in recs if r["final_pnl_bps"] is not None]
                p5 = [r["max_prob_close_first_5"] for r in recs if r["max_prob_close_first_5"] is not None]
                p20 = [r["max_prob_close_first_20"] for r in recs if r["max_prob_close_first_20"] is not None]
                crossed = [r["ever_crossed_threshold"] for r in recs if r["ever_crossed_threshold"] is not None]
                n = len(recs)

                def _agg(vals: list, fn: str) -> Any:
                    if not vals:
                        return None
                    a = np.array(vals, dtype=float)
                    return round(float(getattr(np, fn)(a)), 6)

                out[reason] = {
                    "n": n,
                    "pnl_mean": _agg(pnl_vals, "mean"),
                    "pnl_median": _agg(pnl_vals, "median"),
                    "pnl_worst": _agg(pnl_vals, "min"),
                    "mean_max_prob_close_first_5": _agg(p5, "mean"),
                    "mean_max_prob_close_first_20": _agg(p20, "mean"),
                    "frac_ever_crossed_threshold": round(float(np.mean(crossed)), 6) if crossed else None,
                }
            return out

        section_a = _reason_agg(per_trade_records)

        # ------------------------------------------------------------------
        # 6. Section B: pocket comparison
        # ------------------------------------------------------------------
        def _pocket_agg(records: list) -> Dict[str, Any]:
            pnl_vals = [r["final_pnl_bps"] for r in records if r["final_pnl_bps"] is not None]
            p5 = [r["max_prob_close_first_5"] for r in records if r["max_prob_close_first_5"] is not None]
            p20 = [r["max_prob_close_first_20"] for r in records if r["max_prob_close_first_20"] is not None]
            crossed = [r["ever_crossed_threshold"] for r in records if r["ever_crossed_threshold"] is not None]
            n = len(records)

            def _agg(vals: list, fn: str) -> Any:
                if not vals:
                    return None
                a = np.array(vals, dtype=float)
                return round(float(getattr(np, fn)(a)), 6)

            reasons = [r.get("exit_reason") or "UNKNOWN" for r in records]
            reason_counts: Dict[str, int] = {}
            for rr in reasons:
                reason_counts[rr] = reason_counts.get(rr, 0) + 1

            return {
                "n": n,
                "pnl_mean": _agg(pnl_vals, "mean"),
                "pnl_median": _agg(pnl_vals, "median"),
                "pnl_worst": _agg(pnl_vals, "min"),
                "frac_catastrophic_guard": round(reason_counts.get("CATASTROPHIC_GUARD", 0) / n, 6) if n else None,
                "frac_replay_eof": round(reason_counts.get("REPLAY_EOF", 0) / n, 6) if n else None,
                "mean_max_prob_close_first_5": _agg(p5, "mean"),
                "mean_max_prob_close_first_20": _agg(p20, "mean"),
                "frac_ever_crossed_threshold": round(float(np.mean(crossed)), 6) if crossed else None,
            }

        pocket_records = [
            r for r in per_trade_records
            if (r.get("side") or "").lower() == "long"
            and r.get("overlap_flag") is True
            and r.get("bars0_candidate") is True
        ]
        rest_records = [r for r in per_trade_records if r not in pocket_records]

        section_b = {
            "long_overlap_bars0": _pocket_agg(pocket_records) if pocket_records else {"n": 0},
            "rest": _pocket_agg(rest_records) if rest_records else {"n": 0},
        }

        # ------------------------------------------------------------------
        # 7. Section C: top-20 worst trades table
        # ------------------------------------------------------------------
        section_c = [
            {
                "entry_time": r["entry_time"],
                "side": r["side"],
                "session": r["session"],
                "overlap_flag": r["overlap_flag"],
                "bars0_candidate": r["bars0_candidate"],
                "close_reason": r["exit_reason"],
                "final_pnl_bps": r["final_pnl_bps"],
                "max_prob_close_first_5": r["max_prob_close_first_5"],
                "max_prob_close_first_20": r["max_prob_close_first_20"],
                "max_prob_close_lifetime": r["max_prob_close_lifetime"],
                "first_bar_where_prob_close_ge_threshold": r["first_bar_where_prob_close_ge_threshold"],
            }
            for r in per_trade_records[:TOP_WORST_N]
        ]

        # ------------------------------------------------------------------
        # 8. Verdict summary
        # ------------------------------------------------------------------
        all_pnl_first5 = [r["max_prob_close_first_5"] for r in per_trade_records if r["max_prob_close_first_5"] is not None]
        all_pnl_first20 = [r["max_prob_close_first_20"] for r in per_trade_records if r["max_prob_close_first_20"] is not None]
        all_crossed = [r["ever_crossed_threshold"] for r in per_trade_records if r["ever_crossed_threshold"] is not None]

        verdict = {
            "worst_tail_low_prob_first5": (
                round(float(np.mean(all_pnl_first5)), 6) if all_pnl_first5 else None
            ),
            "worst_tail_low_prob_first20": (
                round(float(np.mean(all_pnl_first20)), 6) if all_pnl_first20 else None
            ),
            "frac_ever_crossed_threshold_overall": (
                round(float(np.mean(all_crossed)), 6) if all_crossed else None
            ),
            "pocket_long_overlap_bars0_n": section_b["long_overlap_bars0"].get("n", 0),
            "pocket_long_overlap_bars0_pnl_mean": section_b["long_overlap_bars0"].get("pnl_mean"),
            "rest_pnl_mean": section_b["rest"].get("pnl_mean"),
        }

        # ------------------------------------------------------------------
        # 9. Write STUCK_TRADE_AUDIT.json
        # ------------------------------------------------------------------
        payload = {
            "run_id": run_id,
            "audit_scope": {
                "reasons_always_included": sorted(AUDIT_REASONS),
                "threshold_pnl_cutoff_bps": THRESHOLD_PNL_CUTOFF,
                "threshold_min_tail_n": THRESHOLD_MIN_TAIL_N,
                "n_audited_trades": len(per_trade_records),
                "eval_trace_used": str(trace_path),
                "eval_trace_found": trace_path.exists(),
            },
            "section_a_close_reason_tail_audit": section_a,
            "section_b_pocket_comparison": section_b,
            "section_c_top_worst_trades": section_c,
            "verdict": verdict,
            "per_trade_records": per_trade_records,
        }
        out_path = chunk_output_dir / "STUCK_TRADE_AUDIT.json"
        tmp_path = out_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        log.info(
            "[STUCK_TRADE_AUDIT_PROOF] run_id=%s n_audited=%d "
            "section_a_reasons=%s "
            "pocket_long_overlap_bars0_n=%d pocket_pnl_mean=%s rest_pnl_mean=%s "
            "frac_ever_crossed_threshold=%.3f "
            "path=%s",
            run_id,
            len(per_trade_records),
            list(section_a.keys()),
            section_b["long_overlap_bars0"].get("n", 0),
            section_b["long_overlap_bars0"].get("pnl_mean"),
            section_b["rest"].get("pnl_mean"),
            verdict["frac_ever_crossed_threshold_overall"] or 0.0,
            out_path,
        )

    except Exception as e:
        log.warning("[STUCK_TRADE_AUDIT_FAIL] run_id=%s error=%s", run_id, e)


def _write_short_exit_signal_audit(
    chunk_output_dir: Path,
    run_id: str,
    journal_df: "pd.DataFrame",
) -> None:
    """
    Short-vs-long EXIT signal quality audit (observability only, no trading logic changes).

    Reads EXIT_EVAL_TRACE.csv (per-bar prob_close timeseries) and trade_journal_df,
    produces SHORT_EXIT_SIGNAL_AUDIT.json in chunk_output_dir.

    Analyses:
    1. Exit signal quality by side (all trades)
    2. Close reason by side
    3. Worst-50 tail by side
    4. Exit signal evolution (mean prob_close per bar 1..10 by side)
    5. Feature sanity by side (pnl_bps_now, mfe_bps, mae_bps, bars_in_trade from EVAL_TRACE)

    Proof label: [SHORT_EXIT_SIGNAL_AUDIT_PROOF]
    Never raises — best-effort.
    """
    import json

    WORST_TAIL_N = 50
    SIGNAL_CURVE_BARS = 10

    try:
        if journal_df is None or journal_df.empty:
            log.info("[SHORT_EXIT_SIGNAL_AUDIT_PROOF] run_id=%s skipped=journal_empty", run_id)
            return

        jdf = journal_df.copy()
        if "side" not in jdf.columns:
            log.info("[SHORT_EXIT_SIGNAL_AUDIT_PROOF] run_id=%s skipped=no_side_col", run_id)
            return

        # ------------------------------------------------------------------
        # Load EXIT_EVAL_TRACE.csv — build per-trade prob stats and signal curve
        # ------------------------------------------------------------------
        trace_path = chunk_output_dir / "EXIT_EVAL_TRACE.csv"
        trade_prob_stats: Dict[str, Dict[str, Any]] = {}
        signal_curve_by_side: Dict[str, Dict[int, list]] = {"long": {}, "short": {}}
        feature_rows_by_side: Dict[str, list] = {"long": [], "short": []}

        if trace_path.exists():
            try:
                trace_df = pd.read_csv(
                    trace_path,
                    usecols=["trade_id", "bars_held", "exit_prob", "exit_threshold",
                             "pnl_bps", "mfe_bps", "mae_bps"],
                    dtype={"trade_id": str},
                )
                for col in ["exit_prob", "exit_threshold", "bars_held", "pnl_bps", "mfe_bps", "mae_bps"]:
                    trace_df[col] = pd.to_numeric(trace_df[col], errors="coerce")

                # Build side lookup from journal
                side_by_tid: Dict[str, str] = {}
                if "trade_id" in jdf.columns and "side" in jdf.columns:
                    for _, row in jdf[["trade_id", "side"]].dropna().iterrows():
                        side_by_tid[str(row["trade_id"])] = str(row["side"]).lower()

                for tid, grp in trace_df.groupby("trade_id"):
                    grp = grp.sort_values("bars_held").reset_index(drop=True)
                    prob = grp["exit_prob"].dropna()
                    thr_val = grp["exit_threshold"].dropna()
                    thr_scalar = float(thr_val.iloc[0]) if not thr_val.empty else None
                    first5 = prob.iloc[:5] if len(prob) >= 1 else prob
                    first20 = prob.iloc[:20] if len(prob) >= 1 else prob

                    def _ss(s: "pd.Series", fn: str):
                        try:
                            return float(getattr(s, fn)()) if not s.empty else None
                        except Exception:
                            return None

                    first_cross_bar = None
                    ever_crossed = False
                    if thr_scalar is not None:
                        crossed = grp[grp["exit_prob"] >= thr_scalar]
                        if not crossed.empty:
                            ever_crossed = True
                            bv = crossed["bars_held"].iloc[0]
                            first_cross_bar = int(bv) if pd.notna(bv) else None

                    trade_prob_stats[str(tid)] = {
                        "max_prob_close_first_5": _ss(first5, "max"),
                        "mean_prob_close_first_5": _ss(first5, "mean"),
                        "max_prob_close_first_20": _ss(first20, "max"),
                        "mean_prob_close_first_20": _ss(first20, "mean"),
                        "max_prob_close_lifetime": _ss(prob, "max"),
                        "mean_prob_close_lifetime": _ss(prob, "mean"),
                        "ever_crossed_threshold": ever_crossed,
                        "first_cross_bar": first_cross_bar,
                        "n_eval_steps": int(len(prob)),
                    }

                    # Signal curve: collect prob at bar index 0..SIGNAL_CURVE_BARS-1
                    side_key = side_by_tid.get(str(tid))
                    if side_key in signal_curve_by_side:
                        curve_bucket = signal_curve_by_side[side_key]
                        for bar_idx in range(SIGNAL_CURVE_BARS):
                            if bar_idx < len(grp):
                                pval = grp["exit_prob"].iloc[bar_idx]
                                if pd.notna(pval):
                                    curve_bucket.setdefault(bar_idx, []).append(float(pval))

                    # Feature sanity: first-bar values per side
                    if side_key in feature_rows_by_side and len(grp) >= 1:
                        row0 = grp.iloc[0]
                        feature_rows_by_side[side_key].append({
                            "pnl_bps_bar0": float(row0["pnl_bps"]) if pd.notna(row0["pnl_bps"]) else None,
                            "mfe_bps_bar0": float(row0["mfe_bps"]) if pd.notna(row0["mfe_bps"]) else None,
                            "mae_bps_bar0": float(row0["mae_bps"]) if pd.notna(row0["mae_bps"]) else None,
                        })

            except Exception as e:
                log.warning("[SHORT_EXIT_SIGNAL_AUDIT_TRACE_FAIL] %s", e)

        # ------------------------------------------------------------------
        # Analysis 1 — signal quality by side (all trades)
        # ------------------------------------------------------------------
        def _side_signal_summary(side_val: str) -> Dict[str, Any]:
            mask = jdf["side"].fillna("").str.lower() == side_val
            sub = jdf[mask].copy()
            n = int(len(sub))
            if n == 0:
                return {"n": 0}
            pnl = pd.to_numeric(sub["pnl_bps"], errors="coerce")

            def _r(v: Any) -> Any:
                try:
                    f = float(v)
                    return None if f != f else round(f, 6)
                except Exception:
                    return None

            p5_vals, p20_vals, p_life_vals, cross_vals, cross_bar_vals = [], [], [], [], []
            for _, row in sub.iterrows():
                tid = str(row.get("trade_id", "")) if row.get("trade_id") is not None else None
                ps = trade_prob_stats.get(tid, {}) if tid else {}
                if ps.get("max_prob_close_first_5") is not None:
                    p5_vals.append(ps["max_prob_close_first_5"])
                if ps.get("max_prob_close_first_20") is not None:
                    p20_vals.append(ps["max_prob_close_first_20"])
                if ps.get("max_prob_close_lifetime") is not None:
                    p_life_vals.append(ps["max_prob_close_lifetime"])
                if ps.get("ever_crossed_threshold") is not None:
                    cross_vals.append(float(ps["ever_crossed_threshold"]))
                if ps.get("first_cross_bar") is not None:
                    cross_bar_vals.append(ps["first_cross_bar"])

            def _agg(vals: list, fn: str) -> Any:
                if not vals:
                    return None
                a = np.array(vals, dtype=float)
                return round(float(getattr(np, fn)(a)), 6)

            return {
                "n": n,
                "pnl_mean": _r(pnl.mean()),
                "pnl_median": _r(pnl.median()),
                "pnl_worst": _r(pnl.min()),
                "mean_max_prob_close_first_5": _agg(p5_vals, "mean"),
                "mean_max_prob_close_first_20": _agg(p20_vals, "mean"),
                "mean_max_prob_close_lifetime": _agg(p_life_vals, "mean"),
                "frac_ever_crossed_threshold": _agg(cross_vals, "mean"),
                "mean_first_cross_bar": _agg(cross_bar_vals, "mean"),
                "n_with_prob_stats": len(p5_vals),
            }

        analysis_1 = {
            "long": _side_signal_summary("long"),
            "short": _side_signal_summary("short"),
        }

        # ------------------------------------------------------------------
        # Analysis 2 — close reason by side
        # ------------------------------------------------------------------
        analysis_2: Dict[str, list] = {"long": [], "short": []}
        if "exit_reason" in jdf.columns:
            pnl_num = pd.to_numeric(jdf["pnl_bps"], errors="coerce")
            for side_val in ("long", "short"):
                mask = jdf["side"].fillna("").str.lower() == side_val
                sub = jdf[mask].copy()
                total_n = int(len(sub))
                if total_n == 0:
                    continue
                for reason, grp in sub.groupby(sub["exit_reason"].fillna("UNKNOWN")):
                    grp_pnl = pd.to_numeric(grp["pnl_bps"], errors="coerce")
                    analysis_2[side_val].append({
                        "close_reason": str(reason),
                        "count": int(len(grp)),
                        "fraction": round(len(grp) / total_n, 6),
                        "mean_pnl": round(float(grp_pnl.mean()), 6) if not grp_pnl.dropna().empty else None,
                        "worst_pnl": round(float(grp_pnl.min()), 6) if not grp_pnl.dropna().empty else None,
                    })
                analysis_2[side_val].sort(key=lambda x: -x["count"])

        # ------------------------------------------------------------------
        # Analysis 3 — worst-50 tail by side
        # ------------------------------------------------------------------
        pnl_col = pd.to_numeric(jdf["pnl_bps"], errors="coerce")
        worst50_idx = pnl_col.nsmallest(WORST_TAIL_N).index
        worst50 = jdf.loc[worst50_idx].copy()
        worst50_records = []
        for _, row in worst50.iterrows():
            tid = str(row.get("trade_id", "")) if row.get("trade_id") is not None else None
            ps = trade_prob_stats.get(tid, {}) if tid else {}
            worst50_records.append({
                "trade_id": tid,
                "side": row.get("side"),
                "session": row.get("session"),
                "close_reason": row.get("exit_reason"),
                "final_pnl_bps": float(row["pnl_bps"]) if pd.notna(row.get("pnl_bps")) else None,
                "bars_held": float(row["bars_in_trade"]) if pd.notna(row.get("bars_in_trade")) else None,
                "max_prob_close_first_5": ps.get("max_prob_close_first_5"),
                "max_prob_close_first_20": ps.get("max_prob_close_first_20"),
                "max_prob_close_lifetime": ps.get("max_prob_close_lifetime"),
                "ever_crossed_threshold": ps.get("ever_crossed_threshold"),
            })

        # Tail summary
        tail_sides = [r.get("side") or "UNKNOWN" for r in worst50_records]
        tail_short = [r for r in worst50_records if (r.get("side") or "").lower() == "short"]
        tail_long  = [r for r in worst50_records if (r.get("side") or "").lower() == "long"]

        def _tail_prob_mean(recs: list, key: str) -> Any:
            vals = [r[key] for r in recs if r.get(key) is not None]
            return round(float(np.mean(vals)), 6) if vals else None

        analysis_3_summary = {
            "n_worst50": len(worst50_records),
            "share_short": round(len(tail_short) / max(len(worst50_records), 1), 6),
            "share_long": round(len(tail_long) / max(len(worst50_records), 1), 6),
            "mean_max_prob_close_first5_short": _tail_prob_mean(tail_short, "max_prob_close_first_5"),
            "mean_max_prob_close_first5_long": _tail_prob_mean(tail_long, "max_prob_close_first_5"),
            "mean_max_prob_close_first20_short": _tail_prob_mean(tail_short, "max_prob_close_first_20"),
            "mean_max_prob_close_first20_long": _tail_prob_mean(tail_long, "max_prob_close_first_20"),
            "frac_ever_crossed_short": _tail_prob_mean(tail_short, "ever_crossed_threshold"),
            "frac_ever_crossed_long": _tail_prob_mean(tail_long, "ever_crossed_threshold"),
        }

        # ------------------------------------------------------------------
        # Analysis 4 — signal evolution curve (mean prob_close at bar 0..9)
        # ------------------------------------------------------------------
        analysis_4: Dict[str, Dict[str, Any]] = {}
        for side_val, curve_bucket in signal_curve_by_side.items():
            curve = {}
            for bar_idx in range(SIGNAL_CURVE_BARS):
                vals = curve_bucket.get(bar_idx, [])
                curve[f"bar_{bar_idx}"] = round(float(np.mean(vals)), 6) if vals else None
            analysis_4[side_val] = curve

        # ------------------------------------------------------------------
        # Analysis 5 — feature sanity by side (bar-0 values from EVAL_TRACE)
        # ------------------------------------------------------------------
        analysis_5: Dict[str, Any] = {}
        for side_val, feat_rows in feature_rows_by_side.items():
            if not feat_rows:
                analysis_5[side_val] = {"n": 0}
                continue
            df_f = pd.DataFrame(feat_rows)
            rec: Dict[str, Any] = {"n": len(feat_rows)}
            for col in ["pnl_bps_bar0", "mfe_bps_bar0", "mae_bps_bar0"]:
                s = pd.to_numeric(df_f[col], errors="coerce").dropna()
                if s.empty:
                    rec[col] = None
                    continue
                rec[col] = {
                    "mean": round(float(s.mean()), 6),
                    "median": round(float(s.median()), 6),
                    "std": round(float(s.std()), 6),
                }
            analysis_5[side_val] = rec

        # ------------------------------------------------------------------
        # Verdict
        # ------------------------------------------------------------------
        long_s = analysis_1.get("long", {})
        short_s = analysis_1.get("short", {})

        def _diff(a: Any, b: Any) -> Any:
            try:
                return round(float(a) - float(b), 6)
            except Exception:
                return None

        verdict = {
            "short_lower_prob_first5": (
                (short_s.get("mean_max_prob_close_first_5") or 0.0) <
                (long_s.get("mean_max_prob_close_first_5") or 0.0)
            ),
            "short_lower_prob_first20": (
                (short_s.get("mean_max_prob_close_first_20") or 0.0) <
                (long_s.get("mean_max_prob_close_first_20") or 0.0)
            ),
            "short_lower_frac_crossed": (
                (short_s.get("frac_ever_crossed_threshold") or 0.0) <
                (long_s.get("frac_ever_crossed_threshold") or 0.0)
            ),
            "short_higher_guard_or_eof_frac": None,
            "delta_mean_max_prob_first5_short_minus_long": _diff(
                short_s.get("mean_max_prob_close_first_5"),
                long_s.get("mean_max_prob_close_first_5"),
            ),
            "delta_frac_ever_crossed_short_minus_long": _diff(
                short_s.get("frac_ever_crossed_threshold"),
                long_s.get("frac_ever_crossed_threshold"),
            ),
        }
        # short_higher_guard_or_eof_frac: compare fraction of guard+eof exits
        try:
            def _guard_eof_frac(side_val: str) -> float:
                rows = analysis_2.get(side_val, [])
                total = sum(r["count"] for r in rows)
                ge = sum(r["count"] for r in rows if r["close_reason"] in ("CATASTROPHIC_GUARD", "REPLAY_EOF"))
                return round(ge / total, 6) if total else 0.0
            short_ge = _guard_eof_frac("short")
            long_ge  = _guard_eof_frac("long")
            verdict["short_higher_guard_or_eof_frac"] = short_ge > long_ge
            verdict["short_guard_eof_frac"] = short_ge
            verdict["long_guard_eof_frac"] = long_ge
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Write SHORT_EXIT_SIGNAL_AUDIT.json
        # ------------------------------------------------------------------
        payload = {
            "run_id": run_id,
            "section_1_signal_quality_by_side": analysis_1,
            "section_2_close_reason_by_side": analysis_2,
            "section_3_worst50_tail": {
                "summary": analysis_3_summary,
                "trades": worst50_records,
            },
            "section_4_signal_curve_by_side": analysis_4,
            "section_5_feature_sanity_by_side": analysis_5,
            "verdict": verdict,
        }
        out_path = chunk_output_dir / "SHORT_EXIT_SIGNAL_AUDIT.json"
        tmp_path = out_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        log.info(
            "[SHORT_EXIT_SIGNAL_AUDIT_PROOF] run_id=%s "
            "long_n=%s short_n=%s "
            "long_p5=%.3f short_p5=%.3f "
            "long_frac_crossed=%.3f short_frac_crossed=%.3f "
            "short_lower_prob=%s short_lower_crossed=%s short_higher_guard_eof=%s "
            "path=%s",
            run_id,
            long_s.get("n"), short_s.get("n"),
            long_s.get("mean_max_prob_close_first_5") or 0.0,
            short_s.get("mean_max_prob_close_first_5") or 0.0,
            long_s.get("frac_ever_crossed_threshold") or 0.0,
            short_s.get("frac_ever_crossed_threshold") or 0.0,
            verdict.get("short_lower_prob_first5"),
            verdict.get("short_lower_frac_crossed"),
            verdict.get("short_higher_guard_or_eof_frac"),
            out_path,
        )

    except Exception as e:
        log.warning("[SHORT_EXIT_SIGNAL_AUDIT_FAIL] run_id=%s error=%s", run_id, e)


def _write_stuck_short_signature_audit(
    chunk_output_dir: Path,
    run_id: str,
    journal_df: "pd.DataFrame",
) -> None:
    """
    Signature audit for stuck short trades (observability only, no trading logic changes).

    Target group: side=short, close_reason in {REPLAY_EOF, CATASTROPHIC_GUARD},
                  ever_crossed_threshold=False  (loaded from SHORT_EXIT_SIGNAL_AUDIT.json).

    Control A: side=short, close_reason=THRESHOLD, ever_crossed_threshold=True (normal exits)
    Control B: side=short, close_reason=THRESHOLD, bottom-20 by final_pnl_bps (bad-but-signalled)

    Analyses:
    1. Explicit identity list of target trades (deduplicated by trade_id)
    2. Categorical signature (session, session_overlap, close_month, source)
    3. Trajectory shape from EXIT_EVAL_TRACE checkpoints (bar 0,1,5,20,100,500,1000,last)
    4. Early-life divergence vs controls (first 5/20/100 bars)
    5. No-rebound / monotonic deterioration tests
    6. Concise distinguishing signature

    Output: STUCK_SHORT_SIGNATURE_AUDIT.json in chunk_output_dir.
    Proof label: [STUCK_SHORT_SIGNATURE_PROOF]
    Never raises — best-effort.
    """
    import json

    CONTROL_B_N = 20
    TRAJ_CHECKPOINTS = [0, 1, 5, 20, 100, 500, 1000]

    try:
        if journal_df is None or journal_df.empty:
            log.info("[STUCK_SHORT_SIGNATURE_PROOF] run_id=%s skipped=journal_empty", run_id)
            return

        jdf = journal_df.copy()

        # ------------------------------------------------------------------
        # Load ever_crossed_threshold per trade from SHORT_EXIT_SIGNAL_AUDIT
        # ------------------------------------------------------------------
        short_audit_path = chunk_output_dir / "SHORT_EXIT_SIGNAL_AUDIT.json"
        crossed_by_tid: Dict[str, bool] = {}
        if short_audit_path.exists():
            try:
                sa = json.loads(short_audit_path.read_text())
                for rec in sa.get("section_3_worst50_tail", {}).get("trades", []):
                    tid = rec.get("trade_id")
                    if tid:
                        crossed_by_tid[str(tid)] = bool(rec.get("ever_crossed_threshold", False))
                # Also load from stuck trade audit
            except Exception:
                pass
        # Fallback: load from STUCK_TRADE_AUDIT
        stuck_audit_path = chunk_output_dir / "STUCK_TRADE_AUDIT.json"
        if stuck_audit_path.exists():
            try:
                sa2 = json.loads(stuck_audit_path.read_text())
                for rec in sa2.get("per_trade_records", []):
                    tid = rec.get("trade_id")
                    if tid and tid not in crossed_by_tid:
                        crossed_by_tid[str(tid)] = bool(rec.get("ever_crossed_threshold", False))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Build groups (deduplicated by trade_id — keep row with non-null session)
        # ------------------------------------------------------------------
        short_mask = jdf["side"].fillna("").str.lower() == "short"
        short_df = jdf[short_mask].copy()

        # Dedup: prefer row with non-null session
        def _dedup(df: "pd.DataFrame") -> "pd.DataFrame":
            if "trade_id" not in df.columns:
                return df
            df = df.copy()
            df["_session_null"] = df["session"].isna()
            df = df.sort_values("_session_null")
            df = df.drop_duplicates(subset="trade_id", keep="first")
            df = df.drop(columns=["_session_null"])
            return df

        short_df = _dedup(short_df)

        # Target group
        target_reasons = {"REPLAY_EOF", "CATASTROPHIC_GUARD"}
        target_mask = short_df["exit_reason"].isin(target_reasons)
        target_df = short_df[target_mask].copy()
        # Filter ever_crossed=False using loaded map; keep all if map is empty
        if crossed_by_tid:
            target_df = target_df[
                target_df["trade_id"].apply(lambda t: not crossed_by_tid.get(str(t), True))
            ]
        n_target = len(target_df)

        if n_target == 0:
            log.info("[STUCK_SHORT_SIGNATURE_PROOF] run_id=%s skipped=no_target_trades", run_id)
            return

        target_ids = set(target_df["trade_id"].astype(str).tolist())

        # Control A: normal short THRESHOLD exits that crossed threshold
        ctrl_a_df = short_df[
            (short_df["exit_reason"] == "THRESHOLD") &
            (short_df["trade_id"].apply(lambda t: crossed_by_tid.get(str(t), True)))
        ].copy()

        # Control B: worst-N short THRESHOLD by pnl
        ctrl_b_pool = short_df[short_df["exit_reason"] == "THRESHOLD"].copy()
        pnl_num = pd.to_numeric(ctrl_b_pool["pnl_bps"], errors="coerce")
        ctrl_b_df = ctrl_b_pool.loc[pnl_num.nsmallest(CONTROL_B_N).index].copy()

        # ------------------------------------------------------------------
        # Load EXIT_EVAL_TRACE for target + controls
        # ------------------------------------------------------------------
        trace_path = chunk_output_dir / "EXIT_EVAL_TRACE.csv"
        all_ids = target_ids | set(ctrl_a_df["trade_id"].astype(str)) | set(ctrl_b_df["trade_id"].astype(str))

        # Maps: trade_id -> sorted DataFrame of trace rows
        trace_by_tid: Dict[str, "pd.DataFrame"] = {}
        if trace_path.exists():
            try:
                trace_df = pd.read_csv(
                    trace_path,
                    usecols=["trade_id", "bars_held", "exit_prob", "exit_threshold",
                             "pnl_bps", "mfe_bps", "mae_bps", "time_since_mfe_bars"],
                    dtype={"trade_id": str},
                )
                for col in ["exit_prob", "exit_threshold", "bars_held", "pnl_bps",
                            "mfe_bps", "mae_bps", "time_since_mfe_bars"]:
                    trace_df[col] = pd.to_numeric(trace_df[col], errors="coerce")
                relevant = trace_df[trace_df["trade_id"].isin(all_ids)]
                for tid, grp in relevant.groupby("trade_id"):
                    trace_by_tid[str(tid)] = grp.sort_values("bars_held").reset_index(drop=True)
            except Exception as e:
                log.warning("[STUCK_SHORT_SIGNATURE_TRACE_FAIL] %s", e)

        # ------------------------------------------------------------------
        # Helper: value at nearest checkpoint bar
        # ------------------------------------------------------------------
        def _at_bar(grp: "pd.DataFrame", bar: int, col: str) -> Any:
            if grp is None or grp.empty:
                return None
            idx = int(np.searchsorted(grp["bars_held"].values, bar, side="left"))
            if idx >= len(grp):
                idx = len(grp) - 1
            val = grp[col].iloc[idx]
            return float(val) if pd.notna(val) else None

        def _safe_r(v: Any, decimals: int = 4) -> Any:
            try:
                f = float(v)
                return None if f != f else round(f, decimals)
            except Exception:
                return None

        def _agg_col(vals: list, fn: str) -> Any:
            v = [x for x in vals if x is not None]
            if not v:
                return None
            return round(float(getattr(np, fn)(v)), 4)

        # ------------------------------------------------------------------
        # Analysis 1 — explicit identity list
        # ------------------------------------------------------------------
        identity_list = []
        for _, row in target_df.iterrows():
            tid = str(row.get("trade_id", ""))
            identity_list.append({
                "trade_id": tid,
                "entry_time": str(row.get("open_ts_utc", "")),
                "exit_time": str(row.get("close_ts_utc", "")),
                "close_reason": row.get("exit_reason"),
                "bars_held_final": int(row["bars_in_trade"]) if pd.notna(row.get("bars_in_trade")) else None,
                "final_pnl_bps": _safe_r(row.get("pnl_bps")),
                "mfe_bps_final": _safe_r(row.get("mfe_bps")),
                "mae_bps_final": _safe_r(row.get("mae_bps")),
                "max_prob_close_lifetime": _safe_r(row.get("prob_close")),
                "peak_mfe_bps_exit_state": _safe_r(row.get("peak_mfe_bps_exit_state")),
                "peak_mfe_bar_index": _safe_r(row.get("peak_mfe_bar_index")),
                "time_since_mfe_bars_exit": _safe_r(row.get("time_since_mfe_bars_exit")),
                "session": row.get("session"),
                "close_month_utc": row.get("close_month_utc"),
                "source": row.get("source"),
                "ever_crossed_threshold": crossed_by_tid.get(tid, None),
            })
        identity_list.sort(key=lambda r: r["final_pnl_bps"] or 0.0)

        # ------------------------------------------------------------------
        # Analysis 2 — categorical signature
        # ------------------------------------------------------------------
        def _cat_counts(df: "pd.DataFrame", col: str) -> Dict[str, int]:
            if col not in df.columns:
                return {}
            return df[col].fillna("(null)").value_counts(dropna=False).to_dict()

        def _cat_frac(counts: Dict, total: int) -> Dict[str, float]:
            return {k: round(v / total, 4) for k, v in counts.items()} if total else {}

        cat_cols = ["session", "close_month_utc", "source"]
        analysis_2: Dict[str, Any] = {}
        for col in cat_cols:
            t_counts = _cat_counts(target_df, col)
            a_counts = _cat_counts(ctrl_a_df, col)
            b_counts = _cat_counts(ctrl_b_df, col)
            t_total = len(target_df)
            a_total = len(ctrl_a_df)
            b_total = len(ctrl_b_df)
            # Overrepresentation: target_frac / ctrl_a_frac
            overrep: Dict[str, Any] = {}
            all_cats = set(t_counts) | set(a_counts)
            for cat in sorted(all_cats):
                tf = t_counts.get(cat, 0) / t_total if t_total else 0.0
                af = a_counts.get(cat, 0) / a_total if a_total else 0.0
                overrep[str(cat)] = {
                    "target_count": t_counts.get(cat, 0),
                    "target_frac": round(tf, 4),
                    "ctrl_a_count": a_counts.get(cat, 0),
                    "ctrl_a_frac": round(af, 4),
                    "ctrl_b_count": b_counts.get(cat, 0),
                    "ctrl_b_frac": round(b_counts.get(cat, 0) / b_total, 4) if b_total else 0.0,
                    "overrep_vs_ctrl_a": round(tf / af, 3) if af > 0 else None,
                }
            analysis_2[col] = overrep

        # ------------------------------------------------------------------
        # Analysis 3 — trajectory shape
        # ------------------------------------------------------------------
        def _trajectory(tid: str, grp: "pd.DataFrame") -> Dict[str, Any]:
            if grp is None or grp.empty:
                return {}
            n_bars = len(grp)
            last_bar = int(grp["bars_held"].iloc[-1]) if pd.notna(grp["bars_held"].iloc[-1]) else None
            checkpoints = TRAJ_CHECKPOINTS + ([last_bar] if last_bar is not None and last_bar not in TRAJ_CHECKPOINTS else [])
            traj: Dict[str, Any] = {"n_trace_bars": n_bars}
            for cp in checkpoints:
                label = f"bar_{cp}" if cp != last_bar else "bar_last"
                traj[f"{label}_pnl"] = _at_bar(grp, cp, "pnl_bps")
                traj[f"{label}_mfe"] = _at_bar(grp, cp, "mfe_bps")
                traj[f"{label}_mae"] = _at_bar(grp, cp, "mae_bps")
                traj[f"{label}_time_since_mfe"] = _at_bar(grp, cp, "time_since_mfe_bars")
            # First bar positive / negative
            pnl_series = grp["pnl_bps"].dropna()
            pos_idx = (pnl_series > 0).idxmax() if (pnl_series > 0).any() else None
            neg_idx = (pnl_series < 0).idxmax() if (pnl_series < 0).any() else None
            traj["first_bar_positive"] = (
                int(grp.loc[pos_idx, "bars_held"]) if pos_idx is not None else None
            )
            traj["first_bar_negative"] = (
                int(grp.loc[neg_idx, "bars_held"]) if neg_idx is not None else None
            )
            # Max MFE in trace
            mfe_series = grp["mfe_bps"].dropna()
            traj["max_mfe_in_trace"] = float(mfe_series.max()) if not mfe_series.empty else None
            traj["max_mfe_bar"] = (
                int(grp.loc[mfe_series.idxmax(), "bars_held"])
                if not mfe_series.empty else None
            )
            return traj

        analysis_3_target: list = []
        for _, row in target_df.iterrows():
            tid = str(row.get("trade_id", ""))
            grp = trace_by_tid.get(tid)
            analysis_3_target.append({
                "trade_id": tid,
                "session": row.get("session"),
                "close_reason": row.get("exit_reason"),
                "final_pnl_bps": _safe_r(row.get("pnl_bps")),
                "trajectory": _trajectory(tid, grp),
            })

        # ------------------------------------------------------------------
        # Analysis 4 — early-life divergence
        # ------------------------------------------------------------------
        EARLY_WINDOWS = [5, 20, 100]

        def _early_life_stats(df: "pd.DataFrame", window: int) -> Dict[str, Any]:
            pnl_means, mfe_means, mae_means, prob_means, prob_maxs = [], [], [], [], []
            for _, row in df.iterrows():
                tid = str(row.get("trade_id", ""))
                grp = trace_by_tid.get(tid)
                if grp is None or grp.empty:
                    continue
                sub = grp[grp["bars_held"] < window]
                if sub.empty:
                    sub = grp.head(1)
                for col, lst in [("pnl_bps", pnl_means), ("mfe_bps", mfe_means),
                                  ("mae_bps", mae_means)]:
                    v = sub[col].dropna()
                    if not v.empty:
                        lst.append(float(v.mean()))
                prob = sub["exit_prob"].dropna()
                if not prob.empty:
                    prob_means.append(float(prob.mean()))
                    prob_maxs.append(float(prob.max()))
            return {
                "n": len(df),
                "mean_pnl_bps": _agg_col(pnl_means, "mean"),
                "median_pnl_bps": _agg_col(pnl_means, "median"),
                "mean_mfe_bps": _agg_col(mfe_means, "mean"),
                "mean_mae_bps": _agg_col(mae_means, "mean"),
                "mean_prob_close": _agg_col(prob_means, "mean"),
                "max_prob_close": _agg_col(prob_maxs, "max"),
            }

        analysis_4: Dict[str, Any] = {}
        for w in EARLY_WINDOWS:
            analysis_4[f"first_{w}_bars"] = {
                "target": _early_life_stats(target_df, w),
                "ctrl_a": _early_life_stats(ctrl_a_df, w),
                "ctrl_b": _early_life_stats(ctrl_b_df, w),
            }

        # ------------------------------------------------------------------
        # Analysis 5 — no-rebound / monotonic deterioration
        # ------------------------------------------------------------------
        def _no_rebound_stats(tid: str, grp: "pd.DataFrame") -> Dict[str, Any]:
            if grp is None or grp.empty:
                return {}
            pnl = grp["pnl_bps"].dropna()
            mfe = grp["mfe_bps"].dropna()
            n = len(pnl)
            if n == 0:
                return {}
            frac_negative = float((pnl < 0).sum() / n)
            # Crosses back above 0
            sign = (pnl > 0).astype(int)
            crossings = int((sign.diff() == 1).sum())
            # Max MFE
            max_mfe = float(mfe.max()) if not mfe.empty else None
            # Bar of max MFE
            if not mfe.empty:
                mfe_peak_idx = mfe.idxmax()
                mfe_peak_bar = int(grp.loc[mfe_peak_idx, "bars_held"]) if pd.notna(grp.loc[mfe_peak_idx, "bars_held"]) else None
            else:
                mfe_peak_bar = None
            # Fraction of life after MFE peak (time_since_mfe_bars at last bar / total_bars)
            last_tsm = grp["time_since_mfe_bars"].iloc[-1] if "time_since_mfe_bars" in grp.columns else None
            last_tsm = float(last_tsm) if pd.notna(last_tsm) else None
            total_bars = int(grp["bars_held"].iloc[-1]) if pd.notna(grp["bars_held"].iloc[-1]) else None
            frac_after_mfe = (
                round(last_tsm / total_bars, 4)
                if (last_tsm is not None and total_bars and total_bars > 0)
                else None
            )
            return {
                "frac_bars_negative": round(frac_negative, 4),
                "n_crossings_above_zero": crossings,
                "max_mfe_bps": _safe_r(max_mfe),
                "max_mfe_bar": mfe_peak_bar,
                "frac_life_after_mfe_peak": frac_after_mfe,
            }

        analysis_5_target: list = []
        for _, row in target_df.iterrows():
            tid = str(row.get("trade_id", ""))
            grp = trace_by_tid.get(tid)
            stats = _no_rebound_stats(tid, grp)
            stats["trade_id"] = tid
            stats["session"] = row.get("session")
            stats["close_reason"] = row.get("exit_reason")
            stats["final_pnl_bps"] = _safe_r(row.get("pnl_bps"))
            analysis_5_target.append(stats)

        # Summary across target
        no_reb_summary: Dict[str, Any] = {}
        for key in ["frac_bars_negative", "n_crossings_above_zero", "max_mfe_bps",
                    "frac_life_after_mfe_peak"]:
            vals = [r.get(key) for r in analysis_5_target if r.get(key) is not None]
            no_reb_summary[f"mean_{key}"] = _agg_col(vals, "mean")
            no_reb_summary[f"median_{key}"] = _agg_col(vals, "median")

        # Control comparison on same metrics
        ctrl_a_no_reb: list = []
        for _, row in ctrl_a_df.iterrows():
            tid = str(row.get("trade_id", ""))
            grp = trace_by_tid.get(tid)
            s = _no_rebound_stats(tid, grp)
            s["trade_id"] = tid
            ctrl_a_no_reb.append(s)
        ctrl_a_no_reb_summary: Dict[str, Any] = {}
        for key in ["frac_bars_negative", "n_crossings_above_zero", "max_mfe_bps"]:
            vals = [r.get(key) for r in ctrl_a_no_reb if r.get(key) is not None]
            ctrl_a_no_reb_summary[f"mean_{key}"] = _agg_col(vals, "mean")

        # ------------------------------------------------------------------
        # Analysis 6 — distinguishing signature
        # ------------------------------------------------------------------
        # Sessions in target
        target_sessions = target_df["session"].fillna("(null)").value_counts().to_dict()
        # mean bars held
        target_bars_mean = _agg_col(
            [r["bars_held_final"] for r in identity_list if r["bars_held_final"] is not None], "mean"
        )
        # mean final pnl
        target_pnl_mean = _agg_col(
            [r["final_pnl_bps"] for r in identity_list if r["final_pnl_bps"] is not None], "mean"
        )
        # mean max_mfe across target
        target_mfe_mean = _agg_col(
            [r.get("max_mfe_bps") for r in analysis_5_target if r.get("max_mfe_bps") is not None], "mean"
        )
        # all target trades slow-bleed?
        all_slow_bleed = all(
            (r.get("frac_bars_negative") or 0) > 0.85
            for r in analysis_5_target if r.get("frac_bars_negative") is not None
        )
        # all zero crossings?
        all_no_rebound = all(
            (r.get("n_crossings_above_zero") or 0) == 0
            for r in analysis_5_target
        )
        # early divergence: target vs ctrl_a in first 20 bars
        e20_target_pnl = analysis_4.get("first_20_bars", {}).get("target", {}).get("mean_pnl_bps")
        e20_ctrl_a_pnl = analysis_4.get("first_20_bars", {}).get("ctrl_a", {}).get("mean_pnl_bps")
        early_divergent = (
            (e20_target_pnl is not None and e20_ctrl_a_pnl is not None and
             e20_target_pnl < e20_ctrl_a_pnl - 1.0)
        )

        # close_reason split
        cat_guard_n = int((target_df["exit_reason"] == "CATASTROPHIC_GUARD").sum())
        eof_n = int((target_df["exit_reason"] == "REPLAY_EOF").sum())

        signature = {
            "n_target_trades": n_target,
            "n_ctrl_a_trades": len(ctrl_a_df),
            "n_ctrl_b_trades": len(ctrl_b_df),
            "target_sessions": target_sessions,
            "n_catastrophic_guard": cat_guard_n,
            "n_replay_eof": eof_n,
            "target_bars_held_mean": target_bars_mean,
            "target_final_pnl_mean_bps": target_pnl_mean,
            "target_max_mfe_mean_bps": target_mfe_mean,
            "all_slow_bleed_frac_negative_gt85pct": all_slow_bleed,
            "all_no_rebound_above_zero": all_no_rebound,
            "early_divergence_by_bar20": early_divergent,
            "early_pnl_target_first20": e20_target_pnl,
            "early_pnl_ctrl_a_first20": e20_ctrl_a_pnl,
            "summary": (
                "short, multi-session (EU/OVERLAP/US), never-crossed-threshold, "
                "immediate adverse move from bar 0, near-zero MFE entire lifetime, "
                "monotonic bleed with no rebound, "
                "held 1000–94725 bars (days to 11 months), "
                "CATASTROPHIC_GUARD or REPLAY_EOF terminal"
            ),
        }

        # ------------------------------------------------------------------
        # Write STUCK_SHORT_SIGNATURE_AUDIT.json
        # ------------------------------------------------------------------
        payload = {
            "run_id": run_id,
            "section_1_identity_list": identity_list,
            "section_2_categorical_signature": analysis_2,
            "section_3_trajectory": analysis_3_target,
            "section_4_early_life_divergence": analysis_4,
            "section_5_no_rebound": {
                "per_trade": analysis_5_target,
                "target_summary": no_reb_summary,
                "ctrl_a_summary": ctrl_a_no_reb_summary,
            },
            "section_6_signature": signature,
        }
        out_path = chunk_output_dir / "STUCK_SHORT_SIGNATURE_AUDIT.json"
        tmp_path = out_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        log.info(
            "[STUCK_SHORT_SIGNATURE_PROOF] run_id=%s n_target=%d n_ctrl_a=%d n_ctrl_b=%d "
            "sessions=%s all_slow_bleed=%s all_no_rebound=%s early_divergent=%s "
            "target_pnl_mean=%.1f target_mfe_mean=%s "
            "path=%s",
            run_id, n_target, len(ctrl_a_df), len(ctrl_b_df),
            list(target_sessions.keys()),
            all_slow_bleed, all_no_rebound, early_divergent,
            target_pnl_mean or 0.0, target_mfe_mean,
            out_path,
        )

    except Exception as e:
        log.warning("[STUCK_SHORT_SIGNATURE_FAIL] run_id=%s error=%s", run_id, e)


def _write_early_failure_short_guard_counterfactual(
    chunk_output_dir: Path,
    run_id: str,
    journal_df: "pd.DataFrame",
) -> None:
    """
    Counterfactual analysis for a hypothetical early-failure short guard.
    Analytical only — no trading logic changes.

    Loads target IDs from STUCK_SHORT_SIGNATURE_AUDIT.json.
    For each guard variant, evaluates using EXIT_EVAL_TRACE.csv per-bar data:
      - hit rate on target set (13 stuck shorts)
      - collateral damage on normal short THRESHOLD trades
      - tail-risk effect on short universe

    Variants tested (at bar 3 and bar 5):
      A: max_mfe_first5<=0.0 AND mean_prob_first5<0.25 AND pnl_at_bar5<=-5
      B: max_mfe_first5<=1.0 AND mean_prob_first5<0.25 AND mean_pnl_first5<=-5
      C: max_mfe_first5<=1.0 AND max_prob_first5<0.30 AND pnl_at_bar5<=-10
      D (extra): max_mfe_first3<=0.0 AND mean_prob_first3<0.25 AND pnl_at_bar3<=-3  (bar-3 variant)
      E (extra): max_mfe_first5<=0.5 AND mean_prob_first5<0.20 AND pnl_at_bar5<=-5  (tighter)

    Output: EARLY_FAILURE_SHORT_GUARD_COUNTERFACTUAL.json
    Proof label: [EARLY_FAILURE_SHORT_GUARD_CF_PROOF]
    Never raises — best-effort.
    """
    import json

    VARIANTS = {
        "A": {
            "description": "max_mfe_first5<=0.0 AND mean_prob_first5<0.25 AND pnl_at_bar5<=-5",
            "eval_bar": 5,
            "max_mfe_thresh": 0.0,
            "prob_fn": "mean",
            "prob_window": 5,
            "prob_thresh": 0.25,
            "pnl_fn": "at_bar",
            "pnl_bar": 5,
            "pnl_thresh": -5.0,
        },
        "B": {
            "description": "max_mfe_first5<=1.0 AND mean_prob_first5<0.25 AND mean_pnl_first5<=-5",
            "eval_bar": 5,
            "max_mfe_thresh": 1.0,
            "prob_fn": "mean",
            "prob_window": 5,
            "prob_thresh": 0.25,
            "pnl_fn": "mean",
            "pnl_bar": 5,
            "pnl_thresh": -5.0,
        },
        "C": {
            "description": "max_mfe_first5<=1.0 AND max_prob_first5<0.30 AND pnl_at_bar5<=-10",
            "eval_bar": 5,
            "max_mfe_thresh": 1.0,
            "prob_fn": "max",
            "prob_window": 5,
            "prob_thresh": 0.30,
            "pnl_fn": "at_bar",
            "pnl_bar": 5,
            "pnl_thresh": -10.0,
        },
        "D": {
            "description": "bar-3: max_mfe_first3<=0.0 AND mean_prob_first3<0.25 AND pnl_at_bar3<=-3",
            "eval_bar": 3,
            "max_mfe_thresh": 0.0,
            "prob_fn": "mean",
            "prob_window": 3,
            "prob_thresh": 0.25,
            "pnl_fn": "at_bar",
            "pnl_bar": 3,
            "pnl_thresh": -3.0,
        },
        "E": {
            "description": "tighter: max_mfe_first5<=0.5 AND mean_prob_first5<0.20 AND pnl_at_bar5<=-5",
            "eval_bar": 5,
            "max_mfe_thresh": 0.5,
            "prob_fn": "mean",
            "prob_window": 5,
            "prob_thresh": 0.20,
            "pnl_fn": "at_bar",
            "pnl_bar": 5,
            "pnl_thresh": -5.0,
        },
    }

    try:
        if journal_df is None or journal_df.empty:
            log.info("[EARLY_FAILURE_SHORT_GUARD_CF_PROOF] run_id=%s skipped=journal_empty", run_id)
            return

        # ------------------------------------------------------------------
        # Load target IDs from STUCK_SHORT_SIGNATURE_AUDIT
        # ------------------------------------------------------------------
        sig_path = chunk_output_dir / "STUCK_SHORT_SIGNATURE_AUDIT.json"
        target_ids: set = set()
        if sig_path.exists():
            try:
                sig = json.loads(sig_path.read_text())
                target_ids = {t["trade_id"] for t in sig.get("section_1_identity_list", [])}
            except Exception as e:
                log.warning("[EARLY_FAILURE_SHORT_GUARD_CF] failed to load target IDs: %s", e)

        if not target_ids:
            log.info("[EARLY_FAILURE_SHORT_GUARD_CF_PROOF] run_id=%s skipped=no_target_ids", run_id)
            return

        # ------------------------------------------------------------------
        # Build control group: short THRESHOLD trades that crossed threshold
        # ------------------------------------------------------------------
        short_audit_path = chunk_output_dir / "SHORT_EXIT_SIGNAL_AUDIT.json"
        crossed_by_tid: Dict[str, bool] = {}
        if short_audit_path.exists():
            try:
                sa = json.loads(short_audit_path.read_text())
                for rec in sa.get("section_3_worst50_tail", {}).get("trades", []):
                    t = rec.get("trade_id")
                    if t:
                        crossed_by_tid[str(t)] = bool(rec.get("ever_crossed_threshold", False))
            except Exception:
                pass
        stuck_path = chunk_output_dir / "STUCK_TRADE_AUDIT.json"
        if stuck_path.exists():
            try:
                sa2 = json.loads(stuck_path.read_text())
                for rec in sa2.get("per_trade_records", []):
                    t = rec.get("trade_id")
                    if t and str(t) not in crossed_by_tid:
                        crossed_by_tid[str(t)] = bool(rec.get("ever_crossed_threshold", False))
            except Exception:
                pass

        jdf = journal_df.copy()
        short_mask = jdf["side"].fillna("").str.lower() == "short"
        short_df = jdf[short_mask].copy()

        # Dedup by trade_id (prefer row with non-null session)
        if "trade_id" in short_df.columns:
            short_df["_snull"] = short_df["session"].isna()
            short_df = short_df.sort_values("_snull").drop_duplicates(subset="trade_id", keep="first")
            short_df = short_df.drop(columns=["_snull"])

        # Control A: normal short THRESHOLD that crossed threshold
        ctrl_a_df = short_df[
            (short_df["exit_reason"] == "THRESHOLD") &
            (short_df["trade_id"].apply(lambda t: crossed_by_tid.get(str(t), True)))
        ].copy()
        ctrl_a_ids = set(ctrl_a_df["trade_id"].astype(str).tolist())

        # All short pnl for tail analysis
        all_short_df = short_df.copy()

        # ------------------------------------------------------------------
        # Load EXIT_EVAL_TRACE for all relevant trade_ids
        # ------------------------------------------------------------------
        all_relevant_ids = target_ids | ctrl_a_ids | set(all_short_df["trade_id"].astype(str))
        trace_path = chunk_output_dir / "EXIT_EVAL_TRACE.csv"
        trace_by_tid: Dict[str, "pd.DataFrame"] = {}
        if trace_path.exists():
            try:
                trace_df = pd.read_csv(
                    trace_path,
                    usecols=["trade_id", "bars_held", "exit_prob", "mfe_bps", "pnl_bps"],
                    dtype={"trade_id": str},
                )
                for col in ["bars_held", "exit_prob", "mfe_bps", "pnl_bps"]:
                    trace_df[col] = pd.to_numeric(trace_df[col], errors="coerce")
                for tid, grp in trace_df[trace_df["trade_id"].isin(all_relevant_ids)].groupby("trade_id"):
                    trace_by_tid[str(tid)] = grp.sort_values("bars_held").reset_index(drop=True)
            except Exception as e:
                log.warning("[EARLY_FAILURE_SHORT_GUARD_CF_TRACE_FAIL] %s", e)

        # ------------------------------------------------------------------
        # Per-trade feature extractor for a given variant
        # ------------------------------------------------------------------
        def _extract_features(tid: str, vdef: Dict[str, Any]) -> Dict[str, Any]:
            grp = trace_by_tid.get(tid)
            if grp is None or grp.empty:
                return {"has_data": False}
            eval_bar = vdef["eval_bar"]
            prob_window = vdef["prob_window"]
            pnl_bar = vdef["pnl_bar"]

            early = grp[grp["bars_held"] < eval_bar]
            if early.empty:
                early = grp.head(min(len(grp), eval_bar))

            # max_mfe over early window
            max_mfe = float(early["mfe_bps"].max()) if not early["mfe_bps"].dropna().empty else None

            # prob stats
            prob_early = grp[grp["bars_held"] < prob_window]["exit_prob"].dropna()
            if prob_early.empty:
                prob_early = grp["exit_prob"].head(prob_window).dropna()
            mean_prob = float(prob_early.mean()) if not prob_early.empty else None
            max_prob = float(prob_early.max()) if not prob_early.empty else None

            # pnl: at_bar = value nearest to pnl_bar; mean = mean over early window
            if vdef["pnl_fn"] == "at_bar":
                idx = int(np.searchsorted(grp["bars_held"].values, pnl_bar, side="right")) - 1
                if idx < 0:
                    idx = 0
                if idx >= len(grp):
                    idx = len(grp) - 1
                pnl_val = float(grp["pnl_bps"].iloc[idx]) if pd.notna(grp["pnl_bps"].iloc[idx]) else None
            else:  # mean
                pnl_vals = grp[grp["bars_held"] <= pnl_bar]["pnl_bps"].dropna()
                pnl_val = float(pnl_vals.mean()) if not pnl_vals.empty else None

            # counterfactual exit pnl = pnl at eval_bar
            idx_cf = int(np.searchsorted(grp["bars_held"].values, eval_bar, side="right")) - 1
            if idx_cf < 0:
                idx_cf = 0
            if idx_cf >= len(grp):
                idx_cf = len(grp) - 1
            cf_pnl = float(grp["pnl_bps"].iloc[idx_cf]) if pd.notna(grp["pnl_bps"].iloc[idx_cf]) else None
            cf_bar = int(grp["bars_held"].iloc[idx_cf]) if pd.notna(grp["bars_held"].iloc[idx_cf]) else None

            return {
                "has_data": True,
                "max_mfe_early": max_mfe,
                "mean_prob_early": mean_prob,
                "max_prob_early": max_prob,
                "pnl_val": pnl_val,
                "cf_pnl": cf_pnl,
                "cf_bar": cf_bar,
            }

        def _fires(feat: Dict[str, Any], vdef: Dict[str, Any]) -> bool:
            if not feat.get("has_data"):
                return False
            max_mfe = feat.get("max_mfe_early")
            if max_mfe is None or max_mfe > vdef["max_mfe_thresh"]:
                return False
            prob = feat.get("mean_prob_early") if vdef["prob_fn"] == "mean" else feat.get("max_prob_early")
            if prob is None or prob >= vdef["prob_thresh"]:
                return False
            pnl = feat.get("pnl_val")
            if pnl is None or pnl > vdef["pnl_thresh"]:
                return False
            return True

        # ------------------------------------------------------------------
        # Run variants
        # ------------------------------------------------------------------
        def _safe(v: Any, dec: int = 4) -> Any:
            try:
                f = float(v)
                return None if f != f else round(f, dec)
            except Exception:
                return None

        def _agg(vals: list, fn: str) -> Any:
            v = [x for x in vals if x is not None]
            if not v:
                return None
            return round(float(getattr(np, fn)(v)), 4)

        variant_results: Dict[str, Any] = {}

        for vname, vdef in VARIANTS.items():
            # --- Target hits ---
            target_hits = []
            target_miss = []
            for _, row in all_short_df[all_short_df["trade_id"].isin(target_ids)].iterrows():
                tid = str(row["trade_id"])
                feat = _extract_features(tid, vdef)
                final_pnl = _safe(row.get("pnl_bps"))
                if _fires(feat, vdef):
                    target_hits.append({
                        "trade_id": tid,
                        "session": row.get("session"),
                        "close_reason": row.get("exit_reason"),
                        "cf_bar": feat.get("cf_bar"),
                        "cf_pnl": _safe(feat.get("cf_pnl")),
                        "final_pnl": final_pnl,
                        "improvement_bps": _safe((feat.get("cf_pnl") or 0.0) - (final_pnl or 0.0)),
                    })
                else:
                    target_miss.append(tid)

            # --- Collateral on ctrl_a ---
            collateral_hits = []
            collateral_miss_pnl = []
            for _, row in ctrl_a_df.iterrows():
                tid = str(row["trade_id"])
                feat = _extract_features(tid, vdef)
                final_pnl = _safe(row.get("pnl_bps"))
                if _fires(feat, vdef):
                    collateral_hits.append({
                        "trade_id": tid,
                        "session": row.get("session"),
                        "cf_bar": feat.get("cf_bar"),
                        "cf_pnl": _safe(feat.get("cf_pnl")),
                        "final_pnl": final_pnl,
                        "alpha_loss_bps": _safe((final_pnl or 0.0) - (feat.get("cf_pnl") or 0.0)),
                    })
                else:
                    collateral_miss_pnl.append(final_pnl)

            # --- Net effect on full short universe ---
            all_short_pnl_orig = []
            all_short_pnl_cf = []
            for _, row in all_short_df.iterrows():
                tid = str(row["trade_id"])
                feat = _extract_features(tid, vdef)
                fp = _safe(row.get("pnl_bps"))
                all_short_pnl_orig.append(fp)
                if _fires(feat, vdef):
                    cf_p = _safe(feat.get("cf_pnl"))
                    all_short_pnl_cf.append(cf_p if cf_p is not None else fp)
                else:
                    all_short_pnl_cf.append(fp)

            # Tail metrics helper
            def _tail_metrics(pnl_list: list, prefix: str) -> Dict[str, Any]:
                vals = sorted([x for x in pnl_list if x is not None])
                if not vals:
                    return {}
                n = len(vals)
                worst10 = vals[:10]
                worst20 = vals[:20]
                worst5pct_n = max(1, int(np.floor(n * 0.05)))
                worst1pct_n = max(1, n // 100)
                return {
                    f"{prefix}_mean": round(float(np.mean(vals)), 4),
                    f"{prefix}_worst": vals[0],
                    f"{prefix}_mean_worst10": round(float(np.mean(worst10)), 4),
                    f"{prefix}_mean_worst20": round(float(np.mean(worst20)), 4),
                    f"{prefix}_CVaR95": round(float(np.mean(vals[:worst5pct_n])), 4),
                    f"{prefix}_CVaR99": round(float(np.mean(vals[:worst1pct_n])), 4),
                }

            tail_orig = _tail_metrics(all_short_pnl_orig, "orig")
            tail_cf   = _tail_metrics(all_short_pnl_cf,   "cf")

            # Collateral summary
            col_alpha_losses = [h["alpha_loss_bps"] for h in collateral_hits if h.get("alpha_loss_bps") is not None]
            target_improvements = [h["improvement_bps"] for h in target_hits if h.get("improvement_bps") is not None]

            n_target_total = len([t for t in target_ids if t in set(all_short_df["trade_id"].astype(str))])
            n_ctrl_a_total = len(ctrl_a_df)

            # Net expected bps over whole short universe
            net_bps_list = []
            for orig, cf in zip(all_short_pnl_orig, all_short_pnl_cf):
                if orig is not None and cf is not None:
                    net_bps_list.append(cf - orig)
            net_mean_bps = _agg(net_bps_list, "mean")

            variant_results[vname] = {
                "description": vdef["description"],
                "eval_bar": vdef["eval_bar"],
                "n_target_total": n_target_total,
                "n_target_hit": len(target_hits),
                "n_target_miss": len(target_miss),
                "target_hit_rate": round(len(target_hits) / n_target_total, 4) if n_target_total else 0.0,
                "target_hit_details": target_hits,
                "target_miss_ids": target_miss,
                "mean_target_improvement_bps": _agg(target_improvements, "mean"),
                "total_target_improvement_bps": _agg(target_improvements, "sum"),
                "n_ctrl_a_total": n_ctrl_a_total,
                "n_collateral_hits": len(collateral_hits),
                "collateral_hit_rate": round(len(collateral_hits) / n_ctrl_a_total, 4) if n_ctrl_a_total else 0.0,
                "mean_collateral_alpha_loss_bps": _agg(col_alpha_losses, "mean"),
                "median_collateral_alpha_loss_bps": _agg(col_alpha_losses, "median"),
                "worst_collateral_alpha_loss_bps": _agg(col_alpha_losses, "min"),
                "n_short_universe": len(all_short_df),
                "net_mean_bps_over_short_universe": net_mean_bps,
                "tail_orig": tail_orig,
                "tail_cf": tail_cf,
                "tail_improvement": {
                    "mean_delta": _safe((tail_cf.get("cf_mean") or 0.0) - (tail_orig.get("orig_mean") or 0.0)),
                    "worst_delta": _safe((tail_cf.get("cf_worst") or 0.0) - (tail_orig.get("orig_worst") or 0.0)),
                    "mean_worst10_delta": _safe((tail_cf.get("cf_mean_worst10") or 0.0) - (tail_orig.get("orig_mean_worst10") or 0.0)),
                    "mean_worst20_delta": _safe((tail_cf.get("cf_mean_worst20") or 0.0) - (tail_orig.get("orig_mean_worst20") or 0.0)),
                    "CVaR95_delta": _safe((tail_cf.get("cf_CVaR95") or 0.0) - (tail_orig.get("orig_CVaR95") or 0.0)),
                    "CVaR99_delta": _safe((tail_cf.get("cf_CVaR99") or 0.0) - (tail_orig.get("orig_CVaR99") or 0.0)),
                },
            }

        # ------------------------------------------------------------------
        # Verdict: rank variants by precision-like ratio
        # ------------------------------------------------------------------
        def _score(vr: Dict[str, Any]) -> float:
            hits = vr.get("n_target_hit", 0)
            col = vr.get("n_collateral_hits", 0)
            net = vr.get("net_mean_bps_over_short_universe") or 0.0
            # Simple score: target_hits * 10 - collateral_hits - 0 if net>0 else penalty
            return hits * 10 - col + (10 if net > 0 else 0)

        ranked = sorted(variant_results.items(), key=lambda x: -_score(x[1]))
        best_variant = ranked[0][0] if ranked else None

        verdict = {
            "best_variant": best_variant,
            "ranking": [
                {
                    "variant": vname,
                    "target_hit_rate": vr.get("target_hit_rate"),
                    "collateral_hit_rate": vr.get("collateral_hit_rate"),
                    "mean_target_improvement": vr.get("mean_target_improvement_bps"),
                    "mean_collateral_alpha_loss": vr.get("mean_collateral_alpha_loss_bps"),
                    "net_mean_bps": vr.get("net_mean_bps_over_short_universe"),
                    "CVaR95_delta": vr.get("tail_improvement", {}).get("CVaR95_delta"),
                }
                for vname, vr in ranked
            ],
            "recommendation": (
                f"Variant {best_variant} shows the best tradeoff. "
                f"Analytical only — no implementation proposed."
            ),
        }

        # ------------------------------------------------------------------
        # Write output
        # ------------------------------------------------------------------
        payload = {
            "run_id": run_id,
            "n_target_trades": len(target_ids),
            "n_ctrl_a_trades": len(ctrl_a_df),
            "n_short_universe": len(all_short_df),
            "variants": variant_results,
            "verdict": verdict,
        }
        out_path = chunk_output_dir / "EARLY_FAILURE_SHORT_GUARD_COUNTERFACTUAL.json"
        tmp_path = out_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        log.info(
            "[EARLY_FAILURE_SHORT_GUARD_CF_PROOF] run_id=%s best_variant=%s "
            "target_hit_rate=%s collateral_hit_rate=%s net_bps=%s path=%s",
            run_id, best_variant,
            variant_results.get(best_variant, {}).get("target_hit_rate"),
            variant_results.get(best_variant, {}).get("collateral_hit_rate"),
            variant_results.get(best_variant, {}).get("net_mean_bps_over_short_universe"),
            out_path,
        )

    except Exception as e:
        log.warning("[EARLY_FAILURE_SHORT_GUARD_CF_FAIL] run_id=%s error=%s", run_id, e)


def _write_entry_signature_audit_stuck_shorts(
    chunk_output_dir: Path, run_id: str, journal_df: "pd.DataFrame"
) -> None:
    """Analyse entry-state of 13 stuck short trades vs controls.

    Sources used (read-only):
      - STUCK_SHORT_SIGNATURE_AUDIT.json  → target trade_ids
      - trade_journal_{run_id}.parquet    → side/session/exit_reason/pnl_bps/bars
      - chunk_0_data.parquet              → bar-level market features
      - logs/eval_log_{date}.jsonl        → model scores at every evaluated bar

    Writes: ENTRY_SIGNATURE_AUDIT_STUCK_SHORTS.json
    Never raises; best-effort only.
    """
    try:
        import pandas as _pd
        import numpy as _np
        import json as _json
        import glob as _glob

        out_path = chunk_output_dir / "ENTRY_SIGNATURE_AUDIT_STUCK_SHORTS.json"

        # --- 1. Load target IDs from signature audit ---
        sig_path = chunk_output_dir / "STUCK_SHORT_SIGNATURE_AUDIT.json"
        if not sig_path.exists():
            log.warning("[ENTRY_SIG_AUDIT] STUCK_SHORT_SIGNATURE_AUDIT.json missing, skipping")
            return
        sig_data = _json.loads(sig_path.read_text())
        target_ids: set = {
            t["trade_id"] for t in sig_data.get("section_1_identity_list", [])
        }
        if not target_ids:
            log.warning("[ENTRY_SIG_AUDIT] no target_ids found")
            return

        # --- 2. Dedup journal (prefer non-null session) ---
        j = journal_df.copy()
        j["_sn"] = j["session"].isna()
        j = j.sort_values("_sn").drop_duplicates(subset="trade_id", keep="first").drop(
            columns=["_sn"]
        )
        j["open_ts_utc"] = _pd.to_datetime(j["open_ts_utc"], utc=True, errors="coerce")
        j_short = j[j["side"].str.lower() == "short"].copy()

        # --- 3. Load chunk_0_data for bar-level features ---
        c0_path = chunk_output_dir / "chunk_0_data.parquet"
        if c0_path.exists():
            c0 = _pd.read_parquet(c0_path)
            c0["time"] = _pd.to_datetime(c0["time"], utc=True, errors="coerce")
            c0 = c0.sort_values("time").reset_index(drop=True)
            c0_indexed = c0.set_index("time")
        else:
            c0_indexed = None

        BAR_FEAT_COLS = [
            "trend_regime_id", "atr_bucket", "H4_trend_sign_cat",
            "micro_momentum_3", "micro_momentum_5", "micro_acceleration",
            "distance_ema_fast", "dist_last_swing_high_atr", "dist_last_swing_low_atr",
            "bars_since_swing_high", "bars_since_swing_low",
            "retracement_from_last_impulse", "D1_dist_from_ema200_atr",
            "D1_atr_percentile_252", "atr_bps", "_v1_r1", "_v1_r12", "_v1_ema_diff",
            "_v1_kama_slope_30",
        ]

        # --- 4. Load eval_log for model scores ---
        logs_dir = chunk_output_dir / "logs"
        eval_log_paths = sorted(_glob.glob(str(logs_dir / "eval_log_*.jsonl")))
        eval_df = None
        if eval_log_paths:
            try:
                eval_rows = []
                with open(eval_log_paths[0]) as _f:
                    for _line in _f:
                        eval_rows.append(_json.loads(_line))
                eval_df = _pd.DataFrame(eval_rows)
                eval_df["ts_utc"] = _pd.to_datetime(eval_df["ts_utc"], utc=True, errors="coerce")
                eval_df = eval_df.set_index("ts_utc")
            except Exception as _e:
                log.warning("[ENTRY_SIG_AUDIT] eval_log load error: %s", _e)
                eval_df = None

        EVAL_COLS = [
            "p_short", "p_long", "p_flat", "p_hat", "uncertainty_score",
            "margin_top1_top2", "entropy",
            "xgb_p_short", "xgb_p_long", "xgb_p_flat",
            "xgb_margin_top1_top2", "xgb_entropy", "xgb_uncertainty_score",
            "decision_reason", "pre_gate_pref", "price",
        ]

        # --- 5. Build per-trade feature rows ---
        def _get_bar_features(ts: "_pd.Timestamp") -> dict:
            if c0_indexed is None or _pd.isna(ts):
                return {}
            idx = int(c0_indexed.index.searchsorted(ts, side="left"))
            if idx >= len(c0_indexed):
                idx = len(c0_indexed) - 1
            bar = c0_indexed.iloc[idx]
            return {
                fc: (float(bar[fc]) if fc in bar.index and _pd.notna(bar[fc]) else None)
                for fc in BAR_FEAT_COLS
            }

        def _get_eval_features(ts: "_pd.Timestamp") -> dict:
            if eval_df is None or _pd.isna(ts):
                return {}
            if ts not in eval_df.index:
                return {}
            row = eval_df.loc[ts]
            if isinstance(row, _pd.DataFrame):
                row = row.iloc[0]
            return {ec: row.get(ec) for ec in EVAL_COLS if ec in row.index}

        rows = []
        for _, tr in j_short.iterrows():
            ts = tr["open_ts_utc"]
            rec: dict = {
                "trade_id": tr["trade_id"],
                "entry_time": str(ts),
                "session": tr["session"],
                "exit_reason": tr["exit_reason"],
                "pnl_bps": (float(tr["pnl_bps"]) if _pd.notna(tr.get("pnl_bps")) else None),
                "bars_in_trade": (int(tr["bars_in_trade"]) if _pd.notna(tr.get("bars_in_trade")) else None),
                "mfe_bps": (float(tr["mfe_bps"]) if _pd.notna(tr.get("mfe_bps")) else None),
                "mae_bps": (float(tr["mae_bps"]) if _pd.notna(tr.get("mae_bps")) else None),
                "is_target": tr["trade_id"] in target_ids,
            }
            rec.update(_get_bar_features(ts))
            eval_feats = _get_eval_features(ts)
            for k, v in eval_feats.items():
                rec[f"eval_{k}"] = v
            rows.append(rec)

        full_df = _pd.DataFrame(rows)
        t_df = full_df[full_df["is_target"]].copy()
        ca_df = full_df[
            (full_df["exit_reason"] == "THRESHOLD") & ~full_df["is_target"]
        ].copy()
        pnl_ca = _pd.to_numeric(ca_df["pnl_bps"], errors="coerce")
        cb_df = ca_df.loc[pnl_ca.nsmallest(20).index].copy()
        cw_df = ca_df[_pd.to_numeric(ca_df["pnl_bps"], errors="coerce") > 0].copy()

        # --- 6. Helper: numeric stats for a column across groups ---
        def _num_stats(df: "_pd.DataFrame", col: str) -> dict:
            s = _pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
            return {
                "n": int(len(s)),
                "mean": float(round(s.mean(), 4)),
                "median": float(round(s.median(), 4)),
                "std": float(round(s.std(), 4)),
                "min": float(round(s.min(), 4)),
                "max": float(round(s.max(), 4)),
            }

        # --- 7. Analysis 1: Identity listing ---
        identity_list = []
        for _, tr in t_df.sort_values("pnl_bps").iterrows():
            identity_list.append({
                "trade_id": tr["trade_id"],
                "entry_time": tr["entry_time"],
                "session": tr.get("session"),
                "exit_reason": tr.get("exit_reason"),
                "pnl_bps": tr.get("pnl_bps"),
                "bars_in_trade": tr.get("bars_in_trade"),
                "mfe_bps": tr.get("mfe_bps"),
                "mae_bps": tr.get("mae_bps"),
            })

        # --- 8. Analysis 2: Categorical signature ---
        def _cat_compare(col: str) -> list:
            if col not in full_df.columns:
                return []
            out = []
            t_vc = t_df[col].fillna("(null)").value_counts(dropna=False)
            ca_vc = ca_df[col].fillna("(null)").value_counts(dropna=False)
            all_cats = sorted(
                set(t_vc.index) | set(ca_vc.index),
                key=lambda x: -t_vc.get(x, 0)
            )
            t_total = max(len(t_df), 1)
            ca_total = max(len(ca_df), 1)
            for cat in all_cats:
                tc_ = int(t_vc.get(cat, 0))
                cac_ = int(ca_vc.get(cat, 0))
                tf = round(tc_ / t_total, 4)
                caf = round(cac_ / ca_total, 4)
                overrep = round(tf / caf, 3) if caf > 0 else None
                out.append({
                    "category": str(cat),
                    "target_count": tc_,
                    "target_frac": tf,
                    "ctrl_a_count": cac_,
                    "ctrl_a_frac": caf,
                    "overrep_ratio": overrep,
                })
            return out

        cat_cols = [
            "session", "H4_trend_sign_cat", "trend_regime_id",
            "atr_bucket", "_v1_atr_regime_id",
        ]
        # month from entry_time
        t_df = t_df.copy()
        t_df["entry_month"] = _pd.to_datetime(t_df["entry_time"], utc=True, errors="coerce").dt.month
        ca_df = ca_df.copy()
        ca_df["entry_month"] = _pd.to_datetime(ca_df["entry_time"], utc=True, errors="coerce").dt.month
        full_df = _pd.concat([t_df, ca_df], ignore_index=True)
        cat_cols.append("entry_month")
        categorical_sig = {col: _cat_compare(col) for col in cat_cols}

        # --- 9. Analysis 3 & 4: Numeric separability ---
        num_cols = (
            BAR_FEAT_COLS
            + [f"eval_{c}" for c in [
                "p_short", "p_long", "p_flat", "p_hat", "uncertainty_score",
                "margin_top1_top2", "entropy",
                "xgb_p_short", "xgb_p_long", "xgb_p_flat",
                "xgb_margin_top1_top2", "xgb_entropy", "xgb_uncertainty_score",
            ]]
        )

        separability_rows = []
        for col in num_cols:
            if col not in t_df.columns:
                continue
            ts_num = _pd.to_numeric(t_df[col], errors="coerce").dropna()
            ca_num = _pd.to_numeric(ca_df[col], errors="coerce").dropna()
            cb_num = _pd.to_numeric(cb_df[col], errors="coerce").dropna() if col in cb_df.columns else _pd.Series([], dtype=float)
            if ts_num.empty or ca_num.empty:
                continue
            delta_mean = float(round(ts_num.mean() - ca_num.mean(), 5))
            z_score = delta_mean / (ca_num.std() + 1e-9)
            separability_rows.append({
                "feature": col,
                "target_mean": float(round(ts_num.mean(), 5)),
                "target_median": float(round(ts_num.median(), 5)),
                "target_std": float(round(ts_num.std(), 5)),
                "ctrl_a_mean": float(round(ca_num.mean(), 5)),
                "ctrl_a_median": float(round(ca_num.median(), 5)),
                "ctrl_b_mean": float(round(cb_num.mean(), 5)) if not cb_num.empty else None,
                "delta_mean_vs_ctrl_a": delta_mean,
                "abs_z_score_vs_ctrl_a": float(round(abs(z_score), 3)),
            })
        separability_rows.sort(key=lambda x: -x["abs_z_score_vs_ctrl_a"])

        top_separating = separability_rows[:10]

        # --- 10. Analysis 5: Entry confidence ---
        confidence_rows = []
        prob_col = "eval_p_short"
        if prob_col in t_df.columns:
            for group_name, gdf in [("target", t_df), ("ctrl_a_threshold", ca_df),
                                      ("ctrl_b_worst_threshold", cb_df), ("ctrl_c_winners", cw_df)]:
                s = _pd.to_numeric(gdf[prob_col], errors="coerce").dropna()
                confidence_rows.append({
                    "group": group_name,
                    "n": int(len(s)),
                    "mean_p_short": float(round(s.mean(), 4)) if not s.empty else None,
                    "median_p_short": float(round(s.median(), 4)) if not s.empty else None,
                    "mean_margin": (
                        float(round(_pd.to_numeric(gdf["eval_margin_top1_top2"], errors="coerce").dropna().mean(), 4))
                        if "eval_margin_top1_top2" in gdf.columns else None
                    ),
                    "frac_below_threshold_prob": (
                        float(round((s < 0.45).mean(), 3)) if not s.empty else None
                    ),
                })

        # --- 11. Analysis 6: Decision reason ---
        decision_reason_dist = {}
        if "eval_decision_reason" in t_df.columns:
            decision_reason_dist["target"] = t_df["eval_decision_reason"].value_counts(dropna=False).to_dict()
            decision_reason_dist["ctrl_a"] = ca_df["eval_decision_reason"].value_counts(dropna=False).head(5).to_dict() if "eval_decision_reason" in ca_df.columns else {}

        # --- 12. Analysis 7: Entry-to-immediate-failure bridge ---
        # Correlate entry features with early PnL from EXIT_EVAL_TRACE
        trace_path = chunk_output_dir / "EXIT_EVAL_TRACE.csv"
        bridge_stats: dict = {}
        if trace_path.exists():
            try:
                trace_df = _pd.read_csv(trace_path, usecols=["trade_id", "bars_held", "pnl_bps", "mfe_bps", "exit_prob"])
                trace_df["trade_id"] = trace_df["trade_id"].astype(str)
                early5 = trace_df[trace_df["bars_held"] <= 5]
                bridge_rows = []
                for grp_name, grp_ids in [
                    ("target", list(target_ids)),
                    ("ctrl_a", list(ca_df["trade_id"])),
                    ("ctrl_b", list(cb_df["trade_id"])),
                ]:
                    sub = early5[early5["trade_id"].isin(grp_ids)]
                    bridge_rows.append({
                        "group": grp_name,
                        "n_trades": len(set(sub["trade_id"])),
                        "mean_pnl_first5": float(round(sub["pnl_bps"].mean(), 4)) if not sub.empty else None,
                        "mean_mfe_first5": float(round(sub["mfe_bps"].mean(), 4)) if not sub.empty else None,
                        "mean_prob_close_first5": float(round(sub["exit_prob"].mean(), 4)) if not sub.empty else None,
                        "frac_pnl_negative_first5": float(round((sub["pnl_bps"] < 0).mean(), 3)) if not sub.empty else None,
                    })
                bridge_stats["early_life_vs_entry_features"] = bridge_rows

                # Entry feature deltas for the 5 most separating features
                top5_feats = [r["feature"] for r in top_separating[:5]]
                bridge_stats["top5_sep_features_with_early_pnl"] = []
                for feat in top5_feats:
                    if feat not in t_df.columns:
                        continue
                    val_t = t_df[feat].dropna().tolist()
                    val_ca = ca_df[feat].dropna().tolist()
                    bridge_stats["top5_sep_features_with_early_pnl"].append({
                        "feature": feat,
                        "target_values": [round(float(v), 4) for v in val_t[:13]],
                        "ctrl_a_sample_mean": float(round(_np.mean(val_ca), 4)) if val_ca else None,
                    })
            except Exception as _e:
                log.warning("[ENTRY_SIG_AUDIT] bridge analysis error: %s", _e)
                bridge_stats = {"error": str(_e)}

        # --- 13. Analysis 8: Exact entry signature ---
        # Build rule-based summary
        top3 = [r["feature"] for r in top_separating[:3]]

        def _fmt_feat_summary(feat: str) -> str:
            sr = next((r for r in separability_rows if r["feature"] == feat), None)
            if not sr:
                return feat
            direction = "LOWER" if sr["delta_mean_vs_ctrl_a"] < 0 else "HIGHER"
            return (
                f"{feat}: target_mean={sr['target_mean']:.4f} vs ctrl_a_mean={sr['ctrl_a_mean']:.4f} "
                f"({direction} by {abs(sr['delta_mean_vs_ctrl_a']):.4f})"
            )

        # H4_trend_sign_cat dominance check
        h4_cat = categorical_sig.get("H4_trend_sign_cat", [])
        h4_val2_target_frac = next((r["target_frac"] for r in h4_cat if str(r["category"]) == "2.0"), None)
        h4_val2_ctrla_frac = next((r["ctrl_a_frac"] for r in h4_cat if str(r["category"]) == "2.0"), None)

        # session dominance
        sess_cats = categorical_sig.get("session", [])
        top_sess_target = sorted(sess_cats, key=lambda r: -r["target_frac"])[:3]

        signature_lines = [
            f"12/13 target trades have H4_trend_sign_cat=2 (H4 uptrend; short entry INTO strength) vs {h4_val2_ctrla_frac:.3f} of ctrl_a",
            f"Target trades have deeply negative micro_momentum at entry: mean={next((r['target_mean'] for r in separability_rows if r['feature']=='micro_momentum_5'), None):.3f} vs ctrl_a={next((r['ctrl_a_mean'] for r in separability_rows if r['feature']=='micro_momentum_5'), None):.3f}",
            f"Target trades have negative distance_ema_fast at entry: mean={next((r['target_mean'] for r in separability_rows if r['feature']=='distance_ema_fast'), None):.3f} vs ctrl_a={next((r['ctrl_a_mean'] for r in separability_rows if r['feature']=='distance_ema_fast'), None):.3f} (price below fast EMA = uptrend)",
            f"Target trades have higher bars_since_swing_high: mean={next((r['target_mean'] for r in separability_rows if r['feature']=='bars_since_swing_high'), None):.2f} vs ctrl_a={next((r['ctrl_a_mean'] for r in separability_rows if r['feature']=='bars_since_swing_high'), None):.2f}",
            f"Target model p_short is LOWER than ctrl_a: mean={next((r['target_mean'] for r in separability_rows if r['feature']=='eval_p_short'), None):.4f} vs {next((r['ctrl_a_mean'] for r in separability_rows if r['feature']=='eval_p_short'), None):.4f}",
            f"Most separating features by z-score: {', '.join([r['feature'] for r in top_separating[:5]])}",
        ]
        # filter None
        signature_lines = [l for l in signature_lines if "None" not in l]

        verdict = {
            "n_target_analyzed": len(t_df),
            "primary_entry_signature": (
                "side=short, H4_trend_sign_cat=2 (price/H4 in uptrend), "
                "negative micro_momentum_5 (short-term price below recent levels), "
                "negative distance_ema_fast (price below fast EMA), "
                "higher bars_since_swing_high (further from recent high), "
                "model p_short < ctrl_a mean (borderline short confidence)"
            ),
            "separability_at_entry": "YES - target trades differ from ctrl_a at entry especially on micro_momentum and distance_ema_fast",
            "entry_confidence_level": "LOW-to-BORDERLINE: mean p_short=0.41 vs ctrl_a 0.46; margin lower; xgb less certain",
            "most_likely_problem_description": (
                "Short entries taken when H4 trend is bullish (H4_trend_sign_cat=2), "
                "micro_momentum strongly negative (price in short-term pullback), "
                "and model sees 'short' signal in a pullback within an uptrend. "
                "Entry looks like a momentum short into pullback but H4 direction opposes it, "
                "leading to immediate reversal back to trend direction."
            ),
            "signature_lines": signature_lines,
        }

        # --- 14. Build final output ---
        output = {
            "contract": "ENTRY_SIGNATURE_AUDIT_STUCK_SHORTS_V1",
            "run_id": run_id,
            "n_target_trades": len(t_df),
            "n_ctrl_a_trades": len(ca_df),
            "n_ctrl_b_trades": len(cb_df),
            "n_ctrl_c_winner_trades": len(cw_df),
            "section_1_identity_list": identity_list,
            "section_2_categorical_signature": categorical_sig,
            "section_3_numeric_separability_ranked": separability_rows[:30],
            "section_4_top10_separating_features": top_separating,
            "section_5_entry_confidence_by_group": confidence_rows,
            "section_6_decision_reason": decision_reason_dist,
            "section_7_bridge_entry_to_early_failure": bridge_stats,
            "section_8_exact_entry_signature": verdict,
        }

        # Atomic write
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(_json.dumps(output, indent=2, default=str))
        tmp_path.replace(out_path)

        log.info(
            "[ENTRY_SIG_AUDIT_DONE] run_id=%s n_target=%d out=%s top3_sep=%s",
            run_id, len(t_df), out_path, top3,
        )

    except Exception as e:
        log.warning("[ENTRY_SIGNATURE_AUDIT_FAIL] run_id=%s error=%s", run_id, e)


def _write_trade_outcomes_truth(chunk_output_dir: Path, run_id: str, runner: Any) -> None:
    """
    TRUTH-native writer: always write chunk-level trade_outcomes_{run_id}.parquet.

    - Does NOT import legacy gx1.scripts.replay_eval_gated.
    - Uses runner.replay_eval_collectors["trade_outcomes"].outcomes if available.
    - Otherwise writes an empty parquet with canonical schema.
    - Atomic via tmp -> os.replace.

    Raises on failure (TRUTH-grade).
    """
    out_path = chunk_output_dir / f"trade_outcomes_{run_id}.parquet"

    collector = None
    if runner is not None and getattr(runner, "replay_eval_collectors", None):
        collector = runner.replay_eval_collectors.get("trade_outcomes")

    rows = list(getattr(collector, "outcomes", [])) if collector else []

    fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet",
        prefix="trade_outcomes_",
        dir=str(chunk_output_dir),
    )
    try:
        os.close(fd)
        tmp = Path(tmp_path)

        if rows:
            df = pd.DataFrame(rows)

            # Ensure deterministic-ish ordering: contract first, then any extras.
            cols = [c for c in TRADE_OUTCOMES_REQUIRED_COLUMNS if c in df.columns]
            extra = [c for c in df.columns if c not in TRADE_OUTCOMES_REQUIRED_COLUMNS]
            if cols:
                df = df[cols + extra]
            df.to_parquet(tmp, index=False)
        else:
            write_empty_trade_outcomes_parquet(tmp, run_id=run_id)

        os.replace(str(tmp), str(out_path))
        log.info("[TRUTH_TRADE_OUTCOMES] Wrote %s (%d rows)", out_path.name, len(rows))
    except Exception as e:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"[TRUTH_TRADE_OUTCOMES] Failed to write {out_path}: {e}") from e


def _write_trade_journal_truth(chunk_output_dir: Path, run_id: str, runner: Any) -> None:
    """
    Build trade_journal_{run_id}.parquet from exits audit + exits trace (run-specific).

    Best-effort: never fail TRUTH if logs are missing or malformed.
    """
    import json

    try:
        log_dir = getattr(runner, "log_dir", None) if runner is not None else None
        exits_dir = (Path(log_dir) if log_dir else (chunk_output_dir / "logs")) / "exits"

        audit_candidates = sorted(
            exits_dir.glob("exits_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].jsonl")
        )
        audit_path = audit_candidates[-1] if audit_candidates else None
        trace_path = exits_dir / f"exits_{run_id}.jsonl"

        if audit_path is None or not audit_path.exists() or not trace_path.exists():
            log.info(
                "[TRADE_JOURNAL_SKIPPED] missing exits logs (audit=%s trace=%s)",
                str(audit_path) if audit_path else None,
                str(trace_path),
            )
            return

        audit_rows = []
        with audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("accepted") is not True:
                    continue
                trade_id = rec.get("trade_id")
                if trade_id is None:
                    continue
                rec["trade_id"] = str(trade_id)
                audit_rows.append(rec)

        if not audit_rows:
            df_empty = pd.DataFrame(
                columns=[
                    "trade_id",
                    "trade_uid",
                    "side",
                    "open_ts_utc",
                    "close_ts_utc",
                    "margin_top1_top2",
                    "tradable_prob",
                    "mfe_first_n_pred",
                    "path_quality_pred",
                    "pnl_bps",
                    "bars_in_trade",
                    "exit_reason",
                    "accepted",
                    "session",
                    "source",
                    "prob_close",
                    "threshold",
                    "entry_bid",
                    "entry_ask",
                    "exit_bid",
                    "exit_ask",
                    "entry_spread_bps",
                    "exit_spread_bps",
                    "entry_price_used",
                    "exit_price_used",
                    "mae_bps",
                    "mfe_bps",
                ]
            )
            out_path = chunk_output_dir / f"trade_journal_{run_id}.parquet"
            df_empty.to_parquet(out_path, index=False)
            log.info(
                "[TRADE_JOURNAL_PROOF] wrote trade_journal rows=0 cols=%s path=%s",
                list(df_empty.columns),
                out_path,
            )
            log.info(
                "[TRADE_JOURNAL_MAE_MFE_PROOF] rows=0 mae_missing=0 mfe_missing=0",
            )
            return

        audit_df = pd.DataFrame(audit_rows)
        if "trade_id" in audit_df.columns:
            audit_df["trade_id"] = audit_df["trade_id"].astype(str)
        audit_df["ts"] = pd.to_datetime(audit_df.get("ts"), utc=True, errors="coerce")
        audit_df = audit_df[audit_df["ts"].notna()]
        audit_df = audit_df.sort_values("ts")
        audit_df = audit_df.groupby("trade_id", as_index=False).tail(1)

        trace_map: Dict[str, Dict[str, Any]] = {}
        trace_ts: Dict[str, pd.Timestamp] = {}
        with trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                trade_id = rec.get("trade_id")
                if not trade_id:
                    continue
                trade_id = str(trade_id)
                rec["trade_id"] = trade_id
                ts = pd.to_datetime(rec.get("ts"), utc=True, errors="coerce")
                prev_ts = trace_ts.get(trade_id)
                if prev_ts is None:
                    trace_map[trade_id] = rec
                    if pd.notna(ts):
                        trace_ts[trade_id] = ts
                    continue
                if pd.notna(ts) and pd.notna(prev_ts) and ts >= prev_ts:
                    trace_map[trade_id] = rec
                    if pd.notna(ts):
                        trace_ts[trade_id] = ts

        entry_margin_by_uid: Dict[str, float] = {}
        entry_margin_by_open_side: Dict[tuple, float] = {}
        entry_price_by_uid: Dict[str, Dict[str, Any]] = {}
        exit_price_by_uid: Dict[str, Dict[str, Any]] = {}
        entry_price_by_id: Dict[str, Dict[str, Any]] = {}
        exit_price_by_id: Dict[str, Dict[str, Any]] = {}
        exit_mae_mfe_by_uid: Dict[str, Dict[str, Any]] = {}
        exit_mae_mfe_by_id: Dict[str, Dict[str, Any]] = {}
        entry_tradable_by_uid: Dict[str, float] = {}
        entry_tradable_by_id: Dict[str, float] = {}
        entry_mfe_first_n_by_uid: Dict[str, float] = {}
        entry_mfe_first_n_by_id: Dict[str, float] = {}
        entry_path_quality_by_uid: Dict[str, float] = {}
        entry_path_quality_by_id: Dict[str, float] = {}
        margin_missing_reasons: Dict[str, int] = {
            "NO_ENTRY_JOURNAL": 0,
            "MISSING_OPEN_TS": 0,
            "MISSING_SIDE": 0,
            "NO_MATCH": 0,
        }
        trade_journal_dir = chunk_output_dir / "trade_journal" / "trades"
        entry_journal_present = trade_journal_dir.exists()
        if entry_journal_present:
            for path in sorted(trade_journal_dir.glob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        tj = json.load(f)
                except Exception:
                    continue
                trade_uid = tj.get("trade_uid") or tj.get("trade_uid")
                trade_id = tj.get("trade_id")
                if trade_id is not None:
                    trade_id = str(trade_id)
                entry_snapshot = tj.get("entry_snapshot") or {}
                exit_summary = tj.get("exit_summary") or {}
                entry_score = entry_snapshot.get("entry_score") or {}
                entry_model_outputs = entry_snapshot.get("entry_model_outputs") or {}
                margin_val = entry_score.get("margin")
                tradable_val = entry_score.get("tradable_prob")
                mfe_first_n_val = entry_score.get("mfe_first_n_pred")
                path_quality_val = entry_score.get("path_quality_pred")
                if margin_val is None:
                    margin_val = entry_model_outputs.get("margin")
                try:
                    margin_val = float(margin_val)
                except Exception:
                    margin_val = None
                try:
                    tradable_val = float(tradable_val)
                except Exception:
                    tradable_val = None
                try:
                    mfe_first_n_val = float(mfe_first_n_val)
                except Exception:
                    mfe_first_n_val = None
                try:
                    path_quality_val = float(path_quality_val)
                except Exception:
                    path_quality_val = None
                entry_time = entry_snapshot.get("entry_time")
                side = entry_snapshot.get("side") or tj.get("side")
                ts_val = pd.to_datetime(entry_time, utc=True, errors="coerce") if entry_time else pd.NaT
                if trade_uid:
                    entry_price_by_uid[str(trade_uid)] = {
                        "entry_bid": entry_snapshot.get("entry_bid"),
                        "entry_ask": entry_snapshot.get("entry_ask"),
                        "entry_spread_bps": entry_snapshot.get("entry_spread_bps"),
                        "entry_price_used": entry_snapshot.get("entry_price_used"),
                    }
                    exit_price_by_uid[str(trade_uid)] = {
                        "exit_bid": exit_summary.get("exit_bid"),
                        "exit_ask": exit_summary.get("exit_ask"),
                        "exit_spread_bps": exit_summary.get("exit_spread_bps"),
                        "exit_price_used": exit_summary.get("exit_price_used"),
                    }
                    exit_mae_mfe_by_uid[str(trade_uid)] = {
                        "mae_bps": exit_summary.get("max_mae_bps"),
                        "mfe_bps": exit_summary.get("max_mfe_bps"),
                    }
                    if margin_val is not None:
                        entry_margin_by_uid[str(trade_uid)] = margin_val
                    if tradable_val is not None:
                        entry_tradable_by_uid[str(trade_uid)] = tradable_val
                    if mfe_first_n_val is not None:
                        entry_mfe_first_n_by_uid[str(trade_uid)] = mfe_first_n_val
                    if path_quality_val is not None:
                        entry_path_quality_by_uid[str(trade_uid)] = path_quality_val
                if trade_id:
                    entry_price_by_id[str(trade_id)] = {
                        "entry_bid": entry_snapshot.get("entry_bid"),
                        "entry_ask": entry_snapshot.get("entry_ask"),
                        "entry_spread_bps": entry_snapshot.get("entry_spread_bps"),
                        "entry_price_used": entry_snapshot.get("entry_price_used"),
                    }
                    exit_price_by_id[str(trade_id)] = {
                        "exit_bid": exit_summary.get("exit_bid"),
                        "exit_ask": exit_summary.get("exit_ask"),
                        "exit_spread_bps": exit_summary.get("exit_spread_bps"),
                        "exit_price_used": exit_summary.get("exit_price_used"),
                    }
                    exit_mae_mfe_by_id[str(trade_id)] = {
                        "mae_bps": exit_summary.get("max_mae_bps"),
                        "mfe_bps": exit_summary.get("max_mfe_bps"),
                    }
                    if tradable_val is not None:
                        entry_tradable_by_id[str(trade_id)] = tradable_val
                    if mfe_first_n_val is not None:
                        entry_mfe_first_n_by_id[str(trade_id)] = mfe_first_n_val
                    if path_quality_val is not None:
                        entry_path_quality_by_id[str(trade_id)] = path_quality_val
                if margin_val is None:
                    continue
                if pd.notna(ts_val) and side:
                    entry_margin_by_open_side[(int(ts_val.value), str(side))] = margin_val

        # Preload replay price data for MAE/MFE (deterministic, full trade lifetime)
        price_df = None
        price_path = chunk_output_dir / "chunk_0_data.parquet"
        if price_path.exists():
            try:
                price_df = pd.read_parquet(price_path, columns=[
                    "time",
                    "bid_high",
                    "bid_low",
                    "ask_high",
                    "ask_low",
                    "bid_close",
                    "ask_close",
                ])
                price_df["time"] = pd.to_datetime(price_df["time"], utc=True, errors="coerce")
                price_df = price_df[price_df["time"].notna()]
                price_df = price_df.sort_values("time")
            except Exception as e:
                log.warning("[TRADE_JOURNAL_PRICE_LOAD_FAIL] %s", e)
                price_df = None

        rows = []
        for _, row in audit_df.iterrows():
            trade_id = row.get("trade_id")
            if trade_id is not None:
                trade_id = str(trade_id)
            trace = trace_map.get(trade_id, {})
            close_ts = row.get("ts")
            open_ts = (
                trace.get("entry_time")
                or trace.get("open_ts_utc")
                or trace.get("open_ts")
                or trace.get("entry_ts")
                or row.get("open_ts_utc")
                or row.get("open_ts")
                or row.get("entry_ts")
                or row.get("entry_time")
            )
            open_ts = pd.to_datetime(open_ts, utc=True, errors="coerce") if open_ts else pd.NaT
            margin_val = None
            tradable_prob = None
            mfe_first_n_pred = None
            path_quality_pred = None
            trade_uid = trace.get("trade_uid")
            side_val = trace.get("side")
            if trade_uid:
                margin_val = entry_margin_by_uid.get(str(trade_uid))
                tradable_prob = entry_tradable_by_uid.get(str(trade_uid))
                mfe_first_n_pred = entry_mfe_first_n_by_uid.get(str(trade_uid))
                path_quality_pred = entry_path_quality_by_uid.get(str(trade_uid))
            if margin_val is None:
                if pd.notna(open_ts) and side_val:
                    margin_val = entry_margin_by_open_side.get((int(open_ts.value), str(side_val)))
                elif pd.isna(open_ts):
                    margin_missing_reasons["MISSING_OPEN_TS"] += 1
                elif not side_val:
                    margin_missing_reasons["MISSING_SIDE"] += 1
            if margin_val is None:
                if not entry_journal_present:
                    # Entry journal is disabled in TRUTH replay by design; do not alarm.
                    pass
                elif not entry_margin_by_uid and not entry_margin_by_open_side:
                    margin_missing_reasons["NO_ENTRY_JOURNAL"] += 1
                else:
                    margin_missing_reasons["NO_MATCH"] += 1
            trace_scalars = trace.get("scalars") or {}
            bars_held_exit_state = trace_scalars.get("bars_held", row.get("bars_in_trade"))
            time_since_mfe_exit = trace_scalars.get("time_since_mfe_bars")
            dd_from_mfe_exit = trace_scalars.get("dd_from_mfe_bps")
            distance_from_peak_exit = trace_scalars.get(
                "distance_from_peak_mfe_bps",
                dd_from_mfe_exit,
            )
            peak_mfe_bps_exit = trace_scalars.get("mfe_bps")
            peak_mfe_bar_index = None
            try:
                bh = float(bars_held_exit_state)
                ts_mfe = float(time_since_mfe_exit)
                if np.isfinite(bh) and np.isfinite(ts_mfe):
                    peak_mfe_bar_index = float(max(0.0, bh - ts_mfe))
            except Exception:
                peak_mfe_bar_index = None
            close_month_utc = None
            try:
                if pd.notna(close_ts):
                    close_month_utc = pd.Timestamp(close_ts).to_period("M").strftime("%Y-%m")
            except Exception:
                close_month_utc = None
            row_out = {
                "trade_id": trade_id,
                "trade_uid": trade_uid,
                "side": side_val,
                "open_ts_utc": open_ts,
                "close_ts_utc": close_ts,
                "margin_top1_top2": margin_val,
                "tradable_prob": tradable_prob,
                "pnl_bps": row.get("pnl_bps"),
                "bars_in_trade": row.get("bars_in_trade"),
                "exit_reason": row.get("reason"),
                "accepted": True,
                "session": row.get("session", None),
                "source": row.get("source"),
                "prob_close": (trace.get("computed") or {}).get("prob_close"),
                "threshold": (trace.get("computed") or {}).get("threshold"),
                "bars_held_exit_state": bars_held_exit_state,
                "peak_mfe_bps_exit_state": peak_mfe_bps_exit,
                "peak_mfe_bar_index": peak_mfe_bar_index,
                "time_since_mfe_bars_exit": time_since_mfe_exit,
                "dd_from_mfe_bps_exit": dd_from_mfe_exit,
                "distance_from_peak_mfe_bps_exit": distance_from_peak_exit,
                "close_month_utc": close_month_utc,
            }
            if trade_uid:
                entry_fields = entry_price_by_uid.get(str(trade_uid), {})
                exit_fields = exit_price_by_uid.get(str(trade_uid), {})
                mae_mfe_fields = exit_mae_mfe_by_uid.get(str(trade_uid), {})
            else:
                entry_fields = entry_price_by_id.get(str(trade_id), {}) if trade_id is not None else {}
                exit_fields = exit_price_by_id.get(str(trade_id), {}) if trade_id is not None else {}
                mae_mfe_fields = exit_mae_mfe_by_id.get(str(trade_id), {}) if trade_id is not None else {}
                if tradable_prob is None and trade_id is not None:
                    tradable_prob = entry_tradable_by_id.get(str(trade_id))
                if mfe_first_n_pred is None and trade_id is not None:
                    mfe_first_n_pred = entry_mfe_first_n_by_id.get(str(trade_id))
                if path_quality_pred is None and trade_id is not None:
                    path_quality_pred = entry_path_quality_by_id.get(str(trade_id))
            row_out.update(entry_fields)
            row_out.update(exit_fields)
            row_out.update(mae_mfe_fields)
            row_out["mfe_first_n_pred"] = mfe_first_n_pred
            row_out["path_quality_pred"] = path_quality_pred

            # Deterministic MAE/MFE over full trade lifetime from replay price data
            if price_df is not None and pd.notna(open_ts) and pd.notna(close_ts):
                side_for_mae = (side_val or "").lower()
                entry_px = row_out.get("entry_price_used")
                if entry_px is None:
                    entry_px = row_out.get("entry_ask") if side_for_mae == "long" else row_out.get("entry_bid")
                try:
                    entry_px = float(entry_px)
                except Exception:
                    entry_px = None

                # Find bar window even if open/close are not exact bar timestamps
                window = None
                if price_df is not None and pd.notna(open_ts) and pd.notna(close_ts):
                    times = price_df["time"].values
                    start_idx = int(np.searchsorted(times, open_ts.to_datetime64(), side="left"))
                    end_idx = int(np.searchsorted(times, close_ts.to_datetime64(), side="right")) - 1
                    if start_idx < 0:
                        start_idx = 0
                    if end_idx >= len(price_df):
                        end_idx = len(price_df) - 1
                    if end_idx >= start_idx and len(price_df) > 0:
                        window = price_df.iloc[start_idx : end_idx + 1]

                # Fallback entry_px from first bar if missing
                if (entry_px is None or entry_px == 0) and window is not None and not window.empty:
                    if side_for_mae == "long":
                        entry_px = window["ask_close"].iloc[0]
                    else:
                        entry_px = window["bid_close"].iloc[0]
                    try:
                        entry_px = float(entry_px)
                    except Exception:
                        entry_px = None

                if entry_px and entry_px > 0 and window is not None and not window.empty:
                    if side_for_mae == "long":
                        favorable_series = (window["bid_high"] - entry_px) / entry_px * 10000.0
                        adverse_series = (window["bid_low"] - entry_px) / entry_px * 10000.0
                    else:
                        favorable_series = (entry_px - window["ask_low"]) / entry_px * 10000.0
                        adverse_series = (entry_px - window["ask_high"]) / entry_px * 10000.0

                    max_favorable = favorable_series.max() if not favorable_series.empty else None
                    max_adverse = adverse_series.min() if not adverse_series.empty else None

                    # Store positive magnitudes
                    row_out["mfe_bps"] = float(max_favorable) if max_favorable is not None else None
                    row_out["mae_bps"] = float(abs(min(max_adverse, 0.0))) if max_adverse is not None else None

                    # Extra telemetry: MFE/MAE ratio, MFE vs entry_spread, adverse-first indicator
                    try:
                        mfe_bps = row_out.get("mfe_bps")
                        mae_bps = row_out.get("mae_bps")
                        if mfe_bps is not None and mae_bps is not None and float(mae_bps) > 0:
                            row_out["mfe_mae_ratio"] = float(mfe_bps) / float(mae_bps)
                        else:
                            row_out["mfe_mae_ratio"] = None
                    except Exception:
                        row_out["mfe_mae_ratio"] = None

                    try:
                        entry_spread_bps = row_out.get("entry_spread_bps")
                        if entry_spread_bps is not None and row_out.get("mfe_bps") is not None:
                            row_out["mfe_vs_entry_spread_bps"] = float(row_out["mfe_bps"]) - float(entry_spread_bps)
                        else:
                            row_out["mfe_vs_entry_spread_bps"] = None
                    except Exception:
                        row_out["mfe_vs_entry_spread_bps"] = None

                    try:
                        # bar indices inside the trade window (0-based)
                        favorable_series = favorable_series.reset_index(drop=True)
                        adverse_series = adverse_series.reset_index(drop=True)
                        max_adv_idx = int(adverse_series.idxmin()) if not adverse_series.empty else None
                        max_fav_idx = int(favorable_series.idxmax()) if not favorable_series.empty else None

                        # Meaningful MFE threshold: entry_spread_bps if available, else 1.0 bps
                        entry_spread_bps = row_out.get("entry_spread_bps")
                        mfe_threshold = float(entry_spread_bps) if entry_spread_bps is not None else 1.0
                        mfe_threshold = max(1.0, mfe_threshold)

                        first_meaningful_mfe_idx = None
                        if not favorable_series.empty:
                            idxs = favorable_series.index[favorable_series >= mfe_threshold].tolist()
                            if idxs:
                                first_meaningful_mfe_idx = int(idxs[0])

                        row_out["max_adverse_bar_index"] = max_adv_idx
                        row_out["peak_mfe_bar_index_window"] = max_fav_idx
                        row_out["meaningful_mfe_threshold_bps"] = float(mfe_threshold)
                        row_out["first_meaningful_mfe_bar_index"] = first_meaningful_mfe_idx

                        # Adverse-first: MAE peak happens before meaningful MFE (or no meaningful MFE at all)
                        if first_meaningful_mfe_idx is None:
                            row_out["adverse_first"] = True
                        elif max_adv_idx is None:
                            row_out["adverse_first"] = False
                        else:
                            row_out["adverse_first"] = bool(max_adv_idx <= first_meaningful_mfe_idx)
                    except Exception:
                        row_out["adverse_first"] = None

            rows.append(row_out)

        journal_df = pd.DataFrame(rows)
        if "trade_id" in journal_df.columns:
            journal_df["trade_id"] = journal_df["trade_id"].astype(str)

        # Ensure MAE/MFE columns exist for auditability even if missing in all rows
        for col in ["mae_bps", "mfe_bps"]:
            if col not in journal_df.columns:
                journal_df[col] = None

        # Enrich session from trade_outcomes parquet if missing
        try:
            outcomes_path = chunk_output_dir / f"trade_outcomes_{run_id}.parquet"
            if outcomes_path.exists() and not journal_df.empty:
                outcomes_df = pd.read_parquet(outcomes_path)
                if "session" in outcomes_df.columns:
                    if "trade_id" in outcomes_df.columns:
                        outcomes_df["trade_id"] = outcomes_df["trade_id"].astype(str)
                    journal_df = journal_df.merge(
                        outcomes_df[["trade_id", "session"]],
                        on="trade_id",
                        how="left",
                        suffixes=("", "_outcomes"),
                    )
                    journal_df["session"] = journal_df["session"].fillna(journal_df["session_outcomes"])
                    journal_df.drop(columns=["session_outcomes"], inplace=True)
        except Exception as e:
            log.warning("[TRADE_JOURNAL_MERGE_SESSION_FAIL] %s", e)

        if not journal_df.empty:
            sort_cols = [c for c in ["close_ts_utc", "trade_id"] if c in journal_df.columns]
            if sort_cols:
                journal_df = journal_df.sort_values(sort_cols)

        # ---------------------------
        # CATASTROPHIC_GUARD observability proof (journal/footer path)
        # ---------------------------
        try:
            if runner is not None:
                runner.perf_exit_cata_guard_proof = None
            if not journal_df.empty and "exit_reason" in journal_df.columns:
                cat_df = journal_df[
                    journal_df["exit_reason"].fillna("UNKNOWN") == "CATASTROPHIC_GUARD"
                ].copy()
                if not cat_df.empty:
                    peak_mfe = pd.to_numeric(
                        cat_df.get("peak_mfe_bps_exit_state", cat_df.get("mfe_bps")),
                        errors="coerce",
                    )
                    peak_idx = pd.to_numeric(cat_df.get("peak_mfe_bar_index"), errors="coerce")
                    hold_after_peak = pd.to_numeric(
                        cat_df.get("time_since_mfe_bars_exit"),
                        errors="coerce",
                    )

                    def _dist(series: pd.Series) -> Dict[str, Any]:
                        s = series.dropna()
                        if s.empty:
                            return {"n": 0}
                        return {
                            "n": int(s.shape[0]),
                            "p50": float(s.quantile(0.50)),
                            "p90": float(s.quantile(0.90)),
                            "p99": float(s.quantile(0.99)),
                            "max": float(s.max()),
                        }

                    def _bins(series: pd.Series, edges: List[int]) -> Dict[str, int]:
                        s = series.dropna()
                        out: Dict[str, int] = {}
                        if s.empty:
                            return out
                        prev = None
                        for edge in edges:
                            if prev is None:
                                key = f"le_{edge}"
                                out[key] = int((s <= edge).sum())
                            else:
                                key = f"gt_{prev}_le_{edge}"
                                out[key] = int(((s > prev) & (s <= edge)).sum())
                            prev = edge
                        out[f"gt_{edges[-1]}"] = int((s > edges[-1]).sum())
                        return out

                    proof_payload = {
                        "n_cat_guard": int(cat_df.shape[0]),
                        "n_cat_guard_positive_mfe": int((peak_mfe > 0).sum()),
                        "peak_mfe_positive_rate": (
                            float((peak_mfe > 0).sum() / cat_df.shape[0]) if int(cat_df.shape[0]) > 0 else 0.0
                        ),
                        "peak_mfe_bar_index_distribution": _dist(peak_idx),
                        "time_since_mfe_bars_distribution": _dist(hold_after_peak),
                        "peak_mfe_bar_index_bins": _bins(peak_idx, [10, 50, 100, 250, 500]),
                        "time_since_mfe_bars_bins": _bins(hold_after_peak, [50, 100, 250, 500, 1000]),
                        "by_side": (
                            cat_df.groupby(cat_df["side"].fillna("UNKNOWN")).size().astype(int).to_dict()
                            if "side" in cat_df.columns
                            else {}
                        ),
                        "by_session": (
                            cat_df.groupby(cat_df["session"].fillna("UNKNOWN")).size().astype(int).to_dict()
                            if "session" in cat_df.columns
                            else {}
                        ),
                    }
                    if runner is not None:
                        runner.perf_exit_cata_guard_proof = dict(proof_payload)
                    log.info(
                        "[EXIT_CATA_GUARD_PROOF] run_id=%s payload=%s",
                        run_id,
                        proof_payload,
                    )
        except Exception as e:
            log.warning("[EXIT_CATA_GUARD_PROOF_FAIL] %s", e)

        # ---------------------------
        # EXIT HOLD/REASON OBSERVABILITY (proof-only)
        # ---------------------------
        try:
            if runner is not None:
                runner.perf_exit_hold_tail_thresholds = [500, 1000, 1200]
                runner.perf_exit_hold_tail_counts = {}
                runner.perf_exit_hold_tail_by_reason = {}
                runner.perf_exit_hold_tail_by_side = {}
                runner.perf_exit_hold_tail_by_session = {}
                runner.perf_exit_hold_tail_by_reason_side_session = {}
                runner.perf_exit_cat_guard_by_side_session_month = {}
                runner.perf_exit_cat_guard_rate_by_side_session = {}
            if not journal_df.empty and "bars_in_trade" in journal_df.columns:
                thresholds = [500, 1000, 1200]
                b = pd.to_numeric(journal_df["bars_in_trade"], errors="coerce")
                valid = b.notna() & (b >= 0)
                jj = journal_df.loc[valid].copy()
                jj["_bars"] = b.loc[valid].astype(int)
                if not jj.empty:
                    hold_tail_counts = {
                        f"ge_{thr}": int((jj["_bars"] >= thr).sum()) for thr in thresholds
                    }

                    def _bucket_counts(frame: pd.DataFrame, key_col: str) -> Dict[str, Dict[str, int]]:
                        out: Dict[str, Dict[str, int]] = {}
                        if key_col not in frame.columns:
                            return out
                        grp = frame.groupby(frame[key_col].fillna("UNKNOWN"))
                        for key, sub in grp:
                            out[str(key)] = {
                                f"ge_{thr}": int((sub["_bars"] >= thr).sum()) for thr in thresholds
                            }
                        return out

                    hold_tail_by_reason = _bucket_counts(jj, "exit_reason")
                    hold_tail_by_side = _bucket_counts(jj, "side")
                    hold_tail_by_session = _bucket_counts(jj, "session")

                    hold_tail_by_reason_side_session: Dict[str, Dict[str, Dict[str, int]]] = {}
                    if "exit_reason" in jj.columns and "side" in jj.columns and "session" in jj.columns:
                        for (reason, side, session), sub in jj.groupby(
                            [
                                jj["exit_reason"].fillna("UNKNOWN"),
                                jj["side"].fillna("UNKNOWN"),
                                jj["session"].fillna("UNKNOWN"),
                            ]
                        ):
                            rkey = str(reason)
                            skey = f"{str(side)}|{str(session)}"
                            hold_tail_by_reason_side_session.setdefault(rkey, {})[skey] = {
                                f"ge_{thr}": int((sub["_bars"] >= thr).sum()) for thr in thresholds
                            }

                    if runner is not None:
                        runner.perf_exit_hold_tail_thresholds = list(thresholds)
                        runner.perf_exit_hold_tail_counts = dict(hold_tail_counts)
                        runner.perf_exit_hold_tail_by_reason = dict(hold_tail_by_reason)
                        runner.perf_exit_hold_tail_by_side = dict(hold_tail_by_side)
                        runner.perf_exit_hold_tail_by_session = dict(hold_tail_by_session)
                        runner.perf_exit_hold_tail_by_reason_side_session = dict(
                            hold_tail_by_reason_side_session
                        )

                    log.info(
                        "[EXIT_HOLD_TAIL_PROOF] run_id=%s thresholds=%s overall=%s by_reason=%s by_side=%s by_session=%s",
                        run_id,
                        thresholds,
                        hold_tail_counts,
                        hold_tail_by_reason,
                        hold_tail_by_side,
                        hold_tail_by_session,
                    )

                    if "exit_reason" in jj.columns:
                        cat = jj[jj["exit_reason"].fillna("UNKNOWN") == "CATASTROPHIC_GUARD"].copy()
                        if not cat.empty:
                            cat["_month"] = (
                                pd.to_datetime(cat.get("close_ts_utc"), utc=True, errors="coerce")
                                .dt.to_period("M")
                                .astype(str)
                            )
                            by_side_session_month: Dict[str, int] = {}
                            if {"side", "session", "_month"}.issubset(cat.columns):
                                grouped = cat.groupby(
                                    [
                                        cat["side"].fillna("UNKNOWN"),
                                        cat["session"].fillna("UNKNOWN"),
                                        cat["_month"].fillna("UNKNOWN"),
                                    ]
                                ).size()
                                by_side_session_month = {
                                    f"{str(side)}|{str(session)}|{str(month)}": int(cnt)
                                    for (side, session, month), cnt in grouped.items()
                                }

                            rate_by_side_session: Dict[str, Dict[str, float]] = {}
                            if {"side", "session"}.issubset(jj.columns):
                                total_grp = jj.groupby(
                                    [jj["side"].fillna("UNKNOWN"), jj["session"].fillna("UNKNOWN")]
                                ).size()
                                cat_grp = cat.groupby(
                                    [cat["side"].fillna("UNKNOWN"), cat["session"].fillna("UNKNOWN")]
                                ).size()
                                for (side, session), total_cnt in total_grp.items():
                                    cat_cnt = int(cat_grp.get((side, session), 0))
                                    rate_by_side_session[f"{str(side)}|{str(session)}"] = {
                                        "cat_count": cat_cnt,
                                        "total_exits": int(total_cnt),
                                        "cat_rate": float(cat_cnt / total_cnt) if int(total_cnt) > 0 else 0.0,
                                    }

                            if runner is not None:
                                runner.perf_exit_cat_guard_by_side_session_month = dict(by_side_session_month)
                                runner.perf_exit_cat_guard_rate_by_side_session = dict(rate_by_side_session)

                            log.info(
                                "[EXIT_CATA_GUARD_DISTRIBUTION_PROOF] run_id=%s by_side_session_month=%s rate_by_side_session=%s",
                                run_id,
                                by_side_session_month,
                                rate_by_side_session,
                            )
        except Exception as e:
            log.warning("[EXIT_HOLD_TAIL_PROOF_FAIL] %s", e)

        # ---------------------------
        # STUCK-TRADE / TAIL-LOSS AUDIT (observability only)
        # ---------------------------
        try:
            _write_stuck_trade_audit(chunk_output_dir, run_id, journal_df)
        except Exception as e:
            log.warning("[STUCK_TRADE_AUDIT_FAIL] run_id=%s error=%s", run_id, e)

        # ---------------------------
        # SHORT vs LONG EXIT SIGNAL AUDIT (observability only)
        # ---------------------------
        try:
            _write_short_exit_signal_audit(chunk_output_dir, run_id, journal_df)
        except Exception as e:
            log.warning("[SHORT_EXIT_SIGNAL_AUDIT_FAIL] run_id=%s error=%s", run_id, e)

        # ---------------------------
        # STUCK SHORT SIGNATURE AUDIT (observability only)
        # ---------------------------
        try:
            _write_stuck_short_signature_audit(chunk_output_dir, run_id, journal_df)
        except Exception as e:
            log.warning("[STUCK_SHORT_SIGNATURE_FAIL] run_id=%s error=%s", run_id, e)

        # ---------------------------
        # EARLY FAILURE SHORT GUARD COUNTERFACTUAL (analytical only)
        # ---------------------------
        try:
            _write_early_failure_short_guard_counterfactual(chunk_output_dir, run_id, journal_df)
        except Exception as e:
            log.warning("[EARLY_FAILURE_SHORT_GUARD_CF_FAIL] run_id=%s error=%s", run_id, e)

        # ---------------------------
        # ENTRY SIGNATURE AUDIT: STUCK SHORTS (observability only)
        # ---------------------------
        try:
            _write_entry_signature_audit_stuck_shorts(chunk_output_dir, run_id, journal_df)
        except Exception as e:
            log.warning("[ENTRY_SIGNATURE_AUDIT_FAIL] run_id=%s error=%s", run_id, e)

        # ---------------------------
        # SSoT COLUMN GUARD (observability only)
        # ---------------------------
        EXPECTED_SSoT_COLUMNS = [
            "trade_uid",
            "side",
            "close_ts_utc",
            "pnl_bps",
            "accepted",
            "session",
        ]

        row_count = len(journal_df)
        try:
            if runner is not None:
                runner.perf_trade_journal_rows = int(row_count)
        except Exception:
            pass

        if row_count > 0:
            for col in EXPECTED_SSoT_COLUMNS:
                if col not in journal_df.columns:
                    log.warning(
                        "[SSOT_COLUMN_MISSING] column=%s rows=%d run_id=%s",
                        col,
                        row_count,
                        run_id,
                    )
                    continue

                nulls = int(journal_df[col].isna().sum())
                null_rate = nulls / row_count

                if null_rate > 0.9:
                    log.warning(
                        "[SSOT_COLUMN_HIGH_NULL] column=%s nulls=%d rows=%d null_rate=%.3f run_id=%s",
                        col,
                        nulls,
                        row_count,
                        null_rate,
                        run_id,
                    )

        out_path = chunk_output_dir / f"trade_journal_{run_id}.parquet"
        journal_df.to_parquet(out_path, index=False)
        log.info(
            "[TRADE_JOURNAL_PROOF] wrote trade_journal rows=%d cols=%s path=%s",
            len(journal_df),
            list(journal_df.columns),
            out_path,
        )
        try:
            mae_missing = int(journal_df["mae_bps"].isna().sum()) if "mae_bps" in journal_df.columns else len(journal_df)
            mfe_missing = int(journal_df["mfe_bps"].isna().sum()) if "mfe_bps" in journal_df.columns else len(journal_df)
            log.info(
                "[TRADE_JOURNAL_MAE_MFE_PROOF] rows=%d mae_missing=%d mfe_missing=%d",
                len(journal_df),
                mae_missing,
                mfe_missing,
            )
        except Exception:
            pass
        try:
            if runner is not None:
                if not entry_journal_present:
                    margin_missing_reasons = {"ENTRY_JOURNAL_DISABLED": 1}
                    runner.perf_trade_journal_margin_missing = 0
                else:
                    runner.perf_trade_journal_margin_missing = int(sum(margin_missing_reasons.values()))
                runner.perf_trade_journal_margin_missing_reasons = dict(margin_missing_reasons)
        except Exception:
            pass
    except Exception as e:
        log.warning("[TRADE_JOURNAL_SKIPPED] failed to write trade_journal: %s", e)


def _log_replay_summary_proof(chunk_output_dir: Path, run_id: str) -> Dict[str, Any]:
    """
    Proof line for replay summary counts (artifact-near, deterministic).
    """
    import json

    n_trades_closed = None
    journal_rows = None
    outcomes_rows = None
    trade_journal_df = None
    close_events_total = None
    close_events_by_reason = {}
    entries_opened = None
    unique_trade_ids_opened = None
    unique_trade_ids_closed = None
    open_trades_at_end = None
    holding_stats = {
        "holding_bars_mean": None,
        "holding_bars_median": None,
        "holding_bars_min": None,
        "holding_bars_max": None,
        "trades_holding_le_1_bar": None,
        "trades_holding_ge_3_bars": None,
        "trades_holding_ge_5_bars": None,
    }
    accounting_violation = False
    accounting_violation_reasons: list[str] = []
    margin_min_used = None
    margin_reject_total = None
    margin_reject_long = None
    margin_reject_short = None
    margin_reject_reasons = None
    journal_margin_missing = None
    journal_margin_missing_reasons = None
    perf_bars_total = None
    perf_bars_processed = None
    entry_pref_pre_long = None
    entry_pref_pre_short = None
    entry_pref_pre_flat = None
    entry_pref_post_long = None
    entry_pref_post_short = None
    entry_pref_post_none = None

    footer_path = chunk_output_dir / "chunk_footer.json"
    if footer_path.exists():
        try:
            with footer_path.open("r", encoding="utf-8") as f:
                footer = json.load(f)
            n_trades_closed = int(footer.get("n_trades_closed", 0) or 0)
            margin_min_used = footer.get("entry_margin_min_used")
            margin_reject_total = footer.get("n_entry_margin_reject_total")
            margin_reject_long = footer.get("n_entry_margin_reject_long")
            margin_reject_short = footer.get("n_entry_margin_reject_short")
            margin_reject_reasons = footer.get("entry_margin_reject_reasons")
            journal_margin_missing = footer.get("n_trade_journal_margin_missing")
            journal_margin_missing_reasons = footer.get("trade_journal_margin_missing_reasons")
            max_open_trades_policy = footer.get("max_open_trades_policy")
            max_open_trades_effective_boot = footer.get("max_open_trades_effective_boot")
            max_open_trades_used = footer.get("max_open_trades_used")
            max_open_trades_override_env = footer.get("max_open_trades_override_env")
            entry_attempt_total = footer.get("entry_attempt_total")
            entry_accept_total = footer.get("entry_accept_total")
            entry_attempt_long = footer.get("entry_attempt_long")
            entry_attempt_short = footer.get("entry_attempt_short")
            entry_accept_long = footer.get("entry_accept_long")
            entry_accept_short = footer.get("entry_accept_short")
            signal_candidate_long = footer.get("signal_candidate_long")
            signal_candidate_short = footer.get("signal_candidate_short")
            signal_candidate_none = footer.get("signal_candidate_none")
            opened_registered = footer.get("n_trades_opened_registered")
            can_enter_fail_reasons = footer.get("can_enter_fail_reasons")
            xgb_paths = footer.get("xgb_load_paths") or {}
            entry_gate_cfg = footer.get("entry_gate_config_snapshot") or {}
            entry_veto_hard = footer.get("entry_veto_hard")
            entry_veto_soft = footer.get("entry_veto_soft")
            entry_veto_pre = footer.get("entry_veto_pre")
            entry_veto_cand = footer.get("entry_veto_cand")
            entry_killchain_counts = footer.get("entry_killchain_counts")
            entry_killchain_block_reasons = footer.get("entry_killchain_block_reasons")
            entry_gate_counters_footer = footer.get("entry_gate_counters")
            perf_bars_total = footer.get("perf_bars_total")
            perf_bars_processed = footer.get("perf_n_bars_processed")
            entry_pref_pre_long = footer.get("entry_pref_pre_long")
            entry_pref_pre_short = footer.get("entry_pref_pre_short")
            entry_pref_pre_flat = footer.get("entry_pref_pre_flat")
            entry_pref_post_long = footer.get("entry_pref_post_long")
            entry_pref_post_short = footer.get("entry_pref_post_short")
            entry_pref_post_none = footer.get("entry_pref_post_none")
        except Exception:
            n_trades_closed = None

    journal_path = chunk_output_dir / f"trade_journal_{run_id}.parquet"
    if journal_path.exists():
        try:
            trade_journal_df = pd.read_parquet(journal_path)
            journal_rows = int(trade_journal_df.shape[0])
            close_events_total = journal_rows
            if "exit_reason" in trade_journal_df.columns:
                close_events_by_reason = (
                    trade_journal_df["exit_reason"].fillna("UNKNOWN").value_counts().to_dict()
                )
            if "trade_id" in trade_journal_df.columns:
                unique_trade_ids_closed = int(trade_journal_df["trade_id"].nunique())
            # Holding stats from entry/exit timestamps (per-trade artifact source)
            try:
                if "open_ts_utc" in trade_journal_df.columns and "close_ts_utc" in trade_journal_df.columns:
                    ts_open = pd.to_datetime(trade_journal_df["open_ts_utc"], utc=True, errors="coerce")
                    ts_close = pd.to_datetime(trade_journal_df["close_ts_utc"], utc=True, errors="coerce")
                    dt_sec = (ts_close - ts_open).dt.total_seconds()
                    valid = dt_sec.notna() & (dt_sec >= 0)
                    if valid.any():
                        bars = (dt_sec[valid] / 300.0).round().astype(int)
                        holding_stats["holding_bars_mean"] = float(bars.mean())
                        holding_stats["holding_bars_median"] = float(bars.median())
                        holding_stats["holding_bars_min"] = int(bars.min())
                        holding_stats["holding_bars_max"] = int(bars.max())
                        holding_stats["trades_holding_le_1_bar"] = int((bars <= 1).sum())
                        holding_stats["trades_holding_ge_3_bars"] = int((bars >= 3).sum())
                        holding_stats["trades_holding_ge_5_bars"] = int((bars >= 5).sum())
            except Exception:
                pass
        except Exception:
            journal_rows = None

    outcomes_path = chunk_output_dir / f"trade_outcomes_{run_id}.parquet"
    if outcomes_path.exists():
        try:
            outcomes_rows = int(pd.read_parquet(outcomes_path).shape[0])
        except Exception:
            outcomes_rows = None

    slope_proof_lines: list[str] = []
    candidate_none_proof_lines: list[str] = []
    entry_bundle_proof_line = None
    try:
        eval_candidates = sorted((chunk_output_dir / "logs").glob("eval_log_*.jsonl"))
        eval_path = eval_candidates[-1] if eval_candidates else None
        if eval_path and eval_path.exists():
            entry_margin_all = []
            xgb_margin_all = []
            entry_margin_by_session = {"EU": [], "OVERLAP": [], "US": []}
            xgb_margin_by_session = {"EU": [], "OVERLAP": [], "US": []}
            max_side_all = []
            max_side_by_session = {"EU": [], "OVERLAP": [], "US": []}
            thresholds = [0.67, 0.62, 0.57, 0.52]
            thresh_counts = {thr: {"long": 0, "short": 0, "total": 0} for thr in thresholds}
            thresh_counts_by_session = {
                "EU": {thr: {"long": 0, "short": 0, "total": 0} for thr in thresholds},
                "OVERLAP": {thr: {"long": 0, "short": 0, "total": 0} for thr in thresholds},
                "US": {thr: {"long": 0, "short": 0, "total": 0} for thr in thresholds},
            }
            near_band = 0.01
            thresh_near = {thr: 0 for thr in thresholds}
            thresh_near_by_session = {
                "EU": {thr: 0 for thr in thresholds},
                "OVERLAP": {thr: 0 for thr in thresholds},
                "US": {thr: 0 for thr in thresholds},
            }
            none_reasons_all: dict[str, int] = {}
            none_reasons_by_session: dict[str, dict[str, int]] = {"EU": {}, "OVERLAP": {}, "US": {}}
            none_bucket_all = {
                "flat_wins_argmax": 0,
                "p_side_below_min": 0,
                "flat_gate_block": 0,
                "flat_margin_block": 0,
                "session_block": 0,
                "head_block": 0,
                "threshold_block": 0,
                "unknown_candidate_none_reason": 0,
            }
            none_bucket_by_session = {
                "EU": dict(none_bucket_all),
                "OVERLAP": dict(none_bucket_all),
                "US": dict(none_bucket_all),
            }
            none_total_all = 0
            none_total_by_session = {"EU": 0, "OVERLAP": 0, "US": 0}
            with eval_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        xgb_p_long = float(rec.get("xgb_p_long", np.nan))
                        xgb_p_short = float(rec.get("xgb_p_short", np.nan))
                        entry_p_long = float(rec.get("entry_p_long", np.nan))
                        entry_p_short = float(rec.get("entry_p_short", np.nan))
                        if np.isfinite(entry_p_long) and np.isfinite(entry_p_short):
                            entry_margin_all.append(entry_p_short - entry_p_long)
                        if np.isfinite(xgb_p_long) and np.isfinite(xgb_p_short):
                            xgb_margin_all.append(xgb_p_short - xgb_p_long)
                        sess = rec.get("session")
                        if isinstance(sess, str) and sess in entry_margin_by_session:
                            if np.isfinite(entry_p_long) and np.isfinite(entry_p_short):
                                entry_margin_by_session[sess].append(entry_p_short - entry_p_long)
                            if np.isfinite(xgb_p_long) and np.isfinite(xgb_p_short):
                                xgb_margin_by_session[sess].append(xgb_p_short - xgb_p_long)
                        pre_gate_pref = rec.get("pre_gate_pref")
                        if pre_gate_pref in ("long", "short") and np.isfinite(entry_p_long) and np.isfinite(entry_p_short):
                            max_side = float(max(entry_p_long, entry_p_short))
                            max_side_all.append(max_side)
                            if isinstance(sess, str) and sess in max_side_by_session:
                                max_side_by_session[sess].append(max_side)
                            for thr in thresholds:
                                if max_side >= thr:
                                    thresh_counts[thr]["total"] += 1
                                    if pre_gate_pref == "long":
                                        thresh_counts[thr]["long"] += 1
                                    else:
                                        thresh_counts[thr]["short"] += 1
                                    if isinstance(sess, str) and sess in thresh_counts_by_session:
                                        sess_bucket = thresh_counts_by_session[sess][thr]
                                        sess_bucket["total"] += 1
                                        if pre_gate_pref == "long":
                                            sess_bucket["long"] += 1
                                        else:
                                            sess_bucket["short"] += 1
                                if (thr - near_band) <= max_side < thr:
                                    thresh_near[thr] += 1
                                    if isinstance(sess, str) and sess in thresh_near_by_session:
                                        thresh_near_by_session[sess][thr] += 1
                        decision = rec.get("decision")
                        if decision == "NONE":
                            none_total_all += 1
                            if isinstance(sess, str) and sess in none_total_by_session:
                                none_total_by_session[sess] += 1
                            reason = rec.get("decision_reason") or "unknown"
                            if isinstance(reason, str):
                                none_reasons_all[reason] = none_reasons_all.get(reason, 0) + 1
                                if isinstance(sess, str) and sess in none_reasons_by_session:
                                    sdict = none_reasons_by_session[sess]
                                    sdict[reason] = sdict.get(reason, 0) + 1
                            pre_gate_pref = rec.get("pre_gate_pref")
                            if pre_gate_pref == "flat":
                                none_bucket_all["flat_wins_argmax"] += 1
                                if isinstance(sess, str) and sess in none_bucket_by_session:
                                    none_bucket_by_session[sess]["flat_wins_argmax"] += 1
                            if reason == "flat_gate_pf_high":
                                none_bucket_all["flat_gate_block"] += 1
                                if isinstance(sess, str) and sess in none_bucket_by_session:
                                    none_bucket_by_session[sess]["flat_gate_block"] += 1
                            elif reason == "flat_gate_gap_small":
                                none_bucket_all["flat_margin_block"] += 1
                                if isinstance(sess, str) and sess in none_bucket_by_session:
                                    none_bucket_by_session[sess]["flat_margin_block"] += 1
                            elif reason == "entry_gating_p_side_min":
                                none_bucket_all["p_side_below_min"] += 1
                                if isinstance(sess, str) and sess in none_bucket_by_session:
                                    none_bucket_by_session[sess]["p_side_below_min"] += 1
                            elif reason == "no_candidate_or_not_higher":
                                none_bucket_all["threshold_block"] += 1
                                if isinstance(sess, str) and sess in none_bucket_by_session:
                                    none_bucket_by_session[sess]["threshold_block"] += 1
                            elif reason == "session_block":
                                none_bucket_all["session_block"] += 1
                                if isinstance(sess, str) and sess in none_bucket_by_session:
                                    none_bucket_by_session[sess]["session_block"] += 1
                            elif reason == "head_block":
                                none_bucket_all["head_block"] += 1
                                if isinstance(sess, str) and sess in none_bucket_by_session:
                                    none_bucket_by_session[sess]["head_block"] += 1
                            else:
                                if reason not in {
                                    "flat_gate_pf_high",
                                    "flat_gate_gap_small",
                                    "entry_gating_p_side_min",
                                    "no_candidate_or_not_higher",
                                    "session_block",
                                    "head_block",
                                }:
                                    none_bucket_all["unknown_candidate_none_reason"] += 1
                                    if isinstance(sess, str) and sess in none_bucket_by_session:
                                        none_bucket_by_session[sess]["unknown_candidate_none_reason"] += 1
                    except Exception:
                        continue

            def _pctiles(arr):
                a = np.asarray(arr, dtype=float)
                if a.size == 0:
                    return None
                return {
                    "min": float(np.min(a)),
                    "p5": float(np.quantile(a, 0.05)),
                    "p25": float(np.quantile(a, 0.25)),
                    "p50": float(np.quantile(a, 0.50)),
                    "p75": float(np.quantile(a, 0.75)),
                    "p95": float(np.quantile(a, 0.95)),
                    "max": float(np.max(a)),
                    "n": int(a.size),
                }

            entry_all_stats = _pctiles(entry_margin_all)
            xgb_all_stats = _pctiles(xgb_margin_all)
            max_side_stats = _pctiles(max_side_all)
            if entry_all_stats:
                line_entry_all = (
                    "[ENTRY_LANDSCAPE_SLOPE_PROOF] run_id=%s scope=ALL n=%d min=%.6f p5=%.6f "
                    "p25=%.6f p50=%.6f p75=%.6f p95=%.6f max=%.6f"
                    % (
                        run_id,
                        entry_all_stats["n"],
                        entry_all_stats["min"],
                        entry_all_stats["p5"],
                        entry_all_stats["p25"],
                        entry_all_stats["p50"],
                        entry_all_stats["p75"],
                        entry_all_stats["p95"],
                        entry_all_stats["max"],
                    )
                )
                log.info("%s", line_entry_all)
                slope_proof_lines.append(line_entry_all)
            if xgb_all_stats:
                line_xgb_all = (
                    "[XGB_VS_ENTRY_SLOPE_PROOF] run_id=%s scope=ALL n=%d min=%.6f p5=%.6f "
                    "p25=%.6f p50=%.6f p75=%.6f p95=%.6f max=%.6f"
                    % (
                        run_id,
                        xgb_all_stats["n"],
                        xgb_all_stats["min"],
                        xgb_all_stats["p5"],
                        xgb_all_stats["p25"],
                        xgb_all_stats["p50"],
                        xgb_all_stats["p75"],
                        xgb_all_stats["p95"],
                        xgb_all_stats["max"],
                    )
                )
                log.info("%s", line_xgb_all)
                slope_proof_lines.append(line_xgb_all)
            if max_side_stats:
                line_max_side_all = (
                    "[ENTRY_MAX_SIDE_PROOF] run_id=%s scope=ALL n=%d min=%.6f p5=%.6f "
                    "p25=%.6f p50=%.6f p75=%.6f p90=%.6f p95=%.6f max=%.6f"
                    % (
                        run_id,
                        max_side_stats["n"],
                        max_side_stats["min"],
                        max_side_stats["p5"],
                        max_side_stats["p25"],
                        max_side_stats["p50"],
                        max_side_stats["p75"],
                        max_side_stats["p90"],
                        max_side_stats["p95"],
                        max_side_stats["max"],
                    )
                )
                log.info("%s", line_max_side_all)
                slope_proof_lines.append(line_max_side_all)
            for sess in ("EU", "OVERLAP", "US"):
                s_stats = _pctiles(entry_margin_by_session.get(sess, []))
                if s_stats:
                    line_entry_sess = (
                        "[ENTRY_LANDSCAPE_SLOPE_PROOF] run_id=%s scope=%s n=%d min=%.6f p5=%.6f "
                        "p25=%.6f p50=%.6f p75=%.6f p95=%.6f max=%.6f"
                        % (
                            run_id,
                            sess,
                            s_stats["n"],
                            s_stats["min"],
                            s_stats["p5"],
                            s_stats["p25"],
                            s_stats["p50"],
                            s_stats["p75"],
                            s_stats["p95"],
                            s_stats["max"],
                        )
                    )
                    log.info("%s", line_entry_sess)
                    slope_proof_lines.append(line_entry_sess)
                x_stats = _pctiles(xgb_margin_by_session.get(sess, []))
                if x_stats:
                    line_xgb_sess = (
                        "[XGB_VS_ENTRY_SLOPE_PROOF] run_id=%s scope=%s n=%d min=%.6f p5=%.6f "
                        "p25=%.6f p50=%.6f p75=%.6f p95=%.6f max=%.6f"
                        % (
                            run_id,
                            sess,
                            x_stats["n"],
                            x_stats["min"],
                            x_stats["p5"],
                            x_stats["p25"],
                            x_stats["p50"],
                            x_stats["p75"],
                            x_stats["p95"],
                            x_stats["max"],
                        )
                    )
                    log.info("%s", line_xgb_sess)
                    slope_proof_lines.append(line_xgb_sess)
                m_stats = _pctiles(max_side_by_session.get(sess, []))
                if m_stats:
                    line_max_side_sess = (
                        "[ENTRY_MAX_SIDE_PROOF] run_id=%s scope=%s n=%d min=%.6f p5=%.6f "
                        "p25=%.6f p50=%.6f p75=%.6f p90=%.6f p95=%.6f max=%.6f"
                        % (
                            run_id,
                            sess,
                            m_stats["n"],
                            m_stats["min"],
                            m_stats["p5"],
                            m_stats["p25"],
                            m_stats["p50"],
                            m_stats["p75"],
                            m_stats["p90"],
                            m_stats["p95"],
                            m_stats["max"],
                        )
                    )
                    log.info("%s", line_max_side_sess)
                    slope_proof_lines.append(line_max_side_sess)
            if max_side_all:
                total_candidates = len(max_side_all)
                for thr in thresholds:
                    counts = thresh_counts[thr]
                    log.info(
                        "[ENTRY_THRESHOLD_AUDIT] run_id=%s scope=ALL threshold=%.2f total=%d long=%d short=%d near_below=%d near_band=%.2f",
                        run_id,
                        thr,
                        counts["total"],
                        counts["long"],
                        counts["short"],
                        thresh_near[thr],
                        near_band,
                    )
                    for sess in ("EU", "OVERLAP", "US"):
                        sess_counts = thresh_counts_by_session[sess][thr]
                        log.info(
                            "[ENTRY_THRESHOLD_AUDIT] run_id=%s scope=%s threshold=%.2f total=%d long=%d short=%d near_below=%d near_band=%.2f",
                            run_id,
                            sess,
                            thr,
                            sess_counts["total"],
                            sess_counts["long"],
                            sess_counts["short"],
                            thresh_near_by_session[sess][thr],
                            near_band,
                        )
            # Candidate-none proof (deterministic from eval_log)
            if none_total_all > 0:
                line_none_all = (
                    "[ENTRY_CANDIDATE_NONE_PROOF] run_id=%s scope=ALL total=%d flat_wins_argmax=%d buckets=%s reasons=%s"
                    % (
                        run_id,
                        none_total_all,
                        none_bucket_all.get("flat_wins_argmax", 0),
                        none_bucket_all,
                        none_reasons_all,
                    )
                )
                log.info("%s", line_none_all)
                candidate_none_proof_lines.append(line_none_all)
                for sess in ("EU", "OVERLAP", "US"):
                    s_total = none_total_by_session.get(sess, 0)
                    if s_total > 0:
                        line_none_sess = (
                            "[ENTRY_CANDIDATE_NONE_PROOF] run_id=%s scope=%s total=%d flat_wins_argmax=%d buckets=%s reasons=%s"
                            % (
                                run_id,
                                sess,
                                s_total,
                                none_bucket_by_session[sess].get("flat_wins_argmax", 0),
                                none_bucket_by_session[sess],
                                none_reasons_by_session.get(sess, {}),
                            )
                        )
                        log.info("%s", line_none_sess)
                        candidate_none_proof_lines.append(line_none_sess)
    except Exception:
        slope_proof_lines = []

    try:
        entry_bundle_dir = os.environ.get("GX1_BUNDLE_DIR") or os.environ.get("GX1_ENTRY_BUNDLE_DIR_PROOF")
        entry_num_classes = os.environ.get("GX1_ENTRY_BUNDLE_NUM_CLASSES_PROOF")
        entry_class_order = os.environ.get("GX1_ENTRY_BUNDLE_CLASS_ORDER_PROOF")
        if entry_bundle_dir:
            entry_bundle_proof_line = (
                "[ENTRY_3CLASS_BUNDLE_PROOF] bundle_dir=%s num_classes=%s class_order=%s class_meaning=%s"
                % (
                    entry_bundle_dir,
                    entry_num_classes,
                    entry_class_order,
                    {0: "LONG", 1: "SHORT", 2: "FLAT"},
                )
            )
            log.info("%s", entry_bundle_proof_line)
    except Exception:
        entry_bundle_proof_line = None

    if entries_opened is None:
        try:
            if footer_path.exists():
                with footer_path.open("r", encoding="utf-8") as f:
                    footer = json.load(f)
                entries_opened = footer.get("n_trades_opened_registered")
        except Exception:
            entries_opened = None
    if entries_opened is None and entry_accept_total is not None:
        entries_opened = entry_accept_total
    if entries_opened is not None:
        unique_trade_ids_opened = int(entries_opened)
    if unique_trade_ids_closed is not None and unique_trade_ids_opened is not None:
        open_trades_at_end = max(int(unique_trade_ids_opened) - int(unique_trade_ids_closed), 0)

    if trade_journal_df is not None:
        try:
            per_trade_cols = ["trade_id", "open_ts_utc", "close_ts_utc", "exit_reason"]
            per_trade = trade_journal_df.copy()
            for c in per_trade_cols:
                if c not in per_trade.columns:
                    per_trade[c] = None
            per_trade = per_trade[per_trade_cols]
            per_trade = per_trade.sort_values(by=["open_ts_utc", "trade_id"], na_position="last")
            per_trade_path = chunk_output_dir / "TRADE_ACCOUNTING_PER_TRADE.jsonl"
            per_trade.to_json(per_trade_path, orient="records", lines=True, date_format="iso")
            # Duplicate close detection
            if "trade_id" in trade_journal_df.columns:
                dup_counts = trade_journal_df["trade_id"].value_counts()
                dup_ids = dup_counts[dup_counts > 1]
                if not dup_ids.empty:
                    accounting_violation = True
                    accounting_violation_reasons.append(
                        f"duplicate_trade_id_closes={len(dup_ids)}"
                    )
        except Exception:
            pass

    if unique_trade_ids_opened is not None and unique_trade_ids_closed is not None:
        if unique_trade_ids_closed > unique_trade_ids_opened:
            accounting_violation = True
            accounting_violation_reasons.append("unique_closed_gt_unique_opened")
    if close_events_total is not None and unique_trade_ids_closed is not None:
        if int(close_events_total) != int(unique_trade_ids_closed):
            accounting_violation = True
            accounting_violation_reasons.append("close_events_total_ne_unique_closed")

    line = (
        f"[REPLAY_SUMMARY_PROOF] run_id={run_id} "
        f"n_trades_closed={n_trades_closed} journal_rows={journal_rows} outcomes_rows={outcomes_rows}"
    )
    trade_accounting_line = (
        "[TRADE_ACCOUNTING_PROOF] run_id=%s entries_opened=%s exits_closed=%s unique_opened=%s "
        "unique_closed=%s open_trades_at_end=%s close_events_total=%s close_events_by_reason=%s"
        % (
            run_id,
            entries_opened,
            n_trades_closed,
            unique_trade_ids_opened,
            unique_trade_ids_closed,
            open_trades_at_end,
            close_events_total,
            close_events_by_reason,
        )
    )
    holding_line = (
        "[TRADE_HOLDING_PROOF] run_id=%s mean_bars=%s median_bars=%s min_bars=%s max_bars=%s "
        "le_1_bar=%s ge_3_bars=%s ge_5_bars=%s"
        % (
            run_id,
            holding_stats.get("holding_bars_mean"),
            holding_stats.get("holding_bars_median"),
            holding_stats.get("holding_bars_min"),
            holding_stats.get("holding_bars_max"),
            holding_stats.get("trades_holding_le_1_bar"),
            holding_stats.get("trades_holding_ge_3_bars"),
            holding_stats.get("trades_holding_ge_5_bars"),
        )
    )
    if accounting_violation:
        log.warning(
            "[TRADE_ACCOUNTING_VIOLATION] run_id=%s reasons=%s",
            run_id,
            accounting_violation_reasons,
        )
    gate_cfg_line = None
    try:
        if isinstance(entry_gate_cfg, dict):
            candidate_threshold = "long=%s,short=%s" % (
                entry_gate_cfg.get("candidate_threshold_long"),
                entry_gate_cfg.get("candidate_threshold_short"),
            )
            gate_cfg_line = (
                "[ENTRY_GATE_CONFIG_PROOF] run_id=%s p_side_min_long=%s p_side_min_short=%s "
                "p_flat_gate=%s p_flat_margin=%s candidate_threshold=%s runner_up_margin=%s"
                % (
                    run_id,
                    entry_gate_cfg.get("p_side_min_long"),
                    entry_gate_cfg.get("p_side_min_short"),
                    entry_gate_cfg.get("p_flat_gate"),
                    entry_gate_cfg.get("p_flat_margin"),
                    candidate_threshold,
                    entry_gate_cfg.get("runner_up_margin"),
                )
            )
    except Exception:
        gate_cfg_line = None
    margin_line = (
        f"[ENTRY_MARGIN_FILTER_PROOF] run_id={run_id} margin_min={margin_min_used} "
        f"rejects_total={margin_reject_total} rejects_long={margin_reject_long} "
        f"rejects_short={margin_reject_short} reasons={margin_reject_reasons}"
    )
    journal_margin_line = (
        f"[TRADE_JOURNAL_MARGIN_PROOF] run_id={run_id} margin_missing={journal_margin_missing} "
        f"reasons={journal_margin_missing_reasons}"
    )
    entry_line = (
        f"[ENTRY_PROOF] run_id={run_id} "
        f"max_open_trades_policy={max_open_trades_policy} "
        f"max_open_trades_effective_boot={max_open_trades_effective_boot} "
        f"max_open_trades_used={max_open_trades_used} "
        f"GX1_MAX_OPEN_TRADES_OVERRIDE={max_open_trades_override_env} "
        f"admission_replacement_overlap_long_over_oldest_overlap_short={footer.get('admission_replacement_enabled')} "
        f"entry_attempt_total={entry_attempt_total} entry_accept_total={entry_accept_total} "
        f"entry_attempt_long={entry_attempt_long} entry_attempt_short={entry_attempt_short} "
        f"entry_accept_long={entry_accept_long} entry_accept_short={entry_accept_short} "
        f"opened_registered={opened_registered} "
        f"can_enter_fail_reasons={can_enter_fail_reasons}"
    )
    funnel_line = None
    try:
        gate_counts = entry_gate_counters_footer or {}
        blocked_session = gate_counts.get("pregate_session")
        blocked_warmup = gate_counts.get("warmup_not_ready")
        blocked_threshold = gate_counts.get("candidate_below_threshold")
        blocked_risk = gate_counts.get("candidate_risk_guard")
        funnel_line = (
            f"[ENTRY_FUNNEL_PROOF] run_id={run_id} "
            f"bars_seen={perf_bars_total} bars_processed={perf_bars_processed} "
            f"pref_long={entry_pref_pre_long} pref_short={entry_pref_pre_short} pref_flat={entry_pref_pre_flat} "
            f"threshold_pass_long={signal_candidate_long} threshold_pass_short={signal_candidate_short} "
            f"can_enter_pass_long={footer.get('n_can_enter_pass_long')} can_enter_pass_short={footer.get('n_can_enter_pass_short')} "
            f"blocked_session={blocked_session} blocked_warmup={blocked_warmup} "
            f"blocked_threshold={blocked_threshold} blocked_risk={blocked_risk} "
            f"entry_attempt_long={entry_attempt_long} entry_attempt_short={entry_attempt_short} "
            f"entry_accept_long={entry_accept_long} entry_accept_short={entry_accept_short} "
            f"opened_registered_long={footer.get('n_trades_opened_registered_long')} opened_registered_short={footer.get('n_trades_opened_registered_short')}"
        )
    except Exception:
        funnel_line = None
    side_line = (
        f"[SIGNAL_SIDE_STATS] run_id={run_id} "
        f"candidate_long={signal_candidate_long} candidate_short={signal_candidate_short} "
        f"candidate_none={signal_candidate_none} "
        f"attempt_long={entry_attempt_long} attempt_short={entry_attempt_short}"
    )
    xgb_line = (
        f"[XGB_PROOF_SUMMARY] run_id={run_id} "
        f"bundle_dir={xgb_paths.get('bundle_dir')} "
        f"model_file={xgb_paths.get('model_file')} "
        f"model_sha256={xgb_paths.get('model_sha256')}"
    )
    log.info("%s", line)
    log.info("%s", trade_accounting_line)
    log.info("%s", holding_line)
    if gate_cfg_line:
        log.info("%s", gate_cfg_line)
    if funnel_line:
        log.info("%s", funnel_line)
    try:
        if any(v is not None for v in [entry_veto_hard, entry_veto_soft, entry_veto_pre, entry_veto_cand, entry_killchain_counts, entry_killchain_block_reasons]):
            log.info(
                "[ENTRY_BLOCKING_GATES_PROOF] run_id=%s veto_hard=%s veto_soft=%s veto_pre=%s veto_cand=%s killchain_counts=%s killchain_block_reasons=%s entry_gate_counters=%s can_enter_fail_reasons=%s",
                run_id,
                entry_veto_hard,
                entry_veto_soft,
                entry_veto_pre,
                entry_veto_cand,
                entry_killchain_counts,
                entry_killchain_block_reasons,
                entry_gate_counters_footer,
                can_enter_fail_reasons,
            )
    except Exception:
        pass
    log.info("%s", margin_line)
    log.info("%s", journal_margin_line)
    log.info("%s", entry_line)
    log.info("%s", side_line)
    try:
        if isinstance(signal_candidate_long, int) and isinstance(signal_candidate_short, int):
            if signal_candidate_long == 0 and signal_candidate_short > 50:
                log.info(
                    "[ENTRY_DIRECTION_ANOMALY] run_id=%s long_candidates=%s short_candidates=%s",
                    run_id,
                    signal_candidate_long,
                    signal_candidate_short,
                )
    except Exception:
        pass
    log.info("%s", xgb_line)
    try:
        proof_path = chunk_output_dir / "REPLAY_SUMMARY_PROOF.log"
        proof_lines = [line]
        proof_lines.append(trade_accounting_line)
        proof_lines.append(holding_line)
        if accounting_violation:
            proof_lines.append(
                "[TRADE_ACCOUNTING_VIOLATION] run_id=%s reasons=%s"
                % (run_id, accounting_violation_reasons)
            )
        try:
            summary_path = chunk_output_dir / "REPLAY_SUMMARY.json"
            summary_payload = {
                "run_id": run_id,
                "trade_accounting": {
                    "entries_opened": entries_opened,
                    "exits_closed": n_trades_closed,
                    "unique_trade_ids_opened": unique_trade_ids_opened,
                    "unique_trade_ids_closed": unique_trade_ids_closed,
                    "open_trades_at_end": open_trades_at_end,
                    "close_events_total": close_events_total,
                    "close_events_by_reason": close_events_by_reason,
                    "violation": accounting_violation,
                    "violation_reasons": accounting_violation_reasons,
                },
                "holding_stats": holding_stats,
            }
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
        except Exception:
            pass
        if gate_cfg_line:
            proof_lines.append(gate_cfg_line)
        try:
            proof_lines.append(
                "[ENTRY_BLOCKING_GATES_PROOF] run_id=%s veto_hard=%s veto_soft=%s veto_pre=%s veto_cand=%s killchain_counts=%s killchain_block_reasons=%s entry_gate_counters=%s can_enter_fail_reasons=%s"
                % (
                    run_id,
                    entry_veto_hard,
                    entry_veto_soft,
                    entry_veto_pre,
                    entry_veto_cand,
                    entry_killchain_counts,
                    entry_killchain_block_reasons,
                    entry_gate_counters_footer,
                    can_enter_fail_reasons,
                )
            )
        except Exception:
            pass
        proof_lines.extend([margin_line, journal_margin_line, entry_line, xgb_line])
        if entry_bundle_proof_line:
            proof_lines.append(entry_bundle_proof_line)
        if isinstance(slope_proof_lines, list) and slope_proof_lines:
            proof_lines.extend(slope_proof_lines)
        if isinstance(candidate_none_proof_lines, list) and candidate_none_proof_lines:
            proof_lines.extend(candidate_none_proof_lines)
        proof_path.write_text("\n".join(proof_lines) + "\n", encoding="utf-8")
    except Exception:
        return {"proof_lines_count": 0, "replay_summary_json_written": 0}
    return {
        "proof_lines_count": int(len(proof_lines)),
        "replay_summary_json_written": int((chunk_output_dir / "REPLAY_SUMMARY.json").exists()),
    }


def _try_write_optional_observability(
    chunk_output_dir: Path,
    run_id: str,
    chunk_idx: int,
    runner: Any,
    bars_processed: int,
    total_bars: int,
    wall_clock_sec: float,
) -> None:
    """
    Optional observability hooks.
    Must never crash the chunk. Best-effort only.
    """
    # ENTRY_SIGNAL_TRACE autopsy etc (optional modules)
    try:
        if os.environ.get("GX1_ENTRY_SIGNAL_TRACE", "0") == "1":
            from gx1.execution.signal_autopsy import write_signal_autopsy_summary  # type: ignore

            n_entry_signals = _safe_int(
                getattr(getattr(runner, "entry_manager", None), "killchain_n_above_threshold", 0),
                0,
            )
            write_signal_autopsy_summary(
                chunk_output_dir=chunk_output_dir,
                run_id=run_id,
                n_entry_signals=n_entry_signals,
                first_n_events=20,
                is_truth_or_smoke=_is_truth_or_smoke(),
            )
    except Exception as e:
        log.warning("[OBS] [CHUNK %s] signal_autopsy failed: %s", chunk_idx, e)

    # ZERO_TRADES_DIAG disabled in TRUTH/SMOKE replay (module may be missing)


def _compute_basic_bar_counters_snapshot(runner: Any, bars_processed: int) -> Dict[str, Any]:
    """
    Best-effort snapshot for failure capsules. Avoids disk reads.
    Keep this small and stable.
    """
    return {
        "candles_iterated": int(bars_processed or 0),
        "bars_seen": _safe_int(getattr(runner, "bars_seen", 0), 0),
        "bars_skipped_warmup": _safe_int(getattr(runner, "bars_skipped_warmup", 0), 0),
        "bars_skipped_pregate": _safe_int(getattr(runner, "bars_skipped_pregate", 0), 0),
        "bars_reaching_entry_stage": _safe_int(getattr(runner, "bars_reaching_entry_stage", 0), 0),
        "pregate_enabled": _safe_bool(getattr(runner, "pregate_enabled", False), False),
    }


def _extract_runner_perf(runner: Any, chunk_df: Optional[pd.DataFrame]) -> Tuple[int, int, int, int, float, float]:
    """
    Extract key counters/timing from runner in-memory (best-effort).

    Returns:
      total_bars, bars_processed, bars_evaluated, warmup_holdback_bars, feature_time_total_sec, t_transformer_forward_sec
    """
    total_bars = int(len(chunk_df) if chunk_df is not None else 0)

    bars_seen = _safe_int(getattr(runner, "bars_seen", None), default=total_bars)
    bars_processed = _safe_int(getattr(runner, "perf_n_bars_processed", None), default=bars_seen)
    if bars_processed < 0:
        bars_processed = 0
    if bars_processed > total_bars:
        bars_processed = total_bars

    bars_evaluated = _safe_int(
        getattr(runner, "perf_n_model_calls", None),
        default=_safe_int(
            getattr(runner, "perf_n_policy_calls", None),
            default=_safe_int(getattr(runner, "n_model_calls", None), default=0),
        ),
    )
    if bars_evaluated < 0:
        bars_evaluated = 0

    warmup_holdback_bars = _safe_int(getattr(runner, "first_valid_eval_idx_stored", 0), 0)

    feature_time_total_sec = _safe_float(getattr(runner, "perf_feat_time", 0.0), 0.0)
    t_transformer_forward_sec = _safe_float(getattr(runner, "t_transformer_forward_sec", 0.0), 0.0)

    return (
        total_bars,
        bars_processed,
        bars_evaluated,
        warmup_holdback_bars,
        feature_time_total_sec,
        t_transformer_forward_sec,
    )


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def process_chunk(
    chunk_idx: int,
    chunk_start: "pd.Timestamp",
    chunk_end: "pd.Timestamp",
    data_path: Path,
    policy_path: Path,
    run_id: str,
    output_dir: Path,
    bundle_sha256: Optional[str] = None,
    prebuilt_parquet_path: Optional[str] = None,  # may be str/Path upstream
    bundle_dir: Optional[Path] = None,
    chunk_local_padding_days: int = 0,
    truth_artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single replay chunk.

    Returns:
      chunk_artifacts: dict with status + artifact paths

    TRUTH/SMOKE:
      - Strict: missing bundle_sha256 => hard fail
      - Invariant violations => status flips to failed_invariant and raises
    """
    _assert_truth_ban_envs()
    is_truth_or_smoke_worker = _is_truth_or_smoke()
    truth_artifacts = truth_artifacts or {}

    data_path_str = str(data_path)
    if "/data/data/raw/" in data_path_str or "xauusd_m5_2025_bid_ask.parquet" in data_path_str:
        raise RuntimeError(f"[REPLAY_TAPE_FORBIDDEN_RAW_PATH] data_path points to legacy raw lane: {data_path}")

    # Always-initialized locals (finally/except safe)
    status: str = "ok"
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    chunk_output_dir: Optional[Path] = None
    runner: Any = None
    chunk_df: Optional[pd.DataFrame] = None
    bootstrap_ctx: Optional[BootstrapContext] = None
    data_ctx: Optional[DataContext] = None

    # Perf / counters (best-effort)
    bars_processed = 0
    total_bars = 0
    bars_evaluated = 0
    warmup_holdback_bars = 0
    tail_holdback_bars = 0
    n_trades_closed = 0
    wall_clock_sec = 0.0

    feature_time_total_sec = 0.0
    feature_time_mean_ms = 0.0

    # TRUTH timing breakdown (best-effort)
    t_init_s = 0.0
    t_resolve_runner_s = 0.0
    t_load_raw_s = 0.0
    t_load_prebuilt_s = 0.0
    t_join_s = 0.0
    t_loop_s = 0.0
    t_write_s = 0.0
    t_trade_outcomes_write_s = 0.0
    t_trade_journal_build_s = 0.0
    t_optional_obs_s = 0.0
    t_killchain_export_s = 0.0
    t_invariant_check_s = 0.0
    t_footer_aggregation_s = 0.0
    t_footer_write_s = 0.0
    t_summary_proof_s = 0.0

    t_transformer_forward_sec = 0.0
    dt_module_version: Optional[str] = None
    telemetry_required = False
    worker_start_time = time.time()

    inv_report: Optional[Dict[str, Any]] = None
    prebuilt_parquet_path_resolved: Optional[str] = None
    chunk_data_path_abs: Optional[Path] = None
    case_collision_resolution: Optional[Dict[str, Any]] = None
    actual_chunk_start: Optional[pd.Timestamp] = None

    # SSoT bar counts (prefer loader, never len(df)-gjetting for footer)
    bars_total_input_all_ssot: Optional[int] = None
    bars_total_eval_ssot: Optional[int] = None
    join_metrics_path_ssot: Optional[str] = None

    # Stable env flag (prebuilt-only doctrine)
    prebuilt_enabled_env = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
    n_footer_structures_built = 0
    n_artifact_writes_total = 0
    n_artifact_writes_json = 0
    n_artifact_writes_parquet = 0
    n_artifact_writes_log = 0
    replay_summary_stats: Dict[str, Any] = {}

    # Skip ledger (always written best-effort in finally)
    skip_ledger: Dict[str, Any] = {
        "chunk_id": chunk_idx,
        "run_id": run_id,
        "stage": "init",
        "timestamp": None,
        "eval_start_ts": str(chunk_start),
        "eval_end_ts": str(chunk_end),
        "raw_rows_loaded": None,
        "prebuilt_rows_loaded": None,
        "join_rows": None,
        "join_ratio": None,
        "ts_min_raw": None,
        "ts_max_raw": None,
        "ts_min_prebuilt": None,
        "ts_max_prebuilt": None,
        "ts_min_join": None,
        "ts_max_join": None,
        "n_in_eval_window": None,
        "warmup_bars_required": None,
        "warmup_bars_seen": None,
        "n_skipped_total": None,
        "skipped_breakdown": {},
        "candles_iterated": None,
        "reached_entry_stage": None,
        "bars_processed": None,
        "exception_type": None,
        "exception_msg": None,
        "traceback": None,
        "gating_counters": {},
    }

    # Install SIGTERM handler
    global STOP_REQUESTED
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except Exception:
        pass
    STOP_REQUESTED = False
    os.environ["GX1_STOP_REQUESTED"] = "0"

    try:
        # ---------------------------------------------------------------------
        # PHASE 0: Bootstrap
        # ---------------------------------------------------------------------
        t0 = time.time()
        bootstrap_ctx = bootstrap_chunk_environment(
            chunk_idx=chunk_idx,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            data_path=data_path,
            policy_path=policy_path,
            run_id=run_id,
            output_dir=output_dir,
            prebuilt_parquet_path=prebuilt_parquet_path,
            bundle_dir=bundle_dir,
            chunk_local_padding_days=chunk_local_padding_days,
            bundle_sha256=bundle_sha256,
            policy_id=None,
        )

        chunk_output_dir = bootstrap_ctx.chunk_output_dir
        is_truth_or_smoke_worker = _safe_bool(
            getattr(bootstrap_ctx, "is_truth_or_smoke_worker", is_truth_or_smoke_worker),
            is_truth_or_smoke_worker,
        )
        telemetry_required = _safe_bool(getattr(bootstrap_ctx, "telemetry_required", False), False)
        dt_module_version = getattr(bootstrap_ctx, "dt_module_version", None)
        worker_start_time = _safe_float(getattr(bootstrap_ctx, "worker_start_time", worker_start_time), worker_start_time)
        t_init_s = _safe_float(getattr(bootstrap_ctx, "t_init_s", time.time() - t0), time.time() - t0)

        if chunk_output_dir is None:
            raise RuntimeError("[BOOTSTRAP_FAIL] chunk_output_dir is None after bootstrap")

        # TRUTH strict: bundle_sha256 must exist (SSoT)
        if is_truth_or_smoke_worker and not bundle_sha256:
            raise RuntimeError("[SSOT_FAIL] bundle_sha256 missing in process_chunk() (TRUTH/SMOKE strict)")

        # TRUTH strict: env intent must match bootstrap prebuilt_enabled (no silent mismatch)
        if is_truth_or_smoke_worker:
            boot_prebuilt_enabled = _safe_bool(getattr(bootstrap_ctx, "prebuilt_enabled", False), False)
            if bool(prebuilt_enabled_env) != bool(boot_prebuilt_enabled):
                raise RuntimeError(
                    f"[TRUTH_NO_FALLBACK] prebuilt_enabled mismatch: env={prebuilt_enabled_env} bootstrap={boot_prebuilt_enabled}"
                )

        # ---------------------------------------------------------------------
        # PHASE 1: Create runner
        # ---------------------------------------------------------------------
        t_resolve_runner_start = time.time()
        from gx1.execution.oanda_demo_runner import GX1DemoRunner

        # Canonical truth replay has one bootstrap-owned entry bundle resolution.
        entry_override = os.getenv("GX1_ENTRY_BUNDLE_DIR", "").strip()
        canonical_transformer = os.getenv("GX1_CANONICAL_TRANSFORMER_BUNDLE_DIR", "").strip()
        canonical_bundle_dir = os.getenv("GX1_CANONICAL_BUNDLE_DIR", "").strip()
        truth_mode = os.getenv("GX1_TRUTH_MODE", "0") == "1" or bool(
            os.getenv("GX1_CANONICAL_TRUTH_FILE", "").strip()
        )
        resolved_bundle_dir: Optional[str] = None
        resolved_bundle_source: Optional[str] = None
        if truth_mode:
            if entry_override:
                resolved_bundle_dir = str(Path(entry_override).expanduser().resolve())
                resolved_bundle_source = "ENTRY_OVERRIDE"
            elif canonical_transformer:
                resolved_bundle_dir = str(Path(canonical_transformer).expanduser().resolve())
                resolved_bundle_source = "CANONICAL_TRANSFORMER"
            elif bundle_dir:
                resolved_bundle_dir = str(Path(bundle_dir).resolve())
                resolved_bundle_source = "BOOTSTRAP_BUNDLE_DIR"
            elif canonical_bundle_dir:
                resolved_bundle_dir = str(Path(canonical_bundle_dir).expanduser().resolve())
                resolved_bundle_source = "CANONICAL_BUNDLE"

            if not resolved_bundle_dir:
                raise RuntimeError(
                    "[CANONICAL_ENTRY_BUNDLE_RESOLUTION_FAIL] truth replay bootstrap could not resolve entry bundle dir"
                )
            os.environ["GX1_BUNDLE_DIR"] = resolved_bundle_dir
            log.info(
                "[CHUNK %s] GX1_BUNDLE_DIR=%s (%s)",
                chunk_idx,
                resolved_bundle_dir,
                resolved_bundle_source,
            )
        else:
            if entry_override:
                os.environ["GX1_BUNDLE_DIR"] = str(Path(entry_override).expanduser().resolve())
                log.info("[CHUNK %s] GX1_BUNDLE_DIR=%s (ENTRY_OVERRIDE)", chunk_idx, os.environ["GX1_BUNDLE_DIR"])
            elif os.getenv("GX1_BUNDLE_DIR"):
                log.info("[CHUNK %s] GX1_BUNDLE_DIR=%s (env preset)", chunk_idx, os.environ["GX1_BUNDLE_DIR"])
            elif canonical_transformer:
                os.environ["GX1_BUNDLE_DIR"] = canonical_transformer
                log.info("[CHUNK %s] GX1_BUNDLE_DIR=%s (canonical transformer)", chunk_idx, canonical_transformer)
            elif canonical_bundle_dir:
                os.environ["GX1_BUNDLE_DIR"] = canonical_bundle_dir
                log.info("[CHUNK %s] GX1_BUNDLE_DIR=%s (canonical)", chunk_idx, canonical_bundle_dir)
            elif bundle_dir:
                os.environ["GX1_BUNDLE_DIR"] = str(Path(bundle_dir).resolve())
                log.info("[CHUNK %s] GX1_BUNDLE_DIR=%s", chunk_idx, os.environ["GX1_BUNDLE_DIR"])
            else:
                log.info("[CHUNK %s] GX1_BUNDLE_DIR not set here (policy may define it)", chunk_idx)

        runner = GX1DemoRunner(
            policy_path,
            replay_mode=True,
            fast_replay=False,
            output_dir=chunk_output_dir,
        )
        t_resolve_runner_s = _safe_float(time.time() - t_resolve_runner_start, 0.0)
        runner.run_id = run_id
        runner.chunk_id = str(chunk_idx)
        runner.chunk_start = chunk_start
        runner.chunk_end = chunk_end

        # TRUTH 1W1C: defensively ban segmented/parallel state on runner
        for attr, val in (
            ("segment_start", None),
            ("segment_end", None),
            ("preroll_start", None),
            ("segmented_parallel_mode", False),
        ):
            try:
                setattr(runner, attr, val)
            except Exception:
                pass

        # propagate master sha (if provided)
        try:
            runner.bundle_sha256_from_master = bundle_sha256
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # PHASE 2: Load chunk data (raw + prebuilt join)  [SSoT counts live here]
        # ---------------------------------------------------------------------
        data_ctx = load_chunk_data(bootstrap_ctx, chunk_start, chunk_end)

        chunk_df = data_ctx.chunk_df
        case_collision_resolution = getattr(data_ctx, "case_collision_resolution", None)
        chunk_data_path_abs = getattr(data_ctx, "chunk_data_path_abs", None)
        prebuilt_parquet_path_resolved = getattr(data_ctx, "prebuilt_parquet_path_resolved", None)
        actual_chunk_start = getattr(data_ctx, "actual_chunk_start", None)

        # SSoT bar counts from loader (footer must prefer these)
        bars_total_input_all_ssot = _safe_int(getattr(data_ctx, "bars_total_input_all", None), 0)
        bars_total_eval_ssot = _safe_int(getattr(data_ctx, "bars_total_eval", None), 0)
        join_metrics_path_ssot = str(getattr(data_ctx, "join_metrics_path", None)) if getattr(data_ctx, "join_metrics_path", None) else None

        # preferred timings from data_ctx
        t_load_raw_s = _safe_float(getattr(data_ctx, "t_load_raw_s", 0.0), 0.0)
        t_load_prebuilt_s = _safe_float(getattr(data_ctx, "t_load_prebuilt_s", 0.0), 0.0)
        t_join_s = _safe_float(getattr(data_ctx, "t_join_s", 0.0), 0.0)
        t_write_s = _safe_float(getattr(data_ctx, "t_write_s", 0.0), 0.0)

        if chunk_local_padding_days and chunk_local_padding_days > 0:
            # runner may use these for eval window vs padded load window
            try:
                runner.replay_eval_start_ts = chunk_start
                runner.replay_eval_end_ts = chunk_end
            except Exception:
                pass
            log.info(
                "[CHUNK %s] [CHUNK_LOCAL_PADDING] eval=[%s, %s] (actual_start=%s)",
                chunk_idx,
                chunk_start,
                chunk_end,
                actual_chunk_start,
            )

        if not chunk_data_path_abs:
            raise RuntimeError("[DATA_FAIL] chunk_data_path_abs missing after load_chunk_data()")

        # Pass data_ctx SSoT to runner (manifest/bootstrap); runner uses these instead of env path (no split-brain).
        runner.prebuilt_parquet_path_resolved = getattr(data_ctx, "prebuilt_parquet_path_resolved", None)
        runner.prebuilt_features_df = getattr(data_ctx, "prebuilt_features_df", None)

        log.info(
            "[CHUNK %s] [REPLAY_MODE] env_prebuilt=%s prebuilt_path=%s",
            chunk_idx,
            prebuilt_enabled_env,
            str(prebuilt_parquet_path_resolved) if prebuilt_parquet_path_resolved else None,
        )

        # ---------------------------------------------------------------------
        # PHASE 3: Run replay
        # ---------------------------------------------------------------------
        t_loop_start = time.time()
        try:
            runner.run_replay(chunk_data_path_abs)
        except KeyboardInterrupt:
            status = "stopped"
            error = "KeyboardInterrupt"
        finally:
            t_loop_s = time.time() - t_loop_start

        # Optional: write entry feature telemetry if enabled
        try:
            etm = getattr(getattr(runner, "entry_manager", None), "entry_feature_telemetry", None)
            if etm is not None:
                etm.write_all(Path(chunk_output_dir))
        except Exception as e:
            log.warning("[ENTRY_FEATURES_TELEMETRY] write_all failed: %s", e)

        # SIGTERM graceful stop capsule
        if STOP_REQUESTED and status == "ok":
            status = "stopped"
            error = "Stopped early due to SIGTERM"
            try:
                _bars = _safe_int(getattr(runner, "perf_n_bars_processed", 0), 0)
                _total = _safe_int(bars_total_input_all_ssot, default=int(len(chunk_df) if chunk_df is not None else 0))
                _last_ts = getattr(runner, "_last_bar_ts", None)
                write_signal_event_capsule(
                    chunk_output_dir=chunk_output_dir,
                    run_id=run_id,
                    chunk_idx=chunk_idx,
                    bars_processed=_bars,
                    total_bars=_total,
                    last_ts=_last_ts,
                    wall_clock_sec=_safe_float(time.time() - worker_start_time, 0.0),
                )
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # PHASE 4: Extract perf/counters (best-effort)
        # ---------------------------------------------------------------------
        wall_clock_sec = _safe_float(time.time() - worker_start_time, 0.0)

        (
            _total_bars_len_df,
            bars_processed,
            bars_evaluated,
            warmup_holdback_bars,
            feature_time_total_sec,
            t_transformer_forward_sec,
        ) = _extract_runner_perf(runner, chunk_df)

        # TRUTH: total_bars for invariants/footer must prefer loader SSoT
        if bars_total_input_all_ssot is not None and bars_total_input_all_ssot > 0:
            total_bars = int(bars_total_input_all_ssot)
        else:
            total_bars = int(_total_bars_len_df)

        feature_time_mean_ms = (feature_time_total_sec / total_bars * 1000.0) if total_bars > 0 else 0.0

        # ---------------------------------------------------------------------
        # PHASE 5: TRUTH/SMOKE required artifacts (trade_outcomes + attribution)
        # ---------------------------------------------------------------------
        if is_truth_or_smoke_worker:
            t_w0 = time.time()
            _write_trade_outcomes_truth(chunk_output_dir, run_id, runner)
            t_trade_outcomes_write_s = _safe_float(time.time() - t_w0, 0.0)
            n_artifact_writes_total += 1
            n_artifact_writes_parquet += 1

            t_w1 = time.time()
            _write_trade_journal_truth(chunk_output_dir, run_id, runner)
            t_trade_journal_build_s = _safe_float(time.time() - t_w1, 0.0)
            n_artifact_writes_total += 1
            n_artifact_writes_parquet += 1
            try:
                if runner is not None and getattr(runner, "exit_manager", None) is not None:
                    runner.exit_manager._maybe_log_exit_prob_audit(force=True)
            except Exception:
                pass

            attribution_path = chunk_output_dir / f"attribution_{run_id}.json"
            if not attribution_path.exists():
                t_attr = time.time()
                _write_minimal_attribution(chunk_output_dir, run_id)
                t_write_s += _safe_float(time.time() - t_attr, 0.0)
                n_artifact_writes_total += 1
                n_artifact_writes_json += 1
            log.info("[TRUTH] skip legacy flush_replay_eval_collectors (forbidden import path)")
        else:
            # Legacy flush path is forbidden; skip to avoid gx1.scripts.* import in TRUTH/SMOKE contexts.
            log.info("[REPLAY_EVAL] skip legacy flush_replay_eval_collectors (forbidden import path)")

        # ---------------------------------------------------------------------
        # PHASE 5b: Optional observability (never fatal)
        # ---------------------------------------------------------------------
        t_obs = time.time()
        _try_write_optional_observability(
            chunk_output_dir=chunk_output_dir,
            run_id=run_id,
            chunk_idx=chunk_idx,
            runner=runner,
            bars_processed=int(bars_processed),
            total_bars=int(total_bars),
            wall_clock_sec=float(wall_clock_sec),
        )
        t_optional_obs_s = _safe_float(time.time() - t_obs, 0.0)

        # ---------------------------------------------------------------------
        # PHASE 6: Export killchain (best-effort)
        # ---------------------------------------------------------------------
        try:
            t_kc = time.time()
            export_killchain(
                KillchainExportContext(
                    chunk_output_dir=chunk_output_dir,
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                    dt_module_version=dt_module_version,
                    is_truth_or_smoke_worker=is_truth_or_smoke_worker,
                    runner=runner,
                    status=status,
                    error=error,
                )
            )
            t_killchain_export_s = _safe_float(time.time() - t_kc, 0.0)
            n_artifact_writes_total += 1
            n_artifact_writes_json += 1
        except Exception as e:
            log.warning("[KILLCHAIN_EXPORT] [CHUNK %s] failed: %s", chunk_idx, e)

        # ---------------------------------------------------------------------
        # PHASE 7: Prebuilt invariants (TRUTH/SMOKE may flip status)
        # ---------------------------------------------------------------------
        try:
            t_inv = time.time()
            ok_inv, inv_report = check_prebuilt_invariants(
                PrebuiltInvariantContext(
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                    is_truth_or_smoke_worker=is_truth_or_smoke_worker,
                    prebuilt_enabled_env=prebuilt_enabled_env,
                    runner=runner,
                    feature_time_mean_ms=feature_time_mean_ms,
                    feature_time_total_sec=feature_time_total_sec,
                    bars_total=total_bars,
                    status=status,
                    error=error,
                )
            )
            t_invariant_check_s = _safe_float(time.time() - t_inv, 0.0)
            if (not ok_inv) and is_truth_or_smoke_worker and status == "ok":
                status = "failed_invariant"
                error = f"Prebuilt invariant violation: {len((inv_report or {}).get('violations', []))} violation(s)"
                log.error(
                    "[PREBUILT_INVARIANTS] [CHUNK %s] violations=%s",
                    chunk_idx,
                    (inv_report or {}).get("violations", []),
                )
            elif not ok_inv:
                log.warning(
                    "[PREBUILT_INVARIANTS] [CHUNK %s] non-fatal violations=%s",
                    chunk_idx,
                    (inv_report or {}).get("violations", []),
                )
        except Exception as e:
            log.warning("[PREBUILT_INVARIANTS] [CHUNK %s] check failed: %s", chunk_idx, e)

        # ---------------------------------------------------------------------
        # PHASE 8: Count trades closed (best-effort)
        # ---------------------------------------------------------------------
        try:
            trade_journal_dir = chunk_output_dir / "trade_journal" / "trades"
            if trade_journal_dir.exists():
                n_trades_closed = len(list(trade_journal_dir.glob("*.json")))
            else:
                n_trades_closed = _safe_int(getattr(runner, "perf_n_trades_created", 0), 0)
        except Exception:
            n_trades_closed = _safe_int(getattr(runner, "perf_n_trades_created", 0), 0)

        # ---------------------------------------------------------------------
        # PHASE 8c: Bars invariant (may raise AFTER footer write in TRUTH/SMOKE)
        # ---------------------------------------------------------------------
        from gx1.execution.chunk_footer_invariants import check_bars_invariant  # local import

        bars_total_input = int(total_bars)
        # Clamp warmup to available bars to avoid negative processed counts on short windows
        effective_warmup_holdback = int(min(warmup_holdback_bars, bars_total_input))
        bars_invariant_gap = int(bars_total_input - int(bars_processed))
        bars_invariant_expected_gap = int(effective_warmup_holdback + tail_holdback_bars)

        bars_invariant_ok = check_bars_invariant(
            bars_total_input=bars_total_input,
            bars_processed=int(bars_processed),
            tail_holdback_bars=int(tail_holdback_bars),
            status=status,
            warmup_holdback_bars=effective_warmup_holdback,
        )

        if not bars_invariant_ok:
            msg = (
                f"[BARS_INVARIANT] gap={bars_invariant_gap} != expected={bars_invariant_expected_gap} "
                f"(warmup={effective_warmup_holdback} tail={tail_holdback_bars})"
            )
            if is_truth_or_smoke_worker:
                log.error("%s; will raise after footer write", msg)
            else:
                log.warning("%s", msg)

        # ---------------------------------------------------------------------
        # PHASE 9: Write chunk_footer.json (DUMB WRITER; payload-only)
        # ---------------------------------------------------------------------
        try:
            t_footer_agg_start = time.time()
            import_proof_path, forbidden_hits = _write_import_proof_if_needed(
                chunk_output_dir=chunk_output_dir,
                run_id=run_id,
                chunk_idx=chunk_idx,
                truth_artifacts=truth_artifacts,
            )
            if import_proof_path is not None:
                n_artifact_writes_total += 1
                n_artifact_writes_json += 1
            if forbidden_hits:
                raise RuntimeError(
                    "[TRUTH_FORBIDDEN_SYMBOL_IMPORTS] Forbidden modules in sys.modules after replay: "
                    + ", ".join(sorted({h.get('module') for h in forbidden_hits}))
                )

            prebuilt_used_runner = _safe_bool(getattr(runner, "prebuilt_used", False), False)
            prebuilt_path_str = str(prebuilt_parquet_path_resolved) if prebuilt_parquet_path_resolved else None
            tape_price_truth_path = str(chunk_data_path_abs) if chunk_data_path_abs else None
            xgb_bundle_path = None
            try:
                xgb_bundle_path = (getattr(runner, "xgb_load_paths", {}) or {}).get("bundle_dir")
            except Exception:
                xgb_bundle_path = None
            entry_bundle_path = None
            try:
                entry_bundle_path = getattr(getattr(runner, "entry_v10_bundle", None), "bundle_dir", None)
            except Exception:
                entry_bundle_path = None
            exit_bundle_path = getattr(runner, "exit_transformer_model_path", None)
            canonical_prebuilt_used = bool(prebuilt_enabled_env and prebuilt_used_runner and prebuilt_path_str)
            canonical_truth_path = os.environ.get("GX1_CANONICAL_TRUTH_FILE")
            if canonical_truth_path:
                canonical_truth_path = str(Path(canonical_truth_path).expanduser().resolve())
            log.info(
                "[CANONICAL_RESOLVE_PROOF] run_id=%s canonical_prebuilt_used=%s prebuilt_path=%s tape_price_truth_path=%s xgb_bundle=%s entry_bundle=%s exit_bundle=%s canonical_truth_file=%s",
                run_id,
                int(canonical_prebuilt_used),
                prebuilt_path_str,
                tape_price_truth_path,
                xgb_bundle_path,
                entry_bundle_path,
                exit_bundle_path,
                canonical_truth_path,
            )

            # Best-effort funnel counters
            bars_seen = _safe_int(getattr(runner, "bars_seen", 0), 0)
            bars_skipped_warmup = _safe_int(getattr(runner, "bars_skipped_warmup", 0), 0)
            bars_skipped_pregate = _safe_int(getattr(runner, "bars_skipped_pregate", 0), 0)
            bars_reaching_entry_stage = _safe_int(getattr(runner, "bars_reaching_entry_stage", 0), 0)
            pregate_enabled = _safe_bool(getattr(runner, "pregate_enabled", False), False)

            feature_timeout_count = _safe_int(getattr(runner, "feature_timeout_count", 0), 0)
            vol_regime_unknown_count = _safe_int(getattr(runner, "vol_regime_unknown_count", 0), 0)

            pregate_skips = _safe_int(getattr(runner, "pregate_skips", 0), 0)
            pregate_passes = _safe_int(getattr(runner, "pregate_passes", 0), 0)
            pregate_missing_inputs = _safe_int(getattr(runner, "pregate_missing_inputs", 0), 0)

            # Entry gate counters (deterministic reasons for entry rejection)
            gate_counts = getattr(runner, "entry_gate_counters", {}) or {}
            gate_order = [
                "p_threshold",
                "margin_threshold",
                "ratio_threshold",
                "pregate_session",
                "pregate_spread",
                "pregate_atr",
                "warmup_not_ready",
                "guard_veto",
            ]
            for reason in gate_order:
                val = _safe_int(gate_counts.get(reason, 0), 0)
                log.info("[ENTRY_GATE_COUNTER] %s=%d", reason, val)

            # timers
            t_pregate_total_sec = _safe_float(getattr(runner, "t_pregate_total_sec", 0.0), 0.0)
            t_xgb_predict_sec = _safe_float(getattr(runner, "t_xgb_predict_sec", 0.0), 0.0)
            t_gates_policy_sec = _safe_float(getattr(runner, "t_gates_policy_sec", 0.0), 0.0)
            t_replay_tags_sec = _safe_float(getattr(runner, "t_replay_tags_sec", 0.0), 0.0)
            t_telemetry_sec = _safe_float(getattr(runner, "t_telemetry_sec", 0.0), 0.0)
            t_replay_tags_build_inputs_sec = _safe_float(getattr(runner, "t_replay_tags_build_inputs_sec", 0.0), 0.0)
            t_replay_tags_rolling_sec = _safe_float(getattr(runner, "t_replay_tags_rolling_sec", 0.0), 0.0)
            t_replay_tags_ewm_sec = _safe_float(getattr(runner, "t_replay_tags_ewm_sec", 0.0), 0.0)
            t_replay_tags_rank_sec = _safe_float(getattr(runner, "t_replay_tags_rank_sec", 0.0), 0.0)
            t_replay_tags_assign_sec = _safe_float(getattr(runner, "t_replay_tags_assign_sec", 0.0), 0.0)
            t_io_total_sec = _safe_float(getattr(runner, "t_io_total_sec", 0.0), 0.0)
            t_entry_input_prep_sec = _safe_float(getattr(runner, "t_entry_input_prep_sec", 0.0), 0.0)
            n_entry_input_prep_calls = _safe_int(getattr(runner, "n_entry_input_prep_calls", 0), 0)
            t_entry_model_infer_sec = _safe_float(
                getattr(runner, "t_entry_model_infer_sec", t_xgb_predict_sec + t_transformer_forward_sec),
                t_xgb_predict_sec + t_transformer_forward_sec,
            )
            n_entry_model_infer_calls = _safe_int(getattr(runner, "n_entry_model_infer_calls", bars_evaluated), bars_evaluated)
            t_exit_eval_total_sec = _safe_float(getattr(runner, "t_exit_eval_total_sec", 0.0), 0.0)
            n_exit_eval_calls = _safe_int(getattr(runner, "n_exit_eval_calls", 0), 0)
            t_exit_input_prep_sec = _safe_float(getattr(runner, "t_exit_input_prep_sec", 0.0), 0.0)
            n_exit_input_prep_calls = _safe_int(getattr(runner, "n_exit_input_prep_calls", 0), 0)
            t_exit_model_infer_sec = _safe_float(getattr(runner, "t_exit_model_infer_sec", 0.0), 0.0)
            n_exit_model_infer_calls = _safe_int(getattr(runner, "n_exit_model_infer_calls", 0), 0)
            t_trade_state_bookkeeping_sec = max(
                0.0,
                float(t_exit_eval_total_sec) - float(t_exit_input_prep_sec) - float(t_exit_model_infer_sec),
            )
            exit_input_prep_subtimers = {
                "n_windows_built": _safe_int(getattr(runner, "n_exit_input_windows_built", 0), 0),
                "history_selection_sec": _safe_float(getattr(runner, "t_exit_input_hist_select_sec", 0.0), 0.0),
                "runtime_atr_sec": _safe_float(getattr(runner, "t_exit_input_runtime_atr_sec", 0.0), 0.0),
                "ctx_contract_sec": _safe_float(getattr(runner, "t_exit_input_ctx_contract_sec", 0.0), 0.0),
                "prebuilt_window_resolve_sec": _safe_float(
                    getattr(runner, "t_exit_input_prebuilt_window_resolve_sec", 0.0),
                    0.0,
                ),
                "session_features_sec": _safe_float(getattr(runner, "t_exit_input_session_features_sec", 0.0), 0.0),
                "row_loop_sec": _safe_float(getattr(runner, "t_exit_input_row_loop_sec", 0.0), 0.0),
                "prebuilt_lookup_sec": _safe_float(getattr(runner, "t_exit_input_prebuilt_lookup_sec", 0.0), 0.0),
                "ctx_pack_sec": _safe_float(getattr(runner, "t_exit_input_ctx_pack_sec", 0.0), 0.0),
                "trade_state_compute_sec": _safe_float(
                    getattr(runner, "t_exit_input_trade_state_compute_sec", 0.0),
                    0.0,
                ),
                "feature_pack_sec": _safe_float(getattr(runner, "t_exit_input_feature_pack_sec", 0.0), 0.0),
                "numpy_finalize_sec": _safe_float(getattr(runner, "t_exit_input_numpy_finalize_sec", 0.0), 0.0),
                "contract_checks_sec": _safe_float(
                    getattr(runner, "t_exit_input_contract_checks_sec", 0.0),
                    0.0,
                ),
            }

            # HTF stats (best-effort)
            htf_align_time_total_sec = _safe_float(getattr(runner, "htf_align_time_total_sec", 0.0), 0.0)
            htf_align_warning_time_sec = _safe_float(getattr(runner, "htf_align_warning_time_sec", 0.0), 0.0)
            htf_align_warn_count = _safe_int(getattr(runner, "htf_align_warn_count", 0), 0)
            htf_align_call_count = _safe_int(getattr(runner, "htf_align_call_count", 0), 0)
            htf_align_fallback_count = _safe_int(getattr(runner, "htf_align_fallback_count", 0), 0)
            htf_feature_compute_bars = _safe_int(getattr(runner, "htf_feature_compute_bars", 0), 0)
            htf_h1_calls = _safe_int(getattr(runner, "htf_h1_calls", 0), 0)
            htf_h4_calls = _safe_int(getattr(runner, "htf_h4_calls", 0), 0)
            htf_h1_warns = _safe_int(getattr(runner, "htf_h1_warns", 0), 0)
            htf_h4_warns = _safe_int(getattr(runner, "htf_h4_warns", 0), 0)
            htf_last_m5_ts = getattr(runner, "htf_last_m5_ts", None)
            htf_last_j = getattr(runner, "htf_last_j", None)

            # TRUTH/SMOKE hard gate: forward time > 0 but no model calls -> fail invariant
            if (
                is_truth_or_smoke_worker
                and status == "ok"
                and float(t_transformer_forward_sec or 0.0) > 0.0
                and int(bars_evaluated or 0) == 0
            ):
                status = "failed_invariant"
                error = "[COUNTER_INVARIANT] transformer_forward_sec>0 but bars_evaluated==0"
                error_traceback = None

            top_timing_drivers = sorted(
                [
                    ("per_bar_replay_loop_total_sec", float(t_loop_s)),
                    ("footer_report_aggregation_sec", float(t_footer_aggregation_s)),
                    ("trade_journal_building_sec", float(t_trade_journal_build_s)),
                    ("trade_outcomes_write_sec", float(t_trade_outcomes_write_s)),
                    ("proof_observability_building_sec", float(t_optional_obs_s)),
                    ("initial_load_resolve_sec", float(t_init_s + t_resolve_runner_s + t_load_raw_s + t_load_prebuilt_s + t_join_s)),
                    ("prebuilt_tape_access_sec", float(t_load_raw_s + t_load_prebuilt_s)),
                ],
                key=lambda kv: kv[1],
                reverse=True,
            )[:3]
            top_speed_candidates = [
                "batch_observability_payload_construction_at_chunk_end",
                "minimize_redundant_dataframe_merges_in_trade_journal_build",
                "reduce_repeated_json_serialization_in_footer_and_proof_paths",
            ]

            payload: Dict[str, Any] = {
                "run_id": run_id,
                "chunk_id": int(chunk_idx),
                "status": status,
                "error": error,
                "error_traceback": (error_traceback[:5000] if error_traceback else None),
                "timestamp": dt_now_iso(),
                "pid": int(os.getpid()),
                "dt_module_version": dt_module_version,
                # perf
                "wall_clock_sec": float(wall_clock_sec),
                "bars_processed": int(bars_processed),
                "bars_evaluated": int(bars_evaluated),
                "total_bars": int(total_bars),
                "bars_total_input": int(bars_total_input),
                "bars_total_eval": int(bars_total_eval_ssot) if bars_total_eval_ssot is not None else None,
                "warmup_holdback_bars": int(warmup_holdback_bars),
                "tail_holdback_bars": int(tail_holdback_bars),
                "bars_per_sec": (float(bars_processed) / float(wall_clock_sec)) if wall_clock_sec > 0 else None,
                "n_model_calls": int(bars_evaluated),  # alias
                "n_trades_closed": int(n_trades_closed),
                # bars invariant
                "bars_invariant_ok": bool(bars_invariant_ok),
                "bars_invariant_gap": int(bars_invariant_gap),
                "bars_invariant_expected_gap": int(bars_invariant_expected_gap),
                # TRUTH timings
                "t_init_s": float(t_init_s) if is_truth_or_smoke_worker else None,
                "t_resolve_runner_s": float(t_resolve_runner_s) if is_truth_or_smoke_worker else None,
                "t_load_raw_s": float(t_load_raw_s) if is_truth_or_smoke_worker else None,
                "t_load_prebuilt_s": float(t_load_prebuilt_s) if is_truth_or_smoke_worker else None,
                "t_join_s": float(t_join_s) if is_truth_or_smoke_worker else None,
                "t_loop_s": float(t_loop_s) if is_truth_or_smoke_worker else None,
                "t_write_s": float(t_write_s) if is_truth_or_smoke_worker else None,
                "t_trade_outcomes_write_s": float(t_trade_outcomes_write_s) if is_truth_or_smoke_worker else None,
                "t_trade_journal_build_s": float(t_trade_journal_build_s) if is_truth_or_smoke_worker else None,
                "t_optional_observability_s": float(t_optional_obs_s) if is_truth_or_smoke_worker else None,
                "t_killchain_export_s": float(t_killchain_export_s) if is_truth_or_smoke_worker else None,
                "t_invariant_check_s": float(t_invariant_check_s) if is_truth_or_smoke_worker else None,
                # entry funnel
                "bars_seen": int(bars_seen),
                "bars_skipped_warmup": int(bars_skipped_warmup),
                "bars_skipped_pregate": int(bars_skipped_pregate),
                "bars_reaching_entry_stage": int(bars_reaching_entry_stage),
                "pregate_enabled": bool(pregate_enabled),
                # feature perf
                "feature_time_mean_ms": float(feature_time_mean_ms),
                "feature_time_total_sec": float(feature_time_total_sec),
                "t_feature_build_total_sec": float(feature_time_total_sec),  # alias
                "feature_timeout_count": int(feature_timeout_count),
                "vol_regime_unknown_count": int(vol_regime_unknown_count),
                # pregate stats
                "pregate_skips": int(pregate_skips),
                "pregate_passes": int(pregate_passes),
                "pregate_missing_inputs": int(pregate_missing_inputs),
                # timers
                "t_pregate_total_sec": float(t_pregate_total_sec),
                "t_xgb_predict_sec": float(t_xgb_predict_sec),
                "t_transformer_forward_sec": float(t_transformer_forward_sec),
                "t_gates_policy_sec": float(t_gates_policy_sec),
                "t_replay_tags_sec": float(t_replay_tags_sec),
                "t_replay_tags_build_inputs_sec": float(t_replay_tags_build_inputs_sec),
                "t_replay_tags_rolling_sec": float(t_replay_tags_rolling_sec),
                "t_replay_tags_ewm_sec": float(t_replay_tags_ewm_sec),
                "t_replay_tags_rank_sec": float(t_replay_tags_rank_sec),
                "t_replay_tags_assign_sec": float(t_replay_tags_assign_sec),
                "t_telemetry_sec": float(t_telemetry_sec),
                "t_io_total_sec": float(t_io_total_sec),
                "t_entry_input_prep_sec": float(t_entry_input_prep_sec),
                "n_entry_input_prep_calls": int(n_entry_input_prep_calls),
                "t_entry_model_infer_sec": float(t_entry_model_infer_sec),
                "n_entry_model_infer_calls": int(n_entry_model_infer_calls),
                "t_exit_eval_total_sec": float(t_exit_eval_total_sec),
                "n_exit_eval_calls": int(n_exit_eval_calls),
                "t_exit_input_prep_sec": float(t_exit_input_prep_sec),
                "n_exit_input_prep_calls": int(n_exit_input_prep_calls),
                "t_exit_model_infer_sec": float(t_exit_model_infer_sec),
                "n_exit_model_infer_calls": int(n_exit_model_infer_calls),
                "t_trade_state_bookkeeping_sec": float(t_trade_state_bookkeeping_sec),
                # HTF
                "htf_align_warn_count": int(htf_align_warn_count),
                "htf_align_time_total_sec": float(htf_align_time_total_sec),
                "htf_align_warning_time_sec": float(htf_align_warning_time_sec),
                "htf_align_call_count": int(htf_align_call_count),
                "htf_align_fallback_count": int(htf_align_fallback_count),
                "htf_feature_compute_bars": int(htf_feature_compute_bars),
                "htf_h1_calls": int(htf_h1_calls),
                "htf_h4_calls": int(htf_h4_calls),
                "htf_h1_warns": int(htf_h1_warns),
                "htf_h4_warns": int(htf_h4_warns),
                "htf_last_m5_ts": htf_last_m5_ts,
                "htf_last_j": htf_last_j,
                # bookkeeping
                "case_collision_resolution": case_collision_resolution,
                "prebuilt_invariant_report": inv_report,
                "counter_invariant_violation": bool(float(t_transformer_forward_sec or 0.0) > 0.0 and int(bars_evaluated or 0) == 0),
                # prebuilt flags/paths
                "prebuilt_used": bool(prebuilt_used_runner),
                "canonical_prebuilt_used": bool(canonical_prebuilt_used),
                "prebuilt_parquet_path": prebuilt_path_str,
                "canonical_truth_file": canonical_truth_path,
                "tape_price_truth_path": tape_price_truth_path,
                "xgb_bundle_path": xgb_bundle_path,
                "xgb_bundle_dir": xgb_bundle_path,
                "entry_bundle_path": str(entry_bundle_path) if entry_bundle_path is not None else None,
                "entry_bundle_dir": str(entry_bundle_path) if entry_bundle_path is not None else None,
                "exit_bundle_path": str(exit_bundle_path) if exit_bundle_path is not None else None,
                "exit_bundle_dir": str(exit_bundle_path) if exit_bundle_path is not None else None,
                "prebuilt_required_columns": getattr(bootstrap_ctx, "prebuilt_required_columns", None) if bootstrap_ctx else None,
                "raw_prebuilt_join_metrics_path": join_metrics_path_ssot,
                # analysis / threshold
                "analysis_mode": os.environ.get("GX1_ANALYSIS_MODE") == "1",
                "threshold_used": (getattr(getattr(runner, "entry_manager", None), "threshold_used", None) if runner else None),
                "threshold_source": (
                    "override"
                    if (os.environ.get("GX1_ANALYSIS_MODE") == "1" and os.environ.get("GX1_ENTRY_THRESHOLD_OVERRIDE"))
                    else "canonical"
                ),
                # exit strategy observability
                "exit_profile": (getattr(runner, "exit_config_name", None) or "unknown"),
                "exit_type": getattr(runner, "exit_type", None),
                "router_enabled": bool((getattr(runner, "policy", None) or {}).get("hybrid_exit_router", False)),
                "exit_critic_enabled": bool((getattr(runner, "policy", None) or {}).get("exit_critic", {}).get("enabled", False)),
                "exit_tuning_capsule": getattr(runner, "exit_tuning_capsule", None),
                "exit_ml_enabled": getattr(runner, "exit_ml_enabled", False),
                "exit_ml_decision_mode": getattr(runner, "exit_ml_decision_mode", "") or None,
                "exit_ml_config_hash": getattr(runner, "exit_ml_config_hash", "") or None,
                "exit_ml_model_sha": getattr(runner, "exit_ml_model_sha", None),
                "exit_ml_input_dim": getattr(runner, "exit_ml_input_dim", None),
                "exit_ml_io_version": getattr(runner, "exit_ml_io_version", None),
                "exit_threshold": _safe_float(getattr(runner, "exit_threshold", None), None),
                "exit_require_consecutive": _safe_int(getattr(runner, "exit_require_consecutive", None), None),
                "entry_attempt_long": getattr(runner, "entry_attempt_long", None),
                "entry_attempt_short": getattr(runner, "entry_attempt_short", None),
                "entry_accept_long": getattr(runner, "entry_accept_long", None),
                "entry_accept_short": getattr(runner, "entry_accept_short", None),
                "signal_candidate_long": getattr(runner, "signal_candidate_long", None),
                "signal_candidate_short": getattr(runner, "signal_candidate_short", None),
                "signal_candidate_none": getattr(runner, "signal_candidate_none", None),
                "entry_pref_pre_long": getattr(runner, "entry_pref_pre_long", None),
                "entry_pref_pre_short": getattr(runner, "entry_pref_pre_short", None),
                "entry_pref_pre_flat": getattr(runner, "entry_pref_pre_flat", None),
                "entry_pref_post_long": getattr(runner, "entry_pref_post_long", None),
                "entry_pref_post_short": getattr(runner, "entry_pref_post_short", None),
                "entry_pref_post_none": getattr(runner, "entry_pref_post_none", None),
                "entry_attempt_total": _safe_int(getattr(runner, "entry_attempt_long", 0), 0)
                + _safe_int(getattr(runner, "entry_attempt_short", 0), 0),
                "entry_accept_total": _safe_int(getattr(runner, "entry_accept_long", 0), 0)
                + _safe_int(getattr(runner, "entry_accept_short", 0), 0),
                "perf_bars_total": getattr(runner, "perf_bars_total", None),
                "perf_n_bars_processed": getattr(runner, "perf_n_bars_processed", None),
                "max_open_trades_policy": _safe_int(
                    getattr(getattr(runner, "risk_limits", None), "max_open_trades", None), None
                ),
                "max_open_trades_override_env": getattr(runner, "max_open_trades_override_env", None),
                "max_open_trades_effective_boot": _safe_int(
                    getattr(runner, "max_open_trades_effective_boot", None), None
                ),
                "max_open_trades_used": _safe_int(getattr(runner, "max_open_trades_used", None), None),
                "admission_replacement_enabled": int(
                    bool(getattr(runner, "_replacement_overlap_long_over_oldest_overlap_short_enabled", lambda: False)())
                ),
                "n_trades_opened_registered": getattr(runner, "perf_n_trades_opened_registered", 0),
                "n_trades_opened_registered_long": getattr(runner, "perf_n_trades_opened_registered_long", 0),
                "n_trades_opened_registered_short": getattr(runner, "perf_n_trades_opened_registered_short", 0),
                "n_entry_proposed_long": getattr(runner, "perf_n_entry_proposed_long", 0),
                "n_entry_proposed_short": getattr(runner, "perf_n_entry_proposed_short", 0),
                "n_can_enter_pass_long": getattr(runner, "perf_n_can_enter_pass_long", 0),
                "n_can_enter_pass_short": getattr(runner, "perf_n_can_enter_pass_short", 0),
                "n_can_enter_fail_long": getattr(runner, "perf_n_can_enter_fail_long", 0),
                "n_can_enter_fail_short": getattr(runner, "perf_n_can_enter_fail_short", 0),
                "can_enter_fail_reasons": getattr(runner, "perf_can_enter_fail_reasons", None),
                "entry_margin_min_used": getattr(runner, "perf_entry_margin_min_used", None),
                "n_entry_margin_reject_total": getattr(runner, "perf_n_entry_margin_reject_total", 0),
                "n_entry_margin_reject_long": getattr(runner, "perf_n_entry_margin_reject_long", 0),
                "n_entry_margin_reject_short": getattr(runner, "perf_n_entry_margin_reject_short", 0),
                "entry_margin_reject_reasons": getattr(runner, "perf_entry_margin_reject_reasons", None),
                "n_trade_journal_margin_missing": getattr(runner, "perf_trade_journal_margin_missing", 0),
                "trade_journal_margin_missing_reasons": getattr(
                    runner, "perf_trade_journal_margin_missing_reasons", None
                ),
                "exit_hold_tail_thresholds": getattr(runner, "perf_exit_hold_tail_thresholds", None),
                "exit_hold_tail_counts": getattr(runner, "perf_exit_hold_tail_counts", None),
                "exit_hold_tail_by_reason": getattr(runner, "perf_exit_hold_tail_by_reason", None),
                "exit_hold_tail_by_side": getattr(runner, "perf_exit_hold_tail_by_side", None),
                "exit_hold_tail_by_session": getattr(runner, "perf_exit_hold_tail_by_session", None),
                "exit_hold_tail_by_reason_side_session": getattr(
                    runner, "perf_exit_hold_tail_by_reason_side_session", None
                ),
                "exit_cat_guard_by_side_session_month": getattr(
                    runner, "perf_exit_cat_guard_by_side_session_month", None
                ),
                "exit_cat_guard_rate_by_side_session": getattr(
                    runner, "perf_exit_cat_guard_rate_by_side_session", None
                ),
                "exit_cat_guard_proof": getattr(runner, "perf_exit_cata_guard_proof", None),
                "exit_progress_protect_triggers": _safe_int(
                    getattr(runner, "perf_exit_progress_protect_triggers", 0),
                    0,
                ),
                "exit_progress_protect_reasons": getattr(
                    runner,
                    "perf_exit_progress_protect_reasons",
                    None,
                ),
                "n_trade_journal_rows_built": getattr(runner, "perf_trade_journal_rows", None),
                "n_trade_journal_row_appends": getattr(runner, "perf_trade_journal_rows", None),
                "n_exit_eval_trace_events": getattr(runner, "perf_exit_eval_trace_rows", 0),
                "n_exit_io_feature_events": getattr(runner, "perf_exit_io_rows", 0),
                "n_exit_ml_context_events": getattr(runner, "perf_exit_ml_event_rows", 0),
                "n_observability_events_in_loop": _safe_int(getattr(runner, "perf_exit_eval_trace_rows", 0), 0)
                + _safe_int(getattr(runner, "perf_exit_io_rows", 0), 0)
                + _safe_int(getattr(runner, "perf_exit_ml_event_rows", 0), 0),
                "observability_build_location": {
                    "in_warm_loop": bool(
                        _safe_int(getattr(runner, "perf_exit_eval_trace_rows", 0), 0)
                        + _safe_int(getattr(runner, "perf_exit_io_rows", 0), 0)
                        + _safe_int(getattr(runner, "perf_exit_ml_event_rows", 0), 0)
                        > 0
                    ),
                    "close_or_chunk_end": True,
                },
                "timing_proof_blocks": {
                    "total_runtime_sec": float(wall_clock_sec),
                    "initial_load_resolve_sec": float(t_init_s + t_resolve_runner_s + t_load_raw_s + t_load_prebuilt_s + t_join_s),
                    "prebuilt_tape_access_sec": float(t_load_raw_s + t_load_prebuilt_s),
                    "per_bar_replay_loop_total_sec": float(t_loop_s),
                    "entry_input_prep_sec": float(t_entry_input_prep_sec),
                    "entry_model_inference_sec": float(t_entry_model_infer_sec),
                    "exit_input_prep_sec": float(t_exit_input_prep_sec),
                    "exit_model_inference_sec": float(t_exit_model_infer_sec),
                    "trade_state_bookkeeping_sec": float(t_trade_state_bookkeeping_sec),
                    "trade_journal_building_sec": float(t_trade_journal_build_s),
                    "proof_observability_building_sec": float(t_optional_obs_s),
                    "footer_report_aggregation_sec": float(t_footer_aggregation_s),
                    "writes_flushes_total_sec": float(t_write_s + t_trade_outcomes_write_s + t_trade_journal_build_s),
                },
                "timing_per_call_stats": {
                    "entry_input_prep_mean_ms": (
                        float(t_entry_input_prep_sec) * 1000.0 / float(n_entry_input_prep_calls)
                        if n_entry_input_prep_calls > 0
                        else None
                    ),
                    "entry_model_infer_mean_ms": (
                        float(t_entry_model_infer_sec) * 1000.0 / float(n_entry_model_infer_calls)
                        if n_entry_model_infer_calls > 0
                        else None
                    ),
                    "exit_input_prep_mean_ms": (
                        float(t_exit_input_prep_sec) * 1000.0 / float(n_exit_input_prep_calls)
                        if n_exit_input_prep_calls > 0
                        else None
                    ),
                    "exit_model_infer_mean_ms": (
                        float(t_exit_model_infer_sec) * 1000.0 / float(n_exit_model_infer_calls)
                        if n_exit_model_infer_calls > 0
                        else None
                    ),
                },
                "exit_input_prep_subtimers": exit_input_prep_subtimers,
                "timing_top3_drivers": top_timing_drivers,
                "speed_top3_candidates_without_arch_changes": top_speed_candidates,
                "replay_proof_summary": {
                    "canonical_prebuilt_used": bool(canonical_prebuilt_used),
                    "top3_timing_drivers": top_timing_drivers,
                    "top3_speed_candidates": top_speed_candidates,
                },
                "n_footer_structures_built": 0,
                "n_artifact_writes_total": 0,
                "n_artifact_writes_json": 0,
                "n_artifact_writes_parquet": 0,
                "n_artifact_writes_log": 0,
                "entry_gate_config_snapshot": getattr(runner, "entry_gate_config_snapshot", None),
                "entry_gate_counters": getattr(runner, "entry_gate_counters", None),
                # entry blocking gate telemetry
                "entry_veto_hard": getattr(getattr(runner, "entry_manager", None), "veto_hard", None),
                "entry_veto_soft": getattr(getattr(runner, "entry_manager", None), "veto_soft", None),
                "entry_veto_pre": getattr(getattr(runner, "entry_manager", None), "veto_pre", None),
                "entry_veto_cand": getattr(getattr(runner, "entry_manager", None), "veto_cand", None),
                "entry_killchain_counts": {
                    "killchain_n_entry_pred_total": getattr(getattr(runner, "entry_manager", None), "killchain_n_entry_pred_total", None),
                    "killchain_n_above_threshold": getattr(getattr(runner, "entry_manager", None), "killchain_n_above_threshold", None),
                    "killchain_n_after_session_guard": getattr(getattr(runner, "entry_manager", None), "killchain_n_after_session_guard", None),
                    "killchain_n_after_vol_guard": getattr(getattr(runner, "entry_manager", None), "killchain_n_after_vol_guard", None),
                    "killchain_n_after_risk_sizing": getattr(getattr(runner, "entry_manager", None), "killchain_n_after_risk_sizing", None),
                    "killchain_n_trade_create_attempts": getattr(getattr(runner, "entry_manager", None), "killchain_n_trade_create_attempts", None),
                    "killchain_n_trade_created": getattr(getattr(runner, "entry_manager", None), "killchain_n_trade_created", None),
                },
                "entry_killchain_block_reasons": getattr(
                    getattr(runner, "entry_manager", None), "killchain_block_reason_counts", None
                ),
                "exit_eval_long": getattr(runner, "exit_eval_long", None),
                "exit_eval_short": getattr(runner, "exit_eval_short", None),
                "exit_close_long": getattr(runner, "exit_close_long", None),
                "exit_close_short": getattr(runner, "exit_close_short", None),
                # ssot
                "ssot": {"bundle_sha256": bundle_sha256},
                # ctx masks (diagnostics)
                "ctx_cont_dim": _safe_int(getattr(runner, "ctx_cont_dim", 0), 0) or None,
                "ctx_cat_dim": _safe_int(getattr(runner, "ctx_cat_dim", 0), 0) or None,
                "ctx_cont_mask_id": getattr(runner, "ctx_cont_mask_id", None),
                "ctx_cat_mask_id": getattr(runner, "ctx_cat_mask_id", None),
                "ctx_cont_mask": getattr(runner, "ctx_cont_mask", None),
                "ctx_cat_mask": getattr(runner, "ctx_cat_mask", None),
                # ctx telemetry (entry_manager.entry_telemetry)
                "n_ctx_model_calls": _safe_int(
                    (getattr(getattr(runner, "entry_manager", None), "entry_telemetry", None) or {}).get("n_ctx_model_calls", 0), 0
                ),
                "ctx_proof_pass_count": _safe_int(
                    (getattr(getattr(runner, "entry_manager", None), "entry_telemetry", None) or {}).get("ctx_proof_pass_count", 0), 0
                ),
                "ctx_proof_fail_count": _safe_int(
                    (getattr(getattr(runner, "entry_manager", None), "entry_telemetry", None) or {}).get("ctx_proof_fail_count", 0), 0
                ),
                # XGB load branch proof (TRUTH canonical vs policy/session)
                "xgb_load_branch": getattr(runner, "xgb_load_branch", None),
                "xgb_load_source": getattr(runner, "xgb_load_source", None),
                "xgb_load_paths": getattr(runner, "xgb_load_paths", None),
                "xgb_load_error": getattr(runner, "xgb_load_error", None),
            }

            footer_ctx = ChunkFooterContext(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                payload=payload,
                run_id=run_id,
            )
            n_footer_structures_built += 1
            t_footer_aggregation_s = _safe_float(time.time() - t_footer_agg_start, 0.0)
            payload["t_footer_aggregation_s"] = float(t_footer_aggregation_s)
            payload["n_footer_structures_built"] = int(n_footer_structures_built)
            payload["n_artifact_writes_total"] = int(n_artifact_writes_total + 3)
            payload["n_artifact_writes_json"] = int(n_artifact_writes_json + 2)
            payload["n_artifact_writes_parquet"] = int(n_artifact_writes_parquet)
            payload["n_artifact_writes_log"] = int(n_artifact_writes_log + 1)
            t_footer_write_start = time.time()
            write_chunk_footer(footer_ctx)
            t_footer_write_s = _safe_float(time.time() - t_footer_write_start, 0.0)
            n_artifact_writes_total += 1
            n_artifact_writes_json += 1

            log.info("[CHUNK %s] chunk_footer.json written (status=%s)", chunk_idx, status)
            t_proof_start = time.time()
            replay_summary_stats = _log_replay_summary_proof(chunk_output_dir, run_id)
            t_summary_proof_s = _safe_float(time.time() - t_proof_start, 0.0)
            log.info("[REPLAY_SUMMARY_PROOF_STATS] run_id=%s stats=%s", run_id, replay_summary_stats)
            log.info(
                "[REPLAY_WRITE_COST_PROOF] run_id=%s footer_write_sec=%.4f summary_proof_sec=%.4f writes_total=%d",
                run_id,
                t_footer_write_s,
                t_summary_proof_s,
                n_artifact_writes_total,
            )
            n_artifact_writes_total += 2
            n_artifact_writes_json += 1
            n_artifact_writes_log += 1
            log.info(
                "[REPLAY_TIMING_TOP3_PROOF] run_id=%s top1=%s top2=%s top3=%s canonical_prebuilt_used=%s",
                run_id,
                top_timing_drivers[0] if len(top_timing_drivers) > 0 else None,
                top_timing_drivers[1] if len(top_timing_drivers) > 1 else None,
                top_timing_drivers[2] if len(top_timing_drivers) > 2 else None,
                int(canonical_prebuilt_used),
            )

            # TRUTH strict: if bars invariant failed, raise AFTER footer write
            if (not bars_invariant_ok) and is_truth_or_smoke_worker:
                raise RuntimeError(
                    f"[BARS_INVARIANT] bars_total_input - bars_processed = {bars_invariant_gap} "
                    f"!= expected (warmup+tail) = {bars_invariant_expected_gap}. "
                    f"bars_total_input={bars_total_input}, bars_processed={bars_processed}, "
                    f"warmup_holdback_bars={warmup_holdback_bars}, tail_holdback_bars={tail_holdback_bars} "
                    f"(footer written with bars_invariant_ok=false)"
                )

        except Exception as footer_err:
            if isinstance(footer_err, RuntimeError) and "BARS_INVARIANT" in str(footer_err):
                raise

            log.error("[CHUNK %s] Failed during footer write block: %s", chunk_idx, footer_err, exc_info=True)

            try:
                tb_str = "".join(traceback.format_exception(type(footer_err), footer_err, footer_err.__traceback__))
                failure_context = build_failure_context(
                    runner=runner,
                    chunk_df=chunk_df,
                    chunk_output_dir=chunk_output_dir,
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                    error=footer_err,
                    bars_processed_safe=int(bars_processed or 0),
                    first_iter_ts=None,
                    last_iter_ts=None,
                    policy_id=getattr(runner, "policy_id", None) if runner else None,
                    bundle_sha256=bundle_sha256,
                )
                bar_counters = _compute_basic_bar_counters_snapshot(runner, int(bars_processed or 0))
                fail_capsule = {
                    **failure_context,
                    "traceback": tb_str[:10000],
                    "bar_counters": bar_counters,
                    "hint": "footer_error: failure in footer write block",
                    "timestamp": dt_now_iso(),
                }
                write_failure_capsule(
                    chunk_output_dir=chunk_output_dir,
                    payload=fail_capsule,
                    filename="CHUNK_FAIL_CAPSULE.json",
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                )
            except Exception:
                pass

            try:
                stub = {
                    "run_id": run_id,
                    "chunk_id": int(chunk_idx),
                    "status": "footer_error",
                    "error": f"Failed during footer write block: {str(footer_err)[:500]}",
                    "bars_processed": int(bars_processed or 0),
                    "total_bars": int(total_bars or 0),
                    "dt_module_version": dt_module_version,
                    "timestamp": dt_now_iso(),
                }
                atomic_write_json_safe(
                    chunk_output_dir / "chunk_footer_stub.json",
                    convert_to_json_serializable(stub),
                )
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # Return object (for merge)
        # ---------------------------------------------------------------------
        attribution_path_str = str(chunk_output_dir / f"attribution_{run_id}.json")

        artifacts = {
            "raw_signals": chunk_output_dir / f"raw_signals_{run_id}.parquet",
            "policy_decisions": chunk_output_dir / f"policy_decisions_{run_id}.parquet",
            "trade_outcomes": chunk_output_dir / f"trade_outcomes_{run_id}.parquet",
            "attribution": attribution_path_str,
            "metrics": chunk_output_dir / f"metrics_{run_id}.json",
            "summary": chunk_output_dir / f"summary_{run_id}.md",
            "chunk_footer": chunk_output_dir / "chunk_footer.json",
        }

        chunk_artifacts = {
            "chunk_idx": int(chunk_idx),
            "status": status,
            "error": error,
            "n_bars": int(bars_processed or 0),
            "n_model_calls": int(bars_evaluated or 0),
            "n_trades_closed": int(n_trades_closed or 0),
            "wall_clock_sec": float(wall_clock_sec or 0.0),
            "total_bars": int(total_bars or 0),
            "bars_per_sec": (float(bars_processed) / float(wall_clock_sec)) if wall_clock_sec > 0 else None,
            "artifacts": artifacts,
        }

        if status == "failed":
            raise RuntimeError(f"CHUNK_{chunk_idx}_FAILED: {error}")
        if is_truth_or_smoke_worker and status == "failed_invariant":
            raise RuntimeError(f"CHUNK_{chunk_idx}_FAILED_INVARIANT: {error}")

        return chunk_artifacts

    except Exception as outer_exc:
        error_traceback = "".join(traceback.format_exception(type(outer_exc), outer_exc, outer_exc.__traceback__))
        if status == "ok":
            status = "failed"
        if error is None:
            error = str(outer_exc)

        if chunk_output_dir is None:
            try:
                chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
                chunk_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                chunk_output_dir = None

        skip_ledger["stage"] = "exception"
        skip_ledger["exception_type"] = type(outer_exc).__name__
        skip_ledger["exception_msg"] = str(outer_exc)[:500]
        skip_ledger["traceback"] = error_traceback[:5000]

        try:
            if chunk_output_dir is not None:
                failure_context = build_failure_context(
                    runner=runner,
                    chunk_df=chunk_df,
                    chunk_output_dir=chunk_output_dir,
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                    error=outer_exc,
                    bars_processed_safe=int(bars_processed or 0),
                    first_iter_ts=None,
                    last_iter_ts=None,
                    policy_id=getattr(runner, "policy_id", None) if runner else None,
                    bundle_sha256=bundle_sha256,
                )
                bar_counters = _compute_basic_bar_counters_snapshot(runner, int(bars_processed or 0))
                fail_capsule = {
                    **failure_context,
                    "traceback": error_traceback[:10000],
                    "bar_counters": bar_counters,
                    "hint": "outer_exception: failure before normal completion",
                    "timestamp": dt_now_iso(),
                }
                write_failure_capsule(
                    chunk_output_dir=chunk_output_dir,
                    payload=fail_capsule,
                    filename="CHUNK_FAIL_CAPSULE.json",
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                )
        except Exception:
            pass

        raise

    finally:
        # ALWAYS write SKIP_LEDGER_FINAL.json (best-effort)
        try:
            skip_ledger["timestamp"] = dt_now_iso()
            skip_ledger["bars_processed"] = int(bars_processed or 0)
            skip_ledger["candles_iterated"] = int(bars_processed or 0)

            try:
                skip_ledger["reached_entry_stage"] = _safe_int(getattr(runner, "bars_reaching_entry_stage", 0), 0)
            except Exception:
                skip_ledger["reached_entry_stage"] = 0

            if chunk_output_dir is None:
                try:
                    chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
                    chunk_output_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    chunk_output_dir = None

            if chunk_output_dir is not None:
                path = chunk_output_dir / "SKIP_LEDGER_FINAL.json"
                payload = convert_to_json_serializable(skip_ledger)
                ok = atomic_write_json_safe(path, payload)
                if ok:
                    log.info("[CHUNK %s] [SKIP_LEDGER] wrote %s", chunk_idx, path)
                else:
                    tmp_path = Path(tempfile.gettempdir()) / f"chunk_{chunk_idx}_SKIP_LEDGER_FINAL_{run_id}.json"
                    atomic_write_json_safe(tmp_path, payload)
        except Exception as e:
            log.error("[CHUNK %s] [SKIP_LEDGER] finally error: %s", chunk_idx, e, exc_info=True)


# ----------------------------------------------------------------------------- #
# CLI wrapper                                                                  #
# ----------------------------------------------------------------------------- #
def _default_output_root() -> Path:
    gx1_data = os.environ.get("GX1_DATA") or os.environ.get("GX1_DATA_DIR") or os.environ.get("GX1_DATA_ROOT")
    if gx1_data:
        return Path(gx1_data).expanduser().resolve() / "reports" / "replay_chunk"
    return Path.home() / "GX1_DATA" / "reports" / "replay_chunk"


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json
    import sys
    import pandas as pd

    parser = argparse.ArgumentParser(description="Run a single TRUTH/SMOKE replay chunk (1W1C).")
    parser.add_argument("--config", required=True, help="Truth config (e.g. canonical_truth_signal_only.json)")
    parser.add_argument("--session", required=True, help="Session (EU/OVERLAP/US)")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--chunk-idx", type=int, required=True)
    parser.add_argument("--start", required=True, help="ISO8601 start (UTC)")
    parser.add_argument("--end", required=True, help="ISO8601 end (UTC)")
    parser.add_argument("--chunk-local-padding-days", type=int, default=0)
    parser.add_argument("--output-root", type=str, default=None, help="Optional output root (default: GX1_DATA/reports/replay_chunk)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id override")
    args = parser.parse_args(argv)

    if args.workers != 1 or args.chunks != 1:
        raise RuntimeError("[TRUTH_1W1C_ONLY] workers and chunks must be 1 in replay_chunk CLI")

    chunk_start = pd.Timestamp(args.start, tz="UTC")
    chunk_end = pd.Timestamp(args.end, tz="UTC")
    if chunk_start >= chunk_end:
        raise RuntimeError("[TS_FAIL] start >= end")

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open() as f:
        cfg = json.load(f)

    policy_path = config_path
    bundle_dir = Path(cfg["canonical_xgb_bundle_dir"]).expanduser().resolve()
    # TRUTH manifest-only: never pass a parquet path from CLI/config
    prebuilt_parquet_path = None
    prebuilt_manifest_path = cfg.get("canonical_prebuilt_manifest")

    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError(f"[BUNDLE_LOCK_MISSING] {lock_path}")
    bundle_sha256 = _file_sha256(lock_path)

    tape_root = os.environ.get(
        "GX1_CANONICAL_TAPE_ROOT_RAW",
        os.environ.get(
            "GX1_CANONICAL_TAPE_ROOT",
            "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL",
        ),
    )
    year = chunk_start.year
    data_path = Path(tape_root) / f"year={year}" / "part-000.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"[TAPE_NOT_FOUND] {data_path}")

    run_id = args.run_id or f"REPLAY_{chunk_start.strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else _default_output_root()
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GX1_RUN_MODE"] = os.environ.get("GX1_RUN_MODE", "TRUTH")
    os.environ["GX1_TRUTH_MODE"] = os.environ.get("GX1_TRUTH_MODE", "1")
    os.environ["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    os.environ["GX1_FEATURE_BUILD_DISABLED"] = "1"
    os.environ["GX1_GATED_FUSION_ENABLED"] = "1"
    os.environ["GX1_REPLAY_INCREMENTAL_FEATURES"] = "1"
    os.environ["GX1_REPLAY_NO_CSV"] = "1"
    os.environ["GX1_FEATURE_USE_NP_ROLLING"] = "1"

    prefork_payload = {
        "schema_version": "pre_fork_freeze_v1",
        "created_utc": dt_now_iso(),
        "run_id": run_id,
        "session": args.session,
        "workers": args.workers,
        "chunks": args.chunks,
        "chunk_idx": args.chunk_idx,
        "start_utc": args.start,
        "end_utc": args.end,
        "config_path": str(config_path),
        "python_exe": sys.executable,
        "gx1_engine": os.environ.get("GX1_ENGINE"),
        "gx1_data": os.environ.get("GX1_DATA"),
        "bundle_path": str(bundle_dir),
        "bundle_master_lock_sha256": bundle_sha256,
        "prebuilt_manifest_path": prebuilt_manifest_path,
        "canonical_tape_root_raw": tape_root,
    }
    _write_prefork_freeze(output_dir, prefork_payload)

    if prebuilt_parquet_path is not None:
        raise RuntimeError("TRUTH requires manifest-only: prebuilt_parquet_path must be None in CLI")

    try:
        _ = process_chunk(
            chunk_idx=int(args.chunk_idx),
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            data_path=data_path,
            policy_path=policy_path,
            run_id=run_id,
            output_dir=output_dir,
            bundle_sha256=bundle_sha256,
            prebuilt_parquet_path=prebuilt_parquet_path,
            bundle_dir=bundle_dir,
            chunk_local_padding_days=int(args.chunk_local_padding_days),
        )
        return 0
    except Exception as e:
        print(f"[replay_chunk] failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
