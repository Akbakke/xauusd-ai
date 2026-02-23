"""
Replay Chunk Runner: Kanonisk wrapper for "kjør én replay-chunk" (TRUTH-grade).

Dette er den kanoniske 1-chunk orchestratoren.

LOCKED GOALS:
- replay_chunk.py er en thin orchestrator (SSoT/flow, ikke “smart” IO)
- Alle writes utenfor orchestrator er atomic/best-effort helpers
- chunk_footer_writer er DUMB WRITER: tar payload dict, skriver chunk_footer.json atomisk
- TRUTH/SMOKE er strict: invariants kan flippe status til failed_invariant
- Ingen segmented/parallel/owner/preroll i TRUTH 1W1C

NOTES:
- Denne fila *kan* lese små ting fra runner i minnet (perf counters etc).
- Denne fila *skal ikke* bygge data/features på nytt (prebuilt-only i TRUTH).
"""

from __future__ import annotations

import logging
import os
import signal
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from gx1.utils.dt_module import now_iso as dt_now_iso

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
# Best-effort bar counters snapshot (used in error capsules)
# TRUTH path: no import from legacy (quarantined) replay scripts.
# -----------------------------------------------------------------------------
def compute_bar_counters_snapshot(runner, bars_processed, chunk_df):
    """Stub for error capsules; full counters live in legacy replay script."""
    return {
        "candles_iterated": int(bars_processed or 0),
        "reached_entry_stage": 0,
        "bars_passed_hard_eligibility": 0,
        "bars_blocked_hard_eligibility": 0,
        "bars_passed_soft_eligibility": 0,
        "bars_blocked_soft_eligibility": 0,
    }


# -----------------------------------------------------------------------------
# Failure helpers (robust, no disk reads required)
# -----------------------------------------------------------------------------
from gx1.execution.chunk_failure import (
    convert_to_json_serializable,
    atomic_write_json_safe,
    write_failure_capsule,
    build_failure_context,
    write_signal_event_capsule,
)

# -----------------------------------------------------------------------------
# Dumb footer writer (payload-only)
# -----------------------------------------------------------------------------
from gx1.execution.chunk_footer_writer import ChunkFooterContext, write_chunk_footer

# -----------------------------------------------------------------------------
# Bootstrap / data loader / exporters / invariants
# -----------------------------------------------------------------------------
from gx1.execution.chunk_bootstrap import bootstrap_chunk_environment, BootstrapContext
from gx1.execution.chunk_data_loader import load_chunk_data, DataContext
from gx1.execution.killchain_export import KillchainExportContext, export_killchain
from gx1.execution.prebuilt_invariants import PrebuiltInvariantContext, check_prebuilt_invariants
from gx1.utils.empty_trade_outcomes import (
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


def _write_minimal_attribution(chunk_output_dir: Path, run_id: str) -> None:
    """Write minimal attribution JSON in TRUTH path when legacy replay_eval_gated is quarantined."""
    path = chunk_output_dir / f"attribution_{run_id}.json"
    data = {
        "mode": "truth_minimal",
        "run_id": run_id,
        "chunks": 1,
        "note": "Legacy replay_eval_gated quarantined; minimal attribution written by TRUTH path.",
    }
    atomic_write_json_safe(path, convert_to_json_serializable(data))


def _write_trade_outcomes_truth(chunk_output_dir: Path, run_id: str, runner: Any) -> None:
    """
    TRUTH-native writer: always write chunk-level trade_outcomes_{run_id}.parquet.
    No import of gx1.scripts.replay_eval_gated. Uses runner.replay_eval_collectors
    if present; otherwise writes empty parquet with canonical schema.
    Atomic write via tmp + os.replace.
    """
    out_path = chunk_output_dir / f"trade_outcomes_{run_id}.parquet"
    collector = None
    if runner and getattr(runner, "replay_eval_collectors", None):
        collector = runner.replay_eval_collectors.get("trade_outcomes")
    rows = list(getattr(collector, "outcomes", [])) if collector else []

    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", prefix="trade_outcomes_", dir=str(chunk_output_dir))
    try:
        os.close(fd)
        tmp = Path(tmp_path)
        if rows:
            df = pd.DataFrame(rows)
            # Ensure column order matches contract; only include existing columns
            cols = [c for c in TRADE_OUTCOMES_REQUIRED_COLUMNS if c in df.columns]
            extra = [c for c in df.columns if c not in TRADE_OUTCOMES_REQUIRED_COLUMNS]
            df = df[cols + extra] if cols else df
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
) -> Dict[str, Any]:
    """
    Process a single replay chunk.

    Returns:
        chunk_artifacts: dict with status + artifact paths
    """
    # -------------------------------------------------------------------------
    # Hard-gates (TRUTH/SMOKE 1W1C)
    # -------------------------------------------------------------------------
    _assert_truth_ban_envs()
    is_truth_or_smoke_worker = _is_truth_or_smoke()

    # -------------------------------------------------------------------------
    # Always-initialized locals (so finally/except always has something sane)
    # -------------------------------------------------------------------------
    status: str = "ok"
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    chunk_output_dir: Optional[Path] = None
    runner: Any = None
    chunk_df: Optional[pd.DataFrame] = None

    # Perf / counters (best-effort)
    bars_processed = 0              # bars iterated by replay loop
    total_bars = 0                  # input bars available to loop (post-join)
    bars_evaluated = 0              # bars where model/policy was called (after warmup)
    warmup_holdback_bars = 0
    tail_holdback_bars = 0          # reserved for deliberate tail skipping (default 0)
    n_trades_closed = 0
    wall_clock_sec = 0.0

    # Feature perf (best-effort)
    feature_time_mean_ms = 0.0
    feature_time_total_sec = 0.0

    # TRUTH timing breakdown (best-effort)
    t_init_s = 0.0
    t_load_raw_s = 0.0
    t_load_prebuilt_s = 0.0
    t_join_s = 0.0
    t_loop_s = 0.0
    t_write_s = 0.0

    # extra telemetry (best-effort)
    dt_module_version: Optional[str] = None
    telemetry_required = False
    worker_start_time = time.time()

    # invariants/export reports
    inv_report: Optional[Dict[str, Any]] = None

    # data loader outputs
    prebuilt_parquet_path_resolved: Optional[str] = None
    chunk_data_path_abs: Optional[Path] = None
    case_collision_resolution: Optional[Dict[str, Any]] = None
    actual_chunk_start: Optional[pd.Timestamp] = None

    # entry funnel counters (best-effort)
    bars_seen = 0
    bars_skipped_warmup = 0
    bars_skipped_pregate = 0
    bars_reaching_entry_stage = 0
    pregate_enabled = False

    # pregate / regimes
    pregate_skips = 0
    pregate_passes = 0
    pregate_missing_inputs = 0
    vol_regime_unknown_count = 0
    feature_timeout_count = 0

    # phase timers (best-effort)
    t_pregate_total_sec = 0.0
    t_xgb_predict_sec = 0.0
    t_transformer_forward_sec = 0.0
    t_gates_policy_sec = 0.0
    t_replay_tags_sec = 0.0
    t_telemetry_sec = 0.0
    t_replay_tags_build_inputs_sec = 0.0
    t_replay_tags_rolling_sec = 0.0
    t_replay_tags_ewm_sec = 0.0
    t_replay_tags_rank_sec = 0.0
    t_replay_tags_assign_sec = 0.0
    t_io_total_sec = 0.0

    # HTF stats (best-effort)
    htf_align_warn_count = 0
    htf_align_time_total_sec = 0.0
    htf_align_warning_time_sec = 0.0
    htf_align_call_count = 0
    htf_align_fallback_count = 0
    htf_feature_compute_bars = 0
    htf_h1_calls = 0
    htf_h4_calls = 0
    htf_h1_warns = 0
    htf_h4_warns = 0
    htf_last_m5_ts = None
    htf_last_j = None

    # stable env flags
    prebuilt_enabled_env = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"

    # -------------------------------------------------------------------------
    # Always-on SKIP_LEDGER (written in finally)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Install SIGTERM handler
    # -------------------------------------------------------------------------
    global STOP_REQUESTED
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except Exception:
        pass
    STOP_REQUESTED = False
    os.environ["GX1_STOP_REQUESTED"] = "0"

    # -------------------------------------------------------------------------
    # Main try
    # -------------------------------------------------------------------------
    try:
        # ---------------------------------------------------------------------
        # PHASE 0: Bootstrap
        # ---------------------------------------------------------------------
        t0 = time.time()
        bootstrap_ctx: BootstrapContext = bootstrap_chunk_environment(
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
        is_truth_or_smoke_worker = _safe_bool(getattr(bootstrap_ctx, "is_truth_or_smoke_worker", is_truth_or_smoke_worker))
        telemetry_required = _safe_bool(getattr(bootstrap_ctx, "telemetry_required", False))
        dt_module_version = getattr(bootstrap_ctx, "dt_module_version", None)
        worker_start_time = _safe_float(getattr(bootstrap_ctx, "worker_start_time", worker_start_time), worker_start_time)

        t_init_s = _safe_float(getattr(bootstrap_ctx, "t_init_s", time.time() - t0), time.time() - t0)

        if chunk_output_dir is None:
            raise RuntimeError("[BOOTSTRAP_FAIL] chunk_output_dir is None after bootstrap")

        # TRUTH hard fail: bundle_sha256 must exist
        if not bundle_sha256:
            raise RuntimeError("[SSOT_FAIL] bundle_sha256 missing in process_chunk()")

        # ---------------------------------------------------------------------
        # PHASE 1: Create runner (worker-local import)
        # ---------------------------------------------------------------------
        from gx1.execution.oanda_demo_runner import GX1DemoRunner

        canonical_bundle_dir = os.getenv("GX1_CANONICAL_BUNDLE_DIR")
        if canonical_bundle_dir:
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
        runner.run_id = run_id
        runner.chunk_id = str(chunk_idx)
        runner.chunk_start = chunk_start
        runner.chunk_end = chunk_end

        # ENTRY_SIGNAL_TRACE: init writer when GX1_ENTRY_SIGNAL_TRACE=1 (TRUTH/SMOKE only)
        signal_trace_writer = None
        if is_truth_or_smoke_worker and os.environ.get("GX1_ENTRY_SIGNAL_TRACE", "0") == "1":
            from gx1.execution.entry_signal_trace import EntrySignalTraceWriter

            trace_path = chunk_output_dir / "ENTRY_SIGNAL_TRACE.jsonl"
            signal_trace_writer = EntrySignalTraceWriter(trace_path, enabled=True, max_lines=500)
            if hasattr(runner, "entry_manager") and runner.entry_manager is not None:
                runner.entry_manager._signal_trace_writer = signal_trace_writer

        # TRUTH 1W1C: ban segmented/parallel state on runner (defensive)
        runner.segment_start = None
        runner.segment_end = None
        runner.preroll_start = None
        runner.segmented_parallel_mode = False

        # propagate master-computed sha
        runner.bundle_sha256_from_master = bundle_sha256

        # ---------------------------------------------------------------------
        # PHASE 2: Load chunk data
        # ---------------------------------------------------------------------
        data_ctx: DataContext = load_chunk_data(bootstrap_ctx, chunk_start, chunk_end)

        chunk_df = data_ctx.chunk_df
        case_collision_resolution = getattr(data_ctx, "case_collision_resolution", None)
        chunk_data_path_abs = getattr(data_ctx, "chunk_data_path_abs", None)
        prebuilt_parquet_path_resolved = getattr(data_ctx, "prebuilt_parquet_path_resolved", None)
        actual_chunk_start = getattr(data_ctx, "actual_chunk_start", None)

        # timings from data_ctx (preferred)
        t_load_raw_s = _safe_float(getattr(data_ctx, "t_load_raw_s", 0.0), 0.0)
        t_load_prebuilt_s = _safe_float(getattr(data_ctx, "t_load_prebuilt_s", 0.0), 0.0)
        t_join_s = _safe_float(getattr(data_ctx, "t_join_s", 0.0), 0.0)
        t_write_s = _safe_float(getattr(data_ctx, "t_write_s", 0.0), 0.0)

        if chunk_local_padding_days and chunk_local_padding_days > 0:
            runner.replay_eval_start_ts = chunk_start
            runner.replay_eval_end_ts = chunk_end
            log.info(
                "[CHUNK %s] [CHUNK_LOCAL_PADDING] eval=[%s, %s] (actual_start=%s)",
                chunk_idx,
                chunk_start,
                chunk_end,
                actual_chunk_start,
            )

        if not chunk_data_path_abs:
            raise RuntimeError("[DATA_FAIL] chunk_data_path_abs missing after load_chunk_data()")

        # Prebuilt plumbing: loader state -> runner (before first on_bar)
        if getattr(data_ctx, "prebuilt_used", False) and getattr(data_ctx, "prebuilt_features_df", None) is not None:
            runner.prebuilt_features_df = data_ctx.prebuilt_features_df
            runner.prebuilt_used = True
            if getattr(data_ctx, "prebuilt_parquet_path_resolved", None):
                runner.prebuilt_parquet_path = data_ctx.prebuilt_parquet_path_resolved
                runner.prebuilt_features_path_resolved = data_ctx.prebuilt_parquet_path_resolved

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

        if STOP_REQUESTED and status == "ok":
            status = "stopped"
            error = "Stopped early due to SIGTERM"
            # Instrumentation: write signal_event capsule for debugging
            if chunk_output_dir is not None:
                _bars = _safe_int(getattr(runner, "perf_n_bars_processed", 0), 0)
                _total = int(len(chunk_df) if chunk_df is not None else 0)
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

        # ---------------------------------------------------------------------
        # PHASE 4: Extract counters/metrics (best-effort)
        # ---------------------------------------------------------------------
        wall_clock_sec = _safe_float(time.time() - worker_start_time, 0.0)

        # total bars in chunk_df (post-join)
        total_bars = int(len(chunk_df) if chunk_df is not None else 0)

        # bars processed (iterated by replay loop)
        # prefer explicit counter if runner provides it, else fall back to bars_seen/total_bars
        bars_seen = _safe_int(getattr(runner, "bars_seen", None), default=total_bars)
        bars_processed = _safe_int(getattr(runner, "perf_n_bars_processed", None), default=bars_seen)

        # evaluated bars = transformer forward calls (n_model_calls); prefer runner.perf_n_model_calls, then telemetry
        bars_evaluated = _safe_int(
            getattr(runner, "perf_n_model_calls", None),
            default=_safe_int(
                getattr(runner, "perf_n_policy_calls", None),
                default=_safe_int(
                    getattr(runner, "n_model_calls", None),
                    default=_safe_int(
                        getattr(
                            getattr(getattr(runner, "entry_manager", None), "entry_feature_telemetry", None),
                            "transformer_forward_calls",
                            None,
                        ),
                        default=0,
                    ),
                ),
            ),
        )

        # warmup holdback (best-effort)
        warmup_holdback_bars = _safe_int(getattr(runner, "first_valid_eval_idx_stored", 0), 0)

        # entry funnel
        bars_skipped_warmup = _safe_int(getattr(runner, "bars_skipped_warmup", 0), 0)
        bars_skipped_pregate = _safe_int(getattr(runner, "bars_skipped_pregate", 0), 0)
        bars_reaching_entry_stage = _safe_int(getattr(runner, "bars_reaching_entry_stage", 0), 0)
        pregate_enabled = _safe_bool(getattr(runner, "pregate_enabled", False), False)

        # feature perf
        feature_time_total_sec = _safe_float(getattr(runner, "perf_feat_time", 0.0), 0.0)
        feature_time_mean_ms = (feature_time_total_sec / total_bars * 1000.0) if total_bars > 0 else 0.0

        # pregate / regimes
        feature_timeout_count = _safe_int(getattr(runner, "feature_timeout_count", 0), 0)
        vol_regime_unknown_count = _safe_int(getattr(runner, "vol_regime_unknown_count", 0), 0)
        pregate_skips = _safe_int(getattr(runner, "pregate_skips", 0), 0)
        pregate_passes = _safe_int(getattr(runner, "pregate_passes", 0), 0)
        pregate_missing_inputs = _safe_int(getattr(runner, "pregate_missing_inputs", 0), 0)

        # timers
        t_pregate_total_sec = _safe_float(getattr(runner, "t_pregate_total_sec", 0.0), 0.0)
        t_xgb_predict_sec = _safe_float(getattr(runner, "t_xgb_predict_sec", 0.0), 0.0)
        t_transformer_forward_sec = _safe_float(getattr(runner, "t_transformer_forward_sec", 0.0), 0.0)
        t_gates_policy_sec = _safe_float(getattr(runner, "t_gates_policy_sec", 0.0), 0.0)
        t_replay_tags_sec = _safe_float(getattr(runner, "t_replay_tags_sec", 0.0), 0.0)
        t_telemetry_sec = _safe_float(getattr(runner, "t_telemetry_sec", 0.0), 0.0)
        t_replay_tags_build_inputs_sec = _safe_float(getattr(runner, "t_replay_tags_build_inputs_sec", 0.0), 0.0)
        t_replay_tags_rolling_sec = _safe_float(getattr(runner, "t_replay_tags_rolling_sec", 0.0), 0.0)
        t_replay_tags_ewm_sec = _safe_float(getattr(runner, "t_replay_tags_ewm_sec", 0.0), 0.0)
        t_replay_tags_rank_sec = _safe_float(getattr(runner, "t_replay_tags_rank_sec", 0.0), 0.0)
        t_replay_tags_assign_sec = _safe_float(getattr(runner, "t_replay_tags_assign_sec", 0.0), 0.0)
        t_io_total_sec = _safe_float(getattr(runner, "t_io_total_sec", 0.0), 0.0)

        # HTF stats
        htf_align_time_total_sec = _safe_float(getattr(runner, "htf_align_time_total_sec", 0.0), 0.0)
        htf_align_warning_time_sec = _safe_float(getattr(runner, "htf_align_warning_time_sec", 0.0), 0.0)
        htf_align_warn_count = _safe_int(getattr(runner, "htf_align_warn_count", 0), 0)

        try:
            if hasattr(runner, "feature_state") and runner.feature_state:
                h1_aligner = getattr(runner.feature_state, "h1_aligner", None)
                h4_aligner = getattr(runner.feature_state, "h4_aligner", None)

                if h1_aligner is not None:
                    s = h1_aligner.get_stats()
                    htf_h1_calls = _safe_int(s.get("call_count", 0), 0)
                    htf_h1_warns = _safe_int(s.get("warn_count", 0), 0)
                    htf_last_m5_ts = s.get("last_m5_ts")
                    htf_last_j = s.get("last_j")

                if h4_aligner is not None:
                    s = h4_aligner.get_stats()
                    htf_h4_calls = _safe_int(s.get("call_count", 0), 0)
                    htf_h4_warns = _safe_int(s.get("warn_count", 0), 0)

                htf_align_call_count = htf_h1_calls + htf_h4_calls
                htf_align_warn_count = htf_h1_warns + htf_h4_warns

                htf_align_fallback_count = _safe_int(getattr(runner.feature_state, "htf_align_fallback_count", 0), 0)
                htf_feature_compute_bars = _safe_int(getattr(runner.feature_state, "htf_feature_compute_bars", 0), 0)
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # PHASE 5: Trade outcomes parquet (TRUTH always) / legacy flush (non-TRUTH)
        # ---------------------------------------------------------------------
        if is_truth_or_smoke_worker:
            try:
                _write_trade_outcomes_truth(chunk_output_dir, run_id, runner)
            except Exception as e:
                log.error("[FLUSH] [CHUNK %s] TRUTH trade_outcomes write failed: %s", chunk_idx, e, exc_info=True)
                raise
        else:
            try:
                if runner and getattr(runner, "replay_eval_collectors", None):
                    from gx1.scripts.replay_eval_gated import flush_replay_eval_collectors

                    flush_replay_eval_collectors(runner, runner.replay_eval_collectors, output_dir=chunk_output_dir)
                    log.info("[FLUSH] [CHUNK %s] collectors flushed", chunk_idx)
                else:
                    log.warning("[FLUSH] [CHUNK %s] no collectors to flush", chunk_idx)
            except Exception as e:
                log.error("[FLUSH] [CHUNK %s] flush failed: %s", chunk_idx, e, exc_info=True)

        # ---------------------------------------------------------------------
        # PHASE 5a: TRUTH/SMOKE minimal attribution when legacy flush unavailable
        # ---------------------------------------------------------------------
        attribution_path = (chunk_output_dir / f"attribution_{run_id}.json") if chunk_output_dir is not None else None
        if is_truth_or_smoke_worker and attribution_path is not None and not attribution_path.exists():
            _write_minimal_attribution(chunk_output_dir, run_id)

        # ---------------------------------------------------------------------
        # PHASE 5b: TRUTH/SMOKE attribution gate (attribution file must exist)
        # ---------------------------------------------------------------------
        if is_truth_or_smoke_worker:
            if attribution_path is None or not attribution_path.exists():
                status = "failed_invariant"
                error = (
                    f"[TRUTH_MISSING_ATTR_PATH] attribution file missing: {attribution_path} "
                    "(required in TRUTH/SMOKE mode)"
                )
                log.error("[ATTRIBUTION_GATE] [CHUNK %s] %s", chunk_idx, error)
        else:
            if attribution_path is not None and not attribution_path.exists():
                log.warning(
                    "[ATTRIBUTION_GATE] [CHUNK %s] attribution file missing (non-TRUTH): %s",
                    chunk_idx,
                    attribution_path,
                )

        # ---------------------------------------------------------------------
        # PHASE 5c: ENTRY_SIGNAL_TRACE - close writer, run autopsy (TRUTH/SMOKE)
        # ---------------------------------------------------------------------
        if signal_trace_writer is not None:
            try:
                signal_trace_writer.close()
                n_entry_signals = _safe_int(
                    getattr(getattr(runner, "entry_manager", None), "killchain_n_above_threshold", 0),
                    0,
                )
                from gx1.execution.signal_autopsy import write_signal_autopsy_summary

                write_signal_autopsy_summary(
                    chunk_output_dir,
                    run_id,
                    n_entry_signals,
                    first_n_events=20,
                    is_truth_or_smoke=is_truth_or_smoke_worker,
                )
            except Exception as e:
                log.warning("[SIGNAL_AUTOPSY] [CHUNK %s] failed: %s", chunk_idx, e)

        # ---------------------------------------------------------------------
        # PHASE 6: Export killchain (best-effort)
        # ---------------------------------------------------------------------
        try:
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
        except Exception as e:
            log.warning("[KILLCHAIN_EXPORT] [CHUNK %s] failed: %s", chunk_idx, e)

        # ---------------------------------------------------------------------
        # PHASE 7: Prebuilt invariants (TRUTH/SMOKE may flip status)
        # ---------------------------------------------------------------------
        try:
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
        # PHASE 8b: ZERO_TRADES_DIAG (TRUTH observability, only when n_trades==0)
        # ---------------------------------------------------------------------
        if is_truth_or_smoke_worker and int(n_trades_closed or 0) == 0:
            try:
                from gx1.execution.zero_trades_diag import write_zero_trades_diag

                thresh_override = "override" if os.environ.get("GX1_ENTRY_THRESHOLD_OVERRIDE") else None
                write_zero_trades_diag(
                    chunk_output_dir=chunk_output_dir,
                    run_id=run_id,
                    chunk_idx=chunk_idx,
                    n_trades_closed=int(n_trades_closed or 0),
                    runner=runner,
                    bars_processed=int(bars_processed or 0),
                    total_bars=int(total_bars or 0),
                    n_model_calls=int(bars_evaluated or 0),
                    threshold_source_override=thresh_override,
                )
            except Exception as e:
                log.warning("[ZERO_TRADES_DIAG] [CHUNK %s] failed: %s", chunk_idx, e)

        # ---------------------------------------------------------------------
        # PHASE 8c: Bars invariant (written to footer; may raise AFTER footer)
        # ---------------------------------------------------------------------
        from gx1.execution.chunk_footer_invariants import check_bars_invariant

        bars_total_input = total_bars
        bars_invariant_gap = int(bars_total_input - bars_processed)
        bars_invariant_expected_gap = int(warmup_holdback_bars + tail_holdback_bars)
        bars_invariant_ok = check_bars_invariant(
            bars_total_input, bars_processed, tail_holdback_bars, status,
            warmup_holdback_bars=warmup_holdback_bars,
        )

        if not bars_invariant_ok:
            if is_truth_or_smoke_worker:
                log.error(
                    "[BARS_INVARIANT] gap=%d != expected=%d; will raise after footer write",
                    bars_invariant_gap,
                    bars_invariant_expected_gap,
                )
            else:
                log.warning(
                    "[BARS_INVARIANT] bars_total_input - bars_processed = %d != tail_holdback_bars = %d",
                    bars_invariant_gap,
                    bars_invariant_expected_gap,
                )

        # ---------------------------------------------------------------------
        # PHASE 9: Write chunk_footer.json (payload-only dumb writer)
        # ---------------------------------------------------------------------
        try:
            prebuilt_used_runner = _safe_bool(getattr(runner, "prebuilt_used", False), False)
            prebuilt_path_str = str(prebuilt_parquet_path_resolved) if prebuilt_parquet_path_resolved else None

            payload: Dict[str, Any] = {
                "run_id": run_id,
                "chunk_id": str(chunk_idx),
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
                "t_load_raw_s": float(t_load_raw_s) if is_truth_or_smoke_worker else None,
                "t_load_prebuilt_s": float(t_load_prebuilt_s) if is_truth_or_smoke_worker else None,
                "t_join_s": float(t_join_s) if is_truth_or_smoke_worker else None,
                "t_loop_s": float(t_loop_s) if is_truth_or_smoke_worker else None,
                "t_write_s": float(t_write_s) if is_truth_or_smoke_worker else None,

                # entry funnel
                "bars_seen": int(bars_seen),
                "bars_skipped_warmup": int(bars_skipped_warmup),
                "bars_skipped_pregate": int(bars_skipped_pregate),
                "bars_reaching_entry_stage": int(bars_reaching_entry_stage),
                "pregate_enabled": bool(pregate_enabled),

                # feature perf
                "feature_time_mean_ms": float(feature_time_mean_ms),
                "feature_time_total_sec": float(feature_time_total_sec),
                "t_feature_build_total_sec": float(feature_time_total_sec),  # alias for perf export required_fields
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
                # TRUTH-safe counter invariant: t_transformer_forward_sec > 0 but n_model_calls == 0 → observability bug
                "counter_invariant_violation": bool(
                    t_transformer_forward_sec > 0 and (bars_evaluated or 0) == 0
                ),

                # prebuilt flags/paths
                "prebuilt_used": bool(prebuilt_used_runner),
                "prebuilt_parquet_path": prebuilt_path_str,
                "prebuilt_required_columns": getattr(bootstrap_ctx, "prebuilt_required_columns", None),

                # threshold / analysis mode (one-glance override verification)
                "analysis_mode": os.environ.get("GX1_ANALYSIS_MODE") == "1",
                "threshold_used": (
                    getattr(getattr(runner, "entry_manager", None), "threshold_used", None)
                    if runner else None
                ),
                "threshold_source": (
                    "override"
                    if (os.environ.get("GX1_ANALYSIS_MODE") == "1" and os.environ.get("GX1_ENTRY_THRESHOLD_OVERRIDE"))
                    else "canonical"
                ),

                # exit strategy observability
                "exit_profile": (
                    "RULE6A_PURE"
                    if (
                        getattr(runner, "exit_type", None) == "EXIT_FARM_V2_RULES_ADAPTIVE"
                        and not (getattr(runner, "policy", None) or {}).get("hybrid_exit_router")
                    )
                    else (getattr(runner, "exit_config_name", None) or "unknown")
                ),
                "exit_type": getattr(runner, "exit_type", None),
                "router_enabled": (
                    False
                    if getattr(runner, "exit_type", None) in ("MASTER_EXIT_V1", "EXIT_TRANSFORMER_V0")
                    else bool((getattr(runner, "policy", None) or {}).get("hybrid_exit_router"))
                ),
                "exit_critic_enabled": (
                    False
                    if getattr(runner, "exit_type", None) in ("MASTER_EXIT_V1", "EXIT_TRANSFORMER_V0")
                    else bool(
                        (getattr(runner, "policy", None) or {}).get("exit_critic", {}).get("enabled", False)
                    )
                ),
                "exit_tuning_capsule": (
                    getattr(runner, "exit_tuning_capsule", None)
                    if getattr(runner, "exit_type", None) == "MASTER_EXIT_V1"
                    else None
                ),
                "exit_ml_enabled": getattr(runner, "exit_ml_enabled", False),
                "exit_ml_decision_mode": getattr(runner, "exit_ml_decision_mode", "") or None,
                "exit_ml_config_hash": getattr(runner, "exit_ml_config_hash", "") or None,
                "exit_ml_model_sha": getattr(runner, "exit_ml_model_sha", None),
                "exit_ml_input_dim": getattr(runner, "exit_ml_input_dim", None),
                "exit_ml_io_version": getattr(runner, "exit_ml_io_version", None),

                # ssot
                "ssot": {"bundle_sha256": bundle_sha256},

                # ctx feature mask (diagnostics / ablation)
                "ctx_cont_dim": _safe_int(getattr(runner, "ctx_cont_dim", 0), 0) or None,
                "ctx_cat_dim": _safe_int(getattr(runner, "ctx_cat_dim", 0), 0) or None,
                "ctx_cont_mask_id": getattr(runner, "ctx_cont_mask_id", None),
                "ctx_cat_mask_id": getattr(runner, "ctx_cat_mask_id", None),
                "ctx_cont_mask": getattr(runner, "ctx_cont_mask", None),
                "ctx_cat_mask": getattr(runner, "ctx_cat_mask", None),
            }

            footer_ctx = ChunkFooterContext(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                payload=payload,
                run_id=run_id,
            )
            write_chunk_footer(footer_ctx)

            log.info("[CHUNK %s] chunk_footer.json written (status=%s)", chunk_idx, status)

            # TRUTH: if bars invariant failed, raise AFTER footer write
            if (not bars_invariant_ok) and is_truth_or_smoke_worker:
                raise RuntimeError(
                    f"[BARS_INVARIANT] bars_total_input - bars_processed = {bars_invariant_gap} "
                    f"!= expected (warmup+tail) = {bars_invariant_expected_gap}. "
                    f"bars_total_input={bars_total_input}, bars_processed={bars_processed}, "
                    f"warmup_holdback_bars={warmup_holdback_bars}, tail_holdback_bars={tail_holdback_bars} (footer written with bars_invariant_ok=false)"
                )

        except Exception as footer_err:
            # If the error is the intentional invariant raise, bubble it.
            if isinstance(footer_err, RuntimeError) and "BARS_INVARIANT" in str(footer_err):
                raise

            log.error("[CHUNK %s] Failed to write chunk_footer.json: %s", chunk_idx, footer_err, exc_info=True)

            # Failure capsule (best-effort)
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
                bar_counters = compute_bar_counters_snapshot(runner, int(bars_processed or 0), chunk_df)
                fail_capsule = {
                    **failure_context,
                    "traceback": tb_str[:10000],
                    "bar_counters": bar_counters,
                    "hint": "footer_error: Failed to write chunk_footer.json",
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

            # Stub footer (never overwrite chunk_footer.json)
            try:
                stub = {
                    "run_id": run_id,
                    "chunk_id": str(chunk_idx),
                    "status": "footer_error",
                    "error": f"Failed to write chunk_footer.json: {str(footer_err)[:500]}",
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
        # attribution: str path (avoids Path/JSON type friction in merge)
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
            "chunk_idx": chunk_idx,
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

        # TRUTH/SMOKE strictness: fail fast if invariants failed
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

        # Ensure chunk_output_dir exists best-effort
        if chunk_output_dir is None:
            try:
                chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
                chunk_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                chunk_output_dir = None

        # Update skip_ledger
        skip_ledger["stage"] = "exception"
        skip_ledger["exception_type"] = type(outer_exc).__name__
        skip_ledger["exception_msg"] = str(outer_exc)[:500]
        skip_ledger["traceback"] = error_traceback[:5000]

        # Failure capsule (best-effort)
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
                bar_counters = compute_bar_counters_snapshot(runner, int(bars_processed or 0), chunk_df)
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
            skip_ledger["reached_entry_stage"] = int(bars_reaching_entry_stage or 0)

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
