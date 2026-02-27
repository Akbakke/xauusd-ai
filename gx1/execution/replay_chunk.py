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

    bars_evaluated = _safe_int(
        getattr(runner, "perf_n_model_calls", None),
        default=_safe_int(
            getattr(runner, "perf_n_policy_calls", None),
            default=_safe_int(getattr(runner, "n_model_calls", None), default=0),
        ),
    )

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
    t_load_raw_s = 0.0
    t_load_prebuilt_s = 0.0
    t_join_s = 0.0
    t_loop_s = 0.0
    t_write_s = 0.0

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
        from gx1.execution.oanda_demo_runner import GX1DemoRunner

        # TRUTH: entry v10_ctx loads from GX1_BUNDLE_DIR; prefer transformer bundle when set
        canonical_transformer = os.getenv("GX1_CANONICAL_TRANSFORMER_BUNDLE_DIR")
        if canonical_transformer:
            os.environ["GX1_BUNDLE_DIR"] = canonical_transformer
            log.info("[CHUNK %s] GX1_BUNDLE_DIR=%s (canonical transformer)", chunk_idx, canonical_transformer)
        elif os.getenv("GX1_CANONICAL_BUNDLE_DIR"):
            canonical_bundle_dir = os.getenv("GX1_CANONICAL_BUNDLE_DIR")
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
            _write_trade_outcomes_truth(chunk_output_dir, run_id, runner)

            attribution_path = chunk_output_dir / f"attribution_{run_id}.json"
            if not attribution_path.exists():
                _write_minimal_attribution(chunk_output_dir, run_id)
            log.info("[TRUTH] skip legacy flush_replay_eval_collectors (forbidden import path)")
        else:
            # Legacy flush path is forbidden; skip to avoid gx1.scripts.* import in TRUTH/SMOKE contexts.
            log.info("[REPLAY_EVAL] skip legacy flush_replay_eval_collectors (forbidden import path)")

        # ---------------------------------------------------------------------
        # PHASE 5b: Optional observability (never fatal)
        # ---------------------------------------------------------------------
        _try_write_optional_observability(
            chunk_output_dir=chunk_output_dir,
            run_id=run_id,
            chunk_idx=chunk_idx,
            runner=runner,
            bars_processed=int(bars_processed),
            total_bars=int(total_bars),
            wall_clock_sec=float(wall_clock_sec),
        )

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
        # PHASE 8c: Bars invariant (may raise AFTER footer write in TRUTH/SMOKE)
        # ---------------------------------------------------------------------
        from gx1.execution.chunk_footer_invariants import check_bars_invariant  # local import

        bars_total_input = int(total_bars)
        bars_invariant_gap = int(bars_total_input - int(bars_processed))
        bars_invariant_expected_gap = int(warmup_holdback_bars + tail_holdback_bars)

        bars_invariant_ok = check_bars_invariant(
            bars_total_input=bars_total_input,
            bars_processed=int(bars_processed),
            tail_holdback_bars=int(tail_holdback_bars),
            status=status,
            warmup_holdback_bars=int(warmup_holdback_bars),
        )

        if not bars_invariant_ok:
            msg = (
                f"[BARS_INVARIANT] gap={bars_invariant_gap} != expected={bars_invariant_expected_gap} "
                f"(warmup={warmup_holdback_bars} tail={tail_holdback_bars})"
            )
            if is_truth_or_smoke_worker:
                log.error("%s; will raise after footer write", msg)
            else:
                log.warning("%s", msg)

        # ---------------------------------------------------------------------
        # PHASE 9: Write chunk_footer.json (DUMB WRITER; payload-only)
        # ---------------------------------------------------------------------
        try:
            import_proof_path, forbidden_hits = _write_import_proof_if_needed(
                chunk_output_dir=chunk_output_dir,
                run_id=run_id,
                chunk_idx=chunk_idx,
                truth_artifacts=truth_artifacts,
            )
            if forbidden_hits:
                raise RuntimeError(
                    "[TRUTH_FORBIDDEN_SYMBOL_IMPORTS] Forbidden modules in sys.modules after replay: "
                    + ", ".join(sorted({h.get('module') for h in forbidden_hits}))
                )

            prebuilt_used_runner = _safe_bool(getattr(runner, "prebuilt_used", False), False)
            prebuilt_path_str = str(prebuilt_parquet_path_resolved) if prebuilt_parquet_path_resolved else None

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
                "prebuilt_parquet_path": prebuilt_path_str,
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
            write_chunk_footer(footer_ctx)

            log.info("[CHUNK %s] chunk_footer.json written (status=%s)", chunk_idx, status)

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
        "GX1_CANONICAL_TAPE_ROOT",
        "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL",
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
        "canonical_tape_root": tape_root,
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