"""
Chunk Failure Handling: Robust, minimal, idempotent error handling helpers.

Goals (SSoT/TRUTH-friendly):
- Never raises (best-effort only).
- No disk reads in error path.
- Atomic writes (tmp -> replace) for JSON artifacts, with fsync.
- JSON-serializable conversion for common GX1 runtime objects (numpy/pandas/path/datetime).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from gx1.utils.dt_module import now_iso as dt_now_iso

log = logging.getLogger(__name__)


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert common non-JSON-serializable objects to JSON-safe types.
    Best-effort, recursive, never raises.

    Handles:
    - numpy scalars/arrays (if numpy available)
    - pandas Timestamp/NaT (if pandas available)
    - datetime
    - Path
    - dict/list/tuple/set
    - Exception
    """
    try:
        # Fast-path primitives
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Path
        if isinstance(obj, Path):
            return str(obj)

        # datetime
        if isinstance(obj, datetime):
            # ISO 8601 is stable and readable
            return obj.isoformat()

        # dict / iterables
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                # JSON keys must be strings
                out[str(k)] = convert_to_json_serializable(v)
            return out

        if isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(x) for x in obj]

        if isinstance(obj, set):
            # deterministic ordering
            return [convert_to_json_serializable(x) for x in sorted(list(obj), key=lambda y: str(y))]

        # Exceptions
        if isinstance(obj, BaseException):
            return {
                "type": type(obj).__name__,
                "message": str(obj),
            }

        # numpy (optional)
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except Exception:
            pass

        # pandas (optional)
        try:
            import pandas as pd  # type: ignore

            if isinstance(obj, pd.Timestamp):
                # Preserve timezone info if present
                return obj.isoformat()
            # Some pandas objects stringify nicely; keep best-effort
        except Exception:
            pass

        # Last resort: string
        return str(obj)
    except Exception:
        # Ultra-last-resort
        try:
            return str(obj)
        except Exception:
            return None


def _atomic_write_json_best_effort(path: Path, payload: Dict[str, Any]) -> bool:
    """
    Internal: write JSON atomically via tmp -> replace with fsync.
    Never raises.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

        return True
    except Exception as e:
        # Best-effort cleanup (allowed: exists() check before unlink)
        try:
            if "tmp_path" in locals() and isinstance(tmp_path, Path) and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

        log.warning(f"[ATOMIC_WRITE] Failed to write {path}: {e}")
        return False


def atomic_write_json_safe(path: Path, payload: Dict[str, Any]) -> bool:
    """
    Safe atomic JSON write that never raises.

    Strategy:
    1) Convert payload to JSON-serializable types (best-effort).
    2) Try gx1.utils.atomic_json.atomic_write_json (if available).
    3) Fallback to our own atomic tmp->replace writer.
    """
    if not isinstance(payload, dict):
        log.warning(f"[ATOMIC_WRITE] Payload is not dict: {type(payload)}")
        return False

    try:
        payload_ser = convert_to_json_serializable(payload)
        if not isinstance(payload_ser, dict):
            # Should not happen, but keep us safe
            log.warning("[ATOMIC_WRITE] Payload conversion did not return dict; forcing dict wrapper")
            payload_ser = {"payload": payload_ser}

        # Try project atomic writer first (may include extra guarantees)
        try:
            from gx1.utils.atomic_json import atomic_write_json  # type: ignore

            atomic_write_json(path, payload_ser)
            return True
        except Exception as atomic_error:
            log.debug(f"[ATOMIC_WRITE] gx1.utils.atomic_json failed, falling back: {atomic_error}")

        # Fallback: atomic tmp->replace
        return _atomic_write_json_best_effort(path, payload_ser)

    except Exception as e:
        log.warning(f"[ATOMIC_WRITE] Failed to write {path}: {e}")
        return False


def write_fatal_capsule(
    *,
    chunk_output_dir: Path,
    chunk_idx: int,
    run_id: str,
    fatal_reason: str,
    error_message: str,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Write fatal capsule for bootstrap-ish failures.
    Best-effort, never raises.

    Writes:
    - WORKER_START.json (atomic)
    - FATAL_ERROR.txt (best-effort)
    """
    ok_any = False
    try:
        capsule: Dict[str, Any] = {
            "timestamp": dt_now_iso(),
            "chunk_id": chunk_idx,
            "run_id": run_id,
            "fatal_reason": fatal_reason,
            "error_message": error_message,
        }
        if extra_fields:
            capsule.update(convert_to_json_serializable(extra_fields))

        chunk_output_dir.mkdir(parents=True, exist_ok=True)

        worker_start_path = chunk_output_dir / "WORKER_START.json"
        ok_any = atomic_write_json_safe(worker_start_path, capsule) or ok_any

        fatal_txt_path = chunk_output_dir / "FATAL_ERROR.txt"
        try:
            lines = [error_message]
            if extra_fields:
                for k, v in extra_fields.items():
                    lines.append(f"{k}: {convert_to_json_serializable(v)}")
            content = "\n".join(lines) + "\n"

            with open(fatal_txt_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            ok_any = True
        except Exception as txt_error:
            log.warning(f"[FATAL_CAPSULE] Failed to write FATAL_ERROR.txt: {txt_error}")

        return ok_any
    except Exception as e:
        log.error(f"[FATAL_CAPSULE] Failed to write fatal capsule: {e}", exc_info=True)
        return ok_any


def write_failure_capsule(
    *,
    chunk_output_dir: Optional[Path],
    payload: Dict[str, Any],
    filename: str = "CHUNK_FAIL_CAPSULE.json",
    chunk_idx: int,
    run_id: str,
) -> Optional[str]:
    """
    Robust writer for failure capsules. Never raises.

    First tries chunk_output_dir, then falls back to /tmp.
    Returns written path string, else None.
    """
    try:
        payload_ser = convert_to_json_serializable(payload)
        if not isinstance(payload_ser, dict):
            payload_ser = {"payload": payload_ser}

        # Try chunk_output_dir
        if chunk_output_dir:
            try:
                chunk_output_dir.mkdir(parents=True, exist_ok=True)
                capsule_path = chunk_output_dir / filename
                if atomic_write_json_safe(capsule_path, payload_ser):
                    return str(capsule_path)
            except Exception:
                pass

        # Fallback to /tmp
        try:
            fallback_path = Path(tempfile.gettempdir()) / f"chunk_{chunk_idx}_{filename}_{run_id}.json"
            if atomic_write_json_safe(fallback_path, payload_ser):
                log.warning(f"[CHUNK {chunk_idx}] Wrote {filename} to fallback: {fallback_path}")
                return str(fallback_path)
        except Exception:
            pass

        return None
    except Exception as e:
        log.warning(f"[CHUNK {chunk_idx}] Failed to write {filename}: {e}")
        return None


def build_failure_context(
    *,
    runner: Optional[Any],
    chunk_df: Optional[Any],
    chunk_output_dir: Optional[Path],  # kept for schema completeness; never read/written here
    chunk_idx: int,
    run_id: str,
    error: Exception,
    bars_processed_safe: int,
    first_iter_ts: Optional[Any],
    last_iter_ts: Optional[Any],
    policy_id: Optional[str],
    bundle_sha256: Optional[str],
) -> Dict[str, Any]:
    """
    Build failure context dict using only safe getattr and primitive derivations.
    No file I/O, no heavy imports, never raises.
    """
    ctx: Dict[str, Any] = {
        "chunk_idx": chunk_idx,
        "run_id": run_id,
        "exception_type": type(error).__name__ if error else None,
        "exception_message": str(error)[:500] if error else None,
        "bars_processed": int(bars_processed_safe) if bars_processed_safe is not None else 0,
        "first_iter_ts": convert_to_json_serializable(first_iter_ts) if first_iter_ts else None,
        "last_iter_ts": convert_to_json_serializable(last_iter_ts) if last_iter_ts else None,
        "policy_id": policy_id,
        "bundle_sha256": bundle_sha256,
        "timestamp": dt_now_iso(),
    }

    # Runner (safe getattr only)
    if runner is not None:
        try:
            ctx["runner_available"] = True
            ctx["prebuilt_used"] = getattr(runner, "prebuilt_used", None)
            ctx["lookup_attempts"] = getattr(runner, "lookup_attempts", 0)
            ctx["lookup_hits"] = getattr(runner, "lookup_hits", 0)
            ctx["lookup_misses"] = getattr(runner, "lookup_misses", 0)
            ctx["prebuilt_lookup_phase"] = getattr(runner, "prebuilt_lookup_phase", "unknown")

            em = getattr(runner, "entry_manager", None)
            if em is not None:
                ctx["entry_manager_available"] = True
                ctx["eval_calls_total"] = getattr(em, "eval_calls_total", None)
                ctx["eval_calls_prebuilt_gate_true"] = getattr(em, "eval_calls_prebuilt_gate_true", None)
                ctx["eval_calls_prebuilt_gate_false"] = getattr(em, "eval_calls_prebuilt_gate_false", None)

                telemetry = getattr(em, "entry_feature_telemetry", None)
                if telemetry is not None:
                    ctx["telemetry_available"] = True
                    ctx["entry_routing_selected_model"] = getattr(telemetry, "entry_routing_selected_model", None)
                    ctx["entry_routing_reason"] = getattr(telemetry, "entry_routing_reason", None)
                    ctx["entry_v10_enabled"] = getattr(telemetry, "entry_v10_enabled", None)
                else:
                    ctx["telemetry_available"] = False
            else:
                ctx["entry_manager_available"] = False
        except Exception:
            ctx["runner_available"] = False

    # chunk_df summary (safe only)
    if chunk_df is not None:
        try:
            ctx["chunk_df_rows"] = int(len(chunk_df))
            if len(chunk_df) > 0:
                idx = getattr(chunk_df, "index", None)
                if idx is not None:
                    ctx["chunk_df_first_ts"] = convert_to_json_serializable(idx[0])
                    ctx["chunk_df_last_ts"] = convert_to_json_serializable(idx[-1])
        except Exception:
            pass

    # Telemetry status from env only (no file checks)
    ctx["telemetry_status"] = {
        "telemetry_required": os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1",
        "collector_initialized": bool(ctx.get("telemetry_available", False)),
        "telemetry_write_attempted": False,
        "telemetry_files_written": [],
        "no_entry_evaluations": False,
        "no_entry_reason": None,
    }

    return convert_to_json_serializable(ctx)


def write_stub_footer(
    *,
    chunk_output_dir: Path,
    chunk_idx: int,
    run_id: str,
    footer_error: Exception,
    bars_processed_safe: int,
    total_bars_safe: int,
    runner: Optional[Any],
    dt_module_version: Optional[str],
) -> bool:
    """
    Write minimal stub footer on error.
    Writes chunk_footer_stub.json and never overwrites chunk_footer.json.
    """
    try:
        stub: Dict[str, Any] = {
            "run_id": run_id,
            "chunk_id": str(chunk_idx),
            "status": "footer_error",
            "error": f"Failed to write chunk_footer.json: {str(footer_error)[:500]}",
            "bars_processed": int(bars_processed_safe) if bars_processed_safe is not None else 0,
            "total_bars": int(total_bars_safe) if total_bars_safe is not None else 0,
            "dt_module_version": dt_module_version,
            "timestamp": dt_now_iso(),
        }

        if runner is not None:
            try:
                stub["prebuilt_used"] = getattr(runner, "prebuilt_used", None)
                stub["lookup_attempts"] = getattr(runner, "lookup_attempts", 0)
                stub["lookup_hits"] = getattr(runner, "lookup_hits", 0)
                stub["lookup_misses"] = getattr(runner, "lookup_misses", 0)
            except Exception:
                pass

        stub_path = chunk_output_dir / "chunk_footer_stub.json"
        ok = atomic_write_json_safe(stub_path, convert_to_json_serializable(stub))
        if ok:
            log.warning(f"[CHUNK {chunk_idx}] Wrote chunk_footer_stub.json (footer_error)")
        return ok
    except Exception as e:
        log.error(f"[CHUNK {chunk_idx}] Failed to write chunk_footer_stub.json: {e}", exc_info=True)
        return False


def write_signal_event_capsule(
    *,
    chunk_output_dir: Path,
    run_id: str,
    chunk_idx: int,
    bars_processed: int,
    total_bars: int,
    last_ts: Optional[Any] = None,
    wall_clock_sec: float = 0.0,
) -> bool:
    """
    Write SIGNAL_EVENT.json when SIGTERM is received during replay.
    Best-effort: reads /proc/self/status for VmRSS/VmHWM, ppid, etc.
    Never raises.
    """
    try:
        payload: Dict[str, Any] = {
            "event": "SIGTERM_RECEIVED",
            "timestamp": dt_now_iso(),
            "run_id": run_id,
            "chunk_idx": chunk_idx,
            "bars_processed": bars_processed,
            "total_bars": total_bars,
            "last_ts": str(last_ts) if last_ts is not None else None,
            "wall_clock_sec": wall_clock_sec,
            "pid": os.getpid(),
            "ppid": None,
        }
        try:
            payload["ppid"] = os.getppid()
        except (AttributeError, OSError):
            pass
        try:
            status_path = Path(f"/proc/{os.getpid()}/status")
            if status_path.exists():
                for line in status_path.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        payload["vmrss_kb"] = line.split(":")[1].strip().split()[0]
                    elif line.startswith("VmHWM:"):
                        payload["vmhwm_kb"] = line.split(":")[1].strip().split()[0]
                    elif line.startswith("PPid:"):
                        payload["ppid_proc"] = line.split(":")[1].strip()
        except Exception:
            pass
        payload["env_hints"] = {
            "GX1_STOP_REQUESTED": os.getenv("GX1_STOP_REQUESTED", "?"),
            "GX1_RUN_MODE": os.getenv("GX1_RUN_MODE", "?"),
        }
        path = chunk_output_dir / "SIGNAL_EVENT.json"
        return atomic_write_json_safe(path, convert_to_json_serializable(payload))
    except Exception as e:
        log.warning(f"[SIGNAL_EVENT] Failed to write capsule: {e}")
        return False