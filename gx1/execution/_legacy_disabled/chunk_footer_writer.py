"""
Chunk Footer Writer (DUMB WRITER)

Contract:
- Does NOT read from disk.
- Does NOT read from runner/telemetry objects.
- Does NOT validate invariants or mutate status/error (except dumb defaults).
- Does NOT write any files except: chunk_footer.json
- Best-effort: never raises; logs on failure.
- Atomic write: tmp -> os.replace()

All "smart" logic must happen upstream (replay_chunk.py / exporters),
and the final footer content must be provided via ctx.payload.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkFooterContext:
    """
    Minimal context for writing chunk_footer.json.

    payload SHOULD already contain:
      - status
      - error (optional)
      - run_id (recommended)
      - chunk_id (optional; will be set from chunk_idx if missing)
      - timestamp (optional; will be set if missing)

    This writer will not validate correctness of those fields.
    """
    chunk_output_dir: Path
    chunk_idx: int
    payload: Dict[str, Any]
    run_id: Optional[str] = None  # optional, only for backfill/log context


def _now_iso_utc() -> str:
    # Local dumb timestamp helper. No dt_module dependency.
    return datetime.now(timezone.utc).isoformat()


def write_chunk_footer(ctx: ChunkFooterContext) -> Dict[str, Any]:
    """
    Build (finalize minimal metadata) and atomically write chunk_footer.json.

    Returns the dict that was attempted written (best-effort).
    Never raises exceptions.
    """
    # 1) Copy payload (do not mutate caller)
    footer: Dict[str, Any]
    if isinstance(ctx.payload, dict):
        footer = dict(ctx.payload)
    else:
        footer = {"payload": ctx.payload}

    # 2) Dumb defaults (deterministic-ish; not "validation")
    footer.setdefault("chunk_id", int(ctx.chunk_idx))
    footer.setdefault("timestamp", _now_iso_utc())
    footer.setdefault("status", "UNKNOWN")  # dumb backstop only

    # Backfill run_id if missing
    if ctx.run_id and "run_id" not in footer:
        footer["run_id"] = ctx.run_id

    # 3) Convert to JSON-serializable (best effort)
    footer_jsonable = _to_jsonable(footer)

    # 4) Atomic write: tmp -> replace (best-effort, no raise)
    out_dir = Path(ctx.chunk_output_dir)
    out_path = out_dir / "chunk_footer.json"
    tmp_path = out_dir / f"chunk_footer.json.tmp.{os.getpid()}"

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(footer_jsonable, f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, out_path)

        log.info(
            "[CHUNK %s] chunk_footer.json written (status=%s, run_id=%s)",
            ctx.chunk_idx,
            footer_jsonable.get("status"),
            footer_jsonable.get("run_id"),
        )

    except Exception as e:
        # Best-effort: log and cleanup tmp; never raise
        log.error(
            "[CHUNK %s] Failed to write chunk_footer.json: %s",
            ctx.chunk_idx,
            e,
            exc_info=True,
        )
        try:
            Path(tmp_path).unlink(missing_ok=True)  # py3.8+ supports missing_ok
        except Exception:
            try:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass

    # Return the *attempted* footer (original types), not jsonable copy
    return footer


def _to_jsonable(x: Any) -> Any:
    """
    Convert arbitrary objects to JSON-serializable types.
    Best-effort. Never raises.
    """
    try:
        return _to_jsonable_inner(x)
    except Exception:
        try:
            return str(x)
        except Exception:
            return "<unserializable>"


def _to_jsonable_inner(x: Any) -> Any:
    # None / primitives
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # dict-like
    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            try:
                ks = k if isinstance(k, str) else str(k)
            except Exception:
                ks = "<key>"
            out[ks] = _to_jsonable(v)
        return out

    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    # pathlib.Path
    if isinstance(x, Path):
        return str(x)

    # bytes
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return str(x)

    # datetime-like: has isoformat()
    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            pass

    # numpy (optional)
    try:
        import numpy as np  # type: ignore
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass

    # pandas-like without importing pandas
    tname = type(x).__name__
    if tname in ("Timestamp", "Timedelta"):
        try:
            return str(x)
        except Exception:
            return "<pandas-like>"

    # fallback
    try:
        return str(x)
    except Exception:
        return "<unserializable>"