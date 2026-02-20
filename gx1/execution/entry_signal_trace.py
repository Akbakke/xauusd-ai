"""
Entry Signal Trace: JSONL writer for entry-signal diagnosis.

When GX1_ENTRY_SIGNAL_TRACE=1, writes chunk_0/ENTRY_SIGNAL_TRACE.jsonl
(one line per entry-signal) with: time, xgb_signal, session, entry_attempted,
entry_taken, final_block_code, risk_snapshot.

Contract: Best-effort, never raises. Hard-cap on max_lines.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

TRACE_VERSION = 1


def build_trace_event_base(
    time: str,
    run_id: str,
    model_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    session: Optional[str] = None,
    threshold_used: Optional[float] = None,
    threshold_source: Optional[str] = None,
    analysis_mode: Optional[bool] = None,
    xgb_prob_long: Optional[float] = None,
    xgb_prob_short: Optional[float] = None,
    xgb_p_hat: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build standard base fields for a trace event."""
    out: Dict[str, Any] = {
        "trace_version": TRACE_VERSION,
        "time": time,
        "run_id": run_id,
        "model_id": model_id,
        "chunk_id": chunk_id,
        "session": session,
        "threshold_used": threshold_used,
        "threshold_source": threshold_source,
        "analysis_mode": analysis_mode,
        "xgb_prob_long": xgb_prob_long,
        "xgb_prob_short": xgb_prob_short,
        "xgb_p_hat": xgb_p_hat,
    }
    out.update(extra)
    return {k: v for k, v in out.items() if v is not None}


class EntrySignalTraceWriter:
    """
    Writes ENTRY_SIGNAL_TRACE.jsonl. Best-effort, never raises.
    Hard-cap on max_lines; drops excess and tracks dropped_lines.
    """

    def __init__(self, path: Path, enabled: bool, max_lines: int = 500) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.max_lines = max(1, int(max_lines))
        self._line_count = 0
        self._dropped_lines = 0
        self._file_handle: Optional[Any] = None

    def append(self, event: Dict[str, Any]) -> None:
        """Append one JSON line. Deterministic key order (sort_keys=True)."""
        if not self.enabled:
            return
        try:
            if self._line_count >= self.max_lines:
                self._dropped_lines += 1
                if self._dropped_lines == 1:
                    log.warning(
                        "[ENTRY_SIGNAL_TRACE] Max lines (%d) reached, dropping further events",
                        self.max_lines,
                    )
                return
            line = json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n"
            if self._file_handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file_handle = open(self.path, "a", encoding="utf-8")
            self._file_handle.write(line)
            self._file_handle.flush()
            self._line_count += 1
        except Exception as e:
            log.warning("[ENTRY_SIGNAL_TRACE] Failed to append: %s", e)

    def close(self) -> None:
        """Close file handle. Idempotent."""
        try:
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
        except Exception as e:
            log.warning("[ENTRY_SIGNAL_TRACE] Failed to close: %s", e)

    @property
    def line_count(self) -> int:
        return self._line_count

    @property
    def dropped_lines(self) -> int:
        return self._dropped_lines
