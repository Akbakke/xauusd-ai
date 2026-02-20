"""
ZERO_TRADES_DIAG: TRUTH-safe diagnostic file when n_trades==0.

Written to chunk_output_dir. Used to diagnose why a run produced 0 trades
(threshold, coverage, filters, reject reasons).
Contract: Best-effort, never raises. Only writes when n_trades_closed==0.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from gx1.execution.chunk_failure import atomic_write_json_safe, convert_to_json_serializable
from gx1.utils.dt_module import now_iso as dt_now_iso

log = logging.getLogger(__name__)

DIAG_VERSION = "1"


def _extract_model_id(run_id: str) -> str:
    """Extract model name from run_id (e.g. MODEL_COMPARE_FULLYEAR_BASE28_... -> BASE28)."""
    for prefix in ("MODEL_COMPARE_FULLYEAR_", "MODEL_COMPARE_Q1_", "MODEL_COMPARE_Q2_", "MODEL_COMPARE_Q3_", "MODEL_COMPARE_Q4_"):
        if run_id.startswith(prefix):
            rest = run_id[len(prefix):]
            # BASE28_20260215_... or BASE28_CTX2PLUS_T1_...
            parts = rest.split("_")
            if parts:
                return str(parts[0])
    return "unknown"


def write_zero_trades_diag(
    chunk_output_dir: Path,
    run_id: str,
    chunk_idx: int,
    n_trades_closed: int,
    runner: Any,
    bars_processed: int,
    total_bars: int,
    n_model_calls: int,
    threshold_source_override: Optional[str] = None,
) -> Optional[Path]:
    """
    Write ZERO_TRADES_DIAG.json when n_trades_closed==0.

    Best-effort: gathers threshold, counts, reject histogram from runner/killchain.
    Never raises. Only writes when n_trades_closed==0.
    threshold_source_override: "override" when GX1_ENTRY_THRESHOLD_OVERRIDE is set.
    """
    if n_trades_closed != 0:
        return None

    out_path = chunk_output_dir / "ZERO_TRADES_DIAG.json"
    model_id = _extract_model_id(run_id)
    has_override = bool(os.environ.get("GX1_ENTRY_THRESHOLD_OVERRIDE"))
    analysis_mode = os.environ.get("GX1_ANALYSIS_MODE") == "1"
    threshold_source = threshold_source_override or ("override" if has_override else "missing")

    payload: Dict[str, Any] = {
        "diag_version": DIAG_VERSION,
        "contract_id": "ZERO_TRADES_DIAG_V1",
        "model_id": model_id,
        "run_id": run_id,
        "chunk_id": chunk_idx,
        "timestamp": dt_now_iso(),
        "analysis_mode": analysis_mode,
        "threshold_used": None,
        "threshold_source": threshold_source,
        "counts": {
            "bars_total_input": int(total_bars),
            "bars_processed": int(bars_processed),
            "bars_evaluated": int(n_model_calls),
            "n_model_calls": int(n_model_calls),
        },
        "n_decisions": None,
        "n_entry_signals": None,
        "n_entries_taken": 0,
        "n_trades_closed": 0,
        "n_rejects_total": None,
        "reject_reason_histogram": {},
        "session_histogram": {},
    }

    try:
        em = getattr(runner, "entry_manager", None) if runner else None
        if em:
            payload["threshold_used"] = getattr(em, "threshold_used", None)
            if threshold_source == "missing" and payload["threshold_used"] is not None:
                payload["threshold_source"] = "canonical"
            payload["n_decisions"] = int(getattr(em, "killchain_n_entry_pred_total", 0) or 0)
            payload["n_entry_signals"] = int(getattr(em, "killchain_n_above_threshold", 0) or 0)
            raw_reasons = getattr(em, "killchain_block_reason_counts", {}) or {}
            payload["reject_reason_histogram"] = dict(sorted((str(k), int(v)) for k, v in raw_reasons.items()))
            payload["n_rejects_total"] = sum(payload["reject_reason_histogram"].values())
    except Exception as e:
        log.warning("[ZERO_TRADES_DIAG] Failed to read entry_manager: %s", e)

    # Bundle SHA (from runner / env)
    try:
        if runner and hasattr(runner, "bundle_sha256"):
            payload["bundle_sha"] = getattr(runner, "bundle_sha256", None)
    except Exception:
        pass
    if payload.get("bundle_sha") is None:
        payload["bundle_sha"] = None

    # Read KILLCHAIN_EXPORT for extra data (normal path, not error path)
    try:
        killchain_path = chunk_output_dir / "KILLCHAIN_EXPORT.json"
        if killchain_path.exists():
            with open(killchain_path, encoding="utf-8") as f:
                kc = json.load(f)
            raw = kc.get("killchain_block_reason_counts") or {}
            payload["reject_reason_histogram"] = dict(sorted((str(k), int(v)) for k, v in raw.items()))
            payload["n_rejects_total"] = sum(payload["reject_reason_histogram"].values())
            payload["killchain_top_block_reason"] = kc.get("killchain_top_block_reason")
            if "session_histogram" in kc:
                payload["session_histogram"] = dict(kc.get("session_histogram") or {})
    except Exception as e:
        log.warning("[ZERO_TRADES_DIAG] Failed to read KILLCHAIN_EXPORT: %s", e)

    # Entry params / policy threshold
    try:
        if runner and hasattr(runner, "entry_params"):
            ep = runner.entry_params or {}
            if "META_THRESHOLD" in ep:
                payload["meta_threshold"] = float(ep.get("META_THRESHOLD"))
    except Exception:
        pass

    payload_json = convert_to_json_serializable(payload)
    if atomic_write_json_safe(out_path, payload_json):
        log.info("[ZERO_TRADES_DIAG] Wrote %s (0 trades)", out_path)
        return out_path
    return None
