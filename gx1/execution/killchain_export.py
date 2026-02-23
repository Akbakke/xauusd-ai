"""
Killchain Export: Deterministic runtime snapshot of killchain / entry-funnel telemetry.

Contract:
- NO disk reads.
- NO private attribute access (no _internal fields).
- NO policy introspection.
- Runtime snapshot only (runner / entry_manager public state).
- Best-effort: never raises.
- Writes exactly one file: KILLCHAIN_EXPORT.json
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gx1.execution.chunk_failure import (
    atomic_write_json_safe,
    convert_to_json_serializable,
)
from gx1.utils.dt_module import now_iso as dt_now_iso

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class KillchainExportContext:
    chunk_output_dir: Path
    chunk_idx: int
    run_id: str
    dt_module_version: Optional[str]
    is_truth_or_smoke_worker: bool
    runner: Any
    status: str
    error: Optional[str] = None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def export_killchain(ctx: KillchainExportContext) -> Optional[Path]:
    """
    Best-effort deterministic export of runtime killchain state.

    Returns:
        Path if written, else None.
    Never raises.
    """
    try:
        payload: Dict[str, Any] = {
            "contract_id": "KILLCHAIN_EXPORT_V1",
            "run_id": ctx.run_id,
            "chunk_idx": ctx.chunk_idx,
            "timestamp": dt_now_iso(),
            "dt_module_version": ctx.dt_module_version,
            "status": ctx.status,
            "error": ctx.error,
        }

        # --------------------------------------------------------------
        # Entry Manager Snapshot
        # --------------------------------------------------------------

        em = getattr(ctx.runner, "entry_manager", None) if ctx.runner else None

        # ---------------- Killchain Core ----------------

        killchain_fields = {
            "killchain_n_entry_pred_total": 0,
            "killchain_n_above_threshold": 0,
            "killchain_n_after_session_guard": 0,
            "killchain_n_after_vol_guard": 0,
            "killchain_n_after_regime_guard": 0,
            "killchain_n_after_risk_sizing": 0,
            "killchain_n_trade_create_attempts": 0,
            "killchain_n_trade_created": 0,
        }

        killchain_block_reason_counts: Dict[str, int] = {}

        if em:
            try:
                for k in killchain_fields:
                    killchain_fields[k] = int(getattr(em, k, 0))

                raw_reasons = getattr(em, "killchain_block_reason_counts", {}) or {}
                killchain_block_reason_counts = {
                    str(k): int(v)
                    for k, v in sorted(raw_reasons.items(), key=lambda kv: kv[0])
                }
            except Exception as e:
                log.warning("[KILLCHAIN_EXPORT] Failed reading killchain core: %s", e)

        payload.update(convert_to_json_serializable(killchain_fields))
        payload["killchain_block_reason_counts"] = convert_to_json_serializable(
            killchain_block_reason_counts
        )

        # Deterministic top reason
        if killchain_block_reason_counts:
            top_reason = sorted(
                killchain_block_reason_counts.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[0][0]
        else:
            top_reason = None

        payload["killchain_top_block_reason"] = top_reason

        # --------------------------------------------------------------
        # Stage 2 (post-vol / post-score guards)
        # --------------------------------------------------------------

        stage2_fields = {
            "killchain_stage2_version": 1,
            "killchain_n_pred_available": 0,
            "killchain_n_pass_score_gate": 0,
            "killchain_n_block_below_threshold": 0,
            "killchain_n_block_spread_guard": 0,
            "killchain_n_block_cost_guard": 0,
            "killchain_n_block_session_time_guard": 0,
            "killchain_n_block_position_limit": 0,
            "killchain_n_block_cooldown": 0,
            "killchain_n_block_risk_guard": 0,
            "killchain_n_block_unknown_post_vol": 0,
        }

        if em:
            try:
                for k in stage2_fields:
                    stage2_fields[k] = int(getattr(em, k, stage2_fields[k]))
            except Exception as e:
                log.warning("[KILLCHAIN_EXPORT] Failed reading stage2 fields: %s", e)

        payload.update(convert_to_json_serializable(stage2_fields))

        # Deterministic top-3 post-vol block reasons
        post_reason_counts = {
            k: stage2_fields.get(k, 0)
            for k in [
                "killchain_n_block_spread_guard",
                "killchain_n_block_cost_guard",
                "killchain_n_block_session_time_guard",
                "killchain_n_block_position_limit",
                "killchain_n_block_cooldown",
                "killchain_n_block_risk_guard",
                "killchain_n_block_unknown_post_vol",
            ]
        }

        post_reason_counts = {k: v for k, v in post_reason_counts.items() if v > 0}

        top_post = sorted(
            post_reason_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:3]

        payload["killchain_stage2_top_post_blocks"] = convert_to_json_serializable(top_post)

        # --------------------------------------------------------------
        # Histogram (score distribution snapshot)
        # --------------------------------------------------------------

        hist_bins = None
        hist_counts = None
        hist_total = 0

        if em:
            try:
                hist_bins = getattr(em, "entry_score_hist_bins", None)
                hist_counts = getattr(em, "entry_score_hist_counts", None)
                hist_total = int(getattr(em, "entry_score_hist_total", 0) or 0)
            except Exception as e:
                log.warning("[KILLCHAIN_EXPORT] Failed reading histogram: %s", e)

        if hist_bins is None:
            hist_bins = [round(i / 100.0, 2) for i in range(101)]
        if hist_counts is None:
            hist_counts = [0 for _ in range(100)]

        payload["entry_score_hist_bins"] = convert_to_json_serializable(hist_bins)
        payload["entry_score_hist_counts"] = convert_to_json_serializable(hist_counts)
        payload["entry_score_hist_total"] = convert_to_json_serializable(hist_total)

        # --------------------------------------------------------------
        # Entry Funnel (runner-level)
        # --------------------------------------------------------------

        entry_funnel = {
            "bars_seen": 0,
            "bars_skipped_warmup": 0,
            "bars_skipped_pregate": 0,
            "bars_reaching_entry_stage": 0,
            "pregate_enabled": False,
        }

        if ctx.runner:
            try:
                entry_funnel["bars_seen"] = int(getattr(ctx.runner, "bars_seen", 0))
                entry_funnel["bars_skipped_warmup"] = int(getattr(ctx.runner, "bars_skipped_warmup", 0))
                entry_funnel["bars_skipped_pregate"] = int(getattr(ctx.runner, "bars_skipped_pregate", 0))
                entry_funnel["bars_reaching_entry_stage"] = int(
                    getattr(ctx.runner, "bars_reaching_entry_stage", 0)
                )
                entry_funnel["pregate_enabled"] = bool(
                    getattr(ctx.runner, "pregate_enabled", False)
                )
            except Exception as e:
                log.warning("[KILLCHAIN_EXPORT] Failed reading entry funnel: %s", e)

        payload["entry_funnel"] = convert_to_json_serializable(entry_funnel)

        # --------------------------------------------------------------
        # Canonical Stage Telemetry (if available)
        # --------------------------------------------------------------

        canonical_stage2 = {}
        canonical_stage3 = {}

        if em and hasattr(em, "entry_feature_telemetry"):
            try:
                t = em.entry_feature_telemetry

                canonical_stage2 = {
                    "stage2_total": getattr(t, "post_vol_guard_reached", None),
                    "stage2_pass": getattr(t, "stage2_pass", None),
                    "stage2_block": getattr(t, "stage2_block", None),
                }

                canonical_stage3 = {
                    "stage3_total": getattr(t, "post_score_gate_reached", None),
                    "stage3_pass": getattr(t, "stage3_pass", None),
                    "stage3_block": getattr(t, "stage3_block", None),
                }
            except Exception as e:
                log.warning("[KILLCHAIN_EXPORT] Failed reading canonical telemetry: %s", e)

        payload["canonical_stage2"] = convert_to_json_serializable(canonical_stage2)
        payload["canonical_stage3"] = convert_to_json_serializable(canonical_stage3)

        # --------------------------------------------------------------
        # Write
        # --------------------------------------------------------------

        export_path = ctx.chunk_output_dir / "KILLCHAIN_EXPORT.json"

        if atomic_write_json_safe(export_path, payload):
            log.info(
                "[KILLCHAIN_EXPORT] Wrote %s (chunk=%s)",
                export_path,
                ctx.chunk_idx,
            )
            return export_path

        log.warning("[KILLCHAIN_EXPORT] Failed writing %s", export_path)
        return None

    except Exception as e:
        log.error("[KILLCHAIN_EXPORT] Unexpected failure: %s", e, exc_info=True)
        return None