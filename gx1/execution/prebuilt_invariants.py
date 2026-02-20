"""
Prebuilt Invariants: Encapsulates all PREBUILT-invariant checks.

Contract:
- No file I/O.
- No exceptions.
- Deterministic output.
- Explicit separation between env-intent and runner-actual.
- JSON-safe report (no Path / enum leakage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from gx1.scripts.add_ctx_cont_columns_to_prebuilt import get_prebuilt_ctx_contract_columns
from gx1.utils.dt_module import now_iso as dt_now_iso

log = logging.getLogger(__name__)


# ============================================================
# Context
# ============================================================

@dataclass(frozen=True)
class PrebuiltInvariantContext:
    """
    Context for PREBUILT invariant checks.

    All data must come from runtime state.
    """

    chunk_idx: int
    run_id: str
    is_truth_or_smoke_worker: bool

    # Env intent
    prebuilt_enabled_env: bool  # GX1_REPLAY_USE_PREBUILT_FEATURES == "1"

    # Runner state
    runner: Any

    # Metrics (already computed upstream)
    feature_time_mean_ms: float
    feature_time_total_sec: float
    bars_total: int

    # Current chunk state
    status: str
    error: Optional[str] = None


# ============================================================
# Invariant Check
# ============================================================

def check_prebuilt_invariants(
    ctx: PrebuiltInvariantContext,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Perform all PREBUILT invariants.

    Returns:
        (ok: bool, report: dict)
    """

    FEATURE_TIME_MEAN_MS_THRESHOLD = 5.0
    violations = []

    # --------------------------------------------------------
    # Extract runner state (best-effort, never fail)
    # --------------------------------------------------------

    runner_prebuilt_used = None
    runner_replay_mode_enum = None
    has_prebuilt_df = False
    lookup_attempts = None
    lookup_hits = None
    lookup_misses = None
    prebuilt_lookup_phase = None
    prebuilt_features_sha256 = None
    prebuilt_features_path_resolved = None
    prebuilt_schema_version = None

    if ctx.runner:
        try:
            runner_prebuilt_used = getattr(ctx.runner, "prebuilt_used", None)

            # Enum -> stable string
            replay_enum = getattr(ctx.runner, "replay_mode_enum", None)
            if replay_enum is not None:
                if hasattr(replay_enum, "value"):
                    runner_replay_mode_enum = str(replay_enum.value)
                elif hasattr(replay_enum, "name"):
                    runner_replay_mode_enum = str(replay_enum.name)
                else:
                    runner_replay_mode_enum = str(replay_enum)

            has_prebuilt_df = (
                hasattr(ctx.runner, "prebuilt_features_df")
                and ctx.runner.prebuilt_features_df is not None
            )

            lookup_attempts = int(getattr(ctx.runner, "lookup_attempts", 0))
            lookup_hits = int(getattr(ctx.runner, "lookup_hits", 0))
            lookup_misses = int(getattr(ctx.runner, "lookup_misses", 0))
            prebuilt_lookup_phase = getattr(ctx.runner, "prebuilt_lookup_phase", None)

            # Loader metadata (if available)
            loader = getattr(ctx.runner, "prebuilt_features_loader", None)
            if loader:
                prebuilt_features_sha256 = getattr(loader, "sha256", None)
                p = getattr(loader, "prebuilt_path_resolved", None)
                prebuilt_features_path_resolved = str(p) if p is not None else None
                prebuilt_schema_version = getattr(loader, "schema_version", None)

            # Fallback metadata
            if prebuilt_features_sha256 is None:
                prebuilt_features_sha256 = getattr(ctx.runner, "prebuilt_features_sha256", None)

            if prebuilt_features_path_resolved is None:
                p = getattr(ctx.runner, "prebuilt_features_path_resolved", None)
                prebuilt_features_path_resolved = str(p) if p is not None else None

            if prebuilt_schema_version is None:
                prebuilt_schema_version = getattr(ctx.runner, "prebuilt_schema_version", None)

        except Exception as e:
            log.warning("[PREBUILT_INVARIANTS] Failed to read runner state: %s", e)

    # --------------------------------------------------------
    # Logging helper
    # --------------------------------------------------------

    def _log_violation(msg: str):
        if ctx.is_truth_or_smoke_worker:
            log.error(msg)
        else:
            log.warning(msg)

    # --------------------------------------------------------
    # Invariant A
    # PREBUILT env enabled => runner.prebuilt_used must be True
    # --------------------------------------------------------

    if ctx.prebuilt_enabled_env and runner_prebuilt_used is not True:
        code = "INV_A_PREBUILT_USED_FALSE"
        msg = (
            "PREBUILT env enabled but runner.prebuilt_used="
            f"{runner_prebuilt_used} (expected True)"
        )
        violations.append({"code": code, "msg": msg})
        _log_violation(msg)

    # --------------------------------------------------------
    # Invariant B
    # PREBUILT env enabled => feature time must be near zero
    # --------------------------------------------------------

    if ctx.prebuilt_enabled_env and ctx.feature_time_mean_ms > FEATURE_TIME_MEAN_MS_THRESHOLD:
        code = "INV_B_FEATURE_TIME_HIGH"
        msg = (
            "PREBUILT env enabled but feature_time_mean_ms="
            f"{ctx.feature_time_mean_ms:.3f} > {FEATURE_TIME_MEAN_MS_THRESHOLD}"
        )
        violations.append({"code": code, "msg": msg})
        _log_violation(msg)

    # --------------------------------------------------------
    # Invariant C
    # PREBUILT env enabled => DF must exist
    # --------------------------------------------------------

    if ctx.prebuilt_enabled_env and not has_prebuilt_df:
        code = "INV_C_PREBUILT_DF_MISSING"
        msg = "PREBUILT env enabled but runner.prebuilt_features_df is missing"
        violations.append({"code": code, "msg": msg})
        _log_violation(msg)

    # --------------------------------------------------------
    # Invariant D (TRUTH/SMOKE)
    # Prebuilt DF must contain all ctx contract columns (required_cont + required_cat)
    # --------------------------------------------------------

    ctx_contract_missing: List[str] = []
    if ctx.prebuilt_enabled_env and has_prebuilt_df and ctx.is_truth_or_smoke_worker:
        try:
            required_cont, required_cat = get_prebuilt_ctx_contract_columns()
            all_required = required_cont + required_cat
            df = getattr(ctx.runner, "prebuilt_features_df", None)
            if df is not None and hasattr(df, "columns"):
                ctx_contract_missing = [c for c in all_required if c not in df.columns]
            if ctx_contract_missing:
                code = "INV_D_CTX_CONTRACT_COLUMNS_MISSING"
                msg = (
                    "[PREBUILT_CTX_CONTRACT] TRUTH/SMOKE prebuilt missing ctx contract columns: "
                    f"missing={ctx_contract_missing}"
                )
                violations.append({"code": code, "msg": msg, "missing": ctx_contract_missing})
                _log_violation(msg)
            else:
                log.info("[PREBUILT_CTX_CONTRACT] required cont+cat present; missing: []")
        except Exception as e:
            log.warning("[PREBUILT_CTX_CONTRACT] Failed to check ctx contract columns: %s", e)
            ctx_contract_missing = ["check_failed"]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    ok = len(violations) == 0

    if ok:
        summary = "OK"
    else:
        first_code = violations[0]["code"]
        summary = f"VIOLATION: {first_code}"
        if len(violations) > 1:
            summary += f" (+{len(violations) - 1} more)"

    # --------------------------------------------------------
    # Final Report (JSON-safe)
    # --------------------------------------------------------

    report = {
        "contract_id": "PREBUILT_INVARIANTS_V1",
        "report_version": 1,

        "ok": ok,
        "summary": summary,

        "run_id": ctx.run_id,
        "chunk_idx": ctx.chunk_idx,
        "timestamp": dt_now_iso(),

        # Context
        "status": ctx.status,
        "error": ctx.error,
        "bars_total": ctx.bars_total,
        "is_truth_or_smoke_worker": ctx.is_truth_or_smoke_worker,

        # Env vs runner
        "prebuilt_enabled_env": ctx.prebuilt_enabled_env,
        "runner_prebuilt_used": runner_prebuilt_used,
        "runner_replay_mode_enum": runner_replay_mode_enum,
        "has_prebuilt_df": has_prebuilt_df,

        # Feature timing
        "feature_time_mean_ms_threshold": FEATURE_TIME_MEAN_MS_THRESHOLD,
        "feature_time_mean_ms_observed": ctx.feature_time_mean_ms,
        "feature_time_total_sec": ctx.feature_time_total_sec,

        # Lookup telemetry
        "lookup_attempts": lookup_attempts,
        "lookup_hits": lookup_hits,
        "lookup_misses": lookup_misses,
        "prebuilt_lookup_phase": prebuilt_lookup_phase,

        # Prebuilt metadata
        "prebuilt_features_sha256": prebuilt_features_sha256,
        "prebuilt_features_path_resolved": prebuilt_features_path_resolved,
        "prebuilt_schema_version": prebuilt_schema_version,

        # Violations
        "violations": violations,

        # Ctx contract (TRUTH/SMOKE)
        "ctx_contract_missing": ctx_contract_missing,
    }

    return ok, report