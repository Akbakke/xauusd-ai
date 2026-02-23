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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
# Helpers (no-throw / json-safe)
# ============================================================

def _now_iso_utc() -> str:
    # Deterministic-ish format. (Value changes with time, but format and code path are stable.)
    return datetime.now(timezone.utc).isoformat()


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_bool(x: Any, default: Optional[bool] = None) -> Optional[bool]:
    try:
        if x is None:
            return default
        return bool(x)
    except Exception:
        return default


def _safe_str(x: Any, default: Optional[str] = None) -> Optional[str]:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _safe_enum_to_str(x: Any) -> Optional[str]:
    # Enum -> stable-ish string
    try:
        if x is None:
            return None
        if hasattr(x, "value"):
            return str(x.value)
        if hasattr(x, "name"):
            return str(x.name)
        return str(x)
    except Exception:
        return None


def _get_required_ctx_columns_from_runner(runner: Any) -> Tuple[List[str], List[str]]:
    """
    NO I/O: derive ctx contract columns from runner/runtime only.
    We accept multiple possible attribute names to avoid coupling.

    Returns:
      (required_cont: list[str], required_cat: list[str])
    """
    required_cont: List[str] = []
    required_cat: List[str] = []

    if runner is None:
        return required_cont, required_cat

    # Candidate attribute names (in priority order)
    cont_candidates = (
        "ctx_cont_required_columns",
        "ctx_cont_columns_required",
        "ctx_cont_feature_names",
        "ctx_cont_names",
        "ORDERED_CTX_CONT_NAMES",  # if runner exposes these
    )
    cat_candidates = (
        "ctx_cat_required_columns",
        "ctx_cat_columns_required",
        "ctx_cat_feature_names",
        "ctx_cat_names",
        "ORDERED_CTX_CAT_NAMES",
    )

    def _extract_list(attr_names: Tuple[str, ...]) -> List[str]:
        for a in attr_names:
            try:
                v = getattr(runner, a, None)
                if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                    return list(v)
            except Exception:
                continue
        return []

    required_cont = _extract_list(cont_candidates)
    required_cat = _extract_list(cat_candidates)

    # If dims exist, and lists are longer, caller may be masking; trim only if dims are sane.
    try:
        cont_dim = _safe_int(getattr(runner, "ctx_cont_dim", None), None)
        cat_dim = _safe_int(getattr(runner, "ctx_cat_dim", None), None)
        if cont_dim is not None and cont_dim > 0 and len(required_cont) >= cont_dim:
            required_cont = required_cont[:cont_dim]
        if cat_dim is not None and cat_dim > 0 and len(required_cat) >= cat_dim:
            required_cat = required_cat[:cat_dim]
    except Exception:
        pass

    return required_cont, required_cat


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

    NEVER raises.
    NO file I/O.
    """
    FEATURE_TIME_MEAN_MS_THRESHOLD = 5.0
    violations: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # Extract runner state (best-effort, never fail)
    # --------------------------------------------------------
    runner_prebuilt_used: Optional[bool] = None
    runner_replay_mode_enum: Optional[str] = None
    has_prebuilt_df = False

    lookup_attempts: Optional[int] = None
    lookup_hits: Optional[int] = None
    lookup_misses: Optional[int] = None
    prebuilt_lookup_phase: Optional[str] = None

    prebuilt_features_sha256: Optional[str] = None
    prebuilt_features_path_resolved: Optional[str] = None
    prebuilt_schema_version: Optional[str] = None

    if ctx.runner is not None:
        try:
            runner_prebuilt_used = _safe_bool(getattr(ctx.runner, "prebuilt_used", None), None)

            replay_enum = getattr(ctx.runner, "replay_mode_enum", None)
            runner_replay_mode_enum = _safe_enum_to_str(replay_enum)

            try:
                has_prebuilt_df = (
                    hasattr(ctx.runner, "prebuilt_features_df")
                    and getattr(ctx.runner, "prebuilt_features_df", None) is not None
                )
            except Exception:
                has_prebuilt_df = False

            lookup_attempts = _safe_int(getattr(ctx.runner, "lookup_attempts", None), None)
            lookup_hits = _safe_int(getattr(ctx.runner, "lookup_hits", None), None)
            lookup_misses = _safe_int(getattr(ctx.runner, "lookup_misses", None), None)
            prebuilt_lookup_phase = _safe_str(getattr(ctx.runner, "prebuilt_lookup_phase", None), None)

            # Loader metadata (if available)
            loader = getattr(ctx.runner, "prebuilt_features_loader", None)
            if loader is not None:
                prebuilt_features_sha256 = _safe_str(getattr(loader, "sha256", None), None)
                p = getattr(loader, "prebuilt_path_resolved", None)
                prebuilt_features_path_resolved = _safe_str(p, None)
                prebuilt_schema_version = _safe_str(getattr(loader, "schema_version", None), None)

            # Fallback metadata
            if prebuilt_features_sha256 is None:
                prebuilt_features_sha256 = _safe_str(getattr(ctx.runner, "prebuilt_features_sha256", None), None)

            if prebuilt_features_path_resolved is None:
                prebuilt_features_path_resolved = _safe_str(
                    getattr(ctx.runner, "prebuilt_features_path_resolved", None), None
                )

            if prebuilt_schema_version is None:
                prebuilt_schema_version = _safe_str(getattr(ctx.runner, "prebuilt_schema_version", None), None)

        except Exception as e:
            # Contract: never throw
            log.warning("[PREBUILT_INVARIANTS] Failed to read runner state: %s", e)

    # --------------------------------------------------------
    # Logging helper
    # --------------------------------------------------------
    def _log_violation(msg: str) -> None:
        try:
            if ctx.is_truth_or_smoke_worker:
                log.error(msg)
            else:
                log.warning(msg)
        except Exception:
            pass

    # --------------------------------------------------------
    # Invariant A
    # PREBUILT env enabled => runner.prebuilt_used must be True
    # --------------------------------------------------------
    if bool(ctx.prebuilt_enabled_env) and runner_prebuilt_used is not True:
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
    ft_mean = _safe_float(ctx.feature_time_mean_ms, 0.0) or 0.0
    if bool(ctx.prebuilt_enabled_env) and ft_mean > FEATURE_TIME_MEAN_MS_THRESHOLD:
        code = "INV_B_FEATURE_TIME_HIGH"
        msg = (
            "PREBUILT env enabled but feature_time_mean_ms="
            f"{ft_mean:.3f} > {FEATURE_TIME_MEAN_MS_THRESHOLD}"
        )
        violations.append({"code": code, "msg": msg})
        _log_violation(msg)

    # --------------------------------------------------------
    # Invariant C
    # PREBUILT env enabled => DF must exist
    # --------------------------------------------------------
    if bool(ctx.prebuilt_enabled_env) and not bool(has_prebuilt_df):
        code = "INV_C_PREBUILT_DF_MISSING"
        msg = "PREBUILT env enabled but runner.prebuilt_features_df is missing"
        violations.append({"code": code, "msg": msg})
        _log_violation(msg)

    # --------------------------------------------------------
    # Invariant D (TRUTH/SMOKE)
    # Prebuilt DF must contain all ctx contract columns derived from runner (NO I/O)
    # --------------------------------------------------------
    ctx_contract_missing: List[str] = []
    required_cont: List[str] = []
    required_cat: List[str] = []

    if bool(ctx.prebuilt_enabled_env) and bool(has_prebuilt_df) and bool(ctx.is_truth_or_smoke_worker):
        try:
            required_cont, required_cat = _get_required_ctx_columns_from_runner(ctx.runner)
            all_required = list(required_cont) + list(required_cat)

            df = getattr(ctx.runner, "prebuilt_features_df", None)
            df_cols = []
            try:
                if df is not None and hasattr(df, "columns"):
                    df_cols = list(df.columns)
            except Exception:
                df_cols = []

            if all_required and df_cols:
                ctx_contract_missing = [c for c in all_required if c not in df_cols]
            elif not all_required:
                # We cannot derive contract from runner; treat as a violation in TRUTH/SMOKE.
                ctx_contract_missing = ["<runner_ctx_contract_unavailable>"]
            else:
                ctx_contract_missing = ["<prebuilt_df_columns_unavailable>"]

            if ctx_contract_missing:
                code = "INV_D_CTX_CONTRACT_COLUMNS_MISSING"
                msg = (
                    "[PREBUILT_CTX_CONTRACT] TRUTH/SMOKE prebuilt missing ctx contract columns "
                    f"(derived from runner): missing={ctx_contract_missing}"
                )
                violations.append(
                    {"code": code, "msg": msg, "missing": list(ctx_contract_missing)}
                )
                _log_violation(msg)
            else:
                try:
                    log.info("[PREBUILT_CTX_CONTRACT] required cont+cat present; missing: []")
                except Exception:
                    pass
        except Exception as e:
            # Contract: never throw
            log.warning("[PREBUILT_CTX_CONTRACT] Failed to check ctx contract columns: %s", e)
            ctx_contract_missing = ["<check_failed>"]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    ok = len(violations) == 0

    if ok:
        summary = "OK"
    else:
        first_code = _safe_str(violations[0].get("code"), "UNKNOWN") or "UNKNOWN"
        summary = f"VIOLATION: {first_code}"
        if len(violations) > 1:
            summary += f" (+{len(violations) - 1} more)"

    # --------------------------------------------------------
    # Final Report (JSON-safe)
    # --------------------------------------------------------
    report: Dict[str, Any] = {
        "contract_id": "PREBUILT_INVARIANTS_V1",
        "report_version": 2,

        "ok": bool(ok),
        "summary": str(summary),

        "run_id": _safe_str(ctx.run_id, "") or "",
        "chunk_idx": _safe_int(ctx.chunk_idx, 0) or 0,
        "timestamp": _now_iso_utc(),

        # Context
        "status": _safe_str(ctx.status, None),
        "error": _safe_str(ctx.error, None),
        "bars_total": _safe_int(ctx.bars_total, 0) or 0,
        "is_truth_or_smoke_worker": bool(ctx.is_truth_or_smoke_worker),

        # Env vs runner
        "prebuilt_enabled_env": bool(ctx.prebuilt_enabled_env),
        "runner_prebuilt_used": runner_prebuilt_used,
        "runner_replay_mode_enum": runner_replay_mode_enum,
        "has_prebuilt_df": bool(has_prebuilt_df),

        # Feature timing
        "feature_time_mean_ms_threshold": float(FEATURE_TIME_MEAN_MS_THRESHOLD),
        "feature_time_mean_ms_observed": float(ft_mean),
        "feature_time_total_sec": float(_safe_float(ctx.feature_time_total_sec, 0.0) or 0.0),

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
        "ctx_contract_required_cont": list(required_cont),
        "ctx_contract_required_cat": list(required_cat),
        "ctx_contract_missing": list(ctx_contract_missing),
    }

    return ok, report