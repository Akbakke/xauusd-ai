"""
Chunk Bootstrap: Environment validation and preflight checks.

ONE UNIVERSE (locked):
- Signal bridge is 7/7 only.
- Context is 6/6 only (ctx_cont=6, ctx_cat=6).
- NO FALLBACKS. NO LEGACY. NO PARTIAL CONTEXT.
- TRUTH/SMOKE is strict and fail-fast (hard fail on any mismatch).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from gx1.contracts.signal_bridge_v1 import (
    ORDERED_CTX_CAT_NAMES_EXTENDED,
    ORDERED_CTX_CONT_NAMES_EXTENDED,
    SEQ_SIGNAL_DIM,
    SNAP_SIGNAL_DIM,
    get_canonical_ctx_contract,
)
from gx1.utils.canonical_prebuilt_resolver import resolve_base28_canonical_from_manifest
from gx1.execution.chunk_failure import (
    atomic_write_json_safe,
    convert_to_json_serializable,
    write_fatal_capsule,
)

log = logging.getLogger(__name__)

# ============================================================================
# ONE UNIVERSE CONSTANTS
# ============================================================================

# Canonical interpreter (SSoT)
REQUIRED_VENV = "/home/andre2/venvs/gx1/bin/python"

# ONE UNIVERSE dims
EXPECTED_XGB_FEATURES_LEN = 28
EXPECTED_CTX_CONT_DIM = 6
EXPECTED_CTX_CAT_DIM = 6
EXPECTED_SEQ_SIGNAL_DIM = 7
EXPECTED_SNAP_SIGNAL_DIM = 7

# Forbidden env vars (TRUTH/SMOKE: segmented/parallel must not exist)
_FORBIDDEN_ENV_VARS_TRUTH = (
    "GX1_SEGMENTED_PARALLEL",
    "GX1_SEGMENT_START",
    "GX1_SEGMENT_END",
    "GX1_PREROLL_START",
    "GX1_OWNER_START",
    "GX1_OWNER_END",
    "GX1_CHUNK_PLAN",
    "GX1_WORKERS",
    "GX1_CHUNKS",
)

# Files we emit here
_WORKER_BOOT_JSON = "WORKER_BOOT.json"
_WORKER_START_JSON = "WORKER_START.json"

# Canonical truth env
_CANONICAL_TRUTH_ENV = "GX1_CANONICAL_TRUTH_FILE"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _read_proc_mem_watermark_best_effort() -> Dict[str, Any]:
    """
    Best-effort memory watermark read (Linux only). Never raises.
    """
    out: Dict[str, Any] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["memory_vmrss_kb"] = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    out["memory_vmhwm_kb"] = int(line.split()[1])
    except Exception as e:
        out["memory_read_error"] = str(e)
    return out


def _ensure_truth_forbidden_env_clean(*, chunk_output_dir: Path, chunk_idx: int, run_id: str) -> None:
    """
    TRUTH/SMOKE: hard fail if forbidden env vars are present (even if set to "0").
    """
    present = {}
    for k in _FORBIDDEN_ENV_VARS_TRUTH:
        if k in os.environ:
            present[k] = os.environ.get(k)

    if present:
        fatal_msg = (
            f"[ENV_FORBIDDEN] [CHUNK {chunk_idx}] Forbidden env vars present in TRUTH/SMOKE: {sorted(present.keys())}"
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="ENV_FORBIDDEN_VARS_PRESENT",
            error_message=fatal_msg,
            extra_fields={"present": present},
        )
        raise RuntimeError(fatal_msg)


def _assert_truth_python_identity(*, chunk_output_dir: Path, chunk_idx: int, run_id: str, python_exe: str) -> None:
    """
    TRUTH/SMOKE: enforce canonical python interpreter.
    """
    if python_exe != REQUIRED_VENV:
        fatal_msg = f"[ENV_IDENTITY_FAIL] Wrong python interpreter. Expected: {REQUIRED_VENV}, Actual: {python_exe}"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="ENV_IDENTITY_FAIL",
            error_message=fatal_msg,
            extra_fields={"expected_python": REQUIRED_VENV, "actual_python": python_exe},
        )
        raise RuntimeError(fatal_msg)


def _assert_one_universe_signal_bridge(*, chunk_output_dir: Path, chunk_idx: int, run_id: str) -> None:
    """
    ONE UNIVERSE: signal bridge dims must be 7/7. No fallback.
    """
    if int(SEQ_SIGNAL_DIM) != EXPECTED_SEQ_SIGNAL_DIM or int(SNAP_SIGNAL_DIM) != EXPECTED_SNAP_SIGNAL_DIM:
        fatal_msg = (
            f"[SIGNAL_BRIDGE_DIM_FAIL] [CHUNK {chunk_idx}] "
            f"Expected SEQ_SIGNAL_DIM={EXPECTED_SEQ_SIGNAL_DIM}, SNAP_SIGNAL_DIM={EXPECTED_SNAP_SIGNAL_DIM} "
            f"but got SEQ_SIGNAL_DIM={int(SEQ_SIGNAL_DIM)}, SNAP_SIGNAL_DIM={int(SNAP_SIGNAL_DIM)}."
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="SIGNAL_BRIDGE_DIM_FAIL",
            error_message=fatal_msg,
            extra_fields={
                "SEQ_SIGNAL_DIM": int(SEQ_SIGNAL_DIM),
                "SNAP_SIGNAL_DIM": int(SNAP_SIGNAL_DIM),
                "expected_SEQ_SIGNAL_DIM": EXPECTED_SEQ_SIGNAL_DIM,
                "expected_SNAP_SIGNAL_DIM": EXPECTED_SNAP_SIGNAL_DIM,
            },
        )
        raise RuntimeError(fatal_msg)


def _run_prebuilt_smoke_test(
    *,
    chunk_idx: int,
    chunk_output_dir: Path,
    run_id: str,
    prebuilt_parquet_path_resolved: str,
    worker_start_info: Dict[str, Any],
    cwd: str,
) -> None:
    """
    Run prebuilt smoke test (pyarrow-only, deterministic retry).
    Verifies file is readable and has at least one row group and rows.
    """
    log.info(f"[CHUNK {chunk_idx}] [SELF_TEST] Running prebuilt smoke test (pyarrow-only)...")

    try:
        import pyarrow.parquet as pq  # type: ignore

        max_retries = 3
        retry_delay_base = 0.1  # 100ms base, deterministic
        smoke_test_passed = False

        last_error: Optional[BaseException] = None
        error_type: Optional[str] = None
        error_traceback: Optional[str] = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(retry_delay_base * (attempt + 1))

                parquet_file = pq.ParquetFile(prebuilt_parquet_path_resolved, memory_map=False)

                metadata = parquet_file.metadata
                num_row_groups = int(metadata.num_row_groups)
                if num_row_groups <= 0:
                    raise RuntimeError("Parquet file has 0 row groups")

                first_row_group = parquet_file.read_row_group(0)
                num_rows = int(len(first_row_group))
                if num_rows <= 0:
                    raise RuntimeError("First row group is empty")

                _ = first_row_group.slice(0, 1)

                log.info(
                    f"[CHUNK {chunk_idx}] [SELF_TEST] ✅ Smoke test passed (pyarrow): "
                    f"row_groups={num_row_groups}, first_row_group_rows={num_rows}"
                )
                smoke_test_passed = True
                break

            except FileNotFoundError as e:
                error_type = "file_not_found"
                last_error = e
                error_traceback = traceback.format_exc()
                log.warning(
                    f"[CHUNK {chunk_idx}] [SELF_TEST] Attempt {attempt + 1}/{max_retries} failed (file_not_found): {e}"
                )

            except PermissionError as e:
                error_type = "permission_denied"
                last_error = e
                error_traceback = traceback.format_exc()
                log.warning(
                    f"[CHUNK {chunk_idx}] [SELF_TEST] Attempt {attempt + 1}/{max_retries} failed (permission_denied): {e}"
                )

            except Exception as e:
                error_str = str(e).lower()
                if ("parquet" in error_str) or ("arrow" in error_str) or ("decode" in error_str):
                    error_type = "parquet_decode"
                else:
                    error_type = "unknown"
                last_error = e
                error_traceback = traceback.format_exc()
                log.warning(
                    f"[CHUNK {chunk_idx}] [SELF_TEST] Attempt {attempt + 1}/{max_retries} failed ({error_type}): {e}"
                )

        if not smoke_test_passed:
            fatal_msg = (
                f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Smoke test failed after {max_retries} attempts. "
                f"Error type={error_type}, last_error={last_error}"
            )
            worker_start_info["exception_full"] = (
                f"{error_type}: {str(last_error)}\n{error_traceback if error_traceback else '(no traceback captured)'}"
            )
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="PREBUILT_SMOKE_TEST_FAIL",
                error_message=fatal_msg,
                extra_fields={
                    "error_type": error_type,
                    "last_error": str(last_error),
                    "traceback": error_traceback if error_traceback else "(no traceback captured)",
                    "prebuilt_parquet_path_resolved": prebuilt_parquet_path_resolved,
                    "cwd": cwd,
                },
            )
            raise RuntimeError(fatal_msg)

    except Exception as e:
        fatal_msg = f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Worker self-test failed: {e}"
        worker_start_info["exception_full"] = f"{str(e)}\n{traceback.format_exc()}"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="PREBUILT_SMOKE_TEST_EXCEPTION",
            error_message=fatal_msg,
            extra_fields={
                "prebuilt_parquet_path_resolved": prebuilt_parquet_path_resolved,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "cwd": cwd,
            },
        )
        raise


def _load_json_file(path: Path) -> Dict[str, Any]:
    _require(path.exists(), f"[BOOTSTRAP] file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _forbid_prune(label: str, value: str, *, allow_ctx6cat6: bool = False) -> None:
    """Hard-fail if value contains forbidden legacy PRUNE markers."""
    upper = value.upper()
    tokens = ("PRUNE14", "PRUNE20", "V13_REFINED3_PRUNE", "REFINED3")
    if allow_ctx6cat6:
        forbidden = tokens
    else:
        forbidden = tokens + ("CTX6CAT6_",)
    if any(bad in upper for bad in forbidden):
        raise RuntimeError(f"LEGACY_PRUNE_FORBIDDEN_IN_TRUTH: {label}={value}")


def _one_universe_required_ctx_columns() -> List[str]:
    """
    ONE UNIVERSE: ctx_cont=6 and ctx_cat=6 always.
    Uses EXTENDED ordered lists, taking first 6 of each.
    """
    cont = list(ORDERED_CTX_CONT_NAMES_EXTENDED[:EXPECTED_CTX_CONT_DIM])
    cat = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:EXPECTED_CTX_CAT_DIM])
    _require(len(cont) == 6, f"[BOOTSTRAP] ORDERED_CTX_CONT_NAMES_EXTENDED[:6] len != 6 (len={len(cont)})")
    _require(len(cat) == 6, f"[BOOTSTRAP] ORDERED_CTX_CAT_NAMES_EXTENDED[:6] len != 6 (len={len(cat)})")
    return cont + cat


def _load_canonical_truth_file(*, chunk_output_dir: Path, chunk_idx: int, run_id: str) -> tuple[Path, Dict[str, Any]]:
    truth_path = os.environ.get(_CANONICAL_TRUTH_ENV)
    if not truth_path:
        fatal_msg = f"[TRUTH_FILE_MISSING_ENV] env {_CANONICAL_TRUTH_ENV} is not set"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="CANONICAL_TRUTH_FILE_MISSING",
            error_message=fatal_msg,
            extra_fields={"env": _CANONICAL_TRUTH_ENV},
        )
        raise RuntimeError(fatal_msg)

    p = Path(truth_path).expanduser().resolve()
    if not p.is_file():
        fatal_msg = f"[TRUTH_FILE_NOT_FOUND] path={str(p)}"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="CANONICAL_TRUTH_FILE_NOT_FOUND",
            error_message=fatal_msg,
            extra_fields={"path": str(p)},
        )
        raise RuntimeError(fatal_msg)

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        fatal_msg = f"[TRUTH_FILE_INVALID_JSON] path={str(p)} err={e}"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="CANONICAL_TRUTH_FILE_INVALID_JSON",
            error_message=fatal_msg,
            extra_fields={"path": str(p), "traceback": traceback.format_exc()},
        )
        raise RuntimeError(fatal_msg) from e

    if not isinstance(obj, dict):
        fatal_msg = f"[TRUTH_FILE_INVALID_JSON] expected dict at top-level, got {type(obj)}"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="CANONICAL_TRUTH_FILE_INVALID_JSON",
            error_message=fatal_msg,
            extra_fields={"path": str(p), "type": str(type(obj))},
        )
        raise RuntimeError(fatal_msg)

    return p, obj


def _truth_get_prebuilt_required_columns(*, truth_obj: Dict[str, Any]) -> List[str]:
    cols = truth_obj.get("prebuilt_required_columns")
    _require(
        isinstance(cols, list) and cols and all(isinstance(x, str) and x for x in cols),
        "TRUTH_PREBUILT_REQUIRED_COLUMNS_MISSING_OR_INVALID: expected non-empty list[str]",
    )
    if len(set(cols)) != len(cols):
        raise RuntimeError("TRUTH_PREBUILT_REQUIRED_COLUMNS_HAS_DUPLICATES")
    return list(cols)


def _truth_resolve_prebuilt_parquet_path(*, truth_obj: Dict[str, Any]) -> str:
    raw = truth_obj.get("canonical_prebuilt_parquet")
    _require(isinstance(raw, str) and raw.strip(), "TRUTH_CANONICAL_PREBUILT_PARQUET_MISSING_OR_INVALID")
    p = Path(raw).expanduser().resolve()
    _require(p.is_file(), f"TRUTH_CANONICAL_PREBUILT_PARQUET_NOT_FOUND: path={str(p)}")
    return str(p)


def _truth_validate_one_universe_contract(
    *,
    chunk_output_dir: Path,
    chunk_idx: int,
    run_id: str,
    truth_obj: Dict[str, Any],
    prebuilt_required_columns: List[str],
) -> Dict[str, Any]:
    """
    ONE UNIVERSE validation (no fallback):
    - canonical_xgb_bundle_dir must exist and have MASTER_MODEL_LOCK.json with ordered_features len=28.
    - canonical_transformer_bundle_dir must exist and have MASTER_TRANSFORMER_LOCK.json with ctx_cont_dim=6 ctx_cat_dim=6.
    - prebuilt_required_columns from truth must:
        - start with ordered_features (exact order)
        - contain ctx12 (ORDERED_CTX_CONT_NAMES_EXTENDED[:6] + ORDERED_CTX_CAT_NAMES_EXTENDED[:6])
    """
    canonical_xgb_bundle_dir_str = str(truth_obj.get("canonical_xgb_bundle_dir") or "").strip()
    canonical_transformer_bundle_dir_str = str(truth_obj.get("canonical_transformer_bundle_dir") or "").strip()

    _require(canonical_xgb_bundle_dir_str, "canonical_xgb_bundle_dir missing in truth config")
    _require(
        canonical_transformer_bundle_dir_str,
        "canonical_transformer_bundle_dir missing in truth config (ONE UNIVERSE requires it)",
    )

    canonical_xgb_bundle_dir = Path(canonical_xgb_bundle_dir_str).expanduser().resolve()
    canonical_transformer_bundle_dir = Path(canonical_transformer_bundle_dir_str).expanduser().resolve()

    # PRUNE kill-switch
    if "PRUNE" in str(canonical_xgb_bundle_dir).upper():
        raise RuntimeError(f"PRUNE bundles are forbidden in canonical TRUTH: {canonical_xgb_bundle_dir}")
    if "PRUNE" in str(canonical_transformer_bundle_dir).upper():
        raise RuntimeError(f"PRUNE bundles are forbidden in canonical TRUTH: {canonical_transformer_bundle_dir}")

    lock_path = canonical_xgb_bundle_dir / "MASTER_MODEL_LOCK.json"
    _require(lock_path.exists(), f"MASTER_MODEL_LOCK.json not found: {lock_path}")
    lock_obj = _load_json_file(lock_path)

    ordered = lock_obj.get("ordered_features")
    _require(
        isinstance(ordered, list) and all(isinstance(x, str) for x in ordered),
        "MASTER_MODEL_LOCK.ordered_features missing or not list[str]",
    )

    n = len(ordered)
    if n != EXPECTED_XGB_FEATURES_LEN:
        fatal_msg = (
            f"[NON_CANON_XGB] [CHUNK {chunk_idx}] ordered_features_len={n} (expected {EXPECTED_XGB_FEATURES_LEN}). "
            f"bundle_dir={canonical_xgb_bundle_dir}"
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="NON_CANON_XGB",
            error_message=fatal_msg,
            extra_fields={"ordered_features_len": n, "canonical_xgb_bundle_dir": str(canonical_xgb_bundle_dir)},
        )
        raise RuntimeError(fatal_msg)

    # Transformer lock must exist and must be 6/6
    trans_lock_path = canonical_transformer_bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
    _require(trans_lock_path.exists(), f"MASTER_TRANSFORMER_LOCK.json missing: {trans_lock_path}")
    trans_lock_obj = _load_json_file(trans_lock_path)

    ctx_cont_dim = trans_lock_obj.get("ctx_cont_dim")
    ctx_cat_dim = trans_lock_obj.get("ctx_cat_dim")
    if int(ctx_cont_dim) != EXPECTED_CTX_CONT_DIM or int(ctx_cat_dim) != EXPECTED_CTX_CAT_DIM:
        fatal_msg = (
            f"[NON_CANON_TRANSFORMER] [CHUNK {chunk_idx}] Expected ctx_cont_dim=6 ctx_cat_dim=6 "
            f"but got ctx_cont_dim={ctx_cont_dim} ctx_cat_dim={ctx_cat_dim}. "
            f"bundle_dir={canonical_transformer_bundle_dir}"
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="NON_CANON_TRANSFORMER",
            error_message=fatal_msg,
            extra_fields={
                "ctx_cont_dim": ctx_cont_dim,
                "ctx_cat_dim": ctx_cat_dim,
                "canonical_transformer_bundle_dir": str(canonical_transformer_bundle_dir),
            },
        )
        raise RuntimeError(fatal_msg)

    ordered_features = list(ordered)

    # Truth columns must start with ordered_features in exact order (no partial/no reordering)
    if prebuilt_required_columns[: len(ordered_features)] != ordered_features:
        fatal_msg = (
            f"[TRUTH_PREBUILT_COLUMNS_MISMATCH] [CHUNK {chunk_idx}] prebuilt_required_columns does not start with "
            f"MASTER_MODEL_LOCK.ordered_features (exact order required)."
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="TRUTH_PREBUILT_COLUMNS_MISMATCH",
            error_message=fatal_msg,
            extra_fields={
                "expected_prefix_len": len(ordered_features),
                "expected_prefix_head": ordered_features[:10],
                "actual_prefix_head": prebuilt_required_columns[:10],
                "canonical_xgb_bundle_dir": str(canonical_xgb_bundle_dir),
            },
        )
        raise RuntimeError(fatal_msg)

    required_ctx_12 = _one_universe_required_ctx_columns()
    missing_ctx = [c for c in required_ctx_12 if c not in set(prebuilt_required_columns)]
    if missing_ctx:
        fatal_msg = f"[TRUTH_PREBUILT_COLUMNS_MISSING_CTX] [CHUNK {chunk_idx}] missing ctx columns: {missing_ctx}"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="TRUTH_PREBUILT_COLUMNS_MISSING_CTX",
            error_message=fatal_msg,
            extra_fields={"missing_ctx": missing_ctx},
        )
        raise RuntimeError(fatal_msg)

    return {
        "canonical_xgb_bundle_dir": str(canonical_xgb_bundle_dir),
        "canonical_transformer_bundle_dir": str(canonical_transformer_bundle_dir),
        "expected_ctx_cont_dim": EXPECTED_CTX_CONT_DIM,
        "expected_ctx_cat_dim": EXPECTED_CTX_CAT_DIM,
    }


@dataclass(frozen=True)
class BootstrapContext:
    """Context containing all bootstrap variables needed for replay execution."""

    # Chunk identity
    chunk_idx: int

    # Paths
    policy_path: Path
    data_path: Path
    output_dir: Path
    chunk_output_dir: Path
    run_id: str

    # Prebuilt
    prebuilt_parquet_path: Optional[str]
    prebuilt_parquet_path_resolved: Optional[str]
    prebuilt_enabled: bool

    # Identity
    bundle_sha256: Optional[str]
    policy_id: Optional[str]
    canonical_truth_file: Optional[Path]

    # Flags
    is_truth_or_smoke_worker: bool
    telemetry_required: bool
    dt_module_version: str

    # Worker info
    worker_start_time: float
    worker_boot_payload: Dict[str, Any]
    worker_start_info: Dict[str, Any]

    # Timing (TRUTH-only)
    t_init_s: float
    t_init_start: Optional[float]

    # Diagnostic info
    cwd: str
    python_exe: str
    output_dir_env: str
    prebuilt_env: str

    # Chunk configuration
    chunk_local_padding_days: int

    # 1W1C (TRUTH locked)
    workers: int = 1
    chunks: int = 1

    # Prebuilt column contract (SSoT)
    prebuilt_required_columns: Optional[List[str]] = None


def bootstrap_chunk_environment(
    *,
    chunk_idx: int,
    chunk_start,
    chunk_end,
    data_path: Path,
    policy_path: Path,
    run_id: str,
    output_dir: Path,
    prebuilt_parquet_path: Optional[str],
    bundle_dir: Optional[Path],
    chunk_local_padding_days: int,
    bundle_sha256: Optional[str] = None,
    policy_id: Optional[str] = None,
) -> BootstrapContext:
    """
    Bootstrap chunk environment: validate, check gates, write WORKER_BOOT/WORKER_START.

    ONE UNIVERSE:
    - Requires signal bridge dims 7/7
    - Requires transformer ctx dims 6/6 (from MASTER_TRANSFORMER_LOCK in canonical bundle dir from truth)
    - Requires prebuilt_required_columns to be sourced from canonical truth config (no fallback)
    - Requires prebuilt parquet path to be sourced/resolved from canonical truth config (no fallback)
    """
    # Determine run mode
    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    is_truth_or_smoke_worker = run_mode in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"

    # TRUTH-only init timing: start as early as possible (fix #2)
    t_init_start: Optional[float] = time.time() if is_truth_or_smoke_worker else None

    # Resolve chunk output dir (absolute)
    chunk_output_dir = (Path(output_dir) / f"chunk_{chunk_idx}").resolve()
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    # Prefork freeze verification (TRUTH/SMOKE only)
    if is_truth_or_smoke_worker:
        prefork_freeze_path = Path(output_dir) / "PRE_FORK_FREEZE.json"
        if not prefork_freeze_path.exists():
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="WORKER_PREFORK_MISSING",
                error_message=f"PRE_FORK_FREEZE.json not found in run root: {output_dir}",
                extra_fields={"output_dir": str(output_dir)},
            )
            raise RuntimeError(
                "[WORKER_PREFORK_MISSING] PRE_FORK_FREEZE.json not found. Worker cannot start without master freeze."
            )

    # dt_module stamping + validation (fail-fast)
    from gx1.utils.dt_module import (  # local import is fine; core infrastructure
        get_dt_module_version,
        validate_dt_module_version,
        now_iso as dt_now_iso,
    )

    validate_dt_module_version()
    dt_module_version = get_dt_module_version()

    # Diagnostics
    cwd = str(Path.cwd())
    python_exe = sys.executable
    output_dir_env = os.getenv("GX1_OUTPUT_DIR", "NOT_SET")
    prebuilt_env = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "NOT_SET")

    # ONE UNIVERSE: always assert signal bridge 7/7 in TRUTH/SMOKE
    if is_truth_or_smoke_worker:
        _assert_one_universe_signal_bridge(chunk_output_dir=chunk_output_dir, chunk_idx=chunk_idx, run_id=run_id)

    # TRUTH/SMOKE: hard enforce identity + forbid env vars
    if is_truth_or_smoke_worker:
        _assert_truth_python_identity(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            python_exe=python_exe,
        )
        _ensure_truth_forbidden_env_clean(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
        )

    # Prebuilt mode (env is the truth of intent)
    prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"

    # Fix #1: TRUTH/SMOKE must require prebuilt-enabled (fail early, not later in loader)
    if is_truth_or_smoke_worker and not prebuilt_enabled:
        fatal_msg = (
            f"[TRUTH_PREBUILT_REQUIRED] [CHUNK {chunk_idx}] "
            "GX1_REPLAY_USE_PREBUILT_FEATURES must be 1 in TRUTH/SMOKE"
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="TRUTH_PREBUILT_REQUIRED",
            error_message=fatal_msg,
            extra_fields={"GX1_REPLAY_USE_PREBUILT_FEATURES": os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES")},
        )
        raise RuntimeError(fatal_msg)

    # Enforce type discipline on arg (even if ignored in TRUTH SSoT path)
    if prebuilt_parquet_path is not None and not isinstance(prebuilt_parquet_path, str):
        fatal_msg = (
            f"[PREBUILT_TYPE_FAIL] [CHUNK {chunk_idx}] prebuilt_parquet_path must be str, "
            f"got {type(prebuilt_parquet_path)} repr={repr(prebuilt_parquet_path)}"
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="PREBUILT_TYPE_FAIL",
            error_message=fatal_msg,
            extra_fields={"type": str(type(prebuilt_parquet_path)), "repr": repr(prebuilt_parquet_path)},
        )
        raise RuntimeError(fatal_msg)

    # Canonical truth file + TRUTH SSoT derivations (only required when prebuilt is enabled)
    canonical_truth_file_val: Optional[Path] = None
    truth_obj: Optional[Dict[str, Any]] = None

    prebuilt_required_columns_val: Optional[List[str]] = None
    prebuilt_parquet_path_resolved: Optional[str] = None

    canonical_xgb_bundle_dir_str: Optional[str] = None
    canonical_transformer_bundle_dir_str: Optional[str] = None
    expected_ctx_cont_dim: Optional[int] = None
    expected_ctx_cat_dim: Optional[int] = None

    if prebuilt_enabled:
        # Load truth (no fallback)
        canonical_truth_file_val, truth_obj = _load_canonical_truth_file(
            chunk_output_dir=chunk_output_dir, chunk_idx=chunk_idx, run_id=run_id
        )

        try:
            # PRUNE kill-switch on truth-config paths (before IO)
            for key in ("canonical_xgb_bundle_dir", "canonical_prebuilt_parquet", "canonical_transformer_bundle_dir"):
                val = str(truth_obj.get(key) or "").strip()
                if val:
                    _forbid_prune(key, val, allow_ctx6cat6=(key == "canonical_transformer_bundle_dir"))

            # Manifest-only resolution (TRUTH/SMOKE): resolve from BASE28_CANONICAL/CURRENT_MANIFEST.json
            manifest_path = Path("/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json")
            truth_manifest_raw = str(truth_obj.get("canonical_prebuilt_manifest") or "").strip()
            if truth_manifest_raw:
                if Path(truth_manifest_raw).expanduser().resolve() != manifest_path.expanduser().resolve():
                    raise RuntimeError(
                        f"PREBUILT_MANIFEST_SPLIT_BRAIN: truth_manifest={truth_manifest_raw} expected={manifest_path}"
                    )
            manifest_info = resolve_base28_canonical_from_manifest(manifest_path)
            prebuilt_parquet_path_resolved = manifest_info["parquet_path"]

            # PRUNE kill-switch on manifest/parquet paths
            _forbid_prune("manifest_path", str(manifest_path))
            _forbid_prune("parquet_path", prebuilt_parquet_path_resolved)

            # TRUTH/SMOKE: caller-supplied prebuilt path is forbidden (manifest is the only source of truth)
            if prebuilt_parquet_path:
                raise RuntimeError(
                    f"SPLIT_BRAIN_PREBUILT: TRUTH requires manifest-only; caller prebuilt path forbidden "
                    f"(caller={Path(prebuilt_parquet_path).expanduser().resolve()})"
                )

            # Split-brain guard: truth canonical_prebuilt_parquet (if present) must match manifest parquet
            truth_parquet_raw = str(truth_obj.get("canonical_prebuilt_parquet") or "").strip()
            if not truth_parquet_raw:
                raise RuntimeError("canonical_prebuilt_parquet missing in truth (must mirror manifest)")
            truth_parquet_resolved = str(Path(truth_parquet_raw).expanduser().resolve())
            if truth_parquet_resolved != prebuilt_parquet_path_resolved:
                raise RuntimeError(
                    f"SPLIT_BRAIN_PREBUILT: truth_parquet={truth_parquet_resolved} manifest_parquet={prebuilt_parquet_path_resolved}"
                )

            # PRUNE kill-switch on resolved parquet path
            upp = prebuilt_parquet_path_resolved.upper()
            if "PRUNE14" in upp or "PRUNE20" in upp:
                raise RuntimeError(f"PRUNE_PREBUILT_FORBIDDEN_IN_TRUTH: {prebuilt_parquet_path_resolved}")

            # Load required columns from schema_manifest adjacent to resolved parquet
            schema_path = Path(prebuilt_parquet_path_resolved).with_suffix(".schema_manifest.json")
            _require(schema_path.exists(), f"TRUTH_PREBUILT_SCHEMA_MANIFEST_NOT_FOUND: path={schema_path}")
            schema_obj = _load_json_file(schema_path)
            prebuilt_required_columns_val = list(schema_obj.get("required_all_features") or [])
            _require(
                isinstance(prebuilt_required_columns_val, list)
                and prebuilt_required_columns_val
                and all(isinstance(x, str) and x for x in prebuilt_required_columns_val),
                "TRUTH_PREBUILT_REQUIRED_COLUMNS_MISSING_OR_INVALID (from schema_manifest.required_all_features)",
            )
            if len(set(prebuilt_required_columns_val)) != len(prebuilt_required_columns_val):
                raise RuntimeError("TRUTH_PREBUILT_REQUIRED_COLUMNS_HAS_DUPLICATES (from schema_manifest)")
            if "time" in prebuilt_required_columns_val:
                # Schema manifest may include time (on-disk column). TRUTH contract excludes time; drop and continue.
                prebuilt_required_columns_val = [c for c in prebuilt_required_columns_val if c != "time"]

            # Validate ONE UNIVERSE contract (locks + ctx dims + columns prefix)
            contract = _truth_validate_one_universe_contract(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                truth_obj=truth_obj,
                prebuilt_required_columns=prebuilt_required_columns_val,
            )
            canonical_xgb_bundle_dir_str = contract["canonical_xgb_bundle_dir"]
            canonical_transformer_bundle_dir_str = contract["canonical_transformer_bundle_dir"]
            expected_ctx_cont_dim = int(contract["expected_ctx_cont_dim"])
            expected_ctx_cat_dim = int(contract["expected_ctx_cat_dim"])

            # ONE SSoT: ctx dims must be 6/6 (CTX6CAT6), sourced from bundle/truth.
            canonical_ctx = get_canonical_ctx_contract()
            if expected_ctx_cont_dim != canonical_ctx["ctx_cont_dim"] or expected_ctx_cat_dim != canonical_ctx["ctx_cat_dim"]:
                raise RuntimeError(
                    f"[CTX_CONTRACT_SPLIT_BRAIN] expected_ctx_cont_dim={expected_ctx_cont_dim} "
                    f"expected_ctx_cat_dim={expected_ctx_cat_dim} canonical={canonical_ctx['ctx_cont_dim']}/{canonical_ctx['ctx_cat_dim']} "
                    f"source={contract.get('ctx_contract_source','unknown')}"
                )

            # Fix #4: bundle_dir split-brain guard in TRUTH/SMOKE
            if is_truth_or_smoke_worker and bundle_dir is not None:
                caller_bundle = str(Path(bundle_dir).expanduser().resolve())
                # Allow only exact canonical dirs (strict)
                if caller_bundle not in (canonical_xgb_bundle_dir_str, canonical_transformer_bundle_dir_str):
                    fatal_msg = (
                        f"[BUNDLE_SPLIT_BRAIN] [CHUNK {chunk_idx}] caller bundle_dir does not match canonical dirs. "
                        f"caller={caller_bundle} canonical_xgb={canonical_xgb_bundle_dir_str} "
                        f"canonical_transformer={canonical_transformer_bundle_dir_str}"
                    )
                    write_fatal_capsule(
                        chunk_output_dir=chunk_output_dir,
                        chunk_idx=chunk_idx,
                        run_id=run_id,
                        fatal_reason="BUNDLE_SPLIT_BRAIN",
                        error_message=fatal_msg,
                        extra_fields={
                            "caller_bundle_dir_resolved": caller_bundle,
                            "canonical_xgb_bundle_dir": canonical_xgb_bundle_dir_str,
                            "canonical_transformer_bundle_dir": canonical_transformer_bundle_dir_str,
                        },
                    )
                    raise RuntimeError(fatal_msg)

        except Exception as e:
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="TRUTH_SSOT_BOOTSTRAP_FAIL",
                error_message=str(e),
                extra_fields={
                    "canonical_truth_file": str(canonical_truth_file_val) if canonical_truth_file_val else None,
                    "traceback": traceback.format_exc(),
                },
            )
            raise RuntimeError(f"[TRUTH_NO_FALLBACK] bootstrap failed: {e}") from e

        # Split-brain guard: if caller passed a path, it must match truth (after resolve)
        if prebuilt_parquet_path:
            caller_resolved = str(Path(prebuilt_parquet_path).expanduser().resolve())
            if caller_resolved != prebuilt_parquet_path_resolved:
                fatal_msg = (
                    f"[PREBUILT_SPLIT_BRAIN] [CHUNK {chunk_idx}] caller prebuilt path does not match truth canonical_prebuilt_parquet. "
                    f"caller_resolved={caller_resolved} truth_resolved={prebuilt_parquet_path_resolved}"
                )
                write_fatal_capsule(
                    chunk_output_dir=chunk_output_dir,
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                    fatal_reason="PREBUILT_SPLIT_BRAIN",
                    error_message=fatal_msg,
                    extra_fields={"caller_resolved": caller_resolved, "truth_resolved": prebuilt_parquet_path_resolved},
                )
                raise RuntimeError(fatal_msg)

    # Existence + size (deterministic: only the resolved path matters)
    prebuilt_exists = False
    prebuilt_size = 0
    if prebuilt_parquet_path_resolved:
        p = Path(prebuilt_parquet_path_resolved)
        prebuilt_exists = p.exists()
        if prebuilt_exists:
            prebuilt_size = int(p.stat().st_size)

    # Build WORKER_BOOT payload (always written; TRUTH/SMOKE requires write success)
    worker_boot_payload: Dict[str, Any] = {
        "dt_module_version": dt_module_version,
        "timestamp": dt_now_iso(),
        "pid": int(os.getpid()),
        "ppid": int(os.getppid()) if hasattr(os, "getppid") else None,
        "cwd": cwd,
        "sys_executable": python_exe,
        "argv_snapshot": list(sys.argv) if hasattr(sys, "argv") else None,
        "chunk_output_dir": str(chunk_output_dir),
        "output_dir": str(Path(output_dir).resolve()),
        "GX1_OUTPUT_DIR": output_dir_env,
        "GX1_RUN_MODE": run_mode,
        "GX1_REPLAY_USE_PREBUILT_FEATURES": os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0"),
        "prebuilt_parquet_path_raw": str(prebuilt_parquet_path) if prebuilt_parquet_path else None,
        "prebuilt_parquet_path_resolved": prebuilt_parquet_path_resolved,
        "prebuilt_exists": bool(prebuilt_exists),
        "prebuilt_size": int(prebuilt_size),
        "chunk_idx": int(chunk_idx),
        "run_id": str(run_id),
        "chunk_start": str(chunk_start),
        "chunk_end": str(chunk_end),
        # ONE UNIVERSE proof
        "signal_bridge_seq_dim": int(SEQ_SIGNAL_DIM),
        "signal_bridge_snap_dim": int(SNAP_SIGNAL_DIM),
        "expected_signal_bridge_seq_dim": EXPECTED_SEQ_SIGNAL_DIM,
        "expected_signal_bridge_snap_dim": EXPECTED_SNAP_SIGNAL_DIM,
        "canonical_truth_file": str(canonical_truth_file_val) if canonical_truth_file_val else None,
        "canonical_xgb_bundle_dir": canonical_xgb_bundle_dir_str,
        "canonical_transformer_bundle_dir": canonical_transformer_bundle_dir_str,
        "expected_ctx_cont_dim": expected_ctx_cont_dim,
        "expected_ctx_cat_dim": expected_ctx_cat_dim,
        "canonical_ctx_contract": get_canonical_ctx_contract(),
        "prebuilt_required_columns": prebuilt_required_columns_val,
    }
    worker_boot_payload.update(_read_proc_mem_watermark_best_effort())

    boot_path = chunk_output_dir / _WORKER_BOOT_JSON
    boot_ok = atomic_write_json_safe(boot_path, convert_to_json_serializable(worker_boot_payload))
    if not boot_ok and is_truth_or_smoke_worker:
        fatal_msg = f"[BOOT_WRITE_FAIL] [CHUNK {chunk_idx}] Failed to write {_WORKER_BOOT_JSON}"
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="BOOT_WRITE_FAIL",
            error_message=fatal_msg,
            extra_fields={"path": str(boot_path)},
        )
        raise RuntimeError(fatal_msg)
    elif not boot_ok:
        log.warning(f"[CHUNK {chunk_idx}] Failed to write {_WORKER_BOOT_JSON} (non-fatal outside TRUTH/SMOKE)")

    # Build WORKER_START info
    worker_start_info: Dict[str, Any] = {
        "chunk_id": int(chunk_idx),
        "run_id": str(run_id),
        "timestamp": dt_now_iso(),
        "prebuilt_parquet_path_raw": str(prebuilt_parquet_path) if prebuilt_parquet_path else None,
        "prebuilt_parquet_path_raw_type": type(prebuilt_parquet_path).__name__ if prebuilt_parquet_path else "None",
        "prebuilt_parquet_path_resolved": prebuilt_parquet_path_resolved,
        "exists": bool(prebuilt_exists),
        "size": int(prebuilt_size),
        "exception_full": None,
    }

    start_path = chunk_output_dir / _WORKER_START_JSON
    _ = atomic_write_json_safe(start_path, convert_to_json_serializable(worker_start_info))

    # If prebuilt enabled: enforce exists/size + run smoke test
    if prebuilt_enabled:
        if not prebuilt_parquet_path_resolved:
            fatal_msg = f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] prebuilt_enabled but resolved path is None"
            worker_start_info["exception_full"] = fatal_msg
            atomic_write_json_safe(start_path, convert_to_json_serializable(worker_start_info))
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="PREBUILT_RESOLVED_NONE",
                error_message=fatal_msg,
                extra_fields={"prebuilt_parquet_path": str(prebuilt_parquet_path)},
            )
            raise RuntimeError(fatal_msg)

        if not prebuilt_exists:
            fatal_msg = (
                f"[PREBUILT_FILE_NOT_FOUND] [CHUNK {chunk_idx}] Prebuilt parquet does not exist: {prebuilt_parquet_path_resolved}"
            )
            worker_start_info["exception_full"] = fatal_msg
            atomic_write_json_safe(start_path, convert_to_json_serializable(worker_start_info))
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="PREBUILT_FILE_NOT_FOUND",
                error_message=fatal_msg,
                extra_fields={"prebuilt_parquet_path_resolved": prebuilt_parquet_path_resolved, "cwd": cwd},
            )
            raise FileNotFoundError(fatal_msg)

        if prebuilt_size <= 0:
            fatal_msg = f"[PREBUILT_FILE_EMPTY] [CHUNK {chunk_idx}] Prebuilt parquet size=0: {prebuilt_parquet_path_resolved}"
            worker_start_info["exception_full"] = fatal_msg
            atomic_write_json_safe(start_path, convert_to_json_serializable(worker_start_info))
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="PREBUILT_FILE_EMPTY",
                error_message=fatal_msg,
                extra_fields={"prebuilt_parquet_path_resolved": prebuilt_parquet_path_resolved, "size": prebuilt_size},
            )
            raise RuntimeError(fatal_msg)

        log.info(f"[CHUNK {chunk_idx}] [SSoT] prebuilt_enabled=1, resolved={prebuilt_parquet_path_resolved}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT] prebuilt exists=True size={prebuilt_size:,} bytes")
        log.info(f"[CHUNK {chunk_idx}] [SSoT] truth_prebuilt_required_columns_len={len(prebuilt_required_columns_val or [])}")

        _run_prebuilt_smoke_test(
            chunk_idx=chunk_idx,
            chunk_output_dir=chunk_output_dir,
            run_id=run_id,
            prebuilt_parquet_path_resolved=prebuilt_parquet_path_resolved,
            worker_start_info=worker_start_info,
            cwd=cwd,
        )

    # Telemetry requirement
    telemetry_required = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
    truth_telemetry = os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
    if (not telemetry_required) and truth_telemetry:
        telemetry_required = True
        log.info("[TELEMETRY] GX1_TRUTH_TELEMETRY=1 enabled -> telemetry_required=True")

    # Fix #2: finalize bootstrap duration correctly
    t_init_s = (time.time() - t_init_start) if t_init_start is not None else 0.0

    # Worker start time (post-bootstrap “ready to run” timestamp)
    worker_start_time = time.time()

    # Final SSoT logs
    log.info(f"[CHUNK {chunk_idx}] [SSoT] cwd={cwd}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] sys.executable={python_exe}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] GX1_OUTPUT_DIR={output_dir_env}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] GX1_REPLAY_USE_PREBUILT_FEATURES={prebuilt_env}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] prebuilt_arg={prebuilt_parquet_path}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] prebuilt_resolved={prebuilt_parquet_path_resolved}")

    if prebuilt_enabled:
        log.info(f"[CHUNK {chunk_idx}] [SSoT] ONE_UNIVERSE ctx_cont_dim=6 ctx_cat_dim=6 (validated)")
        log.info(f"[CHUNK {chunk_idx}] [SSoT] canonical_truth_file={canonical_truth_file_val}")

    return BootstrapContext(
        chunk_idx=chunk_idx,
        policy_path=Path(policy_path),
        data_path=Path(data_path),
        output_dir=Path(output_dir),
        chunk_output_dir=chunk_output_dir,
        run_id=run_id,
        prebuilt_parquet_path=prebuilt_parquet_path,
        prebuilt_parquet_path_resolved=prebuilt_parquet_path_resolved,
        prebuilt_enabled=prebuilt_enabled,
        bundle_sha256=bundle_sha256,
        policy_id=policy_id,
        canonical_truth_file=canonical_truth_file_val,
        is_truth_or_smoke_worker=is_truth_or_smoke_worker,
        telemetry_required=telemetry_required,
        dt_module_version=dt_module_version,
        worker_start_time=worker_start_time,
        worker_boot_payload=worker_boot_payload,
        worker_start_info=worker_start_info,
        t_init_s=float(t_init_s),
        t_init_start=t_init_start,
        cwd=cwd,
        python_exe=python_exe,
        output_dir_env=output_dir_env,
        prebuilt_env=prebuilt_env,
        chunk_local_padding_days=int(chunk_local_padding_days or 0),
        workers=1,
        chunks=1,
        prebuilt_required_columns=prebuilt_required_columns_val,
    )