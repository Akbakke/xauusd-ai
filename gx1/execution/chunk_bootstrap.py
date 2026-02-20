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
)
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


def _compute_prebuilt_required_columns_one_universe(
    *,
    chunk_output_dir: Path,
    chunk_idx: int,
    run_id: str,
    canonical_truth_file: Path,
) -> Dict[str, Any]:
    """
    ONE UNIVERSE SSoT:
    - Truth file must exist.
    - canonical_xgb_bundle_dir must exist and have MASTER_MODEL_LOCK.json with ordered_features len=28.
    - canonical_transformer_bundle_dir must exist and have MASTER_TRANSFORMER_LOCK.json with ctx_cont_dim=6 ctx_cat_dim=6.
    - required columns = ordered_features (28) + missing(ctx12) appended.
    """
    truth_obj = _load_json_file(canonical_truth_file)

    canonical_xgb_bundle_dir_str = str(truth_obj.get("canonical_xgb_bundle_dir") or "").strip()
    canonical_transformer_bundle_dir_str = str(truth_obj.get("canonical_transformer_bundle_dir") or "").strip()

    _require(canonical_xgb_bundle_dir_str, "canonical_xgb_bundle_dir missing in truth config")
    _require(canonical_transformer_bundle_dir_str, "canonical_transformer_bundle_dir missing in truth config (ONE UNIVERSE requires it)")

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
    _require(isinstance(ordered, list) and all(isinstance(x, str) for x in ordered), "MASTER_MODEL_LOCK.ordered_features missing or not list[str]")

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

    required_ctx_12 = _one_universe_required_ctx_columns()

    base_set = set(ordered)
    extra = [c for c in required_ctx_12 if c not in base_set]
    required_all = list(ordered) + extra

    # Ensure the full ctx12 is present in required_all
    missing_ctx = [c for c in required_ctx_12 if c not in set(required_all)]
    _require(not missing_ctx, f"[BOOTSTRAP_FAIL] required_all missing ctx columns: {missing_ctx}")

    return {
        "canonical_xgb_bundle_dir": str(canonical_xgb_bundle_dir),
        "canonical_transformer_bundle_dir": str(canonical_transformer_bundle_dir),
        "expected_ctx_cont_dim": EXPECTED_CTX_CONT_DIM,
        "expected_ctx_cat_dim": EXPECTED_CTX_CAT_DIM,
        "ordered_features": list(ordered),
        "required_ctx_12": required_ctx_12,
        "prebuilt_required_columns": required_all,
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
    - Requires prebuilt_required_columns to include XGB 28 + ctx 12
    - No legacy, no defaults, no fallback.
    """
    # Determine run mode
    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    is_truth_or_smoke_worker = run_mode in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"

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

    # Prebuilt mode (env is the truth of intent; arg is required if enabled)
    prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"

    # Enforce type discipline
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

    if prebuilt_enabled and not prebuilt_parquet_path:
        fatal_msg = (
            f"[PREBUILT_MISSING] [CHUNK {chunk_idx}] prebuilt_parquet_path arg is None/empty "
            f"but GX1_REPLAY_USE_PREBUILT_FEATURES=1 (prebuilt required)."
        )
        write_fatal_capsule(
            chunk_output_dir=chunk_output_dir,
            chunk_idx=chunk_idx,
            run_id=run_id,
            fatal_reason="PREBUILT_MISSING",
            error_message=fatal_msg,
            extra_fields={"GX1_REPLAY_USE_PREBUILT_FEATURES": prebuilt_env},
        )
        raise RuntimeError(fatal_msg)

    # Canonical truth file (required in ONE UNIVERSE prebuilt mode)
    canonical_truth_file_val: Optional[Path] = None
    canonical_truth_file_env = os.getenv("GX1_CANONICAL_TRUTH_FILE")
    if canonical_truth_file_env:
        canonical_truth_file_val = Path(canonical_truth_file_env)

    if prebuilt_enabled:
        if not canonical_truth_file_val or not canonical_truth_file_val.exists():
            fatal_msg = (
                "[TRUTH_NO_FALLBACK] GX1_CANONICAL_TRUTH_FILE missing or does not exist; "
                "required in ONE UNIVERSE prebuilt mode."
            )
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="CANONICAL_TRUTH_FILE_MISSING",
                error_message=fatal_msg,
                extra_fields={"GX1_CANONICAL_TRUTH_FILE": canonical_truth_file_env},
            )
            raise RuntimeError(fatal_msg)

    prebuilt_parquet_path_resolved: Optional[str] = None
    prebuilt_exists = False
    prebuilt_size = 0

    if prebuilt_parquet_path:
        try:
            prebuilt_parquet_path_resolved = str(Path(prebuilt_parquet_path).resolve())
            p = Path(prebuilt_parquet_path_resolved)

            if not p.is_absolute():
                fatal_msg = (
                    f"[PREBUILT_PATH_NOT_ABSOLUTE] [CHUNK {chunk_idx}] Prebuilt path is not absolute: "
                    f"{prebuilt_parquet_path_resolved}"
                )
                write_fatal_capsule(
                    chunk_output_dir=chunk_output_dir,
                    chunk_idx=chunk_idx,
                    run_id=run_id,
                    fatal_reason="PREBUILT_PATH_NOT_ABSOLUTE",
                    error_message=fatal_msg,
                    extra_fields={"prebuilt_parquet_path_resolved": prebuilt_parquet_path_resolved},
                )
                raise RuntimeError(fatal_msg)

            prebuilt_exists = p.exists()
            if prebuilt_exists:
                prebuilt_size = int(p.stat().st_size)

        except Exception as e:
            fatal_msg = f"[PREBUILT_RESOLVE_FAIL] [CHUNK {chunk_idx}] Failed to resolve/stat prebuilt path: {e}"
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="PREBUILT_RESOLVE_FAIL",
                error_message=fatal_msg,
                extra_fields={
                    "prebuilt_parquet_path": str(prebuilt_parquet_path),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    # -------------------------------------------------------------------------
    # ONE UNIVERSE: compute prebuilt_required_columns (TRUTH/SMOKE: no fallback)
    # -------------------------------------------------------------------------
    prebuilt_required_columns_val: Optional[List[str]] = None
    canonical_xgb_bundle_dir_str: Optional[str] = None
    canonical_transformer_bundle_dir_str: Optional[str] = None
    expected_ctx_cont_dim: Optional[int] = None
    expected_ctx_cat_dim: Optional[int] = None

    if prebuilt_enabled:
        try:
            ssot = _compute_prebuilt_required_columns_one_universe(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                canonical_truth_file=canonical_truth_file_val,  # type: ignore[arg-type]
            )
            canonical_xgb_bundle_dir_str = ssot["canonical_xgb_bundle_dir"]
            canonical_transformer_bundle_dir_str = ssot["canonical_transformer_bundle_dir"]
            expected_ctx_cont_dim = ssot["expected_ctx_cont_dim"]
            expected_ctx_cat_dim = ssot["expected_ctx_cat_dim"]
            prebuilt_required_columns_val = ssot["prebuilt_required_columns"]

            _require(int(expected_ctx_cont_dim) == 6, "ONE UNIVERSE violation: expected_ctx_cont_dim != 6")
            _require(int(expected_ctx_cat_dim) == 6, "ONE UNIVERSE violation: expected_ctx_cat_dim != 6")

        except Exception as e:
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="PREBUILT_REQUIRED_COLUMNS_FAIL",
                error_message=str(e),
                extra_fields={
                    "canonical_truth_file": str(canonical_truth_file_val) if canonical_truth_file_val else None,
                    "canonical_xgb_bundle_dir": canonical_xgb_bundle_dir_str,
                    "canonical_transformer_bundle_dir": canonical_transformer_bundle_dir_str,
                    "traceback": traceback.format_exc(),
                },
            )
            raise RuntimeError(
                "[TRUTH_NO_FALLBACK] Failed to compute ONE UNIVERSE prebuilt_required_columns (28 + ctx12). "
                f"Error: {e}"
            ) from e

        if not prebuilt_required_columns_val:
            fatal_msg = "[TRUTH_NO_FALLBACK] prebuilt_required_columns missing in ONE UNIVERSE prebuilt mode."
            write_fatal_capsule(
                chunk_output_dir=chunk_output_dir,
                chunk_idx=chunk_idx,
                run_id=run_id,
                fatal_reason="PREBUILT_REQUIRED_COLUMNS_MISSING",
                error_message=fatal_msg,
                extra_fields={"canonical_truth_file": str(canonical_truth_file_val) if canonical_truth_file_val else None},
            )
            raise RuntimeError(fatal_msg)

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

    # TRUTH-only init timing
    worker_start_time = time.time()
    t_init_start: Optional[float] = time.time() if is_truth_or_smoke_worker else None
    t_init_s = (time.time() - t_init_start) if t_init_start is not None else 0.0

    # Final SSoT logs
    log.info(f"[CHUNK {chunk_idx}] [SSoT] cwd={cwd}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] sys.executable={python_exe}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] GX1_OUTPUT_DIR={output_dir_env}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] GX1_REPLAY_USE_PREBUILT_FEATURES={prebuilt_env}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] prebuilt_arg={prebuilt_parquet_path}")
    log.info(f"[CHUNK {chunk_idx}] [SSoT] prebuilt_resolved={prebuilt_parquet_path_resolved}")

    if prebuilt_enabled:
        log.info(f"[CHUNK {chunk_idx}] [SSoT] ONE_UNIVERSE ctx_cont_dim=6 ctx_cat_dim=6")

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