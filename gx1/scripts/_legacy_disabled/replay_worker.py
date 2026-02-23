#!/home/andre2/venvs/gx1/bin/python
"""
Subprocess worker for processing a single chunk in isolation.

Purpose:
- Run chunk processing in completely isolated subprocess
- No multiprocessing, no IPC - pure subprocess
- Exit code: 0 = success, !=0 = failure
- Writes WORKER_BOOT.json, WORKER_END.json, chunk_footer.json
"""

import sys
import os

REQUIRED_VENV = "/home/andre2/venvs/gx1/bin/python"

if sys.executable != REQUIRED_VENV:
    raise RuntimeError(
        f"[ENV_IDENTITY_FAIL] Wrong python interpreter\n"
        f"Expected: {REQUIRED_VENV}\n"
        f"Actual:   {sys.executable}\n"
        f"Hint: source ~/venvs/gx1/bin/activate"
    )

import argparse
import sys
import os
import time
import signal
import faulthandler
import traceback
from pathlib import Path
import json

# Enable faulthandler immediately (before any imports that might crash)
_fault_dump_file = None
if "GX1_FAULTHANDLER_OUTPUT" in os.environ:
    _fault_dump_file = open(os.environ["GX1_FAULTHANDLER_OUTPUT"], "w")
else:
    _fault_dump_file = sys.stderr

faulthandler.enable(file=_fault_dump_file, all_threads=True)

# Register SIGUSR1 for manual dump
def _dump_traceback(signum, frame):
    faulthandler.dump_traceback(file=_fault_dump_file, all_threads=True)

signal.signal(signal.SIGUSR1, _dump_traceback)

# Import after faulthandler setup
import logging
import pandas as pd

# Import from replay_eval_gated_parallel
from gx1.scripts.replay_eval_gated_parallel import (
    process_chunk,
    split_data_into_chunks,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr for worker logs
    ]
)
log = logging.getLogger(__name__)


def main():
    # B0.4: Write WORKER_BOOT.json as first action (before any other work)
    # This proves the worker process started, even if it crashes immediately
    try:
        # Parse minimal args first to get output_dir
        parser = argparse.ArgumentParser(description="Process a single chunk in subprocess")
        parser.add_argument("--chunk-id", type=int, required=True, help="Chunk ID to process (0-indexed)")
        parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
        args_minimal, _ = parser.parse_known_args()
        
        # Resolve output_dir early
        output_dir = Path(args_minimal.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_output_dir = output_dir / f"chunk_{args_minimal.chunk_id}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write WORKER_BOOT.json immediately (before any imports or heavy work)
        worker_boot_early = {
            "chunk_id": args_minimal.chunk_id,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "sys.executable": sys.executable,
            "sys.version": sys.version.split()[0],
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "stage": "worker_entrypoint",
            "imports_ok": False,  # Will be updated after imports
        }
        worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
        with open(worker_boot_path, "w") as f:
            json.dump(worker_boot_early, f, indent=2)
        log.info(f"[WORKER_BOOT] ✅ Early WORKER_BOOT.json written (stage=worker_entrypoint)")
    except Exception as boot_error:
        # If we can't even write WORKER_BOOT.json, log to stderr
        print(f"[WORKER_BOOT] FATAL: Failed to write early WORKER_BOOT.json: {boot_error}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Continue anyway - maybe we can still write it later
    
    # Now parse full args
    parser = argparse.ArgumentParser(description="Process a single chunk in subprocess")
    parser.add_argument("--chunk-id", type=int, required=True, help="Chunk ID to process (0-indexed)")
    parser.add_argument("--chunk-start", type=str, required=True, help="Chunk start timestamp (ISO format)")
    parser.add_argument("--chunk-end", type=str, required=True, help="Chunk end timestamp (ISO format)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data parquet")
    parser.add_argument("--policy-yaml", type=str, required=True, help="Path to policy YAML")
    parser.add_argument("--bundle-dir", type=str, required=True, help="Path to bundle directory")
    parser.add_argument("--prebuilt-parquet", type=str, required=True, help="Path to prebuilt parquet")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID")
    parser.add_argument("--bundle-sha256", type=str, required=True, help="Bundle SHA256")
    parser.add_argument("--chunk-local-padding-days", type=int, default=0, help="Chunk-local padding days (TRUTH/SMOKE)")
    
    args = parser.parse_args()
    
    # Resolve paths
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup fault dump file
    chunk_output_dir = output_dir / f"chunk_{args.chunk_id}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update WORKER_BOOT.json with imports_ok=True (we got this far)
    worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
    try:
        if worker_boot_path.exists():
            with open(worker_boot_path, "r") as f:
                worker_boot_early = json.load(f)
            worker_boot_early["imports_ok"] = True
            with open(worker_boot_path, "w") as f:
                json.dump(worker_boot_early, f, indent=2)
    except Exception:
        pass  # Non-fatal
    fault_dump_path = chunk_output_dir / "WORKER_FAULT_DUMP.txt"
    global _fault_dump_file
    _fault_dump_file = open(fault_dump_path, "w")
    faulthandler.enable(file=_fault_dump_file, all_threads=True)
    
    log.info(f"[WORKER] Starting chunk {args.chunk_id} in subprocess")
    log.info(f"[WORKER] Output dir: {chunk_output_dir}")
    
    # A) Load prebuilt parquet explicitly (required for PREBUILT mode)
    prebuilt_df = None
    prebuilt_loader = None
    prebuilt_path_resolved = None
    has_prebuilt_df = False
    replay_mode_enum = None
    
    # Initialize worker_boot_data early (for error cases)
    worker_boot_data = {
        "chunk_id": args.chunk_id,
        "timestamp": time.time(),
        "pid": os.getpid(),
        "replay_mode_enum": None,
        "has_prebuilt_df": False,
        "prebuilt_path": None,
        "prebuilt_df_rows": 0,
        "prebuilt_df_cols": 0,
        "prebuilt_df_first_ts": None,
        "prebuilt_df_last_ts": None,
    }
    
    try:
        # Validate prebuilt path exists
        prebuilt_path = Path(args.prebuilt_parquet)
        prebuilt_path_resolved = prebuilt_path.resolve()
        
        log.info(f"[WORKER_PREFLIGHT] prebuilt_path={prebuilt_path_resolved}")
        worker_boot_data["prebuilt_path"] = str(prebuilt_path_resolved)
        
        if not prebuilt_path_resolved.exists():
            fatal_msg = f"[PREBUILT_LOAD_FAIL] Prebuilt parquet file does not exist: {prebuilt_path_resolved}"
            log.error(fatal_msg)
            # Write FATAL capsule
            fatal_capsule_path = chunk_output_dir / "PREBUILT_LOAD_FAIL.json"
            with open(fatal_capsule_path, "w") as f:
                json.dump({
                    "error_type": "PREBUILT_LOAD_FAIL",
                    "error_msg": fatal_msg,
                    "prebuilt_path": str(prebuilt_path_resolved),
                    "exists": False,
                    "chunk_id": args.chunk_id,
                    "timestamp": time.time(),
                }, f, indent=2)
            # Write WORKER_BOOT.json with error
            worker_boot_data["fatal_error"] = fatal_msg
            worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
            with open(worker_boot_path, "w") as f:
                json.dump(worker_boot_data, f, indent=2)
            return 1
        
        prebuilt_size = prebuilt_path_resolved.stat().st_size
        if prebuilt_size < 1000:  # Minimal threshold (1KB)
            fatal_msg = f"[PREBUILT_LOAD_FAIL] Prebuilt parquet file is too small (size={prebuilt_size} bytes): {prebuilt_path_resolved}"
            log.error(fatal_msg)
            fatal_capsule_path = chunk_output_dir / "PREBUILT_LOAD_FAIL.json"
            with open(fatal_capsule_path, "w") as f:
                json.dump({
                    "error_type": "PREBUILT_LOAD_FAIL",
                    "error_msg": fatal_msg,
                    "prebuilt_path": str(prebuilt_path_resolved),
                    "exists": True,
                    "size_bytes": prebuilt_size,
                    "chunk_id": args.chunk_id,
                    "timestamp": time.time(),
                }, f, indent=2)
            # Write WORKER_BOOT.json with error
            worker_boot_data["fatal_error"] = fatal_msg
            worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
            with open(worker_boot_path, "w") as f:
                json.dump(worker_boot_data, f, indent=2)
            return 1
        
        log.info(f"[WORKER_PREFLIGHT] prebuilt_path exists=True, size_bytes={prebuilt_size:,}")
        
        # Load prebuilt parquet using PrebuiltFeaturesLoader
        try:
            from gx1.execution.prebuilt_features_loader import PrebuiltFeaturesLoader
            from gx1.utils.replay_mode import ReplayMode
            
            prebuilt_loader = PrebuiltFeaturesLoader(prebuilt_path_resolved)
            prebuilt_df = prebuilt_loader.df
            has_prebuilt_df = True
            replay_mode_enum = ReplayMode.PREBUILT
            
            log.info(
                f"[WORKER_PREFLIGHT] ✅ Prebuilt loaded: {len(prebuilt_df):,} rows, {len(prebuilt_df.columns)} columns"
            )
            log.info(
                f"[WORKER_PREFLIGHT] Prebuilt index: type={type(prebuilt_df.index).__name__}, "
                f"tz={getattr(prebuilt_df.index, 'tz', None)}, "
                f"first_ts={prebuilt_df.index[0] if len(prebuilt_df) > 0 else 'N/A'}, "
                f"last_ts={prebuilt_df.index[-1] if len(prebuilt_df) > 0 else 'N/A'}"
            )
            
            # Update worker_boot_data with loaded prebuilt info
            worker_boot_data["replay_mode_enum"] = replay_mode_enum.value
            worker_boot_data["has_prebuilt_df"] = has_prebuilt_df
            worker_boot_data["prebuilt_df_rows"] = len(prebuilt_df)
            worker_boot_data["prebuilt_df_cols"] = len(prebuilt_df.columns)
            worker_boot_data["prebuilt_df_first_ts"] = str(prebuilt_df.index[0]) if len(prebuilt_df) > 0 else None
            worker_boot_data["prebuilt_df_last_ts"] = str(prebuilt_df.index[-1]) if len(prebuilt_df) > 0 else None
            
        except Exception as load_error:
            fatal_msg = f"[PREBUILT_LOAD_FAIL] Failed to load prebuilt parquet: {load_error}"
            log.error(fatal_msg, exc_info=True)
            # Write FATAL capsule with traceback
            fatal_capsule_path = chunk_output_dir / "PREBUILT_LOAD_FAIL.json"
            with open(fatal_capsule_path, "w") as f:
                json.dump({
                    "error_type": "PREBUILT_LOAD_FAIL",
                    "error_msg": fatal_msg,
                    "prebuilt_path": str(prebuilt_path_resolved),
                    "exists": True,
                    "size_bytes": prebuilt_size,
                    "exception_type": type(load_error).__name__,
                    "exception_str": str(load_error),
                    "traceback": traceback.format_exception(type(load_error), load_error, load_error.__traceback__),
                    "chunk_id": args.chunk_id,
                    "timestamp": time.time(),
                }, f, indent=2)
            # Write WORKER_BOOT.json with error
            worker_boot_data["fatal_error"] = fatal_msg
            worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
            with open(worker_boot_path, "w") as f:
                json.dump(worker_boot_data, f, indent=2)
            return 1
        
        # Write WORKER_BOOT.json with prebuilt status (before hard invariant check)
        worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
        with open(worker_boot_path, "w") as f:
            json.dump(worker_boot_data, f, indent=2)
        
        log.info(f"[WORKER_BOOT] ✅ Wrote WORKER_BOOT.json: has_prebuilt_df={has_prebuilt_df}, replay_mode_enum={replay_mode_enum.value if replay_mode_enum else None}")
        
        # Hard fail if PREBUILT mode but no DF (after WORKER_BOOT.json is written)
        if replay_mode_enum == ReplayMode.PREBUILT and not has_prebuilt_df:
            fatal_msg = "[PREBUILT_MODE_BUT_NO_DF] replay_mode_enum=PREBUILT but has_prebuilt_df=False"
            log.error(fatal_msg)
            fatal_capsule_path = chunk_output_dir / "PREBUILT_MODE_BUT_NO_DF.json"
            with open(fatal_capsule_path, "w") as f:
                json.dump({
                    "error_type": "PREBUILT_MODE_BUT_NO_DF",
                    "error_msg": fatal_msg,
                    "replay_mode_enum": replay_mode_enum.value if replay_mode_enum else None,
                    "has_prebuilt_df": has_prebuilt_df,
                    "chunk_id": args.chunk_id,
                    "timestamp": time.time(),
                }, f, indent=2)
            # Update WORKER_BOOT.json with fatal error
            worker_boot_data["fatal_error"] = fatal_msg
            with open(worker_boot_path, "w") as f:
                json.dump(worker_boot_data, f, indent=2)
            return 1
        
    except Exception as boot_error:
        log.error(f"[WORKER] ❌ Failed during worker boot: {boot_error}", exc_info=True)
        traceback.print_exc(file=sys.stderr)
        # Try to write WORKER_BOOT.json with error
        try:
            worker_boot_data["fatal_error"] = str(boot_error)
            worker_boot_data["fatal_error_type"] = type(boot_error).__name__
            worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
            with open(worker_boot_path, "w") as f:
                json.dump(worker_boot_data, f, indent=2)
        except Exception:
            pass
        return 1
    
    try:
        # Parse timestamps
        chunk_start = pd.Timestamp(args.chunk_start)
        chunk_end = pd.Timestamp(args.chunk_end)
        
        log.info(f"[WORKER] Chunk {args.chunk_id}: {chunk_start} to {chunk_end}")
        
        # Signal-only TRUTH: prebuilt path is propagated via explicit --prebuilt-parquet arg, not env.
        os.environ["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
        
        # Call process_chunk directly (same code path as multiprocessing worker)
        start_time = time.time()
        
        chunk_result = process_chunk(
            chunk_idx=args.chunk_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            data_path=Path(args.data_path),
            policy_path=Path(args.policy_yaml),
            run_id=args.run_id,
            output_dir=output_dir,
            bundle_sha256=args.bundle_sha256,
            prebuilt_parquet_path=str(prebuilt_path_resolved),  # Pass resolved absolute path
            bundle_dir=Path(args.bundle_dir),
            chunk_local_padding_days=int(getattr(args, "chunk_local_padding_days", 0) or 0),
        )
        
        elapsed = time.time() - start_time
        log.info(f"[WORKER] ✅ Chunk {args.chunk_id} completed successfully in {elapsed:.1f}s")
        
        # Verify chunk_footer.json exists
        chunk_footer_path = chunk_output_dir / "chunk_footer.json"
        if not chunk_footer_path.exists():
            log.error(f"[WORKER] ❌ chunk_footer.json MISSING: {chunk_footer_path}")
            return 1
        
        with open(chunk_footer_path, "r") as f:
            footer = json.load(f)
        
        if footer.get("status") != "ok":
            log.error(f"[WORKER] ❌ chunk_footer.json status is not 'ok': {footer.get('status')}")
            return 1
        
        log.info(f"[WORKER] ✅ chunk_footer.json status: ok")
        
        # Verify WORKER_END.json exists
        worker_end_path = chunk_output_dir / "WORKER_END.json"
        if not worker_end_path.exists():
            log.error(f"[WORKER] ❌ WORKER_END.json MISSING: {worker_end_path}")
            return 1
        
        log.info(f"[WORKER] ✅ WORKER_END.json exists")
        
        # Log peak RSS (VmHWM) from WORKER_END.json for observability
        try:
            with open(worker_end_path, "r") as f:
                worker_end = json.load(f)
            memory_vmhwm_mb = worker_end.get("memory_vmhwm_mb", None)
            if memory_vmhwm_mb is not None:
                log.info(f"[WORKER] Peak memory (VmHWM): {memory_vmhwm_mb:.2f} MB")
        except Exception as e:
            log.warning(f"[WORKER] Failed to read memory from WORKER_END.json: {e}")
        
        return 0
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        log.error(f"[WORKER] ❌ Chunk {args.chunk_id} failed after {elapsed:.1f}s: {e}", exc_info=True)
        
        # Dump full traceback to stderr (master will capture)
        traceback.print_exc(file=sys.stderr)
        
        # Dump full traceback to file
        error_dump_path = chunk_output_dir / "WORKER_ERROR_DUMP.txt"
        with open(error_dump_path, "w") as f:
            f.write(f"Chunk {args.chunk_id} execution failed after {elapsed:.1f}s\n")
            f.write(f"Exception type: {type(e).__name__}\n")
            f.write(f"Exception message: {str(e)}\n")
            f.write("\nFull traceback:\n")
            f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        
        log.error(f"[WORKER] Error dump written to: {error_dump_path}")
        
        # Check if FATAL capsule exists
        fatal_path = chunk_output_dir / "FATAL_ERROR.txt"
        if fatal_path.exists():
            log.info(f"[WORKER] FATAL_ERROR.txt exists: {fatal_path}")
            with open(fatal_path, "r") as f:
                log.info(f"[WORKER] FATAL content: {f.read()[:500]}")
        
        return 1
    
    finally:
        # Close fault dump file
        if _fault_dump_file and _fault_dump_file != sys.stderr:
            _fault_dump_file.close()
        
        log.info(f"[WORKER] Chunk {args.chunk_id} subprocess exiting")


if __name__ == "__main__":
    sys.exit(main())
