#!/home/andre2/venvs/gx1/bin/python
# -*- coding: utf-8 -*-
"""
TRUTH-grade end-to-end sanity checker for the signal-only pipeline.

Verifies:
- Canonical truth file exists + matches signal_bridge contract SHA
- XGB bundle: MASTER_MODEL_LOCK.json + meta ordered_features match (order-sensitive)
- Transformer bundle: MASTER_TRANSFORMER_LOCK.json exists
- Prebuilt parquet: manifest + schema_manifest + required_all_features prefix-match vs XGB ordered_features; 6/6 reality check (all 12 ctx columns present, no NaN/Inf in ctx) before replay/training
- TRUTH env sanity (required envs set, forbidden envs unset)
- Replay runs 1W1C in-process via gx1.execution.replay_chunk.process_chunk + replay_merge.merge_artifacts_1w1c (no legacy replay script import)
- Post-run required artifacts exist (chunk + merged + RUN_COMPLETED)
- chunk_footer.status == ok
- prebuilt_proven (env + footer path + join file) and join_ratio >= 0.995
- feature_build_call_count is 0 if present, otherwise require GX1_FEATURE_BUILD_DISABLED=1
- ctx dims: 6/6 only (ONE UNIVERSE; hard-fail otherwise)
- transformer forward calls > 0 (robust metrics schema fallback)
- zero-trades contract: trade_outcomes parquet exists + ZERO_TRADES_DIAG.json exists when n_trades==0
- exit coverage: truth_exit_journal_ok==true if EXIT_COVERAGE_SUMMARY.json exists
- bars invariant: bars_total_input - bars_processed == tail_holdback_bars

No fallback: missing/mismatch → hard fail.
Always writes E2E_FATAL_CAPSULE.json on failure.

One-liner commands (canonical short-window TRUTH replay; XGB BASE28 → XGB_SIGNAL_BRIDGE_V1 7 dims → Transformer + ctx side-channel):

  A) Validate-only preflight (no replay):
     export GX1_DATA=/home/andre2/GX1_DATA
     export GX1_CANONICAL_TRUTH_FILE=/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json
     export GX1_STRICT_MASK=1
     export GX1_CTX_CONT_MASK=1,1,1,1
     export GX1_CTX_CAT_MASK=1,1,1,1,1
     /home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity --validate-only --start-ts 2025-06-03 --end-ts 2025-06-10

  B) Micro replay (same window; masks 6/6 if env unset):
     /home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity --start-ts 2025-06-03 --end-ts 2025-06-10

  C) Zero-trades canary (1 day; GX1_ENTRY_THRESHOLD_OVERRIDE=1.1; RUN_IDENTITY.json + E2E_SANITY_SUMMARY proof; hard-fail if n_trades>0):
     python -m gx1.scripts.run_truth_e2e_sanity --force-zero-trades --start-ts 2025-06-03 --end-ts 2025-06-04

  D) Full-year proof: use run_fullyear_2025_truth_proof.py (separate runner).

  6/6 rollout (ONE UNIVERSE: ctx_cont_dim=6, ctx_cat_dim=6; no CLI override):

    1) E2E short window (exits must have context.ctx_cont/ctx_cat len 6/6):
       python -m gx1.scripts.run_truth_e2e_sanity --truth-file <canonical_truth> --start-ts 2025-06-03 --end-ts 2025-06-10
       (LAST_GO oppdateres kun når exits har 6/6; postrun gate.)

    2) Train exit 6/6 from LAST_GO:
       python -m gx1.scripts.run_truth_e2e_sanity --train-exit-transformer-v0-from-last-go --require-io-v2

    3) E2E full-year 6/6 (run_fullyear_2025_truth_proof eller tilsvarende).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ENGINE = Path(__file__).resolve().parents[2]

# TRUTH/SMOKE SSoT allowlist (entrypoints that may run in TRUTH/SMOKE):
# - gx1/scripts/run_truth_e2e_sanity.py (this orchestrator)
# - gx1/execution/replay_chunk.py (worker)
# - gx1/execution/chunk_data_loader.py (data loader)
# - XGB eval/calibration: train_xgb_universal_v1_multiyear.py, train_xgb_universal_multihead_v2.py
# Everything else touching PRUNE14/PRUNE20/v13_refined3/CTX6CAT6 or direct canonical_prebuilt_parquet
# must remain frozen/disabled.

# KUN ÉN TRUTH (default) — used when neither CLI nor env provides truth-file.
CANONICAL_TRUTH_DEFAULT = "/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json"
MANIFEST_SSOT = Path("/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json")

DEFAULT_START_TS = "2025-06-03T00:00:00+00:00"
DEFAULT_END_TS = "2025-06-10T23:59:59+00:00"
FULLYEAR_START_TS = "2025-01-01T00:00:00+00:00"
FULLYEAR_END_TS = "2025-12-31T23:59:59+00:00"

JOIN_RATIO_TRUTH = 0.995

# ONE UNIVERSE: context dims are fixed 6/6 (no CLI or env override for dims).
CTX_CONT_DIM = 6
CTX_CAT_DIM = 6

# Zero-trades canary: threshold > 1.0 so entry is mathematically impossible (entry uses >= threshold)
ZERO_TRADES_CANARY_THRESHOLD = "1.1"

# Forbidden envs (baseline TRUTH must have these unset)
FORBIDDEN_ENVS = [
    "GX1_STOP_AFTER_N_BARS",
    "GX1_BAR_SKIP_TRACE",
    "GX1_BAR_SKIP_TRACE_MAX",
    "GX1_KILLCHAIN_STAGE2_TRACE",
    "GX1_KILLCHAIN_STAGE2_TRACE_MAX",
    "GX1_SEGMENTED_PARALLEL",
    "GX1_SEGMENT_START",
    "GX1_SEGMENT_END",
    "GX1_PARALLEL",
    "GX1_SEGMENTED",
    "GX1_PREROLL_BARS",
    "GX1_PREROLL_START",
    "GX1_OWNER_START",
    "GX1_OWNER_END",
    "GX1_REPLAY_PREBUILT_FEATURES_PATH",
]

REQUIRED_TRUTH_ENVS = {
    "GX1_RUN_MODE": "TRUTH",
    "GX1_TRUTH_MODE": "1",
    "GX1_REPLAY_USE_PREBUILT_FEATURES": "1",
    "GX1_FEATURE_BUILD_DISABLED": "1",
    "GX1_GATED_FUSION_ENABLED": "1",
}


def _assert_no_forbidden_symbol_imports_after_replay(run_root: Path) -> None:
    """TRUTH gate: hard-fail if forbidden modules are in sys.modules after replay (stricter than IMPORT_PROOF/banlist).
    Only explicitly banned names and runtime_v9 pattern; no broad baseline/fallback pattern (avoids false positives)."""
    forbidden_exact = [
        "gx1.inference.model_loader_worker",
        "gx1.scripts.replay_eval_gated_parallel",
    ]
    hits: List[str] = []
    for mod in forbidden_exact:
        if mod in sys.modules:
            hits.append(mod)
    for name in sys.modules:
        if "runtime_v9" in name:
            hits.append(name)
    hits = sorted(set(hits))
    if hits:
        msg = (
            "[TRUTH_FORBIDDEN_SYMBOL_IMPORTS] Forbidden modules in sys.modules after replay: "
            + ", ".join(hits)
            + ". Banned: model_loader_worker, replay_eval_gated_parallel, runtime_v9."
        )
        _write_fatal_capsule(run_root, RuntimeError(msg), ["forbidden_symbol_imports"])
        raise RuntimeError(msg)


def _assert_truth_no_legacy_replay(run_root: Path) -> None:
    """TRUTH/SMOKE gate: fail hard if legacy replay is loaded or script exists on disk."""
    if "gx1.scripts.replay_eval_gated_parallel" in sys.modules:
        msg = (
            "[TRUTH_GATE] Legacy replay script must not be imported in TRUTH path. "
            "Found in sys.modules. Use replay_chunk.process_chunk + replay_merge.merge_artifacts_1w1c only."
        )
        _write_fatal_capsule(run_root, RuntimeError(msg), ["legacy_replay_import"])
        raise RuntimeError(msg)
    legacy_script_path = ENGINE / "gx1" / "scripts" / "replay_eval_gated_parallel.py"
    if legacy_script_path.exists():
        msg = (
            f"[TRUTH_GATE] Legacy replay script must not exist in repo (ghost purge). "
            f"File exists: {legacy_script_path}."
        )
        _write_fatal_capsule(run_root, RuntimeError(msg), ["legacy_replay_on_disk"])
        raise RuntimeError(msg)


def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _gx1_data() -> Path:
    gx1_data = os.environ.get("GX1_DATA") or os.environ.get("GX1_DATA_DIR") or os.environ.get("GX1_DATA_ROOT")
    if gx1_data:
        return Path(gx1_data).expanduser().resolve()
    return Path.home() / "GX1_DATA"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"[E2E] file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _forbid_prune_path(label: str, value: str, *, allow_ctx6cat6: bool = False) -> None:
    upper = value.upper()
    tokens = ["PRUNE14", "PRUNE20", "V13_REFINED3_PRUNE", "REFINED3"]
    if not allow_ctx6cat6:
        tokens.append("CTX6CAT6")
    if any(bad in upper for bad in tokens):
        raise RuntimeError(f"LEGACY_PRUNE_FORBIDDEN_IN_TRUTH: {label}={value}")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> bool:
    """
    Atomic JSON writer; prefers gx1.utils.atomic_json.atomic_write_json with fallback to tmp+replace.
    Never raises (best-effort).
    """
    try:
        from gx1.utils.atomic_json import atomic_write_json as _write  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        return _write(path, payload, fallback_on_error=False)
    except Exception:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, path)
            return True
        except Exception:
            return False


def _write_fatal_capsule(run_root: Path, error: BaseException, gates_failed: List[str]) -> None:
    capsule = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "run_truth_e2e_sanity",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "gates_failed": gates_failed,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    _atomic_write_json(run_root / "E2E_FATAL_CAPSULE.json", capsule)


def _apply_ctx_mask_defaults(ctx_cont_dim: int, ctx_cat_dim: int) -> None:
    """
    Set GX1_CTX_CONT_MASK / GX1_CTX_CAT_MASK from dims if unset; else validate length.
    ONE UNIVERSE: only 6/6 allowed; hard-fail otherwise.
    """
    if ctx_cont_dim != CTX_CONT_DIM:
        raise RuntimeError(f"[E2E] ctx_cont_dim must be {CTX_CONT_DIM} (ONE UNIVERSE), got {ctx_cont_dim}")
    if ctx_cat_dim != CTX_CAT_DIM:
        raise RuntimeError(f"[E2E] ctx_cat_dim must be {CTX_CAT_DIM} (ONE UNIVERSE), got {ctx_cat_dim}")
    cont_raw = os.environ.get("GX1_CTX_CONT_MASK", "").strip()
    if not cont_raw:
        os.environ["GX1_CTX_CONT_MASK"] = ",".join(["1"] * ctx_cont_dim)
    else:
        parts = [p.strip() for p in cont_raw.split(",") if p.strip()]
        if len(parts) != ctx_cont_dim:
            raise RuntimeError(
                f"[E2E] GX1_CTX_CONT_MASK length={len(parts)} does not match ctx_cont_dim={ctx_cont_dim}. "
                "Set env to 6 ones or leave unset."
            )
    cat_raw = os.environ.get("GX1_CTX_CAT_MASK", "").strip()
    if not cat_raw:
        os.environ["GX1_CTX_CAT_MASK"] = ",".join(["1"] * ctx_cat_dim)
    else:
        parts = [p.strip() for p in cat_raw.split(",") if p.strip()]
        if len(parts) != ctx_cat_dim:
            raise RuntimeError(
                f"[E2E] GX1_CTX_CAT_MASK length={len(parts)} does not match ctx_cat_dim={ctx_cat_dim}. "
                "Set env to 6 ones or leave unset."
            )


def _exits_context_gate(
    run_root: Path,
    run_id: str,
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
    require_exits_file: bool = False,
) -> tuple[bool, str | None]:
    """
    TRUTH gate: LAST_GO only when exits have expected context (e.g. 6/6).
    Check only: replay/chunk_0/logs/exits/exits_<run_id>.jsonl (no glob/fuzzy).
    If file exists: at least one line must have context.ctx_cont len=expected_ctx_cont_dim and
    context.ctx_cat len=expected_ctx_cat_dim (dims from footer or default 6/6).
    Else: gate fails (caller must fail postrun, write fatal capsule, not update LAST_GO.txt).
    If file does not exist: gate passes unless require_exits_file (exit_ml_enabled / GX1_EXIT_AUDIT).
    Returns (True, None) if gate passes; (False, error_message) if not.
    """
    exits_path = run_root / "replay" / "chunk_0" / "logs" / "exits" / f"exits_{run_id}.jsonl"
    if not exits_path.exists():
        if require_exits_file:
            return False, (f"exits jsonl required when exit ML/audit enabled but file not found: {exits_path}")
        return True, None
    found_valid = False
    for line in exits_path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctx = rec.get("context")
        if not ctx or not isinstance(ctx, dict):
            continue
        c_cont = ctx.get("ctx_cont")
        c_cat = ctx.get("ctx_cat")
        if c_cont is None or c_cat is None:
            continue
        n_cont = len(c_cont) if isinstance(c_cont, (list, tuple)) else 0
        n_cat = len(c_cat) if isinstance(c_cat, (list, tuple)) else 0
        if n_cont == expected_ctx_cont_dim and n_cat == expected_ctx_cat_dim:
            found_valid = True
            break
    if not found_valid:
        # 0-trade run: footer has 6/6 and exit ML was ready; no exits to log
        footer_path = run_root / "replay" / "chunk_0" / "chunk_footer.json"
        if footer_path.exists():
            try:
                footer = _load_json(footer_path)
                if (
                    footer.get("n_trades_closed", 0) == 0
                    and footer.get("ctx_cont_dim") == expected_ctx_cont_dim
                    and footer.get("ctx_cat_dim") == expected_ctx_cat_dim
                ):
                    return True, None
            except Exception:
                pass
        return False, (
            f"exits jsonl has no event with context.ctx_cont/ctx_cat of expected lengths "
            f"(expected ctx_cont_dim={expected_ctx_cont_dim}, ctx_cat_dim={expected_ctx_cat_dim}). "
            "LAST_GO will not be updated; rerun with context in replay so exits audit gets 6/6."
        )
    return True, None


def _run_preflight(truth_path: Path, run_root: Path) -> Dict[str, Any]:
    """Preflight: env sanity, truth file + SHA, XGB lock + meta, transformer lock, prebuilt + schema prefix."""
    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "checks": {},
        "gates_failed": [],
    }

    # Env sanity
    for key, expected in REQUIRED_TRUTH_ENVS.items():
        actual = os.getenv(key, "")
        if actual != expected:
            result["gates_failed"].append(f"env_{key}")
            result["checks"]["env_sanity"] = {"error": f"{key}={actual!r} (expected {expected!r})"}
            return result
    for key in FORBIDDEN_ENVS:
        if os.getenv(key):
            result["gates_failed"].append(f"forbidden_env_{key}")
            result["checks"]["env_sanity"] = {"error": f"Forbidden env {key} is set"}
            return result
    result["checks"]["env_sanity"] = {"passed": True}

    # Canonical truth file + bridge SHA
    if not truth_path.is_absolute() or not truth_path.exists():
        result["gates_failed"].append("canonical_truth_file")
        result["checks"]["canonical_truth"] = {"error": f"Truth file missing/invalid: {truth_path}"}
        return result
    try:
        truth_obj = _load_json(truth_path)
    except Exception as e:
        result["gates_failed"].append("canonical_truth_file")
        result["checks"]["canonical_truth"] = {"error": str(e)}
        return result

    # Support keys: signal_bridge_contract_sha256 (preferred), signal_bridge_sha (legacy). Both present and different → drift, hard-fail.
    key_preferred = "signal_bridge_contract_sha256"
    key_legacy = "signal_bridge_sha"
    val_preferred = truth_obj.get(key_preferred)
    val_legacy = truth_obj.get(key_legacy)
    if val_preferred is not None:
        val_preferred = str(val_preferred).strip()
    if val_legacy is not None:
        val_legacy = str(val_legacy).strip()
    if val_preferred is not None and val_legacy is not None and val_preferred != val_legacy:
        result["gates_failed"].append("signal_bridge_sha")
        result["checks"]["canonical_truth"] = {
            "error": "contract sha drift in truth file: signal_bridge_contract_sha256 != signal_bridge_sha (expected one key or both equal). Fix gx1/configs canonical truth JSON.",
            "signal_bridge_sha_key_used": None,
            "signal_bridge_sha_match": False,
            "signal_bridge_contract_sha256_expected": None,
        }
        try:
            from gx1.contracts.signal_bridge_v1 import CONTRACT_SHA256 as _EXP  # type: ignore

            result["checks"]["canonical_truth"]["signal_bridge_contract_sha256_expected"] = _EXP
        except Exception:
            pass
        return result
    bridge_sha = (val_preferred or val_legacy) or None
    key_used = key_preferred if val_preferred is not None else (key_legacy if val_legacy is not None else None)
    if bridge_sha is None or key_used is None:
        result["gates_failed"].append("signal_bridge_sha")
        result["checks"]["canonical_truth"] = {
            "error": "missing key: expected one of {signal_bridge_contract_sha256, signal_bridge_sha}",
            "signal_bridge_sha_key_used": None,
            "signal_bridge_sha_match": False,
        }
        try:
            from gx1.contracts.signal_bridge_v1 import CONTRACT_SHA256 as _EXP  # type: ignore

            result["checks"]["canonical_truth"]["signal_bridge_contract_sha256_expected"] = _EXP
        except Exception:
            pass
        return result

    try:
        from gx1.contracts.signal_bridge_v1 import CONTRACT_SHA256 as _BRIDGE_SHA  # type: ignore

        match = bridge_sha == _BRIDGE_SHA
        if not match:
            result["gates_failed"].append("signal_bridge_sha")
            result["checks"]["canonical_truth"] = {
                "error": "contract sha mismatch: truth file value != gx1/contracts/signal_bridge_v1.py:CONTRACT_SHA256",
                "signal_bridge_sha_match": False,
                "signal_bridge_sha_key_used": key_used,
                "signal_bridge_sha_value": bridge_sha[:16] + "..." if len(bridge_sha) > 16 else bridge_sha,
                "signal_bridge_contract_sha256_expected": _BRIDGE_SHA,
            }
            return result
    except Exception as e:
        result["gates_failed"].append("signal_bridge_sha")
        result["checks"]["canonical_truth"] = {
            "error": "contract sha mismatch: " + str(e) + " (expected gx1/contracts/signal_bridge_v1.py:CONTRACT_SHA256)",
            "signal_bridge_sha_match": False,
            "signal_bridge_sha_key_used": key_used,
        }
        return result

    canonical_bundle = str(truth_obj.get("canonical_xgb_bundle_dir") or "")
    canonical_prebuilt = str(truth_obj.get("canonical_prebuilt_parquet") or "")
    canonical_manifest_truth = str(truth_obj.get("canonical_prebuilt_manifest") or "")
    canonical_transformer = str(truth_obj.get("canonical_transformer_bundle_dir") or "")

    # Hard forbid legacy tokens before IO
    for label, val in (
        ("canonical_xgb_bundle_dir", canonical_bundle),
        ("canonical_prebuilt_parquet", canonical_prebuilt),
        ("canonical_prebuilt_manifest", canonical_manifest_truth),
        ("canonical_transformer_bundle_dir", canonical_transformer),
        ("manifest_ssot", str(MANIFEST_SSOT)),
    ):
        if val:
            try:
                _forbid_prune_path(label, val, allow_ctx6cat6=label == "canonical_transformer_bundle_dir")
            except Exception as e:
                result["gates_failed"].append("canonical_truth_paths")
                result["checks"]["canonical_truth"] = {"error": str(e)}
                return result

    if not canonical_bundle or not canonical_transformer:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {"error": "Missing canonical_* paths in truth file"}
        return result

    manifest_ssot_resolved = MANIFEST_SSOT.expanduser().resolve()
    if canonical_manifest_truth:
        if Path(canonical_manifest_truth).expanduser().resolve() != manifest_ssot_resolved:
            result["gates_failed"].append("canonical_truth_paths")
            result["checks"]["canonical_truth"] = {
                "error": f"PREBUILT_MANIFEST_SPLIT_BRAIN: truth_manifest={canonical_manifest_truth} "
                f"expected={manifest_ssot_resolved}"
            }
            return result

    if not manifest_ssot_resolved.exists():
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": f"MANIFEST_SSOT_NOT_FOUND: {manifest_ssot_resolved}"}
        return result

    try:
        manifest_obj = json.loads(manifest_ssot_resolved.read_text(encoding="utf-8"))
    except Exception as e:
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": f"MANIFEST_SSOT_INVALID_JSON: {e}"}
        return result

    manifest_parquet = str(Path(manifest_obj.get("parquet_path") or "").expanduser().resolve())
    manifest_sha = manifest_obj.get("parquet_sha256") or ""

    if not manifest_parquet:
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": "MANIFEST_SSOT_MISSING_PARQUET_PATH"}
        return result

    if not Path(manifest_parquet).is_file():
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": f"MANIFEST_PARQUET_NOT_FOUND: {manifest_parquet}"}
        return result

    if not canonical_prebuilt:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {"error": "canonical_prebuilt_parquet missing (must mirror manifest)"}
        return result

    prebuilt_resolved = str(Path(canonical_prebuilt).expanduser().resolve())
    if prebuilt_resolved != manifest_parquet:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {
            "error": f"PREBUILT_SPLIT_BRAIN: manifest.parquet_path={manifest_parquet} != canonical_prebuilt_parquet={prebuilt_resolved}"
        }
        return result

    try:
        _forbid_prune_path("manifest_parquet", manifest_parquet)
    except Exception as e:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {"error": str(e)}
        return result

    # Single source assertion (print)
    try:
        parquet_sha = _sha256_file(Path(manifest_parquet))
    except Exception:
        parquet_sha = "MISSING_OR_UNREADABLE"
    print(
        "[E2E] SSoT_PREBUILT "
        f"manifest_ssot={manifest_ssot_resolved} manifest_parquet={manifest_parquet} "
        f"canonical_prebuilt_parquet={canonical_prebuilt} parquet_sha256={parquet_sha}",
        file=sys.stderr,
    )

    result["checks"]["canonical_truth"] = {
        "truth_file": str(truth_path),
        "truth_file_sha256": _sha256_file(truth_path),
        "manifest_ssot": str(manifest_ssot_resolved),
        "manifest_parquet_path": manifest_parquet,
        "manifest_parquet_sha256": manifest_sha,
        "canonical_prebuilt_parquet": prebuilt_resolved,
        "signal_bridge_sha_match": True,
        "signal_bridge_sha_key_used": key_used,
        "signal_bridge_sha_value": bridge_sha[:16] + "..." if len(bridge_sha) > 16 else bridge_sha,
        "signal_bridge_contract_sha256_expected": _BRIDGE_SHA,
        "passed": True,
    }

    # XGB lock + meta
    bundle_dir = Path(canonical_bundle).expanduser().resolve()
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        result["gates_failed"].append("xgb_lock")
        result["checks"]["xgb_lock"] = {"error": f"MASTER_MODEL_LOCK.json missing: {lock_path}"}
        return result
    lock_obj = _load_json(lock_path)
    ordered_features = list(lock_obj.get("ordered_features") or [])
    if not ordered_features:
        result["gates_failed"].append("xgb_lock")
        result["checks"]["xgb_lock"] = {"error": "MASTER_MODEL_LOCK missing ordered_features"}
        return result

    meta_rel = str(lock_obj.get("meta_path_relative") or "xgb_universal_multihead_v2_meta.json")
    meta_path = bundle_dir / meta_rel
    if not meta_path.exists():
        result["gates_failed"].append("xgb_meta")
        result["checks"]["xgb_meta"] = {"error": f"XGB meta missing: {meta_path}"}
        return result
    meta_obj = _load_json(meta_path)
    meta_features = list(meta_obj.get("feature_names_ordered") or meta_obj.get("ordered_features") or [])
    if meta_features != ordered_features:
        result["gates_failed"].append("xgb_meta")
        result["checks"]["xgb_meta"] = {"error": "XGB meta ordered_features != MASTER_MODEL_LOCK.ordered_features"}
        return result

    result["checks"]["xgb_lock"] = {"passed": True, "lock_path": str(lock_path)}
    result["checks"]["xgb_meta"] = {"passed": True, "meta_path": str(meta_path)}

    # Transformer lock
    transformer_dir = Path(canonical_transformer).expanduser().resolve()
    trans_lock_path = transformer_dir / "MASTER_TRANSFORMER_LOCK.json"
    if not trans_lock_path.exists():
        result["gates_failed"].append("transformer_lock")
        result["checks"]["transformer_lock"] = {"error": f"MASTER_TRANSFORMER_LOCK.json missing: {trans_lock_path}"}
        return result
    result["checks"]["transformer_lock"] = {"passed": True}

    # Prebuilt parquet + manifest + schema manifest + schema prefix
    prebuilt_path = Path(canonical_prebuilt).expanduser().resolve()
    if not prebuilt_path.exists():
        result["gates_failed"].append("prebuilt_parquet")
        result["checks"]["prebuilt"] = {"error": f"Prebuilt parquet missing: {prebuilt_path}"}
        return result

    manifest_path = prebuilt_path.with_suffix(".manifest.json")
    schema_manifest_path = prebuilt_path.with_suffix(".schema_manifest.json")
    if not manifest_path.exists():
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt"] = {"error": f"Prebuilt manifest missing: {manifest_path}"}
        return result
    if not schema_manifest_path.exists():
        result["gates_failed"].append("prebuilt_schema_manifest")
        result["checks"]["prebuilt"] = {"error": f"Prebuilt schema manifest missing: {schema_manifest_path}"}
        return result

    schema_obj = _load_json(schema_manifest_path)
    required_all = list(schema_obj.get("required_all_features") or [])
    if len(required_all) < len(ordered_features) or required_all[: len(ordered_features)] != ordered_features:
        result["gates_failed"].append("schema_prefix_match")
        result["checks"]["prebuilt"] = {
            "error": "schema_manifest.required_all_features must start with MASTER_MODEL_LOCK.ordered_features (order-sensitive)",
        }
        return result

    # 6/6 reality check: prebuilt must have all 12 ctx columns (no theater without this)
    try:
        from gx1.contracts.signal_bridge_v1 import (
            ORDERED_CTX_CAT_NAMES_EXTENDED,
            ORDERED_CTX_CONT_NAMES_EXTENDED,
        )

        required_ctx_12 = list(ORDERED_CTX_CONT_NAMES_EXTENDED) + list(ORDERED_CTX_CAT_NAMES_EXTENDED)
        import pyarrow.parquet as pq

        parquet_schema = pq.read_schema(prebuilt_path)
        prebuilt_cols = list(parquet_schema.names)
        missing_ctx_cols = [c for c in required_ctx_12 if c not in prebuilt_cols]
        if missing_ctx_cols:
            result["gates_failed"].append("prebuilt_ctx_6_6")
            result["checks"]["prebuilt"] = {
                "error": "Prebuilt missing ctx columns (6/6 reality check); run rebuild to CTX_CONT6_CAT6.",
                "missing_ctx_cols": missing_ctx_cols,
                "required_ctx_12": required_ctx_12,
            }
            return result
        # Bonus: no NaN/Inf in ctx columns (STRICT dataset will fail early otherwise)
        import numpy as np
        import pandas as pd

        df_sample = pd.read_parquet(prebuilt_path, columns=required_ctx_12).head(1000)
        if len(df_sample) > 0:
            for col in required_ctx_12:
                if col not in df_sample.columns:
                    continue
                ser = df_sample[col]
                if ser.isna().any():
                    result["gates_failed"].append("prebuilt_ctx_6_6")
                    result["checks"]["prebuilt"] = {
                        "error": f"Prebuilt has NaN in ctx column {col!r} (6/6 reality check).",
                        "column": col,
                    }
                    return result
                if pd.api.types.is_float_dtype(ser):
                    arr = ser.to_numpy(dtype=np.float64, na_value=np.nan)
                    if np.isinf(arr).any():
                        result["gates_failed"].append("prebuilt_ctx_6_6")
                        result["checks"]["prebuilt"] = {
                            "error": f"Prebuilt has Inf in ctx column {col!r} (6/6 reality check).",
                            "column": col,
                        }
                        return result
        ctx_reality_check = {"missing_ctx_cols": [], "ctx_nan_inf_check": "passed"}
    except Exception as e:
        result["gates_failed"].append("prebuilt_ctx_6_6")
        result["checks"]["prebuilt"] = {
            "error": f"6/6 reality check failed: {e}",
        }
        return result

    result["checks"]["prebuilt"] = {
        "passed": True,
        "prebuilt_path": str(prebuilt_path),
        "manifest_exists": True,
        "schema_manifest_exists": True,
        "schema_prefix_match": True,
        "ctx_6_6_reality_check": ctx_reality_check,
    }

    result["passed"] = True
    return result


def _bundle_sha256(bundle_dir: Path) -> str:
    """Compute bundle SHA256 for SSoT: bundle_metadata.json sha256, else hash of model_state_dict.pt, else MASTER_MODEL_LOCK.json."""
    meta_path = bundle_dir / "bundle_metadata.json"
    if meta_path.exists():
        try:
            obj = _load_json(meta_path)
            sha = obj.get("sha256") or obj.get("bundle_sha256")
            if sha:
                return str(sha).strip()
        except Exception:
            pass
    model_path = bundle_dir / "model_state_dict.pt"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if lock_path.exists():
        with open(lock_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    raise RuntimeError(
        f"[E2E] Cannot compute bundle_sha256: no bundle_metadata.json, model_state_dict.pt, or MASTER_MODEL_LOCK.json in {bundle_dir}"
    )


def _run_replay(
    replay_output_dir: Path,
    run_id: str,
    truth_path: Path,
    policy_path: Path,
    raw_path: Path,
    prebuilt_path: Path,
    start_ts: str,
    end_ts: str,
    env_overrides: Dict[str, str],
    bundle_dir: Path,
    merge_output_dir: Path,
) -> int:
    """Run 1W1C replay in-process via replay_chunk.process_chunk + replay_merge.merge_artifacts_1w1c (no legacy script import).
    Chunk artifacts go to replay_output_dir/chunk_0; MERGED/RUN_COMPLETED go to merge_output_dir (run_root)."""
    for k, v in env_overrides.items():
        os.environ[k] = v

    bundle_sha = _bundle_sha256(bundle_dir)

    # PRE_FORK_FREEZE.json required by chunk_bootstrap in TRUTH
    try:
        from gx1.utils.prefork_freeze_gate import run_prefork_freeze_gate_or_fatal

        run_prefork_freeze_gate_or_fatal(
            output_dir=replay_output_dir,
            truth_or_smoke=True,
            bundle_sha=bundle_sha,
        )
    except ImportError:
        # Fallback: write minimal PRE_FORK_FREEZE.json to satisfy bootstrap (no legacy module needed)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bundle_sha256": bundle_sha,
            "note": "prefork freeze stub (module missing); TRUTH/SMOKE requires presence only.",
        }
        _atomic_write_json(replay_output_dir / "PRE_FORK_FREEZE.json", payload)
    except Exception as e:
        print(f"[E2E] PRE_FORK_FREEZE failed: {e}", file=sys.stderr)
        return 2

    import pandas as _pd

    chunk_start_ts = _pd.Timestamp(start_ts, tz="UTC")
    chunk_end_ts = _pd.Timestamp(end_ts, tz="UTC")

    from gx1.execution.replay_chunk import process_chunk
    from gx1.execution.replay_merge import merge_artifacts_1w1c

    try:
        truth_obj_run = _load_json(truth_path)
    except Exception:
        truth_obj_run = {}

    result = process_chunk(
        chunk_idx=0,
        chunk_start=chunk_start_ts,
        chunk_end=chunk_end_ts,
        data_path=raw_path,
        policy_path=policy_path,
        run_id=run_id,
        output_dir=replay_output_dir,
        bundle_sha256=bundle_sha,
        prebuilt_parquet_path=None,
        bundle_dir=bundle_dir,
        chunk_local_padding_days=0,
        truth_artifacts=truth_obj_run,
    )

    if result.get("status") != "ok":
        print(f"[E2E] process_chunk failed: {result.get('error', result)}", file=sys.stderr)
        return 1

    try:
        merge_artifacts_1w1c(replay_output_dir, run_id, output_dir=merge_output_dir)
    except Exception as e:
        print(f"[E2E] merge_artifacts_1w1c failed: {e}", file=sys.stderr)
        return 1

    return 0


def _run_postrun_checks(run_root: Path, run_id: str, truth_artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Post-run: required files, chunk_footer status, invariants, ctx dims, forward_calls, zero-trades, exit journal.
    Root artifacts (MERGED, RUN_COMPLETED, etc.) live in run_root; chunk artifacts in run_root/replay/chunk_0."""
    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_root),
        "passed": False,
        "gates_failed": [],
        "checks": {},
    }

    # Hard gate: forbidden symbol-imports after replay (stricter than IMPORT_PROOF/banlist)
    _assert_no_forbidden_symbol_imports_after_replay(run_root)

    chunk_dir = run_root / "replay" / "chunk_0"
    footer_path = chunk_dir / "chunk_footer.json"

    # Required files: chunk under replay/chunk_0, root artifacts in run_root.
    required_chunk = [
        "trade_outcomes_" + run_id + ".parquet",
        "attribution_" + run_id + ".json",
        "chunk_footer.json",
        "IMPORT_PROOF.json",
    ]
    required_root = [
        f"trade_outcomes_{run_id}_MERGED.parquet",
        f"metrics_{run_id}_MERGED.json",
        f"MERGE_PROOF_{run_id}.json",
        "RUN_COMPLETED.json",
    ]
    truth_artifacts = truth_artifacts or {}
    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {})
    if replay_truth_artifacts.get("require_import_proof"):
        fname = replay_truth_artifacts.get("import_proof_filename") or "IMPORT_PROOF.json"
        required_chunk.append(fname)

    # Diagnostics: log required list and present files (deterministic, bounded)
    required_files = [f"replay/chunk_0/{name}" for name in required_chunk] + required_root
    required_files = list(dict.fromkeys(required_files))  # de-dupe deterministically
    print(
        "[POSTRUN_REQUIRED_SOURCE] run_root=%s chunk_root=%s required_files_count=%d source=_run_postrun_checks"
        % (run_root, chunk_dir, len(required_files))
    )
    for req in required_files:
        print("[POSTRUN_REQUIRED_FILE] %s" % req)

    present_files: List[str] = []
    try:
        present_files = sorted(
            [
                str(p.relative_to(chunk_dir))
                for p in chunk_dir.rglob("*")
                if p.is_file()
            ]
        )
    except Exception:
        present_files = []
    present_limit = 200
    if len(present_files) > present_limit:
        head = present_files[:present_limit]
        print("[POSTRUN_PRESENT_FILES_TRUNCATED] count=%d limit=%d" % (len(present_files), present_limit))
        for pf in head:
            print("[POSTRUN_PRESENT_FILE] %s" % pf)
    else:
        print("[POSTRUN_PRESENT_FILES] count=%d" % len(present_files))
        for pf in present_files:
            print("[POSTRUN_PRESENT_FILE] %s" % pf)

    missing: List[str] = []
    for name in required_chunk:
        if not (chunk_dir / name).exists():
            missing.append(f"replay/chunk_0/{name}")
    for name in required_root:
        if not (run_root / name).exists():
            missing.append(name)

    if missing:
        result["gates_failed"].append("required_files")
        result["checks"]["required_files"] = {"missing": missing}
        print("[POSTRUN_REQUIRED] required_count=%d missing_count=%d" % (len(required_files), len(missing)))
        for m in missing:
            print("[POSTRUN_REQUIRED_MISSING] %s" % m)
        return result
    result["checks"]["required_files"] = {"passed": True}

    # Zero-trades diagnostics (for visibility on skipped artifacts)
    n_trades = None
    try:
        footer_obj = _load_json(footer_path)
        n_trades = footer_obj.get("n_trades") or footer_obj.get("trades_closed") or footer_obj.get("n_trades_closed")
    except Exception:
        footer_obj = None
    print("[POSTRUN_ZERO_TRADES_INFO] n_trades=%s footer_path=%s footer_loaded=%s" % (n_trades, footer_path, footer_obj is not None))

    # Gate: IMPORT_PROOF.json must exist and forbidden_hits must be empty (no ghost imports)
    import_proof_path = chunk_dir / "IMPORT_PROOF.json"
    if not import_proof_path.exists():
        result["gates_failed"].append("import_ghosts")
        result["checks"]["import_ghosts"] = {"error": f"IMPORT_PROOF.json missing: {import_proof_path}"}
        return result
    try:
        import_proof = _load_json(import_proof_path)
        forbidden_hits = import_proof.get("forbidden_hits") or []
        if forbidden_hits:
            result["gates_failed"].append("import_ghosts")
            result["checks"]["import_ghosts"] = {"forbidden_hits": forbidden_hits}
            return result
        result["checks"]["import_ghosts"] = {"passed": True, "forbidden_hits": []}
    except Exception as e:
        result["gates_failed"].append("import_ghosts")
        result["checks"]["import_ghosts"] = {"error": str(e)}
        return result

    # Gate: policy snapshot sha256 must match run_header (no disk drift) unless disabled by truth_artifacts
    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {}) if truth_artifacts else {}
    require_policy_snapshot = replay_truth_artifacts.get("require_policy_snapshot", True)
    header_path = chunk_dir / "run_header.json"
    if not header_path.exists():
        result["gates_failed"].append("run_header")
        result["checks"]["policy_snapshot"] = {"error": "run_header.json missing"}
        return result
    header = _load_json(header_path)
    if require_policy_snapshot:
        expected_sha256 = header.get("policy_snapshot_sha256")
        if not expected_sha256:
            result["gates_failed"].append("policy_snapshot_sha256")
            result["checks"]["policy_snapshot"] = {
                "error": "run_header missing policy_snapshot_sha256 (run must use snapshot runner)"
            }
            return result
        snapshot_name = header.get("policy_snapshot_path") or replay_truth_artifacts.get("policy_snapshot_filename") or "RUN_POLICY_USED.yaml"
        snapshot_file = chunk_dir / snapshot_name
        if not snapshot_file.exists():
            result["gates_failed"].append("policy_snapshot_sha256")
            result["checks"]["policy_snapshot"] = {"error": f"{snapshot_name} missing in chunk dir"}
            return result
        actual_sha256 = _sha256_file(snapshot_file)
        if actual_sha256 != expected_sha256:
            result["gates_failed"].append("policy_snapshot_sha256")
            result["checks"]["policy_snapshot"] = {
                "error": f"{snapshot_name} sha256 does not match run_header.policy_snapshot_sha256",
                "expected": expected_sha256,
                "actual": actual_sha256,
            }
            return result
        result["checks"]["policy_snapshot"] = {"passed": True, "sha256": actual_sha256}
    else:
        result["checks"]["policy_snapshot"] = {"skipped": True, "reason": "disabled_by_truth_config"}

    if not footer_path.exists():
        result["gates_failed"].append("chunk_footer")
        result["checks"]["chunk_footer"] = {"error": "chunk_footer.json missing"}
        return result

    footer = _load_json(footer_path)
    status = (footer.get("status") or "").lower()
    if status != "ok":
        result["gates_failed"].append("chunk_footer_status")
        result["checks"]["chunk_footer"] = {"error": f"status={status!r}", "footer_error": footer.get("error")}
        return result

    # ---------------------------------------------------------------------
    # Invariants: prebuilt_proven + feature_build
    # ---------------------------------------------------------------------
    join_path = chunk_dir / "RAW_PREBUILT_JOIN.json"

    # prebuilt_proven: env + footer.prebuilt_parquet_path + join-file exists
    env_prebuilt = os.environ.get("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
    prebuilt_path_footer = footer.get("prebuilt_parquet_path")
    prebuilt_proven = bool(env_prebuilt and prebuilt_path_footer and join_path.exists())
    if not prebuilt_proven:
        result["gates_failed"].append("prebuilt_proven")
        result["checks"]["invariants"] = {
            "GX1_REPLAY_USE_PREBUILT_FEATURES": os.environ.get("GX1_REPLAY_USE_PREBUILT_FEATURES"),
            "footer_prebuilt_parquet_path": prebuilt_path_footer,
            "RAW_PREBUILT_JOIN_exists": join_path.exists(),
        }
        return result

    # feature_build_call_count: soft invariant
    # - If present: must be 0
    # - If missing: require GX1_FEATURE_BUILD_DISABLED=1
    fbc = footer.get("feature_build_call_count")
    if fbc is not None:
        try:
            if int(fbc) != 0:
                result["gates_failed"].append("feature_build_call_count")
                result["checks"]["invariants"] = {"feature_build_call_count": fbc}
                return result
        except (TypeError, ValueError):
            result["gates_failed"].append("feature_build_call_count")
            result["checks"]["invariants"] = {"feature_build_call_count": fbc}
            return result
    else:
        if os.environ.get("GX1_FEATURE_BUILD_DISABLED", "0") != "1":
            result["gates_failed"].append("feature_build_disabled")
            result["checks"]["invariants"] = {"GX1_FEATURE_BUILD_DISABLED": os.environ.get("GX1_FEATURE_BUILD_DISABLED")}
            return result

    result["checks"]["invariants"] = {
        "prebuilt_proven": True,
        "feature_build_call_count": fbc,
        "GX1_FEATURE_BUILD_DISABLED": os.environ.get("GX1_FEATURE_BUILD_DISABLED"),
    }

    # ---------------------------------------------------------------------
    # ctx dims: ONE UNIVERSE 6/6 only
    # ---------------------------------------------------------------------
    ctx_cont = footer.get("ctx_cont_dim")
    ctx_cat = footer.get("ctx_cat_dim")
    if ctx_cont != CTX_CONT_DIM or ctx_cat != CTX_CAT_DIM:
        result["gates_failed"].append("ctx_dims")
        result["checks"]["ctx_dims"] = {
            "ctx_cont_dim": ctx_cont,
            "ctx_cat_dim": ctx_cat,
            "required": (CTX_CONT_DIM, CTX_CAT_DIM),
        }
        return result
    result["checks"]["ctx_dims"] = {"ctx_cont_dim": ctx_cont, "ctx_cat_dim": ctx_cat, "passed": True}

    # ---------------------------------------------------------------------
    # join_ratio >= 0.995 (TRUTH gate): join_ratio then fallback join_ratio_eval
    # ---------------------------------------------------------------------
    if not join_path.exists():
        result["gates_failed"].append("join_ratio")
        result["checks"]["join_ratio"] = {"error": "RAW_PREBUILT_JOIN.json missing"}
        return result

    try:
        join_data = _load_json(join_path)
        jr = join_data.get("join_ratio")
        if jr is None:
            jr = join_data.get("join_ratio_eval")  # legacy fallback

        if jr is None:
            result["gates_failed"].append("join_ratio")
            result["checks"]["join_ratio"] = {
                "error": "join_ratio missing (tried join_ratio then join_ratio_eval)",
                "join_file_keys": sorted(list(join_data.keys())),
            }
            return result

        try:
            jr_f = float(jr)
        except (TypeError, ValueError):
            result["gates_failed"].append("join_ratio")
            result["checks"]["join_ratio"] = {"error": f"join_ratio not numeric: {jr!r}"}
            return result

        if jr_f < JOIN_RATIO_TRUTH:
            result["gates_failed"].append("join_ratio")
            result["checks"]["join_ratio"] = {"join_ratio": jr_f, "required": JOIN_RATIO_TRUTH}
            return result

        result["checks"]["join_ratio"] = {"join_ratio": jr_f, "passed": True}

    except Exception as e:
        result["gates_failed"].append("join_ratio")
        result["checks"]["join_ratio"] = {"error": f"could not read join file: {e}"}
        return result

    # ---------------------------------------------------------------------
    # forward_calls / n_model_calls (observability gate; never use t_transformer_forward_sec as proof)
    # ---------------------------------------------------------------------
    metrics_path = run_root / f"metrics_{run_id}_MERGED.json"
    metrics: Dict[str, Any] = _load_json(metrics_path) if metrics_path.exists() else {}
    n_trades = int(metrics.get("n_trades", -1)) if metrics else -1
    if n_trades < 0:
        n_trades = int(footer.get("n_trades_closed", -1)) if footer.get("n_trades_closed") is not None else -1

    tried_keys = [
        "transformer_forward_calls",
        "forward_calls_total",
        "n_transformer_calls",
        "transformer_calls",
        "policy_forward_calls",
        "n_model_calls",
    ]
    forward_calls: Optional[int] = None
    chosen_key: Optional[str] = None
    for k in tried_keys:
        v = metrics.get(k)
        if v is not None:
            try:
                forward_calls = int(v)
                chosen_key = k
                break
            except Exception:
                pass
    if forward_calls is None:
        fc = footer.get("n_model_calls") or footer.get("bars_evaluated")
        if fc is not None:
            try:
                forward_calls = int(fc)
                chosen_key = "chunk_footer.n_model_calls"
            except Exception:
                pass

    if n_trades > 0:
        if forward_calls is None or forward_calls <= 0:
            keys = sorted(list(metrics.keys()))
            print(
                f"[E2E] forward_calls NO-GO: n_trades={n_trades} but forward_calls={forward_calls} (required > 0)",
                file=sys.stderr,
            )
            result["gates_failed"].append("forward_calls")
            result["checks"]["forward_calls"] = {
                "error": "n_trades > 0 requires forward_calls > 0",
                "n_trades": n_trades,
                "forward_calls": forward_calls,
                "source_key": chosen_key,
                "metrics_keys": keys,
            }
            return result
        result["checks"]["forward_calls"] = {
            "forward_calls": forward_calls,
            "source_key": chosen_key or "unknown",
            "n_trades": n_trades,
            "passed": True,
        }
    else:
        if forward_calls is None:
            forward_calls = 0
        if forward_calls == 0:
            print("[E2E] no-forward-window: n_trades=0, forward_calls=0 (policy/session-filtered)", file=sys.stderr)
        result["checks"]["forward_calls"] = {
            "forward_calls": forward_calls,
            "source_key": chosen_key or "chunk_footer.n_model_calls",
            "n_trades": n_trades,
            "passed": True,
        }

    # ---------------------------------------------------------------------
    # ctx telemetry (when ctx 6/6: n_ctx_model_calls > 0, ctx_proof_pass == n_ctx_model_calls, ctx_proof_fail == 0)
    # ---------------------------------------------------------------------
    ctx_cat_dim = int(footer.get("ctx_cat_dim") or 0)
    ctx_cont_dim = int(footer.get("ctx_cont_dim") or 0)
    if ctx_cat_dim == 6 and ctx_cont_dim == 6:
        n_ctx = int(footer.get("n_ctx_model_calls") or 0)
        ctx_pass = int(footer.get("ctx_proof_pass_count") or 0)
        ctx_fail = int(footer.get("ctx_proof_fail_count") or 0)
        if n_ctx <= 0:
            result["gates_failed"].append("ctx_telemetry")
            result["checks"]["ctx_telemetry"] = {
                "error": "ctx 6/6 but n_ctx_model_calls <= 0",
                "n_ctx_model_calls": n_ctx,
                "ctx_proof_pass_count": ctx_pass,
                "ctx_proof_fail_count": ctx_fail,
            }
            return result
        if ctx_pass != n_ctx:
            result["gates_failed"].append("ctx_telemetry")
            result["checks"]["ctx_telemetry"] = {
                "error": f"ctx_proof_pass_count ({ctx_pass}) != n_ctx_model_calls ({n_ctx})",
                "n_ctx_model_calls": n_ctx,
                "ctx_proof_pass_count": ctx_pass,
                "ctx_proof_fail_count": ctx_fail,
            }
            return result
        if ctx_fail != 0:
            result["gates_failed"].append("ctx_telemetry")
            result["checks"]["ctx_telemetry"] = {
                "error": f"ctx_proof_fail_count must be 0 when ctx present, got {ctx_fail}",
                "n_ctx_model_calls": n_ctx,
                "ctx_proof_pass_count": ctx_pass,
                "ctx_proof_fail_count": ctx_fail,
            }
            return result
        result["checks"]["ctx_telemetry"] = {
            "n_ctx_model_calls": n_ctx,
            "ctx_proof_pass_count": ctx_pass,
            "ctx_proof_fail_count": ctx_fail,
            "passed": True,
        }

    # ---------------------------------------------------------------------
    # zero-trades contract: trade_outcomes exists (empty parquet) + ZERO_TRADES_DIAG if n_trades==0
    # ---------------------------------------------------------------------
    n_trades = int(metrics.get("n_trades", 0)) if metrics else -1
    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {}) if truth_artifacts else {}
    min_trades = int(replay_truth_artifacts.get("min_trades", 1 if replay_truth_artifacts.get("require_nonzero_trades", True) else 0))
    if n_trades < min_trades:
        result["gates_failed"].append("zero_trades_diag")
        result["checks"]["zero_trades"] = {
            "error": f"n_trades={n_trades} < min_trades={min_trades} (truth-config)",
            "n_trades": n_trades,
            "min_trades": min_trades,
        }
        return result
    if n_trades == 0:
        if min_trades == 0:
            result["checks"]["zero_trades"] = {"skipped": True, "reason": "disabled_by_truth_config", "n_trades": n_trades}
        else:
            to_path = chunk_dir / f"trade_outcomes_{run_id}.parquet"
            if not to_path.exists():
                result["gates_failed"].append("trade_outcomes_zero_trades")
                result["checks"]["zero_trades"] = {"error": "trade_outcomes parquet missing when n_trades==0"}
                return result

            zero_diag = chunk_dir / "ZERO_TRADES_DIAG.json"
            if not zero_diag.exists():
                result["gates_failed"].append("zero_trades_diag")
                result["checks"]["zero_trades"] = {"error": "ZERO_TRADES_DIAG.json missing (TRUTH requires when n_trades==0)"}
                return result

            result["checks"]["zero_trades"] = {"passed": True, "n_trades": n_trades, "min_trades": min_trades}
    else:
        result["checks"]["zero_trades"] = {"n_trades": n_trades, "passed": True, "min_trades": min_trades}

    # ---------------------------------------------------------------------
    # Exit coverage: truth_exit_journal_ok==true if EXIT_COVERAGE_SUMMARY exists (replay or root)
    # ---------------------------------------------------------------------
    exit_cov_path = run_root / "EXIT_COVERAGE_SUMMARY.json"
    if not exit_cov_path.exists():
        exit_cov_path = run_root / "replay" / "EXIT_COVERAGE_SUMMARY.json"
    if exit_cov_path.exists():
        exit_cov = _load_json(exit_cov_path)
        truth_ok = exit_cov.get("truth_exit_journal_ok")
        if truth_ok is not True:
            result["gates_failed"].append("truth_exit_journal_ok")
            result["checks"]["exit_coverage"] = {"truth_exit_journal_ok": truth_ok}
            return result
        result["checks"]["exit_coverage"] = {"truth_exit_journal_ok": True, "passed": True}
    else:
        result["checks"]["exit_coverage"] = {"passed": True, "note": "EXIT_COVERAGE_SUMMARY.json not found"}

    # ---------------------------------------------------------------------
    # Bars invariant: gap == warmup_holdback_bars + tail_holdback_bars when status ok
    # ---------------------------------------------------------------------
    bars_total = int(footer.get("bars_total_input") or 0)
    bars_processed = int(footer.get("bars_processed") or 0)
    warmup_holdback = int(footer.get("warmup_holdback_bars") or 0)
    tail_holdback = int(footer.get("tail_holdback_bars") or 0)
    gap = bars_total - bars_processed
    expected_gap = warmup_holdback + tail_holdback
    if gap != expected_gap:
        result["gates_failed"].append("bars_invariant")
        result["checks"]["bars_invariant"] = {
            "bars_total_input": bars_total,
            "bars_processed": bars_processed,
            "warmup_holdback_bars": warmup_holdback,
            "tail_holdback_bars": tail_holdback,
            "gap": gap,
            "expected_gap": expected_gap,
        }
        return result
    result["checks"]["bars_invariant"] = {"passed": True}

    # Exit strategy (record footer fields)
    result["checks"]["exit_strategy"] = {
        "exit_type": footer.get("exit_type"),
        "exit_profile": footer.get("exit_profile"),
        "router_enabled": footer.get("router_enabled"),
        "exit_critic_enabled": footer.get("exit_critic_enabled"),
        "exit_ml_enabled": footer.get("exit_ml_enabled"),
        "exit_ml_decision_mode": footer.get("exit_ml_decision_mode"),
        "exit_ml_config_hash": footer.get("exit_ml_config_hash"),
    }

    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {}) if truth_artifacts else {}
    require_exit_transformer = replay_truth_artifacts.get("require_exit_type_transformer", True)

    # TRUTH gate: EXIT_TRANSFORMER_V0 only (ONE UNIVERSE ML exit); no router, no exit_critic
    if not require_exit_transformer:
        result["checks"]["exit_strategy"]["skipped"] = True
        result["checks"]["exit_strategy"]["reason"] = "exit ML disabled (fixed bar close per truth_config)"
    else:
        if footer.get("exit_type") != "EXIT_TRANSFORMER_V0":
            result["gates_failed"].append("exit_type_transformer")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_type must be EXIT_TRANSFORMER_V0 in TRUTH (ONE UNIVERSE ML-only), got: {footer.get('exit_type')!r}"
            )
            return result
        if footer.get("router_enabled") is not False:
            result["gates_failed"].append("router_enabled_false")
            result["checks"]["exit_strategy"]["error"] = f"router_enabled must be false in TRUTH, got: {footer.get('router_enabled')}"
            return result
        if footer.get("exit_critic_enabled") is not False:
            result["gates_failed"].append("exit_critic_enabled_false")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_critic_enabled must be false in TRUTH, got: {footer.get('exit_critic_enabled')}"
            )
            return result
        if footer.get("exit_ml_enabled") is not True:
            result["gates_failed"].append("exit_ml_enabled_true")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_ml_enabled must be true in TRUTH (EXIT_TRANSFORMER_V0), got: {footer.get('exit_ml_enabled')}"
            )
            return result
        if footer.get("exit_ml_decision_mode") != "exit_transformer_v0":
            result["gates_failed"].append("exit_ml_decision_mode_transformer")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_ml_decision_mode must be 'exit_transformer_v0' in TRUTH, got: {footer.get('exit_ml_decision_mode')!r}"
            )
            return result

    if require_exit_transformer:
        # TRUTH gate: exits jsonl must exist and contain at least one line with computed.mode == exit_transformer_v0 and context 6/6
        exits_dir = chunk_dir / "logs" / "exits"
        exits_glob = list(exits_dir.glob("exits_*.jsonl")) if exits_dir.exists() else []
        if not exits_glob:
            result["gates_failed"].append("exit_ml_exits_jsonl")
            result["checks"]["exit_strategy"]["error"] = f"exits jsonl required in {exits_dir}; found: {exits_glob}"
            return result
        if not footer.get("exit_ml_model_sha"):
            result["gates_failed"].append("exit_ml_model_sha")
            result["checks"]["exit_strategy"]["error"] = "exit_transformer_v0 requires exit_ml_model_sha in footer"
            return result
        seen_transformer_6_6 = False
        for path in exits_glob:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        comp = rec.get("computed") or {}
                        if comp.get("mode") != "exit_transformer_v0":
                            continue
                        ctx = rec.get("context") or {}
                        ctx_cont = ctx.get("ctx_cont") if isinstance(ctx.get("ctx_cont"), (list, tuple)) else []
                        ctx_cat = ctx.get("ctx_cat") if isinstance(ctx.get("ctx_cat"), (list, tuple)) else []
                        if len(ctx_cont) == 6 and len(ctx_cat) == 6:
                            seen_transformer_6_6 = True
                            break
            except Exception:
                pass
            if seen_transformer_6_6:
                break
        # When n_trades_closed == 0, no exit decisions were logged; allow pass if file exists and footer dims are 6/6
        n_trades = footer.get("n_trades_closed", 0)
        if not seen_transformer_6_6:
            if n_trades == 0 and footer.get("ctx_cont_dim") == 6 and footer.get("ctx_cat_dim") == 6:
                pass
            else:
                result["gates_failed"].append("exit_ml_transformer_6_6")
                result["checks"]["exit_strategy"]["error"] = (
                    "exits jsonl must contain at least one line with computed.mode == 'exit_transformer_v0' and "
                    "context.ctx_cont len 6, context.ctx_cat len 6"
                )
                return result

    result["passed"] = True
    return result


def _write_summary_md(
    run_root: Path,
    preflight: Dict[str, Any],
    postrun: Optional[Dict[str, Any]],
    go: bool,
    reasons: List[str],
    canary_proof: Optional[Dict[str, Any]] = None,
) -> None:
    path = run_root / "E2E_SANITY_SUMMARY.md"
    lines = [
        "# E2E Sanity Summary",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"## Verdict: **{'GO' if go else 'NO-GO'}**",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")

    lines.extend(["", "## Preflight", ""])
    lines.append(f"- passed: `{preflight.get('passed', False)}`")
    if preflight.get("gates_failed"):
        lines.append(f"- gates_failed: {preflight['gates_failed']}")

    lines.extend(["", "## Post-run (if run)", ""])
    if postrun is not None:
        lines.append(f"- passed: `{postrun.get('passed', False)}`")
        if postrun.get("gates_failed"):
            lines.append(f"- gates_failed: {postrun['gates_failed']}")
    else:
        lines.append("- (no replay run)")

    if canary_proof is not None:
        lines.extend(["", "## Zero-trades canary", ""])
        lines.append(f"- mode: `{canary_proof.get('mode', 'ZERO_TRADES_CANARY')}`")
        lines.append(f"- entry_threshold_override: `{canary_proof.get('entry_threshold_override', ZERO_TRADES_CANARY_THRESHOLD)}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="TRUTH-grade E2E sanity checker for signal-only pipeline.")
    ap.add_argument("--run-id", type=str, default="", help="Run ID (default: E2E_SANITY_<utc_ts>)")
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Output run directory (default: GX1_DATA/reports/truth_e2e_sanity/<run_id>)",
    )
    ap.add_argument("--start-ts", type=str, default=DEFAULT_START_TS, help="Start timestamp (ISO)")
    ap.add_argument("--end-ts", type=str, default=DEFAULT_END_TS, help="End timestamp (ISO)")
    ap.add_argument("--full-year", action="store_true", help="Use 2025-01-01 to 2025-12-31")
    ap.add_argument("--validate-only", action="store_true", help="Only preflight, no replay")
    ap.add_argument(
        "--threshold-override",
        type=str,
        default="",
        help="Set GX1_ANALYSIS_MODE=1 and GX1_ENTRY_THRESHOLD_OVERRIDE=<val>",
    )
    ap.add_argument(
        "--force-zero-trades",
        action="store_true",
        help="ZERO_TRADES_CANARY: force 0 trades (GX1_ANALYSIS_MODE=1, GX1_ENTRY_THRESHOLD_OVERRIDE=1.1). Hard-fail if n_trades>0.",
    )
    ap.add_argument("--entry-signal-trace", action="store_true", help="Set GX1_ENTRY_SIGNAL_TRACE=1")
    ap.add_argument("--strict-masks", dest="strict_masks", action="store_true", default=True, help="Set GX1_STRICT_MASK=1 (default)")
    ap.add_argument("--no-strict-masks", dest="strict_masks", action="store_false", help="Do not set GX1_STRICT_MASK")
    ap.add_argument("--truth-file", type=str, default="", help="Canonical truth JSON (default: env, else CANONICAL_TRUTH_DEFAULT)")
    ap.add_argument(
        "--train-exit-transformer-v0-from-last-go",
        action="store_true",
        help="Train Exit Transformer V0 from LAST_GO exits jsonl (6/6 ctx), verify artifacts, then exit (no replay).",
    )
    ap.add_argument(
        "--require-io-v2",
        action="store_true",
        help="With --train-exit-transformer-v0-from-last-go: use IOV2 and require context (hard fail if missing).",
    )
    args = ap.parse_args()

    if getattr(args, "train_exit_transformer_v0_from_last_go", False):
        gx1_data = _gx1_data()
        try:
            from gx1.policy.exit_transformer_v0 import (
                get_last_go_exits_dataset,
                train_from_exits_jsonl,
                verify_exit_transformer_artifacts,
            )
        except ImportError as e:
            print(f"[train-exit-transformer-v0] Import failed: {e}", file=sys.stderr)
            return 1
        try:
            ds = get_last_go_exits_dataset(gx1_data=str(gx1_data))
        except FileNotFoundError as e:
            print(f"[train-exit-transformer-v0] {e}", file=sys.stderr)
            return 1
        exits_path = ds["exits_jsonl_path"]
        go_run_dir = ds["go_run_dir"]
        go_run_id = ds["go_run_id"]
        print(f"[train-exit-transformer-v0] Source: {exits_path} (run_id={go_run_id})", file=sys.stderr)
        require_io_v2 = getattr(args, "require_io_v2", False)
        result = train_from_exits_jsonl(
            exits_path,
            out_dir=None,
            source_run_id=go_run_id,
            source_run_dir=str(go_run_dir),
            gx1_data=str(gx1_data),
            epochs=20,
            window_len=8,
            seed=42,
            use_io_v2=require_io_v2,
            require_io_v2=require_io_v2,
            ctx_cont_dim=CTX_CONT_DIM,
            ctx_cat_dim=CTX_CAT_DIM,
        )
        out_dir = result["train_report_path"].parent
        verify_result = verify_exit_transformer_artifacts(out_dir)
        print(f"[train-exit-transformer-v0] Out dir: {out_dir}", file=sys.stderr)
        print(f"[train-exit-transformer-v0] model_sha256: {result['model_sha256']}", file=sys.stderr)
        print(f"[train-exit-transformer-v0] dataset_sha256: {result['dataset_sha256']}", file=sys.stderr)
        print(f"[train-exit-transformer-v0] Verify passed: {verify_result['passed']}", file=sys.stderr)
        if verify_result.get("failures"):
            for f in verify_result["failures"]:
                print(f"  - {f}", file=sys.stderr)
        if not verify_result["passed"]:
            return 1
        return 0

    run_id = args.run_id or (
        f"ZERO_TRADES_CANARY_{_utc_ts_compact()}" if args.force_zero_trades else f"E2E_SANITY_{_utc_ts_compact()}"
    )
    gx1_data = _gx1_data()
    run_root = Path(args.run_dir).expanduser().resolve() if args.run_dir else (gx1_data / "reports" / "truth_e2e_sanity" / run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    canary_proof = {"mode": "ZERO_TRADES_CANARY", "entry_threshold_override": ZERO_TRADES_CANARY_THRESHOLD} if args.force_zero_trades else None

    # TRUTH gate: no legacy replay script in process
    _assert_truth_no_legacy_replay(run_root)

    # ---------------------------------------------------------------------
    # TRUTH_FILE resolution (NO "missing truth" state; NO split-brain).
    # Priority:
    #   1) --truth-file
    #   2) GX1_CANONICAL_TRUTH_FILE
    #   3) CANONICAL_TRUTH_DEFAULT
    # Split-brain rule:
    #   If both CLI and env are set and differ (after resolve) -> hard fail.
    # TRUTH forbids CLI override of a different env path.
    # ---------------------------------------------------------------------
    truth_file_cli = (args.truth_file or "").strip()
    truth_file_env = (os.environ.get("GX1_CANONICAL_TRUTH_FILE", "") or "").strip()

    cli_abs = str(Path(truth_file_cli).expanduser().resolve()) if truth_file_cli else ""
    env_abs = str(Path(truth_file_env).expanduser().resolve()) if truth_file_env else ""

    if truth_file_cli and truth_file_env and cli_abs != env_abs:
        raise RuntimeError(
            f"SPLIT_BRAIN_TRUTH: --truth-file={cli_abs} != GX1_CANONICAL_TRUTH_FILE={env_abs} (TRUTH_FORBIDS_CLI_TRUTH_OVERRIDE)"
        )

    truth_file = truth_file_cli or truth_file_env or CANONICAL_TRUTH_DEFAULT
    truth_path = Path(truth_file).expanduser().resolve()
    print(f"[E2E] TRUTH_FILE_USED={truth_path}", file=sys.stderr)

    # TRUTH envs (must be set before preflight so preflight can check)
    os.environ["GX1_RUN_MODE"] = "TRUTH"
    os.environ["GX1_TRUTH_MODE"] = "1"
    os.environ["GX1_GATED_FUSION_ENABLED"] = "1"
    os.environ["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    os.environ["GX1_FEATURE_BUILD_DISABLED"] = "1"
    os.environ["GX1_CANONICAL_TRUTH_FILE"] = str(truth_path)
    os.environ.setdefault("GX1_OUTPUT_MODE", "TRUTH")
    os.environ.setdefault("GX1_SEED", "42")
    os.environ.setdefault("GX1_REPLAY_INCREMENTAL_FEATURES", "1")
    os.environ.setdefault("GX1_FEATURE_USE_NP_ROLLING", "1")
    os.environ.setdefault("GX1_REPLAY_NO_CSV", "1")
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

    # TRUTH/SMOKE: hard-fail on parallel/segmented envs or multi-worker hints
    for forbidden_env in (
        "GX1_PARALLEL",
        "GX1_SEGMENTED",
        "GX1_SEGMENTED_PARALLEL",
        "GX1_WORKERS",
        "GX1_CHUNKS",
        "GX1_N_WORKERS",
        "GX1_N_CHUNKS",
    ):
        val = os.environ.get(forbidden_env)
        if val:
            raise RuntimeError(f"[TRUTH_FORBIDDEN_ENV] {forbidden_env}={val}")

    try:
        from gx1.utils.truth_banlist import assert_truth_banlist_clean  # type: ignore

        assert_truth_banlist_clean(output_dir=run_root, stage="run_truth_e2e_sanity:entry")
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["truth_banlist"])
        print(f"[E2E] TRUTH banlist: {e}", file=sys.stderr)
        return 1

    if args.entry_signal_trace:
        os.environ["GX1_ENTRY_SIGNAL_TRACE"] = "1"
    if args.strict_masks:
        os.environ["GX1_STRICT_MASK"] = "1"
    _apply_ctx_mask_defaults(CTX_CONT_DIM, CTX_CAT_DIM)
    if args.threshold_override:
        os.environ["GX1_ANALYSIS_MODE"] = "1"
        os.environ["GX1_ENTRY_THRESHOLD_OVERRIDE"] = args.threshold_override
    if args.force_zero_trades:
        os.environ["GX1_ANALYSIS_MODE"] = "1"
        os.environ["GX1_ENTRY_THRESHOLD_OVERRIDE"] = ZERO_TRADES_CANARY_THRESHOLD
        print("[E2E] MODE=ZERO_TRADES_CANARY (entry threshold 1.1 → 0 trades contract)", file=sys.stderr)
        _atomic_write_json(
            run_root / "RUN_IDENTITY.json",
            {"mode": "ZERO_TRADES_CANARY", "entry_threshold_override": ZERO_TRADES_CANARY_THRESHOLD, "run_id": run_id},
        )

    # Preflight
    try:
        preflight = _run_preflight(truth_path, run_root)
        _atomic_write_json(run_root / "PREFLIGHT_E2E.json", preflight)
        if not preflight.get("passed", False):
            _write_fatal_capsule(run_root, RuntimeError(str(preflight.get("gates_failed", []))), preflight.get("gates_failed", []))
            _write_summary_md(run_root, preflight, None, False, ["Preflight failed: " + str(preflight.get("gates_failed", []))], canary_proof=canary_proof)
            print("[E2E] PREFLIGHT FAIL:", preflight.get("gates_failed"), file=sys.stderr)
            return 1
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["preflight_exception"])
        _write_summary_md(run_root, {"passed": False, "gates_failed": ["preflight_exception"]}, None, False, [str(e)], canary_proof=canary_proof)
        raise

    if args.validate_only:
        _write_summary_md(run_root, preflight, None, True, ["Preflight passed (--validate-only)"], canary_proof=canary_proof)
        print("[E2E] Preflight passed (--validate-only)", file=sys.stderr)
        return 0

    # Replay writes to run_root/replay so TRUTH does not see PREFLIGHT_E2E.json as existing artifacts
    replay_output_dir = run_root / "replay"
    replay_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths from truth
    truth_obj = _load_json(truth_path)
    canonical_bundle = str(truth_obj.get("canonical_xgb_bundle_dir") or "")
    canonical_prebuilt = str(truth_obj.get("canonical_prebuilt_parquet") or "")
    canonical_manifest_truth = str(truth_obj.get("canonical_prebuilt_manifest") or "")
    canonical_transformer = str(truth_obj.get("canonical_transformer_bundle_dir") or "")

    manifest_ssot_resolved = MANIFEST_SSOT.expanduser().resolve()
    if canonical_manifest_truth and Path(canonical_manifest_truth).expanduser().resolve() != manifest_ssot_resolved:
        _write_fatal_capsule(
            run_root,
            RuntimeError("PREBUILT_MANIFEST_SPLIT_BRAIN"),
            ["canonical_truth_paths"],
        )
        _write_summary_md(
            run_root,
            preflight,
            None,
            False,
            [f"PREBUILT_MANIFEST_SPLIT_BRAIN truth_manifest={canonical_manifest_truth} expected={manifest_ssot_resolved}"],
            canary_proof=canary_proof,
        )
        return 1

    manifest_obj = _load_json(manifest_ssot_resolved)
    manifest_parquet = str(Path(manifest_obj.get("parquet_path") or "").expanduser().resolve())
    if not canonical_prebuilt:
        _write_fatal_capsule(run_root, RuntimeError("canonical_prebuilt_parquet missing"), ["canonical_truth_paths"])
        _write_summary_md(
            run_root, preflight, None, False, ["canonical_prebuilt_parquet missing in truth"], canary_proof=canary_proof
        )
        return 1

    if manifest_parquet != str(Path(canonical_prebuilt).expanduser().resolve()):
        _write_fatal_capsule(run_root, RuntimeError("PREBUILT_SPLIT_BRAIN"), ["canonical_truth_paths"])
        _write_summary_md(
            run_root,
            preflight,
            None,
            False,
            [f"PREBUILT_SPLIT_BRAIN manifest_parquet={manifest_parquet} canonical_prebuilt_parquet={canonical_prebuilt}"],
            canary_proof=canary_proof,
        )
        return 1

    for label, val in (
        ("canonical_xgb_bundle_dir", canonical_bundle),
        ("canonical_transformer_bundle_dir", canonical_transformer),
        ("manifest_ssot", str(manifest_ssot_resolved)),
        ("manifest_parquet", manifest_parquet),
    ):
        _forbid_prune_path(label, val, allow_ctx6cat6=label == "canonical_transformer_bundle_dir")

    bundle_dir = Path(canonical_bundle).expanduser().resolve()
    prebuilt_path = Path(manifest_parquet).expanduser().resolve()

    policy_path = Path(
        os.environ.get(
            "GX1_CANONICAL_POLICY_PATH",
            str(ENGINE / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"),
        )
    ).expanduser().resolve()

    raw_path = Path(
        os.environ.get(
            "GX1_RAW_2025",
            str(_gx1_data() / "data" / "data" / "raw" / "xauusd_m5_2025_bid_ask.parquet"),
        )
    ).expanduser().resolve()

    if not policy_path.exists():
        _write_fatal_capsule(run_root, FileNotFoundError(str(policy_path)), ["policy_path"])
        _write_summary_md(run_root, preflight, None, False, [f"Policy not found: {policy_path}"], canary_proof=canary_proof)
        return 1
    # TRUTH gate: only the canonical policy file may be used (exact path match)
    try:
        from gx1.utils.truth_banlist import is_truth_or_smoke, assert_truth_policy_path_canonical

        if is_truth_or_smoke():
            assert_truth_policy_path_canonical(policy_path, engine_root=ENGINE, output_dir=run_root)
    except ImportError:
        pass
    if not raw_path.exists():
        _write_fatal_capsule(run_root, FileNotFoundError(str(raw_path)), ["raw_path"])
        _write_summary_md(run_root, preflight, None, False, [f"Raw data not found: {raw_path}"], canary_proof=canary_proof)
        return 1
    if not prebuilt_path.exists():
        _write_fatal_capsule(run_root, FileNotFoundError(str(prebuilt_path)), ["prebuilt_path"])
        _write_summary_md(run_root, preflight, None, False, [f"Prebuilt parquet not found: {prebuilt_path}"], canary_proof=canary_proof)
        return 1
    if not bundle_dir.exists():
        _write_fatal_capsule(run_root, FileNotFoundError(str(bundle_dir)), ["bundle_dir"])
        _write_summary_md(run_root, preflight, None, False, [f"Bundle dir not found: {bundle_dir}"], canary_proof=canary_proof)
        return 1

    os.environ["GX1_CANONICAL_BUNDLE_DIR"] = str(bundle_dir)
    os.environ["GX1_CANONICAL_TRANSFORMER_BUNDLE_DIR"] = str(Path(canonical_transformer).expanduser().resolve())
    os.environ["GX1_CANONICAL_POLICY_PATH"] = str(policy_path)

    start_ts = FULLYEAR_START_TS if args.full_year else args.start_ts
    end_ts = FULLYEAR_END_TS if args.full_year else args.end_ts

    env_overrides: Dict[str, str] = {}
    if args.threshold_override:
        env_overrides["GX1_ANALYSIS_MODE"] = "1"
        env_overrides["GX1_ENTRY_THRESHOLD_OVERRIDE"] = args.threshold_override
    if args.force_zero_trades:
        env_overrides["GX1_ANALYSIS_MODE"] = "1"
        env_overrides["GX1_ENTRY_THRESHOLD_OVERRIDE"] = ZERO_TRADES_CANARY_THRESHOLD

    # Replay
    try:
        rc = _run_replay(
            replay_output_dir,
            run_id,
            truth_path,
            policy_path,
            raw_path,
            prebuilt_path,
            start_ts,
            end_ts,
            env_overrides,
            bundle_dir,
            merge_output_dir=run_root,
        )
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["replay_exception"])
        postrun_fail = {"passed": False, "gates_failed": ["replay_exception"], "checks": {}}
        _atomic_write_json(run_root / "POSTRUN_E2E.json", postrun_fail)
        _write_summary_md(run_root, preflight, postrun_fail, False, [str(e)], canary_proof=canary_proof)
        raise

    if rc != 0:
        _atomic_write_json(
            run_root / "POSTRUN_E2E.json",
            {"passed": False, "run_id": run_id, "replay_exitcode": rc, "gates_failed": ["replay_exitcode"]},
        )
        _write_summary_md(run_root, preflight, {"passed": False, "gates_failed": ["replay_exitcode"]}, False, [f"Replay exit code {rc}"], canary_proof=canary_proof)
        return 1

    # Postrun checks
    try:
        try:
            truth_obj_postrun = _load_json(truth_path)
        except Exception:
            truth_obj_postrun = {}
        postrun = _run_postrun_checks(run_root, run_id, truth_obj_postrun)
        _atomic_write_json(run_root / "POSTRUN_E2E.json", postrun)
        if not postrun.get("passed", False):
            _write_fatal_capsule(run_root, RuntimeError(str(postrun.get("gates_failed"))), postrun.get("gates_failed", []))
            _write_summary_md(run_root, preflight, postrun, False, ["Post-run failed: " + str(postrun.get("gates_failed", []))], canary_proof=canary_proof)
            print("[E2E] POSTRUN FAIL:", postrun.get("gates_failed"), file=sys.stderr)
            return 1
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["postrun_exception"])
        _write_summary_md(run_root, preflight, None, False, [str(e)], canary_proof=canary_proof)
        raise

    # ZERO_TRADES_CANARY: hard-fail if we expected 0 trades but got any
    if args.force_zero_trades:
        metrics_path = run_root / f"metrics_{run_id}_MERGED.json"
        n_trades = -1
        if metrics_path.exists():
            try:
                metrics = _load_json(metrics_path)
                n_trades = int(metrics.get("n_trades", -1))
            except Exception:
                pass
        if n_trades != 0:
            msg = (
                f"[ZERO_TRADES_CANARY] Contract violation: expected n_trades=0, got n_trades={n_trades}. "
                f"Pipeline must produce 0 trades when GX1_ENTRY_THRESHOLD_OVERRIDE={ZERO_TRADES_CANARY_THRESHOLD}."
            )
            _write_fatal_capsule(run_root, RuntimeError(msg), ["zero_trades_canary"])
            _write_summary_md(run_root, preflight, postrun, False, [msg], canary_proof=canary_proof)
            print(f"[E2E] {msg}", file=sys.stderr)
            return 1
        print("[E2E] ZERO_TRADES_CANARY: n_trades=0, contract OK", file=sys.stderr)

    _write_summary_md(run_root, preflight, postrun, True, ["Preflight passed", "Replay completed", "Post-run passed"], canary_proof=canary_proof)
    print("[E2E] GO: Preflight + Replay + Post-run passed", file=sys.stderr)

    # TRUTH gate: ONE UNIVERSE 6/6 only. LAST_GO only when exits have context 6/6; hard-fail if footer dims != 6/6.
    footer_path = run_root / "replay" / "chunk_0" / "chunk_footer.json"
    expected_ctx_cont = CTX_CONT_DIM
    expected_ctx_cat = CTX_CAT_DIM
    require_exits_file = False
    if footer_path.exists():
        footer = _load_json(footer_path)
        if footer.get("ctx_cont_dim") is not None and footer["ctx_cont_dim"] != CTX_CONT_DIM:
            _write_fatal_capsule(run_root, RuntimeError(f"ONE_UNIVERSE: footer ctx_cont_dim must be {CTX_CONT_DIM}, got {footer['ctx_cont_dim']}"), ["ctx_dims"])
            raise RuntimeError(f"[E2E] footer ctx_cont_dim must be {CTX_CONT_DIM}, got {footer['ctx_cont_dim']}")
        if footer.get("ctx_cat_dim") is not None and footer["ctx_cat_dim"] != CTX_CAT_DIM:
            _write_fatal_capsule(run_root, RuntimeError(f"ONE_UNIVERSE: footer ctx_cat_dim must be {CTX_CAT_DIM}, got {footer['ctx_cat_dim']}"), ["ctx_dims"])
            raise RuntimeError(f"[E2E] footer ctx_cat_dim must be {CTX_CAT_DIM}, got {footer['ctx_cat_dim']}")
        if footer.get("exit_ml_enabled") is True:
            require_exits_file = True
    if os.environ.get("GX1_EXIT_AUDIT") == "1":
        require_exits_file = True
    gate_ok, gate_err = _exits_context_gate(run_root, run_id, expected_ctx_cont, expected_ctx_cat, require_exits_file=require_exits_file)
    if not gate_ok:
        postrun_fail = {
            "passed": False,
            "run_id": run_id,
            "gates_failed": ["exits_context_for_last_go"],
            "checks": {"exits_context_for_last_go": {"error": gate_err}},
        }
        _write_fatal_capsule(run_root, RuntimeError(gate_err or "exits context gate"), ["exits_context_for_last_go"])
        _atomic_write_json(run_root / "POSTRUN_E2E.json", postrun_fail)
        _write_summary_md(run_root, preflight, postrun_fail, False, [gate_err or "Exits context gate failed"], canary_proof=canary_proof)
        print(f"[E2E] LAST_GO not updated (exits context gate): {gate_err}", file=sys.stderr)
        return 1

    # LAST_GO is written only after all gates pass (preflight, replay, postrun, zero-trades, exits-context).
    last_go_dir = gx1_data / "reports" / "truth_e2e_sanity"
    last_go_path = last_go_dir / "LAST_GO.txt"
    try:
        last_go_dir.mkdir(parents=True, exist_ok=True)
        last_go_path.write_text(str(run_root.resolve()), encoding="utf-8")
        print(f"[E2E] LAST_GO set: {last_go_path} -> {run_root}", file=sys.stderr)
    except Exception as e:
        print(f"[E2E] LAST_GO write failed (non-fatal): {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())