#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRY_V10_CTX Depth Ladder Training Script (baseline vs L+1)

Goal:
- Train ENTRY_V10_CTX with a depth-ladder variant where ONLY transformer depth changes.

Non-negotiables:
- CTX contract is ONE UNIVERSE: CTX6CAT6 (6/6)
- DO NOT modify trading logic; ONLY architecture depth (num_layers)
- Fail-fast if the training backend does not actually apply the requested depth
  (prevents silent baseline training).

Usage:
    python gx1/scripts/train_entry_v10_ctx_depth_ladder.py \
        --variant baseline|lplus1 \
        --data <parquet> \
        --out-dir checkpoints/entry_v10_ctx_depth_ladder/ \
        --feature-meta-path <json> \
        --seq-scaler-path <path optional> \
        --snap-scaler-path <path optional> \
        --seed 42 \
        --epochs 10 \
        --device auto
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract

try:
    from gx1.models.entry_v10.entry_v10_ctx_train import (
        main as train_main,
        set_seed,
        set_thread_limits,
    )
except ImportError:
    def _missing_train(*_args, **_kwargs):
        raise RuntimeError(
            "ENTRY_V10_CTX train module is missing; depth ladder wrapper cannot run training."
        )
    train_main = _missing_train

    def set_seed(*_args, **_kwargs):
        raise RuntimeError("ENTRY_V10_CTX train module is missing; set_seed unavailable.")

    def set_thread_limits(*_args, **_kwargs):
        raise RuntimeError("ENTRY_V10_CTX train module is missing; set_thread_limits unavailable.")


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ============================================================================
# BASELINE ARCHITECTURE (LOCKED EXCEPT DEPTH)
# ============================================================================
_CTX = get_canonical_ctx_contract()
_CTX_CAT_DIM = int(_CTX["ctx_cat_dim"])
_CTX_CONT_DIM = int(_CTX["ctx_cont_dim"])
_CTX_TAG = str(_CTX["tag"])


def _hard_gate_ctx6cat6() -> None:
    if _CTX_TAG != "CTX6CAT6" or _CTX_CAT_DIM != 6 or _CTX_CONT_DIM != 6:
        raise RuntimeError(
            f"CTX_CONTRACT_SPLIT_BRAIN: expected CTX6CAT6 (6/6) but got tag={_CTX_TAG} "
            f"ctx_cat_dim={_CTX_CAT_DIM} ctx_cont_dim={_CTX_CONT_DIM}"
        )


BASELINE_CONFIG: Dict[str, Any] = {
    "variant": "v10_ctx",
    "num_layers": 3,
    "d_model": 128,
    "n_heads": 4,
    "dim_feedforward": None,  # backend default (commonly d_model*4)
    "dropout": 0.05,
    "seq_len": 30,
    "ctx_cat_dim": _CTX_CAT_DIM,
    "ctx_cont_dim": _CTX_CONT_DIM,
    "ctx_tag": _CTX_TAG,
    "ctx_emb_dim": 42,
    "ctx_embedding_dim": 8,
}

LPLUS1_CONFIG: Dict[str, Any] = {
    **BASELINE_CONFIG,
    "num_layers": 4,  # baseline + 1
    "depth_ladder_delta": +1,
}


def validate_config_diff(baseline_cfg: Dict[str, Any], variant_cfg: Dict[str, Any], allowed_diff: Set[str]) -> None:
    """
    FATAL if config differs beyond allowed parameters.
    This ensures we only change depth.
    """
    all_keys = set(baseline_cfg.keys()) | set(variant_cfg.keys())
    diff_keys = set()

    for key in all_keys:
        if baseline_cfg.get(key) != variant_cfg.get(key):
            if key not in allowed_diff:
                diff_keys.add(key)

    if diff_keys:
        raise RuntimeError(
            "DEPTH_LADDER_ILLEGAL_DIFF: Config differs beyond allowed parameters. "
            f"diff_keys={sorted(diff_keys)} allowed_diff={sorted(allowed_diff)}"
        )


def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_bundle_sha256(bundle_dir: Path) -> str:
    """
    Deterministic short SHA over key bundle files (best-effort).
    """
    files_to_hash = [
        "model_state_dict.pt",
        "bundle_metadata.json",
        "MASTER_TRANSFORMER_LOCK.json",
        "feature_contract_hash.txt",
    ]
    hasher = hashlib.sha256()
    for filename in files_to_hash:
        fp = bundle_dir / filename
        if fp.exists():
            hasher.update(fp.read_bytes())
    return hasher.hexdigest()[:16]


def _copy_into_bundle(bundle_dir: Path, src: Path, dest_name: str, fatal_label: str) -> str:
    if not src.exists():
        raise RuntimeError(f"{fatal_label}: {src}")
    dest = bundle_dir / dest_name
    shutil.copy2(src, dest)
    return dest.name


def finalize_bundle_with_meta_and_scalers(
    bundle_dir: Path,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
) -> None:
    """
    Make bundle self-sufficient: copy feature_meta/scalers and update bundle_metadata.json and lock.
    Hard-fails if required bundle components are missing.
    """
    bundle_dir = bundle_dir.resolve()
    meta_json = bundle_dir / "bundle_metadata.json"
    model_path = bundle_dir / "model_state_dict.pt"
    lock_path = bundle_dir / "MASTER_TRANSFORMER_LOCK.json"

    if not model_path.exists():
        raise RuntimeError(f"BUNDLE_PACKAGING_MISSING_MODEL: {model_path}")
    if not meta_json.exists():
        raise RuntimeError(f"BUNDLE_PACKAGING_MISSING_METADATA: {meta_json}")
    if not lock_path.exists():
        raise RuntimeError(f"BUNDLE_PACKAGING_MISSING_LOCK: {lock_path}")

    # Copy artifacts
    feature_meta_rel = _copy_into_bundle(
        bundle_dir,
        feature_meta_path,
        "entry_v10_ctx_feature_meta.json",
        "BUNDLE_PACKAGING_MISSING_FEATURE_META",
    )

    seq_scaler_rel = None
    snap_scaler_rel = None
    if seq_scaler_path is not None:
        seq_scaler_rel = _copy_into_bundle(
            bundle_dir,
            seq_scaler_path,
            seq_scaler_path.name,
            "BUNDLE_PACKAGING_MISSING_SCALER",
        )
    if snap_scaler_path is not None:
        snap_scaler_rel = _copy_into_bundle(
            bundle_dir,
            snap_scaler_path,
            snap_scaler_path.name,
            "BUNDLE_PACKAGING_MISSING_SCALER",
        )

    # Update metadata (relative paths)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    meta["feature_meta_path"] = feature_meta_rel
    if seq_scaler_rel:
        meta["seq_scaler_path"] = seq_scaler_rel
    if snap_scaler_rel:
        meta["snap_scaler_path"] = snap_scaler_rel
    meta["git_commit"] = get_git_commit()
    meta["depth_ladder_wrapper"] = True
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Update lock similarly (fail if malformed)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["feature_meta_path"] = feature_meta_rel
    if seq_scaler_rel:
        lock["seq_scaler_path"] = seq_scaler_rel
    if snap_scaler_rel:
        lock["snap_scaler_path"] = snap_scaler_rel
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")


def _read_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_num_layers_from_bundle(bundle_dir: Path) -> Optional[int]:
    """
    Best-effort extraction of num_layers from bundle_metadata.json or lock.
    If training backend doesn't write it, we can't verify -> hard fail upstream.
    """
    meta = _read_json_if_exists(bundle_dir / "bundle_metadata.json") or {}
    lock = _read_json_if_exists(bundle_dir / "MASTER_TRANSFORMER_LOCK.json") or {}

    # common places
    candidates = [
        meta.get("num_layers"),
        meta.get("model", {}).get("num_layers") if isinstance(meta.get("model"), dict) else None,
        meta.get("seq_cfg", {}).get("num_layers") if isinstance(meta.get("seq_cfg"), dict) else None,
        lock.get("num_layers"),
        lock.get("model", {}).get("num_layers") if isinstance(lock.get("model"), dict) else None,
        lock.get("seq_cfg", {}).get("num_layers") if isinstance(lock.get("seq_cfg"), dict) else None,
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            return int(c)
        except Exception:
            continue
    return None


def train_depth_ladder_variant(
    *,
    variant: str,
    data_path: Path,
    out_dir: Path,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    seq_len: int,
) -> Dict[str, Any]:
    """
    Train a depth ladder variant by calling the existing ENTRY_V10_CTX training entrypoint,
    while requesting a depth override through environment variables.

    After training:
    - bundle must exist (model_state_dict.pt + bundle_metadata.json + MASTER_TRANSFORMER_LOCK.json)
    - bundle metadata/lock must reflect the requested num_layers (otherwise hard-fail)
    """
    _hard_gate_ctx6cat6()

    if variant == "baseline":
        config = BASELINE_CONFIG.copy()
    elif variant == "lplus1":
        config = LPLUS1_CONFIG.copy()
        validate_config_diff(BASELINE_CONFIG, config, allowed_diff={"num_layers", "depth_ladder_delta"})
    else:
        raise ValueError("variant must be baseline|lplus1")

    variant_out_dir = (out_dir / variant.upper()).resolve()
    variant_out_dir.mkdir(parents=True, exist_ok=True)

    # determinism
    set_seed(seed)
    set_thread_limits()

    # Device handling: pass through to backend; keep wrapper deterministic
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device

    log.info(f"\n{'='*80}")
    log.info(f"DEPTH_LADDER TRAIN: {variant.upper()} (num_layers={config['num_layers']})")
    log.info(f"{'='*80}")
    log.info(f"Data: {data_path}")
    log.info(f"Out : {variant_out_dir}")
    log.info(f"Seed: {seed}  Device: {resolved_device}")
    log.info(f"CTX : tag={_CTX_TAG} cont={_CTX_CONT_DIM} cat={_CTX_CAT_DIM}")

    # We will call train_main() by overriding sys.argv (the backend script expects CLI args)
    original_argv = sys.argv.copy()
    original_env = dict(os.environ)

    try:
        # Depth ladder request (backend MUST honor these; we verify after)
        os.environ["GX1_DEPTH_LADDER_MODE"] = "1"
        os.environ["GX1_DEPTH_LADDER_VARIANT"] = variant
        os.environ["GX1_DEPTH_LADDER_NUM_LAYERS"] = str(config["num_layers"])

        # Build args for backend trainer
        train_args = [
            "--data", str(data_path),
            "--out_dir", str(variant_out_dir),
            "--feature_meta_path", str(feature_meta_path),
            "--seq_len", str(seq_len),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--learning_rate", str(learning_rate),
            "--seed", str(seed),
            "--device", str(resolved_device),
        ]
        if seq_scaler_path is not None:
            train_args.extend(["--seq_scaler_path", str(seq_scaler_path)])
        if snap_scaler_path is not None:
            train_args.extend(["--snap_scaler_path", str(snap_scaler_path)])

        sys.argv = ["-m", "gx1.models.entry_v10.entry_v10_ctx_train"] + train_args

        log.info("[TRAIN] Calling ENTRY_V10_CTX trainer...")
        train_main()  # must perform training and write bundle artifacts

        # Post-run: required artifacts
        model_path = variant_out_dir / "model_state_dict.pt"
        meta_path = variant_out_dir / "bundle_metadata.json"
        lock_path = variant_out_dir / "MASTER_TRANSFORMER_LOCK.json"
        missing = [str(p) for p in [model_path, meta_path, lock_path] if not p.exists()]
        if missing:
            raise RuntimeError(f"TRAIN_OUTPUT_MISSING: expected bundle files missing: {missing}")

        # Ensure bundle is self-sufficient (copy meta/scalers and update meta/lock)
        finalize_bundle_with_meta_and_scalers(
            bundle_dir=variant_out_dir,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
        )

        # Verify depth override was actually applied (prevent silent baseline)
        got_layers = _extract_num_layers_from_bundle(variant_out_dir)
        if got_layers is None:
            raise RuntimeError(
                "DEPTH_LADDER_VERIFY_FAILED: Could not find num_layers in bundle metadata/lock. "
                "Backend trainer must write num_layers (or seq_cfg.num_layers) into bundle_metadata.json "
                "or MASTER_TRANSFORMER_LOCK.json, otherwise this wrapper cannot verify correctness."
            )
        if int(got_layers) != int(config["num_layers"]):
            raise RuntimeError(
                "DEPTH_LADDER_VERIFY_FAILED: Backend did not apply requested depth. "
                f"requested_num_layers={config['num_layers']} got_num_layers={got_layers}. "
                "This would be silent corruption; failing hard."
            )

        bundle_sha = compute_bundle_sha256(variant_out_dir)
        log.info(f"[OK] bundle_sha={bundle_sha} num_layers={got_layers}")

        return {
            "variant": variant,
            "requested_num_layers": int(config["num_layers"]),
            "verified_num_layers": int(got_layers),
            "out_dir": str(variant_out_dir),
            "bundle_sha": bundle_sha,
            "device": resolved_device,
        }

    finally:
        # Restore argv and env to avoid pollution
        sys.argv = original_argv
        os.environ.clear()
        os.environ.update(original_env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ENTRY_V10_CTX Depth Ladder variants (baseline vs L+1)")
    parser.add_argument("--variant", type=str, required=True, choices=["baseline", "lplus1"])
    parser.add_argument("--data", type=Path, required=True, help="Path to training dataset (parquet)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--feature-meta-path", type=Path, required=True, help="Path to feature_meta.json")
    parser.add_argument("--seq-scaler-path", type=Path, default=None, help="Path to seq scaler (optional)")
    parser.add_argument("--snap-scaler-path", type=Path, default=None, help="Path to snap scaler (optional)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seq-len", type=int, default=30)

    args = parser.parse_args()

    res = train_depth_ladder_variant(
        variant=args.variant,
        data_path=args.data,
        out_dir=args.out_dir,
        feature_meta_path=args.feature_meta_path,
        seq_scaler_path=args.seq_scaler_path,
        snap_scaler_path=args.snap_scaler_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        seq_len=args.seq_len,
    )

    log.info("\n✅ DEPTH_LADDER DONE")
    log.info(f"   variant={res['variant']} requested_layers={res['requested_num_layers']} verified_layers={res['verified_num_layers']}")
    log.info(f"   out_dir={res['out_dir']}")
    log.info(f"   bundle_sha={res['bundle_sha']}")


if __name__ == "__main__":
    main()