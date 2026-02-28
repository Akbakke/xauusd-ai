#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRY_V10_CTX canonical baseline training entrypoint (ONE UNIVERSE only).

Non-negotiables:
- Baseline-only (no depth ladder / no L+1). Any other variant must hard-fail.
- CTX contract locked: CTX6CAT6 (6/6), signal_bridge_id=XGB_SIGNAL_BRIDGE_V1 (7-dim).
- Deterministic, fail-fast, no fallback paths.

Usage (canonical):
    python gx1/scripts/train_entry_v10_ctx_depth_ladder.py \\
        --variant baseline \\
        --data <train parquet or manifest> \\
        --feature-meta-path <json> \\
        --out-dir <bundle output dir> \\
        --epochs 10 --batch_size 256 --lr 3e-4 --seq_len 30 --seed 42
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
from typing import Any, Dict, Optional

import torch

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract

try:
    import gx1.models.entry_v10.entry_v10_ctx_train as train_mod
    from gx1.models.entry_v10.entry_v10_ctx_train import (
        main as train_main,
        set_seed,
        set_thread_limits,
    )
except Exception as e:
    # Fail fast with the real import error (no fallback)
    raise RuntimeError(
        f"[TRAIN_IMPORT_FAIL] Failed to import entry_v10_ctx_train: {type(e).__name__}: {e}"
    ) from e


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
    feature_meta_sha = hashlib.sha256(feature_meta_path.read_bytes()).hexdigest()

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
    meta["feature_meta_sha256"] = feature_meta_sha
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


def train_depth_ladder_variant(
    *,
    variant: str,
    data_path: Path,
    out_dir: Path,
    feature_meta_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    seq_len: int,
) -> Dict[str, Any]:
    """
    Train ENTRY_V10_CTX (baseline only) via canonical trainer.
    Uses dataset_dir derived from data_path (parquet -> parent dir; manifest -> manifest path).
    """
    _hard_gate_ctx6cat6()

    if variant != "baseline":
        raise ValueError("variant must be baseline (depth ladder disabled)")
    config = BASELINE_CONFIG.copy()

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
    log.info(f"CANONICAL TRAIN: {variant.upper()} (num_layers={config['num_layers']})")
    log.info(f"{'='*80}")
    log.info(f"Data: {data_path}")
    log.info(f"Out : {variant_out_dir}")
    log.info(f"Seed: {seed}  Device: {resolved_device}")
    log.info(f"CTX : tag={_CTX_TAG} cont={_CTX_CONT_DIM} cat={_CTX_CAT_DIM}")

    # We will call train_main() by overriding sys.argv (the backend script expects CLI args)
    original_argv = sys.argv.copy()
    original_env = dict(os.environ)

    try:
        # Derive dataset args for canonical trainer (no --data)
        dataset_manifest: Optional[Path] = None
        dataset_dir: Optional[Path] = None
        if str(data_path).endswith(".manifest.json"):
            dataset_manifest = data_path
            dataset_dir = data_path.parent
        elif str(data_path).endswith(".parquet"):
            dataset_manifest = None
            dataset_dir = data_path.parent
        else:
            raise RuntimeError(f"Unsupported data path (expect .parquet or .manifest.json): {data_path}")

        # Build args for backend trainer (canonical CLI)
        train_args = [
            "--train",
            "--dataset_dir", str(dataset_dir),
            "--out_bundle_dir", str(variant_out_dir),
        ]
        train_path_log = None
        val_path_log = None
        if data_path.suffix.lower() == ".parquet":
            train_args.extend(["--dataset_train_parquet", str(data_path.resolve())])
            if data_path.stem.endswith("_train"):
                val_path_log = data_path.with_name(data_path.stem[: -len("_train")] + "_val.parquet").resolve()
            train_path_log = data_path.resolve()
        train_args.extend([
            "--seq_len", str(seq_len),
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--seed", str(seed),
            "--device", str(resolved_device),
        ])
        if dataset_manifest is not None:
            train_args.extend(["--dataset_manifest", str(dataset_manifest)])

        sys.argv = ["gx1.models.entry_v10.entry_v10_ctx_train"] + train_args

        log.info(
            "[CANONICAL_TRAIN_ENTRYPOINT] data=%s out=%s train=%s val=%s",
            data_path,
            variant_out_dir,
            train_path_log or "infer-via-trainer",
            val_path_log or "infer-via-trainer",
        )
        log.info("[TRAIN] Calling ENTRY_V10_CTX trainer...")
        train_main()  # must perform training and write bundle artifacts

        # Post-run: required artifacts (baseline; lock optional for verification)
        model_path = variant_out_dir / "model_state_dict.pt"
        meta_path = variant_out_dir / "bundle_metadata.json"
        lock_path = variant_out_dir / "MASTER_TRANSFORMER_LOCK.json"
        missing = [str(p) for p in [model_path, meta_path] if not p.exists()]
        if missing:
            raise RuntimeError(f"TRAIN_OUTPUT_MISSING: expected bundle files missing: {missing}")

        # Ensure bundle is self-sufficient (copy meta/scalers and update meta/lock)
        finalize_bundle_with_meta_and_scalers(
            bundle_dir=variant_out_dir,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=None,
            snap_scaler_path=None,
        )

        # Baseline verification: contract/dims only (no num_layers requirement)
        meta_json = _read_json_if_exists(meta_path) or {}
        if not meta_json:
            raise RuntimeError("BASELINE_VERIFY_FAILED: bundle_metadata.json empty or unreadable")
        if meta_json.get("signal_bridge_id") != "XGB_SIGNAL_BRIDGE_V1":
            raise RuntimeError(
                f"BASELINE_VERIFY_FAILED: signal_bridge_id mismatch: {meta_json.get('signal_bridge_id')}"
            )
        if meta_json.get("ctx_tag") != "CTX6CAT6":
            raise RuntimeError(f"BASELINE_VERIFY_FAILED: ctx_tag mismatch: {meta_json.get('ctx_tag')}")
        if int(meta_json.get("ctx_cont_dim") or -1) != 6:
            raise RuntimeError(f"BASELINE_VERIFY_FAILED: ctx_cont_dim mismatch: {meta_json.get('ctx_cont_dim')}")
        if int(meta_json.get("ctx_cat_dim") or -1) != 6:
            raise RuntimeError(f"BASELINE_VERIFY_FAILED: ctx_cat_dim mismatch: {meta_json.get('ctx_cat_dim')}")

        bundle_sha = compute_bundle_sha256(variant_out_dir)
        log.info(f"[OK] bundle_sha={bundle_sha}")

        return {
            "variant": variant,
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
    parser = argparse.ArgumentParser(description="Train ENTRY_V10_CTX baseline (canonical entrypoint)")
    parser.add_argument("--variant", type=str, required=True, choices=["baseline"])
    parser.add_argument("--data", type=Path, required=True, help="Path to training dataset (.parquet or .manifest.json)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--feature-meta-path", type=Path, required=True, help="Canonical feature_meta.json to copy into bundle")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seq_len", type=int, default=30)

    args = parser.parse_args()

    res = train_depth_ladder_variant(
        variant=args.variant,
        data_path=args.data,
        out_dir=args.out_dir,
        feature_meta_path=args.feature_meta_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        seq_len=args.seq_len,
    )

    log.info("\n✅ DEPTH_LADDER DONE")
    log.info(f"   variant={res['variant']}")
    log.info(f"   out_dir={res['out_dir']}")
    log.info(f"   bundle_sha={res['bundle_sha']}")


if __name__ == "__main__":
    main()