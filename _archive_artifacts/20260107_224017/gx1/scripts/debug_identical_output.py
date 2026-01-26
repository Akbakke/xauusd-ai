#!/usr/bin/env python3
"""
Debug script to find root cause of identical outputs between gated fusion and baseline.

Runs comprehensive investigation:
- Policy/config tracing
- Artifact verification (paths + hashes)
- Input tensor proof
- Intervention tests (perturbations)
- Decision stack dominance check
- Final verdict

Usage:
    python -m gx1.scripts.debug_identical_output \
        --gated_checkpoint models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION \
        --baseline_checkpoint models/entry_v10_ctx/FULLYEAR_2025_BASELINE_NO_GATE \
        --val_data data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_val.parquet \
        --policy_config gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml \
        --output_dir reports/debug
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from gx1.models.entry_v10.entry_v10_ctx_dataset import EntryV10CtxDataset
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def get_git_info() -> Dict[str, str]:
    """Get git commit and dirty state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        commit = "unknown"
    
    try:
        dirty = subprocess.check_output(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        )
        is_dirty = "clean"
    except:
        is_dirty = "dirty"
    
    return {"commit": commit, "dirty": is_dirty}


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_model(checkpoint_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load model and return with metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load metadata
    metadata_path = checkpoint_dir / "bundle_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Load model state
    state_dict_path = checkpoint_dir / "model_state_dict.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Model state not found: {state_dict_path}")
    
    state_dict = torch.load(state_dict_path, map_location=device)
    
    # Infer dimensions from state dict
    if "seq_encoder.input_proj.weight" in state_dict:
        actual_seq_dim = state_dict["seq_encoder.input_proj.weight"].shape[1]
    else:
        actual_seq_dim = metadata.get("seq_input_dim", 13) + 3
    
    if "snap_encoder.mlp.0.weight" in state_dict:
        actual_snap_dim = state_dict["snap_encoder.mlp.0.weight"].shape[1]
    else:
        actual_snap_dim = metadata.get("snap_input_dim", 85) + 3
    
    # Create model
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=actual_seq_dim,
        snap_input_dim=actual_snap_dim,
        max_seq_len=metadata.get("max_seq_len", 30),
        variant="v10_ctx",
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
        ctx_embedding_dim=8,
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    metadata["seq_input_dim"] = actual_seq_dim
    metadata["snap_input_dim"] = actual_snap_dim
    
    return model, metadata


def trace_artifacts(
    policy_config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
) -> Dict:
    """Trace all artifacts loaded by policy and compute hashes."""
    log.info("[DEBUG_IDENTITY] Tracing artifacts...")
    
    with open(policy_config_path) as f:
        policy = yaml.safe_load(f)
    
    artifacts = {}
    
    # Model artifacts
    entry_models = policy.get("entry_models", {})
    
    # V10_CTX model
    if entry_models.get("v10_ctx", {}).get("enabled", False):
        bundle_dir = entry_models["v10_ctx"].get("bundle_dir", "")
        if bundle_dir:
            bundle_path = Path(bundle_dir).resolve()
            if bundle_path.exists():
                model_path = bundle_path / "model_state_dict.pt"
                metadata_path = bundle_path / "bundle_metadata.json"
                
                if model_path.exists():
                    artifacts["v10_ctx_model"] = {
                        "path": str(model_path.resolve()),
                        "hash": compute_file_hash(model_path),
                        "size_bytes": model_path.stat().st_size,
                    }
                
                if metadata_path.exists():
                    artifacts["v10_ctx_metadata"] = {
                        "path": str(metadata_path.resolve()),
                        "hash": compute_file_hash(metadata_path),
                        "size_bytes": metadata_path.stat().st_size,
                    }
    
    # XGB models
    xgb_config = entry_models.get("xgb", {})
    for session in ["EU", "US", "OVERLAP", "ASIA"]:
        key = f"{session.lower()}_model_path"
        if key in xgb_config:
            xgb_path = Path(xgb_config[key]).resolve()
            if xgb_path.exists():
                artifacts[f"xgb_{session.lower()}"] = {
                    "path": str(xgb_path.resolve()),
                    "hash": compute_file_hash(xgb_path),
                    "size_bytes": xgb_path.stat().st_size,
                }
    
    # Feature meta
    feature_meta_path = policy.get("feature_meta_path", "")
    if feature_meta_path:
        meta_path = Path(feature_meta_path).resolve()
        if meta_path.exists():
            artifacts["feature_meta"] = {
                "path": str(meta_path.resolve()),
                "hash": compute_file_hash(meta_path),
                "size_bytes": meta_path.stat().st_size,
            }
    
    # Scalers
    seq_scaler_path = policy.get("seq_scaler_path", "")
    if seq_scaler_path:
        scaler_path = Path(seq_scaler_path).resolve()
        if scaler_path.exists():
            artifacts["seq_scaler"] = {
                "path": str(scaler_path.resolve()),
                "hash": compute_file_hash(scaler_path),
                "size_bytes": scaler_path.stat().st_size,
            }
    
    snap_scaler_path = policy.get("snap_scaler_path", "")
    if snap_scaler_path:
        scaler_path = Path(snap_scaler_path).resolve()
        if scaler_path.exists():
            artifacts["snap_scaler"] = {
                "path": str(scaler_path.resolve()),
                "hash": compute_file_hash(scaler_path),
                "size_bytes": scaler_path.stat().st_size,
            }
    
    # Calibration directory
    calibration_dir = policy.get("calibration_dir", "")
    if calibration_dir:
        cal_dir = Path(calibration_dir).resolve()
        if cal_dir.exists():
            # Find all calibrator files
            for cal_file in cal_dir.rglob("*.joblib"):
                rel_path = cal_file.relative_to(cal_dir)
                artifacts[f"calibrator_{rel_path}"] = {
                    "path": str(cal_file.resolve()),
                    "hash": compute_file_hash(cal_file),
                    "size_bytes": cal_file.stat().st_size,
                }
    
    # Checkpoint artifacts
    checkpoint_path = checkpoint_dir / "model_state_dict.pt"
    if checkpoint_path.exists():
        artifacts["checkpoint_model"] = {
            "path": str(checkpoint_path.resolve()),
            "hash": compute_file_hash(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        }
    
    # Save artifacts JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_json_path = output_dir / f"IDENTICAL_OUTPUT_ARTIFACTS_{timestamp}.json"
    with open(artifacts_json_path, "w") as f:
        json.dump(artifacts, f, indent=2)
    
    log.info(f"[DEBUG_IDENTITY] ✅ Artifacts traced: {len(artifacts)} files")
    log.info(f"[DEBUG_IDENTITY] ✅ Saved: {artifacts_json_path}")
    
    return artifacts


def capture_input_tensors(
    dataset: EntryV10CtxDataset,
    device: torch.device,
    output_dir: Path,
    n_samples: int = 64,
) -> Dict:
    """Capture input tensors for analysis."""
    log.info(f"[DEBUG_IDENTITY] Capturing {n_samples} input tensors...")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_samples,
        shuffle=False,
        num_workers=0,
    )
    
    batch = next(iter(dataloader))
    
    seq_x = batch["seq_x"].to(device)
    snap_x = batch["snap_x"].to(device)
    ctx_cat = batch["ctx_cat"].to(device)
    ctx_cont = batch["ctx_cont"].to(device)
    
    # Extract XGB channels
    xgb_seq = seq_x[:, :, 13:16].cpu().numpy()  # p_cal, margin, uncertainty_score
    xgb_snap = snap_x[:, 85:88].cpu().numpy()  # p_cal, margin, p_hat
    
    # Statistics
    stats = {
        "seq_x": {
            "shape": list(seq_x.shape),
            "dtype": str(seq_x.dtype),
            "nan_count": int(torch.isnan(seq_x).sum().item()),
            "inf_count": int(torch.isinf(seq_x).sum().item()),
            "min": float(seq_x.min().item()),
            "max": float(seq_x.max().item()),
            "mean": float(seq_x.mean().item()),
        },
        "snap_x": {
            "shape": list(snap_x.shape),
            "dtype": str(snap_x.dtype),
            "nan_count": int(torch.isnan(snap_x).sum().item()),
            "inf_count": int(torch.isinf(snap_x).sum().item()),
            "min": float(snap_x.min().item()),
            "max": float(snap_x.max().item()),
            "mean": float(snap_x.mean().item()),
        },
        "xgb_seq_channels": {
            "p_cal": {
                "min": float(xgb_seq[:, :, 0].min()),
                "max": float(xgb_seq[:, :, 0].max()),
                "mean": float(xgb_seq[:, :, 0].mean()),
                "std": float(xgb_seq[:, :, 0].std()),
                "is_constant": bool(np.allclose(xgb_seq[:, :, 0], xgb_seq[0, 0, 0])),
            },
            "margin": {
                "min": float(xgb_seq[:, :, 1].min()),
                "max": float(xgb_seq[:, :, 1].max()),
                "mean": float(xgb_seq[:, :, 1].mean()),
                "std": float(xgb_seq[:, :, 1].std()),
                "is_constant": bool(np.allclose(xgb_seq[:, :, 1], xgb_seq[0, 0, 1])),
            },
            "uncertainty_score": {
                "min": float(xgb_seq[:, :, 2].min()),
                "max": float(xgb_seq[:, :, 2].max()),
                "mean": float(xgb_seq[:, :, 2].mean()),
                "std": float(xgb_seq[:, :, 2].std()),
                "is_constant": bool(np.allclose(xgb_seq[:, :, 2], xgb_seq[0, 0, 2])),
            },
        },
        "xgb_snap_channels": {
            "p_cal": {
                "min": float(xgb_snap[:, 0].min()),
                "max": float(xgb_snap[:, 0].max()),
                "mean": float(xgb_snap[:, 0].mean()),
                "std": float(xgb_snap[:, 0].std()),
                "is_constant": bool(np.allclose(xgb_snap[:, 0], xgb_snap[0, 0])),
            },
            "margin": {
                "min": float(xgb_snap[:, 1].min()),
                "max": float(xgb_snap[:, 1].max()),
                "mean": float(xgb_snap[:, 1].mean()),
                "std": float(xgb_snap[:, 1].std()),
                "is_constant": bool(np.allclose(xgb_snap[:, 1], xgb_snap[0, 1])),
            },
            "p_hat": {
                "min": float(xgb_snap[:, 2].min()),
                "max": float(xgb_snap[:, 2].max()),
                "mean": float(xgb_snap[:, 2].mean()),
                "std": float(xgb_snap[:, 2].std()),
                "is_constant": bool(np.allclose(xgb_snap[:, 2], xgb_snap[0, 2])),
            },
        },
        "ctx_cat": {
            "shape": list(ctx_cat.shape),
            "dtype": str(ctx_cat.dtype),
            "nan_count": int(torch.isnan(ctx_cat.float()).sum().item()),
            "min": int(ctx_cat.min().item()),
            "max": int(ctx_cat.max().item()),
            "unique_values": [int(v) for v in torch.unique(ctx_cat).cpu().numpy()],
        },
        "ctx_cont": {
            "shape": list(ctx_cont.shape),
            "dtype": str(ctx_cont.dtype),
            "nan_count": int(torch.isnan(ctx_cont).sum().item()),
            "inf_count": int(torch.isinf(ctx_cont).sum().item()),
            "min": float(ctx_cont.min().item()),
            "max": float(ctx_cont.max().item()),
            "mean": float(ctx_cont.mean().item()),
            "is_constant": bool(torch.allclose(ctx_cont, ctx_cont[0])),
        },
    }
    
    # Save NPZ dump
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = output_dir / f"IDENTICAL_OUTPUT_BATCH_{timestamp}.npz"
    np.savez_compressed(
        npz_path,
        seq_x=seq_x.cpu().numpy(),
        snap_x=snap_x.cpu().numpy(),
        ctx_cat=ctx_cat.cpu().numpy(),
        ctx_cont=ctx_cont.cpu().numpy(),
        xgb_seq=xgb_seq,
        xgb_snap=xgb_snap,
    )
    
    log.info(f"[DEBUG_IDENTITY] ✅ Input tensors captured: {npz_path}")
    
    # HARD FAIL checks
    if stats["xgb_seq_channels"]["p_cal"]["is_constant"]:
        raise RuntimeError("HARD_FAIL: XGB seq p_cal channel is constant!")
    if stats["xgb_snap_channels"]["p_cal"]["is_constant"]:
        raise RuntimeError("HARD_FAIL: XGB snap p_cal channel is constant!")
    if stats["ctx_cont"]["is_constant"]:
        raise RuntimeError("HARD_FAIL: ctx_cont is constant!")
    
    return stats, batch


def run_intervention_tests(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """Run intervention tests with perturbations."""
    log.info("[DEBUG_IDENTITY] Running intervention tests...")
    
    model.eval()
    results = {}
    
    # Baseline
    with torch.no_grad():
        seq_x = batch["seq_x"].to(device)
        snap_x = batch["snap_x"].to(device)
        ctx_cat = batch["ctx_cat"].to(device)
        ctx_cont = batch["ctx_cont"].to(device)
        session_id = batch["session_id"].to(device)
        vol_regime_id = batch["vol_regime_id"].to(device)
        trend_regime_id = batch["trend_regime_id"].to(device)
        
        baseline_output = model(
            seq_x=seq_x,
            snap_x=snap_x,
            session_id=session_id,
            vol_regime_id=vol_regime_id,
            trend_regime_id=trend_regime_id,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )
        baseline_logit = baseline_output["direction_logit"].cpu().numpy()
        baseline_gate = baseline_output.get("gate", torch.zeros(len(seq_x), 1)).cpu().numpy()
    
    results["baseline"] = {
        "logit_mean": float(baseline_logit.mean()),
        "logit_std": float(baseline_logit.std()),
        "logit_min": float(baseline_logit.min()),
        "logit_max": float(baseline_logit.max()),
        "gate_mean": float(baseline_gate.mean()) if baseline_gate is not None else 0.0,
    }
    
    # Test 1: Zero XGB channels
    seq_x_zero = seq_x.clone()
    snap_x_zero = snap_x.clone()
    seq_x_zero[:, :, 13:16] = 0.0
    snap_x_zero[:, 85:88] = 0.0
    
    with torch.no_grad():
        output_zero = model(
            seq_x=seq_x_zero,
            snap_x=snap_x_zero,
            session_id=session_id,
            vol_regime_id=vol_regime_id,
            trend_regime_id=trend_regime_id,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )
        logit_zero = output_zero["direction_logit"].cpu().numpy()
    
    delta_zero = np.abs(logit_zero - baseline_logit)
    results["zero_xgb"] = {
        "max_abs_delta": float(delta_zero.max()),
        "mean_abs_delta": float(delta_zero.mean()),
        "p95_abs_delta": float(np.percentile(delta_zero, 95)),
        "decision_flips": int((np.sign(logit_zero) != np.sign(baseline_logit)).sum()),
    }
    
    # Test 2: Randomize XGB channels
    seq_x_rand = seq_x.clone()
    snap_x_rand = snap_x.clone()
    perm = np.random.permutation(len(seq_x))
    seq_x_rand[:, :, 13:16] = seq_x[perm][:, :, 13:16]
    snap_x_rand[:, 85:88] = snap_x[perm][:, 85:88]
    
    with torch.no_grad():
        output_rand = model(
            seq_x=seq_x_rand,
            snap_x=snap_x_rand,
            session_id=session_id,
            vol_regime_id=vol_regime_id,
            trend_regime_id=trend_regime_id,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )
        logit_rand = output_rand["direction_logit"].cpu().numpy()
    
    delta_rand = np.abs(logit_rand - baseline_logit)
    results["randomize_xgb"] = {
        "max_abs_delta": float(delta_rand.max()),
        "mean_abs_delta": float(delta_rand.mean()),
        "p95_abs_delta": float(np.percentile(delta_rand, 95)),
        "decision_flips": int((np.sign(logit_rand) != np.sign(baseline_logit)).sum()),
    }
    
    # Test 3: Zero ctx
    ctx_cat_zero = torch.zeros_like(ctx_cat)
    ctx_cont_zero = torch.zeros_like(ctx_cont)
    
    with torch.no_grad():
        output_ctx_zero = model(
            seq_x=seq_x,
            snap_x=snap_x,
            session_id=session_id,
            vol_regime_id=vol_regime_id,
            trend_regime_id=trend_regime_id,
            ctx_cat=ctx_cat_zero,
            ctx_cont=ctx_cont_zero,
        )
        logit_ctx_zero = output_ctx_zero["direction_logit"].cpu().numpy()
    
    delta_ctx = np.abs(logit_ctx_zero - baseline_logit)
    results["zero_ctx"] = {
        "max_abs_delta": float(delta_ctx.max()),
        "mean_abs_delta": float(delta_ctx.mean()),
        "p95_abs_delta": float(np.percentile(delta_ctx, 95)),
        "decision_flips": int((np.sign(logit_ctx_zero) != np.sign(baseline_logit)).sum()),
    }
    
    # Test 4: Randomize sequence order
    seq_x_perm = seq_x[:, torch.randperm(seq_x.size(1)), :]
    
    with torch.no_grad():
        output_seq_perm = model(
            seq_x=seq_x_perm,
            snap_x=snap_x,
            session_id=session_id,
            vol_regime_id=vol_regime_id,
            trend_regime_id=trend_regime_id,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )
        logit_seq_perm = output_seq_perm["direction_logit"].cpu().numpy()
    
    delta_seq = np.abs(logit_seq_perm - baseline_logit)
    results["randomize_seq"] = {
        "max_abs_delta": float(delta_seq.max()),
        "mean_abs_delta": float(delta_seq.mean()),
        "p95_abs_delta": float(np.percentile(delta_seq, 95)),
        "decision_flips": int((np.sign(logit_seq_perm) != np.sign(baseline_logit)).sum()),
    }
    
    # Test 5: Force gate (if gated fusion enabled)
    # Note: This test requires patching the gated_fusion module directly
    # We'll skip it if the model doesn't have gated_fusion or if it's disabled
    gated_fusion_enabled = os.getenv("GX1_GATED_FUSION_ENABLED", "1") == "1"
    if gated_fusion_enabled and hasattr(model, "gated_fusion") and model.gated_fusion is not None:
        # Patch the gated_fusion.forward method to force gate values
        original_gated_forward = model.gated_fusion.forward
        
        def forced_gate_forward(xgb_state, seq_emb, ctx_emb, snap_raw_emb=None):
            # Call original to get fused and gate
            fused, gate = original_gated_forward(xgb_state, seq_emb, ctx_emb, snap_raw_emb)
            # Force gate to test value (we'll test 0 and 1 separately)
            force_value = float(os.getenv("GX1_DEBUG_FORCE_GATE", "0.5"))
            gate = torch.full_like(gate, force_value)
            # Recompute fused with forced gate
            # fused = gate * xgb_proj + (1 - gate) * transformer_proj
            xgb_proj = model.gated_fusion.xgb_proj(xgb_state)
            transformer_input = torch.cat([seq_emb, ctx_emb], dim=-1)
            transformer_proj = model.gated_fusion.transformer_proj(transformer_input)
            fused = gate * xgb_proj + (1 - gate) * transformer_proj
            return fused, gate
        
        # Test gate=0
        os.environ["GX1_DEBUG_FORCE_GATE"] = "0.0"
        model.gated_fusion.forward = forced_gate_forward
        
        with torch.no_grad():
            output_gate0 = model(
                seq_x=seq_x,
                snap_x=snap_x,
                session_id=session_id,
                vol_regime_id=vol_regime_id,
                trend_regime_id=trend_regime_id,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
            )
            logit_gate0 = output_gate0["direction_logit"].cpu().numpy()
        
        delta_gate0 = np.abs(logit_gate0 - baseline_logit)
        results["force_gate_0"] = {
            "max_abs_delta": float(delta_gate0.max()),
            "mean_abs_delta": float(delta_gate0.mean()),
            "p95_abs_delta": float(np.percentile(delta_gate0, 95)),
            "decision_flips": int((np.sign(logit_gate0) != np.sign(baseline_logit)).sum()),
        }
        
        # Test gate=1
        os.environ["GX1_DEBUG_FORCE_GATE"] = "1.0"
        with torch.no_grad():
            output_gate1 = model(
                seq_x=seq_x,
                snap_x=snap_x,
                session_id=session_id,
                vol_regime_id=vol_regime_id,
                trend_regime_id=trend_regime_id,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
            )
            logit_gate1 = output_gate1["direction_logit"].cpu().numpy()
        
        delta_gate1 = np.abs(logit_gate1 - baseline_logit)
        results["force_gate_1"] = {
            "max_abs_delta": float(delta_gate1.max()),
            "mean_abs_delta": float(delta_gate1.mean()),
            "p95_abs_delta": float(np.percentile(delta_gate1, 95)),
            "decision_flips": int((np.sign(logit_gate1) != np.sign(baseline_logit)).sum()),
        }
        
        # Restore original forward
        model.gated_fusion.forward = original_gated_forward
        del os.environ["GX1_DEBUG_FORCE_GATE"]
    else:
        log.warning("[DEBUG_IDENTITY] Gated fusion not enabled or not available, skipping force gate tests")
    
    # HARD FAIL checks
    if results["zero_xgb"]["max_abs_delta"] < 1e-6 and results["randomize_seq"]["max_abs_delta"] < 1e-6:
        log.error(
            "HARD_FAIL: Model not consuming XGB channels or sequence! "
            f"Zero XGB delta: {results['zero_xgb']['max_abs_delta']}, "
            f"Randomize seq delta: {results['randomize_seq']['max_abs_delta']}"
        )
        # Don't raise, let report be generated
    
    if gated_fusion_enabled and "force_gate_0" in results:
        if results["force_gate_0"]["max_abs_delta"] < 1e-6:
            log.error(
                "HARD_FAIL: Gated fusion not affecting output! "
                f"Force gate=0 delta: {results['force_gate_0']['max_abs_delta']}"
            )
            # Don't raise, let report be generated
    
    log.info("[DEBUG_IDENTITY] ✅ Intervention tests complete")
    return results


def generate_report(
    git_info: Dict,
    env_vars: Dict,
    policy_path: Path,
    artifacts: Dict,
    input_stats: Dict,
    intervention_results: Dict,
    output_dir: Path,
) -> Path:
    """Generate comprehensive debug report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"IDENTICAL_OUTPUT_ROOT_CAUSE_{timestamp}.md"
    
    with open(report_path, "w") as f:
        f.write("# Identical Output Root Cause Investigation\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Git info
        f.write("## Git Information\n\n")
        f.write(f"- **Commit:** `{git_info['commit']}`\n")
        f.write(f"- **Dirty State:** `{git_info['dirty']}`\n\n")
        
        # Policy
        f.write("## Policy Configuration\n\n")
        f.write(f"- **Policy Path:** `{policy_path.resolve()}`\n\n")
        
        # Env vars
        f.write("## Environment Variables\n\n")
        f.write("| Variable | Value |\n")
        f.write("|----------|-------|\n")
        for key, value in sorted(env_vars.items()):
            f.write(f"| `{key}` | `{value}` |\n")
        f.write("\n")
        
        # Artifacts
        f.write("## Artifact Verification\n\n")
        f.write("| Artifact | Path | Hash (SHA256) | Size (bytes) |\n")
        f.write("|----------|------|---------------|--------------|\n")
        for name, info in sorted(artifacts.items()):
            f.write(f"| `{name}` | `{info['path']}` | `{info['hash'][:16]}...` | {info['size_bytes']} |\n")
        f.write("\n")
        
        # Input tensor proof
        f.write("## Input Tensor Proof\n\n")
        f.write("### Shapes\n\n")
        f.write(f"- `seq_x`: {input_stats['seq_x']['shape']}\n")
        f.write(f"- `snap_x`: {input_stats['snap_x']['shape']}\n")
        f.write(f"- `ctx_cat`: {input_stats['ctx_cat']['shape']}\n")
        f.write(f"- `ctx_cont`: {input_stats['ctx_cont']['shape']}\n\n")
        
        f.write("### XGB Channels (seq_x[:, :, 13:16])\n\n")
        for channel, stats in input_stats["xgb_seq_channels"].items():
            f.write(f"- **{channel}**: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                   f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                   f"constant={stats['is_constant']}\n")
        f.write("\n")
        
        f.write("### XGB Channels (snap_x[:, 85:88])\n\n")
        for channel, stats in input_stats["xgb_snap_channels"].items():
            f.write(f"- **{channel}**: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                   f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                   f"constant={stats['is_constant']}\n")
        f.write("\n")
        
        f.write("### Context Features\n\n")
        f.write(f"- `ctx_cat`: unique_values={input_stats['ctx_cat']['unique_values']}\n")
        f.write(f"- `ctx_cont`: min={input_stats['ctx_cont']['min']:.4f}, "
               f"max={input_stats['ctx_cont']['max']:.4f}, "
               f"constant={input_stats['ctx_cont']['is_constant']}\n\n")
        
        # Intervention tests
        f.write("## Intervention Tests\n\n")
        f.write("| Test | Max Abs Delta | Mean Abs Delta | P95 Abs Delta | Decision Flips |\n")
        f.write("|------|---------------|----------------|---------------|----------------|\n")
        for test_name, test_results in intervention_results.items():
            if test_name == "baseline":
                continue
            f.write(f"| {test_name} | {test_results['max_abs_delta']:.6f} | "
                   f"{test_results['mean_abs_delta']:.6f} | "
                   f"{test_results['p95_abs_delta']:.6f} | "
                   f"{test_results['decision_flips']} |\n")
        f.write("\n")
        
        # Verdict
        f.write("## Verdict\n\n")
        
        # Analyze results
        zero_xgb_delta = intervention_results.get("zero_xgb", {}).get("max_abs_delta", 0.0)
        randomize_seq_delta = intervention_results.get("randomize_seq", {}).get("max_abs_delta", 0.0)
        
        if zero_xgb_delta < 1e-6 and randomize_seq_delta < 1e-6:
            f.write("**ROOT CAUSE: Model not consuming XGB channels or sequence inputs.**\n\n")
            f.write("**Fix:** Check model forward pass wiring, feature indices, and input mapping.\n\n")
        elif zero_xgb_delta < 1e-6:
            f.write("**ROOT CAUSE: XGB channels not affecting output.**\n\n")
            f.write("**Fix:** Verify XGB channel extraction and integration in model.\n\n")
        elif randomize_seq_delta < 1e-6:
            f.write("**ROOT CAUSE: Sequence not affecting output.**\n\n")
            f.write("**Fix:** Verify sequence encoder and temporal dependencies.\n\n")
        else:
            f.write("**ROOT CAUSE: Threshold/post-gates dominating decisions.**\n\n")
            f.write("**Fix:** Compare raw logits distributions, not thresholded decisions. "
                   "Consider adjusting evaluation methodology.\n\n")
        
        f.write("## Minimal Fix\n\n")
        f.write("(To be determined based on root cause analysis)\n\n")
    
    log.info(f"[DEBUG_IDENTITY] ✅ Report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Debug identical output root cause")
    parser.add_argument("--policy", type=str, required=True, help="Policy config path (alias for --policy_config)")
    parser.add_argument("--policy_config", type=str, help="Policy config path")
    parser.add_argument("--gated_checkpoint", type=str,
                       default="models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION")
    parser.add_argument("--baseline_checkpoint", type=str,
                       default="models/entry_v10_ctx/FULLYEAR_2025_BASELINE_NO_GATE")
    parser.add_argument("--val_data", type=str,
                       default="data/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION_val.parquet")
    parser.add_argument("--feature_meta_path", type=str,
                       default="gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json")
    parser.add_argument("--seq_scaler_path", type=str,
                       default="gx1/models/entry_v9/nextgen_2020_2025_clean/seq_scaler.joblib")
    parser.add_argument("--snap_scaler_path", type=str,
                       default="gx1/models/entry_v9/nextgen_2020_2025_clean/snap_scaler.joblib")
    parser.add_argument("--output_dir", type=str, default="reports/debug")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # Handle --policy alias
    if args.policy and not args.policy_config:
        args.policy_config = args.policy
    
    if not args.policy_config:
        raise ValueError("--policy or --policy_config is required")
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    log.info(f"[DEBUG_IDENTITY] Using device: {device}")
    
    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Git info
    git_info = get_git_info()
    log.info(f"[DEBUG_IDENTITY] Git commit: {git_info['commit'][:8]}, dirty: {git_info['dirty']}")
    
    # Env vars
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("GX1_")}
    log.info(f"[DEBUG_IDENTITY] GX1_* env vars: {len(env_vars)}")
    
    # Policy config
    policy_path = Path(args.policy_config)
    with open(policy_path) as f:
        policy = yaml.safe_load(f)
    
    log.info(f"[DEBUG_IDENTITY] Policy: {policy_path.resolve()}")
    
    # Load feature meta
    with open(args.feature_meta_path) as f:
        feature_meta = json.load(f)
    
    seq_feature_names = feature_meta.get("seq_features", [])
    snap_feature_names = feature_meta.get("snap_features", [])
    
    # Trace artifacts
    artifacts = trace_artifacts(policy_path, Path(args.gated_checkpoint), output_dir)
    
    # Load dataset
    val_df = pd.read_parquet(args.val_data)
    dataset = EntryV10CtxDataset(
        df=val_df,
        seq_feature_names=seq_feature_names,
        snap_feature_names=snap_feature_names,
        feature_meta_path=args.feature_meta_path,
        seq_scaler_path=Path(args.seq_scaler_path),
        snap_scaler_path=Path(args.snap_scaler_path),
        seq_len=30,
        lookback=30,
        policy_config=policy,
        warmup_bars=288,
        mode="train",
        allow_dummy_ctx=False,
    )
    
    # Capture input tensors
    input_stats, batch = capture_input_tensors(dataset, device, output_dir, n_samples=args.n_samples)
    
    # Load gated model
    gated_model, _ = load_model(Path(args.gated_checkpoint), device)
    
    # Run intervention tests
    intervention_results = run_intervention_tests(gated_model, batch, device, output_dir)
    
    # Generate report
    report_path = generate_report(
        git_info,
        env_vars,
        policy_path,
        artifacts,
        input_stats,
        intervention_results,
        output_dir,
    )
    
    log.info(f"[DEBUG_IDENTITY] ✅ Investigation complete. Report: {report_path}")


if __name__ == "__main__":
    main()
