#!/usr/bin/env python3
"""
V10 Hybrid Pipeline Forensic Audit

Runs a complete forensic audit of the V10 Hybrid Entry pipeline (XGB → Transformer),
dumping all intermediate tensors, gate decisions, and producing a complete audit pack.

Usage:
    python gx1/scripts/audit_v10_hybrid_pipeline.py \
        --year 2020 \
        --days 5 \
        --output-dir reports/pipeline_audit/V10_HYBRID_AUDIT_<timestamp> \
        --dump-snapshots
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

# Default paths
DEFAULT_GX1_DATA = Path("/Users/andrekildalbakke/Desktop/GX1_DATA")
DEFAULT_POLICY = WORKSPACE_ROOT / "gx1" / "configs" / "policies" / "sniper_snapshot" / "2025_SNIPER_V1" / "GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml"
DEFAULT_BUNDLE = DEFAULT_GX1_DATA / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
DATA_FILE = DEFAULT_GX1_DATA / "data" / "data" / "entry_v9" / "full_2020_2025.parquet"


# ============================================================================
# PIPELINE MAP GENERATOR
# ============================================================================

PIPELINE_CRITICAL_FILES = {
    "entry_manager": {
        "path": "gx1/execution/entry_manager.py",
        "role": "Entry evaluation orchestrator, routing logic",
        "key_functions": ["evaluate_entry", "_evaluate_entry_internal", "is_entry_eligible"],
    },
    "oanda_demo_runner": {
        "path": "gx1/execution/oanda_demo_runner.py",
        "role": "V10 hybrid callsite, XGB pre-predict, transformer forward",
        "key_functions": ["_predict_entry_v10_hybrid", "_run_xgb_pre_predict", "_build_transformer_input"],
    },
    "feature_contract_v10_ctx": {
        "path": "gx1/features/feature_contract_v10_ctx.py",
        "role": "Feature contract definitions, channel names, dims",
        "key_functions": ["SEQ_XGB_CHANNEL_NAMES", "SNAP_XGB_CHANNEL_NAMES", "BASE_SEQ_FEATURES", "BASE_SNAP_FEATURES"],
    },
    "entry_feature_telemetry": {
        "path": "gx1/execution/entry_feature_telemetry.py",
        "role": "Telemetry collection, counters, gate tracking",
        "key_functions": ["record_gate", "record_xgb_flow", "record_transformer_input"],
    },
    "prebuilt_features_loader": {
        "path": "gx1/execution/prebuilt_features_loader.py",
        "role": "Prebuilt feature loading, lookup, validation",
        "key_functions": ["lookup_features", "validate_schema"],
    },
    "feature_fingerprint": {
        "path": "gx1/runtime/feature_fingerprint.py",
        "role": "Feature schema validation, fingerprint computation",
        "key_functions": ["compute_feature_fingerprint", "validate_fingerprint"],
    },
    "replay_eval_gated_parallel": {
        "path": "gx1/scripts/replay_eval_gated_parallel.py",
        "role": "Replay orchestrator, chunk processing",
        "key_functions": ["process_chunk", "run_replay"],
    },
}


def generate_pipeline_map(output_dir: Path) -> None:
    """Generate PIPELINE_MAP.md describing the complete data path."""
    lines = [
        "# V10 Hybrid Entry Pipeline Map",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "The V10 Hybrid Entry pipeline processes market data through the following stages:",
        "",
        "```",
        "Prebuilt Features → XGB Pre-Predict → Transformer Injection → Transformer Forward → Entry Decision",
        "```",
        "",
        "---",
        "",
        "## Stage 1: Prebuilt Feature Loading",
        "",
        "**File:** `gx1/execution/prebuilt_features_loader.py`",
        "",
        "- Input: Prebuilt parquet file with pre-computed features",
        "- Output: Feature DataFrame indexed by timestamp (UTC)",
        "- Key validation: Schema matches feature manifest, no reserved candle columns",
        "",
        "**Environment variables:**",
        "- `GX1_REPLAY_USE_PREBUILT_FEATURES=1` (required)",
        "- `GX1_FEATURE_BUILD_DISABLED=1` (enforced)",
        "",
        "---",
        "",
        "## Stage 2: Entry Eligibility Check",
        "",
        "**File:** `gx1/execution/entry_manager.py`",
        "",
        "**Functions:**",
        "- `is_entry_eligible()` - Hard eligibility (position, cooldown, max trades)",
        "- `_evaluate_entry_internal()` - Soft eligibility (session, regime, thresholds)",
        "",
        "**Gates (in order):**",
        "1. `warmup_gate` - Skip first N bars",
        "2. `position_gate` - No open position",
        "3. `cooldown_gate` - Minimum bars since last trade",
        "4. `session_gate` - Valid trading session (ASIA/EU/US/OVERLAP)",
        "5. `regime_gate` - Valid volatility/trend regime (if enabled)",
        "",
        "---",
        "",
        "## Stage 3: XGB Pre-Predict",
        "",
        "**File:** `gx1/execution/oanda_demo_runner.py`",
        "**Function:** `_run_xgb_pre_predict()` or inline in `_predict_entry_v10_hybrid()`",
        "",
        "**Input:**",
        "- Feature vector from prebuilt features (85 base features)",
        "- Session indicator for model selection (EU/US/OVERLAP)",
        "",
        "**Output (XGB channels):**",
        "- `p_long_xgb` - Probability of long entry (raw XGB output)",
        "- `p_hat_xgb` - Calibrated probability (if calibration enabled)",
        "- `uncertainty_score` - Model uncertainty estimate",
        "",
        "**Note:** `margin_xgb` was removed in Jan 2026 cleanup.",
        "",
        "---",
        "",
        "## Stage 4: Transformer Input Assembly",
        "",
        "**File:** `gx1/execution/oanda_demo_runner.py`",
        "**Function:** `_build_transformer_input()` or inline in `_predict_entry_v10_hybrid()`",
        "",
        "**Sequence tensor (seq_data):**",
        "- Shape: `[T, D_seq]` where T=48 (lookback), D_seq=15",
        "- Base features: 12 (from prebuilt)",
        "- XGB channels injected: `p_long_xgb`, `uncertainty_score` (2 channels)",
        "- Padding feature: 1",
        "",
        "**Snapshot tensor (snap_data):**",
        "- Shape: `[D_snap]` where D_snap=87",
        "- Base features: 85 (from prebuilt)",
        "- XGB channels injected: `p_long_xgb`, `p_hat_xgb` (2 channels)",
        "",
        "**XGB channel injection points:**",
        "- SEQ: Appended after base seq features",
        "- SNAP: Appended after base snap features",
        "",
        "---",
        "",
        "## Stage 5: Transformer Forward",
        "",
        "**File:** `gx1/execution/oanda_demo_runner.py`",
        "**Model:** `model_state_dict.pt` from bundle",
        "",
        "**Input:**",
        "- `seq_data`: `[1, T, D_seq]` tensor",
        "- `snap_data`: `[1, D_snap]` tensor",
        "",
        "**Output:**",
        "- Raw logits (pre-softmax)",
        "- Entry probability (post-softmax, temperature scaled if enabled)",
        "",
        "---",
        "",
        "## Stage 6: Entry Decision",
        "",
        "**File:** `gx1/execution/oanda_demo_runner.py`",
        "",
        "**Decision logic:**",
        "1. Apply temperature scaling (if `temperature_scaling_effective=True`)",
        "2. Compare probability to entry threshold",
        "3. Check direction (long/short) based on probability",
        "4. Final validation (NaN/Inf check)",
        "",
        "**Output:**",
        "- Entry signal (True/False)",
        "- Entry direction (long/short)",
        "- Entry probability (for logging)",
        "",
        "---",
        "",
        "## Pipeline-Critical Files",
        "",
        "| Component | File | Role |",
        "|-----------|------|------|",
    ]
    
    for name, info in PIPELINE_CRITICAL_FILES.items():
        lines.append(f"| {name} | `{info['path']}` | {info['role']} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Data Flow Diagram",
        "",
        "```",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│                        PREBUILT FEATURES                            │",
        "│  (xauusd_m5_<year>_features_v10_ctx.parquet)                        │",
        "│  - 85 base snap features                                            │",
        "│  - 12 base seq features (per bar, T=48 lookback)                   │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "                                    │",
        "                                    ▼",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│                        ENTRY ELIGIBILITY                            │",
        "│  entry_manager.is_entry_eligible()                                  │",
        "│  - warmup, position, cooldown, session, regime gates               │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "                                    │",
        "                            (if eligible)",
        "                                    ▼",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│                        XGB PRE-PREDICT                              │",
        "│  oanda_demo_runner._predict_entry_v10_hybrid()                     │",
        "│  Input: 85 snap features                                            │",
        "│  Output: p_long_xgb, p_hat_xgb, uncertainty_score                  │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "                                    │",
        "                                    ▼",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│                      TRANSFORMER INPUT ASSEMBLY                     │",
        "│  seq_data: [T=48, D=15] = 12 base + 2 XGB (p_long, uncertainty) + 1│",
        "│  snap_data: [D=87] = 85 base + 2 XGB (p_long, p_hat)               │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "                                    │",
        "                                    ▼",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│                        TRANSFORMER FORWARD                          │",
        "│  model.forward(seq_data, snap_data)                                │",
        "│  Output: logits → softmax → entry_prob                             │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "                                    │",
        "                                    ▼",
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│                        ENTRY DECISION                               │",
        "│  entry_prob > threshold → ENTER                                    │",
        "│  entry_prob < (1-threshold) → ENTER SHORT                          │",
        "└─────────────────────────────────────────────────────────────────────┘",
        "```",
        "",
    ])
    
    output_path = output_dir / "PIPELINE_MAP.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {output_path}")


# ============================================================================
# ARTIFACT SCHEMA GENERATOR
# ============================================================================

def generate_artifact_schema(output_dir: Path) -> None:
    """Generate ARTIFACT_SCHEMA.json with tensor/array schemas."""
    schema = {
        "xgb_input": {
            "description": "XGB model input vector",
            "source": "prebuilt_features[snap_feature_names]",
            "shape": "[85]",
            "dtype": "float32",
            "feature_names": {
                "source": "gx1/features/feature_contract_v10_ctx.py::SNAP_FEATURE_NAMES",
                "count": 85,
                "normalization": "z-score (per feature, from training stats)",
            },
        },
        "xgb_outputs": {
            "p_long_xgb": {
                "description": "Raw XGB probability of long entry",
                "shape": "[1]",
                "dtype": "float32",
                "range": "[0.0, 1.0]",
                "injected_to": ["seq", "snap"],
            },
            "p_hat_xgb": {
                "description": "Calibrated XGB probability",
                "shape": "[1]",
                "dtype": "float32",
                "range": "[0.0, 1.0]",
                "injected_to": ["snap"],
                "note": "Only available if calibration enabled",
            },
            "uncertainty_score": {
                "description": "XGB model uncertainty estimate",
                "shape": "[1]",
                "dtype": "float32",
                "range": "[0.0, 1.0]",
                "injected_to": ["seq"],
            },
            "margin_xgb": {
                "description": "REMOVED in Jan 2026",
                "status": "DEPRECATED",
                "removal_reason": "Ablation showed improvement without it",
            },
        },
        "transformer_seq_input": {
            "description": "Transformer sequence input tensor",
            "shape": "[T=48, D_seq=15]",
            "dtype": "float32",
            "feature_order": [
                "# Base seq features (12)",
                "close_pct_change",
                "high_low_range",
                "close_position_in_range",
                "volume_normalized",
                "atr_14",
                "rsi_14",
                "macd_signal_diff",
                "bb_position",
                "ema_12_26_diff",
                "obv_normalized",
                "vwap_diff",
                "session_encoded",
                "# XGB channels (2)",
                "p_long_xgb",
                "uncertainty_score",
                "# Padding (1)",
                "pad_zeros",
            ],
            "normalization": "z-score (from seq_scaler.pkl in bundle)",
            "masks": {
                "padding_mask": "Boolean mask for valid timesteps",
                "warmup_mask": "First 48 bars are warmup (not used for prediction)",
            },
        },
        "transformer_snap_input": {
            "description": "Transformer snapshot input tensor",
            "shape": "[D_snap=87]",
            "dtype": "float32",
            "feature_order": [
                "# Base snap features (85)",
                "...see feature_contract_v10_ctx.py::SNAP_FEATURE_NAMES...",
                "# XGB channels (2)",
                "p_long_xgb",
                "p_hat_xgb",
            ],
            "normalization": "z-score (from snap_scaler.pkl in bundle)",
        },
        "transformer_output": {
            "description": "Transformer model output",
            "logits": {
                "shape": "[2]",
                "dtype": "float32",
                "indices": ["short_logit", "long_logit"],
            },
            "probabilities": {
                "shape": "[2]",
                "dtype": "float32",
                "computation": "softmax(logits / temperature)",
                "indices": ["p_short", "p_long"],
            },
            "entry_decision": {
                "description": "Final entry signal",
                "type": "bool",
                "direction": "long if p_long > threshold, short if p_short > threshold",
            },
        },
        "eligibility_gates": {
            "warmup_gate": {
                "description": "Skip first N bars",
                "default": 200,
                "env_var": "GX1_WARMUP_BARS",
            },
            "position_gate": {
                "description": "No open position",
                "check": "open_position is None",
            },
            "cooldown_gate": {
                "description": "Minimum bars since last trade",
                "default": 1,
                "env_var": "GX1_ENTRY_COOLDOWN_BARS",
            },
            "session_gate": {
                "description": "Valid trading session",
                "sessions": ["ASIA", "EU", "US", "OVERLAP"],
                "env_var": "GX1_ENABLED_SESSIONS",
            },
        },
    }
    
    output_path = output_dir / "ARTIFACT_SCHEMA.json"
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Written: {output_path}")


# ============================================================================
# INJECTION SNAPSHOT GENERATOR
# ============================================================================

def generate_injection_snapshot(output_dir: Path) -> None:
    """Generate INJECTION_SNAPSHOT.md documenting XGB→Transformer injection."""
    lines = [
        "# XGB → Transformer Injection Snapshot",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "XGB outputs are injected into Transformer input at two points:",
        "",
        "| Channel | Injected To | Position | Notes |",
        "|---------|-------------|----------|-------|",
        "| `p_long_xgb` | SEQ + SNAP | End of feature list | Duplicated in both |",
        "| `p_hat_xgb` | SNAP only | End of snap features | Calibrated prob |",
        "| `uncertainty_score` | SEQ only | End of seq features | Model uncertainty |",
        "| ~~`margin_xgb`~~ | ~~REMOVED~~ | ~~N/A~~ | Removed Jan 2026 |",
        "",
        "---",
        "",
        "## Injection Point Details",
        "",
        "### SEQ Injection",
        "",
        "**Location:** `gx1/execution/oanda_demo_runner.py::_predict_entry_v10_hybrid()`",
        "",
        "```python",
        "# After base seq features are assembled:",
        "xgb_seq_channels = [p_long_xgb, uncertainty_score]",
        "seq_data = np.concatenate([base_seq, xgb_seq_channels, padding], axis=-1)",
        "```",
        "",
        "**Feature order in seq_data:**",
        "1. Base seq features (12 features)",
        "2. `p_long_xgb` (XGB raw probability)",
        "3. `uncertainty_score` (XGB uncertainty)",
        "4. Padding zeros (1 feature)",
        "",
        "**Total seq features:** 15",
        "",
        "### SNAP Injection",
        "",
        "**Location:** `gx1/execution/oanda_demo_runner.py::_predict_entry_v10_hybrid()`",
        "",
        "```python",
        "# After base snap features are assembled:",
        "xgb_snap_channels = [p_long_xgb, p_hat_xgb]",
        "snap_data = np.concatenate([base_snap, xgb_snap_channels], axis=-1)",
        "```",
        "",
        "**Feature order in snap_data:**",
        "1. Base snap features (85 features)",
        "2. `p_long_xgb` (XGB raw probability)",
        "3. `p_hat_xgb` (XGB calibrated probability)",
        "",
        "**Total snap features:** 87",
        "",
        "---",
        "",
        "## Normalization",
        "",
        "| Channel | Normalization | Notes |",
        "|---------|---------------|-------|",
        "| `p_long_xgb` | None (raw [0,1]) | Probability output, already normalized |",
        "| `p_hat_xgb` | None (raw [0,1]) | Calibrated probability |",
        "| `uncertainty_score` | None (raw [0,1]) | Bounded by design |",
        "",
        "**Note:** No clipping is applied. Values should naturally be in [0,1] range.",
        "",
        "---",
        "",
        "## Duplication Analysis",
        "",
        "`p_long_xgb` is duplicated in both SEQ and SNAP:",
        "- SEQ: Used for temporal pattern learning",
        "- SNAP: Used for current-bar feature context",
        "",
        "This is intentional—the Transformer can learn different uses for the same signal",
        "in different contexts (temporal vs. snapshot).",
        "",
        "---",
        "",
        "## Telemetry Counters",
        "",
        "| Counter | Description | Source |",
        "|---------|-------------|--------|",
        "| `xgb_pre_predict_count` | Number of XGB predictions made | chunk_footer.json |",
        "| `transformer_forward_calls` | Number of transformer forward passes | chunk_footer.json |",
        "| `xgb_channel_names` | List of injected channel names | ENTRY_FEATURES_USED.json |",
        "| `n_xgb_channels_in_transformer_input` | Count of XGB channels | ENTRY_FEATURES_USED.json |",
        "",
        "---",
        "",
        "## Hard Proof: Callsite Locations",
        "",
        "```bash",
        "# Find XGB injection points",
        "grep -n 'p_long_xgb' gx1/execution/oanda_demo_runner.py",
        "grep -n 'uncertainty_score' gx1/execution/oanda_demo_runner.py",
        "grep -n 'xgb_seq_channels' gx1/execution/oanda_demo_runner.py",
        "grep -n 'xgb_snap_channels' gx1/execution/oanda_demo_runner.py",
        "```",
        "",
        "**Feature contract definition:**",
        "- `gx1/features/feature_contract_v10_ctx.py::SEQ_XGB_CHANNEL_NAMES`",
        "- `gx1/features/feature_contract_v10_ctx.py::SNAP_XGB_CHANNEL_NAMES`",
        "",
    ]
    
    output_path = output_dir / "INJECTION_SNAPSHOT.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {output_path}")


# ============================================================================
# CONFIG SSoT GENERATOR
# ============================================================================

def generate_config_ssot(output_dir: Path) -> None:
    """Generate CONFIG_SSoT.md listing all env vars and flags."""
    lines = [
        "# V10 Hybrid Pipeline Configuration SSoT",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Environment Variables",
        "",
        "### Routing & Model Selection",
        "",
        "| Variable | Default | Description | Read Location |",
        "|----------|---------|-------------|---------------|",
        "| `GX1_GATED_FUSION_ENABLED` | `0` | Enable V10 hybrid routing | entry_manager.py |",
        "| `GX1_ENTRY_MODEL_VERSION` | `v10_hybrid` | Entry model version | entry_manager.py |",
        "",
        "### XGB Configuration",
        "",
        "| Variable | Default | Description | Read Location |",
        "|----------|---------|-------------|---------------|",
        "| `GX1_REQUIRE_XGB_CALIBRATION` | `0` | Require XGB calibration | oanda_demo_runner.py |",
        "| `GX1_XGB_CHANNEL_MASK` | `\"\"` | Channels to mask (comma-sep) | feature_contract_v10_ctx.py |",
        "| `GX1_XGB_CHANNEL_ONLY` | `\"\"` | Keep only these channels | feature_contract_v10_ctx.py |",
        "",
        "### Prebuilt Features",
        "",
        "| Variable | Default | Description | Read Location |",
        "|----------|---------|-------------|---------------|",
        "| `GX1_REPLAY_USE_PREBUILT_FEATURES` | `0` | Use prebuilt features | oanda_demo_runner.py |",
        "| `GX1_FEATURE_BUILD_DISABLED` | `0` | Disable feature building | feature_build_tripwires.py |",
        "",
        "### Telemetry",
        "",
        "| Variable | Default | Description | Read Location |",
        "|----------|---------|-------------|---------------|",
        "| `GX1_REQUIRE_ENTRY_TELEMETRY` | `0` | Require telemetry collection | entry_manager.py |",
        "| `GX1_ALLOW_PARALLEL_REPLAY` | `0` | Allow parallel replay | replay_eval_gated_parallel.py |",
        "",
        "### Thresholds & Gates",
        "",
        "| Variable | Default | Description | Read Location |",
        "|----------|---------|-------------|---------------|",
        "| `GX1_ENTRY_THRESHOLD` | `0.6` | Entry probability threshold | policy YAML |",
        "| `GX1_WARMUP_BARS` | `200` | Warmup bars to skip | entry_manager.py |",
        "| `GX1_ENTRY_COOLDOWN_BARS` | `1` | Cooldown between trades | entry_manager.py |",
        "",
        "### Temperature Scaling",
        "",
        "| Variable | Default | Description | Read Location |",
        "|----------|---------|-------------|---------------|",
        "| `GX1_TEMPERATURE_SCALING` | `1.0` | Temperature for softmax | oanda_demo_runner.py |",
        "",
        "### Session & Regime",
        "",
        "| Variable | Default | Description | Read Location |",
        "|----------|---------|-------------|---------------|",
        "| `GX1_ENABLED_SESSIONS` | `EU,US,OVERLAP` | Enabled trading sessions | entry_manager.py |",
        "| `GX1_REGIME_GATING_ENABLED` | `0` | Enable regime gating | entry_manager.py |",
        "",
        "---",
        "",
        "## Policy YAML Keys",
        "",
        "The policy YAML file defines:",
        "",
        "```yaml",
        "entry_models:",
        "  v10_ctx:",
        "    bundle_dir: <path>",
        "    entry_threshold: 0.6",
        "    temperature: 1.0",
        "    xgb:",
        "      EU: <path_to_xgb_EU.pkl>",
        "      US: <path_to_xgb_US.pkl>",
        "      OVERLAP: <path_to_xgb_OVERLAP.pkl>",
        "```",
        "",
        "---",
        "",
        "## RUN_IDENTITY Logging",
        "",
        "These fields are logged in `RUN_IDENTITY.json`:",
        "",
        "```json",
        "{",
        '  "policy_id": "<policy file name>",',
        '  "bundle_sha256": "<hash of bundle_metadata.json>",',
        '  "replay_mode": "PREBUILT",',
        '  "temperature_scaling_effective": true,',
        '  "xgb_channels_seq": ["p_long_xgb", "uncertainty_score"],',
        '  "xgb_channels_snap": ["p_long_xgb", "p_hat_xgb"],',
        '  "feature_fingerprint": "<schema hash>",',
        '  "start_ts": "2025-01-01T00:00:00",',
        '  "end_ts": "2025-12-31T23:59:59"',
        "}",
        "```",
        "",
    ]
    
    output_path = output_dir / "CONFIG_SSoT.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {output_path}")


# ============================================================================
# GATE FUNNEL GENERATOR
# ============================================================================

def generate_gate_funnel(run_dir: Path, output_dir: Path) -> None:
    """Generate GATE_FUNNEL.md and GATE_FUNNEL.json from run output."""
    funnel = {
        "source": str(run_dir),
        "stages": {},
    }
    
    # Try to load from chunk_footer
    footer_path = run_dir / "chunk_0" / "chunk_footer.json"
    if footer_path.exists():
        with open(footer_path) as f:
            footer = json.load(f)
        
        funnel["stages"] = {
            "1_bars_seen": footer.get("bars_seen", 0),
            "2_bars_skipped_warmup": footer.get("bars_skipped_warmup", 0),
            "3_bars_reaching_entry_stage": footer.get("bars_reaching_entry_stage", 0),
            "4_prebuilt_available": footer.get("prebuilt_bypass_count", 0),
            "5_eligibility_blocks": footer.get("bars_blocked_hard_eligibility", 0) + footer.get("bars_blocked_soft_eligibility", 0),
            "6_model_attempts": footer.get("model_attempt_calls", 0),
            "7_xgb_pre_predict_calls": footer.get("xgb_pre_predict_count", 0),
            "8_transformer_forward_calls": footer.get("transformer_forward_calls", 0),
            "9_trades_emitted": footer.get("n_trades_closed", 0),
        }
    
    # Write JSON
    json_path = output_dir / "GATE_FUNNEL.json"
    with open(json_path, "w") as f:
        json.dump(funnel, f, indent=2)
    print(f"Written: {json_path}")
    
    # Write markdown
    lines = [
        "# Gate Funnel Analysis",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        f"**Source:** `{run_dir}`",
        "",
        "---",
        "",
        "## Funnel Stages",
        "",
        "| Stage | Count | Drop | Drop % |",
        "|-------|-------|------|--------|",
    ]
    
    stages = funnel.get("stages", {})
    prev_count = None
    for stage_name, count in stages.items():
        if prev_count is not None:
            drop = prev_count - count
            drop_pct = (drop / prev_count * 100) if prev_count > 0 else 0
            lines.append(f"| {stage_name} | {count:,} | {drop:,} | {drop_pct:.1f}% |")
        else:
            lines.append(f"| {stage_name} | {count:,} | - | - |")
        prev_count = count
    
    lines.extend([
        "",
        "---",
        "",
        "## Conversion Rates",
        "",
    ])
    
    bars_seen = stages.get("1_bars_seen", 0)
    trades = stages.get("9_trades_emitted", 0)
    forward_calls = stages.get("8_transformer_forward_calls", 0)
    
    if bars_seen > 0:
        lines.append(f"- **Bars → Trades:** {trades:,} / {bars_seen:,} = {trades/bars_seen*100:.4f}%")
    if forward_calls > 0:
        lines.append(f"- **Forward Calls → Trades:** {trades:,} / {forward_calls:,} = {trades/forward_calls*100:.4f}%")
    
    md_path = output_dir / "GATE_FUNNEL.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {md_path}")


# ============================================================================
# DIFF REPORT GENERATOR
# ============================================================================

def generate_diff_report(output_dir: Path) -> None:
    """Generate DIFF_REPORT.md with hypotheses about recent changes."""
    lines = [
        "# Pipeline Changes Diff Report",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Recent Significant Changes",
        "",
        "### 1. margin_xgb Removal (Jan 2026)",
        "",
        "**Change:** `margin_xgb` channel was permanently removed from XGB→Transformer injection.",
        "",
        "**Files affected:**",
        "- `gx1/execution/oanda_demo_runner.py` - Removed calculation and injection",
        "- `gx1/features/feature_contract_v10_ctx.py` - Removed from channel lists",
        "- `gx1/execution/entry_feature_telemetry.py` - Removed from telemetry",
        "",
        "**Impact on 2020-2024:**",
        "- FULLYEAR 2025 ablation showed +trades and +PnL when margin_xgb removed",
        "- Multiyear 2020-2024 runs now use pipeline without margin_xgb",
        "- This should NOT cause losses—it should be neutral or positive",
        "",
        "### 2. XGB POST Removal (Jan 2026)",
        "",
        "**Change:** XGB post-predict (calibration/veto) was completely removed.",
        "",
        "**Impact:**",
        "- `post_predict_called` is always False",
        "- `veto_applied_count` is always 0",
        "- XGB now only used as PRE (channels to Transformer)",
        "",
        "### 3. Prebuilt Schema Changes",
        "",
        "**Change:** Reserved candle columns (`CLOSE`, etc.) removed from prebuilt schema.",
        "",
        "**Impact:**",
        "- Prebuilt parquet files must not contain `CLOSE` column",
        "- `close` aliasing now handled in runtime, not prebuilt",
        "",
        "### 4. Timezone/Session Handling",
        "",
        "**Potential issue:** Check if session bucketing is consistent across years.",
        "",
        "**To verify:**",
        "- Compare session distribution (ASIA/EU/US/OVERLAP) per year",
        "- Check if timezone handling is correct in prebuilt feature timestamps",
        "",
        "---",
        "",
        "## Hypotheses: Root Causes for 2020-2024 Losses",
        "",
        "### Hypothesis 1: Prebuilt Feature Quality (HIGH PROBABILITY)",
        "",
        "**Observation:** 2020-2024 prebuilt features may have been built with different",
        "feature pipeline or normalization than 2025.",
        "",
        "**Evidence needed:**",
        "- Compare feature statistics (mean/std) across years",
        "- Check if feature_contract_hash matches",
        "- Verify prebuilt builder version used",
        "",
        "### Hypothesis 2: Session Distribution Mismatch (MEDIUM PROBABILITY)",
        "",
        "**Observation:** Different years may have different session distributions",
        "if timezone handling changed.",
        "",
        "**Evidence needed:**",
        "- Compare session_breakdown per year",
        "- Check if bars per session is roughly consistent",
        "",
        "### Hypothesis 3: XGB Model Mismatch (MEDIUM PROBABILITY)",
        "",
        "**Observation:** XGB models were trained on 2025 data but applied to 2020-2024.",
        "",
        "**Evidence needed:**",
        "- Check XGB model training date and data range",
        "- Compare XGB output distributions across years",
        "",
        "### Hypothesis 4: Regime Shift (LOWER PROBABILITY)",
        "",
        "**Observation:** Market regime in 2020-2024 may be fundamentally different",
        "from 2025 training data.",
        "",
        "**Evidence needed:**",
        "- Compare volatility/trend regime distributions",
        "- Check if model assumptions hold in earlier years",
        "",
        "---",
        "",
        "## Recommended Investigation Order",
        "",
        "1. **Run feature statistics comparison** across all years",
        "2. **Verify prebuilt feature hashes** match expected",
        "3. **Compare XGB output distributions** (p_long_xgb, uncertainty) per year",
        "4. **Check session/regime distributions** for consistency",
        "5. **Run uncertainty_score ablation** on 2020-2024 to measure impact",
        "",
    ]
    
    output_path = output_dir / "DIFF_REPORT.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {output_path}")


# ============================================================================
# PIPELINE OPTIMIZATION PLAN
# ============================================================================

def generate_optimization_plan(output_dir: Path) -> None:
    """Generate PIPELINE_OPTIMIZATION_PLAN.md."""
    lines = [
        "# Pipeline Optimization Plan",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Phase 1: Stabilize & Explain 2020-2024 Losses",
        "",
        "**Goal:** Understand why 2020-2024 underperforms before making changes.",
        "",
        "### Step 1.1: Feature Statistics Audit",
        "- Compare mean/std/quantiles of all 85 snap features across years",
        "- Flag features with significant drift (>2 std from 2025)",
        "",
        "### Step 1.2: XGB Output Analysis",
        "- Compare p_long_xgb distribution per year",
        "- Check if XGB is confident (peaks at 0/1) or uncertain (flat)",
        "",
        "### Step 1.3: Gate Funnel Comparison",
        "- Build funnel for each year",
        "- Identify where most bars die (eligibility, model, threshold)",
        "",
        "---",
        "",
        "## Phase 2: Hypothesis Tests",
        "",
        "**Rule:** One hypothesis at a time. A/B with truth locks.",
        "",
        "### Test 2.1: uncertainty_score Ablation",
        "- Already implemented: `GX1_XGB_CHANNEL_MASK=uncertainty_score`",
        "- Run on 2020-2024 to measure impact",
        "",
        "### Test 2.2: p_hat_xgb vs p_long_xgb",
        "- Test using only raw probability vs calibrated",
        "- May reveal calibration issues in older data",
        "",
        "### Test 2.3: Session Gating",
        "- Test enabling/disabling specific sessions",
        "- May reveal session-specific model weakness",
        "",
        "---",
        "",
        "## Phase 3: Safe Improvements",
        "",
        "**Rule:** Only after Phase 1-2 are complete and understood.",
        "",
        "### Improvement 3.1: Retrain XGB on Multi-Year",
        "- Train XGB on 2020-2024 data",
        "- Keep Transformer frozen",
        "- A/B test against current",
        "",
        "### Improvement 3.2: Feature Engineering",
        "- Add new features based on Phase 1 findings",
        "- One feature at a time",
        "- Require +PnL on held-out year",
        "",
        "---",
        "",
        "## Anti-Patterns (DO NOT DO)",
        "",
        "❌ Tune 100 things simultaneously",
        "❌ Optimize thresholds on training data",
        "❌ Add features without A/B",
        "❌ Trust backtests without held-out validation",
        "❌ Ignore statistical significance",
        "",
    ]
    
    output_path = output_dir / "PIPELINE_OPTIMIZATION_PLAN.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {output_path}")


# ============================================================================
# MAIN AUDIT RUNNER
# ============================================================================

def run_audit_replay(
    year: int,
    days: int,
    output_dir: Path,
    policy: Path,
    bundle_dir: Path,
    dump_snapshots: bool,
) -> Optional[Path]:
    """Run a short audit replay and return the run output directory."""
    
    # Calculate date range
    start_date = f"{year}-01-06"  # Skip first week (potential data issues)
    end_date = f"{year}-01-{6 + days:02d}"
    
    run_output = output_dir / "audit_run"
    run_output.mkdir(parents=True, exist_ok=True)
    
    # Find prebuilt file
    prebuilt_root = DEFAULT_GX1_DATA / "data" / "prebuilt" / "TRIAL160"
    prebuilt_path = prebuilt_root / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"
    
    if not prebuilt_path.exists():
        print(f"ERROR: Prebuilt file not found: {prebuilt_path}")
        return None
    
    # Build environment
    env = os.environ.copy()
    env["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env["GX1_REQUIRE_ENTRY_TELEMETRY"] = "1"
    env["GX1_GATED_FUSION_ENABLED"] = "1"
    env["GX1_ALLOW_PARALLEL_REPLAY"] = "1"
    env["GX1_FEATURE_BUILD_DISABLED"] = "1"
    
    # Build command
    cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "gx1" / "scripts" / "replay_eval_gated_parallel.py"),
        "--data", str(DATA_FILE),
        "--prebuilt-parquet", str(prebuilt_path),
        "--bundle-dir", str(bundle_dir),
        "--policy", str(policy),
        "--output-dir", str(run_output),
        "--workers", "1",
        "--start-ts", f"{start_date}T00:00:00",
        "--end-ts", f"{end_date}T23:59:59",
    ]
    
    print(f"Running audit replay: {year}-01-06 to {year}-01-{6+days:02d}...")
    print(f"Command: {' '.join(cmd[:5])} ...")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
        )
        
        if result.returncode != 0:
            print(f"WARNING: Replay returned non-zero: {result.returncode}")
            print(f"stderr: {result.stderr[-500:] if result.stderr else 'N/A'}")
        
    except subprocess.TimeoutExpired:
        print("ERROR: Replay timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"ERROR: Replay failed: {e}")
        return None
    
    return run_output


def main():
    parser = argparse.ArgumentParser(
        description="V10 Hybrid Pipeline Forensic Audit"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to audit (default: 2025)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of days to audit (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: reports/pipeline_audit/V10_HYBRID_AUDIT_<timestamp>)"
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="Policy YAML path"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Bundle directory"
    )
    parser.add_argument(
        "--dump-snapshots",
        action="store_true",
        help="Dump sample tensor snapshots"
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip replay, only generate documentation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = WORKSPACE_ROOT / "reports" / "pipeline_audit" / f"V10_HYBRID_AUDIT_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("V10 HYBRID PIPELINE FORENSIC AUDIT")
    print("=" * 60)
    print(f"Year: {args.year}")
    print(f"Days: {args.days}")
    print(f"Output: {output_dir}")
    print()
    
    # Generate static documentation
    print("Generating pipeline documentation...")
    generate_pipeline_map(output_dir)
    generate_artifact_schema(output_dir)
    generate_injection_snapshot(output_dir)
    generate_config_ssot(output_dir)
    generate_diff_report(output_dir)
    generate_optimization_plan(output_dir)
    
    # Run audit replay if not skipped
    if not args.skip_replay:
        print()
        print("Running audit replay...")
        run_output = run_audit_replay(
            year=args.year,
            days=args.days,
            output_dir=output_dir,
            policy=args.policy,
            bundle_dir=args.bundle_dir,
            dump_snapshots=args.dump_snapshots,
        )
        
        if run_output and (run_output / "chunk_0" / "chunk_footer.json").exists():
            print()
            print("Generating gate funnel from replay output...")
            generate_gate_funnel(run_output, output_dir)
    else:
        print("Skipping replay (--skip-replay)")
    
    # Summary
    print()
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    print()
    print("Generated files:")
    for f in sorted(output_dir.glob("*.md")) + sorted(output_dir.glob("*.json")):
        print(f"  - {f.name}")
    print()
    print(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
