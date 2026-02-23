#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Feature Manifest (SSoT) - Feature Control Plane v2.0

This script generates a machine-readable feature manifest from prebuilt parquet
and known feature families. The manifest serves as Single Source of Truth for
feature definitions, families, contracts, and usage.

Usage:
    python3 gx1/scripts/generate_feature_manifest.py \
        --prebuilt-parquet data/features/xauusd_m5_2025_features_v10_ctx.parquet \
        --output-json gx1/feature_manifest_v1.json \
        --output-md docs/FEATURE_MANIFEST.md
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Known feature families (from codebase analysis)
FEATURE_FAMILIES = {
    "atr": ["atr", "atr14", "atr50", "atr_z", "atr_regime_id"],
    "regime": ["atr_regime_id", "trend_regime", "vol_regime"],
    "session": ["session_id", "is_EU", "is_US", "is_OVERLAP"],
    "htf": ["h1_", "h4_", "htf_"],
    "microstructure": ["body_pct", "wick_asym", "body_share", "clv"],
    "trend": ["ema", "slope", "roc", "pos_vs_ema"],
    "zscore": ["zscore", "_z_"],
    "bollinger": ["bb_", "bb_squeeze", "bb_bandwidth"],
    "sequence": ["ema20_slope", "ema100_slope", "pos_vs_ema200", "std50", "atr50", "atr_z", "roc20", "roc100", "body_pct", "wick_asym", "session_id", "atr_regime_id", "trend_regime_tf24h"],
    # NEW: Feature Control Plane v2.0 families
    "price": ["mid", "range"],  # Raw price fields (CLOSE is input alias, not prebuilt feature)
    "htf_derived": ["_v1h1_", "_v1h4_"],  # HTF-derived features
    "model_output": ["prob_long", "prob_short", "prob_neutral"],  # Model outputs (not features)
    "returns": ["ret_", "roc"],  # Return features
    "volatility": ["rvol_", "vol_ratio"],  # Volatility features
    "risk": ["brain_risk_score"],  # Risk features
    "meta": ["side"],  # Meta/label fields
}

# Feature prefixes for family detection
FAMILY_PREFIXES = {
    "smc_": "smc_pack_v1",  # DEL 4: SMC Starter Pack v1
    "_v1_": "basic_v1",
    "_v1h1_": "htf_derived",  # NEW: HTF-derived H1 features
    "_v1h4_": "htf_derived",  # NEW: HTF-derived H4 features
    "atr": "atr",
    "session": "session",
    "h1_": "htf",
    "h4_": "htf",
    "htf_": "htf",
    "body_": "microstructure",
    "wick_": "microstructure",
    "ema": "trend",
    "roc": "trend",
    "slope": "trend",
    "zscore": "zscore",
    "_z_": "zscore",
    "bb_": "bollinger",
    "ret_": "returns",  # NEW: Return features
    "rvol_": "volatility",  # NEW: Volatility features
}

# Known sequence features (from sequence_features.py)
SEQUENCE_FEATURES = {
    "ema20_slope", "ema100_slope", "pos_vs_ema200", "std50", "atr50", "atr_z",
    "roc20", "roc100", "body_pct", "wick_asym", "session_id", "atr_regime_id",
    "trend_regime_tf24h"
}

# Features that are safe in PREBUILT (loaded from parquet, not built)
PREBUILT_SAFE = True  # All features in prebuilt parquet are safe

# Features that are live-only (not in prebuilt)
LIVE_ONLY_FEATURES = set()  # Will be populated if feature is not in prebuilt

# Consumers mapping
CONSUMERS = {
    "XGB": [],  # XGB calibrators (entry scores only)
    "TRANSFORMER": [],  # Transformer model (seq + snap features)
    "GATE_ONLY": ["session_id", "atr_regime_id", "is_EU", "is_US", "is_OVERLAP"],
    "LOG_ONLY": ["prob_long", "prob_short", "prob_neutral"],
}

# NEW: Feature Control Plane v2.0 - Explicit classification rules for previously "unknown" features
UNKNOWN_FEATURE_CLASSIFICATIONS = {
    # Price fields (NOTE: CLOSE is NOT a prebuilt feature - it's an input alias from candles.close)
    # CLOSE should never appear in prebuilt schema (dropped by sanitize_feature_columns)
    # If CLOSE appears here, the prebuilt schema is contaminated
    # "CLOSE": {
    #     "family": "price",
    #     "units": "price",
    #     "stage": "input_alias",  # DEL 2: CLOSE is input alias, not prebuilt feature
    #     "causal": True,
    #     "depends_on": ["candles"],
    #     "consumer_contract": "TRANSFORMER_SNAP",
    # },
    "mid": {
        "family": "price",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "range": {
        "family": "price",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    # HTF-derived features
    "_v1h1_atr": {
        "family": "htf_derived",
        "units": "bps",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h1"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h1_ema_diff": {
        "family": "htf_derived",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h1"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h1_rsi14_z": {
        "family": "htf_derived",
        "units": "zscore",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h1"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h1_slope3": {
        "family": "htf_derived",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h1"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h1_slope5": {
        "family": "htf_derived",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h1"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h1_vwap_drift": {
        "family": "htf_derived",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h1"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h4_atr": {
        "family": "htf_derived",
        "units": "bps",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h4"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h4_ema_diff": {
        "family": "htf_derived",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h4"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h4_rsi14_z": {
        "family": "htf_derived",
        "units": "zscore",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h4"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h4_slope3": {
        "family": "htf_derived",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h4"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "_v1h4_slope5": {
        "family": "htf_derived",
        "units": "price",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles", "htf_h4"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    # Model outputs (not features, but logged)
    "prob_long": {
        "family": "model_output",
        "units": "ratio",
        "stage": "derived_from_model",
        "causal": True,
        "depends_on": ["model_output"],
        "consumer_contract": "LOG",
    },
    "prob_short": {
        "family": "model_output",
        "units": "ratio",
        "stage": "derived_from_model",
        "causal": True,
        "depends_on": ["model_output"],
        "consumer_contract": "LOG",
    },
    "prob_neutral": {
        "family": "model_output",
        "units": "ratio",
        "stage": "derived_from_model",
        "causal": True,
        "depends_on": ["model_output"],
        "consumer_contract": "LOG",
    },
    # Return features
    "ret_1": {
        "family": "returns",
        "units": "ratio",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "ret_5": {
        "family": "returns",
        "units": "ratio",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "ret_20": {
        "family": "returns",
        "units": "ratio",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    # Volatility features
    "rvol_20": {
        "family": "volatility",
        "units": "ratio",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "rvol_60": {
        "family": "volatility",
        "units": "ratio",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    "vol_ratio": {
        "family": "volatility",
        "units": "ratio",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    # Risk features
    "brain_risk_score": {
        "family": "risk",
        "units": "ratio",  # 0..1 range (assumed, will be validated)
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
    # Meta/label fields
    "side": {
        "family": "meta",
        "units": "int",
        "stage": "prebuilt_only",
        "causal": True,
        "depends_on": ["candles"],
        "consumer_contract": "TRANSFORMER_SNAP",
    },
}

def detect_feature_family(feature_name: str) -> str:
    """Detect feature family from feature name."""
    # NEW: Check explicit classifications first
    if feature_name in UNKNOWN_FEATURE_CLASSIFICATIONS:
        return UNKNOWN_FEATURE_CLASSIFICATIONS[feature_name]["family"]
    
    feature_lower = feature_name.lower()
    
    # Check exact matches first
    for family, features in FEATURE_FAMILIES.items():
        if feature_name in features:
            return family
    
    # Check prefixes
    for prefix, family in FAMILY_PREFIXES.items():
        if feature_name.startswith(prefix):
            return family
    
    # Check if sequence feature
    if feature_name in SEQUENCE_FEATURES:
        return "sequence"
    
    # Default
    return "unknown"

def detect_timeframe(feature_name: str) -> str:
    """Detect timeframe from feature name."""
    # DEL 4: SMC features are M5-based (computed on M5 bars)
    if feature_name.startswith("smc_"):
        return "M5"
    elif "h1_" in feature_name.lower() or "_h1" in feature_name.lower() or feature_name.startswith("_v1h1_"):
        return "H1"
    elif "h4_" in feature_name.lower() or "_h4" in feature_name.lower() or feature_name.startswith("_v1h4_"):
        return "H4"
    elif "d1_" in feature_name.lower() or "_d1" in feature_name.lower():
        return "D1"
    else:
        return "M5"  # Default to M5

def detect_units(feature_name: str) -> str:
    """Detect units from feature name."""
    # NEW: Check explicit classifications first
    if feature_name in UNKNOWN_FEATURE_CLASSIFICATIONS:
        return UNKNOWN_FEATURE_CLASSIFICATIONS[feature_name]["units"]
    
    if "bps" in feature_name.lower() or ("atr" in feature_name.lower() and "z" not in feature_name.lower()):
        return "bps"
    elif "zscore" in feature_name.lower() or "_z_" in feature_name.lower():
        return "zscore"
    elif "pct" in feature_name.lower() or "ratio" in feature_name.lower():
        return "ratio"
    elif "id" in feature_name.lower() or feature_name.endswith("_id"):
        return "int"
    elif feature_name.startswith("is_") or feature_name.startswith("has_"):
        return "flag"
    elif feature_name in ["CLOSE", "mid", "range"]:
        return "price"
    elif feature_name.startswith("ret_") or feature_name.startswith("roc"):
        return "ratio"
    elif feature_name.startswith("rvol_") or "vol_ratio" in feature_name:
        return "ratio"
    elif feature_name in ["prob_long", "prob_short", "prob_neutral", "brain_risk_score"]:
        return "ratio"
    else:
        return "unknown"

def detect_normalization(feature_name: str) -> str:
    """Detect normalization method from feature name."""
    # DEL 4: SMC features with "_atr" suffix are ATR-normalized in bps
    if feature_name.startswith("smc_") and "_atr" in feature_name.lower():
        return "atr_h1"  # NEW: Explicit normalization contract
    elif "atr" in feature_name.lower() and ("norm" in feature_name.lower() or "_z_" in feature_name.lower()):
        return "atr_h1"  # NEW: Explicit normalization contract
    elif "zscore" in feature_name.lower() or "_z_" in feature_name.lower():
        return "z_rolling(10,100)"  # NEW: Explicit normalization contract
    else:
        return "none"

def detect_consumers(feature_name: str) -> List[str]:
    """Detect which consumers use this feature."""
    consumers = []
    
    # NEW: Check explicit classifications first
    if feature_name in UNKNOWN_FEATURE_CLASSIFICATIONS:
        consumer_contract = UNKNOWN_FEATURE_CLASSIFICATIONS[feature_name]["consumer_contract"]
        if consumer_contract == "TRANSFORMER_SNAP":
            consumers.append("TRANSFORMER")
        elif consumer_contract == "TRANSFORMER_SEQ":
            consumers.append("TRANSFORMER")
        elif consumer_contract == "LOG":
            consumers.append("LOG_ONLY")
        elif consumer_contract == "GATE":
            consumers.append("GATE_ONLY")
        return consumers
    
    # DEL 4: SMC features are initially LOG_ONLY (not used by model yet)
    if feature_name.startswith("smc_"):
        consumers.append("LOG_ONLY")
        return consumers  # Early return for SMC features
    
    # Check GATE_ONLY
    if feature_name in CONSUMERS["GATE_ONLY"]:
        consumers.append("GATE_ONLY")
    
    # Check LOG_ONLY
    if feature_name in CONSUMERS["LOG_ONLY"]:
        consumers.append("LOG_ONLY")
    
    # Check if sequence feature (TRANSFORMER)
    if feature_name in SEQUENCE_FEATURES:
        consumers.append("TRANSFORMER")
    
    # Check if _v1_ feature (likely TRANSFORMER snapshot)
    if feature_name.startswith("_v1_"):
        consumers.append("TRANSFORMER")
    
    # Default: TRANSFORMER (most features go to transformer)
    if not consumers:
        consumers.append("TRANSFORMER")
    
    return consumers

def get_source_module(feature_name: str) -> str:
    """Get source module for feature."""
    # DEL 4: SMC features come from smc_pack_v1
    if feature_name.startswith("smc_"):
        return "gx1.features.smc_pack_v1"
    elif feature_name in SEQUENCE_FEATURES:
        return "gx1.seq.sequence_features"
    elif feature_name.startswith("_v1_"):
        return "gx1.features.basic_v1"
    elif feature_name.startswith("_v1h1_") or feature_name.startswith("_v1h4_"):
        return "gx1.features.htf_aggregator"
    elif "h1_" in feature_name.lower() or "h4_" in feature_name.lower():
        return "gx1.features.htf_aggregator"
    else:
        return "gx1.features.runtime_sniper_core"

def compute_lineage_hash(
    name: str,
    source_module: str,
    timeframe: str,
    units_contract: str,
    normalization_contract: str,
    causal: bool,
    key_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute lineage hash for feature.
    
    Lineage hash is SHA256 over (name, source_module, timeframe, units_contract, normalization_contract, causal, key_params).
    This ensures that if any of these change, the hash changes, enabling detection of feature drift.
    """
    key_params_str = json.dumps(key_params or {}, sort_keys=True)
    hash_input = f"{name}|{source_module}|{timeframe}|{units_contract}|{normalization_contract}|{causal}|{key_params_str}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]  # First 16 chars for readability

def detect_stage(feature_name: str) -> str:
    """Detect feature stage."""
    # NEW: Check explicit classifications first
    if feature_name in UNKNOWN_FEATURE_CLASSIFICATIONS:
        return UNKNOWN_FEATURE_CLASSIFICATIONS[feature_name]["stage"]
    
    # Model outputs
    if feature_name in ["prob_long", "prob_short", "prob_neutral"]:
        return "derived_from_model"
    
    # Gate-only features
    if feature_name in CONSUMERS["GATE_ONLY"]:
        return "gate_only"
    
    # Log-only features
    if feature_name in CONSUMERS["LOG_ONLY"]:
        return "log_only"
    
    # Default: prebuilt_only (all features in prebuilt parquet are prebuilt_only)
    return "prebuilt_only"

def detect_causal(feature_name: str) -> bool:
    """Detect if feature is causal (no lookahead leakage)."""
    # NEW: Check explicit classifications first
    if feature_name in UNKNOWN_FEATURE_CLASSIFICATIONS:
        return UNKNOWN_FEATURE_CLASSIFICATIONS[feature_name]["causal"]
    
    # DEL 4: SMC features have explicit causal confirmation delay
    if feature_name.startswith("smc_"):
        return True  # Causal confirmation delay (R=5 bars) ensures no lookahead leakage
    
    # Default: assume causal (all prebuilt features should be causal)
    return True

def detect_depends_on(feature_name: str) -> List[str]:
    """Detect feature dependencies."""
    # NEW: Check explicit classifications first
    if feature_name in UNKNOWN_FEATURE_CLASSIFICATIONS:
        return UNKNOWN_FEATURE_CLASSIFICATIONS[feature_name]["depends_on"]
    
    # HTF features depend on HTF aggregators
    if feature_name.startswith("_v1h1_") or "h1_" in feature_name.lower():
        return ["candles", "htf_h1"]
    elif feature_name.startswith("_v1h4_") or "h4_" in feature_name.lower():
        return ["candles", "htf_h4"]
    
    # Model outputs depend on model
    if feature_name in ["prob_long", "prob_short", "prob_neutral"]:
        return ["model_output"]
    
    # Default: depends on candles
    return ["candles"]

def detect_consumer_contract(feature_name: str) -> str:
    """Detect consumer contract."""
    # NEW: Check explicit classifications first
    if feature_name in UNKNOWN_FEATURE_CLASSIFICATIONS:
        return UNKNOWN_FEATURE_CLASSIFICATIONS[feature_name]["consumer_contract"]
    
    # Sequence features
    if feature_name in SEQUENCE_FEATURES:
        return "TRANSFORMER_SEQ"
    
    # Gate-only features
    if feature_name in CONSUMERS["GATE_ONLY"]:
        return "GATE"
    
    # Log-only features
    if feature_name in CONSUMERS["LOG_ONLY"]:
        return "LOG"
    
    # Default: TRANSFORMER_SNAP (snapshot features)
    return "TRANSFORMER_SNAP"

def generate_feature_manifest(
    prebuilt_parquet_path: Path,
    output_json_path: Path,
    output_md_path: Path,
    allowlist_unknown: bool = False,
) -> Dict[str, Any]:
    """
    Generate feature manifest from prebuilt parquet.
    
    Args:
        prebuilt_parquet_path: Path to prebuilt features parquet
        output_json_path: Path to output JSON manifest
        output_md_path: Path to output Markdown manifest
        allowlist_unknown: If True, allow unknown features (for initial generation)
    
    Returns:
        Manifest dict
    """
    log.info(f"Reading prebuilt parquet: {prebuilt_parquet_path}")
    
    # Read parquet schema (efficient, no full load)
    parquet_file = pq.ParquetFile(prebuilt_parquet_path)
    schema = parquet_file.schema
    
    # Extract feature names (exclude metadata columns)
    metadata_cols = {"time", "ts", "__index_level_0__"}
    feature_names = [field.name for field in schema if field.name not in metadata_cols]
    
    # DEL 6: Hardening tripwire - hard-fail if CLOSE or other reserved columns found
    from gx1.runtime.column_collision_guard import RESERVED_CANDLE_COLUMNS
    reserved_found = []
    for col in feature_names:
        if col.lower() in RESERVED_CANDLE_COLUMNS:
            reserved_found.append(col)
    
    if reserved_found:
        raise RuntimeError(
            f"[MANIFEST_GENERATOR_FAIL] Prebuilt schema contains reserved candle columns: {reserved_found}. "
            f"Reserved columns (case-insensitive): {sorted(RESERVED_CANDLE_COLUMNS)}. "
            f"Prebuilt schema is contaminated - CLOSE and other reserved columns must be dropped before writing parquet. "
            f"Check that build_fullyear_features_parquet.py sanitize_feature_columns() is working correctly."
        )
    
    log.info(f"Found {len(feature_names)} features in prebuilt parquet (schema clean: no reserved columns)")
    
    # Generate manifest entries
    manifest_features = []
    unknown_features = []
    
    for feature_name in sorted(feature_names):
        family = detect_feature_family(feature_name)
        timeframe = detect_timeframe(feature_name)
        units = detect_units(feature_name)
        normalization = detect_normalization(feature_name)
        consumers = detect_consumers(feature_name)
        source_module = get_source_module(feature_name)
        stage = detect_stage(feature_name)
        causal = detect_causal(feature_name)
        depends_on = detect_depends_on(feature_name)
        consumer_contract = detect_consumer_contract(feature_name)
        
        # Units contract (normalized version of units)
        units_contract = units if units != "unknown" else "unknown"
        
        # Normalization contract (normalized version of normalization)
        normalization_contract = normalization if normalization != "none" else "none"
        
        # Detect if sequence feature (has lookback)
        lookback_bars = None
        if feature_name in SEQUENCE_FEATURES:
            # Sequence features have lookback (from sequence_features.py)
            if "ema20" in feature_name:
                lookback_bars = 20
            elif "ema100" in feature_name:
                lookback_bars = 100
            elif "ema200" in feature_name:
                lookback_bars = 200
            elif "std50" in feature_name or "atr50" in feature_name:
                lookback_bars = 50
            elif "roc20" in feature_name:
                lookback_bars = 20
            elif "roc100" in feature_name:
                lookback_bars = 100
            else:
                lookback_bars = 1  # Default for sequence features
        
        # Key params for lineage hash (minimal, just lookback for now)
        key_params = {"lookback_bars": lookback_bars} if lookback_bars else {}
        
        # Compute lineage hash
        lineage_hash = compute_lineage_hash(
            name=feature_name,
            source_module=source_module,
            timeframe=timeframe,
            units_contract=units_contract,
            normalization_contract=normalization_contract,
            causal=causal,
            key_params=key_params,
        )
        
        entry = {
            "name": feature_name,
            "family": family,
            "timeframe": timeframe,
            "lookback_bars": lookback_bars,
            "units": units,
            "normalization": normalization,
            "safe_in_prebuilt": PREBUILT_SAFE,
            "live_only": feature_name in LIVE_ONLY_FEATURES,
            "leakage_risk_notes": (
                "Causal confirmation delay (R=5 bars) ensures no lookahead leakage" 
                if feature_name.startswith("smc_") else None
            ),  # DEL 4: SMC features are causal with explicit delay
            "consumers": consumers,
            "source_module": source_module,
            "how_computed": f"Computed by {source_module} (see FEATURE_MAP.md for details)",
            # NEW: Feature Control Plane v2.0 contracts
            "stage": stage,
            "causal": causal,
            "depends_on": depends_on,
            "units_contract": units_contract,
            "normalization_contract": normalization_contract,
            "consumer_contract": consumer_contract,
            "lineage_hash": lineage_hash,
        }
        
        manifest_features.append(entry)
        
        if family == "unknown":
            unknown_features.append(feature_name)
    
    # Check for unknown features
    if unknown_features and not allowlist_unknown:
        log.error(f"Found {len(unknown_features)} unknown features:")
        for feat in unknown_features[:10]:
            log.error(f"  - {feat}")
        if len(unknown_features) > 10:
            log.error(f"  ... and {len(unknown_features) - 10} more")
        raise RuntimeError(
            f"[FEATURE_MANIFEST_FAIL] Found {len(unknown_features)} unknown features. "
            f"Add them to FEATURE_FAMILIES or UNKNOWN_FEATURE_CLASSIFICATIONS, or use --allowlist-unknown for initial generation."
        )
    elif unknown_features:
        log.warning(f"Found {len(unknown_features)} unknown features (allowlisted for initial generation)")
    
    # Build manifest
    manifest = {
        "version": "2.0",  # NEW: Bump to v2.0 for Feature Control Plane
        "schema_version": "2.0",  # NEW: Explicit schema version
        "generated_date": pd.Timestamp.now().isoformat(),
        "prebuilt_parquet_path": str(prebuilt_parquet_path),
        "total_features": len(manifest_features),
        "features": manifest_features,
    }
    
    # Write JSON
    log.info(f"Writing JSON manifest to: {output_json_path}")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Write Markdown
    log.info(f"Writing Markdown manifest to: {output_md_path}")
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_content = f"""# Feature Manifest

**Version:** {manifest['version']}  
**Schema Version:** {manifest['schema_version']}  
**Generated:** {manifest['generated_date']}  
**Prebuilt Parquet:** `{prebuilt_parquet_path}`  
**Total Features:** {manifest['total_features']}

## Purpose

This manifest serves as Single Source of Truth (SSoT) for feature definitions, families, contracts, and usage.
It is generated from prebuilt parquet and known feature families.

## Feature Control Plane v2.0

This manifest includes Feature Control Plane contracts:
- **stage**: prebuilt_only | runtime_only | derived_from_model | gate_only | log_only
- **causal**: true/false (no lookahead leakage)
- **depends_on**: List of dependencies (candles, htf_h1, htf_h4, model_output, ...)
- **units_contract**: bps | ratio | flag | zscore | price | int
- **normalization_contract**: none | atr_h1 | z_rolling(10,100) | other
- **consumer_contract**: TRANSFORMER_SEQ | TRANSFORMER_SNAP | XGB | GATE | LOG
- **lineage_hash**: SHA256 hash over (name, source_module, timeframe, units_contract, normalization_contract, causal, key_params)

## Feature Definitions

| Name | Family | Timeframe | Lookback | Units | Normalization | Stage | Causal | Consumer Contract | Lineage Hash |
|------|--------|-----------|----------|-------|---------------|-------|--------|-------------------|--------------|
"""
    
    for feat in manifest_features:
        md_content += f"| `{feat['name']}` | {feat['family']} | {feat['timeframe']} | {feat['lookback_bars'] or 'N/A'} | {feat['units']} | {feat['normalization']} | {feat['stage']} | {feat['causal']} | {feat['consumer_contract']} | `{feat['lineage_hash']}` |\n"
    
    md_content += f"""
## Feature Families

"""
    
    # Group by family
    by_family = {}
    for feat in manifest_features:
        family = feat['family']
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(feat['name'])
    
    for family, features in sorted(by_family.items()):
        md_content += f"### {family.title()} ({len(features)} features)\n\n"
        md_content += ", ".join([f"`{f}`" for f in sorted(features)]) + "\n\n"
    
    md_content += f"""
## Unknown Features

"""
    
    if unknown_features:
        md_content += f"**Warning:** {len(unknown_features)} unknown features found:\n\n"
        for feat in unknown_features:
            md_content += f"- `{feat}`\n"
    else:
        md_content += "No unknown features.\n"
    
    md_content += f"""
## References

- **Feature Map:** `reports/repo_audit/FEATURE_MAP.md`
- **Data Contract:** `docs/DATA_CONTRACT.md`
- **Feature Control Plane:** `docs/FEATURE_CONTROL_PLANE.md`
- **Prebuilt Loader:** `gx1/execution/prebuilt_features_loader.py`
"""
    
    with open(output_md_path, "w") as f:
        f.write(md_content)
    
    log.info("✅ Feature manifest generated successfully")
    
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Generate Feature Manifest (SSoT) - Feature Control Plane v2.0")
    parser.add_argument(
        "--prebuilt-parquet",
        type=Path,
        required=True,
        help="Path to prebuilt features parquet",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("gx1/feature_manifest_v1.json"),
        help="Path to output JSON manifest",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/FEATURE_MANIFEST.md"),
        help="Path to output Markdown manifest",
    )
    parser.add_argument(
        "--allowlist-unknown",
        action="store_true",
        help="Allow unknown features (for initial generation)",
    )
    
    args = parser.parse_args()
    
    if not args.prebuilt_parquet.exists():
        raise FileNotFoundError(f"Prebuilt parquet not found: {args.prebuilt_parquet}")
    
    manifest = generate_feature_manifest(
        prebuilt_parquet_path=args.prebuilt_parquet,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
        allowlist_unknown=args.allowlist_unknown,
    )
    
    log.info(f"✅ Generated manifest with {manifest['total_features']} features")
    log.info(f"   Schema version: {manifest['schema_version']}")
    log.info(f"   JSON: {args.output_json}")
    log.info(f"   Markdown: {args.output_md}")

if __name__ == "__main__":
    main()
