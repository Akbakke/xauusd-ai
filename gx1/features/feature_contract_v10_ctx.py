"""
DEL 2: Canonical feature contract for V10_CTX entry model.

This module defines the single source of truth for V10_CTX feature dimensions.
All code that builds, validates, or uses V10_CTX features must import from here.
"""

# Feature contract version
FEATURE_CONTRACT_VERSION = "v10_ctx_2025_01"

# Base feature counts (from feature_meta.json)
# NOTE: Must match the model's expected dimensions (from train_config.json)
BASE_SEQ_FEATURES = 13
BASE_SNAP_FEATURES = 85

# XGB channels added at runtime
# NOTE: margin_xgb was REMOVED as of 2026-01-25, but XGB_CHANNELS=3 is kept for model compatibility
# The model was trained with 3 XGB channels, so we keep the dimension but only fill 2 channels
# Active channels: seq=[p_long_xgb, uncertainty_score], snap=[p_long_xgb, p_hat_xgb]
XGB_CHANNELS = 3

# Total feature dimensions (base + XGB channels)
TOTAL_SEQ_FEATURES = BASE_SEQ_FEATURES + XGB_CHANNELS  # 16
TOTAL_SNAP_FEATURES = BASE_SNAP_FEATURES + XGB_CHANNELS  # 88

# XGB channel indices
SEQ_XGB_CHANNEL_START = BASE_SEQ_FEATURES  # 13
SEQ_XGB_CHANNEL_END = TOTAL_SEQ_FEATURES  # 16
SNAP_XGB_CHANNEL_START = BASE_SNAP_FEATURES  # 85
SNAP_XGB_CHANNEL_END = TOTAL_SNAP_FEATURES  # 88

# XGB channel names (for debugging)
# NOTE: margin_xgb was REMOVED as of 2026-01-25 based on FULLYEAR 2025 ablation showing it's harmful
SEQ_XGB_CHANNEL_NAMES = ["p_long_xgb", "uncertainty_score"]
SNAP_XGB_CHANNEL_NAMES = ["p_long_xgb", "p_hat_xgb"]


def validate_seq_features(seq_data, context: str = "unknown") -> bool:
    """
    Validate sequence feature dimensions.
    
    Args:
        seq_data: Sequence data array (should be 2D: [seq_len, TOTAL_SEQ_FEATURES])
        context: Context string for error messages
    
    Returns:
        True if valid
    
    Raises:
        RuntimeError: If dimensions don't match contract
    """
    if seq_data.ndim != 2:
        raise RuntimeError(
            f"FEATURE_CONTRACT_MISMATCH: seq_data must be 2D, got {seq_data.ndim}D "
            f"(context: {context})"
        )
    
    actual_seq_dim = seq_data.shape[-1]
    if actual_seq_dim != TOTAL_SEQ_FEATURES:
        raise RuntimeError(
            f"FEATURE_CONTRACT_MISMATCH: seq_input_dim mismatch. "
            f"Contract expects {TOTAL_SEQ_FEATURES} (base={BASE_SEQ_FEATURES} + XGB={XGB_CHANNELS}), "
            f"got {actual_seq_dim} (context: {context})"
        )
    
    return True


def validate_snap_features(snap_data, context: str = "unknown") -> bool:
    """
    Validate snapshot feature dimensions.
    
    Args:
        snap_data: Snapshot data array (should be 1D: [TOTAL_SNAP_FEATURES])
        context: Context string for error messages
    
    Returns:
        True if valid
    
    Raises:
        RuntimeError: If dimensions don't match contract
    """
    if snap_data.ndim != 1:
        raise RuntimeError(
            f"FEATURE_CONTRACT_MISMATCH: snap_data must be 1D, got {snap_data.ndim}D "
            f"(context: {context})"
        )
    
    actual_snap_dim = snap_data.shape[-1]
    if actual_snap_dim != TOTAL_SNAP_FEATURES:
        raise RuntimeError(
            f"FEATURE_CONTRACT_MISMATCH: snap_input_dim mismatch. "
            f"Contract expects {TOTAL_SNAP_FEATURES} (base={BASE_SNAP_FEATURES} + XGB={XGB_CHANNELS}), "
            f"got {actual_snap_dim} (context: {context})"
        )
    
    return True


def get_contract_summary() -> dict:
    """Get contract summary for logging."""
    return {
        "version": FEATURE_CONTRACT_VERSION,
        "base_seq_features": BASE_SEQ_FEATURES,
        "base_snap_features": BASE_SNAP_FEATURES,
        "xgb_channels": XGB_CHANNELS,
        "total_seq_features": TOTAL_SEQ_FEATURES,
        "total_snap_features": TOTAL_SNAP_FEATURES,
        "seq_xgb_indices": f"{SEQ_XGB_CHANNEL_START}-{SEQ_XGB_CHANNEL_END-1}",
        "snap_xgb_indices": f"{SNAP_XGB_CHANNEL_START}-{SNAP_XGB_CHANNEL_END-1}",
    }


# ============================================================================
# XGB CHANNEL MASKING (DEL C - per-channel ablation)
# ============================================================================

# All possible XGB channel names (union of seq and snap)
ALL_XGB_CHANNEL_NAMES = list(set(SEQ_XGB_CHANNEL_NAMES) | set(SNAP_XGB_CHANNEL_NAMES))
# ["p_long_xgb", "margin_xgb", "uncertainty_score", "p_hat_xgb"]

def parse_xgb_channel_mask():
    """
    Parse environment variables for XGB channel masking.
    
    Environment variables:
        GX1_XGB_CHANNEL_MASK: Comma-separated list of channels to DROP (set to 0)
        GX1_XGB_CHANNEL_ONLY: Comma-separated list of channels to KEEP (others set to 0)
    
    Returns:
        dict with:
            - masked_channels: list of channel names to mask (set to 0)
            - kept_channels: list of channel names to keep
            - effective_seq_channels: list of active seq channel names
            - effective_snap_channels: list of active snap channel names
    """
    import os
    
    # REGRESSION GUARD: margin_xgb was REMOVED as of 2026-01-25
    # Hard-fail in truth mode if margin_xgb is referenced
    is_truth_run = os.getenv("GX1_TRUTH_MODE", "0") == "1" or os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
    
    mask_str = os.getenv("GX1_XGB_CHANNEL_MASK", "")
    only_str = os.getenv("GX1_XGB_CHANNEL_ONLY", "")
    
    if mask_str and only_str:
        raise RuntimeError(
            "CHANNEL_MASK_CONFLICT: Cannot set both GX1_XGB_CHANNEL_MASK and GX1_XGB_CHANNEL_ONLY. "
            "Use only one."
        )
    
    if mask_str:
        # DROP specified channels
        masked_channels = [c.strip() for c in mask_str.split(",") if c.strip()]
        # REGRESSION GUARD: Check for removed margin_xgb
        if "margin_xgb" in masked_channels:
            if is_truth_run:
                raise RuntimeError(
                    "MARGIN_XGB_REGRESSION: margin_xgb was REMOVED as of 2026-01-25. "
                    "It cannot be masked because it no longer exists in the pipeline."
                )
            else:
                import warnings
                warnings.warn(
                    "margin_xgb was REMOVED as of 2026-01-25. Ignoring it in GX1_XGB_CHANNEL_MASK.",
                    UserWarning
                )
                masked_channels = [ch for ch in masked_channels if ch != "margin_xgb"]
        # Validate
        for ch in masked_channels:
            if ch not in ALL_XGB_CHANNEL_NAMES:
                raise RuntimeError(
                    f"CHANNEL_MASK_INVALID: Unknown channel '{ch}' in GX1_XGB_CHANNEL_MASK. "
                    f"Valid channels: {ALL_XGB_CHANNEL_NAMES}"
                )
        kept_channels = [ch for ch in ALL_XGB_CHANNEL_NAMES if ch not in masked_channels]
    elif only_str:
        # KEEP only specified channels
        kept_channels = [c.strip() for c in only_str.split(",") if c.strip()]
        # REGRESSION GUARD: Check for removed margin_xgb
        if "margin_xgb" in kept_channels:
            if is_truth_run:
                raise RuntimeError(
                    "MARGIN_XGB_REGRESSION: margin_xgb was REMOVED as of 2026-01-25. "
                    "It cannot be kept because it no longer exists in the pipeline."
                )
            else:
                import warnings
                warnings.warn(
                    "margin_xgb was REMOVED as of 2026-01-25. Ignoring it in GX1_XGB_CHANNEL_ONLY.",
                    UserWarning
                )
                kept_channels = [ch for ch in kept_channels if ch != "margin_xgb"]
        # Validate
        for ch in kept_channels:
            if ch not in ALL_XGB_CHANNEL_NAMES:
                raise RuntimeError(
                    f"CHANNEL_ONLY_INVALID: Unknown channel '{ch}' in GX1_XGB_CHANNEL_ONLY. "
                    f"Valid channels: {ALL_XGB_CHANNEL_NAMES}"
                )
        masked_channels = [ch for ch in ALL_XGB_CHANNEL_NAMES if ch not in kept_channels]
    else:
        # No masking - keep all channels
        masked_channels = []
        kept_channels = ALL_XGB_CHANNEL_NAMES.copy()
    
    # Compute effective channels per injection point
    effective_seq_channels = [ch for ch in SEQ_XGB_CHANNEL_NAMES if ch not in masked_channels]
    effective_snap_channels = [ch for ch in SNAP_XGB_CHANNEL_NAMES if ch not in masked_channels]
    
    return {
        "masked_channels": masked_channels,
        "kept_channels": kept_channels,
        "effective_seq_channels": effective_seq_channels,
        "effective_snap_channels": effective_snap_channels,
        "n_effective_seq": len(effective_seq_channels),
        "n_effective_snap": len(effective_snap_channels),
    }


def get_channel_mask_indices():
    """
    Get indices of channels to mask (set to 0) in seq and snap arrays.
    
    Returns:
        dict with:
            - seq_mask_indices: list of indices in seq array to zero out
            - snap_mask_indices: list of indices in snap array to zero out
            - mask_info: full mask info dict
    """
    mask_info = parse_xgb_channel_mask()
    
    # Build index lists for masking
    seq_mask_indices = []
    for i, ch in enumerate(SEQ_XGB_CHANNEL_NAMES):
        if ch in mask_info["masked_channels"]:
            seq_mask_indices.append(SEQ_XGB_CHANNEL_START + i)
    
    snap_mask_indices = []
    for i, ch in enumerate(SNAP_XGB_CHANNEL_NAMES):
        if ch in mask_info["masked_channels"]:
            snap_mask_indices.append(SNAP_XGB_CHANNEL_START + i)
    
    return {
        "seq_mask_indices": seq_mask_indices,
        "snap_mask_indices": snap_mask_indices,
        "mask_info": mask_info,
    }


def apply_channel_mask(seq_data, snap_data, mask_indices=None):
    """
    Apply channel masking to seq and snap data arrays.
    
    Channels in masked_channels are set to 0.0 (zeroed out).
    
    Args:
        seq_data: 2D numpy array [seq_len, TOTAL_SEQ_FEATURES]
        snap_data: 1D numpy array [TOTAL_SNAP_FEATURES]
        mask_indices: Optional pre-computed mask indices (from get_channel_mask_indices())
    
    Returns:
        (seq_data, snap_data, mask_info) - arrays are modified in-place
    """
    import numpy as np
    
    if mask_indices is None:
        mask_indices = get_channel_mask_indices()
    
    # Zero out masked seq channels
    for idx in mask_indices["seq_mask_indices"]:
        seq_data[:, idx] = 0.0
    
    # Zero out masked snap channels
    for idx in mask_indices["snap_mask_indices"]:
        snap_data[idx] = 0.0
    
    return seq_data, snap_data, mask_indices["mask_info"]
