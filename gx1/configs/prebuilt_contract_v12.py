"""
SSoT Canonical Price/ATR Contract for Prebuilt V12

Defines canonical column names for price and ATR in prebuilt parquet files.
Used by compute_payoffs() and training scripts.
"""

import json
from pathlib import Path
from typing import Dict

# Default canonical mapping (fallback if config file missing)
DEFAULT_CANONICAL_PRICE_COL = "mid"
DEFAULT_CANONICAL_ATR_COL = "atr"


def load_canonical_price_atr_mapping(workspace_root: Path = None) -> Dict[str, str]:
    """
    Load SSoT canonical price/ATR column mapping.
    
    Args:
        workspace_root: Optional workspace root path (defaults to auto-detect)
    
    Returns:
        Dict with 'price_col' and 'atr_col' keys
    """
    if workspace_root is None:
        # Auto-detect workspace root (assume we're in gx1/configs/)
        workspace_root = Path(__file__).parent.parent.parent
    
    contract_path = workspace_root / "gx1" / "configs" / "prebuilt_contract_v12.json"
    if contract_path.exists():
        with open(contract_path, "r") as f:
            contract = json.load(f)
        return {
            "price_col": contract.get("canonical_price_col", DEFAULT_CANONICAL_PRICE_COL),
            "atr_col": contract.get("canonical_atr_col", DEFAULT_CANONICAL_ATR_COL),
        }
    # Default fallback (should not happen in production)
    return {
        "price_col": DEFAULT_CANONICAL_PRICE_COL,
        "atr_col": DEFAULT_CANONICAL_ATR_COL,
    }
