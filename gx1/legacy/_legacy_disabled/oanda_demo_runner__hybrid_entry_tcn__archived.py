# ARCHIVED: HYBRID_ENTRY/TCN entry path removed from runtime; non-canonical in CANONICAL_TRUTH_SIGNAL_ONLY_V1.

from __future__ import annotations

import json as jsonlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def init_hybrid_entry_tcn(self) -> None:
    """Original HYBRID_ENTRY init (TCN load)."""
    hybrid_entry_cfg = self.policy.get("hybrid_entry", {})
    hybrid_entry_enabled = hybrid_entry_cfg.get("enabled", False)

    if hybrid_entry_enabled:
        # Load TCN entry model
        tcn_cfg = self.policy.get("tcn", {})
        tcn_model_path = Path(tcn_cfg.get("model_path", "gx1/models/seq/entry_fast_tcn_2025Q3.pt"))
        tcn_meta_path = Path(tcn_cfg.get("meta_path", "gx1/models/seq/entry_fast_tcn_2025Q3.meta.json"))

        if not tcn_model_path.exists():
            log.warning("[HYBRID_ENTRY] TCN model not found: %s. Hybrid entry disabled.", tcn_model_path)
            self.entry_tcn_model = None
            self.entry_tcn_meta = None
            self.entry_tcn_scaler = None
            self.entry_tcn_feats = None
            self.entry_tcn_lookback = None
        elif not tcn_meta_path.exists():
            log.warning("[HYBRID_ENTRY] TCN meta not found: %s. Hybrid entry disabled.", tcn_meta_path)
            self.entry_tcn_model = None
            self.entry_tcn_meta = None
            self.entry_tcn_scaler = None
            self.entry_tcn_feats = None
            self.entry_tcn_lookback = None
        else:
            try:
                import torch
                from gx1.seq.model_tcn import TempCNN, TempCNNConfig  # type: ignore[reportMissingImports]
                from sklearn.preprocessing import RobustScaler

                # Load meta
                with open(tcn_meta_path, "r") as f:
                    meta = jsonlib.load(f)

                self.entry_tcn_feats = meta["feats"]
                self.entry_tcn_lookback = meta.get("lookback", 864)

                # Reconstruct scaler
                scaler_dict = meta["scaler"]
                scaler = RobustScaler()
                if scaler_dict.get("center_") is not None:
                    scaler.center_ = np.array(scaler_dict["center_"])
                if scaler_dict.get("scale_") is not None:
                    scaler.scale_ = np.array(scaler_dict["scale_"])
                scaler.quantile_range = tuple(scaler_dict.get("quantile_range", [25, 75]))
                scaler.with_centering = scaler_dict.get("with_centering", True)
                scaler.with_scaling = scaler_dict.get("with_scaling", True)
                self.entry_tcn_scaler = scaler

                # Build model config
                arch_cfg = meta.get("seq_cfg", {})
                tcn_config = TempCNNConfig(
                    in_features=len(self.entry_tcn_feats),
                    hidden=arch_cfg.get("hidden", 64),
                    depth=arch_cfg.get("depth", 3),
                    kernel=arch_cfg.get("kernel", 3),
                    dilations=tuple(arch_cfg.get("dilations", (1, 2, 4))),
                    dropout=arch_cfg.get("dropout", 0.10),
                    head_dropout=arch_cfg.get("head_dropout", 0.10),
                    pool=arch_cfg.get("pool", "avg"),
                    attn_heads=arch_cfg.get("attn_heads", 4),
                )

                # Load model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.entry_tcn_model = TempCNN(tcn_config)

                # Load state dict (handle both direct state_dict and wrapped dict)
                state_dict = torch.load(tcn_model_path, map_location=device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    # Wrapped dict with "state_dict" key
                    state_dict_to_load = state_dict["state_dict"]
                else:
                    # Direct state_dict
                    state_dict_to_load = state_dict

                self.entry_tcn_model.load_state_dict(state_dict_to_load, strict=False)
                self.entry_tcn_model.to(device)
                self.entry_tcn_model.eval()
                self.entry_tcn_device = device
                self.entry_tcn_meta = meta

                log.info(
                    "[TCN ENTRY LOADED] model_path=%s meta_path=%s lookback=%d features=%d device=%s",
                    tcn_model_path,
                    tcn_meta_path,
                    self.entry_tcn_lookback,
                    len(self.entry_tcn_feats),
                    device,
                )
            except Exception as e:
                log.error("[HYBRID_ENTRY] Failed to load TCN model: %s. Hybrid entry disabled.", e, exc_info=True)
                self.entry_tcn_model = None
                self.entry_tcn_meta = None
                self.entry_tcn_scaler = None
                self.entry_tcn_feats = None
                self.entry_tcn_lookback = None
    else:
        log.info("[HYBRID_ENTRY] Hybrid entry disabled in policy")
        self.entry_tcn_model = None
        self.entry_tcn_meta = None
        self.entry_tcn_scaler = None
        self.entry_tcn_feats = None
        self.entry_tcn_lookback = None


def predict_entry_tcn(self, candles: pd.DataFrame, entry_bundle) -> Optional[Any]:
    """Original _predict_entry_tcn implementation."""
    if self.entry_tcn_model is None or self.entry_tcn_feats is None:
        log.warning("[TCN_PREDICT] Model or features not loaded - model=%s features=%s", self.entry_tcn_model is not None, self.entry_tcn_feats is not None)
        return None
    try:
        import torch
        from gx1.seq.sequence_features import build_sequence_features
        from gx1.seq.dataset import SeqWindowDataset  # type: ignore[reportMissingImports]
        from gx1.features.basic_v1 import build_basic_v1
        from gx1.utils.ts_utils import ensure_ts_column
        from gx1.execution.live_features import infer_session_tag
        from gx1.tuning.feature_manifest import align_features  # type: ignore[reportMissingImports]
        from gx1.tuning.feature_manifest import load_manifest  # type: ignore[reportMissingImports]
        # (Original body omitted for brevity in archive)
    except Exception as e:
        log.error("[TCN_PREDICT] TCN prediction error: %s", e, exc_info=True)
        return None


if __name__ == "__main__":
    print("ARCHIVED HYBRID_ENTRY/TCN path - non-executable reference")
