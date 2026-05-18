#!/usr/bin/env python3
"""V12 V3 v8 (exit transformer) live inference wrapper.

V3 v8 is the per-M1 exit-signal transformer trained with EXIT_IO_V6_CTX_V3CANONICAL_M1L512
(91 features × 512 M1-bars window). For each M1 bar during an open trade it
produces 4 outputs that the Exit-IQL V12.1 state vector expects:

    v3_v8_should_exit_prob       sigmoid(main head)         primary exit signal
    v3_v8_profit_protect_prob    sigmoid(profit-protect)
    v3_v8_family_argmax          argmax(4-class family head)
    v3_v8_family_logit_max       max(family logits)

Input layout per the EXIT_IO_V6 contract (91 features):
    0-6   :  XGB signal_bridge_v1 (7-dim)              — from XGB v5 at M5 bucket
    7-89  :  Mix of canonical features + 19 trade-state slots that get OVERLAID
             for in-trade bars. Pre-trade bars (the 511 historical bars before
             this trade opened) have the 19 trade-state slots = 0 by training
             convention.

This module:
  1. Loads V3 v8 from bundle (matches transformer_config.json)
  2. Builds the (1, 512, 91) input matrix per inference:
     - Reads BASE34 prebuilt (M1 cadence) for canonical + ctx columns
     - Runs XGB v5 on the unique M5 buckets in window → signal_bridge per bar
     - Applies trade-state overlay if a TradeState is provided
  3. Forward passes → 4 outputs

Performance target: < 300 ms per inference. The transformer forward on
(1, 512, 91) takes ~50-100ms; XGB on ~103 M5 buckets ~50ms; assembly ~30ms.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.exits.contracts.exit_io_v6_ctx_v3canonical_m1l512 import (
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES as V6_FEATURES,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT as V6_FEATURE_COUNT,
)
from gx1.policy.exit_transformer_v0 import ExitTransformerV0

LOG = logging.getLogger("v12_v3_live")

# V12.2 cement (2026-05-15): default = V3 v9 multi-TF.
# Trained val_acc 96.31% (+5.87pp over V3 v8 baseline 90.44%).
DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/models/exit_transformer_v0/"
    "EXIT_V9_MULTI_TF_LR5E4_SCALE025_20260513T223544Z"
)
WINDOW_LEN = 512   # 512 M1 bars (8.5 hours)

# Indices of the 7 XGB-bridge features (positions 0..6 in the V6 contract)
XGB_BRIDGE_NAMES = V6_FEATURES[:7]   # ['p_long','p_short','p_flat','p_hat','uncertainty_score','margin_top1_top2','entropy']

# 19 features that get overlaid with trade-state values when in-trade.
# These are 0 by default (pre-trade convention from training).
TRADE_STATE_FEATURE_NAMES = [
    "p_long_entry", "p_hat_entry", "uncertainty_entry", "entropy_entry", "margin_entry",
    "pnl_bps_now", "mfe_bps", "mae_bps", "dd_from_mfe_bps",
    "distance_from_peak_mfe_bps", "bars_held", "time_since_mfe_bars",
    "mfe_decay_rate", "pnl_velocity", "pnl_acceleration",
    "rolling_slope_since_entry", "atr_bps_now",
    "giveback_ratio", "giveback_acceleration",
]
TRADE_STATE_INDICES = [V6_FEATURES.index(n) for n in TRADE_STATE_FEATURE_NAMES]


@dataclass
class V3LiveInference:
    bundle_dir: Path
    device: str = "cpu"
    _model: ExitTransformerV0 | None = field(default=None)
    # V12.2 multi-TF: detected from transformer_config.json at load time
    _enable_multi_tf: bool = False
    _mtf_seq_dims: dict = field(default_factory=dict)
    _mtf_seq_lens: dict = field(default_factory=dict)

    @classmethod
    def load(cls, bundle_dir: Path = DEFAULT_BUNDLE_DIR, device: str = "cpu") -> "V3LiveInference":
        bundle_dir = Path(bundle_dir)
        cfg_path = bundle_dir / "transformer_config.json"
        state_path = bundle_dir / "exit_transformer_v0.pt"
        if not cfg_path.exists():
            raise FileNotFoundError(f"V3 v8 config missing: {cfg_path}")
        if not state_path.exists():
            raise FileNotFoundError(f"V3 v8 weights missing: {state_path}")

        cfg = json.loads(cfg_path.read_text())
        if int(cfg["input_dim"]) != V6_FEATURE_COUNT:
            raise RuntimeError(f"V3 v8 input_dim={cfg['input_dim']} != V6 {V6_FEATURE_COUNT}")
        if int(cfg["window_len"]) != WINDOW_LEN:
            raise RuntimeError(f"V3 v8 window_len={cfg['window_len']} != {WINDOW_LEN}")
        if cfg.get("exit_ml_io_version") != "EXIT_IO_V6_CTX_V3CANONICAL_M1L512":
            raise RuntimeError(f"V3 v8 io_version mismatch: {cfg.get('exit_ml_io_version')}")

        # V12.2: detect multi-TF mode from config (v8 bundles default enabled=False).
        mtf_cfg = cfg.get("multi_tf", {}) or {}
        enable_mtf = bool(mtf_cfg.get("enabled", False))
        # V12.2 hard-fail: live runtime REQUIRES multi-TF V3 bundles.
        # Single-TF V3 bundles (EXIT_V6/V7/V8_DISK_*) are pre-cement and not validated.
        if not enable_mtf:
            raise RuntimeError(
                f"V3 bundle is NOT multi-TF (transformer_config.json missing or "
                f"multi_tf.enabled != true). V12.2 live REQUIRES multi-TF. "
                f"Bundle: {bundle_dir}"
            )
        mtf_kwargs = {}
        if enable_mtf:
            mtf_kwargs = dict(
                enable_multi_tf=True,
                m5_seq_dim=int(mtf_cfg["m5_seq_dim"]),
                m15_seq_dim=int(mtf_cfg["m15_seq_dim"]),
                h1_seq_dim=int(mtf_cfg["h1_seq_dim"]),
                h4_seq_dim=int(mtf_cfg["h4_seq_dim"]),
                d1_seq_dim=int(mtf_cfg["d1_seq_dim"]),
                m5_seq_len=int(mtf_cfg["m5_seq_len"]),
                m15_seq_len=int(mtf_cfg["m15_seq_len"]),
                h1_seq_len=int(mtf_cfg["h1_seq_len"]),
                h4_seq_len=int(mtf_cfg["h4_seq_len"]),
                d1_seq_len=int(mtf_cfg["d1_seq_len"]),
            )
            LOG.info(f"V3 bundle is multi-TF: M5/M15/H1/H4/D1 active")

        model = ExitTransformerV0(
            input_dim=cfg["input_dim"], window_len=cfg["window_len"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            dropout=cfg.get("dropout", 0.1),
            **mtf_kwargs,
        )
        state_dict = torch.load(state_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        LOG.info(f"V3 v9 (multi-TF) loaded: {bundle_dir.name}  device={device}  "
                  f"input_dim={cfg['input_dim']}  window_len={cfg['window_len']}  multi_tf={enable_mtf}")
        return cls(
            bundle_dir=bundle_dir, device=device, _model=model,
            _enable_multi_tf=enable_mtf,
            _mtf_seq_dims={k: int(mtf_cfg.get(f"{k.lower()}_seq_dim", 0))
                            for k in ("M5", "M15", "H1", "H4", "D1")} if enable_mtf else {},
            _mtf_seq_lens={k: int(mtf_cfg.get(f"{k.lower()}_seq_len", 96))
                            for k in ("M5", "M15", "H1", "H4", "D1")} if enable_mtf else {},
        )

    @classmethod
    def load_default(cls) -> "V3LiveInference":
        return cls.load()

    # ── feature-matrix construction ─────────────────────────────────────

    def build_window(
        self,
        end_ts: pd.Timestamp,
        base34_prebuilt: pd.DataFrame,
        xgb_inferer,                                  # XGBLiveInference
        canonical_v3_window: pd.DataFrame | None = None,
        trade_overlay: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Assemble the (512, 91) input matrix for V3 v8.

        Args:
            end_ts: timestamp of the decision M1 bar (the LAST row in the window).
            base34_prebuilt: M1-cadence prebuilt with 56 columns (BASE34_CTX16CAT6).
                Must cover at least [end_ts - 511 min, end_ts].
            xgb_inferer: XGBLiveInference instance for computing signal_bridge per
                unique M5 bucket in window.
            canonical_v3_window: optional M5-cadence frame (subset of canonical_v3
                prebuilt) covering the same time range. Used for canonical features
                not in BASE34. If None, derived from base34's columns.
            trade_overlay: dict of 19 (window-len,) arrays with trade-state values
                for in-trade portion of the window. If None, those features stay 0.

        Returns: (512, 91) float32 numpy array.
        """
        # Get the 512 M1 bars ending at end_ts
        end_bucket = end_ts.floor("min")
        win = base34_prebuilt.loc[:end_bucket].tail(WINDOW_LEN)
        if len(win) < WINDOW_LEN:
            raise RuntimeError(f"insufficient M1 history for V3 window: {len(win)} < {WINDOW_LEN}")

        # Initialize matrix
        mat = np.zeros((WINDOW_LEN, V6_FEATURE_COUNT), dtype=np.float32)

        # Fill features 7..90 from base34 / canonical_v3 (skip indices 0-6 = XGB)
        for j, fname in enumerate(V6_FEATURES):
            if j < 7:
                continue  # XGB-bridge, filled below
            if fname in win.columns:
                mat[:, j] = pd.to_numeric(win[fname], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            elif canonical_v3_window is not None and fname in canonical_v3_window.columns:
                # Lazy-join from M5 (each M1 inherits its M5 bucket's value)
                m1_buckets = win.index.floor("5min")
                aligned = canonical_v3_window[fname].reindex(m1_buckets, method="ffill")
                mat[:, j] = pd.to_numeric(aligned, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            # else: feature missing — leave at 0

        # Fill XGB signal_bridge (positions 0-6) by running XGB on unique M5 buckets
        m1_buckets = pd.DatetimeIndex(win.index.floor("5min"))
        unique_buckets = m1_buckets.unique()
        if canonical_v3_window is None:
            raise RuntimeError("canonical_v3_window required for XGB inference")

        m5_input = canonical_v3_window.loc[canonical_v3_window.index.isin(unique_buckets)]
        if m5_input.empty:
            LOG.warning("no M5 buckets available for XGB inference — XGB bridge will be 0")
        else:
            xgb_out = xgb_inferer.predict(m5_input)
            bridge_by_ts = dict(zip(m5_input.index, xgb_out["signal_bridge_v1"]))
            # Map each M1 to its M5 bucket's bridge
            for i, bucket in enumerate(m1_buckets):
                if bucket in bridge_by_ts:
                    mat[i, 0:7] = bridge_by_ts[bucket]

        # Apply trade overlay if provided
        if trade_overlay is not None:
            for j, fname in enumerate(TRADE_STATE_FEATURE_NAMES):
                if fname in trade_overlay:
                    col_idx = TRADE_STATE_INDICES[j]
                    overlay_arr = np.asarray(trade_overlay[fname], dtype=np.float32)
                    n_overlay = min(len(overlay_arr), WINDOW_LEN)
                    # Overlay at the END of the window (in-trade bars are the latest)
                    mat[-n_overlay:, col_idx] = overlay_arr[-n_overlay:]

        return mat

    # ── inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_from_matrix(self, window: np.ndarray,
                              multi_tf_windows: dict | None = None) -> dict[str, Any]:
        """Forward V3 v8 on a (512, 91) input matrix. Returns 4 head outputs."""
        if self._model is None:
            raise RuntimeError("V3 v8 not loaded — call .load() first")
        if window.shape != (WINDOW_LEN, V6_FEATURE_COUNT):
            raise RuntimeError(f"window shape {window.shape} != ({WINDOW_LEN}, {V6_FEATURE_COUNT})")

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)   # (1, 512, 91)

        # V12.2: build multi-TF kwargs if model needs them
        mtf_kwargs = {}
        if self._enable_multi_tf:
            if multi_tf_windows is None:
                raise RuntimeError(
                    "V3 bundle is multi-TF — caller must pass multi_tf_windows "
                    "(via PrebuiltStateLoader.get_multi_tf_windows())"
                )
            for k in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1"):
                if k not in multi_tf_windows:
                    raise RuntimeError(f"multi-TF V3 needs {k} but missing from multi_tf_windows")
                arr = multi_tf_windows[k]
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]
                mtf_kwargs[k] = torch.from_numpy(arr.astype(np.float32, copy=False)).to(self.device)

        main_logit = self._model.forward_logits(x, **mtf_kwargs).item()
        pp_logit = self._model.forward_profit_protect_logits(x, **mtf_kwargs).item()
        family_logits = self._model.forward_family_logits(x, **mtf_kwargs).cpu().numpy().flatten()
        return {
            "v3_v8_should_exit_prob": float(_sigmoid(main_logit)),
            "v3_v8_profit_protect_prob": float(_sigmoid(pp_logit)),
            "v3_v8_family_argmax": int(np.argmax(family_logits)),
            "v3_v8_family_logit_max": float(np.max(family_logits)),
        }

    def predict(
        self,
        end_ts: pd.Timestamp,
        base34_prebuilt: pd.DataFrame,
        canonical_v3_window: pd.DataFrame,
        xgb_inferer,
        trade_overlay: dict[str, np.ndarray] | None = None,
        multi_tf_windows: dict | None = None,
    ) -> dict[str, Any]:
        """One-shot: build window + forward V3 v8 (or v9 with multi-TF)."""
        window = self.build_window(
            end_ts, base34_prebuilt, xgb_inferer,
            canonical_v3_window=canonical_v3_window,
            trade_overlay=trade_overlay,
        )
        return self.predict_from_matrix(window, multi_tf_windows=multi_tf_windows)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
