#!/usr/bin/env python3
"""V12 V10 v3 live inference wrapper.

Loads the cemented entry_v10_ctx transformer (V10 v3 / canonical_v3 / 6yr
BS512) and produces per-bar entry-context outputs that feed Entry-IQL v2.

V10 v3 input contract (from MASTER_TRANSFORMER_LOCK.json):
    seq_x     : (B, 96, 37)  → 96-bar M5 history of
                                 [signal_bridge_v1 (7) | price_state (30)]
    snap_x    : (B, 37)      → snapshot of the decision-bar (same 37 fields)
    ctx_cont  : (B, 45)      → ORDERED_CTX_CONT_NAMES_V3
    ctx_cat   : (B, 6)       → ORDERED_CTX_CAT_NAMES_V3 (int64)

Outputs per bar:
    direction_logits   (3,)  → [LONG=0, SHORT=1, FLAT=2] pre-softmax
    direction_probs    (3,)  → softmax of direction_logits
    path_quality       (1,)  → aux regression head
    mfe_first_n        (1,)  → aux regression head (predicted forward MFE)
    tradable_prob      (1,)  → sigmoid(tradable_logit)
    bad_path_prob      (1,)  → sigmoid(bad_path_logit)
    clean_edge_prob    (1,)  → sigmoid(clean_edge_logit)
    survival_prob      (1,)  → sigmoid(survival_logit)

V10 is a transformer — it looks back at the prior 95 M5 bars plus the
current one. Therefore the input DataFrame must have at least 96 rows
of warm history before the decision bar.

Usage:
    v10 = V10LiveInference.load_default()
    # augmented_cv3: from augment_canonical_v3() — has ALL ctx_cont/cat columns
    # bridge: (n, 7) from XGBLiveInference.predict()["signal_bridge_v1"]
    out = v10.predict(augmented_cv3, bridge, end_idx=-1)  # latest bar
    print(out["direction_probs"], out["tradable_prob"])
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

from gx1.contracts.signal_bridge_v3 import (
    PER_BAR_PRICE_STATE_FIELDS_V3,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_CTX_CAT_NAMES_V3,
    SEQ_SIGNAL_DIM_V3,
    CTX_CONT_DIM_V3,
    CTX_CAT_DIM_V3,
    DEFAULT_SEQ_LEN_V3,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer

LOG = logging.getLogger("v12_v10_live")

DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/models/models/entry_v10_ctx/"
    "ENTRY_V10_CTX__RETRAIN_2026Q2_BIDIR_SMC_CANONICAL_V3_6YR_BS512_20260506T120938Z"
)

# V10 v3 cemented config (from MASTER_TRANSFORMER_LOCK.json):
SEQ_LEN = 96
SEQ_DIM = 37   # 7 bridge + 30 price_state
SNAP_DIM = 37
CTX_CONT_DIM = 45
CTX_CAT_DIM = 6


@dataclass
class V10LiveInference:
    bundle_dir: Path
    device: str = "cpu"            # CPU is fast enough for single-bar inference
    _model: EntryV10CtxHybridTransformer | None = field(default=None)

    @classmethod
    def load(cls, bundle_dir: Path = DEFAULT_BUNDLE_DIR,
              device: str = "cpu") -> "V10LiveInference":
        bundle_dir = Path(bundle_dir)
        lock_path = bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
        if not lock_path.exists():
            raise FileNotFoundError(f"V10 lock not found: {lock_path}")
        lock = json.loads(lock_path.read_text())

        # Sanity-check contract dims vs signal_bridge_v3 constants
        if int(lock["seq_input_dim"]) != SEQ_DIM:
            raise RuntimeError(f"V10 bundle seq_input_dim={lock['seq_input_dim']} != {SEQ_DIM}")
        if int(lock["seq_len"]) != SEQ_LEN:
            raise RuntimeError(f"V10 bundle seq_len={lock['seq_len']} != {SEQ_LEN}")
        if int(lock["ctx_cont_dim"]) != CTX_CONT_DIM:
            raise RuntimeError(f"V10 bundle ctx_cont_dim={lock['ctx_cont_dim']} != {CTX_CONT_DIM}")
        if int(lock["ctx_cat_dim"]) != CTX_CAT_DIM:
            raise RuntimeError(f"V10 bundle ctx_cat_dim={lock['ctx_cat_dim']} != {CTX_CAT_DIM}")

        # Build model with bundle's hyperparameters; load weights
        model = EntryV10CtxHybridTransformer(
            seq_input_dim=int(lock["seq_input_dim"]),
            snap_input_dim=int(lock["snap_input_dim"]),
            seq_len=int(lock["seq_len"]),
            ctx_cont_dim=int(lock["ctx_cont_dim"]),
            ctx_cat_dim=int(lock["ctx_cat_dim"]),
        )
        state_dict_path = bundle_dir / lock["model_path_relative"]
        if not state_dict_path.exists():
            raise FileNotFoundError(f"V10 weights not found: {state_dict_path}")
        state_dict = torch.load(str(state_dict_path), map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        LOG.info(f"V10 v3 loaded: {bundle_dir.name}  device={device}")
        return cls(bundle_dir=bundle_dir, device=device, _model=model)

    @classmethod
    def load_default(cls) -> "V10LiveInference":
        return cls.load()

    # ── input matrix building ─────────────────────────────────────────────

    @staticmethod
    def _build_seq_matrix(df: pd.DataFrame, bridge: np.ndarray) -> np.ndarray:
        """Per-bar (n, 37) sequence matrix: 7 signal_bridge + 30 price_state."""
        n = len(df)
        out = np.zeros((n, SEQ_DIM), dtype=np.float32)
        if bridge.shape != (n, 7):
            raise RuntimeError(f"bridge shape {bridge.shape} != ({n}, 7)")
        out[:, 0:7] = bridge.astype(np.float32)
        missing = [c for c in PER_BAR_PRICE_STATE_FIELDS_V3 if c not in df.columns]
        if missing:
            raise RuntimeError(f"missing PER_BAR_PRICE_STATE cols: {missing}")
        for j, fname in enumerate(PER_BAR_PRICE_STATE_FIELDS_V3):
            out[:, 7 + j] = pd.to_numeric(df[fname], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return out

    @staticmethod
    def _build_ctx_cont(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        out = np.zeros((n, CTX_CONT_DIM), dtype=np.float32)
        for j, fname in enumerate(ORDERED_CTX_CONT_NAMES_V3):
            if fname in df.columns:
                out[:, j] = pd.to_numeric(df[fname], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            elif fname == "is_ASIA" and "session_id" in df.columns:
                out[:, j] = (df["session_id"].astype(int) == 0).astype(np.float32).to_numpy()
            else:
                raise RuntimeError(f"missing CTX_CONT col: {fname}")
        return out

    @staticmethod
    def _build_ctx_cat(df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        out = np.zeros((n, CTX_CAT_DIM), dtype=np.int64)
        for j, fname in enumerate(ORDERED_CTX_CAT_NAMES_V3):
            if fname not in df.columns:
                raise RuntimeError(f"missing CTX_CAT col: {fname}")
            out[:, j] = pd.to_numeric(df[fname], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        return out

    # ── prediction ─────────────────────────────────────────────────────

    def predict(
        self,
        augmented_cv3: pd.DataFrame,
        bridge: np.ndarray,
        end_idx: int = -1,
    ) -> dict[str, Any]:
        """V10 forward pass for the M5 bar at `end_idx` of `augmented_cv3`.

        Args:
            augmented_cv3: DataFrame from augment_canonical_v3(). Must have
                at least 96 rows of preceding history before end_idx.
            bridge: (n, 7) signal_bridge_v1 matrix from XGBLiveInference.predict()
                — must be aligned 1:1 with augmented_cv3 rows.
            end_idx: integer index of the decision bar. -1 = latest.

        Returns dict with V10 head outputs (numpy 1-d arrays, batch=1).
        """
        if self._model is None:
            raise RuntimeError("V10 not loaded — call .load() first")

        if end_idx < 0:
            end_idx = len(augmented_cv3) + end_idx
        if end_idx < SEQ_LEN - 1:
            raise RuntimeError(
                f"insufficient history: end_idx={end_idx} needs ≥{SEQ_LEN-1} prior bars"
            )

        # Window: end_idx-95 .. end_idx (inclusive) = 96 bars
        start_idx = end_idx - SEQ_LEN + 1
        window = augmented_cv3.iloc[start_idx: end_idx + 1]
        bridge_window = bridge[start_idx: end_idx + 1]

        # Build input matrices for the 96-bar window
        seq_np = self._build_seq_matrix(window, bridge_window)        # (96, 37)
        # snap = last bar (= window's last row)
        snap_np = seq_np[-1:].copy()                                   # (1, 37)
        ctx_cont_np = self._build_ctx_cont(window.iloc[-1:])          # (1, 45)
        ctx_cat_np = self._build_ctx_cat(window.iloc[-1:])            # (1, 6)

        # Tensors (add batch dim to seq)
        seq_t = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)      # (1, 96, 37)
        snap_t = torch.from_numpy(snap_np).to(self.device)                  # (1, 37)
        ctx_cont_t = torch.from_numpy(ctx_cont_np).to(self.device)          # (1, 45)
        ctx_cat_t = torch.from_numpy(ctx_cat_np).to(self.device)            # (1, 6)

        with torch.no_grad():
            out = self._model(seq_t, snap_t, ctx_cat=ctx_cat_t, ctx_cont=ctx_cont_t)

        dir_logits = out["direction_logits"].cpu().numpy()[0]             # (3,)
        dir_probs = _softmax(dir_logits)
        return {
            "direction_logits": dir_logits,
            "direction_probs": dir_probs,                                  # [P(long), P(short), P(flat)]
            "path_quality": float(out["path_quality"].cpu().numpy()[0, 0]),
            "mfe_first_n": float(out["mfe_first_n"].cpu().numpy()[0, 0]),
            "tradable_prob": float(_sigmoid(out["tradable_logit"].cpu().numpy()[0, 0])),
            "bad_path_prob": float(_sigmoid(out["bad_path_logit"].cpu().numpy()[0, 0])),
            "clean_edge_prob": float(_sigmoid(out["clean_edge_logit"].cpu().numpy()[0, 0])),
            "survival_prob": float(_sigmoid(out["survival_logit"].cpu().numpy()[0, 0])),
            "decision_ts": window.index[-1].isoformat(),
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
