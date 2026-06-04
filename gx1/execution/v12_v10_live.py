#!/usr/bin/env python3
"""V12 V10 v3 live inference wrapper.

Loads the cemented entry_v10_ctx transformer (V10 v3 / canonical_v3 / 6yr
BS512) and produces per-bar entry-context outputs that feed Entry-IQL v2.

V10 v3 input contract (from MASTER_TRANSFORMER_LOCK.json):
    seq_x     : (B, 96, 41)  → 96-bar M5 history of
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

# 2026-05-19: v3+ cement — V10 multi-TF with 4 aux heads (tf_agreement,
# path_quality_log_var/std, position_size, hold_horizon). State_dict verified
# to contain all 8 head weights. Forward-outcome candidates (50,217) were built
# using this exact bundle, so the downstream chain (Entry-IQL v3+ → Phase 1 →
# per-bar → Exit-IQL v3+) is self-consistent.
# 2026-05-29: default is now contract-driven via gx1_guards.load_decision_artifact("v10_entry").
# A/B override: set GX1_V10_BUNDLE_DIR env var to bypass and use a different bundle.
import os as _os
def _resolve_default_v10_bundle() -> Path:
    env_override = _os.environ.get("GX1_V10_BUNDLE_DIR")
    if env_override:
        return Path(env_override)
    # FAIL-CLOSED (2026-06-03 audit): NO hardcoded fallback. A guard failure MUST
    # propagate — never silently substitute the deleted pre-COSTFIX V3PLUS_v2 bundle.
    # The "fail-soft for smoke tests" branch had no actual test consumer.
    from gx1_guards.artifacts import load_decision_artifact
    return Path(load_decision_artifact("v10_entry"))
DEFAULT_BUNDLE_DIR = _resolve_default_v10_bundle()

# V10 v3 cemented config (from MASTER_TRANSFORMER_LOCK.json):
SEQ_LEN = 96
SEQ_DIM = SEQ_SIGNAL_DIM_V3    # contract-derived: 7 bridge + 34 price_state = 41 (was 37)
SNAP_DIM = SEQ_SIGNAL_DIM_V3
CTX_CONT_DIM = CTX_CONT_DIM_V3  # one-truth: track the contract (69 w/ group-A parity), not a hardcoded 45
# R4 completion (2026-06-04): contract-driven, mirroring CTX_CONT_DIM above. Was a half-applied-R4
# hardcoded `6` — the V10 LIVE loader's split-brain twin of the trainer/bootstrap dims R4 made
# contract-driven. At GX1_REGIME_V4=1 (ctx_cat=5) the old literal rejected the 5-cat regime bundle at
# load() (5 != 6) and _build_ctx_cat allocated (n,6) while iterating 5 names. Cement (flag=0) = 6, unchanged.
CTX_CAT_DIM = CTX_CAT_DIM_V3


@dataclass
class V10LiveInference:
    bundle_dir: Path
    device: str = "cpu"            # CPU is fast enough for single-bar inference
    _model: EntryV10CtxHybridTransformer | None = field(default=None)
    # V12.2 multi-TF: detected from bundle_metadata.json at load time
    _enable_multi_tf: bool = False
    _mtf_seq_lens: dict = field(default_factory=dict)
    _mtf_seq_dims: dict = field(default_factory=dict)

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

        # V12.2: read bundle_metadata.json for multi-TF config (optional —
        # v3 bundles don't have this section, default to v3 mode).
        meta_path = bundle_dir / "bundle_metadata.json"
        multi_tf_cfg = {}
        tf_input_scale_cfg: dict = {}
        _enable_pos_enc = False  # persistent=False buffer → must come from metadata, not state_dict
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                multi_tf_cfg = meta.get("multi_tf", {}) or {}
                _enable_pos_enc = bool(meta.get("enable_pos_enc", False))
                tf_input_scale_cfg = meta.get("tf_input_scale", {}) or {}
            except Exception as exc:
                LOG.warning(f"bundle_metadata.json read failed: {exc} — assuming v3 mode")
        enable_mtf = bool(multi_tf_cfg.get("enabled", False))
        # V12.2 hard-fail: live runtime REQUIRES multi-TF V10 bundles.
        # Single-TF V10 bundles (V11_BAD_PATH_FIXED, CANONICAL_V3_6YR_BS512, etc.)
        # are pre-cement and not validated for live use. Refuse to load them.
        if not enable_mtf:
            raise RuntimeError(
                f"V10 bundle is NOT multi-TF (bundle_metadata.json missing or "
                f"multi_tf.enabled != true). V12.2 live REQUIRES multi-TF. "
                f"Bundle: {bundle_dir}"
            )
        mtf_kwargs = {}
        if enable_mtf:
            # V2 (BASE76): M5 is a 5th multi-TF branch. Must read v2_mode/m5_* or
            # strict load fails on m5_proj/m5_encoder + the 5-branch fuse weights.
            _v2_mode = bool(multi_tf_cfg.get("v2_mode", False))
            mtf_kwargs = dict(
                enable_multi_tf=True,
                m15_seq_dim=int(multi_tf_cfg["m15_seq_dim"]),
                h1_seq_dim=int(multi_tf_cfg["h1_seq_dim"]),
                h4_seq_dim=int(multi_tf_cfg["h4_seq_dim"]),
                d1_seq_dim=int(multi_tf_cfg["d1_seq_dim"]),
                m15_seq_len=int(multi_tf_cfg["m15_seq_len"]),
                h1_seq_len=int(multi_tf_cfg["h1_seq_len"]),
                h4_seq_len=int(multi_tf_cfg["h4_seq_len"]),
                d1_seq_len=int(multi_tf_cfg["d1_seq_len"]),
                enable_multi_tf_m5=_v2_mode,
                m5_seq_dim=int(multi_tf_cfg.get("m5_seq_dim", 0)) if _v2_mode else 0,
                m5_seq_len=int(multi_tf_cfg.get("m5_seq_len", multi_tf_cfg["m15_seq_len"])) if _v2_mode else 96,
            )
            LOG.info(f"V10 bundle is multi-TF (v2_mode={_v2_mode}): H1/H4/D1 dim={multi_tf_cfg['h1_seq_dim']} len={multi_tf_cfg['h1_seq_len']}")

        # Detect V10 v3+ aux heads by presence in state_dict so model is
        # built with matching parameters (else load_state_dict fails strict).
        state_dict_path = bundle_dir / lock["model_path_relative"]
        if not state_dict_path.exists():
            raise FileNotFoundError(f"V10 weights not found: {state_dict_path}")
        state_dict = torch.load(str(state_dict_path), map_location=device, weights_only=False)
        v3plus_kwargs = dict(
            enable_tf_agreement_head="head_tf_agreement.weight" in state_dict,
            enable_path_quality_variance_head="head_path_quality_log_var.weight" in state_dict,
            enable_position_size_head="head_position_size.weight" in state_dict,
            enable_hold_horizon_head="head_hold_horizon.weight" in state_dict,
        )
        if any(v3plus_kwargs.values()):
            LOG.info(f"V10 v3+ aux heads detected in bundle: {[k for k,v in v3plus_kwargs.items() if v]}")
        # 2026-05-21: q_head present iff bundle was distilled from Entry-IQL teacher.
        # Detect via state_dict key. When True, the predict() result includes
        # q_per_action_v1 + advantage_over_skip_v1, exposing Entry-IQL knowledge
        # baked into the V10 backbone via Phase 3a distillation.
        enable_q_head = "q_head.weight" in state_dict
        if enable_q_head:
            LOG.info("V10 q_head (distilled) detected — predict() will emit q_per_action")

        # 2026-05-26: dip-head / forecast-head / cross-TF-attn — detect from
        # state_dict keys so the new dip-aware bundles load strict (else their
        # weights are unexpected_keys → load fails).
        _enable_dip_head = "head_dip.weight" in state_dict
        _enable_forecast_head = "head_forecast.weight" in state_dict
        _enable_cross_tf_attn = any(k.startswith("cross_tf_attn.") or k == "tf_gate_logits" for k in state_dict)
        _enable_timing_head = "head_timing.weight" in state_dict
        _enable_tail_risk_head = "head_tail_risk.weight" in state_dict
        _enable_vol_forecast_head = "head_vol_forecast.weight" in state_dict

        # 2026-06-02: per-TF learnable input scale (V10 v5+). Detect via
        # state_dict; if present, init values come from metadata so Parameter
        # is constructed identically to training time (state_dict then
        # overwrites with learned values). Fallback to user's prior if
        # metadata missing.
        _enable_tf_input_scale = any(
            f"tf_input_scale_{tf}" in state_dict for tf in ("m5", "m15", "h1", "h4", "d1")
        )
        # BIG-9 (2026-06-03): FiLM regime-conditioning — rebuild the module iff the bundle
        # carries regime_film.* weights (real params, so state_dict detection is authoritative).
        _enable_regime_film = any(str(k).startswith("regime_film.") for k in state_dict)
        _tf_inits = (tf_input_scale_cfg.get("init") or {})
        _tf_scale_kwargs = dict(
            enable_tf_input_scale=_enable_tf_input_scale,
            tf_input_scale_init_m5=float(_tf_inits.get("m5", 1.0)),
            tf_input_scale_init_m15=float(_tf_inits.get("m15", 1.0)),
            tf_input_scale_init_h1=float(_tf_inits.get("h1", 0.7)),
            tf_input_scale_init_h4=float(_tf_inits.get("h4", 0.5)),
            tf_input_scale_init_d1=float(_tf_inits.get("d1", 0.3)),
        ) if _enable_tf_input_scale else {}
        if _enable_tf_input_scale:
            _learned = tf_input_scale_cfg.get("learned") or {}
            LOG.info(
                "V10 per-TF input scale enabled — learned: %s",
                {k: round(float(v), 4) for k, v in _learned.items()} if _learned else "(missing in meta)"
            )
        if (_enable_dip_head or _enable_forecast_head or _enable_cross_tf_attn
                or _enable_timing_head or _enable_tail_risk_head or _enable_vol_forecast_head):
            LOG.info(f"V10 dip-aware bundle: dip={_enable_dip_head} forecast={_enable_forecast_head} "
                     f"cross_tf_attn={_enable_cross_tf_attn} timing={_enable_timing_head} "
                     f"tail_risk={_enable_tail_risk_head} vol_forecast={_enable_vol_forecast_head}")

        # Build model with bundle's hyperparameters; load weights
        model = EntryV10CtxHybridTransformer(
            seq_input_dim=int(lock["seq_input_dim"]),
            snap_input_dim=int(lock["snap_input_dim"]),
            seq_len=int(lock["seq_len"]),
            ctx_cont_dim=int(lock["ctx_cont_dim"]),
            ctx_cat_dim=int(lock["ctx_cat_dim"]),
            **mtf_kwargs,
            **v3plus_kwargs,
            enable_q_head=enable_q_head,
            enable_pos_enc=_enable_pos_enc,
            enable_regime_film=_enable_regime_film,
            enable_dip_head=_enable_dip_head,
            enable_forecast_head=_enable_forecast_head,
            enable_cross_tf_attn=_enable_cross_tf_attn,
            enable_timing_head=_enable_timing_head,
            enable_tail_risk_head=_enable_tail_risk_head,
            enable_vol_forecast_head=_enable_vol_forecast_head,
            **_tf_scale_kwargs,
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        LOG.info(f"V10 v3 loaded: {bundle_dir.name}  device={device}  multi_tf={enable_mtf}")
        return cls(
            bundle_dir=bundle_dir, device=device, _model=model,
            _enable_multi_tf=enable_mtf,
            _mtf_seq_lens={k: int(multi_tf_cfg.get(f"{k.lower()}_seq_len", 96))
                            for k in ("M15", "H1", "H4", "D1")} if enable_mtf else {},
            _mtf_seq_dims={k: int(multi_tf_cfg.get(f"{k.lower()}_seq_dim", 0))
                            for k in ("M15", "H1", "H4", "D1")} if enable_mtf else {},
        )

    @classmethod
    def load_default(cls) -> "V10LiveInference":
        return cls.load()

    # ── input matrix building ─────────────────────────────────────────────

    @staticmethod
    def _build_seq_matrix(df: pd.DataFrame, bridge: np.ndarray) -> np.ndarray:
        """Per-bar (n, 41) sequence matrix: 7 signal_bridge + 34 price_state (incl. 4 volume)."""
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
        multi_tf_windows: dict | None = None,
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

        # Build input matrices for the SEQ_LEN-bar window.
        # 2026-06-02 fix (audit MEDIUM-#4): use constant names in comments
        # instead of stale hardcoded numbers. Actual dims: SEQ_DIM is
        # contract-derived (signal_bridge_v3.SEQ_SIGNAL_DIM_V3 = 41 for V3),
        # CTX_CONT_DIM is also contract-derived (105 for V3+, may shift).
        # See module-level constants SEQ_DIM / CTX_CONT_DIM / CTX_CAT_DIM_V3.
        seq_np = self._build_seq_matrix(window, bridge_window)        # (SEQ_LEN, SEQ_DIM)
        snap_np = seq_np[-1:].copy()                                   # (1, SEQ_DIM)
        ctx_cont_np = self._build_ctx_cont(window.iloc[-1:])          # (1, CTX_CONT_DIM)
        ctx_cat_np = self._build_ctx_cat(window.iloc[-1:])            # (1, CTX_CAT_DIM_V3)

        # Tensors (add batch dim to seq)
        seq_t = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)      # (1, SEQ_LEN, SEQ_DIM)
        snap_t = torch.from_numpy(snap_np).to(self.device)                  # (1, SEQ_DIM)
        ctx_cont_t = torch.from_numpy(ctx_cont_np).to(self.device)          # (1, CTX_CONT_DIM)
        ctx_cat_t = torch.from_numpy(ctx_cat_np).to(self.device)            # (1, CTX_CAT_DIM_V3)

        # V12.2: pass multi-TF windows when bundle requires them
        mtf_kwargs = {}
        if self._enable_multi_tf:
            if multi_tf_windows is None:
                raise RuntimeError(
                    "V10 bundle is multi-TF — caller must pass multi_tf_windows "
                    "(via PrebuiltStateLoader.get_multi_tf_windows())"
                )
            required_tfs = ["seq_m15", "seq_h1", "seq_h4", "seq_d1"]
            # V2 (BASE76): M5 is a 5th branch — require seq_m5 only when the model
            # was built with it, else forward() raises SEQ_M5_REQUIRED.
            if getattr(self._model.cfg, "enable_multi_tf_m5", False):
                required_tfs.append("seq_m5")
            for k in required_tfs:
                if k not in multi_tf_windows:
                    raise RuntimeError(f"multi-TF V10 needs {k} but missing from multi_tf_windows")
                arr = multi_tf_windows[k]
                # Add batch dim if needed
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]
                # The loader returns a uniform n_bars (96) for every TF, but the
                # model tapers per-TF (BASE76: h4=48, d1=30). Trim to the model's
                # expected per-TF seq_len (most-recent bars) or forward() raises
                # *_LEN_MISMATCH. No-op when they're equal.
                exp_len = int(getattr(self._model.cfg, f"{k[4:]}_seq_len"))  # seq_h4 -> h4_seq_len
                if arr.shape[1] > exp_len:
                    arr = arr[:, -exp_len:, :]
                mtf_kwargs[k] = torch.from_numpy(arr.astype(np.float32, copy=False)).to(self.device)

        with torch.no_grad():
            out = self._model(seq_t, snap_t, ctx_cat=ctx_cat_t, ctx_cont=ctx_cont_t, **mtf_kwargs)

        dir_logits = out["direction_logits"].cpu().numpy()[0]             # (3,)
        dir_probs = _softmax(dir_logits)
        result = {
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
        # V10 v3+ aux heads (Targets 1-4) — only emitted by bundles trained
        # with enable_*_head=True. Older v_FIXED bundles don't have these
        # keys in the model output, so we add them conditionally and the
        # live pipeline / journal logs whichever are present.
        if "tf_agreement_logit" in out:
            result["tf_agreement_pred"] = float(_sigmoid(out["tf_agreement_logit"].cpu().numpy()[0, 0]))
        if "path_quality_log_var" in out:
            # Clamp same range as training (-5, 5) before exp — un-trained
            # bundles can output extreme log_var values that overflow.
            log_var = float(np.clip(out["path_quality_log_var"].cpu().numpy()[0, 0], -5.0, 5.0))
            result["path_quality_log_var"] = log_var
            result["path_quality_std"] = float(np.exp(log_var / 2.0))
        if "position_size_logit" in out:
            result["position_size_pred"] = float(_sigmoid(out["position_size_logit"].cpu().numpy()[0, 0]))
        if "hold_horizon_logit" in out:
            hh_pred = float(_sigmoid(out["hold_horizon_logit"].cpu().numpy()[0, 0]))
            # Map [0,1] back to bars (max 1440 = 24h).
            result["hold_horizon_pred"] = hh_pred
            result["hold_horizon_bars_pred"] = int(round(hh_pred * 1440))
        # Dip-analysis head (18): dir×K×{dip_p50,dip_p90,recovery_p50}. Surface raw
        # so Entry-IQL / runner can act on the dip-risk profile (don't enter at top
        # of a dip). Forecast head (4): cum future return @ K{1,5,12,24} bps.
        if "dip_pred" in out:
            result["dip_pred_v1"] = out["dip_pred"].cpu().numpy()[0].astype(float).tolist()
        if "forecast_pred" in out:
            result["forecast_pred_v1"] = out["forecast_pred"].cpu().numpy()[0].astype(float).tolist()
        if "timing_pred" in out:
            result["timing_pred_v1"] = out["timing_pred"].cpu().numpy()[0].astype(float).tolist()
        if "tail_risk_pred" in out:
            result["tail_risk_pred_v1"] = out["tail_risk_pred"].cpu().numpy()[0].astype(float).tolist()
        if "vol_forecast_pred" in out:
            result["vol_forecast_pred_v1"] = out["vol_forecast_pred"].cpu().numpy()[0].astype(float).tolist()
        # Distilled q_head (Phase 3a): [q_skip, q_long, q_short] in bps.
        if "q_per_action" in out:
            q = out["q_per_action"].cpu().numpy()[0].astype(float)
            result["q_per_action_v1"] = q.tolist()
            q_skip, q_long, q_short = float(q[0]), float(q[1]), float(q[2])
            result["q_skip_v1"] = q_skip
            result["q_take_long_v1"] = q_long
            result["q_take_short_v1"] = q_short
            result["q_advantage_over_skip_v1"] = float(max(q_long, q_short) - q_skip)
            # Argmax action id matches Entry-IQL ACTION_LABELS_V1.
            result["q_action_id_v1"] = int(np.argmax(q))
        return result


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
