# gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS


def _assert_shape(name: str, t: torch.Tensor, nd: int) -> None:
    if not isinstance(t, torch.Tensor):
        raise RuntimeError(f"TYPE_MISMATCH: {name} is not a torch.Tensor (got {type(t)})")
    if t.dim() != nd:
        raise RuntimeError(f"SHAPE_MISMATCH: {name}.dim={t.dim()} expected={nd} shape={tuple(t.shape)}")


def _assert_finite(name: str, t: torch.Tensor) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NONFINITE: {name} contains NaN/Inf")


_ANCHOR_FIELDS = ("p_long", "p_short", "p_flat")
_ANCHOR_IDX = tuple(ORDERED_FIELDS.index(f) for f in _ANCHOR_FIELDS)

# ── Dip-analysis head layout (V10 entry) — risk-aware, multi-horizon, distributional.
# Output index = flatten over (direction, horizon, target) in this order. The
# trainer's pinball loss and any consumer MUST use this same layout (documented,
# not magic numbers). dip_p50/p90 = conditional quantiles of mae_before_mfe (dip
# depth if taking now); recovery_p50 = median mfe-after-dip. See memory
# project_gx1_dip_aware_entry_timing.
DIP_DIRECTIONS = ("long", "short")
DIP_HORIZONS = (12, 48, 96)                       # M5 bars
DIP_TARGETS = ("dip_p50", "dip_p90", "recovery_p50")
DIP_HEAD_DIM = len(DIP_DIRECTIONS) * len(DIP_HORIZONS) * len(DIP_TARGETS)  # = 18

# ── Self-supervised forecast head (#5) — predict cumulative future return (bps)
# at several M5 horizons. Self-supervised (target = realized future return, no
# labels). Forces the representation to capture forward price dynamics → richer
# dip/momentum encoding that all heads + the IQL benefit from.
FORECAST_HORIZONS = (1, 5, 12, 24)                # M5 bars ahead
FORECAST_HEAD_DIM = len(FORECAST_HORIZONS)        # = 4

# ── Dip-timing head (2026-05-26) — predicts WHEN, not just how-deep. Completes
# "don't enter at the TOP of a dip": dip_bottom_frac = bar-of-dip-bottom / K and
# time_to_mfe_frac = bar-of-favorable-peak / K, both ∈[0,1]. Layout flattens over
# (direction, horizon, target) in this exact order. Targets are builder columns
# y_dip_bottom_frac_{dir}_K{K} / y_time_to_mfe_frac_{dir}_K{K}.
TIMING_DIRECTIONS = ("long", "short")
TIMING_HORIZONS = (12, 48, 96)
TIMING_TARGETS = ("dip_bottom_frac", "time_to_mfe_frac")
TIMING_HEAD_DIM = len(TIMING_DIRECTIONS) * len(TIMING_HORIZONS) * len(TIMING_TARGETS)  # = 12

# ── Tail-risk head (2026-05-26) — p90 (pinball q=0.9) of the WORST adverse
# excursion over the full K horizon (regardless of mfe ordering) → stop placement
# / risk sizing. Layout flattens over (direction, horizon). Target column
# y_tail_mae_{dir}_K{K}.
TAIL_RISK_DIRECTIONS = ("long", "short")
TAIL_RISK_HORIZONS = (12, 48, 96)
TAIL_RISK_QUANTILE = 0.9
TAIL_RISK_HEAD_DIM = len(TAIL_RISK_DIRECTIONS) * len(TAIL_RISK_HORIZONS)  # = 6

# ── Volatility-forecast head (2026-05-26) — realized forward vol (std of 1-bar
# returns, bps) over K bars; direction-agnostic. Feeds sizing + regime awareness.
# Target column y_vol_fwd_K{K}.
VOL_FORECAST_HORIZONS = (12, 48, 96)
VOL_FORECAST_HEAD_DIM = len(VOL_FORECAST_HORIZONS)  # = 3


@dataclass(frozen=True)
class CtxModelConfig:
    seq_input_dim: int
    snap_input_dim: int
    seq_len: int
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: Optional[int] = None
    dropout: float = 0.05
    ctx_cat_dim: int = 6
    ctx_cont_dim: int = 6
    # simple, robust embedding: one shared vocab for all ctx_cat slots
    ctx_cat_vocab: int = 1024
    ctx_cat_emb_dim: int = 8
    # Keep ctx as correction, not primary driver
    ctx_cat_scale: float = 0.25
    ctx_cont_scale: float = 0.25
    # Anchored entry
    residual_scale: float = 0.35
    anchor_eps: float = 1e-6
    # XAU direction repair (2026-07-10): optional learned anchor gate +
    # hierarchical trade/side/utility heads. Defaults OFF so existing bundles
    # keep strict-load and identical forward behaviour.
    enable_anchor_gate: bool = False
    anchor_gate_init: float = 1.0
    enable_hierarchical_entry_heads: bool = False
    enable_hierarchical_direction_composition: bool = False
    hierarchical_composition_residual_logit_cap: float = 0.0
    hierarchical_composition_residual_side_neutral: bool = False
    hierarchical_composition_public_flat_from_trade: bool = False
    hierarchical_public_direction_composition: str = "logprob"
    enable_hierarchical_public_trade_head: bool = False
    enable_hierarchical_public_side_head: bool = False
    enable_hierarchical_ctx_prior_adapter: bool = False
    hierarchical_ctx_prior_adapter_scale: float = 0.0
    enable_hierarchical_ctx_direction_calibration: bool = False
    hierarchical_ctx_direction_calibration_scale: float = 0.0
    hierarchical_ctx_direction_calibration_cap: float = 0.0
    enable_side_validity_head: bool = False
    enable_trendline_rail_head: bool = False
    trendline_rail_output_dim: int = 4
    # ── Multi-TF extension (V12.2) ────────────────────────────────────
    # When disabled (default), model behaves identically to v3: no extra
    # layers created, no extra parameters in state_dict, no extra compute.
    # When enabled, model adds M15/H1/H4/D1 sequence encoders + second-stage
    # fusion that combines v3-fused vector with multi-TF pool.
    # Bundle-stored config decides at load-time whether to enable.
    enable_multi_tf: bool = False
    m15_seq_dim: int = 0         # 0 → branch disabled even if enable_multi_tf=True
    h1_seq_dim: int = 0
    h4_seq_dim: int = 0
    d1_seq_dim: int = 0
    m15_seq_len: int = 96        # ~24 hours at M15 cadence
    h1_seq_len: int = 96         # ~4 days at H1 cadence
    h4_seq_len: int = 96         # ~16 days at H4 cadence
    d1_seq_len: int = 96         # ~3 months at D1 cadence
    multi_tf_num_layers: int = 2 # smaller encoders per TF (lower TF count → less compute)
    multi_tf_scale: float = 0.5  # cap multi-TF contribution to final fusion
    # ── Cross-TF attention fusion (2026-05-26) ────────────────────────
    # When enabled, replaces the static concat→MLP multi-TF fusion with: treat
    # the N per-TF pools as a sequence of N tokens → cross-TF attention (each TF
    # attends to the others) → learnable per-TF gate (softmax weights) → output.
    # Lets the model learn WHICH TF matters WHEN (M5 for timing, D1 for regime),
    # regime-dependent, instead of a fixed mix. Zero-init output → starts as a
    # no-op correction (stable). State-dict matches when off (gated, default OFF).
    enable_cross_tf_attn: bool = False
    # ── 2026-06-02: per-TF learnable input scaling ────────────────────
    # V10 cement learned essentially equal weight per TF (cross_tf_attn gates
    # barely moved from zero-init). To break that uniformity, scale each TF
    # input by a LEARNABLE scalar initialized with a user-specified prior
    # that down-weights macro (D1, H4) relative to micro (M5, M15).
    # Default off so older bundles load with strict=True.
    enable_tf_input_scale: bool = False
    tf_input_scale_init_m5: float = 1.0
    tf_input_scale_init_m15: float = 1.0
    tf_input_scale_init_h1: float = 0.7
    tf_input_scale_init_h4: float = 0.5
    tf_input_scale_init_d1: float = 0.3
    # ── V2 extension (2026-05-22): M5 as multi-TF input ───────────────
    # When enabled (V10 v4+), V10 receives BOTH:
    #   - seq_x (M5 raw price-state, 37 features × 96 bars)   ← existing path
    #   - seq_m5 (M5 V2 features 25 × 96 bars from multi-TF cache) ← NEW
    # These are different feature representations of the same M5 history.
    # Fuse layer expands from 4×d_model → 5×d_model input.
    # Defaults OFF so V1 v3+ bundles load with strict=True.
    enable_multi_tf_m5: bool = False
    m5_seq_dim: int = 0          # 0 → m5 multi-TF branch disabled
    m5_seq_len: int = 96         # ~8 hours at M5 cadence
    # ── Distillation Q-head (V13 prep) ────────────────────────────────
    # When enabled, adds nn.Linear(d_model, 3) producing q_per_action that
    # mirrors Entry-IQL Q-values (skip / long / short). Zero-initialised so
    # an empty (un-distilled) head outputs zeros — identical baseline output.
    # State-dict matches v_FIXED exactly when disabled (no params added).
    enable_q_head: bool = False
    # ── Multi-TF → direction head (2026-06-06, forceful MTF→dir) ───────
    # Dedicated nn.Linear(d_model,3) reads the GATED cross-TF repr (`pooled`,
    # NON-zero-init — NOT mtf_correction which is damped to 0 by zero-init
    # cross_tf_out) + an aux CE forces the 5 multi-TF streams to predict
    # direction. Adds mtf_dir_scale*logits to direction_logits. Default OFF
    # → no params added → strict-load unchanged.
    enable_mtf_direction_head: bool = False
    mtf_dir_scale_init: float = 0.2
    # ── TF-agreement head (V10 v3+ Target 1) ──────────────────────────
    # When enabled, adds nn.Linear(d_model, 1) producing tf_agreement_pred
    # in [0,1] (after sigmoid). Trained against y_tf_agreement_score label
    # computed from multi-TF trend-sign agreement with D1. Live inference
    # exposes it so the runner can gate entries on regime-conflict signal.
    # State-dict matches v_FIXED exactly when disabled (no params added).
    # Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 1.
    enable_tf_agreement_head: bool = False
    # ── Heteroscedastic path_quality (V10 v3+ Target 2) ───────────────
    # When enabled, adds nn.Linear(d_model, 1) producing log-variance for
    # path_quality. Used with Gaussian NLL loss instead of MSE so model
    # naturally learns uncertainty: high variance on regime-conflict
    # samples, low variance on clean setups. Live runner gets mean +
    # variance so it can gate on signal-to-noise instead of raw mean.
    # State-dict matches v_FIXED exactly when disabled (no params added).
    # Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 2.
    enable_path_quality_variance_head: bool = False
    # ── Position-size head (V10 v3+ Target 3) ─────────────────────────
    # When enabled, adds nn.Linear(d_model, 1) producing position_size_pred
    # ∈ [0,1] (after sigmoid). Trained against (mfe + mae) / atr label
    # (realized signed edge in ATR units), so predictions map to:
    #   < 0.3  → 0.25× base units (very risky)
    #   0.3-0.5 → 0.5×
    #   0.5-0.7 → 1.0× (default)
    #   > 0.7  → 2.0× (high conviction)
    # Live runner converts prediction to a units-multiplier at order time.
    # State-dict matches v_FIXED exactly when disabled (no params added).
    # Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 3.
    enable_position_size_head: bool = False
    # ── Hold-horizon head (V10 v3+ Target 4) ──────────────────────────
    # When enabled, adds nn.Linear(d_model, 1) producing expected_hold_pred
    # ∈ [0,1] (after sigmoid). Trained against realized_hold_bars / 1440
    # (max 24h horizon normalized). Live runner uses to set per-trade
    # max_bars_in_trade and Strategy F overlay reads it to know when to
    # cut early in stale trades. State-dict matches v_FIXED exactly when
    # disabled. Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 4.
    enable_hold_horizon_head: bool = False
    # ── Dip-analysis head (2026-05-26) ────────────────────────────────
    # When enabled, adds nn.Linear(d_model, DIP_HEAD_DIM=18) regressing the
    # distributional dip profile (dip_p50/p90 of mae_before_mfe + recovery_p50)
    # over {long,short}×K. Forces the transformer to EXPLICITLY represent
    # dip-state internally — so its pooled z (and every downstream head + the
    # Entry-IQL that consumes it) encodes "are we about to enter at the top of a
    # dip". This is the signal R_WAIT_OPP rewards. State-dict matches when off.
    enable_dip_head: bool = False
    # ── Self-supervised forecast head (#5) ────────────────────────────
    # nn.Linear(d_model, FORECAST_HEAD_DIM): predicts cumulative future return
    # (bps) at FORECAST_HORIZONS. Self-supervised aux objective → representation
    # captures forward dynamics. State-dict matches when off (gated, default OFF).
    enable_forecast_head: bool = False
    # ── Dip-timing / tail-risk / vol-forecast heads (2026-05-26) ──────
    # All gated, state-dict matches when off. timing: WHEN dip bottoms + WHEN
    # favorable peak hits (TIMING_HEAD_DIM=12). tail_risk: p90 worst-adverse over
    # full horizon (TAIL_RISK_HEAD_DIM=6, pinball q=0.9). vol_forecast: forward
    # realized vol (VOL_FORECAST_HEAD_DIM=3). See module-level *_HEAD_DIM layout.
    enable_timing_head: bool = False
    enable_tail_risk_head: bool = False
    enable_vol_forecast_head: bool = False
    # ── Regime FiLM (2026-06-03, BIG-9) ───────────────────────────────
    # When enabled, a small MLP over the ctx_cat embedding (incl. the repaired
    # trend_regime_id/vol_regime slots) produces (gamma,beta) that FiLM-modulate a
    # SEPARATE z_dir feeding ONLY the direction head: z_dir = (1+gamma)*z + beta. This
    # lets DIRECTION adapt per-regime (e.g. suppress the mean-revert-short residual in a
    # strong uptrend) without perturbing z for the aux heads / downstream IQL. The output
    # Linear is ZERO-INIT so gamma=0,beta=0 at cold start -> z_dir==z -> bit-identical.
    # Default OFF; the regime-robust V10 retrain opts in (a sample-efficient inductive bias,
    # NOT a new capability — validate it earns its keep on 2026 OOT side-symmetry).
    enable_regime_film: bool = False
    # ── Specialist feature-family fusion (2026-06-28 foundation seq146) ──
    # Optional specialist encoders over audited seq/snap feature groups:
    # structure_swing, smc_liquidity, trend_ema, vol_compression,
    # momentum_flow and session_regime. Default OFF so existing bundles have
    # identical module structure and state_dict behaviour.
    enable_specialist_fusion: bool = False
    specialist_num_layers: int = 1
    specialist_fusion_scale: float = 0.25
    # ── Positional encoding (temporal order) ──────────────────────────
    # When enabled, adds sinusoidal positional encoding to the base seq and
    # every per-TF sequence BEFORE its transformer encoder. Without it, the
    # encoder + mean-pool are fully permutation-invariant: the model is blind
    # to the temporal ORDER of the window and only sees an unordered bag of
    # bars. Default OFF so v_FIXED bundles keep bit-identical forward
    # behaviour; the buffer is persistent=False so state_dict is unchanged.
    enable_pos_enc: bool = False


class EntryV10CtxHybridTransformer(nn.Module):
    """
    Minimal, strict CTX model used by:
      - gx1/models/entry_v10/entry_v10_bundle.py
      - gx1/rl/entry_v10/train_entry_transformer_v10.py (CTX variant)

    Forward signature (expected by docs/usage):
        out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
        out["direction_logits"]  -> (B, 3)  # classes: 0=LONG, 1=SHORT, 2=FLAT
        out["path_quality"]      -> (B, 1)  # auxiliary regression (runtime gate)
        out["mfe_first_n"]       -> (B, 1)  # auxiliary regression (runtime gate)
        out["tradable_logit"]    -> (B, 1)  # auxiliary (binary) tradable head
        out["bad_path_logit"]    -> (B, 1)  # auxiliary (binary) early-adverse / MAE-first head
        out["clean_edge_logit"]  -> (B, 1)  # auxiliary (binary) premium clean-edge head
        out["survival_logit"]    -> (B, 1)  # auxiliary (binary) survives-first-adverse head
    """

    def __init__(
        self,
        *,
        seq_input_dim: int,
        snap_input_dim: int,
        seq_len: int,
        ctx_cont_dim: int = 6,
        ctx_cat_dim: int = 6,
        residual_scale: float = 0.35,
        anchor_eps: float = 1e-6,
        enable_anchor_gate: bool = False,
        anchor_gate_init: float = 1.0,
        enable_hierarchical_entry_heads: bool = False,
        enable_hierarchical_direction_composition: bool = False,
        hierarchical_composition_residual_logit_cap: float = 0.0,
        hierarchical_composition_residual_side_neutral: bool = False,
        hierarchical_composition_public_flat_from_trade: bool = False,
        hierarchical_public_direction_composition: str = "logprob",
        enable_hierarchical_public_trade_head: bool = False,
        enable_hierarchical_public_side_head: bool = False,
        enable_hierarchical_ctx_prior_adapter: bool = False,
        hierarchical_ctx_prior_adapter_scale: float = 0.0,
        enable_hierarchical_ctx_direction_calibration: bool = False,
        hierarchical_ctx_direction_calibration_scale: float = 0.0,
        hierarchical_ctx_direction_calibration_cap: float = 0.0,
        enable_side_validity_head: bool = False,
        enable_trendline_rail_head: bool = False,
        trendline_rail_output_dim: int = 4,
        # Multi-TF extension (V12.2). All default to OFF — model behaves
        # identically to v3 unless a bundle explicitly enables them.
        enable_multi_tf: bool = False,
        m15_seq_dim: int = 0,
        h1_seq_dim: int = 0,
        h4_seq_dim: int = 0,
        d1_seq_dim: int = 0,
        m15_seq_len: int = 96,
        h1_seq_len: int = 96,
        h4_seq_len: int = 96,
        d1_seq_len: int = 96,
        multi_tf_num_layers: int = 2,
        multi_tf_scale: float = 0.5,
        enable_cross_tf_attn: bool = False,
        # V2 extension: enable M5 as 5th multi-TF branch
        enable_multi_tf_m5: bool = False,
        m5_seq_dim: int = 0,
        m5_seq_len: int = 96,
        # Distillation Q-head (V13 prep). Default OFF so v_FIXED bundles
        # load with strict=True. When True, adds q_head linear layer.
        enable_q_head: bool = False,
        enable_mtf_direction_head: bool = False,
        mtf_dir_scale_init: float = 0.2,
        # V10 v3+ aux heads (Targets 1-4). All OFF by default so v_FIXED
        # bundles continue to load with strict=True.
        enable_tf_agreement_head: bool = False,
        enable_path_quality_variance_head: bool = False,
        enable_position_size_head: bool = False,
        enable_hold_horizon_head: bool = False,
        enable_dip_head: bool = False,
        enable_forecast_head: bool = False,
        enable_timing_head: bool = False,
        enable_tail_risk_head: bool = False,
        enable_vol_forecast_head: bool = False,
        enable_pos_enc: bool = False,
        enable_regime_film: bool = False,   # 2026-06-03 BIG-9 (default OFF = bit-parity)
        enable_specialist_fusion: bool = False,
        specialist_input_indices: Optional[Dict[str, list[int]]] = None,
        specialist_num_layers: int = 1,
        specialist_fusion_scale: float = 0.25,
        # 2026-06-02: per-TF learnable input scaling (V10 v5+). All default
        # OFF so legacy bundles continue to load strict=True.
        enable_tf_input_scale: bool = False,
        tf_input_scale_init_m5: float = 1.0,
        tf_input_scale_init_m15: float = 1.0,
        tf_input_scale_init_h1: float = 0.7,
        tf_input_scale_init_h4: float = 0.5,
        tf_input_scale_init_d1: float = 0.3,
    ) -> None:
        super().__init__()
        if seq_input_dim <= 0 or snap_input_dim <= 0 or seq_len <= 0:
            raise RuntimeError(
                f"INVALID_INIT: seq_input_dim={seq_input_dim} snap_input_dim={snap_input_dim} seq_len={seq_len}"
            )
        if enable_multi_tf:
            if min(m15_seq_dim, h1_seq_dim, h4_seq_dim, d1_seq_dim) <= 0:
                raise RuntimeError(
                    f"MULTI_TF_DIM_INVALID: when enable_multi_tf=True, all of m15/h1/h4/d1_seq_dim must be >0. "
                    f"Got m15={m15_seq_dim} h1={h1_seq_dim} h4={h4_seq_dim} d1={d1_seq_dim}"
                )
        if enable_multi_tf_m5 and m5_seq_dim <= 0:
            raise RuntimeError(
                f"MULTI_TF_M5_DIM_INVALID: enable_multi_tf_m5=True requires m5_seq_dim > 0; got {m5_seq_dim}"
            )
        if enable_trendline_rail_head and int(trendline_rail_output_dim) < 4:
            raise RuntimeError(
                "TRENDLINE_RAIL_OUTPUT_DIM_INVALID: trendline rail head requires at least 4 outputs; "
                f"got {int(trendline_rail_output_dim)}"
            )
        if enable_side_validity_head and not enable_hierarchical_entry_heads:
            raise RuntimeError("SIDE_VALIDITY_HEAD_REQUIRES_HIERARCHICAL_ENTRY_HEADS")
        if enable_hierarchical_ctx_prior_adapter and not enable_hierarchical_entry_heads:
            raise RuntimeError("HIERARCHICAL_CTX_PRIOR_ADAPTER_REQUIRES_HIERARCHICAL_ENTRY_HEADS")
        if enable_hierarchical_ctx_prior_adapter and int(ctx_cat_dim) <= 0:
            raise RuntimeError(
                "HIERARCHICAL_CTX_PRIOR_ADAPTER_REQUIRES_CTX_CAT: "
                f"ctx_cat_dim={int(ctx_cat_dim)}"
            )
        if enable_hierarchical_ctx_direction_calibration and not enable_hierarchical_direction_composition:
            raise RuntimeError("HIERARCHICAL_CTX_DIRECTION_CALIBRATION_REQUIRES_COMPOSITION")
        if enable_hierarchical_ctx_direction_calibration and int(ctx_cat_dim) <= 0:
            raise RuntimeError(
                "HIERARCHICAL_CTX_DIRECTION_CALIBRATION_REQUIRES_CTX_CAT: "
                f"ctx_cat_dim={int(ctx_cat_dim)}"
            )
        if enable_hierarchical_public_trade_head and not enable_hierarchical_direction_composition:
            raise RuntimeError("HIERARCHICAL_PUBLIC_TRADE_HEAD_REQUIRES_COMPOSITION")
        if enable_hierarchical_public_side_head and not enable_hierarchical_direction_composition:
            raise RuntimeError("HIERARCHICAL_PUBLIC_SIDE_HEAD_REQUIRES_COMPOSITION")
        if enable_hierarchical_direction_composition and not enable_hierarchical_entry_heads:
            raise RuntimeError("HIERARCHICAL_DIRECTION_COMPOSITION_REQUIRES_HIERARCHICAL_ENTRY_HEADS")
        if hierarchical_composition_public_flat_from_trade and not enable_hierarchical_direction_composition:
            raise RuntimeError("HIERARCHICAL_COMPOSITION_PUBLIC_FLAT_FROM_TRADE_REQUIRES_COMPOSITION")
        hierarchical_public_direction_composition = str(hierarchical_public_direction_composition or "logprob").strip().lower()
        if hierarchical_public_direction_composition not in {"logprob", "margin", "margin_centered", "margin_maxnorm"}:
            raise RuntimeError(
                "HIERARCHICAL_PUBLIC_DIRECTION_COMPOSITION_INVALID: "
                f"got {hierarchical_public_direction_composition!r}, "
                "expected 'logprob', 'margin', 'margin_centered', or 'margin_maxnorm'"
            )
        if float(hierarchical_ctx_prior_adapter_scale) < 0.0:
            raise RuntimeError(
                "HIERARCHICAL_CTX_PRIOR_ADAPTER_SCALE_INVALID: "
                f"got {float(hierarchical_ctx_prior_adapter_scale)}"
            )
        if float(hierarchical_ctx_direction_calibration_scale) < 0.0:
            raise RuntimeError(
                "HIERARCHICAL_CTX_DIRECTION_CALIBRATION_SCALE_INVALID: "
                f"got {float(hierarchical_ctx_direction_calibration_scale)}"
            )
        if float(hierarchical_ctx_direction_calibration_cap) < 0.0:
            raise RuntimeError(
                "HIERARCHICAL_CTX_DIRECTION_CALIBRATION_CAP_INVALID: "
                f"got {float(hierarchical_ctx_direction_calibration_cap)}"
            )
        if float(hierarchical_composition_residual_logit_cap) < 0.0:
            raise RuntimeError(
                "HIERARCHICAL_COMPOSITION_RESIDUAL_LOGIT_CAP_INVALID: "
                f"got {float(hierarchical_composition_residual_logit_cap)}"
            )

        self.cfg = CtxModelConfig(
            seq_input_dim=seq_input_dim,
            snap_input_dim=snap_input_dim,
            seq_len=seq_len,
            ctx_cont_dim=int(ctx_cont_dim),
            ctx_cat_dim=int(ctx_cat_dim),
            residual_scale=float(residual_scale),
            anchor_eps=float(anchor_eps),
            enable_anchor_gate=bool(enable_anchor_gate),
            anchor_gate_init=float(anchor_gate_init),
            enable_hierarchical_entry_heads=bool(enable_hierarchical_entry_heads),
            enable_hierarchical_direction_composition=bool(enable_hierarchical_direction_composition),
            hierarchical_composition_residual_logit_cap=float(hierarchical_composition_residual_logit_cap),
            hierarchical_composition_residual_side_neutral=bool(
                hierarchical_composition_residual_side_neutral
            ),
            hierarchical_composition_public_flat_from_trade=bool(
                hierarchical_composition_public_flat_from_trade
            ),
            hierarchical_public_direction_composition=hierarchical_public_direction_composition,
            enable_hierarchical_public_trade_head=bool(enable_hierarchical_public_trade_head),
            enable_hierarchical_public_side_head=bool(enable_hierarchical_public_side_head),
            enable_hierarchical_ctx_prior_adapter=bool(enable_hierarchical_ctx_prior_adapter),
            hierarchical_ctx_prior_adapter_scale=float(hierarchical_ctx_prior_adapter_scale),
            enable_hierarchical_ctx_direction_calibration=bool(enable_hierarchical_ctx_direction_calibration),
            hierarchical_ctx_direction_calibration_scale=float(
                hierarchical_ctx_direction_calibration_scale
            ),
            hierarchical_ctx_direction_calibration_cap=float(hierarchical_ctx_direction_calibration_cap),
            enable_side_validity_head=bool(enable_side_validity_head),
            enable_trendline_rail_head=bool(enable_trendline_rail_head),
            trendline_rail_output_dim=int(trendline_rail_output_dim),
            enable_multi_tf=bool(enable_multi_tf),
            m15_seq_dim=int(m15_seq_dim),
            h1_seq_dim=int(h1_seq_dim),
            h4_seq_dim=int(h4_seq_dim),
            d1_seq_dim=int(d1_seq_dim),
            m15_seq_len=int(m15_seq_len),
            h1_seq_len=int(h1_seq_len),
            h4_seq_len=int(h4_seq_len),
            d1_seq_len=int(d1_seq_len),
            enable_multi_tf_m5=bool(enable_multi_tf_m5),
            m5_seq_dim=int(m5_seq_dim),
            m5_seq_len=int(m5_seq_len),
            multi_tf_num_layers=int(multi_tf_num_layers),
            multi_tf_scale=float(multi_tf_scale),
            enable_cross_tf_attn=bool(enable_cross_tf_attn),
            enable_q_head=bool(enable_q_head),
            enable_mtf_direction_head=bool(enable_mtf_direction_head),
            mtf_dir_scale_init=float(mtf_dir_scale_init),
            enable_tf_agreement_head=bool(enable_tf_agreement_head),
            enable_path_quality_variance_head=bool(enable_path_quality_variance_head),
            enable_position_size_head=bool(enable_position_size_head),
            enable_hold_horizon_head=bool(enable_hold_horizon_head),
            enable_dip_head=bool(enable_dip_head),
            enable_forecast_head=bool(enable_forecast_head),
            enable_timing_head=bool(enable_timing_head),
            enable_tail_risk_head=bool(enable_tail_risk_head),
            enable_vol_forecast_head=bool(enable_vol_forecast_head),
            enable_pos_enc=bool(enable_pos_enc),
            enable_regime_film=bool(enable_regime_film),
            enable_specialist_fusion=bool(enable_specialist_fusion),
            specialist_num_layers=int(specialist_num_layers),
            specialist_fusion_scale=float(specialist_fusion_scale),
            enable_tf_input_scale=bool(enable_tf_input_scale),
            tf_input_scale_init_m5=float(tf_input_scale_init_m5),
            tf_input_scale_init_m15=float(tf_input_scale_init_m15),
            tf_input_scale_init_h1=float(tf_input_scale_init_h1),
            tf_input_scale_init_h4=float(tf_input_scale_init_h4),
            tf_input_scale_init_d1=float(tf_input_scale_init_d1),
        )

        d_model = int(self.cfg.d_model)
        n_heads = int(self.cfg.n_heads)
        num_layers = int(self.cfg.num_layers)
        dropout = float(self.cfg.dropout)
        d_ff = int(self.cfg.dim_feedforward) if self.cfg.dim_feedforward else int(d_model * 4)

        # Project signal-only inputs into transformer dimension
        self.seq_proj = nn.Linear(int(seq_input_dim), d_model)
        self.snap_proj = nn.Linear(int(snap_input_dim), d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Context encoders
        self.ctx_cat_emb = nn.Embedding(int(self.cfg.ctx_cat_vocab), int(self.cfg.ctx_cat_emb_dim))
        self.ctx_cont_proj = nn.Linear(int(self.cfg.ctx_cont_dim), d_model)

        # Combine: pooled_seq + snap + ctx_cat + ctx_cont
        ctx_cat_flat_dim = int(self.cfg.ctx_cat_dim) * int(self.cfg.ctx_cat_emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_model + ctx_cat_flat_dim + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Specialist feature-family fusion. Each specialist sees only its
        # audited seq feature subset and produces one token. A gated mixture
        # over those tokens adds a zero-init residual correction to z_v3.
        self._specialist_names: tuple[str, ...] = tuple()
        if self.cfg.enable_specialist_fusion:
            cleaned: Dict[str, list[int]] = {}
            raw_indices = specialist_input_indices or {}
            for raw_name, raw_idx in raw_indices.items():
                name = str(raw_name).strip()
                if not name:
                    continue
                idx = sorted({int(i) for i in list(raw_idx or [])})
                if not idx:
                    continue
                if min(idx) < 0 or max(idx) >= int(seq_input_dim):
                    raise RuntimeError(
                        f"SPECIALIST_INDEX_OOB: {name} has indices outside [0,{int(seq_input_dim) - 1}]"
                    )
                cleaned[name] = idx
            if not cleaned:
                raise RuntimeError("SPECIALIST_FUSION_REQUIRES_INDICES")
            self._specialist_names = tuple(cleaned.keys())
            self.specialist_proj = nn.ModuleDict(
                {name: nn.Linear(len(idx), d_model) for name, idx in cleaned.items()}
            )
            specialist_layers = max(1, int(self.cfg.specialist_num_layers))

            def _mk_specialist_enc() -> nn.TransformerEncoder:
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                return nn.TransformerEncoder(layer, num_layers=specialist_layers)

            self.specialist_encoder = nn.ModuleDict(
                {name: _mk_specialist_enc() for name in self._specialist_names}
            )
            for name, idx in cleaned.items():
                self.register_buffer(
                    f"specialist_idx_{name}",
                    torch.tensor(idx, dtype=torch.long),
                    persistent=False,
                )
            self.specialist_gate = nn.Linear(d_model, len(self._specialist_names))
            self.specialist_out = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.specialist_out.weight)
            nn.init.zeros_(self.specialist_out.bias)
            self.register_buffer("specialist_fusion_scale", torch.tensor(float(self.cfg.specialist_fusion_scale)))

        # 3-class direction head: LONG / SHORT / FLAT
        self.head_direction = nn.Linear(d_model, 3)
        # Post-hoc direction calibration (temperature + per-class bias). Identity
        # (None) by default; the bundle loader sets it from
        # bundle_metadata["direction_calibration"] via set_direction_calibration().
        # Plain attribute (NOT a Parameter/buffer) so existing bundles strict-load
        # unchanged and bundles without the metadata key are bit-identical. Applied
        # in forward on the FINAL direction_logits so audit and live serve see the
        # same calibrated outputs by construction. (Vedtak
        # SMART_SEQ520_candidate_train_20260703 — FLAT-rate non-stationarity leg.)
        self._direction_cal: Optional[Tuple[float, torch.Tensor]] = None
        # Post-hoc PATH-head calibration (2026-07-05, cand#1 path-inversion lesson):
        # affine on path_quality (a*x+b) + Platt on bad_path_logit (x/T+b).
        # Identity when never installed -> bit-identical for existing bundles.
        self._path_cal: Optional[Tuple[float, float, float, float]] = None
        # V10-7 (2026-06-03 all-models scan): zero-init the direction RESIDUAL head so cold-start
        # delta_logits=0 -> direction == the (balanced) XGB anchor, and the residual is learned
        # from zero. head_direction was the lone correction head NOT zero-init (q_head/FiLM/
        # cross_tf_out/multi_tf_fuse all are). Only affects fresh-train init; loading a trained
        # bundle overwrites these weights, so it's bit-parity-neutral for existing bundles.
        nn.init.zeros_(self.head_direction.weight)
        nn.init.zeros_(self.head_direction.bias)
        if self.cfg.enable_anchor_gate:
            self.head_anchor_gate = nn.Linear(d_model, 3)
            nn.init.zeros_(self.head_anchor_gate.weight)
            init = min(max(float(self.cfg.anchor_gate_init), 1e-4), 1.0 - 1e-4)
            nn.init.constant_(self.head_anchor_gate.bias, math.log(init / (1.0 - init)))
        # Regime FiLM (BIG-9): condition direction on the ctx_cat embedding (regime slots).
        # ZERO-INIT final layer -> gamma=beta=0 at init -> z_dir==z -> bit-parity with cement.
        if bool(getattr(self.cfg, "enable_regime_film", False)):
            self.regime_film = nn.Sequential(
                nn.Linear(ctx_cat_flat_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),   # -> (gamma, beta)
            )
            nn.init.zeros_(self.regime_film[-1].weight)
            nn.init.zeros_(self.regime_film[-1].bias)
        # Auxiliary heads that remain active in the canonical runtime lane.
        self.head_path_quality = nn.Linear(d_model, 1)
        self.head_mfe_first_n = nn.Linear(d_model, 1)
        self.head_tradable = nn.Linear(d_model, 1)
        self.head_bad_path = nn.Linear(d_model, 1)
        # Replay-oriented quality heads used by training/audit. Runtime may ignore them.
        self.head_clean_edge = nn.Linear(d_model, 1)
        self.head_survival = nn.Linear(d_model, 1)
        if self.cfg.enable_hierarchical_entry_heads:
            self.head_trade = nn.Linear(d_model, 1)
            self.head_side = nn.Linear(d_model, 2)              # LONG / SHORT conditional on trade
            if self.cfg.enable_hierarchical_public_trade_head:
                self.head_public_trade = nn.Linear(d_model, 1)  # public trade / FLAT threshold
            if self.cfg.enable_hierarchical_public_side_head:
                self.head_public_side = nn.Linear(d_model, 2)   # public LONG / SHORT conditional on trade
            self.head_side_utility = nn.Linear(d_model, 2)      # expected path utility, bps
            self.head_side_bad_path = nn.Linear(d_model, 2)     # side-specific bad-path logits
            self.head_side_mae = nn.Linear(d_model, 2)          # side-specific expected MAE, bps
            if self.cfg.enable_hierarchical_ctx_prior_adapter:
                self.hierarchical_ctx_prior_adapter = nn.Linear(ctx_cat_flat_dim, 3)
            if self.cfg.enable_hierarchical_ctx_direction_calibration:
                self.hierarchical_ctx_direction_calibration = nn.Linear(ctx_cat_flat_dim, 3)
            if self.cfg.enable_side_validity_head:
                self.head_side_validity = nn.Linear(d_model, 2)  # side-specific valid-trade logits
            nn.init.zeros_(self.head_trade.bias)
            nn.init.zeros_(self.head_side.bias)
            if self.cfg.enable_hierarchical_public_side_head:
                nn.init.zeros_(self.head_public_side.weight)
                nn.init.zeros_(self.head_public_side.bias)
            if self.cfg.enable_hierarchical_public_trade_head:
                nn.init.zeros_(self.head_public_trade.weight)
                nn.init.zeros_(self.head_public_trade.bias)
            nn.init.zeros_(self.head_side_utility.bias)
            nn.init.zeros_(self.head_side_bad_path.bias)
            nn.init.zeros_(self.head_side_mae.bias)
            if self.cfg.enable_hierarchical_ctx_prior_adapter:
                nn.init.zeros_(self.hierarchical_ctx_prior_adapter.weight)
                nn.init.zeros_(self.hierarchical_ctx_prior_adapter.bias)
            if self.cfg.enable_hierarchical_ctx_direction_calibration:
                nn.init.zeros_(self.hierarchical_ctx_direction_calibration.weight)
                nn.init.zeros_(self.hierarchical_ctx_direction_calibration.bias)
            if self.cfg.enable_side_validity_head:
                nn.init.zeros_(self.head_side_validity.bias)
        if self.cfg.enable_trendline_rail_head:
            self.head_trendline_rail = nn.Linear(d_model, int(self.cfg.trendline_rail_output_dim))
            nn.init.zeros_(self.head_trendline_rail.bias)

        # ── Distillation Q-head (V13 prep) ────────────────────────────
        # Only instantiated when enable_q_head=True. Zero-init so a fresh
        # (un-distilled) head outputs all-zeros — matches "no IQL signal"
        # baseline and keeps training stable until KL-loss pulls it.
        if self.cfg.enable_q_head:
            self.q_head = nn.Linear(d_model, 3)  # [q_skip, q_long, q_short]
            nn.init.zeros_(self.q_head.weight)
            nn.init.zeros_(self.q_head.bias)

        # ── Multi-TF → direction head (forceful MTF→dir, 2026-06-06) ───
        # Reads the GATED cross-TF representation (`pooled`, captured in forward —
        # NON-zero-init, unlike mtf_correction) and contributes a dedicated,
        # non-zeroable term to direction_logits. Requires the cross-TF branch
        # (that's where `pooled` lives). Small-normal init on the Linear (NOT zero
        # — zero-init is the trap that made the model ignore multi-TF) + a learnable
        # non-zero scale so it informs direction from epoch 0. No params when off.
        if (self.cfg.enable_mtf_direction_head and self.cfg.enable_multi_tf
                and self.cfg.enable_cross_tf_attn):
            self.head_mtf_direction = nn.Linear(d_model, 3)  # [long, short, flat]
            nn.init.normal_(self.head_mtf_direction.weight, std=0.02)
            nn.init.zeros_(self.head_mtf_direction.bias)
            self.mtf_dir_scale = nn.Parameter(
                torch.tensor(float(self.cfg.mtf_dir_scale_init))
            )

        # ── TF-agreement head (V10 v3+ Target 1) ──────────────────────
        # Only instantiated when enable_tf_agreement_head=True. Outputs
        # raw logit; caller applies sigmoid for [0,1] probability.
        if self.cfg.enable_tf_agreement_head:
            self.head_tf_agreement = nn.Linear(d_model, 1)

        # ── Path-quality variance head (V10 v3+ Target 2) ─────────────
        # Outputs log-variance for path_quality. Combined with the
        # existing head_path_quality (mean) to form a heteroscedastic
        # Gaussian prediction. Loss = 0.5 * (log_var + (y-mu)^2 / var).
        # Init bias to 0 → variance = 1 baseline; lets the model start
        # at the same effective MSE as before until it learns to vary.
        if self.cfg.enable_path_quality_variance_head:
            self.head_path_quality_log_var = nn.Linear(d_model, 1)
            nn.init.zeros_(self.head_path_quality_log_var.bias)

        # ── Position-size head (V10 v3+ Target 3) ─────────────────────
        # Outputs raw logit; caller applies sigmoid for [0,1] probability.
        if self.cfg.enable_position_size_head:
            self.head_position_size = nn.Linear(d_model, 1)

        # ── Hold-horizon head (V10 v3+ Target 4) ──────────────────────
        # Outputs raw logit; caller applies sigmoid for [0,1] probability
        # then multiplies by 1440 to get expected hold-bars.
        if self.cfg.enable_hold_horizon_head:
            self.head_hold_horizon = nn.Linear(d_model, 1)
        if self.cfg.enable_dip_head:
            # 18 outputs: (dir{long,short} × K{12,48,96} × {dip_p50,dip_p90,recovery_p50})
            self.head_dip = nn.Linear(d_model, DIP_HEAD_DIM)
        if self.cfg.enable_forecast_head:
            self.head_forecast = nn.Linear(d_model, FORECAST_HEAD_DIM)  # cum. future ret (bps) @ horizons
        if self.cfg.enable_timing_head:
            self.head_timing = nn.Linear(d_model, TIMING_HEAD_DIM)  # dip-bottom/peak bar-frac (see TIMING_* layout)
        if self.cfg.enable_tail_risk_head:
            self.head_tail_risk = nn.Linear(d_model, TAIL_RISK_HEAD_DIM)  # p90 worst-adverse (see TAIL_RISK_* layout)
        if self.cfg.enable_vol_forecast_head:
            self.head_vol_forecast = nn.Linear(d_model, VOL_FORECAST_HEAD_DIM)  # fwd realized vol (bps) @ horizons

        # ── Multi-TF encoders (V12.2) — only instantiated when enabled ──
        # Each TF gets its own lightweight TransformerEncoder + linear projection.
        # When disabled, NO parameters are added → state_dict matches v3 exactly,
        # so existing v3 bundles load with strict=True.
        if self.cfg.enable_multi_tf:
            mtf_dropout = dropout
            mtf_layers = int(self.cfg.multi_tf_num_layers)
            self.m15_proj = nn.Linear(int(self.cfg.m15_seq_dim), d_model)
            self.h1_proj = nn.Linear(int(self.cfg.h1_seq_dim), d_model)
            self.h4_proj = nn.Linear(int(self.cfg.h4_seq_dim), d_model)
            self.d1_proj = nn.Linear(int(self.cfg.d1_seq_dim), d_model)
            def _mk_enc():
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                    dropout=mtf_dropout, batch_first=True,
                    activation="gelu", norm_first=True,
                )
                return nn.TransformerEncoder(layer, num_layers=mtf_layers)
            self.m15_encoder = _mk_enc()
            self.h1_encoder = _mk_enc()
            self.h4_encoder = _mk_enc()
            self.d1_encoder = _mk_enc()
            # V2 (2026-05-22): if enable_multi_tf_m5=True, add 5th branch for M5
            # V2-feature stream. State_dict gains m5_proj + m5_encoder weights when
            # this flag is on; v_FIXED bundles (flag=False) load unchanged.
            n_tf_branches = 4
            if self.cfg.enable_multi_tf_m5:
                self.m5_proj = nn.Linear(int(self.cfg.m5_seq_dim), d_model)
                self.m5_encoder = _mk_enc()
                self._expected_m5_seq_dim = int(self.cfg.m5_seq_dim)
                n_tf_branches = 5
            # V12.2 v2: ADDITIVE residual fusion. multi_tf_fuse operates ONLY on
            # multi-TF pools (not concatenated with z_v3). Output is a small
            # CORRECTION that's added to z_v3 — preserves v3 baseline behavior
            # when multi-TF is uninformative (random init → near-zero output).
            self.multi_tf_fuse = nn.Sequential(
                nn.Linear(n_tf_branches * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            nn.init.zeros_(self.multi_tf_fuse[-1].bias)
            nn.init.normal_(self.multi_tf_fuse[-1].weight, std=0.01)
            # ── Cross-TF attention + learnable per-TF gates (#3+#4) ──
            # Each per-TF pool becomes a token; attention lets TFs attend across
            # each other; a learnable softmax gate weights TFs (regime-dependent).
            # Zero-init output → starts as no-op (stable cold start).
            if self.cfg.enable_cross_tf_attn:
                self.cross_tf_attn = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                        dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
                    ),
                    num_layers=1,
                )
                self.tf_gate_logits = nn.Parameter(torch.zeros(n_tf_branches))  # learnable per-TF weight
                self.cross_tf_out = nn.Linear(d_model, d_model)
                nn.init.zeros_(self.cross_tf_out.weight)
                nn.init.zeros_(self.cross_tf_out.bias)
            self._expected_m15_seq_dim = int(self.cfg.m15_seq_dim)
            self._expected_h1_seq_dim = int(self.cfg.h1_seq_dim)
            self._expected_h4_seq_dim = int(self.cfg.h4_seq_dim)
            self._expected_d1_seq_dim = int(self.cfg.d1_seq_dim)
            self.register_buffer("multi_tf_scale", torch.tensor(float(self.cfg.multi_tf_scale)))
            # 2026-06-02 fix (V10 short-bias rooted in equal-weight TF fusion):
            # Per-TF learnable input scalars initialized with user prior.
            # Cement's tf_gate_logits learned essentially 1/5 uniform (gates barely
            # moved from zero-init). To break that uniformity and let the model
            # actually USE M5 micro-signal vs D1 macro-signal, we scale per-TF
            # inputs BEFORE encoding. Initialized to enable_multi_tf_m5 prior:
            #   M5=1.0, M15=1.0, H1=0.7, H4=0.5, D1=0.3
            # Model can adjust these during training (gradient through encoders +
            # final loss). When cfg.enable_tf_input_scale=True these are active.
            self._enable_tf_input_scale = bool(getattr(self.cfg, "enable_tf_input_scale", False))
            if self._enable_tf_input_scale:
                init_m5 = float(getattr(self.cfg, "tf_input_scale_init_m5", 1.0))
                init_m15 = float(getattr(self.cfg, "tf_input_scale_init_m15", 1.0))
                init_h1 = float(getattr(self.cfg, "tf_input_scale_init_h1", 0.7))
                init_h4 = float(getattr(self.cfg, "tf_input_scale_init_h4", 0.5))
                init_d1 = float(getattr(self.cfg, "tf_input_scale_init_d1", 0.3))
                self.tf_input_scale_m5 = nn.Parameter(torch.tensor(init_m5))
                self.tf_input_scale_m15 = nn.Parameter(torch.tensor(init_m15))
                self.tf_input_scale_h1 = nn.Parameter(torch.tensor(init_h1))
                self.tf_input_scale_h4 = nn.Parameter(torch.tensor(init_h4))
                self.tf_input_scale_d1 = nn.Parameter(torch.tensor(init_d1))

        # Strict markers (useful for debugging)
        self._expected_seq_dim = int(seq_input_dim)
        self._expected_snap_dim = int(snap_input_dim)
        self._expected_seq_len = int(seq_len)
        self._expected_ctx_cat_dim = int(self.cfg.ctx_cat_dim)
        self._expected_ctx_cont_dim = int(self.cfg.ctx_cont_dim)
        # Anchored residual scale stored in state_dict for replay parity
        self.register_buffer("residual_scale", torch.tensor(float(self.cfg.residual_scale)))
        self.register_buffer("anchor_eps", torch.tensor(float(self.cfg.anchor_eps)))
        self.register_buffer(
            "hierarchical_composition_residual_logit_cap",
            torch.tensor(float(self.cfg.hierarchical_composition_residual_logit_cap)),
        )

        # ── Positional encoding buffers (persistent=False → not in state_dict) ──
        self.enable_pos_enc = bool(self.cfg.enable_pos_enc)
        if self.enable_pos_enc:
            self.register_buffer("pos_enc", self._sinusoidal_pe(int(seq_len), d_model), persistent=False)
            if self.cfg.enable_multi_tf:
                self.register_buffer("pos_enc_m15", self._sinusoidal_pe(int(self.cfg.m15_seq_len), d_model), persistent=False)
                self.register_buffer("pos_enc_h1", self._sinusoidal_pe(int(self.cfg.h1_seq_len), d_model), persistent=False)
                self.register_buffer("pos_enc_h4", self._sinusoidal_pe(int(self.cfg.h4_seq_len), d_model), persistent=False)
                self.register_buffer("pos_enc_d1", self._sinusoidal_pe(int(self.cfg.d1_seq_len), d_model), persistent=False)
                if self.cfg.enable_multi_tf_m5:
                    self.register_buffer("pos_enc_m5", self._sinusoidal_pe(int(self.cfg.m5_seq_len), d_model), persistent=False)

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding, shape (1, seq_len, d_model)."""
        pe = torch.zeros(int(seq_len), int(d_model))
        position = torch.arange(0, int(seq_len), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, int(d_model), 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        n_cos = pe[:, 1::2].size(1)
        pe[:, 1::2] = torch.cos(position * div_term[:n_cos])
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    def _add_pe(self, t: torch.Tensor, buf_name: str) -> torch.Tensor:
        """Add positional encoding (sliced to current seq len) if enabled."""
        if not getattr(self, "enable_pos_enc", False):
            return t
        pe = getattr(self, buf_name)
        return t + pe[:, : t.size(1)]

    def _anchor_logits_from_snap(self, snap_x: torch.Tensor) -> torch.Tensor:
        # Anchor from XGB probs: [p_long, p_short, p_flat] in SIGNAL_BRIDGE_V1 order
        probs = snap_x[:, _ANCHOR_IDX].float()
        eps = float(self.anchor_eps.item())
        probs = torch.clamp(probs, min=eps, max=1.0)
        anchor_logits = torch.log(probs)
        return anchor_logits.detach()

    def set_direction_calibration(self, temperature: float, bias: torch.Tensor) -> None:
        """Install post-hoc direction calibration (fitted on a recent held-out
        window, stored in bundle_metadata["direction_calibration"], applied by
        the bundle loader). direction_logits -> logits/temperature + bias.
        Identity when never called. Fail-loud on bad values."""
        t = float(temperature)
        if not (t > 0.0) or not torch.isfinite(bias).all() or tuple(bias.shape) != (3,):
            raise ValueError(
                f"[ENTRY_DIRECTION_CAL] invalid calibration: temperature={temperature} bias_shape={tuple(bias.shape)}"
            )
        self._direction_cal = (t, bias.detach().clone().float())

    def set_path_calibration(
        self,
        path_quality_scale: float,
        path_quality_shift: float,
        bad_path_temperature: float,
        bad_path_bias: float,
    ) -> None:
        """Install post-hoc path-head calibration (fitted on held-out val,
        stored in bundle_metadata["path_calibration"], applied by the loader).
        path_quality -> scale*x + shift; bad_path_logit -> x/T + b.
        Identity when never called. Fail-loud on bad values. NOTE: affine/Platt
        recalibration fixes MAGNITUDE, never SIGN — a wrong-sign selected-subset
        correlation (the cand#1 defect) must fail the slice-audit gate and be
        fixed by recipe/retrain, not papered over here."""
        vals = (float(path_quality_scale), float(path_quality_shift), float(bad_path_temperature), float(bad_path_bias))
        import math as _math
        if not all(_math.isfinite(v) for v in vals) or vals[2] <= 0.0:
            raise ValueError(f"[ENTRY_PATH_CAL] invalid calibration: {vals}")
        self._path_cal = vals

    def forward(
        self,
        seq_x: torch.Tensor,
        snap_x: torch.Tensor,
        *,
        ctx_cat: torch.Tensor,
        ctx_cont: torch.Tensor,
        seq_m15: Optional[torch.Tensor] = None,
        seq_h1: Optional[torch.Tensor] = None,
        seq_h4: Optional[torch.Tensor] = None,
        seq_d1: Optional[torch.Tensor] = None,
        # V10 base seq IS M5 — accept seq_m5 from shared dataset but ignore it.
        seq_m5: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _assert_shape("seq_x", seq_x, 3)     # (B,T,D)
        _assert_shape("snap_x", snap_x, 2)   # (B,D)
        _assert_shape("ctx_cat", ctx_cat, 2) # (B,ctx_cat_dim)
        _assert_shape("ctx_cont", ctx_cont, 2) # (B,ctx_cont_dim)

        B, T, Dseq = seq_x.shape
        if int(Dseq) != self._expected_seq_dim:
            raise RuntimeError(f"SEQ_DIM_MISMATCH: got={int(Dseq)} expected={self._expected_seq_dim}")
        if int(T) != self._expected_seq_len:
            raise RuntimeError(f"SEQ_LEN_MISMATCH: got={int(T)} expected={self._expected_seq_len}")

        if int(snap_x.shape[1]) != self._expected_snap_dim:
            raise RuntimeError(f"SNAP_DIM_MISMATCH: got={int(snap_x.shape[1])} expected={self._expected_snap_dim}")

        if int(ctx_cat.shape[1]) != self._expected_ctx_cat_dim:
            raise RuntimeError(
                f"CTX_CAT_DIM_MISMATCH: got={int(ctx_cat.shape[1])} expected={self._expected_ctx_cat_dim}"
            )
        if int(ctx_cont.shape[1]) != self._expected_ctx_cont_dim:
            raise RuntimeError(
                f"CTX_CONT_DIM_MISMATCH: got={int(ctx_cont.shape[1])} expected={self._expected_ctx_cont_dim}"
            )

        # Hard finite checks
        _assert_finite("seq_x", seq_x)
        _assert_finite("snap_x", snap_x)
        _assert_finite("ctx_cont", ctx_cont)

        # ctx_cat must be integer
        if ctx_cat.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            raise RuntimeError(f"CTX_CAT_DTYPE_MISMATCH: expected integer dtype, got {ctx_cat.dtype}")

        # Range guard for embedding vocab
        mx = int(ctx_cat.max().item()) if ctx_cat.numel() > 0 else 0
        if mx >= int(self.cfg.ctx_cat_vocab):
            raise RuntimeError(f"CTX_CAT_OOB: max_id={mx} >= vocab={int(self.cfg.ctx_cat_vocab)}")

        # Encode
        seq_h = self.seq_proj(seq_x)                  # (B,T,d)
        seq_h = self._add_pe(seq_h, "pos_enc")        # temporal order (no-op if disabled)
        seq_h = self.encoder(seq_h)                   # (B,T,d)
        seq_pool = seq_h.mean(dim=1)                  # (B,d)

        snap_h = self.snap_proj(snap_x)               # (B,d)

        cat_emb = self.ctx_cat_emb(ctx_cat.long())    # (B,6,emb)
        cat_flat = cat_emb.reshape(B, -1)             # (B,6*emb)

        cont_h = self.ctx_cont_proj(ctx_cont.float()) # (B,d)
        cat_flat = cat_flat * float(self.cfg.ctx_cat_scale)
        cont_h = cont_h * float(self.cfg.ctx_cont_scale)

        fused = torch.cat([seq_pool, snap_h, cat_flat, cont_h], dim=1)
        z_v3 = self.fuse(fused)

        specialist_gate = None
        if self.cfg.enable_specialist_fusion and hasattr(self, "specialist_gate"):
            pools = []
            for name in self._specialist_names:
                idx = getattr(self, f"specialist_idx_{name}").to(seq_x.device)
                seq_part = seq_x.index_select(dim=2, index=idx)
                spec_h = self.specialist_proj[name](seq_part)
                spec_h = self._add_pe(spec_h, "pos_enc")
                spec_pool = self.specialist_encoder[name](spec_h).mean(dim=1)
                pools.append(spec_pool)
            specialist_tokens = torch.stack(pools, dim=1)
            specialist_logits = self.specialist_gate(z_v3)
            specialist_gate = torch.softmax(specialist_logits, dim=1)
            specialist_pool = (specialist_tokens * specialist_gate.unsqueeze(-1)).sum(dim=1)
            specialist_correction = self.specialist_out(specialist_pool)
            _assert_finite("specialist_correction", specialist_correction)
            _assert_finite("specialist_gate", specialist_gate)
            z_v3 = z_v3 + self.specialist_fusion_scale.to(specialist_correction.dtype) * specialist_correction

        # ── Multi-TF second-stage fusion (V12.2 + V2 M5 extension) ──
        # Only active when the model was constructed with enable_multi_tf=True.
        # Missing tensors are a train/serve contract violation; silently falling
        # back to the single-TF path makes live/replay parity meaningless.
        _mtf_inputs_present = all(t is not None for t in (seq_m15, seq_h1, seq_h4, seq_d1))
        if self.cfg.enable_multi_tf and not _mtf_inputs_present:
            missing = [
                name
                for name, value in (
                    ("seq_m15", seq_m15),
                    ("seq_h1", seq_h1),
                    ("seq_h4", seq_h4),
                    ("seq_d1", seq_d1),
                )
                if value is None
            ]
            raise RuntimeError(
                "MULTI_TF_INPUTS_MISSING: enable_multi_tf=True requires "
                f"m15/h1/h4/d1 tensors; missing={missing}"
            )
        if self.cfg.enable_multi_tf and _mtf_inputs_present:
            for name, t, exp_len, exp_dim in (
                ("seq_m15", seq_m15, self.cfg.m15_seq_len, self._expected_m15_seq_dim),
                ("seq_h1", seq_h1, self.cfg.h1_seq_len, self._expected_h1_seq_dim),
                ("seq_h4", seq_h4, self.cfg.h4_seq_len, self._expected_h4_seq_dim),
                ("seq_d1", seq_d1, self.cfg.d1_seq_len, self._expected_d1_seq_dim),
            ):
                _assert_shape(name, t, 3)
                if int(t.shape[1]) != int(exp_len):
                    raise RuntimeError(f"{name.upper()}_LEN_MISMATCH: got={int(t.shape[1])} expected={exp_len}")
                if int(t.shape[2]) != int(exp_dim):
                    raise RuntimeError(f"{name.upper()}_DIM_MISMATCH: got={int(t.shape[2])} expected={exp_dim}")
                _assert_finite(name, t)

            # 2026-06-02: per-TF input scaling (learnable). Scales each TF
            # sequence BEFORE projection so the encoders see "down-weighted"
            # versions of the macro TFs (default init: H1=0.7, H4=0.5, D1=0.3).
            # When enable_tf_input_scale=False (legacy), these multiplications
            # are skipped and the model is bit-identical to pre-2026-06-02.
            if getattr(self, "_enable_tf_input_scale", False):
                seq_m15_in = seq_m15 * self.tf_input_scale_m15
                seq_h1_in = seq_h1 * self.tf_input_scale_h1
                seq_h4_in = seq_h4 * self.tf_input_scale_h4
                seq_d1_in = seq_d1 * self.tf_input_scale_d1
            else:
                seq_m15_in, seq_h1_in, seq_h4_in, seq_d1_in = seq_m15, seq_h1, seq_h4, seq_d1
            m15_pool = self.m15_encoder(self._add_pe(self.m15_proj(seq_m15_in), "pos_enc_m15")).mean(dim=1)   # (B,d)
            h1_pool = self.h1_encoder(self._add_pe(self.h1_proj(seq_h1_in), "pos_enc_h1")).mean(dim=1)
            h4_pool = self.h4_encoder(self._add_pe(self.h4_proj(seq_h4_in), "pos_enc_h4")).mean(dim=1)
            d1_pool = self.d1_encoder(self._add_pe(self.d1_proj(seq_d1_in), "pos_enc_d1")).mean(dim=1)

            # V2: 5th branch for M5 V2-features (different from seq_x raw price-state).
            pool_list = [m15_pool, h1_pool, h4_pool, d1_pool]
            if self.cfg.enable_multi_tf_m5:
                if seq_m5 is None:
                    raise RuntimeError(
                        "SEQ_M5_REQUIRED: enable_multi_tf_m5=True but seq_m5 is None"
                    )
                _assert_shape("seq_m5", seq_m5, 3)
                if int(seq_m5.shape[1]) != int(self.cfg.m5_seq_len):
                    raise RuntimeError(
                        f"SEQ_M5_LEN_MISMATCH: got={int(seq_m5.shape[1])} expected={self.cfg.m5_seq_len}"
                    )
                if int(seq_m5.shape[2]) != int(self._expected_m5_seq_dim):
                    raise RuntimeError(
                        f"SEQ_M5_DIM_MISMATCH: got={int(seq_m5.shape[2])} expected={self._expected_m5_seq_dim}"
                    )
                _assert_finite("seq_m5", seq_m5)
                # 2026-06-02 per-TF scale (applied to M5 V2 branch as well)
                if getattr(self, "_enable_tf_input_scale", False):
                    seq_m5_in = seq_m5 * self.tf_input_scale_m5
                else:
                    seq_m5_in = seq_m5
                m5_pool = self.m5_encoder(self._add_pe(self.m5_proj(seq_m5_in), "pos_enc_m5")).mean(dim=1)   # (B,d)
                pool_list.append(m5_pool)

            if self.cfg.enable_cross_tf_attn and hasattr(self, "cross_tf_attn"):
                # (B, n_tf, d): each TF pool is a token → cross-TF attention
                tf_tokens = torch.stack(pool_list, dim=1)
                tf_attended = self.cross_tf_attn(tf_tokens)            # (B, n_tf, d)
                gate = torch.softmax(self.tf_gate_logits, dim=0)       # (n_tf,) learnable
                pooled = (tf_attended * gate.view(1, -1, 1)).sum(dim=1)  # (B, d) gated combine
                # Forceful MTF→dir: capture the GATED, NON-zero-init repr (NOT
                # mtf_correction, which zero-init cross_tf_out damps to ~0).
                mtf_repr = pooled
                mtf_correction = self.cross_tf_out(pooled)
            else:
                mtf_combined = torch.cat(pool_list, dim=1)
                mtf_repr = None   # mtf_direction head requires the cross-TF branch (see __init__ guard)
                mtf_correction = self.multi_tf_fuse(mtf_combined)
            scale = float(self.multi_tf_scale.item())
            z = z_v3 + scale * mtf_correction
        else:
            z = z_v3   # v3-identical path
            mtf_repr = None

        # Regime FiLM (BIG-9): modulate a SEPARATE z_dir for the direction head only,
        # leaving z untouched for the aux heads + downstream. Zero-init -> z_dir==z at cold
        # start (bit-parity). cat_emb (B,ctx_cat_dim,emb) was computed above; flatten as the
        # regime conditioner (includes the repaired trend_regime_id/vol_regime slots).
        if bool(getattr(self.cfg, "enable_regime_film", False)) and hasattr(self, "regime_film"):
            film = self.regime_film(cat_emb.reshape(cat_emb.shape[0], -1))   # (B, 2*d_model)
            gamma, beta = film.chunk(2, dim=1)
            z_dir = (1.0 + gamma) * z + beta
        else:
            z_dir = z
        delta_logits = self.head_direction(z_dir)   # (B,3)
        anchor_logits = self._anchor_logits_from_snap(snap_x)
        anchor_gate = None
        if hasattr(self, "head_anchor_gate"):
            anchor_gate = torch.sigmoid(self.head_anchor_gate(z_dir))
            _assert_finite("anchor_gate", anchor_gate)
            anchor_mean = anchor_logits.mean(dim=1, keepdim=True)
            gated_anchor_logits = anchor_mean + anchor_gate * (anchor_logits - anchor_mean)
            direction_logits = gated_anchor_logits + (
                self.residual_scale.to(delta_logits.dtype) * delta_logits
            )
        else:
            direction_logits = anchor_logits + (self.residual_scale.to(delta_logits.dtype) * delta_logits)
        # Forceful MTF→dir (2026-06-06): dedicated, non-zeroable multi-TF term on
        # direction logits. Only when the head exists (cross-TF branch + enabled).
        # Emits mtf_dir_logits for the aux CE that forces the multi-TF repr to
        # predict direction (ENTRY_MTF_DIR_AUX_WEIGHT in the trainer).
        mtf_dir_logits = None
        if hasattr(self, "head_mtf_direction") and mtf_repr is not None:
            mtf_dir_logits = self.head_mtf_direction(mtf_repr)   # (B,3)
            _assert_finite("mtf_dir_logits", mtf_dir_logits)
            direction_logits = direction_logits + (
                self.mtf_dir_scale.to(direction_logits.dtype) * mtf_dir_logits
            )

        path_quality = self.head_path_quality(z)
        mfe_first_n = self.head_mfe_first_n(z)
        tradable_logit = self.head_tradable(z)
        bad_path_logit = self.head_bad_path(z)
        clean_edge_logit = self.head_clean_edge(z)
        survival_logit = self.head_survival(z)
        if self._path_cal is not None:
            _pq_a, _pq_b, _bp_t, _bp_b = self._path_cal
            path_quality = path_quality * _pq_a + _pq_b
            bad_path_logit = bad_path_logit / _bp_t + _bp_b
        _assert_finite("path_quality", path_quality)
        _assert_finite("mfe_first_n", mfe_first_n)
        _assert_finite("tradable_logit", tradable_logit)
        _assert_finite("bad_path_logit", bad_path_logit)
        _assert_finite("clean_edge_logit", clean_edge_logit)
        _assert_finite("survival_logit", survival_logit)

        raw_direction_logits = direction_logits
        out = {
            "direction_logits": direction_logits,
            "raw_direction_logits": raw_direction_logits,
            "anchor_logits": anchor_logits,
            "delta_logits": delta_logits,
            **({"anchor_gate": anchor_gate} if anchor_gate is not None else {}),
            **({"mtf_dir_logits": mtf_dir_logits} if mtf_dir_logits is not None else {}),
            "path_quality": path_quality,
            "mfe_first_n": mfe_first_n,
            "tradable_logit": tradable_logit,
            "bad_path_logit": bad_path_logit,
            "clean_edge_logit": clean_edge_logit,
            "survival_logit": survival_logit,
        }
        if specialist_gate is not None:
            out["specialist_gate"] = specialist_gate
        if self.cfg.enable_hierarchical_entry_heads and hasattr(self, "head_trade"):
            trade_logit = self.head_trade(z)
            public_trade_logit = self.head_public_trade(z) if hasattr(self, "head_public_trade") else trade_logit
            side_logits = self.head_side(z)
            hierarchical_ctx_prior = None
            if hasattr(self, "hierarchical_ctx_prior_adapter"):
                hierarchical_ctx_prior = self.hierarchical_ctx_prior_adapter(cat_flat)
                _assert_finite("hierarchical_ctx_prior", hierarchical_ctx_prior)
                prior_scale = float(getattr(self.cfg, "hierarchical_ctx_prior_adapter_scale", 0.0))
                trade_logit = trade_logit + prior_scale * hierarchical_ctx_prior[:, :1]
                if hasattr(self, "head_public_trade"):
                    public_trade_logit = public_trade_logit + prior_scale * hierarchical_ctx_prior[:, :1]
                side_logits = side_logits + prior_scale * hierarchical_ctx_prior[:, 1:3]
            side_utility = self.head_side_utility(z)
            side_bad_path_logit = self.head_side_bad_path(z)
            side_mae = self.head_side_mae(z)
            side_validity_logit = self.head_side_validity(z) if hasattr(self, "head_side_validity") else None
            _assert_finite("trade_logit", trade_logit)
            _assert_finite("side_logits", side_logits)
            _assert_finite("side_utility", side_utility)
            _assert_finite("side_bad_path_logit", side_bad_path_logit)
            _assert_finite("side_mae", side_mae)
            if side_validity_logit is not None:
                _assert_finite("side_validity_logit", side_validity_logit)
            if bool(getattr(self.cfg, "enable_hierarchical_direction_composition", False)):
                _assert_finite("public_trade_logit", public_trade_logit)
                public_side_logits = (
                    self.head_public_side(z)
                    if hasattr(self, "head_public_side")
                    else side_logits
                )
                _assert_finite("public_side_logits", public_side_logits)
                public_direction_composition = str(
                    getattr(self.cfg, "hierarchical_public_direction_composition", "logprob")
                ).strip().lower()
                if public_direction_composition in {"margin", "margin_centered", "margin_maxnorm"}:
                    public_trade_margin = public_trade_logit.reshape(-1)
                    public_side_for_composition = public_side_logits
                    if public_direction_composition == "margin_centered":
                        public_side_for_composition = (
                            public_side_logits - public_side_logits.mean(dim=1, keepdim=True)
                        )
                    elif public_direction_composition == "margin_maxnorm":
                        public_side_for_composition = (
                            public_side_logits - public_side_logits.max(dim=1, keepdim=True).values
                        )
                    composed_direction_logits = torch.stack(
                        (
                            public_trade_margin + public_side_for_composition[:, 0],
                            public_trade_margin + public_side_for_composition[:, 1],
                            -public_trade_margin,
                        ),
                        dim=1,
                    ).to(dtype=raw_direction_logits.dtype)
                else:
                    trade_log_prob = nn.functional.logsigmoid(public_trade_logit.reshape(-1))
                    flat_log_prob = nn.functional.logsigmoid(-public_trade_logit.reshape(-1))
                    side_log_probs = nn.functional.log_softmax(public_side_logits, dim=1)
                    composed_direction_logits = torch.stack(
                        (
                            trade_log_prob + side_log_probs[:, 0],
                            trade_log_prob + side_log_probs[:, 1],
                            flat_log_prob,
                        ),
                        dim=1,
                    ).to(dtype=raw_direction_logits.dtype)
                _assert_finite("composed_direction_logits", composed_direction_logits)
                residual_direction_logits = self.residual_scale.to(delta_logits.dtype) * delta_logits
                if bool(getattr(self.cfg, "hierarchical_composition_public_flat_from_trade", False)):
                    neutral_residual = residual_direction_logits.mean(dim=1)
                    residual_direction_logits = torch.stack(
                        (
                            neutral_residual,
                            neutral_residual,
                            neutral_residual,
                        ),
                        dim=1,
                    )
                elif bool(getattr(self.cfg, "hierarchical_composition_residual_side_neutral", False)):
                    trade_residual = residual_direction_logits[:, :2].mean(dim=1)
                    residual_direction_logits = torch.stack(
                        (
                            trade_residual,
                            trade_residual,
                            residual_direction_logits[:, 2],
                        ),
                        dim=1,
                    )
                residual_cap = self.hierarchical_composition_residual_logit_cap.to(
                    device=residual_direction_logits.device,
                    dtype=residual_direction_logits.dtype,
                )
                if bool((residual_cap > 0.0).item()):
                    residual_direction_logits = residual_cap * torch.tanh(
                        residual_direction_logits / residual_cap
                    )
                _assert_finite("hierarchical_residual_direction_logits", residual_direction_logits)
                direction_logits = composed_direction_logits + residual_direction_logits.to(
                    dtype=composed_direction_logits.dtype
                )
                if hasattr(self, "hierarchical_ctx_direction_calibration"):
                    ctx_direction_calibration_logits = self.hierarchical_ctx_direction_calibration(cat_flat)
                    _assert_finite(
                        "hierarchical_ctx_direction_calibration_logits",
                        ctx_direction_calibration_logits,
                    )
                    ctx_direction_calibration_logits = (
                        float(getattr(self.cfg, "hierarchical_ctx_direction_calibration_scale", 0.0))
                        * ctx_direction_calibration_logits
                    )
                    calibration_cap = float(
                        getattr(self.cfg, "hierarchical_ctx_direction_calibration_cap", 0.0)
                    )
                    if calibration_cap > 0.0:
                        calibration_cap_t = torch.as_tensor(
                            calibration_cap,
                            device=ctx_direction_calibration_logits.device,
                            dtype=ctx_direction_calibration_logits.dtype,
                        )
                        ctx_direction_calibration_logits = calibration_cap_t * torch.tanh(
                            ctx_direction_calibration_logits / calibration_cap_t
                        )
                    _assert_finite(
                        "hierarchical_ctx_direction_calibration_logits_capped",
                        ctx_direction_calibration_logits,
                    )
                    direction_logits = direction_logits + ctx_direction_calibration_logits.to(
                        dtype=direction_logits.dtype
                    )
                    out["hierarchical_ctx_direction_calibration_logits"] = (
                        ctx_direction_calibration_logits
                    )
                out["direction_logits"] = direction_logits
                out["hierarchical_direction_base_logits"] = composed_direction_logits
                out["hierarchical_direction_residual_logits"] = residual_direction_logits
                if hasattr(self, "head_public_side"):
                    out["public_side_logits"] = public_side_logits
                if hasattr(self, "head_public_trade"):
                    out["public_trade_logit"] = public_trade_logit
                out["hierarchical_direction_residual_side_neutral"] = torch.full(
                    (direction_logits.shape[0], 1),
                    1.0
                    if (
                        bool(getattr(self.cfg, "hierarchical_composition_residual_side_neutral", False))
                        or bool(getattr(self.cfg, "hierarchical_composition_public_flat_from_trade", False))
                    )
                    else 0.0,
                    device=direction_logits.device,
                    dtype=direction_logits.dtype,
                )
                out["hierarchical_direction_public_flat_from_trade"] = torch.full(
                    (direction_logits.shape[0], 1),
                    1.0 if bool(getattr(self.cfg, "hierarchical_composition_public_flat_from_trade", False)) else 0.0,
                    device=direction_logits.device,
                    dtype=direction_logits.dtype,
                )
                out["hierarchical_public_direction_composition_margin"] = torch.full(
                    (direction_logits.shape[0], 1),
                    1.0 if public_direction_composition in {"margin", "margin_centered", "margin_maxnorm"} else 0.0,
                    device=direction_logits.device,
                    dtype=direction_logits.dtype,
                )
                out["hierarchical_public_direction_composition_margin_centered"] = torch.full(
                    (direction_logits.shape[0], 1),
                    1.0 if public_direction_composition == "margin_centered" else 0.0,
                    device=direction_logits.device,
                    dtype=direction_logits.dtype,
                )
                out["hierarchical_public_direction_composition_margin_maxnorm"] = torch.full(
                    (direction_logits.shape[0], 1),
                    1.0 if public_direction_composition == "margin_maxnorm" else 0.0,
                    device=direction_logits.device,
                    dtype=direction_logits.dtype,
                )
                out["hierarchical_direction_composed"] = torch.ones(
                    (direction_logits.shape[0], 1),
                    device=direction_logits.device,
                    dtype=direction_logits.dtype,
                )
            out.update(
                {
                    "trade_logit": trade_logit,
                    "side_logits": side_logits,
                    "side_utility": side_utility,
                    "side_bad_path_logit": side_bad_path_logit,
                    "side_mae": side_mae,
                    **(
                        {"hierarchical_ctx_prior": hierarchical_ctx_prior}
                        if hierarchical_ctx_prior is not None
                        else {}
                    ),
                    **({"side_validity_logit": side_validity_logit} if side_validity_logit is not None else {}),
                }
            )
        # Post-hoc direction calibration (identity unless the bundle loader
        # installed metadata-fitted values). Applied after optional hierarchical
        # composition so audit, replay and live serve consume the same public
        # direction_logits.
        if self._direction_cal is not None:
            _cal_t, _cal_b = self._direction_cal
            direction_logits = direction_logits / _cal_t + _cal_b.to(
                device=direction_logits.device, dtype=direction_logits.dtype
            )
            out["direction_logits"] = direction_logits

        _assert_finite("direction_logits", direction_logits)
        if self.cfg.enable_trendline_rail_head and hasattr(self, "head_trendline_rail"):
            trendline_rail_logits = self.head_trendline_rail(z)
            _assert_finite("trendline_rail_logits", trendline_rail_logits)
            out["trendline_rail_logits"] = trendline_rail_logits
        # Distillation Q-head — only emit when enabled in this bundle.
        if self.cfg.enable_q_head and hasattr(self, "q_head"):
            q_per_action = self.q_head(z)  # (B, 3) — [q_skip, q_long, q_short]
            _assert_finite("q_per_action", q_per_action)
            out["q_per_action"] = q_per_action
        # TF-agreement head — only emit when enabled in this bundle.
        if self.cfg.enable_tf_agreement_head and hasattr(self, "head_tf_agreement"):
            tf_agreement_logit = self.head_tf_agreement(z)  # (B, 1) — raw logit
            _assert_finite("tf_agreement_logit", tf_agreement_logit)
            out["tf_agreement_logit"] = tf_agreement_logit
        # Path-quality variance head — only emit when enabled in this bundle.
        if self.cfg.enable_path_quality_variance_head and hasattr(self, "head_path_quality_log_var"):
            path_quality_log_var = self.head_path_quality_log_var(z)  # (B, 1)
            _assert_finite("path_quality_log_var", path_quality_log_var)
            out["path_quality_log_var"] = path_quality_log_var
        # Position-size head — only emit when enabled in this bundle.
        if self.cfg.enable_position_size_head and hasattr(self, "head_position_size"):
            position_size_logit = self.head_position_size(z)  # (B, 1) — raw logit
            _assert_finite("position_size_logit", position_size_logit)
            out["position_size_logit"] = position_size_logit
        # Hold-horizon head — only emit when enabled in this bundle.
        if self.cfg.enable_hold_horizon_head and hasattr(self, "head_hold_horizon"):
            hold_horizon_logit = self.head_hold_horizon(z)  # (B, 1) — raw logit
            _assert_finite("hold_horizon_logit", hold_horizon_logit)
            out["hold_horizon_logit"] = hold_horizon_logit
        if self.cfg.enable_dip_head and hasattr(self, "head_dip"):
            dip_pred = self.head_dip(z)  # (B, 18) — dip risk profile (see DIP_* layout)
            _assert_finite("dip_pred", dip_pred)
            out["dip_pred"] = dip_pred
        if self.cfg.enable_forecast_head and hasattr(self, "head_forecast"):
            forecast_pred = self.head_forecast(z)  # (B, 4) — cum future ret (bps) @ FORECAST_HORIZONS
            _assert_finite("forecast_pred", forecast_pred)
            out["forecast_pred"] = forecast_pred
        if self.cfg.enable_timing_head and hasattr(self, "head_timing"):
            timing_pred = self.head_timing(z)  # (B, 12) — dip-bottom/peak bar-frac (see TIMING_* layout)
            _assert_finite("timing_pred", timing_pred)
            out["timing_pred"] = timing_pred
        if self.cfg.enable_tail_risk_head and hasattr(self, "head_tail_risk"):
            tail_risk_pred = self.head_tail_risk(z)  # (B, 6) — p90 worst-adverse (see TAIL_RISK_* layout)
            _assert_finite("tail_risk_pred", tail_risk_pred)
            out["tail_risk_pred"] = tail_risk_pred
        if self.cfg.enable_vol_forecast_head and hasattr(self, "head_vol_forecast"):
            vol_forecast_pred = self.head_vol_forecast(z)  # (B, 3) — fwd realized vol (bps) @ VOL_FORECAST_HORIZONS
            _assert_finite("vol_forecast_pred", vol_forecast_pred)
            out["vol_forecast_pred"] = vol_forecast_pred
        return out
