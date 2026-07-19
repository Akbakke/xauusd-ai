# gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    HIDDEN_DIM as EXACT_EVIDENCE_FUSION_HIDDEN_DIM,
    INPUT_DIM as EXACT_EVIDENCE_FUSION_INPUT_DIM,
    INPUTS as EXACT_EVIDENCE_FUSION_OUTPUTS,
)


def _assert_shape(name: str, t: torch.Tensor, nd: int) -> None:
    if not isinstance(t, torch.Tensor):
        raise RuntimeError(f"TYPE_MISMATCH: {name} is not a torch.Tensor (got {type(t)})")
    if t.dim() != nd:
        raise RuntimeError(f"SHAPE_MISMATCH: {name}.dim={t.dim()} expected={nd} shape={tuple(t.shape)}")


def _assert_finite(name: str, t: torch.Tensor) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NONFINITE: {name} contains NaN/Inf")


EXACT_TRENDLINE_RAIL_OUTPUT_DIM = 6
EXACT_SPECIALIST_NAMES = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)

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
# labels). Forces the representation to capture forward price dynamics for the
# shared representation and every learned evidence head.
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
    """Configuration for the one supported Entry model-native architecture.

    There are deliberately no component/head enable flags.  Every instance has
    the full M5/M15/H1/H4/D1 stack, positional encoding, cross-TF attention,
    learnable TF scales, regime FiLM, eight specialists and every supervised
    evidence head.  Values here tune dimensions/scales; they cannot remove a
    decision-path component.
    """

    seq_input_dim: int
    snap_input_dim: int
    seq_len: int
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: Optional[int] = None
    dropout: float = 0.05
    ctx_cat_dim: int = MODEL_NATIVE_CTX_CAT_DIM
    ctx_cont_dim: int = MODEL_NATIVE_CTX_CONT_DIM
    # simple, robust embedding: one shared vocab for all ctx_cat slots
    ctx_cat_vocab: int = 1024
    ctx_cat_emb_dim: int = 8
    # Keep ctx as correction, not primary driver
    ctx_cat_scale: float = 0.25
    ctx_cont_scale: float = 0.25
    # Exact M5/M15/H1/H4/D1 input contract.
    m15_seq_dim: int = 0
    h1_seq_dim: int = 0
    h4_seq_dim: int = 0
    d1_seq_dim: int = 0
    m15_seq_len: int = 96        # ~24 hours at M15 cadence
    h1_seq_len: int = 96         # ~4 days at H1 cadence
    h4_seq_len: int = 96         # ~16 days at H4 cadence
    d1_seq_len: int = 96         # ~3 months at D1 cadence
    multi_tf_num_layers: int = 2 # smaller encoders per TF (lower TF count → less compute)
    multi_tf_scale: float = 0.5  # fixed positive multi-TF representation scale
    # Per-TF learnable input-scaling priors.
    tf_input_scale_init_m5: float = 1.0
    tf_input_scale_init_m15: float = 1.0
    tf_input_scale_init_h1: float = 0.7
    tf_input_scale_init_h4: float = 0.5
    tf_input_scale_init_d1: float = 0.3
    # The exact path receives seq_x (513 model-native signals × 96 M5 bars)
    # plus separately declared M5/M15/H1/H4/D1 tensors. Each timeframe width is
    # artifact-bound; none is inferred from these defaults or silently padded.
    m5_seq_dim: int = 0
    m5_seq_len: int = 96         # ~8 hours at M5 cadence
    # Mandatory specialist and fusion hyperparameters.
    specialist_num_layers: int = 1
    specialist_fusion_scale: float = 0.25


class EntryV10CtxHybridTransformer(nn.Module):
    """
    Minimal, strict CTX model used by:
      - gx1/models/entry_v10/entry_v10_bundle.py
      - gx1/rl/entry_v10/train_entry_transformer_v10.py (CTX variant)

    Forward signature (expected by docs/usage):
        out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
        out["direction_logits"]  -> (B, 3)  # classes: 0=LONG, 1=SHORT, 2=FLAT
        out["public_trade_flat_decision_logits"] -> (B, 2)  # TRADE=max(LONG,SHORT), FLAT
        out["path_quality"]      -> (B, 1)  # auxiliary learned path evidence
        out["mfe_first_n"]       -> (B, 1)  # auxiliary learned excursion evidence
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
        ctx_cont_dim: int = MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim: int = MODEL_NATIVE_CTX_CAT_DIM,
        m15_seq_dim: int,
        h1_seq_dim: int,
        h4_seq_dim: int,
        d1_seq_dim: int,
        m15_seq_len: int = 96,
        h1_seq_len: int = 96,
        h4_seq_len: int = 96,
        d1_seq_len: int = 96,
        multi_tf_num_layers: int = 2,
        multi_tf_scale: float = 0.5,
        m5_seq_dim: int,
        m5_seq_len: int = 96,
        specialist_input_indices: Dict[str, list[int]],
        specialist_num_layers: int = 1,
        specialist_fusion_scale: float = 0.25,
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
        if (
            int(ctx_cont_dim) != MODEL_NATIVE_CTX_CONT_DIM
            or int(ctx_cat_dim) != MODEL_NATIVE_CTX_CAT_DIM
        ):
            raise RuntimeError(
                "[ENTRY_MODEL_NATIVE_CTX_DIM_INVALID] "
                f"ctx_cont_dim={int(ctx_cont_dim)} expected={MODEL_NATIVE_CTX_CONT_DIM} "
                f"ctx_cat_dim={int(ctx_cat_dim)} expected={MODEL_NATIVE_CTX_CAT_DIM}"
            )
        if min(m5_seq_dim, m15_seq_dim, h1_seq_dim, h4_seq_dim, d1_seq_dim) <= 0:
            raise RuntimeError(
                "MULTI_TF_DIM_INVALID: exact architecture requires positive "
                f"M5/M15/H1/H4/D1 dims; got m5={m5_seq_dim} m15={m15_seq_dim} "
                f"h1={h1_seq_dim} h4={h4_seq_dim} d1={d1_seq_dim}"
            )
        mandatory_positive_scales = {
            "multi_tf_scale": multi_tf_scale,
            "specialist_fusion_scale": specialist_fusion_scale,
            "tf_input_scale_init_m5": tf_input_scale_init_m5,
            "tf_input_scale_init_m15": tf_input_scale_init_m15,
            "tf_input_scale_init_h1": tf_input_scale_init_h1,
            "tf_input_scale_init_h4": tf_input_scale_init_h4,
            "tf_input_scale_init_d1": tf_input_scale_init_d1,
        }
        invalid_scales = {
            name: value
            for name, value in mandatory_positive_scales.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        }
        if invalid_scales:
            raise RuntimeError(
                "MANDATORY_REPRESENTATION_SCALE_INVALID: exact architecture requires "
                f"finite positive scales; got {invalid_scales}"
            )
        self.cfg = CtxModelConfig(
            seq_input_dim=seq_input_dim,
            snap_input_dim=snap_input_dim,
            seq_len=seq_len,
            ctx_cont_dim=int(ctx_cont_dim),
            ctx_cat_dim=int(ctx_cat_dim),
            m15_seq_dim=int(m15_seq_dim),
            h1_seq_dim=int(h1_seq_dim),
            h4_seq_dim=int(h4_seq_dim),
            d1_seq_dim=int(d1_seq_dim),
            m15_seq_len=int(m15_seq_len),
            h1_seq_len=int(h1_seq_len),
            h4_seq_len=int(h4_seq_len),
            d1_seq_len=int(d1_seq_len),
            m5_seq_dim=int(m5_seq_dim),
            m5_seq_len=int(m5_seq_len),
            multi_tf_num_layers=int(multi_tf_num_layers),
            multi_tf_scale=float(multi_tf_scale),
            specialist_num_layers=int(specialist_num_layers),
            specialist_fusion_scale=float(specialist_fusion_scale),
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

        # Exact eight-specialist contract.  Do not normalize malformed input:
        # names, index types, uniqueness and range are all evidence contracts.
        if tuple(specialist_input_indices) != EXACT_SPECIALIST_NAMES:
            raise RuntimeError(
                "SPECIALIST_NAMES_MISMATCH: "
                f"got={tuple(specialist_input_indices)} expected={EXACT_SPECIALIST_NAMES}"
            )
        cleaned: Dict[str, list[int]] = {}
        seen_indices: set[int] = set()
        for name in EXACT_SPECIALIST_NAMES:
            raw_idx = specialist_input_indices[name]
            if not isinstance(raw_idx, list) or not raw_idx:
                raise RuntimeError(f"SPECIALIST_INDICES_INVALID: {name}")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_idx):
                raise RuntimeError(f"SPECIALIST_INDEX_TYPE_INVALID: {name}")
            if raw_idx != sorted(raw_idx) or len(raw_idx) != len(set(raw_idx)):
                raise RuntimeError(f"SPECIALIST_INDEX_ORDER_OR_DUPLICATE: {name}")
            if min(raw_idx) < 0 or max(raw_idx) >= int(seq_input_dim):
                raise RuntimeError(
                    f"SPECIALIST_INDEX_OOB: {name} has indices outside [0,{int(seq_input_dim) - 1}]"
                )
            overlap = seen_indices.intersection(raw_idx)
            if overlap:
                raise RuntimeError(f"SPECIALIST_INDEX_OVERLAP: {name} overlap={sorted(overlap)}")
            seen_indices.update(raw_idx)
            cleaned[name] = list(raw_idx)
        self._specialist_names = EXACT_SPECIALIST_NAMES
        self.specialist_proj = nn.ModuleDict(
            {name: nn.Linear(len(idx), d_model) for name, idx in cleaned.items()}
        )
        specialist_layers = int(self.cfg.specialist_num_layers)
        if specialist_layers <= 0:
            raise RuntimeError("SPECIALIST_NUM_LAYERS_INVALID")

        def _mk_encoder(layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=layers)

        self.specialist_encoder = nn.ModuleDict(
            {name: _mk_encoder(specialist_layers) for name in self._specialist_names}
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
        self.register_buffer(
            "specialist_fusion_scale",
            torch.tensor(float(self.cfg.specialist_fusion_scale)),
        )

        # Exact direction and supervised evidence heads.
        self.head_direction = nn.Linear(d_model, 3)
        self._direction_cal: Optional[Tuple[float, torch.Tensor]] = None
        self._path_cal: Optional[Tuple[float, float, float, float]] = None
        self.regime_film = nn.Sequential(
            nn.Linear(ctx_cat_flat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.regime_film[-1].weight)
        nn.init.zeros_(self.regime_film[-1].bias)

        self.head_path_quality = nn.Linear(d_model, 1)
        self.head_mfe_first_n = nn.Linear(d_model, 1)
        self.head_tradable = nn.Linear(d_model, 1)
        self.head_bad_path = nn.Linear(d_model, 1)
        self.head_clean_edge = nn.Linear(d_model, 1)
        self.head_survival = nn.Linear(d_model, 1)
        self.head_trade = nn.Linear(d_model, 1)
        self.head_side = nn.Linear(d_model, 2)
        self.head_side_utility = nn.Linear(d_model, 2)
        self.head_side_bad_path = nn.Linear(d_model, 2)
        self.head_side_mae = nn.Linear(d_model, 2)
        self.head_side_validity = nn.Linear(d_model, 2)
        self.head_trendline_rail = nn.Linear(d_model, EXACT_TRENDLINE_RAIL_OUTPUT_DIM)
        self.head_mtf_direction = nn.Linear(d_model, 3)
        self.head_tf_agreement = nn.Linear(d_model, 1)
        self.head_path_quality_log_var = nn.Linear(d_model, 1)
        self.head_position_size = nn.Linear(d_model, 1)
        self.head_dip = nn.Linear(d_model, DIP_HEAD_DIM)
        self.head_forecast = nn.Linear(d_model, FORECAST_HEAD_DIM)
        self.head_timing = nn.Linear(d_model, TIMING_HEAD_DIM)
        self.head_tail_risk = nn.Linear(d_model, TAIL_RISK_HEAD_DIM)
        self.head_vol_forecast = nn.Linear(d_model, VOL_FORECAST_HEAD_DIM)
        self.evidence_fusion_norm = nn.LayerNorm(EXACT_EVIDENCE_FUSION_INPUT_DIM)
        self.evidence_fusion_in = nn.Linear(
            EXACT_EVIDENCE_FUSION_INPUT_DIM, EXACT_EVIDENCE_FUSION_HIDDEN_DIM
        )
        self.evidence_fusion_out = nn.Linear(EXACT_EVIDENCE_FUSION_HIDDEN_DIM, 3)

        nn.init.zeros_(self.head_trade.bias)
        nn.init.zeros_(self.head_side.bias)
        for head in (self.head_side_utility, self.head_side_bad_path, self.head_side_mae):
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.head_side_validity.bias)
        nn.init.zeros_(self.head_trendline_rail.bias)
        nn.init.normal_(self.head_mtf_direction.weight, std=0.02)
        nn.init.zeros_(self.head_mtf_direction.bias)
        nn.init.zeros_(self.head_path_quality_log_var.bias)
        nn.init.xavier_uniform_(self.evidence_fusion_in.weight)
        nn.init.zeros_(self.evidence_fusion_in.bias)
        nn.init.xavier_uniform_(self.evidence_fusion_out.weight)
        nn.init.zeros_(self.evidence_fusion_out.bias)

        # Exact five-timeframe stack with cross-TF attention and learnable scales.
        mtf_layers = int(self.cfg.multi_tf_num_layers)
        if mtf_layers <= 0:
            raise RuntimeError("MULTI_TF_NUM_LAYERS_INVALID")
        self.m5_proj = nn.Linear(int(self.cfg.m5_seq_dim), d_model)
        self.m15_proj = nn.Linear(int(self.cfg.m15_seq_dim), d_model)
        self.h1_proj = nn.Linear(int(self.cfg.h1_seq_dim), d_model)
        self.h4_proj = nn.Linear(int(self.cfg.h4_seq_dim), d_model)
        self.d1_proj = nn.Linear(int(self.cfg.d1_seq_dim), d_model)
        self.m5_encoder = _mk_encoder(mtf_layers)
        self.m15_encoder = _mk_encoder(mtf_layers)
        self.h1_encoder = _mk_encoder(mtf_layers)
        self.h4_encoder = _mk_encoder(mtf_layers)
        self.d1_encoder = _mk_encoder(mtf_layers)
        self.cross_tf_attn = _mk_encoder(1)
        self.tf_gate_logits = nn.Parameter(torch.zeros(5))
        self.cross_tf_out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.cross_tf_out.weight)
        nn.init.zeros_(self.cross_tf_out.bias)
        self._expected_m5_seq_dim = int(self.cfg.m5_seq_dim)
        self._expected_m15_seq_dim = int(self.cfg.m15_seq_dim)
        self._expected_h1_seq_dim = int(self.cfg.h1_seq_dim)
        self._expected_h4_seq_dim = int(self.cfg.h4_seq_dim)
        self._expected_d1_seq_dim = int(self.cfg.d1_seq_dim)
        self.register_buffer("multi_tf_scale", torch.tensor(float(self.cfg.multi_tf_scale)))
        self.tf_input_scale_m5 = nn.Parameter(torch.tensor(float(self.cfg.tf_input_scale_init_m5)))
        self.tf_input_scale_m15 = nn.Parameter(torch.tensor(float(self.cfg.tf_input_scale_init_m15)))
        self.tf_input_scale_h1 = nn.Parameter(torch.tensor(float(self.cfg.tf_input_scale_init_h1)))
        self.tf_input_scale_h4 = nn.Parameter(torch.tensor(float(self.cfg.tf_input_scale_init_h4)))
        self.tf_input_scale_d1 = nn.Parameter(torch.tensor(float(self.cfg.tf_input_scale_init_d1)))

        # Strict markers (useful for debugging)
        self._expected_seq_dim = int(seq_input_dim)
        self._expected_snap_dim = int(snap_input_dim)
        self._expected_seq_len = int(seq_len)
        self._expected_ctx_cat_dim = int(self.cfg.ctx_cat_dim)
        self._expected_ctx_cont_dim = int(self.cfg.ctx_cont_dim)
        # Positional encoding is mandatory and covers every sequence branch.
        self.register_buffer("pos_enc", self._sinusoidal_pe(int(seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_m5", self._sinusoidal_pe(int(self.cfg.m5_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_m15", self._sinusoidal_pe(int(self.cfg.m15_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_h1", self._sinusoidal_pe(int(self.cfg.h1_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_h4", self._sinusoidal_pe(int(self.cfg.h4_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_d1", self._sinusoidal_pe(int(self.cfg.d1_seq_len), d_model), persistent=False)

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
        """Add the mandatory positional encoding for this sequence branch."""
        pe = getattr(self, buf_name)
        return t + pe[:, : t.size(1)]

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
        Identity when never called. Calibration is report-only for direction
        fusion, which always consumes the raw pre-calibration tensors."""
        vals = (float(path_quality_scale), float(path_quality_shift), float(bad_path_temperature), float(bad_path_bias))
        import math as _math
        if not all(_math.isfinite(v) for v in vals) or vals[0] <= 0.0 or vals[2] <= 0.0:
            raise ValueError(f"[ENTRY_PATH_CAL] invalid calibration: {vals}")
        self._path_cal = vals

    def _assemble_direction_evidence(
        self,
        pre_fusion_outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Assemble the immutable ordered 75-wide evidence contract."""
        evidence_parts: list[torch.Tensor] = []
        batch_size: Optional[int] = None
        for output_name, expected_width in EXACT_EVIDENCE_FUSION_OUTPUTS:
            if output_name not in pre_fusion_outputs:
                raise RuntimeError(
                    f"EVIDENCE_FUSION_INPUT_MISSING: output={output_name}"
                )
            value = pre_fusion_outputs[output_name]
            if value.ndim != 2 or int(value.shape[1]) != expected_width:
                raise RuntimeError(
                    "EVIDENCE_FUSION_INPUT_SHAPE_INVALID: "
                    f"{output_name} shape={tuple(value.shape)} expected=(B,{expected_width})"
                )
            if batch_size is None:
                batch_size = int(value.shape[0])
            elif int(value.shape[0]) != batch_size:
                raise RuntimeError(
                    "EVIDENCE_FUSION_BATCH_MISMATCH: "
                    f"{output_name} rows={int(value.shape[0])} expected={batch_size}"
                )
            _assert_finite(f"evidence_fusion.{output_name}", value)
            evidence_parts.append(value)
        evidence_vector = torch.cat(evidence_parts, dim=1)
        if int(evidence_vector.shape[1]) != EXACT_EVIDENCE_FUSION_INPUT_DIM:
            raise RuntimeError(
                "EVIDENCE_FUSION_WIDTH_INVALID: "
                f"got={int(evidence_vector.shape[1])} "
                f"expected={EXACT_EVIDENCE_FUSION_INPUT_DIM}"
            )
        return evidence_vector

    def _fuse_direction_evidence(
        self,
        pre_fusion_outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        evidence_vector = self._assemble_direction_evidence(pre_fusion_outputs)
        evidence_hidden = nn.functional.gelu(
            self.evidence_fusion_in(self.evidence_fusion_norm(evidence_vector))
        )
        raw_direction_logits = self.evidence_fusion_out(evidence_hidden)
        _assert_finite("raw_direction_logits", raw_direction_logits)
        return raw_direction_logits

    def forward(
        self,
        seq_x: torch.Tensor,
        snap_x: torch.Tensor,
        *,
        ctx_cat: torch.Tensor,
        ctx_cont: torch.Tensor,
        seq_m5: torch.Tensor,
        seq_m15: torch.Tensor,
        seq_h1: torch.Tensor,
        seq_h4: torch.Tensor,
        seq_d1: torch.Tensor,
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
        seq_h = self._add_pe(seq_h, "pos_enc")        # mandatory temporal order
        seq_h = self.encoder(seq_h)                   # (B,T,d)
        seq_pool = seq_h.mean(dim=1)                  # (B,d)

        snap_h = self.snap_proj(snap_x)               # (B,d)

        cat_emb = self.ctx_cat_emb(ctx_cat.long())    # (B,ctx_cat_dim,emb)
        cat_flat = cat_emb.reshape(B, -1)             # (B,ctx_cat_dim*emb)

        cont_h = self.ctx_cont_proj(ctx_cont.float()) # (B,d)
        cat_flat = cat_flat * float(self.cfg.ctx_cat_scale)
        cont_h = cont_h * float(self.cfg.ctx_cont_scale)

        fused = torch.cat([seq_pool, snap_h, cat_flat, cont_h], dim=1)
        z_v3 = self.fuse(fused)

        pools = []
        for name in self._specialist_names:
            idx = getattr(self, f"specialist_idx_{name}").to(seq_x.device)
            seq_part = seq_x.index_select(dim=2, index=idx)
            spec_h = self._add_pe(self.specialist_proj[name](seq_part), "pos_enc")
            pools.append(self.specialist_encoder[name](spec_h).mean(dim=1))
        specialist_tokens = torch.stack(pools, dim=1)
        specialist_gate = torch.softmax(self.specialist_gate(z_v3), dim=1)
        specialist_pool = (specialist_tokens * specialist_gate.unsqueeze(-1)).sum(dim=1)
        specialist_correction = self.specialist_out(specialist_pool)
        _assert_finite("specialist_correction", specialist_correction)
        _assert_finite("specialist_gate", specialist_gate)
        z_v3 = z_v3 + self.specialist_fusion_scale.to(specialist_correction.dtype) * specialist_correction

        # Exact five-timeframe second-stage fusion.  Every branch is required;
        # there is no single-TF or four-TF fallback.
        tf_inputs = (
            ("seq_m5", seq_m5, self.cfg.m5_seq_len, self._expected_m5_seq_dim),
            ("seq_m15", seq_m15, self.cfg.m15_seq_len, self._expected_m15_seq_dim),
            ("seq_h1", seq_h1, self.cfg.h1_seq_len, self._expected_h1_seq_dim),
            ("seq_h4", seq_h4, self.cfg.h4_seq_len, self._expected_h4_seq_dim),
            ("seq_d1", seq_d1, self.cfg.d1_seq_len, self._expected_d1_seq_dim),
        )
        for name, tensor, exp_len, exp_dim in tf_inputs:
            _assert_shape(name, tensor, 3)
            if int(tensor.shape[0]) != B:
                raise RuntimeError(f"{name.upper()}_BATCH_MISMATCH")
            if int(tensor.shape[1]) != int(exp_len):
                raise RuntimeError(
                    f"{name.upper()}_LEN_MISMATCH: got={int(tensor.shape[1])} expected={exp_len}"
                )
            if int(tensor.shape[2]) != int(exp_dim):
                raise RuntimeError(
                    f"{name.upper()}_DIM_MISMATCH: got={int(tensor.shape[2])} expected={exp_dim}"
                )
            _assert_finite(name, tensor)

        pool_list = []
        for name, tensor, _exp_len, _exp_dim in tf_inputs:
            suffix = name.removeprefix("seq_")
            scaled = tensor * getattr(self, f"tf_input_scale_{suffix}")
            projected = getattr(self, f"{suffix}_proj")(scaled)
            encoded = getattr(self, f"{suffix}_encoder")(
                self._add_pe(projected, f"pos_enc_{suffix}")
            )
            pool_list.append(encoded.mean(dim=1))
        tf_tokens = torch.stack(pool_list, dim=1)
        tf_attended = self.cross_tf_attn(tf_tokens)
        tf_gate = torch.softmax(self.tf_gate_logits, dim=0)
        mtf_repr = (tf_attended * tf_gate.view(1, -1, 1)).sum(dim=1)
        mtf_correction = self.cross_tf_out(mtf_repr)
        _assert_finite("mtf_repr", mtf_repr)
        _assert_finite("mtf_correction", mtf_correction)
        z = z_v3 + self.multi_tf_scale.to(mtf_correction.dtype) * mtf_correction

        # Regime FiLM (BIG-9): modulate a SEPARATE z_dir for the direction head only,
        # leaving z untouched for the aux heads + downstream. Zero-init -> z_dir==z at cold
        # start (bit-parity). cat_emb (B,ctx_cat_dim,emb) was computed above; flatten as the
        # Learned regime conditioning from the exact categorical context tensor.
        film = self.regime_film(cat_emb.reshape(cat_emb.shape[0], -1))
        gamma, beta = film.chunk(2, dim=1)
        z_dir = (1.0 + gamma) * z + beta
        model_native_logits = self.head_direction(z_dir)   # (B,3)
        mtf_dir_logits = self.head_mtf_direction(mtf_repr)
        _assert_finite("model_native_logits", model_native_logits)
        _assert_finite("mtf_dir_logits", mtf_dir_logits)

        path_quality_raw = self.head_path_quality(z)
        mfe_first_n = self.head_mfe_first_n(z)
        tradable_logit = self.head_tradable(z)
        bad_path_logit_raw = self.head_bad_path(z)
        clean_edge_logit = self.head_clean_edge(z)
        survival_logit = self.head_survival(z)
        path_quality = path_quality_raw
        bad_path_logit = bad_path_logit_raw
        if self._path_cal is not None:
            _pq_a, _pq_b, _bp_t, _bp_b = self._path_cal
            path_quality = path_quality * _pq_a + _pq_b
            bad_path_logit = bad_path_logit / _bp_t + _bp_b
        _assert_finite("path_quality_raw", path_quality_raw)
        _assert_finite("bad_path_logit_raw", bad_path_logit_raw)
        _assert_finite("path_quality", path_quality)
        _assert_finite("mfe_first_n", mfe_first_n)
        _assert_finite("tradable_logit", tradable_logit)
        _assert_finite("bad_path_logit", bad_path_logit)
        _assert_finite("clean_edge_logit", clean_edge_logit)
        _assert_finite("survival_logit", survival_logit)

        out = {
            "model_native_logits": model_native_logits,
            "mtf_dir_logits": mtf_dir_logits,
            "path_quality_raw": path_quality_raw,
            "path_quality": path_quality,
            "mfe_first_n": mfe_first_n,
            "tradable_logit": tradable_logit,
            "bad_path_logit_raw": bad_path_logit_raw,
            "bad_path_logit": bad_path_logit,
            "clean_edge_logit": clean_edge_logit,
            "survival_logit": survival_logit,
        }
        out["specialist_gate"] = specialist_gate
        trade_logit = self.head_trade(z)
        side_logits = self.head_side(z)

        side_utility = self.head_side_utility(z)
        side_bad_path_logit = self.head_side_bad_path(z)
        side_mae = self.head_side_mae(z)
        side_validity_logit = self.head_side_validity(z)

        for output_name, value in (
            ("trade_logit", trade_logit),
            ("side_logits", side_logits),
            ("side_utility", side_utility),
            ("side_bad_path_logit", side_bad_path_logit),
            ("side_mae", side_mae),
            ("side_validity_logit", side_validity_logit),
        ):
            _assert_finite(output_name, value)

        # Compute every remaining supervised head from the pre-fusion shared
        # representation.  These raw causal outputs then enter exactly one
        # learned three-class evidence projector; the projector never consumes
        # final direction logits or post-fit path calibration.
        exact_outputs = {
            "trendline_rail_logits": self.head_trendline_rail(z),
            "tf_agreement_logit": self.head_tf_agreement(z),
            "path_quality_log_var": self.head_path_quality_log_var(z),
            "position_size_logit": self.head_position_size(z),
            "dip_pred": self.head_dip(z),
            "forecast_pred": self.head_forecast(z),
            "timing_pred": self.head_timing(z),
            "tail_risk_pred": self.head_tail_risk(z),
            "vol_forecast_pred": self.head_vol_forecast(z),
        }
        for output_name, value in exact_outputs.items():
            _assert_finite(output_name, value)

        pre_fusion_outputs = {
            "model_native_logits": model_native_logits,
            "mtf_dir_logits": mtf_dir_logits,
            "path_quality_raw": path_quality_raw,
            "mfe_first_n": mfe_first_n,
            "tradable_logit": tradable_logit,
            "bad_path_logit_raw": bad_path_logit_raw,
            "clean_edge_logit": clean_edge_logit,
            "survival_logit": survival_logit,
            "trade_logit": trade_logit,
            "side_logits": side_logits,
            "side_utility": side_utility,
            "side_bad_path_logit": side_bad_path_logit,
            "side_mae": side_mae,
            "side_validity_logit": side_validity_logit,
            **exact_outputs,
        }
        raw_direction_logits = self._fuse_direction_evidence(pre_fusion_outputs)

        # Sole model-native decision path.  No sibling head, context head or
        # base-direction head can bypass this learned fusion, and no handwritten
        # sign/penalty/cap changes a class margin.
        direction_logits = raw_direction_logits

        out.update(
            {
                "direction_logits": direction_logits,
                "raw_direction_logits": raw_direction_logits,
                "trade_logit": trade_logit,
                "side_logits": side_logits,
                "side_utility": side_utility,
                "side_bad_path_logit": side_bad_path_logit,
                "side_mae": side_mae,
                "side_validity_logit": side_validity_logit,
            }
        )
        out.update(exact_outputs)
        # Post-hoc direction calibration (identity unless the bundle loader
        # installed metadata-fitted values). It is fitted only after training and
        # applied after the sole learned fusion for audit/live parity.
        if self._direction_cal is not None:
            _cal_t, _cal_b = self._direction_cal
            direction_logits = direction_logits / _cal_t + _cal_b.to(
                device=direction_logits.device, dtype=direction_logits.dtype
            )
            out["direction_logits"] = direction_logits

        _assert_finite("direction_logits", direction_logits)
        # Canonical public trade-vs-FLAT decision surface. This is deliberately
        # derived from the final LONG/SHORT/FLAT logits after the sole learned
        # fusion and immutable fitted calibration so training guards cannot prove a
        # different binary policy than the public three-class output.
        public_trade_flat_decision_logits = torch.stack(
            (
                direction_logits[:, :2].max(dim=1).values,
                direction_logits[:, 2],
            ),
            dim=1,
        )
        _assert_finite(
            "public_trade_flat_decision_logits",
            public_trade_flat_decision_logits,
        )
        out["public_trade_flat_decision_logits"] = public_trade_flat_decision_logits
        return out
