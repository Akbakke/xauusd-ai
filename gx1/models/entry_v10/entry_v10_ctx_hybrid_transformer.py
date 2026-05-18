# gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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
    # ── Distillation Q-head (V13 prep) ────────────────────────────────
    # When enabled, adds nn.Linear(d_model, 3) producing q_per_action that
    # mirrors Entry-IQL Q-values (skip / long / short). Zero-initialised so
    # an empty (un-distilled) head outputs zeros — identical baseline output.
    # State-dict matches v_FIXED exactly when disabled (no params added).
    enable_q_head: bool = False
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
        # Distillation Q-head (V13 prep). Default OFF so v_FIXED bundles
        # load with strict=True. When True, adds q_head linear layer.
        enable_q_head: bool = False,
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

        self.cfg = CtxModelConfig(
            seq_input_dim=seq_input_dim,
            snap_input_dim=snap_input_dim,
            seq_len=seq_len,
            ctx_cont_dim=int(ctx_cont_dim),
            ctx_cat_dim=int(ctx_cat_dim),
            residual_scale=float(residual_scale),
            anchor_eps=float(anchor_eps),
            enable_multi_tf=bool(enable_multi_tf),
            m15_seq_dim=int(m15_seq_dim),
            h1_seq_dim=int(h1_seq_dim),
            h4_seq_dim=int(h4_seq_dim),
            d1_seq_dim=int(d1_seq_dim),
            m15_seq_len=int(m15_seq_len),
            h1_seq_len=int(h1_seq_len),
            h4_seq_len=int(h4_seq_len),
            d1_seq_len=int(d1_seq_len),
            multi_tf_num_layers=int(multi_tf_num_layers),
            multi_tf_scale=float(multi_tf_scale),
            enable_q_head=bool(enable_q_head),
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

        # 3-class direction head: LONG / SHORT / FLAT
        self.head_direction = nn.Linear(d_model, 3)
        # Auxiliary heads that remain active in the canonical runtime lane.
        self.head_path_quality = nn.Linear(d_model, 1)
        self.head_mfe_first_n = nn.Linear(d_model, 1)
        self.head_tradable = nn.Linear(d_model, 1)
        self.head_bad_path = nn.Linear(d_model, 1)
        # Replay-oriented quality heads used by training/audit. Runtime may ignore them.
        self.head_clean_edge = nn.Linear(d_model, 1)
        self.head_survival = nn.Linear(d_model, 1)

        # ── Distillation Q-head (V13 prep) ────────────────────────────
        # Only instantiated when enable_q_head=True. Zero-init so a fresh
        # (un-distilled) head outputs all-zeros — matches "no IQL signal"
        # baseline and keeps training stable until KL-loss pulls it.
        if self.cfg.enable_q_head:
            self.q_head = nn.Linear(d_model, 3)  # [q_skip, q_long, q_short]
            nn.init.zeros_(self.q_head.weight)
            nn.init.zeros_(self.q_head.bias)

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
            # V12.2 v2: ADDITIVE residual fusion. multi_tf_fuse operates ONLY on
            # multi-TF pools (not concatenated with z_v3). Output is a small
            # CORRECTION that's added to z_v3 — preserves v3 baseline behavior
            # when multi-TF is uninformative (random init → near-zero output).
            # Previously: z = multi_tf_fuse([z_v3, mtf]) which DESTROYED v3 path
            # and forced model to relearn baseline from scratch.
            self.multi_tf_fuse = nn.Sequential(
                nn.Linear(4 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            # Zero-init last layer's bias so initial multi-TF correction ≈ 0 →
            # model starts exactly as v3 baseline + tiny noise.
            nn.init.zeros_(self.multi_tf_fuse[-1].bias)
            nn.init.normal_(self.multi_tf_fuse[-1].weight, std=0.01)
            self._expected_m15_seq_dim = int(self.cfg.m15_seq_dim)
            self._expected_h1_seq_dim = int(self.cfg.h1_seq_dim)
            self._expected_h4_seq_dim = int(self.cfg.h4_seq_dim)
            self._expected_d1_seq_dim = int(self.cfg.d1_seq_dim)
            self.register_buffer("multi_tf_scale", torch.tensor(float(self.cfg.multi_tf_scale)))

        # Strict markers (useful for debugging)
        self._expected_seq_dim = int(seq_input_dim)
        self._expected_snap_dim = int(snap_input_dim)
        self._expected_seq_len = int(seq_len)
        self._expected_ctx_cat_dim = int(self.cfg.ctx_cat_dim)
        self._expected_ctx_cont_dim = int(self.cfg.ctx_cont_dim)
        # Anchored residual scale stored in state_dict for replay parity
        self.register_buffer("residual_scale", torch.tensor(float(self.cfg.residual_scale)))
        self.register_buffer("anchor_eps", torch.tensor(float(self.cfg.anchor_eps)))

    def _anchor_logits_from_snap(self, snap_x: torch.Tensor) -> torch.Tensor:
        # Anchor from XGB probs: [p_long, p_short, p_flat] in SIGNAL_BRIDGE_V1 order
        probs = snap_x[:, _ANCHOR_IDX].float()
        eps = float(self.anchor_eps.item())
        probs = torch.clamp(probs, min=eps, max=1.0)
        anchor_logits = torch.log(probs)
        return anchor_logits.detach()

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

        # ── Multi-TF second-stage fusion (V12.2) ──
        # Only active when the model was constructed with enable_multi_tf=True
        # AND the caller provides all four TF tensors. If model is v3-mode but
        # caller mistakenly passes multi-TF inputs, we ignore them (no-op).
        _mtf_inputs_present = all(
            t is not None for t in (seq_m15, seq_h1, seq_h4, seq_d1)
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

            m15_pool = self.m15_encoder(self.m15_proj(seq_m15)).mean(dim=1)   # (B,d)
            h1_pool = self.h1_encoder(self.h1_proj(seq_h1)).mean(dim=1)
            h4_pool = self.h4_encoder(self.h4_proj(seq_h4)).mean(dim=1)
            d1_pool = self.d1_encoder(self.d1_proj(seq_d1)).mean(dim=1)
            # V12.2 v2: ADDITIVE residual. mtf_fuse takes ONLY multi-TF pools
            # (no z_v3 in input), output is a correction added to z_v3 with
            # multi_tf_scale. Random init → ≈0 correction → starts as v3 baseline.
            mtf_combined = torch.cat([m15_pool, h1_pool, h4_pool, d1_pool], dim=1)
            mtf_correction = self.multi_tf_fuse(mtf_combined)
            scale = float(self.multi_tf_scale.item())
            z = z_v3 + scale * mtf_correction
        else:
            z = z_v3   # v3-identical path

        delta_logits = self.head_direction(z)      # (B,3)
        anchor_logits = self._anchor_logits_from_snap(snap_x)
        direction_logits = anchor_logits + (self.residual_scale.to(delta_logits.dtype) * delta_logits)

        # Hard output finite checks
        _assert_finite("direction_logits", direction_logits)
        path_quality = self.head_path_quality(z)
        mfe_first_n = self.head_mfe_first_n(z)
        tradable_logit = self.head_tradable(z)
        bad_path_logit = self.head_bad_path(z)
        clean_edge_logit = self.head_clean_edge(z)
        survival_logit = self.head_survival(z)
        _assert_finite("path_quality", path_quality)
        _assert_finite("mfe_first_n", mfe_first_n)
        _assert_finite("tradable_logit", tradable_logit)
        _assert_finite("bad_path_logit", bad_path_logit)
        _assert_finite("clean_edge_logit", clean_edge_logit)
        _assert_finite("survival_logit", survival_logit)

        out = {
            "direction_logits": direction_logits,
            "anchor_logits": anchor_logits,
            "delta_logits": delta_logits,
            "path_quality": path_quality,
            "mfe_first_n": mfe_first_n,
            "tradable_logit": tradable_logit,
            "bad_path_logit": bad_path_logit,
            "clean_edge_logit": clean_edge_logit,
            "survival_logit": survival_logit,
        }
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
        return out
