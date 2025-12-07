"""
ENTRY_V9 NEXTGEN - Transformer architecture with multi-task learning and regime-conditioning.

Features:
- Multi-task heads: direction_logit, early_move_logit, quality_score
- Regime-conditioning: session_id, vol_regime, trend_regime embeddings
- Residual fusion of seq + snap + regime features
- Configurable architecture (d_model, n_heads, num_layers, dim_ff)
"""

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for sequence data."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor [batch, seq_len, d_model]

        Returns:
            Tensor [batch, seq_len, d_model] with positional encoding added
        """
        bsz, seq_len, d_model = x.shape
        if seq_len > self.max_seq_len:
            # Truncate to max_seq_len instead of raising error
            x = x[:, -self.max_seq_len:, :]
            seq_len = self.max_seq_len
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_emb = self.embedding(positions).unsqueeze(0)
        return x + pos_emb


class SeqTransformerEncoder(nn.Module):
    """Transformer encoder for M5 sequences with causal masking."""

    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.05,
        dim_feedforward: Optional[int] = None,
        pooling: str = "mean",
        use_positional_encoding: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be one of {'mean', 'last'}.")

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pooling = pooling
        self.use_positional_encoding = use_positional_encoding
        self.causal = causal

        self.input_proj = nn.Linear(input_dim, d_model)

        if use_positional_encoding:
            self.pos_encoding = LearnedPositionalEncoding(max_seq_len, d_model)
        else:
            self.pos_encoding = None

        d_ff = dim_feedforward if dim_feedforward is not None else d_model * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @property
    def output_dim(self) -> int:
        return self.d_model

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Build causal mask: True = mask out."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        return mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor [batch, seq_len, n_seq_features]

        Returns:
            Tensor [batch, d_model]
        """
        bsz, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # [B, L, d_model]

        # Add positional encoding
        if self.pos_encoding:
            x = self.pos_encoding(x)

        # Apply causal mask if needed
        mask = None
        if self.causal:
            mask = self._build_causal_mask(seq_len, x.device)

        # Transformer encoder
        x = self.encoder(x, mask=mask)  # [B, L, d_model]

        # Pooling
        if self.pooling == "mean":
            x = x.mean(dim=1)  # [B, d_model]
        elif self.pooling == "last":
            x = x[:, -1, :]  # [B, d_model]

        return x


class SnapshotEncoder(nn.Module):
    """MLP encoder for snapshot features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 128, 64),
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one value.")

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self._output_dim = hidden_dims[-1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor [batch, n_snapshot_features]

        Returns:
            Tensor [batch, hidden_dims[-1]]
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x of shape [B, F], got {x.shape}.")
        return self.mlp(x)


class RegimeEmbeddings(nn.Module):
    """Embeddings for regime features: session, vol_regime, trend_regime."""

    def __init__(
        self,
        session_vocab_size: int = 3,  # EU, OVERLAP, US
        vol_regime_vocab_size: int = 4,  # LOW, MEDIUM, HIGH, EXTREME
        trend_regime_vocab_size: int = 3,  # UP, DOWN, RANGE
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.session_embedding = nn.Embedding(session_vocab_size, embedding_dim)
        self.vol_regime_embedding = nn.Embedding(vol_regime_vocab_size, embedding_dim)
        self.trend_regime_embedding = nn.Embedding(
            trend_regime_vocab_size, embedding_dim
        )
        
        # LEAKAGE DEBUG: Track if we've logged a warning (once per instance)
        self._warned_session = False
        self._warned_vol = False
        self._warned_trend = False

    @property
    def output_dim(self) -> int:
        return self.embedding_dim * 3  # session + vol + trend

    def forward(
        self, session_id: Tensor, vol_regime_id: Tensor, trend_regime_id: Tensor
    ) -> Tensor:
        """
        Args:
            session_id: Tensor [batch] with values 0, 1, 2 (EU, OVERLAP, US)
            vol_regime_id: Tensor [batch] with values 0, 1, 2, 3 (LOW, MEDIUM, HIGH, EXTREME)
            trend_regime_id: Tensor [batch] with values 0, 1, 2 (UP, DOWN, RANGE)

        Returns:
            Tensor [batch, embedding_dim * 3]
        """
        # LEAKAGE DEBUG: Check for out-of-range IDs before clamping (log once)
        if not self._warned_session:
            session_min, session_max = session_id.min().item(), session_id.max().item()
            if session_min < 0 or session_max >= self.session_embedding.num_embeddings:
                print(f"[V9] WARNING: session_id out of range detected before clamping (min={session_min}, max={session_max}, vocab_size={self.session_embedding.num_embeddings})")
                self._warned_session = True
        
        if not self._warned_vol:
            vol_min, vol_max = vol_regime_id.min().item(), vol_regime_id.max().item()
            if vol_min < 0 or vol_max >= self.vol_regime_embedding.num_embeddings:
                print(f"[V9] WARNING: vol_regime_id out of range detected before clamping (min={vol_min}, max={vol_max}, vocab_size={self.vol_regime_embedding.num_embeddings})")
                self._warned_vol = True
        
        if not self._warned_trend:
            trend_min, trend_max = trend_regime_id.min().item(), trend_regime_id.max().item()
            if trend_min < 0 or trend_max >= self.trend_regime_embedding.num_embeddings:
                print(f"[V9] WARNING: trend_regime_id out of range detected before clamping (min={trend_min}, max={trend_max}, vocab_size={self.trend_regime_embedding.num_embeddings})")
                self._warned_trend = True
        
        # Clamp regime IDs to valid ranges (defensive check)
        session_id = torch.clamp(session_id, 0, self.session_embedding.num_embeddings - 1)
        vol_regime_id = torch.clamp(vol_regime_id, 0, self.vol_regime_embedding.num_embeddings - 1)
        trend_regime_id = torch.clamp(trend_regime_id, 0, self.trend_regime_embedding.num_embeddings - 1)
        
        session_emb = self.session_embedding(session_id)  # [B, emb_dim]
        vol_emb = self.vol_regime_embedding(vol_regime_id)  # [B, emb_dim]
        trend_emb = self.trend_regime_embedding(trend_regime_id)  # [B, emb_dim]

        # Concatenate all regime embeddings
        regime_emb = torch.cat([session_emb, vol_emb, trend_emb], dim=-1)  # [B, emb_dim*3]
        return regime_emb


class EntryV9Transformer(nn.Module):
    """
    ENTRY_V9 NEXTGEN Transformer with multi-task heads and regime-conditioning.

    Input:
        - seq_x: [batch, seq_len, n_seq_features]
        - snap_x: [batch, n_snapshot_features]
        - session_id: [batch] (0=EU, 1=OVERLAP, 2=US)
        - vol_regime_id: [batch] (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
        - trend_regime_id: [batch] (0=UP, 1=DOWN, 2=RANGE)

    Output (dict):
        - direction_logit: [batch] (binary classification)
        - early_move_logit: [batch] (binary: MFE before MAE within horizon)
        - quality_score: [batch] (regression: normalized MFE-MAE gap)
    """

    def __init__(
        self,
        seq_input_dim: int,
        snap_input_dim: int,
        max_seq_len: int,
        seq_cfg: Optional[Mapping[str, Any]] = None,
        snap_cfg: Optional[Mapping[str, Any]] = None,
        regime_cfg: Optional[Mapping[str, Any]] = None,
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.1,
        head_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        seq_cfg = dict(seq_cfg or {})
        snap_cfg = dict(snap_cfg or {})
        regime_cfg = dict(regime_cfg or {})

        # Sequence encoder
        self.seq_encoder = SeqTransformerEncoder(
            input_dim=seq_input_dim,
            max_seq_len=max_seq_len,
            d_model=seq_cfg.get("d_model", 128),
            n_heads=seq_cfg.get("n_heads", 4),
            num_layers=seq_cfg.get("num_layers", 3),
            dropout=seq_cfg.get("dropout", 0.05),
            dim_feedforward=seq_cfg.get("dim_feedforward"),
            pooling=seq_cfg.get("pooling", "mean"),
            use_positional_encoding=seq_cfg.get("use_positional_encoding", True),
            causal=seq_cfg.get("causal", True),
        )

        # Snapshot encoder
        self.snap_encoder = SnapshotEncoder(
            input_dim=snap_input_dim,
            hidden_dims=tuple(snap_cfg.get("hidden_dims", (256, 128, 64))),
            use_layernorm=snap_cfg.get("use_layernorm", True),
            dropout=snap_cfg.get("dropout", 0.0),
        )

        # Regime embeddings
        self.regime_embeddings = RegimeEmbeddings(
            session_vocab_size=regime_cfg.get("session_vocab_size", 3),
            vol_regime_vocab_size=regime_cfg.get("vol_regime_vocab_size", 4),
            trend_regime_vocab_size=regime_cfg.get("trend_regime_vocab_size", 3),
            embedding_dim=regime_cfg.get("embedding_dim", 16),
        )

        # Fusion: seq + snap + regime
        fused_in_dim = (
            self.seq_encoder.output_dim
            + self.snap_encoder.output_dim
            + self.regime_embeddings.output_dim
        )

        # Residual fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fused_in_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        self.fused_dim = fusion_hidden_dim

        # Multi-task heads
        head_hidden = head_hidden_dim or self.fused_dim

        def make_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(self.fused_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(fusion_dropout * 0.5),
                nn.Linear(head_hidden, 1),
            )

        # Direction head (binary classification)
        self.head_direction = make_head()

        # Early move head (binary: MFE before MAE within horizon)
        self.head_early_move = make_head()

        # Quality score head (regression: normalized MFE-MAE gap)
        self.head_quality = make_head()

    def forward(
        self,
        seq_x: Tensor,
        snap_x: Tensor,
        session_id: Tensor,
        vol_regime_id: Tensor,
        trend_regime_id: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            seq_x: [batch, seq_len, n_seq_features]
            snap_x: [batch, n_snapshot_features]
            session_id: [batch] (0=EU, 1=OVERLAP, 2=US)
            vol_regime_id: [batch] (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
            trend_regime_id: [batch] (0=UP, 1=DOWN, 2=RANGE)

        Returns:
            Dict with:
                - direction_logit: [batch]
                - early_move_logit: [batch]
                - quality_score: [batch]
        """
        # Encode sequences
        seq_emb = self.seq_encoder(seq_x)  # [B, d_seq]

        # Encode snapshots
        snap_emb = self.snap_encoder(snap_x)  # [B, d_snap]

        # Encode regimes
        regime_emb = self.regime_embeddings(
            session_id, vol_regime_id, trend_regime_id
        )  # [B, d_regime]

        # Fuse: seq + snap + regime
        fused = torch.cat([seq_emb, snap_emb, regime_emb], dim=-1)  # [B, d_seq + d_snap + d_regime]
        fused = self.fusion(fused)  # [B, d_fused]

        # Multi-task heads
        direction_logit = self.head_direction(fused).squeeze(-1)  # [B]
        early_move_logit = self.head_early_move(fused).squeeze(-1)  # [B]
        quality_score = self.head_quality(fused).squeeze(-1)  # [B]

        return {
            "direction_logit": direction_logit,
            "early_move_logit": early_move_logit,
            "quality_score": quality_score,
        }


def build_entry_v9_model(config: Dict[str, Any]) -> EntryV9Transformer:
    """
    Build ENTRY_V9 model from config.

    Config structure:
        model:
            name: "entry_v9"
            seq_input_dim: int
            snap_input_dim: int
            max_seq_len: int
            seq_cfg:
                d_model: int (default: 128)
                n_heads: int (default: 4)
                num_layers: int (default: 3)
                dim_feedforward: int (optional)
                dropout: float (default: 0.05)
            snap_cfg:
                hidden_dims: list[int] (default: [256, 128, 64])
                use_layernorm: bool (default: True)
                dropout: float (default: 0.0)
            regime_cfg:
                embedding_dim: int (default: 16)
            fusion_hidden_dim: int (default: 128)
            fusion_dropout: float (default: 0.1)
            head_hidden_dim: int (optional)
    """
    model_cfg = config.get("model", {})
    if model_cfg.get("name") != "entry_v9":
        raise ValueError(f"Expected model.name='entry_v9', got {model_cfg.get('name')}")

    return EntryV9Transformer(
        seq_input_dim=model_cfg["seq_input_dim"],
        snap_input_dim=model_cfg["snap_input_dim"],
        max_seq_len=model_cfg["max_seq_len"],
        seq_cfg=model_cfg.get("seq_cfg", {}),
        snap_cfg=model_cfg.get("snap_cfg", {}),
        regime_cfg=model_cfg.get("regime_cfg", {}),
        fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 128),
        fusion_dropout=model_cfg.get("fusion_dropout", 0.1),
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
    )

