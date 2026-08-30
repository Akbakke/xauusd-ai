# gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    ENTRY_EXIT_RESOLUTION_RATIO,
    EXIT_FEATURE_MAX_SEQUENCE_BARS,
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_decision_token_v1 import (
    ENTRY_DECISION_TOKEN_COMPONENTS,
    ENTRY_DECISION_TOKEN_DIM,
    ENTRY_DECISION_TOKEN_SOURCE_DIM,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_DIP_DIRECTIONS as DIP_DIRECTIONS,  # noqa: F401
    MODEL_NATIVE_DIP_OUTPUT_DIM as DIP_HEAD_DIM,
    MODEL_NATIVE_DIP_OUTPUT_TARGETS as DIP_TARGETS,  # noqa: F401
    MODEL_NATIVE_AUX_RISK_HORIZONS as TIMING_HORIZONS,  # noqa: F401
    MODEL_NATIVE_TIMING_DIRECTIONS as TIMING_DIRECTIONS,  # noqa: F401
    MODEL_NATIVE_TIMING_OUTPUT_DIM as TIMING_HEAD_DIM,
    MODEL_NATIVE_TIMING_TARGETS as TIMING_TARGETS,  # noqa: F401
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    MIN_EFFECTIVE_SCALE as TF_INPUT_SCALE_MIN_EFFECTIVE,
    NEUTRAL_EFFECTIVE_INIT as TF_INPUT_SCALE_NEUTRAL_INIT,
    TF_NAMES as TF_INPUT_SCALE_NAMES,
    raw_tf_input_scale_from_effective,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    EXPECTED_SURFACES as INPUT_NORMALIZATION_SURFACES,
    require_input_normalization_contract,
)
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
)
from gx1.contracts.unified_exit_episode_pack_v1 import (
    UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS,
    UNIFIED_EXIT_EPISODE_STATE_COUNT,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_ENCODER_LAYERS,
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)


def _assert_shape(name: str, t: torch.Tensor, nd: int) -> None:
    if not isinstance(t, torch.Tensor):
        raise RuntimeError(f"TYPE_MISMATCH: {name} is not a torch.Tensor (got {type(t)})")
    if t.dim() != nd:
        raise RuntimeError(f"SHAPE_MISMATCH: {name}.dim={t.dim()} expected={nd} shape={tuple(t.shape)}")


def _assert_finite(name: str, t: torch.Tensor) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NONFINITE: {name} contains NaN/Inf")


_EXIT_TOKEN_AXIS_KERNEL_ROW_CHUNK = 32_768


def _apply_exit_token_axis_encoder(
    encoder: nn.Module,
    rows: torch.Tensor,
    *,
    row_chunk_size: int = _EXIT_TOKEN_AXIS_KERNEL_ROW_CHUNK,
) -> torch.Tensor:
    """Apply a token-axis encoder without exceeding CUDA batch-grid limits.

    Rows are independent batch items. Concatenating differentiable encoder
    calls is algebraically identical to one call in eval mode and preserves
    the complete graph in training; no state, feature or target is sampled or
    detached.
    """

    _assert_shape("exit_token_axis_rows", rows, 3)
    if (
        isinstance(row_chunk_size, bool)
        or not isinstance(row_chunk_size, int)
        or row_chunk_size < 1
    ):
        raise RuntimeError("UNIFIED_EXIT_TOKEN_AXIS_CHUNK_INVALID")
    if int(rows.shape[0]) <= row_chunk_size:
        return encoder(rows)
    return torch.cat(
        [
            encoder(rows[left : left + row_chunk_size])
            for left in range(
                0,
                int(rows.shape[0]),
                row_chunk_size,
            )
        ],
        dim=0,
    )


@dataclass(frozen=True)
class UnifiedExitIncrementalCarry:
    """Learned recurrent state carried between authoritative M1 closures."""

    step_count: int
    batch_size: int
    local_global_hidden: torch.Tensor
    local_family_hidden: Mapping[str, torch.Tensor]
    mtf_family_hidden: Mapping[str, Mapping[str, torch.Tensor]]
    mtf_current_raw: Mapping[str, torch.Tensor]
    path_hidden: torch.Tensor


# Candidate batch-8 profiling showed only 5.7 GiB peak allocated inside the
# model under a hard 12 GiB process fence. CUDA checkpoint recomputation was
# therefore throughput cost at this geometry: it re-ran every encoder layer
# during backward while retaining less than the safely available VRAM.
# CPU training keeps the bounded policy; CUDA has an independent allocator
# fence in the trainer before this path is entered.
TRAIN_ACTIVATION_CHECKPOINT_POLICY = "cuda_disabled_cpu_checkpointed_v2"
CUDA_TRAIN_ACTIVATION_CHECKPOINT_ENABLED = False
MODEL_ARCHITECTURE_SCHEMA_VERSION = "entry_v10_ctx_hybrid_transformer_v8"
MODEL_OUTPUT_SCHEMA_VERSION = "entry_v10_ctx_model_outputs_v8"
_UNIT_TEST_ARCHITECTURE_SENTINEL = object()


def _memory_bounded_transformer_encoder(
    encoder: nn.TransformerEncoder,
    src: torch.Tensor,
    *,
    src_key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the exact encoder under the source-bound activation policy.

    Entry M5, all five MTF branches, and Exit M1 keep the same layers, rows,
    features, batch semantics, dropout stream, and gradients. CUDA candidate
    batch-8 training retains activations within the trainer's 12 GiB allocator
    fence, avoiding a second encoder forward during backward. CPU training
    retains the memory-bounded recomputation path. Evaluation and live
    inference always use the ordinary encoder path.
    """

    if not (encoder.training and torch.is_grad_enabled()):
        return encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )
    if src.is_cuda and not CUDA_TRAIN_ACTIVATION_CHECKPOINT_ENABLED:
        return encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )

    hidden = src
    for layer in encoder.layers:
        def _layer_forward(
            value: torch.Tensor,
            *,
            exact_layer: nn.TransformerEncoderLayer = layer,
        ) -> torch.Tensor:
            return exact_layer(
                value,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=False,
            )

        hidden = _torch_checkpoint(
            _layer_forward,
            hidden,
            use_reentrant=False,
            preserve_rng_state=True,
        )
    if encoder.norm is not None:
        hidden = encoder.norm(hidden)
    return hidden


EXACT_TRENDLINE_EVENT_OUTPUT_DIM = 4
EXACT_SPECIALIST_NAMES = MODEL_NATIVE_TRAINING_SPECIALISTS
EXACT_CTX_CAT_DOMAINS = MODEL_NATIVE_CTX_CAT_DOMAINS
if tuple(EXACT_CTX_CAT_DOMAINS) != MODEL_NATIVE_CTX_CAT_FIELDS:
    raise RuntimeError("ENTRY_MODEL_NATIVE_CTX_CAT_DOMAIN_ORDER_INVALID")

# ── Dip-analysis head layout (V10 entry) — risk-aware, multi-horizon, distributional.
# Output index = flatten over (direction, horizon, target) in this order. The
# trainer's pinball loss and any consumer MUST use this same layout (documented,
# not magic numbers). dip_p50/p90 = conditional quantiles of mae_before_mfe (dip
# depth if taking now); recovery_p50 = median mfe-after-dip. See memory
# project_gx1_dip_aware_entry_timing.
DIP_HORIZONS = TIMING_HORIZONS

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
    """Internal immutable configuration for the exact Entry architecture.

    Every recipe-owned MTF, dropout and specialist-capacity value is required.
    The class provides storage only; it is never an alternate configuration
    loader or a source of compatibility defaults.
    """

    seq_input_dim: int
    snap_input_dim: int
    seq_len: int
    dropout: float
    multi_tf_num_layers: int
    m15_seq_dim: int
    h1_seq_dim: int
    h4_seq_dim: int
    d1_seq_dim: int
    m15_seq_len: int
    h1_seq_len: int
    h4_seq_len: int
    d1_seq_len: int
    multi_tf_scale: float
    m5_seq_dim: int
    m5_seq_len: int
    specialist_num_layers: int
    specialist_fusion_scale: float
    cross_family_fusion_scale: float
    ctx_cat_dim: int
    ctx_cont_dim: int
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: Optional[int] = None
    ctx_cat_emb_dim: int = 8
    ctx_cat_scale: float = 0.25
    ctx_cont_scale: float = 0.25


class EntryV10CtxHybridTransformer(nn.Module):
    """
    Minimal, strict CTX model used by:
      - gx1/models/entry_v10/entry_v10_bundle.py
      - gx1/rl/entry_v10/train_entry_transformer_v10.py (CTX variant)

    Forward signature (expected by docs/usage):
        out = model(
            seq_x,
            snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        seq_m15=seq_m15,
            seq_h1=seq_h1,
            seq_h4=seq_h4,
            seq_d1=seq_d1,
        )
        out["entry_action_q_bps"] -> (B, 3)  # LONG, SHORT, FLAT raw bps
        out["side_mae_bps"]      -> (B, 2)  # raw LONG/SHORT adverse excursion
        out["trendline_event_logits"] -> (B, EXACT_TRENDLINE_EVENT_OUTPUT_DIM)
    """

    def __init__(
        self,
        *,
        seq_input_dim: int,
        snap_input_dim: int,
        seq_len: int,
        dropout: float,
        ctx_cont_dim: int = MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim: int = MODEL_NATIVE_CTX_CAT_DIM,
        m15_seq_dim: int,
        h1_seq_dim: int,
        h4_seq_dim: int,
        d1_seq_dim: int,
        m15_seq_len: int,
        h1_seq_len: int,
        h4_seq_len: int,
        d1_seq_len: int,
        multi_tf_num_layers: int,
        multi_tf_scale: float,
        m5_seq_dim: int,
        m5_seq_len: int,
        specialist_input_indices: Dict[str, list[int]],
        specialist_ctx_cont_indices: Dict[str, list[int]],
        specialist_ctx_cont_nominal_indices: Dict[str, list[int]],
        specialist_ctx_cat_indices: Dict[str, list[int]],
        multi_tf_specialist_input_indices: Dict[str, list[int]],
        temporal_alias_signal_indices: list[int],
        temporal_alias_ctx_cont_indices: list[int],
        input_normalization: Mapping[str, object],
        specialist_num_layers: int,
        specialist_fusion_scale: float,
        cross_family_fusion_scale: float,
    ) -> None:
        unit_test_shape = (
            getattr(self, "_gx1_unit_test_architecture_token", None)
            is _UNIT_TEST_ARCHITECTURE_SENTINEL
        )
        if unit_test_shape:
            object.__delattr__(self, "_gx1_unit_test_architecture_token")
        else:
            architecture = current_entry_exit_architecture_observation()
            architecture["shared_surface"] = {
                "signal_dim": seq_input_dim,
                "snap_dim": snap_input_dim,
                "ctx_cont_dim": ctx_cont_dim,
                "ctx_cat_dim": ctx_cat_dim,
            }
            architecture["schemas"]["input_normalization"] = (
                input_normalization.get("schema_version")
                if isinstance(input_normalization, Mapping)
                else None
            )
            architecture["local_specialists"] = (
                list(specialist_input_indices)
                if isinstance(specialist_input_indices, Mapping)
                else specialist_input_indices
            )
            architecture["mtf_specialists"] = (
                list(multi_tf_specialist_input_indices)
                if isinstance(multi_tf_specialist_input_indices, Mapping)
                else multi_tf_specialist_input_indices
            )
            architecture["mtf"]["per_tf_widths"] = {
                "M5": m5_seq_dim,
                "M15": m15_seq_dim,
                "H1": h1_seq_dim,
                "H4": h4_seq_dim,
                "D1": d1_seq_dim,
            }
            architecture["mtf"]["per_tf_window_bars"] = {
                "M5": m5_seq_len,
                "M15": m15_seq_len,
                "H1": h1_seq_len,
                "H4": h4_seq_len,
                "D1": d1_seq_len,
            }
            architecture["entry"]["sequence_bars"] = seq_len
            architecture["exit"]["sequence_bars"] = (
                seq_len * ENTRY_EXIT_RESOLUTION_RATIO
            )
            architecture["exit"]["max_path_bars"] = (
                UNIFIED_EXIT_MAX_PATH_BARS
            )
            require_entry_exit_production_architecture(
                architecture,
                context="ENTRY_V10_MODEL_CONSTRUCTION",
            )
        super().__init__()
        if seq_input_dim <= 0 or snap_input_dim <= 0 or seq_len <= 0:
            raise RuntimeError(
                f"INVALID_INIT: seq_input_dim={seq_input_dim} snap_input_dim={snap_input_dim} seq_len={seq_len}"
            )
        if (
            isinstance(dropout, bool)
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise RuntimeError(
                "MODEL_DROPOUT_INVALID: dropout must be explicit, finite and "
                f"in [0,1); got {dropout!r}"
            )
        if int(seq_input_dim) != int(snap_input_dim):
            raise RuntimeError(
                "SEQ_SNAP_DIM_MISMATCH: bit-identical current-bar contract "
                f"requires equal widths; seq={seq_input_dim} snap={snap_input_dim}"
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
        if (
            isinstance(multi_tf_num_layers, bool)
            or not isinstance(multi_tf_num_layers, int)
            or multi_tf_num_layers <= 0
        ):
            raise RuntimeError(
                "MULTI_TF_NUM_LAYERS_INVALID: exact architecture requires "
                f"a positive explicit integer; got {multi_tf_num_layers!r}"
            )
        mandatory_positive_scales = {
            "multi_tf_scale": multi_tf_scale,
            "specialist_fusion_scale": specialist_fusion_scale,
            "cross_family_fusion_scale": cross_family_fusion_scale,
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
            dropout=float(dropout),
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
            cross_family_fusion_scale=float(cross_family_fusion_scale),
        )

        d_model = int(self.cfg.d_model)
        if d_model != ENTRY_DECISION_TOKEN_DIM:
            raise RuntimeError(
                "ENTRY_DECISION_TOKEN_MODEL_WIDTH_MISMATCH: "
                f"d_model={d_model} token_dim={ENTRY_DECISION_TOKEN_DIM}"
            )
        n_heads = int(self.cfg.n_heads)
        num_layers = int(self.cfg.num_layers)
        dropout = float(self.cfg.dropout)
        d_ff = int(self.cfg.dim_feedforward) if self.cfg.dim_feedforward else int(d_model * 4)

        if (
            not isinstance(temporal_alias_signal_indices, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in temporal_alias_signal_indices
            )
            or temporal_alias_signal_indices
            != sorted(set(temporal_alias_signal_indices))
            or any(
                value < 0 or value >= int(snap_input_dim)
                for value in temporal_alias_signal_indices
            )
        ):
            raise RuntimeError("TEMPORAL_ALIAS_SIGNAL_INDICES_INVALID")
        if (
            not isinstance(temporal_alias_ctx_cont_indices, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in temporal_alias_ctx_cont_indices
            )
            or len(temporal_alias_ctx_cont_indices)
            != len(temporal_alias_signal_indices)
            or len(set(temporal_alias_ctx_cont_indices))
            != len(temporal_alias_ctx_cont_indices)
            or any(
                value < 0 or value >= int(ctx_cont_dim)
                for value in temporal_alias_ctx_cont_indices
            )
        ):
            raise RuntimeError("TEMPORAL_ALIAS_CTX_CONT_INDICES_INVALID")

        if not isinstance(input_normalization, Mapping):
            raise RuntimeError("ENTRY_INPUT_NORMALIZATION_CONTRACT_MISSING")
        normalization_surfaces = input_normalization.get("surfaces")
        if not isinstance(normalization_surfaces, Mapping):
            raise RuntimeError("ENTRY_INPUT_NORMALIZATION_SURFACES_MISSING")
        normalization_field_names = {
            surface: list(
                normalization_surfaces.get(surface, {}).get(
                    "field_names",
                    [],
                )
            )
            if isinstance(normalization_surfaces.get(surface), Mapping)
            else []
            for surface in INPUT_NORMALIZATION_SURFACES
        }
        normalized_input_contract = require_input_normalization_contract(
            input_normalization,
            expected_field_names=normalization_field_names,
            expected_ctx_cat_names=MODEL_NATIVE_CTX_CAT_FIELDS,
        )
        expected_surface_widths = {
            "signal": int(seq_input_dim),
            "ctx_cont": int(ctx_cont_dim),
            "mtf_m5": int(self.cfg.m5_seq_dim),
            "mtf_m15": int(self.cfg.m15_seq_dim),
            "mtf_h1": int(self.cfg.h1_seq_dim),
            "mtf_h4": int(self.cfg.h4_seq_dim),
            "mtf_d1": int(self.cfg.d1_seq_dim),
        }
        if any(
            len(normalization_field_names[surface]) != width
            for surface, width in expected_surface_widths.items()
        ):
            raise RuntimeError("ENTRY_INPUT_NORMALIZATION_WIDTH_MISMATCH")
        normalized_alias_pairs = [
            (
                int(alias["signal_index"]),
                int(alias["ctx_cont_index"]),
            )
            for alias in normalized_input_contract["temporal_aliases"]
        ]
        if normalized_alias_pairs != list(
            zip(
                temporal_alias_signal_indices,
                temporal_alias_ctx_cont_indices,
            )
        ):
            raise RuntimeError("ENTRY_INPUT_NORMALIZATION_ALIAS_MISMATCH")
        self._input_normalization_contract = normalized_input_contract
        self._input_normalization_field_names = normalization_field_names
        self._input_norm_categorical_domains = {
            surface: {
                str(field): tuple(int(value) for value in domain)
                for field, domain in normalized_input_contract["surfaces"][
                    surface
                ]["categorical_domains"].items()
            }
            for surface in INPUT_NORMALIZATION_SURFACES
        }
        self.register_buffer(
            "input_norm_contract_sha256",
            torch.tensor(
                list(
                    bytes.fromhex(
                        str(normalized_input_contract["contract_sha256"])
                    )
                ),
                dtype=torch.uint8,
            ),
        )
        for surface in INPUT_NORMALIZATION_SURFACES:
            surface_contract = normalized_input_contract["surfaces"][surface]
            prefix = f"input_norm_{surface}"
            self.register_buffer(
                f"{prefix}_center",
                torch.tensor(surface_contract["center"], dtype=torch.float32),
            )
            self.register_buffer(
                f"{prefix}_scale",
                torch.tensor(surface_contract["scale"], dtype=torch.float32),
            )
            self.register_buffer(
                f"{prefix}_binary_mask",
                torch.tensor(
                    surface_contract["binary_mask"],
                    dtype=torch.bool,
                ),
            )
            self.register_buffer(
                f"{prefix}_categorical_mask",
                torch.tensor(
                    surface_contract["categorical_mask"],
                    dtype=torch.bool,
                ),
            )
        # These Python tuples are immutable mirrors of the validated buffer
        # masks.  They are used only to choose which already-device-resident
        # tensor columns receive nominal validation/embeddings.  Keeping them
        # on the host avoids materialising a CUDA buffer through ``tolist`` in
        # every forward call; the registered masks remain the mathematical
        # authority for normalization.
        self._input_norm_categorical_index_tuples = {
            surface: tuple(
                torch.nonzero(
                    getattr(self, f"input_norm_{surface}_categorical_mask"),
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )
            for surface in INPUT_NORMALIZATION_SURFACES
        }
        self.register_buffer(
            "input_norm_alias_signal_indices",
            torch.tensor(temporal_alias_signal_indices, dtype=torch.long),
        )
        self.register_buffer(
            "input_norm_alias_ctx_cont_indices",
            torch.tensor(temporal_alias_ctx_cont_indices, dtype=torch.long),
        )
        ctx_cat_domains = normalized_input_contract["ctx_cat"]["domains"]
        self.register_buffer(
            "input_norm_ctx_cat_min",
            torch.tensor(
                [
                    min(ctx_cat_domains[name])
                    for name in MODEL_NATIVE_CTX_CAT_FIELDS
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "input_norm_ctx_cat_max",
            torch.tensor(
                [
                    max(ctx_cat_domains[name])
                    for name in MODEL_NATIVE_CTX_CAT_FIELDS
                ],
                dtype=torch.long,
            ),
        )
        generic_snap_indices = [
            index
            for index in range(int(snap_input_dim))
            if index not in set(temporal_alias_signal_indices)
        ]
        if not generic_snap_indices:
            raise RuntimeError("GENERIC_SNAP_INPUT_EMPTY_AFTER_ALIAS_EXCLUSION")

        # Project signal-only inputs into transformer dimension. Exact
        # ctx_cont aliases remain temporal seq evidence, but their byte-equal
        # current-bar snap copies cannot enter this generic snapshot path.
        # Keep this immutable Python membership set beside the tensor buffer.
        # The latter is the device index owner; the former prevents every
        # Forward/Exit call from synchronizing a CUDA index tensor merely to
        # decide which nominal embeddings belong in this projection.
        self._generic_snap_index_set = frozenset(generic_snap_indices)
        self.seq_proj = nn.Linear(int(seq_input_dim), d_model)
        self.snap_proj = nn.Linear(len(generic_snap_indices), d_model)
        self.register_buffer(
            "generic_snap_idx",
            torch.tensor(generic_snap_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "temporal_alias_signal_idx",
            torch.tensor(temporal_alias_signal_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "temporal_alias_ctx_cont_idx",
            torch.tensor(temporal_alias_ctx_cont_indices, dtype=torch.long),
            persistent=False,
        )
        signal_categorical_indices = self._input_norm_categorical_index_tuples[
            "signal"
        ]
        self.signal_nominal_embeddings = nn.ModuleDict(
            {
                str(index): nn.Embedding(
                    len(
                        self._input_norm_categorical_domains["signal"][
                            normalization_field_names["signal"][index]
                        ]
                    ),
                    d_model,
                )
                for index in signal_categorical_indices
            }
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Context encoders. Raw context never receives an independent global
        # projection: every field first enters its one exact family token.
        self.ctx_cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(domain), int(self.cfg.ctx_cat_emb_dim))
                for domain in EXACT_CTX_CAT_DOMAINS.values()
            ]
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_model + d_model, d_model),
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
        expected_indices = set(range(int(seq_input_dim)))
        if seen_indices != expected_indices:
            missing = sorted(expected_indices - seen_indices)
            unexpected = sorted(seen_indices - expected_indices)
            raise RuntimeError(
                "SPECIALIST_INDEX_COVERAGE_INVALID: "
                f"missing={missing[:20]} total_missing={len(missing)} "
                f"unexpected={unexpected[:20]} total_unexpected={len(unexpected)}"
            )
        self._specialist_input_index_sets = {
            name: frozenset(indices) for name, indices in cleaned.items()
        }

        def _clean_context_partition(
            raw: Dict[str, list[int]],
            *,
            width: int,
            label: str,
        ) -> Dict[str, list[int]]:
            if tuple(raw) != EXACT_SPECIALIST_NAMES:
                raise RuntimeError(
                    f"{label}_SPECIALIST_NAMES_MISMATCH: "
                    f"got={tuple(raw)} expected={EXACT_SPECIALIST_NAMES}"
                )
            cleaned_partition: Dict[str, list[int]] = {}
            seen: set[int] = set()
            for specialist in EXACT_SPECIALIST_NAMES:
                values = raw[specialist]
                if not isinstance(values, list):
                    raise RuntimeError(f"{label}_INDICES_INVALID: {specialist}")
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in values
                ):
                    raise RuntimeError(
                        f"{label}_INDEX_TYPE_INVALID: {specialist}"
                    )
                if values != sorted(set(values)):
                    raise RuntimeError(
                        f"{label}_INDEX_ORDER_OR_DUPLICATE: {specialist}"
                    )
                if any(value < 0 or value >= int(width) for value in values):
                    raise RuntimeError(f"{label}_INDEX_OOB: {specialist}")
                overlap = seen.intersection(values)
                if overlap:
                    raise RuntimeError(
                        f"{label}_INDEX_OVERLAP: {specialist} "
                        f"overlap={sorted(overlap)}"
                    )
                seen.update(values)
                cleaned_partition[specialist] = list(values)
            expected = set(range(int(width)))
            if seen != expected:
                raise RuntimeError(
                    f"{label}_INDEX_COVERAGE_INVALID: "
                    f"missing={sorted(expected - seen)} "
                    f"unexpected={sorted(seen - expected)}"
                )
            return cleaned_partition

        cleaned_ctx_cont = _clean_context_partition(
            specialist_ctx_cont_indices,
            width=int(self.cfg.ctx_cont_dim),
            label="SPECIALIST_CTX_CONT",
        )
        cleaned_ctx_cat = _clean_context_partition(
            specialist_ctx_cat_indices,
            width=int(self.cfg.ctx_cat_dim),
            label="SPECIALIST_CTX_CAT",
        )
        if (
            not isinstance(specialist_ctx_cont_nominal_indices, Mapping)
            or set(specialist_ctx_cont_nominal_indices)
            != set(EXACT_SPECIALIST_NAMES)
        ):
            raise RuntimeError(
                "SPECIALIST_CTX_CONT_NOMINAL_SPECIALIST_SET_INVALID"
            )
        cleaned_ctx_cont_nominal: Dict[str, list[int]] = {}
        seen_nominal: set[int] = set()
        for specialist in EXACT_SPECIALIST_NAMES:
            values = specialist_ctx_cont_nominal_indices[specialist]
            if (
                not isinstance(values, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in values
                )
                or values != sorted(set(values))
            ):
                raise RuntimeError(
                    f"SPECIALIST_CTX_CONT_NOMINAL_INDEX_INVALID: {specialist}"
                )
            if any(value not in cleaned_ctx_cont[specialist] for value in values):
                raise RuntimeError(
                    f"SPECIALIST_CTX_CONT_NOMINAL_OWNER_INVALID: {specialist}"
                )
            overlap = seen_nominal.intersection(values)
            if overlap:
                raise RuntimeError(
                    "SPECIALIST_CTX_CONT_NOMINAL_INDEX_OVERLAP: "
                    f"{specialist} overlap={sorted(overlap)}"
                )
            seen_nominal.update(values)
            cleaned_ctx_cont_nominal[specialist] = list(values)
        cleaned_ctx_cont_numeric = {
            specialist: [
                index
                for index in cleaned_ctx_cont[specialist]
                if index not in set(cleaned_ctx_cont_nominal[specialist])
            ]
            for specialist in EXACT_SPECIALIST_NAMES
        }
        self._specialist_ctx_cont_nominal_index_tuples = {
            name: tuple(indices)
            for name, indices in cleaned_ctx_cont_nominal.items()
        }
        self._specialist_names = EXACT_SPECIALIST_NAMES
        if (
            not isinstance(multi_tf_specialist_input_indices, dict)
            or tuple(multi_tf_specialist_input_indices)
            != EXACT_SPECIALIST_NAMES
        ):
            raise RuntimeError("MULTI_TF_SPECIALIST_INDEX_CONTRACT_INVALID")
        if len(
            {
                int(self.cfg.m5_seq_dim),
                int(self.cfg.m15_seq_dim),
                int(self.cfg.h1_seq_dim),
                int(self.cfg.h4_seq_dim),
                int(self.cfg.d1_seq_dim),
            }
        ) != 1:
            raise RuntimeError(
                "MULTI_TF_SPECIALIST_WIDTH_SPLIT_BRAIN: all timeframe surfaces "
                "must declare the same ordered feature contract"
            )
        cleaned_mtf: dict[str, list[int]] = {}
        seen_mtf: set[int] = set()
        for specialist in EXACT_SPECIALIST_NAMES:
            indices = multi_tf_specialist_input_indices[specialist]
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in indices
                )
                or indices != sorted(set(indices))
                or any(
                    value < 0 or value >= int(self.cfg.m5_seq_dim)
                    for value in indices
                )
            ):
                raise RuntimeError(
                    f"MULTI_TF_SPECIALIST_INDEX_INVALID: {specialist}"
                )
            overlap = seen_mtf.intersection(indices)
            if overlap:
                raise RuntimeError(
                    "MULTI_TF_SPECIALIST_INDEX_OVERLAP: "
                    f"{specialist} overlap={sorted(overlap)}"
                )
            seen_mtf.update(indices)
            cleaned_mtf[specialist] = list(indices)
        if seen_mtf != set(range(int(self.cfg.m5_seq_dim))):
            raise RuntimeError(
                "MULTI_TF_SPECIALIST_INDEX_COVERAGE_INVALID: every field must "
                "have exactly one family owner"
            )
        self._multi_tf_specialist_index_tuples = {
            name: tuple(indices) for name, indices in cleaned_mtf.items()
        }
        self.multi_tf_specialist_routing_schema_version = (
            MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION
        )
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
            return nn.TransformerEncoder(
                layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )

        self.specialist_encoder = nn.ModuleDict(
            {name: _mk_encoder(specialist_layers) for name in self._specialist_names}
        )
        self.specialist_ctx_cont_proj = nn.ModuleDict(
            {
                name: nn.Linear(len(indices), d_model)
                for name, indices in cleaned_ctx_cont_numeric.items()
                if indices
            }
        )
        self.specialist_ctx_cont_nominal_embeddings = nn.ModuleDict(
            {
                str(index): nn.Embedding(5, d_model)
                for index in sorted(seen_nominal)
            }
        )
        self.specialist_ctx_cont_nominal_out = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(len(indices) * d_model),
                    nn.Linear(len(indices) * d_model, d_model),
                )
                for name, indices in cleaned_ctx_cont_nominal.items()
                if indices
            }
        )
        self.specialist_ctx_cat_proj = nn.ModuleDict(
            {
                name: nn.Linear(
                    len(indices) * int(self.cfg.ctx_cat_emb_dim),
                    d_model,
                )
                for name, indices in cleaned_ctx_cat.items()
                if indices
            }
        )
        self.specialist_context_out = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.GELU(),
                    nn.Linear(d_model, d_model, bias=False),
                )
                for name in self._specialist_names
                if (
                    cleaned_ctx_cont_numeric[name]
                    or cleaned_ctx_cont_nominal[name]
                    or cleaned_ctx_cat[name]
                )
            }
        )
        self.family_context_global = nn.Sequential(
            nn.LayerNorm(len(self._specialist_names) * d_model),
            nn.Linear(len(self._specialist_names) * d_model, d_model),
            nn.GELU(),
        )
        # Specialists first reason over their own temporal fields, then attend
        # to one another.  This makes structure×trend, S/R×momentum,
        # SMC×candles, etc. learned interactions instead of isolated votes.
        self.specialist_token_identity = nn.Parameter(
            torch.empty(1, len(self._specialist_names), d_model)
        )
        nn.init.normal_(self.specialist_token_identity, std=0.02)
        self.specialist_cross_attn = _mk_encoder(1)
        for name, idx in cleaned.items():
            self.register_buffer(
                f"specialist_idx_{name}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )
        for name, idx in cleaned_mtf.items():
            self.register_buffer(
                f"multi_tf_specialist_idx_{name}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )
        for name, idx in cleaned_ctx_cont_numeric.items():
            self.register_buffer(
                f"specialist_ctx_cont_idx_{name}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )
        for name, idx in cleaned_ctx_cont_nominal.items():
            self.register_buffer(
                f"specialist_ctx_cont_nominal_idx_{name}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )
        for name, idx in cleaned_ctx_cat.items():
            self.register_buffer(
                f"specialist_ctx_cat_idx_{name}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )
        self.specialist_gate = nn.Linear(d_model, len(self._specialist_names))
        self.specialist_token_gate = nn.Linear(d_model, 1)
        self.specialist_out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.specialist_out.weight)
        nn.init.zeros_(self.specialist_out.bias)
        self.register_buffer(
            "specialist_fusion_scale",
            torch.tensor(float(self.cfg.specialist_fusion_scale)),
        )

        # Sole Entry authority.  The raw-bps Q head consumes the learned local,
        # final local+MTF, MTF-only and family-context representations directly.
        # No auxiliary prediction, handcrafted evidence vector or calibrated
        # probability can sit between this joint representation and argmax.
        self.entry_q_joint_norm = nn.LayerNorm(4 * d_model)
        self.entry_q_joint_in = nn.Linear(4 * d_model, d_model)
        self.head_entry_action_q = nn.Linear(d_model, 3)
        self.head_side_mae = nn.Linear(d_model, 2)
        self.head_trendline_event = nn.Linear(
            d_model, EXACT_TRENDLINE_EVENT_OUTPUT_DIM
        )
        self.head_position_size = nn.Linear(d_model, 1)
        self.head_dip = nn.Linear(d_model, DIP_HEAD_DIM)
        self.head_forecast = nn.Linear(d_model, FORECAST_HEAD_DIM)
        self.head_timing = nn.Linear(d_model, TIMING_HEAD_DIM)
        self.head_tail_risk = nn.Linear(d_model, TAIL_RISK_HEAD_DIM)
        self.head_vol_forecast = nn.Linear(d_model, VOL_FORECAST_HEAD_DIM)
        # Neutral equal initialization carries no task or direction preference.
        # The trainer applies the one contract-owned uncertainty formula and
        # these parameters travel inside the ordinary model state_dict.
        self.task_log_variances = nn.ParameterDict(
            {
                task_name: nn.Parameter(torch.zeros((), dtype=torch.float32))
                for task_name in JOINT_TASK_NAMES
            }
        )
        self.entry_decision_token = nn.Sequential(
            nn.LayerNorm(ENTRY_DECISION_TOKEN_SOURCE_DIM),
            nn.Linear(
                ENTRY_DECISION_TOKEN_SOURCE_DIM,
                ENTRY_DECISION_TOKEN_DIM,
            ),
            nn.GELU(),
        )

        nn.init.zeros_(self.head_side_mae.bias)
        nn.init.zeros_(self.head_trendline_event.bias)
        nn.init.xavier_uniform_(self.entry_q_joint_in.weight)
        nn.init.zeros_(self.entry_q_joint_in.bias)
        nn.init.xavier_uniform_(self.head_entry_action_q.weight)
        nn.init.zeros_(self.head_entry_action_q.bias)
        nn.init.xavier_uniform_(self.entry_decision_token[1].weight)
        nn.init.zeros_(self.entry_decision_token[1].bias)

        # One shared five-timeframe-capable, eight-family multi-resolution
        # stack. Entry consumes M15/H1/H4/D1; Exit consumes
        # M5/M15/H1/H4/D1. Each
        # timeframe exposes the same one-owner family surface, but feature gates
        # are independent per family×timeframe and conditioned on the current
        # learned state.  Shared family encoders preserve semantic sample
        # efficiency; axial attention then cooperates both across timeframes
        # within a family and across families within a timeframe.
        mtf_layers = int(self.cfg.multi_tf_num_layers)
        if mtf_layers <= 0:
            raise RuntimeError("MULTI_TF_NUM_LAYERS_INVALID")
        mtf_reference_names = normalization_field_names["mtf_m5"]
        mtf_categorical_indices = self._input_norm_categorical_index_tuples[
            "mtf_m5"
        ]
        for tf_name in TF_INPUT_SCALE_NAMES[1:]:
            surface = f"mtf_{tf_name}"
            if (
                normalization_field_names[surface] != mtf_reference_names
                or self._input_norm_categorical_index_tuples[surface]
                != mtf_categorical_indices
                or self._input_norm_categorical_domains[surface]
                != self._input_norm_categorical_domains["mtf_m5"]
            ):
                raise RuntimeError(
                    "ENTRY_INPUT_NORMALIZATION_MTF_CATEGORICAL_SPLIT_BRAIN"
                )
        self._multi_tf_categorical_index_set = frozenset(
            mtf_categorical_indices
        )
        self._multi_tf_specialist_categorical_positions = {
            name: tuple(
                (local_position, global_index)
                for local_position, global_index in enumerate(
                    self._multi_tf_specialist_index_tuples[name]
                )
                if global_index in self._multi_tf_categorical_index_set
            )
            for name in self._specialist_names
        }
        self.mtf_nominal_embeddings = nn.ModuleDict(
            {
                f"{tf_name}_{index}": nn.Embedding(
                    len(
                        self._input_norm_categorical_domains[f"mtf_{tf_name}"][
                            mtf_reference_names[index]
                        ]
                    ),
                    d_model,
                )
                for tf_name in TF_INPUT_SCALE_NAMES
                for index in mtf_categorical_indices
            }
        )
        self.mtf_family_proj = nn.ModuleDict(
            {
                name: nn.Linear(len(cleaned_mtf[name]), d_model)
                for name in self._specialist_names
            }
        )
        self.mtf_family_encoder = nn.ModuleDict(
            {
                name: _mk_encoder(mtf_layers)
                for name in self._specialist_names
            }
        )
        self.mtf_feature_context_gate = nn.ModuleDict()
        for tf_name in TF_INPUT_SCALE_NAMES:
            for name in self._specialist_names:
                key = f"{tf_name}__{name}"
                gate = nn.Linear(d_model, len(cleaned_mtf[name]))
                # 2*sigmoid(0)=1: every feature×timeframe path starts neutral;
                # no timeframe receives a hand-authored preference.
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
                self.mtf_feature_context_gate[key] = gate
        self.family_axis_attn = _mk_encoder(1)
        self.timeframe_axis_attn = _mk_encoder(1)
        self.family_tf_token_identity = nn.Parameter(
            torch.empty(1, 5, len(self._specialist_names), d_model)
        )
        nn.init.normal_(self.family_tf_token_identity, std=0.02)
        cooperation_token_count = 5 * len(self._specialist_names)
        self.family_tf_context_gate = nn.Linear(
            d_model,
            cooperation_token_count,
        )
        self.family_tf_token_gate = nn.Linear(d_model, 1)
        self.family_tf_cooperation_out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.family_tf_cooperation_out.weight)
        nn.init.zeros_(self.family_tf_cooperation_out.bias)
        self.family_tf_token_order = tuple(
            f"{tf_name}:{specialist}"
            for tf_name in TF_INPUT_SCALE_NAMES
            for specialist in self._specialist_names
        )
        self.entry_family_tf_token_order = tuple(
            f"{tf_name.lower()}:{specialist}"
            for tf_name in ENTRY_MTF_CONTEXT_TIMEFRAMES
            for specialist in self._specialist_names
        )
        self.exit_family_tf_token_order = tuple(
            f"{tf_name.lower()}:{specialist}"
            for tf_name in EXIT_MTF_CONTEXT_TIMEFRAMES
            for specialist in self._specialist_names
        )
        self.cross_tf_attn = _mk_encoder(1)
        self.tf_token_identity = nn.Parameter(torch.empty(1, 5, d_model))
        nn.init.normal_(self.tf_token_identity, std=0.02)
        self.tf_gate_logits = nn.Parameter(torch.zeros(5))
        self.tf_context_gate = nn.Linear(d_model, 5)
        self.tf_token_gate = nn.Linear(d_model, 1)
        self.cross_tf_out = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.cross_tf_out.weight)
        nn.init.zeros_(self.cross_tf_out.bias)
        self._expected_m5_seq_dim = int(self.cfg.m5_seq_dim)
        self._expected_m15_seq_dim = int(self.cfg.m15_seq_dim)
        self._expected_h1_seq_dim = int(self.cfg.h1_seq_dim)
        self._expected_h4_seq_dim = int(self.cfg.h4_seq_dim)
        self._expected_d1_seq_dim = int(self.cfg.d1_seq_dim)
        self.register_buffer("multi_tf_scale", torch.tensor(float(self.cfg.multi_tf_scale)))
        # State keys retain their historical names, but every timeframe starts
        # from the same contract-owned neutral identity.  Learned gates/scales
        # may diverge from data; no wrapper supplies a timeframe preference.
        for tf_name in TF_INPUT_SCALE_NAMES:
            self.register_parameter(
                f"tf_input_scale_{tf_name}",
                nn.Parameter(
                    torch.tensor(
                        raw_tf_input_scale_from_effective(
                            TF_INPUT_SCALE_NEUTRAL_INIT
                        ),
                        dtype=torch.float32,
                    )
                ),
            )
        self.register_buffer(
            "cross_family_fusion_scale",
            torch.tensor(float(self.cfg.cross_family_fusion_scale)),
        )

        # Unified lifecycle head. The full-stack Entry encoder produces the
        # frozen ``z`` below; Exit consumes that exact representation plus the
        # literal causal M1 prefix in this same model.  The path encoder adds
        # post-entry evidence, but cannot replace or bypass the Entry state.
        self.exit_path_proj = nn.Linear(UNIFIED_EXIT_PATH_FEATURE_DIM, d_model)
        self.exit_path_encoder = _mk_encoder(
            UNIFIED_EXIT_PATH_ENCODER_LAYERS
        )
        self.exit_side_embedding = nn.Embedding(2, d_model)
        self.exit_entry_query_norm = nn.LayerNorm(d_model)
        self.exit_entry_path_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.exit_fuse = nn.Sequential(
            nn.LayerNorm(5 * d_model),
            nn.Linear(5 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head_exit_action = nn.Linear(d_model, 2)

        # Episode-native Exit. Entry retains its transformer semantics above;
        # these causal GRU scans are dedicated to the long-lived Exit state.
        # The global route sees the complete signal only to learn cross-family
        # residuals. Each of the eight disjoint family routes sees only its
        # owner fields. No pooled/static temporal summary is used.
        self.exit_episode_global_gru = nn.GRU(
            d_model, d_model, batch_first=True
        )
        self.exit_episode_family_gru = nn.ModuleDict(
            {
                name: nn.GRU(d_model, d_model, batch_first=True)
                for name in self._specialist_names
            }
        )
        self.exit_episode_family_cross_attn = _mk_encoder(1)
        self.exit_episode_family_gate = nn.Linear(
            d_model, len(self._specialist_names)
        )
        self.exit_episode_family_token_gate = nn.Linear(d_model, 1)
        self.exit_episode_family_out = nn.Linear(d_model, d_model)
        self.exit_episode_local_fuse = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )
        # Family encoders are shared across native clocks while each history is
        # scanned independently. Exact per-state gathers happen only after the
        # causal scan, so no configured rolling window is re-encoded.
        self.exit_episode_mtf_family_gru = nn.ModuleDict(
            {
                name: nn.GRU(d_model, d_model, batch_first=True)
                for name in self._specialist_names
            }
        )
        self.exit_episode_family_axis_attn = _mk_encoder(1)
        self.exit_episode_timeframe_axis_attn = _mk_encoder(1)
        self.exit_episode_family_tf_context_gate = nn.Linear(
            d_model, 5 * len(self._specialist_names)
        )
        self.exit_episode_family_tf_token_gate = nn.Linear(d_model, 1)
        self.exit_episode_family_tf_out = nn.Linear(d_model, d_model)
        self.exit_episode_path_gru = nn.GRU(
            d_model, d_model, batch_first=True
        )
        self.exit_episode_fuse = nn.Sequential(
            nn.LayerNorm(5 * d_model),
            nn.Linear(5 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # Strict markers (useful for debugging)
        self._expected_seq_dim = int(seq_input_dim)
        self._expected_snap_dim = int(snap_input_dim)
        self._expected_seq_len = int(seq_len)
        self._expected_ctx_cat_dim = int(self.cfg.ctx_cat_dim)
        self._expected_ctx_cont_dim = int(self.cfg.ctx_cont_dim)
        # Positional encoding is mandatory and covers every sequence branch.
        if int(seq_len) * int(ENTRY_EXIT_RESOLUTION_RATIO) > int(
            EXIT_FEATURE_MAX_SEQUENCE_BARS
        ):
            raise RuntimeError(
                "UNIFIED_EXIT_FEATURE_SEQUENCE_CAPACITY_INVALID: "
                f"entry_seq_len={int(seq_len)} ratio={int(ENTRY_EXIT_RESOLUTION_RATIO)} "
                f"capacity={int(EXIT_FEATURE_MAX_SEQUENCE_BARS)}"
            )
        self.register_buffer("pos_enc", self._sinusoidal_pe(int(seq_len), d_model), persistent=False)
        self.register_buffer(
            "pos_enc_exit_feature",
            self._sinusoidal_pe(int(EXIT_FEATURE_MAX_SEQUENCE_BARS), d_model),
            persistent=False,
        )
        self.register_buffer("pos_enc_m5", self._sinusoidal_pe(int(self.cfg.m5_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_m15", self._sinusoidal_pe(int(self.cfg.m15_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_h1", self._sinusoidal_pe(int(self.cfg.h1_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_h4", self._sinusoidal_pe(int(self.cfg.h4_seq_len), d_model), persistent=False)
        self.register_buffer("pos_enc_d1", self._sinusoidal_pe(int(self.cfg.d1_seq_len), d_model), persistent=False)
        self.register_buffer(
            "pos_enc_exit",
            self._sinusoidal_pe(UNIFIED_EXIT_MAX_PATH_BARS, d_model),
            persistent=False,
        )

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
        if int(t.size(1)) > int(pe.size(1)):
            raise RuntimeError(
                "POSITIONAL_ENCODING_CAPACITY_EXCEEDED: "
                f"buffer={buf_name} sequence={int(t.size(1))} capacity={int(pe.size(1))}"
            )
        return t + pe[:, : t.size(1)]

    def require_input_normalization_state(self) -> None:
        """Prove persistent buffers still match the immutable metadata contract."""

        contract = self._input_normalization_contract
        expected_contract_hash = torch.tensor(
            list(bytes.fromhex(str(contract["contract_sha256"]))),
            dtype=torch.uint8,
        )
        if not torch.equal(
            self.input_norm_contract_sha256.detach().cpu(),
            expected_contract_hash,
        ):
            raise RuntimeError("ENTRY_INPUT_NORMALIZATION_STATE_HASH_MISMATCH")
        for surface in INPUT_NORMALIZATION_SURFACES:
            surface_contract = contract["surfaces"][surface]
            expected = {
                "center": torch.tensor(
                    surface_contract["center"], dtype=torch.float32
                ),
                "scale": torch.tensor(
                    surface_contract["scale"], dtype=torch.float32
                ),
                "binary_mask": torch.tensor(
                    surface_contract["binary_mask"], dtype=torch.bool
                ),
                "categorical_mask": torch.tensor(
                    surface_contract["categorical_mask"], dtype=torch.bool
                ),
            }
            for suffix, expected_value in expected.items():
                observed = getattr(
                    self,
                    f"input_norm_{surface}_{suffix}",
                ).detach().cpu()
                if not torch.equal(observed, expected_value):
                    raise RuntimeError(
                        "ENTRY_INPUT_NORMALIZATION_STATE_BUFFER_MISMATCH: "
                        f"surface={surface} field={suffix}"
                    )
            expected_categorical_indices = tuple(
                index
                for index, is_categorical in enumerate(
                    surface_contract["categorical_mask"]
                )
                if bool(is_categorical)
            )
            if (
                self._input_norm_categorical_index_tuples[surface]
                != expected_categorical_indices
            ):
                raise RuntimeError(
                    "ENTRY_INPUT_NORMALIZATION_CATEGORICAL_INDEX_CACHE_MISMATCH: "
                    f"surface={surface}"
                )
        expected_alias_signal = torch.tensor(
            [
                int(alias["signal_index"])
                for alias in contract["temporal_aliases"]
            ],
            dtype=torch.long,
        )
        expected_alias_ctx = torch.tensor(
            [
                int(alias["ctx_cont_index"])
                for alias in contract["temporal_aliases"]
            ],
            dtype=torch.long,
        )
        if (
            not torch.equal(
                self.input_norm_alias_signal_indices.detach().cpu(),
                expected_alias_signal,
            )
            or not torch.equal(
                self.input_norm_alias_ctx_cont_indices.detach().cpu(),
                expected_alias_ctx,
            )
        ):
            raise RuntimeError("ENTRY_INPUT_NORMALIZATION_STATE_ALIAS_MISMATCH")
        ctx_domains = contract["ctx_cat"]["domains"]
        expected_ctx_min = torch.tensor(
            [min(ctx_domains[name]) for name in MODEL_NATIVE_CTX_CAT_FIELDS],
            dtype=torch.long,
        )
        expected_ctx_max = torch.tensor(
            [max(ctx_domains[name]) for name in MODEL_NATIVE_CTX_CAT_FIELDS],
            dtype=torch.long,
        )
        if (
            not torch.equal(
                self.input_norm_ctx_cat_min.detach().cpu(),
                expected_ctx_min,
            )
            or not torch.equal(
                self.input_norm_ctx_cat_max.detach().cpu(),
                expected_ctx_max,
            )
        ):
            raise RuntimeError(
                "ENTRY_INPUT_NORMALIZATION_STATE_CTX_CAT_DOMAIN_MISMATCH"
            )

    def _normalize_input_surface(
        self,
        raw: torch.Tensor,
        *,
        surface: str,
    ) -> torch.Tensor:
        center = getattr(self, f"input_norm_{surface}_center").to(
            device=raw.device,
            dtype=torch.float32,
        )
        scale = getattr(self, f"input_norm_{surface}_scale").to(
            device=raw.device,
            dtype=torch.float32,
        )
        binary_mask = getattr(
            self,
            f"input_norm_{surface}_binary_mask",
        ).to(raw.device)
        categorical_mask = getattr(
            self,
            f"input_norm_{surface}_categorical_mask",
        ).to(raw.device)
        raw_float = raw.float()
        if int(raw_float.shape[-1]) != int(center.numel()):
            raise RuntimeError(
                f"ENTRY_INPUT_NORMALIZATION_RUNTIME_WIDTH_MISMATCH: {surface}"
            )
        field_names = self._input_normalization_field_names[surface]
        if surface.startswith("mtf_") and "ema_stack_aligned_v2" in field_names:
            ema_stack_index = field_names.index("ema_stack_aligned_v2")
            ema_stack = raw_float[..., ema_stack_index]
            if bool(
                (
                    (ema_stack != -1.0)
                    & (ema_stack != 0.0)
                    & (ema_stack != 1.0)
                )
                .any()
                .item()
            ):
                raise RuntimeError(
                    "ENTRY_INPUT_NORMALIZATION_MTF_EMA_STACK_DOMAIN_INVALID: "
                    f"surface={surface}"
                )
        if bool(binary_mask.any().item()):
            binary_values = raw_float[..., binary_mask]
            if bool(
                ((binary_values != 0.0) & (binary_values != 1.0)).any().item()
            ):
                raise RuntimeError(
                    f"ENTRY_INPUT_NORMALIZATION_BINARY_VALUE_INVALID: {surface}"
                )
        if bool(categorical_mask.any().item()):
            domains = self._input_norm_categorical_domains[surface]
            for index in self._input_norm_categorical_index_tuples[surface]:
                values = raw_float[..., index]
                rounded = values.round()
                domain = domains[field_names[index]]
                if (
                    not torch.equal(values, rounded)
                    or bool(
                        (
                            (rounded < min(domain))
                            | (rounded > max(domain))
                        )
                        .any()
                        .item()
                    )
                ):
                    raise RuntimeError(
                        "ENTRY_INPUT_NORMALIZATION_CATEGORICAL_VALUE_INVALID: "
                        f"surface={surface} field={field_names[index]}"
                    )
        identity_mask = binary_mask | categorical_mask
        # Preserve the mathematical asinh transform over the full finite
        # float32 input domain.  A float32 affine ratio can overflow before
        # asinh maps it back to a small finite value when TRAIN learned a very
        # small but non-zero scale.  The float64 intermediate is converted
        # straight back to the model's float32 activation surface; no input is
        # clipped, saturated or otherwise rewritten.
        affine = (
            raw_float.to(dtype=torch.float64)
            - center.to(dtype=torch.float64)
        ) / scale.to(dtype=torch.float64)
        normalized = torch.where(
            identity_mask,
            raw_float.to(dtype=torch.float64),
            torch.asinh(affine),
        ).to(dtype=raw_float.dtype)
        _assert_finite(f"normalized.{surface}", normalized)
        return normalized

    def _effective_tf_input_scale(self, suffix: str) -> torch.Tensor:
        raw = getattr(self, f"tf_input_scale_{suffix}")
        effective = (
            nn.functional.softplus(raw)
            + float(TF_INPUT_SCALE_MIN_EFFECTIVE)
        )
        _assert_finite(f"tf_input_scale_effective_{suffix}", effective)
        return effective

    def _build_family_context_tokens(
        self,
        ctx_cont: torch.Tensor,
        ctx_cat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(ctx_cont.shape[0])
        cat_emb = torch.stack(
            [
                embedding(ctx_cat[:, index].long())
                for index, embedding in enumerate(self.ctx_cat_embeddings)
            ],
            dim=1,
        )
        family_context_parts = []
        for name in self._specialist_names:
            context_token = torch.zeros(
                batch_size,
                int(self.cfg.d_model),
                device=ctx_cont.device,
                dtype=ctx_cont.dtype,
            )
            cont_idx = getattr(
                self,
                f"specialist_ctx_cont_idx_{name}",
            ).to(ctx_cont.device)
            if int(cont_idx.numel()) > 0:
                cont_values = ctx_cont.float().index_select(
                    dim=1,
                    index=cont_idx,
                )
                context_token = context_token + self.specialist_ctx_cont_proj[
                    name
                ](cont_values) * float(self.cfg.ctx_cont_scale)
            nominal_idx = getattr(
                self,
                f"specialist_ctx_cont_nominal_idx_{name}",
            ).to(ctx_cont.device)
            if int(nominal_idx.numel()) > 0:
                nominal_values = ctx_cont.float().index_select(
                    dim=1,
                    index=nominal_idx,
                )
                rounded = nominal_values.round()
                if (
                    not bool(torch.isfinite(nominal_values).all().item())
                    or not bool(torch.equal(nominal_values, rounded))
                    or bool(((rounded < 0) | (rounded >= 5)).any().item())
                ):
                    raise RuntimeError(
                        "CTX_CONT_NOMINAL_DOMAIN_INVALID: "
                        f"specialist={name} expected_exact_integer_domain=0..4"
                    )
                embedded = torch.cat(
                    [
                        self.specialist_ctx_cont_nominal_embeddings[str(index)](
                            rounded[:, position].long()
                        )
                        for position, index in enumerate(
                            self._specialist_ctx_cont_nominal_index_tuples[name]
                        )
                    ],
                    dim=1,
                )
                context_token = context_token + (
                    self.specialist_ctx_cont_nominal_out[name](embedded)
                    * float(self.cfg.ctx_cont_scale)
                )
            cat_idx = getattr(
                self,
                f"specialist_ctx_cat_idx_{name}",
            ).to(ctx_cat.device)
            if int(cat_idx.numel()) > 0:
                cat_values = cat_emb.index_select(
                    dim=1,
                    index=cat_idx,
                ).reshape(batch_size, -1)
                context_token = context_token + self.specialist_ctx_cat_proj[
                    name
                ](cat_values) * float(self.cfg.ctx_cat_scale)
            if name in self.specialist_context_out:
                context_token = self.specialist_context_out[name](
                    context_token
                )
            family_context_parts.append(context_token)
        family_context_tokens = torch.stack(family_context_parts, dim=1)
        global_context_h = self.family_context_global(
            family_context_tokens.reshape(batch_size, -1)
        )
        _assert_finite("family_context_tokens", family_context_tokens)
        _assert_finite("global_context_h", global_context_h)
        return family_context_tokens, global_context_h

    def _assemble_entry_decision_token_source(
        self,
        components: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Assemble all declared pre-argmax Entry evidence in exact order.

        Component order and widths come from the immutable token contract.  No
        block receives a handwritten scale, vote or direction preference.
        """

        expected_names = tuple(name for name, _width in ENTRY_DECISION_TOKEN_COMPONENTS)
        if tuple(components) != expected_names:
            raise RuntimeError(
                "ENTRY_DECISION_TOKEN_COMPONENT_ORDER_INVALID: "
                f"observed={tuple(components)} expected={expected_names}"
            )
        rows: int | None = None
        ordered: list[torch.Tensor] = []
        for name, width in ENTRY_DECISION_TOKEN_COMPONENTS:
            value = components[name]
            _assert_shape(f"entry_decision_token.{name}", value, 2)
            if int(value.shape[1]) != int(width):
                raise RuntimeError(
                    "ENTRY_DECISION_TOKEN_COMPONENT_WIDTH_INVALID: "
                    f"name={name} observed={tuple(value.shape)} width={width}"
                )
            if rows is None:
                rows = int(value.shape[0])
            elif int(value.shape[0]) != rows:
                raise RuntimeError(
                    "ENTRY_DECISION_TOKEN_COMPONENT_BATCH_MISMATCH: "
                    f"name={name} rows={int(value.shape[0])} expected={rows}"
                )
            _assert_finite(f"entry_decision_token.{name}", value)
            ordered.append(value)
        source = torch.cat(ordered, dim=1)
        if int(source.shape[1]) != ENTRY_DECISION_TOKEN_SOURCE_DIM:
            raise RuntimeError("ENTRY_DECISION_TOKEN_SOURCE_WIDTH_INVALID")
        return source

    def _project_entry_decision_token(
        self,
        components: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the sole learned projection to the exact assembled source."""

        source = self._assemble_entry_decision_token_source(components)
        rows = int(source.shape[0])
        token = self.entry_decision_token(source)
        if tuple(token.shape) != (rows, ENTRY_DECISION_TOKEN_DIM):
            raise RuntimeError("ENTRY_DECISION_TOKEN_OUTPUT_SHAPE_INVALID")
        _assert_finite(UNIFIED_EXIT_MODEL_REPRESENTATION_KEY, token)
        return token

    def _encode_shared_feature_base(
        self,
        *,
        seq_x: torch.Tensor,
        snap_x: torch.Tensor,
        ctx_cat: torch.Tensor,
        ctx_cont: torch.Tensor,
        surface_label: str,
        expected_seq_len: int,
        positional_encoding: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one causal surface with the one shared specialist stack.

        Entry and Exit deliberately call this exact method and module set.  The
        only permitted difference is the causal clock/sequence length: Entry is
        M5 and Exit is five M1 bars per Entry bar.  No Exit-specific feature
        projection or specialist implementation exists here.
        """

        _assert_shape(f"{surface_label}.seq_x", seq_x, 3)
        _assert_shape(f"{surface_label}.snap_x", snap_x, 2)
        _assert_shape(f"{surface_label}.ctx_cat", ctx_cat, 2)
        _assert_shape(f"{surface_label}.ctx_cont", ctx_cont, 2)
        batch_size, sequence_len, signal_dim = seq_x.shape
        if int(sequence_len) != int(expected_seq_len):
            raise RuntimeError(
                f"{surface_label.upper()}_SEQ_LEN_MISMATCH: "
                f"got={int(sequence_len)} expected={int(expected_seq_len)}"
            )
        if int(signal_dim) != self._expected_seq_dim:
            raise RuntimeError(
                f"{surface_label.upper()}_SIGNAL_DIM_MISMATCH: "
                f"got={int(signal_dim)} expected={self._expected_seq_dim}"
            )
        if tuple(snap_x.shape) != (batch_size, self._expected_snap_dim):
            raise RuntimeError(
                f"{surface_label.upper()}_SNAP_SHAPE_MISMATCH: "
                f"got={tuple(snap_x.shape)} expected=({batch_size},{self._expected_snap_dim})"
            )
        if tuple(ctx_cat.shape) != (batch_size, self._expected_ctx_cat_dim):
            raise RuntimeError(
                f"{surface_label.upper()}_CTX_CAT_DIM_MISMATCH: "
                f"got={tuple(ctx_cat.shape)} expected=({batch_size},{self._expected_ctx_cat_dim})"
            )
        if tuple(ctx_cont.shape) != (batch_size, self._expected_ctx_cont_dim):
            raise RuntimeError(
                f"{surface_label.upper()}_CTX_CONT_DIM_MISMATCH: "
                f"got={tuple(ctx_cont.shape)} expected=({batch_size},{self._expected_ctx_cont_dim})"
            )

        _assert_finite(f"{surface_label}.seq_x", seq_x)
        _assert_finite(f"{surface_label}.snap_x", snap_x)
        _assert_finite(f"{surface_label}.ctx_cont", ctx_cont)
        if not torch.equal(seq_x[:, -1, :], snap_x):
            raise RuntimeError(f"{surface_label.upper()}_SEQ_LAST_SNAP_NOT_BIT_IDENTICAL")
        alias_signal_idx = self.temporal_alias_signal_idx.to(snap_x.device)
        alias_ctx_idx = self.temporal_alias_ctx_cont_idx.to(ctx_cont.device)
        if int(alias_signal_idx.numel()) > 0 and not torch.equal(
            snap_x.index_select(dim=1, index=alias_signal_idx),
            ctx_cont.index_select(dim=1, index=alias_ctx_idx),
        ):
            raise RuntimeError(f"{surface_label.upper()}_SNAP_CTX_CONT_ALIAS_NOT_BIT_IDENTICAL")

        if ctx_cat.dtype not in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        ):
            raise RuntimeError(f"{surface_label.upper()}_CTX_CAT_DTYPE_MISMATCH")
        ctx_cat_min = self.input_norm_ctx_cat_min.to(ctx_cat.device)
        ctx_cat_max = self.input_norm_ctx_cat_max.to(ctx_cat.device)
        for index, field in enumerate(EXACT_CTX_CAT_DOMAINS):
            values = ctx_cat[:, index]
            if bool(
                ((values < ctx_cat_min[index]) | (values > ctx_cat_max[index]))
                .any()
                .item()
            ):
                raise RuntimeError(
                    f"{surface_label.upper()}_CTX_CAT_DOMAIN_INVALID: field={field}"
                )

        seq_n = self._normalize_input_surface(seq_x, surface="signal")
        snap_n = self._normalize_input_surface(snap_x, surface="signal")
        ctx_cont_n = self._normalize_input_surface(ctx_cont, surface="ctx_cont")
        signal_categorical_mask = self.input_norm_signal_categorical_mask.to(seq_n.device)
        seq_numeric = seq_n.masked_fill(signal_categorical_mask.view(1, 1, -1), 0.0)
        snap_numeric = snap_n.masked_fill(signal_categorical_mask.view(1, -1), 0.0)

        seq_h = self.seq_proj(seq_numeric)
        for index_text, embedding in self.signal_nominal_embeddings.items():
            index = int(index_text)
            seq_h = seq_h + embedding(seq_x[..., index].long())
        seq_h = self._add_pe(seq_h, positional_encoding)
        seq_h = _memory_bounded_transformer_encoder(self.encoder, seq_h)
        seq_pool = seq_h.mean(dim=1)

        generic_snap_idx = self.generic_snap_idx.to(snap_x.device)
        snap_h = self.snap_proj(
            snap_numeric.index_select(dim=1, index=generic_snap_idx)
        )
        generic_snap_set = self._generic_snap_index_set
        for index_text, embedding in self.signal_nominal_embeddings.items():
            index = int(index_text)
            if index in generic_snap_set:
                snap_h = snap_h + embedding(snap_x[:, index].long())

        family_context_tokens, global_context_h = self._build_family_context_tokens(
            ctx_cont_n,
            ctx_cat,
        )
        z_v3 = self.fuse(torch.cat([seq_pool, snap_h, global_context_h], dim=1))

        pools = []
        for specialist_position, name in enumerate(self._specialist_names):
            idx = getattr(self, f"specialist_idx_{name}").to(seq_x.device)
            seq_part = seq_numeric.index_select(dim=2, index=idx)
            spec_h = self._add_pe(
                self.specialist_proj[name](seq_part),
                positional_encoding,
            )
            specialist_index_set = self._specialist_input_index_sets[name]
            for index_text, embedding in self.signal_nominal_embeddings.items():
                index = int(index_text)
                if index in specialist_index_set:
                    spec_h = spec_h + embedding(seq_x[..., index].long())
            temporal_pool = _memory_bounded_transformer_encoder(
                self.specialist_encoder[name],
                spec_h,
            ).mean(dim=1)
            pools.append(temporal_pool + family_context_tokens[:, specialist_position, :])
        specialist_tokens = torch.stack(pools, dim=1)
        specialist_tokens = _memory_bounded_transformer_encoder(
            self.specialist_cross_attn,
            specialist_tokens + self.specialist_token_identity,
        )
        specialist_gate = torch.softmax(
            self.specialist_gate(z_v3)
            + self.specialist_token_gate(specialist_tokens).squeeze(-1),
            dim=1,
        )
        specialist_pool = (
            specialist_tokens * specialist_gate.unsqueeze(-1)
        ).sum(dim=1)
        specialist_correction = self.specialist_out(specialist_pool)
        _assert_finite(f"{surface_label}.specialist_correction", specialist_correction)
        _assert_finite(f"{surface_label}.specialist_gate", specialist_gate)
        z_v3 = z_v3 + self.specialist_fusion_scale.to(
            specialist_correction.dtype
        ) * specialist_correction
        _assert_finite(f"{surface_label}.feature_base_representation", z_v3)
        return z_v3, specialist_gate, global_context_h

    def _encode_multi_tf_route(
        self,
        *,
        base_representation: torch.Tensor,
        tf_inputs: Tuple[Tuple[str, torch.Tensor, int, int], ...],
        route_timeframes: tuple[str, ...],
        route_label: str,
    ) -> Dict[str, torch.Tensor]:
        """Fuse one canonical route with the single shared V4 MTF stack."""

        _assert_shape(f"{route_label}.base_representation", base_representation, 2)
        batch_size = int(base_representation.shape[0])
        route = tuple(tf.lower() for tf in route_timeframes)
        expected_routes = {
            "entry": tuple(tf.lower() for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES),
            "exit": tuple(tf.lower() for tf in EXIT_MTF_CONTEXT_TIMEFRAMES),
        }
        if route_label not in expected_routes or route != expected_routes[route_label]:
            raise RuntimeError(
                f"MODEL_NATIVE_MTF_ROUTE_INVALID: label={route_label!r} route={route!r}"
            )
        if tuple(item[0] for item in tf_inputs) != route:
            raise RuntimeError(
                f"MODEL_NATIVE_MTF_INPUT_ORDER_INVALID: label={route_label!r}"
            )
        route_count = len(route)
        route_indices = torch.tensor(
            [TF_INPUT_SCALE_NAMES.index(tf) for tf in route],
            dtype=torch.long,
            device=base_representation.device,
        )
        for suffix, tensor, expected_len, expected_dim in tf_inputs:
            name = f"{route_label}_seq_{suffix}"
            _assert_shape(name, tensor, 3)
            if int(tensor.shape[0]) != batch_size:
                raise RuntimeError(f"{name.upper()}_BATCH_MISMATCH")
            if int(tensor.shape[1]) != int(expected_len):
                raise RuntimeError(
                    f"{name.upper()}_LEN_MISMATCH: "
                    f"got={int(tensor.shape[1])} expected={expected_len}"
                )
            if int(tensor.shape[2]) != int(expected_dim):
                raise RuntimeError(
                    f"{name.upper()}_DIM_MISMATCH: "
                    f"got={int(tensor.shape[2])} expected={expected_dim}"
                )
            _assert_finite(name, tensor)

        family_grid_rows: list[torch.Tensor] = []
        feature_gate_rows: list[torch.Tensor] = []
        for suffix, tensor, _expected_len, expected_dim in tf_inputs:
            surface = f"mtf_{suffix}"
            normalized_tf = self._normalize_input_surface(tensor, surface=surface)
            categorical_mask = getattr(
                self,
                f"input_norm_{surface}_categorical_mask",
            ).to(normalized_tf.device)
            numeric_tf = normalized_tf.masked_fill(
                categorical_mask.view(1, 1, -1),
                0.0,
            )
            effective_scale = self._effective_tf_input_scale(suffix)
            full_feature_gate = torch.zeros(
                batch_size,
                int(expected_dim),
                device=tensor.device,
                dtype=normalized_tf.dtype,
            )
            family_tokens: list[torch.Tensor] = []
            for specialist in self._specialist_names:
                family_idx = getattr(
                    self,
                    f"multi_tf_specialist_idx_{specialist}",
                ).to(tensor.device)
                feature_gate = 2.0 * torch.sigmoid(
                    self.mtf_feature_context_gate[
                        f"{suffix}__{specialist}"
                    ](base_representation)
                )
                full_feature_gate = full_feature_gate.scatter(
                    1,
                    family_idx.view(1, -1).expand(batch_size, -1),
                    feature_gate,
                )
                family_values = numeric_tf.index_select(dim=2, index=family_idx)
                family_values = (
                    family_values
                    * feature_gate.unsqueeze(1)
                    * effective_scale
                )
                projected = self.mtf_family_proj[specialist](family_values)
                for local_position, global_index in (
                    self._multi_tf_specialist_categorical_positions[specialist]
                ):
                    nominal = self.mtf_nominal_embeddings[
                        f"{suffix}_{global_index}"
                    ](tensor[..., global_index].long())
                    projected = projected + nominal * (
                        feature_gate[:, local_position].view(batch_size, 1, 1)
                        * effective_scale
                    )
                encoded = _memory_bounded_transformer_encoder(
                    self.mtf_family_encoder[specialist],
                    self._add_pe(projected, f"pos_enc_{suffix}"),
                )
                family_tokens.append(encoded.mean(dim=1))
            family_grid_rows.append(torch.stack(family_tokens, dim=1))
            feature_gate_rows.append(full_feature_gate)

        family_tf_feature_gate = torch.stack(feature_gate_rows, dim=1)
        family_grid = torch.stack(family_grid_rows, dim=1)
        expected_grid_shape = (route_count, len(self._specialist_names))
        if tuple(family_grid.shape[1:3]) != expected_grid_shape:
            raise RuntimeError(
                f"{route_label.upper()}_FAMILY_TF_GRID_SHAPE_INVALID: "
                f"observed={tuple(family_grid.shape)} expected={expected_grid_shape}"
            )
        route_family_identity = self.family_tf_token_identity.index_select(
            1,
            route_indices,
        )
        family_grid = family_grid + route_family_identity
        family_axis = family_grid.permute(0, 2, 1, 3).reshape(
            batch_size * len(self._specialist_names),
            route_count,
            int(self.cfg.d_model),
        )
        family_axis = _memory_bounded_transformer_encoder(
            self.family_axis_attn,
            family_axis,
        )
        family_grid = family_axis.reshape(
            batch_size,
            len(self._specialist_names),
            route_count,
            int(self.cfg.d_model),
        ).permute(0, 2, 1, 3)
        timeframe_axis = family_grid.reshape(
            batch_size * route_count,
            len(self._specialist_names),
            int(self.cfg.d_model),
        )
        timeframe_axis = _memory_bounded_transformer_encoder(
            self.timeframe_axis_attn,
            timeframe_axis,
        )
        family_grid = timeframe_axis.reshape(
            batch_size,
            route_count,
            len(self._specialist_names),
            int(self.cfg.d_model),
        )
        _assert_finite(f"{route_label}.family_tf_feature_gate", family_tf_feature_gate)
        _assert_finite(f"{route_label}.family_grid", family_grid)

        cooperation_tokens = family_grid.reshape(
            batch_size,
            route_count * len(self._specialist_names),
            int(self.cfg.d_model),
        )
        full_family_context_logits = self.family_tf_context_gate(
            base_representation
        ).reshape(
            batch_size,
            len(TF_INPUT_SCALE_NAMES),
            len(self._specialist_names),
        )
        route_family_context_logits = full_family_context_logits.index_select(
            1,
            route_indices,
        ).reshape(batch_size, -1)
        family_tf_cooperation_gate = torch.softmax(
            route_family_context_logits
            + self.family_tf_token_gate(cooperation_tokens).squeeze(-1),
            dim=1,
        )
        family_gate_by_tf = family_tf_cooperation_gate.reshape(
            batch_size,
            route_count,
            len(self._specialist_names),
        )
        family_gate_within_tf = family_gate_by_tf / family_gate_by_tf.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1e-12)
        tf_tokens = (
            family_grid * family_gate_within_tf.unsqueeze(-1)
        ).sum(dim=2)
        tf_attended = _memory_bounded_transformer_encoder(
            self.cross_tf_attn,
            tf_tokens + self.tf_token_identity.index_select(1, route_indices),
        )
        tf_gate = torch.softmax(
            self.tf_gate_logits.index_select(0, route_indices).view(1, -1)
            + self.tf_context_gate(base_representation).index_select(
                1,
                route_indices,
            )
            + self.tf_token_gate(tf_attended).squeeze(-1),
            dim=1,
        )
        mtf_repr = (tf_attended * tf_gate.unsqueeze(-1)).sum(dim=1)
        mtf_correction = self.cross_tf_out(mtf_repr)
        cooperation_pool = (
            cooperation_tokens * family_tf_cooperation_gate.unsqueeze(-1)
        ).sum(dim=1)
        cooperation_correction = self.family_tf_cooperation_out(cooperation_pool)
        for name, value in (
            ("mtf_repr", mtf_repr),
            ("mtf_correction", mtf_correction),
            ("tf_gate", tf_gate),
            ("family_tf_cooperation_gate", family_tf_cooperation_gate),
            ("cooperation_correction", cooperation_correction),
        ):
            _assert_finite(f"{route_label}.{name}", value)
        representation = (
            base_representation
            + self.multi_tf_scale.to(mtf_correction.dtype) * mtf_correction
            + self.cross_family_fusion_scale.to(cooperation_correction.dtype)
            * cooperation_correction
        )
        _assert_finite(f"{route_label}.mtf_fused_representation", representation)
        return {
            "representation": representation,
            "mtf_repr": mtf_repr,
            "tf_gate": tf_gate,
            "family_tf_cooperation_gate": family_tf_cooperation_gate,
            "family_tf_feature_gate": family_tf_feature_gate,
        }

    def _forward_exit_causal_episode(
        self,
        *,
        entry_decision_representation: torch.Tensor,
        exit_local_history_x: torch.Tensor,
        exit_state_ctx_cat: torch.Tensor,
        exit_state_ctx_cont: torch.Tensor,
        exit_path_x: torch.Tensor,
        exit_mtf_histories: Mapping[str, torch.Tensor],
        exit_mtf_gathers: Mapping[str, torch.Tensor],
        exit_mtf_history_lengths: Mapping[str, torch.Tensor],
        require_full_episode: bool,
    ) -> Dict[str, torch.Tensor]:
        """Shared causal scan for full episodes and online prefix replay."""

        _assert_shape(
            "entry_decision_representation",
            entry_decision_representation,
            2,
        )
        _assert_shape("exit_local_history_x", exit_local_history_x, 3)
        _assert_shape("exit_state_ctx_cat", exit_state_ctx_cat, 3)
        _assert_shape("exit_state_ctx_cont", exit_state_ctx_cont, 3)
        _assert_shape("exit_path_x", exit_path_x, 4)
        batch_size = int(entry_decision_representation.shape[0])
        state_count = int(exit_state_ctx_cont.shape[1])
        warm_rows = EXIT_FEATURE_SEQUENCE_BARS - 1
        d_model = int(self.cfg.d_model)
        if (
            int(entry_decision_representation.shape[1]) != d_model
            or not 1 <= state_count <= UNIFIED_EXIT_EPISODE_STATE_COUNT
            or tuple(exit_local_history_x.shape)
            != (
                batch_size,
                warm_rows + state_count,
                self._expected_seq_dim,
            )
            or tuple(exit_state_ctx_cont.shape)
            != (batch_size, state_count, self._expected_ctx_cont_dim)
            or tuple(exit_state_ctx_cat.shape)
            != (batch_size, state_count, self._expected_ctx_cat_dim)
            or tuple(exit_path_x.shape)
            != (
                batch_size,
                2,
                state_count,
                UNIFIED_EXIT_PATH_FEATURE_DIM,
            )
            or (
                require_full_episode
                and (
                    state_count != UNIFIED_EXIT_EPISODE_STATE_COUNT
                    or int(exit_local_history_x.shape[1])
                    != UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS
                )
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_INPUT_SHAPE_INVALID")
        for name, value in (
            ("entry_decision_representation", entry_decision_representation),
            ("exit_local_history_x", exit_local_history_x),
            ("exit_state_ctx_cont", exit_state_ctx_cont),
            ("exit_path_x", exit_path_x),
        ):
            _assert_finite(name, value)
        if exit_state_ctx_cat.dtype not in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        ):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_CTX_CAT_DTYPE_INVALID")

        signal_n = self._normalize_input_surface(
            exit_local_history_x, surface="signal"
        )
        categorical_mask = self.input_norm_signal_categorical_mask.to(
            signal_n.device
        )
        signal_numeric = signal_n.masked_fill(
            categorical_mask.view(1, 1, -1), 0.0
        )
        global_input = self.seq_proj(signal_numeric)
        for index_text, embedding in self.signal_nominal_embeddings.items():
            index = int(index_text)
            global_input = global_input + embedding(
                exit_local_history_x[..., index].long()
            )
        global_sequence, _ = self.exit_episode_global_gru(global_input)
        global_state = global_sequence[:, warm_rows:, :]

        family_sequences: list[torch.Tensor] = []
        for name in self._specialist_names:
            indices = getattr(self, f"specialist_idx_{name}").to(
                signal_numeric.device
            )
            family_input = self.specialist_proj[name](
                signal_numeric.index_select(2, indices)
            )
            owned = self._specialist_input_index_sets[name]
            for index_text, embedding in self.signal_nominal_embeddings.items():
                index = int(index_text)
                if index in owned:
                    family_input = family_input + embedding(
                        exit_local_history_x[..., index].long()
                    )
            family_sequence, _ = self.exit_episode_family_gru[name](
                family_input
            )
            family_sequences.append(family_sequence[:, warm_rows:, :])
        family_temporal = torch.stack(family_sequences, dim=2)

        flat_ctx_cont = exit_state_ctx_cont.reshape(
            batch_size * state_count, -1
        )
        flat_ctx_cat = exit_state_ctx_cat.reshape(batch_size * state_count, -1)
        ctx_cont_n = self._normalize_input_surface(
            flat_ctx_cont, surface="ctx_cont"
        )
        family_context, global_context = self._build_family_context_tokens(
            ctx_cont_n,
            flat_ctx_cat,
        )
        family_context = family_context.reshape(
            batch_size, state_count, len(self._specialist_names), d_model
        )
        global_context = global_context.reshape(
            batch_size, state_count, d_model
        )
        family_tokens = family_temporal + family_context
        family_tokens = self.exit_episode_family_cross_attn(
            family_tokens.reshape(
                batch_size * state_count,
                len(self._specialist_names),
                d_model,
            )
            + self.specialist_token_identity
        ).reshape(
            batch_size, state_count, len(self._specialist_names), d_model
        )
        current_raw = exit_local_history_x[:, warm_rows:, :]
        current_n = signal_n[:, warm_rows:, :]
        current_numeric = current_n.masked_fill(
            categorical_mask.view(1, 1, -1), 0.0
        )
        generic_idx = self.generic_snap_idx.to(current_numeric.device)
        current_snap = self.snap_proj(
            current_numeric.index_select(2, generic_idx)
        )
        generic_set = self._generic_snap_index_set
        for index_text, embedding in self.signal_nominal_embeddings.items():
            index = int(index_text)
            if index in generic_set:
                current_snap = current_snap + embedding(
                    current_raw[..., index].long()
                )
        local_state = self.exit_episode_local_fuse(
            torch.cat((global_state, current_snap, global_context), dim=2)
        )
        family_gate = torch.softmax(
            self.exit_episode_family_gate(local_state)
            + self.exit_episode_family_token_gate(family_tokens).squeeze(-1),
            dim=2,
        )
        family_correction = self.exit_episode_family_out(
            (family_tokens * family_gate.unsqueeze(-1)).sum(dim=2)
        )
        local_state = local_state + self.specialist_fusion_scale.to(
            local_state.dtype
        ) * family_correction

        expected_tf_names = tuple(
            timeframe.lower() for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES
        )
        if (
            tuple(exit_mtf_histories) != expected_tf_names
            or tuple(exit_mtf_gathers) != expected_tf_names
            or tuple(exit_mtf_history_lengths) != expected_tf_names
        ):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_MTF_ORDER_INVALID")
        tf_family_states: list[torch.Tensor] = []
        tf_feature_gate_rows: list[torch.Tensor] = []
        for tf_name in expected_tf_names:
            history = exit_mtf_histories[tf_name]
            gather = exit_mtf_gathers[tf_name]
            history_lengths = exit_mtf_history_lengths[tf_name]
            _assert_shape(f"exit_mtf_history_{tf_name}", history, 3)
            _assert_shape(f"exit_mtf_gather_{tf_name}", gather, 2)
            if (
                int(history.shape[0]) != batch_size
                or int(history.shape[2])
                != getattr(self, f"_expected_{tf_name}_seq_dim")
                or tuple(gather.shape) != (batch_size, state_count)
                or tuple(history_lengths.shape) != (batch_size,)
                or history_lengths.dtype not in (torch.int64, torch.int32)
                or gather.dtype not in (torch.int64, torch.int32)
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_EPISODE_MTF_INPUT_INVALID:{tf_name}"
                )
            invalid_gather_or_length = (
                (history_lengths < 1).any()
                | (history_lengths > int(history.shape[1])).any()
                | (gather < 0).any()
                | (gather >= int(history.shape[1])).any()
                | (gather >= history_lengths[:, None]).any()
                | (gather[:, 1:] < gather[:, :-1]).any()
            )
            if bool(invalid_gather_or_length.item()):
                raise RuntimeError(
                    f"UNIFIED_EXIT_EPISODE_MTF_INPUT_INVALID:{tf_name}"
                )
            padding_rows = torch.arange(
                int(history.shape[1]), device=history.device
            )[None, :] >= history_lengths[:, None]
            if bool((history[padding_rows] != 0).any().item()):
                raise RuntimeError(
                    f"UNIFIED_EXIT_EPISODE_MTF_PADDING_INVALID:{tf_name}"
                )
            _assert_finite(f"exit_mtf_history_{tf_name}", history)
            surface = f"mtf_{tf_name}"
            normalized = self._normalize_input_surface(
                history, surface=surface
            )
            mtf_cat_mask = getattr(
                self, f"input_norm_{surface}_categorical_mask"
            ).to(normalized.device)
            numeric = normalized.masked_fill(
                mtf_cat_mask.view(1, 1, -1), 0.0
            )
            family_states: list[torch.Tensor] = []
            current_field_index = gather.unsqueeze(-1).expand(
                -1, -1, int(history.shape[2])
            )
            current_numeric_fields = numeric.gather(1, current_field_index)
            current_raw_fields = history.gather(1, current_field_index)
            full_feature_gate = torch.zeros(
                batch_size,
                state_count,
                int(history.shape[2]),
                dtype=numeric.dtype,
                device=numeric.device,
            )
            for name in self._specialist_names:
                indices = getattr(
                    self, f"multi_tf_specialist_idx_{name}"
                ).to(numeric.device)
                projected = self.mtf_family_proj[name](
                    numeric.index_select(2, indices)
                )
                for _local_position, global_index in (
                    self._multi_tf_specialist_categorical_positions[name]
                ):
                    projected = projected + self.mtf_nominal_embeddings[
                        f"{tf_name}_{global_index}"
                    ](history[..., global_index].long())
                encoded, _ = self.exit_episode_mtf_family_gru[name](
                    projected
                    * self._effective_tf_input_scale(tf_name).to(
                        projected.dtype
                    )
                )
                gather_index = gather.unsqueeze(-1).expand(-1, -1, d_model)
                gathered_state = encoded.gather(1, gather_index)
                feature_gate = 2.0 * torch.sigmoid(
                    self.mtf_feature_context_gate[f"{tf_name}__{name}"](
                        local_state.reshape(batch_size * state_count, d_model)
                    )
                ).reshape(batch_size, state_count, -1)
                full_feature_gate = full_feature_gate.scatter(
                    2,
                    indices.view(1, 1, -1).expand(
                        batch_size, state_count, -1
                    ),
                    feature_gate,
                )
                current_owned_numeric = current_numeric_fields.index_select(
                    2, indices
                )
                current_residual = self.mtf_family_proj[name](
                    current_owned_numeric
                    * feature_gate
                    * self._effective_tf_input_scale(tf_name).to(
                        current_owned_numeric.dtype
                    )
                )
                for local_position, global_index in (
                    self._multi_tf_specialist_categorical_positions[name]
                ):
                    nominal = self.mtf_nominal_embeddings[
                        f"{tf_name}_{global_index}"
                    ](current_raw_fields[..., global_index].long())
                    current_residual = current_residual + nominal * (
                        feature_gate[..., local_position].unsqueeze(-1)
                    )
                family_states.append(gathered_state + current_residual)
            tf_family_states.append(torch.stack(family_states, dim=2))
            tf_feature_gate_rows.append(full_feature_gate)
        family_tf = torch.stack(tf_family_states, dim=2)
        identity = self.family_tf_token_identity[:, : len(expected_tf_names)]
        family_tf = family_tf + identity.unsqueeze(1)
        family_rows = family_tf.reshape(
            batch_size * state_count * len(expected_tf_names),
            len(self._specialist_names),
            d_model,
        )
        family_rows = _apply_exit_token_axis_encoder(
            self.exit_episode_family_axis_attn,
            family_rows,
        )
        family_axis = family_rows.reshape(
            batch_size,
            state_count,
            len(expected_tf_names),
            len(self._specialist_names),
            d_model,
        )
        timeframe_rows = family_axis.permute(0, 1, 3, 2, 4).reshape(
            batch_size * state_count * len(self._specialist_names),
            len(expected_tf_names),
            d_model,
        )
        timeframe_rows = _apply_exit_token_axis_encoder(
            self.exit_episode_timeframe_axis_attn,
            timeframe_rows,
        )
        cooperation_tokens = timeframe_rows.reshape(
            batch_size,
            state_count,
            len(self._specialist_names),
            len(expected_tf_names),
            d_model,
        ).permute(0, 1, 3, 2, 4)
        cooperation_logits = self.exit_episode_family_tf_context_gate(
            local_state
        ).reshape(
            batch_size,
            state_count,
            len(expected_tf_names),
            len(self._specialist_names),
        ) + self.exit_episode_family_tf_token_gate(
            cooperation_tokens
        ).squeeze(-1)
        cooperation_gate = torch.softmax(
            cooperation_logits.reshape(batch_size, state_count, -1), dim=2
        ).reshape_as(cooperation_logits)
        family_tf_feature_gate = torch.stack(tf_feature_gate_rows, dim=2)
        mtf_correction = self.exit_episode_family_tf_out(
            (cooperation_tokens * cooperation_gate.unsqueeze(-1)).sum(
                dim=(2, 3)
            )
        )
        market_state = local_state + self.multi_tf_scale.to(
            local_state.dtype
        ) * mtf_correction

        path_flat = exit_path_x.reshape(
            batch_size * 2, state_count, UNIFIED_EXIT_PATH_FEATURE_DIM
        )
        path_encoded, _ = self.exit_episode_path_gru(
            self.exit_path_proj(path_flat)
        )
        path_encoded = path_encoded.reshape(
            batch_size, 2, state_count, d_model
        )
        side_index = torch.arange(2, device=exit_path_x.device).view(1, 2)
        side_state = self.exit_side_embedding(side_index).view(
            1, 2, 1, d_model
        ).expand(batch_size, -1, state_count, -1)
        token = entry_decision_representation.view(
            batch_size, 1, 1, d_model
        ).expand(-1, 2, state_count, -1)
        local_side = local_state.unsqueeze(1).expand(-1, 2, -1, -1)
        mtf_side = mtf_correction.unsqueeze(1).expand(-1, 2, -1, -1)
        hidden = self.exit_episode_fuse(
            torch.cat(
                (token, local_side, mtf_side, side_state, path_encoded), dim=3
            )
        )
        q_values = self.head_exit_action(hidden)
        valid = torch.ones_like(q_values, dtype=torch.bool)
        terminal_mask = torch.zeros(
            (batch_size, 2, state_count),
            dtype=torch.bool,
            device=q_values.device,
        )
        terminal_reason_index = torch.zeros(
            (batch_size, 2, state_count),
            dtype=torch.long,
            device=q_values.device,
        )
        if state_count == UNIFIED_EXIT_EPISODE_STATE_COUNT:
            valid[:, :, -1, 0] = False
            terminal_mask[:, :, -1] = True
            # 1 = current capacity terminal. Zero means non-terminal. This is
            # explicit so a later economic/data terminal contract can replace
            # capacity without changing recurrent encoder semantics.
            terminal_reason_index[:, :, -1] = 1
        for name, value in (
            ("exit_episode_local_state", local_state),
            ("exit_episode_family_gate", family_gate),
            ("exit_episode_family_tf_gate", cooperation_gate),
            ("exit_episode_path_state", path_encoded),
            ("exit_action_q_bps", q_values),
        ):
            _assert_finite(name, value)
        return {
            "exit_action_q_bps": q_values,
            "exit_action_valid_mask": valid,
            "exit_episode_lengths": torch.full(
                (batch_size, 2),
                state_count,
                dtype=torch.long,
                device=q_values.device,
            ),
            "exit_terminal_mask": terminal_mask,
            "exit_terminal_reason_index": terminal_reason_index,
            "exit_specialist_gate": family_gate,
            "exit_tf_gate": cooperation_gate.sum(dim=3),
            "exit_family_tf_cooperation_gate": cooperation_gate,
            "exit_family_tf_feature_gate": family_tf_feature_gate,
            "exit_episode_market_state": market_state,
            "exit_episode_path_state": path_encoded,
        }

    def forward_exit_episode(
        self,
        *,
        entry_decision_representation: torch.Tensor,
        exit_local_history_x: torch.Tensor,
        exit_state_ctx_cat: torch.Tensor,
        exit_state_ctx_cont: torch.Tensor,
        exit_path_x: torch.Tensor,
        exit_mtf_histories: Mapping[str, torch.Tensor],
        exit_mtf_gathers: Mapping[str, torch.Tensor],
        exit_mtf_history_lengths: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Emit [B,2,512,2] Q values from one causal episode scan."""

        return self._forward_exit_causal_episode(
            entry_decision_representation=entry_decision_representation,
            exit_local_history_x=exit_local_history_x,
            exit_state_ctx_cat=exit_state_ctx_cat,
            exit_state_ctx_cont=exit_state_ctx_cont,
            exit_path_x=exit_path_x,
            exit_mtf_histories=exit_mtf_histories,
            exit_mtf_gathers=exit_mtf_gathers,
            exit_mtf_history_lengths=exit_mtf_history_lengths,
            require_full_episode=True,
        )

    def forward_exit_incremental_prefix(
        self,
        *,
        entry_decision_representation: torch.Tensor,
        exit_local_history_x: torch.Tensor,
        exit_state_ctx_cat: torch.Tensor,
        exit_state_ctx_cont: torch.Tensor,
        exit_path_x: torch.Tensor,
        exit_mtf_histories: Mapping[str, torch.Tensor],
        exit_mtf_gathers: Mapping[str, torch.Tensor],
        exit_mtf_history_lengths: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Online/replay prefix owner with the exact episode weights/state."""

        return self._forward_exit_causal_episode(
            entry_decision_representation=entry_decision_representation,
            exit_local_history_x=exit_local_history_x,
            exit_state_ctx_cat=exit_state_ctx_cat,
            exit_state_ctx_cont=exit_state_ctx_cont,
            exit_path_x=exit_path_x,
            exit_mtf_histories=exit_mtf_histories,
            exit_mtf_gathers=exit_mtf_gathers,
            exit_mtf_history_lengths=exit_mtf_history_lengths,
            require_full_episode=False,
        )

    def export_exit_incremental_carry_tensor_state(
        self, carry: UnifiedExitIncrementalCarry
    ) -> dict[str, torch.Tensor]:
        """Flatten one recurrent carry into its exact persistence surface."""

        if not isinstance(carry, UnifiedExitIncrementalCarry):
            raise RuntimeError("UNIFIED_EXIT_INCREMENTAL_CARRY_TYPE_INVALID")
        tensors = {
            "local_global_hidden": carry.local_global_hidden,
            "path_hidden": carry.path_hidden,
        }
        tensors.update(
            {
                f"local_family_hidden/{family}": carry.local_family_hidden[
                    family
                ]
                for family in self._specialist_names
            }
        )
        for tf_name in ("m5", "m15", "h1", "h4", "d1"):
            tensors[f"mtf_current_raw/{tf_name}"] = carry.mtf_current_raw[
                tf_name
            ]
            tensors.update(
                {
                    f"mtf_family_hidden/{tf_name}/{family}": (
                        carry.mtf_family_hidden[tf_name][family]
                    )
                    for family in self._specialist_names
                }
            )
        self._require_exit_incremental_carry_tensor_state(
            tensors,
            batch_size=carry.batch_size,
        )
        return {name: value.detach() for name, value in tensors.items()}

    def _require_exit_incremental_carry_tensor_state(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        d_model = int(self.cfg.d_model)
        expected_shapes: dict[str, tuple[int, ...]] = {
            "local_global_hidden": (1, batch_size, d_model),
            "path_hidden": (1, batch_size * 2, d_model),
        }
        expected_shapes.update(
            {
                f"local_family_hidden/{family}": (1, batch_size, d_model)
                for family in self._specialist_names
            }
        )
        for tf_name in ("m5", "m15", "h1", "h4", "d1"):
            expected_shapes[f"mtf_current_raw/{tf_name}"] = (
                batch_size,
                int(getattr(self, f"_expected_{tf_name}_seq_dim")),
            )
            expected_shapes.update(
                {
                    f"mtf_family_hidden/{tf_name}/{family}": (
                        1,
                        batch_size,
                        d_model,
                    )
                    for family in self._specialist_names
                }
            )
        if set(tensors) != set(expected_shapes):
            raise RuntimeError("UNIFIED_EXIT_INCREMENTAL_CARRY_KEYS_INVALID")
        for name, expected_shape in expected_shapes.items():
            value = tensors[name]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected_shape
                or value.dtype != torch.float32
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_INCREMENTAL_CARRY_TENSOR_INVALID:{name}"
                )
            _assert_finite(name, value)

    def restore_exit_incremental_carry_tensor_state(
        self,
        *,
        step_count: int,
        batch_size: int,
        tensors: Mapping[str, torch.Tensor],
    ) -> UnifiedExitIncrementalCarry:
        """Restore only a byte-validated exact model-shaped carry."""

        if (
            isinstance(step_count, bool)
            or not isinstance(step_count, int)
            or not 1 <= step_count <= UNIFIED_EXIT_EPISODE_STATE_COUNT
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise RuntimeError("UNIFIED_EXIT_INCREMENTAL_CARRY_IDENTITY_INVALID")
        values = {name: value.to(torch.float32) for name, value in tensors.items()}
        self._require_exit_incremental_carry_tensor_state(
            values, batch_size=batch_size
        )
        return UnifiedExitIncrementalCarry(
            step_count=step_count,
            batch_size=batch_size,
            local_global_hidden=values["local_global_hidden"],
            local_family_hidden={
                family: values[f"local_family_hidden/{family}"]
                for family in self._specialist_names
            },
            mtf_family_hidden={
                tf_name: {
                    family: values[
                        f"mtf_family_hidden/{tf_name}/{family}"
                    ]
                    for family in self._specialist_names
                }
                for tf_name in ("m5", "m15", "h1", "h4", "d1")
            },
            mtf_current_raw={
                tf_name: values[f"mtf_current_raw/{tf_name}"]
                for tf_name in ("m5", "m15", "h1", "h4", "d1")
            },
            path_hidden=values["path_hidden"],
        )

    def forward_exit_incremental_step(
        self,
        *,
        entry_decision_representation: torch.Tensor,
        exit_local_rows_x: torch.Tensor,
        exit_state_ctx_cat: torch.Tensor,
        exit_state_ctx_cont: torch.Tensor,
        exit_path_row_x: torch.Tensor,
        exit_mtf_new_rows: Mapping[str, torch.Tensor],
        carry: Optional[UnifiedExitIncrementalCarry],
    ) -> tuple[Dict[str, torch.Tensor], UnifiedExitIncrementalCarry]:
        """Advance one closed M1 state with genuine recurrent carry.

        The first call consumes exactly 479 pre-entry rows plus state zero.
        Later calls consume one new M1 row.  Each MTF value contains only bars
        newly closed since the previous call and may therefore have length
        zero.  Hidden tensors are never detached by this owner.
        """

        if self.training:
            raise RuntimeError("UNIFIED_EXIT_INCREMENTAL_STEP_REQUIRES_EVAL")
        _assert_shape("entry_decision_representation", entry_decision_representation, 2)
        _assert_shape("exit_local_rows_x", exit_local_rows_x, 3)
        _assert_shape("exit_state_ctx_cat", exit_state_ctx_cat, 2)
        _assert_shape("exit_state_ctx_cont", exit_state_ctx_cont, 2)
        _assert_shape("exit_path_row_x", exit_path_row_x, 3)
        batch_size = int(entry_decision_representation.shape[0])
        d_model = int(self.cfg.d_model)
        first = carry is None
        expected_local_rows = EXIT_FEATURE_SEQUENCE_BARS if first else 1
        step_count = 1 if first else int(carry.step_count) + 1
        expected_tf_names = tuple(
            timeframe.lower() for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES
        )
        if (
            int(entry_decision_representation.shape[1]) != d_model
            or tuple(exit_local_rows_x.shape)
            != (batch_size, expected_local_rows, self._expected_seq_dim)
            or tuple(exit_state_ctx_cat.shape)
            != (batch_size, self._expected_ctx_cat_dim)
            or tuple(exit_state_ctx_cont.shape)
            != (batch_size, self._expected_ctx_cont_dim)
            or tuple(exit_path_row_x.shape)
            != (batch_size, 2, UNIFIED_EXIT_PATH_FEATURE_DIM)
            or not 1 <= step_count <= UNIFIED_EXIT_EPISODE_STATE_COUNT
            or tuple(exit_mtf_new_rows) != expected_tf_names
            or (
                carry is not None
                and (
                    carry.batch_size != batch_size
                    or carry.step_count != step_count - 1
                )
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_INCREMENTAL_STEP_INPUT_INVALID")
        for name, value in (
            ("entry_decision_representation", entry_decision_representation),
            ("exit_local_rows_x", exit_local_rows_x),
            ("exit_state_ctx_cont", exit_state_ctx_cont),
            ("exit_path_row_x", exit_path_row_x),
        ):
            _assert_finite(name, value)

        signal_n = self._normalize_input_surface(
            exit_local_rows_x, surface="signal"
        )
        categorical_mask = self.input_norm_signal_categorical_mask.to(
            signal_n.device
        )
        signal_numeric = signal_n.masked_fill(
            categorical_mask.view(1, 1, -1), 0.0
        )
        global_input = self.seq_proj(signal_numeric)
        for index_text, embedding in self.signal_nominal_embeddings.items():
            index = int(index_text)
            global_input = global_input + embedding(
                exit_local_rows_x[..., index].long()
            )
        prior_global = None if first else carry.local_global_hidden
        global_sequence, global_hidden = self.exit_episode_global_gru(
            global_input, prior_global
        )
        global_state = global_sequence[:, -1:, :]

        family_temporal_rows: list[torch.Tensor] = []
        next_local_family_hidden: dict[str, torch.Tensor] = {}
        for family_name in self._specialist_names:
            indices = getattr(self, f"specialist_idx_{family_name}").to(
                signal_numeric.device
            )
            family_input = self.specialist_proj[family_name](
                signal_numeric.index_select(2, indices)
            )
            owned = self._specialist_input_index_sets[family_name]
            for index_text, embedding in self.signal_nominal_embeddings.items():
                index = int(index_text)
                if index in owned:
                    family_input = family_input + embedding(
                        exit_local_rows_x[..., index].long()
                    )
            prior = (
                None
                if first
                else carry.local_family_hidden[family_name]
            )
            family_sequence, family_hidden = self.exit_episode_family_gru[
                family_name
            ](family_input, prior)
            family_temporal_rows.append(family_sequence[:, -1:, :])
            next_local_family_hidden[family_name] = family_hidden
        family_temporal = torch.stack(family_temporal_rows, dim=2)

        ctx_cont_n = self._normalize_input_surface(
            exit_state_ctx_cont, surface="ctx_cont"
        )
        family_context, global_context = self._build_family_context_tokens(
            ctx_cont_n, exit_state_ctx_cat
        )
        family_context = family_context.view(
            batch_size, 1, len(self._specialist_names), d_model
        )
        global_context = global_context.view(batch_size, 1, d_model)
        family_tokens = self.exit_episode_family_cross_attn(
            (family_temporal + family_context).reshape(
                batch_size, len(self._specialist_names), d_model
            )
            + self.specialist_token_identity
        ).view(batch_size, 1, len(self._specialist_names), d_model)
        current_raw = exit_local_rows_x[:, -1:, :]
        current_n = signal_n[:, -1:, :]
        current_numeric = current_n.masked_fill(
            categorical_mask.view(1, 1, -1), 0.0
        )
        generic_idx = self.generic_snap_idx.to(current_numeric.device)
        current_snap = self.snap_proj(
            current_numeric.index_select(2, generic_idx)
        )
        generic_set = self._generic_snap_index_set
        for index_text, embedding in self.signal_nominal_embeddings.items():
            index = int(index_text)
            if index in generic_set:
                current_snap = current_snap + embedding(
                    current_raw[..., index].long()
                )
        local_state = self.exit_episode_local_fuse(
            torch.cat((global_state, current_snap, global_context), dim=2)
        )
        family_gate = torch.softmax(
            self.exit_episode_family_gate(local_state)
            + self.exit_episode_family_token_gate(family_tokens).squeeze(-1),
            dim=2,
        )
        local_state = local_state + self.specialist_fusion_scale.to(
            local_state.dtype
        ) * self.exit_episode_family_out(
            (family_tokens * family_gate.unsqueeze(-1)).sum(dim=2)
        )

        next_mtf_hidden: dict[str, dict[str, torch.Tensor]] = {}
        next_mtf_current_raw: dict[str, torch.Tensor] = {}
        tf_family_states: list[torch.Tensor] = []
        tf_feature_gate_rows: list[torch.Tensor] = []
        for tf_name in expected_tf_names:
            new_rows = exit_mtf_new_rows[tf_name]
            _assert_shape(f"exit_mtf_new_rows_{tf_name}", new_rows, 3)
            if (
                int(new_rows.shape[0]) != batch_size
                or int(new_rows.shape[2])
                != getattr(self, f"_expected_{tf_name}_seq_dim")
                or (first and int(new_rows.shape[1]) < 1)
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_INCREMENTAL_MTF_INPUT_INVALID:{tf_name}"
                )
            _assert_finite(f"exit_mtf_new_rows_{tf_name}", new_rows)
            if int(new_rows.shape[1]) > 0:
                current_tf_raw = new_rows[:, -1, :]
            elif first:
                raise RuntimeError(
                    f"UNIFIED_EXIT_INCREMENTAL_MTF_INITIAL_HISTORY_EMPTY:{tf_name}"
                )
            else:
                current_tf_raw = carry.mtf_current_raw[tf_name]
            next_mtf_current_raw[tf_name] = current_tf_raw
            surface = f"mtf_{tf_name}"
            current_tf_n = self._normalize_input_surface(
                current_tf_raw[:, None, :], surface=surface
            )[:, 0, :]
            mtf_cat_mask = getattr(
                self, f"input_norm_{surface}_categorical_mask"
            ).to(current_tf_n.device)
            current_tf_numeric = current_tf_n.masked_fill(mtf_cat_mask, 0.0)
            full_feature_gate = torch.zeros(
                batch_size,
                1,
                int(current_tf_raw.shape[1]),
                dtype=current_tf_n.dtype,
                device=current_tf_n.device,
            )
            family_states: list[torch.Tensor] = []
            next_mtf_hidden[tf_name] = {}
            if int(new_rows.shape[1]) > 0:
                normalized_new = self._normalize_input_surface(
                    new_rows, surface=surface
                )
                numeric_new = normalized_new.masked_fill(
                    mtf_cat_mask.view(1, 1, -1), 0.0
                )
            for family_name in self._specialist_names:
                indices = getattr(
                    self, f"multi_tf_specialist_idx_{family_name}"
                ).to(current_tf_n.device)
                prior = (
                    None
                    if first
                    else carry.mtf_family_hidden[tf_name][family_name]
                )
                if int(new_rows.shape[1]) > 0:
                    projected = self.mtf_family_proj[family_name](
                        numeric_new.index_select(2, indices)
                    )
                    for _local_position, global_index in (
                        self._multi_tf_specialist_categorical_positions[
                            family_name
                        ]
                    ):
                        projected = projected + self.mtf_nominal_embeddings[
                            f"{tf_name}_{global_index}"
                        ](new_rows[..., global_index].long())
                    encoded, hidden = self.exit_episode_mtf_family_gru[
                        family_name
                    ](
                        projected
                        * self._effective_tf_input_scale(tf_name).to(
                            projected.dtype
                        ),
                        prior,
                    )
                    gathered_state = encoded[:, -1:, :]
                else:
                    hidden = prior
                    gathered_state = hidden[-1].unsqueeze(1)
                if hidden is None:
                    raise RuntimeError("UNIFIED_EXIT_INCREMENTAL_MTF_HIDDEN_MISSING")
                next_mtf_hidden[tf_name][family_name] = hidden
                feature_gate = 2.0 * torch.sigmoid(
                    self.mtf_feature_context_gate[
                        f"{tf_name}__{family_name}"
                    ](local_state[:, 0, :])
                ).unsqueeze(1)
                full_feature_gate = full_feature_gate.scatter(
                    2,
                    indices.view(1, 1, -1).expand(batch_size, 1, -1),
                    feature_gate,
                )
                current_residual = self.mtf_family_proj[family_name](
                    current_tf_numeric.index_select(1, indices).unsqueeze(1)
                    * feature_gate
                    * self._effective_tf_input_scale(tf_name).to(
                        current_tf_n.dtype
                    )
                )
                for local_position, global_index in (
                    self._multi_tf_specialist_categorical_positions[family_name]
                ):
                    current_residual = current_residual + self.mtf_nominal_embeddings[
                        f"{tf_name}_{global_index}"
                    ](current_tf_raw[:, global_index].long()).unsqueeze(1) * (
                        feature_gate[..., local_position].unsqueeze(-1)
                    )
                family_states.append(gathered_state + current_residual)
            tf_family_states.append(torch.stack(family_states, dim=2))
            tf_feature_gate_rows.append(full_feature_gate)

        family_tf = torch.stack(tf_family_states, dim=2)
        identity = self.family_tf_token_identity[:, : len(expected_tf_names)]
        family_rows = self.exit_episode_family_axis_attn(
            (family_tf + identity.unsqueeze(1)).reshape(
                batch_size * len(expected_tf_names),
                len(self._specialist_names),
                d_model,
            )
        )
        family_axis = family_rows.reshape(
            batch_size,
            1,
            len(expected_tf_names),
            len(self._specialist_names),
            d_model,
        )
        timeframe_rows = self.exit_episode_timeframe_axis_attn(
            family_axis.permute(0, 1, 3, 2, 4).reshape(
                batch_size * len(self._specialist_names),
                len(expected_tf_names),
                d_model,
            )
        )
        cooperation_tokens = timeframe_rows.reshape(
            batch_size,
            1,
            len(self._specialist_names),
            len(expected_tf_names),
            d_model,
        ).permute(0, 1, 3, 2, 4)
        cooperation_logits = self.exit_episode_family_tf_context_gate(
            local_state
        ).reshape(
            batch_size,
            1,
            len(expected_tf_names),
            len(self._specialist_names),
        ) + self.exit_episode_family_tf_token_gate(
            cooperation_tokens
        ).squeeze(-1)
        cooperation_gate = torch.softmax(
            cooperation_logits.reshape(batch_size, 1, -1), dim=2
        ).reshape_as(cooperation_logits)
        mtf_correction = self.exit_episode_family_tf_out(
            (cooperation_tokens * cooperation_gate.unsqueeze(-1)).sum(
                dim=(2, 3)
            )
        )

        path_flat = exit_path_row_x.reshape(
            batch_size * 2, 1, UNIFIED_EXIT_PATH_FEATURE_DIM
        )
        prior_path = None if first else carry.path_hidden
        path_encoded, path_hidden = self.exit_episode_path_gru(
            self.exit_path_proj(path_flat), prior_path
        )
        path_encoded = path_encoded.reshape(batch_size, 2, 1, d_model)
        side_index = torch.arange(2, device=exit_path_row_x.device).view(1, 2)
        side_state = self.exit_side_embedding(side_index).view(
            1, 2, 1, d_model
        ).expand(batch_size, -1, -1, -1)
        token = entry_decision_representation.view(
            batch_size, 1, 1, d_model
        ).expand(-1, 2, -1, -1)
        local_side = local_state.unsqueeze(1).expand(-1, 2, -1, -1)
        mtf_side = mtf_correction.unsqueeze(1).expand(-1, 2, -1, -1)
        hidden = self.exit_episode_fuse(
            torch.cat(
                (token, local_side, mtf_side, side_state, path_encoded), dim=3
            )
        )
        q_values = self.head_exit_action(hidden)
        valid = torch.ones_like(q_values, dtype=torch.bool)
        terminal_mask = torch.zeros(
            batch_size, 2, 1, dtype=torch.bool, device=q_values.device
        )
        terminal_reason_index = torch.zeros(
            batch_size, 2, 1, dtype=torch.long, device=q_values.device
        )
        if step_count == UNIFIED_EXIT_EPISODE_STATE_COUNT:
            valid[..., 0] = False
            terminal_mask[..., 0] = True
            terminal_reason_index[..., 0] = 1
        output = {
            "exit_action_q_bps": q_values,
            "exit_action_valid_mask": valid,
            "exit_episode_lengths": torch.full(
                (batch_size, 2),
                step_count,
                dtype=torch.long,
                device=q_values.device,
            ),
            "exit_terminal_mask": terminal_mask,
            "exit_terminal_reason_index": terminal_reason_index,
            "exit_specialist_gate": family_gate,
            "exit_tf_gate": cooperation_gate.sum(dim=3),
            "exit_family_tf_cooperation_gate": cooperation_gate,
            "exit_family_tf_feature_gate": torch.stack(
                tf_feature_gate_rows, dim=2
            ),
        }
        next_carry = UnifiedExitIncrementalCarry(
            step_count=step_count,
            batch_size=batch_size,
            local_global_hidden=global_hidden,
            local_family_hidden=next_local_family_hidden,
            mtf_family_hidden=next_mtf_hidden,
            mtf_current_raw=next_mtf_current_raw,
            path_hidden=path_hidden,
        )
        return output, next_carry

    def forward(
        self,
        seq_x: torch.Tensor,
        snap_x: torch.Tensor,
        *,
        ctx_cat: torch.Tensor,
        ctx_cont: torch.Tensor,
        seq_m15: torch.Tensor,
        seq_h1: torch.Tensor,
        seq_h4: torch.Tensor,
        seq_d1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_v3, specialist_gate, global_context_h = self._encode_shared_feature_base(
            seq_x=seq_x,
            snap_x=snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
            surface_label="entry_m5",
            expected_seq_len=self._expected_seq_len,
            positional_encoding="pos_enc",
        )
        entry_mtf = self._encode_multi_tf_route(
            base_representation=z_v3,
            tf_inputs=(
                ("m15", seq_m15, self.cfg.m15_seq_len, self._expected_m15_seq_dim),
                ("h1", seq_h1, self.cfg.h1_seq_len, self._expected_h1_seq_dim),
                ("h4", seq_h4, self.cfg.h4_seq_len, self._expected_h4_seq_dim),
                ("d1", seq_d1, self.cfg.d1_seq_len, self._expected_d1_seq_dim),
            ),
            route_timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
            route_label="entry",
        )
        z = entry_mtf["representation"]
        mtf_repr = entry_mtf["mtf_repr"]
        tf_gate = entry_mtf["tf_gate"]
        family_tf_cooperation_gate = entry_mtf["family_tf_cooperation_gate"]
        family_tf_feature_gate = entry_mtf["family_tf_feature_gate"]
        entry_q_joint_source = torch.cat(
            (z_v3, z, mtf_repr, global_context_h), dim=1
        )
        entry_q_joint_hidden = nn.functional.gelu(
            self.entry_q_joint_in(
                self.entry_q_joint_norm(entry_q_joint_source)
            )
        )
        entry_action_q_bps = self.head_entry_action_q(entry_q_joint_hidden)
        _assert_finite("entry_q_joint_hidden", entry_q_joint_hidden)
        _assert_finite("entry_action_q_bps", entry_action_q_bps)

        out = {
            "entry_action_q_bps": entry_action_q_bps,
            "entry_q_joint_hidden": entry_q_joint_hidden,
        }
        out["specialist_gate"] = specialist_gate
        out["tf_gate"] = tf_gate
        out["family_tf_cooperation_gate"] = family_tf_cooperation_gate
        out["family_tf_feature_gate"] = family_tf_feature_gate
        # Genuine numeric outcomes and forward events remain representation
        # auxiliaries. Threshold-derived trade/side/clean-edge/survival heads
        # are intentionally absent, so they cannot shape the Entry Q backbone.
        exact_outputs = {
            "side_mae_bps": self.head_side_mae(z),
            "trendline_event_logits": self.head_trendline_event(z),
            "position_size_logit": self.head_position_size(z),
            "dip_pred": self.head_dip(z),
            "forecast_pred": self.head_forecast(z),
            "timing_pred": torch.sigmoid(self.head_timing(z)),
            "tail_risk_pred": self.head_tail_risk(z),
            "vol_forecast_pred": self.head_vol_forecast(z),
        }
        for output_name, value in exact_outputs.items():
            _assert_finite(output_name, value)

        out.update(exact_outputs)
        entry_token_components = {
            "local_model_native_representation": z_v3,
            "final_model_native_representation": z,
            "multi_timeframe_representation": mtf_repr,
            "family_context_representation": global_context_h,
            "entry_q_joint_hidden": entry_q_joint_hidden,
            "entry_action_q_bps": entry_action_q_bps,
        }
        entry_decision_representation = self._project_entry_decision_token(
            entry_token_components
        )
        entry_decision_token_source = (
            self._assemble_entry_decision_token_source(
                entry_token_components
            )
        )
        _assert_finite(
            UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
            entry_decision_representation,
        )
        out[UNIFIED_EXIT_MODEL_REPRESENTATION_KEY] = (
            entry_decision_representation
        )
        out["entry_decision_token_source"] = entry_decision_token_source
        return out


def _build_unit_test_entry_v10_ctx_hybrid_transformer(
    **kwargs: object,
) -> EntryV10CtxHybridTransformer:
    """Construct a private small-shape model that production inputs cannot select."""

    model = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    object.__setattr__(
        model,
        "_gx1_unit_test_architecture_token",
        _UNIT_TEST_ARCHITECTURE_SENTINEL,
    )
    EntryV10CtxHybridTransformer.__init__(model, **kwargs)
    return model
