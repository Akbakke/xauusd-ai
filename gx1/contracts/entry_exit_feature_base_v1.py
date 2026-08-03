"""One shared Entry/Exit feature-base and resolution contract.

Entry and Exit are two decisions in one model bundle.  They do not own
separate feature implementations or specialist taxonomies.  The canonical
feature base emits the same ordered semantic surface at two causal clocks:
M5 for Entry and M1 for Exit.  Exit may add its literal closed-M1 execution
path, but that path is not a replacement for the shared specialist surface.
"""
from __future__ import annotations

from typing import Any, Mapping

from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SEQ_LEN,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_CLOSURE_CONTRACT,
)


ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION = (
    "gx1_entry_exit_shared_feature_base_contract_v1"
)
ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION = (
    "gx1_entry_exit_enriched_causal_frame_v1"
)
ENTRY_DECISION_TIMEFRAME = "M5"
EXIT_DECISION_TIMEFRAME = "M1"
ENTRY_DECISION_BAR_SECONDS = 300
EXIT_DECISION_BAR_SECONDS = 60
ENTRY_EXIT_RESOLUTION_RATIO = 5
ENTRY_EXIT_SHARED_ENCODER = "entry_v10_ctx_hybrid_transformer_shared_specialists_v1"
EXIT_FEATURE_MAX_SEQUENCE_BARS = 512
ENTRY_FEATURE_SEQUENCE_BARS = MODEL_NATIVE_SEQ_LEN
EXIT_FEATURE_SEQUENCE_BARS = (
    ENTRY_FEATURE_SEQUENCE_BARS * ENTRY_EXIT_RESOLUTION_RATIO
)
EXIT_FEATURE_ROW_CLOCK = "consecutive_authoritative_closed_m1_source_rows"


def entry_exit_shared_feature_base_contract() -> dict[str, Any]:
    """Return the exact contract every Entry/Exit producer must bind."""

    return {
        "schema_version": ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
        "instrument": "XAU_USD",
        "single_feature_owner": True,
        "single_specialist_taxonomy": True,
        "specialist_families": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "specialist_family_count": len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "ordered_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "mandatory_signal_dim": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "context_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "context_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "entry": {
            "decision_timeframe": ENTRY_DECISION_TIMEFRAME,
            "decision_bar_seconds": ENTRY_DECISION_BAR_SECONDS,
            "surface_role": "shared_specialist_feature_surface",
        },
        "exit": {
            "decision_timeframe": EXIT_DECISION_TIMEFRAME,
            "decision_bar_seconds": EXIT_DECISION_BAR_SECONDS,
            "surface_role": "shared_specialist_feature_surface_plus_closed_m1_path",
            "requires_entry_m5_representation": True,
            "requires_shared_m1_feature_surface": True,
            "path_is_additive_not_replacement": True,
            "m1_sequence_resolution_ratio": ENTRY_EXIT_RESOLUTION_RATIO,
            "row_clock": EXIT_FEATURE_ROW_CLOCK,
            "source_absence_contract": CANONICAL_NATIVE_CLOSURE_CONTRACT,
            "synthetic_gap_fill_allowed": False,
        },
        "resolution_ratio_m1_per_m5": ENTRY_EXIT_RESOLUTION_RATIO,
        "shared_encoder": ENTRY_EXIT_SHARED_ENCODER,
        "same_dataset_run_id_required": True,
        "same_split_boundaries_required": True,
        "same_train_normalization_state_required": True,
        "future_feature_reuse_forbidden": True,
        "separate_feature_implementations_forbidden": True,
        "exit_feature_max_sequence_bars": EXIT_FEATURE_MAX_SEQUENCE_BARS,
        "entry_feature_sequence_bars": ENTRY_FEATURE_SEQUENCE_BARS,
        "exit_feature_sequence_bars": EXIT_FEATURE_SEQUENCE_BARS,
    }


def require_entry_exit_shared_feature_base_contract(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Fail closed unless ``value`` is exactly the shared-base contract."""

    expected = entry_exit_shared_feature_base_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(
            f"{context}_ENTRY_EXIT_SHARED_FEATURE_BASE_CONTRACT_MISMATCH"
        )
    return dict(value)
