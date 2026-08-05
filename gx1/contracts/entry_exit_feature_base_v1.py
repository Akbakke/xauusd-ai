"""One shared Entry/Exit feature-owner and resolution contract.

Entry and Exit are two decisions in one model bundle.  They do not own
separate feature implementations or specialist taxonomies.  The canonical
feature owners independently compute the same ordered semantic surface from
closed native M5 rows for Entry and closed native M1 rows for Exit.  There is
no combined M1/M5 package in front of the owners and no value copying between
resolutions.  Exit may add its literal closed-M1 execution path, but that path
is not a replacement for the shared specialist surface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    group_features_by_specialist,
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
from gx1.utils.artifact_primitives_v1 import (
    require_immutable_artifact,
    sha256_file,
)


ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION = (
    "gx1_entry_exit_shared_feature_base_contract_v4"
)
ENTRY_EXIT_FEATURE_OWNER_RESOLUTION_SCHEMA_VERSION = (
    "gx1_entry_exit_feature_owner_resolution_contract_v3"
)
ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION = (
    "gx1_entry_exit_enriched_causal_frame_v3"
)
ENTRY_DECISION_TIMEFRAME = "M5"
EXIT_DECISION_TIMEFRAME = "M1"
ENTRY_DECISION_BAR_SECONDS = 300
EXIT_DECISION_BAR_SECONDS = 60
ENTRY_EXIT_RESOLUTION_RATIO = 5
ENTRY_MTF_CONTEXT_TIMEFRAMES = ("M15", "H1", "H4", "D1")
EXIT_MTF_CONTEXT_TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
ENTRY_MTF_CONTEXT_COUNT = len(ENTRY_MTF_CONTEXT_TIMEFRAMES)
EXIT_MTF_CONTEXT_COUNT = len(EXIT_MTF_CONTEXT_TIMEFRAMES)
ENTRY_EXIT_SHARED_ENCODER = "entry_v10_ctx_hybrid_transformer_shared_specialists_v2"
EXIT_FEATURE_MAX_SEQUENCE_BARS = 512
ENTRY_FEATURE_SEQUENCE_BARS = MODEL_NATIVE_SEQ_LEN
EXIT_FEATURE_SEQUENCE_BARS = (
    ENTRY_FEATURE_SEQUENCE_BARS * ENTRY_EXIT_RESOLUTION_RATIO
)
EXIT_FEATURE_ROW_CLOCK = "consecutive_authoritative_closed_m1_source_rows"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def entry_exit_feature_owner_resolution_contract() -> dict[str, Any]:
    """Bind every specialist owner to independent native M5 and M1 runs."""

    grouped = group_features_by_specialist(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
    )
    invalid_groups = {
        name: fields
        for name, fields in grouped.items()
        if name not in MODEL_NATIVE_TRAINING_SPECIALISTS and fields
    }
    owner_counts = {
        owner: len(grouped[owner])
        for owner in MODEL_NATIVE_TRAINING_SPECIALISTS
    }
    if (
        invalid_groups
        or any(count <= 0 for count in owner_counts.values())
        or sum(owner_counts.values())
        != MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
    ):
        raise RuntimeError("ENTRY_EXIT_FEATURE_OWNER_COVERAGE_INVALID")

    return {
        "schema_version": ENTRY_EXIT_FEATURE_OWNER_RESOLUTION_SCHEMA_VERSION,
        "pre_owner_combined_m1_m5_source_package_allowed": False,
        "independent_native_resolution_computation_required": True,
        "cross_resolution_value_copy_allowed": False,
        "same_owner_implementation_required": True,
        "same_ordered_field_semantics_required": True,
        "resolution_specific_values_expected": True,
        "mtf_construction": "closed_ohlcv_before_feature_computation",
        "mtf_feature_owner": "native_m5_v4",
        "mtf_feature_owner_count": 1,
        "mtf_source": "exact_native_m5_closed_ohlcv",
        "legacy_local_owner_mtf_computation_allowed": False,
        "computed_feature_resampling_allowed": False,
        "entry_route": {
            "local_timeframe": ENTRY_DECISION_TIMEFRAME,
            "mtf_context_timeframes": list(ENTRY_MTF_CONTEXT_TIMEFRAMES),
        },
        "exit_route": {
            "local_timeframe": EXIT_DECISION_TIMEFRAME,
            "mtf_context_timeframes": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
        },
        "mandatory_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "mandatory_feature_count_by_owner": owner_counts,
        "owners": {
            owner: {
                "M5": {
                    "consumer": "ENTRY",
                    "source": "closed_native_m5_rows",
                    "bar_seconds": ENTRY_DECISION_BAR_SECONDS,
                },
                "M1": {
                    "consumer": "EXIT",
                    "source": "closed_native_m1_rows",
                    "bar_seconds": EXIT_DECISION_BAR_SECONDS,
                },
            }
            for owner in MODEL_NATIVE_TRAINING_SPECIALISTS
        },
    }


def entry_exit_shared_feature_base_contract() -> dict[str, Any]:
    """Return the exact contract every Entry/Exit producer must bind."""

    return {
        "schema_version": ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
        "instrument": "XAU_USD",
        "single_feature_owner": True,
        "single_mtf_feature_owner": "native_m5_v4",
        "legacy_mtf_feature_owners_forbidden": True,
        "single_specialist_taxonomy": True,
        "specialist_families": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "specialist_family_count": len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "feature_owner_resolution_contract": (
            entry_exit_feature_owner_resolution_contract()
        ),
        "ordered_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "mandatory_signal_dim": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "context_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "context_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "entry": {
            "decision_timeframe": ENTRY_DECISION_TIMEFRAME,
            "decision_bar_seconds": ENTRY_DECISION_BAR_SECONDS,
            "surface_role": "shared_specialist_feature_surface",
            "mtf_context_timeframes": list(ENTRY_MTF_CONTEXT_TIMEFRAMES),
            "local_timeframe_may_repeat_in_mtf": False,
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
            "mtf_context_timeframes": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
            "local_timeframe_may_repeat_in_mtf": False,
        },
        "resolution_ratio_m1_per_m5": ENTRY_EXIT_RESOLUTION_RATIO,
        "shared_encoder": ENTRY_EXIT_SHARED_ENCODER,
        "same_dataset_run_id_required": True,
        "same_split_boundaries_required": True,
        "same_train_normalization_state_required": True,
        "future_feature_reuse_forbidden": True,
        "combined_pre_owner_source_package_forbidden": True,
        "cross_resolution_value_copy_forbidden": True,
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


def require_entry_exit_enriched_source_binding(
    source: Path,
    *,
    dataset_run_id: str,
    pair_generation_id: str,
    timeframe: str,
    context: str,
) -> dict[str, str]:
    """Validate the exact enriched-source sidecar for either surface."""

    source = Path(source)
    sidecar = Path(f"{source}.manifest.json")
    if (
        timeframe not in {ENTRY_DECISION_TIMEFRAME, EXIT_DECISION_TIMEFRAME}
        or not source.is_absolute()
        or source.resolve() != source
        or source.is_symlink()
        or not source.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
    ):
        raise RuntimeError(f"{context}_SOURCE_ARTIFACT_INVALID")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context}_SOURCE_MANIFEST_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context}_SOURCE_MANIFEST_INVALID")
    unhashed = dict(payload)
    declared_hash = unhashed.pop("manifest_sha256", None)
    source_hash = sha256_file(source)
    bar_seconds = (
        EXIT_DECISION_BAR_SECONDS
        if timeframe == EXIT_DECISION_TIMEFRAME
        else ENTRY_DECISION_BAR_SECONDS
    )
    if (
        payload.get("schema_version")
        != ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION
        or payload.get("decision") != "PASS"
        or payload.get("dataset_run_id") != dataset_run_id
        or payload.get("pair_generation_id") != pair_generation_id
        or payload.get("timeframe") != timeframe
        or payload.get("base_bar_seconds") != bar_seconds
        or payload.get("output_parquet") != str(source)
        or payload.get("output_parquet_sha256") != source_hash
        or declared_hash != _canonical_sha256(unhashed)
    ):
        raise RuntimeError(f"{context}_SOURCE_LINEAGE_INVALID")
    require_entry_exit_shared_feature_base_contract(
        payload.get("shared_feature_base_contract"),
        context=context,
    )
    rank_raw = payload.get("rank_reference_npz")
    rank_hash = payload.get("rank_reference_sha256")
    if not isinstance(rank_raw, str) or not isinstance(rank_hash, str):
        raise RuntimeError(f"{context}_SOURCE_RANK_LINEAGE_INVALID")
    rank_path = require_immutable_artifact(
        Path(rank_raw),
        expected_sha256=rank_hash,
        context=f"{context}_SOURCE_RANK",
    )
    return {
        "manifest_path": str(sidecar),
        "manifest_sha256": sha256_file(sidecar),
        "schema_version": ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
        "source_sha256": source_hash,
        "rank_reference_npz": str(rank_path),
        "rank_reference_sha256": rank_hash,
    }


def require_entry_exit_feature_surface_identity(
    value: Mapping[str, Any] | Any,
    *,
    expected_timeframe: str,
    expected_ordered_fields: Sequence[str],
    expected_signal_manifest_path: str,
    expected_signal_manifest_sha256: str,
    expected_rank_reference_sha256: str,
    context: str,
) -> dict[str, Any]:
    """Prove one native-resolution surface matches the shared Entry contract."""

    if expected_timeframe not in {ENTRY_DECISION_TIMEFRAME, EXIT_DECISION_TIMEFRAME}:
        raise RuntimeError(f"{context}_ENTRY_EXIT_TIMEFRAME_INVALID")
    fields = [str(field) for field in expected_ordered_fields]
    if not fields or len(fields) != len(set(fields)):
        raise RuntimeError(f"{context}_ENTRY_EXIT_ORDERED_FIELDS_INVALID")
    expected = {
        "anchor_timeframe": expected_timeframe,
        "feature_field_order": fields,
        "feature_field_order_sha256": _canonical_sha256(fields),
        "seq_structure_manifest": expected_signal_manifest_path,
        "seq_structure_manifest_sha256": expected_signal_manifest_sha256,
        "rank_reference_sha256": expected_rank_reference_sha256,
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
    }
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"{context}_ENTRY_EXIT_RESOLUTION_SURFACE_IDENTITY_MISMATCH"
        )
    mismatches = [
        key for key, expected_value in expected.items()
        if value.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            f"{context}_ENTRY_EXIT_RESOLUTION_SURFACE_IDENTITY_MISMATCH: "
            f"fields={mismatches}"
        )
    return dict(value)
