import inspect

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
import copy

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    training_objective_contract_metadata,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
    canonical_json_sha256,
)
from gx1.contracts.entry_model_native_readiness_v1 import MODEL_NATIVE_ACTIVE_HEADS
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    NEUTRAL_EFFECTIVE_INIT,
    build_tf_input_scale_contract,
    raw_tf_input_scale_from_effective,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_FEATURE_NAMES_SHA256_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    require_multi_tf_resolution_pyramid,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION,
    require_multi_tf_specialist_routing_v4,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from tests.model_native_input_normalization_support import (
    decision_window_coverage_fixture,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    direction_evidence_fusion_metadata,
)
from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
    COMPONENT_PARAMETERS,
    PARAMETER_SHAPES,
    REFERENCE as MOVEMENT_REFERENCE,
    SCHEMA_VERSION as MOVEMENT_SCHEMA_VERSION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
    unified_entry_exit_contract_metadata,
)
from gx1.models.entry_v10.entry_v10_bundle import (
    _ENTRY_HEAD_STATE_KEYS,
    _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS,
    _MODEL_NATIVE_REQUIRED_SPECIALISTS,
    _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS,
    _infer_entry_bundle_capabilities,
    _require_exact_model_native_bundle_metadata,
    _require_model_native_learned_component_liveness,
    load_entry_v10_ctx_bundle,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _build_active_head_names,
)
from tests.model_native_context_routing_support import (
    context_routing_for_ordered_signal_names,
)
from tests.model_native_input_normalization_support import (
    input_normalization_fit_population_proof_fixture,
    input_normalization_fixture,
)


def test_v10_metadata_active_heads_include_enabled_model_native_heads() -> None:
    got = _build_active_head_names()

    assert got == [*MODEL_NATIVE_ACTIVE_HEADS, "unified_exit"]


def test_bundle_contract_requires_the_same_model_unified_exit_head() -> None:
    assert "unified_exit" in _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS
    assert _ENTRY_HEAD_STATE_KEYS["unified_exit"] == {
        "head_exit_action.weight",
        "head_exit_action.bias",
    }


def test_v10_bundle_capabilities_accept_declared_model_native_heads_from_state_dict() -> (
    None
):
    state_dict = {}
    for prefix in [
        "head_direction",
        "head_path_quality",
        "head_mfe_first_n",
        "head_tradable",
        "head_bad_path",
        "head_clean_edge",
        "head_survival",
        "head_dip",
        "head_forecast",
        "head_timing",
        "head_tail_risk",
        "head_vol_forecast",
        "head_mtf_direction",
        "head_trendline_rail",
    ]:
        state_dict[f"{prefix}.weight"] = object()
        state_dict[f"{prefix}.bias"] = object()
    meta = {
        "supports_context_features": True,
        "train_recipe": {
            "active_heads": [
                "direction",
                "dip",
                "forecast",
                "timing",
                "tail_risk",
                "vol_forecast",
                "mtf_direction",
                "trendline_rail",
            ]
        },
    }

    got = _infer_entry_bundle_capabilities(meta, state_dict)

    assert "dip" in got["supported_heads"]
    assert "forecast" in got["supported_heads"]
    assert "timing" in got["supported_heads"]
    assert "tail_risk" in got["supported_heads"]
    assert "vol_forecast" in got["supported_heads"]
    assert "mtf_direction" in got["supported_heads"]
    assert "trendline_rail" in got["supported_heads"]
    assert got["supports_context_features"] is True


def _exact_model_native_metadata() -> tuple[dict, dict]:
    direction_contract = model_direction_decision_contract_metadata()
    unified_contract = unified_entry_exit_contract_metadata()
    native_subset_years = {
        "year=2026": {
            "rows": 20,
            "canonical_rows_sha256": "3" * 64,
        }
    }
    m1_authority = {
        "schema_version": UNIFIED_EXIT_M1_AUTHORITY_SCHEMA_VERSION,
        "m1_source_path": "/immutable/pair/base28.parquet",
        "m1_source_sha256": "2" * 64,
        "base28_native_m1_subset_proof": {
            "method": "exact_base28_rows_are_native_m1_subset_v1",
            "rows": 20,
            "years": native_subset_years,
            "proof_sha256": canonical_json_sha256(
                native_subset_years
            ),
        },
    }
    unified_exit_training_evidence = {
        "schema_version": "gx1_unified_exit_training_evidence_v1",
        "decision": "PASS",
        "shared_model_state_dict": True,
        "entry_representation_surface": "shared_feature_representation",
        "future_outcomes_used_as_model_inputs": False,
        "exit_action_loss_weight": 1.0,
        "lifecycle": {
            "schema_version": (
                UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
            ),
            "future_outcomes_used_as_model_inputs": False,
            "m1_source_path": m1_authority["m1_source_path"],
            "m1_source_sha256": m1_authority["m1_source_sha256"],
            "m1_authority": m1_authority,
            "m1_authority_sha256": canonical_json_sha256(
                m1_authority
            ),
        },
        "selected_checkpoint_validation": {
            "unified_exit_action_loss_mean": 0.5,
            "unified_exit_action_rows": 20,
            "unified_exit_hold_rows": 10,
            "unified_exit_now_rows": 10,
            "unified_exit_action_accuracy": 0.6,
        },
        "selected_checkpoint_parameter_movement": {
            "all_exit_components_moved": True,
        },
    }
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.metadata_contract_fixture"
        )
    )
    training_objective = training_objective_contract_metadata(
        {name: 1.0 for name in REQUIRED_POSITIVE_LOSS_WEIGHTS}
    )
    context_routing = context_routing_for_ordered_signal_names(
        list(signal_contract["fields"])
    )
    movement = {
        "schema_version": MOVEMENT_SCHEMA_VERSION,
        "reference": MOVEMENT_REFERENCE,
        "selected_checkpoint_epoch": 1,
        "parameter_deltas": {
            key: {
                "shape": list(shape),
                "max_abs_delta": 0.1,
                "l2_delta": 0.2,
                "changed": True,
            }
            for key, shape in PARAMETER_SHAPES.items()
        },
        "component_changed": {key: True for key in COMPONENT_PARAMETERS},
        "output_rows_distinct": True,
        "decision": "PASS",
    }
    mtf_contract = {
        "enabled": True,
        "v4_mode": True,
        "m5_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "m5_seq_len": 96,
        "m15_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "m15_seq_len": 64,
        "h1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "h1_seq_len": 96,
        "h4_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "h4_seq_len": 48,
        "d1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "d1_seq_len": 30,
        "multi_tf_num_layers": 1,
        "multi_tf_scale": 0.5,
        "feature_contract": "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V2",
        "matrix_contract": HTF_V4_MATRIX_CONTRACT,
        "feature_names": list(MULTI_TF_PER_BAR_FEATURES_V4),
        "feature_names_sha256": MULTI_TF_FEATURE_NAMES_SHA256_V4,
        "closed_bar_target_availability": True,
        "target_availability_shift_minutes": 5.0,
    }
    per_tf_seq_lens = {
        "M5": mtf_contract["m5_seq_len"],
        "M15": mtf_contract["m15_seq_len"],
        "H1": mtf_contract["h1_seq_len"],
        "H4": mtf_contract["h4_seq_len"],
        "D1": mtf_contract["d1_seq_len"],
    }
    mtf_specialist_indices = {
        name: list(indices)
        for name, indices in require_multi_tf_specialist_routing_v4(
            MULTI_TF_PER_BAR_FEATURES_V4
        ).items()
    }
    mtf_contract.update(
        {
            "resolution_pyramid": require_multi_tf_resolution_pyramid(
                per_tf_seq_lens
            ),
            "decision_window_coverage": decision_window_coverage_fixture(
                per_tf_seq_lens
            ),
            "specialist_routing_schema_version": (
                MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION
            ),
            "specialist_input_indices": mtf_specialist_indices,
            "family_tf_token_order": [
                f"{tf}:{specialist}"
                for tf in ("m5", "m15", "h1", "h4", "d1")
                for specialist in mtf_specialist_indices
            ],
        }
    )
    tf_inits = {
        name: NEUTRAL_EFFECTIVE_INIT
        for name in ("m5", "m15", "h1", "h4", "d1")
    }
    tf_scale_contract = build_tf_input_scale_contract(
        init_effective=tf_inits,
        learned_raw={
            name: raw_tf_input_scale_from_effective(value)
            for name, value in tf_inits.items()
        },
    )
    input_normalization = input_normalization_fixture(
        signal_names=list(signal_contract["fields"]),
        mtf_names=list(MULTI_TF_PER_BAR_FEATURES_V4),
        per_tf_seq_lens=per_tf_seq_lens,
        dataset_run_id="MODEL_NATIVE_DATASET_PYTEST_V1",
    )
    shared = {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "dropout": 0.05,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "ordered_signal_names": list(signal_contract["fields"]),
        "ordered_ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ordered_ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        "model_native_signal_contract": signal_contract,
        "model_native_training_objective": training_objective,
        "model_native_direction_evidence_fusion": (
            direction_evidence_fusion_metadata()
        ),
        "model_native_learned_component_movement": movement,
        "context_specialist_routing": copy.deepcopy(context_routing),
        "input_normalization": input_normalization,
        "input_normalization_fit_population_proof": (
            input_normalization_fit_population_proof_fixture(
                input_normalization
            )
        ),
        "multi_tf": mtf_contract,
        "tf_input_scale": tf_scale_contract,
        "aux_head_target_contract": model_native_aux_target_contract_metadata(),
        "run_lineage": {
            "schema_version": "entry_model_native_training_run_lineage_v1",
            "training_run_id": "MODEL_NATIVE_TRAINING_PYTEST_V1",
            "dataset_run_id": "MODEL_NATIVE_DATASET_PYTEST_V1",
        },
        "unified_entry_exit_contract": unified_contract,
        "unified_exit_training_evidence": unified_exit_training_evidence,
    }
    lock = {
        **copy.deepcopy(shared),
        "direction_decision_contract": copy.deepcopy(direction_contract),
    }
    specialist_indices = {name: [] for name in _MODEL_NATIVE_REQUIRED_SPECIALISTS}
    for index in range(MODEL_NATIVE_SIGNAL_DIM):
        specialist_indices[
            _MODEL_NATIVE_REQUIRED_SPECIALISTS[
                index % len(_MODEL_NATIVE_REQUIRED_SPECIALISTS)
            ]
        ].append(index)
    meta = {
        **copy.deepcopy(shared),
        "supports_context_features": True,
        "anchored_entry_enabled": False,
        "anchor_source": None,
        "anchor_gate": {"enabled": False},
        "direction_decision_contract": copy.deepcopy(direction_contract),
        "train_recipe": {
            "active_heads": sorted(_MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS)
        },
        "enable_pos_enc": True,
        "enable_regime_film": True,
        "hierarchical_entry_heads": {"enabled": True},
        "trendline_rail_head": {
            "enabled": True,
            "output_dim": 6,
            "hand_written_direction_pressure": False,
            "direction_mapping": "direct_learned_evidence_fusion",
        },
        "model_native_state_contract": {
            "fixture": True,
            "entry_run_id": "MODEL_NATIVE_DATASET_PYTEST_V1",
        },
        "specialist_fusion": {
            "enabled": True,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "input_indices": specialist_indices,
            "context_routing": copy.deepcopy(context_routing),
            "trainable_specialists": list(_MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "num_layers": 1,
            "fusion_scale": 0.25,
            "cross_family_fusion_scale": 0.25,
        },
    }
    return meta, lock


def test_model_native_bundle_metadata_contract_is_exact_and_complete() -> None:
    meta, lock = _exact_model_native_metadata()

    _require_exact_model_native_bundle_metadata(meta, lock)


@pytest.mark.parametrize(
    "field",
    ("feature_meta_path", "seq_scaler_path", "snap_scaler_path"),
)
@pytest.mark.parametrize("owner", ("meta", "lock"))
def test_model_native_bundle_rejects_stale_compatibility_artifact_declarations(
    field: str,
    owner: str,
) -> None:
    meta, lock = _exact_model_native_metadata()
    payload = meta if owner == "meta" else lock
    payload[field] = None

    with pytest.raises(
        RuntimeError,
        match="STALE_COMPATIBILITY_ARTIFACT_FORBIDDEN",
    ):
        _require_exact_model_native_bundle_metadata(meta, lock)


def test_model_native_bundle_loader_has_no_stale_compatibility_arguments() -> None:
    parameters = inspect.signature(load_entry_v10_ctx_bundle).parameters

    assert set(parameters) == {"bundle_dir", "device"}


@pytest.mark.parametrize(
    "field",
    ("feature_meta_path", "seq_scaler_path", "snap_scaler_path"),
)
def test_model_native_bundle_loader_rejects_stale_compatibility_arguments(
    field: str,
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        load_entry_v10_ctx_bundle(
            bundle_dir="/bundle/path/is/not/inspected",
            **{field: "/stale/compatibility/artifact"},
        )


def test_model_native_bundle_rejects_collapsed_training_and_dataset_lineage() -> None:
    meta, lock = _exact_model_native_metadata()
    for payload in (meta, lock):
        payload["run_lineage"]["training_run_id"] = payload["run_lineage"][
            "dataset_run_id"
        ]

    with pytest.raises(RuntimeError, match="RUN_LINEAGE_ROLES_COLLAPSED"):
        _require_exact_model_native_bundle_metadata(meta, lock)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda meta, lock: (
                meta["multi_tf"].pop("h4_seq_len"),
                lock["multi_tf"].pop("h4_seq_len"),
            ),
            "MTF_METADATA_MISSING",
        ),
        (
            lambda meta, lock: meta["train_recipe"]["active_heads"].remove(
                "trendline_rail"
            ),
            "ACTIVE_COMPONENTS_MISMATCH",
        ),
        (
            lambda meta, lock: meta.update(
                {"hierarchical_direction_composition": {"enabled": True}}
            ),
            "STALE_DIRECTION_ARTIFACT_FORBIDDEN",
        ),
        (
            lambda meta, lock: meta.update({"enable_pos_enc": False}),
            "FULL_STACK_COMPONENT_REQUIRED",
        ),
        (
            lambda meta, lock: meta.update({"enable_regime_film": False}),
            "FULL_STACK_COMPONENT_REQUIRED",
        ),
        (lambda meta, lock: lock.update({"ctx_cont_dim": 3}), "META_LOCK_SPLIT_BRAIN"),
        (
            lambda meta, lock: meta["model_native_training_objective"][
                "configurable_positive_loss_weights"
            ].update({"ENTRY_AUX_BAD_PATH_WEIGHT": 0.0}),
            "META_LOCK_SPLIT_BRAIN",
        ),
    ],
)
def test_model_native_bundle_metadata_contract_fails_closed(
    mutation,
    error: str,
) -> None:
    meta, lock = _exact_model_native_metadata()
    mutation(meta, lock)

    with pytest.raises(RuntimeError, match=error):
        _require_exact_model_native_bundle_metadata(meta, lock)


def _live_model_native_state() -> dict[str, torch.Tensor]:
    state = {
        key: torch.ones(1, dtype=torch.float32)
        for keys in _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS.values()
        for key in keys
    }
    state.update(
        {
            "evidence_fusion_norm.weight": torch.ones(96),
            "evidence_fusion_norm.bias": torch.zeros(96),
            "evidence_fusion_in.weight": torch.ones(128, 96),
            "evidence_fusion_in.bias": torch.zeros(128),
            "evidence_fusion_out.weight": torch.arange(
                3 * 128, dtype=torch.float32
            ).reshape(3, 128)
            + 1.0,
            "evidence_fusion_out.bias": torch.zeros(3),
        }
    )
    mtf_routing = require_multi_tf_specialist_routing_v4(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    for timeframe in ("m5", "m15", "h1", "h4", "d1"):
        for specialist, indices in mtf_routing.items():
            state[
                "mtf_feature_context_gate."
                f"{timeframe}__{specialist}.weight"
            ] = torch.ones(len(indices), 1, dtype=torch.float32)
    return state


def test_model_native_learned_component_liveness_accepts_moved_blocks() -> None:
    _require_model_native_learned_component_liveness(_live_model_native_state())


@pytest.mark.parametrize("component", _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS)
def test_model_native_learned_component_liveness_rejects_pass_through_block(
    component: str,
) -> None:
    state = _live_model_native_state()
    for key in _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS[component]:
        state[key] = torch.zeros(1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match=f"{component}:zero_init_pass_through"):
        _require_model_native_learned_component_liveness(state)


@pytest.mark.parametrize(
    ("component", "bias_key"),
    (
        ("specialist_fusion_output", "specialist_out.bias"),
        ("regime_film", "regime_film.2.bias"),
        ("cross_tf_output", "cross_tf_out.bias"),
        ("family_tf_cooperation_output", "family_tf_cooperation_out.bias"),
    ),
)
def test_model_native_learned_component_liveness_rejects_bias_only_movement(
    component: str,
    bias_key: str,
) -> None:
    state = _live_model_native_state()
    for key in _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS[component]:
        state[key] = torch.zeros(1, dtype=torch.float32)
    state[bias_key] = torch.ones(1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match=f"{component}:zero_init_pass_through"):
        _require_model_native_learned_component_liveness(state)


def test_model_native_learned_component_liveness_rejects_retired_mtf_scale() -> None:
    state = _live_model_native_state()
    state["mtf_dir_scale"] = torch.tensor(0.2)

    with pytest.raises(RuntimeError, match="retired_direction_state"):
        _require_model_native_learned_component_liveness(state)


def test_model_native_liveness_rejects_dead_feature_timeframe_gate_row() -> None:
    state = _live_model_native_state()
    key = (
        "mtf_feature_context_gate."
        "h4__trend_ema_encoder.weight"
    )
    state[key][3] = 0.0

    with pytest.raises(
        RuntimeError,
        match="feature_tf_context_gate:.*zero_init_pass_through_rows",
    ):
        _require_model_native_learned_component_liveness(state)
