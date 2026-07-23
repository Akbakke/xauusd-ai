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
from gx1.contracts.entry_model_native_readiness_v1 import MODEL_NATIVE_ACTIVE_HEADS
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    build_tf_input_scale_contract,
    raw_tf_input_scale_from_effective,
)
from gx1.features.htf_features import (
    HTF_V2_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_NAMES_SHA256_V2,
    MULTI_TF_PER_BAR_FEATURES_V2,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
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
)
from gx1.models.entry_v10.entry_v10_bundle import (
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

    assert got == list(MODEL_NATIVE_ACTIVE_HEADS)


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
        "v2_mode": True,
        "m5_seq_dim": 25,
        "m5_seq_len": 96,
        "m15_seq_dim": 25,
        "m15_seq_len": 64,
        "h1_seq_dim": 25,
        "h1_seq_len": 96,
        "h4_seq_dim": 25,
        "h4_seq_len": 48,
        "d1_seq_dim": 25,
        "d1_seq_len": 30,
        "multi_tf_scale": 0.5,
        "feature_contract": "MULTI_TF_PER_BAR_V2",
        "matrix_contract": HTF_V2_MATRIX_CONTRACT,
        "feature_names": list(MULTI_TF_PER_BAR_FEATURES_V2),
        "feature_names_sha256": MULTI_TF_FEATURE_NAMES_SHA256_V2,
        "closed_bar_target_availability": True,
        "target_availability_shift_minutes": 5.0,
    }
    tf_inits = {"m5": 1.0, "m15": 0.7, "h1": 0.7, "h4": 0.5, "d1": 0.3}
    tf_scale_contract = build_tf_input_scale_contract(
        init_effective=tf_inits,
        learned_raw={
            name: raw_tf_input_scale_from_effective(value)
            for name, value in tf_inits.items()
        },
    )
    input_normalization = input_normalization_fixture(
        signal_names=list(signal_contract["fields"]),
        mtf_names=list(MULTI_TF_PER_BAR_FEATURES_V2),
        per_tf_seq_lens={
            "M5": mtf_contract["m5_seq_len"],
            "M15": mtf_contract["m15_seq_len"],
            "H1": mtf_contract["h1_seq_len"],
            "H4": mtf_contract["h4_seq_len"],
            "D1": mtf_contract["d1_seq_len"],
        },
        dataset_run_id="MODEL_NATIVE_DATASET_PYTEST_V1",
    )
    shared = {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
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
    }
    lock = {
        **copy.deepcopy(shared),
        "neutral_xgb_bridge": False,
        "xgb_bridge_source": None,
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
        "neutral_xgb_bridge": False,
        "xgb_bridge_source": None,
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

    assert set(parameters) == {"bundle_dir", "device", "is_replay"}


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
