from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

RETIRED_ENTRY_EXIT_PLANNING_COMPONENTS = (
    "gx1/models/exit_sequence_transformer/__init__.py",
    "gx1/models/exit_sequence_transformer/train_v1.py",
    "gx1/scripts/audit_entry_exit_feature_alignment_v1.py",
    "gx1/scripts/audit_entry_exit_handoff_readiness_v1.py",
    "gx1/scripts/audit_entry_exit_model_dataset_slice_robustness_v1.py",
    "gx1/scripts/audit_entry_exit_per_bar_reconstruction_v1.py",
    "gx1/scripts/audit_entry_exit_split_leakage_v1.py",
    "gx1/scripts/audit_entry_exit_transformer_architecture_readiness_v1.py",
    "gx1/scripts/audit_entry_exit_transformer_post_train_contract_v1.py",
    "gx1/scripts/audit_entry_exit_transformer_train_execution_review_v1.py",
    "gx1/scripts/audit_entry_exit_transformer_trainer_wrapper_readiness_v1.py",
    "gx1/scripts/materialize_entry_exit_model_dataset_readiness_v1.py",
    "gx1/scripts/materialize_entry_exit_per_bar_handoff_v1.py",
    "gx1/scripts/materialize_entry_exit_state_reward_contract_v1.py",
    "gx1/scripts/materialize_entry_exit_transformer_pretrain_manifest_v1.py",
    "gx1/scripts/materialize_entry_exit_transformer_train_enablement_package_v1.py",
    "gx1/scripts/materialize_entry_exit_transformer_training_plan_readiness_v1.py",
    "scripts/run_entry_exit_transformer_train.sh",
    "gx1/research/iql_training_harness_stub_v1.py",
)

RETIRED_SEPARATE_EXIT_COMPONENTS = (
    "gx1/execution/v12_exit_iql_live.py",
    "gx1/runtime/exit_iql_v2_adapter.py",
    "gx1/scripts/exit_iql_artifact_primitives_v1.py",
    "gx1/scripts/exit_iql_multi_head_gpu_core_v1.py",
    "gx1/scripts/materialize_build_exit_iql_v2.py",
    "gx1/policy/exit_transformer_v0.py",
)


def test_retired_entry_exit_planning_chain_is_physically_absent() -> None:
    present = [path for path in RETIRED_ENTRY_EXIT_PLANNING_COMPONENTS if (REPO / path).exists()]
    assert present == []


def test_separate_exit_runtime_and_policy_components_are_absent() -> None:
    present = [
        path
        for path in RETIRED_SEPARATE_EXIT_COMPONENTS
        if (REPO / path).exists()
    ]
    assert present == []


def test_entry_control_surface_keeps_retired_handoff_and_transformer_routes_closed() -> None:
    control = (REPO / "scripts/entry_next_edge_control.sh").read_text(encoding="utf-8")
    assert "  entry-exit-handoff)" not in control
    assert "  entry-exit-transformer-train)" not in control
