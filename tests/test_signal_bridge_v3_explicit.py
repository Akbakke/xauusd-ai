from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def test_soft_selector_and_tabular_shadow_tombstones_are_absent() -> None:
    absent = (
        "gx1/contracts/signal_bridge_active.py",
        "gx1/runtime/entry_tabular_no_xgb_candidate.py",
        "scripts/run_entry_tabular_no_xgb_shadow_only.sh",
        "gx1/scripts/run_truth_e2e_sanity.py",
        "gx1/scripts/run_fullyear_2025_truth_proof.py",
        "gx1/scripts/launch_truth_monday_week_replay_v1.py",
        "scripts/run_replay.sh",
        "scripts/active/run_replay.sh",
        "scripts/active/replay_entry_exit_parallel.py",
        "gx1/execution/chunk_bootstrap.py",
        "gx1/execution/chunk_data_loader.py",
        "gx1/execution/chunk_failure.py",
        "gx1/execution/replay_merge.py",
        "gx1/execution/replay_eval_collectors.py",
        "gx1/execution/replay_pregate.py",
        "gx1/features/feature_contract_v10_ctx.py",
    )
    assert [path for path in absent if (REPO / path).exists()] == []


def test_retired_context_default_builder_is_absent() -> None:
    # The live model-native state builder owns the exact 142/5 context surface.
    # The old helper silently substituted ASIA, neutral trend, fixed spread and
    # fixed ATR values, and had no production caller.
    assert not (REPO / "gx1/execution/entry_context_features.py").exists()


def test_model_native_candidate_and_exit_evidence_paths_remain() -> None:
    required = (
        "gx1/scripts/materialize_entry_candidate_replay_evidence_v1.py",
        "gx1/scripts/materialize_entry_candidate_replay_trade_log_v1.py",
        "gx1/scripts/verify_entry_replay_readiness_v1.py",
        "gx1/scripts/exit_iql_multi_head_gpu_core_v1.py",
        "gx1/runtime/exit_iql_v2_adapter.py",
    )
    assert [path for path in required if not (REPO / path).is_file()] == []


def test_deleted_shadow_launcher_has_no_guardrail_route() -> None:
    source = _text("gx1/contracts/entry_model_native_readiness_v1.py")
    assert "run_entry_tabular_no_xgb_shadow_only" not in source
    assert "direct_no_xgb_shadow_launcher_blocked" not in source


def test_retired_foundation_activation_lane_is_absent() -> None:
    absent = (
        "gx1/scripts/audit_entry_foundation_worktree_hygiene_v1.py",
        "gx1/scripts/materialize_entry_foundation_smoke_dataset_v1.py",
        "gx1/scripts/plan_entry_foundation_activation_v1.py",
        "gx1/scripts/apply_entry_foundation_activation_v1.py",
        "gx1/scripts/run_entry_foundation_activation_post_apply_v1.py",
        "gx1/scripts/audit_entry_smart_dataset_post_rebuild_readiness_v1.py",
        "scripts/stage_entry_foundation_cleanup.sh",
    )
    assert [path for path in absent if (REPO / path).exists()] == []


def test_hand_written_entry_sizing_overlays_are_absent() -> None:
    absent = (
        "gx1/features/regime_adaptive_sizing_v1.py",
        "gx1/runtime/overlays/entry_v10_1_size_overlay.py",
        "gx1/sniper/policy/sniper_q4_atrend_size_overlay.py",
        "gx1/sniper/policy/sniper_q4_cchop_size_overlay.py",
        "gx1/sniper/policy/sniper_q4_eu_timing_size_overlay.py",
        "gx1/sniper/policy/sniper_regime_size_overlay.py",
        "gx1/sniper/policy/runtime_regime_inputs.py",
    )
    assert [path for path in absent if (REPO / path).exists()] == []


def test_retired_entry_critic_and_timing_filter_chain_is_absent() -> None:
    # These random-forest critics were an optional SNIPER shadow/filter lane.
    # Model-native Entry owns direction, timing and size heads directly; active
    # Exit critic files are deliberately outside this tombstone.
    absent = (
        "gx1/rl/entry_critic_runtime.py",
        "gx1/rl/train_entry_critic_v1.py",
        "gx1/rl/train_entry_timing_critic_v1.py",
    )
    assert [path for path in absent if (REPO / path).exists()] == []


def test_retired_farm_trade_log_alias_schema_is_absent() -> None:
    assert not (REPO / "gx1/execution/trade_log_schema.py").exists()


def test_hand_written_entry_policy_filters_are_absent() -> None:
    absent = (
        "gx1/policy/entry_policy_sniper_core.py",
        "gx1/policy/entry_policy_sniper_v10_ctx.py",
        "gx1/policy/farm_guards.py",
        "gx1/policy/risk_guard_v1.py",
        "gx1/policy/trial160_loader.py",
        "gx1/configs/entry_configs/ENTRY_V10_CTX_SNIPER_REPLAY.yaml",
        "gx1/configs/entry_configs/ENTRY_V10_CTX_STRICT_REPLAY.yaml",
        "gx1/configs/entry_configs/ENTRY_V10_CTX_STRICT_REPLAY_V2.yaml",
        "policies/sniper_trial160_prod.json",
        "policies/trial160_prod_v1.json",
    )
    assert [path for path in absent if (REPO / path).exists()] == []


def test_retired_entry_research_authority_is_absent() -> None:
    assert not (REPO / "gx1/analysis/shadow_meta_v1.py").exists()
    forbidden_name_parts = (
        "monday",
        "r5_2",
        "r6_",
        "truth_entry",
        "truth_rl",
        "truth_trade",
        "protector_first",
    )
    survivors = [
        path.name
        for root in (REPO / "gx1/scripts", REPO / "tests")
        for path in root.glob("*")
        if path.is_file()
        and any(part in path.name.lower() for part in forbidden_name_parts)
    ]
    assert survivors == []
