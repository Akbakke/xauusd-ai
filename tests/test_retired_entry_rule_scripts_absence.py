from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RETIRED_ENTRY_RULE_PATHS = (
    "gx1/entry/overlap_overlay.py",
    "gx1/research/posthoc_session_strategyf_eval.py",
    "gx1/execution/entry_feature_telemetry.py",
    "gx1/tools/debug/analyze_entry_v10_label_quality.py",
    "gx1/tools/debug/validate_entry_v10_features.py",
    "gx1/tuning/feature_manifest.py",
    "gx1/scripts/materialize_entry_feature_ai_inventory_v1.py",
    "gx1/scripts/run_replay_eval_chain_compute.sh",
    "gx1/scripts/audit_entry_chart_geometry_challenger_v1.py",
    "gx1/scripts/materialize_entry_session_regime_interaction_manifest_v1.py",
    "gx1/scripts/materialize_entry_smc_liquidity_quality_manifest_v1.py",
    "gx1/scripts/materialize_entry_specialist_challenger_extension_manifest_v1.py",
    "gx1/scripts/materialize_entry_trend_ema_extension_manifest_v1.py",
    "gx1/scripts/materialize_entry_momentum_flow_challenger_manifest_v1.py",
    "gx1/scripts/evaluate_entry_selective_edge_v1.py",
    "gx1/scripts/entry_foundation_smoke_train_event_ledger_v1.py",
    "gx1/scripts/materialize_model_native_rank_reference_v1.py",
    "gx1/scripts/materialize_sequence_structure_features_v1.py",
    "gx1/scripts/augment_canonical_v3_with_missing_features.py",
    "gx1/scripts/run_forward_outcome_rebuild.sh",
    "gx1/utils/external_tree_sidecar_feature_names_ssot.py",
    "gx1/execution/v12_counterfactual_replay.py",
    "gx1/research/exit_netcapture.py",
)

RETIRED_ENTRY_RULE_MODULES = (
    "gx1.entry.overlap_overlay",
    "gx1.research.posthoc_session_strategyf_eval",
    "gx1.execution.entry_feature_telemetry",
    "gx1.tools.debug.analyze_entry_v10_label_quality",
    "gx1.tools.debug.validate_entry_v10_features",
    "gx1.tuning.feature_manifest",
    "gx1.scripts.materialize_entry_feature_ai_inventory_v1",
    "gx1.scripts.audit_entry_chart_geometry_challenger_v1",
    "gx1.scripts.materialize_entry_session_regime_interaction_manifest_v1",
    "gx1.scripts.materialize_entry_smc_liquidity_quality_manifest_v1",
    "gx1.scripts.materialize_entry_specialist_challenger_extension_manifest_v1",
    "gx1.scripts.materialize_entry_trend_ema_extension_manifest_v1",
    "gx1.scripts.materialize_entry_momentum_flow_challenger_manifest_v1",
    "gx1.scripts.evaluate_entry_selective_edge_v1",
    "gx1.scripts.entry_foundation_smoke_train_event_ledger_v1",
    "gx1.scripts.materialize_model_native_rank_reference_v1",
    "gx1.scripts.materialize_sequence_structure_features_v1",
    "gx1.scripts.augment_canonical_v3_with_missing_features",
    "gx1.utils.external_tree_sidecar_feature_names_ssot",
    "gx1.execution.v12_counterfactual_replay",
    "gx1.research.exit_netcapture",
)


def test_retired_entry_rule_scripts_remain_absent() -> None:
    present = [path for path in RETIRED_ENTRY_RULE_PATHS if (REPO_ROOT / path).exists()]
    assert present == []


def test_retired_entry_rule_modules_are_not_importable() -> None:
    importable = []
    for module in RETIRED_ENTRY_RULE_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            importable.append(module)
    assert importable == []


def test_environment_example_has_no_retired_direction_or_sizing_knobs() -> None:
    source = (REPO_ROOT / "env.example").read_text(encoding="utf-8")
    forbidden = (
        "PROD_BASELINE",
        "SHADOW_REGIME",
        "SESSION_WINDOW",
        "TRENDINESS_",
        "SHADOW_PIECEWISE",
        "PIECEWISE_POLICY",
        "THROTTLE_",
        "META_THRESHOLD",
        "I_UNDERSTAND_LIVE_TRADING",
        "OANDA_API_KEY",
    )
    assert all(token not in source for token in forbidden)
