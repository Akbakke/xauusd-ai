import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_entry_exit_feature_alignment_v1 import build_parser, run


BASE_STATE = [
    "running_pnl_bps",
    "running_mfe_bps",
    "running_mae_bps",
    "running_giveback_bps",
    "bars_held",
    "spread_bps",
    "atr_bps",
    "session",
    "vol_regime",
    "side",
    "entry_score",
    "entry_p_long",
    "entry_p_short",
    "entry_p_flat",
    "entry_path_quality_pred",
    "entry_bad_path_prob",
]
FULL_ALIGNMENT = [
    "entry_hh_state",
    "entry_bos_age",
    "entry_smc_sweep_reclaim",
    "entry_false_breakout_pressure",
    "entry_h1_ema_trend",
    "entry_mtf_tf_agreement",
    "entry_range_compression",
    "entry_momentum_impulse",
    "entry_structure_swing_gate_weight",
    "entry_smc_liquidity_gate_weight",
    "entry_trend_ema_gate_weight",
    "entry_vol_compression_gate_weight",
    "entry_momentum_flow_gate_weight",
    "entry_session_regime_gate_weight",
]
SEQ215_ALIGNMENT = [
    "entry_chart_geometry_fib_zone_pressure",
    "entry_chart_geometry_trendline_channel_pressure",
    "entry_price_action_candle_engulfing_pressure",
    "entry_price_action_candle_doji_indecision",
    "entry_chart_geometry_gate_weight",
    "entry_price_action_candle_gate_weight",
]
SMART_ALIGNMENT = [
    "entry_trend_ema_stack_alignment_score",
    "entry_smc_liquidity_sweep_low_quality_long",
    "entry_structure_swing_swing_leg_quality_up",
    "entry_momentum_flow_micro_impulse_norm",
    "entry_session_regime_session_boundary_risk",
    "entry_vol_compression_squeeze_state_score",
    "entry_chart_geometry_trendline_channel_confluence_pressure",
    "entry_price_action_candle_pattern_close_pressure_signed",
    "entry_support_resistance_sr_memory_nearest_level_proximity",
    "entry_mtf_confluence_trend_tf_conflict",
]
PROVENANCE = ["entry_trade_id", "entry_iql_policy_id", "entry_replay_identity_hash"]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_fixture(
    tmp_path: Path,
    *,
    full_alignment: bool,
    seq215_alignment: bool = False,
    smart_alignment: bool = False,
) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    state = [
        *BASE_STATE,
        *(FULL_ALIGNMENT if full_alignment else []),
        *(SEQ215_ALIGNMENT if seq215_alignment else []),
        *(SMART_ALIGNMENT if smart_alignment else []),
    ]
    schema = {
        "state_feature_names": state,
        "numeric_state_features": [field for field in state if field not in {"session", "vol_regime", "side"}],
        "categorical_state_features": ["session", "vol_regime", "side"],
        "provenance_columns": PROVENANCE,
    }
    schema_path = _write_json(root / "feature_schema.json", schema)
    shards: dict[str, str] = {}
    for split in ("train", "val", "test"):
        rows = []
        for idx in range(3):
            row = {
                "entry_trade_id": f"{split}_{idx}",
                "entry_iql_policy_id": "entry_iql_student",
                "entry_replay_identity_hash": "abc",
                "running_pnl_bps": float(idx),
                "running_mfe_bps": float(idx + 1),
                "running_mae_bps": float(idx + 2),
                "running_giveback_bps": float(idx + 3),
                "bars_held": idx,
                "spread_bps": 1.0 + idx,
                "atr_bps": 10.0 + idx,
                "session": "US" if idx % 2 else "EU",
                "vol_regime": "HIGH" if idx % 2 else "LOW",
                "side": "LONG" if idx % 2 else "SHORT",
                "entry_score": 0.8 + idx / 100.0,
                "entry_p_long": 0.6,
                "entry_p_short": 0.3,
                "entry_p_flat": 0.1,
                "entry_path_quality_pred": 2.0 + idx,
                "entry_bad_path_prob": 0.2 + idx / 100.0,
            }
            for pos, field in enumerate([*FULL_ALIGNMENT, *SEQ215_ALIGNMENT, *SMART_ALIGNMENT]):
                if field in state:
                    row[field] = float(idx + pos + 1)
            rows.append(row)
        path = root / f"{split}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        shards[split] = str(path)
    report = {
        "decision": "ENTRY_EXIT_MODEL_DATASET_READY_FOR_EXIT_TRANSFORMER_READINESS_REVIEW",
        "feature_schema_json": str(schema_path),
        "model_dataset_shards": shards,
    }
    return _write_json(root / "model_dataset.json", report)


def _args(tmp_path: Path, model_dataset: Path, *, contract_mode: str = "foundation_seq146") -> argparse.Namespace:
    return argparse.Namespace(
        model_dataset_json=str(model_dataset),
        out_dir=str(tmp_path / "out"),
        entry_specialist_contract_mode=contract_mode,
        fail_on_not_ready=False,
        quiet=True,
    )


def test_entry_exit_feature_alignment_passes_with_all_market_families(tmp_path: Path) -> None:
    model_dataset = _write_fixture(tmp_path, full_alignment=True)

    report = run(_args(tmp_path, model_dataset))

    assert report["decision"] == "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
    assert report["entry_specialist_contract_mode"] == "foundation_seq146"
    assert report["required_entry_specialist_gate_fields"] == [
        "entry_structure_swing_gate_weight",
        "entry_smc_liquidity_gate_weight",
        "entry_trend_ema_gate_weight",
        "entry_vol_compression_gate_weight",
        "entry_momentum_flow_gate_weight",
        "entry_session_regime_gate_weight",
    ]
    assert report["family_review"]["ready"] is True
    skipped = set(report["family_review"]["skipped_families"])
    assert {"chart_geometry_context", "price_action_candle_context"}.issubset(skipped)
    assert "smart_trend_ema_layer_context" in skipped
    assert "smart_mtf_confluence_context" in skipped
    assert report["exit_training_allowed"] is False
    assert report["exit_iql_allowed"] is False
    assert report["trainer_started"] is False


def test_entry_exit_feature_alignment_blocks_missing_market_families(tmp_path: Path) -> None:
    model_dataset = _write_fixture(tmp_path, full_alignment=False)

    report = run(_args(tmp_path, model_dataset))

    assert report["decision"] == "BLOCKED_BY_ENTRY_EXIT_FEATURE_ALIGNMENT"
    missing = set(report["family_review"]["missing_families"])
    assert "structure_swing" in missing
    assert "smc_liquidity" in missing
    assert "trend_ema_mtf" in missing
    assert "momentum_flow" in missing
    failed = {row["check"] for row in report["failures"]}
    assert "Entry-to-Exit market mechanism families are present as model state" in failed


def test_entry_exit_feature_alignment_blocks_seq215_without_challenger_state_and_gates(tmp_path: Path) -> None:
    model_dataset = _write_fixture(tmp_path, full_alignment=True)

    report = run(_args(tmp_path, model_dataset, contract_mode="challenger_seq215"))

    assert report["decision"] == "BLOCKED_BY_ENTRY_EXIT_FEATURE_ALIGNMENT"
    assert report["entry_specialist_contract_mode"] == "challenger_seq215"
    assert report["required_entry_specialist_gate_fields"][-2:] == [
        "entry_chart_geometry_gate_weight",
        "entry_price_action_candle_gate_weight",
    ]
    missing = set(report["family_review"]["missing_families"])
    assert "chart_geometry_context" in missing
    assert "price_action_candle_context" in missing
    assert "entry_specialist_gate_outputs" in missing
    gate_review = report["family_review"]["families"]["entry_specialist_gate_outputs"]
    assert "entry_chart_geometry_gate_weight" in gate_review["missing_required_fields"]
    assert "entry_price_action_candle_gate_weight" in gate_review["missing_required_fields"]


def test_entry_exit_feature_alignment_passes_seq215_with_challenger_state_and_eight_gates(tmp_path: Path) -> None:
    model_dataset = _write_fixture(tmp_path, full_alignment=True, seq215_alignment=True)

    report = run(_args(tmp_path, model_dataset, contract_mode="challenger_seq215"))

    assert report["decision"] == "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
    assert report["entry_specialist_contract_mode"] == "challenger_seq215"
    assert report["family_review"]["ready"] is True
    assert report["family_review"]["families"]["chart_geometry_context"]["ok"] is True
    assert report["family_review"]["families"]["price_action_candle_context"]["ok"] is True
    assert len(report["family_review"]["families"]["entry_specialist_gate_outputs"]["present_fields"]) == 8


def test_entry_exit_feature_alignment_blocks_smart_seq520_without_eight_specialist_state(tmp_path: Path) -> None:
    model_dataset = _write_fixture(tmp_path, full_alignment=True)

    report = run(_args(tmp_path, model_dataset, contract_mode="smart_seq520_candidate"))

    assert report["decision"] == "BLOCKED_BY_ENTRY_EXIT_FEATURE_ALIGNMENT"
    assert report["entry_specialist_contract_mode"] == "smart_seq520_candidate"
    assert report["required_entry_specialist_gate_fields"][-2:] == [
        "entry_chart_geometry_gate_weight",
        "entry_price_action_candle_gate_weight",
    ]
    missing = set(report["family_review"]["missing_families"])
    assert "chart_geometry_context" in missing
    assert "price_action_candle_context" in missing
    assert "entry_specialist_gate_outputs" in missing
    assert "smart_trend_ema_layer_context" in missing
    assert "smart_smc_liquidity_quality_context" in missing
    assert "smart_structure_swing_derivation_context" in missing
    assert "smart_momentum_flow_context" in missing
    assert "smart_session_regime_interaction_context" in missing
    assert "smart_vol_compression_context" in missing
    assert "smart_chart_geometry_context" in missing
    assert "smart_price_action_candle_context" in missing
    assert "smart_support_resistance_memory_context" in missing
    assert "smart_mtf_confluence_context" in missing


def test_entry_exit_feature_alignment_blocks_smart_seq520_without_smart_layer_state(tmp_path: Path) -> None:
    model_dataset = _write_fixture(tmp_path, full_alignment=True, seq215_alignment=True)

    report = run(_args(tmp_path, model_dataset, contract_mode="smart_seq520_candidate"))

    assert report["decision"] == "BLOCKED_BY_ENTRY_EXIT_FEATURE_ALIGNMENT"
    missing = set(report["family_review"]["missing_families"])
    assert "chart_geometry_context" not in missing
    assert "price_action_candle_context" not in missing
    assert "entry_specialist_gate_outputs" not in missing
    assert "smart_trend_ema_layer_context" in missing
    assert "smart_mtf_confluence_context" in missing


def test_entry_exit_feature_alignment_passes_smart_seq520_with_smart_layer_state(tmp_path: Path) -> None:
    model_dataset = _write_fixture(tmp_path, full_alignment=True, seq215_alignment=True, smart_alignment=True)

    report = run(_args(tmp_path, model_dataset, contract_mode="smart_seq520_candidate"))

    assert report["decision"] == "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
    assert report["entry_specialist_contract_mode"] == "smart_seq520_candidate"
    assert report["family_review"]["ready"] is True
    assert "chart_geometry_context" not in report["family_review"]["skipped_families"]
    assert "price_action_candle_context" not in report["family_review"]["skipped_families"]
    assert len(report["family_review"]["families"]["entry_specialist_gate_outputs"]["present_fields"]) == 8
    assert report["required_smart_layer_families"] == [
        "trend_ema_smart_layer",
        "smc_liquidity_quality_layer",
        "structure_swing_derivation_layer",
        "momentum_flow_smart_layer",
        "session_regime_interaction_layer",
        "vol_compression_smart_layer",
        "chart_geometry_smart2_layer",
        "price_action_candle_smart3_layer",
        "support_resistance_memory_layer",
        "mtf_confluence_layer",
    ]
    for family in (
        "smart_trend_ema_layer_context",
        "smart_smc_liquidity_quality_context",
        "smart_structure_swing_derivation_context",
        "smart_momentum_flow_context",
        "smart_session_regime_interaction_context",
        "smart_vol_compression_context",
        "smart_chart_geometry_context",
        "smart_price_action_candle_context",
        "smart_support_resistance_memory_context",
        "smart_mtf_confluence_context",
    ):
        assert report["family_review"]["families"][family]["ok"] is True


def test_entry_exit_feature_alignment_parser_accepts_smart_seq520_mode() -> None:
    args = build_parser().parse_args(["--entry-specialist-contract-mode", "smart_seq520_candidate"])

    assert args.entry_specialist_contract_mode == "smart_seq520_candidate"
