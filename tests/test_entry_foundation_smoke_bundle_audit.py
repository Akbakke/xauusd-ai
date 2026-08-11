from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    FOUNDATION_AUDIT_POLICY_SHA256,
    FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    MANIFEST_NAME as BUNDLE_COMMIT_MANIFEST_NAME,
    write_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    require_smoke_bundle_audit_contract,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    training_objective_contract_metadata,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    INPUTS as DIRECTION_EVIDENCE_INPUTS,
    direction_evidence_fusion_metadata,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_TIMING_OUTPUT_DIM,
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_ORDER as OFFLINE_RL_ACTION_ORDER,
    HORIZON_BARS as OFFLINE_RL_HORIZON_BARS,
    REWARD_SCALE_BPS as OFFLINE_RL_REWARD_SCALE_BPS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
)
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
from gx1.scripts import audit_entry_foundation_smoke_bundle_v1 as audit
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    PREDICTION_EVIDENCE_SCHEMA_VERSION,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SIGNAL_DIM,
)


STAMP = "20260716T120000123456Z"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _signal_contract() -> dict:
    return model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.smoke_bundle_fixture"
        )
    )


def _objective() -> dict:
    return training_objective_contract_metadata(
        {name: 1.0 for name in REQUIRED_POSITIVE_LOSS_WEIGHTS}
    )


def _prediction_frame(rows: int = 768) -> pd.DataFrame:
    labels = np.arange(rows, dtype=np.int64) % 3
    probabilities = np.full((rows, 3), 0.005, dtype=np.float64)
    probabilities[np.arange(rows), labels] = 0.99
    logits = np.log(probabilities)
    public = np.column_stack([np.maximum(logits[:, 0], logits[:, 1]), logits[:, 2]])
    x = np.linspace(-3.0, 3.0, rows)
    path = np.linspace(-20.0, 20.0, rows)
    gate = np.full((rows, len(MODEL_NATIVE_REQUIRED_SPECIALISTS)), 0.05)
    gate[np.arange(rows), np.arange(rows) % gate.shape[1]] = 0.65
    frame = pd.DataFrame(
        {
            "split": np.where(np.arange(rows) < rows // 2, "val", "test"),
            "model": "entry_model_native_smoke",
            "time": pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"),
            "y_direction": labels,
            "pred_direction": labels,
            "p_long": probabilities[:, 0],
            "p_short": probabilities[:, 1],
            "p_flat": probabilities[:, 2],
            "direction_logits": [row.tolist() for row in logits],
            "public_trade_flat_decision_logits": [row.tolist() for row in public],
            "selection_score_mode": "model_direction_argmax",
            "public_trade_probability": np.where(labels == 2, 0.01, 0.99),
            "public_flat_probability": np.where(labels == 2, 0.99, 0.01),
            "public_trade_flat_margin": public[:, 0] - public[:, 1],
            "public_trade_flat_hard_decision": (labels == 2).astype(np.int64),
            "session": np.asarray(("ASIA", "EU", "OVERLAP", "US"))[np.arange(rows) % 4],
            "vol_regime": (np.arange(rows) % 2).astype(str),
            "tradable_prob": 0.6 + 0.2 * np.tanh(x),
            "path_quality_pred": path,
            "mfe_first_n_pred": x + 4.0,
            "bad_path_prob": np.linspace(0.95, 0.05, rows),
            "clean_edge_prob": 0.6 + 0.2 * np.tanh(x),
            "survival_prob": 0.7 + 0.1 * np.tanh(x),
            "tf_agreement_prob": 0.5 + 0.2 * np.tanh(x),
            "path_quality_log_var": np.linspace(-1.0, 1.0, rows),
            "position_size_pred": np.linspace(0.1, 0.9, rows),
            "p_trade": np.where(labels == 2, 0.05, 0.95),
            "specialist_gate": [row.tolist() for row in gate],
            "path_quality_bps": path,
            "y_bad_path": (path < 0.0).astype(np.int64),
            "mfe_first_n_bps": x + 4.0,
            "y_tradable": (labels != 2).astype(np.int64),
            "y_position_size_target": np.linspace(0.1, 0.9, rows),
            "y_long_path_utility_bps": path + 5.0,
            "y_short_path_utility_bps": path - 5.0,
            "long_path_utility_pred_bps": path + 5.0,
            "short_path_utility_pred_bps": path - 5.0,
        }
    )
    vectors = {**dict(DIRECTION_EVIDENCE_INPUTS), "raw_direction_logits": 3}
    for column, width in vectors.items():
        values = np.column_stack([x + index / 10.0 for index in range(width)])
        frame[column] = (
            values[:, 0]
            if width == 1
            else [row.tolist() for row in values]
        )
    timing_layout = model_native_aux_target_contract_metadata()[
        "turning_point_timing"
    ]["layout"]
    timing = np.zeros((rows, MODEL_NATIVE_TIMING_OUTPUT_DIM), dtype=np.float64)
    ordinal = np.arange(rows, dtype=np.float64)
    for item in timing_layout:
        index = int(item["index"])
        base = np.mod(ordinal + 7.0 * index, 101.0) / 100.0
        if (
            item["target"] == "dip_bottom_frac"
            and int(item["horizon_bars"]) == 12
        ):
            direction_id = 0 if item["direction"] == "long" else 1
            values = np.where(
                labels == direction_id,
                0.05 + 0.15 * base,
                0.30 + 0.65 * base,
            )
        else:
            values = 0.05 + 0.90 * base
        frame[item["target_column"]] = values
        timing[:, index] = values
    frame["timing_pred"] = [row.tolist() for row in timing]
    q_targets = np.zeros(
        (rows, len(OFFLINE_RL_ACTION_ORDER), len(OFFLINE_RL_HORIZON_BARS)),
        dtype=np.float64,
    )
    base = np.mod(np.arange(rows, dtype=np.float64), 97.0) / 96.0
    for horizon_index, horizon in enumerate(OFFLINE_RL_HORIZON_BARS):
        long_reward = np.where(
            labels == 0,
            60.0 + 5.0 * base + horizon_index,
            -30.0 + 5.0 * base,
        )
        short_reward = np.where(
            labels == 1,
            60.0 + 5.0 * base + horizon_index,
            -30.0 + 5.0 * base,
        )
        q_targets[:, 0, horizon_index] = long_reward
        q_targets[:, 1, horizon_index] = short_reward
        frame[f"y_action_value_long_K{horizon}"] = long_reward
        frame[f"y_action_value_short_K{horizon}"] = short_reward
        frame[f"y_action_value_flat_K{horizon}"] = 0.0
    q_values = q_targets / float(OFFLINE_RL_REWARD_SCALE_BPS)
    value = q_values.max(axis=1) - 0.05
    advantage = q_values - value[:, None, :]
    frame["action_value"] = [row.tolist() for row in q_values.reshape(rows, -1)]
    frame["expectile_value"] = [row.tolist() for row in value]
    frame["action_advantage"] = [
        row.tolist() for row in advantage.reshape(rows, -1)
    ]
    return frame


def test_training_objective_proof_requires_positive_meta_lock_identity(tmp_path: Path) -> None:
    objective = _objective()
    _write_json(tmp_path / "bundle_metadata.json", {"model_native_training_objective": objective})
    _write_json(
        tmp_path / "MASTER_TRANSFORMER_LOCK.json",
        {"model_native_training_objective": objective},
    )

    report = audit._model_native_training_objective_contract_report(
        bundle_dir=tmp_path,
        metadata={"model_native_training_objective": objective},
    )

    assert report["decision"] == "PASS"
    assert report["meta_lock_exact"] is True
    assert report["objective"] == objective


def test_training_objective_proof_rejects_zero_or_split_brain(tmp_path: Path) -> None:
    objective = _objective()
    broken = json.loads(json.dumps(objective))
    broken["configurable_positive_loss_weights"][REQUIRED_POSITIVE_LOSS_WEIGHTS[0]] = 0.0
    _write_json(tmp_path / "bundle_metadata.json", {"model_native_training_objective": objective})
    _write_json(
        tmp_path / "MASTER_TRANSFORMER_LOCK.json",
        {"model_native_training_objective": broken},
    )

    report = audit._model_native_training_objective_contract_report(
        bundle_dir=tmp_path,
        metadata={"model_native_training_objective": objective},
    )

    assert report["decision"] == "FAIL"
    assert report["meta_lock_exact"] is False
    assert report["failures"]


def test_direction_metrics_fail_closed_below_immutable_edge_policy() -> None:
    frame = _prediction_frame(384)
    clean = audit._direction_metrics(frame, context="fixture")
    frame.loc[frame.index[:80], "pred_direction"] = 2
    degraded = audit._direction_metrics(frame, context="fixture")

    assert clean["decision"] == "PASS"
    assert clean["trade_direction_precision"] == 1.0
    assert set(clean["prediction_counts"]) == {"LONG", "SHORT", "FLAT"}

    assert degraded["accuracy"] < clean["accuracy"]
    assert degraded["trade_direction_precision"] <= clean["trade_direction_precision"]
    assert degraded["decision"] == "FAIL"
    assert any(
        "precision" in failure.lower() and "below" in failure.lower()
        for failure in degraded["failures"]
    ), degraded["failures"]


def test_direction_metrics_reject_tiny_perfect_support() -> None:
    tiny = audit._direction_metrics(_prediction_frame(30), context="tiny")

    assert tiny["trade_direction_precision"] == 1.0
    assert all(value == 1.0 for value in tiny["precision"].values())
    assert tiny["decision"] == "FAIL"
    assert tiny["trade_rows"] < tiny["minimum_trade_rows"]
    assert any("required support" in failure for failure in tiny["failures"])


def test_wilson_lower_is_finite_and_support_sensitive() -> None:
    assert audit._wilson_lower(0, 0) == 0.0
    assert 0.0 < audit._wilson_lower(1, 1) < audit._wilson_lower(100, 100) < 1.0
    with pytest.raises(ValueError, match="outside"):
        audit._wilson_lower(2, 1)


def test_specialist_gate_requires_all_eight_live_and_nonconstant() -> None:
    frame = _prediction_frame(384)
    passed = audit._specialist_gate_contract(frame, split="val")
    frame["specialist_gate"] = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(frame)
    failed = audit._specialist_gate_contract(frame, split="val")

    assert passed["decision"] == "PASS"
    assert tuple(passed["mean_weight"]) == MODEL_NATIVE_REQUIRED_SPECIALISTS
    assert failed["decision"] == "FAIL"
    assert any("pass-through" in failure or "never top-ranked" in failure for failure in failed["failures"])


def test_active_head_evidence_is_exact_and_hold_is_forbidden() -> None:
    frame = _prediction_frame(384)
    passed = audit._active_head_evidence_contract(frame, split="val")
    frame["hold_horizon_pred"] = 0.5
    failed = audit._active_head_evidence_contract(frame, split="val")

    assert passed["decision"] == "PASS"
    assert passed["active_heads"] == list(MODEL_NATIVE_ACTIVE_HEADS)
    assert passed["blocked_heads"] == list(MODEL_NATIVE_BLOCKED_HEADS)
    assert failed["decision"] == "FAIL"
    assert any("hold_horizon" in failure for failure in failed["failures"])


def test_turning_point_evidence_proves_top_bottom_alignment_and_pockets() -> None:
    frame = _prediction_frame(384)
    passed = audit._turning_point_evidence_contract(frame, split="val")

    broken = frame.copy()
    broken["timing_pred"] = [[0.5] * MODEL_NATIVE_TIMING_OUTPUT_DIM] * len(broken)
    failed = audit._turning_point_evidence_contract(broken, split="val")

    assert passed["decision"] == "PASS"
    assert set(passed["near_turn_pockets"]) == {"BOTTOM", "TOP"}
    assert all(
        row["direction_precision"] == 1.0
        and row["timing_precision"] == 1.0
        for row in passed["near_turn_pockets"].values()
    )
    assert failed["decision"] == "FAIL"
    assert failed["failures"]


def test_offline_rl_evidence_requires_target_ranking_and_q_v_parity() -> None:
    frame = _prediction_frame(384)
    passed = audit._offline_rl_evidence_contract(frame, split="val")

    broken = frame.copy()
    broken["action_advantage"] = [[0.0] * 9] * len(broken)
    failed = audit._offline_rl_evidence_contract(broken, split="val")

    assert passed["decision"] == "PASS"
    assert all(
        row["accuracy"] == 1.0
        for row in passed["reward_argmax_ranking"].values()
    )
    assert passed["advantage_max_abs_error"] == pytest.approx(0.0)
    assert failed["decision"] == "FAIL"
    assert any("Advantage" in failure for failure in failed["failures"])


def test_parser_has_no_defaults_or_retired_aliases() -> None:
    parser = audit.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    required = [
        "--bundle-dir", f"/tmp/bundle_{STAMP}",
        "--dataset-dir", f"/tmp/dataset_{STAMP}",
        "--val-manifest-json", "/tmp/val.manifest.json",
        "--predictions-parquet", f"/tmp/selective_edge_predictions_{STAMP}.parquet",
        "--predictions-sha256", "a" * 64,
        "--prediction-report-json", f"/tmp/ENTRY_CANDIDATE_SELECTIVE_EDGE_{STAMP}.json",
        "--target-audit-json", f"/tmp/ENTRY_TARGET_FOUNDATION_AUDIT_{STAMP}.json",
        "--specialist-audit-json", f"/tmp/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_{STAMP}.json",
        "--pretrain-audit-json", f"/tmp/XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_{STAMP}.json",
        "--out-dir", "/tmp/out",
        "--device", "cpu",
    ]
    args = parser.parse_args(required)
    assert args.device == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--smart-seq520"])


def test_bundle_contract_uses_strict_loader_and_proves_full_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / f"entry_model_native_bundle_{STAMP}"
    bundle.mkdir()
    state_path = bundle / "model_state_dict.pt"
    state_path.write_bytes(b"state")
    state_sha = audit._sha256_file(state_path)
    signal = _signal_contract()
    direction = model_direction_decision_contract_metadata()
    specialist = {
        "enabled": True,
        "input_indices": {
            name: [index]
            for index, name in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        },
        "trainable_specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "specialist_model_contract": MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
        "specialist_model_contract_valid": True,
        "specialist_model_contract_set_exact": True,
        "specialist_model_contract_owned_objectives_match": True,
        "specialist_model_contract_signal_families_match": True,
        "specialist_model_contract_support_heads_match": True,
        "specialist_model_contract_model_roles_match": True,
    }
    metadata = {
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": 96,
        "ctx_cont_dim": 142,
        "ctx_cat_dim": 5,
        "state_dict_sha256": state_sha,
        "model_native_signal_contract": signal,
        "direction_decision_contract": direction,
        "multi_tf": {"enabled": True, "v4_mode": True},
        "enable_pos_enc": True,
        "enable_regime_film": True,
        "specialist_fusion": specialist,
        "model_native_direction_evidence_fusion": (
            direction_evidence_fusion_metadata()
        ),
        "sanity_bundle": False,
    }
    lock = {"model_sha256": state_sha}
    (bundle / "bundle_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _write_json(bundle / "MASTER_TRANSFORMER_LOCK.json", lock)
    write_bundle_commit_manifest(
        bundle_dir=bundle.resolve(),
        artifact_names=(
            "MASTER_TRANSFORMER_LOCK.json",
            "bundle_metadata.json",
            "model_state_dict.pt",
        ),
        bundle_kind="trained",
        created_at_utc="2026-07-16T12:00:00+00:00",
    )
    state_keys = {
        "cross_tf_attn.in_proj_weight",
        "tf_gate_logits",
        "tf_token_identity",
        "tf_context_gate.weight",
        "tf_token_gate.weight",
        "cross_tf_out.weight",
        "head_mtf_direction.weight",
        "head_mtf_direction.bias",
        "regime_film.2.weight",
        "specialist_encoder.structure_swing_encoder.layers.0.weight",
        "specialist_token_identity",
        "specialist_cross_attn.layers.0.weight",
        "specialist_token_gate.weight",
        "family_tf_token_identity",
        "family_axis_attn.layers.0.weight",
        "timeframe_axis_attn.layers.0.weight",
        "mtf_family_encoder.structure_swing_encoder.layers.0.weight",
        "mtf_feature_context_gate.m5__structure_swing_encoder.weight",
        "family_tf_context_gate.weight",
        "family_tf_token_gate.weight",
        "family_tf_cooperation_out.weight",
        "evidence_fusion_norm.weight",
        "evidence_fusion_norm.bias",
        "evidence_fusion_in.weight",
        "evidence_fusion_in.bias",
        "evidence_fusion_out.weight",
        "evidence_fusion_out.bias",
        *(f"tf_input_scale_{tf}" for tf in ("m5", "m15", "h1", "h4", "d1")),
    }
    model = SimpleNamespace(
        pos_enc=audit.torch.tensor(1.0),
        pos_enc_m5=audit.torch.tensor(1.0),
        pos_enc_m15=audit.torch.tensor(1.0),
        pos_enc_h1=audit.torch.tensor(1.0),
        pos_enc_h4=audit.torch.tensor(1.0),
        pos_enc_d1=audit.torch.tensor(1.0),
        state_dict=lambda: {key: audit.torch.tensor(1.0) for key in state_keys},
    )
    calls: list[dict] = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(transformer_model=model)

    monkeypatch.setattr(audit, "load_entry_v10_ctx_bundle", fake_loader)

    report, _, _, loaded = audit._bundle_contract_report(
        bundle_dir=bundle,
        device="cpu",
    )

    assert report["decision"] == "PASS", report["failures"]
    assert all(value is True for key, value in report["full_stack"].items() if key != "multi_tf_timeframes")
    assert calls == [
        {
            "bundle_dir": bundle,
            "device": "cpu",
        }
    ]
    assert loaded is not None


def test_run_publishes_exact_consumer_contract_without_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / f"entry_model_native_bundle_{STAMP}"
    dataset = tmp_path / f"entry_model_native_dataset_{STAMP}"
    out_dir = tmp_path / "audit_output"
    bundle.mkdir()
    dataset.mkdir()
    objective = _objective()
    direction = model_direction_decision_contract_metadata()
    signal = _signal_contract()
    metadata = {
        "model_native_training_objective": objective,
        "model_native_signal_contract": signal,
        "direction_decision_contract": direction,
    }
    _write_json(bundle / "bundle_metadata.json", metadata)
    _write_json(
        bundle / "MASTER_TRANSFORMER_LOCK.json",
        {"model_native_training_objective": objective},
    )
    (bundle / "model_state_dict.pt").write_bytes(b"model-native-state")
    write_bundle_commit_manifest(
        bundle_dir=bundle.resolve(),
        artifact_names=(
            "MASTER_TRANSFORMER_LOCK.json",
            "bundle_metadata.json",
            "model_state_dict.pt",
        ),
        bundle_kind="trained",
        created_at_utc="2026-07-16T12:00:00+00:00",
    )
    val_manifest = _write_json(dataset / "xau_val.manifest.json", {"fixture": True})
    predictions = tmp_path / f"selective_edge_predictions_{STAMP}.parquet"
    predictions.write_bytes(b"immutable-predictions")
    prediction_report = _write_json(
        tmp_path / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{STAMP}.json",
        {"schema_version": "entry_candidate_selective_edge_v1"},
    )
    audits = {
        "target": _write_json(tmp_path / f"ENTRY_TARGET_FOUNDATION_AUDIT_{STAMP}.json", {}),
        "specialist": _write_json(tmp_path / f"ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_{STAMP}.json", {}),
        "pretrain": _write_json(tmp_path / f"XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_{STAMP}.json", {}),
    }
    frame = _prediction_frame()

    manifest_report = {
        "decision": "PASS",
        "failures": [],
        "splits": {
            split: {
                "decision": "PASS",
                "failures": [],
                "path": str(path),
                "sha256": audit._sha256_file(path),
                "parquet_path": str(dataset / f"xau_{split}.parquet"),
                "parquet_sha256": "a" * 64,
                "model_native_signal_contract_sha256": "b" * 64,
            }
            for split, path in (("val", val_manifest),)
        },
    }
    monkeypatch.setattr(
        audit,
        "_dataset_manifest_contract",
        lambda **_: (manifest_report, signal),
    )

    audit_schemas = {
        "target": FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
        "specialist": "entry_specialist_feature_group_audit_v1",
        "pretrain": "xau_direction_repair_pretrain_audit_v2",
    }

    def fake_input_audit_contract(
        *,
        name: str,
        path: Path,
        dataset_dir: Path,
        expected_split_artifacts: object,
    ):
        assert expected_split_artifacts == manifest_report["splits"]
        report = {
            "path": str(audits[name]),
            "sha256": audit._sha256_file(audits[name]),
            "schema_version": audit_schemas[name],
            "decision": "PASS",
            "failures": [],
        }
        if name in {"target", "specialist"}:
            report.update(foundation_audit_policy_binding())
            report["data_splits"] = list(FOUNDATION_AUDIT_DATA_SPLITS)
            report["foundation_audit_policy_enforcement"] = (
                foundation_audit_policy_enforcement(name)
            )
        return report, {}

    monkeypatch.setattr(audit, "_input_audit_contract", fake_input_audit_contract)
    bundle_contract = {
        "decision": "PASS",
        "failures": [],
        "commit_path": str((bundle / BUNDLE_COMMIT_MANIFEST_NAME).resolve()),
        "commit_sha256": audit._sha256_file(
            bundle / BUNDLE_COMMIT_MANIFEST_NAME
        ),
        "metadata_path": str((bundle / "bundle_metadata.json").resolve()),
        "metadata_sha256": audit._sha256_file(bundle / "bundle_metadata.json"),
        "lock_path": str((bundle / "MASTER_TRANSFORMER_LOCK.json").resolve()),
        "lock_sha256": audit._sha256_file(bundle / "MASTER_TRANSFORMER_LOCK.json"),
        "state_path": str((bundle / "model_state_dict.pt").resolve()),
        "state_sha256": audit._sha256_file(bundle / "model_state_dict.pt"),
        "state_sha256_matches_metadata_and_lock": True,
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": 96,
        "ctx_cont_dim": 142,
        "ctx_cat_dim": 5,
        "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
        "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
        "specialist_groups": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "specialist_model_contract_sha256": audit.MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT_SHA256,
        "full_stack": {"all": True},
    }
    monkeypatch.setattr(
        audit,
        "_bundle_contract_report",
        lambda **_: (bundle_contract, metadata, direction, object()),
    )
    evidence = {
        "schema_version": PREDICTION_EVIDENCE_SCHEMA_VERSION,
        "evidence_stage": "pre_calibration",
        "authoritative": False,
        "runtime_head_evidence_authoritative": False,
        "path": str(predictions.resolve()),
        "sha256": audit._sha256_file(predictions),
        "rows": int((frame["split"].astype(str) == "val").sum()),
        "splits": ["val"],
        "models": ["entry_model_native_smoke"],
    }
    resolver_calls: list[dict] = []

    def fake_prediction_resolver(*_, **kwargs):
        resolver_calls.append(kwargs)
        return (
            predictions.resolve(),
            {
                "schema_version": "entry_candidate_selective_edge_v1",
                "evidence_stage": "pre_calibration",
            },
            evidence,
        )

    monkeypatch.setattr(
        audit,
        "resolve_and_validate_prediction_evidence",
        fake_prediction_resolver,
    )
    monkeypatch.setattr(
        audit.pd,
        "read_parquet",
        lambda _: frame.loc[
            frame["split"].astype(str) == "val"
        ].reset_index(drop=True),
    )

    report = audit.run(
        Namespace(
            bundle_dir=str(bundle),
            dataset_dir=str(dataset),
            val_manifest_json=str(val_manifest),
            predictions_parquet=str(predictions),
            predictions_sha256=audit._sha256_file(predictions),
            prediction_report_json=str(prediction_report),
            target_audit_json=str(audits["target"]),
            specialist_audit_json=str(audits["specialist"]),
            pretrain_audit_json=str(audits["pretrain"]),
            out_dir=str(out_dir),
            device="cpu",
            quiet=True,
        )
    )

    assert report["decision"] == "PASS", report["failures"]
    assert resolver_calls == [
        {
            "expected_sha256": audit._sha256_file(predictions),
            "prediction_report_path": prediction_report,
            "bundle_dir": bundle,
            "dataset_dir": dataset,
            "expected_stage": "pre_calibration",
            "expected_splits": ("val",),
        }
    ]
    normalized = require_smoke_bundle_audit_contract(report, context="TEST")
    assert normalized["head_contract"]["active_heads"] == list(MODEL_NATIVE_ACTIVE_HEADS)
    assert normalized["specialist_contract"]["specialists"] == list(
        MODEL_NATIVE_REQUIRED_SPECIALISTS
    )
    assert normalized["foundation_audit_policy_sha256"] == (
        FOUNDATION_AUDIT_POLICY_SHA256
    )
    assert not list(out_dir.glob("*latest*"))
    assert len(list(out_dir.glob("ENTRY_MODEL_NATIVE_SMOKE_BUNDLE_AUDIT_*.json"))) == 1

    forged = json.loads(json.dumps(report))
    forged["input_audits"]["specialist"]["foundation_audit_policy"][
        "specialist_liveness"
    ]["min_feature_active_rate"] = 0.0
    with pytest.raises(RuntimeError, match="POLICY_PAYLOAD_INVALID"):
        require_smoke_bundle_audit_contract(forged, context="FORGED")


def test_bundle_must_be_timestamped_and_dataset_must_be_explicit(tmp_path: Path) -> None:
    path = tmp_path / "bundle_latest"
    path.mkdir()
    with pytest.raises(RuntimeError, match="latest"):
        audit._timestamped_directory(path, label="bundle")
    plain = tmp_path / "bundle_without_stamp"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="UTC stamp"):
        audit._timestamped_directory(plain, label="bundle")
    assert audit._explicit_immutable_directory(plain, label="dataset") == plain.resolve()
