import numpy as np
import json

from gx1.features.entry_specialist_feature_groups_v1 import (
    CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT,
    CHALLENGER_SEQ215_TRAINING_SPECIALISTS,
    SPECIALIST_MODEL_CONTRACT,
)
from gx1.scripts.audit_entry_foundation_smoke_bundle_v1 import (
    _direction_metrics,
    _gate_stats,
    _head_contract_report,
    _majority_baseline_accuracy,
    _path_calibration_recipe_contract,
    _pretrain_manifest_contract_report,
    _sha256_file,
    _spearman,
    _specialist_gate_failures,
    _specialist_model_contract_report,
)
from gx1.scripts.verify_entry_training_readiness_v1 import (
    EXPECTED_ACTIVE_TRAINING_HEADS,
    EXPECTED_BLOCKED_HEADS,
)

REQUIRED_SPECIALISTS = [
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
]


def _specialist_model_contract_payload() -> dict:
    return json.loads(json.dumps(SPECIALIST_MODEL_CONTRACT))


def _challenger_seq215_specialist_model_contract_payload() -> dict:
    return json.loads(json.dumps(CHALLENGER_SEQ215_SPECIALIST_MODEL_CONTRACT))


def test_majority_baseline_accuracy_counts_largest_class() -> None:
    labels = np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)
    assert _majority_baseline_accuracy(labels) == 0.5


def test_direction_metrics_reports_accuracy_and_prediction_distribution() -> None:
    logits = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 4.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 2, 2], dtype=np.int64)

    metrics = _direction_metrics(logits, labels)

    assert metrics["rows"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["majority_baseline_accuracy"] == 0.5
    assert metrics["beats_majority_baseline"] is True
    assert metrics["prediction_counts"] == {"LONG": 2, "SHORT": 1, "FLAT": 1}


def test_gate_stats_checks_row_sums_and_mean_weights() -> None:
    gate = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float32,
    )

    stats = _gate_stats(gate, ["structure", "smc", "vol"])

    assert stats["finite"] is True
    assert stats["specialist_count"] == 3
    assert stats["row_sum_max_abs_error"] < 1e-6
    assert stats["active_specialist_count_gt_1pct"] == 3
    assert abs(stats["mean_weight"]["vol"] - 0.65) < 1e-6


def test_specialist_gate_failures_accepts_live_multi_specialist_gate() -> None:
    names = [
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
        "session_regime_encoder",
    ]
    gate = np.full((4, len(names)), 1.0 / len(names), dtype=np.float32)

    failures = _specialist_gate_failures(
        split="val",
        gate_report=_gate_stats(gate, names),
        specialist_names=names,
        required_specialists=tuple(names),
        min_active_specialists=6,
        min_gate_entropy=0.05,
    )

    assert failures == []


def test_specialist_gate_failures_rejects_collapsed_gate() -> None:
    names = [
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
        "session_regime_encoder",
    ]
    gate = np.array([[0.999, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002]], dtype=np.float32)

    failures = _specialist_gate_failures(
        split="test",
        gate_report=_gate_stats(gate, names),
        specialist_names=names,
        required_specialists=tuple(names),
        min_active_specialists=6,
        min_gate_entropy=0.05,
    )

    assert any("collapsed" in failure for failure in failures)


def test_specialist_model_contract_report_accepts_exact_contract() -> None:
    report = _specialist_model_contract_report(_specialist_model_contract_payload())

    assert report["decision"] == "PASS"
    assert report["contract_mode"] == "foundation_seq146"
    assert report["valid"] is True
    assert report["set_exact"] is True
    assert report["owned_objectives_match"] is True
    assert report["support_heads_match"] is True
    assert report["signal_families_match"] is True
    assert report["model_roles_match"] is True
    assert report["failures"] == []


def test_specialist_model_contract_report_accepts_challenger_seq215_contract() -> None:
    report = _specialist_model_contract_report(
        _challenger_seq215_specialist_model_contract_payload(),
        contract_mode="challenger_seq215",
    )

    assert report["decision"] == "PASS"
    assert report["contract_mode"] == "challenger_seq215"
    assert report["valid"] is True
    assert report["set_exact"] is True
    assert report["observed_specialists"] == sorted(CHALLENGER_SEQ215_TRAINING_SPECIALISTS)
    assert "chart_geometry_encoder" in report["observed_specialists"]
    assert "price_action_candle_encoder" in report["observed_specialists"]


def test_specialist_model_contract_report_rejects_challenger_contract_under_foundation_mode() -> None:
    report = _specialist_model_contract_report(_challenger_seq215_specialist_model_contract_payload())

    assert report["decision"] == "FAIL"
    assert report["valid"] is False
    assert report["set_exact"] is False
    assert any("set mismatch" in failure for failure in report["failures"])


def test_specialist_model_contract_report_rejects_tampered_contract() -> None:
    contract = _specialist_model_contract_payload()
    contract["structure_swing_encoder"]["owned_objectives"] = []
    contract["smc_liquidity_encoder"]["supports_heads"] = []

    report = _specialist_model_contract_report(contract)

    assert report["decision"] == "FAIL"
    assert report["valid"] is False
    assert report["set_exact"] is True
    assert report["owned_objectives_match"] is False
    assert report["support_heads_match"] is False
    assert any("owned objectives mismatch" in failure for failure in report["failures"])
    assert any("support heads mismatch" in failure for failure in report["failures"])


def test_specialist_gate_failures_rejects_collapsed_required_group_with_extra_group() -> None:
    names = [
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
        "session_regime_encoder",
        "price_action_candle_encoder",
    ]
    gate = np.array([[0.18, 0.17, 0.16, 0.16, 0.0, 0.16, 0.17]], dtype=np.float32)

    failures = _specialist_gate_failures(
        split="test",
        gate_report=_gate_stats(gate, names),
        specialist_names=names,
        required_specialists=(
            "structure_swing_encoder",
            "smc_liquidity_encoder",
            "trend_ema_encoder",
            "vol_compression_encoder",
            "momentum_flow_encoder",
            "session_regime_encoder",
        ),
        min_active_specialists=6,
        min_gate_entropy=0.05,
    )

    assert any("required specialist gate weights collapsed" in failure for failure in failures)
    assert any("non-required specialists" in failure for failure in failures)


def test_specialist_gate_failures_rejects_extra_ungated_group() -> None:
    names = [
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
        "session_regime_encoder",
        "price_action_candle_encoder",
    ]
    gate = np.full((3, len(names)), 1.0 / len(names), dtype=np.float32)

    failures = _specialist_gate_failures(
        split="val",
        gate_report=_gate_stats(gate, names),
        specialist_names=names,
        required_specialists=(
            "structure_swing_encoder",
            "smc_liquidity_encoder",
            "trend_ema_encoder",
            "vol_compression_encoder",
            "momentum_flow_encoder",
            "session_regime_encoder",
        ),
        min_active_specialists=6,
        min_gate_entropy=0.05,
    )

    assert any("non-required specialists" in failure for failure in failures)


def test_specialist_gate_failures_rejects_missing_required_specialist() -> None:
    names = [
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
    ]
    gate = np.full((2, len(names)), 1.0 / len(names), dtype=np.float32)

    failures = _specialist_gate_failures(
        split="val",
        gate_report=_gate_stats(gate, names),
        specialist_names=names,
        required_specialists=(
            "structure_swing_encoder",
            "smc_liquidity_encoder",
            "trend_ema_encoder",
            "vol_compression_encoder",
            "momentum_flow_encoder",
            "session_regime_encoder",
        ),
        min_active_specialists=5,
        min_gate_entropy=0.05,
    )

    assert any("metadata missing required specialists" in failure for failure in failures)


def test_spearman_reports_negative_bad_path_relation() -> None:
    bad_path_prob = np.array([0.9, 0.7, 0.2, 0.1], dtype=np.float32)
    path_quality = np.array([-8.0, -2.0, 5.0, 12.0], dtype=np.float32)

    assert _spearman(bad_path_prob, path_quality) == -1.0


def test_path_calibration_recipe_contract_requires_full_batch_path_quality_rank() -> None:
    meta = {
        "train_recipe": {
            "active_heads": ["direction", "path_quality", "bad_path"],
            "path_quality_rank_full_batch": True,
            "path_quality_rank_weight": 2.0,
            "path_quality_rank_margin": 0.25,
            "path_quality_rank_quantile": 0.25,
            "bad_path_quality_rank_weight": 2.0,
            "bad_path_quality_rank_margin": 0.25,
            "bad_path_quality_rank_quantile": 0.25,
        }
    }

    report = _path_calibration_recipe_contract(meta, {})

    assert report["decision"] == "PASS"
    assert report["path_quality_rank_full_batch"] is True


def test_path_calibration_recipe_contract_rejects_old_path_quality_recipe() -> None:
    meta = {
        "train_recipe": {
            "active_heads": ["direction", "path_quality", "bad_path"],
            "path_quality_rank_weight": 0.0,
            "bad_path_quality_rank_weight": 2.0,
            "bad_path_quality_rank_margin": 0.25,
            "bad_path_quality_rank_quantile": 0.25,
        }
    }

    report = _path_calibration_recipe_contract(meta, {})

    assert report["decision"] == "FAIL"
    assert any("path_quality_rank_weight" in failure for failure in report["failures"])
    assert any("path_quality_rank_full_batch" in failure for failure in report["failures"])


def test_head_contract_report_accepts_supported_declared_forward_outputs() -> None:
    target_audit = {
        "decision": "PASS",
        "target_head_contract": {
            "active_training_heads": ["direction", "path_quality", "tf_agreement"],
            "blocked_heads": ["hold_horizon"],
        },
    }
    capabilities = {
        "supported_heads": ["direction", "path_quality", "tf_agreement"],
        "declared_active_heads": ["direction", "path_quality", "tf_agreement"],
        "state_dict_heads": ["direction", "path_quality", "tf_agreement"],
    }
    split_reports = {
        "val": {
            "output_keys_seen": ["direction_logits", "path_quality", "tf_agreement_logit"],
            "output_shapes_seen": {
                "direction_logits": [[3]],
                "path_quality": [[1]],
                "tf_agreement_logit": [[1]],
            },
        },
        "test": {
            "output_keys_seen": ["direction_logits", "path_quality", "tf_agreement_logit"],
            "output_shapes_seen": {
                "direction_logits": [[3]],
                "path_quality": [[1]],
                "tf_agreement_logit": [[1]],
            },
        },
    }

    report = _head_contract_report(
        target_audit=target_audit,
        capabilities=capabilities,
        split_reports=split_reports,
    )

    assert report["decision"] == "PASS"
    assert report["failures"] == []


def test_head_contract_report_rejects_missing_active_output_and_blocked_head() -> None:
    target_audit = {
        "decision": "PASS",
        "target_head_contract": {
            "active_training_heads": ["direction", "path_quality", "tf_agreement"],
            "blocked_heads": ["hold_horizon"],
        },
    }
    capabilities = {
        "supported_heads": ["direction", "path_quality", "hold_horizon"],
        "declared_active_heads": ["direction", "path_quality"],
        "state_dict_heads": ["direction", "path_quality", "hold_horizon"],
    }
    split_reports = {
        "val": {
            "output_keys_seen": ["direction_logits", "path_quality", "hold_horizon_logit"],
            "output_shapes_seen": {
                "direction_logits": [[3]],
                "path_quality": [[2]],
                "hold_horizon_logit": [[1]],
            },
        },
    }

    report = _head_contract_report(
        target_audit=target_audit,
        capabilities=capabilities,
        split_reports=split_reports,
    )

    assert report["decision"] == "FAIL"
    assert any("missing active heads" in failure for failure in report["failures"])
    assert any("bundle contains blocked heads" in failure for failure in report["failures"])
    assert any("missing forward outputs" in failure for failure in report["failures"])
    assert any("wrong forward output shapes" in failure for failure in report["failures"])
    assert any("blocked heads emitted forward outputs" in failure for failure in report["failures"])


def test_head_contract_report_rejects_unapproved_extra_head() -> None:
    target_audit = {
        "decision": "PASS",
        "target_head_contract": {
            "active_training_heads": ["direction", "path_quality", "tf_agreement"],
            "blocked_heads": ["hold_horizon"],
        },
    }
    capabilities = {
        "supported_heads": ["direction", "path_quality", "tf_agreement", "experimental_alpha"],
        "declared_active_heads": ["direction", "path_quality", "tf_agreement", "experimental_alpha"],
        "state_dict_heads": ["direction", "path_quality", "tf_agreement", "experimental_alpha"],
    }
    split_reports = {
        "val": {
            "output_keys_seen": ["direction_logits", "path_quality", "tf_agreement_logit"],
            "output_shapes_seen": {
                "direction_logits": [[3]],
                "path_quality": [[1]],
                "tf_agreement_logit": [[1]],
            },
        },
    }

    report = _head_contract_report(
        target_audit=target_audit,
        capabilities=capabilities,
        split_reports=split_reports,
    )

    assert report["decision"] == "FAIL"
    assert any("unapproved heads" in failure for failure in report["failures"])


def test_pretrain_manifest_contract_accepts_exact_provenance(tmp_path) -> None:
    artifacts = {}
    for name in (
        "training_readiness",
        "feature_audit",
        "target_audit",
        "specialist_audit",
        "foundation_guardrails",
        "smoke_dataset_manifest",
        "worktree_hygiene",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"artifact": name}), encoding="utf-8")
        artifacts[name] = path
    artifact_fingerprints = {
        name: {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        for name, path in artifacts.items()
        if name != "training_readiness"
    }
    bundle_dir = tmp_path / "bundle"
    dataset_dir = tmp_path / "smoke_dataset"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    split_hashes = {
        split: {
            "source_manifest_sha256": "a" * 64,
            "out_parquet_sha256": "b" * 64,
            "out_manifest_sha256": "c" * 64,
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "schema_version": "entry_foundation_smoke_train_run_manifest_v1",
        "run_mode": "manifest_only",
        "out_bundle_dir": str(bundle_dir),
        "promotion_shadow_live_allowed": False,
        "trainer_started_by_manifest_writer": False,
        "inputs": {
            "smoke_dataset_dir": str(dataset_dir),
            "training_readiness_json": str(artifacts["training_readiness"]),
            "feature_audit_json": str(artifacts["feature_audit"]),
            "target_audit_json": str(artifacts["target_audit"]),
            "specialist_audit_json": str(artifacts["specialist_audit"]),
            "foundation_guardrails_json": str(artifacts["foundation_guardrails"]),
            "smoke_dataset_manifest": str(artifacts["smoke_dataset_manifest"]),
            "worktree_hygiene_json": str(artifacts["worktree_hygiene"]),
        },
        "artifact_sha256": {key: _sha256_file(path) for key, path in artifacts.items()},
        "preflight_contracts": {
            "training_readiness": {
                "foundation_contract_ready_for_smoke": True,
                "artifact_provenance_decision": "PASS",
                "artifact_fingerprints": artifact_fingerprints,
            },
            "feature_foundation": {
                "decision": "PASS",
                "foundation_objective_coverage_all_present": True,
                "foundation_objective_coverage": [{"objective": "hh_hl_lh_ll", "missing_count": 0}],
                "foundation_objective_liveness_all_live": True,
                "foundation_objective_liveness": [
                    {
                        "split": "val",
                        "objective": "hh_hl_lh_ll",
                        "missing_count": 0,
                        "nonfinite_count": 0,
                        "near_constant_count": 0,
                    }
                ],
                "foundation_source_field_liveness_all_live": True,
                "min_required_source_active_rate": 0.0001,
                "min_required_source_active_count": 1,
                "foundation_source_field_liveness": [
                    {
                        "split": "val",
                        "source_field": "snap.smc_bos_up",
                        "observed": True,
                        "nonfinite_count": 0,
                        "active_count": 12,
                        "active_rate": 0.1,
                        "near_constant": False,
                    }
                ],
                "foundation_source_fields_by_split": {"val": {"source_missing_count": 0}},
            },
            "target_foundation": {
                "decision": "PASS",
                "active_training_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "blocked_heads": list(EXPECTED_BLOCKED_HEADS),
            },
            "specialist_contract": {
                "architecture_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "architecture_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "trainable_specialists": REQUIRED_SPECIALISTS,
                "excluded_specialist_groups": {
                    "neutral_bridge_anchor": 7,
                    "price_action_candle_encoder": 3,
                },
                "specialist_model_contract_valid": True,
                "specialist_model_contract_failures": [],
                "specialist_model_contract": _specialist_model_contract_payload(),
                "foundation_objective_routing_all_present_and_expected": True,
                "foundation_objective_routing": [{"objective": "hh_hl_lh_ll", "missing_count": 0, "misrouted_count": 0}],
                "specialist_input_liveness_all_live": True,
                "specialist_input_liveness": [
                    {
                        "split": "val",
                        "specialist": "structure_swing_encoder",
                        "live_feature_count": 10,
                        "min_required_live_feature_count": 10,
                        "nonfinite_count": 0,
                    }
                ],
            },
            "smoke_dataset": {
                "schema_version": "entry_foundation_seq146_smoke_dataset_v1",
                "audit_provenance_schema_version": "entry_foundation_smoke_dataset_audit_provenance_v1",
                "audit_provenance_all_artifacts_present": True,
                "audit_provenance_all_artifact_hashes_present": True,
                "audit_provenance_artifacts": {
                    key: {
                        "path": str(artifacts[key]),
                        "sha256": _sha256_file(artifacts[key]),
                        "decision": "PASS",
                    }
                    for key in ("feature_audit", "target_audit", "specialist_audit")
                },
                "split_hashes": split_hashes,
            },
            "worktree_hygiene": {
                "decision": "BLOCKED_BY_DIRTY_GIT",
                "foundation_cleanup_critical_gate_review": {
                    "critical_gate_path_count": 12,
                    "ok_count": 12,
                    "missing_from_repo": [],
                    "dirty_missing_from_stage": [],
                },
            },
        },
    }
    path = tmp_path / "pretrain_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "PASS"
    assert report["failures"] == []
    assert report["specialist_required_training_set_exact"] is True
    assert report["specialist_trainable_set_exact"] is True
    assert report["specialist_model_contract_valid"] is True
    assert report["specialist_model_contract_set_exact"] is True
    assert report["specialist_model_contract_owned_objectives_match"] is True
    assert report["worktree_critical_gate_review_ok"] is True

    manifest["preflight_contracts"]["worktree_hygiene"]["foundation_cleanup_critical_gate_review"][
        "dirty_missing_from_stage"
    ] = ["scripts/entry_next_edge_control.sh"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert report["worktree_critical_gate_review_ok"] is False
    assert any("critical gate review is not complete" in failure for failure in report["failures"])
    manifest["preflight_contracts"]["worktree_hygiene"]["foundation_cleanup_critical_gate_review"][
        "dirty_missing_from_stage"
    ] = []

    manifest["preflight_contracts"]["specialist_contract"]["trainable_specialists"] = [
        *REQUIRED_SPECIALISTS,
        "price_action_candle_encoder",
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert report["specialist_required_training_set_exact"] is True
    assert report["specialist_trainable_set_exact"] is False
    assert any("trainable set is not exact" in failure for failure in report["failures"])

    manifest["preflight_contracts"]["specialist_contract"]["trainable_specialists"] = REQUIRED_SPECIALISTS
    manifest["preflight_contracts"]["specialist_contract"]["specialist_model_contract_valid"] = False
    manifest["preflight_contracts"]["specialist_contract"]["specialist_model_contract_failures"] = [
        "forced invalid contract"
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert report["specialist_model_contract_valid"] is False
    assert any("specialist model contract" in failure for failure in report["failures"])


def test_pretrain_manifest_contract_rejects_hash_and_routing_failures(tmp_path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    dataset_dir = tmp_path / "smoke_dataset"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    manifest = {
        "schema_version": "entry_foundation_smoke_train_run_manifest_v1",
        "out_bundle_dir": str(bundle_dir),
        "promotion_shadow_live_allowed": False,
        "trainer_started_by_manifest_writer": False,
        "inputs": {
            "smoke_dataset_dir": str(dataset_dir),
            "training_readiness_json": str(artifact),
            "feature_audit_json": str(artifact),
            "target_audit_json": str(artifact),
            "specialist_audit_json": str(artifact),
            "foundation_guardrails_json": str(artifact),
            "smoke_dataset_manifest": str(artifact),
            "worktree_hygiene_json": str(artifact),
        },
        "artifact_sha256": {
            key: "bad"
            for key in (
                "training_readiness",
                "feature_audit",
                "target_audit",
                "specialist_audit",
                "foundation_guardrails",
                "smoke_dataset_manifest",
                "worktree_hygiene",
            )
        },
        "preflight_contracts": {
            "training_readiness": {
                "foundation_contract_ready_for_smoke": True,
                "artifact_provenance_decision": "FAIL",
                "artifact_fingerprints": {},
            },
            "feature_foundation": {
                "decision": "PASS",
                "foundation_objective_coverage_all_present": True,
                "foundation_objective_coverage": [{"objective": "hh_hl_lh_ll", "missing_count": 0}],
                "foundation_objective_liveness_all_live": True,
                "foundation_objective_liveness": [
                    {
                        "split": "val",
                        "objective": "hh_hl_lh_ll",
                        "missing_count": 0,
                        "nonfinite_count": 0,
                        "near_constant_count": 0,
                    }
                ],
                "foundation_source_field_liveness_all_live": True,
                "min_required_source_active_rate": 0.0001,
                "min_required_source_active_count": 1,
                "foundation_source_field_liveness": [
                    {
                        "split": "val",
                        "source_field": "snap.smc_bos_up",
                        "observed": True,
                        "nonfinite_count": 0,
                        "active_count": 12,
                        "active_rate": 0.1,
                        "near_constant": False,
                    }
                ],
                "foundation_source_fields_by_split": {"val": {"source_missing_count": 0}},
            },
            "target_foundation": {
                "decision": "PASS",
                "active_training_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "blocked_heads": list(EXPECTED_BLOCKED_HEADS),
            },
            "specialist_contract": {
                "architecture_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "architecture_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "trainable_specialists": REQUIRED_SPECIALISTS,
                "specialist_model_contract_valid": True,
                "specialist_model_contract_failures": [],
                "specialist_model_contract": _specialist_model_contract_payload(),
                "foundation_objective_routing_all_present_and_expected": True,
                "foundation_objective_routing": [{"objective": "hh_hl_lh_ll", "missing_count": 0, "misrouted_count": 1}],
                "specialist_input_liveness_all_live": True,
                "specialist_input_liveness": [
                    {
                        "split": "val",
                        "specialist": "structure_swing_encoder",
                        "live_feature_count": 10,
                        "min_required_live_feature_count": 10,
                        "nonfinite_count": 0,
                    }
                ],
            },
            "worktree_hygiene": {
                "decision": "BLOCKED_BY_DIRTY_GIT",
                "foundation_cleanup_critical_gate_review": {
                    "critical_gate_path_count": 12,
                    "ok_count": 12,
                    "missing_from_repo": [],
                    "dirty_missing_from_stage": [],
                },
            },
        },
    }
    path = tmp_path / "pretrain_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert any("artifact hash mismatch" in failure for failure in report["failures"])
    assert any("specialist objective not exactly routed" in failure for failure in report["failures"])


def test_pretrain_manifest_contract_accepts_candidate_train_manifest(tmp_path) -> None:
    artifacts = {}
    for name in ("candidate_readiness", "smoke_bundle_audit", "specialist_audit"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"artifact": name}), encoding="utf-8")
        artifacts[name] = path
    candidate_fingerprints = {
        name: {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        for name, path in artifacts.items()
        if name in {"smoke_bundle_audit", "specialist_audit"}
    }
    bundle_dir = tmp_path / "candidate_bundle"
    dataset_dir = tmp_path / "foundation_dataset"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    manifest = {
        "schema_version": "entry_foundation_candidate_train_run_manifest_v1",
        "out_bundle_dir": str(bundle_dir),
        "promotion_shadow_live_allowed": False,
        "trainer_started_by_manifest_writer": False,
        "inputs": {
            "candidate_dataset_dir": str(dataset_dir),
            "candidate_readiness_json": str(artifacts["candidate_readiness"]),
            "smoke_bundle_audit_json": str(artifacts["smoke_bundle_audit"]),
            "specialist_audit_json": str(artifacts["specialist_audit"]),
        },
        "artifact_sha256": {key: _sha256_file(path) for key, path in artifacts.items()},
        "preflight_contracts": {
            "candidate_readiness": {
                "decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
                "artifact_provenance_decision": "PASS",
                "artifact_fingerprints": candidate_fingerprints,
            },
            "smoke_edge_audit": {
                "decision": "PASS",
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "specialist_groups": REQUIRED_SPECIALISTS,
                "pretrain_manifest_contract_decision": "PASS",
                "feature_objective_coverage_all_present": True,
                "feature_objective_liveness_all_live": True,
                "feature_source_field_liveness_all_live": True,
                "specialist_active_heads_match_target": True,
                "specialist_blocked_heads_match_target": True,
                "specialist_objective_routing_all_present_and_expected": True,
                "specialist_model_contract_valid": True,
                "specialist_model_contract_set_exact": True,
                "specialist_model_contract_owned_objectives_match": True,
                "smoke_dataset_audit_provenance_all_artifacts_present": True,
                "smoke_dataset_audit_provenance_all_artifact_hashes_present": True,
                "worktree_critical_gate_review_ok": True,
            },
            "target_foundation": {
                "decision": "PASS",
                "active_training_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "blocked_heads": list(EXPECTED_BLOCKED_HEADS),
            },
            "specialist_contract": {
                "architecture_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "architecture_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "trainable_specialists": REQUIRED_SPECIALISTS,
                "excluded_specialist_groups": {
                    "neutral_bridge_anchor": 7,
                    "price_action_candle_encoder": 3,
                },
                "specialist_model_contract_valid": True,
                "specialist_model_contract_failures": [],
                "specialist_model_contract": _specialist_model_contract_payload(),
                "foundation_objective_routing_all_present_and_expected": True,
                "specialist_input_liveness_all_live": True,
                "specialist_input_liveness": [
                    {
                        "split": "val",
                        "specialist": "structure_swing_encoder",
                        "live_feature_count": 10,
                        "min_required_live_feature_count": 10,
                        "nonfinite_count": 0,
                    }
                ],
            },
        },
    }
    path = tmp_path / "candidate_pretrain_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "PASS"
    assert report["failures"] == []
    assert report["specialist_required_training_set_exact"] is True
    assert report["specialist_trainable_set_exact"] is True
    assert report["specialist_model_contract_valid"] is True
    assert report["specialist_model_contract_set_exact"] is True
    assert report["specialist_model_contract_owned_objectives_match"] is True
    assert report["smoke_edge_required_specialists_exact"] is True
    assert report["smoke_edge_specialist_groups_exact"] is True
    assert report["smoke_edge_specialist_model_contract_valid"] is True
    assert report["smoke_edge_specialist_model_contract_set_exact"] is True
    assert report["smoke_edge_specialist_model_contract_owned_objectives_match"] is True
    assert report["smoke_dataset_audit_provenance_all_artifacts_present"] is True
    assert report["smoke_dataset_audit_provenance_all_artifact_hashes_present"] is True
    assert report["smoke_edge_worktree_critical_gate_review_ok"] is True

    manifest["preflight_contracts"]["smoke_edge_audit"]["worktree_critical_gate_review_ok"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert report["smoke_edge_worktree_critical_gate_review_ok"] is False
    assert any("worktree critical gate review" in failure for failure in report["failures"])

    manifest["preflight_contracts"]["smoke_edge_audit"]["worktree_critical_gate_review_ok"] = True
    manifest["preflight_contracts"]["specialist_contract"]["trainable_specialists"] = [
        *REQUIRED_SPECIALISTS,
        "price_action_candle_encoder",
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert report["specialist_required_training_set_exact"] is True
    assert report["specialist_trainable_set_exact"] is False
    assert any("trainable set is not exact" in failure for failure in report["failures"])

    manifest["preflight_contracts"]["specialist_contract"]["trainable_specialists"] = REQUIRED_SPECIALISTS
    manifest["preflight_contracts"]["smoke_edge_audit"]["specialist_groups"] = [
        *REQUIRED_SPECIALISTS,
        "price_action_candle_encoder",
    ]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert report["smoke_edge_required_specialists_exact"] is True
    assert report["smoke_edge_specialist_groups_exact"] is False
    assert any("specialist groups were not exact" in failure for failure in report["failures"])


def test_pretrain_manifest_contract_accepts_candidate_latest_refresh_with_timestamped_match(tmp_path) -> None:
    readiness_dir = tmp_path / "candidate_readiness_reports"
    readiness_dir.mkdir()
    candidate_readiness_timestamped = readiness_dir / "ENTRY_CANDIDATE_READINESS_20260630T020634Z.json"
    candidate_readiness_latest = readiness_dir / "ENTRY_CANDIDATE_READINESS_latest.json"
    candidate_readiness_timestamped.write_text(json.dumps({"artifact": "candidate_readiness"}), encoding="utf-8")
    candidate_readiness_latest.write_text(json.dumps({"artifact": "candidate_readiness_refreshed"}), encoding="utf-8")

    artifacts = {"candidate_readiness": candidate_readiness_latest}
    timestamped_artifacts = {}
    for name in ("smoke_bundle_audit", "specialist_audit"):
        artifact_dir = tmp_path / f"{name}_reports"
        artifact_dir.mkdir()
        timestamped = artifact_dir / f"{name}_20260630T020634Z.json"
        latest = artifact_dir / f"{name}_latest.json"
        payload = {"artifact": name}
        timestamped.write_text(json.dumps(payload), encoding="utf-8")
        latest.write_text(json.dumps(payload), encoding="utf-8")
        artifacts[name] = latest
        timestamped_artifacts[name] = timestamped
    candidate_fingerprints = {
        name: {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        for name, path in artifacts.items()
        if name in {"smoke_bundle_audit", "specialist_audit"}
    }
    bundle_dir = tmp_path / "candidate_bundle"
    dataset_dir = tmp_path / "foundation_dataset"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    manifest = {
        "schema_version": "entry_foundation_candidate_train_run_manifest_v1",
        "out_bundle_dir": str(bundle_dir),
        "promotion_shadow_live_allowed": False,
        "trainer_started_by_manifest_writer": False,
        "inputs": {
            "candidate_dataset_dir": str(dataset_dir),
            "candidate_readiness_json": str(candidate_readiness_latest),
            "smoke_bundle_audit_json": str(timestamped_artifacts["smoke_bundle_audit"]),
            "specialist_audit_json": str(timestamped_artifacts["specialist_audit"]),
        },
        "artifact_sha256": {
            "candidate_readiness": _sha256_file(candidate_readiness_timestamped),
            "smoke_bundle_audit": _sha256_file(timestamped_artifacts["smoke_bundle_audit"]),
            "specialist_audit": _sha256_file(timestamped_artifacts["specialist_audit"]),
        },
        "preflight_contracts": {
            "candidate_readiness": {
                "decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
                "artifact_provenance_decision": "PASS",
                "artifact_fingerprints": candidate_fingerprints,
            },
            "smoke_edge_audit": {
                "decision": "PASS",
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "specialist_groups": REQUIRED_SPECIALISTS,
                "pretrain_manifest_contract_decision": "PASS",
                "feature_objective_coverage_all_present": True,
                "feature_objective_liveness_all_live": True,
                "feature_source_field_liveness_all_live": True,
                "specialist_active_heads_match_target": True,
                "specialist_blocked_heads_match_target": True,
                "specialist_objective_routing_all_present_and_expected": True,
                "specialist_model_contract_valid": True,
                "specialist_model_contract_set_exact": True,
                "specialist_model_contract_owned_objectives_match": True,
                "smoke_dataset_audit_provenance_all_artifacts_present": True,
                "smoke_dataset_audit_provenance_all_artifact_hashes_present": True,
                "worktree_critical_gate_review_ok": True,
            },
            "target_foundation": {
                "decision": "PASS",
                "active_training_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "blocked_heads": list(EXPECTED_BLOCKED_HEADS),
            },
            "specialist_contract": {
                "architecture_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "architecture_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "trainable_specialists": REQUIRED_SPECIALISTS,
                "specialist_model_contract_valid": True,
                "specialist_model_contract_failures": [],
                "specialist_model_contract": _specialist_model_contract_payload(),
                "foundation_objective_routing_all_present_and_expected": True,
                "specialist_input_liveness_all_live": True,
            },
        },
    }
    path = tmp_path / "candidate_pretrain_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "PASS"
    candidate_check = report["artifact_hash_checks"]["candidate_readiness"]
    assert candidate_check["path"] == str(candidate_readiness_latest.resolve())
    assert candidate_check["resolved_path"] == str(candidate_readiness_timestamped.resolve())
    assert candidate_check["resolution"] == "timestamped_sibling_by_expected_sha256"
    assert candidate_check["mutable_latest_observed"] == _sha256_file(candidate_readiness_latest)


def test_pretrain_manifest_contract_exposes_candidate_variant_from_contract_mode(tmp_path) -> None:
    bundle_dir = tmp_path / "candidate_bundle"
    dataset_dir = tmp_path / "candidate_dataset"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    manifest = {
        "schema_version": "entry_foundation_candidate_train_run_manifest_v1",
        "specialist_contract_mode": "smart_seq520_candidate",
        "out_bundle_dir": str(bundle_dir),
        "promotion_shadow_live_allowed": False,
        "trainer_started_by_manifest_writer": False,
        "inputs": {"candidate_dataset_dir": str(dataset_dir)},
        "artifact_sha256": {},
        "preflight_contracts": {},
    }
    path = tmp_path / "candidate_pretrain_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["manifest_variant"] == "smart_seq520_candidate"
    assert report["candidate_variant"] == "smart_seq520_candidate"


def test_pretrain_manifest_contract_rejects_candidate_without_smoke_dataset_provenance(tmp_path) -> None:
    artifacts = {}
    for name in ("candidate_readiness", "smoke_bundle_audit", "specialist_audit"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"artifact": name}), encoding="utf-8")
        artifacts[name] = path
    candidate_fingerprints = {
        name: {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        for name, path in artifacts.items()
        if name in {"smoke_bundle_audit", "specialist_audit"}
    }
    bundle_dir = tmp_path / "candidate_bundle"
    dataset_dir = tmp_path / "foundation_dataset"
    bundle_dir.mkdir()
    dataset_dir.mkdir()
    manifest = {
        "schema_version": "entry_foundation_candidate_train_run_manifest_v1",
        "out_bundle_dir": str(bundle_dir),
        "promotion_shadow_live_allowed": False,
        "trainer_started_by_manifest_writer": False,
        "inputs": {
            "candidate_dataset_dir": str(dataset_dir),
            "candidate_readiness_json": str(artifacts["candidate_readiness"]),
            "smoke_bundle_audit_json": str(artifacts["smoke_bundle_audit"]),
            "specialist_audit_json": str(artifacts["specialist_audit"]),
        },
        "artifact_sha256": {key: _sha256_file(path) for key, path in artifacts.items()},
        "preflight_contracts": {
            "candidate_readiness": {
                "decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
                "artifact_provenance_decision": "PASS",
                "artifact_fingerprints": candidate_fingerprints,
            },
            "smoke_edge_audit": {
                "decision": "PASS",
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "specialist_groups": REQUIRED_SPECIALISTS,
                "pretrain_manifest_contract_decision": "PASS",
                "feature_objective_coverage_all_present": True,
                "feature_objective_liveness_all_live": True,
                "feature_source_field_liveness_all_live": True,
                "specialist_active_heads_match_target": True,
                "specialist_blocked_heads_match_target": True,
                "specialist_objective_routing_all_present_and_expected": True,
                "specialist_model_contract_valid": True,
                "specialist_model_contract_set_exact": True,
                "specialist_model_contract_owned_objectives_match": True,
                "smoke_dataset_audit_provenance_all_artifacts_present": False,
                "smoke_dataset_audit_provenance_all_artifact_hashes_present": False,
                "worktree_critical_gate_review_ok": True,
            },
            "target_foundation": {
                "decision": "PASS",
                "active_training_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "blocked_heads": list(EXPECTED_BLOCKED_HEADS),
            },
            "specialist_contract": {
                "architecture_active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
                "architecture_blocked_heads": list(EXPECTED_BLOCKED_HEADS),
                "required_training_specialists": REQUIRED_SPECIALISTS,
                "trainable_specialists": REQUIRED_SPECIALISTS,
                "specialist_model_contract_valid": True,
                "specialist_model_contract_failures": [],
                "specialist_model_contract": _specialist_model_contract_payload(),
                "foundation_objective_routing_all_present_and_expected": True,
                "specialist_input_liveness_all_live": True,
            },
        },
    }
    path = tmp_path / "candidate_pretrain_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    report = _pretrain_manifest_contract_report(path, expected_bundle_dir=bundle_dir, expected_dataset_dir=dataset_dir)

    assert report["decision"] == "FAIL"
    assert any("smoke-dataset audit artifacts" in failure for failure in report["failures"])
    assert any("smoke-dataset audit artifact hashes" in failure for failure in report["failures"])
