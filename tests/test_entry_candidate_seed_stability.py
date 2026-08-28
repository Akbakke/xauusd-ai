from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.verify_entry_candidate_seed_stability_v1 import (
    _classify_raw_q_regime,
    run,
)


def test_seed_regime_classification_rejects_flat_and_side_collapse() -> None:
    assert _classify_raw_q_regime(np.array([2] * 100))["regime"] == "flat_drift"
    assert _classify_raw_q_regime(np.array([0] * 100))["regime"] == "long_side_collapse"
    assert _classify_raw_q_regime(np.array([1] * 100))["regime"] == "short_side_collapse"
    assert _classify_raw_q_regime(np.array([0, 1, 2, 0, 1]))["regime"] == "mixed_raw_q_actions"


def test_five_seed_gate_requires_same_substrate_and_mixed_actions(tmp_path) -> None:
    reports: list[str] = []
    contract = {"contract": "same"}
    recipe_identity = {
        "git_commit": "a" * 40,
        "execution_tier": "canonical",
        "seq_len": 8,
        "dropout": 0.1,
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.001,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0,
        "grad_clip_norm": 1.0,
        "weight_decay": 0.0,
        "model_architecture_schema_version": "test",
        "model_output_schema_version": "test",
        "model_native_signal_contract": contract,
        "model_native_training_objective": contract,
        "recipe_source_provenance": contract,
        "aux_head_target_contract": contract,
        "m1_feature_surface_binding": contract,
        "sequence_source_reconstruction": contract,
        "prefreeze_test_seal_lineage": contract,
        "input_normalization": contract,
        "multi_tf": contract,
        "specialist_fusion": contract,
        "context_specialist_routing": contract,
    }
    for seed in (11, 12, 13, 14, 15):
        bundle = tmp_path / f"bundle_{seed}.json"
        bundle.write_text(
            json.dumps(
                {
                    **recipe_identity,
                    "seed": seed,
                    "run_lineage": {
                        "training_run_id": f"run-{seed}",
                        "dataset_run_id": "dataset",
                    },
                }
            ),
            encoding="utf-8",
        )
        bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
        prediction = tmp_path / f"predictions_{seed}.parquet"
        pd.DataFrame(
            {"split": ["val"] * 6, "pred_direction": [0, 1, 2, 0, 1, 2]}
        ).to_parquet(prediction, index=False)
        prediction_sha = hashlib.sha256(prediction.read_bytes()).hexdigest()
        report = tmp_path / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_20260820T12000{seed % 10}000000Z.json"
        report.write_text(
            json.dumps(
                {
                    "decision": "PASS",
                    "failures": [],
                    "evidence_stage": "pre_calibration",
                    "outcome_economics": "gross_spread_inclusive_research_only",
                    "production_authority_ready": False,
                    "edge_claim_allowed": False,
                    "preregistered_selective_edge": {
                        "coverage_grid": [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
                        "decision": "FAIL",
                    },
                    "bundle_metadata_path": str(bundle),
                    "bundle_metadata_sha256": bundle_sha,
                    "predictions_path": str(prediction),
                    "prediction_evidence": {
                        "path": str(prediction),
                        "sha256": prediction_sha,
                    },
                    "dataset_dir": "/tmp/same_dataset",
                    "model_native_signal_contract": contract,
                    "dataset_signal_contract": contract,
                    "direction_decision_contract": contract,
                }
            ),
            encoding="utf-8",
        )
        reports.append(str(report))
    result = run(
        SimpleNamespace(
            selective_edge_report=reports,
            out_dir=str(tmp_path / "events"),
            quiet=True,
        )
    )
    assert result["decision"] == "PASS"
    assert result["seeds"] == [11, 12, 13, 14, 15]


def test_seed_gate_rejects_a_different_training_recipe(tmp_path) -> None:
    reports: list[str] = []
    for seed in (1, 2, 3, 4, 5):
        bundle = tmp_path / f"bundle_{seed}.json"
        metadata = {
            "seed": seed,
            "git_commit": "a" * 40,
            "execution_tier": "canonical",
            "seq_len": 8,
            "dropout": 0.1,
            "batch_size": 2,
            "epochs": 2 if seed == 5 else 1,
            "lr": 0.001,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
            "grad_clip_norm": 1.0,
            "weight_decay": 0.0,
            "model_architecture_schema_version": "test",
            "model_output_schema_version": "test",
            "model_native_signal_contract": {"same": 1},
            "model_native_training_objective": {"same": 1},
            "recipe_source_provenance": {"same": 1},
            "aux_head_target_contract": {"same": 1},
            "m1_feature_surface_binding": {"same": 1},
            "sequence_source_reconstruction": {"same": 1},
            "prefreeze_test_seal_lineage": {"same": 1},
            "input_normalization": {"same": 1},
            "multi_tf": {"same": 1},
            "specialist_fusion": {"same": 1},
            "context_specialist_routing": {"same": 1},
            "run_lineage": {
                "training_run_id": f"run-{seed}",
                "dataset_run_id": "dataset",
            },
        }
        bundle.write_text(json.dumps(metadata), encoding="utf-8")
        prediction = tmp_path / f"predictions_{seed}.parquet"
        pd.DataFrame({"split": ["val"] * 3, "pred_direction": [0, 1, 2]}).to_parquet(prediction, index=False)
        prediction_sha = hashlib.sha256(prediction.read_bytes()).hexdigest()
        report = tmp_path / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_20260821T12000{seed}000000Z.json"
        report.write_text(json.dumps({
            "decision": "PASS", "failures": [], "evidence_stage": "pre_calibration",
            "outcome_economics": "gross_spread_inclusive_research_only",
            "production_authority_ready": False, "edge_claim_allowed": False,
            "preregistered_selective_edge": {"coverage_grid": [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01], "decision": "FAIL"},
            "bundle_metadata_path": str(bundle),
            "bundle_metadata_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "predictions_path": str(prediction),
            "prediction_evidence": {"path": str(prediction), "sha256": prediction_sha},
            "dataset_dir": "/tmp/same_dataset", "model_native_signal_contract": {"same": 1},
            "dataset_signal_contract": {"same": 1}, "direction_decision_contract": {"same": 1},
        }), encoding="utf-8")
        reports.append(str(report))
    with pytest.raises(RuntimeError, match="SHARED_SAME_RECIPE_IDENTITY_MISMATCH"):
        run(SimpleNamespace(selective_edge_report=reports, out_dir=str(tmp_path / "events"), quiet=True))
