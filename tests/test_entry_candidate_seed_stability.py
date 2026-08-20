from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
    for seed in (11, 12, 13, 14, 15):
        bundle = tmp_path / f"bundle_{seed}.json"
        bundle.write_text(json.dumps({"seed": seed}), encoding="utf-8")
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
