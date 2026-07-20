import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
    build_prediction_evidence_declaration,
    parquet_schema_descriptor,
    parquet_schema_sha256,
    resolve_and_validate_prediction_evidence,
    sha256_file,
)
from tests.model_native_turning_point_support import (
    turning_point_prediction_columns,
)
from tests.model_native_offline_rl_support import offline_rl_prediction_columns


STAMP = "20260716T120000123456Z"


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["val", "test"],
            "model": ["candidate", "candidate"],
            "time": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"], utc=True
            ),
            "y_direction": [0, 2],
            "pred_direction": [0, 2],
            "p_long": [0.7, 0.1],
            "p_short": [0.1, 0.1],
            "p_flat": [0.2, 0.8],
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 2,
            "public_trade_probability": [0.7 / 0.9, 0.1 / 0.9],
            "public_flat_probability": [0.2 / 0.9, 0.8 / 0.9],
            "public_trade_flat_margin": [np.log(0.7 / 0.2), np.log(0.1 / 0.8)],
            "public_trade_flat_hard_decision": [0, 1],
            "direction_logits": [
                [np.log(0.7), np.log(0.1), np.log(0.2)],
                [np.log(0.1), np.log(0.1), np.log(0.8)],
            ],
            "public_trade_flat_decision_logits": [
                [np.log(0.7), np.log(0.2)],
                [np.log(0.1), np.log(0.8)],
            ],
            **turning_point_prediction_columns(2),
            **offline_rl_prediction_columns(2),
        }
    )


def _event(tmp_path: Path) -> dict:
    bundle = tmp_path / "bundle"
    dataset = tmp_path / "dataset"
    out = tmp_path / "out"
    bundle.mkdir()
    dataset.mkdir()
    out.mkdir()
    metadata = {
        "state_dict_sha256": "a" * 64,
        "direction_decision_contract": model_direction_decision_contract_metadata(),
    }
    (bundle / "bundle_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions = out / f"selective_edge_predictions_{STAMP}.parquet"
    atomic_write_parquet_immutable(_predictions(), predictions)
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions,
        bundle_dir=bundle,
        bundle_metadata=metadata,
        requested_splits=["val", "test"],
    )
    report_path = out / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{STAMP}.json"
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": "2026-07-16T12:00:00.123456+00:00",
        "decision": "PASS",
        "failures": [],
        "bundle_dir": str(bundle.resolve()),
        "dataset_dir": str(dataset.resolve()),
        "splits": ["test", "val"],
        "models": ["candidate"],
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "prediction_evidence": evidence,
        "predictions_path": str(predictions.resolve()),
        "bundle_metadata_sha256": evidence["bundle_metadata_sha256"],
        "model_state_dict_sha256": evidence["model_state_dict_sha256"],
        "json_path": str(report_path.resolve()),
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "bundle": bundle,
        "dataset": dataset,
        "out": out,
        "predictions": predictions,
        "untimestamped": out / "selective_edge_predictions.parquet",
        "report_path": report_path,
        "report": report,
    }


def _rewrite_report(event: dict, report: dict) -> None:
    event["report_path"].write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")


def test_authoritative_event_requires_explicit_matching_report(tmp_path: Path) -> None:
    event = _event(tmp_path)

    direct, _, direct_evidence = resolve_and_validate_prediction_evidence(
        event["predictions"],
        prediction_report_path=event["report_path"],
        bundle_dir=event["bundle"],
        dataset_dir=event["dataset"],
        expected_split="test",
        expected_model="candidate",
    )

    assert direct == event["predictions"].resolve()
    assert direct_evidence == event["report"]["prediction_evidence"]


def test_authorizing_consumer_rejects_untimestamped_prediction_path(tmp_path: Path) -> None:
    event = _event(tmp_path)

    with pytest.raises(RuntimeError, match="not a timestamped authoritative"):
        resolve_and_validate_prediction_evidence(
            event["untimestamped"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )


def test_authorizing_consumer_rejects_mismatched_explicit_report(tmp_path: Path) -> None:
    event = _event(tmp_path)
    wrong = event["out"] / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120001123456Z.json"

    with pytest.raises(RuntimeError, match="does not match"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=wrong,
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )


def test_authorizing_consumer_rejects_older_pass_when_newer_event_exists(
    tmp_path: Path,
) -> None:
    event = _event(tmp_path)
    newer = event["out"] / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120001123456Z.json"
    newer.write_text(
        json.dumps(
            {
                "schema_version": "entry_candidate_selective_edge_v1",
                "created_utc": "2026-07-16T12:00:01.123456+00:00",
                "decision": "FAIL",
                "failures": ["newer evidence revoked the old PASS"],
                "json_path": str(newer.resolve()),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not the newest"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )


def test_immutable_prediction_event_refuses_overwrite(tmp_path: Path) -> None:
    event = _event(tmp_path)

    with pytest.raises(RuntimeError, match="already exists"):
        atomic_write_parquet_immutable(_predictions(), event["predictions"])


def test_consumer_rejects_prediction_parquet_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    tampered = _predictions()
    tampered.loc[0, "pred_direction"] = 1
    tampered.to_parquet(event["predictions"], index=False)

    with pytest.raises(RuntimeError, match="parquet SHA-256 mismatch"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )


def test_consumer_rejects_declared_row_or_schema_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    report = dict(event["report"])
    report["prediction_evidence"] = dict(report["prediction_evidence"])
    report["prediction_evidence"]["rows"] = 999
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="row-count mismatch"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )

    report["prediction_evidence"]["rows"] = 2
    report["prediction_evidence"]["columns"] = report["prediction_evidence"]["columns"][:-1]
    _rewrite_report(event, report)
    with pytest.raises(RuntimeError, match="column schema mismatch"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )


def test_consumer_rejects_mode_even_when_attacker_rehashes_parquet(tmp_path: Path) -> None:
    event = _event(tmp_path)
    tampered = _predictions()
    tampered["selection_score_mode"] = "expected_utility"
    tampered.to_parquet(event["predictions"], index=False)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    descriptor = parquet_schema_descriptor(event["predictions"])
    evidence.update(
        {
            "sha256": sha256_file(event["predictions"]),
            "parquet_schema": descriptor,
            "parquet_schema_sha256": parquet_schema_sha256(descriptor),
        }
    )
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="direction mode mismatch"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )


def test_consumer_rejects_report_direction_contract_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    report = dict(event["report"])
    report["direction_decision_contract"] = dict(report["direction_decision_contract"])
    report["direction_decision_contract"]["runtime_direction_thresholds_allowed"] = True
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="direction_decision_contract mismatch"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )


def test_consumer_rejects_bundle_metadata_hash_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    metadata_path = event["bundle"] / "bundle_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["unbound_change"] = True
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="bundle metadata SHA-256 mismatch"):
        resolve_and_validate_prediction_evidence(
            event["predictions"],
            prediction_report_path=event["report_path"],
            bundle_dir=event["bundle"],
            dataset_dir=event["dataset"],
        )
