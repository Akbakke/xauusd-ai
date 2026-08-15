import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_REQUIRED_FIELDS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    PREDICTION_EVIDENCE_SCHEMA_VERSION,
    PRE_CALIBRATION_EVIDENCE_STAGE,
    RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
    RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION,
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
from tests.model_native_sizing_support import runtime_head_prediction_columns


STAMP = "20260716T120000123456Z"
# Row 0 is a unique LONG argmax, row 1 a unique FLAT argmax, in raw bps.
ENTRY_ACTION_Q_BPS = [[2.0, -1.0, 0.0], [-1.0, -2.0, 1.5]]


def _entry_q_margin(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[-1] - ordered[-2]


def _predictions(splits: tuple[str, ...] = ("val",)) -> pd.DataFrame:
    entry_q = [list(row) for row in ENTRY_ACTION_Q_BPS]
    directions = [int(np.argmax(np.asarray(row))) for row in entry_q]
    frame = pd.DataFrame(
        {
            "split": ["val", "test"],
            "model": ["candidate", "candidate"],
            "time": pd.to_datetime(
                ["2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"], utc=True
            ),
            "pred_direction": directions,
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 2,
            "entry_action_q_bps": entry_q,
            "entry_action_q_margin_bps": [_entry_q_margin(row) for row in entry_q],
            "model_direction_index": directions,
            "session_id": [2, 3],
            "session": ["OVERLAP", "US"],
            "position_size_logit": [-0.1, 0.4],
            "position_size_pred": [
                float(1.0 / (1.0 + np.exp(0.1))),
                float(1.0 / (1.0 + np.exp(-0.4))),
            ],
            **turning_point_prediction_columns(2),
        }
    )
    return frame.loc[frame["split"].isin(splits)].reset_index(drop=True)


def _event(
    tmp_path: Path,
    *,
    evidence_stage: str = PRE_CALIBRATION_EVIDENCE_STAGE,
) -> dict:
    stage_splits = {
        PRE_CALIBRATION_EVIDENCE_STAGE: ("val",),
        RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE: ("test",),
    }
    splits = stage_splits[evidence_stage]
    runtime_head = evidence_stage == RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    frame = _predictions(splits)
    if runtime_head:
        runtime_columns = runtime_head_prediction_columns(frame)
        heads = [
            json.loads(payload)
            for payload in runtime_columns["runtime_head_evidence_json"]
        ]
        # Every parquet column that is also a runtime-head field is a
        # *duplicate* of the sealed envelope and must be bit-identical to it,
        # so the fixture republishes the head's own values rather than a
        # second, independently authored copy.
        for name in frame.columns:
            if name in MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_REQUIRED_FIELDS:
                frame[name] = [head[name] for head in heads]
        for name, values in runtime_columns.items():
            frame[name] = values
    atomic_write_parquet_immutable(frame, predictions)
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions,
        bundle_dir=bundle,
        bundle_metadata=metadata,
        evidence_stage=evidence_stage,
        requested_splits=list(splits),
    )
    report_path = out / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{STAMP}.json"
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": "2026-07-16T12:00:00.123456+00:00",
        "decision": "PASS",
        "failures": [],
        "evidence_stage": evidence_stage,
        "bundle_dir": str(bundle.resolve()),
        "dataset_dir": str(dataset.resolve()),
        "splits": list(splits),
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
        "evidence_stage": evidence_stage,
        "expected_splits": splits,
        "expected_sha256": evidence["sha256"],
    }


def _rewrite_report(event: dict, report: dict) -> None:
    event["report_path"].write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")


def _resolve(event: dict, **overrides):
    requested_path = overrides.pop("requested_path", event["predictions"])
    kwargs = {
        "expected_sha256": event["expected_sha256"],
        "prediction_report_path": event["report_path"],
        "bundle_dir": event["bundle"],
        "dataset_dir": event["dataset"],
        "expected_stage": event["evidence_stage"],
        "expected_splits": event["expected_splits"],
    }
    kwargs.update(overrides)
    return resolve_and_validate_prediction_evidence(requested_path, **kwargs)


def test_pre_calibration_requires_exact_val_and_is_non_authoritative(
    tmp_path: Path,
) -> None:
    event = _event(tmp_path)

    direct, _, direct_evidence = _resolve(
        event,
        expected_model="candidate",
    )

    assert direct == event["predictions"].resolve()
    assert direct_evidence == event["report"]["prediction_evidence"]
    assert direct_evidence["schema_version"] == PREDICTION_EVIDENCE_SCHEMA_VERSION
    assert direct_evidence["evidence_stage"] == PRE_CALIBRATION_EVIDENCE_STAGE
    assert direct_evidence["splits"] == ["val"]
    assert direct_evidence["authoritative"] is False
    assert direct_evidence["runtime_head_evidence_authoritative"] is False


def test_runtime_authoritative_requires_exact_test_and_head_envelope(
    tmp_path: Path,
) -> None:
    runtime = _event(
        tmp_path,
        evidence_stage=RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
    )
    _, _, evidence = _resolve(runtime, expected_model="candidate")

    assert evidence["schema_version"] == RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION
    assert evidence["evidence_stage"] == RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE
    assert evidence["splits"] == ["test"]
    assert evidence["authoritative"] is True
    assert evidence["runtime_head_evidence_authoritative"] is True


def test_resolver_signature_requires_sha_stage_and_exact_splits() -> None:
    parameters = inspect.signature(
        resolve_and_validate_prediction_evidence
    ).parameters

    for name in ("expected_sha256", "expected_stage", "expected_splits"):
        assert parameters[name].default is inspect.Parameter.empty
    assert "expected_split" not in parameters
    assert "require_runtime_head_evidence" not in parameters


@pytest.mark.parametrize(
    ("stage", "splits"),
    [
        (PRE_CALIBRATION_EVIDENCE_STAGE, ["val", "test"]),
        (RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE, ["test", "val"]),
        (RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE, ["val"]),
    ],
)
def test_declaration_rejects_every_cross_stage_or_combined_split_set(
    tmp_path: Path,
    stage: str,
    splits: list[str],
) -> None:
    event = _event(tmp_path)
    metadata = json.loads(
        (event["bundle"] / "bundle_metadata.json").read_text(encoding="utf-8")
    )

    with pytest.raises(RuntimeError, match="stage/split mismatch"):
        build_prediction_evidence_declaration(
            predictions_path=event["predictions"],
            bundle_dir=event["bundle"],
            bundle_metadata=metadata,
            evidence_stage=stage,
            requested_splits=splits,
        )


def test_resolver_rejects_cross_stage_expectation(tmp_path: Path) -> None:
    event = _event(tmp_path)

    with pytest.raises(RuntimeError, match="evidence_stage mismatch"):
        _resolve(
            event,
            expected_stage=RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
            expected_splits=("test",),
        )
    with pytest.raises(RuntimeError, match="stage/split mismatch"):
        _resolve(event, expected_splits=("test",))
    with pytest.raises(RuntimeError, match="exact tuple"):
        _resolve(event, expected_splits=["val"])


@pytest.mark.parametrize(
    ("stage", "legacy_schema"),
    [
        (
            PRE_CALIBRATION_EVIDENCE_STAGE,
            "entry_candidate_model_direction_prediction_evidence_v2",
        ),
        (
            RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
            "entry_candidate_model_direction_prediction_evidence_v5",
        ),
    ],
)
def test_resolver_rejects_legacy_schema_without_exact_stage(
    tmp_path: Path,
    stage: str,
    legacy_schema: str,
) -> None:
    event = _event(tmp_path, evidence_stage=stage)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    evidence["schema_version"] = legacy_schema
    evidence.pop("evidence_stage")
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="schema_version mismatch"):
        _resolve(event)


def test_resolver_rejects_legacy_combined_val_test_event(tmp_path: Path) -> None:
    event = _event(tmp_path)
    combined = _predictions(("val", "test"))
    combined.to_parquet(event["predictions"], index=False)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    descriptor = parquet_schema_descriptor(event["predictions"])
    combined_sha = sha256_file(event["predictions"])
    evidence.update(
        {
            "schema_version": "entry_candidate_model_direction_prediction_evidence_v2",
            "authoritative": True,
            "sha256": combined_sha,
            "rows": 2,
            "splits": ["val", "test"],
            "parquet_schema": descriptor,
            "parquet_schema_sha256": parquet_schema_sha256(descriptor),
        }
    )
    evidence.pop("evidence_stage")
    report["splits"] = ["val", "test"]
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="schema_version mismatch"):
        _resolve(event, expected_sha256=combined_sha)


def test_resolver_rejects_missing_or_mutated_exact_declarations(
    tmp_path: Path,
) -> None:
    event = _event(tmp_path)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    evidence.pop("evidence_stage")
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="evidence_stage mismatch"):
        _resolve(event)

    report["prediction_evidence"] = dict(event["report"]["prediction_evidence"])
    report["splits"] = ["val", "test"]
    _rewrite_report(event, report)
    with pytest.raises(RuntimeError, match="split declaration mismatch"):
        _resolve(event)


@pytest.mark.parametrize(
    "field",
    ["authoritative", "runtime_head_evidence_authoritative"],
)
def test_resolver_rejects_non_boolean_authority_declaration(
    tmp_path: Path,
    field: str,
) -> None:
    event = _event(tmp_path)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    evidence[field] = "false"
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="authority"):
        _resolve(event)


def test_runtime_head_rejects_divergent_flat_sizing_truth(
    tmp_path: Path,
) -> None:
    event = _event(
        tmp_path,
        evidence_stage=RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
    )
    tampered = pd.read_parquet(event["predictions"])
    tampered.loc[0, "position_size_logit"] = (
        float(tampered.loc[0, "position_size_logit"]) + 0.25
    )
    tampered.to_parquet(event["predictions"], index=False)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    evidence["sha256"] = sha256_file(event["predictions"])
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(
        RuntimeError,
        match="duplicated field mismatch.*position_size_logit",
    ):
        _resolve(
            event,
            expected_sha256=evidence["sha256"],
        )


def test_runtime_head_rejects_divergent_entry_q_column(
    tmp_path: Path,
) -> None:
    event = _event(
        tmp_path,
        evidence_stage=RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
    )
    tampered = pd.read_parquet(event["predictions"])
    # Keep the same winning action so the argmax/pred_direction check passes
    # and only the sealed-envelope comparison can reject the row.
    diverged = list(tampered.loc[0, "entry_action_q_bps"])
    diverged[int(tampered.loc[0, "pred_direction"])] += 1.0
    tampered.at[0, "entry_action_q_bps"] = diverged
    tampered.loc[0, "entry_action_q_margin_bps"] = _entry_q_margin(diverged)
    tampered.to_parquet(event["predictions"], index=False)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    evidence["sha256"] = sha256_file(event["predictions"])
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="Entry-Q mismatch"):
        _resolve(
            event,
            expected_sha256=evidence["sha256"],
        )


def test_authorizing_consumer_rejects_untimestamped_prediction_path(tmp_path: Path) -> None:
    event = _event(tmp_path)
    event["untimestamped"].write_bytes(event["predictions"].read_bytes())

    with pytest.raises(RuntimeError, match="not a timestamped authoritative"):
        _resolve(
            event,
            requested_path=event["untimestamped"],
        )


def test_authorizing_consumer_rejects_mismatched_explicit_report(tmp_path: Path) -> None:
    event = _event(tmp_path)
    wrong = event["out"] / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120001123456Z.json"
    wrong.write_text(event["report_path"].read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        _resolve(
            event,
            prediction_report_path=wrong,
        )


def test_resolver_rejects_symlink_paths_and_unpinned_hash(tmp_path: Path) -> None:
    event = _event(tmp_path)
    # Must still satisfy the timestamped-name contract, so this isolates the
    # symlink rejection rather than re-testing the mutable-locator check.
    prediction_link = (
        event["out"] / "selective_edge_predictions_20260716T120000999999Z.parquet"
    )
    prediction_link.symlink_to(event["predictions"])
    report_link = event["out"] / "report-link.json"
    report_link.symlink_to(event["report_path"])

    with pytest.raises(RuntimeError, match="regular non-symlink"):
        _resolve(event, requested_path=prediction_link)
    with pytest.raises(RuntimeError, match="regular non-symlink"):
        _resolve(event, prediction_report_path=report_link)
    with pytest.raises(RuntimeError, match="expected_sha256"):
        _resolve(event, expected_sha256="not-a-sha")
    with pytest.raises(RuntimeError, match="caller-pinned expected SHA-256"):
        _resolve(event, expected_sha256="0" * 64)


def test_resolver_requires_exact_explicit_parent_lineage(tmp_path: Path) -> None:
    event = _event(tmp_path)
    wrong_bundle = tmp_path / "wrong-bundle"
    wrong_dataset = tmp_path / "wrong-dataset"
    wrong_bundle.mkdir()
    wrong_dataset.mkdir()

    with pytest.raises(RuntimeError, match="bundle_dir mismatch"):
        _resolve(event, bundle_dir=wrong_bundle)
    with pytest.raises(RuntimeError, match="dataset_dir mismatch"):
        _resolve(event, dataset_dir=wrong_dataset)


def test_explicit_parent_resolution_does_not_scan_newest_sibling(
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

    direct, _, _ = _resolve(event)
    assert direct == event["predictions"].resolve()


def test_immutable_prediction_event_refuses_overwrite(tmp_path: Path) -> None:
    event = _event(tmp_path)

    with pytest.raises(RuntimeError, match="already exists"):
        atomic_write_parquet_immutable(_predictions(), event["predictions"])


def test_consumer_rejects_prediction_parquet_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    tampered = _predictions()
    tampered.loc[0, "pred_direction"] = 1
    tampered.to_parquet(event["predictions"], index=False)

    with pytest.raises(RuntimeError, match="caller-pinned expected SHA-256"):
        _resolve(event)


def test_consumer_rejects_declared_row_or_schema_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    report = dict(event["report"])
    report["prediction_evidence"] = dict(report["prediction_evidence"])
    report["prediction_evidence"]["rows"] = 999
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="row-count mismatch"):
        _resolve(event)

    report["prediction_evidence"]["rows"] = 1
    report["prediction_evidence"]["columns"] = report["prediction_evidence"]["columns"][:-1]
    _rewrite_report(event, report)
    with pytest.raises(RuntimeError, match="column schema mismatch"):
        _resolve(event)


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
        _resolve(
            event,
            expected_sha256=evidence["sha256"],
        )


def test_consumer_rejects_tied_entry_q_even_when_parquet_is_rehashed(
    tmp_path: Path,
) -> None:
    event = _event(tmp_path)
    tampered = _predictions()
    # An exact LONG/FLAT tie in raw bps has no unique argmax and must fail
    # closed even when the attacker republishes a matching hash.
    tied = [2.0, -1.0, 2.0]
    tampered.at[0, "entry_action_q_bps"] = tied
    tampered.loc[0, "entry_action_q_margin_bps"] = _entry_q_margin(tied)
    tampered.to_parquet(event["predictions"], index=False)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    evidence["sha256"] = sha256_file(event["predictions"])
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="no unique top action"):
        _resolve(
            event,
            expected_sha256=evidence["sha256"],
        )


def test_consumer_rejects_direction_that_is_not_the_entry_q_argmax(
    tmp_path: Path,
) -> None:
    event = _event(tmp_path)
    tampered = _predictions()
    tampered.loc[0, "pred_direction"] = 1
    tampered.to_parquet(event["predictions"], index=False)
    report = dict(event["report"])
    evidence = dict(report["prediction_evidence"])
    evidence["sha256"] = sha256_file(event["predictions"])
    report["prediction_evidence"] = evidence
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="mismatches Entry Q argmax"):
        _resolve(
            event,
            expected_sha256=evidence["sha256"],
        )


def test_runtime_declaration_rejects_bundle_carrying_retired_calibration(
    tmp_path: Path,
) -> None:
    event = _event(
        tmp_path,
        evidence_stage=RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
    )
    metadata = json.loads(
        (event["bundle"] / "bundle_metadata.json").read_text(encoding="utf-8")
    )
    metadata["direction_calibration"] = {"enabled": True, "temperature": 1.0}

    with pytest.raises(RuntimeError, match="retired calibration"):
        build_prediction_evidence_declaration(
            predictions_path=event["predictions"],
            bundle_dir=event["bundle"],
            bundle_metadata=metadata,
            evidence_stage=RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE,
            requested_splits=["test"],
        )


def test_consumer_rejects_report_direction_contract_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    report = dict(event["report"])
    report["direction_decision_contract"] = dict(report["direction_decision_contract"])
    report["direction_decision_contract"]["runtime_direction_thresholds_allowed"] = True
    _rewrite_report(event, report)

    with pytest.raises(RuntimeError, match="direction_decision_contract mismatch"):
        _resolve(event)


def test_consumer_rejects_bundle_metadata_hash_tamper(tmp_path: Path) -> None:
    event = _event(tmp_path)
    metadata_path = event["bundle"] / "bundle_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["unbound_change"] = True
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="bundle metadata SHA-256 mismatch"):
        _resolve(event)
