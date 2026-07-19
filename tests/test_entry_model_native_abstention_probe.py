from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_abstention_probe_v1 import (
    HISTORICAL_ROLE,
    MODEL_NATIVE_ROLE,
    ROW_SCHEMA_VERSION,
    SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION,
    UTILITY_DEFINITION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.scripts import verify_entry_model_native_abstention_probe_v1 as probe
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
    build_prediction_evidence_declaration,
)


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, split: str, counts: tuple[int, int, int]) -> str:
    return _write_json(
        path,
        {
            "schema_version": probe.HISTORICAL_MANIFEST_SCHEMA,
            "manifest_variant": probe.HISTORICAL_MANIFEST_VARIANT,
            "expected_seq_snap_width": probe.HISTORICAL_SIGNAL_DIM,
            "foundation_smoke_dataset_v1": {
                "split": split,
                "sample_rows": sum(counts),
                "label_counts": {str(index): value for index, value in enumerate(counts)},
            },
        },
    )


def _registry(path: Path, *, available: bool, artifact_dir: Path) -> str:
    artifact = artifact_dir.with_suffix(".json")
    artifact.write_text('{"historical":true}\n', encoding="utf-8")
    return _write_json(
        path,
        {
            "schema_version": "gx1_artifact_selection_v2",
            "active": {},
            "retired": {
                "v10_entry": {
                    "status": "REJECTED_BY_XAU_SEQ513_MODEL_NATIVE_CONTRACT",
                    "artifact_present": True,
                },
                "entry_iql": {
                    "path": str(artifact.resolve()) if available else None,
                    "sha256": (
                        hashlib.sha256(artifact.read_bytes()).hexdigest()
                        if available
                        else None
                    ),
                    "status": "HISTORICAL_COMPARISON_ONLY" if available else "RETIRED_ARTIFACT_ABSENT",
                    "artifact_present": available,
                }
            },
        },
    )


def _base_args(tmp_path: Path, *, registry_available: bool = False) -> argparse.Namespace:
    values: dict[str, object] = {
        "benchmark_evidence_json": None,
        "benchmark_evidence_sha256": None,
        "learned_probe_evidence_json": None,
        "learned_probe_evidence_sha256": None,
        "out_dir": str(tmp_path / "reports"),
        "quiet": True,
    }
    for split, counts in (
        ("train", (1357, 1338, 1400)),
        ("val", (496, 510, 530)),
        ("test", (507, 513, 516)),
    ):
        path = tmp_path / f"{split}.manifest.json"
        values[f"{split}_smoke_manifest"] = str(path)
        values[f"{split}_smoke_manifest_sha256"] = _manifest(path, split, counts)
    registry_path = tmp_path / "PROJECT_STATE_artifacts.json"
    values["artifact_registry_json"] = str(registry_path)
    values["artifact_registry_sha256"] = _registry(
        registry_path,
        available=registry_available,
        artifact_dir=tmp_path / "historical_entry_iql",
    )
    return argparse.Namespace(**values)


def _register_benchmark(
    args: argparse.Namespace,
    evidence_path: Path,
    evidence_sha256: str,
) -> None:
    registry_path = Path(args.artifact_registry_json)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["retired"]["entry_iql"] = {
        "path": str(evidence_path.resolve()),
        "sha256": evidence_sha256,
        "status": "HISTORICAL_COMPARISON_ONLY",
        "artifact_present": True,
    }
    args.artifact_registry_sha256 = _write_json(registry_path, payload)


def _bind_learned_prediction_lineage(
    args: argparse.Namespace,
    tmp_path: Path,
    directions: tuple[int, ...],
) -> None:
    bundle = (tmp_path / "learned-bundle").resolve()
    dataset = (tmp_path / "learned-dataset").resolve()
    event_dir = (tmp_path / "learned-predictions").resolve()
    bundle.mkdir()
    dataset.mkdir()
    event_dir.mkdir()
    metadata = {
        "state_dict_sha256": "a" * 64,
        "direction_decision_contract": model_direction_decision_contract_metadata(),
    }
    (bundle / "bundle_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    probabilities = np.asarray(
        [
            (0.8, 0.1, 0.1) if direction == 0
            else (0.1, 0.8, 0.1) if direction == 1
            else (0.1, 0.1, 0.8)
            for direction in directions
        ],
        dtype=np.float64,
    )
    logits = np.log(probabilities)
    public_logits = np.column_stack(
        [np.max(logits[:, :2], axis=1), logits[:, 2]]
    )
    public_exp = np.exp(public_logits)
    public_probabilities = public_exp / public_exp.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        {
            "split": ["test"] * len(directions),
            "model": ["candidate"] * len(directions),
            "time": pd.to_datetime(
                [
                    f"2026-06-0{index + 1}T00:00:00Z"
                    for index in range(len(directions))
                ],
                utc=True,
            ),
            "pred_direction": list(directions),
            "y_direction": list(directions),
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE]
            * len(directions),
            "direction_logits": logits.tolist(),
            "public_trade_flat_decision_logits": public_logits.tolist(),
            "p_long": probabilities[:, 0],
            "p_short": probabilities[:, 1],
            "p_flat": probabilities[:, 2],
            "public_trade_probability": public_probabilities[:, 0],
            "public_flat_probability": public_probabilities[:, 1],
            "public_trade_flat_margin": public_logits[:, 0]
            - public_logits[:, 1],
            "public_trade_flat_hard_decision": np.argmax(
                public_logits,
                axis=1,
            ),
        }
    )
    predictions = (
        event_dir / "selective_edge_predictions_20260716T120000123456Z.parquet"
    )
    atomic_write_parquet_immutable(frame, predictions)
    declaration = build_prediction_evidence_declaration(
        predictions_path=predictions,
        bundle_dir=bundle,
        bundle_metadata=metadata,
        requested_splits=["test"],
    )
    report_path = (
        event_dir / "ENTRY_CANDIDATE_SELECTIVE_EDGE_20260716T120000123456Z.json"
    )
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": "2026-07-16T12:00:00.123456+00:00",
        "decision": "PASS",
        "failures": [],
        "bundle_dir": str(bundle),
        "dataset_dir": str(dataset),
        "splits": ["test"],
        "models": ["candidate"],
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "prediction_evidence": declaration,
        "predictions_path": str(predictions),
        "bundle_metadata_sha256": declaration["bundle_metadata_sha256"],
        "model_state_dict_sha256": declaration["model_state_dict_sha256"],
        "json_path": str(report_path),
    }
    report_sha = _write_json(report_path, report)
    args.learned_predictions_parquet = str(predictions)
    args.learned_predictions_sha256 = hashlib.sha256(
        predictions.read_bytes()
    ).hexdigest()
    args.learned_prediction_report_json = str(report_path)
    args.learned_prediction_report_sha256 = report_sha
    args.learned_bundle_dir = str(bundle)
    args.learned_dataset_dir = str(dataset)


def _selection_evidence(
    tmp_path: Path,
    *,
    name: str,
    role: str,
    takes: tuple[bool, ...],
    utilities: tuple[float, ...],
) -> tuple[Path, str]:
    rows_path = tmp_path / f"{name}.jsonl"
    universe_rows: list[tuple[str, str, float]] = []
    rows: list[dict[str, object]] = []
    for index, (take, utility) in enumerate(zip(takes, utilities, strict=True)):
        sample_id = f"sample-{index}"
        time_utc = f"2026-06-0{index + 1}T00:00:00Z"
        row: dict[str, object] = {
            "sample_id": sample_id,
            "time_utc": time_utc,
            "take": take,
            "realized_net_utility_bps": utility,
            "costs_included": True,
        }
        if role == MODEL_NATIVE_ROLE:
            row.update(
                {
                    "model_direction_index": 0 if take else 2,
                    "calibrated_argmax": True,
                }
            )
        rows.append(row)
        universe_rows.append((sample_id, time_utc, utility))
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    take_values = [utility for take, utility in zip(takes, utilities, strict=True) if take]
    skip_values = [utility for take, utility in zip(takes, utilities, strict=True) if not take]
    take_mean = sum(take_values) / len(take_values)
    skip_mean = sum(skip_values) / len(skip_values)
    universe_sha = hashlib.sha256(
        json.dumps(
            sorted(universe_rows), separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    evidence_path = tmp_path / f"{name}.evidence.json"
    evidence_sha = _write_json(
        evidence_path,
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "role": role,
            "evaluation_scope": "strict_oot",
            "split": "test",
            "utility_definition": UTILITY_DEFINITION,
            "evaluation_universe_sha256": universe_sha,
            "authority": {
                "direction": False,
                "fallback": False,
                "launch": False,
                "live": False,
            },
            "row_evidence": {
                "schema_version": ROW_SCHEMA_VERSION,
                "format": "jsonl",
                "path": str(rows_path),
                "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            },
            "metrics": {
                "rows": len(rows),
                "take_rows": len(take_values),
                "skip_rows": len(skip_values),
                "coverage": len(take_values) / len(rows),
                "take_ev_net_bps": take_mean,
                "skip_ev_net_bps": skip_mean,
                "take_skip_separation_net_bps": take_mean - skip_mean,
            },
        },
    )
    return evidence_path, evidence_sha


def test_missing_benchmark_emits_report_only_block_with_label_probe(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    report = probe.run(args)

    assert report["decision"] == probe.BLOCK_DECISION
    assert report["empirical_comparison_performed"] is False
    assert report["empirical_gate_passed"] is False
    assert "HISTORICAL_ENTRY_IQL_BENCHMARK_BYTES_NOT_REGISTERED" in report["blockers"]
    assert "EXACT_HISTORICAL_SELECTION_BENCHMARK_EVIDENCE_MISSING" in report["blockers"]
    assert report["inputs"]["historical_smoke_manifests"]["val"]["label_counts"] == {
        "LONG": 496,
        "SHORT": 510,
        "FLAT": 530,
    }
    assert report["probe_diagnostics"]["active_recipe"]["all_required_weights_positive"] is True
    assert report["probe_diagnostics"]["active_recipe"]["empirical_abstention_edge_proven"] is False
    assert report["direction_authority"] is False
    assert report["rebuild_authorized"] is False
    assert report["training_authorized"] is False
    assert report["side_effects"]["parquet_read"] is False


def test_bound_manifest_hash_mismatch_fails_before_report(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    args.test_smoke_manifest_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        probe.run(args)
    assert not (tmp_path / "reports").exists()


def test_registry_rejects_retired_entry_records_under_active(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    registry_path = Path(args.artifact_registry_json)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["active"]["entry_iql"] = payload["retired"]["entry_iql"]
    args.artifact_registry_sha256 = _write_json(registry_path, payload)

    with pytest.raises(RuntimeError, match="non-active Entry records under active"):
        probe.run(args)


def test_repository_registry_has_only_true_decision_roles_under_active() -> None:
    payload = json.loads(Path("PROJECT_STATE_artifacts.json").read_text(encoding="utf-8"))

    assert "v10_entry" not in payload["active"]
    assert "entry_iql" not in payload["active"]
    assert payload["retired"]["v10_entry"]["status"].startswith("REJECTED")
    assert payload["retired"]["entry_iql"] == {
        "path": None,
        "sha256": None,
        "status": "RETIRED_ARTIFACT_ABSENT",
        "artifact_present": False,
        "in_sample_only": False,
        "note": (
            "The formerly registered historical Entry-IQL directory is absent as of "
            "2026-07-19. No benchmark bytes are registered, and Entry-IQL is not an "
            "Entry authority or fallback."
        ),
    }


def test_exact_aligned_evidence_can_pass_comparison_but_grants_no_authority(tmp_path: Path) -> None:
    args = _base_args(tmp_path)
    utilities = (8.0, 4.0, -3.0, -5.0)
    historical_path, historical_sha = _selection_evidence(
        tmp_path,
        name="historical",
        role=HISTORICAL_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    learned_path, learned_sha = _selection_evidence(
        tmp_path,
        name="learned",
        role=MODEL_NATIVE_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    _register_benchmark(args, historical_path, historical_sha)
    _bind_learned_prediction_lineage(args, tmp_path, (0, 0, 2, 2))
    args.benchmark_evidence_json = str(historical_path)
    args.benchmark_evidence_sha256 = historical_sha
    args.learned_probe_evidence_json = str(learned_path)
    args.learned_probe_evidence_sha256 = learned_sha

    report = probe.run(args)

    assert report["decision"] == probe.PASS_DECISION
    assert report["empirical_comparison"]["passed"] is True
    assert report["rebuild_authorized"] is False
    assert report["training_authorized"] is False
    assert report["launch_authorized"] is False
    assert report["live_authorized"] is False
    assert report["side_effects"]["parquet_read"] is True
    assert report["inputs"]["model_native_prediction_lineage"][
        "direction_rows_exact"
    ] is True


def test_historical_evidence_must_be_the_exact_registered_artifact(
    tmp_path: Path,
) -> None:
    args = _base_args(tmp_path, registry_available=True)
    utilities = (8.0, 4.0, -3.0, -5.0)
    historical_path, historical_sha = _selection_evidence(
        tmp_path,
        name="unregistered-historical",
        role=HISTORICAL_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    learned_path, learned_sha = _selection_evidence(
        tmp_path,
        name="learned-for-mismatched-registry",
        role=MODEL_NATIVE_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    args.benchmark_evidence_json = str(historical_path)
    args.benchmark_evidence_sha256 = historical_sha
    args.learned_probe_evidence_json = str(learned_path)
    args.learned_probe_evidence_sha256 = learned_sha

    report = probe.run(args)

    assert report["empirical_comparison_performed"] is False
    assert (
        "HISTORICAL_SELECTION_BENCHMARK_IS_NOT_EXACT_REGISTERED_ARTIFACT"
        in report["blockers"]
    )


def test_learned_evidence_without_prediction_lineage_cannot_compare(
    tmp_path: Path,
) -> None:
    args = _base_args(tmp_path)
    utilities = (8.0, 4.0, -3.0, -5.0)
    historical_path, historical_sha = _selection_evidence(
        tmp_path,
        name="historical-no-learned-lineage",
        role=HISTORICAL_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    learned_path, learned_sha = _selection_evidence(
        tmp_path,
        name="learned-no-lineage",
        role=MODEL_NATIVE_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    _register_benchmark(args, historical_path, historical_sha)
    args.benchmark_evidence_json = str(historical_path)
    args.benchmark_evidence_sha256 = historical_sha
    args.learned_probe_evidence_json = str(learned_path)
    args.learned_probe_evidence_sha256 = learned_sha

    report = probe.run(args)

    assert report["empirical_comparison_performed"] is False
    assert "EXACT_MODEL_NATIVE_PREDICTION_LINEAGE_MISSING" in report["blockers"]


def test_model_native_rows_cannot_add_a_threshold_selector(tmp_path: Path) -> None:
    args = _base_args(tmp_path, registry_available=True)
    path, sha = _selection_evidence(
        tmp_path,
        name="learned",
        role=MODEL_NATIVE_ROLE,
        takes=(True, True, False, False),
        utilities=(8.0, 4.0, -3.0, -5.0),
    )
    rows_path = tmp_path / "learned.jsonl"
    rows = rows_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["take"] = False
    rows[0] = json.dumps(first, sort_keys=True)
    rows_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["row_evidence"]["sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    sha = _write_json(path, payload)

    args.learned_probe_evidence_json = str(path)
    args.learned_probe_evidence_sha256 = sha
    report = probe.run(args)
    assert any("take must equal calibrated argmax" in blocker for blocker in report["blockers"])


def test_learned_rows_must_match_exact_prediction_event_direction(
    tmp_path: Path,
) -> None:
    args = _base_args(tmp_path)
    utilities = (8.0, 4.0, -3.0, -5.0)
    historical_path, historical_sha = _selection_evidence(
        tmp_path,
        name="historical-direction-lineage",
        role=HISTORICAL_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    learned_path, _ = _selection_evidence(
        tmp_path,
        name="learned-direction-lineage",
        role=MODEL_NATIVE_ROLE,
        takes=(True, True, False, False),
        utilities=utilities,
    )
    learned_rows = tmp_path / "learned-direction-lineage.jsonl"
    rows = learned_rows.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["model_direction_index"] = 1
    rows[0] = json.dumps(first, sort_keys=True)
    learned_rows.write_text("\n".join(rows) + "\n", encoding="utf-8")
    learned_payload = json.loads(learned_path.read_text(encoding="utf-8"))
    learned_payload["row_evidence"]["sha256"] = hashlib.sha256(
        learned_rows.read_bytes()
    ).hexdigest()
    learned_sha = _write_json(learned_path, learned_payload)
    _register_benchmark(args, historical_path, historical_sha)
    _bind_learned_prediction_lineage(args, tmp_path, (0, 0, 2, 2))
    args.benchmark_evidence_json = str(historical_path)
    args.benchmark_evidence_sha256 = historical_sha
    args.learned_probe_evidence_json = str(learned_path)
    args.learned_probe_evidence_sha256 = learned_sha

    report = probe.run(args)

    assert report["empirical_comparison_performed"] is False
    assert any(
        "learned evidence direction differs from prediction event" in blocker
        for blocker in report["blockers"]
    )


def test_control_surface_exposes_only_report_route() -> None:
    source = Path("scripts/entry_next_edge_control.sh").read_text(encoding="utf-8")
    assert "model-native-abstention-probe)" in source
    assert "verify_entry_model_native_abstention_probe_v1" in source
    assert "--benchmark-evidence-json" not in source
    assert "--learned-probe-evidence-json" not in source
    help_text = probe.build_parser().format_help()
    for option in (
        "--learned-predictions-parquet",
        "--learned-predictions-sha256",
        "--learned-prediction-report-json",
        "--learned-prediction-report-sha256",
        "--learned-bundle-dir",
        "--learned-dataset-dir",
    ):
        assert option in help_text
