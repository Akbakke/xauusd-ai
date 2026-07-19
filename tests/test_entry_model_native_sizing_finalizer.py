from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import finalize_entry_model_native_sizing_v1 as sizing_finalizer
from gx1.scripts.finalize_entry_model_native_sizing_v1 import (
    SizingFinalizationError,
    adopt_learned_sizing,
    bind_bundle_sizing_calibration,
    finalize_test_sizing_proof,
)
from tests.model_native_sizing_support import (
    write_passing_sizing_calibration_and_proof,
)


def test_sizing_split_bindings_come_only_from_prediction_report(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    bound_manifest = dataset_dir / "entry_model_native_test.manifest.json"
    bound_parquet = dataset_dir / "entry_model_native_test.parquet"
    bound_manifest.write_text("{}\n", encoding="utf-8")
    bound_parquet.write_bytes(b"bound")
    (dataset_dir / "decoy_test.parquet").write_bytes(b"decoy")
    report = {
        "dataset_signal_contract": {
            "splits": {
                "test": {
                    "manifest_path": str(bound_manifest.resolve()),
                    "manifest_sha256": "a" * 64,
                    "parquet_path": str(bound_parquet.resolve()),
                    "parquet_sha256": "b" * 64,
                }
            }
        }
    }

    assert sizing_finalizer._dataset_split_bindings(
        report,
        dataset_dir,
        ("test",),
    ) == {
        "test": {
            "manifest_path": str(bound_manifest.resolve()),
            "manifest_sha256": "a" * 64,
            "parquet_path": str(bound_parquet.resolve()),
            "parquet_sha256": "b" * 64,
        }
    }
    source = Path(sizing_finalizer.__file__).read_text(encoding="utf-8")
    helper_source = source[
        source.index("def _dataset_split_bindings(") : source.index(
            "\ndef _prediction_provenance(",
        )
    ]
    assert ".glob(" not in helper_source


def test_canonical_fit_and_bundle_binding_clones_exact_pristine_bundle(
    tmp_path: Path,
) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    source_bundle = tmp_path / "source_bundle"
    final_bundle = evidence["bundle_dir"]
    required_inventory = {
        "bundle_metadata.json",
        "MASTER_TRANSFORMER_LOCK.json",
        "model_state_dict.pt",
    }

    source_metadata = json.loads(
        (source_bundle / "bundle_metadata.json").read_text(encoding="utf-8")
    )
    source_lock = json.loads(
        (source_bundle / "MASTER_TRANSFORMER_LOCK.json").read_text(encoding="utf-8")
    )
    final_metadata = json.loads(
        (final_bundle / "bundle_metadata.json").read_text(encoding="utf-8")
    )
    final_lock = json.loads(
        (final_bundle / "MASTER_TRANSFORMER_LOCK.json").read_text(encoding="utf-8")
    )

    assert {path.name for path in source_bundle.iterdir()} == required_inventory
    assert {path.name for path in final_bundle.iterdir()} == required_inventory
    assert "model_native_sizing_calibration" not in source_metadata
    assert "model_native_sizing_calibration" not in source_lock
    assert final_metadata["model_native_sizing_calibration"] == final_lock[
        "model_native_sizing_calibration"
    ]
    assert final_metadata["model_native_sizing_calibration"] == evidence[
        "bundle_calibration"
    ]
    assert evidence["calibration"]["fit_scope"] == "TRAIN_VAL_ONLY"
    assert evidence["calibration"]["fit_splits"] == ["train", "val"]
    proof = evidence["proof"]
    assert proof["account_capacity_grid"]["decision"] == "PASS"
    assert set(proof["account_capacity_grid"]["scenarios"]) == {
        "small",
        "medium",
        "large",
    }
    for scenario in proof["paired_oos_utility"]["scenarios"].values():
        assert scenario["historical_1_unit_control"]["decision"] == "PASS"
        assert (
            scenario["equal_total_continuous_allocation_control"]["decision"]
            == "PASS"
        )
        assert scenario["rounded_equal_total_allocation_diagnostic"]["role"] == (
            "diagnostic_only_not_admission"
        )

    (source_bundle / "stale_policy.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SizingFinalizationError, match="inventory must be exact"):
        bind_bundle_sizing_calibration(
            source_bundle_dir=source_bundle,
            output_bundle_dir=tmp_path / "must_not_exist_bundle",
            calibration_path=Path(evidence["calibration_artifact"]["json_path"]),
        )
    assert not (tmp_path / "must_not_exist_bundle").exists()


def test_label_horizon_proof_is_diagnostic_and_capital_adoption_is_blocked(
    tmp_path: Path,
) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    proof_path = Path(evidence["oos_proof_artifact"]["json_path"])
    with pytest.raises(SizingFinalizationError, match="structurally BLOCKED"):
        adopt_learned_sizing(
            bundle_dir=evidence["bundle_dir"],
            calibration_path=Path(evidence["calibration_artifact"]["json_path"]),
            proof_path=proof_path,
            authority_root=evidence["authority_root"],
            accepted_via_vedtak="UNIT_FINAL_SIZING_ADOPTION",
        )

    assert evidence["proof"]["decision"] == "PASS"
    assert evidence["proof"]["evaluation_scope"] == (
        "FULL_TEST_LABEL_HORIZON_SIZING_HEAD_DIAGNOSTIC_ONLY"
    )
    newest = max(
        (evidence["authority_root"] / "adoption").glob(
            "ENTRY_MODEL_NATIVE_SIZING_ADOPTION_*.json"
        )
    )
    terminal = json.loads(newest.read_text(encoding="utf-8"))
    assert terminal["decision"] == "FAIL"
    assert terminal["attempted_stage"] == "adoption"


def test_failed_proof_refresh_publishes_newer_terminal_fail(tmp_path: Path) -> None:
    evidence = write_passing_sizing_calibration_and_proof(tmp_path)
    proof_dir = evidence["authority_root"] / "proof"
    passing_path = Path(evidence["oos_proof_artifact"]["json_path"])

    with pytest.raises(SizingFinalizationError):
        finalize_test_sizing_proof(
            calibration_path=Path(evidence["calibration_artifact"]["json_path"]),
            oos_source_path=tmp_path / "outside_family" / "missing.json",
            authority_root=evidence["authority_root"],
        )

    newest = max(proof_dir.glob("ENTRY_MODEL_NATIVE_SIZING_OOS_PROOF_*.json"))
    terminal = json.loads(newest.read_text(encoding="utf-8"))
    assert newest.name > passing_path.name
    assert terminal["decision"] == "FAIL"
    assert terminal["attempted_stage"] == "proof"
    assert terminal["failures"]
