from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gx1.contracts.unified_exit_economics_objective_v2 import SECONDS_PER_YEAR, MARK_TO_MARKET_REWARD_ACCOUNTING, LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import file_sha256
from gx1.scripts.build_unified_exit_training_economics_readiness_v1 import (
    CAPITAL_HURDLE_METHOD_RECEIPT_SCHEMA_VERSION,
    build_training_economics_readiness,
    build_val_economics_reference,
    seal_capital_hurdle_method_receipt,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _receipt(tmp_path: Path) -> Path:
    source = (tmp_path / "adopted_capital_policy.txt").resolve()
    source.write_text("synthetic policy method; rate is supplied by owner\n")
    receipt = seal_capital_hurdle_method_receipt(
        {
            "schema_version": CAPITAL_HURDLE_METHOD_RECEIPT_SCHEMA_VERSION,
            "decision": "PASS",
            "method": "prospective_capital_policy_v1",
            "owner_kind": "project_preregistered_prospective_policy",
            "applicable_splits": ["train", "val"],
            "fitted_splits": [],
            "validation_or_test_used": False,
            "train_split_sha256": "1" * 64,
            "train_fold_sha256": "2" * 64,
            "source_lineage_sha256": "3" * 64,
            "policy_sha256": "4" * 64,
            "effective_annual_return_hurdle": 0.10,
            "annual_continuous_hurdle_rate": math.log1p(0.10),
            "rate_conversion_formula": "rho=ln(1+effective_annual_return)",
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": SECONDS_PER_YEAR,
            "source_method_artifact_path": str(source),
            "source_method_artifact_sha256": file_sha256(source),
            "test_data_used": False,
        }
    )
    path = (tmp_path / "hurdle_method.receipt.json").resolve()
    _write(path, receipt)
    return path


@pytest.mark.parametrize("reward_accounting", ["terminal_cash_v2", MARK_TO_MARKET_REWARD_ACCOUNTING, LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING])
def test_builder_reads_rho_from_train_method_and_val_reuses_it(tmp_path: Path, reward_accounting) -> None:
    receipt_path = _receipt(tmp_path)
    train_path = (tmp_path / "train.economics_readiness.json").resolve()
    result = build_training_economics_readiness(
        output_path=train_path,
        method_receipt_path=receipt_path,
        train_split_sha256="1" * 64,
        train_fold_sha256="2" * 64,
        source_lineage_sha256="3" * 64,
        policy_sha256="4" * 64,
        publish=True,
        reward_accounting=reward_accounting,
    )
    assert result["validated_annual_continuous_hurdle_rate"] == math.log1p(0.10)
    val = build_val_economics_reference(
        output_path=(tmp_path / "val.reference.json").resolve(),
        train_readiness_path=train_path,
        val_split_sha256="5" * 64,
        publish=True,
    )
    assert val["annual_continuous_hurdle_rate"] == math.log1p(0.10)
    assert val["rho_refit_for_val"] is False
    assert val["train_readiness_file_sha256"] == file_sha256(train_path)


def test_builder_rejects_tampered_source_method(tmp_path: Path) -> None:
    receipt_path = _receipt(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    Path(receipt["source_method_artifact_path"]).write_text("changed\n")
    with pytest.raises(RuntimeError, match="METHOD_RECEIPT_INVALID"):
        build_training_economics_readiness(
            output_path=tmp_path / "out.json",
            method_receipt_path=receipt_path,
            train_split_sha256="1" * 64,
            train_fold_sha256="2" * 64,
            source_lineage_sha256="3" * 64,
            policy_sha256="4" * 64,
            publish=False,
        )



def test_train_and_val_readiness_cannot_overwrite_existing_evidence(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    train = tmp_path / "train.json"
    kwargs = dict(
        output_path=train, method_receipt_path=receipt,
        train_split_sha256="1" * 64, train_fold_sha256="2" * 64,
        source_lineage_sha256="3" * 64, policy_sha256="4" * 64, publish=True,
    )
    build_training_economics_readiness(**kwargs)
    before = train.read_bytes()
    with pytest.raises(RuntimeError, match="OUTPUT_EXISTS"):
        build_training_economics_readiness(**kwargs)
    assert train.read_bytes() == before
    val = tmp_path / "val.json"
    val.write_bytes(b"preserved VAL evidence")
    with pytest.raises(RuntimeError, match="OUTPUT_EXISTS"):
        build_val_economics_reference(
            output_path=val, train_readiness_path=train,
            val_split_sha256="5" * 64, publish=True,
        )
    assert val.read_bytes() == b"preserved VAL evidence"


@pytest.mark.parametrize("owner", ["build_unified_exit_training_economics_readiness_v1", "build_unified_exit_no_cap_authority_v1"])
def test_late_file_collision_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str,
) -> None:
    import importlib
    module = importlib.import_module("gx1.scripts." + owner)
    publish = module._publish_file_noreplace
    destination = tmp_path / "immutable.json"

    def collide(stage: Path, final: Path) -> None:
        final.write_bytes(b"other attempt")
        publish(stage, final)

    monkeypatch.setattr(module, "_publish_file_noreplace", collide)
    with pytest.raises(RuntimeError, match="already exists"):
        module._write_atomic(destination, {"decision": "PASS"})
    assert destination.read_bytes() == b"other attempt"
    failed = list(tmp_path.glob(".immutable.json.staging.*"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text()) == {"decision": "PASS"}


@pytest.mark.parametrize("owner", ["build_unified_exit_training_economics_readiness_v1", "build_unified_exit_no_cap_authority_v1"])
def test_corrupt_staging_is_rejected_before_final_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str,
) -> None:
    import importlib
    module = importlib.import_module("gx1.scripts." + owner)
    read = Path.read_bytes
    destination = tmp_path / "immutable.json"

    def corrupt_read(path: Path) -> bytes:
        payload = read(path)
        return payload + b" " if ".staging." in path.name else payload

    monkeypatch.setattr(Path, "read_bytes", corrupt_read)
    with pytest.raises(RuntimeError, match="STAGED_BYTES_INVALID"):
        module._write_atomic(destination, {"decision": "PASS"})
    assert not destination.exists()
    failed = list(tmp_path.glob(".immutable.json.staging.*"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text()) == {"decision": "PASS"}
