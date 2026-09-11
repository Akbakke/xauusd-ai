from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract,
)
from gx1.contracts.unified_exit_selected_sampler_v1 import (
    build_selected_sampler_artifact,
    file_sha256,
    require_selected_sampler_artifact,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    candidate_path = tmp_path / "candidates.json"
    predecessor_path = tmp_path / "v3.json"
    root_path = tmp_path / "v4.json"
    equivalence_path = tmp_path / "equivalence.json"
    receipt_path = tmp_path / "receipt.json"
    sampler = build_random_access_sampler_contract(
        split="train",
        source_lineage_sha256="1" * 64,
        transition_budget_per_epoch=65536,
        transitions_per_entry=4,
        entry_pair_population=65295,
    )
    candidates = {
        "schema_version": "gx1_unified_exit_sampler_benchmark_candidate_set_v1",
        "candidate_set_sha256": "2" * 64,
        "candidates": [
            {
                "sampler_contract": sampler,
                "selected": False,
                "status": "BENCHMARK_PENDING",
            }
        ],
    }
    _write(candidate_path, candidates)
    predecessor = {
        "root_sha256": "f2797d10a9a97a028fe3c532a3ed10849766673e860327fb53139338512d6d64",
        "test_accessed": False,
    }
    _write(predecessor_path, predecessor)
    root = {
        "root_sha256": "25f911a0d03595ee5bbbea5fa6a66da98bed91eb24bdb0f561f6870a719d007f",
        "test_accessed": False,
    }
    _write(root_path, root)
    equivalence = {
        "schema_version": "gx1_unified_exit_random_access_index_v3_to_v4_equivalence_v1",
        "decision": "PASS",
        "benchmark_receipt_transfer_to_v4_authorized": True,
        "only_parent_entry_coordinate_and_binding_fields_added": True,
        "selection_uses_outcome_values": False,
        "test_accessed": False,
        "sampler_candidate_set_sha256": candidates["candidate_set_sha256"],
        "predecessor_root": {
            "path": str(predecessor_path),
            "sha256": file_sha256(predecessor_path),
        },
        "sampled_transition_schedules": [
            {
                "sampler_contract_sha256": sampler["contract_sha256"],
                "schedule_streams_byte_identical": True,
                "transition_budget_per_epoch": 65536,
            }
        ],
    }
    equivalence["receipt_sha256"] = canonical_sha256(equivalence)
    _write(equivalence_path, equivalence)
    receipt = {
        "schema_version": "gx1_unified_exit_random_access_train_benchmark_v2",
        "decision": "PASS",
        "candidate_selection_performed": True,
        "selection_uses_outcome_values": False,
        "test_data_used": False,
        "selected_candidate": {
            "batch_size": 16,
            "transition_budget_per_epoch": 65536,
            "sampler_contract_sha256": sampler["contract_sha256"],
        },
        "run_bindings": {
            "files": {
                "candidate_set": {
                    "path": str(candidate_path),
                    "sha256": file_sha256(candidate_path),
                },
                "root_manifest": {
                    "path": str(predecessor_path),
                    "sha256": file_sha256(predecessor_path),
                },
            }
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write(receipt_path, receipt)
    return receipt_path, candidate_path, root_path, equivalence_path


def test_selected_sampler_is_bound_to_measured_v3_and_equivalent_v4(
    tmp_path: Path,
) -> None:
    receipt, candidates, root, equivalence = _fixture(tmp_path)
    artifact = build_selected_sampler_artifact(
        benchmark_receipt_path=receipt,
        candidate_set_path=candidates,
        random_access_root_path=root,
        equivalence_receipt_path=equivalence,
    )
    assert require_selected_sampler_artifact(artifact) == artifact
    assert (
        artifact["batch_size"] == 16
        and artifact["transition_budget_per_epoch"] == 65536
    )
    tampered = copy.deepcopy(artifact)
    tampered["batch_size"] = 8
    with pytest.raises(RuntimeError):
        require_selected_sampler_artifact(tampered, verify_files=False)
