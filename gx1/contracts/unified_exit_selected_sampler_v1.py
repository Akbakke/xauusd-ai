"""Immutable admission contract for the benchmark-selected TRAIN sampler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    require_random_access_sampler_contract,
)

SCHEMA_VERSION = "gx1_unified_exit_selected_sampler_v1"
BENCHMARK_SCHEMA = "gx1_unified_exit_random_access_train_benchmark_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved != path or not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_PATH_INVALID")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_JSON_INVALID")
    return value


def build_selected_sampler_artifact(
    *,
    benchmark_receipt_path: Path,
    candidate_set_path: Path,
    random_access_root_path: Path,
    equivalence_receipt_path: Path,
) -> dict[str, Any]:
    receipt = _read(benchmark_receipt_path)
    candidate_set = _read(candidate_set_path)
    root = _read(random_access_root_path)
    equivalence = _read(equivalence_receipt_path)
    receipt_data = dict(receipt)
    claimed_receipt = receipt_data.pop("receipt_sha256", None)
    selected = receipt.get("selected_candidate")
    if (
        receipt.get("schema_version") != BENCHMARK_SCHEMA
        or receipt.get("decision") != "PASS"
        or claimed_receipt != canonical_sha256(receipt_data)
        or receipt.get("candidate_selection_performed") is not True
        or receipt.get("selection_uses_outcome_values") is not False
        or receipt.get("test_data_used") is not False
        or not isinstance(selected, Mapping)
        or selected.get("batch_size") != 16
        or selected.get("transition_budget_per_epoch") != 65536
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_BENCHMARK_INVALID")
    run_files = receipt.get("run_bindings", {}).get("files", {})
    predecessor_root_path = Path(
        str(equivalence.get("predecessor_root", {}).get("path", ""))
    )
    expected_files = {
        "candidate_set": candidate_set_path,
        "root_manifest": predecessor_root_path,
    }
    for name, path in expected_files.items():
        binding = run_files.get(name)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str(path)
            or binding.get("sha256") != file_sha256(path)
        ):
            raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_SOURCE_MISMATCH")
    sampler_sha = selected.get("sampler_contract_sha256")
    equivalence_data = dict(equivalence)
    claimed_equivalence = equivalence_data.pop("receipt_sha256", None)
    schedule_proofs = [
        proof
        for proof in equivalence.get("sampled_transition_schedules", [])
        if isinstance(proof, Mapping)
        and proof.get("sampler_contract_sha256") == sampler_sha
    ]
    predecessor = equivalence.get("predecessor_root")
    if (
        equivalence.get("schema_version")
        != "gx1_unified_exit_random_access_index_v3_to_v4_equivalence_v1"
        or equivalence.get("decision") != "PASS"
        or claimed_equivalence != canonical_sha256(equivalence_data)
        or equivalence.get("benchmark_receipt_transfer_to_v4_authorized") is not True
        or equivalence.get("only_parent_entry_coordinate_and_binding_fields_added")
        is not True
        or equivalence.get("selection_uses_outcome_values") is not False
        or equivalence.get("test_accessed") is not False
        or equivalence.get("sampler_candidate_set_sha256")
        != candidate_set.get("candidate_set_sha256")
        or not isinstance(predecessor, Mapping)
        or predecessor.get("path") != str(predecessor_root_path)
        or predecessor.get("sha256") != file_sha256(predecessor_root_path)
        or len(schedule_proofs) != 1
        or schedule_proofs[0].get("schedule_streams_byte_identical") is not True
        or schedule_proofs[0].get("transition_budget_per_epoch") != 65536
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_EQUIVALENCE_INVALID")
    matching = [
        row.get("sampler_contract")
        for row in candidate_set.get("candidates", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("sampler_contract"), Mapping)
        and row["sampler_contract"].get("contract_sha256") == sampler_sha
    ]
    if len(matching) != 1:
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_CANDIDATE_MISSING")
    sampler = require_random_access_sampler_contract(matching[0])
    if sampler.get("split") != "train":
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_SPLIT_INVALID")
    if (
        sampler["transition_budget_per_epoch"] != 65536
        or sampler["entry_pairs_per_epoch"] != 16384
        or sampler["transitions_per_entry"] != 4
        or root.get("root_sha256")
        != "25f911a0d03595ee5bbbea5fa6a66da98bed91eb24bdb0f561f6870a719d007f"
        or root.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_BINDING_INVALID")
    value = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "benchmark_receipt": {
            "path": str(benchmark_receipt_path),
            "file_sha256": file_sha256(benchmark_receipt_path),
            "receipt_sha256": claimed_receipt,
        },
        "candidate_set": {
            "path": str(candidate_set_path),
            "file_sha256": file_sha256(candidate_set_path),
            "candidate_set_sha256": candidate_set.get("candidate_set_sha256"),
        },
        "random_access_root": {
            "path": str(random_access_root_path),
            "file_sha256": file_sha256(random_access_root_path),
            "root_sha256": root["root_sha256"],
        },
        "v3_to_v4_equivalence": {
            "path": str(equivalence_receipt_path),
            "file_sha256": file_sha256(equivalence_receipt_path),
            "receipt_sha256": claimed_equivalence,
        },
        "selected_sampler_contract": sampler,
        "selected_sampler_contract_sha256": sampler["contract_sha256"],
        "batch_size": 16,
        "transition_budget_per_epoch": 65536,
        "entry_pairs_per_epoch": 16384,
        "selection_uses_outcome_values": False,
        "test_data_used": False,
    }
    value["artifact_sha256"] = canonical_sha256(value)
    return value


def require_selected_sampler_artifact(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("artifact_sha256", None)
    sampler = require_random_access_sampler_contract(
        value.get("selected_sampler_contract", {})
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("decision") != "PASS"
        or sampler.get("split") != "train"
        or claimed != canonical_sha256(data)
        or value.get("selected_sampler_contract_sha256") != sampler["contract_sha256"]
        or value.get("batch_size") != 16
        or value.get("transition_budget_per_epoch") != 65536
        or value.get("entry_pairs_per_epoch") != 16384
        or value.get("selection_uses_outcome_values") is not False
        or value.get("test_data_used") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_INVALID")
    if verify_files:
        for name in (
            "benchmark_receipt",
            "candidate_set",
            "random_access_root",
            "v3_to_v4_equivalence",
        ):
            binding = value.get(name)
            path = (
                Path(str(binding.get("path", "")))
                if isinstance(binding, Mapping)
                else Path()
            )
            if (
                not path.is_absolute()
                or not path.is_file()
                or path.is_symlink()
                or file_sha256(path) != binding.get("file_sha256")
            ):
                raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_FILE_INVALID")
    return dict(value)


__all__ = (
    "SCHEMA_VERSION",
    "build_selected_sampler_artifact",
    "require_selected_sampler_artifact",
)
