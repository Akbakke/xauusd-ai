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


DIRECT_SELECTION_MODE = "direct_same_root_benchmark_v1"


def _build_direct_selected_sampler(
    *, benchmark_receipt_path: Path, candidate_set_path: Path, random_access_root_path: Path,
) -> dict[str, Any]:
    from gx1.contracts.unified_exit_random_access_train_factory_v1 import _candidate_contracts
    from gx1.contracts.unified_exit_random_access_index_v1 import (
        FULL_POPULATION_ROOT_SCHEMA_VERSION, RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
        require_random_access_index_root, require_random_access_index_manifest,
    )
    from gx1.scripts.benchmark_unified_exit_random_access_train_v1 import require_measured_sampler_benchmark

    contracts = _candidate_contracts(candidate_set_path)
    candidates = _read(candidate_set_path)
    receipt = require_measured_sampler_benchmark(
        _read(benchmark_receipt_path), sampler_contracts=contracts,
    )
    root = require_random_access_index_root(_read(random_access_root_path))
    train = root["splits"]["train"]
    selected = receipt["selected_candidate"]
    sampler = contracts[selected["transition_budget_per_epoch"]]
    if (
        root["schema_version"] != FULL_POPULATION_ROOT_SCHEMA_VERSION
        or root.get("sampler_selection_status") != "BLOCKED_PENDING_TRAIN_ONLY_BENCHMARK"
        or root.get("selected_sampler_contract_sha256") is not None
        or train.get("entry_row_count") != sampler["entry_pair_population"]
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_DIRECT_ROOT_INVALID")
    run_files = receipt.get("run_bindings", {}).get("files", {})
    for name, path in (("candidate_set", candidate_set_path), ("root_manifest", random_access_root_path)):
        if run_files.get(name) != {"path": str(path), "sha256": file_sha256(path)}:
            raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_SOURCE_MISMATCH")
    # These are the existing index producer's TRAIN metadata names. Reject a
    # role-swapped source before following its pointer, including sealed TEST.
    manifest_path = Path(str(train.get("manifest_path", "")))
    if not manifest_path.is_absolute() or manifest_path.name != "train.manifest.json":
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_TRAIN_MANIFEST_REQUIRED")
    manifest = require_random_access_index_manifest(_read(manifest_path), expected_split="train")
    if (
        manifest["schema_version"] != RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION
        or manifest["manifest_sha256"] != train.get("manifest_sha256")
        or manifest.get("entry_row_count") != sampler["entry_pair_population"]
        or manifest.get("parent_entry_source_rows") != sampler["entry_pair_population"]
        or manifest.get("index_parquet_sha256") != train.get("index_parquet_sha256")
        or manifest.get("index_parquet_path") != train.get("index_parquet_path")
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_TRAIN_MANIFEST_INVALID")
    bundle_binding = manifest["source_bindings"].get("final_bindings_bundle", {})
    bundle_path = Path(str(bundle_binding.get("path", "")))
    if not bundle_path.is_absolute() or bundle_path.name != "FINAL_BINDINGS_BUNDLE.json":
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_FINAL_BINDINGS_REQUIRED")
    bundle = _read(bundle_path)
    bundle_data = dict(bundle)
    bundle_sha = bundle_data.pop("bundle_sha256", None)
    if (
        file_sha256(bundle_path) != bundle_binding.get("sha256")
        or bundle.get("schema_version") != "gx1_unified_exit_pilot_final_bindings_bundle_v1"
        or bundle.get("decision") != "BLOCKED_PENDING_TRAIN_ONLY_SAMPLER_BENCHMARK"
        or bundle.get("test_accessed") is not False
        or bundle_sha != canonical_sha256(bundle_data)
        or bundle_sha != root.get("final_bindings_bundle_sha256")
        or bundle.get("sampler_benchmark_candidates") != candidates
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_FINAL_BINDINGS_INVALID")
    value = {
        "schema_version": SCHEMA_VERSION, "decision": "PASS",
        "selection_mode": DIRECT_SELECTION_MODE,
        "reference_workload": receipt["reference_workload"],
        "input_geometry": receipt.get("run_bindings", {}).get("input_geometry"),
        "benchmark_design": run_files.get("chronological_design"),
        "benchmark_receipt": {"path": str(benchmark_receipt_path),
            "file_sha256": file_sha256(benchmark_receipt_path), "receipt_sha256": receipt["receipt_sha256"]},
        "candidate_set": {"path": str(candidate_set_path),
            "file_sha256": file_sha256(candidate_set_path), "candidate_set_sha256": candidates["candidate_set_sha256"]},
        "random_access_root": {"path": str(random_access_root_path),
            "file_sha256": file_sha256(random_access_root_path), "root_sha256": root["root_sha256"]},
        "train_index_manifest": {"path": str(manifest_path),
            "file_sha256": file_sha256(manifest_path), "manifest_sha256": manifest["manifest_sha256"]},
        "final_bindings_bundle": {"path": str(bundle_path),
            "file_sha256": file_sha256(bundle_path), "bundle_sha256": bundle_sha},
        "selected_sampler_contract": sampler,
        "selected_sampler_contract_sha256": sampler["contract_sha256"],
        "batch_size": selected["batch_size"],
        "transition_budget_per_epoch": sampler["transition_budget_per_epoch"],
        "entry_pairs_per_epoch": sampler["entry_pairs_per_epoch"],
        "selection_uses_outcome_values": False, "test_data_used": False,
    }
    value["artifact_sha256"] = canonical_sha256(value)
    return value


def build_selected_sampler_artifact(
    *,
    benchmark_receipt_path: Path,
    candidate_set_path: Path,
    random_access_root_path: Path,
    equivalence_receipt_path: Path | None = None,
) -> dict[str, Any]:
    if equivalence_receipt_path is None:
        return _build_direct_selected_sampler(
            benchmark_receipt_path=benchmark_receipt_path,
            candidate_set_path=candidate_set_path,
            random_access_root_path=random_access_root_path,
        )
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
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_INVALID")
    direct = value.get("selection_mode") == DIRECT_SELECTION_MODE
    if "selection_mode" in value and not direct:
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_MODE_INVALID")
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
        or value.get("transition_budget_per_epoch") != (sampler["transition_budget_per_epoch"] if direct else 65536)
        or value.get("entry_pairs_per_epoch") != (sampler["entry_pairs_per_epoch"] if direct else 16384)
        or value.get("selection_uses_outcome_values") is not False
        or value.get("test_data_used") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_INVALID")
    if direct:
        if "v3_to_v4_equivalence" in value or sampler["transitions_per_entry"] != 4:
            raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_INVALID")
        if verify_files:
            for name in ("benchmark_receipt", "candidate_set", "random_access_root"):
                binding = value.get(name)
                if not isinstance(binding, Mapping):
                    raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_FILE_INVALID")
                path = Path(str(binding.get("path", "")))
                _read(path)
                if file_sha256(path) != binding.get("file_sha256"):
                    raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_FILE_INVALID")
            rebuilt = _build_direct_selected_sampler(
                benchmark_receipt_path=Path(value["benchmark_receipt"]["path"]),
                candidate_set_path=Path(value["candidate_set"]["path"]),
                random_access_root_path=Path(value["random_access_root"]["path"]),
            )
            if rebuilt != value:
                raise RuntimeError("UNIFIED_EXIT_SELECTED_SAMPLER_DIRECT_BINDING_INVALID")
        return dict(value)
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
