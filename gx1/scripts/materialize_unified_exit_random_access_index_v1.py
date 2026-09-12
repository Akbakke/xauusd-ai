#!/usr/bin/env python3
"""Publish the minimal TRAIN/VAL random-access lifecycle index."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    file_sha256,
    require_no_cap_authority,
    require_no_cap_authority_sources,
)
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
    require_split_sequence_binding,
)
from gx1.contracts.unified_exit_random_access_index_v1 import (
    RANDOM_ACCESS_INDEX_V1_COLUMNS,
    VAL_REVISION_ROOT_SCHEMA_VERSION,
    require_val_index_revision_root,
    RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION,
    RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
    build_random_access_index_v2,
    canonical_sha256,
    index_stream_sha256,
    parent_entry_mapping_sha256,
    require_random_access_index_manifest,
    require_random_access_index_root,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    schedule_random_access_epoch,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SOURCE_PATH_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SOURCE_JSON_INVALID"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_SOURCE_JSON_INVALID")
    return value


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _sealed_json(path: Path, value: dict[str, Any], key: str) -> None:
    payload = dict(value)
    payload[key] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _semantic_stream_sha256(frame: pd.DataFrame) -> str:
    columns = tuple(
        name for name in RANDOM_ACCESS_INDEX_V1_COLUMNS if name != "row_identity_sha256"
    )
    digest = __import__("hashlib").sha256()
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        digest.update(
            json.dumps(list(row), separators=(",", ":"), allow_nan=False).encode(
                "ascii"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _schedule_stream_sha256(
    *, contract: dict[str, Any], counts: np.ndarray, epochs: int
) -> str:
    digest = __import__("hashlib").sha256()
    for epoch_index in range(epochs):
        samples = schedule_random_access_epoch(
            sampler_contract=contract,
            epoch_index=epoch_index,
            successor_transition_count_by_entry=[int(value) for value in counts],
        )
        for sample in samples:
            digest.update(canonical_sha256(sample).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _build_equivalence_receipt(
    *,
    predecessor_root_path: Path,
    manifests: dict[str, dict[str, Any]],
    output_dir: Path,
    stage: Path,
    final_bundle: dict[str, Any],
) -> dict[str, Any]:
    predecessor = require_random_access_index_root(_read_json(predecessor_root_path))
    split_proofs: dict[str, Any] = {}
    train_counts: np.ndarray | None = None
    for split in ("train", "val"):
        old_path = Path(predecessor["splits"][split]["index_parquet_path"])
        old = pd.read_parquet(old_path)
        new = pd.read_parquet(stage / f"{split}.random_access_index.parquet")
        old_semantic = _semantic_stream_sha256(old)
        new_semantic = _semantic_stream_sha256(new)
        if (
            old_semantic != new_semantic
            or len(old) != len(new)
            or not np.array_equal(
                old["successor_transition_count"].to_numpy(dtype="<i8"),
                new["successor_transition_count"].to_numpy(dtype="<i8"),
            )
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{split.upper()}_SEMANTIC_DRIFT"
            )
        if split == "train":
            train_counts = new["successor_transition_count"].to_numpy(dtype="<i8")
        split_proofs[split] = {
            "row_count": len(new),
            "unchanged_semantic_columns": list(
                name
                for name in RANDOM_ACCESS_INDEX_V1_COLUMNS
                if name != "row_identity_sha256"
            ),
            "predecessor_semantic_stream_sha256": old_semantic,
            "successor_semantic_stream_sha256": new_semantic,
            "semantic_streams_byte_identical": True,
            "added_columns": ["parent_entry_row_index"],
            "successor_manifest_sha256": manifests[split]["manifest_sha256"],
        }
    if train_counts is None:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_EQUIVALENCE_INVALID")
    candidate_set = final_bundle["sampler_benchmark_candidates"]
    schedules = []
    for candidate in candidate_set["candidates"]:
        contract = candidate["sampler_contract"]
        entries_per_epoch = int(contract["entry_pairs_per_epoch"])
        cycle_epochs = -(-len(train_counts) // entries_per_epoch)
        stream = _schedule_stream_sha256(
            contract=contract, counts=train_counts, epochs=cycle_epochs
        )
        schedules.append(
            {
                "sampler_contract_sha256": contract["contract_sha256"],
                "source_lineage_sha256": contract["source_lineage_sha256"],
                "transition_budget_per_epoch": contract["transition_budget_per_epoch"],
                "population_cycle_epochs": cycle_epochs,
                "predecessor_schedule_stream_sha256": stream,
                "successor_schedule_stream_sha256": stream,
                "schedule_streams_byte_identical": True,
            }
        )
    receipt = {
        "schema_version": "gx1_unified_exit_random_access_index_v3_to_v4_equivalence_v1",
        "decision": "PASS",
        "predecessor_root": _binding(predecessor_root_path),
        "predecessor_root_sha256": predecessor["root_sha256"],
        "successor_output_dir": str(output_dir),
        "split_proofs": split_proofs,
        "sampler_candidate_set_sha256": candidate_set["candidate_set_sha256"],
        "sampled_transition_schedules": schedules,
        "only_parent_entry_coordinate_and_binding_fields_added": True,
        "benchmark_receipt_transfer_to_v4_authorized": True,
        "selection_uses_outcome_values": False,
        "test_accessed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _paths(root: Path, split: str, final_bindings_dir: Path) -> dict[str, Path]:
    suffix = split.upper()
    return {
        "entry_parquet": root / "ENTRY_WINDOW" / f"{split}.parquet",
        "entry_manifest": root / "ENTRY_WINDOW" / f"{split}.manifest.json",
        "m1_child": root / "M1_CHILD_VIEWS_V1" / f"{split}.m1.parquet",
        "m1_child_manifest": root / "M1_CHILD_VIEWS_V1" / f"{split}.manifest.json",
        "summary_manifest": root / f"SUMMARY_FIT_{suffix}_V1" / "manifest.json",
        "successor_counts": root
        / f"SUMMARY_FIT_{suffix}_V1"
        / "successor_transition_counts.npy",
        "closure_authority": root
        / f"CLOSURE_AUTHORITY_{suffix}_V2"
        / "market_closure_authority.json",
        "sequence_binding": final_bindings_dir
        / f"SPLIT_SEQUENCE_BINDING_{suffix}.json",
        "first_state_bridge": final_bindings_dir
        / f"FIRST_STATE_ENTRY_BRIDGE_{suffix}.json",
        "economic_authority": root
        / "ECONOMICS"
        / f"{split}.economic_authority.no_cap.v1.json",
        "economic_counts": root
        / "ECONOMICS"
        / f"{split}.economic_counts.no_cap.v1.json",
    }


def _build_split(
    *,
    pilot_root: Path,
    split: str,
    output_dir: Path,
    final_output_dir: Path,
    composite: dict[str, Any],
    final_bundle_path: Path,
    final_bindings_dir: Path,
    source_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    paths = _paths(pilot_root, split, final_bindings_dir)
    if source_overrides is not None:
        if split != "val" or set(source_overrides) != {
            "summary_manifest", "successor_counts", "closure_authority"
        }:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_OVERRIDE_INVALID")
        paths.update(source_overrides)
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise RuntimeError(
            f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{split.upper()}_SOURCE_MISSING"
        )
    entry_manifest = _read_json(paths["entry_manifest"])
    child_manifest = _read_json(paths["m1_child_manifest"])
    summary = _read_json(paths["summary_manifest"])
    sequence = _read_json(paths["sequence_binding"])
    bridge = _read_json(paths["first_state_bridge"])
    authority = _read_json(paths["economic_authority"])
    counts = np.load(paths["successor_counts"], allow_pickle=False)
    entries = pd.read_parquet(paths["entry_parquet"], columns=["time"])
    child = pd.read_parquet(paths["m1_child"], columns=["time", "bid_open", "ask_open"])
    parent_path = Path(str(child_manifest.get("parent_m1_path", "")))
    parent_manifest_path = Path(str(child_manifest.get("parent_m1_manifest_path", "")))
    parent_entry_binding = entry_manifest.get("source_parquet", {})
    parent_entry_manifest_binding = entry_manifest.get("source_manifest", {})
    parent_entry_path = Path(str(parent_entry_binding.get("path", "")))
    parent_entry_manifest_path = Path(
        str(parent_entry_manifest_binding.get("path", ""))
    )
    if (
        entry_manifest.get("decision") != "PASS"
        or entry_manifest.get("split") != split
        or entry_manifest.get("test_accessed") is not False
        or entry_manifest.get("output_parquet_sha256")
        != file_sha256(paths["entry_parquet"])
        or child_manifest.get("decision") != "PASS"
        or child_manifest.get("split") != split
        or child_manifest.get("test_accessed") is not False
        or child_manifest.get("output_parquet_sha256") != file_sha256(paths["m1_child"])
        or child_manifest.get("parent_m1_sha256") != file_sha256(parent_path)
        or child_manifest.get("parent_m1_manifest_sha256")
        != file_sha256(parent_manifest_path)
        or parent_entry_binding.get("sha256") != file_sha256(parent_entry_path)
        or parent_entry_manifest_binding.get("sha256")
        != file_sha256(parent_entry_manifest_path)
        or summary.get("decision") != "PASS"
        or summary.get("split") != split
        or summary.get("test_accessed") is not False
        or summary.get("entry_pair_population") != len(entries)
        or summary.get("child_parquet_sha256")
        != entry_manifest.get("output_parquet_sha256")
        or summary.get("m1_source_sha256")
        != child_manifest.get("output_parquet_sha256")
        or summary.get("successor_counts_sha256")
        != __import__("hashlib")
        .sha256(np.ascontiguousarray(counts, dtype="<i8").tobytes())
        .hexdigest()
    ):
        raise RuntimeError(
            f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{split.upper()}_LINEAGE_INVALID"
        )
    sequence = require_split_sequence_binding(
        sequence, expected_split=split, expected_entry_rows=len(entries)
    )
    if (
        sequence["binding_sha256"]
        != bridge.get("bindings", {}).get("entry_sequence_audit")
        or bridge.get("decision") != "PASS"
        or bridge.get("split") != split
        or bridge.get("entry_row_count") != len(entries)
        or bridge.get("test_accessed") is not False
        or bridge.get("normalization_mode") != "frozen_train_transform"
        or bridge.get("bindings", {}).get("train_normalization")
        != composite["composite_normalization_sha256"]
        or bridge.get("bindings", {}).get("m1_source")
        != child_manifest["output_parquet_sha256"]
    ):
        raise RuntimeError(
            f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{split.upper()}_BRIDGE_INVALID"
        )
    authority = require_no_cap_authority(
        authority,
        expected_split=split,
        expected_dataset_run_id=authority.get("dataset_run_id"),
        expected_terminal_state_counts_sha256=authority.get(
            "terminal_state_counts_sha256"
        ),
    )
    require_no_cap_authority_sources(authority)
    if authority["no_observed_economic_terminals"] is not True:
        raise RuntimeError(
            f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{split.upper()}_TERMINAL_INVALID"
        )
    parent = pd.read_parquet(parent_path, columns=["time"])
    parent_entries = pd.read_parquet(parent_entry_path, columns=["time"])
    entry_clock = pd.DatetimeIndex(pd.to_datetime(entries["time"], utc=True)).as_unit(
        "ns"
    )
    child_clock = pd.DatetimeIndex(pd.to_datetime(child["time"], utc=True)).as_unit(
        "ns"
    )
    first_state = entry_clock.asi8 + 300_000_000_000
    starts = np.searchsorted(child_clock.asi8, first_state)
    parent_entry_clock = pd.DatetimeIndex(
        pd.to_datetime(parent_entries["time"], utc=True)
    ).as_unit("ns")
    frame, parent_offset = build_random_access_index_v2(
        split=split,
        entry_times=entry_clock,
        parent_entry_times=parent_entry_clock,
        child_m1_times=child_clock,
        parent_m1_times=pd.DatetimeIndex(pd.to_datetime(parent["time"], utc=True)),
        successor_transition_counts=counts,
        entry_bid=child["bid_open"].to_numpy()[starts],
        entry_ask=child["ask_open"].to_numpy()[starts],
        episode_binding_sha256_by_entry=bridge[
            "first_state_episode_binding_sha256_by_entry"
        ],
        entry_fill_binding_sha256_by_entry=bridge["entry_fill_binding_sha256_by_entry"],
    )
    parent_entry_rows = np.ascontiguousarray(
        frame["parent_entry_row_index"].to_numpy(dtype="<i8")
    )
    parent_entry_rows_sha256 = (
        __import__("hashlib").sha256(parent_entry_rows.tobytes()).hexdigest()
    )
    child_entry_clock_sha256 = (
        __import__("hashlib")
        .sha256(np.ascontiguousarray(entry_clock.asi8, dtype="<i8").tobytes())
        .hexdigest()
    )
    if parent_entry_rows_sha256 != entry_manifest.get(
        "source_row_indices_sha256"
    ) or child_entry_clock_sha256 != entry_manifest.get("clock_sha256"):
        raise RuntimeError(
            f"UNIFIED_EXIT_RANDOM_ACCESS_INDEX_{split.upper()}_PARENT_ENTRY_MAPPING_INVALID"
        )
    parquet_path = output_dir / f"{split}.random_access_index.parquet"
    frame.to_parquet(parquet_path, index=False)
    source_bindings = {name: _binding(path) for name, path in paths.items()}
    source_bindings["parent_m1"] = _binding(parent_path)
    source_bindings["parent_m1_manifest"] = _binding(parent_manifest_path)
    source_bindings["parent_entry_parquet"] = _binding(parent_entry_path)
    source_bindings["parent_entry_manifest"] = _binding(parent_entry_manifest_path)
    source_bindings["composite_normalization"] = _binding(
        final_bindings_dir / "COMPOSITE_NORMALIZATION.json"
    )
    source_bindings["final_bindings_bundle"] = _binding(final_bundle_path)
    manifest = {
        "schema_version": RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
        "decision": "PASS",
        "split": split,
        "dataset_run_id": authority["dataset_run_id"],
        "entry_row_count": len(frame),
        "successor_transition_total": int(counts.sum(dtype=np.int64)),
        "parent_m1_row_offset": parent_offset,
        "parent_m1_source_sha256": child_manifest["parent_m1_sha256"],
        "m1_source_sha256": child_manifest["output_parquet_sha256"],
        "entry_binding_sha256": sequence["binding_sha256"],
        "child_entry_clock_sha256": child_entry_clock_sha256,
        "parent_entry_clock_sha256": __import__("hashlib")
        .sha256(np.ascontiguousarray(parent_entry_clock.asi8, dtype="<i8").tobytes())
        .hexdigest(),
        "parent_entry_source_rows": len(parent_entry_clock),
        "parent_entry_source_sha256": parent_entry_binding["sha256"],
        "parent_entry_manifest_sha256": parent_entry_manifest_binding["sha256"],
        "parent_entry_row_indices_sha256": parent_entry_rows_sha256,
        "parent_entry_mapping_sha256": parent_entry_mapping_sha256(frame),
        "gap_classification_source_sha256": summary["closure_authority_sha256"],
        "split_end_utc": authority["coverage_end_utc"],
        "index_parquet_path": str(
            final_output_dir / f"{split}.random_access_index.parquet"
        ),
        "index_parquet_sha256": file_sha256(parquet_path),
        "index_stream_sha256": index_stream_sha256(frame),
        "episode_binding_stream_sha256": bridge["bridge_stream_sha256"],
        "entry_fill_binding_stream_sha256": bridge["entry_fill_binding_stream_sha256"],
        "sequence_binding_sha256": sequence["binding_sha256"],
        "composite_normalization_sha256": composite["composite_normalization_sha256"],
        "economic_authority_sha256": authority["authority_sha256"],
        "economic_terminal_count": 0,
        "split_end_is_right_censor": True,
        "storage_granularity": "one_row_per_entry",
        "full_prefix_states_stored": False,
        "chunk_pointers_stored": False,
        "target_q_stored": False,
        "source_bindings": source_bindings,
        "test_accessed": False,
    }
    _sealed_json(output_dir / f"{split}.manifest.json", manifest, "manifest_sha256")
    manifest = _read_json(output_dir / f"{split}.manifest.json")
    require_random_access_index_manifest(
        manifest,
        expected_split=split,
        index_frame=frame,
    )
    return manifest


def publish(
    *,
    pilot_root: Path,
    output_dir: Path,
    final_bindings_dir: Path | None = None,
    predecessor_root_path: Path,
) -> dict[str, Any]:
    pilot_root = pilot_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    bindings_dir = (
        final_bindings_dir.expanduser().resolve()
        if final_bindings_dir is not None
        else pilot_root / "FINAL_BINDINGS_V1"
    )
    composite_path = bindings_dir / "COMPOSITE_NORMALIZATION.json"
    final_bundle_path = bindings_dir / "FINAL_BINDINGS_BUNDLE.json"
    composite = require_composite_normalization_binding(_read_json(composite_path))
    final_bundle = _read_json(final_bundle_path)
    if (
        final_bundle.get("schema_version")
        != "gx1_unified_exit_pilot_final_bindings_bundle_v1"
        or final_bundle.get("test_accessed") is not False
        or final_bundle.get("bundle_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in final_bundle.items()
                if key != "bundle_sha256"
            }
        )
        or final_bundle.get("composite_normalization", {}).get(
            "composite_normalization_sha256"
        )
        != composite["composite_normalization_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_FINAL_BINDINGS_INVALID")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        manifests = {
            split: _build_split(
                pilot_root=pilot_root,
                split=split,
                output_dir=stage,
                final_output_dir=output_dir,
                composite=composite,
                final_bundle_path=final_bundle_path,
                final_bindings_dir=bindings_dir,
            )
            for split in ("train", "val")
        }
        predecessor_path = predecessor_root_path.expanduser().resolve()
        equivalence = _build_equivalence_receipt(
            predecessor_root_path=predecessor_path,
            manifests=manifests,
            output_dir=output_dir,
            stage=stage,
            final_bundle=final_bundle,
        )
        _sealed_json(
            stage / "V3_TO_V4_EQUIVALENCE.json",
            {
                key: value
                for key, value in equivalence.items()
                if key != "receipt_sha256"
            },
            "receipt_sha256",
        )
        equivalence = _read_json(stage / "V3_TO_V4_EQUIVALENCE.json")
        root = {
            "schema_version": RANDOM_ACCESS_INDEX_V2_ROOT_SCHEMA_VERSION,
            "decision": "PASS",
            "allowed_splits": ["train", "val"],
            "storage_granularity": "one_row_per_entry",
            "full_prefix_states_stored": False,
            "chunk_pointers_stored": False,
            "composite_normalization_sha256": composite[
                "composite_normalization_sha256"
            ],
            "final_bindings_bundle_sha256": final_bundle["bundle_sha256"],
            "sampler_selection_status": final_bundle["sampler_benchmark_candidates"][
                "decision"
            ],
            "selected_sampler_contract_sha256": final_bundle[
                "sampler_benchmark_candidates"
            ]["selected_sampler_contract_sha256"],
            "predecessor_equivalence": {
                "path": str(output_dir / "V3_TO_V4_EQUIVALENCE.json"),
                "sha256": file_sha256(stage / "V3_TO_V4_EQUIVALENCE.json"),
                "receipt_sha256": equivalence["receipt_sha256"],
                "benchmark_receipt_transfer_to_v4_authorized": True,
            },
            "splits": {
                split: {
                    "index_parquet_path": str(
                        output_dir / f"{split}.random_access_index.parquet"
                    ),
                    "index_parquet_sha256": manifests[split]["index_parquet_sha256"],
                    "manifest_path": str(output_dir / f"{split}.manifest.json"),
                    "manifest_sha256": manifests[split]["manifest_sha256"],
                    "entry_row_count": manifests[split]["entry_row_count"],
                    "successor_transition_total": manifests[split][
                        "successor_transition_total"
                    ],
                }
                for split in ("train", "val")
            },
            "test_accessed": False,
        }
        _sealed_json(stage / "ROOT.json", root, "root_sha256")
        require_random_access_index_root(_read_json(stage / "ROOT.json"))
        if output_dir.exists():
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_OUTPUT_EXISTS")
        os.replace(stage, output_dir)
        published_root = require_random_access_index_root(
            _read_json(output_dir / "ROOT.json")
        )
        for split in ("train", "val"):
            published_index = output_dir / f"{split}.random_access_index.parquet"
            published_frame = pd.read_parquet(published_index)
            require_random_access_index_manifest(
                _read_json(output_dir / f"{split}.manifest.json"),
                expected_split=split,
                index_frame=published_frame,
                index_path=published_index,
                verify_sources=True,
            )
        return published_root
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def publish_val_revision(
    *, pilot_root: Path, output_dir: Path, final_bindings_dir: Path,
    predecessor_root_path: Path,
) -> dict[str, Any]:
    """Rebuild only corrected VAL lifecycle bindings; preserve the seed TRAIN root."""
    pilot_root = pilot_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    bindings_dir = final_bindings_dir.expanduser().resolve()
    predecessor_path = predecessor_root_path.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_OUTPUT_EXISTS")
    predecessor_binding = _binding(predecessor_path)
    old = require_random_access_index_root(_read_json(predecessor_path))
    composite_path = bindings_dir / "COMPOSITE_NORMALIZATION.json"
    bundle_path = bindings_dir / "FINAL_BINDINGS_BUNDLE.json"
    composite = require_composite_normalization_binding(_read_json(composite_path))
    bundle = _read_json(bundle_path)
    recipe_path = Path(bundle["recipe_path"])
    recipe = _read_json(recipe_path)
    if (file_sha256(recipe_path) != bundle["recipe_file_sha256"]
            or recipe["recipe_sha256"] != bundle["recipe_sha256"]
            or recipe["recipe_sha256"] != canonical_sha256(
                {k: v for k, v in recipe.items() if k != "recipe_sha256"})
            or bundle.get("decision") != "BLOCKED_PENDING_TRAIN_ONLY_SAMPLER_BENCHMARK"
            or bundle.get("val_fit_rows") != 0
            or bundle.get("test_fit_rows") != 0
            or bundle.get("test_accessed") is not False
            or composite["composite_normalization_sha256"] != old["composite_normalization_sha256"]):
        raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_RECIPE_INVALID")
    overrides = {}
    for name in ("summary_manifest", "successor_counts", "closure_authority"):
        binding = recipe["splits"]["val"][name]
        path = Path(binding["path"])
        if not path.is_absolute() or path.is_symlink() or _binding(path) != binding:
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_SOURCE_INVALID")
        overrides[name] = path
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        manifest = _build_split(
            pilot_root=pilot_root, split="val", output_dir=stage,
            final_output_dir=output_dir, composite=composite,
            final_bundle_path=bundle_path, final_bindings_dir=bindings_dir,
            source_overrides=overrides,
        )
        frame = pd.read_parquet(stage / "val.random_access_index.parquet")
        require_random_access_index_manifest(
            manifest, expected_split="val", index_frame=frame, verify_sources=True,
        )
        old_manifest = require_random_access_index_manifest(
            _read_json(Path(old["splits"]["val"]["manifest_path"])), expected_split="val",
        )
        old_path = Path(old["splits"]["val"]["index_parquet_path"])
        old_frame = pd.read_parquet(old_path)
        require_random_access_index_manifest(
            old_manifest, expected_split="val", index_frame=old_frame,
            index_path=old_path, verify_sources=True,
        )
        changed_columns = {
            "successor_transition_count", "lifecycle_state_count",
            # The fill witness includes the episode hash, which binds the calendar.
            # Compare actual entry quotes and all row coordinates below.
            "episode_binding_sha256", "entry_fill_binding_sha256", "row_identity_sha256",
        }
        unchanged_columns = [name for name in frame.columns if name not in changed_columns]
        if (len(frame) != 5508 or list(frame.columns) != list(old_frame.columns)
                or not frame[unchanged_columns].equals(old_frame[unchanged_columns])
                or (frame["successor_transition_count"] < old_frame["successor_transition_count"]).any()):
            raise RuntimeError("UNIFIED_EXIT_VAL_REVISION_COORDINATE_OR_FILL_DRIFT")
        root = {k: v for k, v in old.items()
                if k not in {"root_sha256", "predecessor_equivalence"}}
        root.update(
            schema_version=VAL_REVISION_ROOT_SCHEMA_VERSION,
            final_bindings_bundle_sha256=bundle["bundle_sha256"],
            val_data_revision={"predecessor_root": predecessor_binding,
                               "predecessor_root_sha256": old["root_sha256"],
                               "changed_split": "val"},
            splits={"train": old["splits"]["train"], "val": {
                "index_parquet_path": str(output_dir / "val.random_access_index.parquet"),
                "index_parquet_sha256": manifest["index_parquet_sha256"],
                "manifest_path": str(output_dir / "val.manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "entry_row_count": manifest["entry_row_count"],
                "successor_transition_total": manifest["successor_transition_total"],
            }},
        )
        # Validate the exact staged manifest before the atomic directory publish.
        staged_root = json.loads(json.dumps(root))
        staged_root["splits"]["val"]["manifest_path"] = str(stage / "val.manifest.json")
        staged_root["root_sha256"] = canonical_sha256(staged_root)
        require_val_index_revision_root(staged_root, expected_predecessor=predecessor_binding)
        _sealed_json(stage / "ROOT.json", root, "root_sha256")
        require_random_access_index_root(_read_json(stage / "ROOT.json"))
        os.replace(stage, output_dir)
        return require_val_index_revision_root(
            _read_json(output_dir / "ROOT.json"), expected_predecessor=predecessor_binding,
        )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--final-bindings-dir", type=Path)
    parser.add_argument("--predecessor-root", type=Path, required=True)
    parser.add_argument("--val-only-revision", action="store_true")
    args = parser.parse_args()
    if args.val_only_revision and args.final_bindings_dir is None:
        parser.error("VAL revision requires --final-bindings-dir")
    print(
        json.dumps(
            (publish_val_revision if args.val_only_revision else publish)(
                pilot_root=args.pilot_root,
                output_dir=args.output_dir,
                final_bindings_dir=args.final_bindings_dir,
                predecessor_root_path=args.predecessor_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
