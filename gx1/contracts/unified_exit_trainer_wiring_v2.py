"""Canonical artifact factory for lifecycle-v2 TRAIN and VAL datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_dataset_adapter_v2 import UnifiedExitDatasetAdapterV2
from gx1.contracts.unified_exit_economic_step_provider_v1 import (
    LazyUnifiedExitEconomicStepProviderV1,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    canonical_sha256,
    file_sha256,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    COMPACT_ROOT_SCHEMA_VERSION,
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"UNIFIED_EXIT_V2_WIRING_{label}_PATH_INVALID")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_V2_WIRING_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"UNIFIED_EXIT_V2_WIRING_{label}_INVALID")
    return value


def _load_random_access_train_artifacts(
    *, paths: Mapping[str, Path], source_owner: Any
) -> dict[str, Any]:
    required = {
        "sampler_contract",
        "summary_fit_manifest",
        "successor_counts",
        "m1_child",
        "m1_child_manifest",
        "market_closure_authority",
        "first_state_bridge",
        "state_view_source",
    }
    if set(paths) != required:
        raise RuntimeError("UNIFIED_EXIT_V2_WIRING_RANDOM_ACCESS_PATHS_INVALID")
    resolved = {name: path.expanduser().resolve() for name, path in paths.items()}
    if any(not path.is_file() or path.is_symlink() for path in resolved.values()):
        raise RuntimeError("UNIFIED_EXIT_V2_WIRING_RANDOM_ACCESS_PATHS_INVALID")
    summary = _read_json(resolved["summary_fit_manifest"], "TRAIN_SUMMARY_FIT")
    child_manifest = _read_json(resolved["m1_child_manifest"], "TRAIN_M1_CHILD")
    counts = np.load(resolved["successor_counts"], allow_pickle=False)
    if (
        counts.dtype != np.dtype("<i8")
        or hashlib.sha256(counts.tobytes()).hexdigest()
        != summary.get("successor_counts_sha256")
        or file_sha256(resolved["m1_child"]) != summary.get("m1_source_sha256")
        or file_sha256(resolved["m1_child_manifest"])
        != summary.get("m1_manifest_sha256")
        or child_manifest.get("output_parquet_sha256")
        != summary.get("m1_source_sha256")
        or child_manifest.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_WIRING_RANDOM_ACCESS_SOURCE_INVALID")
    child_clock = pd.DatetimeIndex(
        pd.to_datetime(
            pd.read_parquet(resolved["m1_child"], columns=["time"])["time"],
            utc=True,
            errors="coerce",
        )
    ).as_unit("ns")
    parent_clock = pd.DatetimeIndex(source_owner._m1_times).as_unit("ns")
    parent_offset = int(np.searchsorted(parent_clock.asi8, child_clock.asi8[0]))
    stop = parent_offset + len(child_clock)
    if (
        child_clock.empty
        or child_clock.hasnans
        or stop > len(parent_clock)
        or not parent_clock[parent_offset:stop].equals(child_clock)
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_WIRING_RANDOM_ACCESS_CLOCK_INVALID")
    return {
        "resolved": resolved,
        "sampler_contract": _read_json(resolved["sampler_contract"], "TRAIN_SAMPLER"),
        "summary_fit_manifest": summary,
        "successor_counts": counts,
        "m1_child_manifest": child_manifest,
        "m1_child_times": child_clock,
        "market_closure_authority": _read_json(
            resolved["market_closure_authority"], "TRAIN_CLOSURE"
        ),
        "first_state_bridge": _read_json(
            resolved["first_state_bridge"], "TRAIN_FIRST_STATE_BRIDGE"
        ),
        "parent_m1_row_offset": parent_offset,
        "state_view_source_sha256": file_sha256(resolved["state_view_source"]),
    }


def _require_root(
    *,
    root_path: Path,
    dataset_run_id: str,
    compact_paths: Mapping[str, Path],
    manifest_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    root = _read_json(root_path, "ROOT_MANIFEST")
    if (
        root.get("schema_version") != COMPACT_ROOT_SCHEMA_VERSION
        or root.get("decision") != "PASS"
        or root.get("dataset_run_id") != dataset_run_id
        or root.get("allowed_splits") != ["train", "val"]
        or root.get("test_accessed") is not False
        or root.get("target_q_stored") is not False
        or root.get("chunk_state_capacity") != 512
        or root.get("manifest_sha256")
        != canonical_sha256(
            {key: value for key, value in root.items() if key != "manifest_sha256"}
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_WIRING_ROOT_MANIFEST_INVALID")
    root_splits = root.get("split_manifests")
    if not isinstance(root_splits, Mapping) or set(root_splits) != {"train", "val"}:
        raise RuntimeError("UNIFIED_EXIT_V2_WIRING_ROOT_MANIFEST_INVALID")
    manifests: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        compact_path = compact_paths[split].expanduser().resolve()
        manifest_path = manifest_paths[split].expanduser().resolve()
        manifest = _read_json(manifest_path, f"{split.upper()}_MANIFEST")
        binding = root_splits[split]
        if (
            not isinstance(binding, Mapping)
            or manifest.get("split") != split
            or manifest.get("dataset_run_id") != dataset_run_id
            or manifest.get("test_accessed") is not False
            or manifest.get("target_q_stored") is not False
            or manifest.get("compact_parquet_sha256") != file_sha256(compact_path)
            or binding.get("manifest_sha256") != manifest.get("manifest_sha256")
            or binding.get("compact_parquet_sha256")
            != manifest.get("compact_parquet_sha256")
            or binding.get("compact_rows") != manifest.get("compact_rows")
            or manifest.get("manifest_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
        ):
            raise RuntimeError(f"UNIFIED_EXIT_V2_WIRING_{split.upper()}_INVALID")
        try:
            frame = pd.read_parquet(compact_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"UNIFIED_EXIT_V2_WIRING_{split.upper()}_COMPACT_INVALID"
            ) from exc
        if len(frame) != int(manifest["compact_rows"]):
            raise RuntimeError(
                f"UNIFIED_EXIT_V2_WIRING_{split.upper()}_COMPACT_INVALID"
            )
        manifests[split] = manifest
        frames[split] = frame
    return root, manifests, frames


def bind_canonical_unified_exit_lifecycle_v2(
    *,
    dataset_run_id: str,
    root_manifest_path: Path,
    compact_paths: Mapping[str, Path],
    compact_manifest_paths: Mapping[str, Path],
    economics_readiness_path: Path,
    cost_authority_paths: Mapping[str, Path],
    datasets: Mapping[str, Any],
    feature_source_owners: Mapping[str, Any],
    random_access_train_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Construct and bind both split adapters before any DataLoader exists."""

    if (
        not dataset_run_id
        or set(compact_paths) != {"train", "val"}
        or set(compact_manifest_paths) != {"train", "val"}
        or set(cost_authority_paths) != {"train", "val"}
        or set(datasets) != {"train", "val"}
        or set(feature_source_owners) != {"train", "val"}
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_WIRING_INVOCATION_INVALID")
    root, manifests, frames = _require_root(
        root_path=root_manifest_path,
        dataset_run_id=dataset_run_id,
        compact_paths=compact_paths,
        manifest_paths=compact_manifest_paths,
    )
    readiness_path = economics_readiness_path.expanduser().resolve()
    readiness = require_unified_exit_unbounded_training_readiness(
        _read_json(readiness_path, "ECONOMICS_READINESS"),
        context="UNIFIED_EXIT_V2_TRAINER_WIRING",
    )
    random_train = (
        _load_random_access_train_artifacts(
            paths=random_access_train_paths,
            source_owner=feature_source_owners["train"],
        )
        if random_access_train_paths is not None
        else None
    )
    adapters: dict[str, UnifiedExitDatasetAdapterV2] = {}
    split_evidence: dict[str, Any] = {}
    for split in ("train", "val"):
        dataset = datasets[split]
        source = feature_source_owners[split]
        if (
            getattr(dataset, "_unified_exit_lifecycle", None) is not None
            or getattr(dataset, "_unified_exit_lifecycle_v2", None) is not None
        ):
            raise RuntimeError("UNIFIED_EXIT_V2_WIRING_LEGACY_OR_DUPLICATE_BINDING")
        provider_options: dict[str, Any] = {}
        if split == "train" and random_train is not None:
            provider_options = {
                "market_closure_authority_path": random_train["resolved"][
                    "market_closure_authority"
                ],
                "market_closure_authority_file_sha256": file_sha256(
                    random_train["resolved"]["market_closure_authority"]
                ),
                "state_m1_source_path": random_train["resolved"]["m1_child"],
                "state_m1_source_manifest_path": random_train["resolved"][
                    "m1_child_manifest"
                ],
                "state_m1_source_file_sha256": file_sha256(
                    random_train["resolved"]["m1_child"]
                ),
                "state_m1_source_manifest_file_sha256": file_sha256(
                    random_train["resolved"]["m1_child_manifest"]
                ),
                "parent_m1_row_offset": random_train["parent_m1_row_offset"],
                "common_successor_transition_counts": random_train["successor_counts"],
                "expected_successor_counts_sha256": random_train[
                    "summary_fit_manifest"
                ]["successor_counts_sha256"],
            }
        provider = LazyUnifiedExitEconomicStepProviderV1(
            compact_rows=frames[split],
            compact_manifest=manifests[split],
            economics_readiness=readiness,
            cost_parameter_authority_path=cost_authority_paths[split],
            **provider_options,
        )
        adapter = UnifiedExitDatasetAdapterV2(
            compact_rows=frames[split],
            compact_manifest=manifests[split],
            source_owner=source,
            epoch_index=0,
            expected_m1_source_sha256=manifests[split]["m1_source_sha256"],
            expected_entry_binding_sha256=manifests[split]["entry_binding_sha256"],
            expected_gap_classification_source_sha256=manifests[split][
                "gap_classification_source_sha256"
            ],
            economics_readiness=readiness,
            economic_exit_step_manifest=provider.economic_exit_step_manifest,
            economic_exit_step_provider=provider,
            mtf_materializer=dataset._get_exit_multi_tf_episode_histories,
            per_tf_seq_lens=dataset.per_tf_seq_lens,
            mtf_cache_identity_sha256=dataset._multi_tf_cache_identity_sha256,
        )
        if split == "train" and random_train is not None:
            adapter.configure_random_access_training_v1(
                sampler_contract=random_train["sampler_contract"],
                successor_transition_counts=random_train["successor_counts"],
                summary_fit_manifest=random_train["summary_fit_manifest"],
                market_closure_authority=random_train["market_closure_authority"],
                normalization_artifact=random_train["summary_fit_manifest"][
                    "lifetime_summary_normalization"
                ],
                first_state_bridge_witness=random_train["first_state_bridge"],
                random_access_m1_times=random_train["m1_child_times"],
                parent_m1_row_offset=random_train["parent_m1_row_offset"],
                expected_child_parquet_sha256=random_train["summary_fit_manifest"][
                    "child_parquet_sha256"
                ],
                expected_state_view_source_sha256=random_train[
                    "state_view_source_sha256"
                ],
            )
        dataset.bind_unified_exit_lifecycle_v2(adapter)
        if (
            getattr(dataset, "_unified_exit_lifecycle", None) is not None
            or getattr(dataset, "_unified_exit_lifecycle_v2", None) is not adapter
        ):
            raise RuntimeError("UNIFIED_EXIT_V2_WIRING_BINDING_INVALID")
        adapters[split] = adapter
        split_evidence[split] = {
            "compact_rows": len(frames[split]),
            "compact_manifest_sha256": manifests[split]["manifest_sha256"],
            "economic_step_manifest_sha256": provider.economic_exit_step_manifest[
                "manifest_sha256"
            ],
            "legacy_exit_target_bound": False,
            "epoch_index": 0,
            "validation_dispatch": (
                "deterministic_full_both_sides_all_chunks" if split == "val" else None
            ),
            "random_access_train_configured": (
                split == "train" and random_train is not None
            ),
        }
    return {
        "schema_version": "gx1_unified_exit_trainer_wiring_v2",
        "decision": "PASS",
        "dataset_run_id": dataset_run_id,
        "root_manifest_path": str(root_manifest_path.expanduser().resolve()),
        "root_manifest_sha256": root["manifest_sha256"],
        "economics_readiness_path": str(readiness_path),
        "economics_readiness_file_sha256": file_sha256(readiness_path),
        "economics_objective_contract_sha256": readiness[
            "economics_objective_contract"
        ]["contract_sha256"],
        "splits": split_evidence,
        "v2_active_before_loader": True,
        "legacy_exit_target_rejected": True,
        "test_data_used": False,
    }


__all__ = ("bind_canonical_unified_exit_lifecycle_v2",)
