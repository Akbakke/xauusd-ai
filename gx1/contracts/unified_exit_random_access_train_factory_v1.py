"""Production TRAIN factory over the native O(Entry) lifecycle index."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
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
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import file_sha256
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    BENCHMARK_BUDGETS,
    BENCHMARK_CANDIDATE_SET_SCHEMA_VERSION,
    canonical_sha256,
)
from gx1.contracts.unified_exit_random_access_index_v1 import (
    require_random_access_index_manifest,
    require_random_access_index_root,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    require_random_access_sampler_contract,
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_{label}_PATH_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_{label}_INVALID"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_{label}_INVALID")
    return value


def _source_path(manifest: Mapping[str, Any], key: str) -> Path:
    binding = manifest.get("source_bindings", {}).get(key)
    if not isinstance(binding, Mapping):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_SOURCE_BINDING_INVALID")
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or file_sha256(path) != binding.get("sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_SOURCE_BINDING_INVALID")
    return path


def _candidate_contracts(path: Path) -> dict[int, dict[str, Any]]:
    candidate_set = _read_json(path, "CANDIDATE_SET")
    data = dict(candidate_set)
    claimed = data.pop("candidate_set_sha256", None)
    candidates = candidate_set.get("candidates")
    if (
        candidate_set.get("schema_version") != BENCHMARK_CANDIDATE_SET_SCHEMA_VERSION
        or candidate_set.get("decision") != "BLOCKED_PENDING_TRAIN_ONLY_BENCHMARK"
        or candidate_set.get("selected_sampler_contract_sha256") is not None
        or candidate_set.get("selection_requires_measured_throughput_and_memory")
        is not True
        or candidate_set.get("selection_uses_outcome_values") is not False
        or candidate_set.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
        or not isinstance(candidates, list)
        or len(candidates) != len(BENCHMARK_BUDGETS)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_CANDIDATES_INVALID")
    by_budget: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("status") != "BENCHMARK_PENDING"
            or candidate.get("selected") is not False
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_CANDIDATES_INVALID")
        contract = require_random_access_sampler_contract(
            candidate.get("sampler_contract", {})
        )
        by_budget[int(contract["transition_budget_per_epoch"])] = contract
    if set(by_budget) != set(BENCHMARK_BUDGETS):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_CANDIDATES_INVALID")
    return by_budget


def build_random_access_train_adapter_factory_v1(
    *,
    root_manifest_path: Path,
    candidate_set_path: Path,
    composite_normalization_path: Path,
    economics_readiness_path: Path,
    train_cost_authority_path: Path,
    train_dataset: Any,
    train_feature_source_owner: Any,
    backup_steps: int = 1,
    reference_policy: Mapping[str, Any] | None = None,
    reference_cutoff_time_ns: int | None = None,
) -> Callable[[int], UnifiedExitDatasetAdapterV2]:
    """Return real candidate adapters without admitting an unbenchmarked sampler."""

    policy = None
    if reference_policy is not None:
        from gx1.contracts.unified_exit_reference_policy_v1 import require_reference_policy_contract
        policy = require_reference_policy_contract(reference_policy)
        if backup_steps != 1:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_REFERENCE_SCOPE_INVALID")
    if type(backup_steps) is not int or backup_steps not in (1, 5):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BACKUP_STEPS_INVALID")
    if reference_cutoff_time_ns is not None and (
            type(reference_cutoff_time_ns) is not int or reference_cutoff_time_ns <= 0 or policy is None):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_REFERENCE_CUTOFF_INVALID")
    root_path = root_manifest_path.expanduser().resolve()
    root = require_random_access_index_root(_read_json(root_path, "ROOT"))
    split_binding = root["splits"]["train"]
    index_path = Path(split_binding["index_parquet_path"])
    manifest_path = Path(split_binding["manifest_path"])
    if file_sha256(index_path) != split_binding["index_parquet_sha256"]:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_INDEX_HASH_INVALID")
    frame = pd.read_parquet(index_path)
    manifest = require_random_access_index_manifest(
        _read_json(manifest_path, "TRAIN_MANIFEST"),
        expected_split="train",
        index_frame=frame,
        index_path=index_path,
        verify_sources=True,
    )
    if (
        manifest["manifest_sha256"] != split_binding["manifest_sha256"]
        or manifest["index_parquet_sha256"] != split_binding["index_parquet_sha256"]
        or root["selected_sampler_contract_sha256"] is not None
        or root["sampler_selection_status"] != "BLOCKED_PENDING_TRAIN_ONLY_BENCHMARK"
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_ROOT_BINDING_INVALID")
    contracts = _candidate_contracts(candidate_set_path)
    composite = require_composite_normalization_binding(
        _read_json(composite_normalization_path, "COMPOSITE_NORMALIZATION")
    )
    if (
        composite["composite_normalization_sha256"]
        != root["composite_normalization_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_NORMALIZATION_INVALID")
    readiness = _read_json(economics_readiness_path, "ECONOMICS_READINESS")
    require_unified_exit_unbounded_training_readiness(
        readiness,
        context="UNIFIED_EXIT_RANDOM_ACCESS_FACTORY",
    )
    summary_path = _source_path(manifest, "summary_manifest")
    counts_path = _source_path(manifest, "successor_counts")
    child_path = _source_path(manifest, "m1_child")
    child_manifest_path = _source_path(manifest, "m1_child_manifest")
    closure_path = _source_path(manifest, "closure_authority")
    bridge_path = _source_path(manifest, "first_state_bridge")
    final_bundle_path = _source_path(manifest, "final_bindings_bundle")
    final_bundle = _read_json(final_bundle_path, "FINAL_BINDINGS")
    final_bundle_data = dict(final_bundle)
    claimed_bundle = final_bundle_data.pop("bundle_sha256", None)
    state_view_binding = final_bundle.get("state_view_source")
    if (
        claimed_bundle != root["final_bindings_bundle_sha256"]
        or claimed_bundle != canonical_sha256(final_bundle_data)
        or not isinstance(state_view_binding, Mapping)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_FINAL_BINDINGS_INVALID")
    state_view_source_path = Path(str(state_view_binding.get("path", "")))
    if (
        not state_view_source_path.is_file()
        or state_view_source_path.is_symlink()
        or file_sha256(state_view_source_path) != state_view_binding.get("sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_STATE_VIEW_INVALID")
    summary = _read_json(summary_path, "SUMMARY")
    closure = _read_json(closure_path, "CLOSURE")
    bridge = _read_json(bridge_path, "FIRST_STATE_BRIDGE")
    counts = np.load(counts_path, allow_pickle=False)
    child_times = pd.DatetimeIndex(
        pd.to_datetime(
            pd.read_parquet(child_path, columns=["time"])["time"],
            utc=True,
            errors="coerce",
        )
    ).as_unit("ns")
    provider_rows = pd.DataFrame(
        {
            "entry_row_index": frame["entry_row_index"],
            "entry_m1_start_row": frame["parent_m1_start_row"],
            "m1_source_sha256": manifest["parent_m1_source_sha256"],
            "long_lifecycle_state_count": frame["lifecycle_state_count"],
            "short_lifecycle_state_count": frame["lifecycle_state_count"],
            "long_economic_terminal": frame["economic_terminal"],
            "short_economic_terminal": frame["economic_terminal"],
        }
    )

    def factory(budget: int) -> UnifiedExitDatasetAdapterV2:
        if budget not in contracts:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FACTORY_BUDGET_INVALID")
        provider = LazyUnifiedExitEconomicStepProviderV1(
            compact_rows=provider_rows,
            compact_manifest=manifest,
            economics_readiness=readiness,
            cost_parameter_authority_path=train_cost_authority_path,
            market_closure_authority_path=closure_path,
            market_closure_authority_file_sha256=file_sha256(closure_path),
            state_m1_source_path=child_path,
            state_m1_source_manifest_path=child_manifest_path,
            state_m1_source_file_sha256=file_sha256(child_path),
            state_m1_source_manifest_file_sha256=file_sha256(child_manifest_path),
            parent_m1_row_offset=manifest["parent_m1_row_offset"],
            common_successor_transition_counts=counts,
            expected_successor_counts_sha256=summary["successor_counts_sha256"],
        )
        adapter = UnifiedExitDatasetAdapterV2.from_random_access_index_v1(
            index_rows=frame,
            index_manifest=manifest,
            source_owner=train_feature_source_owner,
            epoch_index=0,
            economics_readiness=readiness,
            economic_exit_step_manifest=provider.economic_exit_step_manifest,
            economic_exit_step_provider=provider,
            mtf_materializer=train_dataset._get_exit_multi_tf_episode_histories,
            per_tf_seq_lens=train_dataset.per_tf_seq_lens,
            mtf_cache_identity_sha256=train_dataset._multi_tf_cache_identity_sha256,
        )
        adapter.configure_random_access_training_v1(
            backup_steps=backup_steps,
            reference_policy=policy,
            reference_cutoff_time_ns=reference_cutoff_time_ns,
            sampler_contract=contracts[budget],
            successor_transition_counts=counts,
            summary_fit_manifest=summary,
            market_closure_authority=closure,
            normalization_artifact=composite["lifetime_summary_normalization"],
            first_state_bridge_witness=bridge,
            random_access_m1_times=child_times,
            parent_m1_row_offset=manifest["parent_m1_row_offset"],
            expected_child_parquet_sha256=summary["child_parquet_sha256"],
            expected_state_view_source_sha256=file_sha256(state_view_source_path),
            expected_composite_normalization_sha256=composite[
                "composite_normalization_sha256"
            ],
        )
        return adapter

    return factory


__all__ = ("build_random_access_train_adapter_factory_v1",)
