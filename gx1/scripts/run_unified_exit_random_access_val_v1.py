"""Strict weight-EMA full-cohort VAL executor for random-access Exit v2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import (
    build_entry_policy_decisions,
)

from gx1.contracts.local_random_access_campaign_v2 import (
    PROGRESS_SCHEMA,
    canonical_sha256 as campaign_sha256,
    read_bound_json,
    require_plan,
    require_progress as require_campaign_progress,
)
from gx1.contracts.unified_exit_economic_step_provider_v1 import (
    LazyUnifiedExitEconomicStepProviderV1,
)
from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleCorpus
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_random_access_cuda_smoke_v1 import (
    require_bootstrap_composite_normalization,
)
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    canonical_sha256,
    file_sha256,
)
from gx1.contracts.unified_exit_random_access_index_v1 import (
    require_random_access_index_manifest,
    require_random_access_index_root,
    require_parent_entry_coordinate_equivalence,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    bind_preserved_v7_input_normalization,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    schedule_random_access_entry_anchors,
)
from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
    load_selected_weight_ema_checkpoint_readonly_v1,
)
from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
    accumulate_route_diagnostics_v1,
    finalize_route_diagnostics_v1,
    run_resumable_random_access_val_evaluation_v1,
)
from gx1.contracts.unified_exit_random_access_val_factory_v1 import (
    RandomAccessValStateFactoryV1,
)
from gx1.contracts.unified_exit_selected_sampler_v1 import (
    require_selected_sampler_artifact,
)
from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
    require_final_train_checkpoint_authority,
    FULL_POPULATION_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset,
    _model_forward_fp32,
    _multi_tf_kwargs_from_batch,
)
from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import (
    _model,
    require_launch_manifest,
)


class _ExactSampler(Sampler[int]):
    def __init__(self, rows: Sequence[int]) -> None:
        self.rows = tuple(int(value) for value in rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


def _read(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_JSON_PATH_INVALID")
    value = json.loads(resolved.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_JSON_INVALID")
    return value


def _load_final_authority(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or file_sha256(resolved) != expected_file_sha256
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_FINAL_AUTHORITY_FILE_INVALID")
    return require_final_train_checkpoint_authority(_read(resolved), verify_files=True)


def _campaign_context(
    authority: Mapping[str, Any],
    *,
    authority_path: Path,
    authority_file_sha256: str,
    progress_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_sha256 = os.environ.get("GX1_CAMPAIGN_PLAN_SHA256")
    plan_file_sha256 = os.environ.get("GX1_CAMPAIGN_PLAN_FILE_SHA256")
    plan_path_raw = os.environ.get("GX1_CAMPAIGN_PLAN_PATH")
    if (
        not isinstance(plan_sha256, str)
        or not isinstance(plan_file_sha256, str)
        or not isinstance(plan_path_raw, str)
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_CAMPAIGN_ENV_INVALID")
    plan_path = Path(plan_path_raw)
    if not plan_path.is_absolute() or plan_path.resolve() != plan_path:
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_CAMPAIGN_PATH_INVALID")
    plan = require_plan(
        read_bound_json(plan_path, plan_file_sha256),
        verify_files=True,
    )
    if plan["plan_sha256"] != plan_sha256:
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_CAMPAIGN_SHA_INVALID")
    invocation_sha256 = os.environ.get("GX1_CAMPAIGN_INVOCATION_SHA256")
    matches = [
        item
        for item in plan["checked_invocations"]
        if item["invocation_sha256"] == invocation_sha256
    ]
    if len(matches) != 1:
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_CAMPAIGN_INVOCATION_INVALID")
    invocation = matches[0]
    authority_binding = {
        "path": str(authority_path),
        "sha256": authority_file_sha256,
    }
    if (
        plan["phase"] != "full_val"
        or plan["final_train_checkpoint_authority"] != authority_binding
        or plan["selected_batch_size"] != authority["selected_batch_size"]
        or (
            authority["schema_version"] != FULL_POPULATION_SCHEMA_VERSION
            and plan["source_commit"] != authority["source_commit"]
        )
        or invocation["kind"] != "full_val_window"
        or Path(invocation["progress_path"]) != progress_path
        or invocation["checkpoint"]["pointer_path"]
        != authority["final_checkpoint_pointer"]["path"]
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_CAMPAIGN_CONTEXT_INVALID")
    return plan, invocation


def _publish_campaign_progress(
    *,
    path: Path,
    authority: Mapping[str, Any],
    plan: Mapping[str, Any],
    invocation: Mapping[str, Any],
    result: Mapping[str, Any],
    rollout_cursor_path: Path,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_CAMPAIGN_PROGRESS_EXISTS")
    decision = result.get("decision")
    complete = decision in {
        "PASS_COMPLETE",
        "COMPLETE_WITH_RIGHT_CENSORING",
    }
    resumable = decision == "PAUSED_RESUMABLE"
    if not complete and not resumable:
        outcome = "FAILED"
    else:
        outcome = "COMPLETE" if complete else "RESUMABLE"
    value = {
        "schema_version": PROGRESS_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "invocation_sha256": invocation["invocation_sha256"],
        "phase": invocation["kind"],
        "epoch_index": int(authority["epoch_index"]),
        "global_optimizer_steps": int(authority["global_optimizer_steps"]),
        "next_batch_offset": int(authority["total_batches"]),
        "total_batches": int(authority["total_batches"]),
        "completed_units": int(
            result.get(
                "entry_pair_cohort_size",
                result.get("completed_entry_pair_count", 0),
            )
        ),
        "total_units": 5_508,
        "epoch_schedule_sha256": authority["epoch_schedule_sha256"],
        "selection_receipt_sha256": authority["gpu_batch_selection_artifact_sha256"],
        "checkpoint_pointer": dict(authority["final_checkpoint_pointer"]),
        "rollout_cursor": {
            "path": str(rollout_cursor_path),
            "sha256": file_sha256(rollout_cursor_path),
        },
        "terminal": True,
        "outcome": outcome,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
    }
    value["progress_sha256"] = campaign_sha256(value)
    checked = require_campaign_progress(
        value,
        plan_sha256=plan["plan_sha256"],
        invocation=invocation,
        expected_selection_receipt_sha256=authority[
            "gpu_batch_selection_artifact_sha256"
        ],
        verify_file=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(checked, sort_keys=True, indent=2) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return checked


def _source_path(manifest: Mapping[str, Any], name: str) -> Path:
    binding = manifest.get("source_bindings", {}).get(name)
    path = (
        Path(str(binding.get("path", ""))).expanduser().resolve()
        if isinstance(binding, Mapping)
        else Path()
    )
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or file_sha256(path) != binding.get("sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_SOURCE_BINDING_INVALID")
    return path


def _assert_clean_source(launch: Mapping[str, Any]) -> None:
    repository = str(launch["source_repo"])
    head = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            repository,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if head != launch["source_commit"] or status:
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_SOURCE_NOT_CLEAN")


def _entry_representations(
    *,
    model: torch.nn.Module,
    dataset: EntryV10CtxDataset,
    parent_rows: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    if len(parent_rows) != 5_508 or len(set(parent_rows)) != len(parent_rows):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_PARENT_ENTRY_MAPPING_INVALID")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=_ExactSampler(parent_rows),
        num_workers=0,
    )
    route_accumulators: dict[str, Any] = {}
    representations: list[torch.Tensor] = []
    entry_q_values: list[torch.Tensor] = []
    consumed = 0
    for batch in loader:
        observed_rows = batch["entry_row_index"].tolist()
        expected_rows = list(parent_rows[consumed : consumed + len(observed_rows)])
        if observed_rows != expected_rows:
            raise RuntimeError("UNIFIED_EXIT_VAL_CLI_PARENT_ENTRY_ORDER_INVALID")
        seq_x = batch["seq_x"].to(device)
        snap_x = batch["snap_x"].to(device)
        ctx_cont = batch["ctx_cont"].to(device)
        ctx_cat = batch["ctx_cat"].to(device)
        with torch.inference_mode():
            output = _model_forward_fp32(
                model,
                seq_x,
                snap_x,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
                **_multi_tf_kwargs_from_batch(batch, device),
            )
        representation = output.get(UNIFIED_EXIT_MODEL_REPRESENTATION_KEY)
        entry_q = output.get("entry_action_q_bps")
        if (
            not isinstance(entry_q, torch.Tensor) or entry_q.dtype != torch.float32
            or entry_q.shape != (len(observed_rows), 3)
            or not bool(torch.isfinite(entry_q).all().item())
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_CLI_ENTRY_Q_INVALID")
        entry_q_values.append(entry_q.detach().cpu())
        if (
            not isinstance(representation, torch.Tensor)
            or representation.ndim != 2
            or representation.shape[0] != len(observed_rows)
            or not bool(torch.isfinite(representation).all().item())
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_CLI_ENTRY_REPRESENTATION_INVALID")
        accumulate_route_diagnostics_v1(
            route_accumulators,
            {
                "exit_specialist_gate": output.get("specialist_gate"),
                "exit_tf_gate": output.get("tf_gate"),
                "exit_family_tf_cooperation_gate": output.get(
                    "family_tf_cooperation_gate"
                ),
                "exit_family_tf_feature_gate": output.get("family_tf_feature_gate"),
            },
        )
        representations.append(representation.detach())
        consumed += len(observed_rows)
    if consumed != 5_508:
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_ENTRY_COHORT_INCOMPLETE")
    diagnostics = finalize_route_diagnostics_v1(route_accumulators)
    diagnostics["surface"] = "entry_full_cohort_forward"
    return torch.cat(representations, dim=0), diagnostics, torch.cat(entry_q_values, dim=0)


def _build_provider(
    *,
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    cost_authority_path: Path,
) -> LazyUnifiedExitEconomicStepProviderV1:
    child_path = _source_path(manifest, "m1_child")
    child_manifest_path = _source_path(manifest, "m1_child_manifest")
    closure_path = _source_path(manifest, "closure_authority")
    counts_path = _source_path(manifest, "successor_counts")
    summary_path = _source_path(manifest, "summary_manifest")
    counts = np.load(counts_path, allow_pickle=False)
    summary = _read(summary_path)
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
    return LazyUnifiedExitEconomicStepProviderV1(
        compact_rows=provider_rows,
        compact_manifest=manifest,
        economics_readiness=readiness,
        cost_parameter_authority_path=cost_authority_path,
        market_closure_authority_path=closure_path,
        market_closure_authority_file_sha256=file_sha256(closure_path),
        state_m1_source_path=child_path,
        state_m1_source_manifest_path=child_manifest_path,
        state_m1_source_file_sha256=file_sha256(child_path),
        state_m1_source_manifest_file_sha256=file_sha256(child_manifest_path),
        parent_m1_row_offset=int(manifest["parent_m1_row_offset"]),
        common_successor_transition_counts=counts,
        expected_successor_counts_sha256=summary["successor_counts_sha256"],
    )


def run(
    *,
    launch_manifest_path: Path,
    final_train_checkpoint_authority_path: Path,
    final_train_checkpoint_authority_file_sha256: str,
    checkpoint_pointer_path: Path,
    campaign_progress_path: Path,
    rollout_progress_path: Path,
    result_path: Path,
    device: torch.device,
    max_forwards_this_invocation: int,
    progress_interval_forwards: int,
    compute_guard_max_model_forwards: int,
    compute_guard_max_materialized_state_views: int,
    compute_guard_max_wall_seconds: float,
) -> dict[str, Any]:
    launch = require_launch_manifest(_read(launch_manifest_path))
    _assert_clean_source(launch)
    authority = _load_final_authority(
        final_train_checkpoint_authority_path,
        final_train_checkpoint_authority_file_sha256,
    )
    plan, invocation = _campaign_context(
        authority,
        authority_path=final_train_checkpoint_authority_path,
        authority_file_sha256=final_train_checkpoint_authority_file_sha256,
        progress_path=campaign_progress_path,
    )
    if Path(__file__).resolve().parents[2] != Path(plan["source_repo"]):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_EXECUTION_SOURCE_INVALID")
    _assert_clean_source(plan)
    files = {name: Path(binding["path"]) for name, binding in launch["files"].items()}
    if (
        authority.get("bootstrap_source_commit", authority["source_commit"]) != launch["source_commit"]
        or authority["launch_manifest"]["path"] != str(launch_manifest_path)
        or authority["launch_manifest"]["sha256"] != file_sha256(launch_manifest_path)
        or authority["launch_manifest_sha256"] != launch["manifest_sha256"]
        or authority["final_checkpoint_pointer"]["path"] != str(checkpoint_pointer_path)
        or authority["random_access_root"] != launch["files"]["random_access_root"]
        or authority["economics_readiness"] != launch["files"]["economics_readiness"]
        or authority["train_cost_authority"] != launch["files"]["train_cost_authority"]
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_FINAL_AUTHORITY_DRIFT")
    selected_batch_size = int(authority["selected_batch_size"])
    selected = require_selected_sampler_artifact(_read(files["selected_sampler"]))
    root_path = files["random_access_root"]
    root = require_random_access_index_root(_read(root_path))
    val_binding = root["splits"]["val"]
    index_path = Path(val_binding["index_parquet_path"])
    index_manifest_path = Path(val_binding["manifest_path"])
    frame = pd.read_parquet(index_path)
    index_manifest = require_random_access_index_manifest(
        _read(index_manifest_path),
        expected_split="val",
        index_frame=frame,
        index_path=index_path,
        verify_sources=False,
    )
    if (
        val_binding["index_parquet_sha256"] != file_sha256(index_path)
        or val_binding["manifest_sha256"] != index_manifest["manifest_sha256"]
        or frame["entry_row_index"].astype("int64").tolist() != list(range(5_508))
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_INDEX_INVALID")
    # The index predates the Entry-notional correction. Keep its immutable
    # sources, and prove the launch-selected parent has exactly the same full
    # clock and row coordinates before any model inference. This call also
    # verifies all original index sources; no source check is skipped.
    parent_coordinate_evidence = require_parent_entry_coordinate_equivalence(
        index_manifest=index_manifest, index_frame=frame, expected_split="val",
        parent_parquet=launch["files"]["entry_val_parquet"],
        parent_manifest=launch["files"]["entry_val_manifest"],
    )
    pointer = _read(checkpoint_pointer_path)
    epoch_index = int(pointer.get("epoch_index", -1))
    if "full_population_schedule" in authority:
        if authority["full_population_schedule"]["epoch_index"] != epoch_index:
            raise RuntimeError("UNIFIED_EXIT_VAL_CLI_FULL_EPOCH_INVALID")
        # The rebuilt final authority verifies the bound full schedule witness.
        epoch_schedule_sha256 = authority["epoch_schedule_sha256"]
    else:
        anchors = schedule_random_access_entry_anchors(
            sampler_contract=selected["selected_sampler_contract"],
            epoch_index=epoch_index,
        )
        epoch_schedule_sha256 = canonical_sha256(
            {
                "epoch_index": epoch_index,
                "selected_sampler_contract_sha256": selected[
                    "selected_sampler_contract_sha256"
                ],
                "child_entry_order": [int(anchor["entry_row_index"]) for anchor in anchors],
            }
        )
    meta = _read(files["source_bundle_metadata"])
    child = require_composite_normalization_binding(
        _read(files["child_composite_normalization"])
    )
    bootstrap = require_bootstrap_composite_normalization(
        _read(files["bootstrap_composite_normalization"])
    )
    child_norm = child["base_feature_normalization"]["artifact"]["contract"]
    old_norm = bootstrap["base_feature_normalization"]["artifact"]["base_artifact"][
        "contract"
    ]
    model = _model(meta, child_norm, device)
    checkpoint_binding = load_selected_weight_ema_checkpoint_readonly_v1(
        pointer_path=checkpoint_pointer_path,
        model=model,
        expected_checkpoint_pointer_file_sha256=authority["final_checkpoint_pointer"][
            "sha256"
        ],
        expected_launch_manifest_sha256=launch["manifest_sha256"],
        expected_selected_sampler_artifact_sha256=selected["artifact_sha256"],
        expected_bootstrap_source_receipt_sha256=launch[
            "bootstrap_source_receipt_sha256"
        ],
        expected_base_normalization_sha256=old_norm["contract_sha256"],
        expected_summary_normalization_sha256=child["lifetime_summary_normalization"][
            "normalization_sha256"
        ],
        expected_batch_size=selected_batch_size,
        expected_epoch_schedule_sha256=authority["epoch_schedule_sha256"],
        expected_weight_ema_decay=float(launch["weight_ema_decay"]),
    )
    if (
        epoch_schedule_sha256 != authority["epoch_schedule_sha256"]
        or selected["artifact_sha256"] != authority["selected_sampler_artifact_sha256"]
        or old_norm["contract_sha256"] != authority["base_normalization_sha256"]
        or child["lifetime_summary_normalization"]["normalization_sha256"]
        != authority["summary_normalization_sha256"]
        or checkpoint_binding["global_step"] != authority["global_optimizer_steps"]
        or checkpoint_binding["checkpoint_file_sha256"]
        != authority["final_checkpoint_state"]["sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_CHECKPOINT_AUTHORITY_MISMATCH")
    bind_preserved_v7_input_normalization(model, old_norm)
    per_tf = {
        name.upper(): int(meta["multi_tf"][f"{name}_seq_len"])
        for name in ("m5", "m15", "h1", "h4", "d1")
    }
    parent_val_path = files["entry_val_parquet"]
    entry_dataset = EntryV10CtxDataset(
        parent_val_path,
        seq_len=int(meta["seq_len"]),
        m5_prebuilt_path=files["m5_prebuilt"],
        per_tf_seq_lens=per_tf,
        multi_tf_closed_bar=True,
        sequence_source_audit_json=files["sequence_source_audit"],
    )
    parent_rows = frame["parent_entry_row_index"].astype("int64").tolist()
    representations, entry_routes, entry_q_values = _entry_representations(
        model=model,
        dataset=entry_dataset,
        parent_rows=parent_rows,
        device=device,
        batch_size=selected_batch_size,
    )
    entry_routes["parent_entry_coordinate_evidence"] = parent_coordinate_evidence
    corpus = UnifiedExitLifecycleCorpus(
        root_manifest_path=files["feature_lifecycle_root"],
        entry_parquets={
            "train": files["entry_train_parquet"],
            "val": files["entry_val_parquet"],
        },
        entry_manifest_bindings={
            "train": {
                "path": str(files["entry_train_manifest"]),
                "sha256": file_sha256(files["entry_train_manifest"]),
            },
            "val": {
                "path": str(files["entry_val_manifest"]),
                "sha256": file_sha256(files["entry_val_manifest"]),
            },
        },
        dataset_run_id=str(launch["dataset_run_id"]),
        splits=("train", "val"),
    )
    readiness = require_unified_exit_unbounded_training_readiness(
        _read(files["economics_readiness"]),
        context="UNIFIED_EXIT_RANDOM_ACCESS_VAL_CLI",
    )
    provider = _build_provider(
        frame=frame,
        manifest=index_manifest,
        readiness=readiness,
        cost_authority_path=files["train_cost_authority"],
    )
    state_factory = RandomAccessValStateFactoryV1.from_artifacts(
        entry_parquet_path=_source_path(index_manifest, "entry_parquet"),
        entry_manifest_path=_source_path(index_manifest, "entry_manifest"),
        child_m1_path=_source_path(index_manifest, "m1_child"),
        child_m1_manifest_path=_source_path(index_manifest, "m1_child_manifest"),
        successor_counts_path=_source_path(index_manifest, "successor_counts"),
        summary_manifest_path=_source_path(index_manifest, "summary_manifest"),
        first_state_bridge_path=_source_path(index_manifest, "first_state_bridge"),
        split_sequence_binding_path=_source_path(index_manifest, "sequence_binding"),
        composite_normalization_path=_source_path(
            index_manifest, "composite_normalization"
        ),
        closure_authority_path=_source_path(index_manifest, "closure_authority"),
        random_access_index_path=index_path,
        random_access_index_manifest_path=index_manifest_path,
        random_access_index_root_path=root_path,
        source_owner=corpus.splits["val"],
        mtf_materializer=entry_dataset._get_exit_multi_tf_episode_histories,
        economic_step_provider=provider,
        economic_step_manifest=provider.economic_exit_step_manifest,
        economics_objective_contract=readiness["economics_objective_contract"],
    )
    contract, adapter = state_factory.bind_rollout(
        entry_decision_representations=representations,
        model_state_sha256=checkpoint_binding["model_state_sha256"],
        checkpoint_file_sha256=checkpoint_binding["checkpoint_file_sha256"],
        compute_guard_max_model_forwards=compute_guard_max_model_forwards,
        compute_guard_max_materialized_state_views=(
            compute_guard_max_materialized_state_views
        ),
        compute_guard_max_wall_seconds=compute_guard_max_wall_seconds,
        resumable_wall_limit=True,
    )
    if contract["entry_pair_cohort_size"] != 5_508:
        raise RuntimeError("UNIFIED_EXIT_VAL_CLI_FULL_COHORT_REQUIRED")
    entry_policy = build_entry_policy_decisions(
        predicted_q_bps=entry_q_values.numpy(),
        entry_row_indices=frame["entry_row_index"].astype("int64").tolist(),
        checkpoint_binding_sha256=checkpoint_binding["binding_sha256"],
    )
    result = run_resumable_random_access_val_evaluation_v1(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
        checkpoint_binding=checkpoint_binding,
        entry_route_diagnostics=entry_routes,
        entry_policy_decisions=entry_policy,
        progress_path=rollout_progress_path,
        result_path=result_path,
        max_forwards_this_invocation=max_forwards_this_invocation,
        policy_batch_size=selected_batch_size,
        progress_interval_forwards=progress_interval_forwards,
    )
    _publish_campaign_progress(
        path=campaign_progress_path,
        authority=authority,
        plan=plan,
        invocation=invocation,
        result=result,
        rollout_cursor_path=rollout_progress_path,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--final-train-checkpoint-authority", type=Path, required=True)
    parser.add_argument("--final-train-checkpoint-authority-file-sha256", required=True)
    parser.add_argument("--checkpoint-pointer", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path, required=True)
    parser.add_argument("--rollout-progress-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--max-forwards-this-invocation", type=int, required=True)
    parser.add_argument("--progress-interval-forwards", type=int, default=64)
    parser.add_argument("--compute-guard-max-model-forwards", type=int, required=True)
    parser.add_argument(
        "--compute-guard-max-materialized-state-views", type=int, required=True
    )
    parser.add_argument("--compute-guard-max-wall-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    result = run(
        launch_manifest_path=args.launch_manifest.resolve(),
        final_train_checkpoint_authority_path=(
            args.final_train_checkpoint_authority.resolve()
        ),
        final_train_checkpoint_authority_file_sha256=(
            args.final_train_checkpoint_authority_file_sha256
        ),
        checkpoint_pointer_path=args.checkpoint_pointer.resolve(),
        campaign_progress_path=args.progress_path.resolve(),
        rollout_progress_path=args.rollout_progress_path.resolve(),
        result_path=args.result_path.resolve(),
        device=torch.device(args.device),
        max_forwards_this_invocation=args.max_forwards_this_invocation,
        progress_interval_forwards=args.progress_interval_forwards,
        compute_guard_max_model_forwards=args.compute_guard_max_model_forwards,
        compute_guard_max_materialized_state_views=(
            args.compute_guard_max_materialized_state_views
        ),
        compute_guard_max_wall_seconds=args.compute_guard_max_wall_seconds,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "campaign_progress_path": str(args.progress_path.resolve()),
                "rollout_progress_path": str(args.rollout_progress_path.resolve()),
                "result_path": str(args.result_path.resolve()),
                "result_file_sha256": (
                    file_sha256(args.result_path.resolve())
                    if args.result_path.resolve().is_file()
                    else None
                ),
                "semantic_result_sha256": result.get("semantic_result_sha256"),
                "pause_sha256": result.get("pause_sha256"),
                "next_state_index": result.get("next_state_index"),
                "model_forward_count": result.get("model_forward_count"),
                "test_data_used": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return (
        0
        if result.get("decision")
        in {
            "PASS_COMPLETE",
            "COMPLETE_WITH_RIGHT_CENSORING",
            "PAUSED_RESUMABLE",
        }
        else 75
    )


if __name__ == "__main__":
    raise SystemExit(main())
