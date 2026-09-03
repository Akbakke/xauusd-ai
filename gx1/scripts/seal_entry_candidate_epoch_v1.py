#!/usr/bin/env python3
"""Export one already VAL-complete candidate epoch as technical-only evidence.

The source candidate session remains immutable and incomplete.  This command
only reads its hash-verified top-k state, records why the requested epoch was
sealed, and delegates the normal immutable bundle export to the canonical
trainer.  It never resumes optimizer work and never reads TEST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from gx1.contracts.unified_exit_gate_evidence_v1 import (
    require_unified_exit_gate_evidence,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10 import entry_v10_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_JSON_INVALID]")
    return value


def _write_failure_report(*, out_bundle: Path, seal: dict, error: RuntimeError) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = out_bundle.parent / (
        f"{out_bundle.name}_EPOCH{seal['selected_epoch']}_TECHNICAL_SEAL_FAILURE_{stamp}.json"
    )
    payload = {
        "schema_version": "gx1_candidate_epoch_technical_seal_failure_v1",
        "decision": "FAIL_NO_BUNDLE_PUBLISHED",
        "authority": {
            "technical_epoch_result_only": True,
            "candidate": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "seal": seal,
        "error": str(error),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def _load_completed_candidate_epoch_for_seal(
    *, out_bundle_dir: Path, completed_epoch: int
) -> dict:
    """Read a terminal one-epoch candidate session without mutating it.

    ``entry_v10_ctx_train_v3`` persists the final candidate state as
    ``complete=true``, ``phase=validation`` and a zero-based ``epoch_index``.
    The original trainer helper accidentally admitted the inverse state, which
    made a correctly completed one-epoch candidate impossible to export.  Keep
    the repair in the export-only boundary for this already hash-bound run; the
    canonical trainer receives the matching permanent correction separately.
    """

    if isinstance(completed_epoch, bool) or int(completed_epoch) < 1:
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_EPOCH_INVALID]")
    resolved_out_bundle_dir = Path(out_bundle_dir).expanduser().resolve()
    session_dir = resolved_out_bundle_dir.parent / (
        ".gx1-candidate-training-session." + resolved_out_bundle_dir.name
    )
    contract = trainer._candidate_training_session_read_json(
        session_dir / "CANDIDATE_TRAINING_SESSION_CONTRACT.json",
        label="EPOCH_SEAL_CONTRACT",
    )
    authority = contract.get("authority")
    if (
        contract.get("schema_version")
        != trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION
        or contract.get("out_bundle_dir") != str(resolved_out_bundle_dir)
        or contract.get("profile") != "candidate"
        or contract.get("execution_tier") != "canonical"
        or not isinstance(authority, dict)
        or authority.get("candidate_training") is not True
        or authority.get("bundle") is not False
    ):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_CONTRACT_INVALID]")
    session = trainer._CandidateTrainingSession(
        out_bundle_dir=resolved_out_bundle_dir,
        contract=contract,
    )
    state = session.load_checkpoint()
    if state is None or not bool(state.get("complete", False)):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_STATE_INVALID]")
    progress = trainer._require_candidate_training_progress(state["training_progress"])
    selection = dict(progress["checkpoint_selection"])
    if (
        int(selection["last_epoch"]) != int(completed_epoch)
        or int(selection["best_epoch"]) != int(completed_epoch)
        or int(state["epoch_index"]) + 1 != int(completed_epoch)
        or state.get("phase") != "validation"
        or progress.get("validation_snapshot") is not None
    ):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_VAL_NOT_COMPLETE]")
    checkpoint = selection.get("best_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or int(checkpoint.get("epoch", -1)) != int(completed_epoch)
        or checkpoint not in selection["top_k_checkpoints"]
        or not isinstance(selection.get("best_state"), dict)
        or not isinstance(selection.get("best_fitted_q_target_state"), dict)
    ):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_SELECTION_INVALID]")
    relative = Path(str(checkpoint.get("path", "")))
    if relative.is_absolute() or relative.parts[:1] != ("top_k",):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_TOP_K_PATH_INVALID]")
    top_k_path = session.directory / relative
    try:
        payload = torch.load(top_k_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_TOP_K_LOAD_INVALID]") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("session_contract_sha256") != session.contract_sha256
        or payload.get("epoch") != int(completed_epoch)
        or payload.get("metric") != checkpoint.get("metric")
        or trainer.canonical_model_state_sha256(payload.get("model_state", {}))
        != trainer.canonical_model_state_sha256(selection["best_state"])
        or trainer.canonical_model_state_sha256(payload.get("target_model_state", {}))
        != trainer.canonical_model_state_sha256(
            selection["best_fitted_q_target_state"]
        )
    ):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_TOP_K_MISMATCH]")
    return {
        **selection,
        "joint_task_supervision_observed": dict(
            progress["joint_task_supervision_observed"]
        ),
        "joint_task_gradient_observed": dict(
            progress["joint_task_gradient_observed"]
        ),
        "session_directory": str(session.directory),
    }


def _require_profiled_unified_exit_gate_evidence_for_epoch_seal(
    *,
    training_profile: str,
    exit_validation: Mapping[str, Any],
    full_trajectory_validation: Mapping[str, Any],
) -> None:
    """Validate the persisted shared Exit gates at their per-side row count.

    The candidate full-trajectory population represents both Long and Short
    sides.  The shared Exit gate accumulator is intentionally invoked once per
    side, so its ``*_rows`` evidence is exactly half that combined population.
    The original bundle boundary compared it against the combined count and
    rejected otherwise valid full-VAL evidence.  This local replacement keeps
    the already hash-bound V9 model and bundle modules unchanged while the
    permanent source-level correction is made after this technical seal.
    """

    if training_profile == "smoke":
        return
    if training_profile != "candidate":
        raise RuntimeError("[ENTRY_BUNDLE_TRAINING_PROFILE_INVALID]")
    allow_static = entry_v10_bundle._candidate_static_exit_gate_provisional(
        exit_validation
    )
    for context, evidence, total_rows in (
        (
            "ENTRY_BUNDLE_SELECTED_CHECKPOINT",
            exit_validation,
            exit_validation.get("unified_exit_population_rows"),
        ),
        (
            "ENTRY_BUNDLE_FULL_TRAJECTORY",
            full_trajectory_validation,
            full_trajectory_validation.get("population_rows"),
        ),
    ):
        if (
            isinstance(total_rows, bool)
            or not isinstance(total_rows, int)
            or total_rows <= 0
            or total_rows % 2 != 0
        ):
            raise RuntimeError("[ENTRY_BUNDLE_UNIFIED_EXIT_GATE_ROWS_INVALID]")
        require_unified_exit_gate_evidence(
            evidence,
            expected_rows=total_rows // 2,
            context=context,
            allow_static_feature_gate_provisional=allow_static,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        "seal one completed candidate epoch without resuming training"
    )
    parser.add_argument("--recipe-json", type=Path, required=True)
    parser.add_argument("--recipe-sha256", type=str, required=True)
    parser.add_argument("--completed-epoch", type=int, required=True)
    args = parser.parse_args()

    recipe_path = args.recipe_json.expanduser().resolve(strict=True)
    if _sha256(recipe_path) != args.recipe_sha256:
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_RECIPE_SHA256_MISMATCH]")
    recipe = _read_json(recipe_path)
    required = {
        "schema_version",
        "run_id",
        "dataset_run_id",
        "out_bundle_dir",
        "trainer_cli",
        "artifact_bindings",
        "test_guard_lineage",
    }
    if not required.issubset(recipe):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_RECIPE_INVALID]")
    cli = recipe["trainer_cli"]
    artifacts = recipe["artifact_bindings"]
    if (
        not isinstance(cli, dict)
        or not isinstance(artifacts, dict)
        or recipe.get("profile") != "candidate"
        or cli.get("subsample_rows") != 0
    ):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_RECIPE_INVALID]")

    out_bundle = Path(recipe["out_bundle_dir"]).expanduser().resolve()
    session_dir = out_bundle.parent / (
        ".gx1-candidate-training-session." + out_bundle.name
    )
    contract_path = session_dir / "CANDIDATE_TRAINING_SESSION_CONTRACT.json"
    pointer_path = session_dir / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
    contract = _read_json(contract_path)
    result = _load_completed_candidate_epoch_for_seal(
        out_bundle_dir=out_bundle,
        completed_epoch=args.completed_epoch,
    )
    checkpoint = result["best_checkpoint"]
    checkpoint_path = session_dir / str(checkpoint["path"])
    if _sha256(checkpoint_path) != str(checkpoint["sha256"]):
        raise RuntimeError("[CANDIDATE_EPOCH_SEAL_CHECKPOINT_SHA256_MISMATCH]")

    def artifact(name: str) -> Path:
        value = artifacts.get(name)
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise RuntimeError(f"[CANDIDATE_EPOCH_SEAL_ARTIFACT_MISSING] {name}")
        path = Path(value["path"]).expanduser().resolve(strict=True)
        if _sha256(path) != value.get("sha256"):
            raise RuntimeError(f"[CANDIDATE_EPOCH_SEAL_ARTIFACT_SHA256_MISMATCH] {name}")
        return path

    trainer._GRAD_CLIP_NORM = float(cli["grad_clip_norm"])
    trainer._WEIGHT_DECAY = float(cli["weight_decay"])
    entry_v10_bundle._require_profiled_unified_exit_gate_evidence = (
        _require_profiled_unified_exit_gate_evidence_for_epoch_seal
    )
    cache_manifest = artifact("multi_tf_cache_manifest")
    os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"] = str(cache_manifest.parent)
    for artifact_name, env_name in (
        ("train_manifest", "GX1_ENTRY_TRAIN_MANIFEST_SHA256"),
        ("val_manifest", "GX1_ENTRY_VAL_MANIFEST_SHA256"),
        ("train_parquet", "GX1_ENTRY_TRAIN_PARQUET_SHA256"),
        ("val_parquet", "GX1_ENTRY_VAL_PARQUET_SHA256"),
        ("m5_prebuilt", "GX1_ENTRY_M5_PREBUILT_SHA256"),
        (
            "unified_exit_lifecycle_manifest",
            "GX1_ENTRY_UNIFIED_EXIT_LIFECYCLE_MANIFEST_SHA256",
        ),
        (
            "train_sequence_source_reconstruction",
            "GX1_ENTRY_TRAIN_SEQUENCE_SOURCE_AUDIT_SHA256",
        ),
        (
            "val_sequence_source_reconstruction",
            "GX1_ENTRY_VAL_SEQUENCE_SOURCE_AUDIT_SHA256",
        ),
    ):
        os.environ[env_name] = str(artifacts[artifact_name]["sha256"])
    seal = {
        "schema_version": "gx1_candidate_epoch_technical_seal_v1",
        "authority": "technical_epoch_result_only",
        "candidate_training": False,
        "promotion": False,
        "paper": False,
        "live": False,
        "test": False,
        "selected_epoch": int(args.completed_epoch),
        "source_session_directory": str(session_dir),
        "source_session_contract_sha256": _sha256(contract_path),
        "source_resume_pointer_sha256": _sha256(pointer_path),
        "selected_checkpoint_path": str(checkpoint_path),
        "selected_checkpoint_sha256": str(checkpoint["sha256"]),
        "source_recipe_path": str(recipe_path),
        "source_recipe_sha256": args.recipe_sha256,
        "sealer_script_sha256": _sha256(Path(__file__).resolve()),
    }
    try:
        trainer.run_train(
        train_parquet=artifact("train_parquet"),
        train_manifest_path=artifact("train_manifest"),
        val_parquet=artifact("val_parquet"),
        unified_exit_lifecycle_manifest_path=artifact(
            "unified_exit_lifecycle_manifest"
        ),
        seq_len=int(cli["seq_len"]),
        seed=int(cli["seed"]),
        device=torch.device("cpu"),
        batch_size=int(cli["batch_size"]),
        epochs=int(cli["epochs"]),
        lr=float(cli["learning_rate"]),
        out_bundle_dir=out_bundle,
        gx1_data_override=str(cli["gx1_data_root"]),
        num_workers=0,
        early_stopping_patience=int(cli["early_stop_patience"]),
        early_stopping_min_delta=float(cli["early_stop_min_delta"]),
        minimum_epochs_before_stop=int(cli["minimum_epochs_before_stop"]),
        save_top_k=int(cli["save_top_k"]),
        m5_prebuilt_path=artifact("m5_prebuilt"),
        specialist_audit_json=artifact("specialist_audit"),
        specialist_contract_mode=trainer.MODEL_NATIVE_CONTRACT_MODE,
        dropout=float(cli["dropout"]),
        multi_tf_num_layers=int(cli["multi_tf_num_layers"]),
        per_tf_seq_len_m5=int(cli["per_tf_seq_len_m5"]),
        per_tf_seq_len_m15=int(cli["per_tf_seq_len_m15"]),
        per_tf_seq_len_h1=int(cli["per_tf_seq_len_h1"]),
        per_tf_seq_len_h4=int(cli["per_tf_seq_len_h4"]),
        per_tf_seq_len_d1=int(cli["per_tf_seq_len_d1"]),
        multi_tf_scale=float(cli["multi_tf_scale"]),
        subsample_rows=0,
        train_time_window_start_utc=None,
        train_time_window_end_utc=None,
        specialist_num_layers=int(cli["specialist_num_layers"]),
        specialist_fusion_scale=float(cli["specialist_fusion_scale"]),
        cross_family_fusion_scale=float(cli["cross_family_fusion_scale"]),
        grad_accum_steps=int(cli["grad_accum_steps"]),
        prefreeze_test_seal_lineage=recipe["test_guard_lineage"],
        recipe_source_provenance=contract["recipe_source_provenance"],
        run_id=str(recipe["run_id"]),
        dataset_run_id=str(recipe["dataset_run_id"]),
        profile="candidate",
        execution_tier="canonical",
        train_sequence_source_audit_json=artifact(
            "train_sequence_source_reconstruction"
        ),
        val_sequence_source_audit_json=artifact(
            "val_sequence_source_reconstruction"
        ),
        candidate_result_override=result,
        candidate_epoch_seal=seal,
        )
    except RuntimeError as exc:
        report = _write_failure_report(
            out_bundle=out_bundle,
            seal=seal,
            error=exc,
        )
        print(f"[CANDIDATE_EPOCH_TECHNICAL_SEAL_FAILURE] {report}")
        raise


if __name__ == "__main__":
    main()
