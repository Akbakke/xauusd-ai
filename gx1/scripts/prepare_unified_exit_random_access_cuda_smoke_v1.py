"""Prepare a blocked, source-bound local random-access CUDA smoke package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256
from gx1.contracts.unified_exit_random_access_cuda_smoke_v1 import (
    build_blocked_smoke_manifest,
    build_bootstrap_base_normalization,
    build_bootstrap_composite_normalization,
    build_bootstrap_source_receipt,
    file_sha256,
    require_smoke_manifest,
)
from gx1.contracts.unified_exit_random_access_index_v1 import (
    require_random_access_index_root,
)

RECIPE_SCHEMA = "gx1_unified_exit_random_access_cuda_smoke_recipe_v1"
_CONTAINER_KEYS = {
    "schema_version",
    "session_contract_sha256",
    "checkpoint_index",
    "phase",
    "epoch_index",
    "next_batch_offset",
    "global_optimizer_steps",
    "epoch_order",
    "model_state",
    "target_model_state",
    "optimizer_state",
    "weight_ema_state",
    "lr_scheduler_state",
    "rng_state",
    "training_progress",
    "complete",
}


def _read(path: Path) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.resolve() != path
        or not path.is_file()
        or path.is_symlink()
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_INPUT_PATH_INVALID")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_INPUT_INVALID")
    return value


def _binding(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError(f"UNIFIED_EXIT_CUDA_SMOKE_{label}_BINDING_INVALID")
    path = Path(str(value["path"]))
    digest = str(value["sha256"])
    if file_sha256(path) != digest:
        raise RuntimeError(f"UNIFIED_EXIT_CUDA_SMOKE_{label}_HASH_INVALID")
    return path, digest


def _keyset_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(("\n".join(sorted(value)) + "\n").encode("utf-8")).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inspect_checkpoint(recipe: Mapping[str, Any]) -> dict[str, Any]:
    pointer_path, pointer_file_sha = _binding(recipe["checkpoint_pointer"], "POINTER")
    session_path, session_file_sha = _binding(
        recipe["checkpoint_session_contract"], "SESSION"
    )
    pointer = _read(pointer_path)
    slot = pointer.get("slot")
    if slot not in (0, 1):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_POINTER_INVALID")
    state_path = pointer_path.parent / f"candidate_training_state_slot_{slot}.pt"
    if (
        not state_path.is_file()
        or state_path.is_symlink()
        or file_sha256(state_path) != pointer.get("state_sha256")
        or session_file_sha != pointer.get("session_contract_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_POINTER_INVALID")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != _CONTAINER_KEYS:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_SCHEMA_INVALID")
    for key in (
        "schema_version",
        "session_contract_sha256",
        "checkpoint_index",
        "phase",
        "epoch_index",
        "next_batch_offset",
        "global_optimizer_steps",
        "complete",
    ):
        if state.get(key) != pointer.get(key):
            raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_POINTER_MISMATCH")
    model = state.get("model_state")
    target = state.get("target_model_state")
    if (
        not isinstance(model, Mapping)
        or not isinstance(target, Mapping)
        or set(model) != set(target)
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_MODEL_INVALID")
    normalization = model.get("input_norm_contract_sha256")
    if (
        not isinstance(normalization, torch.Tensor)
        or normalization.dtype != torch.uint8
        or normalization.numel() != 32
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_CHECKPOINT_NORMALIZATION_INVALID")
    has_v2 = any(
        name == "unified_exit_random_access_architecture_sha256"
        or name.startswith("exit_random_access_")
        for name in model
    )
    return {
        "state_path": str(state_path),
        "state_file_sha256": str(pointer["state_sha256"]),
        "pointer_path": str(pointer_path),
        "pointer_file_sha256": pointer_file_sha,
        "session_contract_path": str(session_path),
        "session_contract_file_sha256": session_file_sha,
        "session_contract_sha256": str(pointer["session_contract_sha256"]),
        "checkpoint_index": int(pointer["checkpoint_index"]),
        "slot": int(slot),
        "phase": str(pointer["phase"]),
        "epoch_index": int(pointer["epoch_index"]),
        "next_batch_offset": int(pointer["next_batch_offset"]),
        "global_optimizer_steps": int(pointer["global_optimizer_steps"]),
        "container_schema_version": str(state["schema_version"]),
        "container_keyset_sha256": _keyset_sha256(state),
        "online_model_state_sha256": canonical_model_state_sha256(model),
        "target_model_state_sha256": canonical_model_state_sha256(target),
        "model_state_keyset_sha256": _keyset_sha256(model),
        "model_state_key_count": len(model),
        "input_normalization_sha256": bytes(
            normalization.detach().cpu().tolist()
        ).hex(),
        "contains_random_access_v2_state": has_v2,
    }


def build_package(recipe_path: Path) -> dict[str, Any]:
    recipe = _read(recipe_path)
    data = dict(recipe)
    claimed = data.pop("recipe_sha256", None)
    if (
        recipe.get("schema_version") != RECIPE_SCHEMA
        or claimed != canonical_sha256(data)
        or recipe.get("test_accessed") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_RECIPE_INVALID")
    repo = Path(str(recipe["source_repo"]))
    source_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    ).stdout.strip()
    if source_commit != recipe["source_commit"]:
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_SOURCE_COMMIT_INVALID")
    checkpoint = _inspect_checkpoint(recipe)
    bootstrap_source = build_bootstrap_source_receipt(
        checkpoint=checkpoint, source_commit=source_commit
    )
    source_metadata_path, source_metadata_sha = _binding(
        recipe["source_bundle_metadata"], "SOURCE_BUNDLE"
    )
    child_base_path, _ = _binding(recipe["child_base_normalization"], "CHILD_BASE")
    child_composite_path, child_composite_sha = _binding(
        recipe["child_composite_normalization"], "CHILD_COMPOSITE"
    )
    child_composite = require_composite_normalization_binding(
        _read(child_composite_path)
    )
    bootstrap_base = build_bootstrap_base_normalization(
        source_bundle_metadata=_read(source_metadata_path),
        source_bundle_metadata_path=str(source_metadata_path),
        source_bundle_metadata_file_sha256=source_metadata_sha,
        checkpoint_input_normalization_sha256=checkpoint["input_normalization_sha256"],
        child_base_artifact=_read(child_base_path),
        pilot_val_start_utc=str(recipe["pilot_val_start_utc"]),
    )
    output_root = Path(str(recipe["output_root"]))
    base_path = output_root / "BOOTSTRAP_BASE_NORMALIZATION.json"
    base_bytes = _json_bytes(bootstrap_base)
    bootstrap_composite = build_bootstrap_composite_normalization(
        bootstrap_base=bootstrap_base,
        base_path=str(base_path),
        base_file_sha256=_bytes_sha(base_bytes),
        child_composite=child_composite,
    )
    composite_path = output_root / "BOOTSTRAP_COMPOSITE_NORMALIZATION.json"
    composite_bytes = _json_bytes(bootstrap_composite)
    artifacts = dict(recipe["artifacts"])
    random_access_root_path, _ = _binding(
        artifacts["random_access_index_root"], "RANDOM_ACCESS_INDEX_ROOT"
    )
    random_access_root = require_random_access_index_root(
        _read(random_access_root_path)
    )
    if (
        random_access_root.get("sampler_selection_status")
        != "BLOCKED_PENDING_TRAIN_ONLY_BENCHMARK"
        or random_access_root.get("selected_sampler_contract_sha256") is not None
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_RANDOM_ACCESS_ROOT_NOT_BLOCKED")
    artifacts["child_composite_normalization"] = {
        "path": str(child_composite_path),
        "sha256": child_composite_sha,
    }
    artifacts["bootstrap_composite_normalization"] = {
        "path": str(composite_path),
        "sha256": _bytes_sha(composite_bytes),
    }
    # Build against a private staging file so every byte is reopened before publish.
    with tempfile.TemporaryDirectory(prefix="gx1-cuda-smoke-") as tmp:
        staged = Path(tmp) / composite_path.name
        staged.write_bytes(composite_bytes)
        staged_artifacts = dict(artifacts)
        staged_artifacts["bootstrap_composite_normalization"] = {
            "path": str(staged),
            "sha256": _bytes_sha(composite_bytes),
        }
        manifest = build_blocked_smoke_manifest(
            source_repo=str(repo),
            source_commit=source_commit,
            bootstrap_source=bootstrap_source,
            bootstrap_base=bootstrap_base,
            artifacts=staged_artifacts,
            coordinator=recipe["coordinator"],
            output_root=str(output_root),
        )
    manifest["artifacts"]["bootstrap_composite_normalization"] = artifacts[
        "bootstrap_composite_normalization"
    ]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return {
        "BOOTSTRAP_SOURCE_RECEIPT.json": bootstrap_source,
        "BOOTSTRAP_BASE_NORMALIZATION.json": bootstrap_base,
        "BOOTSTRAP_COMPOSITE_NORMALIZATION.json": bootstrap_composite,
        "CUDA_SMOKE_MANIFEST.json": manifest,
    }


def materialize(
    recipe_path: Path, output_dir: Path, *, publish: bool
) -> dict[str, Any]:
    outputs = build_package(recipe_path)
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_CUDA_SMOKE_OUTPUT_EXISTS")
    if publish:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
        )
        try:
            for name, value in outputs.items():
                (staging / name).write_bytes(_json_bytes(value))
            # Rebuild manifest paths from staging-independent bytes and validate the final object now.
            require_smoke_manifest(
                outputs["CUDA_SMOKE_MANIFEST.json"], verify_files=False
            )
            os.replace(staging, output_dir)
            require_smoke_manifest(
                _read(output_dir / "CUDA_SMOKE_MANIFEST.json"), verify_files=True
            )
        except BaseException:
            if staging.exists():
                for child in staging.iterdir():
                    child.unlink()
                staging.rmdir()
            raise
    return {
        "decision": outputs["CUDA_SMOKE_MANIFEST.json"]["decision"],
        "published": publish,
        "output_dir": str(output_dir),
        "manifest_sha256": outputs["CUDA_SMOKE_MANIFEST.json"]["manifest_sha256"],
        "checkpoint": outputs["BOOTSTRAP_SOURCE_RECEIPT.json"]["checkpoint"],
        "bootstrap_composite_normalization_sha256": outputs[
            "BOOTSTRAP_COMPOSITE_NORMALIZATION.json"
        ]["composite_normalization_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    result = materialize(
        args.recipe.resolve(), args.output_dir.resolve(), publish=args.publish
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
