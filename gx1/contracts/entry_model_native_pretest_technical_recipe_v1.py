"""Strict TRAIN/VAL-only recipe for the smallest model-native technical run.

This route exists because a pre-TEST dataset intentionally has no physical
TEST artifacts.  It authorizes only an explicitly bounded technical smoke or
a later frozen external candidate; it never authorizes TEST evaluation,
selection, promotion, shadow, paper or live activity.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PrefreezeTestSealLineageError,
    require_pretest_or_prefreeze_test_guard_lineage_metadata,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id


SCHEMA_VERSION = "entry_model_native_pretest_technical_recipe_v1"
DECISION = "PASS"
TECHNICAL_SCOPE = "strict_train_val_only_no_test_evaluation_v1"
PROFILES = frozenset({"smoke", "candidate"})
SOURCE_BINDING_KEYS = frozenset(
    {"path", "sha256", "size_bytes", "mtime_ns", "device", "inode"}
)
ARTIFACT_BINDING_KEYS = frozenset({"path", "sha256"})
RECIPE_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "created_utc",
        "technical_scope",
        "profile",
        "run_id",
        "dataset_run_id",
        "dataset_dir",
        "out_bundle_dir",
        "test_guard_lineage",
        "artifact_bindings",
        "artifact_bindings_sha256",
        "trainer_cli",
        "trainer_cli_sha256",
        "source_commit",
        "source_bindings",
        "source_bindings_sha256",
        "activation_authority",
        "report_only",
        "side_effects_started",
    }
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "train_manifest",
        "val_manifest",
        "train_parquet",
        "val_parquet",
        "dataset_build_proof",
        "full_input_liveness",
        "feature_audit",
        "target_audit",
        "specialist_audit",
        "execution_causality_audit",
        "train_sequence_source_reconstruction",
        "val_sequence_source_reconstruction",
        "unified_exit_lifecycle_manifest",
        "m5_prebuilt",
        "multi_tf_cache_manifest",
    }
)
SIDE_EFFECTS_ZERO = {
    "training": False,
    "replay": False,
    "iql_distillation": False,
    "shadow": False,
    "live": False,
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class PretestTechnicalRecipeError(RuntimeError):
    """The bounded pre-TEST technical recipe is invalid."""


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _absolute(value: Any, *, label: str) -> Path:
    raw = str(value) if isinstance(value, (str, Path)) else ""
    path = Path(raw)
    if (
        not raw
        or not path.is_absolute()
        or str(path) != raw
        or any(part in {".", ".."} for part in path.parts)
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise PretestTechnicalRecipeError(f"{label}: expected canonical absolute path")
    return path


def _sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if _SHA_RE.fullmatch(text) is None:
        raise PretestTechnicalRecipeError(f"{label}: expected lowercase SHA-256")
    return text


def _mapping(value: Any, *, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != keys:
        raise PretestTechnicalRecipeError(f"{label}: keys are not exact")
    return value


def _require_artifact_bindings(
    value: Any,
    *,
    dataset_dir: Path,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or frozenset(value) != REQUIRED_ARTIFACTS:
        raise PretestTechnicalRecipeError("artifact bindings are not exact")
    normalized: dict[str, dict[str, str]] = {}
    for name in sorted(REQUIRED_ARTIFACTS):
        binding = _mapping(value[name], keys=ARTIFACT_BINDING_KEYS, label=f"{name} binding")
        path = _absolute(binding.get("path"), label=f"{name} path")
        sha256 = _sha(binding.get("sha256"), label=f"{name} sha256")
        if "test" in path.name.lower():
            raise PretestTechnicalRecipeError(f"{name}: TEST-like artifact forbidden")
        if name in {
            "train_manifest",
            "val_manifest",
            "train_parquet",
            "val_parquet",
            "dataset_build_proof",
            "full_input_liveness",
            "feature_audit",
            "target_audit",
            "specialist_audit",
            "execution_causality_audit",
            "train_sequence_source_reconstruction",
            "val_sequence_source_reconstruction",
            "unified_exit_lifecycle_manifest",
        } and path.parent != dataset_dir and dataset_dir not in path.parents:
            raise PretestTechnicalRecipeError(f"{name}: must remain in dataset lineage")
        normalized[name] = {"path": str(path), "sha256": sha256}
    for split in ("train", "val"):
        if not normalized[f"{split}_manifest"]["path"].endswith(f"_{split}.manifest.json"):
            raise PretestTechnicalRecipeError(f"{split}: manifest identity mismatch")
        if not normalized[f"{split}_parquet"]["path"].endswith(f"_{split}.parquet"):
            raise PretestTechnicalRecipeError(f"{split}: parquet identity mismatch")
    return normalized


def _require_source_bindings(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise PretestTechnicalRecipeError("source bindings are missing")
    normalized: dict[str, dict[str, Any]] = {}
    for name, binding in value.items():
        if not isinstance(name, str) or not name:
            raise PretestTechnicalRecipeError("source binding key invalid")
        row = _mapping(binding, keys=SOURCE_BINDING_KEYS, label=f"source binding {name}")
        path = _absolute(row.get("path"), label=f"source binding {name} path")
        sha256 = _sha(row.get("sha256"), label=f"source binding {name} sha256")
        integers: dict[str, int] = {}
        for field in ("size_bytes", "mtime_ns", "device", "inode"):
            raw = row.get(field)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise PretestTechnicalRecipeError(f"source binding {name}.{field} invalid")
            integers[field] = raw
        normalized[name] = {"path": str(path), "sha256": sha256, **integers}
    return normalized


def require_pretest_technical_recipe_metadata(
    value: Mapping[str, Any] | Any,
    *,
    expected_profile: str | None = None,
    expected_run_id: str | None = None,
    expected_dataset_run_id: str | None = None,
    expected_dataset_dir: str | Path | None = None,
    expected_out_bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate recipe content only; it never reads data or TEST artifacts."""

    recipe = _mapping(value, keys=RECIPE_KEYS, label="pre-TEST technical recipe")
    if (
        recipe.get("schema_version") != SCHEMA_VERSION
        or recipe.get("decision") != DECISION
        or recipe.get("technical_scope") != TECHNICAL_SCOPE
        or recipe.get("activation_authority") is not False
        or recipe.get("report_only") is not True
        or recipe.get("side_effects_started") != SIDE_EFFECTS_ZERO
        or not isinstance(recipe.get("created_utc"), str)
        or not str(recipe["created_utc"]).strip()
    ):
        raise PretestTechnicalRecipeError("pre-TEST technical recipe safety boundary invalid")
    profile = str(recipe.get("profile") or "")
    if profile not in PROFILES or (expected_profile is not None and profile != expected_profile):
        raise PretestTechnicalRecipeError("pre-TEST technical recipe profile mismatch")
    run_id = require_entry_run_id(recipe.get("run_id"))
    dataset_run_id = require_entry_run_id(recipe.get("dataset_run_id"))
    if run_id == dataset_run_id:
        raise PretestTechnicalRecipeError("training run and dataset run must differ")
    if expected_run_id is not None and run_id != require_entry_run_id(expected_run_id):
        raise PretestTechnicalRecipeError("pre-TEST technical recipe run ID mismatch")
    if (
        expected_dataset_run_id is not None
        and dataset_run_id != require_entry_run_id(expected_dataset_run_id)
    ):
        raise PretestTechnicalRecipeError("pre-TEST technical recipe dataset run ID mismatch")
    dataset_dir = _absolute(recipe.get("dataset_dir"), label="dataset_dir")
    out_bundle_dir = _absolute(recipe.get("out_bundle_dir"), label="out_bundle_dir")
    if expected_dataset_dir is not None and str(dataset_dir) != str(expected_dataset_dir):
        raise PretestTechnicalRecipeError("pre-TEST technical recipe dataset dir mismatch")
    if expected_out_bundle_dir is not None and str(out_bundle_dir) != str(expected_out_bundle_dir):
        raise PretestTechnicalRecipeError("pre-TEST technical recipe output dir mismatch")
    try:
        guard = require_pretest_or_prefreeze_test_guard_lineage_metadata(
            recipe.get("test_guard_lineage"),
            expected_dataset_run_id=dataset_run_id,
            expected_dataset_dir=dataset_dir,
        )
    except (PrefreezeTestSealLineageError, ValueError) as exc:
        raise PretestTechnicalRecipeError("pre-TEST technical recipe guard invalid") from exc
    if guard.get("schema_version") == "entry_model_native_prefreeze_test_seal_lineage_v1":
        raise PretestTechnicalRecipeError("technical recipe must use unopened pre-TEST guard")
    bindings = _require_artifact_bindings(recipe.get("artifact_bindings"), dataset_dir=dataset_dir)
    if recipe.get("artifact_bindings_sha256") != canonical_json_sha256(bindings):
        raise PretestTechnicalRecipeError("artifact binding hash mismatch")
    for split in ("train", "val"):
        for kind in ("manifest", "parquet"):
            key = f"{split}_{kind}"
            guard_key = key
            if bindings[key] != guard[guard_key]:
                raise PretestTechnicalRecipeError(f"{key}: differs from unopened-TEST guard")
    trainer_cli = recipe.get("trainer_cli")
    if not isinstance(trainer_cli, Mapping) or not trainer_cli:
        raise PretestTechnicalRecipeError("trainer CLI contract missing")
    if recipe.get("trainer_cli_sha256") != canonical_json_sha256(trainer_cli):
        raise PretestTechnicalRecipeError("trainer CLI contract hash mismatch")
    if _GIT_SHA_RE.fullmatch(str(recipe.get("source_commit") or "")) is None:
        raise PretestTechnicalRecipeError("source commit invalid")
    source_bindings = _require_source_bindings(recipe.get("source_bindings"))
    if recipe.get("source_bindings_sha256") != canonical_json_sha256(source_bindings):
        raise PretestTechnicalRecipeError("source binding hash mismatch")
    return json.loads(json.dumps(recipe, sort_keys=True, allow_nan=False))
