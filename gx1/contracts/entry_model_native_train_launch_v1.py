"""Fail-closed launch contract for the two model-native seq513 trainers.

The shell wrappers deliberately contain no artifact discovery or recipe
defaults.  This module validates an explicit immutable evidence set and emits
only the audited, allowlisted trainer environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from gx1.contracts.entry_full_input_liveness_v1 import (
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
    require_foundation_audit_report_policy,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS,
    require_model_native_aux_target_contract,
    require_model_native_aux_target_emission_contract,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    require_offline_rl_contract_metadata,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    READY_DECISION as POST_REBUILD_READY_DECISION,
    SCHEMA_VERSION as POST_REBUILD_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    require_smoke_bundle_audit_contract,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    model_native_recipe_env_contract_metadata,
    require_model_native_recipe_env,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES,
)


SCHEMA_VERSION = "entry_model_native_seq513_train_launch_contract_v2"
RECIPE_AUDIT_SCHEMA = "entry_model_native_seq513_train_recipe_audit_v1"
PRETRAIN_AUDIT_SCHEMA = "xau_direction_repair_pretrain_audit_v2"
TRAINER_RELATIVE_PATH = "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
CAPPED_RUNNER_RELATIVE_PATH = "scripts/gx1_capped_run.sh"
CONTROL_SURFACE_RELATIVE_PATH = "scripts/entry_next_edge_control.sh"
LAUNCH_CONTRACT_RELATIVE_PATH = (
    "gx1/contracts/entry_model_native_train_launch_v1.py"
)
RECIPE_CONTRACT_RELATIVE_PATH = (
    "gx1/contracts/entry_model_native_train_recipe_v1.py"
)
RECIPE_PRODUCER_RELATIVE_PATH = (
    "gx1/scripts/materialize_entry_model_native_seq513_train_recipe_audit_v1.py"
)
AUX_TARGET_CONTRACT_RELATIVE_PATH = (
    "gx1/contracts/entry_model_native_aux_targets_v3.py"
)

REQUIRED_SPECIALISTS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)
REQUIRED_RAIL_FEATURES = tuple(
    name
    for name in CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES
    if "_rail_" in name
)

_MISSING_REQUIRED_OBJECTIVE_WEIGHTS = sorted(
    set(REQUIRED_POSITIVE_LOSS_WEIGHTS) - set(MODEL_NATIVE_RECIPE_ENV_KEYS)
)
if _MISSING_REQUIRED_OBJECTIVE_WEIGHTS:
    raise RuntimeError(
        "MODEL_NATIVE_TRAIN_LAUNCH_OBJECTIVE_WEIGHTS_MISSING: "
        + ",".join(_MISSING_REQUIRED_OBJECTIVE_WEIGHTS)
    )

_STAMP_RE = re.compile(r"(?:^|[^0-9])20[0-9]{6}T[0-9]{6}(?:[0-9]{6})?Z(?:[^0-9]|$)")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CAP_RE = re.compile(r"^[1-9][0-9]*[KMGT]$")
_MUTABLE_POINTER_RE = re.compile(r"(?:^|[/_.-])latest(?:[/_.-]|$)", re.IGNORECASE)

_COMMON_BINDING_KEYS = (
    "train_manifest_json",
    "val_manifest_json",
    "test_manifest_json",
    "train_parquet",
    "val_parquet",
    "test_parquet",
    "m5_prebuilt_path",
    "post_rebuild_readiness_json",
    "full_input_liveness_audit_json",
    "feature_audit_json",
    "target_audit_json",
    "specialist_audit_json",
    "pretrain_audit_json",
    "trainability_readiness_json",
)
_PROFILE_BINDING_KEYS = {
    "smoke": (
        "smoke_manifest_json",
        "smoke_readiness_json",
    ),
    "candidate": (
        "candidate_readiness_json",
        "smoke_bundle_audit_json",
    ),
}
_LARGE_ARTIFACT_KEYS = {
    "train_parquet",
    "val_parquet",
    "test_parquet",
    "m5_prebuilt_path",
}
TRAINER_ARTIFACT_HASH_ENV = {
    "train_manifest_json": "GX1_ENTRY_TRAIN_MANIFEST_SHA256",
    "val_manifest_json": "GX1_ENTRY_VAL_MANIFEST_SHA256",
    "test_manifest_json": "GX1_ENTRY_TEST_MANIFEST_SHA256",
    "train_parquet": "GX1_ENTRY_TRAIN_PARQUET_SHA256",
    "val_parquet": "GX1_ENTRY_VAL_PARQUET_SHA256",
    "test_parquet": "GX1_ENTRY_TEST_PARQUET_SHA256",
}


class LaunchContractError(RuntimeError):
    """The immutable launch evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_binding(path: Path, *, content_sha256: str | None = None) -> dict[str, Any]:
    """Build the binding shape expected from an upstream audit.

    Audit producers may pass a previously computed hash for large artifacts;
    launch validation checks their stat identity without rereading many GB.
    """

    resolved = path.expanduser().resolve(strict=True)
    stat_result = resolved.stat()
    digest = content_sha256 or sha256_file(resolved)
    return {
        "path": str(resolved),
        "sha256": str(digest),
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
    }


def recipe_env_sha256(env_map: Mapping[str, str]) -> str:
    return canonical_json_sha256({str(key): str(value) for key, value in sorted(env_map.items())})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser message is platform-specific
        raise LaunchContractError(f"{label} is not readable JSON: {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} root must be an object: {path}")
    return payload


def _resolved_explicit_path(raw: str, label: str, *, directory: bool = False) -> Path:
    _require(bool(raw), f"{label} is required")
    path = Path(raw).expanduser()
    _require(path.is_absolute(), f"{label} must be absolute: {raw}")
    _require(not _MUTABLE_POINTER_RE.search(str(path)), f"{label} uses a mutable pointer: {raw}")
    _require(path.exists(), f"{label} does not exist: {raw}")
    _require(not path.is_symlink(), f"{label} must not be a symlink: {raw}")
    resolved = path.resolve(strict=True)
    _require(str(resolved) == str(path), f"{label} must already be canonical: {raw}")
    if directory:
        _require(resolved.is_dir(), f"{label} must be a directory: {raw}")
    else:
        _require(resolved.is_file(), f"{label} must be a regular file: {raw}")
    return resolved


def _resolved_output_path(raw: str) -> Path:
    _require(bool(raw), "out_bundle_dir is required")
    path = Path(raw).expanduser()
    _require(path.is_absolute(), f"out_bundle_dir must be absolute: {raw}")
    _require(not _MUTABLE_POINTER_RE.search(str(path)), f"out_bundle_dir uses a mutable pointer: {raw}")
    _require(bool(_STAMP_RE.search(str(path))), f"out_bundle_dir must contain an immutable UTC timestamp: {raw}")
    _require(not path.exists(), f"out_bundle_dir must not already exist: {raw}")
    parent = path.parent.resolve(strict=True)
    _require(parent.is_dir(), f"out_bundle_dir parent must exist: {parent}")
    return parent / path.name


def _zero_failure(
    payload: Mapping[str, Any],
    *,
    label: str,
    schema: str,
    decision: str,
) -> None:
    _require(payload.get("schema_version") == schema, f"{label} schema mismatch")
    _require(payload.get("decision") == decision, f"{label} decision must be {decision}")
    _require(payload.get("failures") == [], f"{label} must declare an explicit empty failures list")


def _validate_binding_map(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_paths: Mapping[str, Path],
) -> None:
    bindings = payload.get("artifact_bindings")
    _require(isinstance(bindings, dict), f"{label} artifact_bindings must be an object")
    _require(set(bindings) == set(expected_paths), f"{label} artifact binding set is not exact")
    _require(
        payload.get("artifact_bindings_sha256") == canonical_json_sha256(bindings),
        f"{label} artifact_bindings_sha256 mismatch",
    )
    for key, expected_path in expected_paths.items():
        binding = bindings.get(key)
        _require(isinstance(binding, dict), f"{label} binding {key} must be an object")
        current = artifact_binding(
            expected_path,
            content_sha256=str(binding.get("sha256") or "") if key in _LARGE_ARTIFACT_KEYS else None,
        )
        _require(_SHA_RE.fullmatch(str(binding.get("sha256") or "")) is not None, f"{label} binding {key} has no sha256")
        _require(binding == current, f"{label} binding {key} does not match the current immutable artifact")


def recipe_source_binding_paths(*, repo: Path, wrapper_path: Path) -> dict[str, Path]:
    return {
        "aux_target_contract": (
            repo / AUX_TARGET_CONTRACT_RELATIVE_PATH
        ).resolve(strict=True),
        "control_surface": (repo / CONTROL_SURFACE_RELATIVE_PATH).resolve(strict=True),
        "launch_contract": (repo / LAUNCH_CONTRACT_RELATIVE_PATH).resolve(strict=True),
        "recipe_contract": (repo / RECIPE_CONTRACT_RELATIVE_PATH).resolve(strict=True),
        "recipe_producer": (repo / RECIPE_PRODUCER_RELATIVE_PATH).resolve(strict=True),
        "wrapper": wrapper_path,
        "trainer": (repo / TRAINER_RELATIVE_PATH).resolve(strict=True),
        "capped_runner": (repo / CAPPED_RUNNER_RELATIVE_PATH).resolve(strict=True),
    }


def recipe_source_bindings(*, repo: Path, wrapper_path: Path) -> dict[str, dict[str, Any]]:
    return {
        key: artifact_binding(path)
        for key, path in recipe_source_binding_paths(
            repo=repo,
            wrapper_path=wrapper_path,
        ).items()
    }


def _validate_source_bindings(
    recipe: Mapping[str, Any],
    *,
    repo: Path,
    wrapper_path: Path,
) -> None:
    expected = recipe_source_binding_paths(repo=repo, wrapper_path=wrapper_path)
    bindings = recipe.get("source_bindings")
    _require(isinstance(bindings, dict), "recipe source_bindings must be an object")
    _require(set(bindings) == set(expected), "recipe source binding set is not exact")
    for key, path in expected.items():
        _require(bindings.get(key) == artifact_binding(path), f"recipe source binding mismatch: {key}")
    _require(
        recipe.get("source_bindings_sha256") == canonical_json_sha256(bindings),
        "recipe source_bindings_sha256 mismatch",
    )


def _validate_source_revision(
    recipe: Mapping[str, Any],
    *,
    repo: Path,
) -> None:
    """Require committed lineage while byte bindings own runtime freshness."""

    source_commit = str(recipe.get("source_commit") or "")
    _require(
        re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
        "recipe audit source_commit is not an exact Git commit",
    )
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", source_commit, "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    _require(
        ancestry.returncode == 0,
        "recipe audit source_commit is not an ancestor of current HEAD",
    )


def _validate_split_manifest(
    manifest: Mapping[str, Any], *, path: Path, parquet: Path, m5_prebuilt: Path
) -> None:
    _require(
        manifest.get("schema_version") == MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
        f"split manifest schema mismatch: {path}",
    )
    _require(manifest.get("manifest_variant") == MODEL_NATIVE_CONTRACT_MODE, f"split manifest mode mismatch: {path}")
    _require(int(manifest.get("expected_seq_snap_width") or 0) == MODEL_NATIVE_SIGNAL_DIM, f"split manifest width mismatch: {path}")
    _require(Path(str(manifest.get("output_data_path") or "")).resolve() == parquet, f"split manifest output path mismatch: {path}")
    extra = manifest.get("extra")
    _require(isinstance(extra, dict), f"split manifest extra contract missing: {path}")
    _require(extra.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE, f"split manifest contract_mode mismatch: {path}")
    _require(extra.get("direction_logit_mode") == MODEL_NATIVE_DIRECTION_LOGIT_MODE, f"split manifest direction mode mismatch: {path}")
    _require(extra.get("neutral_xgb_bridge") is False, f"split manifest declares a bridge: {path}")
    signal_contract = extra.get("model_native_signal_contract")
    _require(isinstance(signal_contract, dict), f"split manifest model-native signal contract missing: {path}")
    try:
        require_model_native_signal_contract(signal_contract, context=str(path))
    except Exception as exc:
        raise LaunchContractError(f"split manifest signal contract invalid: {path}: {exc}") from exc
    signal_bridge = extra.get("signal_bridge")
    _require(isinstance(signal_bridge, dict), f"split manifest signal surface missing: {path}")
    _require(int(signal_bridge.get("seq_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM, f"split manifest seq width mismatch: {path}")
    _require(int(signal_bridge.get("snap_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM, f"split manifest snap width mismatch: {path}")
    _require(signal_bridge.get("fields") == signal_contract.get("fields"), f"split manifest ordered fields mismatch: {path}")
    try:
        require_model_native_aux_target_emission_contract(
            extra.get("aux_head_target_contract"),
            context=str(path),
        )
    except Exception as exc:
        raise LaunchContractError(
            f"split manifest aux-target emission contract invalid: {path}: {exc}"
        ) from exc
    inputs = manifest.get("inputs")
    _require(isinstance(inputs, dict), f"split manifest inputs missing: {path}")
    _require(
        Path(str(inputs.get("source_parquet") or "")).resolve() == m5_prebuilt,
        f"split manifest source parquet does not match m5_prebuilt_path: {path}",
    )
    state_contract = extra.get("model_native_state_contract")
    _require(
        isinstance(state_contract, dict),
        f"split manifest model-native state contract missing: {path}",
    )
    _require(
        Path(str(state_contract.get("rank_reference_source_parquet") or "")).resolve()
        == m5_prebuilt,
        f"split manifest rank-reference source mismatch: {path}",
    )
    _require(
        _SHA_RE.fullmatch(
            str(state_contract.get("rank_reference_source_parquet_sha256") or "")
        )
        is not None,
        f"split manifest source parquet hash missing: {path}",
    )


def _validate_feature_audit_signal_partition(feature: Mapping[str, Any]) -> None:
    signal_contract = feature.get("model_native_signal_contract")
    _require(
        isinstance(signal_contract, Mapping),
        "feature audit model-native signal contract missing",
    )
    try:
        require_model_native_signal_contract(
            signal_contract,
            context="TRAIN_LAUNCH_FEATURE_AUDIT",
        )
    except Exception as exc:
        raise LaunchContractError(
            f"feature audit model-native signal contract invalid: {exc}"
        ) from exc

    base_fields = tuple(str(value) for value in signal_contract.get("base_fields", ()))
    selected_fields = tuple(
        str(value) for value in signal_contract.get("selected_fields", ())
    )
    mandatory_prefix = selected_fields[:MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT]
    ranked_remainder = selected_fields[MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:]
    ranking_sha256 = str(feature.get("feature_ranking_sha256") or "")

    _require(
        int(feature.get("model_native_signal_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM,
        "feature audit total signal width mismatch",
    )
    _require(
        int(feature.get("base_signal_dim") or 0) == MODEL_NATIVE_BASE_SIGNAL_DIM,
        "feature audit base signal width mismatch",
    )
    _require(
        tuple(str(value) for value in feature.get("base_signal_fields", ()))
        == MODEL_NATIVE_BASE_FIELDS
        and base_fields == MODEL_NATIVE_BASE_FIELDS,
        "feature audit ordered base fields mismatch",
    )
    _require(
        int(feature.get("selected_feature_count") or 0)
        == MODEL_NATIVE_SELECTED_FEATURE_COUNT
        and int(feature.get("manifest_selected_feature_count") or 0)
        == MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "feature audit selected width mismatch",
    )
    _require(
        int(feature.get("mandatory_selected_feature_count") or 0)
        == MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        and int(feature.get("manifest_mandatory_selected_feature_count") or 0)
        == MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        and mandatory_prefix == MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        "feature audit mandatory full-stack prefix/order mismatch",
    )
    _require(
        int(feature.get("ranked_remainder_feature_count") or 0)
        == MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        and int(feature.get("manifest_ranked_remainder_feature_count") or 0)
        == MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        and len(ranked_remainder) == MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        and feature.get("ranked_remainder_fields_sha256")
        == canonical_json_sha256(list(ranked_remainder)),
        "feature audit ranked remainder mismatch",
    )
    _require(
        feature.get("feature_ranking_fit_scope") == "train_only"
        and len(ranking_sha256) == 64
        and all(character in "0123456789abcdef" for character in ranking_sha256),
        "feature audit TRAIN-only ranking binding mismatch",
    )


def _validate_post_rebuild_readiness(
    post_rebuild: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Path],
    dataset_dir: Path,
) -> dict[str, str]:
    _zero_failure(
        post_rebuild,
        label="post-rebuild readiness",
        schema=POST_REBUILD_SCHEMA_VERSION,
        decision=POST_REBUILD_READY_DECISION,
    )
    _require(
        Path(str(post_rebuild.get("dataset_dir") or "")).resolve() == dataset_dir,
        "post-rebuild readiness dataset mismatch",
    )
    _require(
        Path(str(post_rebuild.get("smoke_dataset_dir") or "")).resolve()
        == dataset_dir,
        "post-rebuild readiness smoke dataset mismatch",
    )
    _require(
        post_rebuild.get("report_only") is True
        and post_rebuild.get("training_allowed") is False,
        "post-rebuild readiness side-effect boundary mismatch",
    )
    side_effects = post_rebuild.get("side_effects_started")
    _require(
        isinstance(side_effects, dict)
        and set(side_effects)
        == {"dataset_rebuild", "training", "replay", "iql_distillation", "shadow", "live"}
        and not any(bool(value) for value in side_effects.values()),
        "post-rebuild readiness side-effect map is not exact zero",
    )
    for key, artifact_key in (
        ("full_input_liveness_contract", "full_input_liveness_audit_json"),
        ("pretrain_audit", "pretrain_audit_json"),
    ):
        binding = post_rebuild.get(key)
        path = artifacts[artifact_key]
        _require(isinstance(binding, dict), f"post-rebuild {key} binding missing")
        _require(
            binding.get("path") == str(path)
            and binding.get("sha256") == sha256_file(path),
            f"post-rebuild {key} binding mismatch",
        )

    split_artifacts = post_rebuild.get("split_artifacts")
    _require(
        isinstance(split_artifacts, dict)
        and set(split_artifacts) == {"train", "val", "test"},
        "post-rebuild split artifact set is not exact",
    )
    large_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        row = split_artifacts[split]
        _require(isinstance(row, dict), f"post-rebuild {split} split binding invalid")
        manifest = artifacts[f"{split}_manifest_json"]
        parquet = artifacts[f"{split}_parquet"]
        expected_manifest_sha = sha256_file(manifest)
        parquet_sha = str(row.get("parquet_sha256") or "")
        _require(
            row.get("manifest_path") == str(manifest)
            and row.get("manifest_sha256") == expected_manifest_sha
            and row.get("parquet_path") == str(parquet)
            and _SHA_RE.fullmatch(parquet_sha) is not None
            and int(row.get("rows") or 0) > 0,
            f"post-rebuild {split} split binding mismatch",
        )
        large_hashes[f"{split}_parquet"] = parquet_sha
    return large_hashes


def _m5_source_hash_from_manifests(
    payloads: Mapping[str, Mapping[str, Any]],
) -> str:
    source_hashes = {
        str(
            (
                payloads[f"{split}_manifest_json"].get("extra") or {}
            ).get("model_native_state_contract", {}).get(
                "rank_reference_source_parquet_sha256"
            )
            or ""
        )
        for split in ("train", "val", "test")
    }
    _require(
        len(source_hashes) == 1
        and _SHA_RE.fullmatch(next(iter(source_hashes))) is not None,
        "split manifests do not bind one exact m5_prebuilt source hash",
    )
    return next(iter(source_hashes))


def _validate_pretrain_audit(
    pretrain: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Path],
    dataset_dir: Path,
) -> None:
    """Validate the real split-native producer schema without invented fields."""

    _zero_failure(
        pretrain,
        label="pretrain audit",
        schema=PRETRAIN_AUDIT_SCHEMA,
        decision="PASS",
    )
    _require(
        Path(str(pretrain.get("dataset_dir") or "")).resolve() == dataset_dir,
        "pretrain audit dataset mismatch",
    )
    _require(
        pretrain.get("data_splits") == ["train", "val", "test"],
        "pretrain audit split contract mismatch",
    )
    _require(
        pretrain.get("require_rail_features") is True
        and pretrain.get("require_inline_seq_structure") is True
        and pretrain.get("require_xau_provenance") is True,
        "pretrain audit required proof toggles are not exact",
    )
    _require(
        tuple(pretrain.get("required_rail_features") or ())
        == REQUIRED_RAIL_FEATURES
        and pretrain.get("missing_rail_features") == [],
        "pretrain audit rail feature proof mismatch",
    )
    split_rows = pretrain.get("splits")
    _require(
        isinstance(split_rows, list) and len(split_rows) == 3,
        "pretrain audit must contain exactly three split reports",
    )
    reports = {
        str(row.get("split") or ""): row
        for row in split_rows
        if isinstance(row, dict)
    }
    _require(
        set(reports) == {"train", "val", "test"},
        "pretrain audit split report set is not exact",
    )
    tape_provenance = pretrain.get("tape_provenance")
    _require(
        isinstance(tape_provenance, dict)
        and set(tape_provenance) == {"train", "val", "test"}
        and tape_provenance["train"]
        == tape_provenance["val"]
        == tape_provenance["test"],
        "pretrain audit XAU tape provenance differs across splits",
    )
    for split in ("train", "val", "test"):
        row = reports[split]
        provenance = row.get("provenance")
        _require(
            row.get("manifest_path") == str(artifacts[f"{split}_manifest_json"])
            and row.get("parquet_path") == str(artifacts[f"{split}_parquet"]),
            f"pretrain audit {split} artifact paths mismatch",
        )
        _require(
            int(row.get("feature_count") or 0) == MODEL_NATIVE_SIGNAL_DIM
            and row.get("core_target_policy")
            == "future_path_and_utility_outcomes_only"
            and row.get("seq_structure_extension_mode")
            == "mandatory_inline_common_causal_history_v1"
            and row.get("missing_polarity_features") == []
            and row.get("missing_target_columns") == [],
            f"pretrain audit {split} feature/target proof mismatch",
        )
        _require(
            isinstance(provenance, dict)
            and provenance.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE
            and provenance.get("direction_logit_mode")
            == MODEL_NATIVE_DIRECTION_LOGIT_MODE
            and provenance.get("neutral_xgb_bridge") is False
            and provenance.get("xgb_bridge_source") == ""
            and provenance.get("xau_tape_provenance")
            == tape_provenance[split],
            f"pretrain audit {split} provenance mismatch",
        )
        signal_contract = provenance.get("model_native_signal_contract")
        _require(
            isinstance(signal_contract, dict),
            f"pretrain audit {split} signal contract missing",
        )
        try:
            require_model_native_signal_contract(
                signal_contract,
                context=f"TRAIN_LAUNCH_PRETRAIN_{split.upper()}",
            )
        except Exception as exc:
            raise LaunchContractError(
                f"pretrain audit {split} signal contract invalid: {exc}"
            ) from exc
        _require(
            provenance.get("fields") == signal_contract.get("fields"),
            f"pretrain audit {split} ordered signal fields mismatch",
        )


def _validate_audits(
    artifacts: Mapping[str, Path],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    dataset_dir: Path,
    profile: str,
) -> None:
    feature = payloads["feature_audit_json"]
    _zero_failure(feature, label="feature audit", schema="entry_feature_foundation_audit_v1", decision="PASS")
    _require(Path(str(feature.get("dataset_dir") or "")).resolve() == dataset_dir, "feature audit dataset mismatch")
    _validate_feature_audit_signal_partition(feature)
    _require(int(feature.get("ctx_cont_dim_v3") or 0) == MODEL_NATIVE_CTX_CONT_DIM, "feature audit continuous context width mismatch")
    _require(int(feature.get("ctx_cat_dim_v3") or 0) == MODEL_NATIVE_CTX_CAT_DIM, "feature audit categorical context width mismatch")

    target = payloads["target_audit_json"]
    _zero_failure(
        target,
        label="target audit",
        schema=FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
        decision="PASS",
    )
    _require(Path(str(target.get("dataset_dir") or "")).resolve() == dataset_dir, "target audit dataset mismatch")
    require_foundation_audit_report_policy(
        target,
        audit_kind="target",
        context="TRAIN_LAUNCH_TARGET_AUDIT",
    )
    require_model_native_aux_target_contract(
        target.get("model_native_aux_target_contract"),
        context="TRAIN_LAUNCH_TARGET_AUDIT",
    )
    offline_rl_target = target.get("offline_rl_target_contract")
    _require(
        isinstance(offline_rl_target, Mapping)
        and offline_rl_target.get("decision") == "PASS"
        and not offline_rl_target.get("failures"),
        "target audit offline-RL target proof failed",
    )
    require_offline_rl_contract_metadata(
        offline_rl_target.get("offline_rl_contract"),
        context="TRAIN_LAUNCH_TARGET_AUDIT",
    )
    target_heads = target.get("target_head_contract")
    _require(
        isinstance(target_heads, Mapping)
        and tuple(target_heads.get("extra_active_target_heads") or ())
        == MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
        and all(
            (target_heads.get("extra_active_target_head_liveness") or {}).get(head)
            is True
            for head in MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
        ),
        "target audit extra active target-head proof failed",
    )

    specialist = payloads["specialist_audit_json"]
    _zero_failure(specialist, label="specialist audit", schema="entry_specialist_feature_group_audit_v1", decision="PASS")
    _require(Path(str(specialist.get("dataset_dir") or "")).resolve() == dataset_dir, "specialist audit dataset mismatch")
    _require(specialist.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE, "specialist audit mode mismatch")
    _require(int(specialist.get("signal_field_count") or 0) == MODEL_NATIVE_SIGNAL_DIM, "specialist audit signal width mismatch")
    _require(int(specialist.get("selected_feature_count") or 0) == MODEL_NATIVE_SELECTED_FEATURE_COUNT, "specialist audit selected width mismatch")
    _require(tuple(specialist.get("required_training_specialists") or ()) == REQUIRED_SPECIALISTS, "specialist audit group order/set mismatch")
    _require(specialist.get("specialist_model_contract_valid") is True, "specialist model contract is not valid")
    _require(specialist.get("signal_routing_all_mapped") is True, "specialist signal routing is incomplete")

    liveness_result = validate_full_input_liveness_artifact(
        artifacts["full_input_liveness_audit_json"],
        expected_dataset_dir=dataset_dir,
        expected_contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )
    _require(liveness_result.get("ok") is True, f"full-input liveness audit invalid: {liveness_result.get('failures')}")

    trainability = payloads["trainability_readiness_json"]
    _zero_failure(
        trainability,
        label="trainability readiness",
        schema="entry_model_native_seq513_trainability_readiness_v1",
        decision="READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW",
    )
    _require(trainability.get("manifest_variant") == MODEL_NATIVE_CONTRACT_MODE, "trainability mode mismatch")
    _require(int(trainability.get("expected_signal_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM, "trainability signal width mismatch")
    _require(tuple(trainability.get("required_training_specialists") or ()) == REQUIRED_SPECIALISTS, "trainability specialist set mismatch")

    if profile == "smoke":
        smoke_manifest = payloads["smoke_manifest_json"]
        _zero_failure(
            smoke_manifest,
            label="smoke manifest",
            schema="entry_model_native_seq513_smoke_manifest_v2",
            decision="READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW",
        )
        _require(smoke_manifest.get("manifest_variant") == MODEL_NATIVE_CONTRACT_MODE, "smoke manifest mode mismatch")
        _require(int(smoke_manifest.get("expected_seq_snap_width") or 0) == MODEL_NATIVE_SIGNAL_DIM, "smoke manifest width mismatch")
        embedded = smoke_manifest.get("smoke_manifest")
        _require(isinstance(embedded, dict), "smoke manifest has no embedded immutable manifest")
        _require(Path(str(embedded.get("out_dir") or "")).resolve() == dataset_dir, "smoke manifest dataset mismatch")
        _require(smoke_manifest.get("manifest_sha256") == canonical_json_sha256(embedded), "smoke embedded manifest hash mismatch")

        smoke_readiness = payloads["smoke_readiness_json"]
        _zero_failure(
            smoke_readiness,
            label="smoke readiness",
            schema="entry_model_native_seq513_smoke_readiness_v2",
            decision="READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW",
        )
        candidate = smoke_readiness.get("smart_candidate")
        _require(isinstance(candidate, dict), "smoke readiness candidate contract missing")
        _require(candidate.get("manifest_variant") == MODEL_NATIVE_CONTRACT_MODE, "smoke readiness mode mismatch")
        _require(int(candidate.get("expected_signal_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM, "smoke readiness signal width mismatch")
        _require(int(candidate.get("expected_selected_feature_count") or 0) == MODEL_NATIVE_SELECTED_FEATURE_COUNT, "smoke readiness selected width mismatch")
    else:
        candidate_readiness = payloads["candidate_readiness_json"]
        _zero_failure(
            candidate_readiness,
            label="candidate readiness",
            schema="entry_candidate_readiness_model_native_v1",
            decision="READY_FOR_CANDIDATE_TRAINING",
        )
        _require(candidate_readiness.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE, "candidate readiness mode mismatch")
        _require(
            int(candidate_readiness.get("sequence_length") or 0)
            == MODEL_NATIVE_SEQ_LEN,
            "candidate readiness sequence length mismatch",
        )
        _require(int(candidate_readiness.get("expected_signal_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM, "candidate readiness signal width mismatch")
        _require(tuple(candidate_readiness.get("required_specialist_groups") or ()) == REQUIRED_SPECIALISTS, "candidate readiness specialist set mismatch")
        _require(candidate_readiness.get("candidate_training_allowed") is True, "candidate readiness does not authorize explicit training")
        _require(
            candidate_readiness.get("promotion_shadow_live_allowed") is False
            and candidate_readiness.get("activation_authority") is False,
            "candidate readiness activation guard mismatch",
        )
        readiness_bindings = candidate_readiness.get("input_bindings")
        _require(
            isinstance(readiness_bindings, dict)
            and set(readiness_bindings)
            == {
                "smoke_bundle_audit",
                "specialist_audit",
                "trainability_readiness",
            },
            "candidate readiness input binding set mismatch",
        )
        expected_readiness_paths = {
            "smoke_bundle_audit": artifacts["smoke_bundle_audit_json"],
            "specialist_audit": artifacts["specialist_audit_json"],
            "trainability_readiness": artifacts["trainability_readiness_json"],
        }
        for name, path in expected_readiness_paths.items():
            _require(
                readiness_bindings.get(name)
                == {"path": str(path), "sha256": sha256_file(path)},
                f"candidate readiness input binding mismatch: {name}",
            )
        _require(
            candidate_readiness.get("input_bindings_sha256")
            == canonical_json_sha256(readiness_bindings),
            "candidate readiness input binding hash mismatch",
        )

        smoke_bundle = payloads["smoke_bundle_audit_json"]
        try:
            require_smoke_bundle_audit_contract(
                smoke_bundle,
                context="CANDIDATE_TRAIN_LAUNCH",
            )
        except RuntimeError as exc:
            raise LaunchContractError(
                f"smoke bundle audit exact contract invalid: {exc}"
            ) from exc


def _validate_recipe_env(recipe: Mapping[str, Any]) -> list[str]:
    env_map = recipe.get("trainer_env")
    try:
        normalized = require_model_native_recipe_env(env_map)  # type: ignore[arg-type]
    except RuntimeError as exc:
        raise LaunchContractError(f"recipe trainer_env is not canonical: {exc}") from exc
    _require(recipe.get("trainer_env_sha256") == recipe_env_sha256(normalized), "recipe trainer_env_sha256 mismatch")
    return [f"{key}={normalized[key]}" for key in MODEL_NATIVE_RECIPE_ENV_KEYS]


def _trainer_cli_contract(args: argparse.Namespace) -> dict[str, Any]:
    gx1_data_root = Path(args.gx1_data_root).expanduser()
    _require(gx1_data_root.is_absolute(), "gx1_data_root must be absolute")
    _require(gx1_data_root.exists() and gx1_data_root.is_dir(), "gx1_data_root must exist")
    _require(str(gx1_data_root.resolve(strict=True)) == str(gx1_data_root), "gx1_data_root must be canonical")
    _require(_CAP_RE.fullmatch(str(args.memory_cap)) is not None, "memory_cap must be an explicit positive K/M/G/T value")
    _require(_CAP_RE.fullmatch(str(args.swap_cap)) is not None, "swap_cap must be an explicit positive K/M/G/T value")

    integer_values = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "early_stop_patience": int(args.early_stop_patience),
        "subsample_rows": int(args.subsample_rows),
    }
    _require(integer_values["seed"] >= 0, "seed must be >= 0")
    for key in ("epochs", "batch_size", "early_stop_patience"):
        _require(integer_values[key] > 0, f"{key} must be > 0")
    _require(integer_values["subsample_rows"] >= 0, "subsample_rows must be >= 0")

    float_values = {
        "learning_rate": float(args.learning_rate),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "grad_clip_norm": float(args.grad_clip_norm),
        "weight_decay": float(args.weight_decay),
        "multi_tf_scale": float(args.multi_tf_scale),
        "specialist_fusion_scale": float(args.specialist_fusion_scale),
    }
    for key, value in float_values.items():
        _require(math.isfinite(value), f"{key} must be finite")
        if key == "early_stop_min_delta":
            _require(value >= 0.0, f"{key} must be >= 0")
        else:
            _require(value > 0.0, f"{key} must be > 0")

    return {
        "device": str(args.device),
        **integer_values,
        **float_values,
        "memory_cap": str(args.memory_cap),
        "swap_cap": str(args.swap_cap),
        "gx1_data_root": str(gx1_data_root),
    }


def build_recipe_audit_payload(
    args: argparse.Namespace,
    *,
    created_utc: str,
) -> dict[str, Any]:
    """Construct and self-validate one recipe without binding its own bytes."""

    run_id = require_entry_run_id(args.run_id)
    profile = str(args.profile)
    _require(profile in _PROFILE_BINDING_KEYS, f"unsupported launch profile: {profile}")
    repo = Path(args.repo).expanduser().resolve(strict=True)
    wrapper_path = Path(args.wrapper_path).expanduser().resolve(strict=True)
    dataset_dir = _resolved_explicit_path(args.dataset_dir, "dataset_dir", directory=True)
    out_bundle_dir = _resolved_output_path(args.out_bundle_dir)
    trainer_cli = _trainer_cli_contract(args)
    artifact_keys = (*_COMMON_BINDING_KEYS, *_PROFILE_BINDING_KEYS[profile])
    artifacts = {
        key: _resolved_explicit_path(str(getattr(args, key)), key)
        for key in artifact_keys
    }
    payloads = {
        key: _read_json(path, key)
        for key, path in artifacts.items()
        if key.endswith("_json")
    }
    expected_large_hashes = _validate_post_rebuild_readiness(
        payloads["post_rebuild_readiness_json"],
        artifacts=artifacts,
        dataset_dir=dataset_dir,
    )
    expected_large_hashes["m5_prebuilt_path"] = _m5_source_hash_from_manifests(
        payloads
    )
    artifact_bindings = {
        key: artifact_binding(
            path,
            content_sha256=(
                expected_large_hashes[key] if key in _LARGE_ARTIFACT_KEYS else None
            ),
        )
        for key, path in artifacts.items()
    }
    source_bindings = recipe_source_bindings(
        repo=repo,
        wrapper_path=wrapper_path,
    )
    recipe = {
        "schema_version": RECIPE_AUDIT_SCHEMA,
        "created_utc": created_utc,
        "decision": "PASS",
        "failures": [],
        "profile": profile,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "expected_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "execution_allowed": True,
        "activation_authority": False,
        "report_only": True,
        "side_effects_started": {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
        },
        "run_id": run_id,
        "dataset_dir": str(dataset_dir),
        "out_bundle_dir": str(out_bundle_dir),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "trainer_cli": trainer_cli,
        "trainer_cli_sha256": canonical_json_sha256(trainer_cli),
        "trainer_env": dict(MODEL_NATIVE_RECIPE_ENV),
        "trainer_env_sha256": recipe_env_sha256(MODEL_NATIVE_RECIPE_ENV),
        "trainer_env_contract": model_native_recipe_env_contract_metadata(),
        "artifact_bindings": artifact_bindings,
        "artifact_bindings_sha256": canonical_json_sha256(artifact_bindings),
        "large_artifact_sha256": {
            key: expected_large_hashes[key]
            for key in sorted(_LARGE_ARTIFACT_KEYS)
        },
        "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_json_sha256(source_bindings),
    }
    validate_launch(args, recipe_payload=recipe)
    return recipe


def validate_launch(
    args: argparse.Namespace,
    *,
    recipe_payload: Mapping[str, Any] | None = None,
) -> list[str]:
    run_id = require_entry_run_id(args.run_id)
    profile = str(args.profile)
    _require(profile in _PROFILE_BINDING_KEYS, f"unsupported launch profile: {profile}")
    repo = Path(args.repo).expanduser().resolve(strict=True)
    wrapper_path = Path(args.wrapper_path).expanduser().resolve(strict=True)
    _require(wrapper_path.parent == repo / "scripts", "wrapper path is outside the repository scripts directory")

    dataset_dir = _resolved_explicit_path(args.dataset_dir, "dataset_dir", directory=True)
    out_bundle_dir = _resolved_output_path(args.out_bundle_dir)
    trainer_cli = _trainer_cli_contract(args)
    artifact_keys = (*_COMMON_BINDING_KEYS, *_PROFILE_BINDING_KEYS[profile])
    artifacts = {
        key: _resolved_explicit_path(str(getattr(args, key)), key)
        for key in artifact_keys
    }
    for key in ("train_parquet", "val_parquet", "test_parquet", "train_manifest_json", "val_manifest_json", "test_manifest_json"):
        _require(artifacts[key].parent == dataset_dir, f"{key} must be directly inside dataset_dir")

    payloads = {
        key: _read_json(path, key)
        for key, path in artifacts.items()
        if key.endswith("_json")
    }
    for split in ("train", "val", "test"):
        _validate_split_manifest(
            payloads[f"{split}_manifest_json"],
            path=artifacts[f"{split}_manifest_json"],
            parquet=artifacts[f"{split}_parquet"],
            m5_prebuilt=artifacts["m5_prebuilt_path"],
        )
    expected_large_hashes = _validate_post_rebuild_readiness(
        payloads["post_rebuild_readiness_json"],
        artifacts=artifacts,
        dataset_dir=dataset_dir,
    )
    expected_large_hashes["m5_prebuilt_path"] = _m5_source_hash_from_manifests(
        payloads
    )
    _validate_audits(artifacts, payloads, dataset_dir=dataset_dir, profile=profile)

    _validate_pretrain_audit(
        payloads["pretrain_audit_json"],
        artifacts=artifacts,
        dataset_dir=dataset_dir,
    )
    # recipe_audit_json is intentionally outside artifact_keys because a recipe
    # cannot bind its own bytes. The producer validates its in-memory payload
    # through this same path before immutable publication.
    if recipe_payload is None:
        recipe_path = _resolved_explicit_path(
            args.recipe_audit_json,
            "recipe_audit_json",
        )
        _require(
            bool(_STAMP_RE.search(str(recipe_path))),
            f"recipe_audit_json must contain an immutable UTC timestamp: {recipe_path}",
        )
        recipe = _read_json(recipe_path, "recipe audit")
    else:
        recipe = dict(recipe_payload)
    _zero_failure(recipe, label="recipe audit", schema=RECIPE_AUDIT_SCHEMA, decision="PASS")
    _require(recipe.get("profile") == profile, "recipe audit profile mismatch")
    _require(recipe.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE, "recipe audit mode mismatch")
    _require(recipe.get("direction_logit_mode") == MODEL_NATIVE_DIRECTION_LOGIT_MODE, "recipe audit direction mode mismatch")
    _require(int(recipe.get("expected_signal_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM, "recipe audit signal width mismatch")
    _require(int(recipe.get("expected_selected_feature_count") or 0) == MODEL_NATIVE_SELECTED_FEATURE_COUNT, "recipe audit selected width mismatch")
    _require(recipe.get("execution_allowed") is True, "recipe audit does not authorize execution")
    _require(
        recipe.get("activation_authority") is False
        and recipe.get("report_only") is True,
        "recipe audit activation/report-only boundary mismatch",
    )
    _require(
        recipe.get("side_effects_started")
        == {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
        },
        "recipe audit side-effect map is not exact zero",
    )
    _require(
        isinstance(recipe.get("created_utc"), str)
        and bool(str(recipe.get("created_utc")).strip()),
        "recipe audit created_utc missing",
    )
    _require(recipe.get("run_id") == run_id, "recipe audit run_id mismatch")
    _require(Path(str(recipe.get("dataset_dir") or "")).resolve() == dataset_dir, "recipe audit dataset mismatch")
    _require(Path(str(recipe.get("out_bundle_dir") or "")).resolve() == out_bundle_dir, "recipe audit output mismatch")
    _validate_source_revision(recipe, repo=repo)
    _require(recipe.get("trainer_cli") == trainer_cli, "recipe audit trainer_cli mismatch")
    _require(recipe.get("trainer_cli_sha256") == canonical_json_sha256(trainer_cli), "recipe audit trainer_cli_sha256 mismatch")
    _require(
        recipe.get("trainer_env_contract")
        == model_native_recipe_env_contract_metadata(),
        "recipe trainer_env contract mismatch",
    )
    _validate_binding_map(
        recipe,
        label="recipe audit",
        expected_paths={key: artifacts[key] for key in artifact_keys},
    )
    _require(
        recipe.get("large_artifact_sha256") == expected_large_hashes,
        "recipe audit large-artifact hashes do not match post-rebuild readiness",
    )
    _validate_source_bindings(
        recipe,
        repo=repo,
        wrapper_path=wrapper_path,
    )
    env_rows = _validate_recipe_env(recipe)
    recipe_bindings = recipe["artifact_bindings"]
    env_rows.extend(
        f"{env_name}={recipe_bindings[key]['sha256']}"
        for key, env_name in TRAINER_ARTIFACT_HASH_ENV.items()
    )
    return env_rows


def build_parser(*, require_recipe_audit: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(_PROFILE_BINDING_KEYS), required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--wrapper-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-bundle-dir", required=True)
    parser.add_argument("--recipe-audit-json", required=require_recipe_audit)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--batch-size", required=True)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument("--early-stop-patience", required=True)
    parser.add_argument("--early-stop-min-delta", required=True)
    parser.add_argument("--grad-clip-norm", required=True)
    parser.add_argument("--weight-decay", required=True)
    parser.add_argument("--multi-tf-scale", required=True)
    parser.add_argument("--specialist-fusion-scale", required=True)
    parser.add_argument("--subsample-rows", required=True)
    parser.add_argument("--memory-cap", required=True)
    parser.add_argument("--swap-cap", required=True)
    parser.add_argument("--gx1-data-root", required=True)
    for key in _COMMON_BINDING_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), dest=key, required=True)
    for profile_keys in _PROFILE_BINDING_KEYS.values():
        for key in profile_keys:
            if not any(action.dest == key for action in parser._actions):
                parser.add_argument("--" + key.replace("_", "-"), dest=key)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required_profile_keys = _PROFILE_BINDING_KEYS[str(args.profile)]
    missing = [key for key in required_profile_keys if not getattr(args, key, None)]
    if missing:
        print(f"FATAL: missing profile evidence: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        env_rows = validate_launch(args)
    except (LaunchContractError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FATAL: model-native train launch contract rejected: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write("\n".join(env_rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
