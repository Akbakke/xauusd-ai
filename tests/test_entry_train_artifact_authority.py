from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_signal_contract_metadata,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    RETIRED_RANK_STATE_FIELDS,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
)
from gx1.features.htf_features import HTF_V4_CACHE_SCHEMA_VERSION
from tests.model_native_signal_support import canonical_model_native_selected_fields


RUN_ID = "MODEL_NATIVE_TRAIN_ARTIFACT_PYTEST_V1"
DATASET_RUN_ID = "MODEL_NATIVE_TRAIN_DATASET_PYTEST_V1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_exact_trainer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith("ENTRY_") or key.startswith("GX1_"):
            monkeypatch.delenv(key)
    for key, value in MODEL_NATIVE_RECIPE_ENV.items():
        monkeypatch.setenv(key, value)
    for env_name in trainer._TRAIN_ARTIFACT_HASH_ENV.values():
        monkeypatch.setenv(env_name, "0" * 64)
    monkeypatch.setenv(trainer._TRAIN_DATASET_RUN_ID_ENV, DATASET_RUN_ID)


def _aux_target_contract(split: str) -> dict[str, object]:
    candidates = {"train": 100, "val": 101}[split]
    return {
        **model_native_aux_target_contract_metadata(),
        "incomplete_tail_rows_total": 96,
        "candidate_rows_before_completeness": candidates,
        "incomplete_candidate_rows_excluded": 96,
        "complete_rows_emitted": candidates - 96,
    }


def _artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    root = (tmp_path / "dataset_20260719T220000123456Z").resolve()
    root.mkdir()
    m5_prebuilt = root / "FULL_PLUS_CTX_v3src.parquet"
    m5_prebuilt.write_bytes(b"immutable-xau-m5-source")
    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.train_artifact_authority"
    )
    signal_contract = model_native_signal_contract_metadata(selected)
    state_contract = {
        "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
        "entry_run_id": DATASET_RUN_ID,
        "feature_history_start_utc": "2021-01-05T00:00:00Z",
        "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
        "split_reset_allowed": False,
        "runtime_rule_free": True,
    }
    cache_dir = root / "MULTI_TF_V4_CACHE"
    cache_dir.mkdir()
    cache_manifest = cache_dir / "manifest.json"
    cache_identity = "a" * 64
    cache_manifest.write_text(
        json.dumps(
            {
                "schema_version": HTF_V4_CACHE_SCHEMA_VERSION,
                "cache_identity_sha256": cache_identity,
                "m5_prebuilt_source": str(m5_prebuilt),
                "m5_prebuilt_source_sha256": _sha(m5_prebuilt),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    mtf_cache_binding = {
        "cache_dir": str(cache_dir),
        "manifest_path": str(cache_manifest),
        "manifest_sha256": _sha(cache_manifest),
        "cache_identity_sha256": cache_identity,
        "m5_prebuilt_source": str(m5_prebuilt),
        "m5_prebuilt_source_sha256": _sha(m5_prebuilt),
    }
    monkeypatch.setenv(trainer._TRAIN_MULTI_TF_CACHE_ENV, str(cache_dir))
    manifests: dict[str, Path] = {}
    parquets: dict[str, Path] = {}
    for split in ("train", "val"):
        parquet = root / f"entry_model_native_{split}.parquet"
        parquet.write_bytes(f"immutable-{split}".encode())
        manifest = root / f"entry_model_native_{split}.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
                    "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                    "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
                    "output_data_path": str(parquet),
                    "inputs": {"source_parquet": str(m5_prebuilt)},
                    "extra": {
                        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
                        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
                        "model_native_signal_contract": signal_contract,
                        "signal_bridge": {
                            "fields": signal_contract["fields"],
                            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                        },
                        "aux_head_target_contract": _aux_target_contract(split),
                        "entry_run_id": DATASET_RUN_ID,
                        "model_native_state_contract": state_contract,
                        "multi_tf_cache_binding": mtf_cache_binding,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifests[split] = manifest
        parquets[split] = parquet
        monkeypatch.setenv(
            trainer._TRAIN_ARTIFACT_HASH_ENV[f"{split}_manifest"], _sha(manifest)
        )
        monkeypatch.setenv(
            trainer._TRAIN_ARTIFACT_HASH_ENV[f"{split}_parquet"], _sha(parquet)
        )
    monkeypatch.setenv(trainer._TRAIN_DATASET_RUN_ID_ENV, DATASET_RUN_ID)
    monkeypatch.setenv(
        trainer._TRAIN_ARTIFACT_HASH_ENV["m5_prebuilt_path"],
        _sha(m5_prebuilt),
    )
    lifecycle_manifest = root / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
    lifecycle_manifest.write_text(
        json.dumps(
            {
                "schema_version": (
                    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
                ),
                "decision": "PASS",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        trainer._TRAIN_ARTIFACT_HASH_ENV[
            "unified_exit_lifecycle_manifest"
        ],
        _sha(lifecycle_manifest),
    )
    return manifests, parquets, m5_prebuilt


def _resolve(
    manifests: dict[str, Path],
    parquets: dict[str, Path],
    m5_prebuilt: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    return trainer._resolve_explicit_train_split_artifacts(
        train_manifest=manifests["train"],
        val_manifest=manifests["val"],
        train_parquet=parquets["train"],
        val_parquet=parquets["val"],
        unified_exit_lifecycle_manifest_path=(
            manifests["train"].parent
            / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
        ),
        m5_prebuilt_path=m5_prebuilt,
        dataset_run_id=DATASET_RUN_ID,
        profile="candidate",
    )


def test_exact_dataset_artifact_identity_including_m5_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)

    observed_manifests, observed_parquets = _resolve(
        manifests,
        parquets,
        m5_prebuilt,
    )

    assert observed_manifests == manifests
    assert observed_parquets == parquets


def test_train_and_val_must_bind_the_same_exact_mtf_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    val_manifest = json.loads(manifests["val"].read_text(encoding="utf-8"))
    val_manifest["extra"]["multi_tf_cache_binding"][
        "cache_identity_sha256"
    ] = "b" * 64
    manifests["val"].write_text(
        json.dumps(val_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        trainer._TRAIN_ARTIFACT_HASH_ENV["val_manifest"],
        _sha(manifests["val"]),
    )

    with pytest.raises(RuntimeError, match="SPLIT_MTF_CACHE_BINDING_MISMATCH"):
        _resolve(manifests, parquets, m5_prebuilt)


def test_launch_cache_path_must_equal_dataset_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    wrong_cache = tmp_path / "WRONG_MTF_CACHE"
    wrong_cache.mkdir()
    monkeypatch.setenv(trainer._TRAIN_MULTI_TF_CACHE_ENV, str(wrong_cache))

    with pytest.raises(RuntimeError, match="cache_dir differs from launch"):
        _resolve(manifests, parquets, m5_prebuilt)


def test_trainer_boundary_requires_exact_recipe_and_allows_only_bound_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_trainer_env(monkeypatch)

    trainer._enforce_canonical_train_env_contract()

    monkeypatch.setenv("ENTRY_DIRECTION_CE_SCALE", "999")
    with pytest.raises(RuntimeError, match="AMBIENT_CONTROL_FORBIDDEN"):
        trainer._enforce_canonical_train_env_contract()


def test_trainer_boundary_allows_only_guard_owned_attended_stage_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_trainer_env(monkeypatch)
    guarded_transport = {
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "300",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "true",
        "GX1_TRAINER_ATTENDED_STAGE_FIFO": "/tmp/guard-created-preflight",
        "GX1_TRAINER_ATTENDED_STAGE_TOKEN": "a" * 64,
    }
    for key, value in guarded_transport.items():
        monkeypatch.setenv(key, value)

    trainer._enforce_canonical_train_env_contract()
    assert set(guarded_transport).issubset(trainer._TRAIN_CAPPED_SCOPE_ENV)


@pytest.mark.parametrize("recipe_key", sorted(MODEL_NATIVE_RECIPE_ENV))
def test_trainer_boundary_rejects_every_missing_recipe_key(
    monkeypatch: pytest.MonkeyPatch,
    recipe_key: str,
) -> None:
    """Each declared recipe key is mandatory; the set comes from the owner.

    Naming a key literally here is the stale-gate class rule 13 forbids: the
    recipe schema is the authority, so the parametrization derives from it.
    """
    _set_exact_trainer_env(monkeypatch)
    monkeypatch.delenv(recipe_key)

    with pytest.raises(RuntimeError, match="TRAIN_RECIPE_ENV_INVALID"):
        trainer._enforce_canonical_train_env_contract()


def test_trainer_boundary_rejects_ambient_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_exact_trainer_env(monkeypatch)

    trainer._enforce_canonical_train_env_contract()

    monkeypatch.setenv("GX1_MTF_TAPERED", "1")
    with pytest.raises(RuntimeError, match="AMBIENT_CONTROL_FORBIDDEN"):
        trainer._enforce_canonical_train_env_contract()


def test_dataset_run_id_must_match_launch_owned_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    monkeypatch.setenv(
        trainer._TRAIN_DATASET_RUN_ID_ENV,
        "DIFFERENT_DATASET_RUN_ID",
    )

    with pytest.raises(RuntimeError, match="DATASET_RUN_ID_ENV_MISMATCH"):
        _resolve(manifests, parquets, m5_prebuilt)


@pytest.mark.parametrize("mode", ("missing", "mismatch"))
def test_hash_env_is_mandatory_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    env_name = trainer._TRAIN_ARTIFACT_HASH_ENV["val_parquet"]
    if mode == "missing":
        monkeypatch.delenv(env_name)
        expected = "HASH_ENV_INVALID"
    else:
        monkeypatch.setenv(env_name, "0" * 64)
        expected = "SHA256_MISMATCH"

    with pytest.raises(RuntimeError, match=expected):
        _resolve(manifests, parquets, m5_prebuilt)


def test_relative_symlink_and_latest_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="PATH_NOT_ABSOLUTE"):
        trainer._resolve_explicit_train_split_artifacts(
            train_manifest=Path("relative_train.manifest.json"),
            val_manifest=manifests["val"],
            train_parquet=parquets["train"],
            val_parquet=parquets["val"],
            unified_exit_lifecycle_manifest_path=(
                manifests["train"].parent
                / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
            ),
            m5_prebuilt_path=m5_prebuilt,
            dataset_run_id=DATASET_RUN_ID,
            profile="candidate",
        )

    symlink = manifests["train"].with_name("symlink_train.manifest.json")
    symlink.symlink_to(manifests["train"])
    with pytest.raises(RuntimeError, match="NOT_REGULAR"):
        _resolve({**manifests, "train": symlink}, parquets, m5_prebuilt)

    latest = manifests["train"].parent / "latest" / manifests["train"].name
    latest.parent.mkdir()
    latest.write_bytes(manifests["train"].read_bytes())
    monkeypatch.setenv(trainer._TRAIN_ARTIFACT_HASH_ENV["train_manifest"], _sha(latest))
    with pytest.raises(RuntimeError, match="PATH_MUTABLE"):
        _resolve({**manifests, "train": latest}, parquets, m5_prebuilt)


def test_split_paths_must_be_four_way_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="PATHS_NOT_DISTINCT"):
        _resolve(
            manifests,
            {**parquets, "val": parquets["train"]},
            m5_prebuilt,
        )


@pytest.mark.parametrize("mutation", ("self_path", "run_id"))
def test_manifest_self_path_and_run_lineage_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    manifest = manifests["val"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "self_path":
        payload["output_data_path"] = str(parquets["train"])
        expected = "SELF_PATH_MISMATCH"
    else:
        payload["extra"]["model_native_state_contract"]["entry_run_id"] = (
            "DIFFERENT_RUN_ID"
        )
        expected = "RUN_ID_LINEAGE_MISMATCH"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setenv(trainer._TRAIN_ARTIFACT_HASH_ENV["val_manifest"], _sha(manifest))

    with pytest.raises(RuntimeError, match=expected):
        _resolve(manifests, parquets, m5_prebuilt)


@pytest.mark.parametrize(
    "mutation",
    ("hash_env", "manifest_source", "state_invalid"),
)
def test_m5_source_is_hash_and_manifest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    expected = "M5_"
    if mutation == "hash_env":
        monkeypatch.setenv(
            trainer._TRAIN_ARTIFACT_HASH_ENV["m5_prebuilt_path"],
            "0" * 64,
        )
        expected = "SHA256_MISMATCH"
    else:
        manifest = manifests["val"]
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if mutation == "manifest_source":
            payload["inputs"]["source_parquet"] = str(parquets["train"])
            expected = "M5_SOURCE_PATH_MISMATCH"
        else:
            payload["extra"]["model_native_state_contract"][
                "feature_history_mode"
            ] = "per_split_reset"
            expected = "STATE_BINDING_MISMATCH"
        manifest.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(
            trainer._TRAIN_ARTIFACT_HASH_ENV["val_manifest"],
            _sha(manifest),
        )

    with pytest.raises(RuntimeError, match=expected):
        _resolve(manifests, parquets, m5_prebuilt)


@pytest.mark.parametrize("retired_field", sorted(RETIRED_RANK_STATE_FIELDS))
def test_split_state_carrying_a_retired_rank_field_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retired_field: str,
) -> None:
    """A split built against the retired fixed top-k rank subsystem is rejected.

    The rank reference no longer exists, so a manifest still declaring one is a
    stale artifact from a superseded dataset generation. Training on it would
    silently mix two feature-selection contracts.
    """
    manifests, parquets, m5_prebuilt = _artifacts(tmp_path, monkeypatch)
    manifest = manifests["val"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["extra"]["model_native_state_contract"][retired_field] = "stale-value"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setenv(
        trainer._TRAIN_ARTIFACT_HASH_ENV["val_manifest"],
        _sha(manifest),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"STATE_BINDING_MISMATCH.*RETIRED_FIELDS_FORBIDDEN.*{retired_field}",
    ):
        _resolve(manifests, parquets, m5_prebuilt)


def test_wrappers_forward_train_val_artifacts_without_inference() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts" / "run_entry_model_native_seq513_train.sh"
    ).read_text(encoding="utf-8")
    train_command = source[source.index("TRAIN_CMD=(") : source.index("RUN_CMD=(")]
    for flag in (
        "--train-manifest-json",
        "--val-manifest-json",
        "--train-parquet",
        "--val-parquet",
        "--prefreeze-test-seal-json",
        "--prefreeze-test-seal-sha256",
    ):
        assert flag in train_command
    assert "--test-manifest-json" not in train_command
    assert "--test-parquet" not in train_command
    assert "--dataset_manifest" not in train_command
    assert "--dataset_train_parquet" not in train_command


def test_trainer_source_has_no_split_discovery_or_stem_inference() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "def _resolve_train_val_parquets" not in source
    assert "def _resolve_test_parquet" not in source
    assert 'glob("*.parquet")' not in source
    assert "inferred from train" not in source
    assert "--test-manifest-json" not in source
    assert "--test-parquet" not in source
    assert "test_metrics" not in source.lower()
