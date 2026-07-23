from __future__ import annotations

import hashlib
import json
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
from tests.model_native_signal_support import canonical_model_native_selected_fields


RUN_ID = "MODEL_NATIVE_TRAIN_ARTIFACT_PYTEST_V1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aux_target_contract(split: str) -> dict[str, object]:
    candidates = {"train": 100, "val": 101, "test": 102}[split]
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
) -> tuple[dict[str, Path], dict[str, Path]]:
    root = (tmp_path / "dataset_20260719T220000123456Z").resolve()
    root.mkdir()
    selected = canonical_model_native_selected_fields(
        remainder_prefix="session_regime.train_artifact_authority"
    )
    signal_contract = model_native_signal_contract_metadata(selected)
    state_contract = {
        "schema_version": "entry_model_native_state_contract_v2",
        "entry_run_id": RUN_ID,
        "rank_fit_start_utc": "2021-03-16T00:00:00Z",
        "rank_fit_end_utc": "2026-03-31T23:59:59Z",
    }
    manifests: dict[str, Path] = {}
    parquets: dict[str, Path] = {}
    for split in ("train", "val", "test"):
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
                        "model_native_state_contract": state_contract,
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
    return manifests, parquets


def _resolve(
    manifests: dict[str, Path],
    parquets: dict[str, Path],
) -> tuple[dict[str, Path], dict[str, Path]]:
    return trainer._resolve_explicit_train_split_artifacts(
        train_manifest=manifests["train"],
        val_manifest=manifests["val"],
        test_manifest=manifests["test"],
        train_parquet=parquets["train"],
        val_parquet=parquets["val"],
        test_parquet=parquets["test"],
        run_id=RUN_ID,
        profile="candidate",
    )


def test_exact_six_artifact_identity_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, parquets = _artifacts(tmp_path, monkeypatch)

    observed_manifests, observed_parquets = _resolve(manifests, parquets)

    assert observed_manifests == manifests
    assert observed_parquets == parquets


@pytest.mark.parametrize("mode", ("missing", "mismatch"))
def test_hash_env_is_mandatory_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    manifests, parquets = _artifacts(tmp_path, monkeypatch)
    env_name = trainer._TRAIN_ARTIFACT_HASH_ENV["val_parquet"]
    if mode == "missing":
        monkeypatch.delenv(env_name)
        expected = "HASH_ENV_INVALID"
    else:
        monkeypatch.setenv(env_name, "0" * 64)
        expected = "SHA256_MISMATCH"

    with pytest.raises(RuntimeError, match=expected):
        _resolve(manifests, parquets)


def test_relative_symlink_and_latest_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, parquets = _artifacts(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="PATH_NOT_ABSOLUTE"):
        trainer._resolve_explicit_train_split_artifacts(
            train_manifest=Path("relative_train.manifest.json"),
            val_manifest=manifests["val"],
            test_manifest=manifests["test"],
            train_parquet=parquets["train"],
            val_parquet=parquets["val"],
            test_parquet=parquets["test"],
            run_id=RUN_ID,
            profile="candidate",
        )

    symlink = manifests["train"].with_name("symlink_train.manifest.json")
    symlink.symlink_to(manifests["train"])
    with pytest.raises(RuntimeError, match="NOT_REGULAR"):
        _resolve({**manifests, "train": symlink}, parquets)

    latest = manifests["train"].parent / "latest" / manifests["train"].name
    latest.parent.mkdir()
    latest.write_bytes(manifests["train"].read_bytes())
    monkeypatch.setenv(trainer._TRAIN_ARTIFACT_HASH_ENV["train_manifest"], _sha(latest))
    with pytest.raises(RuntimeError, match="PATH_MUTABLE"):
        _resolve({**manifests, "train": latest}, parquets)


def test_split_paths_must_be_six_way_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests, parquets = _artifacts(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="PATHS_NOT_DISTINCT"):
        _resolve(manifests, {**parquets, "val": parquets["train"]})


@pytest.mark.parametrize("mutation", ("self_path", "run_id"))
def test_manifest_self_path_and_run_lineage_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manifests, parquets = _artifacts(tmp_path, monkeypatch)
    manifest = manifests["test"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "self_path":
        payload["output_data_path"] = str(parquets["val"])
        expected = "SELF_PATH_MISMATCH"
    else:
        payload["extra"]["model_native_state_contract"]["entry_run_id"] = (
            "DIFFERENT_RUN_ID"
        )
        expected = "RUN_ID_LINEAGE_MISMATCH"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setenv(trainer._TRAIN_ARTIFACT_HASH_ENV["test_manifest"], _sha(manifest))

    with pytest.raises(RuntimeError, match=expected):
        _resolve(manifests, parquets)


def test_wrappers_forward_all_six_explicit_artifacts_without_inference() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "run_entry_model_native_seq513_smoke_train.sh",
        "run_entry_model_native_seq513_candidate_train.sh",
    ):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        train_command = source[source.index("TRAIN_CMD=(") : source.index("RUN_CMD=(")]
        for flag in (
            "--train-manifest-json",
            "--val-manifest-json",
            "--test-manifest-json",
            "--train-parquet",
            "--val-parquet",
            "--test-parquet",
        ):
            assert flag in train_command
        assert "--dataset_manifest" not in train_command
        assert "--dataset_train_parquet" not in train_command


def test_trainer_source_has_no_split_discovery_or_stem_inference() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "def _resolve_train_val_parquets" not in source
    assert "def _resolve_test_parquet" not in source
    assert 'glob("*.parquet")' not in source
    assert "inferred from train" not in source
