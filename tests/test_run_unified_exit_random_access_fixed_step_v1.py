from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import (
    _FreshWeightEma,
    _ParentSampler,
    _absolute_optimizer_step,
    _bind_multi_tf_cache_from_source_bundle_metadata,
)


def test_parent_sampler_converts_batch_cursor_to_parent_row_offset() -> None:
    rows = tuple(range(100, 200))
    sampler = _ParentSampler(rows, batch_offset=3, batch_size=16)
    assert list(sampler)[:16] == list(range(148, 164))
    assert len(sampler) == 52


def test_parent_child_entry_mapping_is_exact_and_fail_closed() -> None:
    dataset = EntryV10CtxDataset.__new__(EntryV10CtxDataset)
    dataset.df = pd.DataFrame({"x": range(6)})
    dataset._random_access_child_index_by_parent = None
    dataset.bind_random_access_entry_coordinate_mapping_v1(
        parent_entry_row_indices=[2, 3, 4], child_entry_row_indices=[0, 1, 2]
    )
    assert dataset._random_access_child_index_by_parent == {2: 0, 3: 1, 4: 2}
    with pytest.raises(RuntimeError, match="ENTRY_MAPPING_INVALID"):
        dataset.bind_random_access_entry_coordinate_mapping_v1(
            parent_entry_row_indices=[2], child_entry_row_indices=[0]
        )


def test_fresh_ema_round_trip_is_exact() -> None:
    model = torch.nn.Linear(3, 2)
    ema = _FreshWeightEma(model, 0.9)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    state = ema.state_dict()
    restored = _FreshWeightEma(model, 0.9)
    restored.load_state_dict(state)
    assert restored.steps == 1
    assert all(
        torch.equal(restored.shadow[name], tensor)
        for name, tensor in state["shadow"].items()
    )


def test_capped_runner_allowlists_only_exact_fixed_step_module() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/gx1_capped_run.sh"
    ).read_text()
    assert (
        "RANDOM_ACCESS_FIXED_STEP_MODULE=gx1.scripts.run_unified_exit_random_access_fixed_step_v1"
        in source
    )
    assert 'if [[ "$module" == "$RANDOM_ACCESS_FIXED_STEP_MODULE" ]]' in source
    assert (
        "random-access fixed-step smoke/train requires the exact attended CUDA stage contract"
        in source
    )


def test_dataloader_iterator_does_not_advance_model_dropout_rng() -> None:
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    loader = DataLoader(
        TensorDataset(torch.arange(64)),
        batch_size=16,
        sampler=_ParentSampler(tuple(range(64)), batch_offset=0, batch_size=16),
        generator=torch.Generator().manual_seed(999),
    )
    next(iter(loader))
    assert torch.equal(torch.rand(4), expected)


def test_checkpoint_global_step_counts_optimizer_steps_not_writes() -> None:
    assert _absolute_optimizer_step(
        initial_global_step=77, start_batch_offset=128, next_batch_offset=192
    ) == 141
    with pytest.raises(RuntimeError, match="CHECKPOINT_CURSOR_INVALID"):
        _absolute_optimizer_step(
            initial_global_step=77, start_batch_offset=128, next_batch_offset=128
        )


def test_capped_runner_has_exact_non_attended_full_val_allowlist() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/gx1_capped_run.sh"
    ).read_text()
    assert "RANDOM_ACCESS_VAL_MODULE=gx1.scripts.run_unified_exit_random_access_val_v1" in source
    assert 'if [[ "$module" == "$RANDOM_ACCESS_VAL_MODULE" ]]' in source
    assert "random-access full VAL requires the exact guarded campaign contract" in source
    assert "GX1_CAMPAIGN_GUARD_LOG_PATH" in source
    assert "require_campaign_plan_environment" in source
    assert "GX1_CAMPAIGN_PLAN_PATH" in source
    assert "GX1_CAMPAIGN_PLAN_FILE_SHA256" in source
    assert "/usr/bin/sha256sum \"$plan_path\"" in source
    assert '--setenv=GX1_CAMPAIGN_PLAN_PATH=' in source
    assert '--setenv=GX1_CAMPAIGN_PLAN_FILE_SHA256=' in source
    assert "set -o noclobber" in source
    for flag in (
        "--launch-manifest",
        "--final-train-checkpoint-authority",
        "--final-train-checkpoint-authority-file-sha256",
        "--checkpoint-pointer",
        "--progress-path",
        "--rollout-progress-path",
        "--result-path",
        "--max-forwards-this-invocation",
        "--progress-interval-forwards",
        "--compute-guard-max-model-forwards",
        "--compute-guard-max-materialized-state-views",
        "--compute-guard-max-wall-seconds",
    ):
        assert flag in source


def _multi_tf_cache_metadata(tmp_path: Path) -> dict:
    cache_dir = tmp_path / "MULTI_TF_V4_CACHE"
    cache_dir.mkdir()
    identity = "a" * 64
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"cache_identity_sha256": identity}, sort_keys=True)
    )
    return {
        "multi_tf": {
            "shared_cache_dir": str(cache_dir),
            "shared_cache_manifest_path": str(manifest_path),
            "shared_cache_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "shared_cache_identity_sha256": identity,
        }
    }


def test_source_bundle_binds_exact_multi_tf_cache_before_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GX1_V10_MULTI_TF_V4_CACHE_DIR", raising=False)
    metadata = _multi_tf_cache_metadata(tmp_path)
    cache_dir = _bind_multi_tf_cache_from_source_bundle_metadata(metadata)
    assert cache_dir == tmp_path / "MULTI_TF_V4_CACHE"
    assert cache_dir.is_absolute()
    assert (
        os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"]
        == str(cache_dir)
    )


def test_multi_tf_cache_binding_fails_closed_when_missing_or_hash_mismatched(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GX1_V10_MULTI_TF_V4_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="CACHE_BINDING_MISSING"):
        _bind_multi_tf_cache_from_source_bundle_metadata({"multi_tf": {}})
    metadata = _multi_tf_cache_metadata(tmp_path)
    metadata["multi_tf"]["shared_cache_manifest_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="CACHE_MANIFEST_MISMATCH"):
        _bind_multi_tf_cache_from_source_bundle_metadata(metadata)
    assert "GX1_V10_MULTI_TF_V4_CACHE_DIR" not in os.environ


def test_multi_tf_cache_binding_rejects_symlink_and_manifest_path_drift(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GX1_V10_MULTI_TF_V4_CACHE_DIR", raising=False)
    metadata = _multi_tf_cache_metadata(tmp_path)
    cache_dir = tmp_path / "MULTI_TF_V4_CACHE"
    real_manifest = cache_dir / "manifest.real.json"
    manifest_path = cache_dir / "manifest.json"
    manifest_path.rename(real_manifest)
    manifest_path.symlink_to(real_manifest)
    metadata["multi_tf"]["shared_cache_manifest_sha256"] = hashlib.sha256(
        real_manifest.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="CACHE_PATH_INVALID"):
        _bind_multi_tf_cache_from_source_bundle_metadata(metadata)

    manifest_path.unlink()
    drifted = cache_dir / "other.json"
    real_manifest.rename(drifted)
    metadata["multi_tf"]["shared_cache_manifest_path"] = str(drifted)
    metadata["multi_tf"]["shared_cache_manifest_sha256"] = hashlib.sha256(
        drifted.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="CACHE_PATH_INVALID"):
        _bind_multi_tf_cache_from_source_bundle_metadata(metadata)
    assert "GX1_V10_MULTI_TF_V4_CACHE_DIR" not in os.environ


@pytest.mark.parametrize("corrupt_checkpoint", [False, True])
def test_guard_model_signal_follows_cpu_checkpoint_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_checkpoint: bool
) -> None:
    from types import SimpleNamespace
    import gx1.scripts.run_unified_exit_random_access_fixed_step_v1 as runner

    class ModelBoundaryReached(Exception):
        pass

    checkpoint_root = tmp_path / "checkpoints"
    files = {name: tmp_path / (name + ".json") for name in (
        "selected_sampler", "random_access_root", "source_bundle_metadata",
        "child_composite_normalization", "bootstrap_composite_normalization",
        "entry_train_parquet", "entry_val_parquet", "m5_prebuilt",
        "sequence_source_audit", "feature_lifecycle_root", "entry_train_manifest",
        "entry_val_manifest", "economics_readiness", "train_cost_authority",
        "train_random_access_index", "checkpoint_pointer",
    )}
    for path in files.values():
        path.write_text("{}")
    state_path = tmp_path / "candidate_training_state_slot_1.pt"
    torch.save({"model_state": {"w": torch.ones(1)},
                "target_model_state": {"w": torch.ones(1)}}, state_path)
    expected_sha = runner.file_sha256(state_path)
    launch = {"files": {k: {"path": str(v)} for k, v in files.items()},
              "checkpoint_dir": str(checkpoint_root), "batch_sizes": [4, 8, 16],
              "source_repo": str(tmp_path), "source_commit": "a" * 40,
              "dataset_run_id": "fixture", "seed": 7}
    documents = {path: {} for path in files.values()}
    documents[files["selected_sampler"]] = {
        "random_access_root": {"path": str(files["random_access_root"])},
        "candidate_set": {"path": str(tmp_path / "candidate.json")},
        "selected_sampler_contract_sha256": "b" * 64, "artifact_sha256": "c" * 64}
    documents[files["source_bundle_metadata"]] = {
        "seq_len": 1, "multi_tf": {name + "_seq_len": 1 for name in ("m5", "m15", "h1", "h4", "d1")}}
    documents[files["child_composite_normalization"]] = {
        "base_feature_normalization": {"artifact": {"contract": {}}}}
    documents[files["bootstrap_composite_normalization"]] = {
        "base_feature_normalization": {"artifact": {"base_artifact": {"contract": {}}}}}
    documents[files["checkpoint_pointer"]] = {
        "slot": 1, "state_sha256": "0" * 64 if corrupt_checkpoint else expected_sha}
    manifest = tmp_path / "launch.json"
    documents[manifest] = launch
    monkeypatch.setattr(runner, "_read", lambda path: documents[path])
    for name in ("require_launch_manifest", "require_selected_sampler_artifact",
                 "require_composite_normalization_binding", "require_bootstrap_composite_normalization"):
        monkeypatch.setattr(runner, name, lambda value: value)
    monkeypatch.setattr(runner.subprocess, "run", lambda args, **kw:
                        SimpleNamespace(stdout="a" * 40 if "rev-parse" in args else ""))
    monkeypatch.setattr(runner, "_set_deterministic", lambda *args: None)
    monkeypatch.setattr(runner, "_bind_multi_tf_cache_from_source_bundle_metadata", lambda meta: None)
    dataset = SimpleNamespace(bind_unified_exit_lifecycle_v2=lambda adapter: None,
                              bind_random_access_entry_coordinate_mapping_v1=lambda **kw: None)
    monkeypatch.setattr(runner, "EntryV10CtxDataset", lambda *args, **kw: dataset)
    monkeypatch.setattr(runner, "UnifiedExitLifecycleCorpus", lambda **kw:
                        SimpleNamespace(splits={"train": object()}))
    adapter = SimpleNamespace(set_epoch_index=lambda epoch: None,
                              random_access_selected_entry_rows_v1=lambda: list(range(16384)))
    monkeypatch.setattr(runner, "build_random_access_train_adapter_factory_v1",
                        lambda **kw: lambda budget: adapter)
    monkeypatch.setattr(runner.pd, "read_parquet", lambda *args, **kw: pd.DataFrame({
        "entry_row_index": range(16384), "parent_entry_row_index": range(16384)}))
    fifo = tmp_path / "guard-stage"
    os.mkfifo(fifo, 0o600)
    fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    monkeypatch.setenv("GX1_TRAINER_ATTENDED_STAGE_FIFO", str(fifo))
    monkeypatch.setenv("GX1_TRAINER_ATTENDED_STAGE_TOKEN", "d" * 64)

    def model_boundary(*args):
        assert runner.file_sha256(state_path) == expected_sha
        assert os.read(fd, 256) == ("gx1_attended_preflight_ready_v1:" + "d" * 64 + "\n").encode()
        assert not (checkpoint_root / "batch_4").exists()
        raise ModelBoundaryReached

    monkeypatch.setattr(runner, "_model", model_boundary)
    try:
        expected_error = RuntimeError if corrupt_checkpoint else ModelBoundaryReached
        with pytest.raises(expected_error, match="SOURCE_STATE_INVALID" if corrupt_checkpoint else None):
            runner.run(manifest_path=manifest, stage="smoke-arm", device=torch.device("cuda"),
                       checkpoint_dir=checkpoint_root / "batch_4", arm_batch_size=4,
                       progress_path=checkpoint_root / "batch_4" / "PROGRESS.json",
                       train_session_manifest_path=None, max_optimizer_steps=None,
                       plan_sha256="e" * 64, invocation_sha256="f" * 64)
        with pytest.raises(BlockingIOError):
            os.read(fd, 256)
    finally:
        os.close(fd)
