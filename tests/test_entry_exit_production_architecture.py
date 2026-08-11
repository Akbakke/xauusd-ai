from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
import torch

from gx1.contracts.entry_exit_production_architecture_v1 import (
    current_entry_exit_architecture_observation,
    entry_exit_production_architecture_contract,
    require_entry_exit_production_architecture,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.models.entry_v10 import entry_v10_bundle as bundle_module
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
)


def test_exact_current_production_architecture_passes() -> None:
    expected = entry_exit_production_architecture_contract()

    assert require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context="TEST_EXACT",
    ) == expected
    assert expected["entry"]["sequence_bars"] == 96
    assert expected["shared_surface"] == {
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": 142,
        "ctx_cat_dim": 5,
    }
    assert expected["mtf"]["per_tf_widths"] == {
        "M5": MULTI_TF_FEATURE_COUNT_V4,
        "M15": MULTI_TF_FEATURE_COUNT_V4,
        "H1": MULTI_TF_FEATURE_COUNT_V4,
        "H4": MULTI_TF_FEATURE_COUNT_V4,
        "D1": MULTI_TF_FEATURE_COUNT_V4,
    }
    assert expected["exit"]["sequence_bars"] == 480
    assert expected["exit"]["max_path_bars"] == 512


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("entry", "sequence_bars"), 95),
        (("shared_surface", "signal_dim"), 512),
        (("schemas", "mtf_cache"), "htf_v3_disk_cache_manifest_v3"),
        (("mtf", "cache_timeframes"), ["M5", "H1", "H4", "D1"]),
        (("entry", "mtf_route"), ["M5", "M15", "H1", "H4", "D1"]),
        (("exit", "sequence_bars"), MODEL_NATIVE_SELECTED_FEATURE_COUNT),
        (("exit", "max_path_bars"), 511),
        (("shared_encoder",), "duplicate_exit_encoder"),
        (("local_specialists",), list(reversed(MODEL_NATIVE_TRAINING_SPECIALISTS))),
        (("mtf_specialists",), list(reversed(MODEL_NATIVE_TRAINING_SPECIALISTS))),
    ],
)
def test_any_fixed_architecture_drift_fails_closed(
    path: tuple[str, ...],
    value: object,
) -> None:
    observed = entry_exit_production_architecture_contract()
    target = observed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_PRODUCTION_ARCHITECTURE_MISMATCH",
    ):
        require_entry_exit_production_architecture(
            observed,
            context="TEST_DRIFT",
        )


def _preallocation_model_kwargs() -> dict[str, object]:
    specialists = list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    routing = {name: [index] for index, name in enumerate(specialists)}
    return {
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": 96,
        "dropout": 0.0,
        "ctx_cont_dim": 142,
        "ctx_cat_dim": 5,
        "m5_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "m15_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "h1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "h4_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "d1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
        "m5_seq_len": 16,
        "m15_seq_len": 64,
        "h1_seq_len": 96,
        "h4_seq_len": 96,
        "d1_seq_len": 252,
        "multi_tf_num_layers": 1,
        "multi_tf_scale": 1.0,
        "specialist_input_indices": routing,
        "specialist_ctx_cont_indices": routing,
        "specialist_ctx_cont_nominal_indices": routing,
        "specialist_ctx_cat_indices": routing,
        "multi_tf_specialist_input_indices": routing,
        "temporal_alias_signal_indices": [],
        "temporal_alias_ctx_cont_indices": [],
        "input_normalization": {
            "schema_version": "entry_model_native_input_normalization_v2"
        },
        "specialist_num_layers": 1,
        "specialist_fusion_scale": 1.0,
        "cross_family_fusion_scale": 1.0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq_len", 95),
        ("m15_seq_dim", 110),
        (
            "input_normalization",
            {"schema_version": "entry_model_native_input_normalization_v1"},
        ),
    ],
)
def test_model_drift_fails_before_torch_module_allocation(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocation_called = False

    def _forbidden_allocation(_self: object) -> None:
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("torch allocation must not be reached")

    monkeypatch.setattr(torch.nn.Module, "__init__", _forbidden_allocation)
    kwargs = _preallocation_model_kwargs()
    kwargs[field] = value

    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_PRODUCTION_ARCHITECTURE_MISMATCH",
    ):
        EntryV10CtxHybridTransformer(**kwargs)

    assert allocation_called is False


def test_exact_model_architecture_reaches_allocation_only_after_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AllocationReached(RuntimeError):
        pass

    def _allocation_reached(_self: object) -> None:
        raise AllocationReached

    monkeypatch.setattr(torch.nn.Module, "__init__", _allocation_reached)

    with pytest.raises(AllocationReached):
        EntryV10CtxHybridTransformer(**_preallocation_model_kwargs())


def test_private_small_model_path_is_not_a_production_input() -> None:
    parameters = inspect.signature(EntryV10CtxHybridTransformer).parameters

    assert "unit_test" not in parameters
    assert "architecture_bypass" not in parameters
    assert "compatibility" not in parameters


def test_wrong_dataset_tf_route_fails_before_manifest_or_parquet_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_read = False
    parquet_read = False

    def _manifest_read(_path: Path) -> dict[str, object]:
        nonlocal manifest_read
        manifest_read = True
        raise AssertionError("manifest read must not be reached")

    def _parquet_read(*_args: object, **_kwargs: object) -> object:
        nonlocal parquet_read
        parquet_read = True
        raise AssertionError("parquet read must not be reached")

    monkeypatch.setattr(trainer, "_signal_contract_from_manifest_path", _manifest_read)
    monkeypatch.setattr(trainer.pd, "read_parquet", _parquet_read)

    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_PRODUCTION_ARCHITECTURE_MISMATCH",
    ):
        trainer.EntryV10CtxDataset(
            Path("/never/read.parquet"),
            seq_len=96,
            m5_prebuilt_path=Path("/never/read-m5.parquet"),
            per_tf_seq_lens={
                "M15": 16,
                "M5": 24,
                "H1": 12,
                "H4": 8,
                "D1": 4,
            },
            multi_tf_closed_bar=True,
        )

    assert manifest_read is False
    assert parquet_read is False


def _minimal_bundle_architecture_payload() -> dict[str, object]:
    specialists = list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    specialist_indices = {name: [index] for index, name in enumerate(specialists)}
    return {
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": 96,
        "ctx_cont_dim": 142,
        "ctx_cat_dim": 5,
        "model_native_signal_contract": {
            "schema_version": entry_exit_production_architecture_contract()[
                "schemas"
            ]["signal"]
        },
        "input_normalization": {
            "schema_version": "entry_model_native_input_normalization_v2"
        },
        "specialist_fusion": {"input_indices": specialist_indices},
        "context_specialist_routing": {"ctx_cont_indices": specialist_indices},
        "multi_tf": {
            "v4_mode": True,
            "matrix_contract": "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V2",
            "entry_route_timeframes": ["M15", "H1", "H4", "D1"],
            "exit_route_timeframes": ["M5", "M15", "H1", "H4", "D1"],
            "m5_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
            "m15_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
            "h1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
            "h4_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
            "d1_seq_dim": MULTI_TF_FEATURE_COUNT_V4,
            "m5_seq_len": 16,
            "m15_seq_len": 64,
            "h1_seq_len": 96,
            "h4_seq_len": 96,
            "d1_seq_len": 252,
            "specialist_input_indices": specialist_indices,
        },
        "unified_entry_exit_contract": {
            "entry_local_timeframe": "M5",
            "exit_local_timeframe": "M1",
            "exit_local_sequence_bars": 480,
            "exit_max_path_bars": 512,
            "shared_feature_base_contract": {
                "shared_encoder": (
                    "entry_v10_ctx_hybrid_transformer_shared_specialists_v2"
                ),
                "entry": {"decision_timeframe": "M5"},
            },
        },
    }


def _write_minimal_bundle_files(
    bundle_dir: Path,
    payload: dict[str, object],
) -> None:
    bundle_dir.mkdir()
    encoded = json.dumps(payload)
    (bundle_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(encoded)
    (bundle_dir / "bundle_metadata.json").write_text(encoded)
    (bundle_dir / "model_state_dict.pt").write_bytes(b"unread")


@pytest.mark.parametrize(
    ("mutator",),
    [
        (lambda payload: payload["multi_tf"].__setitem__("m15_seq_dim", 110),),
        (
            lambda payload: payload["input_normalization"].__setitem__(
                "schema_version",
                "entry_model_native_input_normalization_v1",
            ),
        ),
    ],
)
def test_bundle_drift_fails_before_commit_or_state_bytes(
    mutator: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = copy.deepcopy(_minimal_bundle_architecture_payload())
    mutator(payload)
    bundle_dir = tmp_path / "bundle"
    _write_minimal_bundle_files(bundle_dir, payload)
    commit_called = False
    state_read = False
    real_read_bytes = Path.read_bytes

    def _commit(_path: Path) -> dict[str, object]:
        nonlocal commit_called
        commit_called = True
        raise AssertionError("commit validation must not be reached")

    def _read_bytes(path: Path) -> bytes:
        nonlocal state_read
        if path.name == "model_state_dict.pt":
            state_read = True
            raise AssertionError("model state must not be read")
        return real_read_bytes(path)

    monkeypatch.setattr(bundle_module, "require_bundle_commit_manifest", _commit)
    monkeypatch.setattr(Path, "read_bytes", _read_bytes)

    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_PRODUCTION_ARCHITECTURE_MISMATCH",
    ):
        bundle_module.load_entry_v10_ctx_bundle(
            bundle_dir=bundle_dir,
            device="cpu",
        )

    assert commit_called is False
    assert state_read is False


def test_exact_bundle_architecture_reaches_commit_before_state_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CommitReached(RuntimeError):
        pass

    bundle_dir = tmp_path / "bundle"
    _write_minimal_bundle_files(
        bundle_dir,
        _minimal_bundle_architecture_payload(),
    )
    state_read = False
    real_read_bytes = Path.read_bytes

    def _commit(_path: Path) -> dict[str, object]:
        raise CommitReached

    def _read_bytes(path: Path) -> bytes:
        nonlocal state_read
        if path.name == "model_state_dict.pt":
            state_read = True
        return real_read_bytes(path)

    monkeypatch.setattr(bundle_module, "require_bundle_commit_manifest", _commit)
    monkeypatch.setattr(Path, "read_bytes", _read_bytes)

    with pytest.raises(CommitReached):
        bundle_module.load_entry_v10_ctx_bundle(
            bundle_dir=bundle_dir,
            device="cpu",
        )

    assert state_read is False
