from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    canonical_sha256,
    file_sha256,
)
from gx1.contracts.unified_exit_trainer_wiring_v2 import (
    bind_canonical_unified_exit_lifecycle_v2,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    COMPACT_ROOT_SCHEMA_VERSION,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _readiness(path: Path) -> None:
    hurdle = economics.seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": economics.CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": "1" * 64,
            "train_fold_sha256": "2" * 64,
            "source_lineage_sha256": "3" * 64,
            "annual_continuous_hurdle_rate": 0.05,
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": economics.SECONDS_PER_YEAR,
            "fit_method": "synthetic_train_only",
            "fit_evidence_sha256": "4" * 64,
        }
    )
    objective = economics.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256="1" * 64,
        expected_train_fold_sha256="2" * 64,
        expected_source_lineage_sha256="3" * 64,
        policy_sha256="5" * 64,
    )
    _write(
        path,
        {
            "schema_version": "gx1_unified_exit_training_economics_readiness_v3",
            "mode": "economics_objective_v2",
            "capital_hurdle_artifact": hurdle,
            "economics_objective_contract": objective,
            "expected_train_split_sha256": "1" * 64,
            "expected_train_fold_sha256": "2" * 64,
            "expected_source_lineage_sha256": "3" * 64,
            "policy_sha256": "5" * 64,
            "proper_policy_certificate_sha256": None,
            "test_data_used": False,
        },
    )


def _artifacts(tmp_path: Path):
    manifests, compacts = {}, {}
    bindings = {}
    for split in ("train", "val"):
        compact = (tmp_path / f"{split}.parquet").resolve()
        pd.DataFrame({"entry_row_index": [0]}).to_parquet(compact, index=False)
        compact_sha = file_sha256(compact)
        manifest = {
            "schema_version": "synthetic",
            "dataset_run_id": "DATASET_RUN",
            "split": split,
            "test_accessed": False,
            "target_q_stored": False,
            "compact_rows": 1,
            "compact_parquet_sha256": compact_sha,
            "m1_source_sha256": "6" * 64,
            "entry_binding_sha256": "7" * 64,
            "gap_classification_source_sha256": "8" * 64,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        manifest_path = (tmp_path / f"{split}.manifest.json").resolve()
        _write(manifest_path, manifest)
        manifests[split], compacts[split] = manifest_path, compact
        bindings[split] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "compact_parquet_sha256": compact_sha,
            "compact_rows": 1,
        }
    root = {
        "schema_version": COMPACT_ROOT_SCHEMA_VERSION,
        "decision": "PASS",
        "dataset_run_id": "DATASET_RUN",
        "allowed_splits": ["train", "val"],
        "test_accessed": False,
        "target_q_stored": False,
        "chunk_state_capacity": 512,
        "split_manifests": bindings,
    }
    root["manifest_sha256"] = canonical_sha256(root)
    root_path = (tmp_path / "root.json").resolve()
    _write(root_path, root)
    readiness = (tmp_path / "readiness.json").resolve()
    _readiness(readiness)
    policies = {}
    for split in ("train", "val"):
        policies[split] = (tmp_path / f"{split}.cost.json").resolve()
        policies[split].write_text("{}\n")
    return root_path, compacts, manifests, readiness, policies


class _Dataset:
    def __init__(self) -> None:
        self._unified_exit_lifecycle = None
        self._unified_exit_lifecycle_v2 = None
        self.per_tf_seq_lens = {"M5": 1, "M15": 1, "H1": 1, "H4": 1, "D1": 1}
        self._multi_tf_cache_identity_sha256 = "9" * 64

    def _get_exit_multi_tf_episode_histories(self, _times):
        return {}

    def bind_unified_exit_lifecycle_v2(self, adapter) -> None:
        self._unified_exit_lifecycle_v2 = adapter


def test_factory_binds_v2_before_loader_and_rejects_legacy(
    monkeypatch, tmp_path: Path
) -> None:
    root, compacts, manifests, readiness, policies = _artifacts(tmp_path)

    class Provider:
        def __init__(self, **kwargs) -> None:
            split = kwargs["compact_manifest"]["split"]
            self.economic_exit_step_manifest = {
                "split": split,
                "manifest_sha256": "a" * 64,
            }

    class Adapter:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    import gx1.contracts.unified_exit_trainer_wiring_v2 as wiring

    monkeypatch.setattr(wiring, "LazyUnifiedExitEconomicStepProviderV1", Provider)
    monkeypatch.setattr(wiring, "UnifiedExitDatasetAdapterV2", Adapter)
    datasets = {"train": _Dataset(), "val": _Dataset()}
    result = bind_canonical_unified_exit_lifecycle_v2(
        dataset_run_id="DATASET_RUN",
        root_manifest_path=root,
        compact_paths=compacts,
        compact_manifest_paths=manifests,
        economics_readiness_path=readiness,
        cost_authority_paths=policies,
        datasets=datasets,
        feature_source_owners={"train": object(), "val": object()},
    )
    assert result["v2_active_before_loader"] is True
    assert result["legacy_exit_target_rejected"] is True
    assert result["splits"]["val"]["validation_dispatch"] == (
        "deterministic_full_both_sides_all_chunks"
    )
    assert all(ds._unified_exit_lifecycle_v2 is not None for ds in datasets.values())
    datasets["train"]._unified_exit_lifecycle = object()
    with pytest.raises(RuntimeError, match="LEGACY_OR_DUPLICATE"):
        bind_canonical_unified_exit_lifecycle_v2(
            dataset_run_id="DATASET_RUN",
            root_manifest_path=root,
            compact_paths=compacts,
            compact_manifest_paths=manifests,
            economics_readiness_path=readiness,
            cost_authority_paths=policies,
            datasets=datasets,
            feature_source_owners={"train": object(), "val": object()},
        )


def test_canonical_trainer_calls_v2_factory_and_epoch_setter_before_dataloaders() -> (
    None
):
    source = Path("gx1/models/entry_v10/entry_v10_ctx_train_v3.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "bind_canonical_unified_exit_lifecycle_v2"
        for call in calls
    )
    run_train = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_train"
    )
    run_calls = [node for node in ast.walk(run_train) if isinstance(node, ast.Call)]
    bind_line = next(
        call.lineno
        for call in run_calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "bind_canonical_unified_exit_lifecycle_v2"
    )
    loader_lines = [
        call.lineno
        for call in run_calls
        if isinstance(call.func, ast.Name) and call.func.id == "DataLoader"
    ]
    assert loader_lines and bind_line < min(loader_lines)
    assert "train_ds.set_unified_exit_lifecycle_v2_epoch(int(epoch))" in source
    assert '"--unified-exit-v2-preflight-only"' in source
