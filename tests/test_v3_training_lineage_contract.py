from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.exits.training.thin_record_dataset import (
    V3_TRAINING_DATASET_PRODUCER_CONTRACT,
    V3_TRAINING_LINEAGE_SCHEMA_VERSION,
    V3_TRAINING_SOURCE_CODE_FILES,
    build_v3_xgb_bridge_source_identity,
    require_authoritative_v3_training_dataset,
    require_reproducible_v3_training_lineage,
    v3_regular_file_binding,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_dataset(root: Path, **semantic_updates: object) -> Path:
    root.mkdir(parents=True)
    xgb_bundle = root / "xgb_bundle"
    xgb_bundle.mkdir()
    (xgb_bundle / "model.joblib").write_bytes(b"unit-xgb")
    xgb_feature_contract = xgb_bundle / "xgb_input_features.json"
    xgb_feature_contract.write_text(
        json.dumps({"features": ["feature_a", "feature_b"]}),
        encoding="utf-8",
    )
    xgb_sanitizer = xgb_bundle / "xgb_input_sanitizer.json"
    xgb_sanitizer.write_text(
        json.dumps({"feature_list": ["feature_a", "feature_b"]}),
        encoding="utf-8",
    )
    xgb_identity = build_v3_xgb_bridge_source_identity(
        bundle_dir=xgb_bundle.resolve(),
        feature_contract_path=xgb_feature_contract.resolve(),
        sanitizer_config_path=xgb_sanitizer.resolve(),
    )
    files = {
        "m1_feature_matrix": "m1_feature_matrix.npy",
        "m1_time_ns": "m1_time_ns.npy",
        "trade_state_overlays": "trade_state_overlays.f32",
        "trade_state_overlays_cols": 19,
        "overlay_index": "overlay_index.parquet",
        "records": "records.jsonl",
    }
    for value in files.values():
        if isinstance(value, str):
            (root / value).write_bytes(f"unit:{value}".encode("utf-8"))
    manifest = {
        "producer_contract_v1": V3_TRAINING_DATASET_PRODUCER_CONTRACT,
        "production_allowed_v1": True,
        "model_native_entry_snapshot_v1": True,
        "exact_t5_fill_v1": True,
        "frozen_entry_snapshot_complete_v1": True,
        "canonical_m1_base_mtf_state_complete_v1": True,
        "xgb_bridge_source_v1": xgb_identity,
        "input_dim": 173,
        "window_len": 512,
        "files": files,
    }
    manifest.update(semantic_updates)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _write_bundle(tmp_path: Path) -> tuple[Path, dict, Path]:
    dataset = _write_dataset(tmp_path / "dataset")
    _, dataset_inventory = require_authoritative_v3_training_dataset(dataset)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    state_path = bundle / "exit_transformer_v0.pt"
    state_path.write_bytes(b"unit-state")
    config = {
        "exit_ml_io_version": "EXIT_IO_V8_REGIME_M1L512",
        "input_dim": 173,
        "window_len": 512,
    }
    config_path = bundle / "transformer_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    m5 = tmp_path / "canonical_m5.parquet"
    m5.write_bytes(b"unit-m5")
    source_inventory = []
    for relative in sorted(V3_TRAINING_SOURCE_CODE_FILES):
        source = bundle / "training_source_v1" / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"unit-source:{relative}".encode("utf-8"))
        source_inventory.append(
            {
                "relative_path": relative,
                **v3_regular_file_binding(
                    source.resolve(),
                    context=f"UNIT_SOURCE[{relative}]",
                ),
            }
        )
    training = {
        "seed": 1337,
        "train_cutoff": "2025-07-01T00:00:00+00:00",
        "val_cutoff": "2026-01-01T00:00:00+00:00",
    }
    lineage = {
        "schema_version": V3_TRAINING_LINEAGE_SCHEMA_VERSION,
        "production_allowed_v1": True,
        "dataset_producer_contract_v1": (
            V3_TRAINING_DATASET_PRODUCER_CONTRACT
        ),
        "dataset_root": str(dataset.resolve()),
        "dataset_files": dataset_inventory,
        "dataset_inventory_sha256": _canonical_sha256(dataset_inventory),
        "m5_prebuilt": v3_regular_file_binding(
            m5.resolve(),
            context="UNIT_M5",
        ),
        "xgb_bridge_source": json.loads(
            json.dumps(
                json.loads(
                    (dataset / "manifest.json").read_text(encoding="utf-8")
                )["xgb_bridge_source_v1"]
            )
        ),
        "source_code_files": source_inventory,
        "source_code_inventory_sha256": _canonical_sha256(source_inventory),
        "split_uid_sha256": {
            "train": hashlib.sha256(b"train").hexdigest(),
            "val": hashlib.sha256(b"val").hexdigest(),
            "test": hashlib.sha256(b"test").hexdigest(),
        },
        "training_recipe_sha256": _canonical_sha256(training),
        "transformer_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "initialization": {
            "mode": "cold",
            "source_state_dict": None,
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "exit_io_version": config["exit_ml_io_version"],
                "model_state_dict_sha256": hashlib.sha256(
                    state_path.read_bytes()
                ).hexdigest(),
                "training": training,
                "training_lineage_v1": lineage,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bundle, config, state_path


def test_reproducible_v3_training_lineage_accepts_exact_self_contained_bytes(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)

    lineage = require_reproducible_v3_training_lineage(
        bundle_dir=bundle.resolve(),
        config=config,
        state_path=state_path.resolve(),
    )

    assert lineage["production_allowed_v1"] is True
    assert lineage["initialization"]["mode"] == "cold"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"production_allowed_v1": 1}, "AUTHORITY_MISSING"),
        ({"exact_t5_fill_v1": False}, "AUTHORITY_MISSING"),
        ({"frozen_entry_snapshot_complete_v1": False}, "AUTHORITY_MISSING"),
        (
            {"canonical_m1_base_mtf_state_complete_v1": False},
            "AUTHORITY_MISSING",
        ),
        ({"model_native_entry_snapshot_v1": False}, "AUTHORITY_MISSING"),
    ],
)
def test_v3_dataset_semantic_authority_is_exact(
    tmp_path: Path,
    updates: dict,
    message: str,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset", **updates)

    with pytest.raises(RuntimeError, match=message):
        require_authoritative_v3_training_dataset(dataset.resolve())


def test_legacy_v3_manifest_without_lineage_fails_closed(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["training_lineage_v1"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="LINEAGE_MISSING_OR_NONCANONICAL"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )


def test_v3_lineage_rehashes_dataset_and_bundle_owned_source_bytes(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)
    dataset_file = tmp_path / "dataset" / "records.jsonl"
    dataset_file.write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="DATASET_INVENTORY_MISMATCH"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )

    bundle, config, state_path = _write_bundle(tmp_path / "second")
    source_file = (
        bundle
        / "training_source_v1"
        / "gx1/policy/exit_transformer_v0.py"
    )
    source_file.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="file bytes differ"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )


def test_v3_lineage_rehashes_xgb_bridge_source_bytes(
    tmp_path: Path,
) -> None:
    bundle, config, state_path = _write_bundle(tmp_path)
    xgb_bundle = tmp_path / "dataset" / "xgb_bundle"
    external_contract = tmp_path / "external_xgb_input_features.json"
    external_contract.write_bytes(
        (xgb_bundle / "xgb_input_features.json").read_bytes()
    )
    with pytest.raises(RuntimeError, match="CONTRACTS_NOT_BUNDLE_OWNED"):
        build_v3_xgb_bridge_source_identity(
            bundle_dir=xgb_bundle.resolve(),
            feature_contract_path=external_contract.resolve(),
            sanitizer_config_path=(
                xgb_bundle / "xgb_input_sanitizer.json"
            ).resolve(),
        )

    (xgb_bundle / "model.joblib").write_bytes(
        b"mutated-xgb"
    )

    with pytest.raises(RuntimeError, match="XGB_BRIDGE_SOURCE"):
        require_reproducible_v3_training_lineage(
            bundle_dir=bundle.resolve(),
            config=config,
            state_path=state_path.resolve(),
        )


def test_v3_parallel_label_spill_preserves_input_uid_order() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gx1/exits/training/disk_labeled_dataset.py"
    ).read_text(encoding="utf-8")

    assert "pool.imap_unordered" not in source
    assert "pool.imap(_spawn_label_one_trade" in source
