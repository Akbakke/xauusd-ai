"""One fixed-year selection keeps full Entry/Exit IDs and all source artifacts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import unified_exit_random_access_index_v1 as contract
from gx1.scripts import materialize_unified_exit_random_access_index_v1 as builder


def _binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _sealed(path, value, seal):
    value = {k: v for k, v in value.items() if k != seal}
    value[seal] = contract.canonical_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value


def _hash_rows(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()


def _fixture(tmp_path, *, omit_split=None):
    data = tmp_path / "full_data"
    data.mkdir()
    clocks = {
        "train": ["2021-06-01", "2025-05-31", "2025-06-01", "2025-09-01", "2026-05-31"],
        "val": ["2026-05-31 23:55", "2026-06-01", "2026-06-15", "2026-06-30 23:45", "2026-07-01"],
    }
    manifests, frames = {}, {}
    for split, times in clocks.items():
        parent = pd.DatetimeIndex(pd.to_datetime(times, utc=True, format="mixed")).as_unit("ns")
        rows = np.arange(len(parent)) if split == "train" else np.array([1, 2, 3])
        if split == omit_split:
            rows = rows[rows != (3 if split == "train" else 2)]
        child = parent[rows]
        parent_path = data / f"parent_{split}.parquet"
        pd.DataFrame({"time": parent, "feature_sentinel": np.arange(len(parent))}).to_parquet(parent_path, index=False)
        parent_manifest_path = data / f"parent_{split}.manifest.json"
        parent_manifest_path.write_text(json.dumps({
            "output_data_path": str(parent_path), "extra": {"pretest_test_guard": {"test_accessed": False}},
            "splits": {"train": {"start": "2021-06-01T00:00:00Z", "end": "2026-05-31T23:55:00Z"}},
        }))
        child_path = data / f"{split}.parquet"
        pd.read_parquet(parent_path).iloc[rows].to_parquet(child_path, index=False)
        child_manifest_path = data / f"{split}.child.manifest.json"
        child_manifest_path.write_text("{}")
        sources = {
            "entry_parquet": _binding(child_path), "entry_manifest": _binding(child_manifest_path),
            "parent_entry_parquet": _binding(parent_path), "parent_entry_manifest": _binding(parent_manifest_path),
        }
        m1 = pd.DatetimeIndex(sorted(t + pd.Timedelta(minutes=m) for t in child for m in (5, 6)))
        frame, _ = contract.build_random_access_index_v2(
            split=split, entry_times=child, parent_entry_times=parent,
            child_m1_times=m1, parent_m1_times=m1, successor_transition_counts=[1] * len(child),
            entry_bid=[100.0] * len(child), entry_ask=[100.1] * len(child),
            episode_binding_sha256_by_entry=["b" * 64] * len(child), entry_fill_binding_sha256_by_entry=["c" * 64] * len(child),
        )
        path = data / f"{split}.random_access_index.parquet"
        frame.to_parquet(path, index=False)
        manifest = _sealed(data / f"{split}.manifest.json", {
            "schema_version": contract.RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION, "decision": "PASS", "split": split,
            "entry_row_count": len(child), "successor_transition_total": len(child), "economic_terminal_count": 0,
            "split_end_is_right_censor": True, "split_end_utc": contract.LATEST_YEAR_WINDOWS[split]["end_utc_exclusive"],
            "storage_granularity": "one_row_per_entry", "full_prefix_states_stored": False,
            "chunk_pointers_stored": False, "target_q_stored": False,
            "index_stream_sha256": contract.index_stream_sha256(frame), "source_bindings": sources,
            "index_parquet_path": str(path), "index_parquet_sha256": _binding(path)["sha256"],
            "child_entry_clock_sha256": _hash_rows(child.asi8), "parent_entry_clock_sha256": _hash_rows(parent.asi8),
            "parent_entry_row_indices_sha256": _hash_rows(rows), "parent_entry_mapping_sha256": contract.parent_entry_mapping_sha256(frame),
            "parent_entry_source_sha256": sources["parent_entry_parquet"]["sha256"],
            "parent_entry_manifest_sha256": sources["parent_entry_manifest"]["sha256"],
            "parent_entry_source_rows": len(parent), "test_accessed": False,
        }, "manifest_sha256")
        manifests[split], frames[split] = manifest, frame
    root = {
        "schema_version": contract.FULL_POPULATION_ROOT_SCHEMA_VERSION, "decision": "PASS", "allowed_splits": ["train", "val"],
        "storage_granularity": "one_row_per_entry", "full_prefix_states_stored": False, "chunk_pointers_stored": False,
        "composite_normalization_sha256": "a" * 64, "test_accessed": False,
        "full_train_population": {"entry_row_count": len(frames["train"]), "parent_entry_source_rows": len(frames["train"]), "train_manifest_sha256": manifests["train"]["manifest_sha256"]},
        "splits": {split: {key: manifest[key] for key in ("index_parquet_path", "index_parquet_sha256", "manifest_sha256", "entry_row_count", "successor_transition_total")}
                   | {"manifest_path": str(data / f"{split}.manifest.json")} for split, manifest in manifests.items()},
    }
    root_path = data / "ROOT.json"
    root = _sealed(root_path, root, "root_sha256")
    return {"source_root_binding": _binding(root_path), "source": root, "frames": frames, "manifests": manifests}


def _root(fixture):
    population = contract.build_latest_year_population(source_root_binding=fixture["source_root_binding"])
    value = {k: v for k, v in fixture["source"].items() if k not in {"root_sha256", "full_train_population"}}
    value.update(schema_version=contract.LATEST_YEAR_ROOT_SCHEMA_VERSION, latest_year_population=population)
    value["root_sha256"] = contract.canonical_sha256(value)
    return value


def test_latest_year_selects_original_full_ids_and_inherits_full_june(tmp_path):
    fixture = _fixture(tmp_path)
    root = _root(fixture)
    expected = {split: {"parquet": manifest["source_bindings"]["parent_entry_parquet"], "manifest": manifest["source_bindings"]["parent_entry_manifest"]}
                for split, manifest in fixture["manifests"].items()}
    assert contract.require_latest_year_index_root(root, expected_parent_bindings=expected) == root
    assert contract.latest_year_selected_entry_rows(fixture["frames"]["train"], split="train").tolist() == [2, 3, 4]
    assert root["splits"] == fixture["source"]["splits"]
    train = root["latest_year_population"]["splits"]["train"]
    assert (train["selected_entry_row_count"], train["index_entry_row_count"]) == (3, 5)
    assert train["selected_parent_entry_row_indices_sha256"] == _hash_rows([2, 3, 4])
    assert root["latest_year_population"]["splits"]["val"]["selected_entry_row_count"] == 3


@pytest.mark.parametrize("split", ["train", "val"])
def test_same_endpoints_and_self_consistent_hashes_cannot_hide_missing_interior_row(tmp_path, split):
    fixture = _fixture(tmp_path, omit_split=split)
    with pytest.raises(RuntimeError, match="COMPLETE_SELECTION_INVALID|FULL_PARENT_REQUIRED"):
        contract.build_latest_year_population(source_root_binding=fixture["source_root_binding"])


def test_latest_year_rejects_another_launch_parent_even_with_same_coordinates(tmp_path):
    root = _root(_fixture(tmp_path))
    expected = {split: {"parquet": deepcopy(proof["parent_entry_parquet"]), "manifest": proof["parent_entry_manifest"]}
                for split, proof in root["latest_year_population"]["splits"].items()}
    expected["train"]["parquet"]["sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="LAUNCH_PARENT_MISMATCH"):
        contract.require_latest_year_index_root(root, expected_parent_bindings=expected)


def test_latest_year_rejects_changed_source_root_bytes(tmp_path):
    root = _root(_fixture(tmp_path))
    path = Path(root["latest_year_population"]["source_root"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="BINDING_INVALID"):
        contract.require_latest_year_index_root(root)


@pytest.mark.parametrize("change", ["window", "mixed_origin", "wrong_schema", "normalization", "selected_ids"])
def test_latest_year_rejects_other_scope_or_data_even_after_rehash(tmp_path, change):
    root = _root(_fixture(tmp_path))
    if change == "window":
        root["latest_year_population"]["splits"]["train"]["start_utc"] = "2025-06-02T00:00:00+00:00"
    elif change == "mixed_origin":
        root["full_train_population"] = {}
    elif change == "wrong_schema":
        root["schema_version"] = contract.RANDOM_ACCESS_INDEX_ROOT_SCHEMA_VERSION
    elif change == "normalization":
        root["composite_normalization_sha256"] = "f" * 64
    else:
        root["latest_year_population"]["splits"]["train"]["selected_child_entry_row_indices_sha256"] = "f" * 64
    root["root_sha256"] = contract.canonical_sha256({k: v for k, v in root.items() if k != "root_sha256"})
    with pytest.raises(RuntimeError, match="LATEST_YEAR"):
        contract.require_latest_year_index_root(root)


@pytest.mark.parametrize("options", [{}, {"full_train_population": True, "latest_year_population": True}, {"latest_year_population": True, "predecessor_root_path": Path("/unused")}, {"latest_year_population": 1}])
def test_publisher_requires_exactly_one_population_mode(tmp_path, options):
    with pytest.raises(RuntimeError, match="POPULATION_OR_PREDECESSOR_REQUIRED"):
        builder.publish(pilot_root=tmp_path, output_dir=tmp_path / "unused", **options)


def test_publisher_latest_year_writes_only_root_without_materializing_data(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    original = {str(path): _binding(path)["sha256"] for path in (tmp_path / "full_data").iterdir()}
    def forbidden(**kwargs):
        raise AssertionError("No split may be materialized for selection-only publication")
    monkeypatch.setattr(builder, "_build_split", forbidden)
    output = tmp_path / "published"
    root = builder.publish(pilot_root=tmp_path, output_dir=output, latest_year_population=True,
                           population_source_root_path=Path(fixture["source_root_binding"]["path"]))
    assert sorted(path.name for path in output.iterdir()) == ["ROOT.json"]
    assert root["splits"] == fixture["source"]["splits"]
    assert {str(path): _binding(path)["sha256"] for path in (tmp_path / "full_data").iterdir()} == original
    assert contract.require_latest_year_index_root(root) == root
