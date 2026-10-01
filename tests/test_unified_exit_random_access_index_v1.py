from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_random_access_index_v1 import (
    RANDOM_ACCESS_INDEX_SCHEMA_VERSION,
    build_random_access_index,
    build_random_access_index_v2,
    canonical_sha256,
    index_stream_sha256,
    parent_entry_mapping_sha256,
    require_random_access_index,
    require_random_access_index_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _built():
    parent = pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC")
    child = parent[5:25]
    entry = pd.DatetimeIndex(
        [child[5] - pd.Timedelta(minutes=5), child[9] - pd.Timedelta(minutes=5)]
    )
    frame, offset = build_random_access_index(
        split="train",
        entry_times=entry,
        child_m1_times=child,
        parent_m1_times=parent,
        successor_transition_counts=np.array([3, 4], dtype="<i8"),
        entry_bid=[100.0, 101.0],
        entry_ask=[100.1, 101.1],
        episode_binding_sha256_by_entry=[SHA_A, SHA_B],
        entry_fill_binding_sha256_by_entry=[SHA_B, SHA_A],
    )
    return frame, offset


def test_index_is_one_entry_row_with_exact_parent_child_coordinates():
    frame, offset = _built()
    assert offset == 5
    assert frame["child_m1_start_row"].tolist() == [5, 9]
    assert frame["parent_m1_start_row"].tolist() == [10, 14]
    assert frame["successor_transition_count"].tolist() == [3, 4]
    assert frame["lifecycle_state_count"].tolist() == [4, 5]
    assert frame["right_censored"].tolist() == [True, True]
    assert frame["economic_terminal"].tolist() == [False, False]
    assert len(set(frame["row_identity_sha256"])) == 2


def test_v2_index_maps_child_entry_to_exact_parent_entry_row():
    parent_m1 = pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC")
    child_m1 = parent_m1[5:25]
    entry = pd.DatetimeIndex(
        [child_m1[5] - pd.Timedelta(minutes=5), child_m1[9] - pd.Timedelta(minutes=5)]
    )
    parent_entry = pd.DatetimeIndex(
        [parent_m1[0], parent_m1[2], entry[0], parent_m1[7], entry[1], parent_m1[12]]
    )
    frame, offset = build_random_access_index_v2(
        split="train",
        entry_times=entry,
        parent_entry_times=parent_entry,
        child_m1_times=child_m1,
        parent_m1_times=parent_m1,
        successor_transition_counts=[3, 4],
        entry_bid=[100.0, 101.0],
        entry_ask=[100.1, 101.1],
        episode_binding_sha256_by_entry=[SHA_A, SHA_B],
        entry_fill_binding_sha256_by_entry=[SHA_B, SHA_A],
    )
    assert offset == 5
    assert frame["parent_entry_row_index"].tolist() == [2, 4]
    assert require_random_access_index(frame) is frame
    assert len(parent_entry_mapping_sha256(frame)) == 64


def test_v2_index_rejects_missing_parent_entry_clock_match():
    parent_m1 = pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC")
    child_m1 = parent_m1[5:25]
    entry = [child_m1[5] - pd.Timedelta(minutes=5)]
    with pytest.raises(RuntimeError, match="PARENT_ENTRY_INVALID"):
        build_random_access_index_v2(
            split="train",
            entry_times=entry,
            parent_entry_times=[parent_m1[1]],
            child_m1_times=child_m1,
            parent_m1_times=parent_m1,
            successor_transition_counts=[3],
            entry_bid=[100.0],
            entry_ask=[100.1],
            episode_binding_sha256_by_entry=[SHA_A],
            entry_fill_binding_sha256_by_entry=[SHA_B],
        )


def test_index_rejects_out_of_range_successor_without_duration_cap():
    parent = pd.date_range("2026-01-01", periods=20, freq="min", tz="UTC")
    child = parent[5:15]
    with pytest.raises(RuntimeError, match="RANGE_INVALID"):
        build_random_access_index(
            split="val",
            entry_times=[child[8] - pd.Timedelta(minutes=5)],
            child_m1_times=child,
            parent_m1_times=parent,
            successor_transition_counts=[2],
            entry_bid=[100.0],
            entry_ask=[100.1],
            episode_binding_sha256_by_entry=[SHA_A],
            entry_fill_binding_sha256_by_entry=[SHA_B],
        )


def test_manifest_seals_no_chunk_or_prefix_contract():
    frame, _ = _built()
    source = {"x": {"path": "/immutable/x", "sha256": SHA_A}}
    value = {
        "schema_version": RANDOM_ACCESS_INDEX_SCHEMA_VERSION,
        "decision": "PASS",
        "split": "train",
        "entry_row_count": 2,
        "successor_transition_total": 7,
        "economic_terminal_count": 0,
        "split_end_is_right_censor": True,
        "storage_granularity": "one_row_per_entry",
        "full_prefix_states_stored": False,
        "chunk_pointers_stored": False,
        "target_q_stored": False,
        "index_stream_sha256": index_stream_sha256(frame),
        "source_bindings": source,
        "test_accessed": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    assert (
        require_random_access_index_manifest(
            value, expected_split="train", index_frame=frame
        )["decision"]
        == "PASS"
    )
    value["chunk_pointers_stored"] = True
    with pytest.raises(RuntimeError, match="MANIFEST_INVALID"):
        require_random_access_index_manifest(
            value, expected_split="train", index_frame=frame
        )


def test_manifest_rejects_staging_path_after_publication(tmp_path):
    frame, _ = _built()
    actual = tmp_path / "train.random_access_index.parquet"
    frame.to_parquet(actual, index=False)
    import hashlib

    source_file = tmp_path / "source.json"
    source_file.write_text("{}", encoding="utf-8")
    source = {
        "x": {
            "path": str(source_file),
            "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        }
    }
    value = {
        "schema_version": RANDOM_ACCESS_INDEX_SCHEMA_VERSION,
        "decision": "PASS",
        "split": "train",
        "entry_row_count": 2,
        "successor_transition_total": 7,
        "economic_terminal_count": 0,
        "split_end_is_right_censor": True,
        "storage_granularity": "one_row_per_entry",
        "full_prefix_states_stored": False,
        "chunk_pointers_stored": False,
        "target_q_stored": False,
        "index_parquet_path": str(tmp_path / ".deleted-stage" / actual.name),
        "index_parquet_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "index_stream_sha256": index_stream_sha256(frame),
        "source_bindings": source,
        "test_accessed": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    with pytest.raises(RuntimeError, match="FILE_INVALID"):
        require_random_access_index_manifest(
            value,
            expected_split="train",
            index_frame=frame,
            index_path=actual,
            verify_sources=True,
        )


def test_native_adapter_constructor_does_not_require_compact_schema():
    from gx1.contracts import unified_exit_economics_objective_v2 as economics
    from gx1.contracts.unified_exit_dataset_adapter_v2 import (
        ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
        UnifiedExitDatasetAdapterV2,
        seal_economic_exit_step_manifest,
    )

    frame, _ = _built()
    manifest = {
        "schema_version": RANDOM_ACCESS_INDEX_SCHEMA_VERSION,
        "decision": "PASS",
        "split": "train",
        "entry_row_count": 2,
        "successor_transition_total": 7,
        "economic_terminal_count": 0,
        "split_end_is_right_censor": True,
        "storage_granularity": "one_row_per_entry",
        "full_prefix_states_stored": False,
        "chunk_pointers_stored": False,
        "target_q_stored": False,
        "index_stream_sha256": index_stream_sha256(frame),
        "source_bindings": {"x": {"path": "/immutable/x", "sha256": SHA_A}},
        "test_accessed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
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
            "fit_method": "unit_train_only",
            "fit_evidence_sha256": "5" * 64,
        }
    )
    objective = economics.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256="1" * 64,
        expected_train_fold_sha256="2" * 64,
        expected_source_lineage_sha256="3" * 64,
        policy_sha256="4" * 64,
    )
    readiness = {
        "schema_version": "gx1_unified_exit_training_economics_readiness_v3",
        "mode": "economics_objective_v2",
        "capital_hurdle_artifact": hurdle,
        "economics_objective_contract": objective,
        "expected_train_split_sha256": "1" * 64,
        "expected_train_fold_sha256": "2" * 64,
        "expected_source_lineage_sha256": "3" * 64,
        "policy_sha256": "4" * 64,
        "proper_policy_certificate_sha256": None,
        "test_data_used": False,
    }
    economic_manifest = seal_economic_exit_step_manifest(
        {
            "schema_version": ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
            "split": "train",
            "lifecycle_manifest_sha256": manifest["manifest_sha256"],
            "economics_objective_contract_sha256": objective["contract_sha256"],
            "economic_step_model_sha256": "d" * 64,
            "economic_step_source_manifest_sha256": "e" * 64,
            "test_data_used": False,
        }
    )

    def provider(*_args):
        raise AssertionError("constructor must remain lazy")

    adapter = UnifiedExitDatasetAdapterV2.from_random_access_index_v1(
        index_rows=frame,
        index_manifest=manifest,
        source_owner=object(),
        epoch_index=0,
        economics_readiness=readiness,
        economic_exit_step_manifest=economic_manifest,
        economic_exit_step_provider=provider,
        mtf_materializer=lambda _times: {},
        per_tf_seq_lens={"M5": 1, "M15": 1, "H1": 1, "H4": 1, "D1": 1},
        mtf_cache_identity_sha256="c" * 64,
    )
    assert adapter._manifest["full_prefix_states_stored"] is False
    assert adapter._manifest["chunk_pointers_stored"] is False
    assert adapter._rows["entry_m1_start_row"].tolist() == [10, 14]


def _source_path_fixture(tmp_path):
    import hashlib
    import json
    from gx1.scripts import materialize_unified_exit_random_access_index_v1 as owner
    def write(path, value, seal=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if seal:
            value = {k: v for k, v in value.items() if k != seal}
            value[seal] = canonical_sha256(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        return value
    def bind(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    data = tmp_path / "adopted_existing_bytes"
    data.mkdir()
    rows, clocks, children, entries = {}, {}, {}, {}
    for split, count in [("train", 2), ("val", 1)]:
        parquet = data / f"original_{split}.parquet"
        parquet.write_bytes(f"synthetic {split} parquet identity".encode())
        rows[split], clocks[split] = count, ("a" if split == "train" else "b") * 64
        entries[split] = parquet
        children[split] = {"parquet_path": str(parquet), "parquet_sha256": bind(parquet)["sha256"],
                           "rows": count, "clock_sha256": clocks[split]}
    windows = {"train": {"start_utc": "2011-06-01T00:00:00+00:00", "end_utc_exclusive": "2025-06-01T00:00:00+00:00"},
               "val": {"start_utc": "2025-06-01T00:00:00+00:00", "end_utc_exclusive": "2026-07-01T00:00:00+00:00"}}
    design_path = tmp_path / "design.json"
    write(design_path, {"schema_version": "gx1_frozen_chronological_learning_design_v1", "calendar": {
        "physical_source_splits": {"train": "train", "control": "val"},
        "physical_coordinate_namespaces_are_separate": True,
        "train_entry_start_inclusive": windows["train"]["start_utc"],
        "train_control_cutoff": windows["val"]["start_utc"],
        "development_control_entry_end_exclusive": windows["val"]["end_utc_exclusive"],
        "source_bindings": {split: {"physical_rows": rows[split], "clock_sha256": clocks[split],
                                   "parquet": bind(entries[split])} for split in rows}}})
    for split, child in children.items():
        path = tmp_path / "entry_metadata" / f"{split}.manifest.json"
        m = write(path, {"output_parquet_path": child["parquet_path"],
            "output_parquet_sha256": child["parquet_sha256"], "rows": child["rows"],
            "window_start_utc": windows[split]["start_utc"],
            "window_end_utc_exclusive": windows[split]["end_utc_exclusive"]}, "manifest_sha256")
        child.update(manifest_path=str(path), manifest_file_sha256=bind(path)["sha256"],
                     manifest_contract_sha256=m["manifest_sha256"])
    admission_path = tmp_path / "admission.json"
    admission = write(admission_path, {"decision": "PASS", "test_accessed": False,
        "chronological_learning_design": bind(design_path), "windows": windows,
        "parent_admission_scope": "frozen_entry_bytes_only_no_parent_m1_states",
        "splits": children}, "witness_sha256")
    recipe = {"schema_version": "gx1_unified_exit_pilot_final_bindings_recipe_v1",
              "test_accessed": False, "child_admission": bind(admission_path), "splits": {}}
    for split in children:
        m1 = tmp_path / "actual_m1_views" / f"{split}.parquet"
        m1.parent.mkdir(exist_ok=True); m1.write_bytes(f"synthetic {split} M1 identity".encode())
        manifest = m1.with_suffix(".manifest.json")
        write(manifest, {"output_parquet": str(m1), "output_parquet_sha256": bind(m1)["sha256"],
            "child_admission_sha256": bind(admission_path)["sha256"],
            "child_parquet_sha256": children[split]["parquet_sha256"],
            "fit_window_start_utc": windows[split]["start_utc"],
            "fit_window_end_utc_exclusive": windows[split]["end_utc_exclusive"]})
        spec = {"m1_source": bind(m1), "m1_manifest": bind(manifest)}
        for name in ["summary_manifest", "successor_counts", "closure_authority"]:
            path = tmp_path / f"{split}_{name}.json";write(path, {"synthetic_identity": name})
            spec[name] = bind(path)
        recipe["splits"][split] = spec
    recipe_path = tmp_path / "recipe.json"
    recipe = write(recipe_path, recipe, "recipe_sha256")
    bundle = {"recipe_path": str(recipe_path), "recipe_file_sha256": bind(recipe_path)["sha256"],
              "recipe_sha256": recipe["recipe_sha256"], "child_admission": recipe["child_admission"]}
    return owner, bundle, recipe, admission, write, bind


def test_index_paths_reuse_exact_adopted_files_and_frozen_calendar(tmp_path):
    owner, bundle, recipe, admission, _, _ = _source_path_fixture(tmp_path)
    for split in ["train", "val"]:
        paths = owner._paths(tmp_path, split, tmp_path/"final", bundle)
        assert paths["entry_parquet"] == Path(admission["splits"][split]["parquet_path"])
        assert paths["m1_child"] == Path(recipe["splits"][split]["m1_source"]["path"])
        assert paths["closure_authority"] == Path(recipe["splits"][split]["closure_authority"]["path"])
        assert paths["learning_design"] == tmp_path/"design.json"
    assert not (tmp_path/"ENTRY_WINDOW").exists()


@pytest.mark.parametrize("mutation", ["source_bytes", "m1_window", "recipe_hash", "design_bytes"])
def test_index_paths_reject_changed_inputs_or_wrong_calendar(tmp_path, mutation):
    owner, bundle, recipe, admission, write, bind = _source_path_fixture(tmp_path)
    if mutation == "source_bytes":
        Path(recipe["splits"]["train"]["summary_manifest"]["path"]).write_text("{}")
    elif mutation == "m1_window":
        import json
        path = Path(recipe["splits"]["train"]["m1_manifest"]["path"])
        m = json.loads(path.read_text());m["fit_window_start_utc"] = "2021-06-01T00:00:00+00:00"
        write(path, m);recipe["splits"]["train"]["m1_manifest"] = bind(path)
        recipe = write(Path(bundle["recipe_path"]), recipe, "recipe_sha256")
        bundle.update(recipe_file_sha256=bind(Path(bundle["recipe_path"]))["sha256"],
                      recipe_sha256=recipe["recipe_sha256"])
    elif mutation == "recipe_hash":
        bundle["recipe_sha256"] = "f" * 64
    else:
        (tmp_path/"design.json").write_text("{}")
    with pytest.raises(RuntimeError):
        owner._paths(tmp_path, "train", tmp_path/"final", bundle)


def test_index_path_resolver_rejects_test_before_file_access(tmp_path, monkeypatch):
    from gx1.scripts import materialize_unified_exit_random_access_index_v1 as owner
    def forbidden(*args, **kwargs):
        raise AssertionError("TEST must be rejected before filesystem access")
    monkeypatch.setattr(Path, "stat", forbidden)
    with pytest.raises(RuntimeError, match="SPLIT_INVALID"):
        owner._paths(tmp_path, "test", tmp_path, {})


@pytest.mark.parametrize("mutation", [None, "dataset", "rows", "start", "end", "counts_path"])
def test_index_economics_uses_entry_identity_population_and_window(tmp_path, mutation):
    from gx1.scripts import materialize_unified_exit_random_access_index_v1 as owner
    from gx1.scripts.build_unified_exit_no_cap_authority_v1 import build_no_cap_authority
    from tests.test_unified_exit_no_cap_authority_v1 import _readiness, _facts
    readiness = tmp_path/"readiness.json";_readiness(readiness)
    result = build_no_cap_authority(output_dir=tmp_path/"authority", dataset_run_id="physical-child",
        split="train", entry_rows=3, coverage_start_utc="2024-01-01T00:00:00+00:00",
        coverage_end_utc="2025-01-01T00:00:00+00:00", economics_readiness_path=readiness,
        economics_fact_manifest_path=_facts(tmp_path), publish=True)
    paths = {"economic_authority": Path(result["authority_path"]), "economic_counts": Path(result["counts_path"])}
    entry = {"pilot_dataset_run_id": "physical-child", "window_start_utc": "2024-01-01T00:00:00+00:00",
             "window_end_utc_exclusive": "2025-01-01T00:00:00+00:00"}
    rows = 3
    if mutation == "dataset":entry["pilot_dataset_run_id"] = "foreign-child"
    elif mutation == "rows":rows = 2
    elif mutation == "start":entry["window_start_utc"] = "2023-01-01T00:00:00+00:00"
    elif mutation == "end":entry["window_end_utc_exclusive"] = "2026-01-01T00:00:00+00:00"
    elif mutation == "counts_path":paths["economic_counts"] = tmp_path/"unbound_counts.json"
    if mutation is None:
        checked = owner._require_index_economics(paths, split="train", entry_manifest=entry, entry_rows=rows)
        assert checked["no_observed_economic_terminals"] is True
    else:
        with pytest.raises(RuntimeError):
            owner._require_index_economics(paths, split="train", entry_manifest=entry, entry_rows=rows)


def test_index_stage_rejects_wrong_file_binding_before_publication(tmp_path):
    import json
    from gx1.scripts import materialize_unified_exit_random_access_index_v1 as owner
    stage = tmp_path/"stage";stage.mkdir()
    frame, _ = _built()
    parquet = stage/"train.random_access_index.parquet";frame.to_parquet(parquet, index=False)
    (stage/"train.manifest.json").write_text(json.dumps({
        "index_parquet_path": str(tmp_path/"final"/parquet.name),
        "index_parquet_sha256": "0"*64}))
    with pytest.raises(RuntimeError, match="STAGED_FILE_INVALID"):
        owner._read_staged_index(stage, tmp_path/"final", "train")
    assert not (tmp_path/"final").exists()
