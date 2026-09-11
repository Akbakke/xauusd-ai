import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_random_access_index_v1 import (
    RANDOM_ACCESS_INDEX_SCHEMA_VERSION,
    build_random_access_index,
    canonical_sha256,
    index_stream_sha256,
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
