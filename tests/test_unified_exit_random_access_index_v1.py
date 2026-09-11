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
