import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_random_access_index_v1 import (
    RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
    build_random_access_index_v2,
    canonical_sha256,
    index_stream_sha256,
    parent_entry_mapping_sha256,
    require_parent_entry_coordinate_equivalence,
)


def test_corrected_parent_requires_the_entire_original_clock(tmp_path):
    def bind(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def clock_sha(clock):
        return hashlib.sha256(clock.as_unit("ns").asi8.astype("<i8").tobytes()).hexdigest()

    def parent_files(name, clock, correction):
        parquet = tmp_path / f"{name}.parquet"
        pd.DataFrame({"time": clock, "target_correction": correction}).to_parquet(parquet)
        manifest = tmp_path / f"{name}.json"
        manifest.write_text(json.dumps({"output_data_path": str(parquet),
            "extra": {"pretest_test_guard": {"test_accessed": False}}}))
        return bind(parquet), bind(manifest)

    m1 = pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC")
    parent = pd.DatetimeIndex([m1[i] for i in [0, 2, 5, 7, 9, 12]])
    child = parent[[2, 4]]
    frame, _ = build_random_access_index_v2(
        split="val", entry_times=child, parent_entry_times=parent,
        child_m1_times=m1[5:25], parent_m1_times=m1,
        successor_transition_counts=[3, 4], entry_bid=[100., 101.],
        entry_ask=[100.1, 101.1], episode_binding_sha256_by_entry=["a"*64, "b"*64],
        entry_fill_binding_sha256_by_entry=["b"*64, "a"*64],
    )
    old_parquet, old_manifest = parent_files("old", parent, [0.] * len(parent))
    child_path = tmp_path / "child.parquet"
    pd.DataFrame({"time": child}).to_parquet(child_path)
    rows = frame["parent_entry_row_index"].to_numpy(dtype="<i8")
    manifest = {
        "schema_version": RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
        "decision": "PASS", "split": "val", "entry_row_count": 2,
        "successor_transition_total": 7, "economic_terminal_count": 0,
        "split_end_is_right_censor": True, "storage_granularity": "one_row_per_entry",
        "full_prefix_states_stored": False, "chunk_pointers_stored": False,
        "target_q_stored": False, "test_accessed": False,
        "index_stream_sha256": index_stream_sha256(frame),
        "source_bindings": {"entry_parquet": bind(child_path),
            "entry_manifest": old_manifest, "parent_entry_parquet": old_parquet,
            "parent_entry_manifest": old_manifest},
        "child_entry_clock_sha256": clock_sha(child),
        "parent_entry_clock_sha256": clock_sha(parent),
        "parent_entry_row_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "parent_entry_mapping_sha256": parent_entry_mapping_sha256(frame),
        "parent_entry_source_sha256": old_parquet["sha256"],
        "parent_entry_manifest_sha256": old_manifest["sha256"],
        "parent_entry_source_rows": len(parent),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    current_parquet, current_manifest = parent_files("corrected", parent, [1.] * len(parent))
    args = dict(index_manifest=manifest, index_frame=frame, expected_split="val")
    evidence = require_parent_entry_coordinate_equivalence(
        **args, parent_parquet=current_parquet, parent_manifest=current_manifest,
    )
    assert evidence["decision"] == "PASS_EXACT_COORDINATES"
    assert evidence["launch_parent_parquet"] != evidence["recorded_parent_parquet"]
    assert evidence["row_coordinates_changed"] is False

    # Child entries and their positions are unchanged; a drift elsewhere in the
    # parent still invalidates the complete clock proof.
    shifted = pd.DatetimeIndex([m1[1], *parent[1:]])
    bad_parquet, bad_manifest = parent_files("bad_clock", shifted, [1.] * len(parent))
    with pytest.raises(RuntimeError, match="COORDINATES_DIFFER"):
        require_parent_entry_coordinate_equivalence(
            **args, parent_parquet=bad_parquet, parent_manifest=bad_manifest,
        )
    with pytest.raises(RuntimeError, match="BINDING_INVALID"):
        require_parent_entry_coordinate_equivalence(
            **args, parent_parquet={**current_parquet, "sha256": "0" * 64},
            parent_manifest=current_manifest,
        )
