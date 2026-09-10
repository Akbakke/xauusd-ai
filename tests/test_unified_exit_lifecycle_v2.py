from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.unified_exit_episode_pack_v2 import (
    UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION,
    require_unified_exit_episode_pack_v2,
    seal_unified_exit_episode_pack_v2,
)
from gx1.contracts.unified_exit_lifecycle_v2 import (
    UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION,
    UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION,
    build_unified_exit_lifecycle_chunks_v2,
    outcome_blind_chunk_index,
    outcome_blind_chunk_permutation,
    require_unified_exit_lifecycle_chunks_v2,
    terminal_state_counts_sha256,
)
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)


def _authority(terminals):
    return {
        "schema_version": UNIFIED_EXIT_ECONOMIC_AUTHORITY_SCHEMA_VERSION,
        "decision": "PASS",
        "authority_artifact_path": "/immutable/train/economic_lifecycle.json",
        "authority_artifact_sha256": "1" * 64,
        "terminal_state_counts_sha256": terminal_state_counts_sha256(terminals),
        "economic_terminal_definition_sha256": "2" * 64,
        "terminal_event_verifier_schema_version": (
            "gx1_economic_terminal_event_verifier_v1"
        ),
        "terminal_events_recomputed_from_train_val_only": True,
        "test_data_used": False,
    }


def test_lifecycle_v2_chunks_continue_beyond_512_and_bind_successor():
    times = pd.date_range("2025-01-01", periods=1300, freq="min", tz="UTC")
    terminals = {(0, 0): 700, (0, 1): None}
    chunks, manifest = build_unified_exit_lifecycle_chunks_v2(
        entry_m1_start_rows=[100],
        m1_times=times,
        split="train",
        split_end=times[1199] + pd.Timedelta(minutes=1),
        terminal_state_count_by_entry_side=terminals,
        economic_lifecycle_authority=_authority(terminals),
        m1_source_sha256="3" * 64,
    )
    long_rows = chunks.loc[chunks["side_index"] == 0].reset_index(drop=True)
    short_rows = chunks.loc[chunks["side_index"] == 1].reset_index(drop=True)
    assert long_rows["chunk_start_bars_in_trade"].tolist() == [0, 512]
    assert long_rows["valid_state_count"].tolist() == [512, 188]
    assert long_rows["successor_available"].tolist() == [True, False]
    assert long_rows["terminal_reason"].tolist() == ["none", "economic_terminal"]
    assert int(long_rows.loc[0, "successor_m1_row"]) == 612
    assert short_rows["chunk_start_bars_in_trade"].tolist() == [0, 512, 1024]
    assert short_rows["valid_state_count"].tolist() == [512, 512, 76]
    assert short_rows["right_censored"].tolist() == [False, False, True]
    assert manifest["capacity_forces_exit"] is False
    assert manifest["maximum_trade_duration_bars"] is None
    assert manifest["split"] == "train"
    require_unified_exit_lifecycle_chunks_v2(
        chunks, manifest=manifest, m1_times=times
    )


def test_lifecycle_v2_rejects_test_and_unproven_economic_authority():
    times = pd.date_range("2025-01-01", periods=600, freq="min", tz="UTC")
    terminals = {(0, 0): 20, (0, 1): 30}
    with pytest.raises(RuntimeError, match="SPLIT_FORBIDDEN"):
        build_unified_exit_lifecycle_chunks_v2(
            entry_m1_start_rows=[100],
            m1_times=times,
            split="test",
            split_end=times[-1] + pd.Timedelta(minutes=1),
            terminal_state_count_by_entry_side=terminals,
            economic_lifecycle_authority=_authority(terminals),
            m1_source_sha256="3" * 64,
        )
    authority = _authority(terminals)
    authority["terminal_events_recomputed_from_train_val_only"] = False
    with pytest.raises(RuntimeError, match="AUTHORITY_INVALID"):
        build_unified_exit_lifecycle_chunks_v2(
            entry_m1_start_rows=[100],
            m1_times=times,
            split="val",
            split_end=times[-1] + pd.Timedelta(minutes=1),
            terminal_state_count_by_entry_side=terminals,
            economic_lifecycle_authority=authority,
            m1_source_sha256="3" * 64,
        )


def test_chunk_schedule_is_deterministic_outcome_blind_permutation():
    identity = {
        "chunk_count": 7,
        "lineage_sha256": "a" * 64,
        "split": "train",
        "entry_row_index": 42,
        "side_index": 1,
    }
    order = outcome_blind_chunk_permutation(**identity)
    assert sorted(order) == list(range(7))
    assert order == outcome_blind_chunk_permutation(**identity)
    assert [
        outcome_blind_chunk_index(epoch_index=epoch, **identity)
        for epoch in range(14)
    ] == list(order) * 2
def _pack(*, chunk_start: int, valid_count: int, successor: bool, censored: bool):
    encoded = chunk_start + valid_count + int(successor)
    warm = EXIT_FEATURE_SEQUENCE_BARS - 1
    base = pd.Timestamp("2025-01-01T00:00:00Z").value
    local_times = base + np.arange(warm + encoded, dtype=np.int64) * 60_000_000_000
    state_times = local_times[warm:]
    action_valid = np.ones((2, valid_count, 2), dtype=np.bool_)
    supervision = action_valid.copy()
    successor_observed = np.ones((2, valid_count), dtype=np.bool_)
    successor_observed[:, -1] = successor
    supervision[..., 0] &= successor_observed
    value = {
        "schema_version": UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION,
        "lifecycle_schema_version": UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION,
        "split": "train",
        "entry_row_index": 4,
        "side_index": 0,
        "chunk_index": 7,
        "entry_m1_start_row": 100,
        "chunk_m1_start_row": 100 + chunk_start,
        "chunk_start_bars_in_trade": chunk_start,
        "valid_state_count": valid_count,
        "encoded_prefix_state_count": encoded,
        "successor_available": successor,
        "successor_prefix_state_index": chunk_start + valid_count if successor else -1,
        "right_censored": censored,
        "terminal_reason": "none",
        "lifecycle_manifest_sha256": "4" * 64,
        "chunk_pointer_stream_sha256": "5" * 64,
        "multi_tf_cache_identity_sha256": "6" * 64,
        "unbounded_exit_training_readiness": {
            "schema_version": "gx1_unified_exit_training_economics_readiness_v2",
            "mode": "elapsed_time_discount_v1",
            "qualification_artifact_path": "/immutable/train/economics.json",
            "qualification_artifact_sha256": "8" * 64,
            "economic_terminal_policy_sha256": "7" * 64,
            "train_capital_hurdle_annual_rate": 0.05,
            "train_capital_hurdle_source_sha256": "9" * 64,
            "hold_running_capital_charge_bps_per_second": 0.0,
            "proper_policy_certificate_sha256": None,
            "test_data_used": False,
        },
        "exit_local_history_x": np.zeros(
            (warm + encoded, MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32
        ),
        "exit_local_history_time_ns": local_times,
        "exit_state_ctx_cont": np.zeros(
            (encoded, MODEL_NATIVE_CTX_CONT_DIM), dtype=np.float32
        ),
        "exit_state_ctx_cat": np.zeros(
            (encoded, MODEL_NATIVE_CTX_CAT_DIM), dtype=np.int64
        ),
        "exit_state_row_time_ns": state_times,
        "exit_decision_time_ns": state_times + 60_000_000_000,
        "exit_path_x": np.zeros(
            (2, encoded, UNIFIED_EXIT_PATH_FEATURE_DIM), dtype=np.float32
        ),
        "exit_entry_bid_ask": np.asarray([[1.0, 1.1], [1.0, 1.1]]),
        "exit_now_reward_bps": np.zeros((2, valid_count), dtype=np.float32),
        "exit_policy_action_valid_mask": action_valid,
        "exit_bellman_target_valid_mask": supervision,
        "exit_successor_observed_mask": successor_observed,
        "exit_state_valid_mask": np.ones((2, valid_count), dtype=np.bool_),
        "exit_terminal_mask": np.zeros((2, valid_count), dtype=np.bool_),
        "exit_terminal_reason_index": np.zeros((2, valid_count), dtype=np.int64),
    }
    for tf in (name.lower() for name in EXIT_MTF_CONTEXT_TIMEFRAMES):
        value[f"exit_mtf_history_{tf}"] = np.zeros(
            (2, MULTI_TF_FEATURE_COUNT_V4), dtype=np.float32
        )
        value[f"exit_mtf_history_time_ns_{tf}"] = np.asarray([base, base + 1])
        value[f"exit_mtf_gather_{tf}"] = np.ones(encoded, dtype=np.int64)
    return seal_unified_exit_episode_pack_v2(value)


@pytest.mark.parametrize(
    ("chunk_start", "valid_count", "successor", "censored"),
    [(0, 512, True, False), (512, 188, False, True)],
)
def test_episode_pack_v2_carries_full_prefix_and_censor_semantics(
    chunk_start, valid_count, successor, censored
):
    pack = _pack(
        chunk_start=chunk_start,
        valid_count=valid_count,
        successor=successor,
        censored=censored,
    )
    require_unified_exit_episode_pack_v2(
        pack,
        per_tf_seq_lens={name: 2 for name in EXIT_MTF_CONTEXT_TIMEFRAMES},
        expected_mtf_cache_identity_sha256="6" * 64,
        context="UNIT",
    )
    assert len(pack["exit_local_history_x"]) == (
        479 + chunk_start + valid_count + int(successor)
    )
