from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

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
from gx1.contracts.unified_exit_gate_evidence_v1 import (
    COOPERATION_GATE_WIDTHS as EXIT_GATE_WIDTHS,
    FEATURE_TF_GATE_SHAPE as EXIT_FEATURE_GATE_SHAPE,
)
from gx1.contracts import unified_exit_economics_objective_v2 as economics_owner
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
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _episode_native_exit_train,
    _fitted_q_targets_for_chunk_v2,
    _new_cooperation_gate_epoch_accumulator,
    _new_feature_tf_gate_epoch_accumulator,
)
from tests.test_unified_exit_random_access_training_v1 import (
    _item as _random_access_item,
    _normalization as _random_access_normalization,
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
    assert order == outcome_blind_chunk_permutation(**{**identity, "side_index": 0})
    assert [
        outcome_blind_chunk_index(epoch_index=epoch, **identity)
        for epoch in range(14)
    ] == list(order) * 2
def _economics_readiness(rho: float = 0.05):
    train_split = "1" * 64
    train_fold = "2" * 64
    lineage = "3" * 64
    policy = "4" * 64
    hurdle = economics_owner.seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": economics_owner.CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": train_split,
            "train_fold_sha256": train_fold,
            "source_lineage_sha256": lineage,
            "annual_continuous_hurdle_rate": rho,
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": economics_owner.SECONDS_PER_YEAR,
            "fit_method": "unit_train_only",
            "fit_evidence_sha256": "5" * 64,
        }
    )
    objective = economics_owner.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256=train_split,
        expected_train_fold_sha256=train_fold,
        expected_source_lineage_sha256=lineage,
        policy_sha256=policy,
    )
    return {
        "schema_version": "gx1_unified_exit_training_economics_readiness_v3",
        "mode": "economics_objective_v2",
        "capital_hurdle_artifact": hurdle,
        "economics_objective_contract": objective,
        "expected_train_split_sha256": train_split,
        "expected_train_fold_sha256": train_fold,
        "expected_source_lineage_sha256": lineage,
        "policy_sha256": policy,
        "proper_policy_certificate_sha256": None,
        "test_data_used": False,
    }


def _pack(
    *,
    chunk_start: int,
    valid_count: int,
    successor: bool,
    censored: bool,
):
    encoded = chunk_start + valid_count + int(successor)
    warm = EXIT_FEATURE_SEQUENCE_BARS - 1
    base = pd.Timestamp("2025-01-01T00:00:00Z").value
    local_times = base + np.arange(warm + encoded, dtype=np.int64) * 60_000_000_000
    state_times = local_times[warm:]
    action_valid = np.ones((valid_count, 2), dtype=np.bool_)
    supervision = action_valid.copy()
    successor_observed = np.ones(valid_count, dtype=np.bool_)
    successor_observed[-1] = successor
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
        "economic_exit_step_manifest_sha256": "a" * 64,
        "economic_exit_step_stream_sha256": "b" * 64,
        "economic_hold_step_stream_sha256": "c" * 64,
        "scheduled_pair_chunk_pointer_sha256": "d" * 64,
        "multi_tf_cache_identity_sha256": "6" * 64,
        "unbounded_exit_training_readiness": _economics_readiness(),
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
            (encoded, UNIFIED_EXIT_PATH_FEATURE_DIM), dtype=np.float32
        ),
        "exit_entry_bid_ask": np.asarray([1.0, 1.1]),
        "exit_now_reward_bps": np.zeros(valid_count, dtype=np.float32),
        "hold_immediate_reward_bps": np.zeros(valid_count, dtype=np.float32),
        "exit_policy_action_valid_mask": action_valid,
        "exit_bellman_target_valid_mask": supervision,
        "exit_successor_observed_mask": successor_observed,
        "exit_state_valid_mask": np.ones(valid_count, dtype=np.bool_),
        "exit_terminal_mask": np.zeros(valid_count, dtype=np.bool_),
        "exit_terminal_reason_index": np.zeros(valid_count, dtype=np.int64),
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
    chunk_start, valid_count, successor, censored, tmp_path
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


def test_lifecycle_v2_right_censors_before_unknown_clock_gap():
    before = pd.date_range("2025-01-01", periods=700, freq="min", tz="UTC")
    after = pd.date_range(before[-1] + pd.Timedelta(minutes=3), periods=30, freq="min")
    times = before.append(after)
    terminals = {(0, 0): None, (0, 1): None}
    chunks, _manifest = build_unified_exit_lifecycle_chunks_v2(
        entry_m1_start_rows=[100],
        m1_times=times,
        split="train",
        split_end=times[-1] + pd.Timedelta(minutes=1),
        terminal_state_count_by_entry_side=terminals,
        economic_lifecycle_authority=_authority(terminals),
        m1_source_sha256="3" * 64,
    )
    for side_index in (0, 1):
        rows = chunks.loc[chunks["side_index"] == side_index]
        assert rows["valid_state_count"].tolist() == [512, 88]
        assert rows.iloc[-1]["right_censored"]


def test_trainer_v2_bootstraps_from_frozen_successor_state(tmp_path):
    pack = _pack(
        chunk_start=0,
        valid_count=3,
        successor=True,
        censored=False,
    )

    class _Target:
        training = False

        def forward_exit_incremental_prefix(self, **kwargs):
            state_count = kwargs["exit_path_x"].shape[2]
            q = torch.zeros((1, 2, state_count, 2), dtype=torch.float32)
            q[0, 0, -1] = torch.tensor([3.0, 7.0])
            return {
                "exit_action_q_bps": q,
                "exit_action_valid_mask": torch.ones_like(q, dtype=torch.bool),
            }

    targets, target_mask, _current = _fitted_q_targets_for_chunk_v2(
        target_model=_Target(),
        target_entry_decision_representation=torch.zeros((1, 256)),
        chunk_pack=pack,
        per_tf_seq_lens={name: 2 for name in EXIT_MTF_CONTEXT_TIMEFRAMES},
        expected_mtf_cache_identity_sha256="6" * 64,
        device=torch.device("cpu"),
    )
    assert targets.shape == (1, 1, 3, 2)
    assert target_mask[0, 0, -1, 0]
    assert 6.99 < float(targets[0, 0, -1, 0]) < 7.0


def test_canonical_trainer_dispatches_to_v2_chunk_consumer():
    pack = _pack(chunk_start=0, valid_count=3, successor=True, censored=False)

    class _Model(torch.nn.Module):

        def __init__(self, *, online):
            super().__init__()
            self.training = False
            self.online = online
            self.task_log_variances = {
                "unified_exit_action": torch.tensor(0.0)
            }

        def forward_exit_incremental_prefix(self, **kwargs):
            state_count = kwargs["exit_path_x"].shape[2]
            token = kwargs["entry_decision_representation"]
            base = token.sum(dim=1).view(1, 1, 1, 1) if self.online else 0.0
            q = torch.zeros((1, 2, state_count, 2), dtype=torch.float32) + base
            if not self.online:
                q[0, 0, -1] = torch.tensor([3.0, 7.0])
            return {
                "exit_action_q_bps": q,
                "exit_action_valid_mask": torch.ones_like(q, dtype=torch.bool),
            }

        def forward_exit_random_access_batch(self, **kwargs):
            state_count = kwargs["m1_local_history_x"].shape[0]
            token = kwargs["entry_decision_representation"]
            base = token.sum(dim=1).view(-1, 1, 1) if self.online else 0.0
            q = torch.zeros((state_count, 2, 2), dtype=torch.float32) + base
            if not self.online:
                q[:, 0, 0] = 3.0
                q[:, 0, 1] = 7.0
            return {
                "exit_action_q_bps": q,
                "exit_action_valid_mask": kwargs["action_valid_mask"],
                "exit_specialist_gate": torch.full(
                    (state_count, EXIT_GATE_WIDTHS["specialist_gate"]), 0.5
                ),
                "exit_tf_gate": torch.full(
                    (state_count, EXIT_GATE_WIDTHS["tf_gate"]), 0.5
                ),
                "exit_family_tf_cooperation_gate": torch.full(
                    (state_count, EXIT_GATE_WIDTHS["family_tf_cooperation_gate"]),
                    0.5,
                ),
                "exit_family_tf_feature_gate": torch.full(
                    (state_count, *EXIT_FEATURE_GATE_SHAPE), 0.5
                ),
            }

    class _Dataset:
        contract, item = _random_access_item()
        view = item["transitions"][0]["state_view"]

        class _Adapter:
            @staticmethod
            def random_access_training_bindings_v1():
                return {
                    "sampler_contract": _Dataset.contract,
                    "normalization_artifact": _random_access_normalization(),
                    "m1_source_sha256": _Dataset.view["m1_source_sha256"],
                    "market_closure_authority_sha256": _Dataset.view[
                        "market_closure_authority_sha256"
                    ],
                    "economic_step_manifest_sha256": _Dataset.view[
                        "economic_step_manifest_sha256"
                    ],
                    "economics_objective_contract_sha256": _Dataset.view[
                        "economics_objective_contract_sha256"
                    ],
                }

            @staticmethod
            def materialize_random_access_training_item_v1(
                entry_row_index, *, outer_batch_index
            ):
                assert entry_row_index == 0
                item = dict(_Dataset.item)
                item["outer_batch_index"] = outer_batch_index
                return item

        _unified_exit_lifecycle_v2 = _Adapter()

    gradients, stats, _entry_targets, entry_valid = _episode_native_exit_train(
        model=_Model(online=True),
        target_model=_Model(online=False),
        entry_decision_representations=torch.zeros((1, 256)),
        target_entry_decision_representations=torch.zeros((1, 256)),
        entry_row_indices=torch.tensor([0]),
        dataset=_Dataset(),
        device=torch.device("cpu"),
        grad_accum_steps=1,
        exit_cooperation_gate_epoch=_new_cooperation_gate_epoch_accumulator(
            EXIT_GATE_WIDTHS
        ),
        exit_feature_tf_gate_epoch=_new_feature_tf_gate_epoch_accumulator(
            EXIT_FEATURE_GATE_SHAPE
        ),
    )
    assert stats["eligible_entry_rows"] == 1
    assert torch.isfinite(gradients).all() and gradients.abs().sum() > 0
    assert entry_valid.all()
