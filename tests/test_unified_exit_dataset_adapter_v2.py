from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.entry_exit_feature_base_v1 import EXIT_MTF_CONTEXT_TIMEFRAMES
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.unified_exit_dataset_adapter_v2 import (
    ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
    UnifiedExitDatasetAdapterV2,
    seal_economic_exit_step_manifest,
)
from gx1.contracts.unified_exit_lifecycle_v2 import unified_exit_lifecycle_v2_contract
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset,
    _episode_native_exit_eval_loss,
    _episode_native_exit_train,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    COMPACT_LIFECYCLE_SCHEMA_VERSION,
    _compact_pointer_stream_sha256,
    build_compact_split,
    pair_chunk_for_epoch,
)


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _readiness(rho=0.05):
    hurdle = economics.seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": economics.CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": "1" * 64,
            "train_fold_sha256": "2" * 64,
            "source_lineage_sha256": "3" * 64,
            "annual_continuous_hurdle_rate": rho,
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
    return {
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


def _component(value, source="7" * 64):
    return {"status": "COMPLETE", "value_bps": value, "source_artifact_sha256": source}


def _steps(*, count, terminal, hurdle_sha, action="exit_now"):
    values = []
    for index in range(count):
        event = (
            "HOLD"
            if action == "hold"
            else "ECONOMIC_TERMINAL"
            if terminal and index == count - 1
            else "EXIT_NOW"
        )
        time_ns = 1_600_000_000_000_000_000 + index * 60_000_000_000
        end_ns = time_ns + 60_000_000_000 if action == "hold" else time_ns
        values.append(
            {
                "schema_version": economics.ECONOMIC_STEP_SCHEMA_VERSION,
                "event_kind": event,
                "interval_start_time_ns": time_ns,
                "interval_end_time_ns": end_ns,
                "gross_price_cashflow": _component(
                    0.0 if action == "hold" else float(index) / 100.0
                ),
                "commission": _component(0.1),
                "execution_slippage": _component(0.1),
                "financing_or_swap": _component(0.0),
                "guaranteed_execution_fee": _component(0.0),
                "risk_utility_penalty": _component(0.0),
                "same_capital_hurdle_running_cost": _component(0.0, hurdle_sha),
                "gap": {
                    "status": "COMPLETE",
                    "classification": (
                        "continuous_m1" if action == "hold" else "instantaneous_execution"
                    ),
                    "source_manifest_sha256": "8" * 64,
                    "classification_artifact_sha256": "9" * 64,
                },
            }
        )
    return values


class _Model:
    training = False

    def __init__(self, *, online):
        self.online = online
        self.task_log_variances = {"unified_exit_action": torch.tensor(0.0)}

    def forward_exit_incremental_prefix(self, **kwargs):
        states = kwargs["exit_path_x"].shape[2]
        token = kwargs["entry_decision_representation"]
        base = token.sum(dim=1).view(1, 1, 1, 1) if self.online else 0.0
        q = torch.zeros((1, 2, states, 2), dtype=torch.float32) + base
        if self.online:
            q[..., 1] = q[..., 1] + 1.0
        else:
            q[..., 1] = 2.0
        return {
            "exit_action_q_bps": q,
            "exit_action_valid_mask": torch.ones_like(q, dtype=torch.bool),
        }


def test_compact_producer_to_dataset_api_to_canonical_trainer():
    times = pd.date_range("2020-01-01", periods=7000, freq="min", tz="UTC")
    entry_start = 4000
    compact = build_compact_split(
        entry_times=[times[entry_start] - pd.Timedelta(minutes=5)],
        m1_times=times,
        split="train",
        split_end=times[-1] + pd.Timedelta(minutes=1),
        terminal_state_count_by_entry_side={(0, 0): 700, (0, 1): 700},
        m1_source_sha256="a" * 64,
        gap_classification_source_sha256="6" * 64,
        entry_binding_sha256="f" * 64,
    )
    lineage = "b" * 64
    epoch = next(
        value
        for value in range(2)
        if pair_chunk_for_epoch(
            chunk_count=2,
            epoch_index=value,
            lineage_sha256=lineage,
            split="train",
            entry_row_index=0,
        ) == 0
    )
    manifest = {
        **unified_exit_lifecycle_v2_contract(),
        "compact_schema_version": COMPACT_LIFECYCLE_SCHEMA_VERSION,
        "split": "train",
        "split_end_utc": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "test_accessed": False,
        "target_q_stored": False,
        "compact_pointer_stream_sha256": _compact_pointer_stream_sha256(compact),
        "schedule_lineage_sha256": lineage,
    }
    manifest["manifest_sha256"] = _sha(manifest)
    readiness = _readiness()
    hurdle_sha = readiness["capital_hurdle_artifact"]["artifact_sha256"]
    streams = {
        "0:0:exit_now": _steps(count=700, terminal=True, hurdle_sha=hurdle_sha),
        "0:0:hold": _steps(count=699, terminal=False, hurdle_sha=hurdle_sha, action="hold"),
        "0:1:exit_now": _steps(count=700, terminal=True, hurdle_sha=hurdle_sha),
        "0:1:hold": _steps(count=699, terminal=False, hurdle_sha=hurdle_sha, action="hold"),
    }
    economic_manifest = seal_economic_exit_step_manifest(
        {
            "schema_version": ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
            "split": "train",
            "lifecycle_manifest_sha256": manifest["manifest_sha256"],
            "economics_objective_contract_sha256": readiness[
                "economics_objective_contract"
            ]["contract_sha256"],
            "economic_step_model_sha256": "d" * 64,
            "economic_step_source_manifest_sha256": "e" * 64,
            "test_data_used": False,
        }
    )

    def provide_step_slice(entry, side, action, start, stop):
        values = streams[f"{entry}:{side}:{action}"][start:stop]
        envelope = {
            "schema_version": "gx1_unified_exit_economic_step_slice_v1",
            "entry_row_index": entry,
            "side_index": side,
            "action": action,
            "start_state_index": start,
            "stop_state_index": stop,
            "steps": values,
            "economic_step_model_sha256": "d" * 64,
            "economic_step_source_manifest_sha256": "e" * 64,
        }
        envelope["slice_sha256"] = _sha(envelope)
        return envelope
    dataset = EntryV10CtxDataset.__new__(EntryV10CtxDataset)
    dataset._unified_exit_lifecycle_v2 = None
    dataset.per_tf_seq_lens = {tf: 2 for tf in EXIT_MTF_CONTEXT_TIMEFRAMES}
    dataset._multi_tf_cache_identity_sha256 = "c" * 64
    dataset._feature_row_offset = 0
    dataset._m1_times = times
    dataset._m1_feature_times = times
    dataset._m1_features = {
        "signal": np.zeros((len(times), MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32),
        "ctx_cont": np.zeros((len(times), MODEL_NATIVE_CTX_CONT_DIM), dtype=np.float32),
        "ctx_cat": np.zeros((len(times), MODEL_NATIVE_CTX_CAT_DIM), dtype=np.int64),
    }
    base_price = np.full(len(times), 1900.0)
    dataset._m1 = {
        "open": base_price,
        "high": base_price + 1.0,
        "low": base_price - 1.0,
        "close": base_price + 0.1,
        "bid_open": base_price - 0.1,
        "bid_high": base_price + 0.9,
        "bid_low": base_price - 1.1,
        "bid_close": base_price,
        "ask_open": base_price + 0.1,
        "ask_high": base_price + 1.1,
        "ask_low": base_price - 0.9,
        "ask_close": base_price + 0.2,
        "mid_open": base_price,
        "mid_high": base_price + 1.0,
        "mid_low": base_price - 1.0,
        "mid_close": base_price + 0.1,
        "volume": np.ones(len(times)),
    }
    dataset._multi_tf_feature_count = MULTI_TF_FEATURE_COUNT_V4
    dataset._multi_tf_feats = {}
    for tf in EXIT_MTF_CONTEXT_TIMEFRAMES:
        frame = pd.DataFrame(index=times)
        frame.attrs["ts_int64"] = np.asarray(times.asi8, dtype=np.int64)
        frame.attrs["feats_np"] = np.zeros(
            (len(times), MULTI_TF_FEATURE_COUNT_V4), dtype=np.float32
        )
        frame.attrs["causal_warmup_rows"] = 0
        dataset._multi_tf_feats[tf] = frame
    adapter = UnifiedExitDatasetAdapterV2(
        compact_rows=compact,
        compact_manifest=manifest,
        source_owner=dataset,
        epoch_index=epoch,
        expected_m1_source_sha256="a" * 64,
        expected_entry_binding_sha256="f" * 64,
        expected_gap_classification_source_sha256="6" * 64,
        economics_readiness=readiness,
        economic_exit_step_manifest=economic_manifest,
        economic_exit_step_provider=provide_step_slice,
        mtf_materializer=dataset._get_exit_multi_tf_episode_histories,
        per_tf_seq_lens=dataset.per_tf_seq_lens,
        mtf_cache_identity_sha256=dataset._multi_tf_cache_identity_sha256,
    )
    dataset.bind_unified_exit_lifecycle_v2(adapter)
    pack = dataset.materialize_exit_training_chunk_v2(0)
    assert pack is not None
    assert pack["valid_state_count"] == 512
    assert pack["encoded_prefix_state_count"] == 513
    assert pack["successor_available"] is True
    assert len(pack["exit_local_history_x"]) == 479 + 513
    assert np.allclose(pack["hold_immediate_reward_bps"], -0.2)
    validation_packs = dataset.materialize_exit_validation_chunks_v2(0)
    assert len(validation_packs) == 4
    assert {(item["side_index"], item["chunk_index"]) for item in validation_packs} == {
        (0, 0), (0, 1), (1, 0), (1, 1)
    }
    gradients, stats, _entry_targets, _entry_valid = _episode_native_exit_train(
        model=_Model(online=True),
        target_model=_Model(online=False),
        entry_decision_representations=torch.zeros((1, 256)),
        target_entry_decision_representations=torch.zeros((1, 256)),
        entry_row_indices=torch.tensor([0]),
        dataset=dataset,
        device=torch.device("cpu"),
        grad_accum_steps=1,
        exit_cooperation_gate_epoch={},
        exit_feature_tf_gate_epoch={},
    )
    assert stats["eligible_entry_rows"] == 1
    assert gradients.abs().sum() > 0
    val_loss, val_stats, _targets, _valid, _realized = (
        _episode_native_exit_eval_loss(
            model=_Model(online=True),
            target_model=_Model(online=False),
            entry_decision_representations=torch.zeros((1, 256)),
            target_entry_decision_representations=torch.zeros((1, 256)),
            entry_row_indices=torch.tensor([0]),
            dataset=dataset,
            device=torch.device("cpu"),
            exit_cooperation_gate_epoch={},
            exit_feature_tf_gate_epoch={},
        )
    )
    assert val_stats["eligible_entry_rows"] == 1
    assert torch.isfinite(val_loss)
    tampered = dict(pack)
    tampered["scheduled_pair_chunk_pointer_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="PACK_SCHEDULE_INVALID"):
        adapter.require_pack(tampered)
    missing = streams.pop(f"0:{pack['side_index']}:hold")
    try:
        with pytest.raises(RuntimeError, match="ECONOMIC_STEPS_MISSING"):
            dataset.materialize_exit_training_chunk_v2(0)
    finally:
        streams[f"0:{pack['side_index']}:hold"] = missing
