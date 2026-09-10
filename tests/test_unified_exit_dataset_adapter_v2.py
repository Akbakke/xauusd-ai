from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
    UnifiedExitDatasetAdapterV2,
    seal_economic_exit_step_manifest,
    seal_economic_training_projection,
)
from gx1.contracts.unified_exit_lifecycle_v2 import unified_exit_lifecycle_v2_contract
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
    build_market_closure_authority,
    seal_exact_market_schedule,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_first_state_entry_bridge_witness,
    build_physical_summary_sample_authority,
    fit_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_training_items,
)
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset,
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
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
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
                        "continuous_m1"
                        if action == "hold"
                        else "instantaneous_execution"
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
        )
        == 0
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
        "0:0:hold": _steps(
            count=699, terminal=False, hurdle_sha=hurdle_sha, action="hold"
        ),
        "0:1:exit_now": _steps(count=700, terminal=True, hurdle_sha=hurdle_sha),
        "0:1:hold": _steps(
            count=699, terminal=False, hurdle_sha=hurdle_sha, action="hold"
        ),
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

    class FastProjectionProvider:
        def __init__(self):
            self.scalar_calls = 0

        def __call__(self, entry, side, action, start, stop):
            self.scalar_calls += 1
            return provide_step_slice(entry, side, action, start, stop)

        def materialize_training_projection(self, entry, side, start, stop, hold_stop):
            contract = readiness["economics_objective_contract"]
            exit_steps = streams[f"{entry}:{side}:exit_now"][start:stop]
            hold_steps = streams[f"{entry}:{side}:hold"][start:hold_stop]
            event_index = {"EXIT_NOW": 0, "HOLD": 1, "ECONOMIC_TERMINAL": 2}
            return seal_economic_training_projection(
                {
                    "schema_version": ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
                    "entry_row_index": entry,
                    "side_index": side,
                    "start_state_index": start,
                    "stop_state_index": stop,
                    "hold_stop_state_index": hold_stop,
                    "exit_event_kind_index": np.asarray(
                        [event_index[step["event_kind"]] for step in exit_steps],
                        dtype="u1",
                    ),
                    "exit_reward_bps": np.asarray(
                        [
                            economics.compose_economic_step(step, contract=contract)[
                                "undiscounted_risk_adjusted_utility_increment_bps"
                            ]
                            for step in exit_steps
                        ],
                        dtype="<f8",
                    ),
                    "hold_event_kind_index": np.asarray(
                        [event_index[step["event_kind"]] for step in hold_steps],
                        dtype="u1",
                    ),
                    "hold_reward_bps": np.asarray(
                        [
                            economics.compose_economic_step(step, contract=contract)[
                                "undiscounted_risk_adjusted_utility_increment_bps"
                            ]
                            for step in hold_steps
                        ],
                        dtype="<f8",
                    ),
                    "economic_step_model_sha256": "d" * 64,
                    "economic_step_source_manifest_sha256": "e" * 64,
                }
            )

    fast_provider = FastProjectionProvider()
    fast_adapter = UnifiedExitDatasetAdapterV2(
        compact_rows=compact,
        compact_manifest=manifest,
        source_owner=dataset,
        epoch_index=epoch,
        expected_m1_source_sha256="a" * 64,
        expected_entry_binding_sha256="f" * 64,
        expected_gap_classification_source_sha256="6" * 64,
        economics_readiness=readiness,
        economic_exit_step_manifest=economic_manifest,
        economic_exit_step_provider=fast_provider,
        mtf_materializer=dataset._get_exit_multi_tf_episode_histories,
        per_tf_seq_lens=dataset.per_tf_seq_lens,
        mtf_cache_identity_sha256=dataset._multi_tf_cache_identity_sha256,
    )
    fast_pack = fast_adapter.materialize(0)
    assert fast_pack is not None
    assert fast_provider.scalar_calls == 0
    for name, old_value in pack.items():
        if isinstance(old_value, np.ndarray):
            new_value = fast_pack[name]
            assert old_value.dtype == new_value.dtype
            assert old_value.shape == new_value.shape
            assert old_value.tobytes() == new_value.tobytes()

    schedule = seal_exact_market_schedule(
        {
            "schema_version": MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
            "decision": "PASS",
            "instrument": "XAU_USD",
            "timeframe": "M1",
            "coverage_start_utc": times[0].isoformat(),
            "coverage_end_utc_exclusive": (
                times[-1] + pd.Timedelta(minutes=1)
            ).isoformat(),
            "interval_semantics": "left_closed_right_open_utc",
            "source_method": "externally_sourced_exact_xau_utc_closure_intervals_v1",
            "source_reference_sha256": "1" * 64,
            "intervals": [],
            "test_data_used": False,
        }
    )
    closure = build_market_closure_authority(
        m1_times=times,
        m1_source_path=Path("/immutable/train.m1.parquet"),
        m1_source_sha256="a" * 64,
        m1_source_manifest_path=Path("/immutable/train.m1.manifest.json"),
        m1_source_manifest_sha256="2" * 64,
        exact_schedule=schedule,
        exact_schedule_path=Path("/immutable/schedule.json"),
        exact_schedule_file_sha256="3" * 64,
    )
    sampler = build_random_access_sampler_contract(
        split="train",
        source_lineage_sha256="4" * 64,
        transition_budget_per_epoch=4,
        transitions_per_entry=4,
        entry_pair_population=1,
    )
    summary_authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[699],
        source_lineage_sha256=sampler["source_lineage_sha256"],
    )
    fit_rows = summary_authority["fit_row_count"]
    fitted = fit_lifetime_summary_normalization(
        values=np.arange(fit_rows * 7, dtype=np.float64).reshape(fit_rows, 7),
        sample_authority=summary_authority,
    )
    summary_manifest = {
        "schema_version": "gx1_unified_exit_pilot_summary_fit_inputs_v1",
        "decision": "PASS",
        "split": "train",
        "source_lineage_sha256": sampler["source_lineage_sha256"],
        "child_admission_sha256": "5" * 64,
        "child_parquet_sha256": "6" * 64,
        "m1_source_sha256": "a" * 64,
        "m1_manifest_sha256": "7" * 64,
        "closure_authority_file_sha256": "8" * 64,
        "closure_authority_sha256": closure["artifact_sha256"],
        "entry_pair_population": 1,
        "successor_counts_sha256": hashlib.sha256(
            np.asarray([699], dtype="<i8").tobytes()
        ).hexdigest(),
        "successor_transition_total": 699,
        "summary_sample_authority": summary_authority,
        "lifetime_summary_normalization": fitted,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }
    summary_manifest["manifest_sha256"] = _sha(summary_manifest)
    witness = build_first_state_entry_bridge_witness(
        split="train",
        entry_times=[times[entry_start] - pd.Timedelta(minutes=5)],
        m1_times=times,
        child_admission_sha256="5" * 64,
        child_parquet_sha256="6" * 64,
        entry_sequence_audit_sha256="9" * 64,
        m1_source_sha256="a" * 64,
        closure_authority_sha256=closure["artifact_sha256"],
        state_view_source_sha256="d" * 64,
        lifetime_summary_registry_sha256=summary_authority[
            "lifetime_summary_registry_sha256"
        ],
        train_normalization_sha256=fitted["normalization_sha256"],
        m1_bid_open=dataset._m1["bid_open"],
        m1_ask_open=dataset._m1["ask_open"],
    )
    fast_provider.market_closure_authority_sha256 = closure["artifact_sha256"]
    fast_provider.economic_exit_step_manifest = economic_manifest
    fast_adapter.configure_random_access_training_v1(
        sampler_contract=sampler,
        successor_transition_counts=[699],
        summary_fit_manifest=summary_manifest,
        market_closure_authority=closure,
        normalization_artifact=fitted,
        first_state_bridge_witness=witness,
        random_access_m1_times=times,
        parent_m1_row_offset=0,
        expected_child_parquet_sha256="6" * 64,
        expected_state_view_source_sha256="d" * 64,
    )
    item = fast_adapter.materialize_random_access_training_item_v1(
        0, outer_batch_index=0
    )
    assert item is not None
    assert len(item["transitions"]) == 4
    assert item["anchor"]["sample"]["state_index"] == 0
    random_bindings = fast_adapter.random_access_training_bindings_v1()
    random_batch = collate_random_access_training_items(
        [item],
        outer_batch_size=1,
        sampler_contract=random_bindings["sampler_contract"],
        normalization_artifact=random_bindings["normalization_artifact"],
        expected_m1_source_sha256=random_bindings["m1_source_sha256"],
        expected_market_closure_authority_sha256=random_bindings[
            "market_closure_authority_sha256"
        ],
        expected_economic_step_manifest_sha256=random_bindings[
            "economic_step_manifest_sha256"
        ],
        expected_economics_objective_contract_sha256=random_bindings[
            "economics_objective_contract_sha256"
        ],
        device=torch.device("cpu"),
    )
    assert random_batch["transition_count"] == 4
    assert random_batch["selected_entry_count"] == 1

    validation_packs = dataset.materialize_exit_validation_chunks_v2(0)
    assert len(validation_packs) == 4
    assert {(item["side_index"], item["chunk_index"]) for item in validation_packs} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    with pytest.raises(RuntimeError, match="RANDOM_ACCESS_TRAIN_NOT_CONFIGURED"):
        _episode_native_exit_train(
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
