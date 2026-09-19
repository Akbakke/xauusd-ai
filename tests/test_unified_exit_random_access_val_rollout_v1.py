from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_lifetime_summary_v1 import LIFETIME_SUMMARY_DIM
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
    build_market_closure_authority,
    m1_clock_sha256,
    seal_exact_market_schedule,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_physical_summary_sample_authority,
    fit_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_random_access_val_rollout_v1 import (
    RandomAccessValRolloutAdapterV1,
    VAL_ENTRY_COHORT_SIZE,
    build_random_access_val_rollout_contract,
    require_random_access_val_rollout_contract,
    require_random_access_val_rollout_result,
    run_random_access_val_rollout,
    unique_active_exit_actions,
)
from gx1.features.htf_features import MULTI_TF_TIMEFRAMES


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_unique_exit_action_rejects_active_tie_but_ignores_inactive_side():
    q = torch.tensor([[[0.0, 2.0], [0.0, 0.0]]])
    assert unique_active_exit_actions(q, np.array([[True, False]]))[0, 0] == 1
    with pytest.raises(RuntimeError, match="TIED_ACTION"):
        unique_active_exit_actions(q, np.array([[True, True]]))


def _readonly(value: object, dtype: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _objective(reward_accounting="terminal_cash_v2") -> dict:
    hurdle = economics.seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": economics.CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": "1" * 64,
            "train_fold_sha256": "2" * 64,
            "source_lineage_sha256": "3" * 64,
            "annual_continuous_hurdle_rate": 0.10,
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": economics.SECONDS_PER_YEAR,
            "fit_method": "train_only_capital_hurdle_fit_v1",
            "fit_evidence_sha256": "4" * 64,
        }
    )
    return economics.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256="1" * 64,
        expected_train_fold_sha256="2" * 64,
        expected_source_lineage_sha256="3" * 64,
        policy_sha256="5" * 64,
        reward_accounting=reward_accounting,
    )


def _normalization() -> dict:
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[2, 20],
        source_lineage_sha256="6" * 64,
    )
    rows = authority["fit_row_count"]
    values = np.arange(rows * LIFETIME_SUMMARY_DIM, dtype=np.float64).reshape(
        rows, LIFETIME_SUMMARY_DIM
    )
    return fit_lifetime_summary_normalization(
        values=values,
        sample_authority=authority,
    )


def _clock(*, gap: str | None = None, tail_rows: int = 3) -> pd.DatetimeIndex:
    context = pd.date_range("2026-06-05T13:00Z", periods=480, freq="min")
    if gap is None:
        return pd.DatetimeIndex(
            list(context)
            + list(
                pd.date_range(
                    context[-1] + pd.Timedelta(minutes=1),
                    periods=tail_rows - 1,
                    freq="min",
                )
            )
        )
    after = (
        context[-1] + pd.Timedelta(days=2, hours=1, minutes=5)
        if gap == "weekend"
        else context[-1] + pd.Timedelta(hours=3)
    )
    return pd.DatetimeIndex(list(context) + [after])


def _closure(clock: pd.DatetimeIndex, *, gap: str | None = None) -> dict:
    intervals = []
    if gap == "weekend":
        intervals.append(
            {
                "kind": "weekend",
                "start_utc": (clock[479] + pd.Timedelta(minutes=1)).isoformat(),
                "end_utc_exclusive": clock[480].isoformat(),
                "source_event_id": "synthetic-pretest-weekend",
            }
        )
    schedule = seal_exact_market_schedule(
        {
            "schema_version": MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
            "decision": "PASS",
            "instrument": "XAU_USD",
            "timeframe": "M1",
            "coverage_start_utc": clock[0].isoformat(),
            "coverage_end_utc_exclusive": (
                clock[-1] + pd.Timedelta(minutes=1)
            ).isoformat(),
            "interval_semantics": "left_closed_right_open_utc",
            "source_method": "externally_sourced_exact_xau_utc_closure_intervals_v1",
            "source_reference_sha256": "7" * 64,
            "intervals": intervals,
            "test_data_used": False,
        }
    )
    return build_market_closure_authority(
        m1_times=clock,
        m1_source_path=Path("/immutable/val-child.parquet"),
        m1_source_sha256="8" * 64,
        m1_source_manifest_path=Path("/immutable/val-child.manifest.json"),
        m1_source_manifest_sha256="9" * 64,
        exact_schedule=schedule,
        exact_schedule_path=Path("/immutable/val-closure.schedule.json"),
        exact_schedule_file_sha256="a" * 64,
    )


class _Policy(nn.Module):
    unified_exit_random_access_architecture_version = RANDOM_ACCESS_MODEL_SCHEMA_VERSION

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.register_buffer(
            "unified_exit_random_access_architecture_sha256",
            torch.tensor(
                list(bytes.fromhex(RANDOM_ACCESS_MODEL_SCHEMA_SHA256)),
                dtype=torch.uint8,
            ),
        )

    def forward_exit_random_access_batch(self, **inputs):
        state = inputs["state_ctx_cont"][:, 0]
        threshold = inputs["entry_decision_representation"][:, :2]
        hold = torch.where(state[:, None] < threshold, 2.0, 0.0) + self.bias
        exit_now = torch.where(state[:, None] >= threshold, 2.0, 0.0) + self.bias
        q = torch.stack((hold, exit_now), dim=2)
        if inputs.get("liquidation_relative_values", False):
            from gx1.contracts.unified_exit_random_access_model_v1 import liquidation_relative_action_values
            q = liquidation_relative_action_values(q)
        return {
            "exit_action_q_bps": q,
            "exit_action_valid_mask": inputs["action_valid_mask"],
        }


class _EconomicProvider:
    def __init__(self, clock, closure_sha, objective, manifest):
        self.clock = clock
        self.market_closure_authority_sha256 = closure_sha
        self.state_m1_source_sha256 = "8" * 64
        self.state_m1_source_manifest_sha256 = "a" * 64
        self.parent_m1_source_sha256 = "b" * 64
        self.parent_m1_source_manifest_sha256 = "c" * 64
        self.parent_m1_row_offset = 100
        self.objective = objective
        self.manifest = manifest

    def __call__(self, entry, side, action, start, stop):
        row = 479 + start
        start_ns = int(self.clock.asi8[row] + 60_000_000_000)
        end_ns = (
            int(self.clock.asi8[row + 1] + 60_000_000_000)
            if action == "hold"
            else start_ns
        )
        elapsed = (end_ns - start_ns) // 1_000_000_000
        classification = (
            "instantaneous_execution"
            if elapsed == 0
            else "continuous_m1"
            if elapsed == 60
            else "declared_market_closure"
        )

        def component(value, source="b" * 64):
            return {
                "status": "COMPLETE",
                "value_bps": float(value),
                "source_artifact_sha256": source,
            }

        step = {
            "schema_version": economics.ECONOMIC_STEP_SCHEMA_VERSION,
            "event_kind": "HOLD" if action == "hold" else "EXIT_NOW",
            "interval_start_time_ns": start_ns,
            "interval_end_time_ns": end_ns,
            "gross_price_cashflow": component(0.0 if action == "hold" else getattr(self, "exit_gross_override", 10.0 + side)),
            "commission": component(0.0 if action == "hold" else 1.0),
            "execution_slippage": component(0.0),
            "financing_or_swap": component(-0.1 if action == "hold" else 0.0),
            "guaranteed_execution_fee": component(0.0),
            "risk_utility_penalty": component(0.2 if action == "hold" else 0.0),
            "same_capital_hurdle_running_cost": component(
                0.0,
                self.objective["capital_hurdle_artifact_sha256"],
            ),
            "gap": {
                "status": "COMPLETE",
                "classification": classification,
                "source_manifest_sha256": "c" * 64,
                "classification_artifact_sha256": "d" * 64,
            },
        }
        if self.objective.get("reward_accounting") in {economics.MARK_TO_MARKET_REWARD_ACCOUNTING, economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING}:
            step["schema_version"] = economics.MARK_TO_MARKET_STEP_SCHEMA_VERSION
            step["successor_liquidation_value"] = component(100.0 * entry + side if action == "hold" else 0.0)
        result = {
            "schema_version": "gx1_unified_exit_economic_step_slice_v1",
            "entry_row_index": entry,
            "side_index": side,
            "action": action,
            "start_state_index": start,
            "stop_state_index": stop,
            "steps": [step],
            "economic_step_model_sha256": self.manifest["economic_step_model_sha256"],
            "economic_step_source_manifest_sha256": self.manifest[
                "economic_step_source_manifest_sha256"
            ],
        }
        result["slice_sha256"] = _sha(result)
        return result


def _state_provider(entry, state_index):
    path_length = min(state_index + 1, 512)
    path_start = state_index + 1 - path_length
    mtf = {}
    for tf in MULTI_TF_TIMEFRAMES:
        suffix = tf.lower()
        mtf[f"exit_mtf_history_{suffix}"] = _readonly([[0.0]], "<f4")
        mtf[f"exit_mtf_history_time_ns_{suffix}"] = _readonly([1], "<i8")
        mtf[f"exit_mtf_gather_{suffix}"] = _readonly([0], "<i8")
    row = entry["entry_m1_start_row"] + state_index
    return {
        "state_index": state_index,
        "m1_row_index": row,
        "bar_start_time_ns": 0,
        "decision_time_ns": 0,
        "m1_local_history_start_row": row - 479,
        "m1_local_history_x": _readonly(np.zeros((480, 1)), "<f4"),
        "state_ctx_cont": _readonly([state_index], "<f4"),
        "state_ctx_cat": _readonly([0], "<i8"),
        "trade_path_start_state_index": path_start,
        "trade_path_length": path_length,
        "trade_path_tail_x": _readonly(np.zeros((2, path_length, 1)), "<f4"),
        "lifetime_summary_x": _readonly(np.zeros((2, 7)), "<f8"),
        "lifetime_summary_sha256_by_side": ["e" * 64, "f" * 64],
        "mtf": mtf,
    }


def _fixture(*, thresholds, counts, gap=None, max_forwards=1_000, reward_accounting="terminal_cash_v2", evaluation_cohort=None):
    tail_rows = max(counts)
    clock = _clock(gap=gap, tail_rows=tail_rows)
    closure = _closure(clock, gap=gap)
    objective = _objective(reward_accounting)
    normalization = _normalization()
    model = _Policy().eval()
    representations = torch.as_tensor(thresholds, dtype=torch.float32)
    entries = [
        {
            "entry_row_index": index,
            "entry_m1_start_row": 479,
            "available_state_count": int(counts[index]),
            "entry_episode_binding_sha256": "1" * 64,
            "entry_fill_binding_sha256": "2" * 64,
        }
        for index in range(VAL_ENTRY_COHORT_SIZE)
    ]
    manifest = {
        "manifest_sha256": "3" * 64,
        "economic_step_model_sha256": "4" * 64,
        "economic_step_source_manifest_sha256": "5" * 64,
    }
    if evaluation_cohort is not None:
        ids = evaluation_cohort["entry_row_indices"]
        entries = [entries[i] for i in ids]
        representations = representations[ids]
    contract = build_random_access_val_rollout_contract(
        evaluation_cohort=evaluation_cohort,
        entries=entries,
        entry_decision_representations=representations,
        source_lineage_sha256="6" * 64,
        m1_source_sha256="8" * 64,
        m1_clock_sha256_value=m1_clock_sha256(clock),
        m1_source_manifest_file_sha256="a" * 64,
        parent_m1_source_sha256="b" * 64,
        parent_m1_source_manifest_sha256="c" * 64,
        parent_m1_row_offset=100,
        market_closure_authority_sha256=closure["artifact_sha256"],
        market_closure_authority_file_sha256="7" * 64,
        economic_step_manifest_sha256=manifest["manifest_sha256"],
        economics_objective_contract_sha256=objective["contract_sha256"],
        normalization_artifact=normalization,
        normalization_file_sha256="8" * 64,
        model_state_sha256=canonical_model_state_sha256(model.state_dict()),
        checkpoint_file_sha256="9" * 64,
        compute_guard_max_model_forwards=max_forwards,
        compute_guard_max_materialized_state_views=int(np.sum(counts)) + 100,
        compute_guard_max_wall_seconds=600.0,
    )

    def state_provider(entry, state_index):
        state = _state_provider(entry, state_index)
        row = entry["entry_m1_start_row"] + state_index
        state["bar_start_time_ns"] = int(clock.asi8[row])
        state["decision_time_ns"] = int(clock.asi8[row] + 60_000_000_000)
        return state

    provider = _EconomicProvider(
        clock,
        closure["artifact_sha256"],
        objective,
        manifest,
    )
    adapter = RandomAccessValRolloutAdapterV1(
        contract=contract,
        entries=entries,
        m1_times=clock,
        market_closure_authority=closure,
        economic_step_provider=provider,
        economic_step_manifest=manifest,
        economics_objective_contract=objective,
        normalization_artifact=normalization,
        state_provider=state_provider,
    )
    return model, representations, adapter, contract


def test_full_5508_entry_both_side_rollout_compacts_and_exits() -> None:
    thresholds = np.zeros((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    thresholds[1_000:2_000] = 1
    thresholds[2_000:] = 2
    counts = np.full(VAL_ENTRY_COHORT_SIZE, 3, dtype=np.int64)
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds,
        counts=counts,
    )
    result = run_random_access_val_rollout(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
    )
    assert require_random_access_val_rollout_result(result, contract=contract) == result
    assert result["decision"] == "PASS_COMPLETE"
    assert result["exited_side_trade_count"] == 11_016
    assert [row["active_entry_count"] for row in result["compaction_trace"]] == [
        5_508,
        4_508,
        3_508,
    ]
    assert result["economic_terminal_count"] == 0
    assert result["capacity_or_512_terminal_count"] == 0


def test_state_513_is_decided_and_512_never_becomes_terminal() -> None:
    thresholds = np.zeros((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.full(VAL_ENTRY_COHORT_SIZE, 514, dtype=np.int64)
    thresholds[0, 0] = 513
    model, representations, adapter, _contract = _fixture(
        thresholds=thresholds,
        counts=counts,
        max_forwards=520,
    )
    result = run_random_access_val_rollout(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
    )
    long_first = result["trade_outcomes"][0]
    assert long_first["status"] == "EXITED"
    assert long_first["exit_state_index"] == 513
    assert result["max_decision_state_index"] == 513
    assert result["capacity_or_512_terminal_count"] == 0


def test_weekend_hold_uses_wall_clock_economics_and_then_exits() -> None:
    thresholds = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.full(VAL_ENTRY_COHORT_SIZE, 2, dtype=np.int64)
    model, representations, adapter, _contract = _fixture(
        thresholds=thresholds,
        counts=counts,
        gap="weekend",
    )
    result = run_random_access_val_rollout(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
    )
    assert result["decision"] == "PASS_COMPLETE"
    assert result["trade_outcomes"][0]["hold_wall_clock_seconds"] > 60
    assert result["trade_outcomes"][0][
        "undiscounted_net_cash_pnl_bps"
    ] == pytest.approx(8.9)


@pytest.mark.parametrize(
    ("gap", "expected_status"),
    (
        (None, "RIGHT_CENSORED_SPLIT_END"),
        ("unknown", "RIGHT_CENSORED_UNKNOWN_SOURCE_GAP"),
    ),
)
def test_hold_at_observation_end_is_censored_never_terminal(
    gap, expected_status
) -> None:
    thresholds = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.ones(VAL_ENTRY_COHORT_SIZE, dtype=np.int64)
    model, representations, adapter, _contract = _fixture(
        thresholds=thresholds,
        counts=counts,
        gap=gap,
    )
    result = run_random_access_val_rollout(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
    )
    assert result["decision"] == "COMPLETE_WITH_RIGHT_CENSORING"
    assert result["trade_outcomes"][0]["status"] == expected_status
    assert result["full_cohort_policy_metrics_authoritative"] is False
    assert result["economic_terminal_count"] == 0


def test_compute_guard_marks_truncation_and_contract_tamper_fails() -> None:
    thresholds = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.full(VAL_ENTRY_COHORT_SIZE, 2, dtype=np.int64)
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds,
        counts=counts,
        max_forwards=1,
    )
    result = run_random_access_val_rollout(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
    )
    assert result["decision"] == "TRUNCATED_NON_AUTHORITATIVE"
    assert result["compute_truncated_side_trade_count"] == 11_016
    assert result["economic_terminal_count"] == 0
    assert result["compute_guard_triggered"] == "max_model_forwards"
    bad = copy.deepcopy(contract)
    bad["capacity_or_512_is_terminal"] = True
    with pytest.raises(RuntimeError, match="VAL_CONTRACT_INVALID"):
        require_random_access_val_rollout_contract(bad)


def _production_factory_fixture(state_count=2):
    from gx1.contracts.unified_exit_dataset_adapter_v2 import _RangeExtrema
    from gx1.contracts.unified_exit_random_access_val_factory_v1 import (
        RandomAccessValStateFactoryV1,
    )
    from gx1.models.entry_v10.direction_decision_contract import (
        UNIFIED_EXIT_PATH_PRICE_FIELDS,
    )

    thresholds = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    counts = np.full(VAL_ENTRY_COHORT_SIZE, state_count, dtype=np.int64)
    model, representations, adapter, _contract = _fixture(
        thresholds=thresholds,
        counts=counts,
    )
    factory = RandomAccessValStateFactoryV1.__new__(RandomAccessValStateFactoryV1)
    factory.entries = adapter.entries
    factory.starts = np.full(VAL_ENTRY_COHORT_SIZE, 479, dtype=np.int64)
    factory.times = adapter.times
    size = len(factory.times)
    base = np.linspace(100.0, 100.2, size, dtype=np.float64)
    factory.prices = {
        name: _readonly(
            base
            + (
                0.02
                if name.startswith("ask_")
                else 0.0
                if name.startswith("bid_")
                else 0.01
            ),
            "<f8",
        )
        for name in UNIFIED_EXIT_PATH_PRICE_FIELDS
    }
    factory.prices["volume"] = _readonly(np.ones(size), "<f8")
    factory.ranges = {
        "bid_high": _RangeExtrema(factory.prices["bid_high"], maximum=True),
        "bid_low": _RangeExtrema(factory.prices["bid_low"], maximum=False),
        "ask_low": _RangeExtrema(factory.prices["ask_low"], maximum=False),
        "ask_high": _RangeExtrema(factory.prices["ask_high"], maximum=True),
    }
    factory.signal = _readonly(np.zeros((size, 1)), "<f4")
    factory.ctx_cont = _readonly(np.maximum(np.arange(size) - 479, 0)[:, None], "<f4")
    factory.ctx_cat = _readonly(np.zeros((size, 1)), "<i8")

    def mtf_materializer(_times):
        result = {}
        for tf in MULTI_TF_TIMEFRAMES:
            suffix = tf.lower()
            result[f"exit_mtf_history_{suffix}"] = np.zeros((1, 1), dtype=np.float32)
            result[f"exit_mtf_history_time_ns_{suffix}"] = np.ones(1, dtype=np.int64)
            result[f"exit_mtf_gather_{suffix}"] = np.zeros(1, dtype=np.int64)
        return result

    factory.mtf_materializer = mtf_materializer
    factory.mtf_feature_dims = {tf: 1 for tf in MULTI_TF_TIMEFRAMES}
    adapter.state_provider = factory.materialize_state
    return factory, adapter, model, representations


def test_production_state_factory_feeds_full_cohort_learned_exit_rollout() -> None:
    factory, adapter, model, representations = _production_factory_fixture()
    state_zero = factory.materialize_state(factory.entries[0], 0)
    assert state_zero["m1_local_history_x"].shape[0] == 480
    assert state_zero["trade_path_length"] == 1
    result = run_random_access_val_rollout(
        model=model,
        entry_decision_representations=representations,
        adapter=adapter,
    )
    assert result["decision"] == "PASS_COMPLETE"
    assert result["exited_side_trade_count"] == 11_016
    assert result["max_decision_state_index"] == 1


@pytest.mark.parametrize("workers", [4, 8])
def test_compact_market_cpu_workers_preserve_states_across_512_boundary(workers):
    factory, adapter, _model, _representations = _production_factory_fixture(520)
    try:
        for index in (0, 1, 511, 512, 513):
            entries = list(range(8))
            baseline = adapter.materialize_active_batch(entries, index)
            cached = adapter.materialize_cached_active_batch(
                entries, index, cached_market_rows={479 + index}, workers=workers,
            )
            for before, after in zip(baseline, cached):
                for name, value in before["state"].items():
                    if name in {"m1_local_history_x", "state_ctx_cat", "state_ctx_cont", "mtf"}:
                        assert after["state"][name] is None
                    elif isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, after["state"][name])
                        assert not after["state"][name].flags.writeable
                    else:
                        assert after["state"][name] == value
                for name in ("successor_observed", "right_censor_reason_if_hold", "transition_closure"):
                    assert before[name] == after[name]
        assert len(factory._val_cpu_pool._processes) == workers
    finally:
        factory.close_val_cpu_workers()



def test_marked_hold_cache_is_bounded_by_intervals_not_entry_prices():
    _, _, adapter, _ = _fixture(
        thresholds=np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32),
        counts=np.full(VAL_ENTRY_COHORT_SIZE, 3, dtype=np.int64),
        reward_accounting=economics.MARK_TO_MARKET_REWARD_ACCOUNTING,
    )
    provider = adapter.economic_step_provider
    provider.materialize_selected_actions = lambda requests: [
        provider(r["entry_row_index"], r["side_index"], r["action"], r["state_index"], r["state_index"] + 1)
        for r in requests]
    requests = [dict(entry_row_index=entry, side_index=side, state_index=state, action="hold")
                for entry in range(100) for side in (0, 1) for state in (0, 1)]
    reference = [adapter.compose_selected_action(**r) for r in requests]
    assert len({x[0]["undiscounted_risk_adjusted_utility_increment_bps"] for x in reference}) == 200
    for batch in (requests, requests[::-1], requests):
        observed = adapter.compose_selected_actions(batch)
        assert observed == (reference if batch is requests else reference[::-1])
        # Both sides in this fixture have identical interval cash costs.
        assert len(adapter._val_composed_hold_cache) == 2


def _train_cutoff_fixture(monkeypatch, *, gap=None, threshold=100.0, max_forwards=1000):
    from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts
    clock = _clock(gap=gap, tail_rows=4)
    # The clock fixture is synthetic; file-binding validation is exercised in
    # test_chronological_measurement_binding with real bound parquet/npz inputs.
    cutoff = int(clock.asi8[479 if gap else 480]) + 60_000_000_000
    scope = {"schema_version": cohorts.TRAIN_ROLLOUT_SCHEMA, "split": "train",
        "source_split": "train", "evaluation_role": "chronological_training_policy_rollout",
        "measurement_only": False, "plan": {}, "measurement_coordinates": {},
        "source_index": {"path": "/synthetic/index", "sha256": "0" * 64},
        "population_rows": VAL_ENTRY_COHORT_SIZE, "entry_row_indices": [0, 1],
        "parent_entry_row_indices": [0, 1], "observation_cutoff_time_ns": cutoff,
        "test_data_used": False}
    scope["cohort_sha256"] = cohorts.canonical_sha256(scope)
    monkeypatch.setattr(cohorts, "build_chronological_train_rollout_cohort", lambda *args: dict(scope))
    thresholds = np.full((VAL_ENTRY_COHORT_SIZE, 2), threshold, dtype=np.float32)
    return _fixture(thresholds=thresholds,
        counts=np.full(VAL_ENTRY_COHORT_SIZE, 2 if gap else 4, dtype=np.int64),
        gap=gap, max_forwards=max_forwards,
        reward_accounting=economics.MARK_TO_MARKET_REWARD_ACCOUNTING, evaluation_cohort=scope)


@pytest.mark.parametrize("gap", [None, "weekend"])
def test_train_cutoff_blocks_future_state_and_economic_reads(monkeypatch, gap):
    _, _, adapter, _ = _train_cutoff_fixture(monkeypatch, gap=gap)
    boundary = 0 if gap else 1
    state = adapter.materialize_state(0, boundary)
    assert state["right_censor_reason_if_hold"] == "observation_cutoff"
    assert state["successor_observed"] is False and state["economic_terminal"] is False
    assert adapter.entries[0]["available_state_count"] == (2 if gap else 4)
    adapter.compose_selected_action(entry_row_index=0, side_index=0, state_index=boundary, action="exit_now")
    def forbidden(*args, **kwargs):
        pytest.fail("provider read beyond the observation boundary")
    adapter.state_provider = forbidden
    adapter.economic_step_provider = forbidden
    calls = [lambda: adapter.materialize_state(0, boundary + 1),
        lambda: adapter.materialize_active_batch([0, 1], boundary + 1),
        lambda: adapter.materialize_cached_active_batch([0, 1], boundary + 1,
            cached_market_rows={480, 481}, workers=8),
        lambda: adapter.compose_selected_action(entry_row_index=0, side_index=0,
            state_index=boundary, action="hold"),
        lambda: adapter.compose_selected_actions([{"entry_row_index":0,"side_index":1,
            "state_index":boundary,"action":"hold"}])]
    for call in calls:
        with pytest.raises(RuntimeError, match="OBSERVATION_CUTOFF_EXCEEDED"):
            call()
