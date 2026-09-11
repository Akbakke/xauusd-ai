from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_dataset_adapter_v2 import (
    ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
    seal_economic_training_projection,
)
from gx1.contracts.unified_exit_economics_objective_v2 import (
    CAPITAL_HURDLE_SCHEMA_VERSION,
    SECONDS_PER_YEAR,
    build_unified_exit_economics_objective_contract,
    seal_train_fitted_capital_hurdle_artifact,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import build_lifetime_summary
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
    build_market_closure_authority,
    seal_exact_market_schedule,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    DURATION_BUCKETS,
    RANDOM_ACCESS_SAMPLE_SCHEMA_VERSION,
    build_random_access_sampler_contract,
    canonical_sha256,
    duration_bucket_for_state,
    schedule_random_access_entry_anchors,
)
from gx1.contracts.unified_exit_random_access_state_view_v1 import (
    materialize_random_access_state_view,
    require_random_access_state_view,
    validate_random_access_m1_source_v1,
)
from gx1.contracts.entry_exit_feature_base_v1 import EXIT_MTF_CONTEXT_TIMEFRAMES


def _clock() -> pd.DatetimeIndex:
    values = pd.date_range("2026-07-01T00:00Z", periods=1_200, freq="min")
    shifted = values.to_series(index=range(len(values)))
    shifted.loc[484:] += pd.Timedelta(days=2)
    return pd.DatetimeIndex(shifted.array)


def _authority(clock: pd.DatetimeIndex, *, known: bool) -> dict:
    intervals = []
    if known:
        intervals.append(
            {
                "kind": "weekend",
                "start_utc": (clock[483] + pd.Timedelta(minutes=1)).isoformat(),
                "end_utc_exclusive": clock[484].isoformat(),
                "source_event_id": "project-clock-weekend-fixture",
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
            "source_reference_sha256": "1" * 64,
            "intervals": intervals,
            "test_data_used": False,
        }
    )
    return build_market_closure_authority(
        m1_times=clock,
        m1_source_path=Path("/immutable/pretest.parquet"),
        m1_source_sha256="2" * 64,
        m1_source_manifest_path=Path("/immutable/pretest.manifest.json"),
        m1_source_manifest_sha256="3" * 64,
        exact_schedule=schedule,
        exact_schedule_path=Path("/immutable/schedule.json"),
        exact_schedule_file_sha256="4" * 64,
    )


def _contract() -> dict:
    return build_random_access_sampler_contract(
        split="train",
        source_lineage_sha256="5" * 64,
        transition_budget_per_epoch=1,
        transitions_per_entry=1,
        entry_pair_population=1,
    )


def _sample(contract: dict, state_index: int) -> dict:
    bucket = duration_bucket_for_state(state_index)
    start, end = DURATION_BUCKETS[bucket]
    stop = min(end or 600, 600)
    sample = {
        "schema_version": RANDOM_ACCESS_SAMPLE_SCHEMA_VERSION,
        "sampler_contract_sha256": contract["contract_sha256"],
        "epoch_index": 0,
        "entry_slot": 0,
        "entry_row_index": 0,
        "sample_slot": 0,
        "duration_bucket_index": bucket,
        "duration_bucket_start_inclusive": start,
        "duration_bucket_stop_exclusive": stop,
        "eligible_transition_count_in_bucket": stop - start,
        "eligible_duration_bucket_count": 6,
        "state_index": state_index,
        "successor_state_index": state_index + 1,
        "sampling_probability": 1.0 / 6.0 / (stop - start),
        "importance_weight": 1.0,
        "sample_role": "bellman_transition",
        "both_sides_share_timeline": True,
        "selection_uses_outcome_values": False,
    }
    sample["sample_sha256"] = canonical_sha256(sample)
    return sample


def _objective() -> dict:
    hurdle = seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": "6" * 64,
            "train_fold_sha256": "7" * 64,
            "source_lineage_sha256": "8" * 64,
            "annual_continuous_hurdle_rate": 0.1,
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": SECONDS_PER_YEAR,
            "fit_method": "fixture_train_only",
            "fit_evidence_sha256": "9" * 64,
        }
    )
    return build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256="6" * 64,
        expected_train_fold_sha256="7" * 64,
        expected_source_lineage_sha256="8" * 64,
        policy_sha256="a" * 64,
    )


class _EconomicProvider:
    def __init__(self, closure_sha: str) -> None:
        self.market_closure_authority_sha256 = closure_sha

    def materialize_training_projection(self, entry, side, start, stop, hold_stop):
        return seal_economic_training_projection(
            {
                "schema_version": ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
                "entry_row_index": entry,
                "side_index": side,
                "start_state_index": start,
                "stop_state_index": stop,
                "hold_stop_state_index": hold_stop,
                "exit_event_kind_index": np.zeros(stop - start, dtype="u1"),
                "exit_reward_bps": np.full(stop - start, 2.0 + side, dtype="<f8"),
                "hold_event_kind_index": np.ones(hold_stop - start, dtype="u1"),
                "hold_reward_bps": np.full(hold_stop - start, -0.1, dtype="<f8"),
                "economic_step_model_sha256": "b" * 64,
                "economic_step_source_manifest_sha256": "c" * 64,
            }
        )


def _materialize(
    state_index: int,
    *,
    known_gap: bool = True,
    counts: tuple[int, int] = (600, 600),
    terminals: tuple[bool, bool] = (False, False),
    anchor: bool = False,
    prevalidate: bool = False,
) -> dict:
    clock = _clock()
    contract = _contract()
    signal = np.arange(len(clock) * 3, dtype=np.float32).reshape(len(clock), 3)
    cont = np.ones((len(clock), 2), dtype=np.float32)
    cat = np.ones((len(clock), 1), dtype=np.int64)

    def path(side: int, start: int, stop: int) -> np.ndarray:
        states = np.arange(start, stop, dtype=np.float32)
        return np.stack((states, np.full_like(states, side)), axis=1)

    def summary(side: int, index: int) -> dict:
        return build_lifetime_summary(
            side=("long", "short")[side],
            bars_in_trade=index + 1,
            elapsed_wall_clock_seconds=(index + 1) * 60,
            current_executable_pnl_bps=0.0,
            cum_mfe_bps=1.0,
            cum_mae_bps=-1.0,
            bars_since_mfe_peak=index,
        )

    manifest = {
        "manifest_sha256": "d" * 64,
        "economic_step_model_sha256": "b" * 64,
        "economic_step_source_manifest_sha256": "c" * 64,
    }
    authority = _authority(clock, known=known_gap)

    def mtf(_times: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for tf in EXIT_MTF_CONTEXT_TIMEFRAMES:
            suffix = tf.lower()
            out[f"exit_mtf_history_{suffix}"] = np.ones((3, 2), dtype=np.float32)
            out[f"exit_mtf_history_time_ns_{suffix}"] = np.arange(3, dtype=np.int64)
            out[f"exit_mtf_gather_{suffix}"] = np.asarray([2], dtype=np.int64)
        return out

    prevalidated = None
    if prevalidate:
        prevalidated = validate_random_access_m1_source_v1(
            m1_times=clock,
            m1_signal=signal,
            m1_ctx_cont=cont,
            m1_ctx_cat=cat,
            m1_source_sha256="2" * 64,
            market_closure_authority=authority,
        )
        clock = prevalidated["times"]
        signal = prevalidated["signal"]
        cont = prevalidated["cont"]
        cat = prevalidated["cat"]
        authority = prevalidated["market_closure_authority"]

    sample = (
        schedule_random_access_entry_anchors(sampler_contract=contract, epoch_index=0)[
            0
        ]
        if anchor
        else _sample(contract, state_index)
    )
    view = materialize_random_access_state_view(
        sampler_contract=contract,
        sample=sample,
        entry_row_index=0,
        entry_m1_start_row=479,
        side_lifecycle_state_counts=counts,
        side_economic_terminal=terminals,
        m1_times=clock,
        m1_signal=signal,
        m1_ctx_cont=cont,
        m1_ctx_cat=cat,
        m1_source_sha256="2" * 64,
        market_closure_authority=authority,
        path_detail_provider=path,
        lifetime_summary_provider=summary,
        mtf_materializer=mtf,
        economic_step_provider=_EconomicProvider(authority["artifact_sha256"]),
        economic_step_manifest=manifest,
        economics_objective_contract=_objective(),
        prevalidated_m1_source=prevalidated,
    )
    require_random_access_state_view(
        view,
        sampler_contract=contract,
        sample=sample,
        expected_m1_source_sha256="2" * 64,
        expected_market_closure_authority_sha256=authority["artifact_sha256"],
        expected_economic_step_manifest_sha256="d" * 64,
        expected_economics_objective_contract_sha256=_objective()["contract_sha256"],
    )
    return view


def test_first_transition_has_480_local_rows_and_growing_trade_path() -> None:
    view = _materialize(0)
    assert view["current"]["m1_local_history_x"].shape == (480, 3)
    assert view["successor"]["m1_local_history_x"].shape == (480, 3)
    assert view["current"]["trade_path_tail_x"].shape == (2, 1, 2)
    assert view["successor"]["trade_path_tail_x"].shape == (2, 2, 2)
    assert view["terminal_mask"].tolist() == [False, False]
    assert view["capacity_or_tail_length_is_terminal"] is False


def test_entry_anchor_materializes_state_zero_outside_loss_budget() -> None:
    view = _materialize(0, anchor=True)
    assert view["current"]["state_index"] == 0
    assert view["sample_role"] == "entry_anchor_no_loss"
    assert view["loss_weight"] == 0.0


def test_deep_transition_uses_rolling_512_tail_and_exact_successor() -> None:
    view = _materialize(530)
    assert view["current"]["trade_path_tail_x"].shape == (2, 512, 2)
    assert view["successor"]["trade_path_tail_x"].shape == (2, 512, 2)
    assert view["current"]["state_index"] == 530
    assert view["successor"]["state_index"] == 531
    assert np.allclose(
        view["immediate_reward_bps"], [[-0.1, 2.0], [-0.1, 3.0]], rtol=0.0, atol=1e-7
    )
    assert view["policy_action_valid_mask"].all()
    assert view["bellman_target_valid_mask"].all()
    assert view["successor_observed_mask"].all()


def test_successor_terminal_masks_hold_before_target_max() -> None:
    view = _materialize(530, counts=(532, 600), terminals=(True, False))
    assert view["successor_terminal_mask"].tolist() == [True, False]
    assert view["successor_policy_action_valid_mask"].tolist() == [
        [False, True],
        [True, True],
    ]


def test_declared_closure_preserves_successor_but_unknown_gap_censors() -> None:
    allowed = _materialize(4)
    assert allowed["transition_closure"]["closure_kind"] == "weekend"
    assert allowed["transition_closure"]["wall_clock_delta_seconds"] > 48 * 60 * 60
    assert 0.0 < allowed["elapsed_wall_clock_gamma"] < 1.0
    with pytest.raises(RuntimeError, match="SUCCESSOR_GAP_CENSORED"):
        _materialize(4, known_gap=False)


def test_split_end_or_side_shorter_than_successor_fails_closed() -> None:
    clock = _clock()
    contract = _contract()
    with pytest.raises(RuntimeError, match="COMMON_TIMELINE_INVALID"):
        materialize_random_access_state_view(
            sampler_contract=contract,
            sample=_sample(contract, 4),
            entry_row_index=0,
            entry_m1_start_row=479,
            side_lifecycle_state_counts=(5, 600),
            side_economic_terminal=(False, False),
            m1_times=clock,
            m1_signal=np.ones((len(clock), 1), dtype=np.float32),
            m1_ctx_cont=np.ones((len(clock), 1), dtype=np.float32),
            m1_ctx_cat=np.ones((len(clock), 1), dtype=np.int64),
            m1_source_sha256="2" * 64,
            market_closure_authority=_authority(clock, known=True),
            path_detail_provider=lambda *_: np.ones((1, 1), dtype=np.float32),
            lifetime_summary_provider=lambda *_: {},
            mtf_materializer=lambda *_: {},
            economic_step_provider=_EconomicProvider("0" * 64),
            economic_step_manifest={},
            economics_objective_contract={},
        )


def test_state_view_hash_tamper_fails_closed() -> None:
    view = _materialize(0)
    contract = _contract()
    sample = _sample(contract, 0)
    view["current"]["trade_path_length"] = 2
    with pytest.raises(RuntimeError, match="STATE_VIEW_INVALID"):
        require_random_access_state_view(
            view,
            sampler_contract=contract,
            sample=sample,
            expected_m1_source_sha256="2" * 64,
            expected_market_closure_authority_sha256=view[
                "market_closure_authority_sha256"
            ],
            expected_economic_step_manifest_sha256="d" * 64,
            expected_economics_objective_contract_sha256=_objective()[
                "contract_sha256"
            ],
        )


def test_prevalidated_source_preserves_exact_state_view_bytes():
    regular = _materialize(530)
    fast = _materialize(530, prevalidate=True)
    assert fast["state_view_sha256"] == regular["state_view_sha256"]
    assert np.array_equal(
        fast["current"]["m1_local_history_x"],
        regular["current"]["m1_local_history_x"],
    )
    assert np.array_equal(
        fast["successor"]["trade_path_tail_x"],
        regular["successor"]["trade_path_tail_x"],
    )
