from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tests.model_native_sizing_support import unverified_learned_sizing_authority
from tests.model_native_offline_rl_support import (
    model_native_mtf_cooperation_evidence,
    offline_rl_evidence,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
)
from gx1.execution.v12_trade_state import (
    CLOSED_M1_PATH_SCHEMA_VERSION,
    M1_RETURNS_WINDOW_MAXLEN,
    PERSISTED_TRADE_STATE_SCHEMA_VERSION,
    TRAJECTORY_HISTORY_MAXLEN,
    TradeState,
    build_trade_broker_account_binding,
    first_full_closed_m1_bar_ts,
    require_model_native_entry_snapshot,
    require_trade_broker_account_binding,
    TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION,
    TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION,
    TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION,
)
from gx1.execution.v12_pipeline import V12Pipeline
from gx1.execution.v12_paper_runner import (
    journal_v12_exit_decision,
    persist_and_journal_v12_exit_bar,
)
from gx1.execution.v12_smart_entry_live import SmartEntryLiveInference
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    canonical_unified_evidence_sha256,
    unified_entry_exit_contract_metadata,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
)
from gx1.contracts.entry_exit_feature_base_v1 import EXIT_FEATURE_SEQUENCE_BARS
from gx1.monitoring.trade_journal import TradeJournal
def _softmax(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    exp = np.exp(array - array.max())
    return (exp / exp.sum()).tolist()


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _logit(value: float) -> float:
    return float(np.log(value / (1.0 - value)))


def _snapshot() -> dict:
    direction_logits = [5.0, 1.0, 0.0]
    public_logits = [5.0, 0.0]
    side_logits = [1.0, -0.5]
    side_bad_logits = [-1.2, 0.7]
    side_validity_logits = [1.4, -0.3]
    mtf_logits = [0.8, -0.2, -0.6]
    rail_logits = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    tf_logit = 0.5
    size_logit = 0.25
    return {
        "decision_ts": "2026-07-16T11:55:00+00:00",
        "runtime_evidence_schema_version": MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "model_policy": MODEL_NATIVE_RUNTIME_POLICY,
        "session_id": 2,
        "session": "OVERLAP",
        "entry_vol_regime_id": 2,
        "entry_vol_regime": "MEDIUM",
        "entry_atr_bucket": 2,
        "entry_spread_bucket": 1,
        "entry_h4_trend_sign_cat": 2,
        "entry_trend_regime_id": 1,
        "entry_trend_regime": "TREND_NEUTRAL",
        "decision_available_ts": "2026-07-16T12:00:00+00:00",
        "entry_signal_latency_sec": 0.0,
        "context_cutoff_ts": "2026-07-16T11:55:00+00:00",
        "context_age_m5_bars": 0,
        "raw_direction_logits": [5.5, 1.1, 0.0],
        "direction_logits": direction_logits,
        "direction_probs": _softmax(direction_logits),
        "model_direction_index": 0,
        "model_direction": "LONG",
        "entry_shared_representation": [
            float(index - 64) / 64.0 for index in range(128)
        ],
        "public_trade_flat_decision_logits": public_logits,
        "public_trade_flat_decision_probs": _softmax(public_logits),
        "public_trade_flat_decision_index": 0,
        "public_trade_flat_decision": "TRADE",
        "selected_side": 0,
        "model_native_logits": [0.5, -0.25, 0.1],
        "path_quality_raw": 1.5,
        "path_quality": 1.5,
        "path_quality_pred": 1.5,
        "mfe_first_n": 12.0,
        "mfe_first_n_pred": 12.0,
        "tradable_logit": 1.0,
        "tradable_prob": _sigmoid(1.0),
        "trade_logit": 0.7,
        "bad_path_logit_raw": -1.0,
        "bad_path_logit": -1.0,
        "bad_path_prob": _sigmoid(-1.0),
        "clean_edge_logit": _logit(0.76),
        "clean_edge_prob": 0.76,
        "survival_logit": _logit(0.68),
        "survival_prob": 0.68,
        "dip_pred": [0.0] * 18,
        "forecast_pred": [0.0] * 4,
        "timing_pred": [0.0] * 12,
        "tail_risk_pred": [0.0] * 6,
        "vol_forecast_pred": [0.0] * 3,
        **offline_rl_evidence(),
        "p_trade": _softmax(public_logits)[0],
        "p_flat_hier": _softmax(public_logits)[1],
        "atr_bps": 9.0,
        "tf_agreement_logit": tf_logit,
        "tf_agreement_pred": _sigmoid(tf_logit),
        "path_quality_log_var": 0.0,
        "path_quality_std": 1.0,
        "position_size_logit": size_logit,
        "position_size_pred": _sigmoid(size_logit),
        "p_long_given_trade": _softmax(side_logits)[0],
        "p_short_given_trade": _softmax(side_logits)[1],
        "side_logits": side_logits,
        "side_probs": _softmax(side_logits),
        "side_utility": [2.4, -0.8],
        "side_bad_path_logit": side_bad_logits,
        "long_bad_path_prob": _sigmoid(side_bad_logits[0]),
        "short_bad_path_prob": _sigmoid(side_bad_logits[1]),
        "side_validity_logit": side_validity_logits,
        "long_validity_prob": _sigmoid(side_validity_logits[0]),
        "short_validity_prob": _sigmoid(side_validity_logits[1]),
        "side_mae": [-3.2, -8.1],
        "mtf_dir_logits": mtf_logits,
        "mtf_dir_probs": _softmax(mtf_logits),
        "mtf_trend_evidence": 0.69,
        "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "specialist_gate": [0.125] * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        **model_native_mtf_cooperation_evidence(),
        "trendline_rail_logits": rail_logits,
        "trendline_rail_probs": [_sigmoid(value) for value in rail_logits],
        "geometry_channel_edge_pressure": 0.42,
        "geometry_rising_support_rail_long_pressure": 0.81,
        "geometry_rising_support_rail_short_trap_pressure": 0.07,
        "geometry_falling_resistance_rail_short_pressure": 0.02,
        "geometry_falling_resistance_rail_long_trap_pressure": 0.03,
        "calibration_version": "direction-cal-v1",
        "direction_calibration_enabled": True,
        "direction_calibration_temperature": 1.1,
        "path_calibration_enabled": True,
        "path_calibration": {
            "enabled": True,
            "version": "path-cal-v1",
            "path_quality_scale": 1.0,
            "path_quality_shift": 0.0,
            "bad_path_temperature": 1.0,
            "bad_path_bias": 0.0,
        },
    }


def _exit_feature_surface() -> dict:
    signal = np.zeros((EXIT_FEATURE_SEQUENCE_BARS, 513), dtype=np.float32)
    return {
        "schema_version": ENTRY_EXIT_FEATURE_SURFACE_SCHEMA_VERSION,
        "decision_time": "2026-07-16T12:01:00+00:00",
        "dataset_run_id": "EXIT_TEST_RUN",
        "feature_base_sha256": "c" * 64,
        "sequence_bars": EXIT_FEATURE_SEQUENCE_BARS,
        "signal": signal,
        "snap": signal[-1].copy(),
        "ctx_cont": np.zeros(142, dtype=np.float32),
        "ctx_cat": np.zeros(5, dtype=np.int64),
    }


def _open(snapshot: dict | None = None) -> TradeState:
    return TradeState.open_unit_normalized_research(
        entry_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
        side="long",
        entry_bid=3300.0,
        entry_ask=3300.2,
        v10_snapshot=_snapshot() if snapshot is None else snapshot,
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )


def _model_bundle_binding() -> dict[str, object]:
    return {
        "schema_version": (
            TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION
        ),
        "bundle_dir": "/immutable/test/model_bundle",
        "bundle_sha256": "b" * 64,
        "operating_point": {
            "selection_score": MODEL_DIRECTION_SELECTION_MODE,
            "max_trades": 1,
        },
    }


def _source_pair_binding() -> dict[str, object]:
    return {
        "schema_version": (
            TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION
        ),
        "pair_generation_id": "c" * 64,
        "pair_manifest_sha256": "d" * 64,
    }


def _head_snapshot() -> dict:
    snapshot = _snapshot()
    for name in (
        "decision_available_ts",
        "entry_signal_latency_sec",
        "context_cutoff_ts",
        "context_age_m5_bars",
    ):
        snapshot.pop(name)
    return {
        "runtime_head_evidence_schema_version": (
            MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
        ),
        **snapshot,
    }


def _valid_closed_m1_bar(
    timestamp: str = "2026-07-16T12:00:00Z",
) -> dict[str, object]:
    return {
        "schema_version": CLOSED_M1_PATH_SCHEMA_VERSION,
        "time": pd.Timestamp(timestamp).isoformat(),
        "complete": True,
        "source_path": "/immutable/test/xau_m1.parquet",
        "source_sha256": "a" * 64,
        "bid_open": 3300.8,
        "bid_high": 3301.5,
        "bid_low": 3299.7,
        "bid_close": 3301.0,
        "ask_open": 3301.0,
        "ask_high": 3301.7,
        "ask_low": 3299.9,
        "ask_close": 3301.2,
        "mid_open": 3300.9,
        "mid_high": 3301.6,
        "mid_low": 3299.8,
        "mid_close": 3301.1,
        "volume": 42,
    }


class _UnifiedExitModel:
    def __init__(self, logits: tuple[float, float]) -> None:
        self.logits = logits
        self.calls: list[dict[str, torch.Tensor]] = []

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "head_exit_action.weight": torch.zeros((2, 1)),
            "head_exit_action.bias": torch.zeros(2),
        }

    def forward_exit_action(
        self,
        **inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.calls.append(inputs)
        logits = torch.tensor([self.logits], dtype=torch.float32)
        return {
            "exit_action_logits": logits,
            "exit_action_probs": torch.softmax(logits, dim=-1),
        }


def _unified_exit_adapter(
    tmp_path: Path,
    *,
    logits: tuple[float, float],
) -> tuple[SmartEntryLiveInference, _UnifiedExitModel]:
    model = _UnifiedExitModel(logits)
    adapter = SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={
            "selection_score": MODEL_DIRECTION_SELECTION_MODE,
            "max_trades": 1,
        },
        _bundle_sha256="b" * 64,
        _model=model,
        _meta={
            "unified_entry_exit_contract": (
                unified_entry_exit_contract_metadata()
            ),
        },
    )
    def _provider(*, decision_time: object, prebuilt_snapshot: object) -> dict:
        del prebuilt_snapshot
        value = _exit_feature_surface()
        value["decision_time"] = pd.Timestamp(decision_time).isoformat()
        return value

    # Unit tests admit an explicit fixture provider. Production binds the
    # hash- and pair-bound parquet provider from model metadata instead.
    adapter._exit_feature_surface_provider = _provider
    return adapter, model


def _bind_test_hold_decision(trade: TradeState) -> None:
    logits = [2.0, -1.0]
    probabilities = _softmax(logits)
    snapshot = trade.require_entry_snapshot()
    path_envelope = trade.build_closed_m1_path_evidence()
    decision = {
        "exit_action_logits": logits,
        "exit_action_probs": probabilities,
        "exit_action_index": 0,
        "action": "HOLD",
        "decision_source": "unified_model",
        "bundle_sha256": "b" * 64,
        "entry_snapshot_sha256": canonical_unified_evidence_sha256(
            snapshot
        ),
        "exit_path_envelope_sha256": (
            canonical_unified_evidence_sha256(path_envelope)
        ),
    }
    decision["output_evidence_sha256"] = (
        canonical_unified_evidence_sha256(decision)
    )
    trade.bind_unified_exit_decision(
        decision,
        expected_bundle_sha256="b" * 64,
    )


def test_unified_exit_uses_frozen_entry_representation_and_exact_path(
    tmp_path: Path,
) -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    adapter, model = _unified_exit_adapter(
        tmp_path,
        logits=(-1.0, 2.0),
    )

    output = adapter.decide_exit(
        entry_snapshot=trade.require_entry_snapshot(),
        exit_path_envelope=trade.build_closed_m1_path_evidence(),
        exit_feature_surface=_exit_feature_surface(),
        entry_bid=trade.entry_bid,
        entry_ask=trade.entry_ask,
        side=trade.side,
    )

    assert output["action"] == "EXIT_NOW"
    assert output["exit_action_index"] == 1
    assert output["decision_source"] == "unified_model"
    assert output["bundle_sha256"] == "b" * 64
    assert len(model.calls) == 1
    assert model.calls[0]["entry_shared_representation"].shape == (1, 128)
    assert model.calls[0]["exit_feature_seq_x"].shape == (1, EXIT_FEATURE_SEQUENCE_BARS, 513)
    assert model.calls[0]["exit_feature_snap_x"].shape == (1, 513)
    assert model.calls[0]["exit_feature_ctx_cat"].shape == (1, 5)
    assert model.calls[0]["exit_feature_ctx_cont"].shape == (1, 142)
    assert model.calls[0]["exit_path_x"].shape == (1, 1, 14)
    assert model.calls[0]["exit_path_lengths"].tolist() == [1]
    assert model.calls[0]["exit_side_index"].tolist() == [0]
    assert model.calls[0]["entry_shared_representation"][0].tolist() == (
        pytest.approx(_snapshot()["entry_shared_representation"])
    )


def test_unified_exit_replay_accepts_exact_pre_sizing_head_snapshot(
    tmp_path: Path,
) -> None:
    trade = _open(_head_snapshot())
    trade.update_bar(**_valid_closed_m1_bar())
    adapter, _model = _unified_exit_adapter(
        tmp_path,
        logits=(-1.0, 2.0),
    )

    output = adapter.decide_exit(
        entry_snapshot=trade.require_entry_snapshot(),
        exit_path_envelope=trade.build_closed_m1_path_evidence(),
        exit_feature_surface=_exit_feature_surface(),
        entry_bid=trade.entry_bid,
        entry_ask=trade.entry_ask,
        side=trade.side,
    )

    assert output["action"] == "EXIT_NOW"
    assert output["entry_snapshot_sha256"] == canonical_unified_evidence_sha256(
        _head_snapshot()
    )


def test_pipeline_commits_one_bar_only_after_same_bundle_exit_decision(
    tmp_path: Path,
) -> None:
    trade = _open()
    adapter, model = _unified_exit_adapter(
        tmp_path,
        logits=(2.0, -1.0),
    )
    expected_bar = _valid_closed_m1_bar()

    class _Loader:
        def refresh_if_changed(self) -> bool:
            return False

        def acquire_serving_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                base28=pd.DataFrame(
                    index=pd.DatetimeIndex(
                        ["2026-07-16T12:00:00Z"]
                    )
                )
            )

        def get_closed_m1_bar(
            self,
            expected_m1: pd.Timestamp,
            *,
            snapshot: SimpleNamespace,
        ) -> dict[str, object]:
            assert expected_m1 in snapshot.base28.index
            assert expected_m1 == pd.Timestamp("2026-07-16T12:00:00Z")
            return dict(expected_bar)

    pipeline = V12Pipeline(
        prebuilt_loader=_Loader(),
        smart_entry=adapter,
    )

    output = pipeline.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-16T12:01:00Z"),
        bid=1.0,
        ask=9999.0,
        on_bar_committed=lambda _trade: None,
    )

    assert output["action"] == "HOLD"
    assert trade.bars_in_trade == 1
    assert trade.last_processed_m1_ts == pd.Timestamp(
        "2026-07-16T12:00:00Z"
    )
    assert len(model.calls) == 1


def test_pipeline_catches_up_every_contiguous_authoritative_m1_bar(
    tmp_path: Path,
) -> None:
    trade = _open()
    adapter, model = _unified_exit_adapter(
        tmp_path,
        logits=(2.0, -1.0),
    )

    class _Loader:
        def refresh_if_changed(self) -> bool:
            return False

        def acquire_serving_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                base28=pd.DataFrame(
                    index=pd.date_range(
                        "2026-07-16T12:00:00Z",
                        periods=2,
                        freq="min",
                    )
                )
            )

        def get_closed_m1_bar(
            self,
            expected_m1: pd.Timestamp,
            *,
            snapshot: SimpleNamespace,
        ) -> dict[str, object]:
            assert expected_m1 in snapshot.base28.index
            return _valid_closed_m1_bar(expected_m1.isoformat())

    pipeline = V12Pipeline(
        prebuilt_loader=_Loader(),
        smart_entry=adapter,
    )

    output = pipeline.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-16T12:02:00Z"),
        bid=3300.0,
        ask=3300.2,
        on_bar_committed=lambda _trade: None,
    )

    assert output["action"] == "HOLD"
    assert trade.bars_in_trade == 2
    assert trade.last_processed_m1_ts == pd.Timestamp(
        "2026-07-16T12:01:00Z"
    )
    assert len(model.calls) == 2


def test_pipeline_catches_up_consecutive_source_rows_across_market_closure(
    tmp_path: Path,
) -> None:
    trade = _open()
    adapter, model = _unified_exit_adapter(
        tmp_path,
        logits=(2.0, -1.0),
    )
    source_index = pd.DatetimeIndex(
        ["2026-07-16T12:00:00Z", "2026-07-18T12:00:00Z"]
    )

    class _Loader:
        def refresh_if_changed(self) -> bool:
            return False

        def acquire_serving_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(base28=pd.DataFrame(index=source_index))

        def get_closed_m1_bar(
            self,
            expected_m1: pd.Timestamp,
            *,
            snapshot: SimpleNamespace,
        ) -> dict[str, object]:
            assert expected_m1 in snapshot.base28.index
            return _valid_closed_m1_bar(expected_m1.isoformat())

    pipeline = V12Pipeline(
        prebuilt_loader=_Loader(),
        smart_entry=adapter,
    )

    output = pipeline.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-18T12:01:00Z"),
        bid=3300.0,
        ask=3300.2,
        on_bar_committed=lambda _trade: None,
    )

    assert output["action"] == "HOLD"
    assert trade.bars_in_trade == 2
    assert trade.last_processed_m1_ts == source_index[-1]
    assert len(model.calls) == 2


def test_pipeline_durably_journals_each_catch_up_bar_before_advancing(
    tmp_path: Path,
) -> None:
    trade = _open()
    trade.trade_id = "trade-batch-journal"
    adapter, model = _unified_exit_adapter(
        tmp_path,
        logits=(2.0, -1.0),
    )

    class _Loader:
        def refresh_if_changed(self) -> bool:
            return False

        def acquire_serving_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                base28=pd.DataFrame(
                    index=pd.date_range(
                        "2026-07-16T12:00:00Z",
                        periods=3,
                        freq="min",
                    )
                )
            )

        def get_closed_m1_bar(
            self,
            expected_m1: pd.Timestamp,
            *,
            snapshot: SimpleNamespace,
        ) -> dict[str, object]:
            assert expected_m1 in snapshot.base28.index
            return _valid_closed_m1_bar(expected_m1.isoformat())

    journal = TradeJournal(
        run_dir=tmp_path / "journal-run",
        run_tag="UNIT_M1_BATCH_DURABILITY",
        header={"meta": {"role": "TEST"}},
        enabled=True,
    )
    journal_record = journal._get_trade_journal(trade_id=trade.trade_id)
    journal_record["entry_snapshot"] = {
        "model_evidence": trade.require_entry_snapshot(),
    }
    state_directory = tmp_path / "states"

    def persist_each_bar(committed_trade: TradeState) -> None:
        persist_and_journal_v12_exit_bar(
            journal,
            committed_trade,
            state_directory=state_directory,
        )
        if committed_trade.bars_in_trade == 1:
            model.logits = (-1.0, 2.0)

    pipeline = V12Pipeline(
        prebuilt_loader=_Loader(),
        smart_entry=adapter,
    )
    output = pipeline.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-16T12:03:00Z"),
        bid=3300.0,
        ask=3300.2,
        on_bar_committed=persist_each_bar,
    )

    assert output["action"] == "EXIT_NOW"
    assert trade.bars_in_trade == 2
    persisted_trade = TradeState.load(
        state_directory / trade.state_filename()
    )
    assert persisted_trade is not None
    assert persisted_trade.to_dict() == trade.to_dict()

    persisted_journal = json.loads(
        (journal.trade_json_dir / f"{trade.trade_id}.json").read_text()
    )
    decisions = persisted_journal["v12_bar_decisions"]
    assert [row["exit_action"] for row in decisions] == [
        "HOLD",
        "EXIT_NOW",
    ]
    assert [row["timestamp"] for row in decisions] == [
        "2026-07-16T12:00:00+00:00",
        "2026-07-16T12:01:00+00:00",
    ]
    assert all(row["exit_action_logits"] for row in decisions)
    assert all(row["exit_action_probs"] for row in decisions)
    assert all(row["output_evidence_sha256"] for row in decisions)
    assert len({row["output_evidence_sha256"] for row in decisions}) == 2


def test_pipeline_handles_two_five_minute_source_publication_batches(
    tmp_path: Path,
) -> None:
    trade = _open()
    adapter, model = _unified_exit_adapter(
        tmp_path,
        logits=(2.0, -1.0),
    )

    class _Loader:
        def __init__(self) -> None:
            self.periods = 5

        def refresh_if_changed(self) -> bool:
            return False

        def acquire_serving_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                base28=pd.DataFrame(
                    index=pd.date_range(
                        "2026-07-16T12:00:00Z",
                        periods=self.periods,
                        freq="min",
                    )
                )
            )

        def get_closed_m1_bar(
            self,
            expected_m1: pd.Timestamp,
            *,
            snapshot: SimpleNamespace,
        ) -> dict[str, object]:
            assert expected_m1 in snapshot.base28.index
            return _valid_closed_m1_bar(expected_m1.isoformat())

    loader = _Loader()
    pipeline = V12Pipeline(
        prebuilt_loader=loader,
        smart_entry=adapter,
    )
    first = pipeline.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-16T12:06:00Z"),
        bid=3300.0,
        ask=3300.2,
        on_bar_committed=lambda _trade: None,
    )
    loader.periods = 10
    second = pipeline.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-16T12:11:00Z"),
        bid=3300.0,
        ask=3300.2,
        on_bar_committed=lambda _trade: None,
    )

    assert first["action"] == second["action"] == "HOLD"
    assert trade.bars_in_trade == 10
    assert trade.last_processed_m1_ts == pd.Timestamp(
        "2026-07-16T12:09:00Z"
    )
    assert len(model.calls) == 10


def test_exit_recovery_loads_trade_bound_bundle_without_active_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from gx1.execution import v12_pipeline as pipeline_module
    from gx1.execution import v12_smart_entry_live as smart_module

    class _Loader:
        def __init__(self) -> None:
            self.loaded = False

        def load(self) -> None:
            self.loaded = True

    bound_m1: dict[str, object] = {}
    adapter = SimpleNamespace(
        _meta={
            "unified_entry_exit_contract": (
                unified_entry_exit_contract_metadata()
            ),
            "m1_feature_surface_binding": {
                "parquet_path": "/tmp/m1_feature_surface.parquet",
                "manifest_path": "/tmp/m1_feature_surface.parquet.manifest.json",
                "dataset_run_id": "UNIT_EXIT_RECOVERY_DATASET",
                "pair_generation_id": "UNIT_EXIT_RECOVERY_PAIR",
                "parquet_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "feature_field_order_sha256": "3" * 64,
            },
        },
        _model=SimpleNamespace(
            state_dict=lambda: {
                "head_exit_action.weight": torch.zeros((2, 1)),
                "head_exit_action.bias": torch.zeros(2),
            }
        ),
        decide_exit=lambda **_kwargs: {},
        bind_admitted_m1_feature_surface=lambda **kwargs: bound_m1.update(
            kwargs
        ),
    )
    observed: dict[str, object] = {}

    def _load_recovery(
        **kwargs: object,
    ) -> SimpleNamespace:
        observed.update(kwargs)
        return adapter

    monkeypatch.setattr(pipeline_module, "PrebuiltStateLoader", _Loader)
    monkeypatch.setattr(
        smart_module.SmartEntryLiveInference,
        "load_immutable_exit_recovery",
        _load_recovery,
    )
    trade = SimpleNamespace(
        model_bundle_binding=_model_bundle_binding(),
        require_entry_snapshot=lambda: _snapshot(),
        sizing_execution_evidence={
            "sizing_application": {
                "sizing_authority_contract": unverified_learned_sizing_authority(),
            },
        },
    )

    pipeline = V12Pipeline.load_exit_recovery(trade)

    assert isinstance(pipeline.prebuilt_loader, _Loader)
    assert pipeline.prebuilt_loader.loaded is True
    assert pipeline.smart_entry is adapter
    assert observed["bundle_dir"] == Path(
        _model_bundle_binding()["bundle_dir"]
    )
    assert observed["expected_bundle_sha256"] == "b" * 64
    assert bound_m1["dataset_run_id"] == "UNIT_EXIT_RECOVERY_DATASET"


def test_first_full_m1_bar_excludes_prefill_intraminute_path() -> None:
    assert first_full_closed_m1_bar_ts(
        pd.Timestamp("2026-07-16T12:00:00Z")
    ) == pd.Timestamp("2026-07-16T12:00:00Z")
    assert first_full_closed_m1_bar_ts(
        pd.Timestamp("2026-07-16T12:00:00.000001Z")
    ) == pd.Timestamp("2026-07-16T12:01:00Z")
    assert first_full_closed_m1_bar_ts(
        pd.Timestamp("2026-07-16T12:00:17.250000Z")
    ) == pd.Timestamp("2026-07-16T12:01:00Z")


def test_trade_state_keeps_learned_sizing_without_hold_rule_sentinel() -> None:
    trade = _open()

    assert trade.require_entry_snapshot()["position_size_pred"] == pytest.approx(
        _snapshot()["position_size_pred"]
    )
    assert "v10_hold_horizon_at_entry_v1" not in trade.require_entry_snapshot()
    assert trade.sizing_execution_evidence["mode"] == (
        "unit_normalized_research_only"
    )
    assert trade.sizing_execution_evidence["executable_order_authority"] is False


def test_executable_trade_state_has_no_default_units_or_sizing_fallback() -> None:
    with pytest.raises(TypeError, match="units"):
        TradeState.open(
            entry_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_snapshot(),
        )


def test_trade_state_rejects_entry_time_not_derived_from_snapshot_timing() -> None:
    with pytest.raises(RuntimeError, match="model-derived minute"):
        TradeState.open_unit_normalized_research(
            entry_ts=pd.Timestamp("2026-07-16T12:01:00Z"),
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_snapshot(),
            normalization_contract="unit_normalized_direction_exit_research_v1",
        )


def test_exit_research_accepts_exact_head_envelope_without_fake_sizing_adoption() -> None:
    trade = _open(_head_snapshot())

    assert trade.require_entry_snapshot() == _head_snapshot()
    assert trade.require_entry_snapshot()["position_size_pred"] == pytest.approx(
        _snapshot()["position_size_pred"]
    )
    restored = TradeState.from_dict(json.loads(json.dumps(trade.to_dict())))
    assert restored.v10_snapshot == _head_snapshot()
    assert restored.sizing_execution_evidence["executable_order_authority"] is False


def test_exit_research_head_requires_exact_t5_fill_and_cannot_open_learned_trade() -> None:
    with pytest.raises(RuntimeError, match="exact T\\+5 replay fill"):
        TradeState.open_unit_normalized_research(
            entry_ts=pd.Timestamp("2026-07-16T12:01:00Z"),
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_head_snapshot(),
            normalization_contract="unit_normalized_direction_exit_research_v1",
        )

    with pytest.raises(RuntimeError, match="exact schema mismatch"):
        TradeState.open(
            entry_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_head_snapshot(),
            units=1,
            sizing_application={},
            fill_transaction_id="virtual:test",
            execution_mode="learned_virtual_dry_run",
            model_bundle_binding=_model_bundle_binding(),
            entry_source_pair_binding=_source_pair_binding(),
        )


def test_broker_account_binding_is_exact_and_mode_owned() -> None:
    binding = build_trade_broker_account_binding(
        environment="practice",
        account_id="101-001-12345678-001",
    )
    assert binding["schema_version"] == (
        TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION
    )
    assert len(binding["account_id_sha256"]) == 64
    assert (
        require_trade_broker_account_binding(
            binding,
            execution_mode="learned_broker_fill",
        )
        == binding
    )
    with pytest.raises(ValueError, match="cannot claim"):
        require_trade_broker_account_binding(
            binding,
            execution_mode="learned_virtual_dry_run",
        )
    with pytest.raises(ValueError, match="exact schema"):
        require_trade_broker_account_binding(
            None,
            execution_mode="learned_broker_fill",
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "direction_logits",
        "path_quality",
        "tf_agreement_logit",
        "path_quality_log_var",
        "position_size_logit",
        "decision_available_ts",
        "model_policy",
        "session_id",
    ],
)
def test_trade_state_rejects_missing_model_native_entry_evidence(missing_key: str) -> None:
    snapshot = _snapshot()
    del snapshot[missing_key]

    with pytest.raises(RuntimeError, match=missing_key):
        _open(snapshot)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("direction_probs", [0.1, 0.8, 0.1]),
        ("public_trade_flat_decision_logits", [4.0, 0.0]),
        ("tf_agreement_pred", 0.01),
        ("path_quality_std", 2.0),
        ("position_size_pred", 0.01),
    ],
)
def test_trade_state_rejects_inconsistent_model_native_entry_evidence(
    key: str,
    value: object,
) -> None:
    snapshot = _snapshot()
    snapshot[key] = value

    with pytest.raises(RuntimeError, match=key):
        require_model_native_entry_snapshot(snapshot)


@pytest.mark.parametrize("key", ["hold_horizon_pred", "hold_horizon_bars_pred"])
def test_trade_state_rejects_blocked_hold_horizon_pass_through(key: str) -> None:
    snapshot = _snapshot()
    snapshot[key] = 96

    with pytest.raises(RuntimeError, match="retired fields"):
        _open(snapshot)


def test_trade_state_load_fails_closed_on_stale_snapshot(tmp_path: Path) -> None:
    trade = _open()
    payload = trade.to_dict()
    del payload["v10_snapshot"]["position_size_logit"]
    path = tmp_path / "open_trade_stale.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to load trade state"):
        TradeState.load(path)


def test_trade_state_update_bar_records_exact_intrabar_state() -> None:
    trade = _open()

    trade.update_bar(**_valid_closed_m1_bar())

    assert trade.bars_in_trade == 1
    assert list(trade.peak_history) == pytest.approx(
        [(3301.5 - trade.entry_ask) / trade.entry_ask * 10_000.0]
    )
    assert list(trade.trough_history) == pytest.approx(
        [(3299.7 - trade.entry_ask) / trade.entry_ask * 10_000.0]
    )
    assert trade.cum_mfe_bps == pytest.approx(
        (3301.5 - trade.entry_ask) / trade.entry_ask * 10_000.0
    )
    assert trade.cum_mae_bps == pytest.approx(
        (3299.7 - trade.entry_ask) / trade.entry_ask * 10_000.0
    )
    assert list(trade.executable_range_bps_history) == pytest.approx(
        [(3301.7 - 3299.7) / 3301.1 * 10_000.0]
    )
    assert trade.last_executable_range_bps == pytest.approx(
        trade.executable_range_bps_history[-1]
    )
    path_evidence = trade.build_closed_m1_path_evidence()
    assert path_evidence["path_rows"] == [_valid_closed_m1_bar()]
    assert len(path_evidence["path_rows_sha256"]) == 64


def test_trade_state_update_bar_requires_all_bid_ask_ohlc_before_mutation() -> None:
    trade = _open()
    bar = _valid_closed_m1_bar()
    del bar["ask_low"]

    with pytest.raises(TypeError, match="ask_low"):
        trade.update_bar(**bar)

    assert trade.bars_in_trade == 0
    assert not trade.pnl_history
    assert not trade.peak_history


def test_trade_state_accepts_forward_market_gap_but_rejects_duplicate_or_reversal() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)
    before = trade.to_dict()

    with pytest.raises(ValueError, match="row clock duplicate/reversal"):
        trade.update_bar(**_valid_closed_m1_bar())
    assert trade.to_dict() == before

    trade.update_bar(
        **_valid_closed_m1_bar("2026-07-16T12:02:00Z")
    )
    assert trade.bars_in_trade == 2

    with pytest.raises(ValueError, match="row clock duplicate/reversal"):
        trade.update_bar(
            **_valid_closed_m1_bar("2026-07-16T12:01:00Z")
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid_close", np.nan),
        ("ask_close", np.inf),
        ("mid_close", 0.0),
        ("bid_high", -1.0),
        ("bid_low", np.nan),
        ("ask_high", np.inf),
        ("ask_low", 0.0),
    ],
)
def test_trade_state_update_bar_rejects_invalid_price_values_without_mutation(
    field: str,
    value: float,
) -> None:
    trade = _open()
    bar = _valid_closed_m1_bar()
    bar[field] = value

    with pytest.raises(ValueError, match="non-finite/non-positive"):
        trade.update_bar(**bar)

    assert trade.bars_in_trade == 0
    assert not trade.pnl_history
    assert not trade.executable_range_bps_history
    assert not trade.closed_m1_path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid_low", 3300.9),
        ("bid_high", 3300.9),
        ("ask_low", 3301.1),
        ("ask_high", 3301.1),
        ("ask_low", 3299.6),
        ("ask_high", 3301.4),
        ("mid_low", 3301.0),
        ("mid_high", 3301.0),
    ],
)
def test_trade_state_update_bar_rejects_invalid_geometry_without_mutation(
    field: str,
    value: float,
) -> None:
    trade = _open()
    bar = _valid_closed_m1_bar()
    bar[field] = value

    with pytest.raises(ValueError, match="M/B/A OHLC geometry invalid"):
        trade.update_bar(**bar)

    assert trade.bars_in_trade == 0
    assert not trade.pnl_history
    assert not trade.executable_range_bps_history
    assert not trade.closed_m1_path


def test_trade_state_from_dict_rejects_missing_intrabar_history() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)
    payload = trade.to_dict()
    del payload["peak_history"]

    with pytest.raises(ValueError, match="missing exact intrabar histories"):
        TradeState.from_dict(payload)


def test_trade_state_from_dict_rejects_synthetic_or_misaligned_intrabar_history() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)
    payload = trade.to_dict()
    payload["executable_range_bps_history"] = [0.0]

    with pytest.raises(ValueError, match="intrabar histories are invalid"):
        TradeState.from_dict(payload)

    payload = trade.to_dict()
    payload["trough_history"] = []
    with pytest.raises(ValueError, match="not aligned"):
        TradeState.from_dict(payload)


def test_trade_state_zero_bar_round_trip_uses_exact_versioned_schema() -> None:
    trade = _open()

    payload = trade.to_dict()
    restored = TradeState.from_dict(json.loads(json.dumps(payload)))

    assert payload["schema_version"] == PERSISTED_TRADE_STATE_SCHEMA_VERSION
    assert restored.bars_in_trade == 0
    assert restored.entry_ts == trade.entry_ts
    assert restored.to_dict() == payload


def test_trade_state_from_dict_requires_every_serialized_field() -> None:
    payload = _open().to_dict()

    for field_name in payload:
        incomplete = json.loads(json.dumps(payload))
        del incomplete[field_name]
        with pytest.raises(ValueError):
            TradeState.from_dict(incomplete)


def test_trade_state_rejects_unexpected_or_retired_persisted_fields() -> None:
    payload = _open().to_dict()
    payload["legacy_entry_filter"] = 0.0

    with pytest.raises(ValueError, match="unexpected=.*legacy_entry_filter"):
        TradeState.from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("entry_ts", "2026-07-16T14:00:00+02:00", "timezone-aware UTC"),
        ("entry_ts", "2026-07-16T12:00:00", "timezone-aware UTC"),
        ("side", "flat", "side must be"),
        ("entry_bid", 0.0, "entry_bid must be positive"),
        ("entry_ask", np.inf, "entry_ask must be finite"),
        ("current_bid", np.nan, "current_bid must be finite"),
        ("current_ask", -1.0, "current_ask must be positive"),
        ("units", 0, "units must be positive"),
        ("units", 1.0, "units must be an exact integer"),
    ],
)
def test_trade_state_rejects_invalid_identity_price_and_unit_fields(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    payload = _open().to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        TradeState.from_dict(payload)


def test_trade_state_rejects_side_model_direction_and_spread_mismatch() -> None:
    payload = _open().to_dict()
    payload["side"] = "short"
    with pytest.raises(ValueError, match="frozen model direction"):
        TradeState.from_dict(payload)

    payload = _open().to_dict()
    payload["entry_spread_bps"] += 0.01
    with pytest.raises(ValueError, match="does not match entry prices"):
        TradeState.from_dict(payload)


def test_trade_state_to_dict_rejects_invalid_in_memory_state() -> None:
    trade = _open()
    trade.units = 0

    with pytest.raises(ValueError, match="units must be positive"):
        trade.to_dict()


def test_trade_state_binds_bar_count_to_all_persisted_history_lengths() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)
    payload = trade.to_dict()
    payload["bars_in_trade"] = 2
    payload["last_processed_m1_ts"] = "2026-07-16T12:01:00+00:00"

    with pytest.raises(ValueError, match="m1_returns_window length"):
        TradeState.from_dict(payload)

    payload = trade.to_dict()
    payload["peak_history"].append(payload["peak_history"][-1])
    with pytest.raises(ValueError, match="not aligned"):
        TradeState.from_dict(payload)


def test_trade_state_accepts_exact_unified_exit_path_capacity() -> None:
    trade = _open()
    start = pd.Timestamp("2026-07-16T12:00:00Z")
    for offset in range(TRAJECTORY_HISTORY_MAXLEN):
        trade.update_bar(
            **_valid_closed_m1_bar(
                (start + pd.Timedelta(minutes=offset)).isoformat()
            )
        )
    _bind_test_hold_decision(trade)

    restored = TradeState.from_dict(trade.to_dict())

    assert len(restored.m1_returns_window) == M1_RETURNS_WINDOW_MAXLEN
    assert len(restored.pnl_history) == TRAJECTORY_HISTORY_MAXLEN
    assert len(restored.closed_m1_path) == TRAJECTORY_HISTORY_MAXLEN
    assert restored.bars_in_trade == TRAJECTORY_HISTORY_MAXLEN


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("current_pnl_bps", np.inf, "current_pnl_bps must be finite"),
        ("cum_mfe_bps", np.nan, "cum_mfe_bps must be finite"),
        ("cum_mae_bps", np.inf, "cum_mae_bps must be finite"),
        (
            "last_executable_range_bps",
            -0.1,
            "last_executable_range_bps must be nonnegative",
        ),
        ("bars_since_mfe_peak", -1, "bars_since_mfe_peak must be nonnegative"),
    ],
)
def test_trade_state_rejects_invalid_running_metrics(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    payload = _open().to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        TradeState.from_dict(payload)

@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("m1_returns_window", [np.nan], "m1_returns_window is invalid"),
        ("pnl_history", [np.inf], "pnl_history is invalid"),
        (
            "executable_range_bps_history",
            [0.0],
            "intrabar histories are invalid",
        ),
    ],
)
def test_trade_state_rejects_nonfinite_or_nonpositive_history(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)
    payload = trade.to_dict()
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        TradeState.from_dict(payload)


def test_trade_state_rejects_intrabar_order_or_close_outside_excursion() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)

    payload = trade.to_dict()
    payload["peak_history"][0] = payload["trough_history"][0] - 0.1
    with pytest.raises(
        ValueError,
        match="derived histories do not match literal path",
    ):
        TradeState.from_dict(payload)

    payload = trade.to_dict()
    payload["pnl_history"][0] = payload["peak_history"][0] + 0.1
    with pytest.raises(
        ValueError,
        match="derived histories do not match literal path",
    ):
        TradeState.from_dict(payload)


def test_trade_state_transactional_commit_preserves_exact_closed_m1_state() -> None:
    trade = _open()
    staged = trade.clone_for_exit_decision()
    staged.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(staged)

    assert trade.bars_in_trade == 0

    trade.commit_complete_exit_bar(staged)
    restored = TradeState.from_dict(trade.to_dict())

    assert restored.last_processed_m1_ts == pd.Timestamp(
        "2026-07-16T12:00:00Z"
    )
    assert restored.bars_in_trade == 1
    assert restored.last_executable_range_bps == pytest.approx(
        restored.executable_range_bps_history[-1]
    )
    assert restored.last_exit_decision == staged.last_exit_decision


def test_trade_state_rejects_tampered_or_missing_persisted_exit_decision() -> None:
    trade = _open()
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)
    payload = trade.to_dict()

    missing = dict(payload)
    missing["last_exit_decision"] = None
    with pytest.raises(ValueError, match="requires its last Exit decision"):
        TradeState.from_dict(missing)

    tampered = json.loads(json.dumps(payload))
    tampered["last_exit_decision"]["action"] = "EXIT_NOW"
    with pytest.raises(ValueError, match="last Exit decision is invalid"):
        TradeState.from_dict(tampered)


def test_persisted_exit_decision_journal_recovery_is_idempotent(
    tmp_path: Path,
) -> None:
    trade = _open()
    trade.trade_id = "trade-unified-exit-recovery"
    trade.update_bar(**_valid_closed_m1_bar())
    _bind_test_hold_decision(trade)
    journal = TradeJournal(
        run_dir=tmp_path,
        run_tag="UNIT_EXIT_RECOVERY",
        header={"meta": {"role": "TEST"}},
        enabled=True,
    )
    journal_record = journal._get_trade_journal(trade_id=trade.trade_id)
    journal_record["entry_snapshot"] = {
        "model_evidence": trade.require_entry_snapshot(),
    }

    journal_v12_exit_decision(journal, trade)
    journal_v12_exit_decision(journal, trade)

    observed = journal._get_trade_journal(trade_id=trade.trade_id)[
        "v12_bar_decisions"
    ]
    assert len(observed) == 1
    assert observed[0]["timestamp"] == trade.last_processed_m1_ts.isoformat()
    assert observed[0]["bid"] == trade.current_bid
    assert observed[0]["ask"] == trade.current_ask


def test_trade_state_load_all_requires_filename_identity_and_sorts(tmp_path: Path) -> None:
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    later_snapshot = _snapshot()
    later_snapshot.update(
        {
            "decision_ts": "2026-07-16T12:00:00+00:00",
            "decision_available_ts": "2026-07-16T12:05:00+00:00",
            "context_cutoff_ts": "2026-07-16T12:00:00+00:00",
        }
    )
    later = TradeState.open_unit_normalized_research(
        entry_ts=pd.Timestamp("2026-07-16T12:05:00Z"),
        side="long",
        entry_bid=3300.0,
        entry_ask=3300.2,
        v10_snapshot=later_snapshot,
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )
    later.trade_id = "later"
    earlier = _open()
    earlier.trade_id = "earlier"
    later.save(state_dir)
    earlier.save(state_dir)

    restored = TradeState.load_all(state_dir)

    assert [trade.trade_id for trade in restored] == ["earlier", "later"]

    bad_path = state_dir / "open_trade_wrong.json"
    bad_payload = _open().to_dict()
    bad_payload["trade_id"] = "actual"
    bad_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="filename/identity mismatch"):
        TradeState.load_all(state_dir)
