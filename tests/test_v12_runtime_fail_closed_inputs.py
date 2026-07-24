from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.execution import v12_pipeline as pipeline_module
from gx1.execution.v12_pipeline import (
    EntryDecisionUnavailable,
    ExitDecisionUnavailable,
    V12Pipeline,
    _exact_closed_m5_row,
    _validated_v3_output,
    require_xgb_v3_chain_identity,
)
from gx1.execution.v12_v3_live import (
    WINDOW_LEN as V3_WINDOW_LEN,
    XGB_BRIDGE_NAMES,
    V3LiveInference,
    _resolve_default_v3_bundle,
    build_v3_base_feature_rows,
    required_closed_m5_keys_for_v3_window,
)
from gx1.execution.v12_xgb_live import _require_ordered_xgb_feature_identity


def test_active_exit_replay_factory_has_no_smart_entry_or_mutable_provider() -> None:
    source = inspect.getsource(V12Pipeline.load_active_exit_replay)

    assert "SmartEntry" not in source
    assert "load_default" not in source
    assert "load_frozen_pair" in source
    with pytest.raises(
        RuntimeError,
        match="REQUIRES_EXACT_SOURCE_TAPE",
    ):
        V12Pipeline.load_active_exit_replay(
            artifact_registry_path=Path("/missing/registry.json"),
            prebuilt_pair_manifest_path=Path("/missing/pair.json"),
            prebuilt_generation_root=Path("/missing/generations"),
            closed_m1_provider=SimpleNamespace(),
        )


class _PassThroughSanitizer:
    def sanitize(
        self,
        frame: pd.DataFrame,
        *,
        feature_list: list[str],
        allow_nan_fill: bool,
        nan_fill_value: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        assert allow_nan_fill is False
        assert nan_fill_value == 0.0
        return frame.loc[:, feature_list].to_numpy(dtype=np.float64), {}


class _SessionProbabilityModel:
    def __init__(
        self,
        probabilities: tuple[float, float, float] = (0.6, 0.3, 0.1),
    ) -> None:
        self.probabilities = probabilities
        self.observed_sessions: list[str] = []

    def predict_proba(
        self,
        frame: pd.DataFrame,
        *,
        session: str,
        feature_list: list[str],
    ) -> SimpleNamespace:
        assert feature_list == ["feature"]
        self.observed_sessions.append(session)
        count = len(frame)
        return SimpleNamespace(
            p_long=np.full(count, self.probabilities[0]),
            p_short=np.full(count, self.probabilities[1]),
            p_flat=np.full(count, self.probabilities[2]),
        )


def _stub_xgb(
    probabilities: tuple[float, float, float] = (0.6, 0.3, 0.1),
):
    from gx1.execution.v12_xgb_live import XGBLiveInference

    model = _SessionProbabilityModel(probabilities)
    inference = XGBLiveInference(
        bundle_dir=Path("."),
        sanitizer_config=Path("sanitizer.json"),
        feature_contract=Path("features.json"),
        _model=model,
        _sanitizer=_PassThroughSanitizer(),
        _features=["feature"],
    )
    return inference, model


def _collector_rows(*times: str) -> pd.DataFrame:
    rows = []
    for offset, time in enumerate(times):
        bid_close = 2400.0 + offset
        ask_close = bid_close + 0.2
        rows.append(
            {
                "time": pd.Timestamp(time),
                "bid_high": bid_close + 0.5,
                "bid_low": bid_close - 0.5,
                "ask_high": ask_close + 0.5,
                "ask_low": ask_close - 0.5,
                "bid_close": bid_close,
                "ask_close": ask_close,
            }
        )
    return pd.DataFrame(rows)


def test_v3_default_bundle_ignores_environment_path_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1_guards import artifacts

    monkeypatch.setenv("GX1_V3_BUNDLE_DIR", "/tmp/unbound-v3")
    monkeypatch.setattr(
        artifacts,
        "load_decision_artifact",
        lambda role: "/registry/bound-v3" if role == "v3_exit" else None,
    )

    assert _resolve_default_v3_bundle() == Path("/registry/bound-v3")


def test_xgb_v3_chain_requires_exact_training_identity() -> None:
    identity = {"identity_sha256": "a" * 64}
    assert (
        require_xgb_v3_chain_identity(
            SimpleNamespace(_runtime_identity=identity),
            SimpleNamespace(
                _training_lineage={"xgb_bridge_source": deepcopy(identity)}
            ),
        )
        == identity
    )

    with pytest.raises(RuntimeError, match="CHAIN_IDENTITY_MISMATCH"):
        require_xgb_v3_chain_identity(
            SimpleNamespace(_runtime_identity=identity),
            SimpleNamespace(
                _training_lineage={
                    "xgb_bridge_source": {"identity_sha256": "b" * 64}
                }
            ),
        )


@pytest.mark.parametrize("session_id", [-1.0, 4.0, 1.5, np.nan, np.inf])
def test_xgb_rejects_invalid_explicit_session_id(session_id: float) -> None:
    inference, model = _stub_xgb()
    frame = pd.DataFrame(
        {"feature": [1.0], "session_id": [session_id]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T12:00:00Z")]),
    )

    with pytest.raises(RuntimeError, match="XGB_SESSION_ID_INVALID"):
        inference.predict(frame)

    assert model.observed_sessions == []


@pytest.mark.parametrize("session", ["", "asia", "INVALID"])
def test_xgb_rejects_invalid_explicit_session_name(session: str) -> None:
    inference, model = _stub_xgb()
    frame = pd.DataFrame(
        {"feature": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T12:00:00Z")]),
    )

    with pytest.raises(RuntimeError, match="XGB_SESSION_INVALID"):
        inference.predict(frame, session=session)

    assert model.observed_sessions == []


def test_xgb_routes_each_row_once_and_emits_exact_bridge() -> None:
    inference, model = _stub_xgb()
    frame = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "session_id": [0.0, 1.0, 2.0, 3.0],
        },
        index=pd.date_range("2026-07-16T06:00:00Z", periods=4, freq="h"),
    )

    result = inference.predict(frame)

    assert model.observed_sessions == ["ASIA", "EU", "OVERLAP", "US"]
    assert result["session"].tolist() == ["ASIA", "EU", "OVERLAP", "US"]
    np.testing.assert_array_equal(
        result["signal_bridge_v1"][:, :3],
        np.asarray([[0.6, 0.3, 0.1]] * 4, dtype=np.float32),
    )
    assert np.isfinite(result["signal_bridge_v1"]).all()


def test_xgb_rejects_non_simplex_head_output() -> None:
    inference, _model = _stub_xgb((0.6, 0.6, 0.1))
    frame = pd.DataFrame(
        {"feature": [1.0], "session_id": [0.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T06:00:00Z")]),
    )

    with pytest.raises(RuntimeError, match="XGB_PROBABILITY_SIMPLEX_INVALID"):
        inference.predict(frame)


def test_xgb_feature_identity_requires_metadata_order_and_sanitizer_parity() -> None:
    _require_ordered_xgb_feature_identity(
        model_feature_names=["a", "b"],
        contract_features=["a", "b"],
        sanitizer_features=["a", "b"],
    )
    with pytest.raises(RuntimeError, match="metadata lacks exact"):
        _require_ordered_xgb_feature_identity(
            model_feature_names=None,
            contract_features=["a", "b"],
            sanitizer_features=["a", "b"],
        )
    with pytest.raises(RuntimeError, match="order differs"):
        _require_ordered_xgb_feature_identity(
            model_feature_names=["b", "a"],
            contract_features=["a", "b"],
            sanitizer_features=["a", "b"],
        )
    with pytest.raises(RuntimeError, match="sanitizer feature order differs"):
        _require_ordered_xgb_feature_identity(
            model_feature_names=["a", "b"],
            contract_features=["a", "b"],
            sanitizer_features=["a"],
        )


def _pipeline(loader: object | None = None, **kwargs: object) -> V12Pipeline:
    return V12Pipeline(
        prebuilt_loader=loader or SimpleNamespace(),
        exit_xgb=object(),
        **kwargs,
    )


class _CanonicalLoader:
    def __init__(
        self,
        cutoff: pd.Timestamp,
        window: pd.DataFrame,
        *,
        base_m1: pd.DataFrame | None = None,
    ) -> None:
        self.cutoff_ts = cutoff
        self.window = window
        self._base28 = base_m1
        self.requested: list[tuple[pd.Timestamp, int]] = []

    def refresh_if_changed(self) -> bool:
        return False

    def get_window(self, end_ts: pd.Timestamp, *, n_bars: int) -> pd.DataFrame:
        self.requested.append((end_ts, n_bars))
        return self.window.copy()


def _canonical_window(end: str) -> pd.DataFrame:
    index = pd.date_range(
        end=pd.Timestamp(end),
        periods=pipeline_module.ENTRY_SEQ_LEN,
        freq="5min",
    )
    return pd.DataFrame({"atr_bps": np.full(len(index), 12.0)}, index=index)


def _v3_m1_source(end: str) -> pd.DataFrame:
    index = pd.date_range(
        end=pd.Timestamp(end),
        periods=V3_WINDOW_LEN,
        freq="min",
    )
    return pd.DataFrame({"m1_source": np.arange(len(index), dtype=float)}, index=index)


def test_entry_canonical_freshness_is_fixed_and_ignores_exit_staleness_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_m1 = _v3_m1_source("2026-07-16T12:05:00Z")
    loader = _CanonicalLoader(
        pd.Timestamp("2026-07-16T11:55:00Z"),
        _canonical_window("2026-07-16T11:55:00Z"),
        base_m1=base_m1,
    )
    pipe = _pipeline(loader)
    monkeypatch.setenv("GX1_MAX_PREBUILT_STALENESS_MIN", "999999")

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:06:00Z"))

    assert raised.value.reason == "entry_canonical_stale"
    assert raised.value.evidence["canonical_cutoff_age_sec"] == 660.0
    assert raised.value.evidence["canonical_cutoff_age_cap_sec"] == 390.0
    assert loader.requested == []

    # The Exit staleness knob cannot authorize a cutoff that omits even one
    # closed-M5 key required by the actual 512-M1 V3 window.
    with pytest.raises(ExitDecisionUnavailable) as exit_raised:
        pipe._refresh_exit_canonical(pd.Timestamp("2026-07-16T12:06:00Z"))
    assert exit_raised.value.reason == "exit_latest_required_closed_m5_unavailable"
    assert exit_raised.value.evidence["required_latest_m5"] == "2026-07-16 12:00:00+00:00"
    assert loader.requested == []


def test_exit_canonical_window_is_derived_from_exact_512_m1_coverage() -> None:
    now = pd.Timestamp("2026-07-16T12:07:00Z")
    decision_m1 = now - pd.Timedelta(minutes=1)
    base_m1 = _v3_m1_source(str(decision_m1))
    required_m5 = required_closed_m5_keys_for_v3_window(decision_m1, base_m1)
    canonical = pd.DataFrame(
        {"atr_bps": np.full(len(required_m5), 12.0)},
        index=required_m5,
    )
    loader = _CanonicalLoader(required_m5[-1], canonical, base_m1=base_m1)
    pipe = _pipeline(loader)

    pipe._refresh_exit_canonical(now)

    assert len(required_m5) > pipeline_module.ENTRY_SEQ_LEN
    assert loader.requested == [(required_m5[-1], len(required_m5))]
    assert pipe._last_exit_augmented is not None
    assert pipe._last_exit_augmented.index.equals(required_m5)
    assert pipe._last_augmented is None


class _ExactBridge:
    def __init__(self) -> None:
        self.observed_index: pd.DatetimeIndex | None = None

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        self.observed_index = pd.DatetimeIndex(frame.index)
        bridge = np.arange(len(frame) * 7, dtype=np.float32).reshape(len(frame), 7)
        return {"signal_bridge_v1": bridge}


def _minimal_v3() -> V3LiveInference:
    features = [*XGB_BRIDGE_NAMES, "atr_bps"]
    return V3LiveInference(
        bundle_dir=Path("."),
        _features=features,
        _feature_count=len(features),
    )


def test_v3_maps_each_m1_row_to_its_exact_latest_closed_m5() -> None:
    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    base_m1 = _v3_m1_source(str(end_m1))
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    canonical = pd.DataFrame(
        {"atr_bps": np.arange(len(required_m5), dtype=float) + 1.0},
        index=required_m5,
    )
    bridge = _ExactBridge()

    matrix = _minimal_v3().build_window(
        end_m1,
        base_m1,
        bridge,
        canonical_v3_window=canonical,
    )

    per_m1_closed_m5 = (
        base_m1.index + pd.Timedelta(minutes=1)
    ).floor("5min") - pd.Timedelta(minutes=5)
    expected_atr = canonical.loc[per_m1_closed_m5, "atr_bps"].to_numpy()
    bridge_positions = required_m5.get_indexer(per_m1_closed_m5)
    expected_bridge = np.arange(
        len(required_m5) * 7,
        dtype=np.float32,
    ).reshape(len(required_m5), 7)[bridge_positions]
    assert required_m5[-1] == pd.Timestamp("2026-07-16T12:00:00Z")
    assert bridge.observed_index is not None
    assert bridge.observed_index.equals(required_m5)
    np.testing.assert_array_equal(matrix[:, :7], expected_bridge)
    np.testing.assert_array_equal(matrix[:, 7], expected_atr.astype(np.float32))
    assert np.isfinite(matrix).all()


def test_closed_m5_mapping_is_exact_for_all_five_m1_phases() -> None:
    from gx1.execution.v12_m1_to_m5_downsample import (
        closed_m5_start_for_m1_bar_labels,
    )

    labels = pd.date_range("2026-07-16T12:00:00Z", periods=5, freq="min")
    observed = closed_m5_start_for_m1_bar_labels(labels)
    expected = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-07-16T11:55:00Z"),
            pd.Timestamp("2026-07-16T11:55:00Z"),
            pd.Timestamp("2026-07-16T11:55:00Z"),
            pd.Timestamp("2026-07-16T11:55:00Z"),
            pd.Timestamp("2026-07-16T12:00:00Z"),
        ]
    )

    assert observed.equals(expected)


def test_v3_phase_is_owned_by_exact_m1_timestamp_not_broadcast_prebuilt() -> None:
    labels = pd.date_range("2026-07-16T12:00:00Z", periods=5, freq="min")
    target = pd.DataFrame(
        {
            # A broad historical BASE28 could carry the wrong, M5-broadcast
            # phase.  The serving owner must ignore it.
            **{f"m5_phase_{phase}": np.ones(5) for phase in range(5)},
        },
        index=labels,
    )
    required_m5 = (
        labels + pd.Timedelta(minutes=1)
    ).floor("5min") - pd.Timedelta(minutes=5)
    canonical = pd.DataFrame(
        {
            **{f"m5_phase_{phase}": np.ones(2) for phase in range(5)},
        },
        index=pd.DatetimeIndex(required_m5.unique()),
    )
    features = [*XGB_BRIDGE_NAMES, *(f"m5_phase_{phase}" for phase in range(5))]

    matrix = build_v3_base_feature_rows(
        target_m1=target,
        volume_history_m1=target,
        canonical_v3=canonical,
        xgb_inferer=_ExactBridge(),
        feature_names=features,
    )

    np.testing.assert_array_equal(matrix[:, 7:], np.eye(5, dtype=np.float32))


def test_shared_v3_builder_requires_volume_prefix_and_is_chunk_invariant() -> None:
    from gx1.features.volume_features import (
        VOLUME_FEATURE_NAMES,
        VOLUME_FEATURE_PREFIX_ROWS,
    )

    source_rows = 700
    target_rows = 64
    index = pd.date_range(
        end=pd.Timestamp("2026-07-16T12:06:00Z"),
        periods=source_rows,
        freq="min",
    )
    full_history = pd.DataFrame(
        {
            "volume": 100.0 + np.square(np.sin(np.arange(source_rows) / 13.0)),
            "close": 2000.0 + np.cos(np.arange(source_rows) / 17.0),
        },
        index=index,
    )
    target = full_history.tail(target_rows)
    closed_m5 = (
        target.index + pd.Timedelta(minutes=1)
    ).floor("5min") - pd.Timedelta(minutes=5)
    canonical = pd.DataFrame(
        {"coverage": 1.0},
        index=pd.DatetimeIndex(closed_m5.unique()),
    )
    features = [*XGB_BRIDGE_NAMES, *VOLUME_FEATURE_NAMES]

    complete = build_v3_base_feature_rows(
        target_m1=target,
        volume_history_m1=full_history,
        canonical_v3=canonical,
        xgb_inferer=_ExactBridge(),
        feature_names=features,
    )
    bounded_history = full_history.tail(
        target_rows + VOLUME_FEATURE_PREFIX_ROWS
    )
    bounded = build_v3_base_feature_rows(
        target_m1=target,
        volume_history_m1=bounded_history,
        canonical_v3=canonical,
        xgb_inferer=_ExactBridge(),
        feature_names=features,
    )
    np.testing.assert_array_equal(complete, bounded)

    with pytest.raises(RuntimeError, match="V3_VOLUME_HISTORY_PREFIX_MISSING"):
        build_v3_base_feature_rows(
            target_m1=target,
            volume_history_m1=full_history.tail(
                target_rows + VOLUME_FEATURE_PREFIX_ROWS - 1
            ),
            canonical_v3=canonical,
            xgb_inferer=_ExactBridge(),
            feature_names=features,
        )


def test_v3_volume_features_use_required_prefix_not_model_window_start() -> None:
    from gx1.features.volume_features import (
        VOLUME_FEATURE_NAMES,
        VOLUME_FEATURE_PREFIX_ROWS,
        compute_volume_features,
    )

    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    source_rows = V3_WINDOW_LEN + VOLUME_FEATURE_PREFIX_ROWS
    index = pd.date_range(end=end_m1, periods=source_rows, freq="min")
    base_m1 = pd.DataFrame(
        {
            "volume": np.arange(1, source_rows + 1, dtype=np.float64),
            "close": 2000.0 + np.sin(np.arange(source_rows) / 11.0),
        },
        index=index,
    )
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    canonical = pd.DataFrame({"coverage": 1.0}, index=required_m5)
    features = [*XGB_BRIDGE_NAMES, *VOLUME_FEATURE_NAMES]
    v3 = V3LiveInference(
        bundle_dir=Path("."),
        _features=features,
        _feature_count=len(features),
    )

    matrix = v3.build_window(
        end_m1,
        base_m1,
        _ExactBridge(),
        canonical_v3_window=canonical,
    )
    expected = compute_volume_features(base_m1)

    for offset, name in enumerate(VOLUME_FEATURE_NAMES, start=7):
        np.testing.assert_array_equal(
            matrix[:, offset],
            expected[name][-V3_WINDOW_LEN:],
        )


def test_v3_volume_features_fail_without_required_prefix() -> None:
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES

    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    index = pd.date_range(end=end_m1, periods=V3_WINDOW_LEN, freq="min")
    base_m1 = pd.DataFrame(
        {
            "volume": np.arange(1, V3_WINDOW_LEN + 1, dtype=np.float64),
            "close": np.linspace(2000.0, 2001.0, V3_WINDOW_LEN),
        },
        index=index,
    )
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    canonical = pd.DataFrame({"coverage": 1.0}, index=required_m5)
    features = [*XGB_BRIDGE_NAMES, *VOLUME_FEATURE_NAMES]
    v3 = V3LiveInference(
        bundle_dir=Path("."),
        _features=features,
        _feature_count=len(features),
    )

    with pytest.raises(RuntimeError, match="V3_VOLUME_HISTORY_MISMATCH"):
        v3.build_window(
            end_m1,
            base_m1,
            _ExactBridge(),
            canonical_v3_window=canonical,
        )


def test_v3_rejects_one_missing_required_closed_m5_key_before_xgb() -> None:
    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    base_m1 = _v3_m1_source(str(end_m1))
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    canonical = pd.DataFrame(
        {"atr_bps": np.ones(len(required_m5) - 1)},
        index=required_m5.delete(len(required_m5) // 2),
    )
    bridge = _ExactBridge()

    with pytest.raises(RuntimeError, match="V3_CANONICAL_CLOSED_M5_COVERAGE_MISSING"):
        _minimal_v3().build_window(
            end_m1,
            base_m1,
            bridge,
            canonical_v3_window=canonical,
        )

    assert bridge.observed_index is None


def test_v3_rejects_nonfinite_active_feature_before_xgb() -> None:
    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    base_m1 = _v3_m1_source(str(end_m1))
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    canonical = pd.DataFrame(
        {"atr_bps": np.ones(len(required_m5))},
        index=required_m5,
    )
    canonical.iloc[len(canonical) // 2, 0] = np.nan
    bridge = _ExactBridge()

    with pytest.raises(RuntimeError, match="V3_ACTIVE_FEATURE_INVALID"):
        _minimal_v3().build_window(
            end_m1,
            base_m1,
            bridge,
            canonical_v3_window=canonical,
        )

    assert bridge.observed_index is None


def test_v3_rejects_missing_active_feature_instead_of_zero_fill() -> None:
    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    base_m1 = _v3_m1_source(str(end_m1))
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    canonical = pd.DataFrame(
        {"unrelated": np.ones(len(required_m5))},
        index=required_m5,
    )
    bridge = _ExactBridge()

    with pytest.raises(RuntimeError, match="V3_ACTIVE_FEATURE_MISSING: atr_bps"):
        _minimal_v3().build_window(
            end_m1,
            base_m1,
            bridge,
            canonical_v3_window=canonical,
        )

    assert bridge.observed_index is None


def test_v3_rejects_nonfinite_xgb_bridge_before_model_inference() -> None:
    class _NonfiniteBridge:
        def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
            values = np.ones((len(frame), 7), dtype=np.float32)
            values[-1, -1] = np.inf
            return {"signal_bridge_v1": values}

    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    base_m1 = _v3_m1_source(str(end_m1))
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    canonical = pd.DataFrame(
        {"atr_bps": np.ones(len(required_m5))},
        index=required_m5,
    )

    with pytest.raises(RuntimeError, match="V3_XGB_BRIDGE_OUTPUT_INVALID"):
        _minimal_v3().build_window(
            end_m1,
            base_m1,
            _NonfiniteBridge(),
            canonical_v3_window=canonical,
        )


def test_v3_rejects_duplicate_canonical_m5_key() -> None:
    end_m1 = pd.Timestamp("2026-07-16T12:06:00Z")
    base_m1 = _v3_m1_source(str(end_m1))
    required_m5 = required_closed_m5_keys_for_v3_window(end_m1, base_m1)
    duplicate_index = required_m5.insert(1, required_m5[0])
    canonical = pd.DataFrame(
        {"atr_bps": np.ones(len(duplicate_index))},
        index=duplicate_index,
    )

    with pytest.raises(RuntimeError, match="V3_CANONICAL_INDEX_INVALID"):
        _minimal_v3().build_window(
            end_m1,
            base_m1,
            _ExactBridge(),
            canonical_v3_window=canonical,
        )


def test_entry_canonical_requires_the_exact_latest_closed_m5() -> None:
    loader = _CanonicalLoader(
        pd.Timestamp("2026-07-16T11:59:00Z"),
        _canonical_window("2026-07-16T11:55:00Z"),
    )
    pipe = _pipeline(loader)

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:05:00Z"))

    assert raised.value.reason == "entry_latest_closed_m5_unavailable"
    assert raised.value.evidence["expected_m5"] == "2026-07-16 12:00:00+00:00"
    assert loader.requested == []


def test_entry_canonical_accepts_only_an_exact_96_bar_fresh_window() -> None:
    window = _canonical_window("2026-07-16T12:00:00Z")
    loader = _CanonicalLoader(pd.Timestamp("2026-07-16T12:00:00Z"), window)
    pipe = _pipeline(loader)

    pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:06:00Z"))

    assert loader.requested == [
        (pd.Timestamp("2026-07-16T12:00:00Z"), pipeline_module.ENTRY_SEQ_LEN)
    ]
    assert pipe._last_augmented is not None
    assert pipe._last_augmented.index[-1] == pd.Timestamp("2026-07-16T12:00:00Z")


def test_entry_canonical_age_does_not_floor_away_subminute_staleness() -> None:
    loader = _CanonicalLoader(
        pd.Timestamp("2026-07-16T12:00:00Z"),
        _canonical_window("2026-07-16T12:00:00Z"),
    )
    pipe = _pipeline(loader)

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe._refresh_entry_canonical(pd.Timestamp("2026-07-16T12:06:31Z"))

    assert raised.value.reason == "entry_canonical_stale"
    assert raised.value.evidence["canonical_cutoff_age_sec"] == 391.0
    assert loader.requested == []


def test_negative_retired_latency_env_cannot_enable_entry_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EntryMustNotRun:
        def predict_live_bar(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("stale Entry state must never reach model inference")

    pipe = _pipeline(smart_entry=_EntryMustNotRun())
    pipe._last_augmented = _canonical_window("2026-07-16T12:00:00Z")
    monkeypatch.setattr(pipe, "_refresh_entry_canonical", lambda _now: None)
    monkeypatch.setenv("GX1_MAX_ENTRY_DECISION_LATENCY_SEC", "-1")

    with pytest.raises(EntryDecisionUnavailable) as raised:
        pipe.make_entry_decision(
            pd.Timestamp("2026-07-16T12:07:00Z"),
            bid=2400.0,
            ask=2400.2,
        )

    assert raised.value.reason == "entry_signal_stale"
    assert raised.value.evidence["entry_signal_latency_sec"] == 120.0
    assert raised.value.evidence["entry_signal_latency_cap_sec"] == 90.0


def test_closed_m1_does_not_substitute_an_older_cached_or_latest_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-07-16T12:10:00Z")
    expected_path = tmp_path / "xauusd_m1_20260716.parquet"
    expected_path.touch()
    pipe = _pipeline()
    pipe._last_m1_atr_minute = pd.Timestamp("2026-07-16T12:08:00Z")
    pipe._last_m1_bar = {"time": pipe._last_m1_atr_minute, "mid_close": 111.0}

    monkeypatch.setattr(pipeline_module, "COLLECTOR_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: _collector_rows("2026-07-16T12:08:00Z"),
    )

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe._refresh_m1_bar(now)

    assert raised.value.reason == "closed_m1_exact_bar_missing"
    assert raised.value.evidence["expected_m1"] == "2026-07-16 12:09:00+00:00"
    assert raised.value.evidence["latest_observed_m1"] == "2026-07-16 12:08:00+00:00"


def test_closed_m1_selects_the_unique_exact_bar_not_a_forming_later_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2026-07-16T12:10:00Z")
    expected_path = tmp_path / "xauusd_m1_20260716.parquet"
    expected_path.touch()
    frame = _collector_rows(
        "2026-07-16T12:09:00Z",
        "2026-07-16T12:10:00Z",
    )
    monkeypatch.setattr(pipeline_module, "COLLECTOR_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: frame.copy(),
    )

    bar = _pipeline()._refresh_m1_bar(now)

    assert bar["time"] == pd.Timestamp("2026-07-16T12:09:00Z")
    assert bar["bid_close"] == 2400.0
    assert bar["mid_close"] == pytest.approx(2400.1)
    assert bar["atr_bps"] > 0.0


def test_closed_m1_midnight_uses_the_expected_bars_calendar_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_path = tmp_path / "xauusd_m1_20260715.parquet"
    expected_path.touch()
    observed_paths: list[Path] = []

    def fake_read(path: Path, **_kwargs: object) -> pd.DataFrame:
        observed_paths.append(Path(path))
        return _collector_rows("2026-07-15T23:59:00Z")

    monkeypatch.setattr(pipeline_module, "COLLECTOR_DIR", tmp_path)
    monkeypatch.setattr(pipeline_module.pd, "read_parquet", fake_read)

    bar = _pipeline()._refresh_m1_bar(pd.Timestamp("2026-07-16T00:00:00Z"))

    assert observed_paths == [expected_path]
    assert bar["time"] == pd.Timestamp("2026-07-15T23:59:00Z")


def test_closed_m1_provider_is_exact_historical_replay_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.Timestamp("2026-07-16T12:09:00Z")

    class _ExactProvider:
        def __init__(self) -> None:
            self.requested: list[pd.Timestamp] = []

        def get_closed_m1_bar(
            self,
            expected_m1: pd.Timestamp,
        ) -> dict[str, object]:
            self.requested.append(expected_m1)
            return _collector_rows(str(expected_m1)).iloc[0].to_dict()

    provider = _ExactProvider()
    pipe = _pipeline(closed_m1_provider=provider)
    monkeypatch.setattr(
        pipeline_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("historical provider must not read live collector")
        ),
    )

    bar = pipe._refresh_m1_bar(pd.Timestamp("2026-07-16T12:10:00Z"))

    assert provider.requested == [expected]
    assert bar["time"] == expected
    assert bar["mid_close"] == pytest.approx(2400.1)


def test_closed_m1_provider_rejects_nearby_bar_substitution() -> None:
    class _WrongBarProvider:
        def get_closed_m1_bar(
            self,
            _expected_m1: pd.Timestamp,
        ) -> dict[str, object]:
            return _collector_rows("2026-07-16T12:08:00Z").iloc[0].to_dict()

    pipe = _pipeline(closed_m1_provider=_WrongBarProvider())

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe._refresh_m1_bar(pd.Timestamp("2026-07-16T12:10:00Z"))

    assert raised.value.reason == "closed_m1_provider_time_mismatch"
    assert raised.value.evidence["expected_m1"] == "2026-07-16 12:09:00+00:00"
    assert raised.value.evidence["observed_m1"] == "2026-07-16 12:08:00+00:00"


def test_exact_closed_m5_rejects_latest_row_substitution() -> None:
    augmented = pd.DataFrame(
        {"atr_bps": [12.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T11:55:00Z")]),
    )

    with pytest.raises(ExitDecisionUnavailable) as raised:
        _exact_closed_m5_row(augmented, pd.Timestamp("2026-07-16T12:07:00Z"))

    assert raised.value.reason == "canonical_exact_m5_missing"
    assert raised.value.evidence["expected_m5"] == "2026-07-16 12:00:00+00:00"


@pytest.mark.parametrize(
    ("output", "q_head_required", "reason"),
    [
        (None, False, "v3_output_invalid"),
        ({}, False, "v3_output_missing"),
        (
            {
                "v3_v8_should_exit_prob": np.nan,
                "v3_v8_profit_protect_prob": 0.2,
                "v3_v8_family_argmax": 1,
                "v3_v8_family_logit_max": 2.0,
            },
            False,
            "v3_output_non_finite",
        ),
        (
            {
                "v3_v8_should_exit_prob": 0.3,
                "v3_v8_profit_protect_prob": 0.2,
                "v3_v8_family_argmax": 1,
                "v3_v8_family_logit_max": 2.0,
            },
            True,
            "v3_output_missing",
        ),
    ],
)
def test_v3_output_contract_fails_closed(
    output: object,
    q_head_required: bool,
    reason: str,
) -> None:
    with pytest.raises(ExitDecisionUnavailable) as raised:
        _validated_v3_output(output, q_head_required=q_head_required)

    assert raised.value.reason == reason


class _FakeTrade:
    def __init__(self, *, quote_pnl_bps: float = 0.0) -> None:
        self.side = "long"
        self.trade_id = "T-1"
        self.bars_in_trade = 0
        self.current_bid = 2400.0
        self.current_ask = 2400.2
        self.current_pnl_bps = 0.0
        self.cum_mfe_bps = 0.0
        self.cum_mae_bps = 0.0
        self.last_atr_bps = 0.0
        self._quote_pnl_bps = quote_pnl_bps
        self.updated_bar: dict[str, float] | None = None
        self.v3_updates: list[dict[str, object]] = []

    def _pnl_bps(self, _bid: float, _ask: float) -> float:
        return self._quote_pnl_bps

    def clone_for_exit_decision(self) -> "_FakeTrade":
        return deepcopy(self)

    def commit_complete_exit_bar(self, staged: "_FakeTrade") -> None:
        self.__dict__.clear()
        self.__dict__.update(deepcopy(staged.__dict__))

    def update_bar(self, **values: float) -> None:
        self.updated_bar = dict(values)
        self.bars_in_trade += 1
        self.current_bid = float(values["bid"])
        self.current_ask = float(values["ask"])

    def build_v3_overlay(self) -> dict[str, np.ndarray]:
        return {}

    def update_v3(self, output: dict[str, object]) -> None:
        self.v3_updates.append(output)

    def build_v3_tracking_features(self) -> dict[str, float]:
        return {}


class _FailingV3:
    _enable_multi_tf = False
    _enable_q_head = False

    def predict(self, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("broken-v3")


class _CompleteV3:
    _enable_multi_tf = False
    _enable_q_head = False

    def predict(self, **_kwargs: object) -> dict[str, object]:
        return {
            "v3_v8_should_exit_prob": 0.7,
            "v3_v8_profit_protect_prob": 0.4,
            "v3_v8_family_argmax": 1,
            "v3_v8_family_logit_max": 2.0,
        }


class _ExitMustNotRun:
    def __init__(self) -> None:
        self.calls = 0

    def decide_for_trade(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("Exit-IQL must not receive failed V3 state")


def _exact_m1_bar() -> dict[str, object]:
    return {
        "time": pd.Timestamp("2026-07-16T12:06:00Z"),
        "bid_high": 2401.0,
        "bid_low": 2399.0,
        "ask_high": 2401.2,
        "ask_low": 2399.2,
        "bid_close": 2400.0,
        "ask_close": 2400.2,
        "mid_close": 2400.1,
        "atr_bps": 9.16,
    }


def test_v3_failure_never_continues_into_exit_iql_zero_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_time = pd.Timestamp("2026-07-16T12:06:00Z")
    loader = SimpleNamespace(
        _base28=pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex([exact_time])),
        cutoff_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
    )
    exit_iql = _ExitMustNotRun()
    pipe = _pipeline(loader, v3=_FailingV3(), exit_iql=exit_iql)
    pipe._last_exit_augmented = pd.DataFrame(
        {"atr_bps": [12.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T12:00:00Z")]),
    )
    monkeypatch.setattr(pipe, "_refresh_m1_bar", lambda _now: _exact_m1_bar())
    monkeypatch.setattr(pipe, "_refresh_exit_canonical", lambda _now: True)
    trade = _FakeTrade()

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe.make_exit_decision(
            trade,
            pd.Timestamp("2026-07-16T12:07:00Z"),
            bid=2400.3,
            ask=2400.5,
        )

    assert raised.value.reason == "v3_inference_failed"
    assert exit_iql.calls == 0
    assert trade.v3_updates == []
    assert trade.updated_bar is None
    assert trade.bars_in_trade == 0


def test_exit_iql_failure_leaves_trade_state_byte_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_time = pd.Timestamp("2026-07-16T12:06:00Z")
    loader = SimpleNamespace(
        _base28=pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex([exact_time])),
        cutoff_ts=pd.Timestamp("2026-07-16T12:00:00Z"),
    )
    exit_iql = _ExitMustNotRun()
    pipe = _pipeline(loader, v3=_CompleteV3(), exit_iql=exit_iql)
    pipe._last_exit_augmented = pd.DataFrame(
        {"atr_bps": [12.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-07-16T12:00:00Z")]),
    )
    monkeypatch.setattr(pipe, "_refresh_m1_bar", lambda _now: _exact_m1_bar())
    monkeypatch.setattr(pipe, "_refresh_exit_canonical", lambda _now: None)
    trade = _FakeTrade()
    before = deepcopy(trade.__dict__)

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe.make_exit_decision(
            trade,
            pd.Timestamp("2026-07-16T12:07:00Z"),
            bid=2400.3,
            ask=2400.5,
        )

    assert raised.value.reason == "exit_iql_decision_failed"
    assert trade.__dict__ == before


def test_missing_canonical_state_is_unavailable_not_synthetic_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = SimpleNamespace(cutoff_ts=pd.Timestamp("2026-07-16T11:55:00Z"))
    pipe = _pipeline(loader)
    trade = _FakeTrade()
    monkeypatch.setattr(pipe, "_refresh_m1_bar", lambda _now: _exact_m1_bar())

    def _canonical_unavailable(_now: pd.Timestamp) -> None:
        raise ExitDecisionUnavailable(
            "canonical_data_unavailable",
            expected_m5="2026-07-16 12:00:00+00:00",
        )

    monkeypatch.setattr(pipe, "_refresh_exit_canonical", _canonical_unavailable)

    with pytest.raises(ExitDecisionUnavailable) as raised:
        pipe.make_exit_decision(
            trade,
            pd.Timestamp("2026-07-16T12:07:00Z"),
            bid=2400.3,
            ask=2400.5,
        )

    assert raised.value.reason == "canonical_data_unavailable"
    assert raised.value.evidence["expected_m5"] == "2026-07-16 12:00:00+00:00"
    assert trade.bars_in_trade == 0
    assert trade.updated_bar is None


def test_fresh_quote_hard_stop_remains_available_before_model_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = _pipeline()
    trade = _FakeTrade(quote_pnl_bps=-95.0)
    monkeypatch.setattr(pipeline_module, "_EXIT_HARD_STOP_BPS", 80.0)
    monkeypatch.setattr(
        pipe,
        "_refresh_m1_bar",
        lambda _now: (_ for _ in ()).throw(AssertionError("collector must not be read")),
    )

    decision = pipe.make_exit_decision(
        trade,
        pd.Timestamp("2026-07-16T12:07:00Z"),
        bid=2300.0,
        ask=2300.2,
    )

    assert decision["action"] == "EXIT_NOW"
    assert decision["decision_source"] == "HARD_MAE_STOP"
    assert decision["decision_safety_scope"] == "fresh_quote_existing_position_close"
    assert trade.bars_in_trade == 0
    assert trade.current_pnl_bps == -95.0


def test_active_runtime_source_has_no_decision_state_substitution() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline_source = (root / "gx1/execution/v12_pipeline.py").read_text(encoding="utf-8")
    runner_source = (root / "gx1/execution/v12_paper_runner.py").read_text(encoding="utf-8")
    exit_start = pipeline_source.index("    def make_exit_decision(")
    exit_end = pipeline_source.index("\n\n# Backwards-compat", exit_start)
    active_exit = pipeline_source[exit_start:exit_end]

    for forbidden in (
        "using zero fallback",
        "zero-fallback V3 state",
        "Use latest available bar as fallback",
        '"error": "no_canonical_data"',
        "current_m1_atr_bps_override=m1_atr_bps if",
        "v3_v8_out = None",
        "trade.update_bar(bid=bid",
    ):
        assert forbidden not in active_exit

    assert "m1_close = (bid + ask) / 2.0" not in runner_source
    assert ".read_parquet(_p" not in runner_source
    assert 'fill_price = float(order_result.get("fill_price") or 0.0)' not in runner_source
    assert 'float(t.get("currentUnits", 0) or 0)' not in runner_source
    assert 'get_open_trades().get("trades", [])' not in runner_source
    assert "fill_price - spread_abs" not in runner_source
    assert "fill_price + spread_abs" not in runner_source
    assert "except ExitDecisionUnavailable as exc:" in runner_source
    assert '"exit_decision": None' in runner_source
    assert "if exit_decision_unavailable:" in runner_source
    assert "FILLED_STATE_UNAVAILABLE_RECOVERY" in runner_source
    assert "EXIT_CLOSE_FAILED" in runner_source
    assert "EXIT_EXECUTION_UNRESOLVED" in runner_source
    assert "BROKER_RECONCILIATION_REQUIRED" in runner_source


def test_missing_trade_id_fails_closed_without_counter_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner,
        "attempt_market_entry",
        lambda _client, side, units: calls.append((side, units)) or {"status": "filled"},
    )
    trade = SimpleNamespace(trade_id=None, side="long", units=7)

    result = runner.attempt_close_trade(object(), trade)

    assert result["status"] == "missing_trade_id"
    assert result["trade_id"] is None
    assert calls == []


def test_empty_trade_id_fails_closed_without_counter_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner,
        "attempt_market_entry",
        lambda _client, side, units: calls.append((side, units)) or {"status": "filled"},
    )
    trade = SimpleNamespace(trade_id="", side="short", units=3)

    result = runner.attempt_close_trade(object(), trade)

    assert result["status"] == "missing_trade_id"
    assert result["trade_id"] is None
    assert calls == []


def test_runtime_launch_lease_rejects_replacement_and_in_check_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.execution import v12_paper_runner as runner
    from gx1.execution import v12_smart_entry_live as smart_live
    from gx1_guards import artifacts

    state_path = tmp_path / "launch.json"
    registry_path = tmp_path / "registry.json"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    state_path.write_text('{"decision":"ALLOW"}\n', encoding="utf-8")
    registry_path.write_text('{"active":{}}\n', encoding="utf-8")
    monkeypatch.setattr(artifacts, "XAU_DIRECTION_LAUNCH_CONTRACT", state_path)
    monkeypatch.setattr(artifacts, "SELECTION_CONTRACT", registry_path)
    monkeypatch.setattr(smart_live, "assert_smart_serving_gate", lambda: {})
    monkeypatch.setattr(
        artifacts,
        "load_decision_entry",
        lambda _role: {
            "path": bundle,
            "xau_direction_launch_state": {
                "accepted_via_vedtak": {
                    "event_sha256": "a" * 64,
                    "vedtak_id": "UNIT_RUNTIME_LEASE",
                }
            },
        },
    )

    lease = runner.require_runtime_entry_launch_lease()
    state_path.write_text('{"decision":"BLOCK"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="replaced or revoked"):
        runner.require_runtime_entry_launch_lease(expected_lease=lease)

    def mutate_during_check() -> dict:
        registry_path.write_text('{"changed":true}\n', encoding="utf-8")
        return {}

    monkeypatch.setattr(
        smart_live,
        "assert_smart_serving_gate",
        mutate_during_check,
    )
    with pytest.raises(RuntimeError, match="changed during lease"):
        runner.require_runtime_entry_launch_lease()


def test_filled_order_with_missing_price_is_explicitly_incomplete_not_zero() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-1",
                "orderID": "order-1",
                "units": "2",
                "tradeOpened": {"tradeID": "trade-1", "units": "2"},
            }
        }
    )

    result = runner.attempt_market_entry(client, "long", units=2)

    assert result["status"] == "filled"
    assert result["trade_id"] == "trade-1"
    assert result["fill_price"] is None


def test_filled_order_units_must_exactly_match_requested_learned_units() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-2",
                "orderID": "order-2",
                "units": "1",
                "price": "2400.2",
                "tradeOpened": {"tradeID": "trade-2", "units": "1"},
            }
        }
    )

    result = runner.attempt_market_entry(client, "long", units=2)

    assert result["status"] == "filled_units_mismatch"
    assert result["requested_signed_units"] == 2
    assert result["filled_signed_units"] == 1
    assert result["fill_units_exact"] is False


def test_mixed_netting_fill_is_never_accepted_as_new_trade_state() -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(
        create_market_order=lambda *_args, **_kwargs: {
            "orderFillTransaction": {
                "id": "tx-mixed",
                "orderID": "order-mixed",
                "units": "5",
                "price": "2400.2",
                "tradeOpened": {"tradeID": "trade-new", "units": "2"},
                "tradesClosed": [{"tradeID": "trade-old", "units": "-3"}],
            }
        }
    )

    result = runner.attempt_market_entry(client, "long", units=5)

    assert result["status"] == "filled_structure_mismatch"
    assert result["fill_units_exact"] is True
    assert result["pure_trade_open"] is False


def _runtime_sizing_authority_for_broker_fact_tests():
    import json

    from gx1.contracts.entry_model_native_sizing_authority_v1 import (
        ValidatedLearnedSizingAuthority,
    )

    return ValidatedLearnedSizingAuthority(
        authority_json="{}",
        adoption_json="{}",
        calibration_json=json.dumps(
            {
                "instrument_constraints": {
                    "instrument": "XAU_USD",
                    "account_currency": "USD",
                    "quote_currency": "USD",
                    "unit_step": 1,
                    "minimum_order_units": 1,
                    "maximum_gross_xau_units": 1000,
                    "margin_rate": 0.05,
                }
            }
        ),
        proof_json="{}",
        joint_proof_json="{}",
        active_exit_registry_projection_json="{}",
        content_hash_key=(),
        file_stats=(),
    )


def _broker_fact_client(
    *,
    hedging_enabled: bool,
    transaction_ids: tuple[str, str, str],
    trades: list[dict] | None = None,
):
    account_tx, instrument_tx, exposure_tx = transaction_ids
    return SimpleNamespace(
        get_account_summary=lambda: {
            "account": {
                "currency": "USD",
                "hedgingEnabled": hedging_enabled,
                "NAV": "10000",
                "balance": "10000",
                "marginAvailable": "1000",
                "marginUsed": "0",
            },
            "lastTransactionID": account_tx,
        },
        get_account_instruments=lambda _instruments: {
            "instruments": [
                {
                    "name": "XAU_USD",
                    "tradeUnitsPrecision": 0,
                    "minimumTradeSize": "1",
                    "maximumOrderUnits": "100000",
                    "marginRate": "0.05",
                }
            ],
            "lastTransactionID": instrument_tx,
        },
        get_open_trades=lambda: {
            "trades": [] if trades is None else trades,
            "lastTransactionID": exposure_tx,
        },
    )


def test_live_sizing_requires_one_coherent_hedging_broker_snapshot() -> None:
    from gx1.execution import v12_paper_runner as runner

    constraints = runner.learned_sizing_runtime_constraints(
        _broker_fact_client(
            hedging_enabled=True,
            transaction_ids=("9001", "9001", "9001"),
        ),
        bid=2400.0,
        ask=2400.2,
        validated_authority=_runtime_sizing_authority_for_broker_fact_tests(),
    )

    assert constraints["account_last_transaction_id"] == "9001"
    assert constraints["instrument_last_transaction_id"] == "9001"
    assert constraints["exposure_last_transaction_id"] == "9001"


@pytest.mark.parametrize(
    ("hedging_enabled", "transaction_ids", "match"),
    [
        (False, ("9001", "9001", "9001"), "hedgingEnabled=true"),
        (True, ("9001", "9002", "9001"), "different lastTransactionID"),
    ],
)
def test_live_sizing_rejects_netting_or_torn_broker_snapshot(
    hedging_enabled: bool,
    transaction_ids: tuple[str, str, str],
    match: str,
) -> None:
    from gx1.execution import v12_paper_runner as runner

    with pytest.raises(RuntimeError, match=match):
        runner.learned_sizing_runtime_constraints(
            _broker_fact_client(
                hedging_enabled=hedging_enabled,
                transaction_ids=transaction_ids,
            ),
            bid=2400.0,
            ask=2400.2,
            validated_authority=_runtime_sizing_authority_for_broker_fact_tests(),
        )


def test_live_entry_reconciles_exact_broker_and_local_xau_trade_ids() -> None:
    from gx1.execution import v12_paper_runner as runner

    empty_client = _broker_fact_client(
        hedging_enabled=True,
        transaction_ids=("9001", "9001", "9001"),
    )
    assert runner.require_broker_xau_trade_reconciliation(
        empty_client,
        local_open_trades=[],
        max_trades=1,
        expected_exposure_transaction_id="9001",
    ) == ()

    broker_trade = {
        "id": "77",
        "instrument": "XAU_USD",
        "currentUnits": "3",
    }
    orphan_client = _broker_fact_client(
        hedging_enabled=True,
        transaction_ids=("9001", "9001", "9001"),
        trades=[broker_trade],
    )
    with pytest.raises(RuntimeError, match="broker/local XAU trade identity mismatch"):
        runner.require_broker_xau_trade_reconciliation(
            orphan_client,
            local_open_trades=[],
            max_trades=1,
            expected_exposure_transaction_id="9001",
        )
    with pytest.raises(RuntimeError, match="at the admitted cap"):
        runner.require_broker_xau_trade_reconciliation(
            orphan_client,
            local_open_trades=[SimpleNamespace(trade_id="77")],
            max_trades=1,
            expected_exposure_transaction_id="9001",
        )
    with pytest.raises(RuntimeError, match="exposure changed"):
        runner.require_broker_xau_trade_reconciliation(
            empty_client,
            local_open_trades=[],
            max_trades=1,
            expected_exposure_transaction_id="9000",
        )


@pytest.mark.parametrize(
    "quote",
    [
        {"bids": [{"price": "2400.0"}], "asks": [{"price": "2400.2"}]},
        {
            "time": "2026-07-16T12:00:00Z",
            "bids": [{"price": "2400.0"}],
            "asks": [{"price": "2399.9"}],
        },
    ],
)
def test_quote_missing_time_or_valid_bid_ask_contract_fails_closed(quote: dict) -> None:
    from gx1.execution import v12_paper_runner as runner

    client = SimpleNamespace(get_pricing=lambda _instruments: {"prices": [quote]})

    with pytest.raises(ValueError):
        runner.get_current_spread_bps(
            client,
            now_utc=pd.Timestamp("2026-07-16T12:00:30Z").to_pydatetime(),
        )


def test_raw_base28_frame_contains_only_exact_native_m1_identity() -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    timestamp = pd.Timestamp("2026-07-16T12:04:00Z")
    m1 = pd.DataFrame(
        {
            column: [2400.0 + offset]
            for offset, column in enumerate(
                incremental.M1_MARKET_IDENTITY_COLUMNS
            )
        },
        index=pd.DatetimeIndex([timestamp]),
    )

    m1["stale_context"] = 999.0
    frame = incremental._build_raw_base28_owned_frame(m1)

    assert tuple(frame.columns) == incremental.RAW_BASE28_COLUMNS
    pd.testing.assert_frame_equal(
        frame,
        m1.loc[:, list(incremental.RAW_BASE28_COLUMNS)].rename_axis("time"),
    )


def test_raw_base28_rejects_missing_native_m1_field() -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    timestamp = pd.Timestamp("2026-07-16T12:04:00Z")
    m1 = pd.DataFrame(
        {
            column: [2400.0 + offset]
            for offset, column in enumerate(incremental.RAW_BASE28_COLUMNS)
            if column != "ask_close"
        },
        index=pd.DatetimeIndex([timestamp]),
    )

    with pytest.raises(RuntimeError, match="RAW_BASE28_M1_FIELDS_MISSING"):
        incremental._build_raw_base28_owned_frame(m1)


@pytest.mark.parametrize(
    ("volume", "error_code"),
    [
        (0.0, "PLUS5_VOLUME_INVALID"),
        (np.nan, "PLUS5_SOURCE_NONFINITE"),
    ],
)
def test_plus5_rejects_unobserved_volume_instead_of_using_one(
    volume: float,
    error_code: str,
) -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    frame = pd.DataFrame(
        {
            "open": [2400.0, 2400.5],
            "high": [2401.0, 2401.5],
            "low": [2399.0, 2399.5],
            "close": [2400.5, 2401.0],
            "volume": [10.0, volume],
        }
    )

    with pytest.raises(RuntimeError, match=error_code):
        incremental._compute_plus5_features(frame)


def test_plus5_rejects_missing_volume_source() -> None:
    from gx1.execution import v12_canonical_incremental as incremental

    frame = pd.DataFrame(
        {
            "open": [2400.0],
            "high": [2401.0],
            "low": [2399.0],
            "close": [2400.5],
        }
    )

    with pytest.raises(RuntimeError, match="PLUS5_SOURCE_MISSING"):
        incremental._compute_plus5_features(frame)


def test_plus5_serve_owner_rejects_zero_volume_instead_of_using_one() -> None:
    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader

    frame = pd.DataFrame(
        {
            "open": [2400.0, 2400.5],
            "high": [2401.0, 2401.5],
            "low": [2399.0, 2399.5],
            "close": [2400.5, 2401.0],
            "volume": [10.0, 0.0],
        }
    )

    with pytest.raises(RuntimeError, match="PLUS5_VOLUME_INVALID"):
        PrebuiltStateLoader()._augment_cv3_with_v1_legacy(frame)


def test_plus5_build_and_serve_delegate_to_identical_formula_owner() -> None:
    from gx1.execution import v12_canonical_incremental as incremental
    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
    from gx1.features.basic_v1 import PLUS5_FEATURES

    n = 64
    close = 2400.0 + np.linspace(0.0, 3.0, n)
    frame = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(10.0, 100.0, n),
        }
    )

    built = incremental._compute_plus5_features(frame)
    served = PrebuiltStateLoader()._augment_cv3_with_v1_legacy(frame)

    pd.testing.assert_frame_equal(
        built[list(PLUS5_FEATURES)],
        served[list(PLUS5_FEATURES)],
    )
