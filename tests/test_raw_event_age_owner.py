"""Raw native-clock event/state age owner and active-source guards."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from gx1.features import event_age_v1
from gx1.features import htf_features
from gx1.features import entry_model_native_feature_layers_v1
from gx1.features import regime_v4_features


def test_raw_event_age_has_honest_prefix_and_no_cap() -> None:
    event = np.zeros(1_205, dtype=np.bool_)
    event[3] = True
    event[1_101] = True
    age = event_age_v1.raw_event_age_bars(event)
    np.testing.assert_array_equal(age[:4], [np.nan, np.nan, np.nan, 0.0])
    assert age[1_003] == 1_000.0
    assert age[1_100] == 1_097.0
    assert age[1_101] == 0.0
    assert age[-1] == 103.0


def test_raw_event_age_chunk_carry_matches_one_shot_exactly() -> None:
    event = np.zeros(1_000, dtype=np.bool_)
    event[[7, 709]] = True
    expected = event_age_v1.raw_event_age_bars(event)
    left = event_age_v1.raw_event_age_bars(event[:503])
    right = event_age_v1.raw_event_age_bars(
        event[503:], initial_age_bars=int(left[-1])
    )
    np.testing.assert_array_equal(np.concatenate((left, right)), expected)


def test_raw_state_age_is_uncapped_and_resets_only_on_real_change() -> None:
    state = np.r_[np.full(702, np.nan), np.zeros(802), np.ones(3)]
    age = event_age_v1.raw_state_age_bars(state)
    assert np.isnan(age[:702]).all()
    assert age[702] == 0.0
    assert age[1_503] == 801.0
    np.testing.assert_array_equal(age[-3:], [0.0, 1.0, 2.0])


def test_scalar_event_age_rejects_future_and_preserves_unknown() -> None:
    assert np.isnan(event_age_v1.raw_event_age_from_last_observed_row(9, None))
    assert event_age_v1.raw_event_age_from_last_observed_row(901, 3) == 898.0
    with pytest.raises(RuntimeError, match="LAST_EVENT_ROW_INVALID"):
        event_age_v1.raw_event_age_from_last_observed_row(3, 4)


def test_active_age_sources_forbid_retired_names_caps_and_log_routes() -> None:
    owner_source = inspect.getsource(event_age_v1)
    active_source = "\n".join(
        (
            inspect.getsource(htf_features.compute_per_bar_features_v4),
            inspect.getsource(
                entry_model_native_feature_layers_v1.build_price_derived_layer
            ),
            inspect.getsource(regime_v4_features),
        )
    )
    for token in (
        "age_norm",
        "trend_age_bars_norm",
        "bars_since_d1_regime_change_v3",
        "d1_dist_roc_288_v3",
        "clip(-5.0, 5.0)",
        "clip(0, 500)",
        "minimum(age, 500)",
        "min(age, 500)",
        "log1p(age)",
    ):
        assert token not in active_source
    for token in ("np.clip", "np.log1p", "math.log1p", "500"):
        assert token not in owner_source
