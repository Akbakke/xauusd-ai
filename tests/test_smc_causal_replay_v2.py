from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features import smc_v1 as smc


def _random_ohlc(rows: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 2_000.0 + np.cumsum(rng.normal(0.0, 1.0, rows))
    open_ = close + rng.normal(0.0, 0.25, rows)
    high = np.maximum(open_, close) + rng.uniform(0.05, 2.0, rows)
    low = np.minimum(open_, close) - rng.uniform(0.05, 2.0, rows)
    return high, low, close


def _frame(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "atr": np.ones(len(close), dtype=np.float64),
        }
    )


def _fixed_high_pivots(monkeypatch, rows: int, indices: tuple[int, ...]) -> None:
    def fixed_pivots(_high, _low, _lookback):
        high_mask = np.zeros(rows, dtype=bool)
        low_mask = np.zeros(rows, dtype=bool)
        high_mask[list(indices)] = True
        return high_mask, low_mask

    monkeypatch.setattr(smc, "_detect_swing_pivots", fixed_pivots)


def test_replay_is_prefix_causal_and_chunk_carry_exact() -> None:
    high, low, close = _random_ohlc(257, seed=91)
    full, full_state = smc.replay_smc_causal_structure_v2(high, low, close)

    state = None
    chunks: list[dict[str, np.ndarray]] = []
    cuts = (0, 1, 7, 43, 91, 180, 257)
    for start, stop in zip(cuts, cuts[1:]):
        observed, state = smc.replay_smc_causal_structure_v2(
            high[start:stop],
            low[start:stop],
            close[start:stop],
            state=state,
        )
        chunks.append(observed)
    assert state == full_state
    for name, expected in full.items():
        actual = np.concatenate([chunk[name] for chunk in chunks])
        np.testing.assert_array_equal(actual, expected)

    for cutoff in (9, 64, 129, 211):
        prefix, _ = smc.replay_smc_causal_structure_v2(
            high[:cutoff], low[:cutoff], close[:cutoff]
        )
        for name, expected in full.items():
            np.testing.assert_array_equal(prefix[name], expected[:cutoff])

    cutoff = 133
    future_high = high.copy()
    future_low = low.copy()
    future_high[cutoff:] += 50.0
    future_low[cutoff:] -= 50.0
    mutated, _ = smc.replay_smc_causal_structure_v2(
        future_high, future_low, close
    )
    for name, expected in full.items():
        np.testing.assert_array_equal(mutated[name][:cutoff], expected[:cutoff])


def test_choch_is_confirmed_opposing_close_break_not_structure_label_flip() -> None:
    bars = np.asarray(
        [
            (105.0, 104.0, 104.5),
            (110.0, 106.0, 108.0),
            (106.5, 105.8, 106.0),
            (102.0, 100.0, 101.0),
            (107.0, 102.5, 105.0),
            (115.0, 107.0, 112.0),
            (109.0, 106.0, 107.0),
            (108.0, 105.0, 106.0),
            (110.0, 105.5, 107.0),
            (112.0, 106.0, 109.0),
            (108.0, 103.0, 105.0),
            (107.0, 103.5, 105.0),
            (104.0, 98.0, 99.0),
        ],
        dtype=np.float64,
    )
    replay, _ = smc.replay_smc_causal_structure_v2(
        bars[:, 0], bars[:, 1], bars[:, 2], swing_lookback=1
    )
    assert replay["swing_state"][8] == 0
    assert replay["swing_state"][10] == 2
    assert replay["swing_state"][11] == 3
    assert replay["choch_down"][11] == 0.0
    assert replay["bos_down"][12] == 1.0
    assert replay["choch_down"][12] == 1.0
    assert replay["choch_up"].sum() + replay["choch_down"].sum() == 1.0

    # Mutation: equality is not a close-through break.
    equal_close = bars[:, 2].copy()
    equal_close[12] = 103.0
    equal_high = bars[:, 0].copy()
    equal_high[12] = max(equal_high[12], equal_close[12])
    mutated, _ = smc.replay_smc_causal_structure_v2(
        equal_high, bars[:, 1], equal_close, swing_lookback=1
    )
    assert mutated["bos_down"][12] == 0.0
    assert mutated["choch_down"][12] == 0.0


def test_sweep_identity_is_one_shot_and_equal_price_new_pivot_rearms(
    monkeypatch,
) -> None:
    high = np.asarray([101, 105, 103, 106, 104, 105, 104, 106, 107, 107], dtype=float)
    low = np.full(len(high), 99.0)
    close = np.asarray([100, 104, 102, 104, 103, 104, 103, 104, 106, 104], dtype=float)
    _fixed_high_pivots(monkeypatch, len(high), (1, 5))

    replay, state = smc.replay_smc_causal_structure_v2(
        high, low, close, swing_lookback=1
    )
    np.testing.assert_array_equal(
        replay["sweep_up_event"],
        np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=float),
    )
    np.testing.assert_array_equal(
        replay["sweep_event_age_bars"],
        np.asarray([np.nan, np.nan, np.nan, 0, 1, 2, 3, 0, 1, 2], dtype=float),
    )
    # Equal price, distinct confirmed pivot index: identity rearms at row 6.
    assert state.consumed_sweep_high == ("high", 5)
    # Row 8 closes through that new level; row 9 is therefore not a sweep of
    # an invalidated structure even though its wick/close straddle 105.
    assert replay["bos_up"][8] == 1.0
    assert replay["sweep_up"][9] == 0.0


def test_sweep_false_gap_and_touch_do_not_rearm_same_level(monkeypatch) -> None:
    high = np.asarray([101, 105, 103, 106, 105, 107], dtype=float)
    low = np.full(len(high), 99.0)
    close = np.asarray([100, 104, 102, 104, 104, 104], dtype=float)
    _fixed_high_pivots(monkeypatch, len(high), (1,))

    replay, _ = smc.replay_smc_causal_structure_v2(
        high, low, close, swing_lookback=1
    )
    assert replay["sweep_up"].tolist() == [0, 0, 0, 1, 0, 1]
    assert replay["sweep_up_event"].tolist() == [0, 0, 0, 1, 0, 0]
    assert replay["sweep_event_age_bars"].tolist()[-3:] == [0, 1, 2]

    wick_touch = high.copy()
    wick_touch[3] = 105.0
    mutated, _ = smc.replay_smc_causal_structure_v2(
        wick_touch, low, close, swing_lookback=1
    )
    assert mutated["sweep_up_event"][3] == 0.0
    assert mutated["sweep_up_event"][5] == 1.0

    close_through = close.copy()
    close_through[3] = 105.5
    through_high = high.copy()
    through_high[3] = 106.0
    mutated, _ = smc.replay_smc_causal_structure_v2(
        through_high, low, close_through, swing_lookback=1
    )
    assert mutated["bos_up"][3] == 1.0
    assert mutated["sweep_up_event"][3] == 0.0
    assert mutated["sweep_up_event"][5] == 0.0


def test_raw_age_seen_and_sided_depths_preserve_double_sweep(
    monkeypatch,
) -> None:
    high = np.asarray([101, 105, 103, 106, 104, 104], dtype=float)
    low = np.asarray([99, 95, 97, 94, 96, 96], dtype=float)
    close = np.asarray([100, 100, 100, 100, 100, 100], dtype=float)

    def fixed_pivots(_high, _low, _lookback):
        high_mask = np.zeros(len(high), dtype=bool)
        low_mask = np.zeros(len(high), dtype=bool)
        high_mask[1] = True
        low_mask[1] = True
        return high_mask, low_mask

    monkeypatch.setattr(smc, "_detect_swing_pivots", fixed_pivots)
    local = smc.compute_smc_features(
        _frame(high, low, close),
        swing_lookback=1,
        include_v30_additions=True,
    )
    np.testing.assert_array_equal(
        local["smc_sweep_event_age_bars"], [np.nan, np.nan, np.nan, 0, 1, 2]
    )
    assert local.loc[3, "smc_sweep_up_event"] == 1.0
    assert local.loc[3, "smc_sweep_down_event"] == 1.0
    assert local.loc[3, "smc_sweep_up_depth_atr"] == 1.0
    assert local.loc[3, "smc_sweep_down_depth_atr"] == 1.0


def test_local_and_mtf_surfaces_are_projections_of_same_native_replay() -> None:
    high, low, close = _random_ohlc(2_000, seed=818)
    frame = _frame(high, low, close)
    local = smc.compute_smc_features(frame, include_v30_additions=True)
    mtf = smc.compute_smc_mtf_primitives_v1(frame)
    valid = np.isfinite(mtf["mtf_smc_pivot_envelope_position"].to_numpy())
    assert valid.any()

    for local_name, mtf_name in (
        ("smc_bos_up", "mtf_smc_bos_up"),
        ("smc_bos_down", "mtf_smc_bos_down"),
        # The MTF spelling gained a ``_state`` suffix on 2026-08-18; the
        # local twin keeps its name until the contract commit renames it, and
        # the values must remain identical across the rename.
        ("smc_sweep_up", "mtf_smc_sweep_up_state"),
        ("smc_sweep_down", "mtf_smc_sweep_down_state"),
        ("smc_sweep_up_depth_atr", "mtf_smc_sweep_up_depth_atr"),
        ("smc_sweep_down_depth_atr", "mtf_smc_sweep_down_depth_atr"),
        ("smc_sweep_up_event", "mtf_smc_sweep_up_event"),
        ("smc_sweep_down_event", "mtf_smc_sweep_down_event"),
        ("smc_bos_displacement_atr", "mtf_smc_bos_displacement_atr"),
        ("smc_pivot_envelope_position", "mtf_smc_pivot_envelope_position"),
        ("smc_sweep_event_age_bars", "mtf_smc_sweep_event_age_bars"),
    ):
        np.testing.assert_array_equal(
            local[local_name].to_numpy()[valid],
            mtf[mtf_name].to_numpy()[valid],
        )
    local_choch = local["smc_choch"].to_numpy()[valid]
    mtf_choch = (
        mtf["mtf_smc_choch_up"].to_numpy()[valid]
        + mtf["mtf_smc_choch_down"].to_numpy()[valid]
    )
    np.testing.assert_array_equal(local_choch, mtf_choch)

    for name in (
        "mtf_smc_bos_up",
        "mtf_smc_bos_down",
        "mtf_smc_choch_up",
        "mtf_smc_choch_down",
        "mtf_smc_sweep_up_event",
        "mtf_smc_sweep_down_event",
    ):
        assert float(mtf[name].sum()) > 0.0, name


def test_v30_wave2_geometry_retirements_are_absent_and_exactly_recoverable() -> None:
    """The three retired geometry columns are gone AND their algebra survives.

    Every ``not in`` below fails against the pre-2026-08-18 owner, where all
    three names are declared and written -- that is what makes this a
    regression test and not a restatement. The identities are the CLAUDE.md
    rule 4 half: each retired column is reproduced exactly from two columns
    that stay in the SAME tuple, hence in the same specialist projection.
    """

    retired = (
        "mtf_geometry_support_break_displacement_atr",
        "mtf_geometry_resistance_break_displacement_atr",
        "mtf_geometry_nearest_level_abs_atr",
    )
    for name in retired:
        assert name not in smc.SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
        assert name not in smc.SMC_MTF_FEATURE_NAMES_V1

    high, low, close = _random_ohlc(2_000, seed=414)
    mtf = smc.compute_smc_mtf_primitives_v1(_frame(high, low, close))
    for name in retired:
        assert name not in mtf.columns

    support = mtf["mtf_geometry_support_dist_atr"].to_numpy(dtype=np.float64)
    resistance = mtf["mtf_geometry_resistance_dist_atr"].to_numpy(
        dtype=np.float64
    )
    live = np.isfinite(support) & np.isfinite(resistance)
    assert live.any()

    # The fixture must actually exercise the ReLU kink on both sides, or the
    # recovery would be proven only on the trivial branch.
    assert (support[live] < 0.0).any() and (support[live] > 0.0).any()
    assert (resistance[live] < 0.0).any() and (resistance[live] > 0.0).any()

    support_break = np.maximum(-support[live], 0.0)
    resistance_break = np.maximum(-resistance[live], 0.0)
    nearest = np.minimum(np.abs(support[live]), np.abs(resistance[live]))
    assert (support_break >= 0.0).all() and (support_break > 0.0).any()
    assert (resistance_break >= 0.0).all() and (resistance_break > 0.0).any()
    assert (nearest >= 0.0).all()
    # Recovered break displacement is non-zero exactly while price sits
    # through the level -- the persistent state the retired column encoded.
    np.testing.assert_array_equal(support_break > 0.0, support[live] < 0.0)
    np.testing.assert_array_equal(
        resistance_break > 0.0, resistance[live] < 0.0
    )
    np.testing.assert_array_equal(
        nearest,
        np.where(
            np.abs(support[live]) <= np.abs(resistance[live]),
            np.abs(support[live]),
            np.abs(resistance[live]),
        ),
    )


def test_v30_wave2_sweep_state_rename_keeps_values_and_drops_old_spelling() -> None:
    """The two per-bar sweep conditions are renamed, not changed.

    ``mtf_smc_sweep_{up,down}`` re-assert on every qualifying bar while the
    ``_event`` siblings fire once per confirmed pivot identity; the ``_state``
    suffix is the only thing that separates the two readings in the name.
    The absence assertions fail against the pre-rename owner.
    """

    assert "mtf_smc_sweep_up" not in smc.SMC_MTF_FEATURE_NAMES_V1
    assert "mtf_smc_sweep_down" not in smc.SMC_MTF_FEATURE_NAMES_V1
    assert "mtf_smc_sweep_up_state" in smc.SMC_MTF_FEATURE_NAMES_V1
    assert "mtf_smc_sweep_down_state" in smc.SMC_MTF_FEATURE_NAMES_V1
    # Position is preserved, so the per-TF column order ahead of the rename is
    # byte-stable and only the label moved.
    assert smc.SMC_MTF_FEATURE_NAMES_V1.index("mtf_smc_sweep_up_state") == 5
    assert smc.SMC_MTF_FEATURE_NAMES_V1.index("mtf_smc_sweep_down_state") == 6

    high, low, close = _random_ohlc(2_000, seed=414)
    frame = _frame(high, low, close)
    mtf = smc.compute_smc_mtf_primitives_v1(frame)
    local = smc.compute_smc_features(frame, include_v30_additions=True)
    assert "mtf_smc_sweep_up" not in mtf.columns
    assert "mtf_smc_sweep_down" not in mtf.columns

    valid = np.isfinite(mtf["mtf_smc_pivot_envelope_position"].to_numpy())
    for local_name, state_name, event_name in (
        ("smc_sweep_up", "mtf_smc_sweep_up_state", "mtf_smc_sweep_up_event"),
        (
            "smc_sweep_down",
            "mtf_smc_sweep_down_state",
            "mtf_smc_sweep_down_event",
        ),
    ):
        state = mtf[state_name].to_numpy()
        event = mtf[event_name].to_numpy()
        np.testing.assert_array_equal(
            local[local_name].to_numpy()[valid], state[valid]
        )
        # The rename is honest: the state genuinely repeats where the event
        # does not, so the two are not two spellings of one column.
        assert np.logical_or(~(event[valid] > 0.0), state[valid] > 0.0).all()
        assert (state[valid] > 0.0).sum() > (event[valid] > 0.0).sum()


def test_equal_width_envelope_is_honestly_unavailable(monkeypatch) -> None:
    rows = 9
    high = np.full(rows, 100.0)
    low = np.full(rows, 100.0)
    close = np.full(rows, 100.0)

    def fixed_pivots(_high, _low, _lookback):
        high_mask = np.zeros(rows, dtype=bool)
        low_mask = np.zeros(rows, dtype=bool)
        high_mask[[1, 3]] = True
        low_mask[[2, 4]] = True
        return high_mask, low_mask

    monkeypatch.setattr(smc, "_detect_swing_pivots", fixed_pivots)
    frame = _frame(high, low, close)
    local = smc.compute_smc_features(frame, swing_lookback=1)
    mtf = smc.compute_smc_mtf_primitives_v1(frame, swing_lookback=1)

    assert local["smc_pivot_envelope_position"].isna().all()
    assert mtf["mtf_smc_pivot_envelope_position"].isna().all()


def test_replay_state_and_owner_source_guards_fail_closed() -> None:
    high, low, close = _random_ohlc(30, seed=5)
    _, state = smc.replay_smc_causal_structure_v2(high, low, close)
    with pytest.raises(RuntimeError, match="SMC_REPLAY_STATE_CONTRACT_INVALID"):
        smc.replay_smc_causal_structure_v2(
            high[:1],
            low[:1],
            close[:1],
            state=replace(state, schema_version="smc_causal_replay_v1"),
        )
    with pytest.raises(RuntimeError, match="SMC_REPLAY_LEVEL_IDENTITY_INVALID"):
        smc.replay_smc_causal_structure_v2(
            high[:1],
            low[:1],
            close[:1],
            state=replace(state, consumed_sweep_high=("low", 1)),
        )
    with pytest.raises(RuntimeError, match="SMC_REPLAY_STATE_CONTRACT_INVALID"):
        smc.replay_smc_causal_structure_v2(
            high[:1],
            low[:1],
            close[:1],
            state=replace(state, high_tail=state.high_tail[1:]),
        )

    module_source = inspect.getsource(smc)
    assert ".resample(" not in module_source
    assert "prev_cond_sweep" not in module_source
    assert "replay_smc_causal_structure_v2(" in inspect.getsource(
        smc.compute_smc_features
    )
    assert "replay_smc_causal_structure_v2(" in inspect.getsource(
        smc.compute_smc_mtf_primitives_v1
    )
    for retired in (
        "smc_sweep_size_atr",
        "smc_bars_since_sweep\"",
        "smc_bars_since_sweep_norm",
        "smc_premium_discount",
        "mtf_smc_bars_since_sweep_event_norm",
    ):
        assert retired not in module_source
    assert "np.clip(" not in inspect.getsource(smc.compute_smc_features)
    assert "np.clip(" not in inspect.getsource(smc.compute_smc_mtf_primitives_v1)


def test_retired_smc_aliases_have_no_active_producer_or_consumer() -> None:
    repo = Path(__file__).resolve().parents[1]
    active_paths = (
        "gx1/features/smc_v1.py",
        "gx1/scripts/materialize_canonical_v3_augment.py",
        "gx1/execution/v12_canonical_incremental.py",
        "gx1/scripts/build_entry_exit_m1_enriched_frame_v1.py",
        "gx1/scripts/augment_forward_outcome_v2.py",
    )
    active_source = "\n".join(
        (repo / relative).read_text(encoding="utf-8") for relative in active_paths
    )
    for retired in (
        "smc_sweep_size_atr",
        '"smc_bars_since_sweep"',
        "smc_bars_since_sweep_norm",
        "smc_premium_discount",
        "smc_premium_state",
        "mtf_smc_bars_since_sweep_event_norm",
        "mtf_smc_premium_discount",
    ):
        assert retired not in active_source
    owner_source = (repo / "gx1/features/smc_v1.py").read_text(encoding="utf-8")
    for retired_formula in ("999", "log1p", "np.clip("):
        assert retired_formula not in owner_source


def test_five_local_additions_are_an_active_model_native_contract() -> None:
    from gx1.features.entry_model_native_feature_layers_v1 import (
        MODEL_NATIVE_SPECIALIST_LAYER_FEATURES,
        SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES,
        build_smc_local_event_layer,
    )

    assert SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES == smc.SMC_V30_ADDITION_NAMES_V1
    assert (
        "smc_local_event_layer",
        smc.SMC_V30_ADDITION_NAMES_V1,
    ) in MODEL_NATIVE_SPECIALIST_LAYER_FEATURES
    assert "include_v30_additions=True" in inspect.getsource(
        build_smc_local_event_layer
    )
