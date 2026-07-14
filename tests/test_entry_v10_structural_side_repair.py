import numpy as np

from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _apply_structural_side_repair,
    _apply_structural_utility_repair,
    _position_size_target_from_repaired_path,
    _repaired_scalar_bad_path_target,
)


def test_structural_side_repair_removes_wrong_side_trap_targets() -> None:
    direction = np.array([1, 1, -1, 0, 0, -1, 0, 1], dtype=np.int32)
    harvest = direction.copy()

    rising_channel_support_touch = np.zeros_like(direction, dtype=bool)
    countertrend_short_trap = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    support_retest_continuation = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    falling_channel_resistance_touch = np.zeros_like(direction, dtype=bool)
    countertrend_long_trap = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
    resistance_retest_continuation = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)

    repaired_direction, repaired_harvest, masks = _apply_structural_side_repair(
        direction,
        harvest,
        rising_channel_support_touch,
        countertrend_short_trap,
        support_retest_continuation,
        falling_channel_resistance_touch,
        countertrend_long_trap,
        resistance_retest_continuation,
    )

    assert repaired_direction.tolist() == [0, -1, -1, 1, -1, -1, 0, 1]
    assert repaired_harvest.tolist() == [0, -1, -1, 1, -1, -1, 0, 1]
    assert masks["short_to_long"].tolist() == [True, False, False, False, False, False, False, False]
    assert masks["short_to_flat"].tolist() == [False, True, False, False, False, False, False, False]
    assert masks["long_to_short"].tolist() == [False, False, False, True, False, False, False, False]
    assert masks["long_to_flat"].tolist() == [False, False, False, False, True, False, False, False]


def test_structural_side_repair_removes_wrong_side_rising_support_targets() -> None:
    direction = np.array([1, 1, -1, 0, 0, -1], dtype=np.int32)
    harvest = direction.copy()

    rising_channel_support_touch = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    support_retest_continuation = np.array([1, 0, 1, 0, 0, 0], dtype=bool)
    falling_channel_resistance_touch = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    resistance_retest_continuation = np.array([0, 0, 0, 1, 0, 1], dtype=bool)
    countertrend_short_trap = np.zeros_like(direction, dtype=bool)
    countertrend_long_trap = np.zeros_like(direction, dtype=bool)

    repaired_direction, repaired_harvest, masks = _apply_structural_side_repair(
        direction,
        harvest,
        rising_channel_support_touch,
        countertrend_short_trap,
        support_retest_continuation,
        falling_channel_resistance_touch,
        countertrend_long_trap,
        resistance_retest_continuation,
    )

    assert repaired_direction.tolist() == [0, -1, 0, 1, -1, 1]
    assert repaired_harvest.tolist() == [0, -1, 0, 1, -1, 1]
    assert not np.any(repaired_direction[rising_channel_support_touch] == 1)
    assert not np.any(repaired_direction[falling_channel_resistance_touch] == 0)
    assert masks["rising_support_pocket"].tolist() == [True, True, True, False, False, False]
    assert masks["falling_resistance_pocket"].tolist() == [False, False, False, True, True, True]


def test_structural_utility_repair_pushes_wrong_side_utility_below_allowed_side() -> None:
    long_util = np.array([-40.0, 80.0, -120.0, -20.0, 15.0], dtype=np.float32)
    short_util = np.array([-2.0, -90.0, -10.0, 30.0, 10.0], dtype=np.float32)
    long_bad = np.zeros(5, dtype=np.float32)
    short_bad = np.zeros(5, dtype=np.float32)
    long_mae = np.array([4.0, 2.0, 10.0, 3.0, 5.0], dtype=np.float32)
    short_mae = np.array([1.0, 8.0, 3.0, 1.0, 6.0], dtype=np.float32)
    anti_short = np.array([1, 1, 0, 0, 1], dtype=bool)
    anti_long = np.array([0, 0, 1, 1, 1], dtype=bool)

    (
        fixed_long_util,
        fixed_short_util,
        fixed_long_bad,
        fixed_short_bad,
        fixed_long_mae,
        fixed_short_mae,
        masks,
    ) = _apply_structural_utility_repair(
        long_util,
        short_util,
        long_bad,
        short_bad,
        long_mae,
        short_mae,
        anti_short,
        anti_long,
        utility_margin_bps=25.0,
        mae_margin_bps=6.0,
    )

    assert fixed_short_util[0] <= fixed_long_util[0] - 25.0
    assert fixed_short_util[1] <= fixed_long_util[1] - 25.0
    assert fixed_long_util[2] <= fixed_short_util[2] - 25.0
    assert fixed_long_util[3] <= fixed_short_util[3] - 25.0
    assert fixed_short_bad[:2].tolist() == [1.0, 1.0]
    assert fixed_long_bad[2:4].tolist() == [1.0, 1.0]
    assert fixed_short_mae[0] >= fixed_long_mae[0] + 6.0
    assert fixed_long_mae[2] >= fixed_short_mae[2] + 6.0
    assert masks["conflict"].tolist() == [False, False, False, False, True]
    assert fixed_long_bad[4] == 1.0
    assert fixed_short_bad[4] == 1.0
    assert fixed_long_util[4] <= 0.0
    assert fixed_short_util[4] <= 0.0
    assert masks["conflict_utility_suppressed"].tolist() == [False, False, False, False, True]


def test_structural_side_repair_forces_conflict_to_flat() -> None:
    direction = np.array([1, 0, 1, 0], dtype=np.int8)
    harvest = direction.copy()
    rising_channel_support_touch = np.array([1, 1, 0, 0], dtype=bool)
    countertrend_short_trap = np.array([0, 0, 0, 0], dtype=bool)
    support_retest_continuation = np.array([1, 0, 0, 0], dtype=bool)
    falling_channel_resistance_touch = np.array([1, 0, 1, 1], dtype=bool)
    countertrend_long_trap = np.array([0, 0, 0, 0], dtype=bool)
    resistance_retest_continuation = np.array([0, 0, 1, 0], dtype=bool)
    short_early_fail = np.array([0, 1, 0, 0], dtype=bool)
    long_early_fail = np.array([0, 0, 0, 1], dtype=bool)

    repaired_direction, repaired_harvest, masks = _apply_structural_side_repair(
        direction,
        harvest,
        rising_channel_support_touch,
        countertrend_short_trap,
        support_retest_continuation,
        falling_channel_resistance_touch,
        countertrend_long_trap,
        resistance_retest_continuation,
        short_early_fail,
        long_early_fail,
    )

    assert repaired_direction.tolist() == [-1, 0, 1, -1]
    assert repaired_harvest.tolist() == [-1, 0, 1, -1]
    assert masks["conflict_to_flat"].tolist() == [True, False, False, False]


def test_post_repair_scalar_bad_path_and_size_follow_repaired_side() -> None:
    quality_side = np.array([0, 1, -1, 0], dtype=np.int8)
    long_bad = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    short_bad = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    scalar_bad = _repaired_scalar_bad_path_target(quality_side, long_bad, short_bad)

    assert scalar_bad.tolist() == [0.0, 0.0, 0.0, 1.0]

    size = _position_size_target_from_repaired_path(
        mfe_first_n_bps=np.array([30.0, 5.0, 0.0, 1.0], dtype=np.float32),
        mae_first_n_bps=np.array([-5.0, -20.0, 0.0, -1.0], dtype=np.float32),
        atr_bps=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        trade_mask=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )

    assert size[0] > 0.5
    assert size[1] < 0.5
    assert size[2] == 0.5
