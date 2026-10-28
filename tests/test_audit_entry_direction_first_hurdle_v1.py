"""Small synthetic tests of first-hurdle coverage and fixed ranking."""
import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    build_dynamic_m1_fill_surface, prepare_dynamic_m1_prefix_index,
)
from gx1.scripts.audit_entry_direction_first_hurdle_v1 import (
    conditional_rank, coverage_report, endpoint_alignment, race_five, race_panel,
)


def synthetic_index():
    times = pd.date_range("2024-05-06T00:00:00Z", periods=80, freq="min")
    frame = pd.DataFrame({"time": times})
    for side in ("bid", "ask"):
        for field in ("open", "high", "low"):
            frame[side + "_" + field] = 100.
    for side in ("bid", "ask"):
        frame.loc[6, side + "_high"] = 100.5
        frame.loc[7, side + "_low"] = 99.5
    index = prepare_dynamic_m1_prefix_index(frame, split_end_exclusive=times[-1] + pd.Timedelta(minutes=1),
                                            source_parquet_sha256="a" * 64)
    fill = build_dynamic_m1_fill_surface(index, m5_decision_times=pd.DatetimeIndex([times[0], times[5]]))
    return index, fill


def test_all_sixty_horizons_keep_first_hit_and_no_event_distinct():
    index, fill = synthetic_index()
    panel = race_panel(index, fill, chunk_parents=2)
    assert panel["outcome"].shape == (2, 60)
    assert panel["outcome"][0, 4] == 1 and panel["observed"][0, 4]
    assert panel["possibilities"][0, 4] == 1
    assert panel["outcome"][1, 4] == 0 and panel["observed"][1, 4]
    assert panel["possibilities"][1, 4] == 4
    coverage = coverage_report(panel)
    assert coverage[4]["counts"]["long_first"] == 1
    assert coverage[4]["counts"]["neither_known"] == 1
    assert coverage[9]["horizon_minutes"] == 10 and coverage[59]["horizon_minutes"] == 60


def test_chunking_does_not_change_labels_or_missing_masks():
    index, fill = synthetic_index()
    first, second = race_panel(index, fill, chunk_parents=1), race_panel(index, fill, chunk_parents=2)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_same_minute_partial_label_is_not_a_known_direction():
    panel = {"outcome": np.zeros((2, 60), np.int8), "observed": np.ones((2, 60), np.bool_),
             "reason": np.zeros((2, 60), np.uint8), "possibilities": np.full((2, 60), 4, np.uint8)}
    panel["observed"][0, 4] = False; panel["reason"][0, 4] = 3; panel["possibilities"][0, 4] = 3
    report = coverage_report(panel)[4]
    assert report["counts"]["same_minute_unknown"] == 1
    assert report["counts"]["neither_known"] == 1
    assert report["possible_sets_bit_order_long_short_neither"]["3"] == 1


def test_fixed_rank_uses_known_long_and_short_only_with_block_bootstrap():
    n = 32
    panel = {"outcome": np.zeros((n, 60), np.int8), "observed": np.ones((n, 60), np.bool_)}
    panel["outcome"][:, 4] = np.where(np.arange(n) % 2, 1, -1)
    panel["outcome"][0, 4] = 0
    panel["observed"][1, 4] = False
    scores = np.where(panel["outcome"][:, 4] == 1, 1., -1.)
    decisions = np.arange(n, dtype=np.int64) * (7 * 86_400_000_000_000)
    result = conditional_rank(panel, scores, decisions, replicates=100, minimum_blocks=8)
    assert result["known_direction_rows"] == n - 2
    assert result["auc"] == 1.
    assert {x["scheme"] for x in result["block_intervals"]} == {"UTC_DAY", "UTC_WEEK_MONDAY"}
    assert "outcome-dependent" in result["selection"]


def test_one_class_cannot_claim_auc():
    panel = {"outcome": np.ones((3, 60), np.int8), "observed": np.ones((3, 60), np.bool_)}
    result = conditional_rank(panel, np.zeros(3), np.arange(3, dtype=np.int64), replicates=2, minimum_blocks=1)
    assert result["status"] == "INSUFFICIENT_BOTH_CLASSES" and result["auc"] is None


def test_inconsistent_coverage_partition_fails_closed():
    panel = {"outcome": np.zeros((1, 60), np.int8), "observed": np.zeros((1, 60), np.bool_),
             "reason": np.full((1, 60), 9, np.uint8), "possibilities": np.full((1, 60), 7, np.uint8)}
    with pytest.raises(RuntimeError, match="COVERAGE_PARTITION"):
        coverage_report(panel)


def test_five_minute_replay_matches_dense_panel():
    index, fill = synthetic_index()
    dense = race_panel(index, fill, chunk_parents=2)
    race = race_five(index, fill)
    np.testing.assert_array_equal(race["outcome"], dense["outcome"][:, 4])
    np.testing.assert_array_equal(race["outcome_observed"], dense["observed"][:, 4])
    np.testing.assert_array_equal(race["unknown_reason"], dense["reason"][:, 4])


def test_endpoint_alignment_keeps_opposite_zero_and_missing_separate():
    race = {"outcome": np.array([1, 1, -1, -1, 1, 0], np.int8),
            "outcome_observed": np.array([True, True, True, True, True, True])}
    endpoint = np.array([1, -1, -1, 0, 0, 1], np.int8)
    observed = np.array([True, True, True, False, True, True])
    result = endpoint_alignment(race, endpoint, observed, np.array([.9, .5, -.8, 0., 0., .2]))
    assert result["known_first_direction"] == 5
    assert result["first_by_endpoint_sign"] == [[1, 1, 1], [1, 0, 0]]
    assert (result["same_direction"], result["opposite_direction"], result["endpoint_zero"],
            result["endpoint_missing"]) == (2, 1, 1, 1)
    assert result["common_observed_nonzero_endpoint"] == 3
    assert result["same_score_auc_on_common_rows"] == {"first_hurdle": 1., "endpoint": 1.}


def test_endpoint_alignment_rejects_malformed_signs():
    race = {"outcome": np.array([1], np.int8), "outcome_observed": np.array([True])}
    with pytest.raises(RuntimeError, match="ALIGNMENT_INPUT"):
        endpoint_alignment(race, np.array([2], np.int8), np.array([True]), np.array([.5]))
