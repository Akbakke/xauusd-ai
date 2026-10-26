"""Synthetic endpoint composition tests; no native model or saved market payload."""
from fractions import Fraction

import numpy as np
import pytest

from gx1.contracts import entry_direction_saved_horizons_v1 as owner


def saved_prices(prices, *, selected=None, cutoff_minute=None):
    prices = np.asarray(prices, dtype=np.float64)
    minute = np.arange(len(prices), dtype=np.int64)
    base = 1704067200 * 10**9
    selected = np.arange(0, len(prices), 5, dtype=np.int64) if selected is None else np.asarray(selected, np.int64)
    n = len(selected)
    cutoff_minute = len(prices) if cutoff_minute is None else cutoff_minute
    cutoff = base + cutoff_minute * owner.MINUTE_NS
    decision = base + selected * owner.MINUTE_NS
    horizons = np.asarray(owner.BASE_HORIZONS, np.int64)
    terminal = selected[:, None] + horizons
    observed = (terminal < len(prices)) & (terminal < cutoff_minute)
    moves = np.zeros((n, 3), np.float64)
    returns = moves.copy()
    rows = np.full((n, 3), -1, np.int64)
    for j in range(3):
        mask = observed[:, j]
        moves[mask, j] = prices[terminal[mask, j]] - prices[selected[mask]]
        returns[mask, j] = moves[mask, j] / prices[selected[mask]] * 10000
        rows[mask, j] = terminal[mask, j]
    return {
        "parent_rows": np.arange(n, dtype=np.int64), "bar_start_ns": decision - 5 * owner.MINUTE_NS,
        "dynamic_entry_m1_row": selected.copy(), "decision_ns": decision,
        "terminal_time_ns": decision[:, None] + horizons * owner.MINUTE_NS,
        "elapsed_wall_seconds": np.broadcast_to(horizons * 60, (n, 3)).copy(),
        "terminal_source_row": rows, "observed": observed, "mid_move": moves,
        "mid_return_bps": returns, "sign": np.sign(moves).astype(np.int8),
    }, cutoff


@pytest.mark.parametrize("kind", ["up", "down", "mixed"])
def test_resolved_labels_equal_direct_endpoint_and_preserve_inputs(kind):
    rng = np.random.default_rng(314159)
    delta = np.full(240, 0.125) if kind == "up" else np.full(240, -0.125) if kind == "down" else rng.normal(0, 0.1, 240)
    prices = 2000 + np.cumsum(delta)
    saved, cutoff = saved_prices(prices)
    before = {k: v.tobytes() for k, v in saved.items()}
    result = owner.derive_direction_horizons(saved, cutoff_ns=cutoff)
    arrays = result["arrays"]
    assert result["horizons_minutes"] == (30, 60)
    assert result["control15"]["resolved_signs_checked"] > 0
    assert all(saved[k].tobytes() == before[k] for k in saved)
    assert all(not a.flags.writeable for a in arrays.values())
    start = saved["dynamic_entry_m1_row"]
    for j, horizon in enumerate((30, 60)):
        valid = arrays["label_resolved"][:, j]
        direct = prices[start[valid] + horizon] - prices[start[valid]]
        assert np.array_equal(arrays["sign"][valid, j], np.sign(direct))
        assert np.array_equal(arrays["terminal_source_row"][valid, j], start[valid] + horizon)
        assert np.all(arrays["mid_move_lower"][valid, j] <= direct)
        assert np.all(arrays["mid_move_upper"][valid, j] >= direct)
        count = result["counts"][j]
        assert sum(count[k] for k in owner.CHAIN_REASONS.values()) == len(start)
        assert count["resolved_nonzero"] + count["certified_zero"] + count["rounding_ambiguous"] == count["observed_chain"]


def test_only_all_zero_components_certify_a_zero_label():
    saved, cutoff = saved_prices(np.full(100, 2000.0))
    result = owner.derive_direction_horizons(saved, cutoff_ns=cutoff)
    arrays = result["arrays"]
    assert arrays["certified_zero"][0].all()
    assert arrays["label_resolved"][0].all()
    assert (arrays["sign"] == 0).all()
    assert not arrays["rounding_ambiguous"].any()


@pytest.mark.parametrize("endpoint", [1.0, 2.0, 3.0])
def test_extreme_cancellation_is_unresolved_instead_of_guessed(endpoint):
    prices = np.full(100, endpoint, np.float64)
    prices[:15] = 1.5
    prices[15:30] = 1e16
    saved, cutoff = saved_prices(prices)
    result = owner.derive_direction_horizons(saved, cutoff_ns=cutoff)
    assert result["arrays"]["observed_chain"][0, 0]
    assert result["arrays"]["rounding_ambiguous"][0, 0]
    assert not result["arrays"]["label_resolved"][0, 0]


def test_nonzero_components_cancelling_to_zero_remain_unresolved():
    prices = np.full(100, 2000.0)
    prices[15:30] = 2010.0
    saved, cutoff = saved_prices(prices)
    result = owner.derive_direction_horizons(saved, cutoff_ns=cutoff)
    assert result["arrays"]["rounding_ambiguous"][0, 0]
    assert not result["arrays"]["certified_zero"][0, 0]


def test_missing_intermediate_parent_preserves_parent_and_separates_reason():
    selected = np.setdiff1d(np.arange(0, 100, 5), [15])
    saved, cutoff = saved_prices(2000 + np.arange(100) * 0.1, selected=selected)
    result = owner.derive_direction_horizons(saved, cutoff_ns=cutoff)
    assert np.array_equal(result["arrays"]["parent_rows"], saved["parent_rows"])
    assert result["arrays"]["chain_reason"][0, 0] == 2
    assert not result["arrays"]["observed_chain"][0, 0]


def test_unobserved_component_is_separate_from_rounding_ambiguity():
    saved, cutoff = saved_prices(2000 + np.arange(100) * 0.1)
    i, j = 3, 2
    saved["observed"][i, j] = False
    for name in ("mid_move", "mid_return_bps", "sign"):
        saved[name][i, j] = 0
    saved["terminal_source_row"][i, j] = -1
    # Also make any 3x5 control ending here unavailable.
    saved["observed"][i, 1] = False
    for name in ("mid_move", "mid_return_bps", "sign"):
        saved[name][i, 1] = 0
    saved["terminal_source_row"][i, 1] = -1
    result = owner.derive_direction_horizons(saved, cutoff_ns=cutoff)
    assert result["arrays"]["chain_reason"][0, 0] == 3
    assert not result["arrays"]["rounding_ambiguous"][0, 0]


def test_exact_right_cutoff_is_censored_with_no_row_loss():
    saved, cutoff = saved_prices(2000 + np.arange(100) * 0.1, cutoff_minute=60)
    result = owner.derive_direction_horizons(saved, cutoff_ns=cutoff)
    arrays = result["arrays"]
    assert arrays["chain_reason"][0, 1] == 1
    assert not arrays["observed_chain"][0, 1]
    assert len(arrays["parent_rows"]) == len(saved["parent_rows"])


@pytest.mark.parametrize("fault,match", [
    ("row", "SHARED_PRICE_ROW"),
    ("clock", "EXACT_BASE_ENDPOINT_CLOCKS"),
    ("duplicate", "PARENT_DECISION_CLOCK"),
    ("dtype", "HEADERS_FINITE"),
    ("masked", "MASKED_STORAGE_OR_SIGN"),
    ("direct15", "CONTROL15_DIFFERENCE_INTERVAL"),
    ("cutoff", "OBSERVED_SOURCE_OR_CUTOFF"),
])
def test_corrupt_inputs_stop_instead_of_becoming_missing_labels(fault, match):
    saved, cutoff = saved_prices(2000 + np.arange(100) * 0.1)
    if fault == "row": saved["terminal_source_row"][0, 1] += 1
    elif fault == "clock": saved["terminal_time_ns"][0, 0] += owner.MINUTE_NS
    elif fault == "duplicate": saved["decision_ns"][1] = saved["decision_ns"][0]
    elif fault == "dtype": saved["sign"] = saved["sign"].astype(np.int64)
    elif fault == "masked": saved["observed"][0, 0] = False
    elif fault == "direct15":
        saved["mid_move"][0, 2] += 1.0
    elif fault == "cutoff": cutoff = saved["terminal_time_ns"][0, 0].item()
    with pytest.raises(RuntimeError, match=match):
        owner.derive_direction_horizons(saved, cutoff_ns=cutoff)


def test_interval_overflow_is_rejected():
    parts = np.full((4, 2), np.finfo(np.float64).max)
    with pytest.raises(RuntimeError, match="INTERVAL_OVERFLOW"):
        owner.outward_sum_bounds(parts)


def test_outward_intervals_contain_exact_endpoint_differences():
    rng = np.random.default_rng(271828)
    samples = rng.uniform(0.5, 2, (5, 200)) * np.exp2(rng.integers(-100, 100, (1, 200)))
    parts = np.ascontiguousarray(np.diff(samples, axis=0))
    low, high = owner.outward_sum_bounds(parts)
    for j in range(samples.shape[1]):
        exact = Fraction.from_float(float(samples[-1, j])) - Fraction.from_float(float(samples[0, j]))
        assert Fraction.from_float(float(low[j])) <= exact <= Fraction.from_float(float(high[j]))
