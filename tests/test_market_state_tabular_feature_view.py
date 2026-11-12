"""Synthetic causal and domain checks for the raw tabular research view."""

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import market_state_tabular_feature_view_v1 as view
from gx1.features import htf_features as htf
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile


def _case():
    fields = compile_compact_entry_execution_profile()["physical_field_names"]
    starts = np.asarray([pd.Timestamp("2025-06-02T21:50:00Z").value,
                         pd.Timestamp("2025-06-02T21:55:00Z").value,
                         pd.Timestamp("2025-06-02T22:00:00Z").value], dtype=np.int64)
    decisions = starts + view.BAR_NS
    parents = np.asarray([4, 7, 9], dtype=np.int64)
    signal = np.zeros((3, len(fields["signal"])), dtype=np.float32)
    ctx_cat = np.asarray([[0], [1], [3]], dtype=np.int64)
    signal_names = list(fields["signal"])
    mask_name = "ctx_cont.gc_d1_observed_unknown_trade_size_valid_observed"
    value_name = mask_name.removesuffix("_observed")
    signal[2, signal_names.index(value_name)] = 3.
    signal[2, signal_names.index(mask_name)] = 1.
    alias_indices = [signal_names.index("ctx_cont." + name) for name in fields["ctx_cont"]]
    ctx_cont = signal[:, alias_indices].copy()
    sources = {}
    for tf in view.TIMEFRAMES:
        step = htf.MULTI_TF_SHIFT[tf]
        last = htf.multi_tf_last_closed_label(pd.Timestamp(int(starts[0]), tz="UTC"), tf)
        times = pd.date_range(start=last - 2 * step, periods=7, freq=step).as_unit("ns").asi8
        values = np.zeros((len(times), len(fields["mtf"])), dtype=np.float32)
        values[:, list(fields["mtf"]).index("atr_bps_14")] = np.arange(len(times)) + 10
        values[:, list(fields["mtf"]).index("mtf_smc_swing_state")] = np.arange(len(times)) % 5
        frame = pd.DataFrame(values,
            index=pd.DatetimeIndex(times.astype("datetime64[ns]"), tz="UTC"),
            columns=fields["mtf"], copy=False)
        frame.attrs.update(ts_int64=times.copy(), feats_np=values,
                           causal_warmup_rows=0,
                           htf_feature_contract=htf.HTF_V4_MATRIX_CONTRACT)
        sources[tf] = frame
    axes = {"source": {"path": "/synthetic/PRICE_TARGETS.npz", "sha256": "a" * 64},
            "rows": len(parents),
            "parent_rows_sha256": view._axis_hash(parents),
            "bar_start_ns_sha256": view._axis_hash(starts),
            "decision_ns_sha256": view._axis_hash(decisions)}
    return dict(parent_rows=parents, bar_start_ns=starts, decision_ns=decisions,
                expected_axes=axes, signal=signal, ctx_cont=ctx_cont,
                ctx_cat=ctx_cat, multi_tf_sources=sources)


def test_exact_last_closed_bar_and_full_five_tf_layout():
    case = _case()
    bound = view.MarketStateTabularFeatureView(**case)
    assert bound.width == 2727 + 1 + 5 * 255 == 4003
    assert len(bound.feature_names) == len(set(bound.feature_names)) == bound.width
    assert len(bound.categorical) == 7
    assert {item["name"] for item in bound.categorical} == {
        "signal.smc_swing_state", "ctx_cat.session_id",
        *("mtf_" + tf.lower() + ".mtf_smc_swing_state" for tf in view.TIMEFRAMES)}
    assert [item["domain"] for item in bound.categorical] == [[0, 1, 2, 3, 4], [0, 1, 2, 3],
                                                                  *([[0, 1, 2, 3, 4]] * 5)]
    positions = np.asarray([0, 2], dtype=np.int64)
    matrix = bound.take(positions, max_rows=2)
    assert matrix.dtype == np.float32 and matrix.shape == (2, bound.width)
    for tf in view.TIMEFRAMES:
        times = case["multi_tf_sources"][tf].attrs["ts_int64"]
        shift_ns = int(htf.MULTI_TF_SHIFT[tf].value)
        expected = np.searchsorted(times, case["decision_ns"] - shift_ns, side="right") - 1
        assert np.array_equal(bound.tf_tails[tf], expected)
        for row in range(3):
            scalar_label = htf.multi_tf_last_closed_label(
                pd.Timestamp(int(case["bar_start_ns"][row]), tz="UTC"), tf)
            assert times[expected[row]] == scalar_label.value
        start = bound.offsets[tf]
        atr = list(compile_compact_entry_execution_profile()["physical_field_names"]["mtf"]).index("atr_bps_14")
        assert np.array_equal(matrix[:, start + atr], case["multi_tf_sources"][tf].attrs["feats_np"][expected[positions], atr])
    # D1 opening at 22:00 UTC is not closed when the M5 issue is at 22:00.
    d1_times = case["multi_tf_sources"]["D1"].attrs["ts_int64"]
    assert d1_times[bound.tf_tails["D1"][1]] + int(htf.MULTI_TF_SHIFT["D1"].value) == case["decision_ns"][1]
    assert d1_times[bound.tf_tails["D1"][1] + 1] + int(htf.MULTI_TF_SHIFT["D1"].value) > case["decision_ns"][1]
    value_name = "signal.ctx_cont.gc_d1_observed_unknown_trade_size_valid"
    mask_name = value_name + "_observed"
    assert matrix[0, bound.feature_names.index(value_name)] == 0.
    assert matrix[0, bound.feature_names.index(mask_name)] == 0.
    assert matrix[1, bound.feature_names.index(value_name)] == 3.
    assert matrix[1, bound.feature_names.index(mask_name)] == 1.


def test_alias_mismatch_and_categorical_domain_fail_closed():
    case = _case()
    case["ctx_cont"][0, 0] = 1.
    bound = view.MarketStateTabularFeatureView(**case)
    with pytest.raises(RuntimeError, match="ACTUAL_CONTEXT_ALIAS"):
        bound.take(np.asarray([0], dtype=np.int64), max_rows=1)
    case = _case()
    signal_names = list(compile_compact_entry_execution_profile()["physical_field_names"]["signal"])
    case["signal"][0, signal_names.index("smc_swing_state")] = 9.
    bound = view.MarketStateTabularFeatureView(**case)
    with pytest.raises(RuntimeError, match="CATEGORICAL_DOMAIN:signal.smc_swing_state"):
        bound.take(np.asarray([0], dtype=np.int64), max_rows=1)
    case = _case()
    tf = "D1"
    mtf_names = list(compile_compact_entry_execution_profile()["physical_field_names"]["mtf"])
    times = case["multi_tf_sources"][tf].attrs["ts_int64"]
    tail = int(np.searchsorted(times, case["decision_ns"][0] - int(htf.MULTI_TF_SHIFT[tf].value), side="right") - 1)
    case["multi_tf_sources"][tf].attrs["feats_np"][tail, mtf_names.index("mtf_smc_swing_state")] = 9.
    bound = view.MarketStateTabularFeatureView(**case)
    with pytest.raises(RuntimeError, match="CATEGORICAL_DOMAIN:mtf_d1.mtf_smc_swing_state"):
        bound.take(np.asarray([0], dtype=np.int64), max_rows=1)


def test_axes_clock_and_chunk_bound_fail_closed():
    case = _case()
    case["expected_axes"]["decision_ns_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="AXIS_BINDING"):
        view.MarketStateTabularFeatureView(**case)
    case = _case()
    case["decision_ns"] = case["decision_ns"] + 1
    with pytest.raises(RuntimeError, match="DECISION_CLOCK"):
        view.MarketStateTabularFeatureView(**case)
    case = _case()
    bound = view.MarketStateTabularFeatureView(**case)
    with pytest.raises(RuntimeError, match="CHUNK_ROWS"):
        bound.take(np.asarray([0, 1], dtype=np.int64), max_rows=1)
    with pytest.raises(RuntimeError, match="CHUNK_ROWS"):
        bound.take(np.asarray([2, 1], dtype=np.int64), max_rows=2)
    # The caller cannot override the physical allocation limit with max_rows.
    bound.parent_rows = np.arange(view.MAX_CHUNK_ROWS + 1, dtype=np.int64)
    with pytest.raises(RuntimeError, match="CHUNK_ROWS"):
        bound.take(np.arange(view.MAX_CHUNK_ROWS + 1, dtype=np.int64),
                   max_rows=view.MAX_CHUNK_ROWS + 1)
