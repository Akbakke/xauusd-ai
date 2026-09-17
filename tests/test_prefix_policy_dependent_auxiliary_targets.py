import copy

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder
from gx1.contracts.entry_causal_m1_target_policy_v1 import materialize_causal_m1_auxiliary_outcomes
from gx1.contracts.entry_causal_m1_position_size_target_policy_v1 import fit_causal_m1_position_size_target_policy
from tests.test_entry_causal_m1_target_policy import _fit, _m1, _sha


@pytest.fixture(scope="module")
def policy_data(tmp_path_factory):
    m1 = _m1()
    m5 = pd.DataFrame({"time": m1.time.iloc[::5].reset_index(drop=True)})
    direction = _fit()
    size = fit_causal_m1_position_size_target_policy(
        closed_m5=m5, closed_m1=m1, entry_causal_m1_target_policy=direction,
        source_parquet_sha256=_sha("m5"), tape_provenance_sha256=_sha("tape"),
        m1_source_sha256=_sha("m1"),
        ecdf_artifact_path=tmp_path_factory.mktemp("prefix_aux") / "ecdf.npy",
    )
    mid = (m1.bid_open + m1.ask_open).to_numpy()[::5]
    m5["high"] = mid + 0.7
    m5["low"] = mid - 0.7
    m5["close"] = mid
    return m1, m5, direction, size


def three_outcomes(policy_data):
    m1, m5, direction, size = policy_data
    out = materialize_causal_m1_auxiliary_outcomes(
        policy=direction, m5_decision_times=m5.time.iloc[:3], closed_m1=m1,
    )
    # Known LONG, SHORT and FLAT cases, with no adverse/path failure trigger.
    for name in ("mfe_long_first_n_bps", "mfe_short_first_n_bps"):
        out[name] = 100.0
    for name in ("mae_long_first_n_bps", "mae_short_first_n_bps",
                 "bad_path_long_first_n", "bad_path_short_first_n"):
        out[name] = 0.0
    out["v11_pnl_long_at_dir_horizon_bps"] = [100.0, -100.0, 0.0]
    out["v11_pnl_short_at_dir_horizon_bps"] = [-100.0, 100.0, 0.0]
    lines = pd.DataFrame(1.0, index=pd.DatetimeIndex(out.time), columns=[
        "y_line_support_touch_held", "y_line_support_touch_mask",
        "y_line_resistance_touch_held", "y_line_resistance_touch_mask",
    ])
    return out, lines


def project(data, out, lines, **kw):
    return builder.build_policy_dependent_entry_auxiliary_targets(
        outcome_frame=out, line_labels=lines,
        direction_target_policy=kw.get("direction", data[2]),
        position_size_target_policy=kw.get("size", data[3]),
    )


def test_shared_targets_preserve_direction_masks_and_side_order(policy_data):
    out, lines = three_outcomes(policy_data)
    before = out.copy(deep=True)
    targets = project(policy_data, out, lines)
    assert targets["y_position_size_mask"].tolist() == [1.0, 1.0, 0.0]
    assert targets["y_position_size_target"][2] == 0.0
    assert targets["y_countertrend_short_trap"].tolist() == [1.0, 0.0, 0.0]
    assert targets["y_countertrend_long_trap"].tolist() == [0.0, 1.0, 0.0]
    assert len(targets) == 14
    assert all(v.dtype == np.float32 for v in targets.values())
    pd.testing.assert_frame_equal(out, before)


def test_absent_line_events_cannot_become_dense_traps(policy_data):
    out, lines = three_outcomes(policy_data)
    lines["y_line_support_touch_mask"] = 0.0
    lines["y_line_resistance_touch_mask"] = 0.0
    targets = project(policy_data, out, lines)
    assert not targets["y_countertrend_short_trap"].any()
    assert not targets["y_countertrend_long_trap"].any()


def test_raw_side_mae_keeps_bps_and_float32(policy_data):
    out, lines = three_outcomes(policy_data)
    out["mae_long_first_n_bps"] = [1.23456789, 6.0, 9.0]
    out["mae_short_first_n_bps"] = [7.0, 2.3456789, 3.0]
    targets = project(policy_data, out, lines)
    np.testing.assert_array_equal(targets["y_long_expected_mae_bps"], out.mae_long_first_n_bps.to_numpy(dtype=np.float32))
    np.testing.assert_array_equal(targets["y_short_expected_mae_bps"], out.mae_short_first_n_bps.to_numpy(dtype=np.float32))


def test_misaligned_registry_times_fail(policy_data):
    out, lines = three_outcomes(policy_data)
    with pytest.raises(RuntimeError, match="CLOCK_MISMATCH"):
        project(policy_data, out, lines.iloc[::-1])


def test_nonbinary_registry_mask_fails(policy_data):
    out, lines = three_outcomes(policy_data)
    lines.iloc[0, 1] = 0.5
    with pytest.raises(RuntimeError, match="LINE_LABEL_INVALID"):
        project(policy_data, out, lines)


def test_wrong_size_policy_cannot_be_applied(policy_data):
    out, lines = three_outcomes(policy_data)
    size = copy.deepcopy(policy_data[3])
    size["entry_causal_m1_target_policy_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="POSITION_SIZE_TARGET_POLICY"):
        project(policy_data, out, lines, size=size)


def materialize(data, *, entry_times=None, cutoff=None, m1=None, m5=None):
    return builder.materialize_policy_dependent_entry_auxiliary_targets(
        entry_times=pd.DatetimeIndex(data[1].time.iloc[40:43]) if entry_times is None else entry_times,
        closed_m1=data[0] if m1 is None else m1,
        closed_m5=data[1] if m5 is None else m5,
        registry_constants={"entry_m5": {"seq_len": 96, "trendline_band_atr": 0.9}},
        direction_target_policy=data[2], position_size_target_policy=data[3],
        supervision_end=pd.Timestamp("2024-01-02T00:00Z") if cutoff is None else cutoff,
    )


def test_materializer_uses_actual_m1_and_registry_owners_without_feature_mutation(policy_data):
    m1_before = policy_data[0].copy(deep=True)
    m5_before = policy_data[1].copy(deep=True)
    actual = materialize(policy_data)
    assert len(actual) == 3 and len(actual.columns) == 15
    assert actual.time.tolist() == policy_data[1].time.iloc[40:43].tolist()
    assert np.isfinite(actual.drop(columns="time").to_numpy()).all()
    pd.testing.assert_frame_equal(m1_before, policy_data[0])
    pd.testing.assert_frame_equal(m5_before, policy_data[1])


def test_exposed_policy_rejected_before_outcomes(policy_data, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Outcome computation must not happen after a rejected fit cutoff")
    monkeypatch.setattr(builder, "materialize_causal_m1_auxiliary_outcomes", forbidden)
    with pytest.raises(RuntimeError, match="POLICY_FIT_AFTER_CUTOFF"):
        materialize(policy_data, cutoff=pd.Timestamp("2024-01-01T08:00Z"))


def test_late_source_rejected_before_outcomes(policy_data, monkeypatch):
    monkeypatch.setattr(builder, "materialize_causal_m1_auxiliary_outcomes", lambda **kw: pytest.fail("late source used"))
    with pytest.raises(RuntimeError, match="SOURCE_AFTER_CUTOFF"):
        materialize(policy_data, cutoff=pd.Timestamp("2024-01-01T08:20Z"))


def test_incomplete_m5_future_is_not_truncated_or_zero_filled(policy_data):
    with pytest.raises(RuntimeError, match="M5_SUPPORT_MISSING"):
        materialize(policy_data, entry_times=pd.DatetimeIndex(policy_data[1].time.iloc[-1:]))


def test_missing_m1_minute_is_not_filled(policy_data):
    m1 = policy_data[0].drop(index=207).reset_index(drop=True)
    with pytest.raises(RuntimeError, match="M1_SUPPORT_MISSING"):
        materialize(policy_data, m1=m1)


def test_later_price_poison_cannot_change_earlier_labels(policy_data):
    before = materialize(policy_data)
    boundary = pd.Timestamp(policy_data[2]["train_end_utc"])
    m1, m5 = policy_data[0].copy(deep=True), policy_data[1].copy(deep=True)
    m1.loc[m1.time > boundary, m1.columns != "time"] += 5000.0
    m5.loc[m5.time > boundary, ["high", "low", "close"]] += 5000.0
    after = materialize(policy_data, m1=m1, m5=m5)
    pd.testing.assert_frame_equal(before, after, check_exact=True)

