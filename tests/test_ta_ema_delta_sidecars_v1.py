import numpy as np
import pandas as pd
import pytest

from gx1.audit.materialize_ta_ema_delta_sidecars_v1 import (
    LOCAL_FIELD, MTF_FIELD, _checked_names, _delta_and_slopes,
)
from gx1.features import entry_specialist_feature_groups_v1 as owners
from gx1.features import htf_features as htf


def test_additive_name_contract_rejects_old_column_reordering():
    local = list(owners.compile_full_ta_entry_execution_profile()['source_field_names']['signal'])
    mtf = list(htf.MULTI_TF_PER_BAR_FEATURES_V4)
    old_local = [name for name in local if name not in {
        LOCAL_FIELD, 'chart.geomline_above_anchor_span_bars',
        'chart.geomline_below_anchor_span_bars',
    }]
    old_mtf = [name for name in mtf if name not in {
        MTF_FIELD, 'geomline_above_anchor_span_bars',
        'geomline_below_anchor_span_bars',
    }]
    assert _checked_names({'feature_field_order': old_local}, {'feature_names': old_mtf}) == (
        old_local, old_mtf,
    )
    old_mtf[0], old_mtf[1] = old_mtf[1], old_mtf[0]
    with pytest.raises(RuntimeError, match='TA_EMA_DELTA_EXACT_ADDITIVE_NAMES'):
        _checked_names({'feature_field_order': old_local}, {'feature_names': old_mtf})


def test_delta_uses_only_past_closed_bars():
    index = pd.date_range('2024-01-01', periods=260, freq='5min', tz='UTC')
    close = 1900.0 + np.arange(len(index), dtype=np.float64) * 0.03
    frame = pd.DataFrame({
        'high': close + 0.2,
        'low': close - 0.2,
        'close': close,
    }, index=index)
    original = _delta_and_slopes(frame)[0]
    changed = frame.copy()
    changed.loc[index[230]:, ['high', 'low', 'close']] += 40.0
    revised = _delta_and_slopes(changed)[0]
    assert np.isfinite(original[204:230]).all()
    np.testing.assert_array_equal(revised[:230], original[:230])


def test_one_use_terminal_success_and_failure_are_distinct():
    from gx1.audit.materialize_ta_ema_delta_sidecars_v1 import _terminal_record

    ref = {'path': '/fixed/receipt', 'sha256': 'a' * 64}
    success = _terminal_record(
        'COMPLETE', plan=ref, claim=ref, result=ref,
        failure=None, source_unchanged=True,
    )
    failure = _terminal_record(
        'FAILED_CONSUMED', plan=ref, claim=ref, result=None,
        failure=ref, source_unchanged=True,
    )
    assert success['exit_code'] == 0 and success['no_relaunch'] is True
    assert failure['exit_code'] == 1 and failure['no_relaunch'] is True
    with pytest.raises(RuntimeError, match='TA_EMA_DELTA_TERMINAL_STATE'):
        _terminal_record(
            'COMPLETE', plan=ref, claim=ref, result=None,
            failure=ref, source_unchanged=True,
        )
