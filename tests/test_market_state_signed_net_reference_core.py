import numpy as np
import pytest

from gx1.audit.market_state_signed_net_reference_core_v1 import score_fold


def _case():
    # Price cells are identical; GC splits both fit and later score outcomes.
    fit = np.array([-1, -1, -1, -1, 1, 1, 1, 1], np.int8)
    score = np.array([-1, -1, 1, 1, 0], np.int8)
    fgc = fit.copy()
    sgc = score.copy()
    fbool = np.ones(len(fit), bool)
    sbool = np.ones(len(score), bool)
    kwargs = dict(
        fit_target=fit, fit_target_valid=fbool,
        fit_ema=np.ones(len(fit), np.int8), fit_ema_valid=fbool,
        fit_swing=np.ones(len(fit), np.int8), fit_swing_valid=fbool,
        fit_gc=fgc, fit_gc_valid=fbool,
        score_target=score, score_target_valid=sbool,
        score_magnitude_atr=np.array([0.3, 0.7, 1.2, 1.4, 0], np.float64),
        score_issues=np.array([1714521600000000000 + i * 300000000000
                               for i in range(len(score))], np.int64),
        score_ema=np.ones(len(score), np.int8), score_ema_valid=sbool,
        score_swing=np.ones(len(score), np.int8), score_swing_valid=sbool,
        score_gc=sgc, score_gc_valid=sbool,
        score_gc_raw=np.array([-0.8, -0.4, 0.3, 0.9, 0], np.float64),
    )
    return kwargs


def test_chronological_gc_addition_on_identical_rows():
    got = score_fold(**_case())
    assert got['score_target_zero'] == 1
    assert got['score_common'] == 4
    assert got['prefix_prior_up'] == 0.5
    assert got['overall']['price_brier_sum'] == got['overall']['prior_brier_sum']
    assert got['overall']['price_plus_gc_brier_sum'] < got['overall']['price_brier_sum']
    assert got['overall']['gc_auc'] == 1.0
    assert sum(x['n'] for x in got['by_abs_D_over_ATR'].values()) == 4
    assert sum(x['n'] for x in got['by_utc_week'].values()) == 4


def test_missing_gc_stays_out_of_every_paired_comparison():
    case = _case()
    case['score_gc_valid'] = case['score_gc_valid'].copy()
    case['score_gc_valid'][3] = False
    got = score_fold(**case)
    assert got['score_target_nonzero'] == 4
    assert got['score_common'] == 3
    assert got['score_missing_gc'] == 1
    assert got['overall']['n'] == 3


def test_rejects_gc_sign_and_magnitude_inconsistency():
    case = _case()
    case['score_gc_raw'] = case['score_gc_raw'].copy()
    case['score_gc_raw'][0] = 0.2
    with pytest.raises(ValueError, match='SCORE_NUMERICS'):
        score_fold(**case)
    case = _case()
    case['score_magnitude_atr'] = case['score_magnitude_atr'].copy()
    case['score_magnitude_atr'][4] = 0.1
    with pytest.raises(ValueError, match='SCORE_NUMERICS'):
        score_fold(**case)
