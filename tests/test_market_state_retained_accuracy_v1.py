"""Synthetic thresholds and masks only; no retained files or model execution."""
import json

import numpy as np
import pytest

from gx1.audit import market_state_retained_accuracy_v1 as owner


def case(bits, *, side=0.0, activity=0.0):
    n=len(bits)
    raw=np.zeros((n,5,2),np.float64)
    raw[:,:,0]=np.asarray(activity).reshape(-1,1) if np.ndim(activity) else activity
    raw[:,:,1]=np.asarray(side).reshape(-1,1) if np.ndim(side) else side
    allowed=np.broadcast_to(np.asarray(bits,np.uint8)[:,None],(n,5)).copy()
    ready=np.ones((n,5),np.bool_)
    return raw,allowed,ready,ready.copy()


def test_perfect_side_ranking_can_have_only_half_threshold_accuracy():
    raw,bits,eligible,mature=case([1,1,4,4],side=[1.,2.,3.,4.],activity=-1000.)
    # Independent pairwise ranking oracle: all true UP logits exceed all DOWN.
    z=raw[:,0,1];up=bits[:,0]==4
    assert float((z[up,None]>z[~up][None,:]).mean())==1.
    measured=owner.score_accuracy(raw,bits,eligible,mature)['M5']
    side=measured['conditional_side']
    assert side['confusion_matrix']==[[0,2],[0,2]]
    assert side['accuracy']==side['balanced_accuracy']==.5
    assert side['recalls']=={'DOWN':0.,'UP':1.}
    assert side['predicted_fractions']=={'DOWN':0.,'UP':1.}
    assert side['always_up_accuracy']==side['always_down_accuracy']==.5
    # Suppressing activity does not alter conditional direction; argmax is FLAT.
    assert measured['three_class']['accuracy']==0.
    assert measured['three_class']['predicted_fractions']['FLAT']==1.


def test_masks_keep_activity_unknown_side_and_singleton_support_separate():
    args=case([1,2,4,5,7,4,1],side=[-1.,-1.,1.,1.,1.,1.,-1.],activity=[1.,-1.,1.,1.,1.,1.,1.])
    raw,bits,eligible,mature=args
    eligible[5,:]=False;mature[6,:]=False
    got=owner.score_accuracy(*args)
    for tf in owner.TIMEFRAMES:
        cell=got[tf];support=cell['support']
        assert support==dict(rows=7,eligible=6,mature=6,eligible_mature=5,ineligible_or_immature=2,
            allowed_counts_on_common={'1':1,'2':1,'4':1,'5':1,'7':1},conditional_side_rows=2,
            activity_rows=4,three_class_rows=3)
        assert cell['conditional_side']['accuracy']==cell['activity']['accuracy']==1.
        assert cell['activity']['confusion_matrix']==[[1,0],[0,3]]
        assert cell['three_class']['always_flat_accuracy']==pytest.approx(1/3)
    # One TF's mask must not change another TF's support.
    eligible[0,1]=False
    changed=owner.score_accuracy(*args)
    assert changed['M5']==got['M5'] and changed['M15']['conditional_side']['count']==1


def test_exact_binary_ties_choose_positive_and_joint_tie_chooses_first_class():
    args=case([1,4,2],side=0.,activity=[np.log(4.),np.log(4.),0.])
    got=owner.score_accuracy(*args)['M5']
    assert got['conditional_side']['exact_ties']==2
    assert got['conditional_side']['confusion_matrix']==[[0,1],[0,1]]
    assert got['activity']['exact_ties']==1 and got['activity']['tie_rule']=='ACTIVE'
    assert got['activity']['confusion_matrix']==[[0,1],[0,2]]
    assert got['three_class']['confusion_matrix']==[[1,0,0],[0,1,0],[1,0,0]]
    assert got['three_class']['exact_ties']==2
    assert got['three_class']['tie_rule']=='FIRST_MAX_DOWN_FLAT_UP'
    assert got['three_class']['accuracy']==pytest.approx(2/3)


def test_zero_negative_logit_is_not_replaced_by_rounded_sigmoid_threshold():
    args=case([1,4],side=[-np.finfo(np.float64).tiny,0.])
    got=owner.score_accuracy(*args)['M5']['conditional_side']
    assert got['accuracy']==1. and got['exact_ties']==1
    bins=owner.fixed_probability_bins(*args)['bins']
    assert bins[2]['known_side_accuracy']==1.


def test_empty_and_one_sided_support_are_defined_without_invented_balanced_accuracy():
    args=case([4,4],side=2.,activity=2.)
    got=owner.score_accuracy(*args)['M5']['conditional_side']
    assert got['accuracy']==1. and got['balanced_accuracy'] is None
    assert got['recalls']=={'DOWN':None,'UP':1.}
    assert got['always_up_accuracy']==1. and got['always_down_accuracy']==0.
    args[2][:]=False
    empty=owner.score_accuracy(*args)
    for cell in empty.values():
        for key in ('conditional_side','activity','three_class'):
            assert cell[key]['count']==0 and cell[key]['correct']==0
            assert cell[key]['accuracy'] is cell[key]['balanced_accuracy'] is None
            assert all(v is None for v in cell[key]['recalls'].values())
    bins=owner.fixed_probability_bins(*args)
    assert len(bins['bins'])==5 and bins['eligible_mature_rows']==0
    assert all(b['count']==0 and b['known_side_accuracy'] is None and b['all_common_row_fraction'] is None
               for b in bins['bins'])
    json.dumps(dict(scores=empty,bins=bins),allow_nan=False)


def test_fixed_bins_include_flat_unknown_and_extreme_endpoints_without_subset_renormalizing():
    args=case([1,2,4,5,7,1,4],side=[-1000.,-2.,-.8,0.,.8,2.,1000.])
    got=owner.fixed_probability_bins(*args)
    assert got['edges']==[0.,.2,.4,.6,.8,1.] and got['eligible_mature_rows']==7
    assert [b['count'] for b in got['bins']]==[2,1,1,1,2]
    assert [b['known_side_count'] for b in got['bins']]==[1,1,0,0,2]
    assert [b['known_side_accuracy'] for b in got['bins']]==[1.,0.,None,None,.5]
    assert got['bins'][0]['allowed_counts']['2']==1
    assert got['bins'][2]['allowed_counts']['5']==1 and got['bins'][3]['allowed_counts']['7']==1
    assert sum(b['all_common_row_fraction'] for b in got['bins'])==pytest.approx(1.)
    assert all(sum(b['allowed_counts'].values())==b['count'] for b in got['bins'])
    assert got['bins'][-1]['upper_inclusive'] is True
    assert all(b['upper_inclusive'] is False for b in got['bins'][:-1])


def test_bin_boundaries_are_exact_half_open_and_final_closed():
    # Test literal probability boundaries independently of inverse-sigmoid rounding.
    values=np.array([0.,np.nextafter(.2,0.),.2,np.nextafter(.4,0.),.4,
        np.nextafter(.6,0.),.6,np.nextafter(.8,0.),.8,1.])
    assert owner._fixed_bin_indices(values).tolist()==[0,0,1,1,2,2,3,3,4,4]


@pytest.mark.parametrize('fault',['raw_shape','raw_dtype','nonfinite','allowed_dtype','invalid_bit','mask_dtype','mask_shape'])
def test_malformed_inputs_fail_closed(fault):
    raw,bits,eligible,mature=case([1,4])
    if fault=='raw_shape':raw=raw[:,None,:,:]
    elif fault=='raw_dtype':raw=raw.astype(np.int64)
    elif fault=='nonfinite':raw[0,0,0]=np.nan
    elif fault=='allowed_dtype':bits=bits.astype(np.int64)
    elif fault=='invalid_bit':bits[0,0]=3
    elif fault=='mask_dtype':eligible=eligible.astype(np.uint8)
    elif fault=='mask_shape':mature=mature[:1]
    for operation in (owner.score_accuracy,owner.fixed_probability_bins):
        with pytest.raises(ValueError):operation(raw,bits,eligible,mature)


def test_readonly_inputs_are_unchanged_and_reports_are_finite_json_scalars():
    args=case([1,2,4,5,7],side=[-1.,0.,1.,1000.,-1000.],activity=[1.,-1.,2.,0.,-3.])
    before=[(a.dtype,a.shape,a.tobytes()) for a in args]
    for a in args:a.flags.writeable=False
    result=dict(scores=owner.score_accuracy(*args),bins=owner.fixed_probability_bins(*args))
    assert [(a.dtype,a.shape,a.tobytes()) for a in args]==before
    assert all(a.flags.writeable is False for a in args)
    json.dumps(result,allow_nan=False)
