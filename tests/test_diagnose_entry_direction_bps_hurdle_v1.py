"""Synthetic checks for the fixed 5-minute +/-10-bps event diagnostic."""
import numpy as np
import pytest
from gx1.contracts import entry_direction_saved_horizons_v1 as endpoint
from gx1.scripts import diagnose_entry_direction_bps_allocation_v1 as diag
from gx1.scripts import diagnose_entry_direction_10min_crosshorizon_v1 as rank
from tests.test_diagnose_entry_direction_10min_crosshorizon_v1 import synthetic_join_inputs, simple_rank_cohorts

diag.np=np


def joined(fixture):
    target,natives,controls,specs,cutoff=fixture
    return diag.five_minute_hurdle_cohorts(target,natives,controls,specs,endpoint_owner=endpoint,cutoff_ns=cutoff)


def test_hurdle_includes_zero_and_excludes_threshold_equality():
    fixture=synthetic_join_inputs();target,natives,controls,specs,_=fixture;first=specs[0]['left']
    for j,value in enumerate((10.,-10.,0.,10.01,-10.01)):
        i=first+j; target['mid_return_bps'][i,1]=value;target['mid_move'][i,1]=value/10000.
        target['sign'][i,1]=np.sign(value);controls['sign'][i,1]=np.sign(value);natives[0]['sign'][j,1]=np.sign(value)
    result=joined(fixture)[0]
    assert result['parent_rows'][:5].tolist() == list(range(first,first+5))
    assert result['labels_long'][:5].tolist() == [0,0,0,1,0]
    assert result['labels_short'][:5].tolist() == [0,0,0,0,1]
    assert result['counts']['observed_zero'] >= 1


def test_fold_end_excludes_immature_even_if_globally_observed():
    fixture=synthetic_join_inputs();target,natives,controls,specs,_=fixture;right=specs[0]['right']
    end=int(target['terminal_time_ns'][right-1,1]);specs[0]['holdout_end_ns']=end
    controls['holdout_end_ns_exclusive'][specs[0]['left']:right]=end
    result=joined(fixture)[0];assert result['counts']['immature'] == 1 and right-1 not in result['parent_rows']


@pytest.mark.parametrize('fault,match', [('clock','HURDLE_CONTROL_decision_ns'),
    ('native','HURDLE_NATIVE_parent_rows'),('fold','HURDLE_REFERENCE_FOLD'),
    ('domain','HURDLE_ET_SCORE'),('bps_sign','HURDLE_TARGET_BPS_SIGN')])
def test_hurdle_join_rejects_misalignment(fault,match):
    fixture=synthetic_join_inputs();target,natives,controls,specs,_=fixture;i=specs[0]['left']
    if fault == 'clock':controls['decision_ns'][i]+=1
    elif fault == 'native':natives[0]['parent_rows'][0]+=1
    elif fault == 'fold':controls['fold_id'][i]=4
    elif fault == 'domain':controls['saved_ET_FULL_probability_up'][i,1]=1.1
    else:target['mid_return_bps'][i,1]*=-1.
    with pytest.raises(RuntimeError,match=match):joined(fixture)


def test_short_inverts_score_and_uses_short_event_label():
    fixture=synthetic_join_inputs();native=fixture[1][0]
    native['logit_up'][:,1]=np.arange(len(native['parent_rows']),dtype=np.float32)
    cohorts=joined(fixture);short=diag.hurdle_rank_cohorts(cohorts,side='short')[0]
    np.testing.assert_array_equal(short['scores']['head_5']['native'],-cohorts[0]['native_score'])
    np.testing.assert_array_equal(short['labels'],cohorts[0]['labels_short'])
    with pytest.raises(RuntimeError,match='HURDLE_SIDE'):diag.hurdle_rank_cohorts(cohorts,side='flat')


def test_absolute_intervals_are_opt_in_and_share_paired_draws():
    cohorts=simple_rank_cohorts()
    old=rank.rank_with_block_uncertainty(cohorts,scheme='utc_week_monday',draws=64,seed=5)
    new=rank.rank_with_block_uncertainty(cohorts,scheme='utc_week_monday',draws=64,seed=5,absolute_auc_intervals=True)
    absolute=new.pop('absolute_auc_intervals');assert new == old
    for head in ('head_5','head_15'):
        assert absolute[head]['native'] == {'low':1.,'high':1.,'level':.99}
        assert absolute[head]['et252'] == {'low':0.,'high':0.,'level':.99}


def test_magnitude_regime_uses_causal_tags_without_future_selection():
    fixture=synthetic_join_inputs();native_folds=fixture[1]
    for native in native_folds:
        n=len(native['parent_rows']);native['context__h1_atr_z_sign']=np.resize(np.array([-1,1],np.int8),n)
    cohorts=joined(fixture);result=diag.magnitude_h1_regime_cohorts(cohorts,native_folds,fixture[3])
    assert len(result['overall'])==4 and len(result['stratified'])==8
    for source,overall,cnt in zip(cohorts,result['overall'],result['counts']):
        np.testing.assert_array_equal(overall['labels'],source['labels_long']|source['labels_short'])
        assert cnt['positive']['parents']+cnt['nonpositive']['parents']==len(source['parent_rows'])
        assert set(np.unique(overall['scores']['head_5']['et252'])).issubset({0.,1.})


@pytest.mark.parametrize('fault,match',[('missing','REGIME_TAG_DOMAIN'),('parent','REGIME_PARENT_JOIN')])
def test_magnitude_regime_rejects_invalid_tags_or_parent_join(fault,match):
    fixture=synthetic_join_inputs();natives=fixture[1]
    for native in natives:native['context__h1_atr_z_sign']=np.full(len(native['parent_rows']),1,np.int8)
    cohorts=joined(fixture)
    if fault=='missing':natives[0]['context__h1_atr_z_sign'][0]=-2
    else:cohorts[0]['parent_rows'][0]+=1
    with pytest.raises(RuntimeError,match=match):diag.magnitude_h1_regime_cohorts(cohorts,natives,fixture[3])
