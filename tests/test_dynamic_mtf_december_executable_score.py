"""Synthetic probabilities and clocks only; no checkpoint, forecast forward or reserved market observations."""
from itertools import product

import numpy as np
import pytest

from gx1.contracts import entry_dynamic_mtf_december_score_v1 as score


def test_phase_probabilities_and_every_interval_against_full_market_paths():
    q = np.array([[.60, .25, .15], [.30, .01, .69]])
    kernels = np.random.default_rng(33).uniform(size=(2, 3, 3, 3))
    kernels /= kernels.sum(axis=-1, keepdims=True)
    start = np.array([1, 0])
    cdf, hazard = score.issued_events(q, kernels, start, 4, interval=True)
    for b, s in enumerate(start):
        previous = np.array([0., 0.])
        for lead in range(1, 5):
            totals = np.zeros(2)
            for path in product(range(3), repeat=lead):
                prob = q[b, path[0]]
                for k in range(1, lead):
                    prob *= kernels[b, k-1, path[k-1], path[k]]
                exits = [i for i, j in enumerate(path) if j != s]
                totals += prob * np.array([bool(exits), bool(exits and s in path[exits[0]+1:])])
            np.testing.assert_allclose(hazard[b, lead-1], (totals-previous)/(1-previous), atol=2e-14)
            previous = totals
        np.testing.assert_allclose(cdf[b], previous)


def test_zero_survival_is_undefined_on_the_same_risk_set_and_cross_midnight_is_exact():
    q = np.array([[0., 1., 0.]])
    k = np.array([[0., 1., 0.], [1., 0., 0.], [1., 0., 0.]])
    _, h = score.issued_events(q, k, np.array([0]), 1440, interval=True)
    assert h[0, 0, :].tolist() == [1., 0.] and np.isnan(h[0, 1, 0]) and h[0, 1, 1] == 1.
    assert np.isnan(h[0, 2, 1])
    c, moments = score._d1_day_sums(np.array([score.DAY - 2*score.MIN]), np.array([[3, 0, 0]]), (h, h))
    assert c[0, :, 0, 0].tolist() == [1, 2] and c[0, :, 0, 2].tolist() == [0, 2]
    assert c[0, :, 1, 2].tolist() == [0, 1] and np.all(c[..., 1] == 0)
    assert moments[0, :, 0, 0, 0].tolist() == [1., 0.]


def test_censoring_stops_before_gap_and_return_can_only_follow_exit():
    states = np.array([[1, 0, -1, 1], [1, 2, 0, 1], [0, 1, 2, 1], [1, 1, 1, 1]])
    known = states >= 0
    traces = score._observed_prefix(states, known, np.array([1, 1, 1, 1]), np.array([True, True, False, True]))
    assert traces.tolist() == [[2, 2, 0], [4, 2, 4], [-1, 0, 0], [4, 0, 0]]
    axis = np.arange(0, 251) * score.MIN
    axis = np.delete(axis, [4, 8])
    right = np.flatnonzero(np.diff(axis) > score.MIN)+1
    assert score._gap_groups(axis, right, right[:1], np.array([0, 5, 9, 8])*score.MIN).tolist() == [3, 2, 0, 4]


def test_storage_uses_actual_npy_headers_and_one_staged_dataset(tmp_path):
    p = score.december_score_storage_plan(max_published_json_bytes=67108864, max_checkpoint_total_bytes=4294967296)
    assert p['total_npy_bytes'] == 499119576 and p['required_free_bytes'] == 4928304600
    assert p['event_npy_total_bytes'] == 3276152 and p['second_npy_copy_bytes'] == 0
    for key, (shape, typ) in score.december_event_layout(score.ROWS).items():
        f = tmp_path / (key + '.npy')
        z = np.lib.format.open_memmap(f, mode='w+', dtype=typ, shape=shape)
        del z
        assert f.stat().st_size == p['event_npy'][key]


def test_outcome_day_estimates_and_episode_ranks_are_distinct_from_row_weighting():
    # Two days with unequal counts, repeated close times within an episode, and a forced gap.
    t = np.array([1, 6, 11, 16, 1441, 1446]) * score.MIN
    y = np.array([0, 0, 2, 2, 0, 2])
    raw = np.array([.2, .3, .8, .7, .9, .1])
    m = np.column_stack((1-raw, np.zeros_like(raw), raw))
    prob = {'model': m, 'markov': np.tile([.5, 0, .5], (6,1)), 'prior': np.tile([.3,.4,.3], (6,1))}
    out, err, ranks, direct = score._endpoint_rows(y, prob, raw, t)
    delta = err['model']['brier'] - err['markov']['brier']
    assert out['scores']['model']['brier']['row']-out['scores']['markov']['brier']['row'] == pytest.approx(delta.mean())
    assert out['scores']['model']['brier']['day']-out['scores']['markov']['brier']['day'] == pytest.approx((delta[:4].mean()+delta[4:].mean())/2)
    assert score._episodes(*direct, ranks)['episodes'] == 4
    ci = score._intervals(t, err, ranks, direct)['proper']['brier_minus_markov']['day']
    assert ci == pytest.approx([delta[:4].mean(), delta[4:].mean()])


def test_registered_full_row_report_on_synthetic_issue_clocks_only():
    import json
    n = score.ROWS
    ids = np.arange(41866, 41866+n)
    issues = score.FIT_END + (np.arange(n)+1)*5*score.MIN
    a = {k: np.zeros(sh, typ) for k, (sh, typ) in score.december_event_layout(n).items()}
    a.update(parent_rows=ids, issue_ns=issues, probabilities=np.broadcast_to(
        np.array([.2, .3, .5], np.float32), (n,5,1440,3)))
    a['state_evidence'][:, :, 0] = ((np.arange(n)%4)-1)[:, None]
    a['state_evidence'][:, :, 1] = (np.arange(n)%3)[:, None]
    a['state_evidence'][:, :, 2:] = (np.arange(n)%2)[:, None, None]
    a['event_p'][:] = .5
    a['candle'][:] = np.arange(n)[:, None] % 3
    a['path_evidence'][:, :, 0] = -1   # Every synthetic event path is censored at the start.
    prior = np.tile([.3, .3, .4], (5, 1))
    grid = np.broadcast_to(prior[:, None, None, :], (5, 4, 1440, 3))
    result = score.registered_report(a, prior, grid)
    assert result['decision'] == 'FAIL_REGISTERED_GATE' and result['any_unestimable']
    assert result['timeframes']['M5']['native_endpoint']['nonflat_rows'] == (n//3)*2
    assert result['timeframes']['D1']['events']['first_return']['gate'] is None
    assert result['timeframes']['D1']['events']['first_return']['total']['unknown_start'] == n
    assert 'missing_clock' in result['timeframes']['H4']['by_path_gap']
    json.dumps(result, allow_nan=False)


def test_batch_extracts_known_negative_unknown_and_positive_events_from_identical_issued_law():
    n=2
    p = np.broadcast_to(np.array([.2,.5,.3], np.float32), (n,5,1440,3))
    kernel = np.broadcast_to(np.array([.2,.5,.3], np.float32), (n,5,1439,3,3))
    t = np.broadcast_to(np.array([.2,.5,.3]), (5,3,3))
    markov = np.broadcast_to(np.array([.2,.5,.3]), (5,4,1440,3))
    i = score.FIT_END + np.array([24*60-2, 24*60+3])*score.MIN
    truth=np.ones((n,5,1440),np.int8)
    truth[0,:,1:3]=0
    observed=np.ones_like(truth,dtype=bool)
    observed[1,:,1:]=False
    truth[~observed]=-1
    labels={'current_state':np.ones((n,5),np.int8),'current_observed':np.ones((n,5),bool),'issue_ns':i,
            'state':truth,'observed':observed,'path_observed':observed,'maturity_ns':
            np.broadcast_to((i[:,None]+np.arange(1,1441)*score.MIN)[:,None,:], truth.shape)}
    evidence=score.batch_evidence(p,kernel,markov,t,labels,np.ones((n,5),bool),np.ones((n,5),np.int8),
                                  np.ones((n,5)),np.array([0,2]))
    assert evidence['path_evidence'][0,4].tolist()==[1440,2,4]
    assert evidence['path_evidence'][1,4].tolist()==[1,0,0]
    assert evidence['d1_hazard_counts'][:, :, 0,0].sum()==3
    assert evidence['d1_hazard_counts'][:, :, 1,0].sum()==5
    assert evidence['state_evidence'][1,0,1] == -1 and evidence['state_evidence'][1,0,2] == 1
    for event in (2,3):
        g=score._aux(evidence,i,4,event)
        assert g['total']['positive']==1 and g['total']['censored']==1
        assert g['total']['model']['brier'] == pytest.approx(g['total']['markov']['brier'],abs=5e-9)


def test_score_disk_guard_checks_each_volume_before_staging(monkeypatch, tmp_path):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    p = score.december_score_storage_plan(max_published_json_bytes=1000, max_checkpoint_total_bytes=2000)
    observation = {'host_base_path':'C:\\WSL', 'host_drive_root':'C:\\'}
    for k in ('guest_available_bytes','host_available_bytes','mounted_host_available_bytes'):
        observation[k] = p['required_free_bytes']
    monkeypatch.setattr(authority, '_observe_wsl_host_storage', lambda root: observation)
    assert score.require_december_score_storage(tmp_path,p)['observation']==observation
    for k in ('guest_available_bytes','host_available_bytes','mounted_host_available_bytes'):
        observation[k] -= 1
        with pytest.raises(ValueError, match='INSUFFICIENT_ACTUAL_DISK'):
            score.require_december_score_storage(tmp_path,p)
        observation[k] += 1
    with pytest.raises(ValueError, match='HOST_IDENTITY'):
        score.require_december_score_storage(tmp_path,p,expected_host=['different','volume'])


def test_score_adapter_requires_the_actual_admission_refs_and_uses_physical_tail(monkeypatch, tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4

    def file(name, content):
        p = tmp_path / (name+'.json')
        raw = json.dumps(content).encode()
        p.write_bytes(raw)
        return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()}

    signs = dict(zip(score.TFS, (-1,-1,-1,1,1)))
    refs = {k:file(k,{}) for k in ('calendar_authority','terminal_checkpoint','fit_terminal_review','target_manifest')}
    frozen = {'parents':41866, 'probabilities':np.full((5,3),1/3).tolist(),
              'transition_probabilities':np.full((5,3,3),1/3).tolist()}
    refs['prior']=file('prior',frozen)
    refs['candle']=file('candle',{'status':'FROZEN_PREFIX_CANDLE_REFERENCE','fit_parents':41866,'fit_cutoff_ns':score.FIT_END,
                                  'panels':{k:{'orientation':v} for k,v in signs.items()}})
    plan=score.december_score_storage_plan(max_published_json_bytes=2048,max_checkpoint_total_bytes=1024)
    admission={'input_role':'score','source_terminal':refs['fit_terminal_review'],'fit_checkpoint':refs['terminal_checkpoint'],
               'target_manifest':refs['target_manifest'],'calendar_authority':refs['calendar_authority'],
               'storage_plan':plan,'scoring_contract_sha256':score.report.december_oof_scoring_contract()['contract_sha256']}
    refs['scoring_admission']=file('bad',dict(admission, input_role='fit'))
    empty=np.array([],np.int64)
    sim={'receipt_refs':refs,'storage_plan':plan,'expected_host':['a','b'],'candle_orientation':signs,
         'gap_after_rows':empty,'published_closure_after_rows':empty}
    clock=score.FIT_END+np.arange(1442)*score.MIN
    series=SimpleNamespace(metadata={'cutoff_ns':score.SCORE_END},arrays={'validity_ns':clock})
    targets={'dynamic_state_series':series,'prior':frozen,'target_manifest':refs['target_manifest']}
    grid=np.full((5,4,1440,3),1/3)
    kwargs={'prepared':{'december_score_inputs':sim},'targets':targets,'view':{'native_endpoint_loss':True,'zero_support_profile':{}},
            'selected':np.arange(41866,47602),'cutoff':score.SCORE_END,'directory':tmp_path,'markov':grid}
    with pytest.raises(ValueError,match='SIGNED_SCORE_ADMISSION'):
        score.DecemberScorePass(**kwargs)
    refs['scoring_admission']=file('admitted',admission)
    monkeypatch.setattr(score,'require_december_score_storage',lambda directory,plan,expected_host:{'plan':plan,'host_identity':expected_host})
    adapter=score.DecemberScorePass(**kwargs)
    assert adapter.storage['host_identity']==['a','b']
    one=np.ones((1,5),dtype=np.int8)
    q=np.ones((1,5,1440),dtype=np.int8)
    i=clock[:1]
    labels={'state':q,'observed':q.astype(bool),'path_observed':q.astype(bool),'issue_ns':i,
            'current_state':one,'current_observed':one.astype(bool),
            'maturity_ns':np.broadcast_to(i[:,None,None]+np.arange(1,1441)*score.MIN,q.shape)}
    used=[]
    def query(series,*,parent_rows,issue_ns,cutoff_ns,lead_minutes,include_first_flat_activity):
        assert include_first_flat_activity and cutoff_ns==score.SCORE_END
        used.extend(lead_minutes)
        return {'state':q[:,:,lead_minutes[0]-1:lead_minutes[0]], 'observed':q[:,:,:1].astype(bool),
                'first_flat_activity_observed':one.astype(bool),'first_flat_activity_is_flat':one.astype(bool)}
    monkeypatch.setattr(score.target,'query_dynamic_mtf_states',query)
    feature={k:labels[k] for k in ('current_state','current_observed')}
    feature['current_path_observed']=one.astype(bool)
    for k in score.TFS:
        seq=np.zeros((1,2,len(MULTI_TF_PER_BAR_FEATURES_V4)),np.float32)
        seq[0,0,adapter.feature]=9.
        seq[0,1,adapter.feature]=2.
        feature['seq_'+k.lower()]=seq
    raw={'features':feature,'witness':{'parent_rows':[41866],'decision_ns':i.tolist()}}
    p=np.full((1,5,1440,3),1/3,np.float32)
    kernels=np.broadcast_to(p[:,:,1:,None,:],(1,5,1439,3,3))
    out=adapter.batch(np.array([41866]),raw,labels,p,kernels)
    assert used==list(score.LEADS) and out['candle'][0].tolist()==[-2,-2,-2,2,2]
    assert out['h4_gap'].tolist()==[0]
