"""Small pure-state/contrast/gate checks; no genuine checkpoint or array decoding."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from gx1.audit import native_future_state_frozen_block_attribution_v1 as g


def states():
    a={'backbone.weight':torch.tensor([1.]),'backbone.registered_buffer':torch.tensor([11.]),'readout.weight':torch.tensor([2.]),'readout.registered_buffer':torch.tensor([12.]),'future_state_model_sha256':torch.tensor([7],dtype=torch.uint8)}
    b={k:v.clone()for k,v in a.items()}
    for k in b:
        if k!='future_state_model_sha256':b[k]+=20
    return a,b


def test_composites_include_every_parameter_and_buffer_with_exact_block_ownership():
    a,b=states();partition=g.partition_states(a,b);assert set().union(*map(set,partition.values()))==set(a)
    for backbone,readout in((864,1080),(1080,864)):
        result=g.composite_state({'864':a,'1080':b},partition,backbone,readout);assert set(result)==set(a)
        for k,v in result.items():
            source=(a if backbone==864 else b)if k.startswith('backbone.')else(a if readout==864 else b)if k.startswith('readout.')else a
            assert torch.equal(v,source[k])


@pytest.mark.parametrize('bad',['unowned','changed_invariant','missing_key','wrong_dtype'])
def test_partition_rejects_omission_unowned_keys_changed_identity_and_bad_headers(bad):
    a,b=states()
    if bad=='unowned':a['unowned_buffer']=b['unowned_buffer']=torch.tensor([1.])
    if bad=='changed_invariant':b['future_state_model_sha256'][0]=8
    if bad=='missing_key':b.pop('readout.registered_buffer')
    if bad=='wrong_dtype':b['backbone.weight']=b['backbone.weight'].double()
    with pytest.raises(ValueError,match='FROZEN_BLOCK_ATTRIBUTION_'):g.partition_states(a,b)


def observations():
    proof={k:'a'*64 for k in('trajectory','position','model_state','optimizer_state','scheduler_state','rng_state','training','optimizer_parameter_names','steps','scores')};snapshot={'proof':copy.deepcopy(proof),'modes':{'':True,'backbone':True,'readout':True},'grad_none':True,'nonpersistent_buffers_sha256':'b'*64};return proof,snapshot


@pytest.mark.parametrize('bad',[None,'model_state','optimizer_state','rng_state','modes','grad_none','nonpersistent_buffer'])
def test_per_pass_mutation_is_rejected_before_any_next_copy_or_final_restore(bad):
    expected,before=observations();after=copy.deepcopy(before)
    if bad in('model_state','optimizer_state','rng_state'):after['proof'][bad]='c'*64
    if bad=='modes':after['modes']['readout']=False
    if bad=='grad_none':after['grad_none']=False
    if bad=='nonpersistent_buffer':after['nonpersistent_buffers_sha256']='d'*64
    if bad:
        with pytest.raises(ValueError,match='PER_PASS_MUTATION_BEFORE_NEXT_COPY'):g.require_arm_invariants(before,after,expected)
        # A later restored live state cannot change the already captured proof strings.
        restored=copy.deepcopy(before);assert restored==before and after!=before
    else:assert all(g.require_arm_invariants(before,after,expected).values())


@pytest.mark.parametrize('bad',[None,'signed_zero','numeric_review','batch_hash'])
def test_self_parity_requires_exact_F32_bits_and_original_numeric_review(bad):
    saved={k:np.zeros((2,5),np.float32)for k in('activity_logits','side_logits')};actual={k:v.copy()for k,v in saved.items()};original={'review':{'verdict':'FAIL'},'batch_score_hashes':[{'side_logits':'a'*64}]};current=copy.deepcopy(original)
    if bad=='signed_zero':actual['side_logits'][0,0]=-0.;assert np.array_equal(actual['side_logits'],saved['side_logits'])
    if bad=='numeric_review':current['review']['verdict']='PASS'
    if bad=='batch_hash':current['batch_score_hashes'][0]['side_logits']='b'*64
    if bad:
        with pytest.raises(ValueError,match='EXACT_SELF_F32_AND_NUMERIC_REVIEW_PARITY'):g.require_self_parity(actual,saved,current,original)
    else:assert g.require_self_parity(actual,saved,current,original)


def review(value):
    def panel():
        return {'count':7,'unique_target_starts':3,'true_positive_count':4,'true_negative_count':3,'model':{'NLL':float(value),'brier':value/10.,'mean_positive_probability':value/10.,'mean_probability_minus_truth_pp':float(value)},'equal_target_weight':{'model':{'NLL':float(value),'brier':value/10.,'mean_positive_probability':value/10.}},'rank_and_variation':{'AUC':value/10.},'calibration':{w:{'mean_probability_minus_truth_pp':float(value)}for w in('row','equal_target')}}
    return {'truth_sha256':'a'*64,'support':{'count':7},'selection_ref':{'path':'/same/selection.json','sha256':'b'*64},'COHORT_score_prior_sha256':'c'*64,'factory_prior_sha256':'d'*64,'spec_sha256':'e'*64,'candidate_TRAIN_comparison_verdict':'FAIL','all_TF':{tf:{task:panel()for task in('activity','side')}for tf in g.train.TF},'by_UTC_target_year':{tf:{str(year):{task:panel()for task in('activity','side')}for year in(2021,2022,2023,2024,2025)}for tf in g.train.TF}}


@pytest.mark.parametrize('unknown',[False,True])
def test_both_conditional_effects_and_interaction_preserve_all_panels_support_and_unknowns(unknown):
    reviews={name:review(v)for name,v in(('SELF864',1.),('SELF1080',8.),('BACKBONE864_READOUT1080',3.),('BACKBONE1080_READOUT864',4.))}
    if unknown:reviews['BACKBONE864_READOUT1080']['by_UTC_target_year']['D1']['2025']['side']['equal_target_weight']=None
    result=g.block_comparisons(reviews);assert set(result['conditional_block_effects'])=={'backbone_given_readout864','backbone_given_readout1080','readout_given_backbone864','readout_given_backbone1080'}
    for tf in g.train.TF:
        panel=result['interaction']['all_TF'][tf]['side'];assert panel['count']==7 and panel['unique_target_starts']==3 and panel['row']['NLL']['interaction']==2. and panel['row_AUC']['interaction']==pytest.approx(.2)
        for year in(2021,2022,2023,2024,2025):
            metric=result['interaction']['by_UTC_target_year'][tf][str(year)]['side']['equal_target']['NLL'];missing=unknown and tf=='D1'and year==2025;assert metric['interaction']==(None if missing else 2.);assert metric['status']==('UNKNOWN'if missing else'DESCRIPTIVE')
    assert result['equal_target_AUC']=='NOT_IMPLEMENTED'and result['new_predictor']is False
    assert json.loads(g.o.development_json_bytes(result))['interaction']['all_TF']['M5']['side']['count']==7


@pytest.mark.parametrize('admitted',[False,True])
def test_gate_opens_execution_only_and_rejects_bool_invocation_counter(admitted):
    binding={'path':'/new/plan.json','sha256':'a'*64};gate=g.gate(binding,admitted);assert g.require_gate(gate,binding,admitted)==gate and gate['execution_authorized']is admitted
    assert all(gate[k]is False for k in('training_authorized','test_authorized','spending_authorized','relaunch_allowed'))
    gate['remaining_invocations']=bool(admitted)
    with pytest.raises(ValueError,match='TYPED_GATE'):g.require_gate(gate,binding,admitted)


def test_all_six_global_flags_stay_closed_in_snapshot(monkeypatch):
    binding={'path':'/new/plan.json','sha256':'a'*64};sr={'path':str(g.SNAPSHOT),'sha256':'b'*64};pref={'path':str(g.o.POLICY),'sha256':'b'*64};policy={k:False for k in g.o.DEV_FLAGS};policy[g.STAGE]={**g.gate(binding,True),'design':g.DESIGN};monkeypatch.setattr(g,'ref',lambda _:sr);monkeypatch.setattr(g,'read',lambda _:policy);assert g.snapshot(binding,pref)==sr
    for flag in g.o.DEV_FLAGS:
        policy[flag]=True
        with pytest.raises(ValueError,match='ALL6_CLOSED'):g.snapshot(binding,pref)
        policy[flag]=False


def test_exact_four_nested_output_allowlist_preserves_inputs_and_forbids_new_checkpoints(tmp_path):
    arms=[{'name':n}for n in('SELF864','SELF1080','BACKBONE864_READOUT1080','BACKBONE1080_READOUT864')];plan={'inputs':{'genuine':{'path':'/old/FINAL864.pt','sha256':'a'*64}},'arms':arms};allowed=g.payload_allowlist(plan,tmp_path);assert allowed=={'/old/FINAL864.pt'}|{str(tmp_path/a['name']/'PASS2_LOGITS.npz')for a in arms}
    for path in allowed:g.o.development_io_audit(allowed,'open',(path,'rb',0))
    for path in(tmp_path/'FINAL1080.pt',tmp_path/'OTHER'/'PASS2_LOGITS.npz'):
        with pytest.raises(RuntimeError,match='PAYLOAD_PATH_FORBIDDEN'):g.o.development_io_audit(allowed,'open',(str(path),'rb',0))
    limits=g.operation_limits();assert len(limits)==26 and limits['model_restores']==3 and limits['checkpoint_loads']==limits['authentic_checkpoint_prechecks']==limits['block_composite_copies']==2 and limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==324 and limits['logit_archive_writes']==limits['logit_archive_readbacks']==4
    assert all(limits[k]==0 for k in('optimizer_steps','backward_calls','autograd_calls','checkpoint_writes','checkpoint_readbacks','normalization_fits','prior_fits','observer_records'))


def test_failed_final_nonpersistent_buffer_gate_cannot_publish_PASS_even_with_authentic_full10(monkeypatch,tmp_path):
    expected,before=observations();final_snapshot=copy.deepcopy(before);final_snapshot['nonpersistent_buffers_sha256']='c'*64;final_restored=False
    with pytest.raises(ValueError,match='PER_PASS_MUTATION_BEFORE_NEXT_COPY'):g.require_arm_invariants(before,final_snapshot,expected)
    published={}
    def publish(root,name,value):
        published.update(json.loads(g.o.development_json_bytes(value)));return {'path':str(root/name),'sha256':'a'*64}
    monkeypatch.setattr(g.o,'development_publish',publish);limits=g.operation_limits();g.publish_state_invariants(tmp_path,limits,limits,[{'checks':{'state':True}}for _ in range(4)],expected,expected,final_restored)
    assert published['status']=='FAILED_CONSUMED'and published['final_restore_all_checks_pass']is False and published['final_authentic1080_all10_restored']is True
