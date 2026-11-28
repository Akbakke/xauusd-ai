"""Pure continuation schedule/phase/scope tests; no model or real payload decode."""
import copy,json
from pathlib import Path
import pytest
from gx1.audit import native_future_state_observed_train_continuation_v1 as c


def source_schedule():
    design=json.loads(Path(c.DESIGN['path']).read_text());producer=json.loads(Path(design['producer_fit_plan']['path']).read_text());return producer,design


def test_continuation_keeps_original432_and_appends_locked_second216_once():
    producer,design=source_schedule();before=copy.deepcopy(producer);result=c.continuation_schedule(producer,design);old=producer['schedule']
    assert producer==before and result['logical_batches']==result['optimizer_updates']==648 and result['passes']==3
    for key in('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags'):
        assert result[key][:432]==old[key]and result[key][432:]==old[key][216:432]
    take=c.o.np.concatenate([c.o.np.asarray(v,c.o.np.int64)for v in result['cohort_takes'][432:]])
    assert c.o.np.array_equal(c.o.np.sort(take),c.o.np.arange(20727,dtype=c.o.np.int64))
    trajectory=c.continuation_trajectory(producer,result);assert trajectory['schedule']==result and trajectory['original_prefix_trajectory_sha256']==producer['metadata']['trajectory']['contract_sha256']
    assert result['original_take_indices_appended']==list(range(216,432))


@pytest.mark.parametrize('bad',['duplicate','wrong_start','nonascending'])
def test_continuation_rejects_unbound_membership_or_order(bad):
    producer,design=source_schedule()
    if bad=='duplicate':producer['schedule']['cohort_takes'][216][0]=producer['schedule']['cohort_takes'][216][1]
    if bad=='wrong_start':design['order']['reuse_original_take_indices_start_inclusive']=0
    if bad=='nonascending':producer['schedule']['cohort_takes'][216].reverse()
    with pytest.raises(ValueError):c.continuation_schedule(producer,design)


def panel(unknown=False):
    return {'count':0 if unknown else 2,'unique_target_starts':0 if unknown else 2,'true_positive_count':0 if unknown else 1,'true_negative_count':0 if unknown else 1,'model':None if unknown else{'NLL':.5,'brier':.2,'mean_positive_probability':.5,'mean_probability_minus_truth_pp':0.},'equal_target_weight':None if unknown else{'model':{'NLL':.5,'brier':.2,'mean_positive_probability':.5}},'rank_and_variation':{'AUC':None if unknown else .6},'calibration':{'row':None if unknown else{'mean_probability_minus_truth_pp':0.},'equal_target':None if unknown else{'mean_probability_minus_truth_pp':0.}}}


def test_FINAL648_comparison_keeps_every_TF_year_worsening_unknown_and_old_verdict():
    all_tf={tf:{task:panel(tf=='H4')for task in('activity','side')}for tf in c.train.TF};years={tf:{str(y):copy.deepcopy(all_tf[tf])for y in range(2021,2026)}for tf in c.train.TF};before={'truth_sha256':'truth','support':{'same':True},'selection_ref':{'fixed':True},'COHORT_score_prior_sha256':'cohort','factory_prior_sha256':'factory','all_TF':all_tf,'by_UTC_target_year':years,'candidate_TRAIN_comparison_verdict':'FAIL'};after=copy.deepcopy(before);after['all_TF']['M5']['side']['model']['NLL']=.7;after['by_UTC_target_year']['D1']['2025']['side']['model']['NLL']=.4
    result=c.compare_reviews(before,after);assert result['actual_before']=='BASELINE432'and result['actual_candidate']=='FINAL648'and result['FINAL648_verdict']=='FAIL'
    assert set(result['all_TF'])==set(c.train.TF)and all(set(v)==set(map(str,range(2021,2026)))for v in result['by_UTC_target_year'].values())
    assert result['all_TF']['M5']['side']['row']['NLL']['FINAL648_minus_BASELINE432']==pytest.approx(.2)
    assert result['by_UTC_target_year']['D1']['2025']['side']['row']['NLL']['FINAL648_minus_BASELINE432']==pytest.approx(-.1)
    assert result['all_TF']['H4']['side']['row_AUC']['FINAL648_minus_BASELINE432']is None
    c.o.development_json_bytes(result)
    after['support']={'changed':True}
    with pytest.raises(ValueError,match='SAME_SCORE_POPULATIONS'):c.compare_reviews(before,after)


def test_exact_new_counts_include_only216updates81terminal_forwards_and216observations():
    limits=c.operation_limits();assert limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==297
    assert limits['backward_calls']==limits['optimizer_steps']==limits['observer_records']==216
    assert limits['checkpoint_loads']==limits['model_restores']==1 and limits['checkpoint_writes']==limits['checkpoint_readbacks']==limits['checkpoint_prepublication_loads']==limits['checkpoint_receipt_writes']==2
    assert limits['logit_archive_writes']==limits['logit_archive_readbacks']==1 and limits['autograd_calls']==limits['normalization_fits']==limits['prior_fits']==0


@pytest.mark.parametrize('admitted',[False,True])
def test_gate_uses_only_new_flat_one_use_authority_and_rejects_bool_integer_alias(admitted):
    binding={'path':'/new/plan.json','sha256':'a'*64};value=c.gate(binding,admitted);assert c.require_gate(value,binding,admitted)==value and value['relaunch_allowed']is value['test_authorized']is value['spending_authorized']is False
    value['remaining_invocations']=bool(value['remaining_invocations'])
    with pytest.raises(ValueError,match='TYPED_GATE'):c.require_gate(value,binding,admitted)


def test_new_prefixed_roles_preserve_original_recovery_design_and_fail_on_any_collision():
    original={'recovery_design':{'path':'/old/lineage.json','sha256':'a'*64}};extra={'observed_continuation_design':c.DESIGN};merged=c.merge_inputs(original,extra);assert merged['recovery_design']==original['recovery_design']and merged['observed_continuation_design']==c.DESIGN
    with pytest.raises(ValueError,match='INPUT_ROLE_COLLISION'):c.merge_inputs(original,{'recovery_design':c.DESIGN})


def test_FINAL648_namespace_logit_and_both_new_checkpoints_are_exact_payload_allowlist(tmp_path):
    plan={'inputs':{'final':{'path':'/original/FINAL432.pt','sha256':'a'*64}}};allowed=c.payload_allowlist(plan,tmp_path)
    assert allowed=={'/original/FINAL432.pt',str(tmp_path/'CURSOR433.pt'),str(tmp_path/'FINAL648.pt'),str(tmp_path/'FINAL648'/'PASS2_LOGITS.npz')}
    for value in allowed:c.o.development_io_audit(allowed,'open',(value,'rb',0))
    for value in(str(tmp_path/'PASS2_LOGITS.npz'),str(tmp_path/'FINAL649.pt')):
        with pytest.raises(RuntimeError,match='PAYLOAD_PATH_FORBIDDEN'):c.o.development_io_audit(allowed,'open',(value,'rb',0))
