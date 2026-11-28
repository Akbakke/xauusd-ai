"""Pure run/design/lineage/schedule/alias checks; no model or payload execution."""
import copy,json
from pathlib import Path
import pytest
from gx1.audit import native_future_state_fixed_continuation_v1 as f
RUN='NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_003'


def metadata():
    paths=f.run_paths(RUN);design=json.loads(Path(paths['design']['path']).read_text());parent=json.loads(Path(design['producer_plan']['path']).read_text());original=json.loads(Path(design['original_producer_plan']['path']).read_text());receipt=json.loads(Path(design['authentic_start_FINAL864_receipt']['path']).read_text());report=json.loads(Path(design['producer_result']['path']).read_text())['report'];return paths,design,parent,original,receipt,report


def test_selected_run_has_canonical_policy_bound_paths_and_no_arbitrary_path():
    paths=metadata()[0];assert paths['name']==RUN and paths['stage']==RUN.lower()and paths['plan']==f.REPO/'configs/research'/(RUN+'.json')and paths['root']==Path('/home/andre2/GX1_RUNS')/RUN
    assert paths['design']['path']==str(f.E/'V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_003.json')and paths['snapshot']==f.E/'V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_20261128_003.json'
    with pytest.raises((ValueError,RuntimeError)):f.run_paths(RUN+'/../../elsewhere')
    with pytest.raises((ValueError,RuntimeError)):f.run_paths('NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20260230_004')
    with pytest.raises((ValueError,RuntimeError)):f.run_paths('NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_999')


@pytest.mark.parametrize('bad',['run_id','end','budget','candidate','tail'])
def test_design_cannot_change_selected_run_fixed_budget_or_candidate(bad):
    paths,design,*_=metadata()
    if bad=='run_id':design['experiment_id']=RUN[:-3]+'004'
    if bad=='end':design['scope']['locked_final_global_cursor']=1296
    if bad=='budget':design['scope']['new_optimizer_updates']=432
    if bad=='candidate':design['phase_alias']['actual_candidate']='FINAL865'
    if bad=='tail':design['order']['repeat_original_take_indices_start_inclusive']=0
    with pytest.raises(ValueError):f.validate_design(RUN,design)


def test_exact864_prefix_plus_original216_tail_has_only1080_candidate():
    paths,design,parent,original,receipt,report=metadata();assert f.validate_design(RUN,design)==(864,1080);schedule=f.continuation_schedule(parent,original,design)
    for key in('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags'):assert schedule[key][:864]==parent['schedule'][key]and schedule[key][864:]==original['schedule'][key][216:432]
    assert schedule['logical_batches']==schedule['optimizer_updates']==1080 and schedule['passes']==5 and f.continuation_trajectory(parent,schedule,paths['design'])['fixed_continuation_design']==paths['design']
    assert f.operation_limits()==design['limits']and design['checkpoint_cursors']==[865,1080]
    f.validate_start_receipt(parent,receipt,design,report,864)


def test_future_fixed_parent_uses_own_generic_fullproof_schema_without_legacy_fallback():
    paths,design,parent,original,receipt,report=metadata();new_schedule=f.continuation_schedule(parent,original,design);new_trajectory=f.continuation_trajectory(parent,new_schedule,paths['design']);future=copy.deepcopy(parent);future.update(schema_version='gx1_native_v3_fixed_continuation_plan_v1',schedule=new_schedule);future['metadata']['trajectory']=new_trajectory;proof=copy.deepcopy(receipt['proof']);proof['trajectory']=f.core.state_sha(new_trajectory);proof['position']=f.core.state_sha(f.train.position(new_schedule,1080));new_ref={'path':'/synthetic/fixed003.json','sha256':'a'*64};new_cp={'path':'/synthetic/FINAL1080.pt','sha256':'b'*64};new_receipt_ref={'path':'/synthetic/FINAL1080.json','sha256':'c'*64};r=copy.deepcopy(receipt);r.update(original_plan=new_ref,execution_plan=new_ref,checkpoint=new_cp,logical_cursor=1080,optimizer_updates=1080,trajectory_sha256=new_trajectory['contract_sha256'],proof=proof);d={'producer_plan':new_ref,'authentic_start_FINAL1080_checkpoint':new_cp,'authentic_start_FINAL1080_receipt':new_receipt_ref};generic_report={'checkpoint_receipts':{'1080':new_receipt_ref},'full_final_state_proof':proof}
    assert f.validate_start_receipt(future,r,d,generic_report,1080)==proof
    future['schema_version']='unregistered-parent'
    with pytest.raises(ValueError,match='UNQUALIFIED_PARENT_SCHEMA'):f.validate_start_receipt(future,r,d,generic_report,1080)


@pytest.mark.parametrize('bad',['producer','cursor','proof'])
def test_actual_parent_rejects_wrong_864_start_lineage(bad):
    paths,design,parent,original,receipt,report=metadata()
    if bad=='producer':receipt['original_plan']=receipt['execution_plan']=design['original_producer_plan']
    if bad=='cursor':receipt['logical_cursor']=receipt['optimizer_updates']=865
    if bad=='proof':receipt['proof']['position']='0'*64
    with pytest.raises(ValueError):f.validate_start_receipt(parent,receipt,design,report,864)


def test_comparison_reuses_old_numeric_owner_with_only_actual_phase_labels(monkeypatch):
    old={'actual_before':'BASELINE432','actual_candidate':'FINAL648','delta':{'BASELINE432':.3,'FINAL648':.4,'FINAL648_minus_BASELINE432':.1},'unknown':None};before=copy.deepcopy(old);monkeypatch.setattr(f.prior,'compare_reviews',lambda a,b:old);new=f.compare_reviews('a','b','BASELINE864','FINAL1080');assert old==before and new['actual_before']=='BASELINE864'and new['actual_candidate']=='FINAL1080'and new['delta']=={'BASELINE864':.3,'FINAL1080':.4,'FINAL1080_minus_BASELINE864':.1}and new['unknown']is None;f.o.development_json_bytes(new)


def test_unique_run_roles_and_exact_dynamic_output_allowlist(tmp_path):
    prefix=f.input_prefix(RUN);next_prefix=f.input_prefix(RUN[:-3]+'004');assert prefix!=next_prefix
    original={prefix+'design':{'path':'/prior/design.json','sha256':'a'*64}};new=f.merge_inputs(original,{next_prefix+'design':{'path':'/new/design.json','sha256':'b'*64}});assert new[prefix+'design']==original[prefix+'design']
    with pytest.raises(ValueError):f.merge_inputs(original,{prefix+'design':{'path':'/wrong/design.json','sha256':'c'*64}})
    plan={'inputs':{'genuine':{'path':'/genuine/FINAL864.pt','sha256':'d'*64}},'starting_cursor':864,'ending_cursor':1080};allowed=f.payload_allowlist(plan,tmp_path);assert allowed=={'/genuine/FINAL864.pt',str(tmp_path/'CURSOR865.pt'),str(tmp_path/'FINAL1080.pt'),str(tmp_path/'FINAL1080'/'PASS2_LOGITS.npz')}
    for path in allowed:f.o.development_io_audit(allowed,'open',(path,'rb',0))
    with pytest.raises(RuntimeError):f.o.development_io_audit(allowed,'open',(str(tmp_path/'FINAL1081.pt'),'rb',0))


@pytest.mark.parametrize('admitted',[False,True])
def test_one_use_gate_stays_per_exact_plan_binding(admitted):
    binding={'path':'/new003/plan.json','sha256':'a'*64};gate=f.gate(binding,admitted);assert f.require_gate(gate,binding,admitted)==gate
    with pytest.raises(ValueError):f.require_gate(gate,{'path':'/old002/plan.json','sha256':'b'*64},admitted)
    gate['remaining_invocations']=bool(gate['remaining_invocations'])
    with pytest.raises(ValueError):f.require_gate(gate,binding,admitted)


def test_changed_policy_design_is_caught_in_bookend_instead_of_losing_terminal(monkeypatch):
    binding={'path':'/canonical003/plan.json','sha256':'a'*64};pref={'path':str(f.o.POLICY),'sha256':'b'*64};boots={'physical_windows_boot_id':'windows','wsl_boot_id':'wsl'};plan={'run_id':RUN,'source_owners':{},'inputs':{},'source_commit':'head'}
    def changed_paths(run_id):raise RuntimeError('policy design binding changed during run')
    monkeypatch.setattr(f,'run_paths',changed_paths);monkeypatch.setattr(f.recovery,'observe_boots',lambda:{'observed':boots,'query_errors':{}});monkeypatch.setattr(f,'ref',lambda p:binding if str(p)==binding['path']else{'path':str(p),'sha256':'c'*64});monkeypatch.setattr(f.subprocess,'check_output',lambda *a,**k:'head\n')
    checks,evidence=f.bookends(plan,binding,pref,boots)
    assert checks=={'plan':True,'sources':True,'inputs':False,'policy':False,'boots':True,'HEAD':True}
    assert evidence['check_errors']['inputs']=={'type':'RuntimeError','message':'policy design binding changed during run'}
