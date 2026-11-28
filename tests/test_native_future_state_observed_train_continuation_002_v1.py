"""Pure metadata/lineage/schedule/alias checks for own FINAL648 -> FINAL864."""
import copy,json
from pathlib import Path
import pytest
from gx1.audit import native_future_state_observed_train_continuation_002_v1 as c


def metadata():
    design=json.loads(Path(c.DESIGN['path']).read_text());parent=json.loads(Path(design['producer_plan']['path']).read_text());original=json.loads(Path(design['unchanged_original_recipe_model_input_owner']['path']).read_text());receipt=json.loads(Path(design['authentic_start_FINAL648_receipt']['path']).read_text());report=json.loads(Path(design['producer_result']['path']).read_text())['report'];return design,parent,original,receipt,report


def test_new864_schedule_preserves_authentic648_prefix_and_exact_original216_tail():
    design,parent,original,receipt,report=metadata();before=copy.deepcopy(parent);schedule=c.continuation_schedule(parent,original,design)
    assert parent==before and schedule['logical_batches']==schedule['optimizer_updates']==864 and schedule['passes']==4
    for key in('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags'):
        assert schedule[key][:648]==parent['schedule'][key]and schedule[key][648:]==original['schedule'][key][216:432]
    for key in('raw_permutation_state_sha256_by_pass','actual_sorted_partition_stream_state_sha256_by_pass'):assert schedule[key][:3]==parent['schedule'][key]and schedule[key][3]==original['schedule'][key][1]
    trajectory=c.continuation_trajectory(parent,schedule);assert trajectory['original_prefix_trajectory_sha256']==parent['metadata']['trajectory']['contract_sha256']and trajectory['schedule']==schedule


@pytest.mark.parametrize('bad',['wrong648hash','duplicate_original_tail','wrong_tail_start'])
def test_schedule_rejects_wrong_parent_or_appended_membership(bad):
    design,parent,original,receipt,report=metadata()
    if bad=='wrong648hash':design['order']['source648_schedule_sha256']='0'*64
    if bad=='duplicate_original_tail':original['schedule']['cohort_takes'][216][0]=original['schedule']['cohort_takes'][216][1]
    if bad=='wrong_tail_start':design['order']['repeat_original_take_indices_start_inclusive']=0
    with pytest.raises(ValueError):c.continuation_schedule(parent,original,design)


@pytest.mark.parametrize('bad',[None,'old432producer','technical_lane','wrong_cursor','stale432trajectory','proof_position'])
def test_start_is_genuine648_under_own_producer_and_trajectory_before_copy(bad):
    design,parent,original,receipt,report=metadata()
    if bad=='old432producer':receipt['original_plan']=receipt['execution_plan']=design['unchanged_original_recipe_model_input_owner']
    if bad=='technical_lane':receipt['producer_lane']='cold'
    if bad=='wrong_cursor':receipt['logical_cursor']=receipt['optimizer_updates']=649
    if bad=='stale432trajectory':parent['metadata']['trajectory']=original['metadata']['trajectory']
    if bad=='proof_position':receipt['proof']['position']='0'*64
    if bad:
        with pytest.raises(ValueError):c.validate_start_receipt(parent,receipt,design,report)
    else:assert c.validate_start_receipt(parent,receipt,design,report)==receipt['proof']and len(receipt['proof'])==10


def test_build_plan_keeps_parent_by_hash_reference_and_has_no_recursive_original_copies(monkeypatch):
    design,parent,original,receipt,report=metadata();identity=parent['physical_input_identity'];extra={'observed_continuation_002_parent_plan':design['producer_plan']};seen=[]
    monkeypatch.setattr(c,'parent_metadata',lambda:(design,parent,original,parent['metadata']['original'],receipt,identity,extra));monkeypatch.setattr(c.core,'future_cuda_require_runtime',lambda runtime:seen.append('fresh'));monkeypatch.setattr(c.train,'runtime_compatible',lambda *a,**k:seen.append('compatible'));monkeypatch.setattr(c.subprocess,'check_output',lambda args,**kwargs:'synthetic-head\n'if args[1]=='rev-parse'else b'old-guard-source');monkeypatch.setattr(c.guard,'native_v3_observed_train_continuation_002_guard_source_bridge',lambda *a,**k:{'synthetic':'guard'})
    plan=c.build_plan({'source_commit':'synthetic-head'})
    assert set(plan['metadata'])=={'parent_plan','original_plan','receipt','trajectory','runtime'}and plan['metadata']['parent_plan']==design['producer_plan']and plan['metadata']['original_plan']==design['unchanged_original_recipe_model_input_owner']
    assert plan['metadata']['receipt']==receipt and plan['authentic_final_checkpoint']==receipt['checkpoint']and plan['starting_cursor']==648 and plan['ending_cursor']==864 and plan['checkpoint_cursors']==[649,864]
    assert len(json.dumps(plan))<len(json.dumps(parent))and seen==['fresh','compatible']


def test_comparison_reuses_qualified_numerics_with_label_only_remap(monkeypatch):
    existing={'actual_before':'BASELINE432','actual_candidate':'FINAL648','all_TF':{'M5':{'NLL':{'BASELINE432':.2,'FINAL648':.3,'FINAL648_minus_BASELINE432':.1},'UNKNOWN':None}},'BASELINE432_verdict':'FAIL','FINAL648_verdict':'FAIL'};before=copy.deepcopy(existing);calls=[]
    def numeric(a,b):calls.append((a,b));return existing
    monkeypatch.setattr(c.prior,'compare_reviews',numeric);result=c.compare_reviews('baseline648','final864')
    assert calls==[('baseline648','final864')]and existing==before and result['actual_before']=='BASELINE648'and result['actual_candidate']=='FINAL864'
    assert result['all_TF']['M5']['NLL']=={'BASELINE648':.2,'FINAL864':.3,'FINAL864_minus_BASELINE648':.1}and result['all_TF']['M5']['UNKNOWN']is None and result['FINAL864_verdict']=='FAIL'
    c.o.development_json_bytes(result)


@pytest.mark.parametrize('admitted',[False,True])
def test_distinct002_gate_and_exact_unchanged_budget(admitted):
    binding={'path':'/new002/plan.json','sha256':'a'*64};gate=c.gate(binding,admitted);assert c.require_gate(gate,binding,admitted)==gate
    assert gate['status']==('ADMITTED_EXACT_V3_OBSERVED_TRAIN_CONTINUATION_002'if admitted else'PREPARED_CLOSED_NOT_ADMITTED')
    gate['remaining_invocations']=bool(gate['remaining_invocations'])
    with pytest.raises(ValueError):c.require_gate(gate,binding,admitted)
    assert c.operation_limits()==metadata()[0]['limits']


def test_002_roles_do_not_replace001_lineage_and_nested_FINAL864_payload_is_allowed(tmp_path):
    original={'observed_continuation_design':{'path':'/original001/design.json','sha256':'a'*64}};merged=c.merge_inputs(original,{'observed_continuation_002_design':c.DESIGN});assert merged['observed_continuation_design']==original['observed_continuation_design']
    with pytest.raises(ValueError):c.merge_inputs(original,{'observed_continuation_design':c.DESIGN})
    allowed=c.payload_allowlist({'inputs':{'genuine':{'path':'/genuine/FINAL648.pt','sha256':'b'*64}}},tmp_path);assert allowed=={'/genuine/FINAL648.pt',str(tmp_path/'CURSOR649.pt'),str(tmp_path/'FINAL864.pt'),str(tmp_path/'FINAL864'/'PASS2_LOGITS.npz')}
    for path in allowed:c.o.development_io_audit(allowed,'open',(path,'rb',0))
    with pytest.raises(RuntimeError,match='PAYLOAD_PATH_FORBIDDEN'):c.o.development_io_audit(allowed,'open',(str(tmp_path/'FINAL648'/'PASS2_LOGITS.npz'),'rb',0))
