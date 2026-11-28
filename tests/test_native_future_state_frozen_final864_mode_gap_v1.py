"""Pure FINAL864 lineage/contract/admission/IO tests; reuse qualified pass owner."""
import copy,json
from pathlib import Path
import pytest
from gx1.audit import native_future_state_frozen_final864_mode_gap_v1 as g


def metadata():
    design=json.loads(Path(g.DESIGN['path']).read_text());parent=json.loads(Path(design['producer_plan']['path']).read_text());receipt=json.loads(Path(design['authentic_FINAL864_receipt']['path']).read_text());report=json.loads(Path(design['producer_result']['path']).read_text())['report'];return design,parent,receipt,report


@pytest.mark.parametrize('bad',[None,'old432producer','old648cursor','technical_lane','wrong_proof'])
def test_only_genuine_864_own_producer_trajectory_full10_can_restore(bad):
    design,parent,receipt,report=metadata()
    if bad=='old432producer':receipt['original_plan']=receipt['execution_plan']=design['original_producer_plan']
    if bad=='old648cursor':receipt['logical_cursor']=receipt['optimizer_updates']=648
    if bad=='technical_lane':receipt['producer_lane']='cold'
    if bad=='wrong_proof':receipt['proof']['position']='0'*64
    if bad:
        with pytest.raises(ValueError):g.validate_final_receipt(parent,receipt,design,report)
    else:assert g.validate_final_receipt(parent,receipt,design,report)==receipt['proof']and len(receipt['proof'])==10


@pytest.mark.parametrize('wrong',['none','cohort_file_SHA','factory_file_SHA','spec_file_SHA'])
def test_prior_and_spec_identity_uses_internal_contract_not_file_SHA(wrong):
    design=metadata()[0]
    def read(key):return json.loads(Path(design[key]['path']).read_text())
    numeric=read('saved_eval_numeric_owner_score');wrapper=read('saved_eval_score_wrapper');cohort=read('frozen_COHORT_prior');factory=read('frozen_factory_prior');spec=read('frozen_numeric_owner_spec')
    if wrong=='cohort_file_SHA':numeric['review']['COHORT_score_prior_sha256']=design['frozen_COHORT_prior']['sha256']
    if wrong=='factory_file_SHA':numeric['review']['factory_prior_sha256']=design['frozen_factory_prior']['sha256']
    if wrong=='spec_file_SHA':numeric['review']['spec_sha256']=design['frozen_numeric_owner_spec']['sha256']
    if wrong=='none':g.validate_score_contracts(numeric,wrapper,design,cohort,factory,spec)
    else:
        with pytest.raises(ValueError,match='INTERNAL_PRIOR_SPEC_CONTRACTS'):g.validate_score_contracts(numeric,wrapper,design,cohort,factory,spec)


@pytest.mark.parametrize('admitted',[False,True])
def test_zero_update_gate_never_opens_training_even_when_execution_admitted(admitted):
    binding={'path':'/new/diagnostic.json','sha256':'a'*64};gate=g.gate(binding,admitted);assert g.require_gate(gate,binding,admitted)==gate and gate['training_authorized']is gate['test_authorized']is gate['spending_authorized']is False
    gate['training_authorized']=True
    with pytest.raises(ValueError,match='TYPED_GATE'):g.require_gate(gate,binding,admitted)


def test_snapshot_requires_all_six_global_flags_false(monkeypatch):
    binding={'path':'/new/plan.json','sha256':'a'*64};sr={'path':str(g.SNAPSHOT),'sha256':'b'*64};pref={'path':str(g.o.POLICY),'sha256':'b'*64};policy={k:False for k in g.o.DEV_FLAGS};policy[g.STAGE]={**g.gate(binding,True),'design':g.DESIGN};monkeypatch.setattr(g,'ref',lambda p:sr);monkeypatch.setattr(g,'read',lambda r:policy)
    assert g.snapshot(binding,pref)==sr
    for flag in g.o.DEV_FLAGS:
        policy[flag]=True
        with pytest.raises(ValueError,match='ALL6_CLOSED'):g.snapshot(binding,pref)
        policy[flag]=False


def test_exact_five_output_archive_allowlist_and_no_checkpoint_outputs(tmp_path):
    allowed=g.payload_allowlist({'inputs':{'genuine':{'path':'/original/FINAL864.pt','sha256':'a'*64}}},tmp_path);expected={'/original/FINAL864.pt'}|{str((tmp_path/(p['label']+'_LOGITS.npz')).resolve())for p in g.o.TRAIN_EVAL_GAP_PASSES};assert allowed==expected
    for path in allowed:g.o.development_io_audit(allowed,'open',(path,'rb',0))
    for name in('TRAIN_DRAW_05_LOGITS.npz','FINAL864.pt'):
        with pytest.raises(RuntimeError,match='PAYLOAD_PATH_FORBIDDEN'):g.o.development_io_audit(allowed,'open',(str(tmp_path/name),'rb',0))


def test_prefixed_input_roles_preserve_original_lineage_and_exact_zero_update_budget():
    original={'recovery_design':{'path':'/original/recovery.json','sha256':'a'*64}};merged=g.merge_inputs(original,{'frozen_final864_mode_gap_design':g.DESIGN});assert merged['recovery_design']==original['recovery_design']
    with pytest.raises(ValueError,match='INPUT_ROLE_COLLISION'):g.merge_inputs(original,{'recovery_design':g.DESIGN})
    limits=g.operation_limits();assert limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==405 and limits['logit_archive_writes']==limits['logit_archive_readbacks']==5 and limits['saved_eval_logit_loads']==1
    assert all(limits[k]==0 for k in('backward_calls','autograd_calls','optimizer_steps','checkpoint_writes','checkpoint_readbacks','observer_records','normalization_fits','prior_fits'))
