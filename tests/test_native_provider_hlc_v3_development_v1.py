"""Focused metadata/IO guards; synthetic values, no CUDA/model or real arrays."""
import copy
import json
import subprocess
from pathlib import Path
import pytest
from gx1.audit import native_provider_hlc_learning_v1 as owner


def protocol():
    return json.loads(owner.DEV_PROTOCOL.read_text())


def test_exact_legacy_source_reversal():
    actual=owner.development_reverse_source((owner.REPO/owner.SELF).read_bytes())
    expected=subprocess.check_output(['git','show',owner.DEV_PARENT+':'+owner.SELF],cwd=owner.REPO)
    assert actual==expected


@pytest.mark.parametrize('mutation',['duplicate','extra_dispatch','legacy_edit'])
def test_source_reversal_rejects_other_changes(mutation):
    raw=(owner.REPO/owner.SELF).read_bytes()
    if mutation=='duplicate':raw+=b'# BEGIN GX1 V3 BOUNDED DEVELOPMENT 20261127\n# END GX1 V3 BOUNDED DEVELOPMENT 20261127\n'
    elif mutation=='extra_dispatch':raw+=b" if sys.argv[1:2]==['--v3-development-plan-sha256']:return main_v3_development()\n"
    else:raw=raw.replace(b'One explicitly admitted',b'Changed legacy admitted',1)
    try:actual=owner.development_reverse_source(raw)
    except ValueError:return
    expected=subprocess.check_output(['git','show',owner.DEV_PARENT+':'+owner.SELF],cwd=owner.REPO)
    assert actual!=expected


def test_recipe_derived_from_canonical_owner():
    p=protocol();recipe=owner.development_recipe(p)
    assert recipe==p['recipe']
    assert recipe['passes']==2 and recipe['train_batch_rows']==96 and recipe['eval_batch_rows']==256
    assert p['fit_checkpoint_cursors']==[1,432]


@pytest.mark.parametrize('field,value',[('passes',1),('seed',17),('allocator_phase_policy',{}),('optimizer',{})])
def test_recipe_rejects_changed_decisions(field,value):
    p=protocol();p['recipe'][field]=value
    with pytest.raises(ValueError):owner.development_recipe(p)


@pytest.mark.parametrize('phase',owner.DEV_PHASES)
def test_exact_phase_gate_and_flags(phase):
    binding={'path':'/synthetic/plan.json','sha256':'1'*64};gate=owner.development_gate(binding,phase,True)
    assert set(gate)=={'status','plan','execution_authorized','remaining_invocations','relaunch_allowed','training_authorized','test_authorized'}
    assert gate['training_authorized']==(phase!='cohort') and gate['remaining_invocations']==1
    policy={key:False for key in owner.DEV_FLAGS}
    if phase!='cohort':policy.update(training_enabled=True,native_training_enabled=True)
    owner.development_validate_flags(policy,phase,True)
    policy['full_epoch_training_allowed']=True
    with pytest.raises(ValueError):owner.development_validate_flags(policy,phase,True)


def test_json_roundtrip_precedes_any_publication():
    assert json.loads(owner.development_json_bytes({'value':(1,True,None)}))=={'value':[1,True,None]}
    with pytest.raises(ValueError):owner.development_json_bytes({'not_finite':float('nan')})
    with pytest.raises(ValueError):owner.development_json_bytes({'unsupported':object()})


@pytest.mark.parametrize('phase,steps',[('cohort',0),('warm',3),('cold',1),('fit',432)])
def test_fixed_operation_budgets(phase,steps):
    limits=owner.development_limits(phase)
    assert limits['optimizer_steps']==steps
    if phase=='fit':
        assert limits['model_forwards']==678 and limits['backward_calls']==432
        assert limits['autograd_calls']==9 and limits['checkpoint_writes']==limits['checkpoint_readbacks']==2
        assert limits['checkpoint_loads']==limits['model_restores']==0


@pytest.mark.parametrize('phase',owner.DEV_PHASES)
def test_payload_allowlist_is_phase_exact(phase):
    plan={'phase':phase,'inputs':{'feature_archive':{'path':'/synthetic/features.npz'},'target_arrays':{'path':'/synthetic/targets.npz'},'selected_cursor2_checkpoint':{'path':'/synthetic/technical/CURSOR002.pt'}}}
    allowed=owner.development_allowed_payloads(plan)
    owner.development_io_audit(allowed,'open',('/synthetic/features.npz','r',0))
    with pytest.raises(RuntimeError):owner.development_io_audit(allowed,'open',('/synthetic/other.npz','r',0))
    if phase=='cold':assert '/synthetic/technical/CURSOR002.pt'in allowed
    else:assert '/synthetic/technical/CURSOR002.pt'not in allowed
    if phase=='fit':
        assert str(owner.DEV_ROOTS[phase]/'FINAL.pt')in allowed
        assert str(owner.DEV_ROOTS[phase]/'INITIAL.pt')not in allowed


@pytest.mark.parametrize('event',['socket.connect','socket.getaddrinfo','urllib.Request'])
def test_network_always_denied(event):
    with pytest.raises(RuntimeError):owner.development_io_audit(set(),event,())
