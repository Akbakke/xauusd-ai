"""Exercise closed native-audit admission before any GPU or checkpoint access."""
import hashlib
import json
from pathlib import Path

import pytest

from gx1.audit import dynamic_compact_input_influence_v1 as native
from gx1.contracts import gx1_capped_execution_v1 as capped


def _closed_fixture(tmp_path, monkeypatch, *, consumed=False):
    repo=tmp_path/'repo'
    repo.mkdir()
    monkeypatch.setattr(native,'REPO',repo)
    plan={'schema_version':native.PLAN_SCHEMA,'run_id':'closed_synthetic_input_audit',
          'policy_stage_key':'closed_input_audit',
          'input_scope':{'path':str(tmp_path/'unused_scope.json'),'sha256':'b'*64},
          'output_directory':str(tmp_path/'unused_output'),'one_shot_invocations':1,
          'max_wall_seconds':3500,'max_forward_calls':218,'reverse_calls':13,
          **{key:False for key in native.FLAGS}}
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(plan)+'\n')
    plan_ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    stage={'execution_authorized':bool(consumed),'planned_invocations':1,
           'completed_invocations':1 if consumed else 0,'relaunch_allowed':False,
           'plan':plan_ref,**{key:False for key in native.FLAGS}}
    # The authoritative policy already has historical NaNs. They must not
    # reopen anything or prevent the exact closed-stage check from running.
    policy={'training_enabled':False,'full_epoch_training_allowed':False,
            'full_val_allowed':False,'legacy_receipts':[float('nan')]*4,
            'closed_input_audit':stage}
    (repo/'NEXT_RUN_POLICY.json').write_text(json.dumps(policy)+'\n')
    def forbidden(*args,**kwargs):
        raise AssertionError('closed plan reached GPU, source subprocess or checkpoint path')
    monkeypatch.setattr(capped,'require_guarded_cuda_producer_execution',forbidden)
    monkeypatch.setattr(native.subprocess,'check_output',forbidden)
    return path,plan_ref,plan


@pytest.mark.parametrize('consumed',[False,True])
def test_closed_or_consumed_policy_rejects_before_gpu_and_source_access(tmp_path,monkeypatch,consumed):
    path,reference,plan=_closed_fixture(tmp_path,monkeypatch,consumed=consumed)
    with pytest.raises(RuntimeError,match='DYNAMIC_COMPACT_NATIVE_INPUT_AUDIT_EXACT_POLICY_STAGE'):
        native.run(plan_path=path,plan_sha256=reference['sha256'],device='cuda',
                   out_dir=Path(plan['output_directory']))
    assert not Path(plan['output_directory']).exists()


def test_new_plan_nonfinite_is_rejected_before_policy_or_gpu_access(tmp_path,monkeypatch):
    path,reference,plan=_closed_fixture(tmp_path,monkeypatch)
    plan['max_wall_seconds']=float('nan')
    path.write_text(json.dumps(plan)+'\n')
    with pytest.raises(RuntimeError,match='DYNAMIC_COMPACT_NATIVE_INPUT_AUDIT_NONFINITE_JSON'):
        native.run(plan_path=path,plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   device='cuda',out_dir=Path(plan['output_directory']))
    assert not Path(plan['output_directory']).exists()
