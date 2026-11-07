"""Focused v2 witness source, target, routing and immutable receipt tests."""
from __future__ import annotations

from contextlib import nullcontext
import json
import subprocess
from pathlib import Path

import pytest

from gx1.audit import verify_market_state_time_query_v2_zero_step_v1 as audit
from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as central


def _plan(tmp_path: Path) -> tuple[dict, str]:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=audit.REPO, text=True).strip()
    value = {
        'schema_version': audit.SCHEMA, 'stage': audit.STAGE,
        'status': 'PREPARED_NOT_ADMITTED', 'source_base_head': head,
        'output_root': str(audit.OUT_ROOT), 'fold_id': audit.FOLD_ID,
        'partition': 'fit', 'sample_parent_rows': audit.PARENT_ROWS,
        'sample_issue_ns': audit.ISSUE_NS,
        'pair_axis_sha256': audit.PAIR_AXIS_SHA256,
        'expected_query_k': audit.EXPECTED_QUERY_K,
        'expected_allowed': audit.EXPECTED_ALLOWED,
        'expected_eligible': audit.EXPECTED_ELIGIBLE,
        'expected_mature': audit.EXPECTED_MATURE,
        'pass_index': audit.PASS_INDEX, 'seed': audit.SEED,
        'max_invocations': 1, 'max_json_bytes': audit.MAX_JSON_BYTES,
        'model_forwards_authorized': 1, 'backward_calls_authorized': 1,
        'optimizer_steps_authorized': 0, 'training_authorized': False,
        'test_data_used': False, 'execution': audit.EXECUTION,
        'limits': audit.LIMITS, 'input_refs': audit._fixed_refs(),
        'expected_view_sha256': audit.EXPECTED_VIEW_SHA256,
        'expected_model_contract_sha256': audit.factorized_time_query_model_contract()['contract_sha256'],
        'expected_loss_contract_sha256': audit.factorized_time_query_loss_contract()['contract_sha256'],
        'expected_scorer_contract_sha256': audit.factorized_time_query_scorer_contract()['contract_sha256'],
        'source_owners': {path: audit._sha(audit.REPO / path)
                          for path in audit.SOURCE_PATHS},
    }
    return value, head


def test_plan_scope_source_owners_and_row_pair_fail_closed(tmp_path):
    if audit.OUT_ROOT.exists():
        pytest.skip('One-use output root already exists')
    plan, head = _plan(tmp_path)
    audit._qualify_plan(plan, {}, None, head=head, out_dir=audit.OUT_ROOT,
                        device='cuda', cap=None)
    for changed in (
        dict(plan, sample_parent_rows=[2, 0]),
        dict(plan, expected_allowed=[[1, 5, 1, 7, 7], [1, 1, 7, 7, 7]]),
        dict(plan, expected_scorer_contract_sha256='0' * 64),
    ):
        with pytest.raises(RuntimeError, match='PLAN_SCOPE'):
            audit._qualify_plan(changed, {}, None, head=head,
                                out_dir=audit.OUT_ROOT, device='cuda', cap=None)
    path = 'gx1/contracts/market_state_time_query_batch_v2.py'
    bad = dict(plan, source_owners={**plan['source_owners'], path: '0' * 64})
    with pytest.raises(RuntimeError, match='SOURCE_OWNERS'):
        audit._qualify_plan(bad, {}, None, head=head,
                            out_dir=audit.OUT_ROOT, device='cuda', cap=None)


def test_actual_fold1_receipt_view_and_target_pair_without_model(tmp_path):
    plan, _ = _plan(tmp_path)
    receipts, view, _ = audit._inspect_inputs(plan)
    assert view['contract_sha256'] == audit.EXPECTED_VIEW_SHA256
    assert len(view['source_bindings']) == 17
    assert view['market_state_time_query_v2'] is True
    assert receipts['fold1']['cohort']['joined_fit_rows'] == 29574
    _, truth = audit._target_identity()
    assert truth['allowed'].tolist() == audit.EXPECTED_ALLOWED
    assert truth['query_k'].tolist() == audit.EXPECTED_QUERY_K
    assert truth['eligible'].tolist() == audit.EXPECTED_ELIGIBLE
    assert truth['mature'].tolist() == audit.EXPECTED_MATURE


def test_validator_routes_v2_preflight_and_run_without_v1(monkeypatch, capsys):
    calls = []

    def preflight(**kwargs):
        calls.append(('preflight', kwargs))
        return {'status': 'SYNTHETIC_V2_PREFLIGHT'}

    def run(**kwargs):
        calls.append(('run', kwargs))
        return {'status': 'SYNTHETIC_V2_RUN'}

    monkeypatch.setattr(audit, 'preflight_read_only', preflight)
    monkeypatch.setattr(audit, 'run', run)
    args = ['--market-state-time-query-v2-zero-step-plan', '/tmp/v2-plan.json',
            '--market-state-time-query-v2-zero-step-plan-sha256', '0' * 64,
            '--device', 'cuda', '--out-dir', '/tmp/v2-output']
    assert central.main(args + ['--preflight']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'SYNTHETIC_V2_PREFLIGHT'
    assert calls[-1][0] == 'preflight'
    assert central.main(args) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'SYNTHETIC_V2_RUN'
    assert calls[-1][0] == 'run'
    assert calls[-1][1]['device'] == 'cuda'
    assert isinstance(calls[-1][1]['plan_path'], Path)


def _synthetic_run_preflight(monkeypatch, tmp_path):
    plan_path = tmp_path / 'PLAN.json'
    plan_path.write_text('{"synthetic":true}\n')
    policy_path = tmp_path / 'POLICY.json'
    policy_path.write_text('{"synthetic":true}\n')
    out_dir = tmp_path / 'RUN'
    plan_ref = audit._ref(plan_path, audit._sha(plan_path))
    policy_ref = audit._ref(policy_path, audit._sha(policy_path))
    plan = {'sample_parent_rows': audit.PARENT_ROWS}

    def preflight(given_path, given_sha, device, given_out):
        assert (given_path, given_sha, device, given_out) == (
            plan_path, plan_ref['sha256'], 'cuda', out_dir)
        return plan, plan_ref, policy_ref, '0' * 40, {}, {}, {}, None

    monkeypatch.setattr(audit, '_preflight', preflight)
    monkeypatch.setattr(audit.old, '_bounded_worker', lambda _: nullcontext())
    monkeypatch.setattr(audit, '_source_bookend', lambda *_: None)
    return plan_path, plan_ref, policy_ref, out_dir


def _refs(root):
    return {path.name: audit._ref(path, audit._sha(path)) for path in root.iterdir()}


def test_success_receipts_and_no_replace(monkeypatch, tmp_path):
    plan_path, plan_ref, policy_ref, out_dir = _synthetic_run_preflight(
        monkeypatch, tmp_path)
    monkeypatch.setattr(audit, '_witness', lambda *_args: {
        'decision': 'SYNTHETIC_V2_GRADIENT_NO_UPDATE', 'fold_id': audit.FOLD_ID,
        'model_forwards': 1, 'backward_calls': 1,
        'optimizer_steps': 0, 'test_data_used': False,
    })
    returned = audit.run(plan_path, plan_ref['sha256'], 'cuda', out_dir)
    refs = _refs(out_dir)
    assert set(refs) == {'CLAIM.json', 'RESULT.json', 'TERMINAL.json'}
    claim = json.loads((out_dir / 'CLAIM.json').read_text())
    result = json.loads((out_dir / 'RESULT.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert claim['plan'] == result['plan'] == terminal['plan'] == plan_ref
    assert claim['policy'] == result['policy'] == policy_ref
    assert result['claim'] == terminal['claim'] == refs['CLAIM.json']
    assert returned == {'result': refs['RESULT.json'], 'terminal': refs['TERMINAL.json']}
    assert terminal['result'] == refs['RESULT.json']
    assert terminal['status'] == result['status'] == 'COMPLETE'
    assert terminal['no_relaunch_under_this_plan'] is True
    assert terminal['model_forwards'] == terminal['backward_calls'] == 1
    with pytest.raises(FileExistsError):
        audit.run(plan_path, plan_ref['sha256'], 'cuda', out_dir)
    assert _refs(out_dir) == refs


def test_failure_receipts_and_no_replace(monkeypatch, tmp_path):
    plan_path, plan_ref, _, out_dir = _synthetic_run_preflight(monkeypatch, tmp_path)

    def fail(*_):
        raise RuntimeError('SYNTHETIC_V2_WITNESS_FAILURE')

    monkeypatch.setattr(audit, '_witness', fail)
    with pytest.raises(RuntimeError, match='SYNTHETIC_V2_WITNESS_FAILURE'):
        audit.run(plan_path, plan_ref['sha256'], 'cuda', out_dir)
    refs = _refs(out_dir)
    assert set(refs) == {'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
    failed = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert failed['claim'] == terminal['claim'] == refs['CLAIM.json']
    assert terminal['failure'] == refs['FAILED.json']
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['model_forwards_upper_bound'] == terminal['backward_calls_upper_bound'] == 1
    assert terminal['no_relaunch_under_this_plan'] is True
