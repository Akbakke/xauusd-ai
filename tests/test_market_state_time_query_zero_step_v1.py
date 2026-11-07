"""Focused source/preflight checks; never starts CUDA or writes a run root."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from gx1.audit import verify_market_state_time_query_zero_step_v1 as audit


def _plan() -> tuple[dict, str]:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=audit.REPO, text=True).strip()
    value = {
        'schema_version': audit.SCHEMA, 'stage': audit.STAGE,
        'status': 'PREPARED_NOT_ADMITTED', 'source_base_head': head,
        'output_root': str(audit.OUT_ROOT),
        'fold_ids': audit.FOLD_IDS, 'partition': 'fit',
        'sample_parent_rows': audit.PARENT_ROWS,
        'sample_issue_ns': audit.ISSUE_NS,
        'pass_index': audit.PASS_INDEX, 'seed': audit.SEED,
        'max_invocations': 1, 'max_json_bytes': audit.MAX_JSON_BYTES,
        'model_forwards_authorized': 2, 'backward_calls_authorized': 2,
        'optimizer_steps_authorized': 0, 'training_authorized': False,
        'test_data_used': False, 'execution': audit.EXECUTION,
        'limits': audit.LIMITS,
        'input_refs': {key: audit._fixed_ref(key) for key in audit.FIXED},
        'expected_view_sha256_by_fold': audit.EXPECTED_VIEW_SHA256_BY_FOLD,
        'expected_model_contract_sha256': audit.time_query_model_contract()['contract_sha256'],
        'expected_loss_contract_sha256': audit.time_query_loss_contract()['contract_sha256'],
        'source_owners': {path: audit._sha(audit.REPO / path)
                          for path in audit.SOURCE_PATHS},
    }
    return value, head


def test_preclaim_plan_scope_and_source_hash_fail_closed():
    if audit.OUT_ROOT.exists():
        pytest.skip('One-use output already exists; plan was consumed')
    plan, head = _plan()
    audit._qualify_plan(plan, {'path': str(audit.PLAN_PATH), 'sha256': '0' * 64},
                        None, head=head, out_dir=audit.OUT_ROOT,
                        device='cuda', cap=None)
    bad = dict(plan, pass_index=1)
    with pytest.raises(RuntimeError, match='PLAN_SCOPE'):
        audit._qualify_plan(bad, {}, None, head=head,
                            out_dir=audit.OUT_ROOT, device='cuda', cap=None)
    bad = dict(plan, input_refs={**plan['input_refs'], 'candidate_tape':
                                  {'path': '/tmp/other.npz', 'sha256': '0' * 64}})
    with pytest.raises(RuntimeError, match='PLAN_SCOPE'):
        audit._qualify_plan(bad, {}, None, head=head,
                            out_dir=audit.OUT_ROOT, device='cuda', cap=None)
    bad = dict(plan, source_owners={**plan['source_owners'],
                                     'gx1/contracts/market_state_time_query_model_v1.py': '0' * 64})
    with pytest.raises(RuntimeError, match='SOURCE_OWNERS'):
        audit._qualify_plan(bad, {}, None, head=head,
                            out_dir=audit.OUT_ROOT, device='cuda', cap=None)


def test_complete_fold1_fold2_receipts_and_source_ref_split():
    if not audit.FIXED['fold2_normalization'][0].exists():
        pytest.skip('Local immutable fold-2 receipt is unavailable')
    runner = audit.REPO / audit.RUNNER_SOURCE
    runner_ref = {'path': str(runner), 'sha256': audit._sha(runner)}
    receipts = audit._inspect_receipts(reviewed_runner_ref=runner_ref)
    assert receipts['historical_lineage']['historical_source_head'] == (
        '39b4ac93af9938a3452e457d3d14d5a3c8a5846a')
    assert set(receipts['historical_lineage']['reviewed_runtime_drift']) == (
        set(audit.HISTORICAL_RUNTIME_BRIDGE) | {audit.RUNNER_SOURCE})
    assert receipts['fold1']['cohort']['joined_fit_rows'] == 29574
    assert receipts['fold1']['profile']['contract_sha256'] == audit.fold1.PROFILE_SHA256
    assert receipts['fold1']['lineage']['current_source_owner_count'] == 25
    plan, _ = _plan()
    for fold_id in audit.FOLD_IDS:
        view = audit._bind_view(plan, receipts, fold_id)
        assert view['contract_sha256'] == audit.EXPECTED_VIEW_SHA256_BY_FOLD[str(fold_id)]
        assert view['research_objective']['contract_sha256'] == (
            audit.time_query_model_contract()['contract_sha256'])
        assert len(view['source_bindings']) == 17
        assert ('unobserved_units_profile' in view) == (fold_id == 1)
        assert view['source_bindings']['gc_gather_source']['sha256'] == (
            audit.HISTORICAL_RUNTIME_BRIDGE['gx1/contracts/entry_gc_mtf_history_v1.py'][
                1 if fold_id == 1 else 0])
    assert len(audit.IMMUTABLE_VIEW_REF_KEYS) == 7
    assert len(audit.HISTORICAL_VIEW_REF_KEYS) == 1
    assert len(audit.CURRENT_VIEW_REF_KEYS) == 9
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
    import numpy as np
    bound = load_bound_time_query_targets()
    for fold_id in audit.FOLD_IDS:
        truth = bound.gather_fit_sampled(
            np.asarray(audit.PARENT_ROWS, dtype=np.int64),
            np.asarray(audit.ISSUE_NS, dtype=np.int64),
            fold_id=fold_id, pass_index=audit.PASS_INDEX, seed=audit.SEED)
        assert truth['query_k'][0].tolist() == audit.EXPECTED_QUERY_K
        assert truth['allowed'][0].tolist() == audit.EXPECTED_ALLOWED
        assert truth['eligible'][0].tolist() == [True, True, False, False, False]


def test_gradient_summary_synthetic_only():
    values = torch.tensor([0.0, 2.0, -3.0])
    report = audit._gradient_summary(values)
    assert report == {'present': True, 'finite': True,
                      'nonzero_count': 2, 'max_abs': 3.0}
    assert audit._gradient_summary(None)['present'] is False


def test_central_parser_routes_preflight_without_run(monkeypatch, capsys):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as central
    calls = []

    def preflight(**kwargs):
        calls.append(('preflight', kwargs))
        return {'status': 'SYNTHETIC_PREFLIGHT_NO_CLAIM'}

    def run(**kwargs):
        calls.append(('run', kwargs))
        return {'status': 'SYNTHETIC_RUN_DISPATCH_ONLY'}

    monkeypatch.setattr(audit, 'preflight_read_only', preflight)
    monkeypatch.setattr(audit, 'run', run)
    args = ['--market-state-time-query-zero-step-plan', '/tmp/synthetic-plan.json',
            '--market-state-time-query-zero-step-plan-sha256', '0' * 64,
            '--device', 'cuda', '--out-dir', '/tmp/synthetic-out']
    assert central.main(args + ['--preflight']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'SYNTHETIC_PREFLIGHT_NO_CLAIM'
    assert calls[-1][0] == 'preflight'
    assert central.main(args) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'SYNTHETIC_RUN_DISPATCH_ONLY'
    assert calls[-1][0] == 'run'
    assert calls[-1][1]['device'] == 'cuda'
    assert isinstance(calls[-1][1]['plan_path'], Path)


def _synthetic_run_preflight(monkeypatch, tmp_path):
    from contextlib import nullcontext

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
        return plan, plan_ref, policy_ref, '0' * 40, {}, {}

    monkeypatch.setattr(audit, '_preflight', preflight)
    monkeypatch.setattr(audit, '_bounded_worker', lambda _began: nullcontext())
    monkeypatch.setattr(audit, '_source_bookend', lambda *_args: None)
    return plan_path, plan_ref, policy_ref, out_dir


def _run_file_refs(out_dir):
    return {path.name: audit._ref(path, audit._sha(path))
            for path in out_dir.iterdir()}


def test_run_writes_claim_result_terminal_and_blocks_relaunch(monkeypatch, tmp_path):
    plan_path, plan_ref, policy_ref, out_dir = _synthetic_run_preflight(
        monkeypatch, tmp_path)
    monkeypatch.setattr(audit, '_witness', lambda *_args: {
        'decision': 'SYNTHETIC_DUAL_BACKWARD_NO_UPDATE',
        'fold_ids': audit.FOLD_IDS,
        'fold_results': {'1': {'fold_id': 1}, '2': {'fold_id': 2}},
        'model_forwards': 2, 'backward_calls': 2,
        'optimizer_steps': 0, 'test_data_used': False,
    })

    returned = audit.run(plan_path, plan_ref['sha256'], 'cuda', out_dir)
    refs = _run_file_refs(out_dir)
    assert set(refs) == {'CLAIM.json', 'RESULT.json', 'TERMINAL.json'}
    claim = json.loads((out_dir / 'CLAIM.json').read_text())
    result = json.loads((out_dir / 'RESULT.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert claim['plan'] == result['plan'] == terminal['plan'] == plan_ref
    assert claim['policy'] == result['policy'] == policy_ref
    assert result['claim'] == terminal['claim'] == refs['CLAIM.json']
    assert terminal['result'] == returned['result'] == refs['RESULT.json']
    assert returned['terminal'] == refs['TERMINAL.json']
    assert terminal['status'] == result['status'] == 'COMPLETE'
    assert claim['fold_ids'] == result['fold_ids'] == audit.FOLD_IDS
    assert set(result['fold_results']) == {'1', '2'}
    assert terminal['model_forwards'] == terminal['backward_calls'] == 2
    assert terminal['optimizer_steps'] == 0
    assert terminal['no_relaunch_under_this_plan'] is True
    assert terminal['output_inventory'] == ['CLAIM.json', 'RESULT.json', 'TERMINAL.json']
    with pytest.raises(FileExistsError):
        audit.run(plan_path, plan_ref['sha256'], 'cuda', out_dir)
    assert _run_file_refs(out_dir) == refs


def test_run_writes_claim_failed_terminal_and_blocks_relaunch(monkeypatch, tmp_path):
    plan_path, plan_ref, _, out_dir = _synthetic_run_preflight(
        monkeypatch, tmp_path)

    def fail_after_claim(*_args):
        raise RuntimeError('SYNTHETIC_WITNESS_FAILURE')

    monkeypatch.setattr(audit, '_witness', fail_after_claim)
    with pytest.raises(RuntimeError, match='SYNTHETIC_WITNESS_FAILURE'):
        audit.run(plan_path, plan_ref['sha256'], 'cuda', out_dir)
    refs = _run_file_refs(out_dir)
    assert set(refs) == {'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
    failure = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert failure['plan'] == terminal['plan'] == plan_ref
    assert failure['claim'] == terminal['claim'] == refs['CLAIM.json']
    assert terminal['failure'] == refs['FAILED.json']
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['model_forwards_upper_bound'] == 2
    assert terminal['backward_calls_upper_bound'] == 2
    assert failure['no_relaunch_under_this_plan'] is True
    assert terminal['no_relaunch_under_this_plan'] is True
    assert terminal['output_inventory'] == ['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
    with pytest.raises(FileExistsError):
        audit.run(plan_path, plan_ref['sha256'], 'cuda', out_dir)
    assert _run_file_refs(out_dir) == refs


def test_admitted_policy_status_and_dual_budget_fail_closed():
    if audit.OUT_ROOT.exists():
        pytest.skip('One-use output already exists; plan was consumed')
    plan, head = _plan()
    plan_ref = {'path': str(audit.PLAN_PATH), 'sha256': '0' * 64}
    cap = {
        'memory_max': audit.MEMORY_BYTES,
        'memory_high': audit.MEMORY_BYTES,
        'swap': audit.SWAP_BYTES,
        'pids': 128, 'class': 'producer', 'execution_mode': 'cuda_producer',
        'lock': {'scope': '/synthetic.scope'},
        'cgroup_path': '/sys/fs/cgroup/synthetic.scope',
    }
    gate = {
        'status': 'ADMITTED', 'decision': 'ADMITTED', 'plan': plan_ref,
        'execution_authorized': True, 'data_reads_authorized': True,
        'model_forwards_authorized': 2, 'backward_calls_authorized': 2,
        'optimizer_steps_authorized': 0, 'training_authorized': False,
        'test_authorized': False, 'claimed_invocations': 0,
        'completed_invocations': 0, 'max_invocations': 1,
        'output_root': str(audit.OUT_ROOT), 'relaunch_allowed': False,
        'checkpoint_reads_authorized': False, 'broker_access_allowed': False,
        'test_data_used': False, 'memory_max': '12G', 'swap_max': '512M',
        'wall_seconds': audit.WALL_SECONDS,
    }
    policy = {
        'training_enabled': False, 'native_training_enabled': False,
        'native_launch_authorized': False, 'full_val_allowed': False,
        audit.STAGE: gate,
    }
    audit._qualify_plan(plan, plan_ref, policy, head=head,
                        out_dir=audit.OUT_ROOT, device='cuda', cap=cap)
    for change in ({'status': 'HELD'}, {'model_forwards_authorized': 1}):
        bad = dict(policy, **{audit.STAGE: dict(gate, **change)})
        with pytest.raises(RuntimeError, match='POLICY'):
            audit._qualify_plan(plan, plan_ref, bad, head=head,
                                out_dir=audit.OUT_ROOT, device='cuda', cap=cap)
