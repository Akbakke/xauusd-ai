"""Small synthetic tests for certified endpoint runs and one-use receipts."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gx1.audit import verify_market_state_endpoint_persistence_v1 as worker


def _arrays(minutes, m5_bits, *, m15_bits=None):
    issues = np.asarray(minutes, dtype=np.int64) * worker.MINUTE_NS
    allowed = np.full((len(issues), 5), 5, dtype=np.uint8)
    allowed[:, 0] = np.asarray(m5_bits, dtype=np.uint8)
    if m15_bits is not None:
        allowed[:, 1] = np.asarray(m15_bits, dtype=np.uint8)
    sources = np.zeros_like(allowed, dtype=np.int64)
    return issues, allowed, sources


def test_certified_side_pairs_and_known_flat_vs_unknown_censor():
    issues, allowed, sources = _arrays(
        [0, 5, 10, 15, 20, 25, 30, 35, 40], [1, 1, 2, 4, 4, 5, 4, 4, 1])
    got = worker.measure_endpoint_persistence(issues, allowed, sources)
    m5 = got['timeframes']['M5']
    assert got['issues'] == 9 and got['continuous_M1_duration_proven'] is False
    assert m5['certified_side_endpoints'] == 7
    assert m5['flat_endpoints'] == 1 and m5['active_unknown_side_endpoints'] == 1
    assert m5['pointwise_own_lead'] == {
        'starting_certified_side': 7, 'missing_exact_future_issue': 1,
        'future_not_certified_side': 2, 'both_certified': 4,
        'same_side': 3, 'opposite_side': 1, 'same_side_rate': 0.75,
        'paired_with_intermediate_clock_gap': 0,
        'paired_across_validity_month': 0,
    }
    flat = m5['sampled_runs']['by_state']['FLAT']
    assert flat['runs'] == 1 and flat['length_endpoints']['median'] == 1
    assert flat['left_boundary']['known_different'] == 1
    assert flat['right_boundary']['known_different'] == 1
    assert flat['fully_bounded_sampled_span_below_own_lead'] == 1
    up = m5['sampled_runs']['by_state']['UP']
    assert up['runs'] == 2 and up['right_boundary']['unknown_side_or_state'] == 1
    assert up['left_boundary']['unknown_side_or_state'] == 1
    assert got['timeframes']['D1']['sampled_runs']['certified_side_combined']['runs'] == 0


def test_validity_month_and_untyped_clock_gap_split_runs_but_pointwise_pairs_remain():
    from datetime import datetime, timezone
    base = int(datetime(2024, 1, 31, 23, 50, tzinfo=timezone.utc).timestamp() // 60)
    minutes = [base + x for x in (0, 5, 10, 15, 25)]
    issues, allowed, sources = _arrays(minutes, [1] * 5, m15_bits=[1] * 5)
    got = worker.measure_endpoint_persistence(issues, allowed, sources)
    m5 = got['timeframes']['M5']
    runs = m5['sampled_runs']['by_state']['DOWN']
    assert runs['runs'] == 3
    assert runs['right_boundary']['validity_month'] == 1
    assert runs['left_boundary']['validity_month'] == 1
    assert runs['right_boundary']['untyped_clock_gap'] == 1
    assert runs['left_boundary']['untyped_clock_gap'] == 1
    assert m5['pointwise_own_lead']['both_certified'] == 3
    assert m5['pointwise_own_lead']['paired_across_validity_month'] == 1
    m15 = got['timeframes']['M15']['pointwise_own_lead']
    assert m15['both_certified'] == 2 and m15['same_side'] == 2
    assert m15['paired_with_intermediate_clock_gap'] == 1
    assert m15['paired_across_validity_month'] == 0


@pytest.mark.parametrize('bad', ['unaligned', 'unsupported_bit', 'missing_source'])
def test_reject_invalid_source_clock_or_mask(bad):
    issues, allowed, sources = _arrays([0, 5, 10], [1, 2, 4])
    if bad == 'unaligned':
        issues[1] += 1
    elif bad == 'unsupported_bit':
        allowed[0, 0] = 3
    else:
        sources[0, 0] = -1
    with pytest.raises(RuntimeError, match='TARGET_AXES'):
        worker.measure_endpoint_persistence(issues, allowed, sources)


def test_preflight_checks_small_refs_and_plan_without_payload_or_claim(monkeypatch, tmp_path):
    plan_path = tmp_path / 'PLAN.json'
    out_dir = tmp_path / 'RUN'
    monkeypatch.setattr(worker, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(worker, 'OUT_ROOT', out_dir)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(worker, '_clean_head', lambda: head)
    monkeypatch.setattr(worker, '_small_reference_chain', lambda: None)
    monkeypatch.setattr(worker.target, 'load_bound_five_endpoint_targets',
                        lambda: pytest.fail('preflight read target payload'))
    owners = {path: hashlib.sha256((worker.REPO / path).read_bytes()).hexdigest()
              for path in worker.SOURCE_PATHS}
    plan = {'schema_version': worker.SCHEMA, 'stage': worker.STAGE,
            'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(out_dir),
            'max_invocations': 1, 'input_refs': worker.INPUT_REFS,
            'source_owners': owners, 'source_base_head': head,
            'execution': worker.EXECUTION, 'limits': worker.LIMITS,
            'metric_contract': worker.METRIC_CONTRACT,
            'command_template': worker.planned_command('{plan_sha256}'),
            'expected_issue_rows': 118706, 'expected_joined_model_rows': 118249,
            'read_only': True, 'test_data_used': False}
    plan_path.write_text(json.dumps(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    got, ref, observed, policy = worker.preflight(plan_path, digest, out_dir)
    assert got == plan and ref['sha256'] == digest and observed == head
    assert policy is None and not out_dir.exists()
    plan['expected_issue_rows'] += 1
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match='PLAN_SCOPE'):
        worker.preflight(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest(), out_dir)


def _fake_run(monkeypatch, tmp_path, *, error=False):
    out_dir = tmp_path / 'RUN'
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(worker, 'preflight', lambda *args, **kwargs:
                        ({'source_owners': {}}, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(worker, '_source_bookend', lambda *args: None)
    if error:
        monkeypatch.setattr(worker, '_load_full_selected',
                            lambda: (_ for _ in ()).throw(RuntimeError('synthetic blocker')))
    else:
        monkeypatch.setattr(worker, '_load_full_selected',
                            lambda: ({'issues': 3, 'timeframes': {}},
                                     {'full_target_issues': 3, 'joined_model_issues': 2}))
    return out_dir


def test_one_use_complete_receipt_without_model_or_test(monkeypatch, tmp_path):
    out_dir = _fake_run(monkeypatch, tmp_path)
    refs = worker.run(worker.PLAN_PATH, 'a' * 64, out_dir)
    assert not (out_dir / 'FAILED.json').exists()
    result = json.loads((out_dir / 'RESULT.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert result['status'] == 'COMPLETE_DESCRIPTIVE_NO_TARGET_SELECTION'
    assert result['model_fits'] == result['target_builds'] == 0
    assert result['continuous_M1_duration_proven'] is False
    assert terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
    assert refs['result']['sha256'] == hashlib.sha256((out_dir / 'RESULT.json').read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        worker.run(worker.PLAN_PATH, 'a' * 64, out_dir)


def test_postclaim_failure_has_terminal_and_no_relaunch(monkeypatch, tmp_path):
    out_dir = _fake_run(monkeypatch, tmp_path, error=True)
    with pytest.raises(RuntimeError, match='synthetic blocker'):
        worker.run(worker.PLAN_PATH, 'a' * 64, out_dir)
    assert not (out_dir / 'RESULT.json').exists()
    failed = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert failed['error_type'] == 'RuntimeError'
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['no_relaunch_under_this_plan'] is True


def test_policy_gate_requires_named_read_only_one_use_and_closed_training(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, 'OUT_ROOT', tmp_path / 'RUN')
    ref = {'path': 'plan', 'sha256': 'a' * 64}
    gate = {'status': 'ADMITTED', 'decision': 'ADMITTED', 'plan': ref,
            'output_root': str(worker.OUT_ROOT),
            'execution_authorized': True, 'data_reads_authorized': True,
            'model_fits_authorized': 0, 'target_builds_authorized': 0,
            'training_authorized': False, 'test_authorized': False,
            'checkpoint_reads_authorized': False,
            'broker_access_allowed': False, 'test_data_used': False,
            'claimed_invocations': 0, 'completed_invocations': 0,
            'max_invocations': 1, 'relaunch_allowed': False,
            'memory_max': '4G', 'swap_max': '512M', 'wall_seconds': 600}
    policy = {'training_enabled': False, 'native_training_enabled': False,
              'native_launch_authorized': False, 'full_val_allowed': False,
              worker.STAGE: gate}
    worker._policy_gate(policy, ref)
    gate['model_fits_authorized'] = 1
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        worker._policy_gate(policy, ref)
    gate['model_fits_authorized'] = 0
    policy['training_enabled'] = True
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        worker._policy_gate(policy, ref)
