import json
from pathlib import Path

import pytest
import torch

from gx1.audit import market_state_ta_ema_delta_campaign_inputs_v1 as adapter


def test_successor_refs_use_new_receipts_and_exact_tape(tmp_path, monkeypatch):
    def put(name, content=b'{}'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    plan = put('gather/PLAN.json')
    result = put('gather/RESULT.json')
    coverage = put('gather/COVERAGE.json')
    terminal = put('gather/TERMINAL.json')
    view = put('view/FEATURE_VIEW.json')
    prior = put('new_prior/ENDPOINT_PRIORS.json')
    put('new_prior/RESULT.json')
    put('new_prior/TERMINAL.json')
    target = put('target/FIT_TARGETS.npz')
    put('target/RESULT.json')
    put('target/TERMINAL.json')
    tape = put('tape/INPUTS.npz', b'tape')
    review = put('FIT_REVIEW.json', json.dumps({'input_archive': {
        'path': str(tape), 'sha256': adapter.legacy.ref(tape)['sha256'],
        'bytes': tape.stat().st_size}}).encode())
    monkeypatch.setattr(adapter.gathered, 'input_refs',
                        lambda: {'view': adapter.legacy.ref(view)})
    monkeypatch.setattr(adapter.gathered, 'PLAN', plan)
    monkeypatch.setattr(adapter.gathered, 'ROOT', result.parent)
    monkeypatch.setattr(adapter, 'PRIORS', prior)
    monkeypatch.setattr(adapter, 'TARGETS', target)
    monkeypatch.setattr(adapter, 'FIT_REVIEW', review)
    refs = adapter.successor_input_refs()
    assert refs['gather_plan'] == adapter.legacy.ref(plan)
    assert refs['gather_result'] == adapter.legacy.ref(result)
    assert refs['gather_coverage'] == adapter.legacy.ref(coverage)
    assert refs['gather_terminal'] == adapter.legacy.ref(terminal)
    assert refs['priors'] == adapter.legacy.ref(prior)
    assert refs['fit_tape'] == adapter.legacy.ref(tape)
    assert refs['fit_targets'] == adapter.legacy.ref(target)
    assert len(refs) == len(set(refs)) == 13


def test_prepare_rejects_unbound_refs_before_provider(monkeypatch):
    monkeypatch.setattr(adapter, 'successor_input_refs',
                        lambda: {'view': {'path': '/exact', 'sha256': 'a' * 64}})
    with pytest.raises(RuntimeError, match='TA_EMA_DELTA_CAMPAIGN_INPUT_INPUT_REFS'):
        adapter.prepare(device=torch.device('cpu'), inputs={})


def test_gather_failure_terminal_is_not_accepted(monkeypatch):
    view_ref = {'path': '/view', 'sha256': 'a' * 64}
    plan_ref = {'path': '/plan', 'sha256': 'b' * 64}
    result_ref = {'path': '/result', 'sha256': 'c' * 64}
    coverage_ref = {'path': '/coverage', 'sha256': 'd' * 64}
    terminal_ref = {'path': '/terminal', 'sha256': 'e' * 64}
    inputs = {'view': view_ref, 'gather_plan': plan_ref,
              'gather_result': result_ref, 'gather_coverage': coverage_ref,
              'gather_terminal': terminal_ref}
    plan = {'schema_version': 'gx1_ta_ema_delta_full_gather_plan_v1',
            'stage': adapter.gathered.STAGE,
            'output_root': str(adapter.gathered.ROOT),
            'inputs': {'view': view_ref}, 'scope': adapter.gathered.SCOPE}
    objects = {'/plan': plan, '/result': {'status': 'COMPLETE'},
               '/coverage': {'status': 'COMPLETE'},
               '/terminal': {'status': 'FAILED_CONSUMED'}}
    monkeypatch.setattr(adapter.gathered, 'input_refs', lambda: {'view': view_ref})
    monkeypatch.setattr(adapter.legacy, 'read',
                        lambda binding: objects[binding['path']])
    with pytest.raises(RuntimeError, match='TA_EMA_DELTA_CAMPAIGN_INPUT_GATHER_RECEIPTS'):
        adapter._receipts(inputs)
