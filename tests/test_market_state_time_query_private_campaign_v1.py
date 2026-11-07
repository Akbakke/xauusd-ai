"""Metadata-only checks for the guarded four-step time-query route."""
from __future__ import annotations

from pathlib import Path

import pytest

from gx1.contracts import market_state_time_query_private_campaign_v1 as scope
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import gx1_capped_execution_v1 as cap
from gx1.scripts import run_entry_native_direction_research_window_v1 as entry
from gx1.scripts import run_market_state_time_query_private_window_v1 as runner


def _ref(name):
    return {'path': '/tmp/gx1_time_query_private_test_' + name,
            'sha256': 'a' * 64}


def test_numeric_recipe_preserves_canonical_controls_and_exact_budget():
    recipe = scope.time_query_private_numerical_recipe()
    assert recipe['schema_version'] == scope.NUMERICAL_SCHEMA
    assert recipe['batch_size'] == 16 and recipe['max_updates'] == 2
    assert recipe['seed'] == 20260911
    assert recipe['precision_policy'] == 'deterministic_fp32'
    assert recipe['optimizer']['name'] == 'torch.optim.AdamW'
    assert recipe['optimizer']['kwargs']['lr'] == 0.0001
    assert recipe['research_objective_sha256'] == scope.MODEL_CONTRACT_SHA256
    assert recipe['time_query_loss_contract_sha256'] == scope.LOSS_CONTRACT_SHA256
    assert scope.PARENT_ROWS == tuple(range(2, 34))


def test_complete_cursor_requires_four_updates_and_full_state():
    recipe = _ref('recipe')
    result = _ref('result')
    full = _ref('full')
    complete = scope.build_time_query_private_cursor(
        recipe_ref=recipe, outcome='COMPLETE', optimizer_updates=4,
        parent_cursor=32, parent_order_sha256='b' * 64,
        full_state=full, result=result)
    assert scope.require_time_query_private_cursor(
        complete, expected_recipe=recipe, verify_files=False)['outcome'] == 'COMPLETE'
    with pytest.raises(Exception, match='CURSOR_COMPLETE'):
        scope.build_time_query_private_cursor(
            recipe_ref=recipe, outcome='COMPLETE', optimizer_updates=3,
            parent_cursor=32, parent_order_sha256='b' * 64,
            full_state=full, result=result)
    with pytest.raises(Exception, match='CURSOR_COMPLETE'):
        scope.build_time_query_private_cursor(
            recipe_ref=recipe, outcome='COMPLETE', optimizer_updates=4,
            parent_cursor=32, parent_order_sha256='b' * 64,
            full_state=None, result=result)


def test_dispatch_reaches_time_query_runner_before_legacy_work(monkeypatch):
    calls = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution',
                        lambda: calls.append('guard') or {'guarded': True})
    monkeypatch.setattr(authority, 'require_native_direction_research_context',
                        lambda *args: (None, None, None, None,
                                      {'run_scope': scope.SCOPE}))
    expected = {'decision': 'private-route'}
    def private_route(**kwargs):
        calls.append(('route', kwargs))
        return expected
    monkeypatch.setattr(runner, 'run_time_query_private_window', private_route)
    actual = entry.run_native_direction_research_window(
        policy_path=Path('/tmp/policy'), policy_file_sha256='a' * 64,
        progress_path=Path('/tmp/progress'))
    assert actual is expected
    assert calls[0] == 'guard'
    assert calls[1][0] == 'route'
    assert len(calls) == 2


@pytest.mark.parametrize('complete', [False, True])
def test_terminal_publication_success_and_failure_without_gpu(
        monkeypatch, tmp_path, complete):
    import json
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common

    output = tmp_path / 'private-research'
    output.mkdir()
    recipe = _ref('recipe')
    window = {'recipe': recipe,
              'campaign_cursor_path': str(tmp_path / 'cursor' / 'pointer.json')}
    plan = {'plan_sha256': 'c' * 64}
    invocation = {'kind': 'synthetic-metadata-only'}
    counters = {'json_bytes': 0,
                'optimizer_updates': 4 if complete else 0}
    state_ref = None
    if complete:
        state_path = tmp_path / 'FULL_STATE.pt'
        state_path.write_bytes(b'synthetic checkpoint ref, no GPU')
        state_ref = common._hash_file(state_path)
    progress_path = tmp_path / 'progress' / 'invocation.json'
    checked = []

    def build_progress(**kwargs):
        return {'outcome': kwargs['outcome'],
                'checkpoint_pointer': kwargs['checkpoint_pointer'],
                'optimizer_updates': kwargs['optimizer_updates'],
                'parent_cursor': kwargs['parent_cursor']}

    def require_progress(value, **kwargs):
        cursor_ref = value['checkpoint_pointer']
        cursor = json.loads(Path(cursor_ref['path']).read_text())
        scope.require_time_query_private_cursor(
            cursor, expected_recipe=recipe, verify_files=True,
            batch_size=16)
        assert kwargs['verify_file'] is True
        checked.append(value['outcome'])
        return value

    monkeypatch.setattr(authority, 'build_research_progress', build_progress)
    monkeypatch.setattr(authority, 'require_research_progress', require_progress)
    runner._publish_claim(output=output, window=window, plan=plan,
                          limits={'synthetic': True}, counters=counters,
                          maximum=100000)
    payload = {
        'schema_version': scope.RESULT_SCHEMA,
        'decision': ('PASS_PRIVATE_MARKET_STATE_TIME_QUERY_RESUME'
                     if complete else 'FAILED'),
        'recipe': recipe,
        'parent_order_sha256': 'b' * 64,
        'parent_cursor': 32 if complete else 0,
        'physical_optimizer_updates': counters['optimizer_updates'],
        'final_state': state_ref,
        'bit_exact_resume': complete,
        'test_data_used': False,
    }
    published = runner._publish_terminal(
        output=output, progress_path=progress_path,
        window=window, plan=plan, invocation=invocation,
        payload=payload, counters=counters, maximum=100000)
    assert Path(published['result']['path']).is_file()
    assert Path(published['checkpoint_pointer']['path']).is_file()
    assert progress_path.is_file()
    assert (output / 'CLAIM.json').is_file()
    assert checked == ['COMPLETE' if complete else 'FAILED']
    with pytest.raises(Exception):
        runner._publish_terminal(
            output=output, progress_path=progress_path,
            window=window, plan=plan, invocation=invocation,
            payload=payload, counters=counters, maximum=100000)
