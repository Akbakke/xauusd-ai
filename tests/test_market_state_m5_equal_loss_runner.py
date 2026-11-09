"""New native qualification routing without genuine inputs or GPU execution."""
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.scripts import run_market_state_native_F_private_window_v1 as runner
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as full
from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as pages
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective


def test_equal_loss_entry_requires_both_existing_ta_and_dedicated_modes(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, 'run_private_window', lambda **kw: seen.append(kw) or 'sentinel')
    assert runner.run_ta_m5_equal_loss_mechanical(
        policy_path='policy', policy_file_sha256='sha', progress_path='progress') == 'sentinel'
    assert seen == [dict(policy_path='policy', policy_file_sha256='sha',
        progress_path='progress', ta_sf=True, dedicated_m5_side=True, m5_equal_observation=True)]


@pytest.mark.parametrize('reject', [False, True])
def test_full_fit_axis_checked_before_first_gather_and_pages_released(monkeypatch, reject):
    events = []
    # Parent IDs intentionally differ from axis positions.
    parents = np.arange(29574, dtype=np.int64) * 3 + 7
    issues = np.arange(29574, dtype=np.int64) * 300_000_000_000
    provider, plan, prepared = SimpleNamespace(), {}, {}
    monkeypatch.setattr(full, '_prepared_fold1',
        lambda *args, **kw: (provider, plan, prepared, parents, issues))
    def require_axes(p, t):
        events.append('axes')
        assert p is parents and t is issues
        if reject:
            raise ValueError('sentinel bad full axis')
    monkeypatch.setattr(objective, 'require_m5_equal_observation_axes', require_axes)
    def gather(*args, **kw):
        left, right = args[-2:]
        events.append(('gather', left, right))
        return {'synthetic': np.zeros((right-left,1),np.float32)}, parents[left:right], issues[left:right]
    monkeypatch.setattr(full, '_feature_batch', gather)
    monkeypatch.setattr(pages, '_release_pages', lambda p, a: events.append('release'))
    if reject:
        with pytest.raises(ValueError, match='bad full axis'):
            runner._load_ta_train_batches({}, {}, np=np, m5_equal_observation=True)
        assert events == ['axes', 'release']
    else:
        batches, reports, actual_provider, actual_plan, order = runner._load_ta_train_batches(
            {}, {}, np=np, m5_equal_observation=True)
        assert events == ['axes', ('gather',0,16), ('gather',16,32), 'release']
        assert actual_provider is provider and actual_plan is plan
        assert np.array_equal(order, parents) and order is not parents
        for i, (_, rows, timestamps) in enumerate(batches):
            assert np.array_equal(rows, parents[i*16:(i+1)*16])
            assert np.array_equal(timestamps, issues[i*16:(i+1)*16])
            assert reports[i]['parent_rows'] == rows.tolist()


def test_equal_loss_terminal_uses_new_root_and_cursor_and_keeps_failed_outcome(monkeypatch):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    events = []
    monkeypatch.setattr(runner.scope, 'TA_M5_EQUAL_LOSS_ROOT', __import__('pathlib').Path('/synthetic-new-root'))
    monkeypatch.setattr(runner.scope, 'TA_M5_EQUAL_LOSS_RESULT_SCHEMA', 'new_schema')
    def publish(path, value, counters, maximum):
        events.append(('publish',str(path),value))
        return {'path':str(path),'sha256':'a'*64}
    monkeypatch.setattr(common, '_json', publish)
    monkeypatch.setattr(runner.scope, 'build_cursor', lambda **kw: events.append(('cursor',kw)) or {'synthetic':True})
    monkeypatch.setattr(authority, 'build_research_progress',
        lambda **kw: events.append(('progress',kw)) or {'synthetic':True})
    monkeypatch.setattr(authority, 'require_research_progress', lambda *args, **kw: None)
    payload = dict(decision='FAILED', recipe={'path':'new-recipe'}, test_data_used=False,
                   final_state=None, parent_cursor=0, parent_order_sha256='b'*64)
    runner._publish_terminal(output='/unused',progress_path='/progress',
        window={'recipe':payload['recipe'],'campaign_cursor_path':'/cursor'},
        plan={'plan_sha256':'c'*64}, invocation=1,payload=payload,
        counters={'optimizer_updates':0},maximum=4096,ta_sf=True,
        dedicated_m5_side=True,m5_equal_observation=True)
    publications=[event for event in events if event[0]=='publish']
    assert publications[0][1]=='/synthetic-new-root/RESULT.json'
    assert publications[1][1]=='/synthetic-new-root/TERMINAL.json'
    assert publications[1][2]['status']=='FAILED' and publications[1][2]['exit_code']==1
    cursor = next(event[1] for event in events if event[0]=='cursor')
    assert cursor['m5_equal_observation'] is True and cursor['outcome']=='FAILED'
    progress = next(event[1] for event in events if event[0]=='progress')
    assert progress['outcome']=='FAILED'
