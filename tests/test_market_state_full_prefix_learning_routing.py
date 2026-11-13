"""Bounded full-prefix routing preserves real tail rows and closes old plans."""
from copy import deepcopy
import numpy as np
import pytest
from gx1.contracts import market_state_ta_full_prefix_learning_v1 as scope
from gx1.contracts import market_state_ta_full_prefix_order_v1 as order_scope
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import local_random_access_campaign_v2 as campaign


def ref(path):
    return dict(path=str(path), sha256='a' * 64)


@pytest.mark.parametrize('outcome,visits', [('FAILED', 0), ('FAILED', 1848),
    ('FAILED', 1849), ('FAILED', 1850), ('FAILED', 3697), ('COMPLETE', 3698)])
@pytest.mark.parametrize('scope', [scope, order_scope], ids=['chronological', 'transposed'])
def test_cursor_and_progress_preserve_six_row_tail_across_two_passes(outcome, visits, scope):
    recipe = ref(scope.RECIPE_PATH)
    cursor_rows = scope.cursor_from_batch_visits(visits)
    cursor = authority.build_research_cursor(recipe_ref=recipe, outcome=outcome,
        optimizer_updates=visits, parent_cursor=cursor_rows, parent_order_sha256=None,
        full_state=None, result=ref(scope.ROOT / 'RESULT.json'), batch_size=16)
    assert authority.require_research_cursor(cursor, expected_recipe=recipe,
        verify_files=False, batch_size=16) == cursor
    pointer = ref(scope.RUNTIME_ROOT / 'research-cursor/RESUME_CURSOR.json')
    invocation = dict(kind=authority.KIND, invocation_sha256='b' * 64,
        direction_research_recipe=recipe, batch_size=16, optimizer_step_budget=3698,
        checkpoint=dict(pointer_path=pointer['path']))
    progress = authority.build_research_progress(window=dict(recipe=recipe),
        plan=dict(plan_sha256='c' * 64), invocation=invocation, checkpoint_pointer=pointer,
        optimizer_updates=visits, parent_cursor=cursor_rows, parent_order_sha256=None, outcome=outcome)
    assert progress['completed_units'] == visits and progress['total_units'] == 3698
    for field, wrong in [('total_units', 768), ('parent_cursor', cursor_rows + 10)]:
        bad = deepcopy(progress); bad[field] = wrong
        bad['progress_sha256'] = campaign.canonical_sha256({k:v for k,v in bad.items() if k != 'progress_sha256'})
        with pytest.raises(RuntimeError):
            authority.require_research_progress(bad, plan_sha256='c' * 64,
                invocation=invocation, expected_selection_receipt_sha256=None, verify_file=False)


def test_old_coverage_recipe_cannot_claim_new_full_prefix_cursor():
    cursor = scope.build_cursor(recipe_ref=ref(scope.RECIPE_PATH), outcome='FAILED',
        optimizer_updates=0, parent_cursor=0, parent_order_sha256=None, full_state=None,
        result=ref(scope.ROOT / 'RESULT.json'))
    cursor['recipe'] = ref(scope.coverage.RECIPE_PATH)
    cursor['cursor_sha256'] = campaign.canonical_sha256({k:v for k,v in cursor.items() if k != 'cursor_sha256'})
    with pytest.raises(RuntimeError):
        authority.require_research_cursor(cursor, expected_recipe=cursor['recipe'], verify_files=False, batch_size=16)


@pytest.mark.parametrize('scope', [scope, order_scope], ids=['chronological', 'transposed'])
def test_exact_recipe_dispatch_and_existing_power_limit(monkeypatch, scope):
    calls = []
    monkeypatch.setattr(campaign, 'require_binding', lambda value, **kw:value)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *a, **kw:dict(run_scope=scope.SCOPE))
    monkeypatch.setattr(scope, 'require_recipe', lambda value, **kw:calls.append(kw) or value)
    args = dict(source_repo=scope.REPO, source_commit=scope.BASE_COMMIT)
    assert authority.require_research_launch_recipe(ref(scope.RECIPE_PATH), **args)['run_scope'] == scope.SCOPE
    with pytest.raises(RuntimeError):
        authority.require_research_launch_recipe(ref(scope.ROOT / 'OTHER.json'), **args)
    assert len(calls) == 1
    recipe = dict(run_scope=scope.SCOPE, fold_id=1, training_enabled=True,
        test_data_used=False, numerical_recipe=dict(batch_size=16))
    assert authority.guarded_research_power_limit(recipe) == 250
    with pytest.raises(RuntimeError):
        authority.guarded_research_power_limit(dict(recipe, test_data_used=True))


@pytest.mark.parametrize('scope', [scope, order_scope], ids=['chronological', 'transposed'])
def test_native_dispatch_retains_guard_and_uses_only_new_scope(monkeypatch, scope):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.audit import run_market_state_ta_m5_objective_benchmark_v1 as worker
    calls = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda:calls.append('guard') or {})
    monkeypatch.setattr(authority, 'require_native_direction_research_context',
        lambda *a:({}, {}, {}, {}, dict(run_scope=scope.SCOPE)))
    monkeypatch.setattr(worker, 'run_objective_benchmark', lambda **kw:calls.append('worker') or kw)
    args = dict(policy_path=scope.ROOT / 'WINDOW.json', policy_file_sha256='a' * 64,
        progress_path=scope.ROOT / 'PROGRESS.json')
    mode = 'full_prefix_order' if scope is order_scope else 'full_prefix_learning'
    assert runner.run_native_direction_research_window(**args) == dict(args, **{mode: True})
    assert calls == ['guard', 'worker']


def test_tail_joins_actual_six_rows_without_padding_or_mask_change():
    from gx1.audit.run_market_state_ta_m5_objective_benchmark_v1 import _require_coverage_observations
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TICK_NS
    rows = np.arange(29574, dtype=np.int64); issues = rows * TICK_NS
    start = 29568; parents = rows[start:]; clocks = issues[start:]
    query = np.tile(np.asarray(NATIVE_K, np.int64), (6, 1))
    truths = {task:dict(parent_rows=parents.copy(), issue_ns=clocks.copy(),
        allowed=np.full((6, 5), 4, np.uint8), mature=np.ones((6, 5), bool)) for task in ('state', 'future')}
    truths['state']['query_k'] = query.copy()
    truths['future']['validity_ns'] = clocks[:, None] + query * TICK_NS
    report = dict(state_query_k=query.copy(), future_native_k=np.asarray(NATIVE_K, np.int64))
    report.update({task+'_'+k:truths[task][k].copy() for task in truths for k in ('allowed', 'mature')})
    args = dict(report=report, parents=parents, clocks=clocks, rows=rows, issues=issues,
        start=start, truths=truths, np=np)
    _require_coverage_observations(**args)
    with pytest.raises(RuntimeError, match='ORIGINAL_TRAIN_AXES'):
        _require_coverage_observations(**dict(args, parents=np.pad(parents, (0, 10))))
    report['future_mature'][0, 0] = False
    with pytest.raises(RuntimeError, match='ACTUAL_TARGETS'):
        _require_coverage_observations(**args)


def test_completed_old_recipe_cannot_claim_order_cursor():
    cursor = order_scope.build_cursor(recipe_ref=ref(order_scope.RECIPE_PATH), outcome='FAILED',
        optimizer_updates=0, parent_cursor=0, parent_order_sha256=None, full_state=None,
        result=ref(order_scope.ROOT / 'RESULT.json'))
    cursor['recipe'] = ref(scope.RECIPE_PATH)
    cursor['cursor_sha256'] = campaign.canonical_sha256({k:v for k,v in cursor.items() if k != 'cursor_sha256'})
    with pytest.raises(RuntimeError, match='FULL_PREFIX_ORDER_RECIPE_PATH'):
        authority.require_research_cursor(cursor, expected_recipe=cursor['recipe'], verify_files=False, batch_size=16)
