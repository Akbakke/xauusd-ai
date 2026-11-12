"""Exact one-use recipe routing and partial-failure accounting; no native work."""
from copy import deepcopy
import pytest
from gx1.contracts import market_state_ta_coverage_benchmark_v1 as scope
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import local_random_access_campaign_v2 as campaign


def ref(path):return dict(path=str(path),sha256='a'*64)


@pytest.mark.parametrize('outcome,visits,updates',[('COMPLETE',768,768),('FAILED',0,0),('FAILED',19,18)])
def test_route_preserves_visit_count_and_strict_completion_budget(outcome,visits,updates):
    recipe=ref(scope.RECIPE_PATH);result=ref(scope.ROOT/'RESULT.json')
    cursor=authority.build_research_cursor(recipe_ref=recipe,outcome=outcome,optimizer_updates=updates,
        parent_cursor=16*visits,parent_order_sha256=None,full_state=None,result=result,batch_size=16)
    assert cursor['schema_version']==scope.CURSOR_SCHEMA
    assert authority.require_research_cursor(cursor,expected_recipe=recipe,verify_files=False,batch_size=16)==cursor
    pointer=ref(scope.RUNTIME_ROOT/'research-cursor/RESUME_CURSOR.json')
    invocation=dict(kind=authority.KIND,invocation_sha256='b'*64,direction_research_recipe=recipe,
        batch_size=16,optimizer_step_budget=768,checkpoint=dict(pointer_path=pointer['path']))
    progress=authority.build_research_progress(window=dict(recipe=recipe),plan=dict(plan_sha256='c'*64),invocation=invocation,
        checkpoint_pointer=pointer,optimizer_updates=updates,parent_cursor=16*visits,parent_order_sha256=None,outcome=outcome)
    assert progress['completed_units']==visits and progress['total_units']==768 and progress['optimizer_updates']==updates
    bad=deepcopy(progress);bad['total_units']=1536
    bad['progress_sha256']=campaign.canonical_sha256({k:v for k,v in bad.items() if k!='progress_sha256'})
    with pytest.raises(RuntimeError):authority.require_research_progress(bad,plan_sha256='c'*64,invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=False)


def test_new_cursor_cannot_be_rebound_to_consumed_control_recipe():
    recipe=ref(scope.RECIPE_PATH)
    cursor=scope.build_cursor(recipe_ref=recipe,outcome='FAILED',optimizer_updates=0,parent_cursor=0,
        parent_order_sha256=None,full_state=None,result=ref(scope.ROOT/'RESULT.json'))
    cursor['recipe']=ref(scope.base.RECIPE_PATH)
    cursor['cursor_sha256']=campaign.canonical_sha256({k:v for k,v in cursor.items() if k!='cursor_sha256'})
    with pytest.raises(RuntimeError):authority.require_research_cursor(cursor,expected_recipe=cursor['recipe'],verify_files=False,batch_size=16)


def test_exact_recipe_dispatch_rejects_lookalike_paths(monkeypatch):
    calls=[]
    monkeypatch.setattr(campaign,'require_binding',lambda value,**kw:value)
    monkeypatch.setattr(campaign,'read_bound_json',lambda *a,**kw:dict(run_scope=scope.SCOPE))
    monkeypatch.setattr(scope,'require_recipe',lambda value,**kw:calls.append(kw) or value)
    assert authority.require_research_launch_recipe(ref(scope.RECIPE_PATH),source_repo=scope.REPO,source_commit=scope.BASE_COMMIT)['run_scope']==scope.SCOPE
    with pytest.raises(RuntimeError):authority.require_research_launch_recipe(ref(scope.ROOT/'OTHER.json'),source_repo=scope.REPO,source_commit=scope.BASE_COMMIT)
    assert len(calls)==1


def test_existing_power_guard_remains250w_and_rejects_test_access():
    recipe=dict(run_scope=scope.SCOPE,fold_id=1,training_enabled=True,test_data_used=False,numerical_recipe=dict(batch_size=16))
    assert authority.guarded_research_power_limit(recipe)==250
    with pytest.raises(RuntimeError):authority.guarded_research_power_limit(dict(recipe,test_data_used=True))


def test_native_coverage_dispatch_keeps_guard_before_exact_worker(monkeypatch):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.audit import run_market_state_ta_m5_objective_benchmark_v1 as worker
    calls=[]
    monkeypatch.setattr(cap,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *a:({},{},{},{},dict(run_scope=scope.SCOPE)))
    monkeypatch.setattr(worker,'run_objective_benchmark',lambda **kw:calls.append('worker') or kw)
    args=dict(policy_path=scope.ROOT/'WINDOW.json',policy_file_sha256='a'*64,progress_path=scope.ROOT/'PROGRESS.json')
    assert runner.run_native_direction_research_window(**args)==dict(args,coverage=True)
    assert calls==['guard','worker']
