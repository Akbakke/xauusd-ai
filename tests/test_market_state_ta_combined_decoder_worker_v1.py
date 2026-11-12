"""Initial-output tolerances and guarded dispatch, using synthetic arrays only."""
from copy import deepcopy

import numpy as np
import pytest

from gx1.audit import run_market_state_ta_m5_objective_benchmark_v1 as worker


def fixture(panel):
    axes = {'parents': np.arange(4096, dtype=np.int64), 'issues': np.arange(4096, dtype=np.int64)}
    for key in (('state_query', 'future_query') if panel == 'TRAIN' else ('query_native',)):
        axes[key] = np.ones((4096, 5), dtype=np.int64)
    for task in ('state', 'future'):
        for key, dtype in (('allowed', np.uint8), ('eligible', np.bool_), ('mature', np.bool_)):
            axes[task + '_' + key] = np.ones((4096, 5), dtype=dtype)
        suffix = '_raw' if panel == 'TRAIN' else '_activity_side_logits'
        axes[task + suffix] = np.full((4096, 5, 2), 3., dtype=np.float32)
        if panel == 'LATER':
            axes[task + '_representation'] = np.zeros((4096, 128), dtype=np.float32)
    measured = (dict(canonical_batch_losses=[1.] * 256, mean_canonical_batch_loss=1.) if panel == 'TRAIN'
                else dict(endpoint_loss={'model': dict(state=1., future=1., combined=2.),
                                         'prior': dict(state=1., future=1., combined=2.)}))
    return dict(arrays=deepcopy(axes), reference=axes, panel=panel, measured=deepcopy(measured),
                control_measurement=measured, np=np)


@pytest.mark.parametrize('panel', ['TRAIN', 'LATER'])
def test_only_cloned_non_m5_logits_use_relative_fp32_tolerance(panel):
    args = fixture(panel)
    suffix = '_raw' if panel == 'TRAIN' else '_activity_side_logits'
    for task in ('state', 'future'):
        args['arrays'][task + suffix][2000, 4, 1] += np.float32(4e-6)
    proof = worker._require_combined_control_arrays(**args)
    assert proof['rows'] == 4096 and proof['axes_targets_exact'] and proof['activity_exact']
    assert all(value > 2e-6 for value in proof['non_m5_side_max_abs'].values())
    assert all(value <= 1 for value in proof['non_m5_side_max_scaled_error'].values())
    assert proof['loss_max_abs'] == 0
    assert proof['m5_representation_arrays_checked'] == (0 if panel == 'TRAIN' else 2)
    assert proof['m5_representations_exact'] is (None if panel == 'TRAIN' else True)


@pytest.mark.parametrize('panel', ['TRAIN', 'LATER'])
@pytest.mark.parametrize('fault', ['axis', 'target', 'dtype', 'shape', 'nan', 'activity', 'm5', 'side', 'loss', 'loss_nan', 'fields'])
def test_initial_parity_rejects_changed_observations_outputs_or_loss(panel, fault):
    args = fixture(panel)
    raw = args['arrays']['future_raw' if panel == 'TRAIN' else 'future_activity_side_logits']
    if fault == 'axis': args['arrays']['parents'][4000] += 1
    elif fault == 'target': args['arrays']['state_allowed'][4000, 4] = 7
    elif fault == 'dtype': args['arrays']['parents'] = args['arrays']['parents'].astype(np.float64)
    elif fault == 'shape': args['arrays']['parents'] = args['arrays']['parents'][:-1]
    elif fault == 'nan': raw[4000, 4, 1] = np.nan
    elif fault == 'activity': raw[4000, 4, 0] = np.nextafter(raw[4000, 4, 0], np.float32(4))
    elif fault == 'm5': raw[4000, 0, 1] = np.nextafter(raw[4000, 0, 1], np.float32(4))
    elif fault == 'side': raw[4000, 4, 1] += np.float32(1e-4)
    elif fault == 'fields': args['arrays']['unbound'] = np.zeros(4096)
    else:
        loss = np.nan if fault == 'loss_nan' else 1.001
        if panel == 'TRAIN': args['measured']['canonical_batch_losses'][250] = loss
        else: args['measured']['endpoint_loss']['model']['state'] = loss
    with pytest.raises(RuntimeError): worker._require_combined_control_arrays(**args)


def test_retained_later_representation_and_prior_must_remain_exact():
    for kind in ('representation', 'prior'):
        args = fixture('LATER')
        if kind == 'representation': args['arrays']['state_representation'][3500, 100] = 1e-7
        else: args['measured']['endpoint_loss']['prior']['state'] += 1e-7
        with pytest.raises(RuntimeError): worker._require_combined_control_arrays(**args)


def test_train_mean_loss_and_missing_batch_are_not_hidden_by_close_outputs():
    for kind in ('mean', 'rows'):
        args = fixture('TRAIN')
        if kind == 'mean': args['measured']['mean_canonical_batch_loss'] += 1e-3
        else: args['measured']['canonical_batch_losses'].pop()
        with pytest.raises(RuntimeError): worker._require_combined_control_arrays(**args)


@pytest.mark.parametrize('coverage,combined', [(True, True), (0, True), (False, 1)])
def test_combined_cannot_reuse_another_worker_mode(coverage, combined):
    with pytest.raises(RuntimeError, match='BENCHMARK_MODE'):
        worker.run_objective_benchmark(policy_path=None, policy_file_sha256=None, progress_path=None,
                                       coverage=coverage, combined_decoder=combined)


def test_native_combined_dispatch_cannot_bypass_guard(monkeypatch):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts import market_state_ta_combined_decoder_benchmark_v1 as scope
    calls = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: calls.append('guard') or {})
    monkeypatch.setattr(authority, 'require_native_direction_research_context',
                        lambda *a: ({}, {}, {}, {}, dict(run_scope=scope.SCOPE)))
    monkeypatch.setattr(worker, 'run_objective_benchmark', lambda **kw: calls.append('worker') or kw)
    args = dict(policy_path=scope.ROOT/'WINDOW.json', policy_file_sha256='a'*64, progress_path=scope.ROOT/'PROGRESS.json')
    assert runner.run_native_direction_research_window(**args) == dict(args, combined_decoder=True)
    assert calls == ['guard', 'worker']
    def stop(): raise RuntimeError('blocked guard')
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', stop)
    with pytest.raises(RuntimeError, match='blocked guard'): runner.run_native_direction_research_window(**args)
    assert calls == ['guard', 'worker']


def test_failure_before_model_keeps_failed_receipt_and_zero_work(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts import market_state_ta_combined_decoder_benchmark_v1 as scope
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    recipe = dict(run_scope=scope.SCOPE, numerical_recipe={}, input_refs={},
                  max_published_json_bytes=1024**2, batch_indices_by_pass={})
    window = dict(recipe={'path':str(scope.RECIPE_PATH),'sha256':'a'*64},
                  output_directory=str(tmp_path/'output'), campaign_cursor_path=str(tmp_path/'CURSOR.json'))
    plan = dict(source_commit='a'*40, plan_sha256='b'*64)
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setattr(authority, 'require_native_direction_research_context', lambda *a: (window,plan,{}, {},recipe))
    monkeypatch.setattr(scope, 'require_recipe', lambda *a,**kw: recipe)
    monkeypatch.setattr(scope, 'read_origin', lambda *a: dict(receipts={},view={},priors={}))
    monkeypatch.setattr(scope, 'ROOT', tmp_path)
    def stop(*args): raise RuntimeError('synthetic bookend failure')
    monkeypatch.setattr(worker.trace, '_bookend', stop)
    published = {}
    def publish(path, payload, *args):
        published[path.name] = deepcopy(payload)
        return dict(path=str(path), sha256='b'*64)
    monkeypatch.setattr(common, '_json', publish)
    monkeypatch.setattr(scope, 'build_cursor', lambda **kw: kw)
    monkeypatch.setattr(authority, 'build_research_progress', lambda **kw: kw)
    monkeypatch.setattr(authority, 'require_research_progress', lambda *a,**kw: None)
    with pytest.raises(RuntimeError, match='synthetic bookend failure'):
        worker.run_objective_benchmark(policy_path=tmp_path/'WINDOW.json', policy_file_sha256='a'*64,
                                       progress_path=tmp_path/'PROGRESS.json', combined_decoder=True)
    result = published['RESULT.json']
    assert result['schema_version'] == scope.RESULT_SCHEMA and result['outcome'] == 'FAILED'
    assert result['initial_model_state_sha256'] is None and result['initial_restore_exact'] is False
    assert result['counts']['optimizer_updates'] == result['counts']['model_constructions'] == 0
    assert result['counts']['control_capture_archive_decodes'] == 0
    assert result['combined_control_parity']['initial'] == {} and published['TERMINAL.json']['exit_code'] == 1


def test_combined_materializer_uses_exact_root_and_existing_cap(monkeypatch):
    from tests import test_market_state_ta_m5_objective_benchmark_v1 as existing
    from gx1.contracts import market_state_ta_combined_decoder_benchmark_v1 as scope
    monkeypatch.setattr(existing, 'scope', scope)
    existing.test_materializer_exact_path_reaches_existing_cap_before_preparation(monkeypatch)


@pytest.mark.parametrize('outcome,visits,updates', [('COMPLETE',768,768),('FAILED',0,0),('FAILED',19,18)])
def test_combined_progress_reuses_existing_failure_accounting(monkeypatch, outcome, visits, updates):
    from tests import test_market_state_coverage_benchmark_routing as existing
    from gx1.contracts import market_state_ta_combined_decoder_benchmark_v1 as scope
    monkeypatch.setattr(existing, 'scope', scope)
    existing.test_route_preserves_visit_count_and_strict_completion_budget(outcome, visits, updates)


def test_combined_recipe_and_power_guards_reject_wrong_paths_and_test(monkeypatch):
    from tests import test_market_state_coverage_benchmark_routing as existing
    from gx1.contracts import market_state_ta_combined_decoder_benchmark_v1 as scope
    monkeypatch.setattr(existing, 'scope', scope)
    existing.test_exact_recipe_dispatch_rejects_lookalike_paths(monkeypatch)
    existing.test_existing_power_guard_remains250w_and_rejects_test_access()
