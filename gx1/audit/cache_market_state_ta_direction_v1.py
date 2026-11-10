"""One admitted frozen FINAL pass caching exact mature fold-1 TA representations."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from gx1.contracts import local_random_access_campaign_v2 as campaign

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_ta_direction_full_cache_20261108_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
SCHEMA = 'gx1_market_state_ta_direction_full_cache_plan_v1'
FINAL_STAGE = 'market_state_ta_m5_equal_loss_final_cache_20261109_001'
FINAL_ROOT = Path('/home/andre2/GX1_RUNS') / FINAL_STAGE.upper()
WORKER = REPO / 'gx1/audit/cache_market_state_ta_direction_v1.py'
DISPATCH = 'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py'
CAP_RUNNER = 'scripts/gx1_capped_run.sh'
HELPER = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_DECODER_DIAGNOSTIC_20261108_001/probe.py')
FIT_ROWS, SCORE_ROWS, BATCH_SIZE, WIDTH = 29574, 17753, 16, 128
ENCODINGS, WALL_SECONDS = 2959, 3600
MAX_CACHE_BYTES = 1024**3


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_DIRECTION_CACHE_' + reason)


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def _read(ref):
    return campaign.read_bound_json(Path(ref['path']), ref['sha256'])


def _ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': campaign.file_sha256(path)}


def _publish(name, value, *, root=ROOT):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    _publish_bytes(root / name, (json.dumps(value, sort_keys=True, indent=2,
                                          allow_nan=False) + '\n').encode())
    return _ref(root / name)


def _helper(ref):
    """The reused CPU helper hides CUDA on import; restore the inherited setting."""
    import torch
    _need(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED_BEFORE_HELPER')
    campaign.require_binding(ref, label='frozen strict-load helper', verify_file=True)
    _need(ref['path'] == str(HELPER), 'HELPER_PATH')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        spec = importlib.util.spec_from_file_location('ta_direction_cache_strict_load', HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _need(not torch.cuda.is_initialized(), 'HELPER_INITIALIZED_CUDA')
    finally:
        if inherited is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = inherited
    _need(os.environ.get('CUDA_VISIBLE_DEVICES') == inherited, 'CUDA_VISIBILITY_RESTORE')
    return module


def _cache_scope(plan_path, out_dir):
    for stage, root, dedicated in ((STAGE, ROOT, False), (FINAL_STAGE, FINAL_ROOT, True)):
        if Path(plan_path) == root / 'PLAN.json' and Path(out_dir) == root:
            return stage, root, dedicated
    _need(False, 'EXACT_CACHE_PATHS')


def _inspect(plan_path, plan_sha256, out_dir, *, execute):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    stage, root, dedicated = _cache_scope(plan_path, out_dir)
    plan_ref = {'path': str(Path(plan_path)), 'sha256': plan_sha256}
    plan = _read(plan_ref)
    keys = {'schema_version', 'stage', 'source_repo', 'source_base_commit',
            'historical_source_commit', 'worker', 'dispatch', 'cap_runner', 'helper',
            'recipe', 'result', 'fit_rows', 'score_rows', 'batch_size', 'decoder_width',
            'include_sampled_S', 'max_backbone_encodings', 'device', 'wall_seconds',
            'training_enabled', 'test_data_used'}
    if dedicated:
        keys.add('fullfit_review')
    _need(set(plan) == keys and plan['schema_version'] == SCHEMA and plan['stage'] == stage
          and plan['source_repo'] == str(REPO) and Path(plan_path) == root / 'PLAN.json'
          and Path(out_dir) == root and plan['fit_rows'] == FIT_ROWS
          and plan['score_rows'] == SCORE_ROWS and plan['batch_size'] == BATCH_SIZE
          and plan['decoder_width'] == WIDTH and plan['include_sampled_S'] is True
          and plan['max_backbone_encodings'] == ENCODINGS and plan['device'] == 'cuda'
          and plan['wall_seconds'] == WALL_SECONDS and plan['training_enabled'] is False
          and plan['test_data_used'] is False, 'EXACT_PLAN')
    for key, path in (('worker', WORKER), ('dispatch', REPO / DISPATCH),
                      ('cap_runner', REPO / CAP_RUNNER), ('helper', HELPER)):
        checked = campaign.require_binding(plan[key], label=key, verify_file=True)
        _need(checked['path'] == str(path), 'SOURCE_PATH:' + key)
    for key in ('source_base_commit', 'historical_source_commit'):
        _need(campaign._COMMIT.fullmatch(plan[key]) is not None, 'COMMIT:' + key)
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor',
                    plan['source_base_commit'], 'HEAD'], check=True)
    recipe, result = _read(plan['recipe']), _read(plan['result'])
    recipe_path = scope.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH if dedicated else scope.TA_RECOVERY_RECIPE_PATH
    run_scope = scope.TA_M5_EQUAL_LOSS_FIT_SCOPE if dedicated else scope.TA_RECOVERY_SCOPE
    runtime = scope.TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT if dedicated else scope.TA_RECOVERY_RUNTIME_ROOT
    _need(plan['recipe']['path'] == str(recipe_path)
          and recipe['recipe_sha256'] == campaign.canonical_sha256({
              k: v for k, v in recipe.items() if k != 'recipe_sha256'})
          and recipe['run_scope'] == run_scope and recipe['fold'] == scope.FOLD
          and recipe['numerical_recipe'] == scope.ta_sf_numerical_recipe(
              dedicated_m5_side=dedicated, m5_equal_observation=dedicated)
          and result['recipe'] == plan['recipe'] and result['source_commit'] == plan['historical_source_commit']
          and plan['result']['path'] == str(runtime / 'prefix-fit/window-0001/RESULT.json')
          and result['outcome'] == result['phase_state'] == 'COMPLETE'
          and result['source_unchanged'] is True and result['test_data_used'] is False
          and result['parent_cursor'] == FIT_ROWS and result['optimizer_updates'] == 1849
          and result['skipped_batches'] == 0, 'HISTORICAL_RECIPE_RESULT')
    if dedicated:
        review = _read(plan['fullfit_review'])
        _need(plan['fullfit_review']['path'] == str(scope.TA_M5_EQUAL_LOSS_FIT_ROOT / 'REVIEW.json')
              and review['status'] == 'COMPLETE_CONSUMED_DIRECTION_DETERIORATED'
              and review['native_source_commit'] == plan['historical_source_commit']
              and review['result'] == plan['result'] and review['final_state'] == result['full_state']
              and review['final_train_measurement'] == result['final_train_measurement']
              and review['later_train_measurement'] == result['oof_measurement']
              and review['matched_train_observations_verified'] is True
              and review['score_means_recalculated'] is True
              and review['source_unchanged'] is True and review['test_data_used'] is False
              and result['dedicated_m5_side'] is True and result['m5_equal_observation'] is True,
              'QUALIFIED_EQUAL_LOSS_FINAL')
    owners = recipe['source_owners']
    _need(set(owners) == scope.required_source_paths(ta_sf=True) and len(owners) == 85,
          'HISTORICAL_85_OWNER_SET')
    current = {}
    historical_git = installed = 0
    replacements = {CAP_RUNNER: plan['cap_runner']}
    if not dedicated:
        replacements[DISPATCH] = plan['dispatch']
    for name, ref in owners.items():
        checked = campaign.require_binding(ref, label='historical owner', verify_file=False)
        _need(checked['path'] == str(REPO / name), 'HISTORICAL_OWNER_PATH')
        if name.startswith('.venv/'):
            campaign.require_binding(ref, label='unchanged installed runtime', verify_file=True)
            installed += 1
        else:
            raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                           plan['historical_source_commit'] + ':' + name])
            _need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'HISTORICAL_GIT:' + name)
            historical_git += 1
        expected = replacements.get(name, ref)
        campaign.require_binding(expected, label='current owner', verify_file=True)
        if name in replacements:
            _need(expected['sha256'] != ref['sha256'], 'DECLARED_ROUTE_CHANGE:' + name)
        current[name] = expected['sha256']
    _need(historical_git == 77 and installed == 8, 'HISTORICAL_OWNER_COUNTS')
    for name, ref in recipe['input_refs'].items():
        campaign.require_binding(ref, label='reused input ' + name, verify_file=True)
    receipts = zero.load_ta_input_receipts(current_source_owners=current)
    view = zero._bind_ta_view(receipts, dedicated_m5_side=dedicated)
    if not dedicated:
        _need(view == _read(recipe['input_refs']['ta_candidate_view']), 'LEGACY_CANDIDATE_VIEW')
    _need(view['contract_sha256'] == recipe['expected_view_sha256'], 'ACTUAL_VIEW')
    scope.require_checkpoint_view(result['feature_view_identity'], recipe=recipe, ta_sf=True)
    baseline = {partition: _read(result[key]) for partition, key in (
        ('fit', 'final_train_measurement'), ('score', 'oof_measurement'))}
    for partition, count in (('fit', FIT_ROWS), ('score', SCORE_ROWS)):
        value = baseline[partition]
        _need(value['evaluated_checkpoint'] == result['full_state']
              and value['optimizer_updates'] == 1849 and value['test_data_used'] is False
              and value['measurement']['rows'] == count
              and value['measurement']['partition'] == partition
              and value['measurement']['feature_view_sha256'] == view['contract_sha256'],
              'BASELINE_PARTITION:' + partition)
    _need(baseline['fit']['evaluated_model_state_sha256']
          == baseline['score']['evaluated_model_state_sha256'], 'BASELINE_SAME_FINAL')
    campaign.require_binding(result['full_state'], label='frozen FINAL', verify_file=True)
    helper = _helper(plan['helper'])
    head, policy_ref = _git('rev-parse', 'HEAD'), None
    if execute:
        _need(_git('branch', '--show-current') == 'work/gx1-current', 'CANONICAL_BRANCH')
        campaign.require_clean_source({'source_repo': str(REPO), 'source_commit': head})
        policy_ref = _ref(REPO / 'NEXT_RUN_POLICY.json')
        policy = _read(policy_ref)
        expected = {'status': 'ADMITTED', 'plan': plan_ref, 'max_invocations': 1,
                    'claimed_invocations': 0, 'completed_invocations': 0, 'relaunch_allowed': False,
                    'execution_authorized': True, 'data_reads_authorized': True,
                    'checkpoint_reads_authorized': True, 'backbone_encodings_authorized': ENCODINGS,
                    'backward_calls_authorized': 0, 'optimizer_steps_authorized': 0,
                    'training_authorized': False}
        gate = policy.get(stage, {})
        _need(all(gate.get(k) == v for k, v in expected.items())
              and policy.get('training_enabled') is False
              and policy.get('full_epoch_training_allowed') is False, 'ADMITTED_POLICY')
        _need(not (root / 'CLAIM.json').exists() and not (root / 'TERMINAL.json').exists(), 'CONSUMED')
    return {'root': root, 'changed_owners': list(replacements),
            'plan': plan, 'plan_ref': plan_ref, 'recipe': recipe, 'result': result,
            'receipts': receipts, 'view': view, 'baseline': baseline, 'helper': helper,
            'head': head, 'policy_ref': policy_ref, 'current_sources': current,
            'historical_git_owners': historical_git, 'installed_runtime_owners': installed}


def preflight_read_only(plan_path, plan_sha256, out_dir):
    if Path(plan_path) == REVERSED_ROOT / 'PLAN.json' or Path(out_dir) == REVERSED_ROOT:
        checked = _inspect_reversed(plan_path, plan_sha256, out_dir, execute=False)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_CHECKPOINT_DESERIALIZATION',
            plan=checked['plan_ref'], view_sha256=checked['view']['contract_sha256'],
            model_state_sha256=REVERSED_MODEL, model_constructions=0, backbone_encodings=0,
            current_source_owners=len(checked['current_sources']), changed_owners=[CAP_RUNNER],
            test_data_used=False)
    checked = _inspect(plan_path, plan_sha256, out_dir, execute=False)
    return {'status': 'READY_NO_CLAIM_NO_MODEL_NO_CHECKPOINT_DESERIALIZATION',
            'plan': checked['plan_ref'], 'view_sha256': checked['view']['contract_sha256'],
            'historical_git_owners': checked['historical_git_owners'],
            'installed_runtime_owners': checked['installed_runtime_owners'],
            'unchanged_current_recipe_owners': len(checked['current_sources']) - len(checked['changed_owners']),
            'declared_changed_owners': checked['changed_owners'],
            'model_constructions': 0, 'backbone_encodings': 0, 'test_data_used': False}


def _side_parity(scores, baseline):
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    report = {}
    for task in ('state', 'future'):
        report[task] = {}
        for tf in TIMEFRAMES:
            got, expected = scores[task][tf]['conditional_side'], baseline['scores']['model'][task][tf]['conditional_side']
            _need(got['count'] == expected['count'] and got['count'] > 0, 'SIDE_COUNT:' + task + ':' + tf)
            metrics = {'count': got['count']}
            for name in ('brier_sum', 'log_sum'):
                actual_mean = got[name] / got['count']
                reference_mean = expected[name] / expected['count']
                delta = abs(actual_mean - reference_mean)
                _need(delta <= 1e-6, 'SIDE_PARITY:' + task + ':' + tf + ':' + name)
                metrics[name.replace('_sum', '')] = {
                    'actual': actual_mean, 'baseline': reference_mean, 'absolute_difference': delta}
            report[task][tf] = metrics
    return report


def _save_cache(partition, arrays, rows, *, partial=False, root=ROOT):
    import numpy as np
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    name = partition.upper() + ('_CACHE_PARTIAL.npz' if partial else '_CACHE.npz')
    stream = io.BytesIO()
    np.savez(stream, **{key: value[:rows] for key, value in arrays.items()})
    payload = stream.getvalue()
    _need(0 < len(payload) <= MAX_CACHE_BYTES, 'CACHE_BYTES')
    _publish_bytes(root / name, payload)
    return {'artifact': _ref(root / name), 'rows': rows, 'partial': partial,
            'arrays': {key: {'shape': [rows, *value.shape[1:]], 'dtype': str(value.dtype)}
                       for key, value in arrays.items()}}


def run(plan_path, plan_sha256, out_dir, *, device):
    if Path(plan_path) == REVERSED_ROOT / 'PLAN.json' or Path(out_dir) == REVERSED_ROOT:
        return _run_reversed(plan_path, plan_sha256, out_dir, device=device)
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_producer_execution, require_capped_lock_ancestry)
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
    from gx1.contracts.market_state_time_query_scorer_v2 import proper_score_sums
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    started = time.monotonic()
    _need(device == 'cuda', 'GPU_ONLY')
    cap = require_guarded_cuda_producer_execution()
    cap['lock'] = require_capped_lock_ancestry()
    _need(cap['cgroup_path'] == '/sys/fs/cgroup' + cap['lock']['scope'], 'LOCK_CGROUP_SCOPE')
    _need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
          and cap['swap'] == 512 * 1024**2, 'EXACT_12G_512M')
    for key, value in (('MAX_POWER_LIMIT_W', '250'), ('MAX_POWER_DRAW_W', '170'),
                       ('MAX_CORE_TEMP_C', '65'), ('MAX_MEMORY_TEMP_C', '80')):
        _need(os.environ.get('GX1_TRAINER_GPU_' + key) == value, 'HARDWARE_GUARD:' + key)
    c = _inspect(plan_path, plan_sha256, out_dir, execute=True)
    recipe, result, receipts, view = (c[k] for k in ('recipe', 'result', 'receipts', 'view'))
    root = c['root']
    def publish(name, value):
        return _publish(name, value, root=root)
    _need(not torch.cuda.is_initialized(), 'CUDA_INIT_ORDER')
    _need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    claim = publish('CLAIM.json', {'plan': c['plan_ref'], 'policy': c['policy_ref'],
        'execution_head': c['head'], 'cap': cap, 'training_enabled': False, 'test_data_used': False})
    provider = prepared = model = None
    arrays, cache_refs, partitions, timings = {}, {}, {}, {}
    active_partition, consumed = None, 0
    counts = {'backbone_encodings': 0, 'model_constructions': 0, 'optimizer_steps': 0,
              'backward_calls': 0, 'resume_state_load_calls': 0, 'resume_state_load_returns': 0}
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('TA direction full cache 3600-second deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1., WALL_SECONDS - 60 - (time.monotonic() - started)))
    def bookend():
        campaign.require_clean_source({'source_repo': str(REPO), 'source_commit': c['head']})
        refs = [c['plan_ref'], c['policy_ref'], *[c['plan'][k] for k in (
            'worker', 'dispatch', 'cap_runner', 'helper', 'recipe', 'result')],
            result['full_state'], result['final_train_measurement'], result['oof_measurement'],
            *recipe['input_refs'].values()]
        if 'fullfit_review' in c['plan']:
            refs.append(c['plan']['fullfit_review'])
        for ref in refs:
            campaign.require_binding(ref, label='cache bookend', verify_file=True)
        for name, digest in c['current_sources'].items():
            _need(campaign.file_sha256(REPO / name) == digest, 'SOURCE_BOOKEND:' + name)
    try:
        tick = time.monotonic()
        numerical = recipe['numerical_recipe']
        trainer._set_deterministic(numerical['seed'], torch.device('cuda'), numerical['precision_policy'])
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _need(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'PRODUCER_NUMERICAL_THREADS')
        provider, provider_plan, prepared, fit_rows, fit_issues = runner._prepared_fold1(receipts, view, np=np)
        endpoint, future, sampled = (load_bound_five_endpoint_targets(),
            load_bound_future_displacement(), load_bound_time_query_targets())
        axes = endpoint.mature_oof_axes(fold_id=1)
        _need(np.array_equal(axes['fit_parent_rows'], fit_rows)
              and np.array_equal(axes['fit_decision_ns'], fit_issues), 'EXACT_FIT_AXIS')
        score_slice = slice(recipe['fold']['oof_start_row'], recipe['fold']['oof_end_row'])
        _need(np.array_equal(axes['score_parent_rows'], prepared['axes']['parent_rows'][score_slice])
              and np.array_equal(axes['score_decision_ns'], prepared['axes']['decision_ns'][score_slice]),
              'EXACT_LATER_TRAIN_AXIS')
        state_model = views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(state_model)
        counts['model_constructions'] = 1
        loaded = c['helper'].load_frozen(result['full_state'], 'final', model,
            recipe=recipe, result=result, metadata={'measurement': c['baseline']['fit']},
            order=torch.from_numpy(fit_rows.copy()), counters=counts)
        model.to(torch.device('cuda'))
        model.eval()
        before = canonical_model_state_sha256(model.state_dict())
        _need(before == loaded['model_sha256'] and not any(p.requires_grad for p in model.parameters()),
              'FROZEN_GPU_MODEL')
        timings['setup_and_load_seconds'] = time.monotonic() - tick
        for partition, rows, issues, expected_rows in (
                ('fit', fit_rows, fit_issues, FIT_ROWS),
                ('score', axes['score_parent_rows'], axes['score_decision_ns'], SCORE_ROWS)):
            tick = time.monotonic()
            active_partition, consumed, arrays = partition, 0, {}
            _need(len(rows) == len(issues) == expected_rows, 'PARTITION_ROWS')
            scores = {}
            def put(key, value, left, right):
                value = np.asarray(value)
                _need(value.shape[0] == right - left and value.dtype.kind in 'biuf',
                      'CACHE_ARRAY:' + key)
                if value.dtype.kind == 'f':
                    _need(np.isfinite(value).all(), 'NONFINITE_CACHE:' + key)
                if key not in arrays:
                    arrays[key] = np.empty((expected_rows, *value.shape[1:]), dtype=value.dtype)
                arrays[key][left:right] = value
            for batch_index, left in enumerate(range(0, expected_rows, BATCH_SIZE)):
                right = min(left + BATCH_SIZE, expected_rows)
                raw, parents, clocks = runner._feature_batch(prepared, view, receipts,
                    rows, issues, left, right, np=np)
                features = {key: torch.from_numpy(value).to('cuda') for key, value in raw.items()}
                native_np = np.broadcast_to(np.asarray(NATIVE_K, np.int64), (len(parents), 5)).copy()
                native = torch.from_numpy(native_np).to('cuda')
                captures = {}
                hooks = [model.state_model.query_head.register_forward_pre_hook(
                    lambda _, args: captures.update(state=args[0].detach())),
                    model.future_head.register_forward_pre_hook(
                    lambda _, args: captures.update(future=args[0].detach()))]
                try:
                    with torch.inference_mode():
                        hidden = model.encode_features(**features)
                        counts['backbone_encodings'] += 1
                        sr = model.state_model.decode_raw_query_grid(hidden, native[:, None])[:, 0]
                        fr = model.decode_native_F_raw(hidden)
                        representations = [captures['state'][:, 0].cpu().numpy().copy(),
                                           captures['future'].cpu().numpy().copy()]
                        logits = [sr.cpu().numpy().copy(), fr.cpu().numpy().copy()]
                        if partition == 'fit':
                            qn = sampled_query_k(clocks, pass_index=0, seed=numerical['seed'])
                            ss = model.state_model.decode_raw_query_grid(
                                hidden, torch.from_numpy(qn).to('cuda')[:, None])[:, 0]
                            train_rep = captures['state'][:, 0].cpu().numpy().copy()
                            train_raw = ss.cpu().numpy().copy()
                finally:
                    for hook in hooks: hook.remove()
                truths = [endpoint.gather(parents, clocks, fold_id=1, partition=partition),
                          future.gather(parents, clocks, fold_id=1, partition=partition)]
                put('parents', parents, left, right); put('issues', clocks, left, right)
                put('query_native', native_np, left, right)
                for index, task in enumerate(('state', 'future')):
                    _need(representations[index].shape == (len(parents), 5, WIDTH), 'REPRESENTATION_WIDTH')
                    truth = truths[index]
                    _need(np.array_equal(truth['parent_rows'], parents)
                          and np.array_equal(truth['issue_ns'], clocks), 'NATIVE_TARGET_AXIS')
                    for field, value in [('representation', representations[index]),
                            ('activity', logits[index][..., 0]), ('original_raw', logits[index]),
                            *[(field, truth[field]) for field in ('allowed', 'eligible', 'mature')]]:
                        put(f'native_{index}_{field}', value, left, right)
                    batch_scores = proper_score_sums(logits[index], truth['allowed'],
                        truth['eligible'], truth['mature'], native_np)['per_tf']
                    if task not in scores:
                        scores[task] = batch_scores
                    else:
                        for tf in TIMEFRAMES:
                            for metric, stats in batch_scores[tf].items():
                                for key, value in stats.items():
                                    scores[task][tf][metric][key] += value
                if partition == 'fit':
                    truth = sampled.gather_fit_sampled(parents, clocks, fold_id=1,
                                                       pass_index=0, seed=numerical['seed'])
                    _need(np.array_equal(truth['query_k'], qn)
                          and np.array_equal(truth['parent_rows'], parents)
                          and np.array_equal(truth['issue_ns'], clocks)
                          and train_rep.shape == (len(parents), 5, WIDTH)
                          and np.array_equal(train_raw[:, 0], logits[0][:, 0]), 'SAMPLED_S_PARITY')
                    put('query_train', qn, left, right)
                    for field, value in [('representation', train_rep), ('activity', train_raw[..., 0]),
                            ('original_raw', train_raw),
                            *[(field, truth[field]) for field in ('allowed', 'eligible', 'mature')]]:
                        put('train_0_' + field, value, left, right)
                consumed = right
                if (batch_index + 1) % 256 == 0:
                    print(json.dumps({'status': 'PROGRESS', 'partition': partition,
                        'completed_rows': consumed, 'completed_batches': batch_index + 1,
                        'backbone_encodings': counts['backbone_encodings'],
                        'elapsed_seconds': time.monotonic() - started}), flush=True)
                del raw, features, hidden, sr, fr
            cache_refs[partition] = _save_cache(partition, arrays, consumed, root=root)
            # Preserve the produced bytes even when their qualification fails.
            parity = _side_parity(scores, c['baseline'][partition]['measurement'])
            partitions[partition] = {'rows': consumed, 'batches': (consumed + 15) // 16,
                'conditional_side_parity': parity, 'scores': scores}
            timings[partition + '_seconds'] = time.monotonic() - tick
            arrays = {}
        _need(counts['backbone_encodings'] == ENCODINGS
              and canonical_model_state_sha256(model.state_dict()) == before
              and all(p.grad is None for p in model.parameters()), 'COUNTS_AND_MODEL_UNCHANGED')
        base._release_pages(provider, prepared); prepared = None
        provider.validate_metadata(provider_plan)
        bookend()
        result_ref = publish('RESULT.json', {'status': 'COMPLETE', 'plan': c['plan_ref'],
            'claim': claim, 'policy': c['policy_ref'], 'execution_head': c['head'],
            'historical_source_commit': c['plan']['historical_source_commit'],
            'checkpoint': loaded, 'view_sha256': view['contract_sha256'],
            'cache_partitions': cache_refs, 'partitions': partitions, 'counts': counts,
            'timings': {**timings, 'elapsed_seconds': time.monotonic() - started},
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'model_state_unchanged': True, 'source_unchanged': True,
            'sampled_S_pass_index': 0, 'sampled_F_reuses_native_1': True,
            'test_data_used': False, 'predictive_value_claimed': False, 'promotion_authorized': False})
        terminal = publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': c['plan_ref'], 'result': result_ref, 'cache_partitions': cache_refs,
            'source_unchanged': True, 'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as exc:
        preservation_error = None
        if arrays and consumed and active_partition not in cache_refs:
            try:
                cache_refs[active_partition] = _save_cache(active_partition, arrays, consumed, partial=True, root=root)
            except Exception as secondary:
                preservation_error = type(secondary).__name__ + ':' + str(secondary)
        failure = publish('FAILED.json', {'status': 'FAILED_CONSUMED', 'plan': c['plan_ref'],
            'claim': claim, 'error': type(exc).__name__ + ':' + str(exc),
            'cache_partitions': cache_refs, 'preservation_error': preservation_error,
            'counts': counts, 'elapsed_seconds': time.monotonic() - started, 'test_data_used': False})
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': c['plan_ref'], 'failure': failure, 'cache_partitions': cache_refs,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_alarm)
        if prepared is not None:
            base._release_pages(provider, prepared)

# Frozen REVERSED weights: a new, exact TRAIN-only producer; old scopes stay sealed.
REVERSED_STAGE = 'market_state_ta_m5_reversed_repr_cache_20261110_001'
REVERSED_ROOT = Path('/home/andre2/GX1_RUNS') / REVERSED_STAGE.upper()
REVERSED_SCHEMA = 'gx1_market_state_ta_m5_reversed_repr_cache_plan_v1'
REVERSED_SOURCE = '2f493a251479484b5876d9fa7b0d0a8ffbd069f4'
REVERSED_MODEL = '174b1a50edd78e102eec11ba80b324e855b6299b7c8789f2cb69df23b878a6f5'
_REVERSED_ORIGIN = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULL_REVERSE_ORDER_20261110_002')
REVERSED_REFS = {
    'reverse_recipe': {'path': str(_REVERSED_ORIGIN / 'RECIPE.json'), 'sha256': '9f8cb54f927e0a52541be93cdbd2c821e9d461c6aff9c27f2def0d11daed14df'},
    'reverse_result': {'path': str(_REVERSED_ORIGIN / 'RESULT.json'), 'sha256': 'be7a403c827e5a2883d7ae6c29ccd345c2ce4feb599da402ec41e601e001347d'},
    'reverse_terminal': {'path': str(_REVERSED_ORIGIN / 'TERMINAL.json'), 'sha256': '40a284e6ee0648c46165e85e28e56494e9378b3e9f0e6457ad93da67d1dcf2fe'},
    'reverse_review': {'path': str(REPO / 'docs/TA_M5_FULL_REVERSE_ORDER_002_REVIEW.json'), 'sha256': '07a82392f2291979b98887f2b8cf73ff5935f31389ff00385e7bf32870fd82eb'},
    'reversed_evidence': {'path': str(_REVERSED_ORIGIN / 'REVERSED_EVIDENCE.pt'), 'sha256': 'fc59501b6a2a016dc642bacf92a6a0ad1a6a10068f2831855fa8c65bec06ceb1'},
    'train_measurement': {'path': str(_REVERSED_ORIGIN / 'REVERSED_TRAIN.json'), 'sha256': '8a53e7af2682190ec42eeab437af6f8ae0df91ab491384dfa57ec459b942fea0'},
    'train_capture': {'path': str(_REVERSED_ORIGIN / 'REVERSED_TRAIN_READOUT.npz'), 'sha256': '31a9ac06ecf10acb77545a2f5665295b8fe09d33e559d36434302f6e2815257b'},
}
REVERSED_BOUNDS = dict(fit_rows=29574, score_rows=0, batch_size=16, decoder_width=128,
    max_backbone_encodings=1849, max_model_constructions=1, max_checkpoint_loads=1,
    max_optimizer_constructions=0, max_backwards=0, max_optimizer_updates=0,
    max_checkpoint_writes=0, max_cache_bytes=67108864, device='cuda', wall_seconds=3600,
    training_enabled=False, test_data_used=False)


def _reversed_plan(plan, plan_path, out_dir):
    fields = {'schema_version', 'stage', 'source_repo', 'source_base_commit', 'worker',
              'dispatch', 'cap_runner', 'runtime_source_override', 'inputs'} | set(REVERSED_BOUNDS)
    _need(set(plan) == fields and plan['schema_version'] == REVERSED_SCHEMA
          and plan['stage'] == REVERSED_STAGE and plan['source_repo'] == str(REPO)
          and Path(plan_path) == REVERSED_ROOT / 'PLAN.json' and Path(out_dir) == REVERSED_ROOT
          and plan['inputs'] == REVERSED_REFS
          and all(type(plan[k]) is type(v) and plan[k] == v for k, v in REVERSED_BOUNDS.items()),
          'REVERSED_EXACT_PLAN')
    _need(isinstance(plan['source_base_commit'], str)
          and campaign._COMMIT.fullmatch(plan['source_base_commit']), 'REVERSED_SOURCE_BASE')
    return plan


def _reversed_sources(plan, recipe):
    """Only the exact new cap selector differs from the consumed 85-owner recipe."""
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    owners = recipe['source_owners']
    _need(set(owners) == fit.required_source_paths(ta_sf=True) and len(owners) == 85,
          'REVERSED_OWNER_SET')
    override = plan['runtime_source_override']
    _need(override == {'original_ref': owners[CAP_RUNNER], 'current_ref': plan['cap_runner']}
          and override['current_ref']['path'] == str(REPO / CAP_RUNNER)
          and override['original_ref']['sha256'] != override['current_ref']['sha256'],
          'REVERSED_CAP_ONLY_BRIDGE')
    current = {}
    for name, ref in owners.items():
        _need(ref['path'] == str(REPO / name), 'REVERSED_OWNER_PATH')
        if not name.startswith('.venv/'):
            raw = subprocess.check_output(['git', '-C', str(REPO), 'show', REVERSED_SOURCE + ':' + name])
            _need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'REVERSED_HISTORICAL_OWNER:' + name)
        live = plan['cap_runner'] if name == CAP_RUNNER else ref
        campaign.require_binding(live, label='reversed current owner ' + name, verify_file=True)
        current[name] = live['sha256']
    return current


def _inspect_reversed(plan_path, plan_sha256, out_dir, *, execute):
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts import market_state_native_F_private_campaign_v1 as reverse
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    plan = _reversed_plan(_read(plan_ref), plan_path, out_dir)
    bound_refs = [plan_ref]
    for key, path in (('worker', WORKER), ('dispatch', REPO / DISPATCH), ('cap_runner', REPO / CAP_RUNNER)):
        _need(plan[key]['path'] == str(path), 'REVERSED_SOURCE_PATH:' + key)
        campaign.require_binding(plan[key], label=key, verify_file=True)
        bound_refs.append(plan[key])
    for key, ref in plan['inputs'].items():
        campaign.require_binding(ref, label='reversed input ' + key, verify_file=True)
        bound_refs.append(ref)
    recipe, result, terminal, measure = (_read(plan['inputs'][k]) for k in (
        'reverse_recipe', 'reverse_result', 'reverse_terminal', 'train_measurement'))
    _read(plan['inputs']['reverse_review'])
    _need(recipe['recipe_sha256'] == campaign.canonical_sha256({k: v for k, v in recipe.items() if k != 'recipe_sha256'})
          and recipe['run_scope'] == reverse.TA_M5_FULL_REVERSE_ORDER_SCOPE
          and recipe['admission_stage'] == 'market_state_ta_m5_full_reverse_order_20261110_002'
          and recipe['batch_indices'] == list(range(1847, -1, -1)) + [1848]
          and recipe['numerical_recipe']['max_updates'] == 1849, 'REVERSED_RECIPE_IDENTITY')
    _need(result['recipe'] == plan['inputs']['reverse_recipe']
          and result['outcome'] == 'COMPLETE' and result['source_commit'] == REVERSED_SOURCE
          and result['decision'] == 'PASS_TA_M5_FULL_REVERSE_ORDER_TECHNICAL'
          and result['source_unchanged'] is True and result['test_data_used'] is False
          and result['evaluation_state_unchanged'] is True and result['all_panels_complete'] is True
          and result['optimizer_updates'] == result['physical_optimizer_updates'] == 1849
          and result['parent_cursor'] == FIT_ROWS and result['skipped_batches'] == 0
          and result['candidate_evidence'] == plan['inputs']['reversed_evidence']
          and result['measurement_refs']['REVERSED_TRAIN'] == plan['inputs']['train_measurement']
          and result['captures']['REVERSED_TRAIN'] == plan['inputs']['train_capture']
          and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
          and terminal['recipe'] == plan['inputs']['reverse_recipe'] and terminal['result'] == plan['inputs']['reverse_result']
          and terminal['no_relaunch_under_this_plan'] is True and terminal['test_data_used'] is False,
          'REVERSED_CONSUMED_ORIGIN')
    _need(measure['evaluated_model_state_sha256'] == REVERSED_MODEL
          and measure['capture'] == plan['inputs']['train_capture'] and measure['optimizer_updates'] == 1849
          and measure['test_data_used'] is False and measure['measurement']['rows'] == FIT_ROWS
          and measure['measurement']['batches'] == 1849, 'REVERSED_MEASUREMENT')
    current = _reversed_sources(plan, recipe)
    # Reuse the native preflight's input/view/prior seams. Its old cap-byte check
    # is replaced only here; the consumed recipe and its validators stay unchanged.
    origin = reverse.read_ta_m5_full_reverse_order_origin(recipe)
    for ref in recipe['input_refs'].values():
        campaign.require_binding(ref, label='reversed reused input', verify_file=True)
        bound_refs.append(ref)
    campaign.require_binding(recipe['trace_worker'], label='reversed capsule producer', verify_file=True)
    bound_refs.append(recipe['trace_worker'])
    receipts = zero.load_ta_input_receipts(current_source_owners=current)
    view = zero._bind_ta_view(receipts, dedicated_m5_side=True)
    _need(view == _read(origin['original_recipe']['input_refs']['ta_candidate_view'])
          and view['contract_sha256'] == recipe['expected_view_sha256']
          and recipe['numerical_recipe'] == origin['original_recipe']['numerical_recipe'],
          'REVERSED_ORIGINAL_VIEW_NUMERICAL')
    fit.require_checkpoint_view(origin['original_result']['feature_view_identity'],
        recipe=origin['original_recipe'], verify_files=True, ta_sf=True)
    priors = _read(recipe['input_refs']['endpoint_priors'])
    objective.require_endpoint_priors(priors, fold_id=1, ta_only=True, dedicated_m5_side=True)
    head = _git('rev-parse', 'HEAD')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor',
                    plan['source_base_commit'], head], check=True)
    policy_ref = _ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = _read(policy_ref); bound_refs.append(policy_ref)
    if execute:
        expected = dict(status='ADMITTED', plan=plan_ref, max_invocations=1, claimed_invocations=0,
            completed_invocations=0, relaunch_allowed=False, execution_authorized=True,
            data_reads_authorized=True, checkpoint_reads_authorized=True,
            backbone_encodings_authorized=1849, backward_calls_authorized=0,
            optimizer_steps_authorized=0, training_authorized=False)
        gate = policy.get(REVERSED_STAGE, {})
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k, v in expected.items())
              and all(policy.get(k) is False for k in (
                  'training_enabled', 'native_training_enabled', 'native_launch_authorized',
                  'full_epoch_training_allowed')),
              'REVERSED_ADMITTED_POLICY')
        _need(_git('branch', '--show-current') == 'work/gx1-current', 'REVERSED_BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO), source_commit=head))
        _need(not any((REVERSED_ROOT / name).exists() for name in ('CLAIM.json', 'RESULT.json', 'TERMINAL.json')),
              'REVERSED_CONSUMED')
    return dict(plan=plan, plan_ref=plan_ref, recipe=recipe, result=result, measurement=measure,
        origin=origin, receipts=receipts, view=view, priors=priors, head=head, policy_ref=policy_ref,
        bound_refs=bound_refs, current_sources=current)


def _load_reversed_weights(model, ck, checked, order, *, torch):
    """Validate the private evidence, copy weights/all buffers; no optimizer/RNG restore."""
    import copy
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    fields = set(('schema_version arm recipe source_commit original_checkpoint canonical_resume_authority '
        'model_state buffer_state optimizer_state rng_state module_training parent_order parent_cursor '
        'physical_optimizer_updates pending_gradients parameter_binding original_numerical_recipe_reference '
        'method_state model_state_sha256 buffer_state_sha256 optimizer_state_sha256 rng_state_sha256 '
        'test_data_used evidence_sha256').split())
    recipe, refs = checked['recipe'], checked['plan']['inputs']
    _need(isinstance(ck, dict) and set(ck) == fields
          and ck['schema_version'] == 'gx1_ta_m5_full_reverse_order_evidence_v1'
          and ck['arm'] == 'REVERSED' and ck['recipe'] == refs['reverse_recipe']
          and ck['source_commit'] == REVERSED_SOURCE
          and ck['original_checkpoint'] == recipe['input_refs']['original_initial_checkpoint']
          and ck['canonical_resume_authority'] is False and ck['test_data_used'] is False
          and ck['physical_optimizer_updates'] == 1849 and ck['parent_cursor'] == FIT_ROWS
          and ck['pending_gradients'] == 0 and ck['model_state_sha256'] == REVERSED_MODEL,
          'REVERSED_EVIDENCE_HEADER')
    _need(ck['evidence_sha256'] == old._tree_sha({k: v for k, v in ck.items() if k != 'evidence_sha256'}),
          'REVERSED_EVIDENCE_SEAL')
    numerical = state._numerical(recipe['numerical_recipe'], model)
    order_sha = old._tree_sha(order)
    _need(ck['original_numerical_recipe_reference'] == numerical
          and ck['parent_order'].dtype == torch.int64 and torch.equal(ck['parent_order'], order)
          and order_sha == checked['result']['parent_order_sha256']
          and ck['method_state'] == dict(batch_indices=recipe['batch_indices'],
              batch_order_sha256=recipe['batch_order_sha256'], full_order_sha256=order_sha,
              original_fit_batch_starts=[16 * b for b in recipe['batch_indices']], pass_index=0,
              all_original_rows_once=True, tail_policy='original_tail_last', loss_and_optimizer_unchanged=True),
          'REVERSED_EVIDENCE_ORDER_NUMERICAL')
    pairs, active, inactive = state._parameters(model)
    binding = dict(schema_version=state.TA_BINDING_SCHEMA, numerical_recipe=numerical,
        feature_view_identity=model.state_model.feature_view_identity, research_objective=model.research_objective,
        active_parameter_names=[n for n, _ in active], inactive_parameter_names=[n for n, _ in inactive],
        parameter_inventory=[dict(name=n, shape=list(v.shape), dtype=str(v.dtype)) for n, v in pairs],
        optimizer_group_parameter_names=[[n for n, _ in active]], ema_created=False,
        scheduler_created=False, target_model_created=False)
    binding['contract_sha256'] = old._tree_sha(binding)
    _need(ck['parameter_binding'] == binding, 'REVERSED_PARAMETER_BINDING')
    old._tensor_map(ck['model_state'], model.state_dict(), 'REVERSED_MODEL')
    old._tensor_map(ck['buffer_state'], dict(model.named_buffers()), 'REVERSED_BUFFERS')
    _need(model_sha(ck['model_state']) == REVERSED_MODEL, 'REVERSED_MODEL_DIGEST')
    for key in ('buffer_state', 'optimizer_state', 'rng_state'):
        _need(old._tree_sha(ck[key]) == ck[key + '_sha256'], 'REVERSED_DIGEST:' + key)
    immutable = {'native_F_objective_sha256'} | {'state_model.' + n for n in old.IMMUTABLE_BUFFER_NAMES | state._QUERY_BUFFERS}
    prefixes = tuple('state_model.' + n for n in old.IMMUTABLE_BUFFER_PREFIXES)
    for name, value in model.named_buffers():
        saved = ck['buffer_state'][name]
        if name in immutable or name.startswith(prefixes):
            _need(torch.equal(saved, value), 'REVERSED_IMMUTABLE_BUFFER:' + name)
        if name in ck['model_state']:
            _need(torch.equal(saved, ck['model_state'][name]), 'REVERSED_PERSISTENT_BUFFER:' + name)
    _need(set(ck['module_training']) == set(dict(model.named_modules()))
          and all(type(v) is bool for v in ck['module_training'].values()), 'REVERSED_MODULE_MODES')
    opt = ck['optimizer_state']; kw = copy.deepcopy(numerical['optimizer']['kwargs']); kw['betas'] = tuple(kw['betas'])
    _need(set(opt) == {'state', 'param_groups'} and len(opt['param_groups']) == 1, 'REVERSED_OPTIMIZER_METADATA')
    group = opt['param_groups'][0]
    _need(set(group) == set(kw) | {'params'} and group['params'] == list(range(len(active)))
          and old._tree_sha({k: group[k] for k in kw}) == old._tree_sha(kw)
          and bool(opt['state']) and set(opt['state']) <= set(group['params']), 'REVERSED_OPTIMIZER_ORDER')
    for index, item in opt['state'].items():
        _need(type(index) is int and set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'REVERSED_ADAM_SCHEMA')
        step = item['step']
        _need(step.ndim == 0 and step.dtype in (torch.float32, torch.float64) and bool(torch.isfinite(step))
              and float(step).is_integer() and 0 < float(step) <= 1849, 'REVERSED_ADAM_STEP')
        for key in ('exp_avg', 'exp_avg_sq'):
            value, parameter = item[key], active[index][1]
            _need(value.shape == parameter.shape and value.dtype == parameter.dtype
                  and bool(torch.isfinite(value).all()), 'REVERSED_ADAM_MOMENT')
        _need(bool((item['exp_avg_sq'] >= 0).all()), 'REVERSED_ADAM_SECOND_MOMENT')
    cuda_rng = ck['rng_state']['torch_cuda']
    _need(isinstance(cuda_rng, list) and bool(cuda_rng) and all(
        isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.ndim == 1 and 0 < v.numel() <= 65536
        for v in cuda_rng), 'REVERSED_STORED_CUDA_RNG')
    old._rng({k: v for k, v in ck['rng_state'].items() if k != 'torch_cuda'},
             device=torch.device('cpu'), seed=numerical['seed'])
    model.load_state_dict(ck['model_state'], strict=True)
    with torch.no_grad():
        for name, value in model.named_buffers(): value.copy_(ck['buffer_state'][name])
    state._require_model_binding(model); model.backbone.require_input_normalization_state()
    _need(model_sha(model.state_dict()) == REVERSED_MODEL
          and old._tree_sha(dict(model.named_buffers())) == ck['buffer_state_sha256'], 'REVERSED_LOADED_EXACT')
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return dict(model_state_sha256=REVERSED_MODEL, buffer_state_sha256=ck['buffer_state_sha256'],
        evidence_sha256=ck['evidence_sha256'], parameter_binding_sha256=binding['contract_sha256'],
        load_kind='WEIGHTS_AND_ALL_BUFFERS_ONLY', optimizer_state_restored=False,
        cuda_rng_restored=False, canonical_resume_authority=False)


def _reversed_batch_parity(captured, baseline, left, right, *, np):
    axes = all(np.array_equal(captured[k], baseline[k][left:right]) for k in ('parents', 'issues'))
    flags, errors = {}, {}
    for task in ('state', 'future'):
        got = captured[task + '_activity_side_logits'][:, 0]
        expected = baseline[task + '_raw'][left:right, 0]
        _need(got.dtype == expected.dtype == np.float32, 'REVERSED_RAW_DTYPE')
        flags[task] = (np.array_equal(got.view(np.uint32), expected.view(np.uint32))
            and np.array_equal(captured['query_native'][:, 0], baseline[task + '_query'][left:right, 0]))
        flags[task + '_masks'] = all(np.array_equal(captured[task + '_' + key][:, 0],
            baseline[task + '_' + key][left:right, 0]) for key in ('allowed', 'eligible', 'mature'))
        errors[task] = float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))
    return dict(M5_raw_bitexact=flags['state'] and flags['future'],
        exact_M5_axes_masks=axes and flags['state_masks'] and flags['future_masks'],
        max_abs_M5_raw_error=errors)


def _save_reversed(arrays, rows, *, partial):
    import numpy as np
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    name = 'FIT_REPRESENTATIONS' + ('_PARTIAL' if partial else '') + '.npz'
    stream = io.BytesIO()
    np.savez(stream, **{k: v[:rows] for k, v in arrays.items()})
    _need(0 < stream.tell() <= REVERSED_BOUNDS['max_cache_bytes'], 'REVERSED_CACHE_BYTES')
    _publish_bytes(REVERSED_ROOT / name, stream.getvalue())
    return _ref(REVERSED_ROOT / name)


def _run_reversed(plan_path, plan_sha256, out_dir, *, device):
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_endpoint_batch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    started = time.monotonic()
    _need(device == 'cuda', 'REVERSED_GPU_ONLY')
    cap = require_guarded_cuda_producer_execution(); cap['lock'] = require_capped_lock_ancestry()
    _need(cap['cgroup_path'] == '/sys/fs/cgroup' + cap['lock']['scope']
          and cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
          and cap['swap'] == 512 * 1024**2, 'REVERSED_EXACT_CAP')
    for key, value in (('MAX_POWER_LIMIT_W', '250'), ('MAX_POWER_DRAW_W', '170'), ('MAX_CORE_TEMP_C', '65'), ('MAX_MEMORY_TEMP_C', '80')):
        _need(os.environ.get('GX1_TRAINER_GPU_' + key) == value, 'REVERSED_HARDWARE_GUARD:' + key)
    checked = _inspect_reversed(plan_path, plan_sha256, out_dir, execute=True)
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'REVERSED_INIT_ORDER')
    publish = lambda name, value: _publish(name, value, root=REVERSED_ROOT)
    claim = publish('CLAIM.json', dict(plan=checked['plan_ref'], policy=checked['policy_ref'],
        source_commit=checked['head'], cap=cap, training_enabled=False, test_data_used=False))
    counts = dict(model_constructions=0, checkpoint_loads=0, backbone_encodings=0, forward_returns=0,
        optimizer_constructions=0, optimizer_updates=0, backwards=0, checkpoint_writes=0,
        resume_state_load_calls=0, resume_state_load_returns=0)
    provider = prepared = model = None; arrays = {}; rows_done = 0; artifact = None; identity = {}
    parity = dict(M5_raw_bitexact=True, exact_M5_axes_masks=True, max_abs_M5_raw_error={'state': 0., 'future': 0.})
    timings = {}; before = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('REVERSED cache 3600-second bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1., WALL_SECONDS - 60 - (time.monotonic() - started)))
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO), source_commit=checked['head']))
        for ref in checked['bound_refs']: campaign.require_binding(ref, label='reversed bookend', verify_file=True)
        for name, digest in checked['current_sources'].items():
            _need(campaign.file_sha256(REPO / name) == digest, 'REVERSED_SOURCE_BOOKEND:' + name)
    try:
        recipe, receipts, view = (checked[k] for k in ('recipe', 'receipts', 'view'))
        numerical = recipe['numerical_recipe']; torch_device = torch.device('cuda')
        trainer._set_deterministic(numerical['seed'], torch_device, numerical['precision_policy'])
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _need(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'REVERSED_THREADS')
        provider, provider_plan, prepared, rows, issues = runner._prepared_fold1(receipts, view, np=np)
        endpoint, future = load_bound_five_endpoint_targets(), load_bound_future_displacement()
        axes = endpoint.mature_oof_axes(fold_id=1)
        _need(np.array_equal(rows, axes['fit_parent_rows']) and np.array_equal(issues, axes['fit_decision_ns'])
              and len(rows) == FIT_ROWS, 'REVERSED_FIT_AXIS')
        keys = ['parents', 'issues'] + [task + '_' + key for task in ('state', 'future')
            for key in ('raw', 'query', 'allowed', 'eligible', 'mature')]
        with np.load(checked['plan']['inputs']['train_capture']['path'], allow_pickle=False) as archive:
            baseline = {key: archive[key].copy() for key in keys}
        _need(np.array_equal(rows, baseline['parents']) and np.array_equal(issues, baseline['issues']), 'REVERSED_CAPTURE_AXIS')
        counts['model_constructions'] += 1
        backbone = views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,
            recipe=checked['origin']['original_recipe'], verify_files=True, ta_sf=True)
        _need(model.research_objective['contract_sha256'] == recipe['expected_model_contract_sha256'], 'REVERSED_MODEL_CONTRACT')
        order, _ = build_reverse_order(rows, recipe['batch_indices'])
        counts['checkpoint_loads'] += 1
        ck = common._load_state(checked['plan']['inputs']['reversed_evidence'], torch=torch,
            maximum=recipe['max_checkpoint_bytes'], counters=counts)
        identity = _load_reversed_weights(model, ck, checked, torch.from_numpy(order.copy()), torch=torch)
        del ck
        model.to(torch_device)
        def signature():
            return dict(model=model_sha(model.state_dict()), buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=torch_device)),
                modes={n: bool(m.training) for n, m in model.named_modules()})
        before = signature()
        _need(before['model'] == REVERSED_MODEL and not any(p.requires_grad for p in model.parameters()), 'REVERSED_FROZEN')
        arrays = {key: np.empty(shape, dtype) for key, (shape, dtype) in runner._ta_m5_capture_layout(FIT_ROWS).items()}
        timings['setup_seconds'] = time.monotonic() - started; tick = time.monotonic()
        for left in range(0, FIT_ROWS, BATCH_SIZE):
            right = min(left + BATCH_SIZE, FIT_ROWS)
            raw, parents, clocks = runner._feature_batch(prepared, view, receipts, rows, issues, left, right, np=np)
            features = {key: torch.from_numpy(value).to(torch_device) for key, value in raw.items()}
            _need(counts['backbone_encodings'] < 1849, 'REVERSED_FORWARD_CAP')
            counts['backbone_encodings'] += 1
            report = evaluate_endpoint_batch(model, features, endpoint, future, parents, clocks,
                fold_id=1, partition='fit', future_weight=1., frozen_priors=checked['priors'], capture_m5_readout=True)
            counts['forward_returns'] += 1
            captured = report['m5_readout']
            runner._ta_m5_capture_batch(arrays, captured, left=left, right=right, parents=parents, issues=clocks)
            rows_done = right
            one = _reversed_batch_parity(captured, baseline, left, right, np=np)
            for key in ('M5_raw_bitexact', 'exact_M5_axes_masks'): parity[key] &= one[key]
            for task in ('state', 'future'):
                parity['max_abs_M5_raw_error'][task] = max(parity['max_abs_M5_raw_error'][task], one['max_abs_M5_raw_error'][task])
            _need(one['M5_raw_bitexact'] and one['exact_M5_axes_masks'], 'REVERSED_M5_NATIVE_PARITY')
            if counts['backbone_encodings'] % 256 == 0:
                print(json.dumps(dict(status='PROGRESS', rows=rows_done, forwards=counts['backbone_encodings'],
                    elapsed_seconds=time.monotonic()-started)), flush=True)
        timings['capture_seconds'] = time.monotonic() - tick
        artifact = _save_reversed(arrays, rows_done, partial=False)
        _need(rows_done == FIT_ROWS and counts['backbone_encodings'] == counts['forward_returns'] == 1849
              and signature() == before and all(p.grad is None for p in model.parameters()), 'REVERSED_UNCHANGED')
        known = {task: int(np.sum(arrays[task+'_eligible'][:, 0] & arrays[task+'_mature'][:, 0]
            & np.isin(arrays[task+'_allowed'][:, 0], (1, 4)))) for task in ('state', 'future')}
        _need(known == {'state': 10010, 'future': 11909}, 'REVERSED_KNOWN_COUNTS')
        base._release_pages(provider, prepared); prepared = None
        provider.validate_metadata(provider_plan); bookend()
        result = publish('RESULT.json', dict(status='COMPLETE', plan=checked['plan_ref'], claim=claim,
            policy=checked['policy_ref'], source_commit=checked['head'], artifact=artifact,
            model_state_sha256=REVERSED_MODEL, identity=identity, rows=FIT_ROWS, batches=1849,
            query_domain='native_endpoints;M5_equal_to_pass0_sampled', parity_scope='M5_only',
            **parity, known_M5=known, counts=counts,
            arrays_metadata={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
            source_unchanged=True, model_state_unchanged=True, evaluation_state_unchanged=True,
            later_forwards=0, training_enabled=False, test_data_used=False,
            timings={**timings, 'elapsed_seconds': time.monotonic()-started}))
        terminal = publish('TERMINAL.json', dict(status='COMPLETE', exit_code=0, plan=checked['plan_ref'],
            result=result, artifact=artifact, source_unchanged=True,
            no_relaunch_under_this_plan=True, test_data_used=False))
        return dict(result=result, terminal=terminal)
    except BaseException as exc:
        preservation_error = None
        if artifact is None and arrays and rows_done:
            try: artifact = _save_reversed(arrays, rows_done, partial=True)
            except Exception as secondary: preservation_error = repr(secondary)
        source_unchanged = False
        try: bookend(); source_unchanged = True
        except Exception as secondary: preservation_error = (preservation_error or '') + ';bookend:' + repr(secondary)
        result = publish('RESULT.json', dict(status='FAILED_CONSUMED', plan=checked['plan_ref'], claim=claim,
            artifact=artifact, rows=rows_done, counts=counts, identity=identity, parity=parity,
            error=type(exc).__name__+':'+str(exc), preservation_error=preservation_error,
            source_unchanged=source_unchanged, training_enabled=False, test_data_used=False))
        publish('TERMINAL.json', dict(status='FAILED', exit_code=1, plan=checked['plan_ref'], result=result,
            artifact=artifact, source_unchanged=source_unchanged, no_relaunch_under_this_plan=True, test_data_used=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.); signal.signal(signal.SIGALRM, previous_alarm)
        if provider is not None:
            if prepared is not None: base._release_pages(provider, prepared)
            provider.close()
