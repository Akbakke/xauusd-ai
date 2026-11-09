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
