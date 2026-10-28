"""Independently rescore a saved later GC-history intervention; no model call."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_gc_mtf_later_verify_20261028_001'
CONTEXT_STAGE = 'independent_direction_gc_context_verify_20261028_001'
DAY_NS = 86400 * 10**9
ALPHA = 0.05 / 12
REPLICATES = 10000
SOURCE_PATHS = (
    'gx1/scripts/verify_direction_gc_mtf_later_followup_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_verify_direction_gc_mtf_later_followup_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_GC_MTF_LATER_VERIFY_' + reason)


def points(labels, logits):
    """Independent rank-sum AUC and stable probability scores."""
    from scipy.stats import rankdata
    require(labels.dtype == np.int8 and logits.dtype == np.float64 and labels.ndim == 1
        and logits.shape == labels.shape and np.isin(labels, (0, 1)).all() and np.isfinite(logits).all(), 'POINT_DOMAIN')
    up = int(labels.sum()); down = len(labels) - up
    require(up > 0 and down > 0, 'POINT_CLASSES')
    auc = (float(rankdata(logits, method='average')[labels == 1].sum()) - up * (up + 1) / 2) / (up * down)
    p = np.empty(len(logits), np.float64); positive = logits >= 0
    p[positive] = 1 / (1 + np.exp(-logits[positive]))
    e = np.exp(logits[~positive]); p[~positive] = e / (1 + e)
    return {'auc': float(auc), 'logloss': float(np.mean(np.logaddexp(0., logits) - labels * logits)),
        'brier': float(np.mean((p - labels) ** 2))}


def independent_auc_draws(labels, scores, ids, weights):
    """Exact duplicate-block AUC via per-block searchsorted, including ties."""
    blocks = weights.shape[1]; wins = np.zeros((blocks, blocks), np.float64)
    positive = np.bincount(ids[labels == 1], minlength=blocks)
    negative = np.bincount(ids[labels == 0], minlength=blocks)
    for i in range(blocks):
        up_scores = scores[(ids == i) & (labels == 1)]
        for j in range(blocks):
            down_scores = np.sort(scores[(ids == j) & (labels == 0)])
            wins[i, j] = np.searchsorted(down_scores, up_scores, side='left').sum() + np.searchsorted(down_scores, up_scores, side='right').sum()
    numerator = np.sum((weights @ wins) * weights, axis=1)
    denominator = 2 * (weights @ positive) * (weights @ negative)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)


def probability_rows(labels, scores):
    p = np.empty(len(scores), np.float64); positive = scores >= 0
    p[positive] = 1 / (1 + np.exp(-scores[positive]))
    e = np.exp(scores[~positive]); p[~positive] = e / (1 + e)
    return {'logloss': np.logaddexp(0., scores) - labels * scores, 'brier': (p - labels) ** 2}


def verify_metrics(pair, corrected, result, *, replicates=REPLICATES):
    """Verify exact sample/target join and both arms on every minute and block draw."""
    fold = result['later_fold']; first, last = fold['oof_start_row'], fold['oof_end_row']
    indices = np.concatenate([np.arange(start, min(start + 16, last), dtype=np.int64)
        for start in range(first, last, 160)])
    n = len(indices); shape = (n, 60)
    require(result['selection'] == {'policy': 'every tenth contiguous 16-row later batch', 'batch_size': 16,
        'batch_stride': 10, 'selected_parents': n, 'selected_batches': len(range(first, last, 160))}, 'SELECTION')
    for key in ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign'):
        require(key in pair and key in corrected and np.array_equal(pair[key], corrected[key][indices]), 'TARGET_JOIN:' + key)
    require(pair['original_logit_up'].shape == pair['ablated_logit_up'].shape == shape
        and pair['original_logit_up'].dtype == pair['ablated_logit_up'].dtype == np.float32, 'LOGIT_SHAPE')
    panels = result['metrics']['per_horizon']; require(len(panels) == 60, 'PANEL_COUNT')
    max_point_error = 0.0; cache = {}
    for j in range(60):
        eligible = pair['observed'][:, j] & (pair['sign'][:, j] != 0) & (pair['terminal_time_ns'][:, j] < fold['oof_cutoff_ns'])
        labels = np.ascontiguousarray((pair['sign'][eligible, j] == 1).astype(np.int8))
        require(panels[j]['horizon_minutes'] == j + 1 and panels[j]['eligible'] == len(labels), 'PANEL:' + str(j + 1))
        for name, field in (('original', 'original_logit_up'), ('ablated', 'ablated_logit_up')):
            z = np.ascontiguousarray(pair[field][eligible, j], dtype=np.float64)
            values = points(labels, z)
            for metric, value in values.items():
                error = abs(value - panels[j]['arms'][name][metric]); max_point_error = max(max_point_error, error)
                require(np.isfinite(error) and error <= 1e-12, 'POINT_REPLAY:' + str(j + 1) + ':' + name + ':' + metric)
        if j in (4, 59):
            cache[j + 1] = (eligible, labels)
    reported = {(x['horizon_minutes'], x['scheme']): x for x in result['metrics']['paired_intervals']}
    require(len(reported) == 4 and result['metrics']['per_interval_alpha'] == ALPHA, 'INTERVAL_SCOPE')
    max_interval_error = 0.0
    for horizon in (5, 60):
        j = horizon - 1; eligible, labels = cache[horizon]
        first_z = np.ascontiguousarray(pair['original_logit_up'][eligible, j], dtype=np.float64)
        second_z = np.ascontiguousarray(pair['ablated_logit_up'][eligible, j], dtype=np.float64)
        days = pair['decision_ns'][eligible] // DAY_NS
        for scheme, keys, seed in (('UTC_DAY', days, 20261027), ('UTC_WEEK_MONDAY', (days + 3) // 7, 20261028)):
            blocks, ids = np.unique(keys, return_inverse=True); k = len(blocks)
            require(k >= 8, 'BLOCK_SUPPORT')
            draws = np.random.default_rng(seed).integers(0, k, size=(replicates, k))
            weights = np.asarray([np.bincount(row, minlength=k) for row in draws], dtype=np.int64)
            a = independent_auc_draws(labels, first_z, ids, weights); b = independent_auc_draws(labels, second_z, ids, weights)
            values = {'auc_ablated_minus_original': b - a}
            first_metrics, second_metrics = probability_rows(labels, first_z), probability_rows(labels, second_z)
            counts = np.bincount(ids, minlength=k); denominator = weights @ counts
            for metric in ('logloss', 'brier'):
                sums = np.bincount(ids, weights=second_metrics[metric] - first_metrics[metric], minlength=k)
                values[metric + '_ablated_minus_original'] = np.divide(weights @ sums, denominator,
                    out=np.full(replicates, np.nan), where=denominator > 0)
            record = reported[(horizon, scheme)]; require(record['blocks'] == k, 'BLOCK_COUNT')
            for name, samples in values.items():
                finite = np.isfinite(samples); valid = int(finite.sum()); invalid = replicates - valid
                expected = record['intervals'][name]
                require(expected['valid_replicates'] == valid and expected['undefined_replicates'] == invalid
                    and expected['per_interval_alpha'] == ALPHA and valid > 0, 'INTERVAL_REPLICATES')
                low, high = np.quantile(samples[finite], [ALPHA / 2, 1 - ALPHA / 2], method='linear')
                error = max(abs(float(low) - expected['low']), abs(float(high) - expected['high']))
                max_interval_error = max(max_interval_error, error)
                require(np.isfinite(error) and error <= 1e-12, 'INTERVAL_REPLAY:' + name)
    return {'selected_parents': n, 'horizons': 60, 'arms': 2, 'paired_interval_families': 4,
        'replicates_per_family': replicates, 'max_absolute_point_error': max_point_error,
        'max_absolute_interval_error': max_interval_error}


def require_prior_same(pair, prior):
    for key in ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'original_logit_up'):
        require(key in pair and key in prior and pair[key].shape == prior[key].shape
            and pair[key].dtype == prior[key].dtype
            and np.ascontiguousarray(pair[key]).tobytes() == np.ascontiguousarray(prior[key]).tobytes(),
            'PRIOR_BYTE_IDENTITY:' + key)


def experiment_spec(mode):
    require(mode in ('history', 'context'), 'MODE')
    if mode == 'context':
        return {'stage': CONTEXT_STAGE, 'stem': 'gx1_direction_gc_context_verify',
            'design': 'DIRECTION_GC_CONTEXT_VERIFY_20261028.json',
            'output': 'DIRECTION_GC_CONTEXT_VERIFY_20261028_001',
            'prior': 'independent_direction_gc_context_followup_20261028_002',
            'result_schema': 'gx1_direction_gc_context_followup_result_v1',
            'peak_power': 'peak_power_draw_w=132.48'}
    return {'stage': STAGE, 'stem': 'gx1_direction_gc_mtf_later_verify',
        'design': 'DIRECTION_GC_MTF_LATER_VERIFY_20261028.json',
        'output': 'DIRECTION_GC_MTF_LATER_VERIFY_20261028_001',
        'prior': 'independent_direction_gc_mtf_later_followup_20261028_001',
        'result_schema': 'gx1_direction_gc_mtf_later_followup_result_v1',
        'peak_power': 'peak_power_draw_w=134.26'}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--experiment', choices=('history', 'context'), default='history')
    args = parser.parse_args(argv)
    spec = experiment_spec(args.experiment)
    from scipy.stats import rankdata  # Load before restricted file scope.
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    del rankdata
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'PRELOADED_MODEL')
    initial = candle.git_state(); require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO / 'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir', 'wall_seconds', 'test_data_used'}
            and plan['schema_version'] == spec['stem'] + '_plan_v1' and plan['stage'] == spec['stage']
            and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO / 'docs' / spec['design']), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == spec['stem'] + '_design_v1'
            and design['inputs'] == plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']); gate = policy[spec['stage']]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
            and all(gate[k] is False for k in ('model_forwards_authorized', 'optimizer_steps_authorized',
                'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO / name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        keys = {'followup_plan', 'result', 'inner_terminal', 'terminal', 'claim', 'guard_log',
            'paired_outputs', 'corrected_target', 'fold_design'}
        if args.experiment == 'context':
            keys.add('previous_pair')
        require(set(plan['inputs']) == keys, 'INPUT_SET')
        meta = {}
        for key in ('followup_plan', 'result', 'inner_terminal', 'terminal', 'claim', 'fold_design'):
            ref = plan['inputs'][key]; meta[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        prior = policy[spec['prior']]
        require(prior['status'] == 'COMPLETE_ONE_SHOT_CONSUMED' and prior['execution_authorized'] is False
            and prior['plan'] == plan['inputs']['followup_plan'] and prior['result'] == plan['inputs']['result']
            and prior['terminal'] == plan['inputs']['terminal'] and prior['claim'] == plan['inputs']['claim'], 'PRIOR_BINDING')
        prior_plan, result, inner, terminal, claim, fold_design = (meta[k] for k in
            ('followup_plan', 'result', 'inner_terminal', 'terminal', 'claim', 'fold_design'))
        require(result['schema_version'] == spec['result_schema'] and result['status'] == 'COMPLETE'
            and result['plan'] == claim['plan'] == terminal['plan'] == plan['inputs']['followup_plan']
            and result['source_head'] == claim['source_head'] and result['later_fold'] == fold_design['folds'][1]
            and result['fold1_reference_logits_byte_identical'] is True and result['model_unchanged'] is True
            and result['forward_calls'] == 221 and result['optimizer_steps'] == 0 and result['test_data_used'] is False
            and result['corrected_target_identity'] == 'exact_original_decimal_midpoint_v1'
            and {k: result['paired_outputs'][k] for k in ('path', 'sha256')} == plan['inputs']['paired_outputs']
            and prior_plan['inputs']['corrected_target'] == plan['inputs']['corrected_target']
            and prior_plan['inputs']['fold_design'] == plan['inputs']['fold_design']
            and (args.experiment != 'context' or (prior_plan['inputs']['prior_paired'] == plan['inputs']['previous_pair']
                and result['later_reference_logits_byte_identical'] is True
                and result['gc_context_layout']['gc_fields'] == 2410
                and result['gc_context_layout']['value_mask_pairs'] == 1205)), 'RESULT_LINEAGE')
        require(inner['status'] == terminal['status'] == 'COMPLETE' and inner['exit_code'] == terminal['exit_code'] == 0
            and inner['source_unchanged'] is inner['inputs_unchanged'] is True and inner['test_data_used'] is False
            and result['later_fold']['oof_cutoff_ns'] < 1782864000000000000, 'TERMINAL_AND_TEST_SEAL')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir']); require(out.parent == ROOT and out.name == spec['output']
        and ROOT.resolve(strict=True) == ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    payload_names = ('paired_outputs', 'corrected_target') + (('previous_pair',) if args.experiment == 'context' else ())
    payloads = [plan['inputs'][k] for k in payload_names]
    logref = plan['inputs']['guard_log']
    metadata = [plan_ref['path'], policy_ref['path'], plan['design']['path'], logref['path']]
    metadata += [plan['inputs'][k]['path'] for k in ('followup_plan', 'result', 'inner_terminal', 'terminal', 'claim', 'fold_design')]
    sys.addaudithook(candle.make_guard(out, [ref['path'] for ref in payloads], metadata,
        source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started = time.monotonic(); previous_handler = signal.getsignal(signal.SIGALRM); previous_profile = sys.getprofile()
    def expired(*unused):
        raise RuntimeError('DIRECTION_GC_MTF_LATER_VERIFY_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(candle.no_queries_or_models)
    stage = out / '.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads + [logref]:
                candle.regular(ref['path']); binding.require_binding(ref, label='input')
            guard = Path(logref['path']).read_text()
            require('event=exit child_status=0' in guard and 'peak_core_temp_c=53' in guard
                and 'peak_memory_temp_c=56' in guard and spec['peak_power'] in guard, 'GUARD_TERMINAL')
            require(Path(plan['inputs']['paired_outputs']['path']).stat().st_size == result['paired_outputs']['bytes'], 'PAIRED_SIZE')
            arrays = {}
            archive_specs = [
                ('pair', plan['inputs']['paired_outputs'], 100 * 1024**2,
                    ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'original_logit_up', 'ablated_logit_up')),
                ('corrected', plan['inputs']['corrected_target'], 800 * 1024**2,
                    ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign'))]
            if args.experiment == 'context':
                archive_specs.append(('previous', plan['inputs']['previous_pair'], 100 * 1024**2,
                    ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'original_logit_up')))
            for label, ref, maximum, names in archive_specs:
                with Path(ref['path']).open('rb') as handle:
                    with zipfile.ZipFile(handle) as archive:
                        require(sum(item.file_size for item in archive.infolist()) <= maximum, 'ARCHIVE_SIZE:' + label)
                    handle.seek(0)
                    with np.load(handle, allow_pickle=False) as archive:
                        require(all(name in archive.files for name in names), 'ARCHIVE_FIELDS:' + label)
                        arrays[label] = {name: archive[name].copy() for name in names}
            verified = verify_metrics(arrays['pair'], arrays['corrected'], result)
            if args.experiment == 'context':
                require_prior_same(arrays['pair'], arrays['previous'])
                verified['all_prior_originals_byte_identical'] = True
            input_snapshot = binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
        output = {'schema_version': spec['stem'] + '_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': args.source_head, 'followup_result': plan['inputs']['result'],
            'paired_outputs': plan['inputs']['paired_outputs'], 'corrected_target': plan['inputs']['corrected_target'],
            'verification': verified, 'method': 'Independent rank-sum point AUC and per-block searchsorted weighted AUC; separate probability and bootstrap calculations.',
            'limits': 'Numerical and target-join verification on one reused later-development sample, not independent generalization, retraining, trading or economic edge.',
            'new_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
        publish(stage / 'RESULT.json', output)
        publish(stage / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
            'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - started, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out / 'OUTPUT')
        publish(out / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'result_path': str(out / 'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out / 'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False})
        raise
    finally:
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
