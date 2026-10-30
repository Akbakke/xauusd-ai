"""One fixed, paired comparison of saved one-/three-pass direction predictions.

The admitted CPU worker opens only two bound OOF prediction archives. It does
not read market inputs or weights, construct a model, fit, or select thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import numpy as np

from gx1.contracts import entry_direction_block_rank_v1 as rank
from gx1.contracts import entry_direction_probability_reference_v1 as probability
from gx1.contracts import gx1_capped_execution_v1 as cap
from gx1.contracts import local_random_access_campaign_v2 as binding
from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
from gx1.scripts import compare_entry_direction_saved_bps_v1 as saved

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_MATCHED_EXPOSURE_1V3_FOLD1_20261027_002')
STAGE = 'direction_matched_exposure_paired_actual_20261028_001'
HORIZONS = (1, 5, 15)
ROWS = 17288
DAY_NS = 86_400_000_000_000
TRAIN_END = 1735689600000000000
LATER_END = 1743465600000000000
REPLICATES, ALPHA = 10000, .05 / 6
GROUPS = (('UTC_DAY', 20261027), ('UTC_WEEK_MONDAY', 20261028))
SOURCE_PATHS = (
    'gx1/scripts/compare_entry_direction_matched_exposure_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
)


def require(ok, reason):
    if not ok:
        raise RuntimeError('MATCHED_EXPOSURE_' + reason)


def paired_draws(labels, old, new, block_ids, weights):
    """Exact paired duplicated-block AUC and row-weighted proper-score draws."""
    n = len(labels)
    require(labels.dtype == np.int8 and labels.shape == (n,) and n > 0
            and np.isin(labels, (0, 1)).all(), 'LABELS')
    require(all(z.dtype == np.float64 and z.shape == (n,) and np.isfinite(z).all()
                for z in (old, new)), 'SCORES')
    require(block_ids.dtype.kind == 'i' and block_ids.shape == (n,)
            and weights.ndim == 2 and weights.shape[1] > 0
            and np.all(block_ids >= 0) and np.all(block_ids < weights.shape[1]), 'BLOCKS')
    k = weights.shape[1]
    used = np.unique(block_ids)
    compact = np.searchsorted(used, block_ids).astype(np.int64)
    selected = weights[:, used].astype(np.float64, copy=False)
    require(np.all(selected.sum(axis=1) > 0), 'EMPTY_BOOTSTRAP')
    auc = {}
    paired_denominator = None
    for name, z in (('one', old), ('three', new)):
        summary = rank.block_rank_pairs(np.ascontiguousarray(labels), np.ascontiguousarray(z), compact)
        numerator, denominator = rank.weighted_rank_pairs(summary, selected)
        require(np.all(denominator > 0), 'SINGLE_CLASS_BOOTSTRAP')
        if paired_denominator is not None:
            require(np.array_equal(denominator, paired_denominator), 'PAIRED_DENOMINATORS')
        paired_denominator = denominator
        auc[name] = numerator / (2 * denominator)
    count = np.bincount(block_ids, minlength=k)
    denominator = weights @ count
    require(np.all(denominator > 0), 'NO_ELIGIBLE_BOOTSTRAP')
    metrics = {name: probability.direction_probability_metrics(np.ascontiguousarray(labels),
        np.ascontiguousarray(z)) for name, z in (('one', old), ('three', new))}
    result = {'one_auc': auc['one'], 'three_auc': auc['three'], 'auc': auc['three'] - auc['one']}
    for metric in ('logloss', 'brier'):
        delta = metrics['three'][metric + '_by_row'] - metrics['one'][metric + '_by_row']
        sums = np.bincount(block_ids, weights=delta, minlength=k)
        result[metric] = (weights @ sums) / denominator
    require(all(np.isfinite(value).all() for value in result.values()), 'UNDEFINED_BOOTSTRAP')
    return result


def interval(draws, alpha=ALPHA):
    require(draws.ndim == 1 and len(draws) > 0 and np.isfinite(draws).all(), 'INTERVAL_DRAWS')
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2], method='linear')
    return {'low': float(low), 'high': float(high), 'alpha': alpha,
            'replicates': len(draws), 'undefined_replicates': 0}


def score_and_compare(one, three, *, cutoff=LATER_END, replicates=REPLICATES,
                      minimum_blocks=8, alpha=ALPHA):
    """Use all fixed parent dates as the bootstrap universe before masking."""
    saved.require_pair(one, three, np=np)
    require(set(one) == set(three), 'PAIR_MEMBERS')
    parents, decision = one['parent_rows'], one['decision_ns']
    require(len(parents) > 0 and np.all(np.diff(parents) > 0) and np.all(np.diff(decision) > 0), 'ORDER')
    days = decision // DAY_NS
    group_data = []
    for name, seed in GROUPS:
        labels = days if name == 'UTC_DAY' else (days + 3) // 7
        blocks, inverse = np.unique(labels, return_inverse=True)
        require(len(blocks) >= minimum_blocks, 'MINIMUM_BLOCKS')
        weights = saved.block_weights(int(len(blocks)), seed=seed, np=np, replicates=replicates)
        require(np.all(weights.sum(axis=1) == len(blocks)), 'BLOCK_WEIGHT_COUNT')
        group_data.append((name, blocks, inverse.astype(np.int64), weights))
    panels = []
    for j, horizon in enumerate(HORIZONS):
        mature = one['terminal_time_ns'][:, j] < cutoff
        eligible = one['observed'][:, j] & (one['sign'][:, j] != 0) & mature
        counts = {'parents': len(parents), 'eligible': int(eligible.sum()),
                  'observed_mature_zero': int((mature & one['observed'][:, j] & (one['sign'][:, j] == 0)).sum()),
                  'mature_missing': int((mature & ~one['observed'][:, j]).sum()),
                  'immature': int((~mature).sum())}
        require(sum(counts[k] for k in ('eligible','observed_mature_zero','mature_missing','immature')) == len(parents), 'COUNT_PARTITION')
        labels = np.ascontiguousarray((one['sign'][eligible, j] == 1).astype(np.int8))
        require(len(labels) > 0 and np.unique(labels).size == 2, 'BOTH_CLASSES')
        arms = {}
        scores = {}
        for name, data in (('one', one), ('three', three)):
            z = np.ascontiguousarray(data['logit_up'][eligible, j], dtype=np.float64)
            scores[name] = z
            m = probability.direction_probability_metrics(labels, z)
            a = probability.direction_rank_diagnostics(labels, z, score_kind='logit_up')
            arms[name] = {'auc': a['auc'], 'logloss': m['logloss'], 'brier': m['brier'],
                          'up': m['up_rows'], 'down': m['down_rows']}
        groups = []
        for name, blocks, inverse, weights in group_data:
            draws = paired_draws(labels, scores['one'], scores['three'], inverse[eligible], weights)
            groups.append({'scheme': name, 'blocks': len(blocks),
                'resampling_population': 'all fixed later parents before eligibility mask',
                'weights_sha256': saved.digest(weights, np),
                'three_minus_one': {metric: interval(draws[metric], alpha) for metric in ('auc','logloss','brier')},
                'individual_auc_descriptive': {arm: interval(draws[arm + '_auc'], alpha) for arm in ('one','three')}})
        panels.append({'horizon_minutes': horizon, 'eligible_rows': len(labels), 'counts': counts, 'arms': arms,
            'three_minus_one': {metric: arms['three'][metric] - arms['one'][metric]
                                for metric in ('auc','logloss','brier')},
            'group_intervals': groups, 'role': 'primary' if horizon == 5 else 'secondary_descriptive'})
    primary = next(item for item in panels if item['horizon_minutes'] == 5)
    support = {metric: all((group['three_minus_one'][metric]['low'] > 0 if metric == 'auc'
                            else group['three_minus_one'][metric]['high'] < 0)
                           for group in primary['group_intervals']) for metric in ('auc','logloss','brier')}
    return {'panels': panels, 'primary_support': support,
            'primary_joint_gate_supported': all(support.values()),
            'primary_auc_improvement_supported_separately': support['auc'],
            'test_data_used': False, 'economic_edge_qualified': False,
            'scope': 'One repeatedly used chronological TRAIN-development fold; no threshold, checkpoint or horizon selection.'}



def compare_family_value_predictions(train, later, *, arm_order, train_cutoff, later_cutoff,
                                     expected_train_rows, expected_later_rows, expected_axis_hashes,
                                     replicates, seed, familywise_alpha, minimum_blocks=8):
    """Pure current60 family comparison; caller binds source/model/data/plan.

    Reuses exact paired rank/proper-score algebra. TRAIN is descriptive learning
    evidence; uncertainty applies only to predeclared five-minute later contrasts.
    Every arm, including failures, must be accounted for before calling.
    """
    require(isinstance(arm_order, (list, tuple)) and len(arm_order) >= 2
        and arm_order[0] == 'baseline' and len(set(arm_order)) == len(arm_order)
        and set(train) == set(later) == set(arm_order), 'FAMILY_ARM_SET')
    require(type(replicates) is int and replicates > 0 and type(seed) is int
        and 0 < familywise_alpha < 1 and train_cutoff < later_cutoff, 'FAMILY_PROTOCOL')
    mandatory = {'parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns',
                 'observed', 'sign', 'logit_up'}
    axis_keys = {'parent_rows', 'bar_start_ns', 'decision_ns'}
    require(isinstance(expected_axis_hashes, dict) and set(expected_axis_hashes) == {'train', 'later'}
        and all(isinstance(v, dict) and set(v) == axis_keys for v in expected_axis_hashes.values()),
        'FAMILY_BOUND_AXIS_HASHES')
    for partition_name, partition, rows, cutoff in (('train', train, expected_train_rows, train_cutoff),
                                     ('later', later, expected_later_rows, later_cutoff)):
        base = partition['baseline']
        require(all(saved.digest(base[k], np) == expected_axis_hashes[partition_name][k] for k in axis_keys),
            'FAMILY_DECLARED_COHORT')
        require(type(rows) is int and rows > 0 and mandatory <= set(base), 'FAMILY_ROWS_OR_KEYS')
        for arm in arm_order:
            data = partition[arm]
            require(set(data) == set(base), 'FAMILY_ARCHIVE_KEYS:' + arm)
            for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
                require(data[key].dtype == np.int64 and data[key].shape == (rows,), 'FAMILY_AXIS:' + key)
            require(np.all(np.diff(data['parent_rows']) > 0)
                and np.all(np.diff(data['decision_ns']) > 0)
                and np.array_equal(data['decision_ns'], data['bar_start_ns'] + 300_000_000_000)
                and np.all(data['decision_ns'] < cutoff), 'FAMILY_CLOCK')
            for key, dtype in (('terminal_time_ns', np.int64), ('observed', np.bool_), ('sign', np.int8)):
                require(data[key].dtype == dtype and data[key].shape == (rows, 60), 'FAMILY_TARGET:' + key)
            require(np.array_equal(data['terminal_time_ns'], data['decision_ns'][:, None]
                + np.arange(1, 61, dtype=np.int64)[None, :] * 60_000_000_000)
                and np.isin(data['sign'], (-1, 0, 1)).all()
                and np.all(data['sign'][~data['observed']] == 0), 'FAMILY_TARGET_DOMAIN')
            z = data['logit_up']
            require(z.dtype in (np.float32, np.float64) and z.shape == (rows, 60)
                and np.isfinite(z).all(), 'FAMILY_LOGITS')
            for key in set(data) - {'logit_up'}:
                require(data[key].shape == base[key].shape and data[key].dtype == base[key].dtype
                    and data[key].tobytes() == base[key].tobytes(), 'FAMILY_MATCHED:' + key)
    require(train['baseline']['parent_rows'][-1] < later['baseline']['parent_rows'][0]
        and np.all(later['baseline']['decision_ns'] >= train_cutoff), 'FAMILY_SPLIT_ORDER')
    priors = []
    for j in range(60):
        b = train['baseline']
        take = b['observed'][:, j] & (b['sign'][:, j] != 0) & (b['terminal_time_ns'][:, j] < train_cutoff)
        up = int(np.sum(b['sign'][take, j] == 1)); n = int(take.sum())
        require(0 < up < n, 'FAMILY_TRAIN_PRIOR_CLASSES')
        priors.append({'eligible': n, 'up': up, 'probability_up': up / n,
                       'logit_up': float(np.log(up / (n - up)))})
    panels = {}
    for label, partition, cutoff in (('train', train, train_cutoff), ('later', later, later_cutoff)):
        output = []
        b = partition['baseline']; rows = len(b['parent_rows'])
        for j in range(60):
            mature = b['terminal_time_ns'][:, j] < cutoff
            take = b['observed'][:, j] & (b['sign'][:, j] != 0) & mature
            y = np.ascontiguousarray((b['sign'][take, j] == 1).astype(np.int8))
            require(len(y) > 0 and np.unique(y).size == 2, 'FAMILY_SCORE_CLASSES')
            counts = {'parents': rows, 'eligible': int(take.sum()),
                'mature_zero': int(np.sum(mature & b['observed'][:, j] & (b['sign'][:, j] == 0))),
                'mature_missing': int(np.sum(mature & ~b['observed'][:, j])),
                'immature': int(np.sum(~mature))}
            require(sum(counts[k] for k in counts if k != 'parents') == rows, 'FAMILY_COUNT_PARTITION')
            metrics = {}
            for arm in list(arm_order) + ['fixed_train_prior']:
                z = (np.full(len(y), priors[j]['logit_up'], np.float64) if arm == 'fixed_train_prior'
                     else np.ascontiguousarray(partition[arm]['logit_up'][take, j], dtype=np.float64))
                m = probability.direction_probability_metrics(y, z)
                metrics[arm] = {k: v for k, v in m.items() if not k.endswith('_by_row')}
                metrics[arm]['auc'] = probability.direction_rank_diagnostics(y, z, score_kind='logit_up')['auc']
            output.append({'horizon_minutes': j + 1, 'counts': counts, 'arms': metrics,
                'always_up_accuracy': float(np.mean(y)),
                'role': 'primary' if j == 4 else 'descriptive'})
        panels[label] = output
    b = later['baseline']; days = b['decision_ns'] // DAY_NS
    groups = []
    for name, labels in (('UTC_DAY', days), ('UTC_WEEK_MONDAY', (days + 3) // 7)):
        blocks, inverse = np.unique(labels, return_inverse=True)
        require(len(blocks) >= minimum_blocks, 'FAMILY_MINIMUM_BLOCKS')
        weights = saved.block_weights(len(blocks), seed=seed, np=np, replicates=replicates)
        require(np.all(weights.sum(axis=1) == len(blocks)), 'FAMILY_RESAMPLES')
        groups.append((name, blocks, inverse.astype(np.int64), weights))
    alpha = familywise_alpha / ((len(arm_order) - 1) * 3 * 2)
    take = b['observed'][:, 4] & (b['sign'][:, 4] != 0) & (b['terminal_time_ns'][:, 4] < later_cutoff)
    y = np.ascontiguousarray((b['sign'][take, 4] == 1).astype(np.int8))
    full_z = np.ascontiguousarray(b['logit_up'][take, 4], dtype=np.float64)
    primary = []
    for arm in arm_order[1:]:
        removed_z = np.ascontiguousarray(later[arm]['logit_up'][take, 4], dtype=np.float64)
        intervals = []
        for name, blocks, ids, weights in groups:
            # old=excluded,new=full: positive AUC / negative proper-score favours full.
            draws = paired_draws(y, removed_z, full_z, ids[take], weights)
            intervals.append({'scheme': name, 'blocks': len(blocks),
                'resampling_population': 'all fixed later parents before eligibility',
                'weights_sha256': saved.digest(weights, np),
                'full_minus_excluded': {m: interval(draws[m], alpha) for m in ('auc','logloss','brier')}})
        metrics = panels['later'][4]['arms']
        support = {m: all((g['full_minus_excluded'][m]['low'] > 0 if m == 'auc'
                           else g['full_minus_excluded'][m]['high'] < 0) for g in intervals)
                   for m in ('auc','logloss','brier')}
        primary.append({'excluded_arm': arm,
            'full_minus_excluded': {m: metrics['baseline'][m] - metrics[arm][m] for m in ('auc','logloss','brier')},
            'intervals': intervals, 'support_by_metric': support})
    return {'schema_version': 'gx1_family_value_saved_comparison_v1',
        'arm_order': list(arm_order), 'panels': panels, 'own_train_prior': priors,
        'primary_five_minute': primary, 'familywise_alpha': familywise_alpha,
        'per_interval_alpha': alpha, 'replicates': replicates, 'seed': seed,
        'interpretation': 'Conditional family value at fixed recipe and unchanged availability; TRAIN descriptive, Q1 reused development.',
        'economic_edge_qualified': False, 'test_data_used': False}


def _metadata_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {'path','sha256'}
            and Path(ref['path']).suffix in ('.json','.py','.sh'), 'METADATA_REF')
    return binding.require_binding(ref, label='matched exposure metadata')


def _git_state():
    def git(*args):
        return subprocess.check_output(['git','-C',str(REPO),*args], text=True).strip()
    return git('rev-parse','HEAD'), git('branch','--show-current'), git('status','--porcelain=v1','--untracked-files=all')


def _install_guard(payload_refs, allowed_json, output):
    allowed = {item['path'] for item in payload_refs}; reads = {}; denials = []
    def guard(event, args):
        if event in ('socket.connect','socket.bind','socket.getaddrinfo','os.system','os.exec','os.posix_spawn'):
            raise RuntimeError('MATCHED_EXPOSURE_FORBIDDEN_PROCESS_OR_NETWORK')
        if event == 'subprocess.Popen':
            allowed_git = [['git','-C',str(REPO),'rev-parse','HEAD'], ['git','-C',str(REPO),'branch','--show-current'],
                           ['git','-C',str(REPO),'status','--porcelain=v1','--untracked-files=all']]
            require(list(args[1]) in allowed_git, 'FORBIDDEN_SUBPROCESS')
        if event == 'import' and args[0].split('.')[0] in ('torch','jax','tensorflow','sklearn'):
            raise RuntimeError('MATCHED_EXPOSURE_FORBIDDEN_MODEL_IMPORT')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(args[0])); flags = args[2] if len(args) > 2 else 0
        write = isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if path.startswith('/home/andre2/GX1_DATA/'):
            raise RuntimeError('MATCHED_EXPOSURE_RAW_DATA')
        if Path(path).suffix.lower() in ('.pt','.pth','.npz','.npy','.parquet','.csv','.pkl') and path not in allowed:
            raise RuntimeError('MATCHED_EXPOSURE_UNBOUND_PAYLOAD')
        if path.startswith('/home/andre2/GX1_RUNS/'):
            if path in allowed:
                require(not write, 'ORIGINAL_WRITE'); reads[path] = reads.get(path, 0) + 1
            elif path in allowed_json:
                require(not write, 'METADATA_WRITE')
            elif path.startswith(str(output) + '/'):
                require(Path(path).suffix not in ('.pt','.pth','.npz','.npy','.parquet','.csv','.pkl'), 'PAYLOAD_OUTPUT')
            else:
                denials.append(path); raise RuntimeError('MATCHED_EXPOSURE_UNBOUND_OPEN')
    sys.addaudithook(guard)
    return reads, denials


def _read_prediction(ref, *, context_spec):
    path = Path(ref['path']); before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink()
            and 0 < before.st_size <= 64 * 1024 * 1024, 'PREDICTION_FILE')
    with path.open('rb') as f:
        inside = os.fstat(f.fileno())
        require((inside.st_dev,inside.st_ino,inside.st_size,inside.st_mtime_ns,inside.st_ctime_ns)
                == (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns), 'OPEN_IDENTITY')
        raw = f.read(64 * 1024 * 1024 + 1)
        after = os.fstat(f.fileno())
    final = path.lstat()
    identity = lambda x: (x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    require(identity(before) == identity(after) == identity(final) and len(raw) == before.st_size
            and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'PREDICTION_SHA_AND_FREEZE')
    return saved.read_predictions(io.BytesIO(raw), signed_bps=False, rows=ROWS, np=np, context_spec=context_spec)


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o444)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','policy-sha256','source-head'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv); start = time.monotonic()
    scope = cap.require_capped_cpu_audit_execution(); lock = cap.require_capped_lock_ancestry()
    require(lock['scope'] == scope['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    expected_git = (args.source_head, 'work/gx1-current', '')
    require(_git_state() == expected_git, 'CLEAN_SOURCE')
    plan_ref = {'path':str(Path(args.plan)), 'sha256':args.plan_sha256}
    policy_ref = {'path':str(REPO/'NEXT_RUN_POLICY.json'), 'sha256':args.policy_sha256}
    plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
    policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
    fields = {'schema_version','stage','review','design','baseline_report','candidate_report','inputs','sources','output_dir','protocol','test_data_used'}
    require(set(plan) == fields and plan['schema_version'] == 'gx1_direction_matched_exposure_paired_plan_v1'
            and plan['stage'] == STAGE and plan['test_data_used'] is False, 'PLAN')
    stage = policy[STAGE]
    require(stage['status'] == 'ADMITTED' and stage['plan'] == plan_ref and stage['execution_authorized'] is True
            and stage['genuine_prediction_reads_authorized'] is True
            and stage['claimed_invocations'] == stage['completed_invocations'] == 0
            and stage['max_invocations'] == 1 and stage['relaunch_allowed'] is False
            and policy['independent_direction_edge_20261027']['current_unit'] == STAGE, 'ADMISSION')
    require(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed','full_val_allowed','native_launch_authorized'))
            and all(stage.get(k) is False for k in ('model_forwards_authorized','optimizer_steps_authorized','checkpoint_reads_authorized','test_data_used','broker_access_allowed')), 'CLOSED_GATES')
    expected_protocol = {'primary_horizon_minutes':5,'horizons_minutes':[1,5,15],
        'comparisons':['auc','logloss','brier'],'groupings':['UTC_DAY','UTC_WEEK_MONDAY'],
        'per_interval_alpha':ALPHA,'quantile_method':'linear','bootstrap_replicates':REPLICATES,'day_seed':20261027,
        'week_seed':20261028,'minimum_blocks':8,'block_population':'all fixed later parents before eligibility masks',
        'no_undefined_primary_draws':True,'no_threshold_or_checkpoint_selection':True}
    require(plan['protocol'] == expected_protocol and stage['protocol'] == expected_protocol, 'PROTOCOL')
    require(set(plan['sources']) == set(SOURCE_PATHS), 'SOURCE_SET')
    for name, ref in plan['sources'].items():
        require(ref['path'] == str(REPO/name), 'SOURCE_PATH'); _metadata_ref(ref)
    for name in ('review','design','baseline_report','candidate_report'):
        _metadata_ref(plan[name])
    review = binding.read_bound_json(Path(plan['review']['path']), plan['review']['sha256'])
    old_report = binding.read_bound_json(Path(plan['baseline_report']['path']), plan['baseline_report']['sha256'])
    new_report = binding.read_bound_json(Path(plan['candidate_report']['path']), plan['candidate_report']['sha256'])
    require(review['status'] == 'COMPLETE_TRAIN_AND_OOF_PAIRED_EVALUATION_PENDING' and review['test_data_used'] is False
            and review['baseline_oof'] == plan['baseline_report'] and review['candidate_oof'] == plan['candidate_report']
            and plan['inputs'] == {'one':review['baseline_predictions'],'three':review['candidate_predictions']}
            and old_report['predictions'] == plan['inputs']['one'] and new_report['predictions'] == plan['inputs']['three'], 'PROVENANCE')
    require(old_report['fold'] == new_report['fold'] and old_report['fold']['oof_cutoff_ns'] == LATER_END
            and old_report['fold']['train_cutoff_ns'] == TRAIN_END and old_report['feature_view_identity'] == new_report['feature_view_identity']
            and old_report['own_prefix_prior'] == new_report['own_prefix_prior'] and old_report['context_map'] == new_report['context_map']
            and old_report['fresh_identity_sha256'] == new_report['fresh_identity_sha256']
            and old_report['own_initial_state'] == new_report['own_initial_state']
            and old_report['test_data_used'] is new_report['test_data_used'] is False, 'REPORT_IDENTITY')
    output = Path(plan['output_dir'])
    require(output.parent == ROOT and output.is_absolute() and not output.exists() and not output.is_symlink(), 'OUTPUT')
    payload_refs = [plan['inputs']['one'], plan['inputs']['three']]
    require(len({x['path'] for x in payload_refs}) == 2 and all(Path(x['path']).suffix == '.npz' for x in payload_refs), 'PAYLOADS')
    output.mkdir(mode=0o700)
    _write_new(output/'CLAIM.json', {'plan':plan_ref,'source_head':args.source_head,'test_data_used':False})
    allowed_json = {plan[name]['path'] for name in ('review','design','baseline_report','candidate_report')} | {plan_ref['path']}
    reads, denials = _install_guard(payload_refs, allowed_json, output)
    one = _read_prediction(payload_refs[0], context_spec=CONTEXT_SPEC)
    three = _read_prediction(payload_refs[1], context_spec=CONTEXT_SPEC)
    require(set(reads) == {x['path'] for x in payload_refs} and all(v == 1 for v in reads.values()) and not denials, 'OPEN_SCOPE')
    require(np.all(one['decision_ns'] >= TRAIN_END) and np.all(one['decision_ns'] < LATER_END), 'LATER_SPLIT')
    result = score_and_compare(one, three)
    for report, arm in ((old_report,'one'),(new_report,'three')):
        for panel, original in zip(result['panels'], report['model_metrics']['per_horizon']):
            require(panel['horizon_minutes'] == original['horizon_minutes'] and panel['eligible_rows'] == original['counts']['eligible_cells']
                    and all(np.isclose(panel['arms'][arm][k], original[k], rtol=1e-12, atol=1e-12)
                            for k in ('logloss','brier')), 'ORIGINAL_SCORE_REPRODUCTION')
    require(_git_state() == expected_git and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
    result.update(schema_version='gx1_direction_matched_exposure_paired_result_v1',
        status='COMPLETE_REUSED_DEVELOPMENT_NO_EDGE_CLAIM', plan=plan_ref, source_head=args.source_head,
        matched_parent_clock_target_mask_contexts=True, original_scores_reproduced=True,
        payload_archives_read=2, checkpoint_reads=0, model_forwards=0, optimizer_steps=0,
        elapsed_seconds=time.monotonic()-start, full_train_assessment_pending=True)
    output_ref = _write_new(output/'RESULT.json', result)
    print(json.dumps({'status':result['status'],'result':output_ref,'primary':next(x for x in result['panels'] if x['horizon_minutes']==5),
                      'primary_support':result['primary_support']}, sort_keys=True))


if __name__ == '__main__':
    main()
