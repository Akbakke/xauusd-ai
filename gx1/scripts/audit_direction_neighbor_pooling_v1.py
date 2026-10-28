"""Fixed neighboring-head rescore of saved 60-minute direction predictions.

This does not construct or run a model and does not admit training or trading.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time
import numpy as np

from gx1.scripts.verify_direction_gc_mtf_later_followup_v1 import points, independent_auc_draws, probability_rows

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_neighbor_pooling_20261028_001'
DAY = 86400 * 10**9
ALPHA = .05 / 12
SOURCES = (
    'gx1/scripts/audit_direction_neighbor_pooling_v1.py',
    'gx1/scripts/verify_direction_gc_mtf_later_followup_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_direction_neighbor_pooling_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_NEIGHBOR_POOLING_' + reason)


def pool_logits(logits):
    """Unweighted nearest horizon mean: 4/5/6 at 5m, two values at edges."""
    z = np.asarray(logits, dtype=np.float64)
    require(z.ndim == 2 and z.shape[1] == 60 and np.isfinite(z).all(), 'LOGITS')
    result = np.empty_like(z)
    result[:, 0] = (z[:, 0] + z[:, 1]) / 2
    result[:, -1] = (z[:, -2] + z[:, -1]) / 2
    result[:, 1:-1] = (z[:, :-2] + z[:, 1:-1] + z[:, 2:]) / 3
    return result


def measure(archive, target, cutoff, logit_name):
    parents = archive['parent_rows']; n = len(parents)
    target_parents = target['parent_rows']; ix = np.searchsorted(target_parents, parents)
    require(parents.dtype == target_parents.dtype == np.int64 and np.all(np.diff(parents) > 0)
        and np.all(ix < len(target_parents)) and np.array_equal(target_parents[ix], parents), 'PARENTS')
    for key in ('decision_ns', 'terminal_time_ns', 'observed'):
        require(np.array_equal(archive[key], target[key][ix]), 'TARGET_JOIN:' + key)
    require(np.all(archive['decision_ns'] < cutoff), 'DECISION_CUTOFF')
    sign = target['sign'][ix]; z = np.asarray(archive[logit_name], dtype=np.float64)
    require(z.shape == sign.shape == (n, 60) and np.isin(sign, (-1, 0, 1)).all(), 'SHAPES')
    pooled = pool_logits(z); panels = []; primary = None
    for j in range(60):
        eligible = archive['observed'][:, j] & (sign[:, j] != 0) & (archive['terminal_time_ns'][:, j] < cutoff)
        y = np.ascontiguousarray((sign[eligible, j] > 0).astype(np.int8))
        require(len(np.unique(y)) == 2, 'OUTCOME_CLASSES')
        a = np.ascontiguousarray(z[eligible, j]); b = np.ascontiguousarray(pooled[eligible, j])
        panels.append({'horizon_minutes': j + 1, 'eligible': len(y), 'actual_up': int(y.sum()),
            'original': points(y, a), 'neighbor_mean': points(y, b)})
        if j == 4: primary = (eligible, y, a, b)
    eligible, y, a, b = primary; days = archive['decision_ns'] // DAY; intervals = []
    for scheme, keys, seed in (('UTC_DAY', days, 20261027), ('UTC_WEEK_MONDAY', (days + 3) // 7, 20261028)):
        blocks, all_ids = np.unique(keys, return_inverse=True); k = len(blocks); require(k >= 8, 'BLOCKS')
        ids = np.ascontiguousarray(all_ids[eligible], dtype=np.int64)
        draws = np.random.default_rng(seed).integers(0, k, size=(10000, k))
        weights = np.asarray([np.bincount(row, minlength=k) for row in draws], dtype=np.int64)
        values = {'auc': independent_auc_draws(y, b, ids, weights) - independent_auc_draws(y, a, ids, weights)}
        ar, br = probability_rows(y, a), probability_rows(y, b)
        denominator = weights @ np.bincount(ids, minlength=k)
        for metric in ('logloss', 'brier'):
            sums = np.bincount(ids, weights=br[metric] - ar[metric], minlength=k)
            values[metric] = np.divide(weights @ sums, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
        bands = {}
        for metric, data in values.items():
            require(np.isfinite(data).all(), 'BOOTSTRAP_CLASSES')
            lo, hi = np.quantile(data, [ALPHA / 2, 1 - ALPHA / 2], method='linear')
            bands[metric] = {'low': float(lo), 'high': float(hi)}
        intervals.append({'scheme': scheme, 'blocks': k, 'intervals_neighbor_minus_original': bands})
    return {'parents': n, 'panels': panels, 'primary_horizon_minutes': 5, 'paired_intervals': intervals}


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('design-sha256', 'policy-sha256', 'source-head'): parser.add_argument('--' + k, required=True)
    args = parser.parse_args()
    from scipy.stats import rankdata
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    del rankdata
    require(not any(k.split('.')[0] in ('torch', 'sklearn', 'joblib') for k in sys.modules), 'MODEL_PRELOAD')
    initial = candle.git_state(); require(initial == (args.source_head, 'work/gx1-current', ''), 'SOURCE')
    design_ref = {'path': str(REPO/'docs/DIRECTION_NEIGHBOR_POOLING_20261028.json'), 'sha256': args.design_sha256}
    policy_ref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        design = binding.read_bound_json(Path(design_ref['path']), design_ref['sha256'])
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']); gate = policy[STAGE]
        require(design['schema_version'] == 'gx1_direction_neighbor_pooling_design_v1' and design['stage'] == STAGE
            and design['test_data_used'] is False and design['bootstrap_replicates'] == 10000
            and design['per_interval_alpha'] == ALPHA and design['cutoff_ns'] == {'q1':1743465600000000000,'q2':1751328000000000000}, 'DESIGN')
        require(gate['design'] == design_ref and gate['status'] == 'ADMITTED' and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False and all(gate[k] is False for k in ('model_forwards_authorized',
                'optimizer_steps_authorized', 'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'GATE')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
            'full_val_allowed', 'native_launch_authorized')), 'TRAINING_CLOSED')
        require(set(design['sources']) == set(SOURCES) and set(design['inputs']) == {'q1', 'q2', 'corrected_target'}, 'REFERENCES')
        for path, ref in design['sources'].items():
            require(ref['path'] == str(REPO/path), 'SOURCE_PATH'); binding.require_binding(ref, label='source')
        control = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    out = ROOT/'DIRECTION_NEIGHBOR_POOLING_20261028_001'
    require(design['output_dir'] == str(out) and ROOT.resolve(strict=True) == ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT')
    out.mkdir(exist_ok=False)
    def publish(path, value): publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json', {'design':design_ref, 'policy':policy_ref, 'source_head':args.source_head, 'execution':proof, 'lock':lock})
    sys.addaudithook(candle.make_guard(out, [r['path'] for r in design['inputs'].values()],
        [design_ref['path'], policy_ref['path']], source_paths=[r['path'] for r in design['sources'].values()]))
    stage = out/'.OUTPUT.staging'; start = time.monotonic()
    def expired(*unused): raise RuntimeError('DIRECTION_NEIGHBOR_POOLING_WALL_LIMIT')
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(600)
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in design['inputs'].values(): binding.require_binding(ref, label='input')
            with np.load(design['inputs']['corrected_target']['path'], allow_pickle=False) as z:
                target = {k:z[k].copy() for k in ('parent_rows','decision_ns','terminal_time_ns','observed','sign')}
            result = {}
            for name, logit in (('q1','logit_up'), ('q2','original_logit_up')):
                with np.load(design['inputs'][name]['path'], allow_pickle=False) as z:
                    archive = {k:z[k].copy() for k in ('parent_rows','decision_ns','terminal_time_ns','observed',logit)}
                require(len(archive['parent_rows']) == {'q1':17288,'q2':1760}[name], 'FIXED_POPULATION')
                result[name] = measure(archive, target, design['cutoff_ns'][name], logit)
            payload = binding.capture_verified_file_snapshot()
        payload.require_unchanged(); control.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock, 'SOURCE_FREEZE')
        result = {'schema_version':'gx1_direction_neighbor_pooling_result_v1', 'status':'COMPLETE', 'design':design_ref,
            'source_head':args.source_head, 'inputs':design['inputs'], 'fixed_rule':'Mean adjacent logits, including center; two at edges.',
            'primary_interval_count':12, 'per_interval_alpha':ALPHA, 'bootstrap_replicates':10000, 'periods':result,
            'limits':'Fixed post-hoc idea using reused development periods, one seed and a nonrandom Q2 sample; not independent validation or retraining.',
            'model_forwards':0, 'optimizer_steps':0, 'checkpoint_reads':0, 'test_data_used':False, 'economic_edge_qualified':False}
        publish(stage/'RESULT.json', result); publish(stage/'TERMINAL.json', {'status':'COMPLETE', 'exit_code':0,
            'source_unchanged':True, 'inputs_unchanged':True, 'elapsed_seconds':time.monotonic()-start})
        publish_bundle_directory_noreplace(stage, out/'OUTPUT')
        publish(out/'TERMINAL.json', {'status':'COMPLETE', 'exit_code':0, 'design':design_ref})
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json', {'status':'FAILED', 'exit_code':1, 'error':type(exc).__name__+': '+str(exc), 'relaunch_allowed':False})
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__': main()
