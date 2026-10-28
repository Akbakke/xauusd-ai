"""Rescore separately saved bps and exposure experiments against exact targets.

No model, checkpoint, optimizer, fresh training, broker or sealed TEST access.
The models were fitted with legacy targets; this does not measure retraining.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

import numpy as np

from gx1.scripts.verify_direction_gc_mtf_later_followup_v1 import (
    points, independent_auc_draws, probability_rows,
)

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_corrected_factors_20261028_001'
SOURCE_PATHS = (
    'gx1/scripts/audit_corrected_bps_exposure_v1.py',
    'gx1/scripts/verify_direction_gc_mtf_later_followup_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_corrected_bps_exposure_v1.py',
)
HORIZONS = (1, 5, 15)
COLS = (0, 4, 14)
CUTOFF = 1743465600000000000
DAY = 86400 * 10**9
ALPHA = 0.05 / 12
REPLICATES = 10000


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('CORRECTED_BPS_EXPOSURE_' + reason)


def validate_join(arms, target):
    """Require identical original rows, context, clocks and correct decimal join."""
    require(set(arms) == {'one', 'bps', 'three'}, 'ARMS')
    base = arms['one']; n = len(base['parent_rows'])
    require(n > 0 and base['parent_rows'].dtype == np.int64 and np.all(np.diff(base['parent_rows']) > 0), 'PARENTS')
    shared = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')
    contexts = sorted(k for k in base if k.startswith('context__'))
    require(len(contexts) > 0 and all(set(k for k in arm if k.startswith('context__')) == set(contexts)
        for arm in arms.values()), 'CONTEXT_INVENTORY')
    for name, arm in arms.items():
        require(all(k in arm and arm[k].dtype == base[k].dtype and arm[k].shape == base[k].shape
            and np.ascontiguousarray(arm[k]).tobytes() == np.ascontiguousarray(base[k]).tobytes()
            for k in shared + tuple(contexts)), 'ORIGINAL_PAIR:' + name)
        z = arm['logit_up']
        require(z.dtype == np.float32 and z.shape == (n, 3) and np.isfinite(z).all(), 'LOGITS:' + name)
    parents = target['parent_rows']; at = np.searchsorted(parents, base['parent_rows'])
    require(parents.dtype == np.int64 and parents.ndim == 1 and np.all(np.diff(parents) > 0)
        and np.all(at < len(parents)) and np.array_equal(parents[at], base['parent_rows']), 'CORRECTED_PARENT_JOIN')
    for name in ('bar_start_ns', 'decision_ns'):
        require(np.array_equal(target[name][at], base[name]), 'CORRECTED_AXIS:' + name)
    for name in ('terminal_time_ns', 'observed'):
        require(np.array_equal(target[name][at][:, COLS], base[name]), 'CORRECTED_CLOCK_MASK:' + name)
    sign = np.ascontiguousarray(target['sign'][at][:, COLS]); old = base['sign']
    require(sign.dtype == np.int8 and sign.shape == old.shape == (n, 3)
        and np.isin(sign, (-1, 0, 1)).all() and np.all(sign[~base['observed']] == 0), 'CORRECTED_SIGN')
    changed = sign != old
    require(np.all(sign[changed] == 0) and np.all(old[changed] != 0), 'NO_OPPOSITE_SIGN_FLIPS')
    return sign, [int(changed[:, j].sum()) for j in range(3)]


def evaluate(arms, corrected_sign, *, replicates=REPLICATES):
    """Three fixed horizons and predeclared paired day/week comparisons at 5m."""
    base = arms['one']; n = len(base['parent_rows']); panels = []; selected = None
    for j, horizon in enumerate(HORIZONS):
        eligible = base['observed'][:, j] & (corrected_sign[:, j] != 0) & (base['terminal_time_ns'][:, j] < CUTOFF)
        labels = np.ascontiguousarray((corrected_sign[eligible, j] > 0).astype(np.int8))
        require(len(labels) > 0 and len(np.unique(labels)) == 2, 'BOTH_CLASSES:' + str(horizon))
        scores = {}
        for name, arm in arms.items():
            z = np.ascontiguousarray(arm['logit_up'][eligible, j], dtype=np.float64)
            scores[name] = {**points(labels, z), 'accuracy': float(np.mean((z > 0) == labels)),
                'predicted_up': int((z > 0).sum())}
        panels.append({'horizon_minutes': horizon, 'eligible': len(labels), 'actual_up': int(labels.sum()),
            'always_long_accuracy': float(labels.mean()), 'arms': scores})
        if horizon == 5: selected = (eligible, labels)
    eligible, labels = selected; days = base['decision_ns'] // DAY; comparisons = []
    for scheme, keys, seed in (('UTC_DAY', days, 20261027), ('UTC_WEEK_MONDAY', (days + 3) // 7, 20261028)):
        blocks, ids_all = np.unique(keys, return_inverse=True); k = len(blocks); require(k >= 8, 'BLOCK_SUPPORT')
        ids = np.ascontiguousarray(ids_all[eligible], dtype=np.int64)
        draws = np.random.default_rng(seed).integers(0, k, size=(replicates, k))
        weights = np.asarray([np.bincount(row, minlength=k) for row in draws], dtype=np.int64)
        z = {name: np.ascontiguousarray(arm['logit_up'][eligible, 1], dtype=np.float64) for name, arm in arms.items()}
        auc = {name: independent_auc_draws(labels, value, ids, weights) for name, value in z.items()}
        proper = {name: probability_rows(labels, value) for name, value in z.items()}
        denominator = weights @ np.bincount(ids, minlength=k)
        for name in ('bps', 'three'):
            samples = {'auc': auc[name] - auc['one']}
            for metric in ('logloss', 'brier'):
                sums = np.bincount(ids, weights=proper[name][metric] - proper['one'][metric], minlength=k)
                samples[metric] = np.divide(weights @ sums, denominator, out=np.full(replicates, np.nan), where=denominator > 0)
            intervals = {}
            for metric, values in samples.items():
                finite = np.isfinite(values); require(finite.all(), 'BOOTSTRAP_UNDEFINED')
                low, high = np.quantile(values, [ALPHA / 2, 1 - ALPHA / 2], method='linear')
                intervals[metric] = {'low': float(low), 'high': float(high)}
            comparisons.append({'candidate': name, 'scheme': scheme, 'blocks': k, 'intervals_candidate_minus_one': intervals})
    return {'parents': n, 'panels': panels, 'primary_horizon_minutes': 5, 'paired_intervals': comparisons,
        'per_interval_alpha': ALPHA, 'bootstrap_replicates': replicates,
        'block_population': 'all fixed later parents before eligibility mask'}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    from scipy.stats import rankdata
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
            and plan['schema_version'] == 'gx1_corrected_bps_exposure_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO / 'docs/DIRECTION_CORRECTED_FACTORS_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_corrected_bps_exposure_design_v1' and design['inputs'] == plan['inputs']
            and design['test_data_used'] is False, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']); gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
            'full_val_allowed', 'native_launch_authorized')) and all(gate[k] is False for k in
            ('model_forwards_authorized', 'optimizer_steps_authorized', 'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS) and all(ref['path'] == str(REPO / name)
            for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for value in plan['sources'].values(): candle.regular(value['path']); binding.require_binding(value, label='source')
        keys = {'baseline', 'bps', 'three', 'corrected', 'corrected_result', 'bps_result', 'exposure_plan', 'exposure_result'}
        require(set(plan['inputs']) == keys, 'INPUT_SET')
        meta = {key: binding.read_bound_json(Path(plan['inputs'][key]['path']), plan['inputs'][key]['sha256'])
            for key in ('corrected_result', 'bps_result', 'exposure_plan', 'exposure_result')}
        corrected, bps, exposure_plan, exposure = (meta[k] for k in ('corrected_result', 'bps_result', 'exposure_plan', 'exposure_result'))
        require(corrected.get('status') == 'COMPLETE' and corrected.get('target') == plan['inputs']['corrected']
            and corrected.get('target_identity') == 'exact_original_decimal_midpoint_v1' and corrected.get('test_data_used') is False, 'CORRECTED_LINEAGE')
        require(bps.get('schema_version') == 'gx1_independent_direction_saved_bps_comparison_v1'
            and bps.get('test_data_used') is False and bps.get('economic_edge_qualified') is False
            and bps['inputs']['baseline_oof'] == plan['inputs']['baseline'] and bps['inputs']['candidate_oof'] == plan['inputs']['bps']
            and bps['initial_same256_logit_bit_parity'] is True, 'BPS_LINEAGE')
        require(exposure.get('status') == 'COMPLETE_REUSED_DEVELOPMENT_NO_EDGE_CLAIM' and exposure.get('test_data_used') is False
            and exposure.get('model_forwards') == exposure.get('optimizer_steps') == 0
            and exposure.get('plan') == plan['inputs']['exposure_plan']
            and exposure_plan['inputs']['one'] == plan['inputs']['baseline']
            and exposure_plan['inputs']['three'] == plan['inputs']['three'], 'EXPOSURE_LINEAGE')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir']); require(out.parent == ROOT and out.name == 'DIRECTION_CORRECTED_FACTORS_20261028_001'
        and ROOT.resolve(strict=True) == ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE'); out.mkdir(exist_ok=False)
    def publish(path, value): publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref, 'source_head': args.source_head, 'execution': proof, 'lock': lock})
    payloads = [plan['inputs'][k] for k in ('baseline', 'bps', 'three', 'corrected')]
    metadata_paths = [plan_ref['path'], policy_ref['path'], plan['design']['path']] + [plan['inputs'][k]['path']
        for k in ('corrected_result', 'bps_result', 'exposure_plan', 'exposure_result')]
    sys.addaudithook(candle.make_guard(out, [x['path'] for x in payloads], metadata_paths,
        source_paths=[x['path'] for x in plan['sources'].values()]))
    started = time.monotonic(); previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(*unused): raise RuntimeError('CORRECTED_BPS_EXPOSURE_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); stage = out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for value in payloads: candle.regular(value['path']); binding.require_binding(value, label='payload')
            arms = {}
            for name, key in (('one', 'baseline'), ('bps', 'bps'), ('three', 'three')):
                with np.load(plan['inputs'][key]['path'], allow_pickle=False) as a: arms[name] = {k: a[k].copy() for k in a.files}
            with np.load(plan['inputs']['corrected']['path'], allow_pickle=False) as a:
                target = {key: a[key].copy() for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')}
            sign, changed = validate_join(arms, target)
            require(len(arms['one']['parent_rows']) == 17288 and changed == [4, 5, 3], 'EXPECTED_CORRECTED_CHANGES')
            for j, horizon in enumerate(HORIZONS):
                ok = arms['one']['observed'][:, j] & (arms['one']['sign'][:, j] != 0) & (arms['one']['terminal_time_ns'][:, j] < CUTOFF)
                labels = np.ascontiguousarray((arms['one']['sign'][ok, j] > 0).astype(np.int8))
                original_bps = next(v for v in bps['points']['later_development'] if v['horizon_minutes'] == horizon)
                original_exposure = next(v for v in exposure['panels'] if v['horizon_minutes'] == horizon)
                for arm, saved, source in (('one', 'baseline', original_bps), ('bps', 'candidate', original_bps), ('three', 'three', original_exposure)):
                    actual = points(labels, np.ascontiguousarray(arms[arm]['logit_up'][ok, j], dtype=np.float64))['auc']
                    expected = source['direction'][saved]['auc'] if arm != 'three' else source['arms'][saved]['auc']
                    require(abs(actual - expected) < 1e-12, 'ORIGINAL_AUC_REPLAY:' + arm + ':' + str(horizon))
            metrics = evaluate(arms, sign)
            payload_snapshot = binding.capture_verified_file_snapshot()
        payload_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock
            and not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'FINAL_SOURCE_LOCK')
        result = {'schema_version': 'gx1_corrected_bps_exposure_result_v1', 'status': 'COMPLETE', 'plan': plan_ref,
            'source_head': args.source_head, 'inputs': plan['inputs'], 'fold': bps['fold'],
            'original_saved_auc_reproduced': True, 'shared_original_baseline_and_contexts': True,
            'corrected_nonzero_to_zero_counts_1_5_15': changed, 'metrics': metrics,
            'limits': 'Models were fitted with original float targets; this only rescores fixed predictions. One reused development fold, one seed, three trained horizons and unproven calendar-block exchangeability. No corrected-target retraining, independent generalization or economic edge.',
            'new_model_forwards': 0, 'optimizer_steps': 0, 'checkpoint_reads': 0, 'test_data_used': False, 'economic_edge_qualified': False}
        publish(stage/'RESULT.json', result); publish(stage/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'source_unchanged': True, 'inputs_unchanged': True, 'elapsed_seconds': time.monotonic()-started, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out/'OUTPUT')
        publish(out/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref, 'result_path': str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False}); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__': raise SystemExit(main())
