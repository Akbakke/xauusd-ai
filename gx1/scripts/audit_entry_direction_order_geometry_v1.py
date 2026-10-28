"""One bounded TRAIN-only comparison of direction-loss batch composition.

No model, predictions, optimizer, fitting, control, TEST, or trading. The
original target archive is sliced to the declared chronological TRAIN prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'direction_order_batch_geometry_20261028_001'
HORIZONS = (1, 5, 15)
MINUTE_NS = 60_000_000_000


def require(condition, reason):
    if not condition:
        raise RuntimeError('ORDER_GEOMETRY_' + reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def typed_order_digest(order):
    order = np.ascontiguousarray(order)
    obj = {'__ndarray__': True, 'dtype': order.dtype.str, 'shape': list(order.shape),
           'sha256': digest(order.tobytes())}
    return digest(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode('ascii'))


def _summarize_order(order, eligible, signs, batch_size):
    n = len(order)
    require(np.array_equal(np.sort(order), np.arange(n)), 'ORDER_PERMUTATION')
    weight = np.zeros(3, np.float64)
    positive_weight = np.zeros(3, np.float64)
    populations = {'up_only': 0, 'down_only': 0, 'both': 0, 'no_eligible_five_minute': 0}
    denominators, last, steps = [], None, 0
    for left in range(0, n, batch_size):
        at = order[left:left + batch_size]
        m = eligible[at]
        y = signs[at]
        per_horizon = m.sum(axis=0)
        up = (m & (y == 1)).sum(axis=0)
        down = (m & (y == -1)).sum(axis=0)
        denominator = int(per_horizon.sum())
        denominators.append(denominator)
        if denominator:
            steps += 1
            weight += per_horizon / denominator
            positive_weight += up / denominator
        key = ('both' if up[1] and down[1] else 'up_only' if up[1]
               else 'down_only' if down[1] else 'no_eligible_five_minute')
        populations[key] += 1
        last = {'parents': len(at), 'eligible_cells': denominator,
                'eligible_by_horizon': per_horizon.tolist(),
                'up_by_horizon': up.tolist(), 'down_by_horizon': down.tolist()}
    # Recompute all batch denominators and five-minute counts with array reductions.
    starts = np.arange(0, n, batch_size)
    m = eligible[order]
    y = signs[order]
    totals = np.add.reduceat(m.sum(axis=1), starts)
    up5 = np.add.reduceat((m[:, 1] & (y[:, 1] == 1)).astype(np.int64), starts)
    down5 = np.add.reduceat((m[:, 1] & (y[:, 1] == -1)).astype(np.int64), starts)
    require(np.array_equal(totals, denominators), 'INDEPENDENT_DENOMINATORS')
    independently = {'up_only': int(np.sum((up5 > 0) & (down5 == 0))),
                     'down_only': int(np.sum((down5 > 0) & (up5 == 0))),
                     'both': int(np.sum((up5 > 0) & (down5 > 0))),
                     'no_eligible_five_minute': int(np.sum((up5 + down5) == 0))}
    require(independently == populations and steps == int(np.count_nonzero(totals)), 'INDEPENDENT_BATCH_COUNTS')
    require(steps > 0 and np.isclose(weight.sum(), steps, atol=1e-10, rtol=0), 'LOSS_COEFFICIENT_MASS')
    return {'parents': n, 'batches': len(denominators), 'nonempty_update_batches': steps,
            'empty_update_batches': len(denominators) - steps,
            'eligible_cells_min_max_per_batch': [int(totals.min()), int(totals.max())],
            'five_minute_batch_classes': populations, 'last_batch': last,
            'mean_batch_loss_coefficient_mass_by_horizon': (weight / steps).tolist(),
            'mean_batch_loss_coefficient_mass_up_by_horizon': (positive_weight / steps).tolist(),
            'scope': 'Exact eligible-cell loss coefficients, not gradients or observed model performance.'}


def compare_orders(parents, decisions, terminals, observed, signs, *, cutoff_ns, batch_size, seed):
    n = len(parents)
    require(n > 0 and parents.dtype == decisions.dtype == terminals.dtype == np.int64
            and parents.shape == decisions.shape == (n,) and terminals.shape == (n, 3)
            and observed.dtype == np.bool_ and observed.shape == (n, 3)
            and signs.dtype == np.int8 and signs.shape == (n, 3), 'TARGET_SHAPES')
    require(type(cutoff_ns) is int and type(batch_size) is int and batch_size > 0
            and type(seed) is int and np.all(np.diff(parents) > 0)
            and np.all(np.diff(decisions) > 0) and np.all(decisions < cutoff_ns), 'PREFIX')
    require(np.array_equal(terminals, decisions[:, None] + np.asarray(HORIZONS, np.int64)[None, :] * MINUTE_NS)
            and np.isin(signs, (-1, 0, 1)).all() and np.all(signs[~observed] == 0), 'TARGET_CLOCK_DOMAIN')
    eligible = observed & (signs != 0) & (terminals < cutoff_ns)
    chrono = np.arange(n, dtype=np.int64)
    mixed = np.random.RandomState(seed).permutation(n).astype(np.int64)
    physical_mixed = np.ascontiguousarray(parents[mixed])
    counts = eligible.sum(axis=0)
    up = (eligible & (signs == 1)).sum(axis=0)
    return {'parent_order_sha256': typed_order_digest(physical_mixed),
            'eligible_by_horizon': counts.tolist(), 'up_by_horizon': up.tolist(),
            'mature_zero_by_horizon': (observed & (signs == 0) & (terminals < cutoff_ns)).sum(axis=0).tolist(),
            'immature_by_horizon': (terminals >= cutoff_ns).sum(axis=0).tolist(),
            'chronological': _summarize_order(chrono, eligible, signs, batch_size),
            'fixed_mixed': _summarize_order(mixed, eligible, signs, batch_size)}


def _same_stat(a, b):
    return all(getattr(a, key) == getattr(b, key) for key in
               ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def read_train_prefix(ref, *, expected_members, n, cutoff_ns):
    path = Path(ref['path'])
    require(path.is_absolute() and not path.is_symlink() and path.resolve(strict=True) == path, 'TARGET_PATH')
    with path.open('rb') as f:
        before = os.fstat(f.fileno())
        sha = hashlib.sha256()
        for part in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(part)
        require(sha.hexdigest() == ref['sha256'], 'TARGET_HASH')
        f.seek(0)
        with np.load(f, allow_pickle=False) as archive:
            require(archive.files == expected_members, 'TARGET_MEMBERS')
            parents = archive['parent_rows'][:n].copy()
            decisions_all = archive['decision_ns'][:n + 1].copy()
            terminals = archive['terminal_time_ns'][:n].copy()
            observed = archive['observed'][:n].copy()
            signs = archive['sign'][:n].copy()
        require(len(decisions_all) == n + 1 and decisions_all[-1] >= cutoff_ns, 'NEXT_PARENT_CUTOFF')
        require(_same_stat(before, os.fstat(f.fileno())) and _same_stat(before, path.stat()), 'TARGET_STABLE_IDENTITY')
    return parents, decisions_all[:n], terminals, observed, signs


def _write_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return digest(data)


def run(plan_path, expected_plan_sha, expected_head):
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
    guard = require_capped_cpu_audit_execution()
    plan_path = Path(plan_path)
    plan_bytes = plan_path.read_bytes()
    require(digest(plan_bytes) == expected_plan_sha, 'PLAN_SHA')
    plan = json.loads(plan_bytes)
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_bytes())
    gate = policy.get(STAGE, {})
    require(gate.get('status') == 'ADMITTED_ONE_CPU_AUDIT' and gate.get('execution_authorized') is True
            and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
            and gate.get('max_invocations') == 1 and gate.get('plan') == {'path': str(plan_path), 'sha256': expected_plan_sha}
            and policy.get('training_enabled') is False and policy.get('native_training_enabled') is False, 'POLICY')
    require(plan.get('stage') == STAGE and plan.get('test_data_used') is False
            and plan.get('model_forwards') == plan.get('optimizer_steps') == 0, 'PLAN_SCOPE')
    source = Path(__file__).resolve(strict=True)
    require(str(source) == plan['source']['path'] and digest(source.read_bytes()) == plan['source']['sha256'], 'SOURCE_SHA')
    branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain=v1', '--untracked-files=all'], text=True)
    require(branch == 'work/gx1-current' and head == expected_head and not dirty, 'CLEAN_SOURCE_IDENTITY')
    for name in ('original_recipe', 'original_result', 'original_input_manifest'):
        ref = plan[name]
        require(digest(Path(ref['path']).read_bytes()) == ref['sha256'], 'BOUND_' + name.upper())
    recipe = json.loads(Path(plan['original_recipe']['path']).read_bytes())
    manifest = json.loads(Path(plan['original_input_manifest']['path']).read_bytes())
    original = json.loads(Path(plan['original_result']['path']).read_bytes())
    require(recipe['input_manifest'] == plan['original_input_manifest'] and manifest['targets'] == plan['target']
            and manifest['archive_members'] == plan['members'] and recipe['fold'] == plan['fold']
            and recipe['numerical_recipe']['seed'] == plan['seed']
            and recipe['numerical_recipe']['batch_size'] == plan['batch_size']
            and original['parent_order_sha256'] == plan['expected_order_sha256'], 'ORIGINAL_BINDINGS')
    out = Path(plan['output_dir'])
    require(out.parent == plan_path.parent and not out.exists() and not out.is_symlink(), 'UNUSED_OUTPUT')
    out.mkdir()
    _write_new(out / 'CLAIM.json', {'stage': STAGE, 'plan_sha256': expected_plan_sha, 'source_head': head})
    try:
        arrays = read_train_prefix(plan['target'], expected_members=plan['members'],
                                   n=plan['fold']['prefix_rows'], cutoff_ns=plan['fold']['train_cutoff_ns'])
        result = compare_orders(*arrays, cutoff_ns=plan['fold']['train_cutoff_ns'],
                                batch_size=plan['batch_size'], seed=plan['seed'])
        require(result['parent_order_sha256'] == plan['expected_order_sha256'], 'ORIGINAL_ORDER_SHA')
        require(result['fixed_mixed']['nonempty_update_batches'] == original['optimizer_updates'], 'ORIGINAL_UPDATE_COUNT')
        result.update({'status': 'COMPLETE_TRAIN_PREFIX_LOSS_GEOMETRY_ONLY', 'stage': STAGE,
                       'source_head': head, 'plan_sha256': expected_plan_sha,
                       'target': plan['target'], 'guard': guard, 'model_forwards': 0,
                       'optimizer_steps': 0, 'test_data_used': False, 'predictive_edge_qualified': False})
        sha = _write_new(out / 'RESULT.json', result)
        _write_new(out / 'TERMINAL.json', {'status': 'COMPLETE', 'result_sha256': sha, 'test_data_used': False})
        print('COMPLETE', sha)
    except BaseException as exc:
        _write_new(out / 'TERMINAL.json', {'status': 'FAILED', 'error': type(exc).__name__ + ':' + str(exc),
                                            'test_data_used': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-head', required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256, args.source_head)


if __name__ == '__main__':
    main()
