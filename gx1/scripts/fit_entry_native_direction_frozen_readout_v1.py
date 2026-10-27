"""One fixed terminal-hidden decoder comparison; no native model or raw-data route."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import types

REPO = Path('/home/andre2/src/GX1_CURRENT')
PROGRAM = Path('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2')
META_SHA = 'a648c47880b6b195d1213741a0826d34c7e475bd75193313a4c26f7723798c7a'
HORIZONS = (1, 5, 15)
MINUTE = 60_000_000_000
DAY = 86_400_000_000_000
PREFIX_CUTOFF = 1735689600000000000
LATER_CUTOFF = 1743465600000000000
PREFIX_ROWS, LATER_ROWS = 47602, 17288
SCHEMES = ('UTC_WEEK_MONDAY', 'UTC_DAY')
HEAD_KEYS = {'head_weight', 'head_bias'}
np = None


def require(ok, reason):
    if not ok:
        raise RuntimeError('TERMINAL_READOUT_' + reason)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def typed_hash_owner(meta, ref):
    # Reuse only the original pure digest function, without importing Exit code.
    source = meta.read_ref(ref).decode()
    tree = ast.parse(source)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_structured_sha256')
    namespace = {'np': np, 'hashlib': hashlib, 'json': json, 'Any': object}
    from collections.abc import Mapping
    namespace['Mapping'] = Mapping
    exec(compile(ast.Module(body=[node], type_ignores=[]), ref['path'], 'exec'), namespace)
    return namespace['_structured_sha256']


def validate_piece(a, contexts, *, parent_start, parent_end, presentation_start,
                   presentation_end, lower_ns, cutoff_ns):
    n = presentation_end - presentation_start
    schema = {
        'parent_rows': ('int64', (n,)), 'bar_start_ns': ('int64', (n,)),
        'decision_ns': ('int64', (n,)), 'terminal_time_ns': ('int64', (n, 3)),
        'observed': ('bool', (n, 3)), 'sign': ('int8', (n, 3)),
        'logit_up': ('float32', (n, 3)), 'hidden': ('float32', (n, 128)),
        'presentation_index': ('int64', (n,)), 'head_weight': ('float32', (3, 128)),
        'head_bias': ('float32', (3,)),
        **{'context__'+c['name']: ('int8', (n,)) for c in contexts},
    }
    require(set(a) == set(schema), 'MEMBER_SET')
    for name, (dtype, shape) in schema.items():
        require(a[name].dtype == np.dtype(dtype) and a[name].shape == shape
                and a[name].flags.c_contiguous and np.isfinite(a[name]).all(), 'ARRAY:' + name)
    require((np.diff(a['parent_rows']) > 0).all()
            and ((a['parent_rows'] >= parent_start) & (a['parent_rows'] < parent_end)).all(), 'PARENT_RANGE_ORDER')
    require(np.array_equal(np.sort(a['presentation_index']),
                           np.arange(presentation_start, presentation_end, dtype=np.int64)), 'PRESENTATION_PERMUTATION')
    d = a['decision_ns']
    require((np.diff(d) > 0).all() and ((d >= lower_ns) & (d < cutoff_ns)).all()
            and np.array_equal(d, a['bar_start_ns'] + 5*MINUTE), 'DECISION_CLOCK')
    require(np.array_equal(a['terminal_time_ns'], d[:, None] + np.array(HORIZONS, np.int64)*MINUTE), 'TARGET_CLOCK')
    require(np.isin(a['sign'], [-1, 0, 1]).all() and (a['sign'][~a['observed']] == 0).all(), 'SIGN_MISSING')
    for c in contexts:
        require(np.isin(a['context__'+c['name']], [v['code'] for v in c['categories']]).all(), 'CONTEXT_DOMAIN')


def assemble_prefix(pieces, contexts, *, rows, cutoff_ns, lower_ns, order_sha, typed_hash):
    start = 0
    for a in pieces:
        end = start + len(a['parent_rows'])
        validate_piece(a, contexts, parent_start=0, parent_end=rows,
                       presentation_start=start, presentation_end=end, lower_ns=lower_ns, cutoff_ns=cutoff_ns)
        for k in HEAD_KEYS:
            require(np.array_equal(a[k], pieces[0][k]), 'SAME_HEAD')
        start = end
    require(start == rows, 'FULL_PREFIX_ROWS')
    merged = {k: np.concatenate([a[k] for a in pieces]) for k in pieces[0] if k not in HEAD_KEYS}
    order = np.argsort(merged['parent_rows'], kind='stable')
    merged = {k: np.ascontiguousarray(v[order]) for k, v in merged.items()}
    merged.update({k: pieces[0][k] for k in HEAD_KEYS})
    require(np.array_equal(merged['parent_rows'], np.arange(rows, dtype=np.int64)), 'EXACT_PREFIX_PARENTS')
    presentation = merged['parent_rows'][np.argsort(merged['presentation_index'], kind='stable')]
    require(typed_hash(presentation) == order_sha, 'FULL_PRESENTATION_HASH')
    validate_piece(merged, contexts, parent_start=0, parent_end=rows, presentation_start=0,
                   presentation_end=rows, lower_ns=lower_ns, cutoff_ns=cutoff_ns)
    return merged


def head_reconstruction(a):
    x, w, b = (a[k].astype(np.float64) for k in ('hidden', 'head_weight', 'head_bias'))
    exact = x @ w.T + b
    # Standard conservative FP32 dot/add bound; not a claim of GPU bit parity.
    u = np.finfo(np.float32).eps / 2
    gamma = (2*128+2)*u / (1-(2*128+2)*u)
    bound = gamma * (np.abs(x) @ np.abs(w).T + np.abs(b)) + np.finfo(np.float32).tiny
    error = np.abs(exact - a['logit_up'].astype(np.float64))
    require((error <= bound).all(), 'HEAD_RECONSTRUCTION_BOUND')
    return {'max_absolute_error': float(error.max()), 'max_rounding_bound': float(bound.max()),
            'bit_parity_claimed': False}


def eligible(a, cutoff):
    return a['observed'] & (a['sign'] != 0) & (a['terminal_time_ns'] < cutoff)


def join_et(a, c, headers, *, source_parent_range=(29849, 118249)):
    for name, spec in headers.items():
        require(c[name].dtype == np.dtype(spec['dtype']) and list(c[name].shape) == spec['shape'], 'ET_HEADER:' + name)
    require(np.array_equal(c['parent_rows'], np.arange(*source_parent_range)), 'ET_FULL_PARENTS')
    indices = np.searchsorted(c['parent_rows'], a['parent_rows'])
    require((indices < len(c['parent_rows'])).all(), 'ET_JOIN_RANGE')
    for k in ('parent_rows', 'decision_ns', 'observed', 'sign', 'terminal_time_ns'):
        require(np.array_equal(a[k], c[k][indices]), 'ET_JOIN:' + k)
    require((c['fold_id'][indices] == 1).all()
            and (c['holdout_end_ns_exclusive'][indices] == LATER_CUTOFF).all()
            and np.array_equal(c['binary_mature'][indices], eligible(a, LATER_CUTOFF)), 'ET_MATURITY_FOLD')
    p = c['saved_ET_FULL_probability_up'][indices]
    require(((p > 0) & (p < 1)).all(), 'ET_FINITE_PROPER_PROBABILITY_NO_CLIPPING')
    require(np.array_equal((p > .5).astype(np.int8), c['saved_ET_FULL_direction_class'][indices]), 'ET_DIRECTION')
    return np.ascontiguousarray(p)


def score_panel(a, predicted, prior, cutoff, owner, io, *, et=None):
    mask = eligible(a, cutoff)
    reports, row_scores = [], []
    for j, h in enumerate(HORIZONS):
        keep = mask[:, j]
        y = np.ascontiguousarray((a['sign'][keep, j] == 1).astype(np.int8))
        require((y == 1).any() and (y == 0).any(), 'SCORE_BOTH_CLASSES')
        arms = {'fitted': predicted[keep, j], 'native': a['logit_up'][keep, j].astype(np.float64),
                'prior': np.full(len(y), prior[j], np.float64)}
        if et is not None:
            arms['ET252'] = np.log(et[keep, j]) - np.log1p(-et[keep, j])
        metrics, rows = {}, {}
        for arm, z in arms.items():
            z = np.ascontiguousarray(z, dtype=np.float64)
            m = owner.direction_probability_metrics(y, z)
            # ET rank uses its original stored probability, avoiding transformed-score ties.
            rank_score = np.ascontiguousarray(et[keep, j]) if arm == 'ET252' else z
            rank = owner.direction_rank_diagnostics(y, rank_score,
                score_kind='probability_up' if arm == 'ET252' else 'logit_up')
            rows[arm] = {'logit': z, 'logloss': m.pop('logloss_by_row'), 'brier': m.pop('brier_by_row')}
            metrics[arm] = {**m, 'rank': rank}
        reports.append({'horizon_minutes': h, 'parents': len(a['parent_rows']), 'eligible_rows': len(y),
            'eligible_coverage': len(y)/len(keep), 'eligible_up_fraction': float(y.mean()),
            'mature_zero_rows': int(np.sum(a['observed'][:, j] & (a['sign'][:, j] == 0)
                                           & (a['terminal_time_ns'][:, j] < cutoff))),
            'eligible_parent_sha256': io.digest(a['parent_rows'][keep], np), 'arms': metrics,
            'always_UP': {'accuracy': float(y.mean()), 'balanced_accuracy': .5, 'auc': .5,
                          'logloss': None, 'brier': None, 'role': 'hard classification reference; no probability assigned'}})
        row_scores.append({'y': y, 'decision': a['decision_ns'][keep], 'arms': rows})
    return reports, row_scores


def paired_intervals(row_scores, io, rank_owner):
    result = []
    for sid, scheme in enumerate(SCHEMES):
        for j, data in enumerate(row_scores):
            days = data['decision']//DAY
            labels = (days+3)//7 if sid == 0 else days
            blocks, inverse = np.unique(labels, return_inverse=True)
            k = len(blocks)
            require(k >= 8, 'EFFECTIVE_BLOCK_COUNT')
            weights = io.block_weights(k, replicates=4000, seed=np.random.SeedSequence([20260911, sid, 1]))
            auc = {}
            for arm in ('native', 'fitted'):
                rank = rank_owner.block_rank_pairs(data['y'], data['arms'][arm]['logit'], inverse)
                numerator, denominator = rank_owner.weighted_rank_pairs(rank, weights)
                require((denominator > 0).all(), 'ALL_BOOTSTRAP_REPLICATES_DEFINED')
                auc[arm] = numerator / (2*denominator)
                if arm == 'native':
                    original_denominator = denominator
                else:
                    require(np.array_equal(denominator, original_denominator), 'PAIRED_DENOMINATORS')
            def ci(values):
                require(np.isfinite(values).all() and len(values) == 4000, 'ALL_FINITE_REPLICATES')
                low, high = np.quantile(values, [.005, .995], method='linear')
                return {'low': float(low), 'high': float(high), 'replicates': 4000, 'alpha': .01}
            proper = {}
            row_count = np.bincount(inverse, minlength=k)
            for metric in ('logloss', 'brier'):
                delta = data['arms']['fitted'][metric] - data['arms']['native'][metric]
                sums = np.bincount(inverse, weights=delta, minlength=k)
                proper[metric] = ci((weights @ sums)/(weights @ row_count))
            result.append({'scheme': scheme, 'horizon_minutes': HORIZONS[j], 'blocks': k,
                'seed_sequence': [20260911, sid, 1], 'weights_sha256': io.digest(weights, np),
                'fitted_minus_native_auc_99': ci(auc['fitted']-auc['native']),
                'individual_auc_99_descriptive': {k: ci(v) for k, v in auc.items()},
                'fitted_minus_native_proper_99_descriptive': proper})
    return result


def operation_watch(io, roles, maxima, fdpaths, phase, options):
    counts = {k+s: 0 for k in maxima for s in ('_calls', '_returns')}
    fit_codes = {code for code, role in roles.items() if role in ('fit_wrapper', 'coordinate_fit')}
    def watch(frame, event, result):
        code = frame.f_code
        if code not in fit_codes:
            io.observe_work(frame, event, result)
        role = roles.get(code)
        if role is None or event not in ('call', 'return'):
            return
        if event == 'call':
            counts[role+'_calls'] += 1
            require(counts[role+'_calls'] <= maxima[role], 'CALL_CAP:' + role)
            if role in ('fit_wrapper', 'coordinate_fit', 'minimize'):
                require(phase[0] == 'prefix_fit', 'PREFIX_FIT_PHASE')
            if role == 'minimize':
                require(frame.f_locals['method'] == 'L-BFGS-B' and frame.f_locals['jac'] is True
                        and frame.f_locals['options'] == options, 'FIXED_SOLVER')
            if role == 'payload_load':
                require(frame.f_locals.get('file') in fdpaths
                        and frame.f_locals.get('allow_pickle') is False, 'PINNED_NPZ_LOAD')
        elif result is not None:
            counts[role+'_returns'] += 1
    return watch, counts


def synthetic_checks(typed_hash, rank_owner, owner, io, readout):
    contexts = [{'name': 'fixture', 'categories': [{'code': 0}]}]
    n = 24
    d = 1714521600000000000 + 5*MINUTE + np.arange(n, dtype=np.int64)*DAY
    cutoff = int(d[-1] + DAY)
    base = {'parent_rows': np.arange(n, dtype=np.int64), 'bar_start_ns': d-5*MINUTE, 'decision_ns': d,
        'terminal_time_ns': d[:, None]+np.array(HORIZONS, np.int64)*MINUTE,
        'observed': np.ones((n, 3), bool), 'sign': np.tile(np.where(np.arange(n)%2, 1, -1).astype(np.int8)[:, None], (1, 3)),
        'logit_up': np.zeros((n, 3), np.float32), 'hidden': np.zeros((n, 128), np.float32),
        'presentation_index': np.empty(n, np.int64), 'head_weight': np.zeros((3, 128), np.float32),
        'head_bias': np.zeros(3, np.float32), 'context__fixture': np.zeros(n, np.int8)}
    order = np.array([17, 2, 23, 6, 13, 0, 21, 8, 11, 4, 19, 10, 15, 22, 1, 12, 9, 16, 5, 20, 3, 18, 7, 14], np.int64)
    base['presentation_index'][order] = np.arange(n)
    def pieces():
        return [{k: v if k in HEAD_KEYS else np.ascontiguousarray(v[np.sort(order[s:s+8])])
                 for k, v in base.items()} for s in (0, 8, 16)]
    def assembly(parts, expected=typed_hash(order)):
        return assemble_prefix(parts, contexts, rows=n, cutoff_ns=cutoff, lower_ns=int(d[0]),
                               order_sha=expected, typed_hash=typed_hash)
    a = assembly(pieces())
    require(all(np.array_equal(a[k], base[k]) for k in base), 'SYNTHETIC_ASSEMBLY')
    cases = ['shuffled_three_piece_assembly_and_typed_hash']
    def rejects(name, fn, reason):
        try:
            fn()
        except RuntimeError as exc:
            require(reason in str(exc), 'SYNTHETIC_WRONG_REJECTION:' + name)
            cases.append(name)
        else:
            raise RuntimeError('SYNTHETIC_EXPECTED_REJECTION:' + name)
    for name, key, mutation, reason in [
        ('duplicate_parent', 'parent_rows', lambda v: v.__setitem__(1, v[0]), 'PARENT_RANGE_ORDER'),
        ('duplicate_presentation', 'presentation_index', lambda v: v.__setitem__(1, v[0]), 'PRESENTATION_PERMUTATION'),
        ('decision_shift', 'decision_ns', lambda v: v.__setitem__(0, v[0]+1), 'DECISION_CLOCK'),
        ('terminal_shift', 'terminal_time_ns', lambda v: v.__setitem__((0, 1), v[0, 1]+MINUTE), 'TARGET_CLOCK'),
        ('missing_nonzero', 'observed', lambda v: v.__setitem__((0, 0), False), 'SIGN_MISSING'),
        ('head_changed', 'head_bias', lambda v: v.__setitem__(0, 1), 'SAME_HEAD'),
    ]:
        p = pieces()
        p[1][key] = p[1][key].copy()
        mutation(p[1][key])
        rejects(name, lambda p=p: assembly(p), reason)
    rejects('wrong_full_order_hash', lambda: assembly(pieces(), '0'*64), 'FULL_PRESENTATION_HASH')
    rejects('missing_partition', lambda: assembly(pieces()[:2]), 'FULL_PREFIX_ROWS')
    require(head_reconstruction(a)['max_absolute_error'] == 0, 'SYNTHETIC_HEAD')
    damaged = dict(a)
    damaged['logit_up'] = a['logit_up'].copy()
    damaged['logit_up'][0, 0] = 1
    rejects('head_prediction_mismatch', lambda: head_reconstruction(damaged), 'HEAD_RECONSTRUCTION_BOUND')
    m = eligible(a, int(a['terminal_time_ns'][0, 1]))
    require(m[0, 0] and not m[0, 1] and not m[0, 2], 'SYNTHETIC_STRICT_MATURITY')
    cases.append('strict_horizon_maturity')
    et = {k: a[k].copy() for k in ('parent_rows', 'decision_ns', 'observed', 'sign', 'terminal_time_ns')}
    et.update(fold_id=np.ones(n, np.int8), holdout_end_ns_exclusive=np.full(n, LATER_CUTOFF, np.int64),
              binary_mature=eligible(a, LATER_CUTOFF), saved_ET_FULL_probability_up=np.full((n, 3), .6),
              saved_ET_FULL_direction_class=np.ones((n, 3), np.int8))
    headers = {k: {'dtype': str(v.dtype), 'shape': list(v.shape)} for k, v in et.items()}
    require(np.array_equal(join_et(a, et, headers, source_parent_range=(0, n)),
                           et['saved_ET_FULL_probability_up']), 'SYNTHETIC_ET_JOIN')
    cases.append('same_parent_clock_label_mask_ET_join')
    et['sign'][0, 0] *= -1
    rejects('ET_label_mismatch', lambda: join_et(a, et, headers, source_parent_range=(0, n)), 'ET_JOIN:sign')
    y = np.array([0, 1, 0, 1, 0, 1], np.int8)
    z = np.array([0., 0., -.2, .3, .4, -.1])
    ids = np.repeat(np.arange(3), 2)
    weights = np.array([[1., 1., 1.], [2., 0., 1.]])
    rr = rank_owner.block_rank_pairs(y, z, ids)
    num, den = rank_owner.weighted_rank_pairs(rr, weights)
    for i, w in enumerate(weights):
        selected = np.concatenate([np.tile(np.flatnonzero(ids == b), int(w[b])) for b in range(3)])
        direct = owner.direction_rank_diagnostics(y[selected], z[selected], score_kind='logit_up')['auc']
        require(num[i]/(2*den[i]) == direct, 'SYNTHETIC_DUPLICATED_BLOCKS')
    cases.append('weighted_auc_matches_explicit_duplicated_rows_with_ties')
    fixture_y = np.tile(np.array([0, 1], np.int8), 12)
    fixture_z = (2*fixture_y-1).astype(np.float64)
    fixture_d = d[0] + np.repeat(np.arange(12), 2)*7*DAY + np.tile([0, 5*MINUTE], 12)
    fixture_rows = {}
    for arm, z0 in [('native', np.zeros(24)), ('fitted', fixture_z)]:
        m0 = owner.direction_probability_metrics(fixture_y, z0)
        fixture_rows[arm] = {'logit': z0, 'logloss': m0['logloss_by_row'], 'brier': m0['brier_by_row']}
    intervals = paired_intervals([{'y': fixture_y, 'decision': fixture_d, 'arms': fixture_rows}]*3, io, rank_owner)
    require(len(intervals) == 6 and all(x['blocks'] == 12
            and x['fitted_minus_native_auc_99']['low'] == .5
            and x['fitted_minus_native_auc_99']['high'] == .5 for x in intervals), 'SYNTHETIC_PAIRED_DRIVER')
    cases.append('paired_week_day_driver_known_half_auc_difference')
    # Separate observer fixture verifies exact-code allowance; no genuine fit invoked.
    roles = {readout.fit_frozen_direction_readout.__code__: 'fit_wrapper',
             readout.fit_prefix_coordinates.__code__: 'coordinate_fit'}
    w, counts = operation_watch(io, roles, {'fit_wrapper': 1, 'coordinate_fit': 1}, set(), ['prefix_fit'], {})
    w(types.SimpleNamespace(f_code=readout.fit_frozen_direction_readout.__code__, f_locals={}), 'call', None)
    w(types.SimpleNamespace(f_code=readout.fit_frozen_direction_readout.__code__, f_locals={}), 'return', {})
    require(counts['fit_wrapper_calls'] == counts['fit_wrapper_returns'] == 1, 'SYNTHETIC_EXACT_FIT_OBSERVER')
    rejects('second_fit_wrapper_rejected', lambda: w(types.SimpleNamespace(
        f_code=readout.fit_frozen_direction_readout.__code__, f_locals={}), 'call', None), 'CALL_CAP')
    cases.append('exact_admitted_code_observed_without_fit')
    denial = {k: 0 for k in io.FORBIDDEN}
    isolated = types.FunctionType(io.observe_work.__code__, {**io.__dict__, 'FORBIDDEN': denial})
    unapproved = compile('def fit_not_admitted(): pass', str(REPO/'gx1/contracts/not_admitted.py'), 'exec')
    namespace = {}
    exec(unapproved, namespace)
    try:
        isolated(types.SimpleNamespace(f_code=namespace['fit_not_admitted'].__code__, f_locals={}), 'call', None)
    except RuntimeError as exc:
        require('unapproved_fits' in str(exc) and denial['unapproved_fits'] == 1, 'SYNTHETIC_FIT_DENIAL')
        cases.append('unrelated_repository_fit_rejected')
    else:
        raise RuntimeError('SYNTHETIC_MISSING_FIT_DENIAL')
    return {'cases': cases, 'case_count': len(cases), 'genuine_payloads': 0, 'genuine_fits': 0}


def main():
    global np
    sys.dont_write_bytecode = True
    hp = PROGRAM/'inspect_direction_gc_parquet_metadata.py'
    raw = hp.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == META_SHA, 'METADATA_HELPER')
    meta = types.ModuleType('terminal_readout_metadata')
    meta.__file__ = str(hp)
    exec(compile(raw, str(hp), 'exec'), meta.__dict__)
    pref = {'path': sys.argv[1], 'sha256': sys.argv[2]}
    plan = meta.read_json(pref)
    polref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': sys.argv[3]}
    policy, head = meta.read_json(polref), sys.argv[4]
    mode = plan['mode']
    require(plan['schema_version'] == 'gx1_terminal_readout_execution_v1' and mode in ('synthetic', 'actual'), 'PLAN')
    st = policy['direction_terminal_readout']
    require(st['status'] == 'ADMITTED' and st['plan'] == pref and st['mode'] == mode
            and st['execution_authorized'] is True and st['max_invocations'] == 1
            and st['claimed_invocations'] == st['completed_invocations'] == 0
            and st['relaunch_allowed'] is False, 'ONE_SHOT_AUTHORITY')
    require(st['saved_payload_reads_authorized'] is (mode == 'actual')
            and st['genuine_readout_fits_authorized'] is (mode == 'actual'), 'FIT_READ_AUTHORITY')
    require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled',
        'native_launch_authorized', 'full_epoch_training_allowed', 'full_val_allowed')), 'NATIVE_GATES')
    require(meta.git_head() == head and plan['wall_seconds'] == (120 if mode == 'synthetic' else 600), 'HEAD_WALL')
    for ref in plan['sources'].values():
        meta.read_ref(ref)
    require(plan['sources']['worker']['path'] == str(Path(__file__).resolve()), 'WORKER_PATH')
    admission = meta.read_json(plan['source_admission'])
    require(str(Path(__file__).relative_to(REPO)) in admission['allowed_repo_files'], 'SOURCE_ADMISSION')
    cap = meta.source_module('terminal_readout_cap', plan['sources']['cap'])
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'LOCK_SCOPE')
    io = meta.source_module('terminal_readout_io', plan['sources']['io'])
    guard = meta.source_module('terminal_readout_guard', plan['sources']['read_guard'])
    meta.PUBLISH = meta.source_module('terminal_readout_pub', plan['sources']['publication'])
    out = Path(plan['out_dir'])
    require(out.parent == PROGRAM and out.parent.resolve(strict=True) == PROGRAM, 'OUTPUT_ROOT')
    out.mkdir()
    claim = meta.publish(out, 'INVOCATION_CLAIM.json', {'plan': pref, 'policy': polref,
                         'source_head': head, 'mode': mode, 'execution': proof, 'lock': lock})
    paths = [x['payload']['path'] for x in plan['units']] + ([plan['et']['payload']['path']] if mode == 'actual' else [])
    require(len(set(paths)) == (5 if mode == 'actual' else 0), 'EXACT_PAYLOAD_SCOPE')
    sys.addaudithook(guard.make_guard(out, paths))
    def expired(*args):
        raise RuntimeError('TERMINAL_READOUT_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(plan['wall_seconds'])
    start, before, previous = time.monotonic(), io.io_state(), sys.getprofile()
    pins, fdpaths, phase, counts = {}, set(), ['setup'], {}
    try:
        sys.path.insert(0, str(REPO))
        import numpy as numpy
        np = numpy
        io.np = np
        from gx1.contracts import entry_native_direction_frozen_readout_v1 as readout
        from gx1.contracts import entry_direction_probability_reference_v1 as owner
        from gx1.contracts import entry_direction_block_rank_v1 as rank_owner
        from scipy.optimize import minimize
        typed_hash = typed_hash_owner(meta, plan['sources']['typed_hash'])
        roles = {np.load.__code__: 'payload_load', readout.fit_frozen_direction_readout.__code__: 'fit_wrapper',
                 readout.fit_prefix_coordinates.__code__: 'coordinate_fit', minimize.__code__: 'minimize',
                 readout.predict_frozen_direction_readout.__code__: 'predict',
                 rank_owner.block_rank_pairs.__code__: 'block_rank',
                 rank_owner.weighted_rank_pairs.__code__: 'weighted_rank'}
        maxima = ({'payload_load': 5, 'fit_wrapper': 1, 'coordinate_fit': 1, 'minimize': 3,
                   'predict': 2, 'block_rank': 12, 'weighted_rank': 12} if mode == 'actual' else
                  {'payload_load': 0, 'fit_wrapper': 0, 'coordinate_fit': 0, 'minimize': 0,
                   'predict': 0, 'block_rank': 13, 'weighted_rank': 13})
        watch, counts = operation_watch(io, roles, maxima, fdpaths, phase, readout.SOLVER_OPTIONS)
        sys.setprofile(watch)
        if mode == 'synthetic':
            require(plan['units'] == [] and plan['et'] is None, 'SYNTHETIC_NO_INPUTS')
            result = synthetic_checks(typed_hash, rank_owner, owner, io, readout)
        else:
            qualification = meta.read_json(plan['source_qualification'])
            require(qualification['status'] == 'PASS_SYNTHETIC_ONLY'
                    and qualification['worker'] == plan['sources']['worker'], 'QUALIFIED_EXACT_WORKER')
            protocol = meta.read_json(plan['scientific_subquestion'])
            require(protocol['scope']['prefix_rows'] == PREFIX_ROWS and protocol['scope']['later_rows'] == LATER_ROWS
                    and protocol['scope']['horizons_minutes'] == list(HORIZONS), 'SCIENTIFIC_SCOPE')
            old = meta.read_json(plan['original_oof_report'])
            contexts = old['context_map']['contexts']
            reports = []
            for item in plan['units']:
                r = meta.read_json(item['result'])
                v = meta.read_json(item['verified_terminal'])
                original = meta.read_json(v['exit_code_source'])
                review = meta.read_json(v['actual_root_review'])
                require(v['status'] == 'COMPLETE' and v['exit_code'] == 0
                        and original['trainer_guard_exit_code'] == original['progress_observer_exit_code'] == 0
                        and original['outcome'] == 'COMPLETE'
                        and v['result'] == item['result'] and review['result'] == item['result'], 'ORIGINAL_TERMINAL')
                require(r['decision'] == 'COMPLETE_FROZEN_READOUT_EXTRACTION'
                        and r['representations'] == {k: item['payload'][k] for k in ('path', 'sha256')}
                        and r['unit'] == item['unit'] and r['test_data_used'] is False, 'EXTRACTION_UNIT')
                reports.append(r)
            for k in ('state', 'feature_view_identity', 'fresh_identity_sha256', 'head_parameters_sha256', 'own_prefix_prior'):
                require(all(r[k] == reports[0][k] for r in reports), 'SAME_FROZEN_' + k)
            require(reports[3]['original_prediction_parity']['bit_identical'] is True
                    and reports[3]['original_prediction_parity']['rows'] == LATER_ROWS
                    and reports[3]['original_prediction_parity']['reference'] == old['predictions'], 'LATER_ORIGINAL_PARITY')
            def load(item, keys):
                ref = item['payload']
                pin = io.pin_artifact(ref)
                pins[ref['path']] = pin
                fdpaths.add('/proc/self/fd/'+str(pin['fd']))
                return io.read_archive(pin, set(keys), np)
            keys = {'parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed',
                    'sign', 'logit_up', 'hidden', 'presentation_index', *HEAD_KEYS,
                    *('context__'+c['name'] for c in contexts)}
            # Only prefix payloads are opened before fitting; later payload remains unopened.
            pieces = [load(item, keys) for item in plan['units'][:3]]
            prefix = assemble_prefix(pieces, contexts, rows=PREFIX_ROWS, cutoff_ns=PREFIX_CUTOFF,
                lower_ns=1714521600000000000, order_sha=protocol['scope']['original_prefix_presentation_sha256'],
                typed_hash=typed_hash)
            del pieces
            prefix_check = head_reconstruction(prefix)
            require(typed_hash({'weight': prefix['head_weight'], 'bias': prefix['head_bias']})
                    == reports[0]['head_parameters_sha256'], 'EXACT_HEAD_HASH')
            prior = np.asarray(reports[0]['own_prefix_prior']['logit_up'], dtype=np.float64)
            mask = eligible(prefix, PREFIX_CUTOFF)
            ups = np.sum(mask & (prefix['sign'] == 1), axis=0)
            downs = np.sum(mask & (prefix['sign'] == -1), axis=0)
            require(ups.tolist() == reports[0]['own_prefix_prior']['eligible_up']
                    and downs.tolist() == reports[0]['own_prefix_prior']['eligible_down']
                    and np.max(np.abs(np.log(ups.astype(float)/downs) - prior)) < 1e-15, 'PREFIX_PRIOR')
            phase[0] = 'prefix_fit'
            state = readout.fit_frozen_direction_readout(prefix['hidden'],
                **{k: prefix[k] for k in ('parent_rows', 'sign', 'observed', 'decision_ns', 'terminal_time_ns')},
                cutoff_ns=PREFIX_CUTOFF, original_weight=prefix['head_weight'], original_bias=prefix['head_bias'], split='train')
            phase[0] = 'fixed_scoring'
            state_ref = meta.publish(out, 'FITTED_PREFIX_STATE.json', plain(state))
            prefix_pred = readout.predict_frozen_direction_readout(prefix['hidden'], state)
            train_metrics, _ = score_panel(prefix, prefix_pred, prior, PREFIX_CUTOFF, owner, io)
            later = load(plan['units'][3], keys)
            validate_piece(later, contexts, parent_start=PREFIX_ROWS, parent_end=PREFIX_ROWS+LATER_ROWS,
                presentation_start=0, presentation_end=LATER_ROWS, lower_ns=PREFIX_CUTOFF, cutoff_ns=LATER_CUTOFF)
            require(np.array_equal(later['parent_rows'], np.arange(PREFIX_ROWS, PREFIX_ROWS+LATER_ROWS)), 'LATER_PARENTS')
            require(typed_hash(later['parent_rows'][np.argsort(later['presentation_index'], kind='stable')])
                    == reports[3]['partition']['full_presentation_parent_rows_sha256'], 'LATER_ORDER')
            require(all(np.array_equal(later[k], prefix[k]) for k in HEAD_KEYS), 'LATER_SAME_HEAD')
            later_check = head_reconstruction(later)
            et_report = meta.read_json(plan['et']['report'])
            et_terminal = meta.read_json(plan['et']['terminal'])
            require(et_report['OOF_predictions'] == plan['et']['payload'] and et_terminal['status'] == 'COMPLETE'
                    and et_terminal['result'] == plan['et']['report'], 'ET_ORIGINAL_TERMINAL')
            et = join_et(later, load(plan['et'], et_report['OOF_array_headers']), et_report['OOF_array_headers'])
            later_pred = readout.predict_frozen_direction_readout(later['hidden'], state)
            later_metrics, later_rows = score_panel(later, later_pred, prior, LATER_CUTOFF, owner, io, et=et)
            for j, panel in enumerate(later_metrics):
                for arm, field in (('native', 'model_metrics'), ('prior', 'prior_metrics')):
                    old_m = old[field]['per_horizon'][j]
                    require(old_m['horizon_minutes'] == HORIZONS[j]
                            and old_m['counts']['eligible_cells'] == panel['eligible_rows'], 'ORIGINAL_COUNTS')
                    for k in ('logloss', 'brier', 'accuracy', 'balanced_accuracy'):
                        require(abs(old_m[k]-panel['arms'][arm][k]) < 1e-14, 'ORIGINAL_SCORE:' + k)
            intervals = paired_intervals(later_rows, io, rank_owner)
            primary = [x for x in intervals if x['horizon_minutes'] == 5]
            require(len(primary) == 2, 'TWO_PRIMARY_SCHEMES')
            result = {'fit_state': state_ref, 'fit_diagnostics': state['fit_diagnostics'],
                'prefix': train_metrics, 'later': later_metrics, 'intervals': intervals,
                'primary_auc5_supported': all(x['fitted_minus_native_auc_99']['low'] > 0 for x in primary),
                'head_reconstruction': {'prefix': prefix_check, 'later': later_check},
                'full_prefix_rows': PREFIX_ROWS, 'later_rows': LATER_ROWS,
                'original_native_prior_scores_reproduced': True,
                'fitted_later_logit_sha256': io.digest(later_pred, np),
                'limitations': protocol['acceptance_and_stops'] + protocol['interpretation'],
                'economic_edge_qualified': False, 'native_model_updated': False}
            require(counts == {k+s: v for k, v in maxima.items() for s in ('_calls', '_returns')}, 'EXACT_CALL_COUNTS')
        require(not any(io.FORBIDDEN.values()), 'NO_FORBIDDEN_WORK')
        io.pinned_bookends(pins)
        used = io.io_state()['rchar']-before['rchar']
        require(used <= plan['max_rchar_bytes'], 'IO_CAP')
        sys.setprofile(previous)
        for path, (identity, digest) in list(meta.PINNED.items()):
            require(meta.regular(path) == identity, 'METADATA_STAT_BOOKEND')
            meta.read_ref({'path': path, 'sha256': digest})
        require(meta.git_head() == head, 'HEAD_BOOKEND')
        summary = meta.publish(out, 'SUMMARY.json', plain({'status': 'COMPLETE', 'mode': mode, 'plan': pref,
            'source_head': head, 'test_data_used': False, **result}))
        terminal = {'status': 'COMPLETE', 'exit_code': 0, 'summary': summary, 'plan': pref, 'policy': polref,
            'source_head': head, 'claim': claim, 'source_unchanged': True, 'inputs_unchanged': True,
            'counters': {**counts, **io.FORBIDDEN}, 'execution': proof, 'lock': lock,
            'rchar_bytes': used, 'elapsed_seconds': time.monotonic()-start, 'test_data_used': False}
    except Exception as exc:
        sys.setprofile(previous)
        meta.publish(out, 'TERMINAL.json', {'status': 'FAILED', 'exit_code': 1, 'plan': pref, 'policy': polref,
            'claim': claim, 'source_head': head, 'error': type(exc).__name__+': '+str(exc),
            'counters': {**counts, **io.FORBIDDEN}, 'elapsed_seconds': time.monotonic()-start, 'test_data_used': False})
        raise
    finally:
        sys.setprofile(previous)
        for pin in pins.values():
            os.close(pin['fd'])
    terminal_ref = meta.publish(out, 'TERMINAL.json', terminal)
    signal.alarm(0)
    print(json.dumps(terminal_ref))


if __name__ == '__main__':
    main()
