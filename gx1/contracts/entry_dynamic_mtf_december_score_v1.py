"""One reserved December score for the sealed five-timeframe forecast; no fit or model construction."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target

TFS = target.TIMEFRAMES
LEADS = tuple(target.LOCAL_WINDOW_M1_BARS[tf] for tf in TFS)
MIN = target.MINUTE_NS
DAY = 1440 * MIN
FIT_END = 1733011200000000000
SCORE_END = 1735689600000000000
ROWS = 5736
EVENTS = ('flat_activity', 'switch_1', 'first_exit', 'first_return')
METHODS = ('model', 'markov')
GAPS = ('no_gap', 'published_only', 'unknown_only', 'published_and_unknown', 'missing_clock')


def _require(ok, msg):
    if not ok:
        raise ValueError('DECEMBER_REGISTERED_SCORE_' + msg)


def _seal(data):
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return {**data, 'contract_sha256': digest}


def december_event_layout(rows):
    _require(type(rows) is int and rows > 0, 'ROW_COUNT')
    return {
        'state_evidence': ((rows, 5, 4), np.int8),  # current, native, FLAT activity, +1 switch
        'path_evidence': ((rows, 5, 3), np.int16),  # prefix length (-1 = unknown start), first exit, first return
        'h4_gap': ((rows,), np.uint8),
        'candle': ((rows, 5), np.float64),
        'event_p': ((rows, 5, 4, 2), np.float64),  # event, then candidate / Markov
        'd1_hazard_counts': ((rows, 2, 2, 4), np.int16),  # UTC day offset, event, risk/y/undefined per method
        'd1_hazard_sums': ((rows, 2, 2, 2, 2), np.float64),  # day, event, method, sum(p²) / sum(p*y)
    }


def december_score_storage_plan(*, max_published_json_bytes, max_checkpoint_total_bytes):
    _require(type(max_published_json_bytes) is int and max_published_json_bytes > 0
             and type(max_checkpoint_total_bytes) is int and max_checkpoint_total_bytes >= 0, 'STORAGE_LIMITS')
    sizes = {}
    for key, (shape, typ) in december_event_layout(ROWS).items():
        dtype = np.dtype(typ)
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': dtype.str, 'fortran_order': False, 'shape': shape})
        sizes[key] = header.tell() + int(np.prod(shape)) * dtype.itemsize
    base = report.dynamic_state_prediction_file_bytes(ROWS, LEADS[-1])
    # The shared staging directory becomes the final directory by no-replace rename on the same filesystem.
    raw = sum(base.values()) + sum(sizes.values())
    return _seal({'schema_version': 'gx1_december_score_storage_v1', 'parents': ROWS, 'base_npy': base, 'event_npy': sizes,
                  'event_npy_total_bytes': sum(sizes.values()), 'total_npy_bytes': raw,
                  'second_npy_copy_bytes': 0, 'staged_sidecar_and_manifest_cap': max_published_json_bytes,
                  'campaign_json_reserve': max_published_json_bytes, 'checkpoint_reserve': max_checkpoint_total_bytes,
                  'required_free_bytes': raw + 2 * max_published_json_bytes + max_checkpoint_total_bytes})


def _law(x):
    a = np.asarray(x, dtype=np.float64)
    mass = a.sum(axis=-1, keepdims=True)
    _require(np.all(np.isfinite(a) & (a >= 0) & (a <= 1))
             and np.all(np.abs(mass - 1.) <= 3 * np.finfo(np.float32).eps), 'ISSUED_PROBABILITY')
    # Remove only the admitted finite softmax roundoff before composing a stochastic law. No clipping or floor.
    return a / mass


def issued_events(first, kernels, start, horizon, *, interval=False):
    """Carry each outside class separately. Conditional surviving phase weights avoid long-horizon underflow."""
    p1 = _law(first)
    k = _law(kernels)
    s = np.asarray(start)
    n = len(s)
    _require(type(horizon) is int and horizon >= 1 and p1.shape == (n, 3)
             and k.shape in ((3, 3), (n, horizon-1, 3, 3)) and np.isin(s, (0, 1, 2)).all(), 'EVENT_KERNELS')
    ii = np.arange(n)
    out = np.zeros((n, horizon, 2), np.float64) if interval else None
    # A=not yet left, B[j]=outside in j, conditional on not returned. Return is absorbing.
    alive = np.ones(n, bool)
    no_exit = np.ones(n, bool)
    a = p1[ii, s].copy()
    b = p1.copy()
    b[ii, s] = 0.
    exit_h = b.sum(axis=1)
    if interval:
        out[:, 0, 0] = exit_h
    with np.errstate(divide='ignore', invalid='ignore'):
        log_exit_survival = np.where(exit_h < .5, np.log1p(-exit_h), np.log(a))
        log_return_survival = np.zeros(n, np.float64)
        no_exit &= a > 0.
        for step in range(1, horizon):
            matrix = np.broadcast_to(k, (n, 3, 3)) if k.ndim == 2 else k[:, step-1]
            source_row = matrix[ii, s, :]
            stay = source_row[ii, s]
            away = source_row.copy()
            away[ii, s] = 0.
            exit_h = away.sum(axis=1)
            # Never subtract a rare transition from 1 to obtain its event mass.
            propagated = np.einsum('bi,bij->bj', b, matrix)
            ret_h = propagated[ii, s].copy()
            propagated[ii, s] = 0.
            if interval:
                out[:, step, 0] = np.where(no_exit, exit_h, np.nan)
                out[:, step, 1] = np.where(alive, ret_h, np.nan)
            b = a[:, None] * away + propagated
            a = a * stay
            total = a + b.sum(axis=1)
            log_exit_survival += np.where(exit_h < .5, np.log1p(-exit_h), np.log(stay))
            log_return_survival += np.where(alive, np.where(ret_h < .5, np.log1p(-ret_h), np.log(total)), 0.)
            no_exit &= stay > 0.
            alive &= total > 0.
            np.divide(a, total, out=a, where=alive)
            np.divide(b, total[:, None], out=b, where=alive[:, None])
    cumulative = np.column_stack((-np.expm1(log_exit_survival), -np.expm1(log_return_survival)))
    _require(np.isfinite(cumulative).all() and np.all((cumulative >= 0.) & (cumulative <= 1.)), 'EVENT_MASS')
    return cumulative, out


def _observed_prefix(states, known, current, start_known):
    n, horizon = known.shape
    length = np.where(known.all(axis=1), horizon, (~known).argmax(axis=1)).astype(np.int16)
    length[~start_known] = -1
    t = np.arange(1, horizon+1)
    exit_flag = (t <= length[:, None]) & (states != current[:, None])
    left = np.where(exit_flag.any(axis=1), exit_flag.argmax(axis=1)+1, 0)
    back_flag = ((left[:, None] > 0) & (t > left[:, None]) & (t <= length[:, None])
                 & (states == current[:, None]))
    back = np.where(back_flag.any(axis=1), back_flag.argmax(axis=1)+1, 0)
    return np.column_stack((length, left, back)).astype(np.int16)


def _d1_day_sums(issues, trace, hazards):
    n = len(issues)
    counts = np.zeros((n, 2, 2, 4), np.int16)
    sums = np.zeros((n, 2, 2, 2, 2), np.float64)
    t = np.arange(1, LEADS[-1]+1)
    offset = ((issues % DAY)[:, None] + t * MIN) // DAY
    for event in (0, 1):
        event_t = trace[:, event+1]
        risk = t <= np.where(event_t > 0, event_t, trace[:, 0])[:, None]
        y = t == event_t[:, None]
        for day in (0, 1):
            take = risk & (offset == day)
            counts[:, day, event, 0] = take.sum(axis=1)
            counts[:, day, event, 1] = (take & y).sum(axis=1)
            for method, h in enumerate(hazards):
                p = h[:, :, event]
                defined = np.isfinite(p)
                counts[:, day, event, 2+method] = (take & ~defined).sum(axis=1)
                _require(np.all((p[defined] >= 0.) & (p[defined] <= 1. + 8*np.finfo(np.float64).eps)), 'HAZARD_DOMAIN')
                z = np.where(take & defined, p, 0.)
                sums[:, day, event, method, 0] = np.square(z).sum(axis=1)
                sums[:, day, event, method, 1] = (z * y).sum(axis=1)
    return counts, sums


def _gap_groups(axis, gaps, published, issues):
    # Right rows are the first physical M1 after each gap. Both endpoints must have a physical clock.
    ends = issues + LEADS[3] * MIN
    i, j = np.searchsorted(axis, issues), np.searchsorted(axis, ends)
    exact = (i < len(axis)) & (j < len(axis))
    exact &= (axis[np.minimum(i, len(axis)-1)] == issues) & (axis[np.minimum(j, len(axis)-1)] == ends)
    def crosses(rows):
        return np.searchsorted(rows, j, side='right') > np.searchsorted(rows, i, side='right')
    return np.where(exact, crosses(published).astype(np.uint8) + 2*crosses(np.setdiff1d(gaps, published)), 4).astype(np.uint8)


def batch_evidence(prob, trans, markov_grid, transition, labels, current_path, activity, candle, gaps, *, reference_events=None):
    """Consume the issued law before joining the realized prefix or forming any hazard risk set."""
    n = len(prob)
    current = labels['current_state']
    co = labels['current_observed']
    states = labels['state']
    eligible = labels['observed'] & (labels['maturity_ns'] < SCORE_END)
    issues = labels['issue_ns']
    data = {name: np.zeros(shape, dtype) for name, (shape, dtype) in december_event_layout(n).items()}
    data['candle'][:] = candle
    data['h4_gap'][:] = gaps
    data['state_evidence'][:, :, 0] = np.where(co, current, -1)
    for tf, h in enumerate(LEADS):
        present = co[:, tf]
        s = np.where(present, current[:, tf], 0)
        case = np.where(present, current[:, tf], 3)
        markov = markov_grid[tf, case]
        future = states[:, tf, :h]
        endpoint = eligible[:, tf, h-1]
        state = data['state_evidence'][:, tf]
        state[:, 1] = np.where(endpoint, future[:, -1], -1)
        state[:, 2] = activity[:, tf]
        state[:, 3] = np.where(present & eligible[:, tf, 0], current[:, tf] != future[:, 0], -1)
        scores = data['event_p'][:, tf]
        scores[:, 0] = np.column_stack((prob[:, tf, h-1, target.FLAT], markov[:, h-1, target.FLAT]))
        other = np.arange(3)[None, :] != s[:, None]
        scores[:, 1] = np.column_stack(((_law(prob[:, tf, 0])*other).sum(axis=1),
                                        (_law(markov[:, 0])*other).sum(axis=1)))
        # Do not let future state, path or censoring enter the issued Markov or model forecasts.
        model_cdf, mh = issued_events(prob[:, tf, 0], trans[:, tf, :h-1], s, h, interval=(tf == 4))
        if reference_events is None:
            ref_cdf, rh = issued_events(transition[tf][s], transition[tf], s, h, interval=(tf == 4))
        else:
            by_start, intervals = reference_events[tf]
            ref_cdf, rh = by_start[s], intervals[s] if tf == 4 else None
        scores[:, 2:, 0], scores[:, 2:, 1] = model_cdf, ref_cdf
        scores[~present, 1:] = np.nan
        start_known = present & current_path[:, tf] if tf == 4 else present
        # Each future minute must be both state- and path-observed. Missing endpoints and immature cells are masked.
        trace = _observed_prefix(future, eligible[:, tf, :h] & labels['path_observed'][:, tf, :h], current[:, tf], start_known)
        data['path_evidence'][:, tf] = trace
        if tf == 4:
            data['d1_hazard_counts'], data['d1_hazard_sums'] = _d1_day_sums(issues, trace, (mh, rh))
    return data


def _number(x):
    return float(x) if np.isfinite(x) else None


def _rank(labels, scores, weights=None):
    if len(labels) == 0 or np.unique(labels).size != 2 or not np.isfinite(scores).all():
        return None
    w = np.ones(len(labels)) if weights is None else weights
    _, rank = np.unique(scores, return_inverse=True)
    pos = np.bincount(rank, weights=w * labels)
    neg = np.bincount(rank, weights=w * (1-labels))
    return float(np.dot(pos, np.cumsum(neg) - .5*neg) / (pos.sum() * neg.sum()))


def _errors(p, truth):
    _require(p.shape == (len(truth), 3) and np.isfinite(p).all() and np.all((p >= 0.) & (p <= 1. + 1e-12)), 'ENDPOINT_DOMAIN')
    y = np.eye(3)[truth]
    with np.errstate(divide='ignore'):
        return {'brier': np.square(p-y).sum(axis=1), 'logloss': -np.log(p[np.arange(len(p)), truth])}


def _direction(labels, probabilities, candle):
    keep = labels != target.FLAT
    y = (labels[keep] == target.UP).astype(np.int8)
    scores = {'candle': candle[keep]}
    for method in METHODS:
        p = probabilities[method][keep]
        denom = p[:, target.DOWN] + p[:, target.UP]
        scores[method] = np.divide(p[:, target.UP], denom, out=np.full(len(p), np.nan), where=denom > 0.)
    return keep, y, scores


def _endpoint_rows(labels, methods, candle, stamps):
    if not len(labels):
        return {'rows': 0, 'days': 0, 'no_eligible_rows': True}, {}, {}, None
    _, day, nday = np.unique(stamps//DAY, return_inverse=True, return_counts=True)
    errors = {k: _errors(p, labels) for k, p in methods.items()}
    proper = {}
    for method, fields in errors.items():
        proper[method] = {}
        for k, vals in fields.items():
            means = np.bincount(day, weights=vals) / nday
            proper[method][k] = {'row': _number(vals.mean()), 'day': _number(means.mean()),
                                  'infinite_rows': int(np.isinf(vals).sum())}
    keep, y, score = _direction(labels, methods, candle)
    _, ids, counts = np.unique(stamps[keep]//DAY, return_inverse=True, return_counts=True)
    auc = {}
    for k, vals in score.items():
        auc[k] = {'row': _rank(y, vals), 'day': _rank(y, vals, 1./counts[ids]), 'undefined_rows': int((~np.isfinite(vals)).sum())}
    classes = np.bincount(labels, minlength=3)
    payload = {'rows': len(labels), 'days': len(nday), 'class_counts': dict(zip(target.STATE_NAMES, classes.tolist())),
               'nonflat_rows': len(y), 'nonflat_days': len(counts), 'direction_both_classes': np.unique(y).size == 2,
               'multiple_outcome_days': len(nday) >= 2, 'scores': proper, 'conditional_auc': auc}
    return payload, errors, score, (y, stamps[keep])


def _bootstrap_weights(days):
    d = len(days)
    rng = np.random.Generator(np.random.PCG64(20260911))
    choices = rng.integers(d, size=(10000, d))
    weights = np.zeros_like(choices)
    np.add.at(weights, (np.arange(len(choices))[:, None], choices), 1)
    return weights


def _quantiles(values):
    return np.quantile(values, [.025, .975], method='linear').tolist() if np.isfinite(values).all() else None


def _intervals(stamps, errors, scores, direction):
    days, inv, freq = np.unique(stamps//DAY, return_inverse=True, return_counts=True)
    if len(days) < 2:
        return {'proper': {}, 'conditional_auc': {}, 'why': 'fewer_than_two_outcome_days'}
    w = _bootstrap_weights(days)
    ci = {}
    for k in ('brier', 'logloss'):
        for ref in ('prior', 'markov'):
            m, r = errors['model'][k], errors[ref][k]
            name = k + '_minus_' + ref
            if not (np.isfinite(m).all() and np.isfinite(r).all()):
                ci[name] = {'row': None, 'day': None}
                continue
            delta = np.bincount(inv, weights=m-r)
            ci[name] = {'row': _quantiles((w @ delta) / (w @ freq)), 'day': _quantiles((w @ (delta/freq))/len(days))}
    y, dt = direction
    ranks = {}
    if np.unique(y).size == 2:
        from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs
        d, bi, n = np.unique(dt//DAY, return_inverse=True, return_counts=True)
        if len(d) >= 2:
            wd = _bootstrap_weights(d)
            for mode, ww in (('row', wd), ('day', wd/n)):
                draws = {}
                for method, value in scores.items():
                    if not np.isfinite(value).all():
                        continue
                    z = block_rank_pairs(y, value, bi)
                    den = (ww @ z['positive']) * (ww @ z['negative'])
                    num = ((ww @ z['twice_wins']) * ww).sum(axis=1)
                    draws[method] = np.divide(num, 2*den, out=np.full(len(ww), np.nan), where=den > 0)
                ranks[mode] = {k: (_quantiles(draws['model']-draws[k]) if k in draws and 'model' in draws else None)
                               for k in ('markov', 'candle')}
    return {'proper': ci, 'conditional_auc': ranks}


def _episodes(y, stamps, scores):
    if not len(y):
        return {'episodes': 0, 'auc': {}}
    starts = np.r_[0, np.flatnonzero((np.diff(stamps) != 5*MIN) | (y[1:] != y[:-1])) + 1]
    lengths = np.diff(np.r_[starts, len(y)])
    return {'episodes': len(starts), 'up': int(y[starts].sum()), 'down': int((y[starts] == 0).sum()),
            'auc': {k: _rank(y[starts], np.add.reduceat(s, starts)/lengths) for k, s in scores.items()}}


def _event_groups(day, current, gap, ledger):
    # columns: same-risk, positive, censored, unknown start, undefined per method, and two raw Brier sums per method.
    keys, ids = np.unique(np.column_stack((day, current, gap)), axis=0, return_inverse=True)
    values = np.zeros((len(keys), 10), np.float64)
    np.add.at(values, ids, ledger)
    def summarize(v):
        n, positive, censored, unknown = map(int, v[:4])
        d = {'risk': n, 'positive': positive, 'nonevent': n-positive, 'censored': censored, 'unknown_start': unknown}
        for i, name in enumerate(METHODS):
            undefined = int(v[4+i])
            sq, py = v[6+i*2:8+i*2]
            # The gate uses exactly the same observed rows for both, also when one method has zero survival support.
            d[name] = {'undefined': undefined, 'sum_p2_defined': float(sq), 'sum_p_y_defined': float(py),
                       'brier': float((sq - 2*py + positive)/n) if n and not undefined else None}
        return d
    return {'total': summarize(values.sum(axis=0)),
            'by_outcome_day_current_and_h4_gap': [{'day_since_epoch': int(k[0]), 'current_code': int(k[1]),
                'gap_code': int(k[2]), **summarize(v)} for k, v in zip(keys, values) if np.any(v)]}


def _aux(data, issues, tf, e):
    h = LEADS[tf]
    state = data['state_evidence'][:, tf]
    trace = data['path_evidence'][:, tf]
    current = state[:, 0]
    gap = data['h4_gap'] if tf == 3 else np.full(len(issues), -1)
    time = issues + (1 if e == 1 else h) * MIN
    event_t = trace[:, e-1] if e >= 2 else None
    _require(e < 2 or np.all((event_t >= 0) & (event_t <= np.maximum(trace[:, 0], 0))), 'TRACE_EVENT_TIME')
    if e >= 2 and tf == 4:
        cnt = data['d1_hazard_counts'][:, :, e-2]
        sums = data['d1_hazard_sums'][:, :, e-2]
        ledger = np.zeros((len(issues), 2, 10), np.float64)
        ledger[..., :2] = cnt[..., :2]
        ledger[..., 4:6] = cnt[..., 2:4]
        ledger[..., 6:] = sums.reshape(len(issues), 2, 4)
        # Count each mother issue once at loss of observability. An unknown start is a distinct stratum.
        known = trace[:, 0] >= 0
        censored = known & (event_t == 0) & (trace[:, 0] < h)
        cday = (issues % DAY + (np.maximum(trace[:, 0], 0)+1).astype(np.int64)*MIN)//DAY
        ledger[np.flatnonzero(censored), cday[censored], 2] = 1
        ledger[:, 0, 3] = ~known
        day = issues[:, None]//DAY + np.arange(2)
        groups = _event_groups(day.ravel(), np.repeat(current, 2), np.repeat(gap, 2), ledger.reshape(-1, 10))
        risk_len = np.where(event_t > 0, event_t, np.maximum(trace[:, 0], 0))
        same_day_minutes = (DAY-1-issues%DAY)//MIN
        _require(np.array_equal(cnt[:, :, 0].sum(axis=1), risk_len)
                 and np.array_equal(cnt[:, 0, 0], np.minimum(risk_len, same_day_minutes))
                 and np.array_equal(cnt[:, :, 1].sum(axis=1), event_t > 0)
                 and np.array_equal(cnt[:, 0, 1], (event_t > 0) & (event_t <= same_day_minutes)), 'D1_PAIRED_EXPOSURE')
    else:
        if e < 2:
            y = state[:, 2+e]
            unknown = (current < 0) if e == 1 else np.zeros(len(issues), bool)
            censored = (y < 0) & ~unknown
        else:
            y = np.where(event_t > 0, 1, np.where(trace[:, 0] == h, 0, -1))
            censored = (trace[:, 0] >= 0) & (y < 0)
            unknown = trace[:, 0] < 0
            time = issues + np.where(event_t > 0, event_t, np.where(censored, trace[:, 0]+1, np.where(unknown, 0, h))).astype(np.int64)*MIN
        take = y >= 0
        p = data['event_p'][:, tf, e]
        ledger = np.zeros((len(issues), 10), np.float64)
        ledger[:, :4] = np.column_stack((take, y == 1, censored, unknown))
        for m in range(2):
            valid = np.isfinite(p[:, m])
            z = np.where(take & valid, p[:, m], 0.)
            ledger[:, 4+m] = take & ~valid
            ledger[:, 6+m*2] = z*z
            ledger[:, 7+m*2] = z*(y == 1)
        groups = _event_groups(time//DAY, current, gap, ledger)
    total = groups['total']
    if e >= 2:
        observed_times = issues[event_t > 0] + event_t[event_t > 0].astype(np.int64)*MIN
        total['known_starts_no_next_minute'] = int((trace[:, 0] == 0).sum())
        if tf == 4:
            # Union of contiguous, observed event-free intervals; repeated issues are distinct exposures.
            none = np.where(event_t > 0, event_t-1, np.maximum(trace[:, 0], 0))
            origin = int(issues[0]//MIN)
            begin = issues//MIN - origin + 1
            bound = int((begin+none).max())+1
            delta = np.bincount(begin, minlength=bound) - np.bincount(begin+none, minlength=bound)
            cover = np.cumsum(delta)
            _require(int(cover.sum()) == total['nonevent'], 'D1_NONEVENT_UNION')
            minutes = np.flatnonzero(cover) + origin
            total['distinct_exposed_nonevent_minutes'] = len(minutes)
            total['distinct_exposed_nonevent_days'] = np.unique(minutes//1440).size
    else:
        observed_times = time[y == 1]
    total['distinct_event_minutes'] = np.unique(observed_times).size
    total['distinct_event_days'] = np.unique(observed_times//DAY).size
    b = [total[k]['brier'] for k in METHODS]
    groups['gate'] = (None if not total['positive'] or not total['nonevent'] or None in b else bool(b[0] <= b[1]))
    return groups


def registered_report(arrays, prior, markov_grid):
    ids, issues = np.asarray(arrays['parent_rows']), np.asarray(arrays['issue_ns'])
    _require(ids.shape == issues.shape == (ROWS,) and np.array_equal(ids, np.arange(41866, 47602))
             and np.all(issues[1:] > issues[:-1]) and FIT_END <= issues[0] <= issues[-1] < SCORE_END, 'RESERVED_AXES')
    result, verdicts = {}, {}
    for tf, name in enumerate(TFS):
        h = LEADS[tf]
        current, state = arrays['state_evidence'][:, tf, :2].T
        known = state >= 0
        stamps = issues+h*MIN
        _require(np.all(stamps[known] < SCORE_END), 'MATURE_ENDPOINT')
        case = np.where(current >= 0, current, 3)
        methods = {'model': np.asarray(arrays['probabilities'][:, tf, h-1], np.float64),
                   'markov': markov_grid[tf, case, h-1], 'prior': np.broadcast_to(prior[tf], (len(issues), 3))}
        selected = {key: p[known] for key, p in methods.items()}
        candle = arrays['candle'][:, tf]
        overall, errs, scores, direction = _endpoint_rows(state[known], selected, candle[known], stamps[known])
        ci = _intervals(stamps[known], errs, scores, direction) if np.any(known) else {'proper': {}, 'conditional_auc': {}}
        def subgroup(mask):
            a = known & mask
            return _endpoint_rows(state[a], {k: p[a] for k, p in methods.items()}, candle[a], stamps[a])[0]
        groups = {s: subgroup(current == k) for k, s in ((-1, 'UNKNOWN'), *enumerate(target.STATE_NAMES))}
        auxiliary = {e: _aux(arrays, issues, tf, i) for i, e in enumerate(EVENTS)}
        gates = {}
        for metric in ('brier', 'logloss'):
            for weight in ('row', 'day'):
                for ref in ('prior', 'markov'):
                    key = metric+'_'+weight+'_vs_'+ref
                    if not np.any(known):
                        gates[key] = None
                    else:
                        x, y = (overall['scores'][k][metric] for k in ('model', ref))
                        a = np.inf if x['infinite_rows'] else x[weight]
                        b = np.inf if y['infinite_rows'] else y[weight]
                        gates[key] = bool(a < b) if np.isfinite(a) or np.isfinite(b) else None
        ranks = overall.get('conditional_auc', {})
        rank = ranks.get('model', {}).get('row')
        baselines = [ranks.get(k, {}).get('row') for k in ('candle', 'markov')]
        gates['row_conditional_auc'] = bool(rank > max(.5, *baselines)) if rank is not None and None not in baselines else None
        ci_markov = ci['proper'].get('brier_minus_markov', {}).get('day')
        gates['day_brier_upper_below_zero'] = bool(ci_markov[-1] < 0) if ci_markov is not None else None
        episode = None
        if tf >= 3:
            episode = _episodes(*direction, scores) if direction is not None else {'episodes': 0, 'auc': {}}
            ea = episode['auc'].get('model')
            gates['episode_auc'] = None if ea is None else bool(ea > .5)
        gates.update({e: value['gate'] for e, value in auxiliary.items()})
        verdicts[name] = gates
        result[name] = {'native_endpoint': overall, 'parents': len(issues), 'eligible': int(known.sum()),
            'immature': int((stamps >= SCORE_END).sum()), 'missing_mature': int(((stamps < SCORE_END) & ~known).sum()),
            'paired_intervals': ci, 'episodes': episode, 'by_current': groups, 'events': auxiliary}
        if tf == 3:
            result[name]['by_path_gap'] = {key: subgroup(arrays['h4_gap'] == code) for code, key in enumerate(GAPS)}
    flags = [value for gates in verdicts.values() for value in gates.values()]
    return {'schema_version': 'gx1_december_reserved_score_v1', 'period': '2024-12', 'parents': len(issues),
            'contract_sha256': report.december_oof_scoring_contract()['contract_sha256'],
            'timeframes': result, 'gates': verdicts,
            'decision': 'PASS_FOR_RESEARCH_REVIEW' if all(v is True for v in flags)
                        else 'FAIL_REGISTERED_GATE' if any(v is False for v in flags) else 'INCONCLUSIVE',
            'any_unestimable': any(v is None for v in flags), 'independent_edge_demonstrated': False,
            'economic_or_trading_authority': False}


def require_december_score_storage(directory, plan, *, expected_host=None):
    from datetime import datetime, timezone
    from gx1.contracts.entry_native_direction_prefix_fit_campaign_v1 import _observe_wsl_host_storage
    _require(plan == december_score_storage_plan(max_published_json_bytes=plan['campaign_json_reserve'],
             max_checkpoint_total_bytes=plan['checkpoint_reserve']), 'STORAGE_PLAN_IDENTITY')
    observed = _observe_wsl_host_storage(Path(directory))
    host = (observed['host_base_path'], observed['host_drive_root'])
    _require(expected_host is None or tuple(expected_host) == host, 'HOST_IDENTITY')
    free = min(observed[k] for k in ('guest_available_bytes', 'host_available_bytes', 'mounted_host_available_bytes'))
    _require(free >= plan['required_free_bytes'], 'INSUFFICIENT_ACTUAL_DISK')
    return {'plan': plan, 'observation': observed, 'host_identity': list(host),
            'observed_utc': datetime.now(timezone.utc).isoformat(), 'reserved': False}


class DecemberScorePass:
    def __init__(self, *, prepared, targets, view, selected, cutoff, directory, markov):
        from gx1.contracts import local_random_access_campaign_v2 as campaign
        from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4

        self.inputs = prepared['december_score_inputs']
        self.series = targets['dynamic_state_series']
        self.prior = np.asarray(targets['prior']['probabilities'], np.float64)
        self.transition = np.asarray(targets['prior']['transition_probabilities'], np.float64)
        self.grid = markov
        refs = self.inputs['receipt_refs']
        for key in ('prior', 'candle', 'fit_terminal_review', 'terminal_checkpoint', 'scoring_admission', 'target_manifest', 'calendar_authority'):
            campaign.require_binding(refs[key], label='score '+key, verify_file=False)
        _require(refs['target_manifest'] == targets['target_manifest'] and targets['prior']['parents'] == 41866
                 and view.get('native_endpoint_loss') is True and view.get('zero_support_profile') is not None
                 and cutoff == SCORE_END and self.series.metadata['cutoff_ns'] == SCORE_END
                 and np.array_equal(selected, np.arange(41866, 47602)), 'SCORE_ROLE')
        admission = campaign.read_bound_json(Path(refs['scoring_admission']['path']), refs['scoring_admission']['sha256'])
        _require(admission.get('input_role') == 'score' and admission.get('source_terminal') == refs['fit_terminal_review']
                 and admission.get('fit_checkpoint') == refs['terminal_checkpoint']
                 and admission.get('target_manifest') == refs['target_manifest']
                 and admission.get('calendar_authority') == refs['calendar_authority']
                 and admission.get('storage_plan') == self.inputs['storage_plan']
                 and admission.get('scoring_contract_sha256') == report.december_oof_scoring_contract()['contract_sha256'],
                 'SIGNED_SCORE_ADMISSION')
        frozen_prior = campaign.read_bound_json(Path(refs['prior']['path']), refs['prior']['sha256'])
        candle = campaign.read_bound_json(Path(refs['candle']['path']), refs['candle']['sha256'])
        signs = self.inputs['candle_orientation']
        _require(frozen_prior == targets['prior'] and candle['status'] == 'FROZEN_PREFIX_CANDLE_REFERENCE'
                 and candle['fit_parents'] == 41866 and candle['fit_cutoff_ns'] == FIT_END
                 and set(signs) == set(TFS) and all(type(signs[tf]) is int and signs[tf] in (-1, 1)
                 and signs[tf] == candle['panels'][tf]['orientation'] for tf in TFS), 'PREFIX_REFERENCES')
        self.signs = np.array([signs[tf] for tf in TFS])
        self.feature = MULTI_TF_PER_BAR_FEATURES_V4.index('close_open_atr')
        self.axis = self.series.arrays['validity_ns']
        self.gaps = np.asarray(self.inputs['gap_after_rows'])
        self.published = np.asarray(self.inputs['published_closure_after_rows'])
        _require(np.issubdtype(self.gaps.dtype, np.integer) and np.issubdtype(self.published.dtype, np.integer)
                 and np.array_equal(self.gaps, np.flatnonzero(np.diff(self.axis) != MIN)+1)
                 and np.array_equal(self.published, np.unique(self.published))
                 and np.isin(self.published, self.gaps).all(), 'BOUND_CLOSURE_ROWS')
        self.storage = require_december_score_storage(directory, self.inputs['storage_plan'], expected_host=self.inputs['expected_host'])
        self.reference_events = []
        for tf, horizon in enumerate(LEADS):
            law = self.transition[tf]
            self.reference_events.append(issued_events(law, law, np.arange(3), horizon, interval=(tf == 4)))
        self.issued_hash = hashlib.sha256()

    def batch(self, ids, raw, labels, probabilities, transitions):
        issues = labels['issue_ns']
        # The native worker already proves model/label historical-current parity in its batch constructor.
        features = raw['features']
        _require(np.array_equal(features['current_state'], labels['current_state'])
                 and np.array_equal(features['current_observed'], labels['current_observed'])
                 and np.array_equal(ids, raw['witness']['parent_rows'])
                 and np.array_equal(issues, raw['witness']['decision_ns']), 'BATCH_ISSUE')
        candle = np.column_stack([features['seq_'+tf.lower()][:, -1, self.feature] for tf in TFS]) * self.signs
        _require(np.isfinite(candle).all(), 'FINITE_PHYSICAL_CANDLE')
        activity = np.empty((len(ids), 5), np.int8)
        for tf, horizon in enumerate(LEADS):
            q = target.query_dynamic_mtf_states(self.series, parent_rows=np.asarray(ids, np.int64), issue_ns=issues,
                    cutoff_ns=SCORE_END, lead_minutes=[horizon], include_first_flat_activity=True)
            _require(np.array_equal(q['state'][:, tf, 0], labels['state'][:, tf, horizon-1])
                     and np.array_equal(q['observed'][:, tf, 0], labels['observed'][:, tf, horizon-1]), 'NATIVE_ACTIVITY_JOIN')
            activity[:, tf] = np.where(q['first_flat_activity_observed'][:, tf], q['first_flat_activity_is_flat'][:, tf], -1)
        for x in (np.asarray(ids, '<i8'), np.asarray(issues, '<i8'), probabilities[:, :, 0], transitions):
            self.issued_hash.update(memoryview(np.ascontiguousarray(x)).cast('B'))
        return batch_evidence(probabilities, transitions, self.grid, self.transition, labels, features['current_path_observed'],
                activity, candle, _gap_groups(self.axis, self.gaps, self.published, issues), reference_events=self.reference_events)

    def finalize(self, arrays, checkpoint_identity):
        _require(hashlib.sha256(np.asarray(arrays['issue_ns'], '<i8').tobytes()).hexdigest()
                 == 'b5a223fae32392ddc45e96c4e8e44f09f43aae91f1eb85f2805dbae3034707b6', 'RESERVED_ISSUE_DIGEST')
        result = registered_report(arrays, self.prior, self.grid)
        return {**result, 'receipt_refs': self.inputs['receipt_refs'], 'storage_check': self.storage,
                'candle_orientation': self.inputs['candle_orientation'], 'issued_kernels_sha256': self.issued_hash.hexdigest(),
                'issued_hash_order': 'each batch int64 ids, issue clocks, FP32 first-state probabilities and FP32 transitions in C order',
                'checkpoint_state_identity': checkpoint_identity,
                'state_evidence_columns': ['historical_current_or_unknown', 'native_endpoint_or_unknown', 'flat_activity_or_unknown', 'change_1_or_unknown'],
                'path_evidence_columns': ['known_consecutive_future_minutes_or_negative_unknown_start', 'first_exit_minute_zero_if_absent', 'first_return_minute_zero_if_absent'],
                'event_order': list(EVENTS), 'methods': list(METHODS), 'h4_gap_codes': list(GAPS),
                'd1_count_axis': ['observed_risk', 'event', 'undefined_model', 'undefined_markov'],
                'd1_sum_axis': ['sum_defined_p_squared', 'sum_defined_p_times_y'],
                'd1_day_offsets': [0, 1], 'direction_weights': 'row or equal UTC target-validity day',
                'event_rate_weights': 'observed at-risk issue-minutes for D1 hazards, qualified parents for other binary events',
                'probability_policy': 'native proper scores use issued marginals; finite stochastic rows are validated then divided by their FP64 sum for phase propagation. No clipping.'}
