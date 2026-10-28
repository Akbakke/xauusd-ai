"""Exploratory fixed past-price direction references on saved development scores.

The prior movement ends five minutes before the decision and its terminal M1
candle is required to close by the decision. No model fitting, trading, TEST,
threshold selection, or claim of a profitable strategy is made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from gx1.contracts import entry_direction_block_rank_v1 as rank
from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights, percentile_interval

REPO = Path('/home/andre2/src/GX1_CURRENT')
MINUTE = 60_000_000_000
DAY = 1440 * MINUTE
REPLICATES = 10000


def require(ok, message):
    if not bool(ok):
        raise RuntimeError('SAVED_PAST_MOVE_DIRECTION_' + message)


def previous_move_cohort(target, rows, cutoff):
    """A lagged five-minute outcome is known before the present decision."""
    decision = target['decision_ns']; parents = target['parent_rows']
    require(rows.dtype == np.int64 and rows.ndim == 1 and len(rows) > 0
        and np.all((0 <= rows) & (rows < len(decision))) and np.all(np.diff(decision) > 0)
        and np.all(np.diff(parents) > 0), 'ROWS')
    now = decision[rows]; requested = now - 10 * MINUTE
    prior = np.searchsorted(decision, requested); safe = np.minimum(prior, len(decision) - 1)
    exists = (prior < len(decision)) & (decision[safe] == requested)
    current_sign = target['sign'][rows, 4]; previous_sign = target['sign'][safe, 4]
    past_terminal = target['terminal_time_ns'][safe, 4]
    require(np.all(past_terminal[exists] == requested[exists] + 5 * MINUTE), 'PAST_TERMINAL_CLOCK')
    eligible = exists & target['observed'][rows, 4] & (current_sign != 0) \
        & (target['terminal_time_ns'][rows, 4] < cutoff) & target['observed'][safe, 4] \
        & (previous_sign != 0) & (past_terminal + MINUTE <= now)
    return eligible, previous_sign


def _ref(path):
    path = Path(path); require(path.is_absolute() and path.is_file(), 'FILE')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def _read(ref):
    require(_ref(ref['path']) == ref, 'SHA:' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def evaluate(y, past, decisions, model=None):
    require(y.dtype == np.int8 and np.isin(y, (0, 1)).all() and len(np.unique(y)) == 2
        and past.shape == decisions.shape == y.shape and np.isin(past, (-1, 1)).all(), 'METRIC_INPUT')
    reversal = np.ascontiguousarray((past < 0).astype(np.float64))
    arms = {'always_long': np.ones(len(y), dtype=np.float64), 'continuation': 1.0 - reversal, 'reversal': reversal}
    if model is not None:
        require(model.shape == y.shape and np.isfinite(model).all(), 'MODEL_INPUT')
        arms['model'] = np.ascontiguousarray(model, dtype=np.float64)
    scores = {name: {'accuracy': float(np.mean((z > (0 if name == 'model' else 0.5)) == y)),
        'auc': float(roc_auc_score(y, z))} for name, z in arms.items()}
    intervals = {}
    days = decisions // DAY
    for scheme, keys, seed in (('UTC_DAY', days, 20261027), ('UTC_WEEK_MONDAY', (days + 3) // 7, 20261028)):
        blocks, ids = np.unique(keys, return_inverse=True); require(len(blocks) >= 8, 'BLOCKS')
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        weights = block_weights(len(blocks), seed=seed, np=np, replicates=REPLICATES)
        counts = np.bincount(ids, minlength=len(blocks)); denominator = weights @ counts
        def accuracy_interval(values):
            sums = np.bincount(ids, weights=values, minlength=len(blocks))
            return percentile_interval((weights @ sums) / denominator, np=np, alpha=0.05)
        hit = (reversal == y).astype(np.float64)
        item = {'blocks': len(blocks), 'reversal_accuracy': accuracy_interval(hit),
            'reversal_minus_always_long_accuracy': accuracy_interval(hit - y)}
        if model is not None:
            samples = []
            for value in (model, reversal):
                pairs = rank.block_rank_pairs(y, np.ascontiguousarray(value, dtype=np.float64), ids)
                numerator, total = rank.weighted_rank_pairs(pairs, weights)
                samples.append(np.divide(numerator, 2 * total, out=np.full(REPLICATES, np.nan), where=total > 0))
            item['reversal_minus_model_auc'] = percentile_interval(samples[1] - samples[0], np=np, alpha=0.05)
        intervals[scheme] = item
    return {'eligible': len(y), 'actual_up': int(y.sum()), 'past_up': int((past > 0).sum()),
        'scores': scores, 'bootstrap_95_percent': intervals}


def main():
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    support = _read(policy['independent_direction_unseen_gc_support_20261028_001']['plan'])
    followup = _read(policy['independent_direction_gc_context_followup_20261028_002']['plan'])
    refs = {'target': support['inputs']['corrected_target'], 'q1': followup['inputs']['oof_predictions'],
        'q2': support['inputs']['sample_pair']}
    for ref in refs.values(): require(_ref(ref['path']) == ref, 'INPUT_HASH')
    with np.load(refs['target']['path'], allow_pickle=False) as a:
        target = {k: a[k].copy() for k in ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')}
    groups = {}
    for name, cutoff in (('train', 1735689600000000000), ('q1', 1743465600000000000), ('q2', 1751328000000000000)):
        if name == 'train':
            rows = np.arange(47602, dtype=np.int64); model = None
        else:
            with np.load(refs[name]['path'], allow_pickle=False) as a:
                parents, decisions = a['parent_rows'].copy(), a['decision_ns'].copy()
                model = a['logit_up' if name == 'q1' else 'original_logit_up'][:, 4].astype(np.float64)
            rows = np.searchsorted(target['parent_rows'], parents).astype(np.int64)
            require(np.all(rows < len(target['parent_rows'])) and np.array_equal(target['parent_rows'][rows], parents)
                and np.array_equal(target['decision_ns'][rows], decisions), 'SAVED_JOIN')
        eligible, previous = previous_move_cohort(target, rows, cutoff)
        require(eligible.any(), 'ELIGIBLE')
        y = np.ascontiguousarray((target['sign'][rows[eligible], 4] > 0).astype(np.int8))
        groups[name] = evaluate(y, previous[eligible], target['decision_ns'][rows[eligible]],
            None if model is None else model[eligible])
        groups[name]['total_parents'] = len(rows)
    result = {'schema_version': 'gx1_saved_past_move_direction_diagnostic_v1',
        'status': 'COMPLETE_EXPLORATORY_SAVED_DATA_ONLY', 'inputs': refs,
        'source': _ref(REPO / 'gx1/scripts/analyze_saved_past_move_direction_v1.py'), 'groups': groups,
        'rule': 'At decision t use the sign of exact M1-open midpoint change from t-10min to t-5min, provided the terminal M1 candle closes no later than t. Score both fixed continuation and reversal against t to t+5min.',
        'limits': 'Exploratory post-inspection comparison on reused development periods; Q2 is a fixed subsample. M1 candle opens lack quote event timestamps. Day/week bootstrap assumes exchangeable blocks; no multiple-selection correction, untouched validation, trading costs or fills. Absolute accuracy against 50% is not economic edge.',
        'new_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False, 'economic_edge_qualified': False}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
