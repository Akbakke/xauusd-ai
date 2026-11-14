"""Fixed fold 2 TRAIN/later readout and paired later F M5 AUC uncertainty."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import (
    TF, direction_readout,
)
from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
    BOOTSTRAP_SEEDS, PANEL_ROWS,
)

KEYS = ('parents', 'issues', 'state_raw', 'future_raw',
        'state_allowed', 'state_eligible', 'state_mature',
        'future_allowed', 'future_eligible', 'future_mature')
DAY_NS = 86400 * 10**9


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('FOLD2_M5_READOUT_' + reason)


def read_capture(binding: dict) -> dict:
    path = Path(binding['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and set(binding) == {'path', 'sha256'}
         and hashlib.sha256(path.read_bytes()).hexdigest() == binding['sha256'],
         'CAPTURE_BINDING')
    with np.load(path, allow_pickle=False) as archive:
        need(set(archive.files) == set(KEYS), 'CAPTURE_FIELDS')
        data = {key: archive[key].copy() for key in KEYS}
    need(data['parents'].shape == data['issues'].shape == (PANEL_ROWS,)
         and data['parents'].dtype == data['issues'].dtype == np.int64
         and np.all(np.diff(data['issues']) > 0), 'CAPTURE_AXES')
    for task in ('state', 'future'):
        need(data[task + '_raw'].shape == (PANEL_ROWS, 5, 2)
             and np.isfinite(data[task + '_raw']).all()
             and all(data[task + '_' + key].shape == (PANEL_ROWS, 5)
                     for key in ('allowed', 'eligible', 'mature')),
             'CAPTURE_' + task.upper())
    return data


def score_capture(data: dict) -> dict:
    """Use the completed INITIAL scorer, adding predeclared known-side log loss."""
    issues = data['issues']
    months = issues.astype('datetime64[ns]').astype('datetime64[M]').astype(str)
    out = {}
    for task in ('state', 'future'):
        truth = {key: data[task + '_' + key] for key in ('allowed', 'eligible', 'mature')}
        cells = direction_readout(data[task + '_raw'], truth, issues, np)
        for month, selected in [('all', np.ones(PANEL_ROWS, dtype=bool)),
                                *((str(m), months == m) for m in np.unique(months))]:
            for j, tf in enumerate(TF):
                known = selected & truth['eligible'][:, j] & truth['mature'][:, j] \
                    & np.isin(truth['allowed'][:, j], (1, 4))
                y = truth['allowed'][known, j] == 4
                logits = data[task + '_raw'][known, j, 1].astype(np.float64)
                cells[month][tf]['log_loss'] = (
                    float(np.logaddexp(0, np.where(y, -logits, logits)).mean())
                    if len(y) else None)
        out[task] = cells
    return out


def weighted_auc(logits: np.ndarray, up: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Integer cluster weights, stable score ties, and half credit for ties."""
    order = np.argsort(logits, kind='stable')
    z, y, w = logits[order], up[order], weights[:, order]
    starts = np.r_[0, 1 + np.flatnonzero(z[1:] != z[:-1])]
    positive = np.add.reduceat(w * y[None, :], starts, axis=1)
    negative = np.add.reduceat(w * (~y)[None, :], starts, axis=1)
    n_up, n_down = positive.sum(1), negative.sum(1)
    wins = (positive * (np.cumsum(negative, axis=1) - .5 * negative)).sum(1)
    result = np.full(len(weights), np.nan)
    np.divide(wins, n_up * n_down, out=result,
              where=(n_up > 0) & (n_down > 0))
    return result


def interval(draws: np.ndarray, point: float) -> dict:
    valid = np.isfinite(draws)
    need(valid.any(), 'EMPTY_BOOTSTRAP')
    lower, upper = np.quantile(draws[valid], [.025, .975])
    return {'point_difference': float(point), 'lower95': float(lower),
            'upper95': float(upper), 'valid_replicates': int(valid.sum()),
            'undefined_replicates': int((~valid).sum())}


def paired_later_f_m5(initial: dict, final: dict) -> dict:
    """Predeclared 2000 paired UTC-day/week AUC draws on fixed later rows."""
    for key in KEYS:
        if not key.endswith('_raw'):
            need(np.array_equal(initial[key], final[key]), 'PAIRED_AXES_TARGETS')
    issues = initial['issues']
    months = issues.astype('datetime64[ns]').astype('datetime64[M]').astype(str)
    need(tuple(np.unique(months)) == ('2025-01', '2025-02', '2025-03'),
         'FIXED_LATER_MONTHS')
    allowed = initial['future_allowed'][:, 0]
    known = initial['future_eligible'][:, 0] & initial['future_mature'][:, 0] \
        & np.isin(allowed, (1, 4))
    up = allowed == 4
    days = issues // DAY_NS
    points = {phase: score_capture(data)['future']
              for phase, data in (('INITIAL', initial), ('FINAL', final))}
    out = {}
    for scheme, groups, seed in (
        ('UTC_DAY', days, BOOTSTRAP_SEEDS['UTC_DAY']),
        ('UTC_MONDAY_WEEK', (days + 3) // 7,
         BOOTSTRAP_SEEDS['UTC_MONDAY_WEEK']),
    ):
        unique, inverse = np.unique(groups, return_inverse=True)
        n_groups = len(unique)
        draws = np.random.default_rng(seed).multinomial(
            n_groups, np.full(n_groups, 1 / n_groups), size=2000)
        per_month = {}
        for month in ('all', '2025-01', '2025-02', '2025-03'):
            mask = known if month == 'all' else known & (months == month)
            y = up[mask]
            need(y.any() and (~y).any(), 'BOTH_CLASSES_' + month)
            weights = draws[:, inverse[mask]].astype(np.float64)
            auc_draws = {phase: weighted_auc(
                data['future_raw'][mask, 0, 1].astype(np.float64), y, weights)
                for phase, data in (('INITIAL', initial), ('FINAL', final))}
            initial_auc = points['INITIAL'][month]['M5']['auc']
            final_auc = points['FINAL'][month]['M5']['auc']
            per_month[month] = {
                'initial_auc': interval(auc_draws['INITIAL'], initial_auc),
                'final_auc': interval(auc_draws['FINAL'], final_auc),
                'final_minus_initial_auc': interval(
                    auc_draws['FINAL'] - auc_draws['INITIAL'],
                    final_auc - initial_auc),
                'known_side_n': int(mask.sum()),
            }
        out[scheme] = {'clusters': n_groups, 'months': per_month}
    primary = out['UTC_DAY']['months']['all'], out['UTC_MONDAY_WEEK']['months']['all']
    qualified = all(cell['final_minus_initial_auc']['lower95'] > 0
                    and cell['final_auc']['lower95'] > .5 for cell in primary)
    return {'schemes': out, 'development_ranking_signal_qualified': qualified,
            'test_data_used': False}
