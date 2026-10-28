"""Hide only GC history rows that closed after the preceding M5 decision.

The native GC history contract represents an unavailable bucket by zero values
and false observation masks. This intervention does not mutate model inputs or
alter XAU history, older GC rows, weights, targets or execution authority.
"""
from __future__ import annotations

import numpy as np
import torch

from gx1.audit.direction_gc_context_intervention_v1 import require

TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
M5_NS = 300_000_000_000


def hide_fresh_gc_history(features, *, witness, decision_ns, prior_available):
    """Mask history buckets closed during the last five minutes as unavailable."""
    require(isinstance(decision_ns, np.ndarray) and decision_ns.dtype == np.int64 and decision_ns.ndim == 1
        and isinstance(prior_available, np.ndarray) and prior_available.dtype == np.bool_
        and prior_available.shape == decision_ns.shape and len(decision_ns) > 0
        and np.all(decision_ns >= M5_NS), 'FRESHNESS_DECISIONS')
    require(isinstance(witness, dict) and set(witness) == set(TIMEFRAMES), 'FRESHNESS_WITNESS')
    result = dict(features); counts = {}
    for tf in TIMEFRAMES:
        key = 'gc_seq_' + tf.lower(); values = features[key]; rows = witness[tf]
        require(isinstance(values, torch.Tensor) and values.ndim == 3 and values.shape[0] == len(decision_ns)
            and isinstance(rows, list) and len(rows) == len(decision_ns)
            and all(isinstance(row, dict) and 'available_ns' in row for row in rows), 'FRESHNESS_SHAPE:' + tf)
        clocks = np.asarray([row['available_ns'] for row in rows], dtype=np.int64)
        require(clocks.shape == tuple(values.shape[:2]) and np.all(clocks >= 0)
            and np.all(np.diff(clocks, axis=1) > 0) and np.all(clocks <= decision_ns[:, None]),
            'FRESHNESS_CLOCK:' + tf)
        hidden = (clocks > (decision_ns - M5_NS)[:, None]) & prior_available[:, None]
        require(np.all(hidden.sum(axis=1) <= 1), 'FRESHNESS_MULTIPLE_BUCKETS:' + tf)
        mask = torch.as_tensor(hidden, dtype=torch.bool, device=values.device)
        updated = values.clone().masked_fill(mask.unsqueeze(-1), 0)
        require(torch.equal(updated[~mask], values[~mask]) and torch.count_nonzero(updated[mask]) == 0,
            'FRESHNESS_PRESERVATION:' + tf)
        changed = mask & torch.any(values != 0, dim=-1)
        counts[tf] = {'hidden_rows': int(mask.sum().item()), 'nonzero_rows_hidden': int(changed.sum().item())}
        result[key] = updated
    return result, counts
