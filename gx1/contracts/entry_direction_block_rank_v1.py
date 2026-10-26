"""Exact block-pair sufficient statistics for conditional direction AUC.

No fitting, data access, block selection or inference policy. Caller owns each.
"""
import numpy as np


def block_rank_pairs(labels, scores, block_ids):
    """Return twice rank wins by positive/negative block, including score ties."""
    if not (isinstance(labels, np.ndarray) and labels.dtype == np.int8
            and labels.ndim == 1 and len(labels) > 0
            and np.isin(labels, (0, 1)).all()):
        raise ValueError("BINARY_LABELS")
    n = len(labels)
    if not (isinstance(scores, np.ndarray) and scores.dtype == np.float64
            and scores.shape == (n,) and np.isfinite(scores).all()):
        raise ValueError("FINITE_SCORES")
    if not (isinstance(block_ids, np.ndarray) and block_ids.dtype.kind == "i"
            and block_ids.shape == (n,) and (block_ids >= 0).all()):
        raise ValueError("BLOCK_IDS")
    k = int(block_ids.max()) + 1
    if not np.array_equal(np.unique(block_ids), np.arange(k)):
        raise ValueError("CONTIGUOUS_BLOCK_IDS")
    order = np.argsort(scores, kind="stable")
    z, y, b = scores[order], labels[order], block_ids[order]
    starts = np.r_[0, np.flatnonzero(z[1:] != z[:-1]) + 1]
    group = np.cumsum(np.r_[0, z[1:] != z[:-1]], dtype=np.int64)
    positive = np.bincount(b[y == 1], minlength=k).astype(np.int64)
    negative = np.bincount(b[y == 0], minlength=k).astype(np.int64)
    twice_wins = np.zeros((k, k), dtype=np.int64)
    for j in range(k):
        neg = np.add.reduceat(((y == 0) & (b == j)).astype(np.int64), starts)
        before = np.cumsum(neg) - neg
        rank = (2 * before + neg)[group[y == 1]]
        # Integer pair counts are exactly represented as doubles at this scope.
        if n * n >= 2**52:
            raise ValueError("EXACT_INTEGER_SCOPE")
        twice_wins[:, j] = np.bincount(b[y == 1], weights=rank,
                                      minlength=k).astype(np.int64)
    return {"twice_wins": twice_wins, "positive": positive, "negative": negative}


def weighted_rank_pairs(summary, weights):
    """Exact duplicate-block rank numerator/denominator for each resample."""
    p, n, u = summary["positive"], summary["negative"], summary["twice_wins"]
    w = np.asarray(weights, dtype=np.float64)
    if (w.ndim != 2 or w.shape[1] != len(p) or len(p) != len(n)
            or u.shape != (len(p), len(p)) or not np.isfinite(w).all()
            or (w < 0).any() or not np.equal(w, np.floor(w)).all()
            or (w.sum(axis=1) == 0).any()):
        raise ValueError("BLOCK_WEIGHTS")
    pairs = (w @ p) * (w @ n)
    twice_wins = np.sum((w @ u) * w, axis=1)
    if (twice_wins < 0).any() or (twice_wins > 2 * pairs).any():
        raise ValueError("PAIR_ACCOUNTING")
    return twice_wins, pairs
