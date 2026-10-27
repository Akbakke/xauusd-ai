"""Score existing 5/15-minute heads on a derived 10-minute endpoint.

Research-only pure array functions: no archive reads, model construction,
training, trading actions, threshold selection, or run authority.
"""
from __future__ import annotations

import numpy as np

from gx1.contracts import entry_direction_block_rank_v1 as block_rank
from gx1.contracts import entry_direction_saved_horizons_v1 as saved_horizons

_DAY_NS = 86400 * 10**9
_SCORE_COLUMNS = (("head_5", 1), ("head_15", 2))
_ARMS = ("native", "et252", "r1", "ref259")
_PRIMARY = (("head_5", "native", "et252"), ("head_15", "native", "et252"))


def _require(condition, reason):
    if not condition:
        raise RuntimeError("DIRECTION_10_CROSS_HEAD_" + reason)


def _same(actual, expected, reason):
    _require(actual.dtype == expected.dtype and actual.shape == expected.shape
             and np.array_equal(actual, expected), reason)


def ten_minute_cohorts(targets, native_folds, controls, fold_specs, *, cutoff_ns):
    """Join exact saved parents and build fixed mature ten-minute cohorts."""
    ten = saved_horizons.derive_ten_minute_direction(targets, cutoff_ns=cutoff_ns)
    _require(ten["control15_bps_rows"] > 0
             and ten["control15"]["resolved_signs_checked"] > 0, "CONTROL15_EMPTY")
    target = ten["arrays"]
    parent = targets["parent_rows"]
    _require(np.array_equal(parent, np.arange(len(parent), dtype=np.int64)), "TARGET_PARENTS")
    ctrl_parent = controls["parent_rows"]
    _require(ctrl_parent.dtype == np.int64 and len(ctrl_parent) > 0
             and np.all(np.diff(ctrl_parent) == 1)
             and ctrl_parent[0] >= 0 and ctrl_parent[-1] < len(parent), "CONTROL_PARENTS")
    for key in ("decision_ns", "terminal_time_ns", "observed", "sign"):
        _same(controls[key], targets[key][ctrl_parent], "CONTROL_TARGET_" + key)
    _require(len(native_folds) == len(fold_specs) == 4, "FOLD_COUNT")
    _require([s["fold_id"] for s in fold_specs] == [1, 2, 3, 4], "FOLD_ORDER")
    result = []
    for native, spec in zip(native_folds, fold_specs):
        fid, left, right, end = (spec[k] for k in
                                  ("fold_id", "left", "right", "holdout_end_ns"))
        _require(type(fid) is int and type(left) is int and type(right) is int
                 and type(end) is int and 0 <= left < right <= len(parent), "FOLD_SPEC")
        rows = np.arange(left, right, dtype=np.int64)
        _same(native["parent_rows"], rows, "NATIVE_PARENTS")
        for key in ("bar_start_ns", "decision_ns", "terminal_time_ns", "observed", "sign"):
            _same(native[key], targets[key][rows], "NATIVE_TARGET_" + key)
        at = np.searchsorted(ctrl_parent, rows)
        _require(np.all(at < len(ctrl_parent))
                 and np.array_equal(ctrl_parent[at], rows), "REFERENCE_JOIN")
        _require(np.all(controls["fold_id"][at] == fid)
                 and np.all(controls["holdout_end_ns_exclusive"][at] == end), "REFERENCE_FOLD")
        native_scores = native["logit_up"]
        _require(native_scores.dtype == np.float32 and native_scores.shape == (len(rows), 3)
                 and np.isfinite(native_scores).all(), "NATIVE_SCORE")
        ref_fields = {"et252": "saved_ET_FULL_probability_up",
                      "r1": "saved_R1_logit_up", "ref259": "logit_up"}
        for name, field in ref_fields.items():
            value = controls[field]
            _require(value.dtype == np.float64 and value.shape == (len(ctrl_parent), 3)
                     and np.isfinite(value).all(), "REFERENCE_SCORE_" + name)
        et = controls[ref_fields["et252"]][at]
        _require(np.all((et >= 0.0) & (et <= 1.0)), "REFERENCE_ET_DOMAIN")
        seen = target["observed_chain"][rows]
        resolved = target["label_resolved"][rows]
        nonzero = target["sign"][rows] != 0
        mature = target["terminal_time_ns"][rows] < end
        mask = seen & resolved & nonzero & mature
        _require(np.any(mask), "EMPTY_FOLD")
        y = (target["sign"][rows][mask] > 0).astype(np.int8)
        _require(np.any(y == 0) and np.any(y == 1), "ONE_CLASS_FOLD")
        score = {}
        for head, column in _SCORE_COLUMNS:
            score[head] = {
                "native": np.ascontiguousarray(native_scores[mask, column], dtype=np.float64),
                **{name: np.ascontiguousarray(controls[field][at][mask, column], dtype=np.float64)
                   for name, field in ref_fields.items()},
            }
        bps_ok = mask & target["bps_sign_consistent"][rows]
        result.append({
            "fold_id": fid,
            "parent_rows": rows[mask],
            "decision_ns": native["decision_ns"][mask],
            "labels": y,
            "scores": score,
            "bps": target["mid_return_bps_approx"][rows][bps_ok],
            "counts": {"parents": len(rows), "chain_observed": int(seen.sum()),
                       "rounding_ambiguous": int((seen & ~resolved).sum()),
                       "certified_zero": int((seen & resolved & ~nonzero).sum()),
                       "fold_end_censored": int((seen & resolved & nonzero & ~mature).sum()),
                       "eligible": int(mask.sum()), "up": int(y.sum()),
                       "down": int(len(y) - y.sum()), "bps_sign_disagreement": int((mask & ~bps_ok).sum())},
        })
    return {"cohorts": result, "control15": ten["control15"],
            "control15_bps_rows": ten["control15_bps_rows"],
            "control15_bps_max_abs_error": ten["control15_bps_max_abs_error"]}


def _block_ids(decision_ns, scheme):
    days = decision_ns // _DAY_NS
    if scheme == "utc_day":
        value = days
    elif scheme == "utc_week_monday":
        value = (days - 4) // 7
    else:
        raise RuntimeError("DIRECTION_10_CROSS_HEAD_BLOCK_SCHEME")
    return np.unique(value, return_inverse=True)


def rank_with_block_uncertainty(cohorts, *, scheme, draws=4000, seed=20261027):
    """Within-fold pair-weighted AUC and paired fixed-block bootstrap."""
    _require(type(draws) is int and draws > 0 and type(seed) is int, "BOOTSTRAP_ARGS")
    rng = np.random.RandomState(seed)
    point_wins = {(h, a): 0 for h, _ in _SCORE_COLUMNS for a in _ARMS}
    draw_wins = {key: np.zeros(draws, np.float64) for key in point_wins}
    point_pairs = 0
    draw_pairs = np.zeros(draws, np.float64)
    details = []
    for cohort in cohorts:
        labels = cohort["labels"]
        days, inverse = _block_ids(cohort["decision_ns"], scheme)
        _require(len(days) >= 8, "FEWER_THAN_EIGHT_BLOCKS")
        block_ids = np.ascontiguousarray(inverse, dtype=np.int64)
        weights = rng.multinomial(len(days), np.full(len(days), 1.0 / len(days)), size=draws)
        ones = np.ones((1, len(days)), np.int64)
        fold_detail = {"fold_id": cohort["fold_id"], "blocks": len(days), "auc": {}}
        first = True
        for head, _ in _SCORE_COLUMNS:
            fold_detail["auc"][head] = {}
            for arm in _ARMS:
                summary = block_rank.block_rank_pairs(labels, cohort["scores"][head][arm], block_ids)
                w, pair = block_rank.weighted_rank_pairs(summary, weights)
                p, onepair = block_rank.weighted_rank_pairs(summary, ones)
                if first:
                    point_pairs += int(onepair[0])
                    draw_pairs += pair
                    first = False
                point_wins[(head, arm)] += int(p[0])
                draw_wins[(head, arm)] += w
                fold_detail["auc"][head][arm] = float(p[0] / (2.0 * onepair[0]))
        details.append(fold_detail)
    _require(point_pairs > 0 and np.all(draw_pairs > 0), "BOOTSTRAP_EMPTY_PAIRS")
    points = {head: {arm: point_wins[(head, arm)] / (2.0 * point_pairs)
                     for arm in _ARMS} for head, _ in _SCORE_COLUMNS}
    draws_auc = {key: value / (2.0 * draw_pairs) for key, value in draw_wins.items()}
    intervals = {}
    for head, first, second in _PRIMARY:
        delta = draws_auc[(head, first)] - draws_auc[(head, second)]
        lo, hi = np.quantile(delta, [0.005, 0.995])
        intervals[head] = {"point_difference": float(points[head][first] - points[head][second]),
                           "low": float(lo), "high": float(hi), "level": 0.99}
    return {"scheme": scheme, "draws": draws, "seed": seed, "point_auc": points,
            "primary_native_minus_et252": intervals, "per_fold": details,
            "within_fold_pairs": point_pairs}


def build_aggregate(joined, *, draws=4000):
    """Return only aggregate diagnostics; no trading or model authority."""
    cohorts = joined["cohorts"]
    day = rank_with_block_uncertainty(cohorts, scheme="utc_day", draws=draws, seed=20261027)
    week = rank_with_block_uncertainty(cohorts, scheme="utc_week_monday", draws=draws, seed=20261028)
    for head, _ in _SCORE_COLUMNS:
        for arm in _ARMS:
            _require(day["point_auc"][head][arm] == week["point_auc"][head][arm], "POINT_SCHEME_PARITY")
    bps = np.concatenate([c["bps"] for c in cohorts])
    _require(len(bps) > 0 and np.isfinite(bps).all(), "BPS_EMPTY")
    return {
        "schema_version": "gx1_saved_cross_head_ten_minute_diagnostic_v1",
        "scope": "FOUR_CHRONOLOGICAL_TRAIN_DEVELOPMENT_FOLDS",
        "target_horizon_minutes": 10,
        "source_score_horizons_minutes": [5, 15],
        "fold_counts": [{"fold_id": c["fold_id"], **c["counts"]} for c in cohorts],
        "control15": joined["control15"],
        "control15_bps_rows": joined["control15_bps_rows"],
        "control15_bps_max_abs_error": joined["control15_bps_max_abs_error"],
        "absolute_return_bps_approx_quantiles_10_50_90": np.quantile(np.abs(bps), [0.1, 0.5, 0.9]).tolist(),
        "bps_rows": len(bps),
        "day_block_bootstrap": day,
        "week_block_bootstrap": week,
        "raw_data_reads": 0, "native_forwards": 0, "optimizer_steps": 0,
        "score_trained_for_ten_minutes": False, "test_data_used": False,
        "prediction_or_economic_edge_qualified": False,
    }
