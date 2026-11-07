"""Pure paired week-block review of M5 certified direction in development TRAIN.

The model recipe, score folds and this gate must be fixed before model fitting.
This calculation grants no TEST, Entry, trade or economic authority.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from gx1.contracts.market_state_five_endpoint_contract_v1 import TIMEFRAMES, VALID_BITS

SEED = 20261107
BOOTSTRAP_REPS = 2000
WEEK_NS = 7 * 24 * 60 * 60 * 1_000_000_000
MONDAY_1970_NS = 4 * 24 * 60 * 60 * 1_000_000_000
_REQUIRED = frozenset(("fold_id", "decision_ns", "allowed", "eligible", "mature",
                       "model_probabilities", "prior_probabilities"))


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError("MARKET_STATE_TABULAR_WEEKLY_" + code)


def _probabilities(value: np.ndarray, rows: int) -> bool:
    return (isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64)
            and value.shape == (rows, len(TIMEFRAMES), 3)
            and bool(np.isfinite(value).all())
            and bool(np.all((value >= 0) & (value <= 1)))
            and bool(np.allclose(value.astype(np.float64).sum(axis=-1), 1.0,
                                 rtol=0, atol=1e-6)))


def _fold_week_sums(item: Mapping, fold_id: int) -> tuple[dict, np.ndarray, np.ndarray]:
    _need(isinstance(item, Mapping) and set(item) == _REQUIRED
          and item["fold_id"] == fold_id, "FOLD_ID")
    decisions = item["decision_ns"]
    _need(isinstance(decisions, np.ndarray) and decisions.dtype == np.int64
          and decisions.ndim == 1 and len(decisions) > 0
          and bool(np.all(decisions > 0)) and bool(np.all(np.diff(decisions) > 0)),
          "DECISION_AXIS")
    rows = len(decisions)
    bits, eligible, mature = (item[name] for name in ("allowed", "eligible", "mature"))
    _need(isinstance(bits, np.ndarray) and bits.dtype == np.uint8
          and bits.shape == (rows, len(TIMEFRAMES)) and bool(np.isin(bits, VALID_BITS).all())
          and all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                  and mask.shape == bits.shape for mask in (eligible, mature)),
          "TARGET_MASKS")
    model, prior = item["model_probabilities"], item["prior_probabilities"]
    _need(_probabilities(model, rows) and _probabilities(prior, rows), "PROBABILITIES")
    selected = eligible[:, 0] & mature[:, 0] & np.isin(bits[:, 0], (1, 4))
    _need(bool(selected.any()), "ZERO_CERTIFIED_SIDE")
    outcome_up = (bits[selected, 0] == 4).astype(np.float64)
    probabilities = []
    for value in (model, prior):
        side = value[selected, 0, :].astype(np.float64, copy=False)
        denominator = side[:, 0] + side[:, 2]
        _need(bool(np.all(denominator > 0)), "UNDEFINED_CONDITIONAL_SIDE")
        probabilities.append(side[:, 2] / denominator)
    delta = np.square(probabilities[1] - outcome_up) - np.square(probabilities[0] - outcome_up)
    weeks = np.floor_divide(decisions[selected] - MONDAY_1970_NS, WEEK_NS)
    week_ids, inverse = np.unique(weeks, return_inverse=True)
    sums = np.bincount(inverse, weights=delta, minlength=len(week_ids)).astype(np.float64)
    counts = np.bincount(inverse, minlength=len(week_ids)).astype(np.int64)
    _need(bool(np.isfinite(sums).all()) and bool(np.all(counts > 0))
          and int(counts.sum()) == int(selected.sum()), "WEEK_AGGREGATION")
    report = {"fold_id": fold_id, "side_count": int(selected.sum()),
              "week_count": len(week_ids), "week_ids": week_ids.tolist(),
              "delta_sum": float(sums.sum(dtype=np.float64)),
              "delta_mean": float(sums.sum(dtype=np.float64) / counts.sum())}
    return report, sums, counts


def paired_m5_weekly_review(folds: Sequence[Mapping]) -> dict:
    """Resample UTC Monday weeks inside each of five fixed chronological folds."""
    _need(isinstance(folds, (tuple, list)) and len(folds) == 5, "FOLD_COUNT")
    reports, blocks = [], []
    previous_last = -1
    for fold_id, item in enumerate(folds, 1):
        report, sums, counts = _fold_week_sums(item, fold_id)
        decisions = item["decision_ns"]
        _need(int(decisions[0]) > previous_last, "FOLD_CHRONOLOGY")
        previous_last = int(decisions[-1])
        reports.append(report)
        blocks.append((sums, counts))
    total_sum = sum(report["delta_sum"] for report in reports)
    total_count = sum(report["side_count"] for report in reports)
    _need(total_count > 0, "TOTAL_SUPPORT")
    pooled = float(total_sum / total_count)
    rng = np.random.default_rng(SEED)
    samples = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for iteration in range(BOOTSTRAP_REPS):
        sample_sum, sample_count = 0.0, 0
        for sums, counts in blocks:
            draw = rng.integers(0, len(sums), size=len(sums))
            sample_sum += float(sums[draw].sum(dtype=np.float64))
            sample_count += int(counts[draw].sum(dtype=np.int64))
        samples[iteration] = sample_sum / sample_count
    lower, upper = np.quantile(samples, (0.025, 0.975), method="linear")
    positive_folds = sum(report["delta_mean"] > 0 for report in reports)
    passed = bool(pooled > 0 and positive_folds >= 4 and lower > 0)
    return {"schema_version": "gx1_market_state_tabular_m5_weekly_review_v1",
            "scope": "development_TRAIN_M5_certified_side_only_no_economic_edge",
            "metric": "paired_conditional_side_Brier_prior_minus_model",
            "folds": reports, "side_count": total_count,
            "pooled_delta": pooled, "positive_fold_count": positive_folds,
            "bootstrap": {"unit": "UTC_Monday_week_within_fold", "seed": SEED,
                          "replicates": BOOTSTRAP_REPS, "ci_level": 0.95,
                          "lower": float(lower), "upper": float(upper)},
            "positive_train_diagnostic": passed,
            "decision": ("POSITIVE_DEVELOPMENT_TRAIN_DIAGNOSTIC" if passed
                         else "NO_POSITIVE_GATE")}
