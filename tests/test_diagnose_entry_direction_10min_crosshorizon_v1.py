"""Synthetic-only tests for saved ten-minute cross-horizon scoring."""
import numpy as np
import pytest

from gx1.scripts import diagnose_entry_direction_10min_crosshorizon_v1 as diag
from tests.test_entry_direction_saved_horizons_v1 import saved_prices


def synthetic_join_inputs():
    # Every five minutes cycle through four unequal endpoint prices.
    # Each ten-minute move is nonzero, with both signs within every fold.
    pattern = np.asarray([100.0, 101.0, 100.2, 99.2], np.float64)
    knots = np.arange(0, 806, 5, dtype=np.float64)
    prices = np.interp(np.arange(805), knots, np.resize(pattern, len(knots)))
    targets, cutoff = saved_prices(prices)
    n = len(targets["parent_rows"])
    controls = {name: targets[name].copy() for name in
                ("parent_rows", "decision_ns", "terminal_time_ns", "observed", "sign")}
    controls["fold_id"] = np.zeros(n, np.int8)
    controls["holdout_end_ns_exclusive"] = np.zeros(n, np.int64)
    controls["saved_ET_FULL_probability_up"] = np.full((n, 3), 0.5, np.float64)
    controls["saved_R1_logit_up"] = np.zeros((n, 3), np.float64)
    controls["logit_up"] = np.zeros((n, 3), np.float64)
    natives, specs = [], []
    for fid in (1, 2, 3, 4):
        left, right = 10 + (fid - 1) * 20, 10 + fid * 20
        rows = np.arange(left, right, dtype=np.int64)
        end = int(targets["decision_ns"][right - 1] + 10 * diag.saved_horizons.MINUTE_NS)
        specs.append({"fold_id": fid, "left": left, "right": right, "holdout_end_ns": end})
        controls["fold_id"][rows] = fid
        controls["holdout_end_ns_exclusive"][rows] = end
        native = {key: targets[key][rows].copy() for key in
                  ("parent_rows", "bar_start_ns", "decision_ns", "terminal_time_ns", "observed", "sign")}
        native["logit_up"] = np.zeros((len(rows), 3), np.float32)
        natives.append(native)
    return targets, natives, controls, specs, cutoff


def test_ten_minute_join_respects_fold_end_and_common_parent_identity():
    target, natives, controls, specs, cutoff = synthetic_join_inputs()
    before = {name: values.tobytes() for name, values in target.items()}
    joined = diag.ten_minute_cohorts(target, natives, controls, specs, cutoff_ns=cutoff)
    assert len(joined["cohorts"]) == 4
    assert joined["control15_bps_rows"] > 0
    for spec, cohort in zip(specs, joined["cohorts"]):
        assert cohort["counts"]["fold_end_censored"] == 1
        assert cohort["counts"]["eligible"] == spec["right"] - spec["left"] - 1
        assert np.all(cohort["decision_ns"] + 10 * diag.saved_horizons.MINUTE_NS < spec["holdout_end_ns"])
        assert np.isin(cohort["labels"], (0, 1)).all()
        assert cohort["counts"]["up"] > 0 and cohort["counts"]["down"] > 0
    assert all(target[k].tobytes() == before[k] for k in target)


@pytest.mark.parametrize("fault,match", [
    ("reference_clock", "CONTROL_TARGET_decision_ns"),
    ("native_parent", "NATIVE_PARENTS"),
    ("reference_fold", "REFERENCE_FOLD"),
    ("et_domain", "REFERENCE_ET_DOMAIN"),
])
def test_ten_minute_join_fails_closed_on_mismatch(fault, match):
    target, natives, controls, specs, cutoff = synthetic_join_inputs()
    left = specs[0]["left"]
    if fault == "reference_clock":
        controls["decision_ns"][left] += diag.saved_horizons.MINUTE_NS
    elif fault == "native_parent":
        natives[0]["parent_rows"][0] += 1
    elif fault == "reference_fold":
        controls["fold_id"][left] = 3
    elif fault == "et_domain":
        controls["saved_ET_FULL_probability_up"][left, 1] = 1.5
    with pytest.raises(RuntimeError, match=match):
        diag.ten_minute_cohorts(target, natives, controls, specs, cutoff_ns=cutoff)


def simple_rank_cohorts():
    cohorts = []
    for fid in (1, 2, 3, 4):
        # Two opposite labels per week preserve both classes in every bootstrap draw.
        days = np.repeat(np.arange(8, dtype=np.int64) * 7 + 20000, 2)
        labels = np.tile(np.array([0, 1], np.int8), 8)
        native = labels.astype(np.float64)
        et = 1.0 - native
        scores = {head: {"native": native.copy(), "et252": et.copy(),
                         "r1": np.zeros(len(labels), np.float64), "ref259": np.zeros(len(labels), np.float64)}
                  for head in ("head_5", "head_15")}
        cohorts.append({"fold_id": fid, "decision_ns": days * diag._DAY_NS,
                        "labels": labels, "scores": scores})
    return cohorts


def test_paired_block_auc_exact_for_perfect_reverse_and_ties():
    cohorts = simple_rank_cohorts()
    out = diag.rank_with_block_uncertainty(cohorts, scheme="utc_week_monday", draws=64, seed=5)
    for head in ("head_5", "head_15"):
        assert out["point_auc"][head] == {"native": 1.0, "et252": 0.0, "r1": 0.5, "ref259": 0.5}
        assert out["primary_native_minus_et252"][head]["low"] == 1.0
        assert out["primary_native_minus_et252"][head]["high"] == 1.0
    assert out["within_fold_pairs"] == 4 * 8 * 8
    assert all(fold["blocks"] == 8 for fold in out["per_fold"])


def test_paired_resampling_of_identical_scores_yields_zero_interval():
    cohorts = simple_rank_cohorts()
    for cohort in cohorts:
        for head in ("head_5", "head_15"):
            cohort["scores"][head]["et252"] = cohort["scores"][head]["native"].copy()
    out = diag.rank_with_block_uncertainty(cohorts, scheme="utc_day", draws=64, seed=5)
    for head in ("head_5", "head_15"):
        assert out["primary_native_minus_et252"][head]["point_difference"] == 0.0
        assert out["primary_native_minus_et252"][head]["low"] == 0.0
        assert out["primary_native_minus_et252"][head]["high"] == 0.0


def test_fewer_than_eight_blocks_stops_instead_of_pretending_uncertainty():
    cohorts = simple_rank_cohorts()
    cohort = cohorts[0]
    cohort["decision_ns"] = cohort["decision_ns"][:14]
    cohort["labels"] = cohort["labels"][:14]
    for head in ("head_5", "head_15"):
        for arm in diag._ARMS:
            cohort["scores"][head][arm] = cohort["scores"][head][arm][:14]
    with pytest.raises(RuntimeError, match="FEWER_THAN_EIGHT_BLOCKS"):
        diag.rank_with_block_uncertainty(cohorts, scheme="utc_week_monday", draws=64, seed=5)
