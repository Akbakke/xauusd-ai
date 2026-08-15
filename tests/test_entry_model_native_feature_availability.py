from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_feature_availability_v1 import (
    SELECTION_POLICY,
    _mandatory_family_counts,
    fit_feature_availability_contract,
    require_feature_availability_contract,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
)


_CANDIDATES = sorted(
    [
        "ctx_cont.bars_since_swing_high",
        "ctx_cont.bars_since_swing_low",
        "ctx_cont.dist_to_R1_atr",
        "ctx_cont.D1_dist_from_ema200_atr",
        "ctx_cont.atr_bps",
        "ctx_cont.d1_close_pct_in_20day_range_canon_v2",
        "ctx_cont.d1_rsi14_canon_v2",
        "ctx_cont.hour_cos",
        "ctx_cont.hour_sin",
        "ctx_cont.swing_high_break_event",
        "ctx_cont.close_distance_below_high_range_fraction",
        "candle.raw_body_signed_range",
    ]
)


def _inputs() -> tuple[np.ndarray, list[str], pd.DatetimeIndex, np.ndarray]:
    rows = 16
    times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    base = np.arange(rows, dtype=np.float32)
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(_CANDIDATES, start=1):
        columns[name] = base * np.float32(index + 1) + np.float32(index**2)
    rare = np.zeros(rows, dtype=np.float32)
    rare[-1] = 1.0
    columns["ctx_cont.swing_high_break_event"] = rare
    matrix = np.column_stack([columns[name] for name in _CANDIDATES]).astype(
        np.float32
    )
    target = np.square(base.astype(np.float64)) - 3.0 * base
    return matrix, list(_CANDIDATES), times, target


def _fit(*, target: np.ndarray | None = None) -> dict:
    matrix, names, times, default_target = _inputs()
    return fit_feature_availability_contract(
        matrix=matrix,
        names=names,
        times=times,
        train_start=times[2],
        train_end=times[-1],
        diagnostic_target=default_target if target is None else target,
    )


def test_all_nonconstant_unique_candidates_are_available_without_top_k() -> None:
    payload = _fit()

    assert payload["selection_policy"] == SELECTION_POLICY
    assert payload["selection_policy"]["fixed_top_k"] is False
    assert payload["selection_policy"]["score_cutoff"] is False
    assert payload["selection_policy"]["family_quota"] is False
    assert payload["diagnostic_target_affects_selection"] is False
    assert payload["available_feature_count"] == len(_CANDIDATES)
    assert payload["excluded_features"] == []
    assert set(payload["family_coverage"]) == set(MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert all(
        row["candidate"] > 0 for row in payload["family_coverage"].values()
    )
    assert require_feature_availability_contract(payload) == payload


def test_exact_duplicate_and_constant_fail_closed_with_repair_evidence() -> None:
    matrix, names, times, target = _inputs()
    constant = matrix.copy()
    constant[:, names.index("ctx_cont.hour_cos")] = 0.25
    with pytest.raises(
        RuntimeError,
        match=r'"train_constant":\["ctx_cont.hour_cos"\]',
    ):
        fit_feature_availability_contract(
            matrix=constant,
            names=names,
            times=times,
            train_start=times[2],
            train_end=times[-1],
            diagnostic_target=target,
        )

    duplicate = matrix.copy()
    duplicate[:, names.index("ctx_cont.bars_since_swing_low")] = duplicate[
        :, names.index("ctx_cont.bars_since_swing_high")
    ]
    with pytest.raises(RuntimeError, match="exact_duplicate_of"):
        fit_feature_availability_contract(
            matrix=duplicate,
            names=names,
            times=times,
            train_start=times[2],
            train_end=times[-1],
            diagnostic_target=target,
        )


def test_single_observed_rare_event_remains_available() -> None:
    payload = _fit()
    row = next(
        row
        for row in payload["features"]
        if row["name"] == "ctx_cont.swing_high_break_event"
    )

    assert row["nonzero_rows"] == 1
    assert row["distinct_value_count"] == 2
    assert row["decision"] == "available"


def test_target_shuffle_changes_diagnostics_but_not_availability() -> None:
    matrix, names, times, target = _inputs()
    original = _fit(target=target)
    shuffled = _fit(target=target[::-1].copy())

    assert original["selection_sha256"] == shuffled["selection_sha256"]
    assert original["available_features"] == shuffled["available_features"]
    assert original["excluded_features"] == shuffled["excluded_features"]
    assert original["diagnostic_target_sha256"] != shuffled[
        "diagnostic_target_sha256"
    ]
    assert matrix.shape[1] == len(names)
    assert times[-1] == pd.Timestamp(original["source_time_max_utc"])


def test_shuffled_rows_and_future_rows_fail_closed() -> None:
    matrix, names, times, target = _inputs()
    permutation = np.arange(len(times))[::-1]
    with pytest.raises(RuntimeError, match="CLOCK_INVALID"):
        fit_feature_availability_contract(
            matrix=matrix[permutation],
            names=names,
            times=times[permutation],
            train_start=times[2],
            train_end=times[-1],
            diagnostic_target=target[permutation],
        )

    future_time = times[-1] + pd.Timedelta(minutes=5)
    with pytest.raises(RuntimeError, match="FUTURE_OR_WINDOW_INVALID"):
        fit_feature_availability_contract(
            matrix=np.vstack([matrix, matrix[-1]]),
            names=names,
            times=times.append(pd.DatetimeIndex([future_time])),
            train_start=times[2],
            train_end=times[-1],
            diagnostic_target=np.append(target, target[-1]),
        )


def test_candidate_pool_mutation_breaks_hash_binding() -> None:
    payload = _fit()
    mutated = copy.deepcopy(payload)
    mutated["candidate_pool"][0] = "ctx_cont.mutated_candidate"

    with pytest.raises(RuntimeError, match="CONTRACT_CANDIDATES_INVALID"):
        require_feature_availability_contract(mutated)
    with pytest.raises(RuntimeError, match="CANDIDATE_POOL_BINDING_INVALID"):
        require_feature_availability_contract(
            payload,
            expected_candidate_pool_sha256="f" * 64,
        )


def test_family_starvation_fails_closed() -> None:
    """A family absent from the WHOLE model input must fail closed.

    Starvation is judged over mandatory ∪ candidate (rule 2g: the gate must
    look at the surface the model actually receives). A family that holds
    every one of its fields as mandatory therefore cannot be starved by an
    empty candidate remainder — dropping a candle candidate proves nothing.
    The families that can still starve are exactly those with no mandatory
    representation, so the fixture drops one of those.
    """

    starvable = sorted(
        name
        for name, count in _mandatory_family_counts().items()
        if count == 0
    )
    assert starvable, "no family can starve; this guard would be vacuous"
    matrix, names, times, target = _inputs()
    dropped = [
        name
        for name in names
        if classify_entry_specialist_feature(name) in set(starvable)
    ]
    assert dropped, "fixture holds no candidate of a starvable family"
    drop_index = [names.index(name) for name in dropped]
    reduced_names = [name for name in names if name not in set(dropped)]
    reduced_matrix = np.delete(matrix, drop_index, axis=1)

    with pytest.raises(RuntimeError, match="CANDIDATE_FAMILY_STARVED"):
        fit_feature_availability_contract(
            matrix=reduced_matrix,
            names=reduced_names,
            times=times,
            train_start=times[2],
            train_end=times[-1],
            diagnostic_target=target,
        )


def test_selection_and_coverage_mutations_fail_closed() -> None:
    payload = _fit()
    selected_mutation = copy.deepcopy(payload)
    selected_mutation["available_features"] = selected_mutation[
        "available_features"
    ][1:]
    with pytest.raises(RuntimeError, match="SELECTION_BINDING_INVALID"):
        require_feature_availability_contract(selected_mutation)

    coverage_mutation = copy.deepcopy(payload)
    first_family = next(iter(coverage_mutation["family_coverage"]))
    coverage_mutation["family_coverage"][first_family]["candidate"] += 1
    with pytest.raises(RuntimeError, match="FAMILY_COVERAGE_INVALID"):
        require_feature_availability_contract(coverage_mutation)
