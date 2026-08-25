from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_causal_m1_target_policy_v1 import (
    causal_m1_direction_targets_from_policy,
    causal_m1_direction_diagnostic_outcome_contract,
    fit_causal_m1_target_policy,
    materialize_causal_m1_auxiliary_outcomes,
    require_causal_m1_target_policy,
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("ascii")).hexdigest()


def _m1(rows: int = 1_100) -> pd.DataFrame:
    time = pd.date_range("2024-01-01T00:00:00Z", periods=rows, freq="min")
    # Repeating movement deliberately gives TRAIN both executable long and
    # short opportunities; the sizing ECDF must not quietly fit one side only.
    index = np.arange(rows, dtype=np.float64)
    base = 2000.0 + np.sin(index / 11.0) * 8.0 + np.sin(index / 23.0) * 3.0
    bid = base
    ask = base + 0.2
    return pd.DataFrame(
        {
            "time": time,
            "bid_open": bid,
            "bid_high": bid + 0.3,
            "bid_low": bid - 0.3,
            "ask_open": ask,
            "ask_high": ask + 0.3,
            "ask_low": ask - 0.3,
        }
    )


def _fit() -> dict[str, object]:
    m1 = _m1()
    m5 = pd.DataFrame({"time": m1["time"].iloc[::5].reset_index(drop=True)})
    return fit_causal_m1_target_policy(
        closed_m5=m5,
        closed_m1=m1,
        train_start=m5.loc[0, "time"],
        train_end=pd.Timestamp("2024-01-01T08:20:00Z"),
        source_parquet_sha256=_sha("m5"),
        tape_provenance_sha256=_sha("tape"),
        m1_source_sha256=_sha("m1"),
    )


def test_fit_is_m1_bound_and_self_validating() -> None:
    policy = _fit()
    assert require_causal_m1_target_policy(policy) == policy
    assert policy["target_contract"]["long_entry_price"] == (
        "ask_open_first_authoritative_m1_at_or_after_entry_decision"
    )
    assert policy["m1_source_sha256"] == _sha("m1")
    assert policy["fit_population_rows"] > 2
    assert policy["path_threshold_population_rows"] == 2 * policy["fit_population_rows"]


def test_fit_rejects_a_gap_when_it_leaves_no_complete_train_population() -> None:
    m1 = _m1().drop(index=250).reset_index(drop=True)
    m5 = pd.DataFrame({"time": pd.date_range("2024-01-01T00:00:00Z", periods=180, freq="5min")})
    with pytest.raises(RuntimeError, match="TRAIN_POPULATION_EMPTY"):
        fit_causal_m1_target_policy(
            closed_m5=m5,
            closed_m1=m1,
            train_start=m5.loc[0, "time"],
            train_end=pd.Timestamp("2024-01-01T08:20:00Z"),
            source_parquet_sha256=_sha("m5"),
            tape_provenance_sha256=_sha("tape"),
            m1_source_sha256=_sha("m1"),
        )


def test_direction_target_is_a_frozen_offline_policy() -> None:
    policy = _fit()
    result = causal_m1_direction_targets_from_policy(
        policy=policy,
        long_executable_pnl_bps=np.array([20.0, -20.0]),
        short_executable_pnl_bps=np.array([-20.0, 20.0]),
    )
    assert result["side"].tolist() == [0, 1]
    assert result["trade"].tolist() == [True, True]
    assert result["long_score_bps"].tolist() == [20.0, -20.0]
    assert result["short_score_bps"].tolist() == [-20.0, 20.0]
    assert causal_m1_direction_diagnostic_outcome_contract(policy)[
        "diagnostic_outcome_label_source"
    ] == "train_fitted_exact_m1_fill_executable_pnl_at_selected_horizon"


def test_materialized_auxiliary_evidence_retains_invalid_m1_rows() -> None:
    policy = _fit()
    m1 = _m1().drop(index=8).reset_index(drop=True)
    output = materialize_causal_m1_auxiliary_outcomes(
        policy=policy,
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True)
        ),
        closed_m1=m1,
    )
    assert bool(output.loc[0, "outcome_valid"]) is False
    assert np.isnan(output.loc[0, "v11_pnl_long_at_dir_horizon_bps"])
    assert output.loc[0, "direction_side"] == -1
