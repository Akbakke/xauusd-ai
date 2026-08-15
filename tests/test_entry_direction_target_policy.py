from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_direction_target_policy_v1 import (
    canonical_entry_target_policy_sha256,
    entry_direction_targets_from_policy,
    fit_entry_direction_target_policy,
    require_entry_direction_target_policy,
)
from tests.entry_direction_target_policy_support import (
    entry_direction_target_policy_fixture,
)


def _fit_frame() -> pd.DataFrame:
    time = pd.date_range("2020-01-01T00:00:00Z", periods=1500, freq="5min")
    phase = np.arange(len(time), dtype=np.float64)
    mid = 1800.0 + 1.5 * np.sin(phase / 9.0) + 0.8 * np.sin(phase / 31.0)
    return pd.DataFrame(
        {"time": time, "bid_close": mid - 0.05, "ask_close": mid + 0.05}
    )


def _fit(frame: pd.DataFrame) -> dict[str, object]:
    time = pd.DatetimeIndex(frame["time"])
    return fit_entry_direction_target_policy(
        closed_m5=frame,
        train_start=time[0],
        train_end=time[1100] + pd.Timedelta(minutes=5),
        source_parquet_sha256="1" * 64,
        tape_provenance_sha256="2" * 64,
    )


def test_entry_direction_policy_is_deterministic_and_train_only() -> None:
    frame = _fit_frame()
    baseline = _fit(frame)
    assert _fit(frame.copy()) == baseline

    mutated = frame.copy()
    mutated.loc[1200:, "bid_close"] += 0.2
    mutated.loc[1200:, "ask_close"] += 0.2
    assert _fit(mutated) == baseline
    assert baseline["val_test_rows_used_for_fit"] == 0
    assert baseline["fit_scope"] == "TRAIN_ONLY"
    assert baseline["handwritten_path_thresholds"] is False
    assert baseline["path_threshold_population_rows"] == (
        2 * baseline["fit_population_rows"]
    )
    thresholds = baseline["path_aux_thresholds_bps"]
    for name in ("mfe", "mae", "path"):
        assert thresholds[f"{name}_low_bps"] <= thresholds[f"{name}_median_bps"]
        assert thresholds[f"{name}_median_bps"] <= thresholds[f"{name}_high_bps"]


def test_entry_direction_policy_rejects_hash_and_knee_tampering() -> None:
    policy = entry_direction_target_policy_fixture()
    tampered_hash = deepcopy(policy)
    tampered_hash["tradable_edge_floor_bps"] = float(
        tampered_hash["tradable_edge_floor_bps"]
    ) + 0.1
    with pytest.raises(RuntimeError, match="ENTRY_TARGET_POLICY_HASH_INVALID"):
        require_entry_direction_target_policy(tampered_hash)

    tampered_knee = deepcopy(policy)
    tampered_knee["selected_direction_horizon_bars"] = (
        int(tampered_knee["selected_direction_horizon_bars"]) + 1
    )
    without_hash = dict(tampered_knee)
    without_hash.pop("policy_sha256")
    tampered_knee["policy_sha256"] = canonical_entry_target_policy_sha256(
        without_hash
    )
    with pytest.raises(RuntimeError, match="ENTRY_TARGET_POLICY_KNEE_INVALID"):
        require_entry_direction_target_policy(tampered_knee)

    tampered_thresholds = deepcopy(policy)
    tampered_thresholds["path_aux_thresholds_bps"]["mfe_low_bps"] = (
        float(tampered_thresholds["path_aux_thresholds_bps"]["mfe_high_bps"])
        + 1.0
    )
    without_hash = dict(tampered_thresholds)
    without_hash.pop("policy_sha256")
    tampered_thresholds["policy_sha256"] = canonical_entry_target_policy_sha256(
        without_hash
    )
    with pytest.raises(RuntimeError, match="ENTRY_TARGET_POLICY_PATH_THRESHOLDS_INVALID"):
        require_entry_direction_target_policy(tampered_thresholds)


def test_dataset_builder_has_no_handwritten_path_threshold_constants() -> None:
    from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder

    source = open(builder.__file__, encoding="utf-8").read()
    for retired in (
        "HARD_NEG_LONG_MIN_MFE_BPS",
        "DEAD_LONG_MAX_MFE_BPS",
        "TEASER_LONG_MAX_MFE_BPS",
        "CLEAN_EDGE_LONG_MFE_MIN_BPS",
        "SURVIVAL_LONG_MFE_MIN_BPS",
    ):
        assert retired not in source
    assert 'direction_target_policy["path_aux_thresholds_bps"]' in source


def test_entry_direction_target_application_has_no_manual_utility_formula() -> None:
    policy = entry_direction_target_policy_fixture()
    floor = float(policy["tradable_edge_floor_bps"])
    result = entry_direction_targets_from_policy(
        policy=policy,
        long_executable_pnl_bps=[floor, floor + 1.0, -2.0],
        short_executable_pnl_bps=[-2.0, -2.0, floor + 1.0],
    )
    assert result["long_score_bps"].tolist() == pytest.approx(
        [floor, floor + 1.0, -2.0]
    )
    assert result["direction"].tolist() == [2, 0, 1]
    assert result["trade"].tolist() == [False, True, True]
