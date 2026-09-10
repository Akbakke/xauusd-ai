from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    build_market_closure_authority,
    build_project_inferred_closure_policy,
    closure_intervals_by_gap_after_row,
    exact_schedule_from_project_policy,
    require_project_inferred_closure_policy,
)


def _segments(pairs: list[tuple[str, str]]) -> pd.DatetimeIndex:
    values: list[pd.Timestamp] = []
    for previous, successor in pairs:
        left = pd.Timestamp(previous)
        right = pd.Timestamp(successor)
        values.extend((left - pd.Timedelta(minutes=1), left, right, right + pd.Timedelta(minutes=1)))
    return pd.DatetimeIndex(sorted(set(values)))


def _train_clock() -> pd.DatetimeIndex:
    return _segments(
        [
            ("2026-06-01T20:59Z", "2026-06-01T22:00Z"),
            ("2026-06-08T20:59Z", "2026-06-08T22:00Z"),
            ("2026-06-15T20:59Z", "2026-06-15T22:00Z"),
            ("2026-06-05T21:59Z", "2026-06-07T22:00Z"),
            ("2026-06-12T21:59Z", "2026-06-14T22:00Z"),
            # Under one hour: must never become an allowed closure rule.
            ("2026-06-17T10:00Z", "2026-06-17T10:31Z"),
        ]
    )


def _policy() -> dict:
    return build_project_inferred_closure_policy(
        train_m1_times=_train_clock(),
        train_m1_source_sha256="a" * 64,
        minimum_daily_support=3,
        minimum_weekend_support=2,
    )


def test_train_clock_fit_qualifies_only_recurring_exact_structural_rules() -> None:
    policy = _policy()
    assert policy["decision"] == "PASS"
    assert policy["fit_splits"] == ["train"]
    assert policy["validation_or_test_used_for_fit"] is False
    assert {rule["closure_kind"] for rule in policy["rules"]} == {
        "daily_maintenance",
        "weekend",
    }
    assert {rule["support_count"] for rule in policy["rules"]} == {2, 3}
    assert policy["false_positive_guards"]["under_one_hour_is_unknown"] is True
    require_project_inferred_closure_policy(
        policy, expected_train_m1_source_sha256="a" * 64
    )


def test_val_is_apply_only_and_unknown_holiday_or_short_gap_censors() -> None:
    clock = _segments(
        [
            ("2026-06-22T20:59Z", "2026-06-22T22:00Z"),
            ("2026-06-26T21:59Z", "2026-06-28T22:00Z"),
            # Thursday to Friday is not a recurring weekend signature.
            ("2026-06-25T12:00Z", "2026-06-26T13:01Z"),
            ("2026-06-24T10:00Z", "2026-06-24T10:31Z"),
        ]
    )
    schedule = exact_schedule_from_project_policy(
        policy=_policy(),
        expected_train_m1_source_sha256="a" * 64,
        target_split="val",
        target_m1_times=clock,
    )
    assert schedule["source_method"] == "project_inferred_pretest_closure_policy_v1"
    authority = build_market_closure_authority(
        m1_times=clock,
        m1_source_path=Path("/immutable/val.parquet"),
        m1_source_sha256="b" * 64,
        m1_source_manifest_path=Path("/immutable/val.manifest.json"),
        m1_source_manifest_sha256="c" * 64,
        exact_schedule=schedule,
        exact_schedule_path=Path("/immutable/val.schedule.json"),
        exact_schedule_file_sha256="d" * 64,
    )
    gaps = closure_intervals_by_gap_after_row(authority)
    allowed = [item for item in gaps.values() if item["successor_across_gap_allowed"]]
    unknown = [item for item in gaps.values() if not item["successor_across_gap_allowed"]]
    assert {item["closure_kind"] for item in allowed} == {
        "daily_maintenance",
        "weekend",
    }
    assert len(unknown) >= 2
    assert all(item["classification"] == "unknown_source_absence" for item in unknown)


def test_policy_tamper_and_test_split_fail_closed() -> None:
    policy = _policy()
    policy["rules"][0]["support_count"] = 1
    with pytest.raises(RuntimeError, match="PROJECT_POLICY_INVALID|PROJECT_RULE_INVALID"):
        require_project_inferred_closure_policy(
            policy, expected_train_m1_source_sha256="a" * 64
        )
    with pytest.raises(RuntimeError, match="PROJECT_SPLIT_INVALID"):
        exact_schedule_from_project_policy(
            policy=_policy(),
            expected_train_m1_source_sha256="a" * 64,
            target_split="test",
            target_m1_times=_train_clock(),
        )
