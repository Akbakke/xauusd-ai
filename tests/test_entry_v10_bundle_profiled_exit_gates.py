"""Smoke and candidate must apply the documented distinct Exit-gate policies."""

from __future__ import annotations

import pytest

import gx1.models.entry_v10.entry_v10_bundle as bundle


def test_smoke_bundle_preserves_diagnostic_exit_gate_without_candidate_veto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, int, str]] = []
    monkeypatch.setattr(
        bundle,
        "require_unified_exit_gate_evidence",
        lambda value, *, expected_rows, context: calls.append(
            (value, expected_rows, context)
        ),
    )

    bundle._require_profiled_unified_exit_gate_evidence(
        training_profile="smoke",
        exit_validation={"unified_exit_population_rows": 32},
        full_trajectory_validation={"population_rows": 0},
    )

    assert calls == []


def test_candidate_bundle_requires_selected_and_full_exit_gate_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, int, str]] = []
    monkeypatch.setattr(
        bundle,
        "require_unified_exit_gate_evidence",
        lambda value, *, expected_rows, context: calls.append(
            (value, expected_rows, context)
        ),
    )
    selected = {"unified_exit_population_rows": 128}
    full = {"population_rows": 512}

    bundle._require_profiled_unified_exit_gate_evidence(
        training_profile="candidate",
        exit_validation=selected,
        full_trajectory_validation=full,
    )

    assert calls == [
        (selected, 128, "ENTRY_BUNDLE_SELECTED_CHECKPOINT"),
        (full, 512, "ENTRY_BUNDLE_FULL_TRAJECTORY"),
    ]


def test_unknown_profile_fails_closed_before_gate_evidence_can_be_accepted() -> None:
    with pytest.raises(RuntimeError, match="TRAINING_PROFILE_INVALID"):
        bundle._require_profiled_unified_exit_gate_evidence(
            training_profile="unknown",
            exit_validation={"unified_exit_population_rows": 1},
            full_trajectory_validation={"population_rows": 1},
        )
