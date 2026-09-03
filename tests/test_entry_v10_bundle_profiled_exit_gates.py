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
        lambda value, *, expected_rows, context, **kwargs: calls.append(
            (value, expected_rows, context, kwargs)
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
        lambda value, *, expected_rows, context, **kwargs: calls.append(
            (value, expected_rows, context, kwargs)
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
        (
            selected,
            64,
            "ENTRY_BUNDLE_SELECTED_CHECKPOINT",
            {"allow_static_feature_gate_provisional": False},
        ),
        (
            full,
            256,
            "ENTRY_BUNDLE_FULL_TRAJECTORY",
            {"allow_static_feature_gate_provisional": False},
        ),
    ]


def test_candidate_bundle_rejects_odd_double_sided_population_before_gate_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bundle,
        "require_unified_exit_gate_evidence",
        lambda **_kwargs: pytest.fail("gate evidence must not run"),
    )

    with pytest.raises(RuntimeError, match="UNIFIED_EXIT_GATE_ROWS_INVALID"):
        bundle._require_profiled_unified_exit_gate_evidence(
            training_profile="candidate",
            exit_validation={"unified_exit_population_rows": 127},
            full_trajectory_validation={"population_rows": 512},
        )


def test_unknown_profile_fails_closed_before_gate_evidence_can_be_accepted() -> None:
    with pytest.raises(RuntimeError, match="TRAINING_PROFILE_INVALID"):
        bundle._require_profiled_unified_exit_gate_evidence(
            training_profile="unknown",
            exit_validation={"unified_exit_population_rows": 1},
            full_trajectory_validation={"population_rows": 1},
        )


def test_candidate_static_exit_gate_requires_exact_provisional_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        bundle,
        "require_unified_exit_gate_evidence",
        lambda _value, *, allow_static_feature_gate_provisional, **_kwargs: calls.append(
            bool(allow_static_feature_gate_provisional)
        ),
    )
    static_failure = "family_tf_feature_gate constant/dead indices=[7]"
    selected = {
        "unified_exit_population_rows": 128,
        "exit_cooperation_gate_health_failures": [static_failure],
        "candidate_exit_static_feature_gate_diagnostic": {
            "schema_version": "gx1_candidate_static_feature_gate_provisional_v1",
            "decision": "PROVISIONAL_STATIC_BUT_OPEN_PENDING_INPUT_INFLUENCE",
            "surface": "unified_exit",
            "nonblocking_for_checkpoint_selection_only": True,
            "required_before_bundle_publication": [
                "hash_bound_full_population_raw_input_liveness",
                "selected_checkpoint_direct_input_influence",
            ],
            "failures": [static_failure],
            "blocking_failures_after_disposition": [],
        },
    }

    bundle._require_profiled_unified_exit_gate_evidence(
        training_profile="candidate",
        exit_validation=selected,
        full_trajectory_validation={"population_rows": 512},
    )
    assert calls == [True, True]

    selected["candidate_exit_static_feature_gate_diagnostic"][
        "blocking_failures_after_disposition"
    ] = ["bad"]
    with pytest.raises(RuntimeError, match="CANDIDATE_STATIC_EXIT_GATE_INVALID"):
        bundle._candidate_static_exit_gate_provisional(selected)


def test_candidate_full_trajectory_requires_loaded_online_and_target_state_bindings() -> None:
    online = "a" * 64
    target = "b" * 64
    report = {
        "online_model_state_sha256": online,
        "target_model_state_sha256": target,
        "state_prediction_stream_sha256": "c" * 64,
    }
    bundle._require_candidate_full_trajectory_bindings(
        report,
        selected_online_model_state_sha256=online,
        target_model_state_sha256=target,
    )

    for field, value in (
        ("online_model_state_sha256", "d" * 64),
        ("target_model_state_sha256", "d" * 64),
        ("state_prediction_stream_sha256", "not-a-sha"),
    ):
        broken = dict(report)
        broken[field] = value
        with pytest.raises(RuntimeError, match="FULL_TRAJECTORY_BINDING_INVALID"):
            bundle._require_candidate_full_trajectory_bindings(
                broken,
                selected_online_model_state_sha256=online,
                target_model_state_sha256=target,
            )
