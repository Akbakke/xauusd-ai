from __future__ import annotations

import copy

import pytest

from gx1.contracts.entry_exit_feature_base_v1 import EXIT_MTF_CONTEXT_TIMEFRAMES
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM
from gx1.contracts.unified_exit_input_influence_v1 import (
    require_unified_exit_input_influence,
    unified_exit_input_influence_layout,
)
from tests.unified_exit_input_influence_support import (
    MULTI_TF_CACHE_IDENTITY_SHA256,
    SELECTED_ONLINE_MODEL_STATE_SHA256,
    UNIFIED_EXIT_LIFECYCLE_ROOT_MANIFEST_SHA256,
    VAL_DATA_SHA256,
    passing_unified_exit_input_influence,
)


def _signal_names() -> list[str]:
    return [
        f"model_native_signal_{index:03d}"
        for index in range(MODEL_NATIVE_SIGNAL_DIM)
    ]


def _require(report: dict[str, object], signal_names: list[str]) -> None:
    require_unified_exit_input_influence(
        report,
        ordered_signal_names=signal_names,
        selected_online_model_state_sha256=SELECTED_ONLINE_MODEL_STATE_SHA256,
        val_data_sha256=VAL_DATA_SHA256,
        multi_tf_cache_identity_sha256=MULTI_TF_CACHE_IDENTITY_SHA256,
        unified_exit_lifecycle_root_manifest_sha256=(
            UNIFIED_EXIT_LIFECYCLE_ROOT_MANIFEST_SHA256
        ),
        context="TEST",
    )


def test_unified_exit_influence_requires_each_native_and_mtf_surface() -> None:
    signal_names = _signal_names()
    report = passing_unified_exit_input_influence(signal_names)
    _require(report, signal_names)

    numeric = report["input_ownership"]["numeric"]  # type: ignore[index]
    for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES:
        assert f"seq_{timeframe.lower()}" in numeric
    assert "seq_signal" in numeric
    assert "snap_signal" not in numeric
    assert "entry_decision_representation" in numeric
    assert "exit_path" in numeric

    expected_layout = unified_exit_input_influence_layout(signal_names)
    assert report["numeric_input_count"] == sum(  # type: ignore[index]
        len(owner["tokens"]) for owner in expected_layout["numeric"].values()
    )


def test_unified_exit_influence_rejects_missing_or_dead_evidence() -> None:
    signal_names = _signal_names()
    report = passing_unified_exit_input_influence(signal_names)

    missing_surface = copy.deepcopy(report)
    missing_surface["numeric"].pop("seq_d1")  # type: ignore[index]
    with pytest.raises(RuntimeError, match="numeric surface set"):
        _require(missing_surface, signal_names)

    dead_field = copy.deepcopy(report)
    metric = next(iter(dead_field["numeric"]["seq_h4"]["metrics"]))  # type: ignore[index]
    dead_field["numeric"]["seq_h4"]["metrics"][metric][  # type: ignore[index]
        "max_abs_exit_margin_gradient"
    ] = 0.0
    with pytest.raises(RuntimeError, match="numeric.seq_h4"):
        _require(dead_field, signal_names)

    wrong_side_axis = copy.deepcopy(report)
    wrong_side_axis["structural"]["exit_side_axis"]["changed_rows"] = 0  # type: ignore[index]
    with pytest.raises(RuntimeError, match="structural.exit_side_axis"):
        _require(wrong_side_axis, signal_names)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("selected_online_model_state_sha256", SELECTED_ONLINE_MODEL_STATE_SHA256),
        ("val_data_sha256", VAL_DATA_SHA256),
        ("multi_tf_cache_identity_sha256", MULTI_TF_CACHE_IDENTITY_SHA256),
        (
            "unified_exit_lifecycle_root_manifest_sha256",
            UNIFIED_EXIT_LIFECYCLE_ROOT_MANIFEST_SHA256,
        ),
    ],
)
def test_unified_exit_influence_rejects_wrong_source_binding(
    field: str,
    expected: str,
) -> None:
    signal_names = _signal_names()
    report = passing_unified_exit_input_influence(signal_names)
    report[field] = ("0" if expected[0] != "0" else "1") * 64

    with pytest.raises(RuntimeError, match=field):
        _require(report, signal_names)
