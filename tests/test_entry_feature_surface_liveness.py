from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.scripts.audit_entry_feature_surface_liveness_v1 import (
    _StreamingFieldStats,
    _cross_clock_harmony,
    _family_liveness,
    _field_liveness,
)


REPO = Path(__file__).resolve().parents[1]
CONTROL = REPO / "scripts/entry_next_edge_control.sh"


def _live_numeric_row() -> dict[str, object]:
    return {
        "row_count": 4,
        "finite_count": 4,
        "nonfinite_count": 0,
        "std": 0.5,
        "value_range": 1.0,
        "active_count": 3,
    }


def _live_categorical_row() -> dict[str, object]:
    return {
        "row_count": 4,
        "finite_count": 4,
        "nonfinite_count": 0,
        "integer_like_count": 4,
        "unique_count": 2,
    }


def test_streaming_stats_is_full_population_and_rejects_constant_fields() -> None:
    stats = _StreamingFieldStats(2)
    stats.update(np.asarray([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32))
    stats.update(np.asarray([[0.0, 3.0], [0.0, 4.0]], dtype=np.float32))
    rows = stats.finalize(["constant", "live"])

    constant_live, constant_reasons = _field_liveness(
        stats=rows["constant"], categorical=False
    )
    live, live_reasons = _field_liveness(stats=rows["live"], categorical=False)

    assert rows["live"]["row_count"] == 4
    assert constant_live is False
    assert "near_constant_std" in constant_reasons
    assert live is True
    assert live_reasons == []


def test_all_eight_actual_owner_families_require_every_field_to_be_live() -> None:
    signal_fields = [
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    ]
    field_stats: dict[str, dict[str, object]] = {
        **{
            f"local.signal.{field}": {
                **_live_numeric_row(),
                "live": True,
            }
            for field in signal_fields
        },
        **{
            f"local.ctx_cont.{field}": {
                **_live_numeric_row(),
                "live": True,
            }
            for field in MODEL_NATIVE_CTX_CONT_FIELDS
        },
        **{
            f"local.ctx_cat.{field}": {
                **_live_categorical_row(),
                "live": True,
            }
            for field in MODEL_NATIVE_CTX_CAT_FIELDS
        },
    }
    report, issues = _family_liveness(
        field_stats=field_stats, signal_fields=signal_fields
    )

    assert issues == []
    assert tuple(report) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert all(row["field_count"] > 0 for row in report.values())
    assert all(row["all_fields_live"] is True for row in report.values())

    first_family = MODEL_NATIVE_TRAINING_SPECIALISTS[0]
    # Select an actual field owned by the first family instead of letting a
    # test fixture invent an unmapped name.
    for candidate in field_stats:
        changed = dict(field_stats)
        changed[candidate] = {**changed[candidate], "live": False}
        changed_report, changed_issues = _family_liveness(
            field_stats=changed, signal_fields=signal_fields
        )
        if changed_report[first_family]["all_fields_live"] is False:
            assert changed_issues == []
            assert candidate in changed_report[first_family]["dead_fields"]
            break
    else:  # pragma: no cover - the exact owner contract always has all eight
        raise AssertionError("first owner family had no actual fields")


def test_m1_m5_harmony_requires_identical_field_and_family_order() -> None:
    signal_fields = [
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    ]
    field_stats: dict[str, dict[str, object]] = {
        **{
            f"local.signal.{field}": {**_live_numeric_row(), "live": True}
            for field in signal_fields
        },
        **{
            f"local.ctx_cont.{field}": {**_live_numeric_row(), "live": True}
            for field in MODEL_NATIVE_CTX_CONT_FIELDS
        },
        **{
            f"local.ctx_cat.{field}": {**_live_categorical_row(), "live": True}
            for field in MODEL_NATIVE_CTX_CAT_FIELDS
        },
    }
    entry_families, entry_issues = _family_liveness(
        field_stats=field_stats, signal_fields=signal_fields
    )
    exit_families, exit_issues = _family_liveness(
        field_stats=field_stats, signal_fields=signal_fields
    )
    assert entry_issues == exit_issues == []

    passed = _cross_clock_harmony(
        entry={"field_stats": field_stats},
        exit_={"field_stats": field_stats},
        entry_families=entry_families,
        exit_families=exit_families,
    )
    assert passed["decision"] == "PASS"
    assert tuple(passed["families"]) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert all(row["entry_exit_equal"] is True for row in passed["families"].values())

    reordered = dict(reversed(list(field_stats.items())))
    failed = _cross_clock_harmony(
        entry={"field_stats": field_stats},
        exit_={"field_stats": reordered},
        entry_families=entry_families,
        exit_families=exit_families,
    )
    assert failed["decision"] == "FAIL"
    assert "qualified_field_order_mismatch" in failed["failures"]


def test_control_route_requires_the_immutable_cross_surface_proof() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "model-native-feature-surface-liveness"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "requires exactly one explicit --run-id" in result.stderr
