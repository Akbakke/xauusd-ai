from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_cross_surface_overlap_v1 import (
    DECISION_ROUTES,
    POLICY_VERSION,
    SCHEMA_VERSION,
    classify_active_duplicate_pairs,
    declared_context_mtf_aliases,
    require_eight_family_coverage,
    validate_cross_surface_overlap_report,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.scripts.audit_entry_cross_surface_feature_overlap_v1 import (
    _TimestampedColumnHashes,
    _scan_decision_surface,
)


def _digest(byte: str) -> str:
    return byte * 64


def test_only_explicit_mtf_context_alias_is_permitted() -> None:
    aliases = sorted(declared_context_mtf_aliases(decision="entry"))
    local_hashes = {
        local: hashlib.sha256(f"{local}:{mtf}".encode("utf-8")).hexdigest()
        for local, mtf in aliases
    }
    mtf_hashes = {
        mtf: local_hashes[local]
        for local, mtf in aliases
    }
    result = classify_active_duplicate_pairs(
        decision="entry",
        local_field_hashes=local_hashes,
        active_mtf_field_hashes=mtf_hashes,
    )

    assert len(result["declared_context_mtf_alias_pairs"]) == len(aliases)
    assert result["missing_declared_context_mtf_alias_pairs"] == []
    assert result["unexpected_active_exact_duplicate_pairs"] == []


def test_unregistered_active_cross_surface_duplicate_fails_closed() -> None:
    result = classify_active_duplicate_pairs(
        decision="exit",
        local_field_hashes={"local.signal.real_feature": _digest("b")},
        active_mtf_field_hashes={"mtf.m5.other_real_feature": _digest("b")},
    )

    assert result["declared_context_mtf_alias_pairs"] == []
    assert result["missing_declared_context_mtf_alias_pairs"]
    assert result["unexpected_active_exact_duplicate_pairs"] == [
        {
            "local_field": "local.signal.real_feature",
            "mtf_field": "mtf.m5.other_real_feature",
            "values_sha256": _digest("b"),
        }
    ]


def test_entry_m5_cache_is_not_an_active_entry_route() -> None:
    assert DECISION_ROUTES["entry"]["local_timeframe"] == "M5"
    assert "M5" not in DECISION_ROUTES["entry"]["active_mtf_timeframes"]
    assert all("mtf.m5." not in pair[1] for pair in declared_context_mtf_aliases(decision="entry"))


def _passing_overlap_report() -> dict:
    report = {
        "schema_version": SCHEMA_VERSION,
        "entry_run_id": "UNIT_CROSS_SURFACE_20260820",
        "decision": "PASS",
        "failures": [],
        "policy": {
            "version": POLICY_VERSION,
            "decision_routes": {
                decision: {
                    "local_timeframe": route["local_timeframe"],
                    "active_mtf_timeframes": list(route["active_mtf_timeframes"]),
                }
                for decision, route in DECISION_ROUTES.items()
            },
        },
        "input_bindings": {"signal_manifest": {"path": "/fixture/signal", "sha256": "a" * 64}},
        "eight_family_coverage": {
            f"family_{index}": {"local_field_count": 1, "mtf_field_count": 1}
            for index in range(8)
        },
    }
    for decision, route in DECISION_ROUTES.items():
        local_hashes = {
            f"local.signal.fixture_{index}": hashlib.sha256(
                f"signal:{decision}:{index}".encode("utf-8")
            ).hexdigest()
            for index in range(MODEL_NATIVE_SIGNAL_DIM)
        }
        local_hashes.update(
            {
                f"local.ctx_cont.{field}": hashlib.sha256(
                    f"context:{decision}:{field}".encode("utf-8")
                ).hexdigest()
                for field in MODEL_NATIVE_CTX_CONT_FIELDS
            }
        )
        active_mtf_hashes = {
            f"mtf.{timeframe.lower()}.{field}": hashlib.sha256(
                f"mtf:{decision}:{timeframe}:{field}".encode("utf-8")
            ).hexdigest()
            for timeframe in route["active_mtf_timeframes"]
            for field in MULTI_TF_PER_BAR_FEATURES_V4
        }
        for local, mtf in declared_context_mtf_aliases(decision=decision):
            digest = hashlib.sha256(f"alias:{decision}:{local}:{mtf}".encode("utf-8")).hexdigest()
            local_hashes[local] = digest
            active_mtf_hashes[mtf] = digest
        classified = classify_active_duplicate_pairs(
            decision=decision,
            local_field_hashes=local_hashes,
            active_mtf_field_hashes=active_mtf_hashes,
        )
        report[decision] = {
            "local_timeframe": route["local_timeframe"],
            "active_mtf_timeframes": list(route["active_mtf_timeframes"]),
            "row_count": 4,
            "local_field_hashes": local_hashes,
            "active_mtf_field_hashes": active_mtf_hashes,
            **classified,
        }
    return report


def test_overlap_report_validation_rejects_a_missing_declared_alias(tmp_path: Path) -> None:
    report_path = tmp_path / "ENTRY_CROSS_SURFACE_INPUT_OVERLAP_20260820T120000000000Z.json"
    report = _passing_overlap_report()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validated = validate_cross_surface_overlap_report(
        report_path.resolve(),
        expected_entry_run_id="UNIT_CROSS_SURFACE_20260820",
        expected_input_bindings={"signal_manifest": {"path": "/fixture/signal", "sha256": "a" * 64}},
    )
    assert validated["decision"] == "PASS"

    report["entry"]["declared_context_mtf_alias_pairs"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="DECLARED_ALIAS"):
        validate_cross_surface_overlap_report(report_path.resolve())


def test_timestamped_hashes_require_exact_values_not_matching_summary_statistics() -> None:
    times = np.asarray([10, 20, 30], dtype=np.int64)
    first = _TimestampedColumnHashes(["one"])
    second = _TimestampedColumnHashes(["two"])
    changed = _TimestampedColumnHashes(["three"])
    first.update(
        timestamps_ns=times,
        values=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        names=["one"],
    )
    second.update(
        timestamps_ns=times,
        values=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        names=["two"],
    )
    changed.update(
        timestamps_ns=times,
        values=np.asarray([[1.0], [2.0], [3.25]], dtype=np.float32),
        names=["three"],
    )

    assert first.result()["one"] == second.result()["two"]
    assert first.result()["one"] != changed.result()["three"]


def test_full_batch_scanner_detects_an_unknown_active_duplicate(tmp_path: Path) -> None:
    signal_fields = list(
        MODEL_NATIVE_BASE_FIELDS
        + MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
        + MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
    )
    rows = 6
    cache_start = pd.Timestamp("2026-01-01T00:00:00Z")
    cache_times = np.asarray(
        (cache_start + pd.to_timedelta(np.arange(2500), unit="min")).asi8,
        dtype=np.int64,
    )
    cache: dict[str, pd.DataFrame] = {}
    for tf_index, timeframe in enumerate(("M5", "M15", "H1", "H4", "D1")):
        values = (
            np.arange(cache_times.size, dtype=np.float32)[:, None]
            + np.arange(len(MULTI_TF_PER_BAR_FEATURES_V4), dtype=np.float32)[None, :]
            * np.float32(0.01)
            + np.float32(tf_index * 1000)
        )
        frame = pd.DataFrame(values, columns=MULTI_TF_PER_BAR_FEATURES_V4)
        frame.attrs["ts_int64"] = cache_times
        frame.attrs["feats_np"] = values
        cache[timeframe] = frame

    times = cache_start + pd.to_timedelta(2000 + np.arange(rows) * 5, unit="min")
    times_ns = np.asarray(times.asi8, dtype=np.int64)
    signal = np.arange(rows * len(signal_fields), dtype=np.float32).reshape(rows, -1)
    # This is not a declared context alias.  It is a byte-identical raw signal
    # and active M15 feature, so the scanner must preserve enough evidence for
    # the policy to reject it.
    cutoff = times_ns + 5 * 60 * 1_000_000_000 - 15 * 60 * 1_000_000_000
    positions = np.searchsorted(cache_times, cutoff, side="right") - 1
    signal[:, 0] = cache["M15"].attrs["feats_np"][positions, 0]
    ctx_cont = np.arange(rows * len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.float32).reshape(rows, -1)
    ctx_cat = np.zeros((rows, MODEL_NATIVE_CTX_CAT_DIM), dtype=np.int64)
    surface = tmp_path / "m5_surface.parquet"
    pq.write_table(
        pa.table(
            {
                "time": pa.array(times),
                "signal": signal.tolist(),
                "ctx_cont": ctx_cont.tolist(),
                "ctx_cat": ctx_cat.tolist(),
            }
        ),
        surface,
    )

    scan = _scan_decision_surface(
        decision="entry",
        surface_path=surface,
        decision_seconds=300,
        signal_fields=signal_fields,
        cache=cache,
        batch_size=2,
    )
    classified = classify_active_duplicate_pairs(
        decision="entry",
        local_field_hashes=scan["local_field_hashes"],
        active_mtf_field_hashes=scan["active_mtf_field_hashes"],
    )

    assert scan["row_count"] == rows
    assert any(
        row["local_field"] == f"local.signal.{signal_fields[0]}"
        and row["mtf_field"] == f"mtf.m15.{MULTI_TF_PER_BAR_FEATURES_V4[0]}"
        for row in classified["unexpected_active_exact_duplicate_pairs"]
    )


def test_all_eight_families_are_present_on_local_and_mtf_planes() -> None:
    local = [
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
        *(f"ctx_cont.{name}" for name in MODEL_NATIVE_CTX_CONT_FIELDS),
    ]
    coverage = require_eight_family_coverage(
        local_fields=local,
        mtf_feature_names=MULTI_TF_PER_BAR_FEATURES_V4,
    )

    assert len(coverage) == 8
    assert all(row["local_field_count"] > 0 for row in coverage.values())
    assert all(row["mtf_field_count"] > 0 for row in coverage.values())


def test_rebuild_runs_cross_surface_audit_before_dataset_builder() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/rebuild_entry_model_native_seq513_dataset.sh"
    ).read_text(encoding="utf-8")

    audit = source.index("audit_entry_cross_surface_feature_overlap_v1")
    builder = source.index("-m gx1.scripts.build_entry_v10_ctx_training_dataset_v3")
    assert audit < builder
    assert "ENTRY_CROSS_SURFACE_INPUT_OVERLAP_" in source
    assert "full cross-surface active-input duplicate audit PASS" in source
