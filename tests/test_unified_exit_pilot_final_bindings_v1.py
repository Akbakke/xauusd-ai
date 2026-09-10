from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    build_composite_normalization_binding,
    build_split_sequence_binding,
    require_composite_normalization_binding,
    require_split_sequence_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_physical_summary_sample_authority,
    fit_lifetime_summary_normalization,
)
from tests.model_native_input_normalization_support import (
    input_normalization_fixture,
)


def _summary() -> dict:
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[8, 9],
        source_lineage_sha256="1" * 64,
    )
    rows = authority["fit_row_count"]
    base = np.arange(rows, dtype=np.float64)[:, None]
    values = np.concatenate(
        [base * (index + 1.0) + index for index in range(7)], axis=1
    )
    return fit_lifetime_summary_normalization(
        values=values,
        sample_authority=authority,
    )


def _base() -> dict:
    contract = input_normalization_fixture(
        signal_names=["signal_a", "signal_b"],
        mtf_names=["mtf_a", "mtf_b"],
    )
    return {
        "schema_version": "gx1_unified_exit_pilot_base_normalization_v1",
        "decision": "PASS",
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "population_witness_sha256": "2" * 64,
        "val_fit_rows": 0,
        "test_fit_rows": 0,
        "test_accessed": False,
    }


def test_composite_normalization_binds_and_revalidates_both_surfaces() -> None:
    value = build_composite_normalization_binding(
        base_artifact=_base(),
        base_path="/immutable/base.json",
        base_file_sha256="3" * 64,
        summary_normalization=_summary(),
        summary_manifest_path="/immutable/summary.json",
        summary_manifest_file_sha256="4" * 64,
        summary_manifest_sha256="5" * 64,
    )
    assert require_composite_normalization_binding(
        value,
        expected_base_contract_sha256=value["base_feature_normalization"][
            "contract_sha256"
        ],
        expected_summary_normalization_sha256=value[
            "lifetime_summary_normalization"
        ]["normalization_sha256"],
    ) == value
    assert value["val_fit_rows"] == 0
    assert value["test_fit_rows"] == 0
    bad = copy.deepcopy(value)
    bad["lifetime_summary_normalization"]["surface"]["scale"][0] *= 2
    with pytest.raises(RuntimeError):
        require_composite_normalization_binding(bad)


def test_split_sequence_binding_proves_first_state_and_split_censor() -> None:
    entry = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="5min")
    m1 = pd.date_range("2025-12-31T23:52:00Z", periods=40, freq="1min")
    value = build_split_sequence_binding(
        split="train",
        entry_times=entry,
        m1_times=m1,
        successor_transition_counts=[4, 5, 6],
        child_admission_file_sha256="1" * 64,
        child_admission_witness_sha256="2" * 64,
        child_parquet_sha256="3" * 64,
        child_manifest_file_sha256="4" * 64,
        child_manifest_contract_sha256="5" * 64,
        m1_source_sha256="6" * 64,
        m1_manifest_file_sha256="7" * 64,
        closure_authority_file_sha256="8" * 64,
        closure_authority_sha256="9" * 64,
    )
    assert require_split_sequence_binding(
        value, expected_split="train", expected_entry_rows=3
    ) == value
    assert value["successor_transition_total"] == 15
    bad = copy.deepcopy(value)
    bad["bindings"]["m1_source"] = "a" * 64
    with pytest.raises(RuntimeError, match="SEQUENCE_BINDING_INVALID"):
        require_split_sequence_binding(
            bad, expected_split="train", expected_entry_rows=3
        )


def test_split_sequence_binding_rejects_successor_past_child_end() -> None:
    entry = pd.DatetimeIndex(["2026-01-01T00:00:00Z"])
    m1 = pd.date_range("2025-12-31T23:59:00Z", periods=8, freq="1min")
    with pytest.raises(RuntimeError, match="SEQUENCE_RANGE_INVALID"):
        build_split_sequence_binding(
            split="val",
            entry_times=entry,
            m1_times=m1,
            successor_transition_counts=[3],
            child_admission_file_sha256="1" * 64,
            child_admission_witness_sha256="2" * 64,
            child_parquet_sha256="3" * 64,
            child_manifest_file_sha256="4" * 64,
            child_manifest_contract_sha256="5" * 64,
            m1_source_sha256="6" * 64,
            m1_manifest_file_sha256="7" * 64,
            closure_authority_file_sha256="8" * 64,
            closure_authority_sha256="9" * 64,
        )
