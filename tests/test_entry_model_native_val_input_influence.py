from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM
from gx1.contracts.entry_model_native_val_input_influence_v1 import (
    COMPARISON_SURFACE,
    COUNTERFACTUAL_DELTA_EPSILON,
    FAMILY_ABLATION_EPSILON,
    NUMERIC_GRADIENT_EPSILON,
    SAMPLE_COUNT,
    SAMPLING_CONTRACT,
    SCHEMA_VERSION,
    SPLIT,
    canonical_json_sha256,
    require_entry_val_input_influence,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    individual_input_influence_layout,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _entry_val_influence_sample,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def _passing_report() -> tuple[dict[str, object], list[str]]:
    names = [f"model_native_signal_{index:03d}" for index in range(MODEL_NATIVE_SIGNAL_DIM)]
    ownership = individual_input_influence_layout(
        names,
        mtf_timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
    )
    numeric = {
        surface: {
            "tokens": row["tokens"],
            "source_indices": row["source_indices"],
            "metrics": {
                token: {
                    "decision": "PASS",
                    "failures": [],
                    "max_abs_entry_action_q_class_margin_gradient": 0.01,
                }
                for token in row["tokens"]
            },
        }
        for surface, row in ownership["numeric"].items()
    }
    def _counterfactual() -> dict[str, object]:
        return {
            "decision": "PASS",
            "failures": [],
            "counterfactual": "valid_owner_manifold_counterfactual",
            "max_abs_entry_action_q_delta_bps": 0.01,
            "changed_rows": SAMPLE_COUNT,
            "total_rows": SAMPLE_COUNT,
        }
    local_context = {
        specialist: {
            "decision": "PASS",
            "failures": [],
            "source_binding_sha256": _SHA_A,
            "max_abs_entry_action_q_delta_bps": 0.01,
            "changed_rows": SAMPLE_COUNT,
            "total_rows": SAMPLE_COUNT,
        }
        for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS
    }
    multi_tf = {
        f"{timeframe.lower()}:{specialist}": {
            "decision": "PASS",
            "failures": [],
            "source_binding_sha256": _SHA_A,
            "max_abs_entry_action_q_delta_bps": 0.01,
            "changed_rows": SAMPLE_COUNT,
            "total_rows": SAMPLE_COUNT,
        }
        for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
        for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "failures": [],
        "required_for_candidate": True,
        "split": SPLIT,
        "sample_count": SAMPLE_COUNT,
        "sampling_contract": SAMPLING_CONTRACT,
        "comparison_surface": COMPARISON_SURFACE,
        "numeric_gradient_epsilon": NUMERIC_GRADIENT_EPSILON,
        "counterfactual_delta_epsilon": COUNTERFACTUAL_DELTA_EPSILON,
        "family_ablation_epsilon": FAMILY_ABLATION_EPSILON,
        "sample_entry_row_indices": list(range(SAMPLE_COUNT)),
        "sample_decision_times_ns": list(range(SAMPLE_COUNT)),
        "val_data_sha256": _SHA_A,
        "multi_tf_cache_identity_sha256": _SHA_B,
        "selected_model_state_dict_sha256": _SHA_C,
        "ordered_signal_names": names,
        "signal_names_sha256": canonical_json_sha256(names),
        "input_ownership": ownership,
        "input_ownership_sha256": canonical_json_sha256(ownership),
        "numeric_input_count": sum(len(row["tokens"]) for row in ownership["numeric"].values()),
        "continuous_manifold_input_count": len(ownership["continuous_manifold"]),
        "categorical_input_count": len(ownership["categorical"]),
        "individual": {
            "numeric": numeric,
            "continuous_manifold": {row["token"]: _counterfactual() for row in ownership["continuous_manifold"]},
            "categorical": {row["token"]: _counterfactual() for row in ownership["categorical"]},
        },
        "family_ablation": {
            "epsilon": FAMILY_ABLATION_EPSILON,
            "sample_count": SAMPLE_COUNT,
            "local_context_routing_sha256": _SHA_D,
            "multi_tf_routing_sha256": _SHA_E,
            "local_context": local_context,
            "multi_tf": multi_tf,
        },
    }, names


def _require(report: dict[str, object], names: list[str]) -> None:
    require_entry_val_input_influence(
        report,
        ordered_signal_names=names,
        val_data_sha256=_SHA_A,
        multi_tf_cache_identity_sha256=_SHA_B,
        selected_model_state_dict_sha256=_SHA_C,
        local_context_routing_sha256=_SHA_D,
        multi_tf_routing_sha256=_SHA_E,
        context="TEST",
    )


def test_entry_val_input_influence_requires_all_physical_and_family_routes() -> None:
    report, names = _passing_report()
    _require(report, names)

    missing_family = copy.deepcopy(report)
    missing_family["family_ablation"]["multi_tf"].pop(next(iter(missing_family["family_ablation"]["multi_tf"])))
    with pytest.raises(RuntimeError, match="MULTI_TF_SET_INVALID"):
        _require(missing_family, names)

    dead_numeric = copy.deepcopy(report)
    surface = next(iter(dead_numeric["individual"]["numeric"]))
    token = next(iter(dead_numeric["individual"]["numeric"][surface]["metrics"]))
    dead_numeric["individual"]["numeric"][surface]["metrics"][token]["max_abs_entry_action_q_class_margin_gradient"] = 0.0
    with pytest.raises(RuntimeError, match="NUMERIC_INVALID"):
        _require(dead_numeric, names)

    wrong_state = copy.deepcopy(report)
    wrong_state["selected_model_state_dict_sha256"] = _SHA_A
    with pytest.raises(RuntimeError, match="BINDING_INVALID"):
        _require(wrong_state, names)


def test_entry_val_probe_reads_eight_direct_evenly_spaced_dataset_positions() -> None:
    class DatasetProbe:
        indices = np.arange(16, dtype=np.int64)
        df = pd.DataFrame(
            {"time": pd.date_range("2025-01-01", periods=16, freq="5min", tz="UTC")}
        )

        def __init__(self) -> None:
            self.calls: list[int] = []

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
            self.calls.append(position)
            return {
                "entry_row_index": torch.tensor(self.indices[position]),
                "seq_x": torch.zeros((2, 1), dtype=torch.float32),
            }

    dataset = DatasetProbe()
    _batch, rows, times = _entry_val_influence_sample(dataset)  # type: ignore[arg-type]
    assert dataset.calls == [0, 2, 4, 6, 8, 10, 12, 15]
    assert rows == dataset.calls
    assert len(times) == SAMPLE_COUNT
