from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    MODEL_NATIVE_AUXILIARY_PREDICTION_VECTOR_WIDTHS,
    PREDICTION_EVIDENCE_STAGE_SPLITS,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_DIP_OUTPUT_DIM,
    MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
    MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS,
    MODEL_NATIVE_TIMING_OUTPUT_DIM,
    MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS,
)
import torch

from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    EVALUATION_COVERAGES,
    _EXTRA_VECTOR_HEADS,
    _append_extra_vector_head_evidence,
    _canonical_live_decision_evidence,
    _concatenate_evidence_chunks,
    _preregistered_hypothesis,
    _research_policy_pnl,
    _require_entry_q_ssot,
    _require_evaluation_mtf_source_provenance,
    _load_val_reference,
    _require_post_prediction_input_stability,
    _require_requested_test_bindings_match_seal,
    _require_selective_edge_stage_split,
    _selection_sort_column,
    _selective_edge_device_arg,
    build_summary,
    run,
    build_metric_rows,
)
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mtf_provenance_binds_distinct_model_and_cache_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The model frame and the MTF cache source are distinct bound objects."""

    model_source = tmp_path / "FULL_PLUS_CTX_v3src.parquet"
    model_source.write_bytes(b"model source")
    cache_source = tmp_path / "m5_enriched.parquet"
    cache_source.write_bytes(b"cache source")
    split_manifest = tmp_path / "val.manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "MULTI_TF_V4_CACHE"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")

    cache_binding = {
        "cache_identity_sha256": "a" * 64,
        "manifest_sha256": _sha256(cache_dir / "manifest.json"),
        "m5_prebuilt_source": str(cache_source),
        "m5_prebuilt_source_sha256": _sha256(cache_source),
    }
    monkeypatch.setattr(
        "gx1.scripts.evaluate_entry_candidate_selective_edge_v1."
        "require_dataset_manifest_multi_tf_cache_binding",
        lambda *_args, **_kwargs: cache_binding,
    )
    monkeypatch.setattr(
        "gx1.scripts.evaluate_entry_candidate_selective_edge_v1."
        "require_multi_tf_v4_cache_binding_files",
        lambda *_args, **_kwargs: cache_binding,
    )
    dataset_contract = {
        "splits": {
            "val": {
                "manifest_path": str(split_manifest),
                "source_frame": {
                    "parquet_path": str(model_source),
                    "parquet_sha256": _sha256(model_source),
                },
            }
        }
    }
    bundle_metadata = {
        "run_lineage": {"dataset_run_id": "pytest-run"},
        "multi_tf": {
            "shared_cache_identity_sha256": cache_binding[
                "cache_identity_sha256"
            ],
            "shared_cache_manifest_sha256": cache_binding["manifest_sha256"],
            "shared_cache_dir": str(cache_dir),
            "shared_cache_manifest_path": str(cache_dir / "manifest.json"),
            "shared_cache_m5_source": str(cache_source),
            "shared_cache_m5_source_sha256": cache_binding[
                "m5_prebuilt_source_sha256"
            ],
        },
    }

    observed = _require_evaluation_mtf_source_provenance(
        dataset_contract=dataset_contract,
        bundle_metadata=bundle_metadata,
        m5_prebuilt=model_source,
        mtf_cache_dir=cache_dir,
    )
    assert observed["source_frame"]["parquet_path"] == str(model_source)
    assert observed["bundle_mtf"]["shared_cache_m5_source"] == str(cache_source)

    with pytest.raises(
        RuntimeError,
        match="SELECTIVE_EDGE_EVALUATION_M5_PROVENANCE_INVALID",
    ):
        _require_evaluation_mtf_source_provenance(
            dataset_contract=dataset_contract,
            bundle_metadata=bundle_metadata,
            m5_prebuilt=cache_source,
            mtf_cache_dir=cache_dir,
        )


def test_post_prediction_integrity_rechecks_every_scored_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No selective-edge report may be written against moving input bytes."""

    initial_bundle = {"model_state_dict_sha256": "a" * 64}
    initial_dataset = {"contract": {"ordered_fields_sha256": "b" * 64}}
    initial_mtf = {"cache_binding": {"cache_identity_sha256": "c" * 64}}
    monkeypatch.setattr(
        "gx1.scripts.evaluate_entry_candidate_selective_edge_v1._bundle_core_integrity_snapshot",
        lambda **_kwargs: initial_bundle,
    )
    monkeypatch.setattr(
        "gx1.scripts.evaluate_entry_candidate_selective_edge_v1._dataset_model_native_contract",
        lambda *_args, **_kwargs: initial_dataset,
    )
    monkeypatch.setattr(
        "gx1.scripts.evaluate_entry_candidate_selective_edge_v1._require_evaluation_mtf_source_provenance",
        lambda **_kwargs: initial_mtf,
    )
    kwargs = {
        "initial_bundle_core": initial_bundle,
        "initial_dataset_contract": initial_dataset,
        "initial_mtf_source_provenance": initial_mtf,
        "bundle_dir": tmp_path,
        "bundle_metadata": {},
        "dataset_dir": tmp_path,
        "splits": ["val"],
        "split_bindings": {"val": {}},
        "m5_prebuilt": tmp_path / "m5.parquet",
        "mtf_cache_dir": tmp_path / "cache",
    }
    _require_post_prediction_input_stability(**kwargs)

    monkeypatch.setattr(
        "gx1.scripts.evaluate_entry_candidate_selective_edge_v1._dataset_model_native_contract",
        lambda *_args, **_kwargs: {"contract": {"ordered_fields_sha256": "d" * 64}},
    )
    with pytest.raises(
        RuntimeError,
        match="SELECTIVE_EDGE_DATASET_CHANGED_DURING_PREDICTION",
    ):
        _require_post_prediction_input_stability(**kwargs)


def test_vector_evidence_widths_match_model_output_owners() -> None:
    """A producer must not substitute physical target count for head width."""

    assert _EXTRA_VECTOR_HEADS == {
        "dip_pred": MODEL_NATIVE_DIP_OUTPUT_DIM,
        "forecast_pred": len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS),
        "timing_pred": MODEL_NATIVE_TIMING_OUTPUT_DIM,
        "tail_risk_pred": len(MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS),
        "vol_forecast_pred": len(MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS),
    }
    assert MODEL_NATIVE_AUXILIARY_PREDICTION_VECTOR_WIDTHS == _EXTRA_VECTOR_HEADS


def test_vector_head_evidence_is_persisted_as_exact_dense_vectors() -> None:
    """Prediction evidence must retain head names, not split them into scalars."""

    outputs = {
        name: torch.full((2, width), float(index + 1), dtype=torch.float32)
        for index, (name, width) in enumerate(_EXTRA_VECTOR_HEADS.items())
    }
    chunks: dict[str, list[np.ndarray]] = {}
    _append_extra_vector_head_evidence(chunks, outputs)
    combined = _concatenate_evidence_chunks(chunks, expected_rows=2)

    assert set(combined) == set(_EXTRA_VECTOR_HEADS)
    for name, width in _EXTRA_VECTOR_HEADS.items():
        assert combined[name].shape == (2, width)
    assert not any(name.rsplit("_", 1)[0] in _EXTRA_VECTOR_HEADS for name in combined)


def test_entry_q_is_the_only_decision_surface_and_ties_fail_closed() -> None:
    q = torch.tensor(
        [[3.0, -1.0, 0.0], [-2.0, 4.0, 0.0], [-1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.equal(_require_entry_q_ssot({"entry_action_q_bps": q}), q)
    with pytest.raises(RuntimeError, match="forbidden legacy"):
        _require_entry_q_ssot(
            {"entry_action_q_bps": q, "anchor_logits": torch.zeros_like(q)}
        )
    with pytest.raises(RuntimeError, match="no unique top action"):
        _require_entry_q_ssot(
            {"entry_action_q_bps": torch.tensor([[1.0, 1.0, 0.0]])}
        )


def test_runtime_test_bindings_must_equal_the_prefreeze_seal(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    seal = {
        "dataset_dir": str(dataset.resolve()),
        "test_manifest": {
            "path": str((dataset / "xau_test.manifest.json").resolve()),
            "sha256": "a" * 64,
        },
        "test_parquet": {
            "path": str((dataset / "xau_test.parquet").resolve()),
            "sha256": "b" * 64,
        },
    }
    bindings = {
        "test": {
            "manifest_path": seal["test_manifest"]["path"],
            "manifest_sha256": seal["test_manifest"]["sha256"],
            "parquet_path": seal["test_parquet"]["path"],
            "parquet_sha256": seal["test_parquet"]["sha256"],
        }
    }
    _require_requested_test_bindings_match_seal(
        bindings,
        dataset_dir=dataset.resolve(),
        seal=seal,
    )
    bindings["test"]["parquet_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="SELECTIVE_EDGE_TEST_SEAL_BINDING_INVALID"):
        _require_requested_test_bindings_match_seal(
            bindings,
            dataset_dir=dataset.resolve(),
            seal=seal,
        )


def test_val_stage_forbids_any_test_val_reference_binding(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="SELECTIVE_EDGE_VAL_STAGE_FORBIDS_VAL_REFERENCE"):
        _load_val_reference(
            str(tmp_path / "forbidden.json"),
            expected_sha256="a" * 64,
            evidence_stage="pre_calibration",
            bundle_dir=tmp_path,
            dataset_dir=tmp_path,
            dataset_contract={},
        )


def test_runtime_authoritative_route_is_locked_without_single_use_release(
    tmp_path: Path,
) -> None:
    class Args:
        evidence_stage = "runtime_authoritative"
        splits = "test"

    with pytest.raises(
        RuntimeError,
        match="SELECTIVE_EDGE_TEST_RELEASE_AUTHORITY_REQUIRED",
    ):
        run(Args())


def test_live_decision_evidence_contains_raw_q_argmax_and_no_probabilities() -> None:
    q = torch.tensor(
        [[3.0, -1.0, 0.0], [-2.0, 4.0, 0.0], [-1.0, -2.0, 0.0]],
        dtype=torch.float32,
    )
    evidence = _canonical_live_decision_evidence(
        {"entry_action_q_bps": q}
    )
    assert set(evidence) == {
        "entry_action_q_bps",
        "entry_action_q_margin_bps",
        "model_direction_index",
        "edge_score",
        "selection_score",
    }
    assert evidence["model_direction_index"].tolist() == [0, 1, 2]
    assert np.array_equal(evidence["entry_action_q_bps"], q.numpy())
    assert np.all(evidence["entry_action_q_margin_bps"] > 0.0)


def test_research_policy_pnl_uses_action_side_and_flat_zero_without_claiming_net() -> None:
    frame = pd.DataFrame(
        {
            "pred_direction": [0, 1, 2],
            "y_long_final_pnl_at_direction_horizon_bps": [5.0, 99.0, 99.0],
            "y_short_final_pnl_at_direction_horizon_bps": [99.0, -3.0, 99.0],
        }
    )
    assert _research_policy_pnl(frame).tolist() == [5.0, -3.0, 0.0]
    with pytest.raises(RuntimeError, match="gross spread-inclusive research outcomes"):
        _research_policy_pnl(
            frame.drop(columns=["y_long_final_pnl_at_direction_horizon_bps"])
        )


def test_selection_sort_is_raw_q_action_value_and_mode_bound() -> None:
    frame = pd.DataFrame(
        {
            "selection_score": [3.0, 2.0],
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 2,
        }
    )
    assert _selection_sort_column(frame) == "selection_score"
    frame.loc[0, "selection_score_mode"] = "probability"
    with pytest.raises(RuntimeError, match="direction mode mismatch"):
        _selection_sort_column(frame)


def test_evidence_chunks_require_exact_row_count_and_shape() -> None:
    combined = _concatenate_evidence_chunks(
        {
            "entry_action_q_bps": [
                np.ones((2, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
            ]
        },
        expected_rows=3,
    )
    assert combined["entry_action_q_bps"].shape == (3, 3)
    with pytest.raises(RuntimeError, match="row mismatch"):
        _concatenate_evidence_chunks(
            {"entry_action_q_bps": [np.ones((2, 3), dtype=np.float32)]},
            expected_rows=3,
        )


def test_preregistered_metrics_use_fixed_grid_and_autocorrelation_null() -> None:
    """A synthetic time-linked signal clears both fixed primary nulls.

    Direction is deliberately attached to the better side on each row; a
    non-zero circular label shift breaks that time alignment, while an iid
    coin-flip has expectation zero.  This exercises the actual two-part gate,
    not only a convenience mean-PnL calculation.
    """

    rng = np.random.default_rng(71)
    rows = 2_048
    state = rng.choice(np.array([-1.0, 1.0]), size=rows)
    long_outcome = state * 5.0 + rng.normal(0.0, 0.1, size=rows)
    short_outcome = -state * 5.0 + rng.normal(0.0, 0.1, size=rows)
    direction = np.where(state > 0.0, 0, 1)
    frame = pd.DataFrame(
        {
            "split": "val",
            "model": "candidate",
            "time": pd.date_range("2025-06-01", periods=rows, freq="5min", tz="UTC"),
            "pred_direction": direction,
            "selection_score": np.linspace(float(rows), 1.0, rows),
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * rows,
            "edge_score": np.ones(rows),
            "y_long_final_pnl_at_direction_horizon_bps": long_outcome,
            "y_short_final_pnl_at_direction_horizon_bps": short_outcome,
        }
    )
    metrics = pd.DataFrame(
        build_metric_rows(frame, top_fracs=list(EVALUATION_COVERAGES))
    )
    assert metrics["coverage_fraction"].tolist() == list(EVALUATION_COVERAGES)
    assert metrics.loc[metrics["coverage_fraction"] == 0.25, "primary_pass"].item()
    assert set(metrics["coin_flip_null_method"]) == {
        "exact_uniform_long_short_expectation"
    }
    hypothesis = _preregistered_hypothesis(
        metrics,
        evidence_stage="pre_calibration",
        val_reference=None,
    )
    assert hypothesis["decision"] == "PASS"
    assert 0.25 in hypothesis["qualifying_coverages"]


def test_summary_uses_emitted_preregistered_scope_and_preserves_booleans() -> None:
    predictions = pd.DataFrame({"split": ["val"], "model": ["candidate"]})
    metrics = pd.DataFrame(
        {
            "split": ["val", "val"],
            "model": ["candidate", "candidate"],
            "scope": [
                "preregistered_raw_q_coverage",
                "preregistered_raw_q_coverage",
            ],
            "group": ["ALL", "ALL"],
            "top_frac": [0.05, 0.10],
            "mean_pnl_bps": [1.25, 2.5],
            "primary_pass": [True, False],
        }
    )

    summary = build_summary(predictions, metrics)["summaries"]

    assert summary == [
        {
            "split": "val",
            "model": "candidate",
            "rows": 1,
            "top5_all_mean_pnl_bps": 1.25,
            "top10_all_mean_pnl_bps": 2.5,
            "top5_primary_pass": True,
            "top10_primary_pass": False,
        }
    ]


def test_selective_edge_auto_is_cpu_and_direct_cuda_fails_closed() -> None:
    assert _selective_edge_device_arg("auto") == "cpu"
    with pytest.raises(RuntimeError, match="GX1_CUDA_PRODUCER_GX1_CAPPED_CLASS_INVALID"):
        _selective_edge_device_arg("cuda")


@pytest.mark.parametrize(
    ("stage", "split"),
    tuple(
        (stage, splits[0])
        for stage, splits in PREDICTION_EVIDENCE_STAGE_SPLITS.items()
    ),
)
def test_stage_split_contract_is_exact(stage: str, split: str) -> None:
    _require_selective_edge_stage_split(
        evidence_stage=stage, split_spec=split
    )
    wrong = "test" if split == "val" else "val"
    with pytest.raises(RuntimeError, match="SELECTIVE_EDGE_STAGE_SPLIT_INVALID"):
        _require_selective_edge_stage_split(
            evidence_stage=stage, split_spec=wrong
        )
