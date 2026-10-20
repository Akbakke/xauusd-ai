"""Small synthetic proofs for TRAIN-owned dynamic M5 loss units.

No model, physical dataset or production checkpoint is needed by these tests.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gx1.contracts import entry_dynamic_m5_loss_scale_v1 as scales
from gx1.contracts import entry_observed_market_v1 as observed
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256

TERMINAL, MFE, MAE, CLOSED_MFE, CLOSED_MAE, FORECAST = scales.SURFACE_NAMES


def _bind(path: Path, value: bytes = b"synthetic lineage only") -> dict[str, str]:
    path.write_bytes(value)
    return {"path": str(path), "sha256": hashlib.sha256(value).hexdigest()}


def _economics() -> dict:
    value = {
        "schema_version": observed.SCHEMA_VERSION + "_economics",
        "cost_authority_sha256": "a" * 64,
        "coverage_start_utc": "2025-01-01T00:00:00Z",
        "coverage_end_utc_exclusive": "2025-02-01T00:00:00Z",
        "round_trip_execution_cost_bps": 4.0,
        "annual_financing_cost_bps": [532.0, 0.0],
        "seconds_per_year": observed.SECONDS_PER_YEAR,
        "historical_cost_truth_qualified": False,
        "capital_hurdle_included": False,
        "risk_utility_included": False,
    }
    # Economics owner hashes compact JSON without the campaign newline.
    return {**value, "economics_sha256": observed._sha(value)}


def _synthetic_loss_scale_lineage(tmp_path):
    quote = _bind(tmp_path / "quote-binding.json")
    index = _bind(tmp_path / "index-binding.json")
    manifest = {
        "extra": {"dynamic_m5_supervision": {
            "source_quote_binding": quote, "index_binding": index,
        }}
    }
    return {
        "fit_partition": "train",
        "train_rows": 3,
        "train_control_cutoff": "2025-02-01T00:00:00Z",
        "train_physical_source": {
            "parquet": _bind(tmp_path / "train.parquet"),
            "manifest": _bind(tmp_path / "train.manifest.json", json.dumps(manifest).encode()),
            "physical_rows": 3,
        },
        "population_witness": _bind(tmp_path / "population.json"),
        "eligible_parent_rows": _bind(tmp_path / "parents.npy"),
        "target_contract": observed.entry_observed_dynamic_m5_path_contract(),
        "economics": _economics(),
        "source_quote_binding": quote,
        "index_binding": index,
    }


@pytest.fixture
def lineage(tmp_path):
    return _synthetic_loss_scale_lineage(tmp_path)


def _synthetic_loaded_loss_scales(tmp_path):
    """Real strict owner output for caller-unit fixtures, never physical admission."""
    tmp_path = tmp_path / "synthetic-loss-unit-owner"
    tmp_path.mkdir(parents=True, exist_ok=True)
    lineage = _synthetic_loss_scale_lineage(tmp_path)
    artifact = _fit(lineage)
    binding = _bind(tmp_path / "loss-scales.json", json.dumps(artifact, allow_nan=False).encode())
    return scales.load_dynamic_m5_loss_scales(binding, expected_lineage=lineage)


def _surfaces(rows=3):
    """Every fit coordinate has real nonzero support; FLAT remains analytic."""
    result = {}
    values = np.array([4097.25, 8193.5, 16385.75], dtype=np.float32)[:rows]
    for name, shape in scales.dynamic_m5_loss_scale_contract()["surface_shapes"].items():
        target = np.broadcast_to(
            values.reshape((rows,) + (1,) * len(shape)), (rows, *shape)
        ).copy()
        if name in (TERMINAL, MFE, CLOSED_MFE):
            target[..., 1] *= -1
        valid = np.ones_like(target, dtype=np.bool_)
        if name == TERMINAL:
            target = np.concatenate((target, np.zeros_like(target[..., :1])), axis=-1)
            valid = np.concatenate((valid, np.ones_like(valid[..., :1])), axis=-1)
        result[name] = (target, valid)
    return result


def _fit(lineage, surfaces=None, chunks=(1, 2)):
    surfaces = _surfaces() if surfaces is None else surfaces
    batches, start = [], 0
    for length in chunks:
        stop = start + length
        batches.append((
            np.arange(start, stop, dtype=np.int64),
            {name: (target[start:stop], valid[start:stop])
             for name, (target, valid) in surfaces.items()},
        ))
        start = stop
    return scales.fit_dynamic_m5_loss_scales(
        batches=batches, lineage=lineage,
        expected_parent_rows=np.arange(lineage["train_rows"], dtype=np.int64),
        fit_split="train",
    )


def _reseal(artifact):
    value = copy.deepcopy(artifact)
    value.pop("artifact_sha256")
    value["artifact_sha256"] = canonical_sha256(value)
    return value


def test_fit_uses_float32_targets_float64_square_sum_exact_whole_train_and_no_mutation(lineage):
    surfaces = _surfaces()
    target, valid = surfaces[MFE]
    target[1, 0, 0], valid[1, 0, 0] = 0.0, False
    before = {name: (x.copy(), mask.copy()) for name, (x, mask) in surfaces.items()}
    result = _fit(lineage, surfaces)
    assert result == _fit(lineage, surfaces, chunks=(3,))
    assert result["fit_rows"] == 3
    assert result["fit_parent_rows_raw_sha256"] == hashlib.sha256(
        np.arange(3, dtype="<i8").tobytes()
    ).hexdigest()
    assert result["test_data_used"] is False
    assert result["model_forwards"] == result["optimizer_updates"] == 0
    for name, (raw, mask) in surfaces.items():
        x, m = (raw[..., :2], mask[..., :2]) if name == TERMINAL else (raw, mask)
        expected_count = m.sum(axis=0, dtype=np.int64)
        expected_sse = np.where(m, x.astype(np.float64) ** 2, 0).sum(axis=0, dtype=np.float64)
        saved = result["surfaces"][name]
        np.testing.assert_array_equal(saved["observed_count"], expected_count)
        np.testing.assert_array_equal(saved["sum_squared_target_bps2"], expected_sse)
        np.testing.assert_array_equal(saved["rms_bps"], np.sqrt(expected_sse / expected_count))
        assert saved["target_float32_storage_sha256"] == hashlib.sha256(
            np.asarray(x, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        assert saved["observed_mask_sha256"] == hashlib.sha256(
            np.ascontiguousarray(m).tobytes()
        ).hexdigest()
        np.testing.assert_array_equal(raw, before[name][0])
        np.testing.assert_array_equal(mask, before[name][1])
    raw = surfaces[FORECAST][0][:, 0]
    assert float(np.sum(raw.astype(np.float64) ** 2)) != float(
        np.sum(raw ** 2, dtype=np.float32)
    )


def test_nonzero_constant_target_rms_is_not_centered_or_replaced(lineage):
    surfaces = _surfaces()
    surfaces[FORECAST][0][:, 0] = np.float32(-7.0)
    result = _fit(lineage, surfaces)
    row = result["surfaces"][FORECAST]
    assert row["observed_count"][0] == 3
    assert row["sum_squared_target_bps2"][0] == 147.0
    assert row["rms_bps"][0] == 7.0
    contract = result["contract"]
    assert contract["centering"] is contract["epsilon_or_unit_fallback"] is False
    assert contract["raw_target_and_output_units"] == "bps"
    assert contract["native_decision_authority_granted"] is False


@pytest.mark.parametrize("fit_split", ["val", "VAL", "control", "CONTROL256", "test", "TEST"])
def test_fit_rejects_every_non_train_partition_before_consuming_targets(lineage, fit_split):
    def forbidden_batches():
        raise AssertionError("non-TRAIN targets were read")
        yield
    with pytest.raises(RuntimeError, match="TRAIN_ONLY_FIT"):
        scales.fit_dynamic_m5_loss_scales(
            batches=forbidden_batches(), lineage=lineage,
            expected_parent_rows=np.arange(3, dtype=np.int64), fit_split=fit_split,
        )


@pytest.mark.parametrize("fault", ["subset", "order", "dtype", "duplicate_batch", "missing_tail", "repeat_epoch"])
def test_fit_requires_exact_ascending_whole_train_once(lineage, fault):
    surfaces = _surfaces()
    parents = np.arange(3, dtype=np.int64)
    batches = [(parents, surfaces)]
    if fault == "subset":
        parents = np.array([0, 2], dtype=np.int64)
    elif fault == "order":
        parents = parents[::-1].copy()
    elif fault == "dtype":
        parents = parents.astype(np.int32)
    elif fault == "duplicate_batch":
        batches = [(np.array([0, 0, 2], dtype=np.int64), surfaces)]
    elif fault == "missing_tail":
        batches = [(np.array([0, 1], dtype=np.int64),
                    {k: (x[:2], m[:2]) for k, (x, m) in surfaces.items()})]
    else:
        batches = batches * 2
    with pytest.raises(RuntimeError, match="(WHOLE_TRAIN|BATCH_POPULATION)"):
        scales.fit_dynamic_m5_loss_scales(
            batches=batches, lineage=lineage, expected_parent_rows=parents, fit_split="train",
        )


@pytest.mark.parametrize("fault", ["zero_rms", "empty_coordinate", "masked_nonzero", "float64_target", "mask_dtype", "nonfinite", "flat_nonzero", "negative_mae"])
def test_fit_rejects_bad_storage_and_zero_coordinates_without_fallback(lineage, fault):
    surfaces = _surfaces()
    if fault == "zero_rms":
        surfaces[FORECAST][0][:, 0] = 0
    elif fault == "empty_coordinate":
        surfaces[MFE][0][:, 0, 0] = 0
        surfaces[MFE][1][:, 0, 0] = False
    elif fault == "masked_nonzero":
        surfaces[MFE][1][0, 0, 0] = False
    elif fault == "float64_target":
        surfaces[MFE] = (surfaces[MFE][0].astype(np.float64), surfaces[MFE][1])
    elif fault == "mask_dtype":
        surfaces[MFE] = (surfaces[MFE][0], surfaces[MFE][1].astype(np.int64))
    elif fault == "nonfinite":
        surfaces[FORECAST][0][0, 0] = np.nan
    elif fault == "flat_nonzero":
        surfaces[TERMINAL][0][0, 0, 2] = 1
    else:
        surfaces[MAE][0][0, 0, 0] = -1
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_LOSS_SCALE"):
        _fit(lineage, surfaces)


@pytest.mark.parametrize("fault", ["count_zero", "count_float", "count_exceeds_train", "sse", "rms", "shape", "missing_surface", "hash", "parent_population", "contract", "lineage", "model_claim"])
def test_resealed_artifact_rejects_semantic_stats_and_lineage_drift(lineage, fault):
    value = _fit(lineage)
    coordinate = value["surfaces"][FORECAST]
    if fault == "count_zero":
        coordinate["observed_count"][0] = 0
    elif fault == "count_float":
        coordinate["observed_count"][0] = 3.0
    elif fault == "count_exceeds_train":
        coordinate["observed_count"][0] = 4
    elif fault == "sse":
        coordinate["sum_squared_target_bps2"][0] += 1
    elif fault == "rms":
        coordinate["rms_bps"][0] += 1
    elif fault == "shape":
        coordinate["rms_bps"].pop()
    elif fault == "missing_surface":
        value["surfaces"].pop(CLOSED_MAE)
    elif fault == "hash":
        coordinate["observed_mask_sha256"] = "z" * 64
    elif fault == "parent_population":
        value["fit_parent_rows_raw_sha256"] = "b" * 64
    elif fault == "contract":
        value["contract"]["centering"] = True
    elif fault == "lineage":
        value["fit_lineage"]["train_control_cutoff"] = "2025-03-01T00:00:00Z"
    else:
        value["model_forwards"] = 1
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_LOSS_SCALE"):
        scales.require_dynamic_m5_loss_scale_artifact(_reseal(value), expected_lineage=lineage)


def test_load_binds_exact_artifact_and_returns_read_only_rms(lineage, tmp_path):
    artifact = _fit(lineage)
    path = tmp_path / "scales.json"
    binding = _bind(path, json.dumps(artifact, allow_nan=False).encode())
    loaded = scales.load_dynamic_m5_loss_scales(binding, expected_lineage=lineage)
    assert loaded["artifact_sha256"] == artifact["artifact_sha256"]
    assert loaded["fit_lineage"] == lineage
    for name, array in loaded["rms"].items():
        assert array.dtype == np.float32 and not array.flags.writeable
        np.testing.assert_array_equal(array, np.asarray(artifact["surfaces"][name]["rms_bps"], np.float32))
        with pytest.raises(ValueError):
            array.flat[0] = 1
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((RuntimeError, ValueError), match="SHA-256 mismatch"):
        scales.load_dynamic_m5_loss_scales(binding, expected_lineage=lineage)


def test_lineage_builder_joins_exact_train_manifest_witness_and_quote(lineage):
    witness = {
        "train_parquet": lineage["train_physical_source"]["parquet"],
        "train_manifest": lineage["train_physical_source"]["manifest"],
        "entry_eligibility_cutoff_utc": lineage["train_control_cutoff"],
        "train_rows": lineage["train_rows"],
        "eligible_parent_rows": lineage["eligible_parent_rows"],
    }
    scope = {
        "training_phase": "entry_only_dynamic_m5_path",
        "physical_sources": {"train": lineage["train_physical_source"]},
        "train_control_cutoff": lineage["train_control_cutoff"],
        "target_contract": lineage["target_contract"], "economics": lineage["economics"],
        "target_m1_source_sha256": lineage["source_quote_binding"]["sha256"],
    }
    population = {"witness": witness, "binding": lineage["population_witness"], "train_rows": 3}
    assert scales.dynamic_m5_loss_fit_lineage(scope, population) == lineage
    changed = copy.deepcopy(scope)
    changed["target_m1_source_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="M1_SOURCE_MISMATCH"):
        scales.dynamic_m5_loss_fit_lineage(changed, population)
    changed = copy.deepcopy(population)
    changed["witness"]["train_rows"] = 2
    with pytest.raises(RuntimeError, match="TRAIN_WITNESS_MISMATCH"):
        scales.dynamic_m5_loss_fit_lineage(scope, changed)


def _runtime_case(surface, *, closed=False):
    shape = scales.dynamic_m5_loss_scale_contract()["surface_shapes"][surface]
    width = 3 if surface == TERMINAL else shape[-1]
    prediction = torch.zeros((2, shape[0], width), dtype=torch.float32)
    target, valid = torch.zeros_like(prediction), torch.zeros_like(prediction, dtype=torch.bool)
    rms = torch.ones(shape, dtype=torch.float32)
    last = shape[0] - 1
    coordinates = [(0, 0, 0, 2., 6., 2.), (0, 0, 1, -4., -10., 3.),
                   (1, last, 0, 8., 4., 4.), (1, last, 1, -3., -6., 1.5)]
    if closed:
        coordinates = [(0, 0, 0, 1., 3., 1.), (0, 0, 1, -2., -6., 2.),
                       (1, last, 0, 3., 9., 3.)]
    for row, query, side, y, p, unit in coordinates:
        if "pre_mfe_mae" in surface:
            y, p = abs(y), abs(p)
        target[row, query, side], prediction[row, query, side] = y, p
        valid[row, query, side], rms[query, side] = True, unit
    if surface == TERMINAL:
        valid[..., 2] = valid[..., 0] & valid[..., 1]
    prediction[0, 1, 0] = 123.0  # Masked predictions are not fabricated supervision.
    return prediction.requires_grad_(), target, valid, rms


def test_runtime_terminal_handcomputed_units_gradients_raw_arrays_and_old_flat_denominator():
    prediction, target, valid, rms = _runtime_case(TERMINAL)
    before = [x.detach().clone() for x in (prediction, target, valid, rms)]
    sse, count = scales.dynamic_m5_scaled_square_sum(
        prediction, target, valid, surface=TERMINAL, rms=rms,
    )
    assert count == 6 and float(sse.detach()) == pytest.approx(13.0)
    loss = sse / count
    assert float(loss.detach()) == pytest.approx(13.0 / 6.0)
    loss.backward()
    expected = torch.zeros_like(prediction)
    expected[0, 0, :2] = torch.tensor([1/3, -2/9])
    expected[1, -1, :2] = torch.tensor([-1/12, -4/9])
    torch.testing.assert_close(prediction.grad, expected)
    for raw, unchanged in zip((prediction, target, valid, rms), before):
        assert torch.equal(raw.detach(), unchanged)
    assert target.grad is None and rms.grad is None


@pytest.mark.parametrize("inclusive,closed", [(MFE, CLOSED_MFE), (MAE, CLOSED_MAE)])
def test_runtime_path_endpoint_scales_and_unequal_counts_pool_with_exact_gradients(inclusive, closed):
    left = _runtime_case(inclusive)
    right = _runtime_case(closed, closed=True)
    sse_left, count_left = scales.dynamic_m5_scaled_square_sum(
        *left[:3], surface=inclusive, rms=left[3],
    )
    sse_right, count_right = scales.dynamic_m5_scaled_square_sum(
        *right[:3], surface=closed, rms=right[3],
    )
    assert (count_left, count_right) == (4, 3)
    assert float(sse_left.detach()) == pytest.approx(13.0)
    assert float(sse_right.detach()) == pytest.approx(12.0)
    loss = (sse_left + sse_right) / (count_left + count_right)
    assert float(loss.detach()) == pytest.approx(25.0 / 7.0)
    loss.backward()
    for prediction, target, valid, rms in (left, right):
        expected = torch.where(valid, 2 * (prediction.detach() - target) / (rms.square() * 7), 0)
        torch.testing.assert_close(prediction.grad, expected)
        assert torch.count_nonzero(prediction.grad[~valid]) == 0


def test_runtime_forecast_scales_base_and_day_coordinates_without_changing_raw_bps():
    width = scales.dynamic_m5_loss_scale_contract()["surface_shapes"][FORECAST][0]
    prediction = torch.zeros((2, width), dtype=torch.float32)
    target, valid = torch.zeros_like(prediction), torch.zeros_like(prediction, dtype=torch.bool)
    rms = torch.ones(width, dtype=torch.float32)
    prediction[0, 0], target[0, 0], rms[0], valid[0, 0] = 6., 2., 2., True
    prediction[1, -1], target[1, -1], rms[-1], valid[1, -1] = -9., -3., 3., True
    prediction[0, 1] = 123.
    prediction.requires_grad_()
    before = [x.detach().clone() for x in (prediction, target, valid, rms)]
    sse, count = scales.dynamic_m5_scaled_square_sum(
        prediction, target, valid, surface=FORECAST, rms=rms,
    )
    assert count == 2 and float(sse.detach()) == pytest.approx(8.0)
    (sse / count).backward()
    expected = torch.zeros_like(prediction)
    expected[0, 0], expected[1, -1] = 1., -2/3
    torch.testing.assert_close(prediction.grad, expected)
    for raw, unchanged in zip((prediction, target, valid, rms), before):
        assert torch.equal(raw.detach(), unchanged)


@pytest.mark.parametrize("fault", ["zero_unit", "nonfinite_unit", "learnable_unit", "unit_shape", "mask_dtype", "masked_target", "nonfinite_prediction", "flat_prediction", "flat_mask"])
def test_runtime_rejects_invalid_units_storage_and_flat_before_backward(fault):
    prediction, target, valid, rms = _runtime_case(TERMINAL)
    if fault == "zero_unit":
        rms[0, 0] = 0
    elif fault == "nonfinite_unit":
        rms[0, 0] = float("nan")
    elif fault == "learnable_unit":
        rms.requires_grad_()
    elif fault == "unit_shape":
        rms = rms[:-1]
    elif fault == "mask_dtype":
        valid = valid.float()
    elif fault == "masked_target":
        target[0, 1, 0] = 1
    elif fault == "nonfinite_prediction":
        prediction = prediction.detach().clone()
        prediction[0, 0, 0] = float("inf")
        prediction.requires_grad_()
    elif fault == "flat_prediction":
        prediction = prediction.detach().clone()
        prediction[0, 0, 2] = 1
        prediction.requires_grad_()
    else:
        valid[0, 0, 2] = False
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_LOSS_SCALE_RUNTIME"):
        scales.dynamic_m5_scaled_square_sum(prediction, target, valid, surface=TERMINAL, rms=rms)
    assert prediction.grad is None


def test_runtime_empty_surface_returns_zero_and_zero_count_without_fabrication():
    shape = scales.dynamic_m5_loss_scale_contract()["surface_shapes"][FORECAST]
    prediction = torch.ones((2, *shape), dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    sse, count = scales.dynamic_m5_scaled_square_sum(
        prediction, target, valid, surface=FORECAST, rms=torch.ones(shape, dtype=torch.float32),
    )
    assert count == 0 and float(sse.detach()) == 0.0
    sse.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
