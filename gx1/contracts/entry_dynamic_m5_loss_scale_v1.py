"""TRAIN-owned loss units for the existing dynamic Entry components.

Raw targets, prediction heads and decision values stay in bps. This owner only
sets a frozen unit for squared residuals; it never chooses a trade or horizon.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256

DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY = "dynamic_m5_loss_scales"
DYNAMIC_M5_LOSS_SCALE_BATCH_KEY = "_dynamic_m5_loss_scales"
SCHEMA_VERSION = "gx1_dynamic_m5_train_target_rms_loss_units_v1"
DYNAMIC_M5_TERMINAL_HORIZON_GROUPS = (("early_1_15", 1, 15), ("later_16_480", 16, 480))
SURFACE_NAMES = (
    "dynamic_terminal_net_markout_bps", "dynamic_mfe_bps",
    "dynamic_pre_mfe_mae_upper_bps", "dynamic_closed_mfe_bps",
    "dynamic_closed_pre_mfe_mae_upper_bps", "forecast_pred",
)


def dynamic_m5_loss_scale_contract() -> dict[str, Any]:
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
    from gx1.contracts.entry_model_native_aux_targets_v3 import MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT
    curves = len(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
    value = {
        "schema_version": SCHEMA_VERSION,
        "statistic": "sqrt(sum_observed_float32_target_squared_in_float64/observed_count)",
        "fit_partition": "train", "fit_population": "all_physical_TRAIN_rows_once_in_ascending_order",
        "centering": False, "epsilon_or_unit_fallback": False,
        "unobserved_coordinate_or_zero_rms": "reject_before_optimizer",
        "query_minutes": list(DYNAMIC_M1_PREFIX_HORIZON_MINUTES),
        "side_order": ["long", "short"],
        "surface_shapes": {name: ([len(MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT)]
            if name == "forecast_pred" else [curves, 2]) for name in SURFACE_NAMES},
        "forecast_layout": [dict(x) for x in MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT],
        "raw_target_and_output_units": "bps", "loss_residual_units": "TRAIN_target_RMS_units",
        "loss": "sum_observed(((prediction_bps-target_bps)/TRAIN_RMS_bps)^2)/observed_count",
        "flat_terminal": "analytic_zero_residual_without_division_with_original_observed_denominator",
        "path_pooling": "separate_endpoint_scales_then_existing_shared_task_observed_denominator",
        "learned_task_weighting_preserved": True, "fixed_relative_task_weights": False,
        "future_outcomes_as_inputs": False, "native_decision_authority_granted": False,
    }
    return {**value, "contract_sha256": canonical_sha256(value)}


def _require_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"fit_partition", "train_rows", "train_control_cutoff", "train_physical_source",
            "population_witness", "eligible_parent_rows", "target_contract", "economics",
            "source_quote_binding", "index_binding"}
    if (not isinstance(value, Mapping) or set(value) != keys
            or value.get("fit_partition") != "train" or type(value.get("train_rows")) is not int
            or value["train_rows"] < 1):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_FIT_LINEAGE_INVALID")
    from gx1.contracts.entry_observed_market_v1 import (
        entry_observed_dynamic_m5_path_contract, require_entry_observed_market_economics,
    )
    if value["target_contract"] != entry_observed_dynamic_m5_path_contract():
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_TARGET_CONTRACT_INVALID")
    require_entry_observed_market_economics(value["economics"])
    source = value["train_physical_source"]
    if (not isinstance(source, Mapping) or source.get("physical_rows") != value["train_rows"]
            or not isinstance(value["train_control_cutoff"], str)):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_FIT_POPULATION_INVALID")
    from gx1.contracts.local_random_access_campaign_v2 import require_binding
    for binding in (source.get("parquet"), source.get("manifest"),
                    value["population_witness"], value["eligible_parent_rows"]):
        require_binding(binding, label="dynamic loss scale TRAIN lineage", verify_file=False)
    canonical_sha256(value)  # Reject non-JSON/nonfinite metadata, never repair it.
    return dict(value)


def dynamic_m5_loss_fit_lineage(scope: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    """Use stable data/target owners, not a later initialization/recipe path."""
    from pathlib import Path
    from gx1.contracts.local_random_access_campaign_v2 import require_binding, read_bound_json
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if scope.get("training_phase") != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_PHASE_INVALID")
    source = scope["physical_sources"]["train"]
    witness = population["witness"]
    if (witness.get("train_parquet") != source["parquet"]
            or witness.get("train_manifest") != source["manifest"]
            or witness.get("entry_eligibility_cutoff_utc") != scope["train_control_cutoff"]
            or witness.get("train_rows") != population["train_rows"]):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_TRAIN_WITNESS_MISMATCH")
    binding = require_binding(source["manifest"], label="dynamic loss scale TRAIN manifest", verify_file=True)
    meta = read_bound_json(Path(binding["path"]), binding["sha256"])["extra"]["dynamic_m5_supervision"]
    if meta["source_quote_binding"]["sha256"] != scope["target_m1_source_sha256"]:
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_M1_SOURCE_MISMATCH")
    return _require_lineage({
        "fit_partition": "train", "train_rows": population["train_rows"],
        "train_control_cutoff": scope["train_control_cutoff"], "train_physical_source": source,
        "population_witness": population["binding"], "eligible_parent_rows": witness["eligible_parent_rows"],
        "target_contract": scope["target_contract"], "economics": scope["economics"],
        "source_quote_binding": meta["source_quote_binding"], "index_binding": meta["index_binding"],
    })


def _target_surface(name: str, target: Any, valid: Any, *, rows: int):
    shape = tuple(dynamic_m5_loss_scale_contract()["surface_shapes"][name])
    if (not isinstance(target, np.ndarray) or target.dtype != np.float32
            or not isinstance(valid, np.ndarray) or valid.dtype != np.bool_
            or target.shape != valid.shape or not np.isfinite(target).all()
            or np.any(target[~valid] != 0)):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_TARGET_STORAGE_INVALID:" + name)
    if name == "dynamic_terminal_net_markout_bps":
        if (target.shape != (rows, shape[0], 3) or np.any(target[..., 2] != 0)
                or not np.array_equal(valid[..., 2], valid[..., 0] & valid[..., 1])):
            raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_FLAT_INVALID")
        target, valid = target[..., :2], valid[..., :2]
    if target.shape != (rows, *shape):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_TARGET_SHAPE_INVALID:" + name)
    if "pre_mfe_mae" in name and np.any(target[valid] < 0):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_NEGATIVE_MAE")
    return target, valid


def fit_dynamic_m5_loss_scales(*, batches: Iterable, lineage: Mapping[str, Any],
                               expected_parent_rows: np.ndarray, fit_split: str) -> dict[str, Any]:
    """Consume target-only chunks, never instantiate a model or read features."""
    if fit_split != "train":
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_TRAIN_ONLY_FIT")
    lineage = _require_lineage(lineage)
    n = lineage["train_rows"]
    if (not isinstance(expected_parent_rows, np.ndarray) or expected_parent_rows.dtype != np.int64
            or not np.array_equal(expected_parent_rows, np.arange(n, dtype=np.int64))):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_WHOLE_TRAIN_REQUIRED")
    contract = dynamic_m5_loss_scale_contract()
    accum = {k: {"count": np.zeros(contract["surface_shapes"][k], np.int64),
        "sumsq": np.zeros(contract["surface_shapes"][k], np.float64),
        "target_hash": hashlib.sha256(), "mask_hash": hashlib.sha256()} for k in SURFACE_NAMES}
    cursor = 0
    for parents, surfaces in batches:
        if (not isinstance(parents, np.ndarray) or parents.dtype != np.int64 or parents.ndim != 1
                or not len(parents) or not np.array_equal(parents, expected_parent_rows[cursor:cursor+len(parents)])
                or not isinstance(surfaces, Mapping) or set(surfaces) != set(SURFACE_NAMES)):
            raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_BATCH_POPULATION_INVALID")
        for name in SURFACE_NAMES:
            target, valid = _target_surface(name, *surfaces[name], rows=len(parents))
            row = accum[name]
            row["count"] += valid.sum(axis=0, dtype=np.int64)
            row["sumsq"] += np.where(valid, target.astype(np.float64)**2, 0).sum(axis=0, dtype=np.float64)
            row["target_hash"].update(np.asarray(target, dtype="<f4").tobytes(order="C"))
            row["mask_hash"].update(np.ascontiguousarray(valid).tobytes())
        cursor += len(parents)
    if cursor != n:
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_WHOLE_TRAIN_INCOMPLETE")
    surfaces = {}
    for name, row in accum.items():
        count, sse = row["count"], row["sumsq"]
        if np.any(count < 1) or not np.isfinite(sse).all() or np.any(sse <= 0):
            raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_EMPTY_OR_ZERO_COORDINATE:" + name)
        rms = np.sqrt(sse/count)
        if not np.isfinite(rms.astype(np.float32)).all() or np.any(rms.astype(np.float32) <= 0):
            raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_FLOAT32_RANGE_INVALID:" + name)
        surfaces[name] = {"observed_count": count.tolist(), "sum_squared_target_bps2": sse.tolist(),
            "rms_bps": rms.tolist(), "target_float32_storage_sha256": row["target_hash"].hexdigest(),
            "observed_mask_sha256": row["mask_hash"].hexdigest()}
    value = {"schema_version": SCHEMA_VERSION+"_artifact", "contract": contract,
        "fit_lineage": lineage, "fit_rows": n,
        "fit_parent_rows_raw_sha256": hashlib.sha256(expected_parent_rows.astype("<i8", copy=False).tobytes()).hexdigest(),
        "surfaces": surfaces, "test_data_used": False, "model_forwards": 0, "optimizer_updates": 0}
    value["artifact_sha256"] = canonical_sha256(value)
    return require_dynamic_m5_loss_scale_artifact(value, expected_lineage=lineage)


def require_dynamic_m5_loss_scale_artifact(value: Mapping[str, Any], *, expected_lineage: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"schema_version", "contract", "fit_lineage", "fit_rows", "fit_parent_rows_raw_sha256", "surfaces",
            "test_data_used", "model_forwards", "optimizer_updates", "artifact_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_ARTIFACT_INVALID")
    result = dict(value); claimed = result.pop("artifact_sha256")
    if (claimed != canonical_sha256(result) or result["schema_version"] != SCHEMA_VERSION+"_artifact"
            or result["contract"] != dynamic_m5_loss_scale_contract()
            or result["fit_lineage"] != _require_lineage(expected_lineage)
            or type(result["fit_rows"]) is not int or result["fit_rows"] != expected_lineage["train_rows"]
            or result["test_data_used"] is not False
            or type(result["model_forwards"]) is not int or result["model_forwards"] != 0
            or type(result["optimizer_updates"]) is not int or result["optimizer_updates"] != 0
            or not isinstance(result["surfaces"], Mapping) or set(result["surfaces"]) != set(SURFACE_NAMES)):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_ARTIFACT_BINDING_INVALID")
    expected_parent_sha = hashlib.sha256(np.arange(result["fit_rows"], dtype="<i8").tobytes()).hexdigest()
    if result["fit_parent_rows_raw_sha256"] != expected_parent_sha:
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_PARENT_POPULATION_INVALID")
    for name in SURFACE_NAMES:
        x = result["surfaces"][name]
        if (not isinstance(x, Mapping) or set(x) != {"observed_count", "sum_squared_target_bps2", "rms_bps",
                "target_float32_storage_sha256", "observed_mask_sha256"}):
            raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_COORDINATE_SCHEMA_INVALID:" + name)
        count, sse, rms = (np.asarray(x[k]) for k in ("observed_count", "sum_squared_target_bps2", "rms_bps"))
        shape = tuple(result["contract"]["surface_shapes"][name])
        if (count.dtype.kind not in "iu" or sse.dtype.kind not in "fi" or rms.dtype.kind not in "fi"
                or count.shape != shape or sse.shape != shape or rms.shape != shape
                or np.any(count < 1) or np.any(count > result["fit_rows"])
                or not np.isfinite(sse).all() or np.any(sse <= 0)
                or not np.isfinite(rms).all() or np.any(rms <= 0)
                or not np.array_equal(rms, np.sqrt(sse.astype(np.float64)/count))
                or not np.isfinite(rms.astype(np.float32)).all() or np.any(rms.astype(np.float32) <= 0)
                or any(not isinstance(x[k], str) or len(x[k]) != 64 or any(c not in "0123456789abcdef" for c in x[k])
                       for k in ("target_float32_storage_sha256", "observed_mask_sha256"))):
            raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_COORDINATE_STATS_INVALID:" + name)
    return {**result, "artifact_sha256": claimed}


def load_dynamic_m5_loss_scales(binding: Mapping[str, Any], *, expected_lineage: Mapping[str, Any]) -> dict[str, Any]:
    from pathlib import Path
    from gx1.contracts.local_random_access_campaign_v2 import require_binding, read_bound_json
    binding = require_binding(binding, label="dynamic M5 TRAIN loss scales", verify_file=True)
    value = require_dynamic_m5_loss_scale_artifact(
        read_bound_json(Path(binding["path"]), binding["sha256"]), expected_lineage=expected_lineage)
    rms = {k: np.asarray(value["surfaces"][k]["rms_bps"], dtype=np.float32) for k in SURFACE_NAMES}
    for array in rms.values():array.setflags(write=False)
    return {"binding": binding, "artifact_sha256": value["artifact_sha256"],
        "contract_sha256": value["contract"]["contract_sha256"], "fit_lineage": value["fit_lineage"], "rms": rms}



def dynamic_m5_loss_scale_identity(loaded: Mapping[str, Any]) -> dict[str, Any]:
    """One serializable dependency identity shared by recipe/session/readouts."""
    from gx1.contracts.local_random_access_campaign_v2 import require_binding
    if (not isinstance(loaded, Mapping)
            or not {"binding", "artifact_sha256", "contract_sha256", "fit_lineage", "rms"}.issubset(loaded)
            or loaded["contract_sha256"] != dynamic_m5_loss_scale_contract()["contract_sha256"]
            or not isinstance(loaded["artifact_sha256"], str) or len(loaded["artifact_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in loaded["artifact_sha256"])):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_LOADED_IDENTITY_INVALID")
    # The loader already verified files, full TRAIN lineage and RMS values.
    # Runtime identity construction is pure metadata; never hash TRAIN per batch.
    return {"binding": require_binding(loaded["binding"], label="dynamic loss scale identity", verify_file=False),
        "artifact_sha256": loaded["artifact_sha256"], "contract_sha256": loaded["contract_sha256"],
        "fit_lineage_sha256": canonical_sha256(loaded["fit_lineage"])}


def dynamic_m5_scaled_square_sum(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                                 *, surface: str, rms: torch.Tensor,
                                 terminal_query_range: tuple[int, int] | None = None) -> tuple[torch.Tensor, int]:
    """Scale errors only; the known FLAT residual contributes zero to its old count."""
    if surface not in SURFACE_NAMES:
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_SURFACE_INVALID")
    if not all(isinstance(x, torch.Tensor) for x in (prediction, target, valid, rms)):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_RUNTIME_INVALID:" + surface)
    shape = tuple(dynamic_m5_loss_scale_contract()["surface_shapes"][surface])
    if (not isinstance(rms, torch.Tensor) or rms.requires_grad or tuple(rms.shape) != shape
            or rms.dtype != torch.float32 or rms.device != prediction.device
            or not bool(torch.isfinite(rms).all()) or bool((rms <= 0).any())
            or prediction.shape != target.shape or target.shape != valid.shape
            or prediction.dtype != torch.float32 or target.dtype != torch.float32 or valid.dtype != torch.bool
            or target.device != prediction.device or valid.device != prediction.device
            or not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all())
            or bool((target[~valid] != 0).any())):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_RUNTIME_INVALID:" + surface)
    count = int(valid.sum())
    if surface == "dynamic_terminal_net_markout_bps":
        if (tuple(prediction.shape[1:]) != (shape[0], 3) or bool((prediction[...,2] != 0).any())
                or bool((target[...,2] != 0).any()) or not torch.equal(valid[...,2], valid[...,0] & valid[...,1])):
            raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_RUNTIME_FLAT_INVALID")
        if terminal_query_range is not None:
            if (type(terminal_query_range) is not tuple or len(terminal_query_range) != 2
                    or any(type(x) is not int for x in terminal_query_range)
                    or terminal_query_range not in tuple((a, b) for _, a, b in DYNAMIC_M5_TERMINAL_HORIZON_GROUPS)):
                raise RuntimeError("DYNAMIC_M5_TERMINAL_QUERY_RANGE_INVALID")
            start, end = terminal_query_range
            prediction, target, valid = (x[:, start-1:end] for x in (prediction, target, valid))
            rms = rms[start-1:end]
            count = int(valid.sum())  # Count genuine LONG/SHORT/analytic-FLAT before slicing sides.
        prediction, target, valid = prediction[...,:2], target[...,:2], valid[...,:2]
    elif terminal_query_range is not None:
        raise RuntimeError("DYNAMIC_M5_TERMINAL_QUERY_RANGE_INVALID")
    elif tuple(prediction.shape[1:]) != shape:
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_RUNTIME_SHAPE_INVALID")
    if "pre_mfe_mae" in surface and (bool((prediction < 0).any()) or bool((target[valid] < 0).any())):
        raise RuntimeError("DYNAMIC_M5_LOSS_SCALE_NEGATIVE_MAE")
    return (((prediction-target)/rms)[valid]).square().sum(), count


def dynamic_m5_terminal_horizon_objective_contract() -> dict[str, Any]:
    """The selected terminal reduction; the frozen111 loss-unit owner is unchanged."""
    value = {
        "schema_version": "gx1_dynamic_m5_terminal_horizon_objective_v1",
        "task": "dynamic_terminal_net_markout_bps",
        "groups": [{"name": name, "query_minutes": [start, end]}
                   for name, start, end in DYNAMIC_M5_TERMINAL_HORIZON_GROUPS],
        "residual_units": "TRAIN_target_RMS_units",
        "loss_unit_contract_sha256": dynamic_m5_loss_scale_contract()["contract_sha256"],
        "group_numerator": "sum_observed_LONG_SHORT_scaled_squared_residuals",
        "group_denominator": "true_observed_LONG_SHORT_analytic_FLAT_cells",
        "reduction": "arithmetic_mean_of_nonempty_group_means",
        "one_nonempty_group": "that_group_mean",
        "no_nonempty_groups": "omit_terminal_task_and_its_log_variance_term",
        "joint_task_precision": "existing_exp_minus_s_times_objective_plus_s_once_per_active_terminal_task",
        "raw_and_pooled_TRAIN_RMS_diagnostics_preserved": True,
        "optimizer_objective_is_pooled_SSE_over_N": False,
        "epsilon_clipping_or_unit_fallback": False,
        "native_decision_authority_granted": False,
    }
    return {**value, "contract_sha256": canonical_sha256(value)}


def dynamic_m5_terminal_horizon_objective(prediction: torch.Tensor, target: torch.Tensor,
                                        valid: torch.Tensor, *, rms: torch.Tensor
                                        ) -> tuple[torch.Tensor | None, dict[str, int]]:
    """Mean observed early/later group means, with the true FLAT-inclusive counts."""
    means, cells = [], {}
    for name, start, end in DYNAMIC_M5_TERMINAL_HORIZON_GROUPS:
        squared, count = dynamic_m5_scaled_square_sum(prediction, target, valid,
            surface="dynamic_terminal_net_markout_bps", rms=rms, terminal_query_range=(start, end))
        cells[name] = count
        if count:
            means.append(squared / count)
    return (torch.stack(means).mean() if len(means) == 2 else means[0] if means else None), cells
