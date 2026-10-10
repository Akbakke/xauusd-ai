"""Entry supervision from observed executable market outcomes, without Exit.

This is a new target authority for the operator decision of 2026-10-09.
It preserves signed market outcomes and the existing TRAIN-owned reference
horizon. That observation horizon is not a trading time limit. Native
admission, stage ownership and serving acceptance must be bound separately.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch

from gx1.contracts.entry_causal_m1_outcomes_v1 import causal_m1_target_contract
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS, EXIT_DECISION_BAR_SECONDS
from gx1.contracts.entry_fitted_q_v1 import ENTRY_FITTED_Q_ACTION_ORDER
from gx1.contracts.entry_model_native_aux_targets_v3 import MODEL_NATIVE_AUX_FORECAST_HORIZONS
from gx1.contracts.unified_exit_economics_objective_v2 import SECONDS_PER_YEAR
from gx1.contracts.unified_exit_prospective_cost_policy_v1 import require_cost_parameter_authority

SCHEMA_VERSION = "gx1_entry_observed_market_v1"
GROSS_TARGET_COLUMNS = (
    "y_long_final_pnl_at_direction_horizon_bps",
    "y_short_final_pnl_at_direction_horizon_bps",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def entry_observed_market_contract() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "action_order": list(ENTRY_FITTED_Q_ACTION_ORDER),
        "gross_target_columns": list(GROSS_TARGET_COLUMNS),
        "quote_contract": causal_m1_target_contract(),
        "horizon_source": "frozen_TRAIN_causal_m1_policy.selected_direction_horizon_bars",
        "horizon_unit_seconds": ENTRY_DECISION_BAR_SECONDS,
        "target": "observed_executable_terminal_cash_return_net_of_declared_costs",
        "target_unit": "raw_bps",
        "flat_target_bps": 0.0,
        "exit_model_used": False,
        "future_outcomes_used_as_model_inputs": False,
        "hindsight_best_exit_used": False,
        "negative_outcomes_clipped": False,
        "capital_hurdle_is_cash_cost": False,
        "holding_time_limit": False,
        "reference_horizon_search_allowed": False,
        "retained_forecast_horizons_m5": list(MODEL_NATIVE_AUX_FORECAST_HORIZONS),
        "score_is_calibrated_probability": False,
        "serving_authority_granted": False,
    }
    return {**value, "contract_sha256": _sha(value)}


def entry_observed_market_economics(
    authority: Mapping[str, Any], *, coverage_start_utc: Any,
    coverage_end_utc: Any, verify_local_sources: bool = True,
) -> dict[str, Any]:
    """Validate the existing cost owner once before materializing target rows."""
    checked = require_cost_parameter_authority(
        authority, expected_coverage_start_utc=coverage_start_utc,
        expected_coverage_end_utc=coverage_end_utc,
        verify_local_sources=verify_local_sources,
    )
    parameters = checked["parameters"]
    # A nonzero account-currency fee would need an explicit notional/FX owner.
    if parameters["guaranteed_execution_fee"]["account_currency_per_execution"] != 0.0:
        raise RuntimeError("ENTRY_OBSERVED_MARKET_FEE_CONVERSION_REQUIRED")
    value = {
        "schema_version": SCHEMA_VERSION + "_economics",
        "cost_authority_sha256": checked["authority_sha256"],
        "coverage_start_utc": checked["coverage_start_utc"],
        "coverage_end_utc_exclusive": checked["coverage_end_utc_exclusive"],
        "round_trip_execution_cost_bps": 2.0 * (
            parameters["commission"]["bps_per_execution"]
            + parameters["execution_slippage"]["central_bps_per_execution"]
        ),
        "annual_financing_cost_bps": [
            parameters["financing_or_swap"][side + "_annual_cost_rate"] * 10_000.0
            for side in ("long", "short")
        ],
        "seconds_per_year": SECONDS_PER_YEAR,
        "historical_cost_truth_qualified": False,
        "capital_hurdle_included": False,
        "risk_utility_included": False,
    }
    return {**value, "economics_sha256": _sha(value)}


def require_entry_observed_market_economics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an immutable projection; native admission must bind its owner."""
    keys = {
        "schema_version", "cost_authority_sha256", "coverage_start_utc",
        "coverage_end_utc_exclusive", "round_trip_execution_cost_bps",
        "annual_financing_cost_bps", "seconds_per_year",
        "historical_cost_truth_qualified", "capital_hurdle_included",
        "risk_utility_included", "economics_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("ENTRY_OBSERVED_MARKET_ECONOMICS_INVALID")
    result = dict(value)
    claimed = result.pop("economics_sha256")
    try:
        digest = _sha(result)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ENTRY_OBSERVED_MARKET_ECONOMICS_INVALID") from exc
    rates = result["annual_financing_cost_bps"]
    authority = result["cost_authority_sha256"]
    values = [result["round_trip_execution_cost_bps"], *(rates if isinstance(rates, list) else [])]
    if (
        digest != claimed or result["schema_version"] != SCHEMA_VERSION + "_economics"
        or not isinstance(authority, str) or len(authority) != 64
        or any(c not in "0123456789abcdef" for c in authority)
        or not isinstance(rates, list) or len(rates) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values)
        or result["seconds_per_year"] != SECONDS_PER_YEAR
        or any(result[k] is not False for k in (
            "historical_cost_truth_qualified", "capital_hurdle_included", "risk_utility_included"
        ))
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_ECONOMICS_INVALID")
    return {**result, "economics_sha256": claimed}


@torch.no_grad()
def build_entry_observed_market_targets(
    *, gross_return_bps: torch.Tensor, elapsed_wall_clock_seconds: torch.Tensor,
    economics: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Net LONG/SHORT markouts and known FLAT=0; never query a model.

    Caller binds complete, split-contained M1 support before emitting rows.
    Missing/censored observations must be excluded by that owner, never
    replaced here with zero, a forecast or an Exit bootstrap.
    """
    cost = require_entry_observed_market_economics(economics)
    gross, elapsed = gross_return_bps, elapsed_wall_clock_seconds
    if (
        not isinstance(gross, torch.Tensor) or gross.ndim != 2
        or gross.shape[0] < 1 or gross.shape[1] != 2
        or gross.dtype not in (torch.float32, torch.float64)
        or not isinstance(elapsed, torch.Tensor) or elapsed.shape != (gross.shape[0],)
        or elapsed.device != gross.device or elapsed.dtype != torch.int64
        or not bool(torch.isfinite(gross).all()) or not bool((elapsed > 0).all())
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_TARGET_INPUT_INVALID")
    # Source target columns are float32; calculate declared costs in float64
    # and perform one final cast, without clipping or renormalizing outcomes.
    annual = torch.tensor(cost["annual_financing_cost_bps"], dtype=torch.float64, device=gross.device)
    financing = elapsed.to(torch.float64)[:, None] * annual[None, :] / SECONDS_PER_YEAR
    net = (gross.to(torch.float64) - cost["round_trip_execution_cost_bps"] - financing).to(gross.dtype)
    targets = torch.cat((net, torch.zeros_like(net[:, :1])), dim=1)
    if not bool(torch.isfinite(targets).all()):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_TARGET_NONFINITE")
    return {
        "target_bps": targets,
        "target_valid": torch.ones_like(targets, dtype=torch.bool),
        "financing_cost_bps": financing,
    }


def require_entry_observed_market_scope(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Bind an explicit successor to existing immutable data and cost owners.

    The base chronological prefix continues to own inputs, normalization and
    row selection. Only this explicit amendment owns Entry supervision and
    gradient staging; the base design's old Entry/Exit target link is inactive.
    """
    from pathlib import Path
    from gx1.contracts.local_random_access_campaign_v2 import require_binding, read_bound_json
    from gx1.contracts.entry_causal_m1_target_policy_v1 import require_causal_m1_target_policy

    def load(binding, label):
        checked = require_binding(binding, label=label, verify_file=True)
        return checked, read_bound_json(Path(checked["path"]), checked["sha256"])

    binding, plan = load(recipe.get("entry_observed_market"), "independent Entry design")
    file_keys = ("entry_train_parquet", "entry_train_manifest", "entry_val_parquet",
                 "entry_val_manifest", "train_cost_authority")
    expected_files = {key: recipe.get("files", {}).get(key) for key in file_keys}
    if (
        set(plan) != {"schema_version", "target_contract", "chronological_prefix", "files",
                      "training_phase", "exit_updates_entry", "native_launch_authorized", "test_data_used"}
        or plan["schema_version"] != SCHEMA_VERSION + "_learning_design"
        or plan["target_contract"] != entry_observed_market_contract()
        or plan["chronological_prefix"] != recipe.get("chronological_prefix")
        or not isinstance(plan["chronological_prefix"], Mapping)
        or set(plan["chronological_prefix"]) != {"design", "normalization_result", "labels_result", "native_coordinates"}
        or plan["files"] != expected_files
        or plan["training_phase"] != "entry_only"
        or any(plan[key] is not False for key in ("exit_updates_entry", "native_launch_authorized", "test_data_used"))
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_DESIGN_INVALID")
    prefix = plan["chronological_prefix"]
    _, design = load(prefix["design"], "Entry input/coordinate design")
    _, labels = load(prefix["labels_result"], "Entry original label support")
    _, label_plan = load(labels["plan"], "Entry original label proof plan")
    _, signal = load(label_plan["input_bindings"]["signal"], "Entry TRAIN target policy source")
    policy = require_causal_m1_target_policy(signal["feature_ranking"]["entry_direction_target_policy"])
    calendar = design["calendar"]
    if (
        labels.get("test_accessed") is not False or labels.get("optimizer_steps") != 0
        or labels.get("forbidden_access_attempts") != []
        or labels.get("identical_frozen_policies_across_physical_splits") is not True
        or policy["fit_scope"] != "TRAIN_ONLY" or policy["val_test_rows_used_for_fit"] != 0
        or policy["policy_sha256"] != signal["feature_ranking"]["entry_direction_target_policy_sha256"]
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_POLICY_OR_SUPPORT_INVALID")
    horizon = policy["selected_direction_horizon_bars"]
    elapsed_seconds = horizon * ENTRY_DECISION_BAR_SECONDS
    for split in ("train", "val"):
        source = calendar["source_bindings"][split]
        proof = labels["splits"][split]
        _, manifest = load(expected_files["entry_" + split + "_manifest"], "Entry physical source manifest")
        if (
            expected_files["entry_" + split + "_parquet"] != source["parquet"]
            or expected_files["entry_" + split + "_manifest"] != source["manifest"]
            or proof["source_manifest"] != source["manifest"]
            or proof["direction_policy_sha256"] != policy["policy_sha256"]
            or manifest["extra"]["diagnostic_outcome_policy_sha256"] != policy["policy_sha256"]
            or proof["rows"] != source["physical_rows"]
            or proof["exact_m1_fill_and_contiguous_outcome_rows"] != proof["rows"]
            or proof["policy_horizon_m5_bars"] != horizon
            or proof["policy_horizon_m1_bars"] * EXIT_DECISION_BAR_SECONDS != elapsed_seconds
        ):
            raise RuntimeError("ENTRY_OBSERVED_MARKET_SOURCE_OR_HORIZON_INVALID")
    _, authority = load(expected_files["train_cost_authority"], "Entry declared net costs")
    economics = entry_observed_market_economics(
        authority, coverage_start_utc=calendar["train_entry_start_inclusive"],
        coverage_end_utc=calendar["development_control_entry_end_exclusive"],
    )
    return {
        "binding": binding, "target_contract": plan["target_contract"],
        "training_phase": plan["training_phase"], "economics": economics,
        "reference_horizon_m5_bars": horizon, "elapsed_wall_clock_seconds": elapsed_seconds,
        "direction_policy_sha256": policy["policy_sha256"],
        "target_m1_source_sha256": policy["m1_source_sha256"],
        "physical_sources": calendar["source_bindings"],
        "train_control_cutoff": calendar["train_control_cutoff"],
        "development_control_entry_end_exclusive": calendar["development_control_entry_end_exclusive"],
    }


def entry_observed_market_identity(scope: Mapping[str, Any]) -> dict[str, Any]:
    """The same target identity is bound in recipe, checkpoint and measurement."""
    return {
        "design": scope["binding"],
        "target_contract": scope["target_contract"],
        "economics": scope["economics"],
        "training_phase": scope["training_phase"],
        "reference_horizon_m5_bars": scope["reference_horizon_m5_bars"],
        "elapsed_wall_clock_seconds": scope["elapsed_wall_clock_seconds"],
        "direction_policy_sha256": scope["direction_policy_sha256"],
        "target_m1_source_sha256": scope["target_m1_source_sha256"],
    }


def bind_entry_observed_market_dataset(dataset: Any, scope: Mapping[str, Any]) -> None:
    """Read the two existing outcome columns once; keep inputs and auxiliaries."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from pathlib import Path
    from gx1.contracts.local_random_access_campaign_v2 import require_binding

    if getattr(dataset, "_entry_observed_market_binding", None) is not None:
        raise RuntimeError("ENTRY_OBSERVED_MARKET_ALREADY_BOUND")
    auxiliary = getattr(dataset, "_policy_dependent_auxiliary_binding", None)
    mask = getattr(dataset, "_policy_dependent_auxiliary_bound_rows", None)
    if (
        not isinstance(auxiliary, Mapping) or auxiliary.get("mode") != "original_physical_split_targets"
        or auxiliary.get("role") not in ("TRAIN", "CONTROL256", "CONTROL_STUDY")
        or (auxiliary.get("role") == "CONTROL_STUDY"
            and not isinstance(auxiliary.get("entry_learning_study"), Mapping))
        or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
        or mask.shape != (len(dataset.df),) or not bool(mask.any())
        or scope["training_phase"] != "entry_only"
        or scope["target_contract"] != entry_observed_market_contract()
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_DATASET_SCOPE_INVALID")
    split = "train" if auxiliary["role"] == "TRAIN" else "val"
    source = scope["physical_sources"][split]
    if (
        Path(dataset.parquet_path) != Path(source["parquet"]["path"])
        or auxiliary.get("parent_entry_parquet") != source["parquet"]
        or auxiliary.get("parent_entry_manifest") != source["manifest"]
        or auxiliary.get("policies", {}).get("direction_policy", {}).get("policy_sha256")
        != scope["direction_policy_sha256"]
        or len(dataset.df) != source["physical_rows"]
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_DATASET_SOURCE_INVALID")
    require_binding(source["parquet"], label="observed Entry physical targets", verify_file=True)
    frame = pq.read_table(dataset.parquet_path, columns=["time", *GROSS_TARGET_COLUMNS]).to_pandas()
    clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
    input_clock = pd.DatetimeIndex(pd.to_datetime(dataset.df["time"], utc=True)).as_unit("ns").asi8
    if (
        len(frame) != len(dataset.df) or not np.array_equal(clock, input_clock)
        or hashlib.sha256(np.asarray(clock, dtype="<i8").tobytes()).hexdigest() != source["clock_sha256"]
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_DATASET_CLOCK_INVALID")
    rows = np.flatnonzero(mask)
    boundary = scope["train_control_cutoff" if split == "train" else "development_control_entry_end_exclusive"]
    if bool((clock[rows] + (ENTRY_DECISION_BAR_SECONDS + scope["elapsed_wall_clock_seconds"]) * 10**9
             > pd.Timestamp(boundary).value).any()):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_OUTCOME_CROSSES_SPLIT")
    gross = frame.iloc[rows][list(GROSS_TARGET_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    observed = build_entry_observed_market_targets(
        gross_return_bps=torch.from_numpy(gross),
        elapsed_wall_clock_seconds=torch.full((len(rows),), scope["elapsed_wall_clock_seconds"], dtype=torch.int64),
        economics=scope["economics"],
    )["target_bps"].numpy()
    targets = np.full((len(frame), 3), np.nan, dtype=np.float32)
    targets[rows] = observed
    targets.setflags(write=False)
    binding = {
        "schema_version": SCHEMA_VERSION + "_dataset",
        "identity": entry_observed_market_identity(scope),
        "role": auxiliary["role"], "physical_source": source,
        "admitted_rows": int(len(rows)),
        "admitted_row_indices_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
        "admitted_target_bps_sha256": hashlib.sha256(np.asarray(observed, dtype="<f4").tobytes()).hexdigest(),
    }
    dataset._entry_observed_market_targets = targets
    dataset._entry_observed_market_binding = {**binding, "binding_sha256": _sha(binding)}


def entry_observed_market_batch_targets(
    batch: Mapping[str, Any], *, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = batch.get("entry_observed_market_target_bps")
    if (
        not isinstance(targets, torch.Tensor) or targets.dtype != torch.float32
        or targets.ndim != 2 or targets.shape[1] != 3
        or targets.shape[0] < 1 or not bool(torch.isfinite(targets).all())
        or not bool((targets[:, 2] == 0.0).all())
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_BATCH_TARGET_INVALID")
    targets = targets.detach().to(device)
    return targets, torch.ones_like(targets, dtype=torch.bool)


def is_exit_owned_parameter(name: str) -> bool:
    """Exit-specific weights, including its token projection and task weight."""
    return (
        name.startswith(("exit_", "head_exit_action.", "entry_decision_token."))
        or name == "task_log_variances.unified_exit_action"
    )


def require_entry_only_gradients(model: torch.nn.Module) -> None:
    """An inactive Exit parameter must have grad=None, even with AdamW decay."""
    inactive = [(name, parameter) for name, parameter in model.named_parameters()
                if is_exit_owned_parameter(name)]
    if not inactive or any(parameter.grad is not None for _, parameter in inactive):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_EXIT_GRADIENT_FORBIDDEN")


def entry_learning_study_loader_order(epoch_order, *, study):
    """Repeat only the declared fit cohort; preserve the durable epoch cursor."""
    import numpy as np
    if (epoch_order.dtype != torch.int64 or epoch_order.ndim != 1
            or not np.array_equal(epoch_order.cpu().numpy(), study["rows"]["epoch_order"])):
        raise RuntimeError("ENTRY_STUDY_NATIVE_ORDER_CHANGED")
    if study["phase"] == "curve":
        return epoch_order
    if study["phase"] != "fit":
        raise RuntimeError("ENTRY_STUDY_PHASE_INVALID")
    order = epoch_order.clone()
    draws = study["phase_spec"]["optimizer_steps"] * 16
    cohort = study["rows"]["fit"]
    if draws > len(order) or len(cohort) != 256 or len(np.unique(cohort)) != 256:
        raise RuntimeError("ENTRY_STUDY_FIT_ORDER_INVALID")
    order[:draws] = torch.as_tensor(np.resize(cohort, draws).copy(), dtype=torch.int64)
    return order


def entry_learning_study_control_dataset(dataset, *, study):
    """Bind a separate read-only evaluation view of the original physical VAL."""
    import copy
    import numpy as np
    original = getattr(dataset, "_policy_dependent_auxiliary_binding", None)
    if (study["phase"] != "curve" or not isinstance(original, Mapping)
            or original.get("role") != "CONTROL256"
            or original.get("mode") != "original_physical_split_targets"
            or original["parent_entry_parquet"] != study["observed_scope"]["physical_sources"]["val"]["parquet"]):
        raise RuntimeError("ENTRY_STUDY_CONTROL_SOURCE_INVALID")
    rows = study["rows"]["control"]
    if len(rows) != 4096 or not np.array_equal(rows, np.unique(rows)) or rows[-1] >= len(dataset.df):
        raise RuntimeError("ENTRY_STUDY_CONTROL_ROWS_INVALID")
    view = copy.copy(dataset)
    mask = np.zeros(len(dataset.df), dtype=bool)
    mask[rows] = True
    mask.setflags(write=False)
    view._policy_dependent_auxiliary_bound_rows = mask
    view._policy_dependent_auxiliary_binding = {
        **original, "role":"CONTROL_STUDY", "rows":len(rows),
        "row_binding":study["plan"]["row_bindings"]["control"],
        "entry_learning_study":study["selection"]}
    view._entry_observed_market_binding = None
    bind_entry_observed_market_dataset(view, study["observed_scope"])
    return view
