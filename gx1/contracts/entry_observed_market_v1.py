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

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    causal_m1_target_contract, M5_PATH_OBSERVATION_HORIZON_BARS,
    M5_PATH_HORIZON_ORIGIN_POLICY_SHA256, M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN,
)
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


from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
M5_PATH_SCHEMA_VERSION = 'gx1_entry_observed_m5_path_v1'
M5_PATH_BATCH_TARGET_KEY = 'entry_observed_m5_path_utility_bps'


def entry_observed_m5_path_contract() -> dict[str, Any]:
    from gx1.contracts.entry_model_native_aux_targets_v3 import MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT
    value = {
        'schema_version': M5_PATH_SCHEMA_VERSION,
        'action_order': list(ENTRY_FITTED_Q_ACTION_ORDER), 'quote_contract': causal_m1_target_contract(),
        'target': 'signed_executable_mfe_minus_conservative_mae_before_first_executable_mfe_bps',
        'value_semantics': 'expected_price_path_utility_not_realizable_cash_return',
        'horizon_m5_bars': M5_PATH_OBSERVATION_HORIZON_BARS,
        'horizon_origin_policy_sha256': M5_PATH_HORIZON_ORIGIN_POLICY_SHA256,
        'horizon_origin': 'operator_reuses_existing_TRAIN_terminal_profit_chord_knee19_not_path_optimality_evidence',
        'peak': 'first_maximum_executable_bid_high_LONG_or_ask_low_SHORT',
        'mae_lower': 'all_prior_M1_buckets_plus_known_peak_bucket_open',
        'mae_upper': 'all_prior_M1_buckets_plus_entire_peak_M1_bucket',
        'primary_mae_supervision': 'conservative_upper_bound_no_assumed_intraminute_order',
        'primary_batch_target_key': M5_PATH_BATCH_TARGET_KEY,
        'label_columns': list(M5_PATH_TARGET_COLUMNS), 'valid_column': M5_PATH_VALID_COLUMN,
        'flat_target_bps': 0.0, 'negative_mfe_and_utility_preserved': True,
        'primary_mask_independent_of_long_horizon_availability': True,
        'support_forecast_layout': list(MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT),
        'support_forecast_masked_per_horizon': True,
        'exit_model_used': False, 'hindsight_extrema_are_supervision_only': True,
        'hindsight_best_exit_cash_claim': False, 'holding_time_limit': False,
        'fixed_loss_limit': False, 'score_is_calibrated_probability': False,
        'serving_authority_granted': False,
    }
    return {**value, 'contract_sha256': _sha(value)}


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
    declared = require_binding(recipe.get("entry_observed_market"), label="independent Entry design", verify_file=True)
    design = read_bound_json(Path(declared["path"]), declared["sha256"])
    if design.get("schema_version") == M5_PATH_SCHEMA_VERSION + "_learning_design":
        scope = _require_observed_market_plan(
            recipe, recipe_key="entry_observed_market", expected_schema=M5_PATH_SCHEMA_VERSION + "_learning_design",
            expected_target_contract=entry_observed_m5_path_contract(), expected_training_phase=ENTRY_ONLY_M5_PATH_TRAINING_PHASE)
        # The path window reuses the declared historical TRAIN decision.
        # A freshly fitted auxiliary policy may select another terminal horizon;
        # it cannot silently change this separately declared path target.
        scope["auxiliary_policy_horizon_m5_bars"] = scope["reference_horizon_m5_bars"]
        scope["reference_horizon_m5_bars"] = M5_PATH_OBSERVATION_HORIZON_BARS
        scope["elapsed_wall_clock_seconds"] = M5_PATH_OBSERVATION_HORIZON_BARS * ENTRY_DECISION_BAR_SECONDS
        for source in scope["physical_sources"].values():
            manifest = read_bound_json(Path(source["manifest"]["path"]), source["manifest"]["sha256"])
            if manifest.get("extra", {}).get("m5_path_supervision", {}).get("contract") != entry_observed_m5_path_contract():
                raise RuntimeError("ENTRY_M5_PATH_SOURCE_MANIFEST_INVALID")
        return scope
    return _require_observed_market_plan(
        recipe, recipe_key="entry_observed_market", expected_schema=SCHEMA_VERSION + "_learning_design",
        expected_target_contract=entry_observed_market_contract(), expected_training_phase="entry_only",
    )


def _require_observed_market_plan(
    recipe: Mapping[str, Any], *, recipe_key: str, expected_schema: str,
    expected_target_contract: Mapping[str, Any], expected_training_phase: str,
    extra_plan_keys: frozenset = frozenset(),
) -> dict[str, Any]:
    """Shared validation of one explicit observed-market learning design (v1 and its
    trading-day-horizon successor) against the frozen chronological prefix, the physical
    sources, the TRAIN-owned auxiliary policy and the declared cost owner."""
    from pathlib import Path
    from gx1.contracts.local_random_access_campaign_v2 import require_binding, read_bound_json
    from gx1.contracts.entry_causal_m1_target_policy_v1 import require_causal_m1_target_policy

    def load(binding, label):
        checked = require_binding(binding, label=label, verify_file=True)
        return checked, read_bound_json(Path(checked["path"]), checked["sha256"])

    binding, plan = load(recipe.get(recipe_key), "independent Entry design")
    file_keys = ("entry_train_parquet", "entry_train_manifest", "entry_val_parquet",
                 "entry_val_manifest", "train_cost_authority")
    expected_files = {key: recipe.get("files", {}).get(key) for key in file_keys}
    if (
        set(plan) != {"schema_version", "target_contract", "chronological_prefix", "files",
                      "training_phase", "exit_updates_entry", "native_launch_authorized", "test_data_used"} | set(extra_plan_keys)
        or plan["schema_version"] != expected_schema
        or plan["target_contract"] != expected_target_contract
        or plan["chronological_prefix"] != recipe.get("chronological_prefix")
        or not isinstance(plan["chronological_prefix"], Mapping)
        or set(plan["chronological_prefix"]) != {"design", "normalization_result", "labels_result", "native_coordinates"}
        or plan["files"] != expected_files
        or plan["training_phase"] != expected_training_phase
        or any(plan[key] is not False for key in ("exit_updates_entry", "native_launch_authorized", "test_data_used"))
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_DESIGN_INVALID")
    prefix = plan["chronological_prefix"]
    _, design = load(prefix["design"], "Entry input/coordinate design")
    _, labels = load(prefix["labels_result"], "Entry original label support")
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        M5_PHYSICAL_TARGET_BINDING_SCHEMA, M5_PRIMARY_OBSERVED_EMISSION_SCOPE,
        require_m5_primary_physical_target_receipt, require_model_native_aux_target_emission_contract,
    )
    if labels.get("schema_version") == M5_PHYSICAL_TARGET_BINDING_SCHEMA:
        require_m5_primary_physical_target_receipt(labels, design_binding=prefix["design"], context="ENTRY_OBSERVED_MARKET")
        if expected_training_phase != ENTRY_ONLY_M5_PATH_TRAINING_PHASE:
            raise RuntimeError("ENTRY_OBSERVED_MARKET_M5_EMISSION_PHASE_MISMATCH")
        calendar = design["calendar"]
        if design.get("scope", {}).get("entry_only_training_phase") != ENTRY_ONLY_M5_PATH_TRAINING_PHASE:
            raise RuntimeError("ENTRY_OBSERVED_MARKET_M5_EMISSION_PHASE_MISMATCH")
        lineage = []
        for split in ("train", "val"):
            source = calendar["source_bindings"][split]
            _, manifest = load(expected_files["entry_" + split + "_manifest"], "Entry M5 physical source manifest")
            extra = manifest["extra"]
            contract = require_model_native_aux_target_emission_contract(extra.get("aux_head_target_contract"), context="ENTRY_OBSERVED_MARKET")
            path = extra.get("m5_path_supervision", {})
            if (expected_files["entry_" + split + "_parquet"] != source["parquet"]
                    or expected_files["entry_" + split + "_manifest"] != source["manifest"]
                    or extra.get("emission_scope") != M5_PRIMARY_OBSERVED_EMISSION_SCOPE
                    or contract.get("emission_scope") != M5_PRIMARY_OBSERVED_EMISSION_SCOPE
                    or path.get("contract") != entry_observed_m5_path_contract()
                    or extra.get("rows") != source["physical_rows"]
                    or path.get("emitted_valid_rows") != source["physical_rows"]
                    or extra["aux_head_target_contract"]["primary_valid_rows_emitted"] != source["physical_rows"]):
                raise RuntimeError("ENTRY_OBSERVED_MARKET_M5_SOURCE_INVALID")
            lineage.append((extra.get("diagnostic_outcome_policy_sha256"), path.get("target_m1_source_sha256")))
        if (lineage[0] != lineage[1] or any(not isinstance(v, str) or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for v in lineage[0])):
            raise RuntimeError("ENTRY_OBSERVED_MARKET_M5_SOURCE_LINEAGE_INVALID")
        policy = {"policy_sha256": lineage[0][0], "m1_source_sha256": lineage[0][1]}
        horizon = entry_observed_m5_path_contract()["horizon_m5_bars"]
        elapsed_seconds = horizon * ENTRY_DECISION_BAR_SECONDS
    else:
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
        "binding": binding, "plan": plan, "target_contract": plan["target_contract"],
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

    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        M5_PRIMARY_OBSERVED_EMISSION_SCOPE, MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
        MODEL_NATIVE_FORECAST_VALID_COLUMNS,
        require_auxiliary_emission_training_phase, require_m5_forecast_observation_storage,
    )
    auxiliary_contract = getattr(dataset, "aux_head_target_contract", {})
    require_auxiliary_emission_training_phase(auxiliary_contract,
        entry_training_phase=scope["training_phase"], context="ENTRY_OBSERVED_MARKET")
    path_phase = scope["training_phase"] == ENTRY_ONLY_M5_PATH_TRAINING_PHASE
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
        or scope["training_phase"] not in ("entry_only", ENTRY_ONLY_M5_PATH_TRAINING_PHASE)
        or scope["target_contract"] != (entry_observed_m5_path_contract() if path_phase else entry_observed_market_contract())
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
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    columns = ([*M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN, *horizons.HORIZON_LABEL_COLUMNS]
               if path_phase else list(GROSS_TARGET_COLUMNS))
    if path_phase and auxiliary_contract.get("emission_scope") == M5_PRIMARY_OBSERVED_EMISSION_SCOPE:
        columns += [*MODEL_NATIVE_FORECAST_TARGET_COLUMNS, *MODEL_NATIVE_FORECAST_VALID_COLUMNS]
    frame = pq.read_table(dataset.parquet_path, columns=["time", *columns]).to_pandas()
    clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
    input_clock = pd.DatetimeIndex(pd.to_datetime(dataset.df["time"], utc=True)).as_unit("ns").asi8
    if (
        len(frame) != len(dataset.df) or not np.array_equal(clock, input_clock)
        or hashlib.sha256(np.asarray(clock, dtype="<i8").tobytes()).hexdigest() != source["clock_sha256"]
    ):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_DATASET_CLOCK_INVALID")
    admitted_mask = mask.copy()
    if path_phase:
        stored_valid = frame[M5_PATH_VALID_COLUMN].to_numpy()
        if not np.isin(stored_valid, (False, True)).all():
            raise RuntimeError("ENTRY_M5_PATH_AVAILABILITY_INVALID")
        finite = np.isfinite(frame[list(M5_PATH_TARGET_COLUMNS)].to_numpy(dtype=np.float64)).all(axis=1)
        if not np.array_equal(finite, stored_valid.astype(bool)):
            raise RuntimeError("ENTRY_M5_PATH_TARGET_AVAILABILITY_MISMATCH")
        admitted_mask &= finite
    rows = np.flatnonzero(admitted_mask)
    if not len(rows):
        raise RuntimeError("ENTRY_M5_PATH_NO_ADMITTED_ROWS")
    boundary = scope["train_control_cutoff" if split == "train" else "development_control_entry_end_exclusive"]
    if bool((clock[rows] + (ENTRY_DECISION_BAR_SECONDS + scope["elapsed_wall_clock_seconds"]) * 10**9
             > pd.Timestamp(boundary).value).any()):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_OUTCOME_CROSSES_SPLIT")
    support = None
    if path_phase:
        values = []
        for side in ("long", "short"):
            prefix = "y_" + side + "_m5_path_"
            mfe = frame.iloc[rows][prefix + "mfe_bps"].to_numpy(dtype=np.float64)
            lower = frame.iloc[rows][prefix + "mae_before_first_mfe_lower_bps"].to_numpy(dtype=np.float64)
            upper = frame.iloc[rows][prefix + "mae_before_first_mfe_upper_bps"].to_numpy(dtype=np.float64)
            utility = frame.iloc[rows][prefix + "path_utility_bps"].to_numpy(dtype=np.float64)
            peak = frame.iloc[rows][prefix + "first_mfe_m1_offset"].to_numpy(dtype=np.float64)
            if (np.any(lower < 0) or np.any(upper < lower) or not np.array_equal(utility, mfe - upper)
                    or np.any(peak < 0) or np.any(peak >= scope["elapsed_wall_clock_seconds"] // EXIT_DECISION_BAR_SECONDS)
                    or not np.equal(peak, np.floor(peak)).all()):
                raise RuntimeError("ENTRY_M5_PATH_TARGET_DOMAIN_INVALID")
            values.append(utility.astype(np.float32))
        observed = np.column_stack((*values, np.zeros(len(rows), np.float32)))
        support = horizons.build_masked_support_forecast_targets(frame=frame, scope=scope,
            split=split, admitted_mask=admitted_mask)
    else:
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
    if support is not None:
        if auxiliary_contract.get("emission_scope") == M5_PRIMARY_OBSERVED_EMISSION_SCOPE:
            m5_forecast_valid = require_m5_forecast_observation_storage(
                frame, context="ENTRY_OBSERVED_MARKET")
        else:
            # The legacy emitted contract proves every M5 forecast complete;
            # the explicit M5 emission contract must provide its own cell masks.
            m5_forecast_valid = np.ones((len(frame), len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS)), dtype=bool)
        m5_forecast_valid.setflags(write=False)
        dataset._entry_observed_market_m5_forecast_valid = m5_forecast_valid
        binding["m5_forecast_observed_cells"] = int(m5_forecast_valid[admitted_mask].sum())
        binding["m5_forecast_valid_sha256"] = hashlib.sha256(m5_forecast_valid.tobytes()).hexdigest()
        support_values, support_valid = support
        dataset._entry_observed_market_support_forecast = support_values
        dataset._entry_observed_market_support_forecast_valid = support_valid
        admitted_mask.setflags(write=False)
        dataset._entry_observed_market_admitted_rows = admitted_mask
        binding["support_forecast_observed_cells"] = int(support_valid.sum())
        binding["support_forecast_valid_sha256"] = hashlib.sha256(support_valid.tobytes()).hexdigest()
        binding["support_forecast_target_sha256"] = hashlib.sha256(support_values[support_valid].tobytes()).hexdigest()
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


def is_inactive_phase_parameter(name: str, entry_training_phase: str) -> bool:
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_TRAINING_PHASES
    if entry_training_phase not in ENTRY_ONLY_TRAINING_PHASES:
        return False
    if is_exit_owned_parameter(name):
        return True
    if entry_training_phase != ENTRY_ONLY_M5_PATH_TRAINING_PHASE:
        return False
    from gx1.contracts.entry_model_native_training_objective_v1 import active_joint_task_names
    from gx1.contracts.entry_model_native_joint_task_weighting_v1 import JOINT_TASK_NAMES
    inactive = set(JOINT_TASK_NAMES) - set(active_joint_task_names(entry_training_phase))
    return name.startswith(("head_side_mae.", "head_trendline_event.", "head_position_size.", "head_dip.",
        "head_timing.", "head_tail_risk.", "head_vol_forecast.")) or name.removeprefix("task_log_variances.") in inactive


def require_entry_only_gradients(model: torch.nn.Module, *, entry_training_phase: str = "entry_only") -> None:
    """An inactive Exit parameter must have grad=None, even with AdamW decay."""
    inactive = [(name, parameter) for name, parameter in model.named_parameters()
                if is_inactive_phase_parameter(name, entry_training_phase)]
    if not inactive or any(parameter.grad is not None for _, parameter in inactive):
        raise RuntimeError("ENTRY_OBSERVED_MARKET_EXIT_GRADIENT_FORBIDDEN")


def entry_learning_study_loader_order(epoch_order, *, study):
    """Repeat only the declared fit cohort; preserve the durable epoch cursor."""
    import numpy as np
    if (epoch_order.dtype != torch.int64 or epoch_order.ndim != 1
            or not np.array_equal(epoch_order.cpu().numpy(), study["rows"]["epoch_order"])):
        raise RuntimeError("ENTRY_STUDY_NATIVE_ORDER_CHANGED")
    if study["phase"] in ("curve", "epoch"):
        return epoch_order
    if "window_days" in study["phase_spec"]:
        # the frozen order restricted to the window rows first (relative order kept), the rest after;
        # the phase budget stops before the window is exhausted, so no epoch ever completes
        window = np.asarray(study["rows"][study["phase"]], dtype=np.int64)
        base = epoch_order.cpu().numpy()
        inside = np.isin(base, window)
        spec = study["phase_spec"]
        if (int(inside.sum()) != len(window) or len(np.unique(window)) != len(window)
                or spec["unique_train_rows"] != spec["optimizer_steps"] * 16
                or spec["unique_train_rows"] >= len(window)):
            raise RuntimeError("ENTRY_STUDY_RECENT_WINDOW_ORDER_INVALID")
        return torch.as_tensor(np.concatenate([base[inside], base[~inside]]).copy(), dtype=torch.int64)
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
    if ((study["phase"] not in ("curve", "epoch") and "window_days" not in study.get("phase_spec", {})) or not isinstance(original, Mapping)
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
    # the observed scope may be the v1 design or its trading-day-horizon successor; the horizons
    # owner dispatches on the scope (the v1 binder rejects the horizon scope by contract)
    from gx1.contracts.entry_observed_market_horizons_v1 import bind_observed_entry_dataset
    bind_observed_entry_dataset(view, study["observed_scope"])
    return view
