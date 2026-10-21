"""Campaign resume bindings for the existing native candidate session.

The campaign cursor binds the canonical TRAIN pointer/state and any active
VAL cursor together. It is a transport receipt, not another training state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.local_random_access_campaign_v2 import (
    canonical_sha256, file_sha256, read_bound_json, require_binding,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import canonical_sha256 as native_sha256
from gx1.contracts.unified_exit_reference_policy_v1 import require_reference_policy_contract
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import native_checkpoint_monitor
from gx1.contracts.unified_exit_random_access_index_v1 import (
    LATEST_YEAR_ROOT_SCHEMA_VERSION, require_random_access_index_root,
)


CURSOR_SCHEMA = "gx1_native_candidate_campaign_cursor_v1"
NATIVE_KIND = "native_candidate_window"
NATIVE_PHASE = "native_candidate"
NATIVE_MODULE = "gx1.scripts.run_unified_exit_native_candidate_window_v1"
WINDOW_SCHEMA = "gx1_native_candidate_window_policy_v1"
PREFIX_RESUME_EQUIVALENCE_SCHEMA = "gx1_prefix_resume_equivalence_plan_v1"

DYNAMIC_M5_TECHNICAL_CURSOR_CEILING = 2
DYNAMIC_M5_TECHNICAL_QUALIFICATION_SCHEMA = "gx1_dynamic_m5_technical_qualification_v1"
DYNAMIC_M5_TECHNICAL_FIXED_SCOPE = {
    "schema_version": DYNAMIC_M5_TECHNICAL_QUALIFICATION_SCHEMA,
    "batch_size": 16, "train_entries": 32, "optimizer_steps": DYNAMIC_M5_TECHNICAL_CURSOR_CEILING,
    "technical_optimizer_updates": 4, "control_optimizer_updates": 0,
    "parity_entries_per_role": 16, "parity_model_forwards": 4,
    "parity_absolute_tolerance_bps": 1e-4, "parity_relative_tolerance": 0.0,
    "max_invocations": 1, "max_invocation_seconds": 12000,
    "teacher_refresh_allowed": False, "full_epoch_training_allowed": False,
    "full_val_allowed": False, "private_weights_reusable": False,
    "native_launch_authorized": False, "test_data_used": False,
}


OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA = "gx1_candidate_optimizer_procedure_transition_v1"
OPTIMIZER_PROCEDURE_TRANSITION_POLICY = "separate_model_and_task_weights_v1"
OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_NAME = "CANDIDATE_OPTIMIZER_PROCEDURE_TRANSITION.json"
OPTIMIZER_PROCEDURE_TRANSITION_RECEIPT_SCHEMA = "gx1_candidate_optimizer_procedure_transition_receipt_v1"
OPTIMIZER_PROCEDURE_ORIGIN_CONTRACT_SHA256 = "19a1286b2c6e977246a8b2f890c8295489135586eb05b194ebf71418c8b47d4c"
OPTIMIZER_PROCEDURE_ORIGIN_POINTER_SHA256 = "d265ce2cba4f494eb818870093d8cb9834be974e2113729232f1ffdcd070ebb7"
OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256 = "9f9aaba84a7f3fd031666b9761d606af03b7c534409cf3049021b023805db974"
OPTIMIZER_PROCEDURE_ORIGIN_CURSOR = {
    "checkpoint_index": 85, "phase": "train", "epoch_index": 1,
    "next_batch_offset": 1152, "global_optimizer_steps": 5233, "complete": False,
}


TRAINING_CONTINUATION_SCHEMA = "gx1_candidate_training_continuation_origin_v1"
TRAINING_CONTINUATION_RECEIPT_NAME = "CANDIDATE_TRAINING_CONTINUATION.json"
TRAINING_CONTINUATION_RECEIPT_SCHEMA = "gx1_candidate_training_continuation_receipt_v1"
TRAINING_CONTINUATION_ORIGIN_CONTRACT_SHA256 = "1f8f3e3a81c3563f93e2c17ed436a36260c76728ff5b78e813d7ee73bb77ec93"
TRAINING_CONTINUATION_ORIGIN_POINTER_SHA256 = "49c48c14c9cd0820ddaaa62b8bef35d4eaccc5c86301c5494ee1d88537501c61"
TRAINING_CONTINUATION_ORIGIN_STATE_SHA256 = "4d1d47484765a7c1ba9af161aa0fb1338e740018ceb8b218e49954e7769f3292"
TRAINING_CONTINUATION_ORIGIN_CURSOR = {
    "checkpoint_index": 87, "phase": "train", "epoch_index": 1,
    "next_batch_offset": 1184, "global_optimizer_steps": 5265, "complete": False,
}
TRAINING_CONTINUATION_STEP_CEILING = 5521


FQI_TARGET_REFRESH_SCHEMA = "gx1_candidate_fqi_target_refresh_origin_v1"
FQI_TARGET_REFRESH_RECEIPT_NAME = "CANDIDATE_FQI_TARGET_REFRESH.json"
FQI_TARGET_REFRESH_RECEIPT_SCHEMA = "gx1_candidate_fqi_target_refresh_receipt_v1"
FQI_TARGET_REFRESH_ORIGIN_CONTRACT_SHA256 = "aa5e390dfdd5bb5c284445438f25dbd76eaa844137ef481192948d6274ab68fa"
FQI_TARGET_REFRESH_ORIGIN_POINTER_SHA256 = "4b47b5156a9d0dea3fe45769d65152713c1c77606f5333b6b2293541c6123bb7"
FQI_TARGET_REFRESH_ORIGIN_STATE_SHA256 = "0fb167f23a41d4f454f7350b73484d533a7fad3e0fca7a51dc516a4d84484f21"
FQI_TARGET_REFRESH_ORIGIN_CURSOR = {
    "checkpoint_index": 91, "phase": "train", "epoch_index": 1,
    "next_batch_offset": 1440, "global_optimizer_steps": 5521, "complete": False,
}
FQI_TARGET_REFRESH_STEP_CEILING = 5777


ENTRY_LEARNABILITY_SCHEMA = "gx1_candidate_entry_learnability_origin_v1"
ENTRY_LEARNABILITY_RECEIPT_NAME = "CANDIDATE_ENTRY_LEARNABILITY.json"
ENTRY_LEARNABILITY_RECEIPT_SCHEMA = "gx1_candidate_entry_learnability_receipt_v1"
ENTRY_LEARNABILITY_ORIGIN_CONTRACT_SHA256 = "25c32556b456ac7f155109993b8eccf9d074c37d167ae3fae1285ecfbffdbdae"
ENTRY_LEARNABILITY_ORIGIN_POINTER_SHA256 = "f508837dfdbc06615fbad52bd824d89f8750944b57a77ca5a13bab1d4798acf8"
ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256 = "ca18cdf27f1ddf22741c893456d2040e1e53ce0cf1ac8e9532cb780973fd5dbe"
ENTRY_LEARNABILITY_ORIGIN_CURSOR = {
    "checkpoint_index": 95, "phase": "train", "epoch_index": 1,
    "next_batch_offset": 1696, "global_optimizer_steps": 5777, "complete": False,
}
ENTRY_LEARNABILITY_STEP_CEILING = 6033
ENTRY_LEARNABILITY_COHORT_SHA256 = "1cea9754bf115b75fc3267337b919feeffd0c0a8fd2d7509f2254eba2a4da889"
ENTRY_LEARNABILITY_TARGET_MODEL_SHA256 = "b8e635dcf6c15ef812f0025fdc088e57ceac703081a290ff682a29920b406a81"
ENTRY_LEARNABILITY_REPLAY_POLICY = {
    "epoch_index": 1, "native_batch_offsets": [1440, 1525, 1610, 1695],
    "batch_offset_start": 1696, "batch_offset_end": 1952,
    "repeats": 64, "parent_row_count": 64,
}


# Continuation of the completed fixed64 diagnostic, never production coverage.
ENTRY_LEARNABILITY_CONTINUATION_CONTRACT_SHA256 = "e208ed4d094f743aa764c23d7074fcc98b49f8b69649469fe99cc02ccf5df0ff"
ENTRY_LEARNABILITY_CONTINUATION_POINTER_SHA256 = "9b84ab0e830b41a89afaf52b1e482f04bb417ef3a418d2595ca64f8c5c0bacbf"
ENTRY_LEARNABILITY_CONTINUATION_STATE_SHA256 = "f513a0931c8b4620baf15dfed99f04a6e3f6042d576515ca599640c9d24cd4aa"
ENTRY_LEARNABILITY_CONTINUATION_CURSOR = {
    "checkpoint_index": 99, "phase": "train", "epoch_index": 1,
    "next_batch_offset": 1952, "global_optimizer_steps": 6033, "complete": False,
}
ENTRY_LEARNABILITY_CONTINUATION_STEP_CEILING = 7057
ENTRY_LEARNABILITY_CONTINUATION_REPLAY_POLICY = {
    **ENTRY_LEARNABILITY_REPLAY_POLICY, "batch_offset_start": 1952,
    "batch_offset_end": 2976, "repeats": 256,
}


def training_continuation_control(origin: Any) -> dict[str, Any]:
    """Preserve stopped87; admit non-replay95 only through its next complete VAL."""
    through_val = (isinstance(origin, Mapping) and isinstance(origin.get("contract"), Mapping)
                   and origin["contract"].get("sha256") == ENTRY_LEARNABILITY_ORIGIN_CONTRACT_SHA256)
    return {
        "contract_sha256": ENTRY_LEARNABILITY_ORIGIN_CONTRACT_SHA256 if through_val else TRAINING_CONTINUATION_ORIGIN_CONTRACT_SHA256,
        "pointer_sha256": ENTRY_LEARNABILITY_ORIGIN_POINTER_SHA256 if through_val else TRAINING_CONTINUATION_ORIGIN_POINTER_SHA256,
        "state_sha256": ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256 if through_val else TRAINING_CONTINUATION_ORIGIN_STATE_SHA256,
        "cursor": dict(ENTRY_LEARNABILITY_ORIGIN_CURSOR if through_val else TRAINING_CONTINUATION_ORIGIN_CURSOR),
        "completed_val_ceiling": 2 if through_val and "exit_backup_steps" not in origin and "exit_reference_policy" not in origin else None,
    }


def native_completed_val_ceiling(recipe: Mapping[str, Any]) -> int | None:
    """Used only together with require_native_run_scope's bound policy checks."""
    origin = recipe.get("candidate_resume_origin")
    if isinstance(origin, Mapping) and origin.get("schema_version") == TRAINING_CONTINUATION_SCHEMA:
        return training_continuation_control(origin)["completed_val_ceiling"]
    return None


def entry_learnability_control(origin: Any = None) -> dict[str, Any]:
    """Select one of the two immutable controls; origin validation binds all hashes."""
    continued = (isinstance(origin, Mapping) and isinstance(origin.get("contract"), Mapping)
                 and origin["contract"].get("sha256") == ENTRY_LEARNABILITY_CONTINUATION_CONTRACT_SHA256)
    return {
        "contract_sha256": ENTRY_LEARNABILITY_CONTINUATION_CONTRACT_SHA256 if continued else ENTRY_LEARNABILITY_ORIGIN_CONTRACT_SHA256,
        "pointer_sha256": ENTRY_LEARNABILITY_CONTINUATION_POINTER_SHA256 if continued else ENTRY_LEARNABILITY_ORIGIN_POINTER_SHA256,
        "state_sha256": ENTRY_LEARNABILITY_CONTINUATION_STATE_SHA256 if continued else ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256,
        "cursor": dict(ENTRY_LEARNABILITY_CONTINUATION_CURSOR if continued else ENTRY_LEARNABILITY_ORIGIN_CURSOR),
        "step_ceiling": ENTRY_LEARNABILITY_CONTINUATION_STEP_CEILING if continued else ENTRY_LEARNABILITY_STEP_CEILING,
        "replay_policy": dict(ENTRY_LEARNABILITY_CONTINUATION_REPLAY_POLICY if continued else ENTRY_LEARNABILITY_REPLAY_POLICY),
    }


def require_entry_learnability_cohort(value: Any) -> dict[str, Any]:
    """Validate the supplied bound cohort; the origin owns its immutable file SHA."""
    if not isinstance(value, Mapping):
        raise RuntimeError("NATIVE_ENTRY_LEARNABILITY_COHORT_INVALID")
    policy = ENTRY_LEARNABILITY_REPLAY_POLICY
    if (type(value.get("epoch_index")) is not int or value["epoch_index"] != policy["epoch_index"]
            or type(value.get("full_training_epoch_index")) is not int
            or value["full_training_epoch_index"] != policy["epoch_index"]
            or value.get("actual_native_batch_offsets") != policy["native_batch_offsets"]
            or any(not isinstance(value.get(key), str) or len(value[key]) != 64
                   or any(char not in "0123456789abcdef" for char in value[key])
                   for key in ("epoch_order_sha256", "selected_sample_plan_sha256"))
            or value.get("selection_uses_losses_or_outcomes") is not False
            or value.get("replacement_of_unfavorable_rows_allowed") is not False
            or any(not isinstance(value.get(key), list) or len(value[key]) != policy["parent_row_count"]
                   or any(type(row) is not int or row < 0 for row in value[key])
                   or len(set(value[key])) != len(value[key]) for key in ("parent_rows", "child_rows"))):
        raise RuntimeError("NATIVE_ENTRY_LEARNABILITY_COHORT_INVALID")
    return dict(value)


def require_entry_learnability_origin(
    origin: Any, *, verify_files: bool = True,
) -> dict[str, Any]:
    """Admit only the bound95 or99 origin for its finite four-batch replay."""
    fields = {"schema_version", "contract", "pointer", "exit_value_initialization",
              "train_population_scope", "gradient_clipping_policy", "cohort"}
    if (type(verify_files) is not bool or not isinstance(origin, Mapping)
            or set(origin) != fields
            or origin.get("schema_version") != ENTRY_LEARNABILITY_SCHEMA
            or origin.get("gradient_clipping_policy") != OPTIMIZER_PROCEDURE_TRANSITION_POLICY
            or origin.get("exit_value_initialization") != "close_now_baseline_v1"
            or origin.get("train_population_scope") != "latest_year_2025_2026_v1"):
        raise RuntimeError("NATIVE_ENTRY_LEARNABILITY_ORIGIN_INVALID")
    contract = require_binding(origin["contract"], label="Entry learnability origin contract", verify_file=verify_files)
    pointer = require_binding(origin["pointer"], label="Entry learnability origin pointer", verify_file=verify_files)
    control = entry_learnability_control(origin)
    if (contract["sha256"] != control["contract_sha256"]
            or pointer["sha256"] != control["pointer_sha256"]):
        raise RuntimeError("NATIVE_ENTRY_LEARNABILITY_ORIGIN_HASH_MISMATCH")
    cohort = require_binding(origin["cohort"], label="Entry learnability cohort", verify_file=verify_files)
    if cohort["sha256"] != ENTRY_LEARNABILITY_COHORT_SHA256:
        raise RuntimeError("NATIVE_ENTRY_LEARNABILITY_COHORT_HASH_MISMATCH")
    if verify_files:
        require_entry_learnability_cohort(read_bound_json(Path(cohort["path"]), cohort["sha256"]))
        current = read_bound_json(Path(pointer["path"]), pointer["sha256"])
        expected = {
            **control["cursor"],
            "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
            "session_contract_sha256": contract["sha256"],
            "state_sha256": control["state_sha256"],
        }
        if (set(current) != set(expected)
                or any(current.get(k) != v or type(current.get(k)) is not type(v)
                       for k, v in expected.items())):
            raise RuntimeError("NATIVE_ENTRY_LEARNABILITY_ORIGIN_CURSOR_INVALID")
        require_binding({
            "path": str(Path(pointer["path"]).parent / "candidate_training_state_slot_0.pt"),
            "sha256": control["state_sha256"],
        }, label="Entry learnability origin state", verify_file=True)
    return dict(origin)


def require_fqi_target_refresh_origin(
    origin: Any, *, verify_files: bool = True,
) -> dict[str, Any]:
    """Admit only stopped checkpoint91 for one explicit online-to-target copy."""
    fields = {"schema_version", "contract", "pointer", "exit_value_initialization",
              "train_population_scope", "gradient_clipping_policy"}
    if (type(verify_files) is not bool or not isinstance(origin, Mapping)
            or set(origin) != fields
            or origin.get("schema_version") != FQI_TARGET_REFRESH_SCHEMA
            or origin.get("gradient_clipping_policy") != OPTIMIZER_PROCEDURE_TRANSITION_POLICY
            or origin.get("exit_value_initialization") != "close_now_baseline_v1"
            or origin.get("train_population_scope") != "latest_year_2025_2026_v1"):
        raise RuntimeError("NATIVE_FQI_TARGET_REFRESH_ORIGIN_INVALID")
    contract = require_binding(origin["contract"], label="FQI refresh origin contract", verify_file=verify_files)
    pointer = require_binding(origin["pointer"], label="FQI refresh origin pointer", verify_file=verify_files)
    if (contract["sha256"] != FQI_TARGET_REFRESH_ORIGIN_CONTRACT_SHA256
            or pointer["sha256"] != FQI_TARGET_REFRESH_ORIGIN_POINTER_SHA256):
        raise RuntimeError("NATIVE_FQI_TARGET_REFRESH_ORIGIN_HASH_MISMATCH")
    if verify_files:
        current = read_bound_json(Path(pointer["path"]), pointer["sha256"])
        expected = {
            **FQI_TARGET_REFRESH_ORIGIN_CURSOR,
            "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
            "session_contract_sha256": contract["sha256"],
            "state_sha256": FQI_TARGET_REFRESH_ORIGIN_STATE_SHA256,
        }
        if (set(current) != set(expected)
                or any(current.get(k) != v or type(current.get(k)) is not type(v)
                       for k, v in expected.items())):
            raise RuntimeError("NATIVE_FQI_TARGET_REFRESH_ORIGIN_CURSOR_INVALID")
        require_binding({
            "path": str(Path(pointer["path"]).parent / "candidate_training_state_slot_0.pt"),
            "sha256": FQI_TARGET_REFRESH_ORIGIN_STATE_SHA256,
        }, label="FQI refresh origin state", verify_file=True)
    return dict(origin)


def require_training_continuation_origin(
    origin: Any, *, verify_files: bool = True,
) -> dict[str, Any]:
    """Admit immutable stopped87 or non-replay95 with the same learning state."""
    fields = {"schema_version", "contract", "pointer", "exit_value_initialization",
              "train_population_scope", "gradient_clipping_policy"}
    if (type(verify_files) is not bool or not isinstance(origin, Mapping)
            or not fields <= set(origin) <= fields | {"exit_backup_steps", "exit_reference_policy"}
            or origin.get("schema_version") != TRAINING_CONTINUATION_SCHEMA
            or origin.get("gradient_clipping_policy") != OPTIMIZER_PROCEDURE_TRANSITION_POLICY
            or origin.get("exit_value_initialization") != "close_now_baseline_v1"
            or origin.get("train_population_scope") != "latest_year_2025_2026_v1"):
        raise RuntimeError("NATIVE_TRAINING_CONTINUATION_ORIGIN_INVALID")
    if "exit_backup_steps" in origin and (
            type(origin["exit_backup_steps"]) is not int or origin["exit_backup_steps"] != 5
            or not isinstance(origin.get("contract"), Mapping)
            or origin["contract"].get("sha256") != ENTRY_LEARNABILITY_ORIGIN_CONTRACT_SHA256):
        raise RuntimeError("NATIVE_TRACE_BACKUP_ORIGIN_INVALID")
    if "exit_reference_policy" in origin:
        require_reference_policy_contract(origin["exit_reference_policy"])
        if ("exit_backup_steps" in origin or not isinstance(origin.get("contract"), Mapping)
                or origin["contract"].get("sha256") != ENTRY_LEARNABILITY_ORIGIN_CONTRACT_SHA256):
            raise RuntimeError("NATIVE_REFERENCE_POLICY_ORIGIN_INVALID")
    control = training_continuation_control(origin)
    contract = require_binding(origin["contract"], label="continuation origin contract", verify_file=verify_files)
    pointer = require_binding(origin["pointer"], label="continuation origin pointer", verify_file=verify_files)
    if (contract["sha256"] != control["contract_sha256"]
            or pointer["sha256"] != control["pointer_sha256"]):
        raise RuntimeError("NATIVE_TRAINING_CONTINUATION_ORIGIN_HASH_MISMATCH")
    if verify_files:
        current = read_bound_json(Path(pointer["path"]), pointer["sha256"])
        expected = {
            **control["cursor"],
            "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
            "session_contract_sha256": contract["sha256"],
            "state_sha256": control["state_sha256"],
        }
        if (set(current) != set(expected)
                or any(current.get(k) != v or type(current.get(k)) is not type(v)
                       for k, v in expected.items())):
            raise RuntimeError("NATIVE_TRAINING_CONTINUATION_ORIGIN_CURSOR_INVALID")
        require_binding({
            "path": str(Path(pointer["path"]).parent / "candidate_training_state_slot_0.pt"),
            "sha256": control["state_sha256"],
        }, label="continuation origin state", verify_file=True)
    return dict(origin)


def require_optimizer_procedure_origin(
    origin: Any, *, verify_files: bool = True,
) -> dict[str, Any]:
    """Admit only the measured, stopped checkpoint85 for a procedure change.

    Historical Exit initialization is a lineage declaration, never a request
    to initialize it again. The trainer owns destination/source/state checks.
    """
    fields = {"schema_version", "contract", "pointer", "exit_value_initialization",
              "train_population_scope", "gradient_clipping_policy"}
    if (type(verify_files) is not bool or not isinstance(origin, Mapping)
            or set(origin) != fields
            or origin.get("schema_version") != OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA
            or origin.get("gradient_clipping_policy") != OPTIMIZER_PROCEDURE_TRANSITION_POLICY
            or origin.get("exit_value_initialization") != "close_now_baseline_v1"
            or origin.get("train_population_scope") != "latest_year_2025_2026_v1"):
        raise RuntimeError("NATIVE_OPTIMIZER_PROCEDURE_ORIGIN_INVALID")
    contract = require_binding(origin["contract"], label="optimizer origin contract", verify_file=verify_files)
    pointer = require_binding(origin["pointer"], label="optimizer origin pointer", verify_file=verify_files)
    if (contract["sha256"] != OPTIMIZER_PROCEDURE_ORIGIN_CONTRACT_SHA256
            or pointer["sha256"] != OPTIMIZER_PROCEDURE_ORIGIN_POINTER_SHA256):
        raise RuntimeError("NATIVE_OPTIMIZER_PROCEDURE_ORIGIN_HASH_MISMATCH")
    if verify_files:
        current = read_bound_json(Path(pointer["path"]), pointer["sha256"])
        expected = {
            **OPTIMIZER_PROCEDURE_ORIGIN_CURSOR,
            "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
            "session_contract_sha256": contract["sha256"],
            "state_sha256": OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256,
        }
        if (set(current) != set(expected)
                or any(current.get(k) != v or type(current.get(k)) is not type(v)
                       for k, v in expected.items())):
            raise RuntimeError("NATIVE_OPTIMIZER_PROCEDURE_ORIGIN_CURSOR_INVALID")
        require_binding({
            "path": str(Path(pointer["path"]).parent / "candidate_training_state_slot_0.pt"),
            "sha256": OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256,
        }, label="optimizer origin state", verify_file=True)
    return dict(origin)


def _require_dynamic_entry_only_preprocessing(artifacts, *, design, normalization, labels):
    """Join the measured Entry-only transform, never an Exit/composite surrogate."""
    import numpy as np
    import pandas as pd
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE as phase
    from gx1.contracts.entry_model_native_aux_targets_v3 import M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
    from gx1.contracts.entry_model_native_input_normalization_v1 import (
        DYNAMIC_M5_FIT_POPULATION, require_input_normalization_contract,
    )
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_FIELDS, MODEL_NATIVE_CTX_CAT_FIELDS, MODEL_NATIVE_SEQ_LEN,
        MODEL_NATIVE_SIGNAL_DIM, require_model_native_signal_contract,
    )
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    from gx1.contracts.entry_full_input_liveness_v1 import validate_full_input_liveness_artifact
    from gx1.contracts.entry_sequence_source_reconstruction_v1 import require_sequence_source_reconstruction_audit
    from gx1.contracts.entry_observed_market_horizons_v1 import require_m5_single_epoch_population
    from gx1.features.htf_features import MULTI_TF_TIMEFRAMES, MULTI_TF_PER_BAR_FEATURES_V4
    from gx1.features import gc_native_context_v1 as gc
    from gx1.models.entry_v10.entry_v10_input_normalization import (
        FIT_POPULATION_PROOF_SCHEMA_VERSION, _canonical_sha256 as normalization_sha256,
    )

    def require(condition, reason):
        if not condition:
            raise RuntimeError("NATIVE_PHYSICAL_PREPROCESSING_DYNAMIC_" + reason)

    def read(binding):
        bound = require_binding(binding, label="dynamic physical preprocessing", verify_file=True)
        return read_bound_json(Path(bound["path"]), bound["sha256"])

    def utc(value):
        stamp = pd.Timestamp(value)
        require(not pd.isna(stamp) and stamp.tzinfo is not None
                and stamp.utcoffset() == pd.Timedelta(0), "CLOCK_INVALID")
        return stamp

    require(all(read(artifacts[key]) == value for key, value in
                (("design", design), ("normalization_result", normalization), ("labels_result", labels))),
            "ARGUMENT_BINDING_MISMATCH")
    require(design.get("schema_version") == "gx1_frozen_chronological_learning_design_v1"
            and design["scope"].get("emission_scope") == M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
            and normalization.get("schema_version") == "gx1_entry_only_dynamic_m5_base_normalization_v1"
            and normalization.get("decision") == "PASS" and normalization.get("entry_training_phase") == phase
            and normalization.get("val_fit_rows") == normalization.get("test_fit_rows") == 0
            and normalization.get("test_accessed") is False
            and normalization.get("native_training_admitted") is False, "ENTRY_ONLY_NORMALIZATION_REQUIRED")
    calendar = design["calendar"]; sources = calendar["source_bindings"]
    start, cutoff, end = map(utc, (calendar["train_entry_start_inclusive"], calendar["train_control_cutoff"],
                                 calendar["development_control_entry_end_exclusive"]))
    require(start < cutoff < end and all(sources["train"][key]["path"] != sources["val"][key]["path"]
            for key in ("parquet", "manifest")), "WINDOW_OR_SOURCE_INVALID")
    inputs = design.get("inputs", {})
    audits = inputs.get("sequence_source_audits", {})
    require(isinstance(audits, Mapping) and set(audits) == {"train", "val"}
            and audits["train"] == normalization.get("sequence_source_audit")
            and inputs.get("full_input_liveness") == normalization.get("full_input_liveness"),
            "INPUT_PROOF_BINDING_MISMATCH")
    manifests, reconstructions = {}, {}
    for split, low, high in (("train", start, cutoff), ("val", cutoff, end)):
        source = sources[split]; proof = labels["splits"][split]
        require(type(source["physical_rows"]) is int and source["physical_rows"] > 0
                and utc(source["declared_window"]["start"]) == low
                and low < utc(source["declared_window"]["end"]) <= high
                and proof["source_parquet"] == source["parquet"] and proof["source_manifest"] == source["manifest"]
                and proof["rows"] == proof["first_prefix_observed_rows"] == source["physical_rows"]
                and proof["physical_clock_sha256"] == source["clock_sha256"]
                and utc(proof["dynamic_m5_supervision"]["index_binding"]["split_end_exclusive"]) == high,
                "SOURCE_OR_POPULATION_MISMATCH")
        manifest = read(source["manifest"]); manifests[split] = manifest
        require_model_native_signal_contract(manifest["extra"].get("model_native_signal_contract"),
                                             context="DYNAMIC_PHYSICAL_PREPROCESSING")
        require(manifest.get("output_data_path") == source["parquet"]["path"]
                and manifest["extra"].get("rows") == source["physical_rows"]
                and manifest["extra"].get("emission_scope") == M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
                and manifest["extra"].get("gc_native_context") == gc.contract(), "MANIFEST_MISMATCH")
        reconstructions[split] = require_sequence_source_reconstruction_audit(read(audits[split]),
            expected_parquet_path=Path(source["parquet"]["path"]), expected_manifest_path=Path(source["manifest"]["path"]),
            expected_parquet_sha256=source["parquet"]["sha256"], expected_manifest_sha256=source["manifest"]["sha256"],
            expected_feature_surface=manifest, expected_rows=source["physical_rows"],
            expected_seq_len=MODEL_NATIVE_SEQ_LEN, expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM)
    require(labels["splits"]["train"]["dynamic_m5_supervision"]["source_quote_binding"]
            == labels["splits"]["val"]["dynamic_m5_supervision"]["source_quote_binding"], "SHARED_QUOTE_MISMATCH")
    signal_names = manifests["train"]["extra"]["model_native_signal_contract"]["fields"]
    field_order = {"signal": signal_names, "ctx_cont": list(MODEL_NATIVE_CTX_CONT_FIELDS), "ctx_cat": list(MODEL_NATIVE_CTX_CAT_FIELDS)}
    live_binding = require_binding(inputs["full_input_liveness"], label="current full-input liveness", verify_file=True)
    live_validation = validate_full_input_liveness_artifact(live_binding["path"], expected_sha256=live_binding["sha256"],
        expected_dataset_dir=Path(sources["train"]["parquet"]["path"]).parent, expected_field_order=field_order,
        expected_manifest_bindings={split: source["manifest"] for split, source in sources.items()})
    require(live_validation.get("ok") is True, "INPUT_LIVENESS_INVALID")
    liveness = read(live_binding)
    fields = {"signal": signal_names, "ctx_cont": list(MODEL_NATIVE_CTX_CONT_FIELDS),
              **{"mtf_" + tf.lower(): list(MULTI_TF_PER_BAR_FEATURES_V4) for tf in MULTI_TF_TIMEFRAMES}}
    contract = require_input_normalization_contract(normalization["contract"], expected_field_names=fields,
        expected_ctx_cat_names=MODEL_NATIVE_CTX_CAT_FIELDS, expected_entry_training_phase=phase)
    require(contract["contract_sha256"] == normalization["contract_sha256"], "NORMALIZATION_CONTRACT_MISMATCH")
    lineage = contract["lineage"]; train_source = sources["train"]; n = train_source["physical_rows"]
    require(lineage["train_parquet_path"] == train_source["parquet"]["path"]
            and lineage["train_parquet_sha256"] == train_source["parquet"]["sha256"]
            and lineage["train_manifest_path"] == train_source["manifest"]["path"]
            and lineage["train_manifest_sha256"] == train_source["manifest"]["sha256"]
            and lineage["train_row_count"] == lineage["entry_train_decision_row_count"] == lineage["context_fit_row_count"] == n
            and lineage["exit_train_decision_row_count"] == lineage["val_fit_row_count"] == lineage["test_fit_row_count"] == 0
            and lineage["per_tf_seq_lens"] == dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)
            and contract["fit_population"] == DYNAMIC_M5_FIT_POPULATION
            and start <= utc(contract["fit_start_utc"]) <= utc(contract["fit_end_utc"]) < cutoff,
            "WHOLE_TRAIN_NORMALIZATION_SCOPE_MISMATCH")
    proof = normalization["fit_population_proof"]
    require(proof.get("schema_version") == FIT_POPULATION_PROOF_SCHEMA_VERSION
            and proof["proof_sha256"] == normalization_sha256({key: value for key, value in proof.items() if key != "proof_sha256"})
            and proof["entry_training_phase"] == phase and proof["fit_population"] == DYNAMIC_M5_FIT_POPULATION
            and proof["fit_scope"] == "train_only"
            and proof["train_decision_row_count"] == proof["entry_train_decision_row_count"] == proof["context_fit_row_count"] == n
            and proof["exit_train_decision_row_count"] == proof["val_fit_row_count"] == proof["test_fit_row_count"] == 0
            and set(proof["local_populations"]) == set(proof["context_populations"]) == {"entry"}
            and proof["local_fit_row_count"] == lineage["local_fit_row_count"], "FIT_POPULATION_PROOF_INVALID")
    require(proof["local_populations"]["entry"]["selected_unique_row_count"] == lineage["local_fit_row_count"]
            == contract["surfaces"]["signal"]["fit_row_count"]
            and contract["surfaces"]["ctx_cont"]["fit_row_count"] == n, "LOCAL_CONTEXT_POPULATION_MISMATCH")
    for tf, seq_len in PRODUCTION_MTF_PER_TF_WINDOW_BARS:
        item = proof["mtf_populations"][tf]; routes = item["routes"]
        require(item["seq_len"] == seq_len and routes["entry"]["enabled"] is True
                and routes["entry"]["decision_row_count"] == n
                and routes["exit"]["enabled"] is False and routes["exit"]["decision_row_count"] == 0
                and item["selected_unique_row_count"] == lineage["per_tf_fit_windows"][tf]["selected_unique_row_count"]
                == contract["surfaces"]["mtf_" + tf.lower()]["fit_row_count"], "MTF_ENTRY_ONLY_POPULATION_MISMATCH")
    source_proof = proof["entry_sequence_source_provenance"]
    require(source_proof["audit_path"] == audits["train"]["path"] and source_proof["audit_sha256"] == audits["train"]["sha256"]
            and source_proof["audit"] == reconstructions["train"], "FIT_RECONSTRUCTION_MISMATCH")
    clock = proof["gc_registry_whole_train_clock"]
    live_clock = liveness["materializer_provenance"]["gc_registry_whole_train_clock"]
    require(clock["exact_match"] is True and clock["rows"] == n
            and clock["population"] == "whole_physical_TRAIN_before_target_availability_selection"
            and clock["physical_train_parquet_path"] == train_source["parquet"]["path"]
            and clock["physical_train_bar_clock_sha256"] == train_source["clock_sha256"]
            and clock["registry_path"] == gc.REGISTRY_PATH and clock["registry_sha256"] == gc.REGISTRY_SHA256
            and clock["rows"] == gc.REGISTRY["train_decision_rows"]
            and clock["train_decision_clock_sha256"] == gc.REGISTRY["train_decision_clock_sha256"]
            and live_clock["exact_match"] is True and live_clock["rows"] == n
            and live_clock["train_decision_clock_sha256"] == clock["train_decision_clock_sha256"]
            and live_clock["physical_train_parquet_sha256"] == train_source["parquet"]["sha256"]
            and live_clock["registry"] == {"path": gc.REGISTRY_PATH, "sha256": gc.REGISTRY_SHA256},
            "CURRENT_REGISTRY_CLOCK_MISMATCH")
    single = require_m5_single_epoch_population(design)
    train_binding = require_binding(calendar["bindings"]["TRAIN_CALENDAR_PARENT_ROWS"], label="physical TRAIN rows", verify_file=True)
    control_binding = require_binding(calendar["bindings"]["CONTROL256_PARENT_ROWS"], label="physical control rows", verify_file=True)
    train = np.load(train_binding["path"], allow_pickle=False); control = np.load(control_binding["path"], allow_pickle=False)
    require(single is not None and single["train_rows"] == n and single["train_parquet"] == train_source["parquet"]
            and single["train_manifest"] == train_source["manifest"]
            and train.dtype == control.dtype == np.dtype("int64") and train.shape == (n,)
            and np.array_equal(train, np.arange(n, dtype=np.int64))
            and np.array_equal(train, single["eligible_parent_rows"])
            and control.shape == (design["selection"]["control_entries"],)
            and len(control) == design["budget"]["later_control_entries"] == 256
            and np.array_equal(control, np.unique(control)) and np.all(control >= 0)
            and np.all(control < sources["val"]["physical_rows"]), "ROW_BINDING_MISMATCH")
    return {"physical_sources": dict(sources), "train_parent_rows": train_binding,
        "control_parent_rows": control_binding, "train_rows": n,
        "entry_training_phase": phase, "entry_only_base_normalization": dict(artifacts["normalization_result"]),
        "base_contract_sha256": contract["contract_sha256"], "normalization_fit_population": DYNAMIC_M5_FIT_POPULATION,
        "full_input_liveness": live_binding, "sequence_source_audits": dict(audits),
        "native_coordinates_bound": False, "input_admission_complete": False, "native_training_admitted": False}


def require_physical_chronological_preprocessing(
    artifacts: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
    normalization: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Join completed full-TRAIN preprocessing without reading market rows.

    This validates data identity, not native coordinates or launch authority.
    The original artifacts stay authoritative; no prefix-fit wrapper is made.
    """
    import numpy as np
    import pandas as pd
    from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
        require_composite_normalization_binding, require_split_sequence_binding,
    )
    from gx1.contracts.unified_exit_pilot_normalization_v1 import (
        require_lifetime_summary_normalization,
        canonical_sha256 as normalization_sha256,
        SUMMARY_SAMPLE_AUTHORITY_SCHEMA_VERSION,
    )

    def require(condition, reason):
        if not condition:
            raise RuntimeError("NATIVE_PHYSICAL_PREPROCESSING_" + reason)

    def read(binding):
        checked = require_binding(binding, label="physical preprocessing", verify_file=True)
        return read_bound_json(Path(checked["path"]), checked["sha256"])

    def utc(value):
        stamp = pd.Timestamp(value)
        require(not pd.isna(stamp) and stamp.tzinfo is not None
                and stamp.utcoffset() == pd.Timedelta(0), "CLOCK_INVALID")
        return stamp

    calendar = design["calendar"]
    sources = calendar.get("source_bindings")
    require(
        calendar.get("physical_source_splits") == {"train": "train", "control": "val"}
        and calendar.get("physical_coordinate_namespaces_are_separate") is True
        and "index" not in calendar and isinstance(sources, Mapping)
        and set(sources) == {"train", "val"},
        "PHYSICAL_SOURCES_INVALID",
    )
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        M5_PHYSICAL_TARGET_BINDING_SCHEMA, require_m5_primary_physical_target_receipt,
        M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA, M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
        require_m5_dynamic_physical_target_receipt,
    )
    dynamic_emission = labels.get("schema_version") == M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA
    m5_emission = dynamic_emission or labels.get("schema_version") == M5_PHYSICAL_TARGET_BINDING_SCHEMA
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    require((design.get("scope", {}).get("entry_only_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
            == dynamic_emission, "DYNAMIC_M5_PHASE_RECEIPT_MISMATCH")
    if m5_emission:
        if dynamic_emission:
            require_m5_dynamic_physical_target_receipt(labels, design_binding=artifacts["design"], context="NATIVE_PHYSICAL_PREPROCESSING")
            return _require_dynamic_entry_only_preprocessing(
                artifacts, design=design, normalization=normalization, labels=labels)
        else:
            require_m5_primary_physical_target_receipt(labels, design_binding=artifacts["design"], context="NATIVE_PHYSICAL_PREPROCESSING")
        from gx1.contracts.entry_model_native_training_objective_v1 import (
            ENTRY_ONLY_M5_PATH_TRAINING_PHASE, ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        )
        expected_phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE if dynamic_emission else ENTRY_ONLY_M5_PATH_TRAINING_PHASE
        require(design.get("scope", {}).get("entry_only_training_phase") == expected_phase
                and (not dynamic_emission or design["scope"].get("emission_scope") == M5_DYNAMIC_OBSERVED_EMISSION_SCOPE)
                and normalization.get("decision") == "PASS_COMPOSITE_AND_FIRST_STATE_BINDINGS_NOT_NATIVE_ADMISSION"
                and normalization.get("test_accessed") is False
                and normalization.get("forbidden_access_attempts") == []
                and normalization.get("normalization_fit") is False
                and all(normalization.get(key) == 0 for key in ("model_forwards", "optimizer_steps")),
                "M5_COMPLETED_RESULTS_REQUIRED")
        require(all(sources["train"][kind]["path"] != sources["val"][kind]["path"]
                    for kind in ("parquet", "manifest")), "SOURCE_MISMATCH")
    else:
        require(
            labels.get("schema_version") == "gx1_native_v38_auxiliary_reuse_precheck_v1"
            and labels.get("decision") == "PASS_FROZEN_TRAIN_POLICIES_AND_COMPLETE_PRETEST_TARGET_CLOCK_SUPPORT"
            and labels.get("identical_frozen_policies_across_physical_splits") is True
            and labels.get("producer_code_unchanged_for_five_target_owners") is True
            and labels.get("test_accessed") is False
            and labels.get("forbidden_access_attempts") == []
            and normalization.get("decision") == "PASS_COMPOSITE_AND_FIRST_STATE_BINDINGS_NOT_NATIVE_ADMISSION"
            and normalization.get("test_accessed") is False
            and normalization.get("forbidden_access_attempts") == []
            and normalization.get("normalization_fit") is False
            and all(event.get(key) == 0 for event in (labels, normalization)
                    for key in ("model_forwards", "optimizer_steps")),
            "COMPLETED_RESULTS_REQUIRED",
        )
        plan = read(labels["plan"])
        inputs = plan["input_bindings"]
        require(plan.get("schema_version") == "gx1_native_v38_auxiliary_reuse_precheck_plan_v1"
                and inputs["design"] == artifacts["design"], "DESIGN_MISMATCH")
        # Compare frozen paths before following any source pointer. Market parquet
        # files and source manifests are deliberately not resolved or opened here.
        require(all(sources[split][kind] == inputs[split + "_" + kind]
                    for split in ("train", "val") for kind in ("parquet", "manifest"))
                and all(sources["train"][kind]["path"] != sources["val"][kind]["path"]
                        for kind in ("parquet", "manifest")), "SOURCE_MISMATCH")
    start = utc(calendar["train_entry_start_inclusive"])
    cutoff = utc(calendar["train_control_cutoff"])
    end = utc(calendar["development_control_entry_end_exclusive"])
    require(start < cutoff < end, "WINDOW_INVALID")
    sequences = {}
    m5_target_proofs = {}
    for split, low, high in (("train", start, cutoff), ("val", cutoff, end)):
        source = sources[split]
        geometry = normalization["splits"][split]
        if not m5_emission:
            proof = labels["splits"][split]
            require(proof["rows"] == proof["exact_m1_fill_and_contiguous_outcome_rows"]
                    == source["physical_rows"]
                    and proof["entry_clock_sha256"] == source["clock_sha256"]
                    and utc(proof["supervision_boundary_exclusive_utc"]) == high,
                    "POPULATION_OR_CLOCK_MISMATCH")
        require(type(source["physical_rows"]) is int and source["physical_rows"] > 0
                and utc(source["declared_window"]["start"]) == low
                and low < utc(source["declared_window"]["end"]) < high
                and geometry["entry_rows"] == source["physical_rows"]
                and geometry["first_state_geometry_exact"] is True
                and geometry["successor_geometry_exact"] is True, "POPULATION_OR_CLOCK_MISMATCH")
        if dynamic_emission:
            # The source-owned receipt scans the actual time/pointer/mask bytes.
            # No inactive legacy37 or fixed95 target union is read here.
            proof = labels["splits"][split]
            require(proof["source_parquet"] == source["parquet"]
                    and proof["source_manifest"] == source["manifest"]
                    and proof["rows"] == proof["first_prefix_observed_rows"] == source["physical_rows"]
                    and proof["physical_clock_sha256"] == source["clock_sha256"]
                    and utc(proof["dynamic_m5_supervision"]["index_binding"]["split_end_exclusive"]) == high,
                    "DYNAMIC_M5_SOURCE_OR_POPULATION_MISMATCH")
            m5_target_proofs[split] = proof
        elif m5_emission:
            import pyarrow.parquet as pq
            from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
                _physical_auxiliary_target_binding, _MODEL_NATIVE_ACTIVE_TARGET_COLS,
            )
            target_frame = pq.read_table(source["parquet"]["path"], columns=["time", *_MODEL_NATIVE_ACTIVE_TARGET_COLS]).to_pandas()
            _, m5_target_proofs[split] = _physical_auxiliary_target_binding(target_frame, parquet_path=Path(source["parquet"]["path"]),
                result_binding=artifacts["labels_result"], expected_design_sha256=artifacts["design"]["sha256"],
                role="TRAIN" if split == "train" else "CONTROL256")
        sequence = require_split_sequence_binding(
            read(normalization["artifacts"]["SPLIT_SEQUENCE_BINDING_" + split.upper() + ".json"]),
            expected_split=split, expected_entry_rows=source["physical_rows"],
        )
        require(sequence["entry_clock_sha256"] == source["clock_sha256"]
                and sequence["bindings"]["child_parquet"] == source["parquet"]["sha256"]
                and sequence["binding_sha256"] == geometry["sequence_binding_sha256"],
                "SEQUENCE_SOURCE_MISMATCH")
        sequences[split] = sequence

    composite_binding = normalization["artifacts"]["COMPOSITE_NORMALIZATION.json"]
    composite = require_composite_normalization_binding(read(composite_binding))
    require(composite["composite_normalization_sha256"]
            == normalization["composite_normalization_sha256"], "COMPOSITE_MISMATCH")
    base = composite["base_feature_normalization"]
    require(read({"path": base["path"], "sha256": base["file_sha256"]}) == base["artifact"],
            "BASE_FILE_MISMATCH")
    contract = base["artifact"]["contract"]
    lineage = contract["lineage"]
    require(
        lineage["train_parquet_path"] == sources["train"]["parquet"]["path"]
        and lineage["train_parquet_sha256"] == sources["train"]["parquet"]["sha256"]
        and lineage["entry_train_decision_row_count"] == sources["train"]["physical_rows"]
        and lineage["val_fit_row_count"] == lineage["test_fit_row_count"] == 0
        and start <= utc(contract["fit_start_utc"]) <= utc(contract["fit_end_utc"]) < cutoff,
        "BASE_TRAIN_SCOPE_MISMATCH",
    )
    summary_binding = composite["summary_fit_manifest"]
    summary = read({"path": summary_binding["path"], "sha256": summary_binding["file_sha256"]})
    unsigned = {k: v for k, v in summary.items() if k != "manifest_sha256"}
    require(summary.get("schema_version") == "gx1_unified_exit_pilot_summary_fit_inputs_v1"
            and summary.get("decision") == "PASS" and summary.get("split") == "train"
            and summary.get("manifest_sha256") == summary_binding["manifest_sha256"]
            == normalization_sha256(unsigned)
            and summary.get("child_parquet_sha256") == sources["train"]["parquet"]["sha256"]
            and summary.get("entry_pair_population") == sources["train"]["physical_rows"]
            and summary.get("val_fit_rows") == summary.get("test_fit_rows") == 0
            and summary.get("test_accessed") is False, "SUMMARY_TRAIN_SCOPE_MISMATCH")
    authority = summary["summary_sample_authority"]
    require(authority.get("schema_version") == SUMMARY_SAMPLE_AUTHORITY_SCHEMA_VERSION
            and authority.get("authority_sha256") == normalization_sha256(
                {k: v for k, v in authority.items() if k != "authority_sha256"})
            and authority["selection_uses_outcome_values"] is False
            and authority["epoch_sampler_independent"] is True
            and authority["successor_counts_sha256"] == summary["successor_counts_sha256"]
            == sequences["train"]["successor_counts_sha256"]
            and summary["successor_transition_total"] == sequences["train"]["successor_transition_total"],
            "SUMMARY_POPULATION_AUTHORITY_MISMATCH")
    lifetime = require_lifetime_summary_normalization(
        summary["lifetime_summary_normalization"],
        expected_sample_authority_sha256=authority["authority_sha256"],
    )
    require(lifetime == composite["lifetime_summary_normalization"]
            and lifetime["train_fit_rows"] == authority["fit_row_count"]
            and authority["entry_pair_population"]
            == sources["train"]["physical_rows"], "SUMMARY_TRANSFORM_MISMATCH")
    if dynamic_emission:
        require(m5_target_proofs["train"]["dynamic_m5_supervision"]["source_quote_binding"]
                == m5_target_proofs["val"]["dynamic_m5_supervision"]["source_quote_binding"],
                "DYNAMIC_M5_SHARED_QUOTE_SOURCE_MISMATCH")
    elif m5_emission:
        require(m5_target_proofs["train"]["policies"] == m5_target_proofs["val"]["policies"],
                "FROZEN_POLICY_MISMATCH")
        require(m5_target_proofs["val"]["source_split"] == "val"
                and m5_target_proofs["val"]["rows"] == 256, "CONTROL_SOURCE_MISMATCH")
    else:
        require(all(labels["splits"]["train"][key] == labels["splits"]["val"][key]
                    for key in ("direction_policy_sha256", "position_size_policy_sha256")),
                "FROZEN_POLICY_MISMATCH")
        control_proof = labels["splits"]["val"]["control256"]
        require(control_proof["physical_coordinate_source"] == "val"
                and control_proof["rows"] == 256
                and control_proof["all_support_checks_passed"] is True, "CONTROL_SOURCE_MISMATCH")
    train_binding = require_binding(calendar["bindings"]["TRAIN_CALENDAR_PARENT_ROWS"],
                                    label="physical TRAIN rows", verify_file=True)
    control_binding = require_binding(calendar["bindings"]["CONTROL256_PARENT_ROWS"],
                                      label="physical control rows", verify_file=True)
    from gx1.contracts.entry_observed_market_horizons_v1 import require_m5_single_epoch_population
    single_epoch = require_m5_single_epoch_population(design)
    train = np.load(train_binding["path"], allow_pickle=False)
    control = np.load(control_binding["path"], allow_pickle=False)
    require(
        train.dtype == control.dtype == np.dtype("int64")
        and train.shape == (sources["train"]["physical_rows"],)
        and np.array_equal(train, np.arange(len(train), dtype=np.int64))
        and (len(train) >= design["budget"]["maximum_trained_entry_rows"] if single_epoch is not None
             else len(train) > design["budget"]["maximum_trained_entry_rows"])
        and control.shape == (design["selection"]["control_entries"],)
        and len(control) == design["budget"]["later_control_entries"] == 256
        and np.array_equal(control, np.unique(control)) and np.all(control >= 0)
        and np.all(control < sources["val"]["physical_rows"])
        and (m5_emission or control_binding == inputs["control256"]),
        "ROW_BINDING_MISMATCH",
    )
    return {
        "physical_sources": dict(sources), "train_parent_rows": train_binding,
        "control_parent_rows": control_binding, "train_rows": len(train),
        "composite_normalization": dict(composite_binding),
        "base_contract_sha256": base["contract_sha256"],
        "summary_normalization_sha256": lifetime["normalization_sha256"],
        "composite_normalization_sha256": composite["composite_normalization_sha256"],
        "native_coordinates_bound": False,
    }


DESIGN_METADATA_COMPLETION_SCHEMA = "gx1_chronological_design_metadata_completion_v1"


def complete_missing_chronological_design_metadata(original_design_binding):
    """Complete only absent scope/status from the original's bound declaration.

    This creates a new design body, not a rewritten benchmark or run authority.
    Historical inputs, clocks, functions and source-owner receipts remain exact.
    """
    original_binding = require_binding(
        original_design_binding, label="original chronological design", verify_file=True)
    original = read_bound_json(Path(original_binding["path"]), original_binding["sha256"])
    if (original.get("schema_version") != "gx1_frozen_chronological_learning_design_v1"
            or any(key in original for key in ("scope", "status", "metadata_completion"))):
        raise RuntimeError("NATIVE_DESIGN_METADATA_COMPLETION_ORIGINAL_INVALID")
    procedure_binding = require_binding(
        original.get("prospective_procedure_source"),
        label="already bound prospective procedure", verify_file=True)
    procedure = read_bound_json(Path(procedure_binding["path"]), procedure_binding["sha256"])
    if (procedure.get("schema_version") != original["schema_version"]
            or not isinstance(procedure.get("scope"), Mapping)
            or not isinstance(procedure.get("status"), str)):
        raise RuntimeError("NATIVE_DESIGN_METADATA_COMPLETION_PROCEDURE_INVALID")
    return {
        **original, "scope": dict(procedure["scope"]), "status": procedure["status"],
        "metadata_completion": {
            "schema_version": DESIGN_METADATA_COMPLETION_SCHEMA,
            "original_design": original_binding,
            "prospective_procedure_source": procedure_binding,
        },
    }


def require_benchmarked_design_identity(benchmark_design_binding, *, design_binding, design):
    """Admit exact identity or the single explicit metadata-only completion.

    Rebuild the expected body from verified original/declaration files. Canonical
    JSON preserves types (True is not 1); no other field or binding may change.
    """
    benchmark_binding = require_binding(
        benchmark_design_binding, label="benchmarked chronological design", verify_file=True)
    current_binding = require_binding(
        design_binding, label="current chronological design", verify_file=True)
    current = read_bound_json(Path(current_binding["path"]), current_binding["sha256"])
    try:
        if canonical_sha256(current) != canonical_sha256(design):
            raise RuntimeError("NATIVE_BENCHMARK_DESIGN_BODY_MISMATCH")
        if benchmark_binding == current_binding:
            return
        expected = complete_missing_chronological_design_metadata(benchmark_binding)
        if canonical_sha256(design) != canonical_sha256(expected):
            raise RuntimeError("NATIVE_BENCHMARK_DESIGN_METADATA_COMPLETION_MISMATCH")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("NATIVE_BENCHMARK_DESIGN_JSON_INVALID") from exc


PHYSICAL_COORDINATES_SCHEMA_V1 = "gx1_physical_native_training_coordinates_v1"
PHYSICAL_COORDINATES_SCHEMA_V2 = "gx1_physical_native_training_coordinates_v2"
PHYSICAL_COORDINATES_SCHEMA_M5_PATH = "gx1_physical_native_training_coordinates_m5_path_v1"
M5_PATH_ELIGIBILITY_RULE = "canonical_m5_path_target_valid_and_available_before_bound_train_cutoff_independent_of_long_horizons"
PHYSICAL_COORDINATES_SCHEMA_DYNAMIC_M5_PATH = "gx1_physical_native_training_coordinates_dynamic_m5_path_v2"
DYNAMIC_M5_PATH_ELIGIBILITY_RULE = "any_genuine_first_m1_prefix_component_before_bound_train_cutoff"
HORIZON_ELIGIBILITY_RULE = "complete_outcomes_at_all_observation_horizons_at_or_before_measured_cutoff"


def _physical_native_sampler(selected_binding=None, *, artifacts, design, physical, dynamic_population_witness=None):
    """Bind the measured legacy sampler or the explicit dynamic population witness."""
    from gx1.contracts.unified_exit_selected_sampler_v1 import (
        DIRECT_SELECTION_MODE, require_selected_sampler_artifact,
    )
    def check(condition, reason):
        if not condition:
            raise RuntimeError("NATIVE_PHYSICAL_COORDINATES_" + reason)
    def read(value):
        value = require_binding(value, label="physical native sampler", verify_file=True)
        return read_bound_json(Path(value["path"]), value["sha256"])
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    dynamic = design.get("scope", {}).get("entry_only_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if dynamic:
        import numpy as np
        from gx1.contracts.entry_observed_market_horizons_v1 import (
            M5_DYNAMIC_SINGLE_EPOCH_POPULATION_SCHEMA, require_m5_single_epoch_population,
        )
        from gx1.scripts.materialize_unified_exit_chronological_coordinates_v1 import _epoch0_order
        check(selected_binding is None and dynamic_population_witness is not None,
              "DYNAMIC_POPULATION_WITNESS_REQUIRED")
        witness_binding = require_binding(dynamic_population_witness,
            label="dynamic physical population witness", verify_file=True)
        single = require_m5_single_epoch_population(design)
        check(single is not None and witness_binding == single["binding"]
              and single["witness"]["schema_version"] == M5_DYNAMIC_SINGLE_EPOCH_POPULATION_SCHEMA
              and single["train_parquet"] == physical["physical_sources"]["train"]["parquet"]
              and single["train_manifest"] == physical["physical_sources"]["train"]["manifest"]
              and single["train_rows"] == physical["train_rows"]
              and np.array_equal(single["eligible_parent_rows"], np.arange(physical["train_rows"], dtype=np.int64)),
              "DYNAMIC_POPULATION_WITNESS_MISMATCH")
        sampler = single["witness"]["native_sampler_contract"]
        check(np.array_equal(_epoch0_order(sampler), single["epoch0_parent_order"]),
              "DYNAMIC_POPULATION_ORDER_MISMATCH")
        return {"population_witness": witness_binding, "native_sampler_contract": sampler}
    check(dynamic_population_witness is None, "DYNAMIC_POPULATION_WITNESS_FORBIDDEN")
    selected = require_selected_sampler_artifact(read(selected_binding))
    check(selected.get("selection_mode") == DIRECT_SELECTION_MODE
          and selected["selected_sampler_contract"]["entry_pair_population"] == physical["train_rows"],
          "SAMPLER_MISMATCH")
    try:
        require_benchmarked_design_identity(
            selected.get("benchmark_design"), design_binding=artifacts["design"], design=design)
    except RuntimeError as exc:
        raise RuntimeError("NATIVE_PHYSICAL_COORDINATES_SAMPLER_MISMATCH") from exc
    manifest_binding = selected["train_index_manifest"]
    manifest = read({"path": manifest_binding["path"], "sha256": manifest_binding["file_sha256"]})
    check(all(manifest["source_bindings"][f"parent_entry_{kind}"]
              == physical["physical_sources"]["train"][kind] for kind in ("parquet", "manifest")),
          "INDEX_PARENT_SOURCE_MISMATCH")
    from gx1.scripts.benchmark_unified_exit_random_access_train_v1 import _reference_workload_from_design
    check(selected["reference_workload"] == _reference_workload_from_design(design), "WORKLOAD_MISMATCH")
    return selected


def require_physical_native_training_coordinates(
    artifacts: Mapping[str, Any], *, design: Mapping[str, Any], physical: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate frozen parent coordinates; native construction replays the order."""
    import numpy as np
    from gx1.contracts.entry_model_native_training_run_lineage_v1 import deterministic_uniform_subsample_indices

    def check(condition, reason):
        if not condition:
            raise RuntimeError("NATIVE_PHYSICAL_COORDINATES_" + reason)
    def bound(value):
        return require_binding(value, label="physical native coordinates", verify_file=True)
    def read(value):
        value = bound(value)
        return read_bound_json(Path(value["path"]), value["sha256"])

    check("native_coordinates" in artifacts, "REQUIRED")
    binding = bound(artifacts["native_coordinates"])
    value = read(binding)
    # v1: every physical TRAIN row is eligible. v2 (operator decision 12.10.2026): the eligible
    # rows are the frozen subset with complete observed outcomes at every trading-day horizon at
    # or before the measured cutoff; the order is the native sampler's full order filtered to it.
    schema = value.get("schema_version")
    check(schema in (PHYSICAL_COORDINATES_SCHEMA_V1, PHYSICAL_COORDINATES_SCHEMA_V2, PHYSICAL_COORDINATES_SCHEMA_M5_PATH, PHYSICAL_COORDINATES_SCHEMA_DYNAMIC_M5_PATH), "IDENTITY_INVALID")
    horizon_subset = schema == PHYSICAL_COORDINATES_SCHEMA_V2
    m5_path_subset = schema == PHYSICAL_COORDINATES_SCHEMA_M5_PATH
    dynamic_m5_subset = schema == PHYSICAL_COORDINATES_SCHEMA_DYNAMIC_M5_PATH
    check(
        value.get("decision") == "TRAIN_COORDINATES_FROZEN_NO_MODEL"
        and value.get("coordinates_sha256") == native_sha256(
            {key:item for key,item in value.items() if key != "coordinates_sha256"})
        and value.get("design") == artifacts["design"]
        and value.get("train_source") == physical["physical_sources"]["train"]["parquet"]
        and value.get("control_source") == physical["physical_sources"]["val"]["parquet"]
        and value.get("control_parent_rows") == physical["control_parent_rows"]
        and value.get("test_data_used") is False
        and value.get("selection_uses_outcome_values") is False
        and all(type(value.get(key)) is int and value[key] == 0
                for key in ("model_forwards", "optimizer_steps", "fits")),
        "IDENTITY_INVALID",
    )
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    check(dynamic_m5_subset == (design.get("scope", {}).get("entry_only_training_phase")
          == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE), "DYNAMIC_M5_PHASE_MISMATCH")
    if dynamic_m5_subset:
        check("selected_sampler" not in value, "DYNAMIC_EXIT_SAMPLER_FORBIDDEN")
        witness_binding = bound(value.get("population_witness"))
        selected = _physical_native_sampler(artifacts=artifacts, design=design, physical=physical,
            dynamic_population_witness=witness_binding)
        sampler_contract = selected["native_sampler_contract"]
    else:
        check("population_witness" not in value, "DYNAMIC_POPULATION_WITNESS_FORBIDDEN")
        selected_binding = bound(value.get("selected_sampler"))
        selected = _physical_native_sampler(selected_binding, artifacts=artifacts, design=design, physical=physical)
        sampler_contract = selected["selected_sampler_contract"]
    bindings = value.get("bindings")
    keys = ("TRAIN_ELIGIBLE_PARENT_ROWS", "TRAIN_NATIVE_EPOCH0_ORDER",
            "TRAIN_NATIVE4096_PARENT_ROWS", "TRAIN256_PROBE_PARENT_ROWS")
    check(isinstance(bindings, Mapping) and set(bindings) == set(keys)
          and (horizon_subset or m5_path_subset or dynamic_m5_subset or bindings["TRAIN_ELIGIBLE_PARENT_ROWS"] == physical["train_parent_rows"]), "ROWS_INVALID")
    arrays = {key: np.load(bound(bindings[key])["path"], allow_pickle=False) for key in keys}
    eligible, order, planned, probe = (arrays[key] for key in keys)
    if horizon_subset:
        eligibility = value.get("eligibility")
        check(isinstance(eligibility, Mapping)
              and set(eligibility) == {"rule", "horizons_design", "entry_eligibility_cutoff_utc", "eligible_rows",
                                       "censored_rows_excluded", "complete_at_all_observation_horizons"}
              and eligibility["rule"] == HORIZON_ELIGIBILITY_RULE
              and eligibility["complete_at_all_observation_horizons"] is True
              and type(eligibility["eligible_rows"]) is int and eligibility["eligible_rows"] == len(eligible)
              and type(eligibility["censored_rows_excluded"]) is int and eligibility["censored_rows_excluded"] >= 0
              and bound(eligibility["horizons_design"]) is not None
              and bindings["TRAIN_ELIGIBLE_PARENT_ROWS"]["path"] != physical["train_parent_rows"]["path"],
              "ELIGIBILITY_INVALID")
        population_ok = (eligible.ndim == 1 and len(eligible) >= 1 and np.all(np.diff(eligible) > 0)
                         and eligible[0] >= 0 and eligible[-1] < physical["train_rows"])
    elif dynamic_m5_subset:
        from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        from gx1.contracts.entry_model_native_aux_targets_v3 import M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
        from gx1.scripts.materialize_unified_exit_chronological_coordinates_v1 import dynamic_m5_eligible_rows, _epoch0_order
        eligibility = value.get("eligibility")
        check(isinstance(eligibility, Mapping)
            and set(eligibility) == {"rule", "training_phase", "emission_scope", "target_contract",
                "entry_eligibility_cutoff_utc", "eligible_rows", "censored_rows_excluded",
                "later_horizons_gate_primary_eligibility", "first_prefix_horizon_minutes"}
            and eligibility["rule"] == DYNAMIC_M5_PATH_ELIGIBILITY_RULE
            and eligibility["training_phase"] == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
            and eligibility["emission_scope"] == M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
            and eligibility["target_contract"] == entry_observed_dynamic_m5_path_contract()
            and design.get("scope", {}).get("entry_only_training_phase") == eligibility["training_phase"]
            and design["scope"].get("emission_scope") == eligibility["emission_scope"]
            and eligibility["entry_eligibility_cutoff_utc"] == design["calendar"]["train_control_cutoff"]
            and eligibility["later_horizons_gate_primary_eligibility"] is False
            and type(eligibility["first_prefix_horizon_minutes"]) is int and eligibility["first_prefix_horizon_minutes"] == 1
            and type(eligibility["eligible_rows"]) is int and eligibility["eligible_rows"] == len(eligible)
            and type(eligibility["censored_rows_excluded"]) is int and eligibility["censored_rows_excluded"] == 0
            and bindings["TRAIN_ELIGIBLE_PARENT_ROWS"]["path"] != physical["train_parent_rows"]["path"],
            "DYNAMIC_M5_ELIGIBILITY_INVALID")
        found = dynamic_m5_eligible_rows(Path(physical["physical_sources"]["train"]["parquet"]["path"]),
            cutoff_utc=eligibility["entry_eligibility_cutoff_utc"])
        population_ok = (found["physical_rows"] == physical["train_rows"]
            and found["censored_rows_excluded"] == found["rows_after_cutoff"] == 0
            and np.array_equal(eligible, found["rows"])
            and np.array_equal(eligible, np.arange(physical["train_rows"], dtype=np.int64)))
        full_order = _epoch0_order(sampler_contract)
        check(np.array_equal(order, full_order), "DYNAMIC_M5_NATIVE_ORDER_MISMATCH")
    elif m5_path_subset:
        from gx1.contracts.entry_observed_market_v1 import entry_observed_m5_path_contract
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
        from gx1.scripts.materialize_unified_exit_chronological_coordinates_v1 import m5_path_eligible_rows, _epoch0_order
        eligibility = value.get("eligibility")
        check(isinstance(eligibility, Mapping)
            and set(eligibility) == {"rule", "training_phase", "target_contract", "entry_eligibility_cutoff_utc",
                "eligible_rows", "censored_rows_excluded", "long_horizons_gate_primary_eligibility"}
            and eligibility["rule"] == M5_PATH_ELIGIBILITY_RULE
            and eligibility["training_phase"] == ENTRY_ONLY_M5_PATH_TRAINING_PHASE
            and eligibility["target_contract"] == entry_observed_m5_path_contract()
            and eligibility["entry_eligibility_cutoff_utc"] == design["calendar"]["train_control_cutoff"]
            and eligibility["long_horizons_gate_primary_eligibility"] is False
            and type(eligibility["eligible_rows"]) is int and eligibility["eligible_rows"] == len(eligible)
            and type(eligibility["censored_rows_excluded"]) is int and eligibility["censored_rows_excluded"] >= 0
            and bindings["TRAIN_ELIGIBLE_PARENT_ROWS"]["path"] != physical["train_parent_rows"]["path"],
            "M5_PATH_ELIGIBILITY_INVALID")
        found = m5_path_eligible_rows(Path(physical["physical_sources"]["train"]["parquet"]["path"]),
            cutoff_utc=eligibility["entry_eligibility_cutoff_utc"])
        population_ok = (found["physical_rows"] == physical["train_rows"]
            and found["censored_rows_excluded"] == eligibility["censored_rows_excluded"]
            and np.array_equal(eligible, found["rows"]))
        full_order = _epoch0_order(selected["selected_sampler_contract"])
        check(np.array_equal(order, full_order[np.isin(full_order, eligible)]), "M5_PATH_NATIVE_ORDER_MISMATCH")
    else:
        population_ok = np.array_equal(eligible, np.arange(physical["train_rows"], dtype=np.int64))
    check(all(array.dtype == np.dtype("int64") and array.ndim == 1 for array in arrays.values())
          and population_ok
          and np.array_equal(np.sort(order), eligible)
          and planned.shape == (design["budget"]["maximum_trained_entry_rows"],)
          and np.array_equal(planned, order[:len(planned)]), "ROWS_INVALID")
    from gx1.contracts.entry_observed_market_horizons_v1 import require_m5_single_epoch_population
    single_epoch = require_m5_single_epoch_population(design)
    if single_epoch is not None:
        from gx1.contracts.entry_observed_market_horizons_v1 import M5_DYNAMIC_SINGLE_EPOCH_POPULATION_SCHEMA
        dynamic_epoch = single_epoch["witness"]["schema_version"] == M5_DYNAMIC_SINGLE_EPOCH_POPULATION_SCHEMA
        check(value["schema_version"] == (PHYSICAL_COORDINATES_SCHEMA_DYNAMIC_M5_PATH if dynamic_epoch else PHYSICAL_COORDINATES_SCHEMA_M5_PATH)
            and (not dynamic_epoch or single_epoch["train_manifest"] == physical["physical_sources"]["train"]["manifest"])
            and single_epoch["train_parquet"] == physical["physical_sources"]["train"]["parquet"]
            and single_epoch["witness"]["native_sampler_contract"] == sampler_contract
            and np.array_equal(eligible, single_epoch["eligible_parent_rows"])
            and np.array_equal(order, single_epoch["epoch0_parent_order"])
            and np.array_equal(planned, order), "SINGLE_EPOCH_WITNESS_MISMATCH")
    positions = deterministic_uniform_subsample_indices(
        population_rows=len(planned), requested_rows=design["budget"]["train_only_probe_entries"],
        seed=design["selection"]["seed"], split_salt=0,
    )
    check(probe.shape == (design["budget"]["train_only_probe_entries"],)
          and np.array_equal(probe, planned[positions]), "PROBE_MISMATCH")
    return {
        "binding": binding, "artifact": value,
        **({"population_witness": selected["population_witness"], "native_sampler_contract": sampler_contract}
           if dynamic_m5_subset else {"selected_sampler_binding": selected_binding, "selected_sampler": selected}),
        "eligible_parent_rows": eligible,
        "epoch0_parent_order": order, "planned_parent_rows": planned, "probe_parent_rows": probe,
    }


def _planned_prefix_optimizer_steps(design: Mapping[str, Any]) -> int:
    from gx1.contracts.entry_observed_market_horizons_v1 import planned_prefix_optimizer_steps
    return planned_prefix_optimizer_steps(design)


def _dynamic_m5_native_design(recipe):
    """Select only the explicit phase in an immutable full chronological design."""
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    value = recipe.get("chronological_prefix")
    if not isinstance(value, Mapping) or "design" not in value:
        return None
    binding = require_binding(value["design"], label="dynamic native design", verify_file=True)
    design = read_bound_json(Path(binding["path"]), binding["sha256"])
    return design if design.get("scope", {}).get("entry_only_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE else None


def _require_dynamic_m5_loss_scale_identity(recipe, *, prefix, observed_scope=None):
    """Strict TRAIN loss-unit binding, independent of a later observed-plan path."""
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import (
        DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY, dynamic_m5_loss_fit_lineage,
        dynamic_m5_loss_scale_identity, load_dynamic_m5_loss_scales,
    )
    from gx1.contracts.entry_observed_market_horizons_v1 import require_observed_entry_scope
    if DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY not in recipe:
        raise RuntimeError("NATIVE_DYNAMIC_M5_LOSS_SCALES_REQUIRED")
    scope = require_observed_entry_scope(recipe) if observed_scope is None else observed_scope
    scales = load_dynamic_m5_loss_scales(recipe[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY],
        expected_lineage=dynamic_m5_loss_fit_lineage(scope, prefix["entry_m5_path_single_epoch"]))
    return dynamic_m5_loss_scale_identity(scales)


def _require_dynamic_m5_native_prefix(recipe):
    """Validate actual physical population/order; this does not authorize a run."""
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_horizons_v1 import require_observed_entry_scope
    prefix = require_chronological_prefix_recipe(recipe)
    physical, single, coordinates = (prefix.get(key) for key in (
        "physical_preprocessing", "entry_m5_path_single_epoch", "physical_coordinates"))
    if (not all(isinstance(value, Mapping) for value in (physical, single, coordinates))
            or physical.get("entry_training_phase") != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
            or single["witness"]["training_phase"] != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
            or coordinates.get("population_witness") != single["binding"]
            or coordinates.get("native_sampler_contract") != single["witness"]["native_sampler_contract"]
            or prefix["train_rows"] != physical["train_rows"] or physical["train_rows"] != single["train_rows"]
            or any(key in recipe for key in ("exit_reference_policy", "exit_backup_steps", "chronological_entry_baseline",
                "chronological_train_only_measurement", "chronological_learning_continuation", "entry_learning_study"))):
        raise RuntimeError("NATIVE_DYNAMIC_M5_PHYSICAL_POPULATION_REQUIRED")
    observed = require_observed_entry_scope(recipe)
    if (not isinstance(observed, Mapping) or observed.get("training_phase") != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
            or observed.get("physical_sources") != physical["physical_sources"]
            or observed.get("target_contract") != single["witness"]["target_contract"]
            or not isinstance(observed.get("query_minutes"), tuple)
            or observed["query_minutes"] != tuple(single["witness"]["target_contract"]["query_contract"]["horizon_minutes"])):
        raise RuntimeError("NATIVE_DYNAMIC_M5_OBSERVED_SCOPE_REQUIRED")
    _require_dynamic_m5_loss_scale_identity(recipe, prefix=prefix, observed_scope=observed)
    return prefix


def _dynamic_m5_compute_profile(recipe):
    """The operator-prescoped two256-row probes each need16 encoder forwards."""
    expected = {"max_model_forwards": 16, "max_state_views": 256,
                "max_wall_seconds": 10800, "progress_interval_forwards": 16}
    value = recipe.get("val_limits")
    if (not isinstance(value, Mapping) or set(value) != set(expected)
            or any(type(value[key]) is not int or value[key] != number for key, number in expected.items())):
        raise RuntimeError("NATIVE_DYNAMIC_M5_COMPUTE_PROFILE_INVALID")
    return dict(expected)


def require_chronological_prefix_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Read the frozen prefix identity; this alone never grants run authority."""
    import numpy as np
    value = recipe.get("chronological_prefix")
    if (not isinstance(value, Mapping)
            or set(value) not in ({"design", "normalization_result", "labels_result"},
                                  {"design", "normalization_result", "labels_result", "native_coordinates"})
            or recipe.get("initialization") != "fresh_existing_model_constructor_no_checkpoint_weights"
            or any(recipe.get(key) is not None for key in ("seed_launch", "seed_authority", "smoke_full_val"))
            or any(key in recipe for key in ("candidate_resume_origin", "native_calibration", "frozen_readout_evaluation", "exit_backup_steps"))):
        raise RuntimeError("NATIVE_PREFIX_RECIPE_INVALID")
    artifacts = {key: require_binding(binding, label="native prefix artifact", verify_file=True)
                 for key, binding in value.items()}
    design, normalization, labels = (read_bound_json(Path(artifacts[key]["path"]), artifacts[key]["sha256"])
                                     for key in ("design", "normalization_result", "labels_result"))
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_SINGLE_EPOCH_SCOPE_KEY, require_m5_single_epoch_population, planned_prefix_training_rows,
    )
    single_epoch = require_m5_single_epoch_population(design)
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    dynamic = design.get("scope", {}).get("entry_only_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if dynamic and ("exit_reference_policy" in recipe
            or design.get("targets", {}).get("dynamic_target_contract") != entry_observed_dynamic_m5_path_contract()
            or "reference_policy" in design.get("targets", {})):
        raise RuntimeError("NATIVE_DYNAMIC_M5_ENTRY_TARGET_REQUIRED")
    if ((single_epoch is None and M5_SINGLE_EPOCH_SCOPE_KEY in recipe)
            or (single_epoch is not None and (recipe.get(M5_SINGLE_EPOCH_SCOPE_KEY) != single_epoch["binding"]
                or "native_coordinates" not in artifacts or recipe.get("trainer_cli", {}).get("epochs") != 1))):
        raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_EXPLICIT_SCOPE_REQUIRED")
    if (design.get("schema_version") != "gx1_frozen_chronological_learning_design_v1"
            or design.get("initialization", {}).get("mode") != recipe["initialization"]
            or design.get("scope", {}).get("test_sealed") is not True
            or design.get("scope", {}).get("same_architecture") is not True
            or design.get("scope", {}).get("one_experiment") is not True
            or design.get("budget", {}).get("planned_optimizer_steps") != _planned_prefix_optimizer_steps(design)
            or design.get("budget", {}).get("maximum_trained_entry_rows") != planned_prefix_training_rows(design)
            or design.get("budget", {}).get("epochs_completed") != (1 if single_epoch is not None else 0)
            or design.get("budget", {}).get("repeat_or_automatic_extension") is not False
            or (not dynamic and require_reference_policy_contract(recipe.get("exit_reference_policy"))
                != require_reference_policy_contract(design["targets"]["reference_policy"]))):
        raise RuntimeError("NATIVE_PREFIX_DESIGN_OR_TARGET_MISMATCH")
    controls = recipe.get("trainer_cli", {})
    from gx1.contracts.entry_observed_market_horizons_v1 import allowed_prefix_training_seeds
    if (type(controls.get("seed")) is not int
            or controls["seed"] not in allowed_prefix_training_seeds(design["initialization"]["seed"])
            or any(type(controls.get(key)) is not type(expected) or controls[key] != expected
                   for key, expected in {"batch_size":16, "learning_rate":0.0001,
                                         "weight_decay":0.0001, "grad_accum_steps":1}.items())):
        raise RuntimeError("NATIVE_PREFIX_TRAINING_CONTROLS_INVALID")
    if any(key in design.get("calendar", {}) for key in (
            "physical_source_splits", "physical_coordinate_namespaces_are_separate", "source_bindings")):
        physical = require_physical_chronological_preprocessing(
            artifacts, design=design, normalization=normalization, labels=labels)
        coordinates = (require_physical_native_training_coordinates(
            artifacts, design=design, physical=physical) if "native_coordinates" in artifacts else None)
        return {"artifacts": artifacts, "design": design,
                **({"entry_m5_path_single_epoch": single_epoch} if single_epoch is not None else {}),
                "train_rows": physical["train_rows"], "physical_preprocessing": physical,
                **({"physical_coordinates": coordinates} if coordinates is not None else {})}
    if "native_coordinates" in artifacts or single_epoch is not None:
        raise RuntimeError("NATIVE_PREFIX_LEGACY_COORDINATES_FORBIDDEN")
    preparation_binding = require_binding(normalization.get("prefix_preparation"), label="native prefix preparation", verify_file=True)
    preparation = read_bound_json(Path(preparation_binding["path"]), preparation_binding["sha256"])
    if (normalization.get("frozen_design") != artifacts["design"]
            or preparation.get("frozen_design") != artifacts["design"]
            or labels.get("prefix_preparation") != preparation_binding):
        raise RuntimeError("NATIVE_PREFIX_DESIGN_OR_TARGET_MISMATCH")
    row_binding = require_binding(preparation["bindings"]["TRAIN_ELIGIBLE_PARENT_ROWS"], label="native prefix rows", verify_file=True)
    rows = np.load(row_binding["path"], allow_pickle=False)
    if (rows.dtype != np.dtype("int64") or rows.ndim != 1 or len(rows) <= 4096
            or not np.array_equal(rows, np.unique(rows)) or np.any(rows < 0)
            or normalization.get("fit_entry_rows") != row_binding):
        raise RuntimeError("NATIVE_PREFIX_POPULATION_INVALID")
    return {"artifacts":artifacts, "design":design, "train_rows":len(rows)}



def require_entry_gradient_diagnostic(recipe, *, invocation_number=None, execution_budget=None):
    """One frozen TRAIN16 gradient contrast; no optimizer or control forwards."""
    prefix = require_chronological_prefix_recipe(recipe)
    if any(k in recipe for k in ("chronological_initial_measurement", "chronological_learning_measurement",
                                  "candidate_resume_origin", "native_calibration", "frozen_readout_evaluation")):
        raise RuntimeError("ENTRY_GRADIENT_MIXED_SCOPE")
    def load(value, label):
        binding = require_binding(value, label=label, verify_file=True)
        return binding, read_bound_json(Path(binding["path"]), binding["sha256"])
    plan_binding, plan = load(recipe.get("entry_gradient_diagnostic"), "Entry gradient plan")
    kind = plan.get("diagnostic_kind")
    joint = kind == "final_joint_update"
    representations = kind == "initial_final_entry_representations"
    main_encoder = representations and plan.get("schema_version") == "gx1_entry_representation_diagnostic_plan_v2"
    signal = joint or representations or kind in ("initial_final_entry_signal", "initial_final_forward_parity", "initial_final_entry_signal_inference_checked")
    if not joint and not representations and kind not in (None, "initial_final_entry_signal", "initial_final_forward_parity", "initial_final_entry_signal_inference_checked"):
        raise RuntimeError("ENTRY_GRADIENT_PLAN_INVALID")
    fixed = {"schema_version":"gx1_entry_gradient_diagnostic_plan_v1", "optimizer_steps":0,
             "train_entries":16, "model_forwards":2, "control_forwards":0, "max_invocations":1,
             "mode":"eval", "selection":"first16_existing_frozen_TRAIN_probe",
             "variants":["detached", "connected"], "test_data_used":False}
    if signal:
        fixed.update(schema_version="gx1_entry_signal_diagnostic_plan_v1", variants=["initial", "final"])
    if joint:
        fixed.update(schema_version="gx1_joint_update_diagnostic_plan_v2",variants=["final"],model_forwards=6)
    if representations:
        fixed.update(schema_version=("gx1_entry_representation_diagnostic_plan_v2" if main_encoder
                                     else "gx1_entry_representation_diagnostic_plan_v1"))
    if kind == "initial_final_forward_parity":
        fixed.update(schema_version="gx1_entry_forward_parity_plan_v1", model_forwards=4,
                     variants=["initial_inference", "initial_gradient", "final_inference", "final_gradient"])
    if kind == "initial_final_entry_signal_inference_checked":
        fixed.update(schema_version="gx1_entry_signal_diagnostic_plan_v2", model_forwards=4)
    if any(type(plan.get(k)) is not type(v) or plan[k] != v for k,v in fixed.items()):
        raise RuntimeError("ENTRY_GRADIENT_PLAN_INVALID")
    _, review = load(plan.get("review"), "completed fixed256 review")
    _, result = load(review.get("final_result"), "fixed final ONLINE")
    _, initial = load(result.get("initialization_result"), "prefix initialization")
    _, raw = load(plan.get("origin_cursor"), "fixed256 origin cursor")
    cursor = require_native_cursor(raw, expected_recipe=raw["recipe"], verify_files=True)
    state = cursor["resume_state"]
    _, origin_recipe = load(raw["recipe"], "completed origin recipe")
    repo = Path(__file__).resolve().parents[2]
    for key in ("gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
                "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"):
        artifact = origin_recipe["source_bindings"]["python:" + key]
        source = require_binding({k: artifact[k] for k in ("path", "sha256")},
                                 label="unchanged model/trainer", verify_file=True)
        if Path(source["path"]) != repo/key:
            raise RuntimeError("ENTRY_GRADIENT_SOURCE_INVALID")
    _, observation = load(plan.get("train_observation"), "frozen TRAIN observation")
    expected_review = (("gx1_causal_entry_fixed256_train_review_v1", "PAIRED_METRICS_COMPLETE_VERDICT_REQUIRED")
                       if signal else ("gx1_native_fixed256_paired_learning_review_v1", "REJECT_EXPANSION_LEARNING_GATE_FAILED"))
    if representations:
        expected_review = (("gx1_main_encoder_normalization_fixed256_train_review_v1" if main_encoder
                            else "gx1_residual_normalization_fixed256_train_review_v1"),
                           "PAIRED_METRICS_COMPLETE_VERDICT_REQUIRED")
    if joint:
        expected_review=("gx1_entry_fuse_normalization_fixed256_train_review_v1", "PAIRED_METRICS_COMPLETE_VERDICT_REQUIRED")
    if (review.get("schema_version") != expected_review[0]
            or review.get("decision") != expected_review[1]
            or result.get("schema_version") != "gx1_native_prefix_final_online_measurement_v1"
            or result.get("selected_model_variant") != "ONLINE" or result.get("optimizer_steps") != 256
            or result.get("frozen_targets_exactly_preserved") is not True
            or result.get("teacher_refreshed") is not False or result.get("test_data_used") is not False
            or review.get("training_pointer") != state.get("training_pointer")
            or review.get("training_state") != state.get("training_state")
            or state.get("global_optimizer_steps") != 256 or state.get("next_batch_offset") != 256
            or state.get("epoch_index") != 0 or state.get("phase") != "train"
            or state.get("complete") is not False or state.get("active_val_cursor") is not None
            or cursor.get("outcome") != "RESUMABLE"
            or initial.get("chronological_prefix") != prefix["artifacts"] or initial.get("files") != recipe.get("files")
            or plan.get("train_observation") != result.get("observations",{}).get("train")
            or observation.get("role") != "train" or observation.get("optimizer_steps") != 256
            or observation.get("test_data_used") is not False
            or observation.get("model_state_sha256") != result.get("model_state_sha256")
            or observation.get("cohort",{}).get("measurement_role") != "train"
            or observation.get("cohort",{}).get("plan") != prefix["artifacts"]["design"]
            or len(observation.get("diagnostics",{}).get("bounded_entry_observations",[])) != 256):
        raise RuntimeError("ENTRY_GRADIENT_ORIGIN_INVALID")
    cached_inputs = None
    initial_observation = None
    if signal:
        _, verdict = load(plan.get("verdict"), "completed causal verdict")
        _, cache_review = load(plan.get("cached_input_review"), "prior cached TRAIN16 inputs")
        _, cache_plan = load(cache_review.get("plan"), "prior cache plan")
        _, cache_cursor = load(cache_plan.get("origin_cursor"), "prior cache origin")
        _, cache_recipe = load(cache_cursor.get("recipe"), "prior cache recipe")
        cached_inputs = require_binding(cache_review.get("input_cache"), label="cached TRAIN16", verify_file=True)
        _, initial_result = load(result.get("initial_measurement"), "saved initial measurement")
        _, initial_observation = load(initial_result.get("observations",{}).get("train"), "saved initial TRAIN outputs")
        if (verdict.get("review") != plan.get("review")
                or verdict.get("decision") != ("REJECT_EXPANSION_ENTRY_FUSE_VALUE_FIT_IMPROVED_ACTIONS_UNCHANGED" if joint else "REJECT_EXPANSION_MAIN_ENCODER_PARTIAL_ENTRY_SIGNAL_NO_DECISION_IMPROVEMENT"
                    if main_encoder else "REJECT_EXPANSION_RESIDUAL_NORMALIZATION_NO_DECISION_IMPROVEMENT"
                    if representations else "REJECT_EXPANSION_CAUSAL_ENTRY_ALL_FLAT_EXIT_FIXED_BY_SIDE")
                or verdict.get("learning_gate_passed") is not False
                or cache_review.get("schema_version") != "gx1_entry_gradient_diagnostic_result_v1"
                or cache_review.get("selection") != "first16_existing_frozen_TRAIN_probe"
                or cache_review.get("optimizer_steps") != 0 or cache_review.get("test_data_used") is not False
                or cache_recipe.get("chronological_prefix") != recipe.get("chronological_prefix")
                or cache_recipe.get("files") != recipe.get("files")
                or initial_observation.get("cohort") != observation.get("cohort")
                or initial_observation.get("role") != "train" or initial_observation.get("optimizer_steps") != 0
                or initial_observation.get("model_state_sha256") != result.get("target_model_state_sha256")
                or initial_observation.get("test_data_used") is not False):
            raise RuntimeError("ENTRY_SIGNAL_CACHE_OR_BASELINE_INVALID")
    if main_encoder or joint:
        # Identical initial tensors do not establish ONLINE function parity.
        measured = require_chronological_learning_measurement(origin_recipe)
        artifacts = measured["artifacts"]
        if (result.get("initial_measurement") != artifacts["initial_measurement_result"]
                or review.get("initial_result") != artifacts["initial_measurement_result"]
                or result.get("initial_measurement_audit") != artifacts["initial_measurement_audit"]
                or result.get("initialization_result") != artifacts["initialization_result"]
                or not initial.get("model_functions")
                or initial.get("model_functions") != initial_result.get("model_functions")
                or initial.get("model_functions") != result.get("model_functions")):
            raise RuntimeError("ENTRY_REPRESENTATION_INITIAL_FUNCTION_INVALID")
    if representations or joint:
        _, audit = load(plan.get("input_binding_audit"), "representation input audit")
        expected_audit = {
            "schema_version":("gx1_joint_update_input_audit_v1" if joint else "gx1_main_encoder_representation_input_audit_v1" if main_encoder
                              else "gx1_residual_representation_input_audit_v1"),
            "decision":"EXACT_INPUT_TARGET_AND_ROW_BINDING_CONFIRMED_NATIVE_DIAGNOSTIC_EXTENSION_REQUIRED",
            "review":plan["review"], "verdict":plan["verdict"],
            "original_input_cache":cached_inputs, "new_training_observation":plan["train_observation"],
            "initial_training_observation":initial_result["observations"]["train"],
            "training_state":state["training_state"], "training_pointer":state["training_pointer"],
            "model_state_sha256":result["model_state_sha256"], "target_model_state_sha256":result["target_model_state_sha256"],
            "entries":16, "batch_identical_to_original_cache":True, "parent_row_order_exact":True,
            "targets_exact":True, "masks_exact":True, "same_prefix_and_file_bindings":True,
            "all_floating_batch_tensors_finite":True, "new_model_forwards":0, "optimizer_steps":0, "test_data_used":False}
        if main_encoder or joint:
            expected_audit.update(initial_measurement=artifacts["initial_measurement_result"],
                initial_measurement_audit=artifacts["initial_measurement_audit"],
                model_functions=initial["model_functions"], initial_targets_and_masks_exact=True)
        if any(type(audit.get(k)) is not type(v) or audit[k] != v for k,v in expected_audit.items()):
            raise RuntimeError("ENTRY_REPRESENTATION_INPUT_BINDING_INVALID")
        cached_inputs = require_binding(audit.get("cache"), label="corrected cached TRAIN16", verify_file=True)
    if kind == "initial_final_entry_signal_inference_checked":
        _, parity = load(plan.get("forward_parity_result"), "completed Entry forward parity")
        if (parity.get("schema_version") != "gx1_entry_forward_parity_result_v1"
                or parity.get("reused_input_cache") != cached_inputs
                or parity.get("training_state") != state["training_state"]
                or parity.get("training_pointer") != state["training_pointer"]
                or parity.get("optimizer_steps") != 0 or parity.get("model_forwards") != 4
                or parity.get("test_data_used") is not False
                or parity.get("model_and_original_checkpoint_preserved") is not True):
            raise RuntimeError("ENTRY_SIGNAL_PARITY_EVIDENCE_INVALID")
        for variant, expected_model in (("initial",result["target_model_state_sha256"]),("final",result["model_state_sha256"])):
            measured = parity.get("measurements",{}).get(variant,{})
            comparisons = measured.get("comparisons",{})
            if (measured.get("model_state_sha256") != expected_model
                    or comparisons.get("inference__cached_reference",{}).get("max_abs_difference_bps") != 0
                    or any(comparisons.get(k,{}).get("changed_actions") != 0 for k in
                           ("inference__cached_reference","gradient__cached_reference","gradient__inference"))):
                raise RuntimeError("ENTRY_SIGNAL_PARITY_EVIDENCE_INVALID")
    policy_binding, policy = load(recipe.get("next_run_policy"), "Entry gradient policy")
    if Path(policy_binding["path"]) != repo/"NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    _require_native_profile_and_economics(recipe, policy, repo)
    expected = {"plan":plan_binding, "chronological_prefix":prefix["artifacts"],
        "run_id":recipe.get("run_id"), "out_bundle_dir":recipe.get("out_bundle_dir"),
        "source_bindings_sha256":recipe.get("source_bindings_sha256"),
        "optimizer_steps":0, "origin_optimizer_steps":256, "max_invocations":1,
        "train_entries":16, "model_forwards":fixed["model_forwards"], "control_forwards":0, "test_data_used":False}
    if (policy.get("training_enabled") is not False or policy.get("entry_gradient_diagnostic") != expected
            or any(k in policy for k in ("chronological_learning_run", "chronological_initial_measurement",
                                         "native_learning_calibration", "frozen_readout_evaluation"))):
        raise RuntimeError("ENTRY_GRADIENT_NOT_AUTHORIZED")
    if invocation_number is not None and (type(invocation_number) is not int or invocation_number != 1):
        raise RuntimeError("ENTRY_GRADIENT_INVOCATION_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != 256
            or execution_budget.get("expected_active_pointer_sha256") is not None
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("ENTRY_GRADIENT_BUDGET_INVALID")
    return {"plan":plan, "plan_binding":plan_binding, "result":result,
            "observation":observation, "origin_resume_state":state,
            "cached_inputs":cached_inputs, "initial_observation":initial_observation}


def require_prefix_measurement_source_binding(recipe, *, measurement=None):
    """Bind physical v38 observations to the recipe's existing complete closure."""
    if "native_coordinates" not in recipe.get("chronological_prefix", {}):
        return {}
    from gx1.contracts.entry_model_native_train_launch_v1 import canonical_json_sha256, artifact_binding
    sources = recipe.get("source_bindings")
    if not isinstance(sources, Mapping) or not sources:
        raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_SOURCE_BINDING_REQUIRED")
    if any(not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
           for value in sources.values()):
        raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_SOURCE_BINDING_INVALID")
    checked = {key: artifact_binding(Path(value["path"])) for key, value in sources.items()}
    digest = canonical_json_sha256(checked)
    if checked != sources or recipe.get("source_bindings_sha256") != digest:
        raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_SOURCE_BINDING_INVALID")
    expected = {"native_recipe_source_bindings": checked,
                "native_recipe_source_bindings_sha256": digest}
    if measurement is not None and any(measurement.get(key) != value for key, value in expected.items()):
        continuation = require_chronological_continuation(recipe)
        if not continuation or continuation.get("resume_equivalence") is not True:
            raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_SOURCE_CHANGED")
        # A pure suffix replay uses the frozen origin baseline only as provenance.
        # Its current closure is still checked above, and the replay owner permits
        # only the named control/restore files to differ from that exact origin.
        origin = continuation["origin_recipe"]
        original = {"native_recipe_source_bindings": origin["source_bindings"],
                    "native_recipe_source_bindings_sha256": origin["source_bindings_sha256"]}
        if any(measurement.get(key) != value for key, value in original.items()):
            raise RuntimeError("NATIVE_PREFIX_MEASUREMENT_SOURCE_CHANGED")
    return expected


def _require_dynamic_initial_artifacts(recipe, prefix, initial, measurement):
    """Rejoin constructor provenance and Entry coordinates, without a teacher forward."""
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE as phase
    physical, single = prefix["physical_preprocessing"], prefix["entry_m5_path_single_epoch"]
    loss_scale_identity = _require_dynamic_m5_loss_scale_identity(recipe, prefix=prefix)
    common = {"entry_training_phase": phase, "target_model_role": "frozen_constructor_anchor_no_forward",
              "dynamic_m5_loss_scales": loss_scale_identity,
              "exit_evaluation": False, "native_launch_authorized": False, "exit_model_forward_count": 0}
    functions = initial.get("model_functions")
    if (any(type(value.get(key)) is not type(expected) or value[key] != expected
            for value in (initial, measurement) for key, expected in common.items())
            or initial.get("entry_only_base_normalization") != physical["entry_only_base_normalization"]
            or initial.get("base_contract_sha256") != physical["base_contract_sha256"]
            or initial.get("population_witness") != single["binding"]
            or initial.get("test_data_used") is not False
            or any(type(initial.get(key)) is not int or initial[key] != 0
                   for key in ("model_forwards", "optimizer_steps", "ema_steps", "normalization_refits"))
            or not isinstance(functions, Mapping) or set(functions) != {"online", "target"}
            or not isinstance(functions["online"], str) or not functions["online"]
            or functions["online"] != functions["target"]
            or any(type(measurement.get(key)) is not int or measurement[key] != 0
                   for key in ("actual_model_forwards", "optimizer_steps"))
            or not isinstance(initial.get("online_model_state_sha256"), str)
            or len(initial["online_model_state_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in initial["online_model_state_sha256"])
            or measurement.get("coordinate_result") != recipe.get("files", {}).get("entry_measurement_coordinates")):
        raise RuntimeError("NATIVE_DYNAMIC_M5_INITIAL_COMPONENT_BINDING_INVALID")
    require_prefix_measurement_source_binding(recipe, measurement=initial)
    require_prefix_measurement_source_binding(recipe, measurement=measurement)


def _require_dynamic_m5_observation(observation, *, identity, cohort):
    """Validate the existing observer's independent components and exact Entry trace."""
    import math
    import hashlib
    import numpy as np
    import pyarrow.parquet as pq
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        DYNAMIC_M5_POINTER_COLUMNS, MODEL_NATIVE_AUX_FORECAST_HORIZONS,
        MODEL_NATIVE_FORECAST_OUTPUT_DIM, MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT, MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT,
    )
    from gx1.contracts.entry_fitted_q_v1 import ENTRY_FITTED_Q_ACTION_ORDER

    def check(condition, reason):
        if not condition:
            raise RuntimeError("NATIVE_DYNAMIC_M5_OBSERVATION_" + reason)

    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)

    def number(value, *, nonnegative=False):
        return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)

    def numeric_array(value, shape, *, counts=False):
        try:
            array = np.asarray(value)
            check(array.shape == shape, "ARRAY_SHAPE_INVALID")
            flat = array.reshape(-1).tolist()
            check(all(type(cell) is int and 0 <= cell <= 256 for cell in flat) if counts else
                  all(number(cell, nonnegative=True) for cell in flat), "ARRAY_DOMAIN_INVALID")
            return array.astype(np.int64 if counts else np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise RuntimeError("NATIVE_DYNAMIC_M5_OBSERVATION_ARRAY_INVALID") from exc

    check(observation.get("target_model_role") == "frozen_constructor_anchor_no_forward"
          and observation.get("exit_evaluation") is False and observation.get("native_launch_authorized") is False,
          "CONSTRUCTOR_ANCHOR_REQUIRED")
    diagnostics = observation.get("diagnostics", {})
    required = {"dynamic_frozen_targets", "dynamic_component_errors", "dynamic_entry_observations", "dynamic_curve_statistics",
        "dynamic_candidate_diagnostic_coverage", "entry_model_forward_count", "exit_model_forward_count",
        "dynamic_candidate_native_authority_admitted", "candidate_target_comparison_uses_prediction_selected_horizons_only",
        "component_masks_have_independent_denominators", "realized_cash_or_native_policy_evaluated"}
    check(isinstance(diagnostics, Mapping) and set(diagnostics) == required
          and type(diagnostics.get("entry_model_forward_count")) is int and diagnostics["entry_model_forward_count"] == 16
          and type(diagnostics.get("exit_model_forward_count")) is int and diagnostics["exit_model_forward_count"] == 0
          and diagnostics.get("dynamic_candidate_native_authority_admitted") is False
          and diagnostics.get("candidate_target_comparison_uses_prediction_selected_horizons_only") is True
          and diagnostics.get("component_masks_have_independent_denominators") is True
          and diagnostics.get("realized_cash_or_native_policy_evaluated") is False, "ENTRY_ONLY_DIAGNOSTICS_REQUIRED")
    parents, times, queries = cohort["parent_entry_row_indices"], cohort["entry_time_ns"], list(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
    frozen = diagnostics["dynamic_frozen_targets"]
    check(isinstance(frozen, Mapping) and set(frozen) == {"entry_supervision_identity", "parent_entry_rows", "entry_time_ns",
          "dynamic_entry_m1_rows", "query_minutes", "base_forecast_horizons", "support_forecast_layout", "components"}
          and frozen["entry_supervision_identity"] == identity and frozen["parent_entry_rows"] == parents
          and frozen["entry_time_ns"] == times and frozen["query_minutes"] == queries
          and frozen["base_forecast_horizons"] == list(MODEL_NATIVE_AUX_FORECAST_HORIZONS)
          and frozen["support_forecast_layout"] == list(MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT), "FROZEN_TARGET_IDENTITY_INVALID")
    # Read just the immutable pointer column; outcome values and feature inputs are not opened.
    wanted = np.asarray(parents, np.int64)
    actual_pointers = np.full(len(parents), -1, np.int64)
    parquet = pq.ParquetFile(cohort["parent_entry_parquet"]["path"])
    check(parquet.metadata.num_rows == cohort["population_rows"], "POINTER_POPULATION_INVALID")
    offset = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=[DYNAMIC_M5_POINTER_COLUMNS[0]], use_threads=False):
        rows = batch.column(0).to_numpy(zero_copy_only=False)
        check(rows.dtype == np.dtype("int64"), "POINTER_DTYPE_INVALID")
        selected = np.flatnonzero((wanted >= offset) & (wanted < offset + len(rows)))
        actual_pointers[selected] = rows[wanted[selected] - offset]
        offset += len(rows)
    check(offset == cohort["population_rows"] and bool((actual_pointers >= 0).all())
          and frozen["dynamic_entry_m1_rows"] == actual_pointers.tolist(), "GLOBAL_POINTER_TRACE_INVALID")
    names = ("dynamic_terminal_net_markout_bps", "dynamic_mfe_bps", "dynamic_pre_mfe_mae_upper_bps",
             "dynamic_closed_mfe_bps", "dynamic_closed_pre_mfe_mae_upper_bps", "forecast_pred")
    components, errors = frozen["components"], diagnostics["dynamic_component_errors"]
    check(isinstance(components, Mapping) and isinstance(errors, Mapping)
          and set(components) == set(errors) == set(names), "COMPONENTS_INVALID")
    for index, name in enumerate(names):
        row, error = components[name], errors[name]
        shape = (MODEL_NATIVE_FORECAST_OUTPUT_DIM,) if index == 5 else (len(queries), 3 if index == 0 else 2)
        count_key = "observed_cells_by_forecast_coordinate" if index == 5 else "observed_cells_by_horizon_and_side"
        check(isinstance(row, Mapping) and set(row) == {"shape", "target_dtype", "mask_dtype", "target_storage_sha256",
              "observed_mask_sha256", "observed_cells", "observed_rows", count_key}
              and row["shape"] == [256, *shape] and row["target_dtype"] == "float32" and row["mask_dtype"] == "bool"
              and digest(row["target_storage_sha256"]) and digest(row["observed_mask_sha256"]), "COMPONENT_STORAGE_INVALID")
        counts = numeric_array(row[count_key], shape, counts=True)
        cells, rows = int(counts.sum()), row["observed_rows"]
        check(type(row["observed_cells"]) is int and row["observed_cells"] == cells
              and type(rows) is int and 0 <= rows <= 256 and (cells == 0) == (rows == 0)
              and (not cells or int(counts.max()) <= rows <= cells), "COMPONENT_COUNTS_INVALID")
        error_keys = {"observed_cells", "observed_rows", "sum_squared_error_bps2", "sum_absolute_error_bps",
            "sum_squared_error_by_forecast_coordinate_bps2" if index == 5 else "sum_squared_error_by_horizon_and_side_bps2",
            "sum_absolute_error_by_forecast_coordinate_bps" if index == 5 else "sum_absolute_error_by_horizon_and_side_bps"}
        check(isinstance(error, Mapping) and set(error) == error_keys | ({"mse_bps2", "mae_bps"} if cells else set())
              and type(error["observed_cells"]) is int and error["observed_cells"] == cells
              and type(error["observed_rows"]) is int and error["observed_rows"] == rows, "ERROR_COUNTS_INVALID")
        for prefix, scalar, average in (("sum_squared_error", "sum_squared_error_bps2", "mse_bps2"),
                                        ("sum_absolute_error", "sum_absolute_error_bps", "mae_bps")):
            array_key = prefix + ("_by_forecast_coordinate" if index == 5 else "_by_horizon_and_side") + ("_bps2" if prefix == "sum_squared_error" else "_bps")
            array = numeric_array(error[array_key], shape)
            check(bool((array[counts == 0] == 0).all()) and number(error[scalar], nonnegative=True)
                  and error[scalar] == float(array.sum())
                  and (not cells or error[average] == float(array.sum()) / cells), "ERROR_AGGREGATION_INVALID")
    for first, second in ((names[1], names[2]), (names[3], names[4])):
        check(components[first]["observed_mask_sha256"] == components[second]["observed_mask_sha256"]
              and components[first]["observed_cells_by_horizon_and_side"] == components[second]["observed_cells_by_horizon_and_side"],
              "PAIRED_PATH_MASK_MISMATCH")
    traces = diagnostics["dynamic_entry_observations"]
    check(isinstance(traces, list) and len(traces) == len(parents) == 256, "TRACE_ROWS_INVALID")
    def trace_component(cell, shape, *, action_order=None):
        keys = {"prediction_bps", "target_bps", "observed"} | ({"action_order"} if action_order else set())
        check(isinstance(cell, Mapping) and set(cell) == keys
              and (action_order is None or cell["action_order"] == action_order), "FULL_CURVE_SCHEMA_INVALID")
        prediction, target, mask = (np.asarray(cell[key], dtype=object) for key in ("prediction_bps", "target_bps", "observed"))
        check(prediction.shape == target.shape == mask.shape == shape, "FULL_CURVE_SHAPE_INVALID")
        check(all(number(value) for value in prediction.flat)
              and all(type(value) is bool for value in mask.flat), "FULL_CURVE_DOMAIN_INVALID")
        check(all(number(value) if valid else value is None for value, valid in zip(target.flat, mask.flat)),
              "FULL_CURVE_TARGET_MASK_INVALID")
        observed = mask.astype(np.bool_)
        return prediction.astype(np.float64), np.where(observed, target, 0).astype(np.float64), observed
    trace_values = {name: [] for name in names}
    actions = dict.fromkeys(ENTRY_FITTED_Q_ACTION_ORDER, 0)
    tied, abstained = 0, 0
    for index, trace in enumerate(traces):
        check(isinstance(trace, Mapping) and set(trace) == {"entry_row_index", "entry_time_ns", "predicted_candidate_quality_bps",
              "diagnostic_action", "diagnostic_abstained", "side_diagnostics", "full_query_curves", "forecast_support"}
              and type(trace["entry_row_index"]) is int and trace["entry_row_index"] == parents[index]
              and type(trace["entry_time_ns"]) is int and trace["entry_time_ns"] == times[index], "TRACE_IDENTITY_INVALID")
        full = trace["full_query_curves"]
        check(isinstance(full, Mapping) and set(full) == {"query_minutes", "components"}
              and isinstance(full["query_minutes"], list) and all(type(value) is int for value in full["query_minutes"])
              and full["query_minutes"] == queries and isinstance(full["components"], Mapping)
              and set(full["components"]) == set(names[:5]), "FULL_CURVE_QUERY_INVALID")
        for component_index, name in enumerate(names[:5]):
            axes = list(ENTRY_FITTED_Q_ACTION_ORDER[:3 if component_index == 0 else 2])
            trace_values[name].append(trace_component(full["components"][name], (len(queries), len(axes)), action_order=axes))
        support = trace["forecast_support"]
        check(isinstance(support, Mapping) and set(support) == {"output_layout", "prediction_bps", "target_bps", "observed"}
              and support["output_layout"] == list(MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT), "FORECAST_TRACE_LAYOUT_INVALID")
        trace_values["forecast_pred"].append(trace_component({key:value for key,value in support.items() if key != "output_layout"},
                                                          (MODEL_NATIVE_FORECAST_OUTPUT_DIM,)))
        p, target, mask = trace_values[names[0]][-1]
        check(bool((p[:, 2] == 0).all()) and bool((target[:, 2] == 0).all())
              and np.array_equal(mask[:, 2], mask[:, 0] & mask[:, 1]), "FULL_CURVE_FLAT_MASK_INVALID")
        for first, second in ((names[1], names[2]), (names[3], names[4])):
            check(np.array_equal(trace_values[first][-1][2], trace_values[second][-1][2]), "FULL_CURVE_PAIRED_MASK_INVALID")
        candidate = trace["predicted_candidate_quality_bps"]
        check(isinstance(candidate, list) and len(candidate) == 3 and all(number(value) for value in candidate)
              and candidate[2] == 0, "CANDIDATE_VALUES_INVALID")
        winners = [position for position,value in enumerate(candidate) if value == max(candidate)]
        action = ENTRY_FITTED_Q_ACTION_ORDER[winners[0]] if len(winners) == 1 else "TIED_FAIL_CLOSED"
        check(trace["diagnostic_action"] == action and type(trace["diagnostic_abstained"]) is bool
              and trace["diagnostic_abstained"] == (action in ("FLAT", "TIED_FAIL_CLOSED")), "CANDIDATE_TRACE_INVALID")
        if action == "TIED_FAIL_CLOSED": tied += 1
        else: actions[action] += 1
        abstained += int(trace["diagnostic_abstained"])
        sides = trace["side_diagnostics"]
        check(isinstance(sides, list) and len(sides) == 2, "SIDE_TRACE_INVALID")
        for side, detail in enumerate(sides):
            check(isinstance(detail, Mapping) and set(detail) == {"side", "predicted_best_horizon_minutes", *names[:5]}
                  and detail["side"] == ENTRY_FITTED_Q_ACTION_ORDER[side], "SIDE_TRACE_INVALID")
            best = detail["predicted_best_horizon_minutes"]
            check(isinstance(best, list) and bool(best) and all(type(value) is int and value in queries for value in best)
                  and best == sorted(set(best)), "HORIZON_TRACE_INVALID")
            for name in names[:5]:
                cell = detail[name]
                check(isinstance(cell, Mapping) and type(cell.get("observed_cells")) is int
                      and 0 <= cell["observed_cells"] <= len(best)
                      and set(cell) == {"observed_cells", "predicted_mean_bps"} | ({"observed_target_mean_bps"} if cell["observed_cells"] else set())
                      and number(cell["predicted_mean_bps"])
                      and (not cell["observed_cells"] or number(cell["observed_target_mean_bps"])), "SIDE_COMPONENT_TRACE_INVALID")
                prediction, target, mask = trace_values[name][-1]
                selected = np.asarray(best, np.int64) - 1
                valid = mask[selected, side]
                check(cell["observed_cells"] == int(valid.sum())
                      and cell["predicted_mean_bps"] == float(prediction[selected, side].mean())
                      and (not valid.any() or cell["observed_target_mean_bps"] == float(target[selected, side][valid].mean())),
                      "FULL_CURVE_SELECTED_TRACE_MISMATCH")
    for component_index, name in enumerate(names):
        shape = (MODEL_NATIVE_FORECAST_OUTPUT_DIM,) if component_index == 5 else (len(queries), 3 if component_index == 0 else 2)
        counts = np.zeros(shape, np.int64)
        squared, absolute = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
        mask_digest, observed_rows = hashlib.sha256(), 0
        # Match the observer's exact batch16/F64 reduction order; no tolerance.
        for offset in range(0, len(traces), 16):
            prediction, target, mask = (np.stack([row[index] for row in trace_values[name][offset:offset + 16]]) for index in range(3))
            mask_digest.update(np.asarray(mask, dtype=np.bool_, order="C").tobytes())
            observed_rows += int(mask.reshape(len(mask), -1).any(axis=1).sum())
            counts += mask.sum(axis=0)
            difference = prediction - target
            squared += np.where(mask, difference * difference, 0).sum(axis=0)
            absolute += np.where(mask, np.abs(difference), 0).sum(axis=0)
        suffix = "_by_forecast_coordinate" if component_index == 5 else "_by_horizon_and_side"
        count_key = "observed_cells" + suffix
        check(mask_digest.hexdigest() == components[name]["observed_mask_sha256"]
              and observed_rows == components[name]["observed_rows"]
              and np.array_equal(counts, np.asarray(components[name][count_key]))
              and np.array_equal(squared, np.asarray(errors[name]["sum_squared_error" + suffix + "_bps2"]))
              and np.array_equal(absolute, np.asarray(errors[name]["sum_absolute_error" + suffix + "_bps"])),
              "FULL_CURVE_AGGREGATE_MISMATCH")
    statistics, coverage = diagnostics["dynamic_curve_statistics"], diagnostics["dynamic_candidate_diagnostic_coverage"]
    action_counts = [actions[name] for name in ENTRY_FITTED_Q_ACTION_ORDER]
    check(isinstance(statistics, Mapping) and type(statistics.get("curve_evaluation_rows")) is int
          and statistics["curve_evaluation_rows"] == 256
          and statistics.get("dynamic_candidate_unique_diagnostic_counts") == action_counts
          and type(statistics.get("dynamic_candidate_tied_diagnostic_rows")) is int
          and statistics["dynamic_candidate_tied_diagnostic_rows"] == tied
          and coverage == {"entries":256, "unique_action_counts":action_counts, "tied_fail_closed_rows":tied,
              "abstained_rows":abstained, "selected_fraction":(256 - abstained) / 256}, "DIAGNOSTIC_COVERAGE_INVALID")
    return dict(observation)


def _require_prefix_fresh_measurement_artifacts(recipe, value):
    """Share the unchanged fresh/source/coordinate checks across explicit scopes."""
    prefix = require_chronological_prefix_recipe(recipe)
    if not isinstance(value, Mapping) or set(value) != {"initialization_result", "measurement_binding_result"}:
        raise RuntimeError("NATIVE_PREFIX_INITIAL_MEASUREMENT_INVALID")
    artifacts = {k: require_binding(v, label="initial measurement artifact", verify_file=True)
                 for k, v in value.items()}
    initial, measurement = (read_bound_json(Path(artifacts[k]["path"]), artifacts[k]["sha256"])
                            for k in ("initialization_result", "measurement_binding_result"))
    repo = Path(__file__).resolve().parents[2]
    model_source = "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py"
    if (initial.get("schema_version") != "gx1_prefix_fresh_initialization_v1"
            or initial.get("decision") != "ACTUAL_NATIVE_FRESH_COMPONENTS_READY_NO_LEARNING_MEASURED"
            or initial.get("chronological_prefix") != prefix["artifacts"]
            or initial.get("files") != recipe.get("files")
            or initial.get("inherited_checkpoint_weights") is not False
            or initial.get("optimizer_state_empty") is not True or initial.get("target_frozen") is not True
            or any(initial.get(k) != 0 for k in ("model_forwards", "optimizer_steps", "ema_steps", "normalization_refits"))
            or initial.get("online_model_state_sha256") != initial.get("target_model_state_sha256")
            or initial.get("online_model_state_sha256") != initial.get("ema_model_state_sha256")
            or initial.get("source_bindings", {}).get(model_source) != file_sha256(repo / model_source)
            or measurement.get("schema_version") != "gx1_prefix_native_measurement_binding_result_v1"
            or measurement.get("decision") != "FROZEN_TRAIN_CONTROL_COORDINATES_AND_NATIVE_MEASUREMENT_BINDING_READY"
            or measurement.get("actual_model_forwards") != 0 or measurement.get("optimizer_steps") != 0
            or measurement.get("test_data_used") is not False):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_ARTIFACT_MISMATCH")
    require_binding(initial["initial_state"], label="saved fresh initial state", verify_file=True)
    coordinates = require_binding(measurement["coordinate_result"], label="frozen measurement coordinates", verify_file=True)
    coordinate_result = read_bound_json(Path(coordinates["path"]), coordinates["sha256"])
    dynamic = _dynamic_m5_native_design(recipe) is not None
    coordinate_design = (coordinate_result.get("chronological_prefix", {}).get("design")
                         if dynamic else coordinate_result.get("design"))
    if coordinate_design != prefix["artifacts"]["design"]:
        raise RuntimeError("NATIVE_PREFIX_INITIAL_DESIGN_MISMATCH")
    if "physical_preprocessing" in prefix:
        from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_measurement_cohort
        if coordinate_result.get("chronological_prefix") != prefix["artifacts"]:
            raise RuntimeError("NATIVE_PREFIX_INITIAL_PHYSICAL_COORDINATES_MISMATCH")
        for role in ("train", "control"):
            build_chronological_measurement_cohort(prefix["artifacts"]["design"], coordinates, role=role)
    if _dynamic_m5_native_design(recipe) is not None:
        prefix = _require_dynamic_m5_native_prefix(recipe)
        _require_dynamic_initial_artifacts(recipe, prefix, initial, measurement)
    return prefix, artifacts, initial, measurement


def require_dynamic_m5_technical_qualification(recipe, *, invocation_number=None, execution_budget=None):
    """Admit one disposable mechanics proof, never a shortened production epoch."""
    key = "entry_dynamic_m5_technical_qualification"
    forbidden = {"chronological_initial_measurement", "chronological_learning_measurement",
        "chronological_learning_continuation", "chronological_train_only_measurement",
        "candidate_resume_origin", "native_calibration", "frozen_readout_evaluation",
        "entry_gradient_diagnostic", "entry_learning_study", "chronological_entry_baseline",
        "frozen_train_policy_evaluation", "frozen_entry_selector_probe"}
    if _dynamic_m5_native_design(recipe) is None or any(name in recipe for name in forbidden):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_MIXED_SCOPE")
    binding = require_binding(recipe.get(key), label="dynamic technical qualification", verify_file=True)
    plan = read_bound_json(Path(binding["path"]), binding["sha256"])
    extra = {"entry_training_phase", "chronological_prefix", "initialization_result", "measurement_binding_result",
        "entry_execution_profile", "files", "run_id", "out_bundle_dir", "source_bindings_sha256",
        "measurement_cohort_sha256", "native_train_parent_rows", "initial_measurement_result", "initial_measurement_audit"}
    if (not isinstance(plan, Mapping) or set(plan) != set(DYNAMIC_M5_TECHNICAL_FIXED_SCOPE) | extra
            or any(type(plan.get(name)) is not type(expected) or plan[name] != expected
                   for name, expected in DYNAMIC_M5_TECHNICAL_FIXED_SCOPE.items())):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_PLAN_INVALID")
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    reference = require_dynamic_m5_execution_profile(recipe)
    if (reference is None or plan["entry_execution_profile"] != reference
            or plan["entry_training_phase"] != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
            or any(plan[name] != recipe.get(name) for name in (
                "chronological_prefix", "files", "run_id", "out_bundle_dir", "source_bindings_sha256"))):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_RECIPE_MISMATCH")
    prefix, artifacts, initial, measurement = _require_prefix_fresh_measurement_artifacts(
        recipe, {name: plan[name] for name in ("initialization_result", "measurement_binding_result")})
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_measurement_cohort
    cohorts = {role: build_chronological_measurement_cohort(prefix["artifacts"]["design"],
        measurement["coordinate_result"], role=role) for role in ("train", "control")}
    parents = prefix["physical_coordinates"]["epoch0_parent_order"][:DYNAMIC_M5_TECHNICAL_FIXED_SCOPE["train_entries"]].tolist()
    if (len(parents) != 32 or not isinstance(plan["native_train_parent_rows"], list)
            or plan["native_train_parent_rows"] != parents
            or any(type(parent) is not int for parent in plan["native_train_parent_rows"])
            or plan["measurement_cohort_sha256"] != {role: value["cohort_sha256"] for role, value in cohorts.items()}):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_ROWS_MISMATCH")
    completed = require_chronological_learning_measurement(recipe,
        initial_measurement_audit=plan["initial_measurement_audit"])
    if (completed["artifacts"]["initial_measurement_result"] != plan["initial_measurement_result"]
            or any(completed["artifacts"][name] != artifacts[name] for name in artifacts)):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_INITIAL_AUDIT_MISMATCH")
    artifacts = completed["artifacts"]
    repo = Path(__file__).resolve().parents[2]
    policy_binding = require_binding(recipe.get("next_run_policy"), label="dynamic technical policy", verify_file=True)
    if Path(policy_binding["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(policy_binding["path"]), policy_binding["sha256"])
    _require_native_profile_and_economics(recipe, policy, repo)
    expected = {"plan": binding, **dict(plan)}
    _expected_observed_entry_binding(recipe, expected)
    if (policy.get("training_enabled") is not True or policy.get("full_epoch_training_allowed") is not False
            or policy.get(key) != expected or any(name in policy for name in (
                "chronological_initial_measurement", "chronological_learning_run", "native_learning_calibration",
                "entry_m5_path_single_epoch_run", "frozen_readout_evaluation"))):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_NOT_AUTHORIZED")
    if invocation_number is not None and (type(invocation_number) is not int or invocation_number != 1):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_INVOCATION_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != DYNAMIC_M5_TECHNICAL_CURSOR_CEILING
            or execution_budget.get("expected_active_pointer_sha256") is not None
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000 or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("NATIVE_DYNAMIC_M5_TECHNICAL_BUDGET_INVALID")
    return {"plan_binding": binding, "plan": dict(plan), "prefix": prefix, "artifacts": artifacts,
        "initialization": initial, "measurement": measurement, "cohorts": cohorts,
        "optimizer_step_ceiling": DYNAMIC_M5_TECHNICAL_CURSOR_CEILING}


def require_chronological_initial_measurement(recipe, *, invocation_number=None, execution_budget=None):
    """One native initial measurement, using saved fresh weights and zero steps."""
    prefix, artifacts, initial, measurement = _require_prefix_fresh_measurement_artifacts(
        recipe, recipe.get("chronological_initial_measurement"))
    repo = Path(__file__).resolve().parents[2]
    binding = require_binding(recipe.get("next_run_policy"), label="initial measurement policy", verify_file=True)
    if Path(binding["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(binding["path"]), binding["sha256"])
    _require_native_profile_and_economics(recipe, policy, repo)
    expected = {**artifacts, "chronological_prefix": prefix["artifacts"], "run_id": recipe.get("run_id"),
        "out_bundle_dir": recipe.get("out_bundle_dir"), "source_bindings_sha256": recipe.get("source_bindings_sha256"),
        "optimizer_steps": 0, "max_invocations": 1, "teacher_refresh_allowed": False,
        "full_epoch_training_allowed": False, "full_val_allowed": False, "test_data_used": False}
    _expected_observed_entry_binding(recipe, expected)
    if "entry_m5_path_single_epoch" in prefix:
        expected["entry_m5_path_single_epoch"] = prefix["entry_m5_path_single_epoch"]["binding"]
    if "chronological_train_only_measurement" in recipe:
        if recipe["chronological_train_only_measurement"] is not True:
            raise RuntimeError("NATIVE_PREFIX_TRAIN_ONLY_MEASUREMENT_INVALID")
        expected["chronological_train_only_measurement"] = True
    if (policy.get("training_enabled") is not False or policy.get("chronological_initial_measurement") != expected
            or any(k in policy for k in ("chronological_learning_run", "native_learning_calibration", "frozen_readout_evaluation"))):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_MEASUREMENT_NOT_AUTHORIZED")
    if invocation_number is not None and (type(invocation_number) is not int or invocation_number != 1):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_INVOCATION_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != 0
            or execution_budget.get("expected_active_pointer_sha256") is not None
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_BUDGET_INVALID")
    return {"artifacts": artifacts, "initialization": initial, "measurement": measurement}


def _require_derived_entry_baseline(recipe, *, initial_result, train_observation):
    """Verify an explicit target derivation; retain the original native receipt."""
    import torch
    from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_state_values
    binding = require_binding(recipe["chronological_entry_baseline"], label="derived Entry baseline", verify_file=True)
    result = read_bound_json(Path(binding["path"]), binding["sha256"])
    baseline_binding = require_binding(result.get("derived_baseline"), label="derived TRAIN targets", verify_file=True)
    baseline = read_bound_json(Path(baseline_binding["path"]), baseline_binding["sha256"])
    original_binding = initial_result["observations"]["train"]
    if (recipe.get("chronological_train_only_measurement") is not True
            or result.get("schema_version") != "gx1_derived_causal_entry_train_baseline_result_v1"
            or result.get("decision") != "CANONICAL_DERIVED_TRAIN_BASELINE_READY_NO_NEW_LEARNING_MEASURED"
            or result.get("source_observations", {}).get("initial") != original_binding
            or result.get("initial_measurement_audit") != recipe["chronological_learning_measurement"]
            or result.get("old_entry_targets_exactly_reconstructed") is not True
            or result.get("new_model_forwards") != 0 or result.get("optimizer_steps") != 0
            or result.get("test_data_used") is not False or result.get("control_observations_accessed") is not False
            or baseline.get("schema_version") != "gx1_derived_causal_entry_train_baseline_v1"
            or baseline.get("role") != "train" or baseline.get("source_observation") != original_binding
            or baseline.get("initial_measurement_audit") != recipe["chronological_learning_measurement"]
            or any(baseline.get(k) != train_observation[k] for k in ("cohort", "model_state_sha256", "target_model_state_sha256"))
            or baseline.get("target_semantics") != "observed_reference_anchor_V_mu_without_hindsight_action"):
        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_BASELINE_INVALID")
    repo = Path(__file__).resolve().parents[2]
    for name in ("gx1/contracts/unified_exit_economic_step_provider_v1.py",
                 "gx1/contracts/unified_exit_reference_policy_v1.py",
                 "gx1/scripts/run_unified_exit_random_access_val_v1.py",
                 "gx1/contracts/unified_exit_random_access_training_v1.py"):
        if result.get("source_bindings", {}).get(name) != {"path":str(repo/name), "sha256":file_sha256(repo/name)}:
            raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_SOURCE_CHANGED")
    if any(result.get("data_bindings", {}).get(k) != recipe["files"][k]
           for k in ("random_access_root", "economics_readiness", "train_cost_authority")):
        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_DATA_CHANGED")
    policy = require_reference_policy_contract(baseline["reference_policy"])
    if policy != require_reference_policy_contract(recipe["exit_reference_policy"]):
        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_POLICY_CHANGED")
    old = train_observation["diagnostics"]["bounded_entry_observations"]
    anchors = train_observation["diagnostics"]["bounded_exit_anchor_observations"]
    liquidation = torch.tensor(baseline["canonical_first_liquidation_bps"], dtype=torch.float32)
    if (len(old) != 256 or len(anchors) != 256 or liquidation.shape != (256, 2)
            or not bool(torch.isfinite(liquidation).all())
            or [x["entry_row_index"] for x in old] != [x["entry_row_index"] for x in anchors]
            or any(x["state_index"] != 0 or x["reference_policy_sha256"] != policy["policy_sha256"] for x in anchors)):
        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_COORDINATES_INVALID")
    hold = torch.tensor([x["target_hold_bps"] for x in anchors], dtype=torch.float32)
    q = torch.stack((hold, torch.zeros_like(hold)), dim=-1)
    values = liquidation + reference_policy_state_values(policy=policy, action_q_bps=q, action_valid_mask=torch.ones_like(q,dtype=torch.bool))
    old_expected = torch.cat((liquidation + hold.clamp_min(0), torch.zeros(256,1)), dim=1)
    targets = torch.cat((values, torch.zeros(256,1)), dim=1).tolist()
    expected_rows = [{**row, "target_q_bps": target} for row,target in zip(old,targets)]
    if (old_expected.tolist() != [x["target_q_bps"] for x in old]
            or baseline.get("entry_observations") != expected_rows):
        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_TARGET_OR_PREDICTION_CHANGED")
    return binding, baseline


def require_chronological_learning_measurement(recipe, *, initial_measurement_audit=None):
    """Bind the completed initial observation; prefix identity is checked by the caller."""
    binding = require_binding(recipe.get("chronological_learning_measurement") if initial_measurement_audit is None
                              else initial_measurement_audit, label="completed initial measurement audit", verify_file=True)
    audit = read_bound_json(Path(binding["path"]), binding["sha256"])
    def load(value, label):
        checked = require_binding(value, label=label, verify_file=True)
        return read_bound_json(Path(checked["path"]), checked["sha256"])
    result = load(audit.get("result"), "initial measurement result")
    require_prefix_measurement_source_binding(recipe, measurement=result)
    initial = load(result.get("initialization_result"), "saved fresh initialization")
    measurement = load(result.get("measurement_binding_result"), "frozen measurement coordinates")
    receipt = load(audit.get("receipt"), "initial native terminal receipt")
    dynamic_prefix = (_require_dynamic_m5_native_prefix(recipe) if _dynamic_m5_native_design(recipe) is not None else None)
    if dynamic_prefix is not None:
        _require_dynamic_initial_artifacts(recipe, dynamic_prefix, initial, measurement)
    expected = initial.get("online_model_state_sha256")
    repo = Path(__file__).resolve().parents[2]
    model_source = "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py"
    if initial.get("source_bindings", {}).get(model_source) != file_sha256(repo / model_source):
        raise RuntimeError("NATIVE_PREFIX_LEARNING_INITIAL_MODEL_SOURCE_CHANGED")
    entry_identity = None
    if _observed_entry_in(recipe):
        from gx1.contracts.entry_observed_market_horizons_v1 import observed_entry_identity, require_observed_entry_scope
        entry_identity = observed_entry_identity(require_observed_entry_scope(recipe))
    if (result.get("entry_observed_market") != entry_identity
            or (dynamic_prefix is not None and result.get("dynamic_m5_loss_scales") != initial.get("dynamic_m5_loss_scales"))):
        raise RuntimeError("NATIVE_PREFIX_INITIAL_ENTRY_TARGET_IDENTITY_MISMATCH")
    initial_roles = set(result.get("observations", {}))
    allowed_roles = ({"train"}, {"train", "control"}) if recipe.get("chronological_train_only_measurement") is True else ({"train", "control"},)
    if (audit.get("schema_version") != "gx1_native_prefix_initial_measurement_audit_v1"
            or audit.get("decision") != "FROZEN_INITIAL_MEASUREMENT_VERIFIED_NO_LEARNING_MEASURED"
            or audit.get("optimizer_steps") != 0 or audit.get("test_data_used") is not False
            or audit.get("fresh_model_optimizer_ema_scheduler_exactly_preserved") is not True
            or audit.get("saved_cpu_python_numpy_rng_exactly_preserved") is not True
            or result.get("schema_version") != "gx1_native_prefix_initial_measurement_v1"
            or result.get("decision") != "FROZEN_INITIAL_TARGETS_AND_PREDICTIONS_READY_NO_LEARNING_MEASURED"
            or result.get("optimizer_steps") != 0 or result.get("teacher_refreshed") is not False
            or result.get("economic_rollout") is not False or result.get("test_data_used") is not False
            or initial.get("chronological_prefix") != recipe.get("chronological_prefix")
            or initial.get("files") != recipe.get("files")
            or not isinstance(expected, str) or len(expected) != 64
            or result.get("model_state_sha256") != expected or result.get("target_model_state_sha256") != expected
            or receipt.get("guard_decision") != "PASS" or receipt.get("outcome") != "RESUMABLE"
            or receipt.get("trainer_guard_exit_code") != 0 or receipt.get("progress_observer_exit_code") != 0
            or receipt.get("test_data_used") is not False
            or initial_roles not in allowed_roles):
        raise RuntimeError("NATIVE_PREFIX_LEARNING_INITIAL_MEASUREMENT_INVALID")
    if dynamic_prefix is not None:
        if (result.get("target_model_role") != "frozen_constructor_anchor_no_forward"
                or result.get("exit_evaluation") is not False or result.get("native_launch_authorized") is not False
                or result.get("model_functions") != initial.get("model_functions")
                or type(result.get("optimizer_steps")) is not int or result["optimizer_steps"] != 0
                or initial_roles != {"train", "control"}):
            raise RuntimeError("NATIVE_DYNAMIC_M5_INITIAL_MEASUREMENT_REQUIRED")
    require_binding(initial["initial_state"], label="saved fresh state", verify_file=True)
    observed = {}
    roles = ("train",) if recipe.get("chronological_train_only_measurement") is True else ("train", "control")
    for role in roles:
        observation = load(result["observations"][role], "initial " + role + " observations")
        observed[role] = observation
        cohort = observation.get("cohort", {})
        if (observation.get("role") != role or observation.get("optimizer_steps") != 0
                or observation.get("entry_observed_market") != entry_identity
                or observation.get("model_state_sha256") != expected
                or observation.get("target_model_state_sha256") != expected
                or observation.get("test_data_used") is not False
                or cohort.get("plan") != recipe["chronological_prefix"]["design"]
                or cohort.get("measurement_coordinates") != measurement.get("coordinate_result")
                or cohort.get("measurement_role") != role
                or len(cohort.get("entry_row_indices", [])) != 256):
            raise RuntimeError("NATIVE_PREFIX_LEARNING_INITIAL_OBSERVATION_INVALID")
        if "native_coordinates" in recipe["chronological_prefix"]:
            from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_measurement_cohort
            if cohort != build_chronological_measurement_cohort(
                    recipe["chronological_prefix"]["design"], measurement["coordinate_result"], role=role):
                raise RuntimeError("NATIVE_PREFIX_LEARNING_PHYSICAL_COHORT_MISMATCH")
        if dynamic_prefix is not None:
            source = dynamic_prefix["physical_preprocessing"]["physical_sources"]["train" if role == "train" else "val"]
            if (cohort.get("parent_entry_parquet") != source["parquet"]
                    or cohort.get("parent_entry_manifest") != source["manifest"]):
                raise RuntimeError("NATIVE_DYNAMIC_M5_INITIAL_OBSERVATION_SOURCE_MISMATCH")
            statistics = observation.get("diagnostics", {}).get("dynamic_curve_statistics", {})
            if (statistics.get("dynamic_m5_loss_scales") != initial.get("dynamic_m5_loss_scales")
                    or statistics.get("dynamic_loss_residual_units") != "TRAIN_target_RMS_units"):
                raise RuntimeError("NATIVE_DYNAMIC_M5_INITIAL_OBSERVATION_LOSS_SCALE_MISMATCH")
            _require_dynamic_m5_observation(observation, identity=entry_identity, cohort=cohort)
    derived = {}
    if "chronological_entry_baseline" in recipe:
        entry_binding, entry_baseline = _require_derived_entry_baseline(
            recipe, initial_result=result, train_observation=observed["train"])
        derived = {"entry_baseline":entry_baseline, "entry_baseline_result":entry_binding}
    return {**derived, "artifacts": {"initialization_result": result["initialization_result"],
                          "measurement_binding_result": result["measurement_binding_result"],
                          "initial_measurement_audit": binding, "initial_measurement_result": audit["result"]},
            "initialization": initial, "measurement": measurement, "initial_measurement": result}



def _require_prefix_resume_equivalence(recipe, *, binding, plan):
    """Replay only the preserved192..256 suffix; this never admits new learning."""
    import numpy as np
    from datetime import datetime
    def load(value, label):
        checked = require_binding(value, label=label, verify_file=True)
        return checked, read_bound_json(Path(checked["path"]), checked["sha256"])
    fixed = {"schema_version": PREFIX_RESUME_EQUIVALENCE_SCHEMA,
        "from_optimizer_steps":192, "stop_after_optimizer_steps":256, "replayed_optimizer_steps":64,
        "maximum_trained_entry_rows":4096, "maximum_additional_entry_rows":0,
        "maximum_replayed_entry_rows":1024, "max_invocations":1, "teacher_refresh_allowed":False,
        "control_forwards":0, "full_epoch_allowed":False, "full_val_allowed":False,
        "test_data_used":False, "automatic_extension_allowed":False, "physical_reboot_required":True,
        "learning_admission":False, "require_bitwise_state_equivalence":True, "run_authority_created":False}
    forbidden = ("candidate_resume_origin", "native_calibration", "frozen_readout_evaluation",
                 "chronological_initial_measurement", "chronological_train_only_measurement",
                 "chronological_entry_baseline", "entry_gradient_diagnostic",
                 "frozen_train_policy_evaluation", "frozen_entry_selector_probe")
    if (any(type(plan.get(k)) is not type(v) or plan.get(k) != v for k,v in fixed.items())
            or any(k in recipe for k in forbidden)
            or not _observed_entry_in(recipe)
            or "chronological_learning_measurement" not in recipe
            or recipe.get("out_bundle_dir") != plan.get("out_bundle_dir")
            or plan.get("comparison_state_fields") != ["model_state","target_model_state","optimizer_state",
                "weight_ema_state","lr_scheduler_state","rng_state","epoch_order","training_progress"]
            or plan.get("comparison_cursor_fields") != ["phase","epoch_index","next_batch_offset",
                                                       "global_optimizer_steps","complete"]
            or plan.get("only_allowed_state_metadata_difference") != ["session_contract_sha256"]):
        raise RuntimeError("NATIVE_PREFIX_RESUME_EQUIVALENCE_SCOPE_INVALID")
    _, review = load(plan.get("origin_review"), "resume equivalence origin review")
    contract_binding, contract = load(plan.get("origin_contract"), "resume equivalence contract")
    _, before = load(plan.get("origin_recipe"), "resume equivalence recipe")
    _, audit = load(plan.get("preflight"), "resume equivalence preflight")
    pointer_binding, pointer = load(plan.get("reference_pointer"), "resume equivalence reference pointer")
    reference = require_binding(plan.get("reference_state"), label="resume equivalence reference state", verify_file=True)
    origin = require_binding(plan.get("origin_state"), label="resume equivalence origin state", verify_file=True)
    directory = Path(pointer_binding["path"]).parent
    if (review.get("schema_version") != "gx1_native_entry_smoke_completion_review_v1"
            or review.get("technical_native_cycle_passed") is not True
            or review.get("source_unchanged") is not True
            or review.get("optimizer_steps") != 256 or review.get("test_data_used") is not False
            or review.get("training_expansion_authorized") is not False
            or review.get("entry_exit_target_independence_verified") is not True
            or review.get("target_model_exactly_preserved") is not True
            or review.get("training_pointer") != pointer_binding
            or review.get("training_state") != reference or review.get("recipe") != plan["origin_recipe"]
            or pointer.get("global_optimizer_steps") != 256 or pointer.get("next_batch_offset") != 256
            or pointer.get("epoch_index") != 0 or pointer.get("phase") != "train" or pointer.get("complete") is not False
            or type(pointer.get("slot")) is not int or pointer["slot"] not in (0,1)
            or pointer.get("session_contract_sha256") != contract_binding["sha256"]
            or pointer.get("state_sha256") != reference["sha256"]
            or Path(reference["path"]) != directory / f"candidate_training_state_slot_{pointer['slot']}.pt"
            or Path(origin["path"]) != directory / f"candidate_training_state_slot_{1-pointer['slot']}.pt"
            or contract.get("out_bundle_dir") != before.get("out_bundle_dir")
            or before.get("out_bundle_dir") == recipe.get("out_bundle_dir")
            or before.get("run_id") == recipe.get("run_id")
            or contract.get("chronological_prefix",{}).get("artifacts") != recipe.get("chronological_prefix")):
        raise RuntimeError("NATIVE_PREFIX_RESUME_EQUIVALENCE_ORIGIN_INVALID")
    mutable = {"source_commit","source_bindings","source_bindings_sha256","run_id","out_bundle_dir",
               "recipe_sha256","next_run_policy","chronological_learning_continuation"}
    if {k:v for k,v in before.items() if k not in mutable} != {k:v for k,v in recipe.items() if k not in mutable}:
        raise RuntimeError("NATIVE_PREFIX_RESUME_EQUIVALENCE_RECIPE_CHANGED")
    allowed = {"wrapper", "python:gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
               "python:gx1/scripts/run_unified_exit_random_access_full_train_v1.py",
               "python:gx1/contracts/unified_exit_native_candidate_campaign_v1.py"}
    old_sources,new_sources = before["source_bindings"],recipe["source_bindings"]
    if (old_sources.keys() != new_sources.keys()
            or any(old_sources[k] != new_sources[k] for k in old_sources if k not in allowed)):
        raise RuntimeError("NATIVE_PREFIX_RESUME_EQUIVALENCE_MODEL_OR_TARGET_SOURCE_CHANGED")
    prefix=contract["chronological_prefix"]
    expected_audit = {"schema_version":"gx1_prefix_resume_equivalence_preflight_v1",
        "origin_contract":contract_binding,"origin_state":origin,"reference_state":reference,
        "reference_pointer":pointer_binding,"origin_recipe":plan["origin_recipe"],"origin_review":plan["origin_review"],
        "from_optimizer_steps":192,"stop_after_optimizer_steps":256,"replayed_optimizer_steps":64,
        "previously_trained_entry_rows":4096,"new_unique_entry_rows":0,"replayed_entry_rows":1024,
        "parent_order":prefix["epoch0_parent_order"],"train_rows":prefix["train_parent_rows"],
        "control_rows":prefix["control_parent_rows"],"entry_order_exact":True,"control_overlap":0,
        "target_model_state_sha256":review["target_model_state_sha256"],
        "original_reference_model_state_sha256":review["model_state_sha256"],
        "physical_sources":prefix["physical_sources"],"native_coordinates":prefix["native_coordinates"],
        "torch_cuda_rng_preserved":True,"new_optimizer_steps":0,"new_model_forwards":0,"test_data_used":False}
    if any(type(audit.get(k)) is not type(v) or audit.get(k) != v for k,v in expected_audit.items()):
        raise RuntimeError("NATIVE_PREFIX_RESUME_EQUIVALENCE_PREFLIGHT_INVALID")
    def rows(key):
        b=require_binding(audit[key],label="resume equivalence rows",verify_file=True)
        a=np.load(b["path"],allow_pickle=False)
        if a.ndim!=1 or a.dtype!=np.dtype("int64"):
            raise RuntimeError("NATIVE_PREFIX_RESUME_EQUIVALENCE_ORDER_INVALID")
        return a
    order,parents,control=(rows(k) for k in ("parent_order","train_rows","control_rows"))
    physical=prefix["physical_sources"]
    _,coordinates=load(prefix["native_coordinates"],"resume equivalence physical coordinates")
    import hashlib
    selected=order[192*16:256*16]
    if (len(order)<4096 or not np.array_equal(np.sort(order),parents)
            or len(np.unique(order[:4096]))!=4096 or len(selected)!=1024
            or hashlib.sha256(selected.astype("<i8",copy=False).tobytes()).hexdigest()!=audit["replayed_parent_rows_sha256"]
            or coordinates.get("train_source")!=physical["train"]["parquet"]
            or coordinates.get("control_source")!=physical["val"]["parquet"]
            or physical["train"]["parquet"]["path"]==physical["val"]["parquet"]["path"]
            or len(order)!=physical["train"]["physical_rows"] or len(control)!=256
            or control.min()<0 or control.max()>=physical["val"]["physical_rows"]
            or datetime.fromisoformat(physical["train"]["last_entry_utc"])>=datetime.fromisoformat(physical["val"]["first_entry_utc"])):
        raise RuntimeError("NATIVE_PREFIX_RESUME_EQUIVALENCE_ORDER_OR_SOURCE_INVALID")
    return {"plan_binding":binding,"plan":plan,"origin_review":review,"origin_contract":contract,
            "origin_contract_binding":contract_binding,"origin_pointer":pointer_binding,
            "origin_recipe":before,"resume_equivalence":True}


def require_chronological_continuation(recipe):
    """One separately bound256-to512 continuation; preserve the completed origin."""
    import numpy as np
    value = recipe.get("chronological_learning_continuation")
    if value is None: return None
    def load(binding, label):
        checked = require_binding(binding, label=label, verify_file=True)
        return checked, read_bound_json(Path(checked["path"]), checked["sha256"])
    binding, plan = load(value, "prefix continuation plan")
    if plan.get("schema_version") == PREFIX_RESUME_EQUIVALENCE_SCHEMA:
        return _require_prefix_resume_equivalence(recipe, binding=binding, plan=plan)
    fixed = {"schema_version":"gx1_prefix_learning_continuation_plan_v1",
        "from_optimizer_steps":256, "stop_after_optimizer_steps":512, "additional_optimizer_steps":256,
        "maximum_trained_entry_rows":8192, "maximum_additional_entry_rows":4096, "max_invocations":1,
        "teacher_refresh_allowed":False, "control_forwards":0, "full_epoch_allowed":False,
        "full_val_allowed":False, "test_data_used":False, "automatic_extension_allowed":False}
    if (any(type(plan.get(k)) is not type(v) or plan.get(k) != v for k,v in fixed.items())
            or recipe.get("chronological_train_only_measurement") is not True
            or "chronological_learning_measurement" not in recipe
            or any(k in recipe for k in ("candidate_resume_origin", "entry_gradient_diagnostic",
                                         "chronological_initial_measurement", "native_calibration", "frozen_readout_evaluation"))):
        raise RuntimeError("NATIVE_PREFIX_CONTINUATION_SCOPE_INVALID")
    _, review = load(plan.get("origin_review"), "continuation origin review")
    contract_binding, contract = load(plan.get("origin_contract"), "continuation origin contract")
    _, before = load(plan.get("origin_recipe"), "continuation origin recipe")
    _, cause = load(plan.get("cause_review"), "joint update review")
    _, audit = load(plan.get("order_audit"), "continuation order audit")
    pointer_binding, pointer = load(review.get("training_pointer"), "continuation origin pointer")
    if (review.get("schema_version") != "gx1_entry_fuse_normalization_fixed256_train_review_v1"
            or review.get("optimizer_steps") != 256 or review.get("same_corrected_targets_masks_cohorts_verified") is not True
            or review.get("original_target_model_preserved") is not True
            or cause.get("schema_version") != "gx1_joint_update_review_v2"
            or cause.get("decision") != "REJECT_AUXILIARY_CONFLICT_AS_CAUSE_ON_MEASURED_TRAIN16"
            or cause.get("training_state") != review.get("training_state")
            or cause.get("training_pointer") != pointer_binding
            or cause.get("scope_consumed") is not True or cause.get("resume_authorized") is not False
            or pointer.get("global_optimizer_steps") != 256 or pointer.get("next_batch_offset") != 256
            or pointer.get("epoch_index") != 0 or pointer.get("phase") != "train" or pointer.get("complete") is not False
            or pointer.get("session_contract_sha256") != contract_binding["sha256"]
            or pointer.get("state_sha256") != review["training_state"]["sha256"]
            or contract.get("out_bundle_dir") != before.get("out_bundle_dir")
            or before.get("out_bundle_dir") == recipe.get("out_bundle_dir")
            or contract.get("chronological_prefix", {}).get("artifacts") != recipe.get("chronological_prefix")):
        raise RuntimeError("NATIVE_PREFIX_CONTINUATION_ORIGIN_INVALID")
    mutable = {"source_commit", "source_bindings", "source_bindings_sha256", "run_id", "out_bundle_dir",
               "recipe_sha256", "next_run_policy", "chronological_learning_continuation"}
    if {k:v for k,v in before.items() if k not in mutable} != {k:v for k,v in recipe.items() if k not in mutable}:
        raise RuntimeError("NATIVE_PREFIX_CONTINUATION_RECIPE_CHANGED")
    allowed_sources = {"wrapper", "python:gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
                       "python:gx1/scripts/run_unified_exit_random_access_full_train_v1.py",
                       "python:gx1/contracts/unified_exit_native_candidate_campaign_v1.py"}
    old_sources, new_sources = before["source_bindings"], recipe["source_bindings"]
    if (old_sources.keys() != new_sources.keys()
            or any(old_sources[k] != new_sources[k] for k in old_sources if k not in allowed_sources)):
        raise RuntimeError("NATIVE_PREFIX_CONTINUATION_MODEL_OR_TARGET_SOURCE_CHANGED")
    prefix = contract["chronological_prefix"]
    audit_fixed = {"schema_version":"gx1_prefix_continuation_order_audit_v1", "origin_optimizer_steps":256,
        "stop_after_optimizer_steps":512, "batch_size":16, "previous_entries":4096, "additional_entries":4096,
        "total_entries":8192, "all_unique":True, "all_within_existing_prefix":True, "control_overlap":0,
        "new_model_forwards":0, "new_targets":0, "new_fits":0, "test_data_used":False}
    if (any(type(audit.get(k)) is not type(v) or audit.get(k) != v for k,v in audit_fixed.items())
            or audit.get("parent_order") != prefix["epoch0_parent_order"]
            or audit.get("train_rows") != prefix["train_parent_rows"]
            or audit.get("control_rows") != prefix["control_parent_rows"]):
        raise RuntimeError("NATIVE_PREFIX_CONTINUATION_ORDER_AUDIT_INVALID")
    def rows(key):
        b = require_binding(audit[key], label="continuation rows", verify_file=True)
        a = np.load(b["path"], allow_pickle=False)
        if a.ndim != 1 or a.dtype != np.dtype("int64"): raise RuntimeError("NATIVE_PREFIX_CONTINUATION_ORDER_INVALID")
        return a
    order, parents, control, selected = (rows(k) for k in ("parent_order", "train_rows", "control_rows", "next_rows"))
    if (len(order) <= 8192 or not np.array_equal(np.sort(order), parents)
            or len(np.unique(order[:8192])) != 8192 or np.intersect1d(order, control).size
            or not np.array_equal(order[4096:8192], selected)):
        raise RuntimeError("NATIVE_PREFIX_CONTINUATION_ORDER_INVALID")
    return {"plan_binding":binding, "plan":plan, "origin_review":review,
            "origin_contract":contract, "origin_contract_binding":contract_binding,
            "origin_pointer":pointer_binding}


def require_m5_single_epoch_run(recipe, *, invocation_number=None, execution_budget=None):
    """One actual M5-eligible TRAIN epoch. Read policy; never create admission."""
    prefix = require_chronological_prefix_recipe(recipe)
    single = prefix.get("entry_m5_path_single_epoch")
    if (single is None or not _observed_entry_in(recipe)
            or "chronological_learning_measurement" not in recipe
            or any(k in recipe for k in ("chronological_initial_measurement", "entry_learning_study",
                "chronological_learning_continuation", "chronological_entry_baseline", "native_calibration",
                "frozen_readout_evaluation", "candidate_resume_origin", "entry_gradient_diagnostic",
                "frozen_train_policy_evaluation", "frozen_entry_selector_probe"))):
        raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_MIXED_OR_UNMEASURED_SCOPE")
    from gx1.contracts.entry_observed_market_horizons_v1 import require_observed_entry_scope
    observed = require_observed_entry_scope(recipe)
    if observed is None or observed["training_phase"] != single["witness"]["training_phase"]:
        raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_TARGET_MISMATCH")
    measured = require_chronological_learning_measurement(recipe)
    repo = Path(__file__).resolve().parents[2]
    binding = require_binding(recipe.get("next_run_policy"), label="single epoch policy", verify_file=True)
    if Path(binding["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(binding["path"]), binding["sha256"])
    _require_native_profile_and_economics(recipe, policy, repo)
    scope = policy.get("entry_m5_path_single_epoch_run")
    windows = scope.get("max_invocations") if isinstance(scope, Mapping) else None
    expected = {"population_witness": single["binding"], "chronological_prefix": prefix["artifacts"],
        "chronological_learning_measurement": measured["artifacts"]["initial_measurement_audit"],
        "run_id": recipe.get("run_id"), "out_bundle_dir": recipe.get("out_bundle_dir"),
        "source_bindings_sha256": recipe.get("source_bindings_sha256"), "max_invocations": windows,
        "optimizer_steps": single["optimizer_steps"], "maximum_trained_entry_rows": single["train_rows"],
        "batch_size": single["witness"]["batch_size"], "drop_last": False, "epochs": 1,
        "final_model_variant": "ONLINE", "teacher_refresh_allowed": False,
        "full_epoch_training_allowed": True, "full_val_allowed": False, "test_data_used": False,
        "automatic_extension_allowed": False}
    _expected_observed_entry_binding(recipe, expected)
    if "chronological_train_only_measurement" in recipe:
        if recipe["chronological_train_only_measurement"] is not True:
            raise RuntimeError("NATIVE_PREFIX_TRAIN_ONLY_MEASUREMENT_INVALID")
        expected["chronological_train_only_measurement"] = True
    dynamic = _dynamic_m5_native_design(recipe) is not None
    if ((policy.get("training_enabled") is not True if dynamic else policy.get("training_enabled") is not False)
            or (dynamic and policy.get("full_epoch_training_allowed") is not True) or scope != expected
            or canonical_sha256(scope) != canonical_sha256(expected)
            or type(windows) is not int or windows < 1
            or any(k in policy for k in ("chronological_learning_run", "native_learning_calibration",
                "frozen_readout_evaluation", "chronological_initial_measurement"))):
        raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_NOT_AUTHORIZED")
    if invocation_number is not None and (type(invocation_number) is not int or not 1 <= invocation_number <= windows):
        raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_INVOCATION_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != single["optimizer_steps"]
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_BUDGET_INVALID")
    return {**prefix, "measurement_scope": measured, "optimizer_step_ceiling": single["optimizer_steps"]}


def require_chronological_prefix_run(recipe, *, invocation_number=None, execution_budget=None):
    """Admit only the explicitly bound single experiment through native guards."""
    prefix = require_chronological_prefix_recipe(recipe)
    repo = Path(__file__).resolve().parents[2]
    binding = require_binding(recipe.get("next_run_policy"), label="native prefix policy", verify_file=True)
    if Path(binding["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(binding["path"]), binding["sha256"])
    _require_native_profile_and_economics(recipe, policy, repo)
    scope = policy.get("chronological_learning_run")
    if not isinstance(scope, Mapping):
        raise RuntimeError("NATIVE_PREFIX_RUN_NOT_AUTHORIZED")
    continuation = require_chronological_continuation(recipe)
    ceiling = continuation["plan"]["stop_after_optimizer_steps"] if continuation else 256
    windows = scope.get("max_invocations")
    expected = {"chronological_prefix":prefix["artifacts"], "run_id":recipe.get("run_id"),
                "out_bundle_dir":recipe.get("out_bundle_dir"), "source_bindings_sha256":recipe.get("source_bindings_sha256"),
                "optimizer_steps":ceiling, "maximum_trained_entry_rows":ceiling * 16, "max_invocations":windows,
                "final_model_variant":"ONLINE", "teacher_refresh_allowed":False,
                "full_epoch_training_allowed":False, "full_val_allowed":False, "test_data_used":False}
    _expected_observed_entry_binding(recipe, expected)
    if continuation:
        expected["chronological_learning_continuation"] = continuation["plan_binding"]
        if windows != 1: raise RuntimeError("NATIVE_PREFIX_CONTINUATION_INVOCATION_INVALID")
    if "chronological_learning_measurement" in recipe:
        measured = require_chronological_learning_measurement(recipe)
        expected["chronological_learning_measurement"] = measured["artifacts"]["initial_measurement_audit"]
        if "chronological_entry_baseline" in recipe:
            expected["chronological_entry_baseline"] = measured["entry_baseline_result"]
    if "chronological_train_only_measurement" in recipe:
        if (recipe["chronological_train_only_measurement"] is not True
                or "chronological_learning_measurement" not in recipe):
            raise RuntimeError("NATIVE_PREFIX_TRAIN_ONLY_MEASUREMENT_INVALID")
        expected["chronological_train_only_measurement"] = True
    dynamic = _dynamic_m5_native_design(recipe) is not None
    if ((policy.get("training_enabled") is not True if dynamic else policy.get("training_enabled") is not False)
            or scope != expected
            or type(windows) is not int or windows < 1
            or type(scope.get("optimizer_steps")) is not int
            or type(scope.get("maximum_trained_entry_rows")) is not int
            or any(scope.get(key) is not False for key in ("teacher_refresh_allowed", "full_epoch_training_allowed", "full_val_allowed", "test_data_used"))
            or any(key in policy for key in ("native_learning_calibration", "frozen_readout_evaluation", "chronological_initial_measurement"))):
        raise RuntimeError("NATIVE_PREFIX_RUN_NOT_AUTHORIZED")
    if invocation_number is not None and (type(invocation_number) is not int or not 1 <= invocation_number <= windows):
        raise RuntimeError("NATIVE_PREFIX_INVOCATION_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != ceiling
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("NATIVE_PREFIX_BUDGET_INVALID")
    return {**prefix, "optimizer_step_ceiling":ceiling, "continuation":continuation}


def require_fresh_physical_campaign_sampler(recipe: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    """Bind current measured CPU sampler provenance, not historical GPU capacity.

    Actual GPU capacity is measured by the guarded, finite native invocation.
    This route grants no full epoch, full VAL, or inherited model authority.
    """
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    prefix_binding = recipe.get("chronological_prefix", {})
    if isinstance(prefix_binding, Mapping) and "design" in prefix_binding:
        design_binding = require_binding(prefix_binding["design"], label="fresh campaign design", verify_file=True)
        design = read_bound_json(Path(design_binding["path"]), design_binding["sha256"])
        if design.get("scope", {}).get("entry_only_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
            prefix = _require_dynamic_m5_native_prefix(recipe)
            binding = prefix["entry_m5_path_single_epoch"]["binding"]
            if "selected_sampler" in recipe.get("files", {}):
                raise RuntimeError("NATIVE_DYNAMIC_M5_EXIT_SAMPLER_FORBIDDEN")
            return binding, binding["sha256"]  # population/order authority; no GPU benchmark claim
    prefix = require_chronological_prefix_recipe(recipe)
    coordinates = prefix.get("physical_coordinates")
    # The materializer and campaign invocation loop own finite scope admission.
    # Repeating it here rebuilds both full physical measurement indexes.
    if coordinates is None:
        raise RuntimeError("NATIVE_FRESH_CAMPAIGN_PHYSICAL_COORDINATES_REQUIRED")
    binding = coordinates["selected_sampler_binding"]
    selected = coordinates["selected_sampler"]
    if recipe.get("files", {}).get("selected_sampler") != binding or selected["batch_size"] != 16:
        raise RuntimeError("NATIVE_FRESH_CAMPAIGN_SAMPLER_MISMATCH")
    return binding, selected["artifact_sha256"]


def require_dynamic_m5_execution_profile(recipe: Mapping[str, Any]) -> dict[str, str] | None:
    """Join one explicit compact model function to its immutable constructor."""
    from gx1.features.entry_specialist_feature_groups_v1 import (
        require_compact_entry_execution_profile_reference,
    )
    binding = require_binding(recipe.get("files", {}).get("source_bundle_metadata"),
                              label="Entry execution-profile constructor", verify_file=True)
    metadata = read_bound_json(Path(binding["path"]), binding["sha256"])
    if not isinstance(metadata, Mapping):
        raise RuntimeError("NATIVE_ENTRY_EXECUTION_PROFILE_METADATA_INVALID")
    declared = recipe.get("entry_execution_profile")
    if metadata.get("entry_execution_profile") != declared:
        raise RuntimeError("NATIVE_ENTRY_EXECUTION_PROFILE_RECIPE_CONSTRUCTOR_MISMATCH")
    reference = require_compact_entry_execution_profile_reference(
        declared, context="NATIVE_ENTRY_EXECUTION_PROFILE")
    if ("entry_execution_profile" in recipe or "entry_execution_profile" in metadata) and _dynamic_m5_native_design(recipe) is None:
        raise RuntimeError("NATIVE_ENTRY_EXECUTION_PROFILE_DYNAMIC_ENTRY_REQUIRED")
    return reference


def require_native_recipe_metadata(
    binding: Mapping[str, str], *, source_repo: Path, source_commit: str,
) -> tuple[dict[str, Any], int]:
    """Check campaign metadata without constructing datasets or model tensors.

    The guarded native entry point still owns full source/data/model validation.
    """
    checked = require_binding(binding, label="native recipe")
    recipe = read_bound_json(Path(checked["path"]), checked["sha256"])
    controls = recipe.get("trainer_cli", {})
    dynamic_design = _dynamic_m5_native_design(recipe)
    if dynamic_design is None and "dynamic_m5_loss_scales" in recipe:
        raise RuntimeError("NATIVE_DYNAMIC_M5_LOSS_SCALE_PHASE_INVALID")
    if dynamic_design is not None:
        import math
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        from gx1.contracts.entry_candidate_checkpoint_policy_v1 import DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
        from gx1.contracts.entry_model_native_train_recipe_v1 import require_model_native_recipe_env
        prefix = _require_dynamic_m5_native_prefix(recipe)
        _dynamic_m5_compute_profile(recipe)
        require_dynamic_m5_execution_profile(recipe)
        files = recipe.get("files", {})
        expected_files = {"entry_train_parquet", "entry_train_manifest", "entry_val_parquet", "entry_val_manifest",
            "m5_prebuilt", "source_bundle_metadata", "sequence_source_audit", "val_sequence_source_audit",
            "entry_only_base_normalization", "entry_measurement_coordinates", "train_cost_authority"}
        fixed = {"epochs": 1, "batch_size": 16, "num_workers": 0, "grad_accum_steps": 1,
            "early_stopping_patience": 5, "early_stopping_min_delta": 0.0, "minimum_epochs_before_stop": 1,
            "save_top_k": 0, "precision_policy": "deterministic_fp32", "learning_rate": 0.0001,
            "weight_decay": 0.0001, "checkpoint_monitor": DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR}
        if (recipe.get("schema_version") != "gx1_unified_exit_random_access_full_train_recipe_v1"
                or recipe.get("profile") != "candidate" or recipe.get("test_data_used") is not False
                or recipe.get("source_repo") != str(source_repo) or recipe.get("source_commit") != source_commit
                or recipe.get("recipe_sha256") != native_sha256({k:v for k,v in recipe.items() if k != "recipe_sha256"})
                or not isinstance(files, Mapping) or set(files) != expected_files
                or set(controls) != set(fixed) | {"seed", "grad_clip_norm"}
                or any(type(controls.get(key)) is not type(value) or controls[key] != value for key,value in fixed.items())
                or type(controls.get("grad_clip_norm")) is not float or not math.isfinite(controls["grad_clip_norm"])
                or controls["grad_clip_norm"] <= 0):
            raise RuntimeError("NATIVE_CANDIDATE_DYNAMIC_RECIPE_METADATA_INVALID")
        require_model_native_recipe_env(recipe.get("recipe_env"), entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
        require_prefix_measurement_source_binding(recipe)
        checked_files = {key: require_binding(value, label="dynamic native input", verify_file=True) for key,value in files.items()}
        physical = prefix["physical_preprocessing"]
        for split in ("train", "val"):
            if any(checked_files[f"entry_{split}_{kind}"] != physical["physical_sources"][split][kind]
                   for kind in ("parquet", "manifest")):
                raise RuntimeError("NATIVE_CANDIDATE_DYNAMIC_SOURCE_MISMATCH")
        normalization = read_bound_json(Path(physical["entry_only_base_normalization"]["path"]), physical["entry_only_base_normalization"]["sha256"])
        lineage = normalization["contract"]["lineage"]
        if (checked_files["entry_only_base_normalization"] != physical["entry_only_base_normalization"]
                or checked_files["sequence_source_audit"] != physical["sequence_source_audits"]["train"]
                or checked_files["val_sequence_source_audit"] != physical["sequence_source_audits"]["val"]
                or checked_files["m5_prebuilt"] != {"path":lineage["m5_prebuilt_path"], "sha256":lineage["m5_prebuilt_sha256"]}
                or recipe.get("dataset_run_id") != lineage["dataset_run_id"]):
            raise RuntimeError("NATIVE_CANDIDATE_DYNAMIC_NORMALIZATION_SOURCE_MISMATCH")
        measurement_binding = checked_files["entry_measurement_coordinates"]
        measurement = read_bound_json(Path(measurement_binding["path"]), measurement_binding["sha256"])
        if (measurement.get("chronological_prefix") != prefix["artifacts"]
                or measurement.get("population_witness") != prefix["entry_m5_path_single_epoch"]["binding"]
                or measurement.get("native_sampler_contract") != prefix["physical_coordinates"]["native_sampler_contract"]
                or measurement.get("schema_version") != "gx1_dynamic_m5_native_coordinate_publication_v1"
                or measurement.get("native_coordinates_complete") is not True
                or measurement.get("entry_measurement_complete") is not False
                or measurement.get("native_launch_admitted") is not False or measurement.get("test_data_used") is not False):
            raise RuntimeError("NATIVE_CANDIDATE_DYNAMIC_COORDINATE_MISMATCH")
        return recipe, prefix["entry_m5_path_single_epoch"]["train_rows"]
    economics_binding = require_binding(recipe.get("files", {}).get("economics_readiness"), label="native economics readiness")
    economics = read_bound_json(Path(economics_binding["path"]), economics_binding["sha256"])
    monitor = native_checkpoint_monitor(economics["economics_objective_contract"])
    if (
        recipe.get("schema_version") != "gx1_unified_exit_random_access_full_train_recipe_v1"
        or recipe.get("profile") != "candidate" or recipe.get("test_data_used") is not False
        or recipe.get("source_repo") != str(source_repo)
        or recipe.get("source_commit") != source_commit
        or recipe.get("recipe_sha256") != native_sha256({k: v for k, v in recipe.items() if k != "recipe_sha256"})
        or any(recipe.get("val_limits", {}).get(k) != v for k, v in
               {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
                "max_wall_seconds": 10800, "progress_interval_forwards": 64}.items())
        or controls.get("epochs") != (1 if "entry_m5_path_single_epoch" in recipe else 30) or controls.get("batch_size") != 16
        or controls.get("early_stopping_patience") != 5
        or controls.get("checkpoint_monitor") != monitor
    ):
        raise RuntimeError("NATIVE_CANDIDATE_RECIPE_METADATA_INVALID")
    root_binding = require_binding(recipe["files"]["random_access_root"], label="native index root")
    root = read_bound_json(Path(root_binding["path"]), root_binding["sha256"])
    if (
        root.get("decision") != "PASS" or root.get("allowed_splits") != ["train", "val"]
        or root.get("test_accessed") is not False
        or root.get("root_sha256") != native_sha256({k: v for k, v in root.items() if k != "root_sha256"})
    ):
        raise RuntimeError("NATIVE_CANDIDATE_INDEX_ROOT_METADATA_INVALID")
    latest_year = root.get("schema_version") == LATEST_YEAR_ROOT_SCHEMA_VERSION
    if latest_year:
        require_random_access_index_root(root)
    split = root["splits"]["train"]
    path = Path(split["manifest_path"])
    manifest = read_bound_json(path, file_sha256(path))
    count = manifest.get("entry_row_count")
    if (
        manifest.get("manifest_sha256") != split["manifest_sha256"]
        or manifest.get("manifest_sha256") != native_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        or type(count) is not int or count < 1
        or (not latest_year and manifest.get("parent_entry_source_rows") != count)
        or (latest_year and split.get("entry_row_count") != count)
    ):
        raise RuntimeError("NATIVE_CANDIDATE_FULL_POPULATION_METADATA_INVALID")
    if latest_year:
        count = root["latest_year_population"]["splits"]["train"]["selected_entry_row_count"]
    if "chronological_prefix" in recipe:
        prefix = require_chronological_prefix_recipe(recipe)
        if prefix["train_rows"] > count or latest_year:
            raise RuntimeError("NATIVE_PREFIX_FULL_PHYSICAL_INDEX_REQUIRED")
        count = (prefix["entry_m5_path_single_epoch"]["train_rows"]
                 if "entry_m5_path_single_epoch" in prefix else prefix["train_rows"])
    return recipe, count


def require_native_calibration_run(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """The two finite native arms compare 32 uninterrupted versus 16+16 steps."""
    if "native_calibration" not in recipe:
        return None
    value = recipe["native_calibration"]
    if (not isinstance(value, Mapping) or set(value) != {
            "schema_version", "arm", "report_only_val"}
            or value["schema_version"] != "gx1_native_learning_calibration_run_v1"
            or type(value["arm"]) is not str or value["arm"] not in {"reference", "split"}
            or type(value["report_only_val"]) is not bool):
        raise RuntimeError("NATIVE_CALIBRATION_RUN_INVALID")
    return dict(value)


def require_reference_learning_plan(recipe: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    binding = require_binding(policy.get("reference_learning_plan"), label="reference learning plan", verify_file=True)
    plan = read_bound_json(Path(binding["path"]), binding["sha256"])
    reference = require_reference_policy_contract(recipe["exit_reference_policy"])
    if (plan.get("schema_version") != "gx1_reference_critic_learning_plan_v1"
            or plan.get("decision") != "FROZEN_COMPARISON_READY"
            or plan.get("reference_policy_sha256") != reference["policy_sha256"]
            or plan.get("origin_state_sha256") != ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256
            or plan.get("teacher_model_state_sha256") != ENTRY_LEARNABILITY_TARGET_MODEL_SHA256
            or not isinstance(recipe.get("source_bindings_sha256"), str)
            or len(recipe["source_bindings_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in recipe["source_bindings_sha256"])
            or plan.get("source_bindings_sha256") != recipe["source_bindings_sha256"]
            or plan.get("split") != "train" or plan.get("test_data_used") is not False
            or plan.get("teacher_refresh_allowed") is not False
            or plan.get("entry_bridge") != "unchanged_initial_teacher_greedy_bridge_no_reference_refresh"
            or plan.get("comparison") != "same_frozen_targets_before_after_and_constant_baselines_by_side_and_month"
            or plan.get("trained_entry_count") != 512 or type(plan.get("trained_entry_count")) is not int
            or plan.get("separate_train_entry_count") != 128 or type(plan.get("separate_train_entry_count")) is not int
            or not isinstance(plan.get("targets"), Mapping)
            or set(plan["targets"]) != {"trained", "separate_train"}):
        raise RuntimeError("NATIVE_REFERENCE_LEARNING_PLAN_INVALID")
    targets = [require_binding(plan["targets"][role], label=f"frozen reference {role}", verify_file=True)
               for role in ("trained", "separate_train")]
    if targets[0]["path"] == targets[1]["path"] or targets[0]["sha256"] == targets[1]["sha256"]:
        raise RuntimeError("NATIVE_REFERENCE_COMPARISON_SEPARATION_INVALID")
    return plan


def _require_native_profile_and_economics(recipe, policy, repo):
    if _dynamic_m5_native_design(recipe) is not None:
        from gx1.contracts.entry_candidate_checkpoint_policy_v1 import DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
        profile = _dynamic_m5_compute_profile(recipe)
        if (policy.get("schema_version") != "gx1_next_native_run_policy_v1"
                or policy.get("canonical_source_repo") != str(repo) or policy.get("canonical_branch") != "work/gx1-current"
                or policy.get("training_module") != NATIVE_MODULE or policy.get("required_val_profile") != profile
                or policy.get("train_batch_size") != 16 or recipe.get("trainer_cli", {}).get("batch_size") != 16
                or recipe.get("trainer_cli", {}).get("checkpoint_monitor") != DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
                or policy.get("precision") != "float32" or policy.get("tf32_allowed") is not False
                or policy.get("native_invocation_seconds") != 12000 or policy.get("outer_guard_seconds") != 13800):
            raise RuntimeError("NATIVE_NEXT_RUN_DYNAMIC_PROFILE_INVALID")
        return profile, policy.get("required_evidence", {}), None
    profile = {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
               "max_wall_seconds": 10800, "progress_interval_forwards": 64}
    if (policy.get("schema_version") != "gx1_next_native_run_policy_v1"
            or policy.get("canonical_source_repo") != str(repo)
            or policy.get("canonical_branch") != "work/gx1-current"
            or policy.get("training_module") != NATIVE_MODULE
            or policy.get("required_val_profile") != profile
            or any(recipe.get("val_limits", {}).get(k) != v for k, v in profile.items())
            or policy.get("train_batch_size") != 16
            or recipe.get("trainer_cli", {}).get("batch_size") != 16
            or policy.get("precision") != "float32" or policy.get("tf32_allowed") is not False
            or policy.get("native_invocation_seconds") != 12000
            or policy.get("outer_guard_seconds") != 13800):
        raise RuntimeError("NATIVE_NEXT_RUN_PROFILE_INVALID")
    evidence = policy.get("required_evidence", {})
    risk_binding = require_binding(evidence.get("risk_objective"), label="native risk objective")
    risk = read_bound_json(Path(risk_binding["path"]), risk_binding["sha256"])
    econ_binding = require_binding(recipe.get("files", {}).get("economics_readiness"), label="native economics")
    economics = read_bound_json(Path(econ_binding["path"]), econ_binding["sha256"])
    objective = economics.get("economics_objective_contract", {})
    if (risk.get("decision") != "PASS" or risk.get("test_data_used") is not False
            or risk.get("maximum_holding_seconds") is not None
            or risk.get("absolute_loss_limit_bps") is not None
            or risk.get("reward_accounting") != "liquidation_advantage_v1"
            or risk.get("economics_objective_schema") != "gx1_unified_exit_economics_objective_v4"
            or objective.get("reward_accounting") != risk["reward_accounting"]
            or objective.get("schema_version") != risk["economics_objective_schema"]):
        raise RuntimeError("NATIVE_NEXT_RUN_RISK_OBJECTIVE_INVALID")
    return profile, evidence, objective



def require_frozen_train_policy_evaluation(recipe, *, invocation_number=None, execution_budget=None):
    """One frozen TRAIN rollout or separately bound Entry representation probe."""
    from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import require_frozen_train_policy_plan
    prefix = require_chronological_prefix_recipe(recipe)
    if any(key in recipe for key in ("chronological_initial_measurement","chronological_learning_measurement",
            "chronological_learning_continuation","chronological_train_only_measurement","chronological_entry_baseline",
            "entry_gradient_diagnostic","frozen_readout_evaluation","candidate_resume_origin","native_calibration")):
        raise RuntimeError("FROZEN_TRAIN_POLICY_MIXED_SCOPE")
    probe_binding = recipe.get("frozen_entry_selector_probe")
    probe = None
    if probe_binding is not None:
        if "frozen_train_policy_evaluation" in recipe:
            raise RuntimeError("FROZEN_ENTRY_PROBE_MIXED_SCOPE")
        probe_binding = require_binding(probe_binding, label="frozen Entry probe", verify_file=True)
        probe = read_bound_json(Path(probe_binding["path"]), probe_binding["sha256"])
        fixed = {"schema_version":"gx1_frozen_entry_representation_probe_v1",
            "stage":"cache_original_entry_representations_only", "optimizer_steps":0,
            "new_fits":0, "max_entry_forwards":16, "entry_batch_size":16,
            "entries":256, "control_forwards":0, "exit_rollout_forwards":0,
            "training_enabled":False, "test_data_used":False}
        if any(type(probe.get(k)) is not type(v) or probe[k] != v for k,v in fixed.items()):
            raise RuntimeError("FROZEN_ENTRY_PROBE_PLAN_INVALID")
    scope = require_frozen_train_policy_plan(
        probe["source_policy_plan"] if probe is not None else recipe.get("frozen_train_policy_evaluation"))
    if probe is not None:
        payloads = {}
        for key in ("completed_policy", "source_result", "dataset_review", "dataset", "review"):
            b = require_binding(probe.get(key), label="frozen Entry probe " + key, verify_file=True)
            payloads[key] = read_bound_json(Path(b["path"]), b["sha256"])
        completed, result = payloads["completed_policy"], payloads["source_result"]
        dataset, reviewed = payloads["dataset"], payloads["dataset_review"]
        if (completed.get("result") != probe["source_result"]
                or completed.get("evaluation_plan") != scope["plan_binding"]
                or completed.get("learning_review_complete") is not True
                or completed.get("optimizer_steps") != 0 or completed.get("exited_side_trade_count") != 512
                or completed.get("model_state_sha256") != scope["completion"]["model_state_sha256"]
                or result.get("decision") != "PASS_COMPLETE"
                or result.get("evaluation_cohort") != scope["cohort"]
                or result.get("exited_side_trade_count") != 512
                or len(result.get("trade_outcomes", [])) != 512
                or any(row.get("status") != "EXITED" for row in result["trade_outcomes"])
                or dataset.get("source_result") != probe["source_result"]
                or dataset.get("source_policy_model_state_sha256") != completed["model_state_sha256"]
                or reviewed.get("dataset") != probe["dataset"]
                or reviewed.get("entries_preserved") != 256 or reviewed.get("negative_labels_preserved") != 409
                or payloads["review"].get("dataset_review_binding") != probe["dataset_review"]
                or [r.get("child_entry_row_index") for r in dataset.get("rows", [])] != scope["cohort"]["entry_row_indices"]
                or [r.get("parent_entry_row_index") for r in dataset["rows"]] != scope["cohort"]["parent_entry_row_indices"]):
            raise RuntimeError("FROZEN_ENTRY_PROBE_SOURCE_OR_COHORT_INVALID")
        scope["entry_selector_probe"] = {"binding":probe_binding,"plan":probe,**payloads}
    repo = Path(__file__).resolve().parents[2]
    binding = require_binding(recipe.get("next_run_policy"), label="frozen TRAIN native policy", verify_file=True)
    if Path(binding["path"]) != repo/"NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(binding["path"]), binding["sha256"])
    _require_native_profile_and_economics(recipe, policy, repo)
    expected = {"plan":scope["plan_binding"],"chronological_prefix":prefix["artifacts"],
        "run_id":recipe.get("run_id"),"out_bundle_dir":recipe.get("out_bundle_dir"),
        "source_bindings_sha256":recipe.get("source_bindings_sha256"),"optimizer_steps":0,
        "origin_optimizer_steps":512,"max_invocations":1,"train_entries":256,"control_forwards":0,
        "val_limits":scope["plan"]["val_limits"],"test_data_used":False}
    scope_key = "frozen_train_policy_evaluation"
    if probe is not None:
        scope_key = "frozen_entry_selector_probe"
        expected.update(plan=probe_binding, new_fits=0, max_entry_forwards=16, exit_rollout_forwards=0)
        if ("frozen_train_policy_evaluation" in policy
                or policy.get("policy_consistent_entry_review") != probe["review"]):
            raise RuntimeError("FROZEN_ENTRY_PROBE_POLICY_REVIEW_INVALID")
    if (policy.get("training_enabled") is not False or policy.get(scope_key) != expected
            or any(key in policy for key in ("chronological_learning_run","native_learning_calibration",
                                            "entry_gradient_diagnostic","frozen_readout_evaluation"))
            or recipe.get("files") != scope["origin_recipe"]["files"]
            or recipe.get("chronological_prefix") != scope["origin_recipe"]["chronological_prefix"]
            or recipe.get("exit_reference_policy") != scope["origin_recipe"]["exit_reference_policy"]
            or recipe.get("val_limits") != scope["plan"]["val_limits"]):
        raise RuntimeError("FROZEN_TRAIN_POLICY_NATIVE_SCOPE_INVALID")
    if invocation_number is not None and (type(invocation_number) is not int or invocation_number != 1):
        raise RuntimeError("FROZEN_TRAIN_POLICY_INVOCATION_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != 512
            or execution_budget.get("max_invocation_seconds") != 12000
            or execution_budget.get("expected_active_pointer_sha256") is not None
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("FROZEN_TRAIN_POLICY_EXECUTION_BUDGET_INVALID")
    return scope


def require_frozen_readout_evaluation(recipe, *, invocation_number=None, execution_budget=None):
    """One bound, read-only native VAL window per arm; no training admission."""
    from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_bounded_val_cohort
    repo = Path(__file__).resolve().parents[2]
    value = recipe.get("frozen_readout_evaluation")
    if (not isinstance(value, Mapping) or set(value) != {"plan", "origin_cursor", "arm"}
            or value.get("arm") not in {"baseline", "candidate"}
            or any(k in recipe for k in ("candidate_resume_origin", "native_calibration", "exit_backup_steps", "exit_reference_policy"))):
        raise RuntimeError("NATIVE_FROZEN_READOUT_RECIPE_INVALID")
    binding = require_binding(recipe.get("next_run_policy"), label="frozen native policy")
    if Path(binding["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(binding["path"]), binding["sha256"])
    _require_native_profile_and_economics(recipe, policy, repo)
    expected = {"plan": value["plan"], "origin_cursor": value["origin_cursor"],
                "allowed_arms": ["baseline", "candidate"], "optimizer_steps": 0,
                "max_invocations_per_arm": 1, "full_val_allowed": False, "test_data_used": False}
    if (policy.get("training_enabled") is not False
            or policy.get("frozen_readout_evaluation") != expected
            or "native_learning_calibration" in policy):
        raise RuntimeError("NATIVE_FROZEN_READOUT_NOT_AUTHORIZED")
    if invocation_number is not None and (type(invocation_number) is not int or invocation_number != 1):
        raise RuntimeError("NATIVE_FROZEN_READOUT_INVOCATION_INVALID")
    cohort = build_bounded_val_cohort(value["plan"])
    plan = read_bound_json(Path(cohort["plan"]["path"]), cohort["plan"]["sha256"])
    cb = require_binding(value["origin_cursor"], label="frozen native origin cursor")
    raw = read_bound_json(Path(cb["path"]), cb["sha256"])
    cursor = require_native_cursor(raw, expected_recipe=raw["recipe"], verify_files=True)
    state = cursor["resume_state"]
    if (len(cohort["entry_row_indices"]) != 256 or cohort["population_rows"] != 5508
            or state["training_state"] != plan["base_online_checkpoint"]
            or state["phase"] != "train" or state["complete"] is not False
            or cursor["outcome"] != "RESUMABLE" or state["active_val_cursor"] is not None
            or recipe.get("files", {}).get("random_access_root") != plan["val_root"]):
        raise RuntimeError("NATIVE_FROZEN_READOUT_ORIGIN_OR_COHORT_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != state["global_optimizer_steps"]
            or execution_budget.get("expected_active_pointer_sha256") is not None
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("NATIVE_FROZEN_READOUT_BUDGET_INVALID")
    return {"cohort":cohort, "plan":plan, "origin_resume_state":state, "arm":value["arm"]}


def require_native_run_scope(
    recipe: Mapping[str, Any], *, invocation_number: int | None = None,
    execution_budget: Mapping[str, Any] | None = None,
) -> int | None:
    """Enforce the operator's bound readiness policy before native model work.

    The sole pre-training exception is a finite, declared TRAIN calibration.
    It uses the normal native session, production profile and machine guards.
    """
    if "entry_dynamic_m5_technical_qualification" in recipe:
        return require_dynamic_m5_technical_qualification(recipe, invocation_number=invocation_number,
            execution_budget=execution_budget)["optimizer_step_ceiling"]
    if "entry_m5_path_single_epoch" in recipe:
        if "chronological_initial_measurement" in recipe:
            require_chronological_initial_measurement(recipe, invocation_number=invocation_number,
                execution_budget=execution_budget)
            if any(k in recipe for k in ("entry_learning_study", "chronological_learning_measurement",
                    "chronological_learning_continuation", "native_calibration", "entry_gradient_diagnostic")):
                raise RuntimeError("NATIVE_M5_SINGLE_EPOCH_MIXED_OR_UNMEASURED_SCOPE")
            return 0
        return require_m5_single_epoch_run(recipe, invocation_number=invocation_number,
            execution_budget=execution_budget)["optimizer_step_ceiling"]
    if "entry_learning_study" in recipe:
        return require_entry_learning_study_run(
            recipe, invocation_number=invocation_number,
            execution_budget=execution_budget)["phase_spec"]["optimizer_steps"]
    observed_replay = False
    if _observed_entry_in(recipe) and "chronological_learning_continuation" in recipe:
        observed_replay = require_chronological_continuation(recipe).get("resume_equivalence") is True
    if _observed_entry_in(recipe) and (
            "chronological_prefix" not in recipe
            or sum(key in recipe for key in ("chronological_initial_measurement", "chronological_learning_measurement")) != 1
            or any(key in recipe for key in ("frozen_train_policy_evaluation", "frozen_entry_selector_probe",
                "entry_gradient_diagnostic", "chronological_entry_baseline", "chronological_learning_continuation",
                "chronological_train_only_measurement", "candidate_resume_origin", "frozen_readout_evaluation",
                "native_calibration") if not (key == "chronological_learning_continuation" and observed_replay))):
        raise RuntimeError("NATIVE_OBSERVED_ENTRY_MIXED_OR_UNMEASURED_SCOPE")
    if "frozen_train_policy_evaluation" in recipe or "frozen_entry_selector_probe" in recipe:
        scope = require_frozen_train_policy_evaluation(recipe, invocation_number=invocation_number,
                                                      execution_budget=execution_budget)
        return scope["origin_resume_state"]["global_optimizer_steps"]
    if "chronological_entry_baseline" in recipe and recipe.get("chronological_train_only_measurement") is not True:
        raise RuntimeError("NATIVE_PREFIX_DERIVED_ENTRY_TRAIN_ONLY_REQUIRED")
    if "chronological_train_only_measurement" in recipe and (
            recipe["chronological_train_only_measurement"] is not True
            or "chronological_prefix" not in recipe or "entry_gradient_diagnostic" in recipe
            or sum(key in recipe for key in ("chronological_initial_measurement", "chronological_learning_measurement")) != 1):
        raise RuntimeError("NATIVE_PREFIX_TRAIN_ONLY_MEASUREMENT_INVALID")
    if "entry_gradient_diagnostic" in recipe:
        scope = require_entry_gradient_diagnostic(recipe, invocation_number=invocation_number, execution_budget=execution_budget)
        return scope["origin_resume_state"]["global_optimizer_steps"]
    if "chronological_initial_measurement" in recipe:
        require_chronological_initial_measurement(recipe, invocation_number=invocation_number,
                                                  execution_budget=execution_budget)
        return 0
    if "chronological_prefix" in recipe:
        scope = require_chronological_prefix_run(recipe, invocation_number=invocation_number, execution_budget=execution_budget)
        return scope["optimizer_step_ceiling"]
    if "frozen_readout_evaluation" in recipe:
        scope = require_frozen_readout_evaluation(recipe, invocation_number=invocation_number,
                                                execution_budget=execution_budget)
        return scope["origin_resume_state"]["global_optimizer_steps"]
    repo = Path(__file__).resolve().parents[2]
    origin = recipe.get("candidate_resume_origin")
    optimizer_transition = (isinstance(origin, Mapping)
                            and origin.get("schema_version") == OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA)
    continuation = (isinstance(origin, Mapping)
                    and origin.get("schema_version") == TRAINING_CONTINUATION_SCHEMA)
    target_refresh = (isinstance(origin, Mapping)
                      and origin.get("schema_version") == FQI_TARGET_REFRESH_SCHEMA)
    entry_learnability = (isinstance(origin, Mapping)
                          and origin.get("schema_version") == ENTRY_LEARNABILITY_SCHEMA)
    if entry_learnability:
        require_entry_learnability_origin(origin)
        replay_control = entry_learnability_control(origin)
    elif optimizer_transition:
        require_optimizer_procedure_origin(origin)
    elif continuation:
        require_training_continuation_origin(origin)
    elif target_refresh:
        require_fqi_target_refresh_origin(origin)
    trace_backup = continuation and "exit_backup_steps" in origin
    reference_mode = continuation and "exit_reference_policy" in origin
    if reference_mode:
        reference = require_reference_policy_contract(origin["exit_reference_policy"])
        if require_reference_policy_contract(recipe.get("exit_reference_policy")) != reference:
            raise RuntimeError("NATIVE_REFERENCE_POLICY_RECIPE_MISMATCH")
    elif "exit_reference_policy" in recipe:
        raise RuntimeError("NATIVE_REFERENCE_POLICY_RECIPE_MISMATCH")

    if ((trace_backup and (type(recipe.get("exit_backup_steps")) is not int or recipe["exit_backup_steps"] != 5))
            or (not trace_backup and "exit_backup_steps" in recipe)):
        raise RuntimeError("NATIVE_TRACE_BACKUP_RECIPE_MISMATCH")
    origin_fields = {"schema_version", "contract", "pointer"}
    optional_origin_fields = {"exit_value_initialization", "train_population_scope"}
    if not (optimizer_transition or continuation or target_refresh or entry_learnability) and (not isinstance(origin, Mapping) or not origin_fields <= set(origin)
            or set(origin) - origin_fields - optional_origin_fields
            or ("train_population_scope" in origin and (
                type(origin["train_population_scope"]) is not str
                or origin["train_population_scope"] != "latest_year_2025_2026_v1"))
            or origin.get("schema_version") != "gx1_candidate_economics_transition_origin_v1"
            or ("exit_value_initialization" in origin and (
                type(origin["exit_value_initialization"]) is not str
                or origin["exit_value_initialization"] != "close_now_baseline_v1"))):
        raise RuntimeError("NATIVE_ECONOMICS_TRANSITION_ORIGIN_REQUIRED")
    completed_val_ceiling = native_completed_val_ceiling(recipe)
    initialization = origin.get("exit_value_initialization")
    calibration = require_native_calibration_run(recipe)
    population = origin.get("train_population_scope")
    require_binding(origin["contract"], label="native origin contract")
    origin_pointer = require_binding(origin["pointer"], label="native origin pointer")
    binding = require_binding(recipe.get("next_run_policy"), label="next native run policy")
    if Path(binding["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(binding["path"]), binding["sha256"])
    if (optimizer_transition or continuation or target_refresh or entry_learnability) and (
            policy.get("training_enabled") is not False
            or policy.get("gradient_clipping_policy") != OPTIMIZER_PROCEDURE_TRANSITION_POLICY
            or (calibration is not None and not (trace_backup or reference_mode))):
        raise RuntimeError("NATIVE_OPTIMIZER_PROCEDURE_BOUNDED_TRAIN_ONLY_REQUIRED")
    if reference_mode:
        if (calibration is None or calibration["arm"] != "reference" or calibration["report_only_val"] is not False
                or require_reference_policy_contract(policy.get("exit_reference_policy")) != reference):
            raise RuntimeError("NATIVE_REFERENCE_POLICY_CALIBRATION_INVALID")
        require_reference_learning_plan(recipe, policy)
    elif "exit_reference_policy" in policy or "reference_learning_plan" in policy:
        raise RuntimeError("NATIVE_REFERENCE_POLICY_POLICY_MISMATCH")
    if trace_backup and (calibration is None or calibration["report_only_val"] is not False):
        raise RuntimeError("NATIVE_TRACE_BACKUP_TRAIN_ONLY_CALIBRATION_REQUIRED")
    if ((trace_backup and (type(policy.get("exit_backup_steps")) is not int or policy["exit_backup_steps"] != 5))
            or (not trace_backup and "exit_backup_steps" in policy)):
        raise RuntimeError("NATIVE_TRACE_BACKUP_POLICY_MISMATCH")
    if (("exit_value_initialization" in policy and (
            type(policy["exit_value_initialization"]) is not str
            or policy["exit_value_initialization"] != "close_now_baseline_v1"))
            or policy.get("exit_value_initialization") != initialization):
        raise RuntimeError("NATIVE_EXIT_VALUE_INITIALIZATION_POLICY_MISMATCH")
    if (("train_population_scope" in policy and (
            type(policy["train_population_scope"]) is not str
            or policy["train_population_scope"] != "latest_year_2025_2026_v1"))
            or policy.get("train_population_scope") != population):
        raise RuntimeError("NATIVE_TRAIN_POPULATION_POLICY_MISMATCH")
    if population is not None:
        root_binding = require_binding(recipe.get("files", {}).get("random_access_root"), label="native training population")
        root = require_random_access_index_root(read_bound_json(Path(root_binding["path"]), root_binding["sha256"]))
        if root["schema_version"] != LATEST_YEAR_ROOT_SCHEMA_VERSION:
            raise RuntimeError("NATIVE_LATEST_YEAR_POPULATION_REQUIRED")
    profile, evidence, objective = _require_native_profile_and_economics(recipe, policy, repo)
    if policy.get("training_enabled") is True:
        if calibration is not None:
            raise RuntimeError("NATIVE_CALIBRATION_CANNOT_ENABLE_FULL_TRAINING")
        for role in ("checkpoint_transition", "learning_calibration", "gpu_batch256_parity",
                     "end_to_end_throughput", "resume_equivalence"):
            proof_binding = require_binding(evidence.get(role), label=f"native {role}")
            proof = read_bound_json(Path(proof_binding["path"]), proof_binding["sha256"])
            if (proof.get("decision") != "PASS" or proof.get("test_data_used") is not False
                    or proof.get("evidence_role") != role
                    or proof.get("economics_objective_contract_sha256") != objective.get("contract_sha256")
                    or proof.get("source_bindings_sha256") != recipe.get("source_bindings_sha256")
                    or proof.get("training_origin_pointer_sha256") != origin_pointer["sha256"]
                    or ("exit_value_initialization" in proof and (
                        type(proof["exit_value_initialization"]) is not str
                        or proof["exit_value_initialization"] != "close_now_baseline_v1"))
                    or proof.get("exit_value_initialization") != initialization
                    or proof.get("training_population_root_sha256") != recipe.get("files", {}).get("random_access_root", {}).get("sha256")
                    or proof.get("native_val_profile") != profile):
                raise RuntimeError("NATIVE_NEXT_RUN_EVIDENCE_NOT_PASS")
        ceiling = None
    elif policy.get("training_enabled") is False and completed_val_ceiling is not None:
        expected_scope = {
            "schema_version": "gx1_native_training_continuation_scope_v2",
            "stop_after_completed_val_epochs": 2,
            "full_epoch_training_allowed": True, "test_data_used": False,
        }
        scope = policy.get("native_learning_calibration")
        if (scope != expected_scope or type(scope.get("stop_after_completed_val_epochs")) is not int
                or scope.get("full_epoch_training_allowed") is not True
                or scope.get("test_data_used") is not False):
            raise RuntimeError("NATIVE_TRAINING_CONTINUATION_VAL_SCOPE_INVALID")
        if invocation_number is not None and (type(invocation_number) is not int or invocation_number < 1):
            raise RuntimeError("NATIVE_CALIBRATION_INVOCATION_INVALID")
        # Reuse the measured technical gates with unchanged model/data/economics.
        # Their historical source/origin bindings remain historical, never rewritten.
        for role in ("checkpoint_transition", "learning_calibration", "gpu_batch256_parity",
                     "end_to_end_throughput", "resume_equivalence"):
            proof_binding = require_binding(evidence.get(role), label=f"native reused {role}")
            proof = read_bound_json(Path(proof_binding["path"]), proof_binding["sha256"])
            if (proof.get("decision") != "PASS" or proof.get("test_data_used") is not False
                    or proof.get("evidence_role") != role
                    or proof.get("economics_objective_contract_sha256") != objective.get("contract_sha256")
                    or proof.get("exit_value_initialization") != initialization
                    or proof.get("training_population_root_sha256") != recipe.get("files", {}).get("random_access_root", {}).get("sha256")
                    or proof.get("native_val_profile") != profile):
                raise RuntimeError("NATIVE_CONTINUATION_REUSED_EVIDENCE_NOT_PASS")
        ceiling = None
    elif policy.get("training_enabled") is False:
        scope = policy.get("native_learning_calibration")
        old_scope = {
            "schema_version": "gx1_native_learning_calibration_scope_v1",
            "optimizer_step_ceilings": [16, 32],
            "full_epoch_training_allowed": False, "test_data_used": False,
        }
        measured_scope = {
            **old_scope, "schema_version": "gx1_native_learning_calibration_scope_v2",
            "reference_optimizer_step_ceiling": 32, "native_report_only_val": True,
        }
        optimizer_scope = {
            "schema_version": "gx1_native_optimizer_procedure_calibration_scope_v1",
            "additional_optimizer_step_ceilings": [16, 32],
            "full_epoch_training_allowed": False, "test_data_used": False,
        }
        continuation_scope = {
            "schema_version": "gx1_native_training_continuation_scope_v1",
            "additional_optimizer_step_ceilings": [256],
            "full_epoch_training_allowed": False, "test_data_used": False,
        }
        refresh_scope = {**continuation_scope, "schema_version": "gx1_native_fqi_target_refresh_scope_v1"}
        replay_scope = {**continuation_scope, "schema_version": "gx1_native_entry_learnability_scope_v1"}
        if entry_learnability:
            replay_scope["additional_optimizer_step_ceilings"] = [
                replay_control["step_ceiling"] - replay_control["cursor"]["global_optimizer_steps"]]
        if reference_mode:
            expected_scope = {
                "schema_version": "gx1_native_reference_policy_scope_v1",
                "additional_optimizer_step_ceilings": [32],
                "reference_additional_optimizer_step_ceiling": 32,
                "full_epoch_training_allowed": False, "test_data_used": False,
            }
            if (not isinstance(scope, Mapping) or scope != expected_scope
                    or type(scope["reference_additional_optimizer_step_ceiling"]) is not int
                    or any(type(n) is not int for n in scope["additional_optimizer_step_ceilings"])
                    or scope["full_epoch_training_allowed"] is not False or scope["test_data_used"] is not False):
                raise RuntimeError("NATIVE_REFERENCE_POLICY_CALIBRATION_SCOPE_INVALID")
        elif trace_backup:
            expected_scope = {
                "schema_version": "gx1_native_frozen_policy_trace_scope_v1",
                "additional_optimizer_step_ceilings": [16, 32],
                "reference_additional_optimizer_step_ceiling": 32,
                "full_epoch_training_allowed": False, "test_data_used": False,
            }
            if (not isinstance(scope, Mapping) or scope != expected_scope
                    or any(type(n) is not int for n in scope["additional_optimizer_step_ceilings"])
                    or type(scope["reference_additional_optimizer_step_ceiling"]) is not int
                    or scope["full_epoch_training_allowed"] is not False
                    or scope["test_data_used"] is not False):
                raise RuntimeError("NATIVE_TRACE_BACKUP_CALIBRATION_SCOPE_INVALID")
        elif continuation or target_refresh or entry_learnability:
            expected_scope = replay_scope if entry_learnability else refresh_scope if target_refresh else continuation_scope
            if (scope != expected_scope or not isinstance(scope, Mapping)
                    or scope.get("full_epoch_training_allowed") is not False
                    or scope.get("test_data_used") is not False
                    or any(type(x) is not int for x in scope["additional_optimizer_step_ceilings"])):
                raise RuntimeError("NATIVE_ENTRY_LEARNABILITY_CALIBRATION_SCOPE_INVALID" if entry_learnability
                                   else "NATIVE_FQI_TARGET_REFRESH_CALIBRATION_SCOPE_INVALID" if target_refresh
                                   else "NATIVE_TRAINING_CONTINUATION_CALIBRATION_SCOPE_INVALID")
        elif optimizer_transition:
            if (scope != optimizer_scope or not isinstance(scope, Mapping)
                    or scope.get("full_epoch_training_allowed") is not False
                    or scope.get("test_data_used") is not False
                    or any(type(x) is not int for x in scope["additional_optimizer_step_ceilings"])):
                raise RuntimeError("NATIVE_OPTIMIZER_PROCEDURE_CALIBRATION_SCOPE_INVALID")
        elif (not isinstance(scope, Mapping)
                or scope not in (old_scope, measured_scope)
                or any(type(x) is not int for x in scope["optimizer_step_ceilings"])
                or scope["full_epoch_training_allowed"] is not False
                or scope["test_data_used"] is not False
                or (calibration is None) != (scope == old_scope)
                or (scope == measured_scope and scope["native_report_only_val"] is not True)):
            raise RuntimeError("NATIVE_TRAINING_BLOCKED_CALIBRATION_SCOPE_REQUIRED")
        trace_ceilings = ([ENTRY_LEARNABILITY_ORIGIN_CURSOR["global_optimizer_steps"] + delta
                           for delta in ([scope["reference_additional_optimizer_step_ceiling"]]
                                         if calibration["arm"] == "reference"
                                         else scope["additional_optimizer_step_ceilings"])]
                          if trace_backup or reference_mode else None)
        ceilings = (trace_ceilings if trace_backup or reference_mode else [replay_control["step_ceiling"]] if entry_learnability else
                    [FQI_TARGET_REFRESH_STEP_CEILING] if target_refresh else
                    [TRAINING_CONTINUATION_STEP_CEILING] if continuation else
                    ([OPTIMIZER_PROCEDURE_ORIGIN_CURSOR["global_optimizer_steps"] + delta
                      for delta in scope["additional_optimizer_step_ceilings"]]
                     if optimizer_transition else
                     ([32] if calibration is not None and calibration["arm"] == "reference"
                      else scope["optimizer_step_ceilings"])))
        if invocation_number is not None:
            if type(invocation_number) is not int or not 1 <= invocation_number <= len(ceilings):
                raise RuntimeError("NATIVE_CALIBRATION_INVOCATION_INVALID")
            ceiling = ceilings[invocation_number - 1]
        elif execution_budget is not None:
            ceiling = execution_budget.get("stop_after_optimizer_steps")
            if type(ceiling) is not int or ceiling not in ceilings:
                raise RuntimeError("NATIVE_CALIBRATION_STEP_CEILING_INVALID")
        elif optimizer_transition or continuation or target_refresh or entry_learnability:
            # Metadata/state-transition validation only. The actual native
            # window still supplies its invocation or execution budget.
            ceiling = max(ceilings)
        else:
            raise RuntimeError("NATIVE_CALIBRATION_BOUNDARY_REQUIRED")
    else:
        raise RuntimeError("NATIVE_NEXT_RUN_ENABLEMENT_INVALID")
    if execution_budget is not None and (
            execution_budget.get("stop_after_optimizer_steps") != ceiling
            or execution_budget.get("stop_after_completed_val_epochs") != completed_val_ceiling
            or (completed_val_ceiling is not None and type(execution_budget.get("stop_after_completed_val_epochs")) is not int)
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("NATIVE_NEXT_RUN_BUDGET_INVALID")
    return ceiling


def require_native_window_policy(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    """Bind a window to one recipe and private campaign output paths."""
    fields = {
        "schema_version", "recipe", "invocation_number", "max_invocation_seconds",
        "budget_path", "progress_path", "campaign_cursor_path",
        "training_session_directory", "test_data_used", "policy_sha256",
    }
    if (
        not isinstance(value, Mapping) or set(value) != fields
        or value["schema_version"] != WINDOW_SCHEMA or value["test_data_used"] is not False
        or type(value["invocation_number"]) is not int or value["invocation_number"] < 1
        or type(value["max_invocation_seconds"]) is not int
        or value["max_invocation_seconds"] != 12000
        or value["policy_sha256"] != canonical_sha256({k: v for k, v in value.items() if k != "policy_sha256"})
    ):
        raise RuntimeError("NATIVE_CANDIDATE_WINDOW_POLICY_INVALID")
    require_binding(value["recipe"], label="native window recipe", verify_file=verify_files)
    for field in ("budget_path", "progress_path", "campaign_cursor_path", "training_session_directory"):
        if not isinstance(value[field], str):
            raise RuntimeError("NATIVE_CANDIDATE_WINDOW_PATH_INVALID")
        path = Path(value[field])
        if not path.is_absolute() or path.resolve() != path or path.is_symlink():
            raise RuntimeError("NATIVE_CANDIDATE_WINDOW_PATH_INVALID")
    if len({value[key] for key in ("budget_path", "progress_path", "campaign_cursor_path")}) != 3:
        raise RuntimeError("NATIVE_CANDIDATE_WINDOW_PATH_COLLISION")
    return dict(value)



def require_complete_val_observation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Require all June outcomes; natural split-end censoring is not a data failure."""
    from collections import Counter
    from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import (
        require_entry_policy_decisions, coupled_entry_exit_policy_metrics,
        marked_entry_exit_policy_metrics,
    )

    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
        RESULT_SCHEMA_VERSION, MARKED_RESULT_SCHEMA_VERSION, LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION,
    )
    if result.get("schema_version") not in {RESULT_SCHEMA_VERSION, MARKED_RESULT_SCHEMA_VERSION, LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION}:
        raise RuntimeError("NATIVE_CANDIDATE_VAL_SCHEMA_INVALID")
    outcomes = result.get("trade_outcomes")
    if (not isinstance(outcomes, list) or len(outcomes) != 11016
            or any(not isinstance(row, Mapping) for row in outcomes)):
        raise RuntimeError("NATIVE_CANDIDATE_FULL_VAL_OUTCOMES_REQUIRED")
    counts = Counter(row.get("status") for row in outcomes)
    censored = counts.get("RIGHT_CENSORED_SPLIT_END", 0)
    if (set(counts) - {"EXITED", "RIGHT_CENSORED_SPLIT_END"}
            or result.get("decision") != ("COMPLETE_WITH_RIGHT_CENSORING" if censored else "PASS_COMPLETE")
            or result.get("test_data_used") is not False
            or result.get("rollout_execution_complete") is not True
            or result.get("entry_pair_cohort_size") != 5508
            or result.get("side_trade_count") != 11016
            or result.get("exited_side_trade_count") != counts.get("EXITED", 0)
            or result.get("right_censored_side_trade_count") != censored
            or result.get("compute_truncated_side_trade_count") != 0):
        raise RuntimeError("NATIVE_CANDIDATE_COMPLETE_OBSERVED_VAL_REQUIRED")
    policy = require_entry_policy_decisions(
        result.get("entry_policy_decisions"), entry_row_indices=list(range(5508)),
        checkpoint_binding_sha256=result["checkpoint_binding_sha256"],
    )
    metrics = coupled_entry_exit_policy_metrics(
        entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True,
    )
    if (result.get("entry_exit_policy_metrics") != metrics
            or result.get("full_cohort_policy_metrics_authoritative")
            is not metrics["full_cohort_authoritative"]):
        raise RuntimeError("NATIVE_CANDIDATE_FULL_VAL_POLICY_METRICS_INVALID")
    relative = result["schema_version"] == LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION
    if ((relative and result.get("exit_q_value_coordinates") != "advantage_over_executable_liquidation_bps")
            or (not relative and "exit_q_value_coordinates" in result)):
        raise RuntimeError("NATIVE_CANDIDATE_VAL_VALUE_COORDINATES_INVALID")
    if result["schema_version"] in {MARKED_RESULT_SCHEMA_VERSION, LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION}:
        marked = marked_entry_exit_policy_metrics(
            entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True,
        )
        if result.get("marked_policy_evaluation") != marked:
            raise RuntimeError("NATIVE_CANDIDATE_MARKED_VAL_METRICS_INVALID")
    elif "marked_policy_evaluation" in result:
        raise RuntimeError("NATIVE_CANDIDATE_MARKED_VAL_SCHEMA_MISMATCH")
    return dict(result)


def require_native_completed_smoke(
    *, plan: Mapping[str, Any], prior: Mapping[str, Any], recipe: Mapping[str, Any],
) -> None:
    from gx1.contracts.local_random_access_campaign_v2 import require_receipt_chain
    from gx1.scripts.local_random_access_campaign_v2 import _receipts

    receipts = require_receipt_chain(prior, _receipts(Path(prior["runtime_root"])), verify_files=True)
    if (
        prior["phase"] != "full_val" or not receipts or receipts[-1]["outcome"] != "COMPLETE"
        or recipe["seed_authority"] != prior["final_train_checkpoint_authority"]
        or plan["final_train_checkpoint_authority"] != recipe["seed_authority"]
        or plan["selection_receipt"] != prior["selection_receipt"]
    ):
        raise RuntimeError("NATIVE_CANDIDATE_COMPLETED_SMOKE_CAMPAIGN_REQUIRED")
    argv = prior["checked_invocations"][receipts[-1]["invocation_number"] - 1]["launcher_argv"]
    binding = require_binding(recipe["smoke_full_val"], label="complete smoke VAL result")
    if argv.count("--result-path") != 1 or binding["path"] != argv[argv.index("--result-path") + 1]:
        raise RuntimeError("NATIVE_CANDIDATE_SMOKE_RESULT_PATH_MISMATCH")
    result = read_bound_json(Path(binding["path"]), binding["sha256"])
    require_complete_val_observation(result)
    if result.get("semantic_result_sha256") != native_sha256(
        {k: v for k, v in result.items() if k != "semantic_result_sha256"}
    ):
        raise RuntimeError("NATIVE_CANDIDATE_COMPLETE_SMOKE_RESULT_BINDING_INVALID")


def require_native_cursor(
    value: Any, *, expected_recipe: Mapping[str, str], verify_files: bool = True,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "recipe", "resume_state", "outcome", "test_data_used", "cursor_sha256"}
        or value["schema_version"] != CURSOR_SCHEMA
        or value["recipe"] != expected_recipe or value["test_data_used"] is not False
        or value["outcome"] not in {"RESUMABLE", "COMPLETE"}
        or value["cursor_sha256"] != canonical_sha256({k: v for k, v in value.items() if k != "cursor_sha256"})
    ):
        raise RuntimeError("NATIVE_CANDIDATE_CAMPAIGN_CURSOR_INVALID")
    require_binding(value["recipe"], label="native candidate recipe", verify_file=verify_files)
    state = value["resume_state"]
    fields = {
        "training_pointer", "training_state", "active_val_cursor", "active_val_model_forwards",
        "epoch_schedule_sha256", "session_contract_sha256", "phase", "epoch_index",
        "next_batch_offset", "global_optimizer_steps", "complete",
    }
    if not isinstance(state, Mapping) or set(state) != fields:
        raise RuntimeError("NATIVE_CANDIDATE_RESUME_STATE_INVALID")
    if (
        state["phase"] not in {"train", "validation"}
        or type(state["complete"]) is not bool
        or state["complete"] != (value["outcome"] == "COMPLETE")
        or any(type(state[key]) is not int or state[key] < 0 for key in (
            "epoch_index", "next_batch_offset", "global_optimizer_steps", "active_val_model_forwards",
        ))
        or state["epoch_index"] > 30
        or (state["epoch_index"] == 30 and not state["complete"])
        or any(not isinstance(state[key], str) or len(state[key]) != 64
               or any(c not in "0123456789abcdef" for c in state[key])
               for key in ("epoch_schedule_sha256", "session_contract_sha256"))
    ):
        raise RuntimeError("NATIVE_CANDIDATE_RESUME_POSITION_INVALID")
    pointer = require_binding(state["training_pointer"], label="native TRAIN pointer", verify_file=verify_files)
    checkpoint = require_binding(state["training_state"], label="native TRAIN state", verify_file=verify_files)
    directory = Path(pointer["path"]).parent
    if (
        Path(pointer["path"]).name != "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
        or Path(checkpoint["path"]).parent != directory
    ):
        raise RuntimeError("NATIVE_CANDIDATE_CHECKPOINT_LAYOUT_INVALID")
    cursor = state["active_val_cursor"]
    if cursor is not None:
        cursor = require_binding(cursor, label="native active VAL cursor", verify_file=verify_files)
        expected = directory / "native_val" / f"epoch_{state['epoch_index'] + 1:04d}" / "ROLLOUT_PROGRESS.json"
        if state["phase"] != "validation" or cursor["path"] != str(expected):
            raise RuntimeError("NATIVE_CANDIDATE_VAL_CURSOR_LAYOUT_INVALID")
        if verify_files:
            raw = read_bound_json(Path(cursor["path"]), cursor["sha256"])
            if raw["model_forward_count"] != state["active_val_model_forwards"]:
                raise RuntimeError("NATIVE_CANDIDATE_VAL_CURSOR_COUNTER_MISMATCH")
    elif state["active_val_model_forwards"] != 0:
        raise RuntimeError("NATIVE_CANDIDATE_VAL_CURSOR_MISSING")
    if verify_files:
        raw = read_bound_json(Path(pointer["path"]), pointer["sha256"])
        if (
            raw.get("schema_version") != "gx1_candidate_training_session_v1"
            or raw.get("state_sha256") != checkpoint["sha256"]
            or type(raw.get("slot")) is not int or raw["slot"] not in (0, 1)
            or Path(checkpoint["path"]).name != f"candidate_training_state_slot_{raw['slot']}.pt"
            or any(raw.get(key) != state[key] for key in (
                "session_contract_sha256", "phase", "epoch_index", "next_batch_offset",
                "global_optimizer_steps", "complete",
            ))
        ):
            raise RuntimeError("NATIVE_CANDIDATE_CANONICAL_POINTER_MISMATCH")
    return dict(value)


def build_native_cursor(
    *, recipe: Mapping[str, str], resume_state: Mapping[str, Any], outcome: str,
) -> dict[str, Any]:
    cursor = {
        "schema_version": CURSOR_SCHEMA, "recipe": dict(recipe),
        "resume_state": dict(resume_state), "outcome": outcome, "test_data_used": False,
    }
    cursor["cursor_sha256"] = canonical_sha256(cursor)
    return require_native_cursor(cursor, expected_recipe=recipe)


OBSERVED_ENTRY_RECIPE_KEYS = ("entry_observed_market", "entry_observed_market_horizons")


def _observed_entry_in(recipe) -> bool:
    return any(key in recipe for key in OBSERVED_ENTRY_RECIPE_KEYS)


def _expected_observed_entry_binding(recipe, expected: dict) -> None:
    """The policy block binds the exact design under its own recipe key (v1 or horizon successor)."""
    from gx1.contracts.entry_observed_market_horizons_v1 import require_observed_entry_scope
    scope = require_observed_entry_scope(recipe)
    if scope is not None:
        key = [k for k in OBSERVED_ENTRY_RECIPE_KEYS if k in recipe][0]
        expected[key] = scope["binding"]
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        if scope.get("training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
            from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY
            expected[DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY] = require_binding(
                recipe.get(DYNAMIC_M5_LOSS_SCALE_RECIPE_KEY), label="dynamic loss-scale policy binding", verify_file=False)


def require_entry_learning_study_run(recipe, *, invocation_number=None, execution_budget=None):
    """Admit the separately authorized finite study; retain every native guard."""
    import numpy as np
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        RECENT_WINDOW_PHASE, recent_window_phase_spec, require_observed_entry_scope,
    )
    selection = recipe.get("entry_learning_study")
    def _phase_name_ok(name):
        import re as _re
        return name in ("fit", "curve") or (isinstance(name, str) and _re.fullmatch(r"recent_[a-z0-9_]{1,40}", name))
    if (not isinstance(selection, Mapping) or set(selection) != {"plan", "phase"}
            or not _phase_name_ok(selection.get("phase"))
            or any(k in recipe for k in ("chronological_initial_measurement",
                "chronological_learning_measurement", "chronological_learning_continuation",
                "chronological_train_only_measurement", "chronological_entry_baseline",
                "candidate_resume_origin", "native_calibration", "entry_gradient_diagnostic",
                "frozen_readout_evaluation", "frozen_train_policy_evaluation", "frozen_entry_selector_probe"))):
        raise RuntimeError("ENTRY_STUDY_MIXED_SCOPE")
    binding = require_binding(selection["plan"], label="Entry learning study", verify_file=True)
    plan = read_bound_json(Path(binding["path"]), binding["sha256"])
    phase = selection["phase"]
    schema = plan.get("schema_version")
    # v1: the 95-minute observed target on the 254-field surface (ENTRY_LEARNING_STUDY_20261010);
    # v2: the 20-trading-day observed target on the NATIVE_D1 surface (operator decision 12.10.2026,
    # minimum scope: one seed, bounded 1024/4096-step phases, control drawn from complete rows only).
    if schema == "gx1_entry_learning_study_v1":
        design_key, model_spec = "entry_observed_market", {"unchanged_architecture":True, "signal_fields":254,
            "feature_families":8, "fresh_initialization_each_phase":True,
            "optimizer_and_loss_unchanged":True, "seed":20260911}
        fixed_phases = {
            "fit": {"optimizer_steps":1024, "train_observation_steps":[0,64,256,1024],
                    "control_observation_steps":[], "max_invocations":1, "unique_train_rows":256},
            "curve": {"optimizer_steps":16384, "train_observation_steps":[0,256,1024,4096,8192,16384],
                      "control_observation_steps":[0,16384], "max_invocations":2,
                      "unique_train_rows":262144}}
    elif schema == "gx1_entry_learning_study_v2":
        design_key, model_spec = "entry_observed_market_horizons", {"unchanged_architecture":True, "signal_fields":315,
            "context_fields":132, "feature_families":8, "context_families":9, "fresh_initialization_each_phase":True,
            "optimizer_and_loss_unchanged":True, "seed":20260911, "loss_horizon_trading_days":20}
        from gx1.contracts.entry_observed_market_horizons_v1 import MINIMUM_PLAN_FIT_OPTIMIZER_STEPS
        fixed_phases = {
            "fit": {"optimizer_steps":MINIMUM_PLAN_FIT_OPTIMIZER_STEPS, "train_observation_steps":[0,64,256,1024],
                    "control_observation_steps":[], "max_invocations":1, "unique_train_rows":256},
            "curve": {"optimizer_steps":4096, "train_observation_steps":[0,256,1024,4096],
                      "control_observation_steps":[0,4096], "max_invocations":1,
                      "unique_train_rows":65536}}
    else:
        raise RuntimeError("ENTRY_STUDY_INPUT_OR_OBJECTIVE_CHANGED")
    # Operator decision 14.10.2026: a v2 successor spec may add the recent-window phase. It binds its
    # predecessor by hash and repeats the completed fit/curve phases byte for byte; the window phase and
    # its rows are derived from the frozen eligible population and verified below.
    window_phases = set(plan.get("phases", {})) - {"fit", "curve"} if schema == "gx1_entry_learning_study_v2" else set()
    if window_phases:
        import re as _re
        if any(not _re.fullmatch(r"recent_[a-z0-9_]{1,40}", name) for name in window_phases):
            raise RuntimeError("ENTRY_STUDY_PHASE_INVALID")
        predecessor_binding = require_binding(plan.get("predecessor"), label="Entry study predecessor", verify_file=True)
        predecessor = read_bound_json(Path(predecessor_binding["path"]), predecessor_binding["sha256"])
        if (predecessor.get("schema_version") != schema or set(predecessor.get("phases", {})) != {"fit", "curve"}
                or any(predecessor["phases"][k] != plan["phases"].get(k) for k in ("fit", "curve"))
                or predecessor.get("row_bindings", {}).get("epoch_order") != plan.get("row_bindings", {}).get("epoch_order")
                or any(predecessor.get(k) != plan.get(k) for k in ("files", "chronological_prefix", design_key, "model"))):
            raise RuntimeError("ENTRY_STUDY_PREDECESSOR_CHANGED")
    elif phase not in ("fit", "curve"):
        raise RuntimeError("ENTRY_STUDY_PHASE_INVALID")
    if phase not in ("fit", "curve") and phase not in window_phases:
        raise RuntimeError("ENTRY_STUDY_PHASE_INVALID")
    other_key = [k for k in OBSERVED_ENTRY_RECIPE_KEYS if k != design_key][0]
    if (plan.get("files") != recipe.get("files")
            or plan.get("chronological_prefix") != recipe.get("chronological_prefix")
            or plan.get(design_key) != recipe.get(design_key) or design_key not in recipe or other_key in recipe
            or recipe.get("initialization") != "fresh_existing_model_constructor_no_checkpoint_weights"
            or set(plan.get("phases", {})) != {"fit", "curve"} | window_phases
            or plan["execution"].get("test_data_used") is not False
            or any(plan["execution"].get(k) is not False for k in (
                "broker_access", "spending", "horizon_changes", "full_epoch_training_allowed",
                "full_val_allowed", "exit_training_authorized_without_entry_qualification"))
            or plan["model"] != model_spec):
        raise RuntimeError("ENTRY_STUDY_INPUT_OR_OBJECTIVE_CHANGED")
    prefix = require_chronological_prefix_recipe(recipe)
    observed = require_observed_entry_scope(recipe)
    rows = {}
    for name, b in plan["row_bindings"].items():
        checked = require_binding(b, label="Entry study rows " + name, verify_file=True)
        a = np.load(checked["path"], allow_pickle=False)
        if a.dtype != np.dtype("int64") or a.ndim != 1 or len(a) < 1 or np.any(a < 0):
            raise RuntimeError("ENTRY_STUDY_ROWS_INVALID")
        rows[name] = a
    n = observed["physical_sources"]["train"]["physical_rows"]
    nv = observed["physical_sources"]["val"]["physical_rows"]
    if schema == "gx1_entry_learning_study_v1":
        expected_rows = {"epoch_order", "fit", "train_probe", "control"}
        train_population = np.arange(n)
        control_expected = np.linspace(0,nv-1,4096,dtype=np.int64)
    else:
        # The epoch order covers exactly the frozen eligible TRAIN rows (complete 20-day outcomes);
        # control is a deterministic uniform subsample of the bound complete VAL population.
        expected_rows = {"epoch_order", "fit", "train_probe", "control", "control_population"}
        train_population = np.asarray(prefix["physical_coordinates"]["eligible_parent_rows"], dtype=np.int64)
        population = rows.get("control_population", np.zeros(0, dtype=np.int64))
        if (len(population) < 4096 or not np.array_equal(population, np.unique(population))
                or population[-1] >= nv
                or not np.array_equal(population, _complete_horizon_control_rows(observed))):
            raise RuntimeError("ENTRY_STUDY_CONTROL_POPULATION_INVALID")
        control_expected = population[np.linspace(0, len(population)-1, 4096, dtype=np.int64)]
    from gx1.contracts.entry_observed_market_horizons_v1 import allowed_prefix_training_seeds
    base_seed = fixed_phases["fit"].get("seed", model_spec["seed"]) if False else model_spec["seed"]
    for name in sorted(window_phases):
        expected_rows |= {name, name + "_train_probe"}
        window_spec = plan["phases"][name]
        window = _recent_window_train_rows(observed, train_population, window_days=window_spec.get("window_days"))
        in_order = rows["epoch_order"][np.isin(rows["epoch_order"], window)]
        if (not np.array_equal(rows.get(name, np.zeros(0, dtype=np.int64)), window)
                or len(in_order) != len(window)
                or not np.array_equal(rows.get(name + "_train_probe", np.zeros(0, dtype=np.int64)),
                                      np.sort(in_order[:4096]))):
            raise RuntimeError("ENTRY_STUDY_RECENT_WINDOW_ROWS_INVALID")
        fixed = recent_window_phase_spec(int(len(window)), window_days=window_spec["window_days"])
        if "seed" in window_spec:
            # a declared per-phase seed must be the design seed or a named variant (one origin: the contract constant)
            if window_spec["seed"] not in allowed_prefix_training_seeds(base_seed):
                raise RuntimeError("ENTRY_STUDY_SEED_NOT_DECLARED")
            fixed["seed"] = window_spec["seed"]
        fixed_phases[name] = fixed
    # the recipe's training seed is the selected phase's declared seed (the design seed when undeclared);
    # v1 studies predate the trainer_cli binding in this gate and carry no window phases
    if schema == "gx1_entry_learning_study_v2":
        selected_seed = fixed_phases.get(phase, {}).get("seed", base_seed)
        if recipe.get("trainer_cli", {}).get("seed", base_seed) != selected_seed:
            raise RuntimeError("ENTRY_STUDY_SEED_NOT_DECLARED")
    if (set(rows) != expected_rows
            or not np.array_equal(np.sort(rows["epoch_order"]), train_population)
            or len(rows["fit"]) != 256 or len(np.unique(rows["fit"])) != 256
            or not np.isin(rows["fit"], rows["epoch_order"][:4096]).all()
            or not np.array_equal(rows["train_probe"], np.sort(rows["epoch_order"][:4096]))
            or not np.array_equal(rows["control"], control_expected)
            or plan["row_bindings"]["epoch_order"]
               != prefix["physical_coordinates"]["artifact"]["bindings"]["TRAIN_NATIVE_EPOCH0_ORDER"]):
        raise RuntimeError("ENTRY_STUDY_POPULATION_CHANGED")
    phase_spec = plan["phases"][phase]
    fixed = fixed_phases[phase]
    if (set(phase_spec) != set(fixed) | {"run_id"}
            or any(type(phase_spec.get(k)) is not type(v) or phase_spec.get(k) != v for k,v in fixed.items())
            or recipe.get("run_id") != phase_spec["run_id"]
            or recipe.get("out_bundle_dir") != str(Path(plan["root"]) / phase.upper() / "CANDIDATE_BUNDLE")):
        raise RuntimeError("ENTRY_STUDY_FINITE_BUDGET_CHANGED")
    repo = Path(__file__).resolve().parents[2]
    pb = require_binding(recipe.get("next_run_policy"), label="Entry study policy", verify_file=True)
    if Path(pb["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(pb["path"]), pb["sha256"])
    _require_native_profile_and_economics(recipe, policy, repo)
    expected = {"selection":selection, "run_id":recipe["run_id"],
        "out_bundle_dir":recipe["out_bundle_dir"],
        "source_bindings_sha256":recipe["source_bindings_sha256"],
        "optimizer_steps":phase_spec["optimizer_steps"],
        "max_invocations":phase_spec["max_invocations"], "test_data_used":False}
    if (policy.get("training_enabled") is not False
            or policy.get("full_epoch_training_allowed") is not False
            or policy.get("full_val_allowed") is not False
            or policy.get("entry_learning_study_run") != expected
            or any(k in policy for k in ("chronological_learning_run", "chronological_initial_measurement",
                "native_learning_calibration", "frozen_readout_evaluation"))):
        raise RuntimeError("ENTRY_STUDY_NOT_AUTHORIZED")
    if invocation_number is not None and (
            type(invocation_number) is not int or not 1 <= invocation_number <= phase_spec["max_invocations"]):
        raise RuntimeError("ENTRY_STUDY_INVOCATION_INVALID")
    if execution_budget is not None and (
            type(execution_budget.get("stop_after_optimizer_steps")) is not int
            or execution_budget["stop_after_optimizer_steps"] != phase_spec["optimizer_steps"]
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("ENTRY_STUDY_EXECUTION_BUDGET_INVALID")
    return {"selection":selection, "plan":plan, "phase":phase, "phase_spec":phase_spec,
            "rows":rows, "observed_scope":observed}


def _recent_window_train_rows(observed_scope, eligible_parent_rows, *, window_days):
    """Eligible TRAIN rows whose decision lies in the last window_days before the measured TRAIN
    eligibility cutoff (sorted parent rows); the clock is read from the bound physical TRAIN parquet."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS
    from gx1.contracts.entry_observed_market_horizons_v1 import recent_window_rows
    eligible = np.asarray(eligible_parent_rows, dtype=np.int64)
    source = observed_scope["physical_sources"]["train"]
    frame = pq.read_table(source["parquet"]["path"], columns=["time"]).to_pandas()
    clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
    if len(clock) != source["physical_rows"] or len(eligible) < 1 or eligible[-1] >= len(clock):
        raise RuntimeError("ENTRY_STUDY_RECENT_WINDOW_SOURCE_INVALID")
    decision = clock[eligible] + ENTRY_DECISION_BAR_SECONDS * 10**9
    cutoff = pd.Timestamp(observed_scope["entry_eligibility_cutoff_utc"]["train"]).value
    return eligible[recent_window_rows(decision, cutoff_ns=cutoff, window_days=window_days)]


def _complete_horizon_control_rows(observed_scope):
    """VAL rows with complete outcomes at all three horizons whose decision lies at or before the
    measured eligibility cutoff (censored rows are excluded, never filled; a complete row after
    the cutoff contradicts the measured design)."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS
    from gx1.contracts.entry_observed_market_horizons_v1 import HORIZON_LABEL_COLUMNS
    source = observed_scope["physical_sources"]["val"]
    frame = pq.read_table(source["parquet"]["path"], columns=["time", *HORIZON_LABEL_COLUMNS]).to_pandas()
    clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
    eligible = clock + ENTRY_DECISION_BAR_SECONDS * 10**9 <= pd.Timestamp(observed_scope["entry_eligibility_cutoff_utc"]["val"]).value
    complete = np.isfinite(frame[HORIZON_LABEL_COLUMNS].to_numpy(dtype=np.float64)).all(axis=1)
    if bool((complete & ~eligible).any()):
        raise RuntimeError("ENTRY_STUDY_CONTROL_CUTOFF_TOO_EARLY")
    return np.flatnonzero(eligible & complete).astype(np.int64)
