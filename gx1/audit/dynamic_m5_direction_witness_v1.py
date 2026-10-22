"""One fixed-TRAIN eval graph: direction/horizon/task gradients, no updates."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

from gx1.audit.dynamic_compact_input_influence_v1 import (
    FLAGS, REPO, _identity, _path, _publish, _read, _reference, _rng_sha, _sha,
)

PLAN_SCHEMA = "gx1_dynamic_m5_direction_witness_plan_v1"
PLAN_SCHEMA_V2 = "gx1_dynamic_m5_direction_witness_plan_v2"
SCOPE_SCHEMA = "gx1_dynamic_m5_direction_witness_scope_v1"
SCOPE_SCHEMA_V2 = "gx1_dynamic_m5_direction_witness_scope_v2"
BINDING_SCHEMA = "gx1_dynamic_m5_direction_witness_binding_v1"
_INPUTS = ("seq_x", "snap_x", "ctx_cont", "ctx_cat", "seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")
_COMPONENTS = ("dynamic_terminal_net_markout_bps", "dynamic_mfe_bps",
               "dynamic_pre_mfe_mae_upper_bps", "dynamic_closed_mfe_bps",
               "dynamic_closed_pre_mfe_mae_upper_bps", "forecast_pred")
_REPORT_HORIZONS = (1, 5, 15, 60, 120, 240, 480)


def _require(value, reason):
    if not value:
        raise RuntimeError("DYNAMIC_M5_DIRECTION_WITNESS_" + reason)


def _require_plan(plan):
    keys = {"schema_version", "run_id", "policy_stage_key", "input_scope",
            "output_directory", "one_shot_invocations", "max_wall_seconds",
            "max_forward_calls", "max_reverse_calls", "headline_horizons", *FLAGS}
    _require(isinstance(plan, dict) and set(plan) == keys
             and plan["schema_version"] in (PLAN_SCHEMA, PLAN_SCHEMA_V2), "PLAN_SCHEMA")
    _require(type(plan["one_shot_invocations"]) is int and plan["one_shot_invocations"] == 1
             and type(plan["max_forward_calls"]) is int and plan["max_forward_calls"] == 1
             and type(plan["max_reverse_calls"]) is int and plan["max_reverse_calls"] == (16 if plan["schema_version"] == PLAN_SCHEMA_V2 else 22)
             and type(plan["max_wall_seconds"]) is int and 0 < plan["max_wall_seconds"] <= (900 if plan["schema_version"] == PLAN_SCHEMA_V2 else 3500)
             and plan["headline_horizons"] == list(_REPORT_HORIZONS)
             and all(plan[key] is False for key in FLAGS), "PLAN_BUDGET_OR_AUTHORITY")
    _require(isinstance(plan["run_id"], str) and bool(plan["run_id"])
             and isinstance(plan["policy_stage_key"], str) and bool(plan["policy_stage_key"]), "PLAN_IDENTITY")


def _preflight(plan_path, plan_sha256, device, out_dir):
    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    proofs = []
    plan = _read(plan_ref, proofs=proofs)
    _require_plan(plan)
    output = Path(out_dir)
    _require(device == "cuda" and output.is_absolute() and output.resolve() == output
             and str(output) == plan["output_directory"] and not output.exists()
             and output.parent.is_dir() and not any(p.is_symlink() for p in output.parents), "FRESH_GUARDED_OUTPUT")
    policy_path = REPO / "NEXT_RUN_POLICY.json"
    policy_ref = {"path": str(policy_path), "sha256": _sha(policy_path)}
    policy_identity = _identity(policy_path)
    policy = _read(policy_ref, _policy_history=True)
    stage = policy.get(plan["policy_stage_key"])
    _require(all(policy.get(k) is False for k in ("training_enabled", "full_epoch_training_allowed", "full_val_allowed"))
             and isinstance(stage, dict) and stage.get("execution_authorized") is True
             and stage.get("plan") == plan_ref and stage.get("planned_invocations") == 1
             and stage.get("completed_invocations") == 0 and stage.get("relaunch_allowed") is False
             and all(stage.get(k) is False for k in FLAGS), "EXACT_POLICY")
    binding_path = _path(plan_path.parent / "DIRECTION_WITNESS_BINDING.json")
    binding_ref = {"path": str(binding_path), "sha256": _sha(binding_path)}
    binding = _read(binding_ref, proofs=proofs)
    _require(set(binding) == {"schema_version", "plan", "source_head", "policy_sha256", "source_bindings"}
             and binding["schema_version"] == BINDING_SCHEMA and binding["plan"] == plan_ref
             and binding["policy_sha256"] == policy_ref["sha256"], "POSTCOMMIT_BINDING")
    sources = _read(binding["source_bindings"], proofs=proofs)
    _require(set(sources) == {"source_head", "files"} and sources["source_head"] == binding["source_head"]
             and isinstance(sources["files"], dict) and bool(sources["files"]), "SOURCES")
    scope = _read(plan["input_scope"], proofs=proofs)
    scope_keys = {"schema_version", "input_packet", "input_packet_metadata",
        "input_preparation_result", "input_preparation_terminal", "constructor_metadata",
        "normalization", "loss_scales", "session", "final_online_result", "packet_layout",
        "helper", "executor"}
    native170 = plan["schema_version"] == PLAN_SCHEMA_V2
    if native170:
        scope_keys |= {"design", "packet_origin", "input_contract_equivalence",
                       "parity_reference", "canonical_pointer_proof"}
    _require(set(scope) == scope_keys and scope["schema_version"] ==
             (SCOPE_SCHEMA_V2 if native170 else SCOPE_SCHEMA), "SCOPE")
    _require(_reference(scope["executor"]) == Path(__file__).resolve()
             and _reference(scope["helper"]) == REPO / "gx1/contracts/entry_dynamic_compact_input_influence_v1.py", "OWNERS")
    for role in ("helper", "executor"):
        record = sources["files"].get(scope[role]["path"])
        _require(isinstance(record, dict) and record["sha256"] == scope[role]["sha256"], "OWNER_SOURCE_JOIN")
    prep_terminal = _read(scope["input_preparation_terminal"], proofs=proofs)
    prep = _read(scope["input_preparation_result"], proofs=proofs)
    _require(prep_terminal["status"] == "COMPLETE" and prep_terminal["exit_code"] == 0
             and prep_terminal["source_unchanged"] is True
             and prep_terminal["result"] == scope["input_preparation_result"]
             and prep_terminal["relaunch_allowed"] is False, "INPUT_PREPARATION_TERMINAL")
    _require(prep["input_packet"] == scope["input_packet"]
             and prep["metadata"] == scope["input_packet_metadata"]
             and prep["genuine_rows"] == 16 and prep["padding_rows"] == 0
             and prep["model_forwards"] == prep["optimizer_updates"] == 0
             and prep["test_data_used"] is False, "INPUT_PREPARATION_RESULT")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    def frozen():
        _require(git("branch", "--show-current") == "work/gx1-current"
                 and git("rev-parse", "HEAD") == binding["source_head"]
                 and not git("status", "--porcelain"), "SOURCE_CHANGED")
        _require(_identity(policy_path) == policy_identity and _sha(policy_path) == policy_ref["sha256"], "POLICY_CHANGED")
        _require(all(_identity(_reference(ref)) == ident for ref, ident in proofs), "INPUT_METADATA_CHANGED")
        for name, record in sources["files"].items():
            path = _path(name)
            _require(path.is_relative_to(REPO) and _identity(path) == record["identity"]
                     and _sha(path) == record["sha256"] and _identity(path) == record["identity"], "SOURCE_FILE_CHANGED")
    frozen()
    return plan, plan_ref, scope, output, frozen


def _packet(scope, metadata):
    import numpy as np
    import torch
    path = _reference(scope["input_packet"])
    before = _identity(path)
    _require(before["st_size"] <= 64 * 1024 * 1024, "PACKET_SIZE")
    raw = path.read_bytes()
    _require(_identity(path) == before and hashlib.sha256(raw).hexdigest() == scope["input_packet"]["sha256"], "PACKET_SHA")
    layout = scope["packet_layout"]
    _require(set(layout) == {"inputs", "batch", "reference_predictions"}
             and set(layout["inputs"]) == set(_INPUTS)
             and set(layout["reference_predictions"]) == set(_COMPONENTS)
             and all(isinstance(v, dict) for v in layout.values()), "PACKET_LAYOUT")
    array_names = [name for group in layout.values() for name in group.values()]
    _require(len(array_names) == len(set(array_names)) and metadata["packet_layout"] == layout, "PACKET_ROLE_ALIAS")
    specs = metadata["arrays"]
    values = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as packet:
        _require(set(packet.files) == set(array_names) == set(specs), "PACKET_INVENTORY")
        for name in array_names:
            value = np.asarray(packet[name])
            spec = specs[name]
            _require(value.dtype.kind in "biuf" and str(value.dtype) == spec["dtype"]
                     and list(value.shape) == spec["shape"] and value.shape[0] == 16
                     and bool(np.isfinite(value).all())
                     and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == spec["sha256"], "PACKET_ARRAY:" + name)
            values[name] = torch.from_numpy(np.array(value, copy=True, order="C"))
    inputs = {key: values[name] for key, name in layout["inputs"].items()}
    batch = {key: values[name] for key, name in layout["batch"].items()}
    references = {key: values[name] for key, name in layout["reference_predictions"].items()}
    return inputs, batch, references, before


def _nest(flat):
    return {**{key: flat[key] for key in ("seq_x", "snap_x", "ctx_cont", "ctx_cat")},
            "mtf": {key: flat[key] for key in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")}}




def _require_native170_physical_batch(batch):
    """The fixed179 physical budget needs real observations in both groups."""
    import torch
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import DYNAMIC_M5_TERMINAL_HORIZON_GROUPS
    from gx1.contracts.entry_observed_market_v1 import DYNAMIC_M5_MARKOUT_VALID_KEY
    valid = batch.get(DYNAMIC_M5_MARKOUT_VALID_KEY)
    _require(isinstance(valid, torch.Tensor) and valid.dtype == torch.bool
        and tuple(valid.shape) == (16, 480, 3)
        and torch.equal(valid[..., 2], valid[..., 0] & valid[..., 1]), "PHYSICAL_TERMINAL_MASK")
    counts = {name: int(valid[:, start - 1:end].sum())
              for name, start, end in DYNAMIC_M5_TERMINAL_HORIZON_GROUPS}
    _require(all(count > 0 for count in counts.values()), "PHYSICAL_TERMINAL_GROUP_COVERAGE")
    return counts

def _read_saved_train_reference(reference, *, proofs):
    """The sole v2 saved-observation read: one bound <=64MiB TRAIN JSON."""
    path = _reference(reference)
    before = _identity(path)
    _require(path.name == "TRAIN_OBSERVATION.json"
             and before["st_size"] <= 64 * 1024 * 1024, "SAVED_TRAIN_JSON_BOUND")
    raw = path.read_bytes()
    _require(_identity(path) == before and hashlib.sha256(raw).hexdigest() == reference["sha256"],
             "SAVED_TRAIN_JSON_SHA")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "SAVED_TRAIN_DUPLICATE_KEY")
            result[key] = value
        return result
    def invalid(value):
        raise RuntimeError("DYNAMIC_M5_DIRECTION_WITNESS_SAVED_TRAIN_NONFINITE_JSON:" + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _require(isinstance(result, dict), "SAVED_TRAIN_JSON_OBJECT")
    proofs.append((dict(reference), before))
    return result


def _require_v2_origin(scope, metadata, constructor, origin_constructor, final, design, equivalence):
    origin = scope["packet_origin"]
    _require(isinstance(origin, dict) and set(origin) == {
        "constructor_metadata", "final_online_result", "online_model_state_sha256"}, "PACKET_ORIGIN")
    _require(metadata["constructor_metadata"] == origin["constructor_metadata"]
        and metadata["final_online_result"] == origin["final_online_result"]
        and metadata["reference_online_model_state_sha256"] == origin["online_model_state_sha256"]
        and metadata["normalization"] == scope["normalization"]
        and metadata["loss_scales"] == scope["loss_scales"]
        and metadata["genuine_rows"] == 16 and metadata["padding_rows"] == 0, "PACKET_ORIGIN_BINDING")
    _require(design["schema_version"] == "gx1_trained_terminal_balance_gradient_witness_design_v1"
        and design["execution_authorized"] is False
        and design["execution_constructor"] == scope["constructor_metadata"]
        and design["origin_constructor"] == origin["constructor_metadata"]
        and design["state"]["final_online_result"] == scope["final_online_result"]
        and design["state"]["online_model_state_sha256"] == scope["session"]["online_model_state_sha256"]
        and design["parity_reference"] == scope["parity_reference"]
        and design["inputs"]["packet"] == scope["input_packet"]
        and design["inputs"]["metadata"] == scope["input_packet_metadata"]
        and design["inputs"]["input_sha256"] == metadata["input_sha256"]
        and design["inputs"]["target_sha256"] == metadata["target_sha256"]
        and final["observations"]["train"] == scope["parity_reference"], "V2_DESIGN_BINDING")
    proof = equivalence["physical_contract_metadata_proof"]
    fields = ("architecture_control_source", "architecture_scalar_fields", "ctx_cat_dim", "ctx_cont_dim",
        "current_input_sources", "dropout", "entry_execution_profile", "fits", "multi_tf",
        "seq_input_dim", "seq_len", "snap_input_dim", "specialist_fusion")
    _require(equivalence["schema_version"] == "gx1_trained_terminal_balance_gradient_witness_preparation_route_v1"
        and proof["packet_origin_constructor147"] == origin["constructor_metadata"]
        and proof["execution_constructor172"] == scope["constructor_metadata"]
        and all(proof[key] is True for key in (
            "same055_normalization", "same057_native_coordinates",
            "same111_artifact_contract_fitlineage_identity", "same_physical_train_ref",
            "same_ten_physical_file_refs_excluding_constructor",
            "same_chronological_prefix_057_observed104_population_witness_profile_dataset111"))
        and all(constructor[key] == origin_constructor[key] for key in fields), "INPUT_CONTRACT_EQUIVALENCE")


def _require_completed_pointer_proof(scope, final, documents, cohort):
    refs = scope["canonical_pointer_proof"]
    _require(isinstance(refs, dict) and set(refs) == {"plan", "terminal", "actual_peer", "result"},
             "POINTER_PROOF_SCOPE")
    plan, terminal, peer = (documents[k] for k in ("plan", "terminal", "actual_peer"))
    _require(plan["schema_version"] == "gx1_saved_direction_comparison_plan_v1"
        and plan["results"]["current_final"] == scope["final_online_result"]
        and plan["loss_scales"] == scope["loss_scales"]
        and terminal["schema_version"] == "gx1_saved_direction_comparison_terminal_v1"
        and terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
        and terminal["source_unchanged"] is True and terminal["test_data_used"] is False
        and terminal["relaunch_allowed"] is False and terminal["result"] == refs["result"]
        and terminal["canonical_parent_pointer_projection_count"] == 6, "COMPLETED_POINTER_TERMINAL")
    alignment = peer["alignment_review"]
    _require(peer["schema_version"] == "gx1_actual_saved_direction_comparison_peer_review_v1"
        and peer["status"] == "PASS_STORED_COMPARISON_INTEGRITY_WITH_PRIMARY_ENTRY_QUALITY_UNQUALIFIED"
        and peer["unresolved_integrity_findings"] == []
        and peer["references"]["PLAN.json"] == refs["plan"]
        and peer["references"]["TERMINAL.json"] == refs["terminal"]
        and peer["references"]["RESULTS.json"] == refs["result"]
        and peer["execution"]["canonical_parent_pointer_projections"] == 6
        and peer["execution"]["terminal_complete_exit0"] is True
        and alignment["role_entries"]["train"] == 256
        and all(alignment[k] is True for k in (
            "actual_six_curve_references_equal_bound_three_version_two_role_metadata",
            "six_pointer_declarations_equal_PLAN",
            "full480_and_existing_canonical_153_152_target_storage_nullable_masks_clock_order_checks_passed_in_completed_worker",
            "cohort_nullable_target_mask_forecast_fields_exact_against_saved168_reference167_bridge",
            "original111_RMS_identity_same_all_versions")), "COMPLETED_POINTER_PEER")
    projection = {"variant": "current_final", "role": "train",
        "path": cohort["parent_entry_parquet"]["path"], "columns": ["dynamic_entry_m1_row"]}
    _require(len(plan["pointer_projections"]) == 6 and plan["pointer_projections"].count(projection) == 1
        and final["observations"]["train"] == scope["parity_reference"], "COMPLETED_POINTER_REFERENCE")
    return {"completed_canonical_projections_reused": 6, "result_reference": refs["result"],
            "result_payload_read_or_rehashed": False}


def _join_saved_train_reference(observation, *, scope, metadata, batch, final, pointer_documents):
    """Pure in-memory joins after the one admitted saved-TRAIN read; no Parquet."""
    import math
    import numpy as np
    import torch
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        MODEL_NATIVE_AUX_FORECAST_HORIZONS, MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT,
    )
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
    )
    _require(observation["role"] == "train" and observation["optimizer_steps"] == final["optimizer_steps"] == 7391
        and observation["model_state_sha256"] == final["model_state_sha256"] ==
            scope["session"]["online_model_state_sha256"]
        and observation["target_model_state_sha256"] == final["target_model_state_sha256"]
        and observation["target_model_role"] == final["target_model_role"] == "frozen_constructor_anchor_no_forward"
        and observation["exit_evaluation"] is observation["native_launch_authorized"] is False,
        "REFERENCE_STATE_ROLE")
    cohort = observation["cohort"]
    proof = _require_completed_pointer_proof(scope, final, pointer_documents, cohort)
    parents, clocks = cohort["parent_entry_row_indices"], cohort["entry_time_ns"]
    _require(len(parents) == len(set(parents)) == len(clocks) == 256
        and parents[:16] == metadata["parent_rows"] and clocks[:16] == metadata["entry_time_ns"]
        and batch["dynamic_entry_parent_row_index"].tolist() == parents[:16]
        and batch["entry_row_index"].tolist() == parents[:16], "REFERENCE_COHORT_ORDER")
    diagnostics = observation["diagnostics"]
    frozen = diagnostics["dynamic_frozen_targets"]
    traces = diagnostics["dynamic_entry_observations"]
    queries = list(range(1, 481))
    _require(frozen["parent_entry_rows"] == parents and frozen["entry_time_ns"] == clocks
        and frozen["entry_supervision_identity"] == final["entry_observed_market"]
        and frozen["query_minutes"] == queries and len(frozen["dynamic_entry_m1_rows"]) == 256
        and frozen["dynamic_entry_m1_rows"][:16] == batch["dynamic_entry_m1_row"].tolist()
        and len(traces) == 256
        and diagnostics["dynamic_curve_statistics"]["dynamic_m5_loss_scales"] == final["dynamic_m5_loss_scales"]
        and diagnostics["dynamic_curve_statistics"]["dynamic_loss_residual_units"] == "TRAIN_target_RMS_units",
        "REFERENCE_FROZEN_IDENTITY")
    base = torch.stack([batch["y_forecast_ret_K" + str(h)] for h in MODEL_NATIVE_AUX_FORECAST_HORIZONS], dim=1)
    targets = {name: batch["y_" + name] for name in _COMPONENTS[:-1]}
    targets["forecast_pred"] = torch.cat((base, batch[SUPPORT_FORECAST_TARGET_BATCH_KEY]), dim=1)
    masks = dict(zip(_COMPONENTS, (
        batch["dynamic_terminal_markout_observed"], batch["dynamic_path_observed"],
        batch["dynamic_path_observed"], batch["dynamic_closed_path_observed"],
        batch["dynamic_closed_path_observed"],
        torch.cat((batch[M5_FORECAST_VALID_BATCH_KEY], batch[SUPPORT_FORECAST_VALID_BATCH_KEY]), dim=1))))
    for name in _COMPONENTS:
        storage, original = frozen["components"][name], metadata["original_target_storage"][name]
        _require(storage["target_dtype"] == "float32" and storage["mask_dtype"] == "bool"
            and storage["shape"] == [256, *targets[name].shape[1:]]
            and storage["target_storage_sha256"] == original["original_target_storage_sha256"]
            and storage["observed_mask_sha256"] == original["original_observed_mask_sha256"]
            and original["comparison_scope_rows"] == 256, "REFERENCE_STORAGE:" + name)
    references = {name: [] for name in _COMPONENTS}
    for row, trace in enumerate(traces[:16]):
        _require(trace["entry_row_index"] == parents[row] and trace["entry_time_ns"] == clocks[row]
            and trace["full_query_curves"]["query_minutes"] == queries
            and trace["forecast_support"]["output_layout"] == list(MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT),
            "REFERENCE_TRACE_ORDER")
        for name in _COMPONENTS:
            cell = trace["forecast_support"] if name == "forecast_pred" else trace["full_query_curves"]["components"][name]
            shape = tuple(targets[name].shape[1:])
            prediction = np.asarray(cell["prediction_bps"], dtype=object)
            target = np.asarray(cell["target_bps"], dtype=object)
            mask = np.asarray(cell["observed"], dtype=object)
            number = lambda x: type(x) in (int, float) and math.isfinite(x)
            _require(prediction.shape == target.shape == mask.shape == shape
                and all(number(x) for x in prediction.flat)
                and all(type(x) is bool for x in mask.flat)
                and all(number(x) if observed else x is None for x, observed in zip(target.flat, mask.flat)),
                "REFERENCE_TRACE_DOMAIN:" + name)
            if name != "forecast_pred":
                _require(cell["action_order"] == ["LONG", "SHORT", *(["FLAT"] if name == _COMPONENTS[0] else [])],
                         "REFERENCE_ACTION_ORDER")
            observed = mask.astype(np.bool_)
            own_mask = masks[name][row].detach().cpu().numpy()
            target_f32 = np.where(observed, target, 0).astype(np.float32)
            own_target = targets[name][row].detach().cpu().numpy()
            _require(np.array_equal(observed, own_mask)
                and np.ascontiguousarray(target_f32[observed]).tobytes() ==
                    np.ascontiguousarray(own_target[observed]).tobytes(), "REFERENCE_TARGET_MASK:" + name)
            values = prediction.astype(np.float32)
            _require(bool(np.isfinite(values).all()), "REFERENCE_F32_FINITE:" + name)
            references[name].append(torch.from_numpy(np.array(values, copy=True, order="C")))
    return {name: torch.stack(values) for name, values in references.items()}, {
        **proof, "saved_train_observation": scope["parity_reference"], "genuine_rows": 16,
        "parent_rows": parents[:16], "entry_time_ns": clocks[:16],
        "full256_original_target_storage_hashes_matched": True,
        "observed_targets_and_masks_exact_in_memory": True, "global_m1_pointer_join_exact": True,
        "nullable_masked_backing_reconstructed": False,
        "packet_target_sha256": metadata["target_sha256"],
        "packet_reference_prediction_role": "immutable151_origin_only_not177_parity_truth",
    }

def run(*, plan_path: Path, plan_sha256: str, device: str, out_dir: Path) -> dict:
    started = time.monotonic()
    plan, plan_ref, scope, output, frozen = _preflight(plan_path, plan_sha256, device, out_dir)
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    hardware, lock = require_guarded_cuda_producer_execution(), require_capped_lock_ancestry()
    import torch
    from gx1.contracts import entry_dynamic_compact_input_influence_v1 as audit
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_model_native_input_normalization_v1 import require_input_normalization_contract
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import (
        DYNAMIC_M5_LOSS_SCALE_BATCH_KEY, load_dynamic_m5_loss_scales, dynamic_m5_loss_scale_identity,
    )
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import _model
    proofs = []
    metadata, constructor, norm_document, final = (_read(scope[k], proofs=proofs) for k in
        ("input_packet_metadata", "constructor_metadata", "normalization", "final_online_result"))
    _require(final["optimizer_steps"] == 7391 and final["selected_model_variant"] == "ONLINE"
             and all(final[k] is False for k in ("test_data_used", "exit_evaluation", "teacher_refreshed", "economic_rollout")), "FINAL_RESULT")
    native170 = plan["schema_version"] == PLAN_SCHEMA_V2
    if native170:
        design = _read(scope["design"], proofs=proofs)
        equivalence = _read(scope["input_contract_equivalence"], proofs=proofs)
        origin_constructor = _read(scope["packet_origin"]["constructor_metadata"], proofs=proofs)
        _require_v2_origin(scope, metadata, constructor, origin_constructor, final, design, equivalence)
    else:
        _require(metadata["constructor_metadata"] == scope["constructor_metadata"]
                 and metadata["normalization"] == scope["normalization"]
                 and metadata["loss_scales"] == scope["loss_scales"]
                 and metadata["genuine_rows"] == 16 and metadata["padding_rows"] == 0
                 and metadata["final_online_result"] == scope["final_online_result"], "PACKET_METADATA_BINDING")
    layout = audit.compile_dynamic_compact_influence_layout()
    normalization = require_input_normalization_contract(norm_document["contract"],
        expected_field_names={"signal": layout["physical_field_names"]["signal"],
            "ctx_cont": layout["physical_field_names"]["ctx_cont"],
            **{"mtf_" + tf: layout["physical_field_names"]["mtf"] for tf in ("m5", "m15", "h1", "h4", "d1")}},
        expected_ctx_cat_names=layout["physical_field_names"]["ctx_cat"],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    flat_inputs, batch, references, packet_identity = _packet(scope, metadata)
    reference_join = None
    if native170:
        physical_group_counts = _require_native170_physical_batch(batch)
        pointer_documents = {key: _read(scope["canonical_pointer_proof"][key], proofs=proofs)
                             for key in ("plan", "terminal", "actual_peer")}
        saved_reference = _read_saved_train_reference(scope["parity_reference"], proofs=proofs)
        references, reference_join = _join_saved_train_reference(saved_reference, scope=scope,
            metadata=metadata, batch=batch, final=final, pointer_documents=pointer_documents)
        reference_join["physical_terminal_group_counts_before_forward"] = physical_group_counts
        del saved_reference, pointer_documents
    cpu_inputs = _nest(flat_inputs)
    audit.require_dynamic_compact_inputs(cpu_inputs, layout=layout)
    input_sha = audit.dynamic_compact_input_sha256(cpu_inputs)
    _require(input_sha == metadata["input_sha256"]
             and len(metadata["parent_rows"]) == len(set(metadata["parent_rows"])) == 16
             and len(metadata["entry_time_ns"]) == 16
             and batch["dynamic_entry_parent_row_index"].tolist() == metadata["parent_rows"], "GENUINE_ORDER")
    session_scope = scope["session"]
    _require(set(session_scope) == {"out_bundle_dir", "contract", "active_pointer", "state", "online_model_state_sha256"}, "SESSION_SCOPE")
    contract, pointer = (_read(session_scope[k], proofs=proofs) for k in ("contract", "active_pointer"))
    bundle = Path(session_scope["out_bundle_dir"])
    session_directory = bundle.parent / (".gx1-candidate-training-session." + bundle.name)
    _require(bundle.is_absolute() and bundle.resolve() == bundle and contract["out_bundle_dir"] == str(bundle)
             and _reference(session_scope["contract"]) == session_directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json"
             and _reference(session_scope["active_pointer"]) == session_directory / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
             and _reference(session_scope["state"]) == session_directory / "candidate_training_state_slot_0.pt", "SESSION_PATH")
    _require(pointer["slot"] == 0 and pointer["checkpoint_index"] == 117 and pointer["phase"] == "train"
             and pointer["epoch_index"] == 0 and pointer["next_batch_offset"] == pointer["global_optimizer_steps"] == 7391
             and pointer["complete"] is False and pointer["state_sha256"] == session_scope["state"]["sha256"], "FINAL_POINTER")
    prefix = contract["chronological_prefix"]
    _require(prefix["entry_training_phase"] == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
             and contract["input_normalization_sha256"] == normalization["contract_sha256"]
             and contract["training"]["precision"] == "deterministic_fp32"
             and contract["training"]["tf32"] is contract["training"]["autocast"] is False
             and final["model_state_sha256"] == session_scope["online_model_state_sha256"], "SESSION_FUNCTION")
    scale_document = _read(scope["loss_scales"], proofs=proofs)
    scales = load_dynamic_m5_loss_scales(scope["loss_scales"], expected_lineage=scale_document["fit_lineage"])
    scale_identity = dynamic_m5_loss_scale_identity(scales)
    _require(scale_identity == prefix["dynamic_m5_loss_scales"] == final["dynamic_m5_loss_scales"], "LOSS_IDENTITY")
    batch[DYNAMIC_M5_LOSS_SCALE_BATCH_KEY] = {"identity": scale_identity,
        "rms": {key: torch.from_numpy(value.copy()).to(device=device) for key, value in scales["rms"].items()}}
    session_identities = {key: _identity(_reference(session_scope[key])) for key in ("contract", "active_pointer", "state")}
    device_object = torch.device(device)
    rng = trainer._attended_session_rng_state(device=device_object)
    rng_digest = _rng_sha(rng)
    threads = torch.get_num_threads()
    model, modes, original_state, online_digest, execution = None, None, None, None, None
    handles = []
    try:
        session = trainer._CandidateTrainingSession(out_bundle_dir=bundle, contract=contract, read_only=True)
        state = session.load_checkpoint()
        _require(state["global_optimizer_steps"] == 7391 and state["session_contract_sha256"] == session.contract_sha256
                 and trainer.canonical_model_state_sha256(state["model_state"]) == session_scope["online_model_state_sha256"], "EXACT_FINAL_ONLINE")
        trainer._set_deterministic(contract["training"]["seed"], device_object, contract["training"]["precision"])
        torch.set_num_threads(1)
        model = _model(constructor, normalization, device_object,
            **trainer._dynamic_m5_model_kwargs(ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE),
            entry_execution_profile=constructor["entry_execution_profile"])
        original_state = state["model_state"]
        model.load_state_dict(original_state, strict=True)
        del state
        model.require_input_normalization_state()
        trainer._require_prefix_online_function(model)
        online_digest = trainer._model_state_sha256(model)
        _require(online_digest == session_scope["online_model_state_sha256"]
                 and all(p.grad is None for p in model.parameters()), "ONLINE_LOAD")
        modes = [(module, module.training) for module in model.modules()]
        model.eval()
        query_inputs, forecast_inputs = [], []
        def capture_query(module, args):
            _require(len(args) == 1 and isinstance(args[0], torch.Tensor), "QUERY_HOOK")
            query_inputs.append(args[0])
        def capture_forecast(module, args):
            _require(len(args) == 1 and isinstance(args[0], torch.Tensor), "FORECAST_HOOK")
            forecast_inputs.append(args[0])
        handles = [model.dynamic_m5_query_projection.register_forward_pre_hook(capture_query),
                   model.head_forecast.register_forward_pre_hook(capture_forecast)]
        inputs = {key: value.to(device_object) for key, value in flat_inputs.items()}
        outputs = trainer._model_forward_fp32(model, inputs["seq_x"], inputs["snap_x"],
            ctx_cat=inputs["ctx_cat"], ctx_cont=inputs["ctx_cont"],
            **{key: inputs[key] for key in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")})
        for handle in handles: handle.remove()
        handles = []
        _require(len(query_inputs) == 2 and len(forecast_inputs) == 1
                 and "entry_action_q_bps" not in outputs and not any(k.startswith("unified_exit") for k in outputs), "ONE_DYNAMIC_GRAPH")
        # Reuse117's source-owned native component tolerance; no new threshold.
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import DYNAMIC_M5_TECHNICAL_FIXED_SCOPE
        tolerance = DYNAMIC_M5_TECHNICAL_FIXED_SCOPE["parity_absolute_tolerance_bps"]
        relative = DYNAMIC_M5_TECHNICAL_FIXED_SCOPE["parity_relative_tolerance"]
        parity = {}
        for key in _COMPONENTS:
            actual = outputs[key].detach().float().cpu()
            expected = references[key]
            _require(actual.shape == expected.shape and actual.dtype == expected.dtype == torch.float32, "REFERENCE_SHAPE:" + key)
            delta = (actual.double() - expected.double()).abs()
            valid = torch.isclose(actual, expected, atol=tolerance, rtol=relative)
            parity[key] = {"cells": actual.numel(), "max_abs_delta_bps": float(delta.max()),
                           "mismatching_cells": int((~valid).sum()), "absolute_tolerance_bps": tolerance, "relative_tolerance": relative}
            _require(bool(valid.all()), "NATIVE_COMPONENT_PARITY:" + key)
        reference_quality = references["dynamic_terminal_net_markout_bps"][..., :2] - references["dynamic_pre_mfe_mae_upper_bps"]
        reference_best = reference_quality.amax(dim=1)
        reference_actions = torch.cat((reference_best, torch.zeros_like(reference_best[..., :1])), dim=1)
        reference_ties = reference_quality == reference_best[:, None, :]
        actual_quality = outputs["dynamic_entry_quality_bps_candidate"].detach().float().cpu()
        actual_actions = outputs["dynamic_entry_action_quality_bps_candidate"].detach().float().cpu()
        _require(torch.allclose(actual_quality[..., :2], reference_quality, atol=tolerance, rtol=relative)
                 and bool((actual_quality[..., 2] == 0).all())
                 and torch.allclose(actual_actions, reference_actions, atol=tolerance, rtol=relative)
                 and torch.equal(outputs["dynamic_entry_quality_best_horizon_mask"].detach().cpu(), reference_ties)
                 and torch.equal(actual_actions == actual_actions.amax(dim=1, keepdim=True),
                                 reference_actions == reference_actions.amax(dim=1, keepdim=True)), "CANDIDATE_TIE_ACTION_PARITY")
        parity["candidate_tie_and_action"] = {"existing_tolerance_bps": tolerance,
            "all_equal_best_horizon_masks_identical": True, "all_action_winner_sets_identical": True}
        gradients = audit.dynamic_m5_direction_witness_gradients(model=model, outputs=outputs,
            batch={**batch, "seq_x": inputs["seq_x"]}, query_inputs=query_inputs,
            forecast_input=forecast_inputs[0], headline_horizons=tuple(plan["headline_horizons"]),
            **({"terminal_objective": "native_170_groupmean_v1"} if native170 else {}))
        _require(gradients["reverse_calls"] == plan["max_reverse_calls"]
                 and gradients["forward_calls"] == 0, "GRADIENT_BUDGET")
        if native170:
            _require(gradients["terminal_group_counts"] == physical_group_counts,
                     "PHYSICAL_TERMINAL_GROUP_COUNTS_CHANGED")
        execution = {"schema_version": ("gx1_dynamic_m5_direction_witness_result_v2" if native170 else
                                         "gx1_dynamic_m5_direction_witness_result_v1"),
            "status": "COMPLETE_LOCAL_EVAL_DIRECTION_GRADIENT_WITNESS_NOT_TRAINING_CAUSE_OR_EDGE",
            "plan": plan_ref, "input_scope": plan["input_scope"], "input_packet": scope["input_packet"],
            "genuine_rows": 16, "padding_rows": 0, "parent_rows": metadata["parent_rows"],
            "entry_time_ns": metadata["entry_time_ns"], "ONLINE_model_state_sha256": online_digest,
            "optimizer_steps_in_loaded_state": 7391, "actual_forward_calls": 1,
            "actual_reverse_calls": gradients["reverse_calls"], "optimizer_updates": 0,
            "input_batch_sha256": input_sha, "loss_scales": scale_identity,
            "component_parity": parity, "gradients": gradients, "hardware": hardware, "lock_ancestry": lock,
            **({"reference_join": reference_join, "packet_origin": scope["packet_origin"],
                "execution_constructor": scope["constructor_metadata"],
                "terminal_optimizer_objective": gradients["terminal_optimizer_objective_contract"]}
               if native170 else {}),
            "observation_mode": "eval", "epoch_training_mode": "train",
            "causality_limit": "Local eval-function gradients at final state do not establish7391-update history, AdamW/clipping/momentum effects or a corrective model.",
            "source_policy_outer_guards_pending": True, **{key: False for key in FLAGS}}
    finally:
        errors = []
        for handle in handles: handle.remove()
        try:
            if model is not None:
                if any(p.grad is not None for p in model.parameters()):
                    errors.append("PARAMETER_GRADS_CHANGED")
                    for p in model.parameters(): p.grad = None
                if online_digest is not None and trainer._model_state_sha256(model) != online_digest:
                    errors.append("ONLINE_CHANGED")
                    model.load_state_dict(original_state, strict=True)
                model.require_input_normalization_state()
        except Exception as error:
            errors.append(type(error).__name__ + ":" + str(error))
        finally:
            if modes is not None:
                for module, mode in modes: module.training = mode
            try:
                trainer._restore_attended_session_rng_state(rng, device=device_object)
                _require(_rng_sha(trainer._attended_session_rng_state(device=device_object)) == rng_digest, "RNG_RESTORE")
            except Exception as error: errors.append(type(error).__name__ + ":" + str(error))
        try:
            torch.set_num_threads(threads)
            _require(torch.get_num_threads() == threads, "THREAD_RESTORE")
        except Exception as error: errors.append(type(error).__name__ + ":" + str(error))
        for key, ident in session_identities.items():
            try:
                path = _reference(session_scope[key])
                _require(_identity(path) == ident and _sha(path) == session_scope[key]["sha256"]
                         and _identity(path) == ident, "SESSION_PRESERVATION:" + key)
            except Exception as error: errors.append(type(error).__name__ + ":" + str(error))
        try:
            _require(_identity(_reference(scope["input_packet"])) == packet_identity
                     and all(_identity(_reference(ref)) == ident for ref, ident in proofs), "PACKET_OR_METADATA_CHANGED")
            _require(audit.dynamic_compact_input_sha256(cpu_inputs) == input_sha, "INPUT_VALUES_CHANGED")
            frozen()
        except Exception as error: errors.append(type(error).__name__ + ":" + str(error))
        _require(not errors, "PRESERVATION:" + ";".join(errors))
    _require(execution is not None and time.monotonic() - started < plan["max_wall_seconds"], "FINAL_COMPLETION")
    execution.update(ONLINE_weights_preserved=True, parameter_grads_preserved=True, modes_restored=True,
        RNG_restored=True, torch_threads_restored=True, session_files_preserved=True,
        source_policy_preserved=True, elapsed_seconds=time.monotonic() - started)
    output.mkdir()
    result_ref = _publish(output, "RESULT.json", execution)
    return {"status": execution["status"], "result": result_ref,
        "actual_forward_calls": 1, "actual_reverse_calls": execution["actual_reverse_calls"],
        "optimizer_updates": 0, "genuine_rows": 16, "padding_rows": 0,
        "source_policy_outer_guards_pending": True}
