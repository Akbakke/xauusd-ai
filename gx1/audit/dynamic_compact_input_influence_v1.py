"""Bounded read-only native Entry input sensitivity; no training authority.

The allowlisted technical validator dispatches here before its legacy parser.
The integrator owns the immutable plan, CUDA producer and source/policy freeze.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import io
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
PLAN_SCHEMA = "gx1_dynamic_compact_native_input_audit_plan_v1"
SCOPE_SCHEMA = "gx1_dynamic_compact_native_input_audit_scope_v1"
FLAGS = ("training_enabled", "full_epoch_training_allowed", "full_val_allowed",
         "native_training_admitted", "test_data_used", "normalization_fit_enabled",
         "exit_evaluation", "broker_access_allowed")
STAT_KEYS = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")


def _require(condition, reason):
    if not condition:
        raise RuntimeError("DYNAMIC_COMPACT_NATIVE_INPUT_AUDIT_" + reason)


def _identity(path):
    info = path.stat()
    return {key: getattr(info, key) for key in STAT_KEYS}


def _path(value):
    path = Path(value)
    _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
             and not any(p.is_symlink() for p in (path, *path.parents)), "CANONICAL_FILE")
    return path


def _reference(value):
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"}
             and isinstance(value["sha256"], str) and len(value["sha256"]) == 64
             and all(c in "0123456789abcdef" for c in value["sha256"]), "REFERENCE")
    return _path(value["path"])


def _sha(path):
    digest = hashlib.sha256()
    before = _identity(path)
    with path.open("rb") as handle:
        _require({k: getattr(os.fstat(handle.fileno()), k) for k in STAT_KEYS} == before, "HASH_DESCRIPTOR")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        _require({k: getattr(os.fstat(handle.fileno()), k) for k in STAT_KEYS} == before, "HASH_CHANGED")
    _require(_identity(path) == before, "HASH_PATH_CHANGED")
    return digest.hexdigest()


def _read(value, *, proofs=None, _policy_history=False):
    path = _reference(value)
    before = _identity(path)
    _require(path.suffix == ".json" and before["st_size"] <= 32 * 1024 * 1024, "JSON_BOUND")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == value["sha256"] and _identity(path) == before, "JSON_SHA")
    def pairs(items):
        result = {}
        for key, item in items:
            _require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = item
        return result
    def invalid(value):
        raise RuntimeError("DYNAMIC_COMPACT_NATIVE_INPUT_AUDIT_NONFINITE_JSON:" + value)
    _require(_policy_history is False or path == REPO / "NEXT_RUN_POLICY.json", "POLICY_HISTORY_PATH")
    # Historical policy aggregates use stdlib NaN syntax. The exact active flags
    # and stage remain strictly typed; new plans/scopes/manifests reject NaN.
    parser = {"object_pairs_hook": pairs}
    if not _policy_history:
        parser["parse_constant"] = invalid
    result = json.loads(raw, **parser)
    _require(isinstance(result, dict), "JSON_OBJECT")
    if proofs is not None:
        proofs.append((dict(value), before))
    return result


def _require_plan(plan):
    keys = {"schema_version", "run_id", "policy_stage_key", "input_scope", "output_directory", "one_shot_invocations", "max_wall_seconds",
            "max_forward_calls", "reverse_calls", *FLAGS}
    _require(isinstance(plan, dict) and set(plan) == keys and plan["schema_version"] == PLAN_SCHEMA,
             "PLAN_SCHEMA")
    _require(type(plan["one_shot_invocations"]) is int and plan["one_shot_invocations"] == 1
             and type(plan["max_wall_seconds"]) is int and 0 < plan["max_wall_seconds"] <= 3500
             and type(plan["max_forward_calls"]) is int and 2 <= plan["max_forward_calls"] <= 218
             and type(plan["reverse_calls"]) is int and plan["reverse_calls"] == 13
             and all(plan[key] is False for key in FLAGS), "PLAN_AUTHORITY")
    _require(isinstance(plan["policy_stage_key"], str) and bool(plan["policy_stage_key"])
             and isinstance(plan["run_id"], str) and bool(plan["run_id"]), "PLAN_IDENTITY")


def _require_policy(plan, plan_ref, policy):
    _require(all(policy.get(key) is False for key in ("training_enabled", "full_epoch_training_allowed", "full_val_allowed")), "GLOBAL_POLICY")
    scope = policy.get(plan["policy_stage_key"])
    _require(isinstance(scope, dict) and scope.get("execution_authorized") is True
             and type(scope.get("planned_invocations")) is int and scope["planned_invocations"] == 1
             and type(scope.get("completed_invocations")) is int and scope["completed_invocations"] == 0
             and scope.get("relaunch_allowed") is False and scope.get("plan") == plan_ref
             and all(scope.get(key) is False for key in FLAGS), "EXACT_POLICY_STAGE")


def _preflight(plan_path, plan_sha256, device, out_dir):
    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan = _read(plan_ref)
    _require_plan(plan)
    _require(device == "cuda", "GUARDED_CUDA_DEVICE_REQUIRED")
    output = Path(out_dir)
    _require(output.is_absolute() and output.resolve() == output and str(output) == plan["output_directory"]
             and not output.exists() and output.parent.is_dir()
             and not any(p.is_symlink() for p in output.parents), "FRESH_OUTPUT")
    policy_path = REPO / "NEXT_RUN_POLICY.json"
    policy_identity, policy_sha = _identity(policy_path), _sha(policy_path)
    policy = _read({"path": str(policy_path), "sha256": policy_sha}, _policy_history=True)
    _require_policy(plan, plan_ref, policy)
    # HEAD and source capture are published only after the clean plan/policy
    # commit. Keeping these fields out of that commit removes the hash cycle.
    binding_path = _path(Path(plan_ref["path"]).parent / "NATIVE_INPUT_AUDIT_BINDING.json")
    binding_identity = _identity(binding_path)
    binding_ref = {"path": str(binding_path), "sha256": _sha(binding_path)}
    binding = _read(binding_ref)
    _require(set(binding) == {"schema_version", "plan", "source_head", "policy_sha256", "source_bindings"}
             and binding["schema_version"] == "gx1_dynamic_compact_native_input_audit_binding_v1"
             and binding["plan"] == plan_ref and binding["policy_sha256"] == policy_sha,
             "POSTCOMMIT_BINDING")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    _require(git("branch", "--show-current") == "work/gx1-current"
             and git("rev-parse", "HEAD") == binding["source_head"] and not git("status", "--porcelain"), "SOURCE_HEAD_OR_DIRTY")
    sources = _read(binding["source_bindings"])
    _require(set(sources) == {"source_head", "files"} and sources["source_head"] == binding["source_head"]
             and isinstance(sources["files"], dict) and bool(sources["files"]), "SOURCE_BINDINGS")
    scope = _read(plan["input_scope"])
    proof_reads = []
    _require(set(scope) == {"schema_version", "input_packet", "input_packet_metadata", "case_manifest",
             "normalization", "constructor_metadata", "session", "helper", "executor",
             "input_render_actual_review", "input_render_terminal", "case_manifest_actual_review", "case_manifest_terminal"}
             and scope["schema_version"] == SCOPE_SCHEMA, "INPUT_SCOPE")
    _require(_reference(scope["executor"]) == Path(__file__).resolve(), "EXECUTOR_IDENTITY")
    _require_actual_proofs(scope, proofs=proof_reads)
    helper = REPO / "gx1/contracts/entry_dynamic_compact_input_influence_v1.py"
    _require(_reference(scope["helper"]) == helper, "HELPER_IDENTITY")
    for ref in (scope["helper"], scope["executor"]):
        record = sources["files"].get(ref["path"])
        _require(isinstance(record, dict) and record["sha256"] == ref["sha256"], "SOURCE_SCOPE_JOIN")
    def frozen():
        _require(git("branch", "--show-current") == "work/gx1-current" and git("rev-parse", "HEAD") == binding["source_head"]
                 and not git("status", "--porcelain"), "SOURCE_CHANGED")
        _require(_identity(policy_path) == policy_identity and _sha(policy_path) == policy_sha, "POLICY_CHANGED")
        _require(_identity(binding_path) == binding_identity and _read(binding_ref) == binding
                 and _read(plan_ref) == plan and _read(binding["source_bindings"]) == sources
                 and _read(plan["input_scope"]) == scope, "BINDING_CHANGED")
        _require(all(_identity(_reference(ref)) == identity for ref, identity in proof_reads), "ACTUAL_PROOF_CHANGED")
        for name, record in sources["files"].items():
            path = _path(name)
            _require(path.is_relative_to(REPO) and _identity(path) == record["identity"]
                     and _sha(path) == record["sha256"] and _identity(path) == record["identity"], "SOURCE_FILE_CHANGED")
    frozen()
    return plan, plan_ref, scope, output, frozen


def _require_actual_proofs(scope, *, proofs):
    for label, review_key, terminal_key, expected_status in (
        ("render", "input_render_actual_review", "input_render_terminal", "PASS_THREE_TRAIN_INPUT_PACKET_ONLY_NOT_NATIVE_INFLUENCE"),
        ("cases", "case_manifest_actual_review", "case_manifest_terminal", "PASS_INPUT_ONLY_FIXED_CASE_MANIFEST_NOT_NATIVE_INFLUENCE"),
    ):
        review, terminal = (_read(scope[key], proofs=proofs) for key in (review_key, terminal_key))
        _require(review["status"] == expected_status and review["terminal"] == scope[terminal_key]
                 and review["actual_result"] == terminal["result"]
                 and review["source_policy_outer_guards_pending"] is False
                 and terminal["status"] == "COMPLETE" and type(terminal["exit_code"]) is int and terminal["exit_code"] == 0
                 and terminal["source_unchanged"] is True and terminal["completed_invocations"] == 1
                 and terminal["relaunch_allowed"] is False, "ACTUAL_PROOF:" + label)
        outer = _read(terminal["result"], proofs=proofs)
        _require(outer["status"] == "PASS" and type(outer["exit_code"]) is int and outer["exit_code"] == 0
                 and outer["source_unchanged"] is outer["policy_unchanged"] is True, "ACTUAL_OUTER_PROOF:" + label)
        if label == "render":
            _require(review["input_packet"] == scope["input_packet"]
                     and review["input_packet_metadata"] == scope["input_packet_metadata"], "ACTUAL_PACKET_PROOF")
        else:
            _require(review["case_manifest"] == scope["case_manifest"]
                     and review["actual_case_result"] == outer["qualified_partial"], "ACTUAL_CASE_PROOF")


def _validate_case_manifest(audit, inputs, layout, normalization, manifest, metadata, scope):
    from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256
    view = audit.dynamic_compact_normalization_f32_view(normalization, layout=layout)
    _require(manifest["schema_version"] == "gx1_dynamic_compact_case_manifest_v1"
             and manifest["input_sha256"] == manifest["input_batch_sha256"] == audit.dynamic_compact_input_sha256(inputs)
             and manifest["layout_sha256"] == canonical_sha256(layout)
             and manifest["normalization_f32_sha256"] == audit._dynamic_compact_normalization_sha256(view)
             and manifest["execution_authorized"] is False
             and manifest["selection_uses_predictions_targets_or_gradients"] is False
             and manifest["genuine_rows"] == 3 and manifest["owner_counts"] == layout["counts"]
             and manifest["skip_rule"] == "execute_every_valid_predeclared_case", "CASE_MANIFEST_IDENTITY")
    order = sorted(range(3), key=lambda row: (manifest["decision_times_ns"][row], manifest["parent_rows"][row]))
    _require(manifest["witness_order"] == order and manifest["parent_rows"] == metadata["unique_parent_rows"], "CASE_WITNESS_ORDER")
    inventory = [(group, owner) for group in ("numeric", "aliases", "categorical", "families") for owner in layout[group]]
    _require(len(manifest["cases"]) == manifest["inventory_label_count"] == len(inventory) == audit.MAX_CASES, "FULL_OWNER_INVENTORY")
    statuses = Counter()
    flat = audit.require_dynamic_compact_inputs(inputs, layout=layout)
    for ordinal, ((group, owner), record) in enumerate(zip(inventory, manifest["cases"])):
        source, index, domain, own, dependent = audit._dynamic_compact_case_source(owner, group, layout)
        kind = "family" if group == "families" else "mask_drop" if dependent else "category" if domain == "source_enum" else "value"
        witness = next((row for row in order if audit._dynamic_compact_case_eligible(flat, layout, owner, group, row)), None)
        _require(type(record["ordinal"]) is int and record["ordinal"] == ordinal and record["owner_group"] == group
                 and record["token"] == owner["token"] and record["kind"] == kind and record["witness_row"] == witness
                 and record["physical_parent_row"] == (None if witness is None else manifest["parent_rows"][witness])
                 and record["decision_time_ns"] == (None if witness is None else manifest["decision_times_ns"][witness])
                 and record["source_surface"] == source and record["source_index"] == index
                 and record["own_observation_indices"] == list(own) and record["time_index"] is None,
                 "CASE_SOURCE_BINDING:" + owner["token"])
        recipe = record["replacement_recipe"]
        _require(recipe["normalization_f32_sha256"] == manifest["normalization_f32_sha256"]
                 and recipe["selection_uses_only_source_masks_and_fixed_time_parent_order"] is True
                 and recipe["retry_allowed"] is False and record["status"] in
                 ("valid", "not_exercised_missing", "source_declared_constant", "invalid_counterfactual"), "CASE_RECIPE")
        if witness is None:
            _require(record["status"] == "not_exercised_missing", "MISSING_CASE_STATUS")
        if record["status"] == "valid":
            _require(witness is not None and isinstance(record["input_sha256"], str) and len(record["input_sha256"]) == 64
                     and all(c in "0123456789abcdef" for c in record["input_sha256"]), "VALID_CASE_DIGEST")
        else:
            _require(record["input_sha256"] is None, "UNEXECUTED_CASE_DIGEST")
        statuses[record["status"]] += 1
    expected_statuses = {key: statuses[key] for key in ("valid", "not_exercised_missing", "source_declared_constant", "invalid_counterfactual")}
    _require(manifest["status_counts"] == expected_statuses and manifest["valid_case_count"] == statuses["valid"]
             and manifest["budget"] == audit.dynamic_compact_influence_budget(statuses["valid"], statuses["valid"])
             and manifest["max_inventory_budget"] == audit.dynamic_compact_influence_budget(len(inventory), len(inventory)), "CASE_COUNTS_BUDGET")
    _require(all(manifest[key] == scope[key] for key in ("input_packet", "input_packet_metadata", "normalization", "helper")), "CASE_OUTER_BINDING")
    return view


def _publish(directory, name, value):
    target = directory / name
    fd, temporary = tempfile.mkstemp(prefix="." + name + ".", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, target)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(target), "sha256": _sha(target)}


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def _route_json(routes, genuine_rows):
    """Copy each projection/field once, rather than synchronizing each owner."""
    import torch
    result = {token: {projection: {} for projection in values} for token, values in routes.items()}
    projections = tuple(next(iter(routes.values())))
    for projection in projections:
        fields = sorted({field for values in routes.values() for field in values[projection]})
        for field in fields:
            tokens = [token for token, values in routes.items() if field in values[projection]]
            values = torch.stack([routes[token][projection][field][:genuine_rows] for token in tokens]).detach().cpu().tolist()
            for token, value in zip(tokens, values):
                result[token][projection][field] = value
    return result


def _rng_sha(value):
    def encode(item):
        if hasattr(item, "detach"):
            array = item.detach().cpu().contiguous().numpy()
            return {"dtype": str(array.dtype), "shape": list(array.shape), "bits": array.tobytes().hex()}
        if isinstance(item, dict):
            return {key: encode(v) for key, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _packet(scope, metadata, manifest, layout):
    import numpy as np
    import torch
    from gx1.contracts.entry_dynamic_compact_input_influence_v1 import require_dynamic_compact_inputs
    path = _reference(scope["input_packet"])
    before = _identity(path)
    _require(metadata["input_packet"] == scope["input_packet"] and metadata["genuine_rows"] == 3
             and metadata["padding_rows"] == 13 and metadata["padded_row_map"] == [0, 1, 2] + [0] * 13
             and metadata["score_mask"] == [True] * 3 + [False] * 13
             and metadata["raw_inputs_not_normalized_or_fitted"] is True, "PACKET_METADATA")
    _require(before["st_size"] <= 32 * 1024 * 1024, "PACKET_BYTE_BOUND")
    shapes = {"seq_x": ("float32", (3, 96, 2725)), "snap_x": ("float32", (3, 2725)),
              "ctx_cont": ("float32", (3, 2542)), "ctx_cat": ("int64", (3, 1)),
              "parent_rows": ("int64", (3,)), "padded_row_map": ("int64", (16,)), "score_mask": ("bool", (16,))}
    for tf, length in metadata["per_tf_seq_lens"].items():
        shapes["seq_" + tf.lower()] = ("float32", (3, length, 253))
    _require(set(metadata["per_tf_seq_lens"]) == {"M5", "M15", "H1", "H4", "D1"}, "PACKET_TIMEFRAMES")
    arrays = {}
    with path.open("rb") as handle:
        _require({k: getattr(os.fstat(handle.fileno()), k) for k in STAT_KEYS} == before, "PACKET_DESCRIPTOR")
        raw = handle.read()
        _require(hashlib.sha256(raw).hexdigest() == scope["input_packet"]["sha256"], "PACKET_PHYSICAL_SHA")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            _require(len(archive.files) == len(shapes) and set(archive.files) == set(shapes), "PACKET_KEYS")
            for key, (dtype, shape) in shapes.items():
                value = archive[key]
                declared = metadata["arrays"][key]
                _require(value.dtype == np.dtype(dtype) and value.shape == shape and np.isfinite(value).all()
                         and declared == {"dtype": dtype, "shape": list(shape), "value_bytes": value.nbytes}, "PACKET_DTYPE_SHAPE:" + key)
                arrays[key] = np.ascontiguousarray(value)
        _require({k: getattr(os.fstat(handle.fileno()), k) for k in STAT_KEYS} == before, "PACKET_DESCRIPTOR_CHANGED")
    _require(_identity(path) == before and arrays["parent_rows"].tolist() == metadata["unique_parent_rows"] == manifest["parent_rows"]
             and arrays["padded_row_map"].tolist() == metadata["padded_row_map"]
             and arrays["score_mask"].tolist() == metadata["score_mask"], "PACKET_IDENTITY_ROWS")
    index = arrays["padded_row_map"]
    inputs = {key: torch.from_numpy(arrays[key][index].copy()) for key in ("seq_x", "snap_x", "ctx_cont", "ctx_cat")}
    inputs["mtf"] = {key: torch.from_numpy(arrays[key][index].copy()) for key in shapes if key.startswith("seq_") and key != "seq_x"}
    require_dynamic_compact_inputs(inputs, layout=layout)
    return inputs, before


def run(*, plan_path: Path, plan_sha256: str, device: str, out_dir: Path) -> dict:
    started = time.monotonic()
    plan, plan_ref, scope, output, frozen = _preflight(plan_path, plan_sha256, device, out_dir)
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    hardware, lock = require_guarded_cuda_producer_execution(), require_capped_lock_ancestry()
    import torch
    from gx1.contracts import entry_dynamic_compact_input_influence_v1 as audit
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_model_native_input_normalization_v1 import require_input_normalization_contract
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import _model
    layout = audit.compile_dynamic_compact_influence_layout()
    metadata_proofs = []
    metadata, manifest = (_read(scope[key], proofs=metadata_proofs) for key in ("input_packet_metadata", "case_manifest"))
    constructor, document = (_read(scope[key], proofs=metadata_proofs) for key in ("constructor_metadata", "normalization"))
    normalization = require_input_normalization_contract(document["contract"],
        expected_field_names={"signal": layout["physical_field_names"]["signal"], "ctx_cont": layout["physical_field_names"]["ctx_cont"],
                              **{"mtf_" + tf: layout["physical_field_names"]["mtf"] for tf in ("m5", "m15", "h1", "h4", "d1")}},
        expected_ctx_cat_names=layout["physical_field_names"]["ctx_cat"], expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    _require(metadata["normalization"] == scope["normalization"] and metadata["constructor_metadata"] == scope["constructor_metadata"], "PACKET_NORMALIZATION_CONSTRUCTOR")
    cpu_inputs, packet_identity = _packet(scope, metadata, manifest, layout)
    _require(audit.dynamic_compact_input_sha256(cpu_inputs) == manifest["input_sha256"], "MANIFEST_PACKET_SHA")
    def stamp_ns(value):
        stamp = datetime.fromisoformat(value)
        _require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, "PACKET_UTC")
        delta = stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (delta.days * 86400 + delta.seconds) * 1000000000 + delta.microseconds * 1000
    decisions = [stamp_ns(item["availability_utc"]) for item in metadata["mtf_windows"]]
    _require(manifest["decision_times_ns"] == decisions and manifest["input_batch_sha256"] == manifest["input_sha256"]
             and all(manifest[key] == scope[key] for key in ("input_packet", "input_packet_metadata", "normalization", "helper")), "CPU_MANIFEST_BINDINGS")
    _reference(manifest["plan"])
    _reference(manifest["input_scope"])
    normalized_view = _validate_case_manifest(audit, cpu_inputs, layout, normalization, manifest, metadata, scope)
    budget = manifest["budget"]
    _require(manifest["execution_authorized"] is False and manifest["inventory_label_count"] == 3446
             and manifest["skip_rule"] == "execute_every_valid_predeclared_case"
             and budget["forward_calls"] == plan["max_forward_calls"] and budget["reverse_calls"] == plan["reverse_calls"], "MANIFEST_BUDGET")
    session_scope = scope["session"]
    _require(set(session_scope) == {"out_bundle_dir", "contract", "active_pointer", "state", "online_model_state_sha256"}, "SESSION_SCOPE")
    contract, pointer = (_read(session_scope[key], proofs=metadata_proofs) for key in ("contract", "active_pointer"))
    bundle = Path(session_scope["out_bundle_dir"])
    expected_directory = bundle.parent / (".gx1-candidate-training-session." + bundle.name)
    _require(bundle.is_absolute() and bundle.resolve() == bundle and contract["out_bundle_dir"] == str(bundle)
             and _reference(session_scope["contract"]) == expected_directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json"
             and _reference(session_scope["active_pointer"]) == expected_directory / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
             and _reference(session_scope["state"]) == expected_directory / "candidate_training_state_slot_0.pt", "SESSION_PATH_OWNER")
    _require(pointer["slot"] == 0 and pointer["checkpoint_index"] == 1 and pointer["phase"] == "train"
             and pointer["epoch_index"] == pointer["next_batch_offset"] == pointer["global_optimizer_steps"] == 0
             and pointer["complete"] is False and pointer["state_sha256"] == session_scope["state"]["sha256"], "INITIAL_POINTER")
    prefix = contract["chronological_prefix"]
    _require(prefix["entry_training_phase"] == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
             and prefix["initial_model_state_sha256"] == session_scope["online_model_state_sha256"]
             and contract["input_normalization_sha256"] == normalization["contract_sha256"]
             and contract["training"]["precision"] == "deterministic_fp32" and contract["training"]["tf32"] is False
             and contract["training"]["autocast"] is False, "SESSION_FUNCTION_BINDING")
    session_identities = {key: _identity(_reference(session_scope[key])) for key in ("contract", "active_pointer", "state")}
    device_object = torch.device(device)
    rng = trainer._attended_session_rng_state(device=device_object)
    rng_digest = _rng_sha(rng)
    original_threads = torch.get_num_threads()
    model, modes, online_digest, original_model_state = None, None, None, None
    forwards = 0
    execution = None
    try:
        # This is the ordinary checkpoint-policy owner, including fresh dependency
        # validation. Never replace it with direct torch.load or a copied state.
        session = trainer._CandidateTrainingSession(out_bundle_dir=bundle, contract=contract, read_only=True)
        state = session.load_checkpoint()
        _require(isinstance(state, dict) and state["global_optimizer_steps"] == 0
                 and state["session_contract_sha256"] == session.contract_sha256, "INITIAL_ONLINE_STATE")
        _require(trainer.canonical_model_state_sha256(state["model_state"]) == session_scope["online_model_state_sha256"], "ONLINE_DIGEST")
        trainer._set_deterministic(contract["training"]["seed"], device_object, contract["training"]["precision"])
        torch.set_num_threads(1)
        _require(torch.get_num_threads() == 1, "GUARDED_NUMERICAL_THREAD_CAP")
        model = _model(constructor, normalization, device_object,
            **trainer._dynamic_m5_model_kwargs(ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE), entry_execution_profile=constructor["entry_execution_profile"])
        original_model_state = state["model_state"]
        model.load_state_dict(original_model_state, strict=True)
        del state
        model.require_input_normalization_state()
        trainer._require_prefix_online_function(model)
        online_digest = trainer._model_state_sha256(model)
        _require(online_digest == session_scope["online_model_state_sha256"] and all(p.grad is None for p in model.parameters()), "MODEL_INITIAL_STATE")
        modes = [(module, module.training) for module in model.modules()]
        model.eval()
        inputs = {key: value.to(device_object) for key, value in cpu_inputs.items() if key != "mtf"}
        inputs["mtf"] = {key: value.to(device_object) for key, value in cpu_inputs["mtf"].items()}
        input_sha = audit.dynamic_compact_input_sha256(inputs)
        def forward(values):
            nonlocal forwards
            _require(forwards < plan["max_forward_calls"] and time.monotonic() - started < plan["max_wall_seconds"], "FORWARD_OR_WALL_BUDGET")
            result = trainer._model_forward_fp32(model, values["seq_x"].to(device_object), values["snap_x"].to(device_object),
                ctx_cat=values["ctx_cat"].to(device_object), ctx_cont=values["ctx_cont"].to(device_object),
                **{key: value.to(device_object) for key, value in values["mtf"].items()})
            forwards += 1
            _require("entry_action_q_bps" not in result and not any(key.startswith("unified_exit") for key in result), "LEGACY_OR_EXIT_OUTPUT")
            return result
        vjp = audit.dynamic_compact_input_vjps(forward, inputs, layout=layout)
        baseline = vjp["outputs"]
        _require(baseline["dynamic_query_horizon_minutes"].numel() == 480
                 and all(bool((baseline[key][..., 2] == 0).all()) for key in
                         ("dynamic_terminal_net_markout_bps", "dynamic_entry_action_quality_bps_candidate", "dynamic_entry_quality_bps_candidate")), "RAW_FLAT_AND_480_QUERIES")
        _require(vjp["forward_calls"] == 1 and vjp["reverse_calls"] == 13, "VJP_COUNTS")
        permutation = torch.arange(15, -1, -1, device=device_object)
        reordered = {key: value.index_select(0, permutation) for key, value in inputs.items() if key != "mtf"}
        reordered["mtf"] = {key: value.index_select(0, permutation) for key, value in inputs["mtf"].items()}
        with torch.no_grad():
            parity = audit.require_dynamic_compact_output_parity(baseline, forward(reordered), inverse_permutation=permutation)
        valid = [record for record in manifest["cases"] if record["status"] == "valid"]
        def cases():
            for record in valid:
                case = audit.make_dynamic_compact_counterfactual(cpu_inputs, layout=layout, token=record["token"],
                    witness_row=record["witness_row"], kind=record["kind"], normalization=normalized_view, time_index=None)
                _require(case["status"] == "valid" and case["route"] == record["route"]
                         and case["time_index"] == record["effective_time_index"]
                         and json.dumps(case["changed_coordinates"], sort_keys=True) == json.dumps(record["changed_coordinates"], sort_keys=True)
                         and audit.dynamic_compact_input_sha256(case["inputs"]) == record["input_sha256"], "FIXED_CASE_REGENERATION:" + record["token"])
                yield case
        responses = []
        for packed in audit.pack_dynamic_compact_cases(cases(), baseline_inputs=cpu_inputs):
            with torch.no_grad():
                actual = forward(packed["inputs"])
                response = audit.dynamic_compact_output_response(baseline, actual,
                    witness_rows=packed["witness_rows"], score_mask=packed["score_mask"].to(device_object))
            scored = sum(bool(value) for value in packed["score_mask"].tolist())
            margin_delta = response["candidate_margin_delta_bps"].detach().cpu().tolist()
            component_delta = {key: value.detach().cpu().tolist() for key, value in response["component_delta_bps"].items()}
            for index in range(scored):
                responses.append({"token": packed["case_tokens"][index], "witness_row": packed["witness_rows"][index],
                    "candidate_margin_delta_bps": margin_delta[index],
                    "component_delta_bps": {key: value[index] for key, value in component_delta.items()}})
            del actual, response, packed
        _require(len(responses) == len(valid) == manifest["valid_case_count"] and forwards == budget["forward_calls"]
                 and audit.dynamic_compact_input_sha256(inputs) == input_sha == manifest["input_sha256"]
                 and audit.dynamic_compact_input_sha256(cpu_inputs) == input_sha, "FIXED_SCHEDULE_COMPLETION")
        routes = _route_json(vjp["routes"], 3)
        baseline_json = {key: value if key == "dynamic_query_horizon_minutes" else value[:3] for key, value in baseline.items()}
        qualities = baseline["dynamic_entry_action_quality_bps_candidate"][:3]
        actions = []
        for row in range(3):
            maximum = qualities[row].max()
            ties = (qualities[row] == maximum).nonzero(as_tuple=False).flatten().tolist()
            actions.append({"physical_parent_row": metadata["unique_parent_rows"][row], "action_argmax_indices": ties,
                "unique_diagnostic_action": ("LONG", "SHORT", "FLAT")[ties[0]] if len(ties) == 1 else None,
                "tie_is_not_native_action_authority": len(ties) != 1})
        execution = {"schema_version": "gx1_dynamic_compact_native_input_audit_result_v1",
            "status": "PASS_FIXED_NATIVE_INPUT_SENSITIVITY_NOT_LEARNING_OR_ADMISSION", "plan": plan_ref,
            "input_scope": plan["input_scope"], "input_packet": scope["input_packet"], "case_manifest": scope["case_manifest"],
            "baseline_full480": _json_value(baseline_json), "baseline_query_count": 480, "raw_analytic_FLAT_zero": True,
            "genuine_rows": 3, "padding_rows": 13, "baseline_actions_diagnostic_only": actions,
            "vjp_routes": _json_value(routes), "finite_case_responses": responses,
            "all_owner_inventory": manifest["cases"], "counterfactual_status_counts": manifest["status_counts"],
            "parity": _json_value(parity), "actual_forward_calls": forwards, "actual_reverse_calls": 13,
            "budget": budget, "input_only_no_targets_provider_dataset_optimizer": True,
            "online_model_state_sha256": online_digest, "input_normalization_sha256": normalization["contract_sha256"],
            "loss_RMS_identity_only_no_loss_or_fit": prefix["dynamic_m5_loss_scales"],
            "hardware": hardware, "lock_ancestry": lock, "executing_numerical_threads": 1,
            "limits": ["Three genuine TRAIN witnesses; thirteen unscored padding rows.",
                       "Current finite counterfactuals and historical VJPs are separate sensitivity evidence.",
                       "No historical finite-case, market-realizability, learned utility, generalization, edge or native action admission is established."],
            "optimizer_updates": 0, "normalization_refits": 0, "source_policy_outer_guards_pending": True,
            **{key: False for key in FLAGS}}
    finally:
        # Restoration is unconditional, even when a preservation comparison fails.
        # Report changed state as a hard failure instead of hiding its restoration.
        preservation_errors = []
        try:
            if model is not None:
                changed_grads = any(parameter.grad is not None for parameter in model.parameters())
                if changed_grads:
                    preservation_errors.append("PARAMETER_GRADS_CHANGED")
                    for parameter in model.parameters():
                        parameter.grad = None
                if online_digest is not None and trainer._model_state_sha256(model) != online_digest:
                    preservation_errors.append("ONLINE_WEIGHTS_CHANGED")
                    model.load_state_dict(original_model_state, strict=True)
                model.require_input_normalization_state()
        except Exception as error:
            preservation_errors.append(type(error).__name__ + ":" + str(error))
        finally:
            if modes is not None:
                for module, mode in modes:
                    module.training = mode
            try:
                trainer._restore_attended_session_rng_state(rng, device=device_object)
                _require(_rng_sha(trainer._attended_session_rng_state(device=device_object)) == rng_digest, "RNG_NOT_RESTORED")
            except Exception as error:
                preservation_errors.append(type(error).__name__ + ":" + str(error))
        try:
            torch.set_num_threads(original_threads)
            _require(torch.get_num_threads() == original_threads, "TORCH_THREADS_NOT_RESTORED")
        except Exception as error:
            preservation_errors.append(type(error).__name__ + ":" + str(error))
        for key, before in session_identities.items():
            try:
                path = _reference(session_scope[key])
                _require(_identity(path) == before and trainer._sha256_file(path) == session_scope[key]["sha256"]
                         and _identity(path) == before, "SESSION_FILE_CHANGED:" + key)
            except Exception as error:
                preservation_errors.append(type(error).__name__ + ":" + str(error))
        try:
            _require(_identity(_reference(scope["input_packet"])) == packet_identity, "INPUT_PACKET_CHANGED")
            _require(all(_identity(_reference(ref)) == identity for ref, identity in metadata_proofs), "EXTERNAL_METADATA_CHANGED")
            frozen()
        except Exception as error:
            preservation_errors.append(type(error).__name__ + ":" + str(error))
        _require(not preservation_errors, "PRESERVATION:" + ";".join(preservation_errors))
    _require(execution is not None and time.monotonic() - started < plan["max_wall_seconds"], "FINAL_COMPLETION")
    execution.update(ONLINE_weights_preserved=True, parameter_grads_preserved=True, modes_restored=True,
                     RNG_restored=True, torch_threads_restored=True, session_files_preserved=True, source_policy_preserved=True,
                     elapsed_seconds=time.monotonic() - started)
    output.mkdir()
    result_ref = _publish(output, "RESULT.json", execution)
    return {"status": execution["status"], "result": result_ref, "budget": budget,
            "actual_forward_calls": forwards, "actual_reverse_calls": 13,
            "valid_cases_executed": len(responses), "genuine_rows": 3, "padding_rows": 13,
            "source_policy_outer_guards_pending": True}
