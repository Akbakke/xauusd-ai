"""One-use TRAIN-only native five-endpoint zero-step source audit.

A separately admitted CUDA producer dispatch calls run(). This module does not
fit, score held-out outcomes, read TEST, or grant model-launch authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
POLICY = REPO / "NEXT_RUN_POLICY.json"
STAGE = "market_state_five_endpoint_zero_step_20261107_003"
SCHEMA = "gx1_market_state_five_endpoint_zero_step_plan_v1"
RUNS = Path("/home/andre2/GX1_RUNS")
FIXED = {
    "input_plan": (REPO / "configs/research/DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_003.json",
                   "4a80b5d15f16e6a790de3a80155043e9fc2177a329256d67f4985597384f8fa5"),
    "input_result": (RUNS / "DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_003/RESULT.json",
                     "3177fee2339bfc81354088cae50f22ac31d191fd4ad7c64c2d047bd68c682b71"),
    "input_terminal": (RUNS / "DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_003/TERMINAL.json",
                       "0ac8a8926a08f0044c30e38df0d257720edcbb5740f2439bbb730d12fbb81300"),
    "normalization": (RUNS / "DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_003/PREFIX_NORMALIZATION.json",
                      "9cbde23587e2296365b799d9f288c27ce3cd9b4023baa19cc77aad3476076109"),
    "constructor": (RUNS / "NATIVE_M5_EARLY_MFE_FRESH_PREPARATION_20261022_187/CONSTRUCTOR_METADATA.json",
                    "8d0fc19b646b142dced37ad86dd0e43a71bb8efe299a9ba4fd2de08e324db2bf"),
    "old_view": (RUNS / "DYNAMIC_M5_OCTOBER_VIEW_BINDING_20261106_001/CONTROL_VIEW.json",
                 "c0d64f1e0f16ffb8a85942567858b5e2a6e6b04482c901472bdc2789d6aec9db"),
    "observer_axes": (RUNS / "DYNAMIC_M5_OCTOBER_VIEW_BINDING_20261106_001/OCTOBER_OBSERVER_AXES.json",
                      "3693ca3c259b7093148c78069cf1e868837441000a12d5db750f3bef8e495a83"),
}
NEW_SOURCE_KEYS = {
    "direction_component_source": "gx1/contracts/entry_native_direction_research_v1.py",
    "five_endpoint_model_source": "gx1/contracts/market_state_five_endpoint_model_v1.py",
    "five_endpoint_contract_source": "gx1/contracts/market_state_five_endpoint_contract_v1.py",
    "five_endpoint_target_binding_source": "gx1/contracts/market_state_five_endpoint_target_binding_v1.py",
}
REQUIRED_OWNERS = frozenset({
    "gx1/audit/verify_market_state_five_endpoint_zero_step_v1.py",
    "gx1/audit/dynamic_mtf_october_input_only.py",
    "gx1/contracts/entry_native_direction_feature_view_v1.py",
    "gx1/contracts/market_state_five_endpoint_model_v1.py",
    "gx1/contracts/market_state_five_endpoint_contract_v1.py",
    "gx1/contracts/market_state_five_endpoint_target_binding_v1.py",
    "gx1/contracts/entry_native_direction_research_v1.py",
    "gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
    "gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py",
    "gx1/contracts/gx1_capped_execution_v1.py",
    "gx1/contracts/model_state_digest_v1.py",
    "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
    "gx1/contracts/entry_native_direction_prefix_units_v1.py",
    "gx1/contracts/local_random_access_campaign_v2.py",
    "gx1/contracts/immutable_event_authority_v1.py",
    "gx1/audit/dynamic_mtf_train_prefix_inputs.py",
})
FEATURE_KEYS = frozenset({"seq_x", "snap_x", "ctx_cat", "ctx_cont"}
    | {prefix + tf for prefix in ("seq_", "gc_seq_")
       for tf in ("m5", "m15", "h1", "h4", "d1")})


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("FIVE_ENDPOINT_ZERO_STEP_" + reason)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path, digest: str) -> dict:
    return {"path": str(path), "sha256": digest}


def _fixed_ref(name: str) -> dict:
    return _ref(*FIXED[name])


def _read_ref(ref: dict, *, max_bytes: int = 8 * 1024 * 1024) -> dict:
    _need(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "REF_SCHEMA")
    _need(isinstance(ref["path"], str) and isinstance(ref["sha256"], str)
          and len(ref["sha256"]) == 64, "REF_TYPES")
    path = Path(ref["path"])
    _need(path.is_absolute() and path.is_file() and 0 < path.stat().st_size <= max_bytes
          and _sha(path) == ref["sha256"], "REF_HASH")
    return json.loads(path.read_text())


def _write(root: Path, name: str, value: dict, *, limit: int) -> dict:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _need(len(data) <= limit, "PUBLICATION_SIZE")
    _need(json.loads(data) == value, "PUBLICATION_STRICT_LOAD")
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    path = root / name
    _publish_bytes(path, data)
    return _ref(path, hashlib.sha256(data).hexdigest())


def _source_refs(old_view: dict, owners: dict, mask_ref: dict) -> dict:
    sources = dict(old_view["source_bindings"])
    _need(sources.pop("dynamic_state_target_source", None) is not None,
          "OLD_VIEW_TARGET_SOURCE")
    for key, path in NEW_SOURCE_KEYS.items():
        sources[key] = _ref(REPO / path, owners[path])
    sources["five_endpoint_corrected_mask"] = mask_ref
    for key, ref in sources.items():
        if key != "five_endpoint_corrected_mask":
            _need(_sha(Path(ref["path"])) == ref["sha256"], "VIEW_SOURCE:" + key)
    return sources


def _qualify_plan(plan: dict, plan_ref: dict, policy: dict | None, *, head: str,
                  out_dir: Path, device: str, cap: dict | None) -> None:
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
    _need(plan.get("schema_version") == SCHEMA and plan.get("stage") == STAGE
          and plan.get("status") == "PREPARED_NOT_ADMITTED"
          and type(plan.get("source_base_head")) is str
          and len(plan["source_base_head"]) == 40
          and subprocess.run(["git", "merge-base", "--is-ancestor",
                              plan["source_base_head"], head],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
          and plan.get("output_root") == str(RUNS / STAGE.upper()) == str(out_dir)
          and plan.get("max_invocations") == 1
          and plan.get("fold_id") == 1 and plan.get("partition") == "fit"
          and plan.get("sample_parent_rows") == [0]
          and type(plan.get("seed")) is int and 0 <= plan["seed"] < 2**32
          and plan.get("model_forwards_authorized") == 1
          and plan.get("optimizer_steps_authorized") == 0
          and plan.get("training_authorized") is False
          and plan.get("test_data_used") is False
          and device == "cuda" and out_dir.is_absolute() and not out_dir.exists(), "PLAN_SCOPE")
    _need(all(plan.get(name) == _fixed_ref(name) for name in FIXED)
          and plan.get("corrected_mask") == _ref(Path(target.REFS["mask"][0]), target.REFS["mask"][1]),
          "FIXED_INPUT_REFS")
    _need(isinstance(plan.get("source_owners"), dict)
          and REQUIRED_OWNERS <= set(plan["source_owners"])
          and all(isinstance(path, str) and not Path(path).is_absolute()
                  and ".." not in Path(path).parts and isinstance(digest, str)
                  and len(digest) == 64 and (REPO / path).is_file()
                  and _sha(REPO / path) == digest
                  for path, digest in plan["source_owners"].items()), "SOURCE_OWNERS")
    _need(isinstance(plan.get("expected_view_sha256"), str)
          and len(plan["expected_view_sha256"]) == 64
          and isinstance(plan.get("expected_objective_sha256"), str)
          and len(plan["expected_objective_sha256"]) == 64
          and isinstance(plan.get("max_json_bytes"), int)
          and 1024 <= plan["max_json_bytes"] <= 1024 * 1024, "EXPECTED_IDENTITIES")
    limits = plan.get("limits")
    _need(isinstance(limits, dict)
          and limits.get("memory_max") == 8 * 1024**3
          and limits.get("memory_high") == 8 * 1024**3
          and limits.get("swap") == 512 * 1024**2
          and type(limits.get("max_wall_seconds")) is int
          and limits["max_wall_seconds"] == 3600, "CAPS")
    if cap is None:
        return
    _need(all(limits[key] == cap.get(key) for key in ("memory_max", "memory_high", "swap"))
          and type(cap.get("pids")) is int and cap["pids"] <= 128, "ACTUAL_CAPS")
    _need(isinstance(policy, dict), "POLICY_SCHEMA")
    gate = policy.get(STAGE)
    expected = {"decision": "ADMITTED", "plan": plan_ref,
                "execution_authorized": True, "data_reads_authorized": True,
                "model_forwards_authorized": 1, "optimizer_steps_authorized": 0,
                "training_authorized": False, "test_authorized": False,
                "claimed_invocations": 0, "completed_invocations": 0,
                "max_invocations": 1, "output_root": str(out_dir)}
    _need(policy.get("training_enabled") is False
          and policy.get("full_val_allowed") is False
          and isinstance(gate, dict)
          and all(gate.get(k) == v for k, v in expected.items()), "POLICY")


def _clean_head() -> str:
    _need(Path.cwd().resolve() == REPO
          and subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
          == "work/gx1-current"
          and subprocess.check_output(["git", "status", "--porcelain"], text=True) == "",
          "CLEAN_REPO")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _checked_plan_ref(plan_path: Path, plan_sha256: str) -> tuple[dict, dict]:
    _need(plan_path == REPO / "configs/research" / (STAGE.upper() + ".json")
          and plan_path.is_absolute() and plan_path.is_file()
          and isinstance(plan_sha256, str) and len(plan_sha256) == 64
          and _sha(plan_path) == plan_sha256, "PLAN_HASH")
    ref = _ref(plan_path, plan_sha256)
    return _read_ref(ref), ref


def _inspect_reusable_receipts() -> None:
    input_plan = _read_ref(_fixed_ref("input_plan"))
    input_result = _read_ref(_fixed_ref("input_result"))
    input_terminal = _read_ref(_fixed_ref("input_terminal"))
    _need(input_result.get("plan") == _fixed_ref("input_plan")
          and input_result.get("prefix_normalization") == _fixed_ref("normalization")
          and input_result.get("model_calls") == 0
          and input_result.get("fit_allowed") is True
          and input_result.get("remaining_blockers") == []
          and input_terminal.get("result") == _fixed_ref("input_result")
          and input_terminal.get("exit_code") == 0
          and input_terminal.get("model_calls") == 0
          and input_plan.get("fit_prefix", {}).get("rows") == 29849
          and input_plan["fit_prefix"].get("cutoff_ns") == 1727740800000000000
          and input_plan.get("evaluation_axis", {}).get("parent_row_range") == [29849, 36172],
          "OCTOBER_INPUT_RECEIPTS")
    _need(isinstance(input_plan.get("source_files_sha256"), dict)
          and len(input_plan["source_files_sha256"]) == 23, "OCTOBER_SOURCE_COUNT")
    for path, digest in input_plan["source_files_sha256"].items():
        _need(isinstance(path, str) and ".." not in Path(path).parts
              and not Path(path).is_absolute() and _sha(REPO / path) == digest,
              "OCTOBER_CURRENT_SOURCE:" + path)
    for key in ("ancestral_264_plan", "ancestral_267_plan"):
        _read_ref(input_plan[key], max_bytes=1024 * 1024)
    for key in ("ancestral_264_worker", "ancestral_267_worker"):
        ref = input_plan[key]
        _need(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
              and isinstance(ref["path"], str) and isinstance(ref["sha256"], str),
              "ANCESTRAL_WORKER_REF:" + key)
        path = Path(ref["path"])
        _need(path.is_absolute() and path.is_file()
              and 0 < path.stat().st_size <= 1024 * 1024
              and _sha(path) == ref["sha256"], "ANCESTRAL_WORKER_HASH:" + key)
    from gx1.audit import dynamic_mtf_october_input_only as prior
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        prior._load_module(input_plan["ancestral_264_worker"],
                           "gx1_five_endpoint_264_preclaim_import")
    finally:
        sys.dont_write_bytecode = previous_bytecode


def _bind_view_metadata(plan: dict) -> tuple[dict, dict, dict, dict, dict]:
    """Bind small source receipts and exact five-endpoint identity before claim."""
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_prefix_units_v1 as units

    normal = _read_ref(_fixed_ref("normalization"))
    constructor = _read_ref(_fixed_ref("constructor"))
    old_view = _read_ref(_fixed_ref("old_view"))
    observer_scope = _read_ref(_fixed_ref("observer_axes"))
    profile = units.october_prefix_units_profile()
    _need(normal["unobserved_units_profile"] == profile
          and old_view["prefix_normalization"]["binding"] == _fixed_ref("normalization")
          and old_view["constructor_metadata"]["binding"] == _fixed_ref("constructor")
          and old_view["october_observer_axes"] == observer_scope,
          "PROFILE_AND_VIEW_ANCESTRY")
    sources = _source_refs(old_view, plan["source_owners"], plan["corrected_mask"])
    view = views.bind_native_direction_feature_view(
        prefix_normalization=normal, prefix_normalization_ref=_fixed_ref("normalization"),
        constructor_metadata=constructor, constructor_metadata_ref=_fixed_ref("constructor"),
        source_bindings=sources, max_batch_rows=16, full_feature_profile=True,
        market_state_five_endpoint=True, unobserved_units_profile=profile,
        october_observer_axes=observer_scope)
    _need(view["contract_sha256"] == plan["expected_view_sha256"]
          and view["research_objective"]["contract_sha256"] == plan["expected_objective_sha256"],
          "NEW_VIEW_IDENTITY")
    return view, normal, constructor, observer_scope, profile


def preflight_read_only(plan_path: Path, plan_sha256: str,
                        out_dir: Path) -> dict:
    """Check small receipts and current owners without CUDA, claim, or payload reads."""
    head = _clean_head()
    plan, ref = _checked_plan_ref(Path(plan_path), plan_sha256)
    _qualify_plan(plan, ref, None, head=head, out_dir=Path(out_dir),
                  device="cuda", cap=None)
    _inspect_reusable_receipts()
    view, _, _, _, _ = _bind_view_metadata(plan)
    return {"status": "READY_NO_CLAIM_NO_PAYLOAD", "plan": ref,
            "view_sha256": view["contract_sha256"],
            "objective_sha256": view["research_objective"]["contract_sha256"],
            "source_head": head, "source_owner_refs_checked": len(plan["source_owners"]),
            "october_source_refs_checked": 23, "claim_created": False,
            "model_forwards": 0, "optimizer_steps": 0, "test_data_used": False}


def _preflight(plan_path: Path, plan_sha256: str, device: str,
               out_dir: Path) -> tuple[dict, dict, dict, str]:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_producer_execution, require_capped_lock_ancestry,
    )
    head = _clean_head()
    plan, plan_ref = _checked_plan_ref(plan_path, plan_sha256)
    policy_ref = _ref(POLICY, _sha(POLICY))
    policy = _read_ref(policy_ref)
    cap = require_guarded_cuda_producer_execution()
    cap["lock"] = require_capped_lock_ancestry()
    _qualify_plan(plan, plan_ref, policy, head=head, out_dir=out_dir,
                  device=device, cap=cap)
    _inspect_reusable_receipts()
    _bind_view_metadata(plan)
    return plan, plan_ref, policy_ref, head



def _readonly_sequence_source_headers(source: dict) -> dict:
    """Seal the two writable October clock arrays before the strict gather."""
    import numpy as np
    _need(isinstance(source, dict)
          and set(source) == {"signal", "times_ns", "sample_positions", "audit"},
          "SEQUENCE_SOURCE_INPUT")
    sealed = dict(source)
    for key in ("times_ns", "sample_positions"):
        values = source[key]
        _need(isinstance(values, np.ndarray) and values.dtype == np.int64
              and values.ndim == 1, "SEQUENCE_HEADER_INPUT:" + key)
        sealed[key] = np.frombuffer(values.tobytes(order="C"), dtype=np.int64)
    return sealed


def _one_forward(plan: dict) -> dict:
    """Decode target-free fit inputs, forward once, then read corrected targets."""
    import numpy as np
    import torch
    from gx1.audit import dynamic_mtf_october_input_only as prior
    from gx1.audit.dynamic_mtf_train_prefix_inputs import _check_payload_stats
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    input_plan = _read_ref(_fixed_ref("input_plan"))
    view, normal, constructor, observer_scope, profile = _bind_view_metadata(plan)
    provider_plan = prior._read_ref(input_plan["ancestral_264_plan"])
    prior._read_ref(input_plan["ancestral_267_plan"])
    provider = prior._load_module(input_plan["ancestral_264_worker"], "gx1_five_endpoint_264_reader")
    with campaign.file_verification_scope():
        info = provider.validate_metadata(provider_plan)
        _check_payload_stats(provider_plan["payloads"])
        axes, full = provider.read_axes(provider_plan, info, np)
        _need(full == input_plan["full_axes"], "FULL_AXES")
        prior._axis_receipt(axes, full, input_plan["fit_prefix"], input_plan["evaluation_axis"])
        fit_rows = input_plan["fit_prefix"]["rows"]
        source_plan = dict(provider_plan,
                           october_gc_transition=input_plan["current_gc_transition"],
                           october_gc_layout=input_plan["current_gc_layout"])
        physical = prior._prepare_prefix_inputs(
            provider, source_plan, info, axes, fit_rows=fit_rows,
            last_issue_ns=int(axes["decision_ns"][fit_rows - 1]))
        snapshot = campaign.capture_verified_file_snapshot()
    selected = np.asarray(plan["sample_parent_rows"], dtype=np.int64)
    fit_axes = {key: axes[key][:fit_rows] for key in ("parent_rows", "bar_start_ns", "decision_ns")}
    observer = views.NativeDirectionObserverAxes(
        view, source=observer_scope["source"], role="fit",
        unobserved_units_profile=profile, **fit_axes)
    raw = views.gather_native_direction_feature_batch(
        view, parent_rows=observer.parent_rows,
        bar_start_ns=observer.bar_start_ns, decision_ns=observer.decision_ns,
        entry_sequence_source=_readonly_sequence_source_headers(physical["entry_sequence_source"]),
        snap_x=physical["current"]["snap"],
        ctx_cont=physical["current"]["ctx_cont"],
        ctx_cat=physical["current"]["ctx_cat"], multi_tf_sources=physical["mtf"],
        gc_surfaces=physical["gc"], gc_layout=physical["gc_layout"],
        batch_parent_rows=selected, observer_axes=observer,
        unobserved_units_profile=profile)
    _need(set(raw["features"]) == FEATURE_KEYS
          and raw["witness"]["parent_rows"] == selected.tolist(), "FEATURE_ONLY_BATCH")
    device = torch.device("cuda")
    _need(torch.cuda.is_available(), "CUDA_AVAILABLE")
    trainer._set_deterministic(plan["seed"], device, trainer.DETERMINISTIC_FP32)
    model = views.build_native_direction_research_model(
        view, prefix_normalization=normal, constructor_metadata=constructor,
        device=device, unobserved_units_profile=profile)
    from gx1.contracts.market_state_five_endpoint_model_v1 import FiveEndpointMarketStateModel
    _need(type(model) is FiveEndpointMarketStateModel
          and model.feature_view_identity["contract_sha256"] == view["contract_sha256"],
          "FRESH_MODEL")
    model.eval()
    initial = canonical_model_state_sha256(model.state_dict())
    features = {key: torch.from_numpy(value).to(device) for key, value in raw["features"].items()}
    _need(set(features) == FEATURE_KEYS, "NO_TARGET_FORWARD_INPUT")
    with torch.no_grad(), trainer._training_mha_fastpath_context(model.backbone):
        outputs = trainer._model_forward_fp32(model, **features)
    _need(set(outputs) == {"logits"}, "OUTPUT_KEYS")
    logits = outputs["logits"]
    _need(logits.dtype == torch.float32 and logits.shape == (1, 5, 3)
          and bool(torch.isfinite(logits).all()), "OUTPUT_SCHEMA")
    p = torch.softmax(logits, dim=-1)
    _need(bool(torch.isfinite(p).all())
          and bool(torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), rtol=0, atol=1e-6)),
          "PROBABILITY_SIMPLEX")
    after = canonical_model_state_sha256(model.state_dict())
    _need(initial == after and all(parameter.grad is None for parameter in model.parameters()),
          "ZERO_UPDATE_STATE")
    # Future labels are opened only after the single feature-only forward.
    bound = target.load_bound_five_endpoint_targets()
    issue = np.asarray([raw["witness"]["decision_ns"][0]], dtype=np.int64)
    truth = bound.gather(selected, issue, fold_id=1, partition="fit")
    _need(truth["allowed"].shape == truth["eligible"].shape == truth["mature"].shape == (1, 5),
          "TARGET_SHAPE")
    snapshot.require_unchanged()
    _check_payload_stats(provider_plan["payloads"])
    return {"audit_scope_limit": "One TRAIN fit row proves only the bound route, shapes and finiteness; it does not prove numerical consumption of all five timeframes or all indicator fields, learning, generalization, or economic edge.",
            "source_refs": {"input_plan": _fixed_ref("input_plan"),
                            "normalization": _fixed_ref("normalization"),
                            "corrected_mask": plan["corrected_mask"]},
            "fresh_unfitted_model": True, "checkpoint_loaded": False,
            "view_sha256": view["contract_sha256"],
            "objective_sha256": view["research_objective"]["contract_sha256"],
            "selected_parent_rows": selected.tolist(), "issue_ns": int(issue[0]),
            "feature_witness": raw["witness"],
            "feature_shapes": {key: list(value.shape) for key, value in features.items()},
            "feature_dtypes": {key: str(value.dtype) for key, value in features.items()},
            "finite_features": True, "output_shape": list(logits.shape),
            "output_dtype": str(logits.dtype), "finite_logits": True,
            "max_probability_sum_error": float((p.sum(-1) - 1).abs().max().item()),
            "initial_model_state_sha256": initial, "after_model_state_sha256": after,
            "target_allowed": truth["allowed"][0].tolist(),
            "target_eligible": truth["eligible"][0].tolist(),
            "target_mature": truth["mature"][0].tolist(),
            "unjoined_target_rows": bound.unjoined_target_rows,
            "source_snapshot_files": snapshot.file_count,
            "model_forwards": 1, "optimizer_steps": 0, "test_data_used": False}


def run(plan_path: Path, plan_sha256: str, device: str, out_dir: Path) -> dict:
    """Admitted dispatcher entry point; an existing output root is never reused."""
    from gx1.contracts.immutable_event_authority_v1 import _fsync_directory

    plan, plan_ref, policy_ref, head = _preflight(
        Path(plan_path), plan_sha256, device, Path(out_dir))
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    claim = None
    try:
        _fsync_directory(root.parent)
        claim = _write(root, "CLAIM.json", {
            "schema_version": "gx1_market_state_five_endpoint_zero_step_claim_v1",
            "plan": plan_ref, "policy": policy_ref, "source_head": head,
            "selected_parent_rows": [0], "invocation": 1}, limit=plan["max_json_bytes"])
        result = _one_forward(plan)
        _need(_sha(Path(plan_ref["path"])) == plan_ref["sha256"]
              and _sha(Path(policy_ref["path"])) == policy_ref["sha256"]
              and subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == head
              and subprocess.check_output(["git", "status", "--porcelain"], text=True) == "",
              "FINAL_SOURCE_IDENTITY")
        result_ref = _write(root, "RESULT.json", {
            "schema_version": "gx1_market_state_five_endpoint_zero_step_result_v1",
            "decision": "PASS_ZERO_STEP_SOURCE_AND_OUTPUT_ONLY", "plan": plan_ref,
            "claim": claim, "source_head": head, "status": "COMPLETE",
            "elapsed_seconds": time.monotonic() - began, **result}, limit=plan["max_json_bytes"])
        terminal = _write(root, "TERMINAL.json", {
            "schema_version": "gx1_market_state_five_endpoint_zero_step_terminal_v1",
            "status": "COMPLETE", "exit_code": 0, "plan": plan_ref, "claim": claim,
            "result": result_ref, "model_forwards": 1, "optimizer_steps": 0,
            "test_data_used": False, "no_relaunch_under_this_plan": True,
            "output_inventory": ["CLAIM.json", "RESULT.json", "TERMINAL.json"]},
            limit=plan["max_json_bytes"])
        return {"result": result_ref, "terminal": terminal}
    except Exception as exc:
        failure = _write(root, "FAILED.json", {
            "schema_version": "gx1_market_state_five_endpoint_zero_step_failure_v1",
            "plan": plan_ref, "claim": claim, "source_head": head,
            "error_type": type(exc).__name__, "error": str(exc)[:2048],
            "no_relaunch_under_this_plan": True}, limit=plan["max_json_bytes"])
        _write(root, "TERMINAL.json", {
            "schema_version": "gx1_market_state_five_endpoint_zero_step_terminal_v1",
            "status": "FAILED", "exit_code": 1, "plan": plan_ref,
            "claim": claim, "failure": failure,
            "model_forwards_upper_bound": 1, "optimizer_steps": 0,
            "test_data_used": False, "no_relaunch_under_this_plan": True,
            "output_inventory": (["CLAIM.json", "FAILED.json", "TERMINAL.json"]
                                 if claim else ["FAILED.json", "TERMINAL.json"])},
            limit=plan["max_json_bytes"])
        raise
