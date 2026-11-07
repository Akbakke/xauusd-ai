"""One-use TRAIN fit gradient witness for the native five-endpoint market-state model.

The source prepares no plan and grants no launch authority. A separately admitted,
guarded CUDA producer may call run exactly once. No optimizer, checkpoint, TEST,
Entry, Exit, or trading path is present.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import verify_market_state_five_endpoint_zero_step_v1 as zero

REPO = zero.REPO
RUNS = zero.RUNS
POLICY = zero.POLICY
STAGE = "market_state_five_endpoint_gradient_bridge_20261107_001"
SCHEMA = "gx1_market_state_five_endpoint_gradient_bridge_plan_v1"
OUT_ROOT = RUNS / STAGE.upper()
MEMORY_BYTES = 12 * 1024**3
SWAP_BYTES = 512 * 1024**2
WALL_SECONDS = 3600
SAMPLE_ROWS = [0, 29848]
ISSUE_NS = [1714521900000000000, 1727740500000000000]
EXECUTION = {
    "class": "producer", "cuda_producer": True, "device": "cuda",
    "module": "gx1.scripts.validate_entry_model_native_technical_checkpoint_v1",
    "flag": "--market-state-five-endpoint-gradient-bridge-plan",
    "memory": "12G", "swap": "512M", "model_forwards": 1,
    "backward_calls": 1, "optimizer_steps": 0,
}
PRIOR_ZERO = {
    "result": zero._ref(
        RUNS / "MARKET_STATE_FIVE_ENDPOINT_ZERO_STEP_20261107_003/RESULT.json",
        "c787c27de60788b30dea83acb859130ea63aa82a68c3a537aa899b7e925c5d91"),
    "terminal": zero._ref(
        RUNS / "MARKET_STATE_FIVE_ENDPOINT_ZERO_STEP_20261107_003/TERMINAL.json",
        "25a1cd3a235730594437935810a298ca6b8e301fe3be332c05b6a663c6a7e7d6"),
}
REQUIRED_OWNERS = zero.REQUIRED_OWNERS | frozenset({
    "gx1/audit/verify_market_state_five_endpoint_gradient_bridge_v1.py",
    "gx1/contracts/market_state_five_endpoint_batch_v1.py",
    "scripts/gx1_capped_run.sh",
    "scripts/gx1_guarded_trainer_exec.sh",
})


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError("FIVE_ENDPOINT_GRADIENT_BRIDGE_" + reason)


def _qualify_plan(plan: dict, plan_ref: dict, policy: dict | None, *,
                  head: str, out_dir: Path, device: str, cap: dict | None) -> None:
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target

    _need(isinstance(plan, dict)
          and plan.get("schema_version") == SCHEMA and plan.get("stage") == STAGE
          and plan.get("status") == "PREPARED_NOT_ADMITTED"
          and plan.get("source_base_head") and type(plan["source_base_head"]) is str
          and len(plan["source_base_head"]) == 40
          and subprocess.run(["git", "merge-base", "--is-ancestor",
                              plan["source_base_head"], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
          and plan.get("output_root") == str(OUT_ROOT) == str(out_dir)
          and out_dir.is_absolute() and not out_dir.exists()
          and device == "cuda"
          and plan.get("fold_id") == 1 and plan.get("partition") == "fit"
          and plan.get("sample_parent_rows") == SAMPLE_ROWS
          and plan.get("sample_issue_ns") == ISSUE_NS
          and plan.get("max_invocations") == 1
          and plan.get("model_forwards_authorized") == 1
          and plan.get("backward_calls_authorized") == 1
          and plan.get("optimizer_steps_authorized") == 0
          and plan.get("training_authorized") is False
          and plan.get("test_data_used") is False
          and type(plan.get("seed")) is int and 0 <= plan["seed"] < 2**32
          and plan.get("execution") == EXECUTION, "PLAN_SCOPE")
    _need(all(plan.get(name) == zero._fixed_ref(name) for name in zero.FIXED)
          and plan.get("corrected_mask") == zero._ref(
              Path(target.REFS["mask"][0]), target.REFS["mask"][1])
          and plan.get("prior_zero_result") == PRIOR_ZERO["result"]
          and plan.get("prior_zero_terminal") == PRIOR_ZERO["terminal"],
          "FIXED_REFS")
    _need(isinstance(plan.get("source_owners"), dict)
          and REQUIRED_OWNERS <= set(plan["source_owners"])
          and all(isinstance(path, str) and not Path(path).is_absolute()
                  and ".." not in Path(path).parts and isinstance(digest, str)
                  and len(digest) == 64 and (REPO / path).is_file()
                  and zero._sha(REPO / path) == digest
                  for path, digest in plan["source_owners"].items()), "SOURCE_OWNERS")
    _need(isinstance(plan.get("expected_view_sha256"), str)
          and len(plan["expected_view_sha256"]) == 64
          and isinstance(plan.get("expected_objective_sha256"), str)
          and len(plan["expected_objective_sha256"]) == 64
          and type(plan.get("max_json_bytes")) is int
          and 1024 <= plan["max_json_bytes"] <= 1024 * 1024, "IDENTITIES")
    limits = plan.get("limits")
    _need(isinstance(limits, dict)
          and limits == {"memory_max": MEMORY_BYTES, "memory_high": MEMORY_BYTES,
                         "swap": SWAP_BYTES, "max_wall_seconds": WALL_SECONDS},
          "CAPS")
    if cap is None:
        return
    _need(all(cap.get(k) == limits[k] for k in ("memory_max", "memory_high", "swap"))
          and type(cap.get("pids")) is int and cap["pids"] <= 128
          and cap.get("class") == "producer"
          and cap.get("execution_mode") == "cuda_producer"
          and isinstance(cap.get("cgroup_path"), str), "ACTUAL_CAPS")
    lock = cap.get("lock")
    _need(isinstance(lock, dict) and isinstance(lock.get("scope"), str)
          and lock["scope"].startswith("/") and lock["scope"].endswith(".scope")
          and cap["cgroup_path"] == "/sys/fs/cgroup" + lock["scope"],
          "LOCK_CGROUP_SCOPE")
    expected_gate = {
        "decision": "ADMITTED", "plan": plan_ref,
        "execution_authorized": True, "data_reads_authorized": True,
        "model_forwards_authorized": 1, "backward_calls_authorized": 1,
        "optimizer_steps_authorized": 0, "training_authorized": False,
        "test_authorized": False, "claimed_invocations": 0,
        "completed_invocations": 0, "max_invocations": 1,
        "output_root": str(out_dir), "relaunch_allowed": False,
        "checkpoint_reads_authorized": False, "broker_access_allowed": False,
        "test_data_used": False,
        "memory_max": "12G", "swap_max": "512M",
        "wall_seconds": WALL_SECONDS,
    }
    gate = policy.get(STAGE) if isinstance(policy, dict) else None
    _need(isinstance(policy, dict)
          and policy.get("training_enabled") is False
          and policy.get("native_training_enabled") is False
          and policy.get("native_launch_authorized") is False
          and policy.get("full_val_allowed") is False
          and isinstance(gate, dict)
          and all(gate.get(k) == value for k, value in expected_gate.items()),
          "POLICY")


def _checked_plan_ref(plan_path: Path, plan_sha256: str) -> tuple[dict, dict]:
    _need(plan_path == REPO / "configs/research" / (STAGE.upper() + ".json")
          and plan_path.is_absolute() and plan_path.is_file()
          and type(plan_sha256) is str and len(plan_sha256) == 64
          and zero._sha(plan_path) == plan_sha256, "PLAN_HASH")
    ref = zero._ref(plan_path, plan_sha256)
    return zero._read_ref(ref), ref


def _inspect_priors() -> None:
    zero._inspect_reusable_receipts()
    result = zero._read_ref(PRIOR_ZERO["result"])
    terminal = zero._read_ref(PRIOR_ZERO["terminal"])
    _need(result.get("decision") == "PASS_ZERO_STEP_SOURCE_AND_OUTPUT_ONLY"
          and result.get("status") == "COMPLETE"
          and result.get("model_forwards") == 1
          and result.get("optimizer_steps") == 0
          and result.get("test_data_used") is False
          and terminal.get("result") == PRIOR_ZERO["result"]
          and terminal.get("exit_code") == 0
          and terminal.get("no_relaunch_under_this_plan") is True,
          "PRIOR_ZERO_RECEIPT")


def preflight_read_only(plan_path: Path, plan_sha256: str,
                        out_dir: Path) -> dict:
    """Check small receipts and current source identities without payload or claim."""
    head = zero._clean_head()
    plan, ref = _checked_plan_ref(Path(plan_path), plan_sha256)
    _qualify_plan(plan, ref, None, head=head, out_dir=Path(out_dir),
                  device="cuda", cap=None)
    _inspect_priors()
    view, _, _, _, _ = zero._bind_view_metadata(plan)
    return {
        "status": "READY_NO_CLAIM_NO_PAYLOAD", "plan": ref,
        "view_sha256": view["contract_sha256"],
        "objective_sha256": view["research_objective"]["contract_sha256"],
        "source_head": head, "source_owner_refs_checked": len(plan["source_owners"]),
        "claim_created": False, "model_forwards": 0,
        "backward_calls": 0, "optimizer_steps": 0, "test_data_used": False,
    }


def _preflight(plan_path: Path, plan_sha256: str, device: str,
               out_dir: Path) -> tuple[dict, dict, dict, str, dict]:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_producer_execution, require_capped_lock_ancestry,
    )
    head = zero._clean_head()
    plan, plan_ref = _checked_plan_ref(plan_path, plan_sha256)
    policy_ref = zero._ref(POLICY, zero._sha(POLICY))
    policy = zero._read_ref(policy_ref)
    cap = require_guarded_cuda_producer_execution()
    cap["lock"] = require_capped_lock_ancestry()
    _qualify_plan(plan, plan_ref, policy, head=head, out_dir=out_dir,
                  device=device, cap=cap)
    _inspect_priors()
    zero._bind_view_metadata(plan)
    return plan, plan_ref, policy_ref, head, cap


def _gradient_summary(gradient, *, row: int | None = None) -> dict:
    import torch
    if gradient is None:
        return {"present": False, "finite": None, "nonzero": False,
                "nonzero_count": 0, "max_abs": None}
    part = gradient if row is None else gradient[row]
    finite = bool(torch.isfinite(part).all())
    magnitude = part.detach().abs()
    return {"present": True, "finite": finite,
            "nonzero": bool(torch.count_nonzero(part).item()),
            "nonzero_count": int(torch.count_nonzero(part).item()),
            "max_abs": float(magnitude.max().item()) if part.numel() else 0.0}


def _evaluate_and_backward(model, logits, features: dict, selected, issues,
                           bound, initial_state_sha256: str) -> dict:
    """One canonical loss backward on row 0; row 1 is evaluated and skipped."""
    import torch
    from gx1.contracts.market_state_five_endpoint_batch_v1 import evaluate_after_forward
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256

    _need(isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
          and tuple(logits.shape) == (2, 5, 3)
          and bool(torch.isfinite(logits).all()), "LOGITS")
    _need(all(isinstance(value, torch.Tensor) and value.shape[0] == 2
              for value in features.values()), "FEATURE_BATCH")
    active = evaluate_after_forward(logits[:1], bound, selected[:1], issues[:1],
                                    fold_id=1, partition="fit")
    skipped = evaluate_after_forward(logits[1:2], bound, selected[1:],
                                     issues[1:], fold_id=1, partition="fit")
    _need(active["count"] > 0 and active["skip_update"] is False
          and skipped["count"] == 0 and skipped["skip_update"] is True
          and float(skipped["loss"].detach().item()) == 0.0
          and active["contract_sha256"] == skipped["contract_sha256"],
          "MATURE_SKIP_SUPPORT")
    active["loss"].backward()
    input_gradients = {}
    for key in sorted(features):
        value = features[key]
        if not value.is_floating_point():
            input_gradients[key] = {"kind": "integer_categorical",
                                    "input_gradient_applicable": False}
            continue
        input_gradients[key] = {
            "informative_row": _gradient_summary(value.grad, row=0),
            "skipped_row": _gradient_summary(value.grad, row=1),
        }
    embedding_gradients = {}
    for name, parameter in model.named_parameters():
        if "embedding" in name.lower():
            embedding_gradients[name] = _gradient_summary(parameter.grad)
    head_gradient = _gradient_summary(model.endpoint_head.weight.grad)
    backbone_gradient = _gradient_summary(
        model.backbone.entry_q_joint_in.weight.grad)
    _need(all(value["informative_row"]["finite"] is not False
              for value in input_gradients.values()
              if "informative_row" in value)
          and all(value["finite"] is not False
                  for value in embedding_gradients.values()),
          "NONFINITE_INPUT_OR_EMBEDDING_GRADIENT")
    _need(head_gradient["present"] and head_gradient["finite"]
          and head_gradient["nonzero"], "HEAD_BACKWARD")
    _need(backbone_gradient["present"] and backbone_gradient["finite"]
          and backbone_gradient["nonzero"], "BACKBONE_BACKWARD")
    _need(all(value["skipped_row"]["finite"] is not False
              and not value["skipped_row"]["nonzero"]
              for value in input_gradients.values()
              if "skipped_row" in value), "SKIPPED_ROW_GRADIENT")
    after = canonical_model_state_sha256(model.state_dict())
    _need(initial_state_sha256 == after, "STATE_CHANGED")
    return {
        "active_loss": float(active["loss"].detach().item()),
        "active_count": active["count"],
        "active_per_tf_counts": active["per_tf_counts"],
        "skipped_loss": float(skipped["loss"].detach().item()),
        "skipped_count": skipped["count"],
        "skipped_per_tf_counts": skipped["per_tf_counts"],
        "skipped_update": skipped["skip_update"],
        "contract_sha256": active["contract_sha256"],
        "head_weight_gradient": head_gradient,
        "backbone_entry_q_joint_in_weight_gradient": backbone_gradient,
        "input_gradients": input_gradients,
        "categorical_embedding_gradients": embedding_gradients,
        "initial_model_state_sha256": initial_state_sha256,
        "after_model_state_sha256": after,
        "backward_calls": 1, "optimizer_steps": 0,
        "ema_updates": 0, "scheduler_steps": 0,
    }


def _one_gradient_witness(plan: dict, cap: dict) -> dict:
    """Gather feature-only fit rows, forward once, then open corrected targets."""
    import numpy as np
    import torch
    from gx1.audit import dynamic_mtf_october_input_only as prior
    from gx1.audit.dynamic_mtf_train_prefix_inputs import _check_payload_stats
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    input_plan = zero._read_ref(zero._fixed_ref("input_plan"))
    view, normal, constructor, observer_scope, profile = zero._bind_view_metadata(plan)
    provider_plan = prior._read_ref(input_plan["ancestral_264_plan"])
    prior._read_ref(input_plan["ancestral_267_plan"])
    provider = prior._load_module(input_plan["ancestral_264_worker"],
                                  "gx1_five_endpoint_gradient_264_reader")
    with campaign.file_verification_scope():
        info = provider.validate_metadata(provider_plan)
        _check_payload_stats(provider_plan["payloads"])
        axes, full = provider.read_axes(provider_plan, info, np)
        _need(full == input_plan["full_axes"], "FULL_AXES")
        prior._axis_receipt(axes, full, input_plan["fit_prefix"],
                            input_plan["evaluation_axis"])
        fit_rows = input_plan["fit_prefix"]["rows"]
        source_plan = dict(provider_plan,
                           october_gc_transition=input_plan["current_gc_transition"],
                           october_gc_layout=input_plan["current_gc_layout"])
        physical = prior._prepare_prefix_inputs(
            provider, source_plan, info, axes, fit_rows=fit_rows,
            last_issue_ns=int(axes["decision_ns"][fit_rows - 1]))
        snapshot = campaign.capture_verified_file_snapshot()
    selected = np.asarray(plan["sample_parent_rows"], dtype=np.int64)
    _need(np.array_equal(axes["parent_rows"][[0, fit_rows - 1]], selected)
          and np.array_equal(axes["decision_ns"][[0, fit_rows - 1]],
                             np.asarray(plan["sample_issue_ns"], np.int64)),
          "EXACT_FIT_ROWS")
    fit_axes = {key: axes[key][:fit_rows]
                for key in ("parent_rows", "bar_start_ns", "decision_ns")}
    observer = views.NativeDirectionObserverAxes(
        view, source=observer_scope["source"], role="fit",
        unobserved_units_profile=profile, **fit_axes)
    raw = views.gather_native_direction_feature_batch(
        view, parent_rows=observer.parent_rows,
        bar_start_ns=observer.bar_start_ns, decision_ns=observer.decision_ns,
        entry_sequence_source=zero._readonly_sequence_source_headers(
            physical["entry_sequence_source"]),
        snap_x=physical["current"]["snap"],
        ctx_cont=physical["current"]["ctx_cont"],
        ctx_cat=physical["current"]["ctx_cat"],
        multi_tf_sources=physical["mtf"], gc_surfaces=physical["gc"],
        gc_layout=physical["gc_layout"], batch_parent_rows=selected,
        observer_axes=observer, unobserved_units_profile=profile)
    _need(set(raw["features"]) == zero.FEATURE_KEYS
          and raw["witness"]["parent_rows"] == selected.tolist()
          and raw["witness"]["decision_ns"] == plan["sample_issue_ns"],
          "FEATURE_ONLY_BATCH")
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
    model.train()
    initial = canonical_model_state_sha256(model.state_dict())
    features = {}
    for key, value in raw["features"].items():
        tensor = torch.from_numpy(value).to(device)
        if tensor.is_floating_point():
            tensor.requires_grad_(True)
        features[key] = tensor
    _need(set(features) == zero.FEATURE_KEYS, "NO_TARGET_FORWARD_INPUT")
    torch.cuda.reset_peak_memory_stats()
    outputs = trainer._model_forward_fp32(model, **features)
    _need(set(outputs) == {"logits"}, "OUTPUT_KEYS")
    logits = outputs["logits"]
    # Target payloads are opened only after the sole feature-only model forward.
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
    bound = target.load_bound_five_endpoint_targets()
    issues = np.asarray(raw["witness"]["decision_ns"], dtype=np.int64)
    witness = _evaluate_and_backward(model, logits, features, selected, issues,
                                     bound, initial)
    snapshot.require_unchanged()
    _check_payload_stats(provider_plan["payloads"])
    return {
        "audit_scope_limit": "Two TRAIN fit rows prove a model-to-loss backward and an empty-support skip only. They do not prove influence of all 253 fields per timeframe, model learning, OOF generalization, or economic edge.",
        "fresh_unfitted_model": True, "checkpoint_loaded": False,
        "model_training_mode": bool(model.training),
        "view_sha256": view["contract_sha256"],
        "objective_sha256": view["research_objective"]["contract_sha256"],
        "selected_parent_rows": selected.tolist(),
        "issue_ns": issues.tolist(),
        "feature_witness": raw["witness"],
        "feature_shapes": {key: list(value.shape) for key, value in features.items()},
        "feature_dtypes": {key: str(value.dtype) for key, value in features.items()},
        "source_snapshot_files": snapshot.file_count,
        "cgroup_scope": {key: cap[key] for key in
                         ("cgroup_path", "memory_max", "memory_high", "swap", "pids")},
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "model_forwards": 1, "test_data_used": False, **witness,
    }



@contextmanager
def _bounded_worker(started: float):
    """Leave time for a FAILED terminal before the outer 3600 s guard stops us."""
    previous = signal.getsignal(signal.SIGALRM)
    remaining_timer = signal.getitimer(signal.ITIMER_REAL)
    _need(previous == signal.SIG_DFL and remaining_timer == (0.0, 0.0),
          "TIMER_OWNER")
    seconds = WALL_SECONDS - (time.monotonic() - started) - 15.0
    _need(seconds > 0, "PREFLIGHT_WALL")

    def expire(_signum, _frame):
        raise TimeoutError("FIVE_ENDPOINT_GRADIENT_BRIDGE_WORKER_WALL_LIMIT")

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(plan_path: Path, plan_sha256: str, device: str, out_dir: Path) -> dict:
    """Admitted dispatcher entry point; the one-use output root cannot be reused."""
    from gx1.contracts.immutable_event_authority_v1 import _fsync_directory

    began = time.monotonic()
    plan, plan_ref, policy_ref, head, cap = _preflight(
        Path(plan_path), plan_sha256, device, Path(out_dir))
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    claim = None
    try:
        _fsync_directory(root.parent)
        claim = zero._write(root, "CLAIM.json", {
            "schema_version": "gx1_market_state_five_endpoint_gradient_bridge_claim_v1",
            "plan": plan_ref, "policy": policy_ref, "source_head": head,
            "selected_parent_rows": SAMPLE_ROWS, "invocation": 1},
            limit=plan["max_json_bytes"])
        with _bounded_worker(began):
            result = _one_gradient_witness(plan, cap)
        _need(zero._sha(Path(plan_ref["path"])) == plan_ref["sha256"]
              and zero._sha(Path(policy_ref["path"])) == policy_ref["sha256"]
              and subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
              == head
              and subprocess.check_output(["git", "status", "--porcelain"], text=True) == "",
              "FINAL_SOURCE_IDENTITY")
        result_ref = zero._write(root, "RESULT.json", {
            "schema_version": "gx1_market_state_five_endpoint_gradient_bridge_result_v1",
            "decision": "PASS_TRAIN_MODEL_LOSS_BACKWARD_AND_SKIP_WITNESS",
            "plan": plan_ref, "claim": claim, "source_head": head,
            "status": "COMPLETE", "elapsed_seconds": time.monotonic() - began,
            **result}, limit=plan["max_json_bytes"])
        terminal = zero._write(root, "TERMINAL.json", {
            "schema_version": "gx1_market_state_five_endpoint_gradient_bridge_terminal_v1",
            "status": "COMPLETE", "exit_code": 0, "plan": plan_ref,
            "claim": claim, "result": result_ref, "model_forwards": 1,
            "backward_calls": 1, "optimizer_steps": 0, "test_data_used": False,
            "no_relaunch_under_this_plan": True,
            "output_inventory": ["CLAIM.json", "RESULT.json", "TERMINAL.json"]},
            limit=plan["max_json_bytes"])
        return {"result": result_ref, "terminal": terminal}
    except Exception as exc:
        failure = zero._write(root, "FAILED.json", {
            "schema_version": "gx1_market_state_five_endpoint_gradient_bridge_failure_v1",
            "plan": plan_ref, "claim": claim, "source_head": head,
            "error_type": type(exc).__name__, "error": str(exc)[:2048],
            "no_relaunch_under_this_plan": True}, limit=plan["max_json_bytes"])
        zero._write(root, "TERMINAL.json", {
            "schema_version": "gx1_market_state_five_endpoint_gradient_bridge_terminal_v1",
            "status": "FAILED", "exit_code": 1, "plan": plan_ref,
            "claim": claim, "failure": failure,
            "model_forwards_upper_bound": 1, "backward_calls_upper_bound": 1,
            "optimizer_steps": 0, "test_data_used": False,
            "no_relaunch_under_this_plan": True,
            "output_inventory": (["CLAIM.json", "FAILED.json", "TERMINAL.json"]
                                 if claim else ["FAILED.json", "TERMINAL.json"])},
            limit=plan["max_json_bytes"])
        raise
