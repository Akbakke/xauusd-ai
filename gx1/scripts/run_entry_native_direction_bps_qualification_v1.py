"""B16 bps qualification inside the existing signed, capped private campaign.

Fresh baseline/candidate mechanics only. No old checkpoint, predictor fit,
later-cohort scoring, new horizon, epoch sweep, or action/Exit authority.
"""
from __future__ import annotations

import gc
from pathlib import Path
import signal
import sys
import time


def require(ok, reason):
    if not ok:
        raise RuntimeError("NATIVE_BPS_QUALIFICATION_" + reason)


def _forward(model, features, *, counters, torch, grad):
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as hybrid
    counters["forwards"] += 1
    require(counters["forwards"] <= 6, "FORWARD_CAP")
    with trainer._training_mha_fastpath_context(model.backbone), (torch.enable_grad() if grad else torch.no_grad()):
        require(not torch.backends.mha.get_fastpath_enabled()
            and not hybrid._BATCH_EQUAL_LENGTH_MTF_EVAL.get(), "CANONICAL_NUMERICS")
        out = trainer._model_forward_fp32(model, **features)
    counters["forward_returns"] += 1
    return out


def _step(model, optimizer, features, target, *, cutoff, counters, torch, probe=False):
    from gx1.contracts.entry_native_direction_bps_v1 import native_direction_bps_loss, native_direction_magnitude_weighted_loss
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    weighted = bool(getattr(model, "magnitude_weighted", False))
    require((weighted != bool(model.signed_bps)) and model.training and all(m.training for m in model.modules()), "TRAIN_MODE")
    out = _forward(model, features, counters=counters, torch=torch, grad=True)
    if weighted:
        result = native_direction_magnitude_weighted_loss(out["logit_up"], rms_bps=model.magnitude_rms_bps,
            raw_weight_mean=model.magnitude_raw_weight_mean, **target, cutoff_ns=cutoff)
    else:
        result = native_direction_bps_loss(out["logit_up"], out["mean_return_bps"],
            rms_bps=model.bps_rms, **target, cutoff_ns=cutoff)
    require(result["update_allowed"] and all(row["eligible_cells"] > 0 for row in result["per_horizon_counts"])
        and (weighted or all(n > 0 for n in result["bps_per_horizon_counts"])), "ALL_THREE_HORIZONS")
    probe_evidence = None
    if probe:
        counters["backwards"] += 1
        require(counters["backwards"] <= 5, "REVERSE_CAP")
        probe_loss = (result["loss"] - result["unweighted_direction_loss"]) if weighted else result["bps_loss"]
        (gradient,) = torch.autograd.grad(probe_loss, (model.backbone.entry_q_joint_in.weight,),
            retain_graph=True, allow_unused=False)
        counters["backward_returns"] += 1
        require(bool(torch.isfinite(gradient).all()) and int(torch.count_nonzero(gradient)) > 0,
            "BPS_ONLY_SHARED_PROJECTION_GRADIENT")
        require(all(p.grad is None for p in model.parameters()), "PROBE_NO_ACCUMULATION")
        probe_evidence = {"joint_projection_nonzero": int(torch.count_nonzero(gradient)),
            "joint_projection_l2": float(torch.linalg.vector_norm(gradient)),
            "scope": ("Weighted minus ordinary direction gradient; no predictive-value claim." if weighted else
                "One bps-only reverse after first mean-head update; no causal/predictive-value claim.")}
        del gradient
    counters["backwards"] += 1
    require(counters["backwards"] <= 5, "REVERSE_CAP")
    result["loss"].backward()
    counters["backward_returns"] += 1
    gradients = common._gradient_evidence(model, torch)
    counters["optimizer_step_calls"] += 1
    require(counters["optimizer_step_calls"] <= 4, "UPDATE_CAP")
    norm = trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)
    counters["optimizer_updates"] += 1
    if weighted:
        return {"weighted_loss": float(result["loss"].detach()),
            "unweighted_direction_loss": float(result["unweighted_direction_loss"].detach()),
            "direction_counts": result["counts"], "direction_horizon_counts": result["per_horizon_counts"],
            "gradient_groups": gradients, "gradient_norm": float(norm), "weight_difference_probe": probe_evidence}
    return {"joint_loss": float(result["loss"].detach()),
        "direction_loss": float(result["direction_loss"].detach()), "bps_loss": float(result["bps_loss"].detach()),
        "direction_counts": result["counts"], "direction_horizon_counts": result["per_horizon_counts"],
        "bps_eligible_cells": result["bps_eligible_cells"], "bps_horizon_counts": result["bps_per_horizon_counts"],
        "gradient_groups": gradients, "gradient_norm": float(norm), "bps_only_probe": probe_evidence}


def run_bps_qualification(*, policy_path: Path, policy_file_sha256: str, progress_path: Path):
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits = require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    window, plan, invocation, marker, launch = authority.require_native_direction_research_context(
        policy_path, policy_file_sha256, progress_path)
    weighted = launch["run_scope"] == authority.WEIGHT_SCOPE
    require(launch["run_scope"] in (authority.BPS_SCOPE, authority.WEIGHT_SCOPE)
        and invocation["batch_size"] == 16, "EXACT_PROFILE")
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    from gx1.contracts.entry_model_native_train_recipe_v1 import (
        native_direction_research_recipe, require_native_direction_research_recipe)
    from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha

    recipe = require_native_direction_research_recipe(launch["numerical_recipe"])
    require(recipe["batch_size"] == 16 and recipe["max_updates"] == 2
        and recipe.get("signed_bps_supervision", False) is (not weighted)
        and recipe.get("magnitude_weighted_direction", False) is weighted, "NUMERICAL_RECIPE")
    baseline_recipe = native_direction_research_recipe(batch_size=16, max_updates=2)
    binding_ref = common._hash_file(launch["execution_binding_path"])
    extra = {}; binding = common._read(binding_ref, extra)
    design = common._read(launch["qualification_design"], extra)
    io_budget = common._read(common.INPUT_DESIGN, extra)["budget"]["max_rchar_bytes"]
    require(io_budget == 137438953472, "REUSED_IO_CEILING")
    proof = common._read(launch["gc_layout_metadata_parity"], extra)
    require(design["model_constructions"] == 4 and design["forwards"] == 6
        and design["reverse_passes"] == 5 and design["physical_optimizer_steps"] == 4, "DESIGN")
    require(proof["status"] == "COMPLETE_CONSUMED" and proof["metadata_exact_parity"] is True
        and proof["test_cases_passed"] == 6 and all(v == 0 for v in proof["counters"].values()), "GC_METADATA_PROOF")
    context = prefix._metadata(launch, common, gc_layout_metadata_parity=launch["gc_layout_metadata_parity"],
        **({"weight_qualification_design": launch["qualification_design"]} if weighted else {}))
    context["info"]["metadata"].update(extra)
    for ref in (window["recipe"], binding["plan"], {"path": str(policy_path), "sha256": policy_file_sha256}):
        common._read(ref, context["info"]["metadata"])
    common._freeze(launch, binding, context)
    output = Path(window["output_directory"]); output.mkdir(parents=True)
    original_prediction = authority.WEIGHT_INITIAL_PREDICTIONS if weighted else None
    denied = common._install_read_guard(context, output, Path(launch["runtime_root"]),
        **({"original_initial_prediction": original_prediction["path"]} if weighted else {}))
    scale_counter = "magnitude_scale_fits" if weighted else "bps_scale_fits"
    counters = {key: 0 for key in ("model_constructions", "model_construction_returns", "optimizer_constructions",
        "forwards", "forward_returns", "backwards", "backward_returns", "optimizer_step_calls", "optimizer_updates",
        "staged_state_load_calls", "staged_state_load_returns", "resume_state_load_calls", "resume_state_load_returns",
        "forbidden_legacy_calls", "json_bytes", scale_counter)}
    started = time.monotonic(); before = common._io(); previous = sys.getprofile()
    device = torch.device("cuda", 0); full_state = initial = initial_identity = None
    error = None; parity = matched = unchanged = False; cursor = 0; steps = []
    order = None; order_sha = "0"*64; scale_ref = input_ref = None
    model = optimizer = None
    common._json(output/"CLAIM.json", {"recipe": window["recipe"], "plan_sha256": plan["plan_sha256"],
        "limits": limits, "additional_worker_ceiling_seconds": 900}, counters, launch["max_published_json_bytes"])

    def expired(*_args):
        raise RuntimeError("NATIVE_BPS_QUALIFICATION_900_SECOND_CEILING")

    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    try:
        sys.setprofile(common._deny_legacy_work(previous, counters))
        counters[scale_counter] += 1
        prepared, targets, view, _context_map, ordered = prefix._prepare(context, launch, np,
            **({"magnitude_weighted": True} if weighted else {"signed_bps": True}))
        selected = ordered[:32].copy()
        order = torch.from_numpy(selected); order_sha = tree_sha(order)
        scale_ref = common._json(output/("OWN_PREFIX_MAG_WEIGHT.json" if weighted else "OWN_PREFIX_BPS_SCALE.json"),
            view["magnitude_weight_contract" if weighted else "bps_scale"], counters, launch["max_published_json_bytes"])
        baseline_view = view_owner.bind_native_direction_feature_view(prefix_normalization=context["fitted"],
            prefix_normalization_ref=context["manifest"]["prefix_normalization"]["output"],
            constructor_metadata=context["constructor"], constructor_metadata_ref=context["manifest"]["constructor_metadata"],
            source_bindings=context["manifest"]["view_source_bindings"], max_batch_rows=16)
        batches = []; support = []
        for left in (0, 16):
            raw, target = prefix._batch(prepared, targets, view, selected[left:left+16], np)
            support.append({"parents": selected[left:left+16].tolist(), "witness_sha256": tree_sha(raw["witness"]),
                "gc": common._gc_support(raw, prepared["gc"]["layout"], np)})
            batches.append(prefix._device_batch(raw, target, torch, device))
        input_ref = common._json(output/"INPUT_QUALIFICATION.json", {"view": view,
            "baseline_view": baseline_view, ("magnitude_weight" if weighted else "bps_scale"): scale_ref, "parent_order_sha256": order_sha,
            "selection": "First32original MT19937seed20260911 own-prefix presentations; no label selection.",
            "batches": support}, counters, launch["max_published_json_bytes"])
        batch_sha = tree_sha(batches)
        del prepared, targets, ordered, raw, target

        def capture(m, opt, offset, updates, recipe_used=recipe):
            return state_owner.capture_native_direction_research_state(m, opt, recipe=recipe_used,
                feature_view_identity=m.feature_view_identity, parent_order=order, cursor=offset,
                optimizer_updates=updates, device=device)

        def evaluate_initial(m, opt, recipe_used):
            before_state = capture(m, opt, 0, 0, recipe_used)
            modes = [(mod, mod.training) for mod in m.modules()]
            try:
                m.eval()
                out = _forward(m, batches[0][0], counters=counters, torch=torch, grad=False)
                result = {k: v.detach().cpu().clone() for k, v in out.items() if k != "horizons_minutes"}
            finally:
                for mod, training in modes: mod.training = training
            require(capture(m, opt, 0, 0, recipe_used)["contract_sha256"] == before_state["contract_sha256"],
                "INITIAL_FORWARD_STATE_PRESERVATION")
            return result

        model, optimizer, _ = common._fresh(context, baseline_view, baseline_recipe, device, counters, max_constructions=4)
        fresh = capture(model, optimizer, 0, 0, baseline_recipe)
        shared_names = tuple(fresh["model_state"])
        shared_sha = tree_sha(fresh["model_state"])
        shared_buffer_names = tuple(fresh["buffer_state"])
        shared_buffer_sha = tree_sha(fresh["buffer_state"])
        baseline_rng_sha = tree_sha(fresh["rng_state"])
        baseline_outputs = evaluate_initial(model, optimizer, baseline_recipe)
        if weighted:
            original_stat = common._stat(original_prediction["path"])
            require(common._hash_file(original_prediction["path"]) == original_prediction, "ORIGINAL_INITIAL_SHA")
            with np.load(original_prediction["path"], allow_pickle=False) as archive:
                require(np.array_equal(archive["parent_rows"][:16], selected[:16])
                    and np.array_equal(archive["decision_ns"][:16], batches[0][1]["decision_ns"].cpu().numpy())
                    and np.array_equal(archive["logit_up"][:16], baseline_outputs["logit_up"].numpy()),
                    "ORIGINAL_INITIAL_OUTPUT_AND_CLOCK")
            require(common._stat(original_prediction["path"]) == original_stat, "ORIGINAL_INITIAL_UNCHANGED")
        del fresh, model, optimizer
        model = optimizer = None; gc.collect(); torch.cuda.empty_cache()

        for trajectory in ("reference", "interrupted", "resumed"):
            cursor = 0
            model, optimizer, _ = common._fresh(context, view, recipe, device, counters, max_constructions=4)
            fresh = capture(model, optimizer, 0, 0)
            require(tree_sha({k: fresh["model_state"][k] for k in shared_names}) == shared_sha
                and tree_sha({k: fresh["buffer_state"][k] for k in shared_buffer_names}) == shared_buffer_sha
                and tree_sha(fresh["rng_state"]) == baseline_rng_sha, "MATCHED_SHARED_PARAMETERS_BUFFERS_AND_RNG")
            identity = tree_sha({k: fresh[k] for k in ("binding", "model_state", "buffer_state",
                "optimizer_state", "rng_state", "module_training")})
            if initial_identity is None:
                initial_identity = identity
                candidate_outputs = evaluate_initial(model, optimizer, recipe)
                require(torch.equal(candidate_outputs["logit_up"], baseline_outputs["logit_up"])
                    and (weighted or torch.equal(candidate_outputs["mean_return_bps"], torch.zeros(16, 3))), "INITIAL_OUTPUT_PARITY")
                matched = True
                initial = common._save_state(output/"INITIAL_STATE.pt", fresh, torch=torch,
                    maximum=launch["max_checkpoint_bytes"], counters=counters)
                initial_inactive = tree_sha({name: p for name, p in model.named_parameters() if not p.requires_grad})
            else:
                require(identity == initial_identity, "FRESH_CANDIDATE_IDENTITY")
            del fresh
            if trajectory == "resumed":
                loaded = common._load_state(interrupted_ref, torch=torch,
                    maximum=launch["max_checkpoint_bytes"], counters=counters)
                restored = state_owner.restore_native_direction_research_state(loaded, model, optimizer, recipe=recipe,
                    feature_view_identity=model.feature_view_identity, parent_order=order, device=device)
                require(restored["cursor"] == 16 and restored["optimizer_updates"] == 1
                    and capture(model, optimizer, 16, 1)["contract_sha256"] == loaded["contract_sha256"], "SERIALIZED_RESTORE")
                del loaded; cursor = 16
            logical = (0, 1) if trajectory == "reference" else (0,) if trajectory == "interrupted" else (1,)
            for index in logical:
                require(time.monotonic()-started < 900, "DEADLINE")
                evidence = _step(model, optimizer, *batches[index], cutoff=launch["fold"]["train_cutoff_ns"],
                    counters=counters, torch=torch, probe=trajectory == "reference" and index == 1)
                cursor = 16*(index+1)
                steps.append({"trajectory": trajectory, "logical_step": index+1, **evidence})
                require(tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})
                    == initial_inactive, "INACTIVE_WEIGHTS_UNCHANGED")
            saved = capture(model, optimizer, cursor, 1 if trajectory == "interrupted" else 2)
            if trajectory == "reference":
                reference_sha = saved["contract_sha256"]
            elif trajectory == "interrupted":
                interrupted_ref = common._save_state(output/"INTERRUPTED_STEP1.pt", saved, torch=torch,
                    maximum=launch["max_checkpoint_bytes"], counters=counters)
            else:
                require(saved["contract_sha256"] == reference_sha, "BIT_EXACT_COMPLETE_RESUME")
                full_state = common._save_state(output/"RESUMED_STEP2.pt", saved, torch=torch,
                    maximum=launch["max_checkpoint_bytes"], counters=counters); parity = True
            del saved, model, optimizer
            model = optimizer = None; gc.collect(); torch.cuda.empty_cache()
        expected = {"model_constructions":4,"model_construction_returns":4,"optimizer_constructions":4,
            "forwards":6,"forward_returns":6,"backwards":5,"backward_returns":5,"optimizer_step_calls":4,
            "optimizer_updates":4,"staged_state_load_calls":3,"staged_state_load_returns":3,
            "resume_state_load_calls":1,"resume_state_load_returns":1,"forbidden_legacy_calls":0,scale_counter:1}
        require(all(counters[k] == v for k, v in expected.items()) and tree_sha(batches) == batch_sha, "EXACT_WORK")
        if weighted:
            require(common._stat(original_prediction["path"]) == original_stat, "ORIGINAL_INITIAL_BOOKEND")
        common._freeze(launch, binding, context); unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ":" + str(exc)
        try: common._freeze(launch, binding, context); unchanged = True
        except Exception: unchanged = False
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler); sys.setprofile(previous)
    after = common._io()
    if after["rchar"] > io_budget:
        error = "RCHAR_BUDGET_EXCEEDED"
    passed = matched and parity and unchanged and error is None
    outcome = "COMPLETE" if passed else "FAILED"
    pass_decision = "PASS_MAGNITUDE_WEIGHT_NATIVE_MECHANICS" if weighted else "PASS_BPS_NATIVE_MECHANICS"
    result = common._json(output/"RESULT.json", {"schema_version":("gx1_native_magnitude_weight_private_qualification_result_v1" if weighted else "gx1_native_bps_private_qualification_result_v1"),
        "decision":pass_decision if passed else "FAILED", "recipe":window["recipe"],
        "execution_binding":binding_ref,"counts":{k:v for k,v in counters.items() if k!="json_bytes"},
        "initial_state":initial,"initial_cuda_identity_sha256":initial_identity,"final_state":full_state,
        ("own_prefix_magnitude_weight" if weighted else "own_prefix_bps_scale"):scale_ref,
        **({"original_initial_prediction":original_prediction,"original_initial_parity":matched} if weighted else {}),
        "input_qualification":input_ref,"matched_initialization":matched,
        "bit_exact_resume":parity,"steps":steps,"parent_order_sha256":order_sha,"parent_cursor":cursor,
        "planned_unique_parents":32,"verified_unique_parents":32 if passed else None,
        "planned_logical_optimizer_steps":2,"physical_optimizer_updates":counters["optimizer_updates"],
        "source_unchanged":unchanged,"metadata_bookends":context["info"]["metadata"],
        "payload_stat_bookends":context["ancestor_plan"]["payloads"],"forbidden_open_paths":denied,
        "limits":limits,"elapsed_seconds":time.monotonic()-started,"io_before":before,"io_after":after,
        "rchar_delta":after["rchar"]-before["rchar"],"rchar_scope":"proc counter excludes mmap faults",
        "private_weights_for_main_fit":False,"predictive_value_qualified":False,"test_data_used":False,
        "learning_boundary":"Two fixed logical B16 steps repeated for resume; no fit or later gain claim.","error":error},
        counters,launch["max_published_json_bytes"])
    checkpoint = common._json(Path(window["campaign_cursor_path"]), authority.build_research_cursor(
        recipe_ref=window["recipe"],outcome=outcome,optimizer_updates=counters["optimizer_updates"],
        parent_cursor=cursor,parent_order_sha256=order_sha,full_state=full_state,result=result,batch_size=16),
        counters,launch["max_published_json_bytes"])
    progress = authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=checkpoint,optimizer_updates=counters["optimizer_updates"],parent_cursor=cursor,
        parent_order_sha256=order_sha,outcome=outcome)
    common._json(progress_path,progress,counters,launch["max_published_json_bytes"])
    authority.require_research_progress(progress,plan_sha256=plan["plan_sha256"],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    common._freeze(launch,binding,context)
    if not passed:
        raise RuntimeError("NATIVE_BPS_QUALIFICATION_FAILED:" + str(error))
    return {"decision":pass_decision,"result":result,
        "progress_path":str(progress_path),"counts":counters,"checkpoint_pointer":checkpoint}
