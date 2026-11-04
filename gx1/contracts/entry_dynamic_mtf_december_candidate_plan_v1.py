"""Registered December research design. This metadata is outside every native launch schema."""
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as campaign_fit
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.entry_dynamic_mtf_state_report_v1 import december_oof_scoring_contract, dynamic_state_prediction_file_bytes
from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import _fixed_order

SCHEMA = "gx1_closed_december_dynamic_mtf_candidate_plan_v1"
REPO = campaign_fit.CANONICAL_REPO
FIT_ROWS, HELD_ROWS = 41866, 5736
FIT_CUTOFF, HELD_CUTOFF = 1733011200000000000, 1735689600000000000
REQUIRED_OWNERS = (
    "gx1/contracts/entry_dynamic_mtf_december_candidate_plan_v1.py",
    "gx1/contracts/entry_dynamic_mtf_state_report_v1.py",
    "gx1/contracts/entry_model_native_train_recipe_v1.py",
    "gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py",
    "gx1/contracts/entry_native_direction_research_campaign_v1.py",
    "gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py",
    "gx1/scripts/run_entry_native_direction_dense_qualification_v1.py", "scripts/gx1_capped_run.sh")
CLOSED = {k: False for k in ("executable", "native_launch_authorized", "training_enabled", "reboot_authorized",
                              "model_calls_authorized", "optimizer_updates_authorized", "held_out_access_authorized",
                              "test_authorized", "broker_authorized", "trading_authorized", "relaunch_allowed")}


def _require(condition, reason):
    if not condition:
        raise ValueError("DECEMBER_CANDIDATE_PLAN_" + reason)


def _read(ref):
    return campaign.read_bound_json(Path(ref["path"]), ref["sha256"])


def _axis_digest(array):
    return hashlib.sha256(np.asarray(array, dtype="<i8").tobytes()).hexdigest()


def _cohort_rows(parents, seed):
    order = _fixed_order(parents, len(parents), seed, np)
    return order, np.sort(order[:32]), order[:256]


def _cohort_counts(query, positions):
    """Sparse TRAIN labels. Count usable endpoints and observable next-minute changes separately."""
    rows = len(positions)
    result = {}
    for j, tf in enumerate(targets.TIMEFRAMES):
        native = j + 1  # query grid is [1, 5, 15, 60, 240, 1440]
        count = {}
        for name, idx in (("first", 0), ("native", native)):
            state = query["state"][positions, j, idx]
            known = query["observed"][positions, j, idx]
            classes = {cls: int(np.count_nonzero(known & (state == k))) for k, cls in enumerate(targets.STATE_NAMES)}
            _require(sum(classes.values()) == int(known.sum()), "COHORT_CLASSES")
            count[name] = {"observed": int(known.sum()), "missing_or_immature": rows - int(known.sum()), "classes": classes}
        paired = (query["current_observed"][positions, j] & query["current_path_observed"][positions, j]
                  & query["observed"][positions, j, 0] & query["path_observed"][positions, j, 0])
        changes = query["state"][positions, j, 0] != query["current_state"][positions, j]
        count["first_minute_known_state_and_path_intervals"] = int(paired.sum())
        count["first_minute_changes"] = int(np.count_nonzero(paired & changes))
        count["first_activity_observed"] = int(query["first_flat_activity_observed"][positions, j].sum())
        result[tf] = count
    return result


def _cohorts(target_ref, geometry, seed):
    manifest = _read(target_ref)
    meta = manifest["metadata"]
    _require(meta["parents"] == FIT_ROWS and meta["cutoff_ns"] == FIT_CUTOFF
             and meta["prefix_lineage"]["parent_slice"] == [0, FIT_ROWS], "PHYSICAL_FIT_ONLY")
    manifest = targets.read_dynamic_mtf_state_series_metadata(target_ref, expected_source_refs=meta["source_refs"],
                                                              expected_cutoff_ns=FIT_CUTOFF)
    # The published earlier fit series was canonically validated. Verify file identities
    # and headers, map read-only, and index only the selected parents and queried TRAIN labels.
    arrays = {}
    for key, desc in manifest["arrays"].items():
        ref = {k: desc[k] for k in ("path", "sha256")}
        campaign.require_binding(ref, label="December physical TRAIN target " + key)
        array = np.load(desc["path"], mmap_mode="r", allow_pickle=False)
        _require(array.dtype.name == desc["dtype"] and list(array.shape) == desc["shape"]
                 and not array.flags.writeable and Path(desc["path"]).stat().st_size == desc["size_bytes"],
                 "FIT_ARRAY_HEADER:" + key)
        arrays[key] = array
    parents, issues = arrays["parent_rows"], arrays["issue_ns"]
    _require(np.array_equal(parents, np.arange(FIT_ROWS)) and np.all(np.diff(issues) > 0)
             and issues[-1] < FIT_CUTOFF and _axis_digest(parents) == geometry["parent_rows_sha256_le_i64"]
             and _axis_digest(issues) == geometry["issue_ns_sha256_le_i64"], "FIT_COHORT_AXES")
    order, private, initial = _cohort_rows(parents, seed)
    selected = np.unique(np.r_[private, initial])
    leads = np.array([1] + [targets.LOCAL_WINDOW_M1_BARS[tf] for tf in targets.TIMEFRAMES], dtype=np.int64)
    query = targets.query_dynamic_mtf_states(targets.DynamicMtfStateSeries(arrays, meta),
        parent_rows=selected, issue_ns=issues[selected].copy(), cutoff_ns=FIT_CUTOFF, lead_minutes=leads,
        include_first_flat_activity=True)
    _require(not np.any(query["observed"] & (query["maturity_ns"] >= FIT_CUTOFF)), "FIT_ONLY_MATURITY")
    batches = [_cohort_counts(query, np.searchsorted(selected, b)) for b in np.split(private, 2)]
    zero = [{"batch": i + 1, "timeframe": tf, "measure": role}
            for i, batch in enumerate(batches) for tf, counts in batch.items()
            for role, n in (("native_endpoint", counts["native"]["observed"]),
                            ("first_state", counts["first"]["observed"]),
                            ("first_activity", counts["first_activity_observed"])) if n == 0]
    return {"parent_order_algorithm": campaign_fit.PARENT_ORDER_ALGORITHM, "seed": seed,
            "one_pass_order_sha256_le_i64": _axis_digest(order), "private_parents_sorted": private.tolist(),
            "private_rule": "first 32 of the fit-only seeded permutation, sort once, two consecutive B16 batches",
            "initial_parents_in_order": initial.tolist(), "initial_rule": "first 256 of the same fit-only permutation",
            "source": target_ref, "private_batch_sparse_support": batches, "private_sparse_support_blockers": zero,
            "initial_sparse_support": _cohort_counts(query, np.searchsorted(selected, initial)),
            "scope": "Target masks and classes at +1/native only. Confirmed +1 changes require known state and path "
                     "at issue and +1. Dense trajectory events, specialist activations, gradients, controls and "
                     "resume are unmeasured. Missing sparse support stops qualification without replacing parents."}


def _numerics(historical, objective_sha):
    old = historical["numerical_recipe"]
    batch = campaign_fit.BATCH_SIZE
    _require(old["max_updates"] == (historical["fold"]["prefix_rows"] + batch - 1) // batch == 2976
             and old["current_state_first_loss_weight"] == old["first_flat_activity_weight"] == 0.03
             and old["seed"] == 20260911 and old["batch_size"] == batch == 16
             and "native_endpoint_loss" not in old, "ORIGINAL_RECIPE")
    def build(maximum):
        recipe = native_direction_research_recipe(batch_size=batch, max_updates=maximum, dynamic_state=True,
            current_state_first_loss_weight=old["current_state_first_loss_weight"],
            first_flat_activity_weight=old["first_flat_activity_weight"], native_endpoint_loss=True)
        changed = {"max_updates", "contract_sha256", "research_objective_sha256", "native_endpoint_loss"}
        _require({k: v for k,v in old.items() if k not in changed} == {k:v for k,v in recipe.items() if k not in changed}
                 and recipe["research_objective_sha256"] == objective_sha, "NUMERICAL_IDENTITY")
        return recipe
    return build(2), build((FIT_ROWS + batch - 1)//batch)


def _candle_signs(candle, refs):
    _require(candle["schema_version"] == "gx1_december_prefix_candle_reference_v1"
             and candle["status"] in ("FROZEN_PREFIX_CANDLE_REFERENCE", "UNDETERMINED_PREFIX_CANDLE_REFERENCE")
             and candle["fit_parents"] == FIT_ROWS and candle["fit_cutoff_ns"] == FIT_CUTOFF
             and candle["feature"] == "close_open_atr" and candle["positive_class"] == "UP"
             and candle["model_calls"] == 0 and candle["held_out_outcomes_read"] is False
             and candle["test_data_used"] is False and all(candle[k] == refs[k] for k in
             ("fit_target_view", "feature_view", "prefix_normalization", "candidate_geometry")), "CANDLE_BINDING")
    panels = candle["panels"]
    _require(isinstance(panels, dict) and set(panels) == set(targets.TIMEFRAMES), "CANDLE_INVENTORY")
    signs = {}
    for tf in targets.TIMEFRAMES:
        p = panels[tf]
        pairs, twice = p["pairs"], p["twice_wins"]
        _require(p["lead_minutes"] == targets.LOCAL_WINDOW_M1_BARS[tf] and
                 all(type(p[k]) is int and p[k] >= 0 for k in ("up", "down", "flat", "observed_native",
                                                               "eligible_rows", "pairs", "twice_wins"))
                 and p["eligible_rows"] == p["up"] + p["down"] and pairs == p["up"] * p["down"]
                 and p["observed_native"] == p["up"] + p["down"] + p["flat"] and twice <= 2*pairs, "CANDLE_PANEL:" + tf)
        expected = (1 if twice >= pairs else -1) if pairs else None
        _require(p["orientation"] == expected and (expected is None or type(p["orientation"]) is int), "CANDLE_SIGN:" + tf)
        signs[tf] = expected
    _require((all(v is not None for v in signs.values())) == (candle["status"] == "FROZEN_PREFIX_CANDLE_REFERENCE"), "CANDLE_STATUS")
    return signs


def _source_freeze(stage, historical):
    branch = subprocess.check_output(["git", "-C", str(REPO), "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    _require(branch == "work/gx1-current" and not subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all"], text=True), "SOURCE_FREEZE")
    _require(set(REQUIRED_OWNERS) <= set(stage["source_owners"]), "SOURCE_INVENTORY")
    owners = {}
    for rel in sorted(set(historical["source_owners"]) | set(stage["source_owners"])):
        path = REPO / rel
        _require(not Path(rel).is_absolute() and ".." not in Path(rel).parts and path.resolve(strict=True) == path
                 and all(part[rel]["path"] == str(path) for part in (historical["source_owners"], stage["source_owners"]) if rel in part),
                 "SOURCE_PATH")
        owners[rel] = {"path": str(path), "sha256": campaign.file_sha256(path)}
        if rel in stage["source_owners"]:
            _require(owners[rel] == stage["source_owners"][rel], "SOURCE_OWNERS_CHANGED:" + rel)
    return {"canonical_repo": str(REPO), "branch": branch, "commit": head, "owners": owners}


def _guards(policy, historical, original_private):
    historic_plan = _read(original_private["plan"])
    first = _read(historic_plan["invocations"][0])
    launcher = first["launcher_argv"][:first["launcher_argv"].index("--")]
    _require(launcher == [str(REPO / "scripts/gx1_capped_run.sh"), "--class", "trainer", "--mem", "20G", "--swap", "512M"],
             "CAPPED_ROUTE")
    hardware = campaign.campaign_hardware_policy(campaign_fit.PHASE)
    hardware["physical_power_limit_w"] = campaign_fit.guarded_prefix_power_limit(historical)
    _require(historic_plan["policy"] == hardware and hardware["fresh_physical_windows_boot_before_every_invocation"] is True
             and first["requires_fresh_windows_boot"] is True
             and policy["native_invocation_seconds"] == 12000 and first["maximum_wall_seconds"] == policy["outer_guard_seconds"] == 13800
             and policy["limits"]["fresh_physical_windows_boot_before_every_invocation"] is True, "EXISTING_HARDWARE_GUARDS")
    return {"source_private_plan_for_limits_only": original_private["plan"], "capacity_argv_prefix": launcher,
            "dispatcher_module": campaign_fit.MODULE, "hardware": hardware,
            "cpu_affinity": policy["limits"]["cpu_affinity"], "numerical_threads": policy["limits"]["numerical_threads"],
            "native_window_seconds": policy["native_invocation_seconds"], "outer_guard_seconds": first["maximum_wall_seconds"],
            "whole_host_idle_before_physical_reboot": True, "next_boot_observation_required": True,
            "unchanged_source_and_immutable_input_bookends": True, "signed_guard_and_one_current_project_lock": True}


def _prior_private_parents(source_recipe):
    rows = source_recipe["qualification_parent_rows"]
    _require(len(rows) == len(set(rows)) == 32 and all(type(p) is int and p >= 0 for p in rows), "ORIGINAL_COHORT")
    reserved = [p for p in rows if FIT_ROWS <= p < FIT_ROWS + HELD_ROWS]
    _require(reserved == [42787, 44748, 45861, 47429], "ORIGINAL_RESERVED_ROWS")
    return reserved


def build_december_candidate_plan(policy, stage, *, candle_reference_ref, candle_reference):
    """Seal one historical candidate design. No model, checkpoint, GPU, OOF labels or launch route are opened."""
    named = "dynamic_mtf_december_candle_and_plan_20261104_001"
    _require(policy.get(named) == stage and all(policy[k] is False for k in ("training_enabled",
             "native_training_enabled", "full_epoch_training_allowed", "full_val_allowed")), "POLICY")
    base = policy["dynamic_mtf_masked_december_input_normalization_20261104_001"]
    data = policy["dynamic_mtf_train_prefix_dec01_20261104_001"]
    original = policy["dynamic_mtf_state_onepass_fit_20261103_001"]
    private_source = policy["dynamic_mtf_state_private_qualification_20261103_002"]
    endpoint = policy["dynamic_mtf_native_endpoint_loss_step1_20261104_001"]
    refs = {"fit_target_view": data["fit_target_view"], "prefix_counts": data["fit_counts"], "prefix_prior": data["fit_prior"],
            "prefix_normalization": base["normalization"], "verified_input_profile": base["verified_review"],
            "normalization_result": base["result"], "feature_view": base["feature_view"],
            "candidate_geometry": base["candidate_geometry"], "endpoint_result": endpoint["result"],
            "historical_recipe": original["recipe"]}
    _require(all(stage.get(k) == v for k,v in refs.items()) and original["training_enabled"] is False
             and private_source["execution_authorized"] is False, "STAGE_BINDINGS")
    # Verification is scoped to this operation; prior native artifacts are design evidence only.
    for ref in refs.values():
        campaign.require_binding(ref, label="registered December evidence")
    _require(Path(candle_reference_ref["path"]).parent == Path(stage["run_root"])
             and _read(candle_reference_ref) == candle_reference, "CANDLE_FILE")
    checked = _read(base["verified_review"])
    view = _read(base["feature_view"])
    old = _read(original["recipe"])
    objective = view["research_objective"]["contract_sha256"]
    _require(checked["status"] == "PASS_MASKED_FIT_PREFIX_AND_FORECAST_INPUT_BINDING_NO_MODEL"
             and checked["feature_view"] == refs["feature_view"] and checked["normalization"] == refs["prefix_normalization"]
             and checked["result"] == refs["normalization_result"] and checked["profile"] == base["profile"]
             and checked["feature_view_contract_sha256"] == view["contract_sha256"]
             and checked["normalized_contract_sha256"] == view["prefix_normalization"]["contract_sha256"]
             and view["prefix_normalization"]["binding"] == refs["prefix_normalization"]
             and view["zero_support_profile"]["contract_sha256"] == base["profile"]["contract_sha256"]
             and view["native_endpoint_loss"] is True and view["full_feature_profile"] is True
             and objective == checked["endpoint_objective_sha256"] == endpoint["new_objective_sha256"]
             and _read(endpoint["result"])["new_objective"] == objective, "VERIFIED_INPUT_OBJECTIVE")
    fit, held = data["geometry"]["fit"], data["geometry"]["held_out"]
    scope = view["reserved_observer_axes"]
    _require(fit["parent_range"] == [0, FIT_ROWS] and held["parent_range"] == [FIT_ROWS, FIT_ROWS + HELD_ROWS]
             and [FIT_ROWS, HELD_ROWS, FIT_CUTOFF, HELD_CUTOFF] == [stage[k] for k in
                  ("prefix_rows", "held_out_rows", "fit_cutoff_ns", "held_out_cutoff_ns")]
             and scope["source"] == data["physical_axes"] and scope["fit"]["cutoff_ns"] == FIT_CUTOFF
             and scope["held_out"]["cutoff_ns"] == HELD_CUTOFF
             and scope["fit"]["rows"] == checked["fitted_parents"] == FIT_ROWS
             and scope["held_out"]["rows"] == checked["reserved_parents"] == HELD_ROWS
             and all(scope[name][k + "_sha256"] == values[external + "_sha256_le_i64"]
                     for name, values in (("fit", fit), ("held_out", held)) for k, external in
                     (("parent_rows","parent_rows"), ("bar_start_ns","bar_start_ns"), ("decision_ns","issue_ns"))),
             "EXACT_DECEMBER_FOLD")
    _require(candle_reference["fit_issue_ns_sha256_le_i64"] == fit["issue_ns_sha256_le_i64"]
             and candle_reference["fit_parent_rows_sha256_le_i64"] == fit["parent_rows_sha256_le_i64"]
             and candle_reference["endpoint_result"] == refs["endpoint_result"]
             and candle_reference["normalization_result"] == refs["normalization_result"], "CANDLE_AXES")
    signs = _candle_signs(candle_reference, refs)
    counts, prior = _read(refs["prefix_counts"]), _read(refs["prefix_prior"])
    _require(counts["prefix_rows"] == prior["parents"] == FIT_ROWS and counts["cutoff_ns"] == FIT_CUTOFF
             and prior["counts"] == counts["class_counts"] and prior["transition_counts"] == counts["transition_counts"],
             "FIT_ONLY_REFERENCES")
    private, main = _numerics(old, objective)
    score = december_oof_scoring_contract()
    _require(score["fit_parents_half_open"] == fit["parent_range"] and score["oof_parents_half_open"] == held["parent_range"]
             and score["fit_cutoff_ns_exclusive"] == FIT_CUTOFF and score["oof_cutoff_ns_exclusive"] == HELD_CUTOFF,
             "SCORING_GEOMETRY")
    guard = _guards(policy, old, private_source)
    previous_private = _read(private_source["recipe"])
    support = _cohorts(refs["fit_target_view"], fit, main["seed"])
    output_bytes = dynamic_state_prediction_file_bytes(HELD_ROWS, len(view["research_objective"]["lead_minutes"]))
    plan = {"schema_version": SCHEMA, "status": "REGISTERED_PLAN_ONLY_NOT_NATIVE_ADMISSION",
        "authority": dict(CLOSED), "source_freeze": _source_freeze(stage, old),
        "bindings": {**refs, "frozen_candle_reference": candle_reference_ref, "private_recipe_lineage": private_source["recipe"]},
        "implementation_boundary": {"registered_plan_is_executable": False,
            "historical_boot_and_checkpoint_provenance_only": True,
            "legacy_fold_policy": "The existing _fold validator admits only its original FOLD_SPECS and rejects this fold.",
            "fit_and_private_targets": "require_dynamic_state_target_binding and _prepare_dynamic_state require a tape "
                "at OOF cutoff. Use only FIT_TARGETS at 2024-12-01 for private and main fit. Do not open a mixed tape.",
            "future_scoring_targets": "Separate physically delimited December [41866,47602) score-only target view at "
                "2025-01-01, published and opened only under separate admission. Never pick qualification parents here.",
            "feature_preparation": "Use the frozen masked FEATURE_VIEW and matching privately bound NativeDirectionObserverAxes "
                "and pass those exact arrays through the ordinary batch route. Historical _prepare_dynamic_state rebuilds "
                "an unmasked view and assumes one fit-plus-OOF population.",
            "scoring": "Bind the registered one-pass reporting contract to the native worker before reserved scoring. "
                "Saved marginal arrays and sequence NLL alone cannot reconstruct first-exit/return hazards."},
        "timeframes": list(targets.TIMEFRAMES), "class_order": list(targets.STATE_NAMES), "native_leads": score["native_lead_minutes"],
        "historical_event_time_only": True, "physical_xau_receive_atr_publication_and_restart_proven": False,
        "observer_axes": scope, "profile": base["profile"], "feature_view_contract_sha256": view["contract_sha256"],
        "objective_sha256": objective, "references": {"prior_markov_scope": "one fit-only frozen prior and transition matrix",
            "candle_orientations": signs, "missing_candle_blocks_scoring": any(s is None for s in signs.values())},
        "qualification": {"role": "private_mechanics_only", "numerical_recipe": private, "cohorts": support,
            "prior_private_reserved_parent_ids": _prior_private_parents(previous_private),
            "prior_private_parents_or_weights_reused": False, "own_zero_update_initial": True,
            "private_weight_transfer_to_main": False, "gradient_contact_required_for_all_native_and_head_groups": True,
            "bit_exact_reference_interrupted_resumed_state_required": True, "max_invocations": 1,
            **{k: previous_private[k] for k in ("max_model_constructions", "max_forwards", "max_backwards",
                "max_private_updates", "max_checkpoint_bytes", "max_published_json_bytes")}},
        "fit": {"prefix_rows": FIT_ROWS, "train_passes": 1, "fit_origins": 1, "seeds": [main["seed"]],
            "numerical_recipe": main, "full_minibatches": FIT_ROWS // main["batch_size"],
            "last_minibatch_parents": FIT_ROWS % main["batch_size"], "fresh_own_initial_before_fit": True,
            "discard_private_weights_and_optimizer_state": True, "initial_train_measurement_rows": old["initial_measurement_rows"],
            "final_train_measurement_rows": FIT_ROWS, "empty_eligible_no_step": True,
            **{k:old[k] for k in ("max_windows", "checkpoint_every_updates", "max_checkpoint_bytes",
                                 "max_checkpoint_total_bytes","max_published_json_bytes","parent_order_algorithm")}},
        "reserved_score": {"parents": HELD_ROWS, "max_complete_scores": 1, "refits": 0, "optimizer_updates": 0,
            "entry_condition": "only after terminal fit with frozen final checkpoint and distinct scoring admission",
            "contract": score, "predictions_created": 0, "outcomes_read": False,
            "base_npy_bytes": output_bytes, "base_npy_total_bytes": sum(output_bytes.values()),
            "event_artifact_size": None, "host_storage_headroom_qualified": False},
        "guards_for_any_future_invocation": guard,
        "open_gates": ["separate hash-bound qualification admission after guarded whole-host idle and safe physical reboot",
            "prefix-only feature and target preparation plus a separate immutable scoring view",
            "actual masked input and all-head gradient contact, own fresh baseline, checkpoint and bit-exact resume",
            "single-pass integrated per-issue event probabilities, matching references and score artifacts",
            "bind the versioned event-output representation and its size ceiling, including temporary storage, then check physical disk",
            "future admission must pin the published plan file SHA and validate all referenced files, source, campaign, resources and host"],
        "economic_or_independent_predictive_edge": None}
    plan["contract_sha256"] = campaign.canonical_sha256(plan)
    return require_closed_december_candidate_plan(plan)


def require_closed_december_candidate_plan(value):
    """Require the registered decision boundary even when an altered mapping is re-sealed."""
    p = dict(value)
    seal = p.pop("contract_sha256", None)
    _require(set(p) == {"schema_version", "status", "authority", "source_freeze", "bindings", "implementation_boundary",
                        "timeframes", "class_order", "native_leads", "historical_event_time_only",
                        "physical_xau_receive_atr_publication_and_restart_proven", "observer_axes", "profile",
                        "feature_view_contract_sha256", "objective_sha256", "references", "qualification", "fit",
                        "reserved_score", "guards_for_any_future_invocation", "open_gates", "economic_or_independent_predictive_edge"},
             "PLAN_FIELDS")
    _require(p.get("schema_version") == SCHEMA and seal == campaign.canonical_sha256(p), "PLAN_SEAL")
    _require(campaign.canonical_sha256(p["authority"]) == campaign.canonical_sha256(CLOSED) and p["status"] == "REGISTERED_PLAN_ONLY_NOT_NATIVE_ADMISSION"
             and "launcher_argv" not in p and "invocations" not in p and "run_scope" not in p, "NO_LAUNCH")
    fit, qualification, scored = p["fit"], p["qualification"], p["reserved_score"]
    scoring = december_oof_scoring_contract()
    base_bytes = dynamic_state_prediction_file_bytes(HELD_ROWS, len(targets.default_trial_lead_minutes()))
    _require(scored["base_npy_bytes"] == base_bytes and scored["base_npy_total_bytes"] == sum(base_bytes.values())
             and scored["event_artifact_size"] is None and scored["host_storage_headroom_qualified"] is False, "SCORE_STORAGE")
    _require(fit["train_passes"] == fit["fit_origins"] == 1 and fit["prefix_rows"] == FIT_ROWS
             and fit["numerical_recipe"]["max_updates"] == (FIT_ROWS+15)//16 and fit["seeds"] == [20260911]
             and fit["last_minibatch_parents"] == FIT_ROWS % 16 and fit["fresh_own_initial_before_fit"] is True
             and fit["discard_private_weights_and_optimizer_state"] is True
             and qualification["private_weight_transfer_to_main"] is False and qualification["max_invocations"] == 1
             and scored["parents"] == HELD_ROWS and scored["max_complete_scores"] == 1 and scored["refits"] == 0
             and scored["optimizer_updates"] == 0 and scored["outcomes_read"] is False and scored["contract"] == scoring,
             "ONE_FIXED_TRAJECTORY")
    for numerical, maximum in ((fit["numerical_recipe"], (FIT_ROWS + 15)//16), (qualification["numerical_recipe"], 2)):
        expected = native_direction_research_recipe(batch_size=16, max_updates=maximum, dynamic_state=True,
            current_state_first_loss_weight=0.03, first_flat_activity_weight=0.03, native_endpoint_loss=True)
        _require(numerical == expected and numerical["research_objective_sha256"] == p["objective_sha256"],
                 "REGISTERED_NUMERICS")
    _require(qualification["prior_private_reserved_parent_ids"] == [42787,44748,45861,47429]
             and qualification["prior_private_parents_or_weights_reused"] is False
             and qualification["max_model_constructions"] == 3 and qualification["max_forwards"] == 21
             and qualification["max_private_updates"] == qualification["max_backwards"] == 4
             and p["observer_axes"]["fit"]["cutoff_ns"] == FIT_CUTOFF
             and p["observer_axes"]["held_out"]["cutoff_ns"] == HELD_CUTOFF
             and p["guards_for_any_future_invocation"]["hardware"]["fresh_physical_windows_boot_before_every_invocation"] is True
             and p["guards_for_any_future_invocation"]["hardware"]["physical_power_limit_w"] == 250, "PHASE_GUARDS")
    order, private, initial = _cohort_rows(np.arange(FIT_ROWS, dtype=np.int64), fit["seeds"][0])
    cohort = qualification["cohorts"]
    _require(cohort["private_parents_sorted"] == private.tolist() and cohort["initial_parents_in_order"] == initial.tolist()
             and cohort["one_pass_order_sha256_le_i64"] == _axis_digest(order), "FIXED_COHORTS")
    return {**p, "contract_sha256": seal}



def read_closed_december_candidate_plan(expected_file):
    """For a future consumer the independent file SHA must come from its own frozen admission."""
    campaign.require_binding(expected_file, label="externally bound December candidate plan")
    return require_closed_december_candidate_plan(_read(expected_file))
