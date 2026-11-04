"""Closed plan validation over deterministic metadata, never a native model."""
from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts import entry_dynamic_mtf_december_candidate_plan_v1 as plan


def _seal(item):
    value = {k: v for k, v in item.items() if k != "contract_sha256"}
    return {**value, "contract_sha256": plan.campaign.canonical_sha256(value)}


def _minimal():
    old = plan.native_direction_research_recipe(batch_size=16, max_updates=2976, dynamic_state=True,
        current_state_first_loss_weight=0.03, first_flat_activity_weight=0.03)
    objective = plan.native_direction_research_recipe(batch_size=16, max_updates=2617, dynamic_state=True,
        current_state_first_loss_weight=0.03, first_flat_activity_weight=0.03, native_endpoint_loss=True)["research_objective_sha256"]
    private, main = plan._numerics({"fold": {"prefix_rows": 47602}, "numerical_recipe": old}, objective)
    order, rows, initial = plan._cohort_rows(np.arange(41866), 20260911)
    sizes = plan.dynamic_state_prediction_file_bytes(5736, 1440)
    return _seal({"source_freeze": {}, "bindings": {}, "implementation_boundary": {}, "references": {},
        "timeframes": list(plan.targets.TIMEFRAMES), "class_order": list(plan.targets.STATE_NAMES),
        "native_leads": plan.december_oof_scoring_contract()["native_lead_minutes"], "profile": {},
        "feature_view_contract_sha256": "a"*64, "historical_event_time_only": True,
        "physical_xau_receive_atr_publication_and_restart_proven": False, "open_gates": [],
        "economic_or_independent_predictive_edge": None, "schema_version": plan.SCHEMA, "status": "REGISTERED_PLAN_ONLY_NOT_NATIVE_ADMISSION",
        "authority": dict(plan.CLOSED), "objective_sha256": objective,
        "observer_axes": {"fit": {"cutoff_ns": plan.FIT_CUTOFF}, "held_out": {"cutoff_ns": plan.HELD_CUTOFF}},
        "guards_for_any_future_invocation": {"hardware": {"fresh_physical_windows_boot_before_every_invocation": True,
                                                            "physical_power_limit_w": 250}},
        "fit": {"prefix_rows": 41866, "fit_origins": 1, "train_passes": 1, "numerical_recipe": main,
                "last_minibatch_parents": 10, "seeds": [20260911], "fresh_own_initial_before_fit": True,
                "discard_private_weights_and_optimizer_state": True},
        "qualification": {"numerical_recipe": private, "max_invocations": 1, "private_weight_transfer_to_main": False,
            "max_model_constructions": 3, "max_forwards": 21, "max_backwards": 4, "max_private_updates": 4,
            "prior_private_reserved_parent_ids": [42787, 44748, 45861, 47429], "prior_private_parents_or_weights_reused": False,
            "cohorts": {"private_parents_sorted": rows.tolist(), "initial_parents_in_order": initial.tolist(),
                        "one_pass_order_sha256_le_i64": plan._axis_digest(order)}},
        "reserved_score": {"parents": 5736, "max_complete_scores": 1, "refits": 0, "optimizer_updates": 0,
                           "outcomes_read": False, "contract": plan.december_oof_scoring_contract(), "base_npy_bytes": sizes,
                           "base_npy_total_bytes": sum(sizes.values()), "event_artifact_size": None, "host_storage_headroom_qualified": False}})


@pytest.mark.parametrize("part,key,value", [
    ("authority","training_enabled",True), ("authority","reboot_authorized",0),
    ("fit","prefix_rows",47602), ("fit","train_passes",2),
    ("qualification","private_weight_transfer_to_main",True), ("reserved_score","max_complete_scores",2),
])
def test_fixed_phases_remain_closed_after_self_reseal(part, key, value):
    valid = _minimal()
    assert plan.require_closed_december_candidate_plan(valid) == valid
    changed = deepcopy(valid)
    changed[part][key] = value
    with pytest.raises(ValueError):
        plan.require_closed_december_candidate_plan(_seal(changed))


def test_fresh_selection_excludes_reserved_parents_and_historical_launch_refuses_the_design():
    artifact = _minimal()
    assert max(artifact["qualification"]["cohorts"]["private_parents_sorted"]) < 41866
    artifact["qualification"]["cohorts"]["private_parents_sorted"][-1] = 42787
    with pytest.raises(ValueError, match="FIXED_COHORTS"):
        plan.require_closed_december_candidate_plan(_seal(artifact))
    for private in (True, False):
        with pytest.raises(RuntimeError, match="FIELDS"):
            plan.campaign_fit.require_dynamic_state_launch_recipe(_minimal(), source_repo=plan.REPO,
                                                                source_commit="a"*40, private_scope=private)


def test_sparse_cohort_counts_preserve_flat_unobserved_and_unknown_paths():
    m, tf, lead = 3, 5, 6
    state = np.broadcast_to(np.array([0, 1, -1], np.int8)[:, None, None], (m, tf, lead))
    query = {"state": state, "observed": state >= 0, "path_observed": state >= 0,
             "current_state": np.full((m, tf), plan.targets.UP, np.int8),
             "current_observed": np.ones((m, tf), np.bool_),
             "current_path_observed": np.arange(m)[:, None] + np.zeros((m,tf)) == 0,
             "first_flat_activity_observed": np.ones((m,tf), bool)}
    got = plan._cohort_counts(query, np.arange(m))
    for tf in plan.targets.TIMEFRAMES:
        assert got[tf]["native"] == {"observed": 2, "missing_or_immature": 1, "classes": {"DOWN":1,"FLAT":1,"UP":0}}
        assert got[tf]["first_minute_changes"] == got[tf]["first_minute_known_state_and_path_intervals"] == 1
        assert got[tf]["first_activity_observed"] == 3


@pytest.mark.parametrize("phase,member,value", [
    ("fit", "batch_size", 8), ("fit", "native_endpoint_loss", False),
    ("qualification", "max_updates", 3), ("qualification", "current_state_first_loss_weight", 0.01)])
def test_resealed_numerical_mismatch_is_not_a_new_qualification_or_fit(phase, member, value):
    item = _minimal()
    numeric = item[phase]["numerical_recipe"]
    numeric[member] = value
    numeric["contract_sha256"] = "1"*64
    with pytest.raises(ValueError, match="REGISTERED_NUMERICS"):
        plan.require_closed_december_candidate_plan(_seal(item))
