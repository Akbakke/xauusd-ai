"""Metadata-only checks for the registered, isolated December phase routes."""
from copy import deepcopy

import pytest

from gx1.contracts import entry_dynamic_mtf_december_native_v1 as dec
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import entry_native_direction_research_campaign_v1 as private


def ref(name):
    return {"path":"/synthetic/" + name, "sha256":"1"*64}


def test_private_recipe_independent_of_authority_and_resealed_mutations(monkeypatch):
    keys = ("historical_recipe","fit_target_view","feature_view","verified_input_profile","prefix_normalization",
            "prefix_counts","prefix_prior","candidate_geometry","frozen_candle_reference")
    refs = {k:ref(k) for k in keys}
    axes = {"rows":47602}
    numeric = {"dynamic_state_supervision":True,"native_endpoint_loss":True,"research_objective_sha256":"1"*64,"batch_size":16,"max_updates":2,"seed":20260911}
    view = {"contract_sha256":"2"*64, "prefix_normalization":{"binding":refs["prefix_normalization"]},
            "reserved_observer_axes":axes, "research_objective":{"contract_sha256":numeric["research_objective_sha256"]}}
    view_sha = dec.campaign.canonical_sha256({"contract_sha256":view["contract_sha256"],"prefix_normalization":view["prefix_normalization"],
                                               "research_objective_sha256":numeric["research_objective_sha256"]})
    cohorts = {"private_parents_sorted":list(range(32)),"initial_parents_in_order":list(range(256)),
               "private_batch_sparse_support":[{tf:{"native":{"observed":n}} for tf in ("M5","M15","H1","H4","D1")} for n in (9,10)]}
    plan = {"bindings":refs, "feature_view_contract_sha256":view["contract_sha256"],"observer_axes":axes,
            "objective_sha256":numeric["research_objective_sha256"],
            "qualification":{"numerical_recipe":numeric,"cohorts":cohorts,"max_private_updates":4,"max_model_constructions":3,
                "max_forwards":21,"max_backwards":4,"max_checkpoint_bytes":100,"max_published_json_bytes":50}}
    files = {"historical_recipe":{"input_manifest":ref("provider")}, "feature_view":view,
             "fit_target_view":{"metadata":{"parents":41866,"cutoff_ns":dec.FOLD["train_cutoff_ns"],"source_refs":{},"test_data_used":False}},
             "verified_input_profile":{"feature_view_contract_sha256":view["contract_sha256"],"normalization":refs["prefix_normalization"]}}
    monkeypatch.setattr(dec, "VIEW_IDENTITY_SHA", view_sha)
    monkeypatch.setattr(dec, "_registered", lambda: plan)
    monkeypatch.setattr(dec, "_read", lambda r: files[r["path"].split("/")[-1]])
    monkeypatch.setattr(dec, "_owners", lambda plan, phase, supplied: {"registered":ref("owner")})
    monkeypatch.setattr(dec.campaign, "require_binding", lambda r, **kwargs: r)
    build = dict(phase="private",runtime_root=dec.ROOTS["private"],admission_stage=dec.STAGES["private"])
    recipe = dec.build_december_native_recipe(**build)
    verify = dict(phase="private",source_repo=dec.REPO,source_commit="a"*40)
    assert dec.require_december_native_recipe(recipe, **verify) == recipe
    assert recipe["endpoint_support_per_batch"] == [[9]*5,[10]*5]
    assert private.guarded_research_power_limit(recipe) == 250
    for key, value in (("qualification_parent_rows", list(range(1,33))), ("initial_parent_rows", []),
                       ("input_role","score"),("target_manifest",ref("later"))):
        changed = {**deepcopy(recipe),key:value}
        changed["recipe_sha256"] = dec.campaign.canonical_sha256({k:v for k,v in changed.items() if k != "recipe_sha256"})
        with pytest.raises(RuntimeError, match="EXACT_REGISTERED_RECIPE"):
            dec.require_december_native_recipe(changed, **verify)
    with pytest.raises(RuntimeError, match="PRIVATE_NO_ORIGIN"):
        dec.build_december_native_recipe(**build, qualification_review=ref("old"))


def test_exact_phase_gate_and_real_endpoint_loss_units():
    fit = {"input_role":"fit","run_scope":dec.FIT_SCOPE,"full_feature_profile":True,"numerical_recipe":{"dynamic_state_supervision":True}}
    gate = {"candidate_plan":dec.PLAN_REF,"input_role":"fit","held_out_features_authorized":False,"held_out_outcomes_authorized":False}
    assert prefix.guarded_prefix_power_limit(fit) == 250
    dec.require_phase_gate(gate,fit)
    with pytest.raises(RuntimeError, match="SEPARATE_ROLE_ADMISSION"):
        dec.require_phase_gate({**gate,"held_out_features_authorized":True},fit)
    with pytest.raises(RuntimeError, match="FRESH_FIT_CHECKPOINT_SCORING_GATE"):
        dec.require_phase_gate({**gate,"input_role":"score","held_out_features_authorized":True,"held_out_outcomes_authorized":True},
                               {**fit,"run_scope":dec.SCORE_SCOPE,"input_role":"score"})
    good = {"native_endpoint_loss":1.04,"native_endpoint_eligible_cells":55,"native_endpoint_eligible_per_tf":[11]*5}
    dec.require_endpoint_step(good,[11]*5)
    for loss in (True, float("inf"), None, -0.4):
        with pytest.raises(RuntimeError, match="PRIVATE_ENDPOINT_SUPPORT"):
            dec.require_endpoint_step({**good,"native_endpoint_loss":loss},[11]*5)
    with pytest.raises(RuntimeError, match="PRIVATE_ENDPOINT_SUPPORT"):
        dec.require_endpoint_step(good,[9]*5)


def test_fit_and_score_cursors_keep_singular_roles(monkeypatch):
    monkeypatch.setattr(dec, "require_checkpoint_view", lambda x:x)
    args = dict(fold=dec.FOLD,feature_view_identity={},phase_state="COMPLETE",outcome="COMPLETE",optimizer_updates=2617,
                parent_cursor=41866,parent_order_sha256="0"*64,full_state=ref("terminal"),own_initial_state=ref("initial"),
                initial_measurement=ref("initial.json"),final_train_measurement=ref("fit.json"),result=ref("result"))
    for phase, rows in (("fit",0),("score",5736)):
        recipe = {"path":str(dec.ROOTS[phase].parent/"RECIPE.json"),"sha256":"0"*64}
        assert prefix.build_prefix_fit_cursor(**args,recipe_ref=recipe,oof_cursor=rows)["oof_cursor"] == rows
        with pytest.raises(Exception, match="CURSOR_COUNTER|CURSOR_COMPLETE"):
            prefix.build_prefix_fit_cursor(**args,recipe_ref=recipe,oof_cursor=5736-rows)
    with pytest.raises(Exception, match="FOLD_FIELDS"):
        prefix.build_prefix_fit_cursor(**args,recipe_ref=ref("unregistered"),oof_cursor=0)


def test_checkpoint_masked_schema_is_exact_and_legacy_unchanged(monkeypatch):
    from types import SimpleNamespace
    from gx1.contracts import entry_native_direction_research_state_v1 as state

    objective = {"contract_sha256":"2"*64}
    ordinary = {k:"a"*64 for k in ("contract_sha256","base_sha256","gc_layout_sha256","gc_normalization_sha256","qualification_sha256")}
    ordinary["binding"] = ref("frozen")
    identity = {"contract_sha256":"1"*64,"research_objective_sha256":objective["contract_sha256"],"prefix_normalization":ordinary}
    model = SimpleNamespace(feature_view_identity=identity,research_objective=objective,native_endpoint_loss=False)
    assert state._view(model, identity) == identity
    extended = {**identity, "prefix_normalization":{**ordinary,"zero_support_profile_sha256":"b"*64,"reserved_observer_axes_sha256":"c"*64}}
    monkeypatch.setattr(dec, "VIEW_IDENTITY_SHA", dec.campaign.canonical_sha256(extended))
    model.feature_view_identity = extended
    with pytest.raises(Exception, match="DECEMBER_CHECKPOINT_OBJECTIVE"):
        state._view(model, extended)
    model.native_endpoint_loss = True
    assert state._view(model, extended) == extended
    tampered = {**extended,"prefix_normalization":{**extended["prefix_normalization"],"reserved_observer_axes_sha256":"b"*64}}
    model.feature_view_identity = tampered
    with pytest.raises(RuntimeError, match="MASKED_CHECKPOINT_VIEW"):
        state._view(model, tampered)

def test_private_native_dispatch_rejects_archived_preflight_recipe(monkeypatch):
    root = dec.ROOTS["private"].parent
    recipe = {"run_scope":dec.PRIVATE_SCOPE}
    monkeypatch.setattr(dec.campaign, "require_binding", lambda r, **kwargs: r)
    monkeypatch.setattr(dec.campaign, "read_bound_json", lambda path, sha: recipe)
    monkeypatch.setattr(dec, "require_december_native_recipe", lambda value, **kwargs: value)
    args = {"source_repo":dec.REPO, "source_commit":"a"*40}
    for filename in ("RECIPE.json", "NATIVE_RECIPE.json"):
        binding = {"path":str(root/filename), "sha256":"1"*64}
        if filename == "RECIPE.json":
            with pytest.raises(private.campaign.RandomAccessCampaignError, match="DECEMBER_RECIPE_PATH"):
                private.require_research_launch_recipe(binding, **args)
        else:
            assert private.require_research_launch_recipe(binding, **args) is recipe
