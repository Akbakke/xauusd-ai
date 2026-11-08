"""Bounded metadata checks for the new TA/M5 mechanical branch; no model/payload."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope
from gx1.contracts import market_state_time_query_fold1_fit_v1 as full


def _seal(value, key="recipe_sha256"):
    body = {k: v for k, v in value.items() if k != key}
    value[key] = (hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if key == "contract_sha256" else campaign.canonical_sha256(body))
    return value


@pytest.fixture(scope="module")
def metadata():
    # Read only the small completed metadata. No checkpoint/tape or model is opened.
    refs = scope.ta_m5_input_refs()
    names = ("ta_input_audit", "ta_input_terminal", "original_candidate_view",
             "ta_candidate_view", "source_review", "source_metadata_check",
             "original_recipe", "original_result", "original_final_measurement",
             "original_endpoint_priors", "ta_endpoint_priors")
    docs = {refs[name]["path"]: scope._read(refs[name], name) for name in names}
    paths = full.required_source_paths(ta_sf=True)
    current = {name: campaign.file_sha256(scope.CANONICAL_REPO / name) for name in paths}
    historical = {name: scope._ta_m5_git_source_sha(scope.TA_M5_REVIEWED_COMMIT, name)
                  for name in paths if not name.startswith(".venv/")
                  and name not in scope.TA_M5_ORCHESTRATION_SOURCE_PATHS}
    return refs, docs, current, historical


@pytest.fixture
def bound(monkeypatch, metadata):
    refs, docs, current, historical = copy.deepcopy(metadata)
    original_require = campaign.require_binding
    # Retain binding syntax validation; these tests never hash/load the checkpoint.
    monkeypatch.setattr(campaign, "require_binding",
        lambda ref, **kw: original_require(ref, label=kw.get("label", "test"), verify_file=False))
    monkeypatch.setattr(scope, "ta_m5_input_refs", lambda: copy.deepcopy(refs))
    monkeypatch.setattr(scope, "_read", lambda ref, label: copy.deepcopy(docs[ref["path"]]))
    monkeypatch.setattr(scope, "_ta_m5_git_source_sha", lambda commit, name: historical[name])
    monkeypatch.setattr(campaign, "file_sha256",
        lambda path: current[str(Path(path).relative_to(scope.CANONICAL_REPO))])
    recipe = scope.build_ta_m5_mechanical_recipe(source_commit=scope.TA_M5_REVIEWED_COMMIT)
    return recipe, refs, docs


def _require(value):
    return scope.require_ta_m5_mechanical_recipe(
        value, source_repo=scope.CANONICAL_REPO, source_commit=scope.TA_M5_REVIEWED_COMMIT)


def test_bound_new_recipe_preserves_old_route_and_exact_work(bound):
    value, refs, _ = bound
    assert _require(value) == value
    assert value["run_scope"] == scope.TA_M5_SCOPE != scope.TA_SCOPE
    assert value["admission_stage"] == scope.TA_M5_STAGE != scope.TA_STAGE
    assert value["numerical_recipe"] == full.ta_sf_numerical_recipe(dedicated_m5_side=True)
    assert value["parent_order_rows"] == 29574 and value["selected_fit_rows"] == 32
    assert (value["max_model_constructions"], value["max_optimizer_constructions"],
            value["max_forwards"], value["max_backwards"], value["max_private_updates"]) == (4, 4, 6, 4, 4)
    origin = value["warmstart_origin"]
    assert origin["original_checkpoint"] == refs["original_checkpoint"]
    assert origin["source_commit"] == scope.TA_M5_ORIGINAL_COMMIT
    assert origin["model_state_sha256"] == scope.TA_M5_ORIGINAL_MODEL_SHA256
    assert (origin["cursor"], origin["optimizer_updates"], origin["skipped_batches"]) == (29574, 1849, 0)
    assert origin["optimizer_migration"] == "fresh_empty_AdamW_cursor0"
    assert value["qualification_weights_for_main_fit"] is False


@pytest.mark.parametrize(("key", "replacement"), [
    ("run_scope", scope.TA_SCOPE), ("max_model_constructions", 3),
    ("max_optimizer_constructions", 3), ("max_forwards", 7),
    ("max_private_updates", 5), ("training_enabled", False),
    ("m5_side_parity_atol", 1e-5), ("m5_side_parity_rtol", 1e-4),
])
def test_resealed_recipe_cannot_expand_or_reinterpret_scope(bound, key, replacement):
    value, _, _ = bound
    value[key] = replacement
    with pytest.raises(Exception, match="TA_M5_RECIPE_SCOPE"):
        _require(_seal(value))


@pytest.mark.parametrize(("key", "replacement"), [
    ("source_commit", "0" * 40), ("model_state_sha256", "0" * 64),
    ("cursor", 0), ("optimizer_migration", "inherit_old_moments"),
])
def test_resealed_origin_cannot_invent_checkpoint_provenance(bound, key, replacement):
    value, _, _ = bound
    value["warmstart_origin"][key] = replacement
    with pytest.raises(Exception, match="TA_M5_RECIPE_WARMSTART"):
        _require(_seal(value))


@pytest.mark.parametrize("damage", ["input_window", "input_source", "prior", "model_digest", "old_cursor"])
def test_physical_metadata_reuse_requires_exact_inputs_priors_and_completed_origin(bound, damage):
    _, refs, docs = bound
    if damage == "input_window":
        view = docs[refs["ta_candidate_view"]["path"]]
        view["window_rows"]["M5"] += 1
        _seal(view, "contract_sha256")
        reason = "TA_M5_EXACT_INPUT_IDENTITY"
    elif damage == "input_source":
        view = docs[refs["ta_candidate_view"]["path"]]
        key = next(key for key in view["source_bindings"] if key != "time_query_v2_model_source")
        view["source_bindings"][key]["sha256"] = "0" * 64
        _seal(view, "contract_sha256")
        reason = "TA_M5_INPUT_SOURCE_BRIDGE"
    elif damage == "prior":
        prior = docs[refs["ta_endpoint_priors"]["path"]]
        prior["tasks"]["state"]["raw_logits"][0][1] += 0.25
        _seal(prior, "contract_sha256")
        reason = "TA_M5_PRIOR_VALUES_CHANGED"
    elif damage == "model_digest":
        docs[refs["original_final_measurement"]["path"]]["evaluated_model_state_sha256"] = "0" * 64
        reason = "TA_M5_ORIGINAL_FINAL_BINDING"
    else:
        docs[refs["original_result"]["path"]]["parent_cursor"] = 4096
        reason = "TA_M5_ORIGINAL_FINAL_BINDING"
    with pytest.raises(Exception, match=reason):
        scope._require_ta_m5_input_proof(refs)


@pytest.mark.parametrize("damage", ["unreviewed_owner", "reviewed_owner", "installed_owner"])
def test_source_bridge_rejects_non_orchestration_drift(bound, damage):
    value, refs, docs = bound
    if damage == "reviewed_owner":
        review = docs[refs["source_review"]["path"]]
        name = "gx1/contracts/market_state_time_query_native_F_model_v1.py"
        review["source_owners"][name] = "0" * 64
        reason = "TA_M5_REVIEWED_SOURCE_SHA"
    else:
        name = (next(name for name in value["source_owners"] if name.startswith(".venv/"))
                if damage == "installed_owner" else "gx1/contracts/entry_model_native_signal_v1.py")
        value["source_owners"][name]["sha256"] = "0" * 64
        _seal(value)
        reason = "TA_M5_UNREVIEWED_SOURCE_DRIFT"
    with pytest.raises(Exception, match=reason):
        _require(value)


def test_physical_validation_cannot_be_disabled(bound):
    with pytest.raises(Exception, match="TA_M5_PHYSICAL_METADATA_REQUIRED"):
        scope.require_ta_m5_mechanical_recipe(bound[0], source_repo=scope.CANONICAL_REPO,
            source_commit=scope.TA_M5_REVIEWED_COMMIT, verify_files=False)


def test_exact_mechanical_gate(bound):
    value, _, _ = bound
    gate = dict(status="ADMITTED", decision="ADMITTED", ta_m5_mechanical_authorized=True,
        fold_id=1, parent_selection=scope.TA_PARENT_SELECTION, selected_fit_rows=32,
        model_constructions_authorized=4, optimizer_constructions_authorized=4,
        model_forwards_authorized=6, backward_calls_authorized=4, optimizer_steps_authorized=4,
        genuine_predictor_fit_authorized=False, test_data_used=False)
    scope.require_ta_m5_mechanical_gate(gate, value)
    gate["optimizer_constructions_authorized"] = 3
    with pytest.raises(Exception, match="TA_M5_POLICY_GATE"):
        scope.require_ta_m5_mechanical_gate(gate, value)


def test_cursor_has_distinct_mode_and_exact_completion_report(bound, monkeypatch):
    ref = {"path": "/tmp/ta-m5-recipe.json", "sha256": "1" * 64}
    report_ref = {"path": "/tmp/ta-m5-result.json", "sha256": "2" * 64}
    state_ref = {"path": "/tmp/ta-m5-state.pt", "sha256": "3" * 64}
    kwargs = dict(recipe_ref=ref, outcome="COMPLETE", optimizer_updates=4,
        parent_cursor=32, parent_order_sha256="4" * 64, full_state=state_ref, result=report_ref)
    value = scope.build_cursor(**kwargs, ta_sf=True, dedicated_m5_side=True)
    assert value["schema_version"] == scope.TA_M5_CURSOR_SCHEMA
    assert scope.build_cursor(**kwargs, ta_sf=True)["schema_version"] == scope.TA_CURSOR_SCHEMA
    assert scope.build_cursor(**kwargs)["schema_version"] == scope.CURSOR_SCHEMA
    with pytest.raises(Exception, match="TA_M5_CURSOR_MODE"):
        scope.build_cursor(**kwargs, dedicated_m5_side=True)
    report = dict(schema_version=scope.TA_M5_RESULT_SCHEMA, recipe=ref,
        parent_order_sha256="4" * 64, parent_cursor=32, physical_optimizer_updates=4,
        final_state=state_ref, test_data_used=False, decision="PASS_TA_M5_MECHANICAL",
        bit_exact_resume=True)
    monkeypatch.setattr(campaign, "read_bound_json", lambda *args: copy.deepcopy(report))
    assert scope.require_cursor(value, expected_recipe=ref, verify_files=True) == value
    report["decision"] = "PASS_TA_SF_MECHANICAL"
    with pytest.raises(Exception, match="CURSOR_RESULT"):
        scope.require_cursor(value, expected_recipe=ref, verify_files=True)
