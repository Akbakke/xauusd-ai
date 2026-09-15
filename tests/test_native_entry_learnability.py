from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as ema
from gx1.contracts.unified_exit_random_access_state_view_v1 import _structured_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_native_optimizer_procedure_scope import _write, optimizer_origin, optimizer_scope, optimizer_history, scope
from tests.test_native_training_continuation import continuation_origin, continuation_scope
from tests.test_native_fqi_target_refresh import fqi_origin, fqi_scope
from tests.test_candidate_economics_transition import _optimizer_procedure_fixture, _identical


def _cohort(order):
    parents = torch.cat([order[offset * 16:(offset + 1) * 16] for offset in (1440, 1525, 1610, 1695)]).tolist()
    return {
        "epoch_index": 1, "full_training_epoch_index": 1,
        "actual_native_batch_offsets": [1440, 1525, 1610, 1695],
        "parent_rows": parents, "child_rows": parents.copy(),
        "epoch_order_sha256": _structured_sha256(order.cpu().numpy()),
        "selected_sample_plan_sha256": "a" * 64,
        "selection_uses_losses_or_outcomes": False,
        "replacement_of_unfavorable_rows_allowed": False,
    }


@pytest.fixture
def learnability_origin(fqi_origin, tmp_path, monkeypatch):
    directory = tmp_path / "stopped95"
    directory.mkdir()
    recipe = _write(tmp_path / "stopped95-recipe.json", {"candidate_resume_origin": fqi_origin})
    contract = _write(directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", {
        "recipe_source_provenance": {"recipe_audit_path": recipe["path"], "recipe_audit_sha256": recipe["sha256"]},
    })
    inherited = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(fqi_origin["contract"]["path"]),
        session_contract_sha256=fqi_origin["contract"]["sha256"],
    )
    _write(directory / native.FQI_TARGET_REFRESH_RECEIPT_NAME, {
        "schema_version": native.FQI_TARGET_REFRESH_RECEIPT_SCHEMA,
        "origin": fqi_origin, "origin_cursor": native.FQI_TARGET_REFRESH_ORIGIN_CURSOR,
        "origin_state_sha256": native.FQI_TARGET_REFRESH_ORIGIN_STATE_SHA256,
        "destination_session_contract_sha256": contract["sha256"],
        "inherited_weight_ema_history": inherited, "ema_internal_steps": 25429,
        "global_optimizer_steps": 5521, "state_preserved": False,
        "optimizer_procedure_changed": False, "identical_future_trajectory_claimed": False,
        "target_model_refreshed": True, "target_refresh_source": "origin_online_model_state",
        "original_target_model_state_sha256": "a" * 64,
        "refreshed_target_model_state_sha256": "b" * 64, "origin_online_model_state_sha256": "b" * 64,
        "training_progress_target_history": "retained_pre_refresh_history",
        "preserved_state_fields": ["model_state", "optimizer_state", "weight_ema_state",
                                   "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"],
    })
    state = directory / "candidate_training_state_slot_0.pt"
    state.write_bytes(b"scope fixture95; no native tensor deserialization")
    state_sha = native.file_sha256(state)
    pointer = _write(directory / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json", {
        **native.ENTRY_LEARNABILITY_ORIGIN_CURSOR,
        "schema_version": "gx1_candidate_training_session_v1", "slot": 0,
        "session_contract_sha256": contract["sha256"], "state_sha256": state_sha,
    })
    cohort = _write(tmp_path / "cohort.json", _cohort(torch.arange(65295, dtype=torch.int64)))
    for suffix, value in [("CONTRACT_SHA256", contract["sha256"]), ("POINTER_SHA256", pointer["sha256"]), ("STATE_SHA256", state_sha)]:
        monkeypatch.setattr(native, "ENTRY_LEARNABILITY_ORIGIN_" + suffix, value)
    monkeypatch.setattr(native, "ENTRY_LEARNABILITY_COHORT_SHA256", cohort["sha256"])
    return {"schema_version": native.ENTRY_LEARNABILITY_SCHEMA, "contract": contract, "pointer": pointer,
            "cohort": cohort, "exit_value_initialization": "close_now_baseline_v1",
            "train_population_scope": "latest_year_2025_2026_v1",
            "gradient_clipping_policy": native.OPTIMIZER_PROCEDURE_TRANSITION_POLICY}


@pytest.fixture
def learnability_scope(fqi_scope, learnability_origin):
    policy, recipe, write, save = fqi_scope
    recipe["candidate_resume_origin"] = learnability_origin
    policy["native_learning_calibration"]["schema_version"] = "gx1_native_entry_learnability_scope_v1"
    save()
    return policy, recipe, write, save


def test_replay_preserves_original_order_and_exact_resume_suffix():
    order = torch.arange(65295, dtype=torch.int64).flip(0)
    original, rng = order.clone(), torch.get_rng_state().clone()
    cohort = _cohort(order)
    expected = torch.tensor(cohort["parent_rows"], dtype=torch.int64).repeat(64)
    for offset in (1696, 1697, 1700, 1824, 1951, 1952):
        replay = trainer._candidate_entry_learnability_order(order, cohort=cohort, epoch_index=1, next_batch_offset=offset)
        assert replay.data_ptr() != order.data_ptr()
        assert torch.equal(order, original) and torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(replay[:1696*16], order[:1696*16])
        assert torch.equal(replay[1952*16:], order[1952*16:])
        assert torch.equal(replay[1696*16:1952*16], expected)
        actual = list(trainer._ExactIndexSampler(replay, batch_offset=offset, batch_size=16))
        assert actual[:(1952-offset)*16] == expected[(offset-1696)*16:].tolist()
        assert len(actual) == 65295-offset*16


@pytest.mark.parametrize("offset", [1695, 1953, -1, True, 1696.0])
def test_replay_rejects_outside_or_ambiguous_cursor(offset):
    order = torch.arange(65295, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        trainer._candidate_entry_learnability_order(order, cohort=_cohort(order), epoch_index=1, next_batch_offset=offset)


@pytest.mark.parametrize("epoch", [0, 2, True, 1.0])
def test_replay_rejects_other_sampler_epoch(epoch):
    order = torch.arange(65295, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        trainer._candidate_entry_learnability_order(order, cohort=_cohort(order), epoch_index=epoch, next_batch_offset=1696)


@pytest.mark.parametrize("fault", ["parent_order", "parent_duplicate", "child_duplicate", "outcome_selection", "replacement", "batch_offsets", "order_digest", "plan_digest"])
def test_replay_never_substitutes_cohort_or_accepts_tamper(fault):
    order = torch.arange(65295, dtype=torch.int64)
    cohort = _cohort(order)
    if fault == "parent_order": cohort["parent_rows"][0], cohort["parent_rows"][1] = cohort["parent_rows"][1], cohort["parent_rows"][0]
    elif fault == "parent_duplicate": cohort["parent_rows"][0] = cohort["parent_rows"][1]
    elif fault == "child_duplicate": cohort["child_rows"][0] = cohort["child_rows"][1]
    elif fault == "outcome_selection": cohort["selection_uses_losses_or_outcomes"] = True
    elif fault == "replacement": cohort["replacement_of_unfavorable_rows_allowed"] = True
    elif fault == "batch_offsets": cohort["actual_native_batch_offsets"][0] += 1
    elif fault == "order_digest": cohort["epoch_order_sha256"] = "0" * 64
    else: cohort["selected_sample_plan_sha256"] = "not-a-digest"
    with pytest.raises(RuntimeError):
        trainer._candidate_entry_learnability_order(order, cohort=cohort, epoch_index=1, next_batch_offset=1696)


def test_scope_admits_only_one_exact_6033_native_control(learnability_scope):
    _, recipe, _, _ = learnability_scope
    assert native.require_native_run_scope(recipe) == 6033
    assert native.require_native_run_scope(recipe, invocation_number=1) == 6033
    assert native.require_native_run_scope(recipe, execution_budget={
        "stop_after_optimizer_steps": 6033, "stop_after_completed_val_epochs": None, "max_invocation_seconds": 12000,
    }) == 6033
    with pytest.raises(RuntimeError, match="INVOCATION_INVALID"):
        native.require_native_run_scope(recipe, invocation_number=2)


@pytest.mark.parametrize("ceiling", [5777, 6032, 6034, 8162, True])
def test_scope_rejects_other_step_budgets(learnability_scope, ceiling):
    _, recipe, _, _ = learnability_scope
    with pytest.raises(RuntimeError, match="STEP_CEILING_INVALID"):
        native.require_native_run_scope(recipe, execution_budget={"stop_after_optimizer_steps": ceiling})


@pytest.mark.parametrize("fault", ["full_training", "old_scope", "extra_steps", "report_only_val"])
def test_scope_never_uses_old_authority_or_whole_epoch(learnability_scope, fault):
    policy, recipe, _, save = learnability_scope
    if fault == "full_training": policy["training_enabled"] = True
    elif fault == "old_scope": policy["native_learning_calibration"]["schema_version"] = "gx1_native_fqi_target_refresh_scope_v1"
    elif fault == "extra_steps": policy["native_learning_calibration"]["additional_optimizer_step_ceilings"] = [256, 512]
    else: recipe["native_calibration"] = {"schema_version": "gx1_native_learning_calibration_run_v1", "arm": "reference", "report_only_val": True}
    save()
    with pytest.raises(RuntimeError): native.require_native_run_scope(recipe, invocation_number=1)


@pytest.mark.parametrize("artifact", ["contract", "pointer", "state", "cohort"])
def test_origin_rejects_any_changed_input(learnability_origin, artifact):
    path = (Path(learnability_origin["pointer"]["path"]).parent / "candidate_training_state_slot_0.pt"
            if artifact == "state" else Path(learnability_origin[artifact]["path"]))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError): native.require_entry_learnability_origin(learnability_origin)


@pytest.mark.parametrize("old_schema,validator", [
    ("FQI_TARGET_REFRESH_SCHEMA", "require_fqi_target_refresh_origin"),
    ("TRAINING_CONTINUATION_SCHEMA", "require_training_continuation_origin"),
    ("OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA", "require_optimizer_procedure_origin"),
])
def test_new_origin_cannot_escape_into_old_production_or_control_authority(learnability_origin, old_schema, validator):
    changed = {k: v for k, v in learnability_origin.items() if k != "cohort"}
    changed["schema_version"] = getattr(native, old_schema)
    with pytest.raises(RuntimeError): getattr(native, validator)(changed)


def test_real_transition_preserves_every_state_and_declares_replay(tmp_path, monkeypatch):
    (old, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch, entry_learnability=True, cohort_factory=_cohort)
    before = old.load_checkpoint()
    old_hashes = {p.name: trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}
    state = trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    for key in before.keys() - {"session_contract_sha256"}: _identical(state[key], before[key])
    assert state["session_contract_sha256"] == new.contract_sha256
    receipt_path = new.directory / native.ENTRY_LEARNABILITY_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    assert receipt["state_preserved"] is True and receipt["optimizer_procedure_changed"] is False
    assert receipt["changed_sample_history"] is True
    assert receipt["data_coverage_advanced"] is False and receipt["production_continuation_allowed"] is False
    assert set(receipt["preserved_state_fields"]) == before.keys() - {"session_contract_sha256"}
    assert receipt["ema_internal_steps"] == 25685
    _identical(state, trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin))
    receipt_path.write_text(json.dumps({**receipt, "production_continuation_allowed": True}))
    with pytest.raises(RuntimeError, match="RECEIPT_MISMATCH"):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    receipt_path.write_bytes(trainer._candidate_training_session_json_bytes(receipt))
    new.save_checkpoint(state)
    _identical(state, new.load_checkpoint())
    with pytest.raises(RuntimeError, match="SESSION_BINDING_INVALID"):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    assert old_hashes == {p.name: trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("fault,error", [
    ("model_source", "UNAPPROVED_SOURCE_CHANGED"), ("data", "MODEL_DATA_OR_TRAINING_CHANGED"),
    ("batch", "MODEL_DATA_OR_TRAINING_CHANGED"), ("clip_cap", "CLIPPING_POLICY_INVALID"),
    ("old_clipping_missing", "CLIPPING_POLICY_INVALID"), ("cursor", "STOPPED_STATE_INVALID"), ("ema_steps", "EMA_HISTORY_INVALID"),
])
def test_transition_refuses_unrelated_learning_or_lineage_changes(tmp_path, monkeypatch, fault, error):
    (_, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch, fault, entry_learnability=True, cohort_factory=_cohort)
    with pytest.raises(RuntimeError, match=error): trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    assert not new._active_path.exists()
    assert not (new.directory / native.ENTRY_LEARNABILITY_RECEIPT_NAME).exists()


def test_history_binds_95_91_87_85_chain_and_diagnostic_restrictions(learnability_origin, tmp_path):
    directory = tmp_path / "replay-session"
    directory.mkdir()
    recipe = _write(tmp_path / "replay-recipe.json", {"candidate_resume_origin": learnability_origin})
    contract = _write(directory / "CANDIDATE_TRAINING_SESSION_CONTRACT.json", {
        "recipe_source_provenance": {"recipe_audit_path": recipe["path"], "recipe_audit_sha256": recipe["sha256"]},
    })
    inherited = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(learnability_origin["contract"]["path"]),
        session_contract_sha256=learnability_origin["contract"]["sha256"],
    )
    cohort = json.loads(Path(learnability_origin["cohort"]["path"]).read_text())
    receipt = {
        "schema_version": native.ENTRY_LEARNABILITY_RECEIPT_SCHEMA,
        "origin": learnability_origin, "origin_cursor": native.ENTRY_LEARNABILITY_ORIGIN_CURSOR,
        "origin_state_sha256": native.ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256,
        "destination_session_contract_sha256": contract["sha256"],
        "inherited_weight_ema_history": inherited, "ema_internal_steps": 25685,
        "global_optimizer_steps": 5777, "state_preserved": True,
        "optimizer_procedure_changed": False, "identical_future_trajectory_claimed": False,
        "changed_sample_history": True, "data_coverage_advanced": False, "production_continuation_allowed": False,
        "fixed_teacher_model_state_sha256": native.ENTRY_LEARNABILITY_TARGET_MODEL_SHA256,
        "replay_policy": {**native.ENTRY_LEARNABILITY_REPLAY_POLICY, "epoch_order_sha256": cohort["epoch_order_sha256"],
                          "selected_sample_plan_sha256": cohort["selected_sample_plan_sha256"]},
        "preserved_state_fields": ["model_state", "target_model_state", "optimizer_state", "weight_ema_state",
                                   "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"],
    }
    path = directory / native.ENTRY_LEARNABILITY_RECEIPT_NAME
    _write(path, receipt)
    kwargs = {"session_contract_path": Path(contract["path"]), "session_contract_sha256": contract["sha256"]}
    history = ema.bind_candidate_weight_ema_history_v1(**kwargs)
    assert history == {"optimizer_step_offset": 19908, "transition_receipt": {"path": str(path), "sha256": native.file_sha256(path)}}
    assert ema._require_candidate_ema_history_offset(history, contract_path=Path(contract["path"])) == 19908
    for field, value in [("state_preserved", False), ("optimizer_procedure_changed", True),
                         ("changed_sample_history", False), ("production_continuation_allowed", True),
                         ("data_coverage_advanced", True), ("replay_policy", {}),
                         ("fixed_teacher_model_state_sha256", "0" * 64), ("ema_internal_steps", 5777)]:
        _write(path, {**receipt, field: value})
        with pytest.raises(RuntimeError): ema.bind_candidate_weight_ema_history_v1(**kwargs)
    _write(path, receipt)
    ancestor = Path(inherited["transition_receipt"]["path"])
    _write(ancestor, {**json.loads(ancestor.read_text()), "unexpected_rewrite": True})
    with pytest.raises(RuntimeError, match="HISTORY_INVALID"): ema.bind_candidate_weight_ema_history_v1(**kwargs)


def test_restore_requires_explicit_diagnostic_receipt_and_preserves_teacher(tmp_path, monkeypatch):
    (old, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch, entry_learnability=True, cohort_factory=_cohort)
    before = old.load_checkpoint()
    state = trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    path = new.directory / native.ENTRY_LEARNABILITY_RECEIPT_NAME
    monkeypatch.setattr(ema, "bind_candidate_weight_ema_history_v1", lambda **kwargs: {
        "optimizer_step_offset": 19908, "transition_receipt": {"path": str(path), "sha256": trainer._sha256_file(path)},
    })
    model = torch.nn.Linear(3, 2)
    target = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    average = trainer._WeightEma(model, .99)
    kwargs = {"session": new, "model": model, "target_model": target, "optimizer": optimizer,
              "weight_ema": average, "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
              "device": torch.device("cpu"), "dataset_rows": 65295}
    with pytest.raises(RuntimeError, match="PRODUCTION_RESUME_FORBIDDEN"):
        trainer._restore_candidate_training_checkpoint(state, **kwargs)
    kwargs["entry_learnability_origin"] = origin
    trainer._restore_candidate_training_checkpoint(state, **kwargs)
    assert target.training is False and all(not p.requires_grad for p in target.parameters())
    _identical(target.state_dict(), before["target_model_state"])
    assert average._steps == 25685
    changed = copy.deepcopy(state)
    changed["target_model_state"]["weight"].add_(1.0)
    with pytest.raises(RuntimeError, match="ENTRY_LEARNABILITY_STATE_INVALID"):
        trainer._restore_candidate_training_checkpoint(changed, **kwargs)
    for field, value in [("global_optimizer_steps", 5778), ("phase", "validation"), ("complete", True), ("next_batch_offset", 1953)]:
        with pytest.raises(RuntimeError): trainer._restore_candidate_training_checkpoint({**state, field: value}, **kwargs)
    saved = path.read_bytes()
    changed_receipt = json.loads(saved)
    changed_receipt["origin"]["cohort"]["sha256"] = "0" * 64
    _write(path, changed_receipt)
    with pytest.raises(RuntimeError, match="RECEIPT_ORIGIN_MISMATCH"):
        trainer._restore_candidate_training_checkpoint(state, **kwargs)
    path.write_bytes(saved)
    monkeypatch.setattr(ema, "bind_candidate_weight_ema_history_v1", lambda **kwargs: {"optimizer_step_offset": 19908,
        "transition_receipt": {"path": str(new.directory / native.FQI_TARGET_REFRESH_RECEIPT_NAME), "sha256": "a" * 64}})
    with pytest.raises(RuntimeError, match="RECEIPT_MISSING"):
        trainer._restore_candidate_training_checkpoint(state, **kwargs)


@pytest.mark.parametrize("count,expected", [(65295, "after_epoch_check"), (16, "cannot complete a TRAIN epoch")])
def test_materializer_keeps_control_before_full_epoch(learnability_scope, tmp_path, monkeypatch, count, expected):
    from gx1.scripts import materialize_local_random_access_campaign_v2 as materializer
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _, recipe, _, _ = learnability_scope
    monkeypatch.setattr(materializer, "_source_commit", lambda _: "a" * 40)
    monkeypatch.setattr(native, "require_native_recipe_metadata", lambda *a, **kw: (recipe, count))
    def stop_before_prior(*_args, **_kwargs): raise RuntimeError("after_epoch_check")
    monkeypatch.setattr(campaign, "read_bound_json", stop_before_prior)
    with pytest.raises(RuntimeError, match=expected):
        materializer.materialize_native_candidate_campaign(
            repo=tmp_path, output=tmp_path / "uncreated", runtime=tmp_path / "uncreated-runtime",
            gpu_uuid="fixture", prepared_boot_path=tmp_path / "boot.json", prepared_boot_file_sha256="a" * 64,
            certificate_path=tmp_path / "certificate", prior_campaign_path=tmp_path / "prior.json",
            prior_campaign_file_sha256="a" * 64, selection_path=tmp_path / "selection.json",
            selection_file_sha256="a" * 64, recipe_path=tmp_path / "recipe.json", recipe_file_sha256="a" * 64, window_count=1,
        )
    assert not (tmp_path / "uncreated").exists() and not (tmp_path / "uncreated-runtime").exists()


@pytest.fixture
def continued_learnability_origin(learnability_origin, tmp_path, monkeypatch):
    directory = tmp_path / 'stopped99'
    directory.mkdir()
    recipe = _write(tmp_path / 'stopped99-recipe.json', {'candidate_resume_origin': learnability_origin})
    contract = _write(directory / 'CANDIDATE_TRAINING_SESSION_CONTRACT.json', {
        'recipe_source_provenance': {'recipe_audit_path': recipe['path'], 'recipe_audit_sha256': recipe['sha256']},
    })
    inherited = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(learnability_origin['contract']['path']),
        session_contract_sha256=learnability_origin['contract']['sha256'])
    cohort = json.loads(Path(learnability_origin['cohort']['path']).read_text())
    _write(directory / native.ENTRY_LEARNABILITY_RECEIPT_NAME, {
        'schema_version': native.ENTRY_LEARNABILITY_RECEIPT_SCHEMA,
        'origin': learnability_origin, 'origin_cursor': native.ENTRY_LEARNABILITY_ORIGIN_CURSOR,
        'origin_state_sha256': native.ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256,
        'destination_session_contract_sha256': contract['sha256'],
        'inherited_weight_ema_history': inherited, 'ema_internal_steps': 25685,
        'global_optimizer_steps': 5777, 'state_preserved': True,
        'optimizer_procedure_changed': False, 'identical_future_trajectory_claimed': False,
        'changed_sample_history': True, 'data_coverage_advanced': False, 'production_continuation_allowed': False,
        'fixed_teacher_model_state_sha256': native.ENTRY_LEARNABILITY_TARGET_MODEL_SHA256,
        'replay_policy': {**native.ENTRY_LEARNABILITY_REPLAY_POLICY,
            'epoch_order_sha256': cohort['epoch_order_sha256'],
            'selected_sample_plan_sha256': cohort['selected_sample_plan_sha256']},
        'preserved_state_fields': ['model_state', 'target_model_state', 'optimizer_state', 'weight_ema_state',
            'lr_scheduler_state', 'rng_state', 'epoch_order', 'training_progress'],
    })
    state = directory / 'candidate_training_state_slot_0.pt'
    state.write_bytes(b'scope fixture99; no native tensor deserialization')
    state_sha = native.file_sha256(state)
    pointer = _write(directory / 'CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json', {
        **native.ENTRY_LEARNABILITY_CONTINUATION_CURSOR,
        'schema_version': 'gx1_candidate_training_session_v1', 'slot': 0,
        'session_contract_sha256': contract['sha256'], 'state_sha256': state_sha,
    })
    for suffix, value in [('CONTRACT_SHA256', contract['sha256']), ('POINTER_SHA256', pointer['sha256']), ('STATE_SHA256', state_sha)]:
        monkeypatch.setattr(native, 'ENTRY_LEARNABILITY_CONTINUATION_' + suffix, value)
    return {**learnability_origin, 'contract': contract, 'pointer': pointer}


@pytest.fixture
def continued_learnability_scope(learnability_scope, continued_learnability_origin):
    policy, recipe, write, save = learnability_scope
    recipe['candidate_resume_origin'] = continued_learnability_origin
    policy['native_learning_calibration']['additional_optimizer_step_ceilings'] = [1024]
    save()
    return policy, recipe, write, save


def test_continued_scope_binds99_without_reinterpreting95(continued_learnability_scope, learnability_origin):
    policy, recipe, _, save = continued_learnability_scope
    assert native.require_native_run_scope(recipe, invocation_number=1) == 7057
    assert native.entry_learnability_control(learnability_origin)['step_ceiling'] == 6033
    assert native.entry_learnability_control(recipe['candidate_resume_origin'])['cursor']['checkpoint_index'] == 99
    with pytest.raises(RuntimeError, match='INVOCATION_INVALID'):
        native.require_native_run_scope(recipe, invocation_number=2)
    recipe['candidate_resume_origin'] = learnability_origin
    save()
    with pytest.raises(RuntimeError, match='CALIBRATION_SCOPE_INVALID'):
        native.require_native_run_scope(recipe)
    policy['native_learning_calibration']['additional_optimizer_step_ceilings'] = [256]
    save()
    assert native.require_native_run_scope(recipe) == 6033


@pytest.mark.parametrize('ceiling', [6033, 7056, 7058, 8162, True])
def test_continued_scope_rejects_wrong_budget(continued_learnability_scope, ceiling):
    _, recipe, _, _ = continued_learnability_scope
    with pytest.raises(RuntimeError, match='STEP_CEILING_INVALID'):
        native.require_native_run_scope(recipe, execution_budget={'stop_after_optimizer_steps': ceiling})


@pytest.mark.parametrize('offset', [1952, 1953, 2208, 2975, 2976])
def test_continued_replay_preserves_order_and_remaining_suffix(offset):
    order = torch.arange(65295, dtype=torch.int64).flip(0)
    original, rng = order.clone(), torch.get_rng_state().clone()
    cohort = _cohort(order)
    origin = {'contract': {'sha256': native.ENTRY_LEARNABILITY_CONTINUATION_CONTRACT_SHA256}}
    replay = trainer._candidate_entry_learnability_order(order, cohort=cohort, epoch_index=1, next_batch_offset=offset, origin=origin)
    expected = torch.tensor(cohort['parent_rows'], dtype=torch.int64).repeat(256)
    assert torch.equal(order, original) and torch.equal(torch.get_rng_state(), rng)
    assert replay.data_ptr() != order.data_ptr()
    assert torch.equal(replay[:1952*16], original[:1952*16])
    assert torch.equal(replay[2976*16:], original[2976*16:])
    assert torch.equal(replay[1952*16:2976*16], expected)
    assert list(trainer._ExactIndexSampler(replay, batch_offset=offset, batch_size=16))[:(2976-offset)*16] == expected[(offset-1952)*16:].tolist()


@pytest.mark.parametrize('offset', [1951, 2977, True, 1952.0])
def test_continued_replay_rejects_wrong_cursor(offset):
    order = torch.arange(65295, dtype=torch.int64)
    origin = {'contract': {'sha256': native.ENTRY_LEARNABILITY_CONTINUATION_CONTRACT_SHA256}}
    with pytest.raises(RuntimeError):
        trainer._candidate_entry_learnability_order(order, cohort=_cohort(order), epoch_index=1, next_batch_offset=offset, origin=origin)


@pytest.mark.parametrize('artifact', ['contract', 'pointer', 'state'])
def test_continued_origin_rejects_changed_checkpoint(continued_learnability_origin, artifact):
    origin = continued_learnability_origin
    path = Path(origin['pointer']['path']).parent / 'candidate_training_state_slot_0.pt' if artifact == 'state' else Path(origin[artifact]['path'])
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(RuntimeError): native.require_entry_learnability_origin(origin)


def test_continued_history_retains_old95_receipt(continued_learnability_origin, learnability_origin):
    origin = continued_learnability_origin
    assert native.require_entry_learnability_origin(origin) == origin
    history = ema.bind_candidate_weight_ema_history_v1(
        session_contract_path=Path(origin['contract']['path']), session_contract_sha256=origin['contract']['sha256'])
    assert history['optimizer_step_offset'] == 19908
    receipt_path = Path(history['transition_receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    assert receipt['origin'] == learnability_origin
    assert receipt['replay_policy']['batch_offset_end'] == 1952
    _write(receipt_path, {**receipt, 'replay_policy': native.ENTRY_LEARNABILITY_CONTINUATION_REPLAY_POLICY})
    with pytest.raises(RuntimeError):
        ema.bind_candidate_weight_ema_history_v1(
            session_contract_path=Path(origin['contract']['path']), session_contract_sha256=origin['contract']['sha256'])


def test_actual_continued_transition_preserves_state_and_declares_new_bound(tmp_path, monkeypatch):
    old_cursor = native.ENTRY_LEARNABILITY_ORIGIN_CURSOR
    monkeypatch.setattr(native, 'ENTRY_LEARNABILITY_ORIGIN_CURSOR', native.ENTRY_LEARNABILITY_CONTINUATION_CURSOR)
    (old, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch, entry_learnability=True, cohort_factory=_cohort)
    monkeypatch.setattr(native, 'ENTRY_LEARNABILITY_ORIGIN_CURSOR', old_cursor)
    for suffix, value in [('CONTRACT_SHA256', origin['contract']['sha256']), ('POINTER_SHA256', origin['pointer']['sha256']), ('STATE_SHA256', native.ENTRY_LEARNABILITY_ORIGIN_STATE_SHA256)]:
        monkeypatch.setattr(native, 'ENTRY_LEARNABILITY_CONTINUATION_' + suffix, value)
    before = old.load_checkpoint()
    state = trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    for key in before.keys() - {'session_contract_sha256'}: _identical(state[key], before[key])
    receipt = json.loads((new.directory / native.ENTRY_LEARNABILITY_RECEIPT_NAME).read_text())
    assert receipt['origin_cursor'] == native.ENTRY_LEARNABILITY_CONTINUATION_CURSOR
    assert receipt['ema_internal_steps'] == 25941
    assert receipt['replay_policy']['batch_offset_start'] == 1952
    assert receipt['replay_policy']['batch_offset_end'] == 2976
    assert receipt['replay_policy']['repeats'] == 256
    assert receipt['production_continuation_allowed'] is False
    assert receipt['state_preserved'] is True and receipt['optimizer_procedure_changed'] is False
    new.save_checkpoint(state)
    _identical(new.load_checkpoint(), state)
