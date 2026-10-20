"""Fixed-step ONLINE comparison through the native owner; synthetic evidence only."""
import copy
import json
import time
from pathlib import Path

import pytest
import torch

from tests.test_native_prefix_initial_measurement import initial_scope, prefix_scope, scope, dynamic_initial_scope, dynamic_initial_observer
from tests.test_native_prefix_recipe import native, runner, _write, _bind, dynamic_native_recipe_metadata
from tests.test_native_prefix_recipe import test_native_campaign_materialization_window_and_final_review_stop as campaign_check
from tests.test_native_prefix_coordinator import PrefixHarness, _prepared, equal_tree, digest, trainer
from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts


@pytest.fixture
def learning_scope(initial_scope, tmp_path):
    policy, recipe, files, seal = initial_scope
    initial_artifacts = recipe.pop('chronological_initial_measurement')
    policy.pop('chronological_initial_measurement')
    measurement = json.loads(Path(initial_artifacts['measurement_binding_result']['path']).read_text())
    observations = {}
    for role in ['train','control']:
        observations[role] = _write(tmp_path/(role+'.json'), {
            'role':role,'optimizer_steps':0,'model_state_sha256':'a'*64,'target_model_state_sha256':'a'*64,
            'test_data_used':False,'cohort':{'plan':recipe['chronological_prefix']['design'],
                'measurement_coordinates':measurement['coordinate_result'],
                'measurement_role':role,'entry_row_indices':list(range(256))}})
    result = {'schema_version':'gx1_native_prefix_initial_measurement_v1',
        'decision':'FROZEN_INITIAL_TARGETS_AND_PREDICTIONS_READY_NO_LEARNING_MEASURED',
        **initial_artifacts,'optimizer_steps':0,'model_state_sha256':'a'*64,'target_model_state_sha256':'a'*64,
        'teacher_refreshed':False,'economic_rollout':False,'test_data_used':False,'observations':observations}
    audit = {'schema_version':'gx1_native_prefix_initial_measurement_audit_v1',
        'decision':'FROZEN_INITIAL_MEASUREMENT_VERIFIED_NO_LEARNING_MEASURED',
        'result':_write(tmp_path/'before.json',result),'optimizer_steps':0,'test_data_used':False,
        'fresh_model_optimizer_ema_scheduler_exactly_preserved':True,'saved_cpu_python_numpy_rng_exactly_preserved':True,
        'receipt':_write(tmp_path/'terminal.json',{'guard_decision':'PASS','outcome':'RESUMABLE',
            'trainer_guard_exit_code':0,'progress_observer_exit_code':0,'test_data_used':False})}
    binding = _write(tmp_path/'audit.json',audit)
    recipe['chronological_learning_measurement'] = binding
    policy['chronological_learning_run'] = {'chronological_prefix':recipe['chronological_prefix'],
        'run_id':recipe['run_id'],'out_bundle_dir':recipe['out_bundle_dir'],
        'source_bindings_sha256':recipe['source_bindings_sha256'],'optimizer_steps':256,
        'maximum_trained_entry_rows':4096,'max_invocations':3,'final_model_variant':'ONLINE',
        'teacher_refresh_allowed':False,'full_epoch_training_allowed':False,'full_val_allowed':False,
        'test_data_used':False,'chronological_learning_measurement':binding}
    seal()
    return policy,recipe,files,seal


def test_learning_scope_binds_initial_evidence_and_fixed_budget(learning_scope):
    _, recipe, _, _ = learning_scope
    for invocation in [1,2,3]: assert native.require_native_run_scope(recipe,invocation_number=invocation) == 256
    value = native.require_chronological_learning_measurement(recipe)
    assert value['initialization']['optimizer_state_empty'] is True
    assert set(value['initial_measurement']['observations']) == {'train','control'}
    budget = {'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    assert native.require_native_run_scope(recipe,execution_budget=budget) == 256
    for steps in [0,255,257]:
        with pytest.raises(RuntimeError,match='PREFIX_BUDGET'):
            native.require_native_run_scope(recipe,execution_budget={**budget,'stop_after_optimizer_steps':steps})


@pytest.mark.parametrize('fault',['unbound_policy','changed_audit','changed_observation','changed_design','mixed_scope'])
def test_learning_admission_rejects_changed_frozen_evidence(learning_scope, fault):
    policy,recipe,_,seal = learning_scope
    ab = recipe['chronological_learning_measurement']; audit = json.loads(Path(ab['path']).read_text())
    if fault == 'unbound_policy': policy['chronological_learning_run'].pop('chronological_learning_measurement')
    elif fault == 'changed_audit': Path(ab['path']).write_text('{}')
    elif fault == 'mixed_scope': policy['chronological_initial_measurement'] = {}
    else:
        result = json.loads(Path(audit['result']['path']).read_text())
        ob = result['observations']['control']
        if fault == 'changed_observation': Path(ob['path']).write_text('{}')
        else:
            observation = json.loads(Path(ob['path']).read_text()); observation['cohort']['plan'] = {'wrong':True}
            result['observations']['control'] = _write(Path(ob['path']),observation)
            audit['result'] = _write(Path(audit['result']['path']),result)
            binding = _write(Path(ab['path']),audit)
            recipe['chronological_learning_measurement'] = binding
            policy['chronological_learning_run']['chronological_learning_measurement'] = binding
    seal()
    with pytest.raises(RuntimeError): native.require_native_run_scope(recipe,invocation_number=1)


@pytest.mark.parametrize('fault',[None,'false','unbound','missing_measurement','initial','diagnostic','no_prefix'])
def test_train_only_measurement_is_explicitly_bound(learning_scope,fault):
    policy,recipe,_,seal=learning_scope
    recipe['chronological_train_only_measurement']=False if fault=='false' else True
    if fault!='unbound':policy['chronological_learning_run']['chronological_train_only_measurement']=True
    if fault=='missing_measurement':recipe.pop('chronological_learning_measurement')
    if fault=='initial':recipe['chronological_initial_measurement']={}
    if fault=='diagnostic':recipe['entry_gradient_diagnostic']={}
    if fault=='no_prefix':recipe.pop('chronological_prefix')
    seal()
    if fault:
        with pytest.raises(RuntimeError):native.require_native_run_scope(recipe,invocation_number=1)
    else:
        assert native.require_native_run_scope(recipe,invocation_number=1)==256


def test_learning_campaign_stops_for_review_at_fixed256(learning_scope,monkeypatch,tmp_path):
    campaign_check(learning_scope,monkeypatch,tmp_path)


@pytest.mark.parametrize('steps,reason',[(128,'invocation_wall_limit'),(256,'optimizer_step_ceiling')])
def test_native_dispatch_restores_initial_state_and_measures_only_final_online(learning_scope,monkeypatch,tmp_path,steps,reason):
    _,recipe,files,seal = learning_scope; rb=seal(); seen=[]
    budget={'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    monkeypatch.setattr(runner.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:seen.append('guard'))
    monkeypatch.setattr(runner.trainer,'_resolve_device',lambda _:torch.device('cpu'))
    monkeypatch.setattr(runner,'_require_native_full_train_recipe',lambda *a:(recipe,files,{}))
    monkeypatch.setattr(runner.launch_owner,'require_candidate_execution_budget',lambda *a,**kw:budget)
    monkeypatch.setattr(runner,'_build_bound_full_train_components',lambda **kw:{'chronological_prefix':kw['chronological_prefix']})
    monkeypatch.setattr(runner,'_restore_prefix_initial_measurement_state',lambda **kw:seen.append('restore'))
    def train(**kw):
        assert kw['execution_budget']['stop_after_optimizer_steps']==256
        seen.append('native_train')
        raise runner.trainer._CandidateExecutionPaused({'reason':reason,'global_optimizer_steps':steps})
    monkeypatch.setattr(runner,'_run_bound_full_train_candidate',train)
    def measure(**kw):
        assert kw['optimizer_steps']==256 and kw['scope']['initial_measurement']['optimizer_steps']==0
        seen.append('final_online'); return {'path':'synthetic','sha256':'a'*64}
    monkeypatch.setattr(runner,'_run_prefix_initial_measurement',measure)
    receipt=tmp_path/'pause.json';receipt.write_text('{}')
    monkeypatch.setattr(runner.trainer,'_write_candidate_execution_pause_receipt',lambda *a,**kw:receipt)
    monkeypatch.setattr(runner,'_native_resume_state',lambda **kw:{'global_optimizer_steps':steps})
    result=runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb['path']),recipe_file_sha256=rb['sha256'],
        execution_budget_path=tmp_path/'budget.json',execution_budget_file_sha256='b'*64)
    assert seen==['guard','restore','native_train']+(['final_online'] if steps==256 else [])
    assert ('chronological_final_measurement' in result['pause']) == (steps==256)


@pytest.mark.parametrize('derived',[pytest.param(False,id='original'),pytest.param(True,id='derived')])
@pytest.mark.parametrize('fault',[None,'target','cohort'])
@pytest.mark.parametrize('train_only',[False,True])
def test_final_online_uses_identical_initial_targets_and_preserves_trained_session(tmp_path,monkeypatch,fault,train_only,derived):
    monkeypatch.setattr(trainer, "_copy_frozen_prefix_reference_model", lambda m, **kw: copy.deepcopy(m).eval().requires_grad_(False))
    if derived and not train_only: pytest.skip('Derived targets require the TRAIN-only scope.')
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    h=PrefixHarness(tmp_path/'run',_prepared(artifacts));output=h.root/'CANDIDATE'
    initial,pause=h.run_prefix(output,0)
    initial_hash=trainer._model_state_sha256(initial)
    scope_value={'initialization':{'online_model_state_sha256':initial_hash},
        'measurement':{'coordinate_result':{'test':True}},
        'artifacts':{'initialization_result':{'test':True},'measurement_binding_result':{'test':True},
                    'initial_measurement_audit':{'test':True}}}
    phase={'final':False}
    measured_roles=[]
    def cohort(*a,role):
        value={'role':role,'parent_entry_row_indices':list(range(256)),'entry_row_indices':list(range(256))}
        if phase['final'] and fault=='cohort':value['changed']=True
        return value
    monkeypatch.setattr(cohorts,'build_chronological_measurement_cohort',cohort)
    def measure(**kw):
        if phase['final'] and train_only:assert kw['evaluation_cohort']['role']=='train'
        measured_roles.append(kw['evaluation_cohort']['role'])
        assert not kw['model'].training and not kw['candidate_target_model'].training
        assert trainer._model_state_sha256(kw['candidate_target_model'])==initial_hash
        prediction=float(kw['model'].weight[0,0].detach())
        delta=1 if phase['final'] and fault=='target' else 0
        entry=[{'entry_row_index':i,'parent_entry_row_index':i,'predicted_q_bps':[prediction,prediction,0],
                'target_q_bps':[i/100+delta,-i/100,0],'target_valid':[True]*3} for i in range(256)]
        if phase['final'] and derived:
            for row in entry: row['target_q_bps'][1] -= 7
        anchor=[{'entry_row_index':i,'state_index':0,'prediction_hold_bps':[prediction]*2,
                 'target_hold_bps':[i/10,-i/10]} for i in range(256)]
        sampled=[{**row,'state_index':j} for row in anchor for j in range(4)]
        torch.rand(3)
        return None,{'bounded_entry_observations':entry,'bounded_exit_anchor_observations':anchor,
                     'bounded_exit_sampled_observations':sampled},None
    monkeypatch.setattr(runner.val,'_entry_representations',measure)
    components={**h.last_kwargs,'train_probe_ds':object()}
    before=runner._run_prefix_initial_measurement(components=components,scope=scope_value,
        recipe={'chronological_prefix':h.data.value},output=output,device=torch.device('cpu'),
        invocation_started=time.monotonic(),pause_evidence=pause)
    scope_value['artifacts']['initial_measurement_result']=before
    scope_value['initial_measurement']=json.loads(Path(before['path']).read_text())
    if derived:
        saved=json.loads(Path(scope_value['initial_measurement']['observations']['train']['path']).read_text())
        entry_rows=copy.deepcopy(saved['diagnostics']['bounded_entry_observations'])
        for row in entry_rows: row['target_q_bps'][1] -= 7
        scope_value['entry_baseline']={'entry_observations':entry_rows}
        scope_value['entry_baseline_result']={'path':'derived-result.json','sha256':'d'*64}
    trained,pause=h.run_prefix(output,256,expected_pointer=digest(h.pointer(output)))
    model_hash=trainer._model_state_sha256(trained); assert model_hash!=initial_hash
    checkpoint=h.state(output); pointer=h.pointer(output).read_bytes()
    rng=trainer._attended_session_rng_state(device=torch.device('cpu'));phase['final']=True
    components={**h.last_kwargs,'train_probe_ds':object()}
    kwargs=dict(components=components,scope=scope_value,recipe={'chronological_prefix':h.data.value},
        output=output,device=torch.device('cpu'),invocation_started=time.monotonic()-1800,
        pause_evidence=pause,optimizer_steps=256)
    if train_only:kwargs['recipe']['chronological_train_only_measurement']=True
    measured_roles.clear()
    if fault:
        with pytest.raises(RuntimeError,match='FROZEN_TARGET_CHANGED|COHORT_OR_TEACHER_CHANGED'):
            runner._run_prefix_initial_measurement(**kwargs)
    else:
        after=runner._run_prefix_initial_measurement(**kwargs);result=json.loads(Path(after['path']).read_text())
        assert result['optimizer_steps']==256 and result['selected_model_variant']=='ONLINE'
        assert measured_roles==(['train'] if train_only else ['train','control'])
        assert result['measurement_roles']==measured_roles and set(result['observations'])==set(measured_roles)
        assert result['frozen_targets_exactly_preserved'] is True and result['target_model_state_sha256']==initial_hash
        assert result['model_state_sha256']==model_hash and result['initial_measurement']==before
        if derived: assert result['derived_entry_target_baseline']==scope_value['entry_baseline_result']
    assert trained.training and h.pointer(output).read_bytes()==pointer
    equal_tree(h.state(output),checkpoint)
    equal_tree(trainer._attended_session_rng_state(device=torch.device('cpu')),rng)
    assert len(h.batches)==256 and h.validation_batches==0 and len(set(h.teacher_hashes))==1


def test_learning_accepts_train_only_initial_but_rejects_old_online_function(learning_scope):
    policy, recipe, _, seal = learning_scope
    ab = recipe['chronological_learning_measurement']
    audit = json.loads(Path(ab['path']).read_text())
    result = json.loads(Path(audit['result']['path']).read_text())
    result['observations'].pop('control')
    audit['result'] = _write(Path(audit['result']['path']), result)
    recipe['chronological_learning_measurement'] = _write(Path(ab['path']), audit)
    policy['chronological_learning_run']['chronological_learning_measurement'] = recipe['chronological_learning_measurement']
    seal()
    with pytest.raises(RuntimeError, match='INITIAL_MEASUREMENT_INVALID'):
        native.require_native_run_scope(recipe)
    recipe['chronological_train_only_measurement'] = True
    policy['chronological_learning_run']['chronological_train_only_measurement'] = True
    seal()
    assert native.require_native_run_scope(recipe, invocation_number=1) == 256
    source = Path(recipe['source_repo']) / 'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py'
    source.write_text('changed online function; initial tensor hash unchanged')
    with pytest.raises(RuntimeError, match='INITIAL_MODEL_SOURCE_CHANGED'):
        native.require_native_run_scope(recipe)


def test_learning_admission_checks_observation_source_before_frozen_cohorts(learning_scope, monkeypatch):
    # Reuse the existing admission fixture. A sentinel proves the check is
    # reached before any later cohort or model construction.
    _,recipe,_,_ = learning_scope
    calls = []
    def reject(value, *, measurement=None):
        assert value is recipe and measurement["optimizer_steps"] == 0
        calls.append(True)
        raise RuntimeError("injected source mismatch")
    monkeypatch.setattr(native,"require_prefix_measurement_source_binding",reject)
    with pytest.raises(RuntimeError,match="injected source mismatch"):
        native.require_chronological_learning_measurement(recipe)
    assert calls == [True]


@pytest.fixture
def dynamic_learning_measurement(dynamic_initial_scope, tmp_path, monkeypatch):
    """Admission dispatch unit; the real six-component scanner is tested separately."""
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    case = dynamic_initial_scope; recipe = case["recipe"]
    initial_artifacts = recipe.pop("chronological_initial_measurement")
    case["policy"].pop("chronological_initial_measurement")
    entry_binding = _write(tmp_path/"dynamic-entry-plan.json", {"unit":True})
    recipe["entry_observed_market"] = entry_binding
    scope = {"binding":entry_binding, "target_contract":case["witness"]["target_contract"],
        "economics":{"unit":True}, "training_phase":case["witness"]["training_phase"],
        "physical_sources":case["physical"]["physical_sources"],
        "query_minutes":tuple(case["witness"]["target_contract"]["query_contract"]["horizon_minutes"]),
        "target_m1_source_sha256":"b"*64}
    monkeypatch.setattr(horizons, "require_observed_entry_scope", lambda value: scope)
    identity = horizons.observed_entry_identity(scope)
    coordinates = case["measurement"]["coordinate_result"]
    cohorts_by_role, observations = {}, {}
    for role, split in (("train","train"), ("control","val")):
        source = case["physical"]["physical_sources"][split]
        cohort = {"plan":recipe["chronological_prefix"]["design"], "measurement_coordinates":coordinates,
            "measurement_role":role, "entry_row_indices":list(range(256)),
            "parent_entry_parquet":source["parquet"], "parent_entry_manifest":source["manifest"]}
        cohorts_by_role[role] = cohort
        observations[role] = {"role":role, "optimizer_steps":0, "model_state_sha256":"a"*64,
            "target_model_state_sha256":"a"*64, "test_data_used":False, "cohort":cohort,
            "entry_observed_market":identity, "dynamic_m5_loss_scales":case["loss_scale_identity"],
        "target_model_role":"frozen_constructor_anchor_no_forward",
            "exit_evaluation":False, "native_launch_authorized":False, "diagnostics":{"unit":True,
                "dynamic_curve_statistics":{"dynamic_m5_loss_scales":case["loss_scale_identity"],
                    "dynamic_loss_residual_units":"TRAIN_target_RMS_units"}}}
    monkeypatch.setattr(cohorts, "build_chronological_measurement_cohort", lambda *args,role: cohorts_by_role[role])
    calls = []
    def checked(observation, *, identity, cohort):
        assert observation["diagnostics"] == {"unit":True, "dynamic_curve_statistics":{
            "dynamic_m5_loss_scales":case["loss_scale_identity"], "dynamic_loss_residual_units":"TRAIN_target_RMS_units"}}
        assert observation["cohort"] == cohort
        assert observation["entry_observed_market"] == identity
        calls.append(observation["role"])
        return dict(observation)
    monkeypatch.setattr(native, "_require_dynamic_m5_observation", checked)
    result = {"schema_version":"gx1_native_prefix_initial_measurement_v1",
        "decision":"FROZEN_INITIAL_TARGETS_AND_PREDICTIONS_READY_NO_LEARNING_MEASURED", **initial_artifacts,
        "optimizer_steps":0, "model_state_sha256":"a"*64, "target_model_state_sha256":"a"*64,
        "teacher_refreshed":False, "economic_rollout":False, "test_data_used":False,
        "entry_observed_market":identity, "dynamic_m5_loss_scales":case["loss_scale_identity"],
        "target_model_role":"frozen_constructor_anchor_no_forward",
        "exit_evaluation":False, "native_launch_authorized":False, "model_functions":case["initial"]["model_functions"],
        **native.require_prefix_measurement_source_binding(recipe)}
    audit = {"schema_version":"gx1_native_prefix_initial_measurement_audit_v1",
        "decision":"FROZEN_INITIAL_MEASUREMENT_VERIFIED_NO_LEARNING_MEASURED", "optimizer_steps":0, "test_data_used":False,
        "fresh_model_optimizer_ema_scheduler_exactly_preserved":True, "saved_cpu_python_numpy_rng_exactly_preserved":True,
        "receipt":_write(tmp_path/"dynamic-native-terminal.json", {"guard_decision":"PASS", "outcome":"RESUMABLE",
            "trainer_guard_exit_code":0, "progress_observer_exit_code":0, "test_data_used":False})}
    def seal():
        result["observations"] = {role:_write(tmp_path/("dynamic-"+role+"-observation.json"), observation)
                                  for role,observation in observations.items()}
        audit["result"] = _write(tmp_path/"dynamic-initial-result.json", result)
        recipe["chronological_learning_measurement"] = _write(tmp_path/"dynamic-initial-audit.json", audit)
        case["seal"]()
    seal()
    return {**case, "result":result, "observations":observations, "audit":audit,
            "cohorts":cohorts_by_role, "observed_calls":calls, "seal_learning":seal}


def test_dynamic_learning_admission_requires_both_same_frozen_entry_cohorts(dynamic_learning_measurement):
    case = dynamic_learning_measurement
    value = native.require_chronological_learning_measurement(case["recipe"])
    assert set(value["initial_measurement"]["observations"]) == {"train","control"}
    assert case["observed_calls"] == ["train","control"]
    assert value["initialization"]["target_model_role"] == "frozen_constructor_anchor_no_forward"


@pytest.mark.parametrize("fault", ["anchor", "exit", "functions", "missing_control", "source", "cohort", "closure", "normalization"])
def test_dynamic_learning_admission_rejects_changed_initial_source_or_measurement(dynamic_learning_measurement, fault):
    case = dynamic_learning_measurement
    if fault == "anchor": case["result"]["target_model_role"] = "teacher"
    elif fault == "exit": case["result"]["exit_evaluation"] = True
    elif fault == "functions": case["result"]["model_functions"] = {"online":"other", "target":"other"}
    elif fault == "missing_control": case["observations"].pop("control")
    elif fault == "source": case["observations"]["train"]["cohort"] = {
        **case["cohorts"]["train"], "parent_entry_parquet":case["physical"]["physical_sources"]["val"]["parquet"]}
    elif fault == "cohort": case["observations"]["train"]["cohort"] = {
        **case["cohorts"]["train"], "entry_row_indices":list(range(1,257))}
    elif fault == "closure": case["result"]["native_recipe_source_bindings_sha256"] = "f"*64
    elif fault == "normalization":
        case["initial"]["base_contract_sha256"] = "f"*64
        case["result"]["initialization_result"] = _write(Path(case["initial_artifacts"]["initialization_result"]["path"]), case["initial"])
    case["seal_learning"]()
    with pytest.raises(RuntimeError): native.require_chronological_learning_measurement(case["recipe"])


@pytest.fixture
def dynamic_technical_scope(dynamic_learning_measurement, tmp_path):
    """Real immutable bindings; physical/model execution remains separately qualified."""
    import numpy as np
    from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
    case = dynamic_learning_measurement
    recipe, policy = case["recipe"], case["policy"]
    profile = compile_compact_entry_execution_profile()
    reference = {name: profile[name] for name in ("name", "contract_sha256")}
    recipe["entry_execution_profile"] = reference
    metadata = case["files"]["source_bundle_metadata"]
    recipe["files"]["source_bundle_metadata"] = _write(metadata, {"entry_execution_profile":reference})
    case["initial"]["files"] = copy.deepcopy(recipe["files"])
    initial_binding = _write(Path(case["initial_artifacts"]["initialization_result"]["path"]), case["initial"])
    case["result"]["initialization_result"] = initial_binding
    case["coordinates"]["epoch0_parent_order"] = np.load(case["witness"]["epoch0_parent_order"]["path"], allow_pickle=False)
    for cohort in case["cohorts"].values():
        cohort["cohort_sha256"] = native.canonical_sha256(cohort)
    case["seal_learning"]()
    audit_binding = recipe.pop("chronological_learning_measurement")
    policy.pop("chronological_learning_run", None)
    policy.update(training_enabled=True, full_epoch_training_allowed=False)
    plan = {**native.DYNAMIC_M5_TECHNICAL_FIXED_SCOPE,
        "entry_training_phase":case["witness"]["training_phase"],
        "chronological_prefix":recipe["chronological_prefix"], "initialization_result":initial_binding,
        "measurement_binding_result":case["result"]["measurement_binding_result"],
        "initial_measurement_result":case["audit"]["result"], "initial_measurement_audit":audit_binding,
        "entry_execution_profile":reference, "files":copy.deepcopy(recipe["files"]),
        "run_id":recipe["run_id"], "out_bundle_dir":recipe["out_bundle_dir"],
        "source_bindings_sha256":recipe["source_bindings_sha256"],
        "measurement_cohort_sha256":{role:value["cohort_sha256"] for role,value in case["cohorts"].items()},
        "native_train_parent_rows":case["coordinates"]["epoch0_parent_order"][:32].tolist()}
    def seal():
        binding = _write(tmp_path/"dynamic-technical-plan.json", plan)
        recipe["entry_dynamic_m5_technical_qualification"] = binding
        policy["entry_dynamic_m5_technical_qualification"] = {
            "plan":binding, **copy.deepcopy(plan), "entry_observed_market":recipe["entry_observed_market"],
            "dynamic_m5_loss_scales":recipe["dynamic_m5_loss_scales"]}
        return case["seal"]()
    seal()
    budget = {"stop_after_optimizer_steps":native.DYNAMIC_M5_TECHNICAL_CURSOR_CEILING,
        "expected_active_pointer_sha256":None, "stop_after_completed_val_epochs":None, "max_invocation_seconds":12000}
    return {**case, "technical_plan":plan, "seal_technical":seal, "technical_budget":budget}


def test_dynamic_technical_gate_binds_actual_initial_audit_four_updates_and_one_public_window(dynamic_technical_scope):
    case = dynamic_technical_scope
    checked = native.require_dynamic_m5_technical_qualification(case["recipe"], invocation_number=1,
        execution_budget=case["technical_budget"])
    assert checked["optimizer_step_ceiling"] == native.DYNAMIC_M5_TECHNICAL_CURSOR_CEILING == 2
    assert checked["plan"]["technical_optimizer_updates"] == 4
    assert checked["plan"]["control_optimizer_updates"] == 0
    assert checked["plan"]["parity_model_forwards"] == 4
    assert checked["artifacts"]["initial_measurement_result"] == case["audit"]["result"]
    assert case["observed_calls"] == ["train", "control"]
    assert native.require_native_run_scope(case["recipe"], invocation_number=1,
        execution_budget=case["technical_budget"]) == 2


@pytest.mark.parametrize("fault", ["training_false", "training_int", "full_epoch", "missing_full_epoch", "updates",
    "updates_bool", "batch", "control_update", "private_reuse", "unbound_policy", "cohort", "parent_rows",
    "profile", "file", "audit_bytes", "audit_status", "initial_result", "source_closure", "mixed_initial",
    "mixed_learning", "mixed_native_epoch"])
def test_dynamic_technical_gate_rejects_scope_authority_or_actual_initial_drift(dynamic_technical_scope, fault, tmp_path):
    case = dynamic_technical_scope
    policy, recipe, plan = case["policy"], case["recipe"], case["technical_plan"]
    if fault == "training_false": policy["training_enabled"] = False
    elif fault == "training_int": policy["training_enabled"] = 1
    elif fault == "full_epoch": policy["full_epoch_training_allowed"] = True
    elif fault == "missing_full_epoch": policy.pop("full_epoch_training_allowed")
    elif fault == "updates": plan["technical_optimizer_updates"] = 5
    elif fault == "updates_bool": plan["optimizer_steps"] = True
    elif fault == "batch": plan["batch_size"] = 8
    elif fault == "control_update": plan["control_optimizer_updates"] = 1
    elif fault == "private_reuse": plan["private_weights_reusable"] = True
    elif fault == "cohort": plan["measurement_cohort_sha256"]["train"] = "f"*64
    elif fault == "parent_rows": plan["native_train_parent_rows"] = list(reversed(plan["native_train_parent_rows"]))
    elif fault == "profile": plan["entry_execution_profile"] = None
    elif fault == "file": plan["files"]["entry_val_manifest"] = plan["files"]["entry_train_manifest"]
    elif fault == "audit_bytes": Path(plan["initial_measurement_audit"]["path"]).write_text("{}")
    elif fault == "audit_status":
        case["audit"]["decision"] = "UNREVIEWED"
        plan["initial_measurement_audit"] = _write(tmp_path/"changed-audit.json", case["audit"])
    elif fault == "initial_result": plan["initial_measurement_result"] = _write(tmp_path/"different-result.json", {})
    elif fault == "source_closure": plan["source_bindings_sha256"] = "f"*64
    elif fault == "mixed_initial": recipe["chronological_initial_measurement"] = {}
    elif fault == "mixed_learning": recipe["chronological_learning_measurement"] = {}
    elif fault == "mixed_native_epoch": policy["entry_m5_path_single_epoch_run"] = {}
    case["seal_technical"]()
    if fault == "unbound_policy":
        policy["entry_dynamic_m5_technical_qualification"]["technical_optimizer_updates"] = 5
        case["seal"]()
    with pytest.raises(RuntimeError):
        native.require_native_run_scope(recipe, invocation_number=1, execution_budget=case["technical_budget"])


@pytest.mark.parametrize("field,value", [("stop_after_optimizer_steps",0), ("stop_after_optimizer_steps",1),
    ("stop_after_optimizer_steps",3), ("stop_after_optimizer_steps",True), ("expected_active_pointer_sha256","a"*64),
    ("stop_after_completed_val_epochs",1), ("max_invocation_seconds",11999), ("resume_probe_val_rows",32)])
def test_dynamic_technical_gate_rejects_other_budget_or_resume_pointer(dynamic_technical_scope, field, value):
    case = dynamic_technical_scope
    with pytest.raises(RuntimeError, match="TECHNICAL_BUDGET"):
        native.require_native_run_scope(case["recipe"], execution_budget={**case["technical_budget"], field:value})
    for invalid in (0, 2, True):
        with pytest.raises(RuntimeError, match="TECHNICAL_INVOCATION"):
            native.require_native_run_scope(case["recipe"], invocation_number=invalid)


def test_dynamic_technical_materializer_uses_real_scope_and_one_ceiling2_window(dynamic_technical_scope, monkeypatch, tmp_path):
    from tests.test_native_prefix_recipe import materializer, _boot
    case = dynamic_technical_scope
    recipe = case["recipe"]; rb = case["seal_technical"](); repo = Path(recipe["source_repo"])
    monkeypatch.setattr(materializer, "_source_commit", lambda _:recipe["source_commit"])
    guards, controllers = {}, {}
    for names, target in ((("runner","guard","query","certificate"), guards),
                          (("controller","observer","campaign_cli"), controllers)):
        for name in names:
            path=repo/"sources"/name; path.parent.mkdir(exist_ok=True); path.write_text(name); target[name]=_bind(path)
    (repo/"scripts").mkdir(); (repo/"scripts/gx1_capped_run.sh").write_text("synthetic")
    (repo/".venv/bin").mkdir(parents=True); (repo/".venv/bin/python").write_text("synthetic")
    monkeypatch.setattr(materializer, "_sources", lambda *a:(guards,controllers))
    boot=tmp_path/"technical-boot.json"; _write(boot,_boot(100,0))
    def materialize(output, windows):
        return materializer.materialize_native_candidate_campaign(repo=repo, output=output,
            runtime=tmp_path/(output.name+"-runtime"), gpu_uuid="GPU-fixture", prepared_boot_path=boot,
            prepared_boot_file_sha256=native.file_sha256(boot), certificate_path=Path(guards["certificate"]["path"]),
            recipe_path=Path(rb["path"]), recipe_file_sha256=rb["sha256"], window_count=windows)
    result=materialize(tmp_path/"technical-campaign",1)
    assert result["plan"]["entry_pairs_per_epoch"] == case["witness"]["train_rows"]
    assert len(result["plan"]["checked_invocations"]) == 1
    with pytest.raises(RuntimeError, match="TECHNICAL_INVOCATION"):
        materialize(tmp_path/"too-many-windows",2)
    assert not (tmp_path/"too-many-windows").exists()
    recipe["chronological_initial_measurement"] = {}; rb=case["seal_technical"]()
    with pytest.raises(RuntimeError, match="TECHNICAL_MIXED_SCOPE"):
        materialize(tmp_path/"mixed-window",1)
    assert not (tmp_path/"mixed-window").exists()


def _dynamic_technical_parity_case(dynamic_initial_observer):
    model, dataset, _, _ = dynamic_initial_observer
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,num_workers=0)
    batch=runner.trainer._materialize_dynamic_m5_batch_targets(next(iter(loader)),dataset)
    original=model.forward
    def forward(*args, **kwargs):
        out=original(*args,**kwargs)
        quality=out["dynamic_terminal_net_markout_bps"][..., :2]-out["dynamic_pre_mfe_mae_upper_bps"]
        out["dynamic_entry_quality_bps_candidate"]=torch.cat((quality,torch.zeros_like(quality[..., :1])),dim=-1)
        return out
    model.forward=forward
    return model,batch


def test_dynamic_curve_parity_checks_complete_queries_masks_and_preserves_online_rng(dynamic_initial_observer):
    model,batch=_dynamic_technical_parity_case(dynamic_initial_observer)
    before=runner.trainer._model_state_sha256(model)
    rng=runner._dynamic_m5_technical_tree_sha256(runner.trainer._attended_session_rng_state(device=torch.device("cpu")))
    result=runner._dynamic_m5_forward_parity(model=model,batch=batch,device=torch.device("cpu"))
    assert result["model_forwards"] == model.forwards == 2
    assert result["optimizer_steps"] == 0 and result["query_and_best_horizon_masks_exact"] is True
    assert result["output_comparisons"]["dynamic_terminal_net_markout_bps"]["shape"] == [16,480,3]
    assert all(row["max_absolute_difference_bps"] <= row["absolute_tolerance_bps"] for row in result["output_comparisons"].values())
    manifest=result["typed_batch_manifest"]
    assert manifest["inputs"]["seq_x"]["shape"][0] == 16
    assert manifest["masks"]["dynamic_path_observed"]["dtype"] == "torch.bool"
    assert manifest["pointers"]["dynamic_entry_parent_row_index"]["dtype"] == "torch.int64"
    assert manifest["targets"]["y_dynamic_mfe_bps"]["shape"] == [16,480,2]
    assert runner.trainer._model_state_sha256(model) == before
    assert runner._dynamic_m5_technical_tree_sha256(runner.trainer._attended_session_rng_state(device=torch.device("cpu"))) == rng
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("fault", ["curve", "forecast", "query", "unknown_query", "horizon_mask", "candidate_tie", "nonfinite", "shape"])
def test_dynamic_curve_parity_rejects_autograd_path_drift(dynamic_initial_observer, fault):
    model,batch=_dynamic_technical_parity_case(dynamic_initial_observer)
    original=model.forward
    def forward(*args,**kwargs):
        out=original(*args,**kwargs)
        if fault == "unknown_query":
            out["dynamic_query_horizon_minutes"]=out["dynamic_query_horizon_minutes"]+1
        if torch.is_grad_enabled():
            if fault == "curve": out["dynamic_mfe_bps"]=out["dynamic_mfe_bps"]+.01
            elif fault == "forecast": out["forecast_pred"]=out["forecast_pred"]+.01
            elif fault == "nonfinite": out["forecast_pred"]=out["forecast_pred"]*float("nan")
            elif fault == "shape": out["dynamic_mfe_bps"]=out["dynamic_mfe_bps"][:,:-1]
            elif fault == "query": out["dynamic_query_horizon_minutes"]=out["dynamic_query_horizon_minutes"]+1
            elif fault == "horizon_mask": out["dynamic_entry_quality_best_horizon_mask"]=~out["dynamic_entry_quality_best_horizon_mask"]
            elif fault == "candidate_tie":
                out["dynamic_entry_action_quality_bps_candidate"]=out["dynamic_entry_action_quality_bps_candidate"].clone()
                out["dynamic_entry_action_quality_bps_candidate"][:,0]=1.
        return out
    model.forward=forward
    with pytest.raises(RuntimeError):
        runner._dynamic_m5_forward_parity(model=model,batch=batch,device=torch.device("cpu"))
    assert all(parameter.grad is None for parameter in model.parameters())



def test_dynamic_curve_parity_rejects_changed_exact_action_ties_inside_numeric_allowance(dynamic_initial_observer):
    model,batch=_dynamic_technical_parity_case(dynamic_initial_observer)
    original=model.forward
    def forward(*args,**kwargs):
        out=original(*args,**kwargs)
        terminal=out["dynamic_terminal_net_markout_bps"].clone()
        terminal[..., :2]=out["dynamic_pre_mfe_mae_upper_bps"]
        if torch.is_grad_enabled(): terminal[...,0] += 1e-5
        out["dynamic_terminal_net_markout_bps"]=terminal
        quality=terminal[...,:2]-out["dynamic_pre_mfe_mae_upper_bps"]
        out["dynamic_entry_quality_bps_candidate"]=torch.cat((quality,torch.zeros_like(quality[...,:1])),dim=-1)
        best=quality.amax(dim=1)
        out["dynamic_entry_action_quality_bps_candidate"]=torch.cat((best,torch.zeros_like(best[:,:1])),dim=-1)
        out["dynamic_entry_quality_best_horizon_mask"]=quality==best[:,None,:]
        return out
    model.forward=forward
    with pytest.raises(RuntimeError,match="ACTION_OR_TIE_CHANGED"):
        runner._dynamic_m5_forward_parity(model=model,batch=batch,device=torch.device("cpu"))


def test_dynamic_technical_guarded_dispatch_uses_exact_scope_before_qualification(dynamic_technical_scope, monkeypatch, tmp_path):
    case=dynamic_technical_scope;recipe=case["recipe"];rb=case["seal_technical"]();seen=[]
    monkeypatch.setattr(runner.trainer,"_require_cuda_trainer_guard_execution",lambda **kw:seen.append("guard"))
    monkeypatch.setattr(runner.trainer,"_resolve_device",lambda _:torch.device("cuda"))
    monkeypatch.setattr(runner,"_require_native_full_train_recipe",lambda *a:(recipe,case["files"],{"unit":True}))
    monkeypatch.setattr(runner.launch_owner,"require_candidate_execution_budget",lambda *a,**kw:case["technical_budget"])
    components={"same_bound_component":True}
    monkeypatch.setattr(runner,"_build_bound_full_train_components",lambda **kw:components)
    def technical(**kwargs):
        assert kwargs["components"] is components
        assert kwargs["scope"]["artifacts"]["initial_measurement_result"] == case["audit"]["result"]
        assert kwargs["scope"]["plan"]["technical_optimizer_updates"] == 4
        seen.append("technical")
        return {"decision":"PRIVATE_TECHNICAL_UNIT"}
    monkeypatch.setattr(runner,"_run_dynamic_m5_technical_qualification",technical)
    result=runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb["path"]),recipe_file_sha256=rb["sha256"],
        execution_budget_path=tmp_path/"unused-budget.json",execution_budget_file_sha256="a"*64)
    assert result == {"decision":"PRIVATE_TECHNICAL_UNIT"} and seen == ["guard","technical"]
    case["policy"]["training_enabled"]=False;case["seal"]();seen.clear()
    with pytest.raises(RuntimeError,match="TECHNICAL_NOT_AUTHORIZED"):
        runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb["path"]),recipe_file_sha256=rb["sha256"],
            execution_budget_path=tmp_path/"unused-budget.json",execution_budget_file_sha256="a"*64)
    assert seen == ["guard"]

def test_dynamic_curve_parity_retains_actual_terminal_drift_and_restores_state(dynamic_initial_observer):
    model, batch = _dynamic_technical_parity_case(dynamic_initial_observer)
    device = torch.device("cpu")
    before = runner.trainer._model_state_sha256(model)
    batch_before = runner._dynamic_m5_technical_tree_sha256(batch)
    rng_before = runner._dynamic_m5_technical_tree_sha256(runner.trainer._attended_session_rng_state(device=device))
    modes_before = [module.training for module in model.modules()]
    original = model.forward
    def drifting_forward(*args, **kwargs):
        out = original(*args, **kwargs)
        if torch.is_grad_enabled():
            terminal = out["dynamic_terminal_net_markout_bps"].clone()
            terminal[..., 0] += .01
            quality = terminal[..., :2] - out["dynamic_pre_mfe_mae_upper_bps"]
            best = quality.amax(dim=1)
            out.update(dynamic_terminal_net_markout_bps=terminal,
                dynamic_entry_quality_bps_candidate=torch.cat((quality, torch.zeros_like(quality[..., :1])), dim=-1),
                dynamic_entry_action_quality_bps_candidate=torch.cat((best, torch.zeros_like(best[:, :1])), dim=-1),
                dynamic_entry_quality_best_horizon_mask=quality == best[:, None, :])
        return out
    model.forward = drifting_forward
    diagnostics = {}
    with pytest.raises(RuntimeError, match="PARITY_VALUES_CHANGED:dynamic_terminal_net_markout_bps"):
        runner._dynamic_m5_forward_parity(model=model, batch=batch, device=device, diagnostics=diagnostics)
    assert model.forwards == diagnostics["model_forwards"] == 2
    assert diagnostics["optimizer_steps"] == 0
    comparison = diagnostics["output_comparisons"]["dynamic_terminal_net_markout_bps"]
    assert comparison["shape"] == [16, 480, 3] and comparison["dtype"] == "torch.float32"
    assert comparison["cells_above_absolute_tolerance"] == 16 * 480
    assert comparison["absolute_tolerance_bps"] == 1e-4 and comparison["relative_tolerance"] == 0.0
    assert comparison["max_absolute_difference_bps"] == pytest.approx(.01, abs=1e-6)
    assert comparison["rms_difference_bps"] == pytest.approx(.01 / (3 ** .5), abs=1e-6)
    assert comparison["max_absolute_difference_index"] == [0, 0, 0]
    assert comparison["left_at_max_absolute_difference_bps"] == pytest.approx(-2.0)
    assert comparison["right_at_max_absolute_difference_bps"] == pytest.approx(-1.99)
    assert comparison["left_max_absolute_output_bps"] == pytest.approx(2.0)
    assert comparison["right_max_absolute_output_bps"] == pytest.approx(2.0)
    assert [item["context"] for item in diagnostics["execution_passes"]] == ["no_grad", "enable_grad"]
    assert [item["grad_enabled"] for item in diagnostics["execution_passes"]] == [False, True]
    assert all(item["model_training"] is False and item["training_module_count"] == 0
        for item in diagnostics["execution_passes"])
    assert diagnostics["model_input_and_gradient_state_preserved"] is True
    assert diagnostics["rng_restored_exactly"] is True and diagnostics["module_modes_restored_exactly"] is True
    assert runner.trainer._model_state_sha256(model) == before
    assert runner._dynamic_m5_technical_tree_sha256(batch) == batch_before
    assert runner._dynamic_m5_technical_tree_sha256(runner.trainer._attended_session_rng_state(device=device)) == rng_before
    assert [module.training for module in model.modules()] == modes_before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_dynamic_curve_parity_records_real_cpu_attention_dispatch_on_existing_forwards(dynamic_initial_observer):
    model, batch = _dynamic_technical_parity_case(dynamic_initial_observer)
    original = model.forward
    def attention_forward(*args, **kwargs):
        # Exercise the real profiler/ATen path without hooks, backend changes or RNG draws.
        q = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
        torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return original(*args, **kwargs)
    model.forward = attention_forward
    diagnostics = {}
    result = runner._dynamic_m5_forward_parity(model=model, batch=batch, device=torch.device("cpu"),
        diagnostics=diagnostics)
    assert result["model_forwards"] == diagnostics["model_forwards"] == model.forwards == 2
    assert result["execution_passes"] == diagnostics["execution_passes"]
    for execution in result["execution_passes"]:
        calls = execution["host_attention_operator_calls"]
        assert calls["aten::scaled_dot_product_attention"] == 1
        assert all(name.startswith("aten::") and type(count) is int and count > 0
            for name, count in calls.items())
        assert execution["device"] == "cpu" and execution["inference_mode_enabled"] is False
        assert execution["model_training"] is False and execution["autocast_enabled"] is False
        assert type(execution["mha_fastpath_enabled"]) is bool
        assert type(execution["deterministic_algorithms_enabled"]) is bool
        assert isinstance(execution["float32_matmul_precision"], str)
    assert diagnostics["rng_restored_exactly"] is True
    assert diagnostics["model_input_and_gradient_state_preserved"] is True


def test_dynamic_technical_caller_publishes_role_failure_and_reraises_original(dynamic_initial_observer, monkeypatch, tmp_path):
    from types import SimpleNamespace
    model, dataset, cohort, _ = dynamic_initial_observer
    cohort = {**cohort, "cohort_sha256": "unit completed cohort"}
    scope = {"plan_binding": {"unit": "completed admission"}, "cohorts": {"train": cohort},
        "plan": {"measurement_cohort_sha256": {"train": cohort["cohort_sha256"]}}}
    # Isolate already-qualified scope/optimizer/restore boundaries. Exercise the actual
    # single caller, immutable publisher and rethrow; this is not native admission.
    monkeypatch.setattr(native, "require_dynamic_m5_technical_qualification", lambda *a, **kw: scope)
    optimizer = SimpleNamespace(state_dict=lambda: {})
    ema = SimpleNamespace(checkpoint_state=lambda: {"steps": 0})
    monkeypatch.setattr(runner, "_component_optimizer_state", lambda **kw: (optimizer, ema, None, {}))
    monkeypatch.setattr(runner, "_restore_prefix_initial_measurement_state", lambda **kw: None)
    monkeypatch.setattr(runner, "_require_dynamic_m5_measurement_source", lambda **kw: kw["cohort"])
    actual_rng = runner.trainer._attended_session_rng_state
    monkeypatch.setattr(runner.trainer, "_attended_session_rng_state",
        lambda **kw: actual_rng(device=torch.device("cpu")))
    failure = RuntimeError("DYNAMIC_M5_TECHNICAL_PARITY_VALUES_CHANGED:dynamic_terminal_net_markout_bps")
    observed_diagnostics = {"model_forwards": 2, "optimizer_steps": 0,
        "output_comparisons": {"dynamic_terminal_net_markout_bps": {"max_absolute_difference_bps": .01}},
        "rng_restored_exactly": True, "model_input_and_gradient_state_preserved": True}
    def fail_parity(*, model, batch, device, diagnostics):
        diagnostics.update(observed_diagnostics)
        raise failure
    monkeypatch.setattr(runner, "_dynamic_m5_forward_parity", fail_parity)
    def forbidden_update(**kwargs):
        raise AssertionError("failure observation reached optimizer coordinator")
    monkeypatch.setattr(runner, "_run_bound_full_train_candidate", forbidden_update)
    identity = dataset._entry_observed_market_binding["identity"]
    components = {"model": model, "effective_train_rows": 32, "train_probe_ds": dataset,
        "entry_observed_market": identity, "native_val_context": {}}
    with pytest.raises(RuntimeError) as caught:
        runner._run_dynamic_m5_technical_qualification(components=components, scope=scope,
            recipe={"trainer_cli": {"learning_rate": .0001, "weight_decay": .0001}}, files={},
            output=tmp_path / "BUNDLE", device=torch.device("cuda"), provenance={},
            execution_budget={"max_invocation_seconds": 12000}, recipe_file_sha256="a" * 64,
            invocation_started=time.monotonic())
    assert caught.value is failure
    path = tmp_path / "DYNAMIC_M5_TECHNICAL_QUALIFICATION" / "PARITY_FAILURE.json"
    report = json.loads(path.read_text())
    assert report["schema_version"] == "gx1_dynamic_m5_technical_parity_failure_v1"
    assert report["error"] == str(failure) and report["failed_role"] == "train"
    assert report["cohort"]["parent_entry_rows"] == cohort["parent_entry_row_indices"][:16]
    assert report["cohort"]["dynamic_entry_m1_rows"] == [row * 5 for row in cohort["parent_entry_row_indices"][:16]]
    assert report["diagnostics"] == observed_diagnostics and report["previous_completed_roles"] == {}
    assert report["optimizer_updates"] == 0 and report["native_training_admitted"] is False
    assert report["test_data_used"] is False
    assert not (path.parent / "RESULT.json").exists()
