"""Passive current-endpoint loss/Jacobian/retained-Adam observation; no fit."""
from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import cache_market_state_ta_direction_v1 as cache
from gx1.audit import observe_market_state_ta_c171_short_heads_v1 as previous
from gx1.audit import ta_current_update_direction_v1 as gradient
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_full_prefix_learning_v1 as full

REPO = cache.REPO
STAGE = 'market_state_ta_current_update_observation_20261113_003'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
WORKER = REPO / 'gx1/audit/observe_market_state_ta_current_updates_v1.py'
NATIVE_ROOT = full.ROOT
PARAMETER_ROOT = ROOT.parent / 'MARKET_STATE_TA_FULL_PREFIX_PARAMETER_REPLAY_20261113_002'
NATIVE_RESULT = dict(path=str(NATIVE_ROOT/'RESULT.json'), sha256='cc5d1ba16cf8823cce63794e6210061f714c5106f04a5b6ab5a7906494cad172')
PARAMETER_RESULT = dict(path=str(PARAMETER_ROOT/'RESULT.json'), sha256='3545cbf2342a0b977cb1860954a15942b0b7617e4cedc7cb435c568798508e6a')
PHASES = ('INITIAL', 'FINAL')
MODELS = dict(INITIAL='bef35dafc43f0b3450cca161049e9daacee55d92d7f1c3187bd82c9c4de972fc',
              FINAL='1542ed994b1d96f8c77481a8263df39ba52aa9d48720773357ecf96cd3ecd2c4')
PANEL_SLOTS = list(range(0, 256, 17))
BATCHES = [0,123,246,369,492,615,738,861,985,1108,1231,1354,1477,1600,1723,1847]
GROUP_COUNTS = dict(query_projection=2, family_axis_attn=12, timeframe_axis_attn=12,
    cross_tf_attn=12, m5_side_heads=4, shared_side_heads=4, tf_activity_heads=20, remaining_encoder=524)
NUMERIC_ROOT = ROOT.parent/'MARKET_STATE_TA_CURRENT_NUMERIC_ROUTES_20261113_001'
NUMERIC_RESULT = dict(path=str(NUMERIC_ROOT/'RESULT.json'),sha256='1c1a92d65e673dd5b91e3c9fda79eb22ed34bd8bbe7f69067181ef05dcb2d9ac')
NUMERIC_VALIDATION = dict(path=str(NUMERIC_ROOT/'REVIEW_VALIDATION.json'),sha256='c0936a6bc5a45b73ee8b18efbacaa941cd7bbdcc252a31eacf031f85a2720015')
NUMERIC_WORKER = 'gx1/audit/observe_market_state_ta_current_numeric_routes_v1.py'
ROUTES = ('NATIVE_NO_GRAD','DYNAMIC_NO_GRAD','DYNAMIC_GRAD')
HELPERS = ('gx1/audit/ta_current_update_direction_v1.py', 'gx1/audit/ta_direction_mixing_gradient_v1.py',
    NUMERIC_WORKER, '.venv/lib/python3.10/site-packages/torch/backends/mha/__init__.py',
    '.venv/lib/python3.10/site-packages/torch/nn/modules/transformer.py')
BOUNDS = dict(batch_size=16, panel_slots=PANEL_SLOTS, original_batch_indices=BATCHES,
    sampled_rows=512, unique_rows=256, model_constructions=1, checkpoint_loads=2,
    archive_decodes=2, input_sidecar_archive_decodes=6, checkpoint_weight_transitions=1,
    forwards=96, grad_forwards=32, autograd_grad_calls=256, matched_batches=64,
    native_reference_matches=32, dynamic_reference_matches=32,
    optimizer_constructions=0, optimizer_updates=0, backwards=0, rng_restores=0,
    checkpoint_writes=0, candidate_writes=0, wall_seconds=900, memory_bytes=12*1024**3,
    max_vector_bytes=2*1024**3, max_published_json_bytes=32*1024**2,
    numerical_threads=8, parameter_group_counts=GROUP_COUNTS,
    raw_rtol=1e-6, raw_atol=2e-6, loss_atol=5e-6, training_enabled=False, test_data_used=False)


def _need(ok, reason):
    if not bool(ok): raise RuntimeError('TA_CURRENT_UPDATE_OBSERVATION_' + reason)


def _leaves(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        _need(set(a) == set(b), 'VIEW_FIELDS')
        return {k:v for name in a for k,v in _leaves(a[name], b[name], prefix+'.'+name if prefix else name).items()}
    return {} if a == b else {prefix:dict(INITIAL=a, FINAL=b)}


def _view_identity(view):
    return dict(contract_sha256=view['contract_sha256'], prefix_normalization=view['prefix_normalization'],
                research_objective_sha256=view['research_objective']['contract_sha256'])


def _metadata():
    native, replay = cache._read(NATIVE_RESULT), cache._read(PARAMETER_RESULT)
    recipe = cache._read(native['recipe']); provenance = cache._read(replay['plan'])
    _need(native['outcome'] == 'COMPLETE' and native['optimizer_updates'] == 3698
        and native['skipped_batches'] == 0 and native['source_unchanged'] is True
        and native['test_data_used'] is False and replay['native_result'] == NATIVE_RESULT
        and replay['actual_model_digests'] == MODELS and replay['source_unchanged'] is True, 'COMPLETED_ENDPOINTS')
    _need(recipe['numerical_recipe']['contract_sha256'] == gradient.NUMERICAL_SHA256, 'NUMERICS')
    owners = dict(recipe['source_owners'])
    # The original recipe is immutable. This metadata-only origin context changes
    # only the allowed orchestration cap reference, never model/data/loss bytes.
    owners[cache.CAP_RUNNER] = cache._ref(REPO/cache.CAP_RUNNER)
    origin_context = copy.deepcopy(recipe); origin_context['source_owners'] = owners
    origin = full.read_origin(origin_context)
    views = {phase:cache._read(ref) for phase,ref in provenance['feature_views'].items()}
    _need(_leaves(views['INITIAL'], views['FINAL']) == provenance['expected_feature_view_differences']
          and views['FINAL'] == origin['view'], 'QUALIFIED_VIEW_PROVENANCE')
    for name in ('constructor_metadata', 'prefix_normalization'):
        refs = [views[phase][name]['binding'] for phase in PHASES]
        _need(refs[0]['sha256'] == refs[1]['sha256'], 'IDENTICAL_NORMALIZATION')
        for ref in refs: campaign.require_binding(ref, label=name, verify_file=True)
    measurements = {}; inputs = dict(native_result=NATIVE_RESULT, parameter_result=PARAMETER_RESULT,
        provenance_plan=replay['plan'], native_recipe=native['recipe'])
    for phase, label, updates in [('INITIAL','INITIAL_TRAIN',0),('FINAL','FULL_PREFIX_PASS2_TRAIN',3698)]:
        wrapper = cache._read(native['measurement_refs'][label]); m = wrapper['measurement']
        _need(wrapper['evaluated_model_state_sha256'] == MODELS[phase]
            and wrapper['evaluation_state_unchanged'] is True and wrapper['test_data_used'] is False
            and wrapper['optimizer_updates'] == updates and wrapper['capture'] == native['panel_captures'][label]
            and [wrapper['panel_batch_indices'][i] for i in PANEL_SLOTS] == BATCHES
            and (m['rows'], m['batches'], m['fold_id'], m['pass_index']) == (4096,256,1,0)
            and m['seed'] == recipe['numerical_recipe']['seed']
            and m['feature_view_sha256'] == recipe['expected_view_sha256']
            and m['model_contract_sha256'] == recipe['expected_model_contract_sha256']
            and m['loss_contract_sha256'] == recipe['expected_loss_contract_sha256']
            and len(m['canonical_batch_losses']) == 256
            and all(type(v) is float and math.isfinite(v) for v in m['canonical_batch_losses']), 'MEASUREMENT:'+phase)
        measurements[phase] = wrapper
        inputs.update({phase+'_measurement':native['measurement_refs'][label], phase+'_capture':wrapper['capture'],
                       phase+'_checkpoint':provenance['states'][phase], phase+'_view':provenance['feature_views'][phase]})
        _need(0 < Path(provenance['states'][phase]['path']).stat().st_size == provenance['state_bytes'][phase] < 128*1024**2, 'STATE_SIZE')
    _need(native['initial_model_state_sha256'] == MODELS['INITIAL']
        and native['arms']['FULL_PREFIX']['evidence_capsule'] == inputs['FINAL_checkpoint']
        and native['arms']['FULL_PREFIX']['final_model_state_sha256'] == MODELS['FINAL'], 'SAVED_ENDPOINT_REFS')
    numeric,validation = cache._read(NUMERIC_RESULT),cache._read(NUMERIC_VALIDATION)
    numeric_plan = cache._read(numeric['plan'])
    _need(numeric['status'] == 'COMPLETE' and numeric['source_unchanged'] is True
        and numeric['model_state_sha256'] == MODELS['INITIAL']
        and numeric['native_retained_control_qualified'] is numeric['dynamic_grad_vs_no_grad_qualified'] is True
        and numeric['cross_route_equivalence_qualified'] is False
        and validation['result'] == NUMERIC_RESULT and validation['native_no_grad_vs_retained_bitexact'] is True
        and validation['dynamic_no_grad_vs_grad_bitexact'] is True, 'QUALIFIED_NUMERIC_PROFILE_REFERENCES')
    for ref in [numeric_plan['worker'], *(numeric_plan['helpers'][name] for name in HELPERS[-2:])]:
        campaign.require_binding(ref,label='unchanged numeric comparator/torch route',verify_file=True)
    inputs.update(numeric_profile_result=NUMERIC_RESULT,numeric_profile_validation=NUMERIC_VALIDATION,
                  numeric_profile_plan=numeric['plan'])
    for ref in inputs.values(): campaign.require_binding(ref, label='current observation input', verify_file=True)
    for ref in owners.values(): campaign.require_binding(ref, label='current observation owner', verify_file=True)
    return dict(origin, recipe=recipe, native=native, measurements=measurements, views=views,
                inputs=inputs, source_owners=owners)


def build_plan(source_base_commit):
    q = _metadata()
    return dict(schema_version='gx1_ta_current_update_observation_plan_v1', stage=STAGE, source_repo=str(REPO),
        source_base_commit=source_base_commit, worker=cache._ref(WORKER), inputs=q['inputs'],
        source_owners=q['source_owners'], helpers={name:cache._ref(REPO/name) for name in HELPERS},
        model_digests=MODELS, feature_view_sha256=q['view']['contract_sha256'],
        numerical_recipe_sha256=gradient.NUMERICAL_SHA256, bounds=BOUNDS,
        aggregation='Mean of 16 intact outcome-blind B16 batches and geometry of their mean vectors are reported separately.',
        interpretation='Endpoint local derivatives and nominal last Adam gradient component; no finite historical causal attribution or next-step prediction.')


def gate_fields(plan_ref):
    return dict(status='ADMITTED', plan=plan_ref, max_invocations=1, remaining_invocations=1,
        claimed_invocations=0, completed_invocations=0, relaunch_allowed=False, execution_authorized=True,
        data_reads_authorized=True, checkpoint_reads_authorized=True, training_authorized=False, bounds=BOUNDS)


def _inspect(plan_path, plan_sha, out_dir, *, execute):
    _need(Path(plan_path) == ROOT/'PLAN.json' and Path(out_dir) == ROOT, 'EXACT_PATHS')
    ref = dict(path=str(plan_path), sha256=plan_sha); plan = cache._read(ref)
    _need(not any((ROOT/name).exists() for name in ('CLAIM.json','RESULT.json','TERMINAL.json')), 'CONSUMED')
    _need(plan == build_plan(plan['source_base_commit']), 'CURRENT_PLAN_BINDINGS')
    head = cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    q = _metadata(); policy_ref = cache._ref(REPO/'NEXT_RUN_POLICY.json'); policy = cache._read(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled','native_launch_authorized',
        'full_epoch_training_allowed','full_val_allowed')), 'NO_TRAINING')
    if execute:
        gate = policy.get(STAGE,{})
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in gate_fields(ref).items()), 'ADMISSION')
        _need(cache._git('branch','--show-current') == 'work/gx1-current', 'BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO), source_commit=head))
    refs = [ref,policy_ref,plan['worker'],*plan['helpers'].values(),*plan['inputs'].values(),*plan['source_owners'].values()]
    return dict(q,plan_ref=ref,plan=plan,head=head,bound_refs=refs)


def preflight(plan_path, plan_sha, out_dir):
    q = _inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_CURRENT_UPDATE_PREFLIGHT', stage=STAGE, plan=q['plan_ref'],
        model_digests=MODELS, feature_view_sha256=q['view']['contract_sha256'], claims=0,
        model_constructions=0, checkpoint_loads=0, array_decodes=0, training_enabled=False, test_data_used=False)


def _groups(model):
    groups = {name:[] for name in GROUP_COUNTS}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        matches = []
        if name.startswith('state_model.query_projection.'): matches.append('query_projection')
        for block in ('family_axis_attn','timeframe_axis_attn','cross_tf_attn'):
            if name.startswith('state_model.backbone.'+block+'.'): matches.append(block)
        if name.startswith(('state_model.m5_side_head.','future_m5_side_head.')): matches.append('m5_side_heads')
        if name.startswith(('state_model.query_head.','future_head.')): matches.append('shared_side_heads')
        if name.startswith(('state_model.tf_activity_heads.','future_tf_activity_heads.')): matches.append('tf_activity_heads')
        if not matches:
            _need(name.startswith('state_model.backbone.'), 'UNCLASSIFIED_PARAMETER:'+name)
            matches.append('remaining_encoder')
        _need(len(matches) == 1, 'OVERLAPPING_GROUP'); groups[matches[0]].append((name,p))
    _need({k:len(v) for k,v in groups.items()} == GROUP_COUNTS, 'ALL_590_PARAMETER_GROUPS')
    _need(sum(p.numel() for entries in groups.values() for _,p in entries)*8*16 <= BOUNDS['max_vector_bytes'], 'VECTOR_MEMORY')
    return groups


def _require_parent_order(order, states, *, torch):
    # Full-prefix RECIPE has no parent_order_sha256; the actual saved endpoints
    # and the qualified prepared axes carry the order. Compare their values.
    expected = (states['INITIAL']['parent_order'], states['FINAL']['selected_parent_order'])
    _need(isinstance(order,torch.Tensor) and order.ndim == 1 and order.numel() > 0
        and order.dtype == torch.int64 and order.device.type == 'cpu', 'PARENT_ORDER_DOMAIN')
    _need(all(isinstance(value,torch.Tensor) and value.dtype == order.dtype
        and value.device == order.device and value.shape == order.shape and torch.equal(value,order)
        for value in expected), 'ACTUAL_PAIRED_PARENT_ORDER')


def _load_endpoint(model, saved, phase, q, order, *, torch, old, state, model_sha):
    recipe = q['recipe']; final = phase == 'FINAL'; binding = saved['parameter_binding' if final else 'binding']
    _need(saved['model_state_sha256'] == model_sha(saved['model_state']) == MODELS[phase]
        and saved['buffer_state_sha256'] == old._tree_sha(saved['buffer_state'])
        and binding['feature_view_identity'] == _view_identity(q['views'][phase])
        and binding['numerical_recipe'] == recipe['numerical_recipe']
        and torch.equal(saved['selected_parent_order' if final else 'parent_order'],order), 'SAVED_IDENTITY:'+phase)
    if final:
        _need(saved['schema_version'] == 'gx1_ta_m5_objective_benchmark_evidence_v1' and saved['arm'] == 'FULL_PREFIX'
            and saved['recipe'] == q['native']['recipe'] and saved['source_commit'] == q['native']['source_commit']
            and saved['physical_optimizer_updates'] == 3698 and saved['training_presentations'] == 59148
            and saved['skipped_batches'] == saved['pending_gradients'] == 0
            and saved['canonical_resume_authority'] is saved['test_data_used'] is False
            and saved['optimizer_state_sha256'] == old._tree_sha(saved['optimizer_state']), 'FINAL_EVIDENCE')
    else:
        _need(saved['schema_version'] == state.TA_STATE_SCHEMA and saved['cursor'] == saved['optimizer_updates'] == saved['skipped_batches'] == 0
            and not saved['optimizer_state']['state'], 'EMPTY_INITIAL')
    pairs,active,inactive = state._parameters(model)
    _need([n for n,_ in active] == binding['active_parameter_names']
        and [n for n,_ in inactive] == binding['inactive_parameter_names']
        and len(active) == 590 and len(inactive) == 179 and all(p.grad is None for _,p in pairs)
        and set(saved['module_training']) == set(dict(model.named_modules())), 'ACTIVE_PARTITION')
    old._tensor_map(saved['model_state'],model.state_dict(),'CURRENT_MODEL')
    old._tensor_map(saved['buffer_state'],dict(model.named_buffers()),'CURRENT_BUFFERS')
    model.load_state_dict(saved['model_state'],strict=True)
    _need(model_sha(model.state_dict()) == MODELS[phase]
        and old._tree_sha(dict(model.named_buffers())) == saved['buffer_state_sha256']
        and model.state_model.feature_view_identity == _view_identity(q['view']), 'LOADED_CURRENT_IDENTITY')
    state._numerical(recipe['numerical_recipe'],model)


def compare_profile(route,piece,baseline,loss,baseline_loss,*,dynamic_reference=None,dynamic_loss=None,np):
    """Use an actual reference from the same route; retain cross-route differences."""
    from gx1.audit.observe_market_state_ta_current_numeric_routes_v1 import compare_pieces
    _need(route in ROUTES,'REFERENCE_ROUTE')
    measured=dict(retained_comparison=compare_pieces(piece,baseline,loss,baseline_loss,np=np),
                  required_reference_qualified=None)
    if route == 'NATIVE_NO_GRAD':
        measured['required_reference_qualified']=measured['retained_comparison']['within_existing_tolerance']
    elif route == 'DYNAMIC_GRAD':
        _need(dynamic_reference is not None and dynamic_loss is not None,'DYNAMIC_REFERENCE')
        measured['dynamic_comparison']=compare_pieces(piece,dynamic_reference,loss,dynamic_loss,np=np)
        measured['required_reference_qualified']=measured['dynamic_comparison']['within_existing_tolerance']
    return measured


def geometry(vectors, adam, delta, *, torch):
    groups = {}; total = {}
    for group,v in vectors.items():
        directions = dict(sgd_total=-v['total'], sgd_future_side=-v['future_all_side'],
            sgd_future_m5_side=-v['future_m5_side'], sgd_future_non_m5_side=v['future_m5_side']-v['future_all_side'],
            sgd_other_than_future_side=v['future_all_side']-v['total'],
            nominal_last_adam=adam[group], actual_endpoint_parameter_delta=delta[group])
        probes = {}
        for probe in gradient.EXTRA_ARMS[1:]:
            probes[probe] = {name:float(torch.dot(v[probe],direction)) for name,direction in directions.items()}
            _need(all(math.isfinite(x) for x in probes[probe].values()), 'GEOMETRY_FINITE')
            target = total.setdefault(probe,{name:0. for name in directions})
            for name,value in probes[probe].items(): target[name] += value
        groups[group] = probes
    return dict(groups=groups, all_active_parameters=total)


def _scalar_report(value):
    # Keep separate missing/zero counts; per-parameter names are already bound by
    # the disjoint inventory. No full vectors or gradient archives are published.
    if isinstance(value,dict): return {k:_scalar_report(v) for k,v in value.items() if k != 'tensor_gradient_state'}
    if isinstance(value,list): return [_scalar_report(v) for v in value]
    return value


def run(plan_path, plan_sha, out_dir, device='cuda'):
    started = time.monotonic(); _need(device == 'cuda','CUDA_ONLY')
    q = _inspect(plan_path,plan_sha,out_dir,execute=True); cap = previous._runtime_cap()
    import numpy as np
    import torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_m5_equal_observation_axes
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _BATCH_EQUAL_LENGTH_MTF_EVAL
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'INIT_ORDER')
    claim = cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,
        max_invocations=1,no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    counts = {k:0 for k in ('model_constructions','checkpoint_loads','resume_state_load_calls','resume_state_load_returns',
        'archive_decodes','input_sidecar_archive_decodes','checkpoint_weight_transitions','forwards','grad_forwards',
        'autograd_grad_calls','matched_batches','native_reference_matches','dynamic_reference_matches','sampled_rows','optimizer_constructions','optimizer_updates',
        'backwards','rng_restores','checkpoint_writes','candidate_writes')}
    provider=prepared=model=None; records=[]; summaries={}; passivity={}; before={}; phase=None; slot=None
    runtime={}; adam_report={}; inventory={}; transition={}
    old_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('Current update observation900-second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']: campaign.require_binding(ref,label='current observation bookend',verify_file=True)
    def signature():
        return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
            rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
            modes={n:m.training for n,m in model.named_modules()},flags={n:p.requires_grad for n,p in model.named_parameters()},
            parameters={n:id(p) for n,p in model.named_parameters()},buffer_ids={n:id(p) for n,p in model.named_buffers()},
            modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
            hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in
                ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
            mha_fastpath=torch.backends.mha.get_fastpath_enabled(),runtime=previous._runtime_profile(torch))
    def outcome(complete, **extra):
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,source_commit=q['head'],counts=counts,
            batch_records=records,phase_summary=summaries,passivity=passivity,numeric_runtime=runtime,
            parameter_inventory=inventory,adam=adam_report,checkpoint_transition=transition,
            model_digests=MODELS,**extra),complete=complete,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,max(1.,900-(time.monotonic()-started)))
    try:
        recipe,view,receipts=q['recipe'],q['view'],q['receipts']; numerical=recipe['numerical_recipe']; cuda=torch.device('cuda')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        runtime=previous._runtime_profile(torch); previous._require_runtime_profile(runtime)
        _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and torch.backends.mha.get_fastpath_enabled() is True,'RUNTIME')
        captures={}; states={}
        for load_phase in PHASES:
            with np.load(q['inputs'][load_phase+'_capture']['path'],allow_pickle=False) as z: captures[load_phase]={k:z[k].copy() for k in z.files}
            counts['archive_decodes']+=1
            _need(set(captures[load_phase]) == set(previous.BASE_KEYS) and all(len(a)==4096 for a in captures[load_phase].values()),'CAPTURE_LAYOUT')
            counts['checkpoint_loads']+=1
            states[load_phase]=common._load_state(q['inputs'][load_phase+'_checkpoint'],torch=torch,maximum=128*1024**2,counters=counts)
        provider,provider_plan,prepared,rows,issues=full.anchor.prepare_fold1(receipts,view,np=np,counters=counts)
        require_m5_equal_observation_axes(rows,issues); order=torch.from_numpy(rows.copy())
        _require_parent_order(order,states,torch=torch)
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions']+=1
        model=NativeFutureDualModel(views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'],constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile']))
        previous._short_parameter_partition(model,state)
        # Validate both actual saved states before any forward. Loading FINAL then
        # INITIAL here would add transitions, so validate FINAL through its hashes
        # and binding before the single later model transition instead.
        final=states['FINAL']; initial=states['INITIAL']
        _need(model_sha(final['model_state']) == final['model_state_sha256'] == MODELS['FINAL']
            and final['parameter_binding']['feature_view_identity'] == _view_identity(q['views']['FINAL'])
            and all(initial['binding'][k] == final['parameter_binding'][k] for k in
                ('parameter_inventory','active_parameter_names','inactive_parameter_names','optimizer_group_parameter_names','research_objective')), 'PAIRED_STATE_BINDINGS')
        _load_endpoint(model,initial,'INITIAL',q,order,torch=torch,old=old,state=state,model_sha=model_sha)
        groups=_groups(model); inventory={g:[dict(name=n,shape=list(p.shape),elements=p.numel()) for n,p in pairs] for g,pairs in groups.items()}
        adam_report,adam=gradient.nominal_last_adam_direction(final,groups,numerical_recipe=numerical,torch=torch)
        delta={g:torch.cat([(final['model_state'][n].double()-initial['model_state'][n].double()).reshape(-1) for n,p in pairs]) for g,pairs in groups.items()}
        model.to(cuda); model.eval(); groups=_groups(model)
        for phase in PHASES:
            if phase == 'FINAL':
                prior=signature(); _load_endpoint(model,final,phase,q,order,torch=torch,old=old,state=state,model_sha=model_sha)
                counts['checkpoint_weight_transitions']+=1; after=signature()
                _need(all(prior[k] == after[k] for k in prior if k not in ('model','buffers')), 'WEIGHTS_ONLY_TRANSITION')
                transition=dict(initial_model=prior['model'],final_model=after['model'],non_weight_state_unchanged=True)
            before[phase]=signature(); sums=None; phase_geometries=[]
            for slot,batch in zip(PANEL_SLOTS,BATCHES):
                raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,batch*16,batch*16+16,np=np)
                features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
                feature_signature={k:old._tree_sha(v) for k,v in features.items()}
                truths=dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                    future=future.gather(parents,clocks,fold_id=1,partition='fit'))
                retained={k:v[slot*16:slot*16+16] for k,v in captures[phase].items()}
                retained_loss=q['measurements'][phase]['measurement']['canonical_batch_losses'][slot]
                dynamic_reference=dynamic_loss=None
                for route in ROUTES:
                    grad_enabled=route == 'DYNAMIC_GRAD'; dynamic=route != 'NATIVE_NO_GRAD'
                    _need(counts['forwards'] < BOUNDS['forwards'] and time.monotonic()-started < 900, 'FORWARD_BOUND')
                    with contextlib.ExitStack() as contexts:
                        contexts.enter_context(torch.set_grad_enabled(grad_enabled))
                        if dynamic:
                            contexts.enter_context(trainer._training_mha_fastpath_context(model.backbone))
                            contexts.enter_context(trainer._training_autocast_context())
                            contexts.enter_context(trainer._training_model_finite_check_context())
                        _need(not model.training and not model.backbone.training and torch.is_grad_enabled() == grad_enabled
                            and not torch.is_inference_mode_enabled() and torch.backends.mha.get_fastpath_enabled() is (not dynamic)
                            and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False, 'FORWARD_PROFILE')
                        counts['forwards']+=1; counts['grad_forwards']+=int(grad_enabled)
                        report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                            seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=batch*16)
                        _need(report['skip_update'] is False and report['loss'].requires_grad == grad_enabled
                            and report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                            and report['loss_contract_sha256'] == recipe['expected_loss_contract_sha256'], 'REPORT_BINDING')
                        piece=dict(parents=parents.copy(),issues=clocks.copy(),state_query=report['state_query_k'].copy(),
                            future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
                        for task in ('state','future'):
                            _need(np.array_equal(truths[task]['parent_rows'],parents) and np.array_equal(truths[task]['issue_ns'],clocks)
                                and np.array_equal(truths[task]['allowed'],report[task+'_allowed'])
                                and np.array_equal(truths[task]['mature'],report[task+'_mature']), 'TARGET_AXES')
                            piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                            for key in ('allowed','eligible','mature'): piece[task+'_'+key]=truths[task][key].copy()
                        loss=float(report['loss'].detach())
                        record=dict(phase=phase,panel_slot=slot,original_batch_index=batch,route=route,
                            grad_enabled=grad_enabled,mha_fastpath=not dynamic,loss=loss)
                        records.append(record)
                        record.update(compare_profile(route,piece,retained,loss,retained_loss,
                            dynamic_reference=dynamic_reference,dynamic_loss=dynamic_loss,np=np))
                        # Cross-profile differences are never converted into an equivalence
                        # claim. Native and gradient forwards each have their own strict gate.
                        _need(record['required_reference_qualified'] is not False,'MATCHED_PROFILE_PARITY:'+route)
                        if route == 'NATIVE_NO_GRAD':
                            counts['native_reference_matches']+=1; counts['matched_batches']+=1
                        elif route == 'DYNAMIC_NO_GRAD': dynamic_reference,dynamic_loss=piece,loss
                        else:
                            counts['dynamic_reference_matches']+=1; counts['matched_batches']+=1
                            measured,vectors=gradient.measure_current_update_direction(report,groups,counts=counts,torch=torch)
                            record.update(gradient=_scalar_report(measured),geometry=geometry(vectors,adam,delta,torch=torch))
                            phase_geometries.append(record['geometry']['all_active_parameters'])
                            if sums is None: sums=vectors
                            else:
                                for g in sums:
                                    for arm in gradient.ARMS: sums[g][arm].add_(vectors[g][arm])
                            del vectors; counts['sampled_rows']+=16
                        record['features_unchanged']={k:old._tree_sha(v) for k,v in features.items()} == feature_signature
                        _need(record['features_unchanged'] and all(p.grad is None for p in model.parameters()), 'PASSIVE_INPUTS_AND_PARAMETERS')
                        del report
                del features,raw,dynamic_reference
            for v in sums.values():
                for value in v.values(): value.div_(len(PANEL_SLOTS))
            summaries[phase]=dict(geometry_of_mean_vectors=geometry(sums,adam,delta,torch=torch),
                mean_of_batch_geometry={probe:{key:sum(v[probe][key] for v in phase_geometries)/16
                    for key in phase_geometries[0][probe]} for probe in phase_geometries[0]})
            del sums
            passivity[phase]=dict(signature_unchanged=signature()==before[phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
            _need(all(passivity[phase].values()), 'PHASE_PASSIVITY')
        for key in counts:
            expected=2 if key in ('resume_state_load_calls','resume_state_load_returns') else BOUNDS[key]
            _need(counts[key] == expected, 'COUNTS:'+key)
        full.anchor.release_inputs(provider,prepared); prepared=None; provider.validate_metadata(provider_plan); bookend()
        return outcome(True,source_unchanged=True,elapsed_seconds=time.monotonic()-started,
            limits=[q['plan']['interpretation'],q['plan']['aggregation'],
                'Eval-mode TRAIN subset; no historical dropout/gradient trajectory reconstructed, no economic or independent generalization claim.',
                'All32 native forwards must match retained controls; all32 grad forwards must match their fresh dynamic no_grad references. Cross-profile differences remain separately reported under unchanged tolerances.',
                'SGD local directions use original losses without clipping or Adam preconditioning; Adam moments include their original clipped history.',
                'Observable probes are all-row mean F-side logits; loss derivatives keep original known-label masks and coefficients.'])
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0); changed=passivity_error=None
        try:
            if phase in before: passivity[phase]=dict(signature_unchanged=signature()==before[phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
        except Exception as other: passivity_error=repr(other)
        try: bookend()
        except Exception as other: changed=repr(other)
        result=outcome(False,source_unchanged=changed is None,active_phase=phase,active_panel_slot=slot,
            error=type(exc).__name__+':'+str(exc),bookend_error=changed,passivity_error=passivity_error,
            gradient_failure=_scalar_report(getattr(exc,'report_json',None)),interpretation_authorized=False)
        raise RuntimeError('TA_CURRENT_UPDATE_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_alarm)
        if prepared is not None: full.anchor.release_inputs(provider,prepared)


def main(argv=None):
    import argparse,json,sys
    supplied=list(sys.argv[1:] if argv is None else argv)
    flags=('--market-state-ta-current-updates-plan','--market-state-ta-current-updates-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True); parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True); parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0] == flag for v in supplied) > 1: parser.error('duplicate flag: '+flag)
    args=parser.parse_args(supplied)
    kwargs=dict(plan_path=args.market_state_ta_current_updates_plan,plan_sha=args.market_state_ta_current_updates_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),sort_keys=True,allow_nan=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
