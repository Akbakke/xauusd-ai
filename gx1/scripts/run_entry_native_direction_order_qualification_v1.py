"""Bound private sign-only B16 qualification; never a fitted predictor."""
from __future__ import annotations

import gc
import signal
import sys
import time
from pathlib import Path


def _require(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_ORDER_QUALIFICATION_' + reason)


def _select_qualification_parents(ordered, diagnostic_ids, design, *, supported, np):
    _require(type(supported) is bool and len(ordered) >= 32 and len(diagnostic_ids) >= 32, 'SELECTION_ARGUMENTS')
    if not supported:
        return np.ascontiguousarray(ordered[:32],dtype=np.int64)
    parents = design.get('parents_sorted') if isinstance(design,dict) else None
    _require(isinstance(parents,list) and len(parents) == 32 and all(type(x) is int for x in parents),
        'SUPPORTED_PARENT_ID_TYPES')
    selected = np.ascontiguousarray(np.sort(diagnostic_ids[:32]),dtype=np.int64)
    expected = np.asarray(parents,dtype=np.int64)
    _require(expected.shape == (32,) and np.array_equal(selected,expected)
        and np.all(selected[1:] > selected[:-1]) and np.isin(selected,ordered).all(), 'PREDECLARED_SUPPORTED_PARENTS')
    return selected


def run_order_qualification(*, policy_path: Path, policy_file_sha256: str, progress_path: Path):
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits = require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    window, plan, invocation, marker, launch = authority.require_native_direction_research_context(
        policy_path, policy_file_sha256, progress_path)
    supported = launch['run_scope'] == authority.ORDER_SUPPORTED_SCOPE
    _require(launch['run_scope'] in (authority.ORDER_PRIVATE_SCOPE,authority.ORDER_SUPPORTED_SCOPE)
        and invocation['batch_size'] == 16, 'EXACT_SCOPE')
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha

    recipe = require_native_direction_research_recipe(launch['numerical_recipe'])
    _require(recipe['batch_size'] == 16 and recipe['max_updates'] == 2
        and recipe.get('signed_bps_supervision', False) is False
        and recipe.get('magnitude_weighted_direction', False) is False, 'SIGN_ONLY_RECIPE')
    binding_ref = common._hash_file(launch['execution_binding_path'])
    extra = {}; binding = common._read(binding_ref, extra)
    io_budget = common._read(common.INPUT_DESIGN, extra)['budget']['max_rchar_bytes']
    _require(io_budget == 137438953472, 'REUSED_IO_CEILING')
    original = common._read(prefix_authority.ORDER_ORIGIN['launch_recipe'], extra)
    initial_report = common._read(prefix_authority.ORDER_ORIGIN['initial_measurement'], extra)
    original_recipe = require_native_direction_research_recipe(original['numerical_recipe'])
    proof = common._read(launch['gc_layout_metadata_parity'], extra)
    selection_design = None
    if supported:
        selection_design = common._read(launch['qualification_design'], extra)
        _require(selection_design['schema_version'] == 'gx1_order_supported_qualification_design_v1'
            and selection_design['status'] == 'DESIGN_ONLY_NO_EXECUTION_AUTHORITY'
            and selection_design['seed'] == prefix.SEED and selection_design['prefix_rows'] == launch['fold']['prefix_rows']
            and selection_design['batch_size'] == 16 and selection_design['test_data_used'] is False, 'SUPPORTED_DESIGN')
        for key in ('prior_failure_review','prior_support_report','prior_support_run_result','prior_support_root_review'):
            common._read(selection_design[key], extra)
    _require(proof['status'] == 'COMPLETE_CONSUMED' and proof['metadata_exact_parity'] is True
        and proof['test_cases_passed'] == 6 and all(v == 0 for v in proof['counters'].values()), 'GC_METADATA_PROOF')
    manifest = common._read(launch['input_manifest'], extra)
    transition = {'original':manifest['view_source_bindings']['direction_component_source'],
        'current':launch['source_owners'][prefix_authority.ORDER_COMPONENT_SOURCE]}
    actual_launch = {**launch, 'run_scope':prefix_authority.ORDER_SCOPE,
        'parent_order_algorithm':prefix_authority.ORDER_ALGORITHM}
    context = prefix._metadata(actual_launch, common, gc_layout_metadata_parity=launch['gc_layout_metadata_parity'],
        order_qualification={'historical_component_transition':transition})
    context['info']['metadata'].update(extra)
    for ref in (window['recipe'], binding['plan'], {'path':str(policy_path), 'sha256':policy_file_sha256},
            launch['qualification_design']):
        common._read(ref, context['info']['metadata'])
    common._freeze(launch, binding, context)
    output = Path(window['output_directory']); output.mkdir(parents=True)
    original_prediction = prefix_authority.ORDER_ORIGIN['initial_predictions']
    denied = common._install_read_guard(context, output, Path(launch['runtime_root']),
        original_initial_prediction=original_prediction['path'])
    counters = {key:0 for key in ('model_constructions','model_construction_returns','optimizer_constructions',
        'forwards','forward_returns','backwards','backward_returns','optimizer_step_calls','optimizer_updates',
        'staged_state_load_calls','staged_state_load_returns','resume_state_load_calls','resume_state_load_returns',
        'forbidden_legacy_calls','json_bytes')}
    evaluation = {'forward_calls':0,'forward_returns':0,'forward_kind_calls':{}}
    before = common._io(); started = time.monotonic(); previous = sys.getprofile()
    device = torch.device('cuda', 0); full_state = initial = initial_identity = None
    error = None; matched = parity = unchanged = False; cursor = 0; steps = []; batch_support = []
    order_sha = '0'*64; model = optimizer = None
    common._json(output/'CLAIM.json', {'recipe':window['recipe'],'plan_sha256':plan['plan_sha256'],
        'limits':limits,'additional_worker_ceiling_seconds':1200}, counters, launch['max_published_json_bytes'])

    def expired(*_args):
        raise RuntimeError('NATIVE_ORDER_QUALIFICATION_1200_SECOND_CEILING')

    old_handler = signal.signal(signal.SIGALRM, expired); signal.alarm(1200)
    try:
        sys.setprofile(common._deny_legacy_work(previous, counters))
        prepared, targets, view, context_map, ordered = prefix._prepare(context, actual_launch, np)
        _require(prefix._prefix_prior(prepared['arrays'], targets, launch['fold'], np)
            == initial_report['own_prefix_prior'], 'ORIGINAL_PREFIX_PRIOR')
        original_stat = common._stat(original_prediction['path'])
        diagnostic_ids = prefix._fixed_order(prepared['arrays']['parent_rows'],
            launch['fold']['prefix_rows'], prefix.SEED, np)[:256]
        selected = _select_qualification_parents(ordered,diagnostic_ids,selection_design,supported=supported,np=np)
        _require(len(selected) == 32 and np.all(selected[1:] > selected[:-1]), 'FIRST_CHRONOLOGICAL_PARENTS')
        order = torch.from_numpy(selected); order_sha = tree_sha(order)
        batches = []
        for left in (0,16):
            raw, target = prefix._batch(prepared, targets, view, selected[left:left+16], np)
            gc_support = common._gc_support(raw,prepared['gc']['layout'],np,allow_empty=not supported)
            if supported:
                _require(all(sum(item['raw_nonzero_values_per_parent']) > 0 for families in gc_support.values()
                    for item in families.values()), 'SUPPORTED_GC_NONZERO_INPUTS')
            batch_support.append({'parent_rows':selected[left:left+16].tolist(),'gc':gc_support})
            batches.append(prefix._device_batch(raw, target, torch, device))
        batch_sha = tree_sha(batches)

        def capture(m, opt, offset, updates, recipe_used=recipe):
            return state_owner.capture_native_direction_research_state(m, opt, recipe=recipe_used,
                feature_view_identity=m.feature_view_identity, parent_order=order, cursor=offset,
                optimizer_updates=updates, device=device)

        for trajectory in ('reference','interrupted','resumed'):
            cursor = 0
            model, optimizer, _ = common._fresh(context, view, recipe, device, counters, max_constructions=3)
            original_fresh = capture(model, optimizer, 0, 0, original_recipe)
            identity = tree_sha({k:original_fresh[k] for k in ('binding','model_state','buffer_state',
                'optimizer_state','rng_state','module_training')})
            _require(identity == initial_report['fresh_identity_sha256'], 'ORIGINAL_FRESH_IDENTITY')
            del original_fresh
            fresh = capture(model, optimizer, 0, 0)
            if initial_identity is None:
                initial_identity = identity
                measured, count = prefix._evaluate(model, prepared, targets, view, context_map, diagnostic_ids,
                    cutoff=launch['fold']['train_cutoff_ns'],kind='INITIAL_TRAIN',counters=evaluation,
                    ceiling=17,device=device,torch=torch,np=np,tree_sha=tree_sha,parity=True)
                _require(count == 256 and measured['initial_parity_passed'], 'ALL_ORIGINAL_INITIAL_ROWS')
                prefix._require_original_initial_match(measured, original_prediction, common=common, np=np)
                del measured
                _require(common._stat(original_prediction['path']) == original_stat, 'ORIGINAL_INITIAL_UNCHANGED')
                matched = True
                initial = common._save_state(output/'INITIAL_STATE.pt',fresh,torch=torch,
                    maximum=launch['max_checkpoint_bytes'],counters=counters)
                initial_inactive = tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})
            else:
                _require(identity == initial_identity and matched, 'SAME_FRESH_IDENTITY_BEFORE_UPDATE')
            del fresh
            if trajectory == 'resumed':
                loaded = common._load_state(interrupted_ref,torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
                restored = state_owner.restore_native_direction_research_state(loaded,model,optimizer,recipe=recipe,
                    feature_view_identity=model.feature_view_identity,parent_order=order,device=device)
                _require(restored['cursor'] == 16 and restored['optimizer_updates'] == 1
                    and capture(model,optimizer,16,1)['contract_sha256'] == loaded['contract_sha256'], 'SERIALIZED_RESTORE')
                del loaded; cursor = 16
            logical = (0,1) if trajectory == 'reference' else (0,) if trajectory == 'interrupted' else (1,)
            for index in logical:
                _require(matched and time.monotonic()-started < 1200, 'PARITY_AND_DEADLINE_BEFORE_UPDATE')
                evidence = common._step(model,optimizer,*batches[index],cutoff=launch['fold']['train_cutoff_ns'],
                    counters=counters,torch=torch)
                cursor = 16*(index+1)
                steps.append({'trajectory':trajectory,'logical_step':index+1,**evidence})
                _require(tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})
                    == initial_inactive, 'INACTIVE_WEIGHTS_UNCHANGED')
            saved = capture(model,optimizer,cursor,1 if trajectory == 'interrupted' else 2)
            if trajectory == 'reference':
                reference_sha = saved['contract_sha256']
            elif trajectory == 'interrupted':
                interrupted_ref = common._save_state(output/'INTERRUPTED_STEP1.pt',saved,torch=torch,
                    maximum=launch['max_checkpoint_bytes'],counters=counters)
            else:
                _require(saved['contract_sha256'] == reference_sha, 'BIT_EXACT_COMPLETE_RESUME')
                full_state = common._save_state(output/'RESUMED_STEP2.pt',saved,torch=torch,
                    maximum=launch['max_checkpoint_bytes'],counters=counters); parity = True
            del saved, model, optimizer
            model = optimizer = None; gc.collect(); torch.cuda.empty_cache()
        expected = {'model_constructions':3,'model_construction_returns':3,'optimizer_constructions':3,
            'forwards':4,'forward_returns':4,'backwards':4,'backward_returns':4,'optimizer_step_calls':4,
            'optimizer_updates':4,'staged_state_load_calls':3,'staged_state_load_returns':3,
            'resume_state_load_calls':1,'resume_state_load_returns':1,'forbidden_legacy_calls':0}
        _require(all(counters[k] == v for k,v in expected.items()) and evaluation['forward_calls'] == 17
            and evaluation['forward_returns'] == 17 and tree_sha(batches) == batch_sha, 'EXACT_WORK')
        _require(common._stat(original_prediction['path']) == original_stat, 'ORIGINAL_INITIAL_BOOKEND')
        common._freeze(launch,binding,context); unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
        try: common._freeze(launch,binding,context); unchanged = True
        except Exception: unchanged = False
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old_handler); sys.setprofile(previous)
    after = common._io()
    if after['rchar'] > io_budget:
        error = 'RCHAR_BUDGET_EXCEEDED'
    passed = matched and parity and unchanged and error is None
    outcome = 'COMPLETE' if passed else 'FAILED'
    result = common._json(output/'RESULT.json', {'schema_version':'gx1_native_order_private_qualification_result_v1',
        'decision':'PASS_ORDER_SIGN_ONLY_NATIVE_MECHANICS' if passed else 'FAILED','recipe':window['recipe'],
        'execution_binding':binding_ref,'counts':{k:v for k,v in counters.items() if k!='json_bytes'},
        'initial_evaluation_counts':evaluation,'initial_state':initial,'initial_identity_sha256':initial_identity,
        'original_initial_prediction':original_prediction,'original_initial_report':prefix_authority.ORDER_ORIGIN['initial_measurement'],
        'historical_component_transition':transition,'final_state':full_state,'matched_initialization':matched,
        'bit_exact_resume':parity,'steps':steps,'batch_gc_support':batch_support,
        'qualification_selection':('predeclared_original_mixed_32_sorted' if supported else 'first_32_chronological'),
        'parent_order_sha256':order_sha,'parent_cursor':cursor,
        'planned_unique_parents':32,'planned_logical_optimizer_steps':2,'physical_optimizer_updates':counters['optimizer_updates'],
        'source_unchanged':unchanged,'metadata_bookends':context['info']['metadata'],
        'payload_stat_bookends':context['ancestor_plan']['payloads'],'forbidden_open_paths':denied,
        'limits':limits,'elapsed_seconds':time.monotonic()-started,'io_before':before,'io_after':after,
        'private_weights_for_main_fit':False,'predictive_value_qualified':False,'test_data_used':False,
        'learning_boundary':'Two chronological B16 steps repeated for resume; no fit or later gain claim.','error':error},
        counters,launch['max_published_json_bytes'])
    checkpoint = common._json(Path(window['campaign_cursor_path']),authority.build_research_cursor(
        recipe_ref=window['recipe'],outcome=outcome,optimizer_updates=counters['optimizer_updates'],parent_cursor=cursor,
        parent_order_sha256=order_sha,full_state=full_state,result=result,batch_size=16),counters,launch['max_published_json_bytes'])
    progress = authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=checkpoint,optimizer_updates=counters['optimizer_updates'],parent_cursor=cursor,
        parent_order_sha256=order_sha,outcome=outcome)
    common._json(progress_path,progress,counters,launch['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    common._freeze(launch,binding,context)
    if not passed:
        raise RuntimeError('NATIVE_ORDER_QUALIFICATION_FAILED:' + str(error))
    return progress
