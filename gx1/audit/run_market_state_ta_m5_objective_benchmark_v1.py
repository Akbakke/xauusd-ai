"""One fixed two-objective learning comparison; no model/checkpoint selection.

Reuse native feature, loss, restoration and evaluation owners. Repeated visits
are research progress, never a canonical full-epoch checkpoint cursor.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import io
from pathlib import Path
import time

from gx1.audit import trace_market_state_ta_m5_original_cuda_prefix_v1 as trace
from gx1.contracts import market_state_ta_anchor_span_short_learning_v1 as anchor


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_M5_OBJECTIVE_BENCHMARK_' + reason)


def _apply_training_update(report, *, model, optimizer, trainer, counters, progress):
    """An empty-side visit must not mutate AdamW or apply weight decay."""
    _need(type(report['skip_update']) is bool, 'SKIP_FLAG')
    loss = float(report['loss'].detach())
    norm = None
    if report['skip_update']:
        progress['skipped_batches'] += 1
        counters['skipped_batches'] += 1
    else:
        counters['backwards'] += 1
        report['loss'].backward()
        counters['backward_returns'] += 1
        counters['optimizer_step_calls'] += 1
        norm = float(trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer))
        counters['optimizer_step_returns'] += 1
        counters['optimizer_updates'] += 1
        progress['optimizer_updates'] += 1
    optimizer.zero_grad(set_to_none=True)
    progress['batch_visits'] += 1
    counters['batch_visits'] += 1
    _need(progress['batch_visits'] == progress['optimizer_updates'] + progress['skipped_batches'],
          'VISIT_ACCOUNTING')
    return dict(loss=loss, gradient_norm=norm, skip_update=report['skip_update'])


def _prepare_later(origin, prepared, receipts, view, *, counters, np):
    """Compose only the already qualified later envelope; no new features/fit."""
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import build_anchor_span_inputs, TIMEFRAMES
    result, coverage, plan = (origin[k] for k in ('later_result', 'later_coverage', 'later_plan'))
    payloads, values, clocks = {}, {}, {}
    cache = prepared['anchor_span_parent_cache']
    _need(len(cache) == 10, 'PARENT_CACHE_COUNT')
    for lane in ('local', *TIMEFRAMES):
        ref = result['surfaces'][lane]['artifact']
        anchor.campaign.require_binding(ref, label=lane+' later span', verify_file=True)
        with np.load(ref['path'], allow_pickle=False) as z:
            payloads[lane] = {k:z[k] for k in z.files}
        counters['input_sidecar_archive_decodes'] += 1
        if lane != 'local':
            index = TIMEFRAMES.index(lane)
            for key, value in (('feats', cache[index]), ('ts', cache[5+index])):
                ref = plan['caches'][lane][key]
                anchor.campaign.require_binding(ref, label=lane+' shared cache', verify_file=True)
                _need(str(Path(value.filename)) == ref['path'] and not value.flags.writeable,
                      'SHARED_PARENT_CACHE')
            values[lane], clocks[lane] = cache[index], cache[5+index]
    old = receipts['fold1']['normal']['parent_prefix_normalization']['base_normalization']
    bound = build_anchor_span_inputs(view, entry_sequence_source=prepared['entry_sequence_source'],
        snap_x=prepared['current']['snap'], ctx_cont=prepared['current']['ctx_cont'],
        ctx_cat=prepared['current']['ctx_cat'], parent_signal_names=old['surfaces']['signal']['field_names'],
        parent_mtf_names=old['surfaces']['mtf_m5']['field_names'], parent_mtf_values=values,
        parent_mtf_times=clocks, span_payloads=payloads, physical_populations=coverage['physical_populations'],
        unobserved_units_profile=receipts['fold1']['profile'], later_coverage=coverage,
        later_axes={k:prepared['axes'][k] for k in ('parent_rows','bar_start_ns','decision_ns')})
    return dict(prepared, cache=bound.mtf_sources, anchor_span_inputs=bound)


def run_objective_benchmark(*, policy_path: Path, policy_file_sha256: str, progress_path: Path) -> dict:
    from gx1.contracts import market_state_ta_m5_objective_benchmark_v1 as scope
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    limits = require_guarded_cuda_trainer_execution()
    window, plan, invocation, marker, recipe = authority.require_native_direction_research_context(
        policy_path, policy_file_sha256, progress_path)
    _need(recipe['run_scope'] == scope.SCOPE, 'ADMITTED_SCOPE')
    scope.require_recipe(recipe, source_repo=scope.REPO, source_commit=plan['source_commit'])
    origin = scope.read_origin(recipe)
    receipts, view, priors = (origin[k] for k in ('receipts','view','priors'))
    numerical, refs = recipe['numerical_recipe'], recipe['input_refs']
    root, output = scope.ROOT, Path(window['output_directory'])
    _need(not any((root/name).exists() for name in ('RESULT.json','TERMINAL.json',
              *(arm+'_EVIDENCE.pt' for arm in scope.ARMS))), 'OUTPUT_EXISTS')
    output.mkdir(parents=True)
    counters = {k:0 for k in ('model_constructions','model_construction_returns','optimizer_constructions',
        'forwards','forward_returns','training_forwards','evaluation_forwards','backwards','backward_returns',
        'optimizer_step_calls','optimizer_step_returns','optimizer_updates','skipped_batches','batch_visits',
        'checkpoint_loads','resume_state_load_calls','resume_state_load_returns','strict_full_state_restore_calls',
        'strict_full_state_restore_returns','initial_rng_imports','checkpoint_writes','candidate_evidence_writes',
        'json_bytes','input_sidecar_archive_decodes','extra_target_gathers','loss_component_owner_calls')}
    arms = {arm:dict(batch_visits=0, optimizer_updates=0, skipped_batches=0, passes_completed=0,
                    fixed_pass0_query_policy=True, training_rng_by_pass={}) for arm in scope.ARMS}
    measurements, measurement_refs, captures, restore_proofs = {}, {}, {}, {}
    timings, training_records, baseline_arrays = {}, {}, {}
    started = time.monotonic(); model_started = None; error = None
    provider = prepared = later_prepared = model = optimizer = initial = None
    source_unchanged = initial_restore_exact = evaluation_unchanged = same_queries = rng_parity = False
    import_proof = {}

    def clock_guard():
        _need(time.monotonic()-started < window['max_invocation_seconds'], 'INVOCATION_DEADLINE')
        if model_started is not None:
            _need(time.monotonic()-model_started < recipe['max_model_seconds'], 'MODEL_DEADLINE')

    try:
        trace._bookend(recipe, plan, policy_path, policy_file_sha256)
        common._json(output/'CLAIM.json', dict(recipe=window['recipe'], plan_sha256=plan['plan_sha256'],
            source_commit=plan['source_commit'], limits=limits, test_data_used=False),
            counters, recipe['max_published_json_bytes'])
        import numpy as np
        import torch
        from gx1.contracts import entry_native_direction_feature_view_v1 as views
        from gx1.contracts import entry_native_direction_research_state_v1 as old
        from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
        from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
        from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
        from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
        from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
        from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch, evaluate_endpoint_batch
        from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
        from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
        _need(torch.cuda.is_available(), 'CUDA_REQUIRED'); device = torch.device('cuda', 0)
        bound, endpoint_bound, future_bound = (load_bound_time_query_targets(),
            load_bound_five_endpoint_targets(), load_bound_future_displacement())
        provider, _, prepared, rows, issues = anchor.prepare_fold1(receipts, view, np=np, counters=counters)
        objective.require_m5_equal_observation_axes(rows, issues)
        order = torch.from_numpy(rows.copy())
        _need(tree_sha(order) == origin['original_result']['parent_order_sha256'], 'ORIGINAL_TRAIN_ORDER')
        train_positions = np.concatenate([np.arange(16*b,16*b+16,dtype=np.int64) for b in scope.TRAIN_BATCH_INDICES])
        later_positions = np.concatenate([np.arange(16*b,16*b+16,dtype=np.int64) for b in scope.LATER_BATCH_INDICES])
        axes = prepared['axes']; later_rows = axes['parent_rows'][29849:47602]; later_issues = axes['decision_ns'][29849:47602]
        expected = endpoint_bound.mature_oof_axes(fold_id=1)
        _need(np.array_equal(later_rows, expected['score_parent_rows'])
              and np.array_equal(later_issues, expected['score_decision_ns']), 'LATER_TARGET_AXES')
        later_prepared = _prepare_later(origin, prepared, receipts, view, counters=counters, np=np)
        later_rows, later_issues = later_rows[later_positions], later_issues[later_positions]
        _need(len(later_rows) == len(train_positions) == 4096
              and len(np.unique(train_positions)) == len(np.unique(later_positions)) == 4096, 'FIXED_PANELS')
        timings['input_seconds'] = time.monotonic()-started; clock_guard()
        model_started = time.monotonic()
        trainer._set_deterministic(numerical['seed'], device, numerical['precision_policy'])
        _need(trainer._GRAD_CLIP_NORM == numerical['grad_clip_norm'], 'UNCHANGED_CLIP')
        counters['model_constructions'] += 1
        backbone = views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(backbone); counters['model_construction_returns'] += 1
        source_ref = refs['short_initial_checkpoint']
        saved = common._load_state(source_ref, torch=torch, maximum=recipe['max_checkpoint_bytes'], counters=counters)
        counters['checkpoint_loads'] += 1
        _need(saved['model_state_sha256'] == model_sha(saved['model_state']) == anchor.PARENT_MODEL_SHA
              and saved['cursor'] == saved['optimizer_updates'] == saved['skipped_batches'] == 0
              and torch.equal(saved['parent_order'], order) and not saved['optimizer_state']['state']
              and tree_sha(saved['rng_state']) == saved['rng_state_sha256'] == recipe['initial_rng_state_sha256'],
              'ACTUAL_PARENT_INITIAL')
        migration = state.import_anchor_span_activity_initial(saved, model,
            source_field_names=origin['source_field_names'],
            source_profile_sha256=origin['source_view']['entry_execution_profile']['contract_sha256'],
            expected_model_state_sha256=anchor.PARENT_MODEL_SHA)
        _need(migration['expanded_parameter_tensors'] == 14 and migration['added_parameters'] == 2314
              and migration['model_state_sha256'] == recipe['expected_initial_model_state_sha256'], 'IMPORTED_INITIAL')
        source_modes, source_rng = migration['module_training'], copy.deepcopy(saved['rng_state'])
        del saved
        model.to(device); model.train(); view_identity = copy.deepcopy(model.state_model.feature_view_identity)
        _need(model.research_objective['contract_sha256'] == recipe['expected_model_contract_sha256'], 'MODEL_CONTRACT')
        counters['optimizer_constructions'] += 1
        optimizer, _ = state.build_fold1_fit_optimizer(model, numerical_recipe=numerical)
        _need(not optimizer.state and {n:bool(m.training) for n,m in model.named_modules()} == source_modes,
              'INITIAL_OPTIMIZER_MODES')
        old._rng(source_rng, device=device, seed=numerical['seed'])
        trainer._restore_attended_session_rng_state(source_rng, device=device); counters['initial_rng_imports'] += 1
        _need(tree_sha(trainer._attended_session_rng_state(device=device)) == recipe['initial_rng_state_sha256'], 'INITIAL_RNG')
        del source_rng, source_modes
        initial = state.capture_fold1_fit_state(model, optimizer, numerical_recipe=numerical,
            feature_view_identity=view_identity, parent_order=order, cursor=0, optimizer_updates=0, skipped_batches=0,
            source_commit=plan['source_commit'], device=device)
        initial_seal = tree_sha(initial)
        import_proof = dict(source_checkpoint=source_ref, migration=migration,
            current_initial_contract_sha256=initial['contract_sha256'], original_rng_restored=True,
            additional_checkpoint_decode=False)
        pairs, _, _ = state._parameters(model)

        def signature():
            return dict(model_state_sha256=model_sha(model.state_dict()),
                buffer_state_sha256=tree_sha(dict(model.named_buffers())),
                optimizer_state_sha256=tree_sha(optimizer.state_dict()),
                rng_state_sha256=tree_sha(trainer._attended_session_rng_state(device=device)),
                module_training={n:bool(m.training) for n,m in model.named_modules()})

        def publish_measurement(name, measured, arrays, model_digest, updates, loss_components=None):
            panel = name.rsplit('_',1)[1]
            if panel in baseline_arrays:
                reference = baseline_arrays[panel]
                _need(set(reference) == set(arrays) and all(np.array_equal(arrays[k], value)
                    for k,value in reference.items() if not (k.endswith('_raw') or k.endswith('_logits')
                                                             or k.endswith('_representation'))), 'MATCHED_PANEL_AXES_TARGETS')
                base = measurements['INITIAL_'+panel]
                if panel == 'TRAIN':
                    _need(measured['observation_chain_sha256'] == base['observation_chain_sha256'], 'MATCHED_TRAIN_OBSERVATIONS')
                else:
                    runner._ta_require_matched_observations(base, measured)
            else:
                baseline_arrays[panel] = arrays
            stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
            _need(0 < stream.tell() <= recipe['max_trace_bytes'], 'CAPTURE_SIZE')
            _publish_bytes(root/(name+'_READOUT.npz'), stream.getvalue())
            captures[name] = common._hash_file(root/(name+'_READOUT.npz'))
            payload = dict(schema_version='gx1_ta_m5_objective_benchmark_measurement_v1',
                measurement=measured, capture=captures[name], evaluated_model_state_sha256=model_digest,
                optimizer_updates=updates, evaluation_objective='FULL_OBJECTIVE',
                loss_components=loss_components,
                panel_batch_indices=list(scope.TRAIN_BATCH_INDICES if panel == 'TRAIN' else scope.LATER_BATCH_INDICES),
                evaluation_state_unchanged=True, canonical_resume_authority=False, test_data_used=False)
            measurement_refs[name] = common._json(root/(name+'.json'), payload, counters, recipe['max_published_json_bytes'])
            measurements[name] = measured

        def fit_report(start, *, m5_direction_only=False):
            raw, parents, clocks = runner._feature_batch(prepared, view, receipts, rows, issues, start,start+16,np=np)
            features = {k:torch.from_numpy(v).to(device) for k,v in raw.items()}
            report = evaluate_fit_batch(model, features, bound, future_bound, parents, clocks, fold_id=1,
                pass_index=0, seed=numerical['seed'], future_weight=1., m5_equal_observation=True,
                fit_batch_start=start, m5_direction_only=m5_direction_only)
            return report, parents, clocks

        def gather(parents, clocks):
            return dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                        future=future_bound.gather(parents,clocks,fold_id=1,partition='fit'))

        def measure_train(label, updates):
            measured, arrays, parts, digest = trace._short_train_panel(model, recipe=recipe, counters=counters,
                evaluate=fit_report, gather_truth=gather, capture=signature, tree_sha=tree_sha, priors=priors,
                clock_guard=clock_guard, np=np, torch=torch, tf_activity_learning=True)
            publish_measurement(label+'_TRAIN',measured,arrays,digest,updates,parts)

        def measure_later(label, updates):
            before = signature(); modules = dict(model.named_modules())
            hooks = {n:(tuple(m._forward_pre_hooks.items()),tuple(m._forward_hooks.items())) for n,m in modules.items()}
            identities = [(n,id(p),p.requires_grad) for n,p in model.named_parameters()]
            _need(all(p.grad is None for _,p in pairs), 'LATER_PENDING_GRADIENTS')
            arrays = {k:np.empty(shape,dtype) for k,(shape,dtype) in runner._ta_m5_capture_layout(4096).items()}
            total = None; model.eval()
            try:
                with torch.inference_mode():
                    for left in range(0,4096,16):
                        clock_guard()
                        raw, parents, clocks = runner._feature_batch(later_prepared,view,receipts,
                            later_rows,later_issues,left,left+16,np=np)
                        features = {k:torch.from_numpy(v).to(device) for k,v in raw.items()}
                        _need(counters['forwards'] < recipe['max_forwards']
                              and counters['evaluation_forwards'] < recipe['max_evaluation_forwards'], 'LATER_FORWARD_BUDGET')
                        counters['forwards'] += 1; counters['evaluation_forwards'] += 1
                        report = evaluate_endpoint_batch(model,features,endpoint_bound,future_bound,parents,clocks,
                            fold_id=1,partition='score',future_weight=1.,frozen_priors=priors,capture_m5_readout=True)
                        counters['forward_returns'] += 1
                        total = objective.merge_endpoint_measurements(total,report['measurement'])
                        runner._ta_m5_capture_batch(arrays,report['m5_readout'],left=left,right=left+16,parents=parents,issues=clocks)
                        del raw, features, report
            finally:
                for n,m in modules.items(): m.training = before['module_training'][n]
                _need(signature() == before and all(p.grad is None for _,p in pairs)
                      and identities == [(n,id(p),p.requires_grad) for n,p in model.named_parameters()]
                      and hooks == {n:(tuple(m._forward_pre_hooks.items()),tuple(m._forward_hooks.items()))
                                    for n,m in model.named_modules()}, 'LATER_EVAL_CHANGED_STATE')
            publish_measurement(label+'_LATER',objective.finalize_endpoint_measurement(total),arrays,
                                before['model_state_sha256'],updates)

        measure_train('INITIAL',0); measure_later('INITIAL',0)
        _need(np.array_equal(baseline_arrays['TRAIN']['parents'],rows[train_positions])
              and np.array_equal(baseline_arrays['LATER']['parents'],later_rows), 'ACTUAL_INITIAL_PANELS')
        timings['initial_seconds'] = time.monotonic()-model_started
        print('INITIAL TRAIN/LATER complete; both arms pending',flush=True)
        for arm in scope.ARMS:
            clock_guard(); counters['strict_full_state_restore_calls'] += 1
            zero = state.restore_fold1_fit_state(initial,model,optimizer,numerical_recipe=numerical,
                feature_view_identity=view_identity,parent_order=order,source_commit=plan['source_commit'],device=device)
            counters['strict_full_state_restore_returns'] += 1
            restored = state.capture_fold1_fit_state(model,optimizer,numerical_recipe=numerical,
                feature_view_identity=view_identity,parent_order=order,source_commit=plan['source_commit'],device=device,**zero)
            restore_proofs[arm] = trace._same_full_state(initial,restored,source=plan['source_commit'],tree_sha=tree_sha,model_sha=model_sha)
            arm_numerical = recipe['objective_recipes'][arm]
            rebound = state.capture_fold1_fit_state(model,optimizer,numerical_recipe=arm_numerical,
                feature_view_identity=view_identity,parent_order=order,source_commit=plan['source_commit'],device=device,**zero)
            _need(zero == dict(cursor=0,optimizer_updates=0,skipped_batches=0) and not optimizer.state
                  and all(tree_sha(restored[k]) == tree_sha(rebound[k]) for k in
                          ('model_state','buffer_state','optimizer_state','rng_state','module_training','parent_order')),
                  'OBJECTIVE_ONLY_ZERO_STEP_REBIND')
            binding = rebound['binding']; del restored,rebound
            progress = arms[arm]; progress.update(initial_model_state_sha256=signature()['model_state_sha256'],
                initial_rng_state_sha256=signature()['rng_state_sha256'])
            training_records[arm] = []; fit_started = time.monotonic()
            for pass_number in scope.PASSES:
                for slot,b in enumerate(scope.TRAIN_BATCH_INDICES):
                    clock_guard(); start = 16*b
                    _need(counters['training_forwards'] < 1536 and counters['forwards'] < recipe['max_forwards'], 'FIT_FORWARD_BUDGET')
                    optimizer.zero_grad(set_to_none=True)
                    counters['training_forwards'] += 1; counters['forwards'] += 1
                    with trainer._training_mha_fastpath_context(model.backbone), trainer._training_autocast_context(), \
                         trainer._training_model_finite_check_context():
                        report, parents, clocks = fit_report(start,m5_direction_only=(arm=='M5_DIRECTION_ONLY'))
                    counters['forward_returns'] += 1
                    _need(report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                          and report['loss_contract_sha256'] == arm_numerical['native_F_loss_contract_sha256']
                          and bool(torch.isfinite(report['loss'])), 'TRAIN_OBJECTIVE_IDENTITY')
                    reference = baseline_arrays['TRAIN']; sl = slice(16*slot,16*slot+16)
                    _need(np.array_equal(parents,reference['parents'][sl]) and np.array_equal(clocks,reference['issues'][sl])
                          and np.array_equal(report['state_query_k'],reference['state_query'][sl])
                          and np.array_equal(report['future_native_k'],reference['future_query'][16*slot])
                          and all(np.array_equal(report[t+'_'+k],reference[t+'_'+k][sl])
                                  for t in ('state','future') for k in ('allowed','mature')), 'FIXED_TRAIN_OBSERVATIONS')
                    record = _apply_training_update(report,model=model,optimizer=optimizer,trainer=trainer,
                                                    counters=counters,progress=progress)
                    training_records[arm].append(dict(pass_number=pass_number,original_batch_index=b,**record))
                    del report
                progress['passes_completed'] = pass_number
                progress['training_rng_state_sha256'] = tree_sha(trainer._attended_session_rng_state(device=device))
                progress['training_rng_by_pass'][str(pass_number)] = progress['training_rng_state_sha256']
                if pass_number in scope.EVALUATION_PASSES:
                    measure_train(arm+'_PASS'+str(pass_number),progress['optimizer_updates'])
                    measure_later(arm+'_PASS'+str(pass_number),progress['optimizer_updates'])
                print(arm+' pass='+str(pass_number)+' visits='+str(progress['batch_visits'])+
                      ' updates='+str(progress['optimizer_updates'])+' skips='+str(progress['skipped_batches']),flush=True)
            timings[arm+'_seconds'] = time.monotonic()-fit_started
            _need(progress['batch_visits'] == 768 and progress['passes_completed'] == 3
                  and state._binding(model,optimizer,arm_numerical,view_identity) == binding
                  and all(p.grad is None and bool(torch.isfinite(p).all()) for _,p in pairs), 'FINAL_ARM_STATE')
            model_state = copy.deepcopy(model.state_dict()); buffers = copy.deepcopy(dict(model.named_buffers()))
            optimizer_state = copy.deepcopy(optimizer.state_dict()); rng = copy.deepcopy(trainer._attended_session_rng_state(device=device))
            old._tensor_map(model_state,model.state_dict(),'EVIDENCE_MODEL')
            old._tensor_map(buffers,dict(model.named_buffers()),'EVIDENCE_BUFFERS')
            old._optimizer_state(optimizer_state,optimizer,model,progress['optimizer_updates'])
            old._rng(rng,device=device,seed=numerical['seed'])
            capsule = dict(schema_version='gx1_ta_m5_objective_benchmark_evidence_v1',arm=arm,recipe=window['recipe'],
                source_commit=plan['source_commit'],original_checkpoint=source_ref,canonical_resume_authority=False,
                model_state=model_state,buffer_state=buffers,optimizer_state=optimizer_state,rng_state=rng,
                module_training={n:bool(m.training) for n,m in model.named_modules()},
                selected_parent_order=torch.from_numpy(rows[train_positions].copy()),batch_visits=progress['batch_visits'],
                passes_completed=3,physical_optimizer_updates=progress['optimizer_updates'],
                skipped_batches=progress['skipped_batches'],pending_gradients=0,parameter_binding=binding,
                numerical_recipe=arm_numerical,query_pass_index=0,model_state_sha256=model_sha(model_state),
                buffer_state_sha256=tree_sha(buffers),optimizer_state_sha256=tree_sha(optimizer_state),
                rng_state_sha256=tree_sha(rng),test_data_used=False)
            capsule['evidence_sha256'] = tree_sha(capsule)
            stream=io.BytesIO(); torch.save(capsule,stream)
            _need(0 < stream.tell() <= recipe['max_candidate_evidence_bytes'], 'EVIDENCE_SIZE'); clock_guard()
            _publish_bytes(root/(arm+'_EVIDENCE.pt'),stream.getvalue()); counters['candidate_evidence_writes'] += 1
            progress['evidence_capsule'] = common._hash_file(root/(arm+'_EVIDENCE.pt'))
            progress['final_model_state_sha256'] = capsule['model_state_sha256']
            del capsule,model_state,buffers,optimizer_state,rng,stream
        _need(tree_sha(initial) == initial_seal, 'INITIAL_PAYLOAD_MUTATED')
        _need(all(arms[a]['initial_model_state_sha256'] == recipe['expected_initial_model_state_sha256']
                  and arms[a]['initial_rng_state_sha256'] == recipe['initial_rng_state_sha256'] for a in scope.ARMS),
              'MATCHED_INITIAL_ARMS')
        expected_counts = dict(model_constructions=1,model_construction_returns=1,optimizer_constructions=1,
            checkpoint_loads=1,initial_rng_imports=1,strict_full_state_restore_calls=2,strict_full_state_restore_returns=2,
            training_forwards=1536,evaluation_forwards=2560,forwards=4096,forward_returns=4096,batch_visits=1536,
            candidate_evidence_writes=2,checkpoint_writes=0,input_sidecar_archive_decodes=12)
        _need(all(counters[k] == v for k,v in expected_counts.items())
              and counters['optimizer_updates'] + counters['skipped_batches'] == 1536
              and all(counters[k] == counters['optimizer_updates'] for k in
                      ('backwards','backward_returns','optimizer_step_calls','optimizer_step_returns'))
              and len(measurement_refs) == len(captures) == 10, 'COMPLETE_COUNTS')
        rng_parity = arms['FULL_OBJECTIVE']['training_rng_by_pass'] == arms['M5_DIRECTION_ONLY']['training_rng_by_pass']
        _need(rng_parity, 'MATCHED_RNG_AFTER_EACH_PASS')
        initial_restore_exact = all(v['all_numerical_state_exact'] for v in restore_proofs.values())
        evaluation_unchanged = same_queries = True
        trace._bookend(recipe,plan,policy_path,policy_file_sha256); clock_guard(); source_unchanged = True
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
    finally:
        if prepared is not None:
            try: anchor.release_inputs(provider,prepared)
            except BaseException as exc: error = (error+'; ' if error else '')+'cleanup: '+type(exc).__name__+': '+str(exc)
        provider=prepared=later_prepared=model=optimizer=initial=None; gc.collect()
    outcome = 'COMPLETE' if error is None and source_unchanged and initial_restore_exact else 'FAILED'
    visits, updates = counters['batch_visits'], counters['optimizer_updates']
    payload = dict(schema_version=scope.RESULT_SCHEMA,outcome=outcome,
        recipe=window['recipe'],source_commit=plan['source_commit'],optimizer_updates=updates,
        physical_optimizer_updates=updates,batch_visits=visits,skipped_batches=counters['skipped_batches'],
        parent_cursor=16*visits,parent_order_sha256=None,full_state=None,final_state=None,
        canonical_resume_authority=False,arms=arms,measurement_refs=measurement_refs,panel_captures=captures,
        measurements=measurements,import_proof=import_proof,restore_proofs=restore_proofs,
        optimizer_initialization='fresh_adamw_empty_state',
        initial_model_state_sha256=import_proof.get('migration',{}).get('model_state_sha256'),
        initial_restore_exact=initial_restore_exact,evaluation_state_unchanged=evaluation_unchanged,
        same_training_queries=same_queries,rng_stream_parity=rng_parity,source_unchanged=source_unchanged,counts=counters,
        training_records=training_records,timings=dict(timings,elapsed_seconds=time.monotonic()-started),error=error,
        primary_endpoint_pass=3,checkpoint_selection=False,test_data_used=False,promotion_authorized=False,
        limits='Fixed4096TRAIN/4096later development, three passes each. Fresh sharedINITIAL and matched inputs/RNG/order; only direct loss changes. Pass3 primary. No untouched generalization, setup-selection, trade-path, profitability or full-capacity claim.')
    result = common._json(root/'RESULT.json',payload,counters,recipe['max_published_json_bytes'])
    common._json(root/'TERMINAL.json',dict(status=outcome,exit_code=0 if outcome=='COMPLETE' else 1,
        result=result,recipe=window['recipe'],source_unchanged=source_unchanged,test_data_used=False,
        no_relaunch_under_this_plan=True),counters,recipe['max_published_json_bytes'])
    cursor = scope.build_cursor(recipe_ref=window['recipe'],outcome=outcome,optimizer_updates=updates,
        parent_cursor=16*visits,parent_order_sha256=None,full_state=None,result=result,batch_size=16)
    cursor_ref = common._json(Path(window['campaign_cursor_path']),cursor,counters,recipe['max_published_json_bytes'])
    progress = authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=cursor_ref,optimizer_updates=updates,parent_cursor=16*visits,parent_order_sha256=None,outcome=outcome)
    common._json(progress_path,progress,counters,recipe['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    if error is not None: raise RuntimeError(error)
    return dict(result=result,checkpoint_pointer=cursor_ref,progress_path=str(progress_path))
