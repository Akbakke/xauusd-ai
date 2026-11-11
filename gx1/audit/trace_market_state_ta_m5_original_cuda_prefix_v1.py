"""Original CUDA prefix replay and one separately admitted c171 short fit.

Both use the same native256-step loop. The legacy replay is read-only and
requires original P4096 parity; the c171 branch binds a fresh optimizer and
matched TRAIN measurements. Neither grants future resume or selection rights.
"""
from __future__ import annotations

import copy
import gc
import io
from pathlib import Path
import time

from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

HEAD_NAMES = (
    'state_model.m5_side_head.weight', 'state_model.m5_side_head.bias',
    'future_m5_side_head.weight', 'future_m5_side_head.bias',
)
FIRST_LOSS = 2.7037248611450195
FIRST_NORM = 6.401226997375488
UPDATES = 256
ROWS = 4096


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_M5_ORIGINAL_CUDA_PREFIX_' + reason)


def _state_identity(value):
    keys = ('source_commit', 'contract_sha256', 'model_state_sha256',
            'buffer_state_sha256', 'optimizer_state_sha256', 'rng_state_sha256',
            'cursor', 'optimizer_updates', 'skipped_batches')
    return {key: value[key] for key in keys}


def _same_full_state(expected, actual, *, source, tree_sha, model_sha):
    """Compare all sealed numerical state; new execution provenance stays new."""
    _need(expected['source_commit'] == source, 'ORIGINAL_STATE_SOURCE')
    _need(expected['contract_sha256'] == tree_sha({
        key: value for key, value in expected.items() if key != 'contract_sha256'}),
        'ORIGINAL_STATE_SEAL')
    _need(set(expected) == set(actual), 'STATE_KEYS')
    for key in ('model_state', 'buffer_state', 'optimizer_state', 'rng_state'):
        digest = model_sha if key == 'model_state' else tree_sha
        _need(digest(expected[key]) == expected[key + '_sha256']
              == digest(actual[key]) == actual[key + '_sha256'],
              'EXACT_' + key.upper())
    for key in set(expected) - {'source_commit', 'contract_sha256',
                               'model_state', 'buffer_state', 'optimizer_state', 'rng_state'}:
        _need(tree_sha(expected[key]) == tree_sha(actual[key]), 'EXACT_' + key.upper())
    return {'all_numerical_state_exact': True,
            'original': _state_identity(expected), 'replayed': _state_identity(actual),
            'compared_source_metadata': False, 'checkpoint_file_hash_parity_claimed': False}


def _bookend(recipe, plan, policy_path, policy_sha):
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _need(campaign.file_sha256(policy_path) == policy_sha, 'WINDOW_CHANGED')
    campaign.require_clean_source(plan)
    for value in (plan['recipe'], recipe['trace_worker'],
                  *recipe['source_owners'].values(), *recipe['input_refs'].values()):
        campaign.require_binding(value, label='original CUDA prefix bookend', verify_file=True)


def _require_execution_bounds(recipe, *, short_learning, non_m5_learning=False):
    """Keep the consumed replay separate from the one admitted c171 experiment."""
    _need(type(short_learning) is type(non_m5_learning) is bool
          and (not non_m5_learning or short_learning), 'SHORT_MODE')
    if short_learning:
        if non_m5_learning:
            from gx1.contracts import market_state_ta_non_m5_short_learning_v1 as short
        else:
            from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
        _need(recipe['run_scope'] == short.SCOPE
              and recipe['max_private_updates'] == recipe['max_backwards'] == UPDATES
              and recipe['max_training_forwards'] == UPDATES
              and recipe['max_evaluation_forwards'] == (512 if non_m5_learning else 32)
              and recipe['max_forwards'] == (768 if non_m5_learning else 288)
              and (not non_m5_learning or recipe['max_model_constructions'] == 2
                  and recipe['max_initial_rng_imports'] == 1 and recipe['dedicated_non_m5_side'] is True)
              and recipe['max_model_seconds'] == 1800
              and recipe['max_checkpoint_writes'] == 2
              and recipe['panel_batch_indices'] == (list(range(256)) if non_m5_learning
                  else [(i * 1847) // 15 for i in range(16)])
              and recipe['optimizer_initialization'] == 'fresh_adamw_empty_state',
              'SHORT_ADMITTED_BOUNDS')
    else:
        _need(recipe['run_scope'] == scope.TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE
              and recipe['max_private_updates'] == recipe['max_forwards']
              == recipe['max_backwards'] == UPDATES
              and recipe['max_model_seconds'] == 720
              and recipe['max_checkpoint_writes'] == 0, 'ADMITTED_BOUNDS')
    _need(recipe['max_trace_bytes'] == 64 * 1024**2
          and recipe['max_checkpoint_loads'] == 2
          and recipe['max_full_state_restores'] == 1, 'ADMITTED_STATE_BOUNDS')


def _short_loss_components(report, truths, *, torch, non_m5_learning=False):
    """Read unchanged owner terms; these are activity+side, before the TF divisor."""
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    raw = report['state_activity_side_logits']
    checked = objective.combined_loss(
        state_raw=raw, future_raw=report['future_activity_side_logits'],
        state_truth=truths['state'], future_truth=truths['future'],
        state_query_k=torch.from_numpy(report['state_query_k']).to(raw.device),
        future_weight=1.0, ta_only=True, dedicated_m5_side=True,
        m5_equal_observation=True, dedicated_non_m5_side=non_m5_learning)
    _need(torch.equal(checked['loss'], report['loss'])
          and checked['contract_sha256'] == report['loss_contract_sha256'],
          'SHORT_ORIGINAL_LOSS_PARITY')
    return dict(total=float(report['loss']), **{
        task + '_per_tf_activity_plus_side_before_divisor': {
            tf: float(value) for tf, value in checked[task]['per_tf_mean_loss'].items()}
        for task in ('state', 'future')})


def _short_train_panel(model, *, recipe, counters, evaluate, gather_truth, capture,
                       tree_sha, priors, clock_guard, np, torch, non_m5_learning=False):
    """Measure the fixed TRAIN panel without altering complete native state."""
    from gx1.audit.run_market_state_ta_m5_paired_batch_order_v1 import sampled_fit_measurement
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    before = capture()
    before_sha = tree_sha(before)
    modules, parameters = dict(model.named_modules()), list(model.named_parameters())
    modes = {name: module.training for name, module in modules.items()}
    identities = [(name, id(value), value.requires_grad) for name, value in parameters]
    hooks = {name: (tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items()))
             for name, module in modules.items()}
    _need(all(value.grad is None for _, value in parameters), 'SHORT_PANEL_PENDING_GRADIENT')
    pieces, losses, components = [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for index in recipe['panel_batch_indices']:
                clock_guard()
                _need(counters['forwards'] < recipe['max_forwards']
                      and counters['evaluation_forwards'] < recipe['max_evaluation_forwards'],
                      'SHORT_PANEL_FORWARD_BUDGET')
                counters['forwards'] += 1
                counters['evaluation_forwards'] += 1
                report, parents, issues = evaluate(16 * index)
                counters['forward_returns'] += 1
                _need(report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                      and report['loss_contract_sha256'] == recipe['expected_loss_contract_sha256']
                      and report['skip_update'] is False and bool(torch.isfinite(report['loss'])),
                      'SHORT_PANEL_LOSS')
                truths = gather_truth(parents, issues)
                counters['extra_target_gathers'] += 2
                block = dict(parents=parents.copy(), issues=issues.copy(),
                    state_query=report['state_query_k'].copy(),
                    future_query=np.broadcast_to(report['future_native_k'], (16, 5)).copy())
                _need(parents.shape == issues.shape == (16,)
                      and parents.dtype == issues.dtype == np.int64, 'SHORT_PANEL_ROWS')
                for task in ('state', 'future'):
                    truth = truths[task]
                    _need(np.array_equal(truth['parent_rows'], parents)
                          and np.array_equal(truth['issue_ns'], issues)
                          and (np.array_equal(truth['query_k'], block['state_query']) if task == 'state'
                               else np.array_equal(report['future_native_k'], np.asarray(NATIVE_K, np.int64))),
                          'SHORT_PANEL_AXES')
                    for name in ('allowed', 'mature'):
                        _need(np.array_equal(truth[name], report[task + '_' + name]),
                              'SHORT_PANEL_TARGET_PARITY')
                    for name in ('allowed', 'eligible', 'mature'):
                        block[task + '_' + name] = truth[name].copy()
                    block[task + '_raw'] = report[task + '_activity_side_logits'].detach().cpu().numpy().copy()
                components.append(_short_loss_components(report, truths, torch=torch, non_m5_learning=non_m5_learning))
                counters['loss_component_owner_calls'] += 1
                pieces.append(block)
                losses.append(float(report['loss']))
    finally:
        for name, module in modules.items():
            module.training = modes[name]
        _need(tree_sha(capture()) == before_sha
              and identities == [(name, id(value), value.requires_grad)
                                 for name, value in model.named_parameters()]
              and hooks == {name: (tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items()))
                            for name, module in model.named_modules()}
              and all(value.grad is None for _, value in model.named_parameters()),
              'SHORT_PANEL_CHANGED_STATE')
    _need(len(pieces) == (256 if non_m5_learning else 16), 'SHORT_PANEL_COMPLETE')
    arrays = {name: np.concatenate([block[name] for block in pieces], axis=0) for name in pieces[0]}
    measurement = sampled_fit_measurement(arrays, losses, priors,
        model_contract_sha256=recipe['expected_model_contract_sha256'],
        loss_contract_sha256=recipe['expected_loss_contract_sha256'],
        feature_view_sha256=recipe['expected_view_sha256'])
    return measurement, arrays, components, before['model_state_sha256']


def run_original_cuda_prefix(*, policy_path: Path, policy_file_sha256: str,
                             progress_path: Path, short_learning: bool = False,
                             non_m5_learning: bool = False) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common

    limits = require_guarded_cuda_trainer_execution()
    window, plan, invocation, marker, recipe = authority.require_native_direction_research_context(
        policy_path, policy_file_sha256, progress_path)
    _require_execution_bounds(recipe, short_learning=short_learning, non_m5_learning=non_m5_learning)
    if short_learning:
        if non_m5_learning:
            from gx1.contracts import market_state_ta_non_m5_short_learning_v1 as short
        else:
            from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
        origin = short.read_origin(recipe)
    else:
        origin = scope.read_ta_m5_original_cuda_prefix_origin(recipe)
    original = origin['original_recipe']
    if non_m5_learning:
        short.require_unchanged_numerics(recipe['numerical_recipe'], original['numerical_recipe'])
    _need((non_m5_learning or recipe['numerical_recipe'] == original['numerical_recipe'])
          and recipe['numerical_recipe']['grad_accum_steps'] == 1
          and recipe['numerical_recipe']['grad_clip_norm'] == 1.0
          and recipe['numerical_recipe']['future_weight'] == 1.0
          and recipe['numerical_recipe']['m5_equal_observation'] is True,
          'UNCHANGED_NUMERICAL_RECIPE')
    refs = recipe['input_refs']
    initial_ref = None if short_learning else refs['original_initial_checkpoint']
    target_ref = None if short_learning else refs['original_p4096_checkpoint']
    source_ref = (recipe['candidate_inputs']['short_initial_checkpoint'] if non_m5_learning
                  else recipe['candidate_inputs']['reversed_evidence'] if short_learning else initial_ref)
    output = Path(window['output_directory'])
    output.mkdir(parents=True)
    root = short.ROOT if short_learning else scope.TA_M5_ORIGINAL_CUDA_PREFIX_ROOT
    _need(not any((root / name).exists() for name in ('TRACE.npz', 'RESULT.json', 'TERMINAL.json')),
          'OUTPUT_EXISTS')
    counters = {key: 0 for key in (
        'model_constructions', 'model_construction_returns', 'optimizer_constructions',
        'forwards', 'forward_returns', 'backwards', 'backward_returns',
        'optimizer_step_calls', 'optimizer_updates', 'checkpoint_loads',
        'resume_state_load_calls', 'resume_state_load_returns',
        'strict_full_state_restore_calls', 'strict_full_state_restore_returns',
        'checkpoint_writes', 'extra_target_gathers', 'json_bytes')}
    if short_learning:
        counters.update({key: 0 for key in ('training_forwards', 'evaluation_forwards',
            'staged_state_load_calls', 'staged_state_load_returns', 'head_tensor_imports',
            'loss_component_owner_calls', 'archive_decodes', 'published_panel_bytes')})
    if non_m5_learning:
        counters['initial_rng_imports'] = 0
    started = time.monotonic()
    model_started = None
    provider = prepared = model = optimizer = initial_saved = target_saved = None
    handles, records, state_proofs = [], [], {}
    order_sha = '0' * 64
    first_step_exact = final_exact = source_unchanged = False
    trace_ref = error = None
    cursor = 0
    timings = {}
    current = {}
    measurements, measurement_refs, panel_captures, import_proof = {}, {}, {}, {}
    initial_restore_exact = evaluation_state_unchanged = False
    final_state_ref = None

    def clock_guard():
        now = time.monotonic()
        _need(now - started < window['max_invocation_seconds'], 'INVOCATION_DEADLINE')
        if model_started is not None:
            _need(now - model_started < recipe['max_model_seconds'], 'MODEL_DEADLINE')

    try:
        _bookend(recipe, plan, policy_path, policy_file_sha256)
        common._json(output / 'CLAIM.json', {
            'recipe': window['recipe'], 'plan_sha256': plan['plan_sha256'],
            'source_commit': plan['source_commit'], 'original_source_commit': recipe['original_source_commit'],
            'checkpoint_inputs': [source_ref] if short_learning else [initial_ref, target_ref],
            'limits': limits, 'test_data_used': False,
        }, counters, recipe['max_published_json_bytes'])
        import numpy as np
        import torch
        from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
        from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
        from gx1.contracts import entry_native_direction_feature_view_v1 as views
        from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
        from gx1.contracts import market_state_time_query_fold1_fit_v1 as full_scope
        from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
        from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
        from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
        from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
        from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
        from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
        from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
            _dynamic_m5_technical_tree_sha256 as tree_sha)

        _need(torch.cuda.is_available(), 'CUDA_REQUIRED')
        device, numerical = torch.device('cuda', 0), recipe['numerical_recipe']
        receipts = (origin['receipts'] if short_learning else zero.load_ta_input_receipts(
            current_source_owners={name: value['sha256'] for name, value in recipe['source_owners'].items()}))
        view = origin['view'] if short_learning else zero._bind_ta_view(receipts, dedicated_m5_side=True)
        view_ref = original['input_refs']['ta_candidate_view']
        _need((non_m5_learning or view == authority.campaign.read_bound_json(Path(view_ref['path']), view_ref['sha256']))
              and view['contract_sha256'] == recipe['expected_view_sha256'], 'EXACT_ORIGINAL_VIEW')
        bound = load_bound_time_query_targets()
        future_bound = load_bound_future_displacement()
        provider, _, prepared, rows, issues = runner._prepared_fold1(receipts, view, np=np)
        objective.require_m5_equal_observation_axes(rows, issues)
        order = torch.from_numpy(rows.copy())
        order_sha = tree_sha(order)
        _need(order_sha == origin['original_result']['parent_order_sha256'], 'ORIGINAL_PARENT_ORDER')
        timings['input_setup_seconds'] = time.monotonic() - started
        clock_guard()
        model_started = time.monotonic()
        trainer._set_deterministic(numerical['seed'], device, numerical['precision_policy'])
        _need(trainer._GRAD_CLIP_NORM == numerical['grad_clip_norm'], 'CANONICAL_CLIP')
        counters['model_constructions'] += 1
        backbone = views.build_native_direction_research_model(
            origin['source_view'] if non_m5_learning else view, prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'], device=torch.device('cpu') if short_learning else device,
            unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(backbone)
        if not short_learning:
            model.to(device)
        counters['model_construction_returns'] += 1
        view_identity = None if non_m5_learning else full_scope.require_checkpoint_view(
            model.state_model.feature_view_identity, recipe=original,
            verify_files=True, ta_sf=True)
        _need(model.research_objective['contract_sha256'] == (original['expected_model_contract_sha256']
              if non_m5_learning else recipe['expected_model_contract_sha256']), 'EXACT_MODEL_CONTRACT')
        if non_m5_learning:
            saved = common._load_state(source_ref, torch=torch,
                maximum=recipe['max_checkpoint_bytes'], counters=counters)
            counters['checkpoint_loads'] += 1
            _need(saved['model_state_sha256'] == model_sha(saved['model_state']) == short.original_short.MODEL_SHA256
                and saved['feature_view_sha256'] == origin['short_recipe']['expected_view_sha256']
                and (saved['cursor'], saved['optimizer_updates'], saved['skipped_batches']) == (0, 0, 0)
                and torch.equal(saved['parent_order'], order)
                and tree_sha(saved['rng_state']) == saved['rng_state_sha256'] == short.INITIAL_RNG_SHA256,
                'NON_M5_ORIGINAL_INITIAL')
            model.load_state_dict(saved['model_state'], strict=True)
            source_modes = dict(saved['module_training'])
            _need(set(source_modes) == set(dict(model.named_modules())), 'NON_M5_SOURCE_MODES')
            for name, module in model.named_modules(): module.training = source_modes[name]
            short.initial.previous._short_parameter_partition(model, state)
            source_initial_rng = copy.deepcopy(saved['rng_state'])
            del saved
            source_model = model
            _need(counters['model_constructions'] < recipe['max_model_constructions'], 'NON_M5_CONSTRUCTOR_BUDGET')
            counters['model_constructions'] += 1
            model = NativeFutureDualModel(views.build_native_direction_research_model(
                view, prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
                device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile']))
            counters['model_construction_returns'] += 1
            migration = state.copy_dedicated_m5_weights_to_non_m5_sides(source_model, model,
                expected_legacy_model_sha256=short.original_short.MODEL_SHA256)
            _need(migration['added_parameters'] == 1032 and len(migration['copied_side_rows']) == 16
                and migration['model_state_sha256'] == recipe['expected_initial_model_state_sha256'],
                'NON_M5_EXACT_MIGRATION')
            del source_model
            counters['head_tensor_imports'] = 16
            model.to(device)
            view_identity = copy.deepcopy(model.state_model.feature_view_identity)
            _need(view_identity == dict(contract_sha256=view['contract_sha256'],
                prefix_normalization=view['prefix_normalization'],research_objective_sha256=view['research_objective']['contract_sha256'])
                and model.research_objective['contract_sha256'] == recipe['expected_model_contract_sha256'],
                'NON_M5_VIEW_IDENTITY')
            import_proof = dict(source_checkpoint=source_ref, migration=migration,
                rng_policy='explicit_original_short002_initial_rng_matched_before_fresh_initial_capture',
                optimizer_initialization=recipe['optimizer_initialization'])
        elif short_learning:
            from gx1.audit import cache_market_state_ta_direction_v1 as cache
            from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
            saved = common._load_state(source_ref, torch=torch,
                maximum=recipe['max_checkpoint_bytes'], counters=counters)
            counters['checkpoint_loads'] += 1
            reversed_order = torch.from_numpy(build_reverse_order(rows, origin['recipe']['batch_indices'])[0].copy())
            source_identity = cache._load_reversed_weights(model, saved, origin, reversed_order, torch=torch)
            del saved, reversed_order
            candidate_refs = recipe['candidate_inputs']
            with np.load(candidate_refs['folded_coefficients']['path'], allow_pickle=False) as archive:
                _need(set(archive.files) == set(HEAD_NAMES), 'SHORT_HEAD_KEYS')
                coefficients = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
            counters['archive_decodes'] += 1
            import_proof = state.import_equal_weight_m5_side_heads(model, coefficients,
                expected_model_state_sha256=cache.REVERSED_MODEL,
                coefficient_artifact=candidate_refs['folded_coefficients'],
                coefficient_recipe=candidate_refs['candidate_plan'])
            counters['head_tensor_imports'] = 4
            _need(model_sha(model.state_dict()) == recipe['expected_initial_model_state_sha256'],
                  'SHORT_EXACT_C171')
            import_proof = dict(weights=source_identity, heads=import_proof,
                rng_policy='declared_seed_and_constructor_rng_no_R1_rng_restore',
                optimizer_initialization=recipe['optimizer_initialization'])
            del coefficients
            model.to(device)
        model.train()
        counters['optimizer_constructions'] += 1
        optimizer, binding = state.build_fold1_fit_optimizer(model, numerical_recipe=numerical)
        if short_learning:
            _need(not optimizer.state, 'SHORT_FRESH_EMPTY_ADAMW')
            if non_m5_learning:
                from gx1.contracts import entry_native_direction_research_state_v1 as old_state
                old_state._rng(source_initial_rng, device=device, seed=numerical['seed'])
                _need(all(dict(model.named_modules())[name].training == mode for name, mode in source_modes.items()),
                      'NON_M5_MATCHED_TRAINING_MODES')
                trainer._restore_attended_session_rng_state(source_initial_rng, device=device)
                counters['initial_rng_imports'] += 1
                _need(tree_sha(trainer._attended_session_rng_state(device=device)) == short.INITIAL_RNG_SHA256,
                      'NON_M5_MATCHED_INITIAL_RNG')
                del source_initial_rng, source_modes
            fresh = state.capture_fold1_fit_state(
                model, optimizer, numerical_recipe=numerical, feature_view_identity=view_identity,
                parent_order=order, cursor=0, optimizer_updates=0, skipped_batches=0,
                source_commit=plan['source_commit'], device=device)
            initial_ref = common._save_state(output / 'INITIAL_C171_STATE.pt', fresh,
                torch=torch, maximum=recipe['max_checkpoint_bytes'], counters=counters)
            counters['checkpoint_writes'] += 1
            del fresh
        initial_saved = common._load_state(initial_ref, torch=torch,
            maximum=recipe['max_checkpoint_bytes'], counters=counters)
        counters['checkpoint_loads'] += 1
        counters['strict_full_state_restore_calls'] += 1
        restored = state.restore_fold1_fit_state(
            initial_saved, model, optimizer, numerical_recipe=numerical,
            feature_view_identity=view_identity, parent_order=order,
            source_commit=plan['source_commit'] if short_learning else recipe['original_source_commit'], device=device)
        counters['strict_full_state_restore_returns'] += 1
        _need(restored == {'cursor': 0, 'optimizer_updates': 0, 'skipped_batches': 0}
              and initial_saved['model_state_sha256'] == recipe['expected_initial_model_state_sha256']
              and not optimizer.state, 'EXACT_ORIGINAL_INITIAL')
        fresh = state.capture_fold1_fit_state(
            model, optimizer, numerical_recipe=numerical, feature_view_identity=view_identity,
            parent_order=order, cursor=0, optimizer_updates=0, skipped_batches=0,
            source_commit=plan['source_commit'], device=device)
        state_proofs['initial'] = _same_full_state(initial_saved, fresh,
            source=plan['source_commit'] if short_learning else recipe['original_source_commit'],
            tree_sha=tree_sha, model_sha=model_sha)
        initial_restore_exact = True
        del fresh

        def measure_short_train(label):
            def capture():
                return state.capture_fold1_fit_state(model, optimizer, numerical_recipe=numerical,
                    feature_view_identity=view_identity, parent_order=order, cursor=cursor,
                    optimizer_updates=counters['optimizer_updates'], skipped_batches=0,
                    source_commit=plan['source_commit'], device=device)
            def evaluate(left):
                raw, parents, clocks = runner._feature_batch(prepared, view, receipts,
                    rows, issues, left, left + 16, np=np)
                features = {name: torch.from_numpy(value).to(device) for name, value in raw.items()}
                with trainer._training_mha_fastpath_context(model.backbone), \
                     trainer._training_autocast_context(), trainer._training_model_finite_check_context():
                    report = evaluate_fit_batch(model, features, bound, future_bound, parents, clocks,
                        fold_id=1, pass_index=0, seed=numerical['seed'], future_weight=1.0,
                        m5_equal_observation=True, fit_batch_start=left)
                return report, parents, clocks
            def gather(parents, clocks):
                return dict(state=bound.gather_fit_sampled(parents, clocks, fold_id=1,
                    pass_index=0, seed=numerical['seed']),
                    future=future_bound.gather(parents, clocks, fold_id=1, partition='fit'))
            measured, arrays, components, digest = _short_train_panel(model, recipe=recipe,
                counters=counters, evaluate=evaluate, gather_truth=gather, capture=capture,
                tree_sha=tree_sha, priors=origin['priors'], clock_guard=clock_guard, np=np, torch=torch,
                non_m5_learning=non_m5_learning)
            if non_m5_learning and label == 'INITIAL_TRAIN':
                state_proofs['full_initial_reference'] = short.require_full_initial_reference(
                    arrays, measured['canonical_batch_losses'], counters, np=np)
            if measurements:
                _need(measured['observation_chain_sha256'] == measurements['INITIAL_TRAIN']['observation_chain_sha256'],
                      'SHORT_MATCHED_PANEL')
            stream = io.BytesIO()
            np.savez_compressed(stream, **arrays)
            _need(sum(value.nbytes for value in arrays.values()) <= recipe['max_trace_bytes']
                  and stream.tell() + counters['published_panel_bytes'] <= recipe['max_trace_bytes'],
                  'SHORT_PANEL_BYTES')
            _publish_bytes(root / (label + '.npz'), stream.getvalue())
            counters['published_panel_bytes'] += stream.tell()
            panel_captures[label] = common._hash_file(root / (label + '.npz'))
            measurements[label] = measured
            measurement_refs[label] = common._json(root / (label + '.json'), {
                'schema_version': 'gx1_ta_m5_c171_short_train_measurement_v1',
                'measurement': measured, 'capture': panel_captures[label],
                'evaluated_model_state_sha256': digest, 'optimizer_updates': counters['optimizer_updates'],
                'panel_batch_indices': recipe['panel_batch_indices'], 'canonical_loss_components': components,
                'component_definition': 'Each per-TF component is activity plus side, before the existing divisor5; future_weight1. No isolated M5-side or rest loss is claimed.',
                'evaluation_state_unchanged': True, 'test_data_used': False},
                counters, recipe['max_published_json_bytes'])

        if short_learning:
            measure_short_train('INITIAL_TRAIN')
        head_params = [(name, dict(model.named_parameters())[name]) for name in HEAD_NAMES]
        _need([tuple(value.shape) for _, value in head_params] == [(1,128),(1,),(1,128),(1,)]
              and all(value.requires_grad and value.dtype == torch.float32 for _, value in head_params),
              'EXACT_HEAD_PARAMETERS')

        def vector(grad=False):
            present = np.asarray([value.grad is not None for _, value in head_params], dtype=np.bool_)
            values = [(value.grad if grad else value) for _, value in head_params]
            arrays = [(value.detach().cpu().numpy().reshape(-1).copy()
                       if value is not None else np.zeros(parameter.numel(), np.float32))
                      for value, (_, parameter) in zip(values, head_params)]
            return np.concatenate(arrays), present

        def moments():
            means, variances, steps, present = [], [], [], []
            for _, parameter in head_params:
                entry = optimizer.state.get(parameter)
                present.append(entry is not None)
                if entry is None:
                    means.append(np.zeros(parameter.numel(),np.float32))
                    variances.append(np.zeros(parameter.numel(),np.float32))
                    steps.append(0)
                else:
                    step = float(entry['step'])
                    _need(step.is_integer() and step > 0, 'OBSERVED_ADAM_STEP')
                    steps.append(int(step))
                    for key, destination in (('exp_avg',means),('exp_avg_sq',variances)):
                        value = entry[key].detach().cpu().numpy().reshape(-1).copy()
                        _need(value.dtype == np.float32 and value.size == parameter.numel()
                              and np.isfinite(value).all(), 'OBSERVED_ADAM_MOMENT')
                        destination.append(value)
            return (np.concatenate(means),np.concatenate(variances),
                    np.asarray(steps,np.int64),np.asarray(present,np.bool_))

        def head_hook(name):
            def observe(_module, inputs):
                _need(name not in current and len(inputs) == 1, 'HEAD_OBSERVER_CALLS')
                value = inputs[0].detach()
                expected = (16,1,128) if name == 'state_representation' else (16,128)
                _need(tuple(value.shape) == expected and value.dtype == torch.float32, 'HEAD_OBSERVER_SHAPE')
                current[name] = value.reshape(16,128).cpu().numpy().copy()
                return None
            return observe

        def observe_step(_optimizer, _args, _kwargs):
            _need('head_grad_after_clip' not in current, 'ONE_ADAM_OBSERVER')
            current['head_grad_after_clip'], current['head_grad_after_clip_present'] = vector(grad=True)
            return None

        handles.extend([
            model.state_model.m5_side_head.register_forward_pre_hook(head_hook('state_representation')),
            model.future_m5_side_head.register_forward_pre_hook(head_hook('future_representation')),
            optimizer.register_step_pre_hook(observe_step)])
        timings['model_and_restore_seconds'] = time.monotonic() - model_started
        fit_started = time.monotonic()
        for left in range(0, ROWS, 16):
            clock_guard()
            current.clear()
            raw, parents, clocks = runner._feature_batch(
                prepared, view, receipts, rows, issues, left, left + 16, np=np)
            features = {name: torch.from_numpy(value).to(device) for name, value in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            current['head_before'], _ = vector()
            if short_learning:
                _need(counters['forwards'] < recipe['max_forwards']
                      and counters['training_forwards'] < recipe['max_training_forwards']
                      and counters['backwards'] < recipe['max_backwards']
                      and counters['optimizer_step_calls'] < recipe['max_private_updates'],
                      'SHORT_TRAIN_BUDGET')
                counters['training_forwards'] += 1
            counters['forwards'] += 1
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                report = evaluate_fit_batch(model, features, bound, future_bound, parents, clocks,
                    fold_id=1, pass_index=0, seed=numerical['seed'], future_weight=numerical['future_weight'],
                    m5_equal_observation=True, fit_batch_start=left)
            counters['forward_returns'] += 1
            _need(report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                  and report['loss_contract_sha256'] == recipe['expected_loss_contract_sha256']
                  and bool(torch.isfinite(report['loss'])) and report['skip_update'] is False,
                  'CANONICAL_LOSS')
            loss = float(report['loss'].detach())
            # Same bound objects, read again only because the report omits eligible.
            truths = {
                'state': bound.gather_fit_sampled(parents, clocks, fold_id=1, pass_index=0, seed=numerical['seed']),
                'future': future_bound.gather(parents, clocks, fold_id=1, partition='fit')}
            counters['extra_target_gathers'] += 2
            current.update(parents=parents.copy(), issues=clocks.copy(),
                state_query=report['state_query_k'].copy(),
                future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
            _need(np.array_equal(truths['state']['query_k'],current['state_query'])
                  and np.array_equal(current['future_query'],
                      np.broadcast_to(model.state_model.query_native_k.detach().cpu().numpy(),(16,5))),
                  'OBSERVED_QUERY_PARITY')
            for task, truth in truths.items():
                _need(np.array_equal(truth['parent_rows'],parents)
                      and np.array_equal(truth['issue_ns'],clocks), 'OBSERVED_CLOCK_PARITY')
                for key in ('allowed','mature'):
                    _need(np.array_equal(truth[key],report[task+'_'+key]), 'OBSERVED_TARGET_PARITY')
                for key in ('allowed','eligible','mature'):
                    current[task+'_'+key] = truth[key].copy()
                info = truth['eligible'] & truth['mature'] & (truth['allowed'] != 7)
                known = info & np.isin(truth['allowed'],(1,4))
                _need(info.sum() == report[task+'_count']
                      and list(info.sum(0)) == [report[task+'_per_tf_counts'][tf] for tf in TIMEFRAMES],
                      'OBSERVED_SUPPORT_PARITY')
                current[task+'_known_count'] = known.sum(0)
                current[task+'_up_count'] = (known & (truth['allowed']==4)).sum(0)
                current[task+'_raw'] = report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
            counters['backwards'] += 1
            report['loss'].backward()
            counters['backward_returns'] += 1
            current['head_grad_before_clip'], current['head_grad_before_clip_present'] = vector(grad=True)
            counters['optimizer_step_calls'] += 1
            norm = trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)
            counters['optimizer_updates'] += 1
            cursor = left + 16
            current['head_after'], _ = vector()
            (current['head_exp_avg_after'],current['head_exp_avg_sq_after'],
             current['head_step_after'],current['head_optimizer_state_present']) = moments()
            _need(set(current) >= {'state_representation','future_representation','head_grad_after_clip'},
                  'OBSERVER_COMPLETENESS')
            current['loss'] = np.asarray(loss,np.float64)
            current['gradient_norm'] = np.asarray(float(norm),np.float64)
            current['start_cursor'] = np.asarray(left,np.int64)
            current['end_cursor'] = np.asarray(left+16,np.int64)
            records.append(dict(current))
            optimizer.zero_grad(set_to_none=True)
            cursor = left + 16
            state._progress(cursor,counters['optimizer_updates'],0,order,numerical)
            if left == 0 and not short_learning:
                first_step_exact = loss == FIRST_LOSS and float(norm) == FIRST_NORM
                _need(first_step_exact, 'ORIGINAL_FIRST_STEP')
            del report, features, raw, truths
        timings['fit_seconds'] = time.monotonic() - fit_started
        for handle in handles:
            handle.remove()
        handles.clear()
        clock_guard()
        _need(cursor == ROWS and counters['optimizer_updates'] == UPDATES, 'PREFIX_COMPLETE')
        if short_learning:
            if non_m5_learning:
                _need(tree_sha(trainer._attended_session_rng_state(device=device)) == short.FINAL_RNG_SHA256,
                      'NON_M5_MATCHED_TERMINAL_RNG')
                state_proofs['original_training_rng_endpoints_matched'] = True
            measure_short_train('FINAL_TRAIN')
            evaluation_state_unchanged = True
        else:
            target_saved = common._load_state(target_ref, torch=torch,
                maximum=recipe['max_checkpoint_bytes'], counters=counters)
            counters['checkpoint_loads'] += 1
            _need((target_saved['cursor'],target_saved['optimizer_updates'],target_saved['skipped_batches'])
                  == (ROWS,UPDATES,0), 'ORIGINAL_P4096_PROGRESS')
        actual = state.capture_fold1_fit_state(
            model,optimizer,numerical_recipe=numerical,feature_view_identity=view_identity,parent_order=order,
            cursor=cursor,optimizer_updates=UPDATES,skipped_batches=0,
            source_commit=plan['source_commit'],device=device)
        if short_learning:
            final_state_ref = common._save_state(output / 'FINAL_C171_SHORT_STATE.pt', actual,
                torch=torch, maximum=recipe['max_checkpoint_bytes'], counters=counters)
            counters['checkpoint_writes'] += 1
            state_proofs['final'] = _state_identity(actual)
        else:
            state_proofs['P004096'] = _same_full_state(target_saved,actual,
                source=recipe['original_source_commit'],tree_sha=tree_sha,model_sha=model_sha)
        final_exact = True
        expected_counts = {'model_constructions':1,'optimizer_constructions':1,
            'forwards':UPDATES,'backwards':UPDATES,'optimizer_step_calls':UPDATES,
            'optimizer_updates':UPDATES,'checkpoint_loads':2,
            'strict_full_state_restore_calls':1,'checkpoint_writes':0}
        if short_learning:
            expected_counts.update(forwards=288,forward_returns=288,backward_returns=UPDATES,
                model_construction_returns=1,training_forwards=256,evaluation_forwards=32,
                checkpoint_writes=2,staged_state_load_calls=2,staged_state_load_returns=2,
                resume_state_load_calls=2,resume_state_load_returns=2,
                strict_full_state_restore_returns=1,head_tensor_imports=4,
                archive_decodes=1,loss_component_owner_calls=32)
        if non_m5_learning:
            expected_counts.update(model_constructions=2,model_construction_returns=2,forwards=768,forward_returns=768,
                evaluation_forwards=512,head_tensor_imports=16,loss_component_owner_calls=512,initial_rng_imports=1,
                extra_target_gathers=1536)
        _need(all(counters[key] == value for key,value in expected_counts.items()),
              'EXACT_EXECUTION_COUNTS')
        _need(initial_saved['contract_sha256'] == tree_sha({
            k:v for k,v in initial_saved.items() if k!='contract_sha256'}), 'INITIAL_PAYLOAD_UNCHANGED')
        del actual
        _bookend(recipe,plan,policy_path,policy_file_sha256)
        clock_guard()
        source_unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        if prepared is not None:
            try:
                from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
                base._release_pages(provider,prepared)
            except BaseException as exc:
                error = (error + '; ' if error else '') + 'cleanup: ' + type(exc).__name__ + ': ' + str(exc)
        prepared = provider = model = optimizer = initial_saved = target_saved = None
        gc.collect()

    # Preserve completed observations on failure. The legacy route never writes weights.
    try:
        if records:
            import numpy as np
            from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
            arrays = {key:np.stack([record[key] for record in records]) for key in records[0]}
            _need(sum(value.nbytes for value in arrays.values()) <= recipe['max_trace_bytes'], 'TRACE_RAW_BYTES')
            stream = io.BytesIO()
            np.savez_compressed(stream,**arrays)
            _need(stream.tell() + counters.get('published_panel_bytes', 0) <= recipe['max_trace_bytes'], 'TRACE_FILE_BYTES')
            _publish_bytes(root/'TRACE.npz',stream.getvalue())
            trace_ref = common._hash_file(root/'TRACE.npz')
    except BaseException as exc:
        error = (error + '; ' if error else '') + 'trace publication: ' + type(exc).__name__ + ': ' + str(exc)
    initial_ok = initial_restore_exact if short_learning else first_step_exact
    outcome = 'COMPLETE' if initial_ok and final_exact and source_unchanged and error is None else 'FAILED'
    final_ref = (final_state_ref if short_learning else target_ref) if outcome == 'COMPLETE' else None
    payload = {
        'schema_version':scope.TA_M5_ORIGINAL_CUDA_PREFIX_RESULT_SCHEMA,
        'outcome':outcome,
        'decision':'PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE' if outcome=='COMPLETE' else 'FAIL_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE',
        'recipe':window['recipe'], 'source_commit':plan['source_commit'],
        'original_source_commit':recipe['original_source_commit'], 'trace_worker':recipe['trace_worker'],
        'parent_cursor':cursor, 'optimizer_updates':counters['optimizer_updates'],
        'physical_optimizer_updates':counters['optimizer_updates'], 'skipped_batches':0,
        'parent_order_sha256':order_sha, 'initial_state':initial_ref, 'targetref':target_ref,
        'final_state':final_ref, 'first_step_exact':first_step_exact, 'final_full_state_exact':final_exact,
        'full_state_comparisons':state_proofs, 'trace':trace_ref, 'traced_steps':len(records),
        'head_parameter_names':list(HEAD_NAMES), 'head_parameter_shapes':[[1,128],[1],[1,128],[1]],
        'head_parameter_spans':[[0,128],[128,129],[129,257],[257,258]],
        'first_step_reference':{'loss':FIRST_LOSS,'gradient_norm':FIRST_NORM},
        'counts':counters, 'timings':dict(timings,elapsed_seconds=time.monotonic()-started),
        'source_unchanged':source_unchanged, 'error':error, 'limits':limits,
        'checkpoint_writes':0, 'canonical_resume_authority':False, 'test_data_used':False,
        'initial_evaluation_replayed':False, 'intermediate_evaluation_forwards':0,
        'qualification_weights_for_main_fit':False, 'predictive_value_claimed':False,
        'scope_limit':'Observer-only original CUDA prefix replay. COMPLETE requires full P4096 numerical-state parity; source metadata and serialization bytes are not claimed identical. No new checkpoint, selection, later scoring or promotion.',
    }
    if short_learning:
        for key in ('targetref','first_step_exact','final_full_state_exact','first_step_reference',
                    'initial_evaluation_replayed','intermediate_evaluation_forwards'):
            payload.pop(key)
        payload.update(schema_version=short.RESULT_SCHEMA,
            decision='PASS_TA_M5_C171_SHORT_LEARNING' if outcome == 'COMPLETE' else 'FAIL_TA_M5_C171_SHORT_LEARNING',
            initial_restore_exact=initial_restore_exact,
            final_state_validated=final_exact, retained_final_state=final_state_ref,
            initial_model_state_sha256=state_proofs.get('initial', {}).get('replayed', {}).get('model_state_sha256'),
            final_model_state_sha256=state_proofs.get('final', {}).get('model_state_sha256'),
            evaluation_state_unchanged=evaluation_state_unchanged,
            optimizer_initialization=recipe['optimizer_initialization'], import_proof=import_proof,
            checkpoint_writes=counters['checkpoint_writes'], panel_batch_indices=recipe['panel_batch_indices'],
            measurement_refs=measurement_refs, measurements=measurements, panel_captures=panel_captures,
            all_panels_complete=set(measurements) == {'INITIAL_TRAIN','FINAL_TRAIN'},
            canonical_loss_unchanged=source_unchanged, candidate_selection=False, later_evaluation_forwards=0,
            parameter_change_observed=(state_proofs['final']['model_state_sha256']
                != recipe['expected_initial_model_state_sha256']) if 'final' in state_proofs else None,
            scope_limit='One unchanged-loss 256-update chronological TRAIN prefix from c171 with fresh empty AdamW; all active parameters and genuine features retained. Same16 intact B16 TRAIN groups before/after, not256 independent samples or held-out generalization. Fresh initial and final states are evidence only; no canonical future resume, selection or predictive-value authority.')
    if non_m5_learning:
        payload.update(decision='PASS_TA_NON_M5_SHORT_LEARNING' if outcome=='COMPLETE' else 'FAIL_TA_NON_M5_SHORT_LEARNING',
            dedicated_non_m5_side=True,full_initial_reference=state_proofs.get('full_initial_reference'),
            original_training_rng_endpoints_matched=state_proofs.get('original_training_rng_endpoints_matched',False),
            scope_limit='One matched256-update/4096chronologicalTRAIN comparison of independent non-M5 sides; original features/targets/loss/AdamW, fresh empty optimizer and matched original initial RNG. Full used4096-row INITIAL/FINAL measurements; no heldout/full-history/selection/promotion claim.')
    result = common._json(root/'RESULT.json',payload,counters,recipe['max_published_json_bytes'])
    common._json(root/'TERMINAL.json', {
        'status':outcome,'exit_code':0 if outcome=='COMPLETE' else 1,'result':result,
        'recipe':window['recipe'],'source_unchanged':source_unchanged,'test_data_used':False,
        'no_relaunch_under_this_plan':True,
        **({'initial_state': initial_ref, 'final_state': final_ref, 'measurement_refs': measurement_refs,
            'optimizer_initialization': recipe['optimizer_initialization'],
            'initial_restore_exact': initial_restore_exact,
            'evaluation_state_unchanged': evaluation_state_unchanged} if short_learning else {})},
        counters,recipe['max_published_json_bytes'])
    cursor_ref = common._json(Path(window['campaign_cursor_path']),authority.build_research_cursor(
        recipe_ref=window['recipe'],outcome=outcome,optimizer_updates=counters['optimizer_updates'],
        parent_cursor=cursor,parent_order_sha256=order_sha,full_state=final_ref,result=result,batch_size=16),
        counters,recipe['max_published_json_bytes'])
    progress = authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=cursor_ref,optimizer_updates=counters['optimizer_updates'],
        parent_cursor=cursor,parent_order_sha256=order_sha,outcome=outcome)
    common._json(progress_path,progress,counters,recipe['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],
        invocation=invocation,expected_selection_receipt_sha256=None,verify_file=True)
    if error is not None:
        raise RuntimeError(error)
    return {'result':result,'checkpoint_pointer':cursor_ref,'progress_path':str(progress_path)}
