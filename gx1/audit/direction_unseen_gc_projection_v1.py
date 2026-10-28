"""Measure the frozen model's direct response to previously unsupported GC inputs.

Only five normalized values at one SMC projection are suppressed. Selection is
based exclusively on input support; existing chronological outputs must replay
byte for byte. No fitting, trading, sealed TEST access, or feature deletion.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from gx1.audit import direction_gc_mtf_followup_v1 as followup
from gx1.audit import direction_gc_mtf_intervention_v1 as intervention
from gx1.scripts.audit_direction_unseen_gc_support_v1 import normalized_values

REPO = followup.REPO
ROOT = followup.ROOT
STAGE = 'independent_direction_unseen_gc_projection_20261028_001'
FIELDS = (
    'gc_d1_observed_proxy_ob_bull_age_bars',
    'gc_d1_observed_proxy_ob_bear_first_retouch',
    'gc_d1_observed_proxy_ob_bear_low',
    'gc_d1_observed_proxy_ob_bear_high',
    'gc_d1_observed_proxy_ob_bear_origin_volume_contracts',
)
SOURCE_PATHS = followup.SOURCE_PATHS + (
    'gx1/audit/direction_unseen_gc_projection_v1.py',
    'gx1/scripts/audit_direction_unseen_gc_support_v1.py',
    'gx1/features/gc_native_context_v1.py',
    'tests/test_direction_unseen_gc_projection_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_UNSEEN_GC_PROJECTION_' + reason)


def select_batches(active, batch_size=16):
    """Original whole batches containing any nonzero input, with no labels."""
    require(isinstance(active, np.ndarray) and active.dtype == np.bool_ and active.ndim == 1
        and type(batch_size) is int and batch_size > 0 and len(active) > 0, 'SELECTION_INPUT')
    starts = np.unique(np.flatnonzero(active) // batch_size) * batch_size
    return [np.arange(i, min(i + batch_size, len(active)), dtype=np.int64) for i in starts]


def install_projection_ablation(module, positions, expected):
    """Clone only the target Linear input; preserve every non-target coordinate."""
    require(isinstance(module, torch.nn.Linear) and len(positions) == len(FIELDS)
        and len(set(positions)) == len(FIELDS) and all(0 <= i < module.in_features for i in positions)
        and isinstance(expected, np.ndarray) and expected.ndim == 2 and expected.shape[1] == len(FIELDS), 'HOOK_SPEC')
    calls = {'count': 0}
    def replace(_module, args):
        require(len(args) == 1 and isinstance(args[0], torch.Tensor)
            and args[0].ndim == 2 and args[0].shape == (len(expected), module.in_features), 'HOOK_INPUT')
        original = args[0]
        actual = original[:, positions].detach().cpu().numpy()
        require(np.allclose(actual, expected, rtol=2e-6, atol=2e-6)
            and np.array_equal(actual != 0, expected != 0), 'MODEL_NORMALIZATION_PARITY')
        altered = original.clone(); altered[:, positions] = 0
        calls['count'] += 1
        return (altered,)
    return module.register_forward_pre_hook(replace), calls


def summarize(original, ablated, active):
    require(original.dtype == ablated.dtype == np.float32 and original.shape == ablated.shape
        and original.ndim == 2 and original.shape[1] == 60 and active.shape == (len(original),)
        and active.dtype == np.bool_ and active.any() and np.isfinite(original).all() and np.isfinite(ablated).all(), 'SUMMARY_INPUT')
    require(np.array_equal(original[~active], ablated[~active]), 'INACTIVE_ROWS_IDENTICAL')
    result = {}
    for label, subset in (('all_selected', np.ones(len(active), dtype=np.bool_)), ('active_inputs', active)):
        delta = ablated[subset].astype(np.float64) - original[subset].astype(np.float64)
        result[label] = {'parents': int(subset.sum()), 'mean_absolute_logit_change_all_60': float(np.abs(delta).mean()),
            'max_absolute_logit_change_all_60': float(np.abs(delta).max()),
            'mean_absolute_logit_change_5m': float(np.abs(delta[:, 4]).mean()),
            'max_absolute_logit_change_5m': float(np.abs(delta[:, 4]).max()),
            'signed_mean_logit_change_5m': float(delta[:, 4].mean()),
            'hard_direction_flips_5m': int(np.count_nonzero((original[subset, 4] > 0) != (ablated[subset, 4] > 0))),
            'hard_direction_flips_all_60': int(np.count_nonzero((original[subset] > 0) != (ablated[subset] > 0)))}
    return result


def run(*, plan_path, plan_sha256, device, out_dir):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.features.gc_native_context_v1 import masking_pairs
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle

    sys.dont_write_bytecode = True
    require(device == 'cuda' and not torch.cuda.is_initialized(), 'EXPLICIT_CUDA')
    plan_ref = {'path': str(candle.regular(plan_path)), 'sha256': plan_sha256}
    source_head, branch, dirty = candle.git_state()
    require(branch == 'work/gx1-current' and not dirty, 'INITIAL_SOURCE')
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir',
            'wall_seconds', 'batch_size', 'original_source_commit', 'test_data_used'}
            and plan['schema_version'] == 'gx1_direction_unseen_gc_projection_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 1800 and plan['batch_size'] == 16 and plan['test_data_used'] is False, 'PLAN')
        require(out_dir == Path(plan['output_dir']) and out_dir.parent == ROOT
            and out_dir.name == 'DIRECTION_UNSEEN_GC_PROJECTION_20261028_001'
            and ROOT.resolve(strict=True) == ROOT and not out_dir.exists() and not out_dir.is_symlink(), 'OUTPUT_SCOPE')
        require(plan['design']['path'] == str(REPO / 'docs/DIRECTION_UNSEEN_GC_PROJECTION_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_direction_unseen_gc_projection_design_v1'
            and design['inputs'] == plan['inputs'] and design['fields'] == list(FIELDS)
            and design['test_data_used'] is False, 'DESIGN')
        policy_path = REPO / 'NEXT_RUN_POLICY.json'
        policy_sha = common._hash_file(str(policy_path))['sha256']
        policy_ref = {'path': str(policy_path), 'sha256': policy_sha}
        policy = binding.read_bound_json(policy_path, policy_sha); gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['checkpoint_reads_authorized'] is True and gate['model_forwards_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
            'full_val_allowed', 'native_launch_authorized')) and all(gate[k] is False for k in
            ('optimizer_steps_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS) and all(ref['path'] == str(REPO / name)
            for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        keys = {'recipe', 'fit_result', 'checkpoint', 'oof_report', 'oof_predictions', 'corrected_target',
            'fold_design', 'source_transition', 'synthetic_test_terminal', 'synthetic_test_plan',
            'context_mmap', 'prefix_normalization', 'support_plan', 'support_result', 'sample_pair',
            'context_followup_result', 'context_verify'}
        require(set(plan['inputs']) == keys, 'INPUT_SET')
        payload_names = ('checkpoint', 'oof_predictions', 'corrected_target', 'context_mmap', 'sample_pair')
        meta = {key: binding.read_bound_json(Path(plan['inputs'][key]['path']), plan['inputs'][key]['sha256'])
            for key in keys - set(payload_names)}
        launch, result, oof = (meta[k] for k in ('recipe', 'fit_result', 'oof_report'))
        support, support_plan, norm = (meta[k] for k in ('support_result', 'support_plan', 'prefix_normalization'))
        previous, verification = meta['context_followup_result'], meta['context_verify']
        require(result['outcome'] == 'COMPLETE' and result['test_data_used'] is False
            and result['recipe'] == plan['inputs']['recipe'] and result['full_state'] == plan['inputs']['checkpoint']
            and result['oof_measurement'] == plan['inputs']['oof_report'] and oof['predictions'] == plan['inputs']['oof_predictions']
            and oof['fold'] == result['fold'] == launch['fold'] and oof['test_data_used'] is False
            and launch['numerical_recipe']['dense_minute_supervision'] is True, 'FIT_LINEAGE')
        require(support['status'] == 'COMPLETE' and support['fields'] == 33 and support['train_normalized_zero_reproduced'] is True
            and support['plan'] == plan['inputs']['support_plan'] and support['context_mmap'] == plan['inputs']['context_mmap']
            and support['test_data_used'] is False and support_plan['inputs']['prefix_normalization'] == plan['inputs']['prefix_normalization']
            and support_plan['inputs']['sample_pair'] == plan['inputs']['sample_pair']
            and support_plan['inputs']['corrected_target'] == plan['inputs']['corrected_target'], 'SUPPORT_LINEAGE')
        require(previous['status'] == verification['status'] == 'COMPLETE' and previous['test_data_used'] is False
            and verification['test_data_used'] is False and verification['followup_result'] == plan['inputs']['context_followup_result']
            and verification['paired_outputs'] == plan['inputs']['sample_pair']
            and {k: previous['paired_outputs'][k] for k in ('path', 'sha256')} == plan['inputs']['sample_pair'], 'PRIOR_PAIR_LINEAGE')
        require(result['feature_view_identity']['prefix_normalization']['binding'] == plan['inputs']['prefix_normalization'], 'NORMALIZATION_LINEAGE')
        intervention.require_synthetic_receipts(plan['inputs'], result, meta['synthetic_test_terminal'], meta['synthetic_test_plan'])
        fold = followup.later_fold(meta['fold_design'], launch['fold'])
        old = subprocess.check_output(['git', 'show', plan['original_source_commit'] + ':' + followup.OWNER], cwd=REPO)
        freeze_launch = followup.source_transition(launch, meta['source_transition'], old, (REPO / followup.OWNER).read_bytes())
        require(len(freeze_launch['source_owners']) == 56, 'FIT_SOURCE_COUNT')
        for ref in freeze_launch['source_owners'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='fit source')
        surface = norm['base_normalization']['surfaces']['ctx_cont']; names = surface['field_names']
        indices = [names.index(name) for name in FIELDS]; pairs = dict(masking_pairs(names)); masks = [pairs[i] for i in indices]
        require(len(indices) == len(set(indices)) == 5 and all(surface['train_transformed_min'][i] ==
            surface['train_transformed_max'][i] == 0 for i in indices) and not (set(indices) & set(masks)), 'FIELD_SUPPORT')
        centers = [surface['center'][i] for i in indices]; scales = [surface['scale'][i] for i in indices]
        binary = [bool(surface['binary_mask'][i] or surface['categorical_mask'][i]) for i in indices]
        control_snapshot = binding.capture_verified_file_snapshot()
    hardware, lock = cap.require_guarded_cuda_producer_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == hardware['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out_dir.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out_dir / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': source_head, 'execution': hardware, 'lock': lock})
    intervention._install_runtime_guard(out_dir, (plan['inputs']['synthetic_test_terminal']['path'],
        plan['inputs']['synthetic_test_plan']['path']))
    start = time.monotonic(); previous_profile = sys.getprofile(); previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(*unused):
        raise RuntimeError('DIRECTION_UNSEEN_GC_PROJECTION_WALL_LIMIT')
    def forbid_updates(frame, event, arg):
        if event == 'call':
            name, path = frame.f_code.co_name, frame.f_code.co_filename.replace('\\', '/')
            require(not (('/torch/optim/' in path and name in ('step', '_init_group'))
                or ('/torch/autograd/' in path and name in ('backward', '_engine_run_backward'))), 'FORBIDDEN_UPDATE')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(forbid_updates)
    stage = out_dir / '.OUTPUT.staging'; handle = None
    try:
        stage.mkdir(exist_ok=False)
        context = window._launch_metadata(launch, common)
        require(context['fitted'] == norm, 'FITTED_NORMALIZATION')
        refs = [plan['inputs'][name] for name in payload_names]
        require(len({ref['path'] for ref in refs}) == len(refs), 'DISTINCT_PAYLOADS')
        for ref in refs:
            before = common._stat(ref['path']); require(common._hash_file(ref['path']) == ref, 'PAYLOAD_SHA')
            context['info']['metadata'][ref['path']] = {'ref': dict(ref), 'stat': before}
        freeze_binding = {'source_head': source_head, 'policy': policy_ref}
        common._freeze(freeze_launch, freeze_binding, context)
        denied = common._install_read_guard(context, out_dir, out_dir, dense_target=context['dense_target'])
        prepared, targets, view, _, _ = window._prepare(context, launch, np, dense_minutes=True)
        with np.load(plan['inputs']['oof_predictions']['path'], allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'logit_up')}
        with np.load(plan['inputs']['sample_pair']['path'], allow_pickle=False) as archive:
            sample = {key: archive[key].copy() for key in ('parent_rows', 'decision_ns', 'original_logit_up')}
        with np.load(plan['inputs']['corrected_target']['path'], allow_pickle=False) as archive:
            axes = {key: archive[key].copy() for key in ('parent_rows', 'bar_start_ns', 'decision_ns')}
        for name in axes:
            require(np.array_equal(axes[name], prepared['arrays'][name]), 'AXIS:' + name)
        origin = launch['fold']; origin_slice = slice(origin['oof_start_row'], origin['oof_end_row'])
        for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            require(np.array_equal(saved[name], axes[name][origin_slice]), 'OOF_AXIS:' + name)
        require(saved['logit_up'].dtype == np.float32 and saved['logit_up'].shape == (17288, 60), 'OOF_LOGITS')
        positions = np.searchsorted(axes['parent_rows'], sample['parent_rows'])
        require(len(positions) == 1760 and np.all(positions >= fold['oof_start_row'])
            and np.all(positions < fold['oof_end_row']) and np.array_equal(axes['parent_rows'][positions], sample['parent_rows'])
            and np.array_equal(axes['decision_ns'][positions], sample['decision_ns'])
            and sample['original_logit_up'].dtype == np.float32 and sample['original_logit_up'].shape == (1760, 60), 'SAMPLE_AXIS')
        raw_context = np.memmap(plan['inputs']['context_mmap']['path'], mode='r', dtype=np.float32, shape=(118249, 2542))
        relevant = np.asarray(indices + masks, dtype=np.int64)
        require(not raw_context.flags.writeable and prepared['current']['ctx_cont'].shape == raw_context.shape
            and np.array_equal(prepared['current']['ctx_cont'][:, relevant], raw_context[:, relevant]), 'ACTUAL_CONTEXT_SOURCE')
        groups = {'q1': np.arange(origin['oof_start_row'], origin['oof_end_row'], dtype=np.int64), 'q2': positions}
        normalized = {}; selected = {}
        for name, rows in groups.items():
            normalized[name] = normalized_values(raw_context[np.ix_(rows, indices)], raw_context[np.ix_(rows, masks)], centers, scales, binary)
            active = np.any(normalized[name] != 0, axis=1); selected[name] = (active, select_batches(active))
        require(int(selected['q1'][0].sum()) == 275 and len(selected['q1'][1]) == 18
            and int(selected['q2'][0].sum()) == 80 and len(selected['q2'][1]) == 5, 'EXPECTED_INPUT_COHORTS')
        del raw_context
        trainer._set_deterministic(launch['numerical_recipe']['seed'], torch.device('cuda', 0), launch['numerical_recipe']['precision_policy'])
        torch.set_num_threads(1); require(torch.get_num_threads() == 1, 'PRODUCER_NUMERICAL_THREADS')
        model = build_native_direction_research_model(view, prefix_normalization=context['fitted'],
            constructor_metadata=context['constructor'], device=torch.device('cuda', 0))
        require(model.feature_view_identity == result['feature_view_identity'], 'VIEW_IDENTITY')
        state = torch.load(plan['inputs']['checkpoint']['path'], map_location='cpu', weights_only=True)
        require(isinstance(state, dict) and set(state) == state_owner.STATE_KEYS
            and state['contract_sha256'] == state_owner._tree_sha({k: v for k, v in state.items() if k != 'contract_sha256'})
            and state['contract_sha256'] == result['durable_checkpoint']['contract_sha256']
            and state['model_state_sha256'] == state_owner.canonical_model_state_sha256(state['model_state'])
            and state['binding']['feature_view_identity'] == model.feature_view_identity
            and state['cursor'] == result['parent_cursor'] and state['optimizer_updates'] == result['optimizer_updates'], 'CHECKPOINT_SEAL')
        model.load_state_dict(state['model_state'], strict=True)
        with torch.no_grad():
            for name, buffer in model.named_buffers(): buffer.copy_(state['buffer_state'][name].to(buffer.device))
        require(state_owner.canonical_model_state_sha256(model.state_dict()) == state['model_state_sha256']
            and state_owner._tree_sha(dict(model.named_buffers())) == state['buffer_state_sha256'], 'LOADED_STATE')
        model.backbone.require_input_normalization_state(); model._require_binding(); model.eval()
        model_sha, buffer_sha = state['model_state_sha256'], state['buffer_state_sha256']; del state
        route = model.backbone.specialist_ctx_cont_idx_smc_liquidity_encoder.detach().cpu().tolist()
        require(all(route.count(i) == 1 for i in indices), 'SMC_OWNER_ROUTE')
        route_positions = [route.index(i) for i in indices]
        projection = model.backbone.specialist_ctx_cont_proj['smc_liquidity_encoder']
        require(projection.in_features == len(route), 'PROJECTION_WIDTH')
        counters = {'forward_calls': 0, 'forward_returns': 0, 'forward_kind_calls': {}}
        ceiling = 2 * sum(len(value[1]) for value in selected.values())
        summaries = {}; arrays = {}
        for name, rows in groups.items():
            active, batches = selected[name]; original = []; ablated = []; kept = []
            reference = saved['logit_up'] if name == 'q1' else sample['original_logit_up']
            for batch in batches:
                require(time.monotonic() - start < plan['wall_seconds'], 'WALL_LIMIT')
                parent_ids = axes['parent_rows'][rows[batch]]
                raw, target = window._batch(prepared, targets, view, parent_ids, np)
                require(np.array_equal(raw['features']['ctx_cont'][:, relevant],
                    prepared['current']['ctx_cont'][np.ix_(rows[batch], relevant)]), 'BATCH_SOURCE_PARITY')
                features, _ = window._device_batch(raw, target, torch, torch.device('cuda', 0))
                z0 = window._forward(model, features, kind=name + '_ORIGINAL', counters=counters,
                    ceiling=ceiling, torch=torch).detach().cpu().numpy().copy()
                intervention.require_exact_replay(z0, np.ascontiguousarray(reference[batch]))
                handle, calls = install_projection_ablation(projection, route_positions, normalized[name][batch])
                try:
                    z1 = window._forward(model, features, kind=name + '_ABLATION', counters=counters,
                        ceiling=ceiling, torch=torch).detach().cpu().numpy().copy()
                finally:
                    handle.remove(); handle = None
                require(calls['count'] == 1 and np.isfinite(z0).all() and np.isfinite(z1).all(), 'HOOK_ACCOUNTING')
                original.append(z0); ablated.append(z1); kept.append(batch)
            kept = np.concatenate(kept); z0 = np.concatenate(original); z1 = np.concatenate(ablated)
            group_active = np.ascontiguousarray(active[kept]); summaries[name] = summarize(z0, z1, group_active)
            arrays.update({name + '_parent_rows': np.ascontiguousarray(axes['parent_rows'][rows[kept]]),
                name + '_decision_ns': np.ascontiguousarray(axes['decision_ns'][rows[kept]]), name + '_active': group_active,
                name + '_original_logit_up': z0, name + '_ablated_logit_up': z1})
        require(counters['forward_calls'] == counters['forward_returns'] == ceiling == 46 and not denied, 'CALL_ACCOUNTING')
        require(state_owner.canonical_model_state_sha256(model.state_dict()) == model_sha
            and state_owner._tree_sha(dict(model.named_buffers())) == buffer_sha
            and all(parameter.grad is None for parameter in model.parameters()), 'MODEL_STATE_UNCHANGED')
        pair_path = stage / 'PAIRED_OUTPUTS.npz'
        with pair_path.open('xb') as f:
            np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        with np.load(pair_path, allow_pickle=False) as a:
            require(set(a.files) == set(arrays) and all(np.array_equal(a[k], value) and a[k].dtype == value.dtype
                for k, value in arrays.items()), 'PAIR_ROUNDTRIP')
        pair_sha = common._hash_file(str(pair_path))['sha256']
        common._freeze(freeze_launch, freeze_binding, context); control_snapshot.require_unchanged()
        require(cap.require_capped_lock_ancestry() == lock and not denied, 'FINAL_SOURCE_LOCK')
        output = {'schema_version': 'gx1_direction_unseen_gc_projection_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': source_head, 'model_state_sha256': model_sha, 'fields': list(FIELDS),
            'physical_indices': indices, 'projection_indices': route_positions, 'batch_size': 16,
            'selected_batches': {'q1': 18, 'q2': 5}, 'metrics': summaries,
            'paired_outputs': {'path': str(out_dir / 'OUTPUT/PAIRED_OUTPUTS.npz'), 'sha256': pair_sha},
            'all_original_batches_byte_identical': True, 'actual_context_source_and_normalization_verified': True,
            'intervention': 'Only five normalized values within the SMC current-context Linear input are set to zero; masks, raw inputs, local aliases, GC history and all other projections remain unchanged.',
            'limits': 'Input-selected reused development cohorts; this internal perturbation may be out of distribution. Local aliases and other routes remain. Magnitude and hard-sign flips do not establish prediction quality or economic edge.',
            'forward_calls': counters['forward_calls'], 'optimizer_steps': 0, 'model_unchanged': True,
            'test_data_used': False, 'economic_edge_qualified': False}
        publish(stage / 'RESULT.json', output)
        publish(stage / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
            'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - start, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out_dir / 'OUTPUT')
        publish(out_dir / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'result_path': str(out_dir / 'OUTPUT/RESULT.json')})
        return output
    except Exception as exc:
        publish(out_dir / 'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False})
        raise
    finally:
        if handle is not None: handle.remove()
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)
