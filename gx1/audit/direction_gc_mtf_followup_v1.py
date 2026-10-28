"""Measure the frozen fold1 direction model's GC-history effect in later fold2.

No optimizer, refit, model selection or TEST access. The first saved fold1
batch must reproduce exactly. Both arms and corrected labels are persisted for
independent paired-score verification on the prespecified later sample.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from gx1.audit import direction_gc_mtf_intervention_v1 as intervention

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_gc_mtf_later_followup_20261028_001'
CONTEXT_STAGE = 'independent_direction_gc_context_followup_20261028_001'
OWNER = 'gx1/scripts/run_entry_native_direction_research_window_v1.py'
SOURCE_PATHS = (
    'gx1/audit/direction_gc_mtf_followup_v1.py',
    'gx1/audit/direction_gc_mtf_intervention_v1.py',
    'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'scripts/gx1_guarded_trainer_exec.sh',
    'tests/test_direction_gc_mtf_followup_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_GC_MTF_LATER_FOLLOWUP_' + reason)


def source_transition(launch, proof, original_bytes, current_bytes):
    """Allow only the documented own-output-directory guard change."""
    old = launch['source_owners'][OWNER]
    require(proof.get('schema_version') == 'gx1_direction_gc_output_directory_guard_fix_v1'
        and proof.get('status') == 'SOURCE_AND_SYNTHETIC_TESTS_ONLY'
        and proof.get('source_before') == old and proof.get('test_data_used') is False
        and proof.get('model_forwards') == proof.get('optimizer_steps') == 0, 'TRANSITION_PROOF')
    new = proof.get('source_after')
    require(isinstance(new, dict) and set(new) == {'path', 'sha256'} and new['path'] == old['path']
        and hashlib.sha256(original_bytes).hexdigest() == old['sha256']
        and hashlib.sha256(current_bytes).hexdigest() == new['sha256'], 'TRANSITION_HASH')
    before = b"    directories={plan['native_cache_dir'],str(Path(plan['gc_surfaces']['path']).parent)}\n"
    after = b"    directories={plan['native_cache_dir'],str(Path(plan['gc_surfaces']['path']).parent),str(output)}\n"
    require(original_bytes.count(before) == 1 and original_bytes.replace(before, after, 1) == current_bytes,
        'TRANSITION_SINGLE_CHANGE')
    return {**launch, 'source_owners': {**launch['source_owners'], OWNER: dict(new)}}


def later_fold(design, origin_fold):
    """Use the already declared second development period after origin fold1."""
    require(isinstance(design.get('folds'), list) and len(design['folds']) >= 2, 'FOLD_DESIGN')
    fold = design['folds'][1]
    require(fold == {'fold_id': 2, 'prefix_rows': 64890, 'oof_start_row': 64890,
        'oof_end_row': 82427, 'train_cutoff_ns': 1743465600000000000,
        'oof_cutoff_ns': 1751328000000000000}
        and origin_fold['oof_end_row'] == fold['oof_start_row']
        and origin_fold['oof_cutoff_ns'] == fold['train_cutoff_ns']
        and origin_fold['train_cutoff_ns'] < fold['train_cutoff_ns'], 'LATER_FOLD')
    return fold


def save_paired(path, *, parents, decisions, terminal, observed, sign, original, ablated):
    """Persist the exact scored paired rows, without weights or raw features."""
    n = len(parents)
    require(parents.dtype == decisions.dtype == np.int64 and parents.shape == decisions.shape == (n,)
        and terminal.dtype == np.int64 and observed.dtype == np.bool_ and sign.dtype == np.int8
        and original.dtype == ablated.dtype == np.float32
        and terminal.shape == observed.shape == sign.shape == original.shape == ablated.shape == (n, 60)
        and n > 0 and np.all(np.diff(parents) > 0) and np.all(np.diff(decisions) > 0)
        and np.isfinite(original).all() and np.isfinite(ablated).all(), 'PAIRED_ARRAYS')
    with path.open('xb') as handle:
        np.savez_compressed(handle, parent_rows=parents, decision_ns=decisions, terminal_time_ns=terminal,
            observed=observed, sign=sign, original_logit_up=original, ablated_logit_up=ablated)
        handle.flush(); os.fsync(handle.fileno())
    with np.load(path, allow_pickle=False) as archive:
        expected = {'parent_rows': parents, 'decision_ns': decisions, 'terminal_time_ns': terminal,
            'observed': observed, 'sign': sign, 'original_logit_up': original, 'ablated_logit_up': ablated}
        require(set(archive.files) == set(expected) and len(archive.files) == len(expected)
            and all(archive[k].dtype == v.dtype and archive[k].shape == v.shape
                and np.ascontiguousarray(archive[k]).tobytes() == np.ascontiguousarray(v).tobytes()
                for k, v in expected.items()), 'PAIRED_ROUNDTRIP')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def run(*, plan_path, plan_sha256, device, out_dir, mode='history'):
    require(mode in ('history', 'context_and_history'), 'EXPERIMENT_MODE')
    context_mode = mode == 'context_and_history'
    experiment_stage = CONTEXT_STAGE if context_mode else STAGE
    experiment_output = ('DIRECTION_GC_CONTEXT_FOLLOWUP_20261028_001' if context_mode
        else 'DIRECTION_GC_MTF_LATER_FOLLOWUP_20261028_001')
    experiment_design = ('DIRECTION_GC_CONTEXT_FOLLOWUP_20261028.json' if context_mode
        else 'DIRECTION_GC_MTF_LATER_FOLLOWUP_20261028.json')
    experiment_schema = 'gx1_direction_gc_context_followup' if context_mode else 'gx1_direction_gc_mtf_later_followup'
    experiment_sources = SOURCE_PATHS + (('gx1/audit/direction_gc_context_intervention_v1.py',
        'tests/test_direction_gc_context_intervention_v1.py') if context_mode else ())
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    sys.dont_write_bytecode = True
    require(device == 'cuda' and not torch.cuda.is_initialized(), 'EXPLICIT_CUDA')
    plan_ref = {'path': str(candle.regular(plan_path)), 'sha256': plan_sha256}
    policy_path = REPO / 'NEXT_RUN_POLICY.json'
    source_head, branch, dirty = candle.git_state()
    require(branch == 'work/gx1-current' and not dirty, 'INITIAL_SOURCE')
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir',
            'wall_seconds', 'batch_size', 'batch_stride', 'original_source_commit', 'test_data_used'}
            and plan['schema_version'] == experiment_schema + '_plan_v1' and plan['stage'] == experiment_stage
            and plan['wall_seconds'] == 1800 and plan['batch_size'] == intervention.BATCH
            and plan['batch_stride'] == intervention.BATCH_STRIDE and plan['test_data_used'] is False, 'PLAN')
        require(str(out_dir) == plan['output_dir'] and out_dir.parent == ROOT
            and out_dir.name == experiment_output
            and ROOT.resolve(strict=True) == ROOT and not out_dir.exists() and not out_dir.is_symlink(), 'OUTPUT_SCOPE')
        require(plan['design']['path'] == str(REPO / 'docs' / experiment_design), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == experiment_schema + '_design_v1'
            and design['inputs'] == plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy_sha = common._hash_file(str(policy_path))['sha256']; policy_ref = {'path': str(policy_path), 'sha256': policy_sha}
        policy = binding.read_bound_json(policy_path, policy_sha); gate = policy[experiment_stage]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['checkpoint_reads_authorized'] is True and gate['model_forwards_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
            and all(gate[key] is False for key in ('optimizer_steps_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(experiment_sources)
            and all(ref['path'] == str(REPO / name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        keys = {'recipe', 'fit_result', 'checkpoint', 'oof_report', 'oof_predictions', 'corrected_target',
            'corrected_result', 'consumer_result', 'consumer_plan', 'synthetic_test_terminal', 'synthetic_test_plan',
            'fold_design', 'source_transition', 'prior_failed_result', 'prior_recovery'}
        if context_mode:
            keys |= {'prior_paired', 'prior_followup_result', 'prior_verify'}
        require(set(plan['inputs']) == keys, 'INPUT_SET')
        binary = {'checkpoint', 'oof_predictions', 'corrected_target'} | ({'prior_paired'} if context_mode else set())
        meta = {}
        for key in keys - binary:
            ref = plan['inputs'][key]; meta[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        launch, result, oof, corrected, consumer = (meta[key] for key in
            ('recipe', 'fit_result', 'oof_report', 'corrected_result', 'consumer_result'))
        require(result['outcome'] == 'COMPLETE' and result['test_data_used'] is False
            and result['recipe'] == plan['inputs']['recipe'] and result['full_state'] == plan['inputs']['checkpoint']
            and result['oof_measurement'] == plan['inputs']['oof_report']
            and oof['predictions'] == plan['inputs']['oof_predictions'] and oof['fold'] == result['fold'] == launch['fold']
            and oof['test_data_used'] is False and launch['numerical_recipe']['dense_minute_supervision'] is True, 'FIT_LINEAGE')
        require(corrected['status'] == 'COMPLETE' and corrected['target'] == plan['inputs']['corrected_target']
            and corrected['target_identity'] == 'exact_original_decimal_midpoint_v1'
            and corrected['test_data_used'] is False and consumer['status'] == 'COMPLETE'
            and consumer['independent_60_horizon_confusion_matrices_reproduced'] is True
            and consumer['plan'] == plan['inputs']['consumer_plan']
            and meta['consumer_plan']['inputs']['corrected'] == plan['inputs']['corrected_target']
            and consumer['test_data_used'] is False, 'CORRECTED_LINEAGE')
        prior = policy['independent_direction_gc_mtf_intervention_20261028_003']
        require(prior['status'] == 'FAILED_ONE_SHOT_CONSUMED' and prior['execution_authorized'] is False
            and prior['inner_result'] == plan['inputs']['prior_failed_result']
            and prior['recovery_result'] == plan['inputs']['prior_recovery']
            and meta['prior_recovery']['status'] == 'COMPLETE'
            and meta['prior_recovery']['original_run_remains_failed'] is True, 'PRIOR_SCOPE')
        if context_mode:
            previous = policy[STAGE]
            require(previous['status'] == 'COMPLETE_ONE_SHOT_CONSUMED'
                and previous['paired_outputs'] == plan['inputs']['prior_paired']
                and previous['result'] == plan['inputs']['prior_followup_result']
                and previous['independent_score_verification'] == plan['inputs']['prior_verify']
                and previous['independent_score_verification_complete'] is True
                and meta['prior_followup_result']['status'] == meta['prior_verify']['status'] == 'COMPLETE'
                and meta['prior_verify']['followup_result'] == plan['inputs']['prior_followup_result']
                and meta['prior_verify']['paired_outputs'] == plan['inputs']['prior_paired']
                and meta['prior_verify']['corrected_target'] == plan['inputs']['corrected_target']
                and {k: meta['prior_followup_result']['paired_outputs'][k] for k in ('path', 'sha256')}
                    == plan['inputs']['prior_paired']
                and meta['prior_followup_result']['test_data_used'] is False
                and meta['prior_verify']['test_data_used'] is False, 'PRIOR_VERIFIED_FOLLOWUP')
        intervention.require_synthetic_receipts(plan['inputs'], result, meta['synthetic_test_terminal'], meta['synthetic_test_plan'])
        fold = later_fold(meta['fold_design'], launch['fold'])
        original_bytes = subprocess.check_output(['git', 'show', plan['original_source_commit'] + ':' + OWNER], cwd=REPO)
        current_bytes = (REPO / OWNER).read_bytes()
        freeze_launch = source_transition(launch, meta['source_transition'], original_bytes, current_bytes)
        require(len(freeze_launch['source_owners']) == 56, 'FIT_SOURCE_COUNT')
        for ref in freeze_launch['source_owners'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='fit source')
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
        raise RuntimeError('DIRECTION_GC_MTF_LATER_FOLLOWUP_WALL_LIMIT')
    def forbid_updates(frame, event, arg):
        if event == 'call':
            name, path = frame.f_code.co_name, frame.f_code.co_filename.replace('\\', '/')
            require(not (('/torch/optim/' in path and name in ('step', '_init_group'))
                or ('/torch/autograd/' in path and name in ('backward', '_engine_run_backward'))), 'FORBIDDEN_UPDATE')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(forbid_updates)
    stage = out_dir / '.OUTPUT.staging'; handles = []
    try:
        stage.mkdir(exist_ok=False)
        context = window._launch_metadata(launch, common)
        payload_names = ('checkpoint', 'oof_predictions', 'corrected_target') + (('prior_paired',) if context_mode else ())
        refs = [plan['inputs'][name] for name in payload_names]
        require(len({ref['path'] for ref in refs}) == len(refs), 'DISTINCT_PAYLOADS')
        for ref in refs:
            before = common._stat(ref['path']); require(common._hash_file(ref['path']) == ref, 'PAYLOAD_SHA')
            context['info']['metadata'][ref['path']] = {'ref': dict(ref), 'stat': before}
        freeze_binding = {'source_head': source_head, 'policy': policy_ref}
        common._freeze(freeze_launch, freeze_binding, context)
        denied = common._install_read_guard(context, out_dir, out_dir, dense_target=context['dense_target'])
        prepared, targets, view, _context_map, _order = window._prepare(context, launch, np, dense_minutes=True)
        with np.load(plan['inputs']['oof_predictions']['path'], allow_pickle=False) as archive:
            names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up')
            require(all(name in archive.files for name in names), 'OOF_FIELDS')
            saved = {name: archive[name].copy() for name in names}
        origin = launch['fold']; origin_slice = slice(origin['oof_start_row'], origin['oof_end_row'])
        for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            require(np.array_equal(saved[name], prepared['arrays'][name][origin_slice]), 'OOF_AXIS:' + name)
        for name in ('terminal_time_ns', 'observed', 'sign'):
            require(np.array_equal(saved[name], targets[name][origin_slice]), 'OOF_ORIGINAL_TARGET:' + name)
        require(saved['logit_up'].dtype == np.float32
            and saved['logit_up'].shape == (origin['oof_end_row'] - origin['oof_start_row'], 60)
            and np.isfinite(saved['logit_up']).all(), 'OOF_LOGITS')
        with np.load(plan['inputs']['corrected_target']['path'], allow_pickle=False) as archive:
            names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')
            require(all(name in archive.files for name in names), 'CORRECTED_FIELDS')
            corrected_arrays = {name: archive[name].copy() for name in names}
        for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            require(np.array_equal(corrected_arrays[name], prepared['arrays'][name]), 'CORRECTED_AXIS:' + name)
        for name in ('terminal_time_ns', 'observed'):
            require(np.array_equal(corrected_arrays[name], targets[name]), 'CORRECTED_CLOCK_MASK:' + name)
        first, last = fold['oof_start_row'], fold['oof_end_row']
        decisions = prepared['arrays']['decision_ns']
        require(np.searchsorted(decisions, fold['train_cutoff_ns'], side='left') == first
            and np.searchsorted(decisions, fold['oof_cutoff_ns'], side='left') == last, 'LATER_CLOCK')
        chosen = intervention.selected_batches(last - first); selection = np.concatenate(chosen)
        require(len(selection) > 0 and len(selection) < last - first and np.all(np.diff(selection) > 0), 'SELECTION')
        prior_pair = None
        gc_layout = None
        if context_mode:
            from gx1.audit import direction_gc_context_intervention_v1 as context_intervention
            from gx1.features.entry_specialist_feature_groups_v1 import compile_gc_mtf_geometry_entry_execution_profile
            gc_profile = compile_gc_mtf_geometry_entry_execution_profile(context['fitted']['gc_mtf_history'])
            require({'name': gc_profile['name'], 'contract_sha256': gc_profile['contract_sha256']}
                == view['entry_execution_profile'], 'CONTEXT_PROFILE_IDENTITY')
            gc_layout = context_intervention.compile_gc_context_layout(gc_profile)
            with np.load(plan['inputs']['prior_paired']['path'], allow_pickle=False) as archive:
                required = ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'original_logit_up')
                require(all(name in archive.files for name in required), 'PRIOR_PAIR_FIELDS')
                prior_pair = {name: archive[name].copy() for name in required}
            rows = first + selection
            for name in ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign'):
                require(np.array_equal(prior_pair[name], corrected_arrays[name][rows]), 'PRIOR_PAIR_AXIS:' + name)
            require(prior_pair['original_logit_up'].dtype == np.float32
                and prior_pair['original_logit_up'].shape == (len(selection), 60), 'PRIOR_PAIR_LOGITS')
        trainer._set_deterministic(launch['numerical_recipe']['seed'], torch.device('cuda', 0),
            launch['numerical_recipe']['precision_policy'])
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
            for name, buffer in model.named_buffers():
                buffer.copy_(state['buffer_state'][name].to(buffer.device))
        require(state_owner.canonical_model_state_sha256(model.state_dict()) == state['model_state_sha256']
            and state_owner._tree_sha(dict(model.named_buffers())) == state['buffer_state_sha256'], 'LOADED_STATE')
        model.backbone.require_input_normalization_state(); model._require_binding(); model.eval()
        model_sha, buffer_sha = state['model_state_sha256'], state['buffer_state_sha256']; del state
        modules = model.backbone.gc_mtf_family_proj; keys = tuple(modules.keys())
        require(len(keys) == 25 and all(any(key.startswith(tf + '__') for key in keys)
            for tf in ('m5', 'm15', 'h1', 'h4', 'd1')), 'GC_ROUTE_SET')
        counters = {'forward_calls': 0, 'forward_returns': 0, 'forward_kind_calls': {}}
        ceiling = 1 + 2 * len(chosen)
        raw, target = window._batch(prepared, targets, view, saved['parent_rows'][:intervention.BATCH], np)
        features, _target = window._device_batch(raw, target, torch, torch.device('cuda', 0))
        replay = window._forward(model, features, kind='REFERENCE_REPLAY', counters=counters,
            ceiling=ceiling, torch=torch).detach().cpu().numpy().copy()
        intervention.require_exact_replay(replay, np.ascontiguousarray(saved['logit_up'][:intervention.BATCH]))
        del raw, target, features, _target, replay
        original, ablated = [], []; hook_calls = {key: 0 for key in keys}; offset = 0
        for indices in chosen:
            require(time.monotonic() - start < plan['wall_seconds'], 'WALL_LIMIT')
            parent_ids = prepared['arrays']['parent_rows'][first + indices]
            raw, target = window._batch(prepared, targets, view, parent_ids, np)
            features, _target = window._device_batch(raw, target, torch, torch.device('cuda', 0))
            z0 = window._forward(model, features, kind='LATER_ORIGINAL', counters=counters,
                ceiling=ceiling, torch=torch).detach().cpu().numpy().copy()
            if context_mode:
                intervention.require_exact_replay(z0, np.ascontiguousarray(
                    prior_pair['original_logit_up'][offset:offset + len(indices)]))
                counterfactual = context_intervention.drop_gc_context(features, gc_layout)
                offset += len(indices)
            else:
                counterfactual = features
            handles, calls = intervention.install_gc_projection_ablation(modules, keys)
            try:
                z1 = window._forward(model, counterfactual, kind='LATER_ABLATION', counters=counters,
                    ceiling=ceiling, torch=torch).detach().cpu().numpy().copy()
            finally:
                for handle in handles:
                    handle.remove()
                handles = []
            require(all(value == 1 for value in calls.values()) and np.isfinite(z0).all() and np.isfinite(z1).all(), 'HOOK_CALLS')
            for key, value in calls.items(): hook_calls[key] += value
            original.append(z0); ablated.append(z1)
            del raw, target, features, counterfactual, _target, z0, z1
        z0, z1 = np.concatenate(original), np.concatenate(ablated)
        require(not context_mode or offset == len(selection), 'LATER_REPLAY_ALL_ROWS')
        require(counters['forward_calls'] == counters['forward_returns'] == ceiling
            and all(value == len(chosen) for value in hook_calls.values()) and not denied, 'CALL_ACCOUNTING')
        require(state_owner.canonical_model_state_sha256(model.state_dict()) == model_sha
            and state_owner._tree_sha(dict(model.named_buffers())) == buffer_sha
            and all(parameter.grad is None for parameter in model.parameters()), 'MODEL_STATE_UNCHANGED')
        rows = first + selection
        parents = np.ascontiguousarray(corrected_arrays['parent_rows'][rows])
        selected_decisions = np.ascontiguousarray(corrected_arrays['decision_ns'][rows])
        selected_terminal = np.ascontiguousarray(corrected_arrays['terminal_time_ns'][rows])
        selected_observed = np.ascontiguousarray(corrected_arrays['observed'][rows])
        selected_sign = np.ascontiguousarray(corrected_arrays['sign'][rows])
        metrics = intervention.paired_metrics(selected_decisions, selected_observed, selected_sign,
            selected_terminal, z0, z1, cutoff=fold['oof_cutoff_ns'])
        metrics['scope'] = 'Every tenth contiguous 16-row batch in the subsequent fixed development period; no refit.'
        pair = save_paired(stage / 'PAIRED_OUTPUTS.npz', parents=parents, decisions=selected_decisions,
            terminal=selected_terminal, observed=selected_observed, sign=selected_sign, original=z0, ablated=z1)
        pair = {'path': str(out_dir / 'OUTPUT/PAIRED_OUTPUTS.npz'), **pair}
        logit_change = np.abs(z1.astype(np.float64) - z0.astype(np.float64))
        common._freeze(freeze_launch, freeze_binding, context); control_snapshot.require_unchanged()
        require(cap.require_capped_lock_ancestry() == lock and not denied, 'FINAL_SOURCE_LOCK')
        output = {'schema_version': experiment_schema + '_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': source_head, 'origin_fold': origin, 'later_fold': fold,
            'model_state_sha256': model_sha, 'source_transition': plan['inputs']['source_transition'],
            'selection': {'policy': 'every tenth contiguous 16-row later batch', 'batch_size': intervention.BATCH,
                'batch_stride': intervention.BATCH_STRIDE, 'selected_parents': len(selection), 'selected_batches': len(chosen)},
            'fold1_reference_logits_byte_identical': True, 'corrected_target_identity': corrected['target_identity'],
            'intervention': ('GC context values and observation masks are zero in the full local sequence and current context; all 25 GC history projections are zeroed after projection.' if context_mode else
                'All 25 GC history projection outputs set to zero after projection; current GC context remains.'),
            'later_reference_logits_byte_identical': True if context_mode else None,
            'gc_context_layout': ({'gc_fields': gc_layout['gc_fields'], 'value_mask_pairs': gc_layout['value_mask_pairs'],
                'aliases_sha256': gc_layout['aliases_sha256']} if context_mode else None),
            'paired_outputs': pair, 'mean_absolute_logit_change': float(logit_change.mean()),
            'max_absolute_logit_change': float(logit_change.max()), 'forward_calls': counters['forward_calls'],
            'optimizer_steps': 0, 'model_unchanged': True, 'metrics': metrics,
            'scope': 'Frozen fold1 model, distinct subsequent development period, fixed sample; no refit or model selection.',
            'limits': ('Previously reused development data; the combined intervention was chosen after the earlier result and can be out of distribution. It suppresses named GC context/aliases and GC history projections, not necessarily every GC-derived quantity. No independent generalization or economic edge.' if context_mode else
                'Previously reused development data; removing an internal route can be out of distribution. The GC current context remains. No independent generalization or economic edge.'),
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
        for handle in handles:
            handle.remove()
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)
