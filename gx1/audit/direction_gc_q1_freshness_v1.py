"""One fixed-Q1, frozen M5 model GC freshness sensitivity experiment.

Fresh and stale predictions use the same parents, targets and model. GC context
aliases are taken from the exact previous M5 decision and newly closed GC
history buckets are hidden. This is not an estimate of local feed latency or
trading economics, and the counterfactual may be out of distribution.
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

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_gc_q1_freshness_20261030_001'
OUTPUT_NAME = 'DIRECTION_GC_Q1_FRESHNESS_20261030_001'
FIT_COMMIT = 'bcedbbd65b4a71a7c57973ed311e2eef988a82b1'
BATCH = 16
STRIDE = 10
SELECTED = 1736
VALID_PRIOR = 1727
SOURCE_PATHS = (
    'gx1/audit/direction_gc_q1_freshness_v1.py',
    'gx1/audit/direction_gc_mtf_intervention_v1.py',
    'gx1/audit/direction_gc_mtf_followup_v1.py',
    'gx1/audit/direction_gc_context_intervention_v1.py',
    'gx1/audit/direction_gc_stale_context_v1.py',
    'gx1/audit/direction_gc_history_freshness_v1.py',
    'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh', 'scripts/gx1_guarded_trainer_exec.sh',
    'tests/test_direction_gc_q1_freshness_v1.py',
    'tests/test_direction_gc_history_freshness_v1.py',
    'tests/test_direction_gc_stale_context_v1.py',
)


def require(value, reason):
    if not bool(value):
        raise RuntimeError('DIRECTION_GC_Q1_FRESHNESS_' + reason)


def previous_rows(decisions, rows):
    """Find exact earlier M5 decisions; a missing clock stays unmodified."""
    require(isinstance(decisions, np.ndarray) and decisions.dtype == np.int64
            and decisions.ndim == 1 and len(decisions) > 0
            and np.all(np.diff(decisions) > 0), 'DECISION_CLOCK')
    require(isinstance(rows, np.ndarray) and rows.dtype == np.int64
            and rows.ndim == 1 and len(rows) > 0
            and np.all((rows >= 0) & (rows < len(decisions))), 'ROWS')
    requested = decisions[rows] - 300_000_000_000
    previous = np.searchsorted(decisions, requested)
    safe = np.minimum(previous, len(decisions) - 1)
    valid = (previous < len(decisions)) & (decisions[safe] == requested)
    safe = np.where(valid, safe, rows)
    require(np.all(safe[valid] < rows[valid])
            and np.all(decisions[safe[valid]] + 300_000_000_000 == decisions[rows[valid]]), 'PRIOR_CLOCK')
    return safe, valid


def fit_source_transition(launch, plan, historical, current):
    """Permit only the added exact-scope selector in the recorded fit runner."""
    key = 'scripts/gx1_capped_run.sh'
    old = launch['source_owners'][key]
    new = plan['sources'][key]
    require(len(launch['source_owners']) == 64 and old['path'] == new['path']
            and hashlib.sha256(historical).hexdigest() == old['sha256']
            and hashlib.sha256(current).hexdigest() == new['sha256'], 'FIT_RUNNER_HASH')
    a = b'# Exact current-profile GC freshness scope, with the existing signed guard.\n'
    b = b'validate_target_command() {\n'
    require(current.count(a) == 1 and current.count(b) == 1, 'SELECTOR_BOUNDARY')
    start, end = current.index(a), current.index(b)
    require(start < end, 'SELECTOR_ORDER')
    stripped = current[:start] + current[end:]
    c = b'      if [[ "${3:-}" == "$TECHNICAL_VALIDATION_PRODUCER_MODULE" \\\n        && "${4:-}" == --direction-gc-q1-freshness-plan ]]; then\n'
    d = b'        TRAINER_GPU_MAX_POWER_LIMIT_W=250\n      fi\n'
    require(stripped.count(c) == 1, 'SELECTOR_BRANCH')
    start = stripped.index(c)
    end = stripped.index(d, start) + len(d)
    stripped = stripped[:start] + stripped[end:]
    require(stripped == historical, 'FIT_RUNNER_UNRELATED_CHANGE')
    return {**launch, 'source_owners': {**launch['source_owners'], key: dict(new)}}


def run(*, plan_path, plan_sha256, device, out_dir):
    from gx1.audit import direction_gc_mtf_intervention_v1 as gc
    from gx1.audit import direction_gc_mtf_followup_v1 as followup
    from gx1.audit import direction_gc_context_intervention_v1 as context_intervention
    from gx1.audit.direction_gc_stale_context_v1 import replace_current_gc_context
    from gx1.audit.direction_gc_history_freshness_v1 import hide_fresh_gc_history
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.features.entry_direction_momentum_profile_v1 import compile_price_momentum_profile, PROFILE_NAME
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    sys.dont_write_bytecode = True
    require(device == 'cuda' and not torch.cuda.is_initialized(), 'EXPLICIT_CUDA')
    require(gc.BATCH == BATCH and gc.BATCH_STRIDE == STRIDE and gc.HEADLINE == (5, 60)
            and gc.REPLICATES == 10000, 'METRIC_CONTRACT')
    plan_ref = {'path': str(candle.regular(plan_path)), 'sha256': plan_sha256}
    policy_path = REPO / 'NEXT_RUN_POLICY.json'
    source_head, branch, dirty = candle.git_state()
    require(branch == 'work/gx1-current' and not dirty, 'SOURCE_CLEAN')
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir',
                              'wall_seconds', 'batch_size', 'batch_stride', 'test_data_used'}
                and plan['schema_version'] == 'gx1_direction_gc_q1_freshness_plan_v1'
                and plan['stage'] == STAGE and plan['wall_seconds'] == 3300
                and plan['batch_size'] == BATCH and plan['batch_stride'] == STRIDE
                and plan['test_data_used'] is False, 'PLAN')
        require(str(out_dir) == plan['output_dir'] and out_dir.parent == ROOT
                and out_dir.name == OUTPUT_NAME and ROOT.resolve(strict=True) == ROOT
                and not out_dir.exists() and not out_dir.is_symlink(), 'OUTPUT')
        require(plan['design']['path'] == str(REPO / 'docs/DIRECTION_GC_Q1_FRESHNESS_DESIGN_20261030.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_direction_gc_q1_freshness_design_v1'
                and design['inputs'] == plan['inputs'] and design['test_data_used'] is False
                and design['selected_parents'] == SELECTED and design['prior_available_parents'] == VALID_PRIOR
                and design['primary_minutes'] == 5 and design['horizons_minutes'] == list(range(1, 61)), 'DESIGN')
        policy_sha = common._hash_file(str(policy_path))['sha256']
        policy_ref = {'path': str(policy_path), 'sha256': policy_sha}
        policy = binding.read_bound_json(policy_path, policy_sha)
        gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref
                and gate['execution_authorized'] is gate['checkpoint_reads_authorized'] is gate['model_forwards_authorized'] is True
                and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
                and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled',
                'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
                and all(gate[k] is False for k in ('optimizer_steps_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
                and all(ref['path'] == str(REPO / name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        keys = {'recipe', 'fit_review', 'fit_result', 'fit_receipt', 'checkpoint', 'oof_report', 'oof_predictions',
                'corrected_target', 'synthetic_test_terminal', 'synthetic_test_plan', 'feasibility', 'profile_preflight'}
        require(set(plan['inputs']) == keys, 'INPUT_SET')
        meta = {}
        for key in keys - {'checkpoint', 'oof_predictions', 'corrected_target'}:
            ref = plan['inputs'][key]
            meta[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        launch, fit, review, receipt, oof = (meta[k] for k in ('recipe', 'fit_result', 'fit_review', 'fit_receipt', 'oof_report'))
        require(launch['recipe_sha256'] == binding.canonical_sha256({k: v for k, v in launch.items() if k != 'recipe_sha256'})
                and launch['run_scope'] == authority.M5_FACTOR_CONTINUATION_SCOPE and launch['candidate_mode'] == 'price_only'
                and launch['train_passes'] == 2 and 'candidate_m1_refs' not in launch
                and launch['dense_target'] == plan['inputs']['corrected_target'], 'RECIPE')
        require(review['status'] == 'PASS_SIGN_TWOPASS_TERMINAL_CHAIN_NOT_EDGE'
                and review['fit_result'] == plan['inputs']['fit_result'] and review['receipt'] == plan['inputs']['fit_receipt']
                and review['terminal_checkpoint'] == plan['inputs']['checkpoint']
                and fit['outcome'] == fit['phase_state'] == 'COMPLETE' and fit['recipe'] == plan['inputs']['recipe']
                and fit['full_state'] == plan['inputs']['checkpoint'] and fit['oof_measurement'] == plan['inputs']['oof_report']
                and fit['parent_cursor'] == 95204 and fit['optimizer_updates'] == 5952 and fit['oof_cursor'] == 17288
                and fit['source_unchanged'] is True and fit['test_data_used'] is False
                and fit['qualification_weights_reused'] is False and not fit['forbidden_open_paths']
                and receipt['outcome'] == 'COMPLETE' and receipt['guard_decision'] == 'PASS' and receipt['trainer_guard_exit_code'] == 0
                and oof['role'] == 'OOF' and oof['fold'] == fit['fold'] == launch['fold']
                and oof['predictions'] == plan['inputs']['oof_predictions'] and oof['test_data_used'] is False, 'FIT_LINEAGE')
        require(meta['feasibility']['status'] == 'OLD_RUNNER_NOT_USABLE_UNCHANGED_NEW_ONE_SHOT_NEEDED_AFTER_FROZEN_FIT'
                and meta['profile_preflight']['status'] == 'PASS_GC_LAYOUT_PARITY_BUT_OLD_RUNNER_NOT_ADMITTED_FOR_CURRENT_PROFILE'
                and meta['feasibility']['profile_preflight'] == plan['inputs']['profile_preflight'], 'PREFLIGHT')
        subprocess.run(['git', 'merge-base', '--is-ancestor', FIT_COMMIT, source_head], cwd=REPO, check=True)
        old = subprocess.check_output(['git', 'show', FIT_COMMIT + ':scripts/gx1_capped_run.sh'], cwd=REPO)
        freeze_launch = fit_source_transition(launch, plan, old, (REPO / 'scripts/gx1_capped_run.sh').read_bytes())
        for name, ref in freeze_launch['source_owners'].items():
            candle.regular(ref['path']); binding.require_binding(ref, label='fit source')
        control_snapshot = binding.capture_verified_file_snapshot()
    hardware, lock = cap.require_guarded_cuda_producer_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == hardware['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    out_dir.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out_dir / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
            'source_head': source_head, 'execution': hardware, 'lock': lock})
    gc._install_runtime_guard(out_dir, (plan['inputs']['synthetic_test_terminal']['path'], plan['inputs']['synthetic_test_plan']['path']))
    start = time.monotonic(); old_profile = sys.getprofile(); old_handler = signal.getsignal(signal.SIGALRM)
    stage = out_dir / '.OUTPUT.staging'
    def expired(*_):
        raise RuntimeError('DIRECTION_GC_Q1_FRESHNESS_WALL_LIMIT')
    def forbid_updates(frame, event, arg):
        if event == 'call':
            name, path = frame.f_code.co_name, frame.f_code.co_filename.replace('\\', '/')
            require(not (('/torch/optim/' in path and name in ('step', '_init_group'))
                    or ('/torch/autograd/' in path and name in ('backward', '_engine_run_backward'))), 'FORBIDDEN_UPDATE')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(forbid_updates)
    try:
        stage.mkdir(exist_ok=False)
        input_launch = window._fit_input_launch(launch, authority)
        context = window._launch_metadata(input_launch, common)
        require(context.get('candidate_inputs') == {'mode': 'price_only'} and 'candidate_m1_refs' not in context, 'M5_CONTEXT')
        payloads = [plan['inputs'][k] for k in ('checkpoint', 'oof_predictions', 'corrected_target')]
        require(len({ref['path'] for ref in payloads}) == len(payloads), 'DISTINCT_PAYLOADS')
        for ref in payloads:
            before = common._stat(ref['path']); require(common._hash_file(ref['path']) == ref, 'PAYLOAD_SHA')
            context['info']['metadata'][ref['path']] = {'ref': dict(ref), 'stat': before}
        freeze = {'source_head': source_head, 'policy': policy_ref}
        common._freeze(freeze_launch, freeze, context)
        denied = common._install_read_guard(context, out_dir, out_dir,
                    dense_target=context['dense_target'], decimal_original=context['decimal_original'])
        prepared, targets, view, _context_map, _ordered = window._prepare(context, input_launch, np, dense_minutes=True)
        names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up')
        with np.load(plan['inputs']['oof_predictions']['path'], allow_pickle=False) as archive:
            require(all(name in archive.files for name in names), 'OOF_FIELDS')
            saved = {name: archive[name].copy() for name in names}
        fold = launch['fold']; first, last = fold['oof_start_row'], fold['oof_end_row']
        require((first, last) == (47602, 64890) and np.array_equal(saved['parent_rows'], np.arange(first, last, dtype=np.int64)), 'Q1_POPULATION')
        for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            require(np.array_equal(saved[name], prepared['arrays'][name][first:last]), 'OOF_AXIS:' + name)
        for name in ('terminal_time_ns', 'observed', 'sign'):
            require(np.array_equal(saved[name], targets[name][first:last]), 'OOF_TARGET:' + name)
        require(saved['logit_up'].dtype == np.float32 and saved['logit_up'].shape == (17288, 60)
                and np.isfinite(saved['logit_up']).all(), 'OOF_LOGITS')
        chosen = gc.selected_batches(last - first); selection = np.concatenate(chosen)
        decisions = prepared['arrays']['decision_ns']; _, validity = previous_rows(decisions, first + selection)
        require(len(selection) == SELECTED and int(validity.sum()) == VALID_PRIOR, 'PRIOR_COHORT')
        gc_profile = compile_price_momentum_profile(context['fitted']['gc_mtf_history'])
        require({'name': gc_profile['name'], 'contract_sha256': gc_profile['contract_sha256']} == view['entry_execution_profile']
                and gc_profile['name'] == PROFILE_NAME, 'PROFILE_IDENTITY')
        gc_layout = context_intervention.compile_gc_context_layout(gc_profile)
        require(gc_layout['aliases_sha256'] == meta['profile_preflight']['aliases_sha256']
                and gc_layout['gc_fields'] == meta['profile_preflight']['gc_fields']
                and gc_layout['value_mask_pairs'] == meta['profile_preflight']['value_mask_pairs'], 'GC_LAYOUT')
        trainer._set_deterministic(launch['numerical_recipe']['seed'], torch.device('cuda', 0), launch['numerical_recipe']['precision_policy'])
        torch.set_num_threads(1); require(torch.get_num_threads() == 1, 'NUMERICAL_THREADS')
        model = build_native_direction_research_model(view, prefix_normalization=context['fitted'],
                    constructor_metadata=context['constructor'], device=torch.device('cuda', 0))
        require(model.feature_view_identity == fit['feature_view_identity'] and model.backbone.entry_execution_profile['name'] == PROFILE_NAME, 'VIEW')
        state = torch.load(plan['inputs']['checkpoint']['path'], map_location='cpu', weights_only=True)
        require(isinstance(state, dict) and set(state) == (state_owner.STATE_KEYS | {'train_passes'})
                and state['schema_version'] == state_owner.MULTIPASS_SCHEMA_VERSION
                and type(state['train_passes']) is int and state['train_passes'] == 2
                and state['contract_sha256'] == state_owner._tree_sha({k: v for k, v in state.items() if k != 'contract_sha256'})
                and state['contract_sha256'] == fit['durable_checkpoint']['contract_sha256']
                and state['model_state_sha256'] == state_owner.canonical_model_state_sha256(state['model_state'])
                and state['binding']['feature_view_identity'] == model.feature_view_identity
                and state['cursor'] == 95204 and state['optimizer_updates'] == 5952, 'CHECKPOINT_SEAL')
        model.load_state_dict(state['model_state'], strict=True)
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(state['buffer_state'][name].to(buffer.device))
        require(state_owner.canonical_model_state_sha256(model.state_dict()) == state['model_state_sha256']
                and state_owner._tree_sha(dict(model.named_buffers())) == state['buffer_state_sha256'], 'LOADED_STATE')
        model.backbone.require_input_normalization_state(); model._require_binding(); model.eval()
        model_sha, buffer_sha = state['model_state_sha256'], state['buffer_state_sha256']; del state
        counters = {'forward_calls': 0, 'forward_returns': 0, 'forward_kind_calls': {}}
        ceiling = 2 * len(chosen); original, stale = [], []
        hidden = {tf: {'hidden_rows': 0, 'nonzero_rows_hidden': 0} for tf in ('M5', 'M15', 'H1', 'H4', 'D1')}
        for ids in chosen:
            require(time.monotonic() - start < plan['wall_seconds'], 'WALL_LIMIT')
            rows_now = first + ids; prior, valid = previous_rows(decisions, rows_now)
            raw, target = window._batch(prepared, targets, view, saved['parent_rows'][ids], np)
            features, _unused = window._device_batch(raw, target, torch, torch.device('cuda', 0))
            z0 = window._forward(model, features, kind='Q1_REPLAY', counters=counters, ceiling=ceiling, torch=torch).detach().cpu().numpy().copy()
            gc.require_exact_replay(z0, np.ascontiguousarray(saved['logit_up'][ids]))
            prior_snap = torch.from_numpy(np.ascontiguousarray(prepared['current']['snap'][prior])).to(torch.device('cuda', 0))
            prior_ctx = torch.from_numpy(np.ascontiguousarray(prepared['current']['ctx_cont'][prior])).to(torch.device('cuda', 0))
            counterfactual = replace_current_gc_context(features, prior_snap, prior_ctx, gc_layout)
            counterfactual, hidden_now = hide_fresh_gc_history(counterfactual, witness=raw['witness']['multi_tf'],
                decision_ns=np.ascontiguousarray(decisions[rows_now]), prior_available=np.ascontiguousarray(valid))
            for tf, record in hidden_now.items():
                for key, value in record.items(): hidden[tf][key] += value
            z1 = window._forward(model, counterfactual, kind='Q1_STALE_GC', counters=counters, ceiling=ceiling, torch=torch).detach().cpu().numpy().copy()
            require(np.isfinite(z1).all() and np.array_equal(z0[~valid], z1[~valid]), 'STALE_MISSING_PRIOR')
            original.append(z0); stale.append(z1)
            del raw, target, features, _unused, prior_snap, prior_ctx, counterfactual, z0, z1
        z0, z1 = np.concatenate(original)[validity], np.concatenate(stale)[validity]
        selected = selection[validity]
        rows = {key: np.ascontiguousarray(saved[key][selected]) for key in ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')}
        require(counters['forward_calls'] == counters['forward_returns'] == ceiling and not denied, 'CALL_ACCOUNTING')
        require(state_owner.canonical_model_state_sha256(model.state_dict()) == model_sha
                and state_owner._tree_sha(dict(model.named_buffers())) == buffer_sha
                and all(parameter.grad is None for parameter in model.parameters()), 'MODEL_UNCHANGED')
        metrics = gc.paired_metrics(rows['decision_ns'], rows['observed'], rows['sign'], rows['terminal_time_ns'], z0, z1, cutoff=fold['oof_cutoff_ns'])
        for j, item in enumerate(metrics['per_horizon']):
            mask = rows['observed'][:, j] & (rows['sign'][:, j] != 0) & (rows['terminal_time_ns'][:, j] < fold['oof_cutoff_ns'])
            labels = rows['sign'][mask, j] == 1
            fresh_dec, stale_dec = z0[mask, j] > 0, z1[mask, j] > 0
            item['arms']['original']['accuracy'] = float(np.mean(fresh_dec == labels))
            item['arms']['ablated']['accuracy'] = float(np.mean(stale_dec == labels))
            item['always_up_accuracy'] = float(np.mean(labels))
            item['direction_changes'] = int(np.count_nonzero(fresh_dec != stale_dec))
            item['mean_absolute_logit_change'] = float(np.mean(np.abs(z0[mask, j].astype(np.float64) - z1[mask, j].astype(np.float64))))
        metrics['scope'] = 'Fixed every-tenth Q1 batch, requiring exact prior M5 decision; paired frozen-model comparison.'
        metrics['limitations'] = 'GC context aliases and all newly closed GC history buckets change together; may be out of distribution. Calendar-block exchangeability, local feed availability and trading economics are not established.'
        pair = followup.save_paired(stage / 'PAIRED_OUTPUTS.npz', parents=rows['parent_rows'], decisions=rows['decision_ns'],
            terminal=rows['terminal_time_ns'], observed=rows['observed'], sign=rows['sign'], original=z0, ablated=z1)
        pair = {'path': str(out_dir / 'OUTPUT/PAIRED_OUTPUTS.npz'), **pair}
        common._freeze(freeze_launch, freeze, context); control_snapshot.require_unchanged()
        require(cap.require_capped_lock_ancestry() == lock and not denied, 'FINAL_SOURCE_LOCK')
        result = {'schema_version': 'gx1_direction_gc_q1_freshness_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': source_head, 'checkpoint': plan['inputs']['checkpoint'], 'fold': fold,
            'selection': {'batch_size': BATCH, 'batch_stride': STRIDE, 'selected_parents': SELECTED,
                          'exact_prior_parents': VALID_PRIOR, 'missing_prior_parents': SELECTED - VALID_PRIOR},
            'exact_saved_q1_logits_reproduced_all_selected': True, 'gc_context_layout': {
                'gc_fields': gc_layout['gc_fields'], 'value_mask_pairs': gc_layout['value_mask_pairs'], 'aliases_sha256': gc_layout['aliases_sha256']},
            'intervention': 'Previous M5 GC context and aliases plus masking of GC history buckets that closed since then; all other inputs remain current.',
            'hidden_gc_history_rows': hidden, 'paired_outputs': pair, 'metrics': metrics,
            'forward_calls': counters['forward_calls'], 'optimizer_steps': 0, 'model_state_unchanged': True,
            'test_data_used': False, 'economic_edge_qualified': False,
            'limitations': 'One reused Q1 development subset and one frozen model; joint intervention may be out of distribution, does not isolate a GC feature, and cannot establish local latency or executable profitability.'}
        publish(stage / 'RESULT.json', result)
        publish(stage / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
                'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - start, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out_dir / 'OUTPUT')
        publish(out_dir / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
                'result_path': str(out_dir / 'OUTPUT/RESULT.json')})
        return result
    except Exception as exc:
        publish(out_dir / 'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
                'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False})
        raise
    finally:
        sys.setprofile(old_profile); signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
