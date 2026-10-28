"""Bounded paired intervention on a trained direction model's GC-history route.

No target or feature is altered. A forward hook suppresses only the projected
GC history contribution before it is added to the XAU family representation;
other GC context paths remain. This is an ablation, not a trading rule.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_gc_mtf_intervention_20261028_002'
MINUTE_NS = 60 * 10**9
DAY_NS = 86400 * 10**9
BATCH = 16
BATCH_STRIDE = 10
HORIZONS = tuple(range(1, 61))
HEADLINE = (5, 60)
REPLICATES = 10000
ALPHA = 0.05 / (len(HEADLINE) * 2 * 3)
SOURCE_PATHS = (
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
    'tests/test_direction_gc_mtf_intervention_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_GC_MTF_INTERVENTION_' + reason)


def selected_batches(rows):
    """Every tenth original contiguous 16-row OOF batch, predeclared."""
    require(type(rows) is int and rows > 0, 'ROWS')
    return [np.arange(start, min(start + BATCH, rows), dtype=np.int64)
        for start in range(0, rows, BATCH * BATCH_STRIDE)]


def paired_metrics(decisions, observed, signs, terminal, baseline, ablated, *, cutoff):
    """Fixed-horizon paired scores, with day/week intervals at 5 and 60 minutes."""
    from gx1.contracts import entry_direction_block_rank_v1 as rank
    from gx1.contracts import entry_direction_probability_reference_v1 as probability
    from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights, percentile_interval
    n = len(decisions)
    require(decisions.dtype == np.int64 and decisions.shape == (n,) and n > 0
        and observed.dtype == np.bool_ and signs.dtype == np.int8 and terminal.dtype == np.int64
        and observed.shape == signs.shape == terminal.shape == baseline.shape == ablated.shape == (n, 60)
        and np.isfinite(baseline).all() and np.isfinite(ablated).all()
        and np.isin(signs, (-1, 0, 1)).all(), 'METRIC_INPUT')
    scores = []
    for j in range(60):
        eligible = observed[:, j] & (signs[:, j] != 0) & (terminal[:, j] < cutoff)
        labels = np.ascontiguousarray((signs[eligible, j] == 1).astype(np.int8))
        require(len(labels) > 0 and np.unique(labels).size == 2, 'BOTH_CLASSES:' + str(j + 1))
        arms = {}
        for name, z in (('original', baseline[eligible, j]), ('ablated', ablated[eligible, j])):
            z = np.ascontiguousarray(z, dtype=np.float64)
            metrics = probability.direction_probability_metrics(labels, z)
            arms[name] = {'auc': probability.direction_rank_diagnostics(labels, z, score_kind='logit_up')['auc'],
                'logloss': float(np.mean(metrics['logloss_by_row'])), 'brier': float(np.mean(metrics['brier_by_row']))}
        scores.append({'horizon_minutes': j + 1, 'eligible': len(labels), 'arms': arms})
    intervals = []
    for horizon in HEADLINE:
        j = horizon - 1
        eligible = observed[:, j] & (signs[:, j] != 0) & (terminal[:, j] < cutoff)
        labels = np.ascontiguousarray((signs[eligible, j] == 1).astype(np.int8))
        actual = np.ascontiguousarray(baseline[eligible, j], dtype=np.float64)
        no_gc = np.ascontiguousarray(ablated[eligible, j], dtype=np.float64)
        days = decisions[eligible] // DAY_NS
        for scheme, block_keys, seed in (('UTC_DAY', days, 20261027), ('UTC_WEEK_MONDAY', (days + 3) // 7, 20261028)):
            blocks, ids = np.unique(block_keys, return_inverse=True)
            require(len(blocks) >= 8, 'BLOCK_SUPPORT:' + scheme)
            weights = block_weights(len(blocks), seed=seed, np=np, replicates=REPLICATES)
            original = rank.block_rank_pairs(labels, actual, ids)
            without = rank.block_rank_pairs(labels, no_gc, ids)
            original_wins, original_pairs = rank.weighted_rank_pairs(original, weights)
            without_wins, without_pairs = rank.weighted_rank_pairs(without, weights)
            auc_original = np.divide(original_wins, 2 * original_pairs, out=np.full(REPLICATES, np.nan), where=original_pairs > 0)
            auc_without = np.divide(without_wins, 2 * without_pairs, out=np.full(REPLICATES, np.nan), where=without_pairs > 0)
            metrics_original = probability.direction_probability_metrics(labels, actual)
            metrics_without = probability.direction_probability_metrics(labels, no_gc)
            result = {'auc_ablated_minus_original': percentile_interval(auc_without - auc_original, np=np, alpha=ALPHA)}
            count = np.bincount(ids, minlength=len(blocks)); denominator = weights @ count
            for metric in ('logloss', 'brier'):
                delta = metrics_without[metric + '_by_row'] - metrics_original[metric + '_by_row']
                sums = np.bincount(ids, weights=delta, minlength=len(blocks))
                values = np.divide(weights @ sums, denominator, out=np.full(REPLICATES, np.nan), where=denominator > 0)
                result[metric + '_ablated_minus_original'] = percentile_interval(values, np=np, alpha=ALPHA)
            intervals.append({'horizon_minutes': horizon, 'scheme': scheme, 'blocks': len(blocks), 'intervals': result})
    return {'per_horizon': scores, 'paired_intervals': intervals, 'per_interval_alpha': ALPHA,
        'scope': 'Prespecified every-tenth contiguous OOF batch on one reused development fold; no retraining.',
        'limitations': 'Paired calendar-block exchangeability is assumed. The intervention removes only the GC history projection, leaving GC context and other paths; it may create an out-of-distribution internal state. No economic edge or causal claim about retraining.'}


def require_exact_replay(actual, expected):
    require(isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
        and actual.dtype == expected.dtype and actual.shape == expected.shape
        and np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes(), 'EXACT_OOF_REPLAY')


def install_gc_projection_ablation(modules, expected_keys):
    """Suppress each named GC projection only; caller always removes handles."""
    require(isinstance(modules, torch.nn.ModuleDict) and tuple(modules.keys()) == tuple(expected_keys)
        and len(expected_keys) > 0, 'GC_MODULE_KEYS')
    calls = {key: 0 for key in expected_keys}; handles = []
    try:
        for key in expected_keys:
            def replace(_module, _args, output, *, name=key):
                require(isinstance(output, torch.Tensor), 'GC_PROJECTION_OUTPUT')
                calls[name] += 1
                return torch.zeros_like(output)
            handles.append(modules[key].register_forward_hook(replace))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles, calls


def _install_runtime_guard(out):
    """Keep the separate historical input guard and forbid network/TEST/writes."""
    def guard(event, args):
        require(event not in ('os.system', 'socket.connect', 'socket.bind', 'socket.sendto',
            'socket.getaddrinfo', 'os.exec', 'os.posix_spawn', 'os.fork', 'os.forkpty',
            'pickle.find_class', 'os.symlink'), 'NETWORK_PROCESS_PICKLE')
        if event == 'subprocess.Popen':
            command = tuple(args[1])
            require(command in (('git', 'rev-parse', 'HEAD'), ('git', 'status', '--porcelain')), 'PROCESS')
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            flags = args[2]
            writing = isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                require(path.is_absolute() and (path == out or out in path.parents) and '..' not in path.parts,
                    'OUTSIDE_OUTPUT_WRITE')
            if str(path).startswith(('/home/andre2/GX1_DATA/', '/home/andre2/GX1_RUNS/')):
                require(not any(part.upper() == 'TEST' or part.upper().startswith(('TEST_', 'TEST.')) for part in path.parts), 'TEST_PATH')
        if event in ('os.remove', 'os.rmdir', 'os.mkdir'):
            path = Path(os.fsdecode(args[0])); require(path == out or out in path.parents, 'OUTSIDE_OUTPUT_MUTATION')
        if event in ('os.rename', 'os.link'):
            paths = [Path(os.fsdecode(value)) for value in args[:2]]
            require(all(path == out or out in path.parents for path in paths), 'OUTSIDE_OUTPUT_MUTATION')
    sys.addaudithook(guard)


def run(*, plan_path, plan_sha256, device, out_dir):
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
            'wall_seconds', 'batch_size', 'batch_stride', 'test_data_used'}
            and plan['schema_version'] == 'gx1_direction_gc_mtf_intervention_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 1800 and plan['batch_size'] == BATCH
            and plan['batch_stride'] == BATCH_STRIDE and plan['test_data_used'] is False, 'PLAN')
        require(str(out_dir) == plan['output_dir'] and out_dir.parent == ROOT
            and out_dir.name == 'DIRECTION_GC_MTF_INTERVENTION_20261028_002'
            and ROOT.resolve(strict=True) == ROOT and not out_dir.exists() and not out_dir.is_symlink(), 'OUTPUT_SCOPE')
        require(plan['design']['path'] == str(REPO / 'docs/DIRECTION_GC_MTF_INTERVENTION_20261028_002.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_direction_gc_mtf_intervention_design_v1'
            and design['inputs'] == plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy_sha = common._hash_file(str(policy_path))['sha256']; policy_ref = {'path': str(policy_path), 'sha256': policy_sha}
        policy = binding.read_bound_json(policy_path, policy_sha); gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['checkpoint_reads_authorized'] is True and gate['model_forwards_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
            and all(gate[key] is False for key in ('optimizer_steps_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO / name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        require(set(plan['inputs']) == {'recipe', 'fit_result', 'checkpoint', 'oof_report', 'oof_predictions',
            'corrected_target', 'corrected_result', 'consumer_result', 'consumer_plan'}, 'INPUT_SET')
        meta = {}
        for key in ('recipe', 'fit_result', 'oof_report', 'corrected_result', 'consumer_result', 'consumer_plan'):
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
        require(len(launch['source_owners']) == 56, 'FIT_SOURCE_COUNT')
        for ref in launch['source_owners'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='fit source')
        control_snapshot = binding.capture_verified_file_snapshot()
    hardware, lock = cap.require_guarded_cuda_producer_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == hardware['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out_dir.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out_dir / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': source_head, 'execution': hardware, 'lock': lock})
    _install_runtime_guard(out_dir)
    start = time.monotonic(); previous_profile = sys.getprofile(); previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(*unused):
        raise RuntimeError('DIRECTION_GC_MTF_INTERVENTION_WALL_LIMIT')
    def forbid_updates(frame, event, arg):
        if event == 'call':
            name, path = frame.f_code.co_name, frame.f_code.co_filename.replace('\\', '/')
            require(not (('/torch/optim/' in path and name in ('step', '_init_group'))
                or ('/torch/autograd/' in path and name in ('backward', '_engine_run_backward'))), 'FORBIDDEN_UPDATE')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(forbid_updates)
    stage = out_dir / '.OUTPUT.staging'
    handles = []
    try:
        stage.mkdir(exist_ok=False)
        context = window._launch_metadata(launch, common)
        refs = [plan['inputs'][name] for name in ('checkpoint', 'oof_predictions', 'corrected_target')]
        require(len({ref['path'] for ref in refs}) == len(refs), 'DISTINCT_PAYLOADS')
        for ref in refs:
            before = common._stat(ref['path']); require(common._hash_file(ref['path']) == ref, 'PAYLOAD_SHA')
            context['info']['metadata'][ref['path']] = {'ref': dict(ref), 'stat': before}
        freeze_binding = {'source_head': source_head, 'policy': policy_ref}
        common._freeze(launch, freeze_binding, context)
        denied = common._install_read_guard(context, out_dir, out_dir, dense_target=context['dense_target'])
        prepared, targets, view, _context_map, _order = window._prepare(context, launch, np, dense_minutes=True)
        with np.load(plan['inputs']['oof_predictions']['path'], allow_pickle=False) as archive:
            names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up')
            require(all(name in archive.files for name in names), 'OOF_FIELDS')
            saved = {name: archive[name].copy() for name in names}
        fold = launch['fold']; first, last = fold['oof_start_row'], fold['oof_end_row']; sl = slice(first, last)
        for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            require(np.array_equal(saved[name], prepared['arrays'][name][sl]), 'OOF_AXIS:' + name)
        for name in ('terminal_time_ns', 'observed', 'sign'):
            require(np.array_equal(saved[name], targets[name][sl]), 'OOF_ORIGINAL_TARGET:' + name)
        require(saved['logit_up'].dtype == np.float32 and saved['logit_up'].shape == (last - first, 60)
            and np.isfinite(saved['logit_up']).all(), 'OOF_LOGITS')
        with np.load(plan['inputs']['corrected_target']['path'], allow_pickle=False) as archive:
            names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')
            require(all(name in archive.files for name in names), 'CORRECTED_FIELDS')
            corrected_arrays = {name: archive[name].copy() for name in names}
        for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            require(np.array_equal(corrected_arrays[name], prepared['arrays'][name]), 'CORRECTED_AXIS:' + name)
        for name in ('terminal_time_ns', 'observed'):
            require(np.array_equal(corrected_arrays[name], targets[name]), 'CORRECTED_CLOCK_MASK:' + name)
        chosen = selected_batches(last - first); selection = np.concatenate(chosen)
        require(len(selection) > 0 and len(selection) < last - first and np.all(np.diff(selection) > 0), 'SELECTION')
        trainer._set_deterministic(launch['numerical_recipe']['seed'], torch.device('cuda', 0),
            launch['numerical_recipe']['precision_policy'])
        # The producer guard permits one CPU numerical worker; the trainer
        # helper configures the training count, so restore the producer count.
        torch.set_num_threads(1)
        require(torch.get_num_threads() == 1, 'PRODUCER_NUMERICAL_THREADS')
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
        original, ablated = [], []; hook_calls = {key: 0 for key in keys}
        for indices in chosen:
            require(time.monotonic() - start < plan['wall_seconds'], 'WALL_LIMIT')
            parent_ids = saved['parent_rows'][indices]
            raw, target = window._batch(prepared, targets, view, parent_ids, np)
            features, _target = window._device_batch(raw, target, torch, torch.device('cuda', 0))
            z0 = window._forward(model, features, kind='REPLAY', counters=counters,
                ceiling=2 * len(chosen), torch=torch).detach().cpu().numpy().copy()
            expected = np.ascontiguousarray(saved['logit_up'][indices])
            require_exact_replay(z0, expected)
            handles, calls = install_gc_projection_ablation(modules, keys)
            try:
                z1 = window._forward(model, features, kind='ABLATION', counters=counters,
                    ceiling=2 * len(chosen), torch=torch).detach().cpu().numpy().copy()
            finally:
                for handle in handles:
                    handle.remove()
                handles = []
            require(all(value == 1 for value in calls.values()) and np.isfinite(z1).all(), 'HOOK_CALLS')
            for key, value in calls.items(): hook_calls[key] += value
            original.append(z0); ablated.append(z1)
            del raw, target, features, _target, z0, z1
        z0, z1 = np.concatenate(original), np.concatenate(ablated)
        require(counters['forward_calls'] == counters['forward_returns'] == 2 * len(chosen)
            and all(value == len(chosen) for value in hook_calls.values()) and not denied, 'CALL_ACCOUNTING')
        require(state_owner.canonical_model_state_sha256(model.state_dict()) == model_sha
            and state_owner._tree_sha(dict(model.named_buffers())) == buffer_sha
            and all(parameter.grad is None for parameter in model.parameters()), 'MODEL_STATE_UNCHANGED')
        rows = first + selection
        metrics = paired_metrics(saved['decision_ns'][selection], corrected_arrays['observed'][rows],
            corrected_arrays['sign'][rows], corrected_arrays['terminal_time_ns'][rows], z0, z1, cutoff=fold['oof_cutoff_ns'])
        logit_change = np.abs(z1.astype(np.float64) - z0.astype(np.float64))
        common._freeze(launch, freeze_binding, context); control_snapshot.require_unchanged()
        require(cap.require_capped_lock_ancestry() == lock and not denied, 'FINAL_SOURCE_LOCK')
        output = {'schema_version': 'gx1_direction_gc_mtf_intervention_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': source_head, 'fold': fold, 'model_state_sha256': model_sha,
            'selection': {'policy': 'every tenth original OOF batch', 'batch_size': BATCH,
                'batch_stride': BATCH_STRIDE, 'selected_parents': len(selection), 'selected_batches': len(chosen)},
            'exact_saved_oof_logit_bytes_reproduced': True, 'corrected_target_identity': corrected['target_identity'],
            'intervention': 'All 25 GC history projection outputs set to zero after projection; current GC context remains.',
            'mean_absolute_logit_change': float(logit_change.mean()), 'max_absolute_logit_change': float(logit_change.max()),
            'forward_calls': counters['forward_calls'], 'optimizer_steps': 0, 'model_unchanged': True,
            'metrics': metrics, 'test_data_used': False, 'economic_edge_qualified': False}
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
