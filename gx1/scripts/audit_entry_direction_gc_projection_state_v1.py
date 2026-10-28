"""Compare frozen initial/trained direction projection parameters; no model forward."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_gc_projection_state_audit_20261028_001'
SOURCE_PATHS = (
    'gx1/scripts/audit_entry_direction_gc_projection_state_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_entry_direction_gc_projection_state_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_GC_PROJECTION_STATE_' + reason)


def matrix_change(first, last, *, shape, rank=False):
    """Aggregate exact FP32 parameter change; never infer feature usefulness."""
    require(isinstance(first, torch.Tensor) and isinstance(last, torch.Tensor)
        and first.device.type == last.device.type == 'cpu' and first.dtype == last.dtype == torch.float32
        and tuple(first.shape) == tuple(last.shape) == tuple(shape)
        and bool(torch.isfinite(first).all()) and bool(torch.isfinite(last).all()), 'TENSOR')
    a = first.detach().numpy(); b = last.detach().numpy()
    delta = b.astype(np.float64) - a.astype(np.float64)
    out = {'shape': list(shape), 'changed_elements': int(np.count_nonzero(a != b)),
        'total_elements': int(a.size), 'initial_nonzero': int(np.count_nonzero(a)),
        'trained_nonzero': int(np.count_nonzero(b)), 'delta_l2': float(np.linalg.norm(delta)),
        'initial_l2': float(np.linalg.norm(a.astype(np.float64))),
        'trained_l2': float(np.linalg.norm(b.astype(np.float64)))}
    if rank:
        require(a.ndim == 2, 'RANK_MATRIX')
        out['initial_numerical_rank'] = int(np.linalg.matrix_rank(a))
        out['trained_numerical_rank'] = int(np.linalg.matrix_rank(b))
        out['rank_method'] = 'numpy matrix_rank default FP32 tolerance'
    return out


def analyze(first, last, profile, *, width):
    require(type(width) is int and width > 0 and isinstance(first, dict) and isinstance(last, dict), 'HEADER')
    require(set(first) == set(last), 'MODEL_KEY_PARITY')
    def pair(name, shape, rank=False):
        require(name in first and name in last, 'MISSING:' + name)
        return matrix_change(first[name], last[name], shape=shape, rank=rank)
    families = profile['local_specialists']
    ctx = {}
    for family in families:
        count = len(profile['ctx_cont_indices_by_specialist'][family])
        # This profile's active context has no nominal context coordinates.
        if count:
            ctx[family] = pair('backbone.specialist_ctx_cont_proj.' + family + '.weight',
                (width, count), rank=family == 'smc_liquidity_encoder')
    gc = {}
    for tf in ('M5', 'M15', 'H1', 'H4', 'D1'):
        gc[tf] = {}
        for family in profile['mtf_specialists']:
            count = len(profile['gc_mtf_family_indices'][tf][family])
            if not count:
                continue
            key = tf.lower() + '__' + family
            prefix = 'backbone.gc_mtf_feature_context_gate.' + key
            gc[tf][family] = {
                'projection': pair('backbone.gc_mtf_family_proj.' + key + '.weight',
                    (width, count), rank=tf == 'M5' and family == 'smc_liquidity_encoder'),
                'context_gate_weight': pair(prefix + '.weight', (count, width)),
                'context_gate_bias': pair(prefix + '.bias', (count,)),
            }
    residuals = {}
    for name in ('specialist_out', 'cross_tf_out', 'family_tf_cooperation_out'):
        residuals[name] = pair('backbone.' + name + '.weight', (width, width))
    return {'context_projection_weights': ctx, 'gc_projection_and_context_gates': gc,
        'family_fusion_output_weights': residuals}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_decision_token_v1 import ENTRY_DECISION_TOKEN_DIM
    from gx1.features.entry_specialist_feature_groups_v1 import (
        compile_gc_mtf_geometry_entry_execution_profile, model_native_context_specialist_routing_contract,
    )
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    # Resolve the existing digest implementation before the file/network guard.
    state_owner._tree_sha({})
    sample = io.BytesIO(); torch.save({'x': torch.tensor([1.])}, sample); sample.seek(0)
    require(torch.equal(torch.load(sample, map_location='cpu', weights_only=True)['x'], torch.tensor([1.])), 'WEIGHTS_ONLY_RUNTIME')
    require(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED')
    initial = candle.git_state(); require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO / 'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir', 'wall_seconds', 'test_data_used'}
            and plan['schema_version'] == 'gx1_direction_gc_projection_state_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO / 'docs/DIRECTION_GC_PROJECTION_STATE_AUDIT_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_direction_gc_projection_state_design_v1'
            and design['inputs'] == plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']); gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['checkpoint_reads_authorized'] is True and gate['max_invocations'] == 1
            and gate['claimed_invocations'] == gate['completed_invocations'] == 0 and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
            and all(gate[k] is False for k in ('model_forwards_authorized', 'optimizer_steps_authorized',
                'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO / name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        require(set(plan['inputs']) == {'recipe', 'fit_result', 'prefix_normalization', 'initial_checkpoint', 'final_checkpoint'}, 'INPUT_SET')
        meta = {}
        for key in ('recipe', 'fit_result', 'prefix_normalization'):
            ref = plan['inputs'][key]; meta[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        recipe, result, normal = meta['recipe'], meta['fit_result'], meta['prefix_normalization']
        require(recipe['numerical_recipe']['dense_minute_supervision'] is True and result['outcome'] == 'COMPLETE'
            and result['test_data_used'] is False and result['recipe'] == plan['inputs']['recipe']
            and result['own_initial_state'] == plan['inputs']['initial_checkpoint']
            and result['full_state'] == plan['inputs']['final_checkpoint']
            and result['feature_view_identity']['prefix_normalization']['binding'] == plan['inputs']['prefix_normalization'], 'FIT_LINEAGE')
        require(len(recipe['source_owners']) == 56, 'FIT_SOURCE_COUNT')
        for ref in recipe['source_owners'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='fit source')
        profile = compile_gc_mtf_geometry_entry_execution_profile(normal['gc_mtf_history'])
        routes = model_native_context_specialist_routing_contract()
        require(ENTRY_DECISION_TOKEN_DIM == 128 and len(profile['local_specialists']) == 8
            and all(not (set(profile['ctx_cont_indices_by_specialist'][f]) & set(routes['ctx_cont_nominal_indices'][f]))
                for f in profile['local_specialists']), 'PROFILE_GEOMETRY')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir'])
    require(ROOT.resolve(strict=True) == ROOT and out.parent == ROOT
        and out.name == 'DIRECTION_GC_PROJECTION_STATE_AUDIT_20261028_001'
        and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    payloads = [plan['inputs'][k] for k in ('initial_checkpoint', 'final_checkpoint')]
    require(len({ref['path'] for ref in payloads}) == 2, 'DISTINCT_CHECKPOINTS')
    metadata = [plan_ref['path'], policy_ref['path'], plan['design']['path']]
    metadata += [plan['inputs'][k]['path'] for k in ('recipe', 'fit_result', 'prefix_normalization')]
    sys.addaudithook(candle.make_guard(out, [ref['path'] for ref in payloads], metadata,
        source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started = time.monotonic(); previous_handler = signal.getsignal(signal.SIGALRM); previous_profile = sys.getprofile()
    def expired(*unused):
        raise RuntimeError('DIRECTION_GC_PROJECTION_STATE_WALL_LIMIT')
    def deny_model_calls(frame, event, arg):
        candle.no_queries_or_models(frame, event, arg)
        candle.no_native_models(frame, event, arg)
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(deny_model_calls)
    stage = out / '.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:
                candle.regular(ref['path']); binding.require_binding(ref, label='checkpoint')
            states = [torch.load(ref['path'], map_location='cpu', weights_only=True) for ref in payloads]
            for state in states:
                require(isinstance(state, dict) and set(state) == state_owner.STATE_KEYS
                    and state['schema_version'] == state_owner.SCHEMA_VERSION
                    and state['contract_sha256'] == state_owner._tree_sha({k: v for k, v in state.items() if k != 'contract_sha256'})
                    and state['model_state_sha256'] == state_owner.canonical_model_state_sha256(state['model_state'])
                    and state['binding']['feature_view_identity'] == result['feature_view_identity'], 'STATE_SEAL')
            first, last = states
            require(first['cursor'] == first['optimizer_updates'] == 0
                and last['cursor'] == result['parent_cursor'] == 47602
                and last['optimizer_updates'] == result['optimizer_updates'] == 2976
                and last['contract_sha256'] == result['durable_checkpoint']['contract_sha256']
                and state_owner._tree_sha(first['binding']) == state_owner._tree_sha(last['binding']), 'TRAIN_BOUNDARY')
            comparison = analyze(first['model_state'], last['model_state'], profile, width=ENTRY_DECISION_TOKEN_DIM)
            input_snapshot = binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock
            and not torch.cuda.is_initialized(), 'FINAL_SOURCE_LOCK')
        result_out = {'schema_version': 'gx1_direction_gc_projection_state_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': args.source_head, 'profile_sha256': profile['contract_sha256'],
            'matched_fit_source_owners': len(recipe['source_owners']), 'initial_updates': 0, 'final_updates': 2976,
            'trained_parent_cursor': 47602, 'comparison': comparison,
            'scope': 'Two hash-bound first-prefix direction checkpoint parameter states; no input data, model forward or optimizer.',
            'limits': 'Parameter updates and numerical matrix rank do not establish output influence, useful interaction, generalization or economic edge.',
            'new_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
        publish(stage / 'RESULT.json', result_out)
        publish(stage / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
            'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - started, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out / 'OUTPUT')
        publish(out / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'result_path': str(out / 'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out / 'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False})
        raise
    finally:
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
