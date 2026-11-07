"""One-use physical TRAIN witness for the factorized time-query objective.

Reuses the qualified fold-1 input and target tape. Two fixed fit rows test a
known M5 side and an ACTIVE row with unknown M5 side; one fresh forward and
backward are allowed. No optimizer, checkpoint, fit, TEST or trade is touched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.audit import verify_market_state_time_query_zero_step_v1 as old
from gx1.contracts import market_state_time_query_binding_v1 as binding
from gx1.contracts.market_state_time_query_loss_v2 import factorized_time_query_loss_contract
from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract
from gx1.contracts.market_state_time_query_scorer_v2 import factorized_time_query_scorer_contract

REPO, RUNS, POLICY = old.REPO, old.RUNS, old.POLICY
STAGE = 'market_state_time_query_v2_zero_step_20261107_001'
SCHEMA = 'gx1_market_state_time_query_v2_zero_step_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
FOLD_ID = 1
PASS_INDEX = old.PASS_INDEX
SEED = old.SEED
PARENT_ROWS = [0, 2]
ISSUE_NS = [1714521900000000000, 1714522500000000000]
PAIR_AXIS_SHA256 = 'd2070a9fe49ba379cb78449c24df8d2e02980cc26f0537d689ddcac35a22380f'
EXPECTED_QUERY_K = [[1, 3, 11, 40, 211], [1, 2, 2, 5, 92]]
EXPECTED_ALLOWED = [[5, 5, 1, 7, 7], [1, 1, 7, 7, 7]]
EXPECTED_ELIGIBLE = [[True, True, True, False, False], [True, True, False, False, False]]
EXPECTED_MATURE = [[True] * 5, [True] * 5]
EXPECTED_VIEW_SHA256 = '5ceac9557d37234090805b8b0edde0f1444ab4e6a140f7b956c6120811e592ac'
MEMORY_BYTES, SWAP_BYTES, WALL_SECONDS = old.MEMORY_BYTES, old.SWAP_BYTES, old.WALL_SECONDS
MAX_JSON_BYTES = old.MAX_JSON_BYTES
EXECUTION = {
    'class': 'producer', 'cuda_producer': True, 'device': 'cuda',
    'module': 'gx1.scripts.validate_entry_model_native_technical_checkpoint_v1',
    'flag': '--market-state-time-query-v2-zero-step-plan',
    'memory': '12G', 'swap': '512M', 'model_forwards': 1,
    'backward_calls': 1, 'optimizer_steps': 0,
}
LIMITS = {'memory_max': MEMORY_BYTES, 'memory_high': MEMORY_BYTES,
          'swap': SWAP_BYTES, 'max_wall_seconds': WALL_SECONDS}
FIXED = old.FIXED
SOURCE_PATHS = old.SOURCE_PATHS | frozenset({
    'gx1/audit/verify_market_state_time_query_v2_zero_step_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_loss_v2.py',
    'gx1/contracts/market_state_time_query_batch_v2.py',
    'gx1/contracts/market_state_time_query_scorer_v1.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
})
V2_MODEL_PATH = 'gx1/contracts/market_state_time_query_model_v2.py'
V2_LOSS_PATH = 'gx1/contracts/market_state_time_query_loss_v2.py'


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('TIME_QUERY_V2_ZERO_STEP_' + reason)


def _sha(path: Path) -> str:
    return old._sha(path)


def _ref(path: Path, sha256: str) -> dict:
    return old._ref(path, sha256)


def _fixed_refs() -> dict:
    return {key: old._fixed_ref(key) for key in FIXED}


def _clean_head() -> str:
    return old._clean_head()


def _checked_plan(path: Path, sha256: str) -> tuple[dict, dict]:
    _need(path == PLAN_PATH and path.is_absolute() and path.is_file()
          and not path.is_symlink() and type(sha256) is str and len(sha256) == 64
          and _sha(path) == sha256, 'PLAN_HASH')
    ref = _ref(path, sha256)
    return old._read_ref(ref), ref


def _source_refs(plan: dict, receipts: dict) -> dict:
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    refs = old._source_refs(plan, receipts, FOLD_ID)
    _need(set(refs) == views.SOURCE_REF_KEYS | views.TIME_QUERY_SOURCE_REF_KEYS,
          'V1_SOURCE_REF_SET')
    del refs['time_query_model_source'], refs['time_query_loss_source']
    for key, path in (('time_query_v2_model_source', V2_MODEL_PATH),
                      ('time_query_v2_loss_source', V2_LOSS_PATH)):
        refs[key] = _ref(REPO / path, plan['source_owners'][path])
        _need(_sha(REPO / path) == refs[key]['sha256'], 'V2_SOURCE_REF:' + key)
    _need(set(refs) == views.SOURCE_REF_KEYS | views.TIME_QUERY_V2_SOURCE_REF_KEYS,
          'V2_SOURCE_REF_SET')
    return refs


def _bind_view(plan: dict, receipts: dict) -> dict:
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    fold1 = receipts['fold1']
    view = views.bind_native_direction_feature_view(
        prefix_normalization=fold1['normal'],
        prefix_normalization_ref=old._fixed_ref('fold1_normalization'),
        constructor_metadata=receipts['constructor'],
        constructor_metadata_ref=old._fixed_ref('constructor'),
        source_bindings=_source_refs(plan, receipts), max_batch_rows=16,
        full_feature_profile=True, market_state_time_query_v2=True,
        unobserved_units_profile=fold1['profile'])
    _need(view['contract_sha256'] == EXPECTED_VIEW_SHA256
          == plan['expected_view_sha256']
          and view['research_objective']['contract_sha256']
          == factorized_time_query_model_contract()['contract_sha256']
          == plan['expected_model_contract_sha256'], 'VIEW_IDENTITY')
    return view


def _target_identity() -> tuple[binding.BoundTimeQueryTargets, dict]:
    rows = np.asarray(PARENT_ROWS, dtype=np.int64)
    issues = np.asarray(ISSUE_NS, dtype=np.int64)
    _need(hashlib.sha256(np.column_stack((rows, issues)).tobytes()).hexdigest()
          == PAIR_AXIS_SHA256, 'PAIR_AXIS_SHA256')
    bound = binding.load_bound_time_query_targets()
    truth = bound.gather_fit_sampled(rows, issues, fold_id=FOLD_ID,
                                     pass_index=PASS_INDEX, seed=SEED)
    _need(truth['parent_rows'].tolist() == PARENT_ROWS
          and truth['issue_ns'].tolist() == ISSUE_NS
          and truth['query_k'].tolist() == EXPECTED_QUERY_K
          and truth['allowed'].tolist() == EXPECTED_ALLOWED
          and truth['eligible'].tolist() == EXPECTED_ELIGIBLE
          and truth['mature'].tolist() == EXPECTED_MATURE
          and truth['allowed'][:, 0].tolist() == [5, 1]
          and bool(truth['eligible'][:, 0].all() and truth['mature'][:, 0].all()),
          'TARGET_PAIR')
    return bound, truth


def _qualify_plan(plan: dict, plan_ref: dict, policy: dict | None, *,
                  head: str, out_dir: Path, device: str, cap: dict | None) -> None:
    _need(isinstance(plan, dict) and plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head], cwd=REPO,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
          and plan.get('output_root') == str(OUT_ROOT) == str(out_dir)
          and out_dir.is_absolute() and not out_dir.exists()
          and device == 'cuda' and plan.get('fold_id') == FOLD_ID
          and plan.get('partition') == 'fit'
          and plan.get('sample_parent_rows') == PARENT_ROWS
          and plan.get('sample_issue_ns') == ISSUE_NS
          and plan.get('pair_axis_sha256') == PAIR_AXIS_SHA256
          and plan.get('expected_query_k') == EXPECTED_QUERY_K
          and plan.get('expected_allowed') == EXPECTED_ALLOWED
          and plan.get('expected_eligible') == EXPECTED_ELIGIBLE
          and plan.get('expected_mature') == EXPECTED_MATURE
          and plan.get('pass_index') == PASS_INDEX and plan.get('seed') == SEED
          and plan.get('max_invocations') == 1
          and plan.get('max_json_bytes') == MAX_JSON_BYTES
          and plan.get('model_forwards_authorized') == 1
          and plan.get('backward_calls_authorized') == 1
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False
          and plan.get('execution') == EXECUTION and plan.get('limits') == LIMITS
          and plan.get('input_refs') == _fixed_refs()
          and plan.get('expected_view_sha256') == EXPECTED_VIEW_SHA256
          and plan.get('expected_model_contract_sha256')
              == factorized_time_query_model_contract()['contract_sha256']
          and plan.get('expected_loss_contract_sha256')
              == factorized_time_query_loss_contract()['contract_sha256']
          and plan.get('expected_scorer_contract_sha256')
              == factorized_time_query_scorer_contract()['contract_sha256'],
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(isinstance(digest, str) and len(digest) == 64
                  and _sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    if cap is None:
        return
    _need(all(cap.get(key) == LIMITS[key]
              for key in ('memory_max', 'memory_high', 'swap'))
          and type(cap.get('pids')) is int and cap['pids'] <= 128
          and cap.get('class') == 'producer'
          and cap.get('execution_mode') == 'cuda_producer'
          and isinstance(cap.get('cgroup_path'), str)
          and isinstance(cap.get('lock'), dict)
          and isinstance(cap['lock'].get('scope'), str)
          and cap['lock']['scope'].startswith('/')
          and cap['lock']['scope'].endswith('.scope')
          and cap['cgroup_path'] == '/sys/fs/cgroup' + cap['lock']['scope'],
          'CAP_AND_LOCK')
    gate = policy.get(STAGE) if isinstance(policy, dict) else None
    expected = {
        'status': 'ADMITTED', 'decision': 'ADMITTED', 'plan': plan_ref,
        'execution_authorized': True, 'data_reads_authorized': True,
        'model_forwards_authorized': 1, 'backward_calls_authorized': 1,
        'optimizer_steps_authorized': 0, 'training_authorized': False,
        'test_authorized': False, 'claimed_invocations': 0,
        'completed_invocations': 0, 'max_invocations': 1,
        'output_root': str(out_dir), 'relaunch_allowed': False,
        'checkpoint_reads_authorized': False, 'broker_access_allowed': False,
        'test_data_used': False, 'memory_max': '12G', 'swap_max': '512M',
        'wall_seconds': WALL_SECONDS,
    }
    _need(isinstance(policy, dict) and policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and isinstance(gate, dict)
          and all(gate.get(key) == value for key, value in expected.items()),
          'POLICY')


def _inspect_inputs(plan: dict) -> tuple[dict, dict, binding.BoundTimeQueryTargets]:
    runner = REPO / old.RUNNER_SOURCE
    receipts = old._inspect_receipts(reviewed_runner_ref=_ref(
        runner, plan['source_owners'][old.RUNNER_SOURCE]))
    view = _bind_view(plan, receipts)
    bound, _ = _target_identity()
    return receipts, view, bound


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    """Exact receipt/view/target gate with no model, feature payload or CLAIM."""
    head = _clean_head()
    plan, plan_ref = _checked_plan(Path(plan_path), plan_sha256)
    _qualify_plan(plan, plan_ref, None, head=head, out_dir=Path(out_dir),
                  device='cuda', cap=None)
    receipts, view, _ = _inspect_inputs(plan)
    return {
        'status': 'READY_NO_CLAIM_NO_RAW_FEATURE_PAYLOAD', 'plan': plan_ref,
        'source_head': head, 'source_owner_refs_checked': len(plan['source_owners']),
        'view_sha256': view['contract_sha256'],
        'model_contract_sha256': factorized_time_query_model_contract()['contract_sha256'],
        'loss_contract_sha256': factorized_time_query_loss_contract()['contract_sha256'],
        'scorer_contract_sha256': factorized_time_query_scorer_contract()['contract_sha256'],
        'pair_axis_sha256': PAIR_AXIS_SHA256,
        'fold1_lineage': receipts['fold1']['lineage'],
        'tape_sha256_checked': old._fixed_ref('candidate_tape')['sha256'],
        'claim_created': False, 'model_forwards': 0, 'backward_calls': 0,
        'optimizer_steps': 0, 'test_data_used': False,
    }


def _preflight(plan_path: Path, plan_sha256: str, device: str,
               out_dir: Path) -> tuple[dict, dict, dict, str, dict, dict, dict,
                                       binding.BoundTimeQueryTargets]:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_producer_execution, require_capped_lock_ancestry,
    )
    head = _clean_head()
    plan, plan_ref = _checked_plan(plan_path, plan_sha256)
    policy_ref = _ref(POLICY, _sha(POLICY))
    policy = old._read_ref(policy_ref, limit=8 * 1024**2)
    cap = require_guarded_cuda_producer_execution()
    cap['lock'] = require_capped_lock_ancestry()
    _qualify_plan(plan, plan_ref, policy, head=head, out_dir=out_dir,
                  device=device, cap=cap)
    receipts, view, bound = _inspect_inputs(plan)
    return plan, plan_ref, policy_ref, head, cap, receipts, view, bound


def _source_bookend(plan: dict, plan_sha256: str, policy_sha256: str,
                    head: str) -> None:
    _need(_clean_head() == head and _sha(PLAN_PATH) == plan_sha256
          and _sha(POLICY) == policy_sha256
          and all(_sha(REPO / path) == digest
                  for path, digest in plan['source_owners'].items())
          and all(_sha(path) == digest for path, digest in FIXED.values()),
          'SOURCE_BOOKEND')


def _witness(plan: dict, cap: dict, receipts: dict, view: dict,
             bound: binding.BoundTimeQueryTargets) -> dict:
    import torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_batch_v2 import FEATURE_KEYS, evaluate_fit_batch
    from gx1.contracts.market_state_time_query_model_v2 import (
        FactorizedTimeQueryMarketStateModel, factorized_log_probabilities,
    )
    from gx1.contracts.market_state_time_query_scorer_v2 import factorized_probabilities
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    chain = receipts['provider_chain']
    provider_plan = chain['provider_plan']
    provider = mature._provider_module()
    info = provider.validate_metadata(provider_plan)
    prepared = mature._prepare_inputs(provider, provider_plan, info, chain['gc_layout'])
    mature._validate_axes(receipts['mature_plan']['cohorts'], prepared['axes'])
    axes = prepared['axes']
    selected = np.asarray(PARENT_ROWS, dtype=np.int64)
    issues = np.asarray(ISSUE_NS, dtype=np.int64)
    at = np.searchsorted(axes['parent_rows'], selected)
    _need(np.array_equal(axes['parent_rows'][at], selected)
          and np.array_equal(axes['decision_ns'][at], issues)
          and int(at[-1]) < receipts['fold1']['cohort']['joined_fit_rows'],
          'EXACT_FOLD1_SAMPLE')
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        axes[key].setflags(write=False)
    sequence = prepared['entry_sequence_source']
    for key in ('signal', 'times_ns', 'sample_positions'):
        sequence[key].setflags(write=False)
    for key in ('snap', 'ctx_cont', 'ctx_cat'):
        prepared['current'][key].setflags(write=False)
    profile = receipts['fold1']['profile']
    raw = views.gather_native_direction_feature_batch(
        view, parent_rows=axes['parent_rows'],
        bar_start_ns=axes['bar_start_ns'], decision_ns=axes['decision_ns'],
        entry_sequence_source=sequence,
        snap_x=prepared['current']['snap'],
        ctx_cont=prepared['current']['ctx_cont'],
        ctx_cat=prepared['current']['ctx_cat'],
        multi_tf_sources=prepared['cache'],
        gc_surfaces=prepared['gc']['surfaces'],
        gc_layout=prepared['gc']['layout'],
        batch_parent_rows=selected, unobserved_units_profile=profile)
    _need(set(raw['features']) == FEATURE_KEYS
          and raw['witness']['parent_rows'] == PARENT_ROWS
          and raw['witness']['decision_ns'] == ISSUE_NS,
          'FEATURE_ONLY_BATCH')
    mature._release_pages(provider, prepared)
    device = torch.device('cuda')
    _need(torch.cuda.is_available(), 'CUDA_AVAILABLE')
    trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
    model = views.build_native_direction_research_model(
        view, prefix_normalization=receipts['fold1']['normal'],
        constructor_metadata=receipts['constructor'], device=device,
        unobserved_units_profile=profile)
    _need(type(model) is FactorizedTimeQueryMarketStateModel
          and model.feature_view_identity['contract_sha256'] == EXPECTED_VIEW_SHA256,
          'FRESH_V2_MODEL')
    model.train()
    initial_state = canonical_model_state_sha256(model.state_dict())
    tensors = {}
    for key, value in raw['features'].items():
        tensor = torch.from_numpy(value).to(device)
        if tensor.is_floating_point():
            tensor.requires_grad_(True)
        tensors[key] = tensor
    torch.cuda.reset_peak_memory_stats()
    with trainer._training_mha_fastpath_context(model.backbone), \
         trainer._training_autocast_context(), \
         trainer._training_model_finite_check_context():
        _need(not torch.backends.mha.get_fastpath_enabled(), 'CANONICAL_MHA')
        outcome = evaluate_fit_batch(model, tensors, bound, selected, issues,
                                     fold_id=FOLD_ID, pass_index=PASS_INDEX,
                                     seed=SEED)
    _need(outcome['count'] == 5 and outcome['skip_update'] is False
          and outcome['query_k'].tolist() == EXPECTED_QUERY_K
          and outcome['allowed'].tolist() == EXPECTED_ALLOWED
          and outcome['eligible'].tolist() == EXPECTED_ELIGIBLE
          and outcome['mature'].tolist() == EXPECTED_MATURE
          and outcome['per_tf_counts'] == {'M5': 2, 'M15': 2,
                                           'H1': 1, 'H4': 0, 'D1': 0}
          and outcome['model_contract_sha256']
              == factorized_time_query_model_contract()['contract_sha256']
          and outcome['loss_contract_sha256']
              == factorized_time_query_loss_contract()['contract_sha256']
          and bool(torch.isfinite(outcome['loss'])), 'TARGET_LOSS_BINDING')
    primitive = outcome['activity_side_logits']
    _need(torch.equal(outcome['logits'], factorized_log_probabilities(primitive)),
          'OUTPUT_FACTORISATION')
    primitive.retain_grad()
    outcome['loss'].backward()
    direct = primitive.grad
    _need(direct is not None and direct.shape == (2, 5, 2)
          and bool(torch.isfinite(direct).all()), 'DIRECT_GRADIENT')
    allowed = torch.from_numpy(outcome['allowed']).to(device)
    eligible = torch.from_numpy(outcome['eligible']).to(device)
    mature_mask = torch.from_numpy(outcome['mature']).to(device)
    side_known = eligible & mature_mask & ((allowed == 1) | (allowed == 4))
    _need(bool(torch.all(direct[..., 1][~side_known] == 0))
          and bool(direct[0, 0, 1] == 0)
          and bool(direct[0, 0, 0] != 0)
          and bool(direct[1, 0, 1] != 0)
          and bool(direct[1, 0, 0] != 0),
          'M5_UNKNOWN_SIDE_AND_KNOWN_SIDE_GRADIENT')
    head = old._gradient_summary(model.query_head.weight.grad)
    projection = old._gradient_summary(model.query_projection[0].weight.grad)
    backbone = old._gradient_summary(model.backbone.entry_q_joint_in.weight.grad)
    _need(all(item['present'] and item['finite'] and item['nonzero_count'] > 0
              for item in (head, projection, backbone)), 'BACKWARD_CONSUMPTION')
    final_state = canonical_model_state_sha256(model.state_dict())
    _need(initial_state == final_state, 'NO_WEIGHT_UPDATE')
    logits_np = primitive.detach().cpu().numpy()
    classes, active, conditional_up = factorized_probabilities(logits_np)
    _need(np.isfinite(classes).all() and np.isfinite(active).all()
          and np.isfinite(conditional_up).all(), 'INITIAL_PROBABILITIES')
    provider.validate_metadata(provider_plan)
    return {
        'decision': 'PASS_FACTORIZED_V2_INITIAL_OUTPUT_AND_DIRECT_GRADIENT_NO_UPDATE',
        'scope_limit': 'One fresh two-row TRAIN fit witness only; no model fit, OOF, calibration, direction value, Entry quality or trading economics.',
        'fold_id': FOLD_ID, 'pass_index': PASS_INDEX, 'seed': SEED,
        'sample_parent_rows': PARENT_ROWS, 'sample_issue_ns': ISSUE_NS,
        'pair_axis_sha256': PAIR_AXIS_SHA256,
        'query_k': outcome['query_k'].tolist(),
        'allowed': outcome['allowed'].tolist(),
        'eligible': outcome['eligible'].tolist(),
        'mature': outcome['mature'].tolist(),
        'per_tf_counts': outcome['per_tf_counts'],
        'informative_count': outcome['count'],
        'loss': float(outcome['loss'].detach().item()),
        'view_sha256': view['contract_sha256'],
        'model_contract_sha256': outcome['model_contract_sha256'],
        'loss_contract_sha256': outcome['loss_contract_sha256'],
        'scorer_contract_sha256': factorized_time_query_scorer_contract()['contract_sha256'],
        'normalization': old._fixed_ref('fold1_normalization'),
        'candidate_tape': old._fixed_ref('candidate_tape'),
        'feature_witness': raw['witness'],
        'feature_shapes': {key: list(value.shape) for key, value in tensors.items()},
        'initial_raw_logits': logits_np.tolist(),
        'initial_activity_probability_by_tf': active.tolist(),
        'initial_conditional_up_probability_by_tf': conditional_up.tolist(),
        'initial_m5_class_probabilities': classes[:, 0, :].tolist(),
        'direct_m5_gradients': {
            'unknown_activity': float(direct[0, 0, 0].item()),
            'unknown_side': float(direct[0, 0, 1].item()),
            'known_activity': float(direct[1, 0, 0].item()),
            'known_side': float(direct[1, 0, 1].item()),
        },
        'query_head_gradient': head, 'query_projection_gradient': projection,
        'backbone_gradient': backbone,
        'initial_model_state_sha256': initial_state,
        'final_model_state_sha256': final_state,
        'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'cuda_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
        'cgroup_path': cap['cgroup_path'],
        'model_forwards': 1, 'backward_calls': 1, 'optimizer_steps': 0,
        'ema_updates': 0, 'scheduler_steps': 0,
        'checkpoint_loaded': False, 'test_data_used': False,
    }


def run(plan_path: Path, plan_sha256: str, device: str, out_dir: Path) -> dict:
    """Admitted one-use guarded CUDA entry point; no fit or optimizer."""
    from gx1.contracts.immutable_event_authority_v1 import _fsync_directory
    began = time.monotonic()
    plan, plan_ref, policy_ref, head, cap, receipts, view, bound = _preflight(
        Path(plan_path), plan_sha256, device, Path(out_dir))
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    claim = None
    try:
        _fsync_directory(root.parent)
        claim = old._write(root, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_time_query_v2_zero_step_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'fold_id': FOLD_ID, 'sample_parent_rows': PARENT_ROWS,
            'invocation': 1, 'test_data_used': False}, limit=MAX_JSON_BYTES)
        with old._bounded_worker(began):
            result = _witness(plan, cap, receipts, view, bound)
        _need(result.get('fold_id') == FOLD_ID
              and result.get('model_forwards') == result.get('backward_calls') == 1
              and result.get('optimizer_steps') == 0, 'WITNESS_COMPLETE')
        _source_bookend(plan, plan_sha256, policy_ref['sha256'], head)
        result_ref = old._write(root, 'RESULT.json', {
            'schema_version': 'gx1_market_state_time_query_v2_zero_step_result_v1',
            'status': 'COMPLETE', 'plan': plan_ref, 'claim': claim,
            'policy': policy_ref, 'source_head': head,
            'elapsed_seconds': time.monotonic() - began, **result,
        }, limit=MAX_JSON_BYTES)
        terminal = old._write(root, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_v2_zero_step_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result_ref,
            'model_forwards': 1, 'backward_calls': 1,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json'],
        }, limit=MAX_JSON_BYTES)
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as exc:
        failed = old._write(root, 'FAILED.json', {
            'schema_version': 'gx1_market_state_time_query_v2_zero_step_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - began,
            'no_relaunch_under_this_plan': True,
        }, limit=MAX_JSON_BYTES)
        old._write(root, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_v2_zero_step_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed,
            'model_forwards_upper_bound': 1, 'backward_calls_upper_bound': 1,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                 if claim else ['FAILED.json', 'TERMINAL.json']),
        }, limit=MAX_JSON_BYTES)
        raise
