"""One admitted frozen FINAL pass caching exact mature fold-1 TA representations."""
from __future__ import annotations

from contextlib import contextmanager

import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from gx1.contracts import local_random_access_campaign_v2 as campaign

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_ta_direction_full_cache_20261108_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
SCHEMA = 'gx1_market_state_ta_direction_full_cache_plan_v1'
FINAL_STAGE = 'market_state_ta_m5_equal_loss_final_cache_20261109_001'
FINAL_ROOT = Path('/home/andre2/GX1_RUNS') / FINAL_STAGE.upper()
WORKER = REPO / 'gx1/audit/cache_market_state_ta_direction_v1.py'
DISPATCH = 'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py'
CAP_RUNNER = 'scripts/gx1_capped_run.sh'
HELPER = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_DECODER_DIAGNOSTIC_20261108_001/probe.py')
FIT_ROWS, SCORE_ROWS, BATCH_SIZE, WIDTH = 29574, 17753, 16, 128
ENCODINGS, WALL_SECONDS = 2959, 3600
MAX_CACHE_BYTES = 1024**3


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_DIRECTION_CACHE_' + reason)


@contextmanager
def observe_ta_fusion_once(backbone, *, batch_rows, max_bytes, torch):
    """Capture one unchanged representation-only forward; no run admission.

    Clones stay on their original device. Gates and representation variation
    describe routing, not predictive utility. Residual coefficients and the
    joint LayerNorm must be accounted for before comparing component sizes.
    """
    _need(type(batch_rows) is int and batch_rows > 0
          and type(max_bytes) is int and max_bytes > 0, 'FUSION_BOUNDS')
    names = ('fuse', 'specialist_out', 'cross_tf_out',
             'family_tf_cooperation_out', 'entry_q_joint_norm')
    modules = {name: getattr(backbone, name, None) for name in names}
    projection = getattr(backbone, 'entry_q_joint_in', None)
    _need(isinstance(backbone, torch.nn.Module)
          and all(isinstance(m, torch.nn.Module) for m in modules.values())
          and len({id(m) for m in modules.values()} | {id(backbone)}) == len(names) + 1
          and isinstance(projection, torch.nn.Linear)
          and isinstance(modules['entry_q_joint_norm'], torch.nn.LayerNorm), 'FUSION_MODULES')
    width = projection.out_features
    families = len(backbone._specialist_names)
    mtf_families = len(backbone._mtf_specialist_names)
    timeframes = backbone.tf_gate_logits.numel()
    feature_width = backbone._mtf_feature_gate_width
    _need(width > 0 and families > 0 and mtf_families > 0 and timeframes > 0
          and type(feature_width) is int and feature_width > 0
          and projection.in_features == 4 * width
          and tuple(modules['entry_q_joint_norm'].normalized_shape) == (4 * width,),
          'FUSION_LAYOUT')
    device = projection.weight.device
    _need(device.type != 'meta' and projection.weight.dtype == torch.float32,
          'FUSION_DEVICE_DTYPE')
    shapes = {name: (batch_rows, width) for name in names[:-1]}
    shapes.update(entry_q_joint_source=(batch_rows, 4 * width),
                  entry_q_joint_hidden=(batch_rows, width),
                  specialist_gate=(batch_rows, families), tf_gate=(batch_rows, timeframes),
                  family_tf_cooperation_gate=(batch_rows, timeframes * mtf_families),
                  family_tf_feature_gate=(batch_rows, timeframes, feature_width))
    sizes = {name: 4 * math.prod(shape) for name, shape in shapes.items()}
    _need(sum(sizes.values()) <= max_bytes, 'FUSION_BYTE_LIMIT')
    captured, handles, seen = {}, [], set()
    failed = False

    def record(name, value):
        nonlocal failed
        try:
            _need(name not in seen, 'FUSION_DUPLICATE:' + name)
            seen.add(name)
            _need(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                  and value.device == device and tuple(value.shape) == shapes[name],
                  'FUSION_TENSOR:' + name)
            detached = value.detach()
            _need(bool(torch.isfinite(detached).all().item()), 'FUSION_NONFINITE:' + name)
            _need(value.numel() * value.element_size() == sizes[name], 'FUSION_BYTES:' + name)
            captured[name] = detached.clone()
        except BaseException:
            failed = True
            raise

    def output_hook(name):
        def hook(_module, _inputs, output):
            record(name, output)
        return hook

    def joint_input_hook(_module, inputs):
        nonlocal failed
        if len(inputs) != 1:
            failed = True
            _need(False, 'FUSION_JOINT_INPUT')
        record('entry_q_joint_source', inputs[0])

    output_names = ('entry_q_joint_hidden', 'specialist_gate', 'tf_gate',
                    'family_tf_cooperation_gate', 'family_tf_feature_gate')

    def backbone_hook(_module, _inputs, output):
        nonlocal failed
        if not isinstance(output, dict) or set(output) != set(output_names):
            failed = True
            _need(False, 'FUSION_OUTPUT_KEYS')
        for name in output_names:
            record(name, output[name])

    try:
        for name in names[:-1]:
            handles.append(modules[name].register_forward_hook(output_hook(name)))
        handles.append(modules['entry_q_joint_norm'].register_forward_pre_hook(joint_input_hook))
        handles.append(backbone.register_forward_hook(backbone_hook))
        yield captured
        _need(not failed and seen == set(shapes) and set(captured) == set(shapes),
              'FUSION_INCOMPLETE')
    except BaseException:
        captured.clear()
        raise
    finally:
        for handle in reversed(handles):
            handle.remove()


def _temporal_methods(backbone):
    return dict(timeframes=['m5','m15','h1','h4','d1'], families=list(backbone._mtf_specialist_names),
        temporal_lengths=[16,64,96,96,252], batch_equal_length_mtf_eval=False,
        grid_axes=['parent','timeframe','family','channel'], cross_tf_axes=['parent','timeframe','channel'],
        moment_axes=['parent','timeframe','family','moment'],
        moment_channels=['raw_rms','within_window_centered_rms'],
        moments='sqrt(mean(x64**2)); sqrt(mean((x64-mean_time(x64))**2)); cast once to float32',
        readout='FP32 mean_time(output) + output_last, then trained family/timeframe identity',
        moments_average_over=['time','channel'], native_means_and_last_dtype='float32',
        persisted_dtype='float32', hooks='parent TransformerEncoder only; no layer hooks',
        family_axis_input_reconstruction='bitexact FP32 output_mean + output_last + trained identity',
        timeframe_axis_input_reconstruction='bitexact preceding family_axis_output',
        input_interpretation='Projected features include context-conditioned gates, embeddings and positional encoding.',
        variance_interpretation='Within-window activation variation and variation between decisions are distinct; neither proves predictive utility.')


@contextmanager
def observe_ta_temporal_once(backbone, *, batch_rows, max_bytes, counts, torch):
    """Observe the exact unmerged 40-route eval; never replace inputs/outputs.

    Temporal inputs already include projection, context gates, embeddings and PE.
    FP64 window RMS is descriptive activation energy, not predictive information.
    Only parent encoders are hooked so child Transformer fastpaths stay intact.
    """
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
        _BATCH_EQUAL_LENGTH_MTF_EVAL, TF_INPUT_SCALE_NAMES)
    _need(type(batch_rows) is int and batch_rows > 0 and type(max_bytes) is int
          and max_bytes > 0, 'TEMPORAL_BOUNDS')
    _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False, 'TEMPORAL_BATCH_MODE')
    timeframes = ('m5','m15','h1','h4','d1'); lengths = (16,64,96,96,252)
    families = tuple(backbone._mtf_specialist_names); cfg = backbone.cfg
    width = cfg.d_model; identity = backbone.family_tf_token_identity
    family_modules = backbone.mtf_family_encoder
    axis_names = ('family_axis_attn','timeframe_axis_attn','cross_tf_attn')
    axes = [getattr(backbone,name) for name in axis_names]
    modules = [family_modules[name] for name in families] + axes
    _need(tuple(TF_INPUT_SCALE_NAMES) == timeframes and len(families) == len(set(families)) == 8
          and tuple(family_modules) == families and type(width) is int and width > 0
          and tuple(getattr(cfg,name+'_seq_len') for name in timeframes) == lengths
          and all(type(getattr(cfg,name+'_seq_len')) is int for name in timeframes)
          and all(isinstance(m,torch.nn.TransformerEncoder) for m in modules)
          and len({id(m) for m in modules}) == 11 and not backbone.training
          and not any(m.training for parent in modules for m in parent.modules()), 'TEMPORAL_LAYOUT')
    _need(isinstance(identity,torch.Tensor) and identity.dtype == torch.float32
          and identity.device.type != 'meta' and tuple(identity.shape) == (1,5,8,width)
          and bool(torch.isfinite(identity.detach()).all().item()), 'TEMPORAL_IDENTITY')
    device = identity.device
    shapes = {name:(batch_rows,5,8,width) for name in (
        'temporal_input_mean','temporal_input_last','temporal_output_mean','temporal_output_last',
        'family_axis_output','timeframe_axis_output')}
    shapes.update({name:(batch_rows,5,8,2) for name in ('temporal_input_moments','temporal_output_moments')})
    shapes.update({name:(batch_rows,5,width) for name in ('cross_tf_input','cross_tf_output')})
    _need(4*sum(math.prod(shape) for shape in shapes.values()) <= max_bytes, 'TEMPORAL_BYTE_LIMIT')
    limits = dict(temporal_family_encoder_calls=1280,temporal_family_axis_calls=32,
        temporal_timeframe_axis_calls=32,temporal_cross_tf_calls=32,temporal_observed_forwards=32)
    _need(all(type(counts.get(k)) is int and 0 <= counts[k] < limit for k,limit in limits.items()),
          'TEMPORAL_COUNTERS')
    captured = {name:torch.empty(shape,dtype=torch.float32,device=device) for name,shape in shapes.items()}
    handles = []; completed = 0; pending = None; stage = 0; failed = False

    def count(name):
        counts[name] += 1
        _need(counts[name] <= limits[name], 'TEMPORAL_CALL_LIMIT:'+name)

    def tensor(value, shape, name):
        _need(isinstance(value,torch.Tensor) and value.dtype == torch.float32
              and value.device == device and tuple(value.shape) == shape, 'TEMPORAL_TENSOR:'+name)
        value = value.detach()
        _need(bool(torch.isfinite(value).all().item()), 'TEMPORAL_NONFINITE:'+name)
        return value

    def input_tensor(args, kwargs, shape, name):
        _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False, 'TEMPORAL_BATCH_MODE')
        _need(not torch.is_grad_enabled() and not backbone.training, 'TEMPORAL_EVAL_ONLY')
        _need(len(args) == 1 and set(kwargs) <= {'src_key_padding_mask'}
              and all(value is None for value in kwargs.values()), 'TEMPORAL_CALL_SIGNATURE:'+name)
        return tensor(args[0],shape,name)

    def exact(actual, expected, name):
        _need(torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32)),
              'TEMPORAL_BIT_PARITY:'+name)

    def summarize(value, prefix, tf, family):
        # The native readout also uses FP32 mean and last; FP64 here is only for RMS.
        captured[prefix+'_mean'][:,tf,family].copy_(value.mean(dim=1))
        captured[prefix+'_last'][:,tf,family].copy_(value[:,-1,:])
        wide = value.to(torch.float64)
        rms = wide.square().mean(dim=(1,2)).sqrt()
        centered = (wide-wide.mean(dim=1,keepdim=True)).square().mean(dim=(1,2)).sqrt()
        moments = torch.stack((rms,centered),dim=1).to(torch.float32)
        _need(bool(torch.isfinite(moments).all().item()), 'TEMPORAL_MOMENTS_NONFINITE')
        captured[prefix+'_moments'][:,tf,family].copy_(moments)

    def guarded(callback):
        def hook(*args):
            nonlocal failed
            try:
                callback(*args)
            except BaseException:
                failed = True
                raise
        return hook

    def family_pre(family):
        def observe(_module, args, kwargs):
            nonlocal pending
            count('temporal_family_encoder_calls')
            _need(stage == 0 and pending is None and completed < 40 and completed % 8 == family,
                  'TEMPORAL_ROUTE_ORDER')
            tf = completed // 8
            value = input_tensor(args,kwargs,(batch_rows,lengths[tf],width),'family_input')
            summarize(value,'temporal_input',tf,family)
            pending = (tf,family)
        return guarded(observe)

    def family_post(family):
        def observe(_module, _args, _kwargs, output):
            nonlocal completed, pending
            _need(pending == (completed//8,family) and completed % 8 == family, 'TEMPORAL_ROUTE_PAIR')
            tf = completed // 8
            summarize(tensor(output,(batch_rows,lengths[tf],width),'family_output'),
                      'temporal_output',tf,family)
            completed += 1; pending = None
        return guarded(observe)

    axis_shapes = ((batch_rows*8,5,width),(batch_rows*5,8,width),(batch_rows,5,width))
    axis_counts = ('temporal_family_axis_calls','temporal_timeframe_axis_calls','temporal_cross_tf_calls')
    def axis_pre(index):
        def observe(_module, args, kwargs):
            nonlocal stage
            count(axis_counts[index])
            _need(completed == 40 and pending is None and stage == 2*index, 'TEMPORAL_AXIS_ORDER')
            value = input_tensor(args,kwargs,axis_shapes[index],axis_names[index]+'_input')
            if index == 0:
                grid = captured['temporal_output_mean'] + captured['temporal_output_last'] + identity.detach()
                expected = grid.permute(0,2,1,3).reshape(axis_shapes[index])
                exact(value,expected,'family_axis_input')
            elif index == 1:
                exact(value,captured['family_axis_output'].reshape(axis_shapes[index]),'timeframe_axis_input')
            else:
                captured['cross_tf_input'].copy_(value)
            stage += 1
        return guarded(observe)

    def axis_post(index):
        def observe(_module, _args, _kwargs, output):
            nonlocal stage
            _need(stage == 2*index+1, 'TEMPORAL_AXIS_PAIR')
            value = tensor(output,axis_shapes[index],axis_names[index]+'_output')
            if index == 0:
                captured['family_axis_output'].copy_(value.reshape(batch_rows,8,5,width).permute(0,2,1,3))
            elif index == 1:
                captured['timeframe_axis_output'].copy_(value.reshape(batch_rows,5,8,width))
            else:
                captured['cross_tf_output'].copy_(value)
            stage += 1
        return guarded(observe)

    try:
        for index,name in enumerate(families):
            handles.append(family_modules[name].register_forward_pre_hook(family_pre(index),with_kwargs=True))
            handles.append(family_modules[name].register_forward_hook(family_post(index),with_kwargs=True))
        for index,module in enumerate(axes):
            handles.append(module.register_forward_pre_hook(axis_pre(index),with_kwargs=True))
            handles.append(module.register_forward_hook(axis_post(index),with_kwargs=True))
        yield captured
        _need(not failed and completed == 40 and pending is None and stage == 6, 'TEMPORAL_INCOMPLETE')
        _need(all(bool(torch.isfinite(value).all().item()) for value in captured.values()), 'TEMPORAL_CAPTURE_NONFINITE')
        count('temporal_observed_forwards')
    except BaseException:
        captured.clear()
        raise
    finally:
        for handle in reversed(handles):
            handle.remove()


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def _read(ref):
    return campaign.read_bound_json(Path(ref['path']), ref['sha256'])


def _ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': campaign.file_sha256(path)}


def _publish(name, value, *, root=ROOT):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    _publish_bytes(root / name, (json.dumps(value, sort_keys=True, indent=2,
                                          allow_nan=False) + '\n').encode())
    return _ref(root / name)


def _helper(ref):
    """The reused CPU helper hides CUDA on import; restore the inherited setting."""
    import torch
    _need(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED_BEFORE_HELPER')
    campaign.require_binding(ref, label='frozen strict-load helper', verify_file=True)
    _need(ref['path'] == str(HELPER), 'HELPER_PATH')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        spec = importlib.util.spec_from_file_location('ta_direction_cache_strict_load', HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _need(not torch.cuda.is_initialized(), 'HELPER_INITIALIZED_CUDA')
    finally:
        if inherited is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = inherited
    _need(os.environ.get('CUDA_VISIBLE_DEVICES') == inherited, 'CUDA_VISIBILITY_RESTORE')
    return module


def _cache_scope(plan_path, out_dir):
    for stage, root, dedicated in ((STAGE, ROOT, False), (FINAL_STAGE, FINAL_ROOT, True)):
        if Path(plan_path) == root / 'PLAN.json' and Path(out_dir) == root:
            return stage, root, dedicated
    _need(False, 'EXACT_CACHE_PATHS')


def _inspect(plan_path, plan_sha256, out_dir, *, execute):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    stage, root, dedicated = _cache_scope(plan_path, out_dir)
    plan_ref = {'path': str(Path(plan_path)), 'sha256': plan_sha256}
    plan = _read(plan_ref)
    keys = {'schema_version', 'stage', 'source_repo', 'source_base_commit',
            'historical_source_commit', 'worker', 'dispatch', 'cap_runner', 'helper',
            'recipe', 'result', 'fit_rows', 'score_rows', 'batch_size', 'decoder_width',
            'include_sampled_S', 'max_backbone_encodings', 'device', 'wall_seconds',
            'training_enabled', 'test_data_used'}
    if dedicated:
        keys.add('fullfit_review')
    _need(set(plan) == keys and plan['schema_version'] == SCHEMA and plan['stage'] == stage
          and plan['source_repo'] == str(REPO) and Path(plan_path) == root / 'PLAN.json'
          and Path(out_dir) == root and plan['fit_rows'] == FIT_ROWS
          and plan['score_rows'] == SCORE_ROWS and plan['batch_size'] == BATCH_SIZE
          and plan['decoder_width'] == WIDTH and plan['include_sampled_S'] is True
          and plan['max_backbone_encodings'] == ENCODINGS and plan['device'] == 'cuda'
          and plan['wall_seconds'] == WALL_SECONDS and plan['training_enabled'] is False
          and plan['test_data_used'] is False, 'EXACT_PLAN')
    for key, path in (('worker', WORKER), ('dispatch', REPO / DISPATCH),
                      ('cap_runner', REPO / CAP_RUNNER), ('helper', HELPER)):
        checked = campaign.require_binding(plan[key], label=key, verify_file=True)
        _need(checked['path'] == str(path), 'SOURCE_PATH:' + key)
    for key in ('source_base_commit', 'historical_source_commit'):
        _need(campaign._COMMIT.fullmatch(plan[key]) is not None, 'COMMIT:' + key)
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor',
                    plan['source_base_commit'], 'HEAD'], check=True)
    recipe, result = _read(plan['recipe']), _read(plan['result'])
    recipe_path = scope.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH if dedicated else scope.TA_RECOVERY_RECIPE_PATH
    run_scope = scope.TA_M5_EQUAL_LOSS_FIT_SCOPE if dedicated else scope.TA_RECOVERY_SCOPE
    runtime = scope.TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT if dedicated else scope.TA_RECOVERY_RUNTIME_ROOT
    _need(plan['recipe']['path'] == str(recipe_path)
          and recipe['recipe_sha256'] == campaign.canonical_sha256({
              k: v for k, v in recipe.items() if k != 'recipe_sha256'})
          and recipe['run_scope'] == run_scope and recipe['fold'] == scope.FOLD
          and recipe['numerical_recipe'] == scope.ta_sf_numerical_recipe(
              dedicated_m5_side=dedicated, m5_equal_observation=dedicated)
          and result['recipe'] == plan['recipe'] and result['source_commit'] == plan['historical_source_commit']
          and plan['result']['path'] == str(runtime / 'prefix-fit/window-0001/RESULT.json')
          and result['outcome'] == result['phase_state'] == 'COMPLETE'
          and result['source_unchanged'] is True and result['test_data_used'] is False
          and result['parent_cursor'] == FIT_ROWS and result['optimizer_updates'] == 1849
          and result['skipped_batches'] == 0, 'HISTORICAL_RECIPE_RESULT')
    if dedicated:
        review = _read(plan['fullfit_review'])
        _need(plan['fullfit_review']['path'] == str(scope.TA_M5_EQUAL_LOSS_FIT_ROOT / 'REVIEW.json')
              and review['status'] == 'COMPLETE_CONSUMED_DIRECTION_DETERIORATED'
              and review['native_source_commit'] == plan['historical_source_commit']
              and review['result'] == plan['result'] and review['final_state'] == result['full_state']
              and review['final_train_measurement'] == result['final_train_measurement']
              and review['later_train_measurement'] == result['oof_measurement']
              and review['matched_train_observations_verified'] is True
              and review['score_means_recalculated'] is True
              and review['source_unchanged'] is True and review['test_data_used'] is False
              and result['dedicated_m5_side'] is True and result['m5_equal_observation'] is True,
              'QUALIFIED_EQUAL_LOSS_FINAL')
    owners = recipe['source_owners']
    _need(set(owners) == scope.required_source_paths(ta_sf=True) and len(owners) == 85,
          'HISTORICAL_85_OWNER_SET')
    current = {}
    historical_git = installed = 0
    replacements = {CAP_RUNNER: plan['cap_runner']}
    if not dedicated:
        replacements[DISPATCH] = plan['dispatch']
    for name, ref in owners.items():
        checked = campaign.require_binding(ref, label='historical owner', verify_file=False)
        _need(checked['path'] == str(REPO / name), 'HISTORICAL_OWNER_PATH')
        if name.startswith('.venv/'):
            campaign.require_binding(ref, label='unchanged installed runtime', verify_file=True)
            installed += 1
        else:
            raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                           plan['historical_source_commit'] + ':' + name])
            _need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'HISTORICAL_GIT:' + name)
            historical_git += 1
        expected = replacements.get(name, ref)
        campaign.require_binding(expected, label='current owner', verify_file=True)
        if name in replacements:
            _need(expected['sha256'] != ref['sha256'], 'DECLARED_ROUTE_CHANGE:' + name)
        current[name] = expected['sha256']
    _need(historical_git == 77 and installed == 8, 'HISTORICAL_OWNER_COUNTS')
    for name, ref in recipe['input_refs'].items():
        campaign.require_binding(ref, label='reused input ' + name, verify_file=True)
    receipts = zero.load_ta_input_receipts(current_source_owners=current)
    view = zero._bind_ta_view(receipts, dedicated_m5_side=dedicated)
    if not dedicated:
        _need(view == _read(recipe['input_refs']['ta_candidate_view']), 'LEGACY_CANDIDATE_VIEW')
    _need(view['contract_sha256'] == recipe['expected_view_sha256'], 'ACTUAL_VIEW')
    scope.require_checkpoint_view(result['feature_view_identity'], recipe=recipe, ta_sf=True)
    baseline = {partition: _read(result[key]) for partition, key in (
        ('fit', 'final_train_measurement'), ('score', 'oof_measurement'))}
    for partition, count in (('fit', FIT_ROWS), ('score', SCORE_ROWS)):
        value = baseline[partition]
        _need(value['evaluated_checkpoint'] == result['full_state']
              and value['optimizer_updates'] == 1849 and value['test_data_used'] is False
              and value['measurement']['rows'] == count
              and value['measurement']['partition'] == partition
              and value['measurement']['feature_view_sha256'] == view['contract_sha256'],
              'BASELINE_PARTITION:' + partition)
    _need(baseline['fit']['evaluated_model_state_sha256']
          == baseline['score']['evaluated_model_state_sha256'], 'BASELINE_SAME_FINAL')
    campaign.require_binding(result['full_state'], label='frozen FINAL', verify_file=True)
    helper = _helper(plan['helper'])
    head, policy_ref = _git('rev-parse', 'HEAD'), None
    if execute:
        _need(_git('branch', '--show-current') == 'work/gx1-current', 'CANONICAL_BRANCH')
        campaign.require_clean_source({'source_repo': str(REPO), 'source_commit': head})
        policy_ref = _ref(REPO / 'NEXT_RUN_POLICY.json')
        policy = _read(policy_ref)
        expected = {'status': 'ADMITTED', 'plan': plan_ref, 'max_invocations': 1,
                    'claimed_invocations': 0, 'completed_invocations': 0, 'relaunch_allowed': False,
                    'execution_authorized': True, 'data_reads_authorized': True,
                    'checkpoint_reads_authorized': True, 'backbone_encodings_authorized': ENCODINGS,
                    'backward_calls_authorized': 0, 'optimizer_steps_authorized': 0,
                    'training_authorized': False}
        gate = policy.get(stage, {})
        _need(all(gate.get(k) == v for k, v in expected.items())
              and policy.get('training_enabled') is False
              and policy.get('full_epoch_training_allowed') is False, 'ADMITTED_POLICY')
        _need(not (root / 'CLAIM.json').exists() and not (root / 'TERMINAL.json').exists(), 'CONSUMED')
    return {'root': root, 'changed_owners': list(replacements),
            'plan': plan, 'plan_ref': plan_ref, 'recipe': recipe, 'result': result,
            'receipts': receipts, 'view': view, 'baseline': baseline, 'helper': helper,
            'head': head, 'policy_ref': policy_ref, 'current_sources': current,
            'historical_git_owners': historical_git, 'installed_runtime_owners': installed}


def preflight_read_only(plan_path, plan_sha256, out_dir):
    if Path(plan_path) == MIXING_GRADIENT_ROOT/'PLAN.json' or Path(out_dir) == MIXING_GRADIENT_ROOT:
        checked = _inspect_fusion_observation(plan_path, plan_sha256, out_dir, execute=False, gradient=True)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_ARRAY_DESERIALIZATION',plan=checked['plan_ref'],
            view_sha256=checked['view']['contract_sha256'],model_constructions=0,backbone_encodings=0,
            archive_decodes=0,autograd_grad_calls=0,current_source_owners=len(checked['current_sources']),
            changed_owners=[CAP_RUNNER],scalar_fits=0,test_data_used=False)
    if Path(plan_path) == TEMPORAL_ROOT/'PLAN.json' or Path(out_dir) == TEMPORAL_ROOT:
        checked = _inspect_fusion_observation(plan_path, plan_sha256, out_dir, execute=False, temporal=True)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_ARRAY_DESERIALIZATION',plan=checked['plan_ref'],
            view_sha256=checked['view']['contract_sha256'],model_constructions=0,backbone_encodings=0,
            archive_decodes=0,current_source_owners=len(checked['current_sources']),changed_owners=[CAP_RUNNER],
            scalar_fits=0,test_data_used=False)
    if Path(plan_path) == MTF_RETAINED_ROOT/'PLAN.json' or Path(out_dir) == MTF_RETAINED_ROOT:
        checked = _inspect_fusion_observation(plan_path, plan_sha256, out_dir, execute=False, retained=True)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_ARRAY_DESERIALIZATION',
            plan=checked['plan_ref'], view_sha256=checked['view']['contract_sha256'],
            candidate_model_state_sha256=MTF_RETAINED_BOUNDS['candidate_model_state_sha256'],
            model_constructions=0, backbone_encodings=0, archive_decodes=0, joint_calls=0,
            current_source_owners=len(checked['current_sources']), changed_owners=[CAP_RUNNER],
            scalar_fits=0, predictor_fits=0, test_data_used=False)
    if Path(plan_path) == FUSION_INITIAL_ROOT / 'PLAN.json' or Path(out_dir) == FUSION_INITIAL_ROOT:
        checked = _inspect_fusion_observation(plan_path, plan_sha256, out_dir, execute=False, initial=True)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_ARRAY_DESERIALIZATION',
            plan=checked['plan_ref'], view_sha256=checked['view']['contract_sha256'],
            candidate_model_state_sha256=FUSION_INITIAL_BOUNDS['candidate_model_state_sha256'],
            model_constructions=0, backbone_encodings=0, archive_decodes=0,
            current_source_owners=len(checked['current_sources']), changed_owners=[CAP_RUNNER],
            scalar_fits=0, test_data_used=False)
    if Path(plan_path) == FUSION_ROOT / 'PLAN.json' or Path(out_dir) == FUSION_ROOT:
        checked = _inspect_fusion_observation(plan_path, plan_sha256, out_dir, execute=False)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_ARRAY_DESERIALIZATION',
            plan=checked['plan_ref'], view_sha256=checked['view']['contract_sha256'],
            candidate_model_state_sha256=FUSION_BOUNDS['candidate_model_state_sha256'],
            model_constructions=0, backbone_encodings=0, archive_decodes=0,
            current_source_owners=len(checked['current_sources']), changed_owners=[CAP_RUNNER],
            scalar_fits=0, test_data_used=False)
    if Path(plan_path) == SHRINK_ROOT / 'PLAN.json' or Path(out_dir) == SHRINK_ROOT:
        checked = _inspect_shrink_score(plan_path, plan_sha256, out_dir, execute=False)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_CHECKPOINT_DESERIALIZATION',
            plan=checked['plan_ref'], view_sha256=checked['view']['contract_sha256'],
            source_model_state_sha256=REVERSED_MODEL, model_constructions=0, backbone_encodings=0,
            current_source_owners=len(checked['current_sources']), changed_owners=[CAP_RUNNER],
            scalar_fits=0, test_data_used=False)
    if Path(plan_path) == SCORE_ROOT / 'PLAN.json' or Path(out_dir) == SCORE_ROOT:
        checked = _inspect_fixed_score(plan_path, plan_sha256, out_dir, execute=False)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_CHECKPOINT_DESERIALIZATION',
            plan=checked['plan_ref'], view_sha256=checked['view']['contract_sha256'],
            source_model_state_sha256=REVERSED_MODEL, model_constructions=0, backbone_encodings=0,
            current_source_owners=len(checked['current_sources']), changed_owners=[CAP_RUNNER],
            test_data_used=False)
    if Path(plan_path) == REVERSED_ROOT / 'PLAN.json' or Path(out_dir) == REVERSED_ROOT:
        checked = _inspect_reversed(plan_path, plan_sha256, out_dir, execute=False)
        return dict(status='READY_NO_CLAIM_NO_MODEL_NO_CHECKPOINT_DESERIALIZATION',
            plan=checked['plan_ref'], view_sha256=checked['view']['contract_sha256'],
            model_state_sha256=REVERSED_MODEL, model_constructions=0, backbone_encodings=0,
            current_source_owners=len(checked['current_sources']), changed_owners=[CAP_RUNNER],
            test_data_used=False)
    checked = _inspect(plan_path, plan_sha256, out_dir, execute=False)
    return {'status': 'READY_NO_CLAIM_NO_MODEL_NO_CHECKPOINT_DESERIALIZATION',
            'plan': checked['plan_ref'], 'view_sha256': checked['view']['contract_sha256'],
            'historical_git_owners': checked['historical_git_owners'],
            'installed_runtime_owners': checked['installed_runtime_owners'],
            'unchanged_current_recipe_owners': len(checked['current_sources']) - len(checked['changed_owners']),
            'declared_changed_owners': checked['changed_owners'],
            'model_constructions': 0, 'backbone_encodings': 0, 'test_data_used': False}


def _side_parity(scores, baseline):
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    report = {}
    for task in ('state', 'future'):
        report[task] = {}
        for tf in TIMEFRAMES:
            got, expected = scores[task][tf]['conditional_side'], baseline['scores']['model'][task][tf]['conditional_side']
            _need(got['count'] == expected['count'] and got['count'] > 0, 'SIDE_COUNT:' + task + ':' + tf)
            metrics = {'count': got['count']}
            for name in ('brier_sum', 'log_sum'):
                actual_mean = got[name] / got['count']
                reference_mean = expected[name] / expected['count']
                delta = abs(actual_mean - reference_mean)
                _need(delta <= 1e-6, 'SIDE_PARITY:' + task + ':' + tf + ':' + name)
                metrics[name.replace('_sum', '')] = {
                    'actual': actual_mean, 'baseline': reference_mean, 'absolute_difference': delta}
            report[task][tf] = metrics
    return report


def _save_cache(partition, arrays, rows, *, partial=False, root=ROOT):
    import numpy as np
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    name = partition.upper() + ('_CACHE_PARTIAL.npz' if partial else '_CACHE.npz')
    stream = io.BytesIO()
    np.savez(stream, **{key: value[:rows] for key, value in arrays.items()})
    payload = stream.getvalue()
    _need(0 < len(payload) <= MAX_CACHE_BYTES, 'CACHE_BYTES')
    _publish_bytes(root / name, payload)
    return {'artifact': _ref(root / name), 'rows': rows, 'partial': partial,
            'arrays': {key: {'shape': [rows, *value.shape[1:]], 'dtype': str(value.dtype)}
                       for key, value in arrays.items()}}


def run(plan_path, plan_sha256, out_dir, *, device):
    if Path(plan_path) == MIXING_GRADIENT_ROOT/'PLAN.json' or Path(out_dir) == MIXING_GRADIENT_ROOT:
        return _run_mixing_gradient_observation(plan_path, plan_sha256, out_dir, device=device)
    if Path(plan_path) == TEMPORAL_ROOT/'PLAN.json' or Path(out_dir) == TEMPORAL_ROOT:
        return _run_fusion_observation(plan_path, plan_sha256, out_dir, device=device, temporal=True)
    if Path(plan_path) == MTF_RETAINED_ROOT/'PLAN.json' or Path(out_dir) == MTF_RETAINED_ROOT:
        return _run_fusion_observation(plan_path, plan_sha256, out_dir, device=device, retained=True)
    if Path(plan_path) == FUSION_INITIAL_ROOT / 'PLAN.json' or Path(out_dir) == FUSION_INITIAL_ROOT:
        return _run_fusion_observation(plan_path, plan_sha256, out_dir, device=device, initial=True)
    if Path(plan_path) == FUSION_ROOT / 'PLAN.json' or Path(out_dir) == FUSION_ROOT:
        return _run_fusion_observation(plan_path, plan_sha256, out_dir, device=device)
    if Path(plan_path) == SHRINK_ROOT / 'PLAN.json' or Path(out_dir) == SHRINK_ROOT:
        return _run_shrink_score(plan_path, plan_sha256, out_dir, device=device)
    if Path(plan_path) == SCORE_ROOT / 'PLAN.json' or Path(out_dir) == SCORE_ROOT:
        return _run_fixed_score(plan_path, plan_sha256, out_dir, device=device)
    if Path(plan_path) == REVERSED_ROOT / 'PLAN.json' or Path(out_dir) == REVERSED_ROOT:
        return _run_reversed(plan_path, plan_sha256, out_dir, device=device)
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_producer_execution, require_capped_lock_ancestry)
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
    from gx1.contracts.market_state_time_query_scorer_v2 import proper_score_sums
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    started = time.monotonic()
    _need(device == 'cuda', 'GPU_ONLY')
    cap = require_guarded_cuda_producer_execution()
    cap['lock'] = require_capped_lock_ancestry()
    _need(cap['cgroup_path'] == '/sys/fs/cgroup' + cap['lock']['scope'], 'LOCK_CGROUP_SCOPE')
    _need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
          and cap['swap'] == 512 * 1024**2, 'EXACT_12G_512M')
    for key, value in (('MAX_POWER_LIMIT_W', '250'), ('MAX_POWER_DRAW_W', '170'),
                       ('MAX_CORE_TEMP_C', '65'), ('MAX_MEMORY_TEMP_C', '80')):
        _need(os.environ.get('GX1_TRAINER_GPU_' + key) == value, 'HARDWARE_GUARD:' + key)
    c = _inspect(plan_path, plan_sha256, out_dir, execute=True)
    recipe, result, receipts, view = (c[k] for k in ('recipe', 'result', 'receipts', 'view'))
    root = c['root']
    def publish(name, value):
        return _publish(name, value, root=root)
    _need(not torch.cuda.is_initialized(), 'CUDA_INIT_ORDER')
    _need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    claim = publish('CLAIM.json', {'plan': c['plan_ref'], 'policy': c['policy_ref'],
        'execution_head': c['head'], 'cap': cap, 'training_enabled': False, 'test_data_used': False})
    provider = prepared = model = None
    arrays, cache_refs, partitions, timings = {}, {}, {}, {}
    active_partition, consumed = None, 0
    counts = {'backbone_encodings': 0, 'model_constructions': 0, 'optimizer_steps': 0,
              'backward_calls': 0, 'resume_state_load_calls': 0, 'resume_state_load_returns': 0}
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('TA direction full cache 3600-second deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1., WALL_SECONDS - 60 - (time.monotonic() - started)))
    def bookend():
        campaign.require_clean_source({'source_repo': str(REPO), 'source_commit': c['head']})
        refs = [c['plan_ref'], c['policy_ref'], *[c['plan'][k] for k in (
            'worker', 'dispatch', 'cap_runner', 'helper', 'recipe', 'result')],
            result['full_state'], result['final_train_measurement'], result['oof_measurement'],
            *recipe['input_refs'].values()]
        if 'fullfit_review' in c['plan']:
            refs.append(c['plan']['fullfit_review'])
        for ref in refs:
            campaign.require_binding(ref, label='cache bookend', verify_file=True)
        for name, digest in c['current_sources'].items():
            _need(campaign.file_sha256(REPO / name) == digest, 'SOURCE_BOOKEND:' + name)
    try:
        tick = time.monotonic()
        numerical = recipe['numerical_recipe']
        trainer._set_deterministic(numerical['seed'], torch.device('cuda'), numerical['precision_policy'])
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _need(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'PRODUCER_NUMERICAL_THREADS')
        provider, provider_plan, prepared, fit_rows, fit_issues = runner._prepared_fold1(receipts, view, np=np)
        endpoint, future, sampled = (load_bound_five_endpoint_targets(),
            load_bound_future_displacement(), load_bound_time_query_targets())
        axes = endpoint.mature_oof_axes(fold_id=1)
        _need(np.array_equal(axes['fit_parent_rows'], fit_rows)
              and np.array_equal(axes['fit_decision_ns'], fit_issues), 'EXACT_FIT_AXIS')
        score_slice = slice(recipe['fold']['oof_start_row'], recipe['fold']['oof_end_row'])
        _need(np.array_equal(axes['score_parent_rows'], prepared['axes']['parent_rows'][score_slice])
              and np.array_equal(axes['score_decision_ns'], prepared['axes']['decision_ns'][score_slice]),
              'EXACT_LATER_TRAIN_AXIS')
        state_model = views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(state_model)
        counts['model_constructions'] = 1
        loaded = c['helper'].load_frozen(result['full_state'], 'final', model,
            recipe=recipe, result=result, metadata={'measurement': c['baseline']['fit']},
            order=torch.from_numpy(fit_rows.copy()), counters=counts)
        model.to(torch.device('cuda'))
        model.eval()
        before = canonical_model_state_sha256(model.state_dict())
        _need(before == loaded['model_sha256'] and not any(p.requires_grad for p in model.parameters()),
              'FROZEN_GPU_MODEL')
        timings['setup_and_load_seconds'] = time.monotonic() - tick
        for partition, rows, issues, expected_rows in (
                ('fit', fit_rows, fit_issues, FIT_ROWS),
                ('score', axes['score_parent_rows'], axes['score_decision_ns'], SCORE_ROWS)):
            tick = time.monotonic()
            active_partition, consumed, arrays = partition, 0, {}
            _need(len(rows) == len(issues) == expected_rows, 'PARTITION_ROWS')
            scores = {}
            def put(key, value, left, right):
                value = np.asarray(value)
                _need(value.shape[0] == right - left and value.dtype.kind in 'biuf',
                      'CACHE_ARRAY:' + key)
                if value.dtype.kind == 'f':
                    _need(np.isfinite(value).all(), 'NONFINITE_CACHE:' + key)
                if key not in arrays:
                    arrays[key] = np.empty((expected_rows, *value.shape[1:]), dtype=value.dtype)
                arrays[key][left:right] = value
            for batch_index, left in enumerate(range(0, expected_rows, BATCH_SIZE)):
                right = min(left + BATCH_SIZE, expected_rows)
                raw, parents, clocks = runner._feature_batch(prepared, view, receipts,
                    rows, issues, left, right, np=np)
                features = {key: torch.from_numpy(value).to('cuda') for key, value in raw.items()}
                native_np = np.broadcast_to(np.asarray(NATIVE_K, np.int64), (len(parents), 5)).copy()
                native = torch.from_numpy(native_np).to('cuda')
                captures = {}
                hooks = [model.state_model.query_head.register_forward_pre_hook(
                    lambda _, args: captures.update(state=args[0].detach())),
                    model.future_head.register_forward_pre_hook(
                    lambda _, args: captures.update(future=args[0].detach()))]
                try:
                    with torch.inference_mode():
                        hidden = model.encode_features(**features)
                        counts['backbone_encodings'] += 1
                        sr = model.state_model.decode_raw_query_grid(hidden, native[:, None])[:, 0]
                        fr = model.decode_native_F_raw(hidden)
                        representations = [captures['state'][:, 0].cpu().numpy().copy(),
                                           captures['future'].cpu().numpy().copy()]
                        logits = [sr.cpu().numpy().copy(), fr.cpu().numpy().copy()]
                        if partition == 'fit':
                            qn = sampled_query_k(clocks, pass_index=0, seed=numerical['seed'])
                            ss = model.state_model.decode_raw_query_grid(
                                hidden, torch.from_numpy(qn).to('cuda')[:, None])[:, 0]
                            train_rep = captures['state'][:, 0].cpu().numpy().copy()
                            train_raw = ss.cpu().numpy().copy()
                finally:
                    for hook in hooks: hook.remove()
                truths = [endpoint.gather(parents, clocks, fold_id=1, partition=partition),
                          future.gather(parents, clocks, fold_id=1, partition=partition)]
                put('parents', parents, left, right); put('issues', clocks, left, right)
                put('query_native', native_np, left, right)
                for index, task in enumerate(('state', 'future')):
                    _need(representations[index].shape == (len(parents), 5, WIDTH), 'REPRESENTATION_WIDTH')
                    truth = truths[index]
                    _need(np.array_equal(truth['parent_rows'], parents)
                          and np.array_equal(truth['issue_ns'], clocks), 'NATIVE_TARGET_AXIS')
                    for field, value in [('representation', representations[index]),
                            ('activity', logits[index][..., 0]), ('original_raw', logits[index]),
                            *[(field, truth[field]) for field in ('allowed', 'eligible', 'mature')]]:
                        put(f'native_{index}_{field}', value, left, right)
                    batch_scores = proper_score_sums(logits[index], truth['allowed'],
                        truth['eligible'], truth['mature'], native_np)['per_tf']
                    if task not in scores:
                        scores[task] = batch_scores
                    else:
                        for tf in TIMEFRAMES:
                            for metric, stats in batch_scores[tf].items():
                                for key, value in stats.items():
                                    scores[task][tf][metric][key] += value
                if partition == 'fit':
                    truth = sampled.gather_fit_sampled(parents, clocks, fold_id=1,
                                                       pass_index=0, seed=numerical['seed'])
                    _need(np.array_equal(truth['query_k'], qn)
                          and np.array_equal(truth['parent_rows'], parents)
                          and np.array_equal(truth['issue_ns'], clocks)
                          and train_rep.shape == (len(parents), 5, WIDTH)
                          and np.array_equal(train_raw[:, 0], logits[0][:, 0]), 'SAMPLED_S_PARITY')
                    put('query_train', qn, left, right)
                    for field, value in [('representation', train_rep), ('activity', train_raw[..., 0]),
                            ('original_raw', train_raw),
                            *[(field, truth[field]) for field in ('allowed', 'eligible', 'mature')]]:
                        put('train_0_' + field, value, left, right)
                consumed = right
                if (batch_index + 1) % 256 == 0:
                    print(json.dumps({'status': 'PROGRESS', 'partition': partition,
                        'completed_rows': consumed, 'completed_batches': batch_index + 1,
                        'backbone_encodings': counts['backbone_encodings'],
                        'elapsed_seconds': time.monotonic() - started}), flush=True)
                del raw, features, hidden, sr, fr
            cache_refs[partition] = _save_cache(partition, arrays, consumed, root=root)
            # Preserve the produced bytes even when their qualification fails.
            parity = _side_parity(scores, c['baseline'][partition]['measurement'])
            partitions[partition] = {'rows': consumed, 'batches': (consumed + 15) // 16,
                'conditional_side_parity': parity, 'scores': scores}
            timings[partition + '_seconds'] = time.monotonic() - tick
            arrays = {}
        _need(counts['backbone_encodings'] == ENCODINGS
              and canonical_model_state_sha256(model.state_dict()) == before
              and all(p.grad is None for p in model.parameters()), 'COUNTS_AND_MODEL_UNCHANGED')
        base._release_pages(provider, prepared); prepared = None
        provider.validate_metadata(provider_plan)
        bookend()
        result_ref = publish('RESULT.json', {'status': 'COMPLETE', 'plan': c['plan_ref'],
            'claim': claim, 'policy': c['policy_ref'], 'execution_head': c['head'],
            'historical_source_commit': c['plan']['historical_source_commit'],
            'checkpoint': loaded, 'view_sha256': view['contract_sha256'],
            'cache_partitions': cache_refs, 'partitions': partitions, 'counts': counts,
            'timings': {**timings, 'elapsed_seconds': time.monotonic() - started},
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'model_state_unchanged': True, 'source_unchanged': True,
            'sampled_S_pass_index': 0, 'sampled_F_reuses_native_1': True,
            'test_data_used': False, 'predictive_value_claimed': False, 'promotion_authorized': False})
        terminal = publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': c['plan_ref'], 'result': result_ref, 'cache_partitions': cache_refs,
            'source_unchanged': True, 'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as exc:
        preservation_error = None
        if arrays and consumed and active_partition not in cache_refs:
            try:
                cache_refs[active_partition] = _save_cache(active_partition, arrays, consumed, partial=True, root=root)
            except Exception as secondary:
                preservation_error = type(secondary).__name__ + ':' + str(secondary)
        failure = publish('FAILED.json', {'status': 'FAILED_CONSUMED', 'plan': c['plan_ref'],
            'claim': claim, 'error': type(exc).__name__ + ':' + str(exc),
            'cache_partitions': cache_refs, 'preservation_error': preservation_error,
            'counts': counts, 'elapsed_seconds': time.monotonic() - started, 'test_data_used': False})
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': c['plan_ref'], 'failure': failure, 'cache_partitions': cache_refs,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_alarm)
        if prepared is not None:
            base._release_pages(provider, prepared)

# Frozen REVERSED weights: a new, exact TRAIN-only producer; old scopes stay sealed.
REVERSED_STAGE = 'market_state_ta_m5_reversed_repr_cache_20261110_001'
REVERSED_ROOT = Path('/home/andre2/GX1_RUNS') / REVERSED_STAGE.upper()
REVERSED_SCHEMA = 'gx1_market_state_ta_m5_reversed_repr_cache_plan_v1'
REVERSED_SOURCE = '2f493a251479484b5876d9fa7b0d0a8ffbd069f4'
REVERSED_MODEL = '174b1a50edd78e102eec11ba80b324e855b6299b7c8789f2cb69df23b878a6f5'
_REVERSED_ORIGIN = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULL_REVERSE_ORDER_20261110_002')
REVERSED_REFS = {
    'reverse_recipe': {'path': str(_REVERSED_ORIGIN / 'RECIPE.json'), 'sha256': '9f8cb54f927e0a52541be93cdbd2c821e9d461c6aff9c27f2def0d11daed14df'},
    'reverse_result': {'path': str(_REVERSED_ORIGIN / 'RESULT.json'), 'sha256': 'be7a403c827e5a2883d7ae6c29ccd345c2ce4feb599da402ec41e601e001347d'},
    'reverse_terminal': {'path': str(_REVERSED_ORIGIN / 'TERMINAL.json'), 'sha256': '40a284e6ee0648c46165e85e28e56494e9378b3e9f0e6457ad93da67d1dcf2fe'},
    'reverse_review': {'path': str(REPO / 'docs/TA_M5_FULL_REVERSE_ORDER_002_REVIEW.json'), 'sha256': '07a82392f2291979b98887f2b8cf73ff5935f31389ff00385e7bf32870fd82eb'},
    'reversed_evidence': {'path': str(_REVERSED_ORIGIN / 'REVERSED_EVIDENCE.pt'), 'sha256': 'fc59501b6a2a016dc642bacf92a6a0ad1a6a10068f2831855fa8c65bec06ceb1'},
    'train_measurement': {'path': str(_REVERSED_ORIGIN / 'REVERSED_TRAIN.json'), 'sha256': '8a53e7af2682190ec42eeab437af6f8ae0df91ab491384dfa57ec459b942fea0'},
    'train_capture': {'path': str(_REVERSED_ORIGIN / 'REVERSED_TRAIN_READOUT.npz'), 'sha256': '31a9ac06ecf10acb77545a2f5665295b8fe09d33e559d36434302f6e2815257b'},
}
REVERSED_BOUNDS = dict(fit_rows=29574, score_rows=0, batch_size=16, decoder_width=128,
    max_backbone_encodings=1849, max_model_constructions=1, max_checkpoint_loads=1,
    max_optimizer_constructions=0, max_backwards=0, max_optimizer_updates=0,
    max_checkpoint_writes=0, max_cache_bytes=67108864, device='cuda', wall_seconds=3600,
    training_enabled=False, test_data_used=False)


def _reversed_plan(plan, plan_path, out_dir):
    fields = {'schema_version', 'stage', 'source_repo', 'source_base_commit', 'worker',
              'dispatch', 'cap_runner', 'runtime_source_override', 'inputs'} | set(REVERSED_BOUNDS)
    _need(set(plan) == fields and plan['schema_version'] == REVERSED_SCHEMA
          and plan['stage'] == REVERSED_STAGE and plan['source_repo'] == str(REPO)
          and Path(plan_path) == REVERSED_ROOT / 'PLAN.json' and Path(out_dir) == REVERSED_ROOT
          and plan['inputs'] == REVERSED_REFS
          and all(type(plan[k]) is type(v) and plan[k] == v for k, v in REVERSED_BOUNDS.items()),
          'REVERSED_EXACT_PLAN')
    _need(isinstance(plan['source_base_commit'], str)
          and campaign._COMMIT.fullmatch(plan['source_base_commit']), 'REVERSED_SOURCE_BASE')
    return plan


def _reversed_sources(plan, recipe):
    """Only the exact new cap selector differs from the consumed 85-owner recipe."""
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    owners = recipe['source_owners']
    if plan.get('non_m5_learning_scope') is not None:
        from gx1.contracts import market_state_ta_non_m5_short_learning_v1 as non_m5
        _need(plan['non_m5_learning_scope'] == non_m5.SCOPE, 'NON_M5_LEARNING_SCOPE')
        return non_m5.current_source_bridge(owners, plan['non_m5_learning_source_owners'])
    if plan.get('tf_activity_initial_scope') is not None:
        from gx1.audit import observe_market_state_ta_tf_activity_initial_v1 as activity
        _need(plan['tf_activity_initial_scope'] == activity.SCOPE, 'TF_ACTIVITY_INITIAL_SCOPE')
        return activity.current_source_bridge(owners, plan['tf_activity_initial_source_owners'])
    if plan.get('non_m5_initial_scope') is not None:
        from gx1.audit import observe_market_state_ta_non_m5_initial_v1 as non_m5
        _need(plan['non_m5_initial_scope'] == non_m5.SCOPE, 'NON_M5_INITIAL_SCOPE')
        return non_m5.current_source_bridge(owners, plan['non_m5_initial_source_owners'])
    if plan.get('native_learning_scope') is not None:
        from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
        _need(plan['native_learning_scope'] == short.SCOPE, 'REVERSED_NATIVE_LEARNING_SCOPE')
        return short.current_source_bridge(owners, plan['native_learning_source_owners'])
    _need(set(owners) == fit.required_source_paths(ta_sf=True) and len(owners) == 85,
          'REVERSED_OWNER_SET')
    override = plan['runtime_source_override']
    _need(override == {'original_ref': owners[CAP_RUNNER], 'current_ref': plan['cap_runner']}
          and override['current_ref']['path'] == str(REPO / CAP_RUNNER)
          and override['original_ref']['sha256'] != override['current_ref']['sha256'],
          'REVERSED_CAP_ONLY_BRIDGE')
    current = {}
    for name, ref in owners.items():
        _need(ref['path'] == str(REPO / name), 'REVERSED_OWNER_PATH')
        if not name.startswith('.venv/'):
            raw = subprocess.check_output(['git', '-C', str(REPO), 'show', REVERSED_SOURCE + ':' + name])
            _need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'REVERSED_HISTORICAL_OWNER:' + name)
        live = plan['cap_runner'] if name == CAP_RUNNER else ref
        campaign.require_binding(live, label='reversed current owner ' + name, verify_file=True)
        current[name] = live['sha256']
    return current


def _qualify_reversed_origin(plan, bound_refs):
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts import market_state_native_F_private_campaign_v1 as reverse
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    recipe, result, terminal, measure = (_read(plan['inputs'][k]) for k in (
        'reverse_recipe', 'reverse_result', 'reverse_terminal', 'train_measurement'))
    _read(plan['inputs']['reverse_review'])
    _need(recipe['recipe_sha256'] == campaign.canonical_sha256({k: v for k, v in recipe.items() if k != 'recipe_sha256'})
          and recipe['run_scope'] == reverse.TA_M5_FULL_REVERSE_ORDER_SCOPE
          and recipe['admission_stage'] == 'market_state_ta_m5_full_reverse_order_20261110_002'
          and recipe['batch_indices'] == list(range(1847, -1, -1)) + [1848]
          and recipe['numerical_recipe']['max_updates'] == 1849, 'REVERSED_RECIPE_IDENTITY')
    _need(result['recipe'] == plan['inputs']['reverse_recipe']
          and result['outcome'] == 'COMPLETE' and result['source_commit'] == REVERSED_SOURCE
          and result['decision'] == 'PASS_TA_M5_FULL_REVERSE_ORDER_TECHNICAL'
          and result['source_unchanged'] is True and result['test_data_used'] is False
          and result['evaluation_state_unchanged'] is True and result['all_panels_complete'] is True
          and result['optimizer_updates'] == result['physical_optimizer_updates'] == 1849
          and result['parent_cursor'] == FIT_ROWS and result['skipped_batches'] == 0
          and result['candidate_evidence'] == plan['inputs']['reversed_evidence']
          and result['measurement_refs']['REVERSED_TRAIN'] == plan['inputs']['train_measurement']
          and result['captures']['REVERSED_TRAIN'] == plan['inputs']['train_capture']
          and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
          and terminal['recipe'] == plan['inputs']['reverse_recipe'] and terminal['result'] == plan['inputs']['reverse_result']
          and terminal['no_relaunch_under_this_plan'] is True and terminal['test_data_used'] is False,
          'REVERSED_CONSUMED_ORIGIN')
    _need(measure['evaluated_model_state_sha256'] == REVERSED_MODEL
          and measure['capture'] == plan['inputs']['train_capture'] and measure['optimizer_updates'] == 1849
          and measure['test_data_used'] is False and measure['measurement']['rows'] == FIT_ROWS
          and measure['measurement']['batches'] == 1849, 'REVERSED_MEASUREMENT')
    current = _reversed_sources(plan, recipe)
    # Reuse the native preflight's input/view/prior seams. Its old cap-byte check
    # is replaced only here; the consumed recipe and its validators stay unchanged.
    native_short = plan.get('native_learning_scope') is not None
    non_m5_initial = plan.get('non_m5_initial_scope') is not None
    tf_activity_initial = plan.get('tf_activity_initial_scope') is not None
    non_m5_learning = plan.get('non_m5_learning_scope') is not None
    _need(sum((native_short, non_m5_initial, non_m5_learning, tf_activity_initial)) <= 1, 'ONE_NATIVE_SCOPE')
    if non_m5_learning:
        from gx1.contracts import market_state_ta_non_m5_short_learning_v1 as non_m5
        from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
        _need(plan['non_m5_learning_scope'] == non_m5.SCOPE, 'NON_M5_LEARNING_SCOPE')
        origin = short.read_reversed_upstream(recipe)
    elif tf_activity_initial:
        from gx1.audit import observe_market_state_ta_tf_activity_initial_v1 as activity
        from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
        _need(plan['tf_activity_initial_scope'] == activity.SCOPE, 'TF_ACTIVITY_INITIAL_SCOPE')
        origin = short.read_reversed_upstream(recipe)
    elif non_m5_initial:
        from gx1.audit import observe_market_state_ta_non_m5_initial_v1 as non_m5
        from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
        _need(plan['non_m5_initial_scope'] == non_m5.SCOPE, 'NON_M5_INITIAL_SCOPE')
        origin = short.read_reversed_upstream(recipe)
    elif native_short:
        from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
        _need(plan['native_learning_scope'] == short.SCOPE, 'REVERSED_NATIVE_LEARNING_SCOPE')
        origin = short.read_reversed_upstream(recipe)
    else:
        origin = reverse.read_ta_m5_full_reverse_order_origin(recipe)
    for name, ref in recipe['input_refs'].items():
        if (native_short or non_m5_initial or non_m5_learning or tf_activity_initial) and name == 'original_trace_helper':
            short.require_historical_trace(ref)
            continue
        campaign.require_binding(ref, label='reversed reused input', verify_file=True)
        bound_refs.append(ref)
    campaign.require_binding(recipe['trace_worker'], label='reversed capsule producer', verify_file=True)
    bound_refs.append(recipe['trace_worker'])
    receipts = zero.load_ta_input_receipts(current_source_owners=current)
    view = zero._bind_ta_view(receipts, dedicated_m5_side=True)
    historical_view = _read(origin['original_recipe']['input_refs']['ta_candidate_view'])
    if tf_activity_initial:
        activity.require_view_source_transition(historical_view, view, plan)
    elif non_m5_initial or non_m5_learning:
        non_m5.require_view_source_transition(historical_view, view, plan)
    else:
        _need(view == historical_view and view['contract_sha256'] == recipe['expected_view_sha256'],
              'REVERSED_ORIGINAL_VIEW')
    _need(historical_view['contract_sha256'] == recipe['expected_view_sha256']
          and recipe['numerical_recipe'] == origin['original_recipe']['numerical_recipe'],
          'REVERSED_ORIGINAL_VIEW_NUMERICAL')
    fit.require_checkpoint_view(origin['original_result']['feature_view_identity'],
        recipe=origin['original_recipe'], verify_files=True, ta_sf=True)
    priors = _read(recipe['input_refs']['endpoint_priors'])
    objective.require_endpoint_priors(priors, fold_id=1, ta_only=True, dedicated_m5_side=True)
    return dict(recipe=recipe, result=result, measurement=measure, current_sources=current,
                origin=origin, receipts=receipts, view=view, priors=priors)


def _inspect_reversed(plan_path, plan_sha256, out_dir, *, execute):
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    plan = _reversed_plan(_read(plan_ref), plan_path, out_dir)
    bound_refs = [plan_ref]
    for key, path in (('worker', WORKER), ('dispatch', REPO / DISPATCH), ('cap_runner', REPO / CAP_RUNNER)):
        _need(plan[key]['path'] == str(path), 'REVERSED_SOURCE_PATH:' + key)
        campaign.require_binding(plan[key], label=key, verify_file=True)
        bound_refs.append(plan[key])
    for key, ref in plan['inputs'].items():
        campaign.require_binding(ref, label='reversed input ' + key, verify_file=True)
        bound_refs.append(ref)
    qualified = _qualify_reversed_origin(plan, bound_refs)
    recipe, result, measure, current, origin, receipts, view, priors = (
        qualified[k] for k in ('recipe', 'result', 'measurement', 'current_sources',
                               'origin', 'receipts', 'view', 'priors'))
    head = _git('rev-parse', 'HEAD')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor',
                    plan['source_base_commit'], head], check=True)
    policy_ref = _ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = _read(policy_ref); bound_refs.append(policy_ref)
    if execute:
        expected = dict(status='ADMITTED', plan=plan_ref, max_invocations=1, claimed_invocations=0,
            completed_invocations=0, relaunch_allowed=False, execution_authorized=True,
            data_reads_authorized=True, checkpoint_reads_authorized=True,
            backbone_encodings_authorized=1849, backward_calls_authorized=0,
            optimizer_steps_authorized=0, training_authorized=False)
        gate = policy.get(REVERSED_STAGE, {})
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k, v in expected.items())
              and all(policy.get(k) is False for k in (
                  'training_enabled', 'native_training_enabled', 'native_launch_authorized',
                  'full_epoch_training_allowed')),
              'REVERSED_ADMITTED_POLICY')
        _need(_git('branch', '--show-current') == 'work/gx1-current', 'REVERSED_BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO), source_commit=head))
        _need(not any((REVERSED_ROOT / name).exists() for name in ('CLAIM.json', 'RESULT.json', 'TERMINAL.json')),
              'REVERSED_CONSUMED')
    return dict(plan=plan, plan_ref=plan_ref, recipe=recipe, result=result, measurement=measure,
        origin=origin, receipts=receipts, view=view, priors=priors, head=head, policy_ref=policy_ref,
        bound_refs=bound_refs, current_sources=current)


def _load_reversed_weights(model, ck, checked, order, *, torch, initial=False):
    """Validate REVERSED or original INITIAL; copy weights/all buffers, never optimizer/RNG."""
    import copy
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    fields = set(('schema_version arm recipe source_commit original_checkpoint canonical_resume_authority '
        'model_state buffer_state optimizer_state rng_state module_training parent_order parent_cursor '
        'physical_optimizer_updates pending_gradients parameter_binding original_numerical_recipe_reference '
        'method_state model_state_sha256 buffer_state_sha256 optimizer_state_sha256 rng_state_sha256 '
        'test_data_used evidence_sha256').split())
    recipe, refs = checked['recipe'], checked['plan']['inputs']
    expected_model = FUSION_INITIAL_BOUNDS['candidate_model_state_sha256'] if initial else REVERSED_MODEL
    if initial:
        _need(isinstance(ck, dict) and set(ck) == state._STATE_KEYS
              and ck['schema_version'] == state.TA_STATE_SCHEMA
              and ck['source_commit'] == checked['origin']['original_result']['source_commit']
              and type(ck['fold_id']) is int and ck['fold_id'] == 1
              and ck['feature_view_sha256'] == recipe['expected_view_sha256']
              and all(type(ck[k]) is int and ck[k] == 0 for k in ('cursor','optimizer_updates','skipped_batches'))
              and ck['model_state_sha256'] == recipe['expected_initial_model_state_sha256'] == expected_model,
              'FUSION_INITIAL_STATE_HEADER')
        _need(ck['contract_sha256'] == old._tree_sha({k:v for k,v in ck.items() if k != 'contract_sha256'}),
              'FUSION_INITIAL_STATE_SEAL')
    else:
        _need(isinstance(ck, dict) and set(ck) == fields
              and ck['schema_version'] == 'gx1_ta_m5_full_reverse_order_evidence_v1'
              and ck['arm'] == 'REVERSED' and ck['recipe'] == refs['reverse_recipe']
              and ck['source_commit'] == REVERSED_SOURCE
              and ck['original_checkpoint'] == recipe['input_refs']['original_initial_checkpoint']
              and ck['canonical_resume_authority'] is False and ck['test_data_used'] is False
              and ck['physical_optimizer_updates'] == 1849 and ck['parent_cursor'] == FIT_ROWS
              and ck['pending_gradients'] == 0 and ck['model_state_sha256'] == expected_model,
              'REVERSED_EVIDENCE_HEADER')
        _need(ck['evidence_sha256'] == old._tree_sha({k:v for k,v in ck.items() if k != 'evidence_sha256'}),
              'REVERSED_EVIDENCE_SEAL')
    numerical = state._numerical(recipe['numerical_recipe'], model)
    order_sha = old._tree_sha(order)
    if initial:
        _need(isinstance(ck['parent_order'], torch.Tensor) and ck['parent_order'].dtype == torch.int64
              and torch.equal(ck['parent_order'], order)
              and order_sha == checked['origin']['original_result']['parent_order_sha256'],
              'FUSION_INITIAL_STATE_ORDER')
    else:
        _need(ck['original_numerical_recipe_reference'] == numerical
              and ck['parent_order'].dtype == torch.int64 and torch.equal(ck['parent_order'], order)
              and order_sha == checked['result']['parent_order_sha256']
              and ck['method_state'] == dict(batch_indices=recipe['batch_indices'],
                  batch_order_sha256=recipe['batch_order_sha256'], full_order_sha256=order_sha,
                  original_fit_batch_starts=[16 * b for b in recipe['batch_indices']], pass_index=0,
                  all_original_rows_once=True, tail_policy='original_tail_last', loss_and_optimizer_unchanged=True),
              'REVERSED_EVIDENCE_ORDER_NUMERICAL')
    pairs, active, inactive = state._parameters(model)
    binding = dict(schema_version=state.TA_BINDING_SCHEMA, numerical_recipe=numerical,
        feature_view_identity=model.state_model.feature_view_identity, research_objective=model.research_objective,
        active_parameter_names=[n for n, _ in active], inactive_parameter_names=[n for n, _ in inactive],
        parameter_inventory=[dict(name=n, shape=list(v.shape), dtype=str(v.dtype)) for n, v in pairs],
        optimizer_group_parameter_names=[[n for n, _ in active]], ema_created=False,
        scheduler_created=False, target_model_created=False)
    binding['contract_sha256'] = old._tree_sha(binding)
    _need(ck['binding' if initial else 'parameter_binding'] == binding, 'REVERSED_PARAMETER_BINDING')
    old._tensor_map(ck['model_state'], model.state_dict(), 'REVERSED_MODEL')
    old._tensor_map(ck['buffer_state'], dict(model.named_buffers()), 'REVERSED_BUFFERS')
    _need(model_sha(ck['model_state']) == expected_model, 'REVERSED_MODEL_DIGEST')
    for key in ('buffer_state', 'optimizer_state', 'rng_state'):
        _need(old._tree_sha(ck[key]) == ck[key + '_sha256'], 'REVERSED_DIGEST:' + key)
    immutable = {'native_F_objective_sha256'} | {'state_model.' + n for n in old.IMMUTABLE_BUFFER_NAMES | state._QUERY_BUFFERS}
    prefixes = tuple('state_model.' + n for n in old.IMMUTABLE_BUFFER_PREFIXES)
    for name, value in model.named_buffers():
        saved = ck['buffer_state'][name]
        if name in immutable or name.startswith(prefixes):
            _need(torch.equal(saved, value), 'REVERSED_IMMUTABLE_BUFFER:' + name)
        if name in ck['model_state']:
            _need(torch.equal(saved, ck['model_state'][name]), 'REVERSED_PERSISTENT_BUFFER:' + name)
    _need(set(ck['module_training']) == set(dict(model.named_modules()))
          and all(type(v) is bool for v in ck['module_training'].values()), 'REVERSED_MODULE_MODES')
    opt = ck['optimizer_state']; kw = copy.deepcopy(numerical['optimizer']['kwargs']); kw['betas'] = tuple(kw['betas'])
    _need(set(opt) == {'state', 'param_groups'} and len(opt['param_groups']) == 1, 'REVERSED_OPTIMIZER_METADATA')
    group = opt['param_groups'][0]
    _need(set(group) == set(kw) | {'params'} and group['params'] == list(range(len(active)))
          and old._tree_sha({k: group[k] for k in kw}) == old._tree_sha(kw)
          and isinstance(opt['state'], dict) and bool(opt['state']) is (not initial)
          and set(opt['state']) <= set(group['params']), 'REVERSED_OPTIMIZER_ORDER')
    for index, item in opt['state'].items():
        _need(type(index) is int and set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'REVERSED_ADAM_SCHEMA')
        step = item['step']
        _need(step.ndim == 0 and step.dtype in (torch.float32, torch.float64) and bool(torch.isfinite(step))
              and float(step).is_integer() and 0 < float(step) <= 1849, 'REVERSED_ADAM_STEP')
        for key in ('exp_avg', 'exp_avg_sq'):
            value, parameter = item[key], active[index][1]
            _need(value.shape == parameter.shape and value.dtype == parameter.dtype
                  and bool(torch.isfinite(value).all()), 'REVERSED_ADAM_MOMENT')
        _need(bool((item['exp_avg_sq'] >= 0).all()), 'REVERSED_ADAM_SECOND_MOMENT')
    cuda_rng = ck['rng_state']['torch_cuda']
    _need(isinstance(cuda_rng, list) and bool(cuda_rng) and all(
        isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.ndim == 1 and 0 < v.numel() <= 65536
        for v in cuda_rng), 'REVERSED_STORED_CUDA_RNG')
    old._rng({k: v for k, v in ck['rng_state'].items() if k != 'torch_cuda'},
             device=torch.device('cpu'), seed=numerical['seed'])
    model.load_state_dict(ck['model_state'], strict=True)
    with torch.no_grad():
        for name, value in model.named_buffers(): value.copy_(ck['buffer_state'][name])
    state._require_model_binding(model); model.backbone.require_input_normalization_state()
    _need(model_sha(model.state_dict()) == expected_model
          and old._tree_sha(dict(model.named_buffers())) == ck['buffer_state_sha256'], 'REVERSED_LOADED_EXACT')
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return dict(model_state_sha256=expected_model, buffer_state_sha256=ck['buffer_state_sha256'],
        **(dict(checkpoint=refs['initial_checkpoint'], source_commit=ck['source_commit'],
                state_seal=ck['contract_sha256']) if initial else dict(evidence_sha256=ck['evidence_sha256'])),
        parameter_binding_sha256=binding['contract_sha256'],
        load_kind='WEIGHTS_AND_ALL_BUFFERS_ONLY', optimizer_state_restored=False,
        cuda_rng_restored=False, canonical_resume_authority=False)


def _reversed_batch_parity(captured, baseline, left, right, *, np):
    axes = all(np.array_equal(captured[k], baseline[k][left:right]) for k in ('parents', 'issues'))
    flags, errors = {}, {}
    for task in ('state', 'future'):
        got = captured[task + '_activity_side_logits'][:, 0]
        expected = baseline[task + '_raw'][left:right, 0]
        _need(got.dtype == expected.dtype == np.float32, 'REVERSED_RAW_DTYPE')
        flags[task] = (np.array_equal(got.view(np.uint32), expected.view(np.uint32))
            and np.array_equal(captured['query_native'][:, 0], baseline[task + '_query'][left:right, 0]))
        flags[task + '_masks'] = all(np.array_equal(captured[task + '_' + key][:, 0],
            baseline[task + '_' + key][left:right, 0]) for key in ('allowed', 'eligible', 'mature'))
        errors[task] = float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))
    return dict(M5_raw_bitexact=flags['state'] and flags['future'],
        exact_M5_axes_masks=axes and flags['state_masks'] and flags['future_masks'],
        max_abs_M5_raw_error=errors)


def _save_reversed(arrays, rows, *, partial):
    import numpy as np
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    name = 'FIT_REPRESENTATIONS' + ('_PARTIAL' if partial else '') + '.npz'
    stream = io.BytesIO()
    np.savez(stream, **{k: v[:rows] for k, v in arrays.items()})
    _need(0 < stream.tell() <= REVERSED_BOUNDS['max_cache_bytes'], 'REVERSED_CACHE_BYTES')
    _publish_bytes(REVERSED_ROOT / name, stream.getvalue())
    return _ref(REVERSED_ROOT / name)


def _run_reversed(plan_path, plan_sha256, out_dir, *, device):
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_endpoint_batch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    started = time.monotonic()
    _need(device == 'cuda', 'REVERSED_GPU_ONLY')
    cap = require_guarded_cuda_producer_execution(); cap['lock'] = require_capped_lock_ancestry()
    _need(cap['cgroup_path'] == '/sys/fs/cgroup' + cap['lock']['scope']
          and cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
          and cap['swap'] == 512 * 1024**2, 'REVERSED_EXACT_CAP')
    for key, value in (('MAX_POWER_LIMIT_W', '250'), ('MAX_POWER_DRAW_W', '170'), ('MAX_CORE_TEMP_C', '65'), ('MAX_MEMORY_TEMP_C', '80')):
        _need(os.environ.get('GX1_TRAINER_GPU_' + key) == value, 'REVERSED_HARDWARE_GUARD:' + key)
    checked = _inspect_reversed(plan_path, plan_sha256, out_dir, execute=True)
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'REVERSED_INIT_ORDER')
    publish = lambda name, value: _publish(name, value, root=REVERSED_ROOT)
    claim = publish('CLAIM.json', dict(plan=checked['plan_ref'], policy=checked['policy_ref'],
        source_commit=checked['head'], cap=cap, training_enabled=False, test_data_used=False))
    counts = dict(model_constructions=0, checkpoint_loads=0, backbone_encodings=0, forward_returns=0,
        optimizer_constructions=0, optimizer_updates=0, backwards=0, checkpoint_writes=0,
        resume_state_load_calls=0, resume_state_load_returns=0)
    provider = prepared = model = None; arrays = {}; rows_done = 0; artifact = None; identity = {}
    parity = dict(M5_raw_bitexact=True, exact_M5_axes_masks=True, max_abs_M5_raw_error={'state': 0., 'future': 0.})
    timings = {}; before = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('REVERSED cache 3600-second bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1., WALL_SECONDS - 60 - (time.monotonic() - started)))
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO), source_commit=checked['head']))
        for ref in checked['bound_refs']: campaign.require_binding(ref, label='reversed bookend', verify_file=True)
        for name, digest in checked['current_sources'].items():
            _need(campaign.file_sha256(REPO / name) == digest, 'REVERSED_SOURCE_BOOKEND:' + name)
    try:
        recipe, receipts, view = (checked[k] for k in ('recipe', 'receipts', 'view'))
        numerical = recipe['numerical_recipe']; torch_device = torch.device('cuda')
        trainer._set_deterministic(numerical['seed'], torch_device, numerical['precision_policy'])
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _need(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'REVERSED_THREADS')
        provider, provider_plan, prepared, rows, issues = runner._prepared_fold1(receipts, view, np=np)
        endpoint, future = load_bound_five_endpoint_targets(), load_bound_future_displacement()
        axes = endpoint.mature_oof_axes(fold_id=1)
        _need(np.array_equal(rows, axes['fit_parent_rows']) and np.array_equal(issues, axes['fit_decision_ns'])
              and len(rows) == FIT_ROWS, 'REVERSED_FIT_AXIS')
        keys = ['parents', 'issues'] + [task + '_' + key for task in ('state', 'future')
            for key in ('raw', 'query', 'allowed', 'eligible', 'mature')]
        with np.load(checked['plan']['inputs']['train_capture']['path'], allow_pickle=False) as archive:
            baseline = {key: archive[key].copy() for key in keys}
        _need(np.array_equal(rows, baseline['parents']) and np.array_equal(issues, baseline['issues']), 'REVERSED_CAPTURE_AXIS')
        counts['model_constructions'] += 1
        backbone = views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,
            recipe=checked['origin']['original_recipe'], verify_files=True, ta_sf=True)
        _need(model.research_objective['contract_sha256'] == recipe['expected_model_contract_sha256'], 'REVERSED_MODEL_CONTRACT')
        order, _ = build_reverse_order(rows, recipe['batch_indices'])
        counts['checkpoint_loads'] += 1
        ck = common._load_state(checked['plan']['inputs']['reversed_evidence'], torch=torch,
            maximum=recipe['max_checkpoint_bytes'], counters=counts)
        identity = _load_reversed_weights(model, ck, checked, torch.from_numpy(order.copy()), torch=torch)
        del ck
        model.to(torch_device)
        def signature():
            return dict(model=model_sha(model.state_dict()), buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=torch_device)),
                modes={n: bool(m.training) for n, m in model.named_modules()})
        before = signature()
        _need(before['model'] == REVERSED_MODEL and not any(p.requires_grad for p in model.parameters()), 'REVERSED_FROZEN')
        arrays = {key: np.empty(shape, dtype) for key, (shape, dtype) in runner._ta_m5_capture_layout(FIT_ROWS).items()}
        timings['setup_seconds'] = time.monotonic() - started; tick = time.monotonic()
        for left in range(0, FIT_ROWS, BATCH_SIZE):
            right = min(left + BATCH_SIZE, FIT_ROWS)
            raw, parents, clocks = runner._feature_batch(prepared, view, receipts, rows, issues, left, right, np=np)
            features = {key: torch.from_numpy(value).to(torch_device) for key, value in raw.items()}
            _need(counts['backbone_encodings'] < 1849, 'REVERSED_FORWARD_CAP')
            counts['backbone_encodings'] += 1
            report = evaluate_endpoint_batch(model, features, endpoint, future, parents, clocks,
                fold_id=1, partition='fit', future_weight=1., frozen_priors=checked['priors'], capture_m5_readout=True)
            counts['forward_returns'] += 1
            captured = report['m5_readout']
            runner._ta_m5_capture_batch(arrays, captured, left=left, right=right, parents=parents, issues=clocks)
            rows_done = right
            one = _reversed_batch_parity(captured, baseline, left, right, np=np)
            for key in ('M5_raw_bitexact', 'exact_M5_axes_masks'): parity[key] &= one[key]
            for task in ('state', 'future'):
                parity['max_abs_M5_raw_error'][task] = max(parity['max_abs_M5_raw_error'][task], one['max_abs_M5_raw_error'][task])
            _need(one['M5_raw_bitexact'] and one['exact_M5_axes_masks'], 'REVERSED_M5_NATIVE_PARITY')
            if counts['backbone_encodings'] % 256 == 0:
                print(json.dumps(dict(status='PROGRESS', rows=rows_done, forwards=counts['backbone_encodings'],
                    elapsed_seconds=time.monotonic()-started)), flush=True)
        timings['capture_seconds'] = time.monotonic() - tick
        artifact = _save_reversed(arrays, rows_done, partial=False)
        _need(rows_done == FIT_ROWS and counts['backbone_encodings'] == counts['forward_returns'] == 1849
              and signature() == before and all(p.grad is None for p in model.parameters()), 'REVERSED_UNCHANGED')
        known = {task: int(np.sum(arrays[task+'_eligible'][:, 0] & arrays[task+'_mature'][:, 0]
            & np.isin(arrays[task+'_allowed'][:, 0], (1, 4)))) for task in ('state', 'future')}
        _need(known == {'state': 10010, 'future': 11909}, 'REVERSED_KNOWN_COUNTS')
        base._release_pages(provider, prepared); prepared = None
        provider.validate_metadata(provider_plan); bookend()
        result = publish('RESULT.json', dict(status='COMPLETE', plan=checked['plan_ref'], claim=claim,
            policy=checked['policy_ref'], source_commit=checked['head'], artifact=artifact,
            model_state_sha256=REVERSED_MODEL, identity=identity, rows=FIT_ROWS, batches=1849,
            query_domain='native_endpoints;M5_equal_to_pass0_sampled', parity_scope='M5_only',
            **parity, known_M5=known, counts=counts,
            arrays_metadata={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
            source_unchanged=True, model_state_unchanged=True, evaluation_state_unchanged=True,
            later_forwards=0, training_enabled=False, test_data_used=False,
            timings={**timings, 'elapsed_seconds': time.monotonic()-started}))
        terminal = publish('TERMINAL.json', dict(status='COMPLETE', exit_code=0, plan=checked['plan_ref'],
            result=result, artifact=artifact, source_unchanged=True,
            no_relaunch_under_this_plan=True, test_data_used=False))
        return dict(result=result, terminal=terminal)
    except BaseException as exc:
        preservation_error = None
        if artifact is None and arrays and rows_done:
            try: artifact = _save_reversed(arrays, rows_done, partial=True)
            except Exception as secondary: preservation_error = repr(secondary)
        source_unchanged = False
        try: bookend(); source_unchanged = True
        except Exception as secondary: preservation_error = (preservation_error or '') + ';bookend:' + repr(secondary)
        result = publish('RESULT.json', dict(status='FAILED_CONSUMED', plan=checked['plan_ref'], claim=claim,
            artifact=artifact, rows=rows_done, counts=counts, identity=identity, parity=parity,
            error=type(exc).__name__+':'+str(exc), preservation_error=preservation_error,
            source_unchanged=source_unchanged, training_enabled=False, test_data_used=False))
        publish('TERMINAL.json', dict(status='FAILED', exit_code=1, plan=checked['plan_ref'], result=result,
            artifact=artifact, source_unchanged=source_unchanged, no_relaunch_under_this_plan=True, test_data_used=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.); signal.signal(signal.SIGALRM, previous_alarm)
        if prepared is not None:
            base._release_pages(provider, prepared)

# A separately admitted, frozen native import plus the missing later TRAIN rows.
SCORE_STAGE = 'market_state_ta_m5_reversed_fixed_native_score_20261110_001'
SCORE_ROOT = Path('/home/andre2/GX1_RUNS') / SCORE_STAGE.upper()
SCORE_SCHEMA = 'gx1_market_state_ta_m5_reversed_fixed_native_score_plan_v1'
SCORE_WEIGHTS_SCHEMA = 'gx1_ta_m5_fixed_native_weights_only_v1'
_FIXED = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_REVERSED_FIXED_READOUT_20261110_001')
SCORE_REFS = {
    **REVERSED_REFS,
    'capture_result': {'path': str(REVERSED_ROOT / 'RESULT.json'), 'sha256': 'ba3767d638132a5056a106a479d6a37afe6e8c44be6e3ca71cb895188f4199b9'},
    'capture_plan': {'path': str(REVERSED_ROOT / 'PLAN.json'), 'sha256': '882470b193f03403e284c40dde7bbf2cc19cda215e9a5a983084b2b788a5b2b8'},
    'capture_completion_review': {'path': str(REVERSED_ROOT / 'CAPTURE_COMPLETION_REVIEW.json'), 'sha256': '3a5f79b70e3975752d2c0b91df91413058feded7e0d04c0b4109d7fa4a696486'},
    'capture_controller_terminal': {'path': str(REVERSED_ROOT / 'CONTROLLER_TERMINAL.json'), 'sha256': 'ff6ef1e4b1096fee9eaf7c1140371225b8a1cb6397c176d4d35a2c0bb7f830e9'},
    'fit_representations': {'path': str(REVERSED_ROOT / 'FIT_REPRESENTATIONS.npz'), 'sha256': '6e99d89a9a1d67372b691959507f7e73bc623b5a5c1488cf1a77c5b87a95ef8c'},
    'readout_plan': {'path': str(_FIXED / 'PLAN.json'), 'sha256': '1a8f270c81f72939e662a4e9cdc32e97a034deff75eec401b3d9c6b7a002b9dc'},
    'readout_result': {'path': str(_FIXED / 'RESULT.json'), 'sha256': '62e596192f628a05bd09b59675a3522044bf6b643a07076aa7c239e5da0d7f40'},
    'readout_terminal': {'path': str(_FIXED / 'TERMINAL.json'), 'sha256': '349f15e5aa62ac986bd0b1bdca4dd6ee15260faaafe6ed02e702e1e367ad0a94'},
    'readout_review': {'path': str(_FIXED / 'NUMERIC_REVIEW.json'), 'sha256': 'f8b952c4257d40ea493a9f40d30f2f2c501d6723ad3f6db35844b33bdb3160a3'},
    'readout_lock': {'path': str(_FIXED / 'READOUT_LOCK.json'), 'sha256': '1908c69e30318605634c2b1be8f0cb4747f8125e8db98f9cf6d1527509e40510'},
    'readout_outputs': {'path': str(_FIXED / 'PRIVATE_READOUTS_AND_PREDICTIONS.npz'), 'sha256': 'e75974bee5eb111aa4e23f2135832d9dfc796a7152a9dd8906710edcae8d4a73'},
    'later_capture': {'path': str(_REVERSED_ORIGIN / 'REVERSED_LATER_READOUT.npz'), 'sha256': 'aedb5f3e0e0ac901eb8066d018363d2721ab46e2ebe33ca3d838c43615582acb'},
    'later_measurement': {'path': str(_REVERSED_ORIGIN / 'REVERSED_LATER.json'), 'sha256': '1eb538c636835f7641341f1266250971851c21391aef7724311c5ffeca2cc694'},
    'initial_score_cache': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_FULL_CACHE_20261108_001/SCORE_CACHE.npz', 'sha256': '8cc964f1ff229bc3c1140a47ed66e256ce90b9ecfbc18c4c4509ba60f63bfee9'},
    'original_imported_outputs': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FINAL_READOUT_20261109_001/PRIVATE_READOUTS_AND_PREDICTIONS.npz', 'sha256': '317deaca134460b73e8ac200dff2741483c7729aa13d6d6870b4aeecf136846b'},
}
SCORE_BOUNDS = dict(fit_rows=29574, score_rows=17753, reused_score_rows=192, batch_size=16,
    decoder_width=128, max_backbone_encodings=1100, max_missing_score_encodings=1098,
    fresh_baseline_forwards=2, max_model_constructions=1, max_checkpoint_loads=2,
    max_candidate_writes=1, max_candidate_bytes=1073741824, max_cache_bytes=67108864,
    max_cached_head_calls=7444, max_reference_linear_calls=2196, head_tensor_imports=4,
    max_optimizer_constructions=0, max_backwards=0, max_optimizer_updates=0,
    max_checkpoint_writes=0, device='cuda', wall_seconds=3600, training_enabled=False,
    test_data_used=False)
SCORE_RANGES = ((0, 64), (8352, 8416), (17680, 17744))


def _fixed_score_plan(plan, plan_path, out_dir):
    fields = {'schema_version', 'stage', 'source_repo', 'source_base_commit', 'worker',
              'dispatch', 'cap_runner', 'runtime_source_override', 'inputs'} | set(SCORE_BOUNDS)
    _need(set(plan) == fields and plan['schema_version'] == SCORE_SCHEMA
          and plan['stage'] == SCORE_STAGE and plan['source_repo'] == str(REPO)
          and Path(plan_path) == SCORE_ROOT / 'PLAN.json' and Path(out_dir) == SCORE_ROOT
          and plan['inputs'] == SCORE_REFS
          and all(type(plan[k]) is type(v) and plan[k] == v for k, v in SCORE_BOUNDS.items()),
          'FIXED_SCORE_EXACT_PLAN')
    _need(isinstance(plan['source_base_commit'], str)
          and campaign._COMMIT.fullmatch(plan['source_base_commit']), 'FIXED_SCORE_SOURCE_BASE')
    return plan


def _inspect_fixed_score(plan_path, plan_sha256, out_dir, *, execute):
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    plan = _fixed_score_plan(_read(plan_ref), plan_path, out_dir)
    refs = plan['inputs']; bound_refs = [plan_ref]
    for key, path in (('worker', WORKER), ('dispatch', REPO / DISPATCH), ('cap_runner', REPO / CAP_RUNNER)):
        _need(plan[key]['path'] == str(path), 'FIXED_SCORE_SOURCE_PATH:' + key)
        campaign.require_binding(plan[key], label=key, verify_file=True); bound_refs.append(plan[key])
    for key, ref in refs.items():
        campaign.require_binding(ref, label='fixed score input ' + key, verify_file=True); bound_refs.append(ref)
    q = _qualify_reversed_origin(plan, bound_refs)
    capture, completion, readout, fit_plan, lock, review, terminal, later = (
        _read(refs[k]) for k in ('capture_result', 'capture_completion_review', 'readout_result',
            'readout_plan', 'readout_lock', 'readout_review', 'readout_terminal', 'later_measurement'))
    _need(capture['status'] == 'COMPLETE' and capture['plan'] == refs['capture_plan']
          and capture['artifact'] == refs['fit_representations'] and capture['model_state_sha256'] == REVERSED_MODEL
          and capture['M5_raw_bitexact'] is True and capture['exact_M5_axes_masks'] is True
          and capture['source_unchanged'] is True and capture['test_data_used'] is False
          and completion['status'] == 'QUALIFIED_CAPTURE_WITH_POST_COMPLETION_CLEANUP_ERROR'
          and completion['result'] == refs['capture_result'] and completion['artifact'] == refs['fit_representations']
          and completion['controller_terminal'] == refs['capture_controller_terminal']
          and completion['worker_exit_code'] == 0 and completion['controller_exit_code'] == 1
          and completion['numeric_checks']['rows'] == FIT_ROWS
          and completion['numeric_checks']['M5_raw_bitexact'] is True,
          'FIXED_SCORE_CAPTURE_QUALIFICATION')
    _need(readout['status'] == 'COMPLETE' and readout['plan'] == refs['readout_plan']
          and readout['readout_lock'] == refs['readout_lock'] and readout['private_outputs'] == refs['readout_outputs']
          and readout['source_unchanged'] is True and readout['test_data_used'] is False
          and readout['selection'] is False and readout['counts']['readout_fits'] == 2
          and lock['plan'] == refs['readout_plan'] and lock['readout_fits'] == 2
          and lock['later_arrays_decoded'] is False and lock['alpha'] == fit_plan['alpha'] == readout['alpha'] == .01
          and all(lock['solvers'][task]['converged'] is True for task in ('state', 'future'))
          and review['status'] == 'PASS_INDEPENDENT_RETAINED_LOGIT_ARITHMETIC'
          and review['result'] == refs['readout_result'] and review['readout_lock'] == refs['readout_lock']
          and review['private_outputs'] == refs['readout_outputs'] and review['test_data_used'] is False
          and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
          and terminal['plan'] == refs['readout_plan'] and terminal['result'] == refs['readout_result'],
          'FIXED_SCORE_LOCKED_READOUT')
    _need(fit_plan['inputs']['fit_representations'] == refs['fit_representations']
          and fit_plan['inputs']['REVERSED_LATER_capture'] == refs['later_capture']
          and fit_plan['inputs']['capture_completion_review'] == refs['capture_completion_review']
          and later['evaluated_model_state_sha256'] == REVERSED_MODEL and later['capture'] == refs['later_capture']
          and later['measurement']['rows'] == 192 and later['test_data_used'] is False
          and q['result']['captures']['REVERSED_LATER'] == refs['later_capture']
          and q['recipe']['input_refs']['initial_score_cache'] == refs['initial_score_cache']
          and q['recipe']['input_refs']['imported_predictions'] == refs['original_imported_outputs'],
          'FIXED_SCORE_MATCHED_INPUTS')
    head = _git('rev-parse', 'HEAD')
    subprocess.run(['git', '-C', str(REPO), 'merge-base', '--is-ancestor', plan['source_base_commit'], head], check=True)
    policy_ref = _ref(REPO / 'NEXT_RUN_POLICY.json'); policy = _read(policy_ref); bound_refs.append(policy_ref)
    if execute:
        gate = policy.get(SCORE_STAGE, {})
        expected = dict(status='ADMITTED', plan=plan_ref, max_invocations=1, claimed_invocations=0,
            completed_invocations=0, relaunch_allowed=False, execution_authorized=True, data_reads_authorized=True,
            checkpoint_reads_authorized=True, backbone_encodings_authorized=1100,
            head_tensor_imports_authorized=4, candidate_weights_writes_authorized=1,
            backward_calls_authorized=0, optimizer_steps_authorized=0, training_authorized=False)
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k, v in expected.items())
              and all(policy.get(k) is False for k in ('training_enabled', 'native_training_enabled',
                  'native_launch_authorized', 'full_epoch_training_allowed')), 'FIXED_SCORE_ADMISSION')
        _need(_git('branch', '--show-current') == 'work/gx1-current', 'FIXED_SCORE_BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO), source_commit=head))
        _need(not any((SCORE_ROOT / k).exists() for k in ('CLAIM.json', 'RESULT.json', 'TERMINAL.json',
                                                       'CANDIDATE_WEIGHTS.pt')), 'FIXED_SCORE_CONSUMED')
    return dict(q, plan=plan, plan_ref=plan_ref, readout=readout, lock=lock, head=head,
                policy_ref=policy_ref, bound_refs=bound_refs)


def _fixed_score_layout():
    reused = [i for left, right in SCORE_RANGES for i in range(left, right)]
    cached_batches = {left // 16 + j for left, right in SCORE_RANGES for j in range((right - left) // 16)}
    missing = [i for i in range((SCORE_ROWS + 15) // 16) if i not in cached_batches]
    _need(len(reused) == len(set(reused)) == 192 and len(cached_batches) == 12
          and len(missing) == 1098 and missing[-1] == 1109, 'FIXED_SCORE_PARTITION')
    return reused, missing


def _fixed_coefficients(archive, lock, *, np, torch):
    tensors = {}
    for task, prefix in (('state', 'state_model.m5_side_head'), ('future', 'future_m5_side_head')):
        beta, rounded = archive[task + '_beta_fp64'], archive[task + '_beta_fp32']
        _need(beta.dtype == np.float64 and rounded.dtype == np.float32 and beta.shape == rounded.shape == (129,)
              and np.isfinite(beta).all() and np.isfinite(rounded).all()
              and hashlib.sha256(beta.tobytes()).hexdigest() == lock['beta_fp64_sha256'][task]
              and np.array_equal(beta.astype(np.float32).view(np.uint32), rounded.view(np.uint32)),
              'FIXED_SCORE_COEFFICIENT_LOCK:' + task)
        tensors[prefix + '.weight'] = torch.from_numpy(rounded[1:].copy().reshape(1, 128))
        tensors[prefix + '.bias'] = torch.from_numpy(rounded[:1].copy())
    return tensors


def _fixed_score_decision(scores):
    cells = {}
    for panel in ('TRAIN', 'LATER_FULL'):
        cells[panel] = {}
        for task in ('state', 'future'):
            values = scores[panel][task]; got = values['CANDIDATE']
            comparisons = {}
            for name, margin in (('INITIAL', 1e-6), ('PRIOR', 1e-8), ('R1', 1e-8)):
                other = values[name]
                _need(got['count'] == other['count'] and got['up_count'] == other['up_count'],
                      'FIXED_SCORE_COMPARISON_OBSERVATIONS')
                comparisons[name] = dict(margin=margin,
                    brier_delta=got['brier']-other['brier'], log_delta=got['log']-other['log'],
                    passes=got['brier'] < other['brier']-margin and got['log'] < other['log']-margin)
            cells[panel][task] = comparisons
    passed = all(cells[p][t][r]['passes'] for p in cells for t in cells[p] for r in ('INITIAL', 'PRIOR'))
    return dict(decision='PASS_ALL_FOUR_INITIAL_PRIOR_CELLS' if passed else 'FAIL_INITIAL_PRIOR_COMPARISON',
                cells=cells, R1_diagnostic_only=True, candidate_selection=False)


def _require_candidate_weights(saved, *, expected, model, torch):
    """Validate the actual decoded private artifact before same-instance loading."""
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    _need(isinstance(saved, dict) and set(saved) == set(expected)
          and saved['schema_version'] == SCORE_WEIGHTS_SCHEMA
          and saved['canonical_resume_authority'] is False and saved['optimizer_state_included'] is False
          and saved['weights_sha256'] == old._tree_sha({k: v for k, v in saved.items() if k != 'weights_sha256'})
          and saved['weights_sha256'] == expected['weights_sha256'], 'FIXED_SCORE_ROUNDTRIP_SEAL')
    _need(model_sha(saved['model_state']) == saved['model_state_sha256'] == expected['model_state_sha256']
          and old._tree_sha(saved['buffer_state']) == saved['buffer_state_sha256'] == expected['buffer_state_sha256'],
          'FIXED_SCORE_ROUNDTRIP_DIGEST')
    old._tensor_map(saved['model_state'], model.state_dict(), 'FIXED_CANDIDATE_MODEL')
    old._tensor_map(saved['buffer_state'], dict(model.named_buffers()), 'FIXED_CANDIDATE_BUFFERS')
    for name, value in saved['model_state'].items():
        _need(torch.equal(value.cpu(), expected['model_state'][name].cpu()), 'FIXED_SCORE_ROUNDTRIP_TENSOR:' + name)
    for name, value in saved['buffer_state'].items():
        _need(torch.equal(value.cpu(), expected['buffer_state'][name].cpu()), 'FIXED_SCORE_ROUNDTRIP_BUFFER:' + name)
    model.load_state_dict(saved['model_state'], strict=True)
    with torch.no_grad():
        for name, value in model.named_buffers(): value.copy_(saved['buffer_state'][name])



def _run_fixed_score(plan_path, plan_sha256, out_dir, *, device):
    import copy
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts import market_state_native_F_private_campaign_v1 as private
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_endpoint_batch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts.run_market_state_native_F_private_window_v1 import _m5_import_score
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    started = time.monotonic()
    _need(device == 'cuda', 'FIXED_SCORE_GPU_ONLY')
    cap = require_guarded_cuda_producer_execution(); cap['lock'] = require_capped_lock_ancestry()
    _need(cap['cgroup_path'] == '/sys/fs/cgroup' + cap['lock']['scope']
          and cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
          and cap['swap'] == 512 * 1024**2, 'FIXED_SCORE_CAP')
    for key, value in (('MAX_POWER_LIMIT_W', '250'), ('MAX_POWER_DRAW_W', '170'), ('MAX_CORE_TEMP_C', '65'), ('MAX_MEMORY_TEMP_C', '80')):
        _need(os.environ.get('GX1_TRAINER_GPU_' + key) == value, 'FIXED_SCORE_HARDWARE:' + key)
    q = _inspect_fixed_score(plan_path, plan_sha256, out_dir, execute=True)
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'FIXED_SCORE_INIT_ORDER')
    publish = lambda name, value: _publish(name, value, root=SCORE_ROOT)
    claim = publish('CLAIM.json', dict(plan=q['plan_ref'], policy=q['policy_ref'], source_commit=q['head'],
        cap=cap, head_tensor_imports_authorized=4, optimizer_updates=0, test_data_used=False))
    counts = dict(model_constructions=0, checkpoint_loads=0, backbone_encodings=0, forward_returns=0,
        missing_score_encodings=0, fresh_baseline_forwards=0, cached_head_calls=0, reference_linear_calls=0,
        head_tensor_imports=0, candidate_writes=0, optimizer_constructions=0, optimizer_updates=0,
        backwards=0, checkpoint_writes=0, resume_state_load_calls=0, resume_state_load_returns=0,
        published_npz_bytes=0)
    provider = prepared = model = None
    arrays = {}; complete = np.zeros(SCORE_ROWS, bool); r1_raw = {}; artifacts = {}
    import_proof = {}; head_parity = {}; baseline_proof = {}; roundtrip = {}; scores = {}; timings = {}
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('Fixed native score 3600-second bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1., WALL_SECONDS - 60 - (time.monotonic() - started)))
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO), source_commit=q['head']))
        for ref in q['bound_refs']: campaign.require_binding(ref, label='fixed score bookend', verify_file=True)
        for name, digest in q['current_sources'].items():
            _need(campaign.file_sha256(REPO / name) == digest, 'FIXED_SCORE_SOURCE_BOOKEND:' + name)
    def save_arrays(name, values):
        stream = io.BytesIO(); np.savez(stream, **values)
        counts['published_npz_bytes'] += stream.tell()
        _need(counts['published_npz_bytes'] <= SCORE_BOUNDS['max_cache_bytes'], 'FIXED_SCORE_OUTPUT_BYTES')
        _publish_bytes(SCORE_ROOT / name, stream.getvalue())
        return _ref(SCORE_ROOT / name)
    def exact(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    try:
        recipe, receipts, view = (q[k] for k in ('recipe', 'receipts', 'view'))
        refs, numerical, cuda = q['plan']['inputs'], recipe['numerical_recipe'], torch.device('cuda')
        trainer._set_deterministic(numerical['seed'], cuda, numerical['precision_policy'])
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _need(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'FIXED_SCORE_THREADS')
        provider, provider_plan, prepared, fit_rows, fit_issues = runner._prepared_fold1(receipts, view, np=np)
        endpoint, future = load_bound_five_endpoint_targets(), load_bound_future_displacement()
        axes = endpoint.mature_oof_axes(fold_id=1)
        later_rows, later_issues = axes['score_parent_rows'], axes['score_decision_ns']
        fold = q['origin']['original_recipe']['fold']
        section = slice(fold['oof_start_row'], fold['oof_end_row'])
        _need(len(later_rows) == SCORE_ROWS and np.array_equal(fit_rows, axes['fit_parent_rows'])
              and np.array_equal(fit_issues, axes['fit_decision_ns'])
              and np.array_equal(later_rows, prepared['axes']['parent_rows'][section])
              and np.array_equal(later_issues, prepared['axes']['decision_ns'][section]), 'FIXED_SCORE_AXES')
        reused, missing = _fixed_score_layout(); reused = np.asarray(reused, np.int64)
        layout = runner._ta_m5_capture_layout(FIT_ROWS)
        panels = {}
        for panel, name in (('TRAIN', 'fit_representations'), ('LATER_192', 'later_capture')):
            with np.load(refs[name]['path'], allow_pickle=False) as archive:
                panels[panel] = {key: archive[key].copy() for key in layout}
        _need(np.array_equal(panels['TRAIN']['parents'], fit_rows)
              and np.array_equal(panels['TRAIN']['issues'], fit_issues)
              and np.array_equal(panels['LATER_192']['parents'], later_rows[reused])
              and np.array_equal(panels['LATER_192']['issues'], later_issues[reused]), 'FIXED_SCORE_CACHED_AXES')
        with np.load(refs['readout_outputs']['path'], allow_pickle=False) as archive:
            locked = {key: archive[key].copy() for key in archive.files}
        coefficients = _fixed_coefficients(locked, q['lock'], np=np, torch=torch)
        for panel, data in panels.items():
            prefix = 'TRAIN' if panel == 'TRAIN' else 'LATER'
            _need(np.array_equal(data['parents'], locked[prefix+'_parents'])
                  and np.array_equal(data['issues'], locked[prefix+'_issues']), 'FIXED_SCORE_READOUT_AXIS')
            for task in ('state', 'future'):
                allowed = data[task+'_allowed'][:, 0]
                known = data[task+'_eligible'][:, 0] & data[task+'_mature'][:, 0] & np.isin(allowed, (1, 4))
                _need(np.array_equal(known, locked[prefix+'_'+task+'_known'])
                      and np.array_equal((allowed == 4).astype(np.float64), locked[prefix+'_'+task+'_y']),
                      'FIXED_SCORE_READOUT_MASKS')
        with np.load(refs['initial_score_cache']['path'], allow_pickle=False) as archive:
            initial_axes = {k: archive[k].copy() for k in ('parents', 'issues')}
            initial_truth = {task: {key: archive['native_'+str(index)+'_'+key].copy()
                for key in ('allowed', 'eligible', 'mature')} for index, task in enumerate(('state', 'future'))}
        _need(np.array_equal(initial_axes['parents'], later_rows) and np.array_equal(initial_axes['issues'], later_issues),
              'FIXED_SCORE_INITIAL_FULL_AXIS')
        with np.load(refs['original_imported_outputs']['path'], allow_pickle=False) as archive:
            initial_logits = {task: archive['score_'+task+'_original_imported'].copy() for task in ('state', 'future')}
            for task in ('state', 'future'):
                truth = initial_truth[task]
                known = truth['eligible'][:, 0] & truth['mature'][:, 0] & np.isin(truth['allowed'][:, 0], (1, 4))
                _need(np.array_equal(known, archive['score_'+task+'_known'])
                      and np.array_equal((truth['allowed'][:, 0] == 4).astype(np.float64), archive['score_'+task+'_y']),
                      'FIXED_SCORE_INITIAL_FULL_MASKS')
        counts['model_constructions'] += 1
        backbone = views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,
            recipe=q['origin']['original_recipe'], verify_files=True, ta_sf=True)
        _need(model.research_objective['contract_sha256'] == recipe['expected_model_contract_sha256'], 'FIXED_SCORE_MODEL_CONTRACT')
        order, _ = build_reverse_order(fit_rows, recipe['batch_indices'])
        counts['checkpoint_loads'] += 1
        saved = common._load_state(refs['reversed_evidence'], torch=torch,
            maximum=recipe['max_checkpoint_bytes'], counters=counts)
        source_identity = _load_reversed_weights(model, saved, q, torch.from_numpy(order.copy()), torch=torch)
        del saved
        model.to(cuda)
        heads = {'state': model.state_model.m5_side_head, 'future': model.future_m5_side_head}
        old_heads = {task: (head.weight.detach().clone(), head.bias.detach().clone()) for task, head in heads.items()}
        def signature():
            return dict(model=model_sha(model.state_dict()), buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
                modes={n: bool(m.training) for n, m in model.named_modules()},
                trainability={n: p.requires_grad for n, p in model.named_parameters()})
        original_signature = signature()
        _need(original_signature['model'] == REVERSED_MODEL and not any(original_signature['trainability'].values()),
              'FIXED_SCORE_FROZEN_SOURCE')
        atol, rtol = private.TA_M5_PARITY_ATOL, private.TA_M5_PARITY_RTOL
        cached = {}
        def cached_heads(label):
            head_parity[label] = {}
            for panel, data in panels.items():
                prefix = 'TRAIN' if panel == 'TRAIN' else 'LATER'
                head_parity[label][panel] = {}
                for task, head in heads.items():
                    pieces = []
                    with torch.inference_mode():
                        for left in range(0, len(data['parents']), 16):
                            x = torch.from_numpy(data[task+'_representation'][left:left+16]).to(cuda)
                            pieces.append(head(x).flatten().cpu().numpy().copy()); counts['cached_head_calls'] += 1
                    z = np.concatenate(pieces)
                    expected = (data[task+'_activity_side_logits'][:, 0, 1] if label == 'R1'
                                else locked[prefix+'_'+task+'_readout_fp32'])
                    known, y = locked[prefix+'_'+task+'_known'], locked[prefix+'_'+task+'_y']
                    got_score, reference_score = _m5_import_score(z, known, y), _m5_import_score(expected, known, y)
                    p = lambda values: np.exp(-np.logaddexp(0., -values.astype(np.float64)))
                    head_parity[label][panel][task] = dict(
                        raw_allclose=bool(np.allclose(z, expected, atol=atol, rtol=rtol)),
                        max_abs_raw_error=float(np.max(np.abs(z.astype(np.float64)-expected.astype(np.float64)))),
                        max_abs_probability_error=float(np.max(np.abs(p(z)-p(expected)))),
                        score_delta={k: got_score[k]-reference_score[k] for k in ('brier', 'log', 'mean_p')},
                        observed_score=got_score, reference_score=reference_score)
                    cached[label+'_'+panel+'_'+task] = z
        cached_heads('R1')
        _need(signature() == original_signature, 'FIXED_SCORE_SOURCE_HEAD_EVAL_MUTATED')
        import_proof = state.import_equal_weight_m5_side_heads(model, coefficients,
            expected_model_state_sha256=REVERSED_MODEL, coefficient_artifact=refs['readout_outputs'],
            coefficient_recipe=refs['readout_plan'])
        counts['head_tensor_imports'] = 4
        candidate_signature = signature()
        _need(candidate_signature['model'] != REVERSED_MODEL
              and all(candidate_signature[k] == original_signature[k] for k in ('buffers', 'rng', 'modes', 'trainability')),
              'FIXED_SCORE_IMPORT_STATE')
        cached_heads('CANDIDATE')
        _need(signature() == candidate_signature, 'FIXED_SCORE_CANDIDATE_HEAD_EVAL_MUTATED')
        artifacts['cached_heads'] = save_arrays('CACHED_HEAD_OUTPUTS.npz', cached)
        _need(all(v['raw_allclose'] for phase in head_parity.values() for panel in phase.values() for v in panel.values()),
              'FIXED_SCORE_ALL_CACHED_HEAD_PARITY')
        scores = {}
        for panel in panels:
            prefix = 'TRAIN' if panel == 'TRAIN' else 'LATER'
            scores[panel] = {}
            for task in heads:
                known, y = locked[prefix+'_'+task+'_known'], locked[prefix+'_'+task+'_y']
                values = dict(CANDIDATE=cached['CANDIDATE_'+panel+'_'+task],
                    R1=locked[prefix+'_'+task+'_REVERSED'], INITIAL=locked[prefix+'_'+task+'_INITIAL'],
                    PRIOR=locked[prefix+'_'+task+'_PRIOR'])
                scores[panel][task] = {name: _m5_import_score(z, known, y) for name, z in values.items()}
        def actual_forward(left, right, baseline=False):
            raw, parents, clocks = runner._feature_batch(prepared, view, receipts,
                later_rows, later_issues, left, right, np=np)
            features = {key: torch.from_numpy(value).to(cuda) for key, value in raw.items()}
            _need(counts['backbone_encodings'] < 1100, 'FIXED_SCORE_FORWARD_CAP')
            counts['backbone_encodings'] += 1
            counts['fresh_baseline_forwards' if baseline else 'missing_score_encodings'] += 1
            report = evaluate_endpoint_batch(model, features, endpoint, future, parents, clocks,
                fold_id=1, partition='score', future_weight=1., frozen_priors=q['priors'], capture_m5_readout=True)
            counts['forward_returns'] += 1
            return report['m5_readout']
        first = actual_forward(0, 16, baseline=True)
        old_first = {key: value[:16] for key, value in panels['LATER_192'].items()}
        for key in ('parents', 'issues', 'query_native', 'state_representation', 'future_representation'):
            _need(exact(first[key], old_first[key]), 'FIXED_SCORE_FRESH_BASELINE_REP:' + key)
        for task in heads:
            name = task+'_activity_side_logits'
            _need(exact(first[name][..., 0], old_first[name][..., 0])
                  and exact(first[name][:, 1:], old_first[name][:, 1:])
                  and np.allclose(first[name][:, 0, 1], cached['CANDIDATE_LATER_192_'+task][:16], atol=atol, rtol=rtol),
                  'FIXED_SCORE_FRESH_BASELINE_OUTPUT:' + task)
            for key in ('allowed', 'eligible', 'mature'):
                _need(exact(first[task+'_'+key], old_first[task+'_'+key]), 'FIXED_SCORE_BASELINE_TRUTH')
        _need(signature() == candidate_signature, 'FIXED_SCORE_BASELINE_MUTATED')
        artifacts['fresh_baseline'] = save_arrays('FRESH_CANDIDATE_BASELINE.npz', first)
        weights = dict(schema_version=SCORE_WEIGHTS_SCHEMA, plan=q['plan_ref'], source_commit=q['head'],
            original_capsule=refs['reversed_evidence'], readout_lock=refs['readout_lock'],
            readout_outputs=refs['readout_outputs'], canonical_resume_authority=False,
            optimizer_state_included=False, model_state=copy.deepcopy(model.state_dict()),
            buffer_state=copy.deepcopy(dict(model.named_buffers())),
            model_state_sha256=candidate_signature['model'], buffer_state_sha256=candidate_signature['buffers'],
            feature_view_identity=copy.deepcopy(model.state_model.feature_view_identity),
            research_objective=copy.deepcopy(model.research_objective),
            module_training=candidate_signature['modes'], trainability=candidate_signature['trainability'],
            test_data_used=False)
        weights['weights_sha256'] = old._tree_sha(weights)
        stream = io.BytesIO(); torch.save(weights, stream)
        _need(0 < stream.tell() <= SCORE_BOUNDS['max_candidate_bytes'], 'FIXED_SCORE_CANDIDATE_BYTES')
        _publish_bytes(SCORE_ROOT/'CANDIDATE_WEIGHTS.pt', stream.getvalue()); counts['candidate_writes'] += 1
        artifacts['candidate_weights'] = _ref(SCORE_ROOT/'CANDIDATE_WEIGHTS.pt')
        counts['checkpoint_loads'] += 1
        decoded = common._load_state(artifacts['candidate_weights'], torch=torch,
            maximum=SCORE_BOUNDS['max_candidate_bytes'], counters=counts)
        _require_candidate_weights(decoded, expected=weights, model=model, torch=torch)
        state._require_model_binding(model); model.backbone.require_input_normalization_state()
        _need(signature() == candidate_signature, 'FIXED_SCORE_ROUNDTRIP_STATE')
        second = actual_forward(0, 16, baseline=True)
        _need(set(first) == set(second) and all(exact(first[k], second[k]) for k in first)
              and signature() == candidate_signature, 'FIXED_SCORE_ROUNDTRIP_FORWARD')
        roundtrip = dict(actual_file_decode=True, same_instance=True, all_model_tensors_exact=True,
            all_buffers_exact=True, raw_outputs_exact=True, representations_exact=True,
            RNG_modes_trainability_unchanged=True, fresh_constructor_tested=False,
            optimizer_restore=False, canonical_resume_authority=False, checkpoint_loads=counts['checkpoint_loads'])
        baseline_proof = dict(rows=16, artifact=artifacts['fresh_baseline'],
            candidate_model_state_sha256=candidate_signature['model'], representations_match_R1_exact=True,
            non_M5_outputs_match_R1_exact=True, candidate_cached_head_parity=True,
            full_population_encoder_baseline=False)
        del weights, decoded, stream, first, second
        arrays = {key: np.empty(shape, dtype) for key, (shape, dtype) in runner._ta_m5_capture_layout(SCORE_ROWS).items()}
        for key, value in panels['LATER_192'].items(): arrays[key][reused] = value
        for task in heads:
            arrays[task+'_activity_side_logits'][reused, 0, 1] = cached['CANDIDATE_LATER_192_'+task]
            r1_raw[task] = np.empty(SCORE_ROWS, np.float32)
            r1_raw[task][reused] = panels['LATER_192'][task+'_activity_side_logits'][:, 0, 1]
        complete[reused] = True
        timings['setup_import_and_roundtrip_seconds'] = time.monotonic()-started
        tick = time.monotonic()
        for n, index in enumerate(missing):
            left, right = 16*index, min(16*index+16, SCORE_ROWS)
            captured = actual_forward(left, right)
            runner._ta_m5_capture_batch(arrays, captured, left=left, right=right,
                parents=later_rows[left:right], issues=later_issues[left:right])
            with torch.inference_mode():
                for task in heads:
                    x = torch.from_numpy(captured[task+'_representation']).to(cuda)
                    if task == 'state': x = x[:, None, :]
                    r1_raw[task][left:right] = torch.nn.functional.linear(
                        x, old_heads[task][0], old_heads[task][1]).flatten().cpu().numpy()
                    counts['reference_linear_calls'] += 1
            complete[left:right] = True
            if (n+1) % 256 == 0:
                print(json.dumps(dict(status='PROGRESS', missing_batches=n+1, rows=int(complete.sum()),
                    elapsed_seconds=time.monotonic()-started)), flush=True)
        timings['missing_later_seconds'] = time.monotonic()-tick
        _need(complete.all() and signature() == candidate_signature and all(p.grad is None for p in model.parameters()),
              'FIXED_SCORE_FINAL_UNCHANGED')
        artifacts['full_later'] = save_arrays('FULL_LATER_READOUT.npz',
            {**arrays, **{task+'_R1_M5_raw': value for task, value in r1_raw.items()}})
        scores['LATER_FULL'] = {}
        for task in heads:
            _need(all(np.array_equal(arrays[task+'_'+key], initial_truth[task][key])
                      for key in ('allowed', 'eligible', 'mature')), 'FIXED_SCORE_FULL_MASKS')
            allowed = arrays[task+'_allowed'][:, 0]
            known = arrays[task+'_eligible'][:, 0] & arrays[task+'_mature'][:, 0] & np.isin(allowed, (1, 4))
            y = (allowed == 4).astype(np.float64)
            _need(int(known.sum()) == {'state': 6096, 'future': 7165}[task], 'FIXED_SCORE_FULL_COUNTS')
            values = dict(CANDIDATE=arrays[task+'_activity_side_logits'][:, 0, 1],
                R1=r1_raw[task], INITIAL=initial_logits[task],
                PRIOR=np.full(SCORE_ROWS, float(q['priors']['tasks'][task]['raw_logits'][0][1])))
            scores['LATER_FULL'][task] = {name: _m5_import_score(z, known, y) for name, z in values.items()}
        decision = _fixed_score_decision(scores)
        ambiguous = []
        for panel, local in (('TRAIN', 'TRAIN'), ('LATER_FULL', 'LATER_192')):
            for task in heads:
                error = head_parity['CANDIDATE'][local][task]['score_delta']
                for reference in ('INITIAL', 'PRIOR'):
                    cell = decision['cells'][panel][task][reference]
                    if cell['passes'] and any(abs(cell[key+'_delta']) <= abs(error[key]) for key in ('brier', 'log')):
                        ambiguous.append(dict(panel=panel, task=task, reference=reference))
        decision['observed_rounding_ambiguity'] = ambiguous
        if ambiguous and decision['decision'].startswith('PASS'):
            decision['decision'] = 'NUMERICALLY_UNCLEAR'
        expected = dict(model_constructions=1, checkpoint_loads=2, backbone_encodings=1100, forward_returns=1100,
            missing_score_encodings=1098, fresh_baseline_forwards=2, cached_head_calls=7444,
            reference_linear_calls=2196, head_tensor_imports=4, candidate_writes=1,
            optimizer_constructions=0, optimizer_updates=0, backwards=0, checkpoint_writes=0,
            resume_state_load_calls=2, resume_state_load_returns=2)
        _need(all(counts[k] == v for k, v in expected.items()), 'FIXED_SCORE_EXACT_COUNTS')
        base._release_pages(provider, prepared); prepared = None
        provider.validate_metadata(provider_plan); bookend()
        result = publish('RESULT.json', dict(status='COMPLETE', plan=q['plan_ref'], claim=claim,
            source_commit=q['head'], artifacts=artifacts, counts=counts, source_identity=source_identity,
            candidate_model_state_sha256=candidate_signature['model'], import_proof=import_proof,
            head_parity=head_parity, fresh_candidate_baseline=baseline_proof, roundtrip=roundtrip,
            scores=scores, comparison=decision, later_reused_rows=192, later_new_rows=17561,
            later_total_rows=SCORE_ROWS, reused_positions=reused.tolist(),
            candidate_later_provenance='192 actual GPU-head outputs on retained R1 inputs plus17561 actual full native outputs',
            R1_later_provenance='192 retained native outputs plus17561 reconstructed affine outputs on the same newly captured inputs',
            prior_ORIGINAL_FINAL_full_later_context=q['readout']['prior_ORIGINAL_FINAL_full_later_context'],
            negative_control_recomputed=False, source_unchanged=True, evaluation_state_unchanged=True,
            head_tensor_imports=4, optimizer_updates=0, training_enabled=False, test_data_used=False,
            canonical_resume_authority=False, candidate_selection=False, promotion_authorized=False,
            limits='Fixed heads only; same-instance weights-only roundtrip, no fresh constructor or optimizer resume. Later TRAIN reused development. R1 and CPU/GPU differences are diagnostic, not a formal global floating-point error bound.',
            timings={**timings, 'elapsed_seconds': time.monotonic()-started}))
        terminal = publish('TERMINAL.json', dict(status='COMPLETE', exit_code=0, plan=q['plan_ref'],
            result=result, artifacts=artifacts, source_unchanged=True,
            no_relaunch_under_this_plan=True, test_data_used=False))
        return dict(result=result, terminal=terminal)
    except BaseException as exc:
        preservation_error = None
        if arrays and complete.any() and 'full_later' not in artifacts:
            try:
                positions = np.flatnonzero(complete)
                artifacts['partial_later'] = save_arrays('PARTIAL_LATER_READOUT.npz',
                    {'score_positions': positions, **{k: v[positions] for k, v in arrays.items()},
                     **{task+'_R1_M5_raw': v[positions] for task, v in r1_raw.items()}})
            except Exception as secondary: preservation_error = repr(secondary)
        source_unchanged = False
        try: bookend(); source_unchanged = True
        except Exception as secondary: preservation_error = (preservation_error or '')+';bookend:'+repr(secondary)
        result = publish('RESULT.json', dict(status='FAILED_CONSUMED', plan=q['plan_ref'], claim=claim,
            artifacts=artifacts, counts=counts, rows=int(complete.sum()), import_proof=import_proof,
            head_parity=head_parity, fresh_candidate_baseline=baseline_proof, roundtrip=roundtrip,
            error=type(exc).__name__+':'+str(exc), preservation_error=preservation_error,
            source_unchanged=source_unchanged, test_data_used=False, canonical_resume_authority=False))
        publish('TERMINAL.json', dict(status='FAILED', exit_code=1, plan=q['plan_ref'], result=result,
            source_unchanged=source_unchanged, no_relaunch_under_this_plan=True, test_data_used=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.); signal.signal(signal.SIGALRM, previous_alarm)
        if prepared is not None:
            base._release_pages(provider, prepared)


# Locked inner-TRAIN confidence correction; all full representations already exist.
SHRINK_STAGE = 'market_state_ta_m5_locked_shrink_native_20261111_001'
SHRINK_ROOT = Path('/home/andre2/GX1_RUNS') / SHRINK_STAGE.upper()
SHRINK_SCHEMA = 'gx1_ta_m5_locked_shrink_native_plan_v1'
_SHRINK = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_INNER_PRIOR_SHRINK_20261111_002')
SHRINK_REFS = {
    **SCORE_REFS,
    'native_plan': {'path': str(SCORE_ROOT/'PLAN.json'), 'sha256': '279cbf0bd77cfbe655c8c4414889f1aa9c62dfb25485ccf428df714d656ae6df'},
    'native_result': {'path': str(SCORE_ROOT/'RESULT.json'), 'sha256': 'e5be50b5bb4eead295051bf8df6f694ecfc6ae46c33386e782e439d2a6007a8f'},
    'native_review': {'path': str(SCORE_ROOT/'NUMERIC_REVIEW.json'), 'sha256': '42135ad7288b66053abd76b107209670ef93eab17da4843493ec34fdb72f9d90'},
    'native_later': {'path': str(SCORE_ROOT/'FULL_LATER_READOUT.npz'), 'sha256': '2e9bc91f8cff9528d0a8821ef4ea0d565d6f921cc092d76112d594cadd3c3938'},
    'shrink_plan': {'path': str(_SHRINK/'PLAN.json'), 'sha256': '476eb63cba3915110c20a8ff935dc754243ea01e66bdd7b18b098ee7cdaaedce'},
    'shrink_result': {'path': str(_SHRINK/'RESULT.json'), 'sha256': '8e127f550d2fce718120c01fa9c6bafe6029742361fd36789ea4c67fc93ff5d0'},
    'shrink_review': {'path': str(_SHRINK/'NUMERIC_REVIEW.json'), 'sha256': '57cba25787986bcbe6b95bd6605c818d93f2e674f995ad2858dac9cee6ffa903'},
    'shrink_lock': {'path': str(_SHRINK/'SLOPE_LOCK.json'), 'sha256': 'b4bd90037482bcb72e2731c4e3a858eaa478bd14103de85d15995673f71913e7'},
    'shrink_outputs': {'path': str(_SHRINK/'PRIVATE_OUTPUTS.npz'), 'sha256': '9b63da8478b411048ce7f4bd0780a36b89b061357fb5866b1ce6055b65dff57b'},
}
SHRINK_BOUNDS = dict(fit_rows=29574, score_rows=17753, batch_size=16, decoder_width=128,
    max_backbone_encodings=2, max_model_constructions=1, max_checkpoint_loads=2,
    max_candidate_writes=1, max_candidate_bytes=1073741824, max_cache_bytes=16777216,
    max_cached_head_calls=5918, head_tensor_imports=4, max_optimizer_constructions=0,
    max_backwards=0, max_optimizer_updates=0, max_checkpoint_writes=0,
    scalar_fits=0, device='cuda', wall_seconds=900, training_enabled=False, test_data_used=False)


def _shrink_plan(plan, plan_path, out_dir):
    fields = {'schema_version', 'stage', 'source_repo', 'source_base_commit', 'worker',
              'dispatch', 'cap_runner', 'runtime_source_override', 'inputs'} | set(SHRINK_BOUNDS)
    _need(set(plan) == fields and plan['schema_version'] == SHRINK_SCHEMA
          and plan['stage'] == SHRINK_STAGE and plan['source_repo'] == str(REPO)
          and Path(plan_path) == SHRINK_ROOT/'PLAN.json' and Path(out_dir) == SHRINK_ROOT
          and plan['inputs'] == SHRINK_REFS
          and all(type(plan[k]) is type(v) and plan[k] == v for k, v in SHRINK_BOUNDS.items()),
          'SHRINK_EXACT_PLAN')
    _need(isinstance(plan['source_base_commit'], str)
          and campaign._COMMIT.fullmatch(plan['source_base_commit']), 'SHRINK_SOURCE_BASE')
    return plan


def _shrink_coefficients(original_coefficients, slope_lock, priors, *, torch):
    result = {}
    expected = {prefix+suffix for prefix in ('state_model.m5_side_head', 'future_m5_side_head')
                for suffix in ('.weight', '.bias')}
    _need(set(original_coefficients) == expected, 'SHRINK_COEFFICIENT_KEYS')
    for task, prefix in (('state', 'state_model.m5_side_head'), ('future', 'future_m5_side_head')):
        a = slope_lock['slopes'][task]['slope']
        b = priors['tasks'][task]['raw_logits'][0][1]
        _need(type(a) is float and 0. < a <= 1. and type(b) in (int, float)
              and float('-inf') < b < float('inf'), 'SHRINK_SLOPE_PRIOR:' + task)
        for suffix, shape in (('.weight', (1, 128)), ('.bias', (1,))):
            value = original_coefficients[prefix+suffix]
            _need(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                  and tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
                  'SHRINK_COEFFICIENT:' + task + suffix)
            x = value.detach().double()
            folded = a*x if suffix == '.weight' else b+a*(x-b)
            result[prefix+suffix] = folded.float()
    return result


def _inspect_shrink_score(plan_path, plan_sha256, out_dir, *, execute):
    pr = {'path': str(plan_path), 'sha256': plan_sha256}
    plan = _shrink_plan(_read(pr), plan_path, out_dir); refs = plan['inputs']; bound = [pr]
    for key, path in (('worker', WORKER), ('dispatch', REPO/DISPATCH), ('cap_runner', REPO/CAP_RUNNER)):
        _need(plan[key]['path'] == str(path), 'SHRINK_SOURCE_PATH:' + key)
        campaign.require_binding(plan[key], label=key, verify_file=True); bound.append(plan[key])
    for key, ref in refs.items():
        campaign.require_binding(ref, label='shrink input '+key, verify_file=True); bound.append(ref)
    q = _qualify_reversed_origin(plan, bound)
    native, native_plan, native_review, shrink, shrink_plan, shrink_review, lock, readout_lock = (
        _read(refs[k]) for k in ('native_result', 'native_plan', 'native_review', 'shrink_result',
            'shrink_plan', 'shrink_review', 'shrink_lock', 'readout_lock'))
    _need(native['status'] == 'COMPLETE' and native['source_unchanged'] and native['evaluation_state_unchanged']
          and native['plan'] == refs['native_plan'] and native['artifacts']['full_later'] == refs['native_later']
          and native_plan['inputs']['fit_representations'] == refs['fit_representations']
          and native_plan['inputs']['readout_outputs'] == refs['readout_outputs']
          and native_plan['inputs']['readout_lock'] == refs['readout_lock']
          and native_review['status'] == 'PASS_INDEPENDENT_RETAINED_LOGIT_ARITHMETIC'
          and native_review['result'] == refs['native_result'] and native_review['exact_full_later_axes_masks'],
          'SHRINK_NATIVE_CACHE_LINEAGE')
    _need(shrink['status'] == 'COMPLETE' and shrink['source_unchanged']
          and shrink['plan'] == refs['shrink_plan'] and shrink['slope_lock'] == refs['shrink_lock']
          and shrink['private_outputs'] == refs['shrink_outputs']
          and shrink['comparison']['decision'] == 'PASS_ALL_FOUR_INITIAL_PRIOR_CELLS'
          and shrink_review['status'] == 'PASS_INDEPENDENT_RETAINED_LOGIT_ARITHMETIC'
          and shrink_review['result'] == refs['shrink_result'] and shrink_review['primary_comparison_pass']
          and lock['plan'] == refs['shrink_plan'] and lock['slopes'] == shrink['slopes']
          and lock['later_arrays_decoded'] is False and lock['scalar_fits'] == shrink['counts']['scalar_fits'] == 2
          and shrink['encoder_oof'] is False and shrink_plan['inputs']['native_result'] == refs['native_result']
          and shrink_plan['inputs']['native_later'] == refs['native_later']
          and readout_lock['plan'] == refs['readout_plan'] and readout_lock['alpha'] == .01,
          'SHRINK_LOCKED_TRANSFORM')
    for task in ('state', 'future'):
        _need(type(lock['slopes'][task]['slope']) is float and 0 < lock['slopes'][task]['slope'] <= 1,
              'SHRINK_NONCOLLAPSED_SLOPE')
    _need(all(x['test_data_used'] is False for x in (native, native_review, shrink, shrink_review)), 'SHRINK_TEST')
    head = _git('rev-parse', 'HEAD')
    subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor',plan['source_base_commit'],head], check=True)
    policy_ref = _ref(REPO/'NEXT_RUN_POLICY.json'); policy = _read(policy_ref); bound.append(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
          'native_launch_authorized','full_epoch_training_allowed')), 'SHRINK_NO_TRAINING')
    if execute:
        gate = policy.get(SHRINK_STAGE, {})
        expected = dict(status='ADMITTED', plan=pr, max_invocations=1, remaining_invocations=1,
            claimed_invocations=0, completed_invocations=0, relaunch_allowed=False, execution_authorized=True,
            data_reads_authorized=True, checkpoint_reads_authorized=True, backbone_encodings_authorized=2,
            head_tensor_imports_authorized=4, candidate_weights_writes_authorized=1,
            backward_calls_authorized=0, optimizer_steps_authorized=0, training_authorized=False)
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in expected.items())
              and all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in SHRINK_BOUNDS.items()),
              'SHRINK_ADMISSION')
        _need(_git('branch','--show-current') == 'work/gx1-current', 'SHRINK_BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
        _need(not any((SHRINK_ROOT/n).exists() for n in ('CLAIM.json','RESULT.json','TERMINAL.json',
              'CANDIDATE_WEIGHTS.pt','FOLDED_COEFFICIENTS.npz')), 'SHRINK_CONSUMED')
    return dict(q, plan=plan, plan_ref=pr, head=head, policy_ref=policy_ref, bound_refs=bound,
                native=native, shrink=shrink, slope_lock=lock, readout_lock=readout_lock)


def _run_shrink_score(plan_path, plan_sha256, out_dir, *, device):
    import copy
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts import market_state_native_F_private_campaign_v1 as private
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_endpoint_batch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts.run_market_state_native_F_private_window_v1 import _m5_import_score
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    started = time.monotonic(); _need(device == 'cuda', 'SHRINK_CUDA_ONLY')
    cap = require_guarded_cuda_producer_execution(); cap['lock'] = require_capped_lock_ancestry()
    _need(cap['cgroup_path'] == '/sys/fs/cgroup'+cap['lock']['scope']
          and cap['memory_max'] == cap['memory_high'] == 12*1024**3 and cap['swap'] == 512*1024**2,
          'SHRINK_CAP')
    for key,value in (('MAX_POWER_LIMIT_W','250'),('MAX_POWER_DRAW_W','170'),('MAX_CORE_TEMP_C','65'),('MAX_MEMORY_TEMP_C','80')):
        _need(os.environ.get('GX1_TRAINER_GPU_'+key) == value, 'SHRINK_HARDWARE:'+key)
    q = _inspect_shrink_score(plan_path,plan_sha256,out_dir,execute=True)
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'SHRINK_INIT_ORDER')
    publish = lambda name,value: _publish(name,value,root=SHRINK_ROOT)
    claim = publish('CLAIM.json',dict(plan=q['plan_ref'],policy=q['policy_ref'],source_commit=q['head'],cap=cap,
                    head_tensor_imports_authorized=4,optimizer_updates=0,test_data_used=False))
    counts = dict(model_constructions=0,checkpoint_loads=0,backbone_encodings=0,forward_returns=0,
        cached_head_calls=0,head_tensor_imports=0,candidate_writes=0,optimizer_constructions=0,
        optimizer_updates=0,backwards=0,checkpoint_writes=0,resume_state_load_calls=0,
        resume_state_load_returns=0,published_npz_bytes=0,scalar_fits=0)
    provider = prepared = None; artifacts = {}; import_proof = {}; scores = {}; parity = {}; roundtrip = {}
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('Locked shrink native 900-second bound')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(1.,900-(time.monotonic()-started)))
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']:campaign.require_binding(ref,label='shrink bookend',verify_file=True)
        for name,digest in q['current_sources'].items():
            _need(campaign.file_sha256(REPO/name)==digest,'SHRINK_OWNER_BOOKEND:'+name)
    def save_arrays(name,values):
        buf=io.BytesIO();np.savez(buf,**values);counts['published_npz_bytes']+=buf.tell()
        _need(counts['published_npz_bytes']<=SHRINK_BOUNDS['max_cache_bytes'],'SHRINK_OUTPUT_BYTES')
        _publish_bytes(SHRINK_ROOT/name,buf.getvalue());return _ref(SHRINK_ROOT/name)
    def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    try:
        recipe,receipts,view=(q[k] for k in ('recipe','receipts','view'));refs=q['plan']['inputs']
        numerical=recipe['numerical_recipe'];cuda=torch.device('cuda')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        _need(torch.get_num_threads()==torch.get_num_interop_threads()==1,'SHRINK_THREADS')
        provider,provider_plan,prepared,fit_rows,fit_issues=runner._prepared_fold1(receipts,view,np=np)
        endpoint,future=load_bound_five_endpoint_targets(),load_bound_future_displacement()
        axes=endpoint.mature_oof_axes(fold_id=1);later_rows,later_issues=axes['score_parent_rows'],axes['score_decision_ns']
        _need(np.array_equal(fit_rows,axes['fit_parent_rows']) and np.array_equal(fit_issues,axes['fit_decision_ns'])
              and len(fit_rows)==FIT_ROWS and len(later_rows)==SCORE_ROWS,'SHRINK_AXES')
        panels={}
        for panel,name,n in (('TRAIN','fit_representations',FIT_ROWS),('LATER_FULL','native_later',SCORE_ROWS)):
            with np.load(refs[name]['path'],allow_pickle=False) as z:
                panels[panel]={k:z[k].copy() for k in runner._ta_m5_capture_layout(n)}
        with np.load(refs['shrink_outputs']['path'],allow_pickle=False) as z:
            ideal={k:z[k].copy() for k in z.files if k.startswith(('TRAIN_','LATER_FULL_'))}
        with np.load(refs['readout_outputs']['path'],allow_pickle=False) as z:
            original=_fixed_coefficients(z,q['readout_lock'],np=np,torch=torch)
        coefficients=_shrink_coefficients(original,q['slope_lock'],q['priors'],torch=torch)
        artifacts['folded_coefficients']=save_arrays('FOLDED_COEFFICIENTS.npz',
            {name:value.cpu().numpy().copy() for name,value in coefficients.items()})
        for panel,data in panels.items():
            rows,issues=(fit_rows,fit_issues) if panel=='TRAIN' else (later_rows,later_issues)
            for key,expected in (('parents',rows),('issues',issues)):
                _need(exact(data[key],expected) and exact(data[key],ideal[panel+'_'+key]),'SHRINK_ORDERED_AXIS')
            for task in ('state','future'):
                allowed=data[task+'_allowed'][:,0]
                k=data[task+'_eligible'][:,0]&data[task+'_mature'][:,0]&np.isin(allowed,(1,4))
                _need(exact(k,ideal[panel+'_'+task+'_known'])
                      and exact((allowed==4).astype(np.float64),ideal[panel+'_'+task+'_y']),'SHRINK_TRUTH')
                _need(np.all(ideal[panel+'_'+task+'_PRIOR']==q['priors']['tasks'][task]['raw_logits'][0][1]),'SHRINK_PRIOR')
        counts['model_constructions']+=1
        backbone=views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile'])
        model=NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,recipe=q['origin']['original_recipe'],verify_files=True,ta_sf=True)
        _need(model.research_objective['contract_sha256']==recipe['expected_model_contract_sha256'],'SHRINK_MODEL_CONTRACT')
        order,_=build_reverse_order(fit_rows,recipe['batch_indices']);counts['checkpoint_loads']+=1
        saved=common._load_state(refs['reversed_evidence'],torch=torch,maximum=recipe['max_checkpoint_bytes'],counters=counts)
        source_identity=_load_reversed_weights(model,saved,q,torch.from_numpy(order.copy()),torch=torch);del saved
        model.to(cuda)
        def signature():
            return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
                modes={n:bool(m.training) for n,m in model.named_modules()},
                trainability={n:p.requires_grad for n,p in model.named_parameters()})
        before=signature()
        _need(before['model']==REVERSED_MODEL and not any(before['trainability'].values()),'SHRINK_FROZEN_SOURCE')
        import_proof=state.import_equal_weight_m5_side_heads(model,coefficients,expected_model_state_sha256=REVERSED_MODEL,
            coefficient_artifact=artifacts['folded_coefficients'],coefficient_recipe=q['plan_ref'])
        counts['head_tensor_imports']=4;candidate=signature()
        _need(candidate['model']!=REVERSED_MODEL and all(candidate[k]==before[k] for k in ('buffers','rng','modes','trainability')),
              'SHRINK_IMPORT_STATE')
        heads={'state':model.state_model.m5_side_head,'future':model.future_m5_side_head};cached={}
        for panel,data in panels.items():
            scores[panel]={};parity[panel]={}
            for task,head in heads.items():
                pieces=[]
                with torch.inference_mode():
                    for left in range(0,len(data['parents']),16):
                        x=torch.from_numpy(data[task+'_representation'][left:left+16]).to(cuda)
                        pieces.append(head(x).flatten().cpu().numpy().copy());counts['cached_head_calls']+=1
                z=np.concatenate(pieces);prefix=panel+'_'+task+'_';k=ideal[prefix+'known'];y=ideal[prefix+'y']
                expected=ideal[prefix+'CANDIDATE'];p=lambda v:np.exp(-np.logaddexp(0.,-v.astype(np.float64)))
                values={'CANDIDATE':z,**{arm:ideal[prefix+arm] for arm in ('INITIAL','PRIOR','R1','W2')},'IDEAL':expected}
                scores[panel][task]={arm:_m5_import_score(v,k,y) for arm,v in values.items()}
                parity[panel][task]=dict(raw_allclose=bool(np.allclose(z,expected,atol=private.TA_M5_PARITY_ATOL,rtol=private.TA_M5_PARITY_RTOL)),
                    max_abs_raw_error=float(np.max(np.abs(z.astype(np.float64)-expected))),
                    max_abs_probability_error=float(np.max(np.abs(p(z)-p(expected)))),
                    score_delta={m:scores[panel][task]['CANDIDATE'][m]-scores[panel][task]['IDEAL'][m] for m in ('brier','log','mean_p')})
                _need(parity[panel][task]['raw_allclose'],'SHRINK_ALL_ROW_PARITY')
                cached[prefix+'CANDIDATE']=z
        _need(counts['cached_head_calls']==5918 and signature()==candidate,'SHRINK_HEAD_EVALUATION_STATE')
        artifacts['cached_heads']=save_arrays('CACHED_HEAD_OUTPUTS.npz',cached)
        def actual_forward():
            _need(counts['backbone_encodings']<2,'SHRINK_FORWARD_LIMIT')
            raw,parents,clocks=runner._feature_batch(prepared,view,receipts,later_rows,later_issues,0,16,np=np)
            features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()};counts['backbone_encodings']+=1
            report=evaluate_endpoint_batch(model,features,endpoint,future,parents,clocks,fold_id=1,partition='score',
                future_weight=1.,frozen_priors=q['priors'],capture_m5_readout=True)
            counts['forward_returns']+=1;return report['m5_readout']
        first=actual_forward();previous={k:v[:16] for k,v in panels['LATER_FULL'].items()}
        for key in ('parents','issues','query_native','state_representation','future_representation'):
            _need(exact(first[key],previous[key]),'SHRINK_FRESH_BASELINE:'+key)
        for task in heads:
            key=task+'_activity_side_logits'
            _need(exact(first[key][...,0],previous[key][...,0]) and exact(first[key][:,1:],previous[key][:,1:])
                and np.allclose(first[key][:,0,1],cached['LATER_FULL_'+task+'_CANDIDATE'][:16],
                    atol=private.TA_M5_PARITY_ATOL,rtol=private.TA_M5_PARITY_RTOL),'SHRINK_BASELINE_OUTPUT')
            for name in ('allowed','eligible','mature'):_need(exact(first[task+'_'+name],previous[task+'_'+name]),'SHRINK_BASELINE_TRUTH')
        _need(signature()==candidate,'SHRINK_BASELINE_STATE')
        artifacts['fresh_baseline']=save_arrays('FRESH_CANDIDATE_BASELINE.npz',first)
        weights=dict(schema_version=SCORE_WEIGHTS_SCHEMA,plan=q['plan_ref'],source_commit=q['head'],
            original_capsule=refs['reversed_evidence'],slope_lock=refs['shrink_lock'],folded_coefficients=artifacts['folded_coefficients'],
            canonical_resume_authority=False,optimizer_state_included=False,model_state=copy.deepcopy(model.state_dict()),
            buffer_state=copy.deepcopy(dict(model.named_buffers())),model_state_sha256=candidate['model'],buffer_state_sha256=candidate['buffers'],
            feature_view_identity=copy.deepcopy(model.state_model.feature_view_identity),research_objective=copy.deepcopy(model.research_objective),
            module_training=candidate['modes'],trainability=candidate['trainability'],test_data_used=False)
        weights['weights_sha256']=old._tree_sha(weights);stream=io.BytesIO();torch.save(weights,stream)
        _need(0<stream.tell()<=SHRINK_BOUNDS['max_candidate_bytes'],'SHRINK_WEIGHTS_BYTES')
        _publish_bytes(SHRINK_ROOT/'CANDIDATE_WEIGHTS.pt',stream.getvalue());counts['candidate_writes']+=1
        artifacts['candidate_weights']=_ref(SHRINK_ROOT/'CANDIDATE_WEIGHTS.pt');counts['checkpoint_loads']+=1
        decoded=common._load_state(artifacts['candidate_weights'],torch=torch,maximum=SHRINK_BOUNDS['max_candidate_bytes'],counters=counts)
        _require_candidate_weights(decoded,expected=weights,model=model,torch=torch)
        state._require_model_binding(model);model.backbone.require_input_normalization_state()
        _need(signature()==candidate,'SHRINK_ROUNDTRIP_STATE')
        second=actual_forward()
        _need(set(first)==set(second) and all(exact(first[k],second[k]) for k in first) and signature()==candidate,'SHRINK_ROUNDTRIP_OUTPUT')
        roundtrip=dict(actual_file_decode=True,same_instance=True,all_model_tensors_exact=True,all_buffers_exact=True,
            raw_outputs_exact=True,representations_exact=True,RNG_modes_trainability_unchanged=True,
            fresh_constructor_tested=False,optimizer_restore=False,canonical_resume_authority=False)
        decision=_fixed_score_decision(scores);ambiguous=[]
        for panel in scores:
            for task in heads:
                for reference in ('INITIAL','PRIOR'):
                    cell=decision['cells'][panel][task][reference]
                    if cell['passes'] and any(-cell[m+'_delta']-cell['margin']<=abs(parity[panel][task]['score_delta'][m]) for m in ('brier','log')):
                        ambiguous.append(dict(panel=panel,task=task,reference=reference))
        decision['all_row_rounding_boundary_ambiguity']=ambiguous
        if ambiguous and decision['decision'].startswith('PASS'):decision['decision']='NUMERICALLY_UNCLEAR'
        expected=dict(model_constructions=1,checkpoint_loads=2,backbone_encodings=2,forward_returns=2,
            cached_head_calls=5918,head_tensor_imports=4,candidate_writes=1,optimizer_constructions=0,
            optimizer_updates=0,backwards=0,checkpoint_writes=0,resume_state_load_calls=2,resume_state_load_returns=2,scalar_fits=0)
        _need(all(counts[k]==v for k,v in expected.items()),'SHRINK_EXACT_COUNTS')
        base._release_pages(provider,prepared);prepared=None;provider.validate_metadata(provider_plan);bookend()
        result=publish('RESULT.json',dict(status='COMPLETE',plan=q['plan_ref'],claim=claim,source_commit=q['head'],counts=counts,
            artifacts=artifacts,source_identity=source_identity,import_proof=import_proof,candidate_model_state_sha256=candidate['model'],
            scores=scores,comparison=decision,all_row_numeric_parity=parity,roundtrip=roundtrip,
            fresh_candidate_baseline=dict(rows=16,artifact=artifacts['fresh_baseline'],representations_and_non_M5_outputs_unchanged=True),
            source_unchanged=True,evaluation_state_unchanged=True,scalar_fits=0,training_enabled=False,test_data_used=False,
            canonical_resume_authority=False,promotion_authorized=False,elapsed_seconds=time.monotonic()-started,
            limits='Actual native CUDA heads on every qualified cached representation plus two full row0 forwards. No all-row fresh encoder, fresh constructor, optimizer resume, untouched generalization or economic-edge claim.'))
        terminal=publish('TERMINAL.json',dict(status='COMPLETE',exit_code=0,plan=q['plan_ref'],result=result,
            source_unchanged=True,no_relaunch_under_this_plan=True,test_data_used=False))
        return dict(result=result,terminal=terminal)
    except BaseException as exc:
        changed=None
        try:bookend()
        except Exception as other:changed=repr(other)
        result=publish('RESULT.json',dict(status='FAILED_CONSUMED',plan=q['plan_ref'],claim=claim,counts=counts,artifacts=artifacts,
            import_proof=import_proof,scores=scores,all_row_numeric_parity=parity,roundtrip=roundtrip,
            error=type(exc).__name__+':'+str(exc),source_unchanged=changed is None,bookend_error=changed,test_data_used=False))
        publish('TERMINAL.json',dict(status='FAILED',exit_code=1,plan=q['plan_ref'],result=result,
            source_unchanged=changed is None,no_relaunch_under_this_plan=True,test_data_used=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);signal.signal(signal.SIGALRM,previous_alarm)
        if prepared is not None:base._release_pages(provider,prepared)


# Passive observation of the already-qualified candidate; no fit or model change.
FUSION_STAGE = 'market_state_ta_fusion_observation_20261111_001'
FUSION_ROOT = Path('/home/andre2/GX1_RUNS') / FUSION_STAGE.upper()
FUSION_SCHEMA = 'gx1_ta_fusion_observation_plan_v1'
FUSION_REFS = {
    **REVERSED_REFS,
    'candidate_plan': {'path': str(SHRINK_ROOT/'PLAN.json'), 'sha256': 'c6ca851b1ed048c7d3f9604687a7c063c071cbda897d6dac2b1de57ff2c000c5'},
    'candidate_result': {'path': str(SHRINK_ROOT/'RESULT.json'), 'sha256': 'd8a4ef4de35df2a0c78bc5efdd45c57d901e4b1ff98ef16fe11187076bdfa612'},
    'candidate_review': {'path': str(SHRINK_ROOT/'NUMERIC_REVIEW.json'), 'sha256': '0464e73f00f06426269f74f01a09d0b6dd276f675a053478062a145dbd2bc73a'},
    'candidate_terminal': {'path': str(SHRINK_ROOT/'TERMINAL.json'), 'sha256': 'ccd02a8a1e79ec4d4dfb11629e2e3dd54499a638bbb6d9cb6a088e2a7312167c'},
    'folded_coefficients': {'path': str(SHRINK_ROOT/'FOLDED_COEFFICIENTS.npz'), 'sha256': 'd3e7461f7907b74978c1037e07e26c27c9a5a44b85e83f213c8e7524496fce3d'},
    'candidate_baseline': {'path': str(SHRINK_ROOT/'FRESH_CANDIDATE_BASELINE.npz'), 'sha256': '8e00069eb8afea6385f8f31180f7a8299435da63723d16cd4a6593d517c353d4'},
}
FUSION_BOUNDS = dict(fit_rows=29574, score_rows=17753, batch_size=16, decoder_width=128,
    batches_per_partition=16, sampled_rows_per_partition=256,
    batch_selection='uniform_complete_batches_floor_i_times_last_over_15',
    max_backbone_encodings=33, max_model_constructions=1, max_checkpoint_loads=1,
    head_tensor_imports=4, max_candidate_writes=0, max_optimizer_constructions=0,
    max_backwards=0, max_optimizer_updates=0, max_checkpoint_writes=0, scalar_fits=0,
    max_cache_bytes=16777216, device='cuda', wall_seconds=900,
    candidate_model_state_sha256='c171736d2de79396dd86a79ad895f8bf2226e5ad8d996abf1dd7f52055e4a467',
    training_enabled=False, test_data_used=False)


FUSION_INITIAL_STAGE = 'market_state_ta_fusion_initial_observation_20261111_001'
FUSION_INITIAL_ROOT = Path('/home/andre2/GX1_RUNS') / FUSION_INITIAL_STAGE.upper()
FUSION_INITIAL_BOUNDS = dict(FUSION_BOUNDS, head_tensor_imports=0,
    candidate_model_state_sha256='ae772fd755cbfa07b1b7bcca8bccd2c8a8889806989a289ddb8267972c281b37')
_INITIAL_ORIGIN = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_FOLD1_FIT_20261109_002')
_INITIAL_BASELINE = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_CENTERED_HEAD_COORDINATES_20261110_001')
FUSION_INITIAL_REFS = {
    **REVERSED_REFS,
    'initial_checkpoint': {'path': str(_INITIAL_ORIGIN/'RUNTIME/prefix-fit/states/INITIAL_FOLD1_STATE.pt'), 'sha256': '80fc8029200c8076bc2faa38014bfd24c75ee71632fb1f45138249ccaca8d484'},
    'initial_baseline': {'path': str(_INITIAL_BASELINE/'INITIAL_LATER_READOUT.npz'), 'sha256': 'ad895e24f4f321b47c7077243b57e769a84eea32b043868271714604aba36b5b'},
    'initial_measurement': {'path': str(_INITIAL_BASELINE/'INITIAL_LATER.json'), 'sha256': 'e1f5a1694bf571a55bc8daab8570abc60597f3d58e7e9421d4ab283f4310b6d9'},
    'trained_result': {'path': str(FUSION_ROOT/'RESULT.json'), 'sha256': '071c2976e9bf26ec88207e3cecb44eda712fda6a2f908717d746ede7f5dcdea5'},
    'trained_terminal': {'path': str(FUSION_ROOT/'TERMINAL.json'), 'sha256': '915bee4127ae296c3fdb10437d1e4406d19a56bd76c5172ba14dba7903191e1f'},
    'trained_review': {'path': str(FUSION_ROOT/'NUMERIC_REVIEW.json'), 'sha256': 'dacc9eaa78a6f2dc8b0e1b1ab5f1140d82bc8415b8d62a705e5cf3d52834c0e2'},
    'trained_train': {'path': str(FUSION_ROOT/'TRAIN_FUSION.npz'), 'sha256': 'fd2ea2971343971179fb2a6d1b86253c57bb8e7fd06c9b7aa53e633e369469e8'},
    'trained_later': {'path': str(FUSION_ROOT/'LATER_FUSION.npz'), 'sha256': '43f423c297d80b4a09467e67e640245ccddb84f8b7c990ce5a6401a3b3184987'},
}
FUSION_MATCH_KEYS = ('parents','issues','query_native', *(
    task+'_'+key for task in ('state','future') for key in ('allowed','eligible','mature')))


MTF_RETAINED_STAGE = 'market_state_ta_mtf_retained_intervention_20261111_001'
MTF_RETAINED_ROOT = Path('/home/andre2/GX1_RUNS') / MTF_RETAINED_STAGE.upper()
MTF_RETAINED_REFS = {**FUSION_REFS, **{k:v for k,v in FUSION_INITIAL_REFS.items() if k.startswith('trained_')}}
MTF_RETAINED_BOUNDS = dict(FUSION_BOUNDS, max_backbone_encodings=0,
    max_joint_calls=64, max_state_decoder_calls=64, max_future_decoder_calls=64,
    max_query_projection_calls=128, max_native_head_calls=128, max_m5_head_calls=128,
    mean_vector_estimates=2, predictor_fits=0,
    intervention='replace_mtf_row_variation_with_fixed_train_panel_means_v1',
    mean_arithmetic='float64_after_minus_base_and_means_cast_once_float32',
    baseline_parity='bitexact_joint_hidden_m5_representations_all_native_logits')


TEMPORAL_STAGE = 'market_state_ta_temporal_boundary_observation_20261111_001'
TEMPORAL_ROOT = Path('/home/andre2/GX1_RUNS') / TEMPORAL_STAGE.upper()
TEMPORAL_REFS = dict(MTF_RETAINED_REFS)
TEMPORAL_BOUNDS = dict(FUSION_BOUNDS,max_cache_bytes=64*1024**2,
    batch_equal_length_mtf_eval=False,temporal_lengths=[16,64,96,96,252],
    max_temporal_family_encoder_calls=1280,max_temporal_family_axis_calls=32,
    max_temporal_timeframe_axis_calls=32,max_temporal_cross_tf_calls=32,
    temporal_capture_layout='fp32_mean_last_and_axis_vectors_fp64_window_moments_v1')


MIXING_GRADIENT_STAGE = 'market_state_ta_mixing_gradient_observation_20261111_001'
MIXING_GRADIENT_ROOT = Path('/home/andre2/GX1_RUNS') / MIXING_GRADIENT_STAGE.upper()
MIXING_GRADIENT_HELPER = REPO/'gx1/audit/ta_direction_mixing_gradient_v1.py'
MIXING_GRADIENT_REFS = dict(FUSION_REFS)
MIXING_GRADIENT_BOUNDS = dict(FUSION_BOUNDS,max_backbone_encodings=17,max_backwards=64,
    feature_partition='fit',max_archive_decodes=1,max_gradient_forwards=16,max_autograd_grad_calls=64,
    parameter_groups=['family_axis_attn','timeframe_axis_attn','cross_tf_attn'],parameter_tensors=36,
    max_retained_gradient_bytes=64*1024**2,max_published_json_bytes=1024**2,
    batch_equal_length_mtf_eval=False,closure_relative_tolerance=1e-4,
    runtime_mode='eval_fixed_sampled_training_queries',gradient_fit_authorized=False)


def _fusion_plan(plan, plan_path, out_dir, *, initial=False, retained=False, temporal=False, gradient=False):
    _need(all(type(v) is bool for v in (initial,retained,temporal,gradient)) and sum((initial,retained,temporal,gradient)) <= 1, 'FUSION_MODE')
    if gradient:
        stage, root, refs, bounds = MIXING_GRADIENT_STAGE, MIXING_GRADIENT_ROOT, MIXING_GRADIENT_REFS, MIXING_GRADIENT_BOUNDS
    elif temporal:
        stage, root, refs, bounds = TEMPORAL_STAGE, TEMPORAL_ROOT, TEMPORAL_REFS, TEMPORAL_BOUNDS
    elif retained:
        stage, root, refs, bounds = MTF_RETAINED_STAGE, MTF_RETAINED_ROOT, MTF_RETAINED_REFS, MTF_RETAINED_BOUNDS
    elif initial:
        stage, root, refs, bounds = FUSION_INITIAL_STAGE, FUSION_INITIAL_ROOT, FUSION_INITIAL_REFS, FUSION_INITIAL_BOUNDS
    else:
        stage, root, refs, bounds = FUSION_STAGE, FUSION_ROOT, FUSION_REFS, FUSION_BOUNDS
    fields = {'schema_version', 'stage', 'source_repo', 'source_base_commit', 'worker',
              'dispatch', 'cap_runner', 'runtime_source_override', 'inputs'} | set(bounds) | ({'gradient_helper'} if gradient else set())
    _need(isinstance(plan, dict) and set(plan) == fields and plan['schema_version'] == FUSION_SCHEMA
          and plan['stage'] == stage and plan['source_repo'] == str(REPO)
          and Path(plan_path) == root/'PLAN.json' and Path(out_dir) == root
          and plan['inputs'] == refs
          and all(type(plan[k]) is type(v) and plan[k] == v for k, v in bounds.items())
          and (not temporal or all(type(v) is int for v in plan['temporal_lengths'])),
          'FUSION_EXACT_PLAN')
    _need(isinstance(plan['source_base_commit'], str)
          and campaign._COMMIT.fullmatch(plan['source_base_commit']), 'FUSION_SOURCE_BASE')
    return plan


def _fusion_batch_indices(rows):
    _need(type(rows) is int and rows in (FIT_ROWS, SCORE_ROWS), 'FUSION_POPULATION')
    count, batch = FUSION_BOUNDS['batches_per_partition'], FUSION_BOUNDS['batch_size']
    last = rows // batch - 1
    indices = tuple(i * last // (count - 1) for i in range(count))
    _need(len(set(indices)) == count and indices[0] == 0 and indices[-1] == last
          and all((i + 1) * batch <= rows for i in indices), 'FUSION_BATCH_SELECTION')
    return indices


def _qualify_fusion_initial(plan, qualified):
    """Metadata-only original INITIAL/baseline and completed matched-panel bindings."""
    refs, recipe = plan['inputs'], qualified['recipe']
    for current, original in (('initial_checkpoint','original_initial_checkpoint'),
                              ('initial_baseline','initial_later_capture'),
                              ('initial_measurement','initial_later_measurement')):
        _need(refs[current] == recipe['input_refs'][original], 'FUSION_INITIAL_ORIGIN:' + current)
    baseline = _read(refs['initial_measurement'])
    positions = [i for left in (0,8352,17680) for i in range(left,left+64)]
    _need(recipe['expected_initial_model_state_sha256'] == plan['candidate_model_state_sha256']
          and baseline['schema_version'] == 'gx1_market_state_ta_sf_fold1_evaluation_v1'
          and baseline['evaluated_checkpoint'] == refs['initial_checkpoint']
          and baseline['evaluated_model_state_sha256'] == plan['candidate_model_state_sha256']
          and baseline['capture'] == refs['initial_baseline'] and baseline['optimizer_updates'] == 0
          and baseline['fold_id'] == 1 and baseline['test_data_used'] is False
          and baseline['full_population_evaluated'] is False and baseline['fixed_subset_positions'] == positions
          and baseline['measurement']['rows'] == 192 and baseline['measurement']['batches'] == 12
          and baseline['measurement']['partition'] == 'score'
          and baseline['measurement']['model_contract_sha256'] == recipe['expected_model_contract_sha256']
          and baseline['measurement']['feature_view_sha256'] == recipe['expected_view_sha256'],
          'FUSION_INITIAL_BASELINE_METADATA')
    _qualify_fusion_trained_panels(refs)


def _qualify_fusion_trained_panels(refs):
    trained, terminal, review = (_read(refs[k]) for k in ('trained_result','trained_terminal','trained_review'))
    indices = {p:list(_fusion_batch_indices(n)) for p,n in (('TRAIN',FIT_ROWS),('LATER',SCORE_ROWS))}
    expected_counts = dict(model_constructions=1,checkpoint_loads=1,backbone_encodings=33,forward_returns=33,
        instrumented_forwards=32,archive_decodes=2,head_tensor_imports=4,candidate_writes=0,
        optimizer_constructions=0,optimizer_updates=0,backwards=0,checkpoint_writes=0,
        resume_state_load_calls=1,resume_state_load_returns=1,scalar_fits=0,sampled_rows=512)
    _need(trained['status'] == terminal['status'] == review['status'] == 'COMPLETE'
          and trained['source_unchanged'] is trained['evaluation_state_unchanged'] is True
          and trained['candidate_model_state_sha256'] == FUSION_BOUNDS['candidate_model_state_sha256']
          and trained['batch_indices'] == indices
          and all(type(trained['counts'].get(k)) is int and trained['counts'][k] == v for k,v in expected_counts.items())
          and trained['passivity'] == dict(uninstrumented_matches_bound_baseline=True,
              instrumented_matches_uninstrumented=True,instrumented_matches_bound_baseline=True)
          and trained['artifacts']['TRAIN'] == refs['trained_train']
          and trained['artifacts']['LATER'] == refs['trained_later']
          and trained['plan']['path'] == str(FUSION_ROOT/'PLAN.json')
          and terminal['result'] == refs['trained_result'] and terminal['plan'] == trained['plan']
          and terminal['exit_code'] == 0 and terminal['source_unchanged'] is True
          and terminal['no_relaunch_under_this_plan'] is True
          and review['schema_version'] == 'gx1_ta_fusion_observation_numeric_review_v1'
          and review['result'] == refs['trained_result'] and review['terminal'] == refs['trained_terminal']
          and review['source_commit'] == trained['source_commit']
          and all(review['partitions'][p]['artifact'] == refs['trained_'+p.lower()]
                  and review['partitions'][p]['rows'] == review['partitions'][p]['unique_parents'] == 256
                  for p in ('TRAIN','LATER'))
          and all(x['test_data_used'] is False for x in (trained,terminal,review)),
          'FUSION_INITIAL_MATCHED_TRAINED_METADATA')


def _inspect_fusion_observation(plan_path, plan_sha256, out_dir, *, execute, initial=False, retained=False, temporal=False, gradient=False):
    pr = {'path': str(plan_path), 'sha256': plan_sha256}
    plan = _fusion_plan(_read(pr), plan_path, out_dir, initial=initial, retained=retained, temporal=temporal, gradient=gradient); refs = plan['inputs']; bound = [pr]
    bounds = MIXING_GRADIENT_BOUNDS if gradient else TEMPORAL_BOUNDS if temporal else MTF_RETAINED_BOUNDS if retained else FUSION_INITIAL_BOUNDS if initial else FUSION_BOUNDS; root = Path(out_dir)
    for key, path in (('worker', WORKER), ('dispatch', REPO/DISPATCH), ('cap_runner', REPO/CAP_RUNNER)):
        _need(plan[key]['path'] == str(path), 'FUSION_SOURCE_PATH:' + key)
        campaign.require_binding(plan[key], label=key, verify_file=True); bound.append(plan[key])
    if gradient:
        _need(plan['gradient_helper']['path'] == str(MIXING_GRADIENT_HELPER), 'GRADIENT_HELPER_PATH')
        campaign.require_binding(plan['gradient_helper'],label='gradient helper',verify_file=True);bound.append(plan['gradient_helper'])
    for key, ref in refs.items():
        campaign.require_binding(ref, label='fusion input '+key, verify_file=True); bound.append(ref)
    q = _qualify_reversed_origin(plan, bound)
    if initial:
        _qualify_fusion_initial(plan, q)
    else:
        candidate_plan, candidate, review, terminal = (_read(refs[k]) for k in (
            'candidate_plan', 'candidate_result', 'candidate_review', 'candidate_terminal'))
        _need(candidate_plan['inputs']['reverse_recipe'] == refs['reverse_recipe']
              and candidate_plan['inputs']['reversed_evidence'] == refs['reversed_evidence']
              and candidate['status'] == 'COMPLETE' and candidate['source_unchanged'] is True
              and candidate['evaluation_state_unchanged'] is True and candidate['plan'] == refs['candidate_plan']
              and candidate['candidate_model_state_sha256'] == FUSION_BOUNDS['candidate_model_state_sha256']
              and candidate['artifacts']['folded_coefficients'] == refs['folded_coefficients']
              and candidate['artifacts']['fresh_baseline'] == refs['candidate_baseline']
              and candidate['fresh_candidate_baseline']['rows'] == BATCH_SIZE
              and candidate['fresh_candidate_baseline']['artifact'] == refs['candidate_baseline']
              and candidate['counts']['head_tensor_imports'] == 4 and candidate['counts']['scalar_fits'] == 0
              and candidate['comparison']['decision'] == 'PASS_ALL_FOUR_INITIAL_PRIOR_CELLS'
              and review['status'] == 'PASS_INDEPENDENT_RETAINED_LOGIT_ARITHMETIC'
              and review['result'] == refs['candidate_result'] and review['plan'] == refs['candidate_plan']
              and review['actual_row0_outputs_bitexact_cached_heads'] is True
              and review['folded_coefficients_exact_from_original_FP32_and_locked_slopes'] is True
              and review['primary_comparison_pass'] is True and terminal['status'] == 'COMPLETE'
              and terminal['exit_code'] == 0 and terminal['result'] == refs['candidate_result']
              and terminal['plan'] == refs['candidate_plan'] and terminal['no_relaunch_under_this_plan'] is True
              and all(x['test_data_used'] is False for x in (candidate, review, terminal)),
              'FUSION_QUALIFIED_CANDIDATE')
    if retained or temporal:
        _qualify_fusion_trained_panels(refs)
    head = _git('rev-parse', 'HEAD')
    subprocess.run(['git','-C',str(REPO),'merge-base','--is-ancestor',plan['source_base_commit'],head], check=True)
    policy_ref = _ref(REPO/'NEXT_RUN_POLICY.json'); policy = _read(policy_ref); bound.append(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
          'native_launch_authorized','full_epoch_training_allowed')), 'FUSION_NO_TRAINING')
    if execute:
        gate = policy.get(plan['stage'], {})
        expected = dict(status='ADMITTED', plan=pr, max_invocations=1, remaining_invocations=1,
            claimed_invocations=0, completed_invocations=0, relaunch_allowed=False, execution_authorized=True,
            data_reads_authorized=True, checkpoint_reads_authorized=True, backbone_encodings_authorized=bounds['max_backbone_encodings'],
            head_tensor_imports_authorized=bounds['head_tensor_imports'], candidate_weights_writes_authorized=0,
            backward_calls_authorized=0, optimizer_steps_authorized=0, training_authorized=False)
        if retained:
            expected['pure_joint_decodes_authorized'] = 64
        if gradient:
            expected.update(backward_calls_authorized=64,gradient_observations_authorized=True,
                later_feature_reads_authorized=False)
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in expected.items())
              and all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in bounds.items()),
              'FUSION_ADMISSION')
        _need(_git('branch','--show-current') == 'work/gx1-current', 'FUSION_BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
        _need(not any(root.glob('*.npz')) and not any((root/n).exists()
              for n in ('CLAIM.json','RESULT.json','TERMINAL.json')), 'FUSION_CONSUMED')
    return dict(q, plan=plan, plan_ref=pr, head=head, policy_ref=policy_ref, bound_refs=bound)


def _fusion_exact_arrays(actual, expected):
    return set(actual) == set(expected) and all(actual[k].shape == expected[k].shape
        and actual[k].dtype == expected[k].dtype and actual[k].tobytes() == expected[k].tobytes()
        for k in expected)


def _save_fusion_arrays(name, arrays, counts, *, np, root=None, max_bytes=None):
    root = FUSION_ROOT if root is None else root
    max_bytes = FUSION_BOUNDS['max_cache_bytes'] if max_bytes is None else max_bytes
    _need(max_bytes != 64*1024**2 or root == TEMPORAL_ROOT, 'FUSION_TEMPORAL_BYTE_ROOT')
    _need(type(max_bytes) is int and max_bytes in (FUSION_BOUNDS['max_cache_bytes'],64*1024**2), 'FUSION_BYTE_BOUND')
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    _need(name in ('UNINSTRUMENTED_LATER_ROW0.npz','TRAIN_FUSION.npz','LATER_FUSION.npz',
                   'TRAIN_FUSION_PARTIAL.npz','LATER_FUSION_PARTIAL.npz','TRAIN_MTF_MEANS.npz',
                   'TRAIN_INTERVENTION.npz','LATER_INTERVENTION.npz','TRAIN_INTERVENTION_PARTIAL.npz',
                   'LATER_INTERVENTION_PARTIAL.npz','BASELINE_PARITY_FAILURE.npz',
                   'TRAIN_GRADIENT_BASELINE.npz','TRAIN_GRADIENT_FORWARD.npz',
                   'TRAIN_TEMPORAL_BOUNDARY.npz','LATER_TEMPORAL_BOUNDARY.npz',
                   'TRAIN_TEMPORAL_BOUNDARY_PARTIAL.npz','LATER_TEMPORAL_BOUNDARY_PARTIAL.npz') and bool(arrays), 'FUSION_ARTIFACT_NAME')
    buf = io.BytesIO(); np.savez(buf, **arrays)
    _need(0 < buf.tell() and counts['published_npz_bytes'] + buf.tell() <= max_bytes,
          'FUSION_OUTPUT_BYTES')
    _publish_bytes(root/name, buf.getvalue())
    counts['published_npz_bytes'] += buf.tell()
    return _ref(root/name)


def _publish_fusion_outcome(plan_ref, payload, *, complete, root=None, json_bytes_bound=None):
    root = FUSION_ROOT if root is None else root
    _need(type(complete) is bool, 'FUSION_OUTCOME_TYPE')
    status = 'COMPLETE' if complete else 'FAILED_CONSUMED'
    value = dict(payload,status=status,plan=plan_ref,training_enabled=False,test_data_used=False,
        predictive_value_claimed=False,promotion_authorized=False,canonical_resume_authority=False)
    def terminal_value(result):
        return dict(status=status,exit_code=0 if complete else 1,plan=plan_ref,result=result,
            source_unchanged=payload['source_unchanged'],no_relaunch_under_this_plan=True,test_data_used=False)
    if json_bytes_bound is not None:
        _need(type(json_bytes_bound) is int and json_bytes_bound > 0, 'GRADIENT_JSON_BOUND_TYPE')
        encode=lambda data:(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        raw=encode(value);preview=dict(path=str(root/'RESULT.json'),sha256=hashlib.sha256(raw).hexdigest())
        claim_bytes=(root/'CLAIM.json').stat().st_size
        _need(claim_bytes+len(raw)+len(encode(terminal_value(preview))) <= json_bytes_bound,'GRADIENT_JSON_BOUND')
    result = _publish('RESULT.json',value,root=root)
    terminal = _publish('TERMINAL.json',terminal_value(result),root=root)
    return dict(result=result,terminal=terminal)


def _run_fusion_observation(plan_path, plan_sha256, out_dir, *, device, initial=False, retained=False, temporal=False):
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_endpoint_batch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    started = time.monotonic(); _need(device == 'cuda', 'FUSION_CUDA_ONLY')
    cap = require_guarded_cuda_producer_execution(); cap['lock'] = require_capped_lock_ancestry()
    _need(cap['cgroup_path'] == '/sys/fs/cgroup'+cap['lock']['scope']
          and cap['memory_max'] == cap['memory_high'] == 12*1024**3 and cap['swap'] == 512*1024**2,
          'FUSION_CAP')
    for key,value in (('MAX_POWER_LIMIT_W','250'),('MAX_POWER_DRAW_W','170'),('MAX_CORE_TEMP_C','65'),('MAX_MEMORY_TEMP_C','80')):
        _need(os.environ.get('GX1_TRAINER_GPU_'+key) == value, 'FUSION_HARDWARE:'+key)
    q = _inspect_fusion_observation(plan_path,plan_sha256,out_dir,execute=True,initial=initial,retained=retained,temporal=temporal)
    root, bounds = Path(out_dir), q['plan']
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'FUSION_INIT_ORDER')
    claim = _publish('CLAIM.json',dict(plan=q['plan_ref'],policy=q['policy_ref'],source_commit=q['head'],
        cap=cap,optimizer_updates=0,test_data_used=False),root=root)
    counts = dict(model_constructions=0,checkpoint_loads=0,backbone_encodings=0,forward_returns=0,
        instrumented_forwards=0,archive_decodes=0,head_tensor_imports=0,candidate_writes=0,
        optimizer_constructions=0,optimizer_updates=0,backwards=0,checkpoint_writes=0,
        resume_state_load_calls=0,resume_state_load_returns=0,published_npz_bytes=0,scalar_fits=0,sampled_rows=0)
    if temporal:
        counts.update(temporal_family_encoder_calls=0,temporal_family_axis_calls=0,
            temporal_timeframe_axis_calls=0,temporal_cross_tf_calls=0,temporal_observed_forwards=0)
    if retained:
        counts.update(joint_calls=0,joint_linear_calls=0,state_decoder_calls=0,state_decoder_returns=0,
            future_decoder_calls=0,future_decoder_returns=0,query_projection_calls=0,native_head_calls=0,
            m5_head_calls=0,mean_vector_estimates=0,predictor_fits=0)
    provider = prepared = None; artifacts = {}; pieces = {}; active = None
    passivity = {}; import_proof = {}; candidate = None; matched_observations = {}; retained_summary = {}
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('Passive TA fusion observation 900-second bound')
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(1.,bounds['wall_seconds']-(time.monotonic()-started)))
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']: campaign.require_binding(ref,label='fusion bookend',verify_file=True)
        for name,digest in q['current_sources'].items():
            _need(campaign.file_sha256(REPO/name)==digest,'FUSION_OWNER_BOOKEND:'+name)
    try:
        recipe,receipts,view=(q[k] for k in ('recipe','receipts','view')); refs=q['plan']['inputs']
        numerical=recipe['numerical_recipe']; cuda=torch.device('cuda')
        if temporal:
            from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _BATCH_EQUAL_LENGTH_MTF_EVAL
            _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False, 'TEMPORAL_BATCH_MODE')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _need(torch.get_num_threads()==torch.get_num_interop_threads()==1,'FUSION_THREADS')
        if not retained:
            provider,provider_plan,prepared,fit_rows,fit_issues=runner._prepared_fold1(receipts,view,np=np)
            endpoint,future=load_bound_five_endpoint_targets(),load_bound_future_displacement()
            axes=endpoint.mature_oof_axes(fold_id=1);later_rows,later_issues=axes['score_parent_rows'],axes['score_decision_ns']
            _need(np.array_equal(fit_rows,axes['fit_parent_rows']) and np.array_equal(fit_issues,axes['fit_decision_ns'])
                  and len(fit_rows)==FIT_ROWS and len(later_rows)==SCORE_ROWS,'FUSION_AXES')
        counts['model_constructions']+=1
        backbone=views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile'])
        model=NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,recipe=q['origin']['original_recipe'],verify_files=True,ta_sf=True)
        _need(model.research_objective['contract_sha256']==recipe['expected_model_contract_sha256'],'FUSION_MODEL_CONTRACT')
        counts['checkpoint_loads']+=1
        checkpoint = refs['initial_checkpoint' if initial else 'reversed_evidence']
        saved=common._load_state(checkpoint,torch=torch,maximum=recipe['max_checkpoint_bytes'],counters=counts)
        if retained:
            _need(isinstance(saved.get('parent_order'),torch.Tensor)
                  and saved['parent_order'].dtype == torch.int64 and saved['parent_order'].shape == (FIT_ROWS,),
                  'RETAINED_CHECKPOINT_PARENT_ORDER')
            # The loader checks this complete order against the immutable result digest,
            # original batch order and method_state; no raw feature/target axis is read.
            order = saved['parent_order'].detach().clone()
        else:
            order = torch.from_numpy((fit_rows.copy() if initial else build_reverse_order(fit_rows,recipe['batch_indices'])[0]).copy())
        source_identity=_load_reversed_weights(model,saved,q,order,torch=torch,initial=initial);del saved
        model.to(cuda)
        def signature():
            return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
                modes={n:bool(m.training) for n,m in model.named_modules()},
                trainability={n:p.requires_grad for n,p in model.named_parameters()},
                parameter_identity={n:id(p) for n,p in model.named_parameters()},
                hooks={n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()})
        before=signature()
        source_model = bounds['candidate_model_state_sha256'] if initial else REVERSED_MODEL
        _need(before['model']==source_model and not any(before['trainability'].values()),'FUSION_FROZEN_SOURCE')
        if not initial:
            with np.load(refs['folded_coefficients']['path'],allow_pickle=False) as z:
                expected={prefix+suffix for prefix in ('state_model.m5_side_head','future_m5_side_head') for suffix in ('.weight','.bias')}
                _need(set(z.files)==expected,'FUSION_COEFFICIENT_KEYS')
                coefficients={name:torch.from_numpy(z[name].copy()) for name in z.files}
            counts['archive_decodes']+=1
            import_proof=state.import_equal_weight_m5_side_heads(model,coefficients,expected_model_state_sha256=REVERSED_MODEL,
                coefficient_artifact=refs['folded_coefficients'],coefficient_recipe=refs['candidate_plan'])
            counts['head_tensor_imports']=4
        candidate=signature()
        _need(candidate['model']==bounds['candidate_model_state_sha256']
              and all(candidate[k]==before[k] for k in before if k!='model'),'FUSION_EXACT_CANDIDATE')
        if retained:
            retained_summary = _run_retained_mtf_panels(model,q,counts,artifacts,passivity,np=np,torch=torch)
        else:
            with np.load(refs['initial_baseline' if initial else 'candidate_baseline']['path'],allow_pickle=False) as z:
                baseline={name:z[name].copy() for name in z.files}
            counts['archive_decodes']+=1
            if initial:
                _need(all(len(v)==192 for v in baseline.values()),'FUSION_INITIAL_BASELINE_ROWS')
                baseline={name:value[:BATCH_SIZE].copy() for name,value in baseline.items()}
            def forward(rows,issues,left,partition,*,instrumented):
                _need(counts['backbone_encodings']<bounds['max_backbone_encodings'],'FUSION_FORWARD_LIMIT')
                raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,left,left+BATCH_SIZE,np=np)
                _need(len(parents)==len(clocks)==BATCH_SIZE,'FUSION_INTACT_BATCH')
                features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
                counts['backbone_encodings']+=1
                def evaluate():
                    report=evaluate_endpoint_batch(model,features,endpoint,future,parents,clocks,fold_id=1,
                        partition=partition,future_weight=numerical['future_weight'],frozen_priors=q['priors'],capture_m5_readout=True)
                    counts['forward_returns']+=1
                    return report['m5_readout']
                if not instrumented: return evaluate(),{}
                with observe_ta_fusion_once(model.backbone,batch_rows=BATCH_SIZE,
                        max_bytes=bounds['max_cache_bytes'],torch=torch) as captured:
                    if temporal:
                        with observe_ta_temporal_once(model.backbone,batch_rows=BATCH_SIZE,
                                max_bytes=bounds['max_cache_bytes'],counts=counts,torch=torch) as boundary:
                            readout=evaluate()
                    else:
                        boundary={};readout=evaluate()
                counts['instrumented_forwards']+=1
                return readout,{**{k:v.cpu().numpy().copy() for k,v in captured.items()},
                    **{'boundary_'+k:v.cpu().numpy().copy() for k,v in boundary.items()}}
            uninstrumented,_=forward(later_rows,later_issues,0,'score',instrumented=False)
            artifacts['uninstrumented']=_save_fusion_arrays('UNINSTRUMENTED_LATER_ROW0.npz',uninstrumented,counts,np=np,root=root,max_bytes=bounds['max_cache_bytes'])
            passivity['uninstrumented_matches_bound_baseline']=_fusion_exact_arrays(uninstrumented,baseline)
            _need(passivity['uninstrumented_matches_bound_baseline'],'FUSION_BOUND_BASELINE_PARITY')
            for panel,rows,issues,partition in (('TRAIN',fit_rows,fit_issues,'fit'),('LATER',later_rows,later_issues,'score')):
                active=panel;pieces={}; matched = None
                if initial or temporal:
                    with np.load(refs['trained_'+panel.lower()]['path'],allow_pickle=False) as z:
                        matched={key:z[key].copy() for key in (z.files if temporal else FUSION_MATCH_KEYS)}
                    counts['archive_decodes']+=1
                    positions=np.asarray([i*BATCH_SIZE+j for i in _fusion_batch_indices(len(rows))
                                          for j in range(BATCH_SIZE)],dtype=np.int64)
                    _need(_fusion_exact_arrays({'parents':rows[positions],'issues':issues[positions]},
                        {k:matched[k] for k in ('parents','issues')}),'FUSION_MATCHED_PARENT_AXES')
                for group,batch_index in enumerate(_fusion_batch_indices(len(rows))):
                    readout,captured=forward(rows,issues,batch_index*BATCH_SIZE,partition,instrumented=True)
                    values={**readout,**{'capture_'+k:v for k,v in captured.items() if not k.startswith('boundary_')}}
                    if temporal:
                        expected_group={k:v[group*BATCH_SIZE:(group+1)*BATCH_SIZE] for k,v in matched.items()}
                        exact=_fusion_exact_arrays(values,expected_group)
                        passivity[panel+'_all_retained_fields_bitexact']=passivity.get(panel+'_all_retained_fields_bitexact',True) and exact
                        if not exact:
                            artifacts['baseline_parity_failure']=_save_fusion_arrays('BASELINE_PARITY_FAILURE.npz',
                                {**{'actual_'+k:v for k,v in values.items()},**{'expected_'+k:v for k,v in expected_group.items()}},
                                counts,np=np,root=root,max_bytes=bounds['max_cache_bytes'])
                            _need(False,'TEMPORAL_RETAINED_PARITY:'+panel+':'+str(group))
                        values={**{k:readout[k] for k in ('parents','issues','query_native')},
                                **{k:v for k,v in captured.items() if k.startswith('boundary_')}}
                    for name,value in values.items(): pieces.setdefault(name,[]).append(value)
                    counts['sampled_rows']+=BATCH_SIZE
                    if panel=='LATER' and batch_index==0:
                        passivity['instrumented_matches_uninstrumented']=_fusion_exact_arrays(readout,uninstrumented)
                        passivity['instrumented_matches_bound_baseline']=_fusion_exact_arrays(readout,baseline)
                        _need(all(passivity.values()),'FUSION_INSTRUMENTED_PARITY')
                combined={k:np.concatenate(v,axis=0) for k,v in pieces.items()}
                _need(all(len(v)==bounds['sampled_rows_per_partition'] for v in combined.values()),'FUSION_PANEL_ROWS')
                if initial:
                    matched_observations[panel]=_fusion_exact_arrays({k:combined[k] for k in FUSION_MATCH_KEYS},matched)
                    _need(matched_observations[panel],'FUSION_MATCHED_OBSERVATIONS')
                artifacts[panel]=_save_fusion_arrays(panel+('_TEMPORAL_BOUNDARY.npz' if temporal else '_FUSION.npz'),combined,counts,np=np,root=root,max_bytes=bounds['max_cache_bytes'])
                pieces={}
        _need(signature()==candidate and all(p.grad is None for p in model.parameters()),'FUSION_STATE_UNCHANGED')
        expected=dict(model_constructions=1,checkpoint_loads=1,backbone_encodings=33,forward_returns=33,
            instrumented_forwards=32,archive_decodes=3 if initial else 2,head_tensor_imports=bounds['head_tensor_imports'],candidate_writes=0,
            optimizer_constructions=0,optimizer_updates=0,backwards=0,checkpoint_writes=0,
            resume_state_load_calls=1,resume_state_load_returns=1,scalar_fits=0,sampled_rows=512)
        if retained:
            expected.update(backbone_encodings=0,forward_returns=0,instrumented_forwards=0,archive_decodes=3,
                joint_calls=64,joint_linear_calls=64,state_decoder_calls=64,state_decoder_returns=64,
                future_decoder_calls=64,future_decoder_returns=64,query_projection_calls=128,
                native_head_calls=128,m5_head_calls=128,mean_vector_estimates=2,predictor_fits=0)
        if temporal:
            expected.update(archive_decodes=4,temporal_family_encoder_calls=1280,temporal_family_axis_calls=32,
                temporal_timeframe_axis_calls=32,temporal_cross_tf_calls=32,temporal_observed_forwards=32)
        _need(all(counts[k]==v for k,v in expected.items()) and all(passivity.values()),'FUSION_EXACT_COUNTS')
        if prepared is not None:
            release=prepared;prepared=None;base._release_pages(provider,release)
            provider.validate_metadata(provider_plan)
        bookend()
        return _publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            source_commit=q['head'],source_identity=source_identity,import_proof=import_proof,
            candidate_model_state_sha256=candidate['model'],passivity=passivity,**retained_summary,
            **(dict(matched_observations=matched_observations,trained_result=refs['trained_result']) if initial else {}),
            **(dict(temporal_methods=_temporal_methods(model.backbone),trained_result=refs['trained_result'],
                observed_calls_exclude_one_uninstrumented_baseline=True) if temporal else {}),
            batch_indices={p:list(_fusion_batch_indices(n)) for p,n in (('TRAIN',FIT_ROWS),('LATER',SCORE_ROWS))},
            source_unchanged=True,evaluation_state_unchanged=True,elapsed_seconds=time.monotonic()-started,
            limitations=['512 rows in 16 intact B16 groups per partition are diagnostic, not 512 independent samples or a quality threshold.',
                'Gate weights and representation variation do not establish predictive utility.',
                'Residual scales and joint LayerNorm must be included in component-size comparisons.'] +
                (['Fixed mean replacement changes normalization and dependence between components; sensitivity is not unique information or edge.',
                  'M5 S/F are predeclared primary; other timeframes are descriptive only, without selection.'] if retained else []) +
                (['Temporal encoder inputs include learned projection, context-conditioned gates, embeddings and positional encoding.',
                  'Within-window activation variance is distinct from variation between decisions; neither proves predictive utility.'] if temporal else [])),complete=True,root=root)
    except BaseException as exc:
        preservation_error=changed=None
        if pieces and active not in artifacts:
            try:
                artifacts[active+'_partial']=_save_fusion_arrays(active+('_TEMPORAL_BOUNDARY_PARTIAL.npz' if temporal else '_FUSION_PARTIAL.npz'),
                    {k:np.concatenate(v,axis=0) for k,v in pieces.items()},counts,np=np,root=root,max_bytes=bounds['max_cache_bytes'])
            except Exception as other: preservation_error=repr(other)
        try:bookend()
        except Exception as other:changed=repr(other)
        _publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            import_proof=import_proof,passivity=passivity,matched_observations=matched_observations,
            **(dict(evaluation_state_unchanged=signature()==candidate) if (retained or temporal) and candidate is not None else {}),
            error=type(exc).__name__+':'+str(exc),
            source_unchanged=changed is None,bookend_error=changed,preservation_error=preservation_error,
            elapsed_seconds=time.monotonic()-started),complete=False,root=root)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);signal.signal(signal.SIGALRM,previous_alarm)
        if prepared is not None:base._release_pages(provider,prepared)


def _retained_mtf_means(source, *, np):
    _need(source.shape == (256,512) and source.dtype == np.float32 and bool(np.isfinite(source).all()),
          'RETAINED_TRAIN_MEAN_SOURCE')
    base, after, pool, _ = np.split(source.astype(np.float64),4,axis=1)
    return dict(mean_mtf_residual=np.mean(after-base,axis=0).astype(np.float32),
                mean_mtf_pool=np.mean(pool,axis=0).astype(np.float32))


def _retained_mtf_intervention(source, means, *, np):
    _need(source.shape == (16,512) and source.dtype == np.float32 and bool(np.isfinite(source).all())
          and set(means) == {'mean_mtf_residual','mean_mtf_pool'}
          and all(v.shape == (128,) and v.dtype == np.float32 and bool(np.isfinite(v).all()) for v in means.values()),
          'RETAINED_INTERVENTION_INPUT')
    changed = source.copy()
    changed[:,128:256] = source[:,:128] + means['mean_mtf_residual']
    changed[:,256:384] = means['mean_mtf_pool']
    _need(bool(np.isfinite(changed).all()) and changed[:,:128].tobytes() == source[:,:128].tobytes()
          and changed[:,384:].tobytes() == source[:,384:].tobytes(), 'RETAINED_PRESERVED_BLOCKS')
    return changed


@contextmanager
def _retained_decoder_hooks(model, counts):
    """Observe actual child-module calls; reject every backbone invocation."""
    handles = []
    def count(name, limit):
        def hook(_module, _inputs):
            counts[name] += 1
            _need(counts[name] <= limit, 'RETAINED_CALL_LIMIT:' + name)
        return hook
    def no_backbone(_module, _inputs):
        counts['backbone_encodings'] += 1
        _need(False, 'RETAINED_BACKBONE_CALL_FORBIDDEN')
    modules = (
        (model.backbone.entry_q_joint_norm,'joint_calls',64),
        (model.backbone.entry_q_joint_in,'joint_linear_calls',64),
        (model.state_model.query_projection,'query_projection_calls',128),
        (model.state_model.query_head,'native_head_calls',128),
        (model.future_head,'native_head_calls',128),
        (model.state_model.m5_side_head,'m5_head_calls',128),
        (model.future_m5_side_head,'m5_head_calls',128))
    try:
        handles.append(model.backbone.register_forward_pre_hook(no_backbone))
        for module,name,limit in modules:
            handles.append(module.register_forward_pre_hook(count(name,limit)))
        yield
    finally:
        for handle in handles: handle.remove()


def _decode_retained_joint(model, source, query, counts, *, torch):
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import _m5_readout_hooks
    _need(source.shape == (16,512) and source.dtype == torch.float32
          and query.shape == (16,5) and query.dtype == torch.int64
          and not any(m.training for m in model.modules()), 'RETAINED_DECODE_INPUT')
    captured, handles = _m5_readout_hooks(model,16)
    try:
        with torch.inference_mode():
            hidden = torch.nn.functional.gelu(model.backbone.entry_q_joint_in(
                model.backbone.entry_q_joint_norm(source)))
            counts['state_decoder_calls'] += 1
            _need(counts['state_decoder_calls'] <= 64, 'RETAINED_STATE_DECODE_LIMIT')
            state = model.state_model.decode_raw_query_grid(hidden,query[:,None,:])[:,0]
            counts['state_decoder_returns'] += 1
            counts['future_decoder_calls'] += 1
            _need(counts['future_decoder_calls'] <= 64, 'RETAINED_FUTURE_DECODE_LIMIT')
            future = model.decode_native_F_raw(hidden)
            counts['future_decoder_returns'] += 1
            _need(set(captured) == {'state_representation','future_representation'}
                  and hidden.shape == (16,128) and state.shape == future.shape == (16,5,2)
                  and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in (hidden,state,future)),
                  'RETAINED_DECODE_OUTPUT')
            return dict(captured,capture_entry_q_joint_hidden=hidden.detach().cpu().numpy().copy(),
                state_activity_side_logits=state.detach().cpu().numpy().copy(),
                future_activity_side_logits=future.detach().cpu().numpy().copy())
    finally:
        for handle in handles: handle.remove()


def _load_retained_panel(ref, counts, *, np):
    from gx1.scripts.run_market_state_time_query_fold1_fit_window_v1 import _ta_m5_capture_layout
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    layout = dict(_ta_m5_capture_layout(256),capture_entry_q_joint_source=((256,512),np.float32),
                  capture_entry_q_joint_hidden=((256,128),np.float32))
    _need(counts['archive_decodes'] < 3, 'RETAINED_ARCHIVE_LIMIT')
    counts['archive_decodes'] += 1
    with np.load(ref['path'],allow_pickle=False) as z:
        _need(set(layout) <= set(z.files), 'RETAINED_PANEL_KEYS')
        arrays = {k:z[k].copy() for k in layout}
    _need(all(arrays[k].shape == shape and arrays[k].dtype == dtype and bool(np.isfinite(arrays[k]).all())
              for k,(shape,dtype) in layout.items())
          and len(np.unique(arrays['parents'])) == 256 and bool((np.diff(arrays['parents']) > 0).all())
          and bool((np.diff(arrays['issues']) > 0).all())
          and np.array_equal(arrays['query_native'],np.broadcast_to(np.asarray(NATIVE_K,dtype=np.int64),(256,5))),
          'RETAINED_PANEL_LAYOUT_AXES')
    return arrays


def _retained_conditional_scores(arrays, *, np):
    from gx1.scripts.run_market_state_native_F_private_window_v1 import _m5_import_score
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    def empty(): return dict(count=0,up_count=0,brier=None,log=None,mean_p=None)
    def pooled(groups, arm):
        count = sum(g[arm]['count'] for g in groups)
        if count == 0: return empty()
        return dict(count=count,up_count=sum(g[arm]['up_count'] for g in groups),**{
            key:sum(g[arm]['count']*g[arm][key] for g in groups if g[arm]['count'])/count
            for key in ('brier','log','mean_p')})
    def delta(original, changed):
        return {k:None if not original['count'] else changed[k]-original[k] for k in ('brier','log')}
    output = {}
    for task in ('state','future'):
        output[task] = {}
        for tf_index,tf in enumerate(TIMEFRAMES):
            allowed = arrays[task+'_allowed'][:,tf_index]
            known = arrays[task+'_eligible'][:,tf_index] & arrays[task+'_mature'][:,tf_index] & np.isin(allowed,(1,4))
            y = allowed == 4; groups = []
            for group in range(16):
                sl = slice(group*16,(group+1)*16); scores = {}
                for arm in ('original','intervention'):
                    logits = arrays[arm+'_'+task+'_activity_side_logits'][sl,tf_index,1]
                    scores[arm] = _m5_import_score(logits,known[sl],y[sl]) if bool(known[sl].any()) else empty()
                groups.append(dict(group=group,**scores,delta=delta(scores['original'],scores['intervention'])))
            original, changed = pooled(groups,'original'), pooled(groups,'intervention')
            output[task][tf] = dict(role='PRIMARY_M5' if tf_index == 0 else 'DESCRIPTIVE_SECONDARY',
                original=original,intervention=changed,delta=delta(original,changed),groups=groups)
    return output


def _run_retained_mtf_panels(model, checked, counts, artifacts, passivity, *, np, torch):
    """One fixed retained-input intervention; the shared runner owns guards/state."""
    root = Path(checked['plan_ref']['path']).parent; refs = checked['plan']['inputs']
    arrays = _load_retained_panel(refs['trained_train'],counts,np=np)
    _need(counts['mean_vector_estimates'] == 0 and counts['archive_decodes'] == 2, 'RETAINED_TRAIN_MEAN_ORDER')
    means = _retained_mtf_means(arrays['capture_entry_q_joint_source'],np=np)
    for value in means.values(): value.flags.writeable = False
    counts['mean_vector_estimates'] = 2
    artifacts['train_means'] = _save_fusion_arrays('TRAIN_MTF_MEANS.npz',means,counts,np=np,root=root)
    artifacts['train_mean_lock'] = _publish('TRAIN_MTF_MEANS.json',dict(plan=checked['plan_ref'],
        source=refs['trained_train'],artifact=artifacts['train_means'],rows=256,
        arithmetic=MTF_RETAINED_BOUNDS['mean_arithmetic'],mean_vector_estimates=2,predictor_fits=0,
        later_archive_opened=False,training_enabled=False,test_data_used=False),root=root)
    conditional_scores = {}
    with _retained_decoder_hooks(model,counts):
        for panel in ('TRAIN','LATER'):
            pieces = {}
            try:
                if panel == 'LATER':
                    campaign.require_binding(artifacts['train_means'],label='locked TRAIN means',verify_file=True)
                    campaign.require_binding(artifacts['train_mean_lock'],label='locked TRAIN mean receipt',verify_file=True)
                    arrays = _load_retained_panel(refs['trained_later'],counts,np=np)
                for group in range(16):
                    left,right = group*16,(group+1)*16
                    original = arrays['capture_entry_q_joint_source'][left:right]
                    query = torch.from_numpy(arrays['query_native'][left:right].copy()).to(model.state_model.query_native_k.device)
                    decoded = _decode_retained_joint(model,torch.from_numpy(original.copy()).to(query.device),query,counts,torch=torch)
                    parity = {k:_fusion_exact_arrays({k:v},{k:arrays[k][left:right]}) for k,v in decoded.items()}
                    for k,ok in parity.items(): passivity[k+'_bitexact'] = passivity.get(k+'_bitexact',True) and ok
                    if not all(parity.values()):
                        failure = {**{k:arrays[k][left:right] for k in FUSION_MATCH_KEYS},
                            **{'actual_'+k:v for k,v in decoded.items()},
                            **{'expected_'+k:arrays[k][left:right] for k in decoded}}
                        artifacts['baseline_parity_failure'] = _save_fusion_arrays('BASELINE_PARITY_FAILURE.npz',failure,counts,np=np,root=root)
                        artifacts['baseline_parity_failure_metadata'] = _publish('BASELINE_PARITY_FAILURE.json',
                            dict(plan=checked['plan_ref'],panel=panel,group=group,source=refs['trained_'+panel.lower()],
                                 parity=parity,artifact=artifacts['baseline_parity_failure'],tolerance_fallback=False),root=root)
                        _need(False,'RETAINED_BASELINE_PARITY:' + ','.join(k for k,ok in parity.items() if not ok))
                    changed = _retained_mtf_intervention(original,means,np=np)
                    altered = _decode_retained_joint(model,torch.from_numpy(changed).to(query.device),query,counts,torch=torch)
                    values = {k:arrays[k][left:right].copy() for k in FUSION_MATCH_KEYS}
                    for task in ('state','future'):
                        name = task+'_activity_side_logits'
                        values['original_'+name],values['intervention_'+name] = decoded[name],altered[name]
                    for k,v in values.items(): pieces.setdefault(k,[]).append(v)
                    counts['sampled_rows'] += 16
                combined = {k:np.concatenate(v,axis=0) for k,v in pieces.items()}
                _need(all(len(v) == 256 for v in combined.values()),'RETAINED_PANEL_ROWS')
                conditional_scores[panel] = _retained_conditional_scores(combined,np=np)
                artifacts[panel] = _save_fusion_arrays(panel+'_INTERVENTION.npz',combined,counts,np=np,root=root)
            except BaseException:
                if pieces and panel not in artifacts:
                    artifacts[panel+'_partial'] = _save_fusion_arrays(panel+'_INTERVENTION_PARTIAL.npz',
                        {k:np.concatenate(v,axis=0) for k,v in pieces.items()},counts,np=np,root=root)
                raise
    return dict(conditional_scores=conditional_scores,trained_result=refs['trained_result'],
        primary_cells=['TRAIN/state/M5','TRAIN/future/M5','LATER/state/M5','LATER/future/M5'],
        intervention=MTF_RETAINED_BOUNDS['intervention'],mean_arithmetic=MTF_RETAINED_BOUNDS['mean_arithmetic'],
        baseline_tolerance_fallback=False,backbone_tripwire_installed=True,model_change_recommended=False)


def _mixing_vector_summary(vectors, *, torch):
    """FP64 geometry only; no optimizer or parameter update is simulated."""
    names = ('state_m5_side','future_m5_side','rest','total')
    _need(set(vectors) == set(names) and all(v.dtype == torch.float64 and v.device.type == 'cpu'
          and v.ndim == 1 and bool(torch.isfinite(v).all()) for v in vectors.values()), 'GRADIENT_VECTORS')
    norms = {n:float(torch.linalg.vector_norm(vectors[n])) for n in names}
    dot = {a:{b:float(torch.dot(vectors[a],vectors[b])) for b in names} for a in names}
    cosine = {a:{b:None if norms[a]*norms[b] == 0 else dot[a][b]/(norms[a]*norms[b]) for b in names} for a in names}
    m5 = vectors['state_m5_side'] + vectors['future_m5_side'];m5norm = float(torch.linalg.vector_norm(m5))
    return dict(norms=norms,dot=dot,cosine=cosine,combined_m5_norm=m5norm,
        combined_m5_dot_rest=float(torch.dot(m5,vectors['rest'])),
        combined_m5_dot_total=float(torch.dot(m5,vectors['total'])),
        combined_m5_cosine_rest=None if m5norm*norms['rest']==0 else float(torch.dot(m5,vectors['rest']))/(m5norm*norms['rest']),
        interpretation='Negative gradient dot products indicate local opposing Euclidean directions. They do not predict an AdamW step or establish a historical training cause.')


def _run_mixing_gradient_observation(plan_path, plan_sha256, out_dir, *, device):
    import numpy as np
    import torch
    from gx1.audit.ta_direction_mixing_gradient_v1 import measure_mixing_gradient_partition, CLOSURE_RELATIVE_L2_LIMIT
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
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
    started=time.monotonic();_need(device=='cuda','GRADIENT_CUDA_ONLY')
    cap=require_guarded_cuda_producer_execution();cap['lock']=require_capped_lock_ancestry()
    _need(cap['cgroup_path']=='/sys/fs/cgroup'+cap['lock']['scope'] and cap['memory_max']==cap['memory_high']==12*1024**3
          and cap['swap']==512*1024**2,'GRADIENT_CAP')
    for key,value in (('MAX_POWER_LIMIT_W','250'),('MAX_POWER_DRAW_W','170'),('MAX_CORE_TEMP_C','65'),('MAX_MEMORY_TEMP_C','80')):
        _need(os.environ.get('GX1_TRAINER_GPU_'+key)==value,'GRADIENT_HARDWARE:'+key)
    q=_inspect_fusion_observation(plan_path,plan_sha256,out_dir,execute=True,gradient=True)
    root=Path(out_dir);bounds=q['plan'];refs=bounds['inputs']
    _need(CLOSURE_RELATIVE_L2_LIMIT == bounds['closure_relative_tolerance'],'GRADIENT_CLOSURE_BINDING')
    claim=_publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],max_invocations=1,
        no_relaunch_under_this_plan=True,test_data_used=False),root=root)
    counts=dict(model_constructions=0,checkpoint_loads=0,backbone_encodings=0,forward_returns=0,
        gradient_forwards=0,archive_decodes=0,head_tensor_imports=0,autograd_grad_calls=0,backwards=0,
        optimizer_constructions=0,optimizer_updates=0,checkpoint_writes=0,candidate_writes=0,
        resume_state_load_calls=0,resume_state_load_returns=0,published_npz_bytes=0,sampled_rows=0,scalar_fits=0)
    provider=prepared=None;model=None;selected=[];artifacts={};records=[];sums={};candidate=None;passivity={}
    previous_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Mixing gradient observation 900-second bound')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(1.,bounds['wall_seconds']-(time.monotonic()-started)))
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']:campaign.require_binding(ref,label='gradient bookend',verify_file=True)
        for name,digest in q['current_sources'].items():_need(campaign.file_sha256(REPO/name)==digest,'GRADIENT_OWNER:'+name)
    try:
        recipe,receipts,view=(q[k] for k in ('recipe','receipts','view'));numerical=recipe['numerical_recipe'];cuda=torch.device('cuda')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        _need(torch.get_num_threads()==torch.get_num_interop_threads()==1 and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'GRADIENT_RUNTIME')
        provider,provider_plan,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        require_m5_equal_observation_axes(rows,issues)
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions']+=1
        backbone=views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile'])
        model=NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,recipe=q['origin']['original_recipe'],verify_files=True,ta_sf=True)
        counts['checkpoint_loads']+=1
        saved=common._load_state(refs['reversed_evidence'],torch=torch,maximum=recipe['max_checkpoint_bytes'],counters=counts)
        order=torch.from_numpy(build_reverse_order(rows,recipe['batch_indices'])[0].copy())
        source_identity=_load_reversed_weights(model,saved,q,order,torch=torch);del saved
        model.to(cuda)
        with np.load(refs['folded_coefficients']['path'],allow_pickle=False) as z:
            expected={prefix+suffix for prefix in ('state_model.m5_side_head','future_m5_side_head') for suffix in ('.weight','.bias')}
            _need(set(z.files)==expected,'GRADIENT_COEFFICIENT_KEYS');coefficients={name:torch.from_numpy(z[name].copy()) for name in z.files}
        counts['archive_decodes']+=1
        import_proof=state.import_equal_weight_m5_side_heads(model,coefficients,expected_model_state_sha256=REVERSED_MODEL,
            coefficient_artifact=refs['folded_coefficients'],coefficient_recipe=refs['candidate_plan']);counts['head_tensor_imports']=4
        def signature():
            return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
                modes={n:bool(m.training) for n,m in model.named_modules()},
                trainability={n:p.requires_grad for n,p in model.named_parameters()},
                parameter_identity={n:id(p) for n,p in model.named_parameters()},
                hooks={n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()})
        candidate=signature();_need(candidate['model']==bounds['candidate_model_state_sha256']
            and not any(candidate['trainability'].values()) and not any(candidate['modes'].values()),'GRADIENT_FROZEN_CANDIDATE')
        all_parameters=dict(model.named_parameters());groups={}
        for group in bounds['parameter_groups']:
            prefix='state_model.backbone.'+group+'.';groups[group]=[(n,p) for n,p in all_parameters.items() if n.startswith(prefix)]
            _need(len(groups[group])==12,'GRADIENT_PARAMETER_GROUP:'+group)
        selected=[p for values in groups.values() for _,p in values]
        _need(len(selected)==len({id(p) for p in selected})==bounds['parameter_tensors'],'GRADIENT_PARAMETER_SCOPE')
        for p in selected:p.requires_grad_(True)
        def forward(left,gradient):
            _need(counts['backbone_encodings']<bounds['max_backbone_encodings'] and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'GRADIENT_FORWARD_BOUND')
            raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,left,left+BATCH_SIZE,np=np)
            _need(len(parents)==BATCH_SIZE and np.array_equal(parents,rows[left:left+BATCH_SIZE])
                  and np.array_equal(clocks,issues[left:left+BATCH_SIZE]),'GRADIENT_INTACT_BATCH')
            features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()};counts['backbone_encodings']+=1
            with torch.set_grad_enabled(gradient),trainer._training_mha_fastpath_context(model.backbone),                 trainer._training_autocast_context(),trainer._training_model_finite_check_context():
                _need(torch.backends.mha.get_fastpath_enabled() is False,'GRADIENT_FASTPATH_PROFILE')
                report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                    seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=left)
            counts['forward_returns']+=1
            if gradient:counts['gradient_forwards']+=1
            _need(report['model_contract_sha256']==recipe['expected_model_contract_sha256'] and report['loss_contract_sha256']==recipe['expected_loss_contract_sha256']
                  and report['skip_update'] is False and bool(torch.isfinite(report['loss'])),'GRADIENT_CANONICAL_LOSS')
            arrays={k:report[k].detach().cpu().numpy().copy() for k in ('state_activity_side_logits','future_activity_side_logits','state_log_probs','future_log_probs')}
            arrays.update({k:report[k].copy() for k in ('state_query_k','future_native_k','state_allowed','future_allowed','state_mature','future_mature')})
            arrays.update(parents=parents.copy(),issues=clocks.copy(),loss=np.asarray(float(report['loss'].detach()),dtype=np.float32))
            return report,arrays
        indices=_fusion_batch_indices(FIT_ROWS)
        baseline_report,baseline=forward(indices[0]*BATCH_SIZE,False);del baseline_report
        artifacts['baseline']=_save_fusion_arrays('TRAIN_GRADIENT_BASELINE.npz',baseline,counts,np=np,root=root)
        for group_index,batch_index in enumerate(indices):
            report,arrays=forward(batch_index*BATCH_SIZE,True)
            if group_index==0:
                artifacts['gradient_forward']=_save_fusion_arrays('TRAIN_GRADIENT_FORWARD.npz',arrays,counts,np=np,root=root)
                passivity['fresh_no_grad_matches_gradient_forward_bitexact']=_fusion_exact_arrays(arrays,baseline)
                _need(passivity['fresh_no_grad_matches_gradient_forward_bitexact'],'GRADIENT_FRESH_BASELINE_PARITY')
            _need(counts['autograd_grad_calls']+4<=bounds['max_autograd_grad_calls'],'GRADIENT_AUTOGRAD_BOUND')
            with torch.enable_grad():
                measurement,vectors=measure_mixing_gradient_partition(report,groups,counts=counts,torch=torch)
            counts['backwards']=counts['autograd_grad_calls'];counts['sampled_rows']+=BATCH_SIZE
            geometry={group:_mixing_vector_summary(v,torch=torch) for group,v in vectors.items()}
            for group,v in vectors.items():
                if group not in sums:sums[group]={arm:x.clone() for arm,x in v.items()}
                else:
                    for arm,x in v.items():sums[group][arm].add_(x)
            retained=sum(x.numel()*x.element_size() for collection in (vectors,sums) for values in collection.values() for x in values.values())
            _need(retained<=bounds['max_retained_gradient_bytes'],'GRADIENT_RETAINED_BYTES')
            records.append(dict(batch_index=batch_index,rows=BATCH_SIZE,loss=float(report['loss'].detach()),
                state_per_tf_counts=report['state_per_tf_counts'],future_per_tf_counts=report['future_per_tf_counts'],
                measurement=measurement,geometry=geometry))
            del report,arrays,vectors,measurement
            _need(all(p.grad is None for p in model.parameters()),'GRADIENT_NO_ACCUMULATED_GRAD')
        means={group:{arm:x/len(indices) for arm,x in values.items()} for group,values in sums.items()}
        aggregate={group:_mixing_vector_summary(v,torch=torch) for group,v in means.items()}
        for p in selected:p.requires_grad_(False)
        _need(signature()==candidate and all(p.grad is None for p in model.parameters()),'GRADIENT_MODEL_UNCHANGED')
        expected=dict(model_constructions=1,checkpoint_loads=1,backbone_encodings=17,forward_returns=17,
            gradient_forwards=16,archive_decodes=1,head_tensor_imports=4,autograd_grad_calls=64,backwards=64,
            optimizer_constructions=0,optimizer_updates=0,checkpoint_writes=0,candidate_writes=0,
            resume_state_load_calls=1,resume_state_load_returns=1,sampled_rows=256,scalar_fits=0)
        _need(all(counts[k]==v for k,v in expected.items()),'GRADIENT_EXACT_COUNTS')
        release=prepared;prepared=None;base._release_pages(provider,release);provider.validate_metadata(provider_plan);bookend()
        payload=dict(claim=claim,counts=counts,artifacts=artifacts,source_commit=q['head'],source_identity=source_identity,
            import_proof=import_proof,candidate_model_state_sha256=candidate['model'],passivity=passivity,
            parameter_names={group:[n for n,_ in pairs] for group,pairs in groups.items()},batch_indices=list(indices),
            batch_records=records,mean_gradient_geometry=aggregate,source_unchanged=True,evaluation_state_unchanged=True,
            elapsed_seconds=time.monotonic()-started,limitations=['Local eval-mode gradient decomposition at the corrected c171 candidate on fixed TRAIN batches only.',
                'Sampled-S/native-F query plan and exact original loss normalization retained. Three encoder groups only; no historical optimizer or whole-model conflict claim.',
                'No dropout, clipping, AdamW, parameter update, predictor fit, feature intervention or later-target selection. Gradient geometry is not edge.'])
        return _publish_fusion_outcome(q['plan_ref'],payload,complete=True,root=root,json_bytes_bound=bounds['max_published_json_bytes'])
    except BaseException as exc:
        counts['backwards']=counts['autograd_grad_calls']
        for p in selected:p.requires_grad_(False)
        changed=None
        try:bookend()
        except Exception as other:changed=repr(other)
        result=_publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,batch_records=records,
            passivity=passivity,failed_gradient_measurement=getattr(exc,'report_json',None),
            error=type(exc).__name__+':'+str(exc),source_unchanged=changed is None,bookend_error=changed),
            complete=False,root=root,json_bytes_bound=bounds['max_published_json_bytes'])
        raise RuntimeError('TA_MIXING_GRADIENT_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        for p in selected:p.requires_grad_(False)
        if prepared is not None:base._release_pages(provider,prepared)
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)
