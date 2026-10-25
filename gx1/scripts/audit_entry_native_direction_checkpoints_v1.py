"""Read-only four-checkpoint276 diagnosis; no fit, step, export or launch alias.

All numerical imports follow the existing native guard and fresh exact admission.
The 194 forwards and six reverse calls are additional genuine computation, not
synthetic qualification. Original state files are decoded once and never written.
This component has no admitted capped-runner/dispatcher launch route. Genuine
execution needs separate integration admission; there is no alternate cap class.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from gx1.contracts import entry_native_direction_checkpoint_diagnostic_v1 as authority


@contextmanager
def _measurement(model, optimizer, *, device, trainer, prefix, tree_sha, evaluation):
    require = authority.require
    require(all(p.grad is None for p in model.parameters()), 'MEASUREMENT_PARTIAL_GRADIENT')
    before = {**prefix._online_identity(model, device, tree_sha, trainer),
        'optimizer': tree_sha(optimizer.state_dict())}
    modes = [(module, module.training) for module in model.modules()]
    rng = trainer._attended_session_rng_state(device=device) if not evaluation else None
    try:
        if evaluation:
            model.eval()
        else:
            require(all(training for _, training in modes), 'CHECKPOINT_TRAIN_MODES')
        yield before
    finally:
        for module, training in modes:
            module.training = training
        # The diagnostic TRAIN-mode forward consumes dropout RNG, deliberately
        # restored without restoring any model/buffer/optimizer mutation.
        if rng is not None:
            trainer._restore_attended_session_rng_state(rng, device=device)
        after = {**prefix._online_identity(model, device, tree_sha, trainer),
            'optimizer': tree_sha(optimizer.state_dict())}
        require(after == before and all(p.grad is None for p in model.parameters()),
            'MEASUREMENT_STATE_CHANGED')


def _module_scope(name):
    module = name.rsplit('.', 1)[0]
    shared = name.startswith(('backbone.mtf_family_encoder.', 'backbone.mtf_family_proj.'))
    return module, ('shared_across_timeframes' if shared else 'named_parameter_route')


def _norm(tensor, torch):
    value = tensor.detach().to(device='cpu', dtype=torch.float64)
    return math.sqrt(float((value * value).sum()))


def parameter_geometry(states, torch):
    """All original named parameters; absent Adam state is never called zero."""
    require = authority.require
    initial = states[0]['model_state']
    names = [item['name'] for item in states[0]['binding']['parameter_inventory']]
    reports = []
    for index, (spec, state) in enumerate(zip(authority.STATE_SPEC, states)):
        require(names == [item['name'] for item in state['binding']['parameter_inventory']],
            'PARAMETER_INVENTORY_IDENTITY')
        active = state['binding']['optimizer_group_parameter_names'][0]
        adam = {name: state['optimizer_state']['state'].get(i) for i, name in enumerate(active)}
        previous = states[max(0, index - 1)]['model_state']
        rows = []
        for name in names:
            value = state['model_state'][name]
            reference = _norm(initial[name], torch)
            delta = value.double() - initial[name].double()
            prior_delta = value.double() - previous[name].double()
            delta_norm = _norm(delta, torch)
            item = adam.get(name)
            module, scope = _module_scope(name)
            rows.append({'parameter': name, 'module': module, 'scope': scope,
                'elements': value.numel(), 'weight_norm': _norm(value, torch),
                'delta_from_0U_norm': delta_norm,
                'delta_from_previous_norm': _norm(prior_delta, torch),
                'changed_elements_from_0U': int(torch.count_nonzero(delta)),
                'relative_delta_from_0U': delta_norm / reference if reference else None,
                'zero_reference_norm': reference == 0,
                'optimizer_eligible': name in adam, 'adam_state_present': item is not None,
                'adam_step': int(item['step']) if item is not None else None,
                'adam_first_moment_norm': _norm(item['exp_avg'], torch) if item is not None else None,
                'adam_second_moment_norm': _norm(item['exp_avg_sq'], torch) if item is not None else None})
        layer_differences = []
        for name in names:
            if '.layers.0.' not in name:
                continue
            for layer in (1, 2):
                other = name.replace('.layers.0.', '.layers.' + str(layer) + '.', 1)
                if other in state['model_state']:
                    difference = state['model_state'][other].double() - state['model_state'][name].double()
                    layer_differences.append({'first': name, 'other': other,
                        'equal': bool(torch.equal(state['model_state'][name], state['model_state'][other])),
                        'difference_norm': _norm(difference, torch)})
        reports.append({'state': spec[0], 'parameters': rows, 'layer_differences': layer_differences})
    return reports


class ActivationRecorder:
    """Temporary exact native module hooks; only scalar/moment summaries leave."""
    def __init__(self, model, torch):
        self.torch = torch
        self.handles = []
        self.expected = {}
        self.records = {}
        self.seen = set()
        self.family_calls = {}
        self.width = int(model.backbone.cfg.d_model)
        self.batch = 16
        b = model.backbone
        self.tfs = ('m5', 'm15', 'h1', 'h4', 'd1')
        families = tuple(b._mtf_specialist_names)
        authority.require(b.entry_family_tf_token_order == tuple(
            tf + ':' + family for tf in self.tfs for family in families), 'ACTIVATION_TF_ORDER')
        self._attach(b.encoder, 'local/global/encoder', (16, b.cfg.seq_len, self.width))
        for family, module in b.specialist_encoder.items():
            self._attach(module, 'local/' + family + '/encoder', (16, b.cfg.seq_len, self.width))
        for family, module in b.mtf_family_encoder.items():
            routes = [(tf, (16, int(getattr(b.cfg, tf + '_seq_len')), self.width)) for tf in self.tfs]
            for tf, shape in routes:
                self.expected['mtf/' + tf + '/' + family + '/encoder'] = shape
            def family_hook(_module, _args, out, family=family, routes=routes):
                at = self.family_calls.get(family, 0)
                authority.require(at < len(routes), 'EXTRA_FAMILY_TF_HOOK')
                tf, shape = routes[at]
                self.family_calls[family] = at + 1
                self._capture('mtf/' + tf + '/' + family + '/encoder', out, shape)
            self.handles.append(module.register_forward_hook(family_hook))
        for name in ('gc_mtf_family_proj', 'gc_mtf_feature_context_gate', 'mtf_feature_context_gate'):
            for key, module in getattr(b, name).items():
                tf = key.split('__', 1)[0]
                shape = ((16, int(getattr(b.cfg, tf + '_seq_len')), self.width)
                    if name == 'gc_mtf_family_proj' else (16, module.out_features))
                self._attach(module, name + '/' + key, shape)
        for name in ('fuse', 'specialist_out', 'cross_tf_out', 'family_tf_cooperation_out',
                'family_context_global', 'entry_q_joint_norm', 'entry_q_joint_in'):
            width = 4 * self.width if name == 'entry_q_joint_norm' else self.width
            self._attach(getattr(b, name), name, (16, width))
        self.expected['direction_head/input'] = (16, self.width)
        self.handles.append(model.direction_head.register_forward_pre_hook(
            lambda _module, args: self._capture('direction_head/input', args[0], (16, self.width))))
        self.scales = {'specialist_out': float(b.specialist_fusion_scale),
            'cross_tf_out': float(b.multi_tf_scale),
            'family_tf_cooperation_out': float(b.cross_family_fusion_scale)}

    def _attach(self, module, name, shape):
        self.expected[name] = shape
        self.handles.append(module.register_forward_hook(
            lambda _module, _args, out, name=name, shape=shape: self._capture(name, out, shape)))

    def begin(self):
        authority.require(not self.seen, 'UNFINISHED_HOOK_FRAME')
        self.family_calls = {}

    def _capture(self, name, value, shape):
        torch = self.torch
        authority.require(name not in self.seen and tuple(value.shape) == shape
            and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'HOOK_SHAPE_OR_DUPLICATE:' + name)
        self.seen.add(name)
        raw = value.detach()
        extras = {}
        if raw.ndim == 3:
            mean = raw.mean(dim=1)
            last = raw[:, -1, :]
            # The unchanged native readout is FP32 mean + last. This observer
            # recomputes that operation; no prediction head is fit on it.
            vector = mean + last
            extras = {'pre_pool_square_mean': float(raw.double().square().mean()),
                'temporal_centered_square_mean': float((raw - mean[:, None, :]).double().square().mean()),
                'last_minus_mean_square_mean': float((last - mean).double().square().mean()),
                'mean_readout_square_mean': float(mean.double().square().mean()),
                'last_readout_square_mean': float(last.double().square().mean())}
        else:
            vector = raw
        vector = vector.to(device='cpu', dtype=torch.float64)
        record = self.records.setdefault(name, {'calls': 0, 'rows': 0,
            'sum': torch.zeros(vector.shape[1], dtype=torch.float64),
            'square_sum': torch.zeros(vector.shape[1], dtype=torch.float64), 'extras': {}})
        record['calls'] += 1
        record['rows'] += len(vector)
        record['sum'] += vector.sum(dim=0)
        record['square_sum'] += vector.square().sum(dim=0)
        for key, number in extras.items():
            record['extras'][key] = record['extras'].get(key, 0.) + number

    def end(self):
        authority.require(self.seen == set(self.expected), 'MISSING_OR_EXTRA_HOOK_ROUTES')
        self.seen.clear()

    def report(self):
        result = {}
        for name, record in self.records.items():
            authority.require(record['calls'] == 16 and record['rows'] == 256, 'HOOK_PANEL_CALLS')
            mean = record['sum'] / record['rows']
            second = record['square_sum'] / record['rows']
            result[name] = {'calls': record['calls'], 'rows': record['rows'],
                'shape_per_call': list(self.expected[name]),
                'rms': math.sqrt(float(second.mean())),
                'mean_vector_norm': math.sqrt(float(mean.square().sum())),
                'between_parent_variance_trace': float((second - mean.square()).clamp_min(0).sum()),
                **{key.removesuffix('_square_mean') + '_rms': math.sqrt(value / record['calls'])
                    for key, value in record['extras'].items()}}
        base = result['fuse']['rms']
        for name, scale in self.scales.items():
            result[name].update({'fixed_fusion_scale': scale,
                'scaled_rms': abs(scale) * result[name]['rms'],
                'scaled_rms_over_local_base_rms': abs(scale) * result[name]['rms'] / base if base else None})
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _profile_budget(budget, codes, phase, previous=None):
    def profile(frame, event, arg):
        code = frame.f_code
        key = codes.get(code)
        if key is not None:
            if event == 'call':
                budget.start(key)
            elif event == 'return':
                budget.returned(key)
        if event == 'call':
            filename, name = code.co_filename, code.co_name
            owner = filename.startswith(authority.REPO + '/') or filename.startswith('/home/andre2/GX1_RUNS/')
            forbidden = ((owner and (name == 'fit' or name.startswith('fit_')
                or name in ('partial_fit', '_optimizer_step_with_finite_gradients', '_save_state',
                    '_save_snapshot', 'load_m1_feature_surface', 'load_current',
                    'query_dynamic_m1_prefix', 'read_dynamic_m5_target_source',
                    'build_dynamic_m5_batch_targets', '_dynamic_m5_optimizer_task_losses')))
                or ('/torch/optim/' in filename and name == 'step')
                or ('/torch/' in filename and name == 'backward')
                or (filename.endswith('/torch/serialization.py') and name == 'save')
                or ('/site-packages/sklearn/' in filename and (name == 'fit' or name.startswith('fit_'))))
            authority.require(not forbidden, 'FORBIDDEN_OPERATION:' + name)
            if filename.endswith('/torch/nn/modules/module.py') and name == '__init__':
                authority.require(phase['name'] == 'CONSTRUCT', 'EXTRA_MODULE_CONSTRUCTION')
        if previous is not None:
            previous(frame, event, arg)
    return profile


def _install_guard(phase, output):
    """Metadata-only preparation, then the existing exact physical read guard.

    All original paths are readonly even when the inherited read guard admits
    them. PT body opens are confined to the four counted load operations.
    """
    denied = []
    def guard(event, args):
        if event in ('os.system', 'os.posix_spawn'):
            denied.append(event)
            raise RuntimeError('NATIVE_DIRECTION_CHECKPOINT_DIAGNOSTIC_SUBPROCESS')
        if event == 'subprocess.Popen':
            command = args[1]
            authority.require(command in (['git', 'rev-parse', 'HEAD'], ['git', 'status', '--porcelain']),
                'UNDECLARED_SUBPROCESS')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
            return
        raw = os.fsdecode(args[0])
        path = os.path.realpath(raw)
        flags = args[2] if len(args) > 2 else 0
        writing = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing:
            authority.require(path.startswith(str(output) + '/'), 'ORIGINAL_OR_UNBOUND_WRITE:' + path)
        if path.startswith(('/home/andre2/GX1_RUNS/', '/home/andre2/GX1_DATA/')) and not path.startswith(str(output) + '/'):
            allowed = (phase['name'] != 'METADATA' or Path(path).suffix in ('.json', '.py'))
            if Path(path).suffix == '.pt':
                allowed = phase['name'] == 'LOAD'
            if not allowed:
                denied.append(path)
                raise RuntimeError('NATIVE_DIRECTION_CHECKPOINT_DIAGNOSTIC_PHASE_OPEN:' + path)
    sys.addaudithook(guard)
    return denied


def _monitored_codes(context, np, torch, prefix, view_owner, trainer, state_owner):
    from gx1.features import htf_features as htf
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_gc_mtf_history_v1 import load_gc_mtf_history_raw_artifact
    from gx1.contracts.entry_native_direction_research_v1 import NativeEntryDirectionResearch
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
    functions = {'torch_load': torch.load, 'numpy_load': np.load,
        'native_forward': EntryV10CtxHybridTransformer.forward,
        'research_forward': NativeEntryDirectionResearch.forward,
        'trainer_forward': trainer._model_forward_fp32,
        'backbone_constructor': EntryV10CtxHybridTransformer.__init__,
        'research_constructor': NativeEntryDirectionResearch.__init__,
        'adamw_constructor': torch.optim.AdamW.__init__,
        'autograd_grad': torch.autograd.grad,
        'engine_reverse': torch.autograd.graph._engine_run_backward,
        'physical_prepare': context['physical'].prepare_inputs,
        'prefix_prepare': prefix._prepare, 'batch': prefix._batch,
        'gather': view_owner.gather_native_direction_feature_batch,
        'axes_loader': context['ancestor'].read_axes,
        'reuse_mapper': context['ancestor'].map_reuse,
        'clock_loader': load_m1_feature_surface_times,
        'cache_loader': htf.load_multi_tf_v4_cache,
        'raw_gc_loader': load_gc_mtf_history_raw_artifact,
        'verified_npy_loader': htf._load_verified_cache_npy}
    codes = {function.__code__: key for key, function in functions.items()}
    authority.require(len(codes) == len(functions), 'DISTINCT_MONITORED_CODE_OWNERS')
    return codes


def _panel(model, optimizer, prepared, targets, view, context_map, selected, *, recorder,
        budget, forward_counts, cutoff, device, np, torch, prefix, trainer, tree_sha, prior):
    from gx1.contracts import entry_native_direction_context_report_v1 as reporter
    from gx1.contracts import entry_direction_probability_reference_v1 as rank_owner
    parts = {key: [] for key in ('parent_rows', 'bar_start_ns', 'decision_ns',
        'terminal_time_ns', 'observed', 'sign', 'logit_up')}
    tags = []
    header = None
    with _measurement(model, optimizer, device=device, trainer=trainer, prefix=prefix,
            tree_sha=tree_sha, evaluation=True) as preservation:
        try:
            for left in range(0, 256, 16):
                chosen = selected[left:left + 16]
                raw, target = prefix._batch(prepared, targets, view, chosen, np)
                tagged = reporter.tag_native_direction_contexts(raw['features'], context_map=context_map,
                    witness=raw['witness'])
                current_header = {k: v for k, v in tagged.items() if k not in ('parent_rows', 'decision_ns', 'tags')}
                if header is None:
                    header = current_header
                authority.require(header == current_header and np.array_equal(tagged['parent_rows'], chosen)
                    and np.array_equal(tagged['decision_ns'], target['decision_ns']), 'PANEL_TAG_JOIN')
                features, _unused = prefix._device_batch(raw, target, torch, device)
                recorder.begin()
                budget.start('panel_forward')
                logits = prefix._forward(model, features, kind='C_FIXED_TRAIN_PANEL',
                    counters=forward_counts, ceiling=194, torch=torch)
                budget.returned('panel_forward')
                recorder.end()
                batch = {'parent_rows': chosen, 'bar_start_ns': raw['witness']['bar_start_ns'],
                    **target, 'logit_up': logits.detach().cpu().numpy()}
                for key in parts:
                    parts[key].append(np.asarray(batch[key]).copy())
                tags.append(tagged['tags'])
                del raw, target, features, _unused, logits, batch
        finally:
            recorder.close()
    arrays = {key: np.ascontiguousarray(np.concatenate(values)) for key, values in parts.items()}
    tag_arrays = {name: np.ascontiguousarray(np.concatenate([item[name] for item in tags])) for name in tags[0]}
    context = {**header, 'parent_rows': arrays['parent_rows'], 'decision_ns': arrays['decision_ns'], 'tags': tag_arrays}
    metrics = reporter.score_native_direction_logits(arrays['logit_up'], sign=arrays['sign'],
        observed=arrays['observed'], decision_ns=arrays['decision_ns'],
        terminal_time_ns=arrays['terminal_time_ns'], cutoff_ns=cutoff, context_tags=context)
    prior_logits = np.broadcast_to(np.asarray(prior['logit_up'], np.float64), (256, 3)).copy()
    prior_metrics = reporter.score_native_direction_logits(prior_logits, sign=arrays['sign'],
        observed=arrays['observed'], decision_ns=arrays['decision_ns'],
        terminal_time_ns=arrays['terminal_time_ns'], cutoff_ns=cutoff, context_tags=context)
    rank_and_bias = []
    for horizon, minutes in enumerate((1, 5, 15)):
        take = arrays['observed'][:, horizon] & (arrays['sign'][:, horizon] != 0) & (arrays['terminal_time_ns'][:, horizon] < cutoff)
        labels = np.ascontiguousarray((arrays['sign'][take, horizon] == 1).astype(np.int8))
        scores = np.ascontiguousarray(arrays['logit_up'][take, horizon], dtype=np.float64)
        probabilities = rank_owner.expit(scores)
        probability_mean = float(probabilities.mean()) if len(scores) else None
        observed_up = float(labels.mean()) if len(scores) else None
        rank_and_bias.append({'horizon_minutes': minutes,
            'rank': rank_owner.direction_rank_diagnostics(labels, scores, score_kind='logit_up'),
            'mean_logit_up': float(scores.mean()) if len(scores) else None,
            'mean_probability_up': probability_mean, 'observed_up_fraction': observed_up,
            'mean_probability_minus_observed_up': probability_mean - observed_up if len(scores) else None})
    return arrays, {'model_metrics': metrics, 'prior_metrics': prior_metrics,
        'rank_and_bias': rank_and_bias, 'activation_geometry': recorder.report(),
        'preservation': preservation,
        'decision_min_ns': int(arrays['decision_ns'].min()),
        'decision_max_ns': int(arrays['decision_ns'].max()),
        'parent_rows_sha256': hashlib.sha256(arrays['parent_rows'].tobytes()).hexdigest(),
        'decision_ns_sha256': hashlib.sha256(arrays['decision_ns'].tobytes()).hexdigest()}, tag_arrays


def horizon_contributions(logits, target, cutoff, torch):
    """Original unweighted BCE, decomposed using its total eligible denominator."""
    from gx1.contracts.entry_native_direction_research_v1 import native_direction_binary_loss
    loss = native_direction_binary_loss(logits, **target, cutoff_ns=cutoff)
    authority.require(loss['update_allowed'], 'GRADIENT_BATCH_HAS_NO_LABELS')
    mask = loss['eligible_mask']
    count = loss['counts']['eligible_cells']
    pieces = []
    for h in range(3):
        labels = (target['sign'][:, h] == 1).to(dtype=logits.dtype)
        values = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, h], labels, reduction='none')
        pieces.append(values[mask[:, h]].sum() / count)
    # Positive reductions may group differently: use the standard accumulated
    # roundoff bound derived from actual cell count and three horizon sums.
    eps = torch.finfo(logits.dtype).eps
    rounding = (count + len(pieces)) * eps
    authority.require(rounding < 1, 'LOSS_ROUNDOFF_DOMAIN')
    authority.require(bool(torch.isclose(sum(pieces), loss['loss'],
        rtol=rounding / (1 - rounding), atol=0)),
        'HORIZON_LOSS_DECOMPOSITION')
    return pieces, loss['counts'], loss['per_horizon_counts']


def gradient_geometry(named_parameters, gradients, torch):
    groups = {}
    for at, (name, parameter) in enumerate(named_parameters):
        module, scope = _module_scope(name)
        group = groups.setdefault(module, {'scope': scope, 'parameters': 0,
            'none_parameters': [0, 0, 0], 'zero_parameters': [0, 0, 0],
            'nonzero_parameters': [0, 0, 0], 'squares': [0., 0., 0.],
            'dots': {'0_1': 0., '0_2': 0., '1_2': 0.}})
        group['parameters'] += 1
        values = []
        for h in range(3):
            gradient = gradients[h][at]
            if gradient is None:
                group['none_parameters'][h] += 1
                values.append(None)
                continue
            authority.require(gradient.shape == parameter.shape and bool(torch.isfinite(gradient).all()), 'GRADIENT_DOMAIN')
            value = gradient.detach().to(device='cpu', dtype=torch.float64)
            square = float(value.square().sum())
            group['squares'][h] += square
            group['nonzero_parameters' if square else 'zero_parameters'][h] += 1
            values.append(value)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            if values[a] is not None and values[b] is not None:
                group['dots'][str(a) + '_' + str(b)] += float((values[a] * values[b]).sum())
    for group in groups.values():
        group['norms'] = [math.sqrt(value) for value in group.pop('squares')]
        group['cosines'] = {}
        for a, b in ((0, 1), (0, 2), (1, 2)):
            denominator = group['norms'][a] * group['norms'][b]
            key = str(a) + '_' + str(b)
            group['cosines'][key] = group['dots'][key] / denominator if denominator else None
    return groups


def _gradient(model, optimizer, prepared, targets, view, selected, *, budget,
        forward_counts, cutoff, device, np, torch, prefix, trainer, tree_sha):
    raw, target = prefix._batch(prepared, targets, view, selected, np)
    features, device_target = prefix._device_batch(raw, target, torch, device)
    named = [(name, p) for name, p in model.named_parameters()
        if name.startswith('backbone.') and p.requires_grad]
    authority.require(named, 'SHARED_TRAINABLE_PARAMETERS')
    with _measurement(model, optimizer, device=device, trainer=trainer, prefix=prefix,
            tree_sha=tree_sha, evaluation=False) as preservation:
        budget.start('gradient_forward')
        logits = prefix._forward(model, features, kind='C_HORIZON_GRADIENT',
            counters=forward_counts, ceiling=194, torch=torch, grad=True)
        budget.returned('gradient_forward')
        pieces, counts, per_horizon = horizon_contributions(logits, device_target, cutoff, torch)
        gradients = []
        for h, piece in enumerate(pieces):
            budget.start('reverse')
            values = torch.autograd.grad(piece, [p for _, p in named], retain_graph=h < 2,
                create_graph=False, allow_unused=True)
            budget.returned('reverse')
            gradients.append(values)
        geometry = gradient_geometry(named, gradients, torch)
        result = {'loss_contributions': [float(piece.detach()) for piece in pieces],
            'counts': counts, 'per_horizon_counts': per_horizon, 'modules': geometry,
            'preservation': preservation, 'mode': 'original_checkpoint_TRAIN_with_restored_dropout_RNG',
            'historical_next_step_reconstructed': False,
            'batch_parent_rows_sha256': hashlib.sha256(selected.tobytes()).hexdigest()}
        del gradients, pieces, logits, values
    return result


def _baseline_arrays(ref, *, common, np, contexts):
    authority.require(common._hash_file(ref['path']) == ref, 'OWN_INITIAL_PREDICTION_BYTES')
    before = common._stat(ref['path'])
    names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up')
    context_names = tuple('context__' + name for name in contexts)
    with np.load(ref['path'], allow_pickle=False) as archive:
        authority.require(archive.files == list(names) + list(context_names), 'OWN_INITIAL_MEMBERS')
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    authority.require(common._stat(ref['path']) == before, 'OWN_INITIAL_ARCHIVE_BOOKEND')
    expected = {'parent_rows': (np.int64, (256,)), 'bar_start_ns': (np.int64, (256,)),
        'decision_ns': (np.int64, (256,)), 'terminal_time_ns': (np.int64, (256, 3)),
        'observed': (np.bool_, (256, 3)), 'sign': (np.int8, (256, 3)), 'logit_up': (np.float32, (256, 3))}
    expected.update({name: (np.int8, (256,)) for name in context_names})
    authority.require(all(array.dtype == expected[name][0] and array.shape == expected[name][1]
        and bool(np.isfinite(array).all()) for name, array in arrays.items()), 'OWN_INITIAL_ARRAY_HEADERS')
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def run_checkpoint_diagnostic(*, plan_path, plan_sha256, admission_path, admission_sha256):
    # Must remain before context, numpy, torch, rank/view/state or device imports.
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits = require_guarded_cuda_trainer_execution()
    sys.dont_write_bytecode = True
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    metadata = {}
    plan_ref = authority.plain_ref({'path': str(Path(plan_path)), 'sha256': plan_sha256})
    admission_ref = authority.plain_ref({'path': str(Path(admission_path)), 'sha256': admission_sha256})
    plan = authority.require_plan(common._read(plan_ref, metadata))
    authority.require_admission(common._read(admission_ref, metadata), plan_ref)
    proposal = authority.require_proposal(common._read(authority.PROPOSAL, metadata))
    policy = common._read(plan['policy'], metadata)
    gate = policy.get(authority.STAGE, {})
    authority.require(gate.get('plan') == plan_ref and gate.get('admission') == admission_ref
        and gate.get('execution_authorized') is True
        and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
        and gate.get('max_invocations') == 1 and gate.get('test_data_used') is False
        and gate.get('optimizer_steps_authorized') is False, 'CURRENT_POLICY_ADMISSION')
    recipe = common._read(proposal['existing_input_refs']['recipe'], metadata)
    qualified = common._read(proposal['existing_input_refs']['first_prefix_qualification'], metadata)
    authority.require(qualified['technical_fit_resume_scoring_qualified'] is True
        and recipe['input_manifest'] == proposal['existing_input_refs']['input_manifest'],
        'QUALIFIED_276_INPUT_MANIFEST')
    authority.require(recipe['fold'] == authority.FOLD and recipe['numerical_recipe']['batch_size'] == 16
        and recipe['numerical_recipe']['max_updates'] == 2976, 'OWN_276_RECIPE')
    initial = common._read(authority.INITIAL_REPORT, metadata)
    authority.require(initial['role'] == 'INITIAL_TRAIN' and initial['predictions'] == authority.INITIAL_PREDICTIONS
        and initial['fold'] == authority.FOLD and initial['initial_grad_nograd_parity'] is True,
        'BOUND_OWN_INITIAL_REPORT')
    refs = []
    for item in proposal['planned_saved_states']:
        declared = common._read(item['declaring_JSON'], metadata)
        ref = item['planned_PT_ref']
        key = 'state' if 'state' in declared else ('own_initial_state' if item['updates'] == 0 else 'full_state')
        authority.require(declared[key] == ref and declared['recipe'] == proposal['existing_input_refs']['recipe']
            and declared['fold'] == authority.FOLD
            and declared['feature_view_identity'] == initial['feature_view_identity'], 'DECLARED_CHECKPOINT_REF')
        if item['updates'] == 0:
            # This RESULT is terminal, with a separate own_initial_state ref.
            # Its end cursor is not the cursor of its declared zero-update file.
            authority.require(declared['initial_measurement'] == authority.INITIAL_REPORT
                and initial['own_initial_state'] == ref == qualified['own_initial_state'], 'DECLARED_OWN_0U')
        else:
            authority.require(declared['parent_cursor'] == item['consumed_parent_cursor']
                and declared['optimizer_updates'] == item['updates'], 'DECLARED_CURSOR_UPDATES')
        if item['role'] == 'terminal_u2976':
            authority.require(declared['outcome'] == declared['phase_state'] == 'COMPLETE', 'COMPLETE_TERMINAL_STATE')
        observed = common._stat(ref['path'])
        authority.require(observed['st_size'] == item['stat_size_bytes']
            and not observed['st_mode'] & 0o222, 'CHECKPOINT_READONLY_STAT_SIZE')
        refs.append(ref)
    required_sources = {**recipe['source_owners'], **proposal['existing_sources_reused_without_change']}
    authority.require(set(plan['source_owners']) == set(required_sources) | set(authority.NEW_FILES)
        and all(plan['source_owners'][key] == ref for key, ref in required_sources.items()), 'EXACT_SOURCE_CLOSURE')
    for ref in plan['source_owners'].values():
        authority.require(common._hash_file(ref['path']) == ref, 'CURRENT_SOURCE_BYTES')
    output = Path(plan['run_root'])
    authority.require(not output.is_symlink() and not output.exists(), 'CONSUMED_RUN_ROOT')
    output.mkdir()
    counts = {'json_bytes': 0, 'resume_state_load_calls': 0, 'resume_state_load_returns': 0}
    budget = authority.CallBudget()
    phase = {'name': 'METADATA'}
    denied = _install_guard(phase, output)
    started = time.monotonic()
    before_io = common._io()
    old_alarm = signal.getsignal(signal.SIGALRM)
    def deadline(_signum, _frame):
        raise TimeoutError('NATIVE_DIRECTION_CHECKPOINT_DIAGNOSTIC_WALL_LIMIT')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(plan['budget']['max_wall_seconds'])
    old_profile = sys.getprofile()
    context = None
    result = {'decision': 'FAILED', 'error': None, 'operation': authority.OPERATION,
        'plan': plan_ref, 'admission': admission_ref, 'test_data_used': False,
        'optimizer_steps': 0, 'limits': list(authority.LIMITS)}
    binding = {'source_head': plan['source_head'], 'policy': plan['policy']}
    common._json(output / 'CLAIM.json', {'plan': plan_ref, 'admission': admission_ref,
        'limits': limits, 'operation_limits': authority.CALL_LIMITS}, counts, plan['budget']['max_json_output_bytes'])
    originals = {ref['path']: common._stat(ref['path']) for ref in refs + [authority.INITIAL_PREDICTIONS]}
    try:
        context = prefix._metadata(recipe, common)
        context['info']['metadata'].update(metadata)
        common._freeze(plan, binding, context)
        guard_context = {**context, 'info': {**context['info'], 'metadata': {
            **context['info']['metadata'], **{ref['path']: {'ref': ref, 'stat': originals[ref['path']]}
                for ref in refs + [authority.INITIAL_PREDICTIONS]}}}}
        delegated_denied = common._install_read_guard(guard_context, output, output)
        import numpy as np
        import torch
        from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
        from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
        from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
        codes = _monitored_codes(context, np, torch, prefix, view_owner, trainer, state_owner)
        sys.setprofile(_profile_budget(budget, codes, phase, old_profile))
        phase['name'] = 'PREPARE'
        prepared, targets, view, context_map, order = prefix._prepare(context, recipe, np)
        prior = prefix._prefix_prior(prepared['arrays'], targets, authority.FOLD, np)
        authority.require(prior == initial['own_prefix_prior'], 'OWN_PREFIX_PRIOR')
        phase['name'] = 'BASELINE'
        saved_initial = _baseline_arrays(authority.INITIAL_PREDICTIONS, common=common, np=np,
            contexts=[name for name, _ in CONTEXT_SPEC])
        phase['name'] = 'LOAD'
        states = []
        for ref, spec in zip(refs, authority.STATE_SPEC):
            budget.start('checkpoint_load')
            state = common._load_state(ref, torch=torch, maximum=plan['budget']['max_checkpoint_bytes'], counters=counts)
            budget.returned('checkpoint_load')
            authority.require(state['schema_version'] == state_owner.SCHEMA_VERSION
                and state['contract_sha256'] == tree_sha({k: v for k, v in state.items() if k != 'contract_sha256'})
                and state['cursor'] == spec[1] and state['optimizer_updates'] == spec[2], 'SAVED_STATE_SEAL_CURSOR')
            states.append(state)
        phase['name'] = 'GEOMETRY'
        geometry = parameter_geometry(states, torch)
        common._json(output / 'PARAMETER_GEOMETRY.json', {'states': geometry,
            'limits': list(authority.LIMITS)}, counts, plan['budget']['max_json_output_bytes'])
        device = torch.device('cuda', 0)
        trainer._set_deterministic(recipe['numerical_recipe']['seed'], device,
            recipe['numerical_recipe']['precision_policy'])
        phase['name'] = 'CONSTRUCT'
        budget.start('model_build')
        model = view_owner.build_native_direction_research_model(view,
            prefix_normalization=context['fitted'], constructor_metadata=context['constructor'], device=device)
        budget.returned('model_build')
        model.train()
        budget.start('optimizer_build')
        optimizer, _binding = state_owner.build_native_direction_research_optimizer(model, recipe=recipe['numerical_recipe'])
        budget.returned('optimizer_build')
        parent_order = torch.from_numpy(order.copy())
        fresh = state_owner.capture_native_direction_research_state(model, optimizer,
            recipe=recipe['numerical_recipe'], feature_view_identity=model.feature_view_identity,
            parent_order=parent_order, cursor=0, optimizer_updates=0, device=device)
        identity_keys = ('binding', 'model_state', 'buffer_state', 'optimizer_state', 'rng_state', 'module_training')
        authority.require(tree_sha({key: fresh[key] for key in identity_keys}) == initial['fresh_identity_sha256']
            == tree_sha({key: states[0][key] for key in identity_keys})
            and model.feature_view_identity == initial['feature_view_identity'], 'OWN_FRESH_INITIAL_IDENTITY')
        del fresh, geometry
        forward_counts = {'forward_calls': 0, 'forward_returns': 0, 'forward_kind_calls': {}}
        panel_refs, gradient_refs = [], []
        for spec, state in zip(authority.STATE_SPEC, states):
            phase['name'] = 'RESTORE'
            state_owner.restore_native_direction_research_state(state, model, optimizer,
                recipe=recipe['numerical_recipe'], feature_view_identity=model.feature_view_identity,
                parent_order=parent_order, device=device)
            for panel_name, left, right in authority.PANELS:
                phase['name'] = 'PANEL'
                recorder = ActivationRecorder(model, torch)
                arrays, measured, tag_arrays = _panel(model, optimizer, prepared, targets, view,
                    context_map, order[left:right], recorder=recorder, budget=budget,
                    forward_counts=forward_counts, cutoff=authority.FOLD['train_cutoff_ns'],
                    device=device, np=np, torch=torch, prefix=prefix, trainer=trainer,
                    tree_sha=tree_sha, prior=prior)
                parity = None
                if spec[0] == 'own0U' and left == 0:
                    actual = {**arrays, **{'context__' + key: value for key, value in tag_arrays.items()}}
                    authority.require(set(actual) == set(saved_initial)
                        and all(actual[key].dtype == saved_initial[key].dtype
                            and np.array_equal(actual[key], saved_initial[key]) for key in actual),
                        'OWN_INITIAL_FIRST256_BIT_PARITY')
                    authority.require(measured['model_metrics'] == initial['model_metrics']
                        and measured['prior_metrics'] == initial['prior_metrics'], 'OWN_INITIAL_REPORT_PARITY')
                    parity = {'reference': authority.INITIAL_PREDICTIONS, 'bit_identical': True,
                        'reference_grad_nograd_parity_reused_not_remeasured': initial['initial_grad_nograd_parity']}
                measured.update({'state': spec[0], 'panel': panel_name,
                    'seen_status': authority.panel_seen_status(left, right, spec[1]),
                    'own_initial_reproduction': parity})
                panel_refs.append(common._json(output / (spec[0] + '__' + panel_name + '.json'),
                    measured, counts, plan['budget']['max_json_output_bytes']))
                del arrays, measured, tag_arrays, recorder
            if spec[0] in ('own0U', 'terminal_u2976'):
                phase['name'] = 'GRADIENT'
                gradient = _gradient(model, optimizer, prepared, targets, view, order[:16],
                    budget=budget, forward_counts=forward_counts,
                    cutoff=authority.FOLD['train_cutoff_ns'], device=device, np=np, torch=torch,
                    prefix=prefix, trainer=trainer, tree_sha=tree_sha)
                gradient.update({'state': spec[0], 'horizons_minutes': [1, 5, 15]})
                gradient_refs.append(common._json(output / (spec[0] + '__HORIZON_GRADIENTS.json'),
                    gradient, counts, plan['budget']['max_json_output_bytes']))
        authority.require(counts['resume_state_load_calls'] == counts['resume_state_load_returns'] == 4
            and forward_counts['forward_calls'] == forward_counts['forward_returns'] == 194,
            'DELEGATED_LOADER_AND_FORWARD_COUNTS')
        authority.require(not denied and not delegated_denied, 'FORBIDDEN_INPUT_OPEN')
        result.update({'decision': 'COMPLETE_READONLY_TRAIN_DIAGNOSIS', 'calls': budget.complete(),
            'source_function_unchanged': True, 'initial_report': authority.INITIAL_REPORT,
            'initial_predictions': authority.INITIAL_PREDICTIONS,
            'states': [{'role': item['role'], 'ref': item['planned_PT_ref']}
                for item in proposal['planned_saved_states']], 'panels': panel_refs,
            'gradients': gradient_refs, 'own_prefix_prior': prior,
            'same_276_initial_function_reproduced': True,
            'context_map': context_map, 'feature_view_identity': model.feature_view_identity,
            'activation_causality_qualified': False, 'capacity_or_convergence_qualified': False,
            'predictive_value_qualified': False, 'economic_edge_qualified': False})
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)[:4096]}
    finally:
        sys.setprofile(old_profile)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        phase['name'] = 'FINAL'
        try:
            authority.require(all(common._stat(path) == before for path, before in originals.items()),
                'ORIGINAL_FILE_STAT_BOOKENDS')
            if context is not None:
                common._freeze(plan, binding, context)
            authority.require(common._io()['rchar'] - before_io['rchar'] <= plan['budget']['max_rchar_bytes'], 'RCHAR_CEILING')
        except Exception as exc:
            result.update({'decision': 'FAILED', 'error': {'type': type(exc).__name__, 'message': str(exc)[:4096]}})
        result.update({'elapsed_seconds': time.monotonic() - started, 'limits_proof': limits,
            'observed_calls': dict(budget.calls), 'observed_returns': dict(budget.returns),
            'denied_phase_opens': denied, 'originals_stat_preserved': all(
                common._stat(path) == before for path, before in originals.items())})
        common._json(output / 'RESULT.json', result, counts, plan['budget']['max_json_output_bytes'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    args = parser.parse_args()
    result = run_checkpoint_diagnostic(plan_path=args.plan, plan_sha256=args.plan_sha256,
        admission_path=args.admission, admission_sha256=args.admission_sha256)
    return 0 if result['decision'] == 'COMPLETE_READONLY_TRAIN_DIAGNOSIS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
