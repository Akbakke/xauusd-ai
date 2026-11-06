"""Role-separated October native inputs over the frozen prefix and development axis.

This is a data adapter, not campaign admission or model authority.
"""
from __future__ import annotations

import hashlib
import mmap
from pathlib import Path


from gx1.audit import dynamic_m5_october_native_batch_binding as batch_owner
from gx1.audit.dynamic_mtf_december_native_inputs import _read_xau_cache_manifest
from gx1.audit import dynamic_mtf_october_input_only as prior
from gx1.contracts import entry_dynamic_m5_october_causal_inputs_v1 as causal
from gx1.contracts import entry_dynamic_m5_october_native_roles_v1 as roles
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as state
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts import local_random_access_campaign_v2 as campaign


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_NATIVE_INPUT_' + reason)


class _PageRelease:
    @staticmethod
    def release_pages(ancestor, prepared, discard):
        for value in prepared.get('october_mmaps', ()):
            handle = getattr(value, '_mmap', None)
            if handle is not None and hasattr(handle, 'madvise'):
                handle.madvise(mmap.MADV_DONTNEED)


def _leaves(manifest, ref, common):
    root = Path(ref['path']).parent
    files, stats = {}, {}
    for name, desc in manifest['arrays'].items():
        path = str(root / (name + '.npy'))
        _require(desc['path'] == path and Path(path).is_file() and not Path(path).is_symlink(),
                 'LEAF_PATH:' + name)
        files[name] = {'path': path, 'sha256': desc['sha256']}
        stats[name] = common._stat(path)
        _require(stats[name]['st_size'] == desc['size_bytes'], 'LEAF_SIZE:' + name)
    return files, stats


def _role_payloads(raw, info, provider, read, metadata):
    """Only files the prefix gather can open; all other 264 payloads stay closed."""
    from gx1.audit.dynamic_mtf_train_prefix_inputs import _check_payload_stats

    root = Path(raw['native_cache_dir'])
    cache = _read_xau_cache_manifest(raw, metadata)
    gc = read(raw['gc_surfaces'])
    paths = {raw['axes']['source']['path'], info['source']['output_parquet']}
    paths.update(provider.reuse_specs(info['reuse']))
    for tf in state.TIMEFRAMES:
        for name in ('ts_npy', 'feats_npy'):
            paths.add(str(root / cache['tfs'][tf][name]))
        paths.update(desc['path'] for desc in gc['surfaces'][tf].values())
    _require(len(paths) == 33 and paths <= set(raw['payloads']), 'EXACT_PHYSICAL_READ_SET')
    files = {path: raw['payloads'][path] for path in paths}
    _check_payload_stats(files)
    return files


def october_native_metadata(launch, common):
    """Read bound metadata only; target leaves are role-specific before opening."""
    metadata = {}
    read = lambda ref: common._read(ref, metadata)
    bound = roles.require_role_binding(launch, read=read)
    batch, view, target = (bound[k] for k in ('batch', 'view', 'target'))
    role = bound['role']
    profile = units.october_prefix_units_profile()
    views._require_view(view, unobserved_units_profile=profile)
    fitted = read(view['prefix_normalization']['binding'])
    constructor = read(view['constructor_metadata']['binding'])
    _require(fitted['contract_sha256'] == view['prefix_normalization']['contract_sha256']
             and batch['fit_prefix']['rows'] == view['prefix']['rows'] == 29849
             and view['october_observer_axes'] == bound['axes'],
             'FIT_VIEW_BINDING')
    prior_result = read(bound['references']['fit_prior'])
    _require(prior_result['parents'] == 29849, 'FIT_PRIOR')
    raw = read(batch['ancestral_264_plan'])
    _require(raw['worker'] == batch['ancestral_264_worker']
             and raw['axes']['source'] == batch['full_axes']['source']
             and raw['axes']['rows'] == batch['full_axes']['rows'], 'PHYSICAL_PROVIDER')
    provider = prior._load_module(batch['ancestral_264_worker'], 'gx1_october_native_264_reader')
    info = provider.validate_metadata(raw)
    role_files = _role_payloads(raw, info, provider, read, metadata)
    info['metadata'].update(metadata)
    for ref in (batch['current_gc_transition'], batch['current_gc_layout']):
        read(ref)
    causal_manifest = read(batch['causal_inputs'])
    _require(causal_manifest['future_labels_included'] is False
             and causal_manifest['observer_axes'] == batch['observer_axes']
             and causal_manifest['target_source_series'] == batch['october_series'],
             'CAUSAL_INPUT_MANIFEST')
    target_refs, target_stats = _leaves(target, launch['target_manifest'], common)
    causal_refs, causal_stats = _leaves(causal_manifest, batch['causal_inputs'], common)
    info['metadata'].update(metadata)
    return {'october_role': role, 'october_arm': bound['arm'], 'october_bound': bound,
            'october_provider_plan': raw, 'october_provider': provider,
            'october_causal_refs': causal_refs, 'october_causal_stats': causal_stats,
            'october_causal_manifest': causal_manifest, 'dynamic_state_launch': launch,
            'state_target_manifest': target, 'state_array_refs': target_refs,
            'state_array_stats': target_stats, 'state_label_audit': {'prior': prior_result},
            'ancestor_plan': {**raw, 'payloads': role_files}, 'physical': _PageRelease, 'ancestor': None,
            'info': info, 'manifest': {'physical_provider_worker': batch['ancestral_264_worker']},
            'fitted': fitted, 'constructor': constructor}


def prepare_october_native(context, launch, np_module):
    """Map only the selected feature prefix, then open its exact target role."""
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    _require(context['dynamic_state_launch'] == launch and context['october_role'] == roles.SCOPE_ROLES[launch['run_scope']],
             'ROLE_CONTEXT')
    batch = context['october_bound']['batch']
    role = context['october_role']
    provider, raw, info = (context[k] for k in ('october_provider', 'october_provider_plan', 'info'))
    stop = 29849 if role == 'fit' else 36172
    with campaign.file_verification_scope():
        axes, full = provider.read_axes(raw, info, np_module)
        _require(full == batch['full_axes'], 'FULL_AXIS_IDENTITY')
        prior._axis_receipt(axes, full, batch['fit_prefix'], batch['evaluation_axis'])
        read_plan = dict(raw, october_gc_transition=batch['current_gc_transition'],
                         october_gc_layout=batch['current_gc_layout'])
        physical = prior._prepare_prefix_inputs(provider, read_plan, info, axes,
            fit_rows=stop, last_issue_ns=int(axes['decision_ns'][stop - 1]))
        tape = causal.load_october_causal_inputs(batch['causal_inputs'],
            observer_axes_ref=batch['observer_axes'], candidate_design_ref=batch['candidate_design'],
            target_series_ref=batch['october_series'], validation_role='fit' if role == 'fit' else 'all')
        series = state.load_dynamic_mtf_state_series(launch['target_manifest'],
            expected_source_refs=batch['target_source_refs'],
            expected_cutoff_ns=batch['fit_prefix']['cutoff_ns'] if role == 'fit' else batch['evaluation_axis']['cutoff_ns'])
        campaign.capture_verified_file_snapshot()
    part = slice(0, 29849) if role == 'fit' else slice(29849, 36172)
    selected = axes['parent_rows'][part]
    issues = axes['decision_ns'][part]
    _require(np_module.array_equal(series.arrays['parent_rows'][part], selected)
             and np_module.array_equal(series.arrays['issue_ns'][part], issues)
             and np_module.array_equal(tape.arrays['parent_rows'][part], selected)
             and np_module.array_equal(tape.arrays['issue_ns'][part], issues), 'ROLE_AXES')
    view = context['october_bound']['view']
    sequence = physical['entry_sequence_source']
    for value in (sequence['signal'], sequence['times_ns'], sequence['sample_positions'],
                  *physical['current'].values()):
        value.setflags(write=False)
    prepared = batch_owner._role_prepared(physical, axes, view, context['october_bound']['axes'], tape,
        role, part, units.october_prefix_units_profile())
    prepared['state_artifact_directory'] = context['state_artifact_directory']
    prepared['october_mmaps'] = [physical['entry_sequence_source']['signal'], *physical['current'].values()]
    if role == 'score':
        prepared['october_score_parent_rows'] = selected
    fit_axes = axes['parent_rows'][:29849]
    _require(hashlib.sha256(fit_axes.astype('<i8', copy=False).tobytes()).hexdigest()
             == view['prefix']['parent_rows_sha256'], 'FIT_ORDER_SOURCE')
    order = worker._fixed_order(fit_axes, len(fit_axes), worker.SEED, np_module)
    leads = np_module.asarray(view['research_objective']['lead_minutes'], np_module.int64)
    _require(len(leads) == 1440 and int(leads[0]) == 1 and int(leads[-1]) == 1440,
             'NATIVE_TARGET_LEADS')
    targets = {'dynamic_state_series': series, 'target_manifest': launch['target_manifest'],
               'target_sources': batch['target_source_refs'], 'prior': context['state_label_audit']['prior'],
               'lead_minutes': leads, 'train_cutoff_ns': batch['fit_prefix']['cutoff_ns'],
               'cutoff_ns': batch['fit_prefix']['cutoff_ns'] if role == 'fit' else batch['evaluation_axis']['cutoff_ns']}
    return prepared, targets, view, None, order
