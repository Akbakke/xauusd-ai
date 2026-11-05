"""One admitted, target-free October M5 Entry input qualification.

The old 264/267 workers supply only their verified read-only input adapters.
Their completed plans are never launched. No model, target or score is loaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from collections.abc import Mapping


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_input_only_20261105_001'


def _require(ok, reason):
    if not ok:
        raise RuntimeError('DYNAMIC_M5_OCTOBER_INPUT_ONLY_' + reason)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ref(path):
    return {'path': str(path), 'sha256': _sha_file(path)}


def _read_ref(ref):
    _require(isinstance(ref, Mapping) and set(ref) == {'path', 'sha256'}
             and _ref(ref['path']) == ref, 'SOURCE_REF')
    return json.loads(Path(ref['path']).read_text())


def _load_module(ref, name):
    _require(_ref(ref['path']) == ref, 'LOADER_SOURCE')
    spec = importlib.util.spec_from_file_location(name, ref['path'])
    _require(spec is not None and spec.loader is not None, 'LOADER_SPEC')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(_ref(ref['path']) == ref, 'LOADER_SOURCE_CHANGED')
    return module


def _admission(plan, policy, plan_ref):
    _require(plan.get('schema_version') == 'gx1_dynamic_m5_october_input_only_plan_v1'
             and plan.get('status') == 'PREPARED_NOT_ADMITTED'
             and plan.get('plan_id') == 'DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_001'
             and plan.get('max_invocations') == 1
             and plan.get('output_root') == '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_001',
             'PLAN_SCOPE')
    _require(policy.get('training_enabled') is False and policy.get('full_val_allowed') is False
             and policy.get('full_epoch_training_allowed') is False, 'POLICY_TRAINING_GATE')
    goal = policy['direction_edge_research_goal_20261105']
    _require(goal.get('training_enabled') is False and goal.get('native_launch_authorized') is False
             and goal.get('test_access_authorized') is False
             and goal.get('broker_orders_or_trading_authorized') is False, 'GOAL_GATES')
    state = policy.get(POLICY_KEY)
    _require(isinstance(state, Mapping) and state.get('status') == 'ADMITTED_ONCE'
             and state.get('plan') == plan_ref and state.get('operator_approved') is True
             and state.get('remaining_invocations') == 1, 'EXPLICIT_ADMISSION_REQUIRED')
    _require(plan['restrictions'] == {'target_arrays': False, 'model_construction': False,
             'model_forward': False, 'optimizer_updates': False, 'test': False,
             'broker_or_network': False, 'orders_or_trading': False, 'spending_usd': 0},
             'RESTRICTIONS')
    return state


def _axis_receipt(axes, full, fit, eval_spec):
    import numpy as np
    _require(set(axes) == {'parent_rows', 'bar_start_ns', 'decision_ns'}
             and set(full) == {'source', 'rows', 'parent_rows_sha256', 'bar_start_ns_sha256', 'decision_ns_sha256'},
             'AXES_KEYS')
    decisions = axes['decision_ns']
    left = int(np.searchsorted(decisions, fit['cutoff_ns'], side='left'))
    right = int(np.searchsorted(decisions, eval_spec['cutoff_ns'], side='left'))
    _require(left == fit['rows'] == eval_spec['parent_row_range'][0]
             and right == eval_spec['parent_row_range'][1]
             and right - left == eval_spec['rows']
             and 0 < left < right <= len(decisions), 'EXACT_OCTOBER_GEOMETRY')
    def hashes(a, b):
        return {name + '_sha256': hashlib.sha256(
            np.ascontiguousarray(axes[name][a:b], dtype='<i8').tobytes()).hexdigest()
            for name in axes}
    _require(all(hashes(0, left)[key] == fit[key] for key in hashes(0, left)),
             'EXACT_FIT_HASHES')
    _require(np.array_equal(axes['bar_start_ns'][left:right] + 300_000_000_000,
                            axes['decision_ns'][left:right])
             and np.all(axes['decision_ns'][left:right] >= fit['cutoff_ns'])
             and np.all(axes['decision_ns'][left:right] < eval_spec['cutoff_ns']),
             'EVALUATION_CLOCK')
    return {'full_axes': full, 'fit_prefix': dict(fit),
            'evaluation': {'parent_row_range': [left, right], 'rows': right-left,
                           'cutoff_ns_exclusive': eval_spec['cutoff_ns'],
                           'first_decision_ns': int(decisions[left]),
                           'last_decision_ns': int(decisions[right-1]), **hashes(left, right)},
            'target_members_decoded': 0, 'evaluation_outcomes_decoded': 0}


def _prepare_prefix_inputs(provider, provider_plan, info, axes, *, fit_rows, last_issue_ns):
    """Map only feature prefixes; later rows contribute clock hashes alone."""
    import numpy as np
    from gx1.audit.dynamic_mtf_december_normalization import _history_views
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times

    source_times = load_m1_feature_surface_times(
        Path(info['source']['output_parquet']), context='OCTOBER_PREFIX_CLOCK_ONLY',
        expected_bar_seconds=300).asi8
    groups = provider.map_reuse(info['reuse'], np)
    positions = np.searchsorted(source_times, axes['bar_start_ns'][:fit_rows]).astype(np.int64, copy=False)
    _require(len(positions) == fit_rows and np.all(positions >= 95)
             and np.all(positions < len(source_times))
             and np.all(positions[1:] > positions[:-1])
             and np.array_equal(source_times[positions], axes['bar_start_ns'][:fit_rows]),
             'EXACT_PREFIX_SOURCE_POINTERS')
    current, source = groups['current051'], groups['source050']
    for start in range(0, fit_rows, 64):
        end = min(start + 64, fit_rows)
        _require(np.array_equal(current['snap'][start:end], source['signal'][positions[start:end]]),
                 'PREFIX_SNAPSHOT_PARITY')
    stop = int(positions[-1]) + 1
    mtf, gc, layout = _history_views(provider_plan,
        provider_plan['october_gc_transition'], provider_plan['october_gc_layout'],
        last_issue_ns, native_data_role=True)
    return {'entry_sequence_source': {'signal': source['signal'][:stop],
                'times_ns': source_times[:stop], 'sample_positions': positions,
                'audit': info['audit']},
            'current': {name: values[:fit_rows] for name, values in current.items()},
            'mtf': mtf, 'gc': gc, 'gc_layout': layout,
            'source050_rows_readable': stop, 'current051_rows_readable': fit_rows}


def _publish(root, name, value, *, max_bytes):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _require(len(raw) <= max_bytes, 'OUTPUT_SIZE:' + name)
    path = root / name
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    return _ref(path)


def run():
    import numpy as np
    from gx1.audit.dynamic_mtf_train_prefix_inputs import _check_payload_stats
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.contracts import entry_native_direction_prefix_normalization_v1 as prefix
    from gx1.contracts import entry_native_direction_prefix_units_v1 as units

    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    _require(Path.cwd().resolve() == REPO and subprocess.check_output(
        ['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
        and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '', 'CLEAN_REPO')
    plan_ref = _ref(PLAN)
    plan = json.loads(PLAN.read_text())
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    _admission(plan, policy, plan_ref)
    _require(plan['profile'] == _ref(REPO / units.PLAN_PATH)
             and plan['fit_prefix'] == units.october_prefix_units_profile()['fit_prefix'],
             'EXACT_UNITS_PROFILE')
    for path, digest in plan['source_files_sha256'].items():
        _require(_sha_file(REPO / path) == digest, 'CURRENT_SOURCE:' + path)
    provider_plan = _read_ref(plan['ancestral_264_plan'])
    ancestral_267 = _read_ref(plan['ancestral_267_plan'])
    _require(ancestral_267['input_provider_plan'] == plan['ancestral_264_plan']
             and ancestral_267['input_provider_worker'] == plan['ancestral_264_worker']
             and ancestral_267['worker'] == plan['ancestral_267_worker']
             and ancestral_267['payloads'] == provider_plan['payloads'],
             'ANCESTRAL_READER_JOIN')
    old_support = _read_ref(plan['prior_october_support'])
    _require(old_support['statistics_fits'] == 0 and old_support['fit_allowed'] is False
             and len(old_support['blockers']) == 153, 'PRIOR_SUPPORT_SCOPE')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)  # A failed or interrupted claim cannot be relaunched.
    began = time.monotonic()
    try:
        provider = _load_module(plan['ancestral_264_worker'], 'gx1_october_264_reader')
        with campaign.file_verification_scope():
            info = provider.validate_metadata(provider_plan)
            _check_payload_stats(provider_plan['payloads'])
            axes, full = provider.read_axes(provider_plan, info, np)
            _require(full == plan['full_axes'], 'FULL_AXIS_IDENTITY')
            axis = _axis_receipt(axes, full, plan['fit_prefix'], plan['evaluation_axis'])
            fit_rows = plan['fit_prefix']['rows']
            source_plan = dict(provider_plan,
                october_gc_transition=plan['current_gc_transition'],
                october_gc_layout=plan['current_gc_layout'])
            prepared = _prepare_prefix_inputs(provider, source_plan, info, axes,
                fit_rows=fit_rows, last_issue_ns=int(axes['decision_ns'][fit_rows - 1]))
            campaign.capture_verified_file_snapshot()
            axis_ref = _publish(root, 'OCTOBER_AXIS.json', axis,
                                max_bytes=plan['max_json_output_bytes'])
            spec = units.october_prefix_units_profile()
        report_source = {'current_plan': plan_ref, 'ancestral_264_plan': plan['ancestral_264_plan'],
                         'ancestral_267_plan': plan['ancestral_267_plan'],
                         'prior_october_support': plan['prior_october_support']}
        q = prefix.qualify_native_direction_prefix_population(
            parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
            decision_ns=axes['decision_ns'], expected_axes=full,
            cutoff_ns=spec['fit_prefix']['cutoff_ns'],
            expected_prefix_parent_rows=axes['parent_rows'][:spec['fit_prefix']['rows']],
            entry_sequence_source=prepared['entry_sequence_source'],
            snap_x=prepared['current']['snap'], ctx_cont=prepared['current']['ctx_cont'],
            ctx_cat=prepared['current']['ctx_cat'], ordered_signal_names=info['source']['feature_field_order'],
            multi_tf_sources=prepared['mtf'], gc_surfaces=prepared['gc'],
            gc_layout=prepared['gc_layout'], base_lineage=provider_plan['base_lineage'],
            source_bindings=report_source, unobserved_units_profile=spec)
        report = q['report']
        _require(report['support'] == old_support['support'], 'SAME_GENUINE_SUPPORT')
        q_ref = _publish(root, 'QUALIFICATION.json', report,
                         max_bytes=plan['max_json_output_bytes'])
        fitted_ref = None
        if report['fit_allowed']:
            fitted = prefix.fit_native_direction_prefix_normalization(q)
            prefix.require_october_units_normalization_bindings(fitted, spec)
            fitted_ref = _publish(root, 'PREFIX_NORMALIZATION.json', fitted,
                                  max_bytes=plan['max_json_output_bytes'])
        _check_payload_stats(provider_plan['payloads'])
        _require(prepared['current051_rows_readable'] == plan['fit_prefix']['rows']
                 and prepared['source050_rows_readable'] <= len(prepared['entry_sequence_source']['times_ns']),
                 'PREFIX_ONLY_FEATURE_ACCESS')
        _require(hashlib.sha256(POLICY.read_bytes()).digest() == hashlib.sha256(policy_bytes).digest(),
                 'POLICY_CHANGED')
        for path, digest in plan['source_files_sha256'].items():
            _require(_sha_file(REPO / path) == digest, 'SOURCE_CHANGED:' + path)
        for key in ('ancestral_264_plan', 'ancestral_264_worker', 'ancestral_267_plan',
                    'ancestral_267_worker', 'prior_october_support', 'current_gc_layout',
                    'current_gc_transition'):
            _require(_ref(plan[key]['path']) == plan[key], 'REF_CHANGED:' + key)
        _require(_ref(PLAN) == plan_ref and subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip() == head
            and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '',
            'SOURCE_WORKTREE_CHANGED')
        result = {'schema_version': 'gx1_dynamic_m5_october_input_only_result_v1',
                  'plan': plan_ref, 'axis': axis_ref, 'qualification': q_ref,
                  'prefix_normalization': fitted_ref,
                  'fit_allowed': report['fit_allowed'], 'remaining_blockers': report['blockers'],
                  'declared_unit_support_count': len(report.get('declared_unit_support', [])),
                  'genuine_input_only': True, 'target_arrays_decoded': 0, 'model_calls': 0,
                  'independent_edge': 'UNKNOWN'}
        result_ref = _publish(root, 'RESULT.json', result, max_bytes=plan['max_json_output_bytes'])
        _publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result_ref,
            'exit_code': 0, 'source_unchanged': True, 'elapsed_seconds': time.monotonic() - began,
            'target_arrays_decoded': 0, 'model_calls': 0, 'no_relaunch_under_this_plan': True},
            max_bytes=plan['max_json_output_bytes'])
        return result_ref
    except Exception as exc:
        _publish(root, 'FAILED.json', {'plan': plan_ref, 'error_type': type(exc).__name__,
            'error': str(exc)[:2048], 'no_relaunch_under_this_plan': True},
            max_bytes=plan['max_json_output_bytes'])
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
