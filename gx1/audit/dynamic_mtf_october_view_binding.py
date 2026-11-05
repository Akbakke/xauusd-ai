"""Bind two metadata-only native views to one strict October development axis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import BATCH_SIZE


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_VIEW_BINDING_20261106_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_view_binding_20261106_001'


def _require(ok, reason):
    if not ok:
        raise RuntimeError('DYNAMIC_M5_OCTOBER_VIEW_' + reason)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ref(path):
    return {'path': str(path), 'sha256': _sha(path)}


def _read(ref):
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
             and _ref(ref['path']) == ref, 'SOURCE_REF')
    return json.loads(Path(ref['path']).read_text())


def _axis_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def _publish(root, name, value):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _require(len(body) <= 1048576, 'OUTPUT_SIZE:' + name)
    path = root / name
    _publish_bytes(path, body)
    return _ref(path)


def _axes(manifest, directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        selected = {}
        for name in ('parent_rows', 'issue_ns'):
            item = manifest['arrays'][name]
            selected[name] = target._load_verified_cache_npy(fd, name + '.npy',
                expected_sha256=item['sha256'], expected_size_bytes=item['size_bytes'],
                label='OCTOBER_VIEW_AXIS/' + name)
            _require(selected[name].dtype == np.int64 and list(selected[name].shape) == item['shape'],
                     'AXIS_HEADER:' + name)
    finally:
        os.close(fd)
    return selected


def _scope(plan, rows, issue):
    _require(len(rows) == len(issue) == 36172 and np.all(np.diff(rows) > 0)
             and np.all(np.diff(issue) > 0) and np.all(issue < plan['evaluation_axis']['cutoff_ns_exclusive']),
             'EXACT_AXIS_SIZE_CLOCK')
    start = issue - 5 * target.MINUTE_NS
    axes = {'parent_rows': rows, 'bar_start_ns': start, 'decision_ns': issue}
    fit, held = plan['fit_prefix'], plan['evaluation_axis']
    n = fit['rows']
    _require(n == held['parent_row_range'][0] and held['parent_row_range'][1] == len(rows),
             'EXACT_PARTITION')
    def hashes(left, right):
        return {name + '_sha256': _axis_sha(value[left:right]) for name, value in axes.items()}
    _require(all(hashes(0, n)[key] == fit[key] for key in hashes(0, n))
             and all(hashes(n, len(rows))[key] == held[key] for key in hashes(n, len(rows)))
             and int(issue[n]) == held['first_decision_ns']
             and int(issue[-1]) == held['last_decision_ns']
             and int(issue[n - 1]) < fit['cutoff_ns'] <= int(issue[n]),
             'EXACT_PARTITION_HASHES')
    return {'source': plan['axis_receipt'], 'rows': len(rows), **hashes(0, len(rows)),
            'fit': {'rows': n, 'cutoff_ns': fit['cutoff_ns'], **hashes(0, n)},
            'held_out': {'rows': held['rows'], 'cutoff_ns': held['cutoff_ns_exclusive'],
                         **hashes(n, len(rows))}}


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '', 'CLEAN_REPO')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    plan_ref = _ref(PLAN)
    plan = _read(plan_ref)
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    state = policy.get(POLICY_KEY)
    _require(plan['schema_version'] == 'gx1_dynamic_m5_october_view_binding_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED' and plan['max_invocations'] == 1
             and plan['output_root'] == '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_VIEW_BINDING_20261106_001'
             and plan['execution']['command'] ==
                 'bash scripts/gx1_capped_run.sh --class audit --mem 2G --swap 512M -- timeout --signal=TERM --kill-after=10s 600s .venv/bin/python -m gx1.audit.dynamic_mtf_october_view_binding'
             and plan['restrictions'] == {'feature_arrays': False, 'outcome_arrays': False,
                 'model_construction': False, 'optimizer_updates': False, 'test_split': False,
                 'broker_or_network': False, 'spending_usd': 0}
             and isinstance(state, dict) and state.get('status') == 'ADMITTED_ONCE'
             and state.get('plan') == plan_ref and state.get('operator_approved') is True
             and state.get('remaining_invocations') == 1
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False,
             'EXPLICIT_ADMISSION_REQUIRED')
    _require(plan['standing_authorization'] == policy['no_cost_local_standing_authorization_20261105']['operator_instruction'],
             'STANDING_AUTHORIZATION')
    _require(all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()),
             'SOURCE_CHANGED')
    old_manifest = _read(plan['historical_input_manifest'])
    constructor = _read(plan['constructor_metadata'])
    candidate_design = _read(plan['candidate_design'])
    baseline = _read(plan['historical_v5_view'])
    fitted = _read(plan['prefix_normalization'])
    axis_receipt = _read(plan['axis_receipt'])
    target_result = _read(plan['october_target_result'])
    series_manifest = target.read_dynamic_mtf_state_series_metadata(
        plan['october_series'], expected_source_refs=plan['target_source_refs'],
        expected_cutoff_ns=plan['evaluation_axis']['cutoff_ns_exclusive'])
    _require(target_result['series'] == plan['october_series']
             and target_result['fit_parent_rows'] == plan['fit_prefix']['rows']
             and target_result['development_parent_rows'] == plan['evaluation_axis']['rows']
             and axis_receipt['fit_prefix'] == plan['fit_prefix']
             and axis_receipt['evaluation'] == plan['evaluation_axis']
             and candidate_design['population']['axis_receipt'] == plan['axis_receipt']
             and old_manifest['constructor_metadata'] == plan['constructor_metadata']
             and fitted['prefix_qualification']['proof']['prefix']['rows'] == plan['fit_prefix']['rows']
             and baseline['current_state_first_loss_weight'] == plan['numerical_recipe']['current_state_first_loss_weight']
             and baseline['first_flat_activity_weight'] == plan['numerical_recipe']['first_flat_activity_weight']
             and baseline['native_endpoint_loss'] is True
             and BATCH_SIZE == plan['max_batch_rows'] == 16,
             'REGISTERED_METADATA')
    provenance = fitted['prefix_qualification']['proof']['source_bindings']
    _require(plan['source_bindings_control']['raw_input_provider_plan'] == provenance['ancestral_264_plan']
             and plan['source_bindings_control']['october_input_plan'] == provenance['current_plan']
             and plan['source_bindings_control']['input_qualification']['path'] ==
                 str(Path(plan['prefix_normalization']['path']).with_name('QUALIFICATION.json')),
             'OCTOBER_INPUT_PROVENANCE')
    for ref in plan['source_bindings_control'].values():
        _read(ref) if ref['path'].endswith('.json') else _require(_ref(ref['path']) == ref, 'CODE_REF')
    _require(plan['source_bindings_candidate'] == {**plan['source_bindings_control'],
                 'confirmed_age_candidate_design': plan['candidate_design']}, 'MATCHED_CANDIDATE_SOURCES')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        axes = _axes(series_manifest, Path(plan['october_series']['path']).parent)
        scope = _scope(plan, axes['parent_rows'], axes['issue_ns'])
        profile = units.october_prefix_units_profile()
        _require(profile == fitted['unobserved_units_profile'], 'EXACT_OCTOBER_PROFILE')
        weights = plan['numerical_recipe']
        args = {'prefix_normalization': fitted, 'prefix_normalization_ref': plan['prefix_normalization'],
                'constructor_metadata': constructor, 'constructor_metadata_ref': plan['constructor_metadata'],
                'max_batch_rows': BATCH_SIZE, 'full_feature_profile': True,
                'dynamic_state_supervision': True, 'native_endpoint_loss': True,
                'current_state_first_loss_weight': weights['current_state_first_loss_weight'],
                'first_flat_activity_weight': weights['first_flat_activity_weight'],
                'unobserved_units_profile': profile, 'october_observer_axes': scope}
        control = views.bind_native_direction_feature_view(**args,
            source_bindings=plan['source_bindings_control'])
        candidate = views.bind_native_direction_feature_view(**args,
            source_bindings=plan['source_bindings_candidate'], confirmed_run_age_input=True)
        _require(all(control[key] == candidate[key] for key in
                 ('prefix_normalization', 'prefix', 'full_axes', 'entry_execution_profile',
                  'physical_field_names_sha256', 'physical_widths', 'october_observer_axes'))
                 and control['research_objective']['contract_sha256'] != candidate['research_objective']['contract_sha256']
                 and candidate['confirmed_run_age_input'] is True
                 and 'confirmed_run_age_input' not in control,
                 'MATCHED_VIEW_IDENTITY')
        scope_ref = _publish(root, 'OCTOBER_OBSERVER_AXES.json', scope)
        control_ref = _publish(root, 'CONTROL_VIEW.json', control)
        candidate_ref = _publish(root, 'CANDIDATE_VIEW.json', candidate)
        _require(_sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and _ref(PLAN) == plan_ref
                 and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = _publish(root, 'RESULT.json', {'schema_version': 'gx1_dynamic_m5_october_view_binding_result_v1',
            'plan': plan_ref, 'observer_axes': scope_ref, 'control_view': control_ref,
            'candidate_view': candidate_ref, 'fit_rows': scope['fit']['rows'],
            'development_rows': scope['held_out']['rows'], 'source_rows_read': 0,
            'target_axis_arrays_read': ['parent_rows', 'issue_ns'], 'outcome_arrays_read': 0,
            'feature_arrays_read': 0, 'model_calls': 0, 'test_split_used': False,
            'independent_edge': 'UNKNOWN'})
        _publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result,
            'exit_code': 0, 'elapsed_seconds': time.monotonic() - started,
            'source_unchanged': True, 'model_calls': 0,
            'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        _publish(root, 'FAILED.json', {'plan': plan_ref,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
