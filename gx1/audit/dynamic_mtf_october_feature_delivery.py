"""One causal, model-free delivery audit of October M5 Entry features and age."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from gx1.audit import dynamic_mtf_october_input_only as prior
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes, _publish_file_noreplace, _fsync_directory


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_FEATURE_DELIVERY_20261106_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_feature_delivery_20261106_001'


def _require(ok, reason):
    if not ok:
        raise RuntimeError('DYNAMIC_M5_OCTOBER_DELIVERY_' + reason)


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


def _publish(root, name, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _require(len(body) <= 1048576, 'OUTPUT_SIZE:' + name)
    path = root / name
    _publish_bytes(path, body)
    return _ref(path)


def _save_causal_inputs(root, arrays, *, plan, axes):
    """Publish only issue-time state/age and axes, never future labels."""
    output = root / 'CAUSAL_INPUTS'
    _require(not output.exists(), 'INPUT_OUTPUT_EXISTS')
    stage = Path(tempfile.mkdtemp(prefix='.CAUSAL_INPUTS.staging.', dir=root))
    descriptors = {}
    for name, values in arrays.items():
        _require(isinstance(values, np.ndarray) and values.shape[0] == 36172
                 and values.dtype.kind in 'ibf' and np.isfinite(values).all(),
                 'CAUSAL_ARRAY:' + name)
        path = stage / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
        _require(np.array_equal(np.load(path, allow_pickle=False), values), 'CAUSAL_ROUNDTRIP:' + name)
        descriptors[name] = {'path': str(output / path.name), 'sha256': _sha(path),
                             'size_bytes': path.stat().st_size, 'dtype': values.dtype.name,
                             'shape': list(values.shape)}
    manifest = {'schema_version': 'gx1_dynamic_m5_october_causal_input_artifact_v1',
                'candidate_design': plan['candidate_design'],
                'target_source_series': plan['october_series'],
                'observer_axes': plan['observer_axes'],
                'fit_parent_rows': 29849, 'development_parent_rows': 6323,
                'cutoff_ns_exclusive': plan['evaluation_axis']['cutoff_ns'],
                'clock_basis': 'historical_event_time_only_online_delivery_not_qualified',
                'future_labels_included': False, 'arrays': descriptors,
                'parent_rows_sha256': hashlib.sha256(axes['parent_rows'].astype('<i8', copy=False).tobytes()).hexdigest(),
                'issue_ns_sha256': hashlib.sha256(axes['decision_ns'].astype('<i8', copy=False).tobytes()).hexdigest()}
    body = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _publish_bytes(stage / 'CAUSAL_INPUTS.json', body)
    (stage / 'CAUSAL_INPUTS.json').chmod(0o444)
    _fsync_directory(stage)
    _publish_file_noreplace(stage, output)
    _fsync_directory(root)
    return {'path': str(output / 'CAUSAL_INPUTS.json'), 'sha256': hashlib.sha256(body).hexdigest()}


def _raw_batch(view, observer, prepared, selected, *, role, profile):
    _require(role in ('fit', 'score'), 'ROLE')
    n = 29849
    part = slice(0, n) if role == 'fit' else slice(n, 36172)
    source = prepared['entry_sequence_source']
    current = prepared['current']
    return views.gather_native_direction_feature_batch(
        view, parent_rows=observer.parent_rows, bar_start_ns=observer.bar_start_ns,
        decision_ns=observer.decision_ns,
        entry_sequence_source={'signal': source['signal'], 'times_ns': source['times_ns'],
                               'sample_positions': source['sample_positions'][part],
                               'audit': source['audit']},
        snap_x=current['snap'][part], ctx_cont=current['ctx_cont'][part],
        ctx_cat=current['ctx_cat'][part], multi_tf_sources=prepared['mtf'],
        gc_surfaces=prepared['gc'], gc_layout=prepared['gc_layout'],
        batch_parent_rows=selected, observer_axes=observer,
        unobserved_units_profile=profile)


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
    _require(plan['schema_version'] == 'gx1_dynamic_m5_october_feature_delivery_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED' and plan['max_invocations'] == 1
             and plan['output_root'] == '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_FEATURE_DELIVERY_20261106_001'
             and plan['execution']['command'] ==
                 'bash scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- timeout --signal=TERM --kill-after=10s 1800s .venv/bin/python -m gx1.audit.dynamic_mtf_october_feature_delivery'
             and plan['restrictions'] == {'model_construction': False, 'optimizer_updates': False,
                 'test_split': False, 'broker_or_network': False, 'spending_usd': 0,
                 'post_october_outcomes': False}
             and isinstance(state, dict) and state.get('status') == 'ADMITTED_ONCE'
             and state.get('plan') == plan_ref and state.get('operator_approved') is True
             and state.get('remaining_invocations') == 1
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False
             and policy['full_epoch_training_allowed'] is False
             and policy['direction_edge_research_goal_20261105']['native_launch_authorized'] is False
             and policy['direction_edge_research_goal_20261105']['test_access_authorized'] is False
             and policy['direction_edge_research_goal_20261105']['broker_orders_or_trading_authorized'] is False,
             'EXPLICIT_ADMISSION_REQUIRED')
    _require(plan['standing_authorization'] == policy['no_cost_local_standing_authorization_20261105']['operator_instruction'],
             'STANDING_AUTHORIZATION')
    _require(all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()),
             'SOURCE_CHANGED')
    control = _read(plan['control_view'])
    candidate = _read(plan['candidate_view'])
    scope = _read(plan['observer_axes'])
    prior_plan = _read(plan['prior_input_plan'])
    prior_result = _read(plan['prior_input_result'])
    age_fit = _read(plan['fit_age_coverage'])
    age_oct = _read(plan['october_age_coverage'])
    _require(control['october_observer_axes'] == candidate['october_observer_axes'] == scope
             and control['prefix_normalization'] == candidate['prefix_normalization']
             and prior_result['fit_allowed'] is True and prior_result['prefix_normalization'] == plan['prefix_normalization']
             and prior_plan['full_axes'] == plan['full_axes']
             and age_fit['parents'] == 29849 and age_oct['parents'] == 6323,
             'QUALIFIED_METADATA')
    profile = units.october_prefix_units_profile()
    views._require_view(control, unobserved_units_profile=profile)
    views._require_view(candidate, unobserved_units_profile=profile)
    original = _read(plan['october_series'])
    _require(original['metadata']['parents'] == 36172 and original['metadata']['cutoff_ns'] ==
             plan['evaluation_axis']['cutoff_ns'], 'EXACT_TARGET_SERIES')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        provider_plan = _read(plan['ancestral_264_plan'])
        provider = prior._load_module(plan['ancestral_264_worker'], 'gx1_october_delivery_264_reader')
        with campaign.file_verification_scope():
            info = provider.validate_metadata(provider_plan)
            axes, full = provider.read_axes(provider_plan, info, np)
            _require(full == plan['full_axes'], 'FULL_AXIS_IDENTITY')
            prior._axis_receipt(axes, full, plan['fit_prefix'], plan['evaluation_axis'])
            read_plan = dict(provider_plan,
                october_gc_transition=plan['current_gc_transition'],
                october_gc_layout=plan['current_gc_layout'])
            prepared = prior._prepare_prefix_inputs(provider, read_plan, info, axes,
                fit_rows=36172, last_issue_ns=int(axes['decision_ns'][36171]))
            campaign.capture_verified_file_snapshot()
        for value in (prepared['entry_sequence_source']['signal'],
                      prepared['entry_sequence_source']['times_ns'],
                      prepared['entry_sequence_source']['sample_positions'],
                      *prepared['current'].values()):
            value.setflags(write=False)
        _require(prepared['current051_rows_readable'] == 36172
                 and len(prepared['entry_sequence_source']['sample_positions']) == 36172
                 and prepared['entry_sequence_source']['times_ns'][-1] == axes['bar_start_ns'][36171],
                 'PREFIX_ONLY_RAW_INPUTS')
        series = target.load_dynamic_mtf_state_series(plan['october_series'],
            expected_source_refs=plan['target_source_refs'],
            expected_cutoff_ns=plan['evaluation_axis']['cutoff_ns'])
        rows = np.ascontiguousarray(axes['parent_rows'][:36172])
        issues = np.ascontiguousarray(axes['decision_ns'][:36172])
        _require(np.array_equal(series.arrays['parent_rows'], rows)
                 and np.array_equal(series.arrays['issue_ns'], issues), 'INPUT_TARGET_AXES')
        current = target.query_historical_dynamic_mtf_confirmed_run_age(
            series, parent_rows=rows, issue_ns=issues,
            cutoff_ns=plan['evaluation_axis']['cutoff_ns'])
        encoded = target.encode_historical_dynamic_mtf_confirmed_run_age(current)
        _require(encoded.shape == (36172, 5, 2)
                 and np.array_equal(encoded[..., 1] > 0, current['current_confirmed_run_age_observed']),
                 'AGE_ENCODED_MASK')
        for col, tf in enumerate(target.TIMEFRAMES):
            for left, right, receipt in ((0, 29849, age_fit), (29849, 36172, age_oct)):
                expected = receipt['timeframes'][tf]
                _require(int(current['current_confirmed_run_age_observed'][left:right, col].sum()) == expected['confirmed_age_observed']
                         and int(current['current_observed'][left:right, col].sum()) == expected['current_state_observed']
                         and int((current['current_observed'][left:right, col] &
                                  current['current_path_observed'][left:right, col]).sum()) == expected['current_state_and_path_observed'],
                         'AGE_PRIOR_COVERAGE:' + tf)
        observers = {}
        for role, part in (('fit', slice(0, 29849)), ('score', slice(29849, 36172))):
            observers[role] = views.NativeDirectionObserverAxes(candidate, source=scope['source'],
                parent_rows=axes['parent_rows'][part], bar_start_ns=axes['bar_start_ns'][part],
                decision_ns=axes['decision_ns'][part], role=role,
                unobserved_units_profile=profile)
        feature_keys = {'seq_x', 'snap_x', 'ctx_cont', 'ctx_cat',
                        *(f'seq_{tf.lower()}' for tf in target.TIMEFRAMES),
                        *(f'gc_seq_{tf.lower()}' for tf in target.TIMEFRAMES)}
        fit_sample = np.asarray([rows[0], rows[1], rows[29847], rows[29848]], dtype=np.int64)
        fit_raw = _raw_batch(candidate, observers['fit'], prepared, fit_sample,
                             role='fit', profile=profile)
        _require(set(fit_raw['features']) == feature_keys, 'FIT_FEATURE_KEYS')
        score_rows = rows[29849:]
        count = 0
        first_batch = None
        for left in range(0, len(score_rows), 16):
            selected = score_rows[left:left + 16]
            raw = _raw_batch(candidate, observers['score'], prepared, selected,
                             role='score', profile=profile)
            _require(set(raw['features']) == feature_keys
                     and raw['witness']['parent_rows'] == selected.tolist()
                     and raw['witness']['raw_values_not_normalized'] is True,
                     'SCORE_FEATURE_DELIVERY')
            if first_batch is None:
                first_batch = raw
            count += len(selected)
        _require(count == 6323, 'EXACT_SCORE_ROWS')
        control_observer = views.NativeDirectionObserverAxes(control, source=scope['source'],
            parent_rows=axes['parent_rows'][29849:36172],
            bar_start_ns=axes['bar_start_ns'][29849:36172],
            decision_ns=axes['decision_ns'][29849:36172], role='score',
            unobserved_units_profile=profile)
        matched = _raw_batch(control, control_observer, prepared, score_rows[:16],
                             role='score', profile=profile)
        _require(all(np.array_equal(matched['features'][key], first_batch['features'][key])
                     for key in feature_keys), 'MATCHED_CONTROL_CANDIDATE_RAW_FEATURES')
        causal = {'parent_rows': rows, 'issue_ns': issues,
                  'current_state': np.ascontiguousarray(current['current_state']),
                  'current_observed': np.ascontiguousarray(current['current_observed']),
                  'current_path_observed': np.ascontiguousarray(current['current_path_observed']),
                  'current_confirmed_run_age_features': encoded}
        input_ref = _save_causal_inputs(root, causal, plan=plan,
            axes={'parent_rows': rows, 'decision_ns': issues})
        _require(_sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and _ref(PLAN) == plan_ref
                 and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = _publish(root, 'RESULT.json', {'schema_version': 'gx1_dynamic_m5_october_feature_delivery_result_v1',
            'plan': plan_ref, 'causal_inputs': input_ref, 'fit_sample_rows_gathered': len(fit_sample),
            'development_rows_gathered': count, 'feature_keys': sorted(feature_keys),
            'control_candidate_first_batch_raw_equal': True,
            'age_input_coverage_matches_prior': True,
            'source_rows_readable': prepared['source050_rows_readable'],
            'current_rows_readable': prepared['current051_rows_readable'],
            'future_labels_in_causal_artifact': False,
            'model_calls': 0, 'test_split_used': False, 'independent_edge': 'UNKNOWN'})
        _publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result,
            'exit_code': 0, 'elapsed_seconds': time.monotonic() - began,
            'source_unchanged': True, 'model_calls': 0,
            'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        _publish(root, 'FAILED.json', {'plan': plan_ref, 'error_type': type(exc).__name__,
            'error': str(exc)[:2048], 'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
