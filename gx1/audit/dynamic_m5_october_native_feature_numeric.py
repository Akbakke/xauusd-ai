"""One capped, model-free October native indicator clock and value audit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from gx1.audit import dynamic_m5_october_native_batch_binding as batch_owner
from gx1.audit import dynamic_mtf_october_feature_delivery as delivery
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.features.htf_features import MULTI_TF_SHIFT


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_native_feature_numeric_20261106_001'
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
SAMPLE_POSITIONS = (0, 1, 29847, 29848, 29849, 29850, 36170, 36171)
FIELDS = ('atr_bps_14', 'ema20_dist_atr', 'mtf_geometry_support_dist_atr',
          'mtf_geometry_resistance_dist_atr')
SOURCE_PATH = 'gx1/audit/dynamic_m5_october_native_feature_numeric.py'
OUTPUT_ROOT = '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_FEATURE_NUMERIC_20261106_001'
COMMAND = ('bash scripts/gx1_capped_run.sh --class audit --mem 4G --swap 512M -- '
           'timeout --signal=TERM --kill-after=10s 1800s .venv/bin/python -m '
           'gx1.audit.dynamic_m5_october_native_feature_numeric')
RESTRICTIONS = {'model_construction': False, 'model_forward': False,
                'optimizer_updates': False, 'test_split': False,
                'broker_or_network': False, 'spending_usd': 0,
                'post_october_outcomes': False}


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_NATIVE_FEATURE_NUMERIC_' + reason)


def _head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _clean():
    return subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == ''


def _source_hashes(plan):
    base = delivery._read(plan['base_plan'])
    expected = set(base['source_files_sha256']) | {SOURCE_PATH}
    hashes = plan['source_files_sha256']
    _require(set(hashes) == expected and hashes[SOURCE_PATH] == delivery._sha(Path(__file__)),
             'SOURCE_INVENTORY')
    _require(all(delivery._sha(REPO / path) == digest for path, digest in hashes.items()),
             'SOURCE_BYTES')
    return base


def _gather(prepared, view, selected, profile):
    return views.gather_native_direction_feature_batch(
        view, parent_rows=prepared['arrays']['parent_rows'],
        bar_start_ns=prepared['arrays']['bar_start_ns'],
        decision_ns=prepared['arrays']['decision_ns'],
        entry_sequence_source=prepared['entry_sequence_source'],
        snap_x=prepared['current']['snap'], ctx_cont=prepared['current']['ctx_cont'],
        ctx_cat=prepared['current']['ctx_cat'], multi_tf_sources=prepared['cache'],
        gc_surfaces=prepared['gc']['surfaces'], gc_layout=prepared['gc']['layout'],
        batch_parent_rows=selected, observer_axes=prepared['observer_axes'],
        unobserved_units_profile=profile)


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO and _clean()
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
                 == 'work/gx1-current', 'CLEAN_REPO')
    head = _head()
    plan_ref = delivery._ref(PLAN)
    plan = delivery._read(plan_ref)
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    stage = policy.get(POLICY_KEY)
    _require(plan['schema_version'] == 'gx1_dynamic_m5_october_native_feature_numeric_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED'
             and plan['base_plan'] == delivery._ref(batch_owner.PLAN)
             and plan['script_sha256'] == delivery._sha(Path(__file__))
             and plan['parent_range'] == [0, 36172]
             and plan['sample_parent_positions'] == list(SAMPLE_POSITIONS)
             and plan['timeframes'] == list(TIMEFRAMES)
             and plan['indicator_fields'] == list(FIELDS)
             and plan['output_root'] == OUTPUT_ROOT
             and plan['execution']['command'] == COMMAND
             and plan['execution']['class'] == 'audit'
             and plan['execution']['memory'] == '4G'
             and plan['execution']['swap'] == '512M'
             and plan['execution']['timeout_seconds'] == 1800
             and plan['execution']['kill_after_seconds'] == 10
             and plan['execution']['cost_usd'] == 0
             and plan['restrictions'] == RESTRICTIONS
             and plan['max_invocations'] == 1
             and isinstance(stage, dict) and stage.get('status') == 'ADMITTED_ONCE'
             and stage.get('plan') == plan_ref
             and stage.get('remaining_invocations') == 1
             and stage.get('model_calls_authorized') == 0
             and stage.get('test_data_used') is False
             and policy['training_enabled'] is False
             and policy['full_val_allowed'] is False,
             'ADMISSION')
    _require(plan['standing_authorization'] ==
             policy['no_cost_local_standing_authorization_20261105']['operator_instruction'],
             'STANDING_AUTHORIZATION')
    base = _source_hashes(plan)
    _require(base['sample_parent_positions'] == {'fit': [0, 1, 29847, 29848],
             'development': [29849, 29850, 36170, 36171]}, 'BASE_SCOPE')
    candidate = delivery._read(base['candidate_view'])
    scope = delivery._read(base['observer_axes'])
    profile = units.october_prefix_units_profile()
    views._require_view(candidate, unobserved_units_profile=profile)
    _require(candidate['full_feature_profile'] is True
             and candidate['october_observer_axes'] == scope, 'CANDIDATE_VIEW')
    root = Path(OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        provider_plan = delivery._read(base['ancestral_264_plan'])
        provider = delivery.prior._load_module(base['ancestral_264_worker'],
                                               'gx1_october_numeric_264_reader')
        with campaign.file_verification_scope():
            info = provider.validate_metadata(provider_plan)
            axes, full = provider.read_axes(provider_plan, info, np)
            _require(full == base['full_axes'], 'FULL_AXES')
            delivery.prior._axis_receipt(axes, full, base['fit_prefix'],
                                         base['evaluation_axis'])
            read_plan = dict(provider_plan, october_gc_transition=base['current_gc_transition'],
                             october_gc_layout=base['current_gc_layout'])
            raw = delivery.prior._prepare_prefix_inputs(
                provider, read_plan, info, axes, fit_rows=36172,
                last_issue_ns=int(axes['decision_ns'][36171]))
            campaign.capture_verified_file_snapshot()
        for value in (raw['entry_sequence_source']['signal'],
                      raw['entry_sequence_source']['times_ns'],
                      raw['entry_sequence_source']['sample_positions'],
                      *raw['current'].values()):
            value.setflags(write=False)
        starts = np.asarray(axes['bar_start_ns'][:36172], dtype=np.int64)
        decisions = np.asarray(axes['decision_ns'][:36172], dtype=np.int64)
        parents = np.asarray(axes['parent_rows'][:36172], dtype=np.int64)
        _require(len(starts) == len(decisions) == len(parents) == 36172
                 and np.array_equal(decisions, starts + 300_000_000_000),
                 'EXACT_PARENT_CLOCK')
        selections = {}
        census = {}
        for tf in TIMEFRAMES:
            frame = raw['mtf'][tf]
            ts = np.asarray(frame.attrs['ts_int64'])
            matrix = np.asarray(frame.attrs['feats_np'])
            warmup = int(frame.attrs['causal_warmup_rows'])
            shift = int(MULTI_TF_SHIFT[tf].value)
            _require(ts.dtype == np.int64 and matrix.dtype == np.float32
                     and matrix.shape == (len(ts), len(frame.columns))
                     and 0 <= warmup < len(ts), 'CACHE_HEADER:' + tf)
            cutoff = decisions - shift
            ids = np.searchsorted(ts, cutoff, side='right') - 1
            _require(np.all(ids >= warmup) and np.all(ids < len(ts))
                     and np.all(ts[ids] + shift <= decisions)
                     and np.all((ids + 1 == len(ts)) |
                                (ts[np.minimum(ids + 1, len(ts) - 1)] > cutoff)),
                     'LATEST_CLOSED:' + tf)
            if tf == 'M5':
                _require(np.array_equal(ts[ids], starts), 'M5_PARENT_CACHE_BARSTART')
            selections[tf] = ids
            fields = {}
            for name in FIELDS:
                _require(name in frame.columns, 'INDICATOR_FIELD:' + tf + ':' + name)
                column = frame.columns.get_loc(name)
                values = matrix[ids, column]
                _require(np.isfinite(values).all(), 'INDICATOR_FINITE:' + tf + ':' + name)
                fields[name] = {'finite_count': int(np.isfinite(values).sum()),
                    'distinct_count': int(np.unique(values).size),
                    'minimum': float(values.min()), 'maximum': float(values.max()),
                    'negative_count': int((values < 0).sum()),
                    'zero_count': int((values == 0).sum()),
                    'positive_count': int((values > 0).sum())}
            census[tf] = {'cache_rows': len(ts), 'warmup_rows': warmup,
                          'latest_closed_parent_count': len(ids),
                          'first_selected_cache_start_ns': int(ts[ids[0]]),
                          'last_selected_cache_start_ns': int(ts[ids[-1]]),
                          'fields': fields}
        sampled = {}
        for role, part, positions in (
            ('fit', slice(0, 29849), SAMPLE_POSITIONS[:4]),
            ('score', slice(29849, 36172), SAMPLE_POSITIONS[4:])):
            selected = np.ascontiguousarray(parents[list(positions)])
            prepared = batch_owner._role_prepared(raw, axes, candidate, scope,
                                                   None, role, part, profile)
            gathered = _gather(prepared, candidate, selected, profile)
            _require(gathered['witness']['parent_rows'] == selected.tolist()
                     and gathered['witness']['raw_values_not_normalized'] is True,
                     'GATHER_WITNESS:' + role)
            for local, position in enumerate(positions):
                item = {'parent_row': int(parents[position]),
                        'decision_ns': int(decisions[position]), 'timeframes': {}}
                for tf in TIMEFRAMES:
                    witness = gathered['witness']['multi_tf'][tf][local]
                    actual_id = int(witness['native_row_indices'][-1])
                    expected_id = int(selections[tf][position])
                    frame = raw['mtf'][tf]
                    _require(actual_id == expected_id
                             and int(witness['available_ns'][-1]) ==
                                 int(frame.attrs['ts_int64'][expected_id]) +
                                 int(MULTI_TF_SHIFT[tf].value),
                             'ACTUAL_GATHER_ASOF:' + role + ':' + tf)
                    actual = gathered['features']['seq_' + tf.lower()][local, -1]
                    expected = frame.attrs['feats_np'][expected_id]
                    _require(actual.dtype == np.float32 and np.isfinite(actual).all(),
                             'ACTUAL_GATHER_FINITE:' + role + ':' + tf)
                    indicator_values = {}
                    for name in FIELDS:
                        column = frame.columns.get_loc(name)
                        one = np.asarray(actual[column], dtype=np.float32)
                        two = np.asarray(expected[column], dtype=np.float32)
                        _require(one.view(np.uint32).item() == two.view(np.uint32).item(),
                                 'TENSOR_CACHE_PARITY:' + role + ':' + tf + ':' + name)
                        indicator_values[name] = float(one)
                    item['timeframes'][tf] = {
                        'cache_row': expected_id,
                        'cache_start_ns': int(frame.attrs['ts_int64'][expected_id]),
                        'available_ns': int(witness['available_ns'][-1]),
                        'raw_indicator_values': indicator_values}
                sampled[str(position)] = item
        _require(delivery._sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and delivery._ref(PLAN) == plan_ref and _head() == head and _clean(),
                 'SOURCE_OR_POLICY_CHANGED')
        _source_hashes(plan)
        result = delivery._publish(root, 'RESULT.json', {
            'schema_version': 'gx1_dynamic_m5_october_native_feature_numeric_result_v1',
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'plan': plan_ref, 'parent_count': len(parents),
            'timeframes': list(TIMEFRAMES), 'indicator_fields': list(FIELDS),
            'asof_and_numeric_census': census, 'sample_tensor_cache_parity': sampled,
            'evidence_scope': 'raw MTF cache as-of selection and raw cache-to-gather tensor parity only',
            'not_measured': ['post_normalization_values', 'family_projection',
                'model_forward', 'feature_influence', 'support_resistance_level_observation'],
            'model_constructions': 0, 'model_forwards': 0, 'optimizer_updates': 0,
            'test_data_used': False, 'independent_edge': 'UNKNOWN',
            'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'elapsed_seconds': time.monotonic() - started})
        delivery._publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result,
            'exit_code': 0, 'source_unchanged': True,
            'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        delivery._publish(root, 'FAILED.json', {'plan': plan_ref,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
