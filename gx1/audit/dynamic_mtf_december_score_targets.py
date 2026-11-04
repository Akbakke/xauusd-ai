"""One separately admitted, post-fit publication of the December-only target and closure tapes."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from gx1.audit import dynamic_mtf_train_prefix_targets as prefix
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.features.htf_features import _load_verified_cache_npy


STAGE = 'dynamic_mtf_december_score_targets_20261104_001'
PARENT_KEYS = {'parent_rows', 'issue_ns', 'issue_source_index'}


def _require(ok, message):
    if not ok:
        raise RuntimeError('DECEMBER_SCORE_TARGET_' + message)


def _load_immutable_slice(manifest, directory, fit_scope, reserved_scope, source_limit, output_limit):
    descriptors = manifest['arrays']
    _require(set(descriptors) == set(target.SERIES_ARRAY_DTYPES)
             and sum(x['size_bytes'] for x in descriptors.values()) == source_limit, 'SOURCE_BYTE_LIMIT')
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        def load(name):
            item = descriptors[name]
            _require(item['path'] == str(directory / (name + '.npy')), 'ARRAY_PATH')
            a = _load_verified_cache_npy(fd, name + '.npy', expected_sha256=item['sha256'],
                                         expected_size_bytes=item['size_bytes'], label='DECEMBER_SCORE/' + name)
            _require(a.dtype.name == item['dtype'] == target.SERIES_ARRAY_DTYPES[name]
                     and list(a.shape) == item['shape'] and not a.flags.writeable, 'SOURCE_ARRAY:' + name)
            return a

        # Inspect only the two independent clocks before selecting any state or price-derived array.
        parent_from, parent_to = fit_scope['rows'], fit_scope['rows'] + reserved_scope['rows']
        original_issues = load('issue_ns')
        _require(original_issues.searchsorted(fit_scope['cutoff_ns']) == parent_from
                 and original_issues.searchsorted(reserved_scope['cutoff_ns']) == parent_to, 'PARENT_CUTS')
        clock = load('validity_ns')
        left, right = (int(clock.searchsorted(s['cutoff_ns'], side='left')) for s in (fit_scope,reserved_scope))
        _require(0 < left < right < len(clock), 'SOURCE_INTERVAL')
        size = right - left
        expected_bytes = 0
        for key, desc in descriptors.items():
            shape = (reserved_scope['rows'] if key in PARENT_KEYS else size, *desc['shape'][1:])
            # The native producer uses np.save and fixed NPY v1 simple numeric headers.
            head = io.BytesIO()
            dt = np.dtype(desc['dtype'])
            np.lib.format.write_array_header_1_0(head, {'descr':dt.str,'fortran_order':False,'shape':shape})
            expected_bytes += head.tell() + int(np.prod(shape)) * dt.itemsize
        _require(expected_bytes <= output_limit, 'OUTPUT_BYTE_LIMIT')
        arrays = {}
        for name in target.SERIES_ARRAY_DTYPES:
            a = original_issues if name == 'issue_ns' else clock if name == 'validity_ns' else load(name)
            row_range = slice(parent_from, parent_to) if name in PARENT_KEYS else slice(left, right)
            arrays[name] = np.array(a[row_range], copy=True)
        pointer = arrays['issue_source_index']
        present = pointer >= 0
        _require(np.all(pointer[~present] == -1) and np.all((pointer[present] >= left) & (pointer[present] < right)),
                 'ISSUE_POINTER_SOURCE')
        pointer[present] -= left
        return arrays, left, right, expected_bytes
    finally:
        os.close(fd)


def materialize_december_score_targets(policy, stage, output_dir):
    """Requires a separate capped, once-claimed after-FIT data route. Does no prediction, scoring or fitting."""
    from gx1.contracts.entry_dynamic_mtf_december_native_v1 import require_december_fit_origin
    prefix.require_capped_cpu_audit_execution()
    lock = prefix.require_capped_lock_ancestry()
    _require(policy[STAGE] == stage and stage['execution_authorized'] is True and stage['score_target_view_authorized'] is True
             and stage['claimed_invocations'] == stage['completed_invocations'] == 0 and stage['max_invocations'] == 1
             and stage['relaunch_allowed'] is False and stage['model_calls_authorized'] is stage['training_enabled'] is False
             and policy['training_enabled'] is False, 'ADMISSION')
    output = Path(output_dir)
    root = Path(stage['run_root'])
    _require(output == root / 'SCORE_TARGETS' and root.is_absolute() and root.resolve(strict=True) == root
             and not any(os.path.lexists(p) for p in (output, root/'SCORE_CALENDAR', root/'SCORE_CLOCK_CLASSIFICATION.json')),
             'FRESH_OUTPUT')
    origin = require_december_fit_origin(stage['fit_terminal_review'])
    _require(origin['fit_checkpoint'] == stage['fit_checkpoint'] and origin['fit_terminal_review'] == stage['fit_terminal_review']
             and stage['candidate_plan'] == policy['dynamic_mtf_december_candle_and_plan_20261104_001']['candidate_plan']
             and stage['source_target_result'] == policy['dynamic_mtf_state_target_audit_20261102_005']['result'],
             'FROZEN_SOURCES')
    plan = prefix.read_bound_json(stage['candidate_plan'])
    scopes = plan['observer_axes']
    fit, reserved = scopes['fit'], scopes['held_out']
    _require(fit['rows'] == 41866 and reserved['rows'] == 5736 and fit['cutoff_ns'] == 1733011200000000000
             and reserved['cutoff_ns'] == 1735689600000000000 and stage['max_source_npy_bytes'] == 65465804
             and stage['max_output_npy_bytes'] == 8888768, 'SCORE_GEOMETRY')
    source = prefix.read_bound_json(stage['source_target_result'])
    series_ref = source['target_manifest']
    _require(series_ref == source['series'], 'UPSTREAM_SERIES')
    m = target.read_dynamic_mtf_state_series_metadata(series_ref, expected_source_refs=source['target_sources'],
                                                       expected_cutoff_ns=source['fold']['series_cutoff_ns'])
    arrays, left, right, expected_bytes = _load_immutable_slice(m, Path(series_ref['path']).parent, fit, reserved,
                                                               stage['max_source_npy_bytes'], stage['max_output_npy_bytes'])
    issues, valid = arrays['issue_ns'], arrays['validity_ns']
    _require(np.array_equal(arrays['parent_rows'], np.arange(fit['rows'], fit['rows']+reserved['rows']))
             and all(hashlib.sha256(a.astype('<i8',copy=False).tobytes()).hexdigest() == reserved[k+'_sha256']
                     for k,a in [('parent_rows',arrays['parent_rows']),('decision_ns',issues),
                                 ('bar_start_ns',issues-target.LOCAL_WINDOW_M1_BARS['M5']*target.MINUTE_NS)])
             and np.all((valid >= fit['cutoff_ns']) & (valid < reserved['cutoff_ns']))
             and arrays['source_row'][0] == m['metadata']['source_row_offset'] + left, 'TARGET_ROLE_IDENTITY')
    starts = valid - target.MINUTE_NS
    calendar_dir = root/'SCORE_CALENDAR'
    closure, calendar, payloads = prefix._prefix_closures(source,m['metadata'],starts,calendar_dir,first_source_row=left)
    meta = {k:m['metadata'][k] for k in ('schema_version','definition','source_refs','gap_count_scope','target_rule_locked','test_data_used')}
    meta.update(cutoff_ns=reserved['cutoff_ns'],rows=len(valid),parents=len(issues),source_row_offset=int(arrays['source_row'][0]),
                supplied_source_rows=len(valid),supplied_clock_sha256=prefix._digest(starts,'<i8'),
                closure_binding=closure,atr_surfaces=prefix._prefix_atr_surfaces(arrays),
                score_lineage={'role':'score','candidate_plan':stage['candidate_plan'],'source_target_result':stage['source_target_result'],
                               'source_series':series_ref,'source_row_slice':[left,right],
                               'parent_slice':[fit['rows'], fit['rows']+reserved['rows']],'fit_origin':origin,
                               'calendar_binding':calendar,'historical_context':'005 precomputed causal state, range and ATR at each retained validity'})
    series = target._series(arrays,meta)
    std = json.loads(payloads['CLOSURE_AUTHORITY.json'])
    gap_rows = [r['gap_after_m1_row']+1 for r in std['intervals']]
    published = [r['gap_after_m1_row']+1 for r in std['intervals'] if r['successor_across_gap_allowed']]
    _require(np.array_equal(gap_rows,np.flatnonzero(np.diff(valid)!=target.MINUTE_NS)+1)
             and prefix.require_capped_lock_ancestry()==lock,'GAP_AND_LOCK')
    prefix._publish_calendar(calendar_dir,payloads)
    new = target.save_dynamic_mtf_state_series(series,output)
    result_manifest = prefix.read_bound_json(new)
    actual_bytes = sum(d['size_bytes'] for d in result_manifest['arrays'].values())
    _require(actual_bytes == expected_bytes, 'PUBLISHED_NPY_BYTES')
    cfile = root/'SCORE_CLOCK_CLASSIFICATION.json'
    c = {'role':'score','target_manifest':new,'candidate_plan':stage['candidate_plan'],'fit_origin':origin,
         'source_calendar':source['calendar_binding'],'qualified_calendar':calendar,
         'gap_after_rows':gap_rows,'published_closure_after_rows':published,
         'row_semantics':'zero-based local first M1 row after gap'}
    cb = prefix._calendar_bytes(c)
    _publish_bytes(cfile,cb)
    return {'status':'MATERIALIZED_RESERVED_ONLY_TARGETS_NO_SCORES','candidate_plan':stage['candidate_plan'],
            'fit_origin':origin,'fit_terminal_review':stage['fit_terminal_review'],'fit_checkpoint':stage['fit_checkpoint'],
            'target_manifest':new,'calendar_authority':{'path':str(cfile),'sha256':hashlib.sha256(cb).hexdigest()},
            'reserved_parents':len(issues),'source_rows':len(valid),'source_row_slice':[left,right],
            'fit_cutoff_ns':fit['cutoff_ns'],'score_cutoff_ns':reserved['cutoff_ns'],
            'source_npy_bytes':stage['max_source_npy_bytes'],'published_npy_bytes':actual_bytes,
            'model_calls':0,'optimizer_updates':0,'score_computed':False,'test_data_used':False}
