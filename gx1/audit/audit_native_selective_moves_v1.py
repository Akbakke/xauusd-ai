"""One-use CPU diagnosis of selective warnings on a complete frozen TRAIN capture.

The one-ATR first-cross predicate is observational, never a training label,
Entry/Exit/stop/hold rule or calibrated event probability. No model lives here.
"""
from __future__ import annotations
import argparse, hashlib, json, signal, subprocess, time, zipfile
from pathlib import Path
if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from gx1.audit import build_native_m5_direction_targets_v1 as native
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
STAGE = 'native_selective_move_audit_20261122_001'
BASENAME = 'NATIVE_SELECTIVE_MOVE_AUDIT_20261122_001'
PLAN = REPO / 'configs/research' / (BASENAME + '.json')
ROOT = Path('/home/andre2/GX1_RUNS') / BASENAME
SELF = 'gx1/audit/audit_native_selective_moves_v1.py'
TEST = 'tests/test_audit_native_selective_moves_v1.py'
PREREG = {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_SELECTIVE_MOVE_PREPARATION_20261122_001/SELECTIVE_MOVE_PREREGISTRATION001.json',
          'sha256': '94e93090a49b62c3a0b17fc0403db6a69993bd140e65f4fa378dd43ed52825c9'}
ADDENDUM = {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_SELECTIVE_MOVE_PREPARATION_20261122_001/FULL_COVERAGE_REVERSAL_SCOPE001.json',
            'sha256': '005422b037ca509e8c2c88a05e73d4e279d0130a2a468b6334752d8608488d50'}
REPORT_PARTITION = {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_SELECTIVE_MOVE_PREPARATION_20261122_001/UP_DOWN_REPORT_PARTITION001.json',
                    'sha256': '8d47b8223cdef9bd8a0520c5e6e88170e59c8cce5d2b23f687d68b4428ed0513'}
TF = ('M5', 'M15', 'H1', 'H4', 'D1')
THRESHOLDS = (.50, .60, .70, .80, .90)
ROWS = 354550
START, UPPER = 1609459200000000000, 1767225600000000000
TICK, MINUTE, DAY, ORIGIN = 300_000_000_000, 60_000_000_000, 86400_000_000_000, 79200_000_000_000
TARGET_FIELDS = ('issue_ns', 'atr14_asof', 'atr_asof_available_ns', 'target_start_ns', 'validity_ns')
CAPTURE_FIELDS = ('issue_ns', 'activity_logits', 'side_logits')
CAPTURE_KEYS = set(CAPTURE_FIELDS) | {'source_rows', 'selected_positions'}
FLAGS = ('training_enabled', 'native_training_enabled', 'native_launch_authorized', 'full_epoch_training_allowed', 'full_val_allowed')
OWNERS = (SELF, TEST, 'gx1/audit/build_native_m5_direction_targets_v1.py',
          'gx1/contracts/xau_tape_provenance_v1.py', 'gx1/contracts/native_m5_direction_io_firewall_v1.py',
          'gx1/contracts/gx1_capped_execution_v1.py', 'gx1/contracts/immutable_event_authority_v1.py', 'scripts/gx1_capped_run.sh')
EXECUTION = {'class': 'audit', 'memory': '4G', 'swap': '512M', 'outer_seconds': 600,
             'internal_seconds': 570, 'kill_after_seconds': 10, 'numerical_threads': 1}
UNKNOWN, DOWN, NONE, UP, BOTH = -2, -1, 0, 1, 2
KIND = {UNKNOWN: 'UNKNOWN', DOWN: 'DOWN_FIRST', NONE: 'NONE', UP: 'UP_FIRST', BOTH: 'BOTH_SAME_BAR'}


def need(ok, why):
    if not bool(ok):
        raise ValueError('NATIVE_SELECTIVE_MOVE_' + why)


def ref(path):
    path = Path(path)
    need(path.is_absolute() and not path.is_symlink() and path.is_file(), 'REGULAR_ABSOLUTE_FILE')
    return {'path': str(path), 'sha256': native.sha(path)}


def read(binding):
    need(type(binding) is dict and set(binding) == {'path', 'sha256'} and ref(binding['path']) == binding, 'JSON_BINDING')
    return json.loads(Path(binding['path']).read_text())


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def require_prereg(value):
    need(value['schema_version'] == 'gx1_native_selective_move_preregistration_v1'
         and value['execution_authorized'] is False, 'PREREG_NOT_AUTHORITY')
    d = value['diagnostic']
    need(set(d) == {'baselines', 'capture_success', 'direction', 'metrics', 'observational_event_predicate',
         'probability_caveat', 'report_all_cells_no_winner_or_threshold_selection', 'score', 'source_price_scope',
         'stage', 'thresholds', 'timeframes', 'warning_selection'}, 'PREREG_DIAGNOSTIC_KEYS')
    need(d['stage'] == STAGE and d['thresholds'] == list(THRESHOLDS) and d['timeframes'] == list(TF)
         and d['score'] == 'sigmoid(activity_logit)*max(sigmoid(side_logit),1-sigmoid(side_logit))'
         and d['direction'] == 'UP if sigmoid(side_logit)>=0.5 elseDOWN'
         and d['report_all_cells_no_winner_or_threshold_selection'] is True, 'PREREG_METHOD')
    event = d['observational_event_predicate']
    need(set(event) == {'anchor', 'event', 'hit_time', 'scale', 'unit', 'unknown', 'window'}
         and 'oneTFATR14' in event['scale'] and 'Firsteligibleissue' in event['anchor']
         and 'existingcanonicalvalidity_ns' in event['window'], 'PREREG_EVENT')
    return d


def validate_capture_metadata(result, capture, capture_plan, prereg):
    r = result['report']
    identity = prereg['model_identity']
    need(result['status'] == 'COMPLETE_NATIVE_FULL_TRAIN_CAPTURE' and result['source_unchanged'] is True
         and result['test_data_used'] is False and r['capture'] == capture, 'FULL_CAPTURE_SUCCESS')
    need(r['rows'] == ROWS and r['batch_rows'] == 256 and r['batch_count'] == 1385
         and set(r['capture_array_headers']) == CAPTURE_KEYS, 'FULL_CAPTURE_GEOMETRY')
    for key, h in r['capture_array_headers'].items():
        need(h['shape'] == ([ROWS, 5] if key.endswith('_logits') else [ROWS])
             and h['dtype'] == ('<f4' if key.endswith('_logits') else '<i8'), 'CAPTURE_HEADERS')
    need(r['reference_producer'] == identity['original_MAIN_result']
         and capture_plan['inputs']['main_plan'] == identity['original_MAIN_plan']
         and capture_plan['inputs']['main_result'] == identity['original_MAIN_result']
         and capture_plan['inputs']['checkpoint_receipt'] == identity['checkpoint_receipt']
         and capture_plan['inputs']['preregistration'] == PREREG, 'ORIGINAL_MAIN_FINAL_PRODUCER')
    need(r['original_FINAL_components']['model_state'] == r['model_state_restored']['model_state_sha256']
         == identity['model_state_sha256'] and r['original_PASS1_batch_hashes_all_bitexact'] is True
         and r['original_PASS1_full_output_and_proper_scores_bitexact'] is True, 'FULL_CAPTURE_BITEXACT')


def native_train_source(manifest_ref):
    manifest = native.source_manifest(manifest_ref)
    need(manifest['market_closure_contract'] == 'oanda_complete_true_scheduled_weekly_closure_excluded_v2', 'WEEKLY_CLOSURE_OWNER')
    parts, metadata = [], {}
    for year in range(2021, 2026):
        path = native.NATIVE / f'year={year}/part-000.parquet'
        part = {'path': str(path), 'sha256': manifest['year_sha256'][f'year={year}'],
                'row_groups': list(range(pq.ParquetFile(path).num_row_groups))}
        meta = native.part_metadata(part)
        need(meta['total_rows'] == manifest['year_rows'][f'year={year}']
             and all(START <= g['first_bar_ns'] <= g['last_bar_ns'] and g['last_bar_ns'] + TICK < UPPER
                     for g in meta['selected_groups']), 'ONLY_FIVE_TRAIN_YEARS')
        parts.append(part)
        metadata[str(path)] = meta
    need(sum(v['total_rows'] for v in metadata.values()) == ROWS, 'NATIVE_TRAIN_ROWS')
    return {'manifest': manifest_ref, 'parts': parts, 'closure_authority': None,
            'issue_start_ns': START, 'cutoff_ns': UPPER}, metadata


def build_plan(*, capture_result, capture, source_base_commit=None):
    prereg = read(PREREG)
    diagnostic = require_prereg(prereg)
    scope = read(ADDENDUM)
    need(scope['schema_version'] == 'gx1_native_full_coverage_reversal_scope_v1'
         and scope['execution_authorized'] is False and scope['original_preregistration'] == PREREG
         and scope['coverage']['TRAIN_rows'] == ROWS and scope['coverage']['timeframes'] == list(TF)
         and scope['coverage']['years'] == list(range(2021, 2026))
         and scope['reversal_observation']['new_training_target_or_fit'] is False
         and scope['reversal_observation']['scores_and_thresholds_unchanged'] is True, 'SCOPE_ADDENDUM')
    partition = read(REPORT_PARTITION)
    need(partition['schema_version'] == 'gx1_native_selective_UP_DOWN_report_partition_v1'
         and partition['execution_authorized'] is False and partition['scope_addendum'] == ADDENDUM
         and partition['method_unchanged'] is True and partition['selector_thresholds_events_baselines_unchanged'] is True,
         'LOCKED_REPORT_PARTITION')
    original = read(prereg['model_identity']['original_MAIN_plan'])
    result = read(capture_result)
    capture_plan = read(result['plan'])
    validate_capture_metadata(result, capture, capture_plan, prereg)
    m = original['metadata']
    dataset = read(m['dataset'])
    need(dataset['native_manifest'] == m['native_manifest'] and dataset['cutoff_ns'] == UPPER
         and dataset['TRAIN_issue_ns_sha256'] == m['TRAIN_issue_ns_sha256']
         and dataset['array_headers']['issue_ns']['shape'] == [ROWS], 'ACTUAL_MAIN_TRAIN_DATASET')
    spec, meta = native_train_source(m['native_manifest'])
    inputs = {'preregistration': PREREG, 'scope_addendum': ADDENDUM, 'report_partition': REPORT_PARTITION, 'main_plan': prereg['model_identity']['original_MAIN_plan'],
              'main_result': prereg['model_identity']['original_MAIN_result'], 'capture_result': capture_result,
              'capture_plan': result['plan'], 'capture': capture, 'dataset': m['dataset'],
              'target_arrays': dataset['arrays'], 'source_snapshot': m['source_snapshot'],
              'native_manifest': m['native_manifest'], 'train_prior': m['prior']}
    need(result['report']['capture_array_headers']['issue_ns'] == dataset['array_headers']['issue_ns'], 'FULL_CAPTURE_TARGET_CLOCK_HEADERS')
    return {'schema_version': 'gx1_native_selective_move_audit_plan_v1', 'stage': STAGE,
            'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(ROOT), 'plan_path': str(PLAN),
            'source_base_commit': source_base_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'source_owners': {n: native.sha(REPO / n) for n in OWNERS}, 'inputs': inputs,
            'source_spec': spec, 'source_metadata': meta, 'columns': list(native.COLUMNS),
            'capture_array_headers': result['report']['capture_array_headers'],
            'target_array_headers': {k: dataset['array_headers'][k] for k in TARGET_FIELDS},
            'TRAIN_issue_ns_sha256': m['TRAIN_issue_ns_sha256'], 'diagnostic': diagnostic,
            'reversal_observation': scope['reversal_observation'], 'report_partitions': partition['partitions'],
            'execution': EXECUTION, 'runtime_versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__},
            'model_forwards': 0, 'checkpoint_loads': 0, 'optimizer_updates': 0, 'fits': 0,
            'CONTROL_VAL_TEST_rows': 0, 'max_invocations': 1, 'automatic_continuation': False}


def source_clean(plan):
    return (subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == 'work/gx1-current'
            and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == plan['source_base_commit']
            and not subprocess.check_output(['git', 'status', '--porcelain', '--', 'gx1', 'scripts', 'tests'], cwd=REPO, text=True)
            and set(plan['source_owners']) == set(OWNERS)
            and all(native.sha(REPO / k) == v for k, v in plan['source_owners'].items()))


def gate_expected(binding, admitted):
    return {'status': 'ADMITTED_EXACT_CPU_ONLY_SELECTIVE_MOVE_AUDIT' if admitted else 'PREPARED_NOT_ADMITTED',
            'plan': binding, 'execution_authorized': admitted, 'remaining_invocations': int(admitted),
            'relaunch_allowed': False, 'training_authorized': False, 'test_authorized': False}


def inspect(plan_sha, *, admitted):
    need(Path.cwd().resolve() == REPO, 'CWD')
    binding = {'path': str(PLAN), 'sha256': plan_sha}
    plan = read(binding)
    need(plan == build_plan(capture_result=plan['inputs']['capture_result'], capture=plan['inputs']['capture'],
                           source_base_commit=plan['source_base_commit']) and source_clean(plan), 'EXACT_PLAN_SOURCE')
    pref = ref(POLICY)
    policy = read(pref)
    need(all(policy.get(k) is False for k in FLAGS) and policy[STAGE] == gate_expected(binding, admitted), 'ALL_TRAINING_CLOSED_ONE_USE_GATE')
    need(not ROOT.exists() and not ROOT.is_symlink(), 'FRESH_OUTPUT_ROOT')
    return plan, binding, pref


def archive_headers(binding, headers, *, exact_inventory=None):
    with zipfile.ZipFile(binding['path']) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and all(n.endswith('.npy') and '/' not in n for n in names), 'NPZ_INVENTORY')
        if exact_inventory is not None:
            need(set(names) == {k + '.npy' for k in exact_inventory}, 'NPZ_EXACT_KEYS')
        for key, h in headers.items():
            with z.open(key + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                need(version in ((1, 0), (2, 0)), 'NPY_VERSION')
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                         else np.lib.format.read_array_header_2_0(stream))
                need(list(shape) == h['shape'] and dtype.str == h['dtype'] and not dtype.hasobject and not fortran, 'NPY_HEADER')


def checked_inputs(plan):
    for binding in plan['inputs'].values():
        need(ref(binding['path']) == binding, 'INPUT_SHA')
    native.checked_inputs(plan)
    archive_headers(plan['inputs']['capture'], plan['capture_array_headers'], exact_inventory=CAPTURE_KEYS)
    archive_headers(plan['inputs']['target_arrays'], plan['target_array_headers'])


def load_fields(binding, fields, headers):
    out = {}
    with np.load(binding['path'], allow_pickle=False) as z:
        for key in fields:
            a = z[key]
            h = headers[key]
            need(a.dtype.str == h['dtype'] and list(a.shape) == h['shape'] and array_sha(a) == h['sha256'], 'PAYLOAD_ARRAY_IDENTITY')
            out[key] = a
    return out


def validate_full_clock(capture, target, starts):
    issue = capture['issue_ns']
    need(issue.dtype == target['issue_ns'].dtype == np.dtype('int64') and issue.shape == target['issue_ns'].shape == starts.shape == (ROWS,)
         and np.array_equal(issue, target['issue_ns']) and np.array_equal(issue, starts + TICK)
         and np.all(np.diff(issue) > 0) and np.all((issue >= START) & (issue < UPPER)), 'HUNDRED_PERCENT_TRAIN_CLOCK')
    for key in ('activity_logits', 'side_logits'):
        need(capture[key].dtype == np.dtype('float32') and capture[key].shape == (ROWS, 5)
             and np.isfinite(capture[key]).all(), 'ALL_FIVE_FINITE_HEADS')


def clock_groups(issues, target_starts, validities):
    need(issues.ndim == target_starts.ndim == validities.ndim == 1 and len(issues) == len(target_starts) == len(validities)
         and len(issues) > 0 and np.all(np.diff(issues) > 0), 'GROUP_CLOCK')
    eligible = (target_starts >= issues) & (validities > target_starts)
    # Invalid clock rows stay as individual UNKNOWN groups; none are discarded.
    change = np.r_[True, (target_starts[1:] != target_starts[:-1]) | ~eligible[1:] | ~eligible[:-1]]
    begins = np.flatnonzero(change)
    ends = np.r_[begins[1:], len(issues)]
    need(all(not eligible[a] or np.all(validities[a:b] == validities[a]) for a, b in zip(begins, ends)), 'GROUP_VALIDITY_CONSTANT')
    return begins, ends, eligible


def first_warnings(scores, begins, threshold):
    need(scores.ndim == 1 and np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), 'SCORE_DOMAIN')
    warnings = np.full(len(begins), -1, np.int64)
    selected = np.flatnonzero(scores >= threshold)
    if len(selected):
        group = np.searchsorted(begins, selected, side='right') - 1
        ids, at = np.unique(group, return_index=True)
        warnings[ids] = selected[at]
    return warnings


def gap_table(starts, *, grid_start, upper, closed):
    grid = np.arange(grid_start, upper, TICK, dtype=np.int64)
    need(closed.dtype == np.bool_ and closed.shape == grid.shape and np.all((starts >= grid_start) & (starts < upper))
         and np.all(starts % TICK == 0) and np.all(np.diff(starts) > 0), 'EXPECTED_GRID')
    present = np.zeros(len(grid), np.bool_)
    at = ((starts - grid_start) // TICK).astype(np.int64)
    present[at] = True
    need(not np.any(present & closed), 'PRICE_INSIDE_DECLARED_WEEKLY_CLOSURE')
    return {'start': grid_start, 'upper': upper,
            'gaps': np.r_[0, np.cumsum(~closed & ~present)], 'expected': np.r_[0, np.cumsum(~closed)]}


def _prefix_count(table, field, start, end):
    a, b = (start - table['start']) // TICK, (end - table['start']) // TICK
    return int(table[field][b] - table[field][a])


def scan_event(starts, prices, *, anchor, end, target_start, atr, atr_available, table):
    out = {'kind': UNKNOWN, 'hit_row': -1, 'hit_start': -1, 'hit_end': -1, 'end_row': -1,
           'anchor_price': np.nan, 'complete': False, 'pretarget_hit': False, 'reason': ''}
    if not (table['start'] <= anchor < end < table['upper'] and anchor <= target_start < end
            and anchor % TICK == end % TICK == target_start % TICK == 0):
        out['reason'] = 'INVALID_OR_IMMATURE_WINDOW'; return out
    if not (np.isfinite(atr) and atr > 0 and 0 <= atr_available <= anchor):
        out['reason'] = 'MISSING_OR_NONCAUSAL_ATR'; return out
    anchor_row = int(np.searchsorted(starts, anchor - TICK, side='right') - 1)
    if anchor_row < 0:
        out['reason'] = 'NO_KNOWN_ANCHOR_PRICE'; return out
    price = float(prices[anchor_row, 3]); out['anchor_price'] = price
    a, b = int(np.searchsorted(starts, anchor)), int(np.searchsorted(starts, end - TICK, side='right'))
    expected = _prefix_count(table, 'expected', anchor, end)
    out['complete'] = bool(expected > 0 and _prefix_count(table, 'gaps', anchor, end) == 0)
    if out['complete'] and b > a:
        out['end_row'] = b - 1
    ups = np.flatnonzero(prices[a:b, 1] >= price + atr)
    downs = np.flatnonzero(prices[a:b, 2] <= price - atr)
    if not len(ups) and not len(downs):
        out['kind'] = NONE if out['complete'] else UNKNOWN
        out['reason'] = '' if out['complete'] else 'INCOMPLETE_NO_HIT_WINDOW'
        return out
    u, d = int(ups[0]) if len(ups) else b - a, int(downs[0]) if len(downs) else b - a
    row = a + min(u, d); hit = int(starts[row])
    if _prefix_count(table, 'gaps', anchor, hit):
        out['reason'] = 'PREFIX_GAP_BEFORE_HIT'; return out
    out.update(kind=BOTH if u == d else UP if u < d else DOWN, hit_row=row,
               hit_start=hit, hit_end=hit + TICK, pretarget_hit=bool(hit < target_start))
    return out


def anchor_past_direction(starts, prices, *, anchor, duration, table):
    past = anchor - duration
    if not (duration > 0 and duration % TICK == 0 and table['start'] + TICK <= past < anchor < table['upper']):
        return UNKNOWN
    # Include the bar ending at the past reference: a missing open bar there
    # cannot silently substitute an older price. Declared closures remain exempt.
    if _prefix_count(table, 'gaps', past - TICK, anchor):
        return UNKNOWN
    a = int(np.searchsorted(starts, anchor - TICK, side='right') - 1)
    p = int(np.searchsorted(starts, past - TICK, side='right') - 1)
    if p < 0 or a < 0:
        return UNKNOWN
    return int(np.sign(prices[a, 3] - prices[p, 3]))


def event_strata(kinds, past):
    known_move = (kinds == UP) | (kinds == DOWN)
    known_past = (past == UP) | (past == DOWN)
    strata = {'REVERSAL': known_move & known_past & (kinds != past),
              'CONTINUATION': known_move & known_past & (kinds == past),
              'NO_PAST_DIRECTION': past == NONE, 'UNKNOWN_PAST': past == UNKNOWN,
              'NO_KNOWN_DIRECTIONAL_EVENT': ~known_move & known_past}
    need(np.all(sum(v.astype(np.int8) for v in strata.values()) == 1), 'EXHAUSTIVE_EVENT_STRATA')
    return strata


def event_arrays(issues, target_starts, validities, atr, atr_available, starts, prices, table):
    begins, ends, eligible = clock_groups(issues, target_starts, validities)
    n = len(begins)
    events = {'kind': np.full(n, UNKNOWN, np.int8), 'hit_row': np.full(n, -1, np.int64),
              'hit_start': np.full(n, -1, np.int64), 'hit_end': np.full(n, -1, np.int64),
              'end_row': np.full(n, -1, np.int64), 'anchor_price': np.full(n, np.nan),
              'complete': np.zeros(n, np.bool_), 'pretarget_hit': np.zeros(n, np.bool_),
              'reason': np.full(n, '', dtype='<U32'), 'begins': begins, 'ends': ends,
              'anchor_ns': issues[begins], 'target_start': target_starts[begins], 'atr': atr[begins],
              'past_direction_at_anchor': np.full(n, UNKNOWN, np.int8)}
    for g, i in enumerate(begins):
        out = scan_event(starts, prices, anchor=int(issues[i]), end=int(validities[i]), target_start=int(target_starts[i]),
                         atr=float(atr[i]), atr_available=int(atr_available[i]), table=table)
        for k, v in out.items(): events[k][g] = v
        events['past_direction_at_anchor'][g] = anchor_past_direction(starts, prices, anchor=int(issues[i]),
            duration=int(validities[i] - target_starts[i]), table=table)
    events['clock_eligible_rows'] = int(eligible.sum())
    return events


def distribution(values):
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    return None if not len(a) else {'count': len(a), 'minimum': float(a.min()), 'mean': float(a.mean()),
                                   'median': float(np.median(a)), 'maximum': float(a.max())}


def classify_warnings(kinds, hit_starts, warning_times, warning_sides, selected):
    moves = (kinds == UP) | (kinds == DOWN)
    correct = selected & moves & (warning_sides == kinds) & (warning_times <= hit_starts)
    wrong = selected & moves & (warning_sides != kinds)
    late = selected & moves & (warning_sides == kinds) & (warning_times > hit_starts)
    return {'correct_timely': correct, 'wrong_direction': wrong, 'late_correct_direction': late,
            'no_hit': selected & (kinds == NONE), 'unknown': selected & (kinds == UNKNOWN),
            'both_same_bar': selected & (kinds == BOTH), 'moves': moves}


def warning_summary(events, warnings, side_by_row, issues, starts, prices):
    selected = warnings >= 0
    w = np.maximum(warnings, 0)
    warning_times, sides = issues[w], side_by_row[w]
    c = classify_warnings(events['kind'], events['hit_start'], warning_times, sides, selected)
    n, good = int(selected.sum()), int(c['correct_timely'].sum())
    uncertain = int(c['unknown'].sum() + c['both_same_bar'].sum())
    counts = {k: int(v.sum()) for k, v in c.items() if k != 'moves'}
    need(sum(counts.values()) == n, 'ALL_SELECTED_CLASSIFIED')
    days = np.unique((issues - ORIGIN) // DAY)
    warned_days, frequencies = np.unique((warning_times[selected] - ORIGIN) // DAY, return_counts=True)
    frequency = dict(zip(warned_days.tolist(), frequencies.tolist()))
    per_day = {str(np.datetime64(int(day * DAY + ORIGIN), 'ns')) + 'Z': int(frequency.get(int(day), 0)) for day in days}
    known = c['moves']
    all_hit_keys = set(zip(events['hit_start'][known].tolist(), events['kind'][known].tolist()))
    captured_keys = set(zip(events['hit_start'][c['correct_timely']].tolist(), events['kind'][c['correct_timely']].tolist()))
    measured = selected & known
    wr = np.searchsorted(starts, warning_times[measured] - TICK, side='right') - 1
    need(np.all(wr >= 0), 'WARNING_PRICE_ASOF')
    progress = (prices[wr, 3] - events['anchor_price'][measured]) * events['kind'][measured] / events['atr'][measured]
    forward_hit = measured & (warning_times <= events['hit_start'])
    hit_rows = events['hit_row'][forward_hit]
    hit_wr = np.searchsorted(starts, warning_times[forward_hit] - TICK, side='right') - 1
    hit_return = (prices[hit_rows, 3] / prices[hit_wr, 3] - 1) * sides[forward_hit] * 10000
    complete = measured & events['complete'] & (events['end_row'] >= 0)
    end_wr = np.searchsorted(starts, warning_times[complete] - TICK, side='right') - 1
    end_return = (prices[events['end_row'][complete], 3] / prices[end_wr, 3] - 1) * sides[complete] * 10000
    progress_by_group = np.full(len(selected), np.nan)
    progress_by_group[measured] = progress
    at_anchor = c['correct_timely'] & (warning_times == events['anchor_ns'])
    later_before_hit = c['correct_timely'] & (warning_times > events['anchor_ns'])
    def stratum_summary(mask):
        selected_n = int((mask & selected).sum())
        correct_n = int((mask & c['correct_timely']).sum())
        unknown_n = int((mask & (c['unknown'] | c['both_same_bar'])).sum())
        known_n = int((mask & known).sum())
        return {'groups': int(mask.sum()), 'eligible_known_moves': known_n, 'captured_known_moves': correct_n,
                'missed_known_moves': known_n - correct_n, 'selected_warnings': selected_n,
                'selected_false_wrong_late_nohit': int((mask & (c['wrong_direction'] | c['late_correct_direction'] | c['no_hit'])).sum()),
                'selected_unknown_or_same_bar_order': unknown_n,
                'classification': {k: int((mask & v).sum()) for k, v in c.items() if k != 'moves'},
                'selected_precision_lower': correct_n / selected_n if selected_n else None,
                'selected_precision_upper': (correct_n + unknown_n) / selected_n if selected_n else None,
                'correct_timely_at_anchor': int((mask & at_anchor).sum()),
                'correct_later_before_1ATR_hit': int((mask & later_before_hit).sum()),
                'correct_timely_progress_ATR': distribution(progress_by_group[mask & c['correct_timely']]),
                'correct_timely_headroom_ATR': distribution(1 - progress_by_group[mask & c['correct_timely']])}
    past = events['past_direction_at_anchor']
    known_past = (past == UP) | (past == DOWN)
    warning_relation = {'OPPOSES_PAST': selected & known_past & (sides != past),
                        'FOLLOWS_PAST': selected & known_past & (sides == past),
                        'NO_PAST_DIRECTION': selected & (past == NONE), 'UNKNOWN_PAST': selected & (past == UNKNOWN)}
    return {'selected_warnings': n, 'classification': counts,
            'correct_timely_at_anchor': int(at_anchor.sum()), 'correct_later_before_1ATR_hit': int(later_before_hit.sum()),
            'event_strata': {k: stratum_summary(v) for k, v in event_strata(events['kind'], past).items()},
            'event_direction': {KIND[d]: stratum_summary(events['kind'] == d) for d in (UP, DOWN)},
            'warning_direction': {name: stratum_summary(selected & (sides == d)) for name, d in (('UP', UP), ('DOWN', DOWN))},
            'reversal_direction': {KIND[d]: stratum_summary((events['kind'] == d) & (past == -d)) for d in (UP, DOWN)},
            'warning_relation_to_past_at_GROUP_ANCHOR': {k: stratum_summary(v) for k, v in warning_relation.items()},
            'timing_limit': 'Anchor and later-before-1ATR-hit are distinct; neither certifies warning before latent physical turn onset.',
            'selected_precision_lower': good / n if n else None, 'selected_precision_upper': (good + uncertain) / n if n else None,
            'known_moves': int(known.sum()), 'captured_known_moves': good, 'missed_known_moves': int(known.sum()) - good,
            'known_move_coverage': good / int(known.sum()) if known.any() else None,
            'known_moves_without_warning': int((known & ~selected).sum()),
            'pretarget_warnings': int((selected & (warning_times < events['target_start'])).sum()),
            'selected_pretarget_hits': int((selected & events['pretarget_hit']).sum()),
            'captured_pretarget_hits': int((c['correct_timely'] & events['pretarget_hit']).sum()),
            'correct_timely_warning_lead_lower_minutes': distribution((events['hit_start'][c['correct_timely']] - warning_times[c['correct_timely']]) / MINUTE),
            'correct_timely_warning_lead_upper_minutes': distribution((events['hit_end'][c['correct_timely']] - warning_times[c['correct_timely']]) / MINUTE),
            'first_warnings_per_observed_24h_day_22UTC': per_day, 'observed_days': len(days),
            'mean_first_warnings_per_observed_day': n / len(days),
            'unique_TF_hitbar_direction_known_moves': len(all_hit_keys), 'unique_TF_hitbar_direction_captured': len(captured_keys),
            'unique_TF_hitbar_direction_missed': len(all_hit_keys - captured_keys),
            'warning_progress_in_observed_event_direction_ATR': distribution(progress),
            'warning_remaining_headroom_ATR': distribution(1 - progress),
            'gross_mid_return_warning_close_to_hitbar_close_bps': distribution(hit_return),
            'gross_mid_return_warning_close_to_complete_window_end_bps': distribution(end_return),
            'costs': 'NOT_MEASURED', 'MAE_MFE': 'NOT_MEASURED', 'PnL': 'NOT_MEASURED',
            'gross_markout_caveat': 'Observed mid-close diagnostic only; no fill/latency/cost or trade result. Hit time is a five-minute interval.'}


def analyse(capture, target, starts, prices, table, prior):
    issues = capture['issue_ns']
    a = np.exp(-np.logaddexp(0., -capture['activity_logits'].astype(np.float64)))
    s = np.exp(-np.logaddexp(0., -capture['side_logits'].astype(np.float64)))
    score, sides = a * np.maximum(s, 1 - s), np.where(s >= .5, UP, DOWN).astype(np.int8)
    report = {}
    for j, tf in enumerate(TF):
        e = event_arrays(issues, target['target_start_ns'][:, j], target['validity_ns'][:, j], target['atr14_asof'][:, j],
                         target['atr_asof_available_ns'][:, j], starts, prices, table)
        groups = len(e['begins'])
        p_a = float(prior['per_task']['activity'][tf]['probabilities'][1])
        p_s = float(prior['per_task']['side'][tf]['probabilities'][1])
        need(0 < p_a < 1 and 0 < p_s < 1, 'FROZEN_PRIOR_PROBABILITY')
        prior_sides = np.full(len(issues), UP if p_s >= .5 else DOWN, np.int8)
        cells = {}
        for threshold in THRESHOLDS:
            warnings = first_warnings(score[:, j], e['begins'], threshold)
            prior_warnings = e['begins'].copy() if p_a * max(p_s, 1 - p_s) >= threshold else np.full(groups, -1, np.int64)
            cells[f'{threshold:.2f}'] = {
                'model': warning_summary(e, warnings, sides[:, j], issues, starts, prices),
                'always_UP_at_model_warning_issues': warning_summary(e, warnings, np.full(len(issues), UP, np.int8), issues, starts, prices),
                'always_DOWN_at_model_warning_issues': warning_summary(e, warnings, np.full(len(issues), DOWN, np.int8), issues, starts, prices),
                'constant_TRAIN_prior_at_model_warning_issues': warning_summary(e, warnings, prior_sides, issues, starts, prices),
                'constant_TRAIN_prior_same_threshold': warning_summary(e, prior_warnings, prior_sides, issues, starts, prices)}
        reasons, reason_counts = np.unique(e['reason'][e['kind'] == UNKNOWN], return_counts=True)
        report[tf] = {'issue_rows': len(issues), 'score_distribution': distribution(score[:, j]),
                      'predicted_UP_fraction': float((sides[:, j] == UP).mean()),
                      'clock_eligible_rows': e['clock_eligible_rows'],
                      'clock_ineligible_rows_kept_UNKNOWN': len(issues) - e['clock_eligible_rows'],
                      'unique_target_window_groups': groups, 'group_rows_total': int((e['ends'] - e['begins']).sum()),
                      'event_counts': {name: int((e['kind'] == k).sum()) for k, name in KIND.items()},
                      'UNKNOWN_reasons': {str(k): int(v) for k, v in zip(reasons, reason_counts)},
                      'pretarget_known_hits': int(e['pretarget_hit'].sum()),
                      'past_direction_at_anchor_counts': {'UP': int((e['past_direction_at_anchor'] == UP).sum()),
                          'DOWN': int((e['past_direction_at_anchor'] == DOWN).sum()),
                          'NO_PAST_DIRECTION': int((e['past_direction_at_anchor'] == NONE).sum()),
                          'UNKNOWN_PAST': int((e['past_direction_at_anchor'] == UNKNOWN).sum())},
                      'constant_TRAIN_prior': {'activity': p_a, 'UP_given_activity': p_s, 'score': p_a * max(p_s, 1 - p_s)},
                      'thresholds': cells, 'independent_episodes_claimed': False}
        need(sum(report[tf]['event_counts'].values()) == groups and report[tf]['group_rows_total'] == len(issues), 'ALL_ROWS_GROUPS_ACCOUNTED')
    return report


def calculate(plan):
    capture = load_fields(plan['inputs']['capture'], CAPTURE_FIELDS, plan['capture_array_headers'])
    target = load_fields(plan['inputs']['target_arrays'], TARGET_FIELDS, plan['target_array_headers'])
    starts, prices, decoded = native.read_source(plan)
    validate_full_clock(capture, target, starts)
    need(array_sha(capture['issue_ns']) == plan['TRAIN_issue_ns_sha256'] and decoded == ROWS
         and prices.shape == (ROWS, 4) and np.isfinite(prices).all() and np.all(prices > 0)
         and np.all(prices[:, 1] >= prices.max(axis=1)) and np.all(prices[:, 2] <= prices.min(axis=1)), 'NATIVE_PRICE_BYTES_CLOCK_DOMAIN')
    prior = read(plan['inputs']['train_prior'])
    need(prior['fit_population']['scope'] == 'TRAIN' and prior['fit_population']['issue_rows'] == ROWS
         and prior['fit_population']['issue_ns_sha256'] == plan['TRAIN_issue_ns_sha256'], 'FROZEN_WHOLE_TRAIN_PRIOR')
    from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
    grid = np.arange(START, UPPER, TICK, dtype=np.int64)
    closed = scheduled_weekly_closure_mask(pd.to_datetime(grid, utc=True))
    table = gap_table(starts, grid_start=START, upper=UPPER, closed=np.asarray(closed, dtype=np.bool_))
    return {'rows': ROWS, 'all_five_timeframes_retained': list(TF), 'full_clock_bytes_match': True,
            'capture_array_fields_decoded': list(CAPTURE_FIELDS), 'target_array_fields_decoded': list(TARGET_FIELDS),
            'native_price_rows_decoded': decoded, 'native_price_columns': list(native.COLUMNS),
            'per_tf': analyse(capture, target, starts, prices, table, prior),
            'unproven': ['TRAIN reuse is not independent generalization or economic edge.',
                'This one-ATR predicate is observational only, not a training/Entry/Exit/stop/hold rule.',
                'Future-state probabilities are not calibrated probabilities of this observational event.',
                'M5 first-hit timing is an interval; both thresholds in the same first bar have unknown order.',
                'Only manifest weekly closures are excluded; missing open bars and immature windows remain UNKNOWN.',
                'Target-window and TF-hitbar-direction counts are not independent market episodes.']}


def publish(name, value):
    path = ROOT / name
    _publish_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    return ref(path)


def run(plan_sha):
    started = time.monotonic()
    cap, lock = require_capped_cpu_audit_execution(), require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3 and cap['swap'] == 512 * 1024**2
         and cap['numerical_threads'] == 1 and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):], 'CAP_LIMITS')
    plan, binding, pref = inspect(plan_sha, admitted=True)
    checked_inputs(plan)
    ROOT.mkdir(parents=True, exist_ok=False)
    claim = publish('CLAIM.json', {'status': 'CLAIMED_ONE_USE', 'stage': STAGE, 'plan': binding, 'policy_at_claim': pref,
                                 'remaining_invocations': 0, 'relaunch_allowed': False, 'test_data_used': False})
    old = signal.getsignal(signal.SIGALRM)
    result_ref = None
    try:
        def expired(*_): raise TimeoutError('SELECTIVE_MOVE_WALL')
        signal.signal(signal.SIGALRM, expired)
        remaining = EXECUTION['internal_seconds'] - (time.monotonic() - started)
        need(remaining > 0, 'TIME_BEFORE_CALCULATE')
        signal.setitimer(signal.ITIMER_REAL, remaining)
        report = calculate(plan)
        checked_inputs(plan)
        need(source_clean(plan) and ref(POLICY) == pref and ref(PLAN) == binding, 'SOURCE_INPUT_POLICY_BOOKEND')
        result_ref = publish('RESULT.json', {'status': 'COMPLETE_NATIVE_SELECTIVE_MOVE_AUDIT', 'stage': STAGE,
            'plan': binding, 'claim': claim, 'policy_at_claim': pref, 'preregistration': PREREG, 'scope_addendum': ADDENDUM, 'report_partition': REPORT_PARTITION, 'report': report,
            'source_unchanged': True, 'model_forwards': 0, 'checkpoint_loads': 0, 'optimizer_updates': 0,
            'fits': 0, 'test_data_used': False, 'economic_edge_proven': False, 'elapsed_seconds': time.monotonic() - started})
        terminal = publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE, 'plan': binding,
            'claim': claim, 'result': result_ref, 'remaining_invocations': 0, 'relaunch_allowed': False,
            'source_unchanged': True, 'test_data_used': False})
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE, 'plan': binding,
            'claim': claim, 'result': result_ref, 'remaining_invocations': 0, 'relaunch_allowed': False,
            'error': type(error).__name__ + ':' + str(error)[:2000], 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, binding, pref = inspect(args.plan_sha256, admitted=False)
        checked_inputs(plan)
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'plan': binding, 'policy': pref, 'claim_created': False,
                          'array_payloads_decoded': 0, 'parquet_rows_decoded': 0, 'models': 0, 'checkpoint_loads': 0}))
    else:
        print(json.dumps(run(args.plan_sha256)))


if __name__ == '__main__':
    main()
