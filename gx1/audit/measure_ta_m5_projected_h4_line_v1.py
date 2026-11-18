"""One-use, read-only M5 close break proxy against a projected closed H4 line.

This is a new research convention, not an existing model feature or a trade.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import time

from gx1.audit import score_ta_ema_final_fit_v1 as replay
from gx1.audit import market_state_ta_ema_delta_campaign_inputs_v1 as ema_inputs
from gx1.audit import run_market_state_ta_m5_decision_fit_v1 as fit_owner
from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import need, read, ref

REPO = replay.REPO
STAGE = 'ta_m5_projected_h4_line_20261118_001'
PLAN = REPO / 'configs/research/TA_M5_PROJECTED_H4_LINE_20261118_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_PROJECTED_H4_LINE_20261118_001')
ORACLE = Path('/home/andre2/GX1_RUNS/H4_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001/RESULT.json')
MAX_SECONDS = 1800
M5_NS = 300 * 10**9
H4_NS = 14400 * 10**9
FIELD_MAP = {'atr_bps_14': 0, 'geomline_above_active_count': 159,
             'geomline_above_dist_atr': 160,
             'geomline_above_slope_atr_per_bar': 161,
             'geomline_below_active_count': 166,
             'geomline_below_dist_atr': 167,
             'geomline_below_slope_atr_per_bar': 168}
NO_H4, OUTSIDE_ORACLE, INCOMPLETE_H4, GAP, NO_LINE, NONE, UP_BREAK, DOWN_BREAK, AMBIGUOUS = range(9)
CODES = ('NO_H4', 'OUTSIDE_ORACLE', 'INCOMPLETE_H4', 'GAP', 'NO_LINE',
         'NONE', 'UP_BREAK', 'DOWN_BREAK', 'AMBIGUOUS')
SOURCES = tuple(dict.fromkeys((*replay.SOURCES,
    'gx1/audit/measure_ta_m5_projected_h4_line_v1.py',
    'gx1/features/htf_features.py', 'gx1/features/trendline_registry_v1.py',
    'tests/test_measure_ta_m5_projected_h4_line_v1.py',
    'tests/test_ta_m5_projected_h4_line_wrapper_v1.py')))


def frozen_refs():
    score = {name: ref(replay.ROOT / file) for name, file in
             (('result', 'RESULT.json'), ('terminal', 'TERMINAL.json'),
              ('capture', 'FIT_SCORE_READOUT.npz'))}
    result, terminal = read(score['result']), read(score['terminal'])
    need(result['status'] == 'COMPLETE_FROZEN_EMA_FOLD1_FIT_SCORE_NO_UPDATE'
         and result['fit_capture'] == score['capture']
         and result['fit_rows'] == replay.FIT_ROWS
         and result['source_unchanged'] is True
         and result['optimizer_updates'] == 0
         and result['test_data_used'] is False
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['result'] == score['result']
         and terminal['remaining_invocations'] == 0, 'FROZEN_FIT_SCORE')
    consumed, prior = replay.consumed_refs()
    need(prior['score_rows'] == ema_inputs.LATER_ROWS
         and prior['final_capture'] == consumed['final_capture']
         and prior['source_unchanged'] is True, 'FROZEN_LATER_SCORE')
    return score, consumed


def qualified_inputs():
    inputs = ema_inputs.successor_input_refs()
    ema_inputs._receipts(inputs)
    old = read(inputs['old_physical_plan'])
    caches = old['caches']['H4']
    enriched = old['inputs']['enriched']
    need(ref(Path(enriched['path'])) == enriched
         and all(ref(Path(caches[key]['path'])) == caches[key]
                 for key in ('feats', 'ts')), 'PHYSICAL_REFS')
    packet = read(inputs['packet'])
    names = packet['parent_anchor_packet']['parent_prefix_normalization'][
        'base_normalization']['surfaces']['mtf_m5']['field_names']
    need(len(names) == 253
         and all(names[index] == name for name, index in FIELD_MAP.items()),
         'H4_FIELD_MAP')
    oracle_ref = ref(ORACLE)
    oracle = read(oracle_ref)
    comparison = oracle['comparison']
    fields = {item['field']: item for item in comparison['per_field']}
    need(oracle['status'] == 'COMPLETE'
         and oracle['decision'] == 'PASS_H4_TRAIN_WINDOW_19_EXERCISED'
         and oracle['cache_refs'] == caches
         and oracle['train_cache_rows']['count'] == 1044
         and comparison['all_train_window_cells_match'] is True
         and all(fields[name]['mismatched_rows'] == 0
                 for name in FIELD_MAP if name != 'atr_bps_14'),
         'H4_PHYSICAL_ORACLE')
    return inputs, enriched, caches, oracle_ref, oracle['train_cache_rows']


def build_plan(*, source_base_commit=None):
    inputs, enriched, caches, oracle_ref, oracle_rows = qualified_inputs()
    score, consumed = frozen_refs()
    head = source_base_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {'schema_version': 'gx1_ta_m5_projected_h4_line_plan_v1',
        'stage': STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_commit': head,
        'source_owners': {name: ref(REPO / name)['sha256'] for name in SOURCES},
        'inputs': inputs, 'enriched': enriched, 'h4_caches': caches,
        'h4_physical_oracle': oracle_ref, 'h4_oracle_rows': oracle_rows,
        'frozen_fit_score': score, 'consumed_ema': consumed,
        'field_map': FIELD_MAP, 'output_root': str(ROOT),
        'fit_rows': replay.FIT_ROWS, 'later_rows': ema_inputs.LATER_ROWS,
        'model_forwards': 0, 'optimizer_updates': 0, 'max_seconds': MAX_SECONDS,
        'h4_grid': 'UTC02_06_10_14_18_22_last_closed',
        'projection': 'L0=H4close +/- dist_atr*ATR; L(D)=L0+slope_atr_per_bar*ATR*(D-H4close_time)/4h',
        'projection_role': 'new_fractional_H4_bar_research_convention',
        'event': 'UP_BREAK iff prior_M5close<=same_line_at_prior_D and current_M5close>line_at_D and close_rises; mirrored DOWN_BREAK; both directions ambiguous',
        'complete_clock': '48_observed_M5_bars_in_closed_H4_plus_contiguous_M5_from_H4close_to_D',
        'evaluation': 'all_events_F_H4_known_side_vs_frozen_model_and_FIT_prior; complete_h480_event_quality_vs_frozen_model_sign_and_always_LONG',
        'execution': {'class': 'audit', 'memory': '4G', 'swap': '512M'},
        'training_authorized': False, 'test_authorized': False,
        'online_authorized': False, 'broker_access_allowed': False,
        'test_data_used': False, 'economic_edge_proven': False}


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                   text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO,
                                    text=True).strip() == 'work/gx1-current'
            and subprocess.run(['git', 'merge-base', '--is-ancestor',
                                plan['source_base_commit'], head], cwd=REPO,
                               capture_output=True).returncode == 0
            and not subprocess.check_output(['git', 'status', '--porcelain'],
                                            cwd=REPO, text=True)
            and set(plan['source_owners']) == set(SOURCES)
            and all(ref(REPO / name)['sha256'] == plan['source_owners'][name]
                    for name in SOURCES))


def inspect(plan_sha, *, admitted):
    need(ref(PLAN)['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OUTPUT')
    plan = read(ref(PLAN))
    need(plan == build_plan(source_base_commit=plan.get('source_base_commit'))
         and Path.cwd().resolve() == REPO and source_clean(plan),
         'FIXED_SCOPE_SOURCE')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    gate = policy.get(STAGE, {})
    previous = policy.get(replay.STAGE, {})
    need(previous.get('status') == 'REVIEWED_CONSUMED_FIT_SF_WEAK_EDGE_UNPROVEN'
         and previous.get('result') == plan['frozen_fit_score']['result']
         and previous.get('terminal') == plan['frozen_fit_score']['terminal']
         and previous.get('fit_capture') == plan['frozen_fit_score']['capture']
         and previous.get('execution_authorized') is False
         and previous.get('remaining_invocations') == 0
         and policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('full_epoch_training_allowed') is False
         and policy.get('native_launch_authorized') is False
         and policy.get('full_val_allowed') is False
         and policy.get('exit_training_and_evaluation_this_wave') is False
         and gate.get('plan') == ref(PLAN)
         and gate.get('output_root') == str(ROOT)
         and gate.get('fit_rows') == replay.FIT_ROWS
         and gate.get('later_rows') == ema_inputs.LATER_ROWS
         and gate.get('model_forwards') == gate.get('optimizer_updates') == 0
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('online_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('test_data_used') is False
         and gate.get('relaunch_allowed') is False, 'CLOSED_POLICY')
    if admitted:
        need(gate.get('status') == 'ADMITTED_ONE_USE'
             and gate.get('execution_authorized') is True
             and gate.get('remaining_invocations') == gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0,
             'ONE_USE_ADMISSION')
    else:
        need(gate.get('status') == 'PREPARED_NOT_ADMITTED'
             and gate.get('execution_authorized') is False
             and gate.get('remaining_invocations') == 0, 'NOT_ADMITTED')
    return plan, policy_ref


def project_line(h4_close, atr_bps, distance, slope, *, above, fraction):
    """Affine extension of the emitted integer-H4-bar geometry to 0<=f<1."""
    import numpy as np
    need(all(np.isfinite(v) for v in (h4_close, atr_bps, distance, slope, fraction))
         and h4_close > 0 and atr_bps > 0 and distance >= 0
         and 0 <= fraction < 1, 'PROJECT_DOMAIN')
    atr = h4_close * atr_bps / 10000.0
    return h4_close + (distance if above else -distance) * atr + slope * atr * fraction


def classify_break(prior_close, current_close, prior_lines, current_lines):
    """Only confirmed M5 closes cross a frozen H4 line; no touch inference."""
    import numpy as np
    need(np.isfinite(prior_close) and np.isfinite(current_close)
         and len(prior_lines) == len(current_lines) <= 2, 'BREAK_INPUT')
    up = down = False
    for before, now in zip(prior_lines, current_lines):
        need(np.isfinite(before) and np.isfinite(now), 'BREAK_LINE')
        up |= prior_close <= before and current_close > now and current_close > prior_close
        down |= prior_close >= before and current_close < now and current_close < prior_close
    if up and down:
        return AMBIGUOUS
    return UP_BREAK if up else DOWN_BREAK if down else NONE


def _axes(plan):
    """Read only axis arrays; all outcome arrays remain unopened."""
    import numpy as np
    targets = plan['inputs']['fit_targets']['path']
    with np.load(targets, allow_pickle=False) as z:
        parent, issues = z['parent_rows'], z['issue_ns']
    need(parent.shape == issues.shape == (ema_inputs.LATER_STOP,)
         and parent.dtype == issues.dtype == np.int64
         and np.all(np.diff(parent) > 0)
         and np.all(np.diff(issues) > 0), 'TARGET_AXIS')
    phases = {'FIT': slice(0, replay.FIT_ROWS),
              'LATER': slice(ema_inputs.LATER_START, ema_inputs.LATER_STOP)}
    for phase, span in phases.items():
        capture = (plan['frozen_fit_score']['capture'] if phase == 'FIT'
                   else plan['consumed_ema']['final_capture'])
        with np.load(capture['path'], allow_pickle=False) as z:
            cp, ci = z['parent_rows'], z['issue_ns']
        need(np.array_equal(cp, parent[span]) and np.array_equal(ci, issues[span]),
             'CAPTURE_TARGET_AXIS_' + phase)
    return {phase: (parent[span], issues[span]) for phase, span in phases.items()}


def _m5_closes(binding, low_ns, high_ns):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(binding['path'])
    names = parquet.schema_arrow.names
    need('time' in names and 'close' in names, 'M5_COLUMNS')
    read_groups = []
    time_col = names.index('time')
    for k in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(k).column(time_col).statistics
        need(stats is not None and stats.has_min_max, 'M5_ROWGROUP_TIME_STATS')
        lo, hi = pd.Timestamp(stats.min), pd.Timestamp(stats.max)
        lo = lo.tz_localize('UTC') if lo.tz is None else lo.tz_convert('UTC')
        hi = hi.tz_localize('UTC') if hi.tz is None else hi.tz_convert('UTC')
        if hi.value >= low_ns and lo.value <= high_ns:
            read_groups.append(k)
    need(read_groups and len(read_groups) <= 20, 'M5_BOUNDED_ROWGROUPS')
    frames = [parquet.read_row_group(k, columns=['time', 'close']).to_pandas()
              for k in read_groups]
    times = np.concatenate([pd.DatetimeIndex(pd.to_datetime(f['time'], utc=True)).asi8
                            for f in frames])
    closes = np.concatenate([f['close'].to_numpy(dtype=np.float64) for f in frames])
    need(len(times) == len(closes) and np.all(np.diff(times) > 0)
         and np.isfinite(closes).all() and np.all(closes > 0)
         and np.all(times % M5_NS == 0), 'M5_CLOCK_CLOSE')
    return times, closes, read_groups


def _closed_h4_and_event_axes(plan, phase_axes, deadline):
    import numpy as np
    import pandas as pd
    from gx1.features.htf_features import multi_tf_bar_label, multi_tf_resample
    combined = np.concatenate([phase_axes[p][1] for p in ('FIT', 'LATER')])
    need(np.all(combined % M5_NS == 0) and np.all(np.diff(combined) > 0),
         'UTC_DECISION_GRID')
    h_start = multi_tf_bar_label(pd.DatetimeIndex(combined, tz='UTC') -
                                  pd.Timedelta(hours=4), 'H4').asi8
    need(np.array_equal(h_start,
         ((combined - H4_NS - 2*3600*10**9) // H4_NS)*H4_NS + 2*3600*10**9),
         'H4_SOURCE_LABEL_PARITY')
    h4_times = np.load(plan['h4_caches']['ts']['path'], mmap_mode='r')
    h4_feats = np.load(plan['h4_caches']['feats']['path'], mmap_mode='r')
    need(h4_times.ndim == 1 and h4_feats.shape == (len(h4_times), 253)
         and h4_times.dtype == np.int64 and np.all(np.diff(h4_times) > 0)
         and np.all((h4_times - 2*3600*10**9) % H4_NS == 0), 'H4_CACHE_AXIS')
    ix = np.searchsorted(h4_times, h_start)
    present = ix < len(h4_times)
    present[present] &= h4_times[ix[present]] == h_start[present]
    oracle = plan['h4_oracle_rows']
    oracle_mask = present & (ix >= oracle['first']) & (ix <= oracle['last_inclusive'])
    times, closes, groups = _m5_closes(plan['enriched'],
                                        int(h_start.min()), int(combined.max()-M5_NS))
    unique_h = np.unique(h_start[oracle_mask])
    complete = {}
    close_h = {}
    for h in unique_h:
        j = int(np.searchsorted(times, h))
        good = (j + 48 <= len(times) and
                np.array_equal(times[j:j+48], h + np.arange(48)*M5_NS))
        complete[int(h)] = good
        if good:
            close_h[int(h)] = float(closes[j+47])
    # Independent V4 resample close parity for every complete closed H4 bar.
    series = pd.Series(closes, index=pd.DatetimeIndex(times, tz='UTC'))
    v4_close = multi_tf_resample(series, 'H4').last()
    need(all(np.isclose(close_h[int(h)], v4_close.loc[pd.Timestamp(h, tz='UTC')],
                        rtol=0, atol=1e-9)
             for h in unique_h if complete[int(h)]), 'V4_H4_CLOSE_PARITY')
    code = np.full(len(combined), NO_H4, dtype=np.uint8)
    parity_rows = 0
    for i, (decision, h, cache_row) in enumerate(zip(combined, h_start, ix)):
        if time.monotonic() >= deadline:
            raise TimeoutError('PROJECTED_H4_LINE_INTERNAL_DEADLINE')
        if not present[i]:
            continue
        if not oracle_mask[i]:
            code[i] = OUTSIDE_ORACLE
            continue
        if not complete.get(int(h), False):
            code[i] = INCOMPLETE_H4
            continue
        # The previous close must use this same H4 snapshot at its close time.
        if decision <= h + H4_NS or decision >= h + 2*H4_NS:
            code[i] = GAP
            continue
        first = h + H4_NS
        n = int((decision - first) // M5_NS)
        j = int(np.searchsorted(times, first))
        if (n < 1 or j+n > len(times) or
            not np.array_equal(times[j:j+n], first + np.arange(n)*M5_NS)):
            code[i] = GAP
            continue
        current = float(closes[j+n-1])
        prior = close_h[int(h)] if n == 1 else float(closes[j+n-2])
        features = h4_feats[cache_row]
        atr_bps = float(features[FIELD_MAP['atr_bps_14']])
        need(np.isfinite(atr_bps) and atr_bps > 0, 'H4_ATR')
        fraction = float((decision - first) / H4_NS)
        prior_fraction = fraction - M5_NS/H4_NS
        lines_before, lines_now = [], []
        for side in ('above', 'below'):
            count = float(features[FIELD_MAP[f'geomline_{side}_active_count']])
            need(np.isfinite(count) and count >= 0 and count == int(count),
                 'H4_LINE_COUNT')
            if not count:
                continue
            distance = float(features[FIELD_MAP[f'geomline_{side}_dist_atr']])
            slope = float(features[FIELD_MAP[f'geomline_{side}_slope_atr_per_bar']])
            h4_close = close_h[int(h)]
            line_at_close = project_line(h4_close, atr_bps, distance, slope,
                                         above=side == 'above', fraction=0.0)
            atr = h4_close*atr_bps/10000.0
            expected = h4_close + (distance if side == 'above' else -distance)*atr
            need(np.isclose(line_at_close, expected, rtol=0, atol=1e-9)
                 and (line_at_close >= h4_close-1e-9 if side == 'above'
                      else line_at_close <= h4_close+1e-9),
                 'H4_INTEGER_SOURCE_FORMULA_PARITY')
            parity_rows += 1
            lines_before.append(project_line(h4_close, atr_bps, distance, slope,
                                             above=side == 'above', fraction=prior_fraction))
            lines_now.append(project_line(h4_close, atr_bps, distance, slope,
                                          above=side == 'above', fraction=fraction))
        code[i] = (classify_break(prior, current, lines_before, lines_now)
                   if lines_now else NO_LINE)
    need(parity_rows > 0, 'NO_LINE_RECONSTRUCTION_EXERCISED')
    return {'code': code, 'h4_index': ix, 'h4_start_ns': h_start,
            'm5_rowgroups': groups, 'complete_h4_bars': int(sum(complete.values())),
            'integer_formula_consistency_lines': parity_rows}


def _readouts(plan, axes, geometry):
    import numpy as np
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import auc_with_ties
    prior = read(plan['inputs']['priors'])
    need(prior['fit_rows'] == replay.FIT_ROWS and
         len(prior['tasks']['future']['raw_logits']) == 5, 'FROZEN_PRIOR')
    prior_logit = float(prior['tasks']['future']['raw_logits'][3][1])
    need(np.isfinite(prior_logit), 'FROZEN_PRIOR_FINITE')
    full = fit_owner.load_targets(plan['inputs']['fit_targets'], ema_inputs.LATER_STOP)
    phase_spans = {'FIT': slice(0, replay.FIT_ROWS),
                   'LATER': slice(ema_inputs.LATER_START, ema_inputs.LATER_STOP)}
    reports, arrays = {}, {}
    cursor = 0
    for phase, span in phase_spans.items():
        parent, issues = axes[phase]
        n = len(parent)
        code = geometry['code'][cursor:cursor+n]
        h4_index = geometry['h4_index'][cursor:cursor+n]
        cursor += n
        targets = fit_owner.target_slice(full, span.start, span.stop)
        need(np.array_equal(parent, targets['parent_rows'])
             and np.array_equal(issues, targets['issue_ns']), 'OUTCOME_AXIS_' + phase)
        capture = plan['frozen_fit_score']['capture'] if phase == 'FIT' else \
                  plan['consumed_ema']['final_capture']
        with np.load(capture['path'], allow_pickle=False) as z:
            allowed = z['native_F_allowed'][:, 3]
            eligible = z['native_F_eligible'][:, 3]
            mature = z['native_F_mature'][:, 3]
            logit = z['native_F_raw'][:, 3, 1].astype(np.float64)
        need(allowed.shape == eligible.shape == mature.shape == logit.shape == (n,)
             and np.isfinite(logit).all(), 'FROZEN_F_SHAPE_' + phase)
        event = np.isin(code, (UP_BREAK, DOWN_BREAK))
        event_up = code == UP_BREAK
        known = eligible & mature & np.isin(allowed, (1, 4))
        known_event = known & event
        actual_up = allowed == 4
        model_up = logit >= 0
        prior_up = prior_logit >= 0
        prob_model = 1.0/(1.0+np.exp(-np.clip(logit, -80, 80)))
        prob_prior = 1.0/(1.0+np.exp(-np.clip(prior_logit, -80, 80)))
        known_report = {'rows': int(known_event.sum()),
            'actual_up_fraction': float(actual_up[known_event].mean()) if known_event.any() else None,
            'event_up_fraction': float(event_up[known_event].mean()) if known_event.any() else None,
            'event_side_accuracy': float((event_up[known_event] == actual_up[known_event]).mean()) if known_event.any() else None,
            'frozen_model_side_accuracy': float((model_up[known_event] == actual_up[known_event]).mean()) if known_event.any() else None,
            'frozen_fit_prior_side_accuracy': float((prior_up == actual_up[known_event]).mean()) if known_event.any() else None,
            'event_side_auc': auc_with_ties(event_up[known_event].astype(np.float64), actual_up[known_event], np),
            'frozen_model_auc': auc_with_ties(logit[known_event], actual_up[known_event], np),
            'frozen_fit_prior_auc': auc_with_ties(np.full(int(known_event.sum()), prior_logit), actual_up[known_event], np),
            'frozen_model_brier': float(np.square(prob_model[known_event]-actual_up[known_event]).mean()) if known_event.any() else None,
            'frozen_fit_prior_brier': float(np.square(prob_prior-actual_up[known_event]).mean()) if known_event.any() else None}
        complete = targets['path_observed'][:, 5].astype(bool)
        path_event = event & complete
        actual_q = targets['net_bps'][:, 5, :].astype(np.float64) - \
                   targets['pre_mfe_mae_upper_bps'][:, 5, :].astype(np.float64)
        need(actual_q.shape == (n, 2)
             and np.isfinite(actual_q[complete]).all(), 'H480_ACTUAL_' + phase)
        ii = np.flatnonzero(path_event)
        event_q = actual_q[ii, np.where(event_up[ii], 0, 1)]
        model_q = actual_q[ii, np.where(model_up[ii], 0, 1)]
        long_q = actual_q[ii, 0]
        h480_report = {'complete_event_rows': len(ii),
            'event_chosen_quality_bps_mean': float(event_q.mean()) if len(ii) else None,
            'frozen_model_sign_quality_bps_mean': float(model_q.mean()) if len(ii) else None,
            'always_long_quality_bps_mean': float(long_q.mean()) if len(ii) else None,
            'event_minus_model_bps_mean': float((event_q-model_q).mean()) if len(ii) else None,
            'event_minus_always_long_bps_mean': float((event_q-long_q).mean()) if len(ii) else None,
            'role': 'overlapping_h480_markout_not_trading_PnL'}
        reports[phase] = {'all_rows': n, 'event_rows': int(event.sum()),
            'event_rows_without_known_F_H4': int((event & ~known).sum()),
            'event_rows_without_complete_h480_path': int((event & ~complete).sum()),
            'coverage_codes': {
            name: int((code == k).sum()) for k, name in enumerate(CODES)},
            'known_F_H4_all_rows': int(known.sum()),
            'known_F_H4_on_events': known_report,
            'h480_complete_all_rows': int(complete.sum()),
            'h480_complete_event_quality': h480_report}
        arrays[phase] = {'parent_rows': parent, 'issue_ns': issues,
            'event_code': code, 'h4_cache_row': h4_index,
            'F_H4_known': known, 'F_H4_up': actual_up,
            'frozen_F_H4_logit': logit.astype(np.float32),
            'h480_complete': complete, 'h480_quality_long_short': actual_q.astype(np.float32)}
    need(cursor == len(geometry['code']), 'PHASE_CURSOR')
    return reports, arrays, prior_logit


def run(plan_sha):
    import numpy as np
    plan, policy_at_claim = inspect(plan_sha, admitted=True)
    started = time.monotonic()
    deadline = started + MAX_SECONDS - 60
    ROOT.mkdir(parents=True, exist_ok=False)
    claim = fit_owner.publish('CLAIM.json', {'status': 'CLAIMED_ONE_USE',
        'stage': STAGE, 'plan': ref(PLAN), 'policy_at_claim': policy_at_claim,
        'source_base_commit': plan['source_base_commit'],
        'max_invocations': 1, 'remaining_invocations': 0,
        'test_data_used': False}, root=ROOT)
    result_ref = None
    try:
        axes = _axes(plan)
        geometry = _closed_h4_and_event_axes(plan, axes, deadline)
        need(time.monotonic() < deadline, 'TIME_BEFORE_OUTCOME')
        reports, arrays, prior_logit = _readouts(plan, axes, geometry)
        need(time.monotonic() < deadline and source_clean(plan)
             and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_at_claim
             and ref(PLAN)['sha256'] == plan_sha
             and ref(Path(plan['enriched']['path'])) == plan['enriched']
             and all(ref(Path(plan['h4_caches'][key]['path'])) == plan['h4_caches'][key]
                     for key in ('feats', 'ts')), 'SOURCE_INPUT_POLICY_TIME_BOOKEND')
        capture = {}
        for phase in ('FIT', 'LATER'):
            for key, value in arrays[phase].items():
                capture[phase + '_' + key] = value
        stream = io.BytesIO()
        np.savez_compressed(stream, **capture)
        capture_ref = fit_owner.publish('EVENT_READOUT.npz', stream.getvalue(), root=ROOT)
        result_ref = fit_owner.publish('RESULT.json', {
            'status': 'COMPLETE_READ_ONLY_PROJECTED_H4_LINE_EVENT_MEASUREMENT',
            'stage': STAGE, 'plan': ref(PLAN), 'claim': claim,
            'policy_at_claim': policy_at_claim, 'source_unchanged': True,
            'input_refs': plan['inputs'], 'enriched': plan['enriched'],
            'h4_caches': plan['h4_caches'], 'h4_physical_oracle': plan['h4_physical_oracle'],
            'frozen_fit_score': plan['frozen_fit_score'], 'consumed_ema': plan['consumed_ema'],
            'event_capture': capture_ref, 'm5_rowgroups_read': geometry['m5_rowgroups'],
            'complete_closed_h4_bars': geometry['complete_h4_bars'],
            'integer_formula_consistency_lines': geometry['integer_formula_consistency_lines'],
            'source_parity_basis': 'bound_H4_physical_oracle_zero_field_mismatches; integer_reconstruction_is_arithmetic_only',
            'frozen_fit_prior_H4_side_logit': prior_logit,
            'partitions': reports, 'model_forwards': 0,
            'optimizer_constructions': 0, 'optimizer_updates': 0,
            'normalization_fits': 0, 'test_data_used': False,
            'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started}, root=ROOT)
        fit_owner.publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'stage': STAGE, 'plan': ref(PLAN), 'claim': claim, 'result': result_ref,
            'policy_at_claim': policy_at_claim, 'source_unchanged': True,
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False}, root=ROOT)
    except BaseException as error:
        fit_owner.publish('TERMINAL.json', {'status': 'FAILED_CONSUMED',
            'exit_code': 1, 'stage': STAGE, 'plan': ref(PLAN), 'claim': claim,
            'result': result_ref, 'policy_at_claim': policy_at_claim,
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False}, root=ROOT)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    need(args.plan == str(PLAN) and args.out_dir == str(ROOT), 'CLI_SCOPE')
    if args.preflight:
        plan, policy = inspect(args.plan_sha256, admitted=False)
        axes = _axes(plan)
        geometry = _closed_h4_and_event_axes(
            plan, axes, time.monotonic() + MAX_SECONDS - 60)
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'plan': ref(PLAN),
            'policy': policy, 'fit_rows': plan['fit_rows'],
            'later_rows': plan['later_rows'], 'model_forwards': 0,
            'optimizer_updates': 0, 'max_seconds': MAX_SECONDS,
            'output_root_absent': True, 'claim_created': False,
            'm5_rowgroups_read': geometry['m5_rowgroups'],
            'complete_closed_h4_bars': geometry['complete_h4_bars'],
            'integer_formula_consistency_lines': geometry['integer_formula_consistency_lines'],
            'price_event_code_counts': {name: int((geometry['code'] == k).sum())
                                        for k, name in enumerate(CODES)},
            'test_data_used': False}, sort_keys=True))
    else:
        run(args.plan_sha256)


if __name__ == '__main__':
    main()
