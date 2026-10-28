"""Bound one-shot audit of source-decimal midpoint signs; never fit or trade."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import gzip
import json
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'independent_direction_decimal_sign_audit_20261028_002'
START_NS = 1714521600000000000  # 2024-05-01T00:00Z
CUTOFF_NS = 1767225600000000000  # 2026-01-01T00:00Z
TEST_NS = 1782864000000000000  # 2026-07-01T00:00Z
MINUTE_NS = 60 * 10**9
PARENTS = 118249
FOLD_TRAIN = 47602
FOLD_LATER_END = 64890
TRAIN_CUTOFF = 1735689600000000000
LATER_CUTOFF = 1743465600000000000
INT64_MAX = np.iinfo(np.int64).max
SOURCE_PATHS = (
    'gx1/scripts/audit_entry_direction_decimal_sign_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/xau_tape_provenance_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_entry_direction_decimal_sign_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_DECIMAL_AUDIT_' + reason)


def price_parts(value):
    """Parse the vendor decimal without arithmetic or a guessed price quantum."""
    require(type(value) is str and 0 < len(value) <= 128, 'PRICE_STRING')
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise RuntimeError('DIRECTION_DECIMAL_AUDIT_PRICE_DECIMAL') from exc
    require(number.is_finite() and number > 0, 'PRICE_DOMAIN')
    parts = number.as_tuple()
    coefficient = int(''.join(map(str, parts.digits)))
    require(parts.sign == 0 and coefficient > 0 and isinstance(parts.exponent, int), 'PRICE_PARTS')
    return coefficient, parts.exponent


def decimal_midpoint_numerators(quotes):
    """Convert source strings to exact common-grid bid+ask numerators.

    The int64 bound is a fail-closed resource/domain constraint, not rounding.
    """
    require(len(quotes) > 0, 'EMPTY_QUOTES')
    parsed_bid, parsed_ask = [], []
    for bid, ask in quotes:
        parsed_bid.append(price_parts(bid)); parsed_ask.append(price_parts(ask))
    precision = max(0, -min(e for _, e in parsed_bid + parsed_ask))
    require(precision <= 18, 'DECIMAL_GRID_TOO_FINE_FOR_INT64')
    sides = []
    for parsed in (parsed_bid, parsed_ask):
        values = np.empty(len(parsed), np.int64)
        for i, (coefficient, exponent) in enumerate(parsed):
            power = precision + exponent
            require(0 <= power <= 18, 'DECIMAL_GRID_EXPONENT')
            multiplier = 10**power
            require(coefficient <= (INT64_MAX // 2) // multiplier, 'DECIMAL_GRID_OVERFLOW')
            values[i] = coefficient * multiplier
        sides.append(values)
    require(np.all(sides[0] <= sides[1]), 'CROSSED_EXACT_QUOTES')
    return sides[0] + sides[1], precision


def exact_signs_for_targets(source_times, source_rows, midpoint_numerators, *, decisions,
        entry_rows, terminal_times, observed, terminal_rows, cutoff_ns, block_rows=1024):
    """Exact sign from separately verified quote row/clock bindings."""
    size, width = terminal_times.shape
    require(source_times.dtype == source_rows.dtype == midpoint_numerators.dtype == np.int64
        and len(source_times) == len(source_rows) == len(midpoint_numerators) > 0
        and np.all(np.diff(source_times) > 0) and np.all(np.diff(source_rows) > 0)
        and np.all(midpoint_numerators > 0), 'SOURCE_AXIS')
    require(decisions.dtype == entry_rows.dtype == terminal_times.dtype == terminal_rows.dtype == np.int64
        and observed.dtype == np.bool_ and decisions.shape == entry_rows.shape == (size,)
        and observed.shape == terminal_rows.shape == (size, width)
        and np.all(decisions < cutoff_ns), 'TARGET_AXES')
    start = np.searchsorted(source_times, decisions)
    require(np.all(start < len(source_times)) and np.array_equal(source_times[start], decisions)
        and np.array_equal(source_rows[start], entry_rows), 'ENTRY_JOIN')
    out = np.zeros((size, width), np.int8)
    for left in range(0, size, block_rows):
        right = min(left + block_rows, size)
        wanted = terminal_times[left:right]
        pos = np.searchsorted(source_times, wanted)
        found = (wanted < cutoff_ns) & (pos < len(source_times))
        found[found] &= source_times[pos[found]] == wanted[found]
        require(np.array_equal(found, observed[left:right]), 'TERMINAL_OBSERVATION')
        rows = np.full(wanted.shape, -1, np.int64)
        rows[found] = source_rows[pos[found]]
        require(np.array_equal(rows, terminal_rows[left:right]), 'TERMINAL_JOIN')
        origin = np.broadcast_to(midpoint_numerators[start[left:right], None], wanted.shape)
        out[left:right][found] = np.sign(midpoint_numerators[pos[found]] - origin[found]).astype(np.int8)
    return out


def confusion_by_horizon(saved, exact, observed, mature):
    require(saved.dtype == exact.dtype == np.int8 and observed.dtype == mature.dtype == np.bool_
        and saved.shape == exact.shape == observed.shape == mature.shape
        and np.isin(saved, (-1, 0, 1)).all() and np.isin(exact, (-1, 0, 1)).all(), 'CONFUSION_DOMAIN')
    rows = []
    for j in range(saved.shape[1]):
        take = observed[:, j] & mature[:, j]
        old, new = saved[take, j], exact[take, j]
        matrix = np.bincount((old.astype(np.int64) + 1)*3 + new.astype(np.int64) + 1, minlength=9).reshape(3, 3)
        rows.append({'horizon_minutes': j + 1, 'observed_mature': int(take.sum()),
            'saved_by_exact_sign_order_minus_zero_plus': matrix.tolist(),
            'saved_nonzero_to_exact_zero': int(((old != 0) & (new == 0)).sum()),
            'saved_zero_to_exact_nonzero': int(((old == 0) & (new != 0)).sum()),
            'nonzero_sign_flips': int((old * new == -1).sum())})
    return rows


def score_versions(predictions, exact, stored, observed, terminal_times, cutoff_ns, report):
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    from gx1.contracts.entry_native_direction_context_report_v1 import _panel
    require(predictions.ndim == 2 and predictions.shape == exact.shape == stored.shape == observed.shape
        == terminal_times.shape and predictions.shape[1] == 60 and np.isfinite(predictions).all(), 'SCORE_SHAPES')
    rows = []
    for j in range(60):
        mature = terminal_times[:, j] < cutoff_ns
        versions = {}
        for name, signs in (('stored', stored), ('source_decimal', exact)):
            panel = _panel(predictions[:, j], signs[:, j], observed[:, j], mature)
            eligible = observed[:, j] & mature & (signs[:, j] != 0)
            labels = np.ascontiguousarray((signs[eligible, j] == 1).astype(np.int8))
            values = np.ascontiguousarray(predictions[eligible, j], dtype=np.float64)
            rank = direction_rank_diagnostics(labels, values, score_kind='logit_up')
            versions[name] = {'eligible': panel['counts']['eligible_cells'], 'auc': rank['auc'],
                'accuracy': panel['accuracy'], 'logloss': panel['logloss'], 'brier': panel['brier']}
        original = report['model_metrics']['per_horizon'][j]
        for key in ('accuracy', 'logloss', 'brier'):
            require(np.isclose(versions['stored'][key], original[key], rtol=0, atol=1e-12), 'ORIGINAL_SCORE:' + key)
        require(versions['stored']['eligible'] == original['counts']['eligible_cells']
            and np.isclose(versions['stored']['auc'], original['rank']['auc'], rtol=0, atol=1e-12), 'ORIGINAL_RANK')
        rows.append({'horizon_minutes': j + 1, **versions})
    return rows


def _read_target(path):
    axis = ('parent_rows', 'bar_start_ns', 'decision_ns', 'dynamic_entry_m1_row')
    fields = axis + ('terminal_time_ns', 'terminal_source_row', 'observed', 'mid_move', 'mid_return_bps', 'sign')
    with Path(path).open('rb') as handle:
        with zipfile.ZipFile(handle) as archive:
            require(len(archive.infolist()) == len(fields)
                and sum(item.file_size for item in archive.infolist()) <= 800 * 1024**2, 'TARGET_ARCHIVE_SIZE')
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            require(set(archive.files) == set(fields), 'TARGET_FIELDS')
            target = {name: archive[name].copy() for name in fields}
    dtypes = {'terminal_time_ns': np.int64, 'terminal_source_row': np.int64, 'observed': np.bool_,
        'mid_move': np.float64, 'mid_return_bps': np.float64, 'sign': np.int8}
    require(all(target[name].dtype == np.int64 and target[name].shape == (PARENTS,) for name in axis)
        and all(target[name].dtype == dtype and target[name].shape == (PARENTS, 60)
            for name, dtype in dtypes.items()), 'TARGET_DTYPE_SHAPE')
    observed, sign = target['observed'], target['sign']
    move, bps, rows = target['mid_move'], target['mid_return_bps'], target['terminal_source_row']
    require(np.array_equal(target['parent_rows'], np.arange(PARENTS, dtype=np.int64))
        and np.all(np.diff(target['decision_ns']) > 0)
        and np.all((target['decision_ns'] >= START_NS) & (target['decision_ns'] < CUTOFF_NS))
        and np.array_equal(target['decision_ns'], target['bar_start_ns'] + 5 * MINUTE_NS)
        and np.array_equal(target['terminal_time_ns'], target['decision_ns'][:, None]
            + np.arange(1, 61, dtype=np.int64)[None, :] * MINUTE_NS)
        and np.isfinite(move).all() and np.isfinite(bps).all()
        and np.isin(sign, (-1, 0, 1)).all()
        and np.array_equal(np.sign(move).astype(np.int8), sign)
        and np.array_equal(np.sign(bps).astype(np.int8), sign)
        and np.all(rows[observed] >= 0) and np.all(rows[~observed] == -1)
        and np.all(sign[~observed] == 0) and np.all(move[~observed] == 0)
        and np.all(bps[~observed] == 0), 'TARGET_HEADER_CLOCK_SIGN')
    return target


def _read_raw_quotes(chunks, source, *, pd, closure_mask, request_bounds):
    """Read only approved pre-TEST chunks; post-cutoff prices are not selected."""
    window = (source['time_ns'] >= START_NS) & (source['time_ns'] < CUTOFF_NS)
    times, source_rows = source['time_ns'][window], source['source_row'][window]
    bids, asks = source['bid_open'][window], source['ask_open'][window]
    actual_times, pairs = [], []
    require(len(chunks) == len(request_bounds), 'CHUNK_BOUNDS')
    for ref, (begin, end) in zip(chunks, request_bounds):
        with gzip.open(ref['path'], 'rt', encoding='utf-8') as handle:
            payload = json.load(handle)
        require(isinstance(payload, dict) and set(payload) == {'schema_version', 'request', 'response'}
            and payload['schema_version'] == 'xau_oanda_native_source_chunk_v2', 'VENDOR_ENVELOPE')
        timestamp = lambda value: pd.Timestamp(value, unit='ns', tz='UTC').strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
        expected_request = {'instrument': 'XAU_USD', 'from': timestamp(begin), 'to': timestamp(end),
                            'granularity': 'M1', 'price': 'MBA'}
        require(payload['request'] == expected_request, 'VENDOR_REQUEST')
        response = payload['response']
        require(isinstance(response, dict) and response.get('instrument') == 'XAU_USD'
            and response.get('granularity') == 'M1' and isinstance(response.get('candles'), list), 'VENDOR_RESPONSE')
        candles = response['candles']
        if not candles:
            continue
        clock = pd.to_datetime([item['time'] for item in candles], utc=True, format='ISO8601').as_unit('ns').asi8
        require(np.all(np.diff(clock) > 0) and np.all((clock >= begin) & (clock < end)), 'VENDOR_CHUNK_CLOCK')
        closed = closure_mask(pd.to_datetime(clock, unit='ns', utc=True))
        selected = np.flatnonzero((clock >= START_NS) & (clock < CUTOFF_NS) & ~closed)
        for i in selected:
            row = candles[int(i)]
            require(row.get('complete') is True and isinstance(row.get('bid'), dict)
                and isinstance(row.get('ask'), dict), 'VENDOR_CANDLE')
            bid, ask = row['bid'].get('o'), row['ask'].get('o')
            require(type(bid) is str and type(ask) is str, 'VENDOR_PRICE_TYPE')
            actual_times.append(int(clock[i])); pairs.append((bid, ask))
    raw_clock = np.asarray(actual_times, dtype=np.int64)
    require(np.array_equal(raw_clock, times), 'VENDOR_CANONICAL_CLOCK')
    raw_bid = np.fromiter((float(b) for b, _ in pairs), dtype=np.float64, count=len(pairs))
    raw_ask = np.fromiter((float(a) for _, a in pairs), dtype=np.float64, count=len(pairs))
    require(np.array_equal(raw_bid, bids) and np.array_equal(raw_ask, asks), 'VENDOR_CANONICAL_FLOAT64')
    midpoint_numerators, precision = decimal_midpoint_numerators(pairs)
    return times, source_rows, midpoint_numerators, precision


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    import pandas as pd
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import pyarrow.dataset
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics  # preload scorer
    from gx1.contracts.entry_native_direction_context_report_v1 import _panel  # preload scorer
    from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'PRELOADED_MODEL')
    initial = candle.git_state()
    require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'chunks', 'sources', 'output_dir',
                'wall_seconds', 'start_ns', 'cutoff_ns', 'test_ns', 'parents', 'test_data_used'}
            and plan['schema_version'] == 'gx1_direction_decimal_sign_audit_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 600 and plan['start_ns'] == START_NS and plan['cutoff_ns'] == CUTOFF_NS
            and plan['test_ns'] == TEST_NS and plan['parents'] == PARENTS and plan['test_data_used'] is False
            and len(plan['chunks']) == 204, 'PLAN')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(plan['design']['path'] == str(REPO/'docs/DIRECTION_DECIMAL_SIGN_AUDIT_20261028.json')
            and design['schema_version'] == 'gx1_direction_decimal_sign_audit_design_v1'
            and design['source_window']['overlapping_vendor_response_chunks'] == 204, 'DESIGN')
        source_window = design['source_window']
        expected_inputs = {'native_manifest': source_window['source_manifest'],
            'native_2024': source_window['native_years']['2024'],
            'native_2025': source_window['native_years']['2025'],
            'dense_targets': {k: design['saved_targets'][k] for k in ('path', 'sha256')},
            'oof_report': design['saved_candidate_oof']['report'],
            'oof_predictions': design['saved_candidate_oof']['predictions']}
        require(plan['inputs'] == expected_inputs and set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO/name) for name, ref in plan['sources'].items()), 'EXACT_INPUTS_SOURCES')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized')), 'GLOBAL_GATES')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir'])
    require(candle.ROOT.resolve(strict=True) == candle.ROOT and out.parent == candle.ROOT
        and out.name == 'DIRECTION_DECIMAL_SIGN_AUDIT_20261028_002'
        and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    inputs = list(plan['inputs'].values()) + plan['chunks']
    require(len({r['path'] for r in inputs}) == len(inputs), 'DUPLICATE_INPUT')
    metadata = [plan_ref['path'], policy_ref['path'], plan['design']['path']]
    sys.addaudithook(candle.make_guard(out, [ref['path'] for ref in inputs], metadata,
        source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started = time.monotonic(); previous_handler = signal.getsignal(signal.SIGALRM); previous_profile = sys.getprofile()
    def expired(*unused):
        raise RuntimeError('DIRECTION_DECIMAL_AUDIT_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds'])
    sys.setprofile(candle.no_queries_or_models)
    stage = out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in inputs:
                candle.regular(ref['path']); binding.require_binding(ref, label='input')
            manifest = json.loads(Path(plan['inputs']['native_manifest']['path']).read_text())
            require(manifest['source_kind'] == 'oanda_native_mba_candles' and manifest['timestamp_semantics'] == 'bar_start_utc'
                and manifest['source_granularity'] == 'M1'
                and manifest['market_closure_contract'] == 'oanda_complete_true_scheduled_weekly_closure_excluded_v2', 'SOURCE_MANIFEST')
            expected, request_bounds = [], []
            root = Path(manifest['out_root'])
            for item in manifest['source_chunks']:
                begin = int(pd.Timestamp(item['request_from_utc']).value)
                end = int(pd.Timestamp(item['request_to_utc_exclusive']).value)
                if end > START_NS and begin < CUTOFF_NS:
                    require(end <= TEST_NS, 'CHUNK_REACHES_SEALED_TEST')
                    expected.append({'path': str(root/item['relative_path']), 'sha256': item['sha256']})
                    request_bounds.append((begin, end))
            require(expected == plan['chunks'], 'EXACT_VENDOR_CHUNK_LIST')
            years = [candle.read_year_quotes(plan['inputs']['native_' + str(year)]['path'], expected_rows=rows,
                year=year, parquet=parquet, arrow=arrow, np=np) for year, rows, offset in candle.YEAR_SPEC]
            source = {key: np.concatenate([part[key] for part in years]) for key in years[0]}
            source['source_row'] = np.concatenate([np.arange(offset, offset + rows, dtype=np.int64)
                for year, rows, offset in candle.YEAR_SPEC])
            times, source_rows, numerators, precision = _read_raw_quotes(plan['chunks'], source, pd=pd,
                closure_mask=scheduled_weekly_closure_mask, request_bounds=request_bounds)
            target = _read_target(plan['inputs']['dense_targets']['path'])
            exact = exact_signs_for_targets(times, source_rows, numerators, decisions=target['decision_ns'],
                entry_rows=target['dynamic_entry_m1_row'], terminal_times=target['terminal_time_ns'],
                observed=target['observed'], terminal_rows=target['terminal_source_row'], cutoff_ns=CUTOFF_NS)
            old, observed, terminal = target['sign'], target['observed'], target['terminal_time_ns']
            groups = {'all_118249': confusion_by_horizon(old, exact, observed, terminal < CUTOFF_NS),
                'fold1_train': confusion_by_horizon(old[:FOLD_TRAIN], exact[:FOLD_TRAIN], observed[:FOLD_TRAIN], terminal[:FOLD_TRAIN] < TRAIN_CUTOFF),
                'fold1_later': confusion_by_horizon(old[FOLD_TRAIN:FOLD_LATER_END], exact[FOLD_TRAIN:FOLD_LATER_END], observed[FOLD_TRAIN:FOLD_LATER_END], terminal[FOLD_TRAIN:FOLD_LATER_END] < LATER_CUTOFF)}
            report = json.loads(Path(plan['inputs']['oof_report']['path']).read_text())
            with np.load(plan['inputs']['oof_predictions']['path'], allow_pickle=False) as archive:
                pred = {name: archive[name].copy() for name in ('parent_rows', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up')}
            sl = slice(FOLD_TRAIN, FOLD_LATER_END)
            require(np.array_equal(pred['parent_rows'], target['parent_rows'][sl])
                and np.array_equal(pred['decision_ns'], target['decision_ns'][sl])
                and np.array_equal(pred['terminal_time_ns'], terminal[sl])
                and np.array_equal(pred['observed'], observed[sl]) and np.array_equal(pred['sign'], old[sl]), 'OOF_EXACT_TARGET_JOIN')
            scores = score_versions(pred['logit_up'], exact[sl], old[sl], observed[sl], terminal[sl], LATER_CUTOFF, report)
            input_snapshot = binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
        result = {'schema_version': 'gx1_direction_decimal_sign_audit_result_v1', 'status': 'COMPLETE',
            'source_head': args.source_head, 'plan': plan_ref, 'source_rows': len(times),
            'decimal_places_in_common_grid': precision, 'source_chunks': len(plan['chunks']),
            'vendor_to_canonical_float64_and_clock_parity': True, 'target_source_rows_and_masks_parity': True,
            'original_oof_scores_reproduced': True, 'confusions': groups, 'later_fixed_logit_scores': scores,
            'scope': 'original decimal bid/ask M1 candle opens, 118249 pre-2026 parents and 60 endpoints; one reused development fold',
            'limits': 'Point scores are not uncertainty, real quote-event timestamps, fills or economic edge; no new targets or fits.',
            'new_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
        publish(stage/'RESULT.json', result)
        publish(stage/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
            'inputs_unchanged': True, 'elapsed_seconds': time.monotonic()-started, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out/'OUTPUT')
        publish(out/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'result_path': str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False})
        raise
    finally:
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
