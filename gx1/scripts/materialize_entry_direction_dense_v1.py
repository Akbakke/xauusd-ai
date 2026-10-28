"""Materialize exact 1–60-minute direction targets from bound M1 quotes.

The legacy pure projection requires byte parity with the saved 1/5/15
endpoints. A separate decimal-grid projection preserves those legacy checks,
then computes a new target identity from caller-qualified exact source ticks.
Neither function grants file access or permission to publish/train. A fold-specific
batch masks all outcomes at or beyond its own cutoff.
"""
from __future__ import annotations

import numpy as np

from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle

HORIZONS = tuple(range(1, 61))
ANCHORS = (1, 5, 15)
MINUTE_NS = 60 * 10**9


def require(ok, reason):
    if not ok:
        raise RuntimeError('DIRECTION_DENSE_TARGET_' + reason)


def build_dense_targets(source, saved, *, cutoff_ns, block_rows=1024):
    """Compute targets without interpolation; fail if any legacy anchor differs."""
    parents, bars, decisions, pointers = (saved[k] for k in
        ('parent_rows', 'bar_start_ns', 'decision_ns', 'dynamic_entry_m1_row'))
    size = len(parents)
    require(type(cutoff_ns) is int and 0 < cutoff_ns <= np.iinfo(np.int64).max
        and type(block_rows) is int and 0 < block_rows <= 4096, 'CONFIG')
    require(size > 0 and all(isinstance(x, np.ndarray) and x.dtype == np.int64
        and x.shape == (size,) for x in (parents, bars, decisions, pointers))
        and np.all(parents >= 0) and np.all(pointers >= 0) and np.all(bars >= 0)
        and np.all(bars <= np.iinfo(np.int64).max - 5 * MINUTE_NS)
        and np.all(np.diff(parents) > 0) and np.all(np.diff(decisions) > 0)
        and np.all(decisions == bars + 5 * MINUTE_NS) and np.all(decisions % (5 * MINUTE_NS) == 0)
        and np.all(decisions < cutoff_ns)
        and np.all(decisions <= np.iinfo(np.int64).max - 60 * MINUTE_NS), 'PARENT_CLOCK')
    for name in ('time_ns', 'source_row', 'open', 'bid_open', 'ask_open'):
        require(name in source and isinstance(source[name], np.ndarray)
            and source[name].ndim == 1 and len(source[name]) == len(source['time_ns']), 'SOURCE:' + name)
    require(source['time_ns'].dtype == source['source_row'].dtype == np.int64
        and np.all(source['source_row'] >= 0) and np.all(np.diff(source['time_ns']) > 0)
        and np.all(np.diff(source['source_row']) > 0), 'SOURCE_CLOCK')
    require(all(source[k].dtype == np.float64 and np.isfinite(source[k]).all()
        and np.all(source[k] > 0) for k in ('open', 'bid_open', 'ask_open'))
        and np.all(source['ask_open'] >= source['bid_open']), 'SOURCE_PRICES')
    entry = candle.gather_quotes(source, decisions, cutoff_ns=cutoff_ns, np=np)
    require(entry['observed'].all() and np.array_equal(entry['source_row'], pointers)
        and np.all(entry['bid_open'] > 0) and np.all(entry['ask_open'] >= entry['bid_open']), 'ENTRY')
    shape = (size, len(HORIZONS))
    target = {'parent_rows': parents.copy(), 'bar_start_ns': bars.copy(),
        'decision_ns': decisions.copy(), 'dynamic_entry_m1_row': pointers.copy(),
        'terminal_time_ns': decisions[:, None] + np.asarray(HORIZONS, np.int64)[None, :] * MINUTE_NS,
        'terminal_source_row': np.full(shape, -1, np.int64), 'observed': np.zeros(shape, np.bool_),
        'mid_move': np.zeros(shape, np.float64), 'mid_return_bps': np.zeros(shape, np.float64),
        'sign': np.zeros(shape, np.int8)}
    for left in range(0, size, block_rows):
        right = min(left + block_rows, size)
        end = candle.gather_quotes(source, target['terminal_time_ns'][left:right], cutoff_ns=cutoff_ns, np=np)
        observed = end['observed']
        require(np.all(end['bid_open'][observed] > 0)
            and np.all(end['ask_open'][observed] >= end['bid_open'][observed]), 'TERMINAL_QUOTES')
        origin = (entry['bid_open'][left:right] + entry['ask_open'][left:right]) / 2.0
        terminal = (end['bid_open'] + end['ask_open']) / 2.0
        move = terminal - origin[:, None]
        bps = 10000.0 * move / origin[:, None]
        require(np.isfinite(move[observed]).all() and np.isfinite(bps[observed]).all(), 'FINITE')
        target['terminal_source_row'][left:right] = end['source_row']
        target['observed'][left:right] = observed
        target['mid_move'][left:right][observed] = move[observed]
        target['mid_return_bps'][left:right][observed] = bps[observed]
        target['sign'][left:right][observed] = np.sign(move[observed]).astype(np.int8)
    columns = np.asarray(ANCHORS, np.int64) - 1
    for name in ('terminal_time_ns', 'terminal_source_row', 'observed', 'mid_move', 'mid_return_bps', 'sign'):
        actual = np.ascontiguousarray(target[name][:, columns])
        expected = np.ascontiguousarray(saved[name])
        require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and actual.tobytes() == expected.tobytes(), 'LEGACY_PARITY:' + name)
    return target


DECIMAL_GRID_SCHEMA = 'gx1_direction_source_decimal_grid_v1'


def build_decimal_grid_dense_targets(source, saved, *, decimal_grid, cutoff_ns, block_rows=1024):
    """Project a distinct target identity from source-verified decimal integer prices.

    The caller must bind the original decimal strings, precision, source rows,
    period and the resulting new artifact identity separately. Float parity alone
    cannot prove original decimal precision. The legacy target/anchor path is
    first validated unchanged; its numeric targets are not reused as new labels.
    """
    require(isinstance(decimal_grid, dict) and set(decimal_grid) ==
        {'schema_version', 'decimal_places', 'bid_ticks', 'ask_ticks'}
        and decimal_grid['schema_version'] == DECIMAL_GRID_SCHEMA
        and type(decimal_grid['decimal_places']) is int
        and 0 <= decimal_grid['decimal_places'] <= 12, 'DECIMAL_GRID_HEADER')
    scale = 10 ** decimal_grid['decimal_places']
    bid, ask = decimal_grid['bid_ticks'], decimal_grid['ask_ticks']
    size = len(source['time_ns'])
    require(all(isinstance(v, np.ndarray) and v.dtype == np.int64 and v.shape == (size,)
        and not v.flags.writeable for v in (bid, ask))
        and np.all(bid > 0) and np.all(ask >= bid) and np.all(ask <= 2**52), 'DECIMAL_GRID_TICKS')
    require(np.array_equal(bid.astype(np.float64) / scale, source['bid_open'])
        and np.array_equal(ask.astype(np.float64) / scale, source['ask_open']), 'DECIMAL_GRID_SOURCE_PARITY')
    target = build_dense_targets(source, saved, cutoff_ns=cutoff_ns, block_rows=block_rows)
    times, numerator = source['time_ns'], bid + ask
    entry_at = np.searchsorted(times, target['decision_ns'])
    require(np.all(entry_at < size) and np.array_equal(times[entry_at], target['decision_ns'])
        and np.array_equal(source['source_row'][entry_at], target['dynamic_entry_m1_row']), 'DECIMAL_GRID_ENTRY')
    for left in range(0, len(entry_at), block_rows):
        right = min(left + block_rows, len(entry_at))
        observed = target['observed'][left:right]
        endpoints = target['terminal_time_ns'][left:right]
        positions = np.searchsorted(times, endpoints)
        require(np.all(positions[observed] < size)
            and np.array_equal(times[positions[observed]], endpoints[observed])
            and np.array_equal(source['source_row'][positions[observed]],
                target['terminal_source_row'][left:right][observed]), 'DECIMAL_GRID_TERMINAL')
        origin = np.broadcast_to(numerator[entry_at[left:right], None], endpoints.shape)
        delta = numerator[positions[observed]] - origin[observed]
        move = delta.astype(np.float64) / (2 * scale)
        bps = (10000.0 * delta.astype(np.float64)) / origin[observed]
        target['mid_move'][left:right][observed] = move
        target['mid_return_bps'][left:right][observed] = bps
        target['sign'][left:right][observed] = np.sign(delta).astype(np.int8)
    require(np.array_equal(np.sign(target['mid_move']).astype(np.int8), target['sign'])
        and np.array_equal(np.sign(target['mid_return_bps']).astype(np.int8), target['sign'])
        and np.all(target['sign'][~target['observed']] == 0), 'DECIMAL_GRID_OUTPUT')
    return target


def causal_target_batch(target, indices, *, cutoff_ns):
    """Expose no future label at the strict fold boundary; keep maturity separate."""
    require(isinstance(indices, np.ndarray) and indices.dtype == np.int64 and indices.ndim == 1
        and len(indices) > 0 and np.all(indices >= 0) and np.all(indices < len(target['parent_rows']))
        and type(cutoff_ns) is int and cutoff_ns > 0, 'BATCH_HEADER')
    decision = target['decision_ns'][indices]
    require(np.all(decision < cutoff_ns), 'BATCH_DECISION_CUTOFF')
    terminal = target['terminal_time_ns'][indices]
    require(terminal.shape == (len(indices), 60)
        and np.array_equal(terminal, decision[:, None] + np.asarray(HORIZONS, np.int64)[None, :] * MINUTE_NS), 'BATCH_CLOCK')
    stored_observed, stored_sign = target['observed'][indices], target['sign'][indices]
    require(terminal.dtype == np.int64 and stored_observed.dtype == np.bool_
        and stored_sign.dtype == np.int8 and stored_observed.shape == stored_sign.shape == terminal.shape
        and np.isin(stored_sign, (-1, 0, 1)).all() and np.all(stored_sign[~stored_observed] == 0), 'BATCH_VALUES')
    mature = terminal < cutoff_ns
    observed = stored_observed & mature
    sign = np.where(observed, stored_sign, np.int8(0)).astype(np.int8)
    return {'parent_rows': target['parent_rows'][indices].copy(), 'decision_ns': decision.copy(),
        'terminal_time_ns': terminal.copy(), 'mature': mature, 'observed': observed, 'sign': sign}


def main(argv=None):
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import sys
    import time
    import zipfile  # Load before the restricted audit scope.
    import zlib

    import pyarrow as arrow
    import pyarrow.parquet as parquet
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import local_random_access_campaign_v2 as binding

    del zipfile, zlib
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    repo, root = candle.REPO, candle.ROOT
    stage_name = 'direction_dense_minute_targets_actual_20261028_001'
    plan_ref = {'path': str(Path(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(repo / 'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
    policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
    fields = {'schema_version', 'stage', 'design', 'dense_design', 'metadata_review', 'inputs',
        'sources', 'folds', 'fold_sources', 'output_dir', 'cutoff_ns', 'parents', 'horizons_minutes',
        'block_rows', 'wall_seconds', 'test_data_used'}
    require(set(plan) == fields and plan['schema_version'] == 'gx1_direction_dense_target_plan_v1'
        and plan['stage'] == stage_name and plan['cutoff_ns'] == candle.CUTOFF
        and plan['parents'] == candle.PARENTS and plan['horizons_minutes'] == list(HORIZONS)
        and plan['block_rows'] == 1024 and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
    gate = policy[stage_name]
    require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
        and gate['genuine_source_reads_authorized'] is True and gate['dense_target_write_authorized'] is True
        and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
        and gate['relaunch_allowed'] is False and policy['independent_direction_edge_20261027']['current_unit'] == stage_name,
        'ADMISSION')
    require(all(policy.get(k) is False for k in ('training_enabled', 'native_training_enabled',
        'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
        and all(gate.get(k) is False for k in ('model_forwards_authorized', 'optimizer_steps_authorized',
            'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
    expected_git = (args.source_head, 'work/gx1-current', '')
    require(candle.git_state() == expected_git, 'CLEAN_SOURCE')
    design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
    dense_design = binding.read_bound_json(Path(plan['dense_design']['path']), plan['dense_design']['sha256'])
    review = binding.read_bound_json(Path(plan['metadata_review']['path']), plan['metadata_review']['sha256'])
    require(dense_design['status'] == 'DESIGN_ONLY_NO_EXECUTION_AUTHORITY'
        and dense_design['constraints']['test_live_paper_broker_orders_exit_promotion'] is False
        and dense_design['current_state']['native_direction_horizons_minutes'] == [1, 5, 15], 'DENSE_DESIGN')
    expected_inputs = design['input_references_from_existing_verified_metadata']
    require(set(plan['inputs']) == {'native_2024', 'native_2025', 'native_manifest', 'price_targets', 'target_metadata'}, 'INPUT_SET')
    for name, ref in plan['inputs'].items():
        require(ref == {k: expected_inputs[name][k] for k in ('path', 'sha256')}, 'INPUT_BINDING:' + name)
    expected_sources = set(candle.SOURCE_PATHS) | {'gx1/scripts/materialize_entry_direction_dense_v1.py'}
    require(set(plan['sources']) == expected_sources, 'SOURCE_SET')
    for name, ref in plan['sources'].items():
        require(ref['path'] == str(repo / name), 'SOURCE_PATH')
        binding.require_binding(ref, label='dense source', verify_file=True)
    folds = plan['folds']
    require(isinstance(folds, list) and len(folds) == 4 and isinstance(plan['fold_sources'], list)
        and len(plan['fold_sources']) == 4, 'FOLDS')
    prior_end = 0
    for index, fold in enumerate(folds, 1):
        fold_ref = plan['fold_sources'][index - 1]
        fold_report = binding.read_bound_json(Path(fold_ref['path']), fold_ref['sha256'])
        require(fold_report['fold'] == fold and fold_report['test_data_used'] is False, 'FOLD_SOURCE')
        require(set(fold) == {'fold_id', 'prefix_rows', 'oof_start_row', 'oof_end_row', 'train_cutoff_ns', 'oof_cutoff_ns'}
            and fold['fold_id'] == index and fold['prefix_rows'] == fold['oof_start_row']
            and (index == 1 or prior_end == fold['oof_start_row'])
            and prior_end <= fold['oof_start_row'] < fold['oof_end_row'] <= candle.PARENTS
            and fold['train_cutoff_ns'] < fold['oof_cutoff_ns'] <= candle.CUTOFF, 'FOLD_IDENTITY')
        prior_end = fold['oof_end_row']
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    output = Path(plan['output_dir'])
    require(output.is_absolute() and output.parent == root and not output.exists() and not output.is_symlink()
        and root.resolve(strict=True) == root, 'OUTPUT')
    output.mkdir(mode=0o700)
    def publish(name, value):
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        return {'path': str(output / name), 'sha256': hashlib.sha256(raw).hexdigest()}
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref, 'source_head': args.source_head,
        'test_data_used': False})
    target_path = output / 'DENSE_TARGETS.npz'
    meta_paths = [plan_ref['path'], policy_ref['path'], plan['design']['path'],
        plan['dense_design']['path'], plan['metadata_review']['path']]
    sys.addaudithook(candle.make_guard(output, [ref['path'] for ref in plan['inputs'].values()],
        meta_paths, output_payload_paths=(str(target_path),)))
    started = time.monotonic()
    def expired(*unused):
        raise RuntimeError('DIRECTION_DENSE_TARGET_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(plan['wall_seconds'])
    previous_profile = sys.getprofile()
    try:
        sys.setprofile(candle.no_queries_or_models)
        with binding.fresh_file_verification_scope():
            for ref in plan['inputs'].values():
                candle.regular(ref['path'])
                binding.require_binding(ref, label='dense input', verify_file=True)
            manifest = json.loads(Path(plan['inputs']['native_manifest']['path']).read_text())
            metadata = json.loads(Path(plan['inputs']['target_metadata']['path']).read_text())
            candle.require_metadata(manifest, metadata, plan['inputs'], review)
            saved = candle.read_targets(plan['inputs']['price_targets']['path'], np=np)
            years = [candle.read_year_quotes(plan['inputs']['native_' + str(year)]['path'],
                expected_rows=rows, year=year, parquet=parquet, arrow=arrow, np=np)
                for year, rows, offset in candle.YEAR_SPEC]
            source = {key: np.concatenate([part[key] for part in years]) for key in years[0]}
            source['source_row'] = np.concatenate([np.arange(offset, offset + rows, dtype=np.int64)
                for year, rows, offset in candle.YEAR_SPEC])
            target = build_dense_targets(source, saved, cutoff_ns=plan['cutoff_ns'], block_rows=plan['block_rows'])
            require(len(target['parent_rows']) == plan['parents']
                and np.array_equal(target['parent_rows'], np.arange(plan['parents'], dtype=np.int64)), 'TARGET_ROWS')
            for fold in folds:
                decision = target['decision_ns']
                require(int(np.searchsorted(decision, fold['train_cutoff_ns'], side='left')) == fold['prefix_rows']
                    and int(np.searchsorted(decision, fold['oof_cutoff_ns'], side='left')) == fold['oof_end_row'], 'FOLD_CLOCK')
            input_snapshot = binding.capture_verified_file_snapshot()
        with target_path.open('xb') as handle:
            np.savez_compressed(handle, **target)
            handle.flush(); os.fsync(handle.fileno())
        require(0 < target_path.stat().st_size <= 512 * 1024**2, 'TARGET_OUTPUT_SIZE')
        target_sha = binding.file_sha256(target_path)
        summaries = []
        for fold in folds:
            groups = []
            for role, first, last, cutoff in (
                ('train', 0, fold['prefix_rows'], fold['train_cutoff_ns']),
                ('oof', fold['oof_start_row'], fold['oof_end_row'], fold['oof_cutoff_ns'])):
                batch = causal_target_batch(target, np.arange(first, last, dtype=np.int64), cutoff_ns=cutoff)
                eligible = batch['observed'] & (batch['sign'] != 0)
                groups.append({'role': role, 'parents': last - first, 'cutoff_ns_exclusive': cutoff,
                    'eligible': eligible.sum(axis=0).astype(np.int64).tolist(),
                    'up': (eligible & (batch['sign'] == 1)).sum(axis=0).astype(np.int64).tolist(),
                    'down': (eligible & (batch['sign'] == -1)).sum(axis=0).astype(np.int64).tolist(),
                    'observed_zero': (batch['observed'] & (batch['sign'] == 0)).sum(axis=0).astype(np.int64).tolist(),
                    'mature_missing': (batch['mature'] & ~batch['observed']).sum(axis=0).astype(np.int64).tolist(),
                    'immature': (~batch['mature']).sum(axis=0).astype(np.int64).tolist()})
                del batch, eligible
            summaries.append({'fold_id': fold['fold_id'], 'groups': groups})
        input_snapshot.require_unchanged()
        require(candle.git_state() == expected_git and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
        report = {'schema_version': 'gx1_direction_dense_target_result_v1', 'status': 'COMPLETE_REAL_TARGETS_ONLY',
            'plan': plan_ref, 'source_head': args.source_head, 'target': {'path': str(target_path),
                'sha256': target_sha, 'bytes': target_path.stat().st_size}, 'parents': len(target['parent_rows']),
            'horizons_minutes': list(HORIZONS), 'global_cutoff_ns_exclusive': plan['cutoff_ns'],
            'legacy_anchor_byte_parity': {'1': True, '5': True, '15': True},
            'fold_summaries': summaries, 'source_inputs': plan['inputs'],
            'cutoff_rule': 'Global archive contains only endpoints before 2026; every fold batch must mask terminal_time >= its own cutoff before loss or evaluation.',
            'scope': 'Historical TRAIN/development candle midpoint outcomes; no model, trading, TEST or quote-event latency.',
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
            'elapsed_seconds': time.monotonic() - started}
        result_ref = publish('RESULT.json', report)
        terminal_ref = publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result_ref, 'source_head': args.source_head, 'source_unchanged': True,
            'execution': proof, 'lock': lock, 'test_data_used': False})
        print(json.dumps({'status': report['status'], 'result': result_ref, 'terminal': terminal_ref,
            'parents': report['parents'], 'horizons': 60, 'target_bytes': report['target']['bytes']}))
    except Exception as exc:
        publish('TERMINAL.json', {'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'source_head': args.source_head, 'error': type(exc).__name__ + ': ' + str(exc),
            'test_data_used': False})
        raise
    finally:
        sys.setprofile(previous_profile)
        signal.alarm(0)


if __name__ == '__main__':
    main()
