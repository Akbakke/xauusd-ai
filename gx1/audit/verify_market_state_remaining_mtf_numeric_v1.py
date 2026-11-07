"""One-use TRAIN-only physical comparison of the 63 residual MTF cache fields.

The nine independent closed-bar oracles own the formulas. This driver binds
old native/cache receipts, fits no parameters, uses no model, and publishes
only compact numeric evidence from one capped CPU invocation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit import verify_m5_m15_h1_trendline_oracle_v1 as prior
from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as h4
from gx1.audit import market_state_remaining_mtf_volume_oracle_v1 as volume
from gx1.audit import market_state_remaining_mtf_squeeze_oracle_v1 as squeeze
from gx1.audit import market_state_remaining_mtf_candle_oracle_v1 as candle
from gx1.audit import market_state_remaining_mtf_momentum_simple5_oracle_v1 as momentum
from gx1.audit import market_state_remaining_mtf_divergence10_oracle_v1 as divergence
from gx1.audit import market_state_remaining_mtf_local_returns_oracle_v1 as returns
from gx1.audit import market_state_remaining_mtf_basic7_oracle_v1 as basic
from gx1.audit import market_state_remaining_mtf_micro_price_oracle_v1 as micro
from gx1.audit import market_state_remaining_mtf_generic5_oracle_v1 as generic

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
STAGE = 'market_state_remaining_mtf_numeric_20261107_001'
SCHEMA = 'gx1_market_state_remaining_mtf_numeric_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
TRAIN_START_NS = prior.TRAIN_START_NS
TRAIN_END_NS = prior.TRAIN_END_NS
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
TF_SCOPE = {
    **prior.TF_SCOPE,
    'H4': {'bar_ns': 14_400_000_000_000, 'seq_len': 96,
           'band_atr': 0.9971390001300923, 'prefix_rows': 23270,
           'first_row': 21464, 'last_row': 22507, 'witnesses': 8},
    'D1': {'bar_ns': 86_400_000_000_000, 'seq_len': 252,
           'band_atr': 0.6968664882937518, 'prefix_rows': 3889,
           'first_row': 3588, 'last_row': 3761, 'witnesses': 4},
}
EXPECTED_COLUMNS = (
    *range(23, 50), *range(111, 126), *range(191, 196),
    *range(197, 202), *range(218, 223), 228, 230, 231, 232, 234, 238,
)
assert len(EXPECTED_COLUMNS) == len(set(EXPECTED_COLUMNS)) == 63
QUALIFIED_INDEX = {
    'path': '/home/andre2/GX1_RUNS/NATIVE_M5_REMAINING_ACTIVE_CORE_NUMERIC_REFERENCE_20261021_133/QUALIFIED_MTF_NUMERIC_INDEX.json',
    'sha256': 'f5022afbdcde059440a327d8ea2b9b06cb5bc07a077b482b45b61302232a292f',
}
POSITION_REFS = {
    'result': {'path': '/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CLOSE_POSITION_NUMERIC_20261028_001/OUTPUT/RESULT.json',
               'sha256': '7f381bdf8d876471fd67ae0b2d8289737b3bdc42d7d40140bc2b43ba11f9598a'},
    'terminal': {'path': '/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CLOSE_POSITION_NUMERIC_20261028_001/OUTPUT/TERMINAL.json',
                 'sha256': '25eae10e00b838d800b65f29f685438e2e47443844c517766f305a61f982a7c2'},
}
TRENDLINE_PHYSICAL_REFS = {
    'M5_M15_H1': {
        'plan': {'path': str(REPO / 'configs/research/M5_M15_H1_TRENDLINE_PHYSICAL_ORACLE_20261107_001.json'),
                 'sha256': '2fa56e3827926dacb4b469433227bb8e153671ff66ddd45a31e57b9df6173d28'},
        'result': {'path': str(RUNS / 'M5_M15_H1_TRENDLINE_PHYSICAL_ORACLE_20261107_001/RESULT.json'),
                   'sha256': '7554f31463aeceebf4f8950aeec2bf251cab3ef3276612f97de3b4a6cd1c0dfb'},
        'terminal': {'path': str(RUNS / 'M5_M15_H1_TRENDLINE_PHYSICAL_ORACLE_20261107_001/TERMINAL.json'),
                     'sha256': '440d9238489a6e907a35fe53f0e695e3d3f3f9a47edb4f6c0f5e2d1751a58658'},
    },
    'H4': {
        'plan': {'path': str(REPO / 'configs/research/H4_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001.json'),
                 'sha256': '51be8cd511db606751d62e85bf1b8348601fd350b62e106116140cbacf4ccb2c'},
        'result': {'path': str(RUNS / 'H4_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001/RESULT.json'),
                   'sha256': '6c4bea5179d2d9231f7e2051d2349b9c7ee97880c93b7aec939a4e10788206da'},
        'terminal': {'path': str(RUNS / 'H4_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001/TERMINAL.json'),
                     'sha256': 'b36b219d75897911e6765f8137053d6a893b107dce35a3ea187e15c4a82ca32f'},
    },
    'D1': {
        'plan': {'path': str(REPO / 'configs/research/D1_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001.json'),
                 'sha256': 'f403ea8ad2e771782c6f58d9fbc9b0ca56fe39092bc3c7d57c18661cbd8a5e2b'},
        'result': {'path': str(RUNS / 'D1_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001/RESULT.json'),
                   'sha256': 'aac68ee7dd349f1e56d3d85da280292706c7e6f414556d9c5842a29e24405b8e'},
        'terminal': {'path': str(RUNS / 'D1_TRENDLINE_LIFECYCLE_PHYSICAL_ORACLE_20261107_001/TERMINAL.json'),
                     'sha256': '405412e20b0d854fb0efbd5a513e617c56082bb8f3a43232f98139f8fd519cf5'},
    },
}
INPUT_REFS = {'trendline': prior.INPUT_REFS, 'qualified_index': QUALIFIED_INDEX,
              'trendline_physical': TRENDLINE_PHYSICAL_REFS,
              'close_position': POSITION_REFS}
LIMITS = {'memory_max': '4G', 'swap_max': '512M', 'wall_seconds': 3600,
          'execution_class': 'audit', 'model_forwards': 0, 'optimizer_steps': 0}
EXECUTION = {'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '4G', 'swap': '512M', 'timeout_seconds': 3600,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.verify_market_state_remaining_mtf_numeric_v1',
             'model_forwards': 0, 'optimizer_steps': 0}
SCOPE = {'train_start_ns': TRAIN_START_NS,
         'train_end_exclusive_ns': TRAIN_END_NS,
         'timeframes': list(TIMEFRAMES), 'tf_scope': TF_SCOPE,
         'feature_columns': list(EXPECTED_COLUMNS), 'test_data_used': False}
SOURCE_PATHS = frozenset({
    'gx1/audit/verify_market_state_remaining_mtf_numeric_v1.py',
    'gx1/audit/verify_m5_m15_h1_trendline_oracle_v1.py',
    'gx1/audit/verify_h4_trendline_lifecycle_oracle_v1.py',
    'gx1/audit/verify_d1_trendline_lifecycle_physical_v1.py',
    'gx1/audit/verify_d1_trendline_lifecycle_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_volume_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_squeeze_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_candle_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_momentum_simple5_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_divergence10_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_local_returns_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_basic7_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_micro_price_oracle_v1.py',
    'gx1/audit/market_state_remaining_mtf_generic5_oracle_v1.py',
    'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/features/volatility_squeeze_state_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
})
ABS_TOL = 2e-6
REL_TOL = 2e-6
TERMINAL_RESERVE_SECONDS = 30


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('REMAINING_MTF_NUMERIC_' + reason)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': _sha(path)}


def _write(root: Path, name: str, value: dict) -> dict:
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(raw) <= 1024 * 1024 and json.loads(raw) == value, 'PUBLICATION')
    path = root / name
    _publish_bytes(path, raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def _groups() -> tuple[tuple[tuple[int, ...], tuple[str, ...]], ...]:
    return (
        (volume.CACHE_COLUMNS, volume.FIELD_NAMES),
        (squeeze.CACHE_COLUMNS, squeeze.FIELDS),
        (candle.CACHE_COLUMNS, candle.CANDLE_CACHE_NAMES),
        (momentum.CACHE_COLUMNS, momentum.FIELDS),
        (divergence.CACHE_COLUMNS, divergence.DIVERGENCE10_CACHE_NAMES),
        (returns.CACHE_COLUMNS, returns.FIELDS),
        (basic.CACHE_COLUMNS, basic.FIELD_NAMES),
        (micro.CACHE_COLUMNS, micro.FIELDS),
        (generic.CACHE_COLUMNS, generic.GENERIC5_CACHE_NAMES),
    )


def _field_contract(manifest: dict, qualified: dict) -> tuple[str, ...]:
    names = manifest.get('feature_names')
    _need(manifest.get('feature_count') == 253 and isinstance(names, list)
          and len(names) == 253 and len(set(names)) == 253,
          'MANIFEST_FEATURES')
    groups = _groups()
    columns = [col for cols, _ in groups for col in cols]
    _need(len(groups) == 9 and len(columns) == 63
          and len(set(columns)) == 63 and tuple(sorted(columns)) == EXPECTED_COLUMNS
          and all(tuple(names[col] for col in cols) == tuple(group_names)
                  and len(cols) == len(group_names)
                  for cols, group_names in groups), 'ORACLE_ORDINAL_NAMES')
    rows = qualified.get('rows')
    _need(qualified.get('cache_manifest') is not None
          and qualified.get('qualified_per_tf') == 158
          and qualified.get('test_data_used') is False
          and isinstance(rows, list) and len(rows) == 790,
          'QUALIFIED_INDEX_SCOPE')
    sets = []
    for tf in TIMEFRAMES:
        records = [row for row in rows if row.get('tf') == tf]
        indexes = [row.get('cache_column') for row in records]
        _need(len(records) == 158 and len(set(indexes)) == 158
              and all(type(col) is int and 0 <= col < 253
                      and row.get('name') == names[col]
                      and row.get('numeric_reference_qualified') is True
                      for col, row in zip(indexes, records)),
              'QUALIFIED_INDEX_' + tf)
        sets.append(set(indexes))
    _need(all(s == sets[0] for s in sets)
          and not (sets[0] & set(range(159, 190)))
          and 236 not in sets[0]
          and set(EXPECTED_COLUMNS) == set(range(253)) - sets[0]
              - set(range(159, 190)) - {236}
          and names[236] == 'close_position_range20',
          'EXACT_63_COMPLEMENT')
    return tuple(names[col] for col in EXPECTED_COLUMNS)


def _small_receipts() -> tuple[dict, dict, dict, dict]:
    inputs, manifest, witnesses = prior._small_receipts()
    _need(inputs.get('qualified_index') == QUALIFIED_INDEX,
          'PINNED_133_INDEX_REF')
    qualified = h4.bound_json(QUALIFIED_INDEX)
    _need(qualified.get('cache_manifest') == inputs['cache_manifest'],
          'INDEX_CACHE_BINDING')
    _field_contract(manifest, qualified)
    from gx1.features.volatility_squeeze_state_v1 import require_volatility_squeeze_artifact_binding
    frozen = require_volatility_squeeze_artifact_binding(
        manifest['volatility_squeeze_artifact_set'])
    _need(frozen.binding() == manifest['volatility_squeeze_artifact_set']
          and all(frozen.require_params(tf)['fit']['low_state_index'] == 0
                  for tf in TIMEFRAMES), 'FROZEN_HMM_PRECLAIM')
    for lane, refs in TRENDLINE_PHYSICAL_REFS.items():
        plan = h4.bound_json(refs['plan'])
        result = h4.bound_json(refs['result'])
        terminal = h4.bound_json(refs['terminal'])
        _need(plan.get('test_data_used') is False
              and result.get('plan') == refs['plan']
              and result.get('status') == terminal.get('status') == 'COMPLETE'
              and result.get('test_data_used') is False
              and terminal.get('test_data_used') is False
              and terminal.get('exit_code') == 0
              and terminal.get('result') == refs['result'],
              'TRENDLINE_PHYSICAL_CHAIN_' + lane)
        if lane == 'M5_M15_H1':
            _need(set(result.get('timeframes', {})) == {'M5', 'M15', 'H1'}
                  and all(result['timeframes'][tf]['comparison']['rows']
                          == TF_SCOPE[tf]['last_row'] - TF_SCOPE[tf]['first_row'] + 1
                          and result['timeframes'][tf]['comparison']['fields'] == 31
                          and result['timeframes'][tf]['comparison']['mismatched_cells'] == 0
                          for tf in ('M5', 'M15', 'H1')),
                  'TRENDLINE_PHYSICAL_31_THREE_TF')
        else:
            _need(result['comparison']['rows']
                  == TF_SCOPE[lane]['last_row'] - TF_SCOPE[lane]['first_row'] + 1
                  and result['comparison']['fields'] == 31
                  and result['comparison']['mismatched_cells'] == 0,
                  'TRENDLINE_PHYSICAL_31_' + lane)
    position = h4.bound_json(POSITION_REFS['result'])
    terminal = h4.bound_json(POSITION_REFS['terminal'])
    _need(position.get('status') == terminal.get('status') == 'COMPLETE'
          and position.get('field') == 'close_position_range20'
          and position.get('all_five_exact') is True
          and position.get('test_data_used') is False
          and terminal.get('exit_code') == 0
          and terminal.get('test_data_used') is False
          and set(position.get('timeframes', {})) == set(TIMEFRAMES)
          and all(position['timeframes'][tf]['comparison']['mismatched_cells'] == 0
                  for tf in TIMEFRAMES), 'COLUMN_236_RECEIPT')
    for tf in ('H4', 'D1'):
        refs = inputs['per_tf'][tf]
        _need(set(refs) == {'feats', 'ts', 'replay_plan', 'replay_result', 'replay_terminal'}
              and refs['feats'] == {
                  'path': str(Path(inputs['cache_manifest']['path']).parent /
                              manifest['tfs'][tf]['feats_npy']),
                  'sha256': manifest['tfs'][tf]['feats_npy_sha256']}
              and refs['ts'] == {
                  'path': str(Path(inputs['cache_manifest']['path']).parent /
                              manifest['tfs'][tf]['ts_npy']),
                  'sha256': manifest['tfs'][tf]['ts_npy_sha256']}
              and manifest['tfs'][tf]['feature_count'] == 253,
              'CACHE_REFS_' + tf)
        replay_plan = h4.bound_json(refs['replay_plan'])
        replay = h4.bound_json(refs['replay_result'])
        replay_terminal = h4.bound_json(refs['replay_terminal'])
        c = TF_SCOPE[tf]
        key = 'd1' if tf == 'D1' else 'tf'
        _need(replay_plan['inputs'][key + '_feats'] == refs['feats']
              and replay_plan['inputs'][key + '_ts'] == refs['ts']
              and replay_plan['inputs']['cache_manifest'] == inputs['cache_manifest']
              and replay.get('status') == replay_terminal.get('status') == 'COMPLETE'
              and replay_terminal.get('exit_code') == 0
              and replay.get('test_data_used') is False
              and replay.get('input_price_precision') == 'native_float64_full_history'
              and replay.get('source_clock_rows') == c['prefix_rows']
              and replay.get('replay_parity') is True
              and replay['all_causal_history_vs_cache']['mismatched_cells'] == 0
              and replay['model_train_period_vs_cache']['rows'] == c['last_row'] - c['first_row'] + 1
              and replay.get('window_configuration') == {
                  'timeframe': tf, 'seq_len': c['seq_len'], 'band_atr': c['band_atr']},
              'REPLAY_RECEIPT_' + tf)
        for name in ('feats', 'ts'):
            path = Path(refs[name]['path'])
            _need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                  'PAYLOAD_HEADER_' + tf + '_' + name)
    return inputs, manifest, witnesses, qualified


def _clean_head() -> str:
    return prior._clean_head()


def _inspect_plan(plan_path: Path, plan_sha256: str, out_dir: Path) -> tuple:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT
          and plan_path.is_file() and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    _need(isinstance(plan, dict) and set(plan) == {
              'schema_version', 'stage', 'status', 'source_base_head',
              'output_root', 'max_invocations', 'input_refs', 'scope',
              'limits', 'execution', 'model_forwards_authorized',
              'optimizer_steps_authorized', 'training_authorized',
              'test_data_used', 'source_owners'}
          and plan.get('schema_version') == SCHEMA and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and type(plan.get('source_base_head')) is str
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
          and plan.get('max_invocations') == 1
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('scope') == SCOPE and plan.get('limits') == LIMITS
          and plan.get('execution') == EXECUTION
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False,
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(isinstance(digest, str) and len(digest) == 64
                  and _sha(REPO / path) == digest
                  for path, digest in owners.items()), 'SOURCE_OWNERS')
    inputs, manifest, witnesses, qualified = _small_receipts()
    return plan, {'path': str(plan_path), 'sha256': plan_sha256}, head, inputs, manifest, witnesses, qualified


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, ref, head, inputs, _manifest, _witnesses, qualified = _inspect_plan(
        plan_path, plan_sha256, out_dir)
    return {'schema_version': 'gx1_remaining_mtf_numeric_preflight_v1',
            'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': ref,
            'source_head': head, 'source_owners_checked': len(plan['source_owners']),
            'prior_qualified_per_tf': qualified['qualified_per_tf'],
            'residual_fields_per_tf': len(EXPECTED_COLUMNS),
            'native_parts_stat_checked': len(inputs['native_parts']),
            'claim_created': False, 'npz_or_raw_payload_read': False,
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def _policy_gate(policy: dict, plan_ref: dict, out_dir: Path) -> None:
    gate = policy.get(STAGE, {})
    _need(policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and gate.get('status') == gate.get('decision') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == 1
          and gate.get('relaunch_allowed') is False
          and gate.get('memory_max') == LIMITS['memory_max']
          and gate.get('swap_max') == LIMITS['swap_max']
          and gate.get('wall_seconds') == LIMITS['wall_seconds']
          and gate.get('test_data_used') is False
          and gate.get('checkpoint_reads_authorized') is False
          and gate.get('broker_access_allowed') is False
          and gate.get('model_forwards_authorized') == 0
          and gate.get('optimizer_steps_authorized') == 0
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('output_root') == str(out_dir), 'POLICY_GATE')


def _cap_gate() -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    _need(cap.get('memory_max') == cap.get('memory_high') == 4 * 1024**3
          and cap.get('swap') == 512 * 1024**2
          and cap.get('pids') == 64
          and cap.get('numerical_threads') == 1
          and lock.get('scope') == cap.get('cgroup_path', '')[len('/sys/fs/cgroup'):]
          and not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib')
                      for name in sys.modules), 'CAP_LOCK_NO_MODEL')
    return {'execution': cap, 'lock': lock}


def _align_rows(cache_ts: np.ndarray, native_ns: np.ndarray, tf: str
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = TF_SCOPE[tf]
    _need(cache_ts.dtype == native_ns.dtype == np.int64
          and cache_ts.ndim == native_ns.ndim == 1
          and len(cache_ts) > c['last_row'] + 1
          and len(native_ns) > 0
          and np.all(np.diff(cache_ts) > 0)
          and np.all(np.diff(native_ns) > 0), 'CLOCK_AXES_' + tf)
    closed = cache_ts[:c['prefix_rows']] + c['bar_ns']
    rows = np.flatnonzero((closed >= TRAIN_START_NS) & (closed < TRAIN_END_NS))
    _need(np.array_equal(rows, np.arange(c['first_row'], c['last_row'] + 1))
          and int(cache_ts[c['last_row'] + 1]) + c['bar_ns'] >= TRAIN_END_NS,
          'TRAIN_ROW_RANGE_' + tf)
    times = np.asarray(cache_ts[rows], dtype=np.int64)
    positions = np.searchsorted(native_ns, times)
    _need(int(positions[-1]) < len(native_ns)
          and np.array_equal(native_ns[positions], times)
          and np.all(np.diff(positions) > 0), 'NATIVE_JOIN_' + tf)
    return rows, positions.astype(np.int64), times


def _require_prerequisites(high, low, close, rsi, atr, cache_rows) -> dict:
    """Bind unrounded RSI/ATR to already-qualified columns 1/0."""
    expected = np.column_stack((atr / close * 1e4, (rsi - 50.0) / 50.0)).astype(np.float32)
    observed = np.asarray(cache_rows[:, [0, 1]], dtype=np.float32)
    _need(expected.shape == observed.shape
          and not np.isinf(expected).any() and not np.isinf(observed).any(),
          'PREREQUISITE_SHAPE')
    mask_equal = np.array_equal(np.isnan(expected), np.isnan(observed))
    finite = np.isfinite(expected) & np.isfinite(observed)
    difference = np.abs(expected.astype(np.float64) - observed.astype(np.float64))
    tolerance = np.maximum(ABS_TOL, REL_TOL * np.abs(expected.astype(np.float64)))
    _need(mask_equal and bool(np.all(difference[finite] <= tolerance[finite])),
          'QUALIFIED_RAW_RSI_ATR_CACHE_BINDING')
    _need(np.all(np.isfinite(high)) and np.all(np.isfinite(low))
          and np.all(np.isfinite(close)) and np.all(low > 0)
          and np.all(low <= close) and np.all(close <= high)
          and np.all(np.isfinite(rsi)) and np.all((rsi >= 0) & (rsi <= 100))
          and np.all(np.isfinite(atr)) and np.all(atr > 0),
          'PREREQUISITE_TRAIN_DOMAIN')
    return {'rows': len(close),
            'rsi14_raw_float64': True, 'atr14_positive_float64': True,
            'qualified_cache_columns': [0, 1],
            'max_cache_absolute_error': float(np.max(difference[finite])) if finite.any() else None}


def _oracle_matrix(bars, positions: np.ndarray, tf: str,
                   squeeze_fit: dict, feature_names: tuple[str, ...],
                   rsi: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """Compute each independent formula on the complete native history prefix."""
    n = len(bars)
    _need(n > 0 and positions.dtype == np.int64 and positions.ndim == 1
          and len(positions) > 0 and positions[-1] == n - 1
          and np.all(np.diff(positions) > 0), 'ORACLE_PREFIX_POSITIONS')
    opened = bars['open'].to_numpy(dtype=np.float64)
    high = bars['high'].to_numpy(dtype=np.float64)
    low = bars['low'].to_numpy(dtype=np.float64)
    close = bars['close'].to_numpy(dtype=np.float64)
    counts = bars['volume'].to_numpy(dtype=np.float64)
    clock = bars.index.asi8 + TF_SCOPE[tf]['bar_ns']
    _need(bool(np.all(clock < TRAIN_END_NS) and np.all(np.diff(clock) > 0)),
          'ORACLE_CLOSED_CLOCK')
    result = np.full((len(positions), len(EXPECTED_COLUMNS)), np.nan, dtype=np.float32)
    column_to_slot = {col: i for i, col in enumerate(EXPECTED_COLUMNS)}
    assigned: set[int] = set()

    def put(cols, names, values):
        matrix = np.asarray(values, dtype=np.float32)
        _need(matrix.shape == (len(positions), len(cols)) and not np.isinf(matrix).any()
              and not (set(cols) & assigned)
              and tuple(feature_names[column_to_slot[col]] for col in cols) == tuple(names),
              'ORACLE_GROUP_' + tf)
        for j, col in enumerate(cols):
            result[:, column_to_slot[col]] = matrix[:, j]
        assigned.update(cols)

    volume_values = volume.independent_closed_tf_price_update_oracle(
        counts, available_ns=clock, train_end_exclusive_ns=TRAIN_END_NS)
    put(volume.CACHE_COLUMNS, volume.FIELD_NAMES, volume_values['values'][positions])
    del volume_values

    fit = squeeze_fit
    squeeze_values = squeeze.closed_squeeze3_reference(
        close, means=fit['bandwidth_mean'], variances=fit['bandwidth_variance'],
        transition_probability=fit['transition_probability'],
        initial_probability=fit['initial_probability'])
    put(squeeze.CACHE_COLUMNS, squeeze.FIELDS,
        np.column_stack([squeeze_values[name][positions] for name in squeeze.FIELDS]))
    del squeeze_values

    put(candle.CACHE_COLUMNS, candle.CANDLE_CACHE_NAMES,
        candle.candle21_from_closed_ohlc(opened, high, low, close, positions=positions))

    momentum_values = momentum.closed_momentum_simple5_reference(
        high, low, close, rsi, atr)
    put(momentum.CACHE_COLUMNS, momentum.FIELDS,
        np.column_stack([momentum_values[name][positions] for name in momentum.FIELDS]))
    del momentum_values

    put(divergence.CACHE_COLUMNS, divergence.DIVERGENCE10_CACHE_NAMES,
        divergence.divergence10_from_closed_hlc_rsi_atr(
            high, low, close, rsi, atr, positions=positions))

    returns_values = returns.closed_local_returns4_reference(close)
    put(returns.CACHE_COLUMNS, returns.FIELDS,
        np.column_stack([returns_values[name][positions] for name in returns.FIELDS]))
    del returns_values

    basic_values = basic.independent_closed_tf_basic7_oracle(
        high, low, close, available_ns=clock,
        train_end_exclusive_ns=TRAIN_END_NS)
    put(basic.CACHE_COLUMNS, basic.FIELD_NAMES, basic_values['values'][positions])
    del basic_values

    micro_values = micro.closed_micro_price5_reference(high, low, close)
    put(micro.CACHE_COLUMNS, micro.FIELDS,
        np.column_stack([micro_values[name][positions] for name in micro.FIELDS]))
    del micro_values

    put(generic.CACHE_COLUMNS, generic.GENERIC5_CACHE_NAMES,
        generic.generic5_from_closed_hlc(high, low, close, positions=positions))
    _need(assigned == set(EXPECTED_COLUMNS), 'ALL_63_ASSIGNED_' + tf)
    return result


def _discrete(name: str) -> bool:
    return (name.endswith(('_flag', '_event', '_age_bars', '_duration_bars',
                           '_last_event_side'))
            or name in {'close_range_observed', 'volatility.bars_in_squeeze',
                        'volatility.squeeze_release_age_bars'})


def _compare(expected: np.ndarray, observed: np.ndarray, rows: np.ndarray,
             times: np.ndarray, names: tuple[str, ...]) -> dict:
    a = np.asarray(expected, dtype=np.float32)
    b = np.asarray(observed, dtype=np.float32)
    _need(a.shape == b.shape == (len(rows), 63)
          and times.shape == rows.shape and len(names) == 63
          and np.all(np.diff(rows) == 1) and np.all(np.diff(times) > 0)
          and not np.isinf(a).any() and not np.isinf(b).any(), 'COMPARE_SHAPE')
    details = []
    examples = []
    total = 0
    for j, name in enumerate(names):
        x = a[:, j]
        y = b[:, j]
        finite = np.isfinite(x) & np.isfinite(y)
        same_missing = np.isnan(x) & np.isnan(y)
        error = np.abs(x[finite].astype(np.float64) - y[finite].astype(np.float64))
        tolerance = (np.zeros(len(error)) if _discrete(name) else
                     np.maximum(ABS_TOL, REL_TOL * np.abs(x[finite].astype(np.float64))))
        matched = same_missing.copy()
        matched[finite] = error <= tolerance
        bad = np.flatnonzero(~matched)
        total += len(bad)
        for i in bad[:max(0, 64 - len(examples))]:
            examples.append({'cache_row': int(rows[i]), 'timestamp_ns': int(times[i]),
                             'cache_column': EXPECTED_COLUMNS[j], 'field': name,
                             'oracle': float(x[i]) if np.isfinite(x[i]) else None,
                             'cache': float(y[i]) if np.isfinite(y[i]) else None})
        expected_finite = np.isfinite(x)
        cache_finite = np.isfinite(y)
        details.append({'cache_column': EXPECTED_COLUMNS[j], 'field': name,
                        'discrete_exact': _discrete(name),
                        'oracle_finite_rows': int(expected_finite.sum()),
                        'cache_finite_rows': int(cache_finite.sum()),
                        'oracle_nonzero_rows': int(np.count_nonzero(expected_finite & (x != 0))),
                        'cache_nonzero_rows': int(np.count_nonzero(cache_finite & (y != 0))),
                        'mismatched_rows': int(len(bad)),
                        'max_finite_absolute_error': float(np.max(error)) if len(error) else None})
    return {'rows': len(rows), 'fields': 63, 'compared_cells': len(rows) * 63,
            'mismatched_cells': int(total), 'all_train_window_cells_match': total == 0,
            'unexercised_fields': [d['field'] for d in details if d['oracle_nonzero_rows'] == 0],
            'float_abs_tolerance': ABS_TOL, 'float_rel_tolerance': REL_TOL,
            'per_field': details, 'mismatch_examples': examples,
            'mismatch_examples_truncated': total > len(examples)}


def _measure(inputs: dict, manifest: dict, qualified: dict) -> dict:
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix
    from gx1.features.htf_features import (
        _resample_ohlcv, build_multi_tf_v4_closed_timestamp_indices,
    )
    from gx1.features.technical_indicators_v1 import wilder_rsi, wilder_atr
    from gx1.features.volatility_squeeze_state_v1 import require_volatility_squeeze_artifact_binding

    stats = {}
    for ref in inputs['native_parts']:
        path, stat = h4.checked(ref)
        stats[str(path)] = stat
    for tf in TIMEFRAMES:
        for key in ('feats', 'ts'):
            path, stat = h4.checked(inputs['per_tf'][tf][key])
            stats[str(path)] = stat
    frozen = require_volatility_squeeze_artifact_binding(
        manifest['volatility_squeeze_artifact_set'])
    _need(frozen.binding() == manifest['volatility_squeeze_artifact_set'],
          'FROZEN_SQUEEZE_BINDING')
    source = load_native_parts_prefix(inputs['native_parts'], TRAIN_END_NS)
    _need(len(source) > 0 and source.index[-1].value < TRAIN_END_NS,
          'NATIVE_TRAIN_ONLY')
    indices = build_multi_tf_v4_closed_timestamp_indices(source.index)
    names = _field_contract(manifest, qualified)
    per_tf = {}
    for tf in TIMEFRAMES:
        print(json.dumps({'stage': STAGE, 'progress': 'start_tf', 'timeframe': tf}), flush=True)
        c = TF_SCOPE[tf]
        refs = inputs['per_tf'][tf]
        index = indices[tf]
        bars = _resample_ohlcv(source, tf).dropna(
            subset=['open', 'high', 'low', 'close']).loc[index]
        _need(bars.index.equals(index), 'RESAMPLE_AXIS_' + tf)
        cache_ts = np.load(refs['ts']['path'], mmap_mode='r', allow_pickle=False)
        cache = np.load(refs['feats']['path'], mmap_mode='r', allow_pickle=False)
        _need(cache_ts.dtype == np.int64 and cache_ts.ndim == 1
              and cache.dtype == np.float32 and cache.ndim == 2
              and cache.shape == (len(cache_ts), manifest['feature_count']),
              'CACHE_SHAPE_' + tf)
        rows, positions, times = _align_rows(cache_ts, index.asi8, tf)
        end = int(positions[-1]) + 1
        bars = bars.iloc[:end]
        rsi = wilder_rsi(bars['close'], 14).to_numpy(dtype=np.float64)
        atr = wilder_atr(bars['high'], bars['low'], bars['close'], 14).to_numpy(dtype=np.float64)
        positive_atr = np.where(atr > 0, atr, np.nan)
        prerequisites = _require_prerequisites(
            bars['high'].to_numpy(dtype=np.float64)[positions],
            bars['low'].to_numpy(dtype=np.float64)[positions],
            bars['close'].to_numpy(dtype=np.float64)[positions],
            rsi[positions], positive_atr[positions],
            np.asarray(cache[np.ix_(rows, [0, 1])]))
        expected = _oracle_matrix(bars, positions, tf, frozen.require_params(tf)['fit'],
                                  names, rsi, positive_atr)
        observed = np.asarray(cache[np.ix_(rows, EXPECTED_COLUMNS)], dtype=np.float32)
        comparison = _compare(expected, observed, rows, times, names)
        per_tf[tf] = {'train_cache_row_first': c['first_row'],
                      'train_cache_row_last_inclusive': c['last_row'],
                      'native_prefix_rows': end,
                      'timestamp_rule': tf + ' label plus native duration closes before TRAIN cutoff',
                      'raw_rsi_atr_prerequisites': prerequisites,
                      'comparison': comparison}
        print(json.dumps({'stage': STAGE, 'progress': 'complete_tf',
                          'timeframe': tf, 'rows': len(rows),
                          'mismatched_cells': comparison['mismatched_cells']}), flush=True)
        del bars, cache, cache_ts, rsi, atr, positive_atr, expected, observed
        gc.collect()
    for path, stat in stats.items():
        _need(h4.regular(Path(path)) == stat, 'FINAL_PAYLOAD_STAT_' + Path(path).name)
    return {'timeframes': per_tf, 'native_parts_hashed': len(inputs['native_parts']),
            'cache_arrays_hashed': 2 * len(TIMEFRAMES),
            'source_m5_rows': len(source),
            'comparison_scope': 'May-Dec 2024 TRAIN cache rows after full 2009-prefix independent closed-bar calculations; no TEST, model, learning or economic claim.',
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def _remaining_alarm_seconds(started: float) -> int:
    # The outer 3600s timeout starts before Python reaches this function. Keep
    # thirty seconds for FAILED/TERMINAL publication and a small import margin.
    seconds = int(LIMITS['wall_seconds'] - (time.monotonic() - started)
                  - TERMINAL_RESERVE_SECONDS)
    _need(seconds > 0, 'WALL_BUDGET_PRECLAIM')
    return seconds


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    started = time.monotonic()
    plan, plan_ref, head, inputs, manifest, _witnesses, qualified = _inspect_plan(
        plan_path, plan_sha256, out_dir)
    cap = _cap_gate()
    policy_ref = _ref(POLICY)
    policy = json.loads(POLICY.read_text())
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
    _policy_gate(policy, plan_ref, out_dir)
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_BEFORE_CLAIM')
    alarm_seconds = _remaining_alarm_seconds(started)
    out_dir.mkdir(parents=True, exist_ok=False)
    claim = None
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError('REMAINING_MTF_NUMERIC_WALL_LIMIT')))
        signal.alarm(alarm_seconds)
        from gx1.contracts.immutable_event_authority_v1 import _fsync_directory
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_remaining_mtf_numeric_claim_v1',
            'stage': STAGE, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'invocation': 1, 'test_data_used': False})
        data = _measure(inputs, manifest, qualified)
        _need(_sha(plan_path) == plan_sha256
              and _sha(POLICY) == policy_ref['sha256']
              and _clean_head() == head
              and all(_sha(REPO / path) == digest
                      for path, digest in plan['source_owners'].items())
              and _sha(out_dir / 'CLAIM.json') == claim['sha256']
              and {entry.name for entry in out_dir.iterdir()} == {'CLAIM.json'},
              'FINAL_IDENTITY')
        comparisons = [data['timeframes'][tf]['comparison'] for tf in TIMEFRAMES]
        mismatches = sum(c['mismatched_cells'] for c in comparisons)
        decision = ('FAIL_63_FIELD_TRAIN_NUMERIC_PARITY' if mismatches else
                    'PASS_63_FIELD_TRAIN_NUMERIC_PARITY_WITH_SUPPORT_GAPS'
                    if any(c['unexercised_fields'] for c in comparisons) else
                    'PASS_63_FIELD_TRAIN_NUMERIC_PARITY_ALL_EXERCISED')
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_remaining_mtf_numeric_result_v1',
            'stage': STAGE, 'status': 'COMPLETE', 'decision': decision,
            'plan': plan_ref, 'claim': claim, 'policy': policy_ref,
            'source_head': head, 'input_refs': INPUT_REFS,
            'source_owners': plan['source_owners'],
            'feature_columns': list(EXPECTED_COLUMNS),
            'feature_names': list(_field_contract(manifest, qualified)),
            'total_compared_cells': sum(c['compared_cells'] for c in comparisons),
            'total_mismatched_cells': mismatches,
            'elapsed_seconds': time.monotonic() - started,
            'cap': cap, **data})
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_remaining_mtf_numeric_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'decision': decision,
            'plan': plan_ref, 'claim': claim, 'result': result,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json']})
        return {'result': result, 'terminal': terminal}
    except Exception as exc:
        failure = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_remaining_mtf_numeric_failure_v1',
            'stage': STAGE, 'plan': plan_ref, 'claim': claim,
            'source_head': head, 'error_type': type(exc).__name__,
            'error': str(exc)[:2048], 'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_remaining_mtf_numeric_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failure,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True,
            'output_inventory': sorted(entry.name for entry in out_dir.iterdir()) + ['TERMINAL.json']})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    value = (preflight_read_only(args.plan, args.plan_sha256, args.out_dir)
             if args.preflight else run(args.plan, args.plan_sha256, args.out_dir))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
