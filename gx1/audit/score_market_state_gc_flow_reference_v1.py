"""One-use, read-only M5 GC flow reference on five later TRAIN folds.

The 049 per-field TRAIN hashes prove the consumed 051 current-context bytes.
The corrected five-endpoint binder owns outcomes. No model, fit or TEST is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import stat
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_five_endpoint_prior_v1 as base
from gx1.audit import market_state_m5_gc_flow_reference_core_v1 as core
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory
from gx1.features import entry_specialist_feature_groups_v1 as owners

REPO, RUNS, POLICY = base.REPO, base.RUNS, base.POLICY
STAGE = 'market_state_gc_flow_reference_20261107_001'
SCHEMA = 'gx1_market_state_gc_flow_reference_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
SOURCE_PATHS = frozenset({
    'gx1/audit/score_market_state_gc_flow_reference_v1.py',
    'gx1/audit/market_state_m5_gc_flow_reference_core_v1.py',
    'gx1/audit/measure_market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/gc_native_context_v1.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/features/micro_structure_v1.py',
    'gx1/features/volume_features.py',
    'scripts/gx1_capped_run.sh',
})
LIMITS = {'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
          'swap': 512 * 1024**2, 'max_wall_seconds': 1800}
EXECUTION = {'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '4G', 'swap': '512M', 'timeout_seconds': 1800,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.score_market_state_gc_flow_reference_v1',
             'model_forwards': 0, 'optimizer_steps': 0}
ROOT261 = RUNS / 'NATIVE_M5_DIRECTION_NATIVE_PREFIX_NORMALIZATION_20261024_261'
ROOT264 = RUNS / 'NATIVE_M5_DIRECTION_NATIVE_PREFIX_INPUT_BINDING_REPAIR_20261024_264'
ROOT267 = RUNS / 'NATIVE_M5_DIRECTION_GC_BOOLEAN_PHYSICAL_PREFIX_NORMALIZATION_20261024_267'
DATA_ROOT = Path('/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003')
CURRENT_MMAP = ROOT261 / 'SUPPORT_SCRATCH/current051/ctx_cont.mmap'
SOURCE_MMAP = ROOT261 / 'SUPPORT_SCRATCH/source050/ctx_cont.mmap'
REFS = {
    'provider_plan': {'path': str(ROOT264 / 'PLAN_SUPPORT.json'),
                      'sha256': '96ff82533042ab069292505b3a44315c13223a9b9d9e9ee669cd757e3f84c086'},
    'decode_binding': {'path': str(ROOT264 / 'DECODE_REUSE_BINDING.json'),
                       'sha256': '48a86db06160126a5a0fab0b6d74f4bd8bd4e24493e1e4fabad97f431ba0a2be'},
    'source_plan': {'path': str(ROOT261 / 'PLAN_SUPPORT.json'),
                    'sha256': '68f968ba182fe8ebd4144242a4498d60029e41d6f457092c126937f32b0f9886'},
    'source_worker': {'path': str(ROOT261 / 'qualify_physical_prefix_support_guarded.py'),
                      'sha256': '521d3aae88926b97813a67fcb923a3a63b6b73e0a68e57a5dc8e36a83c5153f2'},
    'source_terminal': {'path': str(ROOT261 / 'TERMINAL_SUPPORT.json'),
                        'sha256': 'c7423d7b7402c9a4c986f3b058e62dfa99b53ee5734aef07139fd746afe80ecb'},
    'source_outer_terminal': {'path': str(ROOT261 / 'OUTER_TERMINAL_SUPPORT.json'),
                              'sha256': '13a5c7a486d43ac63fd3670490c8bb507613b0ec01027db2e3ff9458ea53c935'},
    'source_reconstruction': {'path': str(RUNS / 'NATIVE_GC_CURRENT_PHYSICAL_METADATA_VIEW_20261019_051/SOURCE_RECONSTRUCTION_train.json'),
                              'sha256': 'f9ddf882de5b2481a274849d5775e23197b2012b6599a52c1078286a85fc2c55'},
    'feature_manifest': {'path': str(DATA_ROOT / 'SURFACES_050/M5_FEATURE_BASE_050.parquet.manifest.json'),
                         'sha256': '2faacdca663e3192e0fd17ef877ef2ca6527b6fc0b9e156e714251a13fc19283'},
    'split_manifest': {'path': str(DATA_ROOT / 'TRAIN_VAL_051/XAU_GC_NATIVE__ENTRY_FITTED_Q_train.manifest.json'),
                       'sha256': 'd6bf1d2ddacf0260e6c23e0496f57086ddfcd3ba971e6fa34b4b433f58cf2ed0'},
    'gc_registry': {'path': str(REPO / 'configs/research/NATIVE_GC_CONTEXT_FIELDS_20261018_049.json'),
                    'sha256': '3e444777099a848cd8795688b57a19122d1ac5454f42902a3495e64aad73a757'},
    'complete_plan': {'path': str(ROOT267 / 'PLAN.json'),
                      'sha256': 'e20ce91dd5569b7f1ab15ce2c8463497d2425b5c307855348150166191d23ab1'},
    'complete_result': {'path': str(ROOT267 / 'RESULT.json'),
                        'sha256': 'e4359717d818e6c7c55382d36a941765cc7c177b19b3eb89a50e11ec11684fe9'},
    'complete_terminal': {'path': str(ROOT267 / 'TERMINAL.json'),
                          'sha256': 'ae53d3274fb84487c247969177dc13506352db509d9ea17b414f290a5df7020a'},
}
INPUT_REFS = {**REFS, 'target': {name: {'path': path, 'sha256': digest}
                               for name, (path, digest) in target.REFS.items()}}
# Every index is checked against both the native owner and the 051 split manifest.
FIELD_COLUMNS = {
    'gc_m5_observed_volume': (552, 553),
    'gc_m5_observed_buy_known': (554, 555),
    'gc_m5_observed_sell_known': (556, 557),
    'gc_m5_observed_unknown': (558, 559),
    'gc_m5_observed_known_delta_contracts': (564, 565),
    'gc_m5_observed_book_ofi_contracts': (798, 799),
}
EXPECTED_FIT_ROWS = (29574, 47338, 64615, 82152, 100224)
EXPECTED_SCORE_ROWS = (17753, 17288, 17537, 18072, 17750)
EXPECTED_CERTIFIED_SIDE = (6096, 5728, 5734, 6129, 5889)
SCOPE = {
    'train_start_ns': target.TRAIN_START_NS,
    'train_end_exclusive_ns': target.FOLD_BOUNDARIES_NS[-1],
    'fold_ids': [1, 2, 3, 4, 5],
    'target': 'M5_k1_corrected_certified_side',
    'fold_axes': 'target.mature_oof_axes_exact_mature_fit_prefix_and_later_TRAIN_score',
    'field_columns': {name: list(pair) for name, pair in FIELD_COLUMNS.items()},
    'mature_fit_rows': list(EXPECTED_FIT_ROWS),
    'score_rows': list(EXPECTED_SCORE_ROWS),
    'certified_side_rows': list(EXPECTED_CERTIFIED_SIDE),
    'core_contract_sha256': core.gc_flow_reference_contract()['contract_sha256'],
    'full_mmap_content_sha_claimed': False,
}
_sha, _ref, _write, _clean_head = base._sha, base._ref, base._write, base._clean_head


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('GC_FLOW_REFERENCE_' + reason)


def _file_stat(path: Path) -> dict:
    _need(path.is_file() and not path.is_symlink(), 'REGULAR:' + path.name)
    s = path.stat()
    _need(stat.S_ISREG(s.st_mode), 'REGULAR_MODE:' + path.name)
    return {key: getattr(s, key) for key in ('st_dev', 'st_ino', 'st_mode',
            'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def _read_json(ref: dict, limit: int) -> dict:
    path = Path(ref['path'])
    before = _file_stat(path)
    _need(0 < before['st_size'] <= limit, 'REF_SIZE:' + path.name)
    raw = path.read_bytes()
    _need(_file_stat(path) == before and len(raw) == before['st_size']
          and hashlib.sha256(raw).hexdigest() == ref['sha256'],
          'REF_HASH:' + path.name)
    return json.loads(raw)


def _masked_hash(values: np.ndarray, observed: np.ndarray) -> str:
    """The immutable 049 digest recipe, including masked stored float32 bytes."""
    _need(values.dtype == np.float32 and observed.dtype == np.bool_
          and values.ndim == observed.ndim == 1 and values.shape == observed.shape,
          'MASKED_HASH_INPUT')
    return hashlib.sha256(observed.astype(np.uint8).tobytes()
                          + values.tobytes()).hexdigest()


def _source_metadata() -> dict:
    provider = _read_json(REFS['provider_plan'], 4 * 1024**2)
    binding = _read_json(REFS['decode_binding'], 4 * 1024**2)
    source_plan = _read_json(REFS['source_plan'], 4 * 1024**2)
    source_terminal = _read_json(REFS['source_terminal'], 4 * 1024**2)
    source_outer = _read_json(REFS['source_outer_terminal'], 4 * 1024**2)
    reconstruction = _read_json(REFS['source_reconstruction'], 4 * 1024**2)
    feature_manifest = _read_json(REFS['feature_manifest'], 4 * 1024**2)
    split = _read_json(REFS['split_manifest'], 64 * 1024**2)
    registry = _read_json(REFS['gc_registry'], 4 * 1024**2)
    completed_plan = _read_json(REFS['complete_plan'], 4 * 1024**2)
    completed = _read_json(REFS['complete_result'], 4 * 1024**2)
    completed_terminal = _read_json(REFS['complete_terminal'], 4 * 1024**2)
    _need(_sha(Path(REFS['source_worker']['path'])) == REFS['source_worker']['sha256']
          and provider.get('decode_reuse_binding') == REFS['decode_binding']
          and provider.get('m5_feature_manifest') == REFS['feature_manifest']
          and provider.get('source_reconstruction') == REFS['source_reconstruction']
          and provider.get('axes', {}).get('source') == INPUT_REFS['target']['model_axes']
          and provider['axes'].get('rows') == 118249
          and provider['axes'].get('decision_rule') == 'bar_start_ns_plus_300000000000'
          and binding.get('source_plan') == REFS['source_plan']
          and binding.get('source_worker') == REFS['source_worker']
          and binding.get('terminal') == REFS['source_terminal']
          and binding.get('outer_terminal') == REFS['source_outer_terminal']
          and binding.get('schema_version') == 'gx1_native_direction_completed_decode_reuse_v1'
          and binding.get('data_fit_or_support_qualified') is False
          and binding.get('completed_source_loader_calls_returns') == [1, 1]
          and binding.get('completed_current_loader_calls_returns') == [1, 1]
          and source_terminal.get('source_unchanged') is True
          and source_terminal.get('test_data_used') is False
          and source_outer.get('source_and_external_bookends_unchanged') is True
          and source_plan.get('worker') == REFS['source_worker'],
          'DECODE_LINEAGE')
    _need(completed.get('status') == 'COMPLETE_FIVE_RAW_QUALIFICATIONS_FOUR_BOOLEAN_PREFIX_NORMALIZATIONS_NO_MODEL_ADMISSION'
          and completed.get('input_provider_plan') == REFS['provider_plan']
          and completed.get('five_complete_raw_parities_before_first_fit') is True
          and completed.get('test_data_used') is False
          and completed.get('plan') == REFS['complete_plan']
          and completed_terminal.get('status') == 'COMPLETE'
          and completed_terminal.get('exit_code') == 0
          and completed_terminal.get('source_unchanged') is True
          and completed_terminal.get('test_data_used') is False
          and completed_terminal.get('result') == REFS['complete_result']
          and completed_terminal.get('plan') == REFS['complete_plan']
          and completed_plan.get('input_provider_plan') == REFS['provider_plan'],
          'COMPLETE_LINEAGE')
    for path, shape in ((CURRENT_MMAP, (118249, 2542)),
                        (SOURCE_MMAP, (1140763, 2542))):
        observed = _file_stat(path)
        _need(observed == binding['scratch_files'][str(path)]
              and observed == completed['reused_scratch'][str(path)]
              and not observed['st_mode'] & 0o222
              and observed['st_size'] == shape[0] * shape[1] * 4,
              'MMAP_STAT:' + path.name)
    checks = reconstruction.get('checks', {})
    _need(reconstruction.get('decision') == 'PASS'
          and reconstruction.get('rows') == 118249
          and reconstruction.get('feature_surface_rows') == 1140763
          and reconstruction.get('manifest_path') == REFS['split_manifest']['path']
          and reconstruction.get('manifest_sha256') == REFS['split_manifest']['sha256']
          and reconstruction.get('feature_surface_manifest_path') == REFS['feature_manifest']['path']
          and reconstruction.get('feature_surface_manifest_sha256') == REFS['feature_manifest']['sha256']
          and all(checks.get(key) is True for key in (
              'every_emitted_timestamp_maps_exactly_once_to_feature_surface',
              'every_sequence_equals_exact_source_surface_history_bit_identical',
              'every_snapshot_equals_exact_source_surface_current_row_bit_identical',
              'source_timestamps_strictly_increasing')),
          'CLOCK_AND_SOURCE_RECONSTRUCTION')
    fields = owners.compile_compact_entry_execution_profile()['physical_field_names']
    ctx = list(fields['ctx_cont'])
    _need(len(ctx) == 2542
          and feature_manifest.get('rows') == 1140763
          and feature_manifest.get('ctx_cont_dim') == 2542
          and feature_manifest.get('feature_field_order') == list(fields['signal'])
          and split['feature_contract']['ctx_cont_names'] == ctx
          and split['extra']['gc_native_context']['registry_path']
              == 'configs/research/NATIVE_GC_CONTEXT_FIELDS_20261018_049.json'
          and split['extra']['gc_native_context']['registry_sha256']
              == REFS['gc_registry']['sha256']
          and registry.get('schema_version') == 'gx1_native_gc_context_registry_v1'
          and registry.get('outcomes_accessed') is False
          and registry.get('train_decision_rows') == 118249
          and registry.get('train_parent_clock_binding', {}).get('bar_clock_sha256')
              == provider['axes']['bar_start_ns_sha256'],
          'FIELD_AND_REGISTRY_LINEAGE')
    reported = registry['field_report']
    for name, (value_col, mask_col) in FIELD_COLUMNS.items():
        _need(ctx[value_col] == name and ctx[mask_col] == name + '_observed'
              and reported[name]['decision_rows'] == 118249
              and len(reported[name]['masked_float32_train_sha256']) == 64,
              'FIELD_INDEX:' + name)
    return {'binding': binding, 'provider': provider, 'registry': registry,
            'current_stat': _file_stat(CURRENT_MMAP),
            'source_stat': _file_stat(SOURCE_MMAP)}


def _inspect_plan(plan_path: Path, plan_sha256: str, out_dir: Path
                  ) -> tuple[dict, dict, str, dict]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT
          and plan_path.is_file() and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    _need(plan.get('schema_version') == SCHEMA and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('scope') == SCOPE
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('limits') == LIMITS
          and plan.get('execution') == EXECUTION
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    owners_map = plan.get('source_owners')
    _need(isinstance(owners_map, dict) and set(owners_map) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest for path, digest in owners_map.items()),
          'SOURCE_OWNERS')
    base._receipt_preflight()
    meta = _source_metadata()
    return plan, {'path': str(plan_path), 'sha256': plan_sha256}, head, meta


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, plan_ref, head, meta = _inspect_plan(plan_path, plan_sha256, out_dir)
    return {'status': 'READY_NO_CLAIM_NO_MMAP_PAYLOAD', 'plan': plan_ref,
            'source_head': head, 'source_owners_checked': len(plan['source_owners']),
            'current_mmap_stat': meta['current_stat'],
            'selected_fields': list(FIELD_COLUMNS), 'claim_created': False,
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def _extract_fields(meta: dict) -> tuple[dict, dict]:
    """Read six fixed full-TRAIN field pairs; prove bytes against 049 registry."""
    before = _file_stat(CURRENT_MMAP)
    _need(before == meta['current_stat'], 'CURRENT_BEFORE')
    backing = np.memmap(CURRENT_MMAP, mode='r', dtype='<f4', shape=(118249, 2542))
    values: dict[str, np.ndarray] = {}
    evidence: dict[str, dict] = {}
    for name, (value_col, mask_col) in FIELD_COLUMNS.items():
        raw = np.array(backing[:, value_col], dtype=np.float32, copy=True)
        observed_raw = np.array(backing[:, mask_col], dtype=np.float32, copy=True)
        _need(np.isfinite(observed_raw).all()
              and np.isin(observed_raw, (0.0, 1.0)).all(), 'MASK_DOMAIN:' + name)
        observed = observed_raw == 1.0
        digest = _masked_hash(raw, observed)
        frozen = meta['registry']['field_report'][name]
        _need(digest == frozen['masked_float32_train_sha256']
              and int(observed.sum()) == frozen['finite_observed']
              and int((~observed).sum()) == frozen['missing']
              and np.isfinite(raw[observed]).all(), 'FIELD_BYTES:' + name)
        values[name] = raw
        values[name + '_observed'] = observed
        evidence[name] = {'value_col': value_col, 'mask_col': mask_col,
                          'masked_float32_train_sha256': digest,
                          'observed': int(observed.sum()),
                          'missing': int((~observed).sum())}
    del backing
    _need(_file_stat(CURRENT_MMAP) == before, 'CURRENT_AFTER')
    return values, evidence


def _flow_parity(values: dict) -> dict:
    volume = values['gc_m5_observed_volume']
    buy = values['gc_m5_observed_buy_known']
    sell = values['gc_m5_observed_sell_known']
    unknown = values['gc_m5_observed_unknown']
    delta = values['gc_m5_observed_known_delta_contracts']
    joint = np.logical_and.reduce([values[name + '_observed'] for name in (
        'gc_m5_observed_volume', 'gc_m5_observed_buy_known',
        'gc_m5_observed_sell_known', 'gc_m5_observed_unknown',
        'gc_m5_observed_known_delta_contracts')])
    _need(np.array_equal((buy[joint] - sell[joint]), delta[joint])
          and np.array_equal((buy[joint] + sell[joint] + unknown[joint]),
                             volume[joint]), 'TRADE_ARITHMETIC')
    return {'joint_observed_rows': int(joint.sum()),
            'delta_exact_mismatches': 0, 'volume_exact_mismatches': 0,
            'volume_field': 'gc_m5_observed_volume'}


def _axis_identity(parent: np.ndarray, decision: np.ndarray,
                   provider: dict, registry: dict) -> dict:
    """Prove parent IDs index current051 rows and share the 049 TRAIN clock."""
    n = provider['axes']['rows']
    _need(parent.dtype == decision.dtype == np.int64
          and parent.shape == decision.shape == (n,)
          and np.array_equal(parent, np.arange(n, dtype=np.int64))
          and bool(np.all(decision > target.M5_NS))
          and bool(np.all(np.diff(decision) > 0)), 'TARGET_PARENT_AXIS')
    axes_sha = {
        'parent_rows_sha256': hashlib.sha256(parent.tobytes()).hexdigest(),
        'bar_start_ns_sha256': hashlib.sha256((decision - target.M5_NS).tobytes()).hexdigest(),
        'decision_ns_sha256': hashlib.sha256(decision.tobytes()).hexdigest(),
    }
    _need(axes_sha['parent_rows_sha256'] == provider['axes']['parent_rows_sha256']
          and axes_sha['bar_start_ns_sha256'] == provider['axes']['bar_start_ns_sha256']
          and axes_sha['bar_start_ns_sha256'] == registry['train_parent_clock_binding']['bar_clock_sha256']
          and axes_sha['decision_ns_sha256'] == registry['train_decision_clock_sha256'],
          'TARGET_GC_CLOCK')
    return axes_sha


def _measure(meta: dict) -> dict:
    bound = target.load_bound_five_endpoint_targets()
    parent = bound.parent_rows
    decision = bound.decision_ns
    axes_sha = _axis_identity(parent, decision, meta['provider'], meta['registry'])
    _need(len(parent) == 118249, 'PHYSICAL_ROW_COUNT')
    values, hashes = _extract_fields(meta)
    parity = _flow_parity(values)
    folds = []
    for fid in SCOPE['fold_ids']:
        axes = bound.mature_oof_axes(fold_id=fid)
        rows, issues = axes['score_parent_rows'], axes['score_decision_ns']
        _need(len(axes['fit_parent_rows']) == EXPECTED_FIT_ROWS[fid - 1]
              and len(rows) == EXPECTED_SCORE_ROWS[fid - 1]
              and np.array_equal(decision[rows], issues)
              and int(rows[0]) >= 0 and int(rows[-1]) < len(parent),
              'SCORE_AXIS:' + str(fid))
        truth = bound.gather(rows, issues, fold_id=fid, partition='score')
        allowed = truth['allowed'][:, 0]
        eligible = truth['eligible'][:, 0]
        mature = truth['mature'][:, 0]
        certified = eligible & mature & np.isin(allowed, (1, 4))
        _need(int(certified.sum()) == EXPECTED_CERTIFIED_SIDE[fid - 1],
              'CERTIFIED_SUPPORT:' + str(fid))
        scored = core.score_fold(
            fold_id=fid, issue_ns=issues, allowed=allowed,
            eligible=eligible, mature=mature,
            buy=values['gc_m5_observed_buy_known'][rows],
            sell=values['gc_m5_observed_sell_known'][rows],
            unknown=values['gc_m5_observed_unknown'][rows],
            buy_observed=values['gc_m5_observed_buy_known_observed'][rows],
            sell_observed=values['gc_m5_observed_sell_known_observed'][rows],
            unknown_observed=values['gc_m5_observed_unknown_observed'][rows],
            ofi=values['gc_m5_observed_book_ofi_contracts'][rows],
            ofi_observed=values['gc_m5_observed_book_ofi_contracts_observed'][rows])
        _need(scored['issue_axis'] == {'count': len(rows),
              'sha256': axes['score_decision_ns_sha256']}
              and scored['overall']['certified_side'] == int(certified.sum()),
              'CORE_SCORE_AXIS:' + str(fid))
        folds.append({'fold_id': fid, 'fit_rows': len(axes['fit_parent_rows']),
                      'fit_parent_rows_sha256': axes['fit_parent_rows_sha256'],
                      'fit_issue_axis_sha256': axes['fit_decision_ns_sha256'],
                      'score_parent_rows_sha256': axes['score_parent_rows_sha256'],
                      'score': scored,
                      'immature_issue_rows': int((~mature).sum()),
                      'eligible_immature_issue_rows': int((eligible & ~mature).sum())})
    _need(_file_stat(CURRENT_MMAP) == meta['current_stat']
          and _file_stat(SOURCE_MMAP) == meta['source_stat'], 'MMAP_BOOKEND')
    return {'scope_limit': 'Six full-TRAIN GC field pairs proven; five later TRAIN M5 k1 score intervals only; no full scratch hash, model, fit, probability calibration, TEST, entry, exit or economic edge.',
            'axis_sha256': axes_sha, 'mmap': {'path': str(CURRENT_MMAP),
            'stat': meta['current_stat'], 'full_content_sha256': None},
            'selected_field_hashes': hashes, 'trade_arithmetic': parity,
            'core_contract': core.gc_flow_reference_contract(),
            'folds': folds, 'model_forwards': 0, 'optimizer_steps': 0,
            'fit_calls': 0, 'test_data_used': False}


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, plan_ref, head, meta = _inspect_plan(plan_path, plan_sha256, out_dir)
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    _need(all(cap.get(key) == value for key, value in LIMITS.items()
              if key != 'max_wall_seconds')
          and cap.get('pids') == 64 and cap.get('numerical_threads') == 1
          and lock.get('scope') == cap.get('cgroup_path', '')[len('/sys/fs/cgroup'):],
          'CAP_LIMITS')
    policy_ref = _ref(POLICY)
    policy = json.loads(POLICY.read_text())
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
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
          and gate.get('memory_max') == '4G'
          and gate.get('swap_max') == '512M'
          and gate.get('wall_seconds') == LIMITS['max_wall_seconds']
          and gate.get('test_data_used') is False
          and gate.get('checkpoint_reads_authorized') is False
          and gate.get('broker_access_allowed') is False
          and gate.get('model_forwards_authorized') == 0
          and gate.get('optimizer_steps_authorized') == 0
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('output_root') == str(OUT_ROOT),
          'POLICY_GATE')
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_BEFORE_CLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    claim = None
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError('GC_FLOW_REFERENCE_WALL_LIMIT')))
        signal.alarm(LIMITS['max_wall_seconds'])
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_gc_flow_reference_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'fold_ids': SCOPE['fold_ids'],
            'lock': lock})
        data = _measure(meta)
        _need(_sha(plan_path) == plan_sha256
              and _sha(POLICY) == policy_ref['sha256']
              and _clean_head() == head
              and _sha(out_dir / 'CLAIM.json') == claim['sha256']
              and {item.name for item in out_dir.iterdir()} == {'CLAIM.json'},
              'FINAL_SOURCE_IDENTITY')
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_gc_flow_reference_result_v1',
            'status': 'COMPLETE', 'decision': 'PASS_TRAIN_GC_FLOW_REFERENCE_ONLY',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'elapsed_seconds': time.monotonic() - began, **data})
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_gc_flow_reference_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json']})
        return {'result': result, 'terminal': terminal}
    except Exception as exc:
        failure = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_gc_flow_reference_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_gc_flow_reference_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failure, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                 if claim else ['FAILED.json', 'TERMINAL.json'])})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight_read_only(args.plan, args.plan_sha256,
                                             args.out_dir), sort_keys=True))
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
