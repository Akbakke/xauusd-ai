"""One fixed paired TRAIN-development ET252/M1-OFI experiment; no native authority.

Original 245 targets and all 3345 baseline coordinates remain unchanged. Three
fresh baseline models must reproduce original 252 states and later probabilities
before any augmented fit. No archived worker is imported or mutated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import mmap
import os
from pathlib import Path
import signal
import sys
import time
import warnings
import zipfile
import numpy as np

from gx1.scripts.audit_entry_direction_candle_midpoints_v1 import regular, git_state, read_targets

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
SCHEMA = 'gx1_direction_last_closed_m1_ofi_plan_v1'
STAGE = 'independent_direction_last_m1_ofi_20261027'
SYNTHETIC_STAGE = 'independent_direction_last_m1_ofi_synthetic_20261027'
TEST_FILES = ('tests/test_compare_entry_direction_last_closed_m1_ofi_v1.py',)
DESIGN = {'path': str(ROOT/'DESIGN.json'), 'sha256': 'a6638b593ebe41eebf1b45d60007cd2a52b04ef0a28cdb174782acb4b77b72b6'}
PROTOCOL = {'path': str(ROOT/'LAST_M1_OFI_PROTOCOL_001.json'), 'sha256': '94eba9f3d5e2e4f97f4ca97d908d07ee4bb53ccd50e09c6b99e42ad94cffe150'}
N, LATER, STOP, PARENTS = 47602, 17288, 64890, 118249
CUT, END, MINUTE = 1735689600000000000, 1743465600000000000, 60000000000
ORIGIN, M1_ROWS = 1714521600000000000, 878400
HORIZONS = (1, 5, 15)
NEW_FIELDS = ('last_closed_m1_ofi_contracts', 'last_closed_m1_ofi_observed')
INPUT_KEYS = {'train_raw', 'price_targets', 'train_m1', 'original252_oof'}
META_KEYS = {'target_metadata', 'source_map204', 'prefix_recipe210', 'original252_result',
    'original252_fold', 'original252_metrics', 'train_m1_summary', 'train_m1_review', 'qualified_roll'}
# Exact existing metadata authorities; the plan cannot silently replace a proof.
META_REFS = {
    'target_metadata': {'path': '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_BASELINE_DATA_20261024_245/OUTPUT/METADATA.json', 'sha256': 'b043ec38eb1b250817fa809ea471fbc136fc73f540b9b4749bbe3ad0113cd282'},
    'source_map204': {'path': '/home/andre2/GX1_RUNS/NATIVE_M5_GC_RELATIVE_COORDINATE_DESIGN_20261023_204/SOURCE_MAP.json', 'sha256': 'c3dd04288d93d9d78d96ac7444db2f9ba887157142513334445eb3c75a138996'},
    'prefix_recipe210': {'path': '/home/andre2/GX1_RUNS/NATIVE_M5_GC_RELATIVE_PREFIX_NORMALIZATION_REPAIR_20261023_210/OUTPUT/baseline_20250101.json', 'sha256': '9990a2f82e9ebd4a5d4896c19f6f6bab6d89c28b14c43b8350c5c6c88cc4a7fd'},
    'original252_result': {'path': '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NONLINEAR_REFERENCE_20261024_252/RESULT.json', 'sha256': '50594fb97a4a2e81f57b90f83332c7e980aff752a120f8505d99eadeca8c3c6c'},
    'original252_fold': {'path': '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NONLINEAR_REFERENCE_20261024_252/FOLD_1_FEATURE_QUALIFICATION.json', 'sha256': '931fdcdc429d5c5c628904f83e0a80a4b923174ab47d4bb7230c37ba1ebac568'},
    'original252_metrics': {'path': '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NONLINEAR_REFERENCE_20261024_252/OUTPUT/METRICS.json', 'sha256': '1d288d097e9695a5c72c27e50b9db70ca4332a7fd2ba270de28609b31345ba88'},
    'train_m1_summary': {'path': '/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/D_TRAIN_M1_REDUCTION_20261026_001/FINAL/SUMMARY.json', 'sha256': '3107449c02e941b56c76f28345ee8b958c3bee654decea7e3460e67705b5fe24'},
    'train_m1_review': {'path': '/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/D_TRAIN_M1_REDUCTION_ACTUAL_REVIEW_001.json', 'sha256': 'f89244376ca6675621e5e7b5af6df89d66f44f774e3a2798619e5ac8054d171b'},
    'qualified_roll': {'path': '/home/andre2/GX1_RUNS/NATIVE_GC_RAW_REBUILD_20261018_001/RESULTS/ROLL_SCHEDULE.json', 'sha256': '71262543da79a79ca0c6a0fa2c2d0f78494fa001d3f2409188d3ad80d4f7dc69'},
}
# Exact original252 configuration, also required against its bound fit metadata.
MODEL_CONFIG = {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini',
    'max_depth': 8, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None,
    'min_impurity_decrease': 0.0, 'min_samples_leaf': 256, 'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 32, 'n_jobs': 1,
    'oob_score': False, 'random_state': 20260911, 'verbose': 0, 'warm_start': False}
SOURCE_PATHS = ('gx1/scripts/compare_entry_direction_last_closed_m1_ofi_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/entry_direction_snapshot_data_v1.py',
    'gx1/contracts/entry_relative_price_snapshot_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py', 'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py', 'gx1/contracts/entry_model_native_bundle_commit_v1.py')


def require(ok, reason):
    if not ok:
        raise RuntimeError('DIRECTION_LAST_M1_OFI_' + reason)


def plain(ref):
    return {key: ref[key] for key in ('path', 'sha256')}


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def owned(path, out):
    p = Path(os.fsdecode(path))
    return (p.is_absolute() and '..' not in p.parts and (p == out or out in p.parents)
        and not any(x.is_symlink() for x in (p, *p.parents)))


def require_plan(plan, design, protocol):
    synthetic = plan.get('mode') == 'synthetic'
    keys = {'schema_version', 'mode', 'design', 'protocol', 'inputs', 'metadata',
        'sources', 'out_dir', 'wall_seconds'} | ({'tests'} if synthetic else set())
    require(set(plan) == keys and plan['mode'] in ('actual', 'synthetic')
        and plan['schema_version'] == SCHEMA + ('_synthetic' if synthetic else ''), 'PLAN_SCHEMA')
    require(plan['design'] == DESIGN and plan['protocol'] == PROTOCOL
        and protocol['design'] == DESIGN and protocol['schema_version'] == 'gx1_independent_direction_last_m1_protocol_v1', 'PROTOCOL_BINDING')
    require(protocol['fold'] == {'id': 1, 'train_parents': N, 'later_parents': LATER,
        'train_cutoff_ns': CUT, 'later_cutoff_ns': END}
        and protocol['arms'] == {'baseline_fields': 3345, 'augmented_fields': 3347,
            'new_fields': list(NEW_FIELDS)}
        and protocol['max_genuine_fits'] == 6 and protocol['horizons_minutes'] == list(HORIZONS)
        and protocol['baseline_before_candidate'] is True, 'FIXED_PROTOCOL')
    require(plan['wall_seconds'] == (180 if synthetic else 1200)
        and protocol['wall_seconds'] == 1200 and protocol['memory_bytes'] == 4294967296, 'WALL_MEMORY')
    require(set(plan['inputs']) == (set() if synthetic else INPUT_KEYS)
        and set(plan['metadata']) == (set() if synthetic else META_KEYS), 'INPUT_METADATA_SET')
    require(plan['metadata'] == ({} if synthetic else META_REFS), 'EXACT_METADATA_REFERENCES')
    for ref in [*plan['inputs'].values(), *plan['metadata'].values()]:
        require(set(ref) == {'path', 'sha256'}, 'PLAIN_REFERENCE')
    require(set(SOURCE_PATHS) | (set(TEST_FILES) if synthetic else set()) <= set(plan['sources']), 'SOURCE_SET')
    for relative, ref in plan['sources'].items():
        require('..' not in Path(relative).parts and not Path(relative).is_absolute()
            and relative.startswith(('gx1/', 'tests/', 'scripts/', 'configs/', '.venv/'))
            and set(ref) == {'path', 'sha256'} and ref['path'] == str(REPO/relative), 'SOURCE_PATH')
    if synthetic:
        require(plan['tests'] == list(TEST_FILES), 'TARGETED_TEST_FILES')
    else:
        expected = design['input_references_from_existing_verified_metadata']
        require(plan['inputs']['price_targets'] == plain(expected['price_targets'])
            and plan['inputs']['train_m1'] == plain(expected['train_m1_ofi'])
            and plan['metadata']['target_metadata'] == expected['target_metadata'], 'UNCHANGED_DESIGN_INPUTS')
    out = Path(plan['out_dir'])
    require(out.is_absolute() and '..' not in out.parts and out.parent == ROOT
        and out.name not in ('', '.', '..') and not out.is_symlink()
        and ROOT.resolve(strict=True) == ROOT, 'OUTPUT_ROOT')
    return out


def make_guard(out, inputs, metadata, sources, *, synthetic):
    allowed = set(inputs) | set(metadata) | set(sources)
    tests = {str(REPO/name) for name in TEST_FILES} & set(sources)
    git_prefix = ('git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null', '-C', str(REPO))
    tails = {('rev-parse', 'HEAD'), ('branch', '--show-current'), ('status', '--porcelain', '--untracked-files=all')}
    def guard(event, args):
        if event == 'subprocess.Popen':
            command = tuple(args[1]); require(command[:len(git_prefix)] == git_prefix
                and command[len(git_prefix):] in tails, 'PROCESS')
        require(event not in ('os.system', 'socket.connect', 'socket.bind', 'socket.sendto',
            'socket.getaddrinfo', 'os.exec', 'os.posix_spawn', 'os.fork', 'os.forkpty',
            'pickle.find_class', 'os.symlink'), 'NETWORK_PROCESS_PICKLE')
        if event == 'import':
            require(args[0].split('.')[0] not in ('tensorflow', 'jax', 'keras'), 'MODEL_IMPORT')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]); p = Path(name)
            write = bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            sealed = any(x.upper() == 'TEST' or x.upper().startswith(('TEST_', 'TEST.')) for x in p.parts)
            require(not sealed or (synthetic and (owned(name, out) or (name in tests and not write))), 'TEST_PATH')
            if write: require(owned(name, out), 'OUTSIDE_OUTPUT_WRITE')
            if '/GX1_DATA/' in name or '/GX1_RUNS/' in name:
                require(name in allowed or owned(name, out), 'UNDECLARED_DATA')
            if p.suffix.lower() in ('.pt', '.pth', '.pkl', '.ckpt', '.parquet', '.csv', '.gz', '.zst', '.dbn'):
                require(synthetic and owned(name, out), 'FORBIDDEN_PAYLOAD')
            if p.suffix.lower() in ('.npz', '.npy'):
                require((name in inputs and not write and not synthetic) or owned(name, out), 'PAYLOAD')
        if event in ('os.remove', 'os.rmdir', 'os.mkdir'):
            require(owned(args[0], out), 'OUTSIDE_OUTPUT_MUTATION')
        if event in ('os.rename', 'os.link'):
            require(owned(args[0], out) and owned(args[1], out), 'OUTSIDE_OUTPUT_MUTATION')
    return guard


def read_archive(path, keys):
    with regular(path).open('rb') as handle:
        with zipfile.ZipFile(handle) as archive:
            info = archive.infolist()
            require(len(info) == len(keys) and {x.filename for x in info} == {x+'.npy' for x in keys}
                and sum(x.file_size for x in info) <= 128*1024**2, 'ARCHIVE_INVENTORY')
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            require(set(archive.files) == set(keys), 'ARCHIVE_KEYS')
            values = {key: archive[key] for key in archive.files}
    require(all(not x.dtype.hasobject and x.flags.c_contiguous for x in values.values()), 'ARCHIVE_HEADERS')
    for value in values.values(): value.setflags(write=False)
    return values


def require_metadata(meta, plan, layout):
    fq, old, metrics, recipe = (meta[k] for k in ('original252_fold', 'original252_result', 'original252_metrics', 'prefix_recipe210'))
    require(fq['status'] == 'PASS_BEFORE_FOLD_FIT' and fq['fold_id'] == 1
        and fq['prefix_parent_rows'] == N and fq['holdout_parent_rows'] == LATER
        and fq['cutoff_ns'] == CUT and fq['holdout_end_ns_exclusive'] == END
        and plain(fq['raw_artifact']) == plan['inputs']['train_raw']
        and fq['raw_source_map'] == plan['metadata']['source_map204']
        and fq['normalization_recipe'] == plan['metadata']['prefix_recipe210']
        and fq['whole_prefix_layout'] == layout['contract'], 'ORIGINAL_FULL_LAYOUT')
    require(old['status'] == 'COMPLETE_FIXED_EXTRATREES_TRAIN_OOF_REFERENCE_NO_NATIVE_OR_TRADE_AUTHORITY'
        and old['source_unchanged'] is True
        and old['test_data_used'] is False and plain(old['OOF_predictions']) == plan['inputs']['original252_oof']
        and plain(old['metrics']) == plan['metadata']['original252_metrics']
        and old['fold_feature_qualifications'][1] == plan['metadata']['original252_fold'], 'ORIGINAL252_RESULT')
    fits = metrics['fits'][3:6]
    require(len(metrics['fits']) == len(old['model_roundtrip_proofs']) == 15
        and all(f['fold_id'] == 1 and f['horizon_minutes'] == h and f['config'] == MODEL_CONFIG
            and f['fit_parent_rows'] == N and f['holdout_prediction_population'] == LATER
            and f['normalization_recipe'] == plan['metadata']['prefix_recipe210']
            and f['normalization_sha256'] == fq['normalization_sha256']
            and f['model_state_sha256'] == old['model_roundtrip_proofs'][3+j]['state_sha256']
            for j, (f, h) in enumerate(zip(fits, HORIZONS))), 'ORIGINAL252_FITS')
    require(recipe['arm'] == 'baseline' and recipe['prefix']['rows'] == N
        and recipe['prefix']['cutoff_ns'] == CUT and plain(recipe['input_artifact']) == plan['inputs']['train_raw']
        and recipe['normalization']['contract_sha256'] == fq['normalization_sha256'], 'PREFIX_RECIPE')
    summary, review, roll = (meta[k] for k in ('train_m1_summary', 'train_m1_review', 'qualified_roll'))
    require(summary['rows'] == review['rows'] == M1_ROWS and plain(summary['train_table']) == plan['inputs']['train_m1']
        and plain(review['train_table']) == plan['inputs']['train_m1']
        and review['summary'] == plan['metadata']['train_m1_summary']
        and review['inner_outer_exit_codes'] == [0, 0]
        and type(review['source_and_policy_bookends_verified']) is int
        and review['source_and_policy_bookends_verified'] == 19
        and review['full_reader_receipts_verified'] == 35 and review['test_data_used'] is False
        and summary['contract_order'] == roll['contract_order'] and len(roll['contract_order']) == 15
        and roll['origin_ns'] == summary['clock_start_ns'] == ORIGIN
        and summary['clock_end_exclusive_ns'] == 1767225600000000000, 'QUALIFIED_TRAIN_M1')
    target = meta['target_metadata']
    require(target['parent_rows'] == PARENTS and target['horizons_minutes'] == list(HORIZONS)
        and target['test_data_used'] is False and plain(target['price_targets']) == plan['inputs']['price_targets'], 'ORIGINAL245_TARGETS')
    return fits


def validate_m1(m1, summary, roll):
    require(set(m1) == set(summary['array_schema']), 'M1_MEMBERS')
    for name, spec in summary['array_schema'].items():
        value = m1[name]
        require(value.dtype.str == spec['dtype_str'] and list(value.shape) == spec['shape']
            and not value.flags.writeable, 'M1_HEADER:' + name)
    clock, ci = m1['bar_start_ns'], m1['contract_index']
    require(np.array_equal(clock, ORIGIN + np.arange(M1_ROWS, dtype=np.int64)*MINUTE)
        and np.array_equal(m1['available_ns'], clock+MINUTE)
        and np.all((ci >= -1) & (ci < len(roll['contract_order'])))
        and np.array_equal(m1['assigned'], ci >= 0), 'M1_CLOCK_CONTRACT_ASSIGNMENT')
    # The summary/review binds the already-qualified selected source, not a new roll fit.
    require(np.isfinite(m1['ofi_contracts'][m1['ofi_valid']]).all()
        and not np.any(m1['ofi_valid'] & (m1['receive_clock_excluded'] | m1['book_event_state_excluded'])), 'M1_VALID_OFI_INCIDENT')


def current_m5_contract(m1, decisions):
    """Producer's five-minute contract domain; no OFI-quality requirement on prior4."""
    require(decisions.dtype == np.int64 and decisions.ndim == 1
        and np.all(decisions % (5*MINUTE) == 0), 'M5_DECISION_GRID')
    wanted = decisions[:, None] + np.arange(-5, 0, dtype=np.int64)[None, :]*MINUTE
    clock = m1['bar_start_ns']; pos = np.searchsorted(clock, wanted)
    inside = np.all(pos < len(clock), axis=1); rows = np.flatnonzero(inside)
    found = np.zeros(len(decisions), bool)
    found[rows] = np.all(clock[pos[rows]] == wanted[rows], axis=1)
    rows = np.flatnonzero(found); ids = m1['contract_index'][pos[rows]]
    valid = np.all(m1['assigned'][pos[rows]], axis=1) & np.all(ids >= 0, axis=1)
    valid &= np.all(ids == ids[:, :1], axis=1)
    result = np.full(len(decisions), -1, np.int16)
    result[rows[valid]] = ids[valid, 0]
    result.setflags(write=False)
    return result


def encode_baseline(path, destination, layout, norm):
    from gx1.contracts.entry_relative_price_snapshot_v1 import encode_normalized_snapshot_block
    raw = np.load(str(regular(path)), mmap_mode='r', allow_pickle=False)
    require(isinstance(raw, np.memmap) and raw.dtype == np.float32
        and raw.shape == (PARENTS, 3318) and not raw.flags.writeable, 'RAW_READONLY_HEADER')
    output = np.memmap(destination, mode='w+', dtype=np.float32, shape=(STOP, 3345))
    sha = hashlib.sha256()
    try:
        for left in range(0, STOP, 1024):
            right = min(left+1024, STOP)
            block, names = encode_normalized_snapshot_block(raw[left:right], layout=layout, normalization=norm)
            require(names == layout['contract']['encoded_columns'] and block.shape == (right-left, 3345)
                and block.dtype == np.float32 and np.isfinite(block).all(), 'BASELINE_ENCODE')
            output[left:right] = block; sha.update(block.tobytes()); del block
            raw._mmap.madvise(mmap.MADV_DONTNEED)
        output.flush()
    finally:
        raw._mmap.close(); output._mmap.close()
    return sha.hexdigest()


def normalize_new(projection, decisions):
    from gx1.contracts.entry_model_native_input_normalization_v1 import fit_surface_normalization, apply_surface_normalization
    observed = projection[NEW_FIELDS[1]]
    raw = np.column_stack((projection[NEW_FIELDS[0]], observed)).astype(np.float32)
    require(np.isfinite(raw).all() and len(raw) == len(decisions), 'NEW_F32_FINITE')
    prefix = decisions < CUT
    require(int(prefix.sum()) == N and np.all(prefix[:N]) and not prefix[N:].any(), 'NEW_OWN_PREFIX')
    selected = prefix & observed
    require(int(selected.sum()) >= 2, 'NEW_PREFIX_OBSERVATIONS')
    stats = fit_surface_normalization(raw[selected], surface='research_last_closed_m1_ofi',
        field_names=list(NEW_FIELDS), row_count=int(selected.sum()),
        semantic_binary_fields=[NEW_FIELDS[1]], semantic_categorical_domains={},
        allow_constant_train_fields=True, allow_unobserved_train_fields=False)
    encoded = apply_surface_normalization(raw, stats)
    encoded[~observed, 0] = 0.0
    require(np.array_equal(encoded[:, 1], observed.astype(np.float32))
        and np.isfinite(encoded).all(), 'NEW_MASK_AFTER_NORMALIZATION')
    encoded.setflags(write=False)
    return encoded, stats


def append_and_check(baseline, augmented, new, *, rows=STOP):
    require(type(rows) is int and 0 < rows <= STOP and new.dtype == np.float32
        and new.shape == (rows, 2) and np.isfinite(new).all(), 'AUGMENTATION_HEADER')
    a = np.memmap(baseline, mode='r', dtype=np.float32, shape=(rows, 3345))
    b = np.memmap(augmented, mode='w+', dtype=np.float32, shape=(rows, 3347))
    sha = hashlib.sha256()
    try:
        for left in range(0, rows, 1024):
            right = min(left+1024, rows)
            b[left:right, :3345] = a[left:right]; b[left:right, 3345:] = new[left:right]
            common = np.ascontiguousarray(b[left:right, :3345])
            require(common.tobytes() == a[left:right].tobytes(), 'ALL3345_BIT_PARITY')
            sha.update(common.tobytes()); del common
            a._mmap.madvise(mmap.MADV_DONTNEED)
        b.flush()
    finally: a._mmap.close(); b._mmap.close()
    return sha.hexdigest()


def forest_identity(model, width):
    require(model.get_params(deep=False) == MODEL_CONFIG and model.n_features_in_ == width
        and model.n_outputs_ == 1 and np.array_equal(model.classes_, [0, 1])
        and model.n_classes_ == 2 and len(model.estimators_) == 32, 'FITTED_FOREST_STATE')
    trees = []
    for tree in model.estimators_:
        state = tree.tree_.__getstate__(); nodes, values = state['nodes'], state['values']
        require(tree.n_features_in_ == width and tree.splitter == 'random'
            and not nodes.dtype.hasobject and values.dtype == np.float64
            and np.isfinite(values).all(), 'TREE_STATE')
        trees.append({'class': type(tree).__module__+'.'+type(tree).__qualname__, 'seed': int(tree.random_state),
            'max_depth': int(state['max_depth']), 'node_count': int(state['node_count']),
            'nodes_dtype': str(nodes.dtype.descr), 'nodes_shape': list(nodes.shape),
            'nodes_sha256': digest(nodes), 'values_shape': list(values.shape), 'values_sha256': digest(values)})
    state = {'class': type(model).__module__+'.'+type(model).__qualname__,
        'config': model.get_params(deep=False), 'classes_dtype': str(model.classes_.dtype),
        'classes_sha256': digest(model.classes_), 'n_features': width, 'n_outputs': 1,
        'tree_count': 32, 'trees': trees}
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_baseline_parity(states, later, expected_states, expected_later):
    require(len(states) == len(expected_states) == 3 and states == expected_states,
        'ORIGINAL252_MODEL_STATE_PARITY')
    require(later.dtype == expected_later.dtype == np.float64 and later.shape == expected_later.shape == (LATER, 3)
        and later.tobytes() == expected_later.tobytes(), 'ORIGINAL252_PROBABILITY_BIT_PARITY')


def probability_scores(y, p):
    # Original252 raw-probability convention; epsilon only defines finite logloss.
    require(y.dtype == np.int8 and y.ndim == 1 and y.shape == p.shape and len(y) > 0
        and np.isin(y, (0, 1)).all() and p.dtype == np.float64 and np.isfinite(p).all()
        and np.all((p >= 0) & (p <= 1)), 'PROBABILITY_SCORE_INPUT')
    q = np.clip(p, np.finfo(np.float64).eps, 1-np.finfo(np.float64).eps)
    loss = -(y*np.log(q)+(1-y)*np.log1p(-q)); brier = (p-y)**2
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    rank = direction_rank_diagnostics(y, np.ascontiguousarray(p), score_kind='probability_up')
    predicted = p > .5; truth = y == 1
    counts = {'TN': int((~truth & ~predicted).sum()), 'FP': int((~truth & predicted).sum()),
        'FN': int((truth & ~predicted).sum()), 'TP': int((truth & predicted).sum())}
    recall_up = counts['TP']/int(truth.sum()) if truth.any() else None
    recall_down = counts['TN']/int((~truth).sum()) if (~truth).any() else None
    bins = np.minimum((p*10).astype(np.int64), 9)
    calibration = [{'bin': j, 'count': int((bins == j).sum()),
        'probability': float(p[bins == j].mean()) if (bins == j).any() else None,
        'up_fraction': float(y[bins == j].mean()) if (bins == j).any() else None} for j in range(10)]
    return {'summary': {'rows': len(y), 'auc': rank['auc'], 'logloss': float(loss.mean()),
        'brier': float(brier.mean()), 'confusion': counts, 'up_recall': recall_up,
        'down_recall': recall_down, 'calibration_fixed10': calibration}, 'logloss': loss, 'brier': brier}


def draw_weights(blocks, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, blocks, size=(2000, blocks))
    weights = np.zeros((2000, blocks), np.int64)
    np.add.at(weights, (np.repeat(np.arange(2000), blocks), draws.reshape(-1)), 1)
    weights.setflags(write=False)
    return weights


def interval(values):
    good = np.isfinite(values); n = int(good.sum())
    bounds = np.quantile(values[good], [.025, .975], method='linear') if n else None
    return {'low': None if bounds is None else float(bounds[0]),
        'high': None if bounds is None else float(bounds[1]), 'replicates': len(values),
        'defined_replicates': n, 'undefined_replicates': int((~good).sum()), 'confidence': .95}


def paired_uncertainty(y, probabilities, decisions):
    from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs, weighted_rank_pairs
    scores = {arm: probability_scores(y, p) for arm, p in probabilities.items()}
    result = {}
    days = decisions//86400000000000
    comparisons = [('B_minus_A', 'B', 'A'), ('A_minus_prior', 'A', 'prior'),
        ('B_minus_prior', 'B', 'prior'), ('A_minus_R1', 'A', 'R1'), ('B_minus_R1', 'B', 'R1')]
    for scheme, labels, seed in (('UTC_DAY', days, 20260911), ('UTC_WEEK_MONDAY', (days+3)//7, 20260912)):
        blocks, ids = np.unique(labels, return_inverse=True); count = len(blocks)
        if count < 8:
            result[scheme] = {'blocks': count, 'status': 'INSUFFICIENT_BLOCKS', 'replicates': 0}
            continue
        weights = draw_weights(count, seed)
        auc = {}
        for arm, p in probabilities.items():
            summary = block_rank_pairs(y, p, ids)
            wins, pairs = weighted_rank_pairs(summary, weights)
            auc[arm] = np.divide(wins, 2*pairs, out=np.full(2000, np.nan), where=pairs > 0)
        denominator = weights @ np.bincount(ids, minlength=count)
        contrasts = {}
        for label, left, right in comparisons:
            detail = {'auc': interval(auc[left]-auc[right])}
            for metric in ('logloss', 'brier'):
                sums = np.bincount(ids, weights=scores[left][metric]-scores[right][metric], minlength=count)
                detail[metric] = interval((weights @ sums)/denominator)
            contrasts[label] = detail
        result[scheme] = {'blocks': count, 'status': 'PAIRED_BLOCK_BOOTSTRAP',
            'replicates': 2000, 'seed': seed, 'weights_sha256': digest(weights), 'contrasts': contrasts}
    return result


def primary_supported(uncertainty):
    for scheme in ('UTC_DAY', 'UTC_WEEK_MONDAY'):
        value = uncertainty[scheme]
        if value['status'] != 'PAIRED_BLOCK_BOOTSTRAP': return False
        pair = value['contrasts']['B_minus_A']
        if any(pair[metric]['undefined_replicates'] for metric in ('auc', 'logloss')): return False
        if not (pair['auc']['low'] is not None and pair['auc']['low'] > 0
                and pair['logloss']['high'] is not None and pair['logloss']['high'] < 0): return False
    return True


def new_field_counts(projection, contract, target):
    seen = projection[NEW_FIELDS[1]]; values = projection[NEW_FIELDS[0]]
    result = {}
    for name, sl, cutoff in (('prefix', slice(0, N), CUT), ('later', slice(N, STOP), END)):
        mask, raw = seen[sl], values[sl]
        result[name] = {'parents': len(mask), 'observed': int(mask.sum()), 'missing': int((~mask).sum()),
            'observed_zero': int((mask & (raw == 0)).sum()), 'observed_nonzero': int((mask & (raw != 0)).sum()),
            'invalid_m5_bucket_contract': int((contract[sl] < 0).sum()), 'per_horizon': []}
        for j, horizon in enumerate(HORIZONS):
            keep = target['observed'][sl, j] & (target['sign'][sl, j] != 0) & (target['terminal_time_ns'][sl, j] < cutoff)
            result[name]['per_horizon'].append({'horizon_minutes': horizon, 'same_label_rows': int(keep.sum()),
                'new_ofi_observed': int((keep & mask).sum()), 'new_ofi_missing': int((keep & ~mask).sum())})
    return result


class Calls:
    def __init__(self, synthetic):
        self.synthetic = synthetic; self.phase = None; self.baseline_passed = False
        self.counts = {'forest_fit': 0, 'tree_fit': 0, 'forest_predict': 0, 'tree_predict': 0, 'new_normalization_fit': 0}
        self.codes = {}
    def register(self, forest_class, tree_class):
        self.codes = {inspect.unwrap(forest_class.fit).__code__: 'forest_fit',
            inspect.unwrap(tree_class._fit).__code__: 'tree_fit',
            inspect.unwrap(forest_class.predict_proba).__code__: 'forest_predict',
            inspect.unwrap(tree_class.predict_proba).__code__: 'tree_predict'}
    def watch(self, frame, event, arg):
        if event == 'call':
            path, name = frame.f_code.co_filename.replace('\\', '/'), frame.f_code.co_name
            require('/gx1/models/' not in path and not ('/torch/nn/modules/module.py' in path and name == '__init__')
                and not ('/torch/optim/' in path and name in ('__init__', 'step', '_init_group'))
                and not ('/torch/cuda/' in path and name == '_lazy_init'), 'NATIVE_MODEL_OR_CUDA')
            label = self.codes.get(frame.f_code)
            if label:
                require(not self.synthetic and self.phase in ('A', 'B')
                    and (self.phase == 'A' or self.baseline_passed), 'UNADMITTED_LEARNING_CALL')
                self.counts[label] += 1
                limit = {'forest_fit': 6, 'tree_fit': 192, 'forest_predict': 12, 'tree_predict': 384}[label]
                require(self.counts[label] <= limit, 'LEARNING_CALL_BUDGET')
            if name == 'fit_surface_normalization' and path.endswith('/entry_model_native_input_normalization_v1.py'):
                if not self.synthetic:
                    require(self.phase == 'NEW_NORM', 'UNADMITTED_STATISTICS_FIT')
                    self.counts['new_normalization_fit'] += 1
                    require(self.counts['new_normalization_fit'] == 1, 'ONE_NEW_STATISTICS_FIT')
            if not self.synthetic and (name in ('fit', 'fit_transform', '_fit', 'partial_fit') and '/sklearn/' in path):
                require(frame.f_code in self.codes, 'UNAPPROVED_SKLEARN_FIT')
        if event == 'c_call':
            require(not getattr(arg, '__name__', '').startswith('_cuda'), 'CUDA')


def fit_predict(path, width, target, expected_fits, calls, forest_class):
    probabilities = np.empty((STOP, 3), np.float64); states = []; diagnostics = []
    for j, horizon in enumerate(HORIZONS):
        keep = target['observed'][:N, j] & (target['sign'][:N, j] != 0) & (target['terminal_time_ns'][:N, j] < CUT)
        y = (target['sign'][:N, j][keep] == 1).astype(np.int8)
        counts = np.bincount(y, minlength=2); old = expected_fits[j]
        require(np.all(counts > 0) and digest(keep) == old['fit_mask_sha256']
            and len(y) == old['fit_rows'] and counts.tolist() == [old['class_counts']['down'], old['class_counts']['up']], 'EXACT_ORIGINAL_FIT_POPULATION')
        matrix = np.memmap(path, mode='r', dtype=np.float32, shape=(STOP, width))
        X = np.empty((len(y), width), np.float32); at = 0
        try:
            for left in range(0, N, 1024):
                right = min(left+1024, N); block = matrix[left:right][keep[left:right]]
                X[at:at+len(block)] = block; at += len(block); del block
                matrix._mmap.madvise(mmap.MADV_DONTNEED)
        finally: matrix._mmap.close()
        require(at == len(y) and X.flags.c_contiguous and np.isfinite(X).all(), 'FIT_MATRIX')
        model = forest_class(**MODEL_CONFIG)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); model.fit(X, y)
        require(not caught, 'FIT_WARNING'); del X; gc.collect()
        state = forest_identity(model, width); states.append(state)
        split = {'ofi_value': 0, 'ofi_mask': 0}
        if width == 3347:
            for tree in model.estimators_:
                split['ofi_value'] += int((tree.tree_.feature == 3345).sum())
                split['ofi_mask'] += int((tree.tree_.feature == 3346).sum())
        matrix = np.memmap(path, mode='r', dtype=np.float32, shape=(STOP, width))
        try:
            for sl in (slice(0, N), slice(N, STOP)):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); p = model.predict_proba(matrix[sl])
                require(not caught and p.dtype == np.float64 and p.shape == (len(matrix[sl]), 2)
                    and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                    and np.all(np.abs(p.sum(axis=1)-1) <= np.finfo(np.float64).eps*32), 'PREDICTION')
                probabilities[sl, j] = p[:, 1]; del p
                matrix._mmap.madvise(mmap.MADV_DONTNEED)
        finally: matrix._mmap.close()
        require(forest_identity(model, width) == state, 'MODEL_UNCHANGED_AFTER_PREDICTION')
        diagnostics.append({'horizon_minutes': horizon, 'fit_rows': len(y), 'fit_mask_sha256': digest(keep),
            'model_state_sha256': state, 'new_field_split_counts': split})
        del model; gc.collect()
    return probabilities, states, diagnostics


def run_actual(plan, meta, stage, publish, calls):
    import sklearn
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import ExtraTreeClassifier
    from gx1.contracts.entry_relative_price_snapshot_v1 import compile_relative_price_snapshot_layout, require_prefix_snapshot_normalization
    from gx1.contracts.entry_direction_snapshot_data_v1 import gather_direction_last_closed_m1_ofi
    require(sklearn.__version__ == '1.7.2', 'SKLEARN_VERSION')
    calls.register(ExtraTreesClassifier, ExtraTreeClassifier)
    layout = compile_relative_price_snapshot_layout(source_contract=meta['source_map204'])
    fits = require_metadata(meta, plan, layout)
    norm = require_prefix_snapshot_normalization(meta['prefix_recipe210']['normalization'], layout=layout)
    target = read_targets(plan['inputs']['price_targets']['path'], np=np)
    require(np.array_equal(target['parent_rows'], np.arange(PARENTS, dtype=np.int64)), 'FULL_PARENT_ORDER')
    for value in target.values(): value.setflags(write=False)
    decisions = target['decision_ns'][:STOP]
    require(int((target['decision_ns'] < CUT).sum()) == N
        and int((target['decision_ns'] < END).sum()) == STOP, 'EXACT_FOLD_CLOCK_COUNTS')
    fq = meta['original252_fold']
    require(digest(target['parent_rows'][:STOP]) == fq['parent_order_sha256']
        and digest(decisions) == fq['decision_clock_sha256'], 'EXACT_FOLD_AXES')
    summary = meta['train_m1_summary']
    m1 = read_archive(plan['inputs']['train_m1']['path'], set(summary['array_schema']))
    validate_m1(m1, summary, meta['qualified_roll'])
    contract = current_m5_contract(m1, decisions)
    projection = gather_direction_last_closed_m1_ofi(m1, parent_rows=target['parent_rows'][:STOP],
        decision_times_ns=decisions, current_m5_contract_index=contract)
    counts = new_field_counts(projection, contract, target)
    del m1; gc.collect()
    keys = set('parent_rows decision_ns fold_id holdout_end_ns_exclusive own_prefix_prior_logit_up sign observed terminal_time_ns raw_ret_1 binary_mature probability_up direction_class saved_R1_probability_up saved_R1_logit_up'.split())
    old = read_archive(plan['inputs']['original252_oof']['path'], keys)
    member = old['fold_id'] == 1
    require(np.array_equal(old['parent_rows'], np.arange(29849, PARENTS, dtype=np.int64))
        and int(member.sum()) == LATER and np.array_equal(old['parent_rows'][member], target['parent_rows'][N:STOP])
        and np.array_equal(old['decision_ns'][member], decisions[N:])
        and old['probability_up'].dtype == np.float64 and old['probability_up'].shape == (88400, 3), 'ORIGINAL252_COHORT')
    for key in ('sign', 'observed', 'terminal_time_ns'):
        require(old[key][member].tobytes() == target[key][N:STOP].tobytes(), 'ORIGINAL252_LABEL_MASK:' + key)
    require(np.all(old['holdout_end_ns_exclusive'][member] == END), 'ORIGINAL252_LATER_CUTOFF')
    calls.phase = 'NEW_NORM'; new, new_stats = normalize_new(projection, decisions); calls.phase = None
    baseline, augmented = stage/'baseline.npy', stage/'augmented.npy'
    baseline_sha = encode_baseline(plan['inputs']['train_raw']['path'], baseline, layout, norm)
    common_sha = append_and_check(baseline, augmented, new)
    require(common_sha == baseline_sha, 'FULL_BASELINE_A_B_HASH')
    publish(stage/'FEATURE_QUALIFICATION.json', {'status': 'PASS_BEFORE_ANY_FOREST_FIT',
        'baseline_f32_sha256': baseline_sha, 'augmented_first3345_f32_sha256': common_sha,
        'new_fields_f32_sha256': digest(new), 'new_normalization': new_stats,
        'new_field_counts': counts, 'prefix_normalization_sha256': norm['contract_sha256'],
        'old_statistics_refitted': False, 'historical_full_matrix_bit_parity_measured': False})
    calls.phase = 'A'
    a, states, diagnostics_a = fit_predict(baseline, 3345, target, fits, calls, ExtraTreesClassifier)
    require_baseline_parity(states, a[N:], [x['model_state_sha256'] for x in fits],
        np.ascontiguousarray(old['probability_up'][member]))
    calls.phase = None; calls.baseline_passed = True
    publish(stage/'BASELINE_PARITY.json', {'status': 'PASS_BEFORE_ANY_AUGMENTED_FIT',
        'model_state_sha256': states, 'later_probability_sha256': digest(a[N:]),
        'original_later_probability_sha256': digest(old['probability_up'][member]), 'diagnostics': diagnostics_a})
    calls.phase = 'B'; b, _, diagnostics_b = fit_predict(augmented, 3347, target, fits, calls, ExtraTreesClassifier); calls.phase = None
    panels, primary = [], None
    for j, horizon in enumerate(HORIZONS):
        keep = target['observed'][N:STOP, j] & (target['sign'][N:STOP, j] != 0) & (target['terminal_time_ns'][N:STOP, j] < END)
        require(np.array_equal(keep, old['binary_mature'][member, j]), 'ORIGINAL252_BINARY_MATURE')
        y = (target['sign'][N:STOP, j][keep] == 1).astype(np.int8)
        prior_z = old['own_prefix_prior_logit_up'][member, j]
        prior = 1/(1+np.exp(-prior_z[keep]))
        probabilities = {'A': np.ascontiguousarray(a[N:, j][keep]), 'B': np.ascontiguousarray(b[N:, j][keep]),
            'prior': np.ascontiguousarray(prior), 'R1': np.ascontiguousarray(old['saved_R1_probability_up'][member, j][keep])}
        uncertainty = paired_uncertainty(y, probabilities, decisions[N:][keep])
        summaries = {arm: probability_scores(y, p)['summary'] for arm, p in probabilities.items()}
        observed_subset = projection[NEW_FIELDS[1]][N:][keep]
        subset = {arm: probability_scores(y[observed_subset], p[observed_subset])['summary']
            for arm, p in probabilities.items()} if observed_subset.any() else None
        prefix_keep = target['observed'][:N, j] & (target['sign'][:N, j] != 0) & (target['terminal_time_ns'][:N, j] < CUT)
        prefix_y = (target['sign'][:N, j][prefix_keep] == 1).astype(np.int8)
        prefix = {arm: probability_scores(prefix_y, np.ascontiguousarray(p[:N, j][prefix_keep]))['summary'] for arm, p in (('A', a), ('B', b))}
        panels.append({'horizon_minutes': horizon, 'prefix': prefix, 'later': summaries,
            'uncertainty': uncertainty, 'later_valid_new_ofi_subset_secondary': subset,
            'later_mask_sha256': digest(keep), 'later_scored_rows': int(keep.sum())})
        if horizon == 5: primary = primary_supported(uncertainty)
    require(calls.counts == {'forest_fit': 6, 'tree_fit': 192, 'forest_predict': 12,
        'tree_predict': 384, 'new_normalization_fit': 1}, 'EXACT_GENUINE_CALL_COUNTS')
    # Preserve predictions, never raw OFI, trees, pickle or model weights.
    np.savez_compressed(stage/'PREDICTIONS.npz', parent_rows=target['parent_rows'][:STOP], decision_ns=decisions,
        probability_A=a, probability_B=b, later_binary_mature=old['binary_mature'][member])
    baseline.unlink(); augmented.unlink()
    prediction_ref = {'path': str(stage.parent/'OUTPUT/PREDICTIONS.npz'),
        'sha256': hashlib.sha256((stage/'PREDICTIONS.npz').read_bytes()).hexdigest()}
    return {'schema_version': 'gx1_direction_last_closed_m1_ofi_result_v1', 'status': 'COMPLETE',
        'primary_supported': primary, 'panels': panels, 'new_field_counts': counts,
        'predictions': prediction_ref, 'sklearn_version': sklearn.__version__,
        'input_scope': 'Same compact 3345-coordinate snapshot in both arms; two isolated research-only OFI fields appended in B.',
        'baseline_diagnostics': diagnostics_a, 'augmented_diagnostics': diagnostics_b,
        'operation_counts': calls.counts, 'protocol': PROTOCOL,
        'scope': 'One fixed-seed paired reused-TRAIN-development augmentation; no causal isolation, native, generalization or economic qualification.'}


def run_synthetic(stage, publish):
    import pytest
    require(os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not os.environ.get('PYTEST_ADDOPTS') and sys.dont_write_bytecode, 'SYNTHETIC_ENVIRONMENT')
    class Results:
        def __init__(self):
            self.nodes = []; self.collection = []; self.reports = []; self.phases = {}; self.deselected = []; self.errors = []
        def pytest_collectreport(self, report):
            self.collection.append({'nodeid': report.nodeid, 'outcome': report.outcome, 'error': None if report.passed else str(report.longrepr)})
        def pytest_collection_finish(self, session): self.nodes = [item.nodeid for item in session.items]
        def pytest_deselected(self, items): self.deselected.extend(item.nodeid for item in items)
        def pytest_runtest_logreport(self, report):
            self.phases.setdefault(report.nodeid, []).append(report.when)
            self.reports.append({'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome, 'error': None if report.passed else str(report.longrepr)})
        def pytest_internalerror(self, excrepr, excinfo): self.errors.append(str(excrepr))
    result = Results(); code = error = None; previous = sys.pycache_prefix
    cache = stage/'python-cache'; cache.mkdir(exist_ok=False)
    try:
        sys.pycache_prefix = str(cache)
        code = pytest.main(['-c', '/dev/null', '--rootdir', str(REPO), '--noconftest', '-s', '-q',
            '-p', 'no:cacheprovider', '-p', 'no:logging', '--assert=plain',
            '--basetemp', str(stage/'pytest-temp'), *TEST_FILES], plugins=[result])
    except Exception as exc: error = exc
    finally:
        sys.pycache_prefix = previous
        publish(stage/'SYNTHETIC_DIAGNOSTICS.json', {'pytest_exit_code': None if code is None else int(code),
            'exception': None if error is None else type(error).__name__+': '+str(error),
            'collected_nodeids': result.nodes, 'collection_reports': result.collection, 'runtest_reports': result.reports,
            'runtest_phases': result.phases, 'deselected': result.deselected, 'internal_errors': result.errors,
            'bytecode_prefix_restored': sys.pycache_prefix == previous})
    if error: raise error
    require(code == 0 and result.nodes and len(set(result.nodes)) == len(result.nodes)
        and {x.split('::')[0] for x in result.nodes} == set(TEST_FILES) and not result.deselected and not result.errors
        and all(x['outcome'] == 'passed' for x in result.collection+result.reports)
        and set(result.phases) == set(result.nodes) and all(x == ['setup', 'call', 'teardown'] for x in result.phases.values()), 'ALL_TARGETED_TEST_PHASES')
    return {'status': 'PASS_SYNTHETIC_ONLY', 'tests': len(result.nodes), 'test_files': list(TEST_FILES),
        'collected_nodeids': result.nodes, 'genuine_payloads': 0, 'genuine_fits': 0, 'native_calls': 0}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'): parser.add_argument('--'+key, required=True)
    args = parser.parse_args(argv)
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    initial = git_state(); require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_HEAD_BRANCH_CLEAN')
    plan_ref = {'path': str(regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        protocol = binding.read_bound_json(Path(plan['protocol']['path']), plan['protocol']['sha256'])
        out = require_plan(plan, design, protocol); synthetic = plan['mode'] == 'synthetic'
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        admitted = policy[SYNTHETIC_STAGE if synthetic else STAGE]
        require(admitted['status'] == 'ADMITTED' and admitted['plan'] == plan_ref
            and admitted['execution_authorized'] is True and admitted['max_invocations'] == 1
            and admitted['claimed_invocations'] == admitted['completed_invocations'] == 0
            and admitted['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized')), 'GLOBAL_GATES')
        for ref in plan['sources'].values(): regular(ref['path']); binding.require_binding(ref, label='source')
        meta = {key: binding.read_bound_json(Path(ref['path']), ref['sha256']) for key, ref in plan['metadata'].items()}
        snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value): publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json', {'plan': plan_ref, 'protocol': PROTOCOL, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    metadata_paths = [plan_ref['path'], policy_ref['path'], DESIGN['path'], PROTOCOL['path'], *[x['path'] for x in plan['metadata'].values()]]
    sys.addaudithook(make_guard(out, [x['path'] for x in plan['inputs'].values()], metadata_paths,
        [x['path'] for x in plan['sources'].values()], synthetic=synthetic))
    calls = Calls(synthetic); previous_profile = sys.getprofile(); previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(*unused): raise RuntimeError('DIRECTION_LAST_M1_OFI_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(calls.watch)
    stage = out/'.OUTPUT.staging'; started = time.monotonic()
    input_snapshot = None; previous_numba_cache = os.environ.get('NUMBA_CACHE_DIR')
    try:
        # Genuine payload hashing is itself capped work and subject to the guard.
        if not synthetic:
            with binding.fresh_file_verification_scope():
                for ref in plan['inputs'].values():
                    regular(ref['path']); binding.require_binding(ref, label='input')
                input_snapshot = binding.capture_verified_file_snapshot()
        stage.mkdir(exist_ok=False)
        # Numba caches its config at first import; do not reuse an old locator.
        require(not any(name == 'numba' or name.startswith('numba.') for name in sys.modules),
            'NUMBA_NOT_PREIMPORTED')
        numba_cache = stage/'numba-cache'; numba_cache.mkdir(exist_ok=False)
        os.environ['NUMBA_CACHE_DIR'] = str(numba_cache)
        result = run_synthetic(stage, publish) if synthetic else run_actual(plan, meta, stage, publish, calls)
        snapshot.require_unchanged()
        if input_snapshot is not None: input_snapshot.require_unchanged()
        require(git_state() == initial, 'FINAL_HEAD_BRANCH_CLEAN')
        require(cap.require_capped_lock_ancestry() == lock, 'FINAL_LOCK_IDENTITY')
        result.update(plan=plan_ref, sources=plan['sources'], inputs=plan['inputs'], metadata=plan['metadata'])
        publish(stage/'RESULT.json', result)
        publish(stage/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
            'test_data_used': False, 'native_calls': 0, 'weights_saved': False, 'operation_counts': calls.counts,
            'elapsed_seconds': time.monotonic()-started})
        publish_bundle_directory_noreplace(stage, out/'OUTPUT')
        publish(out/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'source_head': args.source_head, 'source_unchanged': True, 'result_path': str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'error': type(exc).__name__+': '+str(exc), 'partial_output_is_invalid': True, 'relaunch_allowed': False})
        raise
    finally:
        if previous_numba_cache is None: os.environ.pop('NUMBA_CACHE_DIR', None)
        else: os.environ['NUMBA_CACHE_DIR'] = previous_numba_cache
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
