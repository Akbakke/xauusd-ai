"""One admitted TRAIN candle-midpoint audit; no index, model, fitting or trading.

Only five exact inputs are allowed. The original eleven-array archive remains
unchanged; the pure comparison must prove legacy parity before alternative-price
statistics. Outputs contain aggregates and hashes, never market arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_candle_audit_20261027'
SYNTHETIC_STAGE = 'independent_direction_candle_synthetic_20261027'
TEST_FILES = ('tests/test_entry_direction_price_projection_v1.py',
    'tests/test_entry_direction_snapshot_data_v1.py',
    'tests/test_audit_entry_direction_candle_midpoints_v1.py')
TEST_HELPERS = ('tests/test_entry_causal_m1_outcomes.py',)
SCHEMA = 'gx1_direction_candle_midpoint_audit_plan_v1'
SUBSET_SHA = 'd938827ef160896f0a27a5cea86bd60e0e8eb5a3561cc0aeb7f1bc006c5e8aab'
DESIGN_SHA = 'a6638b593ebe41eebf1b45d60007cd2a52b04ef0a28cdb174782acb4b77b72b6'
CUTOFF = 1767225600000000000
PARENTS = 118249
COLUMNS = ('time', 'open', 'bid_open', 'ask_open')
YEAR_SPEC = ((2024, 354866, 5076522), (2025, 353499, 5431388))
COUNTS = {'observed': [118237, 117799, 116936], 'missing': [12, 450, 1313],
          'observed_zero': [476, 194, 99]}
TARGET_DTYPES = dict(parent_rows='int64', bar_start_ns='int64', dynamic_entry_m1_row='int64',
    decision_ns='int64', terminal_time_ns='int64', elapsed_wall_seconds='int64',
    terminal_source_row='int64', observed='bool', mid_move='float64',
    mid_return_bps='float64', sign='int8')
SOURCE_PATHS = ('gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/entry_direction_price_projection_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py')


def require(ok, reason):
    if not ok:
        raise RuntimeError('DIRECTION_CANDLE_AUDIT_' + reason)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
        and not any(p.is_symlink() for p in (path, *path.parents))
        and stat.S_ISREG(path.lstat().st_mode), 'CANONICAL_REGULAR_FILE')
    return path


def git_state():
    prefix = ['git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null', '-C', str(REPO)]
    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}
    return tuple(subprocess.check_output(prefix + tail, env=env, text=True).strip()
        for tail in (['rev-parse', 'HEAD'], ['branch', '--show-current'],
                     ['status', '--porcelain', '--untracked-files=all']))


def require_plan(plan, design):
    mode = plan.get('mode')
    fields = {'schema_version', 'mode', 'design', 'metadata_review', 'inputs', 'sources', 'out_dir', 'wall_seconds'}
    fields |= {'tests'} if mode == 'synthetic' else {'expected_counts'}
    require(mode in ('actual', 'synthetic') and set(plan) == fields
        and plan['schema_version'] == (SCHEMA + '_synthetic' if mode == 'synthetic' else SCHEMA), 'PLAN_SCHEMA')
    require(plan['design'] == {'path': str(ROOT/'DESIGN.json'), 'sha256': DESIGN_SHA}
        and plan['metadata_review']['path'] == str(ROOT/'INPUT_METADATA_REVIEW.json'), 'DESIGN_BINDING')
    expected = design['input_references_from_existing_verified_metadata']
    require(set(plan['inputs']) == (set() if mode == 'synthetic' else
        {'native_2024', 'native_2025', 'native_manifest', 'price_targets', 'target_metadata'}), 'FIVE_INPUTS')
    for key, ref in plan['inputs'].items():
        require(set(ref) == {'path', 'sha256'} and ref == {k: expected[key][k] for k in ref}, 'INPUT_BINDING:' + key)
    require(plan['wall_seconds'] == 180 and (plan['tests'] == list(TEST_FILES) if mode == 'synthetic'
        else plan['expected_counts'] == COUNTS), 'COUNTS_OR_WALL')
    out = Path(plan['out_dir'])
    require(out.parent == ROOT and out.name not in ('', '.', '..') and not out.is_symlink()
        and ROOT.resolve(strict=True) == ROOT, 'OUTPUT_ROOT')
    require(isinstance(plan['sources'], dict)
        and (set(SOURCE_PATHS) | (set(TEST_FILES + TEST_HELPERS) if mode == 'synthetic' else set())) <= set(plan['sources']), 'SOURCE_SET')
    for relative, ref in plan['sources'].items():
        require(set(ref) == {'path', 'sha256'} and ref['path'] == str(REPO/relative)
            and relative.startswith(('gx1/', 'tests/', 'scripts/'))
            and '..' not in Path(relative).parts, 'SOURCE_PATH')
    require(design['phase_A']['scope'] == {'cutoff_exclusive_utc': '2026-01-01T00:00:00Z',
        'decision': 'M5 bar_start + 300 seconds', 'horizons_minutes': [1, 5, 15],
        'input_columns': list(COLUMNS), 'parents': PARENTS, 'year_files': [2024, 2025]}
        and design['phase_A']['expected_existing_counts'] == COUNTS, 'DESIGN_SCOPE')
    return out


def require_metadata(manifest, metadata, inputs, review):
    require(metadata['schema_version'] == 'gx1_direction_baseline_data_metadata_v1'
        and metadata['parent_rows'] == PARENTS and metadata['horizons_minutes'] == [1, 5, 15]
        and metadata['full_parent_order_retained'] is True and metadata['test_data_used'] is False
        and {k: metadata['price_targets'][k] for k in ('path', 'sha256')} == inputs['price_targets'], 'TARGET_METADATA')
    authority = metadata['target_report']['canonical_source_metadata']['source_quote_binding']['authority']
    proof = authority['base28_native_m1_subset_proof']
    require(authority['native_m1_manifest_path'] == inputs['native_manifest']['path']
        and authority['native_m1_manifest_sha256'] == inputs['native_manifest']['sha256']
        and proof['method'] == 'exact_quote_complete_pretest_rows_are_native_m1_subset_v1'
        and proof['proof_sha256'] == review['source_subset_proof_sha256'] == SUBSET_SHA
        and manifest['canonical_rows_sha256'] == authority['native_m1_canonical_rows_sha256']
        and manifest['source_chunks_sha256'] == authority['native_m1_source_chunks_sha256'], 'NATIVE_SUBSET_PROOF')
    require(manifest['timestamp_semantics'] == 'bar_start_utc'
        and manifest['source_kind'] == 'oanda_native_mba_candles'
        and manifest['completion_field'] == 'complete' and manifest['completion_value'] is True, 'NATIVE_SEMANTICS')
    for year, rows, offset in YEAR_SPEC:
        label = 'year=' + str(year)
        before = sum(value for key, value in manifest['year_rows'].items() if key < label)
        require(manifest['year_rows'][label] == proof['years'][label]['rows'] == rows
            and manifest['year_sha256'][label] == inputs['native_' + str(year)]['sha256']
            and before == offset, 'YEAR_ROWS_HASH_OFFSET:' + label)


def read_year_quotes(path, *, expected_rows, year, parquet, arrow, np):
    # Explicit Python file object avoids a directory/dataset discovery route.
    with regular(path).open('rb') as handle:
        table = parquet.read_table(handle, columns=list(COLUMNS), use_threads=False)
    require(table.column_names == list(COLUMNS) and table.num_rows == expected_rows
        and all(table[name].null_count == 0 for name in COLUMNS), 'YEAR_COLUMNS_ROWS_NULLS')
    clock = table['time']
    require(arrow.types.is_timestamp(clock.type) and clock.type.tz in (None, 'UTC'), 'YEAR_TIME_TYPE')
    times = clock.cast(arrow.timestamp('ns', tz=clock.type.tz)).cast(arrow.int64()).to_numpy()
    low = np.datetime64(str(year) + '-01-01', 'ns').astype(np.int64)
    high = np.datetime64(str(year + 1) + '-01-01', 'ns').astype(np.int64)
    require(times.dtype == np.int64 and np.all(np.diff(times) > 0)
        and np.all((times >= low) & (times < high)) and np.all(times % (60*10**9) == 0), 'YEAR_CLOCK')
    result = {'time_ns': np.ascontiguousarray(times)}
    for name in COLUMNS[1:]:
        require(arrow.types.is_float64(table[name].type), 'YEAR_PRICE_TYPE:' + name)
        value = table[name].to_numpy()
        require(value.dtype == np.float64 and np.isfinite(value).all() and np.all(value > 0), 'YEAR_PRICE:' + name)
        result[name] = np.ascontiguousarray(value)
    require(np.all(result['bid_open'] <= result['ask_open']), 'YEAR_CROSSED_QUOTES')
    return result


def gather_quotes(source, wanted, *, cutoff_ns, np):
    require(wanted.dtype == np.int64 and wanted.ndim in (1, 2), 'WANTED_CLOCK')
    times = source['time_ns']; pos = np.searchsorted(times, wanted)
    found = (pos < len(times)) & (wanted < cutoff_ns)
    found[found] &= times[pos[found]] == wanted[found]
    out = {'time_ns': np.where(found, wanted, 0).astype(np.int64),
           'source_row': np.full(wanted.shape, -1, np.int64), 'observed': found.copy()}
    out['source_row'][found] = source['source_row'][pos[found]]
    for name in COLUMNS[1:]:
        out[name] = np.zeros(wanted.shape, np.float64)
        out[name][found] = source[name][pos[found]]
    return out


def read_targets(path, *, np):
    import zipfile
    with regular(path).open('rb') as handle:
        with zipfile.ZipFile(handle) as archive:
            require(len(archive.infolist()) == 11 and sum(x.file_size for x in archive.infolist()) <= 64*1024**2,
                'ARCHIVE_BOUNDS')
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            require(set(archive.files) == set(TARGET_DTYPES), 'ELEVEN_TARGET_KEYS')
            saved = {name: archive[name].copy() for name in archive.files}
    for name, dtype in TARGET_DTYPES.items():
        shape = (PARENTS,) if name in ('parent_rows', 'bar_start_ns', 'dynamic_entry_m1_row', 'decision_ns') else (PARENTS, 3)
        require(saved[name].dtype == np.dtype(dtype) and saved[name].shape == shape, 'TARGET_DTYPE_SHAPE:' + name)
    require(np.all(np.diff(saved['parent_rows']) > 0) and np.all(np.diff(saved['decision_ns']) > 0)
        and np.array_equal(saved['decision_ns'], saved['bar_start_ns'] + 300*10**9)
        and np.all((saved['decision_ns'] > 0) & (saved['decision_ns'] < CUTOFF))
        and np.all(saved['decision_ns'] % (300*10**9) == 0)
        and np.array_equal(saved['terminal_time_ns'], saved['decision_ns'][:,None]
            + np.asarray([1,5,15], np.int64)[None,:]*60*10**9), 'PARENT_DECISION_CLOCK')
    require(saved['observed'].sum(axis=0).tolist() == COUNTS['observed']
        and (~saved['observed']).sum(axis=0).tolist() == COUNTS['missing']
        and (saved['observed'] & (saved['sign'] == 0)).sum(axis=0).tolist() == COUNTS['observed_zero'], 'TARGET_COUNTS')
    return saved


def make_guard(out, inputs, metadata, *, synthetic=False, source_paths=(), output_payload_paths=()):
    allowed = set(inputs) | set(metadata)
    test_sources = {str(REPO/name) for name in TEST_FILES + TEST_HELPERS} & set(source_paths)
    git_prefix = ('git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null', '-C', str(REPO))
    git_tails = {('rev-parse', 'HEAD'), ('branch', '--show-current'), ('status', '--porcelain', '--untracked-files=all')}
    def owned(value):
        p = Path(os.fsdecode(value))
        return (p.is_absolute() and '..' not in p.parts and (p == out or out in p.parents)
            and not any(x.is_symlink() for x in (p, *p.parents)))
    output_payloads = set(output_payload_paths)
    require(all(isinstance(name, str) and Path(name).suffix == '.npz' and owned(name)
        for name in output_payloads), 'OUTPUT_PAYLOAD_SCOPE')
    def guard(event, args):
        if event == 'subprocess.Popen':
            command = tuple(args[1])
            require(command[:len(git_prefix)] == git_prefix and command[len(git_prefix):] in git_tails,
                'PROCESS')
        require(event not in ('os.system', 'socket.connect', 'socket.bind', 'socket.sendto', 'socket.getaddrinfo',
            'os.exec', 'os.posix_spawn', 'os.fork', 'os.forkpty', 'pickle.find_class', 'os.symlink'), 'NETWORK_PROCESS_PICKLE')
        if event == 'import':
            require(args[0].split('.')[0] not in (('sklearn', 'joblib', 'tensorflow', 'jax', 'keras') if synthetic else
                ('torch', 'sklearn', 'joblib', 'tensorflow', 'jax', 'keras')), 'MODEL_IMPORT')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]); p = Path(name)
            writing = bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            sealed = any(part.upper() == 'TEST' or part.upper().startswith(('TEST_', 'TEST.')) for part in p.parts)
            require(not sealed or (synthetic and (owned(name) or (name in test_sources and not writing))), 'TEST_PATH')
            if writing:
                require(owned(name), 'OUTSIDE_OUTPUT_WRITE')
            if '/GX1_DATA/' in name or '/GX1_RUNS/' in name:
                require(name in allowed or owned(name), 'UNDECLARED_DATA')
            if p.suffix.lower() in ('.parquet', '.npz', '.npy', '.csv', '.gz', '.zst', '.dbn', '.pt', '.pth', '.pkl', '.ckpt'):
                require((synthetic and owned(name)) or (name in inputs and not writing)
                    or (name in output_payloads and owned(name)), 'PAYLOAD')
        if event in ('os.remove', 'os.rmdir', 'os.mkdir'):
            require(owned(args[0]), 'OUTSIDE_OUTPUT_MUTATION')
        if event in ('os.rename', 'os.link'):
            require(owned(args[0]) and owned(args[1]), 'OUTSIDE_OUTPUT_MUTATION')
    return guard


def no_queries_or_models(frame, event, arg):
    if event == 'call':
        path = frame.f_code.co_filename.replace('\\', '/')
        name = frame.f_code.co_name
        require('/gx1/models/' not in path and not path.endswith('/entry_causal_m1_outcomes_v1.py')
            and name not in ('project_direction_price_endpoints', 'project_direction_minute_curve'), 'MODEL_INDEX_PATH_CALL')


def no_native_models(frame, event, arg):
    if event == 'call':
        path = frame.f_code.co_filename.replace('\\', '/')
        require('/gx1/models/' not in path
            and not (path.endswith('/torch/nn/modules/module.py') and frame.f_code.co_name == '__init__')
            and not ('/torch/optim/' in path and frame.f_code.co_name in ('__init__', 'step', '_init_group')),
            'SYNTHETIC_NATIVE_MODEL_CALL')
        require(not (path.endswith('/torch/cuda/__init__.py') and frame.f_code.co_name == '_lazy_init'),
            'SYNTHETIC_CUDA')
    if event == 'c_call':
        require(not getattr(arg, '__name__', '').startswith('_cuda'), 'SYNTHETIC_CUDA')


def run_synthetic(stage, *, np, publish):
    import pytest
    require(os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not os.environ.get('PYTEST_ADDOPTS') and sys.dont_write_bytecode,
        'SYNTHETIC_ENVIRONMENT')
    class Results:
        def __init__(self):
            self.collected = []; self.phases = {}; self.by_file = {}
            self.collection_reports = []; self.runtest_reports = []
            self.deselected = []; self.internal_errors = []
        def pytest_collectreport(self, report):
            self.collection_reports.append({'nodeid': report.nodeid, 'outcome': report.outcome,
                'longrepr': None if report.passed else str(report.longrepr)})
        def pytest_collection_finish(self, session):
            self.collected = [item.nodeid for item in session.items]
            self.by_file = {name: sum(node.split('::')[0] == name for node in self.collected) for name in TEST_FILES}
        def pytest_deselected(self, items):
            self.deselected.extend(item.nodeid for item in items)
        def pytest_runtest_logreport(self, report):
            self.phases.setdefault(report.nodeid, []).append(report.when)
            self.runtest_reports.append({'nodeid': report.nodeid, 'when': report.when,
                'outcome': report.outcome, 'longrepr': None if report.passed else str(report.longrepr)})
        def pytest_internalerror(self, excrepr, excinfo):
            self.internal_errors.append(str(excrepr))
    result = Results(); code = None; error = None
    cache = stage/'python-cache'; cache.mkdir(exist_ok=False)
    require(not any(cache.iterdir()), 'SYNTHETIC_EMPTY_BYTECODE_PREFIX')
    previous_prefix = sys.pycache_prefix
    try:
        sys.pycache_prefix = str(cache)
        code = pytest.main(['-c', '/dev/null', '--rootdir', str(REPO), '--noconftest', '-s', '-q',
            '-p', 'no:cacheprovider', '-p', 'no:logging', '--assert=plain',
            '--basetemp', str(stage/'pytest-temp'), *TEST_FILES], plugins=[result])
    except Exception as exc:
        error = exc
    finally:
        sys.pycache_prefix = previous_prefix
        publish(stage/'SYNTHETIC_DIAGNOSTICS.json', {
            'schema_version': 'gx1_direction_candle_synthetic_diagnostics_v1',
            'pytest_exit_code': None if code is None else int(code),
            'pytest_exception': None if error is None else type(error).__name__ + ': ' + str(error),
            'collected_nodeids': result.collected, 'tests_by_file': result.by_file,
            'collection_reports': result.collection_reports, 'runtest_reports': result.runtest_reports,
            'runtest_phases': result.phases, 'deselected_nodeids': result.deselected,
            'internal_errors': result.internal_errors, 'bytecode_prefix_restored': sys.pycache_prefix == previous_prefix})
    if error is not None:
        raise error
    require(len(result.collected) > 0 and len(set(result.collected)) == len(result.collected), 'COLLECTED_CASES')
    require({node.split('::')[0] for node in result.collected} == set(TEST_FILES), 'COLLECTED_FILES')
    require(not result.deselected, 'DESELECTED_TESTS')
    require(int(code) == 0 and not result.internal_errors
        and all(report['outcome'] == 'passed' for report in result.collection_reports)
        and all(report['outcome'] == 'passed' for report in result.runtest_reports)
        and set(result.phases) == set(result.collected)
        and all(value == ['setup', 'call', 'teardown'] for value in result.phases.values()), 'ALL_TEST_PHASES_PASSED')
    return {'schema_version': 'gx1_direction_candle_synthetic_qualification_v1', 'status': 'PASS_SYNTHETIC_ONLY',
        'tests': len(result.collected), 'tests_by_file': result.by_file, 'collected_nodeids': result.collected,
        'runtest_phases': result.phases, 'skipped': 0, 'deselected': 0, 'genuine_payloads': 0,
        'native_calls': 0, 'fits': 0, 'optimizer_updates': 0, 'predictive_value_qualified': False}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    # Imports happen before the guard; no payload may be touched here.
    import numpy as np
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import pyarrow.dataset  # Resolve optional Arrow imports before sealing I/O.
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_direction_price_projection_v1 import compare_direction_native_midpoint
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'PRELOADED_MODEL')
    initial = git_state()
    require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_HEAD_BRANCH_CLEAN')
    plan_ref = {'path': str(regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        out = require_plan(plan, design); synthetic = plan['mode'] == 'synthetic'
        review = binding.read_bound_json(Path(plan['metadata_review']['path']), plan['metadata_review']['sha256'])
        require(review['design'] == plan['design'] and review['source_subset_proof_sha256'] == SUBSET_SHA, 'METADATA_REVIEW')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        admitted = policy[SYNTHETIC_STAGE if synthetic else STAGE]
        require(admitted['status'] == 'ADMITTED' and admitted['plan'] == plan_ref
            and admitted['execution_authorized'] is True and admitted['max_invocations'] == 1
            and admitted['claimed_invocations'] == admitted['completed_invocations'] == 0
            and admitted['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized')), 'GLOBAL_GATES')
        for ref in plan['sources'].values():
            regular(ref['path']); binding.require_binding(ref, label='source')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    metadata_paths = [plan_ref['path'], policy_ref['path'], plan['design']['path'], plan['metadata_review']['path']]
    sys.addaudithook(make_guard(out, [x['path'] for x in plan['inputs'].values()], metadata_paths, synthetic=synthetic,
        source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started = time.monotonic(); previous_profile = sys.getprofile()
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(*unused):
        raise RuntimeError('DIRECTION_CANDLE_AUDIT_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds'])
    sys.setprofile(no_native_models if synthetic else no_queries_or_models)
    stage = out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        if synthetic:
            report = run_synthetic(stage, np=np, publish=publish)
            input_snapshot = control_snapshot
        else:
            with binding.fresh_file_verification_scope():
                for ref in plan['inputs'].values():
                    regular(ref['path']); binding.require_binding(ref, label='input')
                review = binding.read_bound_json(Path(plan['metadata_review']['path']), plan['metadata_review']['sha256'])
                manifest = json.loads(Path(plan['inputs']['native_manifest']['path']).read_text())
                metadata = json.loads(Path(plan['inputs']['target_metadata']['path']).read_text())
                require_metadata(manifest, metadata, plan['inputs'], review)
                saved = read_targets(plan['inputs']['price_targets']['path'], np=np)
                years = [read_year_quotes(plan['inputs']['native_' + str(y)]['path'], expected_rows=n,
                    year=y, parquet=parquet, arrow=arrow, np=np) for y, n, offset in YEAR_SPEC]
                source = {key: np.concatenate([part[key] for part in years]) for key in years[0]}
                source['source_row'] = np.concatenate([np.arange(offset, offset+n, dtype=np.int64) for y,n,offset in YEAR_SPEC])
                require(np.all(np.diff(source['time_ns']) > 0), 'COMBINED_CLOCK')
                entry = gather_quotes(source, saved['decision_ns'], cutoff_ns=CUTOFF, np=np)
                terminal = gather_quotes(source, saved['terminal_time_ns'], cutoff_ns=CUTOFF, np=np)
                require(entry['observed'].all() and np.array_equal(entry['source_row'], saved['dynamic_entry_m1_row']), 'ENTRY_SOURCE_ROWS')
                require(np.array_equal(terminal['observed'], saved['observed'])
                    and np.array_equal(terminal['source_row'], saved['terminal_source_row']), 'TERMINAL_SOURCE_ROWS_MASKS')
                result = compare_direction_native_midpoint(saved, entry_quotes=entry, terminal_quotes=terminal, cutoff_ns=CUTOFF)
                hashes = {role: {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    for key, value in arrays.items()} for role, arrays in
                    (('source', source), ('saved', saved), ('entry', entry), ('terminal', terminal), ('comparison', result['arrays']))}
                input_snapshot = binding.capture_verified_file_snapshot()
            report = {key: value for key, value in result.items() if key != 'arrays'}
            report.update(array_sha256=hashes, inputs=plan['inputs'], plan=plan_ref, source_head=args.source_head,
                native_year_scope=[{'year': y, 'rows': n, 'global_offset': offset} for y,n,offset in YEAR_SPEC],
                evidence_class='measured_on_bound_TRAIN_candles',
                boundary='Candle-open midpoint comparison only; no quote-event timing, model, generalization or economic evidence.')
        input_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(git_state() == initial, 'FINAL_HEAD_BRANCH_CLEAN')
        require(cap.require_capped_lock_ancestry() == lock, 'FINAL_LOCK_IDENTITY')
        publish(stage/'RESULT.json', report)
        publish(stage/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'source_head': args.source_head, 'source_unchanged': True, 'inputs_unchanged': True,
            'elapsed_seconds': time.monotonic()-started, 'test_data_used': False,
            'native_calls': 0, 'fits': 0, 'optimizer_updates': 0})
        publish_bundle_directory_noreplace(stage, out/'OUTPUT')
        publish(out/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'result_path': str(out/'OUTPUT/RESULT.json'), 'plan': plan_ref})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'error': type(exc).__name__ + ': ' + str(exc), 'elapsed_seconds': time.monotonic()-started,
            'partial_output_is_invalid': True, 'relaunch_allowed': False})
        raise
    finally:
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
