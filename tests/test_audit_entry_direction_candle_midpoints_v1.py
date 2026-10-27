"""Synthetic admission, exact-reader and I/O boundary checks; no market input."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as audit


@pytest.fixture
def admission(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    keys = ('native_2024', 'native_2025', 'native_manifest', 'price_targets', 'target_metadata')
    refs = {key: {'path': str(tmp_path/(key+'.json')), 'sha256': 'a'*64} for key in keys}
    design = {'input_references_from_existing_verified_metadata': refs,
        'phase_A': {'scope': {'cutoff_exclusive_utc': '2026-01-01T00:00:00Z',
            'decision': 'M5 bar_start + 300 seconds', 'horizons_minutes': [1, 5, 15],
            'input_columns': list(audit.COLUMNS), 'parents': audit.PARENTS, 'year_files': [2024, 2025]},
            'expected_existing_counts': copy.deepcopy(audit.COUNTS)}}
    plan = {'schema_version': audit.SCHEMA, 'mode': 'actual',
        'design': {'path': str(tmp_path/'DESIGN.json'), 'sha256': audit.DESIGN_SHA},
        'metadata_review': {'path': str(tmp_path/'INPUT_METADATA_REVIEW.json'), 'sha256': 'b'*64},
        'inputs': copy.deepcopy(refs), 'sources': {name: {'path': str(audit.REPO/name), 'sha256': 'c'*64}
            for name in audit.SOURCE_PATHS}, 'expected_counts': copy.deepcopy(audit.COUNTS),
        'out_dir': str(tmp_path/'actual'), 'wall_seconds': 180}
    return plan, design


def test_exact_actual_plan(admission):
    plan, design = admission
    assert audit.require_plan(plan, design) == Path(plan['out_dir'])


@pytest.mark.parametrize('fault', ['2026', 'extra_input', 'missing_source', 'other_source', 'wall', 'counts', 'outside_output'])
def test_actual_admission_rejects_scope_drift(admission, fault):
    plan, design = admission
    if fault == '2026': plan['inputs']['native_2025']['path'] += '/year=2026'
    elif fault == 'extra_input': plan['inputs']['extra'] = plan['inputs']['native_2024']
    elif fault == 'missing_source': plan['sources'].pop(audit.SOURCE_PATHS[0])
    elif fault == 'other_source': plan['sources'][audit.SOURCE_PATHS[0]]['path'] += '.other'
    elif fault == 'wall': plan['wall_seconds'] += 1
    elif fault == 'counts': plan['expected_counts']['missing'][0] += 1
    else: plan['out_dir'] = str(Path(plan['out_dir']).parent.parent/'escaped')
    with pytest.raises(RuntimeError, match='DIRECTION_CANDLE_AUDIT_'):
        audit.require_plan(plan, design)


def test_synthetic_plan_has_no_payload_or_optional_test_selection(admission):
    plan, design = admission
    plan.update(schema_version=audit.SCHEMA+'_synthetic', mode='synthetic', inputs={}, tests=list(audit.TEST_FILES))
    plan.pop('expected_counts')
    for name in audit.TEST_FILES:
        plan['sources'][name] = {'path': str(audit.REPO/name), 'sha256': 'd'*64}
    assert audit.require_plan(plan, design) == Path(plan['out_dir'])
    plan['tests'] = plan['tests'][:2]
    with pytest.raises(RuntimeError, match='COUNTS_OR_WALL'):
        audit.require_plan(plan, design)


def test_exact_quote_join_keeps_missing_and_cutoff_separate():
    source = {'time_ns': np.array([60, 120, 180, 300], np.int64),
        'source_row': np.array([5076522, 5076523, 5076524, 5076525], np.int64),
        **{key: np.array([100, 101, 102, 104], np.float64) for key in audit.COLUMNS[1:]}}
    wanted = np.array([[60, 240, 300], [120, 180, 360]], np.int64)
    original = {key: value.copy() for key, value in source.items()}
    result = audit.gather_quotes(source, wanted, cutoff_ns=300, np=np)
    assert result['observed'].tolist() == [[True, False, False], [True, True, False]]
    assert result['source_row'].tolist() == [[5076522, -1, -1], [5076523, 5076524, -1]]
    assert np.all(result['time_ns'][~result['observed']] == 0)
    for name in audit.COLUMNS[1:]: assert np.all(result[name][~result['observed']] == 0)
    for name in source: np.testing.assert_array_equal(source[name], original[name])


def test_quote_join_does_not_use_nearest_older_quote():
    source = {'time_ns': np.array([60, 180], np.int64), 'source_row': np.array([8, 9], np.int64),
        **{key: np.ones(2, np.float64) for key in audit.COLUMNS[1:]}}
    result = audit.gather_quotes(source, np.array([120], np.int64), cutoff_ns=300, np=np)
    assert not result['observed'].any() and result['source_row'][0] == -1


def test_reader_uses_only_four_columns_and_preserves_double_prices(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path/'synthetic.parquet'
    times = np.array(['2024-01-02T00:00', '2024-01-02T00:01'], 'datetime64[ns]')
    pq.write_table(pa.table({'time': times, 'open': [100.123456789, 101.25],
        'bid_open': [100., 101.], 'ask_open': [100.5, 101.5], 'forbidden_extra': [1, 2]}), path)
    calls = []
    def reader(handle, **kwargs):
        calls.append(kwargs)
        return pq.read_table(handle, **kwargs)
    actual = audit.read_year_quotes(path, expected_rows=2, year=2024,
        parquet=SimpleNamespace(read_table=reader), arrow=pa, np=np)
    assert calls == [{'columns': list(audit.COLUMNS), 'use_threads': False}]
    assert actual['open'].dtype == np.float64 and actual['open'][0] == 100.123456789


@pytest.mark.parametrize('fault', ['year', 'rows', 'crossed', 'null'])
def test_reader_rejects_wrong_clock_or_quote_domain(tmp_path, fault):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path/'synthetic.parquet'
    values = {'time': np.array(['2024-01-02T00:00'], 'datetime64[ns]'), 'open': [100.],
        'bid_open': [99.], 'ask_open': [101.]}
    if fault == 'year': values['time'] = np.array(['2026-01-02T00:00'], 'datetime64[ns]')
    elif fault == 'crossed': values['bid_open'] = [102.]
    elif fault == 'null': values['open'] = pa.array([None], type=pa.float64())
    pq.write_table(pa.table(values), path)
    with pytest.raises(RuntimeError, match='YEAR_'):
        audit.read_year_quotes(path, expected_rows=2 if fault == 'rows' else 1,
            year=2024, parquet=pq, arrow=pa, np=np)


@pytest.mark.parametrize('name', ['torch', 'sklearn', 'joblib'])
def test_genuine_guard_blocks_model_imports(tmp_path, name):
    guard = audit.make_guard(tmp_path, [], [])
    with pytest.raises(RuntimeError, match='MODEL_IMPORT'):
        guard('import', (name,))


def test_guards_separate_synthetic_fixture_payload_from_genuine_data(tmp_path):
    fake = str(tmp_path/'fixture.npz')
    flags = os.O_WRONLY | os.O_CREAT
    audit.make_guard(tmp_path, [], [], synthetic=True)('open', (fake, 'wb', flags))
    with pytest.raises(RuntimeError, match='PAYLOAD'):
        audit.make_guard(tmp_path, [], [])('open', (fake, 'wb', flags))
    with pytest.raises(RuntimeError, match='UNDECLARED_DATA'):
        audit.make_guard(tmp_path, [], [], synthetic=True)('open', ('/home/andre2/GX1_DATA/fake.parquet', 'rb', 0))


def test_guard_permits_only_readonly_fixed_git_commands(tmp_path):
    guard = audit.make_guard(tmp_path, [], [])
    prefix = ['git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null', '-C', str(audit.REPO)]
    guard('subprocess.Popen', ('git', prefix+['rev-parse', 'HEAD'], None, {}))
    with pytest.raises(RuntimeError, match='PROCESS'):
        guard('subprocess.Popen', ('git', prefix+['reset', '--hard'], None, {}))


def test_guards_reject_outside_writes_and_network(tmp_path):
    guard = audit.make_guard(tmp_path, [], [])
    with pytest.raises(RuntimeError, match='OUTSIDE_OUTPUT_WRITE'):
        guard('open', (str(tmp_path.parent/'outside.json'), 'w', os.O_WRONLY))
    with pytest.raises(RuntimeError, match='NETWORK_PROCESS_PICKLE'):
        guard('socket.connect', ())


def test_source_path_symlink_is_not_admitted(tmp_path):
    path = tmp_path/'real.json'; path.write_text('{}')
    link = tmp_path/'alias.json'
    # The execution guard forbids symlink creation; inject the already-known
    # symlink result without making one in the synthetic qualification process.
    with pytest.MonkeyPatch.context() as patch:
        old = Path.resolve
        patch.setattr(Path, 'resolve', lambda self, **kw: path if self == link else old(self, **kw))
        with pytest.raises(RuntimeError, match='CANONICAL_REGULAR_FILE'):
            audit.regular(link)


def test_native_model_and_cuda_calls_are_rejected():
    frame = SimpleNamespace(f_code=SimpleNamespace(co_filename='/tmp/gx1/models/example.py', co_name='__init__'))
    with pytest.raises(RuntimeError, match='SYNTHETIC_NATIVE_MODEL_CALL'):
        audit.no_native_models(frame, 'call', None)
    with pytest.raises(RuntimeError, match='SYNTHETIC_CUDA'):
        audit.no_native_models(frame, 'c_call', SimpleNamespace(__name__='_cuda_init'))


def _metadata(admission):
    plan, design = admission
    years = {'year='+str(year): {'rows': rows} for year,rows,offset in audit.YEAR_SPEC}
    proof = {'method': 'exact_quote_complete_pretest_rows_are_native_m1_subset_v1',
        'proof_sha256': audit.SUBSET_SHA, 'years': years}
    authority = {'native_m1_manifest_path': plan['inputs']['native_manifest']['path'],
        'native_m1_manifest_sha256': plan['inputs']['native_manifest']['sha256'],
        'base28_native_m1_subset_proof': proof, 'native_m1_canonical_rows_sha256': 'e'*64,
        'native_m1_source_chunks_sha256': 'f'*64}
    metadata = {'schema_version': 'gx1_direction_baseline_data_metadata_v1', 'parent_rows': audit.PARENTS,
        'horizons_minutes': [1,5,15], 'full_parent_order_retained': True, 'test_data_used': False,
        'price_targets': plan['inputs']['price_targets'],
        'target_report': {'canonical_source_metadata': {'source_quote_binding': {'authority': authority}}}}
    manifest = {'timestamp_semantics': 'bar_start_utc', 'source_kind': 'oanda_native_mba_candles',
        'completion_field': 'complete', 'completion_value': True, 'year_rows': {'year=2023': 5076522,
        'year=2024': 354866, 'year=2025': 353499},
        'year_sha256': {label: plan['inputs']['native_'+label[-4:]]['sha256'] for label in years},
        'canonical_rows_sha256': 'e'*64, 'source_chunks_sha256': 'f'*64}
    return manifest, metadata, plan['inputs'], {'source_subset_proof_sha256': audit.SUBSET_SHA}


def test_metadata_proof_binds_native_manifest_years_and_global_offsets(admission):
    audit.require_metadata(*_metadata(admission))


@pytest.mark.parametrize('fault', ['proof', 'year_hash', 'offset', 'canonical_rows'])
def test_metadata_proof_drift_stops_before_row_mapping(admission, fault):
    manifest, metadata, inputs, review = _metadata(admission)
    if fault == 'proof': review['source_subset_proof_sha256'] = 'a'*64
    elif fault == 'year_hash': manifest['year_sha256']['year=2024'] = 'b'*64
    elif fault == 'offset': manifest['year_rows']['year=2023'] += 1
    else: manifest['canonical_rows_sha256'] = 'a'*64
    with pytest.raises(RuntimeError, match='NATIVE_SUBSET_PROOF|YEAR_ROWS_HASH_OFFSET'):
        audit.require_metadata(manifest, metadata, inputs, review)


@pytest.mark.parametrize('fault', ['dtype', 'terminal_clock'])
def test_target_reader_rejects_archive_dtype_or_elapsed_clock(tmp_path, monkeypatch, fault):
    count = 2
    monkeypatch.setattr(audit, 'PARENTS', count)
    monkeypatch.setattr(audit, 'COUNTS', {'observed': [count]*3, 'missing': [0]*3, 'observed_zero': [count]*3})
    decisions = np.asarray([1735733100000000000, 1735733400000000000], np.int64)
    saved = {name: np.zeros((count,) if name in ('parent_rows', 'bar_start_ns', 'dynamic_entry_m1_row',
        'decision_ns') else (count,3), dtype) for name,dtype in audit.TARGET_DTYPES.items()}
    saved.update(parent_rows=np.arange(count, dtype=np.int64), decision_ns=decisions,
        bar_start_ns=decisions-300*10**9,
        terminal_time_ns=decisions[:,None]+np.asarray([1,5,15], np.int64)[None,:]*60*10**9,
        observed=np.ones((count,3), bool))
    if fault == 'dtype': saved['sign'] = saved['sign'].astype(np.int64)
    else: saved['terminal_time_ns'][0,0] += 1
    path = tmp_path/'synthetic_targets.npz'; np.savez(path, **saved)
    with pytest.raises(RuntimeError, match='TARGET_DTYPE_SHAPE|PARENT_DECISION_CLOCK'):
        audit.read_targets(path, np=np)
