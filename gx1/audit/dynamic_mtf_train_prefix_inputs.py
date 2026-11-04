"""Verify the saved physical support for the registered dynamic-state TRAIN prefix."""
import ast
import hashlib
import stat
import subprocess
from pathlib import Path

from gx1.contracts import local_random_access_campaign_v2 as campaign


def _require(ok, reason):
    if not ok:
        raise ValueError('DYNAMIC_TRAIN_PREFIX_SUPPORT_' + reason)


def _read(ref):
    campaign.require_binding(ref, label='prefix metadata', verify_file=False)
    return campaign.read_bound_json(Path(ref['path']), ref['sha256'])


def _check_payload_stats(payloads):
    keys = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    for filename, item in payloads.items():
        path = Path(filename)
        observed = path.lstat()
        _require(path.is_absolute() and path.resolve() == path and stat.S_ISREG(observed.st_mode)
                 and item['stat'] == {key: getattr(observed, key) for key in keys}, 'PAYLOAD_IDENTITY:' + filename)


def _qualification_code(source):
    # The optional relative-GC fit was added after this support measurement.
    # Keep the whole qualification module, imports and helpers except the fitter.
    module = ast.parse(source)
    module.body = [node for node in module.body if not (
        isinstance(node, ast.FunctionDef) and node.name == 'fit_native_direction_prefix_normalization')]
    return ast.dump(module, include_attributes=False)


def _masked_blockers(report, current):
    _require(report['fit_allowed'] is False and report['statistics_fits'] == 0, 'FIT_ROLE')
    support = report['support']
    _require(len(support) == 12 and set(support) == set(current['support']), 'SURFACE_SET')
    missing = {}
    for surface, data in support.items():
        _require(all(data[key] == current['support'][surface][key] for key in
                     ('field_names', 'binary_fields', 'masking_pairs', 'categorical_domains')),
                 'FIELD_OR_MASK:' + surface)
        names, counts = data['field_names'], data['observed_counts']
        _require(len(names) == len(counts) == len(set(names)), 'SUPPORT_WIDTH:' + surface)
        masks = dict(data['masking_pairs'])
        for i, count in enumerate(counts):
            if count == 0:
                _require(i in masks and data['observed_min'][i] is data['observed_max'][i] is None
                         and data['nonzero_observed_counts'][i] == 0, 'MASKED_EMPTY:' + names[i])
                missing.setdefault(surface, []).append(names[i])
    rows = [(b['surface'], b['field'], b['reason']) for b in report['blockers']]
    expected = [(s, f, 'no_observed_prefix_values') for s, fields in missing.items() for f in fields]
    _require(sorted(rows) == sorted(expected) and set(missing) == {'signal', 'ctx_cont', 'gc_mtf_d1'},
             'BLOCKER_PARITY')
    fields = missing['gc_mtf_d1']
    _require(len(fields) == 51 and set(fields) == set(missing['ctx_cont'])
             and {'ctx_cont.' + field for field in fields} == set(missing['signal']), 'ALIAS_PARITY')
    return fields, {s: len(v) for s, v in missing.items()}


def verify_train_prefix_input_support(policy):
    """One metadata join. The enclosing runner owns admission and publication."""
    repo = Path(__file__).resolve().parents[2]
    gate = policy['current_work']['dynamic_mtf_train_gates_20261104']
    geom = gate['oof_geometry']
    raw = policy['dynamic_m5_direction_gc_boolean_physical_prefix_normalization_267']
    verified = policy['dynamic_mtf_state_prefix_normalization_20261101_001']
    historical = policy['dynamic_mtf_state_onepass_fit_20261103_001']
    _require(policy['training_enabled'] is gate['model_calls_authorized'] is False
             and gate['fit_parents'] == 29849 and gate['fit_cutoff_utc_exclusive'] == '2024-10-01T00:00:00Z',
             'REGISTERED_SCOPE')
    with campaign.file_verification_scope():
        recipe = _read(historical['recipe'])
        inputs = _read(recipe['input_manifest'])
        physical_plan = _read(raw['plan'])
        source_plan = _read(verified['plan'])
        _require(recipe['full_feature_profile'] is True and inputs['axes'] == geom['source']
                 and inputs['physical_provider_plan'] == source_plan['physical_provider_plan'] == raw['plan']
                 and inputs['physical_provider_worker'] == source_plan['physical_provider_worker'] == raw['worker']
                 and physical_plan['payloads'] == source_plan['payloads'] and len(physical_plan['payloads']) == 48,
                 'PHYSICAL_PROVIDER')
        payloads = physical_plan['payloads']
        _check_payload_stats(payloads)
        result = _read(raw['result'])
        terminal = _read(raw['terminal'])
        review = _read(raw['actual_qualification_review'])
        _require(terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0 and terminal['source_unchanged'] is True
                 and terminal['result'] == review['result'] == raw['result']
                 and terminal['plan'] == review['plan'] == result['plan'] == raw['plan']
                 and result['source_stat_bookends_unchanged'] is True and result['payloads'] == payloads
                 and result['qualifications'] == review['qualifications'] == raw['qualifications'], 'SAVED_TERMINAL')
        matches = [q for q in result['qualifications'] if q['prefix_parent_rows'] == gate['fit_parents']
                   and q['cutoff_ns'] == 1727740800000000000]
        _require(len(matches) == 1 and matches[0]['fit_allowed'] is False
                 and matches[0]['blocker_count'] == review['excluded_fold0_missing_observation_blockers'] == 153,
                 'EXCLUDED_PREFIX')
        prior = _read(matches[0]['report'])
        _require(inputs['prefix_qualification'] == geom['prior_prefix_qualification'], 'CURRENT_QUALIFICATION')
        current = _read(inputs['prefix_qualification'])
        proof, comparison = prior['proof'], current['proof']
        block, prefix = geom['blocks'][0], proof['prefix']
        _require(block['role'] == 'fit' and block['parent_range'] == [0, gate['fit_parents']]
                 and block['parents'] == prefix['rows'] == gate['fit_parents']
                 and prefix['cutoff_ns'] == matches[0]['cutoff_ns']
                 and prefix['decision_ns_sha256'] == block['issue_ns_sha256_le_i64']
                 and prefix['bar_start_ns_sha256'] == block['bar_start_ns_sha256_le_i64']
                 and prefix['parent_rows_sha256'] == block['parent_rows_sha256_le_i64']
                 and proof['all_physical_fields_preserved'] is comparison['all_physical_fields_preserved'] is True
                 and all(proof[key] == comparison[key] for key in ('base_lineage', 'full_axes', 'window_rows'))
                 and proof['full_axes']['source'] == geom['source'], 'PREFIX_IDENTITY')
        original_sources = proof['source_bindings']
        active_sources = comparison['source_bindings']
        _require(original_sources['current_plan'] == raw['plan'] and active_sources['current_plan'] == verified['plan']
                 and original_sources['current_source_owners'] == physical_plan['source_owners']
                 and active_sources['current_source_owners'] == source_plan['source_owners']
                 and original_sources['ancestral264_input_provenance'] == active_sources['ancestral264_input_provenance'],
                 'BOUND_SOURCES')
        for name in ('current_gc_layout', 'current_gc_raw_transition'):
            for sources in (original_sources, active_sources):
                campaign.require_binding(sources[name], label=name)
            _require(original_sources[name]['sha256'] == active_sources[name]['sha256'], 'RAW_GC:' + name)
        # Verify current physical owners. The campaign/admission implementation is outside this raw support claim.
        for name in physical_plan['source_owners']:
            ref = source_plan['source_owners'][name]
            _require(ref == recipe['source_owners'][name] and ref['path'] == str(repo / name), 'OWNER:' + name)
            campaign.require_binding(ref, label=name)
        name = 'gx1/contracts/entry_native_direction_prefix_normalization_v1.py'
        original = subprocess.check_output(['git', 'show', original_sources['current_source_head'] + ':' + name], cwd=repo)
        _require(hashlib.sha256(original).hexdigest() == physical_plan['source_owners'][name]['sha256']
                 and _qualification_code(original) == _qualification_code((repo / name).read_bytes()),
                 'QUALIFICATION_OWNER_CHANGED')
        fields, counts = _masked_blockers(prior, current)
        _check_payload_stats(payloads)
        campaign.capture_verified_file_snapshot()
    return {'status': 'BLOCK', 'fit_allowed': False, 'statistics_fits': 0, 'normalization': None,
            'reason': 'no_observed_prefix_values', 'source': 'saved_canonical_qualification_verified_by_metadata_join',
            'registered_fit_prefix': prefix, 'qualifier': matches[0]['report'], 'actual_review': raw['actual_qualification_review'],
            'terminal': raw['terminal'], 'physical_result': raw['result'], 'physical_plan': raw['plan'],
            'input_manifest': recipe['input_manifest'], 'comparison_qualification': inputs['prefix_qualification'],
            'qualification_owner': source_plan['source_owners'][name], 'gc_layout': active_sources['current_gc_layout'],
            'gc_raw_transition': active_sources['current_gc_raw_transition'], 'physical_surfaces': len(prior['support']),
            'physical_payload_stat_bookends': len(payloads), 'blockers': sum(counts.values()),
            'blocked_coordinates_by_surface': counts, 'population_rows_by_blocked_surface': {
                surface: prior['support'][surface]['population_rows'] for surface in counts},
            'unique_gc_d1_fields': fields, 'decoded_feature_or_target_arrays': 0, 'model_calls': 0}


def review_train_gc_d1_donor(terminal_ref, cutoff_ns):
    """Measure support at the exact D1-close snapshot of the bound 024 GC donor."""
    import numpy as np
    import pyarrow.parquet as pq

    terminal = _read(terminal_ref)
    _require(terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
             and terminal['test_data_used'] is False, 'GC_DONOR_TERMINAL')
    root = Path(terminal_ref['path']).parent
    refs = {}

    def frame(name, axis, columns):
        item = terminal['artifacts'][name]
        source = {'path': str(root / name), 'sha256': item['sha256']}
        refs[name] = campaign.require_binding(source, label=name)
        _require((root / name).stat().st_size == item['bytes'], 'DONOR_SIZE')
        return pq.read_table(source['path'], columns=[axis, *columns],
                             filters=[(axis, '<', cutoff_ns)], use_threads=False).to_pandas().set_index(axis)

    profile = ['observed_profile_' + name + '_valid'
               for name in ('rolling15', 'rolling60', 'anchored_segment', 'session_observed', 'session_complete')]
    daily = frame('TRADE_FEATURES_D1.parquet', 'available_ns', [
        'observed_profile_snapshot_available_ns', 'observed_profile_source_coverage_fraction',
        'observed_proxy_bos_down', 'observed_proxy_ob_bear_state_valid', *profile])
    minute = frame('M1_GC_INPUTS.parquet', 'bar_start_ns', ['records', 'trades', 'volume'])
    times = daily.index.to_numpy(dtype=np.int64)
    one_minute = 60_000_000_000
    _require(len(times) and np.all(np.diff(times) > 0) and minute.index.is_unique
             and np.array_equal(daily['observed_profile_snapshot_available_ns'].to_numpy(), times)
             and np.all(times % (1440 * one_minute) == 1320 * one_minute), 'GC_SNAPSHOT_CLOCK')

    def summarize(first_minute):
        positions = times[:, None] + one_minute * np.arange(first_minute, first_minute + 5)
        _require(np.all(positions < cutoff_ns), 'GC_MINUTE_BOUNDARY')
        block = minute.loc[positions.reshape(-1)]
        trades = block['trades'].to_numpy().reshape(len(times), 5)
        return {'minutes': len(block), 'records': int(block['records'].sum()),
                'trades': int(trades.sum()), 'volume': int(block['volume'].sum()),
                'dates_with_trades': int((trades > 0).any(axis=1).sum())}

    return {'terminal': terminal_ref, 'artifacts': refs, 'cutoff_ns': cutoff_ns,
            'daily_source_rows': len(times), 'snapshot_hour_utc': 22,
            'snapshot_m5': summarize(-5), 'same_dates_one_hour_earlier': summarize(-65),
            'same_dates_next_five_minutes_diagnostic_only': summarize(0),
            'profile_valid_counts': {col: int(daily[col].sum()) for col in profile},
            'profile_source_coverage_zero': int((daily['observed_profile_source_coverage_fraction'] == 0).sum()),
            'bear_bos_known': int(daily['observed_proxy_bos_down'].notna().sum()),
            'bear_bos_down': int((daily['observed_proxy_bos_down'] == 1).sum()),
            'bear_state_observed': int(daily['observed_proxy_ob_bear_state_valid'].sum()),
            'historical_exchange_calendar_claimed': False}
