"""Data-route isolation for the single registered masked December dynamic-state experiment."""
import hashlib
import mmap
import resource
from pathlib import Path

import numpy as np

from gx1.audit.dynamic_mtf_december_normalization import _history_views
from gx1.audit.dynamic_mtf_train_prefix_inputs import _check_payload_stats
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.models.entry_v10.entry_v10_input_normalization import MODEL_NATIVE_SEQ_LEN


def _require(value, reason):
    if not value:
        raise RuntimeError('DECEMBER_NATIVE_INPUT_' + reason)


class _PageRelease:
    @staticmethod
    def release_pages(ancestor, prepared, discard):
        for a in prepared['december_mmaps']:
            handle = getattr(a, '_mmap', None)
            if handle is not None and hasattr(handle, 'madvise'):
                handle.madvise(mmap.MADV_DONTNEED)


def _role(launch):
    from gx1.contracts import entry_dynamic_mtf_december_native_v1 as native
    expected = {native.PRIVATE_SCOPE: 'fit', native.FIT_SCOPE: 'fit', native.SCORE_SCOPE: 'score'}
    mode = expected.get(launch['run_scope'])
    _require(mode in ('fit', 'score') and launch['input_role'] == mode, 'ROLE_SCOPE')
    return mode


def _read_xau_cache_manifest(raw, metadata):
    """Read only the V4 manifest whose exact file identity was pinned by the physical input plan."""
    lineage = raw['base_lineage']
    path = Path(lineage['mtf_cache_manifest_path'])
    _require(path == Path(raw['native_cache_dir']) / 'manifest.json', 'XAU_CACHE_MANIFEST_PATH')
    ref = {'path': str(path), 'sha256': lineage['mtf_cache_manifest_sha256']}
    item = raw['payloads'][str(path)]
    _require(item['sha256'] == ref['sha256'] and item['verification'] == 'canonical_fresh_hash',
             'XAU_CACHE_MANIFEST_ROLE')
    inventory = {str(path): item}
    _check_payload_stats(inventory)
    value = campaign.read_bound_json(path, ref['sha256'])
    _check_payload_stats(inventory)
    metadata[str(path)] = {'ref': ref, 'stat': dict(item['stat'])}
    return value


def december_native_metadata(launch, common):
    """Verify small, recursively bound metadata and derive the exact role-specific data-open set."""
    role = _role(launch)
    metadata = {}

    def read(ref):
        return common._read(ref, metadata)

    candidate = read(launch['candidate_plan'])
    bind = candidate['bindings']
    _require(launch['feature_view'] == bind['feature_view'] and launch['prefix_normalization'] == bind['prefix_normalization']
             and launch['prefix_prior'] == bind['prefix_prior'] and launch['label_audit'] == bind['prefix_counts'],
             'REGISTERED_INPUT_BINDINGS')
    if role == 'fit':
        _require(launch['target_manifest'] == bind['fit_target_view'] and 'scoring_binding' not in launch,
                 'FIT_TARGET_BEFORE_OPEN')
        scoring = None
    else:
        scoring = read(launch['scoring_binding'])
        _require(launch['target_manifest'] == scoring['target_manifest'] and launch['target_manifest'] != bind['fit_target_view']
                 and scoring['candidate_plan'] == launch['candidate_plan']
                 and scoring['fit_checkpoint'] == launch['score_origin']['fit_checkpoint']
                 and scoring['storage_plan'] == launch['score_storage_plan'], 'SCORE_TARGET_BEFORE_OPEN')
        clock_role = read(scoring['calendar_authority'])
        _require(clock_role['candidate_plan'] == launch['candidate_plan'] and clock_role['fit_origin'] == launch['score_origin']
                 and clock_role['role'] == 'score' and clock_role['target_manifest'] == launch['target_manifest'],
                 'SCORE_CALENDAR_PROVENANCE')
        read(clock_role['qualified_calendar']['closure_authority'])
    view, fitted = read(launch['feature_view']), read(launch['prefix_normalization'])
    views._require_view(view)
    _require(view['contract_sha256'] == candidate['feature_view_contract_sha256']
             and fitted['prefix_qualification']['proof']['prefix'] == view['prefix']
             and fitted['zero_support_profile'] == view['zero_support_profile']
             and view['native_endpoint_loss'] is True and view['full_feature_profile'] is True
             and view['dynamic_state_supervision'] is True and candidate['observer_axes'] == view['reserved_observer_axes'],
             'FROZEN_VIEW')
    physical = fitted['prefix_qualification']['proof']['source_bindings']
    _require(launch['input_manifest'] == physical['input_manifest'], 'HISTORICAL_PROVIDER_METADATA_ONLY')
    manifest = read(launch['input_manifest'])
    raw = read(physical['ancestral264_plan'])
    source = read(raw['m5_feature_manifest'])
    audit = read(physical['source_reconstruction'])
    reuse = read(physical['decode_reuse'])
    normalization_result = read(bind['normalization_result'])
    _require(normalization_result['feature_view'] == launch['feature_view']
             and normalization_result['normalization'] == launch['prefix_normalization']
             and normalization_result['source_summary']['clock_and_mmap_bindings'] == physical
             and raw['source_reconstruction'] == physical['source_reconstruction']
             and raw['decode_reuse_binding'] == physical['decode_reuse']
             and physical['current_gc_layout'] == manifest['view_source_bindings']['raw_gc_layout']
             and physical['current_gc_raw_transition'] == manifest['view_source_bindings']['raw_gc_transition']
             and audit['feature_surface_manifest_sha256'] == raw['m5_feature_manifest']['sha256']
             and audit['feature_surface_path'] == source['output_parquet']
             and audit['feature_surface_sha256'] == source['output_parquet_sha256']
             and source['decision'] == audit['decision'] == 'PASS' and all(audit['checks'].values()), 'STEP2_RAW_LINEAGE')
    layout = read(physical['current_gc_layout'])
    transition = read(physical['current_gc_raw_transition'])
    gc = read(raw['gc_surfaces'])
    line = raw['base_lineage']
    xau = _read_xau_cache_manifest(raw, metadata)
    _require(line == view['base_lineage'] and transition['binding'] == raw['gc_surfaces']
             and transition['current_layout_sha256'] == layout['contract_sha256']
             and transition['array_descriptors'] == gc['surfaces'] and not transition['normalization_applied'],
             'RAW_HISTORY_LINEAGE')
    # Only the four actually consumed immutable decode files are mapped. The other two decode files
    # and the 245 parent/price archive have no read role in the native job.
    decode_root = Path(reuse['source_worker']['path']).parent / 'SUPPORT_SCRATCH'
    scratch = {name: str(decode_root / group / (name + '.mmap')) for group, name in
               [('source050', 'signal'), ('current051', 'snap'), ('current051', 'ctx_cont'), ('current051', 'ctx_cat')]}
    files = {source['output_parquet']: raw['payloads'][source['output_parquet']]}
    _require(files[source['output_parquet']]['sha256'] == source['output_parquet_sha256'], 'CLOCK_PARQUET_BINDING')
    for path in scratch.values():
        item = raw['payloads'][path]
        _require(item['stat'] == reuse['scratch_files'][path] and not item['stat']['st_mode'] & 0o222
                 and item['sha256'] is None, 'READ_ONLY_DECODE')
        files[path] = item
    for tf in target.TIMEFRAMES:
        info = xau['tfs'][tf]
        desc = gc['surfaces'][tf]
        refs = [(d['path'], d['sha256']) for d in desc.values()]
        for name in ('ts_npy', 'feats_npy'):
            refs.append((str(Path(raw['native_cache_dir']) / info[name]), info[name + '_sha256']))
        for path, digest in refs:
            _require(raw['payloads'][path]['sha256'] == digest, 'HISTORY_SOURCE:' + tf)
            files[path] = raw['payloads'][path]
    _check_payload_stats(files)
    meta = read(launch['target_manifest'])
    scope = view['reserved_observer_axes']['fit' if role == 'fit' else 'held_out']
    _require(meta['metadata']['parents'] == scope['rows'] and meta['metadata']['cutoff_ns'] == scope['cutoff_ns']
             and meta['metadata']['source_refs'] == launch['target_sources'] and meta['metadata']['test_data_used'] is False
             and set(meta['arrays']) == set(target.SERIES_ARRAY_DTYPES), 'PHYSICAL_TARGET_ROLE')
    refs, stats = {}, {}
    for name, desc in meta['arrays'].items():
        _require(desc['path'] == str(Path(launch['target_manifest']['path']).parent / (name + '.npy')), 'TARGET_LEAF')
        refs[name] = {'path': desc['path'], 'sha256': desc['sha256']}
        stats[name] = common._stat(desc['path'])
        _require(stats[name]['st_size'] == desc['size_bytes'], 'TARGET_SIZE:' + name)
    prior, counts = read(launch['prefix_prior']), read(launch['label_audit'])
    _require(prior['parents'] == counts['prefix_rows'] == view['prefix']['rows']
             and prior['counts'] == counts['class_counts'] and prior['transition_counts'] == counts['transition_counts'],
             'FROZEN_FIT_PRIOR')
    constructor = read(manifest['constructor_metadata'])
    read_files = {**files, **{r['path']: {'sha256': r['sha256'], 'stat': stats[k], 'verification': 'role_target'}
                              for k, r in refs.items()}}
    return {'december_role': role, 'december_target_ref': launch['target_manifest'], 'dynamic_state_launch': launch, 'state_target_manifest': meta,
            'state_array_refs': refs, 'state_array_stats': stats, 'december_read_files': read_files,
            'ancestor_plan': {**raw, 'payloads': files}, 'physical': _PageRelease, 'ancestor': None,
            'info': {'metadata': metadata, 'audit': audit}, 'manifest': manifest,
            'fitted': fitted, 'constructor': constructor, 'state_label_audit': {'prior': prior},
            'december_sources': {'scratch': scratch, 'source': source, 'audit': audit, 'physical': physical,
                                'step2': normalization_result['source_summary'], 'view': view, 'candidate': candidate,
                                'scoring': scoring}}


def _role_clock(path, *, first, last):
    """Read only the timestamp column in the closed historical interval, never nested market tensors."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    dt = pq.read_schema(path).field('time').type
    _require(pa.types.is_timestamp(dt) and dt.unit == 'ns', 'FEATURE_CLOCK_UNIT')
    filters = [('time', '>=', pa.scalar(int(first), type=dt)), ('time', '<=', pa.scalar(int(last), type=dt))]
    col = pq.read_table(path, columns=['time'], filters=filters, use_threads=False)['time'].combine_chunks()
    _require(col.null_count == 0, 'FEATURE_CLOCK_NULL')
    return col.cast(pa.int64()).to_numpy(zero_copy_only=False)


def _map_role(path, stat, *, begin, end, total_rows, width, dtype):
    typ = np.dtype(dtype)
    step = width * typ.itemsize
    _require(0 <= begin < end <= total_rows and stat['st_size'] == total_rows * step
             and not stat['st_mode'] & 0o222, 'PHYSICAL_SLICE')
    return np.memmap(path, dtype=typ, offset=begin * step, shape=(end - begin, width), mode='r')


def prepare_december_native(context, launch, np):
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    role = _role(launch)
    _require(context['december_role'] == role and context['dynamic_state_launch'] == launch, 'ROLE_CONTEXT')
    raw = context['ancestor_plan']
    src = context['december_sources']
    view = src['view']
    scope = view['reserved_observer_axes']['fit' if role == 'fit' else 'held_out']
    with campaign.file_verification_scope():
        series = target.load_dynamic_mtf_state_series(launch['target_manifest'], expected_source_refs=launch['target_sources'],
                                                      expected_cutoff_ns=scope['cutoff_ns'])
        issues = series.arrays['issue_ns']
        axes = views.NativeDirectionObserverAxes(view, source=view['reserved_observer_axes']['source'],
                                                parent_rows=series.arrays['parent_rows'], bar_start_ns=issues - views.BAR_NS,
                                                decision_ns=issues, role=role)
        first_position = 0 if role == 'fit' else view['prefix']['rows']
        n = len(issues)
        source = src['source']
        # Use only clocks here. The signed 050 table starts before the first TRAIN parent and is
        # the same immutable backing as the step-2 reconstruction. No legacy target archive is opened.
        clock = _role_clock(source['output_parquet'], first=0, last=int(axes.bar_start_ns[-1]))
        positions = np.searchsorted(clock, axes.bar_start_ns)
        _require(len(clock) <= source['rows'] and np.all(positions < len(clock))
                 and np.all(np.diff(clock)>0) and np.all(clock % views.BAR_NS == 0) and positions[-1]+1 == len(clock)
                 and np.array_equal(clock[positions], axes.bar_start_ns) and positions[0] >= MODEL_NATIVE_SEQ_LEN - 1,
                 'PHYSICAL_SOURCE_CLOCK')
        if role == 'fit':
            _require(len(clock) == src['step2']['source050_rows_through_last_issue'], 'STEP2_SOURCE_CLOCK')
        left = int(positions[0]) - MODEL_NATIVE_SEQ_LEN + 1
        stop = int(positions[-1]) + 1
        local_clock = clock[left:stop].copy()
        local_positions = positions - left
        local_clock.setflags(write=False)
        local_positions.setflags(write=False)
        scratch = src['scratch']
        source_total = source['rows']
        parent_total = src['audit']['rows']
        signal = _map_role(scratch['signal'], raw['payloads'][scratch['signal']]['stat'], begin=left, end=stop,
                           total_rows=source_total, width=source['signal_dim'], dtype='float32')
        current = {}
        for name, width, dtype in (('snap', source['signal_dim'], 'float32'),
                                   ('ctx_cont', source['ctx_cont_dim'], 'float32'), ('ctx_cat', source['ctx_cat_dim'], 'int64')):
            current[name] = _map_role(scratch[name], raw['payloads'][scratch[name]]['stat'], begin=first_position,
                                      end=first_position+n, total_rows=parent_total, width=width, dtype=dtype)
        mtf, gc, layout = _history_views(raw, src['physical']['current_gc_raw_transition'],
                                        src['physical']['current_gc_layout'], int(issues[-1]), native_data_role=True,
                                        fit_row_limits=src['step2'] if role == 'fit' else None)
        if role == 'fit':
            _require({k:len(v) for k,v in mtf.items()} == src['step2']['xau_raw_rows_through_last_issue']
                     and {k:len(v['available_ns']) for k,v in gc.items()} == src['step2']['gc_raw_rows_through_last_issue'],
                     'STEP2_HISTORY_CLOCKS')
        _check_payload_stats(raw['payloads'])
        campaign.capture_verified_file_snapshot()
    fit_axes = axes.parent_rows if role == 'fit' else np.arange(view['prefix']['rows'], dtype=np.int64)
    _require(hashlib.sha256(fit_axes.astype('<i8', copy=False).tobytes()).hexdigest() == view['prefix']['parent_rows_sha256'],
             'CHECKPOINT_FIT_ORDER_SOURCE')
    order = worker._fixed_order(fit_axes, len(fit_axes), worker.SEED, np)
    prepared = {'arrays': {k:getattr(axes,k) for k in ('parent_rows','bar_start_ns','decision_ns')},
                'observer_axes': axes, 'current': current, 'cache': mtf, 'gc': {'surfaces': gc,'layout':layout},
                'entry_sequence_source': {'signal':signal,'times_ns':local_clock,'sample_positions':local_positions,
                                          'audit':src['audit']}, 'state_artifact_directory':context['state_artifact_directory'],
                'december_mmaps': [signal,*current.values()], 'december_role':role,
                'physical_source_rows': [left,stop]}
    targets = {'dynamic_state_series':series, 'target_manifest':launch['target_manifest'],
               'target_sources':launch['target_sources'], 'prior':context['state_label_audit']['prior'],
               'lead_minutes':np.asarray(view['research_objective']['lead_minutes'],np.int64),
               'train_cutoff_ns':view['prefix']['cutoff_ns'], 'cutoff_ns':scope['cutoff_ns']}
    if role == 'score':
        prepared['december_score_parent_rows'] = axes.parent_rows
        prepared['december_score_inputs'] = _score_inputs(context, launch, series)
    return prepared, targets, view, None, order

def _score_inputs(context, launch, series):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    s = context['december_sources']['scoring']
    cal = common._read(s['calendar_authority'], context['info']['metadata'])
    _require(cal['role'] == 'score' and cal['target_manifest'] == launch['target_manifest']
             and cal['candidate_plan'] == launch['candidate_plan'], 'SCORE_CALENDAR_ROLE')
    _require(cal['qualified_calendar'] == series.metadata['score_lineage']['calendar_binding']
             and series.metadata['score_lineage']['fit_origin'] == launch['score_origin'], 'SCORE_SERIES_CALENDAR')
    closure = common._read(cal['qualified_calendar']['closure_authority'],context['info']['metadata'])
    _require(closure['artifact_sha256'] == series.metadata['closure_binding']['authority_sha256']
             and cal['gap_after_rows'] == [r['gap_after_m1_row']+1 for r in closure['intervals']]
             and cal['published_closure_after_rows'] == [r['gap_after_m1_row']+1 for r in closure['intervals']
                                                        if r['successor_across_gap_allowed']], 'SCORE_GAP_SEMANTICS')
    all_gaps = np.asarray(cal['gap_after_rows'], dtype=np.int64)
    published = np.asarray(cal['published_closure_after_rows'], dtype=np.int64)
    _require(np.array_equal(all_gaps, np.flatnonzero(np.diff(series.arrays['validity_ns']) != target.MINUTE_NS) + 1)
             and np.array_equal(np.unique(published), published) and np.isin(published, all_gaps).all(),
             'SCORE_CALENDAR_AXES')
    candidate = context['december_sources']['candidate']
    return {'candle_orientation':candidate['references']['candle_orientations'], 'storage_plan':s['storage_plan'],
            'gap_after_rows':all_gaps, 'published_closure_after_rows':published,
            'receipt_refs':{'prior':launch['prefix_prior'], 'candle':candidate['bindings']['frozen_candle_reference'],
                            'fit_terminal_review':launch['score_origin']['fit_terminal_review'],
                            'terminal_checkpoint':launch['score_origin']['fit_checkpoint'],
                            'scoring_admission':launch['scoring_binding'], 'target_manifest':launch['target_manifest'],
                            'target_result':s['target_result'], 'target_terminal':s['target_terminal'],
                            'calendar_authority':s['calendar_authority']}}


def preflight_december_private_inputs(launch, *, qualification_output, runtime_root):
    """One target/feature route check. The capped caller owns claim, budget and terminal publication."""
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
    from gx1.contracts import entry_dynamic_mtf_december_native_v1 as authority
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_gc_mtf_entry_execution_profile
    from gx1.scripts import run_entry_native_direction_dense_qualification_v1 as private
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(launch['run_scope'] == authority.PRIVATE_SCOPE and _role(launch) == 'fit', 'PRIVATE_PREFLIGHT_ONLY')
    before = common._io()
    context = worker._dynamic_state_metadata(launch, common)
    context['state_artifact_directory'] = Path(qualification_output)
    denied = common._install_read_guard(context, Path(qualification_output), Path(runtime_root))
    prepared = None
    summaries = []
    try:
        prepared, targets, view, _, ordered = worker._prepare_dynamic_state(context, launch, np)
        fixed = private.select_dynamic_state_qualification_parents(ordered, {'parents_sorted':launch['qualification_parent_rows']}, np=np)
        profile = compile_full_gc_mtf_entry_execution_profile(context['fitted']['gc_mtf_history'])
        for start in (0, 16):
            selected = fixed[start:start+16]
            raw, truth = worker._batch(prepared, targets, view, selected, np)
            support = private._dynamic_state_input_support(raw, profile, prepared['gc']['layout'], common=common, np=np)
            observed = truth['observed']
            summaries.append({'parent_rows':selected.tolist(), 'family_nonzero':support['family_raw_nonzero_counts'],
                              'features':{k:{'shape':list(v.shape), 'dtype':v.dtype.name,
                                             'sha256':hashlib.sha256(v.tobytes()).hexdigest()}
                                          for k,v in raw['features'].items()},
                              'mature_observed_by_tf':np.count_nonzero(observed,axis=(0,2)).tolist(),
                              'target_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest()
                                               for k,v in truth.items() if isinstance(v,np.ndarray)}})
        _check_payload_stats(context['ancestor_plan']['payloads'])
        for name, r in context['state_array_refs'].items():
            _require(common._stat(r['path']) == context['state_array_stats'][name], 'PREFLIGHT_TARGET_BOOKEND')
        _require(not denied, 'PREFLIGHT_DENIED_READ')
        after = common._io()
        return {'status':'PASS_FIT_ONLY_TWO_REAL_B16_DATA_BATCHES', 'input_role':'fit',
                'fit_parents':len(ordered), 'fit_target':launch['target_manifest'], 'feature_view':launch['feature_view'],
                'restricted_raw_payloads':len(context['ancestor_plan']['payloads']),
                'role_target_arrays':len(context['state_array_refs']), 'source_rows':prepared['physical_source_rows'],
                'batches':summaries, 'denied_reads':denied, 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'io_before':before,'io_after':after,'model_calls':0,'optimizer_updates':0,'held_out_outcomes_read':False}
    finally:
        if prepared is not None:
            _PageRelease.release_pages(None, prepared, None)
