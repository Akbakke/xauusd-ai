"""Exact metadata plus synthetic gather equivalence; no model, market arrays or fit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts import entry_anchor_span_feature_view_v1 as extension
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.models.entry_v10.entry_v10_input_normalization import _hash_int64_indices
from tests.test_entry_native_direction_prefix_normalization_v1 import prefix_case


def read(ref):
    path=Path(ref['path'])
    if not path.is_file():pytest.skip('immutable research metadata unavailable on this host')
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==ref['sha256']
    return json.loads(data)


def ref(name):return dict(path='/synthetic/'+name,sha256='1'*64)


def binding_case(spec):
    parent=read(spec['parent_reference'])
    base=read(spec['normalization_evidence']['expanded_base'])
    units=(None if spec['units_sha'] is None else read(spec['normalization_evidence']['units_profile']))
    constructor=extension.build_anchor_span_constructor_metadata(read(extension.PARENT_CONSTRUCTOR_REF))
    packet=extension.build_anchor_span_prefix_packet(parent,base,unobserved_units_profile=units)
    proof=parent['prefix_qualification']['proof']['source_bindings']
    sources={key:ref(key) for key in views.SOURCE_REF_KEYS|views.TIME_QUERY_V2_SOURCE_REF_KEYS|views.ANCHOR_SPAN_SOURCE_REF_KEYS}
    sources.update(input_qualification=spec['qualification_reference'],
        raw_input_provider_plan=proof['provider_plan'],raw_gc_layout=proof['boolean_raw_gc_layout'],
        raw_gc_transition=proof['boolean_raw_domain_transition'],time_query_mature_prefix_plan=proof['current_plan'],
        anchor_span_inputs_source=dict(path=extension.OVERLAY_OWNER_PATH,sha256='1'*64))
    for key,path in [('mtf_window_source','gx1/features/htf_features.py'),
                     ('gc_gather_source','gx1/contracts/entry_gc_mtf_history_v1.py'),
                     ('owner_profile_source','gx1/features/entry_specialist_feature_groups_v1.py')]:
        sources[key]=dict(path=str(views.time_query_targets.ROOT/path),sha256=proof['current_source_owners'][path])
    for key,name in [('time_query_v2_model_source','market_state_time_query_model_v2.py'),
                     ('time_query_v2_loss_source','market_state_time_query_loss_v2.py')]:
        sources[key]=dict(path=str(views.time_query_targets.ROOT/'gx1/contracts'/name),sha256='1'*64)
    path,digest=views.time_query_targets.CANDIDATE_REFS['tape']
    sources['time_query_candidate_tape']=dict(path=str(path),sha256=digest)
    return dict(prefix_normalization=packet,prefix_normalization_ref=ref('packet.json'),
        constructor_metadata=constructor,constructor_metadata_ref=ref('constructor.json'),
        source_bindings=sources,max_batch_rows=16,unobserved_units_profile=units,
        full_feature_profile=True,full_ta_feature_profile=True,market_state_time_query_v2=True,
        dedicated_m5_side=True,dedicated_tf_activity=True)


def test_immutable_fold2_span_metadata_rejects_changed_feature_profile():
    with pytest.raises(RuntimeError, match='ANCHOR_SPAN_NORMALIZATION_CONSUMER_BINDING'):
        binding_case(extension.FOLD2_BINDING)


def test_immutable_fold1_span_metadata_rejects_changed_feature_profile():
    with pytest.raises(RuntimeError, match='ANCHOR_SPAN_NORMALIZATION_CONSUMER_BINDING'):
        binding_case(extension.FOLD1_BINDING)


def _require_legacy_bound_profile():
    gc = views.owners.compile_compact_entry_execution_profile()['physical_field_names']
    ta = views.owners.compile_full_ta_entry_execution_profile()['physical_field_names']
    if (len(gc['signal']), len(gc['mtf']), len(ta['signal']), len(ta['mtf'])) != (2727, 255, 317, 255):
        pytest.skip('immutable 317/255 span-view receipts belong to the old feature profile')


@pytest.fixture(scope='module')
def binding():
    _require_legacy_bound_profile()
    return binding_case(extension.FOLD1_BINDING)


@pytest.fixture(scope='module')
def fold2_binding():
    _require_legacy_bound_profile()
    return binding_case(extension.FOLD2_BINDING)


def test_full_gc_fold2_metadata_binds_expanded_raw_and_parent_gc(fold2_binding):
    args=deepcopy(fold2_binding)
    for key in ('full_ta_feature_profile','dedicated_m5_side','dedicated_tf_activity'):
        del args[key]
    view=views.bind_native_direction_feature_view(**args)
    assert views.bind_native_direction_feature_view(**args,expected=view)==view
    assert view['physical_widths']==dict(signal=2727,ctx_cont=2542,ctx_cat=1,mtf=255)
    assert view['entry_execution_profile']['name']==views.owners.FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
    assert view['prefix_normalization']['gc_normalization_sha256']==extension.FOLD2_BINDING['gc_normalization_sha']
    assert 'ta_base_normalization_sha256' not in view['prefix_normalization']
    assert 'full_ta_feature_profile' not in view
    assert view['research_objective']==extension.factorized_time_query_model_contract(ta_only=False)


def test_full_gc_fold1_rejects_incompatible_gc_unit_profile(binding):
    args=deepcopy(binding)
    for key in ('full_ta_feature_profile','dedicated_m5_side','dedicated_tf_activity'):
        del args[key]
    with pytest.raises(RuntimeError,match='UNITS_BINDING'):
        views.bind_native_direction_feature_view(**args)


def test_exact_metadata_dispatch_preserves_parent_gc_and_new_geometry(binding):
    view=views.bind_native_direction_feature_view(**binding)
    assert views.bind_native_direction_feature_view(**binding,expected=view)==view
    assert view['physical_widths']==dict(signal=317,ctx_cont=132,ctx_cat=1,mtf=255)
    packet=binding['prefix_normalization']
    assert packet['gc_mtf_history']==packet['parent_prefix_normalization']['gc_mtf_history']
    assert view['constructor_scalar_config']==extension.SCALARS
    assert view['prefix_normalization']['ta_base_normalization_sha256']==extension.TA_CONTRACT_SHA
    assert view['anchor_span_input_extension']['execution_authorized'] is False


@pytest.mark.parametrize('fault',['old_units','old_packet','changed_old_stat','changed_constructor','missing_overlay','invalid_decoder_flag','marker_removed'])
def test_metadata_rejects_unbound_or_resealed_changes(binding,fault):
    args=deepcopy(binding)
    if fault=='old_units':args['unobserved_units_profile']=extension.units.mature_fold1_prefix_units_profile()
    elif fault=='old_packet':args['prefix_normalization']=args['prefix_normalization']['parent_prefix_normalization']
    elif fault=='changed_old_stat':
        packet=args['prefix_normalization'];packet['base_normalization']['surfaces']['signal']['center'][0]+=1
        packet['contract_sha256']=views._sha({k:v for k,v in packet.items() if k!='contract_sha256'})
    elif fault=='changed_constructor':args['constructor_metadata']['dropout']=.1
    elif fault=='missing_overlay':del args['source_bindings']['anchor_span_inputs_source']
    elif fault=='invalid_decoder_flag':args['dedicated_non_m5_side']=1
    else:
        view=views.bind_native_direction_feature_view(**args);del view['anchor_span_input_extension']
        view['contract_sha256']=views._sha({k:v for k,v in view.items() if k!='contract_sha256'})
        with pytest.raises(RuntimeError):views._require_view(view,unobserved_units_profile=args['unobserved_units_profile'])
        return
    with pytest.raises(RuntimeError):views.bind_native_direction_feature_view(**args)



@pytest.mark.parametrize('side', [False, True])
def test_span_view_composes_decoders_without_changing_inputs_or_normalization(binding, side):
    original = views.bind_native_direction_feature_view(**binding)
    combined = views.bind_native_direction_feature_view(**binding, dedicated_non_m5_side=side)
    assert views.bind_native_direction_feature_view(
        **binding, dedicated_non_m5_side=side, expected=combined) == combined
    changed = {key for key in set(original) | set(combined)
               if original.get(key) != combined.get(key)}
    assert changed == ({'dedicated_non_m5_side', 'research_objective', 'contract_sha256'}
                       if side else set())
    assert combined['prefix_normalization'] == original['prefix_normalization']
    assert combined['physical_field_names_sha256'] == original['physical_field_names_sha256']
    assert combined['anchor_span_input_extension'] == original['anchor_span_input_extension']
    if side:
        assert combined['research_objective']['dedicated_non_m5_side'] is True
        assert combined['research_objective']['dedicated_tf_activity'] is True
        with pytest.raises(RuntimeError, match='EXPECTED_VIEW'):
            views.bind_native_direction_feature_view(
                **binding, dedicated_non_m5_side=True, expected=original)


@pytest.mark.parametrize('bad', [False, 1, 'true'])
def test_combined_span_view_rejects_resealed_nonliteral_side_flag(binding, bad):
    combined = views.bind_native_direction_feature_view(**binding, dedicated_non_m5_side=True)
    combined['dedicated_non_m5_side'] = bad
    combined['contract_sha256'] = views._sha({k: v for k, v in combined.items() if k != 'contract_sha256'})
    with pytest.raises(RuntimeError):
        views._require_view(combined, unobserved_units_profile=binding['unobserved_units_profile'])


def readonly(value):
    value=np.ascontiguousarray(value);value.setflags(write=False);return value


def make_gather_case(monkeypatch, *, later=False, full_gc=False):
    # Metadata identity is tested separately above; this small fixture isolates
    # real canonical window/cropping behavior with no market-value reads.
    monkeypatch.setattr(views,'_require_view',lambda view,**kw:view)
    c=prefix_case();fields,_,_=views._canonical()
    local_names=['chart.'+n for n in overlay.SPAN_FIELDS]
    source=c['entry_sequence_source']['signal']
    for j,name in enumerate(local_names):source[:,fields['signal'].index(name)]=np.arange(len(source))%11+j+1
    source[:,fields['signal'].index(overlay.EMA_DELTA_LOCAL_FIELD)]=np.arange(len(source),dtype=np.float32)/7-5
    c['snap_x']=source[c['entry_sequence_source']['sample_positions']].copy()
    # Span additions are not historical context aliases; all old aliases stay.
    for tf,frame in c['multi_tf_sources'].items():
        values=np.array(frame.to_numpy(dtype=np.float32,copy=False),copy=True,order='C')
        for j,name in enumerate(overlay.SPAN_FIELDS):values[:,fields['mtf'].index(name)]=np.arange(len(values))%13+j+1
        values[:,fields['mtf'].index(overlay.EMA_DELTA_MTF_FIELD)]=np.arange(len(values),dtype=np.float32)/9-3
        rebuilt=pd.DataFrame(values,index=frame.index,columns=frame.columns,copy=False)
        rebuilt.attrs.update(frame.attrs)
        rebuilt.attrs['feats_np']=values
        c['multi_tf_sources'][tf]=rebuilt
    for key in ('parent_rows','bar_start_ns','decision_ns','snap_x','ctx_cont','ctx_cat'):c[key]=readonly(c[key])
    for key in ('signal','times_ns','sample_positions'):c['entry_sequence_source'][key]=readonly(c['entry_sequence_source'][key])
    for frame in c['multi_tf_sources'].values():
        frame.attrs['feats_np']=readonly(frame.to_numpy(dtype=np.float32,copy=False))
        frame.attrs['ts_int64']=readonly(frame.attrs['ts_int64'])
    base_view=dict(full_feature_profile=True,full_axes=dict(rows=6),
        prefix=dict(rows=4,cutoff_ns=c['cutoff_ns']),max_batch_rows=2,
        contract_sha256='a'*64,
        prefix_normalization=dict(gc_layout_sha256=c['gc_layout']['contract_sha256']))
    if not full_gc:base_view['full_ta_feature_profile']=True
    original={k:c[k] for k in ('parent_rows','bar_start_ns','decision_ns','entry_sequence_source','snap_x','ctx_cont','ctx_cat','multi_tf_sources')}
    original.update(gc_surfaces=c['gc_surfaces'] if full_gc else None,
                    gc_layout=c['gc_layout'] if full_gc else None,
                    batch_parent_rows=np.array([12,15] if later else [5,9],np.int64))
    expected=views.gather_native_direction_feature_batch(base_view,**original)
    keep=[i for i,n in enumerate(fields['signal']) if n not in local_names]
    kept_mtf=[i for i,n in enumerate(fields['mtf']) if n not in overlay.SPAN_FIELDS]
    local_parent=readonly(source[:,keep]);snap=readonly(c['snap_x'][:,keep])
    seq={**c['entry_sequence_source'],'signal':local_parent}
    populations={};payloads={};mtf_values={};mtf_times={}
    def sidecar(lane,rows,times,spans):
        shift=views.BAR_NS if lane=='local' else int(overlay.htf.MULTI_TF_SHIFT[lane].value)
        payloads[lane]=dict(row_indices=rows,times_ns=times,native_bar_positions=rows+100,anchor_span_bars=spans)
        populations[lane]=dict(selected_unique_row_count=len(rows),
            selected_row_indices_sha256=_hash_int64_indices(rows,namespace='entry_local_m5_rows' if lane=='local' else 'mtf:'+lane),
            available_ns_sha256=hashlib.sha256((times+shift).astype('<i8').tobytes()).hexdigest())
    rows=np.arange(5,108 if later else 105,dtype=np.int64)
    sidecar('local',rows,seq['times_ns'][rows],source[np.ix_(rows,[fields['signal'].index(n) for n in local_names])])
    for tf,frame in c['multi_tf_sources'].items():
        mtf_values[tf]=readonly(frame.attrs['feats_np'][:,kept_mtf]);mtf_times[tf]=frame.attrs['ts_int64']
        all_ids=[]
        for start in c['bar_start_ns'][:6 if later else 4]:
            windows=views.get_model_native_multi_tf_route_windows(c['multi_tf_sources'],decision_bar_start=pd.Timestamp(int(start),tz='UTC'),per_tf_seq_lens=dict(views.history.WINDOW_ROWS),route_timeframes=views.history.TIMEFRAMES,base_bar_duration=pd.Timedelta(seconds=300),return_row_indices=True)
            all_ids.extend(windows[tf][1].tolist())
        rows=np.unique(all_ids).astype(np.int64)
        sidecar(tf,rows,mtf_times[tf][rows],frame.attrs['feats_np'][np.ix_(rows,[fields['mtf'].index(n) for n in overlay.SPAN_FIELDS])])
    view={**base_view,'anchor_span_input_extension':{'synthetic_only':True}}
    build=dict(entry_sequence_source=seq,snap_x=snap,ctx_cont=c['ctx_cont'],ctx_cat=c['ctx_cat'],
        parent_signal_names=[fields['signal'][i] for i in keep],parent_mtf_names=[fields['mtf'][i] for i in kept_mtf],
        parent_mtf_values=mtf_values,parent_mtf_times=mtf_times,span_payloads=payloads,physical_populations=populations,unobserved_units_profile=None)
    if later:
        # The exact 118249-row metadata gate is tested below without allocating
        # full-width features. This fixture isolates real native gather/window
        # behavior on two synthetic later rows, including original row IDs.
        later_binding=dict(later_parent_positions=readonly(np.array([4,5],np.int64)),
            later_parent_rows=readonly(c['parent_rows'][4:6].copy()),
            later_decision_ns=readonly(c['decision_ns'][4:6].copy()),
            later_coverage_sha256='b'*64,
            later_qualification=('/home/andre2/GX1_RUNS/SYNTHETIC/RESULT.json','c'*64))
        monkeypatch.setattr(overlay,'_require_later_coverage',lambda *a,**kw:later_binding)
        build.update(later_coverage={'synthetic_only':True,
            'score_end_exclusive_ns':int(c['decision_ns'][-1])+views.BAR_NS},later_axes={
            key:c[key] for key in ('parent_rows','bar_start_ns','decision_ns')})
    bound=overlay.build_anchor_span_inputs(view,**build)
    args={**original,'entry_sequence_source':seq,'snap_x':snap,'multi_tf_sources':bound.mtf_sources,'anchor_span_inputs':bound}
    return view,args,expected,build


def test_composed_batch_matches_full_current_arrays_and_original_native_row_witnesses(monkeypatch):
    view,args,expected,_=make_gather_case(monkeypatch)
    output=views.gather_native_direction_feature_batch(view,**args)
    for key,value in expected['features'].items():
        assert output['features'][key].dtype==value.dtype
        assert output['features'][key].tobytes()==value.tobytes(),key
    assert output['witness']==expected['witness']
    assert all(rows[0]>0 for rows in args['anchor_span_inputs'].mtf_parent_rows.values())


def test_new_ema_sidecars_match_full_native_values_and_reject_mismatch(monkeypatch):
    view,args,expected,build=make_gather_case(monkeypatch)
    local_name,mtf_name=overlay.EMA_DELTA_LOCAL_FIELD,overlay.EMA_DELTA_MTF_FIELD
    source=build['entry_sequence_source']['signal']
    local_at=build['parent_signal_names'].index(local_name)
    local_values=source[:,local_at].copy()
    build['entry_sequence_source']={**build['entry_sequence_source'],
        'signal':readonly(np.delete(source,local_at,axis=1))}
    build['snap_x']=readonly(np.delete(build['snap_x'],local_at,axis=1))
    build['parent_signal_names'].pop(local_at)
    mtf_values={}
    for tf,parent in build['parent_mtf_values'].items():
        at=build['parent_mtf_names'].index(mtf_name)
        mtf_values[tf]=parent[:,at].copy()
        build['parent_mtf_values'][tf]=readonly(np.delete(parent,at,axis=1))
    build['parent_mtf_names'].pop(at)
    sha=lambda a,dtype:hashlib.sha256(np.ascontiguousarray(a,dtype=dtype).tobytes()).hexdigest()
    payloads,surfaces,refs={},{},{}
    for lane in ('local',*overlay.TIMEFRAMES):
        span=build['span_payloads'][lane]
        rows,times=span['row_indices'],span['times_ns']
        values=(local_values if lane=='local' else mtf_values[lane])[rows].astype(np.float32)
        payloads[lane]=dict(row_indices=rows.copy(),times_ns=times.copy(),
                            delta_5_atr=np.ascontiguousarray(values))
        refs[lane]=ref(lane+'_EMA_DELTA.npz')
        surfaces[lane]=dict(artifact=refs[lane],span_artifact=ref(lane+'_ANCHOR_SPANS.npz'),
            row_indices_sha256=sha(rows,'<i8'),times_ns_sha256=sha(times,'<i8'),
            delta_5_atr_sha256=sha(values,'<f4'),rows=len(rows),
            first_time_ns=int(times[0]),last_time_ns=int(times[-1]),old_slope_parity_rows=1)
    manifest=dict(schema_version=overlay.EMA_DELTA_MANIFEST_SCHEMA,status='COMPLETE',
        cutoff_ns=overlay.COLD_FULL_LATER_END_NS,
        source=ref('source.parquet'),local_surface_manifest=ref('source.json'),
        mtf_cache_manifest=ref('cache.json'),span_reuse_result=ref('span.json'),
        fields=dict(local=local_name,mtf=mtf_name),surfaces=surfaces,
        test_data_used=False,model_forwards=0,optimizer_steps=0)
    manifest_ref=ref('MANIFEST.json')
    result=dict(schema_version=overlay.EMA_DELTA_RESULT_SCHEMA,status='COMPLETE',
        manifest=manifest_ref,surfaces=deepcopy(surfaces),source=manifest['source'],
        cutoff_ns=manifest['cutoff_ns'],fit_rows=29574,later_parent_start=29849,
        later_rows=17489,test_data_used=False,model_forwards=0,optimizer_steps=0)
    result_ref=ref('RESULT.json')
    view['physical_widths']=dict(signal=318,ctx_cont=132,ctx_cat=1,mtf=256)
    view['ema_delta_input_extension']=dict(schema_version=overlay.EMA_DELTA_INPUT_SCHEMA,
        physical_input_result=result_ref,sidecars=refs,fit_rows=29574,
        score_parent_start=29849,score_rows=17489,
        score_end_exclusive_ns=overlay.COLD_FULL_LATER_END_NS,execution_authorized=False)
    bundle=dict(result_ref=result_ref,result=result,manifest_ref=manifest_ref,
                manifest=manifest,payloads=payloads)
    build['ema_delta_bundle']=bundle
    bound=overlay.build_anchor_span_inputs(view,**build)
    args.update(entry_sequence_source=build['entry_sequence_source'],snap_x=build['snap_x'],
        multi_tf_sources=bound.mtf_sources,anchor_span_inputs=bound)
    output=views.gather_native_direction_feature_batch(view,**args)
    assert output['witness']==expected['witness']
    for key,value in expected['features'].items():
        assert output['features'][key].tobytes()==value.tobytes(),key
    later={'physical_sidecars':{lane:item['span_artifact'] for lane,item in surfaces.items()}}
    assert set(overlay._require_ema_delta_bundle(
        view,bundle,build['span_payloads'],later))=={'local',*overlay.TIMEFRAMES}
    later['physical_sidecars']['M15']=ref('different_span.npz')
    with pytest.raises(RuntimeError,match='EMA_LATER_SPAN_REF_AND_AXES:M15'):
        overlay._require_ema_delta_bundle(view,bundle,build['span_payloads'],later)
    with pytest.raises(RuntimeError,match='EMA_BUNDLE_REQUIRED_OR_UNEXPECTED'):
        overlay.build_anchor_span_inputs(view,**{k:v for k,v in build.items() if k!='ema_delta_bundle'})
    changed=deepcopy(bundle)
    changed['payloads']['M15']['delta_5_atr'][0]+=1
    with pytest.raises(RuntimeError,match='EMA_SURFACE_ARRAYS:M15'):
        overlay.build_anchor_span_inputs(view,**{**build,'ema_delta_bundle':changed})
    changed=deepcopy(bundle)
    changed['payloads']['M15']['times_ns']+=1
    changed['manifest']['surfaces']['M15']['times_ns_sha256']=sha(
        changed['payloads']['M15']['times_ns'],'<i8')
    changed['manifest']['surfaces']['M15']['first_time_ns']+=1
    changed['manifest']['surfaces']['M15']['last_time_ns']+=1
    changed['result']['surfaces']=deepcopy(changed['manifest']['surfaces'])
    with pytest.raises(RuntimeError,match='UNCOVERED_OR_DIFFERENT_CLOCK'):
        overlay.build_anchor_span_inputs(view,**{**build,'ema_delta_bundle':changed})


def test_full_gc_composed_batch_preserves_all_xau_and_gc_values(monkeypatch):
    view,args,expected,_=make_gather_case(monkeypatch,full_gc=True)
    output=views.gather_native_direction_feature_batch(view,**args)
    assert all('gc_seq_'+tf.lower() in output['features'] for tf in overlay.TIMEFRAMES)
    assert set(output['features'])==set(expected['features'])
    for key,value in expected['features'].items():
        assert output['features'][key].tobytes()==value.tobytes(),key
    assert output['witness']==expected['witness']


@pytest.mark.parametrize('fault',['later_parent','missing_overlay','different_source','mutable_backing','changed_clock','changed_population'])
def test_composed_input_scope_clock_and_backing_guards(monkeypatch,fault):
    view,args,_,build=make_gather_case(monkeypatch)
    if fault=='later_parent':args['batch_parent_rows']=np.array([12],np.int64)
    elif fault=='missing_overlay':args['anchor_span_inputs']=None
    elif fault=='different_source':args['entry_sequence_source']={**args['entry_sequence_source'],'signal':readonly(args['entry_sequence_source']['signal'].copy())}
    elif fault=='mutable_backing':args['multi_tf_sources']['M5'].attrs['feats_np'].setflags(write=True)
    elif fault=='changed_clock':
        args['entry_sequence_source']['times_ns'].setflags(write=True)
    else:
        build['physical_populations']['M5']['selected_row_indices_sha256']='f'*64
        with pytest.raises(RuntimeError,match='QUALIFIED_POPULATION'):overlay.build_anchor_span_inputs(view,**build)
        return
    with pytest.raises(RuntimeError):views.gather_native_direction_feature_batch(view,**args)


def later_metadata_case(fold_id=1):
    """Synthetic axes only; no retained market values or large feature matrices."""
    panel=overlay.later_panel(fold_id=fold_id)
    start=panel['score_start_ns']
    decisions=start+(np.arange(118249,dtype=np.int64)-panel['score_parent_start'])*views.BAR_NS
    axes=dict(parent_rows=readonly(3*np.arange(118249,dtype=np.int64)+2),
              bar_start_ns=readonly(decisions-views.BAR_NS),decision_ns=readonly(decisions))
    sha=lambda a:hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()
    view=dict(contract_sha256='a'*64,full_axes=dict(rows=118249,**{
        k+'_sha256':sha(a) for k,a in axes.items()}),
        prefix=dict(rows=panel['prefix_rows'],cutoff_ns=panel['prefix_cutoff_ns'],normalization_identity='unchanged'))
    populations={k:dict(selected_unique_row_count=3,selected_row_indices_sha256='d'*64,
                       available_ns_sha256='e'*64) for k in ('local',*overlay.TIMEFRAMES)}
    selected=(panel['score_parent_start']+np.array(panel['batch_indices'],dtype=np.int64)[:,None]*16
              +np.arange(16,dtype=np.int64)).reshape(-1)
    coverage=dict(schema_version=overlay.LATER_COVERAGE_SCHEMA,status='COMPLETE',
        qualification=dict(path='/home/andre2/GX1_RUNS/SYNTHETIC_LATER/RESULT.json',sha256='f'*64),
        view_sha256=view['contract_sha256'],full_axes=deepcopy(view['full_axes']),
        normalization_prefix=deepcopy(view['prefix']),score_parent_start=panel['score_parent_start'],score_rows=panel['score_rows'],
        score_start_ns=panel['score_start_ns'],score_end_exclusive_ns=panel['score_end_exclusive_ns'],
        batch_size=16,batch_indices=panel['batch_indices'],
        selected_parent_rows_sha256=sha(axes['parent_rows'][selected]),
        selected_decision_ns_sha256=sha(axes['decision_ns'][selected]),
        physical_populations=deepcopy(populations))
    return view,coverage,axes,populations,selected


def cold_full_metadata_case():
    view,coverage,axes,populations,_=later_metadata_case(1)
    panel=overlay.cold_full_later_panel()
    # The 65-minute closed-market gap leaves the full 1,440-minute target
    # mature before the October score starts, as in the bound prefix.
    decisions=axes['decision_ns'].copy()
    decisions[:panel['prefix_rows']]-=65*60*10**9
    axes['decision_ns']=readonly(decisions)
    axes['bar_start_ns']=readonly(decisions-views.BAR_NS)
    sha=lambda a:hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()
    view['full_axes']={**view['full_axes'],**{
        k+'_sha256':sha(a) for k,a in axes.items()}}
    view['prefix']['last_decision_ns']=int(decisions[panel['prefix_rows']-1])
    coverage.update(schema_version=overlay.COLD_FULL_LATER_COVERAGE_SCHEMA,
        full_axes=deepcopy(view['full_axes']),normalization_prefix=deepcopy(view['prefix']),
        score_parent_start=panel['score_parent_start'],score_rows=panel['score_rows'],
        score_start_ns=panel['score_start_ns'],score_end_exclusive_ns=panel['score_end_exclusive_ns'],
        batch_size=panel['batch_size'],full_batches=panel['full_batches'],tail_rows=panel['tail_rows'])
    del coverage['batch_indices']
    positions=np.arange(panel['score_parent_start'],panel['score_parent_start']+panel['score_rows'],dtype=np.int64)
    coverage['selected_parent_rows_sha256']=sha(axes['parent_rows'][positions])
    coverage['selected_decision_ns_sha256']=sha(axes['decision_ns'][positions])
    root='/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002/'
    sidecars={lane:dict(path=root+lane+'_ANCHOR_SPANS.npz',sha256='1'*64)
              for lane in ('local',*overlay.TIMEFRAMES)}
    coverage['physical_sidecars']=deepcopy(sidecars)
    return view,coverage,axes,populations,sidecars,positions


def test_cold_full_later_covers_all_mature_rows_including_single_tail():
    view,coverage,axes,populations,sidecars,positions=cold_full_metadata_case()
    before=deepcopy((view,coverage,populations,sidecars))
    actual=overlay._require_later_coverage(view,coverage,axes,populations,sidecar_refs=sidecars)
    assert (view,coverage,populations,sidecars)==before
    assert len(positions)==17489 and positions[0]==29849 and positions[-1]==47337
    assert coverage['full_batches']*coverage['batch_size']+coverage['tail_rows']==len(positions)
    for key,expected in [('later_parent_positions',positions),
                         ('later_parent_rows',axes['parent_rows'][positions]),
                         ('later_decision_ns',axes['decision_ns'][positions])]:
        assert np.array_equal(actual[key],expected) and not actual[key].flags.writeable
        assert not any(np.shares_memory(actual[key],a) for a in axes.values())
    assert len(actual['later_coverage_sha256'])==64
    assert actual['later_qualification']==tuple(coverage['qualification'][k] for k in ('path','sha256'))


@pytest.mark.parametrize('fault',['view','prefix','maturity','full_axes','population','sidecar_binding',
    'sidecar_sha','sidecar_path','batch_count','tail','score_count','selected_axis_hash',
    'axis_changed','qualification'])
def test_cold_full_later_fails_closed_on_wrong_binding_or_coverage(fault):
    view,coverage,axes,populations,sidecars,_=cold_full_metadata_case()
    if fault=='view':coverage['view_sha256']='b'*64
    elif fault=='prefix':coverage['normalization_prefix']['rows']+=1
    elif fault=='maturity':view['prefix']['last_decision_ns']+=views.BAR_NS
    elif fault=='full_axes':coverage['full_axes']['rows']-=1
    elif fault=='population':coverage['physical_populations']['M5']['available_ns_sha256']='0'*64
    elif fault=='sidecar_binding':sidecars['M5']['sha256']='2'*64
    elif fault=='sidecar_sha':coverage['physical_sidecars']['M5']['sha256']='g'*64;sidecars=deepcopy(coverage['physical_sidecars'])
    elif fault=='sidecar_path':coverage['physical_sidecars']['M5']['path']='/home/andre2/GX1_RUNS/other/M5_ANCHOR_SPANS.npz';sidecars=deepcopy(coverage['physical_sidecars'])
    elif fault=='batch_count':coverage['full_batches']-=1
    elif fault=='tail':coverage['tail_rows']=0
    elif fault=='score_count':coverage['score_rows']-=1
    elif fault=='selected_axis_hash':coverage['selected_decision_ns_sha256']='0'*64
    elif fault=='axis_changed':
        a=axes['parent_rows'].copy();a[40000]+=1;axes['parent_rows']=readonly(a)
    else:coverage['qualification']['sha256']='g'*64
    with pytest.raises(RuntimeError,match='LATER_COLD'):
        overlay._require_later_coverage(view,coverage,axes,populations,sidecar_refs=sidecars)


@pytest.mark.parametrize('fold_id,first,last,count',[(1,29849,47592,4096),(2,47602,64881,4096),(3,64890,82425,17536)])
def test_fixed_later_metadata_owns_only_selected_axes_without_changing_train_normalization(fold_id,first,last,count):
    view,coverage,axes,populations,positions=later_metadata_case(fold_id)
    before=deepcopy((view,coverage,populations))
    actual=overlay._require_later_coverage(view,coverage,axes,populations)
    assert (view,coverage,populations)==before
    assert len(positions)==count and positions[0]==first and positions[-1]==last
    for key,expected in [('later_parent_positions',positions),
                         ('later_parent_rows',axes['parent_rows'][positions]),
                         ('later_decision_ns',axes['decision_ns'][positions])]:
        assert np.array_equal(actual[key],expected) and not actual[key].flags.writeable
        assert not any(np.shares_memory(actual[key],a) for a in axes.values())
    assert len(actual['later_coverage_sha256'])==64
    assert actual['later_qualification']==tuple(coverage['qualification'][k] for k in ('path','sha256'))
    coverage['batch_indices'][0]=1
    assert actual['later_parent_positions'][0]==first


@pytest.mark.parametrize('fault',[
    'view','full_axes','normalization','population','batch_selection','score_count','boolean_count',
    'unfinished','ref_path','ref_sha','selected_axis_hash','mutable_axis','changed_axis','extra_key'])
@pytest.mark.parametrize('fold_id',[1,2,3])
def test_later_metadata_rejects_scope_axis_or_provenance_changes(fault,fold_id):
    view,coverage,axes,populations,_=later_metadata_case(fold_id)
    if fault=='view':coverage['view_sha256']='b'*64
    elif fault=='full_axes':coverage['full_axes']['rows']-=1
    elif fault=='normalization':coverage['normalization_prefix']['cutoff_ns']+=views.BAR_NS
    elif fault=='population':coverage['physical_populations']['M5']['available_ns_sha256']='0'*64
    elif fault=='batch_selection':coverage['batch_indices'][1]+=1
    elif fault=='score_count':coverage['score_rows']-=1
    elif fault=='boolean_count':coverage['batch_size']=True
    elif fault=='unfinished':coverage['status']='PREPARED'
    elif fault=='ref_path':coverage['qualification']['path']='/home/andre2/GX1_RUNS/../RESULT.json'
    elif fault=='ref_sha':coverage['qualification']['sha256']='g'*64
    elif fault=='selected_axis_hash':coverage['selected_parent_rows_sha256']='0'*64
    elif fault=='mutable_axis':axes['parent_rows'].setflags(write=True)
    elif fault=='changed_axis':
        a=axes['parent_rows'].copy();a[30000]+=1;axes['parent_rows']=readonly(a)
    else:coverage['allow_oof']=True
    with pytest.raises(RuntimeError,match='LATER_'):
        overlay._require_later_coverage(view,coverage,axes,populations)


def test_later_metadata_rejects_resealed_clock_outside_fixed_score_interval():
    view,coverage,axes,populations,_=later_metadata_case()
    # Resealing full-axis hashes cannot move the score panel into a later year.
    for key in ('bar_start_ns','decision_ns'):
        a=axes[key].copy();a[29849:]+=120*24*3600*10**9;axes[key]=readonly(a)
        view['full_axes'][key+'_sha256']=hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()
    coverage['full_axes']=deepcopy(view['full_axes'])
    with pytest.raises(RuntimeError,match='LATER_SCORE_INTERVAL'):
        overlay._require_later_coverage(view,coverage,axes,populations)


def test_separate_later_adapter_matches_full_arrays_and_original_window_rows(monkeypatch):
    view,args,expected,_=make_gather_case(monkeypatch,later=True)
    before=deepcopy(view['prefix'])
    output=views.gather_native_direction_feature_batch(view,**args)
    assert view['prefix']==before
    for name,value in expected['features'].items():
        assert output['features'][name].dtype==value.dtype
        assert output['features'][name].tobytes()==value.tobytes(),name
    assert output['witness']==expected['witness']
    assert all(np.all(np.diff(rows)==1) for rows in args['anchor_span_inputs'].mtf_parent_rows.values())


@pytest.mark.parametrize('fault',['train_parent','wrong_decision','mutable_selected_axis'])
def test_later_gather_rejects_other_rows_clocks_or_mutable_binding(monkeypatch,fault):
    view,args,_,_=make_gather_case(monkeypatch,later=True)
    if fault=='train_parent':args['batch_parent_rows']=np.array([5],np.int64)
    elif fault=='wrong_decision':
        a=args['decision_ns'].copy();a[4]+=views.BAR_NS;args['decision_ns']=readonly(a)
    else:args['anchor_span_inputs'].later_parent_rows.setflags(write=True)
    with pytest.raises(RuntimeError,match='LATER_'):
        views.gather_native_direction_feature_batch(view,**args)


def test_later_envelope_rejects_packed_mtf_holes_even_with_updated_population_hash(monkeypatch):
    view,_,_,build=make_gather_case(monkeypatch,later=True)
    payload=build['span_payloads']['M5']
    keep=np.arange(len(payload['row_indices']))!=3
    for key in payload:payload[key]=payload[key][keep]
    pop=build['physical_populations']['M5']
    pop['selected_unique_row_count']=len(payload['row_indices'])
    pop['selected_row_indices_sha256']=_hash_int64_indices(payload['row_indices'],namespace='mtf:M5')
    pop['available_ns_sha256']=hashlib.sha256((payload['times_ns']+views.BAR_NS).astype('<i8').tobytes()).hexdigest()
    with pytest.raises(RuntimeError,match='SIDECAR_ARRAYS:M5'):
        overlay.build_anchor_span_inputs(view,**build)


def test_later_builder_requires_source_pointers_to_match_bound_full_clock(monkeypatch):
    view,_,_,build=make_gather_case(monkeypatch,later=True)
    source=build['entry_sequence_source']
    positions=source['sample_positions'].copy();positions[5]-=1
    build['entry_sequence_source']={**source,'sample_positions':readonly(positions)}
    with pytest.raises(RuntimeError,match='LATER_PARENT_SOURCE_CLOCK'):
        overlay.build_anchor_span_inputs(view,**build)


def test_train_builder_does_not_accept_unbound_later_axes(monkeypatch):
    view,_,_,build=make_gather_case(monkeypatch)
    with pytest.raises(RuntimeError,match='UNBOUND_LATER_AXES'):
        overlay.build_anchor_span_inputs(view,**build,later_axes={})


def test_fold2_view_uses_own_prefix_without_inheriting_units(binding, fold2_binding):
    old=views.bind_native_direction_feature_view(**binding)
    new=views.bind_native_direction_feature_view(**fold2_binding)
    assert views.bind_native_direction_feature_view(**fold2_binding,expected=new)==new
    views._require_view(new,unobserved_units_profile=None)
    assert new['prefix']['rows']==47338 and new['prefix']['cutoff_ns']==1735603200000000000
    assert new['prefix_normalization']['contract_sha256']==extension.FOLD2_BINDING['packet_sha']
    assert new['prefix_normalization']['ta_base_normalization_sha256']==extension.FOLD2_BINDING['ta_sha']
    assert 'unobserved_units_profile' not in new
    assert 'unobserved_units_profile_sha256' not in new['prefix_normalization']
    for key in ('physical_widths','physical_field_names_sha256','canonical_routing_sha256',
                'constructor_scalar_config','entry_execution_profile','research_objective'):
        assert new[key]==old[key]
    packet=fold2_binding['prefix_normalization']
    assert packet['unobserved_units_profile'] is None
    assert packet['gc_mtf_history']==packet['parent_prefix_normalization']['gc_mtf_history']
    assert extension.require_anchor_span_prefix_packet(packet)==packet


@pytest.mark.parametrize('fault',['fold1_units','cutoff','marker','normalization','parent_source','null_units_key'])
def test_fold2_view_rejects_resealed_wrong_scope_or_units(binding,fold2_binding,fault):
    args=deepcopy(fold2_binding)
    if fault=='fold1_units':
        args['unobserved_units_profile']=binding['unobserved_units_profile']
        with pytest.raises(RuntimeError,match='FOLD2_UNOBSERVED_UNITS_FORBIDDEN'):
            views.bind_native_direction_feature_view(**args)
        return
    view=views.bind_native_direction_feature_view(**args)
    if fault=='cutoff':view['prefix']['cutoff_ns']+=300*10**9
    elif fault=='marker':view[extension.MARKER_KEY]['parent_prefix_reference']=extension.FOLD1_BINDING['parent_reference']
    elif fault=='normalization':view['prefix_normalization']['base_sha256']=extension.BASE_CONTRACT_SHA
    elif fault=='parent_source':view['prefix_source_bindings_sha256']='0'*64
    else:view['unobserved_units_profile']=None
    view['contract_sha256']=views._sha({k:v for k,v in view.items() if k!='contract_sha256'})
    with pytest.raises(RuntimeError):views._require_view(view,unobserved_units_profile=None)


def test_retained_fold1_view_reconstructs_identically(binding):
    retained=read(dict(path='/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001/FEATURE_VIEW.json',
        sha256='2b15bcfe7fec09d83dfc6af71fe406b6c8153aa5d0a5d9262d112fea3ace3d60'))
    args={**binding,'prefix_normalization_ref':retained['prefix_normalization']['binding'],
          'constructor_metadata_ref':retained['constructor_metadata']['binding'],
          'source_bindings':retained['source_bindings'],'max_batch_rows':retained['max_batch_rows'],
          'dedicated_non_m5_side':retained.get('dedicated_non_m5_side',False)}
    assert views.bind_native_direction_feature_view(**args,expected=retained)==retained


def test_fold2_later_panel_rejects_fold1_selection_and_unqualified_prefix():
    view,coverage,axes,populations,_=later_metadata_case(2)
    coverage['batch_indices']=list(overlay.LATER_BATCH_INDICES)
    with pytest.raises(RuntimeError,match='LATER_FIXED_PANEL'):
        overlay._require_later_coverage(view,coverage,axes,populations)
    view,coverage,axes,populations,_=later_metadata_case(2)
    view['prefix']['cutoff_ns']+=views.BAR_NS
    coverage['normalization_prefix']=deepcopy(view['prefix'])
    with pytest.raises(RuntimeError,match='LATER_QUALIFIED_PREFIX'):
        overlay._require_later_coverage(view,coverage,axes,populations)


def test_later_adapter_uses_qualified_end_instead_of_fold1_global(monkeypatch):
    view,_,_,build=make_gather_case(monkeypatch,later=True)
    # The exact metadata binding is isolated in its own tests; the real array
    # guard must use the validated coverage cutoff for either supported panel.
    payload=build['span_payloads']['local']
    build['later_coverage']['score_end_exclusive_ns']=int(payload['times_ns'][-1])+views.BAR_NS
    with pytest.raises(RuntimeError,match='QUALIFIED_POPULATION:local'):
        overlay.build_anchor_span_inputs(view,**build)


def test_fold2_ta_anchor_span_accepts_direct_tf_only_with_five_side_heads(fold2_binding):
    args = deepcopy(fold2_binding)
    args.update(dedicated_non_m5_side=True, direct_tf_hidden=True)
    view = views.bind_native_direction_feature_view(**args)
    assert view['direct_tf_hidden'] is True
    assert view['dedicated_non_m5_side'] is True
    assert view['research_objective'] == extension.factorized_time_query_model_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True,
        dedicated_tf_activity=True, direct_tf_hidden=True)
    assert views.bind_native_direction_feature_view(**args, expected=view) == view
    del args['dedicated_non_m5_side']
    with pytest.raises(RuntimeError, match='TF_DECODER_ARGUMENTS'):
        views.bind_native_direction_feature_view(**args)
