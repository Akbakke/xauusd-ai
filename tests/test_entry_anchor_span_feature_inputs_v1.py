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


@pytest.fixture(scope='module')
def binding():
    parent=read(extension.extension.PARENT_ARTIFACT)
    base=read(extension.NORMALIZATION_EVIDENCE['expanded_base'])
    units=read(extension.NORMALIZATION_EVIDENCE['units_profile'])
    constructor=extension.build_anchor_span_constructor_metadata(read(extension.PARENT_CONSTRUCTOR_REF))
    packet=extension.build_anchor_span_prefix_packet(parent,base,unobserved_units_profile=units)
    proof=parent['prefix_qualification']['proof']['source_bindings']
    sources={key:ref(key) for key in views.SOURCE_REF_KEYS|views.TIME_QUERY_V2_SOURCE_REF_KEYS|views.ANCHOR_SPAN_SOURCE_REF_KEYS}
    sources.update(input_qualification=extension.PARENT_QUALIFICATION_REF,
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


def test_exact_metadata_dispatch_preserves_parent_gc_and_new_geometry(binding):
    view=views.bind_native_direction_feature_view(**binding)
    assert views.bind_native_direction_feature_view(**binding,expected=view)==view
    assert view['physical_widths']==dict(signal=317,ctx_cont=132,ctx_cat=1,mtf=255)
    packet=binding['prefix_normalization']
    assert packet['gc_mtf_history']==packet['parent_prefix_normalization']['gc_mtf_history']
    assert view['constructor_scalar_config']==extension.SCALARS
    assert view['prefix_normalization']['ta_base_normalization_sha256']==extension.TA_CONTRACT_SHA
    assert view['anchor_span_input_extension']['execution_authorized'] is False


@pytest.mark.parametrize('fault',['old_units','old_packet','changed_old_stat','changed_constructor','missing_overlay','other_decoder','marker_removed'])
def test_metadata_rejects_unbound_or_resealed_changes(binding,fault):
    args=deepcopy(binding)
    if fault=='old_units':args['unobserved_units_profile']=extension.units.mature_fold1_prefix_units_profile()
    elif fault=='old_packet':args['prefix_normalization']=args['prefix_normalization']['parent_prefix_normalization']
    elif fault=='changed_old_stat':
        packet=args['prefix_normalization'];packet['base_normalization']['surfaces']['signal']['center'][0]+=1
        packet['contract_sha256']=views._sha({k:v for k,v in packet.items() if k!='contract_sha256'})
    elif fault=='changed_constructor':args['constructor_metadata']['dropout']=.1
    elif fault=='missing_overlay':del args['source_bindings']['anchor_span_inputs_source']
    elif fault=='other_decoder':args['dedicated_non_m5_side']=True
    else:
        view=views.bind_native_direction_feature_view(**args);del view['anchor_span_input_extension']
        view['contract_sha256']=views._sha({k:v for k,v in view.items() if k!='contract_sha256'})
        with pytest.raises(RuntimeError):views._require_view(view,unobserved_units_profile=args['unobserved_units_profile'])
        return
    with pytest.raises(RuntimeError):views.bind_native_direction_feature_view(**args)


def readonly(value):
    value=np.ascontiguousarray(value);value.setflags(write=False);return value


def make_gather_case(monkeypatch, *, later=False):
    # Metadata identity is tested separately above; this small fixture isolates
    # real canonical window/cropping behavior with no market-value reads.
    monkeypatch.setattr(views,'_require_view',lambda view,**kw:view)
    c=prefix_case();fields,_,_=views._canonical()
    local_names=['chart.'+n for n in overlay.SPAN_FIELDS]
    source=c['entry_sequence_source']['signal']
    for j,name in enumerate(local_names):source[:,fields['signal'].index(name)]=np.arange(len(source))%11+j+1
    c['snap_x']=source[c['entry_sequence_source']['sample_positions']].copy()
    # Span additions are not historical context aliases; all old aliases stay.
    for tf,frame in c['multi_tf_sources'].items():
        values=np.array(frame.to_numpy(dtype=np.float32,copy=False),copy=True,order='C')
        for j,name in enumerate(overlay.SPAN_FIELDS):values[:,fields['mtf'].index(name)]=np.arange(len(values))%13+j+1
        rebuilt=pd.DataFrame(values,index=frame.index,columns=frame.columns,copy=False)
        rebuilt.attrs.update(frame.attrs)
        rebuilt.attrs['feats_np']=values
        c['multi_tf_sources'][tf]=rebuilt
    for key in ('parent_rows','bar_start_ns','decision_ns','snap_x','ctx_cont','ctx_cat'):c[key]=readonly(c[key])
    for key in ('signal','times_ns','sample_positions'):c['entry_sequence_source'][key]=readonly(c['entry_sequence_source'][key])
    for frame in c['multi_tf_sources'].values():
        frame.attrs['feats_np']=readonly(frame.to_numpy(dtype=np.float32,copy=False))
        frame.attrs['ts_int64']=readonly(frame.attrs['ts_int64'])
    base_view=dict(full_ta_feature_profile=True,full_axes=dict(rows=6),prefix=dict(rows=4,cutoff_ns=c['cutoff_ns']),max_batch_rows=2,contract_sha256='a'*64)
    original={k:c[k] for k in ('parent_rows','bar_start_ns','decision_ns','entry_sequence_source','snap_x','ctx_cont','ctx_cat','multi_tf_sources')}
    original.update(gc_surfaces=None,gc_layout=None,
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
        monkeypatch.setattr(overlay,'_require_later_coverage',lambda *a:later_binding)
        monkeypatch.setattr(overlay,'LATER_SCORE_END_NS',int(c['decision_ns'][-1])+views.BAR_NS)
        build.update(later_coverage={'synthetic_only':True},later_axes={
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


def later_metadata_case():
    """Synthetic axes only; no retained market values or large feature matrices."""
    start=overlay.LATER_SCORE_START_NS
    decisions=start+(np.arange(118249,dtype=np.int64)-29849)*views.BAR_NS
    axes=dict(parent_rows=readonly(3*np.arange(118249,dtype=np.int64)+2),
              bar_start_ns=readonly(decisions-views.BAR_NS),decision_ns=readonly(decisions))
    sha=lambda a:hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()
    view=dict(contract_sha256='a'*64,full_axes=dict(rows=118249,**{
        k+'_sha256':sha(a) for k,a in axes.items()}),
        prefix=dict(rows=29574,cutoff_ns=1727654400000000000,normalization_identity='unchanged'))
    populations={k:dict(selected_unique_row_count=3,selected_row_indices_sha256='d'*64,
                       available_ns_sha256='e'*64) for k in ('local',*overlay.TIMEFRAMES)}
    selected=(29849+np.array([i*1108//255 for i in range(256)],dtype=np.int64)[:,None]*16
              +np.arange(16,dtype=np.int64)).reshape(-1)
    coverage=dict(schema_version=overlay.LATER_COVERAGE_SCHEMA,status='COMPLETE',
        qualification=dict(path='/home/andre2/GX1_RUNS/SYNTHETIC_LATER/RESULT.json',sha256='f'*64),
        view_sha256=view['contract_sha256'],full_axes=deepcopy(view['full_axes']),
        normalization_prefix=deepcopy(view['prefix']),score_parent_start=29849,score_rows=17753,
        score_start_ns=overlay.LATER_SCORE_START_NS,score_end_exclusive_ns=overlay.LATER_SCORE_END_NS,
        batch_size=16,batch_indices=[i*1108//255 for i in range(256)],
        selected_parent_rows_sha256=sha(axes['parent_rows'][selected]),
        selected_decision_ns_sha256=sha(axes['decision_ns'][selected]),
        physical_populations=deepcopy(populations))
    return view,coverage,axes,populations,selected


def test_fixed_later_metadata_owns_only_selected_axes_without_changing_train_normalization():
    view,coverage,axes,populations,positions=later_metadata_case()
    before=deepcopy((view,coverage,populations))
    actual=overlay._require_later_coverage(view,coverage,axes,populations)
    assert (view,coverage,populations)==before
    assert len(positions)==4096 and positions[0]==29849 and positions[-1]==47592
    for key,expected in [('later_parent_positions',positions),
                         ('later_parent_rows',axes['parent_rows'][positions]),
                         ('later_decision_ns',axes['decision_ns'][positions])]:
        assert np.array_equal(actual[key],expected) and not actual[key].flags.writeable
        assert not any(np.shares_memory(actual[key],a) for a in axes.values())
    assert len(actual['later_coverage_sha256'])==64
    assert actual['later_qualification']==tuple(coverage['qualification'][k] for k in ('path','sha256'))
    coverage['batch_indices'][0]=1
    assert actual['later_parent_positions'][0]==29849


@pytest.mark.parametrize('fault',[
    'view','full_axes','normalization','population','batch_selection','score_count','boolean_count',
    'unfinished','ref_path','ref_sha','selected_axis_hash','mutable_axis','changed_axis','extra_key'])
def test_later_metadata_rejects_scope_axis_or_provenance_changes(fault):
    view,coverage,axes,populations,_=later_metadata_case()
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
