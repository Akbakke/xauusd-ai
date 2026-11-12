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


def make_gather_case(monkeypatch):
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
    original.update(gc_surfaces=None,gc_layout=None,batch_parent_rows=np.array([5,9],np.int64))
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
    rows=np.arange(5,105,dtype=np.int64)
    sidecar('local',rows,seq['times_ns'][rows],source[np.ix_(rows,[fields['signal'].index(n) for n in local_names])])
    for tf,frame in c['multi_tf_sources'].items():
        mtf_values[tf]=readonly(frame.attrs['feats_np'][:,kept_mtf]);mtf_times[tf]=frame.attrs['ts_int64']
        all_ids=[]
        for start in c['bar_start_ns'][:4]:
            windows=views.get_model_native_multi_tf_route_windows(c['multi_tf_sources'],decision_bar_start=pd.Timestamp(int(start),tz='UTC'),per_tf_seq_lens=dict(views.history.WINDOW_ROWS),route_timeframes=views.history.TIMEFRAMES,base_bar_duration=pd.Timedelta(seconds=300),return_row_indices=True)
            all_ids.extend(windows[tf][1].tolist())
        rows=np.unique(all_ids).astype(np.int64)
        sidecar(tf,rows,mtf_times[tf][rows],frame.attrs['feats_np'][np.ix_(rows,[fields['mtf'].index(n) for n in overlay.SPAN_FIELDS])])
    view={**base_view,'anchor_span_input_extension':{'synthetic_only':True}}
    build=dict(entry_sequence_source=seq,snap_x=snap,ctx_cont=c['ctx_cont'],ctx_cat=c['ctx_cat'],
        parent_signal_names=[fields['signal'][i] for i in keep],parent_mtf_names=[fields['mtf'][i] for i in kept_mtf],
        parent_mtf_values=mtf_values,parent_mtf_times=mtf_times,span_payloads=payloads,physical_populations=populations,unobserved_units_profile=None)
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
