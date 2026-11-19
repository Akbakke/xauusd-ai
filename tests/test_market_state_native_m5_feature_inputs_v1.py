"""Native-provider wiring/availability mechanics; numeric owners have own tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import market_state_native_m5_feature_inputs_v1 as owner
from gx1.features.volatility_squeeze_state_v1 import VolatilitySqueezeArtifactSet


def _fixture(tmp_path,monkeypatch,*,missing_macro_row=None,wrong_calibration_source=False):
    n=400;index=pd.date_range('2024-05-01',periods=n,freq='5min',tz='UTC')
    close=2000.+np.sin(np.arange(n)/9.)
    raw=pd.DataFrame(index=index)
    for prefix,spread in (('',0.),('bid_',-.1),('ask_',.1)):
        for suffix,delta in (('open',0.),('high',1.),('low',-1.),('close',0.)):
            raw[prefix+suffix]=close+spread+delta
    raw['volume']=1+np.arange(n)%37;raw=raw.loc[:,list(owner.RAW_COLUMNS)]
    path=tmp_path/'MANIFEST.json'
    native={'schema_version':'xau_canonical_native_source_v3','instrument':'XAU_USD','timeframe':'M5',
        'source_granularity':'M5','source_kind':'oanda_native_mba_candles','prices':'MBA',
        'bar_duration_seconds':300,'decision_available_offset_seconds':300,'timestamp_semantics':'bar_start_utc',
        'completion_field':'complete','completion_value':True,'year_sha256':{'year=2024':'a'*64}}
    path.write_text(json.dumps(native));ref={'path':str(path),'sha256':owner.hashlib.sha256(path.read_bytes()).hexdigest()}
    spec={'manifest':ref,'cutoff_ns':int(index[-1].value)+2*owner.BAR_NS,
        'parts':[{'path':str(tmp_path/'year=2024/part-000.parquet'),'sha256':'a'*64,'row_groups':[0]}]}
    source=owner.bind_native_m5_feature_source(raw,native_m5_manifest=ref,source_spec=spec)
    profile=owner.owners.compile_full_ta_entry_execution_profile();fields=profile['physical_field_names']
    frames={}
    for tf in owner.TIMEFRAMES:
        if tf=='M5':clock=index
        else:
            width=int(owner.htf.MULTI_TF_SHIFT[tf].value)
            last=(spec['cutoff_ns']-width-1)//width*width
            clock=pd.DatetimeIndex(pd.to_datetime(last-np.arange(399,-1,-1,dtype=np.int64)*width,utc=True))
        row=np.arange(len(clock))[:,None];column=np.arange(len(fields['mtf']))[None,:]
        values=(1.+np.sin(row*.07+column*.11)).astype(np.float32)
        for j,name in enumerate(fields['mtf']):
            if name in owner.norm.MTF_SEMANTIC_BINARY_FIELDS:values[:,j]=(np.arange(len(clock))%2).astype(np.float32)
            if name in owner.norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS:values[:,j]=np.arange(len(clock))%5
            if name in owner.norm.signed_ternary_unit_fields('mtf_'+tf.lower()):values[:,j]=np.arange(len(clock))%3-1
        frame=pd.DataFrame(values,index=clock,columns=fields['mtf'])
        scalar_fields=owner.htf.MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[tf]
        scalars=(np.arange(len(clock),dtype=np.float32)[:,None]+np.arange(len(scalar_fields),dtype=np.float32)[None,:])
        import torch
        from tests.test_entry_v10_ctx_model_shapes import _valid_range_observation_input
        values=_valid_range_observation_input(torch.from_numpy(values),fields['mtf']).numpy()
        frame=pd.DataFrame(values,index=clock,columns=fields['mtf'],copy=False)
        frame.attrs.update(ts_int64=clock.asi8.copy(),feats_np=values,causal_warmup_rows=0,
            htf_feature_contract=owner.htf.HTF_V4_MATRIX_CONTRACT,
            model_native_mtf_scalar_fields_v4=scalar_fields,model_native_mtf_scalars_np_v4=scalars,
            model_native_mtf_scalar_warmup_rows_v4=0,
            model_native_mtf_scalar_contract_v4=owner.htf.MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4)
        frames[tf]=frame
    def build(*args,**kwargs):
        scope=kwargs.get('retained_model_scope')
        return {tf:owner.htf._retain_model_native_feature_frame_v4(frame,timeframe=tf,scope=scope)
                for tf,frame in frames.items()}
    monkeypatch.setattr(owner.htf,'build_multi_tf_per_bar_features_v4',build)
    native_ref=({**ref,'sha256':'f'*64} if wrong_calibration_source else ref)
    provenance={'native_m5_manifest_artifact':native_ref['path'],'native_m5_manifest_sha256':native_ref['sha256']}
    registry={'contract_sha256':'b'*64,'entry_m5':{'seq_len':96,'trendline_band_atr':.3},
        'declared_train_window_start':'2009-06-01T00:00:00+00:00',
        'declared_train_window_end':'2013-01-01T22:00:00+00:00',
        'provenance':{'entry_m5_trendline_band':{'source_provenance':provenance}}}
    monkeypatch.setattr(owner.htf,'require_native_m5_registry_constants',lambda x:x)
    squeeze_params={'M5':{'source_provenance':provenance,
        'declared_train_window_start':registry['declared_train_window_start'],
        'declared_train_window_end':registry['declared_train_window_end']}}
    artifacts=VolatilitySqueezeArtifactSet(tmp_path/'SQUEEZE.json','c'*64,'d'*64,'e'*64,
        squeeze_params,{},clocks=owner.TIMEFRAMES)
    monkeypatch.setattr(owner,'require_volatility_squeeze_artifact_set',lambda x:x)
    monkeypatch.setattr(VolatilitySqueezeArtifactSet,'binding',lambda self:{'native_five_test_binding':'c'*64})
    def attach(frame,**kwargs):
        for j,name in enumerate(fields['ctx_cont']):
            if name not in owner.macro.D1_MACRO_CONTEXT_FIELDS:
                positions=raw.index.get_indexer(frame.index)
                if name in owner.norm.signed_ternary_unit_fields('ctx_cont'):
                    frame[name]=(positions%3-1).astype(np.float32)
                elif name in owner.norm.native_m5_binary_fields('ctx_cont',fields['ctx_cont']):
                    frame[name]=(positions%2).astype(np.float32)
                else:
                    frame[name]=np.cos(positions*.03+j*.17).astype(np.float32)
    monkeypatch.setattr(owner.htf,'attach_model_native_mtf_scalars_v4',attach)
    monkeypatch.setattr(owner.htf,'project_multi_tf_v4_scalars',lambda *a,**k:{})
    from gx1.features import trendline_registry_v1 as trend
    def local(source,**kwargs):
        values=np.arange(len(source),dtype=np.float64)[:,None]+np.arange(len(trend.TRENDLINE_REGISTRY_FEATURE_NAMES_V1))[None,:]
        return pd.DataFrame(values,index=source.index,columns=trend.TRENDLINE_REGISTRY_FEATURE_NAMES_V1),None
    monkeypatch.setattr(trend,'compute_trendline_registry_features_v1',local)
    def macro(times,components):
        positions=raw.index.get_indexer(pd.DatetimeIndex(times)-pd.Timedelta(minutes=5))
        frame=pd.DataFrame({name:np.sin(positions*.07+j*.13)
                            for j,name in enumerate(owner.macro.D1_MACRO_CONTEXT_FIELDS)})
        if missing_macro_row is not None:frame.loc[positions==missing_macro_row,owner.macro.D1_MACRO_CONTEXT_FIELDS[0]]=np.nan
        frame.insert(0,'decision_time',times);return frame
    monkeypatch.setattr(owner.macro,'build_d1_macro_context',macro)
    monkeypatch.setattr(owner.macro,'d1_macro_context_components',lambda spec:{'test_fixture':True})
    return raw,source,registry,artifacts,frames


def _build(fixture,**kwargs):
    raw,source,registry,artifacts,_=fixture
    scope={'issue_start_ns':int(raw.index[0].value)+owner.BAR_NS,
           'issue_end_ns':source['source_spec']['cutoff_ns'],**kwargs}
    return owner.build_native_m5_feature_surfaces(raw,source_binding=source,v29_registry_constants=registry,
        volatility_squeeze_artifacts=artifacts,**scope)


def test_raw_provider_routes_every_canonical_field_and_all_nine_forward_keys(tmp_path,monkeypatch):
    fixture=_fixture(tmp_path,monkeypatch);surfaces=_build(fixture)
    provider=owner.NativeM5FeatureProvider(surfaces)
    fields=provider.profile['physical_field_names']
    assert {k:len(v) for k,v in fields.items()}=={'signal':318,'ctx_cont':132,'ctx_cat':1,'mtf':256}
    assert set(surfaces['field_lineage'])==set(fields['signal'])
    assert len(provider.profile['local_specialists'])==8 and len(provider.profile['macro_context_fields'])==61
    assert not provider.input_ready[:95].any() and provider.input_ready[95:].all()
    selected=np.array([150,399],np.int64);batch=provider.gather(selected,max_rows=2)
    assert set(batch)=={'seq_x','snap_x','ctx_cont','ctx_cat','seq_m5','seq_m15','seq_h1','seq_h4','seq_d1'}
    np.testing.assert_array_equal(batch['seq_x'][:,-1],batch['snap_x'])
    np.testing.assert_array_equal(batch['snap_x'][:,provider.profile['temporal_alias_signal_indices']],
                                  batch['ctx_cont'][:,provider.profile['temporal_alias_ctx_cont_indices']])
    for tf in owner.TIMEFRAMES:
        length=owner.WINDOWS[tf];assert batch['seq_'+tf.lower()].shape==(2,length,256)
        for i,row in enumerate(selected):
            closes=provider.tf_times[tf]+int(owner.htf.MULTI_TF_SHIFT[tf].value)
            expected=np.flatnonzero(closes<=provider.issue_ns[row])[-length:]
            np.testing.assert_array_equal(batch['seq_'+tf.lower()][i],provider.tf_values[tf][expected])
    report=provider.support_report(selected)
    assert report['input_ready_rows']==2
    assert all(all(count==s['physical_rows'] for count in s['observed_counts']) for s in report['surfaces'].values())


def test_missing_macro_is_preserved_and_removes_exactly_consuming_local_windows(tmp_path,monkeypatch):
    surfaces=_build(_fixture(tmp_path,monkeypatch,missing_macro_row=200))
    provider=owner.NativeM5FeatureProvider(surfaces)
    name='ctx_cont.'+owner.macro.D1_MACRO_CONTEXT_FIELDS[0]
    col=provider.profile['physical_field_names']['signal'].index(name)
    assert np.isnan(surfaces['signal'][200,col]) and not surfaces['observed'][200,col]
    assert provider.input_ready[199] and not provider.input_ready[200:296].any() and provider.input_ready[296]
    report=provider.readiness_report(np.arange(95,400,dtype=np.int64))
    assert report['failed_conditions']['current_features_observed']==1
    assert report['failed_conditions']['local_window_observed']==96
    assert report['input_not_ready']==96
    with pytest.raises(RuntimeError,match='BATCH_NOT_OBSERVED'):
        provider.gather(np.array([200],np.int64),max_rows=1)


def test_calibration_source_mismatch_rejected_before_feature_computation(tmp_path,monkeypatch):
    fixture=_fixture(tmp_path,monkeypatch,wrong_calibration_source=True)
    monkeypatch.setattr(owner.htf,'build_multi_tf_per_bar_features_v4',lambda *a,**k:pytest.fail('computed mixed-source features'))
    with pytest.raises(RuntimeError,match='CALIBRATION_NATIVE_SOURCE_JOIN'):_build(fixture)


def test_legacy_six_clock_artifacts_rejected_before_nested_squeeze_reads(tmp_path,monkeypatch):
    raw,source,registry,artifacts,frames=_fixture(tmp_path,monkeypatch)
    from dataclasses import replace
    from gx1.features.volatility_squeeze_state_v1 import VOLATILITY_SQUEEZE_CLOCKS
    artifacts=replace(artifacts,clocks=VOLATILITY_SQUEEZE_CLOCKS)
    monkeypatch.setattr(owner,'require_volatility_squeeze_artifact_set',lambda *a,**k:pytest.fail('M1 artifact read'))
    with pytest.raises(RuntimeError,match='NATIVE_FIVE_SQUEEZE_REQUIRED'):
        owner.build_native_m5_feature_surfaces(raw,source_binding=source,v29_registry_constants=registry,
            volatility_squeeze_artifacts=artifacts,issue_start_ns=int(raw.index[0].value)+owner.BAR_NS,
            issue_end_ns=source['source_spec']['cutoff_ns'])


def test_raw_source_cutoff_and_bytes_cannot_be_relabelled(tmp_path,monkeypatch):
    raw,source,registry,artifacts,_=_fixture(tmp_path,monkeypatch)
    changed=raw.copy();changed.iloc[-1,0]+=.01
    with pytest.raises(RuntimeError,match='RAW_SOURCE_REBIND'):
        owner.build_native_m5_feature_surfaces(changed,source_binding=source,v29_registry_constants=registry,
            volatility_squeeze_artifacts=artifacts,issue_start_ns=int(raw.index[0].value)+owner.BAR_NS,
            issue_end_ns=source['source_spec']['cutoff_ns'])
    spec=source['source_spec'];spec={**spec,'cutoff_ns':int(raw.index[-1].value)+owner.BAR_NS}
    with pytest.raises(RuntimeError,match='SOURCE_EXCLUSIVE_CLOSE_CUTOFF'):
        owner.bind_native_m5_feature_source(raw,native_m5_manifest=source['native_m5_manifest'],source_spec=spec)


def test_fresh_normalization_uses_unique_physical_consumption_union(tmp_path,monkeypatch):
    provider=owner.NativeM5FeatureProvider(_build(_fixture(tmp_path,monkeypatch)))
    seen={}
    def fit(part,*,surface,field_names,**kwargs):
        seen[surface]=part.row_indices.copy()
        return {'fit_row_count':len(part.row_indices),'field_names':list(field_names)}
    monkeypatch.setattr(owner.norm,'fit_surface_normalization',fit)
    monkeypatch.setattr(owner.norm,'share_temporal_alias_stats_from_signal',lambda ctx,*a,**k:ctx)
    monkeypatch.setattr(owner.norm,'fit_ctx_cat_contract',lambda part,**k:{'fit_row_count':len(part.row_indices)})
    monkeypatch.setattr(owner.norm,'build_native_m5_input_normalization_contract',lambda **kwargs:kwargs)
    rows=np.array([150,151,300],np.int64);cutoff=int(provider.issue_ns[-1]+owner.BAR_NS)
    result=owner.fit_native_m5_feature_normalization(provider,fit_source_rows=rows,cutoff_ns=cutoff)
    expected=np.unique(np.concatenate([np.arange(r-95,r+1) for r in rows]))
    np.testing.assert_array_equal(seen['signal'],expected)
    np.testing.assert_array_equal(seen['ctx_cont'],rows)
    for tf in owner.TIMEFRAMES:
        expected=np.unique(np.concatenate([np.arange(provider.tf_tails[tf][r]-owner.WINDOWS[tf]+1,
                                                     provider.tf_tails[tf][r]+1) for r in rows]))
        np.testing.assert_array_equal(seen['mtf_'+tf.lower()],expected)
    assert result['fit_population']['issue_rows']==3
    assert result['feature_provider_contract_sha256']==owner.native_m5_feature_provider_contract()['contract_sha256']


def test_binding_rejects_old_metadata_and_future_masks_without_any_file_read(monkeypatch):
    contract=owner.native_m5_feature_provider_contract()
    value={'schema_version':owner.BINDING_SCHEMA,'source_bar_seconds':300,'timeframes':list(owner.TIMEFRAMES),
        'source_clock_contract':owner.CLOCK_CONTRACT,'feature_provider_contract_sha256':contract['contract_sha256'],
        'source_axes_sha256':'a'*64,'input_normalization_sha256':'b'*64,
        'execution_profile':contract['execution_profile'],'warmstart_used':False}
    monkeypatch.setattr(Path,'read_bytes',lambda *a:pytest.fail('unexpected file IO'))
    assert owner.require_native_m5_input_binding(owner._seal(value))['warmstart_used'] is False
    for key in ('m1_context','future_eligible','parent_legacy_view'):
        with pytest.raises(RuntimeError,match='INPUT_BINDING_KEYS'):
            owner.require_native_m5_input_binding(owner._seal({**value,key:'unused'}))


def test_fitted_parameters_are_not_available_to_earlier_forecasts(tmp_path,monkeypatch):
    fixture=_fixture(tmp_path,monkeypatch)
    end='2024-05-01T10:00:00+00:00'
    fixture[2]['declared_train_window_end']=end
    fixture[3].params_by_clock['M5']['declared_train_window_end']=end
    surfaces=_build(fixture);provider=owner.NativeM5FeatureProvider(surfaces)
    assert surfaces['parameter_available_ns']==int(pd.Timestamp(end).value)
    assert not provider.input_ready[provider.issue_ns<int(pd.Timestamp(end).value)].any()
    assert provider.input_ready[provider.issue_ns>=int(pd.Timestamp(end).value)].all()


def test_normalization_cannot_fit_reused_development_val(tmp_path,monkeypatch):
    provider=owner.NativeM5FeatureProvider(_build(_fixture(tmp_path,monkeypatch)))
    monkeypatch.setattr(owner.norm,'fit_surface_normalization',lambda *a,**k:pytest.fail('statistics touched VAL'))
    with pytest.raises(RuntimeError,match='FIT_POPULATION'):
        owner.fit_native_m5_feature_normalization(provider,fit_source_rows=np.array([150,151],np.int64),
            cutoff_ns=owner.norm.NATIVE_M5_TRAIN_FIT_CUTOFF_NS+owner.BAR_NS)


def test_real_provider_normalization_fit_alias_share_and_contract_boundary(tmp_path,monkeypatch):
    """Actual seven-surface fitter and validators, with canonical synthetic values."""
    import torch
    from tests.test_entry_v10_ctx_model_shapes import _valid_range_observation_input
    surfaces=_build(_fixture(tmp_path,monkeypatch))
    fields=surfaces['provider_contract']['field_names'];n=len(surfaces['bar_start_ns'])
    def matrix(names,rows,*,mtf=False):
        axis=np.arange(rows,dtype=np.float32)[:,None]
        columns=np.arange(len(names),dtype=np.float32)[None,:]
        values=(np.sin(axis*.17+columns*.13)*(.05+columns*.001)).astype(np.float32)
        domains=(owner.norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS if mtf
                 else owner.norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS)
        for name,domain in domains.items():
            if name in names:values[:,names.index(name)]=np.asarray(domain)[np.arange(rows)%len(domain)]
        surface='mtf_m5' if mtf else 'ctx_cont' if list(names)==list(fields['ctx_cont']) else 'signal'
        for name in owner.norm.native_m5_binary_fields(surface,names):
            values[:,names.index(name)]=np.arange(rows)%2
        for name in owner.norm.signed_ternary_unit_fields(surface):
            if name in names:values[:,names.index(name)]=np.arange(rows)%3-1
        return _valid_range_observation_input(torch.from_numpy(values),names).numpy()
    signal=matrix(fields['signal'],n);context=matrix(fields['ctx_cont'],n)
    profile=owner.owners.compile_full_ta_entry_execution_profile()
    signal[:,list(profile['temporal_alias_signal_indices'])]=context[:,list(profile['temporal_alias_ctx_cont_indices'])]
    surfaces['signal']=owner._owned(signal,np.float32);surfaces['ctx_cont']=owner._owned(context,np.float32)
    surfaces['observed']=owner._owned(np.isfinite(signal),np.bool_)
    for tf,frame in list(surfaces['multi_tf_sources'].items()):
        values=owner._owned(matrix(list(frame.columns),len(frame),mtf=True),np.float32)
        replacement=pd.DataFrame(values,index=frame.index,columns=frame.columns,copy=False)
        replacement.attrs=dict(frame.attrs);replacement.attrs['feats_np']=values
        surfaces['multi_tf_sources'][tf]=replacement
    provider=owner.NativeM5FeatureProvider(surfaces)
    selected=np.arange(150,400,dtype=np.int64)
    cutoff=int(provider.issue_ns[-1]+owner.BAR_NS)
    normalization=owner.fit_native_m5_feature_normalization(provider,fit_source_rows=selected,cutoff_ns=cutoff)
    assert owner.norm.require_native_m5_input_normalization_contract(normalization)==normalization
    assert len(normalization['temporal_aliases'])==132
    for alias in normalization['temporal_aliases']:
        assert set(alias)=={'signal_index','ctx_cont_index','signal_field','ctx_cont_field','specialist'}
        assert alias['specialist']==owner.owners.classify_entry_specialist_feature(alias['signal_field'])
        for key in ('center','scale','binary_mask','categorical_mask','scale_source'):
            assert normalization['surfaces']['signal'][key][alias['signal_index']]==normalization['surfaces']['ctx_cont'][key][alias['ctx_cont_index']]
    batch=provider.gather(selected[-2:],max_rows=2)
    for key,array in batch.items():
        if key=='ctx_cat':continue
        surface='signal' if key in ('seq_x','snap_x') else 'ctx_cont' if key=='ctx_cont' else 'mtf_'+key[4:]
        transformed=owner.norm.apply_surface_normalization(array,normalization['surfaces'][surface])
        assert np.isfinite(transformed).all() and transformed.shape==array.shape
    binding=provider.input_binding(normalization)
    assert binding['input_normalization_sha256']==normalization['contract_sha256']


def test_bounded_materialization_preserves_full_state_values_global_ids_and_real_statistics(tmp_path,monkeypatch):
    fixture=_fixture(tmp_path,monkeypatch)
    from gx1.features import trendline_registry_v1 as trend
    owner_calls=[];original_build=owner.htf.build_multi_tf_per_bar_features_v4
    original_trend=trend.compute_trendline_registry_features_v1
    def full_history_build(source,**kwargs):
        owner_calls.append(('mtf',len(source)))
        return original_build(source,**kwargs)
    def full_history_trend(source,**kwargs):
        owner_calls.append(('local_trend',len(source)))
        return original_trend(source,**kwargs)
    monkeypatch.setattr(owner.htf,'build_multi_tf_per_bar_features_v4',full_history_build)
    monkeypatch.setattr(trend,'compute_trendline_registry_features_v1',full_history_trend)
    full=owner.NativeM5FeatureProvider(_build(fixture))
    bounded_surfaces=_build(fixture,issue_start_ns=int(full.issue_ns[200]),issue_end_ns=int(full.issue_ns[300]))
    bounded=owner.NativeM5FeatureProvider(bounded_surfaces)
    assert owner_calls==[('mtf',400),('local_trend',400),('mtf',400),('local_trend',400)]
    np.testing.assert_array_equal(bounded.issue_ns,full.issue_ns)
    np.testing.assert_array_equal(bounded.source_row,full.source_row)
    np.testing.assert_array_equal(bounded.retained_source_rows,np.arange(105,300))
    np.testing.assert_array_equal(bounded_surfaces['retained_issue_rows'],np.arange(200,300))
    assert not bounded.input_ready[:200].any() and not bounded.input_ready[300:].any()
    assert bounded.input_ready[200:300].all()
    for key in ('signal','ctx_cont','ctx_cat','observed'):
        np.testing.assert_array_equal(bounded_surfaces[key],full.surfaces[key][105:300])
        assert not np.shares_memory(bounded_surfaces[key],full.surfaces[key])
    for tf in owner.TIMEFRAMES:
        frame=bounded_surfaces['multi_tf_sources'][tf]
        original_ids=frame.attrs['owner_row_indices']
        positions=np.searchsorted(full.tf_source_rows[tf],original_ids)
        np.testing.assert_array_equal(full.tf_source_rows[tf][positions],original_ids)
        np.testing.assert_array_equal(frame.attrs['feats_np'],full.tf_values[tf][positions])
        np.testing.assert_array_equal(frame.attrs['ts_int64'],full.tf_times[tf][positions])
        np.testing.assert_array_equal(frame.attrs['model_native_mtf_scalars_np_v4'],
            full.surfaces['multi_tf_sources'][tf].attrs['model_native_mtf_scalars_np_v4'][positions])
        assert np.shares_memory(frame.to_numpy(copy=False),frame.attrs['feats_np'])
        assert not np.shares_memory(frame.attrs['feats_np'],full.tf_values[tf])
    chosen=np.array([200,299],np.int64)
    for key,values in bounded.gather(chosen,max_rows=2).items():
        np.testing.assert_array_equal(values,full.gather(chosen,max_rows=2)[key])
    selected=np.arange(200,300,dtype=np.int64);cutoff=int(full.issue_ns[300])
    full_norm=owner.fit_native_m5_feature_normalization(full,fit_source_rows=selected,cutoff_ns=cutoff)
    bounded_norm=owner.fit_native_m5_feature_normalization(bounded,fit_source_rows=selected,cutoff_ns=cutoff)
    assert full_norm['surfaces']==bounded_norm['surfaces']
    assert full_norm['ctx_cat']==bounded_norm['ctx_cat']
    assert full_norm['fit_population']==bounded_norm['fit_population']
    assert full_norm['source_axes_sha256']!=bounded_norm['source_axes_sha256']
    assert bounded.support_report(selected)['retained_scope']['local_rows']==195


def test_bounded_local_matrix_allocations_follow_consumption_not_complete_source(tmp_path,monkeypatch):
    fixture=_fixture(tmp_path,monkeypatch)
    raw=fixture[0];stack=owner.np.column_stack;allocations=[]
    def bounded_stack(arrays):
        arrays=list(arrays)
        if len(arrays)==318:allocations.append((len(arrays[0]),318))
        return stack(arrays)
    monkeypatch.setattr(owner.np,'column_stack',bounded_stack)
    surfaces=_build(fixture,issue_start_ns=int(raw.index[200].value)+owner.BAR_NS,
                    issue_end_ns=int(raw.index[300].value)+owner.BAR_NS)
    assert allocations==[(195,318)]
    assert surfaces['signal'].nbytes==195*318*4
    assert surfaces['ctx_cont'].nbytes==195*132*4
    assert surfaces['observed'].nbytes==195*318
    assert surfaces['bar_start_ns'].shape==(400,)
    assert surfaces['source_binding']['rows']==400
