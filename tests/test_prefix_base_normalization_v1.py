"""Actual base/MTF fit owners on synthetic prefix data; no market/model run."""
import copy
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.scripts import materialize_unified_exit_pilot_normalization_inputs_v1 as inputs
from gx1.scripts import materialize_unified_exit_pilot_base_normalization_v1 as base
from tests.test_materialize_unified_exit_pilot_normalization_inputs_v1 import (
    _population_fixture, _fixed, _write_json, _sha256_file,
)
from tests.test_entry_v10_input_normalization_fit import _signal_names, _fill_semantic_categoricals, _mtf_frame, _mtf_values
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM, MODEL_NATIVE_CTX_CONT_FIELDS, MODEL_NATIVE_CTX_CAT_FIELDS
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS, MTF_SEMANTIC_CATEGORICAL_DOMAINS, MTF_SEMANTIC_BINARY_FIELDS, SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4, MULTI_TF_SHIFT, MULTI_TF_RESAMPLE_RULES, multi_tf_bar_label


def _prepare(root, monkeypatch, *, poison=False):
    root.mkdir()
    f, kwargs, _ = _population_fixture(root)
    names = _signal_names()
    cutoff = pd.Timestamp('2025-06-01T00:06Z').value
    rows_path = root/'fit_rows.npy'; np.save(rows_path, np.array([0, 1], dtype='<i8'))
    def surface(path, times, bar_ns):
        signal = (np.arange(len(times))[:, None]*.1 + np.arange(MODEL_NATIVE_SIGNAL_DIM)[None, :]*.01).astype('float32')
        _fill_semantic_categoricals(signal, names, axis_rows=len(times))
        ctx = (np.arange(len(times))[:, None]*.013 + np.arange(len(MODEL_NATIVE_CTX_CONT_FIELDS))[None, :]*.005).astype('float32')
        for name,domain in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS.items():
            ctx[:, list(MODEL_NATIVE_CTX_CONT_FIELDS).index(name)] = np.asarray(domain)[np.arange(len(times)) % len(domain)]
        for alias in base._derive_temporal_aliases(names):
            ctx[:, alias['ctx_cont_index']] = signal[:, alias['signal_index']]
        cat = np.zeros((len(times), len(MODEL_NATIVE_CTX_CAT_FIELDS)), dtype='int64')
        future = times.asi8 + bar_ns > cutoff
        if poison:
            cols = [i for i,n in enumerate(names) if n not in SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS]
            signal[np.ix_(future, cols)] += 5000
            ctx_cols = [i for i,n in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS) if n not in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS]
            ctx[np.ix_(future, ctx_cols)] += 5000
            for alias in base._derive_temporal_aliases(names):
                ctx[:, alias['ctx_cont_index']] = signal[:, alias['signal_index']]
        table = pa.table({'time':pa.array(times), 'signal':_fixed(signal, signal.shape[1], pa.float32()),
            'ctx_cont':_fixed(ctx, ctx.shape[1], pa.float32()), 'ctx_cat':_fixed(cat, cat.shape[1], pa.int64())})
        pq.write_table(table,path)
        return signal,ctx,cat
    sig,ctx,cat = surface(f['surface'], f['times'], 300_000_000_000)
    positions = np.array([95,96,97])
    child_table = pa.table({'time':pa.array(f['times'][positions]),
        'snap':_fixed(sig[positions], MODEL_NATIVE_SIGNAL_DIM, pa.float32()),
        'ctx_cont':_fixed(ctx[positions], ctx.shape[1], pa.float32()),
        'ctx_cat':_fixed(cat[positions], cat.shape[1], pa.int64())})
    pq.write_table(child_table, f['child_path'])
    f['child']['decision']='PASS'
    f['child']['splits']['train'].update(rows=3, parquet_sha256=_sha256_file(f['child_path']),
        clock_sha256=inputs._clock_hash(f['times'][positions].asi8))
    m1_times = pd.DatetimeIndex(pd.read_parquet(kwargs['m1_source_path'])['time']).as_unit('ns')
    surface(kwargs['m1_feature_base_path'], m1_times, 60_000_000_000)
    mf=inputs._read_json(kwargs['m1_feature_base_manifest_path'],'FIXTURE')
    mf['output_parquet_sha256']=_sha256_file(kwargs['m1_feature_base_path']);_write_json(kwargs['m1_feature_base_manifest_path'],mf)
    f['parent']['feature_contract']={'signal_bridge_fields':names}
    parent=root/'parent.json';_write_json(parent,f['parent'])
    admission=root/'admission.json';_write_json(admission,f['child'])
    cutoff_args=dict(fit_entry_rows_path=rows_path,fit_cutoff_time_ns=cutoff)
    pop=inputs.build_train_normalization_population_witness(**kwargs, **cutoff_args)
    pop_path=root/'population.json';_write_json(pop_path,pop)
    view={'decision':'PASS','parent_train_manifest':{'path':str(parent),'sha256':_sha256_file(parent)},
        'train_normalization_population_witness':{'contract_sha256':pop['contract_sha256'],'file_sha256':_sha256_file(pop_path)},
        'normalization_fit_population':pop['normalization_fit_population']}
    view_path=root/'view.json';_write_json(view_path,view)
    cache={}
    for tf,rule in MULTI_TF_RESAMPLE_RULES.items():
        end=multi_tf_bar_label(pd.DatetimeIndex([pd.Timestamp(cutoff,tz='UTC')+pd.Timedelta(days=2)]),tf)[0]
        times=pd.date_range(end=end,periods=1000,freq=rule).as_unit('ns')
        values=_mtf_values(len(times))
        if poison:
            future=times.asi8 + MULTI_TF_SHIFT[tf].value > cutoff
            cols=[i for i,n in enumerate(MULTI_TF_PER_BAR_FEATURES_V4) if n not in MTF_SEMANTIC_CATEGORICAL_DOMAINS and n not in MTF_SEMANTIC_BINARY_FIELDS]
            values[np.ix_(future,cols)]+=5000
        cache[tf]=_mtf_frame(times.asi8,values)
    cache_dir=root/'cache';cache_dir.mkdir()
    _write_json(cache_dir/'manifest.json',{'builder_version':'synthetic', 'feature_names':list(MULTI_TF_PER_BAR_FEATURES_V4)})
    monkeypatch.setattr(base,'load_multi_tf_v4_cache',lambda path:cache)
    call=dict(child_admission_path=admission,normalization_view_path=view_path,population_witness_path=pop_path,
        m5_feature_path=f['surface'],m5_prebuilt_path=f['surface'],m1_feature_path=kwargs['m1_feature_base_path'],
        mtf_cache_dir=cache_dir,output_path=root/'normalization.json',publish=False,**cutoff_args)
    return call,pop,view


def test_actual_fit_surfaces_ignore_all_later_features(tmp_path,monkeypatch):
    call,pop,_=_prepare(tmp_path/'original',monkeypatch)
    original=base.fit_base(**call)
    poisoned,pop_changed,_=_prepare(tmp_path/'poisoned',monkeypatch,poison=True)
    changed=base.fit_base(**poisoned)
    assert pop['train_entry_decision_rows']==2 and pop['exit_m1_current_unique_rows']==6
    assert pop['entry_m5_local_unique_rows']==97
    assert pop['m1_source']['right_censor_time_utc_exclusive']=='2025-06-02T00:00:00Z'
    assert pop['normalization_fit_population']['maximum_observed_decision_time_ns']==call['fit_cutoff_time_ns']
    for name in base.EXPECTED_SURFACES:
        assert original['contract']['surfaces'][name]==changed['contract']['surfaces'][name],name
    assert original['contract']['ctx_cat']==changed['contract']['ctx_cat']
    assert original['contract']['lineage']['entry_train_decision_row_count']==2
    assert original['contract']['lineage']['exit_train_decision_row_count']==6
    assert pop['m1_feature_base']['sha256']!=pop_changed['m1_feature_base']['sha256']


@pytest.mark.parametrize('mutation',['missing_opt_in','wrong_cutoff','wrong_rows','future_m5','future_m1','changed_witness'])
def test_bad_prefix_binding_rejected_before_statistics(tmp_path,monkeypatch,mutation):
    call,pop,view=_prepare(tmp_path/'case',monkeypatch)
    monkeypatch.setattr(base,'fit_surface_normalization',lambda *a,**k:pytest.fail('fit must not start'))
    if mutation=='missing_opt_in':
        call['fit_entry_rows_path']=None;call['fit_cutoff_time_ns']=None
    elif mutation=='wrong_cutoff':call['fit_cutoff_time_ns']-=1
    elif mutation=='wrong_rows':np.save(call['fit_entry_rows_path'],np.array([1],dtype='<i8'))
    elif mutation=='changed_witness':pop['entry_m5_local_unique_rows']+=1
    else:
        key='entry_m5_local_intervals' if mutation=='future_m5' else 'exit_m1_current_intervals'
        pop[key][-1]['end_row_exclusive']+=1
        pop['contract_sha256']=inputs._canonical_sha256({k:v for k,v in pop.items() if k!='contract_sha256'})
        view['train_normalization_population_witness']['contract_sha256']=pop['contract_sha256']
    _write_json(call['population_witness_path'],pop)
    if mutation!='changed_witness':view['train_normalization_population_witness']['file_sha256']=_sha256_file(call['population_witness_path'])
    _write_json(call['normalization_view_path'],view)
    with pytest.raises(RuntimeError,match='PREFIX_'):
        base.fit_base(**call)


def test_population_cutoff_uses_close_and_rejects_later_entry(tmp_path):
    _,kwargs,_=_population_fixture(tmp_path)
    rows=tmp_path/'rows.npy';np.save(rows,np.array([0],dtype='<i8'))
    cutoff=pd.Timestamp('2025-06-01T00:02Z').value
    exact=inputs.build_train_normalization_population_witness(**kwargs,fit_entry_rows_path=rows,fit_cutoff_time_ns=cutoff)
    less=inputs.build_train_normalization_population_witness(**kwargs,fit_entry_rows_path=rows,fit_cutoff_time_ns=cutoff-1)
    assert exact['exit_m1_current_unique_rows']==2 and less['exit_m1_current_unique_rows']==1
    np.save(rows,np.array([1],dtype='<i8'))
    with pytest.raises(RuntimeError,match='PREFIX_ENTRY_AFTER_CUTOFF'):
        inputs.build_train_normalization_population_witness(**kwargs,fit_entry_rows_path=rows,fit_cutoff_time_ns=cutoff)
