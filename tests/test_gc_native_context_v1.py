"""Mask/normalization algebra only; real-data native wiring has its own gate."""
import numpy as np
import pytest
import torch
from gx1.features import gc_native_context_v1 as gc
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    fit_surface_normalization, apply_surface_normalization, invert_surface_normalization,
    require_surface_normalization,
)


def test_observed_population_excludes_missing_storage_and_roundtrips():
    names = [gc.VALUE_FIELDS[0], gc.OBSERVED_FIELDS[0]]
    raw = np.array([[0, 0], [10, 1], [20, 1], [30, 1], [0, 0]], np.float32)
    contract = fit_surface_normalization(raw, surface='ctx_cont', field_names=names)
    require_surface_normalization(contract, surface='ctx_cont', field_names=names)
    assert contract['center'][0] == 20
    assert contract['scale'][0] == 10
    assert contract['binary_mask'] == [0, 1]
    normalized = apply_surface_normalization(raw, contract)
    np.testing.assert_allclose(contract['train_transformed_min'], normalized.min(axis=0), atol=1e-7)
    np.testing.assert_allclose(contract['train_transformed_max'], normalized.max(axis=0), atol=1e-7)
    assert normalized[0, 0] == normalized[-1, 0] == 0
    np.testing.assert_allclose(invert_surface_normalization(normalized, contract), raw, atol=1e-6)
    tensor = torch.tensor(raw, requires_grad=True)
    preliminary = torch.asinh((tensor - torch.tensor(contract['center'])) / torch.tensor(contract['scale']))
    result = gc.mask_normalized_torch(tensor, preliminary, names)
    result[:, 0].sum().backward()
    assert tensor.grad[0, 0] == tensor.grad[-1, 0] == 0
    assert tensor.grad[1, 0] != 0


def test_nonzero_value_behind_absent_mask_is_rejected():
    names = [gc.VALUE_FIELDS[0], gc.OBSERVED_FIELDS[0]]
    raw = np.array([[5, 0], [10, 1], [20, 1]], np.float32)
    with pytest.raises(RuntimeError, match='MASKED_STORAGE'):
        fit_surface_normalization(raw, surface='ctx_cont', field_names=names)
    with pytest.raises(RuntimeError, match='MASKED_STORAGE'):
        gc.mask_normalized_torch(torch.tensor(raw), torch.tensor(raw), names)


def test_missing_physical_train_observations_fail_closed():
    names = [gc.VALUE_FIELDS[0], gc.OBSERVED_FIELDS[0]]
    with pytest.raises(RuntimeError, match='NO_PHYSICAL_TRAIN_OBSERVATIONS'):
        fit_surface_normalization(np.zeros((4, 2), np.float32), surface='ctx_cont', field_names=names)


def test_every_value_has_exactly_one_availability_and_specialist_owner():
    assert len(gc.VALUE_FIELDS) == len(gc.OBSERVED_FIELDS)
    assert set(gc.SPECIALIST_OWNERS) == set(gc.CONTEXT_FIELDS)
    assert len(gc.masking_pairs(gc.CONTEXT_FIELDS)) == len(gc.VALUE_FIELDS)
    assert len(gc.masking_pairs(['ctx_cont.' + n for n in gc.CONTEXT_FIELDS])) == len(gc.VALUE_FIELDS)
    with pytest.raises(RuntimeError, match='OBSERVATION_FIELD_MISSING'):
        gc.masking_pairs([gc.VALUE_FIELDS[0]])


def test_gc_liveness_labels_do_not_claim_dense_activity_or_learning():
    from gx1.contracts.entry_full_input_liveness_v1 import classify_field_status
    from gx1.features.gc_native_context_v1 import VALUE_FIELDS, OBSERVED_FIELDS
    stats = dict(row_count=100, finite_count=100, nonfinite_count=0,
                 min=0.0, max=0.0, std=0.0, value_range=0.0, active_count=0, active_rate=0.0)
    assert classify_field_status(split='train',surface='ctx_cont',field=VALUE_FIELDS[0],stats=stats)[0] == 'OBSERVED_MASKED_SOURCE'
    assert classify_field_status(split='train',surface='signal',field='ctx_cont.'+OBSERVED_FIELDS[0],stats=stats)[0] == 'STRUCTURAL_OBSERVATION_MASK'
    assert classify_field_status(split='train',surface='ctx_cont',field=OBSERVED_FIELDS[0],stats={**stats,'min':0.5,'max':0.5})[0] == 'FAIL'
    assert classify_field_status(split='train',surface='ctx_cont',field='hour_cos',stats=stats)[0] == 'FAIL'


def _bound_source_fixture(tmp_path):
    import json
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    clock=pd.DatetimeIndex(['2024-05-01T00:05Z','2024-05-01T00:10Z','2024-05-01T00:15Z']).asi8
    path=tmp_path/'TRADE_FEATURES_M5.parquet'
    pq.write_table(pa.table({'available_ns':clock,'value':[10.,20.,30.],'value_copy':[10.,20.,30.],
        'valid':[True,False,True],'snapshot':clock,'constant':[7.,7.,7.],
        'empty':[float('nan')]*3,'flag':[False,True,True],
        'observed_profile_snapshot_age_minutes':[0.,0.,0.],'observed_spot_basis_age_minutes':[0.,0.,0.]}),path)
    binding={'path':str(path),'sha256':gc._stream_sha256(path)}
    terminal=tmp_path/'TERMINAL.json'
    terminal.write_text(json.dumps({'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,
        'artifacts':{path.name:{'sha256':binding['sha256'],'bytes':path.stat().st_size}}}))
    return clock,path,binding,{'path':str(terminal),'sha256':gc._stream_sha256(terminal)}


def test_latest_bucket_respects_validity_without_backsearch_and_keeps_legacy_semantics(tmp_path,monkeypatch):
    import pandas as pd
    clock,path,binding,_=_bound_source_fixture(tmp_path)
    item={'name':'fixture','table':path.name,'column':'value','owner':'momentum_flow_encoder',
        'availability_field':'fixture_observed','validity_column':'valid','source_clock_column':'snapshot'}
    registry={'fields':[item],'tables':{path.name:binding}}
    sources=gc._load_sources(registry)
    monkeypatch.setattr(gc,'REGISTRY',registry)
    monkeypatch.setattr(gc,'CONTEXT_FIELDS',('fixture','fixture_observed'))
    monkeypatch.setattr(gc,'_sources',lambda:sources)
    decisions=pd.DatetimeIndex(['2024-05-01T00:05Z','2024-05-01T00:07Z','2024-05-01T00:10Z','2024-05-01T00:12Z'])
    actual=gc.build_context(decisions)
    assert actual['fixture'].tolist()==[10.,10.,0.,0.]
    assert actual['fixture_observed'].tolist()==[1.,1.,0.,0.]
    del item['validity_column'];del item['source_clock_column']
    assert gc.build_context(decisions)['fixture'].tolist()==[10.,10.,20.,20.]


@pytest.mark.parametrize('column',['observed_profile_snapshot_age_minutes','observed_spot_basis_age_minutes'])
def test_decision_age_advances_and_future_source_rejects_exact_nanoseconds(tmp_path,column):
    import pandas as pd
    import pyarrow as pa
    clock,path,binding,_=_bound_source_fixture(tmp_path)
    item={'name':'age','table':path.name,'column':column,
        'source_clock_column':'snapshot','transform':'decision_age_minutes'}
    sources=gc._load_sources({'fields':[item],'tables':{path.name:binding}})
    decisions=pd.DatetimeIndex(['2024-05-01T00:07Z','2024-05-01T00:09Z','2024-05-01T00:12Z']).asi8
    joined=np.searchsorted(clock,decisions,side='right')-1
    values,observed=gc._field_values(item,*sources[path.name],decisions,joined)
    assert values.tolist()==[2.,4.,2.] and observed.all()
    table=sources[path.name][1]
    future=clock.copy();future[1]+=1
    table=table.set_column(table.schema.get_field_index('snapshot'),'snapshot',pa.array(future))
    with pytest.raises(RuntimeError,match='FUTURE_SOURCE_CLOCK'):
        gc._field_values(item,clock,table,decisions,joined)


def test_source_reader_projects_columns_streams_hash_and_limits_train_prefix(tmp_path,monkeypatch):
    from pathlib import Path
    clock,path,binding,_=_bound_source_fixture(tmp_path)
    item={'name':'value','table':path.name,'column':'value','validity_column':'valid','source_clock_column':'snapshot'}
    original=gc.pq.read_table;calls=[]
    def projected(*args,**kwargs):
        calls.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(gc.pq,'read_table',projected)
    monkeypatch.setattr(Path,'read_bytes',lambda *_:(_ for _ in ()).throw(AssertionError('whole-file read_bytes forbidden')))
    sources=gc._load_sources({'fields':[item],'tables':{path.name:binding}},decision_stop_ns=clock[1])
    assert len(sources[path.name][0])==2
    assert set(calls[0]['columns'])=={'available_ns','value','valid','snapshot'}
    assert calls[0]['filters']==[('available_ns','<=',int(clock[1]))]


@pytest.mark.parametrize("prefix,surface", [("", "ctx_cont"), ("ctx_cont.", "signal")])
@pytest.mark.parametrize("observed_values", [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
def test_masked_structural_value_extrema_follow_binary_identity(prefix, surface, observed_values):
    # Use an actual source-declared masked structural value in the active registry.
    value, observed = next(
        (value, observed)
        for value, observed in zip(gc.VALUE_FIELDS, gc.OBSERVED_FIELDS)
        if value in gc.STRUCTURAL_VALUE_FIELDS
    )
    names = [prefix + value, prefix + observed]
    raw = np.array(
        [[0.0, 0.0], [observed_values[0], 1.0], [observed_values[1], 1.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    contract = fit_surface_normalization(raw, surface=surface, field_names=names)
    require_surface_normalization(contract, surface=surface, field_names=names)
    assert contract["binary_mask"] == [1, 1]
    assert contract["center"] == [0.0, 0.0] and contract["scale"] == [1.0, 1.0]
    normalized = apply_surface_normalization(raw, contract)
    np.testing.assert_array_equal(normalized, raw)
    np.testing.assert_array_equal(contract["train_transformed_min"], normalized.min(axis=0))
    np.testing.assert_array_equal(contract["train_transformed_max"], normalized.max(axis=0))
    np.testing.assert_array_equal(invert_surface_normalization(normalized, contract), raw)

    # Neither fixing the fitted extrema nor masked storage relaxes the binary guard.
    invalid = raw.copy()
    invalid[1, 0] = 0.5
    with pytest.raises(RuntimeError, match="ENTRY_INPUT_NORMALIZATION_BINARY_DOMAIN_INVALID"):
        fit_surface_normalization(invalid, surface=surface, field_names=names)
    corrupt = dict(contract)
    corrupt["train_transformed_max"] = [float(np.arcsinh(1.0)), 1.0]
    with pytest.raises(RuntimeError, match="ENTRY_INPUT_NORMALIZATION_BINARY_CONTRACT_INVALID"):
        require_surface_normalization(corrupt, surface=surface, field_names=names)


def test_structural_value_identity_is_explicit_even_constant_and_checks_all_runtime_paths(monkeypatch):
    from gx1.contracts.entry_full_input_liveness_v1 import classify_field_status
    name=gc.VALUE_FIELDS[0];observed=gc.OBSERVED_FIELDS[0];names=[name,observed]
    monkeypatch.setattr(gc,'STRUCTURAL_VALUE_FIELDS',frozenset({name}))
    raw=np.array([[0.,1.],[0.,1.],[0.,1.]],np.float32)
    contract=fit_surface_normalization(raw,surface='ctx_cont',field_names=names)
    assert contract['binary_mask']==[1,1]
    assert contract['center']==[0.,0.] and contract['scale']==[1.,1.]
    np.testing.assert_array_equal(apply_surface_normalization(raw,contract),raw)
    stats=dict(row_count=3,finite_count=3,nonfinite_count=0,min=0.,max=0.,std=0.,value_range=0.,active_count=0,active_rate=0.)
    assert classify_field_status(split='train',surface='ctx_cont',field=name,stats=stats)[0]=='STRUCTURAL_OBSERVATION_MASK'
    broken=raw.copy();broken[0,0]=0.5
    with pytest.raises(RuntimeError,match='STRUCTURAL_VALUE_DOMAIN'):gc.require_masked_storage_numpy(broken,names)
    with pytest.raises(RuntimeError,match='STRUCTURAL_VALUE_DOMAIN'):
        gc.mask_normalized_torch(torch.tensor(broken),torch.tensor(broken),names)


def test_registry_uses_only_train_decisions_retains_constants_and_reports_sample_duplicates(tmp_path,monkeypatch):
    import pandas as pd
    clock,path,binding,terminal=_bound_source_fixture(tmp_path)
    base={'table':path.name,'owner':'momentum_flow_encoder','role':'market_value',
        'validity_column':'valid','source_clock_column':'snapshot'}
    fields=[{**base,'name':'a','column':'value'},{**base,'name':'b','column':'value_copy'},
        {**base,'name':'formula_alias','column':'value'},
        {**base,'name':'constant','column':'constant','validity_column':None},
        {**base,'name':'absent','column':'empty'},
        {**base,'name':'flag','column':'flag','validity_column':None,'role':'structural_observation_mask'}]
    candidates={'schema_version':'fixture','fields':fields}
    monkeypatch.setattr(gc,'_completion_candidate_map',lambda:candidates)
    decisions=pd.DatetimeIndex(['2024-05-01T00:05Z','2024-05-01T00:07Z','2024-05-01T00:10Z','2024-05-01T00:12Z'])
    registry=gc.build_registry_from_train(candidates,{path.name:binding},decisions,
        train_start='2024-05-01T00:00Z',train_end='2024-05-01T00:13Z',source_terminal=terminal,
        retained_registry={'fields':[],'tables':{}})
    assert {row['name'] for row in registry['fields']}=={'a','b','constant','flag'}
    assert next(row for row in registry['fields'] if row['name']=='constant')['train_constant'] is True
    assert next(row for row in registry['fields'] if row['name']=='flag')['role']=='structural_observation_mask'
    assert registry['unsupported']==[{'name':'absent','reason':'NO_GENUINE_TRAIN_OBSERVATION'}]
    assert registry['exact_duplicate_aliases'][0]['removed']=='formula_alias'
    assert any({'a','b'}.issubset(group) for group in map(set,registry['equal_train_vector_groups']))
    assert registry['duplicate_sample_equality_is_formula_proof'] is False
    assert registry['train_decision_rows']==4
    with pytest.raises(RuntimeError,match='TRAIN_SCOPE'):
        gc.build_registry_from_train(candidates,{path.name:binding},decisions,
            train_start='2024-05-01T00:00Z',train_end='2024-05-01T00:12Z',source_terminal=terminal,
            retained_registry={'fields':[],'tables':{}})


def test_registry_retains_original_basis_and_namespaces_overlapping_table_binding(tmp_path,monkeypatch):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    clock,path,binding,terminal=_bound_source_fixture(tmp_path)
    legacy=tmp_path/'legacy.parquet';pq.write_table(pa.table({'available_ns':clock,'legacy':[9.,9.,9.]}),legacy)
    basis=tmp_path/'basis.parquet';pq.write_table(pa.table({'available_ns':clock,'basis':[1.,2.,3.]}),basis)
    def old(name,key,column):return {'name':name,'table':key,'column':column,'owner':'momentum_flow_encoder','availability_field':name+'_observed'}
    retained={'fields':[old('legacy_value',path.name,'legacy'),old('spot_basis','BASIS_M5.parquet','basis')],
        'tables':{path.name:{'path':str(legacy),'sha256':gc._stream_sha256(legacy)},
        'BASIS_M5.parquet':{'path':str(basis),'sha256':gc._stream_sha256(basis)}}}
    candidates={'fields':[{'name':'new_value','table':path.name,'column':'value','owner':'momentum_flow_encoder'}]}
    monkeypatch.setattr(gc,'_completion_candidate_map',lambda:candidates)
    registry=gc.build_registry_from_train(candidates,{path.name:binding},pd.DatetimeIndex(['2024-05-01T00:05Z','2024-05-01T00:10Z']),
        train_start='2024-05-01T00:00Z',train_end='2024-05-01T00:11Z',source_terminal=terminal,retained_registry=retained)
    assert {item['name'] for item in registry['fields']}=={'legacy_value','spot_basis','new_value'}
    assert next(item for item in registry['fields'] if item['name']=='legacy_value')['table']=='retained::'+path.name
    assert registry['tables']['BASIS_M5.parquet']['path']==str(basis)


def test_registry_clock_admission_requires_whole_exact_physical_train_not_eligible_subset():
    import hashlib
    import pandas as pd
    decisions=pd.DatetimeIndex(['2024-05-01T00:05Z','2024-05-01T00:10Z','2024-05-01T00:15Z'])
    registry={'train_decision_rows':3,'train_decision_clock_sha256':hashlib.sha256(decisions.asi8.tobytes()).hexdigest()}
    assert gc.require_registry_train_clock_match(decisions,registry)['exact_match'] is True
    with pytest.raises(RuntimeError,match='PHYSICAL_TRAIN_CLOCK_MISMATCH'):
        gc.require_registry_train_clock_match(decisions[:2],registry)
    shifted=decisions+pd.Timedelta(minutes=5)
    with pytest.raises(RuntimeError,match='PHYSICAL_TRAIN_CLOCK_MISMATCH'):
        gc.require_registry_train_clock_match(shifted,registry)
