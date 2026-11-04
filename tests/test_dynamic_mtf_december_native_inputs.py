import hashlib
import os

import numpy as np
import pytest

from gx1.audit import dynamic_mtf_december_native_inputs as route
from gx1.audit import dynamic_mtf_december_score_targets as score
from gx1.audit.dynamic_mtf_december_normalization import _raw_array
from gx1.contracts import entry_dynamic_mtf_december_native_v1 as authority
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


def _ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_native_role_builds_readonly_subrange_not_a_whole_table(tmp_path):
    p = tmp_path/'raw.mmap'
    np.arange(20, dtype=np.float32).tofile(p)
    os.chmod(p, 0o444)
    st = {'st_mode':p.stat().st_mode, 'st_size':p.stat().st_size}
    a = route._map_role(str(p),st,begin=2,end=5,total_rows=10,width=2,dtype='float32')
    assert a.tolist() == [[4.,5.],[6.,7.],[8.,9.]] and not a.flags.writeable
    assert a.offset == 16 and a.nbytes == 24
    with pytest.raises(RuntimeError,match='PHYSICAL_SLICE'):
        route._map_role(str(p),st,begin=2,end=11,total_rows=10,width=2,dtype='float32')
    f = tmp_path/'raw.npy'; np.save(f,np.arange(20,dtype=np.float32).reshape(10,2))
    limited = _raw_array(_ref(f),shape=(10,2),dtype='float32',stop=3)
    assert limited.shape == (3,2) and not limited.flags.writeable
    assert len(limited._mmap) == limited.offset+limited.nbytes < f.stat().st_size


def test_score_and_fit_roles_check_frozen_disjoint_axes(monkeypatch):
    monkeypatch.setattr(views,'_require_view',lambda _v:None)
    parent = np.arange(7,dtype=np.int64)
    bar = (parent+10)*views.BAR_NS
    clock = bar+views.BAR_NS
    axes = dict(parent_rows=parent,bar_start_ns=bar,decision_ns=clock)
    src = {'path':'/tmp/original-clock','sha256':'2'*64}
    def section(sl, cut):
        return {'rows':len(parent[sl]), 'cutoff_ns':cut,
                **{k+'_sha256':hashlib.sha256(v[sl].tobytes()).hexdigest() for k,v in axes.items()}}
    scope = {**section(slice(None),int(clock[-1])+1),'source':src, 'fit':section(slice(0,4),int(clock[4])),
             'held_out':section(slice(4,None),int(clock[-1])+1)}
    v={'reserved_observer_axes':scope,'contract_sha256':'1'*64}
    held={k:a[4:] for k,a in axes.items()}
    bound=views.NativeDirectionObserverAxes(v,source=src,role='score',**held)
    assert np.array_equal(bound.parent_rows,[4,5,6]) and not bound.parent_rows.flags.writeable
    with pytest.raises(RuntimeError,match='OBSERVER_ROW_COUNT'):
        views.NativeDirectionObserverAxes(v,source=src,role='fit',**held)
    with pytest.raises(RuntimeError,match='OBSERVER_FIT_AXIS'):
        views.NativeDirectionObserverAxes(v,source=src,role='score',**{**held,'parent_rows':parent[:3]})
    fit=views.NativeDirectionObserverAxes(v,source=src,role='fit',**{k:a[:4]for k,a in axes.items()})
    assert np.array_equal(fit.parent_rows,[0,1,2,3])


def test_private_with_wrong_role_or_reserved_target_fails_before_any_payload_open():
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    class Reader:
        def __init__(self):
            self.opened=[]
        def _read(self, ref, unused):
            self.opened.append(ref)
            return {'bindings':{k:k for k in ('feature_view','prefix_normalization','prefix_prior','prefix_counts','fit_target_view')}}
    r=Reader()
    bad = {'run_scope':authority.PRIVATE_SCOPE,'input_role':'score'}
    with pytest.raises(RuntimeError,match='ROLE_SCOPE'):
        worker._dynamic_state_metadata(bad,r)
    assert not r.opened
    l={'run_scope':authority.PRIVATE_SCOPE,'input_role':'fit','candidate_plan':'frozen-plan','feature_view':'feature_view',
       'prefix_normalization':'prefix_normalization','prefix_prior':'prefix_prior','label_audit':'prefix_counts',
       'target_manifest':'reserved-target'}
    with pytest.raises(RuntimeError,match='FIT_TARGET_BEFORE_OPEN'):
        worker._dynamic_state_metadata(l,r)
    assert r.opened==['frozen-plan']


def test_reserved_target_slice_excludes_prior_future_cells_and_rebases_missing_pointer(tmp_path):
    parent_clock=np.array([10,20,30,40,50,60],np.int64)
    marker=np.array([10,20,30,32,45,50,60,92],np.int64)
    cols={'issue_ns':parent_clock,'parent_rows':np.arange(6,dtype=np.int64),
          'issue_source_index':np.array([0,1,2,-1,5,6],np.int64),'validity_ns':marker,'source_row':np.arange(8,dtype=np.int64)}
    for k,dt in target.SERIES_ARRAY_DTYPES.items():
        if k not in cols:cols[k]=np.zeros((len(marker),5),dtype=dt)
    specs={}
    for k,v in cols.items():
        f=tmp_path/(k+'.npy');np.save(f,v);specs[k]={**_ref(f),'shape':list(v.shape),'size_bytes':f.stat().st_size,'dtype':v.dtype.name}
    total=sum(d['size_bytes']for d in specs.values())
    a,l,r,sz=score._load_immutable_slice({'arrays':specs},tmp_path,{'rows':2,'cutoff_ns':30},
                                         {'rows':4,'cutoff_ns':61},total,5000)
    assert (l,r)==(2,7) and a['parent_rows'].tolist()==[2,3,4,5]
    assert a['validity_ns'].tolist()==[30,32,45,50,60] and a['issue_source_index'].tolist()==[0,-1,3,4]
    assert sz==sum(v.nbytes+128 for v in a.values())
    with pytest.raises(RuntimeError,match='OUTPUT_BYTE_LIMIT'):
        score._load_immutable_slice({'arrays':specs},tmp_path,{'rows':2,'cutoff_ns':30},{'rows':4,'cutoff_ns':61},total,sz-1)


def test_source_clock_projection_excludes_future_prices(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    f=tmp_path/'source.parquet'; ts=pa.array([100,200,300,400],type=pa.timestamp('ns',tz='UTC'))
    pq.write_table(pa.table({'time':ts,'not_used':np.array([77.,88.,99.,111.])}),f)
    assert route._role_clock(f,first=100,last=300).tolist()==[100,200,300]


def test_only_the_bound_xau_cache_manifest_can_exceed_generic_metadata_limit(tmp_path):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common

    path = tmp_path / 'manifest.json'
    path.write_bytes(b'{"role":"test"}' + b' ' * (16 * 1024**2))
    ref = _ref(path)
    item = {'sha256': ref['sha256'], 'stat': common._stat(path), 'verification': 'canonical_fresh_hash'}
    raw = {'native_cache_dir': str(tmp_path),
           'base_lineage': {'mtf_cache_manifest_path': str(path), 'mtf_cache_manifest_sha256': ref['sha256']},
           'payloads': {str(path): item}}
    with pytest.raises(RuntimeError, match='SMALL_PLAINREF'):
        common._read(ref, {})
    metadata = {}
    assert route._read_xau_cache_manifest(raw, metadata) == {'role':'test'}
    assert metadata == {str(path): {'ref':ref, 'stat':item['stat']}}
    with pytest.raises(RuntimeError, match='XAU_CACHE_MANIFEST_PATH'):
        route._read_xau_cache_manifest({**raw, 'native_cache_dir': str(tmp_path/'other')}, {})
    with pytest.raises(RuntimeError, match='XAU_CACHE_MANIFEST_ROLE'):
        route._read_xau_cache_manifest({**raw, 'payloads': {str(path): {**item,'verification':'other'}}}, {})
    bad_size = {**item, 'stat': {**item['stat'], 'st_size': item['stat']['st_size'] + 1}}
    with pytest.raises(ValueError, match='PAYLOAD_IDENTITY'):
        route._read_xau_cache_manifest({**raw,'payloads':{str(path):bad_size}}, {})
