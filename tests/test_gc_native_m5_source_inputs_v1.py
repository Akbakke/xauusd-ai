"""Causal native M5 source clocks and independently checked raw payloads."""
import hashlib

import numpy as np
import pytest

from gx1.contracts import gc_native_m5_source_inputs_v1 as native
from gx1.contracts import gc_continuous_source_v1 as source
from gx1.contracts import gc_source_disposition_v1 as timing


def clock(days):
    return dict(schema_version=source.NATIVE_M5_REDUCTION_CLOCK_SCHEMA,
        start_ns=timing.STUDY_START_NS, end_ns=timing.STUDY_START_NS+days*native.DAY_NS,
        bar_duration_ns=native.NATIVE_M5_INTERVAL_NS)


def test_roll_uses_previous_complete_day_directly_on_native_m5_and_never_backwards():
    volumes=np.array([[10,20,1,1000],[2,3,30,1]], np.uint64)
    rows,days,basis=native.selected_native_m5_contracts(volumes,[(2024,6),(2024,8)],source_clock=clock(4))
    assert days.tolist()==[-1,0,0,1]
    assert basis.tolist()==[-1,0,1,2]
    assert rows.shape==(4*288,)
    assert np.all(rows[:288]==-1)
    assert rows[288]==0 and rows[576]==0 and rows[864]==1
    changed=volumes.copy();changed[:,2:]=[[0,0],[999,999]]
    actual=native.selected_native_m5_contracts(changed,[(2024,6),(2024,8)],source_clock=clock(4))[0]
    np.testing.assert_array_equal(actual[:864],rows[:864])


def test_roll_ties_and_absence_preserve_unknown_then_current_contract():
    rows,days,basis=native.selected_native_m5_contracts(np.array([[5,7,0,0],[5,1,0,0]],np.uint64),
        [(2024,6),(2024,8)],source_clock=clock(4))
    assert days.tolist()==[-1,-1,0,0]
    assert basis.tolist()==[-1,-1,1,1]
    assert np.all(rows[:576]==-1)


@pytest.mark.parametrize('values',[np.ones((2,3),np.float64),np.ones((2,2),np.uint64)])
def test_roll_rejects_inexact_volume_or_wrong_day_axis(values):
    with pytest.raises(ValueError,match='DAILY_VOLUME_CLOCK'):
        native.selected_native_m5_contracts(values,[(2024,6),(2024,8)],source_clock=clock(3))


def test_raw_hash_gate_rejects_changed_payload_with_same_file_size(tmp_path,monkeypatch):
    monkeypatch.setattr(native,'GC_ROOT',tmp_path)
    path=tmp_path/'original.gz';path.write_bytes(b'original')
    item=dict(path=str(path),bytes=8,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert native.verify_native_raw_payload(item)['sha256']==item['sha256']
    path.write_bytes(b'changed!')
    with pytest.raises(ValueError,match='RAW_PAYLOAD_HASH'):
        native.verify_native_raw_payload(item)


def test_raw_hash_gate_rejects_symlink_escape(tmp_path,monkeypatch):
    root=tmp_path/'raw';root.mkdir();monkeypatch.setattr(native,'GC_ROOT',root)
    outside=tmp_path/'elsewhere';outside.write_bytes(b'raw')
    link=root/'alias';link.symlink_to(outside)
    with pytest.raises(ValueError,match='RAW_PATH'):
        native.verify_native_raw_payload(dict(path=str(link),bytes=3,sha256=hashlib.sha256(b'raw').hexdigest()))
