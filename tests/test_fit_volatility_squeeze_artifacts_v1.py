import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import fit_volatility_squeeze_artifacts_v1 as producer
from gx1.features import volatility_squeeze_state_v1 as owner


def _fixture(tmp_path, *, times=None, cutoff=None):
    root = tmp_path / 'native'
    year = root / 'year=2024'
    year.mkdir(parents=True)
    if times is None:
        times = pd.date_range('2024-01-01T00:00:00Z', periods=3 * 288, freq='5min')
    close = 2000 + np.arange(len(times)) * .001
    frame = pd.DataFrame(dict(time=times, open=close, high=close + 1,
                              low=close - 1, close=close, volume=np.ones(len(times))))
    part = year / 'part-000.parquet'
    frame.to_parquet(part, row_group_size=len(frame))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = root / 'MANIFEST.json'
    manifest.write_text(json.dumps(dict(
        schema_version='xau_canonical_native_source_v3', instrument='XAU_USD',
        timeframe='M5', source_granularity='M5', source_kind='oanda_native_mba_candles',
        prices='MBA', bar_duration_seconds=300, decision_available_offset_seconds=300,
        timestamp_semantics='bar_start_utc', completion_field='complete',
        completion_value=True, year_sha256={'year=2024': digest(part)})))
    native = dict(path=str(manifest), sha256=digest(manifest))
    plan = dict(schema_version='gx1_native_m5_squeeze_fit_plan_v1',
                native_m5_manifest=native,
                timeframes=list(owner.VOLATILITY_SQUEEZE_DIRECTION_CLOCKS),
                source_clock_contract=owner.VOLATILITY_SQUEEZE_DIRECTION_CLOCK_CONTRACT,
                fit_start_ns=int(times[0].value),
                fit_cutoff_ns=int(pd.Timestamp(cutoff if cutoff is not None else '2024-01-03T00:00:00Z').value))
    plan['source_spec'] = dict(manifest=native, cutoff_ns=plan['fit_cutoff_ns'],
                              parts=[dict(path=str(part), sha256=digest(part), row_groups=[0])])
    path = tmp_path / 'PLAN.json'
    path.write_text(json.dumps(plan))
    return path, plan, part


def test_native_producer_proves_five_raw_frames_and_excludes_cutoff(tmp_path, monkeypatch):
    path, plan, _ = _fixture(tmp_path)
    calls = []

    def fit(frames, **kwargs):
        calls.append((frames, kwargs))
        manifest = kwargs['output_dir'] / 'manifest.json'
        manifest.write_text(json.dumps(dict(artifacts={
            tf: dict(train_ohlcv_rows=len(frame), train_ohlcv_sha256=owner._ohlcv_sha256(frame))
            for tf, frame in frames.items()})))
        return manifest

    monkeypatch.setattr(owner, 'fit_volatility_squeeze_artifact_manifest', fit)
    output = tmp_path / 'fit'
    output.mkdir()
    manifest, proof = producer.fit_native_m5_artifacts_from_plan(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), output_dir=output)
    frames, kwargs = calls[0]
    assert tuple(frames) == owner.VOLATILITY_SQUEEZE_DIRECTION_CLOCKS
    assert tuple(kwargs['source_provenance_by_clock']) == tuple(frames)
    assert manifest.is_file()
    assert proof['fit_frames']['M5']['rows'] == 575
    for tf, info in proof['fit_frames'].items():
        assert info['last_available_ns'] < plan['fit_cutoff_ns']
        assert info['ohlcv_sha256'] == owner._ohlcv_sha256(frames[tf])
    assert frames['H1'].iloc[0]['open'] == frames['M5'].iloc[0]['open']
    assert frames['H1'].iloc[0]['close'] == frames['M5'].iloc[11]['close']


def test_native_producer_rejects_changed_raw_bytes_before_fit(tmp_path, monkeypatch):
    path, _, part = _fixture(tmp_path)
    part.write_bytes(part.read_bytes() + b'changed')
    monkeypatch.setattr(owner, 'fit_volatility_squeeze_artifact_manifest',
                        lambda *args, **kwargs: pytest.fail('fit reached'))
    with pytest.raises(RuntimeError, match='PART_IDENTITY_INVALID'):
        producer.fit_native_m5_artifacts_from_plan(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output_dir=tmp_path)


def test_native_producer_rejects_val_fit_before_raw_io(tmp_path, monkeypatch):
    path, plan, _ = _fixture(tmp_path)
    plan['fit_cutoff_ns'] = pd.Timestamp('2026-07-01T00:00:00Z').value
    path.write_text(json.dumps(plan))
    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, 'ParquetFile', lambda *args, **kwargs: pytest.fail('raw IO reached'))
    with pytest.raises(RuntimeError, match='FIT_SCOPE'):
        producer.fit_native_m5_artifacts_from_plan(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output_dir=tmp_path)


def test_native_producer_rejects_undeclared_minute_metadata_before_raw_io(tmp_path, monkeypatch):
    path, plan, _ = _fixture(tmp_path)
    plan['m1_source'] = dict(path=str(tmp_path / 'forbidden'), sha256='a' * 64)
    path.write_text(json.dumps(plan))
    import pyarrow.parquet as pq
    monkeypatch.setattr(pq, 'ParquetFile', lambda *args, **kwargs: pytest.fail('raw IO reached'))
    with pytest.raises(RuntimeError, match='NATIVE_M5_PROVENANCE_INVALID'):
        producer.fit_native_m5_artifacts_from_plan(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output_dir=tmp_path)



def test_native_fit_frames_skip_only_empty_gaps_and_keep_observed_partial_bins(tmp_path):
    from gx1.features.htf_features import _resample_ohlcv
    times=pd.date_range('2024-01-04T00:00:00Z','2024-01-09T00:00:00Z',freq='5min')
    weekend=(times>=pd.Timestamp('2024-01-05T22:00:00Z')) & (times<pd.Timestamp('2024-01-07T23:00:00Z'))
    daily=(times>=pd.Timestamp('2024-01-08T21:00:00Z')) & (times<pd.Timestamp('2024-01-08T22:00:00Z'))
    missing_member=times==pd.Timestamp('2024-01-08T10:05:00Z')
    times=times[~(weekend|daily|missing_member)]
    path,plan,_=_fixture(tmp_path,times=times,cutoff='2024-01-09T00:00:00Z')
    frames,_,proof=producer.native_m5_fit_frames_from_plan(
        path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    m5=frames['M5'];raw_m15=_resample_ohlcv(m5,'M15')
    empty_label=pd.Timestamp('2024-01-06T12:00:00Z')
    assert raw_m15.loc[empty_label,['open','high','low','close']].isna().all()
    assert raw_m15.loc[empty_label,'volume']==0 and empty_label not in frames['M15'].index
    # Independent integer grouping proves every observed fit bucket remains.
    for clock,frame in frames.items():
        grid=owner.volatility_squeeze_bar_grid(clock);duration=grid['duration_ns'];offset=grid['origin_offset_ns']
        labels=(m5.index.asi8-offset)//duration*duration+offset
        unique=np.unique(labels)
        expected=unique[(unique>=plan['fit_start_ns']) & (unique+duration<plan['fit_cutoff_ns'])]
        np.testing.assert_array_equal(frame.index.asi8,expected)
        owner._validate_closed_ohlcv(frame,timeframe=clock,context='GAP_REGRESSION')
        assert proof['fit_frames'][clock]['rows']==len(expected)
    partial_label=pd.Timestamp('2024-01-08T10:00:00Z')
    members=m5.loc[[partial_label,partial_label+pd.Timedelta(minutes=10)]]
    observed=frames['M15'].loc[partial_label]
    np.testing.assert_array_equal(observed[['open','high','low','close','volume']].to_numpy(),
        np.array([members['open'].iloc[0],members['high'].max(),members['low'].min(),
                  members['close'].iloc[-1],members['volume'].sum()]))
    # Sunday reopening and a daily closure leave observed partial H4 bins.
    assert frames['H4'].loc[pd.Timestamp('2024-01-07T22:00:00Z'),'volume']==36
    assert frames['H4'].loc[pd.Timestamp('2024-01-08T18:00:00Z'),'volume']==36
    assert pd.Timestamp('2024-01-08T21:00:00Z') not in frames['H1'].index


@pytest.mark.parametrize('invalid',['all_prices_nan','open_nan','zero_volume'])
def test_native_fit_frames_retain_actual_invalid_bars_for_strict_rejection(tmp_path,invalid):
    path,plan,part=_fixture(tmp_path)
    frame=pd.read_parquet(part)
    label=pd.Timestamp('2024-01-01T10:00:00Z')
    selected=(frame['time']>=label)&(frame['time']<label+pd.Timedelta(minutes=15))
    if invalid=='zero_volume':frame.loc[selected,'volume']=0
    else:frame.loc[selected,['open','high','low','close'] if invalid=='all_prices_nan' else ['open']]=np.nan
    frame.to_parquet(part,row_group_size=len(frame))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest_path=Path(plan['native_m5_manifest']['path'])
    manifest=json.loads(manifest_path.read_text());manifest['year_sha256']['year=2024']=digest(part)
    manifest_path.write_text(json.dumps(manifest))
    plan['native_m5_manifest']['sha256']=digest(manifest_path)
    plan['source_spec']['parts'][0]['sha256']=digest(part)
    path.write_text(json.dumps(plan))
    frames,_,_=producer.native_m5_fit_frames_from_plan(path,expected_sha256=digest(path))
    assert label in frames['M15'].index
    if invalid=='all_prices_nan':assert frames['M15'].loc[label,['open','high','low','close']].isna().all()
    elif invalid=='open_nan':assert np.isnan(frames['M15'].loc[label,'open'])
    else:assert frames['M15'].loc[label,'volume']==0
    with pytest.raises(RuntimeError,match='SOURCE_NONFINITE|OHLCV_GEOMETRY_INVALID'):
        owner._validate_closed_ohlcv(frames['M15'],timeframe='M15',context='INVALID_REGRESSION')
