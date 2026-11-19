import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import fit_volatility_squeeze_artifacts_v1 as producer
from gx1.features import volatility_squeeze_state_v1 as owner


def _fixture(tmp_path):
    root = tmp_path / 'native'
    year = root / 'year=2024'
    year.mkdir(parents=True)
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
                fit_cutoff_ns=int(pd.Timestamp('2024-01-03T00:00:00Z').value))
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
