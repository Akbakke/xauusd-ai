import json
from types import SimpleNamespace
import pandas as pd
from gx1.scripts import research_openbb_source_review_v1 as audit


def test_keyless_gold_scope_and_no_fred_call(tmp_path,monkeypatch):
    import sys
    calls=[]
    def cot(**kw):
        calls.append(kw)
        return SimpleNamespace(to_dataframe=lambda **_:pd.DataFrame({'date':['2024-05-07'],'cftc_contract_market_code':['088691']}))
    monkeypatch.delenv('FRED_API_KEY',raising=False)
    monkeypatch.setitem(sys.modules,'openbb',SimpleNamespace(obb=SimpleNamespace(cftc=SimpleNamespace(cot=cot))))
    monkeypatch.setattr(audit.importlib.metadata,'version',lambda _: '2.0.0')
    plan={'packages':{'openbb-cftc':'2.0.0'},'CFTC_code':'CFTC_088691','start_date':'2024-05-01','end_date':'2026-06-30','end_exclusive':'2026-07-01','max_reports':200,'maximum_output_bytes':100000}
    audit.provider_review(plan,tmp_path)
    out=json.loads((tmp_path/'PROVIDER_RESULT.json').read_text())
    assert out['FRED_status']=='BLOCKED_MISSING_FRED_API_KEY'
    assert out['point_in_time_admitted'] is False
    assert [c['report_type'] for c in calls]==['legacy','disaggregated']
    assert all(c['futures_only'] and c['code']=='CFTC_088691' and c['end_date']=='2026-06-30' for c in calls)
    assert out['requests']['disaggregated']['rows']==1


def test_wrong_market_is_rejected_without_error_url(tmp_path,monkeypatch):
    import sys
    def cot(**kw):return SimpleNamespace(to_dataframe=lambda **_:pd.DataFrame({'date':['2024-05-07'],'cftc_contract_market_code':['OTHER']}))
    monkeypatch.delenv('FRED_API_KEY',raising=False)
    monkeypatch.setitem(sys.modules,'openbb',SimpleNamespace(obb=SimpleNamespace(cftc=SimpleNamespace(cot=cot))))
    monkeypatch.setattr(audit.importlib.metadata,'version',lambda _: '2.0.0')
    p={'packages':{'openbb-cftc':'2.0.0'},'CFTC_code':'CFTC_088691','start_date':'2024-05-01','end_date':'2026-06-30','end_exclusive':'2026-07-01','max_reports':200,'maximum_output_bytes':100000}
    audit.provider_review(p,tmp_path)
    d=json.loads((tmp_path/'PROVIDER_RESULT.json').read_text())
    assert all(r=={'status':'FAILED','exception':'RuntimeError'} for r in d['requests'].values())
    assert not list(tmp_path.glob('*.parquet'))


def test_shutdown_report_date_is_not_publication_date():
    import numpy as np
    from gx1.scripts.research_openbb_source_review_v1 import known_cot_delay_diagnostic_v1
    reports=pd.DatetimeIndex(['2025-09-23','2025-09-30'],tz='UTC')
    t=pd.DatetimeIndex(['2025-10-06T12:00Z','2025-10-08T12:00Z','2025-11-19T20:30Z','2025-11-19T20:35Z']).asi8
    actual=known_cot_delay_diagnostic_v1(t,reports,{'2025-09-30':'2025-11-19T20:30:00Z'})
    np.testing.assert_array_equal(actual,[False,True,True,False])
