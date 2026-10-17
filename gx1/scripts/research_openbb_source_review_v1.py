"""Small isolated OpenBB V5 source audit. No native registration or model fit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def provider_review(plan, directory):
    """Only the two preregistered Gold requests; no FRED request without a key."""
    from openbb import obb
    versions={n:importlib.metadata.version(n) for n in plan['packages']}
    if versions!=plan['packages']:raise RuntimeError('OPENBB_VERSION_MISMATCH')
    out={'versions':versions,'requests':{},'FRED_status':'BLOCKED_MISSING_FRED_API_KEY',
         'point_in_time_admitted':False,'native_features_added':0}
    key=os.environ.get('FRED_API_KEY')
    if key:
        # No ambient user-settings lookup: a credential must be explicitly supplied.
        obb.user.credentials.fred_api_key=key
        try:
            obj=obb.fred.economy.fred_series(symbol=plan['FRED_series'],start_date=plan['start_date'],end_date=plan['end_date'],use_cache=False,provider='fred')
            rows=obj.to_dataframe(index=None);rows.to_json(directory/'FRED_VALUES.json',orient='records',date_format='iso')
            out['FRED_status']='CURRENT_VINTAGE_ONLY_NOT_PIT';out['FRED_rows']=len(rows)
        except Exception as exc:out['FRED_status']='REQUEST_FAILED_'+type(exc).__name__
    for kind in ('legacy','disaggregated'):
        try:
            obj=obb.cftc.cot(code=plan['CFTC_code'],report_type=kind,futures_only=True,
                measure='all',start_date=plan['start_date'],end_date=plan['end_date'],provider='cftc')
            frame=obj.to_dataframe(index=None)
            if len(frame)>plan['max_reports']:raise RuntimeError('OPENBB_REPORT_BUDGET')
            # Provider rows must carry the declared market identity, not other gold markets.
            if set(frame.cftc_contract_market_code.astype(str))!={'088691'}:raise RuntimeError('OPENBB_MARKET_SCOPE')
            dates=__import__('pandas').to_datetime(frame.date,utc=True)
            if not ((dates>=plan['start_date'])&(dates<plan['end_exclusive'])).all():raise RuntimeError('OPENBB_DATE_SCOPE')
            path=directory/(kind.upper()+'.parquet');frame.to_parquet(path,index=False)
            if path.stat().st_size>plan['maximum_output_bytes']:raise RuntimeError('OPENBB_SIZE_BUDGET')
            out['requests'][kind]={'status':'RETRIEVED_CURRENT_VINTAGE_NOT_PIT','rows':len(frame),'columns':list(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception as exc:
            # Error strings can include credential-bearing URLs; record type only.
            out['requests'][kind]={'status':'FAILED','exception':type(exc).__name__}
    (directory/'PROVIDER_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binding',type=Path)
    parser.add_argument('--provider-plan',type=Path);parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.provider_plan:
        plan=json.loads(args.provider_plan.read_text())
        if args.directory!=Path(plan['output_root'])/'.RESULTS.staging' or not (Path(plan['output_root'])/'EXECUTION_CLAIM.json').exists():raise RuntimeError('OPENBB_CHILD_SCOPE')
        from openbb_core.app.service.system_service import SystemService
        settings=args.directory/'SETTINGS'
        SystemService(openbb_directory=str(settings),user_settings_path=str(settings/'user_settings.json'),system_settings_path=str(settings/'system_settings.json'),logging_handlers=['noop'])
        from openbb import obb
        obb.user.preferences.cache_directory=str(args.directory/'CACHE')
        provider_review(plan,args.directory);return
    from gx1.scripts import gc_source_disposition_v1 as io
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    import numpy as np
    import pandas as pd
    identity='OPENBB_FREE_SOURCE_REVIEW_20261017_014';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    if args.binding!=root/'EXECUTION_BINDING.json':raise RuntimeError('OPENBB_BINDING_PATH')
    binding=io.read_json(args.binding);path=io.REPO/'configs/research'/f'{identity}.json'
    io.verify_digest(path,binding['plan_sha256'],'PLAN');plan=io.read_json(path)
    io.verify_digest(io.REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'POLICY')
    io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'APPROVAL')
    approval=io.read_json(root/'OPERATOR_APPROVAL.json');policy=io.read_json(io.REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['openbb_source_review_014']
    if (gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0 or gate['execution_authorized'] is not False or policy['training_enabled'] is not False or approval['approved_plan_sha256']!=binding['plan_sha256'] or approval['max_invocations']!=1 or io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=io.REPO,text=True)):raise RuntimeError('OPENBB_AUTHORITY')
    for rel,digest in binding['code_sha256'].items():io.verify_digest(io.REPO/rel,digest,'CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    directory=root/'.RESULTS.staging';directory.mkdir(exist_ok=False)
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head']})
    subprocess.run([plan['environment_python'],str(Path(__file__).resolve()),'--provider-plan',str(path),'--directory',str(directory)],check=True,timeout=plan['provider_timeout_seconds'])
    result=io.read_json(directory/'PROVIDER_RESULT.json')
    from gx1.scripts import research_ta_campaign_v1 as existing
    spec=io.read_json(io.REPO/'configs/research/D1_MACRO_CONTEXT_SOURCES_20261011_001.json')
    cot=existing.parse_cot_gold_archives(spec);treasury=existing.parse_treasury_yields(spec['drivers']['treasury'])
    result['existing_cot']={'rows':len(cot),'first':str(cot.as_of.min()),'last':str(cot.as_of.max()),'category':'legacy noncommercial; not disaggregated Managed Money'}
    result['existing_treasury']={'rows':len(treasury),'first':str(treasury.Date.min()),'last':str(treasury.Date.max())}
    legacy=directory/'LEGACY.parquet'
    if legacy.exists():
        new=pd.read_parquet(legacy);new['as_of']=pd.to_datetime(new.date,utc=True)
        merged=cot.merge(new,on='as_of',suffixes=('_old','_new'))
        mapping={'open_interest':'open_interest_all','noncomm_long':'non_commercial_positions_long_all','noncomm_short':'non_commercial_positions_short_all','comm_long':'commercial_positions_long_all','comm_short':'commercial_positions_short_all'}
        result['legacy_comparison']={'matched_reports':len(merged),'fields':{}}
        for old,newname in mapping.items():
            delta=merged[old].to_numpy(float)-merged[newname].to_numpy(float)
            result['legacy_comparison']['fields'][old]={'different':int(np.count_nonzero(delta)),'max_abs':float(np.max(np.abs(delta))) if len(delta) else None}
    disagg=directory/'DISAGGREGATED.parquet'
    if disagg.exists():
        df=pd.read_parquet(disagg).sort_values('date');L=df.managed_money_positions_long_all.astype(float);S=df.managed_money_positions_short_all.astype(float);oi=df.open_interest_all.astype(float)
        ratio=(L-S)/oi;past=ratio.shift(1).rolling(plan['position_history_reports'],min_periods=plan['position_history_reports'])
        sd=past.std(ddof=0);z=(ratio-past.mean())/sd.replace(0,np.nan)
        candidate=pd.DataFrame({'as_of':df.date,'mm_long':L,'mm_short':S,'open_interest':oi,'mm_net_over_oi':ratio,'mm_long_change1report':L.diff(),'mm_short_change1report':S.diff(),'mm_net_over_oi_past52_z':z,'availability_verified':False})
        candidate.to_parquet(directory/'MANAGED_MONEY_REVIEW_ONLY.parquet',index=False)
        result['managed_money_candidate']={'rows':len(df),'past52_z_rows':int(z.notna().sum()),'publication_timestamp_in_provider':False,'training_admitted':False}
    result.update(status='SOURCE_REVIEW_COMPLETE_NOT_FEATURE_ADMISSION',source_head=binding['source_head'],limits=proof,TEST_accessed=False,fit_started=False)
    for rel,digest in binding['code_sha256'].items():io.verify_digest(io.REPO/rel,digest,'CODE_POST')
    io.verify_digest(io.REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'POLICY_POST')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=io.REPO,text=True):raise RuntimeError('OPENBB_SOURCE_CHANGED')
    io.write_json(directory/'RESULT.json',result)
    io.write_json(directory/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(directory/'RESULT.json')})
    publish_bundle_directory_noreplace(directory,root/'RESULTS');print('OPENBB_SOURCE_REVIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
