"""Separate Entry research diagnostics; no native/production target authority.

Strict terminal version freezes delay=0 before outcome inspection. The NY
calendar is an explicitly UNQUALIFIED historical candidate: current regular
hours cannot establish 2011-2025 holiday/regime history. No fitted thresholds,
Exit teacher, feature construction or hindsight exit selection occurs here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NS=10**9
MINUTE=60*NS
DAY=86400*NS
STRICT_TERMINAL_MAX_DELAY_SECONDS=0
STRICT_VARIANT='native_observed_sessions_exact_terminal_research_v1'
CALENDAR_CANDIDATE='xau_ny_regular_hours_historical_UNQUALIFIED_v1'
NY='America/New_York'

def regular_calendar(start_ns: int, end_ns: int):
    start=pd.Timestamp(start_ns,tz='UTC').tz_convert(NY).tz_localize(None).normalize()-pd.Timedelta(days=40)
    end=pd.Timestamp(end_ns,tz='UTC').tz_convert(NY).tz_localize(None).normalize()+pd.Timedelta(days=40)
    days=pd.date_range(start,end,freq='D')
    days=days[np.isin(days.dayofweek,[6,0,1,2,3])]
    opens=days+pd.Timedelta(hours=18,minutes=5)
    closes=days+pd.Timedelta(days=1,hours=16,minutes=59)
    return {'open_local':opens,'open_ns':opens.tz_localize(NY).tz_convert('UTC').asi8,
            'close_ns':closes.tz_localize(NY).tz_convert('UTC').asi8,
            'historical_calendar_qualified':False,'holiday_overrides_applied':False}

def is_regular_open(times, calendar):
    times=np.asarray(times,np.int64)
    i=np.searchsorted(calendar['open_ns'],times,side='right')-1
    c=np.clip(i,0,len(calendar['open_ns'])-1)
    return (i>=0)&(times>=calendar['open_ns'][c])&(times<calendar['close_ns'][c])

def calendar_terminal(times, bid, ask, decisions, horizon, split_end, calendar):
    """Never skip a missing weekday session because its prices are absent.

    Same LOCAL wall-clock offset from the NY session open, across DST. This is
    a candidate only, not an approved historical calendar/target replacement.
    """
    times=np.asarray(times,np.int64);decisions=np.asarray(decisions,np.int64)
    session=np.searchsorted(calendar['open_ns'],decisions,side='right')-1
    i=np.clip(session,0,len(calendar['open_ns'])-1)
    ti=np.minimum(i+horizon,len(calendar['open_ns'])-1)
    local=pd.DatetimeIndex(decisions,tz='UTC').tz_convert(NY).tz_localize(None)
    offset=local-calendar['open_local'][i]
    desired_local=calendar['open_local'][ti]+offset
    desired=desired_local.tz_localize(NY).tz_convert('UTC').asi8
    entry=np.searchsorted(times,decisions);exit=np.searchsorted(times,desired)
    ec=np.minimum(entry,len(times)-1);xc=np.minimum(exit,len(times)-1)
    valid=is_regular_open(decisions,calendar)&(session>=0)&(i+horizon<len(calendar['open_ns']))
    valid &= (entry<len(times))&(times[ec]==decisions)&(exit<len(times))&(times[xc]==desired)
    valid &= (desired<calendar['close_ns'][ti])&(desired<=split_end)
    gross=np.full((len(decisions),2),np.nan)
    gross[valid,0]=(bid[xc[valid]]-ask[ec[valid]])/ask[ec[valid]]*10000
    gross[valid,1]=(bid[ec[valid]]-ask[xc[valid]])/bid[ec[valid]]*10000
    return {'valid':valid,'entry':ec,'exit':xc,'desired_ns':desired,'gross':gross,
            'historical_calendar_qualified':False,'training_authorized':False}

def strict_terminal_mask(reference, times):
    delay=(np.asarray(times)[reference['exit']]-reference['target_ns'])//NS
    return reference['valid'] & (delay>=0) & (delay<=STRICT_TERMINAL_MAX_DELAY_SECONDS),delay

def range_extrema(values, starts, ends, *, maximum):
    """Exact vectorized segment-tree range queries [start,end), no lookahead inputs."""
    a=np.asarray(values,float);starts=np.asarray(starts,np.int64);ends=np.asarray(ends,np.int64)
    if np.any(starts<0) or np.any(ends>len(a)) or np.any(ends<=starts):
        raise ValueError('ENTRY_RESEARCH_EXTREMA_RANGE')
    size=1<<(len(a)-1).bit_length();sentinel=-np.inf if maximum else np.inf
    tree=np.full(2*size,sentinel);tree[size:size+len(a)]=a
    op=np.maximum if maximum else np.minimum
    width=size
    while width>1:
        tree[width//2:width]=op(tree[width:2*width:2],tree[width+1:2*width:2]);width//=2
    left=starts+size;right=ends+size;out=np.full(len(starts),sentinel)
    while np.any(left<right):
        active=left<right;hit=active&((left&1)==1);out[hit]=op(out[hit],tree[left[hit]]);left[hit]+=1
        hit=active&((right&1)==1);right[hit]-=1;out[hit]=op(out[hit],tree[right[hit]])
        left//=2;right//=2
    return out

def gap_inventory(times,calendar):
    """Classify against CURRENT regular template; historical exceptions unresolved."""
    t=np.asarray(times,np.int64)
    starts=t[:-1]+MINUTE;ends=t[1:];take=ends>starts;starts=starts[take];ends=ends[take]
    expected=np.concatenate([np.arange(max(int(a),int(t[0])),min(int(b),int(t[-1])+MINUTE),MINUTE,dtype=np.int64)
                             for a,b in zip(calendar['open_ns'],calendar['close_ns']) if b>t[0] and a<=t[-1]])
    missing_open=np.searchsorted(expected,ends)-np.searchsorted(expected,starts)
    minutes=(ends-starts)//MINUTE
    return pd.DataFrame({'start_ns':starts,'end_ns':ends,'missing_minutes':minutes,
                         'missing_during_current_regular_template':missing_open,
                         'current_template_closed_minutes':minutes-missing_open,
                         'classification':np.where(missing_open==0,'REGULAR_TEMPLATE_CLOSED_HISTORICAL_UNQUALIFIED','MISSING_QUOTES_OR_UNQUALIFIED_HOLIDAY')})
