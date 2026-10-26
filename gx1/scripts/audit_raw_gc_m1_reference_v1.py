"""Independent raw GC MBP-1 to M1 references for offline source qualification.

No production reducers, model, fit, broker or implicit raw-file execution.
The admitted caller owns study/TEST seal, exact inventory and final receipts.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import os
import stat as statmod
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

# Independent values only: no production reducer, raw parser, or target reader.
_SOURCE_TRADE_MINUTE134=60000000000
_SOURCE_TRADE_U64134=(1<<64)-1
_SOURCE_TRADE_UNDEF_PRICE134=(1<<63)-1
_SOURCE_TRADE_UNDEF_SIZE134=(1<<32)-1
_SOURCE_TRADE_COLUMNS134=('ts_recv','ts_event','rtype','publisher_id','instrument_id','action','side','depth',
    'price','size','flags','ts_in_delta','sequence','bid_px_00','ask_px_00','bid_sz_00','ask_sz_00',
    'bid_ct_00','ask_ct_00','symbol')
_SOURCE_TRADE_SIGNATURE134=('ts_recv','ts_event','sequence','action','side','price','size','flags')
_SOURCE_TRADE_METRICS134=('records','trades','volume','buy','sell','unknown','invalid_trade',
    'bad_recv_trade','gap','snapshot_trade','duplicate_trade')
_SOURCE_TRADE_QUALITY134=('records','clean_complete_bbo','bad_bbo','gap','bad_recv','snapshot','reset',
    'duplicate_trade_signature','full_record_duplicate_trade','signature_collision','file_seam')
_SOURCE_TRADE_SIZE134=('size_sum_contracts','size_sum_squared_contracts2','size_min_contracts',
    'size_max_contracts','buy_trade_records','sell_trade_records','unknown_trade_records')


def _source_trade_group_sums134(values,starts,label):
    """Checked exact integer grouping; the input grouping comes from the caller."""
    values=np.asarray(values);starts=np.asarray(starts)
    if values.dtype!=np.dtype('uint64') or values.ndim!=1 or starts.dtype!=np.dtype('int64'):
        raise RuntimeError('RAW_TRADE134_GROUP_SCHEMA')
    if not len(starts):return np.zeros(0,np.uint64)
    counts=np.diff(np.r_[starts,len(values)])
    if starts[0]!=0 or np.any(counts<=0):raise RuntimeError('RAW_TRADE134_GROUP_SCHEMA')
    # This product is a Python integer, never a wrapped uint64 or float bound.
    if int(values.max())*int(counts.max())<=_SOURCE_TRADE_U64134:
        return np.add.reduceat(values,starts,dtype=np.uint64)
    out=np.empty(len(starts),np.uint64)
    for group,(begin,count) in enumerate(zip(starts,counts)):
        total=sum(map(int,values[int(begin):int(begin+count)]))
        if total>_SOURCE_TRADE_U64134:raise RuntimeError('RAW_TRADE134_UINT64_OVERFLOW:'+label)
        out[group]=total
    return out


def _source_trade_accumulate134(target,local,weights,label):
    local=np.asarray(local)
    if not len(local):return
    if local.dtype!=np.dtype('int64') or np.any(local[1:]<local[:-1]):
        raise RuntimeError('RAW_TRADE134_UNORDERED_MINUTE_GROUPS')
    starts=np.r_[0,np.flatnonzero(local[1:]!=local[:-1])+1].astype(np.int64)
    unique=local[starts]
    added=(np.diff(np.r_[starts,len(local)]).astype(np.uint64) if weights is None else
        _source_trade_group_sums134(weights,starts,label))
    old=target[unique]
    if np.any(old>np.uint64(_SOURCE_TRADE_U64134)-added):
        raise RuntimeError('RAW_TRADE134_UINT64_OVERFLOW:'+label)
    target[unique]=old+added


class RawTradeQualityReference134:
    """One declared file, bounded local M1 arrays and one preceding raw row.

    Root owns CSV parsing, exact twenty-column schema/identity/order/window
    validation and EOF/hash admission. No file is opened by this reference.
    """
    def __init__(self,meta):
        names=('start_ns','end_ns','study_origin_ns','tick_size','raw_symbol','instrument_id','publisher_id')
        if any(name not in meta for name in names):raise RuntimeError('RAW_TRADE134_META')
        self.meta={name:meta[name] for name in names}
        for name in names:
            if name!='raw_symbol' and (isinstance(meta[name],bool) or not isinstance(meta[name],int)):
                raise RuntimeError('RAW_TRADE134_META')
        self.start=meta['start_ns'];self.end=meta['end_ns'];self.origin=meta['study_origin_ns'];self.tick=meta['tick_size']
        if (not 0<=self.origin<=self.start<self.end<=(1<<63)-1 or self.tick<=0
            or any(value%_SOURCE_TRADE_MINUTE134 for value in (self.start,self.end,self.origin))):
            raise RuntimeError('RAW_TRADE134_META')
        self.width=(self.end-self.start)//_SOURCE_TRADE_MINUTE134
        self.offset=(self.start-self.origin)//_SOURCE_TRADE_MINUTE134
        self.metrics={name:np.zeros(self.width,np.uint64) for name in _SOURCE_TRADE_METRICS134}
        self.quality={name:np.zeros(self.width,np.uint64) for name in _SOURCE_TRADE_QUALITY134}
        self.ohlc={name:np.zeros(self.width,np.int64) for name in ('open_tick','high_tick','low_tick','close_tick')}
        self.trade_size={name:np.zeros(self.width,np.uint64) for name in _SOURCE_TRADE_SIZE134}
        self.trade_size['size_min_contracts'][:]=np.uint64(_SOURCE_TRADE_U64134)
        self.previous=None;self.rows=0;self.batches=0;self.finished=False

    def _empty_context(self):
        out={'minute_index':np.zeros(0,np.int64),'local_index':np.zeros(0,np.int64),
            'trusted_trade':np.zeros(0,bool),'duplicate_trade_signature':np.zeros(0,bool),
            'full_record_duplicate_trade':np.zeros(0,bool),
            'ladder':{'minute_index':np.zeros(0,np.int64),'price_tick':np.zeros(0,np.int64),
                'buy':np.zeros(0,np.uint64),'sell':np.zeros(0,np.uint64),'unknown':np.zeros(0,np.uint64)}}
        return self._readonly(out)

    @staticmethod
    def _readonly(context):
        for key,value in context.items():
            if isinstance(value,dict):
                for array in value.values():array.setflags(write=False)
            else:value.setflags(write=False)
        return context

    def consume(self,data):
        if self.finished:raise RuntimeError('RAW_TRADE134_ALREADY_FINISHED')
        if set(data)!=set(_SOURCE_TRADE_COLUMNS134):raise RuntimeError('RAW_TRADE134_COLUMNS')
        if any(not isinstance(data[name],np.ndarray) or data[name].ndim!=1 for name in _SOURCE_TRADE_COLUMNS134):
            raise RuntimeError('RAW_TRADE134_ARRAY_SCHEMA')
        recv=data['ts_recv'];n=len(recv)
        if any(len(data[name])!=n for name in _SOURCE_TRADE_COLUMNS134):raise RuntimeError('RAW_TRADE134_ARRAY_SHAPE')
        if recv.dtype!=np.dtype('uint64'):raise RuntimeError('RAW_TRADE134_RECV_DTYPE')
        if not n:return self._empty_context()
        if int(recv.min())<self.start or int(recv.max())>=self.end:
            raise RuntimeError('RAW_TRADE134_FILE_WINDOW')
        local=(recv.astype(np.int64)-self.start)//_SOURCE_TRADE_MINUTE134
        minute=local+self.offset
        action,side,flags,price,size=(data[name] for name in ('action','side','flags','price','size'))
        trade=action=='T';bad_recv=(flags&8)!=0;snapshot=(flags&32)!=0;gap=(flags&4)!=0
        invalid=trade&((size==0)|(size==_SOURCE_TRADE_UNDEF_SIZE134)|(price<=0)|
            (price==_SOURCE_TRADE_UNDEF_PRICE134)|(price%self.tick!=0))
        trusted=trade&~bad_recv&~snapshot&~invalid
        duplicate=trade.copy()
        duplicate[0]=bool(trade[0] and self.previous is not None and all(
            data[name][0]==self.previous[name] for name in _SOURCE_TRADE_SIGNATURE134))
        for name in _SOURCE_TRADE_SIGNATURE134:duplicate[1:]&=data[name][1:]==data[name][:-1]
        suspects=np.flatnonzero(duplicate);full=np.zeros(n,bool)
        if len(suspects):
            same=np.ones(len(suspects),bool);before=np.maximum(suspects-1,0)
            for name in _SOURCE_TRADE_COLUMNS134:
                equal=data[name][suspects]==data[name][before]
                if suspects[0]==0:equal[0]=self.previous is not None and data[name][0]==self.previous[name]
                same&=equal
            full[suspects]=same
        bid,ask,bsz,asz=(data[name] for name in ('bid_px_00','ask_px_00','bid_sz_00','ask_sz_00'))
        bad_bbo=((bid<=0)|(ask<=0)|(bid==_SOURCE_TRADE_UNDEF_PRICE134)|(ask==_SOURCE_TRADE_UNDEF_PRICE134)|
            (bid>=ask)|(bid%self.tick!=0)|(ask%self.tick!=0)|(bsz==0)|(asz==0)|
            (bsz==_SOURCE_TRADE_UNDEF_SIZE134)|(asz==_SOURCE_TRADE_UNDEF_SIZE134))
        reset=action=='R';clean=((flags&128)!=0)&~bad_bbo&~gap&~bad_recv&~snapshot&~reset
        seam=np.zeros(n,bool)
        if self.rows==0:seam[0]=True
        metric_masks={'records':~bad_recv,'trades':trusted,'invalid_trade':invalid,'bad_recv_trade':trade&bad_recv,
            'gap':gap,'snapshot_trade':trade&snapshot,'duplicate_trade':duplicate}
        for name,take in metric_masks.items():
            _source_trade_accumulate134(self.metrics[name],local[take],None,name)
        quality_masks={'records':np.ones(n,bool),'clean_complete_bbo':clean,'bad_bbo':bad_bbo,'gap':gap,
            'bad_recv':bad_recv,'snapshot':snapshot,'reset':reset,'duplicate_trade_signature':duplicate,
            'full_record_duplicate_trade':full,'signature_collision':duplicate&~full,'file_seam':seam}
        for name,take in quality_masks.items():
            _source_trade_accumulate134(self.quality[name],local[take],None,'quality.'+name)
        indices=np.flatnonzero(trusted);ladder=self._empty_context()['ladder']
        if len(indices):
            loc=local[indices];glob=minute[indices];ticks=(price[indices]//self.tick).astype(np.int64)
            quantities=size[indices].astype(np.uint64);sides=side[indices]
            _source_trade_accumulate134(self.metrics['volume'],loc,quantities,'volume')
            _source_trade_accumulate134(self.trade_size['size_sum_contracts'],loc,quantities,'size_sum_contracts')
            _source_trade_accumulate134(self.trade_size['size_sum_squared_contracts2'],loc,
                quantities*quantities,'size_sum_squared_contracts2')
            for code,label in (('B','buy'),('A','sell'),('N','unknown')):
                take=sides==code
                _source_trade_accumulate134(self.metrics[label],loc[take],quantities[take],label)
                _source_trade_accumulate134(self.trade_size[label+'_trade_records'],loc[take],None,label+'_trade_records')
            starts=np.r_[0,np.flatnonzero(loc[1:]!=loc[:-1])+1].astype(np.int64)
            unique=loc[starts];ends=np.r_[starts[1:],len(loc)]-1
            empty=self.ohlc['open_tick'][unique]==0
            self.ohlc['open_tick'][unique[empty]]=ticks[starts[empty]]
            self.ohlc['close_tick'][unique]=ticks[ends]
            self.ohlc['high_tick'][unique]=np.maximum(self.ohlc['high_tick'][unique],np.maximum.reduceat(ticks,starts))
            lows=np.minimum.reduceat(ticks,starts)
            self.ohlc['low_tick'][unique]=np.where(empty,lows,np.minimum(self.ohlc['low_tick'][unique],lows))
            minima=np.minimum.reduceat(quantities,starts);maxima=np.maximum.reduceat(quantities,starts)
            self.trade_size['size_min_contracts'][unique]=np.minimum(self.trade_size['size_min_contracts'][unique],minima)
            self.trade_size['size_max_contracts'][unique]=np.maximum(self.trade_size['size_max_contracts'][unique],maxima)
            # Different grouping algorithm from the producer's recarray/unique:
            # sort (global minute,tick), identify adjacent changes, reduce once.
            order=np.lexsort((ticks,glob));g=glob[order];p=ticks[order];q=quantities[order];s=sides[order]
            first=np.r_[0,np.flatnonzero((g[1:]!=g[:-1])|(p[1:]!=p[:-1]))+1].astype(np.int64)
            ladder={'minute_index':g[first],'price_tick':p[first]}
            for code,label in (('B','buy'),('A','sell'),('N','unknown')):
                ladder[label]=_source_trade_group_sums134(np.where(s==code,q,np.uint64(0)),first,'ladder.'+label)
        self.previous={name:(data[name][-1].item() if isinstance(data[name][-1],np.generic) else data[name][-1])
            for name in _SOURCE_TRADE_COLUMNS134}
        self.rows+=n;self.batches+=1
        return self._readonly({'minute_index':minute,'local_index':local,'trusted_trade':trusted,
            'duplicate_trade_signature':duplicate,'full_record_duplicate_trade':full,'ladder':ladder})

    def finish(self):
        if self.finished:raise RuntimeError('RAW_TRADE134_ALREADY_FINISHED')
        if not self.rows:raise RuntimeError('RAW_TRADE134_EMPTY_FILE')
        for i in range(self.width):
            volume=sum(int(self.metrics[name][i]) for name in ('buy','sell','unknown'))
            records=sum(int(self.trade_size[name+'_trade_records'][i]) for name in ('buy','sell','unknown'))
            if (volume!=int(self.metrics['volume'][i]) or int(self.trade_size['size_sum_contracts'][i])!=volume
                    or records!=int(self.metrics['trades'][i])):
                raise RuntimeError('RAW_TRADE134_INTERNAL_CONSERVATION')
        self.trade_size['size_min_contracts'][self.metrics['trades']==0]=0
        minute=np.arange(self.offset,self.offset+self.width,dtype=np.int64)
        report={'raw_rows':self.rows,'nonempty_batches':self.batches,'local_M1_rows':self.width,
            'first_global_minute_index':self.offset,'last_global_minute_index':self.offset+self.width-1,
            'source_metric_count':11,'quality_counter_count':11,'trade_size_field_count':7,
            'trusted_T_records':sum(map(int,self.metrics['trades'])),'T_duplicates_retained':True,
            'full_record_fields_compared':list(_SOURCE_TRADE_COLUMNS134),'previous_full_record_carries_across_batch':True,
            'state_scope':'one file; new instance clears duplicate predecessor and marks first raw row as file seam',
            'all_integer_arithmetic_checked':True,'ladder_scope':'current batch only; root merges admitted fragments',
            'ladder_grouping':'lexsort(minute,price_tick) then checked adjacent reduceat',
            'identity_schema_order_and_EOF_admission_owned_by_root':True,'production_reducers_called':False,
            'raw_files_opened_by_reference':0,'floating_reference_values':0,'training_enabled':False}
        self.finished=True
        for mapping in (self.metrics,self.quality,self.ohlc,self.trade_size):
            for value in mapping.values():value.setflags(write=False)
        minute.setflags(write=False)
        return {'minute_index':minute,'source_metrics11':self.metrics,'quality11':self.quality,
            'ohlc':self.ohlc,'trade_size7':self.trade_size,'reference_report':report}

# Integration fragment: generic numpy is supplied by the capped audit module.
_BOOK_MINUTE134 = 60_000_000_000
_BOOK_U134 = 2.**-53
_BOOK_ETA134 = float.fromhex('0x0.0000000000001p-1022')
_BOOK_NAMES134 = ('ofi_contracts','bid_refill_contracts','ask_refill_contracts',
    'bid_depletion_contracts','ask_depletion_contracts','quote_updates',
    'tw_imbalance','tw_spread_ticks','tw_bid_contracts','tw_ask_contracts',
    'tw_microprice_offset_ticks','mid_response_ticks')
_BOOK_ORDER134 = ('tw_bid_orders','tw_ask_orders','tw_order_count_imbalance',
    'tw_bid_contracts_per_order','tw_ask_contracts_per_order')


def _book_up134(value):
    value=np.asarray(value,dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value<0.):raise RuntimeError('RAW_BOOK_BOUND_DOMAIN')
    out=np.nextafter(value,np.inf)
    if not np.isfinite(out).all():raise RuntimeError('RAW_BOOK_BOUND_OVERFLOW')
    return out


def _book_add134(a,b):
    a,b=np.broadcast_arrays(a,b);out=_book_up134(a+b)
    return np.where((a==0.)&(b==0.),0.,out)


def _book_mul134(a,b):
    a,b=np.broadcast_arrays(a,b);out=_book_up134(a*b)
    return np.where((a==0.)|(b==0.),0.,out)


def _book_div134(a,b):
    a,b=np.broadcast_arrays(a,b)
    if np.any(~np.isfinite(b)) or np.any(b<=0.):raise RuntimeError('RAW_BOOK_BOUND_DENOMINATOR')
    out=_book_up134(a/b)
    return np.where(a==0.,0.,out)


def _book_rho134(value):
    # |RN(x)-x| <= u/(1-u)*|RN(x)| + eta. Exact zero nodes are
    # identified from their integer-zero numerators/products, not samples.
    v=np.asarray(value,dtype=np.float64)
    if not np.isfinite(v).all():raise RuntimeError('RAW_BOOK_FLOAT_OVERFLOW')
    bound=_book_add134(_book_div134(_book_mul134(_BOOK_U134,np.abs(v)),1.-_BOOK_U134),_BOOK_ETA134)
    return np.where(v==0.,0.,bound)


def _book_gamma134(count):
    n=np.asarray(count)
    if np.any(n<0) or np.any(n>=2**53):raise RuntimeError('RAW_BOOK_ACCUMULATION_COUNT')
    nu=_book_mul134(n.astype(np.float64),_BOOK_U134)
    den=np.nextafter(1.-nu,-np.inf)
    if np.any(den<=0.):raise RuntimeError('RAW_BOOK_ACCUMULATION_DOMAIN')
    return _book_div134(nu,den)


def _book_group_upper134(values,starts,counts):
    # A reduction of k nonnegative numbers uses at most k-1 additions.
    summed=np.add.reduceat(values,starts,axis=0)
    gamma=_book_gamma134(counts-1)
    if values.ndim==2:gamma=gamma[:,None]
    den=np.nextafter(1.-gamma,-np.inf)
    if np.any(den<=0.):raise RuntimeError('RAW_BOOK_MASS_REDUCTION_DOMAIN')
    return _book_div134(summed,den)


class RawBookReference134:
    """Independent per-file all-F_LAST state; no clean-event or candidate gate.

    Integer transitions and coverage use exact guarded ledgers. Weighted means
    use independent stable grouped reductions, with a source AND reference
    accumulation envelope. No production reducer or target enters this class.
    """
    def __init__(self,meta):
        self.meta=dict(meta)
        for key in ('start_ns','end_ns','study_origin_ns','tick_size'):
            v=self.meta[key]
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)):raise RuntimeError('RAW_BOOK_META_INTEGER')
        self.start=int(meta['start_ns']);self.end=int(meta['end_ns']);self.origin=int(meta['study_origin_ns']);self.tick=int(meta['tick_size'])
        if (self.tick<=0 or self.start<self.origin or self.end<=self.start or self.end>=2**63
                or (self.start-self.origin)%_BOOK_MINUTE134 or (self.end-self.origin)%_BOOK_MINUTE134):
            raise RuntimeError('RAW_BOOK_META_CLOCK')
        self.rows=(self.end-self.start)//_BOOK_MINUTE134
        self.minute_index=np.arange(self.rows,dtype=np.int64)+(self.start-self.origin)//_BOOK_MINUTE134
        self.covered=np.zeros(self.rows,np.int64);self.order_covered=np.zeros(self.rows,np.int64)
        self.transition=np.zeros((self.rows,6),np.int64);self.transition_mass=self.transition.copy()
        self.transition_count=np.zeros(self.rows,np.int64)
        self.tw_sum=np.zeros((self.rows,10));self.tw_mass=self.tw_sum.copy();self.tw_input_error=self.tw_sum.copy()
        self.tw_count=np.zeros((self.rows,2),np.int64)
        self.first=np.full(self.rows,np.nan);self.last=self.first.copy()
        self.first_time=np.full(self.rows,np.iinfo(np.int64).max,np.int64)
        self.last_time=np.full(self.rows,np.iinfo(np.int64).min,np.int64)
        self.previous=None;self.finished=False;self.last_received=None
        self.batches=0;self.records=0;self.final_records=0;self.integration_segments=0;self.order_segments=0

    def _groups(self,index):
        index=np.asarray(index,np.int64)
        if index.ndim!=1 or (len(index) and (index[0]<0 or index[-1]>=self.rows or np.any(index[1:]<index[:-1]))):
            raise RuntimeError('RAW_BOOK_LOCAL_INDEX')
        starts=np.r_[0,np.flatnonzero(index[1:]!=index[:-1])+1]
        return index[starts],starts,np.diff(np.r_[starts,len(index)])

    def _push_transition(self,index,terms):
        if not len(index):return
        ids,starts,counts=self._groups(index);limit=np.iinfo(np.int64).max
        # Guard before reduceat; both exact sum and absolute mass fit int64.
        biggest=int(np.max(np.abs(terms)))
        if biggest*len(terms)>limit:raise RuntimeError('RAW_BOOK_INTEGER_BATCH_OVERFLOW')
        added=np.add.reduceat(terms,starts,axis=0);mass=np.add.reduceat(np.abs(terms),starts,axis=0)
        if np.any(self.transition_mass[ids]>limit-mass):raise RuntimeError('RAW_BOOK_INTEGER_LEDGER_OVERFLOW')
        if np.any(self.transition_count[ids]>limit-counts):raise RuntimeError('RAW_BOOK_COUNT_OVERFLOW')
        self.transition[ids]+=added;self.transition_mass[ids]+=mass;self.transition_count[ids]+=counts

    def _push_tw(self,index,reference,wire,error_reference,error_wire,duration,offset):
        if not len(index):return
        ids,starts,counts=self._groups(index);dt=duration.astype(np.float64)[:,None]
        terms=reference*dt;wire_terms=wire*dt
        if not np.isfinite(terms).all() or not np.isfinite(wire_terms).all():raise RuntimeError('RAW_BOOK_FLOAT_OVERFLOW')
        error=_book_add134(_book_mul134(dt,_book_add134(error_reference,error_wire)),
            _book_add134(_book_rho134(terms),_book_rho134(wire_terms)))
        added=np.add.reduceat(terms,starts,axis=0)
        columns=slice(offset,offset+5);new=self.tw_sum[ids,columns]+added
        if not np.isfinite(new).all():raise RuntimeError('RAW_BOOK_FLOAT_OVERFLOW')
        self.tw_sum[ids,columns]=new
        self.tw_mass[ids,columns]=_book_add134(self.tw_mass[ids,columns],_book_group_upper134(np.abs(terms),starts,counts))
        self.tw_input_error[ids,columns]=_book_add134(self.tw_input_error[ids,columns],_book_group_upper134(error,starts,counts))
        group=0 if offset==0 else 1
        if np.any(self.tw_count[ids,group]>np.iinfo(np.int64).max-counts):raise RuntimeError('RAW_BOOK_COUNT_OVERFLOW')
        self.tw_count[ids,group]+=counts

    def _integrate(self,state,left,right):
        # This method is called separately for the leading and trailing boundary
        # intervals. Their original phase order owns strict first/last ties.
        take=(right>left)&(state[:,1]>0)&(state[:,2]>state[:,1])&(state[:,3]>0)&(state[:,4]>0)
        if not np.any(take):return
        state=state[take];left=left[take];right=right[take]
        index=(left-self.start)//_BOOK_MINUTE134;ids,starts,counts=self._groups(index)
        duration=right-left
        if np.any(duration<=0) or np.any(duration>_BOOK_MINUTE134):raise RuntimeError('RAW_BOOK_INTERVAL_DOMAIN')
        added=np.add.reduceat(duration,starts)
        if np.any(self.covered[ids]>_BOOK_MINUTE134-added):raise RuntimeError('RAW_BOOK_COVERAGE_OVERFLOW')
        self.covered[ids]+=added;self.integration_segments+=len(index)
        bid,ask=state[:,1],state[:,2]
        # All raw prices have already been divided with signed integer floor.
        # Reject unsafe int64 addition before computing the half-tick midpoint.
        limit=np.iinfo(np.int64).max
        if np.any(bid>limit-ask):raise RuntimeError('RAW_BOOK_PRICE_INTEGER_OVERFLOW')
        total=bid+ask
        if np.any(total>2**53):raise RuntimeError('RAW_BOOK_MIDPOINT_EXACT_DOMAIN')
        mid=total.astype(np.float64)*.5
        q=state[:,3].astype(np.float64);r=state[:,4].astype(np.float64)
        numerator=(state[:,3]-state[:,4]).astype(np.float64);denominator=(state[:,3]+state[:,4]).astype(np.float64)
        spread=(ask-bid).astype(np.float64)
        imbalance=numerator/denominator
        # Independent real microprice-offset formula versus the source's
        # ratio-then-multiply wire. The latter is used for its bound only.
        product=spread*numerator;ref_micro=product/(2.*denominator)
        wire_product=spread*imbalance;wire_micro=wire_product*.5
        er_micro=_book_add134(_book_div134(_book_rho134(product),2.*denominator),_book_rho134(ref_micro))
        es_micro=_book_mul134(.5,_book_add134(_book_mul134(spread,_book_rho134(imbalance)),_book_rho134(wire_product)))
        reference=np.column_stack((imbalance,spread,q,r,ref_micro));wire=reference.copy();wire[:,4]=wire_micro
        er=np.zeros_like(reference);es=er.copy();er[:,0]=_book_rho134(imbalance);es[:,0]=er[:,0]
        er[:,4]=er_micro;es[:,4]=es_micro
        self._push_tw(index,reference,wire,er,es,duration,0)
        bc,ac=state[:,5],state[:,6]
        observed=(bc>0)&(ac>0)&(bc<2**32-1)&(ac<2**32-1)
        if np.any(observed):
            oi=index[observed];od=duration[observed];oi_ids,oi_starts,oi_counts=self._groups(oi)
            oadded=np.add.reduceat(od,oi_starts)
            if np.any(self.order_covered[oi_ids]>_BOOK_MINUTE134-oadded):raise RuntimeError('RAW_BOOK_ORDER_COVERAGE_OVERFLOW')
            self.order_covered[oi_ids]+=oadded;self.order_segments+=len(oi)
            b=bc[observed].astype(np.float64);a=ac[observed].astype(np.float64)
            im=(bc[observed]-ac[observed]).astype(np.float64)/(b+a)
            perbid=q[observed]/b;perask=r[observed]/a
            v=np.column_stack((b,a,im,perbid,perask));e=np.zeros_like(v);e[:,2:]=_book_rho134(v[:,2:])
            self._push_tw(oi,v,v,e,e,od,5)
        first=starts;last=starts+counts-1
        earlier=left[first]<self.first_time[ids];later=right[last]>self.last_time[ids]
        self.first[ids[earlier]]=mid[first[earlier]];self.first_time[ids[earlier]]=left[first[earlier]]
        self.last[ids[later]]=mid[last[later]];self.last_time[ids[later]]=right[last[later]]

    def consume(self,data,context):
        if self.finished:raise RuntimeError('RAW_BOOK_ALREADY_FINISHED')
        names=('ts_recv','flags','bid_px_00','ask_px_00','bid_sz_00','ask_sz_00','bid_ct_00','ask_ct_00')
        if any(name not in data for name in names):raise RuntimeError('RAW_BOOK_SOURCE_FIELDS')
        recv=np.asarray(data['ts_recv']);n=len(recv)
        if recv.ndim!=1 or recv.dtype!=np.dtype('uint64') or any(np.shape(data[k])!=(n,) for k in names):raise RuntimeError('RAW_BOOK_SOURCE_SHAPE')
        if not n:return
        if (np.any(recv<self.start) or np.any(recv>=self.end) or np.any(recv[1:]<recv[:-1])
                or (self.last_received is not None and int(recv[0])<self.last_received)):
            raise RuntimeError('RAW_BOOK_SOURCE_CLOCK')
        local=np.asarray(context['local_index'])
        if local.dtype!=np.dtype('int64') or not np.array_equal(local,(recv.astype(np.int64)-self.start)//_BOOK_MINUTE134):raise RuntimeError('RAW_BOOK_CONTEXT_CLOCK')
        self.batches+=1;self.records+=n;self.last_received=int(recv[-1])
        take=(np.asarray(data['flags'])&128)!=0
        if not np.any(take):return
        columns=[recv[take].astype(np.int64)]
        for key in ('bid_px_00','ask_px_00'):
            raw=np.asarray(data[key])
            if raw.dtype!=np.dtype('int64'):raise RuntimeError('RAW_BOOK_PRICE_SCHEMA')
            columns.append(raw[take]//self.tick)
        for key in ('bid_sz_00','ask_sz_00','bid_ct_00','ask_ct_00'):
            raw=np.asarray(data[key])
            if raw.dtype!=np.dtype('uint32'):raise RuntimeError('RAW_BOOK_SIZE_SCHEMA')
            columns.append(raw[take].astype(np.int64))
        current=np.column_stack(columns);self.final_records+=len(current);before=np.empty_like(current)
        before[0]=current[0] if self.previous is None else self.previous;before[1:]=current[:-1]
        anchor=np.ones(len(current),bool)
        if self.previous is None:anchor[0]=False
        b,a,q,r=(current[:,j] for j in (1,2,3,4));B,A,Q,R=(before[:,j] for j in (1,2,3,4))
        eligible=anchor&(B>0)&(A>B)&(b>0)&(a>b)
        if np.any(eligible):
            # Piecewise signed book flow, independent of the producer's four
            # indicator-product expression; no aggressor or cancellation claim.
            bidflow=np.where(b>B,q,np.where(b==B,q-Q,-Q))
            askflow=np.where(a<A,-r,np.where(a==A,R-r,R))
            terms=np.column_stack((bidflow+askflow,np.where((b==B)&(q>Q),q-Q,0),
                np.where((a==A)&(r>R),r-R,0),np.where((b==B)&(q<Q),Q-q,0),
                np.where((a==A)&(r<R),R-r,0),np.count_nonzero(current[:,1:5]!=before[:,1:5],axis=1)>0))
            self._push_transition((current[eligible,0]-self.start)//_BOOK_MINUTE134,terms[eligible])
        left=before[:,0];right=current[:,0];active=anchor&(right>=left)
        boundary=self.start+((left-self.start)//_BOOK_MINUTE134+1)*_BOOK_MINUTE134
        self._integrate(before[active],left[active],np.minimum(right,boundary)[active])
        cross=active&(right>boundary)
        if np.any(cross):
            ending=self.start+(right-self.start)//_BOOK_MINUTE134*_BOOK_MINUTE134
            second=np.maximum(boundary,ending)
            self._integrate(before[cross],second[cross],right[cross])
        self.previous=current[-1].copy()

    def finish(self):
        if self.finished:raise RuntimeError('RAW_BOOK_ALREADY_FINISHED')
        self.finished=True
        if self.previous is not None:
            left=int(self.previous[0]);right=min(self.end,self.start+((left-self.start)//_BOOK_MINUTE134+1)*_BOOK_MINUTE134)
            if self.end<left:raise RuntimeError('RAW_BOOK_EOF_BEFORE_STATE')
            self._integrate(self.previous[None,:],np.array([left],np.int64),np.array([right],np.int64))
        book={};bounds={};count=self.transition_count[:,None]
        mass=self.transition_mass.astype(np.float64);mass_upper=_book_up134(mass)
        integer_error=_book_add134(_book_mul134(_book_gamma134(count),mass_upper),_book_rho134(self.transition.astype(np.float64)))
        integer_error=np.where(self.transition_mass<=2**53,0.,integer_error)
        for j,name in enumerate(_BOOK_NAMES134[:6]):book[name]=self.transition[:,j].astype(np.float64);bounds[name]=integer_error[:,j]
        full=self.covered==_BOOK_MINUTE134;order_full=self.order_covered==_BOOK_MINUTE134
        for j,name in enumerate(_BOOK_NAMES134[6:11]+_BOOK_ORDER134):
            group=0 if j<5 else 1;gamma=_book_gamma134(self.tw_count[:,group])
            M=self.tw_mass[:,j];E=self.tw_input_error[:,j]
            pair_sum_error=_book_add134(E,_book_add134(_book_mul134(gamma,_book_add134(M,E)),_book_mul134(gamma,M)))
            value=self.tw_sum[:,j]/float(_BOOK_MINUTE134)
            source_value_magnitude=_book_div134(_book_add134(np.abs(self.tw_sum[:,j]),pair_sum_error),float(_BOOK_MINUTE134))
            error=_book_add134(_book_div134(pair_sum_error,float(_BOOK_MINUTE134)),
                _book_add134(_book_rho134(value),_book_rho134(source_value_magnitude)))
            mask=full if j<5 else order_full
            book[name]=np.where(mask,value,np.nan);bounds[name]=np.where(mask,error,np.nan)
        response=self.last-self.first;response_error=_book_mul134(2.,_book_rho134(np.where(np.isfinite(response),response,0.)))
        book['mid_response_ticks']=np.where(full,response,np.nan);bounds['mid_response_ticks']=np.where(full,response_error,np.nan)
        for name,value in [('covered_ns',self.covered),('order_covered_ns',self.order_covered),
                ('opening_mid_tick',self.first),('closing_mid_tick',self.last)]:
            book[name]=value;bounds[name]=np.where(np.isfinite(value),0.,np.nan)
        names=(*_BOOK_NAMES134,*_BOOK_ORDER134,'covered_ns','order_covered_ns','opening_mid_tick','closing_mid_tick')
        book={name:book[name] for name in names};bounds={name:bounds[name] for name in names}
        return {'minute_index':self.minute_index,'book':book,'bounds':bounds,'reference_report':{
            'schema_version':'raw_book_reference_134_v1','rows':self.rows,'records':self.records,'batches':self.batches,
            'F_LAST_records':self.final_records,'integration_segments':self.integration_segments,'order_segments':self.order_segments,
            'full_price_coverage_rows':int(np.count_nonzero(full)),'full_order_coverage_rows':int(np.count_nonzero(order_full)),
            'per_file_state_reset':True,'batch_state_carry':True,'event_clean_mask_used':False,
            'gap_integration':'only_leading_and_trailing_boundary_minutes','EOF':'first_boundary_clamp_only',
            'floating_accumulation':'source_and_reference_gamma_n_absolute_mass_plus_per_node_input_error',
            'transition_integer_mass_max':int(self.transition_mass.max()),
            'maximum_TW_additions_per_minute':int(self.tw_count.max()),'production_reducer_calls':0}}

# Independent event diagnostics. Input validation/reading belongs to the shared reader.
_EVENT134_MINUTE = 60_000_000_000
_EVENT134_I64_MAX = (1 << 63)-1
_EVENT134_I64_MIN = -(1 << 63)
_EVENT134_U = 2.**-53
_EVENT134_LATENCIES = ('send_to_capture','receive_minus_event','matching_clock')
EVENT134_REDUCERS = {
    'qualified_final_event_count':'sum','spread_state_min_ticks':'min','spread_state_max_ticks':'max',
    'spread_change_count':'sum','spread_expansion_count':'sum','spread_contraction_count':'sum',
    'spread_expansion_sum_ticks':'sum','spread_contraction_sum_ticks':'sum',
    'spread_max_expansion_ticks':'max','spread_max_contraction_ticks':'max',
    'receive_interval_count':'sum','receive_interval_sum_ns':'sum',
    'receive_interval_min_ns':'min','receive_interval_max_ns':'max',
    'ts_in_delta_clamped_low_count':'sum','ts_in_delta_clamped_high_count':'sum',
    'undefined_event_clock_count':'sum','negative_send_to_capture_count':'sum',
    'negative_matching_clock_count':'sum',
    **{p+s:op for p in _EVENT134_LATENCIES for s,op in
       (('_count','sum'),('_sum_ns','sum'),('_min_ns','min'),('_max_ns','max'))},
}
EVENT134_FIELDS = tuple(EVENT134_REDUCERS)+tuple(p+'_mean_ns' for p in
    ('receive_interval',)+_EVENT134_LATENCIES)+('qualified_covered_ns','qualified_events_per_observed_second')


def _event134_up(x):
    x=float(x)
    if x<0. or not np.isfinite(x):raise RuntimeError('RAW_EVENT134_RADIUS_DOMAIN')
    return 0. if x==0. else float(np.nextafter(x,np.inf))


def _event134_sum_radius(total,mass,count):
    # Every cast and every sequential addition is covered. If all partial
    # integer sums fit in53bits, all these particular operations are exact.
    if mass <= 1 << 53:return 0.
    k=count+1
    if k >= 1 << 53:raise RuntimeError('RAW_EVENT134_SUM_OPERATION_DOMAIN')
    ku=_event134_up(float(k)*_EVENT134_U)
    lower=float(np.nextafter(1.-ku,-np.inf))
    if lower<=0.:raise RuntimeError('RAW_EVENT134_SUM_OPERATION_DOMAIN')
    gamma=_event134_up(ku/lower)
    represented=float(total)
    residual=abs(int(represented)-total)
    return _event134_up(_event134_up(gamma*_event134_up(float(mass)))+_event134_up(float(residual)))


def _event134_quotient(numerator,denominator,radius):
    out=np.full(len(numerator),np.nan);bound=out.copy()
    valid=np.isfinite(numerator)&np.isfinite(denominator)&(denominator>0.)
    out[valid]=numerator[valid]/denominator[valid]
    exact=valid&(radius==0.);bound[exact]=0.
    uncertain=valid&(radius>0.)
    if np.any(uncertain):
        lo=np.nextafter(numerator[uncertain]-radius[uncertain],-np.inf)
        hi=np.nextafter(numerator[uncertain]+radius[uncertain],np.inf)
        lower=np.nextafter(lo/denominator[uncertain],-np.inf)
        upper=np.nextafter(hi/denominator[uncertain],np.inf)
        bound[uncertain]=np.nextafter(np.maximum(out[uncertain]-lower,upper-out[uncertain]),np.inf)
    return out,bound


class RawEventReference134:
    """One per-file raw-event ledger; all F_LAST states survive batch seams."""
    def __init__(self,meta):
        self.meta=dict(meta)
        for name in ('start_ns','end_ns','study_origin_ns','tick_size'):
            if isinstance(meta.get(name),bool) or not isinstance(meta.get(name),(int,np.integer)):
                raise RuntimeError('RAW_EVENT134_META')
        self.start=int(meta['start_ns']);self.end=int(meta['end_ns']);self.origin=int(meta['study_origin_ns'])
        self.tick=int(meta['tick_size'])
        if (self.tick<=0 or self.origin<0 or self.start<self.origin or self.end<=self.start or self.end>_EVENT134_I64_MAX
                or any(t%_EVENT134_MINUTE for t in (self.start,self.end,self.origin))):
            raise RuntimeError('RAW_EVENT134_META')
        self.n=(self.end-self.start)//_EVENT134_MINUTE
        self.minute_index=np.arange(self.n,dtype=np.int64)+(self.start-self.origin)//_EVENT134_MINUTE
        self._values={k:([0]*self.n if op=='sum' else [None]*self.n) for k,op in EVENT134_REDUCERS.items()}
        self._mass={k:[0]*self.n for k,op in EVENT134_REDUCERS.items() if op=='sum'}
        self._operations={k:[0]*self.n for k in self._mass}
        self.previous=None;self.rows=0;self.final_rows=0;self.clean_rows=0;self.batches=0;self.finished=False

    def _accumulate(self,name,minute,values):
        if not len(minute):return
        # Stable minute grouping changes no event qualification or predecessor.
        # Integer sufficient statistics are independent of grouping order.
        m=np.asarray(minute,np.int64);v=np.asarray(values,np.int64)
        if np.any(m[1:]<m[:-1]):
            order=np.argsort(m,kind='stable');m=m[order];v=v[order]
        first=np.r_[0,np.flatnonzero(m[1:]!=m[:-1])+1];counts=np.diff(np.r_[first,len(m)])
        op=EVENT134_REDUCERS[name]
        if op=='sum':
            largest=max(abs(int(v.min())),abs(int(v.max())))
            safe=largest*int(counts.max())<=_EVENT134_I64_MAX
            if safe:
                sums=np.add.reduceat(v,first);masses=np.add.reduceat(np.abs(v),first)
            else:
                # Rare wide signed clock domains cannot wrap a NumPy int64.
                sums=[sum(map(int,v[s:s+c])) for s,c in zip(first,counts)]
                masses=[sum(abs(int(x)) for x in v[s:s+c]) for s,c in zip(first,counts)]
            for i,s,count,total,mass in zip(m[first],first,counts,sums,masses):
                j=int(i);self._values[name][j]+=int(total);self._mass[name][j]+=int(mass)
                self._operations[name][j]+=int(count)
        else:
            reduced=(np.minimum.reduceat if op=='min' else np.maximum.reduceat)(v,first)
            for i,value in zip(m[first],reduced):
                j=int(i);previous=self._values[name][j];new=int(value)
                self._values[name][j]=new if previous is None else (min(previous,new) if op=='min' else max(previous,new))

    def consume(self,data,context):
        if self.finished:raise RuntimeError('RAW_EVENT134_FINISHED')
        required=('ts_recv','ts_event','flags','action','bid_px_00','ask_px_00','bid_sz_00','ask_sz_00','ts_in_delta')
        if any(k not in data for k in required):raise RuntimeError('RAW_EVENT134_SOURCE_SCHEMA')
        lengths={len(np.asarray(data[k])) for k in required}
        if len(lengths)!=1:raise RuntimeError('RAW_EVENT134_SOURCE_SHAPE')
        n=lengths.pop();self.rows+=n;self.batches+=1
        if not n:return
        ts=np.asarray(data['ts_recv'])
        if ts.dtype!=np.dtype('uint64') or np.any(ts>_EVENT134_I64_MAX):raise RuntimeError('RAW_EVENT134_RECEIVE_DOMAIN')
        recv=ts.astype(np.int64)
        if np.any((recv<self.start)|(recv>=self.end)):raise RuntimeError('RAW_EVENT134_FILE_WINDOW')
        local=(recv-self.start)//_EVENT134_MINUTE
        if ('local_index' not in context or not np.array_equal(context['local_index'],local)
                or 'minute_index' not in context or not np.array_equal(context['minute_index'],local+self.minute_index[0])):
            raise RuntimeError('RAW_EVENT134_CONTEXT_CLOCK')
        flags=np.asarray(data['flags']);take=(flags&128)!=0
        if not np.any(take):return
        time=recv[take];minute=local[take]
        bid=np.asarray(data['bid_px_00'],np.int64)[take]//self.tick
        ask=np.asarray(data['ask_px_00'],np.int64)[take]//self.tick
        bs=np.asarray(data['bid_sz_00'],np.int64)[take];az=np.asarray(data['ask_sz_00'],np.int64)[take]
        clean=((flags[take]&44)==0)&(bid>0)&(ask>bid)&(bs>0)&(az>0)&(np.asarray(data['action'])[take]!='R')
        spread=np.where(clean,ask-bid,0)
        prior_time=np.r_[time[0] if self.previous is None else self.previous[0],time[:-1]]
        prior_spread=np.r_[0 if self.previous is None else self.previous[1],spread[:-1]]
        prior_clean=np.r_[False if self.previous is None else self.previous[2],clean[:-1]]
        transition=clean&prior_clean&(time>=prior_time)
        one=np.ones(len(time),np.int64)
        def add(name,mask,value):self._accumulate(name,minute[mask],np.asarray(value,np.int64)[mask])
        add('qualified_final_event_count',clean,one)
        add('spread_state_min_ticks',clean,spread);add('spread_state_max_ticks',clean,spread)
        add('spread_state_min_ticks',transition,prior_spread);add('spread_state_max_ticks',transition,prior_spread)
        change=spread-prior_spread;up=np.maximum(change,0);down=np.maximum(-change,0)
        add('spread_change_count',transition&(change!=0),one)
        add('spread_expansion_count',transition&(change>0),one);add('spread_contraction_count',transition&(change<0),one)
        add('spread_expansion_sum_ticks',transition,up);add('spread_contraction_sum_ticks',transition,down)
        add('spread_max_expansion_ticks',transition,up);add('spread_max_contraction_ticks',transition,down)
        interval=time-prior_time
        for suffix,value in (('count',one),('sum_ns',interval),('min_ns',interval),('max_ns',interval)):
            add('receive_interval_'+suffix,transition,value)
        raw_delta=np.asarray(data['ts_in_delta'])
        if raw_delta.dtype!=np.dtype('int32'):raise RuntimeError('RAW_EVENT134_DELTA_DOMAIN')
        delta=raw_delta[take].astype(np.int64);low=delta==-(1<<31);high=delta==(1<<31)-1
        raw_event=np.asarray(data['ts_event'])
        if raw_event.dtype!=np.dtype('uint64'):raise RuntimeError('RAW_EVENT134_EVENT_CLOCK_DOMAIN')
        raw_event=raw_event[take];defined=raw_event<=_EVENT134_I64_MAX
        add('ts_in_delta_clamped_low_count',clean&low,one);add('ts_in_delta_clamped_high_count',clean&high,one)
        add('undefined_event_clock_count',clean&~defined,one)
        event=np.where(defined,raw_event,0).astype(np.int64)
        # recv and definedevent are nonnegative signed64, so their difference
        # cannot overflow. A subsequent matching subtraction needs its proof.
        difference=time-event
        matching_mask=clean&defined&~low&~high
        active=np.flatnonzero(matching_mask)
        if len(active):
            d=difference[active];x=delta[active]
            bad=np.zeros(len(x),bool);positive=x>0;negative=x<0
            bad[positive]=d[positive]<_EVENT134_I64_MIN+x[positive]
            bad[negative]=d[negative]>_EVENT134_I64_MAX+x[negative]
            if np.any(bad):raise RuntimeError('RAW_EVENT134_MATCHING_CLOCK_OVERFLOW')
        matching=np.zeros(len(time),np.int64);matching[active]=difference[active]-delta[active]
        for prefix,value,mask in (
            ('send_to_capture',delta,clean&~low&~high),
            ('receive_minus_event',difference,clean&defined),
            ('matching_clock',matching,matching_mask)):
            for suffix,vector in (('_count',one),('_sum_ns',value),('_min_ns',value),('_max_ns',value)):
                add(prefix+suffix,mask,vector)
        add('negative_send_to_capture_count',clean&~low&~high&(delta<0),one)
        add('negative_matching_clock_count',matching_mask&(matching<0),one)
        self.previous=(int(time[-1]),int(spread[-1]),bool(clean[-1]))
        self.final_rows+=len(time);self.clean_rows+=int(np.count_nonzero(clean))

    def finish(self,book_result):
        if self.finished:raise RuntimeError('RAW_EVENT134_FINISHED')
        if not np.array_equal(book_result['minute_index'],self.minute_index):raise RuntimeError('RAW_EVENT134_BOOK_CLOCK')
        covered=np.asarray(book_result['book']['covered_ns'])
        if covered.dtype!=np.dtype('int64') or covered.shape!=(self.n,) or np.any((covered<0)|(covered>_EVENT134_MINUTE)):
            raise RuntimeError('RAW_EVENT134_BOOK_COVERAGE')
        values={};bounds={};max_mass=max_operations=0;nonexact_sums=0
        for name,op in EVENT134_REDUCERS.items():
            seq=self._values[name]
            v=np.asarray([np.nan if x is None else float(x) for x in seq],np.float64)
            e=np.where(np.isfinite(v),0.,np.nan)
            if op=='sum':
                for i,(total,mass,count) in enumerate(zip(seq,self._mass[name],self._operations[name])):
                    if name.endswith('_count') and abs(total)>1<<53:raise RuntimeError('RAW_EVENT134_COUNT_PRECISION')
                    e[i]=_event134_sum_radius(total,mass,count)
                    max_mass=max(max_mass,mass);max_operations=max(max_operations,count);nonexact_sums+=bool(e[i])
            if np.isinf(v).any() or np.isinf(e).any():raise RuntimeError('RAW_EVENT134_OUTPUT_OVERFLOW')
            values[name]=v;bounds[name]=e
        for prefix in ('receive_interval',)+_EVENT134_LATENCIES:
            values[prefix+'_mean_ns'],bounds[prefix+'_mean_ns']=_event134_quotient(
                values[prefix+'_sum_ns'],values[prefix+'_count'],bounds[prefix+'_sum_ns'])
        values['qualified_covered_ns']=covered.astype(np.float64);bounds['qualified_covered_ns']=np.zeros(self.n)
        # covered<=60e9 is an exact F64 integer. This explicit denominator wire
        # preserves source /1e9 then division, with exact event count<=2^53.
        seconds=covered.astype(np.float64)/1e9
        values['qualified_events_per_observed_second'],bounds['qualified_events_per_observed_second']=_event134_quotient(
            values['qualified_final_event_count'],seconds,bounds['qualified_final_event_count'])
        self.finished=True
        return {'minute_index':self.minute_index.copy(),'event':values,'bounds':bounds,'reference_report':{
            'source_rows':self.rows,'F_LAST_rows':self.final_rows,'clean_final_rows':self.clean_rows,
            'batches':self.batches,'per_file_reset':True,'previous_all_F_LAST_carried_across_batches':True,
            'integer_sufficient_statistics_exact':True,'integer_group_reductions_overflow_checked':True,
            'maximum_absolute_integer_mass':str(max_mass),'maximum_sequential_sum_operations':max_operations,
            'sum_coordinates_requiring_roundoff_radius':nonexact_sums,'source_sum_radius':'gamma_(n+1)*absolute_mass+reference_residual',
            'candidate_mask_applied':False,'production_helpers_called':False,'target_cache_used':False}}

# Integrator-owned input and comparison boundary; stdlib/Arrow/NumPy only.
RAW134_MINUTE = 60_000_000_000
RAW134_COLUMN_TYPES = {
    'ts_recv':'uint64','ts_event':'uint64','rtype':'uint8','publisher_id':'uint16',
    'instrument_id':'uint32','action':'string','side':'string','depth':'uint8',
    'price':'int64','size':'uint32','flags':'uint8','ts_in_delta':'int32','sequence':'uint32',
    'bid_px_00':'int64','ask_px_00':'int64','bid_sz_00':'uint32','ask_sz_00':'uint32',
    'bid_ct_00':'uint32','ask_ct_00':'uint32','symbol':'string'}
RAW134_IDENTITY = ('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')


def raw134_identity(path):
    path=Path(path)
    if not path.is_absolute() or path.resolve(strict=True)!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise RuntimeError('RAW134_PATH_IDENTITY')
    stat=path.stat()
    if not statmod.S_ISREG(stat.st_mode):raise RuntimeError('RAW134_NOT_REGULAR')
    return {k:getattr(stat,k) for k in RAW134_IDENTITY}


class _Raw134HashReader(io.RawIOBase):
    def __init__(self,raw):
        super().__init__();self.raw=raw;self.digest=hashlib.sha256();self.bytes_read=0
    def readable(self):return True
    def readinto(self,buffer):
        count=self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count]);self.bytes_read+=count
        return count


class _Raw134CSVReader:
    def __init__(self,compressed,limit):
        self.compressed=compressed;self.limit=limit;self.bytes_read=0;self.closed=False
    def readable(self):return True
    def _count(self,value):
        self.bytes_read+=len(value)
        if self.bytes_read>self.limit:raise RuntimeError('RAW134_DECOMPRESSED_BYTE_CAP')
        return value
    def read(self,size=-1):
        if size is None or size<0:size=self.limit-self.bytes_read+1
        return self._count(self.compressed.read(min(size,self.limit-self.bytes_read+1)))
    def readline(self,size):return self._count(self.compressed.readline(size))


class RawMBPBatchReader134:
    """Exactly one compressed read/hash and one twenty-column CSV parse.

    Only natural Arrow exhaustion plus actual gzip EOF/CRC, raw SHA, byte caps,
    unchanged file identity and at least one validated record admit a receipt.
    Closing a prefix iterator never produces a complete-file receipt.
    """
    def __init__(self,item,*,block_size=4*1024*1024):
        self.item=dict(item);self.block_size=block_size;self.used=False;self.receipt=None
        if type(block_size) is not int or not 1024<=block_size<=4*1024*1024:
            raise RuntimeError('RAW134_BLOCK_SIZE')
        for k in ('start_ns','end_ns','bytes','max_records','instrument_id','publisher_id'):
            if type(item.get(k)) is not int or item[k]<0:raise RuntimeError('RAW134_METADATA')
        if (item['end_ns']<=item['start_ns'] or item['end_ns']>=2**63
                or item['start_ns']%RAW134_MINUTE or item['end_ns']%RAW134_MINUTE
                or not item['bytes'] or not item['max_records']
                or not isinstance(item.get('raw_symbol'),str) or not item['raw_symbol']
                or not isinstance(item.get('sha256'),str) or len(item['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in item['sha256'])
                or item.get('compression')!='gzip' or item.get('pretty_ts') is not False
                or item.get('pretty_px') is not False):raise RuntimeError('RAW134_METADATA')
        self.path=Path(item['path']);self.before=raw134_identity(self.path)
        if self.before['st_size']!=item['bytes']:raise RuntimeError('RAW134_BOUND_FILE_SIZE')

    def __iter__(self):
        if self.used:raise RuntimeError('RAW134_READER_CONSUMED')
        self.used=True;rows=batches=0;previous=None
        with self.path.open('rb',buffering=0) as raw:
            opened={k:getattr(os.fstat(raw.fileno()),k) for k in RAW134_IDENTITY}
            if opened!=self.before:raise RuntimeError('RAW134_OPEN_IDENTITY')
            tee=_Raw134HashReader(raw)
            with io.BufferedReader(tee,buffer_size=1024*1024) as buffered:
                with gzip.GzipFile(fileobj=buffered,mode='rb') as compressed:
                    csv_stream=_Raw134CSVReader(compressed,self.item['max_records'])
                    header=csv_stream.readline(8193)
                    if len(header)>8192 or not header.endswith(b'\n'):raise RuntimeError('RAW134_CSV_HEADER')
                    try:names=next(csv.reader([header.decode('utf-8').rstrip('\r\n')],strict=True))
                    except (UnicodeError,csv.Error,StopIteration) as exc:raise RuntimeError('RAW134_CSV_HEADER') from exc
                    if len(names)!=20 or len(set(names))!=20 or set(names)!=set(RAW134_COLUMN_TYPES):
                        raise RuntimeError('RAW134_TWENTY_COLUMNS')
                    schema={k:pa.type_for_alias(v) for k,v in RAW134_COLUMN_TYPES.items()}
                    reader=pacsv.open_csv(csv_stream,
                        read_options=pacsv.ReadOptions(column_names=names,use_threads=False,block_size=self.block_size),
                        parse_options=pacsv.ParseOptions(newlines_in_values=False),
                        convert_options=pacsv.ConvertOptions(column_types=schema,include_columns=list(RAW134_COLUMN_TYPES)))
                    for batch in reader:
                        if any(batch.column(i).null_count for i in range(batch.num_columns)):
                            raise RuntimeError('RAW134_NULL_RECORD')
                        data={name:batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
                              for name in RAW134_COLUMN_TYPES}
                        recv=data['ts_recv'];n=len(recv)
                        if not n:continue
                        if (int(recv[0])<self.item['start_ns'] or int(recv[-1])>=self.item['end_ns']
                                or np.any(recv[1:]<recv[:-1]) or (previous is not None and int(recv[0])<previous)):
                            raise RuntimeError('RAW134_RECEIVE_ORDER_OR_WINDOW')
                        if (np.any(data['rtype']!=1) or np.any(data['publisher_id']!=self.item['publisher_id'])
                                or np.any(data['instrument_id']!=self.item['instrument_id'])
                                or np.any(data['symbol']!=self.item['raw_symbol'])
                                or not np.isin(data['action'],('A','C','M','R','T','F','N')).all()
                                or not np.isin(data['side'],('A','B','N')).all()):
                            raise RuntimeError('RAW134_RECORD_IDENTITY_OR_DOMAIN')
                        rows+=n;batches+=1
                        if rows>self.item['max_records']:raise RuntimeError('RAW134_RECORD_CAP')
                        previous=int(recv[-1])
                        for array in data.values():array.setflags(write=False)
                        yield data
                    if csv_stream.read(1)!=b'':raise RuntimeError('RAW134_INCOMPLETE_CSV_EOF')
                    if not rows:raise RuntimeError('RAW134_EMPTY_FILE')
                if tee.bytes_read!=self.item['bytes'] or tee.digest.hexdigest()!=self.item['sha256']:
                    raise RuntimeError('RAW134_COMPRESSED_SHA_OR_SIZE')
                if raw134_identity(self.path)!=self.before:raise RuntimeError('RAW134_FINAL_IDENTITY')
                self.receipt={'path':str(self.path),'sha256':tee.digest.hexdigest(),'bytes':tee.bytes_read,
                    'decompressed_csv_bytes':csv_stream.bytes_read,'raw_records':rows,'batches':batches,
                    'column_types':dict(RAW134_COLUMN_TYPES),'first_record_file_scope':self.item['start_ns'],
                    'end_exclusive_ns':self.item['end_ns'],'file_identity':self.before,
                    'actual_EOF_and_gzip_CRC_verified':True,'full_file_admitted':True,
                    'compressed_payload_passes':1,'CSV_parse_passes':1}


def extract_raw_gc_train_prefix134(item,destination,*,cutoff_ns,compression_level,max_line_bytes):
    """Copy a hash-bound receive-time prefix without parsing market fields.

    The caller binds the selected crossing source and its prior full-file time-order
    qualification. Only the header and ts_recv tokens are interpreted here.
    The derived gzip is a new local artifact; the original source never receives
    a full-EOF/CRC receipt from this prefix operation. Failed sibling stages
    remain for the caller's evidence inventory; this function never unlinks.
    """
    from gx1.contracts.immutable_event_authority_v1 import (
        _publish_file_noreplace, _fsync_directory,
    )
    declared=RawMBPBatchReader134(item)
    item=declared.item;source_path=declared.path;before=declared.before
    if (type(cutoff_ns) is not int or cutoff_ns%RAW134_MINUTE
            or not item['start_ns']<cutoff_ns<item['end_ns']):
        raise RuntimeError('RAW134_PREFIX_CUTOFF')
    if type(compression_level) is not int or not 0<=compression_level<=9:
        raise RuntimeError('RAW134_PREFIX_COMPRESSION_LEVEL')
    if type(max_line_bytes) is not int or max_line_bytes<1:
        raise RuntimeError('RAW134_PREFIX_LINE_LIMIT')
    destination=Path(destination);parent=destination.parent
    if (not destination.is_absolute() or parent.resolve(strict=True)!=parent
            or any(p.is_symlink() for p in (parent,*parent.parents)) or not parent.is_dir()):
        raise RuntimeError('RAW134_PREFIX_DESTINATION')
    stage=parent/('.'+destination.name+'.stage')
    if destination.exists() or destination.is_symlink():
        raise RuntimeError('RAW134_PREFIX_OUTPUT_EXISTS')
    if stage.exists() or stage.is_symlink():
        raise RuntimeError('RAW134_PREFIX_STAGE_EXISTS')

    def same_source(raw):
        opened={k:getattr(os.fstat(raw.fileno()),k) for k in RAW134_IDENTITY}
        if opened!=before or raw134_identity(source_path)!=before:
            raise RuntimeError('RAW134_PREFIX_SOURCE_IDENTITY')

    source_csv_bytes=rows=inspected=derived_csv_bytes=0
    csv_digest=hashlib.sha256();previous=None;first_outside=None
    buffer=bytearray(1024*1024)
    with source_path.open('rb',buffering=0) as raw:
        same_source(raw)
        opaque=_Raw134HashReader(raw)
        while opaque.readinto(buffer):
            pass
        if opaque.bytes_read!=item['bytes'] or opaque.digest.hexdigest()!=item['sha256']:
            raise RuntimeError('RAW134_PREFIX_COMPRESSED_SHA_OR_SIZE')
        same_source(raw)
        raw.seek(0)
        fd=os.open(stage,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as output:
            with gzip.GzipFile(fileobj=raw,mode='rb') as compressed:
                def bounded_line(limit):
                    nonlocal source_csv_bytes
                    value=compressed.readline(min(limit,item['max_records']-source_csv_bytes+1))
                    source_csv_bytes+=len(value)
                    if source_csv_bytes>item['max_records']:
                        raise RuntimeError('RAW134_PREFIX_DECOMPRESSED_BYTE_CAP')
                    return value

                with gzip.GzipFile(filename='',fileobj=output,mode='wb',
                        compresslevel=compression_level,mtime=0) as derived:
                    header=bounded_line(min(8193,max_line_bytes+1))
                    if len(header)>min(8192,max_line_bytes) or not header.endswith(b'\n'):
                        raise RuntimeError('RAW134_PREFIX_CSV_HEADER')
                    try:
                        names=next(csv.reader([header.decode('utf-8').rstrip('\r\n')],strict=True))
                    except (UnicodeError,csv.Error,StopIteration) as exc:
                        raise RuntimeError('RAW134_PREFIX_CSV_HEADER') from exc
                    if (len(names)!=20 or len(set(names))!=20 or set(names)!=set(RAW134_COLUMN_TYPES)
                            or names[0]!='ts_recv'):
                        raise RuntimeError('RAW134_PREFIX_HEADER_TWENTY_COLUMNS_RECV_FIRST')
                    derived.write(header);csv_digest.update(header);derived_csv_bytes+=len(header)
                    # A uint64 has at most twenty decimal digits. Interpret only
                    # that token before the cutoff gate; the bounded read may
                    # include opaque bytes after its separator or gzip read-ahead.
                    token_limit=len(str((1<<64)-1))+1
                    while True:
                        head=bounded_line(token_limit)
                        if not head:
                            raise RuntimeError('RAW134_PREFIX_BOUNDARY_NOT_FOUND')
                        token,separator,_=head.partition(b',')
                        if not separator or not token or not token.isdigit() or len(token)>=token_limit:
                            raise RuntimeError('RAW134_PREFIX_RECEIVE_TOKEN')
                        recv=int(token)
                        if recv>(1<<64)-1:
                            raise RuntimeError('RAW134_PREFIX_RECEIVE_TOKEN')
                        if (recv<item['start_ns'] or recv>=item['end_ns']
                                or (previous is not None and recv<previous)):
                            raise RuntimeError('RAW134_PREFIX_RECEIVE_ORDER_OR_WINDOW')
                        inspected+=1
                        if inspected>item['max_records']:
                            raise RuntimeError('RAW134_PREFIX_RECORD_CAP')
                        if recv>=cutoff_ns:
                            first_outside=recv
                            break
                        previous=recv
                        line=head if head.endswith(b'\n') else head+bounded_line(max_line_bytes-len(head)+1)
                        if len(line)>max_line_bytes or not line.endswith(b'\n'):
                            raise RuntimeError('RAW134_PREFIX_CSV_LINE')
                        derived.write(line);csv_digest.update(line)
                        derived_csv_bytes+=len(line);rows+=1
                    if not rows:
                        raise RuntimeError('RAW134_PREFIX_EMPTY_TRAIN')
                same_source(raw)
            output.flush();os.fsync(output.fileno())
        # Hash the completed new gzip, not its ancestor's bytes or identity.
        stage_identity=raw134_identity(stage)
        with stage.open('rb',buffering=0) as staged:
            opened={k:getattr(os.fstat(staged.fileno()),k) for k in RAW134_IDENTITY}
            if opened!=stage_identity:
                raise RuntimeError('RAW134_PREFIX_STAGE_IDENTITY')
            hashed=_Raw134HashReader(staged)
            while hashed.readinto(buffer):
                pass
            if hashed.bytes_read!=stage_identity['st_size'] or raw134_identity(stage)!=stage_identity:
                raise RuntimeError('RAW134_PREFIX_STAGE_IDENTITY')
        same_source(raw)
        _publish_file_noreplace(stage,destination)
        _fsync_directory(parent)

    ancestor={'path':str(source_path),'sha256':item['sha256'],'bytes':item['bytes'],
        'start_ns':item['start_ns'],'end_ns':item['end_ns'],'file_identity':before}
    provenance={'schema_version':'gx1_local_raw_gc_train_csv_prefix134_v1','source':ancestor,
        'cutoff_exclusive_ns':cutoff_ns,'copied_csv_bytes_sha256':csv_digest.hexdigest(),
        'original_header_and_train_line_bytes_preserved':True}
    derived_item={k:item[k] for k in ('start_ns','instrument_id','publisher_id','raw_symbol',
        'compression','pretty_ts','pretty_px')}
    derived_item.update(path=str(destination),end_ns=cutoff_ns,bytes=hashed.bytes_read,
        sha256=hashed.digest.hexdigest(),max_records=derived_csv_bytes,local_derivation=provenance)
    return {'derived_item':derived_item,'extraction_receipt':{
        'schema_version':'gx1_raw_gc_train_prefix_extraction134_v1','source':ancestor,
        'source_full_opaque_SHA_verified':True,'source_opaque_hash_bytes':opaque.bytes_read,
        'source_same_descriptor_hash_then_prefix':True,'source_identity_bookends':True,
        'source_full_EOF_and_gzip_CRC_verified':False,'source_full_file_admitted':False,
        'stop_reason':'FIRST_RECEIVE_AT_OR_AFTER_CUTOFF','first_outside_receive_ns':first_outside,
        'receive_order_scope':'TRAIN_prefix_and_first_outside_timestamp',
        'source_decompressed_bytes_returned':source_csv_bytes,
        'timestamps_inspected':inspected,'raw_records_copied':rows,
        'derived_decompressed_csv_bytes':derived_csv_bytes,'copied_csv_bytes_sha256':csv_digest.hexdigest(),
        'header_fields':names,'compression_level':compression_level,'max_line_bytes':max_line_bytes,
        'market_fields_parsed':0,'derived_reader_EOF_CRC_qualification_pending':True,
        'derived_item':derived_item}}


def compare_raw_values134(reference,target,*,bounds=None,label='M1'):
    """Exact integer/domain comparisons or precomputed F64 per-cell radii."""
    if set(reference)!=set(target) or (bounds is not None and set(bounds)!=set(reference)):
        raise RuntimeError('RAW134_COMPARE_FIELDS:'+label)
    report={}
    for name,expected in reference.items():
        expected=np.asarray(expected);actual=np.asarray(target[name])
        if expected.ndim!=1 or expected.shape!=actual.shape:raise RuntimeError('RAW134_COMPARE_SHAPE:'+name)
        if bounds is None:
            if expected.dtype.kind not in 'iu' or actual.dtype.kind not in 'iu':raise RuntimeError('RAW134_INTEGER_DOMAIN:'+name)
            left,right=expected,actual
            if expected.dtype.kind!=actual.dtype.kind:
                signed=expected if expected.dtype.kind=='i' else actual
                if np.any(signed<0):raise RuntimeError('RAW134_EXACT_MISMATCH:'+label+':'+name)
                left,right=expected.astype(np.uint64),actual.astype(np.uint64)
            if not np.array_equal(left,right):raise RuntimeError('RAW134_EXACT_MISMATCH:'+label+':'+name)
            report[name]={'cells':len(expected),'arithmetic':'EXACT_INTEGER'}
        else:
            radius=np.asarray(bounds[name])
            if radius.shape!=expected.shape or expected.dtype!=np.dtype('float64') or actual.dtype!=np.dtype('float64'):
                raise RuntimeError('RAW134_FLOAT_SCHEMA:'+name)
            finite=np.isfinite(expected)
            if (not np.array_equal(np.isnan(expected),np.isnan(actual)) or np.isinf(expected).any() or np.isinf(actual).any()
                    or not np.array_equal(np.isnan(radius),np.isnan(expected)) or np.any(radius[finite]<0.)
                    or not np.isfinite(radius[finite]).all()):raise RuntimeError('RAW134_FLOAT_DOMAIN:'+name)
            # Subtraction is rounded upward: it cannot hide a bound violation.
            delta=np.abs(actual[finite]-expected[finite]);upper=np.where(delta==0.,0.,np.nextafter(delta,np.inf))
            if np.any(upper>radius[finite]):raise RuntimeError('RAW134_BOUND_MISMATCH:'+label+':'+name)
            report[name]={'cells':len(expected),'finite':int(np.count_nonzero(finite)),
                'NaN':int(np.count_nonzero(~finite)),'arithmetic':'F64_PRECOMPUTED_RADIUS',
                'max_abs_error':float(delta.max(initial=0.)),'max_radius':float(radius[finite].max(initial=0.))}
    return report


def reduce_raw_file134(item,meta,*,on_ladder=None,block_size=4*1024*1024):
    """Compute all independent ledgers from the same immutable decoded batch.

    No production code, target reading, fit or model call. Caller admission and
    final source/target guards remain mandatory before qualification/publishing.
    """
    if any(item[k]!=meta[k] for k in ('start_ns','end_ns','raw_symbol','instrument_id','publisher_id')):
        raise RuntimeError('RAW134_COORDINATOR_METADATA')
    reader=RawMBPBatchReader134(item,block_size=block_size)
    trade=RawTradeQualityReference134(meta);book=RawBookReference134(meta);event=RawEventReference134(meta)
    for data in reader:
        context=trade.consume(data);book.consume(data,context);event.consume(data,context)
        if on_ladder is not None:on_ladder(context['ladder'])
    if reader.receipt is None:raise RuntimeError('RAW134_COORDINATOR_INCOMPLETE_EOF')
    t=trade.finish();b=book.finish();e=event.finish(b)
    if not np.array_equal(t['minute_index'],b['minute_index']) or not np.array_equal(t['minute_index'],e['minute_index']):
        raise RuntimeError('RAW134_COORDINATOR_M1_CLOCK')
    return {'raw_receipt':reader.receipt,'trade':t,'book':b,'event':e}


RAW134_LADDER_FIELDS=('minute_index','price_tick','buy','sell','unknown')


def canonical_ladder134(mapping):
    if set(mapping)!=set(RAW134_LADDER_FIELDS):raise RuntimeError('RAW134_LADDER_FIELDS')
    n=len(mapping['minute_index']);data={}
    for name in RAW134_LADDER_FIELDS:
        v=np.asarray(mapping[name])
        if v.shape!=(n,) or v.dtype.kind not in 'iu' or np.any(v<0):raise RuntimeError('RAW134_LADDER_DOMAIN')
        if name in ('minute_index','price_tick'):
            if np.any(v>np.iinfo(np.int64).max):raise RuntimeError('RAW134_LADDER_DOMAIN')
            data[name]=v.astype(np.int64)
        else:data[name]=v.astype(np.uint64)
    if not n:return data
    order=np.lexsort((data['price_tick'],data['minute_index']));data={k:v[order] for k,v in data.items()}
    starts=np.r_[0,np.flatnonzero((data['minute_index'][1:]!=data['minute_index'][:-1])|
        (data['price_tick'][1:]!=data['price_tick'][:-1]))+1].astype(np.int64)
    out={k:data[k][starts] for k in RAW134_LADDER_FIELDS[:2]}
    for k in RAW134_LADDER_FIELDS[2:]:out[k]=_source_trade_group_sums134(data[k],starts,'target_ladder.'+k)
    return out


class RawLadderComparator134:
    """Only the unfinished final raw minute survives between decoded batches.

    Caller supplies one verified target file-window slice. Full EOF validation
    must finish before finish() can contribute to an admitted per-file receipt.
    """
    def __init__(self,target):
        self.target=canonical_ladder134(target);self.pending=None;self.cursor=0;self.finished=False;self.last_minute=None
    def _compare(self,data):
        n=len(data['minute_index'])
        if not n:return
        actual={k:v[self.cursor:self.cursor+n] for k,v in self.target.items()}
        compare_raw_values134(data,actual,label='UNFILTERED_PRICE_SIDE_LADDER')
        self.cursor+=n
    def consume(self,data):
        if self.finished:raise RuntimeError('RAW134_LADDER_FINISHED')
        if not len(data['minute_index']):return
        batch=canonical_ladder134(data)
        if self.last_minute is not None and int(batch['minute_index'][0])<self.last_minute:
            raise RuntimeError('RAW134_LADDER_BACKWARD_MINUTE')
        self.last_minute=int(batch['minute_index'][-1])
        if self.pending is not None:batch=canonical_ladder134({k:np.r_[self.pending[k],batch[k]] for k in RAW134_LADDER_FIELDS})
        complete=batch['minute_index']<batch['minute_index'][-1]
        self._compare({k:v[complete] for k,v in batch.items()})
        self.pending={k:v[~complete] for k,v in batch.items()}
    def finish(self):
        if self.finished:raise RuntimeError('RAW134_LADDER_FINISHED')
        self.finished=True
        if self.pending is not None:self._compare(self.pending)
        if self.cursor!=len(self.target['minute_index']):raise RuntimeError('RAW134_LADDER_MISSING_ROWS')
        return {'qualified_price_side_levels':self.cursor,'exact_fields':list(RAW134_LADDER_FIELDS),
            'T_duplicates_retained':True,'raw_record_retention':'only pending last-minute grouped levels'}


def compare_reduced_file134(reduced,m1,event,trade_size,selected_indices):
    """Compare a declared file window; independent roll proof remains separate."""
    t,b,e=reduced['trade'],reduced['book'],reduced['event'];index=t['minute_index']
    if not np.array_equal(index,b['minute_index']) or not np.array_equal(index,e['minute_index']):
        raise RuntimeError('RAW134_COMPARE_REFERENCE_CLOCK')
    wanted=np.asarray(selected_indices)
    if wanted.dtype!=np.dtype('int64') or wanted.ndim!=1 or np.any(wanted[1:]<=wanted[:-1]):
        raise RuntimeError('RAW134_COMPARE_SELECTED_CLOCK')
    local=wanted-(int(index[0]) if len(index) else 0)
    if np.any((local<0)|(local>=len(index))):raise RuntimeError('RAW134_COMPARE_SELECTED_CLOCK')
    integers={**t['source_metrics11'],**{'quality_'+k:v for k,v in t['quality11'].items()},**t['ohlc']}
    integers.update({k:v for k,v in b['book'].items() if v.dtype.kind in 'iu'})
    floats={k:v for k,v in b['book'].items() if v.dtype.kind=='f'}
    if set(m1)!=set(integers)|set(floats):raise RuntimeError('RAW134_COMPARE_M1_FIELDS')
    if set(event)!=set(e['event'])|{'minute_index'} or set(trade_size)!=set(t['trade_size7'])|{'minute_index'}:
        raise RuntimeError('RAW134_COMPARE_SELECTED_FIELDS')
    clock=compare_raw_values134({'event':wanted,'trade_size':wanted},
        {'event':event['minute_index'],'trade_size':trade_size['minute_index']},label='SELECTED_CLOCK')
    report={'unfiltered_integer':compare_raw_values134(integers,{k:m1[k] for k in integers},label='001_RAW_M1'),
        'unfiltered_float':compare_raw_values134(floats,{k:m1[k] for k in floats},
            bounds={k:b['bounds'][k] for k in floats},label='001_RAW_BOOK'),
        'selected_clock':clock,
        'selected_event':compare_raw_values134({k:v[local] for k,v in e['event'].items()},
            {k:event[k] for k in e['event']},bounds={k:v[local] for k,v in e['bounds'].items()},label='024_RAW_EVENT'),
        'selected_trade_size':compare_raw_values134({k:v[local] for k,v in t['trade_size7'].items()},
            {k:trade_size[k] for k in t['trade_size7']},label='024_RAW_TRADE_SIZE')}
    return report


# Independent completed-UTC-day selection and restored024 observation masks.
_MASK134_MINUTE = 60_000_000_000
_MASK134_DAY = 1440*_MASK134_MINUTE
_MASK134_ORIGIN = 1714521600000000000  # 2024-05-01T00:00:00Z
_MASK134_END = 1782864000000000000     # 2026-07-01T00:00:00Z


def reference_roll134(volumes,maturities):
    """Unique prior-day volume leader among expiries not earlier than current."""
    v=np.asarray(volumes)
    if v.ndim!=2 or v.dtype.kind not in 'iu' or not all(v.shape) or np.any(v<0):
        raise RuntimeError('RAW_MASK134_DAILY_VOLUME_DOMAIN')
    symbols,days=v.shape
    if symbols>32767 or len(maturities)!=symbols:
        raise RuntimeError('RAW_MASK134_MATURITY_IDENTITY')
    identities=[]
    for maturity in maturities:
        if (not isinstance(maturity,(tuple,list)) or len(maturity)!=2 or
                any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) for x in maturity)):
            raise RuntimeError('RAW_MASK134_MATURITY_IDENTITY')
        identities.append(tuple(map(int,maturity)))
    if len(set(identities))!=symbols:raise RuntimeError('RAW_MASK134_MATURITY_IDENTITY')
    selected=np.full(days,-1,np.int16);basis=np.full(days,-1,np.int32)
    identity=None;last_basis=-1
    for day in range(1,days):
        history=day-1
        eligible=[i for i in range(symbols) if int(v[i,history])>0 and
                  (identity is None or identities[i]>=identities[identity])]
        leaders=sorted(((int(v[i,history]),i) for i in eligible),reverse=True)
        if leaders and (len(leaders)==1 or leaders[0][0]>leaders[1][0]):
            identity=leaders[0][1];last_basis=history
        selected[day]=-1 if identity is None else identity;basis[day]=last_basis
    return selected,basis


def reference_roll_minutes134(minutes,selected_daily):
    """The new daily identity starts at00:01UTC, not its00:00label."""
    t=np.asarray(minutes);daily=np.asarray(selected_daily)
    if (t.ndim!=1 or t.dtype!=np.dtype('int64') or not len(t) or np.any(t%_MASK134_MINUTE)
            or np.any(t<_MASK134_ORIGIN) or np.any(t>=_MASK134_END) or np.any(t[1:]<=t[:-1])
            or daily.ndim!=1 or daily.dtype.kind not in 'iu' or not len(daily)
            or np.any(daily < -1) or np.any(daily>32767)):
        raise RuntimeError('RAW_MASK134_ROLL_MINUTE_DOMAIN')
    # Search on explicit daily availability stamps keeps the one-minute
    # availability offset separate from the prior-day selection algorithm.
    availability=_MASK134_ORIGIN+np.arange(len(daily),dtype=np.int64)*_MASK134_DAY+_MASK134_MINUTE
    pos=np.searchsorted(availability,t,side='right')-1
    if np.any(t>=_MASK134_ORIGIN+len(daily)*_MASK134_DAY):raise RuntimeError('RAW_MASK134_ROLL_DAILY_COVERAGE')
    out=np.full(len(t),-1,np.int16);defined=pos>=0;out[defined]=daily[pos[defined]]
    return out


def _mask134_stamp(text):
    from datetime import datetime,timezone
    if not isinstance(text,str):raise RuntimeError('RAW_MASK134_INCIDENT_INTERVAL')
    stamp=datetime.fromisoformat(text.replace('Z','+00:00'))
    if stamp.tzinfo is None:raise RuntimeError('RAW_MASK134_INCIDENT_INTERVAL')
    delta=stamp-datetime(1970,1,1,tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*1_000_000_000+delta.microseconds*1000


def reference_masks134(minutes,selected,metrics,quality,covered,ordercovered,manifest):
    """Final restored024 masks; duplicate T signatures remain diagnostic only."""
    t=np.asarray(minutes);s=np.asarray(selected)
    if (t.dtype!=np.dtype('int64') or t.ndim!=1 or not len(t) or np.any(t%_MASK134_MINUTE)
            or np.any(np.diff(t)!=_MASK134_MINUTE) or np.any(t<_MASK134_ORIGIN) or np.any(t>=_MASK134_END)
            or s.shape!=t.shape or s.dtype.kind not in 'iu' or np.any(s < -1)):
        raise RuntimeError('RAW_MASK134_M1_CLOCK')
    admitted=manifest.get('trusted_trade_admission',{})
    if (manifest.get('schema_version')!='gx1_gc_vendor_source_incidents_v1' or manifest.get('dataset')!='GLBX.MDP3'
            or manifest.get('nontrade_action_semantics_admitted') is not False or not manifest.get('incidents')
            or admitted.get('reject_duplicate_signature') is not False or admitted.get('deduplicate_T') is not False):
        raise RuntimeError('RAW_MASK134_RESTORED_INCIDENT_SCOPE')
    required_metrics=('trades','invalid_trade','gap','bad_recv_trade','snapshot_trade','unknown')
    required_quality=('records','clean_complete_bbo','bad_bbo','gap','bad_recv','snapshot','reset','file_seam')
    arrays={}
    for prefix,mapping,names in (('m',metrics,required_metrics),('q',quality,required_quality)):
        for name in names:
            x=np.asarray(mapping[name])
            if x.shape!=t.shape or x.dtype.kind not in 'iu' or np.any(x<0):
                raise RuntimeError('RAW_MASK134_COUNTER_DOMAIN:'+prefix+name)
            arrays[prefix+name]=x
    cover,order=np.asarray(covered),np.asarray(ordercovered)
    if any(x.shape!=t.shape or x.dtype.kind not in 'iu' or np.any(x<0) or np.any(x>_MASK134_MINUTE) for x in (cover,order)):
        raise RuntimeError('RAW_MASK134_COVERAGE_DOMAIN')
    volume=(s>=0)&(t+_MASK134_MINUTE<_MASK134_END)&(arrays['mtrades']>0)
    for name in ('invalid_trade','gap','bad_recv_trade','snapshot_trade'):volume &= arrays['m'+name]==0
    clean=(arrays['qrecords']>0)&(arrays['qclean_complete_bbo']>0)
    for name in ('bad_bbo','gap','bad_recv','snapshot','reset','file_seam'):clean &= arrays['q'+name]==0
    preceding=np.zeros(len(t),bool)
    preceding[1:]=clean[:-1]&(s[:-1]==s[1:])&(s[1:]>=0)
    ofi=(s>=0)&clean&preceding
    clock_excluded=np.zeros(len(t),bool);state_excluded=np.zeros(len(t),bool)
    anchors=np.flatnonzero(clean&(s>=0))
    for incident in manifest['incidents']:
        left=_mask134_stamp(incident['start_inclusive_utc']);right=_mask134_stamp(incident['end_exclusive_utc'])
        if left>=right or not incident.get('incident_id') or not incident.get('source_url'):
            raise RuntimeError('RAW_MASK134_INCIDENT_INTERVAL')
        # Half-open intersection: minute_end>incident_start andminute_start<incident_end.
        first=int(np.searchsorted(t+_MASK134_MINUTE,left,side='right'))
        stop=int(np.searchsorted(t,right,side='left'))
        if first>=stop:continue
        clock_excluded[first:stop]=True
        at=int(np.searchsorted(anchors,stop,side='left'))
        recovery=int(anchors[at])+1 if at<len(anchors) else len(t)
        state_excluded[first:recovery]=True
    volume &= ~clock_excluded;ofi &= ~state_excluded
    result={'volume_valid':volume,'signed_valid':volume&(arrays['munknown']==0),
        'ofi_valid':ofi,'bbo_valid':ofi&(cover==_MASK134_MINUTE),
        'order_count_valid':ofi&(order==_MASK134_MINUTE),
        'receive_clock_excluded':clock_excluded,'book_event_state_excluded':state_excluded}
    for x in result.values():x.setflags(write=False)
    return result
