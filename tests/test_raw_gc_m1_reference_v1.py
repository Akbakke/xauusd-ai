"""Literal independent raw references and bounded synthetic gzip/CSV fixtures."""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import numpy as np
import pytest
from gx1.scripts import audit_raw_gc_m1_reference_v1 as a

class TestRawTradeQualityReference134:
    START=1714521600000000000
    MINUTE=60000000000
    TICK=100000000

    @classmethod
    def _meta(cls,*,start=None,minutes=3):
        start=cls.START if start is None else start
        return dict(start_ns=start,end_ns=start+minutes*cls.MINUTE,study_origin_ns=cls.START,
            tick_size=cls.TICK,raw_symbol='GCM4',instrument_id=42,publisher_id=7)

    @classmethod
    def _row(cls,index=0,**patch):
        recv=patch.get('ts_recv',cls.START+index+1)
        result=dict(ts_recv=recv,ts_event=recv-10,rtype=1,publisher_id=7,instrument_id=42,
            action='T',side='B',depth=0,price=1000*cls.TICK,size=1,flags=128,ts_in_delta=10,
            sequence=index+1,bid_px_00=999*cls.TICK,ask_px_00=1001*cls.TICK,
            bid_sz_00=10,ask_sz_00=20,bid_ct_00=2,ask_ct_00=3,symbol='GCM4')
        result.update(patch);return result

    @staticmethod
    def _data(rows):
        schema={'ts_recv':'uint64','ts_event':'uint64','rtype':'uint8','publisher_id':'uint16',
            'instrument_id':'uint32','action':'object','side':'object','depth':'uint8','price':'int64',
            'size':'uint32','flags':'uint8','ts_in_delta':'int32','sequence':'uint32','bid_px_00':'int64',
            'ask_px_00':'int64','bid_sz_00':'uint32','ask_sz_00':'uint32','bid_ct_00':'uint32',
            'ask_ct_00':'uint32','symbol':'object'}
        data={name:np.asarray([row[name] for row in rows],dtype=dtype) for name,dtype in schema.items()}
        for value in data.values():value.setflags(write=False)
        return data

    @classmethod
    def _literal_rows(cls):
        first=cls._row(size=3)
        return [first,dict(first),dict(first,bid_ct_00=5),
            cls._row(3,ts_recv=cls.START+2,side='A',price=1001*cls.TICK,size=2),
            cls._row(4,ts_recv=cls.START+3,side='N',price=999*cls.TICK,size=5),
            cls._row(5,ts_recv=cls.START+4,size=0),cls._row(6,ts_recv=cls.START+5,size=(1<<32)-1),
            cls._row(7,ts_recv=cls.START+6,price=(1<<63)-1),
            cls._row(8,ts_recv=cls.START+7,price=-1),cls._row(9,ts_recv=cls.START+8,price=1000*cls.TICK+1),
            cls._row(10,ts_recv=cls.START+9,size=7,flags=136),
            cls._row(11,ts_recv=cls.START+10,side='A',size=9,flags=160),
            cls._row(12,ts_recv=cls.START+11,size=4,flags=132),
            cls._row(13,ts_recv=cls.START+12,action='R',side='N',price=0,size=0),
            cls._row(14,ts_recv=cls.START+13,action='M',side='N',price=0,size=0,ask_px_00=999*cls.TICK),
            cls._row(15,ts_recv=cls.START+14,action='A',side='N',price=0,size=0,flags=0),
            cls._row(16,ts_recv=cls.START+cls.MINUTE+1,side='A',price=1002*cls.TICK,size=2)]

    @staticmethod
    def _ladder(contexts):
        result={}
        for context in contexts:
            table=context['ladder'];keys=list(zip(table['minute_index'],table['price_tick']))
            assert keys==sorted(set(keys))
            for row,key in enumerate(keys):
                sums=result.setdefault(tuple(map(int,key)),[0,0,0])
                for j,name in enumerate(('buy','sell','unknown')):sums[j]+=int(table[name][row])
        return result

    def test_literal_all_eleven_source_and_quality_counters_OHLC_sizes_and_ladder(self):
        data=self._data(self._literal_rows());reference=a.RawTradeQualityReference134(self._meta())
        context=reference.consume(data);result=reference.finish()
        expected={'records':15,'trades':6,'volume':20,'buy':13,'sell':2,'unknown':5,
            'invalid_trade':5,'bad_recv_trade':1,'gap':1,'snapshot_trade':1,'duplicate_trade':2}
        second={'records':1,'trades':1,'volume':2,'sell':2}
        assert set(result['source_metrics11'])==set(expected)
        for name,count in expected.items():
            np.testing.assert_array_equal(result['source_metrics11'][name],[count,second.get(name,0),0])
        quality={'records':16,'clean_complete_bbo':10,'bad_bbo':1,'gap':1,'bad_recv':1,'snapshot':1,
            'reset':1,'duplicate_trade_signature':2,'full_record_duplicate_trade':1,'signature_collision':1,'file_seam':1}
        assert set(result['quality11'])==set(quality)
        for name,count in quality.items():
            np.testing.assert_array_equal(result['quality11'][name],[count,1 if name in ('records','clean_complete_bbo') else 0,0])
        for name,expected_ticks in {'open_tick':[1000,1002,0],'high_tick':[1001,1002,0],
                                  'low_tick':[999,1002,0],'close_tick':[1000,1002,0]}.items():
            np.testing.assert_array_equal(result['ohlc'][name],expected_ticks)
        sizes={'size_sum_contracts':[20,2,0],'size_sum_squared_contracts2':[72,4,0],
            'size_min_contracts':[2,2,0],'size_max_contracts':[5,2,0],
            'buy_trade_records':[4,0,0],'sell_trade_records':[1,1,0],'unknown_trade_records':[1,0,0]}
        assert set(result['trade_size7'])==set(sizes)
        for name,value in sizes.items():np.testing.assert_array_equal(result['trade_size7'][name],value)
        np.testing.assert_array_equal(np.flatnonzero(context['trusted_trade']),[0,1,2,3,4,12,16])
        np.testing.assert_array_equal(np.flatnonzero(context['duplicate_trade_signature']),[1,2])
        np.testing.assert_array_equal(np.flatnonzero(context['full_record_duplicate_trade']),[1])
        assert self._ladder([context])=={(0,999):[0,0,5],(0,1000):[13,0,0],(0,1001):[0,2,0],(1,1002):[0,2,0]}
        assert result['reference_report']['trusted_T_records']==7

    def test_batch_partition_equivalence_retains_T_duplicates_and_full_previous_record(self):
        data=self._data(self._literal_rows());snapshots={name:value.copy() for name,value in data.items()}
        whole=a.RawTradeQualityReference134(self._meta());one_context=whole.consume(data);expected=whole.finish()
        for sizes in ((1,1,2,3,10),(1,)*17,(6,11)):
            owner=a.RawTradeQualityReference134(self._meta());contexts=[];start=0
            owner.consume(self._data([]))
            for size in sizes:
                contexts.append(owner.consume({name:value[start:start+size] for name,value in data.items()}));start+=size
            actual=owner.finish();assert start==17
            for key in ('source_metrics11','quality11','ohlc','trade_size7'):
                for name,value in expected[key].items():np.testing.assert_array_equal(actual[key][name],value)
            for name in ('minute_index','local_index','trusted_trade','duplicate_trade_signature','full_record_duplicate_trade'):
                np.testing.assert_array_equal(np.concatenate([context[name] for context in contexts]),one_context[name])
            assert self._ladder(contexts)==self._ladder([one_context])
            assert actual['reference_report']['raw_rows']==17
        for name,value in data.items():
            np.testing.assert_array_equal(value,snapshots[name]);assert not value.flags.writeable
        for name,value in one_context.items():
            arrays=value.values() if isinstance(value,dict) else (value,)
            assert all(not array.flags.writeable for array in arrays)

    @pytest.mark.parametrize('field',('rtype','publisher_id','instrument_id','depth','ts_in_delta',
        'bid_px_00','ask_px_00','bid_sz_00','ask_sz_00','bid_ct_00','ask_ct_00','symbol'))
    def test_signature_suspect_checks_each_remaining_field_across_batch_seam(self,field):
        # Root separately rejects identity violations; this tests all20-field equality itself.
        first=self._row();second=dict(first)
        second[field]=(first[field]+'X' if field=='symbol' else first[field]+(self.TICK if field.endswith('px_00') else 1))
        owner=a.RawTradeQualityReference134(self._meta())
        owner.consume(self._data([first]));context=owner.consume(self._data([second]));result=owner.finish()
        assert context['duplicate_trade_signature'].tolist()==[True]
        assert context['full_record_duplicate_trade'].tolist()==[False]
        assert result['quality11']['signature_collision'].tolist()==[1,0,0]
        assert result['source_metrics11']['trades'].tolist()==[2,0,0]

    @pytest.mark.parametrize('field,value',(('ts_recv',1714521600000000002),('ts_event',1714521600000000000),
        ('sequence',2),('action','M'),('side','A'),('price',100100000000),('size',2),('flags',132)))
    def test_each_signature_field_difference_prevents_duplicate(self,field,value):
        first=self._row();second=dict(first);second[field]=value
        owner=a.RawTradeQualityReference134(self._meta());owner.consume(self._data([first]))
        context=owner.consume(self._data([second]));result=owner.finish()
        assert not context['duplicate_trade_signature'][0] and not context['full_record_duplicate_trade'][0]
        assert not result['quality11']['signature_collision'].any()

    @pytest.mark.parametrize('patch',({'price':0},{'price':-100000000},{'price':(1<<63)-1},
        {'price':100000000001},{'size':0},{'size':(1<<32)-1}))
    def test_zero_UNDEF_negative_offtick_trades_are_counted_but_not_trusted(self,patch):
        owner=a.RawTradeQualityReference134(self._meta());context=owner.consume(self._data([self._row(**patch)]));result=owner.finish()
        assert context['trusted_trade'].tolist()==[False] and len(context['ladder']['price_tick'])==0
        assert result['source_metrics11']['invalid_trade'].tolist()==[1,0,0]
        assert result['source_metrics11']['trades'].tolist()==[0,0,0]
        assert result['quality11']['clean_complete_bbo'].tolist()==[1,0,0]
        assert all(not value.any() for value in result['ohlc'].values())
        assert all(not value.any() for value in result['trade_size7'].values())

    @pytest.mark.parametrize('patch',({'bid_px_00':0},{'ask_px_00':0},{'bid_px_00':(1<<63)-1},
        {'ask_px_00':(1<<63)-1},{'bid_px_00':99900000001},{'ask_px_00':99900000000},
        {'bid_sz_00':0},{'ask_sz_00':0},{'bid_sz_00':(1<<32)-1},{'ask_sz_00':(1<<32)-1}))
    def test_bad_BBO_quality_does_not_silently_reclassify_genuine_T(self,patch):
        owner=a.RawTradeQualityReference134(self._meta());context=owner.consume(self._data([self._row(**patch)]));result=owner.finish()
        assert context['trusted_trade'].tolist()==[True]
        assert result['quality11']['bad_bbo'].tolist()==[1,0,0]
        assert not result['quality11']['clean_complete_bbo'].any()
        assert result['source_metrics11']['trades'].tolist()==[1,0,0]

    @pytest.mark.parametrize('flags,trusted',((136,False),(160,False),(132,True),(172,False)))
    def test_flag_domains_overlap_without_dropping_gap_T_quantity(self,flags,trusted):
        owner=a.RawTradeQualityReference134(self._meta());context=owner.consume(self._data([self._row(flags=flags,size=7)]));result=owner.finish()
        assert bool(context['trusted_trade'][0]) is trusted
        assert int(result['source_metrics11']['volume'][0])==(7 if trusted else 0)
        assert int(result['source_metrics11']['bad_recv_trade'][0])==int(bool(flags&8))
        assert int(result['source_metrics11']['snapshot_trade'][0])==int(bool(flags&32))
        assert int(result['source_metrics11']['gap'][0])==int(bool(flags&4))

    def test_file_reset_marks_new_seam_and_does_not_deduplicate_prior_file(self):
        data=self._data([self._row()])
        first=a.RawTradeQualityReference134(self._meta());first.consume(data);first.finish()
        second=a.RawTradeQualityReference134(self._meta());context=second.consume(data);result=second.finish()
        assert not context['duplicate_trade_signature'].any()
        assert result['quality11']['file_seam'].tolist()==[1,0,0]

    def test_local_file_axis_keeps_empty_minutes_and_absolute_study_indices(self):
        start=self.START+2*self.MINUTE;owner=a.RawTradeQualityReference134(self._meta(start=start,minutes=2))
        context=owner.consume(self._data([self._row(ts_recv=start+1,side='N',size=5)]));result=owner.finish()
        np.testing.assert_array_equal(result['minute_index'],[2,3])
        np.testing.assert_array_equal(context['local_index'],[0]);np.testing.assert_array_equal(context['minute_index'],[2])
        np.testing.assert_array_equal(result['source_metrics11']['unknown'],[5,0])
        assert self._ladder([context])=={(2,1000):[0,0,5]}

    @pytest.mark.parametrize('split',(False,True))
    def test_trusted_uint32_size_square_overflow_is_rejected_without_wrap(self,split):
        rows=[self._row(0,size=(1<<32)-2),self._row(1,size=(1<<32)-2)]
        owner=a.RawTradeQualityReference134(self._meta())
        with pytest.raises(RuntimeError,match='RAW_TRADE134_UINT64_OVERFLOW:size_sum_squared_contracts2'):
            if split:owner.consume(self._data(rows[:1]));owner.consume(self._data(rows[1:]))
            else:owner.consume(self._data(rows))

    def test_checked_group_and_persistent_uint64_additions_reject_overflow(self):
        maximum=(1<<64)-1
        with pytest.raises(RuntimeError,match='RAW_TRADE134_UINT64_OVERFLOW:hand'):
            a._source_trade_group_sums134(np.asarray([maximum,1],np.uint64),np.asarray([0],np.int64),'hand')
        np.testing.assert_array_equal(a._source_trade_group_sums134(np.asarray([maximum,1],np.uint64),
            np.asarray([0,1],np.int64),'hand'),np.asarray([maximum,1],np.uint64))
        owner=a.RawTradeQualityReference134(self._meta());owner.metrics['records'][0]=maximum
        with pytest.raises(RuntimeError,match='RAW_TRADE134_UINT64_OVERFLOW:records'):
            owner.consume(self._data([self._row()]))

    def test_empty_file_and_post_finish_replay_fail_closed(self):
        owner=a.RawTradeQualityReference134(self._meta())
        with pytest.raises(RuntimeError,match='RAW_TRADE134_EMPTY_FILE'):owner.finish()
        data=self._data([self._row()]);owner.consume(data);owner.finish()
        with pytest.raises(RuntimeError,match='RAW_TRADE134_ALREADY_FINISHED'):owner.consume(data)
        with pytest.raises(RuntimeError,match='RAW_TRADE134_ALREADY_FINISHED'):owner.finish()

    def test_raw_unsigned_undefined_event_clock_is_not_float_coerced_or_T_deduplicated(self):
        first=self._row(ts_event=(1<<64)-1,ts_in_delta=-(1<<31))
        owner=a.RawTradeQualityReference134(self._meta())
        owner.consume(self._data([first]));context=owner.consume(self._data([dict(first)]))
        assert owner.previous['ts_event']==(1<<64)-1
        assert owner.previous['ts_recv']==self.START+1 and owner.previous['ts_in_delta']==-(1<<31)
        result=owner.finish()
        assert context['duplicate_trade_signature'].tolist()==[True]
        assert context['full_record_duplicate_trade'].tolist()==[True]
        assert result['source_metrics11']['trades'].tolist()==[2,0,0]

    def test_signature_predecessor_is_previous_raw_record_not_previous_T(self):
        first=self._row();middle=dict(first,action='N',side='N')
        owner=a.RawTradeQualityReference134(self._meta())
        owner.consume(self._data([first,middle]));context=owner.consume(self._data([dict(first)]));result=owner.finish()
        assert not context['duplicate_trade_signature'].any()
        assert result['source_metrics11']['trades'].tolist()==[2,0,0]
        assert not result['quality11']['duplicate_trade_signature'].any()

# Integration fragment; the capped test namespace supplies a, np and pytest.
class TestRawBookReference134:
    @staticmethod
    def _meta(minutes=1,start=0,tick=100_000_000):
        return {'start_ns':start,'end_ns':start+minutes*60_000_000_000,'study_origin_ns':0,
            'tick_size':tick,'raw_symbol':'GCM4','instrument_id':669,'publisher_id':1}

    @staticmethod
    def _data(rows,tick=100_000_000,flags=None,action=None):
        n=len(rows);x=np.asarray(rows,np.int64);clock=x[:,0].astype(np.uint64)
        data={'ts_recv':clock,'ts_event':clock.copy(),'rtype':np.ones(n,np.uint8),
            'publisher_id':np.ones(n,np.uint16),'instrument_id':np.full(n,669,np.uint32),
            'action':np.full(n,'T') if action is None else np.asarray(action),
            'side':np.full(n,'B'),'depth':np.zeros(n,np.uint8),'price':np.full(n,100*tick,np.int64),
            'size':np.ones(n,np.uint32),'flags':np.full(n,128,np.uint8) if flags is None else np.asarray(flags,np.uint8),
            'ts_in_delta':np.zeros(n,np.int32),'sequence':np.arange(n,dtype=np.uint32),
            'bid_px_00':x[:,1]*tick,'ask_px_00':x[:,2]*tick,
            'bid_sz_00':x[:,3].astype(np.uint32),'ask_sz_00':x[:,4].astype(np.uint32),
            'bid_ct_00':x[:,5].astype(np.uint32),'ask_ct_00':x[:,6].astype(np.uint32),'symbol':np.full(n,'GCM4')}
        for v in data.values():v.setflags(write=False)
        return data

    @staticmethod
    def _context(data,meta):
        local=(data['ts_recv'].astype(np.int64)-meta['start_ns'])//60_000_000_000
        minute=local+(meta['start_ns']-meta['study_origin_ns'])//60_000_000_000
        context={'local_index':local,'minute_index':minute,'trusted_trade':np.zeros(len(local),bool)}
        for v in context.values():v.setflags(write=False)
        return context

    @classmethod
    def _run(cls,rows,*,minutes=1,start=0,tick=100_000_000,cuts=(),flags=None,action=None):
        meta=cls._meta(minutes,start,tick);data=cls._data(rows,tick,flags,action)
        saved={k:v.copy() for k,v in data.items()};owner=a.RawBookReference134(meta)
        edges=(0,*cuts,len(rows))
        for left,right in zip(edges[:-1],edges[1:]):
            batch={k:v[left:right] for k,v in data.items()};owner.consume(batch,cls._context(batch,meta))
        result=owner.finish()
        for k,v in data.items():
            np.testing.assert_array_equal(v,saved[k]);assert not v.flags.writeable
        return result

    @staticmethod
    def _rational(result,name,numerator,denominator=1,row=0):
        from fractions import Fraction
        value=float(result['book'][name][row]);bound=float(result['bounds'][name][row])
        assert np.isfinite(value) and np.isfinite(bound) and bound>=0.
        assert abs(Fraction.from_float(value)-Fraction(numerator,denominator))<=Fraction.from_float(bound)

    @staticmethod
    def _four_states():
        S=1_000_000_000
        return [(0,100,102,10,20,2,4),(15*S,100,102,14,18,2,3),
            (30*S,101,103,12,16,3,4),(45*S,100,103,8,12,2,3)]

    @pytest.mark.parametrize('cuts',((),(1,),(2,),(1,2,3)))
    def test_literal_signed_transitions_TW_and_own_order_means(self,cuts):
        result=self._run(self._four_states(),cuts=cuts)
        expected={'ofi_contracts':(28,1),'bid_refill_contracts':(4,1),'ask_refill_contracts':(0,1),
            'bid_depletion_contracts':(0,1),'ask_depletion_contracts':(6,1),'quote_updates':(3,1),
            'tw_imbalance':(-673,3360),'tw_spread_ticks':(9,4),'tw_bid_contracts':(11,1),
            'tw_ask_contracts':(33,2),'tw_microprice_offset_ticks':(-757,3360),'mid_response_ticks':(1,2),
            'tw_bid_orders':(9,4),'tw_ask_orders':(7,2),'tw_order_count_imbalance':(-23,105),
            'tw_bid_contracts_per_order':(5,1),'tw_ask_contracts_per_order':(19,4),
            'opening_mid_tick':(101,1),'closing_mid_tick':(203,2)}
        assert tuple(result['book'])==(*a._BOOK_NAMES134,*a._BOOK_ORDER134,'covered_ns','order_covered_ns','opening_mid_tick','closing_mid_tick')
        for name,(num,den) in expected.items():self._rational(result,name,num,den)
        assert result['book']['covered_ns'].dtype==np.dtype('int64')
        assert result['book']['covered_ns'][0]==result['book']['order_covered_ns'][0]==60_000_000_000
        for name in a._BOOK_NAMES134[:6]:assert result['bounds'][name][0]==0.
        assert result['reference_report']['F_LAST_records']==4

    def test_all_final_rows_ignore_event_clean_flags_action_and_trade_context(self):
        rows=self._four_states();normal=self._run(rows)
        dirty=self._run(rows,flags=(128|8|32|4,128,128|32,128|4),action=('R','A','C','M'))
        for name in normal['book']:
            np.testing.assert_array_equal(normal['book'][name],dirty['book'][name])
            np.testing.assert_array_equal(normal['bounds'][name],dirty['bounds'][name])
        assert dirty['reference_report']['event_clean_mask_used'] is False

    def test_nonfinal_intermediate_quote_never_changes_book_state(self):
        rows=self._four_states();inserted=rows[:1]+[(10_000_000_000,1000,1001,1,1,1,1)]+rows[1:]
        actual=self._run(inserted,flags=(128,0,128,128,128));expected=self._run(rows)
        for name in expected['book']:np.testing.assert_array_equal(actual['book'][name],expected['book'][name])

    @pytest.mark.parametrize('badcount',(0,2**32-1))
    def test_order_coverage_is_separate_from_full_price_coverage(self,badcount):
        rows=[(0,100,102,10,20,badcount,4),(30_000_000_000,100,102,10,20,2,4)]
        result=self._run(rows)
        assert result['book']['covered_ns'][0]==60_000_000_000
        assert result['book']['order_covered_ns'][0]==30_000_000_000
        assert np.isfinite(result['book']['tw_imbalance'][0])
        for name in a._BOOK_ORDER134:assert np.isnan(result['book'][name][0]) and np.isnan(result['bounds'][name][0])

    def test_long_gap_only_observes_its_two_boundary_minutes(self):
        S=1_000_000_000
        result=self._run([(10*S,100,102,10,20,2,4),(190*S,102,104,10,20,2,4)],minutes=4,cuts=(1,))
        np.testing.assert_array_equal(result['book']['covered_ns'],[50*S,0,0,60*S])
        np.testing.assert_array_equal(result['book']['order_covered_ns'],[50*S,0,0,60*S])
        assert np.isnan(result['book']['tw_bid_contracts'][:3]).all()
        self._rational(result,'mid_response_ticks',2,row=3)
        self._rational(result,'ofi_contracts',30,row=3)
        assert np.isnan(result['book']['opening_mid_tick'][1:3]).all()

    def test_EOF_never_carries_state_through_unobserved_later_minutes(self):
        result=self._run([(10_000_000_000,100,102,10,20,2,4)],minutes=3)
        np.testing.assert_array_equal(result['book']['covered_ns'],[50_000_000_000,0,0])
        assert np.isnan(result['book']['tw_spread_ticks']).all()
        assert result['book']['quote_updates'].tolist()==[0.,0.,0.]

    def test_file_boundary_starts_new_anchor_and_global_minute_indices(self):
        first=self._run([(0,100,102,10,20,2,4)])
        second=self._run([(70_000_000_000,110,112,5,6,1,2)],start=60_000_000_000)
        assert first['minute_index'].tolist()==[0] and second['minute_index'].tolist()==[1]
        assert second['book']['covered_ns'].tolist()==[50_000_000_000]
        assert second['book']['ofi_contracts'].tolist()==[0.]

    @pytest.mark.parametrize('cuts',((),(1,),(1,2,3)))
    def test_equal_receive_times_keep_row_order_but_no_zero_duration_TW(self,cuts):
        S=1_000_000_000
        result=self._run([(0,100,102,10,10,2,2),(0,101,103,20,10,2,2),
            (30*S,101,103,25,10,2,2),(30*S,102,104,30,10,2,2)],cuts=cuts)
        self._rational(result,'ofi_contracts',75);self._rational(result,'quote_updates',3)
        self._rational(result,'bid_refill_contracts',5);self._rational(result,'tw_bid_contracts',25)
        self._rational(result,'opening_mid_tick',102);self._rational(result,'closing_mid_tick',103)
        self._rational(result,'mid_response_ticks',1)
        assert result['book']['covered_ns'][0]==60*S

    def test_invalid_geometry_drops_only_its_operational_state_coverage(self):
        S=1_000_000_000
        result=self._run([(0,100,102,10,10,2,2),(15*S,101,100,10,10,2,2),(30*S,100,102,10,10,2,2)])
        assert result['book']['covered_ns'][0]==45*S
        assert result['book']['quote_updates'][0]==0.
        assert np.isnan(result['book']['tw_spread_ticks'][0])

    def test_zero_displayed_size_still_participates_in_signed_transition(self):
        S=1_000_000_000
        result=self._run([(0,100,102,10,10,2,2),(15*S,100,102,0,10,2,2),(30*S,100,102,20,10,2,2)])
        self._rational(result,'ofi_contracts',10);self._rational(result,'bid_depletion_contracts',10)
        self._rational(result,'bid_refill_contracts',20);self._rational(result,'quote_updates',2)
        assert result['book']['covered_ns'][0]==45*S
        assert np.isnan(result['book']['tw_bid_contracts'][0])

    def test_signed_integer_floor_is_preserved_before_invalid_geometry_gate(self):
        # A negative fractional raw tick is floored, never truncated towardzero.
        meta=self._meta(tick=10);data=self._data([(0,-1,2,10,10,2,2)],tick=10)
        data=dict(data);data['bid_px_00']=np.array([-1],np.int64);data['bid_px_00'].setflags(write=False)
        owner=a.RawBookReference134(meta);owner.consume(data,self._context(data,meta))
        assert owner.previous[1]==-1
        assert owner.finish()['book']['covered_ns'].tolist()==[0]

    def test_unsafe_midpoint_int64_addition_is_rejected_before_wrap(self):
        with pytest.raises(RuntimeError,match='RAW_BOOK_PRICE_INTEGER_OVERFLOW'):
            self._run([(0,2**62,2**62+1,1,1,1,1)],tick=1)

    def test_wrong_shared_context_clock_and_second_finish_are_rejected(self):
        meta=self._meta();data=self._data([(0,100,102,10,10,2,2)])
        owner=a.RawBookReference134(meta);context=self._context(data,meta);context=dict(context);context['local_index']=np.array([1],np.int64)
        with pytest.raises(RuntimeError,match='RAW_BOOK_CONTEXT_CLOCK'):owner.consume(data,context)
        owner=a.RawBookReference134(meta);owner.consume(data,self._context(data,meta));owner.finish()
        with pytest.raises(RuntimeError,match='RAW_BOOK_ALREADY_FINISHED'):owner.finish()

class TestRawEventReference134:
    @staticmethod
    def _meta(start=0,end=120_000_000_000):
        return {'start_ns':start,'end_ns':end,'study_origin_ns':0,'tick_size':10,
                'raw_symbol':'GCM4','instrument_id':1,'publisher_id':1}

    @staticmethod
    def _data(times,*,ask=None,bid=None,flags=None,action=None,delta=None,event=None):
        n=len(times)
        d={'ts_recv':np.asarray(times,np.uint64),'ts_event':np.asarray(times if event is None else event,np.uint64),
           'rtype':np.ones(n,np.uint8),'publisher_id':np.ones(n,np.uint16),'instrument_id':np.ones(n,np.uint32),
           'action':np.asarray(['M']*n if action is None else action),'side':np.asarray(['N']*n),
           'depth':np.zeros(n,np.uint8),'price':np.full(n,1000,np.int64),'size':np.ones(n,np.uint32),
           'flags':np.asarray([128]*n if flags is None else flags,np.uint8),
           'ts_in_delta':np.asarray([0]*n if delta is None else delta,np.int32),'sequence':np.arange(n,dtype=np.uint32),
           'bid_px_00':np.asarray([1000]*n if bid is None else bid,np.int64),
           'ask_px_00':np.asarray([1020]*n if ask is None else ask,np.int64),
           'bid_sz_00':np.full(n,5,np.uint32),'ask_sz_00':np.full(n,7,np.uint32),
           'bid_ct_00':np.ones(n,np.uint32),'ask_ct_00':np.ones(n,np.uint32),'symbol':np.asarray(['GCM4']*n)}
        for x in d.values():x.setflags(write=False)
        return d

    @staticmethod
    def _context(data,meta):
        receive=data['ts_recv'].astype(np.int64);minute=(receive-meta['study_origin_ns'])//60_000_000_000
        local=(receive-meta['start_ns'])//60_000_000_000
        minute.setflags(write=False);local.setflags(write=False)
        return {'minute_index':minute,'local_index':local}

    @classmethod
    def _run(cls,data,*,meta=None,cuts=(),covered=None):
        meta=cls._meta() if meta is None else meta;owner=a.RawEventReference134(meta)
        points=(0,)+tuple(cuts)+(len(data['ts_recv']),)
        for lo,hi in zip(points,points[1:]):
            batch={k:x[lo:hi] for k,x in data.items()};owner.consume(batch,cls._context(batch,meta))
        n=(meta['end_ns']-meta['start_ns'])//60_000_000_000
        minute=np.arange(n,dtype=np.int64)+(meta['start_ns']-meta['study_origin_ns'])//60_000_000_000
        coverage=np.full(n,60_000_000_000,np.int64) if covered is None else np.asarray(covered,np.int64)
        return owner.finish({'minute_index':minute,'book':{'covered_ns':coverage}})

    def test_hand_events_ties_bad_previous_reset_clamps_and_three_clocks(self):
        data=self._data([10,10,20,30,40,50,60,70],ask=[1020,1040,1030,1050,1060,1080,1100,1020],
            flags=[128,128,132,128,128,128,160,128],action=['M','M','M','M','R','M','M','M'],
            delta=[5,-2,7,-(1<<31),3,(1<<31)-1,1,4],event=[5,20,15,21,35,40,59,(1<<64)-1])
        result=self._run(data);v=result['event'];e=result['bounds']
        assert len(v)==37 and tuple(v)==a.EVENT134_FIELDS
        expected={'qualified_final_event_count':5,'spread_state_min_ticks':2,'spread_state_max_ticks':8,
            'spread_change_count':1,'spread_expansion_count':1,'spread_contraction_count':0,
            'spread_expansion_sum_ticks':2,'spread_contraction_sum_ticks':0,
            'spread_max_expansion_ticks':2,'spread_max_contraction_ticks':0,
            'receive_interval_count':1,'receive_interval_sum_ns':0,'receive_interval_min_ns':0,'receive_interval_max_ns':0,
            'ts_in_delta_clamped_low_count':1,'ts_in_delta_clamped_high_count':1,'undefined_event_clock_count':1,
            'negative_send_to_capture_count':1,'negative_matching_clock_count':1,
            'send_to_capture_count':3,'send_to_capture_sum_ns':7,'send_to_capture_min_ns':-2,'send_to_capture_max_ns':5,
            'receive_minus_event_count':4,'receive_minus_event_sum_ns':14,'receive_minus_event_min_ns':-10,'receive_minus_event_max_ns':10,
            'matching_clock_count':2,'matching_clock_sum_ns':-8,'matching_clock_min_ns':-8,'matching_clock_max_ns':0,
            'receive_interval_mean_ns':0,'send_to_capture_mean_ns':7/3,'receive_minus_event_mean_ns':3.5,
            'matching_clock_mean_ns':-4,'qualified_covered_ns':60_000_000_000,'qualified_events_per_observed_second':1/12}
        assert set(expected)==set(v)
        for name,wanted in expected.items():assert v[name][0]==wanted and e[name][0]==0
        assert v['qualified_final_event_count'][1]==0
        assert np.isnan(v['spread_state_min_ticks'][1]) and np.isnan(v['matching_clock_mean_ns'][1])
        assert result['reference_report']['candidate_mask_applied'] is False

    def test_tick_floor_clean_geometry_undefined_sizes_and_nonfinal_rows(self):
        data=self._data([1,2,3,4,5,6,7],bid=[1009,1009,0,-1,1000,1000,1000],
            ask=[1029,1009,1050,1050,1009,1030,1040],flags=[128,128,128,128,128,0,128])
        # Positive UINT32 sentinel size remains accepted by the event owner;
        # this is deliberately different from order-count/timeweight masks.
        data['bid_sz_00']=np.array([5,5,5,5,5,5,(1<<32)-1],np.uint32);data['bid_sz_00'].setflags(write=False)
        v=self._run(data)['event']
        assert v['qualified_final_event_count'][0]==2
        assert v['spread_state_min_ticks'][0]==2 and v['spread_state_max_ticks'][0]==4
        assert v['receive_interval_count'][0]==0  # Bad final predecessor survives the nonfinal row.

    def test_prior_spread_is_added_to_new_minute_and_batches_are_identical(self):
        data=self._data([59_999_999_990,60_000_000_010],ask=[1040,1020])
        full=self._run(data,covered=[60_000_000_000,0]);split=self._run(data,cuts=(1,),covered=[60_000_000_000,0])
        for group in ('event','bounds'):
            for name in full[group]:np.testing.assert_array_equal(full[group][name],split[group][name])
        v=full['event']
        assert v['spread_state_min_ticks'][1]==2 and v['spread_state_max_ticks'][1]==4
        assert v['spread_contraction_count'][1]==1 and v['spread_contraction_sum_ticks'][1]==2
        assert v['receive_interval_count'][1]==1 and v['receive_interval_sum_ns'][1]==20
        assert np.isnan(v['qualified_events_per_observed_second'][1])

    def test_file_boundary_reset_does_not_invent_a_cross_file_transition(self):
        first=self._run(self._data([59_999_999_990],ask=[1040]),meta=self._meta(end=60_000_000_000))
        second=self._run(self._data([60_000_000_010],ask=[1020]),meta=self._meta(start=60_000_000_000))
        assert first['event']['receive_interval_count'][0]==second['event']['receive_interval_count'][0]==0
        assert second['event']['spread_state_max_ticks'][0]==2
        np.testing.assert_array_equal(second['minute_index'],[1])

    def test_backward_receive_event_breaks_only_that_transition_and_updates_predecessor(self):
        v=self._run(self._data([20,10,30],ask=[1020,1050,1030]))['event']
        assert v['qualified_final_event_count'][0]==3 and v['receive_interval_count'][0]==1
        assert v['receive_interval_sum_ns'][0]==20 and v['spread_contraction_sum_ticks'][0]==2
        assert v['spread_expansion_count'][0]==0

    def test_defined_uint64_boundary_and_signed_integer_group_overflow_are_proved(self):
        maximum=(1<<63)-1
        data=self._data([10,20,30],event=[maximum,maximum,(1<<64)-1])
        result=self._run(data);v=result['event'];e=result['bounds']
        exact=-2*maximum+30
        assert v['receive_minus_event_count'][0]==2 and v['undefined_event_clock_count'][0]==1
        assert v['receive_minus_event_sum_ns'][0]==float(exact)
        source=0.
        for x in (10-maximum,20-maximum):source+=float(x)
        assert abs(source-v['receive_minus_event_sum_ns'][0])<=e['receive_minus_event_sum_ns'][0]
        assert e['receive_minus_event_sum_ns'][0]>0
        assert int(result['reference_report']['maximum_absolute_integer_mass'])>=abs(exact)
        assert v['negative_matching_clock_count'][0]==2

    def test_signed_cancellation_roundoff_bound_comes_from_input_mass_not_target(self):
        wide=1<<60;data=self._data([100,200,300],event=[0,wide+200,0])
        result=self._run(data);v=result['event'];e=result['bounds'];source=0.
        for x in (100,-wide,300):source+=float(x)
        assert v['receive_minus_event_sum_ns'][0]==float(400-wide)
        assert e['receive_minus_event_sum_ns'][0]>0
        assert abs(source-v['receive_minus_event_sum_ns'][0])<=e['receive_minus_event_sum_ns'][0]
        source_mean=source/3
        assert abs(source_mean-v['receive_minus_event_mean_ns'][0])<=e['receive_minus_event_mean_ns'][0]
        assert result['reference_report']['target_cache_used'] is False

    def test_every_batch_partition_preserves_all_event_values_and_radii(self):
        data=self._data([10,10,20,30,40,50],ask=[1020,1040,1030,1060,1020,1040],
            flags=[128,128,132,128,128,128],delta=[0,-2,1,3,4,5],event=[9,12,15,31,35,49])
        full=self._run(data)
        for cuts in ((1,),(2,4),(1,2,3,4,5)):
            split=self._run(data,cuts=cuts)
            for group in ('event','bounds'):
                for name in full[group]:np.testing.assert_array_equal(full[group][name],split[group][name])

    def test_matching_subtraction_overflow_fails_before_numeric_reduction(self):
        data=self._data([0],event=[(1<<63)-1],delta=[2]);owner=a.RawEventReference134(self._meta())
        with pytest.raises(RuntimeError,match='RAW_EVENT134_MATCHING_CLOCK_OVERFLOW'):
            owner.consume(data,self._context(data,self._meta()))

    def test_context_book_domain_and_finish_are_strict(self):
        data=self._data([10]);meta=self._meta();owner=a.RawEventReference134(meta)
        with pytest.raises(RuntimeError,match='RAW_EVENT134_CONTEXT_CLOCK'):
            owner.consume(data,{'minute_index':np.array([0]),'local_index':np.array([1])})
        owner=a.RawEventReference134(meta);owner.consume(data,self._context(data,meta))
        with pytest.raises(RuntimeError,match='RAW_EVENT134_BOOK_COVERAGE'):
            owner.finish({'minute_index':np.array([0,1],np.int64),'book':{'covered_ns':np.array([1.,2.])}})
        owner.finish({'minute_index':np.array([0,1],np.int64),'book':{'covered_ns':np.array([1,2],np.int64)}})
        with pytest.raises(RuntimeError,match='RAW_EVENT134_FINISHED'):owner.consume(data,self._context(data,meta))

def _raw_reader_fixture134(tmp_path,rows=None,header=None):
    names=list(a.RAW134_COLUMN_TYPES) if header is None else header
    base=dict(ts_recv=0,ts_event=0,rtype=1,publisher_id=1,instrument_id=1551,
        action='T',side='B',depth=0,price=100_000_000_000,size=2,flags=128,
        ts_in_delta=0,sequence=1,bid_px_00=99_900_000_000,ask_px_00=100_100_000_000,
        bid_sz_00=3,ask_sz_00=5,bid_ct_00=1,ask_ct_00=2,symbol='GCG5')
    rows=[{}] if rows is None else rows
    string=io.StringIO(newline='');writer=csv.writer(string,lineterminator='\n');writer.writerow(names)
    for change in rows:
        value=dict(base,**change);writer.writerow([value[k] for k in names])
    data=string.getvalue().encode();path=tmp_path/'fixture.csv.gz';compressed=gzip.compress(data,mtime=0);path.write_bytes(compressed)
    item={'path':str(path),'start_ns':0,'end_ns':a.RAW134_MINUTE,'bytes':len(compressed),
        'max_records':len(data),'instrument_id':1551,'publisher_id':1,'raw_symbol':'GCG5',
        'compression':'gzip','pretty_ts':False,'pretty_px':False,'sha256':hashlib.sha256(compressed).hexdigest()}
    meta={**item,'study_origin_ns':0,'tick_size':100_000_000}
    return item,meta,data


class TestRaw134SharedReader:
    def test_one_parse_typed_readonly_complete_hash(self,tmp_path):
        item,meta,csv_bytes=_raw_reader_fixture134(tmp_path,[{},dict(ts_recv=1,ts_event=1,sequence=2,side='A',size=3)])
        reader=a.RawMBPBatchReader134(item,block_size=1024);arrays=list(reader)
        assert sum(len(x['ts_recv']) for x in arrays)==2
        for data in arrays:
            assert list(data)==list(a.RAW134_COLUMN_TYPES)
            for name,kind in a.RAW134_COLUMN_TYPES.items():
                assert data[name].dtype==np.dtype('object' if kind=='string' else kind)
                assert not data[name].flags.writeable
        assert reader.receipt['actual_EOF_and_gzip_CRC_verified']
        assert reader.receipt['sha256']==item['sha256']
        assert reader.receipt['decompressed_csv_bytes']==len(csv_bytes)
        assert reader.receipt['compressed_payload_passes']==reader.receipt['CSV_parse_passes']==1
        with pytest.raises(RuntimeError,match='CONSUMED'):list(reader)

    def test_prefix_never_admits_full_file_even_if_read_ahead(self,tmp_path):
        item,meta,_=_raw_reader_fixture134(tmp_path)
        reader=a.RawMBPBatchReader134(item,block_size=1024);iterator=iter(reader);next(iterator);iterator.close()
        assert reader.receipt is None

    @pytest.mark.parametrize('field,value',[('rtype',2),('publisher_id',2),('instrument_id',1552),
        ('symbol','GCH5'),('action','X'),('side','X'),('ts_recv',-1),('ts_recv',60_000_000_000),('size',2**32)])
    def test_identity_domain_window_and_arrow_integer_limits(self,tmp_path,field,value):
        item,meta,_=_raw_reader_fixture134(tmp_path,[{field:value}])
        with pytest.raises((RuntimeError,a.pa.ArrowInvalid)):list(a.RawMBPBatchReader134(item,block_size=1024))

    def test_receive_order_with_timestamp_ties_keeps_row_order(self,tmp_path):
        item,meta,_=_raw_reader_fixture134(tmp_path,[{'sequence':2},{'sequence':1}])
        out=list(a.RawMBPBatchReader134(item,block_size=1024))
        assert np.concatenate([x['sequence'] for x in out]).tolist()==[2,1]
        item,meta,_=_raw_reader_fixture134(tmp_path,[{'ts_recv':1},{'ts_recv':0}])
        with pytest.raises(RuntimeError,match='RECEIVE_ORDER'):list(a.RawMBPBatchReader134(item,block_size=1024))

    def test_wrong_sha_and_decompression_cap_are_not_receipts(self,tmp_path):
        item,meta,_=_raw_reader_fixture134(tmp_path)
        reader=a.RawMBPBatchReader134(dict(item,sha256='0'*64),block_size=1024)
        with pytest.raises(RuntimeError,match='COMPRESSED_SHA'):list(reader)
        assert reader.receipt is None
        reader=a.RawMBPBatchReader134(dict(item,max_records=12),block_size=1024)
        with pytest.raises(RuntimeError,match='DECOMPRESSED_BYTE_CAP'):list(reader)
        assert reader.receipt is None

    def test_bad_gzip_crc_even_with_bound_compressed_hash(self,tmp_path):
        item,meta,_=_raw_reader_fixture134(tmp_path);path=Path(item['path']);body=bytearray(path.read_bytes());body[-8]^=1
        path.write_bytes(body);item['sha256']=hashlib.sha256(body).hexdigest();reader=a.RawMBPBatchReader134(item,block_size=1024)
        with pytest.raises((OSError,EOFError)):list(reader)
        assert reader.receipt is None

    def test_column_count_duplicate_names_null_and_symlink(self,tmp_path):
        for names in [list(a.RAW134_COLUMN_TYPES)[:-1],list(a.RAW134_COLUMN_TYPES)[:-1]+['ts_recv']]:
            item,meta,_=_raw_reader_fixture134(tmp_path,header=names)
            with pytest.raises(RuntimeError,match='TWENTY_COLUMNS'):list(a.RawMBPBatchReader134(item,block_size=1024))
        item,meta,_=_raw_reader_fixture134(tmp_path,[dict(price='')])
        with pytest.raises(RuntimeError,match='NULL_RECORD'):list(a.RawMBPBatchReader134(item,block_size=1024))
        item,meta,_=_raw_reader_fixture134(tmp_path);link=tmp_path/'link.gz';link.symlink_to(Path(item['path']))
        with pytest.raises(RuntimeError,match='PATH_IDENTITY'):a.RawMBPBatchReader134(dict(item,path=str(link)))

    def test_complete_coordinator_consumes_all_ledgers_once(self,tmp_path):
        item,meta,_=_raw_reader_fixture134(tmp_path,[{},dict(ts_recv=1_000_000_000,ts_event=1_000_000_000,side='A',size=3,sequence=2)])
        ladders=[];out=a.reduce_raw_file134(item,meta,on_ladder=ladders.append,block_size=1024)
        assert out['raw_receipt']['raw_records']==2
        assert out['trade']['source_metrics11']['trades'].tolist()==[2]
        assert out['trade']['source_metrics11']['volume'].tolist()==[5]
        assert out['trade']['trade_size7']['size_sum_squared_contracts2'].tolist()==[13]
        assert out['book']['book']['covered_ns'].tolist()==[a.RAW134_MINUTE]
        assert out['event']['event']['qualified_final_event_count'].tolist()==[2.]
        assert sum(sum(map(int,x['buy']))+sum(map(int,x['sell'])) for x in ladders)==5
        assert all(out[k]['reference_report']['production_reducers_called'] is False for k in ['trade'])


    def test_receive_order_is_checked_across_arrow_batches(self,tmp_path):
        rows=[dict(ts_recv=i,ts_event=i,sequence=i) for i in range(100)]
        item,meta,_=_raw_reader_fixture134(tmp_path,rows)
        first=a.RawMBPBatchReader134(item,block_size=1024);stream=iter(first);width=len(next(stream)['ts_recv']);stream.close()
        assert 0<width<len(rows) and first.receipt is None
        rows[width]['ts_recv']=rows[width-1]['ts_recv']-1
        item,meta,_=_raw_reader_fixture134(tmp_path,rows);reader=a.RawMBPBatchReader134(item,block_size=1024)
        with pytest.raises(RuntimeError,match='RECEIVE_ORDER'):list(reader)
        assert reader.receipt is None

    def test_identity_change_after_yield_cannot_publish_receipt(self,tmp_path):
        item,meta,_=_raw_reader_fixture134(tmp_path);reader=a.RawMBPBatchReader134(item,block_size=1024)
        stream=iter(reader);next(stream);path=Path(item['path']);before=path.stat();a.os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1_000_000))
        with pytest.raises(RuntimeError,match='FINAL_IDENTITY'):next(stream)
        assert reader.receipt is None


class TestRaw134TrainPrefixExtraction:
    """Synthetic-only qualification of the separate TRAIN byte-copy boundary."""
    @staticmethod
    def _source(tmp_path,rows=None,header=None):
        if rows is None:
            rows=[dict(ts_recv=0),dict(ts_recv=a.RAW134_MINUTE)]
        item,_,data=_raw_reader_fixture134(tmp_path,rows,header)
        item['end_ns']=2*a.RAW134_MINUTE
        return item,data

    @staticmethod
    def _extract(tmp_path,item,**changes):
        args=dict(cutoff_ns=a.RAW134_MINUTE,compression_level=1,max_line_bytes=4096)
        args.update(changes)
        return a.extract_raw_gc_train_prefix134(item,tmp_path/'prefix.csv.gz',**args)

    @pytest.mark.parametrize('count',(1,257))
    def test_exact_prefix_ties_buffers_and_unchanged_reader_roundtrip(self,tmp_path,count):
        rows=[dict(ts_recv=i//2,ts_event=i,sequence=i+1) for i in range(count)]
        rows.append(dict(ts_recv=a.RAW134_MINUTE,ts_event=a.RAW134_MINUTE))
        item,data=self._source(tmp_path,rows)
        result=self._extract(tmp_path,item);derived=result['derived_item'];receipt=result['extraction_receipt']
        expected=b''.join(data.splitlines(keepends=True)[:-1])
        path=Path(derived['path'])
        assert gzip.decompress(path.read_bytes())==expected
        assert derived['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']
        assert derived['bytes']==path.stat().st_size
        assert derived['max_records']==len(expected) and derived['end_ns']==a.RAW134_MINUTE
        assert derived['local_derivation']['source']['sha256']==item['sha256']
        assert 'receipt_path' not in derived and 'receipt_sha256' not in derived
        assert receipt['copied_csv_bytes_sha256']==hashlib.sha256(expected).hexdigest()
        assert receipt['source_opaque_hash_bytes']==item['bytes']
        assert receipt['source_same_descriptor_hash_then_prefix'] and receipt['source_identity_bookends']
        assert receipt['source_full_EOF_and_gzip_CRC_verified'] is False
        assert receipt['source_full_file_admitted'] is False and receipt['market_fields_parsed']==0
        assert receipt['first_outside_receive_ns']==a.RAW134_MINUTE
        assert receipt['raw_records_copied']==count and receipt['timestamps_inspected']==count+1
        assert not (tmp_path/'.prefix.csv.gz.stage').exists()
        reader=a.RawMBPBatchReader134(derived,block_size=1024);batches=list(reader)
        assert np.concatenate([v['ts_recv'] for v in batches]).tolist()==[i//2 for i in range(count)]
        assert np.concatenate([v['sequence'] for v in batches]).tolist()==list(range(1,count+1))
        assert reader.receipt['actual_EOF_and_gzip_CRC_verified']
        assert reader.receipt['full_file_admitted'] and reader.receipt['raw_records']==count
        assert reader.receipt['decompressed_csv_bytes']==len(expected)

    def test_future_poison_and_overlong_market_tail_are_opaque(self,tmp_path,monkeypatch):
        item,data=self._source(tmp_path)
        expected=b''.join(data.splitlines(keepends=True)[:-1])
        poisoned=expected+str(a.RAW134_MINUTE).encode()+b',\xff"'+b'X'*100000+b'\n'
        body=gzip.compress(poisoned,mtime=0);Path(item['path']).write_bytes(body)
        item.update(bytes=len(body),max_records=len(poisoned),sha256=hashlib.sha256(body).hexdigest())
        def forbidden(*args,**kwargs):
            raise AssertionError('Extractor must not call the market CSV parser')
        with monkeypatch.context() as patch:
            patch.setattr(a.pacsv,'open_csv',forbidden)
            result=self._extract(tmp_path,item,max_line_bytes=len(expected.splitlines(keepends=True)[0])+1)
        assert gzip.decompress(Path(result['derived_item']['path']).read_bytes())==expected
        assert result['extraction_receipt']['source_decompressed_bytes_returned']<=len(expected)+21
        assert result['extraction_receipt']['market_fields_parsed']==0

    @pytest.mark.parametrize('token',('',-1,'+1','1.0','"1"',str(1<<64),'0'*21))
    def test_timestamp_tokens_fail_without_publishing(self,tmp_path,token):
        item,_=self._source(tmp_path,[dict(ts_recv=token),dict(ts_recv=a.RAW134_MINUTE)])
        with pytest.raises(RuntimeError,match='RECEIVE_TOKEN'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()
        assert (tmp_path/'.prefix.csv.gz.stage').exists()

    @pytest.mark.parametrize('fault',('missing','duplicate','receive_not_first'))
    def test_header_owner_and_receive_position_are_explicit(self,tmp_path,fault):
        names=list(a.RAW134_COLUMN_TYPES)
        if fault=='missing':names=names[:-1]
        elif fault=='duplicate':names[-1]='ts_recv'
        else:names[0],names[1]=names[1],names[0]
        item,_=self._source(tmp_path,header=names)
        with pytest.raises(RuntimeError,match='HEADER_TWENTY_COLUMNS_RECV_FIRST'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()

    @pytest.mark.parametrize('fault',('decreasing','below_start','outside_source_end'))
    def test_receive_order_and_original_source_window(self,tmp_path,fault):
        changes={}
        if fault=='decreasing':
            rows=[dict(ts_recv=2),dict(ts_recv=1),dict(ts_recv=a.RAW134_MINUTE)]
        elif fault=='below_start':
            rows=[dict(ts_recv=a.RAW134_MINUTE-1),dict(ts_recv=2*a.RAW134_MINUTE)]
            changes['cutoff_ns']=2*a.RAW134_MINUTE
        else:
            rows=[dict(ts_recv=0),dict(ts_recv=2*a.RAW134_MINUTE)]
        item,_=self._source(tmp_path,rows)
        if fault=='below_start':item.update(start_ns=a.RAW134_MINUTE,end_ns=3*a.RAW134_MINUTE)
        with pytest.raises(RuntimeError,match='RECEIVE_ORDER_OR_WINDOW'):
            self._extract(tmp_path,item,**changes)
        assert not (tmp_path/'prefix.csv.gz').exists()

    @pytest.mark.parametrize('fault',('decompressed_cap','train_line_cap'))
    def test_explicit_byte_and_train_line_limits(self,tmp_path,fault):
        rows=None if fault=='decompressed_cap' else [
            dict(ts_recv=0,symbol='X'*5000),dict(ts_recv=a.RAW134_MINUTE)]
        item,_=self._source(tmp_path,rows)
        if fault=='decompressed_cap':item['max_records']=12
        with pytest.raises(RuntimeError,match='DECOMPRESSED_BYTE_CAP|CSV_LINE'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()

    def test_natural_original_EOF_before_boundary_is_explicitly_rejected(self,tmp_path):
        item,_=self._source(tmp_path,[dict(ts_recv=0)])
        with pytest.raises(RuntimeError,match='BOUNDARY_NOT_FOUND'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()
        assert (tmp_path/'.prefix.csv.gz.stage').exists()

    def test_empty_train_prefix_is_not_an_admitted_source(self,tmp_path):
        item,_=self._source(tmp_path,[dict(ts_recv=a.RAW134_MINUTE)])
        with pytest.raises(RuntimeError,match='EMPTY_TRAIN'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()

    @pytest.mark.parametrize('change',(
        {'cutoff_ns':0},{'cutoff_ns':2*a.RAW134_MINUTE},{'cutoff_ns':True},
        {'compression_level':-1},{'max_line_bytes':0},
    ))
    def test_explicit_recipe_arguments_fail_closed(self,tmp_path,change):
        item,_=self._source(tmp_path)
        with pytest.raises(RuntimeError,match='PREFIX_CUTOFF|COMPRESSION_LEVEL|LINE_LIMIT'):
            self._extract(tmp_path,item,**change)
        assert not (tmp_path/'prefix.csv.gz').exists()

    def test_wrong_opaque_SHA_stops_before_prefix_and_stage(self,tmp_path):
        item,_=self._source(tmp_path);item['sha256']='0'*64
        with pytest.raises(RuntimeError,match='COMPRESSED_SHA_OR_SIZE'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()
        assert not (tmp_path/'.prefix.csv.gz.stage').exists()

    @pytest.mark.parametrize('identity_call',(3,4))
    def test_source_identity_changes_after_hash_or_prefix_cannot_publish(self,tmp_path,monkeypatch,identity_call):
        item,_=self._source(tmp_path);original=a.raw134_identity;calls=0
        def tamper(path):
            nonlocal calls
            if Path(path)==Path(item['path']):
                calls+=1
                if calls==identity_call:
                    stat=Path(path).stat()
                    a.os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
            return original(path)
        monkeypatch.setattr(a,'raw134_identity',tamper)
        with pytest.raises(RuntimeError,match='SOURCE_IDENTITY'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()

    @pytest.mark.parametrize('existing',('destination','stage'))
    def test_existing_outputs_and_failed_stages_are_never_replaced(self,tmp_path,existing):
        item,_=self._source(tmp_path)
        path=tmp_path/('prefix.csv.gz' if existing=='destination' else '.prefix.csv.gz.stage')
        path.write_bytes(b'preserve')
        with pytest.raises(RuntimeError,match='OUTPUT_EXISTS|STAGE_EXISTS'):
            self._extract(tmp_path,item)
        assert path.read_bytes()==b'preserve'

    def test_publication_failure_preserves_completed_stage(self,tmp_path,monkeypatch):
        from gx1.contracts import immutable_event_authority_v1 as publication
        item,data=self._source(tmp_path)
        def fail(source,destination):
            raise OSError('injected publication failure')
        monkeypatch.setattr(publication,'_publish_file_noreplace',fail)
        with pytest.raises(OSError,match='injected publication failure'):
            self._extract(tmp_path,item)
        assert not (tmp_path/'prefix.csv.gz').exists()
        assert gzip.decompress((tmp_path/'.prefix.csv.gz.stage').read_bytes())==b''.join(
            data.splitlines(keepends=True)[:-1])

    def test_publication_race_uses_real_no_replace_authority(self,tmp_path,monkeypatch):
        from gx1.contracts import immutable_event_authority_v1 as publication
        item,_=self._source(tmp_path);original=publication._publish_file_noreplace
        def race(source,destination):
            Path(destination).write_bytes(b'concurrent output')
            original(source,destination)
        monkeypatch.setattr(publication,'_publish_file_noreplace',race)
        with pytest.raises(publication.ImmutableEventAuthorityError,match='already exists'):
            self._extract(tmp_path,item)
        assert (tmp_path/'prefix.csv.gz').read_bytes()==b'concurrent output'
        assert (tmp_path/'.prefix.csv.gz.stage').exists()

    def test_derived_reader_prefix_still_cannot_emit_full_file_receipt(self,tmp_path):
        item,_=self._source(tmp_path)
        result=self._extract(tmp_path,item)
        reader=a.RawMBPBatchReader134(result['derived_item'],block_size=1024)
        stream=iter(reader);next(stream);stream.close()
        assert reader.receipt is None


class TestRaw134Comparison:
    def test_integer_sign_and_above_float_precision_are_exact(self):
        huge=np.array([2**63+1],np.uint64)
        assert a.compare_raw_values134({'x':huge},{'x':huge})['x']['cells']==1
        with pytest.raises(RuntimeError,match='EXACT_MISMATCH'):
            a.compare_raw_values134({'x':huge},{'x':np.array([2**63-1],np.int64)})
        with pytest.raises(RuntimeError,match='EXACT_MISMATCH'):
            a.compare_raw_values134({'x':np.array([2**64-1],np.uint64)},{'x':np.array([-1],np.int64)})

    def test_frozen_radius_domains_and_nan_availability(self):
        x=np.array([1.,np.nan]);same=x.copy();zero=np.array([0.,np.nan])
        a.compare_raw_values134({'x':x},{'x':same},bounds={'x':zero})
        actual=np.array([np.nextafter(1.,np.inf),np.nan]);delta=actual[0]-1.
        a.compare_raw_values134({'x':x},{'x':actual},bounds={'x':np.array([np.nextafter(delta,np.inf),np.nan])})
        for target,bounds in [(actual,zero),(np.array([1.,0.]),zero),(same,np.array([-1.,np.nan])),(same,np.array([np.inf,np.nan]))]:
            with pytest.raises(RuntimeError):a.compare_raw_values134({'x':x},{'x':target},bounds={'x':bounds})



class TestRaw134StreamingLadder:
    @staticmethod
    def _batch(rows):
        return {k:np.array([row[j] for row in rows],dtype=np.int64 if j<2 else np.uint64)
            for j,k in enumerate(a.RAW134_LADDER_FIELDS)}
    def test_exact_merge_across_batch_and_empty_minute(self):
        target=self._batch([(0,100,3,0,0),(0,100,2,1,0),(2,101,0,0,4)])
        c=a.RawLadderComparator134(target)
        c.consume(self._batch([(0,100,3,0,0)]));assert c.cursor==0
        c.consume(self._batch([(0,100,2,1,0),(2,101,0,0,4)]));assert c.cursor==1
        report=c.finish();assert report['qualified_price_side_levels']==2
    def test_wrong_side_quantity_and_missing_target_levels_reject(self):
        target=self._batch([(0,100,3,0,0)])
        c=a.RawLadderComparator134(target);c.consume(self._batch([(0,100,0,3,0)]))
        with pytest.raises(RuntimeError,match='EXACT_MISMATCH'):c.finish()
        c=a.RawLadderComparator134(target)
        with pytest.raises(RuntimeError,match='MISSING_ROWS'):c.finish()
    def test_grouped_unsigned_overflow_never_wraps(self):
        with pytest.raises(RuntimeError,match='UINT64_OVERFLOW'):
            a.canonical_ladder134(self._batch([(0,100,2**64-1,0,0),(0,100,1,0,0)]))
    def test_m1_float_integer_dispatch_and_selected_clock(self,tmp_path):
        item,meta,_=_raw_reader_fixture134(tmp_path)
        out=a.reduce_raw_file134(item,meta,block_size=1024);t,b,e=out['trade'],out['book'],out['event']
        m1={**t['source_metrics11'],**{'quality_'+k:v for k,v in t['quality11'].items()},**t['ohlc'],**b['book']}
        event={'minute_index':e['minute_index'],**e['event']};size={'minute_index':t['minute_index'],**t['trade_size7']}
        result=a.compare_reduced_file134(out,m1,event,size,t['minute_index'])
        assert result['unfiltered_integer']['covered_ns']['arithmetic']=='EXACT_INTEGER'
        assert result['unfiltered_float']['tw_spread_ticks']['finite']==1
        with pytest.raises(RuntimeError,match='SELECTED_CLOCK'):
            a.compare_reduced_file134(out,m1,dict(event,minute_index=np.array([1],np.int64)),size,t['minute_index'])


class TestRawSelectionMasks134:
    MINUTE=60_000_000_000
    DAY=86_400_000_000_000
    ORIGIN=1714521600000000000
    END=1782864000000000000

    @staticmethod
    def _manifest(start='2025-09-17T13:34:28Z',end='2025-09-17T13:50:40Z'):
        return {'schema_version':'gx1_gc_vendor_source_incidents_v1','dataset':'GLBX.MDP3',
            'nontrade_action_semantics_admitted':False,'incidents':[{'incident_id':'literal',
                'source_url':'https://example.invalid/declared-incident','start_inclusive_utc':start,'end_exclusive_utc':end}],
            'trusted_trade_admission':{'reject_duplicate_signature':False,'deduplicate_T':False}}

    @classmethod
    def _fixture(cls,n,start=None):
        start=cls.ORIGIN+2*cls.DAY if start is None else start
        clock=start+np.arange(n,dtype=np.int64)*cls.MINUTE
        selected=np.zeros(n,np.int16)
        metrics={name:np.zeros(n,np.uint64) for name in ('records','trades','volume','buy','sell','unknown',
            'invalid_trade','bad_recv_trade','gap','snapshot_trade','duplicate_trade')}
        for name in ('records','trades','volume','buy'):metrics[name][:]=1
        quality={name:np.zeros(n,np.uint64) for name in ('records','clean_complete_bbo','bad_bbo','gap','bad_recv',
            'snapshot','reset','duplicate_trade_signature','full_record_duplicate_trade','signature_collision','file_seam')}
        quality['records'][:]=quality['clean_complete_bbo'][:]=1
        return clock,selected,metrics,quality,np.full(n,cls.MINUTE,np.int64),np.full(n,cls.MINUTE,np.int64)

    @staticmethod
    def _clock(text,n):
        start=np.datetime64(text,'ns').astype(np.int64)
        return start+np.arange(n,dtype=np.int64)*60_000_000_000

    def test_prior_complete_day_unique_tie_hold_zero_and_no_backward_maturity(self):
        volume=np.array([[5,7,9,0,100,100,100,0],
                         [5,5,9,10,0,5,0,0],
                         [0,0,0,2,0,5,8,0]],np.uint64)
        chosen,basis=a.reference_roll134(volume,[(2024,6),(2024,8),(2024,12)])
        np.testing.assert_array_equal(chosen,[-1,-1,0,0,1,1,1,2])
        np.testing.assert_array_equal(basis,[-1,-1,1,1,3,3,3,6])
        # A future current-day column never chooses the current day's identity.
        changed=volume.copy();changed[:,-1]=[0,0,1000000]
        np.testing.assert_array_equal(a.reference_roll134(changed,[(2024,6),(2024,8),(2024,12)])[0],chosen)

    def test_unique_same_identity_refreshes_basis_and_every_prefix_is_causal(self):
        volume=np.array([[1,2,3,0],[0,0,0,100]],np.uint64);maturity=[(2024,6),(2024,8)]
        chosen,basis=a.reference_roll134(volume,maturity)
        np.testing.assert_array_equal(chosen,[-1,0,0,0]);np.testing.assert_array_equal(basis,[-1,0,1,2])
        for stop in (1,2,3):
            p,b=a.reference_roll134(volume[:,:stop],maturity)
            np.testing.assert_array_equal(p,chosen[:stop]);np.testing.assert_array_equal(b,basis[:stop])

    def test_roll_uses_actual_maturity_order_rather_than_symbol_index(self):
        volume=np.array([[0,20,0],[10,1,100]],np.uint64)
        chosen,_=a.reference_roll134(volume,[(2025,2),(2024,6)])
        np.testing.assert_array_equal(chosen,[-1,1,0])
        with pytest.raises(RuntimeError,match='RAW_MASK134_MATURITY_IDENTITY'):
            a.reference_roll134(volume,[(2024,6),(2024,6)])

    def test_roll_changes_at_zero_zero_one_UTC_and_keeps_first_day_unselected(self):
        daily=np.array([-1,0,1],np.int16)
        offsets=np.array([0,1,1439,1440,1441,2879,2880,2881],np.int64)
        selected=a.reference_roll_minutes134(self.ORIGIN+offsets*self.MINUTE,daily)
        np.testing.assert_array_equal(selected,[-1,-1,-1,-1,0,0,0,1])

    def test_literal_seven_masks_keep_duplicate_T_and_separate_volume_book_coverage(self):
        t,s,m,q,covered,order=self._fixture(10);s[:]=[-1,0,0,0,1,1,1,1,1,1]
        q['file_seam'][1]=1;q['bad_bbo'][3]=1;q['records'][3]=2
        m['invalid_trade'][2]=1;m['records'][2]=2;q['records'][2]=2
        m['unknown'][5]=1;m['buy'][5]=0
        m['trades'][6]=m['volume'][6]=m['buy'][6]=0
        m['gap'][7]=q['gap'][7]=1;q['records'][7]=2
        m['duplicate_trade'][9]=q['duplicate_trade_signature'][9]=q['full_record_duplicate_trade'][9]=1
        covered[6]-=1;order[9]-=1
        out=a.reference_masks134(t,s,m,q,covered,order,self._manifest())
        assert set(out)=={'volume_valid','signed_valid','ofi_valid','bbo_valid','order_count_valid',
                         'receive_clock_excluded','book_event_state_excluded'}
        expected_volume=[False,True,False,True,True,True,False,False,True,True]
        expected_signed=[False,True,False,True,True,False,False,False,True,True]
        expected_ofi=[False,False,False,False,False,True,True,False,False,True]
        np.testing.assert_array_equal(out['volume_valid'],expected_volume)
        np.testing.assert_array_equal(out['signed_valid'],expected_signed)
        np.testing.assert_array_equal(out['ofi_valid'],expected_ofi)
        np.testing.assert_array_equal(out['bbo_valid'],np.array(expected_ofi)&(covered==self.MINUTE))
        np.testing.assert_array_equal(out['order_count_valid'],np.array(expected_ofi)&(order==self.MINUTE))
        assert not out['bbo_valid'][6] and out['order_count_valid'][6]
        assert out['bbo_valid'][9] and not out['order_count_valid'][9]
        assert not out['receive_clock_excluded'].any() and not out['book_event_state_excluded'].any()
        assert all(x.dtype==np.dtype('bool') and not x.flags.writeable for x in out.values())

    def test_published_partial_minute_incident_excludes_through_first_clean_anchor(self):
        t=self._clock('2025-09-17T13:33:00',22);_,s,m,q,c,o=self._fixture(22,start=int(t[0]))
        q['bad_bbo'][18]=1;q['file_seam'][19]=1;s[20:]=1
        out=a.reference_masks134(t,s,m,q,c,o,self._manifest())
        expected_clock=np.zeros(22,bool);expected_clock[1:18]=True
        expected_state=np.zeros(22,bool);expected_state[1:21]=True
        np.testing.assert_array_equal(out['receive_clock_excluded'],expected_clock)
        np.testing.assert_array_equal(out['book_event_state_excluded'],expected_state)
        assert not out['ofi_valid'][20] and out['ofi_valid'][21]  # Anchor itself remains excluded.
        assert out['volume_valid'][18] and out['volume_valid'][19] and out['volume_valid'][20]

    def test_incident_without_later_clean_anchor_excludes_all_remaining_state(self):
        t=self._clock('2025-09-17T13:33:00',22);_,s,m,q,c,o=self._fixture(22,start=int(t[0]))
        q['bad_recv'][18:]=1
        out=a.reference_masks134(t,s,m,q,c,o,self._manifest())
        assert not out['book_event_state_excluded'][0] and out['book_event_state_excluded'][1:].all()
        assert out['receive_clock_excluded'].sum()==17 and not out['receive_clock_excluded'][18:].any()

    def test_incident_interval_is_half_open_and_recovery_anchor_is_inclusive(self):
        t=self._clock('2025-09-17T13:33:00',5);_,s,m,q,c,o=self._fixture(5,start=int(t[0]))
        out=a.reference_masks134(t,s,m,q,c,o,self._manifest('2025-09-17T13:34:00Z','2025-09-17T13:35:00Z'))
        np.testing.assert_array_equal(out['receive_clock_excluded'],[False,True,False,False,False])
        np.testing.assert_array_equal(out['book_event_state_excluded'],[False,True,True,False,False])
        np.testing.assert_array_equal(out['ofi_valid'],[False,False,False,True,True])

    def test_final_minute_strict_less_than_applies_to_trade_not_invented_book_gate(self):
        t,s,m,q,c,o=self._fixture(3,start=self.END-3*self.MINUTE)
        out=a.reference_masks134(t,s,m,q,c,o,self._manifest())
        np.testing.assert_array_equal(out['volume_valid'],[True,True,False])
        np.testing.assert_array_equal(out['signed_valid'],[True,True,False])
        np.testing.assert_array_equal(out['ofi_valid'],[False,True,True])
        np.testing.assert_array_equal(out['bbo_valid'],[False,True,True])
        np.testing.assert_array_equal(out['order_count_valid'],[False,True,True])

    @pytest.mark.parametrize('fault',['missing_restoration','reject_signature','deduplicate'])
    def test_manifest_cannot_silently_restore_another_admission(self,fault):
        t,s,m,q,c,o=self._fixture(3);manifest=self._manifest()
        if fault=='missing_restoration':manifest.pop('trusted_trade_admission')
        elif fault=='reject_signature':manifest['trusted_trade_admission']['reject_duplicate_signature']=True
        else:manifest['trusted_trade_admission']['deduplicate_T']=True
        with pytest.raises(RuntimeError,match='RAW_MASK134_RESTORED_INCIDENT_SCOPE'):
            a.reference_masks134(t,s,m,q,c,o,manifest)

    def test_gap_in_minute_axis_cannot_manufacture_a_previous_clean_anchor(self):
        t,s,m,q,c,o=self._fixture(3);t[1:]+=self.MINUTE
        with pytest.raises(RuntimeError,match='RAW_MASK134_M1_CLOCK'):
            a.reference_masks134(t,s,m,q,c,o,self._manifest())
