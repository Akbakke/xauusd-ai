"""Causal observed-volume roll and indicator input masks; no model features.

One genuine authority for the continuous GC input series. UTC days are not
exchange sessions. Volume means observed electronic T records, never orders,
settlement volume, OANDA volume or guaranteed complete historical flow.
"""
from __future__ import annotations

import gzip
from typing import Any
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.contracts.gc_order_flow_source_v1 import (
    UNDEF_PRICE, F_BAD_TS_RECV, F_MAYBE_BAD_BOOK, F_SNAPSHOT,
)

DAY_NS = 24 * timing.HOUR_NS
MINUTES_PER_DAY = DAY_NS // timing.MINUTE_NS
# Parser memory bound only; changes neither data selection nor indicator rules.
CSV_BLOCK_BYTES = 4 * 1024**2
METRICS = ('records', 'trades', 'volume', 'buy', 'sell', 'unknown',
           'invalid_trade', 'bad_recv_trade', 'gap', 'snapshot_trade', 'duplicate_trade')
COLUMNS = {
    'ts_recv': pa.uint64(), 'instrument_id': pa.uint32(),
    'publisher_id': pa.uint16(), 'symbol': pa.string(),
    'action': pa.string(), 'side': pa.string(), 'size': pa.uint32(),
    'price': pa.int64(), 'flags': pa.uint8(),
    'sequence': pa.uint32(), 'ts_event': pa.uint64(),
}


def scan_symbol(items: list[dict], tick_size: int, *, on_batch=None, extra_columns=None, on_file_start=None) -> dict[str, Any]:
    """One bounded Arrow stream per outright. Integer quantities remain exact."""
    width = (timing.WINDOW_END_NS - timing.STUDY_START_NS) // timing.MINUTE_NS
    out = {name: np.zeros(width, dtype=np.uint64) for name in METRICS}
    file_reports = []
    columns=dict(COLUMNS)
    if extra_columns is not None:columns.update(extra_columns)
    symbol = items[0]['raw_symbol']
    for item in sorted(items, key=lambda x: x['start_ns']):
        count = 0
        previous = None
        previous_signature = None
        file_bad_recv_trades = 0
        raw_trades = 0
        if on_file_start is not None:on_file_start(item)
        with gzip.open(item['path'], 'rb') as stream:
            reader = pacsv.open_csv(stream,
                read_options=pacsv.ReadOptions(use_threads=False, block_size=CSV_BLOCK_BYTES),
                convert_options=pacsv.ConvertOptions(column_types=columns, include_columns=list(columns)))
            for batch in reader:
                if any(batch.column(i).null_count for i in range(batch.num_columns)):
                    raise ValueError('GC_CONTINUOUS_NULL_INPUT')
                data = {name: batch.column(batch.schema.get_field_index(name)) for name in columns}
                recv = data['ts_recv'].to_numpy(zero_copy_only=False)
                count += len(recv)
                if (count > item['max_records'] or np.any(recv < item['start_ns'])
                        or np.any(recv >= item['end_ns']) or np.any(recv[1:] < recv[:-1])
                        or (previous is not None and int(recv[0]) < previous)):
                    raise ValueError('GC_CONTINUOUS_TIME_OR_BUDGET_INVALID')
                previous = int(recv[-1])
                for name, expected in (('symbol', symbol), ('instrument_id', item['instrument_id']),
                                       ('publisher_id', item['publisher_id'])):
                    if not pc.all(pc.equal(data[name], expected)).as_py():
                        raise ValueError('GC_CONTINUOUS_IDENTITY_INVALID')
                for name, allowed in (('action', timing.ALLOWED_ACTIONS), ('side', timing.ALLOWED_SIDES)):
                    if not pc.all(pc.is_in(data[name], value_set=pa.array(sorted(allowed)))).as_py():
                        raise ValueError('GC_CONTINUOUS_ACTION_OR_SIDE_INVALID')
                idx = (recv.astype(np.int64) - timing.STUDY_START_NS) // timing.MINUTE_NS
                if np.any(idx < 0) or np.any(idx >= width):
                    raise ValueError('GC_CONTINUOUS_TEST_OR_WINDOW_VIOLATION')
                trade = pc.equal(data['action'], 'T').to_numpy(zero_copy_only=False)
                raw_trades += int(trade.sum())
                side = {s: pc.equal(data['side'], s).to_numpy(zero_copy_only=False) for s in ('A','B','N')}
                flags = data['flags'].to_numpy(zero_copy_only=False)
                size = data['size'].to_numpy(zero_copy_only=False).astype(np.uint64)
                price = data['price'].to_numpy(zero_copy_only=False)
                bad_recv = (flags & F_BAD_TS_RECV) != 0
                file_bad_recv_trades += int(np.count_nonzero(trade & bad_recv))
                signature = [data[name].to_numpy(zero_copy_only=False) for name in
                             ('ts_recv','ts_event','sequence','action','side','price','size','flags')]
                duplicate = trade.copy()
                duplicate[0] &= previous_signature == tuple(a[0] for a in signature)
                for a in signature:
                    duplicate[1:] &= a[1:] == a[:-1]
                previous_signature = tuple(a[-1] for a in signature)
                invalid = trade & ((size == 0) | (size == timing.UNDEF_SIZE) |
                                   (price <= 0) | (price == UNDEF_PRICE) | (price % tick_size != 0))
                trusted_trade = trade & ~bad_recv & ~invalid & ((flags & F_SNAPSHOT) == 0)
                def add(name, take, weights=None):
                    np.add.at(out[name], idx[take], np.ones(int(take.sum()), dtype=np.uint64)
                              if weights is None else weights[take])
                add('records', ~bad_recv)
                add('trades', trusted_trade)
                add('volume', trusted_trade, size)
                for s,name in (('B','buy'),('A','sell'),('N','unknown')):
                    add(name, trusted_trade & side[s], size)
                add('invalid_trade', invalid)
                add('bad_recv_trade', trade & bad_recv)
                add('gap', (flags & F_MAYBE_BAD_BOOK) != 0)
                add('snapshot_trade', trade & ((flags & F_SNAPSHOT) != 0))
                add('duplicate_trade', duplicate)
                if on_batch is not None:
                    if extra_columns is not None:data['_duplicate_trade_signature']=pa.array(duplicate)
                    on_batch(data, idx, trusted_trade)
        if count == 0:
            raise ValueError('GC_CONTINUOUS_EMPTY_FILE')
        file_reports.append({'path': item['path'], 'records': count,
            'start_ns': item['start_ns'], 'end_ns': item['end_ns'],
            'bad_recv_trades': file_bad_recv_trades, 'raw_trade_records': raw_trades})
        print(f'GC_CONTINUOUS_FILE_COMPLETE {symbol} records={count}', flush=True)
    out['symbol'] = symbol
    out['files'] = file_reports
    return out


def choose_daily_contract(volumes: np.ndarray, maturities: list[tuple[int,int]]) -> tuple[np.ndarray,np.ndarray]:
    """Choose day d only from completed d-1 observed volumes; no backward roll.

    No trades/ties keep the previous identity. An initial tie means no choice.
    Zero current-day data never triggers a substitution. Returning to an earlier
    expiry is forbidden even if its observed volume subsequently increases.
    """
    symbols, days = volumes.shape
    if len(maturities) != symbols or len(set(maturities)) != symbols:
        raise ValueError('GC_CONTINUOUS_MATURITY_IDENTITY_INVALID')
    selected = np.full(days, -1, dtype=np.int16)
    basis_day = np.full(days, -1, dtype=np.int32)
    current = -1
    for day in range(1, days):
        candidates = [s for s in range(symbols) if volumes[s, day-1] > 0 and
                      (current < 0 or maturities[s] >= maturities[current])]
        if candidates:
            largest = max(int(volumes[s,day-1]) for s in candidates)
            winners = [s for s in candidates if int(volumes[s,day-1]) == largest]
            if len(winners) == 1:
                current = winners[0]
                basis_day[day] = day-1
        if basis_day[day] < 0:
            basis_day[day] = basis_day[day-1]
        selected[day] = current
    return selected, basis_day


def row_masks(selected: np.ndarray, values: dict[str,np.ndarray], eligible: np.ndarray) -> dict[str,np.ndarray]:
    base = ((selected >= 0) & eligible & (values['trades'] > 0) &
            (values['invalid_trade'] == 0) & (values['gap'] == 0) &
            (values['bad_recv_trade'] == 0) & (values['snapshot_trade'] == 0) &
            (values['duplicate_trade'] == 0))
    signed = base & (values['unknown'] == 0)
    return {'volume': base, 'volume_profile': base, 'footprint': signed,
            'delta': signed, 'cvd': signed}


def reset_segments(selected: np.ndarray, valid: np.ndarray, minute_ns: np.ndarray) -> np.ndarray:
    """Segment relative CVD/profile at every UTC date, roll, or invalid minute.

    No cross-segment continuation or exchange-session-complete claim is made.
    """
    segment = np.full(len(valid), -1, dtype=np.int64)
    counter = -1
    for i in range(len(valid)):
        if not valid[i]:
            continue
        if (i == 0 or not valid[i-1] or selected[i] != selected[i-1] or
                minute_ns[i] // DAY_NS != minute_ns[i-1] // DAY_NS or
                minute_ns[i] - minute_ns[i-1] != timing.MINUTE_NS):
            counter += 1
        segment[i] = counter
    return segment


def admission_decisions(months: np.ndarray, masks: dict[str,np.ndarray],
                        selected_bad_timestamp_trades: int) -> dict:
    """Structural admission only, with exact support in every fixed study month.

    No fitted coverage tolerance: a completely absent month fails. Any selected
    Trade with unreliable ts_recv rejects causal trade indicators globally;
    its true location cannot be repaired with a guessed time padding.
    """
    expected_months = np.arange(np.datetime64('2024-05'), np.datetime64('2026-07'), dtype='datetime64[M]')
    result = {}
    for name, mask in masks.items():
        counts = {str(m): int(np.count_nonzero(mask & (months == m))) for m in expected_months}
        reasons = []
        if selected_bad_timestamp_trades:
            reasons.append('SELECTED_TRADE_RECEIVE_CLOCK_UNRELIABLE_NO_PROVEN_LOCALIZATION')
        if any(v == 0 for v in counts.values()):
            reasons.append('NO_VALID_ROWS_IN_AT_LEAST_ONE_FIXED_STUDY_MONTH')
        result[name] = {'decision': 'REJECT' if reasons else 'ADMIT',
            'scope': 'masked observed-record inputs for a separately frozen offline feature builder only',
            'valid_m1_rows': int(mask.sum()), 'monthly_valid_m1_rows': counts, 'reasons': reasons,
            'complete_flow_or_predictive_value_proven': False}
    for name, reason in (
        ('sweeps','MBP1_HAS_NO_MULTI_LEVEL_OR_INDIVIDUAL_ORDER_HISTORY; price_sweep_proxy_requires_its_own_formula'),
        ('order_blocks','INSTITUTIONAL_ORDER_BLOCK_TRUTH_NOT_OBSERVED; causal_price_structure_requires_existing_SMC_owner')):
        result[name] = {'decision':'REJECT','reasons':[reason]}
    return result


# Optional L1 audit columns. The default trade parser/004 rules are unchanged.
L1_AUDIT_COLUMNS = {
    'rtype':pa.uint8(),'depth':pa.uint8(),'ts_in_delta':pa.int32(),
    'bid_px_00':pa.int64(),'ask_px_00':pa.int64(),
    'bid_sz_00':pa.uint32(),'ask_sz_00':pa.uint32(),
    'bid_ct_00':pa.uint32(),'ask_ct_00':pa.uint32(),
}
L1_QUALITY_METRICS=('records','clean_complete_bbo','bad_bbo','gap','bad_recv',
                    'snapshot','reset','duplicate_trade_signature',
                    'full_record_duplicate_trade','signature_collision','file_seam')

class L1QualityCollector:
    """Selected-contract, per-M1 quality only; does not compute a predictor.

    Full-record comparisons cover all 20 input fields, only for existing
    consecutive Trade-signature suspects. BBO snapshots are self-contained L1
    observations; candidate continuous windows need a clean prior-minute anchor.
    No claim of whole-feed sequence completeness or hidden order observability.
    """
    def __init__(self,wanted_minute_indices,tick_size):
        self.wanted=np.asarray(wanted_minute_indices,np.int64);self.tick_size=tick_size
        self.values={k:np.zeros(len(self.wanted),np.uint64) for k in L1_QUALITY_METRICS}
        self.previous_full=None;self.first_in_file=True
    def file_start(self,item):
        self.previous_full=None;self.first_in_file=True
    def __call__(self,data,idx,trusted_trade):
        names=[k for k in data if not k.startswith('_')]
        signature=data['_duplicate_trade_signature'].to_numpy(zero_copy_only=False)
        suspects=np.flatnonzero(signature);same=np.ones(len(suspects),bool)
        if len(suspects):
            for name in names:
                arr=data[name];current=arr.take(pa.array(suspects)).to_numpy(zero_copy_only=False)
                before=arr.take(pa.array(np.maximum(suspects-1,0))).to_numpy(zero_copy_only=False).copy()
                if suspects[0]==0:
                    if self.previous_full is None:same[0]=False
                    else:before[0]=self.previous_full[name]
                same &= current==before
        full=np.zeros(len(idx),bool);full[suspects]=same
        self.previous_full={name:data[name][-1].as_py() for name in names}
        pos=np.searchsorted(self.wanted,idx);clip=np.minimum(pos,max(0,len(self.wanted)-1))
        if not len(self.wanted):return
        selected=(pos<len(self.wanted))&(self.wanted[clip]==idx)
        if not selected.any():self.first_in_file=False;return
        def arr(name):return data[name].to_numpy(zero_copy_only=False)
        flags=arr('flags');bid=arr('bid_px_00');ask=arr('ask_px_00')
        bsz=arr('bid_sz_00');asz=arr('ask_sz_00')
        bad=((bid<=0)|(ask<=0)|(bid==UNDEF_PRICE)|(ask==UNDEF_PRICE)|(bid>=ask)|
             (bid%self.tick_size!=0)|(ask%self.tick_size!=0)|(bsz==0)|(asz==0)|
             (bsz==timing.UNDEF_SIZE)|(asz==timing.UNDEF_SIZE))
        gap=(flags&F_MAYBE_BAD_BOOK)!=0;recv=(flags&F_BAD_TS_RECV)!=0;snap=(flags&F_SNAPSHOT)!=0
        reset=pc.equal(data['action'],'R').to_numpy(zero_copy_only=False)
        complete=(flags&128)!=0
        clean=complete&~bad&~gap&~recv&~snap&~reset
        seam=np.zeros(len(idx),bool)
        if self.first_in_file:seam[0]=True
        metrics={'records':np.ones(len(idx),bool),'clean_complete_bbo':clean,'bad_bbo':bad,
          'gap':gap,'bad_recv':recv,'snapshot':snap,'reset':reset,
          'duplicate_trade_signature':signature,'full_record_duplicate_trade':full,
          'signature_collision':signature&~full,'file_seam':seam}
        for key,take in metrics.items():
            chosen = take & selected
            if chosen.any():
                unique,counts=np.unique(pos[chosen],return_counts=True)
                self.values[key][unique]+=counts.astype(np.uint64)
        self.first_in_file=False
