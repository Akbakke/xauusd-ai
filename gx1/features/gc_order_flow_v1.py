"""Offline observed GC trade features. No fit, native field registration or orders.

The continuous-source owner alone validates records/masks. This owner consumes
its admitted masks and receives only its validated Trade batches. Prices remain
unadjusted outright ticks; all quantities are exact observed contracts.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gx1.contracts import gc_continuous_source_v1 as source
from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.features.htf_features import MULTI_TF_SHIFT, multi_tf_bar_label
from gx1.features.smc_v1 import replay_volume_order_block_proxies_v1, VOLUME_ORDER_BLOCK_PROXY_FIELDS_V1

SCHEMA_VERSION = 'gc_observed_trade_features_v1'
# Explicit feature recipe inputs, fixed before any outcome access.
PROFILE_QUANTILES = (0.25, 0.5, 0.75)
PRICE_SCALE = 10**9
PROFILE_FIELDS = ('vwap_usd', 'poc_usd', 'q25_usd', 'median_usd', 'q75_usd', 'price_levels')
FOOTPRINT_FIELDS = ('max_level_delta_contracts', 'min_level_delta_contracts',
                    'buy_poc_usd', 'sell_poc_usd')
CVD_FIELDS = ('cvd_segment_delta_contracts', 'cvd_segment_volume_contracts', 'cvd_segment_minutes')


def histogram_features(ticks, buy, sell, unknown, tick_size):
    """Exact integer ladder reduction; volume quantiles use the lower crossing.

    POC ties choose the lowest price. Side POC is undefined when that side has
    no observed volume. No imbalance threshold or hidden-order inference.
    """
    ticks = np.asarray(ticks, dtype=np.int64)
    buy, sell, unknown = [np.asarray(a, dtype=np.uint64) for a in (buy, sell, unknown)]
    if not (ticks.ndim == 1 and len(ticks) and buy.shape == sell.shape == unknown.shape == ticks.shape):
        raise ValueError('GC_HISTOGRAM_SHAPE_INVALID')
    order = np.argsort(ticks, kind='stable')
    ticks, buy, sell, unknown = [a[order] for a in (ticks, buy, sell, unknown)]
    if np.any(np.diff(ticks) <= 0):
        raise ValueError('GC_HISTOGRAM_PRICES_NOT_UNIQUE')
    total = buy+sell+unknown
    volume = sum(map(int,total))
    if volume <= 0:
        raise ValueError('GC_HISTOGRAM_EMPTY_VOLUME')
    scale = tick_size/PRICE_SCALE
    cumulative = np.cumsum(total, dtype=np.uint64)
    out = {'volume_contracts': volume,
           'vwap_usd': sum(int(p)*int(v) for p,v in zip(ticks,total))/volume*scale,
           'poc_usd': float(ticks[np.argmax(total)])*scale,
           'price_levels': len(ticks)}
    # Exact rational thresholds avoid conversion of large contract counts to float.
    for name, denominator, numerator in (('q25_usd',4,1),('median_usd',2,1),('q75_usd',4,3)):
        threshold = (volume*numerator+denominator-1)//denominator
        out[name] = float(ticks[np.searchsorted(cumulative,threshold)])*scale
    delta = [int(b)-int(s) for b,s in zip(buy,sell)]
    out.update(max_level_delta_contracts=max(delta), min_level_delta_contracts=min(delta))
    for side, quantities in (('buy',buy),('sell',sell)):
        out[side+'_poc_usd'] = float(ticks[np.argmax(quantities)])*scale if np.any(quantities) else np.nan
    return out


class TradeLadderCollector:
    """A streaming callback of the source owner, never a second record parser."""
    def __init__(self, path: Path, selected, valid, symbol_index, tick_size, *, reference_bars=()):
        self.selected, self.valid = selected, valid
        self.symbol_index, self.tick_size = symbol_index, tick_size
        self.ohlc = np.zeros((len(selected),4),dtype=np.int64)
        self.writer = pq.ParquetWriter(path, pa.schema([
            ('minute_index',pa.int32()),('price_tick',pa.int64()),
            ('buy',pa.uint64()),('sell',pa.uint64()),('unknown',pa.uint64())]), compression='zstd')
        self.rows = 0
        self.reference_bars = frozenset(reference_bars)
        self.reference_trades = {}

    def __call__(self, data, idx, trusted_trade):
        take = trusted_trade & self.valid[idx] & (self.selected[idx] == self.symbol_index)
        if not np.any(take):
            return
        minutes = idx[take]
        ticks = data['price'].to_numpy(zero_copy_only=False)[take]//self.tick_size
        size = data['size'].to_numpy(zero_copy_only=False)[take].astype(np.uint64)
        side = data['side'].to_numpy(zero_copy_only=False)[take]
        for bar in self.reference_bars.intersection(set((minutes//5).tolist())):
            refs = self.reference_trades.setdefault(int(bar),[])
            for j in np.flatnonzero(minutes//5==bar):
                refs.append((int(ticks[j]),int(size[j]),str(side[j])))
        unique, first, counts = np.unique(minutes,return_index=True,return_counts=True)
        empty = self.ohlc[unique,0] == 0
        self.ohlc[unique[empty],0] = ticks[first[empty]]
        self.ohlc[unique,3] = ticks[first+counts-1]
        lo = np.minimum.reduceat(ticks,first)
        hi = np.maximum.reduceat(ticks,first)
        self.ohlc[unique,1] = np.maximum(self.ohlc[unique,1],hi)
        self.ohlc[unique,2] = np.where(empty,lo,np.minimum(self.ohlc[unique,2],lo))
        keys = np.rec.fromarrays([minutes.astype(np.int32),ticks],names='minute,price')
        levels, inverse = np.unique(keys,return_inverse=True)
        quantities = {}
        for value,name in (('B','buy'),('A','sell'),('N','unknown')):
            a = np.zeros(len(levels),dtype=np.uint64)
            np.add.at(a,inverse[side==value],size[side==value])
            quantities[name] = a
        self.writer.write_table(pa.table({'minute_index':levels['minute'], 'price_tick':levels['price'], **quantities}))
        self.rows += int(take.sum())

    def close(self):
        self.writer.close()


def timeframe_skeleton(minutes, selected, metrics, masks, ohlc, timeframe, tick_size):
    labels = multi_tf_bar_label(pd.DatetimeIndex(minutes,tz='UTC'),timeframe).asi8
    stamps, first, inverse, counts = np.unique(labels,return_index=True,return_inverse=True,return_counts=True)
    duration = MULTI_TF_SHIFT[timeframe].value
    expected = duration//timing.MINUTE_NS
    valid = (counts==expected) & np.logical_and.reduceat(masks['volume'],first)
    valid &= np.minimum.reduceat(selected,first)==np.maximum.reduceat(selected,first)
    signed = valid & np.logical_and.reduceat(masks['delta'],first)
    out = {'bar_start_ns':stamps, 'available_ns':stamps+duration,
           'contract_index':selected[first], 'volume_valid':valid, 'signed_valid':signed}
    for name in ('volume','buy','sell','unknown','trades'):
        out[name] = np.add.reduceat(metrics[name],first,dtype=np.uint64)
    scale = tick_size/PRICE_SCALE
    out['open_usd'] = ohlc[first,0]*scale
    out['high_usd'] = np.maximum.reduceat(ohlc[:,1],first)*scale
    out['low_usd'] = np.minimum.reduceat(ohlc[:,2],first)*scale
    out['close_usd'] = ohlc[first+counts-1,3]*scale
    for name in PROFILE_FIELDS+FOOTPRINT_FIELDS+CVD_FIELDS:
        out[name] = np.full(len(stamps),np.nan)
    out['delta_contracts'] = out['buy'].astype(np.int64)-out['sell'].astype(np.int64)
    out['known_aggressor_fraction'] = np.divide(out['buy']+out['sell'],out['volume'],
        out=np.full(len(stamps),np.nan),where=out['volume']>0)
    return out,inverse


def consume_ladder(path: Path, surfaces, minute_maps, tick_size):
    """Reduce bounded Arrow batches into one currently open histogram per TF."""
    states = {tf:None for tf in surfaces}
    def flush(tf):
        state = states[tf]
        if state is None:
            return
        bar,histogram = state
        ticks = sorted(histogram)
        a = np.array([histogram[p] for p in ticks],dtype=np.uint64)
        values = histogram_features(ticks,a[:,0],a[:,1],a[:,2],tick_size)
        surface = surfaces[tf]
        if np.isfinite(surface['vwap_usd'][bar]):
            raise RuntimeError('GC_FEATURE_PROFILE_MULTIPLE_CONTRACTS')
        if values['volume_contracts'] != int(surface['volume'][bar]):
            raise RuntimeError('GC_FEATURE_LADDER_VOLUME_PARITY_FAILED')
        for name in PROFILE_FIELDS:
            surface[name][bar] = values[name]
        if surface['signed_valid'][bar]:
            for name in FOOTPRINT_FIELDS:
                surface[name][bar] = values[name]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        a = {n:batch.column(batch.schema.get_field_index(n)).to_numpy() for n in batch.schema.names}
        for tf,surface in surfaces.items():
            bars = minute_maps[tf][a['minute_index']]
            take = surface['volume_valid'][bars]
            if not np.any(take):
                continue
            bars,ticks = bars[take],a['price_tick'][take]
            quantities = np.column_stack([a[n][take] for n in ('buy','sell','unknown')])
            unique,first,counts = np.unique(bars,return_index=True,return_counts=True)
            for bar,start,count in zip(unique,first,counts):
                if states[tf] is None or states[tf][0] != bar:
                    flush(tf)
                    states[tf] = (int(bar),{})
                hist = states[tf][1]
                levels,inv = np.unique(ticks[start:start+count],return_inverse=True)
                sums = np.zeros((len(levels),3),dtype=np.uint64)
                np.add.at(sums,inv,quantities[start:start+count])
                for price,qty in zip(levels,sums):
                    key = int(price)
                    if key in hist:
                        hist[key] += qty
                    else:
                        hist[key] = qty.copy()
    for tf in surfaces:
        flush(tf)


def add_structure_and_cvd(surface, minute_cvd, minute_maps, minutes):
    valid = surface['volume_valid']
    labels = surface['bar_start_ns']
    duration = int(surface['available_ns'][0]-labels[0])
    split = np.r_[True, ~valid[:-1] | ~valid[1:] |
        (np.diff(labels)!=duration) | (np.diff(surface['contract_index'])!=0)]
    for name in VOLUME_ORDER_BLOCK_PROXY_FIELDS_V1:
        surface[name] = np.full(len(valid),np.nan)
    starts = np.flatnonzero(valid & split)
    for begin in starts:
        end = begin+1
        while end < len(valid) and valid[end] and not split[end]:
            end += 1
        data = replay_volume_order_block_proxies_v1(*[surface[n][begin:end] for n in
            ('open_usd','high_usd','low_usd','close_usd','volume')])
        for name,values in data.items():
            if name not in surface:
                surface[name] = np.full(len(valid),np.nan)
            surface[name][begin:end] = values
    end_index = np.searchsorted(minutes,surface['available_ns']-timing.MINUTE_NS)
    inside = end_index<len(minutes)
    for name,values in minute_cvd.items():
        take = inside & surface['signed_valid']
        surface[name][take] = values[end_index[take]]
    # Invalid observations stay missing, never neutral flow or inactive structure.
    unsigned = ('volume','trades','unknown','known_aggressor_fraction','open_usd','high_usd','low_usd','close_usd')+PROFILE_FIELDS
    signed = ('buy','sell','delta_contracts')+FOOTPRINT_FIELDS+CVD_FIELDS
    if np.any(surface['volume'] > 2**53):
        raise RuntimeError('GC_FEATURE_FLOAT64_QUANTITY_EXACTNESS_BOUND_EXCEEDED')
    for names,mask in ((unsigned,valid),(signed,surface['signed_valid'])):
        for name in names:
            value = surface[name].astype(np.float64)
            value[~mask] = np.nan
            surface[name] = value


def relative_cvd(minutes, selected, masks, metrics):
    segments = source.reset_segments(selected,masks['cvd'],minutes)
    out = {name:np.full(len(minutes),np.nan) for name in CVD_FIELDS}
    previous = -1
    delta = volume = count = 0
    for i in np.flatnonzero(masks['cvd']):
        if segments[i] != previous:
            delta = volume = count = 0
        previous = segments[i]
        delta += int(metrics['buy'][i])-int(metrics['sell'][i])
        volume += int(metrics['volume'][i])
        count += 1
        for name,value in zip(CVD_FIELDS,(delta,volume,count)):
            out[name][i] = value
    return out
