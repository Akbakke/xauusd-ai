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
OBSERVED_FOOTPRINT_FIELDS = {
    'max_level_delta_contracts': 'observed_known_max_level_delta_contracts',
    'min_level_delta_contracts': 'observed_known_min_level_delta_contracts',
    'buy_poc_usd': 'observed_buy_known_poc_usd',
    'sell_poc_usd': 'observed_sell_known_poc_usd',
}


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
    def __init__(self, path: Path | None, selected, valid, symbol_index, tick_size, *, reference_bars=()):
        self.selected, self.valid = selected, valid
        self.symbol_index, self.tick_size = symbol_index, tick_size
        self.ohlc = np.zeros((len(selected),4),dtype=np.int64)
        self.writer = None if path is None else pq.ParquetWriter(path, pa.schema([
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
        if self.writer is None:
            self.rows += int(take.sum())
            return
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
        if self.writer is not None:
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
    # Preserve the strict complete-clock domain above. The separate observed
    # domain describes only admitted M1 bytes, never a complete exchange tape.
    # Expected minutes are the fixed UTC bucket duration, not a guessed calendar.
    accepted = np.asarray(masks['volume'], bool)
    accepted_count = np.add.reduceat(accepted.astype(np.int64), first)
    signed_count = np.add.reduceat((accepted & masks['delta']).astype(np.int64), first)
    contract_valid = (np.minimum.reduceat(selected, first) >= 0)
    contract_valid &= np.minimum.reduceat(selected, first) == np.maximum.reduceat(selected, first)
    observed = (counts == expected) & contract_valid & (accepted_count > 0)
    out.update(observed_valid=observed, observed_contract_valid=contract_valid,
        observed_clock_minutes=counts, observed_expected_clock_minutes=np.full(len(stamps), expected, np.int64),
        observed_accepted_minutes=accepted_count, observed_signed_minutes=signed_count,
        observed_coverage_fraction=accepted_count / expected,
        observed_signed_minute_fraction=np.divide(signed_count, accepted_count,
            out=np.full(len(stamps), np.nan), where=accepted_count > 0))
    for source_name, target in (('volume', 'volume'), ('buy', 'buy_known'),
            ('sell', 'sell_known'), ('unknown', 'unknown'), ('trades', 'trades')):
        quantities = np.add.reduceat(np.where(accepted, metrics[source_name], 0), first, dtype=np.uint64)
        if np.any(quantities > 2**53):
            raise RuntimeError('GC_FEATURE_FLOAT64_QUANTITY_EXACTNESS_BOUND_EXCEEDED')
        value = quantities.astype(np.float64)
        value[~observed] = np.nan
        out['observed_' + target] = value
    out['observed_known_delta_contracts'] = out['observed_buy_known'] - out['observed_sell_known']
    out['observed_known_aggressor_fraction'] = np.divide(
        out['observed_buy_known'] + out['observed_sell_known'], out['observed_volume'],
        out=np.full(len(stamps), np.nan), where=out['observed_volume'] > 0)
    out['observed_signed_complete'] = observed & (out['observed_unknown'] == 0)
    positions = np.arange(len(minutes), dtype=np.int64)
    opening = np.minimum.reduceat(np.where(accepted, positions, len(minutes)), first)
    closing = np.maximum.reduceat(np.where(accepted, positions, -1), first)
    opening = np.minimum(opening, len(minutes) - 1)
    closing = np.maximum(closing, 0)
    observed_prices = {
        'open_usd': ohlc[opening, 0] * scale,
        'high_usd': np.maximum.reduceat(np.where(accepted, ohlc[:, 1], 0), first) * scale,
        'low_usd': np.minimum.reduceat(np.where(accepted, ohlc[:, 2], np.iinfo(np.int64).max), first) * scale,
        'close_usd': ohlc[closing, 3] * scale,
    }
    for name, values in observed_prices.items():
        out['observed_' + name] = np.where(observed, values, np.nan)
    # Minute-grid nanoseconds in this source are exactly representable as
    # float64; NaN preserves absence without inventing a timestamp sentinel.
    for name, index in (('first', opening), ('last', closing)):
        timestamps = minutes[index].astype(np.int64)
        encoded = timestamps.astype(np.float64)
        if not np.array_equal(encoded.astype(np.int64), timestamps):
            raise RuntimeError('GC_FEATURE_OBSERVED_TIMESTAMP_PRECISION')
        out['observed_' + name + '_minute_ns'] = np.where(observed, encoded, np.nan)
    for name in PROFILE_FIELDS:
        out['observed_' + name] = np.full(len(stamps), np.nan)
    for name in OBSERVED_FOOTPRINT_FIELDS.values():
        out[name] = np.full(len(stamps), np.nan)
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
        if surface['observed_valid'][bar]:
            if np.isfinite(surface['observed_vwap_usd'][bar]):
                raise RuntimeError('GC_FEATURE_PROFILE_MULTIPLE_CONTRACTS')
            for j, name in enumerate(('buy_known', 'sell_known', 'unknown')):
                if sum(map(int, a[:, j])) != int(surface['observed_' + name][bar]):
                    raise RuntimeError('GC_FEATURE_OBSERVED_SIDE_VOLUME_PARITY_FAILED:' + name)
            if values['volume_contracts'] != int(surface['observed_volume'][bar]):
                raise RuntimeError('GC_FEATURE_OBSERVED_LADDER_VOLUME_PARITY_FAILED')
            for name in PROFILE_FIELDS:
                surface['observed_' + name][bar] = values[name]
            # These are the KNOWN-side contributions, even when unknown volume
            # exists. They never assert the sign of the unobserved aggressors.
            for source_name, target in OBSERVED_FOOTPRINT_FIELDS.items():
                surface[target][bar] = values[source_name]
        if surface['volume_valid'][bar]:
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
            take = (surface['volume_valid'] | surface['observed_valid'])[bars]
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


def add_observed_structure(surface):
    """Existing SMC formulas on admitted observed candles, with coverage intact.

    This is an observed-candle proxy. Missing intrabar minutes do not reveal
    their price path. No institutional order identity or complete-market claim
    follows. Missing candles, contract changes and clock gaps reset the proxy.
    """
    valid = np.asarray(surface['observed_valid'], bool)
    labels = np.asarray(surface['bar_start_ns'], np.int64)
    available = np.asarray(surface['available_ns'], np.int64)
    if not len(valid) or np.any(available - labels != available[0] - labels[0]):
        raise ValueError('GC_OBSERVED_STRUCTURE_CLOCK')
    duration = int(available[0] - labels[0])
    first_observed = np.asarray(surface['observed_first_minute_ns'], np.float64)
    last_observed = np.asarray(surface['observed_last_minute_ns'], np.float64)
    boundary_gap = np.full(len(valid), np.nan)
    adjacent = valid[1:] & valid[:-1]
    boundary_gap[1:] = np.where(adjacent,
        np.maximum((first_observed[1:] - last_observed[:-1]) / timing.MINUTE_NS - 1, 0), np.nan)
    split = np.r_[True, ~valid[:-1] | ~valid[1:] |
        (np.diff(labels) != duration) | (np.diff(surface['contract_index']) != 0) |
        (boundary_gap[1:] > 0)]
    surface['observed_proxy_boundary_gap_minutes'] = boundary_gap
    for name in VOLUME_ORDER_BLOCK_PROXY_FIELDS_V1:
        surface['observed_proxy_' + name] = np.full(len(valid), np.nan)
    for begin in np.flatnonzero(valid & split):
        end = begin + 1
        while end < len(valid) and valid[end] and not split[end]:
            end += 1
        values = [np.asarray(surface['observed_' + name])[begin:end] for name in
            ('open_usd', 'high_usd', 'low_usd', 'close_usd', 'volume')]
        if not all(np.isfinite(value).all() for value in values):
            raise RuntimeError('GC_OBSERVED_STRUCTURE_NONFINITE')
        data = replay_volume_order_block_proxies_v1(*values)
        for name, value in data.items():
            surface['observed_proxy_' + name][begin:end] = value
    return surface


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

# Research v2 adds families without changing any v1 formula or native registry.
L1_FIELDS_V2 = ('ofi_contracts','bid_refill_contracts','ask_refill_contracts',
    'bid_depletion_contracts','ask_depletion_contracts','quote_updates',
    'tw_imbalance','tw_spread_ticks','tw_bid_contracts','tw_ask_contracts',
    'tw_microprice_offset_ticks','mid_response_ticks')
L1_ORDER_COUNT_FIELDS_V1 = ('tw_bid_orders', 'tw_ask_orders', 'tw_order_count_imbalance',
    'tw_bid_contracts_per_order', 'tw_ask_contracts_per_order')
VALUE_AREA_FRACTION_V2 = (7,10)  # preregistered 70%, never fitted to outcomes


def aggregate_observed_book_v1(minutes, selected, book, timeframe):
    """Aggregate already-qualified M1 book bytes, without a raw-event replay.

    Additive flow uses its own OFI mask. BBO and displayed-order means use
    their distinct admitted covered_ns weights. The M1 source discarded
    partial-minute means, so those cannot be reconstructed here. Quote response
    joins the first and last admitted BBO states and exposes their M1 timestamps;
    it is not a return over a fully observed interval when coverage is partial.
    """
    minutes = np.asarray(minutes, np.int64)
    selected = np.asarray(selected, np.int64)
    if (minutes.ndim != 1 or not len(minutes) or selected.shape != minutes.shape or
            np.any(np.diff(minutes) <= 0) or np.any(minutes % timing.MINUTE_NS)):
        raise ValueError('GC_OBSERVED_BOOK_MINUTE_CLOCK')
    required = ('ofi_valid', 'bbo_valid', 'covered_ns', 'order_count_valid',
        'order_covered_ns', 'opening_mid_tick', 'closing_mid_tick') + L1_FIELDS_V2 + L1_ORDER_COUNT_FIELDS_V1
    arrays = {name: np.asarray(book[name]) for name in required}
    if any(value.shape != minutes.shape for value in arrays.values()):
        raise ValueError('GC_OBSERVED_BOOK_SHAPE')
    for name in ('ofi_valid', 'bbo_valid', 'order_count_valid'):
        if not np.isin(arrays[name], (0, 1)).all():
            raise ValueError('GC_OBSERVED_BOOK_MASK')
    covered, order_covered = arrays['covered_ns'], arrays['order_covered_ns']
    if (not np.isfinite(covered).all() or not np.isfinite(order_covered).all() or
            np.any(covered < 0) or np.any(covered > timing.MINUTE_NS) or
            np.any(order_covered < 0) or np.any(order_covered > covered) or
            np.any(covered != covered.astype(np.int64)) or
            np.any(order_covered != order_covered.astype(np.int64))):
        raise ValueError('GC_OBSERVED_BOOK_COVERAGE')
    flow = arrays['ofi_valid'].astype(bool)
    bbo = arrays['bbo_valid'].astype(bool)
    orders = arrays['order_count_valid'].astype(bool)
    if np.any(bbo & (covered != timing.MINUTE_NS)) or np.any(orders & (order_covered != timing.MINUTE_NS)):
        raise ValueError('GC_OBSERVED_BOOK_PARTIAL_MEAN_UNAVAILABLE')
    if np.any(bbo & ~flow) or np.any(orders & ~bbo):
        raise ValueError('GC_OBSERVED_BOOK_MASK_DEPENDENCY')
    for names, mask in ((L1_FIELDS_V2[:6], flow),
            (L1_FIELDS_V2[6:11] + ('opening_mid_tick', 'closing_mid_tick'), bbo),
            (L1_ORDER_COUNT_FIELDS_V1, orders)):
        if any(not np.isfinite(arrays[name][mask]).all() for name in names):
            raise ValueError('GC_OBSERVED_BOOK_QUALIFIED_NONFINITE')
    labels = multi_tf_bar_label(pd.DatetimeIndex(minutes, tz='UTC'), timeframe).asi8
    stamps, first, counts = np.unique(labels, return_index=True, return_counts=True)
    duration = MULTI_TF_SHIFT[timeframe].value
    expected = duration // timing.MINUTE_NS
    same = (np.minimum.reduceat(selected, first) >= 0)
    same &= np.minimum.reduceat(selected, first) == np.maximum.reduceat(selected, first)
    bucket = (counts == expected) & same
    out = {'bar_start_ns': stamps, 'available_ns': stamps + duration,
        'contract_index': selected[first], 'observed_book_contract_valid': same,
        'observed_book_clock_minutes': counts,
        'observed_book_expected_clock_minutes': np.full(len(stamps), expected, np.int64)}
    flow_count = np.add.reduceat(flow.astype(np.int64), first)
    bbo_count = np.add.reduceat(bbo.astype(np.int64), first)
    order_count = np.add.reduceat(orders.astype(np.int64), first)
    bbo_time = np.add.reduceat(np.where(bbo, covered, 0).astype(np.int64), first)
    order_time = np.add.reduceat(np.where(orders, order_covered, 0).astype(np.int64), first)
    flow_valid = bucket & (flow_count > 0)
    bbo_valid = bucket & (bbo_time > 0)
    order_valid = bucket & (order_time > 0)
    out.update(observed_book_flow_valid=flow_valid,
        observed_book_flow_accepted_minutes=flow_count,
        observed_book_flow_coverage_fraction=flow_count / expected,
        observed_book_valid=bbo_valid, observed_book_accepted_minutes=bbo_count,
        observed_book_covered_ns=bbo_time, observed_book_coverage_fraction=bbo_time / duration,
        observed_order_valid=order_valid, observed_order_accepted_minutes=order_count,
        observed_order_covered_ns=order_time, observed_order_coverage_fraction=order_time / duration)
    for name in L1_FIELDS_V2[:6]:
        value = np.add.reduceat(np.where(flow, arrays[name], 0.0), first)
        out['observed_book_' + name] = np.where(flow_valid, value, np.nan)
    for names, mask, weights, total, valid, prefix in (
            (L1_FIELDS_V2[6:11], bbo, covered, bbo_time, bbo_valid, 'observed_book_'),
            (L1_ORDER_COUNT_FIELDS_V1, orders, order_covered, order_time, order_valid, 'observed_order_')):
        for name in names:
            weighted = np.add.reduceat(np.where(mask, arrays[name], 0.0) * np.where(mask, weights, 0), first)
            value = np.divide(weighted, total, out=np.full(len(stamps), np.nan), where=total > 0)
            out[prefix + name] = np.where(valid, value, np.nan)
    positions = np.arange(len(minutes), dtype=np.int64)
    opening = np.minimum.reduceat(np.where(bbo, positions, len(minutes)), first)
    closing = np.maximum.reduceat(np.where(bbo, positions, -1), first)
    opening = np.minimum(opening, len(minutes) - 1)
    closing = np.maximum(closing, 0)
    opened = arrays['opening_mid_tick'][opening]
    closed = arrays['closing_mid_tick'][closing]
    out['observed_book_opening_mid_tick'] = np.where(bbo_valid, opened, np.nan)
    out['observed_book_closing_mid_tick'] = np.where(bbo_valid, closed, np.nan)
    out['observed_book_mid_response_ticks'] = np.where(bbo_valid, closed - opened, np.nan)
    span = (minutes[closing] - minutes[opening]) // timing.MINUTE_NS + 1
    out['observed_book_response_span_minutes'] = np.where(bbo_valid, span, np.nan)
    out['observed_book_response_contiguous'] = bbo_valid & (bbo_count == span)
    for name, index in (('first', opening), ('last', closing)):
        timestamps = minutes[index]
        encoded = timestamps.astype(np.float64)
        if not np.array_equal(encoded.astype(np.int64), timestamps):
            raise RuntimeError('GC_FEATURE_OBSERVED_TIMESTAMP_PRECISION')
        out['observed_book_' + name + '_minute_ns'] = np.where(bbo_valid, encoded, np.nan)
    return out


def bbo_transition_v2(previous, current):
    """Cont et al. L1 OFI; refill/depletion only at unchanged best price.

    Columns are receive ns, bid/ask integer ticks, bid/ask contracts. A size
    change is a best-quote proxy, not an identified cancellation or order.
    """
    p,c = np.asarray(previous),np.asarray(current)
    b,a,q,r = c[:,1],c[:,2],c[:,3],c[:,4]
    B,A,Q,R = p[:,1],p[:,2],p[:,3],p[:,4]
    ofi=(b>=B)*q-(b<=B)*Q-(a<=A)*r+(a>=A)*R
    return np.column_stack((ofi,(b==B)*np.maximum(q-Q,0),
        (a==A)*np.maximum(r-R,0),(b==B)*np.maximum(Q-q,0),
        (a==A)*np.maximum(R-r,0),np.any(c[:,1:]!=p[:,1:],axis=1)))


class L1FeatureCollectorV2:
    """Streaming final-event L1 reducer, piecewise constant on ts_recv.

    Integrals use the state strictly before each interval's right boundary.
    All values are later gated by the unchanged 012 clean-prior-minute mask.
    A file seam clears the anchor. Missing interior minutes are not filled.
    """
    def __init__(self,wanted,tick_size,*,origin_ns=timing.STUDY_START_NS,reference_minutes=(),include_order_counts=False):
        self.wanted=np.asarray(wanted,np.int64);self.tick_size=tick_size
        self.origin_ns=origin_ns;self.previous=None;self.end_ns=None
        self.sums=np.zeros((len(wanted),12),np.float64)
        self.covered=np.zeros(len(wanted),np.int64)
        self.first=np.full(len(wanted),np.nan);self.last=self.first.copy()
        self.first_time=np.full(len(wanted),np.iinfo(np.int64).max,np.int64)
        self.last_time=np.full(len(wanted),np.iinfo(np.int64).min,np.int64)
        self.reference_minutes=frozenset(reference_minutes);self.references=[]
        self.include_order_counts=include_order_counts
        if include_order_counts:
            self.order_sums=np.zeros((len(wanted),len(L1_ORDER_COUNT_FIELDS_V1)),np.float64)
            self.order_covered=np.zeros(len(wanted),np.int64)
    def file_start(self,item):
        if self.previous is not None:self.finish(self.end_ns)
        self.previous=None;self.end_ns=item['end_ns']
    def _positions(self,minute):
        pos=np.searchsorted(self.wanted,minute)
        take=pos<len(self.wanted)
        take[take] &= self.wanted[pos[take]]==minute[take]
        return pos,take
    def _integrate(self,p,left,right):
        minute=(left-self.origin_ns)//timing.MINUTE_NS
        pos,take=self._positions(minute)
        take &= (right>left)&(p[:,1]>0)&(p[:,2]>p[:,1])&(p[:,3]>0)&(p[:,4]>0)
        if not take.any():return
        p=p[take];pos=pos[take];duration=right[take]-left[take]
        q,r=p[:,3].astype(float),p[:,4].astype(float)
        mid=(p[:,1]+p[:,2])/2
        imbalance=(q-r)/(q+r)
        vals=np.column_stack((imbalance,p[:,2]-p[:,1],q,r,
             (p[:,2]-p[:,1])*imbalance/2))
        np.add.at(self.covered,pos,duration)
        for j in range(5):np.add.at(self.sums[:,6+j],pos,vals[:,j]*duration)
        if self.include_order_counts:
            bc,ac=p[:,5],p[:,6]
            observed=(bc>0)&(ac>0)&(bc<timing.UNDEF_SIZE)&(ac<timing.UNDEF_SIZE)
            qty=np.column_stack((bc,ac,(bc-ac)/(bc.astype(float)+ac),
                np.divide(q,bc,out=np.full(len(q),np.nan),where=bc>0),
                np.divide(r,ac,out=np.full(len(r),np.nan),where=ac>0)))
            np.add.at(self.order_covered,pos[observed],duration[observed])
            for j in range(len(L1_ORDER_COUNT_FIELDS_V1)):
                np.add.at(self.order_sums[:,j],pos[observed],qty[observed,j]*duration[observed])
        unique,first=np.unique(pos,return_index=True)
        earliest=left[take][first];replace=earliest<self.first_time[unique]
        self.first[unique[replace]]=mid[first[replace]];self.first_time[unique[replace]]=earliest[replace]
        last=np.r_[first[1:]-1,len(pos)-1];latest=right[take][last]
        replace=latest>self.last_time[unique]
        self.last[unique[replace]]=mid[last[replace]];self.last_time[unique[replace]]=latest[replace]
    def consume(self,events):
        c=np.asarray(events,np.int64)
        if not len(c):return
        p=np.vstack((c[0] if self.previous is None else self.previous,c[:-1]))
        transition=bbo_transition_v2(p[:,:5],c[:,:5])
        minute=(c[:,0]-self.origin_ns)//timing.MINUTE_NS
        pos,take=self._positions(minute)
        anchor=np.ones(len(c),bool)
        if self.previous is None:anchor[0]=False
        take &= anchor&(p[:,1]>0)&(p[:,2]>p[:,1])&(c[:,1]>0)&(c[:,2]>c[:,1])
        for j in range(6):np.add.at(self.sums[:,j],pos[take],transition[take,j])
        left=p[:,0].copy();right=c[:,0]
        active=anchor&(right>=left)
        # At most the two boundary minutes are integrated across a long gap.
        # Any unobserved middle minute keeps coverage zero and cannot qualify.
        boundary=self.origin_ns+(((left-self.origin_ns)//timing.MINUTE_NS)+1)*timing.MINUTE_NS
        self._integrate(p[active],left[active],np.minimum(right,boundary)[active])
        cross=active&(right>boundary)
        if cross.any():
            endstart=self.origin_ns+((right-self.origin_ns)//timing.MINUTE_NS)*timing.MINUTE_NS
            second=np.maximum(boundary,endstart)
            self._integrate(p[cross],second[cross],right[cross])
        self.previous=c[-1].copy()
    def __call__(self,data,idx,trusted_trade):
        flags=data['flags'].to_numpy(zero_copy_only=False)
        take=(flags&source.L1_F_LAST)!=0
        cols=[data[n].to_numpy(zero_copy_only=False)[take].astype(np.int64) for n in
          ('ts_recv','bid_px_00','ask_px_00','bid_sz_00','ask_sz_00')]
        cols[1]//=self.tick_size;cols[2]//=self.tick_size
        if self.include_order_counts:
            cols.extend(data[n].to_numpy(zero_copy_only=False)[take].astype(np.int64)
                        for n in ('bid_ct_00','ask_ct_00'))
        c=np.column_stack(cols)
        if self.reference_minutes and len(c):
            m=(c[:,0]-self.origin_ns)//timing.MINUTE_NS
            ref=np.isin(m,list(self.reference_minutes))
            if ref.any():self.references.append(c[ref].copy())
        self.consume(c)
    def finish(self,end_ns=None):
        if self.previous is None:return
        end=self.end_ns if end_ns is None else end_ns
        if end<self.previous[0]:raise ValueError('L1_FINISH_BEFORE_LAST_EVENT')
        p=self.previous[None,:];left=p[:,0];right=np.array([end],np.int64)
        boundary=self.origin_ns+(((left-self.origin_ns)//timing.MINUTE_NS)+1)*timing.MINUTE_NS
        self._integrate(p,left,np.minimum(right,boundary))
        # No state carry to an unobserved later minute at EOF/prefix boundary.
        self.previous=self.previous.copy();self.previous[0]=end
    def values(self,candidate):
        mask=np.asarray(candidate,bool)&(self.covered==timing.MINUTE_NS)
        out={}
        for j,name in enumerate(L1_FIELDS_V2):
            if j<6:v=self.sums[:,j].copy()
            elif j<11:v=self.sums[:,j]/timing.MINUTE_NS
            else:v=self.last-self.first
            v[~(np.asarray(candidate,bool) if j<6 else mask)]=np.nan;out[name]=v
        return out,mask


    def order_count_values(self,candidate):
        """Observed BBO order counts, isolated from the unchanged 12-field book surface."""
        if not self.include_order_counts:
            raise RuntimeError('L1_ORDER_COUNTS_NOT_REQUESTED')
        valid=np.asarray(candidate,bool)&(self.order_covered==timing.MINUTE_NS)
        out={}
        for j,name in enumerate(L1_ORDER_COUNT_FIELDS_V1):
            values=self.order_sums[:,j]/timing.MINUTE_NS
            values[~valid]=np.nan
            out[name]=values
        return out,valid


def value_area_v2(histogram):
    """Contiguous 70% area grown from POC; equal neighbor volume chooses low.

    Empty intervening ticks belong to the price interval, not fake volume.
    This is not a volume quantile interval. Returns integer POC, VAL, VAH.
    """
    if not histogram or any(int(v)<=0 for v in histogram.values()):raise ValueError('VA_EMPTY_OR_NONPOSITIVE')
    prices=list(range(min(histogram),max(histogram)+1));vol=[int(histogram.get(p,0)) for p in prices]
    p=max(range(len(prices)),key=lambda i:(vol[i],-prices[i]));lo=hi=p
    used=vol[p];total=sum(vol)
    while used*10<total*7:
        l=vol[lo-1] if lo else -1;r=vol[hi+1] if hi+1<len(prices) else -1
        if l>=r:lo-=1;used+=vol[lo]
        else:hi+=1;used+=vol[hi]
    return prices[p],prices[lo],prices[hi]


def profile_values_v2(histogram,tick_usd):
    poc,low,high=value_area_v2(histogram);volume=sum(map(int,histogram.values()))
    return {'vwap_usd':sum(int(p)*int(v) for p,v in histogram.items())/volume*tick_usd,
       'poc_usd':poc*tick_usd,'val_usd':low*tick_usd,'vah_usd':high*tick_usd,
       'volume_contracts':volume}


def dynamic_profiles_v2(starts,histograms,valid,contracts,close,tick_usd):
    """Real-trade rolling profiles and explicit incomplete session diagnostics.

    Session anchor: 18:00 America/New_York, DST-aware, not a historical holiday
    calendar. Strict session mask demands every minute since that anchor under
    004. Native-only ladders may not meet this; incomplete values are diagnostics.
    A separate causal anchor restarts at each rejected bar/roll/session.
    """
    from collections import deque,Counter
    stamps=pd.DatetimeIndex(starts,tz='UTC').tz_convert('America/New_York')
    days=stamps.tz_localize(None).normalize()-pd.to_timedelta((stamps.hour<18).astype(np.int64),unit='D')
    anchors=(days+pd.Timedelta(hours=18)).tz_localize('America/New_York').tz_convert('UTC').asi8
    out={};profiles=('rolling15','rolling60','anchored_segment','session_observed')
    names=('vwap_usd','poc_usd','val_usd','vah_usd','volume_contracts')
    for family in profiles:
        for name in names:out[family+'_'+name]=np.full(len(starts),np.nan)
        out[family+'_valid']=np.zeros(len(starts),bool)
    out['session_complete_valid']=np.zeros(len(starts),bool)
    out['session_accepted_minutes']=np.zeros(len(starts),np.int64)
    out['session_expected_minutes']=(starts+5*timing.MINUTE_NS-anchors)//timing.MINUTE_NS
    rolling=deque();segment=Counter();session=Counter();accepted=0
    for i,start in enumerate(starts):
        reset=(i==0 or contracts[i]!=contracts[i-1] or anchors[i]!=anchors[i-1])
        if reset:session=Counter();accepted=0
        if reset or i==0 or not valid[i-1] or start-starts[i-1]!=5*timing.MINUTE_NS:
            rolling.clear();segment=Counter()
        if not valid[i]:rolling.clear();segment=Counter();continue
        hist=histograms[i]
        if not hist:raise RuntimeError('PROFILE_VALID_WITHOUT_TRADES')
        rolling.append(hist)
        if len(rolling)>12:rolling.popleft()
        segment.update(hist);session.update(hist);accepted+=5
        families={'anchored_segment':segment,'session_observed':session}
        for n in (3,12):
            if len(rolling)>=n:
                combined=Counter()
                for h in list(rolling)[-n:]:combined.update(h)
                families['rolling'+str(n*5)]=combined
        for name,h in families.items():
            for key,v in profile_values_v2(h,tick_usd).items():out[name+'_'+key][i]=v
            out[name+'_valid'][i]=True
        out['session_accepted_minutes'][i]=accepted
        out['session_complete_valid'][i]=(accepted==out['session_expected_minutes'][i])
    # GC-relative distances and reactions to the PREVIOUS admitted rolling level.
    for family in ('rolling15','rolling60','anchored_segment'):
        for level in ('vwap','poc','val','vah'):
            price=out[family+'_'+level+'_usd'];out[family+'_'+level+'_distance_ticks']=(close-price)/tick_usd
            previous=np.r_[np.nan,price[:-1]]
            same=np.r_[False,(np.diff(starts)==5*timing.MINUTE_NS)&(contracts[1:]==contracts[:-1])]
            reaction=(close-previous)/tick_usd;reaction[~(same&valid&np.r_[False,valid[:-1]])]=np.nan
            out[family+'_'+level+'_prior_level_response_ticks']=reaction
    return out


def exact_feature_readback_v2(expected,actual):
    """Arrow defaults treat NaN != NaN; preserve schema, nulls and exact values."""
    if not expected.schema.equals(actual.schema,check_metadata=True):return False
    for name in expected.column_names:
        a,b=expected[name],actual[name]
        if not a.is_null().equals(b.is_null()):return False
        if pa.types.is_floating(a.type):
            if not np.array_equal(a.to_numpy(),b.to_numpy(),equal_nan=True):return False
        elif not a.equals(b):return False
    return True


def full_gc_minute_selection_v2(minutes,roll_schedule):
    """Same 004 contract decision, independent of the spot source's clock.

    Selection is known at roll_at_ns <= minute START, exactly matching 004.
    The 00:00 minute retains the prior choice when a new choice arrives 00:01.
    """
    stamps=np.asarray(minutes,np.int64)
    dates=np.array([r['roll_at_ns'] for r in roll_schedule['schedule']],np.int64)
    if np.any(np.diff(dates)<=0):raise ValueError('GC_FULL_CLOCK_SCHEDULE_ORDER')
    ci=np.array([-1 if r['selected_symbol'] is None else roll_schedule['contract_order'].index(r['selected_symbol']) for r in roll_schedule['schedule']],np.int16)
    pos=np.searchsorted(dates,stamps,side='right')-1
    out=ci[np.maximum(pos,0)].copy();out[pos<0]=-1
    return out
