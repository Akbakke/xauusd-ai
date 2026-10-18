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
    out['observed_avg_executed_trade_contracts'] = np.divide(out['observed_volume'], out['observed_trades'],
        out=np.full(len(stamps), np.nan), where=out['observed_trades'] > 0)
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
    for name in PROFILE_FIELDS + ('val_usd', 'vah_usd'):
        out['observed_' + name] = np.full(len(stamps), np.nan)
    for name in OBSERVED_FOOTPRINT_FIELDS.values():
        out[name] = np.full(len(stamps), np.nan)
    return out,inverse


def consume_ladder(path: Path, surfaces, minute_maps, tick_size, *, profile_output=None, profile_state_max_bytes=None):
    """Reduce bounded Arrow batches into one currently open histogram per TF."""
    states = {tf:None for tf in surfaces}
    profile_state = None
    profile_anchors = None
    if profile_output is not None:
        profile_state = DynamicProfileStateV2(tick_size / PRICE_SCALE, max_state_bytes=profile_state_max_bytes)
        profile_anchors = profile_anchors_v2(surfaces['M5']['bar_start_ns'])
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
            total_histogram = {price: sum(map(int, histogram[price])) for price in ticks}
            _, value_low, value_high = value_area_v2(total_histogram)
            surface['observed_val_usd'][bar] = value_low * tick_size / PRICE_SCALE
            surface['observed_vah_usd'][bar] = value_high * tick_size / PRICE_SCALE
            if tf == 'M5' and profile_state is not None:
                row = profile_state.step(surface['bar_start_ns'][bar], total_histogram, True,
                    surface['contract_index'][bar], surface['observed_close_usd'][bar], profile_anchors[bar],
                    int(surface['observed_accepted_minutes'][bar]))
                for name, value in row.items():profile_output[name][bar] = value
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
            # The source is contract-major, not globally chronological. Keep
            # contiguous runs in stream order so an open late-contract bucket
            # receives this batch's continuation before an earlier contract
            # can trigger its conservation check. Never sort or merge revisits.
            boundaries = np.r_[0, np.flatnonzero(np.diff(bars) != 0) + 1, len(bars)]
            first, counts = boundaries[:-1], np.diff(boundaries)
            for bar,start,count in zip(bars[first],first,counts):
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
    return {'profile_stream_peak_state_bytes': 0 if profile_state is None else profile_state.peak_state_bytes}


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

def known_cumulative_contributions_v1(minutes, selected, masks, metrics):
    """Known B/A and retained N on the existing causal volume-segment clock.

    Unknown aggressors contribute only to unknown quantity. This is never
    a full signed CVD. UTC-date, roll and invalid/silent-minute resets reuse
    the canonical segment owner rather than guessing a session calendar.
    """
    names = ('buy_contracts', 'sell_contracts', 'unknown_contracts', 'volume_contracts',
             'known_delta_contracts', 'unknown_fraction', 'observed_minutes')
    out = {name: np.full(len(minutes), np.nan) for name in names}
    segment = source.reset_segments(selected, masks['volume'], minutes)
    previous = -1
    buy = sell = unknown = volume = count = 0
    for i in np.flatnonzero(masks['volume']):
        if segment[i] != previous:
            buy = sell = unknown = volume = count = 0
        previous = segment[i]
        buy += int(metrics['buy'][i]); sell += int(metrics['sell'][i])
        unknown += int(metrics['unknown'][i]); volume += int(metrics['volume'][i]); count += 1
        if volume != buy + sell + unknown or volume <= 0 or volume > 2**53:
            raise RuntimeError('GC_KNOWN_CUMULATIVE_CONSERVATION')
        values = (buy, sell, unknown, volume, buy - sell, unknown / volume, count)
        for name, value in zip(names, values):out[name][i] = value
    return out


def add_known_cumulative_contributions_v1(surface, minute_values, minutes):
    """Project the exact last observed M1, never a later/full-signed endpoint."""
    valid = surface['observed_valid']
    end = surface['observed_last_minute_ns']
    idx = np.searchsorted(minutes, np.where(valid, end, minutes[0]).astype(np.int64))
    clip = np.minimum(idx, len(minutes) - 1)
    if np.any(valid & ((idx >= len(minutes)) | (minutes[clip] != end))):
        raise RuntimeError('GC_KNOWN_CUMULATIVE_ENDPOINT')
    for name, value in minute_values.items():
        surface['observed_known_cumulative_' + name] = np.where(valid, value[clip], np.nan)
    return surface


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


EVENT_LATENCY_DOMAINS_V1 = ('send_to_capture', 'receive_minus_event', 'matching_clock')
EVENT_DIAGNOSTIC_REDUCERS_V1 = {
    'qualified_final_event_count': 'sum', 'spread_state_min_ticks': 'min', 'spread_state_max_ticks': 'max',
    'spread_change_count': 'sum', 'spread_expansion_count': 'sum', 'spread_contraction_count': 'sum',
    'spread_expansion_sum_ticks': 'sum', 'spread_contraction_sum_ticks': 'sum',
    'spread_max_expansion_ticks': 'max', 'spread_max_contraction_ticks': 'max',
    'receive_interval_count': 'sum', 'receive_interval_sum_ns': 'sum',
    'receive_interval_min_ns': 'min', 'receive_interval_max_ns': 'max',
    'ts_in_delta_clamped_low_count': 'sum', 'ts_in_delta_clamped_high_count': 'sum',
    'undefined_event_clock_count': 'sum', 'negative_send_to_capture_count': 'sum',
    'negative_matching_clock_count': 'sum',
    **{prefix + suffix: operation for prefix in EVENT_LATENCY_DOMAINS_V1 for suffix, operation in
        (('_count', 'sum'), ('_sum_ns', 'sum'), ('_min_ns', 'min'), ('_max_ns', 'max'))},
}


def aggregate_observed_event_diagnostics_v1(minutes, selected, events, timeframe):
    """Aggregate exact event sufficient statistics with separate denominators."""
    labels = multi_tf_bar_label(pd.DatetimeIndex(minutes, tz='UTC'), timeframe).asi8
    stamps, first, counts = np.unique(labels, return_index=True, return_counts=True)
    duration = MULTI_TF_SHIFT[timeframe].value
    expected = duration // timing.MINUTE_NS
    same = (np.minimum.reduceat(selected, first) >= 0)
    same &= np.minimum.reduceat(selected, first) == np.maximum.reduceat(selected, first)
    count = np.asarray(events['qualified_final_event_count'])
    accepted = np.isfinite(count) & (count > 0)
    n = np.add.reduceat(accepted.astype(np.int64), first)
    valid = (counts == expected) & same & (n > 0)
    out = {'observed_event_valid': valid, 'observed_event_accepted_minutes': n}
    reduced = {}
    for name, operation in EVENT_DIAGNOSTIC_REDUCERS_V1.items():
        values = np.asarray(events[name]); finite = np.isfinite(values) & accepted
        if operation == 'sum':value = np.add.reduceat(np.where(finite, values, 0.0), first)
        elif operation == 'min':value = np.minimum.reduceat(np.where(finite, values, np.inf), first)
        else:value = np.maximum.reduceat(np.where(finite, values, -np.inf), first)
        reduced[name] = np.where(valid & np.isfinite(value), value, np.nan)
    covered = np.add.reduceat(np.where(accepted, events['qualified_covered_ns'], 0), first)
    reduced['qualified_covered_ns'] = np.where(valid, covered, np.nan)
    reduced['qualified_events_per_observed_second'] = np.divide(reduced['qualified_final_event_count'], covered / 1e9,
        out=np.full(len(stamps), np.nan), where=valid & (covered > 0))
    for prefix in ('receive_interval',) + EVENT_LATENCY_DOMAINS_V1:
        denominator = reduced[prefix + '_count']
        reduced[prefix + '_mean_ns'] = np.divide(reduced[prefix + '_sum_ns'], denominator,
            out=np.full(len(stamps), np.nan), where=denominator > 0)
        out['observed_event_' + prefix + '_valid'] = valid & (denominator > 0)
    out['observed_event_coverage_fraction'] = np.where(valid, covered / duration, np.nan)
    for name, values in reduced.items():out['observed_event_' + name] = values
    return out


class L1FeatureCollectorV2:
    """Streaming final-event L1 reducer, piecewise constant on ts_recv.

    Integrals use the state strictly before each interval's right boundary.
    All values are later gated by the unchanged 012 clean-prior-minute mask.
    A file seam clears the anchor. Missing interior minutes are not filled.
    """
    def __init__(self,wanted,tick_size,*,origin_ns=timing.STUDY_START_NS,reference_minutes=(),include_order_counts=False,include_event_diagnostics=False):
        self.wanted=np.asarray(wanted,np.int64);self.tick_size=tick_size
        self.origin_ns=origin_ns;self.previous=None;self.end_ns=None
        self.sums=np.zeros((len(wanted),12),np.float64)
        self.covered=np.zeros(len(wanted),np.int64)
        self.first=np.full(len(wanted),np.nan);self.last=self.first.copy()
        self.first_time=np.full(len(wanted),np.iinfo(np.int64).max,np.int64)
        self.last_time=np.full(len(wanted),np.iinfo(np.int64).min,np.int64)
        self.reference_minutes=frozenset(reference_minutes);self.references=[]
        self.include_order_counts=include_order_counts
        self.include_event_diagnostics=include_event_diagnostics
        self.event_previous=None
        if include_event_diagnostics:
            self.event_stats = {name: np.full(len(wanted), np.inf if operation == 'min' else
                -np.inf if operation == 'max' else 0.0) for name, operation in EVENT_DIAGNOSTIC_REDUCERS_V1.items()}
        if include_order_counts:
            self.order_sums=np.zeros((len(wanted),len(L1_ORDER_COUNT_FIELDS_V1)),np.float64)
            self.order_covered=np.zeros(len(wanted),np.int64)
    def file_start(self,item):
        if self.previous is not None:self.finish(self.end_ns)
        self.previous=None;self.end_ns=item['end_ns'];self.event_previous=None
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
    def _consume_event_diagnostics(self, data, take, c):
        if not len(c):return
        flags = data['flags'].to_numpy(zero_copy_only=False)[take]
        clean = ((flags & (source.F_BAD_TS_RECV | source.F_MAYBE_BAD_BOOK | source.F_SNAPSHOT)) == 0)
        clean &= (c[:, 1] > 0) & (c[:, 2] > c[:, 1]) & (c[:, 3] > 0) & (c[:, 4] > 0)
        clean &= np.asarray(data['action'].to_numpy(zero_copy_only=False)[take]) != 'R'
        p = np.vstack((c[0] if self.event_previous is None else self.event_previous[0], c[:-1]))
        prior_clean = np.r_[False if self.event_previous is None else self.event_previous[1], clean[:-1]]
        minute = (c[:, 0] - self.origin_ns) // timing.MINUTE_NS
        pos, wanted = self._positions(minute)
        def add(name, mask, values):
            selected = wanted & mask
            if not selected.any():return
            operation = EVENT_DIAGNOSTIC_REDUCERS_V1[name]
            function = np.add.at if operation == 'sum' else np.minimum.at if operation == 'min' else np.maximum.at
            function(self.event_stats[name], pos[selected], np.asarray(values)[selected])
        one = np.ones(len(c));spread = c[:, 2] - c[:, 1]
        transition = clean & prior_clean & (c[:, 0] >= p[:, 0])
        change = spread - (p[:, 2] - p[:, 1])
        add('qualified_final_event_count', clean, one)
        add('spread_state_min_ticks', clean, spread);add('spread_state_max_ticks', clean, spread)
        # Include the prior observed state active before the next event.
        for name in ('spread_state_min_ticks', 'spread_state_max_ticks'):add(name, transition, p[:, 2] - p[:, 1])
        add('spread_change_count', transition & (change != 0), one)
        add('spread_expansion_count', transition & (change > 0), one)
        add('spread_contraction_count', transition & (change < 0), one)
        expansion = np.maximum(change, 0);contraction = np.maximum(-change, 0)
        add('spread_expansion_sum_ticks', transition, expansion);add('spread_contraction_sum_ticks', transition, contraction)
        add('spread_max_expansion_ticks', transition, expansion);add('spread_max_contraction_ticks', transition, contraction)
        interval = c[:, 0] - p[:, 0]
        add('receive_interval_count', transition, one);add('receive_interval_sum_ns', transition, interval)
        add('receive_interval_min_ns', transition, interval);add('receive_interval_max_ns', transition, interval)
        delta = data['ts_in_delta'].to_numpy(zero_copy_only=False)[take].astype(np.int64)
        event_raw = data['ts_event'].to_numpy(zero_copy_only=False)[take]
        defined = event_raw <= np.iinfo(np.int64).max
        low = delta == np.iinfo(np.int32).min;high = delta == np.iinfo(np.int32).max
        add('ts_in_delta_clamped_low_count', clean & low, one);add('ts_in_delta_clamped_high_count', clean & high, one)
        add('undefined_event_clock_count', clean & ~defined, one)
        event = np.where(defined, event_raw, 0).astype(np.int64)
        receive_minus_event = c[:, 0] - event
        matching = receive_minus_event - delta
        for prefix, values, mask in (
                ('send_to_capture', delta, clean & ~low & ~high),
                ('receive_minus_event', receive_minus_event, clean & defined),
                ('matching_clock', matching, clean & defined & ~low & ~high)):
            add(prefix + '_count', mask, one);add(prefix + '_sum_ns', mask, values)
            add(prefix + '_min_ns', mask, values);add(prefix + '_max_ns', mask, values)
        add('negative_send_to_capture_count', clean & ~low & ~high & (delta < 0), one)
        add('negative_matching_clock_count', clean & defined & ~low & ~high & (matching < 0), one)
        self.event_previous = (c[-1].copy(), bool(clean[-1]))
    def event_diagnostics(self, candidate):
        if not self.include_event_diagnostics:raise RuntimeError('L1_EVENT_DIAGNOSTICS_NOT_REQUESTED')
        mask = np.asarray(candidate, bool)
        out = {name: np.where(mask & np.isfinite(values), values, np.nan) for name, values in self.event_stats.items()}
        for prefix in ('receive_interval',) + EVENT_LATENCY_DOMAINS_V1:
            out[prefix + '_mean_ns'] = np.divide(out[prefix + '_sum_ns'], out[prefix + '_count'],
                out=np.full(len(mask), np.nan), where=out[prefix + '_count'] > 0)
        out['qualified_covered_ns'] = np.where(mask, self.covered, np.nan)
        out['qualified_events_per_observed_second'] = np.divide(out['qualified_final_event_count'], self.covered / 1e9,
            out=np.full(len(mask), np.nan), where=mask & (self.covered > 0))
        return out
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
        if self.include_event_diagnostics:self._consume_event_diagnostics(data, take, c)
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


DYNAMIC_PROFILE_FAMILIES_V2 = ('rolling15', 'rolling60', 'anchored_segment', 'session_observed')
DYNAMIC_PROFILE_VALUE_FIELDS_V2 = ('vwap_usd', 'poc_usd', 'val_usd', 'vah_usd', 'volume_contracts')


def profile_anchors_v2(starts):
    stamps = pd.DatetimeIndex(starts, tz='UTC').tz_convert('America/New_York')
    days = stamps.tz_localize(None).normalize() - pd.to_timedelta((stamps.hour < 18).astype(np.int64), unit='D')
    return (days + pd.Timedelta(hours=18)).tz_localize('America/New_York').tz_convert('UTC').asi8


def dynamic_profile_arrays_v2(starts):
    out = {}
    for family in DYNAMIC_PROFILE_FAMILIES_V2:
        for name in DYNAMIC_PROFILE_VALUE_FIELDS_V2:out[family + '_' + name] = np.full(len(starts), np.nan)
        out[family + '_valid'] = np.zeros(len(starts), bool)
    out['session_complete_valid'] = np.zeros(len(starts), bool)
    out['session_accepted_minutes'] = np.zeros(len(starts), np.int64)
    out['session_expected_minutes'] = (starts + 5 * timing.MINUTE_NS - profile_anchors_v2(starts)) // timing.MINUTE_NS
    for family in ('rolling15', 'rolling60', 'anchored_segment'):
        for level in ('vwap', 'poc', 'val', 'vah'):
            for suffix in ('distance_ticks', 'prior_level_response_ticks'):
                out[family + '_' + level + '_' + suffix] = np.full(len(starts), np.nan)
    return out


class DynamicProfileStateV2:
    """The existing M5/NY-session formula with one bounded streaming state.

    Global contract-major source order is safe: each contract change resets
    state; within one contract the source must advance. No all-history histogram
    array is retained. The measured state bound excludes caller-owned outputs.
    """
    def __init__(self, tick_usd, *, max_state_bytes=None):
        from collections import Counter, deque
        self.tick_usd = tick_usd; self.max_state_bytes = max_state_bytes
        self.rolling = deque(); self.segment = Counter(); self.session = Counter()
        self.accepted = 0; self.previous = None; self.peak_state_bytes = 0
        self.field_names = tuple(dynamic_profile_arrays_v2(np.array([0], np.int64)))
    def _measure(self):
        import sys
        containers = [self.segment, self.session, *self.rolling]
        measured = sys.getsizeof(self.rolling) + sum(sys.getsizeof(c) +
            sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in c.items()) for c in containers)
        if self.previous is not None:
            measured += sys.getsizeof(self.previous[4]) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k,v in self.previous[4].items())
        self.peak_state_bytes = max(self.peak_state_bytes, measured)
        if self.max_state_bytes is not None and measured > self.max_state_bytes:
            raise RuntimeError('GC_PROFILE_STREAM_STATE_MEMORY_BOUND')
    def step(self, start, histogram, valid, contract, close, anchor, accepted_minutes=5):
        from collections import Counter
        previous = self.previous
        reset = previous is None or contract != previous[1] or anchor != previous[2]
        if previous is not None and contract == previous[1] and start <= previous[0]:
            raise RuntimeError('GC_PROFILE_CONTRACT_STREAM_ORDER')
        if reset:self.session = Counter(); self.accepted = 0
        if reset or previous is None or not previous[3] or start - previous[0] != 5 * timing.MINUTE_NS:
            self.rolling.clear(); self.segment = Counter()
        result = {name: np.nan for name in self.field_names}
        for family in DYNAMIC_PROFILE_FAMILIES_V2:result[family + '_valid'] = False
        result.update(session_complete_valid=False, session_accepted_minutes=0,
            session_expected_minutes=(start + 5 * timing.MINUTE_NS - anchor) // timing.MINUTE_NS)
        if not valid:
            self.rolling.clear(); self.segment = Counter()
        else:
            if not histogram or not 0 < accepted_minutes <= 5:raise RuntimeError('PROFILE_VALID_WITHOUT_TRADES')
            self.rolling.append(histogram)
            if len(self.rolling) > 12:self.rolling.popleft()
            self.segment.update(histogram); self.session.update(histogram); self.accepted += accepted_minutes
            families = {'anchored_segment': self.segment, 'session_observed': self.session}
            for n in (3, 12):
                if len(self.rolling) >= n:
                    combined = Counter()
                    for hist in list(self.rolling)[-n:]:combined.update(hist)
                    families['rolling' + str(n * 5)] = combined
            for family, hist in families.items():
                for name, value in profile_values_v2(hist, self.tick_usd).items():result[family + '_' + name] = value
                result[family + '_valid'] = True
            result['session_accepted_minutes'] = self.accepted
            result['session_complete_valid'] = self.accepted == result['session_expected_minutes']
            for family in ('rolling15', 'rolling60', 'anchored_segment'):
                for level in ('vwap', 'poc', 'val', 'vah'):
                    key = family + '_' + level
                    result[key + '_distance_ticks'] = (close - result[key + '_usd']) / self.tick_usd
                    same = previous is not None and start - previous[0] == 5 * timing.MINUTE_NS and contract == previous[1] and previous[3]
                    if same:result[key + '_prior_level_response_ticks'] = (close - previous[4][key + '_usd']) / self.tick_usd
        self.previous = (start, contract, anchor, valid, result)
        self._measure()
        return result


def dynamic_profiles_v2(starts,histograms,valid,contracts,close,tick_usd):
    """Existing fixed15/60minute profiles, not3/12 higher-timeframe bars."""
    out = dynamic_profile_arrays_v2(starts); anchors = profile_anchors_v2(starts)
    state = DynamicProfileStateV2(tick_usd)
    for i, start in enumerate(starts):
        row = state.step(start, histograms[i], valid[i], contracts[i], close[i], anchors[i])
        for name, value in row.items():out[name][i] = value
    return out


def add_projected_m5_profiles_v2(surface, m5_available, profiles, source_coverage):
    """Latest exact closed M5 snapshot on each TF clock; never backsearch gaps."""
    available = surface['available_ns']
    idx = np.searchsorted(m5_available, available, side='right') - 1
    clip = np.clip(idx, 0, len(m5_available) - 1)
    matched = (idx >= 0) & (m5_available[clip] == available)
    for name, values in profiles.items():
        selected = values[clip]
        surface['observed_profile_' + name] = (matched & selected if selected.dtype == bool
            else np.where(matched, selected, np.nan))
    surface['observed_profile_source_coverage_fraction'] = np.where(matched, source_coverage[clip], np.nan)
    surface['observed_profile_snapshot_available_ns'] = np.where(matched, m5_available[clip], np.nan)
    surface['observed_profile_snapshot_age_minutes'] = np.where(matched, (available - m5_available[clip]) / timing.MINUTE_NS, np.nan)
    return surface


def completion_native_candidates_v1():
    """Declared field/owner/domain/clock inventory, never name-heuristic routing.

    Identity and absolute snapshot timestamps remain metadata. Latency ns are
    measured differences, not absolute identities. Actual TRAIN support and
    duplicate/constant reports are still required before registry admission.
    """
    groups = []
    def add(columns, owner, validity, clock='available_ns', role='market_value'):
        groups.append({'columns': list(columns), 'owner': owner, 'validity_column': validity,
            'source_clock_column': clock, 'role': role})
    flow = 'momentum_flow_encoder';liquidity = 'smc_liquidity_encoder'
    volatility = 'vol_compression_encoder';regime = 'session_regime_encoder'
    add(('observed_volume','observed_buy_known','observed_sell_known','observed_unknown','observed_trades',
        'observed_avg_executed_trade_contracts','observed_known_delta_contracts','observed_known_aggressor_fraction',
        'observed_accepted_minutes','observed_signed_minutes','observed_coverage_fraction','observed_signed_minute_fraction'), flow, 'observed_valid')
    add(('observed_open_usd','observed_high_usd','observed_low_usd','observed_close_usd'), 'price_action_candle_encoder', 'observed_valid')
    add(('observed_spot_basis_bps','observed_spot_basis_age_minutes'), flow,
        'observed_spot_basis_valid', 'observed_spot_basis_source_available_ns', 'same_closed_m1_cross_venue_basis')
    add(tuple('observed_' + name for name in PROFILE_FIELDS + ('val_usd','vah_usd')), liquidity, 'observed_valid')
    add(tuple(OBSERVED_FOOTPRINT_FIELDS.values()), flow, 'observed_valid')
    add(tuple('observed_known_cumulative_' + name for name in ('buy_contracts','sell_contracts','unknown_contracts',
        'volume_contracts','known_delta_contracts','unknown_fraction','observed_minutes')), flow, 'observed_valid')
    add(tuple('observed_proxy_' + name for name in VOLUME_ORDER_BLOCK_PROXY_FIELDS_V1) +
        ('observed_proxy_boundary_gap_minutes',), liquidity, 'observed_valid')
    profile_names = tuple(dynamic_profile_arrays_v2(np.array([0], np.int64)))
    add(tuple('observed_profile_' + name for name in profile_names) +
        ('observed_profile_source_coverage_fraction','observed_profile_snapshot_age_minutes'), liquidity,
        None, 'observed_profile_snapshot_available_ns', 'closed_m5_clock_context')
    add(tuple('observed_book_' + name for name in L1_FIELDS_V2[:6]) +
        ('observed_book_flow_accepted_minutes','observed_book_flow_coverage_fraction'), flow, 'observed_book_flow_valid')
    add(tuple('observed_book_' + name for name in ('tw_imbalance','tw_bid_contracts','tw_ask_contracts',
        'tw_microprice_offset_ticks','mid_response_ticks','opening_mid_tick','closing_mid_tick')) +
        ('observed_book_accepted_minutes','observed_book_coverage_fraction','observed_book_response_span_minutes',
         'observed_book_response_contiguous'), flow, 'observed_book_valid')
    add(('observed_book_tw_spread_ticks',), volatility, 'observed_book_valid')
    add(tuple('observed_order_' + name for name in L1_ORDER_COUNT_FIELDS_V1) +
        ('observed_order_accepted_minutes','observed_order_coverage_fraction'), flow, 'observed_order_valid')
    spread_fields = ('spread_state_min_ticks','spread_state_max_ticks','spread_change_count','spread_expansion_count',
        'spread_contraction_count','spread_expansion_sum_ticks','spread_contraction_sum_ticks',
        'spread_max_expansion_ticks','spread_max_contraction_ticks')
    add(tuple('observed_event_' + name for name in spread_fields), volatility, 'observed_event_valid')
    add(('observed_event_qualified_final_event_count','observed_event_qualified_events_per_observed_second',
        'observed_event_accepted_minutes','observed_event_coverage_fraction'), flow, 'observed_event_valid')
    for prefix in ('receive_interval',) + EVENT_LATENCY_DOMAINS_V1:
        add(tuple('observed_event_' + prefix + suffix for suffix in ('_count','_min_ns','_max_ns','_mean_ns')),
            regime, 'observed_event_' + prefix + '_valid')
    add(tuple('observed_event_' + name for name in ('ts_in_delta_clamped_low_count','ts_in_delta_clamped_high_count',
        'undefined_event_clock_count','negative_send_to_capture_count','negative_matching_clock_count')), regime, 'observed_event_valid')
    masks = ('observed_valid','observed_contract_valid','observed_signed_complete','observed_book_contract_valid',
        'observed_book_flow_valid','observed_book_valid','observed_order_valid','observed_event_valid') + tuple(
        'observed_event_' + prefix + '_valid' for prefix in ('receive_interval',) + EVENT_LATENCY_DOMAINS_V1)
    add(masks, regime, None, role='structural_observation_mask')
    records = []
    for tf in ('M5','M15','H1','H4','D1'):
        for group in groups:
            for column in group['columns']:
                records.append({'name': 'gc_' + tf.lower() + '_' + column,
                    'table': 'TRADE_FEATURES_' + tf + '.parquet', 'column': column,
                    **{key: value for key, value in group.items() if key != 'columns'}})
    identities = [(row['table'], row['column']) for row in records]
    if len(set(identities)) != len(identities):raise RuntimeError('GC_COMPLETION_CANDIDATE_OWNER_DUPLICATE')
    return {'schema_version':'gc_completion_native_candidates_v1','fields':records,
        'metadata_columns':['bar_start_ns','available_ns','contract_index','observed_first_minute_ns',
            'observed_last_minute_ns','observed_book_first_minute_ns','observed_book_last_minute_ns',
            'observed_profile_snapshot_available_ns','observed_spot_basis_source_available_ns'],
        'profile_clock_semantics':'M5 physical15/60minute and NYsession snapshots on exact closed TF clock, not native TF rolling bars',
        'new_registry_admitted':False,'actual_physical_train_support_required':True}


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
