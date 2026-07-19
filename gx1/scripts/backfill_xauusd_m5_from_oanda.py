#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent XAU_USD M5 Candle Backfill from OANDA + Raw Repair Mode.

- Default: backfill new M5 bid/ask candles (idempotent merge)
- Repair mode: scan an existing raw parquet for timestamp gaps, fetch missing candles,
  and write a repaired parquet + manifest with deterministic proofs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import struct
import sys
import time
import urllib.request
import lzma
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from gx1.execution.oanda_client import OandaClient, OandaClientConfig  # noqa: E402
from gx1.execution.oanda_credentials import load_oanda_credentials  # noqa: E402
from gx1.utils.env_loader import load_dotenv_if_present  # noqa: E402
from gx1.utils.granularity import granularity_to_minutes, granularity_to_pandas_freq  # noqa: E402
from gx1_guards.gates import require_retrain_vedtak  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Default paths
_DEFAULT_TAPE_ROOT = os.getenv(
    "GX1_CANONICAL_TAPE_ROOT",
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL",
)
_DEFAULT_TAPE_YEAR = datetime.now(timezone.utc).year
DEFAULT_CANDLE_FILE = Path(_DEFAULT_TAPE_ROOT) / f"year={_DEFAULT_TAPE_YEAR}" / "part-000.parquet"
INSTRUMENT = "XAU_USD"
GRANULARITY = "M5"
DEFAULT_DUKA_CACHE_DIR = Path(os.getenv("GX1_DATA", "/home/andre2/GX1_DATA")).expanduser() / "data" / "external" / "dukascopy_cache"
DEFAULT_DUKA_SYMBOL = "XAUUSD"
DUKA_SCALE_CANDIDATES = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]


def _load_oanda_client(prod_baseline: bool = False) -> OandaClient:
    creds = load_oanda_credentials(prod_baseline=prod_baseline)
    config = OandaClientConfig(
        api_key=creds.api_token,
        account_id=creds.account_id,
        env=creds.env,
    )
    return OandaClient(config)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_candles_bid_ask(
    client: OandaClient,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: str = "M5",
    chunk_days: int = 15,
) -> pd.DataFrame:
    """
    Fetch candles with bid/ask prices from OANDA using paged requests.
    Returns a DataFrame indexed by time (UTC) with bid/ask and mid columns.
    """
    log.info(f"[FETCH] {instrument} {granularity} {start} -> {end}")
    all_chunks: List[pd.DataFrame] = []
    freq = granularity_to_pandas_freq(granularity)
    if chunk_days <= 0:
        chunk_days = 1
    if chunk_days == 1:
        print("[RAW_OANDA_FETCH_CHUNKING_PROOF] chunk_days=1", flush=True)
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days), end)
        params = {
            "from": current_start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "to": current_end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "granularity": granularity,
            "price": "MBA",
        }
        try:
            data = client._request("GET", f"/instruments/{instrument}/candles", params=params)
        except Exception as e:
            log.error(f"[FETCH] chunk {current_start} -> {current_end} failed: {e}")
            current_start = current_end
            continue
        candles = data.get("candles", [])
        rows = []
        for c in candles:
            if not c.get("complete", False):
                continue
            raw_time = pd.to_datetime(c["time"])
            if raw_time.tzinfo is None:
                raw_time = raw_time.tz_localize("UTC")
            else:
                raw_time = raw_time.tz_convert("UTC")
            t = raw_time.floor(freq)
            mid = c.get("mid", {})
            bid = c.get("bid", mid)
            ask = c.get("ask", mid)
            rows.append(
                {
                    "time": t,
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": float(c.get("volume", 0)),
                    "bid_open": float(bid.get("o", mid.get("o", 0))),
                    "bid_high": float(bid.get("h", mid.get("h", 0))),
                    "bid_low": float(bid.get("l", mid.get("l", 0))),
                    "bid_close": float(bid.get("c", mid.get("c", 0))),
                    "ask_open": float(ask.get("o", mid.get("o", 0))),
                    "ask_high": float(ask.get("h", mid.get("h", 0))),
                    "ask_low": float(ask.get("l", mid.get("l", 0))),
                    "ask_close": float(ask.get("c", mid.get("c", 0))),
                }
            )
        if rows:
            df_chunk = pd.DataFrame(rows).set_index("time")
            all_chunks.append(df_chunk)
            log.info(f"[FETCH] chunk {current_start.date()}->{current_end.date()} rows={len(df_chunk)}")
        time.sleep(0.25)
        current_start = current_end
    if not all_chunks:
        return pd.DataFrame()
    df = pd.concat(all_chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def merge_candles(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([existing, new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _ensure_ts_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if "time" in df.columns:
        ts_col = "time"
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.set_index(ts_col)
    else:
        ts_col = "index"
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()
    return df, ts_col


def _is_expected_closed_gap(prev_ts: pd.Timestamp, now_ts: pd.Timestamp) -> bool:
    """
    Treat 21:55–23:00 UTC as expected closed window (no quotes).
    If the entire gap (prev+5min .. now-5min) lies within that daily window, accept it.
    """
    gap_start = prev_ts + pd.Timedelta(minutes=5)
    gap_end = now_ts - pd.Timedelta(minutes=5)
    same_day = gap_start.date() == gap_end.date()
    start_ok = gap_start.time() >= dtime(hour=21, minute=55)
    end_ok = gap_end.time() <= dtime(hour=23, minute=0)
    return same_day and start_ok and end_ok


def _dukascopy_url(symbol: str, hour_start: pd.Timestamp) -> str:
    # Dukascopy datafeed months are zero-based (January=00).
    month_zero_based = int(hour_start.month) - 1
    return (
        "https://datafeed.dukascopy.com/datafeed/"
        f"{symbol}/{hour_start:%Y}/{month_zero_based:02d}/{hour_start:%d}/{hour_start:%H}h_ticks.bi5"
    )


def _download_dukascopy_bi5(
    url: str,
    dest: Path,
    max_retries: int = 3,
    backoff: Iterable[float] = (0.5, 1.0, 2.0),
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        size = dest.stat().st_size
        if size > 0:
            return size
        dest.unlink(missing_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    attempt = 0
    for delay in backoff:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("empty response")
            dest.write_bytes(data)
            size = dest.stat().st_size
            if size <= 0:
                raise RuntimeError("zero bytes after write")
            return size
        except Exception as e:
            if attempt >= max_retries:
                raise RuntimeError(f"download failed after {attempt} attempts: {e}")
            time.sleep(delay)
    raise RuntimeError("download failed")


def _dukascopy_hour_probably_closed(symbol: str, hour_start: pd.Timestamp) -> bool:
    symbol_upper = str(symbol or "").upper()
    if symbol_upper not in {"XAUUSD", "XAU_USD"}:
        return False
    hour = pd.Timestamp(hour_start)
    if hour.tzinfo is None:
        hour = hour.tz_localize("UTC")
    else:
        hour = hour.tz_convert("UTC")
    weekday = int(hour.weekday())
    if weekday == 5:
        return True
    if weekday == 6 and int(hour.hour) < 22:
        return True
    if weekday == 4 and int(hour.hour) >= 22:
        return True
    return False


def _parse_bi5_ticks(
    path: Path,
    hour_start: pd.Timestamp,
) -> pd.DataFrame:
    raw = path.read_bytes()
    if not raw:
        raise RuntimeError(f"empty bi5 file: {path}")
    try:
        decompressed = lzma.decompress(raw)
    except Exception as e:
        raise RuntimeError(f"lzma decompress failed: {e}")
    if len(decompressed) % 20 != 0:
        raise RuntimeError(f"bi5 payload not aligned to 20 bytes: {len(decompressed)}")
    rows = []
    for time_ms, bid_i, ask_i, bid_vol, ask_vol in struct.iter_unpack(">iiiff", decompressed):
        rows.append((time_ms, bid_i, ask_i, bid_vol, ask_vol))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time_ms", "bid_i", "ask_i", "bid_vol", "ask_vol"])
    ts = hour_start + pd.to_timedelta(df["time_ms"], unit="ms")
    df["time"] = pd.to_datetime(ts, utc=True)
    df = df.drop(columns=["time_ms"])
    df = df[["time", "bid_i", "ask_i", "bid_vol", "ask_vol"]]
    return df


def _autodetect_scale(
    bid_i: np.ndarray,
    ask_i: np.ndarray,
    candidates: Iterable[int] = DUKA_SCALE_CANDIDATES,
) -> int:
    best_scale = None
    best_score = None
    for s in candidates:
        bid = bid_i / s
        ask = ask_i / s
        median_bid = float(np.nanmedian(bid))
        ask_ge_bid_ratio = float(np.mean(ask >= bid))
        if 100 <= median_bid <= 10_000 and ask_ge_bid_ratio >= 0.99:
            score = ask_ge_bid_ratio * 100000 - abs(median_bid - 2000)
            if best_score is None or score > best_score:
                best_score = score
                best_scale = s
                best_median = median_bid
                best_ratio = ask_ge_bid_ratio
    if best_scale is None:
        raise RuntimeError("[DUKA_SCALE_FAIL] no scale candidates matched plausibility")
    print(
        "[DUKA_SCALE_PROOF] "
        f"scale={best_scale} median_bid={best_median:.6f} ask_ge_bid_ratio={best_ratio:.6f}",
        flush=True,
    )
    return int(best_scale)


def _resample_ticks_to_bars(df_ticks: pd.DataFrame, granularity: str = "M5") -> pd.DataFrame:
    df = df_ticks.copy()
    df = df.set_index("time").sort_index()
    df["bid"] = df["bid_i"]
    df["ask"] = df["ask_i"]
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    freq = granularity_to_pandas_freq(granularity)

    def _ohlc(series: pd.Series) -> pd.DataFrame:
        return series.resample(freq).agg(["first", "max", "min", "last"])

    bid_ohlc = _ohlc(df["bid"])
    ask_ohlc = _ohlc(df["ask"])
    mid_ohlc = _ohlc(df["mid"])
    vol = (df["bid_vol"] + df["ask_vol"]).resample(freq).sum(min_count=1)

    out = pd.DataFrame(
        {
            "open": mid_ohlc["first"],
            "high": mid_ohlc["max"],
            "low": mid_ohlc["min"],
            "close": mid_ohlc["last"],
            "bid_open": bid_ohlc["first"],
            "bid_high": bid_ohlc["max"],
            "bid_low": bid_ohlc["min"],
            "bid_close": bid_ohlc["last"],
            "ask_open": ask_ohlc["first"],
            "ask_high": ask_ohlc["max"],
            "ask_low": ask_ohlc["min"],
            "ask_close": ask_ohlc["last"],
            "volume": vol,
        }
    )
    out = out.dropna(subset=["bid_close", "ask_close", "close"], how="any")
    return out


def _resample_ticks_to_m5(df_ticks: pd.DataFrame) -> pd.DataFrame:
    return _resample_ticks_to_bars(df_ticks, "M5")


def _dukascopy_patch(
    missing_ts: List[pd.Timestamp],
    symbol: str,
    cache_dir: Path,
    max_hours: int,
    granularity: str = "M5",
    allow_missing_hours: bool = False,
) -> pd.DataFrame:
    if not missing_ts:
        return pd.DataFrame()
    missing_ts = sorted(set(pd.to_datetime(missing_ts, utc=True)))
    start_ts = missing_ts[0]
    end_ts = missing_ts[-1]
    hour_start = start_ts.floor("h")
    hour_end = end_ts.floor("h")
    hours = pd.date_range(hour_start, hour_end, freq="h", tz="UTC")
    if len(hours) > max_hours:
        raise RuntimeError(f"[DUKA_HOUR_LIMIT] needed_hours={len(hours)} max_hours={max_hours}")

    total_bytes = 0
    downloaded = 0
    skipped_hours = 0
    tick_frames = []
    for hour in hours:
        url = _dukascopy_url(symbol, hour)
        rel = Path(symbol) / f"{hour:%Y/%m/%d}/{hour:%H}h_ticks.bi5"
        path = cache_dir / rel
        if allow_missing_hours and _dukascopy_hour_probably_closed(symbol, hour):
            skipped_hours += 1
            continue
        try:
            size = _download_dukascopy_bi5(url, path)
        except Exception:
            if not allow_missing_hours:
                raise
            skipped_hours += 1
            continue
        total_bytes += size
        downloaded += 1
        ticks = _parse_bi5_ticks(path, hour)
        if not ticks.empty:
            tick_frames.append(ticks)

    print(
        "[RAW_DUKA_FETCH_PROOF] "
        f"files_needed={len(hours)} files_downloaded={downloaded} files_skipped={skipped_hours} bytes_total={total_bytes}",
        flush=True,
    )

    if not tick_frames:
        return pd.DataFrame()

    df_ticks = pd.concat(tick_frames, ignore_index=True)

    ask_ge_bid_ratio = float(np.mean(df_ticks["ask_i"].to_numpy() >= df_ticks["bid_i"].to_numpy()))
    if ask_ge_bid_ratio < 0.99:
        swapped_ratio = float(np.mean(df_ticks["bid_i"].to_numpy() >= df_ticks["ask_i"].to_numpy()))
        if swapped_ratio >= 0.99:
            df_ticks = df_ticks.rename(columns={"bid_i": "bid_tmp"})
            df_ticks = df_ticks.rename(columns={"ask_i": "bid_i"})
            df_ticks = df_ticks.rename(columns={"bid_tmp": "ask_i"})
            print(
                "[DUKA_SWAP_PROOF] "
                f"ask_ge_bid_ratio_before={ask_ge_bid_ratio:.6f} ask_ge_bid_ratio_after={swapped_ratio:.6f}",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"[DUKA_BID_ASK_FAIL] ask_ge_bid_ratio={ask_ge_bid_ratio:.6f} swapped_ratio={swapped_ratio:.6f}"
            )

    scale = _autodetect_scale(df_ticks["bid_i"].to_numpy(), df_ticks["ask_i"].to_numpy())
    df_ticks["bid_i"] = df_ticks["bid_i"] / scale
    df_ticks["ask_i"] = df_ticks["ask_i"] / scale

    print(
        "[DUKA_BI5_PARSE_PROOF] "
        f"ticks={len(df_ticks)} ts_min={df_ticks['time'].min()} ts_max={df_ticks['time'].max()}",
        flush=True,
    )

    bars = _resample_ticks_to_bars(df_ticks, granularity)
    return bars


def _dukascopy_fetch_range(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol: str,
    cache_dir: Path,
    max_hours: int,
    granularity: str,
) -> pd.DataFrame:
    step = pd.Timedelta(minutes=granularity_to_minutes(granularity))
    if end <= start:
        return pd.DataFrame()
    expected = list(pd.date_range(start, end - step, freq=granularity_to_pandas_freq(granularity), tz="UTC"))
    if not expected:
        return pd.DataFrame()
    bars = _dukascopy_patch(
        missing_ts=expected,
        symbol=symbol,
        cache_dir=cache_dir,
        max_hours=max_hours,
        granularity=granularity,
        allow_missing_hours=True,
    )
    if bars.empty:
        return bars
    bars = bars.loc[(bars.index >= start) & (bars.index < end)].sort_index()
    return bars


def _gap_check(
    df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, allow_expected_closed: bool = False
) -> Tuple[int, int]:
    subset = df.loc[start_ts:end_ts]
    if subset.empty:
        return 0, 0
    subset_series = subset.index.to_series()
    gaps_found = 0
    max_dt_sec = 0
    for pos in range(1, len(subset_series)):
        prev_ts = subset_series.iloc[pos - 1]
        now_ts = subset_series.iloc[pos]
        delta_sec = (now_ts - prev_ts).total_seconds()
        if delta_sec > max_dt_sec:
            max_dt_sec = int(delta_sec)
        if delta_sec <= 300:
            continue
        if allow_expected_closed and _is_expected_closed_gap(prev_ts, now_ts):
            continue
        gaps_found += 1
    print(
        "[LOCAL_GAP_CHECK] "
        f"rows={len(subset)} max_dt_sec={max_dt_sec} gaps_found={gaps_found}",
        flush=True,
    )
    return max_dt_sec, gaps_found


def _repair_raw_candles(
    raw_in: Path,
    raw_out: Path,
    start_ts: str,
    end_ts: str,
    granularity: str,
    instrument: str,
    dukascopy_enabled: bool,
    dukascopy_cache_dir: Path,
    dukascopy_symbol: str,
    dukascopy_max_hours: int,
) -> dict:
    raw_in = raw_in.expanduser().resolve()
    raw_out = raw_out.expanduser().resolve()

    df_raw = pd.read_parquet(raw_in)
    df_raw, ts_origin = _ensure_ts_index(df_raw)
    df_raw = df_raw.loc[pd.Timestamp(start_ts, tz="UTC") : pd.Timestamp(end_ts, tz="UTC")]
    df_raw = df_raw.sort_index()

    ts_idx = df_raw.index
    ts_series = ts_idx.to_series().reset_index(drop=True)
    diffs = ts_series.diff().dt.total_seconds()
    gaps = diffs[diffs > 300]
    gaps_found = int(len(gaps))
    first_gap = None
    if gaps_found > 0:
        gi = gaps.index[0]
        first_gap = {
            "prev_ts": ts_series.iloc[gi - 1] if gi - 1 >= 0 else None,
            "now_ts": ts_series.iloc[gi],
            "spacing_sec": gaps.iloc[0],
        }
    print(
        "[RAW_GAP_SCAN_PROOF] "
        f"n_rows={len(df_raw)} gaps_found={gaps_found} first_gap={first_gap}",
        flush=True,
    )
    for idx in gaps.index:
        prev_ts = ts_series.iloc[idx - 1] if idx - 1 >= 0 else None
        now_ts = ts_series.iloc[idx]
        spacing = diffs.loc[idx]
        print(
            f"[RAW_GAP_ITEM] prev_ts={prev_ts} now_ts={now_ts} spacing_sec={spacing}",
            flush=True,
        )

    print("[RAW_NO_SYNTH_POLICY_PROOF] synth_disabled=1", flush=True)
    filled_segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    fill_frames: List[pd.DataFrame] = []
    filled_from_oanda = 0
    filled_from_duka = 0
    expected_closed = 0

    if gaps_found > 0:
        client = _load_oanda_client(prod_baseline=False)
        for idx in gaps.index:
            prev_ts = ts_idx[idx - 1]
            now_ts = ts_idx[idx]
            gap_start = prev_ts + pd.Timedelta(minutes=5)
            gap_end = now_ts - pd.Timedelta(minutes=5)
            if gap_start > gap_end:
                continue
            if _is_expected_closed_gap(prev_ts, now_ts):
                expected_closed += 1
                print(
                    "[RAW_EXPECTED_CLOSED_GAP_PROOF] "
                    f"prev_ts={prev_ts} now_ts={now_ts} gap_start={gap_start} gap_end={gap_end} "
                    "window=21:55-23:00",
                    flush=True,
                )
                continue
            filled_segments.append((gap_start, gap_end))
            fetched = fetch_candles_bid_ask(
                client=client,
                instrument=instrument,
                start=gap_start,
                end=gap_end,
                granularity=granularity,
                chunk_days=1,
            )
            if not fetched.empty:
                fill_frames.append(fetched)
                filled_from_oanda += len(fetched)
                print(
                    f"[RAW_GAP_FILL] gap_start={gap_start} gap_end={gap_end} rows={len(fetched)} "
                    f"ts_min={fetched.index.min()} ts_max={fetched.index.max()}",
                    flush=True,
                )

            expected = pd.date_range(gap_start, gap_end, freq="5min", tz="UTC")
            existing = set(df_raw.index)
            if not fetched.empty:
                existing |= set(fetched.index)
            missing = [ts for ts in expected if ts not in existing]
            if not missing:
                continue
            if not dukascopy_enabled:
                if fetched.empty:
                    raise RuntimeError("[RAW_REPAIR_FAIL] OANDA returned 0 rows and Dukascopy is disabled")
                raise RuntimeError("[RAW_REPAIR_FAIL] remaining missing bars after OANDA fetch; Dukascopy disabled")

            duka_df = _dukascopy_patch(
                missing_ts=missing,
                symbol=dukascopy_symbol,
                cache_dir=dukascopy_cache_dir,
                max_hours=dukascopy_max_hours,
            )
            if duka_df.empty:
                if fetched.empty:
                    raise RuntimeError(
                        "[RAW_REPAIR_FAIL] OANDA returned 0 rows and Dukascopy patch failed; no synth allowed"
                    )
                raise RuntimeError(
                    "[RAW_REPAIR_FAIL] remaining missing bars after Dukascopy patch"
                )
            duka_df = duka_df.loc[duka_df.index.isin(missing)]
            missing_after = [ts for ts in missing if ts not in duka_df.index]
            print(
                "[RAW_DUKA_PATCH_PROOF] "
                f"missing_before={len(missing)} filled_from_duka={len(duka_df)} "
                f"missing_after={len(missing_after)}",
                flush=True,
            )
            if missing_after:
                raise RuntimeError(
                    "[RAW_REPAIR_FAIL] remaining missing bars after Dukascopy patch"
                )
            fill_frames.append(duka_df)
            filled_from_duka += len(duka_df)

    if fill_frames:
        df_fill = pd.concat(fill_frames).sort_index().drop_duplicates(keep="last")
    else:
        df_fill = pd.DataFrame(columns=df_raw.columns)
    print(f"[RAW_GAP_FILL_SUMMARY] total_fill_rows={len(df_fill)}", flush=True)
    actionable_gaps = gaps_found - expected_closed
    if actionable_gaps > 0 and len(df_fill) == 0:
        raise RuntimeError(
            "[RAW_REPAIR_FAIL] OANDA returned 0 rows and Dukascopy patch failed; no synth allowed"
        )

    if not df_fill.empty:
        df_fill = df_fill.reindex(columns=df_raw.columns)

    merged = pd.concat([df_raw, df_fill]).sort_index().drop_duplicates(keep="last")
    if not merged.index.is_monotonic_increasing:
        merged = merged.sort_index()

    if not merged.index.is_unique:
        raise RuntimeError("[RAW_REPAIR_FAIL] duplicate timestamps after merge")

    start_ts_utc = pd.Timestamp(start_ts, tz="UTC")
    end_ts_utc = pd.Timestamp(end_ts, tz="UTC")
    _, gaps_after = _gap_check(merged, start_ts_utc, end_ts_utc, allow_expected_closed=True)
    if gaps_after != 0:
        raise RuntimeError(f"[RAW_REPAIR_FAIL] gaps remain after repair gaps_after={gaps_after}")

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    merged_reset = merged.reset_index()
    # Ensure time column is present for downstream loaders
    merged_reset = merged_reset.rename(columns={"index": "time"})
    merged_reset.to_parquet(raw_out)
    sha_out = _sha256_file(raw_out)

    print(
        "[RAW_REPAIR_PROOF] "
        f"filled_bars={len(df_fill)} gaps_after={gaps_after} "
        f"ts_min={merged.index.min()} ts_max={merged.index.max()} sha256_out={sha_out}",
        flush=True,
    )

    manifest = {
        "raw_in": str(raw_in),
        "raw_out": str(raw_out),
        "n_rows_out": int(len(merged_reset)),
        "ts_min": merged.index.min().isoformat() if len(merged) else None,
        "ts_max": merged.index.max().isoformat() if len(merged) else None,
        "filled_bars": int(len(df_fill)),
        "counts": {
            "gaps_found": gaps_found,
            "filled_from_oanda": int(filled_from_oanda),
            "filled_from_duka": int(filled_from_duka),
            "expected_closed": int(expected_closed),
        },
        "source": "oanda_raw + dukascopy_patch",
        "patch_window": {"start": start_ts_utc.isoformat(), "end": end_ts_utc.isoformat()},
        "synth_disabled": 1,
        "gaps_after": gaps_after,
        "sha256_out": sha_out,
        "filled_intervals": [
            {"start": s.isoformat(), "end": e.isoformat()} for s, e in filled_segments
        ],
    }
    manifest_path = raw_out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _floor_for_granularity(ts: pd.Timestamp, granularity: str) -> pd.Timestamp:
    return ts.floor(granularity_to_pandas_freq(granularity))


def _write_canonical_time_column(df: pd.DataFrame, candle_file: Path) -> None:
    out = df.sort_index().copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise RuntimeError("[BACKFILL_CANDLES] final dataframe must have DatetimeIndex")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = "time"
    out = out.reset_index()
    out.to_parquet(candle_file, index=False)


def backfill_main(
    instrument: str,
    granularity: str,
    end_ts: str | None = None,
    *,
    dukascopy_enabled: bool = False,
    dukascopy_cache_dir: Path = DEFAULT_DUKA_CACHE_DIR,
    dukascopy_symbol: str = DEFAULT_DUKA_SYMBOL,
    dukascopy_max_hours: int = 72,
) -> dict:
    log.info("=" * 60)
    log.info("XAU_USD M5 Candle Backfill (Idempotent)")
    log.info("=" * 60)

    load_dotenv_if_present()
    client = None
    try:
        client = _load_oanda_client(prod_baseline=False)
        log.info("Initialized OANDA client")
    except Exception as e:
        if not dukascopy_enabled:
            log.error(f"Failed to initialize OANDA client: {e}")
            return {"success": False, "error": str(e)}
        log.warning("[BACKFILL_CANDLES] OANDA client init failed; Dukascopy fallback enabled: %s", e)

    if os.environ.get("GX1_RAW_2025", "").strip():
        raise RuntimeError("[RAW_LANE_FORBIDDEN] GX1_RAW_2025 is set; raw lane is retired")

    candle_file = DEFAULT_CANDLE_FILE
    if "/data/data/raw/" in str(candle_file) or "xauusd_m5_2025_bid_ask.parquet" in str(candle_file):
        raise RuntimeError(f"[RAW_LANE_FORBIDDEN] candle_file points to raw lane: {candle_file}")
    existing_df = pd.DataFrame()

    if candle_file.exists():
        try:
            existing_df = pd.read_parquet(candle_file)
            existing_df, _ = _ensure_ts_index(existing_df)
            log.info(f"Loaded {len(existing_df):,} existing candles")
            if len(existing_df) > 0:
                log.info(f"  Existing range: {existing_df.index.min()} to {existing_df.index.max()}")
        except Exception as e:
            log.warning(f"Failed to load existing parquet: {e}")
            existing_df = pd.DataFrame()
    else:
        log.info("Candle file does not exist - will create new file")

    now_utc = pd.Timestamp.now(tz="UTC")
    target_utc = pd.Timestamp(end_ts) if end_ts else now_utc
    if target_utc.tzinfo is None:
        target_utc = target_utc.tz_localize("UTC")
    else:
        target_utc = target_utc.tz_convert("UTC")
    now_utc_floor = _floor_for_granularity(target_utc, granularity)
    step = pd.Timedelta(minutes=granularity_to_minutes(granularity))

    if existing_df.empty or len(existing_df) == 0:
        from_time = pd.Timestamp("2025-01-01", tz="UTC")
        to_time = now_utc_floor
        log.info(f"No existing candles - fetching from {from_time.date()} to {to_time.date()}")
    else:
        last_ts = existing_df.index.max()
        log.info(f"Last existing candle: {last_ts}")
        next_bar = last_ts + step

        if last_ts >= now_utc_floor - step:
            log.info(f"[BACKFILL_CANDLES] No new candles to backfill (last_ts={last_ts}, now={now_utc_floor})")
            return {
                "success": True,
                "new_candles": 0,
                "from_time": last_ts,
                "to_time": now_utc_floor,
                "total_candles": len(existing_df),
            }

        from_time = next_bar
        to_time = now_utc_floor
        log.info(f"Fetching candles from {from_time} to {to_time}")

    try:
        if client is None:
            raise RuntimeError("OANDA_CLIENT_UNAVAILABLE")
        new_df = fetch_candles_bid_ask(
            client=client,
            instrument=instrument,
            start=from_time,
            end=to_time,
            granularity=granularity,
        )
    except Exception as e:
        if not dukascopy_enabled:
            log.error(f"Failed to fetch candles: {e}")
            return {"success": False, "error": str(e)}
        log.warning("[BACKFILL_CANDLES] OANDA fetch failed; trying Dukascopy tick fallback: %s", e)
        try:
            new_df = _dukascopy_fetch_range(
                start=from_time,
                end=to_time,
                symbol=dukascopy_symbol,
                cache_dir=Path(dukascopy_cache_dir),
                max_hours=int(dukascopy_max_hours),
                granularity=granularity,
            )
        except Exception as duka_e:
            log.error("[BACKFILL_CANDLES] Dukascopy fallback failed: %s", duka_e)
            return {"success": False, "error": f"OANDA failed: {e}; Dukascopy failed: {duka_e}"}

    if new_df.empty:
        if dukascopy_enabled:
            log.info("[BACKFILL_CANDLES] OANDA returned empty; trying Dukascopy tick fallback")
            new_df = _dukascopy_fetch_range(
                start=from_time,
                end=to_time,
                symbol=dukascopy_symbol,
                cache_dir=Path(dukascopy_cache_dir),
                max_hours=int(dukascopy_max_hours),
                granularity=granularity,
            )
        if new_df.empty:
            log.info(f"[BACKFILL_CANDLES] No new candles fetched for {from_time} to {to_time}")
            return {
                "success": True,
                "new_candles": 0,
                "from_time": from_time,
                "to_time": to_time,
                "total_candles": len(existing_df),
            }

    log.info(f"[BACKFILL_CANDLES] Fetched {len(new_df):,} new candles")
    log.info(f"[BACKFILL_CANDLES] New range: {new_df.index.min()} to {new_df.index.max()}")

    if existing_df.empty:
        final_df = new_df
    else:
        log.info("Merging with existing candles...")
        final_df = merge_candles(existing_df, new_df)
        log.info(f"Merged: {len(existing_df):,} existing + {len(new_df):,} new = {len(final_df):,} total")

    candle_file.parent.mkdir(parents=True, exist_ok=True)
    _write_canonical_time_column(final_df, candle_file)

    log.info("=" * 60)
    log.info("✅ Candle Backfill Complete!")
    log.info("=" * 60)
    log.info(f"[BACKFILL_CANDLES] Added {len(new_df):,} candles from {from_time} to {to_time}. Total now: {len(final_df):,} bars.")

    return {
        "success": True,
        "new_candles": len(new_df),
        "from_time": from_time,
        "to_time": to_time,
        "total_candles": len(final_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OANDA XAUUSD M5 backfill/repair")
    parser.add_argument(
        "--vedtak",
        required=True,
        help="Explicit auditable decision ID authorizing this external-data operation",
    )
    parser.add_argument("--instrument", default=INSTRUMENT)
    parser.add_argument("--granularity", default=GRANULARITY)
    parser.add_argument("--repair-mode", action="store_true", help="Enable raw repair mode")
    parser.add_argument("--raw-in", type=Path, help="Input raw parquet (for repair)")
    parser.add_argument("--raw-out", type=Path, help="Output repaired parquet (for repair)")
    parser.add_argument("--start-ts", help="Start ts (UTC ISO) for repair")
    parser.add_argument("--end-ts", help="End ts (UTC ISO) for repair")
    parser.add_argument("--dukascopy-enabled", action="store_true", default=False, help="Enable Dukascopy patch (repair)")
    parser.add_argument("--dukascopy-disabled", action="store_true", default=False, help="Disable Dukascopy patch (repair)")
    parser.add_argument("--dukascopy-cache-dir", type=Path, default=DEFAULT_DUKA_CACHE_DIR)
    parser.add_argument("--dukascopy-symbol", default=DEFAULT_DUKA_SYMBOL)
    parser.add_argument("--dukascopy-max-hours", type=int, default=72)
    parser.add_argument("--test-auth", action="store_true", help="Sanity test OANDA auth (accounts + 1 candle)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # This must run before environment loading, credentials, network access,
    # cache creation, or repaired/canonical dataset writes.
    require_retrain_vedtak(args.vedtak)
    load_dotenv_if_present()

    if args.test_auth:
        try:
            client = _load_oanda_client(prod_baseline=False)
            print("[OANDA_AUTH_TEST] credentials_loaded=1 env=practice_or_live", flush=True)
            acct = client._request("GET", "/accounts")
            print(f"[OANDA_AUTH_TEST] accounts_status=ok n_accounts={len(acct.get('accounts', []))}", flush=True)
            candles = client._request(
                "GET",
                f"/instruments/{args.instrument}/candles",
                params={"granularity": args.granularity, "count": 1, "price": "MBA"},
            )
            print(f"[OANDA_AUTH_TEST] candles_status=ok candles_len={len(candles.get('candles', []))}", flush=True)
            return 0
        except Exception as e:
            print(f"[OANDA_AUTH_TEST_FAIL] {e}", flush=True)
            return 1

    if args.repair_mode:
        if not all([args.raw_in, args.raw_out, args.start_ts, args.end_ts]):
            log.error("[REPAIR_MODE] Missing required args: --raw-in --raw-out --start-ts --end-ts")
            return 1
        if args.dukascopy_disabled:
            dukascopy_enabled = False
        elif args.dukascopy_enabled:
            dukascopy_enabled = True
        else:
            dukascopy_enabled = True
        manifest = _repair_raw_candles(
            raw_in=Path(args.raw_in),
            raw_out=Path(args.raw_out),
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            granularity=args.granularity,
            instrument=args.instrument,
            dukascopy_enabled=dukascopy_enabled,
            dukascopy_cache_dir=Path(args.dukascopy_cache_dir),
            dukascopy_symbol=args.dukascopy_symbol,
            dukascopy_max_hours=int(args.dukascopy_max_hours),
        )
        log.info("[REPAIR_MODE] Completed. Manifest written alongside output.")
        log.debug("Manifest: %s", manifest)
        return 0

    result = backfill_main(
        instrument=args.instrument,
        granularity=args.granularity,
        end_ts=args.end_ts,
        dukascopy_enabled=bool(args.dukascopy_enabled),
        dukascopy_cache_dir=Path(args.dukascopy_cache_dir),
        dukascopy_symbol=str(args.dukascopy_symbol),
        dukascopy_max_hours=int(args.dukascopy_max_hours),
    )
    if not result.get("success", False):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
