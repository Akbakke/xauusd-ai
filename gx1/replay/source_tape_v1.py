"""Immutable read-only price source for bounded offline replay lookup."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.io.price_glitch_guard import assert_no_price_scale_glitch
from gx1.models.entry_v10.direction_decision_contract import (
    canonical_closed_m1_bar,
)


@dataclass(frozen=True)
class SourceTape:
    """Hash-bound chronological M1 bid/ask source for offline replay."""

    source_path: Path
    source_sha256: str
    source_size_bytes: int
    times: np.ndarray
    index: pd.Index
    mid_open: np.ndarray
    mid_high: np.ndarray
    mid_low: np.ndarray
    mid_close: np.ndarray
    bid_open: np.ndarray
    ask_open: np.ndarray
    bid_close: np.ndarray
    ask_close: np.ndarray
    bid_high: np.ndarray
    bid_low: np.ndarray
    ask_high: np.ndarray
    ask_low: np.ndarray
    volume: np.ndarray
    m5_bucket_index: pd.Index
    m5_bucket_first_indices: np.ndarray
    m5_bucket_last_indices: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "SourceTape":
        requested_path = Path(path).expanduser()
        if requested_path.is_symlink() or not requested_path.is_file():
            raise RuntimeError(
                f"source tape must be a regular non-symlinked file: {requested_path}"
            )
        source_path = requested_path.resolve(strict=True)
        stat_before = source_path.stat()
        columns = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "bid_open",
            "ask_open",
            "bid_close",
            "ask_close",
            "bid_high",
            "bid_low",
            "ask_high",
            "ask_low",
            "volume",
        ]
        source = pd.read_parquet(source_path, columns=columns)
        assert_no_price_scale_glitch(
            source,
            context="MODEL_NATIVE_REPLAY_SOURCE_TAPE",
        )
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat_after = source_path.stat()
        before_identity = (
            stat_before.st_dev,
            stat_before.st_ino,
            stat_before.st_size,
            stat_before.st_mtime_ns,
            stat_before.st_ctime_ns,
        )
        after_identity = (
            stat_after.st_dev,
            stat_after.st_ino,
            stat_after.st_size,
            stat_after.st_mtime_ns,
            stat_after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise RuntimeError(f"source tape changed while loading: {source_path}")
        if source.empty:
            raise RuntimeError(f"source tape is empty: {source_path}")
        source["time"] = pd.to_datetime(source["time"], utc=True, errors="coerce")
        if source["time"].isna().any():
            raise RuntimeError(f"source tape contains invalid time rows: {path}")
        if not source["time"].is_monotonic_increasing:
            raise RuntimeError(
                f"source tape rows are not strictly chronological: {path}"
            )
        source = source.reset_index(drop=True)
        if source["time"].duplicated().any():
            raise RuntimeError(f"source tape contains duplicate time rows: {path}")
        source_index = pd.Index(source["time"])
        if not source_index.equals(source_index.floor("min")):
            raise RuntimeError(f"source tape timestamps must be minute-aligned: {path}")
        m5_buckets = source_index.floor("5min")
        first_mask = ~m5_buckets.duplicated(keep="first")
        last_mask = ~m5_buckets.duplicated(keep="last")
        tape = cls(
            source_path=source_path,
            source_sha256=digest.hexdigest(),
            source_size_bytes=int(stat_after.st_size),
            times=source["time"].to_numpy(),
            index=source_index,
            mid_open=source["open"].to_numpy(np.float64),
            mid_high=source["high"].to_numpy(np.float64),
            mid_low=source["low"].to_numpy(np.float64),
            mid_close=source["close"].to_numpy(np.float64),
            bid_open=source["bid_open"].to_numpy(np.float64),
            ask_open=source["ask_open"].to_numpy(np.float64),
            bid_close=source["bid_close"].to_numpy(np.float64),
            ask_close=source["ask_close"].to_numpy(np.float64),
            bid_high=source["bid_high"].to_numpy(np.float64),
            bid_low=source["bid_low"].to_numpy(np.float64),
            ask_high=source["ask_high"].to_numpy(np.float64),
            ask_low=source["ask_low"].to_numpy(np.float64),
            volume=source["volume"].to_numpy(np.int64),
            m5_bucket_index=pd.Index(m5_buckets[first_mask]),
            m5_bucket_first_indices=np.flatnonzero(first_mask).astype(np.int64),
            m5_bucket_last_indices=np.flatnonzero(last_mask).astype(np.int64),
        )
        tape._require_shape_contract()
        return tape

    @property
    def source_binding(self) -> dict[str, Any]:
        """Return the exact source identity used by offline proof producers."""

        return {
            "path": str(self.source_path),
            "sha256": self.source_sha256,
            "size_bytes": self.source_size_bytes,
        }

    def _require_shape_contract(self) -> None:
        expected = len(self.times)
        if len(self.index) != expected or not self.index.is_unique:
            raise RuntimeError("source tape time index is not unique and shape-aligned")
        if (
            len(self.m5_bucket_index) == 0
            or not self.m5_bucket_index.is_unique
            or not self.m5_bucket_index.is_monotonic_increasing
            or len(self.m5_bucket_first_indices) != len(self.m5_bucket_index)
            or len(self.m5_bucket_last_indices) != len(self.m5_bucket_index)
            or np.any(self.m5_bucket_first_indices > self.m5_bucket_last_indices)
        ):
            raise RuntimeError("source tape observed-M5 bucket index is invalid")
        for name in (
            "mid_open",
            "mid_high",
            "mid_low",
            "mid_close",
            "bid_open",
            "ask_open",
            "bid_close",
            "ask_close",
            "bid_high",
            "bid_low",
            "ask_high",
            "ask_low",
            "volume",
        ):
            if len(getattr(self, name)) != expected:
                raise RuntimeError(
                    f"source tape column length mismatch: {name}={len(getattr(self, name))} "
                    f"expected={expected}"
                )
        price_arrays = {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in (
                "mid_open",
                "mid_high",
                "mid_low",
                "mid_close",
                "bid_open",
                "ask_open",
                "bid_close",
                "ask_close",
                "bid_high",
                "bid_low",
                "ask_high",
                "ask_low",
            )
        }
        if any(
            not np.isfinite(values).all() or np.any(values <= 0.0)
            for values in price_arrays.values()
        ):
            raise RuntimeError("source tape contains non-finite/non-positive prices")
        bid_open = price_arrays["bid_open"]
        ask_open = price_arrays["ask_open"]
        bid_close = price_arrays["bid_close"]
        ask_close = price_arrays["ask_close"]
        bid_high = price_arrays["bid_high"]
        bid_low = price_arrays["bid_low"]
        ask_high = price_arrays["ask_high"]
        ask_low = price_arrays["ask_low"]
        mid_open = price_arrays["mid_open"]
        mid_high = price_arrays["mid_high"]
        mid_low = price_arrays["mid_low"]
        mid_close = price_arrays["mid_close"]
        volume = np.asarray(self.volume)
        if volume.dtype.kind not in "iu" or np.any(volume < 0):
            raise RuntimeError("source tape contains invalid negative volume")
        invalid_geometry = (
            np.any(ask_open < bid_open)
            or np.any(ask_close < bid_close)
            or np.any(bid_low > np.minimum(bid_open, bid_close))
            or np.any(bid_high < np.maximum(bid_open, bid_close))
            or np.any(ask_low > np.minimum(ask_open, ask_close))
            or np.any(ask_high < np.maximum(ask_open, ask_close))
            or np.any(ask_low < bid_low)
            or np.any(ask_high < bid_high)
            or np.any(mid_low > np.minimum(mid_open, mid_close))
            or np.any(mid_high < np.maximum(mid_open, mid_close))
        )
        if invalid_geometry:
            raise RuntimeError("source tape contains invalid literal M/B/A OHLC geometry")

    def indices_for_times(self, sample_times: pd.Series) -> np.ndarray:
        self._require_shape_contract()
        parsed = pd.to_datetime(sample_times, utc=True, errors="coerce")
        if parsed.isna().any():
            raise RuntimeError(
                f"{int(parsed.isna().sum())} replay fill times are invalid"
            )
        indices = self.index.get_indexer(parsed)
        if np.any(indices < 0):
            raise RuntimeError(
                f"{int((indices < 0).sum())} replay fill times are missing from source tape"
            )
        return indices.astype(np.int64, copy=False)

    def get_closed_m1_bar(self, expected_m1: pd.Timestamp) -> dict[str, Any]:
        """Return one exact, hash-bound, complete closed M1 bar."""

        self._require_shape_contract()
        expected = pd.Timestamp(expected_m1)
        if expected.tzinfo is None:
            expected = expected.tz_localize("UTC")
        else:
            expected = expected.tz_convert("UTC")
        position = int(self.index.get_indexer([expected])[0])
        if position < 0:
            raise RuntimeError(f"source tape lacks exact closed M1 bar: {expected}")
        return canonical_closed_m1_bar(
            m1_bar_ts=expected,
            complete=True,
            source_path=str(self.source_path),
            source_sha256=self.source_sha256,
            bid_open=float(self.bid_open[position]),
            bid_high=float(self.bid_high[position]),
            bid_low=float(self.bid_low[position]),
            bid_close=float(self.bid_close[position]),
            ask_open=float(self.ask_open[position]),
            ask_high=float(self.ask_high[position]),
            ask_low=float(self.ask_low[position]),
            ask_close=float(self.ask_close[position]),
            mid_open=float(self.mid_open[position]),
            mid_high=float(self.mid_high[position]),
            mid_low=float(self.mid_low[position]),
            mid_close=float(self.mid_close[position]),
            volume=int(self.volume[position]),
        )

    def get_open_quote(self, exact_minute: pd.Timestamp) -> dict[str, Any]:
        """Return the exact bid/ask open quote for one offline replay step."""

        self._require_shape_contract()
        expected = pd.Timestamp(exact_minute)
        if expected.tzinfo is None:
            expected = expected.tz_localize("UTC")
        else:
            expected = expected.tz_convert("UTC")
        position = int(self.index.get_indexer([expected])[0])
        if position < 0:
            raise RuntimeError(f"source tape lacks exact open quote: {expected}")
        return {
            "time": expected,
            "bid": float(self.bid_open[position]),
            "ask": float(self.ask_open[position]),
        }

    def label_horizon_indices(
        self,
        *,
        decision_time: pd.Timestamp,
        horizon_m5_bars: int,
    ) -> tuple[int, int]:
        """Resolve an M5 label horizon on the authoritative observed M1 clock."""

        self._require_shape_contract()
        decision = pd.Timestamp(decision_time)
        if decision.tzinfo is None:
            decision = decision.tz_localize("UTC")
        else:
            decision = decision.tz_convert("UTC")
        if pd.isna(decision) or decision != decision.floor("5min"):
            raise RuntimeError(
                f"label-horizon decision must be an exact M5 timestamp: {decision_time!r}"
            )
        if (
            isinstance(horizon_m5_bars, bool)
            or not isinstance(horizon_m5_bars, (int, np.integer))
            or int(horizon_m5_bars) <= 0
        ):
            raise RuntimeError(
                f"M5 label horizon must be a positive integer: {horizon_m5_bars!r}"
            )
        decision_bucket = int(self.m5_bucket_index.get_indexer([decision])[0])
        if decision_bucket < 0:
            raise RuntimeError(f"source tape lacks decision M5 bucket: {decision}")
        target_bucket = decision_bucket + int(horizon_m5_bars)
        if target_bucket >= len(self.m5_bucket_index):
            raise RuntimeError(
                "source tape does not cover the full observed-M5 label horizon: "
                f"decision={decision} horizon={int(horizon_m5_bars)}"
            )
        fill_time = decision + pd.Timedelta(minutes=5)
        fill_index = int(self.index.get_indexer([fill_time])[0])
        if fill_index < 0:
            raise RuntimeError(
                f"source tape lacks exact T+5 Entry fill row: {fill_time}"
            )
        exit_index = int(self.m5_bucket_last_indices[target_bucket])
        if exit_index < fill_index:
            raise RuntimeError(
                "observed-M5 label horizon ends before the exact Entry fill: "
                f"fill_index={fill_index} exit_index={exit_index}"
            )
        return fill_index, exit_index
