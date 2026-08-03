"""Exact offline execution primitive for immutable model-native Entry replay.

The persisted LONG/SHORT/FLAT argmax fixes direction before this module runs.
Every non-FLAT row is simulated independently as a unit-normalized price-path
diagnostic and exits only at its dataset label horizon.  This module does not
simulate orders or apply position size, and has no direction selector,
threshold, occupancy rule, risk filter, or early-exit variant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.io.price_glitch_guard import assert_no_price_scale_glitch
from gx1.models.entry_v10.direction_decision_contract import (
    canonical_closed_m1_bar,
)

LABEL_HORIZON_EXIT_MODE = "label_horizon"
OFFLINE_REPLAY_EXECUTION_CODE_PATH = "gx1/execution/model_native_entry_replay_v1.py"
OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE = "offline_direction_argmax_diagnostic"
UNIT_NORMALIZED_PNL_MODE = "unit_normalized_price_path_bps"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _policy_hash(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def label_horizon_exit_policy_config() -> dict[str, Any]:
    """Return the only permitted Entry replay exit configuration."""

    return {
        "schema_version": "entry_replay_label_horizon_exit_v3",
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "exit_mode": LABEL_HORIZON_EXIT_MODE,
        "label_horizon_timeframe": "M5",
        "source_row_clock": "consecutive_authoritative_closed_m1_source_rows",
        "exit_index_authority": "observed_m5_bucket_plus_label_horizon_bars",
        "row_simulation_mode": "independent",
        "one_trade_per_non_flat_argmax_row": True,
        "pnl_measurement_mode": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "filters_applied": False,
        "occupancy_filter_allowed": False,
        "cooldown_allowed": False,
        "max_trades_per_day_allowed": False,
        "daily_loss_limit_allowed": False,
        "take_profit_stop_loss_allowed": False,
        "mfe_protect_allowed": False,
        "invalid_path_skip_allowed": False,
    }


def label_horizon_exit_policy_contract() -> dict[str, Any]:
    """Return the immutable exact-exit contract embedded in replay evidence."""

    params = label_horizon_exit_policy_config()
    return {
        "schema_version": "entry_replay_label_horizon_exit_contract_v3",
        "offline_only": True,
        "promotion_shadow_live_allowed": False,
        "code_path": OFFLINE_REPLAY_EXECUTION_CODE_PATH,
        "params": params,
        "config_hash": _policy_hash(params),
    }


@dataclass(frozen=True)
class SourceTape:
    """Immutable chronological bid/ask tape for exact offline replay.

    The same object can feed the live Exit policy through its explicit
    ``get_closed_m1_bar`` seam. That keeps historical replay on the production
    policy path while binding every returned bar to one exact source file.
    """

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
        """Exact source identity for replay manifests and proof producers."""

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
        if (
            volume.dtype.kind not in "iu"
            or np.any(volume < 0)
        ):
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

    def get_closed_m1_bar(
        self,
        expected_m1: pd.Timestamp,
    ) -> dict[str, Any]:
        """Return the unique exact M1 bar required by ``V12Pipeline``."""

        self._require_shape_contract()
        expected = pd.Timestamp(expected_m1)
        if expected.tzinfo is None:
            expected = expected.tz_localize("UTC")
        else:
            expected = expected.tz_convert("UTC")
        position = int(self.index.get_indexer([expected])[0])
        if position < 0:
            raise RuntimeError(
                f"source tape lacks exact closed M1 bar: {expected}"
            )
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

    def get_open_quote(
        self,
        exact_minute: pd.Timestamp,
    ) -> dict[str, Any]:
        """Return the exact fresh bid/ask open quote for one replay step."""

        self._require_shape_contract()
        expected = pd.Timestamp(exact_minute)
        if expected.tzinfo is None:
            expected = expected.tz_localize("UTC")
        else:
            expected = expected.tz_convert("UTC")
        position = int(self.index.get_indexer([expected])[0])
        if position < 0:
            raise RuntimeError(
                f"source tape lacks exact open quote: {expected}"
            )
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
        """Resolve an M5 label horizon on the authoritative observed M1 clock.

        The dataset label horizon counts observed M5 bars, not M1 rows and not
        wall-clock minutes. Market closures therefore advance through the next
        observed M5 bucket without synthesizing missing source rows.
        """

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
            raise RuntimeError(
                f"source tape lacks decision M5 bucket: {decision}"
            )
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

    def simulate_label_horizon_trade(
        self,
        *,
        decision_time: pd.Timestamp,
        horizon_m5_bars: int,
        side: int,
    ) -> dict[str, Any]:
        start, end = self.label_horizon_indices(
            decision_time=decision_time,
            horizon_m5_bars=horizon_m5_bars,
        )
        return self._simulate_trade_indices(
            start=start,
            end=end,
            side=side,
            held_bars=int(horizon_m5_bars),
        )

    def simulate_trade(
        self,
        *,
        start_idx: int,
        horizon_bars: int,
        side: int,
    ) -> dict[str, Any]:
        """Simulate exactly one independent LONG/SHORT label-horizon trade.

        Invalid paths raise instead of returning a sentinel, because silently
        skipping a non-FLAT model decision would bias replay precision.
        """

        start = int(start_idx)
        horizon = int(horizon_bars)
        end = start + horizon
        return self._simulate_trade_indices(
            start=start,
            end=end,
            side=side,
            held_bars=horizon,
        )

    def _simulate_trade_indices(
        self,
        *,
        start: int,
        end: int,
        side: int,
        held_bars: int,
    ) -> dict[str, Any]:
        self._require_shape_contract()
        if side not in (0, 1):
            raise RuntimeError(f"replay trade side must be LONG=0 or SHORT=1: {side}")
        if start < 0 or start >= len(self.times):
            raise RuntimeError(f"replay start index is outside source tape: {start}")
        if held_bars <= 0:
            raise RuntimeError(f"label horizon must be positive: {held_bars}")
        if end < start or end >= len(self.times):
            raise RuntimeError(
                "source tape does not cover the full label horizon: "
                f"start={start} horizon={held_bars} end={end} rows={len(self.times)}"
            )

        future = slice(start, end + 1)
        if side == 0:
            entry_price = float(self.ask_open[start])
            exit_price = float(self.bid_close[end])
            favorable = np.asarray(self.bid_high[future], dtype=np.float64)
            adverse = np.asarray(self.bid_low[future], dtype=np.float64)
            gross_pnl_bps = (exit_price - entry_price) / entry_price * 1e4
            mfe_bps = (float(np.max(favorable)) - entry_price) / entry_price * 1e4
            mae_bps = (entry_price - float(np.min(adverse))) / entry_price * 1e4
        else:
            entry_price = float(self.bid_open[start])
            exit_price = float(self.ask_close[end])
            favorable = np.asarray(self.ask_low[future], dtype=np.float64)
            adverse = np.asarray(self.ask_high[future], dtype=np.float64)
            gross_pnl_bps = (entry_price - exit_price) / entry_price * 1e4
            mfe_bps = (entry_price - float(np.min(favorable))) / entry_price * 1e4
            mae_bps = (float(np.max(adverse)) - entry_price) / entry_price * 1e4

        if (
            not np.isfinite(entry_price)
            or entry_price <= 0.0
            or not np.isfinite(exit_price)
            or exit_price <= 0.0
            or not np.isfinite(favorable).all()
            or not np.isfinite(adverse).all()
            or bool((favorable <= 0.0).any())
            or bool((adverse <= 0.0).any())
            or not np.isfinite(gross_pnl_bps)
            or not np.isfinite(mfe_bps)
            or not np.isfinite(mae_bps)
        ):
            raise RuntimeError(
                "source tape contains an invalid price path for a non-FLAT row: "
                f"start={start} end={end} side={side}"
            )

        entry_time = pd.Timestamp(self.times[start])
        exit_time = pd.Timestamp(self.times[end])
        if pd.isna(entry_time) or pd.isna(exit_time):
            raise RuntimeError(
                f"source tape contains invalid timestamps: start={start} end={end}"
            )
        return {
            "entry_src_idx": start,
            "exit_src_idx": end,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl_bps": float(gross_pnl_bps),
            "mfe_bps": float(mfe_bps),
            "mae_bps": float(mae_bps),
            "held_bars": held_bars,
            "exit_reason": LABEL_HORIZON_EXIT_MODE,
        }
