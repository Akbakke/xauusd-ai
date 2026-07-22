#!/usr/bin/env python3
"""V12 live pipeline orchestrator.

Single entry point for the model-native XAU serving stack:
    ENTRY (per closed M5 row):
        cv3+BASE28 prebuilts → ModelNativeStateBuilder (exact 513 signals)
        + 142 continuous / 5 categorical context fields
        → contract-resolved, calibrated model-native Entry bundle
        → final model LONG/SHORT/FLAT argmax → SKIP/TAKE
    EXIT (per M1, unchanged joint-replay-proven chain):
        XGB bridge (M5, asof) + V3 exit transformer + Exit-IQL (+ overlays)
                                                          ↓ if TAKE
                                                       open TradeState
                                                          ↓ per-M1
                                                       Exit-IQL V12.1 → HOLD/EXIT
The legacy XGB→V10→Entry-IQL entry chain is RETIRED (2026-07-05; bundles
physically gone 2026-07-07) — v10/entry_iql fields remain only for offline
replay drivers that construct the pipeline explicitly.

Encapsulates model loading (~300 ms one-time at startup) and provides
two main inference methods:

  .make_entry_decision(now_minute, bid, ask)
      No open trade → returns SKIP / TAKE_LONG_NOW / TAKE_SHORT_NOW
      from the model's exact LONG/SHORT/FLAT argmax plus diagnostics.

  .make_exit_decision(trade, now_minute, bid, ask, m1_close=None)
      Open trade → advances trade state, returns HOLD / EXIT_NOW
      with the full bar_state (for journaling). Any supplied m1_close is a
      parity assertion only; the exact collector bar remains authoritative.

Used by v12_paper_runner.py to drive live trade decisions.

"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC,
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    ModelNativeRuntimeEvidenceError,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CONTRACT_MODE
from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
from gx1.execution.v12_xgb_live import XGBLiveInference as ExitXGBLiveInference
from gx1.execution.v12_model_native_state_live import SEQ_LEN_MODEL_NATIVE as ENTRY_SEQ_LEN
from gx1.execution.v12_exit_iql_live import ExitIQLLiveInference, let_winners_run_hold
from gx1.execution.v12_v3_live import V3LiveInference
from gx1.execution.v12_trade_state import TradeState

LOG = logging.getLogger("v12_pipeline")


# HARD MAE-STOP (risk overlay, 2026-06-17, default 0 = OFF). The learned Exit-IQL does NOT hard-cap
# adverse excursion — it holds through deep MAE to scratch a win, keeping a 95% win-rate but a brutal
# left tail (baseline worst single trade −416 bps MAE; it IS the bulk of the cap-3 account DD). User
# vedtak: stop "tåle 500 i minus i 8 timer for 16 i pluss". When the live unrealized PnL drops to
# −GX1_EXIT_HARD_STOP_BPS, force EXIT_NOW regardless of the Exit-IQL action. NOT gated by PURE_PHASE6
# (a risk overlay we WANT live). Validated (OOT baseline hard-stop sim): −80 caps every trade at −80
# for −1.7% total PnL (−120 = −0.8%); bilateral; the marginal/live trades it most protects have worse
# MAE so the live benefit is larger. Reversible: GX1_EXIT_HARD_STOP_BPS=0.
_EXIT_HARD_STOP_BPS = float(os.environ.get("GX1_EXIT_HARD_STOP_BPS", "0") or "0")


COLLECTOR_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
CANONICAL_M1_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
ENTRY_DECISION_AVAILABILITY_LAG = pd.Timedelta(minutes=5)
# Immutable live Entry freshness contract.  The replay fill convention makes an
# M5 row available at T+5; a materially later live fill is a different trade.
# These values are deliberately not environment-configurable. The runtime
# evidence contract is the single numeric owner of the 90-second limit.
ENTRY_MAX_DECISION_LATENCY_SEC = float(MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC)
ENTRY_MAX_CANONICAL_CUTOFF_AGE_SEC = (
    ENTRY_DECISION_AVAILABILITY_LAG.total_seconds()
    + ENTRY_MAX_DECISION_LATENCY_SEC
)


class EntryDecisionUnavailable(RuntimeError):
    """No model direction exists for this poll; never synthesize FLAT/SKIP."""

    def __init__(self, reason: str, **evidence: Any):
        self.reason = str(reason)
        self.evidence = dict(evidence)
        super().__init__(f"{self.reason}: {self.evidence}")


class ExitDecisionUnavailable(RuntimeError):
    """No authoritative Exit model decision exists for this M1 cadence step.

    The exception is deliberately distinct from HOLD.  Missing market state,
    stale bars, or failed model inputs may not be converted into an Exit action.
    Fresh broker quotes remain available to the hard-stop safety path before
    this exception is raised.
    """

    def __init__(self, reason: str, **evidence: Any):
        self.reason = str(reason)
        self.evidence = dict(evidence)
        super().__init__(f"{self.reason}: {self.evidence}")


REQUIRED_V3_DECISION_FIELDS = (
    "v3_v8_should_exit_prob",
    "v3_v8_profit_protect_prob",
    "v3_v8_family_argmax",
    "v3_v8_family_logit_max",
)


def _utc_ts(ts: pd.Timestamp | Any) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _latest_closed_m5_start(now_minute: pd.Timestamp) -> pd.Timestamp:
    """Return the latest M5 bar-start whose OHLC is closed at this wall-clock minute."""
    return _utc_ts(now_minute).floor("5min") - ENTRY_DECISION_AVAILABILITY_LAG


def _exact_closed_m5_row(
    augmented: pd.DataFrame,
    now_minute: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Series]:
    """Return the unique canonical row for the latest actually closed M5 bar."""
    expected = _latest_closed_m5_start(now_minute)
    if augmented is None or augmented.empty:
        raise ExitDecisionUnavailable(
            "canonical_window_empty",
            expected_m5=str(expected),
        )
    try:
        observed_index = pd.to_datetime(augmented.index, utc=True, errors="coerce")
    except Exception as exc:  # noqa: BLE001 - convert to structured unavailable evidence
        raise ExitDecisionUnavailable(
            "canonical_index_invalid",
            expected_m5=str(expected),
            error_type=type(exc).__name__,
        ) from exc
    positions = np.flatnonzero(observed_index == expected)
    if len(positions) != 1:
        valid = observed_index[~pd.isna(observed_index)]
        raise ExitDecisionUnavailable(
            "canonical_exact_m5_missing" if len(positions) == 0 else "canonical_exact_m5_duplicate",
            expected_m5=str(expected),
            matches=int(len(positions)),
            latest_observed_m5=str(valid.max()) if len(valid) else "",
        )
    row = augmented.iloc[int(positions[0])]
    if isinstance(row, pd.DataFrame):
        raise ExitDecisionUnavailable(
            "canonical_exact_m5_duplicate",
            expected_m5=str(expected),
            matches=int(len(row)),
        )
    return expected, row


def _validated_v3_output(v3_v8_out: Any, *, q_head_required: bool) -> dict[str, Any]:
    if not isinstance(v3_v8_out, dict):
        raise ExitDecisionUnavailable(
            "v3_output_invalid",
            observed_type=type(v3_v8_out).__name__,
        )
    required = list(REQUIRED_V3_DECISION_FIELDS)
    if q_head_required:
        required.extend(("v3_q_hold_v1", "v3_q_exit_v1"))
    missing = [name for name in required if name not in v3_v8_out]
    if missing:
        raise ExitDecisionUnavailable("v3_output_missing", missing_fields=missing)
    invalid: list[str] = []
    for name in required:
        try:
            value = float(v3_v8_out[name])
        except (TypeError, ValueError):
            invalid.append(name)
            continue
        if not np.isfinite(value):
            invalid.append(name)
    if invalid:
        raise ExitDecisionUnavailable("v3_output_non_finite", invalid_fields=invalid)
    for name in ("v3_v8_should_exit_prob", "v3_v8_profit_protect_prob"):
        value = float(v3_v8_out[name])
        if not 0.0 <= value <= 1.0:
            raise ExitDecisionUnavailable(
                "v3_output_probability_out_of_range",
                field=name,
                value=value,
            )
    family = float(v3_v8_out["v3_v8_family_argmax"])
    if family not in (0.0, 1.0, 2.0, 3.0):
        raise ExitDecisionUnavailable(
            "v3_output_family_invalid",
            value=family,
        )
    return v3_v8_out


def _entry_decision_latency_fields(
    now_minute: pd.Timestamp,
    decision_m5: pd.Timestamp,
) -> dict[str, Any]:
    """Live/replay parity fields for a smart entry decision.

    Smart rows are labeled by the M5 bar start T. The earliest replay fill is
    the M1 open at T+5, so a live decision made materially later is a different
    trade and must not be silently executed as if it were the replay fill.
    """
    now_ts = _utc_ts(now_minute)
    decision_ts = _utc_ts(decision_m5)
    available_ts = decision_ts + ENTRY_DECISION_AVAILABILITY_LAG
    latency_sec = (now_ts - available_ts).total_seconds()
    fields: dict[str, Any] = {
        "decision_ts": str(decision_ts),
        "decision_available_ts": str(available_ts),
        "entry_signal_latency_sec": float(latency_sec),
        "entry_signal_latency_min": float(latency_sec / 60.0),
        "entry_signal_latency_cap_sec": ENTRY_MAX_DECISION_LATENCY_SEC,
        "entry_signal_stale": bool(latency_sec > ENTRY_MAX_DECISION_LATENCY_SEC),
    }
    return fields


@dataclass
class V12Pipeline:
    prebuilt_loader: PrebuiltStateLoader
    # Required solely by the separately admitted V3/Exit stack.  It is never
    # passed to model-native Entry state, inference, or direction selection.
    exit_xgb: ExitXGBLiveInference
    exit_iql: "ExitIQLLiveInference | None" = None
    v3: V3LiveInference | None = None     # V3 v8 — used for exit decisions
    # SMART entry adapter. It loads only when the artifact selector and newest
    # XAU direction launch contract admit the exact hashed bundle.
    smart_entry: "object | None" = None
    _last_smart_bucket: pd.Timestamp | None = None
    # Cache for the most recent model-native augmented window (refreshed per M5).
    # XGB remains available to exit/V3 only and never enters Entry direction.
    _last_augmented_bucket: pd.Timestamp | None = None
    _last_augmented: pd.DataFrame | None = None
    # Exact closed-M1 cache.  A cache hit is admitted only when its timestamp is
    # the unique expected bar for this cadence step; an older cached bar is never
    # a substitute for missing collector state.
    _last_m1_atr_minute: pd.Timestamp | None = None
    # V4 (R13): full intrabar OHLC of the latest CLOSED M1 bar (one source for the
    # V3 overlay's intrabar peak/trough/atr AND current_atr_bps_v1).
    _last_m1_bar: dict | None = None
    @classmethod
    def load_default(cls) -> "V12Pipeline":
        """Fail-closed SMART serving stack.

        Entry is the exact contract-admitted model-native bundle via
        SmartEntryLiveInference. The retired V10→Entry-IQL direction chain is
        absent. Exit remains the separately contract-resolved stack.
        """
        t0 = time.perf_counter()
        loader = PrebuiltStateLoader()
        loader.load()
        exit_xgb = ExitXGBLiveInference.load_default()
        from gx1.execution.v12_smart_entry_live import SmartEntryLiveInference, assert_smart_serving_gate
        assert_smart_serving_gate()
        smart_entry = SmartEntryLiveInference.load()
        exit_iql = ExitIQLLiveInference.load_default()
        v3 = V3LiveInference.load_default()   # V3 v8 exit transformer
        # exit-side multi-TF tables (V3) — loader-owned, refreshed by its async cycle
        if getattr(v3, "_enable_multi_tf", False):
            LOG.info("V12.2: building multi-TF features on PrebuiltStateLoader (one-time)")
            loader.build_multi_tf_features()
        # entry-side smart context (float32 MTF + full-frame overrides) — heavy
        # (~2 min); built BLOCKING once here (mandatory initial snapshot). Every
        # later cv3 cutoff advance refreshes it in a BACKGROUND thread
        # (predict_live_bar / maybe_schedule_ctx_refresh — serving-wave gap 3),
        # so neither entry nor the per-M1 exit loop ever stalls on it again.
        smart_entry.refresh_multi_tf(loader._cv3)
        LOG.info(f"V12Pipeline loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
        LOG.info(
            "  prebuilt cutoff: %s  entry=%s "
            "exit_mtf=%s",
            loader.cutoff_ts,
            MODEL_NATIVE_CONTRACT_MODE,
            getattr(v3, "_enable_multi_tf", False),
        )
        return cls(
            prebuilt_loader=loader,
            exit_xgb=exit_xgb,
            exit_iql=exit_iql,
            v3=v3,
            smart_entry=smart_entry,
        )

    def _refresh_m1_bar(self, now_minute: pd.Timestamp) -> dict[str, Any]:
        """Load the unique latest CLOSED M1 bar or fail closed with evidence.

        This exact bar is the one truth for bid/ask closes, intrabar OHLC, and
        ``current_atr_bps_v1``.  Neither an older cache entry nor a latest-row
        lookup is permitted to fabricate the missing cadence state.
        """
        cur_min = _utc_ts(now_minute).floor("min")
        expected = cur_min - pd.Timedelta(minutes=1)
        if self._last_m1_atr_minute == expected and self._last_m1_bar is not None:
            exact_cached_bar = self._last_m1_bar
            return exact_cached_bar

        path = COLLECTOR_DIR / f"xauusd_m1_{expected.strftime('%Y%m%d')}.parquet"
        if not path.is_file():
            raise ExitDecisionUnavailable(
                "closed_m1_file_missing",
                expected_m1=str(expected),
                path=str(path),
            )
        columns = (
            "time",
            "bid_high",
            "bid_low",
            "ask_high",
            "ask_low",
            "ask_close",
            "bid_close",
        )
        try:
            df = pd.read_parquet(path, columns=list(columns))
            observed_times = pd.to_datetime(df["time"], utc=True, errors="coerce")
        except Exception as exc:  # noqa: BLE001 - convert I/O/schema failure to evidence
            raise ExitDecisionUnavailable(
                "closed_m1_read_failed",
                expected_m1=str(expected),
                path=str(path),
                error_type=type(exc).__name__,
            ) from exc

        positions = np.flatnonzero(observed_times == expected)
        if len(positions) != 1:
            valid = observed_times[~pd.isna(observed_times)]
            raise ExitDecisionUnavailable(
                "closed_m1_exact_bar_missing" if len(positions) == 0 else "closed_m1_exact_bar_duplicate",
                expected_m1=str(expected),
                matches=int(len(positions)),
                latest_observed_m1=str(valid.max()) if len(valid) else "",
                path=str(path),
            )

        row = df.iloc[int(positions[0])]
        price_names = columns[1:]
        prices: dict[str, float] = {}
        invalid_fields: list[str] = []
        for name in price_names:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                invalid_fields.append(name)
                continue
            if not np.isfinite(value) or value <= 0.0:
                invalid_fields.append(name)
            else:
                prices[name] = value
        if invalid_fields:
            raise ExitDecisionUnavailable(
                "closed_m1_prices_invalid",
                expected_m1=str(expected),
                invalid_fields=invalid_fields,
                path=str(path),
            )

        range_invalid = (
            prices["bid_low"] > prices["bid_close"]
            or prices["bid_close"] > prices["bid_high"]
            or prices["ask_low"] > prices["ask_close"]
            or prices["ask_close"] > prices["ask_high"]
            or prices["ask_close"] < prices["bid_close"]
        )
        if range_invalid:
            raise ExitDecisionUnavailable(
                "closed_m1_ohlc_invalid",
                expected_m1=str(expected),
                prices=prices,
                path=str(path),
            )

        mid_close = (prices["ask_close"] + prices["bid_close"]) / 2.0
        atr_bps = (prices["ask_high"] - prices["bid_low"]) / mid_close * 1e4
        if not np.isfinite(atr_bps) or atr_bps <= 0.0:
            raise ExitDecisionUnavailable(
                "closed_m1_atr_invalid",
                expected_m1=str(expected),
                atr_bps=float(atr_bps),
                path=str(path),
            )

        bar: dict[str, Any] = {
            "time": expected,
            **prices,
            "mid_close": float(mid_close),
            "atr_bps": float(atr_bps),
        }
        self._last_m1_bar = bar
        self._last_m1_atr_minute = expected
        return bar

    # ── canonical_v3 builds (cached per M5 bucket) ─────────────────────

    def _refresh_entry_canonical(self, now_minute: pd.Timestamp) -> None:
        """Load the exact latest closed M5 window under the live Entry contract.

        Entry freshness is intentionally separate from ``_refresh_exit_canonical``:
        the latter retains the independently admitted Exit stack's historical
        staleness semantics. Entry has no runtime knob, no cutoff clipping to
        an older row, and no boolean soft-unavailable path. Every violation is
        structured evidence that no model direction exists for this poll.
        """
        now_ts = _utc_ts(now_minute)
        expected_m5 = _latest_closed_m5_start(now_ts)
        try:
            changed = bool(self.prebuilt_loader.refresh_if_changed())
        except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence
            raise EntryDecisionUnavailable(
                "entry_canonical_refresh_failed",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
                error_type=type(exc).__name__,
            ) from exc
        if changed:
            self._last_augmented_bucket = None
            self._last_augmented = None

        try:
            raw_cutoff = getattr(self.prebuilt_loader, "cutoff_ts", None)
        except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_unavailable",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
                error_type=type(exc).__name__,
            ) from exc
        if raw_cutoff is None:
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_missing",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
            )
        try:
            cutoff = _utc_ts(raw_cutoff)
        except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_invalid",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
                cutoff=repr(raw_cutoff),
                error_type=type(exc).__name__,
            ) from exc
        if pd.isna(cutoff):
            raise EntryDecisionUnavailable(
                "entry_canonical_cutoff_invalid",
                now_minute=str(now_ts),
                expected_m5=str(expected_m5),
                cutoff=repr(raw_cutoff),
            )

        cutoff_age_sec = float((now_ts - cutoff).total_seconds())
        freshness_evidence = {
            "now_minute": str(now_ts),
            "expected_m5": str(expected_m5),
            "canonical_cutoff": str(cutoff),
            "canonical_cutoff_age_sec": cutoff_age_sec,
            "canonical_cutoff_age_cap_sec": ENTRY_MAX_CANONICAL_CUTOFF_AGE_SEC,
        }
        if cutoff_age_sec > ENTRY_MAX_CANONICAL_CUTOFF_AGE_SEC:
            raise EntryDecisionUnavailable(
                "entry_canonical_stale",
                **freshness_evidence,
            )
        if cutoff < expected_m5:
            raise EntryDecisionUnavailable(
                "entry_latest_closed_m5_unavailable",
                **freshness_evidence,
            )

        cur_bucket = expected_m5.floor("5min")
        augmented = self._last_augmented
        if self._last_augmented_bucket != cur_bucket or augmented is None:
            try:
                augmented = self.prebuilt_loader.get_window(
                    expected_m5,
                    n_bars=ENTRY_SEQ_LEN,
                )
            except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence
                raise EntryDecisionUnavailable(
                    "entry_canonical_window_read_failed",
                    **freshness_evidence,
                    error_type=type(exc).__name__,
                ) from exc

        if augmented is None or augmented.empty:
            raise EntryDecisionUnavailable(
                "entry_canonical_window_empty",
                **freshness_evidence,
            )
        if len(augmented) != ENTRY_SEQ_LEN:
            raise EntryDecisionUnavailable(
                "entry_canonical_history_mismatch",
                **freshness_evidence,
                observed_bars=int(len(augmented)),
                required_bars=int(ENTRY_SEQ_LEN),
            )
        try:
            observed_index = pd.to_datetime(augmented.index, utc=True, errors="coerce")
        except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence
            raise EntryDecisionUnavailable(
                "entry_canonical_index_invalid",
                **freshness_evidence,
                error_type=type(exc).__name__,
            ) from exc
        if (
            observed_index.hasnans
            or not observed_index.is_monotonic_increasing
            or not observed_index.is_unique
        ):
            raise EntryDecisionUnavailable(
                "entry_canonical_index_invalid",
                **freshness_evidence,
                has_nat=bool(observed_index.hasnans),
                monotonic=bool(observed_index.is_monotonic_increasing),
                unique=bool(observed_index.is_unique),
            )
        observed_latest = observed_index[-1]
        if observed_latest != expected_m5:
            raise EntryDecisionUnavailable(
                "entry_canonical_exact_m5_missing",
                **freshness_evidence,
                observed_latest_m5=str(observed_latest),
            )

        self._last_augmented_bucket = cur_bucket
        self._last_augmented = augmented

    def _refresh_exit_canonical(self, now_minute: pd.Timestamp) -> bool:
        """Refresh the Exit stack's augmented window from disk prebuilt.

        This retains the separately admitted Exit stack's existing behavior and
        configurable operational staleness cap. Entry must use the strict,
        immutable ``_refresh_entry_canonical`` contract above.

        Returns True if data available, False only if prebuilt is empty or
        history insufficient (early-history edge cases).
        """
        # Hot-reload prebuilts from disk if incremental updater extended them.
        # Invalidates cached window if cutoff advanced.
        if self.prebuilt_loader.refresh_if_changed():
            self._last_augmented_bucket = None
            self._last_augmented = None
        cutoff = self.prebuilt_loader.cutoff_ts
        # FAIL-CLOSED staleness cap (2026-06-03 audit): the clip below silently decides on
        # stale features if the canonical-incremental daemon stalls/dies (now>cutoff frozen).
        # Refuse model inference when the prebuilt is older than the cap. Live-only by construction:
        # replay always has now<=cutoff so this never fires there.
        import os as _os
        _max_stale_min = float(_os.environ.get("GX1_MAX_PREBUILT_STALENESS_MIN", "30"))
        if cutoff is not None and now_minute > cutoff:
            _age_min = (now_minute - cutoff).total_seconds() / 60.0
            if _age_min > _max_stale_min:
                LOG.error(f"[PREBUILT_STALE] cutoff {cutoff} is {_age_min:.0f} min behind now "
                          f"{now_minute} (> {_max_stale_min} cap) — MODEL UNAVAILABLE (canonical daemon stalled?)")
                return False
        # Clip to latest CLOSED M5 start. A wall-clock poll at 12:07 must not
        # read the 12:05 row, which is unavailable until 12:10.
        latest_closed_m5 = _latest_closed_m5_start(now_minute)
        effective_ts = latest_closed_m5 if cutoff is None or latest_closed_m5 <= cutoff else cutoff
        cur_bucket = effective_ts.floor("5min")
        if self._last_augmented_bucket == cur_bucket and self._last_augmented is not None:
            return True

        # Read 96-bar window directly from canonical_v3 + BASE28 prebuilts.
        # Identical values to what V12 cascade trainings saw — no live recompute.
        augmented = self.prebuilt_loader.get_window(effective_ts, n_bars=ENTRY_SEQ_LEN)
        if augmented.empty:
            LOG.warning(f"prebuilt empty for {effective_ts} (cutoff={cutoff}) — system not ready")
            return False
        if len(augmented) < ENTRY_SEQ_LEN:
            LOG.warning(f"only {len(augmented)} bars (need {ENTRY_SEQ_LEN}) — early-history bar")
            return False

        self._last_augmented_bucket = cur_bucket
        self._last_augmented = augmented
        return True

    # ── entry decision ────────────────────────────────────────────────

    def make_entry_decision(
        self,
        now_minute: pd.Timestamp,
        bid: float,
        ask: float,
    ) -> dict[str, Any]:
        """Run the contract-admitted model-native Entry policy for the latest
        closed M5 bar.

        One decision per M5 row (the replay cadence): prediction row time T =
        M5 bar-START label; the row lands in cv3 after the bar closes at T+5,
        so decision availability ≈ T+5 (+ daemon latency) and the live fill is
        the first quote after that — the operating point's 'M1 open at T+5'
        convention (RUN_MANIFEST entry_fill_convention; live pays real latency
        slippage vs the replay's exact T+5 open).

        Returns a model decision only when a new closed M5 state was scored:
            action: SKIP / TAKE_LONG_NOW / TAKE_SHORT_NOW
            model_direction: LONG / SHORT / FLAT
            action_id / edge_score / p_long / p_short / p_flat / session
            decision_ts: ISO timestamp of the M5 row decided
            _v10_snapshot: exit-bound diagnostic head snapshot

        Operational no-data/stale/cadence states raise EntryDecisionUnavailable;
        they are not allowed to masquerade as a model FLAT/SKIP decision.
        """
        if self.smart_entry is None:
            # fail-closed: the legacy V10->Entry-IQL chain is RETIRED (bundles
            # physically gone); a pipeline without smart_entry must never trade.
            raise RuntimeError(
                "[SMART_ENTRY] V12Pipeline has no smart_entry adapter — the legacy entry "
                "chain is RETIRED; construct via load_default() or pass smart_entry."
            )
        self._refresh_entry_canonical(now_minute)

        augmented = self._last_augmented

        # The accepted model-native model requires its exact 96-bar history.
        if len(augmented) != ENTRY_SEQ_LEN:
            raise EntryDecisionUnavailable(
                "canonical_history_mismatch",
                observed_bars=int(len(augmented)),
                required_bars=int(ENTRY_SEQ_LEN),
            )

        # ── SMART entry: one decision per NEW closed M5 row ──────────────────
        decision_m5 = augmented.index[-1]
        latency_fields = _entry_decision_latency_fields(
            now_minute,
            decision_m5,
        )
        if self._last_smart_bucket is not None and decision_m5 <= self._last_smart_bucket:
            raise EntryDecisionUnavailable("awaiting_new_m5_bar", **latency_fields)
        if latency_fields["entry_signal_stale"]:
            self._last_smart_bucket = decision_m5
            LOG.warning(
                "[ENTRY_SIGNAL_STALE] decision_m5=%s available=%s now=%s "
                "latency=%.0fs cap=%.0fs — MODEL UNAVAILABLE (no backlog execution)",
                decision_m5,
                latency_fields["decision_available_ts"],
                _utc_ts(now_minute),
                latency_fields["entry_signal_latency_sec"],
                ENTRY_MAX_DECISION_LATENCY_SEC,
            )
            raise EntryDecisionUnavailable("entry_signal_stale", **latency_fields)
        # Serving-wave gap 3: predict_live_bar NEVER blocks on the ~2-min smart-
        # context refresh (background thread + last-completed-snapshot, staleness
        # journaled as context_age_m5_bars) — the per-M1 exit loop is no longer
        # starved at every cv3 cutoff advance.
        from gx1.execution.v12_smart_entry_live import SmartContextStaleError
        try:
            head = self.smart_entry.predict_live_bar(self.prebuilt_loader, decision_m5)
        except SmartContextStaleError as exc:
            # fail-closed on rotten context (> GX1_SMART_CTX_MAX_STALENESS_M5 bars):
            # Emit no model direction and do NOT mark the bucket decided — retry until
            # the background refresh catches up (or the bar is superseded).
            LOG.warning(f"[SMART_ENTRY] {exc} — MODEL UNAVAILABLE (refresh in background; retry next poll)")
            raise EntryDecisionUnavailable(
                "smart_ctx_stale_refresh_pending",
                context_age_m5_bars=exc.age,
                context_cutoff_ts=str(exc.ctx_cutoff),
                **latency_fields,
            ) from exc
        except Exception as exc:  # noqa: BLE001 — fail closed, keep exits alive
            LOG.error(
                f"[SMART_ENTRY] state/forward failed for {decision_m5}: {exc} — "
                "no model direction emitted; exit management continues"
            )
            raise EntryDecisionUnavailable(
                "smart_entry_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                **latency_fields,
            ) from exc
        # exit-bound snapshot atr_bps = RAW live cv3 row value at T — the
        # joint-replay-proven convention (RUN_MANIFEST snapshot_policy), NOT the
        # state-builder's offline-derived atr_bps.
        if "atr_bps" not in augmented.columns:
            raise EntryDecisionUnavailable(
                "model_native_atr_missing",
                decision_m5=str(decision_m5),
                **latency_fields,
            )
        _atr_raw = float(pd.to_numeric(augmented.iloc[-1]["atr_bps"], errors="coerce"))
        if not np.isfinite(_atr_raw) or _atr_raw <= 0.0:
            raise EntryDecisionUnavailable(
                "model_native_atr_invalid",
                decision_m5=str(decision_m5),
                atr_bps=_atr_raw,
                **latency_fields,
            )
        try:
            decision = self.smart_entry.decide(head, atr_bps=_atr_raw)
            if not isinstance(decision, dict) or not decision:
                raise RuntimeError(
                    "model-native decision adapter returned no exact decision mapping"
                )
        except Exception as exc:  # noqa: BLE001 - no invalid head may become FLAT
            LOG.error(
                "[SMART_ENTRY] decision contract failed for %s: %s — "
                "no model direction emitted; exit management continues",
                decision_m5,
                exc,
            )
            raise EntryDecisionUnavailable(
                "model_native_direction_decision_invalid",
                decision_m5=str(decision_m5),
                error_type=type(exc).__name__,
                error=str(exc),
                **latency_fields,
            ) from exc
        decision.update(latency_fields)
        snapshot_raw = decision.get("_v10_snapshot")
        if not isinstance(snapshot_raw, dict) or not snapshot_raw:
            raise EntryDecisionUnavailable(
                "model_native_runtime_evidence_missing",
                decision_m5=str(decision_m5),
            )
        timing_evidence = {
            key: decision[key]
            for key in MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
            if key in decision
        }
        if set(timing_evidence) != set(
            MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
        ):
            raise EntryDecisionUnavailable(
                "model_native_timing_evidence_incomplete",
                missing_fields=sorted(
                    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
                    - set(timing_evidence)
                ),
                decision_m5=str(decision_m5),
            )
        executable_snapshot = dict(snapshot_raw)
        executable_snapshot.update(timing_evidence)
        try:
            executable_snapshot = require_model_native_runtime_evidence(
                executable_snapshot,
                context="V12_PIPELINE_ENTRY",
            )
        except ModelNativeRuntimeEvidenceError as exc:
            raise EntryDecisionUnavailable(
                "model_native_runtime_evidence_invalid",
                decision_m5=str(decision_m5),
                error=str(exc),
            ) from exc
        if decision.get("policy") != executable_snapshot["model_policy"]:
            raise EntryDecisionUnavailable(
                "model_native_policy_mismatch",
                decision_policy=decision.get("policy"),
                snapshot_policy=executable_snapshot["model_policy"],
            )
        decision["_v10_snapshot"] = executable_snapshot
        # Consume the M5 cadence slot only after ATR, model decision and the
        # complete runtime-evidence contract have all passed.  A transient
        # validation failure must remain retryable on this same fresh bar; it
        # is not a model decision and may not silently suppress the bar.
        self._last_smart_bucket = decision_m5
        # (Legacy V10->Entry-IQL forward, distilled-Q swap and the in-process
        # Entry-IQL shadow were REMOVED with the retired legacy chain — git
        # history holds the implementation; shadow_entry_iql stays None.)

        # The model is accepted once per newly closed M5 row.  `_last_smart_bucket`
        # above is cadence/state freshness, not a direction override; no rule-based
        # LONG/SHORT→FLAT mutation is permitted after the model argmax.
        return decision

    # ── exit decision ────────────────────────────────────────────────

    def make_exit_decision(
        self,
        trade: TradeState,
        now_minute: pd.Timestamp,
        bid: float,
        ask: float,
        m1_close: float | None = None,
    ) -> dict[str, Any]:
        """Return one contract-complete Exit decision for the exact closed M1.

        Missing/stale market state and inference failures raise
        :class:`ExitDecisionUnavailable`; they are not synthetic HOLD actions.
        The only pre-model action is the fresh-quote hard stop, retained solely
        to close an already-open position when model inputs are unavailable.
        """
        now_ts = _utc_ts(now_minute).floor("min")
        try:
            quote_bid = float(bid)
            quote_ask = float(ask)
        except (TypeError, ValueError) as exc:
            raise ExitDecisionUnavailable(
                "fresh_quote_invalid",
                now_minute=str(now_ts),
                bid=repr(bid),
                ask=repr(ask),
            ) from exc
        if (
            not np.isfinite(quote_bid)
            or not np.isfinite(quote_ask)
            or quote_bid <= 0.0
            or quote_ask <= 0.0
            or quote_ask < quote_bid
        ):
            raise ExitDecisionUnavailable(
                "fresh_quote_invalid",
                now_minute=str(now_ts),
                bid=quote_bid,
                ask=quote_ask,
            )

        quote_pnl_bps = float(trade._pnl_bps(quote_bid, quote_ask))
        if not np.isfinite(quote_pnl_bps):
            raise ExitDecisionUnavailable(
                "fresh_quote_pnl_invalid",
                now_minute=str(now_ts),
                bid=quote_bid,
                ask=quote_ask,
            )

        # Execution-risk recovery, not model state: a fresh broker quote may
        # trigger the configured catastrophe floor even if collector/model state
        # is unavailable.  Do not fabricate a closed M1 bar or advance cadence.
        if _EXIT_HARD_STOP_BPS > 0.0 and quote_pnl_bps <= -_EXIT_HARD_STOP_BPS:
            trade.current_bid = quote_bid
            trade.current_ask = quote_ask
            trade.current_pnl_bps = quote_pnl_bps
            trade.cum_mae_bps = min(float(trade.cum_mae_bps), quote_pnl_bps)
            LOG.info(f"[HARD_MAE_STOP] {trade.side} trade_id={getattr(trade, 'trade_id', '?')} "
                     f"pnl={quote_pnl_bps:+.1f}bps <= -{_EXIT_HARD_STOP_BPS:.0f} → force EXIT_NOW "
                     f"pre-canonical (bars={trade.bars_in_trade}, mae={trade.cum_mae_bps:+.1f})")
            return {
                "action": "EXIT_NOW", "action_id": 1, "stub": False,
                "decision_source": "HARD_MAE_STOP",
                "decision_safety_scope": "fresh_quote_existing_position_close",
                "bars_in_trade": int(trade.bars_in_trade),
                "current_pnl_bps": quote_pnl_bps,
            }

        m1_bar = self._refresh_m1_bar(now_ts)
        authoritative_m1_close = float(m1_bar["mid_close"])
        if m1_close is not None:
            try:
                supplied_m1_close = float(m1_close)
            except (TypeError, ValueError) as exc:
                raise ExitDecisionUnavailable(
                    "closed_m1_close_invalid",
                    expected_m1=str(m1_bar["time"]),
                    supplied=repr(m1_close),
                ) from exc
            if not np.isfinite(supplied_m1_close) or not np.isclose(
                supplied_m1_close,
                authoritative_m1_close,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ExitDecisionUnavailable(
                    "closed_m1_close_mismatch",
                    expected_m1=str(m1_bar["time"]),
                    supplied=supplied_m1_close,
                    authoritative=authoritative_m1_close,
                )

        try:
            canonical_ready = self._refresh_exit_canonical(now_ts)
        except ExitDecisionUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - structured no-decision evidence
            raise ExitDecisionUnavailable(
                "canonical_refresh_failed",
                now_minute=str(now_ts),
                expected_m5=str(_latest_closed_m5_start(now_ts)),
                decision_m1=str(m1_bar["time"]),
                error_type=type(exc).__name__,
            ) from exc
        if not canonical_ready:
            raise ExitDecisionUnavailable(
                "canonical_data_unavailable",
                now_minute=str(now_ts),
                expected_m5=str(_latest_closed_m5_start(now_ts)),
                decision_m1=str(m1_bar["time"]),
                cutoff=str(getattr(self.prebuilt_loader, "cutoff_ts", "")),
            )

        augmented = self._last_augmented
        decision_m5, cv3_row = _exact_closed_m5_row(augmented, now_ts)

        # last_atr_bps = latest M5 atr (journal/diagnostic + from_dict backfill only).
        # NOTE: the V3 overlay's atr_bps_now is NO LONGER sourced here — V4 records a
        # per-M1-bar atr in update_bar (intrabar (ask_high-bid_low)/mid), the one-truth
        # builder basis. This M5 value is kept for the per-bar journal field.
        if "atr_bps" not in cv3_row.index:
            raise ExitDecisionUnavailable(
                "canonical_atr_missing",
                decision_m5=str(decision_m5),
            )
        try:
            canonical_atr_bps = float(cv3_row["atr_bps"])
        except (TypeError, ValueError) as exc:
            raise ExitDecisionUnavailable(
                "canonical_atr_invalid",
                decision_m5=str(decision_m5),
                value=repr(cv3_row["atr_bps"]),
            ) from exc
        if not np.isfinite(canonical_atr_bps) or canonical_atr_bps <= 0.0:
            raise ExitDecisionUnavailable(
                "canonical_atr_invalid",
                decision_m5=str(decision_m5),
                value=canonical_atr_bps,
            )

        if self.v3 is None:
            raise ExitDecisionUnavailable("v3_not_loaded", decision_m1=str(m1_bar["time"]))
        if self.exit_iql is None:
            raise ExitDecisionUnavailable("exit_iql_not_loaded", decision_m1=str(m1_bar["time"]))

        base_m1 = getattr(self.prebuilt_loader, "_base28", None)
        if base_m1 is None or not isinstance(base_m1, pd.DataFrame) or base_m1.empty:
            raise ExitDecisionUnavailable(
                "v3_base_m1_unavailable",
                decision_m1=str(m1_bar["time"]),
            )
        try:
            base_m1_index = pd.to_datetime(base_m1.index, utc=True, errors="coerce")
        except Exception as exc:  # noqa: BLE001 - structured contract evidence
            raise ExitDecisionUnavailable(
                "v3_base_m1_index_invalid",
                decision_m1=str(m1_bar["time"]),
                error_type=type(exc).__name__,
            ) from exc
        base_matches = int(np.count_nonzero(base_m1_index == m1_bar["time"]))
        if base_matches != 1:
            valid_base_index = base_m1_index[~pd.isna(base_m1_index)]
            raise ExitDecisionUnavailable(
                "v3_base_exact_m1_missing" if base_matches == 0 else "v3_base_exact_m1_duplicate",
                decision_m1=str(m1_bar["time"]),
                matches=base_matches,
                latest_observed_m1=str(valid_base_index.max()) if len(valid_base_index) else "",
            )

        # Advance only from the authoritative closed bar.  The fresh quote above
        # is risk telemetry; it must not become the model's M1 state.
        trade.update_bar(
            bid=float(m1_bar["bid_close"]),
            ask=float(m1_bar["ask_close"]),
            m1_close=authoritative_m1_close,
            bid_high=float(m1_bar["bid_high"]),
            bid_low=float(m1_bar["bid_low"]),
            ask_high=float(m1_bar["ask_high"]),
            ask_low=float(m1_bar["ask_low"]),
        )
        trade.last_atr_bps = canonical_atr_bps

        # Run V3 v8 inference with trade-state overlay (B3 wire-up)
        try:
            overlay = trade.build_v3_overlay() if trade.bars_in_trade > 0 else None
            # V12.2: fetch multi-TF windows if V3 bundle requires them
            v3_mtf_windows = None
            if getattr(self.v3, "_enable_multi_tf", False):
                v3_mtf_windows = self.prebuilt_loader.get_multi_tf_windows(pd.Timestamp(m1_bar["time"]))
                if not v3_mtf_windows:
                    raise ExitDecisionUnavailable(
                        "v3_multi_tf_unavailable",
                        decision_m1=str(m1_bar["time"]),
                    )
            v3_v8_out = self.v3.predict(
                end_ts=pd.Timestamp(m1_bar["time"]),
                base34_prebuilt=base_m1,
                canonical_v3_window=augmented,
                xgb_inferer=self.exit_xgb,
                trade_overlay=overlay,
                multi_tf_windows=v3_mtf_windows,
            )
            v3_v8_out = _validated_v3_output(
                v3_v8_out,
                q_head_required=bool(getattr(self.v3, "_enable_q_head", False)),
            )
        except ExitDecisionUnavailable:
            self._v3_fail_strikes = getattr(self, "_v3_fail_strikes", 0) + 1
            LOG.error(
                "[V3_DECISION_UNAVAILABLE] contract failure strike=%d decision_m1=%s",
                self._v3_fail_strikes,
                m1_bar["time"],
            )
            raise
        except Exception as exc:
            self._v3_fail_strikes = getattr(self, "_v3_fail_strikes", 0) + 1
            LOG.error(
                "[V3_DECISION_UNAVAILABLE] inference failure strike=%d decision_m1=%s: %s",
                self._v3_fail_strikes,
                m1_bar["time"],
                exc,
            )
            raise ExitDecisionUnavailable(
                "v3_inference_failed",
                decision_m1=str(m1_bar["time"]),
                decision_m5=str(decision_m5),
                error_type=type(exc).__name__,
                consecutive_failures=int(self._v3_fail_strikes),
            ) from exc
        self._v3_fail_strikes = 0
        # Update running V3 statistics only after the complete output contract
        # has passed; partial/NaN output must not contaminate the next bar.
        trade.update_v3(v3_v8_out)

        # C1 fix 2026-05-19: training uses per-M1-bar (ask_high - bid_low)/mid bps
        # (typical 3-7 bps), live had been using canonical M5 ATR14 (10-50 bps).
        # 10× distribution shift on a feature Exit-IQL depends on.
        m1_atr_bps = float(m1_bar["atr_bps"])
        try:
            rec, bar_state = self.exit_iql.decide_for_trade(
                trade,
                cv3_row,
                v3_v8_out=v3_v8_out,
                current_m1_atr_bps_override=m1_atr_bps,
                now_minute=pd.Timestamp(m1_bar["time"]),
            )
        except Exception as exc:  # noqa: BLE001 - never turn model failure into HOLD
            raise ExitDecisionUnavailable(
                "exit_iql_decision_failed",
                decision_m1=str(m1_bar["time"]),
                decision_m5=str(decision_m5),
                error_type=type(exc).__name__,
            ) from exc
        if not isinstance(bar_state, dict):
            raise ExitDecisionUnavailable(
                "exit_iql_bar_state_invalid",
                decision_m1=str(m1_bar["time"]),
                observed_type=type(bar_state).__name__,
            )
        # Inject the 7 V3-tracking running-stats (max-prob-since-entry, consecutive-
        # exits, acceleration…) into bar_state. These are DERIVED running stats over
        # the V3 prob across the trade — DISTINCT from the 4 raw v3_v8_* outputs above
        # (which ARE in the cement). This is a belt-and-suspenders re-call of the same
        # values build_bar_state() already wrote (v12_exit_iql_live.py:433), kept as a
        # forward-compat hook: the CURRENT CLEAN exit cement was NOT built with these 7,
        # so the featurizer drops them (hence the benign [EXIT_IQL_V3_TRACKING_MISSING]
        # load warning) — train==serve holds. A future refit must define and prove
        # a new vedtak-gated dataset contract explicitly before consuming them.
        bar_state.update(trade.build_v3_tracking_features())
        # Surface raw Exit-IQL Q-values for diagnostics (was None before, made
        # debugging premature exits impossible). q_per_action_v1 = [q_hold, q_exit].
        try:
            iql_rec = rec.iql_recommendation_v1
            q_values = iql_rec.q_per_action_v1
            q_hold = float(q_values[0])
            q_exit = float(q_values[1])
            q_advantage = float(iql_rec.advantage_exit_over_hold_v1)
            raw_action_id = float(rec.action_id_v1)
            v3_exit_prob = float(rec.v3_should_exit_prob_v1)
            raw_action_label = str(rec.action_label_v1)
            raw_decision_source = str(rec.decision_source_v1)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ExitDecisionUnavailable(
                "exit_iql_output_invalid",
                decision_m1=str(m1_bar["time"]),
                error_type=type(exc).__name__,
            ) from exc
        output_values = (q_hold, q_exit, q_advantage, raw_action_id, v3_exit_prob)
        if not all(np.isfinite(value) for value in output_values):
            raise ExitDecisionUnavailable(
                "exit_iql_output_non_finite",
                decision_m1=str(m1_bar["time"]),
            )
        if raw_action_id not in (0.0, 1.0):
            raise ExitDecisionUnavailable(
                "exit_iql_action_invalid",
                decision_m1=str(m1_bar["time"]),
                action_id=raw_action_id,
                action_label=raw_action_label,
            )
        expected_exit_label = {0: "HOLD", 1: "EXIT_NOW"}[int(raw_action_id)]
        if raw_action_label != expected_exit_label:
            raise ExitDecisionUnavailable(
                "exit_iql_action_contract_mismatch",
                decision_m1=str(m1_bar["time"]),
                action_id=raw_action_id,
                action_label=raw_action_label,
                expected_label=expected_exit_label,
            )
        if not 0.0 <= v3_exit_prob <= 1.0 or not raw_decision_source:
            raise ExitDecisionUnavailable(
                "exit_iql_output_invalid",
                decision_m1=str(m1_bar["time"]),
                v3_should_exit_prob=v3_exit_prob,
                decision_source=raw_decision_source,
            )
        # Diagnostic: cast every bar_state value to a JSON-serializable scalar
        # so the runner can log the full 204-feature state to journal.
        bar_state_clean = {}
        for k, v in bar_state.items():
            try:
                bar_state_clean[k] = float(v)
            except (TypeError, ValueError):
                bar_state_clean[k] = str(v)
        # HARD MAE-STOP risk overlay (default-OFF). One-truth, applies live + in any pipeline replay.
        # If the trade is more than the stop underwater right now, force EXIT_NOW — caps the adverse
        # excursion so a position can never sit deep in the red grinding for a small scratch-win.
        _action_label = raw_action_label
        _action_id = int(raw_action_id)
        _decision_source = raw_decision_source
        # LET-WINNERS-RUN overlay (default-OFF; ONE-TRUTH with the phase6 gate): suppress a profit-EXIT_NOW
        # while the trade is in profit AND still near its MFE peak, so a winner rides until a real trailing
        # giveback (Strategy-F) / hard-stop closes it — addresses the held_too_short continuation-miss leak.
        if _action_id == 1 and let_winners_run_hold(float(trade.current_pnl_bps), float(trade.cum_mfe_bps)):
            _action_label = "HOLD"
            _action_id = 0
            _decision_source = "LET_WINNERS_RUN"
        if _EXIT_HARD_STOP_BPS > 0.0 and _action_id != 1 and float(trade.current_pnl_bps) <= -_EXIT_HARD_STOP_BPS:
            _action_label = "EXIT_NOW"
            _action_id = 1
            _decision_source = "HARD_MAE_STOP"
            LOG.info(f"[HARD_MAE_STOP] {trade.side} trade_id={getattr(trade, 'trade_id', '?')} "
                     f"pnl={trade.current_pnl_bps:+.1f}bps <= -{_EXIT_HARD_STOP_BPS:.0f} → force EXIT_NOW "
                     f"(bars={trade.bars_in_trade}, mae={trade.cum_mae_bps:+.1f})")
        return {
            "action": _action_label,
            "action_id": _action_id,
            "decision_source": _decision_source,
            "decision_m1": str(m1_bar["time"]),
            "decision_m5": str(decision_m5),
            "v3_should_exit_prob": v3_exit_prob,
            "v3_degraded": False,
            "q_hold": q_hold,
            "q_exit": q_exit,
            "q_advantage": q_advantage,
            "bar_state": bar_state_clean,
            "bars_in_trade": int(trade.bars_in_trade),
            "current_pnl_bps": float(trade.current_pnl_bps),
            "cum_mfe_bps": float(trade.cum_mfe_bps),
            "cum_mae_bps": float(trade.cum_mae_bps),
            "stub": False,
        }


# Backwards-compat module-level callable so existing paper-runner code
# that calls `make_v12_decision(features_snapshot)` can be lifted with
# minimal changes. The caller provides a singleton V12Pipeline.

_GLOBAL_PIPELINE: V12Pipeline | None = None


def get_global_pipeline() -> V12Pipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = V12Pipeline.load_default()
    return _GLOBAL_PIPELINE
