"""ONE-TRUTH regime-adaptive LONG-sizing recalibration overlay.

Default-OFF, reversible serve-time SIZING overlay (a sibling to DIPFIX and the
conviction-gate). It de-sizes V10 LONG entries by the recent REALIZED long-win in
the CURRENT trend_regime — a causal, past-only rolling statistic — so the book
automatically shrinks long exposure in regimes where recent longs have been
losing (the 2026 down-regime overconfidence the user flagged) and rides full size
where recent longs win. SHORTS are never touched.

    mult = clip(recent_regime_long_win / base, floor, 1.0)          # longs only

THE LEVER (sweep 2026-06-25, condition=trend_regime, K=20-30, base=0.55,
floor=0.3): on held-to-K96 terminal it improved 2026 Sharpe 0.322->0.370 (+0.048)
AND both 2026 halves (+0.043/+0.042) AND pre-2026 (+0.016), halved the worst trade
in both periods, and cut pre-2026 drawdown 58% — the first session-lever positive
OOT in BOTH periods AND within-2026. It is risk-adjusted (higher Sharpe / lower
tail, ~10% less total exposure), not free total PnL.

TRAIN==SERVE BY CONSTRUCTION: the live runner AND the offline phase6/replay both
import this class and feed it the SAME realized-outcome stream in chronological
order, so the multiplier sequence is identical. IMPORTANT: the live "win" is the
EXIT-STACK realized pnl (pnl_bps > 0) — NOT the held-to-K96 terminal the sweep
approximated — so the gate-fix replay MUST re-derive the multiplier on exit-stack
realized outcomes before any flip (the de-size and the exit-cap may overlap on the
tail). The sweep validated the MECHANISM; the gate-fix validates the exit-realized
number. Nothing here changes a model or a contract; it scales OANDA units only,
clamped, instant-rollback via GX1_REGIME_RECAL=0.

WHY A NEW FILE (rule 7): the rolling per-regime outcome buffer is genuinely-new
shared cross-trade STATE used by two callers (live runner + offline gate).
size_units() in the runner owns conviction/vol sizing but is stateless; DIPFIX
owns action overlays. Neither can hold a persisted rolling deque. Import this; do
not re-implement the rolling-win anywhere else.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

# Canonical 3-class trend_regime labels (== universe / DIPFIX / V10 ctx one-hot).
TREND_REGIMES = ("TREND_DOWN", "TREND_NEUTRAL", "TREND_UP")

# Swept-best defaults (2026-06-25). All env-overridable; DEFAULT-OFF.
DEFAULT_K = 25          # rolling window of recent matured LONG takes per regime
DEFAULT_BASE = 0.55     # win-rate at which size is un-touched (mult -> 1.0)
DEFAULT_FLOOR = 0.30    # hardest de-size (never smaller than 0.30x)
DEFAULT_WARMUP = 12     # min observations in a regime before de-sizing (else 1.0x)


def desize_multiplier(recent_long_win: float, *, base: float = DEFAULT_BASE,
                      floor: float = DEFAULT_FLOOR) -> float:
    """Pure formula: clip(recent_long_win / base, floor, 1.0).

    recent_long_win in [0,1] = realized win-rate of the last K long takes in the
    current trend_regime. base is the 'no de-size' win-rate; floor caps how small
    the multiplier can get. Capped at 1.0 (this overlay only ever REDUCES long
    size, never inflates it — overconviction is the failure mode we correct)."""
    return min(max(recent_long_win / max(base, 1e-9), floor), 1.0)


class RegimeLongWinTracker:
    """Rolling per-regime realized-long-win tracker (causal, past-only).

    Feed matured LONG outcomes in CHRONOLOGICAL order via record(); read the
    current de-size multiplier via multiplier(regime). Used identically by the
    live runner (incremental, persisted across restarts) and the offline gate
    (fresh, fed in time order) -> identical multiplier stream = train==serve.
    """

    def __init__(self, *, k: int = DEFAULT_K, base: float = DEFAULT_BASE,
                 floor: float = DEFAULT_FLOOR, warmup: int = DEFAULT_WARMUP,
                 state_path: str | os.PathLike | None = None) -> None:
        self.k = int(k)
        self.base = float(base)
        self.floor = float(floor)
        self.warmup = int(warmup)
        self._buf: dict[str, deque] = {r: deque(maxlen=self.k) for r in TREND_REGIMES}
        self.state_path = Path(state_path) if state_path else None
        if self.state_path is not None and self.state_path.exists():
            self._load()

    # ── core ────────────────────────────────────────────────────────────────
    def record(self, regime: str, win: bool) -> None:
        """Append a matured LONG outcome (win = realized pnl_bps > 0) to its regime.

        Unknown/None regime is dropped (cannot attribute it). Persists if a
        state_path was given (so a runner restart keeps a warm window)."""
        if regime not in self._buf:
            return
        self._buf[regime].append(1 if win else 0)
        if self.state_path is not None:
            self._save()

    def recent_win(self, regime: str) -> float | None:
        """Realized win-rate over the buffered window for `regime`, or None if the
        window is below warmup (insufficient evidence to recalibrate)."""
        buf = self._buf.get(regime)
        if buf is None or len(buf) < self.warmup:
            return None
        return sum(buf) / len(buf)

    def multiplier(self, regime: str) -> float:
        """De-size multiplier in [floor, 1.0] for a LONG in `regime`.

        Returns 1.0 (no-op) when the regime is unknown or below warmup — fail-safe:
        a thin/cold window never de-sizes."""
        w = self.recent_win(regime)
        if w is None:
            return 1.0
        return desize_multiplier(w, base=self.base, floor=self.floor)

    def n_obs(self, regime: str) -> int:
        buf = self._buf.get(regime)
        return len(buf) if buf is not None else 0

    # ── persistence ─────────────────────────────────────────────────────────
    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"k": self.k, "buf": {r: list(self._buf[r]) for r in TREND_REGIMES}}
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.state_path)  # atomic
        except Exception:
            # persistence is best-effort; a write failure must never break trading
            pass

    def _load(self) -> None:
        try:
            payload = json.loads(self.state_path.read_text())
            for r in TREND_REGIMES:
                vals = payload.get("buf", {}).get(r, [])
                # honor the CURRENT k (maxlen) even if the saved window was wider
                self._buf[r] = deque((int(v) for v in vals[-self.k:]), maxlen=self.k)
        except Exception:
            pass


# ── env-gated live config (DEFAULT-OFF; no-op unless GX1_REGIME_RECAL=1) ──────
REGIME_RECAL_ON = os.environ.get("GX1_REGIME_RECAL", "0") == "1"
REGIME_RECAL_K = int(os.environ.get("GX1_REGIME_RECAL_K", str(DEFAULT_K)))
REGIME_RECAL_BASE = float(os.environ.get("GX1_REGIME_RECAL_BASE", str(DEFAULT_BASE)))
REGIME_RECAL_FLOOR = float(os.environ.get("GX1_REGIME_RECAL_FLOOR", str(DEFAULT_FLOOR)))
REGIME_RECAL_WARMUP = int(os.environ.get("GX1_REGIME_RECAL_WARMUP", str(DEFAULT_WARMUP)))


def build_live_tracker(state_path: str | os.PathLike | None) -> RegimeLongWinTracker:
    """Construct a tracker from the live env config (swept-best defaults)."""
    return RegimeLongWinTracker(
        k=REGIME_RECAL_K, base=REGIME_RECAL_BASE, floor=REGIME_RECAL_FLOOR,
        warmup=REGIME_RECAL_WARMUP, state_path=state_path,
    )
