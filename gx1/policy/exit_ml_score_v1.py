"""
Exit ML Score v1 – deterministic linear scorer for ML-exit decision.

Uses signal-bridge + trade-state inputs. No randomness, no network.
Weights and threshold from config (policy/yaml or ExitTuningCapsule); SSoT.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Default weights (overridden by config)
DEFAULT_EXIT_ML_SCORE_V1 = {
    "w0": 0.0,
    "w1": 0.15,   # drawdown_from_mfe_atr
    "w2": 0.05,   # entropy_slope
    "w3": -0.1,   # delta_p_long (negative: p_long dropping -> exit)
    "w4": 0.08,   # uncertainty_score
    "w5": -0.2,   # p_long_now - p_long_entry (negative: conviction drop -> exit)
    "threshold": 0.5,
}


@dataclass
class ExitMLContext:
    """Causal context for ML exit decision (per bar, per trade)."""

    # Trade state (current bar)
    pnl_bps: float = 0.0
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    bars_held: int = 0
    time_since_mfe_bars: int = 0
    drawdown_from_mfe_bps: float = 0.0
    drawdown_from_mfe_atr: Optional[float] = None
    atr_bps: Optional[float] = None
    session: Optional[str] = None
    spread_bps: Optional[float] = None

    # Current signal bridge
    p_long: Optional[float] = None
    p_short: Optional[float] = None
    p_flat: Optional[float] = None
    p_hat: Optional[float] = None
    uncertainty_score: Optional[float] = None
    margin_top1_top2: Optional[float] = None
    entropy: Optional[float] = None

    # Entry snapshot (frozen at entry)
    p_long_entry: Optional[float] = None
    p_hat_entry: Optional[float] = None
    uncertainty_entry: Optional[float] = None
    entropy_entry: Optional[float] = None
    margin_entry: Optional[float] = None

    # Deltas (over 1/3/5 bars)
    dp_long_1: Optional[float] = None
    dp_long_3: Optional[float] = None
    dp_long_5: Optional[float] = None
    dentropy_3: Optional[float] = None
    duncertainty_3: Optional[float] = None
    entropy_slope: Optional[float] = None  # e.g. (entropy - entropy_3bar_ago) or simple delta

    def to_audit_dict(self) -> Dict[str, Any]:
        """For jsonl audit; only finite numbers and present fields."""
        d: Dict[str, Any] = {}
        for k, v in (
            ("pnl_bps", self.pnl_bps),
            ("mfe_bps", self.mfe_bps),
            ("mae_bps", self.mae_bps),
            ("bars_held", self.bars_held),
            ("time_since_mfe_bars", self.time_since_mfe_bars),
            ("drawdown_from_mfe_bps", self.drawdown_from_mfe_bps),
            ("drawdown_from_mfe_atr", self.drawdown_from_mfe_atr),
            ("p_long", self.p_long),
            ("p_hat", self.p_hat),
            ("uncertainty_score", self.uncertainty_score),
            ("entropy", self.entropy),
            ("p_long_entry", self.p_long_entry),
            ("p_hat_entry", self.p_hat_entry),
            ("uncertainty_entry", self.uncertainty_entry),
            ("entropy_entry", self.entropy_entry),
            ("dp_long_1", self.dp_long_1),
            ("dp_long_3", self.dp_long_3),
            ("dp_long_5", self.dp_long_5),
            ("dentropy_3", self.dentropy_3),
            ("duncertainty_3", self.duncertainty_3),
            ("entropy_slope", self.entropy_slope),
        ):
            if v is not None and (not isinstance(v, float) or math.isfinite(v)):
                d[k] = round(v, 6) if isinstance(v, float) else v
        return d


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def compute_exit_score_and_decision(
    ctx: ExitMLContext,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[float, str, str]:
    """
    Deterministic: exit_score = w0 + w1*dd_mfe_atr + w2*entropy_slope + w3*delta_p_long
        + w4*uncertainty + w5*p_long_delta_vs_entry (p_long_delta_vs_entry = p_long_now - p_long_entry).
    Returns (exit_score, decision, reason) with decision in ("HOLD", "EXIT"), reason "ML_SCORE_EXIT" or "".
    """
    cfg = config or {}
    w0 = float(cfg.get("w0", DEFAULT_EXIT_ML_SCORE_V1["w0"]))
    w1 = float(cfg.get("w1", DEFAULT_EXIT_ML_SCORE_V1["w1"]))
    w2 = float(cfg.get("w2", DEFAULT_EXIT_ML_SCORE_V1["w2"]))
    w3 = float(cfg.get("w3", DEFAULT_EXIT_ML_SCORE_V1["w3"]))
    w4 = float(cfg.get("w4", DEFAULT_EXIT_ML_SCORE_V1["w4"]))
    w5 = float(cfg.get("w5", DEFAULT_EXIT_ML_SCORE_V1["w5"]))
    threshold = float(cfg.get("threshold", DEFAULT_EXIT_ML_SCORE_V1["threshold"]))

    dd_atr = _safe_float(ctx.drawdown_from_mfe_atr) or 0.0
    entropy_slope = _safe_float(ctx.entropy_slope) or 0.0
    delta_p_long = _safe_float(ctx.dp_long_3) or _safe_float(ctx.dp_long_1) or 0.0
    uncertainty = _safe_float(ctx.uncertainty_score) or 0.0
    p_long_now = _safe_float(ctx.p_long) or 0.0
    p_long_entry = _safe_float(ctx.p_long_entry) or 0.0
    # p_long_delta_vs_entry = now - entry; negative means conviction dropped (w5 encourages exit)
    p_long_delta_vs_entry = p_long_now - p_long_entry

    score = (
        w0
        + w1 * dd_atr
        + w2 * entropy_slope
        + w3 * delta_p_long
        + w4 * uncertainty
        + w5 * p_long_delta_vs_entry
    )

    if score > threshold:
        return (float(score), "EXIT", "ML_SCORE_EXIT")
    return (float(score), "HOLD", "")


def exit_ml_config_hash(config: Dict[str, Any]) -> str:
    """Deterministic hash of config for footer/audit."""
    canonical = json.dumps({k: config[k] for k in sorted(config) if k in ("w0", "w1", "w2", "w3", "w4", "w5", "threshold")}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
