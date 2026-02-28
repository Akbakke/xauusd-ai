# Risk Guard V1
#
# Purpose: Reduce tail risk with minimal EV loss by:
# - Blocking entries during extreme spread/vol spikes
# - Enforcing cooldown after entry
# - Optional session-specific clamps (US / HIGH vol)

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# Reason codes
REASON_SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
REASON_ATR_TOO_HIGH = "ATR_TOO_HIGH"
REASON_VOL_EXTREME = "VOL_EXTREME"
REASON_SESSION_UNKNOWN = "SESSION_UNKNOWN"
REASON_COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
REASON_US_SPREAD_TOO_HIGH = "US_SPREAD_TOO_HIGH"
REASON_OVERLAP_SPREAD_TOO_HIGH = "OVERLAP_SPREAD_TOO_HIGH"

GUARD_NAME = "RISK_GUARD_V1"
GUARD_VERSION = "1.0.0"
_GUARD_IMPL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GUARD_ID_STRING = f"{GUARD_NAME}::{GUARD_VERSION}::{_GUARD_IMPL_SHA256}"


class RiskGuardV1:
    """
    Risk guard to reduce tail risk.

    Does not modify signals; only returns block/allow decisions.
    Session-specific probability clamps are exposed via get_session_clamp()
    and must be applied by the caller (EntryManager).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize guard with configuration.
        Args:
            config: Guard configuration dict (from YAML)
        """
        # Canonical key (no legacy naming)
        self.cfg = (config or {}).get("risk_guard_v1", {})
        self.enabled = bool(self.cfg.get("enabled", False))

        # Optional identity validation
        cfg_guard_id = (self.cfg or {}).get("guard_id")
        if cfg_guard_id and isinstance(cfg_guard_id, dict):
            cfg_name = cfg_guard_id.get("name")
            cfg_ver = cfg_guard_id.get("version")
            cfg_sha = cfg_guard_id.get("impl_sha256")
            if cfg_name != GUARD_NAME or cfg_ver != GUARD_VERSION or cfg_sha != _GUARD_IMPL_SHA256:
                raise RuntimeError(
                    f"[RISK_GUARD_ID_MISMATCH] guard_id mismatch: "
                    f"cfg={{name:{cfg_name},version:{cfg_ver},sha:{cfg_sha}}} "
                    f"expected={{name:{GUARD_NAME},version:{GUARD_VERSION},sha:{_GUARD_IMPL_SHA256}}}"
                )

        # Global thresholds (units: bps)
        self.block_spread_bps = float(self.cfg.get("block_if_spread_bps_gte", 3500))
        self.block_atr_bps = float(self.cfg.get("block_if_atr_bps_gte", 25.0))
        self.block_vol_regimes = list(self.cfg.get("block_if_vol_regime_in", ["EXTREME"]))

        # Missing-data policy (avoid silent “None means OK” unless explicitly allowed)
        self.allow_if_missing_spread = bool(self.cfg.get("allow_if_missing_spread", True))
        self.allow_if_missing_atr = bool(self.cfg.get("allow_if_missing_atr", True))
        self.allow_if_missing_session = bool(self.cfg.get("allow_if_missing_session", False))

        # Session configs
        self.us_cfg = dict(self.cfg.get("us_session", {}) or {})
        self.overlap_cfg = dict(self.cfg.get("overlap_session", {}) or {})

        # Cooldown tracking (per-run state)
        self._last_entry_bar_index: Optional[int] = None

    def should_block(
        self,
        entry_snapshot: Dict[str, Any],
        feature_context: Dict[str, Any],
        policy_state: Dict[str, Any],
        current_bar_index: int,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Determine if entry should be blocked.
        Returns:
            (should_block, reason_code, details_dict)
        """
        if not self.enabled:
            return (False, None, {})

        details: Dict[str, Any] = {}

        # Cooldown check
        cooldown_bars = int(self.cfg.get("cooldown_bars_after_entry", 1))
        if not self.cooldown_allows(current_bar_index, cooldown_bars):
            details["cooldown_bars"] = cooldown_bars
            details["last_entry_bar"] = self._last_entry_bar_index
            details["current_bar"] = current_bar_index
            return (True, REASON_COOLDOWN_ACTIVE, details)

        # Session
        session = policy_state.get("session")
        if not session:
            session = entry_snapshot.get("session")
        if not session or session == "UNKNOWN":
            if not self.allow_if_missing_session:
                details["session"] = session
                return (True, REASON_SESSION_UNKNOWN, details)

        # Spread bps (NO 'or' because 0.0 is a valid float)
        spread_bps = feature_context.get("spread_bps")
        if spread_bps is None:
            spread_bps = entry_snapshot.get("spread_bps")

        if spread_bps is None:
            if not self.allow_if_missing_spread:
                details["spread_bps"] = None
                return (True, REASON_SPREAD_TOO_HIGH, details)

        # ATR bps
        atr_bps = feature_context.get("atr_bps")
        if atr_bps is None:
            atr_bps = entry_snapshot.get("atr_bps")

        if atr_bps is None:
            if not self.allow_if_missing_atr:
                details["atr_bps"] = None
                return (True, REASON_ATR_TOO_HIGH, details)

        # Vol regime
        vol_regime = policy_state.get("vol_regime")
        if not vol_regime:
            vol_regime = entry_snapshot.get("vol_regime")
        if vol_regime in self.block_vol_regimes:
            details["vol_regime"] = vol_regime
            return (True, REASON_VOL_EXTREME, details)

        # Global spread threshold
        if spread_bps is not None and float(spread_bps) >= self.block_spread_bps:
            details["spread_bps"] = float(spread_bps)
            details["threshold"] = self.block_spread_bps
            return (True, REASON_SPREAD_TOO_HIGH, details)

        # Global ATR threshold
        if atr_bps is not None and float(atr_bps) >= self.block_atr_bps:
            details["atr_bps"] = float(atr_bps)
            details["threshold"] = self.block_atr_bps
            return (True, REASON_ATR_TOO_HIGH, details)

        # Session-specific spread blocks only (clamps are NOT blocks)
        session_upper = (session or "").upper()
        if session_upper == "US":
            block_if_spread = self.us_cfg.get("block_if_spread_bps_gte", None)
            if block_if_spread is not None and spread_bps is not None and float(spread_bps) >= float(block_if_spread):
                details["spread_bps"] = float(spread_bps)
                details["threshold"] = float(block_if_spread)
                return (True, REASON_US_SPREAD_TOO_HIGH, details)

        if session_upper == "OVERLAP":
            block_if_spread = self.overlap_cfg.get("block_if_spread_bps_gte", None)
            if block_if_spread is not None and spread_bps is not None and float(spread_bps) >= float(block_if_spread):
                details["spread_bps"] = float(spread_bps)
                details["threshold"] = float(block_if_spread)
                return (True, REASON_OVERLAP_SPREAD_TOO_HIGH, details)

        return (False, None, details)

    def cooldown_allows(self, current_bar_index: int, cooldown_bars: int) -> bool:
        if self._last_entry_bar_index is None:
            return True
        return (int(current_bar_index) - int(self._last_entry_bar_index)) >= int(cooldown_bars)

    def record_entry(self, current_bar_index: int) -> None:
        self._last_entry_bar_index = int(current_bar_index)

    def get_session_clamp(self, session: Optional[str]) -> Optional[float]:
        """
        Return extra_min_prob_long clamp for session if configured.
        This does NOT block by itself; caller should apply clamp to thresholds.
        """
        session_upper = (session or "").upper()
        if session_upper == "US":
            clamp = float(self.us_cfg.get("extra_min_prob_long", 0.0) or 0.0)
            return clamp if clamp > 0 else None
        if session_upper == "OVERLAP":
            clamp = float(self.overlap_cfg.get("extra_min_prob_long", 0.0) or 0.0)
            return clamp if clamp > 0 else None
        return None

    def get_guard_id(self) -> Dict[str, str]:
        return {
            "name": GUARD_NAME,
            "version": GUARD_VERSION,
            "impl_sha256": _GUARD_IMPL_SHA256,
            "id_string": GUARD_ID_STRING,
        }