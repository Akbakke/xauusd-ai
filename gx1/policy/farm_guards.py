"""farm_guards re-export shim.

The original module was moved to the legacy/disabled tree at
`gx1/legacy/_legacy_disabled/policy_farm_exit/farm_guards.py` but several
active modules still import from `gx1.policy.farm_guards`
(notably `gx1.policy.entry_policy_sniper_core`).

Rather than dual-maintain or move the file back, this shim re-exports the
public symbols from the legacy module so existing imports keep working.

This is intentional: the legacy code is functionally correct for the v10_ctx
sniper guard flow; only its location was relocated.
"""
from gx1.legacy._legacy_disabled.policy_farm_exit.farm_guards import (  # noqa: F401
    sniper_guard_v1,
    _extract_session_vol_regime,
)

__all__ = ["sniper_guard_v1", "_extract_session_vol_regime"]
