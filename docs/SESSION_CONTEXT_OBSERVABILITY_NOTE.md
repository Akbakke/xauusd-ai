# Session Context Observability Note

**Date:** 2026-03-11  
**Scope:** Session observability + sequence context (no policy change)

## Summary
This update makes **ASIA observable** in the session context while keeping **ASIA non-tradable** in policy/execution.  
Session derivation is unified to UTC boundaries:
- ASIA: 22:00–07:00
- EU: 07:00–12:00
- OVERLAP: 12:00–16:00
- US: 16:00–22:00

## Observed vs Tradable
- **Observed session (market context):** `session_id`  
  Mapping: `ASIA=0, EU=1, OVERLAP=2, US=3`
- **Tradable window (policy only):** `session_tradable` flag  
  `1` for EU/OVERLAP/US, `0` for ASIA

No new gates or policy changes were introduced. Existing policy still enforces its tradable window.

## New/Updated Features
Added to existing feature pipeline (prebuilt/runtime):
- `session_id` now includes ASIA (mapping above)
- `is_ASIA`, `is_OVERLAP` (alongside `is_EU`, `is_US`)
- `minutes_since_session_open`
- `minutes_to_next_session_boundary`
- `session_change_flag`
- `session_tradable`

These features increase observability without changing execution policy.

## Transformer Context (24/7)
Sequence features now infer `session_id` directly from timestamps (SSoT).  
This enables **24/7 context visibility** across ASIA → EU → OVERLAP → US, as long as the underlying candle stream is continuous.

## Contracts / Compatibility
- `ctx_cat` dimension unchanged (still 6).  
- **Semantic change:** `session_id` mapping updated to include ASIA.  
  This is a behavior change and should be treated as a contract update for any trained models.

## Policy Safety
ASIA remains **non-tradable** in policy/execution. No new gates were added.

