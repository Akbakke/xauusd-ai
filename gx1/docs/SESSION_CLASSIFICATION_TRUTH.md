# Session Classification - Truth Documentation

> **Status:** Read-only truth documentation  
> **Date:** 2026-01-30  
> **Purpose:** Document canonical session classification rules

## Executive Summary

**Session classification is determined by UTC hour using `infer_session_tag()` function.**

**Source:** `gx1/execution/live_features.py::infer_session_tag()`

---

## 1. Session Boundaries (UTC)

| Session Name | Start (UTC) | End (UTC) | UTC Hours | Notes |
|--------------|-------------|-----------|-----------|-------|
| **ASIA** | 22:00 | 07:00 (next day) | 22-06:59 | Covers late US close to early EU pre-open |
| **EU** | 07:00 | 12:00 | 7-11 | European session |
| **OVERLAP** | 12:00 | 16:00 | 12-15 | EU/US overlap period |
| **US** | 16:00 | 22:00 | 16-21 | US session |

**Timezone:** All timestamps are converted to UTC before classification.

**Overlap Rules:**
- OVERLAP is a distinct session (12:00-16:00 UTC)
- No gaps or ambiguities: all 24 hours are covered by exactly one session

---

## 2. Implementation

**Function:** `gx1/execution/live_features.py::infer_session_tag(timestamp: pd.Timestamp) -> str`

**Logic:**
```python
if 7 <= hour < 12:
    return "EU"
elif 12 <= hour < 16:
    return "OVERLAP"
elif 16 <= hour < 22:
    return "US"
else:
    # Hours 22:00-06:59 UTC: ASIA session
    return "ASIA"
```

**Invariants:**
- Always returns one of: `"EU"`, `"US"`, `"OVERLAP"`, `"ASIA"`
- Never returns `"UNKNOWN"` or `"ALL"`
- Defaults to `"ASIA"` for invalid/NaN timestamps (with warning)

**Timezone Handling:**
- If timestamp is timezone-aware: converts to UTC
- If timestamp is timezone-naive: assumes UTC

---

## 3. Session Token Mapping

**For Transformer session tokens (one-hot encoding):**

| Session | Token Index | Token Name |
|---------|-------------|------------|
| ASIA | 0 | `session_is_asia` |
| EU | 1 | `session_is_eu` |
| OVERLAP | 2 | `session_is_overlap` |
| US | 3 | `session_is_us` |

**One-Hot Invariant:**
- Exactly one token = 1.0 per bar
- Sum of all tokens = 1.0 (within tolerance)

---

## 4. References

- **Implementation:** `gx1/execution/live_features.py::infer_session_tag()`
- **Usage:** Called throughout codebase for session routing and classification
- **Session Tokens:** `gx1/features/feature_contract_v10_ctx.py::SESSION_TOKEN_NAMES`
