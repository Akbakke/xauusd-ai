# SNIPER Entry Policy - Reuse Map

**Generated:** 2025-12-17  
**Purpose:** Map existing infrastructure for building EU/London/NY Sniper entry policy  
**Status:** Pre-implementation analysis - DO NOT implement until this map is reviewed

---

## Executive Summary

**Goal:** Build a new "Sniper" entry policy for EU/London/NY sessions that reuses 100% of existing infrastructure. Only new components: entry policy logic + regime detector + YAML config.

**Key Finding:** Existing "SHORT_SNIPER" policies are experimental SHORT-only variants, NOT session-based. We need a NEW session-based Sniper for EU/London/NY.

---

## A. Existing Infrastructure Inventory

### A1. Entry Policies (V9/V10)

| Klasse/Fil | Versjon | Brukt av YAML Configs | Avhengigheter |
|------------|---------|----------------------|---------------|
| `gx1/policy/entry_v9_policy_farm_v2b.py` | V9 FARM_V2B | `ENTRY_V9_FARM_V2B_PROD.yaml` (prod_snapshot) | `farm_guards.farm_brutal_guard_v2`, ENTRY_V9 model, meta-model (optional) |
| `gx1/policy/entry_v9_policy_farm_v2.py` | V9 FARM_V2 | Legacy FARM_V2 configs | `farm_guards.farm_brutal_guard_v2`, ENTRY_V9 model |
| `gx1/models/entry_v9/entry_v9_transformer.py` | V9 Model | All V9 entry policies | Feature manifest, scaler, runtime features |

**Key Functions:**
- `apply_entry_v9_policy_farm_v2b()` - Main entry policy function
- Uses `farm_brutal_guard_v2()` - Hard filter: ASIA + (LOW ∪ MEDIUM) ONLY
- `min_prob_long` threshold (default 0.68-0.72)
- `allowed_sessions: ["ASIA"]` enforced in config

**Reuse Strategy:**
- ✅ **Gjenbruk 100%**: Entry model (ENTRY_V9), feature builder, meta-model infrastructure
- ⚠️ **Modifiser**: Entry policy function to allow EU/London/NY sessions
- ⚠️ **Ny fil**: `entry_v9_policy_sniper.py` (wrapper around existing logic, different session/regime gates)

---

### A2. Exit Policies & Routers

| Klasse/Fil | Versjon | Brukt av YAML Configs | Avhengigheter |
|------------|---------|----------------------|---------------|
| `gx1/policy/exit_farm_v2_rules.py` | FARM_V2_RULES | `FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml` | PnL calculator, trade state |
| `gx1/policy/exit_hybrid_controller.py` | HYBRID_ROUTER_V1/V2/V3 | `GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml` | Exit router model (V3), range features |
| `gx1/core/hybrid_exit_router.py` | V3_RANGE | PROD_BASELINE | Decision tree model, range features |

**Key Functions:**
- `ExitFarmV2Rules` - RULE5 (sniper) + RULE6A (adaptive scalper)
- `ExitModeSelector.choose_exit_profile()` - Routes between RULE5/RULE6A
- V3 router uses: `atr_pct`, `spread_pct`, `range_pos`, `distance_to_range`, `range_edge_dist_atr`

**Reuse Strategy:**
- ✅ **Gjenbruk 100%**: Exit engine, router logic, RULE5/RULE6A profiles
- ✅ **Gjenbruk**: Same exit configs (`FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml`)
- ✅ **Gjenbruk**: Hybrid router V3 model (if Sniper uses same range features)

---

### A3. Regime & Session Logic

| Klasse/Fil | Funksjon | Session Support | Regime Support |
|------------|----------|-----------------|----------------|
| `gx1/regime/farm_regime.py` | `infer_farm_regime()` | ASIA ONLY | LOW, MEDIUM (ATR-based) |
| `gx1/execution/live_features.py` | `infer_session_tag()` | EU, US, OVERLAP, ASIA | Vol bucket (LOW/MID/HIGH) |
| `gx1/policy/farm_guards.py` | `farm_brutal_guard_v2()` | ASIA ONLY | LOW, MEDIUM |

**Session Boundaries (UTC):**
- **ASIA**: 22:00-07:00
- **EU**: 07:00-12:00
- **OVERLAP**: 12:00-16:00 (EU/US overlap)
- **US**: 16:00-22:00

**Reuse Strategy:**
- ✅ **Gjenbruk**: `infer_session_tag()` - Already supports EU/US/OVERLAP
- ⚠️ **Ny fil**: `gx1/regime/sniper_regime.py` - New regime detector for EU/London/NY
- ⚠️ **Modifiser**: Entry policy to use sniper_regime instead of farm_regime

---

### A4. Feature Builders

| Klasse/Fil | Funksjon | Features | Reuse Status |
|------------|----------|---------|--------------|
| `gx1/execution/live_features.py` | `build_live_entry_features()` | All V9 features, ATR, vol bucket, session | ✅ 100% reuse |
| `gx1/features/runtime_v9.py` | `build_v9_runtime_features()` | Sequence + snapshot features | ✅ 100% reuse |
| `gx1/features/basic_v1.py` | `build_basic_v1()` | Core OHLC features | ✅ 100% reuse |

**Key Features Available:**
- Vol features: `atr_bps`, `atr_adr_ratio`, `vol_bucket`
- Range features: `range_pos`, `distance_to_range`, `range_edge_dist_atr` (computed in EntryManager)
- Trend features: EMA, ADX (from basic_v1)
- Session: `session_tag` (EU/US/OVERLAP/ASIA)
- Spread: `spread_pct`, `spread_bps`

**Reuse Strategy:**
- ✅ **Gjenbruk 100%**: All feature builders
- ✅ **Gjenbruk**: Feature manifest alignment
- ✅ **Gjenbruk**: ATR/vol regime computation

---

### A5. Runner & Execution Infrastructure

| Klasse/Fil | Komponent | Reuse Status |
|------------|-----------|--------------|
| `gx1/execution/oanda_demo_runner.py` | `GX1DemoRunner` | ✅ 100% reuse |
| `gx1/execution/entry_manager.py` | `EntryManager` | ✅ 100% reuse |
| `gx1/execution/exit_manager.py` | `ExitManager` | ✅ 100% reuse |
| `gx1/execution/broker_client.py` | `BrokerClient` | ✅ 100% reuse |
| `gx1/monitoring/trade_journal.py` | `TradeJournal` | ✅ 100% reuse |
| `gx1/monitoring/reconcile_oanda.py` | Reconciliation | ✅ 100% reuse |
| `gx1/execution/oanda_backfill.py` | Backfill | ✅ 100% reuse |

**Reuse Strategy:**
- ✅ **Gjenbruk 100%**: All execution infrastructure
- ✅ **Gjenbruk**: Warmup gates, backfill, preflight checks
- ✅ **Gjenbruk**: Trade journaling, reconciliation, run_header generation

---

### A6. Active & Prod Snapshot Policies

**PROD_BASELINE (Authoritative):**
- `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- Entry: `ENTRY_V9_FARM_V2B_PROD.yaml` (ASIA + LOW/MEDIUM)
- Exit: `FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml`
- Router: HYBRID_ROUTER_V3_RANGE

**Existing SNIPER Policies (EXPERIMENTAL - SHORT ONLY):**
- `ENTRY_V9_FARM_V2B_EXIT_A_SHORT_SNIPER_Q1.yaml` - ⚠️ SHORT-only, ASIA session
- `GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_SHORT_SNIPER_Q1.yaml` - ⚠️ SHORT-only, ASIA session
- **NOT RELEVANT**: These are for SHORT trades, not EU/London/NY sessions

**Reuse Strategy:**
- ✅ **Gjenbruk**: Exit configs (RULE5/RULE6A)
- ✅ **Gjenbruk**: Hybrid router V3 model (if range features compatible)
- ⚠️ **Ny fil**: Sniper entry config YAML (EU/London/NY sessions)

---

## B. Sniper Design Definition

### B1. Sniper Goals (EU/London/NY)

**Regimes:**
- Vol spike detection (HIGH volatility OK for Sniper)
- Trend continuation (EMA/ADX-based)
- Session overlap (London open 08:00 UTC, NY open 13:00 UTC)
- Higher frequency than FARM (more trades, but still guardrails)

**Session Windows:**
- **EU**: 07:00-12:00 UTC (London session)
- **OVERLAP**: 12:00-16:00 UTC (EU/US overlap - high liquidity)
- **US**: 16:00-22:00 UTC (NY session)
- **NOT ASIA**: 22:00-07:00 UTC (excluded)

**Regime Requirements:**
- Allow HIGH volatility (unlike FARM which is LOW/MEDIUM only)
- Trend continuation preferred (EMA up, ADX > threshold)
- Session overlap periods prioritized

**Exit Engine:**
- ✅ **Gjenbruk**: Same Hybrid router V3 + RULE5/RULE6A
- ✅ **Gjenbruk**: Same exit configs
- ⚠️ **Optional**: Different router thresholds for Sniper (config-driven)

---

### B2. Reuse Baseline (100% Infrastructure)

| Komponent | Status | Notes |
|-----------|--------|-------|
| **Runner** | ✅ Gjenbruk | `GX1DemoRunner` - unchanged |
| **Execution** | ✅ Gjenbruk | `EntryManager` / `ExitManager` - unchanged |
| **Journaling** | ✅ Gjenbruk | `TradeJournal` - 100% coverage |
| **Backfill/Warmup** | ✅ Gjenbruk | Same gates (288 bars), same backfill logic |
| **Inventory/Run Header** | ✅ Gjenbruk | Unchanged |
| **Reconciliation** | ✅ Gjenbruk | Same OANDA reconciliation |
| **Feature Builders** | ✅ Gjenbruk | `build_live_entry_features()` - unchanged |
| **Entry Model** | ✅ Gjenbruk | ENTRY_V9 transformer - unchanged |
| **Exit Engine** | ✅ Gjenbruk | RULE5/RULE6A + Hybrid router V3 - unchanged |

---

## C. Minimal New Files Required

### C1. New YAML Policy (Active)

**File:** `gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`

**Structure:**
```yaml
meta:
  router_version: V3_RANGE
  role: CANARY  # NOT PROD_BASELINE initially
  description: >
    SNIPER entry policy for EU/London/NY sessions.
    Higher frequency than FARM, allows HIGH volatility.
    Uses same exit engine (Hybrid router V3 + RULE5/RULE6A).

policy_name: GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY
version: "GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY"
mode: "LIVE"  # or "REPLAY"
instrument: "XAU_USD"
timeframe: "M5"
warmup_bars: 288

entry_config: gx1/configs/policies/active/ENTRY_V9_SNIPER_LONDON_NY.yaml
exit_config: gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml

exit_policies:
  rule5: gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml
  rule6a: gx1/configs/exits/FARM_EXIT_V2_RULES_ADAPTIVE_v1.yaml

hybrid_exit_router:
  version: HYBRID_ROUTER_V3
  model_path: gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl
  v3_range_edge_cutoff: 1.0  # Same as PROD_BASELINE

execution:
  dry_run: false
  max_open_trades: 3  # Higher than FARM (1-2)

logging:
  level: INFO
```

---

### C2. New Entry Config YAML

**File:** `gx1/configs/policies/active/ENTRY_V9_SNIPER_LONDON_NY.yaml`

**Structure:**
```yaml
entry_v9_policy_sniper:
  enabled: true
  min_prob_long: 0.65  # Lower than FARM (0.68) for higher frequency
  min_prob_profitable: 0.0
  enable_profitable_filter: false
  require_trend_up: true  # Trend continuation preferred
  
  allow_low_vol: true
  allow_medium_vol: true
  allow_high_vol: true    # KEY DIFFERENCE: HIGH vol allowed
  allow_extreme_vol: false
  
  allow_short: false  # Long-only (same as FARM)

entry_models:
  v9:
    enabled: true
    model_dir: gx1/models/entry_v9/nextgen_2020_2025_clean  # Same as FARM

meta_model:
  enabled: true
  model_path: gx1/models/farm_entry_meta/baseline_model.pkl  # Same as FARM
  feature_cols_path: gx1/models/farm_entry_meta/feature_cols.json

allowed_sessions: ["EU", "OVERLAP", "US"]  # KEY DIFFERENCE: Not ASIA
allowed_vol_regimes: ["LOW", "MEDIUM", "HIGH"]  # KEY DIFFERENCE: HIGH allowed
allowed_trend_regimes: []  # Trend checked via require_trend_up

guard:
  enabled: true

entry_gating:
  p_side_min:
    long: 0.65  # Lower threshold for higher frequency
    short: 1.0  # Blocks shorts
  margin_min:
    long: 0.50
    short: 0.50
  side_ratio_min: 1.25
  sticky_bars: 1
```

---

### C3. New Entry Policy Python File

**File:** `gx1/policy/entry_v9_policy_sniper.py`

**Strategy:**
- **Gjenbruk**: Core logic from `entry_v9_policy_farm_v2b.py`
- **Modifiser**: Session guard (EU/OVERLAP/US instead of ASIA)
- **Modifiser**: Vol regime guard (allow HIGH volatility)
- **Modifiser**: Regime detector (use `sniper_regime.py` instead of `farm_regime.py`)

**Key Changes:**
```python
# Instead of:
from gx1.regime.farm_regime import infer_farm_regime
from gx1.policy.farm_guards import farm_brutal_guard_v2

# Use:
from gx1.regime.sniper_regime import infer_sniper_regime
from gx1.policy.sniper_guards import sniper_brutal_guard_v1

# Session check:
allowed_sessions = ["EU", "OVERLAP", "US"]  # Not ASIA

# Vol regime check:
allowed_vol_regimes = ["LOW", "MEDIUM", "HIGH"]  # HIGH allowed
```

---

### C4. New Regime Detector (Optional - Prefer Extension)

**File:** `gx1/regime/sniper_regime.py`

**Strategy:**
- **Gjenbruk**: Session inference from `live_features.infer_session_tag()`
- **Gjenbruk**: ATR regime computation (already in features)
- **Ny logikk**: Combine session + vol regime for Sniper regimes

**Function:**
```python
def infer_sniper_regime(session_id: str, atr_regime_id: str) -> str:
    """
    Return Sniper-regime string based on session + ATR-regime.
    
    Sniper scope:
    - EU + (LOW|MEDIUM|HIGH) -> "SNIPER_EU_LOW/MEDIUM/HIGH"
    - OVERLAP + (LOW|MEDIUM|HIGH) -> "SNIPER_OVERLAP_LOW/MEDIUM/HIGH"
    - US + (LOW|MEDIUM|HIGH) -> "SNIPER_US_LOW/MEDIUM/HIGH"
    - ASIA -> "SNIPER_OUT_OF_SCOPE"
    """
```

**Alternative (Prefer):**
- Extend `farm_regime.py` to support Sniper regimes
- Add `infer_sniper_regime()` function in same file
- Reuse existing session/ATR logic

---

### C5. New Guard Module (Optional - Prefer Extension)

**File:** `gx1/policy/sniper_guards.py`

**Strategy:**
- **Gjenbruk**: Guard structure from `farm_guards.py`
- **Modifiser**: Session check (EU/OVERLAP/US instead of ASIA)
- **Modifiser**: Vol regime check (allow HIGH)

**Function:**
```python
def sniper_brutal_guard_v1(row, context: str = "") -> bool:
    """
    Brutal guard for SNIPER: Allows EU/OVERLAP/US + (LOW|MEDIUM|HIGH) volatility.
    """
    # Check session: EU, OVERLAP, or US (not ASIA)
    # Check vol: LOW, MEDIUM, or HIGH (not EXTREME)
```

**Alternative (Prefer):**
- Extend `farm_guards.py` to support Sniper guards
- Add `sniper_brutal_guard_v1()` function in same file
- Reuse existing guard infrastructure

---

### C6. New Run Script

**File:** `scripts/run_practice_live_sniper_london_ny.sh`

**Strategy:**
- **Gjenbruk**: Structure from `run_live_force_one_trade_any_session.sh`
- **Modifiser**: Policy path to Sniper YAML
- **Gjenbruk**: All preflight checks, backfill, warmup logic

---

## D. Implementation Plan (After Review)

### Phase 1: Regime & Guards (Minimal)
1. Create `gx1/regime/sniper_regime.py` (or extend `farm_regime.py`)
2. Create `gx1/policy/sniper_guards.py` (or extend `farm_guards.py`)
3. Test regime inference on sample data

### Phase 2: Entry Policy (Core)
1. Create `gx1/policy/entry_v9_policy_sniper.py`
2. Reuse ENTRY_V9 model, feature builder, meta-model
3. Test entry policy on sample signals

### Phase 3: Config & Scripts
1. Create `ENTRY_V9_SNIPER_LONDON_NY.yaml`
2. Create `GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`
3. Create `run_practice_live_sniper_london_ny.sh`

### Phase 4: Testing
1. Dry-run replay on 1-2 weeks of London/NY periods
2. Verify trade journal entries
3. Verify exit engine works identically
4. Compare frequency vs FARM

---

## E. Reuse Summary

### ✅ 100% Reuse (No Changes)
- `GX1DemoRunner` - Execution runner
- `EntryManager` / `ExitManager` - Trade lifecycle
- `BrokerClient` - OANDA API
- `TradeJournal` - Trade logging
- `build_live_entry_features()` - Feature builder
- `build_v9_runtime_features()` - Runtime features
- ENTRY_V9 transformer model - Entry predictions
- Exit engine (RULE5/RULE6A) - Exit logic
- Hybrid router V3 - Exit routing
- Backfill/warmup - Historical data
- Reconciliation - OANDA matching
- Run header/inventory - Metadata

### ⚠️ Extend/Modify (Minimal Changes)
- Entry policy function - New `entry_v9_policy_sniper.py` (wrapper around existing logic)
- Regime detector - New `sniper_regime.py` (or extend `farm_regime.py`)
- Guard module - New `sniper_guards.py` (or extend `farm_guards.py`)

### 🆕 New Files (Minimal)
1. `gx1/policy/entry_v9_policy_sniper.py` (~300 lines, mostly copy-paste from farm_v2b)
2. `gx1/regime/sniper_regime.py` (~50 lines, or extend farm_regime.py)
3. `gx1/policy/sniper_guards.py` (~100 lines, or extend farm_guards.py)
4. `gx1/configs/policies/active/ENTRY_V9_SNIPER_LONDON_NY.yaml` (~60 lines)
5. `gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml` (~80 lines)
6. `scripts/run_practice_live_sniper_london_ny.sh` (~200 lines, copy-paste from existing script)

**Total New Code:** ~800 lines (mostly config + wrapper functions)

---

## F. Commands (After Implementation)

### Sniper Practice-Live
```bash
export OANDA_ENV=practice
export OANDA_API_TOKEN=<token>
export OANDA_ACCOUNT_ID=<account>
./scripts/run_practice_live_sniper_london_ny.sh
```

### Sniper Short Replay
```bash
python3 -m gx1.execution.oanda_demo_runner \
  gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml \
  --replay-mode \
  --data-path data/raw/xauusd_m5_2025_bid_ask.parquet \
  --start-date 2025-01-15 \
  --end-date 2025-01-22
```

### Inventory + Baseline Fingerprint Check
```bash
python3 gx1/scripts/audit_runs_inventory.py --infer-role --write-inventory
python3 gx1/analysis/prod_baseline_proof.py \
  --run gx1/wf_runs/SNIPER_LONDON_NY_<timestamp> \
  --prod-policy gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml
```

---

## G. Key Decisions

### Decision 1: Extend vs New Files
**Recommendation:** Extend existing files (`farm_regime.py`, `farm_guards.py`) rather than creating new ones. This reduces duplication and maintains consistency.

**Rationale:**
- Regime/guard logic is similar (session + vol check)
- Easier to maintain if logic is centralized
- Can add `infer_sniper_regime()` and `sniper_brutal_guard_v1()` to existing files

### Decision 2: Entry Policy Wrapper
**Recommendation:** Create new `entry_v9_policy_sniper.py` that wraps existing `entry_v9_policy_farm_v2b.py` logic but uses Sniper guards/regimes.

**Rationale:**
- Entry policy logic is complex (~300 lines)
- Different session/regime gates require different code paths
- Easier to test and maintain if separated

### Decision 3: Exit Engine Reuse
**Recommendation:** Reuse 100% of exit engine (RULE5/RULE6A + Hybrid router V3).

**Rationale:**
- Exit logic is session-agnostic
- Range features work for any session
- No need to change exit engine for Sniper

### Decision 4: Role/Meta
**Recommendation:** Use `meta.role: CANARY` initially (not PROD_BASELINE).

**Rationale:**
- Sniper is new, needs validation
- Can promote to PROD_BASELINE after testing
- Follows same pattern as FARM evolution

---

## H. Open Questions

1. **Router Thresholds:** Should Sniper use different router thresholds (atr_low_pct, atr_high_pct) than FARM?
   - **Recommendation:** Start with same thresholds, tune later if needed

2. **Warmup Bars:** Should Sniper use same warmup (288 bars) or different?
   - **Recommendation:** Same warmup (288 bars) for consistency

3. **Max Open Trades:** Should Sniper allow more concurrent trades than FARM?
   - **Recommendation:** Start with 3 (vs FARM's 1-2), tune based on results

4. **Trend Filter:** Should Sniper require trend continuation (`require_trend_up: true`)?
   - **Recommendation:** Yes, for initial version. Can relax later.

---

**Status:** ✅ Ready for review. DO NOT implement until this map is approved.

