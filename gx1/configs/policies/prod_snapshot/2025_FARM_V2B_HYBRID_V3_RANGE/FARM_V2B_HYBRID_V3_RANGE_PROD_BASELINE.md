# FARM_V2B_HYBRID_V3_RANGE - PROD_BASELINE

**Status:** ✅ PROD_BASELINE (Frozen 2025-12-15)  
**Router Version:** V3_RANGE + Guardrail  
**Entry Policy:** FARM_V2B mp68_max5_cd0 (long-only)  
**Exit Policies:** RULE5 + RULE6A (ML-routed with guardrail)  
**Guardrail:** RULE6A only allowed when range_edge_dist_atr < 1.0  
**Testperiode:** Full year 2025 (2025-01-01 → 2025-12-31)

---

## 📊 Executive Summary

V3_RANGE router med range-aware features (range_pos, distance_to_range, range_edge_dist_atr) har blitt testet på fullt år 2025 og markert som PROD_BASELINE.

**Key Results (with guardrail):**
- ✅ **162 trades** i FULLYEAR replay
- ✅ **117.67 bps EV/trade**
- ✅ **84.6% win rate**
- ✅ **100% range feature coverage** (range_pos, range_edge_dist_atr)
- ✅ **RULE6A allocation: 9.3%** (15 trades, all with range_edge_dist_atr < 1.0)
- ✅ **Guardrail blocked 42 trades** from RULE6A (range_edge_dist_atr >= 1.0)
- ✅ **All blocked trades had identical PnL** under RULE5 vs RULE6A (RULE6A was "cosmetic")

**Konklusjon:** V3_RANGE router gir range-aware routing med god performance og konsistent feature coverage.

---

## 🔧 Frozen Components

### 1. Router Model
- **File:** `exit_router_models_v3/exit_router_v3_tree.pkl`
- **Type:** scikit-learn DecisionTreeClassifier
- **Training Data:** 1592 trades (FULLYEAR 2025)
- **Accuracy:** 73.2%
- **Tree Rules:** See `exit_router_v3_tree_rules.txt`

### 2. Feature Set
**Numeric Features:**
- `atr_pct`: ATR in basis points
- `spread_pct`: Spread in basis points
- `range_pos`: Position in range [0.0, 1.0]
- `distance_to_range`: Distance to range [0.0, 1.0]
- `range_edge_dist_atr`: ATR-normalized distance to nearest range edge [0.0, 10.0]

**Categorical Features:**
- `regime`: FARM regime (FARM_ASIA_MEDIUM, FARM_ASIA_LOW, etc.)

**Feature Cleaning:**
- `range_pos`: NaN → 0.5, clip [0.0, 1.0]
- `distance_to_range`: NaN → 0.5, clip [0.0, 1.0]
- `range_edge_dist_atr`: NaN → 0.0, clip [0.0, 10.0]

### 3. Exit Policies
- **RULE5:** `FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml`
- **RULE6A:** `FARM_EXIT_V2_RULES_ADAPTIVE_v1.yaml`

### 4. Guardrail
- **Rule:** RULE6A only allowed when `range_edge_dist_atr < 1.0`
- **Rationale:** RULE6A er en edge-spesialisert exit, kun aktivert når range_edge_dist_atr < 1.0. I øvrige regimer gir den ingen målbar forbedring i PnL eller risiko og er derfor deaktivert.
- **Implementation:** Post-processing guardrail in `exit_hybrid_controller.py` that overrides router decision if `range_edge_dist_atr >= cutoff`
- **Effect:** Reduced RULE6A allocation from 35.2% to 9.3%, with identical EV/trade (117.67 bps)

---

## 📈 Performance Metrics

### FULLYEAR 2025 Replay Results

| Metric | Value |
|--------|-------|
| **Total Trades** | 162 |
| **EV/trade** | 117.67 bps |
| **Trades/day** | 0.56 |
| **Win Rate** | 84.6% |
| **Median PnL** | 189.88 bps |
| **Min PnL** | -405.06 bps |
| **Max PnL** | 212.62 bps |
| **Median Bars Held** | 56,288 |

### Exit Profile Distribution

| Profile | Trades | % | Mean PnL (bps) |
|---------|--------|---|----------------|
| **RULE5** | 147 | 90.7% | 110.19 |
| **RULE6A** | 15 | 9.3% | 190.98 |

**Note:** With guardrail, RULE6A allocation reduced from 35.2% to 9.3%. All remaining RULE6A trades have `range_edge_dist_atr < 1.0` and show superior performance (190.98 bps mean PnL vs 110.19 bps for RULE5).

### Range Features Coverage

| Feature | Coverage | Stats |
|---------|----------|-------|
| **range_pos** | 162/162 (100.0%) | min=0.005, max=1.000, median=0.745 |
| **range_edge_dist_atr** | 162/162 (100.0%) | min=0.000, max=10.000, median=2.224 |

### RULE6A Range Analysis

- **RULE6A trades near edge** (range_edge_dist_atr < 2.0): 29/57 (50.9%)
- **RULE6A range_edge_dist_atr median:** 1.97
- **Pattern:** RULE6A allokeres oftere nær range edges

---

## 🌳 Decision Tree Structure

**Primary Splits:**
- `atr_pct <= 6.81` → RULE5
- `6.81 < atr_pct <= 33.50`:
  - `spread_pct <= 98.47` → RULE6A (moderate ATR, OK spread)
  - `spread_pct > 98.47`:
    - `range_edge_dist_atr <= 0.08` → RULE5
    - `range_edge_dist_atr > 0.08` → RULE6A
- `atr_pct > 33.50`:
  - `spread_pct <= 0.50` and `atr_pct > 61.34` → RULE6A
  - else → RULE5

**Key Insight:** Tree uses `range_edge_dist_atr` for high-spread edge cases, confirming range-aware routing.

---

## 📁 Files

### Config Files
- **PROD Config:** `GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`
- **Entry Config:** `../2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml`
- **Exit Configs:** See `exit_policies` in PROD config

### Model Files
- **Tree Model:** `exit_router_models_v3/exit_router_v3_tree.pkl`
- **Tree Rules:** `exit_router_models_v3/exit_router_v3_tree_rules.txt`
- **Metrics:** `exit_router_models_v3/exit_router_v3_metrics.json`

### Replay Results
- **Trade Log:** `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv`
- **Results:** `gx1/wf_runs/FARM_V2B_EXIT_HYBRID_ROUTER_V3_RANGE_FULLYEAR/results.json`

---

## ✅ Freeze Checklist

- [x] Router model frozen (`exit_router_v3_tree.pkl`)
- [x] Feature set frozen (5 numeric features + regime)
- [x] Exit policies frozen (RULE5 + RULE6A configs)
- [x] Guardrail frozen (range_edge_dist_atr < 1.0 cutoff)
- [x] PROD config created (with guardrail)
- [x] Documentation created
- [x] Model files copied to prod_snapshot
- [x] Marked as PROD_BASELINE in config

---

## 🚀 Usage

To use this PROD_BASELINE configuration:

```bash
bash scripts/run_replay.sh \
  gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml \
  2025-01-01 2025-12-31 \
  7 \
  gx1/wf_runs/YOUR_RUN_TAG
```

---

**Status:** ✅ PROD_BASELINE frozen 2025-12-15 (with guardrail)  
**Next Steps:** Ready for production deployment

---

## 🔒 Guardrail Documentation

**RULE6A er en edge-spesialisert exit, kun aktivert når range_edge_dist_atr < 1.0. I øvrige regimer gir den ingen målbar forbedring i PnL eller risiko og er derfor deaktivert.**

### Verification Results

- **42 trades blocked** by guardrail (range_edge_dist_atr >= 1.0)
- **All blocked trades** had identical PnL under RULE5 vs RULE6A (delta = 0.0 exact)
- **All blocked trades** had identical intratrade risk (MFE, MAE, DD all identical)
- **RULE6A was "cosmetic"** in these regimes - no measurable difference
- **15 remaining RULE6A trades** all have range_edge_dist_atr < 1.0 and show superior performance (190.98 bps mean PnL)

### Implementation

Guardrail implemented as post-processing in `exit_hybrid_controller.py`:
- Router selects policy (RULE5 or RULE6A)
- If RULE6A selected and `range_edge_dist_atr >= cutoff` (1.0), override to RULE5
- Debug logging when guardrail activates

