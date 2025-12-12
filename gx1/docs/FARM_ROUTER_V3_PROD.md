# FARM Exit Router V3 - Production Baseline

## Overview

The FARM exit routing system uses machine learning to dynamically choose between two exit policies:
- **RULE5**: Conservative sniper exit (default)
- **RULE6A**: Adaptive scalper exit (selective use)

Four router variants are available, with **V3 recommended as the production baseline** for 2025+ on XAUUSD.

---

## Router Variants

### V3 (HYBRID_ROUTER_V3) - **PROD_BASELINE**

**Status:** Recommended production baseline

**Description:**
- ML decision-tree based router trained on raw `atr_pct`/`spread_pct` values
- Conservative variant with risk haircuts applied
- Uses raw feature values (no standardization required)

**Performance (2025 full-year backtest):**
- EV/trade: ≈148–150 bps
- Win rate: ≈88%
- RULE6A share: ≈5–7% of trades
- Total PnL: 24,266 bps (163 trades)

**Routing Logic:**
- Ultralav ATR (≤6.81): → RULE5
- Moderat ATR (6.81–31.34): → RULE6A only if `spread_pct ≤ 78.10`, `atr_pct > 18.75`, and MEDIUM regime
- Høy ATR (>31.34): → RULE6A only if `spread_pct ≤ 0.50`, `atr_pct > 61.03`, and MEDIUM regime
- All trades: Long-only, entry in ASIA session

**Use Case:** Standard production trading on XAUUSD M5 timeframe.

---

### ADAPTIVE (HYBRID_ROUTER_ADAPTIVE) - **HIGH_VOLUME_MODE**

**Status:** High-volume/exploratory mode

**Description:**
- System-level adaptive router that switches between V1 (aggressive) and V2B (conservative) based on market conditions
- Highest trade count and highest total PnL
- Lower EV per trade than V3 but much higher frequency

**Performance (2025 full-year backtest):**
- EV/trade: ≈111 bps
- Win rate: ≈88%
- RULE6A share: ≈1.7% of trades
- Total PnL: 39,969 bps (360 trades)

**Use Case:** High-volume trading or exploratory modes where maximum trade frequency is desired.

---

### V2 (HYBRID_ROUTER_V2) - **RESEARCH_ONLY**

**Status:** Research/legacy only (not recommended for production)

**Description:**
- Early ML-based router with aggressive RULE6A allocation
- Over-aggressive RULE6A usage leads to lower EV per trade and win rate

**Performance (2025 full-year backtest):**
- EV/trade: ≈71 bps
- Win rate: ≈72%
- RULE6A share: ≈41% of trades
- Total PnL: 9,951 bps (141 trades)

**Use Case:** Reference implementation showing what over-aggressive RULE6A usage looks like. Not recommended for production.

---

### V2B (HYBRID_ROUTER_V2B) - **ARCHIVED_SAFE_MODE**

**Status:** Archived (too conservative)

**Description:**
- Ultra-conservative router with near-zero RULE6A allocation
- Overly defensive approach results in lowest EV per trade and win rate

**Performance (2025 full-year backtest):**
- EV/trade: ≈46 bps
- Win rate: ≈59%
- RULE6A share: ≈0% of trades
- Total PnL: 7,273 bps (158 trades)

**Use Case:** Historical experiment demonstrating "too safe" routing. Not recommended for production.

---

## Router Roles Summary

| Router | Role | EV/trade | Win Rate | RULE6A % | Status |
|--------|------|----------|----------|----------|--------|
| **V3** | `PROD_BASELINE` | 148–150 bps | 88% | 5–7% | ✅ Recommended |
| **ADAPTIVE** | `HIGH_VOLUME_MODE` | 111 bps | 88% | 1.7% | High-frequency mode |
| **V2** | `RESEARCH_ONLY` | 71 bps | 72% | 41% | ❌ Too aggressive |
| **V2B** | `ARCHIVED_SAFE_MODE` | 46 bps | 59% | 0% | ❌ Too conservative |

---

## Production Recommendation

**Use HYBRID_ROUTER_V3 as the production baseline** for 2025+ on XAUUSD.

V3 provides the best balance of:
- High EV per trade (148–150 bps)
- High win rate (88%)
- Conservative RULE6A allocation (5–7%)
- Robust risk management (cuts high-spread RULE6A, requires MEDIUM regime)

Configuration: `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_FULLYEAR.yaml`

---

## Implementation

Router implementations are in:
- `gx1/core/hybrid_exit_router.py` - Router logic
- `gx1/policy/exit_hybrid_controller.py` - Integration with EntryManager

Training and analysis:
- `gx1/analysis/router_training_v3.py` - Model training
- `gx1/analysis/compare_exit_routers_fullyear.py` - Performance comparison

