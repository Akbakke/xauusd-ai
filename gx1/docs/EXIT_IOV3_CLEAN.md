# EXIT IOV3_CLEAN contract

Frozen exit-transformer input contract. ONE UNIVERSE 6/6 context only. One representation per concept (no duplicates).

**Version:** `IOV3_CLEAN`  
**Feature count:** 35  
**This contract is append-only. Breaking changes require version bump.**

---

## Feature list (order = index)

| # | Name | Meaning |
|---|------|--------|
| 0 | p_long | Current long probability (signal bridge) |
| 1 | p_short | Current short probability |
| 2 | p_flat | Flat probability (reserved) |
| 3 | p_hat | Calibrated top-1 probability |
| 4 | uncertainty_score | Model uncertainty |
| 5 | margin_top1_top2 | Margin between top-1 and top-2 |
| 6 | entropy | Prediction entropy |
| 7 | p_long_entry | Long prob at entry (frozen snapshot) |
| 8 | p_hat_entry | Calibrated prob at entry |
| 9 | uncertainty_entry | Uncertainty at entry |
| 10 | entropy_entry | Entropy at entry |
| 11 | margin_entry | Margin at entry |
| 12 | pnl_bps_now | Current PnL in bps |
| 13 | mfe_bps | Max favorable excursion in bps |
| 14 | mae_bps | Max adverse excursion in bps |
| 15 | dd_from_mfe_bps | Drawdown from MFE (pnl_bps_now - mfe_bps); single representation |
| 16 | bars_held | Bars since entry |
| 17 | time_since_mfe_bars | Bars since MFE; single representation |
| 18 | atr_bps_now | ATR in bps (current bar) |
| 19–24 | ctx_cont_0 … ctx_cont_5 | Continuous context (6) |
| 25–30 | ctx_cat_0 … ctx_cat_5 | Categorical context (6) |
| 31 | pnl_over_atr | pnl_bps_now / atr_bps_now |
| 32 | spread_over_atr | spread_bps_now / atr_bps_now |
| 33 | price_dist_from_entry_bps | (price_now - entry_price) / entry_price * 10000 |
| 34 | price_dist_from_entry_over_atr | price_dist_from_entry_bps / atr_bps_now |

---

## Design choices (one representation per concept)

- **Drawdown:** Only at index 15 (dd_from_mfe_bps). Removed duplicate drawdown_from_peak_bps.
- **Time since MFE:** Only at index 17 (time_since_mfe_bars). Removed duplicate bars_since_mfe.
- **Raw vs ATR-normalized:** Both kept: pnl_bps_now (12) and pnl_over_atr (31); spread in state and spread_over_atr (32); price_dist_from_entry_bps (33) and price_dist_from_entry_over_atr (34).
- **State keys:** Runner may send `pnl_bps` or `pnl_bps_now`; both accepted (no fallback for missing required keys).

---

## TRUTH/SMOKE gate

In TRUTH/SMOKE runs, exit transformer must use **IOV3_CLEAN** and **input_dim = 35**. Otherwise: `RuntimeError`.

---

## Row layout (runtime)

- `row["state"]`: entry_price, price_now, spread_bps_now, pnl_bps_now or pnl_bps, mfe_bps, atr_bps_now, time_since_mfe_bars, …
- `row["signals"]`, `row["entry_snapshot"]`: IOV2-compatible.
- `row["context"]["ctx_cont"]`: length 6; `row["context"]["ctx_cat"]`: length 6.

SSoT: `gx1/contracts/exit_transformer_io_v3.py` (`ORDERED_EXIT_FEATURES_V3`, `EXIT_IO_FEATURE_COUNT`).

---

## IOV1 / IOV2

- **IOV1** and **IOV2** are used transitively by IOV3. In TRUTH/SMOKE, only IOV3_CLEAN (35 dims) is allowed at runtime.
