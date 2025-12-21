# Baseline Map - GX1 Trading System

**Purpose:** Quick reference for policy paths, artifact locations, and expected journal fields.

---

## PROD_BASELINE Policy

**Policy Path:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_V3_RANGE_PROD.yaml`

**Entry Config:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID/ENTRY_V9_FARM_V2B_PROD.yaml`

**Exit Config:** `gx1/configs/exits/FARM_EXIT_V2_RULES_A_mp68_max5_rule5_cd0.yaml`

**Router Model:** `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/exit_router_models_v3/exit_router_v3_tree.pkl`

**Feature Manifest:** Located in entry model directory (e.g., `gx1/models/entry_v9/feature_manifest.json`)

**Meta Role:** `PROD_BASELINE`

**Sessions:** ASIA only (via `farm_brutal_guard_v2`)

**Vol Regimes:** LOW, MEDIUM (via `allow_medium_vol` config)

---

## SNIPER Snapshot Policy

**Policy Path:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`

**Entry Config:** `gx1/configs/policies/active/ENTRY_V9_SNIPER_LONDON_NY.yaml`

**Exit Config:** Same as PROD_BASELINE (reuses exit engine)

**Router Model:** Same as PROD_BASELINE (reuses router model)

**Feature Manifest:** Same as PROD_BASELINE (reuses feature manifest)

**Meta Role:** `SNIPER_CANARY`

**Sessions:** EU, OVERLAP, US (via `allowed_sessions` config)

**Vol Regimes:** LOW, MEDIUM, HIGH (via `allowed_vol_regimes` config)

---

## Run Header Artifacts

**Location:** `<run_dir>/run_header.json`

**Always Generated:** Yes (for both LIVE and REPLAY modes)

**Contains:**
- `git_commit`: Git commit hash
- `policy_path`: Path to policy YAML
- `artifacts.policy.sha256`: Policy file hash
- `artifacts.entry_model.sha256`: Entry model hash (if available)
- `artifacts.feature_manifest.sha256`: Feature manifest hash
- `artifacts.router.sha256`: Router model hash (if hybrid exit enabled)
- `artifacts.exit_config.sha256`: Exit config hash
- `meta.role`: Policy role (PROD_BASELINE, SNIPER_CANARY, etc.)
- `instrument`: Trading instrument (e.g., "XAU_USD")
- `timeframe`: Candle timeframe (e.g., "M5")
- `warmup_bars`: Required warmup bars
- `timestamp`: Run start timestamp
- `run_tag`: Run identifier

**Replay-Specific:**
- `replay.eval_start_ts`: Evaluation window start
- `replay.eval_end_ts`: Evaluation window end
- `replay.start_ts`: Full data start
- `replay.end_ts`: Full data end

---

## Trade Journal Index Fields

**Location:** `<run_dir>/trade_journal/trade_journal_index.csv`

**Required Fields (always present):**
- `trade_id`: Unique trade identifier
- `entry_time`: Entry timestamp (ISO8601 UTC)
- `exit_time`: Exit timestamp (ISO8601 UTC, empty if open)
- `side`: Trade side (long/short)
- `exit_profile`: Exit profile (RULE5, RULE6A, etc.)
- `pnl_bps`: Realized PnL in basis points
- `exit_reason`: Exit reason (RULE_A_PROFIT, RULE_A_TRAILING, REPLAY_EOF, etc.)

**Entry Context Fields (parity with FARM):**
- `session`: Trading session (EU, OVERLAP, US, ASIA)
- `vol_regime`: Volatility regime (LOW, MEDIUM, HIGH, EXTREME)
- `trend_regime`: Trend regime (TREND_UP, TREND_DOWN, NEUTRAL, UNKNOWN)
- `atr_bps`: ATR in basis points (at entry)
- `spread_bps`: Spread in basis points (at entry)
- `range_pos`: Range position [0.0, 1.0] (at entry)
- `distance_to_range`: Distance to range [0.0, 1.0] (at entry)

**Router Fields:**
- `router_version`: Router version (V3, V3_RANGE, etc.)
- `router_decision`: Raw router decision (RULE5, RULE6A)
- `range_edge_dist_atr`: Range edge distance (ATR-normalized)
- `guardrail_applied`: Whether guardrail was applied (bool)

**Execution Fields (LIVE only):**
- `oanda_trade_id`: OANDA trade ID
- `oanda_last_txn_id`: OANDA last transaction ID
- `execution_status`: Execution status (OK, REJECTED, UNKNOWN)

**Merge Metadata (after merge):**
- `source_chunk`: Source chunk label (root, chunk_0, etc.)
- `trade_file`: Trade JSON filename

---

## Field Sources

**Session:**
- Primary: `policy_state["session"]` (set by `infer_session_tag()`)
- Fallback: `current_row["session"]` (from features)
- Fallback: `infer_session_tag(timestamp)` (direct inference)

**Vol Regime:**
- Primary: `policy_state["brain_vol_regime"]` (Big Brain V1)
- Fallback: `policy_state["vol_regime"]`
- Fallback: `policy_state["farm_regime"]` (extract vol part, e.g., "LOW" from "ASIA_LOW")
- Fallback: `current_row["_v1_atr_regime_id"]` (map 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)

**Trend Regime:**
- Primary: `policy_state["brain_trend_regime"]` (Big Brain V1)
- Fallback: `policy_state["trend_regime"]`
- Default: "UNKNOWN" (if not available)

**Range Features:**
- Source: `trade.extra` (computed in EntryManager)
- Fields: `range_pos`, `distance_to_range`, `range_edge_dist_atr`
- Also available in `feature_context.range` in trade JSON

**ATR/Spread:**
- Source: `feature_context.atr` and `feature_context.spread` in trade JSON
- ATR: `atr_bps` (basis points)
- Spread: `spread_bps` (basis points, computed from `spread_pct * 10000`)

---

## Verification Commands

**Check run_header exists:**
```bash
ls -l gx1/wf_runs/<RUN_TAG>/run_header.json
```

**Check run_header artifacts:**
```bash
jq '.artifacts' gx1/wf_runs/<RUN_TAG>/run_header.json
```

**Check journal index fields:**
```bash
head -1 gx1/wf_runs/<RUN_TAG>/trade_journal/trade_journal_index.csv
```

**Check vol_regime coverage:**
```bash
python3 -c "import pandas as pd; df=pd.read_csv('gx1/wf_runs/<RUN_TAG>/trade_journal/trade_journal_index.csv'); print(f'Total: {len(df)}, With vol_regime: {df[\"vol_regime\"].notna().sum()}, Empty: {df[\"vol_regime\"].isna().sum()}')"
```

**Check session coverage:**
```bash
python3 -c "import pandas as pd; df=pd.read_csv('gx1/wf_runs/<RUN_TAG>/trade_journal/trade_journal_index.csv'); print(df['session'].value_counts())"
```

---

## Notes

- **Backward Compatibility:** Old runs without new fields will have empty strings in CSV (not errors)
- **Replay Parity:** Replay mode now generates `run_header.json` and populates all journal fields (same as LIVE)
- **Merge Safety:** Merge script preserves all columns from source indexes, even if they differ between chunks

