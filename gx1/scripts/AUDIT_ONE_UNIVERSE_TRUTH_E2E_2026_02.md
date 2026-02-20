# ONE UNIVERSE TRUTH E2E – Code Audit Report (Read-Only)

**Date:** 2026-02  
**Scope:** Actual runtime wiring for Entry and Exit in TRUTH E2E pipeline. No code changes; observation and trace only.

---

## A) What is TRUE today (as implemented)

### Entry input spec: seq/snap dims + ctx dims, and where each comes from

| Element | Dim / content | Source (file: location) |
|--------|----------------|--------------------------|
| **seq_x** | `(lookback, 7)` | Built in `gx1/execution/oanda_demo_runner.py`: ~9711–9716. Loop over `ORDERED_FIELDS` (from `gx1.contracts.signal_bridge_v1`) fills `seq_data[:, j]` from `field_to_seq[fname]` (p_long_seq, p_short_seq, …). `ORDERED_FIELDS` has 7 elements → last dim 7. Validated by `validate_seq_signal(seq_x_np, ...)` (signal_bridge_v1) at ~9723. |
| **snap_x** | `(7,)` | Same block: `snap_data[j]` from `field_to_snap[fname]`. Validated by `validate_snap_signal(snap_x_np, ...)` at ~9724. |
| **ctx_cont** | 6 | From `entry_context_features.to_tensor_continuous(expected_ctx_cont_dim)`. `expected_ctx_cont_dim` comes from bundle metadata (`bundle_meta.get("expected_ctx_cont_dim", 2)` at ~9825 and ~9875). Runner also sets `self.ctx_cont_dim` from metadata at ~3210–3211 and enforces 6/6 at ~3213–3216 (ONE_UNIVERSE_CTX_DIM_MISMATCH). So when bundle is canonical (6/6), runtime uses 6. |
| **ctx_cat** | 6 | From `entry_context_features.to_tensor_categorical(expected_ctx_cat_dim)`. `expected_ctx_cat_dim` from bundle metadata (`bundle_meta.get("expected_ctx_cat_dim", 6)` at ~9826, ~9876). Same 6/6 enforcement at load time. |
| **ctx feature names and order** | 6 cont + 6 cat | **Cont:** `gx1/contracts/signal_bridge_v1.py`: `ORDERED_CTX_CONT_NAMES_EXTENDED` = `["atr_bps", "spread_bps", "D1_dist_from_ema200_atr", "H1_range_compression_ratio", "D1_atr_percentile_252", "M15_range_compression_ratio"]` (lines 63–76). **Cat:** `ORDERED_CTX_CAT_NAMES_EXTENDED` = `["session_id", "trend_regime_id", "vol_regime_id", "atr_bucket", "spread_bucket", "H4_trend_sign_cat"]` (lines 86–96). Dataset uses these in `gx1/rl/entry_v10/dataset_v10.py` at ~106–110: `ORDERED_CTX_CONT_NAMES_EXTENDED` / `ORDERED_CTX_CAT_NAMES_EXTENDED` from signal_bridge_v1, sliced to `ctx_cont_dim`/`ctx_cat_dim` (6 each). |

**Summary:** Entry at runtime uses seq (T, 7), snap (7), ctx_cont 6, ctx_cat 6. The 7 come from XGB → signal bridge (ORDERED_FIELDS). The 12 ctx names and order come from `signal_bridge_v1.ORDERED_CTX_*_NAMES_EXTENDED`.

---

### Exit input spec: exact fields and dims, and where each comes from

| Element | Content | Source (file: location) |
|--------|---------|--------------------------|
| **First 7 (bridge signals)** | p_long, p_short, p_flat, p_hat, uncertainty_score, margin_top1_top2, entropy | `gx1/contracts/exit_transformer_io_v1.py`: `ORDERED_SIGNAL_FIELDS` (lines 19–28). Same semantic 7 as entry signal bridge. |
| **Next 5 (entry snapshot)** | p_long_entry, p_hat_entry, uncertainty_entry, entropy_entry, margin_entry | `ORDERED_ENTRY_SNAPSHOT_FIELDS` (lines 30–37). |
| **Next 7 (trade state)** | pnl_bps_now, mfe_bps, mae_bps, dd_from_mfe_bps, bars_held, time_since_mfe_bars, atr_bps_now | `ORDERED_TRADE_STATE_FIELDS` (lines 39–47). |
| **IOV1 total** | 19 dims | `FEATURE_DIM = 19` (line 52). |
| **IOV2** | IOV1 + ctx_cont (6) + ctx_cat (6) = 31 dims | `gx1/contracts/exit_transformer_io_v2.py`: `ordered_feature_names_v2`, `feature_dim_v2` (6/6 defaults at 33–34, 49–53). Row→vector: `_row_to_iov1_slice` (84–123) builds first 19 from row["state"], row["signals"], row["entry_snapshot"]; `row_to_feature_vector_v2` (127–165) appends ctx from row["context"]["ctx_cont"] and ["ctx_cat"]. |
| **Exits jsonl row layout** | signals: p_long_now, p_short_now, p_hat_now, uncertainty_score, margin_top1_top2, entropy; entry_snapshot: 5; state: 7 | `exit_transformer_io_v2.py` `_row_to_iov1_slice` (99–122): explicit mapping from row keys to vec indices. |

**Runtime snapshot builder:** `gx1/execution/live_features.py` `build_live_exit_snapshot` (266–359) returns a **DataFrame** with columns such as entry_id, atr_bps, mfe_so_far, mae_so_far, net_move_bps, etc. That schema is for **rules/snapshot** (e.g. MASTER_EXIT_V1 rules), not the exit transformer’s 19/31-dim vector. The exit transformer’s vector is built from **exits jsonl rows** (replay) via `row_to_feature_vector_v2` / `_row_to_iov1_slice`, or from `ExitMLContext` via `ml_ctx_to_feature_vector` (IOV1) in live path when exit ML is used.

**Conclusion:** Exit uses the same 7 bridge signals (first 7 of IOV1), plus entry_snapshot (5), trade_state (7). IOV2 adds ctx 6/6. So exit **does** consume the same upstream 7 and, when IOV2 is used, the same ctx 6/6.

---

### IO version selection for exit: mechanism + resolved version in TRUTH

| Aspect | Where | What |
|--------|--------|------|
| **Selection mechanism** | Policy YAML + runner | `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/MASTER_EXIT_V1_A.yaml`: `exit_ml.decider_enabled: false`, `mode: score_v1`. Exit transformer block commented out (34–38). Runner at `oanda_demo_runner.py` ~2185–2206: if `decider_enabled` and mode `exit_transformer_v0`, loads decider from `exit_transformer` config. |
| **TRUTH resolved** | MASTER_EXIT_V1_A.yaml | **Exit transformer is not used in TRUTH** with canonical policy: `decider_enabled: false`. So no exit transformer model is loaded; no IO version is “selected” at runtime for TRUTH. |
| **When transformer is used** | Config + env | If `exit_ml.decider_enabled: true` and `exit_ml.exit_transformer.enabled: true`, runner loads `ExitTransformerDecider`. Decider’s `input_dim` comes from `exit_transformer_config.json` (saved at train time). If `input_dim == 19` → IOV1 validate; else → IOV2 validate (`exit_transformer_v0.py` ~232–235). So **saved config’s `input_dim`** (19 vs 31) is the resolved IO version. |

**File references:**  
- Policy: `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/MASTER_EXIT_V1_A.yaml` (lines 21–38).  
- Runner: `gx1/execution/oanda_demo_runner.py` (2160–2208).  
- Decider: `gx1/policy/exit_transformer_v0.py` (228–235, 245–267).

---

## B) Contracts & invariants

### Signal bridge contract (7 dims) and SHA

- **File:** `gx1/contracts/signal_bridge_v1.py`
- **Ordered fields (7):** `ORDERED_FIELDS` = `["p_long", "p_short", "p_flat", "p_hat", "uncertainty_score", "margin_top1_top2", "entropy"]` (lines 46–52).
- **Dims:** `SEQ_SIGNAL_DIM = SNAP_SIGNAL_DIM = len(ORDERED_FIELDS)` → 7 (54–55).
- **SHA:** `CONTRACT_SHA256 = hashlib.sha256(("|".join(ORDERED_FIELDS)).encode("utf-8")).hexdigest()` (57). Truth file stores `"signal_bridge_contract_sha256":"dde7271db04cf51608bf18cfd1cbdbb7dc631556be4be64e9c82f21ed328f2aa"`; preflight can compare against this.
- **Other dims forbidden in TRUTH/SMOKE:** `validate_seq_signal` / `validate_snap_signal` (129–185) require last dim `SEQ_SIGNAL_DIM` / `SNAP_SIGNAL_DIM` (7). `ALLOWED_CTX_CONT_DIMS = (6,)`, `ALLOWED_CTX_CAT_DIMS = (6,)` (103–104); `validate_bundle_ctx_contract_in_strict` enforces expected_ctx_cont_dim/expected_ctx_cat_dim in those tuples. So 7 for signal and 6/6 for ctx are the only allowed dims in contract validation.

---

### ctx contract (6/6) and exact ordered names

- **ctx_cont (6):** `ORDERED_CTX_CONT_NAMES_EXTENDED` =  
  `["atr_bps", "spread_bps", "D1_dist_from_ema200_atr", "H1_range_compression_ratio", "D1_atr_percentile_252", "M15_range_compression_ratio"]`  
  (`signal_bridge_v1.py` 63–76).
- **ctx_cat (6):** `ORDERED_CTX_CAT_NAMES_EXTENDED` =  
  `["session_id", "trend_regime_id", "vol_regime_id", "atr_bucket", "spread_bucket", "H4_trend_sign_cat"]`  
  (86–96).
- **Prebuilt required ctx columns:** Exactly these 12 names (6 cont + 6 cat), no “entire EXTENDED list” beyond 6+6. See below.

---

### Prebuilt ctx reality check: exactly what is required

- **E2E preflight** (`run_truth_e2e_sanity.py` 456–474): Imports `ORDERED_CTX_CONT_NAMES_EXTENDED` and `ORDERED_CTX_CAT_NAMES_EXTENDED` from `gx1.contracts.signal_bridge_v1`, builds `required_ctx_12 = list(ORDERED_CTX_CONT_NAMES_EXTENDED) + list(ORDERED_CTX_CAT_NAMES_EXTENDED)`. So **exactly 12 columns** (the two extended lists, which are already length 6 each). Checks parquet schema for these 12; then samples 1000 rows and checks no NaN/Inf in those columns.
- **Chunk bootstrap** (`chunk_bootstrap.py` 286–295, 380–388): `_one_universe_required_ctx_columns()` returns `ORDERED_CTX_CONT_NAMES_EXTENDED[:6] + ORDERED_CTX_CAT_NAMES_EXTENDED[:6]` (same 12). `required_all = ordered_features (28) + extra` where extra = ctx columns not already in XGB 28. So prebuilt is required to have **XGB 28 + these 12 ctx columns** (and no “extended” list beyond 6+6).

**Defined in:**  
- `gx1/contracts/signal_bridge_v1.py`: ORDERED_CTX_CONT_NAMES_EXTENDED, ORDERED_CTX_CAT_NAMES_EXTENDED (6 each).  
- `gx1/scripts/run_truth_e2e_sanity.py`: 458–462 (required_ctx_12).  
- `gx1/execution/chunk_bootstrap.py`: 286–295, 380–384.

---

## C) Evidence

### Key file paths opened

1. `gx1/configs/canonical_truth_signal_only.json`
2. `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`
3. `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/MASTER_EXIT_V1_A.yaml`
4. `gx1/contracts/signal_bridge_v1.py`
5. `gx1/contracts/exit_transformer_io_v1.py`
6. `gx1/contracts/exit_transformer_io_v2.py`
7. `gx1/rl/entry_v10/dataset_v10.py`
8. `gx1/execution/oanda_demo_runner.py` (sections ~3180–3220, ~4250–4260, ~9695–9910, ~2160–2210)
9. `gx1/execution/chunk_bootstrap.py` (1–50, 282–398)
10. `gx1/execution/live_features.py` (266–359)
11. `gx1/execution/entry_context_features.py` (import of signal_bridge_v1)
12. `gx1/scripts/run_truth_e2e_sanity.py` (115–130, 193–220, 418–518, 555–610, 638–725, 1355–1382)
13. `gx1/policy/exit_transformer_v0.py` (22–29, 188–267, 302–307, 426–427, 464–472)

### Key snippets (minimal)

- **Canonical truth keys:** `canonical_xgb_bundle_dir`, `canonical_prebuilt_parquet`, `canonical_transformer_bundle_dir`, `signal_bridge_contract_sha256` (all present in JSON).
- **Policy exit:** `exit_config: .../MASTER_EXIT_V1_A.yaml`; that file has `exit.type: MASTER_EXIT_V1`, `exit_ml.decider_enabled: false` → no router, no exit_critic, no exit transformer in TRUTH.
- **Entry 7/7:** Runner builds seq/snap from `ORDERED_FIELDS` and validates with `validate_seq_signal` / `validate_snap_signal` (runner ~9711–9724).
- **Prebuilt 12 ctx:** `required_ctx_12 = list(ORDERED_CTX_CONT_NAMES_EXTENDED) + list(ORDERED_CTX_CAT_NAMES_EXTENDED)` (run_truth_e2e_sanity 458–462).
- **Replay calls:** `from gx1.execution.replay_chunk import process_chunk` and `from gx1.execution.replay_merge import merge_artifacts_1w1c` (run_truth_e2e_sanity 584–585); `process_chunk(...)` at 587, `merge_artifacts_1w1c(...)` at 606.
- **Legacy replay:** Gate at 115–129 checks `gx1.scripts.replay_eval_gated_parallel` not in sys.modules and script file quarantined.

### Runtime execution

No validate-only or micro replay was run for this report. If run, evidence would be:

- **Validate-only:** `run_dir` from args or default `GX1_DATA/reports/truth_e2e_sanity/<run_id>`; preflight result in memory / E2E_FATAL_CAPSULE if failed.
- **Micro replay:** `run_dir`; `replay/chunk_0/chunk_footer.json`: `ctx_cont_dim`, `ctx_cat_dim` (expect 6, 6), `bars_processed`, `exit_type` (MASTER_EXIT_V1), `exit_ml_enabled` (false), etc.; `replay/chunk_0/logs/exits/exits_<run_id>.jsonl` for `context.ctx_cont` / `context.ctx_cat` lengths (6/6 when exits audit writes context).

---

## D) Risks / ambiguities (observe only)

1. **Runner ctx defaults from bundle_meta:** In the entry hot path, `expected_ctx_cont_dim = bundle_meta.get("expected_ctx_cont_dim", 2)` and `expected_ctx_cat_dim = bundle_meta.get("expected_ctx_cat_dim", 6)` (e.g. oanda_demo_runner ~9825–9826, ~9875–9876). If a bundle’s metadata omitted `expected_ctx_cont_dim`, runtime would default to **2** (not 6). Canonical bundle is expected to have 6/6 in metadata; runner also enforces 6/6 at bundle load (3213–3216). So behavior is correct for canonical TRUTH, but the default “2” remains a latent fallback if metadata is wrong or missing.
2. **Exit transformer not used in TRUTH:** With MASTER_EXIT_V1_A.yaml, exit ML decider is disabled. So the “same 7 + ctx 6/6” exit path is **implemented** (IOV2 contract and row_to_feature_vector_v2) but **not exercised** in the canonical TRUTH run. Exits jsonl still gets context 6/6 written when exit audit/journaling is enabled (e.g. for LAST_GO / exit training).
3. **build_live_exit_snapshot vs exit transformer input:** `build_live_exit_snapshot` returns a DataFrame with different column names (e.g. mfe_so_far, mae_so_far, net_move_bps). The exit transformer’s 19/31-dim vector is built from exits jsonl (row→vector) or from ExitMLContext (ml_ctx_to_feature_vector). So “where to add more features for exit transformer” is the **contract + row/context mapping** (exit_transformer_io_v1/v2 and the code that fills row["state"], row["signals"], row["entry_snapshot"], row["context"]), not the DataFrame from `build_live_exit_snapshot` unless that DataFrame is later converted into that row format.
4. **Doc vs code:** Policy YAML comments mention “exit_transformer_v0” and “window_len: 64”; implementation matches. E2E docstring says “6/6 reality check (all 12 ctx columns present)” and code enforces exactly those 12 names from the two EXTENDED lists (no extra columns required).

---

**End of report.**
