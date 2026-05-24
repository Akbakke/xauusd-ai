# Risiko-oversikt: filer/bundles som kan kjøre feil/gammel kode

**Generert:** 2026-05-24 etter dyp audit (kl 19:00)

Denne fila identifiserer hvor pipelinen kan kjøre på feil/gammel kode hvis vi ikke er bevisste. Sortert etter alvorlighet.

---

## 🚨 KRITISK — vil bryte live ved restart

### 1. v12_entry_iql_live.py (linje 63-73)
**Hardkodet bundle-sti:** `BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED133[7-9, 0-1]_20260521T111046Z`
- **Status:** Alle 5 seed-bundles SLETTET i Tier-A cleanup 2026-05-24
- **Impact:** Live OANDA paper-runner-restart → `load_default()` feiler med missing bundle
- **Fix:** Etter retrain av ny vinner-Entry → update DEFAULT_BUNDLE_DIR + ENSEMBLE_BUNDLE_DIRS

### 2. v12_exit_iql_live.py (linje 138-141)
**Hardkodet bundle-sti:** `BUILD_EXIT_IQL_PER_BAR_DATASET_V12_..._V3TRACKED_..._ZCLAMP8M_TRAINED_20260520T190905Z_LOCK`
- **Status:** ALLE Exit-IQL-bundles SLETTET (Tier-B + senere)
- **Impact:** Live restart → `V12.4_HARD_LOCKED` RuntimeError på linje ~227
- **Fix:** Etter Exit-retrain → update `V12_4_APPROVED_BUNDLE` til ny bundle-navn ELLER endre hard-lock til dynamisk lookup

### 3. v12_v10_live.py (linje 67-72)
**V10 bundle:** `/home/andre2/GX1_DATA/models/models/entry_v10_ctx/ENTRY_V10_V3PLUS_v2_20260518T135516Z`
- **Status:** OK, bundle eksisterer
- **Risiko:** Hard-failer hvis bundle endres til ikke-multi-TF. Multi-TF input config bygges fra `bundle_metadata.json`.
- **Action:** Ved V10 v4-retrain MÅ bundle_metadata.json beholde `multi_tf.enabled=true`.

---

## ⚠️ HØY RISIKO — feil dataset blir brukt

### 4. Forward-outcome datasets — flere generasjoner finnes

På disk akkurat nå:
- `CANDIDATE_FORWARD_OUTCOME_V3PLUS_PORTFOLIO_PLUS5_20260521T110559Z_LOCK` (base — V10 v2 outputs IKKE inkludert)
- `CANDIDATE_FORWARD_OUTCOME_V3PLUS_PORTFOLIO_PLUS5_..._V10V2_FULL_20260523T150309Z` (V10 v2 inputs joined)
- `CANDIDATE_FORWARD_OUTCOME_V3PLUS_PORTFOLIO_PLUS5_..._V10V2_RESCORED_20260523T143509Z`
- `CANDIDATE_FORWARD_OUTCOME_V3PLUS_PORTFOLIO_PLUS5_..._V10V2_AUGMENTED_20260523T150542Z` (forrige base, V10 v2 + V2 multi-TF)
- `CANDIDATE_FORWARD_OUTCOME_V3PLUS_PLUS5_FIXED_FULL_20260524T152700Z_LOCK` (Bug 1+2+3 fix anvendt, MEN bygd på V10V2_AUGMENTED uten 5-TF dip/struct)
- `CANDIDATE_FORWARD_OUTCOME_V3PLUS_PLUS5_FIXED_V3_20260524T161239Z_LOCK` (4 dist-features for M15+D1, MEN 5-TF dip/struct expansion ikke applied)

**Problem:** Hvis vi sender retrain `--forward-outcome-dir` til feil versjon, mister vi bugfixes.
**Fix:** Bruk **ALLTID** den siste versjonen (FIXED_V3 så lenge ikke ny er bygd). Aller helst opprett symlink eller alias.

### 5. Per-bar datasets — samme problem

- `BUILD_EXIT_IQL_PER_BAR_DATASET_V12_V3PLUS_FULL_20260519T012648Z_LOCK_V3V2_FULL_V2STATE_20260523T191540Z` (V12 swing-K, 266 cols, OLD candidate-gen)
- `EXIT_IQL_PER_BAR_DATASET_V2_M1_SCALP_V3PLUS_FIXED_V3_20260524T161821Z_LOCK` (scalp K + Bug-fixes, 351 cols)

**Problem:** Phase 6 i Optuna-scripts i `/tmp/phase6_optuna.py` peker fortsatt mot V12 swing-K dataset.
**Fix:** Oppdater Phase 6-scripts til å peke på V2_M1_SCALP_V3PLUS_FIXED_V3.

### 6. V12 reports source (V12_2_REPORTS_20260514T161504Z)
- Inneholder OLD candidate-generation. Brukt for OLD per-bar V12 dataset.
- **V3PLUS-pipelinen bruker IKKE denne** — bruker INFERENCE_BATCH_CANDIDATES_V3PLUS_20260518T170242Z.
- **Risiko:** Hvis vi pår et script som default leser fra V12_2_REPORTS → får ugyldige V10 v2-only candidates.
- **Sjekk:** materialize_inference_batch_candidates_v3_v1.py + augment_forward_outcome_v2.py må peke på V3PLUS reports.

---

## ⚠️ MIDDELS RISIKO — stille feil-fallback

### 7. `.get(col, 0.0)` silent zero-fill
Steder dette skjer:
- `v12_entry_iql_live.py:254-258` — V10 v3+ aux heads (tf_agreement_pred, etc.). Hvis bundle eldre enn 2026-05-21 → 0-fill → Q-vektor kollaps.
- `entry_iql_v2_adapter.py:189-195` — numeriske features. Silent 0-fill med WARN.
- `materialize_build_entry_iql_v2.py:build_state_matrix()` — for hver col i NUMERIC_STATE_COLS som ikke finnes i df, append zeros.
- `v12_exit_iql_live.py:372` — canonical feature fallback.

**Risiko:** Feature blir DEAD uten å crash, kvalitetstap usynlig.
**Mitigation:** Logg WARN ved silent 0-fill (allerede gjort i adapter).

### 8. K_HORIZONS mismatch trener-vs-dataset
- Trener-konstanter:
  - Entry: `K_HORIZONS = [12, 24, 48, 96, 144, 192]` (M5 bars)
  - Exit V2: `K_HORIZONS = [1, 4, 12, 48, 144, 240]` (M1 bars, scalp)
- Dataset må ha `hold_max_pnl_K{K}_v1` for hver K i trener-listen.
- **NY:** `gx1/rl/dataset_horizon_constants.py` har consolidated definitions.
- **Risiko:** Hvis builder kjøres med gammel K_HORIZONS, dataset får feil kolonner → trener silent fallback til `exit_now`.

### 9. V2_BUILD_MODE environment variable
- `GX1_BUILD_IQL_V2=1` og `GX1_BUILD_EXIT_IQL_V2=1` styrer om V2-features lastes inn i state.
- **Hvis missing:** State faller tilbake til V1 → mister 125 per-TF + 32 group-A features.
- **Fix:** Alltid sett begge env vars eksplisitt i kjørings-scripts.

### 10. SMC features konstant verdi
- `smc_choch_canon_v1` har std=0.01, nnz<5% — sparse men levende.
- `smc_premium_state` har std=0.31 — alive.
- Hvis canonical_v3 prebuilt rebygges med endret SMC-logikk: training/live blir misaligned.

---

## 🟡 LAV RISIKO — gammel kode som bør ryddes

### 11. _legacy_disabled scripts (76 filer)
- `/home/andre2/src/GX1_ENGINE/gx1/scripts/_legacy_disabled/` — gamle scripts som ikke importeres
- **Sikker å slette,** men beholder for now for arkeologisk verdi.

### 12. _legacy_pre_v12 execution
- `/home/andre2/src/GX1_ENGINE/gx1/execution/_legacy_pre_v12/` — pre-V12 runners
- Sikker å slette hvis ikke vi kjører pre-V12 backtester.

### 13. Multiple Phase 6 implementasjoner
- `/tmp/phase6_optuna.py` — current
- 5 gamle Phase 6 scripts SLETTET i dag

### 14. Test bundles i /home/andre2/GX1_DATA/reports/truth_e2e_sanity/
- `/tmp/SMOKE_*` dirs — kan slettes etter hver runde
- Audit-scripts kan også ryddes når ikke i bruk

---

## ✅ TRYGGE KODESTIER (verifisert dette session)

- `gx1/scripts/materialize_build_entry_iql_v2.py` — current
- `gx1/scripts/materialize_build_exit_iql_v2.py` — current
- `gx1/scripts/augment_forward_outcome_v2.py` — current (Bug 1+2+3 anvendt)
- `gx1/scripts/augment_per_bar_v2_from_forward_outcome.py` — current (Bug 4 anvendt)
- `gx1/features/htf_features.py` — current (ffill + warmup + d1_vwap fix)
- `gx1/features/group_a_features.py` — current (M15+D1 dist)
- `gx1/rl/dataset_horizon_constants.py` — NY (canonical K_HORIZONS)
- `gx1/rl/reward_defs.py` — NY (canonical reward variant names)

---

## Bundle inventar — KEEP-liste (det vi kjører på)

### Entry-IQL
- `BUILD_ENTRY_IQL_V2_R_WAIT_OPP_K48_LAM05/10_*_LOCK`
- `BUILD_ENTRY_IQL_V2_R_WAIT_OPP_K96_LAM05/10/20/30/50_*_LOCK`
- `BUILD_ENTRY_IQL_V2_R_HYBRID_K96_TOL20/40_*_LOCK`
(9 bundles, ~7 MB total)

### Exit-IQL
- INGEN på disk akkurat nå — alle slettet i cleanup
- Må retraines

### Per-bar datasets
- `EXIT_IQL_PER_BAR_DATASET_V2_M1_SCALP_V3PLUS_FIXED_V3_20260524T161821Z_LOCK` (current, 351 cols, 2.4 GB)
- `BUILD_EXIT_IQL_PER_BAR_DATASET_V12_..._V3V2_FULL_V2STATE_20260523T191540Z` (gammel V12 swing-K, fortsatt brukt av nåværende Phase 6 cache)

### Forward-outcome
- `CANDIDATE_FORWARD_OUTCOME_V3PLUS_PLUS5_FIXED_V3_20260524T161239Z_LOCK` (current, 518 cols)
- Andre versjoner: kan slettes etter Phase 6 viser ny vinner

### Source data
- `INFERENCE_BATCH_CANDIDATES_V3PLUS_20260518T170242Z` (V3PLUS candidate source — IKKE slett)
- `CANONICAL_FEATURES_V3` + `CANONICAL_V3_PREBUILT` (SMC source — IKKE slett)
- M1 + M5 tapes (`xauusd_m*_bid_ask__CANONICAL`) — IKKE slett

---

## Anbefalt prosedyre før hver retrain

1. **Verifiser dataset:** `ls -ld $LATEST_FWD` + sjekk timestamp = today
2. **Verifiser env vars:** `echo $GX1_BUILD_IQL_V2 $GX1_BUILD_EXIT_IQL_V2` (begge `1`)
3. **Verifiser K_HORIZONS:** import gx1/rl/dataset_horizon_constants og sjekk match med dataset
4. **Smoke-test først:** `--sample-n-rows 5000 --budget fast`
5. **Sjekk state matrix dim:** state vector skal være ~265 for Entry, ~280 for Exit (etter 5-TF expansion)
6. **Verifiser featurer alive:** kjør validate_entry_all.py + audit på 2026 sample før commit til full retrain
