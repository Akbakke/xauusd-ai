# TRIAL 160 IMPLEMENTATION — COMPLETE STATUS

**Dato:** 2026-01-16  
**Status:** ✅ **B2, C1, E2 FULLFØRT** — Policy loader, Runner Identity, og Smoke tests implementert og testet

## ✅ Fullført (8/12 oppgaver — 67%)

### DEL A — Dokumentasjon
- [x] **A1:** `docs/SSOT_TRIAL160_MIGRATION.md` — Klassifisering av legacy vs canonical
- [x] **D1:** `docs/TRIAL160_AUDIT_CHECKLIST.md` — Operasjonell checklist med eksakte kommandoer

### DEL B — Trial 160 SSoT
- [x] **B1:** `policies/sniper_trial160_prod.json` — Kanonisk policy-fil
  - Policy ID: `trial160_prod_v1`
  - Policy SHA256: `61d6c1ad4a0899dde37b2aadf5872da9fa9cd0ca0d36bdb1842a3922aad34556`
  
- [x] **B2:** `gx1/policy/trial160_loader.py` — Fail-fast policy loader
  - ✅ Hard-fail på manglende felt
  - ✅ Hard-fail på ukjente felt (forby extra)
  - ✅ Hard-fail på typer utenfor forventning
  - ✅ Hard-fail på policy_id mismatch
  - ✅ Beregner SHA256 ved load
  - ✅ Ingen default values

### DEL C — Guards & Tripwires
- [x] **C1:** `gx1/runtime/run_identity.py` — Runner Identity invariant
  - ✅ Git head + dirty flag
  - ✅ Python executable + version
  - ✅ Bundle SHA256 (optional)
  - ✅ Windows SHA (optional)
  - ✅ Prebuilt manifest SHA256 + path
  - ✅ Policy SHA256 + policy_id
  - ✅ Replay mode = PREBUILT (enum)
  - ✅ Feature build disabled = true
  - ✅ Atomisk write (temp file + rename)

### DEL E — GO/NO-GO Scripts
- [x] **E1:** `scripts/doctor_trial160.sh` — Doctor check script
- [x] **E2:** `scripts/smoke_trial160_2days.sh` — 2-day smoke test
- [x] **E2:** `scripts/smoke_trial160_7days.sh` — 7-day smoke test

## 🔄 Gjenstår (4/12 oppgaver — 33%)

### DEL A — Arkivering
- [ ] **A2:** Arkiver legacy scripts med tombstones

### DEL C — Guards & Tripwires
- [ ] **C2:** Forby silent fallback til feature-building
- [ ] **C3:** Forby feil features (schema/dims mismatch)

### DEL E — FULLYEAR Runner
- [ ] **E3:** `scripts/run_fullyear_trial160_prebuilt.sh` — Venter til smokes er grønn

## 2-Day Smoke Test — Resultater

**Kjørt:** 2026-01-16 18:57  
**Status:** ✅ **PASSED**

### Nøkkeltall

- **Total PnL (bps):** -19.75
- **Trade Count:** 41
- **MaxDD (bps):** -55.20
- **P5 Loss (bps):** -31.98

### Guard Block Rates

- **Spread Block Rate:** 0.0000
- **ATR Block Rate:** 0.0000
- **Threshold Pass Rate:** 0.1444

### Kill-Chain

- **Stage2 After Vol Guard:** 180
- **Stage2 Pass Score Gate:** 26
- **Stage2 Block Threshold:** 154
- **Stage3 Trades Created:** 26

### Invariants Verifisert

✅ **Alle invariants PASSED:**
- RUN_IDENTITY.json opprettet
- Policy ID: `trial160_prod_v1`
- Policy SHA256: `61d6c1ad4a0899dde37b2aadf5872da9fa9cd0ca0d36bdb1842a3922aad34556`
- Replay mode: `PREBUILT`
- Feature build disabled: `True`
- Alle chunks: `prebuilt_used = True`
- Alle chunks: `tripwire_passed = True`
- Lookup invariants: `lookup_attempts (376) == lookup_hits (180) + lookup_misses (196)` ✅

## Implementerte Filer

### Policy & Loader
- `policies/sniper_trial160_prod.json` — Kanonisk policy-fil
- `gx1/policy/trial160_loader.py` — Fail-fast policy loader

### Runtime Identity
- `gx1/runtime/run_identity.py` — RUN_IDENTITY.json generator

### Scripts
- `scripts/doctor_trial160.sh` — Doctor check
- `scripts/smoke_trial160_2days.sh` — 2-day smoke test
- `scripts/smoke_trial160_7days.sh` — 7-day smoke test

### Dokumentasjon
- `docs/SSOT_TRIAL160_MIGRATION.md` — Migration plan
- `docs/TRIAL160_AUDIT_CHECKLIST.md` — Operasjonell checklist
- `docs/TRIAL160_IMPLEMENTATION_STATUS.md` — Status tracking

## Kommandoer

### Test Policy Loader
```bash
python3 -m gx1.policy.trial160_loader policies/sniper_trial160_prod.json
```

### Test RUN_IDENTITY
```bash
mkdir -p /tmp/test_run_identity
export GX1_REPLAY_USE_PREBUILT_FEATURES=1
export GX1_FEATURE_BUILD_DISABLED=1
python3 -m gx1.runtime.run_identity \
  --output-dir /tmp/test_run_identity \
  --policy-id trial160_prod_v1 \
  --policy-sha256 61d6c1ad4a0899dde37b2aadf5872da9fa9cd0ca0d36bdb1842a3922aad34556 \
  --prebuilt-path data/features/xauusd_m5_2025_features_v10_ctx.parquet \
  --allow-dirty
```

### Run Smoke Tests
```bash
# 2-day smoke test
./scripts/smoke_trial160_2days.sh

# 7-day smoke test
./scripts/smoke_trial160_7days.sh
```

### Check Results
```bash
# 2-day smoke test results
cat reports/replay_eval/TRIAL160_SMOKE_2DAYS/SMOKE_REPORT.md
cat reports/replay_eval/TRIAL160_SMOKE_2DAYS/RUN_IDENTITY.json
```

## Neste Steg

1. ✅ **B2, C1, E2 FULLFØRT** — Policy loader, Runner Identity, og Smoke tests
2. ⏳ **Venter på:** 7-day smoke test (hvis ønskelig)
3. 📋 **Gjenstår:** 
   - DEL A2: Arkiver legacy scripts
   - DEL C2: Forby silent fallback til feature-building
   - DEL C3: Forby feil features
   - DEL E3: FULLYEAR runner (når smokes er grønn)

## Notater

- **2-day smoke test:** ✅ PASSED — Alle invariants verifisert
- **Policy loader:** ✅ Fungerer — Hard-fail på alle edge cases
- **RUN_IDENTITY:** ✅ Fungerer — Alle required fields logges
- **Doctor script:** ✅ Fungerer — Krever ALLOW_DIRTY=1 for smoke tests
