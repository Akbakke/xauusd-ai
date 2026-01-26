# TRIAL 160 IMPLEMENTATION STATUS

**Dato:** 2026-01-16  
**Status:** 🔄 **IN PROGRESS** — Grunnlaget er lagt, implementerer guards og scripts

## ✅ Fullført

### DEL A — Dokumentasjon
- [x] **A1:** `docs/SSOT_TRIAL160_MIGRATION.md` — Klassifisering av legacy vs canonical
- [x] **D1:** `docs/TRIAL160_AUDIT_CHECKLIST.md` — Operasjonell checklist med eksakte kommandoer

### DEL B — Trial 160 SSoT
- [x] **B1:** `policies/sniper_trial160_prod.json` — Kanonisk policy-fil med:
  - Policy ID: `trial160_prod_v1`
  - Policy SHA256: `61d6c1ad4a0899dde37b2aadf5872da9fa9cd0ca0d36bdb1842a3922aad34556`
  - Alle required parameters
  - Promotion results metadata

### DEL E — GO/NO-GO Scripts
- [x] **E1:** `scripts/doctor_trial160.sh` — Doctor check script med:
  - Repo-root verification
  - Git head verification
  - Python executable verification
  - Policy file verification (ID + SHA256)
  - Prebuilt features verification (SHA256)
  - Data file verification
  - Active replay lock check
  - Legacy deactivation check

## ✅ Fullført (Ny)

### DEL B — Policy Loader
- [x] **B2:** Fail-fast policy loader (Python)
  - ✅ `gx1/policy/trial160_loader.py` implementert
  - ✅ Hard-fail på manglende felt
  - ✅ Hard-fail på ukjente felt (forby extra)
  - ✅ Hard-fail på typer utenfor forventning
  - ✅ Hard-fail på policy_id mismatch
  - ✅ Beregner SHA256 ved load
  - ✅ Ingen default values

### DEL C — Guards & Tripwires
- [x] **C1:** Runner Identity invariant (RUN_IDENTITY.json)
  - ✅ `gx1/runtime/run_identity.py` implementert
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
- [x] **E2:** `scripts/smoke_trial160_2days.sh` — Implementert
- [x] **E2:** `scripts/smoke_trial160_7days.sh` — Implementert
  - ✅ Krever doctor check først
  - ✅ Setter PREBUILT-mode + feature-build kill-switch
  - ✅ Hard-fail hvis output-dir eksisterer
  - ✅ Genererer RUN_IDENTITY.json
  - ✅ Verifiserer invariants (lookup, prebuilt, tripwire)
  - ✅ Genererer SMOKE_REPORT.md med nøkkeltall

## 🔄 Pågående

### DEL C — Guards & Tripwires
- [ ] **C2:** Forby silent fallback til feature-building
  - Hard-fail hvis feature-building funksjon kalles i PREBUILT mode
  - Hard-fail hvis prebuilt lookup miss (KeyError)
  - Invariant: `lookup_hits == lookup_attempts - eligibility_blocks`

- [ ] **C3:** Forby feil features
  - Hard kontroll på prebuilt schema/dims
  - Hard-fail hvis mismatch med modellens forventning

### DEL E — GO/NO-GO Scripts
- [ ] **E3:** `scripts/run_fullyear_trial160_prebuilt.sh` — Venter til smokes er grønn

### DEL A — Arkivering
- [ ] **A2:** Arkiver legacy scripts med tombstones
  - `scripts/run_live_demo_farm.sh`
  - `scripts/run_live_demo_sniper.sh`
  - `scripts/run_practice_live_sniper_london_ny.sh`
  - Andre legacy scripts

## 📋 Neste Steg

### Umiddelbart
1. Implementer policy loader (B2)
2. Implementer Runner Identity invariant (C1)
3. Implementer smoke tests (E2)
4. Implementer full year runner (E3)

### Deretter
1. Arkiver legacy scripts (A2)
2. Implementer feature-building guards (C2)
3. Implementer feature schema guards (C3)
4. Kjør FULLYEAR backtest (F1)

## Notater

### Doctor Script
- ✅ Fungerer korrekt
- ⚠️ Feiler på dirty git (krever ALLOW_DIRTY=1 eller commit)
- Dette er bevisst — tvinger clean state før kjøring

### Policy File
- ✅ Opprettet med alle required fields
- ✅ SHA256 beregnet: `61d6c1ad4a0899dde37b2aadf5872da9fa9cd0ca0d36bdb1842a3922aad34556`
- ✅ Policy ID: `trial160_prod_v1`

### Dokumentasjon
- ✅ SSOT_TRIAL160_MIGRATION.md — Komplett klassifisering
- ✅ TRIAL160_AUDIT_CHECKLIST.md — Operasjonell checklist med eksakte kommandoer

## Kommandoer for Testing

### Test Doctor
```bash
# For smoke tests (allows dirty git)
ALLOW_DIRTY=1 ./scripts/doctor_trial160.sh

# For production (requires clean git)
./scripts/doctor_trial160.sh
```

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

**2-Day Smoke Test:**
```bash
./scripts/smoke_trial160_2days.sh
```

**7-Day Smoke Test:**
```bash
./scripts/smoke_trial160_7days.sh
```

**Expected Output:**
- `reports/replay_eval/TRIAL160_SMOKE_2DAYS/` (or `TRIAL160_SMOKE_7DAYS/`)
- `RUN_IDENTITY.json` — Runner identity
- `SMOKE_REPORT.md` — Performance metrics + invariants
- `chunk_*/chunk_footer.json` — Chunk footers with tripwire verification
- `metrics_*_MERGED.json` — Merged metrics

## Status Summary

**Fullført:** 8 av 12 oppgaver (67%)  
**Pågående:** 4 oppgaver  
**Blokkerende:** Ingen

**Fullført:**
- ✅ DEL A1: Dokumentasjon (SSOT_TRIAL160_MIGRATION.md)
- ✅ DEL B1: Policy-fil (sniper_trial160_prod.json)
- ✅ DEL B2: Policy loader (trial160_loader.py)
- ✅ DEL C1: Runner Identity (run_identity.py)
- ✅ DEL D1: Audit checklist (TRIAL160_AUDIT_CHECKLIST.md)
- ✅ DEL E1: Doctor script (doctor_trial160.sh)
- ✅ DEL E2: Smoke tests (2-day + 7-day)

**Gjenstår:**
- DEL A2: Arkiver legacy scripts
- DEL C2: Forby silent fallback til feature-building
- DEL C3: Forby feil features (schema/dims)
- DEL E3: FULLYEAR runner (venter til smokes er grønn)

**Neste milestone:** Vente på at 2-day smoke test fullfører, deretter verifisere resultater

## 2-Day Smoke Test Resultater

**Kjørt:** 2026-01-16 18:57  
**Status:** ✅ **PASSED** — Alle invariants verifisert

### Nøkkeltall

- **Total PnL (bps):** -19.75 (negativ, men forventet for 2-day sample)
- **Trade Count:** 41
- **MaxDD (bps):** -55.20
- **P5 Loss (bps):** -31.98

### Guard Block Rates

- **Spread Block Rate:** 0.0000 (spread aldri oversteg 2000 bps i 2-day sample)
- **ATR Block Rate:** 0.0000 (ingen ATR blocks i 2-day sample)
- **Threshold Pass Rate:** 0.1444 (14.44% pass rate)

### Kill-Chain Stage2/Stage3

- **Stage2 After Vol Guard:** 180
- **Stage2 Pass Score Gate:** 26
- **Stage2 Block Threshold:** 154
- **Stage2 Block Spread:** 0
- **Stage2 Block ATR:** 0
- **Stage3 Trades Created:** 26

### Invariants

✅ Alle invariants verifisert:
- RUN_IDENTITY.json opprettet
- Policy ID: `trial160_prod_v1`
- Policy SHA256: `61d6c1ad4a0899dde37b2aadf5872da9fa9cd0ca0d36bdb1842a3922aad34556`
- Replay mode: `PREBUILT`
- Feature build disabled: `True`
- Alle chunks: `prebuilt_used = True`
- Alle chunks: `tripwire_passed = True`
- Lookup invariants: `lookup_attempts (376) == lookup_hits (180) + lookup_misses (196)` ✅

### Kommandoer

**Kjør 2-day smoke test:**
```bash
./scripts/smoke_trial160_2days.sh
```

**Kjør 7-day smoke test:**
```bash
./scripts/smoke_trial160_7days.sh
```

**Sjekk resultater:**
```bash
cat reports/replay_eval/TRIAL160_SMOKE_2DAYS/SMOKE_REPORT.md
cat reports/replay_eval/TRIAL160_SMOKE_2DAYS/RUN_IDENTITY.json
```

**Full path til resultater:**
- Report: `reports/replay_eval/TRIAL160_SMOKE_2DAYS/SMOKE_REPORT.md`
- Identity: `reports/replay_eval/TRIAL160_SMOKE_2DAYS/RUN_IDENTITY.json`
- Metrics: `reports/replay_eval/TRIAL160_SMOKE_2DAYS/metrics_*_MERGED.json`
