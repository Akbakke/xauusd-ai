# SSoT TRIAL 160 MIGRATION — Legacy vs Canonical

**Dato:** 2026-01-16  
**Status:** 🔄 **IN PROGRESS** — Migration til Trial 160 som single source of truth

## Mål

Etablere Trial 160 som **eneste gyldige pipeline** for replay/live, med harde guards som forhindrer:
- Feil features
- Feil policy
- Feil datasett
- Feil env
- Silent fallbacks

## Canonical Pipeline (Trial 160)

### Eneste Gyldige Pipeline

**Replay:**
- **Entrypoint:** `scripts/run_fullyear_trial160_prebuilt.sh`
- **Mode:** PREBUILT only (hard requirement)
- **Features:** V10_CTX gated fusion (prebuilt)
- **Policy:** `policies/sniper_trial160_prod.json` (SSoT)
- **Data:** `data/raw/xauusd_m5_2025_bid_ask.parquet`
- **Prebuilt:** `data/features/xauusd_m5_2025_features_v10_ctx.parquet`

**Live:**
- **Entrypoint:** `scripts/run_live_trial160.sh` (TODO: implementer)
- **Mode:** PREBUILT only (hard requirement)
- **Features:** V10_CTX gated fusion (prebuilt)
- **Policy:** `policies/sniper_trial160_prod.json` (SSoT)

### Trial 160 Parameters (SSoT)

```json
{
  "policy_id": "trial160_prod_v1",
  "entry_threshold": 0.102,
  "max_concurrent_positions": 2,
  "risk_guard_block_atr_bps_gte": 13.73,
  "risk_guard_block_spread_bps_gte": 2000,
  "risk_guard_cooldown_bars_after_entry": 2
}
```

**Policy SHA256:** (beregnes ved opprettelse)

## Legacy — Arkiveres/Deaktiveres

### Legacy Scripts (FARM/SNIPER)

**FARM Legacy:**
- `scripts/run_live_demo_farm.sh` → **ARKIVER**
- `scripts/go_practice_live_asia.sh` → **ARKIVER**
- `scripts/watch_live_demo_farm.sh` → **ARKIVER**

**SNIPER Legacy (gammel):**
- `scripts/run_live_demo_sniper.sh` → **ARKIVER**
- `scripts/run_practice_live_sniper_london_ny.sh` → **ARKIVER**
- `scripts/run_live_demo_sniper_p4_canary.sh` → **ARKIVER**

**Legacy Replay Scripts:**
- `scripts/run_sniper_entry_v10_1_flat_fullyear_2025.sh` → **ARKIVER**
- `scripts/run_sniper_entry_v10_1_flat_ungated_fullyear_2025.sh` → **ARKIVER**
- `scripts/run_replay_fullyear_2025_parallel.sh` → **ARKIVER** (hvis ikke Trial 160)

### Legacy Policies

**FARM Policies:**
- `gx1/configs/policies/prod_snapshot/2025_FARM_V2B_HYBRID_V3_RANGE/` → **BEHOLD** (arkivert, ikke aktiv)
- `gx1/configs/policies/active/GX1_V11_OANDA_DEMO_V9_FARM_*.yaml` → **BEHOLD** (arkivert, ikke aktiv)

**SNIPER Legacy Policies:**
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml` → **BEHOLD** (arkivert, ikke aktiv)
- `gx1/configs/policies/active/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_*.yaml` → **BEHOLD** (arkivert, ikke aktiv)

### Legacy Code Paths

**Feature Building (forby i PREBUILT mode):**
- `gx1/features/basic_v1.py::build_basic_v1()` → **HARD-FAIL** hvis kalt i PREBUILT mode
- `gx1/execution/live_features.py::build_live_entry_features()` → **HARD-FAIL** hvis kalt i PREBUILT mode
- `gx1/features/runtime_v10_ctx.py::build_v10_ctx_runtime_features()` → **HARD-FAIL** hvis kalt i PREBUILT mode

**Legacy Entry Policies:**
- `gx1/policy/entry_v9_policy_farm_v2b.py` → **BEHOLD** (arkivert, ikke aktiv)
- `gx1/policy/entry_v9_policy_sniper.py` → **BEHOLD** (arkivert, ikke aktiv)

## Migration Plan

### Fase 1: Klassifisering ✅

- [x] Dokumenter canonical vs legacy
- [x] Identifiser alle legacy scripts
- [x] Identifiser alle legacy policies

### Fase 2: Arkivering

- [ ] Opprett `archive/legacy_20260116/`
- [ ] Flytt legacy scripts til archive
- [ ] Legg inn tombstones i original paths
- [ ] Verifiser at ingen scripts kan kjøre legacy uten `ALLOW_LEGACY=1`

### Fase 3: Trial 160 SSoT

- [ ] Opprett `policies/sniper_trial160_prod.json`
- [ ] Implementer fail-fast policy loader
- [ ] Verifiser policy_id + sha256 logging

### Fase 4: Guards & Tripwires

- [ ] Implementer Runner Identity invariant (RUN_IDENTITY.json)
- [ ] Forby silent fallback til feature-building
- [ ] Forby feil features (schema/dims mismatch)

### Fase 5: GO/NO-GO Scripts

- [ ] `scripts/doctor_trial160.sh`
- [ ] `scripts/smoke_trial160_2days.sh`
- [ ] `scripts/smoke_trial160_7days.sh`
- [ ] `scripts/run_fullyear_trial160_prebuilt.sh`

### Fase 6: Verifisering

- [ ] Kjør doctor + smoke tests
- [ ] Kjør FULLYEAR backtest
- [ ] Generer FULLYEAR_TRIAL160_REPORT.md
- [ ] Verifiser alle invariants

## Tombstone Format

Legacy scripts skal erstattes med tombstones som hard-feiler:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEGACY REMOVED — Tombstone

This script has been archived as part of Trial 160 migration.
Use Trial 160 pipeline instead.

Archived: 2026-01-16
Archive location: archive/legacy_20260116/
"""

import sys

def main():
    if os.getenv("ALLOW_LEGACY") != "1":
        print("ERROR: Legacy script removed.")
        print("This script has been archived as part of Trial 160 migration.")
        print("Use Trial 160 pipeline instead:")
        print("  scripts/run_fullyear_trial160_prebuilt.sh (replay)")
        print("  scripts/run_live_trial160.sh (live)")
        print("")
        print("To override (NOT RECOMMENDED):")
        print("  ALLOW_LEGACY=1 ./scripts/<legacy_script>")
        sys.exit(1)
    
    # Original script logic (only if ALLOW_LEGACY=1)
    # ... (moved to archive)

if __name__ == "__main__":
    main()
```

## Verifisering

### Check Legacy Scripts

```bash
# Sjekk at ingen legacy scripts kan kjøres uten ALLOW_LEGACY=1
git grep -n "run_live_demo_farm|run_live_demo_sniper|FARM|SNIPER" scripts/ | grep -v "ALLOW_LEGACY"
```

### Check Canonical Pipeline

```bash
# Verifiser at Trial 160 pipeline er eneste gyldige
./scripts/doctor_trial160.sh
```

## Status

**Nåværende status:** 🔄 **IN PROGRESS**

**Neste steg:**
1. Implementer Trial 160 policy-fil
2. Implementer guards og tripwires
3. Arkiver legacy scripts
4. Verifiser hele kjeden
