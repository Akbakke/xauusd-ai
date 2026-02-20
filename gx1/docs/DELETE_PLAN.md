# Policy cleanup: DELETE_PLAN

**Mål:** Én policyfil (canonical). Alt annet under `gx1/configs/policies/**` som ikke er nødvendig for den, er kandidater til sletting.

**Canonical policy:** `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml`

---

## KEEP (behold)

Kun disse filene er nødvendige for TRUTH og canonical run:

| Relative path | Rolle |
|---------------|--------|
| `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml` | Canonical policy (the one) |
| `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/MASTER_EXIT_V1_A.yaml` | exit_config referert av canonical |
| `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/SNIPER_RISK_GUARD_V1.yaml` | risk_guard.config_path referert av canonical |

Ingen andre filer under `gx1/configs/policies/` trengs for å kjøre TRUTH/E2E med canonical policy.

---

## DELETE_CANDIDATE (eksplisitte paths)

Følgende filer og mapper skal slettes (etter review). Ingen globs – eksakt liste.

### Hele mapper (rm -rf)

- `gx1/configs/policies/active`
- `gx1/configs/policies/prod_snapshot`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1`

### Enkeltfiler under sniper_snapshot/2025_SNIPER_V1

- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY_EXIT_RULE6A_PURE.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_LIVE_TRIAL160_V10_CTX.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_DEMO_SNIPER_P4_CANARY.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P1_PLONG_080.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P2_REGIME_BLOCKS.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P4_1.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P4_1_EXITCRITIC.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P4_COMBINED.yaml`

### Enkeltfiler under sniper_snapshot/2025_SNIPER_V1/exits

- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_ADAPTIVE.yaml`
- `gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml`

---

## Rekkefølge før sletting (hygiene-gate)

Før du kjører rm-kommandoene nedenfor:

- **Step 0:** Kjør `python gx1/scripts/check_policy_hygiene.py` (skal PASS på Python-delen; YAML-listen feiler inntil Step 3 er kjørt).
- **Step 1:** Hardkodede defaults er patchet til canonical / env (gjort i one-policy PR).
- **Step 2:** Kjør hygiene igjen: `python gx1/scripts/check_policy_hygiene.py` – fortsatt forventet FAIL på YAML inntil sletting.
- **Step 3:** Execute rm-kommandoene nedenfor. Etter sletting skal hygiene scriptet gi full PASS.

---

## Kommandoer (ikke kjørt automatisk)

Kjør fra repo root (`GX1_ENGINE`).

### Preview (antall filer og total størrelse som vil fjernes)

```bash
# Tell filer og mapper som vil slettes (inkl. innhold i mapper)
find gx1/configs/policies/active gx1/configs/policies/prod_snapshot gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1 -type f 2>/dev/null | wc -l
find gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1 -maxdepth 1 -type f ! -name 'GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml' ! -name 'SNIPER_RISK_GUARD_V1.yaml' -print 2>/dev/null | wc -l
find gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits -type f ! -name 'MASTER_EXIT_V1_A.yaml' -print 2>/dev/null | wc -l
du -sh gx1/configs/policies/active gx1/configs/policies/prod_snapshot gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1 2>/dev/null
```

### Execute (slett eksakt disse paths – ingen globs)

```bash
# Mapper
rm -rf gx1/configs/policies/active
rm -rf gx1/configs/policies/prod_snapshot
rm -rf gx1/configs/policies/sniper_snapshot/2025_SNIPER_V10_1

# Filer i sniper_snapshot/2025_SNIPER_V1 (ikke exits/)
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_REPLAY_V10_CTX_VERIFY_EXIT_RULE6A_PURE.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_LIVE_TRIAL160_V10_CTX.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_SNIPER_TRAIN_V10_CTX_GATED.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_DEMO_SNIPER_P4_CANARY.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_OANDA_PRACTICE_LIVE_SNIPER_LONDON_NY.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P1_PLONG_080.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P2_REGIME_BLOCKS.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P4_1.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P4_1_EXITCRITIC.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_SNIPER_CANARY_P4_COMBINED.yaml

# Filer i exits/
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_ADAPTIVE.yaml
rm -f gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/exits/SNIPER_EXIT_RULES_A_P4_1.yaml
```

---

**Merk:** Scripts bruker nå kun canonical path eller `GX1_CANONICAL_POLICY_PATH` (one-policy PR). Hygiene-gate `check_policy_hygiene.py` hindrer regresjon på både YAML-liste og hardkodede policy-paths i Python.
