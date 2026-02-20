# BASE28-only routing verification

**Mål:** Bekrefte at model_compare_2025 kun støtter BASE28 (og evt. BASE28_CTX2PLUS_T1). Ingen PRUNE14/PRUNE20 i config eller i `run_model_compare_2025.py`.

---

## rg-kommandoer og forventede resultater

### 1) Config: ingen PRUNE14/PRUNE20 som modellnøkler

```bash
cd /home/andre2/src/GX1_ENGINE
rg -n '"PRUNE14"|"PRUNE20"' gx1/configs/model_compare_2025/model_configs.json
```

**Forventet:** Ingen treff. (Eventuelle treff som kun er del av en sti-streng, f.eks. `TRANSFORMER_SIGNAL_ONLY_PRUNE14_...`, er tillatt som path, ikke som config key. Sjekk manuelt at `models`-nøklene kun er `BASE28` og evt. `BASE28_CTX2PLUS_T1`.)

```bash
jq '.models | keys' gx1/configs/model_compare_2025/model_configs.json
```

**Forventet:** `["BASE28", "BASE28_CTX2PLUS_T1"]` (eller `["BASE28"]` hvis CTX2PLUS er tatt ut).

---

### 2) run_model_compare_2025.py: ingen PRUNE14/PRUNE20

```bash
rg -n 'PRUNE14|PRUNE20' gx1/scripts/run_model_compare_2025.py
```

**Forventet:** 0 treff.

---

### 3) Docs: kun historikkrapport og PRUNE14-diagnose

```bash
rg -l 'PRUNE14|PRUNE20' docs/
```

**Forventet:** Kun filer som er tillatt som historikk:
- `docs/REPORT_prune_noise_removal.md` (audit-evidence)
- `docs/PRUNE14_DIAGNOSIS_PLAN.md` (historisk diagnose)

Ingen treff i `docs/REPORT_model_compare_config_validation.md` som anbefaler PRUNE14/PRUNE20 for model_compare. Eventuell omtale i REPORT_model_compare_config_validation er kun i tabellrad om PRUNE14_DIAGNOSIS_PLAN (andre domener).

---

### 4) Verifikasjonskommando (--validate_only)

```bash
python -m gx1.scripts.run_model_compare_2025 --validate_only --models BASE28
```

**Forventet:** exit 0. Evidence skrevet til `$GX1_DATA/reports/model_compare_2025/model_compare_validation_evidence.json`.

---

### 5) Ukjent modell: hard-fail med tilgjengelige = config-nøkler

```bash
python -m gx1.scripts.run_model_compare_2025 --validate_only --models PRUNE20
```

**Forventet:** exit 1. Stderr inneholder `UNKNOWN_MODEL` og `available=` med kun de modellene som finnes i config (typisk `BASE28`, evt. `BASE28_CTX2PLUS_T1`).

---

## Oppsummering

| Sjekk | Kommando / handling | Forventet |
|-------|---------------------|-----------|
| Config models-keys | `jq '.models | keys' gx1/configs/model_compare_2025/model_configs.json` | Kun BASE28 (og evt. BASE28_CTX2PLUS_T1) |
| Script fri for PRUNE | `rg -n 'PRUNE14|PRUNE20' gx1/scripts/run_model_compare_2025.py` | 0 treff |
| Validate BASE28 | `--validate_only --models BASE28` | exit 0, evidence-fil |
| Unknown model | `--validate_only --models PRUNE20` | exit 1, UNKNOWN_MODEL, available=list fra config |

Deterministic evidence writing beholdt; ingen endring i trading-/replay-semantikk.
