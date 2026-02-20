# Model Compare 2025 — config routing og validering (SSoT-strict)

**Dato:** 2025-02-16  
**Mål:** Verifisere at config lastes og valideres på én måte (SSoT); `--validate_only` kjører strict validator og skriver evidence.

---

## A) rg-verifisert config-bruk

### `rg -n "model_compare_2025"` (utvalg)

| Fil | Linje | Treff |
|-----|-------|--------|
| gx1/scripts/run_model_compare_2025.py | 37–38 | OUT_ROOT, CONFIGS_PATH |
| (MODEL_SELECTION.md fjernet; BASE28-only) | — | — |
| gx1/configs/model_compare_2025/README.md | 20, 33, … | Bruk av script |
| docs/PRUNE14_DIAGNOSIS_PLAN.md | 23, 33, … | Run-dir, script |
| scripts/run_model_compare_2025.sh | 8, 18 | Wrapper som kaller .py |

### `rg -n "model_configs\.json"`

| Fil | Linje |
|-----|-------|
| gx1/scripts/run_model_compare_2025.py | 38 |

Kun én definisjon: `CONFIGS_PATH = ENGINE / "gx1" / "configs" / "model_compare_2025" / "model_configs.json"`.

### `rg -n "CONFIGS_PATH"`

| Fil | Linje | Bruk |
|-----|-------|------|
| gx1/scripts/run_model_compare_2025.py | 38 | Definisjon |
| gx1/scripts/run_model_compare_2025.py | 56, 62 | _load_configs(), CONFIGS_PATH leses |
| gx1/scripts/run_model_compare_2025.py | 1009 | validate_model_compare_config_strict(CONFIGS_PATH, …) ved --validate_only |

### `rg -n "_load_configs"`

| Fil | Linje |
|-----|-------|
| gx1/scripts/run_model_compare_2025.py | 56 | Definisjon |
| gx1/scripts/run_model_compare_2025.py | 990 | main() kaller _load_configs() for config-dict |

### `rg -n "_resolve_model_paths"`

| Fil | Linje | Bruk |
|-----|-------|------|
| gx1/scripts/run_model_compare_2025.py | 57 | Definisjon |
| gx1/scripts/run_model_compare_2025.py | 81, 223, 520 | _validate_model_paths, validate_model_compare_config_strict, run_single_job bruker _resolve_model_paths |

### `rg -n "--validate_only"`

| Fil | Linje |
|-----|-------|
| gx1/scripts/run_model_compare_2025.py | 979 | argparse --validate_only |
| gx1/configs/model_compare_2025/README.md | 11 | Check status without starting |

---

## Oppsummering: hvor config lastes og brukes

- **Lastes:** `gx1/scripts/run_model_compare_2025.py` — `CONFIGS_PATH` peker på `gx1/configs/model_compare_2025/model_configs.json`. `_load_configs()` leser denne filen og returnerer config-dict (kalles fra `main()`).
- **Brukes:** Samme script bruker config via `_resolve_model_paths(config, model)` for å få `canonical_xgb_bundle_dir`, `canonical_prebuilt_parquet`, `canonical_transformer_bundle_dir` og `signal_bridge_contract_sha256` per modell. Brukes i validering og `run_single_job`.
- **Validering:** Ved `--validate_only` kalles `validate_model_compare_config_strict(CONFIGS_PATH, truth_mode, run_dir=OUT_ROOT, models=models)`. Ingen replay/training; kun strict validering og evidence-fil.

---

## B) Strict validator og evidence-format

- **Funksjon:** `validate_model_compare_config_strict(config_path, truth_mode, run_dir, models) -> dict`
- **Returnerer:** Evidence-dict med stabil format: `{"truth_mode": bool, "config_path": str, "models": [{"model": str, "status": "OK"|"FAIL", "reason": str}, ...]}`.
- **Sjekker (hard-fail):** paths (xgb/transformer = dir, prebuilt = file), MASTER_MODEL_LOCK.json / MASTER_TRANSFORMER_LOCK.json, prebuilt prefix vs XGB `ordered_features`, signal_bridge_contract_sha256, ctx dims ved manifest, lock model_sha256 vs fil.
- **Hjelpere:** `_load_json`, `_sha256_file`, `_get_prebuilt_column_prefix` (schema_manifest først, deretter parquet; ekskluderer time/ts/timestamp/__index_level_0__).

---

## C) --validate_only

- Kaller `validate_model_compare_config_strict(CONFIGS_PATH, truth_mode, run_dir=OUT_ROOT, models=models)`.
- `truth_mode = (GX1_TRUTH_MODE == "1") or (GX1_RUN_MODE in {"TRUTH", "SMOKE"})`.
- Exit 0 ved full pass; exit 1 ved `RuntimeError` (melding + «evidence skrevet til …»).

---

## F) Verifikasjon

```bash
python -m gx1.scripts.run_model_compare_2025 --validate_only --models BASE28
```

**Forventet:** exit 0; fil `$GX1_DATA/reports/model_compare_2025/model_compare_validation_evidence.json` finnes og inneholder `"config_path"`, `"truth_mode"`, `"models"` med status OK.

**Ved feil:** exit 1; samme evidence-fil med siste modell `"status": "FAIL"` og `"reason"` satt.

---

## Resultat

| Sjekk | Status |
|-------|--------|
| Config lastes kun fra CONFIGS_PATH i run_model_compare_2025.py | **PASS** |
| _resolve_model_paths er eneste kilde til model paths fra config | **PASS** |
| --validate_only kjører strict validator, exit 0/1, evidence skrevet | **PASS** |
| Evidence-format inkluderer truth_mode, config_path, models | **PASS** |

**Konklusjon: PASS.** Config-routing og validering er SSoT-strict i `run_model_compare_2025.py`.

---

## Paranoid hardening (nyere endringer)

**Mål:** Idiot-sikker validering uten fallback eller silent skip; deterministisk evidence; presise feilmeldinger.

### 1) Ukjente modeller (SSoT: config)

- **main():** Etter `_load_configs()` sjekkes at alle `models` finnes i `config["models"]`. Hvis ikke: exit 1 med `UNKNOWN_MODEL: requested=[...] unknown=[...] available=[...]`. Ingen continue eller warn.
- **Validator:** Samme sjekk i starten av `validate_model_compare_config_strict`; ved ukjente modeller skrives evidence og raises `RuntimeError` med samme meldingsformat.

### 2) Manifest-SSoT og ctx-dims

- **Prebuilt-prefix:** XGB lock `ordered_features` er SSoT; prebuilt-kolonneliste (schema_manifest eller parquet) må matche prefix eksakt i rekkefølge.
- **Ctx dims/names:** Transformer `MASTER_TRANSFORMER_LOCK.json` og/eller `bundle_metadata.json` er SSoT for `expected_ctx_cont_dim`/`expected_ctx_cat_dim` og `ordered_ctx_*_names`.
- **Dataset-manifest:** Hvis `{prebuilt.stem}.manifest.json` finnes, MÅ dims/names der matche transformer (SSoT). Hvis manifest mangler: ctx-dims-sjekk hoppes over og i evidence settes `ctx_manifest_ok: "skipped_no_manifest"`. Ingen infer fra data.

### 3) Transformer model_path_relative + sha

- `MASTER_TRANSFORMER_LOCK.json` må ha `model_path_relative` som peker på en fil innenfor transformer bundle-dir (resolve + `relative_to`; ingen escape ut av mappen).
- Hvis lock har `model_sha256`: beregnes sha256 på `bundle_dir / model_path_relative` og det gjøres hard-fail ved mismatch.
- `model_state_dict.pt` er ikke lenger hardkodet; filen kommer utelukkende fra lock.

### 4) signal_bridge_contract_sha256 mot alle kilder

- Config `signal_bridge_contract_sha256` må matche transformer lock *og* (hvis den finnes) `bundle_metadata.json`.
- Feilmelding ved mismatch inkluderer alle tre: `config=... transformer_lock=... bundle_metadata=...`.

### 5) Evidence: utvidet format, deterministisk

- Beholdt: `config_path`, `truth_mode`, `models`.
- Per modell: `checks` med faste nøkler (samme rekkefølge hver gang):  
  `ctx_manifest_ok`, `prebuilt_prefix_ok`, `sha_ok`, `signal_bridge_sha_ok`, `transformer_lock_ok`, `transformer_model_file_ok`, `xgb_lock_ok`.
- Top-nøkler skrevet i fast rekkefølge (`config_path`, `models`, `truth_mode`); ingen `sort_keys` som endrer meningsinnhold (array-rekkefølge beholdt).

### 6) Ripgrep-audit

- `rg -n "validate_model_compare_config_strict|model_compare_validation_evidence\.json|CONFIGS_PATH" gx1/ scripts/ docs/`  
  Treff: `gx1/scripts/run_model_compare_2025.py` (definisjon og bruk), `gx1/tests/test_model_compare_config_validation.py` (tester), `docs/REPORT_model_compare_config_validation.md` (dokumentasjon). Ingen treff i `scripts/`. Ingen alternative validatorer eller gamle navn i bruk.

### 7) Tester

- **gx1/tests/test_model_compare_config_validation.py:**  
  - `test_unknown_model_fails_and_writes_evidence`: ukjent modell gir RuntimeError med UNKNOWN_MODEL og evidence-fil med FAIL + checks.  
  - `test_evidence_format_has_checks_keys`: sjekker at EVIDENCE_TOP_KEYS, CHECK_KEYS, MODEL_REC_KEYS er definert som forventet.
