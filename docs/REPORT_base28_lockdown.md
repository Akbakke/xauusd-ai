# BASE28 lockdown: canonical model and single-model only

**Mål:** Låse BASE28 som eneste kanoniske/standard modellvalg i model-compare og truth-config routing. Alt annet (f.eks. BASE28_CTX2PLUS_T1) er eksplisitt "research-only" og krever eksplisitt flagg. Ingen fallback, ingen silent skip; feil hard-failer tidlig.

---

## A) Endringer (SSoT)

### Config (gx1/configs/model_compare_2025/model_configs.json)

- **canonical_model:** Toppnivå-felt lagt til: `"canonical_model": "BASE28"`. SSoT for default modellvalg.
- **models:** Uendret: kun `BASE28` og `BASE28_CTX2PLUS_T1`.
- **signal_bridge_contract_sha256:** Uendret og fortsatt brukt.

### Script (gx1/scripts/run_model_compare_2025.py)

- **argparse --models:** default er tom streng `""` (ikke "BASE28"). Resolveres til `config["canonical_model"]` via `_parse_models(s, config)`.
- **_get_canonical_model(config):** Returnerer `config["canonical_model"]`; hard-fail hvis mangler eller ikke i `config["models"]`.
- **_parse_model(s, config) -> str:** Tar `(s, config)`; returnerer canonical eller eksplisitt modell; aldri hardkodet BASE28.
  - Tom / `"CANONICAL"` / kanonisk navn fra config → `[canonical_model]`.
  - `"ALL"` → hard-fail med melding: "ALL is banned in this harness".
  - Flere modeller (komma/space) → hard-fail: "Multi-model runs are banned; run one at a time."
  - Ellers én modell → `[parts[0]]`.
- **validate_model_compare_config_strict:** Validerer alltid `canonical_model` først (og skriver den i evidence). Ved manglende eller ugyldig `canonical_model` skrives evidence med FAIL og raise.
- **Evidence:** To tydelige formater:
  - **Normal validering:** Toppnøkler `("config_path", "canonical_model", "models", "truth_mode")`; `models` er liste av per-modell records med `checks`. `_write_evidence(run_dir, payload)` bruker `keys=None` → EVIDENCE_TOP_KEYS.
  - **UNKNOWN_MODEL:** Flat payload, ingen `"models"`-nøkkel. Nøkler: `config_path`, `canonical_model`, `truth_mode`, `requested_models`, `unknown_models`, `available_models`, `status`, `reason`. `_write_evidence(run_dir, payload, keys=EVIDENCE_UNKNOWN_FAIL_KEYS)`.
- **--validate_only uten --models:** Validerer canonical model (BASE28).
- **Guard:** `--models BASE28,BASE28_CTX2PLUS_T1` (eller andre flere) → hard-fail. `--models ALL` → hard-fail.
- **WARNING:** Ved `--models BASE28_CTX2PLUS_T1` skrives til stderr: "RUNNING NON-CANONICAL MODEL: BASE28_CTX2PLUS_T1 (research-only)." (ikke en fail).
- **--parallel:** Run cases in parallel (threshold sweep), not per-model. Each case = 1W1C TRUTH, no chunking.

---

## B) rg-kommandoer og forventede treff

### canonical_model

```bash
rg -n "canonical_model" gx1/
```

**Forventet (utvalg):**  
- `gx1/configs/model_compare_2025/model_configs.json`: 1 (toppnivå-nøkkel).  
- `gx1/scripts/run_model_compare_2025.py`: EVIDENCE_TOP_KEYS, _get_canonical_model, _fail_unknown_models, validate_model_compare_config_strict (canonical check, ordered_models, evidence_payload), _parse_models docstring, main (default, canonical_model var, parallel guard).  
- `gx1/configs/model_compare_2025/README.md`: beskrivelse av default/canonical.

### ALL / Multi-model / banned

```bash
rg -n "ALL\\b|Multi-model|banned" gx1/scripts/run_model_compare_2025.py
```

**Forventet:**  
- Docstring: "Multi-model and \"ALL\" are banned".  
- _parse_models: "ALL" -> hard-fail; "multi-model banned"; "[FAIL_FAST] --models ALL is banned"; "Multi-model runs are banned; run one at a time."  
- main: "--parallel requires exactly one model (multi-model runs are banned)."

### BASE28_CTX2PLUS_T1

```bash
rg -n "BASE28_CTX2PLUS_T1" gx1/ docs/
```

**Forventet:** Kun config (nøkkel under `models`), eksplisitte docs (README, rapporter) og CLI-hjelp/varsler – **aldri** som default eller auto-inkludert i kjørevei.  
- `gx1/configs/model_compare_2025/model_configs.json`: 1 (models-key).  
- `gx1/scripts/run_model_compare_2025.py`: docstring (research-only), --validate_only eksempel, ALL-feilmelding eksempel, --models help-tekst.  
- `gx1/configs/model_compare_2025/README.md`: research-only, optional.  
- `gx1/execution/zero_trades_diag.py`: kun kommentar (run_id-eksempel).  
- `docs/REPORT_*.md`: historikk/verifikasjon.

---

## C) Kommandoer som må fungere

| Kommando | Forventet |
|----------|-----------|
| `python -m gx1.scripts.run_model_compare_2025 --validate_only` | Validerer canonical (BASE28). Exit 0. Evidence inneholder `canonical_model`. |
| `python -m gx1.scripts.run_model_compare_2025 --validate_only --models BASE28_CTX2PLUS_T1` | Validerer BASE28_CTX2PLUS_T1. Exit 0. WARNING på stderr. |
| `python -m gx1.scripts.run_model_compare_2025 --models BASE28` | Kjører BASE28 (kanonisk). Exit 0 ved ok. |

---

## D) Kommandoer som må feile (med forventet feilmelding)

| Kommando | Forventet feilmelding |
|----------|------------------------|
| `... --models BASE28,BASE28_CTX2PLUS_T1` | `[FAIL_FAST] Multi-model runs are banned; run one at a time. Requested: BASE28,BASE28_CTX2PLUS_T1` |
| `... --models ALL` | `[FAIL_FAST] --models ALL is banned in this harness. Use a single model (e.g. BASE28 or BASE28_CTX2PLUS_T1).` |

---

## E) Evidence-format

**Normal validering (validate_model_compare_config_strict OK/FAIL per model):**

- Toppnøkler: `config_path`, `canonical_model`, `models`, `truth_mode`
- `models`: liste av per-modell records med `checks`, `model`, `reason`, `status`
- `_write_evidence(run_dir, payload)` med `keys=None` (default EVIDENCE_TOP_KEYS)

**UNKNOWN_MODEL (flat payload, aldri per-modell records):**

- Nøkler: `config_path`, `canonical_model`, `truth_mode`, `requested_models`, `unknown_models`, `available_models`, `status`, `reason`
- `"models"` finnes IKKE i UNKNOWN_MODEL-evidence
- `_write_evidence(run_dir, payload, keys=EVIDENCE_UNKNOWN_FAIL_KEYS)`

---

## F) Self-check / tester

I `gx1/tests/test_model_compare_config_validation.py`:

- **test_validate_only_without_models_validates_canonical:** `--validate_only` uten `--models` validerer canonical (forventer exit 0).
- **test_models_all_fails_with_banned_message:** `--models ALL` må feile med melding om at ALL er banned.
- **test_models_multi_fails_with_multi_model_banned:** `--models BASE28,BASE28_CTX2PLUS_T1` må feile med multi-model banned.
- **test_models_base28_ctx2plus_logs_non_canonical_warning:** `--models BASE28_CTX2PLUS_T1` må logge "RUNNING NON-CANONICAL … (research-only)" til stderr.

Kjør: `pytest gx1/tests/test_model_compare_config_validation.py -v`

---

## G) Threshold sweep og cases

- **--thresholds "0.44,0.46,0.48":** Oppretter N cases (én per threshold). Hver case kjører samme model_id med `GX1_ENTRY_THRESHOLD_OVERRIDE` satt.
- **Output:** `OUT_ROOT/model_id/cases/thr_0p48/...` (deterministisk navn).
- **--parallel:** Parallelliser cases (threshold sweep), ikke modeller. `--max_parallel` (default 4) begrenser samtidige case-subprocesser.
- **cases_index.json:** I topp-run_dir; lister cases med case_id, run_root, env_overrides.

Eksempel:
```bash
python -m gx1.scripts.run_model_compare_2025 --thresholds "0.44,0.46,0.48" --parallel --max_parallel 3
```

---

## H) Ikke endret

- Ingen endringer i `run_fullyear_2025_truth_proof.py` eller trading/engine-semantikk.  
- Kun routing, SSoT, CLI-guards og docs.
