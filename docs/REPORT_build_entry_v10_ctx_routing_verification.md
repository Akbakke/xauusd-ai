# Rapport: Verifikasjon av routing for build_entry_v10_ctx_training_dataset

**Dato:** 2025-02-16  
**Mål:** Bekreft at base-script ikke lenger brukes og at routing er 100 % til _signal_only, _legacy og gx1.datasets.entry_v10_ctx_legacy.

---

## 1) rg-kommandoer (output inkludert)

### 1.1 `rg -n "build_entry_v10_ctx_training_dataset\.py" . --glob '!docs/AUDIT_*'`

```
./docs/FULLYEAR_2025_ROBUST_PIPELINE.md:7:The old name `build_entry_v10_ctx_training_dataset.py` is deprecated and stubbed (fail-fast).
./docs/FULLYEAR_2025_TRAINING_PIPELINE_FIX.md:7:The old name `build_entry_v10_ctx_training_dataset.py` is deprecated and stubbed (fail-fast).
./gx1/scripts/build_entry_v10_ctx_training_dataset.py:9:    "[DEPRECATED] build_entry_v10_ctx_training_dataset.py is removed. "
```

**Vurdering:** Kun kanonisk deprecation-melding (2 docs) og stub-tekst (1). Ingen bruk av base-script.  
**PASS**

---

### 1.2 `rg -n "gx1\.scripts\.build_entry_v10_ctx_training_dataset\b" .`

Kun treff i `docs/AUDIT_build_entry_v10_ctx_training_dataset_references.md` (historikk / «Before»-tabell og sjekkliste). Ingen treff i kode (gx1/, scripts/).  
**PASS**

---

### 1.3 `rg -n "python -m gx1\.scripts\.build_entry_v10_ctx_training_dataset\b" .`

Kun i audit-dokumentet (beskrivelse av at base ikke skal brukes). Ingen faktiske CLI-kall.  
**PASS**

---

### 1.4 `rg -n "build_entry_v10_ctx_training_dataset\b" gx1/ scripts/ docs/ .github/ Makefile`

- **gx1/:** Kun `gx1/scripts/build_entry_v10_ctx_training_dataset.py:9` (stub-melding).
- **scripts/:** Ingen treff.
- **docs/:** AUDIT (historikk) + FULLYEAR_2025_*.md:7 (kanonisk setning).
- **.github/ Makefile:** Ingen treff.

Ingen kode eller runbooks som bruker base-navnet uten suffiks.  
**PASS**

---

### 1.5 `rg -n "(importlib|__import__|exec\(|eval\(|subprocess\.run\(|os\.system\()" gx1/ scripts/ .github/ Makefile`

Treff er f.eks. `subprocess.run` (git, andre verktøy), `model.eval()`, `importlib.import_module` (generisk). Ingen av dem invokerer `build_entry_v10_ctx_training_dataset` (uten suffiks).  
**PASS**

---

## 2) Entrypoints

| Kommando | Resultat |
|----------|----------|
| `python -m gx1.scripts.build_entry_v10_ctx_training_dataset_legacy --help` | Exit 0, usage vises. |
| `python -m gx1.scripts.build_entry_v10_ctx_training_dataset_signal_only --help` | Exit 0, usage vises. |
| `python -m gx1.scripts.build_entry_v10_ctx_training_dataset` | Exit 1, RuntimeError med `[DEPRECATED]` og peker til _signal_only / _legacy. |

**PASS** (alle tre som forventet)

---

## 3) Base script feiler som forventet

Kjørt:  
`/home/andre2/venvs/gx1/bin/python -m gx1.scripts.build_entry_v10_ctx_training_dataset`

Output:

```
RuntimeError: [DEPRECATED] build_entry_v10_ctx_training_dataset.py is removed. Use build_entry_v10_ctx_training_dataset_signal_only.py or build_entry_v10_ctx_training_dataset_legacy.py.
```

**PASS**

---

## 4) Imports (compute_session_histogram / build_dataset / write_manifest)

| Fil | Import |
|-----|--------|
| `gx1/tests/test_session_histogram_logging.py:5` | `from gx1.datasets.entry_v10_ctx_legacy import compute_session_histogram` |
| `gx1/tests/test_entry_v10_ctx_dataset_contract.py:135` | `from gx1.datasets.entry_v10_ctx_legacy import build_dataset` |
| `gx1/tests/test_entry_v10_ctx_dataset_contract.py:144` | `from gx1.datasets.entry_v10_ctx_legacy import write_manifest` |
| `gx1/datasets/entry_v10_ctx_legacy.py` | Re-eksporterer fra `gx1.scripts.build_entry_v10_ctx_training_dataset_legacy` |

Ingen importer direkte fra `gx1.scripts.build_entry_v10_ctx_training_dataset`. Tester bruker stabil flate `gx1.datasets.entry_v10_ctx_legacy`.  
**PASS**

---

## 5) Rydding – DELETE_CANDIDATE (kun forslag, ikke slettet)

| Fil | Begrunnelse | Risiko |
|-----|-------------|--------|
| *(ingen)* | Ingen filer som *kun* har gammelt navn i kommentar/historikk og som bør slettes. Stub-scriptet beholdes med vilje (fail-fast). FULLYEAR-docs har kun kanonisk deprecation-setning. AUDIT er bevis/historikk. | - |

Valgfri oppdatering i audit-dokumentet: oppdater sjekkliste slik at «Legacy/signal_only --help» og «Base script fails» markeres som fullført (hak av), siden de nå er verifisert.

---

## Oppsummering

| Punkt | Status |
|-------|--------|
| 1.1 rg "build_entry_v10_ctx_training_dataset\.py" (ekskl. AUDIT) | **PASS** |
| 1.2 rg "gx1\.scripts\.build_entry_v10_ctx_training_dataset\b" | **PASS** |
| 1.3 rg "python -m gx1\.scripts\.build_entry_v10_ctx_training_dataset\b" | **PASS** |
| 1.4 rg token i gx1/ scripts/ docs/ .github/ Makefile | **PASS** |
| 1.5 rg dynamic dispatch | **PASS** |
| 2 Entrypoints _legacy og _signal_only --help | **PASS** |
| 3 Base script feiler med [DEPRECATED] | **PASS** |
| 4 Imports bruker entry_v10_ctx_legacy | **PASS** |
| 5 DELETE_CANDIDATE | Ingen kandidater; valgfri audit-sjekkliste-oppdatering. |

**Konklusjon:** Base-script brukes ikke. Routing er korrekt til _signal_only, _legacy og `gx1.datasets.entry_v10_ctx_legacy`. Ingen rester som krever patch.

---

## Valgfri patch (audit-sjekkliste)

Hvis du vil oppdatere audit-dokumentet slik at verifiserte punkter er haked av:

```diff
--- a/docs/AUDIT_build_entry_v10_ctx_training_dataset_references.md
+++ b/docs/AUDIT_build_entry_v10_ctx_training_dataset_references.md
@@ -116,9 +116,9 @@
 - [x] **0** CLI invocations of base script in scripts, CI, Makefile, hooks.
-- [ ] **Legacy/signal_only --help:** `python -m gx1.scripts.build_entry_v10_ctx_training_dataset_legacy --help` and `..._signal_only --help` succeed.
-- [ ] **Base script fails:** `python -m gx1.scripts.build_entry_v10_ctx_training_dataset` exits with DEPRECATED RuntimeError.
+- [x] **Legacy/signal_only --help:** `python -m gx1.scripts.build_entry_v10_ctx_training_dataset_legacy --help` and `..._signal_only --help` succeed.
+- [x] **Base script fails:** `python -m gx1.scripts.build_entry_v10_ctx_training_dataset` exits with DEPRECATED RuntimeError.
 - [ ] **pytest:** `pytest gx1/tests/test_session_histogram_logging.py gx1/tests/test_entry_v10_ctx_dataset_contract.py -v` pass.
```

Pytest-punktet står uendret (avhengig av miljø/legacy-moduler).

---

## Beslutningskjede og mal

- **AUDIT** peker hit: [AUDIT_build_entry_v10_ctx_training_dataset_references.md](AUDIT_build_entry_v10_ctx_training_dataset_references.md) inneholder «Final verification: see REPORT_…».
- **ADR-logg:** [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) har oppføring for dette subsystemet.

**Frosset område:** Dette er et *done subsystem*. Eventuelle endringer skal være eksplisitte, kontrakt-endringer eller nye filer — ikke gjenbruk av gamle navn.

**Mal for andre monolitter:** Prosessen *audit → stub → stabil importflate → report → GO* er oppskriften å gjenbruke når dere deler opp andre monolitter i GX1.
