# Rapport: Routing for train_entry_v10_ctx_transformer_signal_only

**Dato:** 2025-02-16  
**Mål:** Canonical train-script er `train_entry_v10_ctx_transformer_signal_only.py`; gamle navn er stub (fail-fast). Ingen runbooks/CLI/tester peker på gamle navn.

---

## rg-kommandoer (output)

### `rg -n "train_entry_v10_ctx_signal_only\.py" .`

```
./gx1/scripts/train_entry_v10_ctx_signal_only.py:9:    "[DEPRECATED] train_entry_v10_ctx_signal_only.py is removed. "
```

Kun treff i stub-filen (deprecation-melding). **PASS**

---

### `rg -n "train_entry_v10_ctx_transformer_signal_only" .`

```
./gx1/scripts/train_entry_v10_ctx_signal_only.py:4:DEPRECATED: This script is removed. Use train_entry_v10_ctx_transformer_signal_only.py.
./gx1/scripts/train_entry_v10_ctx_signal_only.py:10:    "Use: python -m gx1.scripts.train_entry_v10_ctx_transformer_signal_only ..."
```

Kun i stub (peker til nytt navn) og i selve det nye scriptet. **PASS**

---

### `rg -n "gx1\.scripts\.train_entry_v10_ctx_signal_only\b" .`

Ingen treff. Ingen importer eller kall av gammel modul. **PASS**

---

### `rg -n "train_entry_v10_ctx_signal_only" .` (bredt)

Kun stub-fil (linje 9). Ingen treff i docs/, .github/, runbooks, tester. **PASS**

---

## Entrypoint --help (exit 0)

```bash
python -m gx1.scripts.train_entry_v10_ctx_transformer_signal_only --help
```

Exit 0. Viser bl.a. `--train-parquet`, `--val-parquet`, `--out-bundle-dir`, `--strict`, `--strict-extended`. **PASS**

---

## Gammelt navn fail-fast

```bash
python -m gx1.scripts.train_entry_v10_ctx_signal_only
```

Output:

```
RuntimeError: [DEPRECATED] train_entry_v10_ctx_signal_only.py is removed. Use: python -m gx1.scripts.train_entry_v10_ctx_transformer_signal_only ...
```

Exit 1. **PASS**

---

## Konklusjon

| Sjekk | Resultat |
|-------|----------|
| rg gamle scriptnavn (.py) | **PASS** (kun stub) |
| rg nytt scriptnavn | **PASS** (stub + ny fil) |
| rg gammel modulpath | **PASS** (0 treff) |
| rg bredt gamle navn | **PASS** (kun stub) |
| Nytt script --help exit 0 | **PASS** |
| Gammelt script fail-fast | **PASS** |

**Konklusjon: PASS.** Canonical script er `gx1/scripts/train_entry_v10_ctx_transformer_signal_only.py`. Gamle navn er erstattet med fail-fast stub. Ingen runbooks/CLI/tester peker på gamle navn.
