# Ghost purge plan — One Universe TRUTH

Mål: Én sann måte å kjøre TRUTH på. Alt annet fjernes fra repo (git rm), ikke deprecated eller quarantine.

---

## 1. Canonical TRUTH entrypoints (SSoT)

Følgende er **allowed to exist** og utgjør den kanoniske TRUTH-path:

| Artifact | Rolle |
|----------|--------|
| `gx1/scripts/run_truth_e2e_sanity.py` | TRUTH E2E entrypoint; kjører replay via replay_chunk + replay_merge |
| `gx1/execution/replay_chunk.py` | 1W1C replay; importerer GX1DemoRunner, kaller process_chunk |
| `gx1/execution/oanda_demo_runner.py` | Runner; policy snapshot, exit loop, replay_eval_collectors |
| `gx1/execution/replay_merge.py` | merge_artifacts_1w1c; brukes av run_truth_e2e_sanity etter process_chunk |
| `gx1/utils/truth_banlist.py` | TRUTH/SMOKE banlist (banned_modules, assert_truth_*), canonical policy path |
| Canonical policy YAML | Path fra truth file (GX1_CANONICAL_TRUTH_FILE); må matche TRUTH_CANONICAL_POLICY_RELATIVE |
| `gx1/scripts/run_fullyear_2025_truth_proof.py` | Thin wrapper som kun invokerer run_truth_e2e_sanity --full-year |

Ingen av disse importerer eller kjører `replay_eval_gated_parallel` eller noe under `gx1/_quarantine`.

---

## 2. Candidates for deletion

### 2.1 `gx1/scripts/replay_eval_gated_parallel.py`

- **Status:** Filen finnes ikke i repo (allerede fjernet eller aldri committet).
- **Bevis:** `rg -l "replay_eval_gated_parallel" gx1/` viser kun referanser i run_truth_e2e_sanity (gate), gx1_doctor, quarantine_paths, docs og _quarantine. Ingen canonical entrypoint importerer den.
- **Handling:** Ikke slett (finnes ikke). Legg til `gx1.scripts.replay_eval_gated_parallel` i `truth_banlist.banned_modules` slik at import → hard-fail. Oppdater gate-melding i run_truth_e2e_sanity til «must not exist in repo (ghost purge)».

### 2.2 Hele `gx1/_quarantine/`

- **Bevis:** Ingen treff fra canonical entrypoints:
  - `run_truth_e2e_sanity`: importerer kun replay_chunk, replay_merge, truth_banlist.
  - `replay_chunk`: importerer oanda_demo_runner, replay_eval_collectors (execution).
  - `oanda_demo_runner`: importerer truth_banlist, policy/exit_transformer_v0, gx1.scripts.replay_eval_gated (flush_replay_eval_collectors).
  - `replay_merge`: ingen import fra _quarantine.
  - `truth_banlist`: ingen import fra _quarantine.
- **rg:** Ingen `from gx1._quarantine` eller `import.*_quarantine` i kode under gx1 utenom tests (test_rule_exit_banlist nevner _quarantine som unntak i docstring/skip-logikk).
- **Handling:** `git rm -r gx1/_quarantine` (hele mappen).

---

## 3. Delete order

1. **Først:** Oppdater kode som refererer til _quarantine eller replay_eval_gated_parallel (se §4), slik at ingen import eller filbane forutsetter _quarantine.
2. **Deretter:** `git rm -r gx1/_quarantine`.
3. Ingen andre filer slettes (replay_eval_gated_parallel.py finnes ikke).

---

## 4. Required replacements

| Sted | Endring |
|------|--------|
| `gx1/utils/truth_banlist.py` | Legg `gx1.scripts.replay_eval_gated_parallel` til `BANLIST.banned_modules`. |
| `gx1/scripts/run_truth_e2e_sanity.py` | I `_assert_truth_no_legacy_replay`: oppdater melding ved «file exists» til at script må ikke finnes i repo (ghost purge). |
| `gx1/tools/gx1_doctor.py` | Legacy-replay-sjekk: ved `legacy_path.exists()` → fail med melding «must not exist (ghost purge)»; ellers OK. |
| `gx1/tests/test_rule_exit_banlist.py` | Skip-logikk som sjekker `_quarantine` i path kan beholdes (gir ingen treff etter sletting) eller forenkles; ingen funksjonell avhengighet av at _quarantine finnes. |
| `gx1/tools/quarantine_paths.py` | Behold `gx1/scripts/replay_eval_gated_parallel.py` i forbidden-liste (så path ikke «gjenopplives»). |
| Docs som peker på slettet/ghost | Oppdater eller fjern referanser til replay_eval_gated_parallel og _quarantine (se §6). |

Ingen av de tillatte filene importerer noe fra _quarantine; ingen erstatning av import nødvendig der.

---

## 5. Hard gates

- **truth_banlist.py:** `gx1.scripts.replay_eval_gated_parallel` i `banned_modules` → hvis noen importerer i TRUTH/SMOKE → `assert_truth_banlist_clean` feiler.
- Eksisterende gate i run_truth_e2e_sanity: scriptet må ikke være i sys.modules og filen må ikke finnes på disk; melding oppdatert til ghost purge.

---

## 6. Verification commands

Etter purge:

```bash
# Preflight / validering
python -m gx1.scripts.run_truth_e2e_sanity --validate-only

# Mikro replay (kort vindu)
python -m gx1.scripts.run_truth_e2e_sanity --start-ts 2025-06-03 --end-ts 2025-06-05  # eller tilsvarende

# No-ghost test (B3)
python -m pytest gx1/tests/test_no_ghost_paths.py -v

# Grep: ingen ghost-tokens i gx1 (unntatt docs/GHOST_PURGE_PLAN.md og test_no_ghost_paths.py)
rg -n "replay_eval_gated_parallel|_DEPRECATED|legacy_replay|exit_farm_v2_rules|exit_master_v1|fixed_bar" gx1 --glob '!*.md' --glob '!**/test_no_ghost_paths.py' --glob '!**/GHOST_PURGE_PLAN.md'
# Forventet: ingen treff under gx1/ (unntatt tillatte unntak i plan/doc og test).
```

---

## 7. Ghost tokens (for test_no_ghost_paths.py)

Disse strengene skal ikke forekomme i gx1-kode (unntatt i docs/GHOST_PURGE_PLAN.md og i test_no_ghost_paths.py selv):

- `replay_eval_gated_parallel`
- `_DEPRECATED`
- `legacy_replay`
- `exit_farm_v2_rules`
- `exit_master_v1`
- `fixed_bar`

Evt. flere som identifiseres i planen (f.eks. andre legacy script-navn som kun fantes under _quarantine).
