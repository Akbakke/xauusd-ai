# STOPP OG SE TILBAKE (MÅ VÆRE GRØNN FØR TUNING)

Dette er en hard sjekkliste før vi gjør *én eneste* ny endring i entry/exit/policy.
Målet er å unngå «feil script / feil YAML / feil datasett / feil output-dir»-feil og å tvinge frem fakta.

## 1) RUN_CTX / ENV (grunnmur)

- [ ] **Repo-root**: Stemmer `RUN_CTX root` med forventet repo (ikke en kopi/backup)?
- [ ] **HEAD**: Stemmer `RUN_CTX head` med forventet commit?
- [ ] **Entrypoint**: Kjørte vi riktig script (f.eks. `scripts/run_fullyear_prebuilt.sh` / `gx1/scripts/replay_eval_gated_parallel.py`)?

## 2) Policy / data / prebuilt (kilde-sannhet)

- [ ] **Policy YAML path**: Stemmer policy path (og er det *den* YAML-en vi tror)?
- [ ] **Policy checksum**: Stemmer checksum for policy YAML (for å unngå «feil fil»)?
- [ ] **Dataset path**: Stemmer dataset path (riktig år, riktig bid/ask, riktig granularitet)?
- [ ] **Prebuilt path**: Stemmer `GX1_REPLAY_PREBUILT_FEATURES_PATH`?
- [ ] **Prebuilt sha**: Stemmer prebuilt sha (fra perf JSON / footer)?

## 3) Output-dir hygiene (ingen gjenbruk)

- [ ] **Ny output-dir**: Output-dir er ny (hard reset), ikke gjenbrukt.
- [ ] **Tripwire**: PREBUILT-tripwires passerte (footer/perf: `tripwire_passed=true`).

## 4) Minimums-bevis (før «hvorfor 0 trades»)

- [ ] **EntryPrediction finnes**: `killchain_n_entry_pred_total > 0`
- [ ] **Signal over threshold finnes**: `killchain_n_above_threshold > 0`
  - Hvis `killchain_n_above_threshold == 0`: dette er *threshold/signal*-problem, ikke post-gates.

## 5) Hvis trades = 0: hva drepte dem?

- [ ] **Trades faktisk 0**: `killchain_n_trade_created == 0`
- [ ] **Dominerende blokkering**: `killchain_top_block_reason` peker på én tydelig årsak.
- [ ] **Smoking gun**:
  - Hvis `killchain_n_above_threshold > 0` og `killchain_n_trade_created == 0` → post-gates dreper.
  - Identifiser *hvilken* (session/vol/regime/risk/position/cooldown).

## 6) STOPP-OG-SE-BAKOVER output (hvor å se)

- **Per chunk**: `chunk_*/chunk_footer.json`
- **Perf**: `perf_*.json`
- **Killchain rapport**: `reports/replay_eval/FULLYEAR_KILLCHAIN_ANALYSIS_<runid>.md`

