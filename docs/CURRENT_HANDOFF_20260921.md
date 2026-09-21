# CURRENT HANDOFF — 2026-09-21 (kveld)

Dette er inngangspunktet for en ny agent. Les i denne rekkefølgen:
1. `CLAUDE.md` (konstitusjonen — regel 1–25 og kapasitetsgrensene er absolutte).
2. Dette dokumentet (nåtilstand + eksakt vei videre).
3. `docs/PIPELINE_FEATURE_FIDELITY_REVIEW_20260921.md` (funnregisteret F-1..F-31 som
   styrer inneværende og neste bølger, med F-16-korreksjonen).
4. `bash scripts/gx1_handover.sh --check` (den eksekverbare statuseieren — utrangerer
   all prosa, også dette dokumentet).
Agent-minnet (`~/.claude/projects/-home-andre2/memory/`) har arbeidsnotatet
`project_gx1_fullsup_candidate_campaign_20260921.md` med env-fil og prosedyredetaljer.

## Hva systemet ER (koblingskart)

- **Én beslutningsautoritet:** unik argmax over `entry_action_q_bps` (LONG/SHORT/FLAT)
  og `unified_exit_action` (HOLD/EXIT_NOW) fra samme bundle/delte encoder. Fitted-Q i
  rå bps; ingen kalibrerte sannsynligheter, ingen post-modell-regler. Ties feiler lukket
  (i VAL: epoch blir uskårbar, ikke fatal — `[VAL_EXACT_TIE_UNSCORABLE]`, trainer-eid).
- **Flaten (etter fidelity-bølgen 21.09, commit `5ec2e537`):** signal v35 = 243 dim
  (26 frossen base + 217 selected hvorav 150 mandatory + 67 kandidater); ctx_cont 71,
  ctx_cat 1 (`session_id`); per-TF-matrise V23 = **190 felt × 5 TF-er** (M5/M15/H1/H4/D1),
  partisjonert eksakt over 8 spesialistfamilier (structure_swing 20, smc_liquidity 50,
  trend_ema 29, vol_compression 5, momentum_flow 22, session_regime 4, chart_geometry 38,
  price_action_candle 22). LES ALLTID tallene ved å eksekvere
  `gx1/contracts/entry_model_native_signal_v1.py` og
  `gx1/features/entry_specialist_feature_groups_v1.require_multi_tf_specialist_routing_v4`
  — aldri fra dette dokumentet (regel 4/13).
- **Kjedens produksjonsrekkefølge** (alt gjennom `scripts/gx1_capped_run.sh`, én tung
  jobb om gangen): squeeze-seksklokke-fit (TRAIN-only, immutable) →
  `scripts/run_seq513_rebuild_chain_v1.sh` (M5-kilde → 5-TF-cache → ranker →
  signalmanifest → M1/M5-featureflater → preflight → kombinert Entry/Exit-dataset) →
  audit-kjede (causality/foundations/liveness/seq-reconstruction/pretrain) →
  recipe-materialisering (smoke+kandidat) → trainability readiness → kandidat-readiness
  (konsumerer eksekvert smoke-bundle-audit) → immutable launch-gate → launcher
  (`run_entry_model_native_pretest_technical_train_v1 --dry-run` så `--execute`).
- **Treningsdrift:** 2-timers guardede segmenter (veggklokke 7200 s, exit 75,
  `reason=wall_clock_limit_7200s`), sesjonen resumeres eksakt via
  `CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json` — relansér samme launcher-kommando.
  Guard: 70 °C kjerne / 80 °C minne / 300 W limit / 310 W draw / 12 GiB (operatørvedtak
  20–21.09). passord.md stashes rundt clean-tree-gates (`git stash push -- passord.md`,
  ALDRI les/commit), poppes etterpå.
- **Ytelse (målt):** ~1,19–1,25 s/steg batch 8 på 240-flaten etter
  effektivitetsbølgen (commit `cd602015`, bit-identitet bevist med seedet probe);
  epoch = ~39k steg ≈ 7,2 segmenter + VAL ~1 t.

## Hvor vi STÅR akkurat nå (21.09 ~23:00)

- Fidelity-bølgen er committet (`5ec2e537`): F-9/10/11/13/14/15/18/19/20/21/22/23/24
  implementert; 190/243-flate; alle tester grønne UNNTATT 9 transisjonelle
  (test_entry_handover_control 7 + train_recipe 1 + audited_dataset 1) som leser den
  EKTE launch-statens pre-bølge-recipes — de grønnes ved state-rebind i steg 5 under.
- **Squeeze-refit kjører** (bakgrunn): identiske inputs som 20260920T112706Z-fitten,
  eneste endring er vindu-eksakt `bollinger_relative_bandwidth` (pandas rolling var
  start-avhengig i siste ulp — avdekket av chunk-carry-testen). Output-katalog i
  env-filas `$SQZ2`.
- Forrige kandidat (`ENTRY_V10_FULLSUP_CANDIDATE_20260921T115456Z`, 240-flaten) ble
  SKROTET på operatørordre ved ~11 200 steg; sesjonskatalogen ligger som evidens under
  `$DS2` — slett aldri for hånd (regel 9; retention-eieren senere).

## EKSAKT VEI VIDERE (i rekkefølge)

1. **Verifiser squeeze-refit**: `$SQZ2/manifest.json` finnes, v3-schema, seks
   `*_params.json`; noter manifest-sha256.
2. **Kjør chain-driveren** med nytt run-id og nytt event-root:
   `scripts/run_seq513_rebuild_chain_v1.sh --run-id PRETEST_V11_<UTC> --event-root
   $ART/PRETEST_V11_ENRICHED_<UTC> --feature-ranking-json <event>/RANKING_<UTC>.json
   --preflight-out-dir <event>/preflight --m1-lifecycle-pair-manifest-json $PAIR
   --m1-lifecycle-pair-generation-root <PAIR_GEN-roten> --registry-fit-inner-end
   $INNER_END --volatility-squeeze-manifest $SQZ2/manifest.json
   --volatility-squeeze-manifest-sha256 <sha> --history-start $HISTORY_START
   --train-start $TRAIN_START --train-end $TRAIN_END --val-start <TRAIN_END+5min>
   --val-end $VAL_END --test-start $VAL_END --test-end <som FULLSUP>`.
   Alle vindusverdier står i env-fila (`gx1_candidate_campaign_env_20260920.sh` i
   minnekatalogen, kopier til scratchpad og `source`). TEST forblir FORSEGLET.
3. **Dataset + audit-kjede**: samme wrapper/audit-mønster som FULLSUP-byggingen 20.09
   (stiene og artefaktnavnene i env-fila/`$DS2` er malen; nytt dataset-root blir
   `$ART/V11_...`). Forvent `SESSION_ID_CANONICAL_MISMATCH`-klasse-abort hvis noe er
   galt — kjeden er fail-closed hele veien; les feilen, reparér årsaken, aldri symptomen.
4. **Recipes → readiness → gate**: mønsteret fra 21.09 (se minnenotatet §"Prosedyre-arv");
   trainer-CLI-malene ligger i minnekatalogen (`smoke_trainer_cli.json` 32-raders smoke,
   `candidate_trainer_cli.json` batch 8/30 epochs/patience 5 — konstitusjonsbundet).
   NB: en 32-raders gate-smoke må EKSEKVERES på nye artefakter før kandidat-readiness
   (smoke-bundle-audit er gate-input; forvent den kjente smoke-skala-FAIL-en
   «never top-ranked» — kandidat-readiness konsumerer nettopp den).
5. **State-rebind + commit**: oppdater `current_source_technical_recipe` (status-løpet
   MATERIALIZED_DRY_RUN_PENDING → PASS → EXECUTED → GATE_READY) og
   `current_pretest_trainability_readiness` i `PROJECT_STATE_xau_direction_launch.json`;
   da grønnes de 9 transisjonelle testene. Kjør `gx1_handover.sh --check` + full suite.
6. **Kandidat-launch** på 243-flaten + nattloop (segment-resume ved hvert exit-75-varsel).
   Epoch-1-VAL: forvent mulig `[VAL_EXACT_TIE_UNSCORABLE]` i epoch 1–2 (undertrent
   modell; ufarlig, epoch diskvalifiseres, treningen fortsetter).

## Etter kandidaten (roadmap, fra fidelity-reviewens §4)

- **Attribusjonsbølge (sub-C, kode uten rebuild):** reaktiver `feature_mask_ablation`
  i `evaluate_entry_candidate_selective_edge_v1` (schema-slot finnes; prior art commit
  `0b139646`); Entry-gate-evidens i bundler (speil `unified_exit_gate_evidence_v1`);
  fiks usefulness-donorplanen (nabobar-degenerasjon); deklarér counterfactual-konvensjon
  (mean-substitusjon, ikke rå 0). Protokollen med regel 2f-grenser står i reviewens §3.
- **Preregistrert evaluering:** `docs/PREREGISTERED_DIRECTION_TEST_20260820.md` er
  frossen (coverage-grid, HAC, sirkulær-skift-null 512 trekk, ≥5 seeds, VAL→TEST-regel).
  Myntkast-nullen beregnes EKSAKT av evaluatoren; −13,16-tallet er dødt.
- **Arkitektur-hypoteser (F-27..F-31)** blir målbare spørsmål når ablasjonsmatrisen har
  kjørt. **DST (F-25)** er en egen preregistrert beslutning. **Hopper-flytting**
  (≤2 500 NOK-ramme) vurderes når epoch 1–3-VAL viser om modellen lærer.

## Kjente feller (ikke gjenta)

- Recipe binder kildebytes: ENHVER kildeendring → ny recipe-kjede; endringer i
  trainer-closuren mens en kandidat kjører brekker neste segment-relansering.
- `--trainer-cli-json` tar JSON-TEKST, ikke filsti; `--m5-prebuilt` = split-manifestets
  `inputs.source_parquet`; capped-runnerens `--cuda-producer` krever absolutt
  kanonisk python-sti; fit-produsentenes `--output-dir` må pre-eksistere.
- Handover-heredocens argv: argv[1]=repo, argv[2]=launch-state (fikset 20.09 — ikke
  reintroduser `.parent`-feilen).
- `_v1_atr14`-navnet finnes fortsatt i basic_v1-DataFrame (ikke-signal-konsumenter);
  signalfeltet er `_v1_atr14_bps`.
