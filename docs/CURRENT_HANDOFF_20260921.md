> **ARKIVERT 2026-09-26 — IKKE JOBB HER.** Denne grenen (`audit/v9-premiere-20260905`) er slått
> inn i den eneste kodebasen `/home/andre2/src/GX1_CURRENT` (branch `work/gx1-current`) og tagget
> `archive/gx1-engine-audit-v9-20260926`. Denne mappen er bare felles git-lagring. Regler, status og
> neste steg: `GX1_CURRENT/GX1_RULES.md`, `GX1_CURRENT/CURRENT_HANDOVER.md`,
> `GX1_CURRENT/docs/CONSOLIDATION_20260926.md`. Alt under er historikk.

# Gjeldende oppdatering 23.09: V12 stoppet, Entry-forbedring aktiv

Les [ENTRY_NATIVE_NSTEP512_RESULT_20260923.md](ENTRY_NATIVE_NSTEP512_RESULT_20260923.md) først,
deretter [ENTRY_BOUNDED_NATIVE_INITIALIZATION_20260923.md](ENTRY_BOUNDED_NATIVE_INITIALIZATION_20260923.md),
deretter [ENTRY_TIME_FEEDBACK_AND_NSTEP_20260923.md](ENTRY_TIME_FEEDBACK_AND_NSTEP_20260923.md),
deretter [ENTRY_TARGET_AND_EMA_DIAGNOSIS_20260923.md](ENTRY_TARGET_AND_EMA_DIAGNOSIS_20260923.md),
deretter [ENTRY_REPAIR_20260923.md](ENTRY_REPAIR_20260923.md).
Checkpoint844 er bevart. Tre native kontroller er fullført:64 steg hver, totalt192.
Siste v10/v9-kontroll fullførte på668,39 sekunder, kilde2eee9ffa, returkode0.
Frossen v9-lærer og kontrollerte målstatistikker var uendret. Onlinev10 fikk ny
før-baseline og lavere TRAIN-MSE, men etter-fit velger Entry SHORT512/512 i begge
splitter. VAL netto vindusmark er-0,8045 Bps, score/utfall-rangering-0,0363,
og83 posisjoner er åpne. Læringsporten er ikke bestått. Ingen jobb er aktiv.
Ikke relanser noen fullført plan eller promoter ettervektene. PC er ikke restartet.
Rutingsrettelsen er numerisk kontrollert; bedre økonomisk seleksjon er ikke vist.
Kostnadskorreksjon alene og direkte Exit-gradient til Entry-hodet forklarer ikke
svak seleksjon i de kontrollerte tilfellene. Én låst analytisk readout-fit på
63 TRAIN /63 senere VAL feilet: VAL+1,3745 netto Bps mot alwaysSHORT+9,2374,
og side-MSE ble dårligere. Ingen vekter er promotert; ingen ny fit av denne prøven.
Gradientkontrollen begrunner ingen reset av usikkerhetsvekter: samlet retning
reduserte begge tapsgrupper lokalt i seks kontrollerte tilfeller. Entry-pris og
alle512 historiske prisutfall per side stemmer på63 TRAIN /63 senere VAL.
Videreføring av alle34 åpne TRAIN-sider er fullført:14 faktiske EXIT,20 fortsatt
åpne ved beregningsbudsjettet; alle34 prefikser bitlike og vektene uendret.
De63 opprinnelige Entry-valgene har nå55 lukkede/8 åpne og netto mark-14,1569 Bps
mot-6,2092 i det første vinduet. Dette er ulik observasjonstid, ikke full livsløps-PnL.
Brukeren har gjenopptatt målet med retning først: LONG bullish / SHORT bearish.
Les ENTRY_DIRECTION_FIRST_20260923.md først. Tidligere spørsmål om portefølje og
tid er ingen forutsetning for dette arbeidet. To retningsdiagnoser er fullført;
en native forecast-warmup er lagt til, men ingen nye markedsoptimizersteg er fullført. Frossen v10-readout bestod
ikke senere VAL. Direkte inputbaseline fant svakt 5-minutters signal på siste
TRAIN-år (51,06 % balansert treff), ikke dokumentert Entry-gevinst.
En paret kontroll viste at original v8 ikke har samme all-bearish forecast som
v10 med originalvektene. Original v8-baseline er nå fullført på1024 TRAIN /512 VAL: ingen horisont
bestod retningsporten. Neste ene native forsøk prioriterer eksisterende forecast-L1
på4096 TRAIN-rader, med original/før/etter på samme512 TRAIN /512 VAL.
Første warmup-oppstart stoppet før trening fordi referanse-loaderen bare tillot
v9. Eksplisitt v8-referanse er rettet og kontrollert mot ekte checkpoint/input.
Se siste avsnitt i ENTRY_DIRECTION_FIRST_20260923.md for feilkvittering og omfang.
Under kildefrys brukes Mac-operatørkopien CURRENT_HANDOVER.md og eksakt runtime-kvittering.
Ingen nye koeffisienter er promotert; ingen større trening eller jobb er aktiv.
Se ENTRY_DIRECTION_FIRST_20260923.md og ENTRY_GOAL_PROGRESS.json under
/home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923 for etterprøvbare bevis.
Tidligere oppstartsoppskrifter nedenfor er historikk. TEST forblir forseglet.

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
- **Flaten (etter duplikat-reparasjonen 22.09, commit `e8a639ca`):** signal v36 = 241 dim
  (24 frossen base + 217 selected hvorav 150 mandatory + 67 kandidater); ctx_cont 71,
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

## Oppdatering 22.09 morgen — duplikat-reparasjon + V12-rebuild

Kjeden under ble kjørt enkeltvis (chain-driveren er feil verktøy for pretest,
se «Kjente feller»). V11 kom helt fram til datasett + audit-kjede, og
**spesialist-auditen fant to eksakte duplikatkolonner** som fidelity-bølgen selv
hadde innført: `_v1_atr14_bps ≡ ctx_cont.atr_bps` og `_v1_ema_diff ≡
macd_line_atr`, begge målt bit-identiske over 464 244 TRAIN-rader. Dette er noe
ANNET enn kryss-flate-aliasene i `627894ac`: der er det to inputplan (ctx-skalar
vs. sekvenshistorikk), her er det samme signalvektor på samme klokke. Reparert i
`e8a639ca` ved å fjerne den redundante halvdelen av hvert par fra base-blokken
— flaten er nå v36 = 241. V11 er dermed skrotet og V12 bygges fra båndet
(V11s enriched-rammer ble korrekt avvist: den delte base-kontrakten binder
`base_feature_count` og `ordered_signal_dim`, som begge flyttet seg).

To driftsfeller bekreftet på nytt samme natt: (1) `pgrep -f <skriptnavn>` matcher
vaktens EGEN kommandolinje og rapporterte «kjører» i åtte timer etter at kjeden
var død — bruk en sentinel-fil som bare jobben selv skriver; (2) verten
blåskjermet 21.09 kl. 21:20 (bugcheck 0xA, `winhvr.sys`, fjerde 0xA på en måned
— se minnet `project_gx1_host_bsod_kills_long_runs_20260921`), så segment-resume
er BSOD-forsikring, ikke bare varmeguard.

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
