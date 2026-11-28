# CURRENT_HANDOVER — GX1

## Autoritet og arbeidssted

- Arbeid bare i `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
- Les `GX1_RULES.md` helt; uendret SHA256
  `2c7f0fc9e564d5986db59157dee6cc947bdfd3260f742bacc464853a28683a58`.
- `NEXT_RUN_POLICY.json` er eneste arbeidsstatus. Originale terminaler,
  prosessobservasjon og hashbundne kvitteringer overstyrer prosa.
- Overtakelse: branch/log, `bash scripts/gx1_handover.sh --check`, dette
  dokumentet og aktuelle policyfelt. Åpne eldre bevis bare ved konkret behov.
- Root eier policy, launch og samlet kontroll; andre agenter har avgrenset eierskap.
- Historikk ligger i Git og originalkvitteringene, ikke i denne overleveringen.
  Siste fulle historikkversjon: `c6a8c6672:CURRENT_HANDOVER.md`.

## Gjeldende status og neste steg

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Aktuelt felt er
native_all_tf_provider_hlc_v3_train_eval_gap_20261128_001. Den originale
read-only NN-planen er faktisk FAILED_CONSUMED etter alle405forwards; en
separat CPU-engangsplan har fullført sammendraget fra bevarte JSON-skår.
Begge porter er stengt/forbrukt med remaining=0 og relaunch=false.
Alle seks globale trenings-/evalueringsflagg er false, active_lane=null.
Neste felt native_v3_frozen_batch_residual_audit_20261128_001 er bare design/
kildeklargjøring, execution=false og remaining=0.
Ingen ny fit eller servingendring er innrømmet. Edge-målet er aktivt/uoppfylt.

Neste konkrete undersøkelse er M5s gjenværende middel-/kalibreringsdrift:
V3_FROZEN_BATCH_RESIDUAL_DESIGN_20261128_001.json låser kontroll av faktisk
tapsnormalisering og batchplan mot lagrede logits/fasit
før en endring i modell, tap eller optimizer vurderes. Gjenbruk nåværende
bevis; ikke gjenta NN eller gamle fitplaner. En ny avgrenset kjøring krever
sin egen eksakte plan/kilde-/input-/hostadmisjon. Ingen dropout-, læringsrate-,
ruting-, taps-, epoke-, terskel- eller kalibratorendring følger av resultatet.

### Bevarte TRAIN-resultater og konkret mangel

Siste læringskandidat er forhåndslåst issue-miks PASS2 fra fryst kilde7471ddf48.
Alle20727TRAIN-issues/1377kildeklokkedøgn2021–2025 er med én gang per pass;
B96/E256, seed20260911, to pass à216steg. Private PCG64-permutasjoner endrer
bare batchmedlemskap; hvert issue beholder sine genuine kausale vinduer.
Alle åtte familier og D1/makro, M5/M15/H1/H4/D1 og fullt koblet None-policy
beholdes. Input, normalisering/prior, fasit/masker, tap, optimizer og vakter
er uendret. Alle432takes er faktisk bundet til source_rows/issue_ns/fasit.

REFERENCE og COLD fullførte2+1tekniske oppdateringer og fullproof10; disse er
ikke initializer/kandidater. Fersk FIT fullførte432updates/597forwards på812,48s,
0checkpointloads/restores/skips og alle seks bookends. Samme factory/full RNG
og faktisk truth-SHA matchet originalINITIAL; INITIAL-output ble gjenbrukt.
Originale stopp/ukjent fysisk arbeid1..216 i eldre armer beholdes som ukjent.

| TF | Låst PASS2 side-AUC | Brier minus COHORT-prior, rad | År-PASS rad/målbar |
|---|---:|---:|---:|
| M5 |0.519403|+0.000398422|1/5 · 1/5|
| M15 |0.518758|-0.000249777|4/5 · 4/5|
| H1 |0.524846|-0.000190210|3/5 · 3/5|
| H4 |0.524267|-0.000416408|5/5 · 4/5|
| D1 |0.614417|-0.008619807|5/5 · 5/5|

Side proper-score består4/5TF under begge vektinger; M5 feiler. Samlet proper-
score og tidsblokkstabilitet er FAIL, learning_qualified=false. Aktivitet og
pooled rank-guard består alle fem. Årskontrollen er18/25rad og17/25målbar,
mot14/25 og16/25 i tidligere kontroller. D1s13968kjente rader er875forskjellige
målbarer. TRAIN-år med whole-TRAIN-statistikk er ikke senere generalisering.
M5 har UP-middelfeil−2,4271pp og DOWN-argmax12721/12739; dette er betinget
side, ikke full treklassehandling eller dokumentert handelsvalg.

PASS1→PASS2 er diagnostikk: M5-AUC0,509672→0,519403 mens Brier forverres
0,000156777 og UP-middelfeil−1,758→−2,427pp. D1-AUC0,561885→0,614417.
Flere epoker er dermed ikke dokumentert som samlet fremgang på alle fem.

### Fullført readout-attribusjon og målt modusgap

Ny CPU-attribusjon autentiserte originale FINAL/fasit og alle10tilstandsbevis/
10sidevekt-/bias-momentslots ved432;0modeller/NN/updates/CUDA. M5s middel-logit
whole-batch→issue-miks endret−0,130168: native bias−0,000760 er0,584% og
logit-minus-bias-rest99,416%. Resten omfatter representasjon/projeksjon og
F32-avrunding samlet, uten isolert årsak. Direkte bias med uendret rest kan
flytte sannsynlighet høyst0,019pp mot observert middelendring−3,252pp.
En enkel biaspatch er ikke begrunnet av dette.

Design003 låste én full B256/no_grad EVAL-paritet og fire sekvensielle
TRAIN-modusdraws på samme frosne FINAL/alle20727issues/alle fem TF og år.
Original NN-plan-SHA:bd3f29a59c8d6c96ea9e20a2c21edde530087d7346c5ed74f1f9cefa2cf9a325.
Produsentkilde:d2a3cb055, SELF67a1267677b23e736029379bb30060b2eed9eb74fc7cf13c28a49c2be5c6ed4e.
Faktisk jobb fullførte405forwards,5score-/logitarkiver og alle tilstandsbevis
på366,68s, men feilet ved JSON-publisering av sammendraget:
ValueError:PROVIDER_HLC_DEV_UNSUPPORTED_JSON:float64. Originalstatusen
FAILED_CONSUMED og alle originale kvitteringer beholdes; planen relanseres aldri.
Ingen hardware-/hostvakt slo ut. EVAL-logitarkivet er byte-eksakt likt lagret
PASS2, SHAe8dd65bc8dc21c83f6d894634a3c16f3464d3426099fb9d3cf81adaa1177048d.

Vekter/buffere, optimizer, steps/scores og tomme gradienter var uendret før
opprydding; bare tilsiktet modulmodus/full RNG ble gjenopprettet. Alle10fulle
sluttbevis og alle seks bookends består. Hver draw startet fra forrige draws
slutt-RNG, uten reseed/reset. All fysisk støtte beholdes per draw.

Minste rettelse var kun native float-cast av descriptive_SE. Berørt
serialiseringsregresjon består. Separat CPU-plan-SHAdd36a379a624dfaeabfc7b9ebe6892c4badb0fbdb9efb3d2c7f68617e9b628a6
fullførte lagrede skår med0nye NN/prediksjoner/updates/NPZ-/checkpointdekodinger,
alle egne bookends og exit0. Nytt DRAW_METRIC_SUMMARY.json ligger bare i egen
NATIVE_V3_TRAIN_EVAL_GAP_AGGREGATION_COMPLETION_20261128_001-root; original
NN-root er urørt. Sammendragets SHA er
88f4e9f9e90d955e6cf21b18bf61551af8dc2232c5b30f64dbef23534e5469c1.

M5s UP-middel er48,456008% i EVAL og48,493718% som middel av fire TRAIN-skår:
modusgap+0,037710pp, fortsatt UP-middelfeil−2,389396pp (draw-SD0,001044pp).
Alle fire draws feiler fortsatt M5s proper-score; de andre fire TF består.
M5s gjennomsnittlige Brier er0,250301143, fortsatt+0,000379132 over prior.
AUC EVAL→TRAIN-skårmiddel: M5.519403→.519607, M15.518758→.518434,
H1.524846→.524725, H4.524267→.524178, D1.614417→.612520.

Dette er samlet frossen modusforskjell (dropout0,05 og eval-only gruppering av
like TF-lengder), ikke isolert dropoutårsak eller B96/backward-treningsbane.
Det lille målte gapet lukker ikke M5-mangelen, men utelukker heller ikke
dropouteffekter under læring. Mean/SD/SE gjelder fire separate skår, aldri
skår av gjennomsnittslogits/-sannsynlighet. Ingen beste draw eller presisjons-/
signifikanspåstand. Rad-AUC og rad-/målbar-NLL/Brier er rapportert; målbar-AUC
er NOT_IMPLEMENTED. Senere generalisering/økonomisk edge er fortsatt ubevist.

Kildekvalifisering:80guard-/kildebrokontroller og36forskjellige leaf-kontroller
består; bare berørte feil ble kjørt om. Én ekstra berørt testkjøring bekreftet
JSON-rettelsen etter faktisk stopp. Gjenbruk disse kvitteringene.
SOURCE-backup, original FINAL-backup og denne bølges FINAL-backup er fysisk
fsync/F_FULLFSYNC/readback-verifisert på Mac. Nytt FINAL-arkiv har168filer,
10 283 919bytes og SHA1e625cf04f211e4b301a61d2fc1eaf5b28db48acef38212404c7c51af40b4fed.
Begge målingsroots, alle skår/tilstandsbevis og CPU-sammendrag er inkludert.
Eksakte arkiv-/manifestbindinger ligger i policy. Originale FINAL-vekter
gjenbruker sin allerede verifiserte backup. Arkivet skiller produsentkilde
fra senere cast; avsluttende kvitteringsbinding/overlevering bevares i Git.
Ingen rådata eller vekter skal i Git.

## Kvalifisert datagrunnlag og funksjon

- Native OANDA practice/XAU_USD M5 MBA finnes fra 2009 til juni 2026.
  Alle kildeår/TF er skannet. Separat rettet provider-observasjonsreferanse
  har 542 753 høyere-TF-barer og 7 055 789 eksakte feltsammenligninger uten avvik.
- 1 410 restaurerte M5-observasjoner er utenfor vedtatt PAIR_WEEKCLOSED-domene.
  Referansen er ikke innført som modellinput eller ny featurehistorikk.
- 354 550 TRAIN-rader (2021–2025) har null tapte/ekstra rader fra lagret råkilde til
  canonical/CAPTURE. Alle 17 870 ukentlig-grid-fravær er allerede fravær i råsvar;
  ingen priser skal oppdiktes eller gap automatisk klassifiseres som stengt.
- V3 har 1 772 410 modne, kjente aktivitetsceller og 340 umodne haler
  (M5/M15/H1/H4/D1: 1/3/12/48/276). Tvetydig side blir aldri FLAT.
  Originale nominal-/kompletthets-/sourcegap-/kausale ATR-masker beholdes.
- Full fersk V3 INITIAL på 354 550 rader og egen C2-kald replay er kvalifisert.
  Låst COHORT-scoring skiller factoryreferanse, INITIAL og COHORT-prior;
  alle fem TF vurderes med rad-/målbarvekting, kalibrering og 25 tidsblokker.
- Åtte genuine spesialistfamilier og D1/makro samarbeider med M5 som hovedklokke
  og M15/H1/H4/D1-kontekst. M30 trenger egen kontrakt-/nyttevurdering.
- De 18 interne recurrence-/squeeze-prefixfunnene ble avkreftet før faktisk
  konsumhistorikk. Ingen ny inputbit eller gjentatt full replay er begrunnet.

## Fortsatt ubevist

13/3/4-statusen er ikke oppgradert av teknisk logging, dataparitet eller replay.
Historiske V1/V2 FAIL er ikke gjeldende V3-nevnere eller nye V3-læringsresultater.

- 03/07: historisk samme-feed markedskalender/quote-tilgjengelighet og relevant
  source-known/asof er uavklart. Fravær alene beviser ikke markedsstenging.
- 09: relevant publisering/revisjonsvintage for makro er fortsatt uavklart.
  44 av 61 felt er OANDA-prisderivater fra 11 instrumenter; øvrige 17 er
  COT4/Treasury9/VIX2/GLD2. Ingen aktuell OANDA44-lookahead er målt.
  May18 Treasury er kjent 2467 s før første faktiske input; Apr1 før faktisk
  input 2022-04-03T22:05Z er UNKNOWN. Deklarert 2022-04-03T04:00Z er et separat,
  sterkere klokkekrav; begge publiseringspåstander er ubevist.
  Ingen automatisk arkiv-/HTTP-retry.
- 11/12: provider-HLC-observasjon er ikke ubetinget markedskompletthet eller
  representativitet. Nominal klokke og observerte providerbarer er ulike bevis.
- 15/19: V3-læring og senere generalisering er ikke kvalifisert. TRAIN-år med
  whole-TRAIN-statistikk er ikke kronologisk OOF eller uavhengig senere evaluering.

Full kalender/vintage er ikke et universelt hinder for alle lokale tekniske
kontroller. En avgrenset fit trenger egen eksakt admission; læring og senere
generalisering må måles som utfall, ikke antas bevist før første fit.

## Varige kjøre- og oppryddingsgrenser

- Det varige M5/MTF/26-måneders-MBP-målet i `GX1_RULES.md` består.
  Målet er nyttige tosidige prognoser og senere LONG/SHORT med liten MAE før MFE;
  GC/orderbok er utsatt, ikke slettet. Metode: `docs/NATIVE_LEARNING.md`.
  Én tung CURRENT-jobb via `scripts/gx1_capped_run.sh`, lås og eksisterende
  hardware-/hostvakter; frosne kilder/policy under kjøring. Stabile lange
  kjøringer kontrolleres omtrent hvert 15.–30. minutt, ikke minuttvis.
- Ingen TEST-tilpasning, M1-retningsinput/fasit/normalisering/utvalg, GC,
  Entry/Exit, broker/live/paper/ordre/spending/promotion i denne bølgen.
  TEST er forseglet; juni 2026 er gjenbrukt utviklings-VAL.
- Ingen fast tapsgrense/maksimal holdetid eller fjerning av genuine familier/TF.
  Teknisk PASS, alltid FLAT/HOLD og TRAIN-score beviser ikke handelsnytte.
- Alle originale MAIN-/recovery-checkpoints, kvitteringer, inputs og backuper
  beholdes. Intet nytt slettevedtak foreligger.
  `full_project_retention_cleanup_20261120_001` er forbrukt med resterende 0.
  DATA/RUNS ryddes bare av `gx1.scripts.cleanup_gx1_evidence_v1` etter regel 9,
  med fersk dry-run, kanonisk prosjektlås, verifisert eierskap/referanseomfang
  og eksakt godkjent inventar.

## Bevis som åpnes ved konkret behov

E betyr /home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001.
Policy binder eksakte SHA, originale kvitteringer og egne snapshots.

- V3_FROZEN_BATCH_RESIDUAL_DESIGN_20261128_001.json: neste stengte CPU-design,
  alle432takes/fem TF og konservasjon, ingen ny NN/treningsadmisjon.
- V3_TRAIN_EVAL_GAP_MEASUREMENT_REVIEW_20261128_001.json: alle fem TF/25årsceller,
  EVAL mot fire draws, mean/SD/SE og autentiserte original-/CPU-kvitteringer.
- V3_TRAIN_EVAL_GAP_SERIALIZATION_FIX_20261128_001.json: eksakt cast, produsent-
  og rettet kildeidentitet samt berørt regresjonstest.
- V3_TRAIN_EVAL_GAP_ACCEPTANCE_20261128_001.json: original faktisk NN-feil.
- V3_TRAIN_EVAL_GAP_AGGREGATION_COMPLETION_PLAN_20261128_001.json og egen root
  RESULT/TERMINAL: separat fullført CPU-plan, ingen NN-relansering.
- V3_TRAIN_EVAL_GAP_SOURCE/FINAL_BACKUP_20261128_001.json og manifeste:
  separate fysiske Mac-backuper; FINAL inneholder også CPU-completion-root.
- V3_DIRECTION_ISSUE_MIX_TRAIN_REVIEW_20261128_001.json og
  V3_ISSUE_MIX_PASS_DIAGNOSTIC_20261128_001.json: låste kandidater og diagnostikk.
- V3_ISSUE_MIX_READOUT_ATTRIBUTION_20261128_001.json: konkret ny arm mot gamle
  JSON-referanser uten å kjøre gamle armer om.
- V3_DIRECTION_ISSUE_MIX_REFERENCE/COLD/FIT_ACCEPTANCE_20261128_001.json og
  SOURCE/REFERENCE/QUALIFIED/FINAL_BACKUP: tidligere fullførte rollekvitteringer.

Aktuell original NN-root:
/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_TRAIN_EVAL_GAP_20261128_001.
Separat CPU-completion-root:
/home/andre2/GX1_RUNS/NATIVE_V3_TRAIN_EVAL_GAP_AGGREGATION_COMPLETION_20261128_001.
Original issue-miks FINAL ligger i DIRECTION_ISSUE_MIX_20261128_001_fit-rooten
bundet i plan/policy. Originalkvitteringer overstyrer prosa.
