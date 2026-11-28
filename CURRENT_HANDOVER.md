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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Fullført bølge eies av
native_all_tf_provider_hlc_v3_block_order_recovery_20261128_001.
Neste kildeforberedelse eies av
native_all_tf_provider_hlc_v3_direction_issue_mix_20261128_001.
Alle tre nye engangsroller er COMPLETE og lukket/forbrukt med remaining=0;
alle seks trenings-/evalueringsflagg er false, active_lane=null og ingen
modelljobb kjører. Appmålet mot dokumenterbar retningsedge med alle genuine
indikatorer og deres samspill er aktivt og uoppfylt.

### Fullført order-kontroll og faktisk kald paritet

Felles fryst kilde var 3a36f58ecc2d81e7b5e567cda294d2d2ba51e51c, med samme
protokoll gjennom REFERENCE, COLD og RECOVERY. Nye produsenter er ærlig bundet;
full fil-SHA tvinges ikke lik mellom forskjellige produsenter.
REFERENCE gjenskapte batch116→131 fra fersk factory/full RNG: alle ti C1-bevis
matcher ekte original-C1. COLD lastet ekte original-C1 i ny prosess, utførte
bare batch131 og matchet referansens C2 på alle ti bevis. Tekniske2+1 updates
er ikke læringskandidater og deres vekter ble ikke brukt til videreføring.

RECOVERY lastet ekte original-C1 på nytt, aldri teknisk C2. Den fullførte
431 nye optimizeroppdateringer/596 forwards til opprinnelig logisk432 på
803,87 s, exit0. Alle seks bookends består i hver av de tre rollene.
Original INITIAL ble bevart: fersk factory/full RNG og faktisk truth-SHA var
fysisk kontrollert før original order-fit. Ingen gammel vektinitializer,
ny prior, normalisering, utvalgsendring eller nye INITIAL-forwards.

Studien beholder20 727 TRAIN-issues fra1377 kildeklokkedøgn2021–2025,
alle åtte familier, D1/makro og M5/M15/H1/H4/D1, fullt koblet None-policy.
B96/E256, to pass à216 batcher, seed20260911. Bare hele batchers rekkefølge
ble endret; genuine vinduer, mål, masker og kanonisk scoring er uendret.
Alle faktiske checkpoints/kvitteringer, opprinnelige tre sluttpaneler og
separate diskbackuper beholdes. Eksakte referanser/SHA ligger i policy.

Original order-plan er fortsatt FAILED_CONSUMED etter host_maintenance_active
02:12:57UTC. Dens C1 er varig; opprinnelig fysisk arbeid før stoppet er
UNKNOWN innen1..216, separat fra ny431-recovery og logisk432. Ingen relaunch,
kill, reboot, host-/vaktendring eller vedlikeholdssperre ble brukt.

### Låst PASS2: TRAIN FAIL, blandet forbedring

Bare forhåndslåst global PASS2 er kandidat; INITIAL/PASS1 er diagnostikk.
Kronologisk kontroll slo COHORT-prioren i side NLL/Brier på2/5TF;
whole-batch-stokkingen gjør det på4/5 under både rad- og målbarvekting.
M15 feiler fortsatt. Samlet proper-score og tidsblokkstabilitet er FAIL,
learning_qualified=false. Aktivitet består alle fem, men er ikke alene
retningslæring. Ingen PASS1-, TF-, seed- eller kriterieutvelgelse etter resultat.

| TF | Side-AUC, stokket | Side Brier minus prior, radvektet | År-PASS/5 |
|---|---:|---:|---:|
| M5 |0,516900|-0,000110352|4|
| M15|0,526010|+0,000089618|1|
| H1 |0,527229|-0,000279881|3|
| H4 |0,530418|-0,000221191|5|
| D1 |0,537875|-0,000284807|3|

H1/H4/D1 forbedres mot kronologisk kontroll; M5/M15 forverres. Totalt16/25
årsceller består mot14/25 før. M5 feiler2024, M15 feiler2021–2024,
H1 feiler2022–2023 og D1 feiler2024–2025. Pooled score skjuler dermed fortsatt
regimefeil. Høyere fire TF har fortsatt100%UP ved binær side-argmax; M5 gir
239/12739DOWN. Dette er ikke full treklassehandling eller påvist tosidig
handelsvalg. Alle8familier og5TF beholdes;4/5er ingen samlet suksess.

### Lagret CPU-attribusjon, ingen ny prediksjon

To CPU-auditer brukte eksisterende FINAL-parametre/outputs og autentisert
fasit, uten modell-/optimizerkonstruksjon, NN, autograd, optimizersteg,
featurecache eller ny fit. Alle input-/kilde-/policy-/HEAD-bookends og alle
checkpointproofs består. Den første lagrede analysens valgfrie momenter ble
UNKNOWN fordi tuple-betas var krevd; produksjon bruker direkte JSON-list.
Originalrapporten beholdes. Et smalt tillegg autentiserte alle10 bias-slots
ved432 mot faktiske direkte konstruktørargumenter, uten å kjøre analysen om.

M15s gjennomsnittlige kjente side-logit økte0,059446 mellom kontrollene.
Direkte native biasendring er bare0,000757; ca98,7% av additiv endring er
lagret logit-minus-bias-rest, som omfatter representasjon/projeksjon og
F32-avrunding. Den isolerer ikke disse komponentene eller en kausal feil.
Fullkohortens analytiske biasderivert bruker103610 kjente aktivitetsceller
som korrekt tapsnevner. Historiske, globalt klippede minibatchmomenter er
et annet objekt; fortegnene gir ingen optimal oppdatering eller årsaksdom.
Ingen enkel biaspatch, kalibrator, rutingendring eller ekstra epokeserie
følger av dette. Teknisk paritet og TRAIN-resultat er ikke senere edge.

### Neste arbeid — kun kildeforberedelse

V3_DIRECTION_ISSUE_MIX_DESIGN_20261128_001.json låser én ny hypotese:
globalt blandet medlemskap i TRAIN-minibatcher, samme20727issues/to pass/
432updates og samme kardinalitet per logisk steg som whole-batch-kontrollen.
87-raderssteget flyttes ikke. To private PCG64-permutasjoner er kildeklokkebaserte
og hashbundet før fit; medlemsindeksene sorteres innen hver batch for å bevare
eksisterende packet-API. Modellruten har ingen carry mellom issues; hver rad
får sine egne genuine kausale vinduer. Ingen generisk inputeier må endres.
Dette er en kontroll av batchsammensetning, ingen bevist forbedring eller årsak.

Neste er minste adapter i eksisterende trenings-/checkpoint-/guard-eiere,
fokuserte negative kontraktkontroller og faktisk fersk C1/C2-kald paritet.
INITIAL gjenbrukes bare etter fersk factory/fullRNG/truth-identitet; tekniske
eller gamle FINAL-vekter er aldri initializer. Samme fem TF, åtte familier,
D1/makro, mål, masker, tapsformel, optimizer, klipping og kanoniske score beholdes.
Bare låst PASS2 vurderes; alle5-/årskriterier og begge vektinger gjelder uendret.
Ny engangsplan, aktuell preflight/admisjon, backup og fryst kilde kreves.
Alle seks globale flagg er false: design åpner ingen kjøring, retry,
kalibrator, ekstra epokeserie, TEST, VAL, Control eller handel.

Kildekvalifisering før den fullførte bølgen bestod73 guard-/kildebrotester og
18 syntetiske leaf-kontroller, syntaks og native kontrakter. Original numerikk
og vakter er uendret. Ny metadata-/handoverkontroll avslutter denne bølgen;
uendrede fullsuiter skal ikke gjentas.

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
Policy binder eksakte SHA, originalkvitteringer og egne policy-snapshots.

- V3_DIRECTION_ISSUE_MIX_DESIGN_20261128_001.json: neste énvariabelhypotese,
  låste kildeposisjonspermutasjoner og stengt kjøringsport.
- V3_BLOCK_ORDER_EXACT_RECOVERY_TRAIN_REVIEW_20261128_001.json:
  låst PASS2, alle fem TF/begge vektinger, år og ærlige ubeviste grenser.
- V3_ORDER_READOUT_ATTRIBUTION_20261128_001.json og
  V3_ORDER_READOUT_MOMENTS_20261128_001.json: bevart CPU-attribusjon og smalt tillegg.
- V3_BLOCK_ORDER_EXACT_RECOVERY_REFERENCE/COLD/RECOVERY_ACCEPTANCE_20261128_001.json:
  faktiske roller, full proof10 og seks bookends; åpne konkrete filnavn fra policy.
- V3_BLOCK_ORDER_EXACT_RECOVERY_SOURCE/QUALIFIED/FINAL_BACKUP_20261128_001.json:
  SOURCE/QUALIFIED/FINAL er fysisk sikret på separat Mac-disk med fsync og
  uavhengig readback. FINAL har178 filer; eksakt arkiv-/manifesthash i policy.
- V3_DIRECTION_BLOCK_ORDER_STOP_REVIEW_20261128_001.json og
  MAINTENANCE_DIAGNOSIS med samme suffiks: original ekstern stopp og bevart C1.

Aktuelle roots har prefiks
/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_BLOCK_ORDER_RECOVERY_20261128_001_
og suffiks reference/cold/recovery. Den kronologiske kontrollen ligger i
NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002_recovery.
Originale kvitteringer overstyrer prosa.
