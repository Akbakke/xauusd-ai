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

NEXT_RUN_POLICY.json er eneste arbeidsstatus. Aktuelt forberedelsesfelt er
native_all_tf_provider_hlc_v3_direction_block_order_20261128_001.
Siste diagnostikkfelt native_all_tf_provider_hlc_v3_direction_gradient_20261128_001
er CLOSED_CONSUMED_COMPLETE. Appmålet mot dokumenterbar retningsedge med alle
genuine indikatorer og deres samspill er aktivt og uoppfylt.

Alle seks trenings-/evalueringsflagg er false og var false også under den
særskilt innrømmede oppdateringsfrie diagnostikken. Alle tidligere fit-,
recovery- og diagnostikkplaner er forbrukt/lukket. Ingen modelljobb kjører.
En historisk godkjenning åpner aldri en ny plan; kilde/designforberedelse
nedenfor er ikke fit-admisjon.

### Siste låste læringsutfall

V3 FIT004-trajektorien ble fullført gjennom RECOVERY002 til global 432:
431 nye optimizeroppdateringer, 596 forwards, 805,35 s, alle seks bookends PASS.
Original C1 var initializer; tekniske vekter ble aldri brukt. REFERENCE002
fersk C1 og separat COLD002-C2 matcher original/referanse på alle ti bevis.
Originalt fysisk arbeid før FIT004-stoppet forblir UNKNOWN innen 1..216.

Bare forhåndslåste PASS2 er kandidat. Samlet TRAIN-verdict er FAIL og
learning_qualified=false. Aktivitet slår prior på alle fem TF, men det er
ikke alene retningslæring. Side NLL/Brier slår COHORT-prioren under begge
vektinger bare på M5/M15. H1/H4/D1 feiler.

| TF | Side-AUC | Side Brier minus prior, radvektet |
|---|---:|---:|
| M5 |0,521052|-0,000265981|
| M15|0,537790|-0,000406239|
| H1 |0,522139|+0,000358074|
| H4 |0,500753|+0,002628812|
| D1 |0,516545|+0,001826896|

H1/H4/D1 har gjennomsnittlig UP-feil +2,60/+5,02/+4,49 prosentpoeng.
M15/H1/H4/D1 har bare UP ved binær argmax. H4/D1 feiler proper-score-
årskontrollen2021–2024 og består2025; M5 feiler2021.
Lagret algebraisk Brier-dekomponering viser nivåbidrag for H1/D1 og svak
sentrert rest/rangering for H4. Ingen kalibrator ble fittet eller innført.

### Faktisk FINAL-diagnostikk

Den nye diagnostikken fullførte på 97,84 s med 3 eval-forwards, 30 autograd-kall
og 0 optimizersteg. Alle seks bookends og alle ti originale FINAL 432-bevis
består; modell, AdamW, full RNG, treningsmodus og tomme parametergradienter er
bevart. Fryst kjøringskilde var 15d258835. Ingen checkpoint/logit/refit-
skriving. Egne heads og delt backbone får ikke-null sidegradient i alle
15 støttede TF-paneler. Ingen fremmede readout-heads fikk sidegradient.

På samme tre låste paneler [0, 107, 214] er normforholdet mellom summert
aktivitet og summert side 6,753/20,495/8,882, med positive cosinusverdier
0,18590/0,06087/0,23657. Samlet rågradient støtter side på hele backbone i
14 av 15 TF-paneler; bare D1/panel 107 er negativ. Dette er ingen universell
aktivitetskade eller frakobling. Alle åtte familier og shared-mix er med;
D1/makroparametre ligger faktisk i session_regime_encoder.

Lokal D1-biasfeil er -17,42/-16,11/+3,48 prosentpoeng, mens pooledTRAIN er
+4,49. Derfor følger ingen global biasjustering av de tre panelene.
Gradientene er lokale, uklippede data-gradienter; de er ikke historisk
kausalbevis, AdamW-oppdateringer, optimale taskvekter eller indikatornytte.

### Neste konkrete arbeid

Én ny kontroll skal isolere batchrekkefølge før endret modell/tap/ruting:
samme 216 hele B96-batcher, siste 87-radersbatch, 20727 TRAIN-rader, to pass og 432
updates. Privat PCG64 med eksisterende recipe-seed 20260911 låser to
permutasjoner uten å flytte modellens globale/dropout RNG. Ingen endring i
rader, rekkefølge innen batch/vindu, features, mål, masker, normalisering, prior,
fullt koblet None, optimizer, klipp eller evalrekkefølge.

Fersk samme factory er initializer; aldri gamle C1/FINAL/tekniske vekter.
Gjenbruk faktisk INITIAL og gamle kalde bevis bare etter eksakt identitets-/
funksjonskontroll og navngitt kildebro; kvalifiser ny schedule/cursor-
binding og negative restorekontroller. Endret ONLINE-funksjon krever
fersk INITIAL og ligger utenfor denne order-only-avgrensningen.
Sammenlign eneste nye låste PASS2 med opprinnelig kronologisk PASS2 og samme
INITIAL/COHORT-prior på alle fem TF/vektinger/år/blokker. Samme absolutte
kriterier gjelder; relativ forbedring av en FAIL er ingen ny kvalifisering.

Order-only-ruten er kildekvalifisert med 64 guard-/kildebrokontroller og
23 ulike leaf-kontroller. Faktisk INITIAL-score, logits, input-/normaliserings-
identitet og 78 kildeeiere er validert uten payloaddekoding eller NN.
Privat RNG, hele batcher, ny kursørbinding og avvisning før parameterkopiering
består. Plan, aktuell runtime, metadata-preflight og separat backup gjenstår
før noen fit-admisjon. Ny ordrespesifikk kald paritet er ikke målt eller påstått.

Dette er en målrettet test av rekkefølge-/suffixhypotesen, ikke en påvist
årsak eller lovet H4-forbedring. Ny kildekvalifisering, eksakt engangsplan,
backup og aktuell guardadmisjon kreves før NN/fit. Ingen automatisk seed-,
epoke-, taps- eller modellserie.

## Bevarte tekniske grenser

Diagnostikkens 51 guard-/kildebrotester og 20 ulike leaf-kontroller består;
faktisk FINAL-metadata og 78 kildefiler er validert. Gamle numeriske eiere og
beståtte kontroller gjenbrukes. Første metadata-preflight stoppet på null
telemetrisamples før child/CLAIM; aktuell signert query og andre metadata-
inspeksjon bestod. Timeout/vakter ble ikke endret. NN-planen ble kjørt én gang.
Originale MoUso-stopp, logger, checkpoints og backuper beholdes; ingen
prosesskill, reboot, vedlikeholdssperre eller svekket vakt ble brukt.

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
- Denne dokumentforkortingen endrer ingen kode, policy, data eller vakter.

## Bevis som åpnes ved konkret behov

E betyr /home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001.
Policy binder eksakte SHA og originale terminaler/snapshots.

- V3_DIRECTION_GRADIENT_ACCEPTANCE_20261128_001.json:
  faktisk 3/30/0, tellinger, proof10 og seks bookends.
- V3_DIRECTION_GRADIENT_REVIEW_20261128_001.json:
  alle tre paneler, per-TF readout-bias/vekter og Gram-avledede familiesummer.
- V3_DIRECTION_BLOCK_ORDER_DESIGN_20261128_001.json:
  neste avgrensede hypotese, bevarte invariants og stengt NN/fit-admisjon.
- V3_DIRECTION_GRADIENT_SOURCE_QUALIFICATION_20261128_001.json:
  fokuserte tester, navngitte bytekildebroer og faktisk metadata uten NN.
- V3_DIRECTION_GRADIENT_SOURCE/FINAL_BACKUP_20261128_001.json:
  separat Mac-disk/fsync/readback,FINAL bare når kvittert.
- V3_FIT004_EXACT_RECOVERY_TRAIN_REVIEW_20261128_002.json og
  RECOVERY_ACCEPTANCE med samme 002-suffiks: opprinnelig låst PASS2 og FINAL 432.

Aktuell diagnostikkroot:
 /home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_GRADIENT_20261128_001
Originale002roots har prefiks
NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002_ og suffiks
reference/cold/recovery. Original FIT004 har egen uendret root.
Originalkvitteringer overstyrer denne prosaen.
