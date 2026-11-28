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
native_all_tf_provider_hlc_v3_block_order_recovery_20261128_001.
Order-feltet native_all_tf_provider_hlc_v3_direction_block_order_20261128_001
er lukket etter ekstern stopp; originalplanen er forbrukt.
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

### Nyeste order-fit — avbrutt, ekte C1 bevart

Den order-only-kilden er kvalifisert med 64 guard-/kildebrokontroller og
23 ulike leaf-kontroller. Metadata-preflight og separat SOURCE-backup består.
Kjøringen brukte fryst kilde e63c927316d251cf96d07b6bbc863323f8d20d89.
Fersk modell og full RNG matchet original INITIAL, og faktisk fasit-SHA ble
kontrollert før første oppdatering. Ingen gamle vekter ble lastet, og ingen
nye INITIAL-forwards ble gjort.

Guard stoppet kl. 02:12:57 UTC på host_maintenance_active, 239,96 s etter claim.
RESULT/TERMINAL er FAILED_CONSUMED med alle seks bookends PASS. MoUsoCoreWorker
PID 4676, parent 5764, ble opprettet 02:12:56.3024120 UTC; journalen viser tilstede-
værelse 02:12:57.6730684. Setup/OOBE/CBS/WU/reboot-in-progress var false/0.
Aktiv installasjon er UNKNOWN. Guardloggen mangler nonce/monotonic; tids-/
reason-/flaggkorrelasjon er ikke bevist eksakt noncebinding. Én signert query
og journalen viste idle 02:27:58 UTC; dette er observert idle, ikke eksakt exit
eller garanti for neste kjøreperiode. Ingen host-/vaktendring.

CURSOR001.json/.pt og CURSOR001_ORDER.json er varige og hashbekreftet.
C1 beviser én varig oppdatering. Totalt faktisk utført er UNKNOWN innen 1..216:
PASS1 publiseres synkront etter update 216 og før 217, men PASS1/PASS2/FINAL
mangler. Det finnes ingen ny order-kandidat eller læringsdom.
Første originalbatch var 116; neste logiske indeks 1 bruker originalbatch 131,
96 rader, samme bundne kilde-/issuehash. ScheduleSHA er
dba16db2b9c04c1fd988a79d3fd713b9719bbfc3514b76b38174b92ccfef19de.
C1-checkpointSHA er
d15de306b8a231b8665f1422ba7c3f7aae211524ffe9ba0970f0ca28f1437e05.
Eksakte øvrige original-/policy-/proofbindinger ligger i det lukkede feltet.

### Neste konkrete arbeid — eksakt videreføring

Design er bundet i V3_BLOCK_ORDER_EXACT_RECOVERY_DESIGN_20261128_001.json.
Adapteren i eksisterende eiere er kildekvalifisert: 73 guard-/kildebrotester
og 18 syntetiske rollekontroller består. Faktisk C1-/protokollmetadata og 78
kildefiler er validert uten NN eller tensor-dekoding. Fysisk paritet gjenstår.
Felles protokoll er bundet i policy. Neste er de tre eksakte rollene:

1. REFERENCE: fersk samme factory/full RNG, opprinnelige batcher 116→131.
   Alle ti C1-bevis må matche ekte order-C1 før C2 godtas. Ny produsent er ekte;
   full fil-SHA skal ikke tvinges lik mellom ulike produsenter.
2. COLD: ny prosess laster ekte order-C1, autentiserer faktisk full produsent/
   runtime/scores/proof, og avviser feil ordreschedule før parameterkopiering.
   Utfør kun batch 131; alle ti C2-bevis må matche referansen.
3. RECOVERY: last ekte order-C1 på nytt, aldri tekniske vekter. Fortsett opprinnelig
   globalindeks 1..431 med 431 nye updates/596 forwards til samme sluttpunkt 432.
   Bevar ordrebindinger, original INITIAL, kanonisk PASS1/PASS2 og tre opprinnelige
   diagnostikkpaneler. Bare global PASS2 er kandidat; ny FINAL 432 er eneste
   gjenstående opprinnelige checkpointposisjon.

Nye order-C1/C2-kalde bevis er foreløpig IKKE MÅLT. Gamle kronologiske kaldbevis
er bare uendret numerisk mekanikk og erstatter ikke denne nye pariteten.
C1 kan legitimt ha null backbone-momenter. Det opprinnelige ukjente fysiske
arbeidet og tekniske replay-oppdateringer rapporteres separat fra logisk 432.

Én felles fryst kilde/protokoll gjennom de tre rollene; administrativ lane-
admisjon bare mellom faktiske terminaler. Hver rolle har ny eksakt engangsplan,
aktuell runtime/preflight/backup og egen uforanderlig policy-snapshot. Alle
gates er nå stengt. Ingen relaunch av order-planen, automatisk retry/fitserie,
nye epoker, tap, ruting, features, prior, normalisering, utvalg eller vaktsvekkelse.
Alle fem TF og genuine familier beholdes. TEST og øvrige eksklusjoner består.

## Bevarte tekniske grenser

Diagnostikkens 51 guard-/kildebrotester og 20 ulike leaf-kontroller består;
faktisk FINAL-metadata og 78 kildefiler er validert. Gamle numeriske eiere og
beståtte kontroller gjenbrukes. Første metadata-preflight stoppet på null
telemetrisamples før child/CLAIM; aktuell signert query og andre metadata-
inspeksjon bestod. Timeout/vakter ble ikke endret. Hver NN-plan ble kjørt én gang.
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
- V3_BLOCK_ORDER_EXACT_RECOVERY_DESIGN_20261128_001.json:
  neste eksakte C1/C2/recovery-scope; ingen ny fit-admisjon.
- V3_DIRECTION_BLOCK_ORDER_STOP_REVIEW_20261128_001.json og
  MAINTENANCE_DIAGNOSIS med samme suffiks: ekstern stopp og bevart order-C1.
- V3_DIRECTION_BLOCK_ORDER_DESIGN_20261128_001.json:
  opprinnelig uendret order-only-hypotese og bevarte invariants.
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
