# Entry: retning først — 23.09.2026

## Gjeldende mål

Brukeren har bedt om å prioritere LONG = bullish og SHORT = bearish.
Tidligere spørsmål om én åpen posisjon eller nettoresultat per tidsenhet er
ikke en forutsetning for dette arbeidet. Ingen slik handelsregel innføres.

Mål fremtidig prisretning fra historikken, med identiske beslutningsklokker,
og skill dette fra lærerens forventning om senere Exit-fortjeneste.
Entry-Qs LONG/SHORT-indekser er korrekt bundet; ingen fortegnsbyttefeil ble funnet.
Den direkte main_direction/mtf_direction-loss-en er avviklet i dagens trener.
Eksisterende forecast-hode lærer faktisk fremtidig close-avkastning med L1.
Forecast er en auxiliary; Entry-Q velger fortsatt etter frossen Exit-policyverdi.

Aktiv kilde ved måling: GX1_ENGINE, audit/v9-premiere-20260905, 8e0edafc.
Original checkpoint844 og alle modell-/trenerfiler er uendret i denne runden.
TEST er forseglet. Ingen GPU-trening eller Exit-rollout ble kjørt.

## 1. Avgrenset retningsfit på frossen v10-representasjon

1024 jevnt fordelte TRAIN-rader og 512 senere juni-VAL-rader.
Én analytisk Ledoit–Wolf-ridge fra eksisterende 128 Entry-Q-hidden-verdier til
fortegnet på faktisk avkastning etter eksisterende 1/5/12/24 M5-steg.
Koeffisientene ble frosset før VAL. Ingen horisont eller terskel ble valgt.
Alle fremtidige labels ble kontrollert mot bundet close-historikk (0,001 Bps toleranse).
Originale vekter var bitlike etter målingen. Returkode 0; 190,42 sekunder.

| Nominell horisont | TRAIN balansert treff | Senere VAL balansert treff |
|---|---:|---:|
| 5 minutter | 62,55 % | 49,87 % |
| 25 minutter | 61,92 % | 50,26 % |
| 60 minutter | 63,96 % | 50,21 % |
| 120 minutter | 63,10 % | 51,87 % |

Ingen av fire forhåndsdefinerte porter bestod. Dagblokk-bootstrap med
korreksjon for fire horisonter inkluderte 50 % for samtlige.
Dette er overtilpasning uten dokumentert senere retningsgevinst.
Prøven er avsluttet; ikke tun, promoter eller relanser den.

## 2. Viktig avgrensning: endret funksjon med gamle vekter

v10-prøven bruker en endret rutingsfunksjon med checkpoint844s originale vekter.
Dens forecast-prognoser var bearish på alle 1024 TRAIN / 512 VAL-rader.
Dette beskriver ikke automatisk originalmodellens innlærte prognosefunksjon.

En paret kontroll på de samme 63 bevarte VAL-inputene viste:

| Funksjon med identiske originalvekter | 5 min LONG/SHORT | 25 min | 60 min | 120 min |
|---|---:|---:|---:|---:|
| Original v8 | 60/3 | 53/10 | 57/6 | 55/8 |
| Endret v10 | 0/63 | 0/63 | 0/63 | 0/63 |

Gjennomsnittlig absolutt prognoseendring: 2,89 / 9,24 / 21,76 / 43,68 Bps.
Original v8s balanserte treff var 48,49 / 53,21 / 51,85 / 58,94 %.
Utvalget er lite, gjenbrukt utviklings-VAL og gir ingen sikker rangering av
horisonter. Ingen horisont er valgt. Returkode 0; 13,83 sekunder.

Konklusjon: attribuer alltid retningsresultater til både modellfunksjon og
vekter. Det negative forecast-fortegnet i første probe kan ikke brukes som
bevis på at original v8 hadde samme kollaps. Eksisterende rutingsgradientfeil
og deres tekniske rettelser er fortsatt dokumentert; de er ikke lønnsomhetsbevis.

## 3. Retning direkte fra eksisterende input på større TRAIN-grunnlag

Eksisterende feature-rangering er kun TRAIN-diagnostikk: beste enkeltfelt hadde
absolutt Spearman 0,02715 mot 19-stegs executable LONG-minus-SHORT-margin.
Den dokumenterer ikke senere retningskvalitet.

Én ny analytisk baseline brukte alle 241 snapshotfelt og 71 kontekstfelt,
samt fire TRAIN-observerte session-kategorier. Ingen felt ble slettet fra modellen.
Middelverdi, skala, koeffisienter og regularisering er tilpasset bare de første
fire årene. Fire grensekryssende rader ble utelatt fra fit.

- Fit: 248 100 TRAIN-rader, fremtidige labels ferdige før 01.06.2025.
- Kronologisk intern kontroll: 65 295 senere TRAIN-rader, juni 2025–mai 2026.
- Samtlige 313 399 rader fikk labels kontrollert mot source-close.
  Største absolutt avvik: 0,00003044 Bps.
- Én fast Ledoit–Wolf-ridge til alle fire retningsmål, ingen parametersøk.
- Ingen juni-VAL eller TEST lest av denne kontrollen.
- Returkode 0; 39,07 sekunder; beskyttet CPU-audit med 4 GiB-grense.

| Horisont | Fit balansert treff | Senere TRAIN-kontroll | Justert intervall |
|---|---:|---:|---:|
| 5 min | 51,90 % | 51,06 % | 50,35–51,60 % |
| 25 min | 52,61 % | 50,55 % | 49,90–51,10 % |
| 60 min | 53,35 % | 50,36 % | 49,35–51,51 % |
| 120 min | 53,73 % | 50,03 % | 48,17–51,68 % |

2000 månedsblokk-resamplinger, seed 0, justert for fire horisonter.
Bare 5-minuttersmålet bestod denne interne diagnostiske porten.
Dette er et svakt signal, ikke dokumentert Entry-forbedring eller netto lønnsomhet.
Metoden bruker snapshots/kontekst; den erstatter ikke modellens tidssekvenser.
Denne fit-en og 128-hidden-prøven bruker forskjellig datamengde og fit-periode;
forskjellen isolerer derfor ikke effekten av representasjonen alene.

## Neste konkrete arbeid

1. Bruk den bevarte originalfunksjonen v8 med checkpoint844 som korrekt
   prognosebaseline på det allerede låste 1024 TRAIN / 512 VAL-utvalget.
   Gjenbruk labels og rad-ID-er. Ingen ny fit eller horisontseleksjon.
2. Avgrens deretter eventuell direkte retningssupervisjon i den eksisterende
   native treneren. Et retningsscore må ikke skrives inn som Bps i Entry-Q
   eller forecast-utdata. Bevar eksisterende features, tidsrammer og Exit-lærer.
3. Krev senere kronologisk retningsgevinst før større trening. Selektiv FLAT
   og samlet kostnadsjustert økonomi gjenstår etter retningsmålet.

Målet er aktivt og ikke oppnådd. Ingen jobb er aktiv ved denne overleveringen.
Totalt 192 native optimizersteg fra tidligere kontroller og tre analytiske
fits i hele reviewet (én tidligere verdi-fit, to retningsdiagnoser).
Ingen av disse nye retningskoeffisientene er promotert. PC er ikke restartet.

## Bevis

Rot: /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923

- ENTRY_DIRECTION_20260923/PLAN.json
  SHA 7187b6b7cc6b3c93ec16a0e8cf328b83990faed33658a5da4df7440c2085ad62
- ENTRY_DIRECTION_20260923/RESULT.json
  SHA 07971cf270a6c55e0ec112638ccc0fcdd1a6d74e5018a2d64d2987efce0156b2
- ENTRY_DIRECTION_20260923/ORIGINAL_FUNCTION_CONTROL.json
  SHA cd8b78a3d2a6ac2ab4c018573be6db2b7a858a1a06291cbeb10cd974cd572edc
- ENTRY_DIRECTION_RAW_TRAIN_20260923/PLAN.json
  SHA e595d7f29c20eaadfbd67d316e34eacdf6661932c20b29d5165b0cc80e870c6e
- ENTRY_DIRECTION_RAW_TRAIN_20260923/RESULT.json
  SHA fdf14fc99e64096e3715e7712048381f757644e8d8c3db5e44d51f5f65c26284

Skript, logger, frosne koeffisienter og caches er bevart i disse runtime-mappene.
Ingen historiske planer skal automatisk relanseres.


## Original v8-baseline og avgrenset native retningslæring

Den planlagte original-v8-baselinen er fullført med returkode 0 på 193,56 sekunder.
Samme 1024 TRAIN /512 VAL-rader og labels som v10-prøven er kontrollert bitlike.
Checkpoint844s råvekter er uendret. Ingen nye fits eller optimizersteg.

| Nominell horisont | Original v8 TRAIN balansert treff | Senere VAL | Justert VAL-intervall |
|---|---:|---:|---:|
| 5 minutter | 51,27 % | 49,55 % | 45,89–52,73 % |
| 25 minutter | 55,13 % | 53,56 % | 49,60–57,62 % |
| 60 minutter | 58,14 % | 51,45 % | 47,96–55,12 % |
| 120 minutter | 59,97 % | 52,20 % | 47,67–56,26 % |

Ingen horisont bestod porten. Original Entry-Q-spread valgte SHORT på alle
1024 TRAIN /512 VAL-rader, selv om forecast ga begge fortegn.
Dette begrunner direkte arbeid med retningslæring; flere gjentakelser av
samme Exit-lærer-fit er ikke neste tiltak.

Bevis: ENTRY_DIRECTION_ORIGINAL_V8_20260923/PLAN.json og RESULT.json.
Plan-SHA: 777be8e2becc606d69bd8275d71a8805214da05f9dff2ab4153063ca0116c794.

### Minste native utvidelse

Et valgfritt, recipe-bundet forecast_only_warmup-flagg gir én avsluttende
research-pass i den eksisterende canonical treneren. Det krever initialisert
FP32-smoke med originalcheckpoint og en SHA-bundet, frossen originalfunksjon.
Candidate, legacy-rute, ufullstendig binding og flaggavvik avvises.

- Bruk eksisterende forecast-hode, L1 i Bps og alle fire eksisterende horisonter.
  Ingen ny modellarkitektur, retningslogit, handelsterskel eller tapsvekt.
- Lær fra observerte fremtidige prisutfall. Ingen Exit-rollouts eller
  bootstrappede Exit-verdimål brukes i denne avgrensede læringspassen.
- Bevar alle eksisterende inputfelt og tidsrammer. Forecast og delte encodere
  mottar gradient; øvrige hoder, Entry-Q-ledd og task-vekter skal være bitlike.
- Mål både originalfunksjon, endret online-funksjon før fit, og etter fit på
  samme 512 TRAIN /512 VAL-rader. Disse er en låst delmengde av smoke-populasjonen.
- Utdata er en researchrapport, før/etter-arrays og bevarte råvekter.
  Ingen bundle, checkpointseleksjon eller promoteringsmyndighet produseres.
- Delte encoderendringer kan påvirke Exit og Entry-Q selv med uendrede hoder.
  Full økonomisk etterkontroll er derfor fortsatt påkrevd før bruk.

Teknisk kontroll: 20 eksisterende recipe-/launcher-kontroller bestod.
En separat syntetisk fire-stegs kjøring kontrollerte flaggbinding, avvisninger,
de tre evalueringsstadiene, endret encoder/forecast, uendrede beskyttede hoder,
like før/etter-rader og labels, og fravær av bundle/promotering. Dette er
implementeringsbevis; ingen av de fire syntetiske stegene er markedsbasert læring.

Neste ene forsøk er forhåndsavgrenset til 4096 TRAIN-rader, batch8,
én passering /512 optimizersteg, og de samme eksisterende recipe-hyperparametrene.
Kjør bare via eksisterende native launcher og gx1_capped_run, med alle vakter.
Ingen full epoch eller full VAL. Ingen horisont velges fra VAL.
En bedre forecast-MAE alene er ikke nok: vurder begge retningsrecalls, balansert
treff og usikkerhet mot både før-fit og originalfunksjonen. All økonomisk
Entry-seleksjon forblir uavklart til separat etterprøvbar måling foreligger.


### Første oppstart stoppet før trening; v8-referansen er rettet

ENTRY_DIRECTION_WARMUP4096_20260923T135523Z (source022a5d8a) avsluttet
14:03:05 UTC med returkode1 etter345,57 sekunder. Referanse-loaderen krevde
v9 og avviste korrekt original-v8 med ENTRY_FROZEN_TEACHER_SOURCE_VERSION_INVALID.
Traceback er før warmup-funksjonen og før første optimizersteg: faktisk0 nye
markedssteg, ingen baseline/ettervekter produsert. Planen er avsluttet.

Minste rettelse: et eksplisitt internt original_direction_reference-flagg
velger v8 bare for denne retningsreferansen. Standard v9-lærer er uendret.
Den virkelige helperen er nå kontrollert mot checkpoint844 og tre bevarte
VAL-input: forecast, Entry-Q og Entry-token er bitlike uavhengig lastet v8.
Normalisering/vekter og Torch/NumPy/Python-RNG er uendret; feil hash og feil
standardversjon avvises. Eksisterende v9-laster fungerer fortsatt.

Vakten avsluttet med child_status1 uten maskinvarebrudd; målte topper:
61 C kjerne,62 C minne,138,21 W og732 MiB GPU-minne.
Ny oppstart krever fersk recipe på rettet, committet kilde. Samme4096-raders
beregningsplan og forhåndslåste læringsport beholdes; dette er ingen ny variant.
Bevis: run/START_FAILURE_DIAGNOSIS.json,
DIRECTION_ORIGINAL_REFERENCE_LOADER_VERIFICATION.json og tilhørende logger.

### Fullført L1-forsøk og neste avgrensede retningssammenligning

ENTRY_DIRECTION_WARMUP4096_20260923T141919Z er fullført på7abae358:
4096TRAIN,512optimizersteg,returkode0. Ingen horisont bestod retningsporten.
På samme512VAL ble balansert treff48,79/50,65/50,39/51,11 prosent.
Lavere forecast-L1 var hovedsakelig korreksjon av skjevhet: bare én horisont
slo en konstant TRAIN-median på VAL, med0,00784Bps lavere MAE.
Frossen råfeature-readout ga50,07–53,17 prosent på sammeVAL, uten støttet fordel.
Snapshot/context-normaliseringen gjorde ingen av312 felt konstante.
Volatilitetsskifte er målt; årsaken til svak retning er ikke bevist.

Én eksplisitt recipe-bundet warmup_direction_bce-variant er nå implementert
i samme eksisterende native research-gren. Den bruker samme fireutgangshode
som retningslogits i dette forsøket, samme arkitektur, input, originale vekter
og optimizerinnstillinger. Det er en kontroll av læringsmålet, ikke en påstand
om at pris-L1 er en kodefeil. Standard L1-gren og normal trening er bevart.

BCE bruker fortegnet til de eksisterende faktisk observerte fremtidsreturene.
Eksakt nullretur gir ingen bullish/bearish-label, utelates fra tapet og telles.
Ingen klasserevekting, ny handelsterskel eller tids-/kostnadsregel.
Retningsscore lagres som direction_score, aldri forecast_bps; originalfunksjonens
Bps er bare retningsreferanse. BCE-vektene er research-logits, ikke Bps-prognoser,
kalibrerte sannsynligheter, handelsbundle eller promotering.

Teknisk kontroll:20 eksisterende recipe-/launcher-tester bestod; syntetisk
fire-stegs kontroll av både L1- og BCE-grenen bestod. Nullmasken har null gradient,
feil retning straffes korrekt, tomme/ikke-endelige/ulike targets avvises,
source-/flaggavvik avvises, beskyttede hoder er uendret, og outputenheter er tydelige.
Dette er implementeringsbevis, ikke markedsbasert læring.

Neste ene markedskontroll bruker samme4096TRAIN,batch8,512steg og512TRAIN/512VAL
som L1-armen. Før-baseline og originalreferanse må være like L1-armens cache.
Retningsporten krever lavere TRAIN-BCE, begge retninger og justerte parede VAL-
intervaller mot original, før og fullført L1-arm. Alle fire horisonter rapporteres.
Ingen full epoch/fullVAL, retuning eller større trening følger automatisk.
Kjør først fra committet ren kilde og fersk recipe via eksisterende vakter.

Runtimebevis under /home/andre2/GX1_RUNS/V12_EPOCH1_REVIEW_20260923:
DIRECTION_BCE_IMPLEMENTATION_VERIFICATION.json,
DIRECTION_WARMUP_L1_REGRESSION_VERIFICATION.json,
direction_bce_targeted_checks.log og DIRECTION_OBJECTIVE_COMPARISON_NEXT.json.
Ingen ny markedstrening er startet ved denne kildeoppdateringen.
