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
