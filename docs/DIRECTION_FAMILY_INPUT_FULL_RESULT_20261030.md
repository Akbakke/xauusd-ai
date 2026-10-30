# Full familieinputkontroll 30.10.2026

Inputkontrollen er fullført på 47 602 TRAIN- og 17 288 senere Q1-foreldre, alle 14 inputflater og 4 057 batcher. Dette måler faktisk inputdekning og intervensjonsmekanikk, ikke lært indikatorverdi.

## Målt på hele det avgrensede datasettet

- GC-D1 aktive verdiceller: 506 745 425 / 2 219 205 240 observerte TRAIN-eksponeringer (22,8345 %), mot 485 795 092 / 805 966 560 senere (60,2748 %). Gjentatte historikkceller er ikke uavhengige observasjoner.
- gc_d1_observed_volume har 0–168 observerte historikkdager under trening, gjennomsnitt 82,8875, mot 169–230 senere, gjennomsnitt 199,3889. Vinduet er 252 dager. Dekningsantallene overlapper ikke; årsakseffekt på prediksjoner er ubevist.
- Ingen aktive verdifelt mangler helt i full TRAIN. Pilotens ti helt manglende TRAIN-felt var et utvalgsfunn.
- Senere nåkontekst mangler 17 aktive verdifelt helt: ti strenge H1-felt for signert handelsflyt og sju D1-felt for bearish proxy order-block. De ti H1-feltene er observerte på bare 36 av 47 602 TRAIN-foreldre og ingen av 17 288 senere foreldre. Delvis kjent flyt finnes i separate observerte felt.
- Kilden krever komplett time, samme kontrakt og gyldig signert flyt i hvert minutt for de strenge H1-feltene. Dette forklarer maskens krav; den faktiske årsaken til hver manglende time er ikke målt. Manglende verdier fylles ikke med gjettet retning.
- Den godkjente kompaktprofilen parkerer fortsatt 71/253 XAU-historikkfelt per tidsramme og 16/2542 kontekstfelt. Parkerte indikatorers prediksjonsverdi er ikke vurdert.

## Konsistens og kjørekostnad

Original terminal og outer er begge 0. Resultatets SHA-256 er 7361baad30ea2f0cc21699952b6336a49e701a6b64c86d1401d2828bdfc83c64. Uavhengig gjennomgang fant null avvik i 23 346 koordinatkontroller og verifiserte kilde-/metadatareferanser. Dette er gjennomgang av lagrede aggregater, ikke ny rådatareplay.

Kilde, policy og metadata ble kontrollert før og etter. Råpayloadenes filidentitet ble kontrollert ved bookends etter eksisterende providerkvalifisering; dette er ikke en ny full råfilhash ved begge ender. Kun tre akse-arrays ble dekodet, null modell-/optimizerkall, null nektede filåpninger og ingen TEST.

De 20 faktiske intervensjonskontrollene tok 5,1531 sekunder mot pilotens 30,8630. Baseline endret ingen verdier. Prislysarmen endret fortsatt null i de første 16 TRAIN-radene, men har støtte i bredere TRAIN og endrer senere vitner; dette beviser ikke frakobling.

Hele jobben tok 10 556,85 sekunder (omtrent 2 t 56 min). Cgroup topp/minnegrense var 4 GiB; max-hendelser 3107, OOM og OOM-kill 0. Swap-topp var 512 MiB. Grensene ble beholdt.

Original RESULT.limits har en foreldet generell setning om pilotens TRAIN-range. Denne setningen gjelder pilotmodus; denne kvitteringens mode=full, full_population_measured=true og eksakte radantall viser full avgrenset TRAIN. Originalkvitteringen er bevart uendret.

## Gjenstående arbeid

Fersk privat native kvalifisering av hver modellvariant, deretter ti serielle matchede ettpass-fit og full TRAIN/Q1-skåring med parrede usikkerhetsintervaller. Ingen familie er hittil dokumentert nyttig eller unyttig av denne kontrollen. Q1 er gjenbrukt utvikling, og handelsverdi er ikke målt.

Originale bevis ligger under /home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/INPUT_FULL_001: RESULT.json, TERMINAL.json, OUTER.json og INDEPENDENT_AGGREGATE_REVIEW_001.json. Eksakte bindinger og neste tillatte steg eies av NEXT_RUN_POLICY.json.
