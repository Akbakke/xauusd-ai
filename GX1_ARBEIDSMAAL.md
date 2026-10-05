# GX1 arbeidsmål - oppdatert 05.10.2026

## Sluttmål

Bygg og kvalifiser en fullstendig automatisk XAUUSD-bot med én kausal
før-TEST-M1-kilde, M5 Entry, M1 Exit og lukket M15/H1/H4/D1-kontekst.
Bevar 254 features, åtte familier, ingen fast tapsgrense og ingen maksimal
holdetid.

## Bevis som kreves

1. Selektiv læring som slår relevante konstanter og samme-risiko-baselines.
2. Senere kronologisk generalisering uten TEST.
3. Positiv kostnadsjustert økonomi med alle valgte og åpne posisjoner.
4. Funksjonell train/serve-paritet for den fryste kandidaten.
5. Offline ordreintensjon, idempotent restart, brokeravstemming og recovery.
6. Fryst modellvalg før forseglet TEST.
7. Egen senere autorisasjon før live/paper, brokerordre eller spending.

## Nåstatus

Komplett M1/M5-grunnlag, v38-features, preprocessing, normalisering,
kostnadspolicy, økonomisk indeks og faktisk featurekilde er kvalifisert.
Makrokjernen DFII10/DTWEXBGS/T10YIE var inkonklusiv og er ikke promotert.
Full B med seks kilder består som separat blokkert mål.

Neste port er én workload-matchet samplerbenchmark. Se
docs/RESTART_POINT_20261005.md og CURRENT_RESTART_POINT.json.
Ingen edge eller lønnsomhet er bevist.
