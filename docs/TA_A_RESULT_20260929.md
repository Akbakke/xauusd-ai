# Måling A — INKONKLUSIV for begge modeller

Den fryste primære 20-D1-målingen gir INKONKLUSIV for ren ridge og HGB.
Ingen modell oppfyller GO mot de avtalte risiko-LONG-, konstant- og trendreferansene.
Punktestimatene gir ingen praktisk begrunnelse for native trening. Usikkerheten
utelukker samtidig ikke små relevante effekter; dette er ikke et bevis mot all teknisk analyse.

## Populasjon og kildebinding

Resultat: /home/andre2/GX1_RUNS/TA_RESEARCH_20260929/MEASUREMENT_A_001/RESULT.json, SHA-256 7835eef9f8021ea63246973f2f3bd6179d0b299f8fa4ccccd357498d95cc0224.
Kildecommit: 0dc4a04b379e46b51dab513825d7b5f49216d080. Registreringshash: c299e53fcd39818d75d5e505838ba1196b4b1d217992201b772b43f28c2a5758.
Finansiering: direkte EFFR, 4 394 eksakt verifiserte virkedager. Kildebytes:
fa83cd8030327dea8529f6ef5bf2f1a0f8a4294bcfd14e8dbf460f9a1d784d19.
To FRED-timeouter og EFFR-skjemaavviket er bevart. Mottatt percentRate-felt
ble bundet før A; ingen ny datahenting eller utfallsstyrt valg.

15 kronologiske årsfolds per horisont, 2011–2025, 4 003 prognoser og
4 002 sammenhengende porteføljeintervaller. Første fill 2011-01-02 20:50 UTC;
siste fill 2025-12-01 23:00 UTC. De siste 20 observerte D1-binene utgår som
forhåndsregistrert felles target-grense. Hele perioden er gjenbrukt utviklingshistorikk,
ikke et urørt holdout. Ingen 2026-årsfil eller TEST-utfall ble åpnet.

Terminal, 30 resultatfiler og hasher er kontrollert. Et etterfølgende capped
audit gjenbrukte bare dagspanel/prognoser/FITS og bekreftet at alle siste
ytre TRAIN-utfall var tilgjengelige senest ved første beslutning i folden.
Ingen ny fit eller omkjøring ble gjort for rapporten.

## Primær økonomi

Netto er prosent av initialkapital over hele perioden, ikke årlig avkastning.
Sharpe bruker null kontantavkastning og den deklarerte 252-annualiseringen.
Samme regel for kausalt risikobudsjett brukes; realisert risiko kan være ulik.
Ujustert kjøp-og-hold er en separat diagnostikk med fast antall enheter.

| Policy | Netto historisk finansiering | Sharpe | Maks fall | Realisert årsvol | Netto uten finansiering |
|---|---:|---:|---:|---:|---:|
| Ridge | 14.50 % | 0.131 | 64.66 % | 14.43 % | 43.56 % |
| HGB | 59.21 % | 0.307 | 50.46 % | 11.80 % | 89.27 % |
| Kausal konstant | 61.04 % | 0.333 | 41.45 % | 10.76 % | 92.18 % |
| Risiko-LONG | 61.04 % | 0.333 | 41.45 % | 10.76 % | 92.18 % |
| Enkel trend | 45.83 % | 0.314 | 26.05 % | 8.80 % | 65.94 % |
| Kjøp-og-hold, ujustert | 156.11 % | 0.435 | 49.52 % | 16.94 % | 197.57 % |

Konstantens fortegn var LONG på samtlige 4 003 prognoser og ga nøyaktig samme
portefølje som risiko-LONG. HGBs positive netto slår ikke denne enkle referansen
på samlet netto, Sharpe eller maksimalt fall. Ridge gjør det vesentlig svakere.
Positiv historisk avkastning alene er derfor ikke dokumentert indikatorverdi.

## Kostnader og sekundær horisont

Alle tall i neste tabell er bps av initialkapital. Spread/slippage belastes
faktiske beholdningsendringer; finansiering inkluderer veggklokke og signert
SHORT-kreditt. Ingen kunstig lukking ved årsskifte eller target-horisont.

| Policy, h20 | Mid-PnL | Spread | Slippage | Finansiering | Netto |
|---|---:|---:|---:|---:|---:|
| Ridge | 6442.56 | 1492.95 | 593.52 | 2906.50 | 1449.58 |
| HGB | 10607.89 | 1126.36 | 554.90 | 3005.85 | 5920.79 |
| Risiko-LONG | 9792.04 | 398.31 | 175.49 | 3114.15 | 6104.10 |
| Enkel trend | 8784.67 | 1452.84 | 738.27 | 2010.60 | 4582.96 |
| Kjøp-og-hold, ujustert | 19778.55 | 14.07 | 7.96 | 4145.24 | 15611.28 |

På sekundær h5 er netto etter historisk finansiering -0,28 % for ridge
og +30,58 % for HGB. Dette er diagnostikk, ingen ny kandidatseleksjon.
Nullfinansiering gir henholdsvis +28,60 % og +58,87 %.

## Modellvalg og senere prognosefeil

| Horisont | Ridge på øvre/nedre gridgrense | Konstant ville slå ridge ved indre valg | HGB valgte konstant | Senere MSE ridge / HGB / konstant |
|---|---:|---:|---:|---:|
| 20 D1 | 4/15 og 2/15 | 4/15 | 9/15 | 13.0586 / 14.7006 / 11.8412 |
| 5 D1 | 4/15 og 1/15 | 4/15 | 8/15 | 2.7140 / 2.6408 / 2.6406 |

MSE er i ATR14-enheter i andre, på alle 4 003 senere prognoser.
Indre modellvalg er TRAIN-intern tilpasning; det er ikke senere generalisering.
Den separate konstante prognosen har lavest samlet senere MSE ved begge horisonter.
Full in-sample TRAIN-MSE er ikke rekjørt for rapportering. H20 ridge ga 3 271 LONG
og 732 SHORT; HGB ga 3 715 LONG og 288 SHORT. Ingen av disse armene var all-FLAT.

## Samtidig inferens og styrke

Alle 96 erklærte endepunkter var statistisk definerte og inngår i samme
max-|t|-korreksjon: 1 999 parede stasjonære bootstrap-trekk, forventet blokk
60 D1-intervaller, alpha 0,05. 86 endepunkter er INKONKLUSIV og 10 NO_GO;
ingen GO. De ti NO_GO er ikke blant de påkrevde primære endepunktene som
bestemmer modellenes samlede dom. Hele familien er publisert i JSON-rapporten.

Her vises h20 mot risiko-LONG med historisk finansiering. Styrke er den
forhåndsregistrerte betingede lokasjonsskiftmodellen, ikke en garanti ved
framtidige regimeskifter. MDE er ved 80 % styrke mot null.

| Modell/statistikk | Estimat | Simultant 95 % intervall | Tre deklarerte effekter | Styrke ved disse | MDE80 |
|---|---:|---:|---|---|---:|
| ridge / mean_delta_bps | -0.6691 | [-4.2407, 2.9025] | 1 / 2 / 5 | 1.2 % / 8.5 % / 87.7 % | 4.5795 |
| ridge / normalized_delta | -0.0048 | [-0.0539, 0.0443] | 0.01 / 0.02 / 0.05 | 1.0 % / 3.3 % / 53.6 % | 0.0634 |
| ridge / sharpe_delta | -0.2014 | [-0.8660, 0.4632] | 0.1 / 0.2 / 0.3 | 0.7 % / 1.6 % / 4.1 % | 0.8591 |
| hgb / mean_delta_bps | 0.0180 | [-2.1470, 2.1830] | 1 / 2 / 5 | 5.4 % / 40.6 % / 99.9 % | 2.7732 |
| hgb / normalized_delta | 0.0001 | [-0.0284, 0.0286] | 0.01 / 0.02 / 0.05 | 2.3 % / 18.8 % / 98.2 % | 0.0363 |
| hgb / sharpe_delta | -0.0255 | [-0.4929, 0.4418] | 0.1 / 0.2 / 0.3 | 1.2 % / 4.5 % / 13.8 % | 0.5979 |

Ridge-minus-LONG er -0,67 bps per intervall, med simultant intervall
[-4,24; +2,90]. HGB-minus-LONG er +0,018 bps, med [-2,15; +2,18].
MDE på henholdsvis 4,58 og 2,77 bps er større enn minste relevante effekt
på 1 bps. Det er derfor upresist å erklære små effekter motbevist.
Dette åpner heller ikke trening: det foreligger ingen positiv merverdibeslutning.

## Årsfordeling

Gjennomsnittlig netto perioderetur i bps med historisk finansiering, h20.
Årstabellen er beskrivende. Ingen år fjernes eller brukes til ny regelvalg.

| År | Intervaller | Ridge | HGB | Risiko-LONG | Trend | Kjøp-og-hold |
|---|---:|---:|---:|---:|---:|---:|
| 2011 | 349 | 4.736 | 3.708 | 3.708 | 3.437 | 2.600 |
| 2012 | 306 | 0.515 | 0.540 | 0.540 | -1.802 | 2.242 |
| 2013 | 267 | -8.645 | -8.996 | -8.996 | 3.711 | -12.225 |
| 2014 | 258 | -3.518 | -6.730 | -1.467 | -1.213 | -0.067 |
| 2015 | 259 | -5.357 | -6.017 | -5.796 | -1.201 | -5.526 |
| 2016 | 258 | -3.928 | 2.430 | 2.063 | 1.391 | 3.388 |
| 2017 | 257 | -4.548 | 5.460 | 4.748 | -4.102 | 4.011 |
| 2018 | 260 | -2.377 | -1.215 | -1.108 | 0.077 | -1.785 |
| 2019 | 259 | 8.339 | 6.138 | 5.413 | 4.025 | 6.500 |
| 2020 | 259 | 6.884 | 6.433 | 6.574 | 4.065 | 10.610 |
| 2021 | 258 | -4.237 | -2.673 | -1.437 | -3.168 | -1.940 |
| 2022 | 258 | -4.225 | 5.308 | -0.242 | -2.243 | -0.473 |
| 2023 | 257 | 3.269 | 1.754 | 2.072 | -0.987 | 4.610 |
| 2024 | 259 | 8.599 | 5.184 | 5.101 | 5.008 | 9.950 |
| 2025 | 238 | 15.855 | 10.720 | 10.580 | 9.689 | 24.351 |

## Evidensgrense og neste arbeid

- Målt: de deklarerte indikatorene, årsfittene, prognosene, modellbasert
  BID/ASK-porteføljeøkonomi og betinget inferens på gjenbrukt 2011–2025-historikk.
- Bevist konsistent: kildehasher, lukket D1/utførelsesklokke, kausale ytre
  target-grenser, gjenbrukte eiere, capped-kjøring og terminal/inventar.
- Ikke undersøkt: faktisk ordreutførelse, historiske brokerswapper, native
  læring, framtidig generalisering, TEST og B/C-resultater.

Kilden har også delvise/helgedøgn: 2011 har 350 predikerte D1-biner,
2012 har 306, senere år nær 258. De er beholdt etter den registrerte
source-absence-kontrakten. 20 D1 betyr 20 observerte biner, og 252 er
konfigurert annualisering; dette skal ikke omtales som en uniform børskalender.
Ingen post-hoc endring av denne klokken er gjort.

Fast initialt risikobudsjett betyr at samme regel ikke gir lik realisert
volatilitet eller konstant giring relativt til løpende egenkapital.
Finansiering er en EFFR-proxy med fast påslag og UTC-døgnkonvensjon,
ikke verifiserte historiske brokerbelastninger. EFFR-metodikken endret seg
i mars 2016. Blokkinferens beskriver ikke alle mulige regimeskifter.

A avsluttes uten ny fit eller parametersøk. Fortsett den vedtatte B-kildekontrollen
og C-registreringen. B må dokumentere faktisk publisering og historiske vintager;
manglende kildebevis skal lukkes uttrykkelig. Ingen native trening åpnes.

[Maskinlesbar aggregatrapport](TA_A_RESULT_20260929.json).
