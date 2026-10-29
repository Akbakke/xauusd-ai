# M5-momentum: kontroll av faktisk modellbidrag

Kandidaten ble trent på M5-entry med lukkede høyere tidsrammer. Kontrollens inngrep setter ett eller alle fire XAU-felt til null ved den normaliserte/gatede inngangen til den felles momentumprojeksjonen på M5, M15, H1, H4 og D1. GC, lokale aliaser, projeksjonsbias og øvrige ruter beholdes. Dette er ikke retrening.

På 1726 modne, observerte utfall med ikke-null femminuttersretning på et forhåndsbestemt utvalg av 1 736 gjenbrukte Q1 2025-foreldre endret ingen av de fire enkeltinngrepene eller det samlede inngrepet en opp/ned-beslutning. Det samme gjelder alle 60 horisonter i dette utvalget og 256 diagnostiske TRAIN-foreldre. Feltverdiene nådde projeksjonsinngangen: 99.603–100.000 % av de registrerte tidsrute-cellene var ikke-null. Cellene inkluderer overlappende historikk og er ikke uavhengige observasjoner.

Gjennomsnittlig absolutt endring i femminutterslogit ved samlet inngrep var 0,0000011575. Original AUC var 0,510522 og AUC med alle fire nullstilt 0,510519 på dette utvalget. Dette delutvalgets punktestimat er ikke bevis på edge; hele den tidligere Q1-sammenligningen hadde AUC 0,496961.

Uavhengig CPU-replay gjenskapte de lagrede originalprediksjonene nøyaktig og kontrollerte 600 original-/inngrepspaneler for AUC, logloss og Brier samt 600 påvirkningspaneler. Maksimal numerisk forskjell var 5,56e-17. Den uavhengige kontrollen gjenskapte ikke bootstrapintervallene eller observatorens inputtellinger.

Originalkjøringen ble fullført på 525 sekunder, med 750 forwardkall og null optimizersteg; modell og kilde var uendret. Signert GPU-vakt passerte: maksimal målt effekt 147,48 W, kjernetemperatur 48 °C, minnetemperatur 52 °C. Konfigurert grense er 250 W. TEST og handel ble ikke brukt.

## Paret AUC-endring ved 5 og 60 minutter

Tallene er inngrep minus original. Intervallene kommer fra originalresultatet, med 10 000 dag-/ukeblokk-bootstraptrekninger og familiejustert alfa 0,0008333 per intervall. De inkluderer null; blokkuavhengighet/utbyttbarhet og generalisering er ikke bevist.

| Felt | Minutter | Δ AUC | Dagintervall | Ukeintervall |
|---|---:|---:|---:|---:|
| rsi14_centered | 5 | +0.00e+00 | [+0.00e+00, +0.00e+00] | [+0.00e+00, +0.00e+00] |
| rsi14_centered | 60 | +7.50e-07 | [+0.00e+00, +9.62e-06] | [+0.00e+00, +8.44e-06] |
| macd_line_atr | 5 | +0.00e+00 | [+0.00e+00, +0.00e+00] | [+0.00e+00, +0.00e+00] |
| macd_line_atr | 60 | +1.50e-06 | [+0.00e+00, +1.92e-05] | [+0.00e+00, +1.69e-05] |
| macd_hist_atr | 5 | +0.00e+00 | [+0.00e+00, +0.00e+00] | [+0.00e+00, +0.00e+00] |
| macd_hist_atr | 60 | +0.00e+00 | [+0.00e+00, +0.00e+00] | [+0.00e+00, +0.00e+00] |
| ret_1 | 5 | +0.00e+00 | [+0.00e+00, +0.00e+00] | [+0.00e+00, +0.00e+00] |
| ret_1 | 60 | +1.50e-06 | [-2.29e-05, +2.53e-05] | [-1.65e-05, +2.58e-05] |
| all_four | 5 | -2.69e-06 | [-2.45e-05, +0.00e+00] | [-2.10e-05, +0.00e+00] |
| all_four | 60 | +1.50e-06 | [-1.90e-05, +2.59e-05] | [-1.53e-05, +2.46e-05] |

## Alle 60 horisonter

Hvert tall er original AUC eller endring i AUC etter nulling, på samme kvalifiserte rader innen horisonten. Antallet kvalifiserte rader varierer mellom horisontene. Ingen av inngrepene endret en binær beslutning i noen av disse panelene.

| Minutter | Rader | Original AUC | Δ RSI14 | Δ MACD-linje | Δ MACD-hist. | Δ ret_1 | Δ alle fire |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1730 | 0.485877 | +0.00e+00 | +2.68e-06 | +0.00e+00 | -2.68e-06 | +0.00e+00 |
| 2 | 1729 | 0.493227 | +0.00e+00 | +0.00e+00 | +0.00e+00 | -2.68e-06 | -1.34e-06 |
| 3 | 1727 | 0.507632 | +0.00e+00 | -1.34e-06 | -1.34e-06 | -1.34e-06 | +0.00e+00 |
| 4 | 1735 | 0.524393 | +1.33e-06 | +1.33e-06 | +1.33e-06 | +3.99e-06 | +3.99e-06 |
| 5 | 1726 | 0.510522 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | -2.69e-06 |
| 6 | 1726 | 0.483895 | -2.69e-06 | -4.03e-06 | -1.34e-06 | -2.69e-06 | -8.73e-06 |
| 7 | 1724 | 0.524738 | +1.35e-06 | -1.35e-06 | +0.00e+00 | +2.69e-06 | +2.69e-06 |
| 8 | 1728 | 0.498675 | +5.36e-06 | +5.36e-06 | +0.00e+00 | +4.02e-06 | +2.68e-06 |
| 9 | 1723 | 0.532019 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.35e-06 |
| 10 | 1718 | 0.511022 | +1.36e-06 | -2.71e-06 | +1.36e-06 | -1.36e-06 | +0.00e+00 |
| 11 | 1715 | 0.522587 | +0.00e+00 | +1.36e-06 | +1.36e-06 | -1.36e-06 | +5.44e-06 |
| 12 | 1716 | 0.536085 | +1.36e-06 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 |
| 13 | 1719 | 0.522679 | +1.36e-06 | +1.36e-06 | +0.00e+00 | +1.36e-06 | +4.07e-06 |
| 14 | 1718 | 0.533600 | -1.36e-06 | -1.36e-06 | +2.72e-06 | -1.36e-06 | -2.04e-06 |
| 15 | 1712 | 0.531633 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.37e-06 | +1.37e-06 |
| 16 | 1709 | 0.538635 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.37e-06 |
| 17 | 1710 | 0.540643 | -1.37e-06 | -2.75e-06 | -1.37e-06 | -1.37e-06 | +0.00e+00 |
| 18 | 1709 | 0.522057 | +1.38e-06 | +1.38e-06 | +0.00e+00 | +1.38e-06 | +1.38e-06 |
| 19 | 1710 | 0.514836 | +0.00e+00 | +0.00e+00 | +1.37e-06 | +1.37e-06 | +0.00e+00 |
| 20 | 1703 | 0.503043 | -6.93e-06 | -6.93e-06 | +0.00e+00 | -1.39e-06 | -8.31e-06 |
| 21 | 1703 | 0.524844 | +0.00e+00 | +0.00e+00 | -1.39e-06 | +0.00e+00 | +1.39e-06 |
| 22 | 1703 | 0.515663 | +1.39e-06 | +1.39e-06 | +1.39e-06 | +0.00e+00 | +1.39e-06 |
| 23 | 1704 | 0.506887 | -5.56e-06 | -4.17e-06 | +0.00e+00 | -4.17e-06 | -6.94e-06 |
| 24 | 1699 | 0.536084 | -1.40e-06 | +0.00e+00 | -2.79e-06 | -1.40e-06 | -5.59e-06 |
| 25 | 1695 | 0.529657 | -2.81e-06 | +0.00e+00 | -1.40e-06 | -2.81e-06 | -1.40e-06 |
| 26 | 1693 | 0.538592 | +1.40e-06 | +1.40e-06 | +0.00e+00 | +1.40e-06 | +4.21e-06 |
| 27 | 1692 | 0.512343 | +0.00e+00 | +1.40e-06 | -1.40e-06 | -2.81e-06 | -2.81e-06 |
| 28 | 1692 | 0.512783 | +1.41e-06 | +1.41e-06 | +1.41e-06 | +0.00e+00 | +0.00e+00 |
| 29 | 1693 | 0.528054 | +0.00e+00 | +7.03e-07 | +7.03e-07 | +2.81e-06 | +5.62e-06 |
| 30 | 1687 | 0.517824 | -1.42e-06 | -1.42e-06 | -1.42e-06 | +0.00e+00 | -7.09e-07 |
| 31 | 1688 | 0.538070 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.41e-06 | +2.83e-06 |
| 32 | 1689 | 0.521849 | +2.12e-06 | +1.41e-06 | +0.00e+00 | +2.82e-06 | +1.13e-05 |
| 33 | 1685 | 0.526378 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +2.84e-06 |
| 34 | 1686 | 0.518485 | +1.42e-06 | +0.00e+00 | +0.00e+00 | +0.00e+00 | -4.25e-06 |
| 35 | 1680 | 0.522468 | +0.00e+00 | -1.43e-06 | +1.43e-06 | +1.43e-06 | +0.00e+00 |
| 36 | 1683 | 0.528995 | +1.42e-06 | +1.42e-06 | +0.00e+00 | +2.85e-06 | +1.42e-06 |
| 37 | 1681 | 0.532017 | +1.43e-06 | +0.00e+00 | +0.00e+00 | +1.43e-06 | +1.43e-06 |
| 38 | 1681 | 0.550398 | +0.00e+00 | +0.00e+00 | +1.43e-06 | +0.00e+00 | +0.00e+00 |
| 39 | 1682 | 0.545312 | -1.43e-06 | -1.43e-06 | +0.00e+00 | +0.00e+00 | +0.00e+00 |
| 40 | 1675 | 0.559321 | -7.21e-07 | +7.21e-07 | +7.21e-07 | -7.21e-07 | -7.21e-07 |
| 41 | 1677 | 0.541429 | -2.87e-06 | -4.31e-06 | +0.00e+00 | +1.44e-06 | -7.18e-06 |
| 42 | 1674 | 0.538416 | -1.44e-06 | -1.44e-06 | +1.44e-06 | -1.44e-06 | -4.32e-06 |
| 43 | 1676 | 0.507770 | +0.00e+00 | -1.43e-06 | -1.43e-06 | -1.43e-06 | -2.87e-06 |
| 44 | 1674 | 0.532633 | +1.44e-06 | +1.44e-06 | +0.00e+00 | +0.00e+00 | +2.89e-06 |
| 45 | 1672 | 0.514947 | +0.00e+00 | +0.00e+00 | +0.00e+00 | -1.45e-06 | -1.45e-06 |
| 46 | 1670 | 0.532985 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.45e-06 |
| 47 | 1669 | 0.547728 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.45e-06 | +0.00e+00 |
| 48 | 1672 | 0.529841 | +0.00e+00 | +1.45e-06 | +0.00e+00 | +0.00e+00 | -1.45e-06 |
| 49 | 1672 | 0.512878 | +0.00e+00 | +0.00e+00 | +1.45e-06 | +0.00e+00 | -1.45e-06 |
| 50 | 1664 | 0.535868 | +0.00e+00 | +0.00e+00 | +0.00e+00 | -1.47e-06 | +0.00e+00 |
| 51 | 1663 | 0.505366 | -1.46e-06 | -2.93e-06 | +0.00e+00 | -2.93e-06 | -2.93e-06 |
| 52 | 1663 | 0.523372 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.47e-06 |
| 53 | 1662 | 0.521511 | +1.47e-06 | +0.00e+00 | +1.47e-06 | +1.47e-06 | +1.47e-06 |
| 54 | 1662 | 0.529689 | +1.47e-06 | +1.47e-06 | +1.47e-06 | -1.47e-06 | +1.47e-06 |
| 55 | 1656 | 0.524949 | -2.23e-06 | -7.43e-07 | -7.43e-07 | -7.43e-07 | +7.43e-07 |
| 56 | 1657 | 0.526345 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.48e-06 | +0.00e+00 |
| 57 | 1658 | 0.521037 | +1.49e-06 | +0.00e+00 | +1.49e-06 | +1.49e-06 | +4.46e-06 |
| 58 | 1657 | 0.515979 | -1.48e-06 | +0.00e+00 | -1.48e-06 | -1.48e-06 | -4.44e-06 |
| 59 | 1657 | 0.514651 | +0.00e+00 | +0.00e+00 | +0.00e+00 | +1.48e-06 | +1.48e-06 |
| 60 | 1651 | 0.518740 | +7.50e-07 | +1.50e-06 | +0.00e+00 | +1.50e-06 | +1.50e-06 |

## Begrensninger

Nulling er en intern intervensjon og kan gi en unaturlig tilstand. Samme informasjon kan finnes i andre inndata eller være korrelert med dem. Resultatet viser at denne ene ruten har svært liten funksjonell påvirkning i denne faste modellen og dette utvalget; det fastslår ikke hvorfor, hva retrening ville oppnådd, eller at modellen har handelsverdi. Fasit er en candle-basert proxy; eksakte quote-tider, kostnader og utførelse er ikke testet.

Originalt resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_MOMENTUM_INTERVENTION_20261029_003/OUTPUT/RESULT.json` SHA-256 `9b18161cec38395bb82debaef8c46caae31a14cdfb4959e12bd40e37ec1231a2`. Uavhengig resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_MOMENTUM_INDEPENDENT_20261029_001/RESULT.json` SHA-256 `3fc7e1b2c43d25e9af8c69b282605547af7750410d343a1ab37ac69989e52396`.
