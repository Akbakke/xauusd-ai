# Frossen M5-modell: GC-ferskhet på Q1 — 30.10.2026

Modellen reagerer på GC-informasjonen, men forskjellen viser ikke en sikker fordel for fersk GC. På 1 717 gyldige femminuttersrader endret 90 (5,24 %) binær retning når navngitt GC-kontekst og nye historikkbøtter ble holdt tilbake. Fersk/tilbakeholdt: AUC 0,547636 / 0,542785, treff 50,67 % / 51,02 %; alltid opp 51,25 %. Dette er et fast delutvalg av gjenbrukte Q1-data, ikke bevis på edge.

AUC-differansen tilbakeholdt minus fersk var -0,004851. Parrede dag-/ukeblokkintervaller var [-0,017980; 0,008880] / [-0,013201; 0,007116]. Begge omfatter null. Femminutters logloss-/Brier-intervallene omfatter også null. Seks minutter: AUC 0,523612 / 0,512985 og treff 52,65 % / 52,07 % på 1 717 rader; beskrivende.

Samme modell hadde AUC 0,506181 og treff 48,94 % på hele Q1-femminutterspopulasjonen, 17 192 gyldige rader. Høyere skår på delutvalget er ikke en matchet GC-effekt. Utvalget består av hver tiende 16-raders batch, ikke tilfeldig eller sessionsstratifisert trekking.

## Metode og grenser

Frosset to-pass-modell, korrigert fasit 1–60 minutter og M5-entry med lukkede høyere tidsrammer. Alle 1 736 valgte Q1-prediksjoner ble gjenskapt eksakt før den tilhørende batchen ble endret; 1 727 hadde eksakt forrige M5-klokke og ble sammenlignet.

Navngitte GC-kontekstverdier, masker og siste lokale aliaser ble erstattet med forrige M5-verdi. Nytt GC-historikkinnhold ble maskert: 1 727 M5-bøtter, 573 M15, 138 H1, 37 H4 og 0 D1. Øvrige input og vekter ble bevart. Felles endring kan gi input utenfor treningsfordelingen, isolerer ingen enkelt feature og beviser ikke lokal datatilgjengelighet eller lønnsomhet.

Fem og 60 minutter har forhåndsdefinerte parrede dag-/ukeblokkbånd for AUC, logloss og Brier, med 10 000 bootstraprepeteringer og per-intervall alfa 0,05/12. Øvrige horisonter er beskrivende. Kalendermessig utbyttbarhet er en antakelse.

Kjøringen hadde 218 forward, null optimizersteg, signert vakt PASS og topp 53/56 °C og 134,6 W. Ingen TEST, trening eller handel. Konfigurert effektgrense var 250 W, senere trekk 23,17 W. En separat CPU-kontroll gjenskapte kohort, fasiter, originallogits, alle 60 skårer og 12 blokkbånd. Maksimalt avvik: 1,2e-16 for skår og 4,4e-18 for intervall.

Resultat SHA-256: `a551132353028cae2561bdd013303d0f90ed8dc12ee11f80f5e593adae3f1581`. Kontroll SHA-256: `6527188090594fbe8ea3d0876063a0f540f8fff8aadb72b8b7b47647a78a95e4`.

## Alle 60 horisonter

Fersk og tilbakeholdt bruker samme rader. Horisontene er prognosehorisonter, ikke entryfrekvens eller holdetid.

| Min | N | Treff fersk | Treff tilbakeholdt | Alltid opp | AUC fersk | AUC tilbakeholdt | Retninger endret |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1721 | 50.55% | 49.39% | 48.29% | 0.4934 | 0.4816 | 168 |
| 2 | 1720 | 50.87% | 50.87% | 49.07% | 0.5085 | 0.5006 | 44 |
| 3 | 1718 | 50.81% | 51.11% | 49.42% | 0.5226 | 0.5159 | 71 |
| 4 | 1726 | 50.58% | 50.46% | 51.27% | 0.5393 | 0.5380 | 84 |
| 5 | 1717 | 50.67% | 51.02% | 51.25% | 0.5476 | 0.5428 | 90 |
| 6 | 1717 | 52.65% | 52.07% | 49.04% | 0.5236 | 0.5130 | 110 |
| 7 | 1715 | 52.83% | 52.83% | 49.56% | 0.5320 | 0.5248 | 196 |
| 8 | 1719 | 53.11% | 52.01% | 50.61% | 0.5374 | 0.5217 | 193 |
| 9 | 1714 | 50.99% | 51.63% | 51.58% | 0.5439 | 0.5437 | 105 |
| 10 | 1709 | 49.85% | 49.74% | 51.73% | 0.5175 | 0.5155 | 80 |
| 11 | 1706 | 51.82% | 51.64% | 50.00% | 0.5312 | 0.5290 | 87 |
| 12 | 1707 | 50.44% | 50.21% | 52.08% | 0.5410 | 0.5389 | 106 |
| 13 | 1710 | 48.89% | 48.89% | 52.40% | 0.5208 | 0.5165 | 70 |
| 14 | 1709 | 50.38% | 50.61% | 52.19% | 0.5440 | 0.5381 | 92 |
| 15 | 1703 | 48.91% | 48.91% | 53.49% | 0.5419 | 0.5398 | 92 |
| 16 | 1700 | 49.71% | 49.53% | 52.53% | 0.5355 | 0.5316 | 85 |
| 17 | 1701 | 49.27% | 49.50% | 53.26% | 0.5269 | 0.5252 | 68 |
| 18 | 1700 | 48.47% | 48.00% | 54.35% | 0.5328 | 0.5279 | 110 |
| 19 | 1701 | 47.68% | 47.27% | 53.38% | 0.5251 | 0.5211 | 101 |
| 20 | 1694 | 48.52% | 48.58% | 53.31% | 0.5382 | 0.5304 | 71 |
| 21 | 1694 | 48.88% | 48.64% | 53.54% | 0.5380 | 0.5352 | 74 |
| 22 | 1694 | 48.29% | 47.99% | 54.25% | 0.5380 | 0.5354 | 99 |
| 23 | 1695 | 48.44% | 48.26% | 54.57% | 0.5326 | 0.5331 | 81 |
| 24 | 1690 | 51.89% | 51.60% | 54.62% | 0.5383 | 0.5344 | 225 |
| 25 | 1686 | 47.51% | 47.33% | 54.74% | 0.5259 | 0.5227 | 75 |
| 26 | 1684 | 48.34% | 48.52% | 53.74% | 0.5307 | 0.5323 | 85 |
| 27 | 1683 | 48.60% | 48.31% | 53.65% | 0.5341 | 0.5372 | 75 |
| 28 | 1683 | 49.73% | 50.51% | 54.01% | 0.5372 | 0.5400 | 127 |
| 29 | 1684 | 48.93% | 48.28% | 54.16% | 0.5284 | 0.5253 | 121 |
| 30 | 1678 | 46.90% | 46.60% | 54.59% | 0.5374 | 0.5365 | 61 |
| 31 | 1679 | 48.18% | 48.78% | 54.08% | 0.5377 | 0.5363 | 90 |
| 32 | 1680 | 46.96% | 47.26% | 54.23% | 0.5369 | 0.5374 | 69 |
| 33 | 1676 | 47.61% | 47.73% | 54.06% | 0.5519 | 0.5513 | 72 |
| 34 | 1677 | 48.84% | 48.42% | 54.38% | 0.5490 | 0.5434 | 83 |
| 35 | 1671 | 48.35% | 48.71% | 54.70% | 0.5502 | 0.5490 | 78 |
| 36 | 1674 | 46.00% | 46.30% | 54.66% | 0.5424 | 0.5399 | 65 |
| 37 | 1672 | 47.91% | 47.61% | 55.08% | 0.5467 | 0.5389 | 137 |
| 38 | 1672 | 47.07% | 46.59% | 54.13% | 0.5509 | 0.5461 | 76 |
| 39 | 1673 | 47.46% | 46.68% | 54.57% | 0.5579 | 0.5464 | 71 |
| 40 | 1666 | 47.60% | 46.88% | 55.28% | 0.5491 | 0.5448 | 90 |
| 41 | 1668 | 47.18% | 47.00% | 54.92% | 0.5586 | 0.5581 | 69 |
| 42 | 1665 | 46.37% | 45.47% | 54.83% | 0.5372 | 0.5364 | 77 |
| 43 | 1667 | 49.55% | 48.83% | 54.29% | 0.5425 | 0.5384 | 110 |
| 44 | 1665 | 47.21% | 46.55% | 55.44% | 0.5603 | 0.5580 | 71 |
| 45 | 1663 | 47.68% | 47.38% | 55.56% | 0.5590 | 0.5560 | 85 |
| 46 | 1661 | 45.94% | 45.51% | 54.97% | 0.5357 | 0.5332 | 63 |
| 47 | 1660 | 48.07% | 47.35% | 54.52% | 0.5513 | 0.5497 | 76 |
| 48 | 1663 | 50.75% | 49.85% | 54.96% | 0.5553 | 0.5503 | 145 |
| 49 | 1663 | 45.94% | 45.52% | 55.74% | 0.5535 | 0.5564 | 89 |
| 50 | 1655 | 46.34% | 45.74% | 56.56% | 0.5638 | 0.5631 | 78 |
| 51 | 1654 | 46.31% | 46.01% | 55.32% | 0.5548 | 0.5543 | 81 |
| 52 | 1654 | 45.71% | 45.83% | 56.11% | 0.5345 | 0.5363 | 68 |
| 53 | 1653 | 50.51% | 50.15% | 56.50% | 0.5566 | 0.5552 | 154 |
| 54 | 1653 | 44.77% | 44.34% | 56.68% | 0.5535 | 0.5529 | 71 |
| 55 | 1647 | 45.72% | 45.78% | 56.77% | 0.5606 | 0.5632 | 77 |
| 56 | 1648 | 46.18% | 45.87% | 56.19% | 0.5649 | 0.5648 | 75 |
| 57 | 1649 | 45.30% | 45.54% | 57.06% | 0.5563 | 0.5571 | 78 |
| 58 | 1648 | 47.69% | 47.15% | 56.13% | 0.5613 | 0.5604 | 97 |
| 59 | 1648 | 45.51% | 45.02% | 56.49% | 0.5407 | 0.5386 | 78 |
| 60 | 1642 | 44.52% | 43.79% | 57.31% | 0.5472 | 0.5447 | 76 |
