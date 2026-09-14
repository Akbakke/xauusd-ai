# Juni-VAL etter første femårs-epoch — 14. september 2026

Treningen er stoppet, og Windows-oppgaven er deaktivert. Epoch 2 hadde automatisk startet før stoppbeskjeden og rakk 320 lagrede optimizersteg. Første epochs uforanderlige EMA og hele VAL-resultatet er bevart. Ingen videre trening før Entry/Exit-problemene er avklart.

## Hva månedsslutt uten modellstyrt lukking betyr

Exit valgte fortsatt HOLD ved siste tilgjengelige juniminutt. Simuleringen mangler senere VAL-priser og avsluttes derfor som avkortet ved månedsslutt. Dette er verken en modellvalgt Exit eller en observert handel med null tap.

11 016 er to hypotetiske sider for hver av 5 508 innganger. Modellen valgte 4 180 LONG, 1 328 SHORT og ingen FLAT. De 7 472 lukkede forløpene inkluderer sider Entry ikke valgte. For den faktisk valgte siden er tallene:

| Resultat | Antall | Gjennomsnittlig netto Bps | Median holdetid | Verste observerte netto MAE |
|---|---:|---:|---:|---:|
| Modellstyrt lukket | 2 227 | +125,62 faktisk lukket | 35,38 timer | −880,06 Bps |
| Fortsatt HOLD ved månedsslutt | 3 281 | −610,55 ved hypotetisk månedssluttslukking | 303,82 timer | −1 369,27 Bps |
| Alle valgte | 5 508 | −312,90 ved hypotetisk månedssluttslukking av de åpne | 169,83 timer | −1 369,27 Bps |

Hypotetisk månedssluttslukking er en diagnostisk beregning, ikke modellens lærte Exit, offisielt full-policy-resultat eller porteføljeavkastning. Den offisielle netto Bps-metrikken er utilgjengelig fordi så mange valgte handler fortsatt er åpne. Positiv statistikk bare for lukkede handler gir et misvisende bilde.

## Entry og Exit bidrar begge

- Blant de 3 281 åpne valgte handlene kom største gunstige utslag før verste ugunstige utslag i omtrent 89 % av tilfellene.
- 72,42 % av disse hadde først minst +20 netto Bps og gikk senere under −20 netto Bps uten modellstyrt lukking. Dette viser konkret at Exit lar tidligere gevinst snu til tap.
- Entry har også ugunstig retning/timing: på alle valgte innganger er avkastningen etter én time i snitt −8,39 netto Bps; bare 38,93 % er positive. Dette er en fast-horisont-diagnose, uavhengig av faktisk Exit.
- Av de åpne valgte handlene hadde 3 175 en hypotetisk motsatt side som modellen faktisk lukket. Det beviser ikke at en motsatt strategi ville være lønnsom på nye data.
- Oppsettets særskilte hold-risikostraff er null på begge sider. Finansiering og øvrige kostnader er med. Det finnes dermed ingen egen eksplisitt MAE-straff i denne kostnadspolicyen; dette alene beviser ikke en kodefeil.

## Færre handler krever et filter som faktisk skiller godt fra dårlig

Q-verdier er forventede Bps, ikke kalibrerte sannsynligheter. Vi undersøkte forskjellen mellom valgt Q og beste alternativ. De 10 % største forskjellene beholdt 551 handler, men ga fortsatt omtrent −218,62 netto Bps per handel i samme hypotetiske sluttberegning. Et slikt høyere filter løser altså ikke problemet alene. Ingen terskel er endret eller valgt ved å optimere på denne ene VAL-måneden.

## Målegrunnlag og neste steg

Analysen bruker faktisk Entry-fill og sidekorrekte bid/ask-sluttkurser per minutt, 4 Bps samlet slippage og bundet finansiering. Alle lukkede nettoresultater ble rekonstruert med maksimal differanse 0,000000000002 Bps. Bevegelse og rekkefølge innen samme minutt er ikke observert; MAE kan være verre intraminutt.

Avklar ønsket holdetid og akseptabel MAE før treningsmål eller exit-/entryregler endres. Deretter velges minste tiltak som adresserer den observerte Entry/Exit-atferden. Ikke start en ny epoch som analyse eller test.

## Fartstiltak

En separat kildekopi åpner for VAL-batch 256, åtte CPU-arbeidere og tre timers VAL-vinduer. Den ytre tidsvakten beholder 30 minutters margin, og temperatur-, strøm-, RAM-, VRAM- og øvrige grenser er uendret. Gamle profiler beholder sine grenser. Kjøringen er ikke aktivert.

På 1 024 identiske tilstander fra juniprisene var CPU-trinnet cirka 17 % raskere med åtte arbeidere/batch 128 og cirka 25 % raskere med åtte arbeidere/batch 256, med identiske tilstandsbytes. Fem repetisjoner per variant; oppstart målt separat, CPU 0–7. Dette er kun materialisering av prissti og livstidssammendrag, ikke hele VAL eller GPU-hastighet. GPU-batch 256 og samlet gjennomstrømning må fortsatt måles før profilen velges. Eksisterende FP32-kontroll krever identiske handlinger og absolutt Q-avvik høyst 0,0001 Bps.

Kilder: `VAL_RESULT_EPOCH_1.json`, `ENTRY_EXIT_DIAGNOSTIC.json`, `ENTRY_EXIT_SUMMARY.json`, `handover_snapshot/CPU_CAPACITY_RESULT_20260914.json`. Ingen TEST-data er brukt.

Forberedt kilde: `2ab85a7548aa0e2132779de26066ecb4bc10c6f6` i `/home/andre2/src/GX1_VAL_CAPACITY_V41`, branch `perf/native-val-windows-20260914`. Ikke aktivert eller migrert. Før eventuell bruk kreves kildebundet profil og målt GPU-paritet/gjennomstrømning; ingen ny treningsstart er autorisert som del av denne analysen.
