# Første P0-utviklingstest — 17.10.2026

**Ingen dokumentert merverdi i denne oppskriften.** Alle åtte tremodellarmer taper netto, også i det billigste kostnadsscenarioet. Ingen familie gir sikker prediktiv forbedring mot CORE. Dette avkrefter ikke all mulig GC-verdi; forsøket gir ikke grunnlag for større trening eller native innføring.

021 og uavhengig revisjon 022 er COMPLETE, exit 0, uendret kilde, TEST=false. 40 små LightGBM-modeller og 40 regeltabeller, ingen HPO eller ny arkitektur. Selve studien tok 140 sekunder under capped audit (4GiB, CPU0–7, én numerisk tråd). Ingen nye rådata eller betalte kall.

## Frosset oppsett

26 eksisterende GC-måneder, mai 2024–juni 2026, er utviklingsdata. 37 181 felles F/L-inputvinduer. Fem ekspanderende fold ga 17 301 beslutninger fra mai 2025 til juni 2026. Tre etiketter passerer foldslutt og utelates fra primær prediksjonsevaluering, men ingen strategibeslutning fjernes ved hjelp av framtidig utfall. Tremodellfamilien har 17 298 identiske evalueringsrader. En ukjent LV-regeltilstand gir én abstensjon; regler sammenlignes på 17 297 felles rader. COMMON_PREDICTION_VIEW.json gir alle seksten modeller på samme 17 297 rader, uten refit.

Etiketten er 10⁴·ln(mid(t+31min)/mid(t+1min)), der t er M5-beslutningen. Primærhorisonten på 30 minutter ble valgt før fitting: kort intradagrespons, begrenset finansieringsbelastning og eksakt M1-bane for hele inputkohorten. 120/360 min er samme prognoses sekundærdiagnostikk, ingen ekstra modeller eller horisontseleksjon. TRAIN har ett døgn embargo og purging av faktiske framtidsutfall. Alle 26 måneder er utvikling; native TEST åpnes ikke.

CORE med 211 felt utelater de samme 61 ukvalifiserte makrofelt og tidligere pausede 27 mønsterfelt fra hver arm. F12 er de nye OFI/BBO-feltene, L8 de strenge rolling 15-nivå-/responsfeltene, V16 de eksisterende volatilitetsfeltene. SMC og kausal MTF beholdes; V måler ekstra eksplisitt volatilitetstilstand, ikke fravær av all volatilitet. Ingen nye ATR-er eller ufullstendige hele-session-profiler.

Tremodell: eksisterende LightGBM 4.6, 64 runder, dybde 4, 15 blader, minst 500 observasjoner per blad. Regelreferansen summerer retningstegn for EMA/OFI/VWAP-kontinuasjon og estimerer TRAIN-gjennomsnitt per tilstand; V bruker TRAIN-RV-terciler. Ingen krav om enstemmighet. Prognosene lærer observerte markedsutfall, uavhengig av Exit.

## Målt prediksjon og økonomi

Tall nedenfor bruker tremodellfamiliens felles 17 298 prediksjonsrader og alle 17 301 strategimuligheter. Netto og drawdown er bps av startkapital ved fast 1 x-notional. Sharpe er kalenderdaglig, annualisert med 365,25; drawdown bruker observerte M1-åpningskvoter.

| Arm | MSE bps² | Rank IC | Netto bps | Sharpe | Maks DD bps | Åpnet / åpen ved slutt |
|---|---:|---:|---:|---:|---:|---:|
| CORE | 422.20 | -0.0159 | -99.11 | -1.10 | 177.38 | 43 / 0 |
| F | 423.73 | -0.0163 | -185.95 | -0.93 | 293.94 | 25 / 0 |
| L | 421.96 | -0.0035 | -470.49 | -2.30 | 582.40 | 91 / 0 |
| V | 422.18 | -0.0154 | -141.44 | -0.45 | 383.27 | 19 / 1 |
| FL | 423.36 | -0.0081 | -355.83 | -1.60 | 393.51 | 32 / 1 |
| FV | 423.32 | -0.0167 | -499.98 | -1.47 | 604.13 | 31 / 0 |
| LV | 422.74 | -0.0121 | -209.26 | -0.84 | 388.82 | 24 / 1 |
| FLV | 423.60 | -0.0176 | -290.20 | -1.21 | 407.78 | 23 / 0 |

Alle samlede rank-IC er negative, og alle kalibreringshelninger er negative. TRAIN-gjennomsnitt som prognose gir MSE420,24, bedre enn alle tremodellene. L forbedrer CORE-MSE med 0,237 bps² og i 4/5 fold, men familiejustert 99,2857%-intervall er[−0,660;1,091]. FLV forbedrer ingen fold og har MSE1,404 bps² høyere enn CORE. Ingen sikker prediktiv gevinst.

| Arm | Utførelseskost/reserve bps | Finansiering bps | Eksponering % av tid | Turnover notional | Netto min–maks, 11 scenarioer |
|---|---:|---:|---:|---:|---:|
| CORE | 172.00 | 0.97 | 0.146 | 86.01 | -272.08 til -12.14 |
| F | 100.00 | 2.34 | 0.352 | 49.99 | -288.28 til -133.61 |
| L | 364.00 | 1.73 | 0.261 | 181.99 | -836.22 til -286.76 |
| V | 76.00 | 2.81 | 0.423 | 37.00 | -220.25 til -100.64 |
| FL | 128.00 | 3.19 | 0.481 | 62.99 | -487.03 til -288.64 |
| FV | 124.00 | 2.39 | 0.359 | 61.96 | -626.37 til -435.60 |
| LV | 96.00 | 2.19 | 0.330 | 47.00 | -307.45 til -159.07 |
| FLV | 92.00 | 1.79 | 0.270 | 45.98 | -383.99 til -242.41 |

Alle 288 åpnede tremodellposisjoner er LONG og åpnes i mai–juni 2026. De fire første foldene er bare FLAT: kostnadskravet overstiger prognosene. Dette dokumenterer verken tålmodighet, selektivitet eller bidireksjonell økonomisk verdi. FLV har 23 handler, CORE43; forskjellen i netto er−191,09 bps. Ukeblokk-intervallet for FLV minus CORE er[−2,442;1,198]bps per kalenderdag etter familiejustering: ikke sikkert. L minus CORE er−371,38 bps med intervallet[−2,583;−0,027]bps/dag; negativt i denne spesifikasjonen, basert på bare siste folds handler.4000 bootstrap-replikasjoner og 62 uker gir betinget usikkerhet, ikke uavhengig bekreftelse.

Regelreferansen er FLAT i sju armer. FLV gir 7 handler, netto−42,83 bps, Sharpe−1,63 og DD51,14 bps. Manglende handler er ingen kvalitetsbestått. Alltid-FLAT gir 0. Alltid-LONG markert ved slutt gir+1706,37 bps, Sharpe 0,38, DD5328,90 bps, finansiering 663,50 bps og 99,98% tids­eksponering. Det er en retningsreferanse med samme nominelle budsjett og kostmodell; faktisk eksponering er langt høyere enn modellene.

Sekundær FLV-IC er−0,0403 ved 120 min (16 907 eksakte baner) og−0,0289 ved 360 min (14 674). Dette er prognoseoverføring fra 30 min, ikke egne horisonters treningsresultater.

## Seleksjon og gjenværende grenser

F/L-kohorten dekker 26,22% av 141 806 native M5-vinduer. Årsdekning:2024=32,37%,2025=18,90%,2026=32,69%. ASIA30,60%, EU36,98%, OVERLAP7,97%, US21,14%. Lav/middels/høy TRAIN-RV-tercil har 39,69/28,91/18,96% dekning. EMA-ned/opp 25,09/27,19%. Månedsdekningen varierer 3,75–38,76%; oktober 2025 er svakest. Alle år og regimer finnes, men kohorten er klart selektert. Ingen masker er svekket, og resultatene kan ikke overføres til manglende vinduer. Fulltabeller ligger i REPRESENTATIVENESS.json.

CFTC-revisjonen 9 desember er dokumentert i CFTC_RESEARCH_RELEASE_HISTORY_V2_20261017.json: rapportdato 10 nov får planlagt publisering 10 des, rapportdato 23 des får 29 des. Opprinnelig novemberplan bevares som historikk; foreldede januar-etterslepsdatoer er utgått. [CFTCs reviderte primærkilde](https://www.cftc.gov/PressRoom/PressReleases/9147-25) og [historiske meldinger](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm) oppgir planlagt publisering. Faktisk levering og historiske verdirevisjoner er fortsatt ikke fullstendig dokumentert. Forskningsruten godtar bare eksplisitt kvalifisert faktisk publisering/vintage, aldri rapportdato+7. Alle 61 makrofelt er derfor ute av P0; november/desember-observasjoner beholdes. Native makroeier endres ikke.

## Regnskap og kontroll

Beslutning etter lukket M5; endring utføres på eksakt neste M1-åpning etter 60 sek. Beslutningen bruker den lineære kontantproksyen ±prognose minus siste lukkede XAU-spread, gebyr og projisert horisontfinansiering, sammenlignet med FLAT=0. Faktisk resultat beregnes med eksakt v3-regnskap. Én posisjon og fast 1 x-notional; ingen fast holdetid eller stopp. Hold gjennom featuremangler, sessionstengning og foldbytte til modellens FLAT/flip. Manglende utførelseskvote avviser endringen. Framtidsmasker styrer etiketter, aldri strategihandlinger.

Faktisk LONG=10⁴(exitBid−entryAsk)/entryAsk; SHORT=10⁴(entryBid−exitAsk)/entryBid. Spread er allerede med. Netto trekker rundturgebyr og årlig kost×faktisk veggklokketid/31 557 600. Sentralt 4 bps og LONG569/SHORT0 årlige bps; scenarioene 2/4/8 bps,0/1/2×LONG-finansiering samt ugunstig finansiering på begge sider. Signalene/handlene holdes fast i alle scenarioene. Gebyr reserveres én gang som full rundtur ved åpning, også åpen sluttposisjon. Historiske finansieringsvilkår er ikke bevist.

Råpriser og kanonisk v3-kosteier rekonstruerer alle 296 handler/åpne markeringer i alle 11 scenarioer; maksimal nettoavvik 1,14×10⁻¹³bps. Alle 16 modellers beslutninger og alle åtte armenes TRAIN/evalueringsrader rekonstrueres eksakt. SHORT er kontrollert algebraisk og i fokuserte kontrakttester; denne strategien produserte ingen faktiske SHORT-handler. Original 021-resultat er bevart;022 og den felles prediksjonsvisningen bruker 0 refits.107 fokuserte tester består etter reparasjon av kvitteringsleseren.

M1-barstempel er ikke dokumentert første-tick/filltid. Ukjent dag-/helligdagskalender og øvrige datagap består; primærmål har eksakte baner. M1-/gapdynamikk kan skjule drawdown. Dager uten kvoter viderefører siste mark; oppsamlet finansiering vises ved neste observerte mark. Sharpe er dermed betinget av denne observerte prisbanen. Ingen virkelig urørt GC-bekreftelsesperiode finnes. Native TEST, fryst baseline, Exit, produksjon og live er urørt. Ingen videre fit er innrømmet.

Frosset oppskrift: configs/research/GC_P0_FIRST_DEVELOPMENT_20261017_021.json. Resultater: /home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_P0_FIRST_DEVELOPMENT_20261017_021/RESULTS. Uavhengig kontroll: tilsvarende GC_P0_DEVELOPMENT_REVIEW_20261017_022/RESULTS. Neste eventuelle studie må ha en ny forhåndsregistrert hypotese som håndterer dårlig kalibrering, seleksjon og fire fold uten handler; flere features eller større trening er ikke begrunnet av dette forsøket.
