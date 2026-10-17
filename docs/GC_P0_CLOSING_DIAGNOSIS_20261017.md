# GX1 P0 — avsluttende diagnose, 17.10.2026

**NO-GO: stopp videre GC/M5-P0-forskning nå.** Vi har ikke funnet en handelsbar GC-edge som begrunner en ny forhåndsregistrert modelltest. Holdingregelen har et tydelig avvik fra 30-minuttersmålet, men den faste horisontkontrollen gir fortsatt negativ netto i alle armer ved sentrale kostnader. Mer modellkompleksitet er ikke begrunnet av disse dataene.

Diagnose 023 er COMPLETE, uten modelltrening, ny inferens, HPO, kalibrator, fortegnsinversjon eller valg av horisont/terskel. Den bruker de lagrede 17 301 beslutningene, de samme 17 298 målgyldige tremodellradene, originale handler og eksisterende native skalarkontekst/M1-priser. Alle 17 301 lagrede 30-minuttersmarkouts rekonstrueres eksakt fra rå bid/ask-mid. Originale 021/022-resultater er urørt. Dette er gjenbrukte utviklingsdata.

## Hvorfor prediksjonene svikter

Prognosenes standardavvik er 1,22–1,39 bps, mot markoutenes 20,50 bps. Mellom 66,26 og 75,29 prosent av prognosene er positive, mens bare 49,84 prosent av markoutene er positive. Fortegnsandelene er ikke et krav om sannsynlighetskalibrering: modellen anslår forventet markout, ikke opp-sannsynlighet. Liten prognosevarians er ikke i seg selv feil for forventede avkastninger; problemet er manglende nyttig samvariasjon og ustabile nivåanslag.

For den lagrede TRAIN-referansen c gjelder eksakt: MSE(p)−MSE(c)=E[(p−c)²]−2E[(y−c)(p−c)]. CORE tilfører 1,498 bps² prognoseenergi og har negativ alignment −0,457, slik at MSE øker 1,955 bps². FLV tilfører 2,240 med alignment −1,119; MSE øker 3,358. Alle åtte armer tilfører variasjon uten tilstrekkelig korrekt tilpasning til framtidsutfall. Dette er en algebraisk forklaring på det målte tapet, ikke en identifikasjon av feil i treningsmekanismen.

Negativ samlet kalibreringshelning skyldes negativ kovarians. For CORE er kovariansen −0,248: −0,023 innen fold og −0,225 mellom foldenes gjennomsnitt. For F og FLV ligger den negative kovariansen hovedsakelig innen fold, henholdsvis −0,790 og −0,693. Én felles nivåjustering forklarer derfor ikke alle modellene. L har svak positiv kovarians innen fold (+0,027), men negativ mellom-fold-komponent (−0,169).

| Fold | Målt 30-min snitt bps | CORE prognosesnitt | FLV prognosesnitt | CORE rank-IC | L rank-IC | FLV rank-IC |
|---|---:|---:|---:|---:|---:|---:|
| dev_01 | -0.236 | 0.438 | 1.074 | 0.0005 | 0.0097 | -0.0112 |
| dev_02 | 0.015 | -0.004 | -0.034 | -0.0455 | 0.0000 | -0.0574 |
| dev_03 | 0.966 | -0.096 | -0.225 | -0.0236 | -0.0231 | -0.0127 |
| dev_04 | 1.086 | 0.373 | 1.600 | -0.0061 | 0.0044 | -0.0130 |
| dev_05 | -0.214 | 1.741 | 1.074 | 0.0256 | 0.0263 | -0.0063 |

Samlet negativ IC betyr ikke at alle fold og regimer er negative. CORE og L har positiv IC/kalibrering i siste fold; FLV har negativ IC i alle fem. FLVs nederste prognosedesil anslår −1,286 bps og observerer +1,400; øverste anslår +3,488 og observerer −0,185. Dette gir ingen stigende rangering av attraktive utfall. Desilene er kun diagnostikk, ikke grunnlag for en omvendt strategi. TRAIN-prognoser er ikke lagret i denne studien; underfit, overfit eller en kausal modellreparasjon kan ikke avgjøres fra disse framtidige prognosene alene.

## Hvorfor bare LONG og bare mai–juni 2026

Beslutningen krever ±prognose > siste lukkede XAU-spread + 4 bps rundtur + projisert finansiering. Median spread per fold er 1,95/1,61/1,89/1,63/1,49 bps: typisk terskel om lag 5,5–6 bps. Det finnes negative prognoser i alle fold, men den høyeste SHORT-nettomarginen i hele åttearmspopulasjonen er fortsatt −0,183 bps. SHORT var tillatt og litt billigere finansiert; beslutningsregelen forbød ikke retningen.

Alle LONG-nettomarginer er negative i de første fire foldene. I siste fold øker prognosespredningen og den positive halen passerer terskelen: CORE gir 68 LONG-signaler som blir 43 innganger; FLV gir 76 signaler og 23 innganger under én-posisjon-regelen. Positiv prognoseskjevhet og kostnadskrav forklarer fortegnet/tidspunktet. Lav F/L-dekning reduserer mulighetene og endringstidspunktene, men tvinger ikke retningen til LONG. Kostnadskravet forklarer FLAT, ikke en skjult positiv edge. Ingen terskel er redusert eller valgt på nytt.

## Holdingkontroll ved nøyaktig 30 minutter

Dagens policy bruker et 30-minutters Entry-mål, men lukker ved neste kvalifiserte FLAT/flip. Median faktisk hold varierer fra 5 til 90 minutter mellom armer, og maksimum er 1 565 minutter. En FLAT-prognose etter nye rundturkostnader er ikke en egen, riktig målsatt Exit-verdi for en allerede åpen posisjon. F-armens lengste posisjon varer 1 565 minutter og har 108 kvalifiserte mellomliggende beslutninger, alle LONG. Den lange posisjonen skyldes dermed vedvarende LONG-signaler, ikke bare fravær av datadekning. Dette er en begrensning ved forskningsregelen; native Exit er ikke undersøkt eller endret.

| Arm | Dynamisk netto bps | Samme innganger, 30-min sum¹ | Én posisjon, fast 30-min netto | Antall fast 30-min |
|---|---:|---:|---:|---:|
| CORE | -99.11 | 48.96 | -8.48 | 34 |
| F | -185.95 | -91.43 | -352.75 | 46 |
| L | -470.49 | -478.40 | -385.93 | 84 |
| V | -141.44 | -49.77 | -52.25 | 26 |
| FL | -355.83 | -237.68 | -532.61 | 52 |
| FV | -499.98 | -328.13 | -404.50 | 35 |
| LV | -209.26 | -225.68 | -297.87 | 28 |
| FLV | -290.20 | -193.11 | -245.46 | 26 |

¹ Samme 288 originale innganger kan gi to samtidige posisjoner ved 30-minuttersmåling. Summen er derfor et hendelsesmarkout, ikke porteføljeavkastning eller et Sharpe-estimat. Den separate én-posisjon-varianten ignorerer mellomliggende signaler fram til planlagt slutt og bruker de samme frosne signalene/kostnadene; den har en annen, eksplisitt rapportert inngangsmengde. Begge kontroller ble bestemt før diagnoseutfallene ble lest.

COREs samme-inngang-markout er svakt positivt: +1,139 bps per inngang. Det bygger på bare fem aktive uker; familiejustert 99,375 prosent ukeblokkintervall er [−7,877;16,250]. CORE inneholder ingen nye F/L/V-felt. Alle sju GC/V-tillegg har negativt samme-inngang-markout sentralt. Den fulle én-posisjon-varianten er negativ i alle åtte armer. Den forbedrer noen og forverrer andre; ingen paired holdingkontrast er sikkert positiv etter familiejustering.

Ved billigste scenario blir CORE +60,63 bps og V +0,59, men begge er negative sentralt og ved høyere kostnad. Alle armer med de nye F- eller L-familiene er negative også ved billigste scenario. Dermed er det ingen dokumentert kostnadsrobust GC-gevinst som holding alene avdekker.

## Informasjon i GC-familiene

Dette kontrollerer informasjonen de lagrede modellprognosene faktisk utnytter, ikke all mulig informasjon i rå OFI/BBO/profiler. For et prognosetillegg d=p_arm−p_CORE er MSE-forbedringen 2E[(y−p_CORE)d]−E[d²]. Enkelte tillegg har positiv residual-alignment, men det er ikke nok til å betale for deres ekstra prognosevariasjon.

| Tillegg mot CORE | MSE-forbedring bps² | Familiejustert 99,286% intervall | Positive fold / 5 |
|---|---:|---:|---:|
| F | -1.539 | [-5.940; 1.334] | 1 |
| L | 0.237 | [-0.660; 1.091] | 4 |
| V | 0.019 | [-1.999; 2.187] | 2 |
| FL | -1.163 | [-4.715; 1.254] | 1 |
| FV | -1.124 | [-3.797; 1.663] | 0 |
| LV | -0.544 | [-2.443; 1.183] | 2 |
| FLV | -1.404 | [-4.102; 1.312] | 0 |

L er det eneste tydelige, men svake deskriptive sporet: +0,237 bps² i 4/5 fold, og rank-IC +0,0202 i middels RV. Intervallet inkluderer null, L ligger fortsatt bak TRAIN-gjennomsnittet i hvert enkelt hovedregime, og både holdingkontroller taper etter kostnader. Ingen ny strategi eller hypotese velges fra dette undergruppefunnet. FLV forbedrer ingen fold.

## Seleksjon, session og regime

Dekningen nedenfor er mot den fryste native M5-forskningspopulasjonen, ikke alle kalender- eller GC-minutter. Alle åtte armer har samme rader; seleksjon skaper derfor ikke ulikt sammenligningsutvalg mellom armene. Den begrenser tolkning utenfor kohorten og endrer sammensetningen mellom TRAIN og fold.

| Fold | F/L-rader / native rader | Dekning % | Markout-std valgt / utelatt bps |
|---|---:|---:|---:|
| dev_01 | 3107 / 16704 | 18.60 | 12.53 / 16.37 |
| dev_02 | 2040 / 16816 | 12.13 | 10.81 / 19.07 |
| dev_03 | 2605 / 15909 | 16.37 | 17.46 / 26.96 |
| dev_04 | 5556 / 16098 | 34.51 | 27.46 / 36.93 |
| dev_05 | 3993 / 10855 | 36.78 | 19.73 / 24.03 |
| initial_TRAIN | 19880 / 65424 | 30.39 | 10.72 / 15.49 |

Valgte markouts har lavere standardavvik enn utelatte i alle fold. Initial TRAIN dekker 30,39 prosent, mot 12,13–36,78 prosent i utviklingsfoldene. Faste initial-TRAIN RV-grenser og EMA-fortegn er gjenbrukt; ingen nye indikatorer eller regimegrenser.

| Session/regime | Evalueringsrader | CORE IC | L IC | FLV IC |
|---|---:|---:|---:|---:|
| session=ASIA | 7525 | -0.0188 | -0.0074 | -0.0356 |
| session=EU | 6191 | -0.0128 | 0.0045 | -0.0069 |
| session=OVERLAP | 1277 | 0.0427 | 0.0249 | 0.0019 |
| session=US | 2305 | -0.0513 | -0.0319 | -0.0033 |
| RV_tercile=0 | 2575 | -0.0215 | -0.0190 | -0.0335 |
| RV_tercile=1 | 4389 | -0.0163 | 0.0202 | -0.0132 |
| RV_tercile=2 | 10334 | -0.0188 | -0.0112 | -0.0202 |
| EMA_trend_sign=-1 | 7860 | -0.0287 | -0.0077 | -0.0271 |
| EMA_trend_sign=1 | 9438 | -0.0126 | -0.0063 | -0.0141 |

Full dekning og alle åtte modellers prediksjonsresultater per fold, session, RV, EMA og fold×session×RV finnes i RESULT.json. Undergrupper er deskriptive, med ulikt n og mange sammenligninger. De er ikke valgte strategier eller dokumenterte funn. Vi beregner ingen modellprestasjon på utelatte rader: slike lagrede prognoser finnes ikke, og manglende data kan ikke gjøres representative med vekting uten ubegrunnede antakelser. Ingen masker er svekket.

## Beslutning og ubeviste grenser

**Anbefaling: NO-GO, avslutt P0 og ikke tren mer.** Det finnes et målt samspill mellom svak prediksjon, nivåskift mellom fold, kostnadsterskel og en holdingregel som ikke følger målehorisonten. Vi har ikke påvist en konkret reparasjon som gir robust økonomisk verdi fra GC-familiene. Dette er ikke bevis for at all GC-mikrostruktur er verdiløs; det er manglende handelsbar edge i den undersøkte forskningen. Ingen inversjon, ny horisont, ny terskel eller mer modellkompleksitet foreslås som fortsettelse.

Bevist konsistent: råetiketter, felles indekser, original prognose-/handelslineage, MSE-/kovariansidentiteter og fast30 mot kanonisk v3-kontantregnskap. Målt på ekte utviklingsdata: tabellene og bootstrap. Ikke undersøkt/bevist: kausal treningsfeil, generell rå-featureinformasjon, virkelig urørt generalisering, historisk brokerfinansiering, tick-nøyaktige fills og risiko inne i manglende M1-intervaller. De tidligere data-/publikasjonsbegrensningene består. TEST, baseline, Entry/Exit og produksjon er urørt. Ingen nye datakall, kostnader eller modeller.

Kanonisk evidens: /home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_P0_CLOSING_DIAGNOSIS_20261017_023/RESULTS. Frosset diagnoseoppskrift: configs/research/GC_P0_CLOSING_DIAGNOSIS_20261017_023.json.

Kontroller: åtte nye algebra-/klokketester og tolv eksisterende P0-kontrakter består. Den fokuserte overleveringspolicytesten består også (21 tester samlet); syntaks, gamle funksjoners AST, beskyttede kilder og publikasjonskontroller verifiseres separat.
