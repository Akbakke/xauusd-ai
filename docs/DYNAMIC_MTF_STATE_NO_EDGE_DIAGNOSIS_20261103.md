# GX1: hvorfor det historiske tilstandspasset ikke slo Markov

## Beslutning

Ingen ny native modellrunde er begrunnet av de lagrede resultatene. Det ene passet lærte på TRAIN, men råmodellen slo ikke den likt informerte, fryste TRAIN-Markov-referansen samlet på senere gjenbrukte utviklingsdata for noen av de fem tidsrammene. Første-minutts skiftesannsynlighet er systematisk feilkalibrert allerede på TRAIN. Én etterfølgende, avgrenset kalibrering på lagrede TRAIN-prediksjoner ga bedre første-minutts Brier og logloss enn Markov også på gjenbrukt utvikling, men dette er ingen uavhengig edge-test og sier ikke at retningsbevegelsen ved lengre ledetider er lært. Årsaken til råmodellens feilkalibrering er ikke kausalt isolert. Økonomisk edge er ikke evaluert.

Den første revisjonen var lesning av bevarte prediksjoner, mål, checkpoints, inputbindinger og kilde. Den senere CPU-kontrollen tilpasset 15 énparameter skifteoffsets på lagret TRAIN-output og scoret disse én gang på TRAIN og gjenbrukt utvikling. Den konstruerte eller kjørte ingen native modell og gjorde ingen optimizeroppdateringer, TEST, Exit, broker eller handel. Fryst treningskilde: `13c6bb8f2f33c50806908673fa33accb5edaba63`; råmodellens ettpass-resultat og den separate kalibreringskontrollen holdes atskilt.

## Bevisstatus per mulig forklaring

| Forklaring | Hva eksisterende bevis viser | Status |
|---|---|---|
| Frakoblet data-/modellbane | Fem tidsrammer og full featureprofil er bundet; 642 aktive tensorer endret seg. GC-D1-historikkstøtten er uten overlapp mellom TRAIN og senere data. Dette måler ikke indikatorenes prediktive verdi. | Ingen frakobling påvist; konkret inputskift, kausal verdi ukjent |
| Historisk klokke/lekkasje | TRAIN-normalisering og målcutoff er bundet før senere data. XAU-M1 og ATR har bare event-time, uten faktisk mottaksklokke. | Historisk sjekk bestått; ONLINE ukjent |
| Feil mål | 0,5 ATR er kandidat. M5 +1-retning er bare 49,8 % observert; D1-FLATT er sjelden; terskelstørrelsen varierer 3,13–70,82 bps mellom tidsrammer. | Konkret mål-/databegrensning; ønsket grense ikke kvalifisert |
| Feil sannsynlighetsnivå | Råmodellen holder M5-tilstand for lenge og skifter D1 for ofte allerede på TRAIN. Én TRAIN-kalibrering forbedret +1-Brier/logloss på gjenbrukt utvikling. | Målt svakhet; årsak i treningen ikke isolert |
| Taps-/modellutforming | Sekvens-NLL vekter alle observerte minutter; første-steg-tillegg er 0,03 hver, og ingen Markov-forankring finnes. | Plausibel mekanisme, ikke kausal test |
| Manglende framtidig retningssignal | UP/NED-AUC er rundt 0,5 ved én full native tidsramme på alle fem. Første-minutts skifter kan rangeres innen kjent nåtilstand. | Kort signal, ingen dokumentert helhorisont-edge |
| Generalisering/økonomi | Råmodellen slår Markov på TRAIN for M5–H4, men taper senere; D1 taper allerede på TRAIN. Kalibratoren bruker in-sample slutt-TRAIN og tidligere brukt utvikling. | Generaliseringsreversering og D1 TRAIN-gap målt; ingen uavhengig eller økonomisk edge |

## Matchet resultat og hvor svikten oppstår

Modell og Markov ble vurdert på identiske modne, observerte målceller innen hver periode. Begge fikk samme historisk observerte nåtilstand; ved ukjent nåtilstand brukte referansen bare fryst TRAIN-prior. Modellens Brier minus Markovs Brier (lavere er bedre) viser **TRAIN-merverdi som snur på senere data** for M5–H4, mens D1 er svak allerede på TRAIN:

| Tidsramme | TRAIN, sluttmodell in-sample | Senere gjenbrukt utvikling | Beskrivende 95 % intervall for senere gap, 14 UTC-uker |
|---|---:|---:|---:|
| M5 | −0,001457 | +0,003365 | +0,001397 til +0,005417 |
| M15 | −0,002361 | +0,002943 | +0,001258 til +0,004853 |
| H1 | −0,003361 | +0,003984 | +0,001611 til +0,006771 |
| H4 | −0,004914 | +0,006860 | +0,004104 til +0,009835 |
| D1 | +0,007528 | +0,023559 | +0,005726 til +0,041588 |

TRAIN er in-sample for sluttmodellen og derfor optimistisk. Reverseringen på M5–H4 er en dokumentert generaliseringssvikt relativt til samme referanse, ikke bevis for én bestemt årsak som regimebytte eller overtilpasning. D1s gap er også en TRAIN-fit-/kalibreringssvikt. Logloss har samme fortegnsmønster: M5–H4 bedre enn Markov på TRAIN, alle fem svakere senere. Brier-gapet var positivt i hver av januar, februar og mars 2025 for hver tidsramme. Ukeblokkintervallene er beskrivende: minuttsmål og foreldre overlapper, og perioden er brukt til utvikling før. De er ikke ny, uavhengig validering.

På hele framtidsgriden er deler av generaliseringsgapet synlig som skift i **kalibrering per klasse**. Tallene nedenfor er aggregerte, sterkt overlappende observerte celler fra de signerte rapportene:

| Tidsramme | TRAIN faktisk OPP | Senere faktisk / modellens middel P(OPP) | Senere faktisk / modellens middel P(FLATT) |
|---|---:|---:|---:|
| M5 | 43,96 % | 44,34 / 43,95 % | 14,50 / 10,57 % |
| M15 | 45,40 % | 46,49 / 45,16 % | 12,97 / 10,41 % |
| H1 | 46,58 % | 50,13 / 45,62 % | 11,66 / 12,14 % |
| H4 | 51,67 % | 57,76 / 51,01 % | 7,03 / 7,11 % |
| D1 | 55,31 % | 66,62 / 52,94 % | 0,78 / 4,92 % |

M5s faktiske klasseandel er nesten uendret mellom periodene, mens modellens middel P(NED) stiger fra 42,43 til 45,48 % og P(FLATT) faller fra 12,65 til 10,57 %. M5-gapet kan altså ikke forklares av et stort skift i **bare** klassepriorene. På D1 endres fasiten tydelig mot OPP og nesten uten FLATT, mens modellens middelprognose følger ikke med. Dette viser en fordeling-/kalibreringssvikt, men skiller ikke årsaken mellom inputregime, delt modellhode, loss og utilstrekkelig fit.

Første-minutt-prognosen avslører en mer presis feil enn samlet skår. Tabellen viser sannsynlighet for **skifte bort fra den kjente nåtilstanden**, aggregert over dennes tre klasser. Tallene i parentes er faktisk skiftrate. Her brukes lagret TRAIN- og utviklingsprediksjon, ingen ny forward.

| Tidsramme / nåtilstand | TRAIN: modell / faktisk | Senere: modell / faktisk | Fryst Markov |
|---|---:|---:|---:|
| M5 NED | 0,100 / 0,230 | 0,087 / 0,237 | 0,220 |
| M5 FLATT | 0,060 / 0,316 | 0,088 / 0,328 | 0,311 |
| M5 OPP | 0,039 / 0,226 | 0,038 / 0,221 | 0,212 |
| D1 NED | 0,104 / 0,002 | 0,092 / 0,001 | 0,001 |
| D1 OPP | 0,028 / 0,001 | 0,030 / 0,001 | 0,001 |

Modellen er altså for treg til å skifte på M5 og for rask på D1 allerede på TRAIN. Det er ikke bare et sjokk i senere data. På de identiske 256 TRAIN-foreldrene gikk første-minutt-Brier fra 0,65861 til 0,52451 på M5, men referansen fikk 0,46718. D1 gikk fra 0,66433 til 0,01368; referansen var nær null i dette lille utvalget uten observerte D1-skifter. Dette viser reell tilpasning som fortsatt bommer på enkel korttidskalibrering; 256-utvalget er ikke en egen generaliseringstest.

### Kort rangeringssignal, men ingen rangering ved én hel tidsramme

Feilkalibreringen betyr ikke at alle lagrede skårer mangler informasjon. Innen **samme kjente nåtilstand** rangerte modellens første-minutts skiftesannsynlighet senere skifter over ikke-skifter: M5 NED AUC 0,678 (684 skifter / 2 883 par), M5 OPP 0,649 (692 / 3 135), M15 NED 0,731 og M15 OPP 0,709. En Markov-sannsynlighet er konstant innen hver slik gruppe og gir AUC 0,5. En selvstendig `sklearn`-beregning gjenskapte tallene; M5/M15-rangeringen var positiv i hver av januar, februar og mars 2025. D1 har bare 7 NED- og 12 OPP-skifter på senere første-minuttspar, så høy D1-AUC er usikker.

TRAIN-definerte desiler viser at nivåfeilen ikke bare kommer fra regimeskiftet i senere data. For M5 med kjent NED ga laveste desil gjennomsnittlig skiftesannsynlighet 0,046 mot 0,090 faktisk på TRAIN og 0,045 mot 0,106 senere. Høyeste desil ga 0,186 mot 0,479 på TRAIN og 0,179 mot 0,524 senere. M5 OPP viser samme kompresjon: høyeste desil 0,080 mot 0,451 på TRAIN og 0,078 mot 0,450 senere. M15 NED har en annen form: laveste desil overpredikerer svakt, mens høyeste underpredikerer. Dette motiverte den senere, separat bundne kalibreringskontrollen nedenfor; på dette tidspunktet var ingen kalibrator tilpasset eller scoret.

Dette er et **historisk, kortvarig rangeringssignal**, ikke en kalibrert sannsynlighet eller handelsedge. På samme mål har modellens første-minutts binære skifte-Brier 0,211172 mot Markovs 0,182923 på M5, og 0,005828 mot 0,001117 på D1. De store rå AUC-tallene for faktisk tilstand ved +1 skyldes i stor grad at nåtilstanden ofte fortsetter; modellen slår Markov litt i slik rangering, men svikten ved lengre ledetider gjenstår.

På de nøyaktig samme observerte, ikke-FLATE senere målene forsvinner UP-mot-DOWN-rangeringen når ledetiden når omtrent én hel native tidsramme. AUC er en rangmåling; den erstatter ikke Brier eller logloss.

En egen lesning av de lagrede **slutt-TRAIN**- og senere prognosene på én full native tidsramme (M5 +5, M15 +15, H1 +60, H4 +240, D1 +1 440 min) skiller manglende variasjon fra manglende korrekt rangering. In-sample TRAIN UP/NED-AUC er henholdsvis 0,499/0,508/0,537/0,563/0,663; senere er den 0,498/0,490/0,487/0,487/0,484. Senere standardavvik for modellens betingede `P(OPP)/(P(OPP)+P(NED))` på samme observerte ikke-FLATT-rader er 0,070/0,041/0,034/0,036/0,033. Skårene varierer altså, men varierer **ikke i riktig retningsrekkefølge** senere. M5s **samlede** rangering ved +5 er nær null allerede in-sample; svak rangering innen kjente nåtilstander er kontrollert separat nedenfor. H1/H4/D1s in-sample-rangering overlever ikke senere. På senere data er middel skår for faktisk OPP svakt *lavere* enn for faktisk NED på alle fem (D1 0,5370 mot 0,5392). Dette er en beskrivende TRAIN-/gjenbrukt utviklingssammenligning med overlappende mål, ikke en årsaks- eller edge-test. [Kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_LEAD_DIRECTION_SCORE_DISTRIBUTION.json) SHA-256 `5e502219f354d69f76e25435bde8947b502ba52cfdb95870899a57f0fd60b179`.

Månedsdeling bekrefter at kontrasten ikke bare skyldes én pooled beregning: D1s in-sample TRAIN-AUC ved +1 440 min var 0,577–0,705 i hver av de åtte TRAIN-månedene; senere januar/februar/mars var 0,522/0,497/0,414. Den tydelige marsinversjonen finnes **innen** måneden. M5 ved +5 min lå 0,485–0,507 i alle åtte TRAIN-måneder og 0,499/0,501/0,490 senere. Dette er fortsatt overlappende, gjenbrukt historikk, og månedsmønsteret skiller ikke tidsdrift, markedsregime, inputskift eller målskift. [Månedskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_LEAD_MONTHLY_DIRECTION_AUC.json) SHA-256 `479a506e958ac75ccbdef905748953d1c3b5f127d0ee632400739c734f42c413`.

En kontroll **innen historisk nåtilstand** viser at D1s in-sample-rangering ikke bare kommer fra å skille nåværende NED fra OPP. På TRAIN var D1 +1 440 AUC 0,653 blant 12 119 rader med kjent NED nå og 0,671 blant 14 763 med kjent OPP nå; på senere utvikling var begge 0,495 (4 365 og 8 469 rader). En Markov-skår er konstant innen hver slik gruppe og har AUC 0,5 der. M5 +5 hadde svak in-sample AUC 0,527–0,532 innen de tre kjente nåtilstandene, selv om samlet AUC var 0,499; senere er de 0,493/0,520/0,508. Enkle stigende skårer for **dato** og antall observerte GC-D1-rader hadde pooled TRAIN D1-AUC 0,5039 og 0,5040, mot modellens 0,6633. Modellskåren hadde Spearman-korrelasjon omtrent −0,009 med begge. Dermed er D1s TRAIN-rangering verken bare nåtilstandsmiks eller en monoton kalender-/GC-dekningsskår. Det sier ikke hvilke indikatorer som bar den, og in-sample-rangeringen kan være overtilpasning. På senere utvikling kollapser den betingede rangeringen; årsaken er fortsatt ikke isolert. [Betinget kontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_STATE_CONDITIONAL_NATIVE_RANK_REVIEW.json) SHA-256 `13aa797e6b606217d10d3f86baa16b3dc48a6b034089a40b3febeccbf0a248db`.

Skårgeometrien viser en mer konkret **mål–prognose-konflikt ved én full tidsramme**. For M5 +5, blant observerte ikke-FLATT-mål, er faktisk OPP-andel etter kjent NED/OPP nå 0,524/0,507 på TRAIN og 0,521/0,518 senere. Modellens middel betingede OPP-skår er derimot 0,437/0,605 og 0,413/0,589. Den skiller sterkt etter nåtilstand selv når den påfølgende retningsandelen er nesten lik. Markov viser også nåtilstandsavhengighet (senere 0,453/0,570), men svakere. Dette forklarer hvorfor svak positiv rangering **innen** noen nåtilstander ikke gir god samlet M5-rangering ved +5; det beviser ikke at nåtilstandsprojeksjonen alene er feilen.

D1 +1 440 viser motsatt rekkefølge mellom fasit og modellens middel på de samme ikke-FLATT-radene: senere er faktisk OPP-andel etter kjent NED/OPP 0,719/0,644, mens modellskåren er 0,534/0,540. På TRAIN er tilsvarende 0,606/0,560 mot 0,554/0,568. Den sterke in-sample-rangeringen **innen** D1-nåtilstand kunne delvis veie opp for denne feilrekkefølgen; senere er den borte. Dette er en deskriptiv forklaring av AUC-fallet, ikke en isolert modell- eller datarotårsak.

Brier per ledeminutt avgrenser også D1s TRAIN-svikt: modellen var dårligere enn Markov ved +1 til +563 min og bedre ved +564 til +1 440 min på slutt-TRAIN in-sample. Ved +1 440 var gapet −0,013175, selv om samlet fullgrid-gap på TRAIN var positivt. På senere utvikling var D1-gapet positivt ved **alle 1 440** ledeminutter; ved +1 440 var det +0,021423. M5 +5-gapet var +0,000796 på TRAIN og +0,004838 senere. Disse er treklasse-Brier på alle observerte mål, adskilt fra den ikke-FLATT-betingede retningsanalysen over. [Gruppe- og horisontkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_STATE_GROUP_HORIZON_SCORE_DECOMPOSITION.json) SHA-256 `62e16783993c640b14b4c0b0f1df271af6837976d90f591caf1b9c472fd9efb2`.

En eksakt **parvis AUC-dekomponering** bekrefter at gruppeforskjellene har praktisk betydning for samlet rangering. For M5 +5 på TRAIN er AUC 0,5142 blant OPP/NED-par med **samme** nåtilstand, men 0,4927 mellom **ulike** nåtilstander; 70,2 % av sammenlignbare par er mellom grupper, slik at pooled AUC blir 0,4991. Senere er innen/mellom 0,4944/0,4995 og pooled 0,4979. For D1 +1 440 er TRAIN innen/mellom 0,6660/0,6615, senere 0,4951/0,4713; pooled faller fra 0,6633 til 0,4841. Det viser at D1 mister rangering også innen samme nåtilstand, mens en feilrekkefølge mellom nåtilstander trekker ytterligere ned. Par deler overlappende mål og er ikke uavhengige hendelser. [Parvis kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_AUC_WITHIN_BETWEEN_STATE_REVIEW.json) SHA-256 `4acc526d424cdd957d279e59efee949762aa7a1bf4142c252b54f9e6f0679d4b`.

Den allerede fryste, **nåtilstandsfrie TRAIN-klasseprioren** gir en skarp kontroll av hvor lenge nåtilstand hjelper de målte prognosene. På senere M5 slår Markov prioren bare ved +1–3 min; fra +4 til +1 440 er priorens Brier lavere. Modellen slår prioren bare ved +1 og +2, der Markov likevel slår modellen. Den slår derfor **ikke begge** forhåndsdefinerte referanser ved noen M5-ledetid. Ved +4 og +5 er prioren best i **hver** av januar, februar og mars, kontrollert på lagrede rader med pooled paritet til signert rapport. Dette viser at M5-prognosen beholder en ugunstig nåtilstandsavhengighet lenger enn de to referansene tilsier i denne perioden; det beviser ikke at nåtilstand er uten informasjon i markedet.

| TF, én full native ledetid | Senere Brier modell / Markov / TRAIN-prior | Senere ledeminutter der modellen slår **begge** referanser |
|---|---:|---:|
| M5 +5 | 0,612158 / 0,607321 / 0,604858 | 0 / 1 440 |
| M15 +15 | 0,596087 / 0,592204 / 0,597435 | 5 / 1 440 |
| H1 +60 | 0,578103 / 0,574463 / 0,579435 | 18 / 1 440 |
| H4 +240 | 0,545288 / 0,538942 / 0,538705 | 36 / 1 440 |
| D1 +1 440 | 0,490322 / 0,468899 / 0,466864 | 0 / 1 440 |

En lesende oppdeling på **samme observerte rader ved én full native ledetid** korrigerer en mulig feiltolkning av M5-krysset: dagens FLATT-tilstand inneholder fortsatt informasjon om senere FLATT, særlig på M15 og H1. Tallene under er faktisk FLATT-andel etter FLATT nå, sammenlignet med etter kjent NED/OPP nå; modell og Markov er gjennomsnittlig `P(FLATT)` når nå er FLATT.

| TF / ledetid | FLATT nå, n | Faktisk FLATT etter FLATT nå / etter NED eller OPP nå | Modell / Markov `P(FLATT)` etter FLATT nå |
|---|---:|---:|---:|
| M5 +5 | 816 | 25,3 / 14,1 % | 17,9 / 24,7 % |
| M15 +15 | 1 626 | 34,9 / 10,2 % | 17,6 / 26,9 % |
| H1 +60 | 1 599 | 27,8 / 7,6 % | 17,7 / 27,5 % |
| H4 +240 | 1 049 | 14,2 / 5,9 % | 7,5 / 14,7 % |

M5, M15 og H1 har høyere faktisk FLATT-andel fra FLATT nå i hver av januar, februar og mars. H4 varierer mer mellom månedene. D1 har bare 119 observerte rader med FLATT nå ved +1 440 og ingen FLATT-fasit i disse; det gir ikke en nyttig FLATT-konklusjon. Markov fanger M5/H1/H4 FLATT-gruppen bedre i nivå, mens modellen legger for lav FLATT-sannsynlighet på disse radene. **Samlet FLATT-Brier** er likevel litt bedre for modellen enn Markov på M5/H1/H4, fordi også de øvrige nåtilstandene teller. På M5 +5 er modellens Brier-gap mot Markov fordelt på NED/FLATT/OPP med +0,003572/−0,000727/+0,001993. Markov taper mot prioren på M5 +5 hovedsakelig i NED og OPP (+0,001266/−0,000121/+0,001318). Dermed er horisontkrysset en svakhet i de **samlede prognosene**, særlig retningskomponentene, og kan ikke brukes som bevis for at all nåtilstand bør glemmes etter +3 minutter. Fasitvinduene overlapper, og historisk event-time er ikke bevist som faktisk mottakstid. [Oppdelingskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_FLAT_CONDITION_AND_BRIER_COMPONENTS.json) SHA-256 `211f51239fb3504e1e1ef5db55c23c06385f8b72d2842eef4ecdb1a993bac676`; null nye modellkall eller fit.

Den **tidsavhengige FLATT-kurven** viser mer enn nivåfeil. Tabellen gjelder bare rader med kjent FLATT nå og observerbar framtidsklasse; hvert felt er faktisk andel / modellens middel `P(FLATT)`. TRAIN er sluttmodellens in-sample-skår, senere er gjenbrukt utvikling.

| TF | TRAIN +1 / én native ledetid | Senere +1 / én native ledetid |
|---|---:|---:|
| M5 +5 | 68,4/94,0 % → 27,6/24,7 % | 67,2/91,2 % → 25,3/17,9 % |
| M15 +15 | 88,5/94,4 % → 40,7/25,2 % | 87,8/92,1 % → 34,9/17,6 % |
| H1 +60 | 96,7/94,0 % → 35,5/25,0 % | 96,2/92,0 % → 27,8/17,7 % |
| H4 +240 | 98,7/90,4 % → 13,3/11,2 % | 98,8/89,0 % → 14,2/7,5 % |

M5 og M15 går fra for høy modellert FLATT-andel ved +1 til for lav ved én native ledetid, **allerede på TRAIN**. På senere M5 er samme fortegnsskifte 67,8/91,4 % ved +1 og 27,4/17,9 % ved +5 når **nøyaktig de samme 751 issue-radene** må være observerbare ved begge ledetider; M15 har samme mønster på 1 620 rader. H1 og H4 ligger gjennomgående for lavt ved de kontrollerte ledetidene. Dette er et målt misforhold mellom modellens marginale FLATT-tilstandskurve og fasit, ikke en påvist kodefeil eller bestemt parameterårsak. En kalibrator som bare justerer +1 kan ikke alene rette fortegnsskiftet. D1s sjeldne FLATT-utfall gir ikke tilsvarende sikker kurvediagnose. [Kurvekvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_FLAT_PERSISTENCE_CURVE_TRAIN_LATER.json) SHA-256 `efeadf38f2822b5701d0b70a97343a7bb82380cc793108f4a26a8ea053888c48`; ingen nye modellkall eller fit.

**FLATT ved en senere ledetid er tilstandsandel, ikke overlevelse uten avbrudd.** I lagrede senere mål startet og sluttet 206 M5-, 567 M15- og 444 H1-rader med FLATT på én native ledetid. På komplette, minuttvise observerte baner hadde henholdsvis 47/206 (22,8 %), 177/567 (31,2 %) og 159/444 (35,8 %) vært OPP eller NED underveis og kommet tilbake til FLATT. TRAIN viser samme type utgang/retur: 215/717, 597/2 042 og 545/1 921. For H4 er bare 76 av 149 senere FLATT→FLATT-baner komplette over +240 minutter; 11 av disse 76 gikk via OPP/NED. D1 hadde ingen slik +1 440-bane. Kurven over må derfor ikke tolkes som varighet til **første** utgang. Den eksisterende marginalscoringen tester slutt-tilstand ved hvert minutt, men den lagrede modellen gir ikke en separat første-utgang-/returprognose å sammenligne her. [Målbanekvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_FLAT_EXIT_REENTRY_TARGET_PATHS.json) SHA-256 `d44c2231dd891e9de953b65143f0a987d254cc98d3615c9c09f3385821c5be02`; null modellkall, fit eller TEST.

FLATT nå kan altså hjelpe **prognosen for senere FLATT-tilstand** uten å fortelle hvilken retning som kommer etterpå. En separat rangkontroll på observerte framtidige OPP/NED-mål med FLATT nå gir følgende betinget OPP-AUC ved én native ledetid. Dette er et etterpåvalgt utvalg for diagnose, **ikke** en inngangsregel: framtidig ikke-FLATT er ukjent ved issue.

| TF / ledetid | Slutt-TRAIN AUC, in-sample | Senere AUC, antall rader / kalenderdager |
|---|---:|---:|
| M5 +5 | 0,530 | 0,520; 610 / 69 |
| M15 +15 | 0,532 | 0,511; 1 059 / 68 |
| H1 +60 | 0,527 | 0,399; 1 155 / 67 |
| H4 +240 | 0,600 | 0,390; 900 / 45 |

H1s senere måneds-AUC er 0,350/0,471/0,370 for januar/februar/mars; H4 er 0,248/0,271/0,590. H4 varierer derfor kraftig. Beregningen er kontrollert både med `sklearn` og eksakt rangsumformel. D1 har 119 senere ikke-FLATT-rader etter FLATT nå, alle OPP og fordelt på bare fire dager; AUC er udefinert. Tallene viser ingen stabil, senere retningsrangering fra FLATT nå, særlig ikke på H1/H4. Radene har sterkt overlappende fasit, utviklingen er gjenbrukt og dette er ikke en uavhengig invers handelshypotese. [Retningskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CURRENT_FLAT_NATIVE_DIRECTION_RANK_TRAIN_LATER.json) SHA-256 `d20f06db650329f34e36cbd553ae685072959059bb1652d0acfb8372d84bd174`; ingen nye modellkall eller fit.

Minuttspørringene overlapper sterkt. «Slår begge» er en beskrivende snittmengde, **ikke** en operativ referanse som velger beste kurve etter å ha sett utviklingsperioden. [Horisontkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_MARKOV_PRIOR_HORIZON_CROSSOVER.json) SHA-256 `edd8d81b1dfe32c9cba22193937751975b23d969e53a97a94ca391ad0bf5ebeb`.

| Tidsramme / ledetid | Modell UP/NED AUC | Markov UP/NED AUC | Modell / Markov Brier, tre klasser |
|---|---:|---:|---:|
| M5 / +5 min | 0,498 | 0,500 | 0,6122 / 0,6073 |
| M15 / +15 min | 0,490 | 0,484 | 0,5961 / 0,5922 |
| H1 / +60 min | 0,487 | 0,489 | 0,5781 / 0,5745 |
| H4 / +240 min | 0,487 | 0,497 | 0,5453 / 0,5389 |
| D1 / +1 440 min | 0,484 | 0,451 | 0,4903 / 0,4689 |

FLAT-sannsynligheten kan fortsatt rangere noe ved enkelte korte ledetider selv når FLAT aldri er modal klasse, for eksempel M15 +15 AUC 0,644 mot Markov 0,589. Den er dårlig kalibrert i samlet skår. Ingen av disse tallene viser at modellen forutser retningsbevegelsen gjennom hele tidsrammen.

På den **etterpå valgte** delmengden der kjent FLAT faktisk ble OPP eller NED ved +1, rangerte modellen retningen (M5 AUC 0,806 på 337 overganger; M15 0,910 på 216). En enkel skår fra siste avsluttede M1-lys, `close − open`, fikk M5 AUC 0,840 og M15 0,874 på nøyaktig samme rader. M1-filen ble hashet på nytt, og alle issue-tider traff eksakt forrige M1-bar. Denne referansen er historisk event-time, og utvalget krever kunnskap om et **fremtidig** skifte; tallene kan ikke brukes som operativt seleksjons- eller merverdibevis.

Den fryste Markov-referansen slo også modellens **binære skiftesannsynlighet** på matchede fremtidige nabopar: M5 0,177985 mot modell 0,178109 i Brier, D1 0,001306 mot 0,001313, og tilsvarende små negative gap på M15/H1/H4. Modellen hadde null eksakte *modale* skiftevarsler, men også en optimal sannsynlighetsmodell kan velge «ingen skifte» når ett-minutts skifter er sjeldne. Null modal treff er derfor ikke alene forklaringen; skår mot samme-informasjonsreferansen er avgjørende.

På senere data er modellens FLAT-Brier klart svakere enn referansens på M5/M15/H1/H4, mens DOWN-celler er bedre og UP-celler svakere. Modellens modal FLAT-recall er nær null over hele banen. Ved +5 minutter er M5-recall null, ved +15 M15 null, ved +60 H1 null, og ved +240 er den null på alle fem. M5s gjennomsnittlige FLAT-sannsynlighet ved +15 er omtrent 0,104 både når fasit er FLAT og ellers. Det viser dårlig separasjon av rolig tilstand utover første steg. M5-gapet mot Markov er positivt både når nåtilstanden er kjent (+0,003742 Brier) og ukjent (+0,002957); uobserverbar M5-retning er derfor ikke en tilstrekkelig forklaring alene.

## Én lagret første-minutts kalibreringskontroll

Etter trygg PC-omstart 599→600 ble én CPU-kontroll gjennomført på de **allerede lagrede** sluttprediksjonene. Det første forsøket ble lukket før scoring: lagret FP32 avrunding av sannsynlighetsrader var inntil 1,44×10⁻⁷, mens kontrollen krevde 10⁻⁸. Den separate erstatningen normaliserte radene i float64 og bestod. Ingen native modellforward eller optimizeroppdatering fant sted. Begge engangsplaner er konsumert og trening er stengt.

Kalibratoren passer én logitforskyvning for skifte ved +1 minutt per tidsramme og kjent nåtilstand (15 tall) på 47 602 TRAIN-foreldre. Den bevarer råmodellens betingede fordeling mellom de to andre klassene ved et skifte. Ved ukjent nåtilstand lar den råprediksjonen stå. Logitforskyvningen er monoton innen hver tidsramme/nåtilstand, så den kan ikke skape den målte innen-gruppe skifte-AUC-en; den retter sannsynlighetsnivået til et allerede lagret rangeringssignal. Én scoring på 17 288 **gjenbrukte** utviklingsforeldre ga følgende matchede treklasse-Brier og logloss; alle sammenligninger bruker identiske observerte målceller og samme historiske nåtilstand:

| Tidsramme | Observerte +1 / foreldre | Brier rå / kalibrert / Markov | Logloss rå / kalibrert / Markov |
|---|---:|---:|---:|
| M5 | 8 613 / 17 288 | 0,460854 / 0,413951 / 0,425444 | 0,853668 / 0,710064 / 0,744901 |
| M15 | 14 614 / 17 288 | 0,210824 / 0,204065 / 0,214564 | 0,406527 / 0,381453 / 0,421248 |
| H1 | 16 894 / 17 288 | 0,072037 / 0,070374 / 0,072472 | 0,166359 / 0,157088 / 0,173128 |
| H4 | 17 144 / 17 288 | 0,026345 / 0,019385 / 0,019559 | 0,088540 / 0,053383 / 0,058679 |
| D1 | 16 987 / 17 288 | 0,010440 / 0,002237 / 0,002257 | 0,058513 / 0,008635 / 0,009160 |

Ved +1 er **tilstandsskifte** sjeldent på lange tidsrammer. Blant senere rader med kjent nåtilstand og observert +1-fasit skiftet M5 i 1 713 / 7 046 (24,3 %), M15 i 1 577 / 14 096 (11,2 %), H1 i 618 / 16 839 (3,7 %), H4 i 168 / 17 140 (1,0 %) og D1 i bare 19 / 16 986 (0,11 %). Uendret tilstand betyr ikke at prisen sto stille; det betyr at det lokale tilstandsmerket ikke skiftet. D1s svært lave +1-Brier må derfor ikke leses som evne til å varsle en ny svingbevegelse.

Den største absolutte Brier-gevinsten mot Markov er ved M5 (+1): 0,011493. Ved D1 er den bare 0,000020. Kalibreringen gjenfinner særlig nivået på skiftesannsynligheten; for eksempel steg M5 NED fra gjennomsnittlig 0,100 til observert 0,230 på TRAIN, mens D1 NED falt fra 0,104 til 0,002. Dette peker mot en konkret sannsynlighetsskala-/base-rate-svikt i råoutput, ikke mot at indikatorene nødvendigvis er ubrukelige.

**Begrensning:** Kalibratoren ble tilpasset prediksjoner fra en modell som selv var trent på hele TRAIN. Utviklingsperioden er tidligere brukt til design. Det finnes ingen kronologisk OOF-kalibrering eller urørt bekreftelse. Resultatet gjelder bare første minutt, og M5s retningsklasse er observert i omtrent halvparten av foreldrerekkene. UP/NED-rangering ved +5/+15/+60/+240/+1 440 minutter forblir rundt 0,5, og FLAT-dekning/selektivitet er ikke løst. Denne kontrollen kvalifiserer verken løpende markedsprognose, Entry eller økonomisk edge.

## Hvorfor lagrede data ikke gir kronologisk kalibreringsbevis

Den eksakte kampanjen har bare tre prediksjonssett: 256 TRAIN-foreldre **før** fit, 47 602 TRAIN-foreldre **etter** hele fit og 17 288 senere utviklingsforeldre. Elleve mellomliggende checkpoints lagrer vekter og TRAIN-cursor, men ingen prediksjoner. Dermed finnes ingen lagrede TRAIN-prognoser som både er fra den trente modellidentiteten og er laget før modellens fit på de samme radene. [Inventarkvitteringen](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_SAME_MODEL_OOF_AVAILABILITY.json) avgrenser påstanden til denne kampanjen.

En fremtidig replay fra mellomliggende checkpoint ville heller ikke automatisk være streng kronologisk OOF: kampanjens inputmanifest binder én normalisering kvalifisert på **alle 47 602 TRAIN-foreldre** før disse checkpointene ble laget. En ekte OOF-protokoll må tilpasse normalisering, modell og kalibrator på bare tidligere TRAIN-prefikser og score senere, modne TRAIN-rader uten refit på dem. Å dele opp den allerede gjenbrukte utviklingsperioden ville ikke gjøre den urørt. Dette er et manglende bevis, ikke en oppfordring til flere løp nå.

## Mål, datatilgang og klokke

Målet følger kausale lokale opp- og nedbevegelser per M5/M15/H1/H4/D1, inkludert reverseringer og FLAT; det bruker ingen fast startpris, og 1–1 440 minutter er målt spørregrid, ikke en handelsgrense. Den bruker brukerens godkjente **kandidatterskel** 0,5 ATR(14) per tidsramme; det lagrede targetmanifestet har `target_rule_locked=false`. Kildeinspeksjon fant ingen byttet tidsrammerekkefølge: target, historisk nåtilstand, modellkoordinater og rapport bruker samme M5–D1-orden. Nåtilstand bindes til eksakt foreldrerad og issue-tid, og fremtidige mål sendes til loss, ikke til forward. Dette er kode- og lagringsbevis, ikke bevis for faktisk ONLINE-tilgjengelighet. Den trente forsøksgriden dekker nøyaktig +1 til +1 440 minutter; det er ingen maksimal tilstands- eller holdetid, men svingbevegelser etter ett døgn er ennå ikke trent eller målt.

M5s mål er selektivt uobserverbart: av 17 288 senere første-minuttsrader var 8 613 retningsklasser observerte (49,8 %). Av de 8 675 ukjente hadde 8 669 flagg for ukjent intraminutt-rekkefølge; flaggene overlapper. Den binære FLAT/ikke-FLAT-aktiviteten kunne likevel avgjøres i 17 284 komplette lokale vinduer. D1 hadde 16 987 observerte første tilstander (98,3 %), men bare 125 FLAT blant 16 136 komplette lokale vinduer (0,8 %). Derfor er M5-retningslæring sterkt sensurert nettopp ved tvetydige M1-barer, mens D1s FLAT-klasse nesten forsvinner i senere data. Ingen rekkefølge inne i et M1-lys skal gjettes for å fylle hullene. En avgrenset lokal filinventering fant bare én time XAU-ticks i utviklingsåret 2025 i Dukascopy-cachen og fire løse testfiler. Dette er ikke de OANDA-tickene som skapte de brukte M1-lysene, og kan ikke reparere TRAIN-/utviklingsmålene uten kildeparitet og full dekning. Inventeringen utelukker ikke andre, ennå uundersøkte kilder.

En ren TRAIN-kontroll av komplette lokale første-minuttsvinduer viser at definisjonen av FLAT er svært terskelfølsom på **alle fem** tidsrammer:

| Tidsramme | Komplette TRAIN-vinduer | Lokal ro ved 0,5 ATR | Hypotetisk ved 0,75 ATR |
|---|---:|---:|---:|
| M5 | 47 561 | 7,7 % | 32,6 % |
| M15 | 47 473 | 11,6 % | 36,0 % |
| H1 | 47 115 | 13,7 % | 38,9 % |
| H4 | 45 817 | 9,3 % | 34,9 % |
| D1 | 37 390 | 3,7 % | 23,8 % |

Dette er bare lokal rekkevidde i første minutt; opp-/nedtilstandene og hele tilstandsbanen må rematerialiseres kausalt for en annen terskel. Tallene godkjenner ingen terskelendring og bruker ikke senere data til å velge en.

Den tidligere, hashbundne TRAIN-målrevisjonen måler også faktisk kandidatterskel ved prognosetid og klassefordelingen over **alle observerte framtidsminutter** i forsøksgriden:

| Tidsramme | Median 0,5 ATR ved TRAIN-issue | NED / FLATT / OPP blant observerte framtidsceller |
|---|---:|---:|
| M5 | 3,13 bps | 41,0 / 15,0 / 44,0 % |
| M15 | 5,71 bps | 40,5 / 14,1 / 45,4 % |
| H1 | 12,17 bps | 39,2 / 14,2 / 46,6 % |
| H4 | 25,26 bps | 39,0 / 9,3 / 51,7 % |
| D1 | 70,82 bps | 41,2 / 3,5 / 55,3 % |

Disse cellene overlapper sterkt og er ikke uavhengige utfall. Tabellen viser at «0,5 ATR» representerer svært ulike absolutte bevegelsesstørrelser per tidsramme. Særlig M5s 3,13 bps er et konkret mål å vurdere mot brukerens ønskede betydning av bevegelse; den er ikke i seg selv bevis på en handelsmulighet eller en feil terskel. Målrevisjonen låste ikke kandidatregelen (`target_rule_locked=false`).

Den bundne normaliseringskvalifiseringen bruker 47 602 TRAIN-foreldre med siste beslutning 31.12.2024 kl. 21:55 UTC, før TRAIN-grensen 01.01.2025 kl. 00:00 UTC; den utførte null nye statistikkfit. Dette viser ingen innblanding av senere utviklingsrader i den kvalifiserte normaliseringen (kvalifisering SHA-256 `51ac757b4c1709d1104a750937ebb07bded36fe890089d436e298e801139d1fc`).

Historisk event-time-sjekk av siste M1-lys og ATR er bestått. Faktisk fysisk mottaks-, prosesserings- og restarttid for disse verdiene er fremdeles ukjent; dette stenger ONLINE-bruk, men er ikke påvist som årsak til det historiske Brier-gapet. Den bundne XAU-M1-filen har 5 959 045 rader og bare `time`, OHLC, bid/ask OHLC og `volume`; ingen mottaks- eller ferdigberegningstid. Kildemanifestet sporer OANDAs historiske `/instruments/XAU_USD/candles` med `complete=true`, M1 barstart og `decision_available_offset_seconds=60`: det er en event-time-regel om når lyset er lukket, ikke en målt ankomstklokke. `M5_ENRICHED.parquet` har tids-/ATR-felt og mange GC-klokkefelter, men disse dokumenterer ikke mottakstid for XAU-M1 eller XAU-ATR. Historiske filer alene kan derfor ikke bevise hva ONLINE-modellen faktisk ville ha mottatt ved issue-tid.

## Modellkonsum og loss

Inputmanifester binder full featureprofil, åtte spesialistfamilier, M5/M15/H1/H4/D1 for XAU og GC samt D1/makro-kontekst. TRAIN-normaliseringskvalifiseringen har ingen blokkeringer og minst én observasjon per felt i alle 12 kontrollerte flater. Den viser også sparsomhet: GC-D1 har median observert feltdekning 40,7 %, og 15 av 399 felt er konstante på TRAIN; 72 av 2 542 `ctx_cont`-felt er konstante. Fullprofilens syntetiske kontrakttest krever at hvert fysisk felt er rutet til nøyaktig én familie. Alle 642 aktive parametertensorer endret seg under passet, også spesialist-/MTF-projeksjoner, nåtilstandsprojeksjon og tilstandshoder. Dette viser tilkoblet modellbane, ikke at hver indikator gir prediktiv verdi eller at OOF-featurefordelingen er stabil.

En tidligere fysisk fullprofilrevisjon er gjenbrukt **bare etter identitetskontroll** mot den aktuelle kampanjens inputmanifest: akser, fold, fysisk provider, konstruktør og prefixnormalisering er like; modellens retningskomponent-kilde og kontrakthash er forskjellige. Den målte GC-D1-historikkdekkingen er helt uten overlapp: TRAIN har 0–168 observerte volumhistorikkrader per forelder (snitt 82,89), senere utvikling 169–230 (snitt 199,39). Revisjonen registrerte også 17 sparsomme felt som var observert på TRAIN, men aldri senere. Dette er en **konkret inputstøtteforskyvning** som kan gjøre generalisering vanskeligere. Tid, regime og historikkmengde endres samtidig; ingen lagret prediksjon kan isolere effekten, og resultatet rettferdiggjør verken å fjerne indikatorer eller å kalle dem verdifulle.

Den fryste modellkilden er identisk med kildehashene i fitoppskriften (`entry_v10_ctx_hybrid_transformer.py` `7808095f…`, `entry_native_direction_research_v1.py` `05c0633f…`). GC-felter normaliseres med observasjonsmasker og legges til XAU-familietokenene før transformerblokken; D1-avlesningen er `mean` over **alle 252** kodede posisjoner pluss siste posisjon (modellkilden linje 361–373 og 2552–2573). Det finnes ingen GC-spesifikk nevner basert på antall observerte historikkrader i denne avlesningen. Dermed møter den trente representasjonen en faktisk annen maskestruktur og mulig annen tokenfordeling senere. XAU-data, posisjon og maskefelter er fortsatt til stede, og vektene kan kompensere; dette er en plausibel mekanisme, ikke bevis for at GC-skiftet forårsaket Brier-gapet eller at avlesningen er feil.

Et nytt **lesende** replay av lagret sluttmodell, mål og fryst Markov gjenskapte D1s samlede Brier-gap på TRAIN (+0,007528) og gjenbrukt utvikling (+0,023559) til seks desimaler. Innen senere utvikling var gapet positivt i fire kvartiler etter antall observerte GC-D1-volumhistorikkrader: 169–183: +0,05653; 184–198: +0,02115; 199–214: +0,00451; 215–230: +0,01355. Det **øker ikke monotont** med mer GC-historikk. Januar/februar/mars hadde henholdsvis +0,04816/+0,00797/+0,01287. Dette svekker en enkel forklaring om at økende historikkmengde alene skaper gapet, men kvartilene er beskrivende og valgt etter målt støtte; historiealder og markedsregime følger kalenderen. Ingen kausal effekt eller uavhengig prognoseverdi er isolert. [Kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_GC_D1_SUPPORT_SCORE_REVIEW.json) SHA-256 `b52859791a9ab84509958c612d4a260694728458394965d3361eb1911e36a2b8`.

Loss er normalisert sekvens-NLL over alle observerte minutter og fem tidsrammer. Dermed fikk M5 bare 12,7 % av de observerte TRAIN-cellene, mens H1 og H4 fikk 23,1 % hver; dette følger hovedsakelig M5s svakere målobservasjon og er ikke en påvist kausal årsak. Første tilstand og første FLAT-aktivitet får egne tillegg med vekt 0,03 hver. Første tilstand kommer fra eget hode; nåtilstanden legges til skjult representasjon via en lært projeksjon. Det finnes ingen eksplisitt identitets- eller Markov-forankring som garanterer at enkel fortsettelse bevares. Felles hoder må lære skiftrater som er svært forskjellige på M5 og D1. For fullstendig observerte baner scorer loss overgangene gitt faktisk forrige tilstand; ved ukjente tilstander marginaliserer den disse. Prognosene bruker fremoverrekursjon over modellens egne sannsynligheter. Små overgangsfeil kan dermed akkumuleres gjennom 1 440 steg. Dette er ikke i seg selv en kodefeil. De komplette modellovergangsmatrisene ble ikke bevart, så denne mekanismen kan ikke isoleres fra det lagrede materialet. Dette er en plausibel mekanisme for målt feilkalibrering, men den er **ikke kausalt isolert**; vektendring eller flere pass kan ikke kalles en påvist løsning.

Kilden gir en mer presis vektingsgrense: for en **fullt observert** bane er sekvens-NLL én første-tilstandsterm og én betinget overgangsterm for hvert senere minutt (forskningskilden linje 681–695). Summen deles på alle observerte celler, mens første tilstand får et eget tillegg på 0,03 ganger middel første-steg-NLL; første FLATT-aktivitet får et separat tillegg på 0,03 (linje 736–764 og 769–789). I de bevarte første/siste TRAIN-batchene er forholdet mellom observerte første-steg-celler og alle celler 65/81 686 = 0,000796 og 10/12 151 = 0,000823. Disse er **øvre grenser** for hvor stor koeffisient middel første-steg-NLL fra fullt observerte baner kan ha i sekvensdelen av akkurat disse batchene; delvis observerte baner kan gi annen gradientvei. Marginalprognosen ved hver senere ledetid avhenger derimot også av startfordelingen. Dette dokumenterer et konkret misforhold mellom den direkte starttilstandsvekten og gjentatt marginal skåring, men viser ikke at en bestemt tapsvekt eller ny arkitektur ville slå Markov.

Den faktiske TRAIN-kvitteringen bekrefter tapsdispatch på to lagrede grensebatcher: rapportert total er `sekvens-NLL + 0,03 × første-tilstand-NLL + 0,03 × første-FLATT-aktivitet-NLL` med avvik under 6×10⁻⁹, og gradientnormen er endelig og positiv. Første batch hadde 81 686 observerte framtidsceller; siste hadde bare 12 151. Tapene 1,0405 og 0,2248 er derfor **ikke** en matchet læringskurve. Den riktige TRAIN-læringssjekken er samme 256 foreldre før/etter fit. Denne kontrollen finner ingen aritmetisk tapsdispatchfeil, men beviser ikke at lossvektene eller designet er gode for framtidig retningspresisjon.

Den signerte senere rapporten viser hvor forskjellig den observerte **sekvens-NLL per celle** er fra logloss for modellens **marginale framtidsprognose** på de samme observerte cellene:

| Tidsramme | Sekvens-NLL/celle | Modellens marginale logloss | Markovs marginale logloss |
|---|---:|---:|---:|
| M5 | 0,728311 | 1,015613 | 1,007498 |
| M15 | 0,398291 | 0,992113 | 0,986921 |
| H1 | 0,166258 | 0,965562 | 0,962454 |
| H4 | 0,056917 | 0,868079 | 0,863076 |
| D1 | 0,011399 | 0,678129 | 0,640929 |

Sekvens-NLL bruker observert tidligere tilstand der hele banen er kjent og marginaliserer ukjente mellomsteg; prognosen må derimot føre **modellens egne** sannsynligheter fremover. Betinget banetap og marginal logloss måler derfor ulike ting, og den lavere sekvens-NLL-en kan ikke brukes som bevis for framtidig presisjon. Særlig D1s 0,0114 mot 0,6781 illustrerer dette. Modellen taper mot Markov i marginal logloss på alle fem; tallforskjellen alene beviser verken feil i rekursjonen eller at en annen lossvekt ville løse problemet.

## Samsvar med brukerens prognosemål

Modellen får historisk nåtilstand og genuine MTF-indikatorer og trenes mot framtidige lokale OPP/NED/FLATT-klasser på alle fem tidsrammer. Den bruker ingen fast prislinje ved prognosestart, og reverseringer kan gi flere framtidige retningsskifter. På dette punktet samsvarer målkontrakten med den løpende markedsbeskrivelsen.

Den godkjente 0,5 ATR-grensen er fortsatt en **forskningskandidat**, ikke en dokumentert grense for «bevegelse verdt å fange». M5-retning er ukjent på omtrent halvparten av +1-radene fordi M1-OHLC mangler hendelsesrekkefølge. D1-FLATT er svært sjelden. Loss teller observerte framtidsminutter, men har ingen eksplisitt målverdi for framtidig bevegelsesstørrelse, nytte eller tidspunktet for første relevante skifte. Dermed vet modellen hvilken **proxy** den skal lære, mens full samsvar med den praktiske OPP/NED/FLATT-bestillingen fortsatt er uavklart. Dette er et mål- og datagap; det er ikke bevis for at indikatorene er uten verdi.

## Neste avgrensede beslutningsgrunnlag

1. Behold nåværende modell som forskningsartefakt og fryst TRAIN-Markov som minste prognosereferanse. Ikke start en serie nye runder for å justere mot den gjenbrukte utviklingsperioden.
2. Dokumenter faktisk mottaks- og beregningstid for XAU M1 og ATR samt restartparitet før ONLINE-adgang. Hvis kilden ikke har slike klokker, behold ONLINE-status ukjent.
3. Avklar målsemantikken med en forhåndsdefinert **TRAIN-only målundersøkelse**: hvor ofte hvert MTF-nivå skal være FLAT for ønsket bevegelse, og om finere, faktisk tidsstemplet XAU-data kan avklare M5s ukjente intraminutt-rekkefølge. En ny terskel må gi full kausal rematerialisering; den kan ikke avledes av denne hypotetiske ro-andelen alene.
4. Før en kalibrator kan kvalifiseres, trengs kronologiske TRAIN-OOF-prediksjoner fra samme modellidentitet **med prefixnormalisering tilpasset bare tidligere rader**, en forhåndslåst kalibreringsregel og et senere ubenyttet målevindu. Den lagrede sluttmodellen og dens mellomliggende checkpoints gir ikke dette alene. En ny native modellutforming bør først begrunnes mot den målte feilen: eksplisitt tidsrammespesifikk skiftehazard og retning, klasse- og ledetidsavhengig påvirkning fra kjent nåtilstand som bevarer målbar informasjon om senere FLATT, direkte marginal skåring av FLATT-tilstandsandelen og retning ved forhåndsvalgte ledetider og eventuelt en TRAIN-fryst Markov-forankret residual. M5-krysset ved +4 er diagnostikk, ingen hard bytteregel; både Markov og nåtilstandsfri prior beholdes som separate fryste referanser. Skår FLATT-tilstandsandel og første utgang/tilbakekomst separat. Fastsett aksept for FLAT, UP/NED, skiftesannsynlighet, samlet Brier/logloss og lengre ledetider før eventuell fit. Mål særskilt om et **nytt** lokalt tilstandsskifte varsles før det skjer, over alle issue-rader inkludert ingen-skifte; utvalg bare av rader som senere faktisk skiftet er etterpåklokt. De nå lagrede rapportene har marginale klassesannsynligheter og sekvens-NLL, men ingen komplette overgangsmatriser; en ny engangsscoring må derfor bevare en forhåndsbestemt, begrenset skifte-/hendelsesskår i samme pass hvis dette skal testes uten flere runder. Bevar reverseringer og ingen maksimal holdetid. Dette er designforslag, ikke påvist forbedring.
5. Brukerens stående tillatelse til korte kvalitetstester åpner ingen automatisk serie nye native runder. En eventuell ny modellkontroll må ha separat eksakt plan, fersk initialbaseline ved endret ONLINE-funksjon, samme ressursvakter og forhåndsdefinert stoppkriterium. Gjenbrukt utviklingsdata kan bare styre design; forseglet TEST er ikke et tuningsett. Uavhengig evidens kommer fra den vedtatte månedlige forward-journalen, før økonomisk Entry-evaluering kan vurderes. Entry, Exit, handel og økonomisk edge er fortsatt utestet.

## Etterprøvbare bevis

De første revisjonene ligger under `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001` og er lesende analyser av bevarte filer. Den separate kalibreringskontrollen ligger under `/home/andre2/GX1_RUNS/DYNAMIC_MTF_SAVED_CALIBRATION_20261103_002`. Ingen rådata eller vekter ligger i Git.

| Fil | SHA-256 | Formål |
|---|---|---|
| `SAVED_PARAMETER_DELTA_AUDIT.json` | `84f85da8d8a85ce10011102272909ff43e2eb652b0245f924618911cf168497e` | Eksakt checkpointbinding og 642/642 endrede aktive tensorer. Relativ ratio for nullinitialisert gruppe er udefinert og brukes ikke. |
| `SAVED_PAIRED_SKILL_REVIEW.json` | `51cd62fc09f59186cbe3710bdf08d9f461a2f9041f3d68388164d2b1a9154a60` | Rapportreplay, matchet Brier/logloss, klasser, måneder og ukeblokker. |
| `SAVED_MARKOV_CHANGE_REVIEW.json` | `6c3057cd0aa2deefd3c9f9e0b4c07b319a64286236c23e7b88e9fa9d1f4398ee` | Matchet binær skifte-Brier; modellens felles overgangsfordeling er ikke lagret, modellscore kommer fra signert rapport. |
| `SAVED_FIRST_TRANSITION_TRAIN_VS_DEVELOPMENT.json` | `0fd6f693f9fdd1b2badd72eb68559512fa8134a2cbfb560ce07c3a4f3af960a9` | Første skiftrate på TRAIN og senere data. |
| `SAVED_256_FIRST_STEP_REVIEW.json` | `e31c0a4ea1fe1629f38b8284ce02b9f7431989b89aae244892c3bdd0dd22c911` | Samme 256 før/etter ved +1 minutt. |
| `SAVED_OOF_FIRST_TARGET_COVERAGE.json` | `e646e12b7d1a4f75ccc31bc63873e25023ab08063bb993b7a83235daa43c39c7` | Første måldekning og ukjent-årsaker. |
| `SAVED_TRAIN_QUIETNESS_FEASIBILITY.json` | `3db113d2b02bbd57fab83dda81f21ff4a8ab9a132cf6dd41cb7420c31daf1f1f` | TRAIN-only lokal ro ved gjeldende og hypotetiske terskler. |
| `SAVED_FIRST_CHANGE_RANK.json` | `4fe8128f99137127a8aac971ccbb4962a588ac8986c4e413be16c6e5203b100d` | Første skifte-AUC innen kjent nåtilstand, TRAIN og senere data. |
| `SAVED_FIRST_CHANGE_RANK_VERIFY.json` | `664f4e4018a82aa67e9ef3bc31778380389573106234d38587d6a71e54ab25be` | Uavhengig AUC-replay og månedssnitt. |
| `SAVED_OOF_STATE_RANK_BY_LEAD.json` | `5f9e2ccd77668728a2acec7b2fa3add417acdac76f157225d44a4767b311adab` | FLAT- og UP/NED-rangering ved seks ledetider. |
| `SAVED_OOF_RANK_VS_MARKOV.json` | `a6dc26ec989deabfb6826d542d8fd7ccb920c4eb4be3bfc04e631719e08d1c09` | Samme Markov-informasjon, AUC og replay av rapportert Brier. |
| `SAVED_FLAT_EXIT_DIRECTION.json` | `82819fa540fc17a2db6b1a64e5748d97946946d0766b8e1e87a5d83cd420021c` | Retningsrangering betinget på observert første FLAT-skifte. |
| `SAVED_LAST_M1_SIMPLE_BASELINE.json` | `1a621570da8c58451b82f06216bc342dc225063f118d2ff7b78fea511c085878` | Enkel kausal historisk M1-skår på samme rader; eksakt kildehash og join. |
| `SCOPED_LOCAL_XAU_TICK_INVENTORY.json` | `a7c5d2b61a0a0a909e4930d100d59b6906bde6ac6a934b5050fe40e0498437d0` | Avgrenset lokal inventering av XAU-tickcache; ingen full eller kildeidentisk dekning påvist. |
| `SAVED_FIRST_CHANGE_RELIABILITY.json` | `8f60b213b2a0a36299e7eb6f4fd59167c53c7c8c87d3351d9d0eec67575aa514` | TRAIN-definerte skårdesiler og skiftefrekvenser før den separate kalibreringskontrollen. |
| `DYNAMIC_MTF_SAVED_CALIBRATION_20261103_002/RESULT.json` | `dadafc3cc5b27b39960a73b4378bf1c89b1b069e5d479d0fec17762fcebf4d41` | Én 15-offset TRAIN-fit og én TRAIN-/gjenbrukt utviklingsscoring av lagrede +1-prediksjoner. |
| `SAVED_SAME_MODEL_OOF_AVAILABILITY.json` | `1250cfa5f27f01e5210b09d027dff97180a9649ff4e3e89bfe51d832037def72` | Tre lagrede prediksjonssett og elleve checkpoints; ingen TRAIN-OOF-prediksjoner i denne kampanjen. |
| `SAVED_LOSS_COMPOSITION_REVIEW.json` | `5d8ab38213a1c78c5ea439c9cf89ae92cbf54c40e1fea052f07de4a8d038a55f` | Faktisk 0,03/0,03-tapsaritmetikk på første/siste TRAIN-batch; ulike batcher er ingen læringskurve. |
| `SAVED_CURRENT_INPUT_LINEAGE_PARITY_REVIEW.json` | `0dbc16a860829f34d260ddf1187b83ea1ba1d23cd4ab1749335258379a77d589` | Aktuell inputmanifestparitet mot tidligere fysisk fullprofilrevisjon og GC-D1-historikkstøtte uten overlapp. |
| `SAVED_GC_D1_SUPPORT_SCORE_REVIEW.json` | `b52859791a9ab84509958c612d4a260694728458394965d3361eb1911e36a2b8` | Lesende D1-Brier per GC-historikkstøtte, med eksakt pooled paritet mot signert TRAIN-/utviklingsrapport; ingen kausal attribusjon. |
| `SAVED_NATIVE_LEAD_DIRECTION_SCORE_DISTRIBUTION.json` | `5e502219f354d69f76e25435bde8947b502ba52cfdb95870899a57f0fd60b179` | Lagret UP/NED-skårspredning og AUC på én full native tidsramme, TRAIN in-sample mot gjenbrukt utvikling og fryst Markov. |
| `SAVED_NATIVE_LEAD_MONTHLY_DIRECTION_AUC.json` | `479a506e958ac75ccbdef905748953d1c3b5f127d0ee632400739c734f42c413` | Samme native-ledetid og betinget UP/NED-skår delt på issue-måned; overlappende, beskrivende rader. |
| `SAVED_CURRENT_STATE_CONDITIONAL_NATIVE_RANK_REVIEW.json` | `13aa797e6b606217d10d3f86baa16b3dc48a6b034089a40b3febeccbf0a248db` | UP/NED-AUC innen historisk nåtilstand på fem TF, D1 per måned og enkle monotone tid-/GC-dekningskontroller. |
| `SAVED_STATE_GROUP_HORIZON_SCORE_DECOMPOSITION.json` | `62e16783993c640b14b4c0b0f1df271af6837976d90f591caf1b9c472fd9efb2` | Signert Brier-gap per ledeminutt og M5/D1 native-ledetid gruppert etter historisk nåtilstand, med eksakt rapportparitet. |
| `SAVED_NATIVE_AUC_WITHIN_BETWEEN_STATE_REVIEW.json` | `4acc526d424cdd957d279e59efee949762aa7a1bf4142c252b54f9e6f0679d4b` | Eksakt pooled AUC delt i OPP/NED-par innen og mellom historiske nåtilstandsgrupper på M5/D1 native-ledetid. |
| `SAVED_MARKOV_PRIOR_HORIZON_CROSSOVER.json` | `edd8d81b1dfe32c9cba22193937751975b23d969e53a97a94ca391ad0bf5ebeb` | Tre fryste Brier-kurver ved alle ledeminutter; M5 +1–15 bekreftet mot lagrede rader per senere måned. |
| `SAVED_NATIVE_FLAT_CONDITION_AND_BRIER_COMPONENTS.json` | `211f51239fb3504e1e1ef5db55c23c06385f8b72d2842eef4ecdb1a993bac676` | Eksakt native-ledetid Brier-paritet, FLATT etter historisk nåtilstand og klassevis skårgap; månedsdeling på gjenbrukt utvikling. |
| `SAVED_CURRENT_FLAT_PERSISTENCE_CURVE_TRAIN_LATER.json` | `efeadf38f2822b5701d0b70a97343a7bb82380cc793108f4a26a8ea053888c48` | FLATT-tilstandsandel fra +1 til én native ledetid, slutt-TRAIN in-sample mot senere gjenbrukt utvikling; samme-rad-kontroll. |
| `SAVED_CURRENT_FLAT_NATIVE_DIRECTION_RANK_TRAIN_LATER.json` | `d20f06db650329f34e36cbd553ae685072959059bb1652d0acfb8372d84bd174` | Betinget OPP/NED-AUC etter historisk FLATT nå ved én native ledetid, TRAIN og senere gjenbrukt utvikling. |
| `SAVED_CURRENT_FLAT_EXIT_REENTRY_TARGET_PATHS.json` | `d44c2231dd891e9de953b65143f0a987d254cc98d3615c9c09f3385821c5be02` | Målbaner med FLATT nå og ved native ledetid: ubrutt FLATT mot utgang og tilbakekomst, med fullstendighetsgrenser. |
| `DYNAMIC_FULL_FEATURE_COVERAGE_REUSE_002/RESULT.json` | `fa99d1358575e7c9882a698a392dcb4343f7c77ef9ce369ab7ea82c5b2416008` | Gjenbrukt fysisk featurestøtte på 47 602 TRAIN/17 288 senere foreldre; ingen individuell indikatorverdi kvalifisert. |
| `RUNTIME/prefix-fit/window-0002/OOF.json` | `6c03081f9058674681ecb827d1b8235c6e3ac197d9675e5871dcd757d348c63c` | Signert sekvens-NLL og marginal logloss/Brier på de samme senere observerte cellene. |
| `TARGET_AUDIT_005/TRAIN_AUDIT.json` | `5eb6be66a7cc887efcdf891ac5af2a9a2aac61cf7a4661e8021a64a963213771` | TRAIN-only klassefordeling og faktisk 0,5 ATR-størrelse på fem tidsrammer; overlappende celler. |
| `DYNAMIC_MTF_SAVED_CALIBRATION_20261103_001/ATTEMPT_001_FAILURE.json` | `3412f68ef1312abee5bf5870f5c3e5bae252b8ebde8b9f59e36c983938ab8b17` | Første forsøk konsumert etter FP32-summeringsvakt, før scoring. |
| `XAU_M1_NATIVE_2009_20260701_PAIR_WEEKCLOSED_20260927/MANIFEST.json` | `138d2f0819d08086d05c3de2a510279b563e7a269e4600b87332926a35017e39` | Kildekontrakt for historiske OANDA-M1-lys; barstempel og 60 s event-time-regel, ingen fysisk ankomstklokke. |

Tidligere signert kampanjeresultat og første FLAT-per-lead-replay står i [engangspassrapporten](DYNAMIC_MTF_STATE_ONEPASS_RESULT_20261103.md). Kildeeiere: `gx1/contracts/entry_dynamic_mtf_state_targets_v1.py`, `gx1/contracts/entry_native_direction_research_v1.py`, `gx1/contracts/entry_dynamic_mtf_state_report_v1.py` og `gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py`.
