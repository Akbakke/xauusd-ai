# GX1: hvorfor det historiske tilstandspasset ikke slo Markov

## Beslutning

Ingen ny native modellrunde er begrunnet av de lagrede resultatene. Det ene passet lærte på TRAIN, men råmodellen slo ikke den likt informerte, fryste TRAIN-Markov-referansen samlet på senere gjenbrukte utviklingsdata for noen av de fem tidsrammene. Første-minutts skiftesannsynlighet er systematisk feilkalibrert allerede på TRAIN. Én etterfølgende, avgrenset kalibrering på lagrede TRAIN-prediksjoner ga bedre første-minutts Brier og logloss enn Markov også på gjenbrukt utvikling, men dette er ingen uavhengig edge-test og sier ikke at retningsbevegelsen ved lengre ledetider er lært. Årsaken til råmodellens feilkalibrering er ikke kausalt isolert. Økonomisk edge er ikke evaluert.

Den første revisjonen var lesning av bevarte prediksjoner, mål, checkpoints, inputbindinger og kilde. Den senere CPU-kontrollen tilpasset 15 énparameter skifteoffsets på lagret TRAIN-output og scoret disse én gang på TRAIN og gjenbrukt utvikling. Den konstruerte eller kjørte ingen native modell og gjorde ingen optimizeroppdateringer, TEST, Exit, broker eller handel. Fryst treningskilde: `13c6bb8f2f33c50806908673fa33accb5edaba63`; råmodellens ettpass-resultat og den separate kalibreringskontrollen holdes atskilt.

## Bevisstatus per mulig forklaring

| Forklaring | Hva eksisterende bevis viser | Status |
|---|---|---|
| Frakoblet data-/modellbane | Fem tidsrammer og full featureprofil er bundet; 642 aktive tensorer endret seg. GC-D1-historikkstøtten er uten overlapp mellom TRAIN og senere data. Dette måler ikke indikatorenes prediktive verdi. | Ingen frakobling påvist; konkret inputskift, kausal verdi ukjent |
| Historisk klokke/lekkasje | TRAIN-normalisering og målcutoff er bundet før senere data. XAU-M1 og ATR har bare event-time, uten faktisk mottaksklokke. | Historisk sjekk bestått; ONLINE ukjent |
| Feil mål | 0,5 ATR er kandidat. M5 +1-retning er 49,8 % observert; D1-FLATT er sjelden. M5-grensen er median 1,97 historisk siterte spreads på TRAIN, men lokal range er ikke fangbar Entry-bevegelse. | Målskala og intraminutt-retning uavklart; ingen terskelendring |
| Feil sannsynlighetsnivå | Råmodellen holder M5-tilstand for lenge ved +1, men gir for lav FLATT-andel ved +5. M15 har samme fortegnsskifte; én TRAIN-kalibrering forbedret bare +1. | Marginal kurvefeil målt også på TRAIN; én offset løser den ikke |
| Taps-/modellutforming | Sekvens-NLL trener betingede baner mens prognosen scores marginalt. Faktorbytte av lagret output viser at modellens betingede OPP/NED-del forverrer senere Brier/logloss mot Markov på alle fem TF. | Outputsvikt lokalisert; ingen konkret parameter-/kodeårsak bevist |
| Manglende framtidig retningssignal | Senere modell-UP/NED-AUC er rundt 0,5 ved én native ledetid på alle fem. En allerede rutet M5-candleverdi gir svak motsatt rangering ved +5; binær FLATT/ikke-FLATT-skår slår Markov på M5 i hver senere måned. | Modellens betingede retningsoutput overser et enkelt historisk rangeringssignal; ingen uavhengig eller økonomisk edge |
| Generalisering/økonomi | M5–H4 fullgrid-Brier snur fra TRAIN in-sample til senere tap; H1/H4/D1s betingede retningsrangering faller særlig. GC-D1-inputstøtte skifter, men årsaken er ikke isolert. | Gjenbrukt utvikling, ingen uavhengig eller økonomisk edge |

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

### D1: sterkt overlappende utfallsrader

Ved D1s native +1 440 minutt er 30 976 observerte ikke-FLATT TRAIN-rader fordelt på bare 164 UTC-utfallsdager; senere er 13 180 rader fordelt på 61 dager. Blant etterfølgende observerte issue-tider med fem minutters mellomrom beholder D1-fasiten samme NED/OPP-klasse i 99,47 % på TRAIN og 99,40 % senere. Den deterministiske inndelingen i sammenhengende like klasseforløp gir 340/134 forløp. Dette er **ikke** 340/134 uavhengige markedsforsøk, men viser hvor sterkt radene er knyttet sammen. Til sammenligning er M5s nabofasit lik i omtrent 49 % av slike par.

| D1, betinget OPP/NED | Radvektet modell-AUC | Lik vekt per UTC-utfallsdag | Én gjennomsnittsskår per klasseforløp |
|---|---:|---:|---:|
| Slutt-TRAIN, in-sample | 0,6633 | 0,6425 | 0,5909 |
| Gjenbrukt senere utvikling | 0,4841 | 0,4684 | 0,4806 |

Dette er ulike vektinger og derfor ulike estimander; forløpene defineres dessuten fra framtidig fasit og brukes bare som følsomhetskontroll. Den høye TRAIN-rangeringen er følsom for analyseenhet, men overlever dagvekting. Den senere svikten består. På alle fem TF er modellens senere treklasse-Brier-gap mot matchet Markov fortsatt positivt når hver utfallsdag får lik vekt. For D1 ved +1 440 min er gapet −0,013175 på TRAIN og +0,021423 senere med ordinær radvekt, mot −0,012426 og +0,023081 med lik dagvekt. De ordinære tallene ble gjenskapt mot den signerte rapporten. Overlappende mål kan dermed overdrive opplevd bevismengde, men dagvekting fjerner ikke den observerte senere tapsforskjellen. [Klyngekontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_TARGET_CLUSTERING_SENSITIVITY.json) SHA-256 `c6fa51b69c3830536b4d08208b90c12a9d957a8d1d5791edc3593c68cbecc306`.

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

### Før og etter på de samme 256 TRAIN-foreldrene

Den lagrede initialmålingen omfatter 256 TRAIN-foreldre. Alle foreldre og issue-tider er eksakt funnet igjen i sluttmålingens 47 602 rader; tallene under bruker derfor **identiske** rader ved én native ledetid per tidsramme. AUC gjelder bare observerte NED/OPP-mål; treklasse-Brier gjelder alle observerte mål. Candle-korrelasjonen er Spearman mellom siste fullførte native `close_open_atr` og modellens betingede OPP-skår.

| TF | NED/OPP-rader | AUC initial → slutt | Candle–OPP-korrelasjon initial → slutt | Treklasse-Brier initial → slutt |
|---|---:|---:|---:|---:|
| M5 +5 | 135 | 0,603 → 0,509 | 0,018 → 0,456 | 0,677 → 0,611 |
| M15 +15 | 201 | 0,487 → 0,493 | 0,054 → 0,187 | 0,679 → 0,588 |
| H1 +60 | 210 | 0,453 → 0,606 | −0,202 → −0,070 | 0,687 → 0,573 |
| H4 +240 | 211 | 0,528 → 0,619 | −0,137 → 0,174 | 0,688 → 0,561 |
| D1 +1 440 | 179 | 0,515 → 0,653 | −0,074 → 0,479 | 0,679 → 0,508 |

For M5 gir en parvis resampling av 36 issue-uker et **beskrivende** 95 %-intervall for slutt minus initial på −0,234 til +0,043 AUC, +0,177 til +0,673 i candle-korrelasjon og −0,115 til −0,022 i Brier. AUC-fallet er dermed usikkert i dette lille, spredte utvalget. Treningen endret outputen og forbedret in-sample treklasseskåren, samtidig som M5-OPP-skåren ble mer knyttet til candlefortegnet. Ingen enkeltparameter eller kausal tapsårsak er identifisert. Utvalget er TRAIN, ikke OOF eller uavhengig generalisering. [Matchet kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_IDENTICAL_256_INITIAL_FINAL_DIRECTION_REVIEW.json) SHA-256 `305b149cfc3a27fe11f05c223dbff12cd15946129ff3e1e7e3dbc133cfa1c87d`; ingen nye modellkall eller optimizeroppdateringer.

### M5: enkel rangering fra allerede rutet candleinput

Som en lesende kontroll valgte vi på TRAIN fortegnet **motsatt** av `close_open_atr` for siste komplette M5-lys og scoret samme regel på lagrede prognoser. Dette feltet er allerede rutet til `price_action_candle_encoder`; cachet M5-bar har start `issue−5 min` og er lukket ved issue etter event-time. Mot den hashbundne M1-kilden stemte fortegnet i `close_open_atr` og den andre rutede candlekroppen med femminuttersbevegelsen på alle 47 284 sammenlignbare TRAIN- og 17 185 senere issue-rader. Den fysiske mottakstiden er fortsatt ukjent.

| M5 +5, observert NED/OPP | Rader | Motsatt `close_open_atr` AUC | Lagret modell AUC | Matchet TRAIN-Markov AUC |
|---|---:|---:|---:|---:|
| Slutt-TRAIN, in-sample modell | 22 832 | 0,5205 | 0,4991 | 0,4924 |
| Gjenbrukt utvikling, jan.–mars 2025 | 7 812 | 0,5185 | 0,4979 | 0,5003 |

På de samme senere ikke-FLATT-radene var faktisk OPP-andel 53,47 % etter en NED-candle og 50,17 % etter en OPP-candle. Modellens middel betingede P(OPP) gikk motsatt vei: 47,68 % og 53,82 %. Den lagrede modellskåren er altså positivt knyttet til candlefortegnet (Spearman 0,478), mens fasiten er svakt negativt knyttet til det (−0,032). TRAIN viser samme fortegnskonflikt. Dette er en observert output-sammenheng; den beviser ikke at modellen direkte vekter akkurat dette feltet feil.

På senere data var enkeltskåren over 0,5 i hver måned (0,5098/0,5278/0,5183). Beskrivende 14-ukers blokkbootstrap ga 95 %-intervall 0,5093–0,5277 for AUC, og 0,0067–0,0345 for forskjellen mot modellen. Blant de 3 273 senere radene med **ukjent** M5-nåtilstand var enkeltskåren 0,5311 mot modellens 0,4887; 3 265 av dem var maskert for både ukjent intraminutt-rekkefølge og uavklart retning. For kjent FLATT nå var enkeltskåren 0,4589, så regelen er ikke ensartet på tvers av tilstander. Dette viser at et svakt retningstegn finnes i et allerede tilgjengelig felt og at råmodellens +5-output ikke utnytter det i denne rangmålingen; det isolerer ikke om tapsmål, representasjon, normalisering, regime eller optimering er årsaken.

Innen samme **ukjente** nåtilstandsgruppe var senere OPP-andel 54,51 % etter NED-candle og 49,32 % etter OPP-candle, mens modellen ga 50,91/53,31 % i gjennomsnitt. Forskjellen er altså ikke bare miks mellom de kjente nåtilstandene. Beskrivende 14-ukers blokkintervall for motsatt-candle-AUC i denne gruppen var 0,5130–0,5529 og for AUC-fordelen mot modellen 0,0144–0,0727. Gruppen er fortsatt valgt etter at framtidig NED/OPP er observert; dette er ingen operativ regel. [Nåtilstandskontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_M5_CANDLE_CURRENT_STATE_STRATIFICATION.json) SHA-256 `1e09c2087d19d66383eaa96abb3aa0a76136f6026577f7fd71ca5e4bc8fd3757`.

Den enkle beslutningen «motsatt candlefortegn» traff 51,59 % av de 7 785 senere ikke-null-radene, mens alltid-OPP traff 51,77 % på de samme radene. AUC gjelder bare framtidig observerte ikke-FLATT-mål, og den senere perioden er gjenbrukt. Dette er derfor ingen full treklasseprognose, uavhengig edge eller handelsregel. Den gir en ekstra forhåndslåst M5-referanse for en eventuell streng TRAIN-OOF-kandidat. Ingen native modellkall eller oppdateringer ble gjort. [Lagret kontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_ROUTED_M5_CANDLE_REVERSAL_AUDIT.json) SHA-256 `081969b226d06f05c551629a09ac2210bcd62d8bff282cca3a9f9f7f93e0c501`.

### Samme enkle candlekontroll på alle fem tidsrammer

For hver tidsramme brukte vi `close_open_atr` fra siste **fullførte native candle** ved issue. Fortegnet (med/mot candle) ble valgt bare fra slutt-TRAIN, så ble den samme regelen målt på gjenbrukt senere utvikling. Alle fem cache-featurefilene er hashkontrollert mot inputmanifestet. Dette er en lesende, betinget NED/OPP-rangering ved én full native ledetid per TF, ikke en ny modell eller et nytt mål.

| TF | TRAIN-valgt fortegn | Senere rader | Candle AUC | Modell AUC | Matchet Markov AUC | Beskrivende 14-ukers candle-intervall |
|---|---|---:|---:|---:|---:|---:|
| M5 | motsatt | 7 812 | 0,5185 | 0,4979 | 0,5003 | 0,5093–0,5277 |
| M15 | motsatt | 12 899 | 0,5123 | 0,4897 | 0,4843 | 0,5027–0,5213 |
| H1 | motsatt | 14 696 | 0,5188 | 0,4871 | 0,4894 | 0,5027–0,5328 |
| H4 | med | 14 894 | 0,4928 | 0,4866 | 0,4967 | 0,4645–0,5225 |
| D1 | med | 13 180 | 0,4989 | 0,4841 | 0,4515 | 0,4081–0,5636 |

M15s candle-AUC var over 0,5 i hver senere måned; H1 var nær 0,5 i januar og høyere i mars. H4/D1s TRAIN-valgte fortegn ga ikke senere rangering over 0,5. På **alle fem** var enkel fortegnsbasert treffrate lavere enn alltid-OPP på de samme ikke-null-radene. Ledetidene og candle-alderen varierer, og målene overlapper sterkt. Fem TRAIN-valg, gjenbrukt utvikling og beskrivende intervaller gir ingen uavhengig eller økonomisk edge. [Fem-TF-kontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_MTF_LAST_CLOSED_CANDLE_NATIVE_DIRECTION_AUDIT.json) SHA-256 `d5e4e26e4fb3307f2317cc75bf364458520d2ac6bfe7b56724d49ecb1a11ef26`.

Den fryste inputbanen er kontrollert særskilt for dette feltet. Prediksjonsissue er eksakt forelderens M5-barstart pluss fem minutter for alle 47 602 TRAIN- og 17 288 senere foreldre. Kildebundet Entry-gather velger siste MTF-bar som er lukket ved issue. `close_open_atr` tilhører candle-spesialisten på alle fem TF, har positiv TRAIN-skala 0,689–0,800, er observert og ikke konstant i TRAIN-normaliseringsutvalget. Den faktiske kildeformelen er en stigende, ikke-klippende `asinh((x−center)/scale)`; feltet har ingen verdi-maske. På de eksakt samme observerte native NED/OPP-radene var rå og rekonstruert normalisert AUC identisk på **alle fem TF i begge perioder**, uten færre unike verdier eller ranginversjoner. Denne kontrollen avkrefter direkte ranginversjon eller klipping i normaliseringen av candlefeltet. Den tester ikke om den komprimerte skalaen påvirket nettverkets tilpasning, om nettverket utnytter feltet godt eller om fysisk XAU/ATR-mottak var klart ved issue. [Normaliseringskontroll](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CANDLE_NORMALIZATION_RANK_PARITY.json) SHA-256 `e208c7c7b8cf1d16c562ecc0846edfc5ad3a175a1785a3206c2c8d13090924e7`.

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

Et eksakt replay av de samme lagrede målene viser et **tapsdekningsgap** ved senere ledetider. Ved M5 +5 er treklasse-retningen observert på 26 315 TRAIN- og 8 980 senere rader, mens fullstendig lokal range avgjør FLATT/ikke-FLATT på 47 566 og 17 285. Det gir ytterligere 21 252 TRAIN- og 8 305 senere rader med kjent **ikke-FLATT**, men ukjent OPP/NED. Ved M15 +15 er tilleggene 6 224 og 2 496; H1 +60 865 og 324. H4/D1 har langt færre slike rader. Ingen av de binære tilleggscellene er FLATT, og de gir **ikke** en retningsetikett. Ved +1 gjenskaper replayet eksakt rapportens binære aktivitetsantall for alle fem TF; M5 har 22 710 ekstra TRAIN- og 8 671 senere rader der.

Forskningskoden legger binær FLATT/ikke-FLATT-NLL til **bare ved +1** med vekt 0,03. På senere ledetider finnes ikke et eksplisitt binært tapsledd for de ekstra cellene; sekvens-NLL kan fortsatt få indirekte informasjon fra observerte nabostater. Det er en konkret mulighet for mer måltilsyn av FLATT-tilstandsandelen uten å gjette intraminutt-retning, men verken ny vekting eller bedre generalisering er testet. [Dekningskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_BINARY_ACTIVITY_LABEL_COVERAGE_NATIVE_LEADS.json) SHA-256 `963115cac4f3b7859a47f8155985c4305d361c9763a867b3db49b9575a862331`; null fit, modellkall og TEST.

Men skåring av **allerede lagrede prediksjoner** viser at flere binære labels ikke er en påvist første rettelse. På samme senere, komplette lokale FLATT/ikke-FLATT-mål ved én native ledetid er binær Brier modell / likt informert Markov:

| TF | Binær Brier modell / Markov | Treklasse-Brier modell / Markov på observerte retningsmål |
|---|---:|---:|
| M5 +5 | 0,063594 / 0,071860 | 0,612158 / 0,607321 |
| M15 +15 | 0,088777 / 0,089531 | 0,596087 / 0,592204 |
| H1 +60 | 0,083223 / 0,083650 | 0,578103 / 0,574463 |
| H4 +240 | 0,060105 / 0,060925 | 0,545288 / 0,538942 |
| D1 +1 440 | 0,008363 / 0,007661 | 0,490322 / 0,468899 |

Binær og treklasse-Brier har ulike mål og radutvalg; bare modell–Markov-parene **innen** hver kolonne kan sammenlignes. På M5s 8 305 senere binære tilleggsceller med ukjent retning er modellens Brier 0,010486 mot Markovs 0,026905. Modellen er dermed allerede bedre på akkurat disse ikke-FLATT-radene uten direkte senere binært tapsledd. M5s binære Brier-fordel finnes i januar/februar/mars hver for seg (−0,00966/−0,00702/−0,00801 modell minus Markov); binær FLATT-AUC er 0,567/0,558/0,565. Rangeringen er sterkest når historisk M5-nåtilstand er ukjent (AUC 0,585 på 8 299 senere rader), men er nær 0,5 innen kjent FLATT nå (0,491). Dette er **gjenbrukt utvikling med overlappende mål**, ikke uavhengig edge, og det løser ikke OPP/NED-rangeringen. +1 binær Brier/logloss og antall gjenskapes eksakt mot signert rapport på alle fem TF i begge perioder. [Skårkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_BINARY_ACTIVITY_SCORE_NATIVE_LEADS.json) SHA-256 `c8bb833a3f0777f362108252a9b2c9dbcbb98aaa7cde8fea3df405ec2a451bb4`; [måned/rang-kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_BINARY_ACTIVITY_MONTHLY_RANK.json) SHA-256 `c440f68e4945b2e1318e7e7e252d7054c9a9240b6ad4755d5d1521f7572d7da3`. Ingen modellkall, fit eller TEST.

En faktorvis **outputkontroll på de samme observerte treklasse-radene** skiller `P(FLATT)` fra betinget `P(NED/OPP | ikke-FLATT)`. Tabellen viser endring i senere treklasse-Brier fra Markov når bare én faktor byttes til modellens lagrede faktor; negative tall er bedre enn Markov. «Full modell» er den faktisk lagrede prognosen.

| TF / native ledetid | Bare modellens FLATT-andel | Bare modellens betingede retning | Full modell |
|---|---:|---:|---:|
| M5 +5 | −0,000751 | +0,004901 | +0,004838 |
| M15 +15 | +0,001295 | +0,002441 | +0,003882 |
| H1 +60 | −0,000532 | +0,004023 | +0,003640 |
| H4 +240 | −0,001508 | +0,007697 | +0,006346 |
| D1 +1 440 | +0,001351 | +0,020229 | +0,021423 |

På senere data **forverrer modellens betingede retningsfordeling Brier og logloss mot Markov på alle fem** når FLATT-sannsynligheten holdes lik Markov. FLATT-faktoren hjelper M5/H1/H4 i Brier på de samme radene, men ikke M15/D1; i logloss hjelper den M5/M15/H1/H4. På TRAIN in-sample hjalp retningsfaktoren H1/H4/D1, men ikke M5/M15. Dette lokaliserer skårsvikten til **outputfaktoren for retning**, særlig generaliseringsfallet på lengre TF; det identifiserer ikke hvilken indikator, tapsvekt eller modellparameter som forårsaker den. Hybridene er regnestykker av lagret output, ikke trenbare kandidater eller tillatte handelsregler, og byttevirkninger gjør at kolonnene ikke summerer eksakt. [Faktorkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_ACTIVITY_DIRECTION_PROBABILITY_SWAP.json) SHA-256 `1e33e1b359d254a3533f48869be884bfb9873f43c46ac83aa4071ecc6a24fbc9`; original modell/Markov Brier og logloss gjenskapes fra signert rapport, null nye modellkall eller fit.

Samme lesende faktorbytte er nå kontrollert på **seks faste ledetider** (+1, +5, +15, +60, +240 og +1 440 min) for alle fem TF på gjenbrukt senere utvikling. Eksakte observerte radantall og original modell-/Markov-Brier stemmer med de signerte ledetidsrapportene; de fem native panelene gjenskaper forrige faktorkvittering innen `1,5×10⁻¹⁴` også for logloss.

| TF | Ledetider der modellretning gir svakere Brier enn Markovretning | Ledetider der modellens FLATT-andel gir bedre Brier |
|---|---:|---:|
| M5 | 6/6 | 5/6 |
| M15 | 5/6 | 5/6 |
| H1 | 3/6 | 5/6 |
| H4 | 5/6 | 2/6 |
| D1 | 6/6 | 0/6 |

Betinget modellretning forverrer 25/30 paneler. På H1 hjelper den ved +1/+5/+15, men forverrer ved +60/+240/+1 440; en ensartet retningskorreksjon over hele horisonten er derfor ikke begrunnet. Full modell er svakere enn Markov i 24/30 paneler. Dette er seks utvalgte, sterkt overlappende ledetider og **ikke** en fullgrid- eller uavhengig test; faktorbytter er ingen kausale parameterinngrep. Funnene prioriterer forhåndslåst, tidsramme- og ledetidsavhengig retningsmåling i en eventuell streng TRAIN-OOF-kandidat, mens M5/H1s nyttige FLATT-informasjon må bevares. [Seks-ledetidskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_SPARSE_LEAD_ACTIVITY_DIRECTION_SWAP.json) SHA-256 `a1c1018b62889340c9e7f0ab965caeec17c68e615d5f77f65311f3e04f52b040`.

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

En separat **TRAIN-only skala-kontroll** sammenligner den lagrede 0,5 ATR-prisgrensen ved issue med bid/ask-spreaden ved siste lukkede M1-lys fra samme hashverifiserte XAU-kilde. 47 591 av 47 602 TRAIN-issue-tider traff både M1-sitat og target-akse eksakt; de øvrige 11 manglet begge. Tallene er historiske sitater, ikke ordreutførelse.

| TF | Median grense / sitert spread, TRAIN | TRAIN-andel der grense ≤ 2× sitert spread |
|---|---:|---:|
| M5 | 1,97× | 51,4 % |
| M15 | 3,60× | 4,1 % |
| H1 | 7,62× | 0,1 % |
| H4 | 15,64× | <0,1 % |
| D1 | 43,37× | <0,1 % |

M5-grensen er 3,126 bps i median mot sitert spread 1,625 bps; på senere gjenbrukt utvikling er tilsvarende 3,113 og 1,866 bps, og 65,2 % av grensene er ≤2 slike spreads. **Lokal høy–lav-rekkevidde er ikke bevegelsen en Entry kunne fanget**, og spreaden her omfatter ikke tidsforsinkelse, glidning eller gebyrer. Dette kvalifiserer ingen handel og velger ingen ny terskel. Det gjør derimot «bevegelse verdt å fange» til et konkret, fortsatt uavklart TRAIN-målvalg, særlig på M5; det er ikke bevist som årsak til svak retningsrangering. [Skalakvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_TRAIN_THRESHOLD_VS_M1_QUOTED_SPREAD.json) SHA-256 `69d133047f2e1e2d943aa376b7001fd2c8ebdcfe97b1ff705bad9669b9adf551`; M1-filen ble hashet mot bundet kildereferanse, ingen modellkall, fit eller TEST.

Den bundne normaliseringskvalifiseringen bruker 47 602 TRAIN-foreldre med siste beslutning 31.12.2024 kl. 21:55 UTC, før TRAIN-grensen 01.01.2025 kl. 00:00 UTC; den utførte null nye statistikkfit. Dette viser ingen innblanding av senere utviklingsrader i den kvalifiserte normaliseringen (kvalifisering SHA-256 `51ac757b4c1709d1104a750937ebb07bded36fe890089d436e298e801139d1fc`).

Historisk event-time-sjekk av siste M1-lys og ATR er bestått. Faktisk fysisk mottaks-, prosesserings- og restarttid for disse verdiene er fremdeles ukjent; dette stenger ONLINE-bruk, men er ikke påvist som årsak til det historiske Brier-gapet. Den bundne XAU-M1-filen har 5 959 045 rader og bare `time`, OHLC, bid/ask OHLC og `volume`; ingen mottaks- eller ferdigberegningstid. Kildemanifestet sporer OANDAs historiske `/instruments/XAU_USD/candles` med `complete=true`, M1 barstart og `decision_available_offset_seconds=60`: det er en event-time-regel om når lyset er lukket, ikke en målt ankomstklokke. `M5_ENRICHED.parquet` har tids-/ATR-felt og mange GC-klokkefelter, men disse dokumenterer ikke mottakstid for XAU-M1 eller XAU-ATR. Historiske filer alene kan derfor ikke bevise hva ONLINE-modellen faktisk ville ha mottatt ved issue-tid.

### Hvor mye av framtidsbanen er sammenhengende observert?

En uavhengig lesning av lagret targetserie og de nøyaktige 47 602 TRAIN-/17 288 senere issue-tidene finner **null komplette 1 440-minutters M1-klokkevinduer i begge perioder**. Dette gjelder også 47 338 TRAIN-/17 013 senere vinduer der alle 1 440 minuttene ligger før periodens cutoff; fraværet skyldes altså ikke bare høyre sensur ved cutoff. Kildeserien har 280 minuttgap; de vanligste validity-stegene over gap er 65 minutter (177 ganger) og 2 945 minutter (45 ganger). Dette er et faktum om den lagrede historiske klokken, ikke bevis for hvilke bevegelser som kunne ses i sanntid. Alle parent×TF-baner med minst én eligible måltilstand er dermed delvis observerte; den fryste `state_chain_negative_log_likelihood` bruker sin partiell-bane-marginalisering for dem, ikke grenen for en fullt observert 1 440-minutters bane. Målceller og tilstøtende par gjenskaper de signerte rapportene eksakt; rapportens metrikk for helt komplette 1 440-minutters vinduer er også 0 på alle fem TF.

| TF | Senere observert tilstand ved native ledetid | Senere sammenhengende observert fra +1 gjennom native ledetid | Median sammenhengende prefiks, minutter |
|---|---:|---:|---:|
| M5 +5 | 8 980 | 4 706 | 0 |
| M15 +15 | 14 659 | 10 668 | 26 |
| H1 +60 | 16 251 | 14 486 | 270 |
| H4 +240 | 15 917 | 13 477 | 580 |
| D1 +1 440 | 13 256 | 0 | 660 |

Null komplette døgnbaner betyr **ikke** null direkte nabopar i sekvensloss. Den samme, mot signert rapport paritetskontrollerte kvitteringen teller observerte tilstøtende målpar. Andelen under er antall slike par delt på antall observerte målceller; dette er celler med en observert forgjenger, **ikke** andel av faktisk gradient eller uavhengige hendelser:

| TF | TRAIN nabopar / observerte celler | Senere nabopar / observerte celler |
|---|---:|---:|
| M5 | 26 106 948 / 31 090 770 = 83,97 % | 8 759 719 / 10 597 435 = 82,66 % |
| M15 | 48 473 313 / 50 035 520 = 96,88 % | 17 281 452 / 17 865 797 = 96,73 % |
| H1 | 56 264 718 / 56 535 402 = 99,52 % | 20 511 276 / 20 602 532 = 99,56 % |
| H4 | 56 470 482 / 56 582 604 = 99,80 % | 20 889 073 / 20 924 619 = 99,83 % |
| D1 | 50 938 295 / 51 025 541 = 99,83 % | 20 774 289 / 20 805 360 = 99,85 % |

Den partielle log-likelihooden marginaliserer ukjente tilstander, men observerte sammenhengende segmenter inneholder fortsatt betingede overgangsledd (fryst modellkilde `state_chain_negative_log_likelihood`). Dermed kan «ingen komplett 1 440-minutters bane» ikke alene forklare svak læring på native ledetid. Parene sier ikke hvor stor gradient hvert TF, skifte eller langt ledeminutt faktisk fikk, og de etablerer ikke marginal prognosepresisjon.

D1 kan altså ha en kjent tilstand ett døgn senere uten at hele mellomliggende banen er kjent. For M5 er første framtidsminutt uobserverbart på omtrent halvparten av senere rader, hovedsakelig på grunn av ukjent rekkefølge i M1-lyset; bare 4 706/17 288 har et komplett observert prefiks til +5 selv om 8 980 har kjent endepunkt ved +5. Dette gjør et krav om «full 1 440-minutters første-skifte-/reverseringsbane» umulig på denne kilden. En framtidig hendelsesskår må rapportere observert sammenhengende prefiks og sensur ved kildegap, mens endepunktstilstand og senere gyldighet scoret separat. Det innfører ingen maksimal holdetid og beviser ikke at partiell-bane-NLL er årsaken til svak retning. [Prefiks-/klokkekvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_DYNAMIC_STATE_PREFIX_OBSERVABILITY.json) SHA-256 `21a685f1367b075f8754ea1e8c7ed5bf529808e0a3d38241e1444d6c1b05c264`.

En separat kontroll av **unike, sammenhengende M1-minuttpar** finner ikke at M5-/M15-skifter samler seg ved ATR-oppdatering: på senere data er M5 20,84 % ATR-oppdateringspar mot 20,41 % av de observerte skiftene, M15 6,90 % mot 6,64 %. H1 er 1,69 % mot 1,82 %. H4/D1 har få slike oppdateringsskifter, så deres høyere relative andel er ustabil. Sammenfall kan uansett ikke bevise at grenseendringen *forårsaket* et skifte. Denne lesende kontrollen nedprioriterer en enkel ATR-grenseoppdateringsforklaring for M5/M15, men låser ikke 0,5 ATR som riktig nytteverdi. [ATR-grensekvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_ATR_BOUNDARY_TARGET_CHANGE_REVIEW.json) SHA-256 `dcadb4d747ee98724fc0b0a0947a12d5b9f1ae1f725863f7b7e40bc028b9590f`.

### Hvor målcellene blir uobserverbare

Den signerte fullgrid-rapporten deler hvert forelder×ledeminutt i **gjensidig utelukkende** observert klasse, manglende M1-spørring, umodent mål eller uavklart retning. Senere har alle TF 14,59 % manglende M1-spørringer og 0,76 % umodne celler; forskjellen ligger særlig i retningen:

| TF | TRAIN observert / uavklart retning | Senere observert / uavklart retning |
|---|---:|---:|
| M5 | 45,36 / 39,72 % | 42,57 / 42,08 % |
| M15 | 72,99 / 12,09 % | 71,77 / 12,88 % |
| H1 | 82,48 / 2,60 % | 82,76 / 1,89 % |
| H4 | 82,55 / 2,53 % | 84,05 / 0,60 % |
| D1 | 74,44 / 10,64 % | 83,57 / 1,08 % |

M5s retningsobservasjon er dermed hovedsakelig begrenset av M1-OHLCs ukjente rekkefølge og uavklart lokal retning, ikke bare av lukkede markedstimer. D1 har derimot et stort **observasjons-/utvalgsskifte**. Den bundne kalenderkvitteringen klassifiserer 280 kildehull: 228 dokumenterte markedspauser og 52 ukjente kildefravær som må sensureres på høyre side; 48 av de ukjente hullene er i TRAIN og fire i senere periode. På den **unike** D1-kildeserien har 27 742 av 28 438 uavklarte TRAIN-minutter også ukjent-kildegap-flagg, mot 1 498 av 1 498 senere. Bare 924 uavklarte TRAIN-minutter har intraminutt-rekkefølgeflagget; flaggene kan overlappe. Dette forklarer i stor grad hvorfor D1-måldekningen er lavere på TRAIN. Det beviser ikke at kildehullene forårsaker modellens senere rangtap: D1-AUC faller også i måneder med langt færre hull, og uobserverte klasser inngår ikke i skåren. Fyll aldri slike hull med gjetninger; ny kilde må ha egen klokke- og eierskapskontroll. [Deknings-/gapkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_TARGET_COVERAGE_GAP_ORIGIN_REVIEW.json) SHA-256 `d9d846dfd7ef1c052347eaae092fe8b7eae928d81b94234434a61c99e384f0ca`.

### Skår før og etter kildegap i prognosebanen

På **identiske observerte endepunkter ved én native ledetid** er modell−Markov-Brier regnet separat etter om banen fra issue krysser et M1-kildegap. Gapene er eksakt matchet til den publiserte kalenderens 280 intervaller; alle målantall og original modell-/Markov-Brier gjenskaper de signerte ledetidsrapportene. På senere utvikling har M5 +5, M15 +15 og H1 +60 **ingen** slike gap blant sine henholdsvis 8 980/14 659/16 251 observerte endepunkter, men modellen taper fortsatt mot Markov. Deres native tap kan derfor ikke skyldes at akkurat disse banene krysser et M1-hull.

H4 +240 skiller seg ut: 14 270 senere rader uten gap har Brier-gap **+0,003288**, mens 1 640 rader over en dokumentert daglig markedspause har **+0,033055**. Bare sju H4-rader krysser et ukjent kildehull; de gir ingen nyttig egen slutning. På TRAIN in-sample er H4-gapet −0,005786 uten gap og +0,007101 over kjent pause. Januar/februar/mars viser senere kjente-pause-gap +0,044393/+0,031181/+0,023388, mot +0,002668/+0,007861/−0,000405 uten gap. Et parvis, **beskrivende** 14-ukers blokkintervall for forskjellen mellom kjent pause og ingen gap er +0,01825 til +0,04153. D1 +1 440 har ingen observerte endepunkter uten kildegap og kan ikke gi tilsvarende kontrast.

Dette er en lokalisering av H4s tap rundt daglig markedspause, ikke bevis for at selve pausen eller en bestemt modellmekanisme forårsaker det: tidspunkt, målfordeling og markedsforhold følger samme gruppe. Den framtidige evalueringen må derfor rapportere gapkryssing separat, samtidig som H4s øvrige tap og de gapfrie M5/M15/H1-tapene fortsatt må løses. [Gapstratifisering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_LEAD_CLOCK_GAP_SCORE_STRATIFICATION.json) SHA-256 `7748a7ddaba314a6b32e0bb36c4e05cfa14191d22b1ba5ed3a35010d5af26a16`; null modellkall eller fit.

På H4 +240 er gapmønsteret også skilt i de lagrede sannsynlighetsfaktorene, med Markovs øvrige faktor holdt fast. På senere **kjente pausebaner** forverrer modellens FLATT-andel Brier med +0,020207 og betingede retning med +0,012473 mot Markov; full modell gir +0,033055. Uten gap hjelper FLATT-andelen (−0,004003), mens retningen forverrer (+0,007160); full modell gir +0,003288. På TRAIN in-sample var FLATT-faktoren allerede svak over kjent pause (+0,008239), men litt bedre uten gap (−0,000594). Dermed er H4s pausetap ikke bare en retningsfaktor; en eventuell framtidig kandidat må måle begge faktorene i den forhåndslåste pausegruppen. Faktorbyttene er ikke årsaksinngrep og summerer ikke eksakt til full modell. [H4-faktorkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_H4_CLOSURE_ACTIVITY_DIRECTION_FACTOR_REVIEW.json) SHA-256 `bb7ecda847a69ca5990f16d64c623590c9bba21026d08b6dbb5d4f9eacb22dfd`; pooled native-resultat er gjenskapt mot tidligere kvittering innen `4×10⁻¹⁶`.

### H4-pause: faktisk FLATT-andel mot sannsynlighet

På de samme lagrede H4 +240-endepunktene er klasseandelene gjenskapt fra eksakt `issue+240`-gyldighetstid, og modell/Markov-Brier og gruppetall stemmer med den signerte gapkvitteringen innen `1×10⁻¹²`. Tallene er overlappende, beskrivende rader; «modell» og «Markov» er gjennomsnittlig **prognosert** FLATT-sannsynlighet, ikke treffrate.

| Periode/gruppe | Rader | Faktisk FLATT | Modellens FLATT | Markovs FLATT |
|---|---:|---:|---:|---:|
| TRAIN in-sample, uten krysset gap | 38 983 | 5,39 % | 9,04 % | 10,07 % |
| TRAIN in-sample, kjent daglig pause | 3 854 | 39,52 % | 9,13 % | 9,99 % |
| Senere utvikling, uten gap | 14 270 | 3,65 % | 6,86 % | 9,90 % |
| Senere utvikling, kjent daglig pause | 1 640 | 30,61 % | 7,08 % | 10,09 % |

Modellen skiller knapt de to senere gruppene i gjennomsnittlig FLATT-sannsynlighet, selv om observert FLATT-andel er langt høyere når en kjent daglig pause krysses. I januar/februar/mars er faktisk pause-FLATT 36,96/18,29/35,86 %, mens modellens middel er 6,83/6,91/7,51 %. På senere pausebaner er faktisk NED 15,30 %, mens modellen tildeler NED 41,65 % i snitt. Dette konkretiserer H4s aktivitets- og klassefordelingsfeil; det forklarer ikke de gapfrie M5/M15/H1-tapene. Ni TRAIN-rader i den tidligere «uten gap»-kvitteringen mangler M1-kilderad ved issue og er oppgitt separat i den nye kvitteringen; senere H4-grupper har ingen slike rader. [Klassekalibreringskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_H4_CLOSURE_CLASS_CALIBRATION.json) SHA-256 `125288e4a688a0a82ef30ca47493b027ed2086256a7248006f0dbcccc938495b`; null nye modellkall eller fit.

Pauseforskjellen består også **innen samme historisk observerte H4-nåtilstand**, som både modellen og matchet Markov får ved issue:

| Nåtilstand | Senere rader pause / uten gap | Faktisk FLATT pause / uten gap | Modellens FLATT pause / uten gap |
|---|---:|---:|---:|
| NED | 486 / 5 147 | 23,25 / 1,90 % | 6,82 / 6,90 % |
| FLATT | 163 / 886 | 44,79 / 8,58 % | 8,21 / 7,36 % |
| OPP | 980 / 8 114 | 32,24 / 4,28 % | 7,02 / 6,77 % |

De små gruppene med ukjent nåtilstand tolkes ikke. Å blande ulike nåtilstander alene forklarer dermed ikke H4s FLATT-underprognose ved pause. Pausetid, lokal range og øvrig markedsregime følger fortsatt gruppen; dette er ikke et kausalt featureinngrep. Vektet sum av alle nåtilstandsgrupper gjenskaper lagret gruppeantall, faktisk FLATT, modellmiddel og begge Brier-skårer innen `2×10⁻¹⁶`. [Nåtilstandskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_H4_CLOSURE_CURRENT_STATE_FLAT_REVIEW.json) SHA-256 `efe8a2049b6a8f13046ea48f517d47a6d62a8ea63b99948a14a0da6ca8290ec7`; null fit eller modellkall.

Den **fryste** fitoppskriftens modellkilde (`entry_native_direction_research_v1.py` SHA-256 `05c0633f…`) bygger prognosekoordinaten bare av tidsramme-one-hot og normalisert ledetid i veggklokkeminutter. `forward` bruker samme koordinat for alle issue-tidspunkt; den har ingen eksplisitt framtidig åpent-minutt-telling eller publisert kalenderpause i queryhodet. Historisk `session_id`, `hour_sin`, `hour_cos` og `dow_sin` er rutet til sesjonsfamilien, slik at indirekte kalenderlæring **er mulig**. Målkontrakten teller derimot lokale vinduer i *observerte aktive M1-barer* og lar ikke kjent stenging flytte vinduet; en 64-minutters daglig pause gir omtrent 176 aktive minutter innen et +240-veggklokkevindu. Dette er et påvist mål-/query-klokkeskille, men ikke en isolert årsak til svikten.

En egen målkontroll finner på samtlige observerte H4 +240-rader i både pause- og gapfri gruppe at `FLATT` er eksakt lik `local_range < threshold`. På senere pausebaner er median lokal prisbredde 8,635 USD mot 13,615 USD uten gap; median terskel er nær lik, 6,877 mot 6,908 USD. Den store FLATT-andelen er dermed faktisk forankret i målregelens lokale prisbredde, ikke en feilaktig FLATT-tildeling eller en stor medianforskyvning i ATR-terskelen. Dette sier fortsatt ikke om 0,5 ATR er riktig brukergrense.

En **retrospektiv** join mot den uavhengig publiserte, regulære dagskalenderen fanger alle 1 640 senere sertifiserte H4-pausebaner. Ujustert kalenderinterseksjon tar også med 47 randrader: hvert querytidspunkt er nøyaktig ett minutt etter publisert pausestart, men er gyldighetstiden for siste observerte M1-bar før kildegapet. Med M1-gyldighetstidens ettminuttsforskyvning blir det 1 640/1 640 og ingen ekstra på disse observerte endepunktene. En framtidig kalenderfeature må følge den samme kausale gyldighetstidskontrakten og dokumentere hva som faktisk var publisert ved issue; den må ikke kopiere etterpå kjent kildeklokke. [Mål-/kalenderkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_H4_CLOSURE_TARGET_RULE_CROSSCHECK.json) SHA-256 `0931d5cad655fd898d05b95dd911cfe4a872ede8dbd2f59393c82a76a55a6b01`; ingen modellkall eller fit.

En direkte, parameterfri kontroll av **bare aktiv-minuttklokken** endrer den fryste Markov-referansens H4 +240-bane fra 240 til 176 planlagte åpne M1-steg på alle observerte kjente pausebaner, med samme TRAIN-overgangsmatrise og samme historiske nåtilstand. På senere 1 640 pausebaner går Markov-Brier fra 0,685745 til 0,683851, mens gjennomsnittlig FLATT knapt flytter seg fra 10,0924 til 10,0942 % mot faktisk 30,61 %. TRAIN in-sample går fra 0,780418 til 0,778838; gapfrie grupper forblir 240 steg og uendret. Modell- og opprinnelig Markov-Brier er gjenskapt eksakt mot den lagrede kvitteringen. Ren omregning av veggklokke til aktiv tid **løser derfor ikke** den målte H4-kalibreringen. En eventuell betinget pause-/regimehypotese må vurderes separat; dette er ikke et årsaksinngrep i modellen. [Aktiv-minutt-kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_H4_ACTIVE_MINUTE_MARKOV_COUNTERFACTUAL.json) SHA-256 `4976853286ec74386ec1e585c0c5d9b7e28f3e996f092e501518cc013be5daa3`; null fit/modellkall.

Den samme klokkeutskiftningen er nå kontrollert ved **én forhåndsdefinert native ledetid på hver av de fem tidsrammene**. På senere observerte endepunkter har M5 +5, M15 +15 og H1 +60 null planlagte lukkede minutter; deres modell−Markov-Brier blir uendret **+0,004838/+0,003882/+0,003640**. For H4 +240 går gapet +0,006346→+0,006541, og for D1 +1 440 går det +0,021423→+0,020951 når samme fryste Markov tar aktive i stedet for veggklokkesteg. Alle fem senere gap er fortsatt ugunstige. Dette er endepunkter, ikke hele bevegelsesbaner; D1 har ikke en gapfri sammenligningsgruppe. Signerte modell- og opprinnelige Markov-tall stemmer innen `6×10⁻¹⁵`. En ren klokkereparasjon av referansen er derfor ikke forklaringen på manglende merverdi. [Fem-TF-klokkekvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_NATIVE_ACTIVE_CLOCK_MARKOV_FIVE_TF_REVIEW.json) SHA-256 `d112c18eeb52291c4af5c3c200f3e9b6bc2f83ca6f59eb58c7dbce28470285b1`; ingen modellfit eller native kall.

En eventuell kalenderkandidat må bruke bare på forhånd publiserte åpningstider som var tilgjengelige ved issue, og gi hodet forventet aktiv tid og pausekryssing per framtidsquery. Den må ikke bruke den etterpå sertifiserte kildeklokke-klassifiseringen som framtidig input. Fastsett mål og strengt kronologisk TRAIN-OOF før en slik énkandidatstest; sammenlign H4 med/uten kjent pause og alle fem tidsrammer mot samme fryste referanser. Ingen ny native kjøring er innrømmet av dette funnet.

## Modellkonsum og loss

Inputmanifester binder full featureprofil, åtte spesialistfamilier, M5/M15/H1/H4/D1 for XAU og GC samt D1/makro-kontekst. TRAIN-normaliseringskvalifiseringen har ingen blokkeringer og minst én observasjon per felt i alle 12 kontrollerte flater. Den viser også sparsomhet: GC-D1 har median observert feltdekning 40,7 %, og 15 av 399 felt er konstante på TRAIN; 72 av 2 542 `ctx_cont`-felt er konstante. Fullprofilens syntetiske kontrakttest krever at hvert fysisk felt er rutet til nøyaktig én familie. Alle 642 aktive parametertensorer endret seg under passet, også spesialist-/MTF-projeksjoner, nåtilstandsprojeksjon og tilstandshoder. Dette viser tilkoblet modellbane, ikke at hver indikator gir prediktiv verdi eller at OOF-featurefordelingen er stabil.

En tidligere fysisk fullprofilrevisjon er gjenbrukt **bare etter identitetskontroll** mot den aktuelle kampanjens inputmanifest: akser, fold, fysisk provider, konstruktør og prefixnormalisering er like; modellens retningskomponent-kilde og kontrakthash er forskjellige. Den målte GC-D1-historikkdekkingen er helt uten overlapp: TRAIN har 0–168 observerte volumhistorikkrader per forelder (snitt 82,89), senere utvikling 169–230 (snitt 199,39). Revisjonen registrerte også 17 sparsomme felt som var observert på TRAIN, men aldri senere. Dette er en **konkret inputstøtteforskyvning** som kan gjøre generalisering vanskeligere. Tid, regime og historikkmengde endres samtidig; ingen lagret prediksjon kan isolere effekten, og resultatet rettferdiggjør verken å fjerne indikatorer eller å kalle dem verdifulle.

Den fryste modellkilden er identisk med kildehashene i fitoppskriften (`entry_v10_ctx_hybrid_transformer.py` `7808095f…`, `entry_native_direction_research_v1.py` `05c0633f…`). GC-felter normaliseres med observasjonsmasker og legges til XAU-familietokenene før transformerblokken; D1-avlesningen er `mean` over **alle 252** kodede posisjoner pluss siste posisjon (modellkilden linje 361–373 og 2552–2573). Det finnes ingen GC-spesifikk nevner basert på antall observerte historikkrader i denne avlesningen. Dermed møter den trente representasjonen en faktisk annen maskestruktur og mulig annen tokenfordeling senere. XAU-data, posisjon og maskefelter er fortsatt til stede, og vektene kan kompensere; dette er en plausibel mekanisme, ikke bevis for at GC-skiftet forårsaket Brier-gapet eller at avlesningen er feil.

Et nytt **lesende** replay av lagret sluttmodell, mål og fryst Markov gjenskapte D1s samlede Brier-gap på TRAIN (+0,007528) og gjenbrukt utvikling (+0,023559) til seks desimaler. Innen senere utvikling var gapet positivt i fire kvartiler etter antall observerte GC-D1-volumhistorikkrader: 169–183: +0,05653; 184–198: +0,02115; 199–214: +0,00451; 215–230: +0,01355. Det **øker ikke monotont** med mer GC-historikk. Januar/februar/mars hadde henholdsvis +0,04816/+0,00797/+0,01287. Dette svekker en enkel forklaring om at økende historikkmengde alene skaper gapet, men kvartilene er beskrivende og valgt etter målt støtte; historiealder og markedsregime følger kalenderen. Ingen kausal effekt eller uavhengig prognoseverdi er isolert. [Kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_GC_D1_SUPPORT_SCORE_REVIEW.json) SHA-256 `b52859791a9ab84509958c612d4a260694728458394965d3361eb1911e36a2b8`.

En separat, på forhånd avgrenset kontroll av **fem faktisk rutede XAU-MTF-cachefelt** per tidsramme bruker nøyaktig siste lukkede native bar ved hver lagret issue-tid. Dette er råverdier før den fryste, monotone TRAIN-normaliseringen. Tabellen viser `trend_state_age_bars` som en tydelig fordelingstest, ikke som bevist årsaksfeature:

| TF | Median TRAIN → senere, issue-vektet | Senere utenfor TRAINs 5.–95.-persentil |
|---|---:|---:|
| M5 | 31 → 29 barer | 4,5 % under / 4,6 % over |
| M15 | 31 → 39 barer | 3,7 % under / 7,1 % over |
| H1 | 25 → 36 barer | 1,9 % under / 17,9 % over |
| H4 | 29 → 49 barer | 1,5 % under / 24,4 % over |
| D1 | 210 → 25 barer | **100 % under** / 0 % over |

D1s TRAIN 5.-persentil er 131 barer og senere maksimum er 57; alle senere issue-rader ligger derfor under denne TRAIN-grensen. De 47 602/17 288 issue-radene gjentar høyere tidsrammebarer, men med hver D1-bar telt én gang er medianene fortsatt 209,5 mot 25,5 på 174/64 unike barer. Andre forhåndsvalgte D1-felt skifter også: `rsi14_centered` median 0,106→0,309, og `atr_bps_14` 141,7→125,4. M5s fem valgte felt har langt mindre medianforskyvning. Dette utvider den målte inputstøtteforskyvningen utover GC-D1-masken; det er **ikke** en full profiltest, ikke et isolert featureablasjonsbevis og ikke en forklaring på M5-tapet. Tid, markedstilstand og målpriorer følger samme periode. [XAU-inputkvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_SELECTED_MTF_INPUT_SHIFT_REVIEW.json) SHA-256 `757ff27003475d1d878ae48c7db76785390556d180fb2982cd77c665360ff436`; fem cachefiler og tidsakser er hashkontrollert, ingen modellkall, fit, TEST eller handel.

Loss er normalisert sekvens-NLL over alle observerte minutter og fem tidsrammer. Dermed fikk M5 bare 12,7 % av de observerte TRAIN-cellene, mens H1 og H4 fikk 23,1 % hver; dette følger hovedsakelig M5s svakere målobservasjon og er ikke en påvist kausal årsak. Første tilstand og første FLAT-aktivitet får egne tillegg med vekt 0,03 hver. Første tilstand kommer fra eget hode; nåtilstanden legges til skjult representasjon via en lært projeksjon. Det finnes ingen eksplisitt identitets- eller Markov-forankring som garanterer at enkel fortsettelse bevares. Felles hoder må lære skiftrater som er svært forskjellige på M5 og D1. For fullstendig observerte baner scorer loss overgangene gitt faktisk forrige tilstand; ved ukjente tilstander marginaliserer den disse. Prognosene bruker fremoverrekursjon over modellens egne sannsynligheter. Små overgangsfeil kan dermed akkumuleres gjennom 1 440 steg. Dette er ikke i seg selv en kodefeil. De komplette modellovergangsmatrisene ble ikke bevart, så denne mekanismen kan ikke isoleres fra det lagrede materialet. Dette er en plausibel mekanisme for målt feilkalibrering, men den er **ikke kausalt isolert**; vektendring eller flere pass kan ikke kalles en påvist løsning.

De allerede signerte rapportene lar oss avgrense hvor ofte et **observert tilstøtende prognosepar** faktisk skifter klasse. Dette er overlappende issue×minutt-par, ikke uavhengige markedshendelser eller direkte mål på gradientandel:

| TF | TRAIN-par med skifte | Senere par med skifte | Direkte NED↔OPP av senere skifter | Modale varslede skifter, modell / matchet Markov senere |
|---|---:|---:|---:|---:|
| M5 | 23,15 % | 23,16 % | 54,74 % | 0 / 0 |
| M15 | 10,38 % | 10,28 % | 71,46 % | 0 / 0 |
| H1 | 3,37 % | 3,41 % | 77,53 % | 0 / 0 |
| H4 | 0,96 % | 0,93 % | 81,55 % | 0 / 0 |
| D1 | 0,12 % | 0,13 % | 90,08 % | 0 / 0 |

Over alle fem TF var 5,70 % av TRAIN-parene og 5,36 % av de senere parene skifter. D1s lange bane domineres sterkt av gjentatt tilstand, mens M5 har mange skifter selv om modellen ikke gir et modalt skiftevarsel. At **begge** probabilistiske prognoser har null modale skifter, betyr at modalt treff alene ikke skiller dem; samlet Brier/logloss og skifte-/retningssannsynlighet må fortsatt sammenlignes. Direkte NED↔OPP utgjør flertallet av observerte skiftepar på alle fem TF, så et rent binært skiftmål ville ikke dekke brukerens retningsmål. Den delvis observerte sekvenslossens faktiske gradientfordeling kan ikke utledes fra disse tellingene. [Skifteeksponering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_ADJACENT_CHANGE_EXPOSURE_REVIEW.json) SHA-256 `15ce8a5163901617eaabbc94da4029e5fd44d8a8b04740991f5800a578234045`.

Den fryste TRAIN-Markov-matrisen ble deretter regnet fram uten ny tilpasning og scoret mot **nøyaktig de samme observerte skifteparene** som modellens signerte rapport. Replayet traff alle fem par- og skifteantall eksakt og Markovs allerede signerte treklasse-Brier innen `5×10⁻¹⁵`. Modell minus Markov i binær Brier for «skifte neste minutt» er:

| TF | TRAIN in-sample, ×10⁻⁶ | Senere gjenbrukt utvikling, ×10⁻⁶ |
|---|---:|---:|
| M5 | −23,354 | +123,335 |
| M15 | −15,431 | +32,815 |
| H1 | −3,595 | +0,838 |
| H4 | +0,481 | +0,955 |
| D1 | +7,712 | +6,644 |

Positivt gap betyr svakere modell. Alle senere punktgap er positive, men svært små mot skifte-Brier-nivåene, som er 0,1781 på M5 og 0,00131 på D1. Modellen viser altså ingen målt senere forbedring i denne **binære, tilstøtende skifteskåren**; dette alene sier ikke om den kan rangere *hvilken retning* neste skifte tar. Modellenes felles overgangssannsynligheter ble ikke lagret, så vi kan ikke blokkresample modell–Markov-gapet eller isolere en parameterårsak fra denne kvitteringen. [Matchet skifte-Brier](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_MARKOV_CHANGE_BRIER_VS_MODEL.json) SHA-256 `c0c0ced9f1f3e2359f0352b6bccdfdcd28dd444f5c6b3781244d46182dae0a09`.

Kilden gir en mer presis vektingsgrense: for en **fullt observert** bane er sekvens-NLL én første-tilstandsterm og én betinget overgangsterm for hvert senere minutt (forskningskilden linje 681–695). Summen deles på alle observerte celler, mens første tilstand får et eget tillegg på 0,03 ganger middel første-steg-NLL; første FLATT-aktivitet får et separat tillegg på 0,03 (linje 736–764 og 769–789). I de bevarte første/siste TRAIN-batchene er forholdet mellom observerte første-steg-celler og alle celler 65/81 686 = 0,000796 og 10/12 151 = 0,000823. Disse er **øvre grenser** for hvor stor koeffisient middel første-steg-NLL fra fullt observerte baner kan ha i sekvensdelen av akkurat disse batchene; delvis observerte baner kan gi annen gradientvei. Marginalprognosen ved hver senere ledetid avhenger derimot også av startfordelingen. Dette dokumenterer et konkret misforhold mellom den direkte starttilstandsvekten og gjentatt marginal skåring, men viser ikke at en bestemt tapsvekt eller ny arkitektur ville slå Markov.

Den faktiske TRAIN-kvitteringen bekrefter tapsdispatch på to lagrede grensebatcher: rapportert total er `sekvens-NLL + 0,03 × første-tilstand-NLL + 0,03 × første-FLATT-aktivitet-NLL` med avvik under 6×10⁻⁹, og gradientnormen er endelig og positiv. Første batch hadde 81 686 observerte framtidsceller; siste hadde bare 12 151. Tapene 1,0405 og 0,2248 er derfor **ikke** en matchet læringskurve. Den nå utførte kontrollen på identiske 256 TRAIN-foreldre står ovenfor: treklasse-Brier bedret seg på alle fem TF, men M5s betingede retnings-AUC-endring er usikker. Tapsdispatchkontrollen finner ingen aritmetisk feil; ingen av kontrollene beviser at lossvektene eller designet er gode for framtidig retningspresisjon.

Den signerte senere rapporten viser hvor forskjellig den observerte **sekvens-NLL per celle** er fra logloss for modellens **marginale framtidsprognose** på de samme observerte cellene:

| Tidsramme | Sekvens-NLL/celle | Modellens marginale logloss | Markovs marginale logloss |
|---|---:|---:|---:|
| M5 | 0,728311 | 1,015613 | 1,007498 |
| M15 | 0,398291 | 0,992113 | 0,986921 |
| H1 | 0,166258 | 0,965562 | 0,962454 |
| H4 | 0,056917 | 0,868079 | 0,863076 |
| D1 | 0,011399 | 0,678129 | 0,640929 |

Sekvens-NLL bruker observert tidligere tilstand der hele banen er kjent og marginaliserer ukjente mellomsteg; prognosen må derimot føre **modellens egne** sannsynligheter fremover. Betinget banetap og marginal logloss måler derfor ulike ting, og den lavere sekvens-NLL-en kan ikke brukes som bevis for framtidig presisjon. Særlig D1s 0,0114 mot 0,6781 illustrerer dette. Modellen taper mot Markov i marginal logloss på alle fem; tallforskjellen alene beviser verken feil i rekursjonen eller at en annen lossvekt ville løse problemet.

### Tilstandsvarighet er ikke i den treklassede prognosekjeden

Den fryste målbyggeren bærer løpende lave/høye prisekstremer når retningen oppdateres; den fryste prognosekontrakten faktoriserer derimot framtidsbanen som førsteordens overgang fra **bare forrige framtidige klasse**, issue-representasjon og ledeminutt. To mulige framtidsbaner med samme issue, ledeminutt og siste klasse, men ulik tid siden siste skifte, får dermed samme overgangsrad. Det er en konkret uttrykksbegrensning for mål med baneavhengige ekstremer, ikke en påvist kodefeil.

På den **unike M1-målserien** er følgende andel av neste observerte minutt et nytt klasseskifte. Hver trippel har tre sammenhengende M1-gyldighetstider og fullt observerbar lokal bane; «fortsatt» betyr at klassen også var lik for to minutter siden, «ny» at den nettopp var byttet inn. Dette er overlappende tripler, ikke uavhengige hendelser.

| Retningstilstand | TRAIN fortsatt / ny | Senere fortsatt / ny | Senere tripler ny |
|---|---:|---:|---:|
| M5 NED/OPP | 27,27 / 12,73 % | 27,11 / 13,76 % | 3 851 |
| M15 NED/OPP | 10,44 / 3,50 % | 10,42 / 3,67 % | 4 412 |
| H1 NED/OPP | 3,12 / 0,78 % | 3,14 / 0,64 % | 2 046 |

M5-FLATT har motsatt mønster på senere strengt observerbare baner: 29,07 % neste-minutts utgang etter fortsatt FLATT, 36,74 % rett etter inntreden. I den bredere masken med observerte klasser, men uten krav om fullt kjent lokal bane, har M5 og M15 høyere retningsskifterate etter fortsatt tilstand i **alle 24 UTC-timer** med minst 20 tripler av hver type. H4/D1 har få nye tilstander og særlig få umiddelbare neste skifter, så de gir ikke samme presisjon.

Dette viser varighets-/historieavhengighet i de **observerte etikettene**. Det beviser ikke at den betingede prosessen etter alle issue-features er ikke-Markov, eller at dette alene forklarer modellens tap mot en også førsteordens Markov-referanse. En framtidig kandidat bør først måle marginal retning, FLATT og første skifte strengt kronologisk; dersom den målte skiftefeilen krever det, kan en liten varighets- eller latent-ekstremtilstand vurderes med samme fryste referanser. Ingen ny arkitektur eller trening følger automatisk av denne diagnosen. [Tilstandsrekkefølge](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_TARGET_SECOND_ORDER_STATE_HAZARD_REVIEW.json) SHA-256 `7550897c63b7aa4b48c47375d79a1c0fb8bae0bb22ab6e0c01d695af545bd2cc`; [streng banesensitivitet](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_TARGET_SECOND_ORDER_PATH_SENSITIVITY.json) SHA-256 `a6707f3ae5557091dddce36f7d9aa2f18dde791b81fc1246ff689f008b7210e1`; ingen fit/modellkall.

### Første framtidsminutt: modellen bruker lite av tilstandsalderen

På de **samme lagrede modellprediksjonene** er første framtidsminutt kontrollert ved hver femminutters prognosestart. Den historiske klassen ved issue og ett minutt før er kausalt kjent; alle tre M1-tilstander og hele lokale banen må være observerbare. Tabellen viser faktisk andel nytt NED/OPP-skifte og modellens gjennomsnittlige sannsynlighet `1−P(nåværende klasse)` på senere gjenbrukt utvikling:

| TF | Nettopp inntrådt: faktisk / modell (rader) | Fortsatt tilstand: faktisk / modell (rader) |
|---|---:|---:|
| M5 | 15,77 / 6,85 % (799) | 29,07 / 6,41 % (2 446) |
| M15 | 4,89 / 5,71 % (879) | 11,41 / 6,77 % (8 124) |
| H1 | 0,82 / 4,72 % (368) | 3,29 / 6,18 % (12 969) |
| H4 | 0 / 4,10 % (124) | 0,89 / 5,58 % (15 023) |
| D1 | 0 / 3,57 % (18) | 0,12 / 5,16 % (15 938) |

H4/D1s ny-tilstand-grupper er små. M5s modellskår beveger seg svakt **motsatt** den observerte aldersforskjellen; på H1/H4/D1 er nivået for skifte for høyt. TRAIN viser samme kvalitative avvik: M5 faktisk/modell 15,08/7,61 % ved ny retning og 28,77/7,39 % ved fortsatt retning. På M5-FLATT er senere faktisk utgang 36,12 % ved ny og 29,41 % ved fortsatt FLATT, mot modell 9,26/8,41 %. Dette er en feil i den lærte første-stegsfordelingen på målt utvalg, ikke bevis for økonomisk edge eller for en bestemt parameterårsak. Full +1 treklasse-Brier og radantall for alle fem TF er gjenskapt mot signert rapport innen `9×10⁻¹⁶`.

Den eksplisitte nåtilstandskodingen i fryst modell er **tre klasseindikatorer + observert-flagg + sti-observert-flagg**. Den får ikke målseriens eksakte alder eller avstand til løpende ekstrem direkte; den øvrige historiske featureprofilen kan gi tilnærminger. En eventuell liten, kausal alder-/ekstremavstandskandidat må først bindes til faktisk tilgjengelig historikk, sammenlignes mot allerede lagret +1-kalibrering og måles på strengt kronologisk TRAIN-OOF samt lengre ledetider. Flere pass eller bare et +1-offset kan ikke antas å rette den betingede retningen. [Første-minuttskvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_FIRST_MINUTE_STATE_AGE_MODEL_CALIBRATION.json) SHA-256 `03da846483e75b4682c00e6b980a0edcb8150721d0c0e028af3e4358df8007f6`; ingen modellkall eller fit.

### Bekreftet varighet ved prognosestart

For å skille «nettopp inntrådt» fra lengre løp telles alder bare fra et **observert klasseskifte** i en sammenhengende M1-bane. Telling nullstilles ved ukjent klasse, ukjent lokal bane eller hull i M1-gyldighetstid. Tabellen bruker lagrede +1-prognoser ved issue og viser faktisk skifte / modellens gjennomsnittlige skiftesannsynlighet på senere gjenbrukt utvikling:

| Nåtilstand | Alder 1 min | Alder 2–4 min | Alder 5–9 min |
|---|---:|---:|---:|
| M5 NED/OPP | 15,77 / 6,85 % (799) | 26,79 / 6,65 % (1 243) | 32,56 / 6,54 % (344) |
| M15 NED/OPP | 4,89 / 5,71 % (879) | 7,99 / 6,23 % (2 229) | 10,25 / 7,38 % (2 440) |
| M5 FLATT, utgang | 36,12 / 9,26 % (263) | 31,52 / 8,87 % (349) | 25,38 / 7,80 % (130) |

M15 NED/OPP ved alder 10–19 min har 14,97 % faktisk skifte mot 7,11 % fra modellen (1 870 rader). TRAIN viser samme hovedmønster: M5 NED/OPP 15,08/25,41/35,41 % faktisk ved alder 1/2–4/5–9, mens modellen gir 7,61/7,71/7,48 %. Aldersgruppe 1 har **eksakt** rad- og sannsynlighetsparitet med den tidligere «nettopp inntrådt»-kvitteringen på begge perioder og alle fem TF. Dette avgrenser en målt første-stegs kalibreringssvikt; gruppene overlapper i tid, er diagnostisk valgt og dekker bare løp som starter innen kjent bane. M1-pauser forlenger ikke den bekreftede alderen. Funnene viser verken at alder er kausalt manglende input etter alle øvrige indikatorer, eller at en alderkandidat vil forbedre marginal treklasse-skår. [Bekreftet-varighet-kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CONFIRMED_RUN_AGE_FIRST_MINUTE_REVIEW.json) SHA-256 `1e579b17826db25b3e953e6b11b18c8386d575b4d637c4a31293cc7b1f083b5b`; ingen fit, native modellkall, optimizer, TEST eller handel.

### Kontroll innen NED og OPP

Den bekreftede alderen er også delt etter **kjent nåtilstand**, slik at ulik NED/OPP-andel ikke alene kan skape mønsteret. På senere M5 stiger faktisk +1-skifte fra 17,28 til 28,53 % for NED og fra 14,21 til 25,08 % for OPP når bekreftet alder går fra 1 til 2–4 minutter; modellen flytter seg bare 9,26→9,05 % og 4,37→4,28 %. TRAIN har samme retning i begge klassene. M15 er mer sammensatt: ved NED følger modellens 8,36→9,52→11,18 % omtrent faktisk 4,68→8,80→10,72 % over alder 1/2–4/5–9, mens ved OPP er modellens 2,95→3,14→3,95 % for lavt mot faktisk 5,12→7,22→9,82 %. Et generelt påslag for alle tilstander følger derfor ikke av funnet. Aggregert radantall, skifter og modellmiddel har eksakt paritet med forrige varighetskvittering. Dette er fortsatt overlappende, gjenbrukt diagnostikk og kontrollerer ikke alle historiske indikatorer. [Klassefordelt kvittering](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001/SAVED_CONFIRMED_RUN_AGE_BY_DIRECTION_STATE.json) SHA-256 `864c97cc7b97a27595ec5bd59f6c5f34ca49879a9c580ec28887ccb88209cb1f`; ingen fit eller modellkall.

## Samsvar med brukerens prognosemål

Modellen får historisk nåtilstand og genuine MTF-indikatorer og trenes mot framtidige lokale OPP/NED/FLATT-klasser på alle fem tidsrammer. Den bruker ingen fast prislinje ved prognosestart, og reverseringer kan gi flere framtidige retningsskifter. På dette punktet samsvarer målkontrakten med den løpende markedsbeskrivelsen.

Den godkjente 0,5 ATR-grensen er fortsatt en **forskningskandidat**, ikke en dokumentert grense for «bevegelse verdt å fange». M5-retning er ukjent på omtrent halvparten av +1-radene fordi M1-OHLC mangler hendelsesrekkefølge. D1-FLATT er svært sjelden. Loss teller observerte framtidsminutter, men har ingen eksplisitt målverdi for framtidig bevegelsesstørrelse, nytte eller tidspunktet for første relevante skifte. Dermed vet modellen hvilken **proxy** den skal lære, mens full samsvar med den praktiske OPP/NED/FLATT-bestillingen fortsatt er uavklart. Dette er et mål- og datagap; det er ikke bevis for at indikatorene er uten verdi.

## Neste avgrensede beslutningsgrunnlag

De neste portene er **beslutninger før eventuell kodeendring og én ny historisk modellkandidat**. Ingen er en autorisert kjøring nå.

| Port | Konkrete krav | Status |
|---|---|---|
| Mål | Behold 0,5 ATR som kandidat inntil TRAIN-only vurdering av «verdt å fange», M5s retningssensur og D1s sjeldne FLATT; enhver endring krever ny kausal målmaterialisering på alle fem TF. | Åpen |
| Kalender og aktiv tid | Ren aktiv-minuttklokke rettet ikke native tapet på fem TF. Vurder bare en eksplisitt framtidsquery som betinger på publisert, ved issue kjent pause/regime og forventet aktiv tid; behold veggklokkeledetid. Bruk aldri etterpå sertifisert M1-kildegap som framtidsinput. Lås separat H4 pause/uten-pause-skår og full fem-TF-skår før kandidat. | Målt feil, kandidat ikke kodet |
| Retningskandidat | Én forhåndsdefinert endring som direkte trener/scorer marginal treklasse og betinget retning ved fastsatte ledetider, bevarer dynamisk FLATT/OPP/NED og all fysisk input. Ingen ekstra binær-loss-runde som første tiltak. | Ikke vedtatt eller kodet |
| Måling | Strengt kronologisk TRAIN-OOF med prefixnormalisering og samme modellidentitet; full treklasse-Brier/logloss, betinget retningsrangering, binær aktivitet og første utgang/retur på observerte prefikser med eksplisitt sensur ved kildegap, separat per TF, med utfallsdagblokker og forløpsfølsomhet på lange TF. Markov, klasseprior og candlefortegn må også bestemmes fra bare hvert tidligere TRAIN-prefiks før OOF-radene scores; native H4-skår må rapporteres separat for baner over dokumentert markedspause. Gjenbrukt utvikling er bare diagnostikk. | Mangler |
| Kjøring | Høyst én eksakt capped kontroll og ett pass, med trygg omstart før modellsjekk, frosset kilde, forhåndslåst stoppregel og ingen automatisk omprøving. | `training_enabled=false`; ingen plan innrømmet |
| Uavhengig bruk | Faktisk XAU-M1/ATR-mottak og restartparitet før ONLINE. Forseglet TEST forblir stengt; bare vedtatt månedlig forward-journal kan gi ny uavhengig prognoseevidens. | ONLINE/edge ukjent |

1. Behold nåværende modell som forskningsartefakt og fryst TRAIN-Markov som minste prognosereferanse. Ikke start en serie nye runder for å justere mot den gjenbrukte utviklingsperioden.
2. Dokumenter faktisk mottaks- og beregningstid for XAU M1 og ATR samt restartparitet før ONLINE-adgang. Hvis kilden ikke har slike klokker, behold ONLINE-status ukjent.
3. Avklar målsemantikken med en forhåndsdefinert **TRAIN-only målundersøkelse**: hvor ofte hvert MTF-nivå skal være FLAT for ønsket bevegelse, og om finere, faktisk tidsstemplet XAU-data kan avklare M5s ukjente intraminutt-rekkefølge. TRAIN-skalaen viser at M5s 0,5 ATR-grense ofte er høyst to historisk siterte spreads; dette avgjør ikke hvilken bevegelse brukeren vil fange. En ny terskel må gi full kausal rematerialisering; den kan ikke avledes av denne hypotetiske ro-andelen alene.
4. Før en kalibrator kan kvalifiseres, trengs kronologiske TRAIN-OOF-prediksjoner fra samme modellidentitet **med prefixnormalisering tilpasset bare tidligere rader**, en forhåndslåst kalibreringsregel og et senere ubenyttet målevindu. Den lagrede sluttmodellen og dens mellomliggende checkpoints gir ikke dette alene. En ny native modellutforming bør først begrunnes mot den målte feilen i betinget retning: tidsrammespesifikk skiftehazard og retningsfordeling, klasse- og ledetidsavhengig påvirkning fra kjent nåtilstand som bevarer målbar informasjon om senere FLATT, direkte marginal skåring av treklassefordelingen og separat FLATT/ikke-FLATT ved forhåndsvalgte ledetider og eventuelt en TRAIN-fryst Markov-forankret residual. M5-krysset ved +4 er diagnostikk, ingen hard bytteregel; både Markov og nåtilstandsfri prior beholdes som separate fryste referanser. Et ekstra binært FLATT/ikke-FLATT-tap ved senere ledetider er foreløpig en lavere prioritert designhypotese: modellen slår allerede Markov på M5s binære +5-mål, også de retningsukjente tilleggscellene. En slik endring må begrunnes med forhåndslåst TRAIN-OOF-forbedring uten å fylle inn ukjent OPP/NED. Skår FLATT-tilstandsandel og første utgang/tilbakekomst separat. Fastsett aksept for FLAT, UP/NED, skiftesannsynlighet, samlet Brier/logloss og lengre ledetider før eventuell fit. Mål særskilt om et **nytt** lokalt tilstandsskifte varsles før det skjer, over alle issue-rader inkludert ingen-skifte; utvalg bare av rader som senere faktisk skiftet er etterpåklokt. De nå lagrede rapportene har marginale klassesannsynligheter og sekvens-NLL, men ingen komplette overgangsmatriser; en ny engangsscoring må derfor bevare en forhåndsbestemt, begrenset skifte-/hendelsesskår i samme pass hvis dette skal testes uten flere runder. Bevar reverseringer og ingen maksimal holdetid. Dette er designforslag, ikke påvist forbedring.
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
| `SAVED_SELECTED_MTF_INPUT_SHIFT_REVIEW.json` | `757ff27003475d1d878ae48c7db76785390556d180fb2982cd77c665360ff436` | Fem forhåndsvalgte, hashbundne XAU-MTF-cachefelt på eksakt lagrede issue-rader: TRAIN mot senere fordeling og unike native barer. |
| `SAVED_GC_D1_SUPPORT_SCORE_REVIEW.json` | `b52859791a9ab84509958c612d4a260694728458394965d3361eb1911e36a2b8` | Lesende D1-Brier per GC-historikkstøtte, med eksakt pooled paritet mot signert TRAIN-/utviklingsrapport; ingen kausal attribusjon. |
| `SAVED_NATIVE_LEAD_DIRECTION_SCORE_DISTRIBUTION.json` | `5e502219f354d69f76e25435bde8947b502ba52cfdb95870899a57f0fd60b179` | Lagret UP/NED-skårspredning og AUC på én full native tidsramme, TRAIN in-sample mot gjenbrukt utvikling og fryst Markov. |
| `SAVED_NATIVE_TARGET_CLUSTERING_SENSITIVITY.json` | `c6fa51b69c3830536b4d08208b90c12a9d957a8d1d5791edc3593c68cbecc306` | D1s utfallsdager/klasseforløp, AUC ved rad-/dag-/forløpsvekting og native treklasse-Brier med rapportparitet. |
| `SAVED_CANDLE_NORMALIZATION_RANK_PARITY.json` | `e208c7c7b8cf1d16c562ecc0846edfc5ad3a175a1785a3206c2c8d13090924e7` | Eksakt issue-klokke og rå/normalisert candle-AUC på alle fem TF, med kildebundet positiv asinh-skala og uten ranginversjon. |
| `SAVED_IDENTICAL_256_INITIAL_FINAL_DIRECTION_REVIEW.json` | `305b149cfc3a27fe11f05c223dbff12cd15946129ff3e1e7e3dbc133cfa1c87d` | Eksakt samme 256 TRAIN-foreldre før og etter passet: native retnings-AUC, candle-korrelasjon og treklasse-Brier på fem TF; M5 parvis ukeblokkintervall. |
| `SAVED_ADJACENT_CHANGE_EXPOSURE_REVIEW.json` | `15ce8a5163901617eaabbc94da4029e5fd44d8a8b04740991f5800a578234045` | Signerte tilstøtende prognosepar: skifteandel, direkte retningsreversering og modale varsler for modell og matchet Markov. |
| `SAVED_MARKOV_CHANGE_BRIER_VS_MODEL.json` | `c0c0ced9f1f3e2359f0352b6bccdfdcd28dd444f5c6b3781244d46182dae0a09` | Fryst TRAIN-Markov regnet fram mot identiske skiftepar; eksakt målparitet og treklasse-Brier-paritet mot signert rapport. |
| `SAVED_SPARSE_LEAD_ACTIVITY_DIRECTION_SWAP.json` | `a1c1018b62889340c9e7f0ab965caeec17c68e615d5f77f65311f3e04f52b040` | Seks faste ledetider på fem TF: modell/Markov treklasse og faktorbytter; native-kvittering og signert ledetidsrapport gjenskapt. |
| `SAVED_DYNAMIC_STATE_PREFIX_OBSERVABILITY.json` | `21a685f1367b075f8754ea1e8c7ed5bf529808e0a3d38241e1444d6c1b05c264` | Null komplette 1 440-minutters kilde-/målbaner, sammenhengende prefiks og native endepunkt på fem TF; eksakt rapportparitet. |
| `SAVED_ATR_BOUNDARY_TARGET_CHANGE_REVIEW.json` | `dcadb4d747ee98724fc0b0a0947a12d5b9f1ae1f725863f7b7e40bc028b9590f` | Unike observerte minuttpar: tilstandsskifter med/uten ny ATR-grense; ingen M5/M15-konsentrasjon ved oppdatering. |
| `SAVED_TARGET_COVERAGE_GAP_ORIGIN_REVIEW.json` | `d9d846dfd7ef1c052347eaae092fe8b7eae928d81b94234434a61c99e384f0ca` | Fullgrid-partisjon av målobservasjon på fem TF og D1s unike kildegapoverlapp; bundet publisert kalender. |
| `SAVED_NATIVE_LEAD_CLOCK_GAP_SCORE_STRATIFICATION.json` | `7748a7ddaba314a6b32e0bb36c4e05cfa14191d22b1ba5ed3a35010d5af26a16` | Native Brier-gap mot matchet Markov etter kildegap/markedsstenging; H4 per senere måned og beskrivende ukeblokkintervall. |
| `SAVED_H4_CLOSURE_ACTIVITY_DIRECTION_FACTOR_REVIEW.json` | `bb7ecda847a69ca5990f16d64c623590c9bba21026d08b6dbb5d4f9eacb22dfd` | H4 +240 faktorbytte for FLATT og betinget retning med/uten kjent pause, TRAIN og senere; native pooled paritet. |
| `SAVED_H4_CLOSURE_CLASS_CALIBRATION.json` | `125288e4a688a0a82ef30ca47493b027ed2086256a7248006f0dbcccc938495b` | H4 +240 faktisk klasseandel mot lagret modell og Markov ved kjent daglig pause og uten gap; signert antall/Brier-paritet. |
| `SAVED_H4_CLOSURE_TARGET_RULE_CROSSCHECK.json` | `0931d5cad655fd898d05b95dd911cfe4a872ede8dbd2f59393c82a76a55a6b01` | H4-FLATT mot lokal range/terskel og publisert dagskalender med M1-gyldighetstidens ettminuttsrand. |
| `SAVED_H4_ACTIVE_MINUTE_MARKOV_COUNTERFACTUAL.json` | `4976853286ec74386ec1e585c0c5d9b7e28f3e996f092e501518cc013be5daa3` | Samme fryste Markov og H4-endepunkter med 176 planlagte aktive mot 240 veggklokkesteg; opprinnelig skårparitet. |
| `SAVED_NATIVE_ACTIVE_CLOCK_MARKOV_FIVE_TF_REVIEW.json` | `d112c18eeb52291c4af5c3c200f3e9b6bc2f83ca6f59eb58c7dbce28470285b1` | Native ledetid på fem TF med planlagt aktivtids-Markov; signert modell/Markov-paritet, fortsatt ugunstig senere gap. |
| `SAVED_H4_CLOSURE_CURRENT_STATE_FLAT_REVIEW.json` | `efe8a2049b6a8f13046ea48f517d47a6d62a8ea63b99948a14a0da6ca8290ec7` | H4-FLATT-feilen ved kjent pause består innen NED/FLATT/OPP nåtilstand; eksakt pooled skårparitet. |
| `SAVED_TARGET_SECOND_ORDER_STATE_HAZARD_REVIEW.json` | `7550897c63b7aa4b48c47375d79a1c0fb8bae0bb22ab6e0c01d695af545bd2cc` | Unike observerte M1-tripler: neste klasseskifte etter fortsatt kontra nylig inntrådt tilstand, TRAIN og senere. |
| `SAVED_TARGET_SECOND_ORDER_PATH_SENSITIVITY.json` | `a6707f3ae5557091dddce36f7d9aa2f18dde791b81fc1246ff689f008b7210e1` | Samme varighetskontroll med streng `path_observed`-maske; originalt trippelutvalg har eksakt paritet. |
| `SAVED_FIRST_MINUTE_STATE_AGE_MODEL_CALIBRATION.json` | `03da846483e75b4682c00e6b980a0edcb8150721d0c0e028af3e4358df8007f6` | Faktisk +1 skifte mot lagret modell/Markov ved ny og fortsatt historisk tilstand; signert treklasse-skårparitet. |
| `SAVED_CONFIRMED_RUN_AGE_FIRST_MINUTE_REVIEW.json` | `1e579b17826db25b3e953e6b11b18c8386d575b4d637c4a31293cc7b1f083b5b` | Bekreftet sammenhengende tilstandsalder ved issue mot faktisk +1-skifte og lagret modell/Markov, TRAIN og senere; alder 1 har eksakt paritet. |
| `SAVED_CONFIRMED_RUN_AGE_BY_DIRECTION_STATE.json` | `864c97cc7b97a27595ec5bd59f6c5f34ca49879a9c580ec28887ccb88209cb1f` | Bekreftet +1-alder delt innen kjent NED/OPP på M5/M15; eksakt paritet mot samlet kvittering. |
| `SAVED_MTF_LAST_CLOSED_CANDLE_NATIVE_DIRECTION_AUDIT.json` | `d5e4e26e4fb3307f2317cc75bf364458520d2ac6bfe7b56724d49ecb1a11ef26` | TRAIN-valgt candlefortegn mot modell og matchet Markov ved én native ledetid på alle fem TF, med beskrivende ukeblokker. |
| `SAVED_M5_CANDLE_CURRENT_STATE_STRATIFICATION.json` | `1e09c2087d19d66383eaa96abb3aa0a76136f6026577f7fd71ca5e4bc8fd3757` | M5 +5 candlefortegn og modellskår innen kjent/ukjent nåtilstand, med ukeblokkintervall for ukjentgruppen. |
| `SAVED_ROUTED_M5_CANDLE_REVERSAL_AUDIT.json` | `081969b226d06f05c551629a09ac2210bcd62d8bff282cca3a9f9f7f93e0c501` | Lesende +5 retningsrangering fra allerede rutet M5-candleverdi mot lagret modell/Markov, kildefortegn og ukeblokker. |
| `SAVED_NATIVE_LEAD_MONTHLY_DIRECTION_AUC.json` | `479a506e958ac75ccbdef905748953d1c3b5f127d0ee632400739c734f42c413` | Samme native-ledetid og betinget UP/NED-skår delt på issue-måned; overlappende, beskrivende rader. |
| `SAVED_CURRENT_STATE_CONDITIONAL_NATIVE_RANK_REVIEW.json` | `13aa797e6b606217d10d3f86baa16b3dc48a6b034089a40b3febeccbf0a248db` | UP/NED-AUC innen historisk nåtilstand på fem TF, D1 per måned og enkle monotone tid-/GC-dekningskontroller. |
| `SAVED_STATE_GROUP_HORIZON_SCORE_DECOMPOSITION.json` | `62e16783993c640b14b4c0b0f1df271af6837976d90f591caf1b9c472fd9efb2` | Signert Brier-gap per ledeminutt og M5/D1 native-ledetid gruppert etter historisk nåtilstand, med eksakt rapportparitet. |
| `SAVED_NATIVE_AUC_WITHIN_BETWEEN_STATE_REVIEW.json` | `4acc526d424cdd957d279e59efee949762aa7a1bf4142c252b54f9e6f0679d4b` | Eksakt pooled AUC delt i OPP/NED-par innen og mellom historiske nåtilstandsgrupper på M5/D1 native-ledetid. |
| `SAVED_MARKOV_PRIOR_HORIZON_CROSSOVER.json` | `edd8d81b1dfe32c9cba22193937751975b23d969e53a97a94ca391ad0bf5ebeb` | Tre fryste Brier-kurver ved alle ledeminutter; M5 +1–15 bekreftet mot lagrede rader per senere måned. |
| `SAVED_NATIVE_FLAT_CONDITION_AND_BRIER_COMPONENTS.json` | `211f51239fb3504e1e1ef5db55c23c06385f8b72d2842eef4ecdb1a993bac676` | Eksakt native-ledetid Brier-paritet, FLATT etter historisk nåtilstand og klassevis skårgap; månedsdeling på gjenbrukt utvikling. |
| `SAVED_CURRENT_FLAT_PERSISTENCE_CURVE_TRAIN_LATER.json` | `efeadf38f2822b5701d0b70a97343a7bb82380cc793108f4a26a8ea053888c48` | FLATT-tilstandsandel fra +1 til én native ledetid, slutt-TRAIN in-sample mot senere gjenbrukt utvikling; samme-rad-kontroll. |
| `SAVED_CURRENT_FLAT_NATIVE_DIRECTION_RANK_TRAIN_LATER.json` | `d20f06db650329f34e36cbd553ae685072959059bb1652d0acfb8372d84bd174` | Betinget OPP/NED-AUC etter historisk FLATT nå ved én native ledetid, TRAIN og senere gjenbrukt utvikling. |
| `SAVED_CURRENT_FLAT_EXIT_REENTRY_TARGET_PATHS.json` | `d44c2231dd891e9de953b65143f0a987d254cc98d3615c9c09f3385821c5be02` | Målbaner med FLATT nå og ved native ledetid: ubrutt FLATT mot utgang og tilbakekomst, med fullstendighetsgrenser. |
| `SAVED_BINARY_ACTIVITY_LABEL_COVERAGE_NATIVE_LEADS.json` | `963115cac4f3b7859a47f8155985c4305d361c9763a867b3db49b9575a862331` | Observert treklasse mot komplett lokal binær FLATT/ikke-FLATT ved +1 og én native ledetid på alle fem TF. |
| `SAVED_BINARY_ACTIVITY_SCORE_NATIVE_LEADS.json` | `c8bb833a3f0777f362108252a9b2c9dbcbb98aaa7cde8fea3df405ec2a451bb4` | Binær Brier/logloss på samme lagrede mål ved +1/native, all/treklasse/binær-tilleggsgrupper; +1-paritet med signert rapport. |
| `SAVED_NATIVE_BINARY_ACTIVITY_MONTHLY_RANK.json` | `c440f68e4945b2e1318e7e7e252d7054c9a9240b6ad4755d5d1521f7572d7da3` | Binær native AUC/Brier per senere måned og M5-nåtilstandsgruppe; rangsumkontroll og pooled paritet. |
| `SAVED_NATIVE_ACTIVITY_DIRECTION_PROBABILITY_SWAP.json` | `1e33e1b359d254a3533f48869be884bfb9873f43c46ac83aa4071ecc6a24fbc9` | Modell/Markov sannsynlighetsfaktorer byttet én om gangen på identiske observerte native treklasse-rader; Brier/logloss-paritet. |
| `SAVED_TRAIN_THRESHOLD_VS_M1_QUOTED_SPREAD.json` | `69d133047f2e1e2d943aa376b7001fd2c8ebdcfe97b1ff705bad9669b9adf551` | TRAIN-kandidatgrense 0,5 ATR mot siste lukkede M1 bid/ask-spread på samme issue-tid; senere periode kun beskrivende. |
| `DYNAMIC_FULL_FEATURE_COVERAGE_REUSE_002/RESULT.json` | `fa99d1358575e7c9882a698a392dcb4343f7c77ef9ce369ab7ea82c5b2416008` | Gjenbrukt fysisk featurestøtte på 47 602 TRAIN/17 288 senere foreldre; ingen individuell indikatorverdi kvalifisert. |
| `RUNTIME/prefix-fit/window-0002/OOF.json` | `6c03081f9058674681ecb827d1b8235c6e3ac197d9675e5871dcd757d348c63c` | Signert sekvens-NLL og marginal logloss/Brier på de samme senere observerte cellene. |
| `TARGET_AUDIT_005/TRAIN_AUDIT.json` | `5eb6be66a7cc887efcdf891ac5af2a9a2aac61cf7a4661e8021a64a963213771` | TRAIN-only klassefordeling og faktisk 0,5 ATR-størrelse på fem tidsrammer; overlappende celler. |
| `DYNAMIC_MTF_SAVED_CALIBRATION_20261103_001/ATTEMPT_001_FAILURE.json` | `3412f68ef1312abee5bf5870f5c3e5bae252b8ebde8b9f59e36c983938ab8b17` | Første forsøk konsumert etter FP32-summeringsvakt, før scoring. |
| `XAU_M1_NATIVE_2009_20260701_PAIR_WEEKCLOSED_20260927/MANIFEST.json` | `138d2f0819d08086d05c3de2a510279b563e7a269e4600b87332926a35017e39` | Kildekontrakt for historiske OANDA-M1-lys; barstempel og 60 s event-time-regel, ingen fysisk ankomstklokke. |

Tidligere signert kampanjeresultat og første FLAT-per-lead-replay står i [engangspassrapporten](DYNAMIC_MTF_STATE_ONEPASS_RESULT_20261103.md). Kildeeiere: `gx1/contracts/entry_dynamic_mtf_state_targets_v1.py`, `gx1/contracts/entry_native_direction_research_v1.py`, `gx1/contracts/entry_dynamic_mtf_state_report_v1.py` og `gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py`.
