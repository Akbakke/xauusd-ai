# GX1: hvorfor det historiske tilstandspasset ikke slo Markov

## Beslutning

Ingen ny modellrunde er begrunnet av de lagrede resultatene. Det ene passet lærte på TRAIN, men modellen slo ikke den likt informerte, fryste TRAIN-Markov-referansen på senere gjenbrukte utviklingsdata for M5, M15, H1, H4 eller D1. Feilen er synlig allerede ved første fremtidige minutt på TRAIN: modellen lærer ikke de observerte, tidsrammespesifikke skiftratene like godt som en enkel referanse med samme historiske nåtilstand. Ingen enkelt kodefeil eller sikker årsak til dette er påvist. Økonomisk edge er ikke evaluert.

Arbeidet her er bare lesning av bevarte prediksjoner, mål, checkpoints, inputbindinger og kilde. Ingen modell ble konstruert eller kjørt, og ingen trening, scoringkampanje, TEST, Exit, broker eller handel ble startet. Fryst treningskilde: `13c6bb8f2f33c50806908673fa33accb5edaba63`; dette dokumentet vurderer resultatene fra ett pass, ikke et nytt eksperiment.

## Matchet resultat og hvor svikten oppstår

Modell og Markov ble vurdert på identiske modne, observerte målceller. Begge fikk samme historisk observerte nåtilstand; ved ukjent nåtilstand brukte referansen bare fryst TRAIN-prior. I 17 288 senere utviklingsforeldre var modellens Brier minus referansens Brier positiv på alle tidsrammer (lavere er bedre):

| Tidsramme | Modell − Markov Brier | Beskrivende 95 % intervall over 14 UTC-ukeblokker |
|---|---:|---:|
| M5 | +0,003365 | +0,001397 til +0,005417 |
| M15 | +0,002943 | +0,001258 til +0,004853 |
| H1 | +0,003984 | +0,001611 til +0,006771 |
| H4 | +0,006860 | +0,004104 til +0,009835 |
| D1 | +0,023559 | +0,005726 til +0,041588 |

Logloss var også dårligere på alle fem. Brier-gapet var positivt i hver av januar, februar og mars 2025 for hver tidsramme. Ukeblokkintervallene er beskrivende: minuttsmål og foreldre overlapper, og perioden er brukt til utvikling før. De er ikke ny, uavhengig validering.

Første-minutt-prognosen avslører en mer presis feil enn samlet skår. Tabellen viser sannsynlighet for **skifte bort fra den kjente nåtilstanden**, aggregert over dennes tre klasser. Tallene i parentes er faktisk skiftrate. Her brukes lagret TRAIN- og utviklingsprediksjon, ingen ny forward.

| Tidsramme / nåtilstand | TRAIN: modell / faktisk | Senere: modell / faktisk | Fryst Markov |
|---|---:|---:|---:|
| M5 NED | 0,100 / 0,230 | 0,087 / 0,237 | 0,220 |
| M5 FLATT | 0,060 / 0,316 | 0,088 / 0,328 | 0,311 |
| M5 OPP | 0,039 / 0,226 | 0,038 / 0,221 | 0,212 |
| D1 NED | 0,104 / 0,002 | 0,092 / 0,001 | 0,001 |
| D1 OPP | 0,028 / 0,001 | 0,030 / 0,001 | 0,001 |

Modellen er altså for treg til å skifte på M5 og for rask på D1 allerede på TRAIN. Det er ikke bare et sjokk i senere data. På de identiske 256 TRAIN-foreldrene gikk første-minutt-Brier fra 0,65861 til 0,52451 på M5, men referansen fikk 0,46718. D1 gikk fra 0,66433 til 0,01368; referansen var nær null i dette lille utvalget uten observerte D1-skifter. Dette viser reell tilpasning som fortsatt bommer på enkel korttidskalibrering; 256-utvalget er ikke en egen generaliseringstest.

Den fryste Markov-referansen slo også modellens **binære skiftesannsynlighet** på matchede fremtidige nabopar: M5 0,177985 mot modell 0,178109 i Brier, D1 0,001306 mot 0,001313, og tilsvarende små negative gap på M15/H1/H4. Modellen hadde null eksakte *modale* skiftevarsler, men også en optimal sannsynlighetsmodell kan velge «ingen skifte» når ett-minutts skifter er sjeldne. Null modal treff er derfor ikke alene forklaringen; skår mot samme-informasjonsreferansen er avgjørende.

På senere data er modellens FLAT-Brier klart svakere enn referansens på M5/M15/H1/H4, mens DOWN-celler er bedre og UP-celler svakere. Modellens modal FLAT-recall er nær null over hele banen. Ved +5 minutter er M5-recall null, ved +15 M15 null, ved +60 H1 null, og ved +240 er den null på alle fem. M5s gjennomsnittlige FLAT-sannsynlighet ved +15 er omtrent 0,104 både når fasit er FLAT og ellers. Det viser dårlig separasjon av rolig tilstand utover første steg. M5-gapet mot Markov er positivt både når nåtilstanden er kjent (+0,003742 Brier) og ukjent (+0,002957); uobserverbar M5-retning er derfor ikke en tilstrekkelig forklaring alene.

## Mål, datatilgang og klokke

Målet følger kausale lokale opp- og nedbevegelser per M5/M15/H1/H4/D1, inkludert reverseringer og FLAT; det bruker ingen fast startpris, og 1–1 440 minutter er målt spørregrid, ikke en handelsgrense. Den vedtatte terskelen er 0,5 ATR(14) per tidsramme. Kildeinspeksjon fant ingen byttet tidsrammerekkefølge: target, historisk nåtilstand, modellkoordinater og rapport bruker samme M5–D1-orden. Nåtilstand bindes til eksakt foreldrerad og issue-tid, og fremtidige mål sendes til loss, ikke til forward. Dette er kode- og lagringsbevis, ikke bevis for faktisk ONLINE-tilgjengelighet.

M5s mål er selektivt uobserverbart: av 17 288 senere første-minuttsrader var 8 613 retningsklasser observerte (49,8 %). Av de 8 675 ukjente hadde 8 669 flagg for ukjent intraminutt-rekkefølge; flaggene overlapper. Den binære FLAT/ikke-FLAT-aktiviteten kunne likevel avgjøres i 17 284 komplette lokale vinduer. D1 hadde 16 987 observerte første tilstander (98,3 %), men bare 125 FLAT blant 16 136 komplette lokale vinduer (0,8 %). Derfor er M5-retningslæring sterkt sensurert nettopp ved tvetydige M1-barer, mens D1s FLAT-klasse nesten forsvinner i senere data. Ingen rekkefølge inne i et M1-lys skal gjettes for å fylle hullene.

En ren TRAIN-kontroll av komplette lokale vinduer viser at definisjonen av FLAT er svært terskelfølsom: ved dagens 0,5 ATR var lokal ro 7,7 % på M5 og 3,7 % på D1; en **hypotetisk** 0,75 ATR-grense ville gitt 32,6 % og 23,8 %. Dette er bare lokal rekkevidde i første minutt; opp-/nedtilstandene og hele tilstandsbanen må rematerialiseres kausalt for en annen terskel. Tallene godkjenner ingen terskelendring og bruker ikke senere data til å velge en.

Historisk event-time-sjekk av siste M1-lys og ATR er bestått. Faktisk fysisk mottaks-, prosesserings- og restarttid for disse verdiene er fremdeles ukjent; dette stenger ONLINE-bruk, men er ikke påvist som årsak til det historiske Brier-gapet.

## Modellkonsum og loss

Inputmanifester binder full featureprofil, åtte spesialistfamilier, M5/M15/H1/H4/D1 for XAU og GC samt D1/makro-kontekst. TRAIN-normaliseringskvalifiseringen har ingen blokkeringer og minst én observasjon per felt i alle 12 kontrollerte flater. Den viser også sparsomhet: GC-D1 har median observert feltdekning 40,7 %, og 15 av 399 felt er konstante på TRAIN; 72 av 2 542 `ctx_cont`-felt er konstante. Fullprofilens syntetiske kontrakttest krever at hvert fysisk felt er rutet til nøyaktig én familie. Alle 642 aktive parametertensorer endret seg under passet, også spesialist-/MTF-projeksjoner, nåtilstandsprojeksjon og tilstandshoder. Dette viser tilkoblet modellbane, ikke at hver indikator gir prediktiv verdi eller at OOF-featurefordelingen er stabil.

Loss er normalisert sekvens-NLL over alle observerte minutter og fem tidsrammer. Første tilstand og første FLAT-aktivitet får egne tillegg med vekt 0,03 hver. Første tilstand kommer fra eget hode; nåtilstanden legges til skjult representasjon via en lært projeksjon. Det finnes ingen eksplisitt identitets- eller Markov-forankring som garanterer at enkel fortsettelse bevares. Felles hoder må lære skiftrater som er svært forskjellige på M5 og D1. Dette er en plausibel mekanisme for målt feilkalibrering, men den er **ikke kausalt isolert**; vektendring eller flere pass kan ikke kalles en påvist løsning.

## Neste avgrensede beslutningsgrunnlag

1. Behold nåværende modell som forskningsartefakt og fryst TRAIN-Markov som minste prognosereferanse. Ikke start en serie nye runder for å justere mot den gjenbrukte utviklingsperioden.
2. Dokumenter faktisk mottaks- og beregningstid for XAU M1 og ATR samt restartparitet før ONLINE-adgang. Hvis kilden ikke har slike klokker, behold ONLINE-status ukjent.
3. Avklar målsemantikken med en forhåndsdefinert **TRAIN-only målundersøkelse**: hvor ofte hvert MTF-nivå skal være FLAT for ønsket bevegelse, og om finere, faktisk tidsstemplet XAU-data kan avklare M5s ukjente intraminutt-rekkefølge. En ny terskel må gi full kausal rematerialisering; den kan ikke avledes av denne hypotetiske ro-andelen alene.
4. Dersom ny design vedtas, gjør første steg og senere skiftehazard eksplisitt tidsrammespesifikk og sammenlign en TRAIN-fryst Markov-forankret residual med dagens frie hode. Definer loss og aksept for FLAT, UP/DOWN, skiftesannsynlighet, kalibrering og total Brier/logloss før ny fit. Bevar begge retninger ved reversering og ingen maksimal holdetid. Dette er et forslag, ikke endret modellkode eller dokumentert forbedring.
5. En eventuell ny engangskontroll, ett pass og én scoring trenger særskilt autorisasjon, fersk initialbaseline ved endret ONLINE-funksjon og samme ressursvakter. Gjenbrukt utviklingsdata kan bare styre design; forseglet TEST er ikke et tuningsett. Uavhengig evidens kommer fra den vedtatte månedlige forward-journalen, før økonomisk Entry-evaluering kan vurderes. Entry, Exit, handel og økonomisk edge er fortsatt utestet.

## Etterprøvbare bevis

Alle nye revisjoner ligger under `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001` og er kun lesende analyser av bevarte filer. Ingen rådata eller vekter ligger i Git.

| Fil | SHA-256 | Formål |
|---|---|---|
| `SAVED_PARAMETER_DELTA_AUDIT.json` | `84f85da8d8a85ce10011102272909ff43e2eb652b0245f924618911cf168497e` | Eksakt checkpointbinding og 642/642 endrede aktive tensorer. Relativ ratio for nullinitialisert gruppe er udefinert og brukes ikke. |
| `SAVED_PAIRED_SKILL_REVIEW.json` | `51cd62fc09f59186cbe3710bdf08d9f461a2f9041f3d68388164d2b1a9154a60` | Rapportreplay, matchet Brier/logloss, klasser, måneder og ukeblokker. |
| `SAVED_MARKOV_CHANGE_REVIEW.json` | `6c3057cd0aa2deefd3c9f9e0b4c07b319a64286236c23e7b88e9fa9d1f4398ee` | Matchet binær skifte-Brier; modellens felles overgangsfordeling er ikke lagret, modellscore kommer fra signert rapport. |
| `SAVED_FIRST_TRANSITION_TRAIN_VS_DEVELOPMENT.json` | `0fd6f693f9fdd1b2badd72eb68559512fa8134a2cbfb560ce07c3a4f3af960a9` | Første skiftrate på TRAIN og senere data. |
| `SAVED_256_FIRST_STEP_REVIEW.json` | `e31c0a4ea1fe1629f38b8284ce02b9f7431989b89aae244892c3bdd0dd22c911` | Samme 256 før/etter ved +1 minutt. |
| `SAVED_OOF_FIRST_TARGET_COVERAGE.json` | `e646e12b7d1a4f75ccc31bc63873e25023ab08063bb993b7a83235daa43c39c7` | Første måldekning og ukjent-årsaker. |
| `SAVED_TRAIN_QUIETNESS_FEASIBILITY.json` | `3db113d2b02bbd57fab83dda81f21ff4a8ab9a132cf6dd41cb7420c31daf1f1f` | TRAIN-only lokal ro ved gjeldende og hypotetiske terskler. |

Tidligere signert kampanjeresultat og første FLAT-per-lead-replay står i [engangspassrapporten](DYNAMIC_MTF_STATE_ONEPASS_RESULT_20261103.md). Kildeeiere: `gx1/contracts/entry_dynamic_mtf_state_targets_v1.py`, `gx1/contracts/entry_native_direction_research_v1.py`, `gx1/contracts/entry_dynamic_mtf_state_report_v1.py` og `gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py`.
