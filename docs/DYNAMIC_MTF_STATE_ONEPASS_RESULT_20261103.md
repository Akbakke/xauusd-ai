# Dynamisk OPP/NED/FLATT: ett historisk pass 3. november 2026

## Konklusjon

Modellen lærer på TRAIN, men gir ingen dokumentert merverdi på senere, gjenbrukte utviklingsdata. På identiske observerbare celler er Brier-skåren dårligere enn en fryst TRAIN-Markov-referanse som får samme eksplisitte historiske nåtilstand. Dette gjelder M5, M15, H1, H4 og D1. Ingen økonomisk edge eller sanntidsklarhet er påvist.

Det er en konkret prognosesvikt: FLAT gjenkjennes ofte ved første fremtidige minutt, men blir raskt nesten aldri modellens mest sannsynlige klasse. På senere data varslet modellen null eksakte fremtidige tilstandsskifter, selv om fasiten inneholder observerte skifter. Skiftetallene nedenfor er overlappende prognosepar, ikke uavhengige markedshendelser. Et slikt modalt resultat alene utelukker ikke probabilistisk informasjon; Brier og logloss mot likt informert referanse gir den bredere kontrollen.

## Eksakt omfang og kjørebevis

- Fryst treningskilde: `13c6bb8f2f33c50806908673fa33accb5edaba63`, branch `work/gx1-current` i `/home/andre2/src/GX1_CURRENT`. Recipe SHA-256 `9ddf77864ff0550cfd3160fa3242e56c1de9df7afcd5acab37ca6e38b648a884`.
- Boot 598: én TRAIN-passering, 47 602 foreldre og 2 976 optimizeroppdateringer på 7 782 sekunder. Boot 599: én scoring, 47 602 TRAIN- og 17 288 senere utviklingsforeldre, null backward/optimizeroppdateringer på 9 100 sekunder. Begge terminalkvitteringer er `PASS`, prosessutgang 0 og `test_data_used=false`.
- Signert vakt målte maksimalt 56/66 °C kjerne/minne og 137,42 W i fit; 57/66 °C og 133,52 W i scoring. Effektgrense 250 W. Windows-oppgaven er deaktivert, og trening er stengt i `NEXT_RUN_POLICY.json`.
- `INITIAL_TRAIN`, `FINAL_TRAIN` og senere `OOF` binder samme målmanifest, kildevisning og fryste TRAIN-prior. Alle lagrede prediksjonsfiler ble hashet på nytt. Kildekoden for både modell og rapport holdt seg fryst under kjøringen.
- Målet er den lokale tilstanden NED/FLATT/OPP med 0,5 kausal Wilder ATR(14), beregnet fortløpende per tidsramme. Oppgang og etterfølgende nedgang bevares som to skifter. Det finnes ingen fast prognose-startpris; 1–1 440 minutter er bare den målte spørrehorisonten, ingen holdetid eller grense for tilstandens varighet.

## Samme 256 TRAIN-foreldre før og etter

Startradene ligger i diagnostisk rekkefølge. Audit matchet alle 256 mot den komplette TRAIN-aksen på foreldrerad og verifiserte identisk issue- og barstarttid. Oppgitt Brier og sekvens-NLL bruker nøyaktig samme modne, observerte mål. Startmålingens modell-, Markov- og priorverdier ble gjenskapt fra lagrede prediksjoner og mål. Lavere er bedre.

| Tidsramme | Brier før | Brier etter, samme 256 | Markov, samme 256 | Sekvens-NLL før → etter |
|---|---:|---:|---:|---:|
| M5 | 0,67952 | 0,61582 | 0,61711 | 1,06877 → 0,72027 |
| M15 | 0,67865 | 0,60716 | 0,60914 | 1,00755 → 0,40035 |
| H1 | 0,68244 | 0,60195 | 0,60487 | 0,98731 → 0,16673 |
| H4 | 0,68636 | 0,55132 | 0,55744 | 0,98262 → 0,06135 |
| D1 | 0,68783 | 0,45877 | 0,44798 | 0,98910 → 0,01153 |

Første og siste batch hadde henholdsvis 1,0405 og 0,2248 i samlet treningstap. Oppskriften binder 55 kildeeiere. Modellen skiftet parameterhash, og siste batch ga ikke-null gradient i alle 134 registrerte parametergrupper. Dette viser faktisk TRAIN-tilpasning og modellkonsum gjennom de bundne spesialist-/MTF-veiene. Det beviser ikke at hver indikator tilfører nyttig prediksjonsinformasjon.

## Senere utviklingsdata

Referansen bruker fryste TRAIN-overgangstall og samme observerte nåtilstand som modellen får, med TRAIN-prior ved ukjent nåtilstand. Brier er aggregert over alle modne, observerte celler på 1–1 440 minutter; lavere er bedre. FLAT-recall er andelen sanne FLAT-celler der FLAT er entydig modal prognose.

| Tidsramme | Modell Brier | Markov Brier | FLAT-recall | Eksakte varslede skifter / observerte skifter |
|---|---:|---:|---:|---:|
| M5 | 0,61688 | 0,61351 | 0,087 % | 0 / 2 029 126 |
| M15 | 0,60531 | 0,60237 | 0,382 % | 0 / 1 775 923 |
| H1 | 0,59157 | 0,58759 | 1,481 % | 0 / 699 014 |
| H4 | 0,53888 | 0,53202 | 3,025 % | 0 / 193 811 |
| D1 | 0,43794 | 0,41438 | 4,914 % | 0 / 27 172 |

Logloss er også dårligere enn Markov på alle fem tidsrammer. På nøyaktig de samme 17 288 senere foreldrene forbedret den nye modellen Brier mot 2. november-varianten på M15/H1/H4/D1, men forverret M5 marginalt; ingen tidsramme slår den nå likt informerte Markov-referansen. Den gamle modellen hadde null modal FLAT-recall over banen; den nye er fortsatt nær null.

Første minutt har derimot FLAT-recall 57,4/86,9/97,0/98,5/95,2 % på M5/M15/H1/H4/D1. Allerede ved 5 minutter er M5-recall 0 %, ved 15 minutter M15 0 %, ved 60 minutter H1 0 %, og ved 240 minutter velger modellen aldri FLAT på noen av de fem tidsrammene. M5-modellens gjennomsnittlige FLAT-sannsynlighet ved +5 minutter er 11,2 % når målet er FLAT og 10,4 % ellers; ved +15 minutter er begge omtrent 10,4 %. Den selvstendige første-minutt-FLAT-aktivitetsscoren er bedre enn Markov på M5/M15/H1, men gjelder bare komplette lokale vinduer og kan ha utvalgsskjevhet. Den løser ikke den senere tilstandsprognosen.

## Tolkning og neste beslutning

Mål-, input-, modell-, loss- og rapportkontraktene henger teknisk sammen, og TRAIN-forbedringen avviser en helt frakoblet treningsbane. Prognosen lærer i stor grad kortsiktig nåtilstand og vanlige fortsettelser; den skiller senere FLAT fra ikke-FLAT svakt og gir ikke en nyttig modal skiftprognose. Den tette sekvenslossens normalisering over observerte celler og de to første-minutt-tilleggene på 0,03 kan favorisere vanlige fortsettelser. Dette er en *hypotese om årsak*, ikke en isolert ablasjon eller bevist kodefeil. M5 har også mange sanne skifter; null treff kan ikke forklares bort som bare D1-klasseubalanse.

Før en eventuell ny modell må faktisk mottakstid og beregningstid for siste M1-lys og ATR dokumenteres. Det historiske event-time-målet viser ikke at informasjonen fantes i sanntid. Deretter bør en ny forhåndsdefinert design eksplisitt modellere sannsynlighet for tilstandsskifte og neste tilstand på alle fem tidsrammer, med TRAIN-bestemte vekter for skifthendelser, kalibrert mot samme Markov-/priorreferanse og uten fremtidige input. Den skal fortsatt telle begge retninger ved reversering og bevare FLAT. Slike endringer krever egen kvalifisering og ny eksakt autorisasjon før en eneste ny passering. Ingen automatisk serie med runder er åpnet.

Januar–mars 2025 er gjenbrukt utviklingsdata, og overlappende minuttceller er ikke uavhengige episoder. TEST er forseglet. Ingen Entry-handler, kostnader, gevinst, Exit, broker, paper eller live er evaluert.

## Bevarte bevis

- Rot: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261103_001/FIT_ONEPASS_001`
- `RUNTIME/receipts/invocation-0001.json` og `invocation-0002.json`: signerte terminalkvitteringer.
- `RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json`, `window-0002/FINAL_TRAIN.json`, `window-0002/OOF.json` og `window-0002/RESULT.json`: resultater og bindinger.
- `ACTUAL_TERMINAL_REVIEW_001.json`: samlet terminalstatus, oppgave og verifiserte resultatbindinger.
- `SAVED_ONEPASS_REVIEW.json` SHA-256 `91f980dac2ac1c14af17bea1a5ffcef717c97b6833f881b42a2e7ece36d306de`: likt 256-utvalg, full prediksjonshash og rapportgjenspilling.
- `SAVED_OOF_FLAT_BY_LEAD.json` SHA-256 `b3f8dfa7b779e8ffcb685542110d46f51cb7e6ebaa06e47b1166d153b6ef138b`: lagrede FLAT-prognoser på seks valgte ledetider.
