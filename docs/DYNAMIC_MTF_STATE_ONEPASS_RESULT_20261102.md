# Dynamisk markedsstatus: resultat av én kort trening, 2. november 2026

## Konklusjon

Ingen nyttig prediktiv fordel er dokumentert. Modellen lærte noe på de samme 256 TRAIN-radene, men på observerbare celler i senere utviklingsdata velger den aldri FLAT som entydig mest sannsynlig klasse. På alle fem forhåndsvalgte tidsrammehorisonter slår den heller ikke den faste TRAIN-prioren, som her alltid velger UP. Dette er ikke en test av økonomisk lønnsomhet.

Den gamle modellen lærte 60 fremtidige terminalretninger uten egen FLAT-klasse. Den nye bruker dynamiske tilstander DOWN/FLAT/UP på M5/M15/H1/H4/D1, terskel 0,5 ganger kausal Wilder ATR(14) per tidsramme, og 1–1 440 minutters prognoseledetid. Ledetid er ikke en handelsregel eller maksimal holdetid. Målene kan vise oppgang etterfulgt av nedgang uten en fast startstrek. Kandidatterskelen er ikke bevist økonomisk optimal, og prognoser utover 24 timer er ikke målt eller implementert her.

## Senere utviklingsdata, ett fast horisontpunkt per tidsramme

De tre treffratene for senere periode bruker samme observerbare klasseutvalg på vedkommende horisont; TRAIN-kolonnen viser en annen periode og et annet utvalg. Prior er TRAIN-klassens majoritet, UP; Markov bruker TRAIN-overganger og den kausalt observerte nåtilstanden når denne er tilgjengelig, ellers TRAIN-prioren. Modellen får ikke denne nåtilstanden eksplisitt. Samme prediksjonsrad kan inngå i andre horisonter, og observasjonene er ikke uavhengige.

| Tidsramme | Ledetid | TRAIN modell | Senere modell | Senere alltid UP | Senere Markov | Senere observerte |
|---|---:|---:|---:|---:|---:|---:|
| M5 | 5 min | 43.83 % | 42.37 % | 44.99 % | 44.11 % | 8 980 |
| M15 | 15 min | 45.66 % | 44.06 % | 46.91 % | 43.66 % | 14 659 |
| H1 | 60 min | 48.38 % | 48.06 % | 50.96 % | 50.96 % | 16 251 |
| H4 | 240 min | 52.97 % | 56.52 % | 58.10 % | 58.10 % | 15 917 |
| D1 | 1440 min | 58.98 % | 62.21 % | 67.43 % | 67.43 % | 13 256 |

På +1 minutt er senere modell/Markov-treffrate M5 51.06/70.56 %, M15 60.06/87.51 %, H1 66.36/96.20 %, H4 67.84/99.00 %, D1 66.76/99.89 %. At D1-tilstanden vanligvis fortsetter til neste minutt gjør en god slik referanse naturlig; det er ikke i seg selv en handelsfordel. Markov får et eksplisitt kausalt input som modellen ikke har, så dette er en informativ systemreferanse, ikke en ren arkitekturduell på identiske inputs.

En ekstra kontroll av lagrede +1-minuttsprognoser viser at modellen har noe sammenheng med kjent nåtilstand, men overser FLAT. I senere periode var M5-nåtilstanden FLAT i 1 028 observerbare tilfeller; i 691 av disse var neste tilstand også FLAT, mens modellen aldri valgte FLAT i denne gruppen. Når M5-nåtilstanden var NED, valgte modellen NED i 1 673 av 2 883 tilfeller; når nåtilstanden var OPP, valgte den OPP i 2 165 av 3 135. For disse 1 028 tilfellene var modellens gjennomsnittlige FLAT-sannsynlighet 14,3 %, mens 67,2 % av neste tilstander var FLAT. Gruppene er dannet i etterkontrollen; modellen fikk ikke nåtilstanden som eksplisitt input. Forskjellen er derfor en betinget observasjon, ikke en vanlig ubetinget kalibreringskurve. Dette viser både delvis informasjon i de øvrige inputene og et tydelig hull, men isolerer ikke årsaken.

## Samme 256 rader før og etter

Radidentiteter, beslutningstider og merkingsmasker ble koblet eksakt. Tallene under aggregerer observerte celler over alle 1–1 440 ledetider på de samme TRAIN-radene; lavere log-loss er bedre. Den opprinnelige initialrapporten ble gjenskapt fra lagrede sannsynligheter og target-masker.

| Tidsramme | Treffer før / etter | Log-loss før / etter |
|---|---:|---:|
| M5 | 21.44 / 44.63 % | 1.120 / 1.011 |
| M15 | 21.66 / 45.62 % | 1.119 / 0.994 |
| H1 | 21.55 / 47.27 % | 1.122 / 0.990 |
| H4 | 18.48 / 52.54 % | 1.126 / 0.907 |
| D1 | 15.69 / 56.33 % | 1.131 / 0.797 |

Dette viser målbar tilpasning på treningsutvalget, ikke generalisering eller at alle indikatorer gir nyttig informasjon. På samme 256 falt FLAT-recall fra om lag 70–78 % ved tilfeldig initialisering til omtrent null etter trening; den initielle modellens samlede presisjon og log-loss var samtidig mye svakere. På senere data er FLAT-recall nøyaktig 0 på alle fem tidsrammer. FLAT finnes likevel i målgrunnlaget: på tvers av observerte senere parent/lead-celler er andelen ca. 14,5 % for M5 og 0,78 % for D1.

En separat sammenligning av de lagrede initial- og sluttvektene viser at alle 641 aktive parametertensorer ble endret, og at alle hadde AdamW-tilstand med 2 976 steg. De 179 inaktive tensorene og de uforanderlige bufferne var uendret. Alle 128 vektene i FLAT-raden i starthodet ble endret, og optimizer-momentene i starthodet og overgangshodet var ikke null. FLAT-hodet var altså ikke fryst eller utelatt fra optimizer. Parameterendringer og ikke-null optimizer-momenter beviser ikke at hver indikator bidrar prediktivt. Kontrollen leste lagrede tensorer på CPU uten å konstruere eller kjøre en modell.

## Skifter og tilgjengelighet

Den opprinnelige metoden valgte aldri et skifte som det mest sannsynlige paret av to påfølgende tilstander, men det betyr ikke at skiftesannsynligheten er null eller at den mest sannsynlige marginale tilstandskurven aldri endrer seg. Den følgende ekstra analysen bruker de lagrede marginalene. Utvalget omfatter også kjent nåtilstand til første prognoseminutt, og utelater par der modellens mest sannsynlige tilstand er uavgjort ved et endepunkt. Faktiske skifter betyr faktiske skifter innen dette felles utvalget. Et treff krever at begge tilstandsendepunktene er observerbare og at både fra-/tilklasse og nøyaktig minutt stemmer. Ukjente minutter blir ikke interpolert. Tallene er overlappende prognosemuligheter, ikke antall unike markedshendelser.

| Tidsramme | Faktiske skifter til NED / FLAT / OPP | Varslede modale skifter til NED / FLAT / OPP | Strenge treff NED / FLAT / OPP |
|---|---:|---:|---:|
| M5 | 790 909 / 467 063 / 772 858 | 4 792 / 0 / 3 214 | 483 / 0 / 431 |
| M15 | 767 157 / 254 895 / 755 447 | 7 331 / 0 / 7 127 | 447 / 0 / 524 |
| H1 | 311 286 / 79 263 / 309 083 | 5 972 / 0 / 9 111 | 205 / 0 / 295 |
| H4 | 86 106 / 18 527 / 89 346 | 3 523 / 0 / 8 753 | 69 / 0 / 86 |
| D1 | 13 000 / 1 325 / 12 866 | 4 742 / 0 / 7 735 | 11 / 0 / 8 |

Den opprinnelige timingen krevde et helt observerbart vindu på 1 440 minutter og fikk null gyldige vinduer. Tilleggsanalysen stopper derfor første-skifte-timing ved første ukjente eller uavgjorte minutt og teller separate observerte endepunktpar også senere. Dette gir et begrenset, men faktisk målbart hendelsesbilde; det rekonstruerer ikke intraminutt-rekkefølge eller bevegelser gjennom hull. Tilsvarende additive felt er nå implementert i den ordinære rapportkoden for fremtidige evalueringer, uten å endre de opprinnelige feltene. De bestod 15 syntetiske tester og en sammenligning mot 256 lagrede utviklingsrader; de tidligere rapportene er ikke skrevet om. M5 hadde kjent nåtilstand og minst ett observerbart, entydig innledende prognoseminutt i 7 046 av 17 288 senere rader.

En egen kontroll på unike, påfølgende observerbare M1-endepunkter undersøkte om M5-skiftene stort sett faller sammen med endringer i ATR-terskelen. I senere utviklingsdata skjedde 6 604 av 8 298 slike skifter mens terskelen var uendret. Terskelen endret seg samtidig med 1 694 skifter; samsvar i tid viser ikke at terskelen forårsaket dem, og ukjente endepunkter er utelatt. Dette gir ikke grunnlag for å endre den vedtatte 0,5-ATR-regelen.

Samlet over alle ledetider er 10 597 435 av 24 894 720 mulige M5-celler observerbare i senere periode. Flere årsaksflagg kan overlappe: dokumenterte markedspauser, ukjente hull, intraminutt-tvetydighet og manglende modning må holdes atskilt. De 228 sertifiserte historiske OANDA-gapene er bare brukt som dokumenterte markedspauser; ingen manglende pris er fylt inn og ingen pause er automatisk FLAT.

## Hva vi vet om årsaken

- En konkret asymmetri er at Markov-referansen bruker nåværende dynamiske tilstand og en tilgjengelighetsmaske, mens modellens forward får genuine indikatorer og tidsrammekoordinater, men ikke denne tilstanden eksplisitt. Modellen kan muligens utlede deler fra historikk; resultatet viser ikke at den gjør det godt. Kausaliteten og ONLINE-tilgjengeligheten til et slikt ekstra input må bevises før det brukes.
- Tapet er sekvensens negative log-sannsynlighet delt på antall observerte celler. For en fullt observert sekvens inngår starttilstanden én gang, deretter en mengde overganger betinget på forrige sanne tilstand. Dette kan gi mye læringssignal til vanlige fortsettelser og for lite til riktig starttilstand og nyttige sjeldne skifter. På de lagrede sluttvektene utgjør første framtidstilstands negative log-sannsynlighet henholdsvis 0,21 %, 0,20 %, 0,40 %, 1,10 % og 5,70 % av summert sekvens-NLL for M5 til D1, målt på TRAIN-radene der første minutt var observert. Dette er en dekomponering av lagrede tapsverdier, ikke en måling av gradientandel eller treningsforløpet. Det er en kodebasert hypotese, ikke en isolert årsak bevist av en ablasjon.
- De åtte spesialistfamiliene, fem tidsrammene og D1/makro er bundet i inputprofilen, og private kontrollbatcher hadde målbar gradientkontakt. Det beviser verken at hver indikator har verdi eller at modellen benytter dem prediktivt. Datatilgjengelighet varierer mellom tidsrammer og perioder.

## Kausal nåtilstand ved beslutningstid

Den lagrede nåtilstanden er kontrollert på alle 47 602 TRAIN- og 17 288 senere rader. Når tilstanden er kjent, har både den ferdige M1-baren og ATR-verdien deklarert tilgjengelighet senest ved beslutningstid; ingen ligger etter denne. De tidligere beståtte fremtidsendringstestene for samme målkode er gjenbrukt. Dette er historisk klokke- og kildekontroll, ikke bevis for leveringstid i ONLINE-løpet.

I senere data var M5-nåtilstanden kjent på 8 987 av 17 288 rader (52,0 %), og alle fem tilstander var kjent på 8 284 rader (47,9 %). Av de 8 987 M5-radene hadde 3 202 kjent tilstand, men uavklart lokal rekkefølge i forløpet. Tilstandstilgjengelighet og sti-tilgjengelighet må derfor ha separate masker; UNKNOWN må ikke bli FLAT.

På samtlige 8 987 rader med kjent M5-tilstand var M5-ATR deklarert tilgjengelig nøyaktig ved beslutningstid, og siste M1-bar hadde også lukketid nøyaktig der. Om meldingen faktisk er mottatt og beregnet før modellen beslutter, er ennå ikke dokumentert. En mulig ONLINE-implementasjon må dokumentere denne hendelsesrekkefølgen og videreføre tilstanden trygt gjennom reelle markedspauser og restarter; en historisk tidslikhet alene er ikke nok.

## Tidskilde og mulig startpunkt for videre design

En kontroll av skjema og Parquet-footer uten datarader viser at den brukte XAU M1-kilden har bartid, men ingen egen navngitt mottaks- eller behandlingstid. GC-relaterte mottaksfelter i M5-kildens skjema dokumenterer ikke levering av XAU M1. Andre logger er ikke undersøkt. En avgrenset statisk søkekontroll fant heller ingen kobling til dette dynamiske målet i de 14 Python-filene under execution/replay, og de undersøkte eldre live-adapterne beskriver egne begrensninger. Faktisk ONLINE-rekkefølge og integrasjon er derfor fortsatt uavklart.

En NumPy-beregning med de lagrede, positive TRAIN-overgangstellingene viser at et mulig modellhode med null residual kan gjenskape Markov-referansen for kjent nåtilstand eller UNKNOWN-fallback. Største forskjell mellom FP32 og float64 var 0,00000895 på fem tidsrammer og 1 440 ledetider. Dette er ikke kontroll av faktisk native modellbane, ny scoring, bedre læring eller edge. Den konkrete kontrakten, maskene, mulige tap og uavklarte kvalifiseringskrav står i [designforslaget](DYNAMIC_STATE_CAUSAL_INPUT_DESIGN_20261102.md).

## Konkrete neste tiltak, ikke igangsatt

1. Lag en kausal ONLINE-representasjon av kjent nåtilstand per tidsramme med eksplisitt UNKNOWN/tilgjengelighetsmaske, tilgjengelig før beslutningstid. Gi modell og baseline samme informasjon; behold genuine indikatorer og alle vakter. Kontroller først at målet ikke lekker fremtidig threshold, ekstrempunkt eller observasjon.
2. Revider tapsdesignet slik at riktig tilstand ved starten og marginale sannsynligheter ved forhåndsvalgte ledetider får direkte treningssignal, samtidig som overgangs-/sekvenstap og begge skifteretninger beholdes. Balansér eventuelt klasser som et ekstra treningsmål, men evaluer uvektet log-loss og Brier på naturlige klassesammensetninger og kontroller kalibrering. Disse er egnede sannsynlighetsscorer; se [Gneiting og Raftery, 2007](https://doi.org/10.1198/016214506000001437). Valg av vekter er en hypotese som krever forhåndsdefinert TRAIN-/kronologisk utviklingskontroll.
3. Behold observasjonsmasker, og skaff høyere tidsoppløsning eller bedre dokumentasjon når det er mulig fremfor å gjette rekkefølge gjennom M1-hull. Utvid ledetidsdomenet på en kontrollert måte hvis swing utover 24 timer skal prognostiseres. Evaluer størrelse, varighet, alle relevante skifter og senere entry-økonomi separat før noen påstand om edge.
4. Gjør ingen flere treningsrunder eller fulltrening automatisk. Forseglede TEST-data, Exit, broker, paper og handel forblir stengt.

## Gjennomføring og kontrollgrenser

Én hovedpassering fullførte 47 602 TRAIN-rader og 2 976 optimizer-oppdateringer på 7980.1 sekunder. Resultat-/cursor-/vekttilstand ble lagret, men avslutningsvalidatoren avviste feilaktig det nye manifestet og prosessen fikk exit 1 uten ordinær kvittering. Den opprinnelige planen og ubrukte vinduer ble stengt.
En egen første scoring brukte lagrede sluttvekter: 47 602 TRAIN-rader og 17 288 senere rader, 4 057 forward-kall, 0 backward og 0 nye oppdateringer på 10616.3 sekunder. Full tilstandsforsegling var identisk med sluttvektene. Scoringen publiserte begge rapportene og alle prediksjonsarrayer, men en postpubliseringskontroll ble stoppet av en manglende eksplisitt readguard-binding; også denne prosessen fikk exit 1 uten ordinær kvittering. Ingen modell eller måling ble kjørt om igjen.
Etterpå er den eksakte bundne filadgangen rettet og testet, og den faktiske COMPLETE progress/cursor/result-kjeden er validert under lesevakten uten modellkall. Alle 15 lagrede arrayfiler er verifisert med fil- og datasha, akser og radidentitet. Uavhengig saved-only replay av alle TRAIN-/senere rader på seks faste ledetider gjenskapte de rapporterte modell-, prior- og Markov-målene. Disse etterkontrollene erstatter ikke eller omskriver de manglende native kvitteringene.

Originale artefakter og sjekksummer (ingen rådata eller vekter er lagt i Git):

- `training_integrity`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/FIT_ONEPASS_001/TRAINING_INTEGRITY.json` — SHA-256 `fab5f0d40e01d27e0d91ca0dd43a004e40f40d2cf4c9a711e56cc5be30843a34`
- `scoring_integrity`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SCORING_RECOVERY_001/SCORING_TERMINAL_INTEGRITY.json` — SHA-256 `a19c91b4278b1e6eec0e6038a2074d52db7fca961fbe5a4c42fb9386de0b3b3e`
- `array_verification`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_ARRAY_VERIFICATION_001/RESULT.json` — SHA-256 `52a5d2dd5f8e5137922e2e8922ccfbf4a6252177306d4e1f710fbc6d42d9157b`
- `paired_256`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/PAIRED_INITIAL_FINAL_256_001/RESULT.json` — SHA-256 `1a67033eca5e87f696c7e9114d712877b321132623f9b4c83bd144a82d1ef414`
- `score_replay`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_SCORE_REPLAY_001/RESULT.json` — SHA-256 `de0cd8b95d797740682af5dd6ccaf09dd01e09cdb3460c8378a485cf37da382f`
- `observed_event_review`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_STATE_EVENTS_001/RESULT.json` — SHA-256 `5c7565867480e38d36af98863592900da24bb11e88ffb9a76d52fce602547d45`
- `observed_event_crosscheck`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_STATE_EVENTS_001/CROSSCHECK.json` — SHA-256 `31062c5824a000d657fa49396c35e778bf6f01653f687ac3d21217a2fb423490`
- `postterminal_readguard`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SCORING_RECOVERY_001/POSTTERMINAL_READ_GUARD_CHECK_001/VALIDATION_RECEIPT.json` — SHA-256 `150566c6ada6b731ad993c589e0d0ee030ce79b80d19103c24b2b5b7d51567d6`
- `train_report`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SCORING_RECOVERY_001/RUNTIME/prefix-fit/window-0001/FINAL_TRAIN.json` — SHA-256 `fd229e358db6a05561bfecfdd3e5c13b83b676b55c9bacd1ec8c17e145ab9d0c`
- `later_report`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SCORING_RECOVERY_001/RUNTIME/prefix-fit/window-0001/OOF.json` — SHA-256 `74b698a8067c220e1abed70684a19f7f77e54d71050e511d944bf569cbba82a7`
- `target_review`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/TARGET_AUDIT_005/LABEL_AUDIT_REVIEW.json` — SHA-256 `f31112cee8e672a6d6b52dd2a83da4e71f50c2abf5c460e1dfeab171a507563f`
- `calendar_review`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/OANDA_CALENDAR_DOWNLOAD_001/VERIFIED_RESULT.json` — SHA-256 `83d40f02fbc06e1028ed3e1ed22ed76ec00d43b43914946302f704e131726af8`

- `additive_report_tests`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_STATE_EVENTS_001/ADDITIVE_REPORT_TEST_RECEIPT_001.json` — SHA-256 `8c1625a8bb2005c0444c53a6c08d4b184812675b74cc8415b632471e0edd07ab`
- `saved_event_implementation_check`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_EVENT_IMPLEMENTATION_CHECK_001/RESULT.json` — SHA-256 `a5e841288923c8f52403a55a809ff387d7c008dd8915a8d7045a5a0c06fe8270`

- `current_state_diagnosis`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/CURRENT_STATE_DIAGNOSIS_001/RESULT.json` — SHA-256 `1bf732d8aaa6a28200858477388a5395981651b81b8f06393ca55a6005e7fbd8`

- `causal_current_state_audit`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/CAUSAL_CURRENT_STATE_AUDIT_002/RESULT.json` — SHA-256 `645ea3517c5e34ee683327024e63a27c3dbf983ea0e62d8bfdaf2f2a8992cdf1`
- `causal_audit_first_attempt_failure`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/CAUSAL_CURRENT_STATE_AUDIT_001/FAILURE.json` — SHA-256 `2288b5c51369e51cf27210ffa5639f0470c956b8dd0ee55f893bf3d69f36402e`

- `saved_state_optimizer_audit`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_STATE_OPTIMIZER_AUDIT_001/RESULT.json` — SHA-256 `4262d3a64134992f5fe0c2fa14ba3df1328a2c71691591f5804362578d77c6f3`
- `state_threshold_change_audit`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/STATE_THRESHOLD_CHANGE_AUDIT_001/RESULT.json` — SHA-256 `ea037190de124e71871caa05ca97fe647b7652da9d5e1f640309e195fd9cb49e`

- `saved_loss_decomposition`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_LOSS_DECOMPOSITION_001/RESULT.json` — SHA-256 `d6b9f3ccc9ca5ef8a76e8018cd56376dfff68a2de0d2c569d2746d016033f535`

TEST/Exit/handel er ikke brukt. Senere Q1 2025 er gjenbrukt utviklingsdata og ikke en forseglet blind test. Ingen handelsregel, kostnader eller økonomisk avkastning er evaluert.

- `online_clock_source_schema_audit`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/ONLINE_CLOCK_SOURCE_SCHEMA_AUDIT_001/RESULT.json` — SHA-256 `cb662168d553d0e1a1109416bde5c802daa4f50544177505d1efea97150a3779`
- `baseline_anchor_feasibility`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/BASELINE_ANCHOR_FEASIBILITY_001/RESULT.json` — SHA-256 `8fb74c6913f6018c0692274189c6054794e781ad8d2295a3b21a15c3f3b3344e`
- `online_execution_source_audit`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/ONLINE_EXECUTION_SOURCE_AUDIT_001/RESULT.json` — SHA-256 `9ec6e0861b54d908e4222d226f3191ce992a813b563512c2723652b4fdb066c4`
