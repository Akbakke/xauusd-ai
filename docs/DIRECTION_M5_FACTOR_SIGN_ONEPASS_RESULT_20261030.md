# Ettpass M5-retningsreferanse: resultat 30.10.2026

På samme gjenbrukte Q1 2025-utviklingsrader traff den fullførte rene retningsmodellen 48,616 % ved fem minutter, mot 51,582 % for alltid opp. AUC var 0,496961. Dette dokumenterer ikke edge i retning eller økonomisk handelsverdi.

Modellen bruker lukkede M5/M15/H1/H4/D1-innganger, åtte spesialistfamilier og D1/makro, uten separat M1-serie eller M1-masker som entryinput. Den har 60 korrigerte retningshoder. Det ble gjennomført ett pass over 47 602 TRAIN-foreldre, 2 976 oppdateringer og scoring av 17 288 senere Q1-foreldre; private kvalifiseringsvekter ble ikke gjenbrukt. Fysisk boot 584, signert ressursvakt og kvitteringskjede er verifisert. Engangsoppgaven er deaktivert. GPU-grensen ble beholdt på 250 W, mens faktisk trekk etter kjøringen var 25,7 W.

Prediksjonsarkivet er byteidentisk med det tidligere M5-price-only-arkivet. Dermed er de tidligere kvalifiserte 10 000 dags-/ukeblokktrekningene og 60 simultane AUC-båndene direkte gjenbrukbare. Det er en reprodusert baseline, ingen ny uavhengig generalisering. Ingen av de 60 horisontene har begge simultane nedre AUC-grenser over 0,5.

På 17 192 identiske, modne, observerte og ikke-null femminuttersutfall hadde den tidligere korrigerte referansen 48,604 % treff og AUC 0,496976. Den gjeldende modellen hadde 48,616 % og 0,496961. Forskjellen i AUC har parrede dags- og ukeintervaller som inkluderer null. Prisfelter og projeksjon/gater ble endret sammen i den eldre sammenligningen; enkeltindikatorers effekt er ikke isolert.

Den allerede fullførte femminuttersvektede modellen traff 50,250 % på samme femminuttersrader, AUC 0,500775, fortsatt under alltid-opp. Femminutters AUC-forskjellen mot denne rene referansen var +0,003814; parrede 98,75 %-intervaller var −0,008555 til +0,016147 på dager og −0,005745 til +0,017847 på uker. Den vektede objektivendringen endret også øvrige horisonters relative vekter. Ingen horisont eller terskel velges fra disse gjenbrukte resultatene.

## Alle 60 framtidige målehorisonter

Tabellen gjelder samme frosne Q1-radsett, med modne, observerte ikke-null utfall per horisont. Antall kan variere mellom horisonter. AUC-båndene er 97,5 % simultane maksimumsavviksbånd innen hver blokkordning; dags-/ukeblokkenes utbyttbarhet er en antakelse. Fem minutter var forhåndsvalgt hovedmål. Alltid opp er en enkel retningsreferanse, ikke en nettoresultatmåling.

| Min | N | Ren treff % | Alltid opp % | 5×10 treff % | Ren AUC | Samtidig dagbånd | Samtidig ukebånd |
|---:|---:|---:|---:|---:|---:|:---|:---|
| 1 | 17211 | 49.980 | 49.695 | 50.044 | 0.499136 | [0.469819, 0.528453] | [0.471472, 0.526799] |
| 2 | 17239 | 49.864 | 50.914 | 50.450 | 0.499827 | [0.470510, 0.529144] | [0.472164, 0.527491] |
| 3 | 17237 | 49.690 | 51.111 | 50.705 | 0.502942 | [0.473625, 0.532259] | [0.475279, 0.530606] |
| 4 | 17253 | 49.719 | 51.301 | 50.658 | 0.501122 | [0.471805, 0.530439] | [0.473458, 0.528785] |
| 5 | 17192 | 48.616 | 51.582 | 50.250 | 0.496961 | [0.467644, 0.526278] | [0.469297, 0.524625] |
| 6 | 17189 | 50.794 | 50.817 | 50.817 | 0.487936 | [0.458619, 0.517253] | [0.460272, 0.515599] |
| 7 | 17192 | 51.408 | 51.326 | 50.012 | 0.506821 | [0.477504, 0.536138] | [0.479158, 0.534485] |
| 8 | 17196 | 51.146 | 51.204 | 50.204 | 0.495583 | [0.466266, 0.524900] | [0.467919, 0.523246] |
| 9 | 17194 | 50.663 | 51.861 | 50.506 | 0.503281 | [0.473964, 0.532598] | [0.475618, 0.530945] |
| 10 | 17133 | 50.762 | 51.912 | 50.978 | 0.496512 | [0.467195, 0.525829] | [0.468848, 0.524175] |
| 11 | 17123 | 50.581 | 51.562 | 50.885 | 0.501697 | [0.472380, 0.531014] | [0.474034, 0.529361] |
| 12 | 17134 | 50.706 | 52.072 | 48.763 | 0.508351 | [0.479034, 0.537668] | [0.480687, 0.536015] |
| 13 | 17137 | 49.945 | 52.010 | 51.351 | 0.506806 | [0.477489, 0.536123] | [0.479142, 0.534469] |
| 14 | 17139 | 51.316 | 52.418 | 52.080 | 0.513002 | [0.483685, 0.542319] | [0.485339, 0.540666] |
| 15 | 17077 | 48.949 | 52.392 | 50.056 | 0.507430 | [0.478113, 0.536747] | [0.479766, 0.535093] |
| 16 | 17071 | 50.706 | 52.282 | 50.706 | 0.513384 | [0.484067, 0.542701] | [0.485721, 0.541048] |
| 17 | 17075 | 50.102 | 52.463 | 49.710 | 0.510389 | [0.481072, 0.539706] | [0.482726, 0.538053] |
| 18 | 17076 | 50.404 | 52.594 | 48.472 | 0.506161 | [0.476844, 0.535478] | [0.478497, 0.533824] |
| 19 | 17080 | 51.013 | 52.611 | 52.465 | 0.513308 | [0.483991, 0.542625] | [0.485644, 0.540971] |
| 20 | 17012 | 51.440 | 52.645 | 49.982 | 0.510734 | [0.481417, 0.540051] | [0.483071, 0.538398] |
| 21 | 17014 | 51.434 | 52.704 | 51.328 | 0.506158 | [0.476841, 0.535475] | [0.478494, 0.533822] |
| 22 | 17017 | 50.902 | 53.047 | 50.914 | 0.513403 | [0.484086, 0.542720] | [0.485739, 0.541066] |
| 23 | 17015 | 52.195 | 52.918 | 51.854 | 0.501725 | [0.472408, 0.531042] | [0.474062, 0.529389] |
| 24 | 17009 | 53.219 | 53.125 | 52.790 | 0.515883 | [0.486566, 0.545200] | [0.488220, 0.543547] |
| 25 | 16954 | 48.313 | 53.120 | 50.513 | 0.507650 | [0.478333, 0.536967] | [0.479986, 0.535314] |
| 26 | 16956 | 50.678 | 53.108 | 51.321 | 0.508982 | [0.479665, 0.538299] | [0.481318, 0.536646] |
| 27 | 16957 | 50.581 | 53.476 | 49.897 | 0.504940 | [0.475623, 0.534257] | [0.477277, 0.532604] |
| 28 | 16952 | 52.377 | 53.386 | 52.130 | 0.512687 | [0.483370, 0.542004] | [0.485023, 0.540350] |
| 29 | 16947 | 50.310 | 53.473 | 49.501 | 0.516652 | [0.487335, 0.545969] | [0.488989, 0.544316] |
| 30 | 16887 | 50.743 | 53.556 | 50.009 | 0.512529 | [0.483212, 0.541846] | [0.484866, 0.540193] |
| 31 | 16889 | 52.904 | 53.479 | 52.028 | 0.508542 | [0.479225, 0.537859] | [0.480878, 0.536206] |
| 32 | 16892 | 52.179 | 53.587 | 50.562 | 0.510198 | [0.480881, 0.539515] | [0.482535, 0.537862] |
| 33 | 16884 | 51.581 | 53.749 | 51.558 | 0.506303 | [0.476986, 0.535620] | [0.478639, 0.533966] |
| 34 | 16889 | 51.862 | 53.692 | 53.893 | 0.514959 | [0.485642, 0.544276] | [0.487296, 0.542623] |
| 35 | 16829 | 51.637 | 53.651 | 50.223 | 0.512624 | [0.483307, 0.541941] | [0.484961, 0.540288] |
| 36 | 16831 | 49.534 | 53.532 | 53.413 | 0.513533 | [0.484216, 0.542850] | [0.485870, 0.541197] |
| 37 | 16828 | 52.104 | 53.999 | 53.952 | 0.519836 | [0.490519, 0.549153] | [0.492173, 0.547500] |
| 38 | 16824 | 51.296 | 53.941 | 50.458 | 0.521267 | [0.491950, 0.550584] | [0.493604, 0.548931] |
| 39 | 16823 | 52.024 | 53.873 | 53.902 | 0.521637 | [0.492320, 0.550954] | [0.493974, 0.549301] |
| 40 | 16761 | 52.628 | 53.905 | 52.604 | 0.525846 | [0.496529, 0.555163] | [0.498182, 0.553510] |
| 41 | 16765 | 50.820 | 53.922 | 53.296 | 0.521948 | [0.492631, 0.551265] | [0.494285, 0.549612] |
| 42 | 16763 | 51.417 | 54.101 | 52.198 | 0.519919 | [0.490602, 0.549236] | [0.492255, 0.547583] |
| 43 | 16767 | 52.890 | 53.951 | 52.037 | 0.511826 | [0.482509, 0.541143] | [0.484162, 0.539489] |
| 44 | 16765 | 51.262 | 54.381 | 52.293 | 0.517269 | [0.487952, 0.546586] | [0.489605, 0.544932] |
| 45 | 16704 | 52.472 | 54.185 | 52.874 | 0.514531 | [0.485214, 0.543848] | [0.486868, 0.542195] |
| 46 | 16697 | 51.848 | 54.201 | 54.207 | 0.516519 | [0.487202, 0.545836] | [0.488856, 0.544183] |
| 47 | 16697 | 50.380 | 54.159 | 48.919 | 0.521428 | [0.492111, 0.550745] | [0.493764, 0.549091] |
| 48 | 16704 | 52.209 | 54.274 | 51.503 | 0.520771 | [0.491454, 0.550088] | [0.493107, 0.548435] |
| 49 | 16695 | 52.393 | 54.429 | 52.842 | 0.517131 | [0.487814, 0.546448] | [0.489468, 0.544795] |
| 50 | 16640 | 51.424 | 54.405 | 52.933 | 0.517706 | [0.488389, 0.547023] | [0.490042, 0.545369] |
| 51 | 16638 | 53.216 | 54.616 | 52.098 | 0.516809 | [0.487492, 0.546126] | [0.489146, 0.544473] |
| 52 | 16638 | 51.424 | 54.418 | 50.457 | 0.514184 | [0.484867, 0.543501] | [0.486520, 0.541847] |
| 53 | 16638 | 52.669 | 54.568 | 51.785 | 0.520023 | [0.490706, 0.549340] | [0.492359, 0.547686] |
| 54 | 16639 | 52.389 | 54.474 | 51.860 | 0.515920 | [0.486603, 0.545237] | [0.488257, 0.543584] |
| 55 | 16574 | 53.011 | 54.706 | 54.024 | 0.522508 | [0.493191, 0.551825] | [0.494845, 0.550172] |
| 56 | 16574 | 51.177 | 54.634 | 53.216 | 0.519985 | [0.490668, 0.549302] | [0.492321, 0.547648] |
| 57 | 16577 | 51.819 | 54.588 | 51.903 | 0.519188 | [0.489871, 0.548505] | [0.491524, 0.546851] |
| 58 | 16573 | 51.795 | 54.565 | 51.638 | 0.516845 | [0.487528, 0.546162] | [0.489181, 0.544508] |
| 59 | 16577 | 52.241 | 54.606 | 49.502 | 0.508427 | [0.479110, 0.537744] | [0.480763, 0.536090] |
| 60 | 16512 | 52.332 | 54.797 | 51.986 | 0.518947 | [0.489630, 0.548264] | [0.491283, 0.546611] |

## Kildekontroll og gjenværende begrensninger

På TRAIN stemte alle 2 796 249 observerte målverdier over de 60 horisontene med de 82 hashkontrollerte originalkildeblokkene for bid-/ask-åpning: observasjonsmasker, retningsfortegn og bps-verdier hadde null avvik. I Q1 stemte 1 015 373 observerte fasitceller mot originalkilden. Dette er historiske lysåpninger; eksakt quotehendelsestid, lokal ankomst, samtidig bid/ask og faktisk fyll er ikke etablert.

RSI14, MACD-linje/-histogram og ret_1 stemmer med uavhengig rekonstruksjon for siste lukkede lys ved de undersøkte Q1-beslutningene på alle fem tidsrammer. En 512-raders stikkprøve bekreftet ret_1 fra rå M5-pris via faktisk signal- og snapshotmatrise og M5-fleroppløsning. Dette dekker ikke alle historiske vinduer eller dokumenterer indikatorenes nytte.

GC-ordrebokgruppen var nominelt gyldig på 47 132 av 47 602 TRAIN-beslutninger mot 17 283 av 17 288 Q1-beslutninger. De 274 ugyldige på 1. mai skyldes at første GC-dag mangler en valgt kontrakt under den kausale regelen om forrige dags volum. Dette er en reell historikkbegrensning, ikke en bevist årsak til manglende edge. Andre ugyldige bøtter har kildens mulige-dårlig-bok-flagg; maskene må ikke svekkes uten kildebevis.

Alle 60 bps-skalaer er uavhengig gjenskapt fra TRAIN, og den private bps-kontrollen er bestått. Neste arbeid er ett separat bundet bps-ettpass og én separat videreføring fra denne rene modellens eksakte terminalcheckpoint for å sammenligne mer treningseksponering. Bps-armen legger både til et eget hode og et bps-tapsledd, slik at disse to endringene ikke isoleres innbyrdes. Før et nytt treningsvindu kreves maskindekkende idle-/skrivekontroll og fysisk restart. TEST, Exit, broker, ordre, live/paper, spending og promotion er stengt.

## Sporbarhet

- Verifisert resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_FACTOR_SIGN_ONEPASS_20261030_001/ACTUAL_FIT_REVIEW_001.json` (SHA-256 `7cf4f4f2fde38d0eb3ce9e104b21262ee0b39b26ecc0861dffed48b4c3aa24de`).
- Sammenligning med alle 60 paneler: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_FACTOR_SIGN_ONEPASS_20261030_001/REUSED_60_HORIZON_COMPARISON_001.json` (SHA-256 `b6439e09cb12062e268a51e0bc389dca122a357d0a9c3ca3aee79e3fea78a983`).
- Etterkontroll av engangsoppgave og effekt: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_FACTOR_SIGN_ONEPASS_20261030_001/POST_FIT_IDLE_POWER_TASKS_001.json` (SHA-256 `4d61863376a2351c7daa04753b17d5d9fc3b60165f49d03d59f3ecc331ab9242`).
- TRAIN-fasit: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_TRAIN_RAW_DECIMAL_FASIT_AUDIT_20261030_001/RESULT.json` (SHA-256 `349fb7696b70db254efc529e0aeb27e87aecb30971730f7563a33608b0fdfd89`).
- GC-first-day: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_FIRST_DAY_ROLL_LIMIT_REVIEW_20261030_001/RESULT.json` (SHA-256 `3787acf6bb3eeba913885c63a988c523f8647c1aa3d7587557dfec1e0a8c8205`).
