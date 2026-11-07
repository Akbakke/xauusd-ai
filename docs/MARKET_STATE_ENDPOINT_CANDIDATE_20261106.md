# Fem-TF markedstilstand ved framtidig endepunkt — kandidat 06.11.2026

**Status: PROPOSED, NOT LOCKED.** Dette er et betinget semantisk forslag før Entry, ikke en valgt fasit, endret kilde/modell, treningsplan eller kjøreautorisasjon. Før valg må en radvis kilde- og maskekontroll vise om dette lokale vindusbildet er en fysisk meningsfull markedstilstand. `training_enabled=false`; TEST forblir forseglet.

## Foreslått mål og klokke

For hver fullført M5-beslutning ved UTC-tid `t` estimeres lokal tilstand `S_TF(v)` ved **ett eksakt framtidig UTC-endepunkt** `v=t+Δ_TF`, der `Δ_TF` er M5=5, M15=15, H1=60, H4=240 og D1=1440 veggklokkeminutter. Dette er ikke avkastning fra prisen ved `t`, en handelsstart eller bevis på at tilstanden varer hele intervallet. Nåtilstanden `S_TF(t)` beregnes med samme definisjon fra informasjon faktisk tilgjengelig ved `t`; alle modellinputs, også lukkede TF-indikatorer, må være kausale da. Fasit ved `v` kan bruke senere pris og siste fysisk tilgjengelige, **lukkede** TF-ATR ved `v`, men ingen av disse verdiene er input ved `t`. Endepunktets M1-bar må finnes på eksakt klokke og være moden før cutoff; et manglende eller umodent endepunkt har ingen fasit.

Bruk de siste `W_TF` faktisk observerte, lukkede M1-barene ved `v`: henholdsvis 5/15/60/240/1440. `R_TF(v)` er lokal high–low-range over M1-rader der hver rads high/low også omfatter forrige observerte close ved sammenhengende minutt og ved dokumentert gjenåpning; etter ukjent gap brukes ny rads open i stedet. Dette er samme range-geometri som i eksisterende målkilde og må etterprøves radvis for en ny versjon. `A_TF(v)>0` er Wilder ATR(14) fra sist fullførte native TF-bar ved eller før `v`. En full lokal historikk må være observert siden kildestart eller siste ukjente kildegap. Eksakt kildekobling, ATR, kalender og cutoff må hashbindes til ny målidentitet.

| Observert prisbevis ved `v` | Tillatt tilstandsmengde |
| --- | --- |
| Fullt `W_TF`, `R < 0,5 A` | `{FLATT}` |
| Fullt `W_TF`, `R ≥ 1,0 A`, konservativ OHLC beviser sluttside | `{NED}` eller `{OPP}` |
| Fullt `W_TF`, `R ≥ 1,0 A`, sluttside ikke bevist | `{NED, OPP}` |
| `0,5 A ≤ R < 1,0 A` | `{NED, FLATT, OPP}` = ukjent |
| Ufullstendig vindu etter ukjent gap, men **bare post-reset** observert range `≥ 1,0 A` | `{NED, OPP}`; ingen sideopprykk |
| Annet ufullstendig vindu, manglende/umoden M1 eller utilgjengelig ATR | `{NED, FLATT, OPP}` = ukjent |

En bred M1-bar kan romme begge retninger i ukjent rekkefølge. En singelside krever det konservative, radvise sluttbeviset i eksisterende `_step_direction`: ved bred bar bare close eksakt på high/low; ellers må en smal bars fortsatte anker og entydige terskelpassering bevise siden. Rangen alene beviser aktivitet, aldri side. Ukjent gap eller utilgjengelig ATR nullstiller sideankeret og den tidlige post-reset-aktivitetssuffiksen. Et tidligere ATR-hull ugyldiggjør ikke i seg selv et senere fullt M1-rangevindu for FLATT dersom ATR er gyldig ved `v`; dette må kontrolleres mot gammel radvis geometri. Ukjent kildegap krever derimot nytt fullt M1-vindu for FLATT. Tidlig post-gap-aktivitet beholder i **denne** kandidaten bare `{NED, OPP}` inntil full vindusstøtte og separat radbevist side foreligger; aggregattall kan ikke promotere den. En dokumentert stengning må være fullt dekket av ekstern eksakt kalender/autoritet: den bevarer observerte barer og gjenåpningshopp, men gir ingen etikett i manglende minutter og teller ikke som vindustid. Delvis dekket eller ukjent gap nullstiller lokal historikk. FLATT krever alltid fullt vindu.

0,5 ATR er brukerens godkjente **kandidat** for minste relevante bevegelse. 1,0 ATR er en hel lokal TF-range og er her et separat aktivitetsbevis; i May–Sep-kontrollen var M5-medianen ca. 3,48–4,29 siterte spreads. Faktorene er ikke valgt fra klassebalanse eller modellscore. Den prøvde historikkavhengige 1,0-inn/0,5-ut-låsen er en mulig **episode**definisjon, men velges ikke som endepunktstilstand: samme lokale range mellom grensene kan da bli FLATT eller AKTIV ut fra fortiden. Skifter, rekkefølge og varighet må måles separat i observerte M1-minutter og ved eksakte UTC-leads, med sensur for OHLC-rekkefølge, stengning, ukjente hull og månedsgrenser. Ingen >50 %-varighetsport eller maksimal holdeperiode legges på et endepunktmål.

## Avledet TRAIN-dekning og åpne porter

May–Sep 2024, 29 849 M5-prognoser per TF. Tallene ble først summert over alle 25 måned×TF-partisjoner fra bevarte 0,5/1,0-semantikk- og stengningsrapporter og er nå kontrollert radvis mot samme kilder. **Dette er fem måneders kandidatdekning, ikke et låst full-TRAIN-mål eller prediksjonsevne.** Eksakt = 0,5-FLATT + 1,0-NED/OPP minus sider uten komplett aktivitetsbevis; delvis = 1,0-aktivitet uten side + separat tidlig post-gap-aktivitetsbevis.

| TF | Eksakt `{FLATT/NED/OPP}` | Andel av 29 849 | Delvis `{NED,OPP}` | Ukjent |
| --- | ---: | ---: | ---: | ---: |
| M5 | 12 404 | 41,56 % | 1 878 | 15 567 |
| M15 | 14 841 | 49,72 % | 468 | 14 540 |
| H1 | 14 896 | 49,90 % | 95 | 14 858 |
| H4 | 14 280 | 47,84 % | 97, derav 34 tidlig | 15 472 |
| D1 | 7 610 | 25,49 % | 906, alle tidlig | 21 333 |

Kilder: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_SEMANTIC_CALIBRATION_20261106_002/RESULT.json` (SHA-256 `bcd9a82ff6e3218cf58539ec0c4d88cdbb80b1e7c497d2991d31aab071033bf7`) og `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_CLOSURE_OBSERVABILITY_20261106_001/RESULT.json` (SHA-256 `bbd2811dbcd04cdf5d3ac01e4f79c30b405cd085c97ce12093648af699203603`). De publiserte ikke en ny radvis målserie. Den separate [radvise kontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_ENDPOINT_CANDIDATE_ROWWISE_20261106_001/RESULT.json) SHA-256 `395e2604029e00b388f0fc97dfa18c58c1ca1f0c8abfe45c399a50af40caccdf` publiserte [tillattemengde-masker](/home/andre2/GX1_RUNS/MARKET_STATE_ENDPOINT_CANDIDATE_ROWWISE_20261106_001/ROW_MASKS.npz) SHA-256 `5de51adef82cdec0bb717564ec5effc02d407f5972570b1ebfd1f8a93ad4a019`. Alle 25 partisjoner matchet de to rapportene, og en uavhengig etterkontroll fant 940 tidlige/ufullstendige AKTIV-rader uten sidepromotering (906 D1 og 34 H4). Målstatus forblir PROPOSED; ingen modellevaluering er gjort.

Den første radvise kontrollen over dekker May–Sep 2024. Den separate [full-TRAIN-kildeautoriteten](/home/andre2/GX1_RUNS/FULL_TRAIN_XAU_CLOSURE_AUTHORITY_20261107_001/RESULT.json) SHA-256 `b58875df44811da3dafde11edbb89347ec49c74d830249f2ed0903a53b502d1c` koblet 591 682 M1-tidsrader til den [publiserte ordinære kalenderen](/home/andre2/GX1_RUNS/XAU_2025_PUBLISHED_CALENDAR_EXTENSION_20261106_003/PUBLISHED_CALENDAR.json): 413 av 508 observerte hull er fullt inne i publiserte stengninger, 95 er UKJENT. Den nye [full-TRAIN-kandidatkontrollen](/home/andre2/GX1_RUNS/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001/RESULT.json) SHA-256 `3a508ce45755c8db2ad8c078d56a6afef8b88a6af3e0a0b159ca388def38ebc5` publiserte [radvise masker](/home/andre2/GX1_RUNS/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001/ROW_MASKS.npz) SHA-256 `ad0e7d56c58b463bdd3d61fe6b2fcf2b206afa56e899dfb36a850e63c5ec3020` for 118 706 M5-utstedelser × fem TF over May 2024–Dec 2025. Alle 100 måned×TF-partisjoner og kildereferanser besto uavhengig kontroll; det var 0 avvik på 131 323 eksakt sammenlignbare celler fra den gamle May–Sep-kontrollen. Dette er **kilde- og maskedekning**, ikke låst mål eller prediksjonsevne. Bare hull fullt inne i kildebundet stengning er kjente; øvrige kildepauser forblir UKJENT. Ingen senere prisrad eller fold brukes til å velge terskler.

| TF | Eksakt av 118 706 | Andel | Delvis `{NED,OPP}` | Ukjent |
| --- | ---: | ---: | ---: | ---: |
| M5 | 47 852 | 40,31 % | 7 659 | 63 195 |
| M15 | 57 691 | 48,60 % | 1 557 | 59 458 |
| H1 | 56 471 | 47,57 % | 316 | 61 919 |
| H4 | 53 967 | 45,46 % | 252 | 64 487 |
| D1 | 39 042 | 32,89 % | 1 537 | 78 127 |

Det meste av M5-UKJENT kommer fra **måldefinisjonens** gråsone `0,5 A ≤ R < 1,0 A`: 62 687 av 63 195 (99,2 %); bare 471 mangler eksakt endepunkt og 37 har annen ukjent støtte. På M15/H1/H4 utgjør gråsonen 57 992/56 316/54 197 ukjente rader. På D1 fordeler 78 127 UKJENT seg på 42 008 i gråsonen, 25 243 uten eksakt endepunkt og 10 876 med annen uavklart støtte. Dette er en mål- og observerbarhetskostnad, ikke mangel på indikatorer eller bevis på at markedet mangler retning. Tersklene flyttes ikke etter disse tallene uten et nytt forhåndsbundet målvedtak.

D1-eksaktdekningen varierer fra 7,8 % i august 2025 til 64,2 % i oktober 2025; senere resultat kan derfor ikke tolkes uten månedlig dekning. En separat capped kontroll av bare TRAIN M1-tidskolonnen fant at alle 39 042 eksakte D1-vinduer inneholder allerede kjente barer ved `t`, typisk 64 av 1 440 (4,44 %). Typisk veggklokkespenn er 1 504 minutter, og 486 eksakte vinduer som krysser en dokumentert helg spenner typisk 4 448 minutter. Dette er **tilstand ved `v`**, ikke bevegelsen fra `t` til `v`; overlappen alene forklarer ikke 50/50-retning. Retningssiden i dagens måleier bruker et fortløpende anker som kan være eldre enn `W_TF` og nullstilles først ved ukjent gap eller manglende ATR. Den fysiske M5-kontrollen nedenfor avklarer lokal side for eksakte 1,0 ATR-retninger; alderen og verdien av kausal nåtilstand som prognosereferanse, andre masker og øvrige TF er fortsatt åpne.

## Alternativ med én forhåndsbundet 1,0 ATR-grense

En separat, **ikke valgt** kildekandidat bruker samme eksakte klokke, konservative sidebevis og hullregler, men kaller et fullt vindu `R < 1,0 A` FLATT og `R ≥ 1,0 A` AKTIV. [RESULT](/home/andre2/GX1_RUNS/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001/RESULT.json) SHA-256 `1bad8a8a88d61a67ee2487e387c246da06f4f4ccf319e7edd8ce35d39d13eba1`, [radvise masker og kalibrering](/home/andre2/GX1_RUNS/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001/ROW_MASKS_AND_RATIOS.npz) SHA-256 `35257429efa4ec553293e1b10bec0993c3c494f04e7f5be72d55877112e590e5`, og [TERMINAL](/home/andre2/GX1_RUNS/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001/TERMINAL.json) SHA-256 `9ce11435f5be8174dd629c5f6c564f8a945cc432fb037afed14de47222798295` dokumenterer én capped, full-TRAIN kildekontroll med exit 0, 0 modellkall og 0 optimizersteg. De 591 682 M1-radene hadde ingen ugyldige/kryssede eller null-spread bid/ask-close-sitater i denne kilden. Kontroll av sitert spread er fysisk målskala, **ikke** utførbar nettoøkonomi.

| TF | Eksakt av 118 706 | Delvis `{NED,OPP}` | Ukjent | NED / FLATT / OPP blant eksakte | Månedsmedian for `1,0 ATR / sitert spread` |
| --- | ---: | ---: | ---: | ---: | ---: |
| M5 | 110 539 (93,12 %) | 7 659 | 508 | 18 921 / 70 869 / 20 749 | 2,97–6,83 |
| M15 | 115 683 (97,45 %) | 1 557 | 1 466 | 21 375 / 70 272 / 24 036 | 5,24–12,43 |
| H1 | 112 787 (95,01 %) | 316 | 5 603 | 20 265 / 68 152 / 24 370 | 10,76–25,60 |
| H4 | 108 164 (91,12 %) | 252 | 10 290 | 20 776 / 62 072 / 25 316 | 21,66–49,78 |
| D1 | 81 050 (68,28 %) | 1 537 | 36 119 | 14 910 / 44 417 / 21 723 | 61,36–125,75 |

En uavhengig kontroll av lagret RESULT/NPZ bekreftet alle 100 måned×TF-partisjoner, forholdstall, gyldighetsmasker, uendrede endepunktsrader og at alle tidligere ikke-UKJENT-masker er bevart. I alt **273 200 gamle gråsoneceller ble FLATT**, uten gjettet side. På M5 er tallet 62 687. Hele TRAIN gir M5-median `1,0 ATR / sitert spread` 4,164 (bare 5,67 % ≤2 spreads), mot 2,082 og 46,24 % ≤2 ved 0,5 ATR. Dette er sitatkvalitet og fysisk skala, ikke gjennomførbar økonomi; den uavhengige kontrollen regnet ikke rå bud/tilbud eller ATR på nytt.

De 62 687 M5-gråsonecellene som 1,0-varianten kaller FLATT hadde selv median lokal high–low-range **3,144 siterte spreads**; 82,71 % var minst 2 spreads og 30,69 % minst 4. Samlede TRAIN-medianer for M15/H1/H4/D1 var 5,672/11,725/24,371/67,113 spreads. Dette er beregnet fra de to hashbundne radfilene som `range/ATR × ATR/sitert spread` bare på gyldige celler. Range er prisbanens spenn, ikke bevist netto retning.

Den gamle hashbundne [0,5 ATR-serien](/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/TARGET_AUDIT_005/STATE_SERIES/SERIES.json) SHA-256 `d2addf8fbb6f9ac9e9ed741bbbd2fe2bfdc07f59553ea28c021ea198d5fbfa8e` dekker den tilgjengelige overlappen fram til 01.04.2025. Kilde- og gyldighetsrader ble koblet eksakt til 1,0-kandidatens gråsoneceller. `Lokal range/(2 × 0,5-terskel)` var identisk med lagret `range/1,0 ATR` på alle koblede rader (maksavvik 0).

| TF | Koblet gammel gråsone | Merket observert NED/OPP ved 0,5 ATR | Andel |
| --- | ---: | ---: | ---: |
| M5 | 34 117 | 22 657 | 66,41 % |
| M15 | 31 180 | 29 506 | 94,63 % |
| H1 | 29 849 | 29 732 | 99,61 % |
| H4 | 28 558 | 28 528 | 99,89 % |
| D1 | 21 612 | 21 597 | 99,93 % |

Disse er **terskelavhengige etiketter fra den gamle eieren**, ikke uavhengig bevis for at 0,5 ATR er meningsfullt, og april–desember 2025 er ikke med i overlappen. Særlig lange TF kan ha eldre retningsanker; den separate lokale sidekontrollen nedenfor gjelder bare eksakte M5-sider ved 1,0 ATR. Tallene viser at 1,0-varianten kaller mange tidligere retningskandidater FLATT. Høyere dekning er dermed endret målbetydning, ikke ny informasjon eller bedre modell. Verken 0,5 eller 1,0 velges etter klasseandel, dekning eller modellscore. Ingen fit, VAL-utvalg, TEST eller handel er gjort.

En ekstra lesende TRAIN-kontroll koblet en nåtilstandsproxy `S_TF(t)` fra eksakt tidligere utstedelse `t−lead` til `S_TF(t+lead)` fra utstedelsen ved `t`. Den krevde samme fysiske M1-kilderad ved `t` og entydig etikett i begge ender; kilderadavvik var 0. Tabellen sammenligner to **faste, ufittede** referanser på *samme* rader innen hver kandidat: behold nåtilstanden og si alltid FLATT. Ulike kandidater har ulike utvalg og skal ikke rangeres ved å sammenligne prosenter på tvers.

| TF | 0,5/1,0: n | Behold nå | Alltid FLATT | 1,0: n | Behold nå | Alltid FLATT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 | 20 514 | 45,66 % | 13,49 % | 104 009 | 53,82 % | 65,63 % |
| M15 | 31 522 | 50,58 % | 15,93 % | 112 861 | 54,58 % | 61,06 % |
| H1 | 29 667 | 46,63 % | 12,54 % | 107 242 | 51,41 % | 59,37 % |
| H4 | 24 951 | 45,57 % | 12,31 % | 97 610 | 45,46 % | 56,70 % |
| D1 | 14 337 | 47,15 % | 3,10 % | 54 640 | 45,51 % | 54,19 % |

I den matchede M5-populasjonen var framtidige NED/FLATT/OPP `8 410/2 768/9 336` med to terskler og `17 062/68 256/18 691` med én terskel. De fire M5-overgangene mellom to entydige sider er identiske i variantene: NED→NED 3 626, NED→OPP 4 370, OPP→NED 4 065 og OPP→OPP 4 197. Blant disse sideparene skifter altså 8 435/16 258 (51,88 %) side fra `t` til `t+5`; dette er overlappende radpar, ikke uavhengige episoder. Én-terskelens økte støtte består i stor grad av FLATT-etiketter, og alltid FLATT slår enkel tilstandspersistens på samtlige TF. Nåtilstandsproxyen er historisk kildekobling, ikke bevist live tilgjengelig modellinput, og denne kontrollen er ikke fit, målvalg, prognosebevis eller økonomisk edge.

## Fysisk TRAIN-kontroll av mellomsonen

Den innrømmede engangskontrollen [RESULT](/home/andre2/GX1_RUNS/MARKET_STATE_GRAY_PHYSICAL_20261107_001/RESULT.json) SHA-256 `0b8392ed9713113c64a251edd8e1daa70521c0738a2995e85b17ef9de639581e`, [TERMINAL](/home/andre2/GX1_RUNS/MARKET_STATE_GRAY_PHYSICAL_20261107_001/TERMINAL.json) SHA-256 `8f83777d03c90794dbe5c534e1647d41ccb9137e3dcd0c9af1f2168105e871fb`, leste 591 682 TRAIN M1-rader og de to hashbundne kandidatemaskene. Capped audit var ferdig på 285,59 s med exit 0, null modell- og optimizersteg, ingen TEST. Den gjenbygde lokal range fra M1 OHLC og sist observerte close ved dokumentert gjenåpning, og fant **eksakt bitparitet** i range/ATR mot lagret forhold på alle 537 909 komplette celler. Det er kildekonsistens for denne formelen og disse dataene, ikke en uavhengig ATR-formelrevisjon.

Netto er close ved eksakt endepunkt minus open i første av de siste `W_TF` faktisk observerte M1-barene. `|netto|/range` beskriver hvor mye av prisbanens spenn som ble beholdt som netto retning. Gruppene er fullt observerte lokale vinduer med range under 0,5 ATR, 0,5–1,0 ATR og minst 1,0 ATR. Alle tall er medianer; hver gråsonecelle beholdes på sin opprinnelige TF og M5-utstedelse.

| TF | Gråsone n | `|netto|/ATR` under / grå / over | `|netto|/range` under / grå / over | Grå `|netto|` i siterte spreads |
| --- | ---: | ---: | ---: | ---: |
| M5 | 62 687 | 0,131 / 0,296 / 0,741 | 0,316 / 0,404 / 0,574 | 1,179 |
| M15 | 57 992 | 0,128 / 0,287 / 0,731 | 0,324 / 0,397 / 0,554 | 2,120 |
| H1 | 56 316 | 0,125 / 0,285 / 0,741 | 0,314 / 0,394 / 0,548 | 4,440 |
| H4 | 54 197 | 0,123 / 0,277 / 0,743 | 0,296 / 0,382 / 0,551 | 8,869 |
| D1 | 42 008 | 0,120 / 0,265 / 0,791 | 0,271 / 0,353 / 0,613 | 22,745 |

Gråsonen ligger mellom de to referansegruppene både i netto bevegelse og retningsandel. M5-gråsonens `|netto|` var 0,216/1,179/3,209 siterte spreads ved p10/median/p90; den månedlige medianen for `|netto|/ATR` lå mellom 0,278 og 0,315. D1 hadde større månedsspredning, 0,188–0,370. Sitert spread er fysisk skala, ikke gjennomførbar nettoøkonomi. M1-close-kryssinger og terminalt antall closes i RESULT er beskrivelser av observerte lukkekurser, ikke intraminuttrekkefølge eller faktisk tilstandsvarighet. Overlappende endepunkter er ikke uavhengige markedsepisoder.

Det er dermed uforsvarlig å velge én 1,0 ATR-grense *bare* fordi den gir høyere dekning: den kaller hele denne mellomsonen FLATT. Et konkret neste målforslag er å la fullt `R<0,5A` bevise FLATT og fullt `R≥0,5A` bevise AKTIV, mens OPP/NED fortsatt krever separat konservativt sidebevis; mellomsonen ville da ha tillatt mengde `{NED,OPP}`. Dette er **forslag, ikke valgt mål**. Fysisk netto er ulik mellom cellene, og verken denne fordelingen eller senere modellscore skal brukes til å tune terskelen. Operatørens valg av betydningen av FLATT/AKTIV avventes; `training_enabled=false` består.

## Lokal M5-side ved 1,0 ATR

En egen, **konsumert** TRAIN-only diagnose gjenbygde eksisterende 1,0 ATR-måleier fra bundne M1/M5/ATR/kildestengninger og sammenlignet hver av de 39 670 eksakte M5-NED/OPP-endepunktene med en kald `_step_direction`-beregning fra bare siste fem observerte M1-barer. [RESULT](/home/andre2/GX1_RUNS/M5_STATE_SIDE_LOCALITY_20261107_001/RESULT.json) SHA-256 `bbda7175c50ad0b0517c04f0cc07aa77e20692762cdc0fa6c78671dcf2539df3` og [TERMINAL](/home/andre2/GX1_RUNS/M5_STATE_SIDE_LOCALITY_20261107_001/TERMINAL.json) SHA-256 `39e66884806bf743252a903c14755b9b12e557fecefad1a45cb56a8575234edf` viser capped exit 0, 0 modellsteg og ingen TEST. Lokal side var **samme på 39 670/39 670**, ukjent 0, motsatt 0, over alle 20 TRAIN-måneder. Etterkontroll bekreftet hasher, gammelt radutvalg og alle månedssummer; den gjorde ikke en ny råpris-/ATR-replay.

Siste overgang til fullhistorikkens side var eldre enn de fem lokale barene i 13 298 rader (33,52 %), nøyaktig samme rader der overgangen var ved eller før prognosetidspunktet. 11 882 rader (29,95 %) hadde en FLATT-observasjon etter denne eldre sideovergangen. De to aldersmålene er **ett overlappende funn**, og overlappende prognoser er ikke uavhengige episoder. Det eldre ankeret er altså ikke nødvendig for å gjenskape **disse eksakte M5-sidene** fra den lokale prisbanen. Diagnosen kontrollerer ikke FLATT, delvis AKTIV, UKJENT, andre TF, hvor tidlig skift varsles, modellæring eller økonomisk edge. Den velger ikke 0,5 mot 1,0 ATR.

Før fit forhåndsbindes disse fem ekspanderende TRAIN-foldene, alle med venstre grense inkludert og høyre ekskludert:

| Fold | Tidligere fit fra 2024-05-01 | Senere skårperiode |
| --- | --- | --- |
| 1 | til 2024-10-01 | 2024-10-01 til 2025-01-01 |
| 2 | til 2025-01-01 | 2025-01-01 til 2025-04-01 |
| 3 | til 2025-04-01 | 2025-04-01 til 2025-07-01 |
| 4 | til 2025-07-01 | 2025-07-01 til 2025-10-01 |
| 5 | til 2025-10-01 | 2025-10-01 til 2026-01-01 |

Fit-rader må ha alle brukte målmodningstider strengt før skårperiodens start; skårrader må ha eksakt framtidig gyldighet før periodens slutt. En eventuell ny nåtilstand som modellinput endrer ONLINE-funksjonen og krever fersk initialbaseline før sammenligning. Fem fold er en kronologisk måleplan, ikke fem tillatelser til modellkjøring.

En **mål-uavhengig** etterkontroll av de to hashbundne kandidatfilene fant bitidentiske `issue_ns`, M5-kilderad, native leads og M1-endepunktsrad på alle 118 706 utstedelser. Tabellen viser antall rader med både eksakt M1-endepunkt og `issue+lead < foldgrense`; den teller ennå ikke om FLATT, AKTIV eller side er fysisk bevist. `N` er antall utstedelser i perioden før TF-spesifikk sensur.

| Fold | Del | N | M5 | M15 | H1 | H4 | D1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | fit | 29 963 | 29 839 | 29 619 | 28 633 | 27 695 | 23 542 |
| 1 | skår | 17 821 | 17 756 | 17 626 | 17 041 | 16 471 | 13 678 |
| 2 | fit | 47 784 | 47 596 | 47 248 | 45 686 | 44 202 | 37 496 |
| 2 | skår | 17 351 | 17 286 | 17 158 | 16 582 | 16 030 | 13 486 |
| 3 | fit | 65 135 | 64 882 | 64 406 | 62 268 | 60 232 | 50 982 |
| 3 | skår | 17 603 | 17 537 | 17 408 | 16 823 | 16 270 | 13 709 |
| 4 | fit | 82 738 | 82 420 | 81 817 | 79 103 | 76 538 | 64 967 |
| 4 | skår | 18 138 | 18 067 | 17 933 | 17 331 | 16 808 | 14 270 |
| 5 | fit | 100 876 | 100 488 | 99 753 | 96 446 | 93 382 | 79 513 |
| 5 | skår | 17 830 | 17 746 | 17 615 | 17 028 | 16 480 | 13 674 |

For hver celle er `N = berettiget + manglende eksakt M1-endepunkt + umodent endepunkt`, med gjensidig utelukkende grunner: først test `issue+lead < grense`, deretter fysisk endepunktsrad. På de fem skårperiodene er 265/276/276/276/265 D1-endepunkter umodne ved høyre grense og ytterligere 3 878/3 589/3 618/3 592/3 891 modne D1-endepunkter fysisk fraværende. En ren splitt på utstedelsestid ville derfor lekket framtidig D1-fasit. Disse tallene er kilde-/klokkeberettigelse på TRAIN, ikke en ny fasit, modellscore eller treningsadmisjon.

Tre sannsynlighetsreferanser fryses før fit. Tidligere TRAIN-andel estimerer `P(AKTIV)` på alle aktivitetskjente rader og `P(OPP|AKTIV)` bare på sertifiserte sider; hver binær andel får fast Laplace-telling `+1/+1`. Egen-TF overgang estimerer de samme to betingede andelene per eksakt kausal nåtilstand, med ukjent/delvis nåtilstand eller tom gruppe tilbake til ubetinget TRAIN-andel. Indikatorreferansen bruker siste kausalt lukkede egen-TF-bar og `z=(close−EMA20)/ATR14`: en binær logistisk mapping med `|z|` for aktivitet og en med `z` for retning, fast L2-styrke 1, bare tidligere målmodne rader og tilbakefall til de smoothede TRAIN-andelene ved én klasse eller ugyldig input. Alle tre bygger `p(NED),p(FLATT),p(OPP)` fra `P(AKTIV)` og `P(OPP|AKTIV)`. Eksakt solver, intercept og sannsynlighetsklipp må bindes i referanseimplementasjonen før score; ingen parameter velges på senere fold. Alle tre estimerer eller tilpasser parametere fra tidligere TRAIN og kjøres derfor ikke mens `training_enabled=false`; kilde-, fold- og oppskriftskontroll kan forberedes uten fit. En mulig billig nåtilstandsreferanse gjenbruker samme TF-maske fra en eldre M5-utstedelse med eksakt gyldighet ved `t`; manglende forgjenger, særlig etter stengning, må falle tilbake til ubetinget prior og rapportere sin dekning. Dette er en avgrenset tilgjengelighetsvariant, ikke bevis på full nåtilstandsdekning.

Skår aktivitet med binær Brier når FLATT/AKTIV er bevist, treklasse med Brier/logtap på singleton, og betinget OPP/NED bare når side er bevist. Rapporter dekning og kalibrering per TF og måned og parvis usikkerhet i ukeblokker, siden femminuttersprognoser overlapper. Sammenlign modell og hver referanse på identiske rader innen hvert mål; ingen samlet PASS fra en pooled totalskår.

For hver fold gjelder fem porter: (1) radvis ny målserie og kilde-/kalender-/ATR-hash for alle TRAIN-måneder, med eksakt `t→v` og gjensidig utelukkende masker; (2) hver folds fit bruker bare tidligere, målmodne TRAIN-rader og egne normaliseringsstatistikker; (3) manglende kilde, warmup, cutoff og overlapp ved foldgrense sensureres, aldri imputeres som FLATT; (4) output, tap og scorer bruker samme tillatte mengde for hver TF ved dens egen native lead (+5/+15/+60/+240/+1440), altså fem mål per M5-prognose; (5) hver senere fold sammenlignes på identiske kvalifiserte rader mot forhåndsvalgt TRAIN-prior og kausal persistens/enkel indikatorreferanse, med dekning, kalibrering og usikkerhet per TF/måned. Senere folds eller TEST velger ikke terskel.

Tre sannsynligheter `p(NED), p(FLATT), p(OPP)` summerer til én. For delvis `{NED,OPP}` er tillatt-mengde-tapet `−log[p(NED)+p(OPP)]`; for singelton brukes vanlig treklasse-logtap. Ukjent mengde med alle tre gir ingen **lokal** klassebegrensning (`−log 1=0`), men en sekvensmodell kan fortsatt få indirekte gradient derfra via senere observerte etiketter; ingen påstand om null totalgradient. Rapporter separat proper aktivitetsmål der FLATT/AKTIV er bevist, retningsmål bare ved bevist singelside, og treklassemål bare ved bevist singleton. Hvert mål bruker sin egen forhåndsdeklarerte maske: aktivitet også på delvis `{NED,OPP}`, betinget retning bare ved bevist singelside og treklasse bare ved singleton. For hvert mål må modell og referanser skåres på identiske rader, tider og TF; tapet må bruke den tilsvarende tillatt-mengde-masken. Kvalifiser ikke retning ved høy dekning alene, og kall for tynn D1-støtte **ukjent** heller enn å flytte grensene. Ingen modellfit, Entry-vurdering eller økonomipåstand følger av dette notatet.
