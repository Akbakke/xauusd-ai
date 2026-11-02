# Dynamisk markedsstatus: resultat av én kort trening, 2. november 2026

## Konklusjon

Ingen nyttig prediktiv fordel er dokumentert. Modellen lærte noe på de samme 256 TRAIN-radene, men på observerbare celler i senere utviklingsdata velger den aldri FLAT som entydig mest sannsynlig klasse. På alle fem forhåndsvalgte tidsrammehorisonter slår den heller ikke den faste TRAIN-prioren, som her alltid velger UP. Dette er ikke en test av økonomisk lønnsomhet.

Den gamle modellen lærte 60 fremtidige terminalretninger uten egen FLAT-klasse. Den nye bruker dynamiske tilstander DOWN/FLAT/UP på M5/M15/H1/H4/D1, terskel 0,5 ganger kausal Wilder ATR(14) per tidsramme, og 1–1 440 minutters prognoseledetid. Ledetid er ikke en handelsregel eller maksimal holdetid. Målene kan vise oppgang etterfulgt av nedgang uten en fast startstrek. Kandidatterskelen er ikke bevist økonomisk optimal, og prognoser utover 24 timer er ikke målt eller implementert her.

Senere kildeundersøkelse viser at første M5-mål kan være ikke-FLAT selv når OPP/NED er uobserverbart: dette gjelder 22 710 av de 47 602 TRAIN-foreldrene. En separat avslått kildevariant kan bruke slike komplette lokale vinduer i et FLAT/ikke-FLAT-tap uten å dikte retning. Tilsvarende antall er 6 886 på M15, 1 027 på H1, 170 på H4 og 51 på D1. En uavhengig kontroll mot de opprinnelige TRAIN-prisene bekreftet alle fem målmaskene og ATR-tersklene rad for rad. Ingen modell er kjørt med endringen, og full objektivspesifikk TRAIN-audit mangler fortsatt. Definisjon, tester og begrensninger står i [designnotatet](DYNAMIC_STATE_CAUSAL_INPUT_DESIGN_20261102.md).

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

## Hvor ofte TRAIN-målet viser skifter

Den lagrede målserien ble telt på nytt for å skille ulike observerte kilde-minuttpar fra eksponering i overlappende prognoser. De vektede overgangsmatrisene ble gjenskapt eksakt fra samme TRAIN-foreldre og 1–1 440-minuttersgrid; ingen modellprediksjon ble beregnet.

| Tidsramme | Ulike observerte minuttskifter | Derav NED → OPP | Derav OPP → NED | Skifter blant observerte prognosepar |
|---|---:|---:|---:|---:|
| M5 | 24 527 | 6 771 | 6 910 | 23,15 % |
| M15 | 20 643 | 7 293 | 7 364 | 10,38 % |
| H1 | 7 701 | 2 969 | 2 992 | 3,37 % |
| H4 | 2 166 | 836 | 837 | 0,96 % |
| D1 | 246 | 109 | 105 | 0,118 % |

Øvrige skifter involverer FLAT. Eksempelvis svarer de 246 D1-minuttskiftene til 60 087 observerte parent-/lead-par fordi prognosevinduene overlapper. De unike minuttene kan selv høre til samme markedsbevegelse og er ikke uavhengige hendelser. Ukjente tilstander og skjult intraminuttrekkefølge er utelatt. Det svært lave D1-skifteforholdet viser hvorfor videreføring er en sterk referanse, men er ikke et bevis for at tapet eller klassefordelingen alene forårsaket manglende edge. Overgangs- og FLAT-målinger må fortsatt rapporteres sammen med treffrate og baseline.

Opptelling: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/TRAIN_TRANSITION_EXPOSURE_001/RESULT.json` (SHA-256 `6434bc6033bb73ef79f9b152727a594cf15a92fee05bd31eeb8b8ab52dd8d63a`). Separat kontroll av aggregataritmetikk og kildeidentitet: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/TRAIN_TRANSITION_EXPOSURE_001/SUMMARY_REVIEW.json` (SHA-256 `db12ff4d51a124d2ffb66f3c5c1f452c844375c80dea3229707933a034836269`).

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

## Første fremtidsmål og oppløsning

På de 47 602 TRAIN-foreldrene var M5-tilstanden ett minutt frem ukjent på **22 744** rader, 47,78 %. På **13 767** av disse var selve det neste M1-lyset bredt nok til å krysse terskelen, mens sluttkursen lå inne i lysintervallet. M1-lyset forteller ikke hvilken rekkefølge høy og lav hadde, og målkoden markerer derfor sluttretningen ukjent. For de resterende 8 977 er denne ene direkte forklaringen ikke påvist. Av 24 858 kjente M5-sluttilstander hadde 8 262 likevel ukjent lokal sti. Tilstand og bevegelsesrekkefølge er derfor ikke det samme.

Et syntetisk kontrollert eksempel viser to prisstier med identisk OHLC og motsatt sluttretning. Det viser informasjonsgrensen i et lys; det beviser ikke at bestemte reverseringer faktisk fant sted i de virkelige lysene. På D1 var 5 549 av 5 600 ukjente førsteceller flagget for et ukjent kildehull i det lokale vinduet; dette er ikke 5 549 separate hull, og diagnoseflagg kan overlappe. Ingen av tallene er modelltreffrate eller bevis på edge.

Kilden brukt til målet er OANDA XAU/USD M1-midtkurser. Et avgrenset filnavnsøk fant ingen S5-/tick-kilde i de kontrollerte XAU-mappene, men det utelukker ikke andre mapper eller generiske filnavn. En tidligere Dukascopy-cache er en annen leverandør og mangler dokumentert komplett dekning og tidsproveniens; den er ikke innrømmet som OANDA-fasit. [OANDA dokumenterer S5-lys](https://developer.oanda.com/rest-live-v20/instrument-df/), men tilgjengelighet for hele den historiske perioden er ikke kontrollert. Ingen brokerkall ble gjort. S5 kan redusere tvetydighet, men kan selv skjule rekkefølge inne i fem sekunder.

En eventuell finere kilde må først identifiseres og kilde-/tidsbindes uten TEST eller brokertilgang. Deretter må komplette underlys aggregeres tilbake mot de samme M1-midtkursene, med avvik og manglende tidsrom eksplisitt maskert. Bar-sluttberegningen må bruke bare data og terskel tilgjengelig ved sluttidspunktet; eventuelle intra-bar-tider krever egen kausal klokke. Før noen trening må en separat targetidentitet, TRAIN-audit og ny initialbaseline kvalifiseres. Eksisterende prognoser må ikke få nye fasiter i ettertid uten tydelig ny sammenligning.

Lagringsspor: årsaksflagg `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/FIRST_STATE_OBSERVABILITY_DIAGNOSTIC_001/RESULT.json` SHA-256 `5bb3c391b3e366d5939856da6c995e2408fbb53051a7f90d60541459f37afc58`; direkte M1-kontroll `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/FIRST_STATE_M1_AMBIGUITY_TRAIN_001/RESULT.json` `7bcf2abd40fdce157f43271870c719286694c07f09c360b9f4132e54d18662af`; syntetisk vitne `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/FIRST_STATE_M1_AMBIGUITY_TRAIN_001/SYNTHETIC_ORDER_WITNESS.json` `d820df94af06d72272252ff3111f51627fc0716ba030d9c270e1dfd3636c910f`; metadatakryssjekk `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/FIRST_STATE_M1_AMBIGUITY_TRAIN_001/INDEPENDENT_REVIEW.json` `acefe74f32428863fabf35761c4a7d6ebe6e4cfef823fe1669569caffb8dc342`; kildeavgrensning `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/XAU_INTRAMINUTE_SOURCE_SCOPE_001/RESULT.json` `ae80bc3b4f13ea217ab5af348f3d0e60a63cb208bbb80eab1708182810354ec1`. Ingen ny modell, optimizer, scoring eller TEST.

## Indikatorprofil og faktisk datadekning

Den gjennomførte dynamiske engangstreningen brukte fullprofilen. Den binder alle deklarerte signalfelt, kontinuerlige kontekstfelt og MTF-felt til åtte spesialistfamilier på M5, M15, H1, H4 og D1, med separate GC-historikker og D1/makro-kontekst. Profilen omfatter 2 725 signalfelt, 2 542 kontinuerlige kontekstfelt, ett kategorifelt og 253 XAU-felt per MTF-historikk; feltene inkluderer masker og aliaser og er ikke et antall uavhengige indikatorer. Profilidentiteten stemmer med de lagrede privatbatchene. Alle 133 deklarerte gradientgrupper hadde kontakt i deres andre tekniske steg, men dette dokumenterer ikke verdien til enkeltindikatorer.

En saved-only sammenkobling har kontrollert samme kilde/provider, eksakt TRAIN- og senere foreldre- og beslutningstidsakse, samt felt, observasjonsmasker og TRAIN-support på tolv fysiske flater mot den tidligere fulle inputrevisjonen. Den tidligere revisjonen målte alle 47 602 TRAIN- og 17 288 senere utviklingsforeldre. Konteksten har 17 verdifelt uten en eneste senere observasjon: ti strenge H1-flytfelt var observert på bare 36 TRAIN-foreldre, og sju bearish D1-proxyfelt på 550. De 17 feltene blir ikke nulltolket som kjente observasjoner.

Separat kjent H1-delta har delvis observert data på 45 214 TRAIN- og 16 560 senere foreldre. I alle de sistnevnte finnes også uklassifisert aggressorvolum; den undersøkte kildekoden krever null ukjente for komplett signert flyt. Delvis kjent delta er derfor ikke et fullstendig signert delta. En separat kildekjede kobler de brukte M5-/H1-/D1-tabellhashene til produsentens opprinnelige terminal. De tre relevante produsentfilene er byteidentiske med versjonene ved denne kjøringens Git-commit. Kilden tar ukjent volum fra rå handelsside N og krever null ukjent i hvert minutt for streng signert flyt. Ingen rå hendelser eller enkelttimer er spilt om igjen, så tellerne dokumenterer dekningen og kildekravet, ikke en ny kontroll av hver hendelse. Ikke gjett manglende handleretning.

De sju D1-proxyfeltene er likevel ikke helt fraværende fra modellens historikkvinduer: I grunnlaget for de senere foreldrene var hvert felt observert på tre av 315 unike D1-kilderader, som ga 51 864 gjenbrukte vindusforekomster. TRAIN-historikken hadde tre av 425 unike kilderader. Kilderadene kan overlappe mellom periodene. Produsentkoden setter sonetilstanden til ukjent etter et manglende TF-lys inntil en ny observert strukturendring etablerer en sone. Den opprinnelige produsentkvitteringen kvalifiserte ikke en historisk GC-børskalender; kontrollen har ikke avgjort om de enkelte hullene var dokumenterte markedspauser eller manglende data, eller om de forklarer disse proxyfeltenes dekning. [CMEs metallguide for 2024](https://www.cmegroup.com/trading/metals/files/metals-prod-guide-2024.pdf) beskriver ordinær Globex-tid, og [CMEs kunngjøring for 9. januar 2025](https://www.cmegroup.com/trading-hours/files/day-of-mourning-january-9-2024.pdf) dekker den ene dagen. Disse er ikke en komplett historisk kalender over unntak. Direkte nedlasting til GX1 ble avvist eller tidsavbrutt. Tre arkiverte CME-dokumenter ble deretter lagret lokalt og verifisert mot arkivindeksen; ingen hull er klassifisert fra dem. Nedlastingskvittering: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/CME_HISTORICAL_HOURS_SOURCES_001/ARCHIVE_DOWNLOAD_RESULT.json` (SHA-256 `f437e0f2539789b9ca06118b6d95bbec9ce79d90d24cf9047c7f98b41b550c3e`).

En avgrenset kontroll av D1-kildens tilgjengelighet og masker, uten pris eller mål, fant 168 observerte kilderader innen TRAIN-beslutningenes tidsrom og 61 i senere utvikling. Av henholdsvis 36 og 13 kilderader etter hopp var 35 og 12 merket søndag UTC. Alle de 49 hadde ukjent bearish sonetilstand. I senere utvikling var strukturgrunnlaget markert klart på 37 kilderader, men sonetilstanden var ukjent på alle 37. Søndagsmerkingen er ikke bevis for at hvert hull var en ordinær stenging; den historiske GC-kalenderen må bindes før dette kan avgjøres og før en eventuell endring av masken. Kontroll: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/D1_GAP_SOURCE_OBSERVATION_001/RESULT.json` (SHA-256 `916c874ba3c4f2e1ba0af186b7c133b079ac9305c3c991d687dad459401fcfcd`).

En senere kandidatsortering sammenlignet de 47 søndagsmerkede hoppene med CMEs ordinære uketider i Chicago-tid. Alle ligger innenfor den nominelle ukestengingen; fire TRAIN- og to senere hopp berører helligdagsperioder i en arkivert CME-filterliste. De øvrige 41 er kandidater, ikke verifiserte faktiske stenginger. Filterlisten er generell og 2025-oppføringene var fremtidsopplysninger ved arkivering; den kan ikke utelukke senere endringer. De to øvrige hoppene ligger på onsdagsmerkede lys. En uavhengig minuttkontroll av 135 360 tidspunkter bekreftet nominelle tider, sommertid og helligdagsfilter. Ingen kilde- eller målmasker er endret. Kandidatresultat: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/D1_GAP_CALENDAR_CANDIDATES_001/RESULT.json` (SHA-256 `69637742f451615c645da67783b70e8c4199b8048d72fab573714ea0c9f8ccc3`); uavhengig kontroll: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/D1_GAP_CALENDAR_CANDIDATES_001/INDEPENDENT_REVIEW.json` (SHA-256 `cf16c7a0522d27074eaa6d2979d612d7152bdfada7c523c536b7e3f2ef5f5af8`).

I et rent motfaktisk forsøk ble nullstillingen bare undertrykt for de 41 uverifiserte helgekandidatene; alle andre nullstillinger ble beholdt. Dette økte kjent bearish D1-sonetilstand fra **3 til 77 av 168** observerte TRAIN-kilderader, men senere utvikling ble fortsatt **0 av 61**. I de senere radene var nedadgående struktur observerbar på 37 dager uten noen observert nedadgående strukturendring som opprettet ny bearish sone. Dette er ikke en endring i modelldata eller bevis for bedre prediksjon. De 21 originale indikatorfeltene ble gjenskapt eksakt fra kilderadene, og ti utvalgte prefikskontroller bestod; den hypotetiske varianten er ikke innrømmet til trening. Resultat: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/D1_COUNTERFACTUAL_ZONE_AUDIT_001/RESULT.json` (SHA-256 `8875df1fad6ce78f3a2beee9903ca8d59ba40bd5f35310661ca52e3d438cc6d2`); strukturkontroll: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/D1_COUNTERFACTUAL_ZONE_AUDIT_001/STRUCTURE_REVIEW.json` (SHA-256 `16c0ab5f4f3e7ce943d581b4e328413e35296514ab1d9f1d283adcfeaa92050d`).

GC-D1-volumets observerte historikk per forelder var 0–168 dager i TRAIN, gjennomsnitt 82,89, mot 169–230 dager senere, gjennomsnitt 199,39. Historikkvinduene overlapper, så dette er ikke uavhengige observasjoner. Dekningsforskjell og ukjent flyt er konkrete begrensninger, men deres årsakseffekt på svak prediksjon er ikke isolert. Ny nåtilstands-/langhorisontvariant krever fortsatt egen native kvalifisering.

Gjenbrukskontrollen, kildehashene og grensene står i `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/DYNAMIC_FULL_FEATURE_COVERAGE_REUSE_002/RESULT.json` (SHA-256 `fa99d1358575e7c9882a698a392dcb4343f7c77ef9ce369ab7ea82c5b2416008`). Ingen modell, optimizer, nye targetdata eller TEST ble brukt. Den separate kontrollen av produsentterminal, tabellidentiteter og kildeversjon står i `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/GC_FLOW_SOURCE_LINEAGE_001/RESULT.json` (SHA-256 `0758f0d26e23255deb5b345b79c4a7a7b783dd0b06fbc56d513c49334b9bfeeb`). Kontroll av D1 nåkontekst, unike historikkrader og kildeadferd: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/D1_PROXY_SUPPORT_REUSE_001/RESULT.json` (SHA-256 `678942de8a1f6ecfbcdd0d9983b5242799e41035ec0f0eab9db1f433fbfd9892`).

## Kausal nåtilstand ved beslutningstid

Den lagrede nåtilstanden er kontrollert på alle 47 602 TRAIN- og 17 288 senere rader. Når tilstanden er kjent, har både den ferdige M1-baren og ATR-verdien deklarert tilgjengelighet senest ved beslutningstid; ingen ligger etter denne. De tidligere beståtte fremtidsendringstestene for samme målkode er gjenbrukt. Dette er historisk klokke- og kildekontroll, ikke bevis for leveringstid i ONLINE-løpet.

I senere data var M5-nåtilstanden kjent på 8 987 av 17 288 rader (52,0 %), og alle fem tilstander var kjent på 8 284 rader (47,9 %). Av de 8 987 M5-radene hadde 3 202 kjent tilstand, men uavklart lokal rekkefølge i forløpet. Tilstandstilgjengelighet og sti-tilgjengelighet må derfor ha separate masker; UNKNOWN må ikke bli FLAT.

På samtlige 8 987 rader med kjent M5-tilstand var M5-ATR deklarert tilgjengelig nøyaktig ved beslutningstid, og siste M1-bar hadde også lukketid nøyaktig der. Om meldingen faktisk er mottatt og beregnet før modellen beslutter, er ennå ikke dokumentert. En mulig ONLINE-implementasjon må dokumentere denne hendelsesrekkefølgen og videreføre tilstanden trygt gjennom reelle markedspauser og restarter; en historisk tidslikhet alene er ikke nok.

## Tidskilde og mulig startpunkt for videre design

En kontroll av skjema og Parquet-footer uten datarader viser at den brukte XAU M1-kilden har bartid, men ingen egen navngitt mottaks- eller behandlingstid. GC-relaterte mottaksfelter i M5-kildens skjema dokumenterer ikke levering av XAU M1. Andre logger er ikke undersøkt. En avgrenset statisk søkekontroll fant heller ingen kobling til dette dynamiske målet i de 14 Python-filene under execution/replay, og de undersøkte eldre live-adapterne beskriver egne begrensninger. Faktisk ONLINE-rekkefølge og integrasjon er derfor fortsatt uavklart.

En NumPy-beregning med de lagrede, positive TRAIN-overgangstellingene viser at et mulig modellhode med null residual kan gjenskape Markov-referansen for kjent nåtilstand eller UNKNOWN-fallback. Største forskjell mellom FP32 og float64 var 0,00000895 på fem tidsrammer og 1 440 ledetider. Dette er ikke kontroll av faktisk native modellbane, ny scoring, bedre læring eller edge. Den konkrete kontrakten, maskene, mulige tap og uavklarte kvalifiseringskrav står i [designforslaget](DYNAMIC_STATE_CAUSAL_INPUT_DESIGN_20261102.md).

## Status for historisk nåinput

Et separat historisk nåinpututtak er implementert med uavhengige masker for kjent tilstand og kjent lokal sti. Det returnerer ingen fremtidsetiketter og merker datatiden uttrykkelig som historisk, ikke dokumentert ONLINE-levering. 34 syntetiske targettester bestod; et CPU-uttak mot alle 64 890 lagrede TRAIN-/utviklingsforeldre stemte eksakt med tidligere kilde-/klokkeanalyse. En ny opt-in kildevariant kobler nåtilstand og begge masker til modellrepresentasjonen og gir et separat vektet første-tilstandstap. Ingen tapsvekt er valgt for trening, og faktisk modellforward, gradienter og native ressurser er ikke testet. Den gamle treningskvitteringen kvalifiserer ikke den endrede kilden. Den normale commit-hookens modelltester ble utelatt fordi de kjører modell; de fokuserte kilde-/tensorkontrollene og separate guardrail-testene bestod. Ingen ny modell, scoring eller trening ble kjørt. Se [designforslaget](DYNAMIC_STATE_CAUSAL_INPUT_DESIGN_20261102.md).

## Lengre tidshorisonter

Lagrede target-endepunkter er nå undersøkt 1–30 kalenderdager frem uten modellkjøring. Alle tre klasser finnes på alle fem tidsrammer for de sju forhåndsvalgte eksempelledetidene i både TRAIN og senere utviklingsdata. Eksempelvis har senere utvikling ved sju døgn 8 194 observerbare M5-endepunkter og 15 618 D1-endepunkter av 17 288 foreldre. Dette er overlappende endepunkter, ikke fullstendig observerte ukesforløp eller dokumentert prediksjon. Den faktisk trente modellen stopper ved 1 440 minutter. En separat kildevariant kan bindes til en lengre tett minuttgrid, med ukjente hull maskert; den er ikke kjørt eller kvalifisert. Se [hele oversikten](DYNAMIC_STATE_CAUSAL_INPUT_DESIGN_20261102.md).

En engangsmåling fant 619,8 GB ledig i WSLs filsystem og 286,8 GB på Windows-disken som inneholder WSL-volumet. De kan ikke brukes om hverandre; begge må kontrolleres på nytt før en eventuell lagringskrevende plan. Dette er ingen plassreservasjon. En ny kildevakt kontrollerer begge volumene før modellstart og før utvidede outputfaser, med en eksplisitt oppskriftsbundet reserve; den er kontrollert under lesevakt, men ikke gjennom en faktisk modellkjøring.

En effektiv TRAIN-telling er implementert og gjenskapte de tidligere 1 440-minutters klassene, overgangene, maskene og prioren eksakt. En eventuell utvidet TRAIN-audit kan forberedes som et uttrykkelig ikke-admittert utkast; ingen ny horisont er valgt. Hele pre-TEST-cachet hash-/strukturkontrolleres ved lasting, mens bare TRAIN-etiketter aggregeres. Se [kontrollgrunnlaget](DYNAMIC_STATE_CAUSAL_INPUT_DESIGN_20261102.md).

En ny TRAIN-only opptelling av et uforpliktende sjudøgnseksempel viser betydelig klasse- og fortsettelsesskjevhet, særlig på D1. Dette er målfordeling, ikke en ny prognosemåling; se [kontrollgrunnlaget](DYNAMIC_STATE_CAUSAL_INPUT_DESIGN_20261102.md).

## Gjenstående tiltak

1. Fullfør en kausal ONLINE-representasjon og modellkobling for kjent nåtilstand per tidsramme med eksplisitt UNKNOWN/tilgjengelighetsmaske, tilgjengelig før beslutningstid. Gi modell og baseline samme informasjon; behold genuine indikatorer og alle vakter. Kontroller først at målet ikke lekker fremtidig threshold, ekstrempunkt eller observasjon.
2. Revider tapsdesignet slik at riktig tilstand ved starten og marginale sannsynligheter ved forhåndsvalgte ledetider får direkte treningssignal, samtidig som overgangs-/sekvenstap og begge skifteretninger beholdes. Balansér eventuelt klasser som et ekstra treningsmål, men evaluer uvektet log-loss og Brier på naturlige klassesammensetninger og kontroller kalibrering. Disse er egnede sannsynlighetsscorer; se [Gneiting og Raftery, 2007](https://doi.org/10.1198/016214506000001437). Valg av vekter er en hypotese som krever forhåndsdefinert TRAIN-/kronologisk utviklingskontroll.
3. Behold observasjonsmasker, og skaff høyere tidsoppløsning eller bedre dokumentasjon når det er mulig fremfor å gjette rekkefølge gjennom M1-hull. Kvalifiser den nye utvidbare ledetidsvarianten og ressursbehovet før swing utover 24 timer prognostiseres. Evaluer størrelse, varighet, alle relevante skifter og senere entry-økonomi separat før noen påstand om edge.
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
- `historical_current_input_accessor`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/HISTORICAL_CURRENT_INPUT_ACCESSOR_001/RESULT.json` — SHA-256 `b4321b1a57ed5e12f8ae01cb01c1e0e99cd76f769ec3365978c9ec6746bf1e1f`
- `historical_current_input_test_receipt`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/HISTORICAL_CURRENT_INPUT_ACCESSOR_001/TEST_RECEIPT.json` — SHA-256 `b2ae8a296f0b0ec9941cb7e18c4cb4b6713b173bf851cd64e418d8db61ca5e24`
- `current_model_wiring_source_review`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/CURRENT_MODEL_WIRING_SOURCE_001/SOURCE_REVIEW.json` — SHA-256 `0096ed929f2e25f8426991185c4ecec4058e59fa8cf61d6fc1a06c8b4030b1b7`
- `saved_long_horizon_target_support`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_LONG_HORIZON_TARGET_SUPPORT_001/RESULT.json` — SHA-256 `b44043df9d6be4a35d666b32f68403d16fa5529c420a821cfb2d4cca26ead3f2`
- `extended_dense_horizon_source_review`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/EXTENDED_DENSE_HORIZON_SOURCE_001/SOURCE_REVIEW.json` — SHA-256 `34e430bafc8b9006491bc3851886ae1874ab797c0d04e1e2969e682fec8bffab`
- `wsl_host_storage_observation`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/WSL_HOST_STORAGE_OBSERVATION_001/RESULT.json` — SHA-256 `8784a98083ebebf241ddd34e04eeb6a6742593b07fa299f6c457648bddcdb2ec`
- `extended_dual_storage_guard`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/EXTENDED_DUAL_STORAGE_GUARD_002/SOURCE_REVIEW.json` — SHA-256 `0067a138830c356aa4ddc60249a06afcc672ae6f737010336ee342d0022d30c3`
- `train_count_and_draft_source_review`: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/TRAIN_COUNT_REUSE_CHECK_003/SOURCE_REVIEW.json` — SHA-256 `2149e88cd7281d36926f41d8ac792b6fb732c49e6f71001e9bdf88b4913cdb3b`
