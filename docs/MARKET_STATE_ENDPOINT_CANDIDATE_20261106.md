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

D1-eksaktdekningen varierer fra 7,8 % i august 2025 til 64,2 % i oktober 2025; senere resultat kan derfor ikke tolkes uten månedlig dekning. En separat capped kontroll av bare TRAIN M1-tidskolonnen fant at alle 39 042 eksakte D1-vinduer inneholder allerede kjente barer ved `t`, typisk 64 av 1 440 (4,44 %). Typisk veggklokkespenn er 1 504 minutter, og 486 eksakte vinduer som krysser en dokumentert helg spenner typisk 4 448 minutter. Dette er **tilstand ved `v`**, ikke bevegelsen fra `t` til `v`; overlappen alene forklarer ikke 50/50-retning. Retningssiden i dagens måleier bruker et fortløpende anker som kan være eldre enn `W_TF` og nullstilles først ved ukjent gap eller manglende ATR. Ankeralder og verdi mot kausal nåtilstand må undersøkes før målet låses.

Før fit forhåndsbindes disse fem ekspanderende TRAIN-foldene, alle med venstre grense inkludert og høyre ekskludert:

| Fold | Tidligere fit fra 2024-05-01 | Senere skårperiode |
| --- | --- | --- |
| 1 | til 2024-10-01 | 2024-10-01 til 2025-01-01 |
| 2 | til 2025-01-01 | 2025-01-01 til 2025-04-01 |
| 3 | til 2025-04-01 | 2025-04-01 til 2025-07-01 |
| 4 | til 2025-07-01 | 2025-07-01 til 2025-10-01 |
| 5 | til 2025-10-01 | 2025-10-01 til 2026-01-01 |

Fit-rader må ha alle brukte målmodningstider strengt før skårperiodens start; skårrader må ha eksakt framtidig gyldighet før periodens slutt. En eventuell ny nåtilstand som modellinput endrer ONLINE-funksjonen og krever fersk initialbaseline før sammenligning. Fem fold er en kronologisk måleplan, ikke fem tillatelser til modellkjøring.

Tre sannsynlighetsreferanser fryses før fit. Tidligere TRAIN-andel estimerer `P(AKTIV)` på alle aktivitetskjente rader og `P(OPP|AKTIV)` bare på sertifiserte sider; hver binær andel får fast Laplace-telling `+1/+1`. Egen-TF overgang estimerer de samme to betingede andelene per eksakt kausal nåtilstand, med ukjent/delvis nåtilstand eller tom gruppe tilbake til ubetinget TRAIN-andel. Indikatorreferansen bruker siste kausalt lukkede egen-TF-bar og `z=(close−EMA20)/ATR14`: en binær logistisk mapping med `|z|` for aktivitet og en med `z` for retning, fast L2-styrke 1, bare tidligere målmodne rader og tilbakefall til de smoothede TRAIN-andelene ved én klasse eller ugyldig input. Alle tre bygger `p(NED),p(FLATT),p(OPP)` fra `P(AKTIV)` og `P(OPP|AKTIV)`. Eksakt solver, intercept og sannsynlighetsklipp må bindes i referanseimplementasjonen før score; ingen parameter velges på senere fold. Alle tre estimerer eller tilpasser parametere fra tidligere TRAIN og kjøres derfor ikke mens `training_enabled=false`; kilde-, fold- og oppskriftskontroll kan forberedes uten fit. En mulig billig nåtilstandsreferanse gjenbruker samme TF-maske fra en eldre M5-utstedelse med eksakt gyldighet ved `t`; manglende forgjenger, særlig etter stengning, må falle tilbake til ubetinget prior og rapportere sin dekning. Dette er en avgrenset tilgjengelighetsvariant, ikke bevis på full nåtilstandsdekning.

Skår aktivitet med binær Brier når FLATT/AKTIV er bevist, treklasse med Brier/logtap på singleton, og betinget OPP/NED bare når side er bevist. Rapporter dekning og kalibrering per TF og måned og parvis usikkerhet i ukeblokker, siden femminuttersprognoser overlapper. Sammenlign modell og hver referanse på identiske rader innen hvert mål; ingen samlet PASS fra en pooled totalskår.

For hver fold gjelder fem porter: (1) radvis ny målserie og kilde-/kalender-/ATR-hash for alle TRAIN-måneder, med eksakt `t→v` og gjensidig utelukkende masker; (2) hver folds fit bruker bare tidligere, målmodne TRAIN-rader og egne normaliseringsstatistikker; (3) manglende kilde, warmup, cutoff og overlapp ved foldgrense sensureres, aldri imputeres som FLATT; (4) output, tap og scorer bruker samme tillatte mengde for hver TF ved dens egen native lead (+5/+15/+60/+240/+1440), altså fem mål per M5-prognose; (5) hver senere fold sammenlignes på identiske kvalifiserte rader mot forhåndsvalgt TRAIN-prior og kausal persistens/enkel indikatorreferanse, med dekning, kalibrering og usikkerhet per TF/måned. Senere folds eller TEST velger ikke terskel.

Tre sannsynligheter `p(NED), p(FLATT), p(OPP)` summerer til én. For delvis `{NED,OPP}` er tillatt-mengde-tapet `−log[p(NED)+p(OPP)]`; for singelton brukes vanlig treklasse-logtap. Ukjent mengde med alle tre gir ingen **lokal** klassebegrensning (`−log 1=0`), men en sekvensmodell kan fortsatt få indirekte gradient derfra via senere observerte etiketter; ingen påstand om null totalgradient. Rapporter separat proper aktivitetsmål der FLATT/AKTIV er bevist, retningsmål bare ved bevist singelside, og treklassemål bare ved bevist singleton. Hvert mål bruker sin egen forhåndsdeklarerte maske: aktivitet også på delvis `{NED,OPP}`, betinget retning bare ved bevist singelside og treklasse bare ved singleton. For hvert mål må modell og referanser skåres på identiske rader, tider og TF; tapet må bruke den tilsvarende tillatt-mengde-masken. Kvalifiser ikke retning ved høy dekning alene, og kall for tynn D1-støtte **ukjent** heller enn å flytte grensene. Ingen modellfit, Entry-vurdering eller økonomipåstand følger av dette notatet.
