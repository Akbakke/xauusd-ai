# Selvstendig revisjon av retning og features, 28. oktober 2026

Status: **PÅGÅR**. Spørsmålet er om genuine, kausale inputs faktisk når modellen, kan læres og gir bedre retning på senere data. Rapporten bruker bundne summeringer, kildekontroll, syntetiske regneeksempler og en separat capped desimalkilde-audit av erklærte data før 2026. Ingen ny modellforward, trening eller TEST-inngang ble åpnet av desimalauditen. Tallenes evidensklasse og ubeviste grenser er oppgitt nedenfor.

## Kjørt 1–60-minutters modell

Den separat admitterte 60-utgangsmodellen er fullført etter fysisk boot 573: 47 602 TRAIN-foreldre, 2 976 oppdateringer, og 17 288 senere utviklingsforeldre. Kilde uendret, TEST ubrukt og signert vakt PASS. Oppgaven er deaktivert. Opprinnelig effektgrense er 160 W og faktisk tomgang var 23,31 W; under trening var øvre grense 250 W og målt effekttopp 137,66 W. En effektgrense er ikke faktisk trekk.

| Minutter | Kvalifiserte senere utfall | AUC | Treffsikkerhet | Fast TRAIN-prior, treff | Modellens logloss | Fast TRAIN-prior, logloss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17 215 | 0,501055 | 49,71 % | 49,69 % | 0,693677 | 0,693228 |
| 5 | 17 197 | 0,500183 | 48,87 % | 51,58 % | 0,694562 | 0,692740 |
| 10 | 17 137 | 0,498516 | 51,63 % | 51,91 % | 0,692777 | 0,692532 |
| 15 | 17 080 | 0,505191 | 49,05 % | 52,39 % | 0,695772 | 0,692181 |
| 30 | 16 890 | 0,511837 | 50,79 % | 53,55 % | 0,695744 | 0,691117 |
| 60 | 16 513 | 0,521054 | 51,96 % | 54,79 % | 0,697610 | 0,689432 |

Dette er **punktestimater**, betinget på observerte ikke-null utfall, fra én gjenbrukt utviklingsfold; antall er forskjellig per horisont. Tabellen er ikke en seleksjon av beste minutt. Den forhåndsdefinerte, simultane dag-/ukeanalysen på alle 60 hoder er nå fullført: ingen minutter har begge nedre båndgrenser over 0,5. Ved 60 minutter er AUC 0,521054 med simultant dagbånd 0,491561–0,550547 og ukebånd 0,493632–0,548476. Ved 5 minutter er kandidatens AUC 0,500183 mot 0,503021 for den gamle modellen på de samme 17 197 kvalifiserte foreldrene; parret dagintervall for differansen er −0,016391 til 0,009926, og ukeintervall −0,016751 til 0,009815. Rangbedring er ikke dokumentert. Kandidatens logloss er 0,694562 mot 0,693514 og Brier 0,250706 mot 0,250183; begge parrede intervaller for kandidat minus gammel er positive under begge blokkskjemaer. Disse to intervallene rapporteres separat og gjelder én gjenbrukt fold. Arkitektur og antall mål endret seg samtidig, så resultatet kan ikke tilskrives én faktor. Blokkutbyttbarhet er ikke etablert. Ingen FLAT-/banekvalitet, uavhengig bekreftelse eller økonomisk edge er bevist. Desimalrevisjonen nedenfor har målt fortegnsfeilen på faste skårer; retrening med korrigerte mål er ikke målt.

På de samme 17 197 femminuttsradene svarer den nye modellen NED på 14 905 (86,67 %) ved fast 0,5-terskel, mens faktisk OPP er 8 870 (51,58 %). Gjennomsnittlig predikert OPP-sannsynlighet er 48,74 %, beregnet fra alle de lagrede kalibreringsbøttene; de to ikke-tomme bøttene dekker 0,4–0,6. Dette viser skjev kalibrering på denne folden. En monoton terskel-/nivåendring uten ny rangering kan ikke endre AUC på samme rader, og resultatet beviser ikke hvorfor rangeringen er svak.

- Treningsresultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_ONEPASS_FIT_20261028_001/RUNTIME/prefix-fit/window-0001/RESULT.json`, SHA-256 `2ce98e2e1967feeaabd6d0e1e4ffffac2178cdafa7cdc5e6561e6cd52f4dc3b8`.
- Alle 60 senere punktmålinger: samme mappe, `OOF.json`, SHA-256 `349e975a2cb22971606133caab848c26e7c84309d375f04077a427a6b1eeb451`.
- Simultan 1–60- og parret 5min-usikkerhet: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_ONEPASS_UNCERTAINTY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `47f20a687853e9ce8617aa96dd1cf0a8c47987b4c580c78f1805ceaeef60a687`. 10 000 trekninger per skjema, 77 UTC-dager og 14 UTC-uker. Bruker de opprinnelige målmaskene; desimaleffekten er kvantifisert separat.
- Vaktkvittering: samme run, `RUNTIME/receipts/invocation-0001.json`, SHA-256 `5f6a242dae28ffa802ca235da64d8512c0b5437f8f6fa8f45082b8f20006f51c`. Idle-kontroll: `POWER_IDLE_RESTORE_001.json`, SHA-256 `86fd4c8e4754d853a84d22772de6c97229d314917dbd09b9d9a529ccaf348d51`.

## Hvilke observasjoner kommer med?

Alle fem radene nedenfor starter med de samme 17 288 senere foreldrene. Tabellen bruker den opprinnelige målidentiteten; desimalauditen dokumenterer de få ekstra nullbevegelsene separat. En fremtidig pris som mangler, en observert nullbevegelse og et endepunkt utenfor foldens cutoff holdes atskilt.

| Minutter | Gyldige ikke-null | Modne, men manglende pris | Modne nullbevegelser | Umodne ved foldgrensen |
| --- | ---: | ---: | ---: | ---: |
| 1 | 17 215 | 0 | 73 | 0 |
| 5 | 17 197 | 63 | 27 | 1 |
| 15 | 17 080 | 189 | 16 | 3 |
| 30 | 16 890 | 378 | 14 | 6 |
| 60 | 16 513 | 756 | 7 | 12 |

Horisontene evalueres derfor på ulike rader: en AUC-forskjell mellom for eksempel 5 og 60 minutter kan også påvirkes av hvem som er med. De manglende radene øker systematisk med horisonten; årsaken til akkurat dette mønsteret er ikke fastslått av disse tellingene alene.

## Dokumenterte feil og begrensninger

1. **Målfeil bekreftet på faktiske data, men den forklarer ikke skårene alene.** Begge eksisterende målbyggere beregner midtpris i float64. En separat, bundet capped audit sammenlignet opprinnelige OANDA-desimalstrenger med kanoniske klokker og floatverdier, målpekerne og alle 60 lagrede endepunkter for 118 249 foreldre før 2026. Av 6 946 049 observerte foreldre/minutt-celler var 976 (0,0141 %) lagret OPP/NED når den eksakte desimalmidtprisen var uendret. Ingen lagrede nullmål var eksakt ikke-null, og ingen reell opp-/nedbevegelse fikk motsatt fortegn. Dette er celletellinger, ikke antall uavhengige markedsperioder. I fold1s TRAIN var 479 av 2 796 249 celler berørt, hvorav 26 ved 5min. I senere fold1 var 164 av 1 015 373 celler berørt, hvorav 5 ved 5min. Ved uendret modellscore og korrekt 5min-nullmaskering blir AUC 0,500175 mot opprinnelig 0,500183; den største absolutte AUC-forskjellen over 60 minutt er 0,000176. Dette er skåring av faste prediksjoner, ikke en kontrafaktisk retrening eller et kausalbevis for at feilen aldri påvirker vektene. Første forsøk stoppet på feil antakelse om arkivomslaget og er konsumert; forsøk 002 brukte verifisert v2-request/response-struktur og fullførte. Kildefeltene har tre desimaler i den undersøkte perioden. Ikke bruk en gjettet epsilon eller generaliser denne presisjonen til andre kilder/perioder. De gamle målene er bevart og en korrigert identitet er materialisert; bruk i modell krever en separat kvalifisert målforbruker.

   Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_SIGN_AUDIT_20261028_002/OUTPUT/RESULT.json`, SHA-256 `9f3441e79017b4e2d00656e9843af3a96ca3c2bef143275b7b28d927aed5b096`; terminal SHA-256 `af58987e8d784a8d8bfa727a207b5da4f15c448d67f51c7f28cfe642746229ff`. [OANDA oppgir prisverdier som desimalstrenger](https://developer.oanda.com/rest-live-v20/pricing-common-df/).
2. **Streng GC-aggressor har lav dekning.** Av 47 602 første TRAIN-beslutninger er strengt kjøp/salg observert på 10 833 M5-, 2 583 M15- og 36 H1-beslutninger. Ti strenge H1-verdifelt bygger i første prefix på de samme tre forskjellige gyldige H1-barene; tell ikke gjentatte beslutninger som nye H1-observasjoner. I undersøkte kildefiler var `signed_valid` alltid falsk for alle 4 747 H4- og 792 D1-rader. Dette følger den strenge definisjonen og er ikke bevis på feilklassifisering. Delvis kjent H1-delta er derimot tilgjengelig for 45 214 beslutninger; H4 og D1 for henholdsvis 45 540 og 36 164. Ukjent side må fortsatt være ukjent, og delvis dekning er ikke full tape. Kildekoden mapper Trade B til kjøp, A til salg og N til ukjent, som samsvarer med [Databentos feltbeskrivelse](https://databento.com/docs/conventions/common-fields-enums-types); dette er semantikkontroll, ikke bevis for at alle verdier er riktige. Faktisk retningsverdi er uavklart.
3. **Tre aktive D1-pris-/volumfelt har ingen tallvariasjon i første TRAIN-prefix.** Bearish GC-orderblokkens lav, høy og opprinnelsesvolum er observert på tre forskjellige D1-barer; hver verdi er lik sitt prefix-senter. Observert verdi normaliseres derfor til null, og manglende verdi maskeres til null i alle tre respektive verdiinnganger. Modellen kan bruke separate tilstedeværelsesmasker, men kan ikke lære hvordan variasjon i disse verdiene virker i denne folden. Dette beviser ikke årsak til 5-minuttersresultatet.
   Samlet er 29 aktive GC-verdifelt i den aktuelle `ctx_cont`-overflaten konstante på alle observasjonene i første prefix på 47 602 beslutninger. Tyve er fire ulike klokke-/kvalitetstellere på fem tidsrammer som alle er null når observert; fire er minste registrerte handelsstørrelse som alltid er én; de øvrige er blant annet profilalder og D1-ordreblokkverdier. Manglende observasjon og tilstedeværelse har egne masker. Derfor kan tallverdiene ikke lære en variabel sammenheng i denne folden, mens tilgjengelighetsmaskene fortsatt kan skille rader. Dette er 29 felt i én overflate, ikke 29 uavhengige feil, og flere kvalitetsfelt kan med rette være konstante. Tilsvarende lokale signalaliaser må ikke telles en gang til. Inventar: `INDEPENDENT_FEATURE_METADATA_AUDIT_20261028_001/feature_inventory.json`, SHA-256 `b08665ee84298655175bd05b38d4c4a0f3d2f7455db243e3224e400d0626c8ea`.
4. **Noen binære hendelser finnes ikke i første prefix.** Det er ingen sanne verdier for observerte D1 bearish/bullish første-retouch-felt i de 550 aktuelle kontekstradene; H4/D1 `signed_complete` er også alltid falske når observert. En hendelsesrespons kan ikke læres fra hendelser som ikke er observert i prefixet. Andre tidsrammer, masker og kontekst beholdes.
5. **D1-sesjonsskifte er en konstant.** `mtf_d1.session_change_flag` er aktivt, men uttrykkelig definert som null fordi D1 lukker ved samme UTC-grense. En tidligere tallkontroll på samme cache fant null på alle 4 147 D1-rader. Feltet skiller ikke D1-rader; øvrig sesjonskontekst er aktiv.
6. **Fem GC-MTF-masker er rutet til en annen familie enn verdien.** `gc_<tf>_observed_book_trailing_support_gap_minutes` går til momentum/flow, mens den gjenbrukte `gc_<tf>_observed_book_valid` går til sesjon/regime. I et syntetisk M5-eksempel gir både gyldig null og manglende verdi samme normaliserte verdi i momentumgrenen. Masken finnes et annet sted, og fellesmodellen kan ha annen informasjon. Ingen global informasjonsmangel eller retningsskade er bevist.
7. **M1-klokke og fyll er svakere dokumentert enn kontraktnavnet antyder.** Kilden er M1-candles der `time` er periodens start og `open` første pris innenfor perioden; selve hendelsestidspunktet finnes ikke i overflaten. [OANDAs definisjon](https://developer.oanda.com/rest-live-v20/instrument-df/) støtter denne tolkningen. Nøyaktig quote-tid, faktisk fyll og eksakt forløpt tid mellom to åpninger er derfor ikke bevist. Dette er ikke en bevist årsak til AUC nær 0,5.
8. **Ulik sesjonsklokke.** XAU D1 starter fast 22 UTC; GC-profilen starter 18 New York-tid, som er 22 UTC om sommeren og 23 UTC om vinteren. Begge er eksplisitte definisjoner, og [CME oppgir GC-handelstider i lokal USA-tid](https://www.cmegroup.com/markets/metals/precious/gold.timeAndSales.html?videoId=6400825166112). Ulikhet er ikke i seg selv en feil; faktisk dekning og effekt må undersøkes.
9. **96 M5-rader dekker ofte mer enn åtte timer.** En separat capped kontroll på den faktiske, hashbundne kildetidsaksen koblet alle 118 249 foreldre til de samme 96 observerte radene modellen bruker. I 41 124 vinduer (34,78 %) er minst ett intervall lengre enn fem minutter. Median samlet vindutid fra første radstart til beslutningen er 480 minutter; 90-persentilen er 540 minutter, 99-persentilen 3 420 minutter (57 timer), og maksimum 4 860 minutter (81 timer). I senere fold1 gjelder hull 6 008 av 17 288 vinduer (34,75 %). Posisjonskodingen teller radplassering, mens time-/sesjonsfelt kan gi indirekte klokkeinformasjon. Noen brudd kan være normale markedsstengninger; årsak er ikke fastslått for hvert hull. På de samme senere foreldrene er beskrivende 5min-AUC 0,502787 uten hull og 0,496839 med hull; gruppene har ulikt marked/klokkeslett, og ingen parret intervensjon eller usikkerhetsintervall viser at hullene forårsaker forskjellen. Å gi modellen eksplisitt forløpt tid er en hypotese som krever separat sammenligning og ny baseline, ikke en bevist forbedring. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_HISTORY_CLOCK_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `9eff85319123a5cee2ae6d0ed6e1b8a33025e77d3ad04684cd5cfa2c0379a374`.
10. **54 GC-gyldighetsfelt er normalisert som kontinuerlige.** 40 har boolsk kildetype; de øvrige 14 er double med 0/1 i undersøkt prefix. En sann verdi kan dermed normaliseres til 0 og falsk til negativ verdi; separat observasjonsmaske finnes. Tilstandene er fremdeles matematisk forskjellige, og skade er ikke målt. Sjekk produsent og domene før eventuell designendring.
11. **Nivåskifte og gamle terskler er generaliseringsrisiko, ikke bevist feil.** I nestet prefixmetadata har 202 av 204 aktive GC-dollarverdier en høyere maks etter utvidelse; dette viser minst én ny høy verdi, ikke frekvens eller årsak. Aktive level-/trendlinjeterskler er tilpasset 2009–2012, kausalt før modelltrening; hvor godt de passer dagens marked er ikke målt.
12. **Featuresemantikk må beholdes.** `smc_sweep_*_avwap_*` på XAU er vektet med OANDA-candlens prisoppdateringsantall, ikke dokumentert utført handelsvolum. GC har egne handelsvolumfelt. SMC-pivoter krever tre lukkede barer for bekreftelse: 15 minutter på M5 og tre timer på H1. Dette er kausale definisjoner, ikke beviste feil.

## Hva er kontrollert, og hva mangler?

- Den aktive lokale profilen har 2 654 fysiske signalposisjoner, inkludert verdier, masker og aliaser; det er ikke 2 654 uavhengige indikatorer. Inventaret dekker 8 947 felt/overflate-kombinasjoner, ikke 8 947 indikatorer. Alle åtte familier og D1/makro beholdes.
- Alle 1 205 registrerte GC-verdifelt finnes i aktiv kontekst og er med i tidligere numerisk indeks for publisert M1 til TF. Feltdekning varierer fra 3 til 227 808 observerte verdier. En separat tidligere rå-MBP-til-valgt-M1-kjede kvalifiserte 66 filer og deres sluttjoin. Dette etablerer ikke at hvert felt lærer prediktivt eller at alle kilde- og historikkspørsmål er lukket.
- Av 890 aktive XAU-MTF-felt/tidsramme-posisjoner matcher 730 både navn og kolonne i tidligere numerisk indeks for samme cache-hash. De resterende 160 er 32 kartgeometrifelt på hver av fem tidsrammer som ikke står i akkurat den indeksen. Det finnes fokuserte syntetiske tester; fravær fra indeksen beviser ikke feil eller at ingen annen kontroll finnes. Egen reell tall-/kausalitetsdekning må avklares før generell PASS.
- Den eldre tre-hoders modellen forbedret 5-minutters TRAIN-AUC fra 0,515888 til 0,533534 ved én til tre pass, men senere AUC gikk fra 0,503021 til 0,500950, med dårligere senere proper scores. Dette er ett gjenbrukt fold, ikke en universell dom over trening.
- En separat énpass-kandidat som tilføyde signed-bps-supervisjon fikk 5min senere AUC 0,496823 mot 0,503021 for baseline på samme 17 197 kvalifiserte rader; korrigerte parrede dag-/ukeintervaller for differansen inkluderer null. Heller ikke bps-feilen slo begge enkle referanser dokumentert. Dette er én seed og én gjenbrukt fold, ikke bevis for at bps-supervisjon generelt er uten verdi. Se `docs/DIRECTION_BPS_SAVED_COMPARISON_RESULT_20261027.json`.
- En tidligere ikke-lineær ET252-referanse oppnådde 5-minutters innen-fold-AUC 0,513092 mot native 0,503735 på 67 112 felles utviklingsrader; korrigerte parvise dag-/ukeintervaller støttet den relative forskjellen. Modellhistorikk, normalisering og arkitektur var ikke identiske. Dette er ikke urørt test eller økonomisk edge.
- D1/makro-feltene er aktive, men historiske COT-revisjonsversjoner og GLD-/VIX-tilgjengelighetsklokker er ikke fullstendig kvalifisert. Koden har forsiktige as-of-regler; framtidslekkasje er ikke bevist.
- En separat kontroll av faktisk lagrede modellprediksjoner brukte det eksisterende, avsluttede H1-EMA-slope-fortegnet på den samme senere folden. Ved negativ slope var faktisk 5min OPP 50,53 % av 5 535 kvalifiserte, og modellens gjennomsnittlige OPP-sannsynlighet 48,76 %; ved positiv slope var tallene 52,08 % av 11 662 og 48,73 %. Den observerte forskjellen er +1,54 prosentpoeng; modellforskjellen er −0,026 prosentpoeng. Forskjellen mellom gruppenes faktiske utfall har korrigert dagintervall −0,80 til +3,81 prosentpoeng og ukeintervall −0,87 til +3,97. Gruppene ble oppdaget på den samme gjenbrukte folden; kalenderblokkers utbyttbarhet er ikke etablert. Dette er en nyttig responsdiagnose, men beviser verken at H1-trend har prediktiv verdi, at feltet er frakoblet eller at modellen ikke bruker kombinasjoner. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_SLOPE_RESPONSE_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `3f2bd5814f2f9d207b81d66ddc7b845023f8a07e3bf1ba7a55699f38ff9f87fe`.
- Den fullførte 60-hoders treningskvitteringen registrerte ikke-null gradienter i GC-projeksjoner og -gater på alle fem tidsrammer både for første batch på 16 foreldre og siste batch på to. Alle 60 output-rader hadde gradient i disse to batchene. Det viser kontakt på gruppenivå i disse målingene, ikke at hvert felt er korrekt eller at modellen har lært nyttig retning.
- De 60 logitene deler en representasjon med 128 dimensjoner. Uvektet BCE fordeler seg over tilgjengelige horisontceller; første prefix har 1,69 % av cellene ved 5 minutter og 49,46 % ved 31–60 minutter. Dette er celletall, ikke målte gradientandeler. Et lite privat 32-foreldre-eksempel ga gradienter i alle 60 hoder; det beviser ikke verdien av hvert felt eller samspill.

## Neste nødvendige arbeid

1. Desimalauditen er fullført og konsumert. Gjenskapelige kildebytes, klokker, målpekere og opprinnelige skårer bestod; kvantifiseringen står over. Ingen ny modell ble evaluert.
2. En separat korrigert målberegning og et genuint 1–60-minutters målarkiv er nå laget fra kildeverifiserte desimalheltall. Alle opprinnelige klokker, radpekere og observasjonsmasker er beholdt, og alle 60 fortegnsmatriser for hele populasjonen og de to fold1-delene gjenskaper uavhengig audit. Alle 976 falske bevegelser er fjernet; ingen reelle OPP/NED er snudd. Det nye arkivet har egen identitet og er ikke koblet til trening. En separat kvalifisert målforbruker og ny baseline kreves før eventuell trening; ingen retningsbedring er derfor dokumentert. Se `docs/DIRECTION_EXACT_DECIMAL_GRID_TARGET_20261028.json` og materialiseringsresultat `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_TARGETS_20261028_001/OUTPUT/RESULT.json`, SHA-256 `71ea36e8fd8048a32aa1136f342264bdbd28122ae674dc1452a3d17207979648`.
3. Den forhåndsdefinerte 60-horisont-evalueringen er fullført og konsumert; ingen minutter har begge simultane bånd over 0,5. Hold gamle og korrigerte mål klart adskilt.
4. Mål faktisk bruk av de åtte eksisterende familiene og fem tidsrammene i den trente modellen med forhåndsdefinerte, parrede inngrepsgrupper på de samme kronologiske foreldrene. Gjenskap først lagrede originalprediksjoner, bevar verdier sammen med observasjonsmasker, og behold all ukjent side som ukjent. Et inngrep viser modellens følsomhet, ikke alene hva en retrent modell kan lære. Kontrollér de 32 kartgeometrifeltene per tidsramme som ikke står i den tidligere tallindeksen, og kvalifiser publiserings-/revisjonsversjoner for D1/makro.
5. Test eksplisitt forløpt tid mellom M5-rader som én separat hypotese mot identisk baseline, fordi faktiske vinduer ofte krysser hull. Behold eksisterende tidsfelt, familier og tidsrammer. En ny ONLINE-funksjon krever fersk initialbaseline; ingen endring er bevist nyttig før parret senere måling.
6. Test prioriteringen av 5min i felles 1–60-oppgaven som en annen enkeltfaktor. Første prefix gir 5min 1,69 % av de kvalifiserte loss-cellene, men celleandelen er ikke målt gradientbidrag. Behold alle 60 mål; sammenlign samme initialisering, inputs, eksponering og kronologiske rader før en eventuell vekting anses bedre. Den korrigerte målfilen må først få en separat kvalifisert forbruker og fersk baseline. Ingen av disse hypotesene gir ny treningstillatelse alene.

Revisjonsinventar og detaljfunn: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/INDEPENDENT_FEATURE_METADATA_AUDIT_20261028_001/MANIFEST_002.json`, SHA-256 `f16a48c01df167cdfaee7cade5c0351a17ae3add1fde15ec0f9751685153dd62`. Den peker til 24 scoped funn; rapporten er ikke en erklæring om fullført feature- eller edgekvalifisering.


## Korrigert målforbruker: faktisk koblingskontroll

Den separate forbrukeren er prøvd på hashbundne mål fra 118 249 foreldre og alle 60 minutter. Originalmålet passerte fortsatt de gamle 1/5/15-byteankrene; den korrigerte fasiten bevarte foreldreakser, klokker, kilderader og masker. Alle uavhengige fortegnsmatriser for hele utvalget og fold1 TRAIN/senere ble gjenskapt. De 17 288 lagrede senere prediksjonsradene og deres opprinnelige mål stemte med originalarkivet. Dette kontrollerer kobling, ikke læring.

Av 6 946 049 observerte bps-celler fikk 6 784 709 en annen flyttallsverdi. Median absolutt forskjell var 6,377e-13 bps, 99-persentil 2,476e-12 bps og maksimum 4,214e-12 bps. Dette er numerisk små målte forskjeller; virkningen av retrening er ikke testet. Fasiten fjerner fortsatt 976 falske fortegn uten observerte motsatte ikke-null-fortegn.

Engangskjøringens resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_CONSUMER_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `87b2240484f48b1c7341467947357e25cda28c8d47fcf91f43b87614fa664632`. Kontrollen brukte ingen modellforward, optimizer, TEST eller handel. En framtidig treningsrute må separat binde og autentisere filene og kontrollene, og ha fersk initialbaseline; den er ikke implementert eller admittert nå.


## Mulig flaskehals i sammensettingen

Kilde og første prefixes normaliseringsmetadata viser at den trente 60-hodersprofilen har åtte spesialistfamilier. I SMC-familien går 1 094 aktive numeriske kontekstkoordinater gjennom én lært 128-dimensjonal projeksjon; 616 er binære og 540 har `_observed`-suffiks. De er derfor ikke 1 094 uavhengige indikatorer. I M5-historien projiseres 316 GC-koordinater og 76 XAU-koordinater hver til 128 verdier, og resultatene summeres før familie-encoderen. GC-koordinatene finnes også navngitt i aktiv gjeldende kontekst.

En 316-til-128 lineær rute har, ved fast kontekststyrt gate, matematiske nullretninger for vilkårlige realvektorer. Dette beviser ikke at slike kombinasjoner finnes i ekte data, at nyttig retning går tapt, eller at hele modellen mangler en alternativ rute. Det peker på en konkret hypotese: mål først modellen med og uten kausalt sammenhengende GC-informasjon på identiske senere rader, etter at den gjenskapte originalprediksjonen er kontrollert. Ingen arkitekturendring eller treningsgevinst er konkludert.

Kilde-/metadata-kvittering: `docs/DIRECTION_GC_PROJECTION_SOURCE_REVIEW_20261028.json`. Ved kildekontrollen matchet alle 56 kildeeierne sine treningshasher. Siden er én read-only katalogvakt rettet; fremtidig replay må binde denne eksakte kildeovergangen. Ingen checkpointlesing, modellforward, trening eller TEST ble brukt i denne kildekontrollen.


## Hva ble faktisk oppdatert i den trente modellen?

En egen CPU-kontroll leste hashbundne start- og sluttcheckpoint etter 2 976 optimizersteg, verifiserte hele state-forseglingen og samme feature-view. Alle tre sammenslåingslagene `specialist_out`, `cross_tf_out` og `family_tf_cooperation_out` startet på null og hadde endret samtlige 16 384 vekter. SMC-kontekstprojeksjonen hadde numerisk rang 128 av 128, både før og etter trening, med NumPys standard FP32-rangtoleranse. Dette viser at grenene har fått parameteroppdateringer; det er ikke en måling av deres bidrag til prognosen.

I M5 SMC ble 40 320 av 40 448 GC-projeksjonsvekter endret, og 315 av 316 gate-biaser endret seg fra null. Også flere andre tidsrammegrupper hadde koordinater uten målbar parameterendring. Enkeltfelt og årsak er foreløpig ikke identifisert; neste kontroll skal binde navn og prefixstøtte før de omtales som ubrukte. AdamW hadde både læringsrate og vektforfall satt i recipe, så uendret FP32-verdi alene er ikke bevis for null gradient eller frakobling.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_PROJECTION_STATE_AUDIT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `57268760afd73589f2ab5d721ac7eb85c8932e44bfc44278093f2f78d4e00159`. Ingen modellforward, nye optimizersteg, TEST eller økonomimåling.


## Navngitte felt uten variasjon i første TRAIN-prefix

Den etterfølgende kontrollen koblet uendrede start-/sluttparametere til eksakte feltnavn og prefixnormalisering. **39 GC-historiekoordinater og 33 kontekstkoordinater hadde transformert TRAIN-støtte nøyaktig 0–0; 31 av historikoordinatene var også uendret i kontekstruten.** Dette er felt/tidsramme-posisjoner, ikke 39 selvstendige indikatorer. Historiegrunnlaget er den registrerte unionen av native rader i TRAIN-vinduene: 47 798 M5, 15 991 M15, 4 080 H1, 1 139 H4 og 425 D1; kontekstgrunnlaget er 47 602 beslutninger. Normaliseringseieren beregner støtte etter de kildeeide observasjonsmaskene.

De 39 hadde uendret projeksjonskolonne, gate-vektrad og gate-bias. En koordinat som er null over disse TRAIN-radene, bidrar ikke med variasjon i akkurat den ruten; separate tilgjengelighetsmasker eller andre korrelerte felt kan fortsatt bære informasjon. Mange posisjoner gjelder klokkekvalitet, mens D1 bearish orderblock-lav/høy/opprinnelsesvolum og første retouch, samt H4/D1 `signed_complete`, også inngår. Konstans er ikke i seg selv en kodefeil.

Som moteksempel hadde den delvis kjente handelsdeltaen ikke-null variasjon i normalisert TRAIN-støtte på alle fem tidsrammer. Dens GC-projeksjonskolonne, gate-vektrad, gate-bias og kontekstkolonne var ikke bit-identiske mellom start og slutt. Det viser variasjon og parameterendring, ikke at deltaen gir riktig retningsprognose. Ukjent aggressor er fortsatt ukjent.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_PROJECTION_FIELDS_AUDIT_20261028_002/OUTPUT/RESULT.json`, SHA-256 `083f330c0a7166114bd2926fd948e4db378e36de99bc5b02f27bcb7344ada40b`. Ingen nye modellforward, optimizersteg, TEST eller handel. Neste reelle spørsmål er hvor mye den ferdigtrente modellens prognoser og senere treff påvirkes når en navngitt rute gripes inn i; dette krever en separat kildebundet kontroll med gjenskapt originalprediksjon.


## Foreløpig intervensjon i trent GC-historierute

Et forhåndsvalgt utvalg tok hver tiende opprinnelige OOF-batch: 109 batcher og 1 736 senere utviklingsforeldre. Alle 60 lagrede logits ble gjenskapt byte for byte før den samme batchen ble kjørt med de 25 GC-historieprojeksjonene midlertidig nullstilt. Gjeldende GC-kontekst og andre ruter forble aktive. Gjennomsnittlig absolutt endring i logit var 0,000523 over alle 60 utganger; største endring var 0,003202. Dette er en intern intervensjon, ikke en modell trent uten GC.

Ved 5min var AUC på de samme 1 726 observerte ikke-null-utfallene 0,514611 med original modell og 0,514270 med nullstilt GC-historie. Parret differanse (nullstilt minus original) hadde korrigert dagintervall −0,000899 til 0,000157 og ukeintervall −0,000887 til 0,000106; begge inkluderer null. Det dokumenterer ikke bedre eller dårligere retning fra denne ruten i utvalget. Dette lille utvalget er ikke samme populasjon som full-fold-AUC rundt 0,500175.

**Evidensgrense:** Produsenten utførte 218 modellkall og skrev en intern COMPLETE-terminal, men sluttet med exit 1 da lesesperren blokkerte mappe-fsync etter atomisk rename. Opprinnelig kjøring forblir FAILED. En separat fullført recovery synkroniserte mappene, hashbandt originalresultatet og gjenskapte originalskårene nøyaktig fra lagrede prediksjoner og korrigert fasit for alle 60 minutter. Ablasjonslogits ble ikke lagret; ablasjonsskårene og intervallene er derfor foreløpige og ikke uavhengig etterregnet. Blokktrekningene forutsetter utbyttbare dager/uker; ingen generalisering eller økonomi er kvalifisert.

Foreløpig resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_INTERVENTION_20261028_003/OUTPUT/RESULT.json`, SHA-256 `c16ebb9d8ef9a6c198cc177bc029e15106860d81f008feda53c03be5a778dbee`. Recovery: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_INTERVENTION_RECOVERY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `ce131292529d42cda9653c1fa263c84be46c7ea0077d8d557a627d26c0cd0359`. Ingen optimizersteg, TEST eller handel.


## Retting av publiseringsvakten

Årsaken til exit 1 var at den eksisterende inputvakten avviste en read-only `O_DIRECTORY`-åpning av jobbens egen outputmappe under `fsync`. Én kodelinje legger bare denne eksakte mappen til samme katalogliste som allerede avviser skriveflagg. Et syntetisk testtilfelle bekrefter at egen mappe kan åpnes som read-only katalog, mens skriveflagg og uautoriserte søskenstier fortsatt avvises. 26 fokuserte tester bestod.

Dette endrer ikke modellfunksjonen eller resultatet av det feilede forsøket. 55 av 56 kildeeierne matcher den historiske treningsrecipe; den ene forskjellen er denne katalogvaktrettingen. En senere modellreplay må separat binde og verifisere kildeovergangen. Se `docs/DIRECTION_GC_OUTPUT_DIRECTORY_GUARD_FIX_20261028.json`.


## Verifisert GC-historieintervensjon i senere utviklingsperiode

En separat, forhåndsbundet måling brukte den samme frosne første-prefixmodellen på hver tiende 16-raders batch fra 2025 Q2: 1 760 foreldre. En lagret fold1-referansebatch ble gjenskapt byte for byte. Deretter ble de 25 GC-historieprojeksjonene midlertidig nullstilt i én arm; gjeldende GC-kontekst og øvrige ruter forble aktive. Begge prediksjonsarmene og korrigerte mål ble lagret uten modellvekter eller rå features.

På 1 753 kvalifiserte 5min-rader var AUC 0,489688 med original modell og 0,489699 med nullstilt GC-historie. Differansen (nullstilt minus original) var +0,000010; korrigert dagintervall var −0,000534 til +0,000536 og ukeintervall −0,000616 til +0,000785. Ved 60min var begge AUC 0,484323 på 1 688 kvalifiserte rader, og også der inkluderte begge intervaller null. Gjennomsnittlig absolutt logitendring over alle 60 utganger var 0,000755. Punktestimater for alle 60 minutter står i resultatet; intervaller var forhåndsvalgt for 5 og 60 minutter.

En separat capped CPU-kontroll gjenkoblet hver lagret rad til den korrigerte fasiten og gjenskapte begge armenes AUC, logloss og Brier ved alle 60 minutter, samt 10 000 dag-/ukeblokktrekninger ved 5/60 minutter med en egen rangsum-/blokksøk-implementasjon. Største skåravvik var 5,56e-17 og største intervallavvik 3,28e-18. Ingen ny modellforward ble brukt i kontrollen.

Dette viser liten målt påvirkning fra den bestemte GC-historieruten i dette utvalget og ingen dokumentert retningsforbedring ved de forhåndsvalgte horisontene. Det viser ikke at all GC-informasjon er uten verdi: kontekstruten står på, intervensjonen kan skape en unaturlig intern tilstand, og perioden er tidligere gjenbrukt utvikling. Blokkinferensen forutsetter utbyttbare kalenderblokker. Ingen generalisering, handel eller nettoøkonomi er kvalifisert.

Modellresultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_LATER_FOLLOWUP_20261028_001/OUTPUT/RESULT.json`, SHA-256 `02d8cf20527f9219ec82b974b0dd741522539cab4f7a6790358d1b4d65cdd5e6`. Uavhengig kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_MTF_LATER_VERIFY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `f641aefdd33805cf467d1d639641d19cb59938fb33d9b2d9e401b9ffc2bb7ba0`. Parede prediksjoner er hashbundet under RUNS og ikke lagt i Git. Ingen optimizersteg, TEST eller handel.


## GC-kontekst og lokal sekvens sammen med GC-historie: uavhengig kontrollert

Den opprinnelige historietesten lot GC være aktiv i lokal sekvens og gjeldende kontekst. Kildeprofilen har 2 410 slike GC-koordinater, fordelt på 1 205 verdi-/observasjonspar. Hvert par har eksakt alias mellom lokal sekvens og kontekst, og maskene styrer ikke andre felter. Den nye kontrollen satte disse parene til manglende i lokal sekvens, snapshot og kontekst samtidig som de 25 GC-historieprojeksjonene ble satt til null. Andre navngitte inndata var uendret. Dette er et kunstig inngrep i en ferdig modell, ikke bevis for at all GC-avledet informasjon er fjernet.

På de samme 1 760 forhåndsvalgte 2025 Q2-foreldrene ble samtlige opprinnelige prognoser for alle 60 minutter gjenskapt byteidentisk. Gjennomsnittlig absolutt endring i modellens logit var **0,07858**, mot **0,000755** i den tidligere testen av historieprojeksjonene alene. Modellen reagerer dermed på denne samlede intervensjonen; det fastslår ikke at den bruker informasjonen riktig eller at enkelte indikatorer er nyttige. Ved fem minutter var AUC **0,489688** original og **0,493635** etter inngrepet på 1 753 kvalifiserte rader. Ved 60 minutter var tilsvarende verdier **0,484323** og **0,494325** på 1 688 rader. Parrede dag-/ukeintervaller for AUC-forskjellen inkluderer null på begge horisontene. Alle 60 punktmål er lagret, men usikkerhetsbånd er kun beregnet for de to forhåndsvalgte horisontene.

Dette er én tidligere gjenbrukt utviklingsperiode, og inngrepet ble valgt etter en tidligere analyse. Undertrykking kan lage en unaturlig modelltilstand. En uavhengig CPU-kontroll gjenskapte både armer og alle 60 minutter, korrigerte målkoblinger og 5/60-minutters blokkbånd med separate algoritmer. Største punktavvik var 5,6e-17 og største intervallavvik 4,4e-18. Tallene viser verken generalisering eller lønnsomhet. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_FOLLOWUP_20261028_002/OUTPUT/RESULT.json`, SHA-256 `3759c43b7eed35b40eb16817592a6a8fa4070bcd2354ee7e7be33b04028566ee`. Ingen optimizersteg, TEST eller handel.


## Konkret femminutters skjevhet i modellens retning

I det korrigerte Q2-utvalget var 880 av 1 753 femminuttersutfall opp (50,20 %), mens modellen ga positivt opp-logit på bare **73 av 1 753** (4,16 %). Med fast terskel 0 var 864 av 1 753 retninger riktige (49,29 %); alltid-LONG som forhåndsvalgt referanse traff 880 (50,20 %). Dette er evaluerte markedsretninger, ikke handler, og det er ikke en optimalisering av terskel. Modellens AUC på de samme radene var 0,489688; en terskeljustering alene endrer ikke AUC.

Det samme mønsteret er synlig utenfor dette utvalget: på de 17 197 kvalifiserte Q1-radene med opprinnelige etiketter var 8 870 utfall opp (51,58 %), mens modellen predikerte opp 2 292 ganger (13,33 %). Femminuttersfasiten i TRAIN-prefix hadde 24 081 opp og 23 227 ned (50,90 % opp, opprinnelige etiketter). En enkel ubalanse mot ned i hele dette treningsprefixet forklarer derfor ikke den sterke nedskjevheten alene.

På de **samme 256** lagrede TRAIN-radene var 128 utfall opp. Ved tilfeldig initialisering ga femminuttershodet opp én gang; etter 2 976 optimizersteg ga det opp 55 ganger. Treningens logloss på disse radene falt fra 0,7848 til 0,6927 og AUC steg fra 0,4410 til 0,5334; dette er et lite treningsutvalg, ikke generalisering. Kildeeieren bruker seedet standardinitialisering for de opprinnelige retningshodene og et uvektet BCE-tap over alle kvalifiserte celler i de 60 horisontene. Initial skjevhet, vedvarende kalibreringsskjevhet og et mulig samspill med treningsmålet er konkrete hypoteser; tallene beviser ikke hvilken mekanisme som forårsaker manglende edge. Verken en ny initialisering, en ny terskel eller ny trening er kvalifisert som forbedring.

De eksakte sammenligningene for hvert minutt 1–60, inkludert kilde- og prediksjonshasher, står i `docs/DIRECTION_SAVED_CALIBRATION_DIAGNOSIS_20261028.json`, SHA-256 `3d33cc50dd960d99a47873b5dcba3b1092cf6968a135952299c8a17ea23a206a`. Uavhengig verifikasjon av GC-målingen: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_CONTEXT_VERIFY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `00a84af7b6edda97e7574d8b464c615574940360150458577a289f1fd0d45e3f`. Ingen nye modellkall, optimizersteg, TEST eller handel.


## Fem og seks minutter fra samme beslutning

En ekstra beskrivende gjennomgang sammenlignet nabohorisontene på nøyaktig de radene der begge fasitene var gyldige. I Q2-utvalget ga modellen ulike faste opp/ned-tegn ved fem og seks minutter på **1 669 av 1 751** rader (95,3 %); de to faktiske utfallene hadde ulike tegn på **224 av 1 751** (12,8 %). I Q1 med opprinnelige etiketter var tallene henholdsvis 14 792 av 17 168 (86,2 %) og 2 431 av 17 168 (14,2 %).

Dette betyr ikke at prognosene bryter en matematisk regel: ulike framtidsminutter kan ha forskjellige utfall. Modellens gjennomsnittlige Q2-sannsynligheter var bare **48,13 % opp ved fem minutter** og **52,03 % ved seks**, og begge sannsynlighetene lå mellom 45 % og 55 % på 94,1 % av de felles radene. En fast 50 %-terskel gjør derfor svak usikkerhet til svært forskjellige binære svar. Dette er et konkret problem for en strategi som skulle tvinge fram retning fra disse skårene; det er ikke bevis for at en ny terskel, abstensregel eller treningsendring gir edge. Samtlige 59 nabopar er lagret i samme diagnose; eksempelet ble valgt etter inspeksjon av gjenbrukte utviklingsdata.


## Sjelden kartinformasjon i første treningsprefix

Den faktiske kompaktprofilen har 32 lokale kartkoordinater, én kontekstkoordinat og 39 kartkoordinater på hver av M5, M15, H1, H4 og D1. Jeg koblet den lagrede feltinventaren til normaliseringen og den brukte modellens hash. Alle disse koordinatene hadde observerte verdier og variasjon i sine respektive TRAIN-populasjoner. Det avkrefter at hele denne familien bare var konstant i dette prefixet, men fastslår ikke at indikatorenes formler er riktige eller at modellen har lært nyttig bruk.

Tre av feltene var svært sjelden ikke-null i M5-historieruten: `geomline_retest_fail_up` **7**, `geomline_retest_fail_down` **11** og `geomchan_apex_proximity` **3** av 47 798 unike native rader. I den lokale signalruten var tilsvarende antall 295, 290 og 5 106 av 47 878 rader. Disse populasjonene er ikke helt de samme, og en native rad kan inngå flere ganger i modellvinduer; tallene er derfor ikke antall uavhengige treningseksempler. Kilden bruker forskjellige kandidatvinduer og bånd: lokal M5 96 rader og 0,98657 ATR, M5-historie 16 rader og 0,91616 ATR. Dette er en dokumentert konfigurasjonsforskjell og en mulig forklaring på ulike forekomster, ikke et bevist avvik eller en målt effekt på retning.

Tersklene for nivå-/trendlinjeregistrene er dessuten tilpasset et historisk vindu som slutter 1. januar 2013, før den aktuelle modelltreningen. Det er ikke framtidslekkasje; hvor godt tersklene fungerer på nyere gullmarked er ikke kvalifisert. De 32 aktive kartposisjonene per tidsramme som mangler i den tidligere numeriske indeksen, trenger fortsatt egen genuin formel- og kausalitetskontroll. Underlagsrapport: `docs/DIRECTION_CHART_TRAIN_SUPPORT_20261028.json`, SHA-256 `8a110c09d5e4332a06b11a0d9445e732d531dd764c1c4f609c0ee26508979959`. Ingen nye modellkall, optimizersteg, TEST eller handel.


## Faktisk H1-bygg fra 2009: tidligere avvik er forklart

Jeg gjenskapte H1-trendlinjene fra de 17 hashbundne native M5-årsdelene, fra 1. juni 2009 og fram til juli 2025. Alle **1 070 080** M5-rader som overlapper den bearbeidede filen hadde eksakt samme klokke og OHLCV. Med den opprinnelige forhistorien, frosne parameterne og den faktiske float64-kildeveien stemte **alle 31 aktive H1-trendlinjefelt byteidentisk på 90 075 cache-rader**; å dele kjøringen og føre tilstand videre ga også identiske resultater. Dermed inngår ingen avvik i de undersøkte TRAIN-, Q1- eller Q2-vinduene. De 14 H1-radene jeg tidligere meldte som ulike oppstod fordi min kontroll feilaktig startet etter at viktig forhistorie var skåret bort. De er ikke en påvist feil i modellens H1-input.

Dette er fortsatt en gjenskaping med samme formel-/tilstandseier, ikke en uavhengig matematisk kvalitetssikring eller bevis på at modellen lærer av feltene. Den separate standard-CLIen har en annen float32-behandling enn den identifiserte native byggeveien; de to kan ikke regnes som ekvivalente uten egen fullhistorisk sammenligning. Ingen cache, modellvekter eller handelsregler er endret. H1-resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_NATIVE_HISTORY_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `94b524013ab9f3cf98e5ad679954b036ba15bd12286e8f87d0c08c7dcfbf4dae`.

## Provenienskorrigering: den faktiske cachen har forhistorie fra 2009

Videre kontroll av den konkrete M5-manifesten og `build_entry_exit_m1_enriched_frame_v1.py` viser at denne cachen ble beregnet fra `native_stage`, før den bearbeidede M5-filen og cachen ble skåret til samme senere tidsakse. Native kilden begynner i juni 2009; den bearbeidede filen vi brukte i replay begynner i juni 2010. Den statefulle trendlinjeberegningen i tidligere kontroller manglet altså opprinnelig forhistorie. Dessuten brukes standard-CLIens float32-cast i en annen byggevei; den er ikke dokumentert som byggeveien for denne cachen.

En separat hypotetisk float32-kjøring med den forkortede kilden avvek i 706 173 H1-celler, langt mer enn double-kjøringen. Den kan ikke brukes til å kalle faktisk cache feil, og betegnelsen «builder-tro» i den historiske planen er trukket tilbake. Neste nødvendige kontroll er å gjenskape fra de hashbundne native årspartene fra 2009, sjekke overlapp mot den bearbeidede kilden, og så sammenligne faktiske modellvinduer. Tidligere originalresultater bevares som avgrensede tester av andre kilderepresentasjoner. Hypotetisk float32-resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_BUILDER_PRECISION_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `f83ea53ff0dea30a03545ab81329b2009ad917e27cc7051fdf69aa27b16418c1`. Ingen modellkall, optimizersteg, TEST eller handel.

## Presisjonskorrigering for D1/H4/H1-kontrollene

De tre kontrollene nedenfor brukte M5-filens opprinnelige double-priser. Ved kontroll av cachebyggeren oppdaget jeg at den først caster `open`, `high`, `low`, `close` og `volume` til float32. Sammenligningene brukte double-priser med forkortet historikk; den påståtte faktiske float32-byggeveien ble siden avkreftet for denne cachen, som forklart ovenfor. Jeg omtaler ikke forskjellene som cachefeil. En separat kontroll må bruke byggekodens presisjon før eventuell feilkonklusjon. Målingene og originalkvitteringene er bevart.

For H1 skilte double-beregningen seg i fire kanalposisjoner på 10 unike H1-rader som inngår i 3 527 av 47 602 TRAIN-vinduer; i Q1 inngikk fire avvikende rader i 3 449 av 17 288 vinduer. I Q2-utvalget var antallet null. Noen felt endret diskret `geomchan_active`, så dette er ikke bare avrunding av ferdige tall. Det viser at endret presisjon kan forandre inngangsverdier i den undersøkte modellen; det viser ikke at double er riktig i alle kontrakter, at dagens cache er feil, eller at forskjellen forklarer manglende edge. H1-resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H1_TRENDLINE_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `2998b8d8ff3b74f9891c03f1640529f94b61654dfb00bee1d09202d764b14923`.

## D1-trendlinjer gjenskapt fra prisdata: eldre avvik

Jeg gjenskapte 31 aktive D1-trendlinjefelt fra de opprinnelige hashbundne M5-prisene i double-presisjon, med samme lukkede D1-klokke, Wilder ATR, frosne parametere og hele den tilgjengelige kausale historikken før 1. juli 2025. Å dele beregningen ved treningsgrensen og føre tilstanden videre ga eksakt samme float32-verdier og manglendehet som én sammenhengende beregning på alle 3 889 D1-rader.

Sammenlignet med den lagrede modellen brukte, er det **7 551 avvikende celler** over hele denne historikken. For de 174 D1-radene som lukkes i den angitte TRAIN-perioden og 127 senere utviklingsrader er alle 31 feltene derimot byteidentiske. Første avvik starter i 2010. Dette er et reelt avvik mellom gjenskaping og cache, men årsak, siste avvikstidspunkt og skillet mellom manglendehet og numeriske verdier er ennå ikke lokalisert. Modellens D1-vinduer inneholder 252 tidligere rader, så samsvar for nåværende TRAIN-rader alene utelukker ikke at eldre avvik er konsumert. Ingen forbedring eller årsak til manglende edge er bevist.

Dette er en gjenskaping med nåværende kildeeier, ikke en uavhengig formelimplementasjon. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_D1_TRENDLINE_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `7e1b469bb33598527976b03ec7fd8f892b6f18e08801de2e5d115c40444970ef`. Ingen modellkall, optimizersteg, TEST eller handel.


Den separate lokaliseringen fant at de 7 551 cellene omfatter **460 forskjeller i manglendehet og 7 091 ulikheter i endelige float32-biter**, fordelt på 910 D1-rader. Siste avvikende rad starter **18. april 2023 kl. 22 UTC**. Med kildens faktiske én-dags lukking og 252-raders D1-vindu ligger ingen av disse radene i de **47 602** treningsvinduene, **17 288** Q1-vinduene eller **1 760** undersøkte Q2-vinduene. Tidligste D1-rad i treningsvinduene starter 9. mai 2023 kl. 22 UTC. Kontroll med den kanoniske window-funksjonen bekreftet første og siste vindu i hver kronologisk gruppe; alle gruppenes beslutningsklokker er stigende. For disse undersøkte vinduene samsvarer altså alle 31 D1-trendlinjefeltene med gjenskapingen.

Dette forklarer ikke modellens manglende retning. Det eldre historiske avviket er bevart og kildeårsaken er ukjent; andre perioder og tidsrammer er ikke dekket. Samme kildeeier på begge sider er fortsatt ikke uavhengig matematisk formelbevis. Lokalisering: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_D1_MISMATCH_LOCALIZATION_20261028_001/OUTPUT/RESULT.json`, SHA-256 `590ab9d42c34f56aeba5be50ac33ec140eb565b3a2bcaf25d28125633c9ee525`. Kildeeierens vinduskontroll: `docs/DIRECTION_D1_WINDOW_OWNER_CHECK_20261028.json`, SHA-256 `72164316e48f1159b2c1f8f0474503ce9cfb7ca329cea80c5b20f5bbefbe544a`. Ingen modellkall, trening, TEST eller handel.


## H4-trendlinjer og faktiske modellvinduer

Den samme kildebundne double-kontrollen mot float32-basert cache ble utført på H4, med H4-klokkens egen tidsforskyvning, ATR, historikk og 96-raders modellvindu. Over 23 270 H4-rader og 31 aktive trendlinjefelt stemte sammenhengende og delt beregning eksakt. Mot lagret cache fant jeg **3 235 eldre celleavvik**: 448 i manglendehet og 2 787 i endelige float32-biter, over 266 H4-rader. Siste avvikende H4-rad starter 2. februar 2023 kl. 14 UTC.

Ingen av avvikene inngår i de 47 602 undersøkte TRAIN-vinduene, 17 288 Q1-vinduene eller 1 760 valgte Q2-vinduene. Første og siste vindu i hver gruppe ble kontrollert med den faktiske cache-sliceren; gruppenes beslutningsklokker er stigende. Dermed samsvarer disse 31 H4-feltene med gjenskapingen på alle undersøkte vindusrader. Også her er historisk avviksårsak ukjent, og gjenskaping med samme eier er ikke uavhengig formelbevis eller dokumentert edge.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_H4_TRENDLINE_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `e0162ea76483cc5e9c1013c3bf4e94711a57c3ceeb2447f471b256aac5ed4d44`. Kildeeierens vinduskontroll: `docs/DIRECTION_H4_WINDOW_OWNER_CHECK_20261028.json`, SHA-256 `5edc05d74cb09cb8f69544c5353ecb786f5637a8913a8418a93b48d417511171`. Ingen modellkall, optimizersteg, TEST eller handel.


## Faktisk M5-bygg fra 2009: 31 trendlinjefelt

Jeg brukte samme opprinnelige M5-kilde og forhistorikk som i H1-kontrollen, men den faktiske M5-klokken, M5-parametrene og 16-raders M5-ruten. Native og bearbeidet OHLCV var like på alle 1 070 080 overlappende M5-rader. Alle **31 aktive trendlinjefelt** stemte byteidentisk mot cachen på disse radene; en sammenhengende beregning og en delt beregning med videreført tilstand ga også identiske float32-verdier. Det er null avvik i de undersøkte TRAIN-, Q1- og valgte Q2-vinduene.

Dette viser at den navngitte M5-dataveien leverer de samme verdiene som gjenskaping med den eksisterende kilden, ikke at feltene isolert er matematisk uavhengig kontrollert eller prediktivt nyttige. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_NATIVE_HISTORY_REPLAY_20261028_001/OUTPUT/RESULT.json`, SHA-256 `d1c39ec3f8ad7edbda23f620ea6ce5fec606eebe9ad763080cc1847c7dedd3ed`. Ingen modellkall, optimizersteg, TEST eller handel.


## Trendlinjefeltene på alle fem tidsrammer med faktisk forhistorie

Med de samme 17 hashbundne native M5-årsdelene og full historikk fra juni 2009 er de **31 aktive trendlinjefeltene** nå gjenskapt mot cache for alle fem tidsrammer før juli 2025. Resultatet er null avvik i float32-biter og manglendehet på **1 070 080 M5-rader, 357 385 M15-rader, 90 075 H1-rader, 23 270 H4-rader og 3 889 D1-rader**. Sammenhengende beregning og delt beregning med videreført tilstand samsvarer også på hver tidsramme. De undersøkte TRAIN-, Q1- og valgte Q2-vinduene har null avvik. De tidligere avvikene ved 2010-start er forklart av utelatt forhistorie i min kontroll; originalkvitteringene er bevart.

Dette er en omfattende kildeeierreplay, ikke en uavhengig matematisk implementasjon av de 31 formlene, ikke bevis for at modellen lærte dem og ikke et edge-resultat. Av de 32 aktive posisjonene per tidsramme som ikke lå i den tidligere numeriske indeksen, er den siste `close_position_range20`; den gjenstår egen uavhengig kontroll. Resultater: M5 `d1c39ec3f8ad7edbda23f620ea6ce5fec606eebe9ad763080cc1847c7dedd3ed`, M15 `1f06f563c2258c9766c53963f8883a3c9f84d6c1dd8b3f1142f67da554442c38`, H1 `94b524013ab9f3cf98e5ad679954b036ba15bd12286e8f87d0c08c7dcfbf4dae`, H4 `debdc7287f3e8f87ffcad1bbe72469446230d0670b24f3f66c0639e5a55b33c0`, D1 `52e8b24b0e0df6a4684847071bd15563c93f931e7cfb5cd5b7117b983db93fd7`. Ingen modellkall, optimizersteg, TEST eller handel.


## Uavhengig formelkontroll: posisjon i 20-perioders prisintervall

For `close_position_range20` beregnet jeg separat høyeste high og laveste low over de siste 20 lukkede native stolpene med NumPy, og deretter `(close − laveste low) / (høyeste high − laveste low)`. Null intervall behandles som manglende. Kildepriser, klokker og felles overlapp ble kontrollert mot de opprinnelige hashbundne 2009-kildene.

Resultatet stemte **byteidentisk på alle fem tidsrammer**, totalt 1 544 699 undersøkte native rader før juli 2025, med null avvik i TRAIN-, Q1- og de valgte Q2-vinduene. Dette er en selvstendig aritmetisk kontroll av én formel, ikke bevis på at den gir bedre prognoser. Sammen med den tidligere cachebundne numeriske indeksen er **735 av 890 aktive MTF-felt-/tidsrammeposisjoner** dekket av uavhengig numerisk kontroll i denne sammenstillingen. De øvrige 155 er 31 trendlinjefelt på fem tidsrammer: de har nå eksakt replay med samme kildeeier, men ikke uavhengig matematisk formelbevis.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CLOSE_POSITION_NUMERIC_20261028_001/OUTPUT/RESULT.json`, SHA-256 `7f381bdf8d876471fd67ae0b2d8289737b3bdc42d7d40140bc2b43ba11f9598a`. Ingen modellkall, optimizersteg, TEST eller handel.


## Faktisk senere støtte for GC-felt modellen ikke fikk variasjon i

Jeg kontrollerte den opprinnelige, skrivebeskyttede og hashbundne `ctx_cont`-filen mot eksakte foreldre-/klokkeakser og modellens egne observasjonsmasker og første-prefix-normalisering. Alle **33** tidligere identifiserte GC-kontekstkoordinater var normalisert null på samtlige **47 602** TRAIN-foreldre. Den tidligere checkpointkontrollen viser at de tilhørende kolonnene i akkurat kontekstprojeksjonen var uendrede fra start til slutt. Dette er ikke det samme som at hele modellen eller andre ruter var uendrede.

På **275 av 17 288** Q1-foreldre fikk `gc_d1_observed_proxy_ob_bull_age_bars` ikke-null verdi. På **80 av 1 760** fast utvalgte Q2-foreldre fikk tre bearish orderblock-verdier ikke-null støtte: low, high og opphavsvolum, med maksimum **7,12**, **7,22** og **12,27** etter den opprinnelige normaliseringen. Retouch-feltet var ikke-null på 32 av disse. Her var verdiene observert, ikke konstruerte erstatninger for ukjente data.

Dette er en konkret blind flekk i én modellrute: modellen fikk ikke variasjon i disse koordinatene i første TRAIN-prefix, men møter variasjon senere. Det beviser ennå ikke hvor mye prognosen påvirkes, om informasjonen finnes andre steder, eller at det er årsaken til svak retning. Neste nødvendige kontroll er et parret inngrep i bare disse navngitte projeksjonskolonnene, på de samme inputdefinerte radene og med gjenskapt originalprognose. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_UNSEEN_GC_SUPPORT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `48b28abd82380efa51ab4ff8912152e586b4535c615c5c7091eae68bc910d3c7`. Ingen nye modellkall, optimizersteg, TEST eller handel.


## Direkte effekt av tidligere usette GC-verdier

Fem navngitte normaliserte GC-verdier var null gjennom første TRAIN-prefix; kolonnene i SMC-kontekstprojeksjonen ble ikke oppdatert. På inputvalgte senere batcher ble bare disse fem verdiene undertrykt i akkurat denne projeksjonen, mens masker, lokale aliaser, GC-historikk og andre innganger ble bevart. Alle de 23 opprinnelige batchene ble gjenskapt byteidentisk. Ingen 5min-retning skiftet på 275 aktive Q1- eller 80 aktive Q2-foreldre. Gjennomsnittlig absolutt sannsynlighetsendring ved 5min var 0,00190 og 0,07463 prosentpoeng; maksimal endring var 0,00903 og 0,25671 prosentpoeng. På alle 60 horisonter skiftet henholdsvis 9 av 16 500 og 47 av 4 800 foreldre-horisontceller fortegn, fordelt på 9 og 28 foreldre. Dette avgrenser virkningen i én rute og ett sett gjenbrukte utviklingsdata; det er ikke en retrening, en årsaksforklaring på manglende edge eller bevis for at GC-feltene er unyttige.

Resultat `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_UNSEEN_GC_PROJECTION_20261028_001/OUTPUT/RESULT.json`, SHA-256 `b4a42e6189a493acd0d45fe47b3972be7fad98f6a469da18e5898ea03ebf86bf`; uavhengig NumPy-kontroll `docs/DIRECTION_UNSEEN_GC_PROJECTION_CHECK_20261028.json`, SHA-256 `15620cc577cfd42ec66b2458f28e4ad87140f921433fd86497c5b30343b4689d`.


## Skjevheter og sammensetning av retningshodene

På nøyaktig de samme 17 157 Q1-radene med gyldige korrigerte utfall for både 5 og 6 minutter predikerte modellen OPP på **13,33 % og 99,38 %**, mens fasiten var OPP på **51,57 % og 50,83 %**. AUC var henholdsvis **0,49987 og 0,49355**. I det tidligere fastsatte Q2-utvalget var andelene **4,17 % og 99,49 %**, mens fasiten var **50,26 % og 50,60 %**, med AUC **0,48967 og 0,46826**. På hele første korrigerte TRAIN-prefix var den faktiske OPP-andelen **50,91 % og 51,00 %** for de to horisontene; dette forklarer ikke de ekstreme prediksjonsandelene som enkel klasseubalanse.

På de samme 256 lagrede TRAIN-radene før trening var 6min-hodets OPP-andel 52,73 %; etter 2 976 oppdateringer var den 98,44 %, mens faktisk OPP var 52,34 %. Det lille utvalget er ikke representativt alene, men viser at den målte skjevheten ikke bare var en uendret initialbias. I Q1 var 5-/6min-hodenes logitkorrelasjon 0,171, i Q2 −0,046; fasitens fortegnskorrelasjoner var 0,717 og 0,744. For alle 59 tilstøtende par var median prediksjonskorrelasjon 0,827 i Q1 og 0,742 i Q2, mens median fasitkorrelasjon var 0,885/0,892. Seks og åtte prediksjonspar hadde negativ korrelasjon. Disse tallene er beskrivende; nærliggende sannsynligheter må ikke være like, og analysevalget kom etter inspeksjon av gjenbrukte utviklingsfold.

Kilden viser tre opprinnelige uavhengig parameteriserte 1/5/15-logiter og 57 øvrige uavhengig parameteriserte logiter på samme 128-dimensjonale representasjon. BCE summeres over alle gyldige tidsmål uten eksplisitt kobling mellom naboene. En felles, tidskoblet utgang er derfor en konkret hypotese å teste mot samme initialbaseline og alle 60 mål, ikke en bevist forbedring. Den målte ned-/oppskjevheten er reell for disse dataene, men den alene forklarer ikke hvorfor AUC er nær 0,5.

Detaljer og alle 59 nabopar: `docs/DIRECTION_HEAD_COMPOSITION_DIAGNOSIS_20261028.json`, SHA-256 `2af5f7bd7ff5c2623318db972b15b0f5bf0efcc9cb523ce0d7b1683c1279717c`. Ingen nye modellforward, optimizersteg, TEST eller handel.


## Enkel, tidsmessig tilgjengelig prisreferanse

Som en utforskende kontroll brukte jeg fortegnet på M1-midtprisendringen fra **10 til 5 minutter før beslutningen**, og krevde at sluttcandle var lukket før beslutningen. Både videreføring og motsatt retning er beregnet; ingen modell er trent. Alle sammenligninger bruker nøyaktig samme gyldige foreldre og korrigerte 5min-fasit. På **17 033 Q1-rader** traff motsatt retning **50,75 %** og alltid OPP **51,61 %**; referansens AUC var **0,50799** mot modellens **0,50052**. Parrede dag- og ukeintervaller for AUC-differansen inkluderer null.

På det tidligere fastsatte **Q2-utvalget på 1 738 rader** traff motsatt retning **52,47 %** mot **50,17 %** for alltid OPP. Referansens AUC var **0,52473** mot modellens **0,48931**; parret AUC-differanse hadde dagintervall **0,01013–0,06163** og ukeintervall **0,01813–0,05451**. Trefferatens forskjell mot alltid OPP hadde dagintervall **−0,80 til +5,36 prosentpoeng** og ukeintervall **+0,11 til +4,44 prosentpoeng**. På første TRAIN-prefix traff samme motsatte regel **50,56 %** mot **50,91 %** for alltid OPP.

Dette er en *etterundersøkelse* av data som allerede er brukt til utvikling, ikke en forhåndsregistrert og urørt bekreftelse. Blokkintervallene forutsetter utbyttbare dager/uker og korrigerer ikke for hvordan idéen ble funnet. M1-candlens åpningspris har ikke eksakt quote-tid, og kostnader/fyll er ikke målt. Resultatet gir en konkret enkel referanse å forhåndsdefinere før senere sammenligning, men dokumenterer verken generalisert retningsedge eller lønnsom trading. Det er heller ikke kontrollert at den eksakte M1-prisendringen inngår direkte i modellens features.

Resultat: `docs/DIRECTION_PAST_MOVE_REFERENCE_DIAGNOSIS_20261028.json`, SHA-256 `1d1273e9e500fcf32586fbc6ce24e1d663d44155b64a8f6e1b77d3cd8ba867e0`; uavhengig kontroll av punktestimater og fem bootstrap-trekninger per blokkmetode er bestått. Ingen nye modellforward, optimizersteg, TEST eller handel.


## Sammenheng før og etter trening

Det er ikke riktig å si at hele modellen er frakoblet eller ikke lærer noen sammensetning. På nøyaktig de samme **256 lagrede TRAIN-foreldrene** økte median korrelasjon mellom de 59 parene av nabologiter fra **0,0303 før trening til 0,7787 etter 2 976 oppdateringer**. Antall negative nabokorrelasjoner falt fra **29 til 7**. For 5/6min økte korrelasjonen fra **−0,0413 til 0,6100** på disse treningsradene, men på alle undersøkte Q1- og Q2-foreldre var den **0,1698 og −0,0454**. Her brukes alle lagrede foreldre, mens den foregående fasitsammenligningen bare brukte foreldre med begge mål gyldige.

Dette viser endret samvariasjon i modellens utganger, ikke at den har lært riktig markedsretning. TRAIN-utvalget er lite, og de senere periodene har ulike populasjoner. Resultat: `docs/DIRECTION_HEAD_TRAIN_LEARNING_CHECK_20261028.json`, SHA-256 `f1befc6b5ce33de2019e5193311411646819390e7ce7b8e51a75285a43f901a2`. Ingen nye modellforward, optimizersteg, TEST eller handel.


## GC-kontekst: utenfor treningsområdet

Jeg sammenlignet de **1 205 eksisterende GC-verdiposisjonene** og deres observasjonsmasker i modellens gjenbrukte gjeldende kontekst. På de 47 602 første TRAIN-foreldrene gjenskapte de beregnede normaliserte minimums- og maksimumsverdiene metadata eksakt for alle 1 205 posisjonene. I Q1 hadde **14 895 av 17 288** foreldre minst én observert verdi utenfor sitt respektive TRAIN-område; medianen var **189** posisjoner per forelder. Av 18 145 892 observerte verdi-celler lå 2 372 322 utenfor; 2 295 565 av disse hadde avstand over 0,1 i normalisert rom. I det faste Q2-utvalget gjaldt minst ett avvik **alle 1 760 foreldre**, med median **204** posisjoner, og 346 859 av 1 814 741 observerte celler var utenfor. Dette er ofte beslektede verdier på flere tidsrammer, ikke 189 eller 204 uavhengige indikatorer. En separat råverdimetode gjenskapte antall celler, foreldre og posisjoner eksakt.

Et konkret eksempel er `gc_m5_observed_vwap_usd`: observert maksimum i TRAIN var **2 801,405**, mot median **2 907,042 i Q1** og **3 323,885 i Q2-utvalget**. Verdien oversteg TRAIN-området på 11 720 av 17 284 observerte Q1-foreldre og alle 1 760 observerte Q2-foreldre. Transformasjonen er fortsatt matematisk definert utenfor området; avviket viser ekstrapolering og mulig generaliseringsrisiko, ikke en ødelagt feature. Relative VWAP-avstander finnes også i profilen.

En beskrivende kontroll av dette ene feltet fant 5min-AUC **0,49445** på 5 529 kvalifiserte Q1-rader innenfor TRAIN-maks og **0,50353** på 11 659 over maksimum. Gjennomsnittlig modellert OPP-sannsynlighet var henholdsvis **49,00 % og 48,61 %**, mens faktisk OPP var **51,02 % og 51,86 %**. Gruppene tilhører ulike markedsperioder og analysen er ikke en kausal intervensjon. Tallene viser derfor ikke at nivåskiftet i dette feltet forklarer svak AUC; en kontrollert sammenligning trengs før endret representasjon kan vurderes som forbedring.

`docs/DIRECTION_GC_CONTEXT_RANGE_DIAGNOSIS_20261028.json`, SHA-256 `a32df96dc55669ef6efc4b5b47a28249adf8cd71f4f5b32362e197d090f27962`, samler kilde-/datahasher og de verifiserte aggregatene fra tre capped, lesende terminalberegninger; filen er et transkribert sammendrag, ikke en egen maskinprodusert kjørekvittering. Ingen nye modellforward, optimizersteg, TEST eller handel.


## Registerperiode mot kronologisk modellfold

Det nåværende GC-feltregisteret er laget for **1. mai 2024 til 1. januar 2026**, og det samme gjelder det eldre registeret som ble beholdt i det. Perioden omfatter dermed Q1 og Q2 2025, selv om den undersøkte modellens første TRAIN-prefix slutter ved årsskiftet 2024/2025. Registerets hash for beslutningsklokkene samsvarer eksakt med alle **118 249** foreldrene. Registeret og det beholdte eldre registeret oppgir at utfall ikke ble brukt. Den nåværende registerkoden beholder felter som er observert og bruker identiske kilde-/formelbindinger for aliaser; den mottar ikke et mål i registerfunksjonen.

Jeg målte observasjonsmaskene i den bundne konteksten: **alle 1 205 beholdte GC-verdifelt var observert minst én gang på de første 47 602 TRAIN-foreldrene**. Alle 1 205 registrets observasjonstall over hele den bredere perioden stemte også eksakt med kontekstfila. Dette strider ikke mot funnet av 33 felt som var *normalisert null* gjennom prefixet: en observert verdi kan være konstant, og en observert hendelse kan være falsk. Den faktiske normaliseringen ble tilpasset første prefix, med siste fit-barstart 31. desember 2024 kl. 21:50 UTC.

Den lengre registerperioden betyr at denne modellen ikke har et register som var uavhengig låst ved årsskiftet 2024/2025. Kontrollen fant ikke direkte fasitlekkasje via dette GC-registeret, men den beviser ikke at en full registergjenskaping på bare prefixet ville gitt identisk skjema, eller at alle andre kilder, publiseringstider og historiske revisjoner er uten lekkasje. Resultat: `docs/DIRECTION_GC_REGISTRY_FOLD_SCOPE_20261028.json`, SHA-256 `6eed8d1142c65fdcf5dd8b6819304e7d742c315ee97215b94409be21e3b1c67d`; observasjonstallene er et sammendrag fra capped terminalkjøring, ikke en egen kjørekvittering. Ingen modellforward, optimizersteg, TEST eller handel.


## Endrer GC-konteksten seg ved M5-lukking?

Jeg koblet hvert Q1- og Q2-utvalgsforelder til den eksakte foregående beslutningen fem minutter tidligere når den fantes. Av 17 288 Q1-foreldre hadde **17 224** en slik forgjenger; av 1 760 Q2-foreldre hadde **1 754** det. Alle de sammenlignbare foreldrene hadde minst én endret observert GC-verdi, med median **209** endrede verdiposisjoner i Q1 og **212** i Q2. Mange posisjoner er beslektede og inkluderer klokke-/alders- og kumulative felter, så antall endringer er ikke antall uavhengige handelssignaler.

For mer konkrete M5-data endret observert GC-VWAP seg på **17 214 av 17 216** Q1-par der begge verdiene var observert og **1 754 av 1 754** i Q2. Delvis kjent GC-delta endret seg på **17 127 av 17 216** og **1 742 av 1 754**. Strengt GC-delta var observert på langt færre foreldre. Dette viser at tallene i gjeldende kontekst oppdateres, ikke at endringene er prediktive, at ukjent aggressorside er kjent eller at faktisk publiseringsforsinkelse er kvalifisert.

En separat frossen modelltest er nå **fullført** på de samme tidligere fastsatte Q2-batchene. I den ene armen beholdt modellen nåværende GC-kontekst; i den andre ble 2 410 GC-verdi-/maskeposisjoner i nåværende kontekst og siste lokale sekvensrad byttet med den eksakte foregående M5-beslutningen. De tidligere lokalradene, GC-MTF-historien, andre inputs, vekter og buffere var uendret. Alle originalbatcher gjenskapte lagrede logits byteidentisk; **1 754 av 1 760** foreldre hadde en eksakt foregående beslutning.

På **1 747** kvalifiserte 5min-utfall var AUC **0,49008 med fersk kontekst** og **0,49944 med forrige kontekst**. Parret differanse gammel minus fersk hadde dagbånd **−0,01189 til +0,03165** og ukebånd **−0,00563 til +0,02544**; begge inkluderer null. På 60min var AUC **0,48514 mot 0,49036**, også med dag-/ukebånd som inkluderer null. Siste GC-kontekst endret 5min-retningens faste fortegn på **87 av 1 747** rader, med gjennomsnittlig absolutt endring i OPP-sannsynlighet **0,515 prosentpoeng** og maksimum **4,112 prosentpoeng**. Dette viser påvirkning på modellen, men ingen dokumentert forbedring av retning.

En uavhengig kontroll koblet alle rader til korrigert fasit, gjenskapte begge armenes AUC/logloss/Brier på alle 60 horisonter og de parrede 10 000 dag-/uketrekningene ved 5 og 60 minutter. Største punktavvik var **5,56e−17** og intervallavvik **4,37e−18**. Den opprinnelige RESULT-fila har en feil tekst i `metrics.limitations` som beskriver forrige historieablasjon; selve `intervention`-feltet og outer `limits` beskriver riktig inngrep. Originalkvitteringen er bevart, den uavhengige kontrollen forklarer feilen, og kildekoden er rettet for fremtidige kjøringer.

Dette er en etterundersøkelse på gjenbrukte utviklingsdata; byttet kan gi en unaturlig kombinasjon. Siden GC-MTF-historien fortsatt er fersk, er dette ikke en fullstendig test av all fersk GC-informasjon. Forutsetningen om utbyttbare kalenderblokker er ikke etablert. Ingen trening, TEST, handler eller økonomisk edge. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_STALE_CONTEXT_20261028_001/OUTPUT/RESULT.json`, SHA-256 `78195edb995d245cf741a5368da82af82e088f77a12e4e6fa6fd45146c9b0492`; uavhengig kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_STALE_CONTEXT_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `a0e997e5a0134d11cc0a09fc3cc91115ea729fd04cae233100efeed32d8adcb8`.


## Hvorfor korrigert fasit ikke allerede er i trening

Jeg har prøvd de faktiske hashbundne filene i de to målkoblingene. Eksisterende treningsbinder godtar originalt 1–60-arkiv, men avviser korrigert desimalarkiv med `NATIVE_DIRECTION_DENSE_DATA_ANCHOR_BYTES:mid_move`, fordi den er laget for byteidentiske eldre 1/5/15-ankre. Den separate desimalforbrukeren godtar den korrigerte strukturen og bevarer foreldreakser, klokker, kilderader og masker. På alle 118 249 foreldre er henholdsvis **56, 40 og 18** originale fortegn endret ved 1, 5 og 15 minutter; dette er tidligere falske bevegelser mot eksakt uendret midtpris.

Det gamle treningsscopet og den private mekanikkvalideringen er eksplisitt bundet til originalfilen. Korreksjonen kan derfor ikke tas i bruk ved å bytte fil eller fjerne ankerkravet. En separat rute må binde og autentisere både original og korrigert fasit med de eksisterende kilde- og forbrukerkvitteringene, kalle desimalforbrukeren, og få fersk initialbaseline og egen kvalifisering før en ny, spesifikt autorisert trening. Det er ingen målt retreningseffekt ennå.

Kilde- og filbindinger samt minst nødvendige opplegg: `docs/DIRECTION_DECIMAL_TRAIN_BINDER_BLOCK_20261028.json`, SHA-256 `56a2773843ae6abfe212ec37e9bd669a7e3b57946d827906c34536eb5252f407`. Bindresultatene er et sammendrag av en capped lesende terminalkontroll, ikke en egen maskinprodusert kjørekvittering. Ingen nye modellforward, optimizersteg, TEST eller handel.


## Separat autentisert korrigert målinngang

Den nye dataeieren `gx1/contracts/entry_native_direction_decimal_input_v1.py` binder originalt og korrigert målarkiv til de kildeverifiserte desimal-, materialiserings- og forbrukerkvitteringene, kontrollerer at den lesende kallerens eldre mål er byteidentiske med riktig fil, og bruker deretter den eksisterende separate desimalforbrukeren. Den gamle treningsbinderens ankerkrav er beholdt. Syntetiske tester av feil fil, kvittering, referanse og akse bestod.

En separat capped kontroll på de ekte filene bestod for alle **118 249 foreldre og 60 horisonter**: fysiske foreldre-/barstart-hasher stemte, originale ankre ble kontrollert, og **976** falske bevegelser ble fjernet uten motsatte ikke-null-fortegn. Dette kvalifiserer selve datakoblingen. Den er **ikke koblet inn i et autorisert treningsløp**, og en fersk initialbaseline, separat kilde-/scopekvalifisering og nytt eksakt treningsvedtak kreves. Ingen effekt på modellens retning er målt ved denne kontrollen.

Genuint datakontrollresultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_BOUND_INPUT_ACTUAL_20261028_001.json`, SHA-256 `8415d195256ea7dd9bc9cd35661c1536fe57c1db330b0969d16a963a3381be19`. Ingen modellforward, optimizersteg, TEST eller handel.


## Fersk GC-kontekst og alle fem GC-historikkruter

For å utvide den forrige kun-konteksttesten brukte jeg de samme **1 760** fastsatte Q2-foreldrene; **1 754** hadde en eksakt beslutning fem minutter tidligere. I den ene armen ble all nåværende GC beholdt. I den andre ble 2 410 navngitte GC-kontekstverdier og masker samt siste lokale aliaser erstattet med forrige M5-verdier, og GC-historikkbøtter som lukket etter forrige beslutning ble markert utilgjengelige med verdier og masker samlet. Eldre GC-historikk og øvrige modellinnganger var uendret. Alle **110** originalbatchene gjenskapte lagrede logits byteidentisk.

Inngrepet skjulte **1 754 M5-, 583 M15-, 143 H1- og 36 H4-rader** med ikke-null GC-data. Ingen D1-rad lukket innenfor de undersøkte femminuttersvinduene, så denne testen kan ikke si hva en fersk D1-rad bidrar med. På **1 747** kvalifiserte 5min-utfall var AUC **0,49008 med fersk GC** og **0,49939 med siste GC skjult**. Parret differanse skjult minus fersk hadde dagbånd **−0,01196 til +0,03160** og ukebånd **−0,00583 til +0,02543**; begge inkluderer null. På 60min var AUC **0,48514 mot 0,49035**, også med dag-/ukebånd som inkluderer null.

Sammenlignet med den tidligere armen der bare gjeldende GC-kontekst var foreldet, endret den ekstra skjulingen av GC-historikk **ett av 1 747** faste 5min-fortegn; gjennomsnittlig absolutt sannsynlighetsendring var **0,00243 prosentpoeng**, maksimum **0,01307**. Den uavhengige kontrollen gjenskapte begge armenes AUC/logloss/Brier for alle 60 horisonter, samt 10 000 parrede dag-/uketrekninger ved 5 og 60 minutter; største punkt- og intervallavvik var henholdsvis **5,56e−17 og 3,04e−18**. Den kontrollerte også sammenligningen med den tidligere kun-kontekstarmen på identiske foreldre.

Dette er ett gjenbrukt utviklingsutvalg og et inngrep som kan skape en unaturlig kombinasjon av inputs. Det viser ikke hvordan en modell retrent uten eller med ferske GC-data ville gjort det, og utelukker ikke GC-avledede felt utenfor de navngitte rutene. Faktisk publiseringsforsinkelse, fyll og økonomi er ikke kvalifisert. Ingen dokumentert retningsedge, trening, TEST eller handel. Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_FULL_STALE_20261028_001/OUTPUT/RESULT.json`, SHA-256 `488c856752b461ade6eff21f89ba258ddbab7dadab336fbc06ab8abbc86ca725`; uavhengig kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_GC_FULL_STALE_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `719dd6f6555927312ce43755fc0f4bf21f0d43655cbc3c1145141ff36eb8f5ac`.


## Bps og ekstra treningspass mot samme korrigerte fasit

De tidligere separat trente bps- og trepassmodellene ble sammenlignet mot eksakt samme lagrede énpass-baseline. 17 288 senere foreldre, alle lagrede kontekstfelt, tidsakser og masker stemte. Desimalfasiten fjernet henholdsvis 4, 5 og 3 falske fortegn ved 1, 5 og 15 minutter; alle tidligere originale 3 × 3 AUC-er ble gjenskapt før den nye målingen.

På **17 192** kvalifiserte 5min-utfall var AUC **0,50298** for baseline, **0,49676** for bps og **0,50096** for tre pass. Parret bps minus baseline hadde dagintervall **−0,01755 til +0,00605** og ukeintervall **−0,02127 til +0,00608**. Tre minus ett pass hadde dagintervall **−0,01002 til +0,00604** og ukeintervall **−0,00802 til +0,00430**. Alle AUC-intervallene inkluderer null. Tre pass hadde dessuten dårligere logloss og Brier med ukeintervall over null, mens tilsvarende dagsintervaller inkluderte null. Intervallene omfatter to faktorer, tre mål og to blokktyper, med forhåndsvalgt korreksjon for 12 intervaller.

En uavhengig capped kontroll beregnet alle punktene med sklearn og alle 10 000 parrede intervaller ved en separat sortert poeng-/blokkmetode; et utvalg vektede AUC-trekninger ble også kontrollert direkte med sklearn. Største punktavvik var 1,11e−16 og intervallavvik 3,04e−18.

Dette er den samme gjenbrukte utviklingsfolden, én seed og de opprinnelige 1/5/15-hodene. Modellene ble trent med gammel fasit, så undersøkelsen **måler ikke effekten av å trene på korrigert fasit**. Full treningsinputparitet er heller ikke bevist for bps. Ingen ny modellkjøring, trening, TEST, handelsresultat eller økonomisk edge.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CORRECTED_FACTORS_20261028_001/OUTPUT/RESULT.json`, SHA-256 `48ba0989f59bc6edfc4feb2f5e5a7295d4c9280ee136eacd05a6f33f297e834e`; uavhengig kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CORRECTED_FACTORS_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `38c8f3246fe14182d4c9ed26e7b86fbd1fed64b80bed83fbcec99123efa5ecc3`.


## Fast nabohodegjennomsnitt

For å teste én konkret konsekvens av forskjellen mellom nabohodene ble lagrede logits for 4, 5 og 6 minutter gitt lik vekt og sammenlignet med det opprinnelige 5min-hodet. Foreldre, klokker og masker stemte eksakt med korrigert mål. Alle 60 horisonter ble beregnet beskrivende; 5min var forhåndsvalgt primærhorisont for intervallene.

I Q1 var AUC **0,50017 før og 0,50278 etter** på 17 192 kvalifiserte utfall. Parret endring hadde dagbånd **−0,00377 til +0,00884** og ukebånd **−0,00222 til +0,00735**. I det faste Q2-utvalget var AUC **0,48969 før og 0,48848 etter** på 1 753 utfall, med dagbånd **−0,02376 til +0,02042** og ukebånd **−0,02094 til +0,01942**. Alle båndene inkluderer null, så det er ikke dokumentert bedre retningsrangering.

Q1 hadde lavere logloss og Brier enn originalhodet med både dag- og ukebånd under null. Dette er forbedring i samlet sannsynlighetsscore relativt til den aktuelle originalmodellen, ikke i seg selv et separat bevis på bedre kalibrering eller edge. I Q2 inkluderte de tilsvarende båndene null. Som et etterfølgende beskrivende sjekkpunkt hadde den faste 5min-andelen fra første TRAIN-prefix (**24 069/47 282 = 0,50905**) lavere punktscore enn nabogjennomsnittet i begge perioder: Q1 Brier **0,24980 mot 0,24997** og Q2 **0,25005 mot 0,25021**. Dette siste er ikke en intervalltest og bør ikke presenteres som bevist forskjell.

En uavhengig kontroll rekonstruerte naboverdier og alle punkter med NumPy/sklearn, og gjenskapte samtlige 10 000 parrede dag-/uketrekninger med separat sorteringsmetode; største punktavvik var 2,23e−16 og intervallavvik 3,91e−18. Begge utviklingsperiodene er tidligere undersøkt og Q2 er et ikke-tilfeldig delutvalg. Dette er en etterundersøkt idé, ikke uavhengig bekreftet forbedring. Modellen er ikke retrent, og TRAIN-referansepunktene er gjengitt fra en separat capped terminalkontroll med verifisert målhash, ikke en egen maskinkvittering. Ingen ny modellforward, optimizersteg, TEST, handel eller økonomisk edge.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_NEIGHBOR_POOLING_20261028_001/OUTPUT/RESULT.json`, SHA-256 `db39faf262d0cbc848dc7dfa26b3e52e5530846ac842afaa117e12db20388b27`; uavhengig kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_NEIGHBOR_POOLING_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `eb382469e6f6d55d0c9011ac8373dffc1a16f6203213506b287191120474b9ff`.


## Treffprosent mot alltid OPP

En separat capped kontroll av de samme lagrede 60-hodeprognosene og korrigert fasit gir for **5 minutter i Q1** treffprosent **48,88 %**, mens en konstant OPP-prognose treffer **51,58 %** på de samme **17 192** utfallene. Modellen valgte OPP på **13,33 %** av radene. I det faste Q2-utvalget treffer modellen **49,29 %** mot **50,20 %** for alltid OPP på **1 753** utfall; modellen valgte OPP på **4,16 %**. Dette er punktmålinger på gjenbrukte utviklingsdata, uten eget parret usikkerhetsintervall for treffprosenten.

Til sammenligning valgte 6min-hodet OPP på **99,38 %** av de kvalifiserte Q1-radene og **99,49 %** i Q2; dette gjelder hvilken side av 50 % sannsynligheten ligger på, ikke at modellen er 99 % sikker. Verdiene er gjengitt fra en capped lesende terminalberegning med hash- og aksekontroll; JSON-sammendraget er ingen selvstendig kjørekvittering. Ingen modellforward, trening, TEST eller handel. Sammendrag: `docs/DIRECTION_HEAD_ACCURACY_DIAGNOSIS_20261028.json`, SHA-256 `c35341662b2ec165b6006270776f7bc0275773bafae8dd85d111c179e097959e`.


## Målbevegelse mot observert candle-spread

Den aktive retningstapsfunksjonen gir kvalifiserte opp-/ned-celler lik vekt, uten å skille små og store prisbevegelser. På eksakt samme Q1- og faste Q2-foreldre som modellen er vurdert på, har jeg derfor koblet korrigerte 1–60min-utfall til native bid- og ask-åpninger på både beslutnings- og sluttcandle. Alle foreldre-, klokke- og kildepekere stemte; prisene og fasitens bevegelser ble kontrollert på det opprinnelige tre-desimalers prisgridet.

Tabellen viser andelen **ikke-null** midtprisbevegelser som var større enn halvparten av entry-spread pluss halvparten av slutt-spread, målt fra candleåpningene. Den er beskrivende og bruker senere utfall kun i denne revisjonen.

| Horisont | Q1: større enn spreadproxy | Fast Q2-utvalg: større enn spreadproxy |
| --- | ---: | ---: |
| 1 minutt | 31,35 % | 46,73 % |
| 5 minutter | 58,00 % | 71,53 % |
| 15 minutter | 75,13 % | 82,94 % |
| 60 minutter | 88,24 % | 91,59 % |

Ved 5 minutter var Q1 **9 972 av 17 192** bevegelser større enn proxyen, **95** like og **7 125** mindre. Median absolutt bevegelse var **2,389 basispunkter**, median kombinert spreadproxy **1,857 basispunkter**, og **23,66 %** av ikke-null-bevegelsene var under ett basispunkt. I Q2-utvalget var **1 254 av 1 753** større, **8** like og **491** mindre. Dette synliggjør at binært fortegn alene er et ufullstendig mål for en senere handelsbeslutning; det fastslår ikke hvorfor modellen har AUC omkring 0,50.

En uavhengig metode beregnet direkte den fordelaktige differansen mellom bid/ask-ticks i ønsket retning og gjenskapte alle tre utfallskategoriene og kvantilene for alle **60 horisonter og begge perioder**. Maksimalt kvantilavvik var null.

**Begrensning:** Candleåpningen er ikke en dokumentert samtidig, omsettelig quote eller et fyll. Analysen inkluderer ikke provisjon, slippage, prisveien før sluttidspunktet, åpne posisjoner eller øvrige handelskostnader, og den beviser verken tap, lønnsomhet eller en handelsregel. Utfallsbasert filtrering er ikke tillatt ved beslutningstidspunktet; eventuelle fremtidige FLAT-/retningsmål må testes med kun kausale inputs. Q1 og Q2 er gjenbrukte utviklingsdata, og Q2 er et fast ikke-tilfeldig delutvalg. Ingen modellforward, trening, TEST eller handel.

Resultat: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CANDLE_SPREAD_20261028_001/OUTPUT/RESULT.json`, SHA-256 `ee4ee92e1e872e3af4121202afe8d4cc1510826955d885fd863bef55eaf099cb`; uavhengig kontroll: `/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_CANDLE_SPREAD_INDEPENDENT_CHECK_20261028_001.json`, SHA-256 `dece4fbd7f3f560de5af60f8197ec9bf13123713115c5d88ae6f2ea00caee6cc`.
