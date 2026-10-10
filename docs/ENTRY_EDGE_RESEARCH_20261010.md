# Entry-edge: avsluttet firestegsforsøk 10.10.2026

De fire avgrensede stegene er fullført. Ingen Entry-edge er dokumentert for denne hypotesen. Den native modellen, tapsvektene, de 254 inputfeltene, åtte familier og opprinnelige checkpoints er uendret. Ny native trening er fortsatt stengt.

## 1. Lærings- og gradientdiagnose

Tre forhåndsvalgte TRAIN-batcher ble målt i både opprinnelig og ferdig smoke-tilstand: seks CPU FP32-forwards og null optimizersteg. Entry-retningskontrasten hadde ikke-null gradient til 538 delte parametertensorer i alle seks målingene. Den samlede hjelpegradienten var svakere enn Entry-gradienten og hadde positiv cosinus mot kontrasten (0.0041–0.1606).

På de tre batchene falt fellesnivåfeilen, mens kontrastfeilen økte litt. Resultatet støtter den eksisterende forklaringen om forbedret nivå/bias uten dokumentert bedre retning. Det gir ingen målt begrunnelse for å koble om gradientene eller endre tapsvekter. Dette er små pre-clipping-målinger med CPU dropout, ikke rekonstruksjon av AdamW eller et kausalt inngrep.

Diagnosen fullførte på 332.64 s med 7.15 GiB peakRSS. Første start ble avvist før claim/forward fordi eksplisitt tom CUDA_VISIBLE_DEVICES manglet; avvisningen og separat korrigert plan er bevart.

## 2. Én kronologisk informasjonshypotese

Hypotesen var at kjent prisreaksjon får mer retningsverdi når den kombineres med plassering ved nivåer og markedsregime.

- Additiv probe: 14 deterministiske felt fra eksisterende pris-, nivå-, sweep-, trend-, ATR- og spreadinformasjon.
- Samspillsprobe:de samme 14 feltene pluss 24 forhåndsdefinerte produkter.
- Samme faste ridge-regularisering:alpha lik antall fit-rader; foldens egne sentrerings-/skaleringstall. Ingen parameter- eller horisontsøk.
- 12 ekspanderende kronologiske TRAIN-folds fra 2014 til mai 2025; fit-utfall måtte være ferdige før neste periode. 48 CPU-fits og 529545 evaluerte rader.
- Register- og squeeze-kalibrering var tilgjengelig innen 2013-01-01T22:00Z. Alle 15 rå inputfelt var endelige på alle 652552 fysiske TRAIN-rader.
- 95-minuttersmålet og kostnadsautoriteten var identiske med det observerte native Entry-målet. Negative utfall ble bevart.

| Modell | Retningskontrast MSE (bps²) | Kontrastkorrelasjon | LONG / SHORT / FLAT |
|---|---:|---:|---:|
| Konstant fra samme historiske fit | 2792.340196 | 0.004517 | 0 / 0 / 529545 |
| Additiv | 2792.151997 | 0.011632 | 0 / 0 / 529545 |
| Samspill | 2792.130096 | 0.012381 | 0 / 0 / 529545 |

Samspillets samlede MSE-forbedring mot konstanten er 0.210100 bps², omtrent 0.007524%. Det er bedre i 6 av 12 perioder. Målt med lik vekt per måned er forbedringen 0.743883 bps²; det familiejusterte 95%-intervallet er [-0.634232,2.371906]. Samspillets ekstra forbedring over den additive proben er 0.103505 bps² per måned, intervall [-0.240387,0.464429]. Usikkerheten inkluderer null.

Intervallene gjenbruker eksisterende stasjonær bootstrap med 1999 parvise trekk, forventet blokklengde tre måneder og Bonferroni-justering over sju deklarerte sammenligninger. De er betingede utviklingsestimater.

## 3. Minste begrunnede oppfølging

Denne faste representasjons-/regulariseringshypotesen avvises for native oppfølging. Begge modeller er svake retningsprober og produserer ingen positiv forventet netto Entry-verdi. Selv den høyeste estimerte verdien er negativ: −2.882342 bps for additiv og −2.039162 bps for samspill.

Ingen modell-, tapsvekt- eller treningsendring er begrunnet av disse målingene. Forsøkets autorisasjon er brukt opp. Vi har ikke senket terskelen, forlenget treningen, endret horisonten eller søkt videre etter et gunstig resultat.

Dette avviser ikke alle mulige prisrepresentasjoner, andre regulariseringer eller alle Entry-hypoteser. En ny undersøkelse må begrunnes med en særskilt kausal informasjonshypotese og et eget avgrenset design.

## 4. Økonomi og kapasitet

Eksisterende kontantregnskap ble brukt på 3967064 M1-markeringer fra 2014-01-02T00:50Z til 2025-05-30T20:59Z. Modellen kan ha én fysisk enhet LONG, SHORT eller FLAT; ingen overlappende posisjoner eller pyramider. Alle endringer fylles på observerte M1 BID/ASK ved beslutningen. Posisjonen videreføres mellom beslutninger;95 minutter brukes bare som målhorisont og er ingen tvungen exit.

Kapital og enhetsstørrelse er identiske med alltid-LONG. Kommisjon, slippage, kalenderbasert finansiering og likvidasjonsreserve for åpne sluttposisjoner inngår. Finansiering går til siste observerte pris, ikke til en senere helgegrense.

| Policy | Posisjonsåpninger | Netto bps av startkapital | Maks. drawdown |
|---|---:|---:|---:|
| Additiv | 0 | 0 | 0 |
| Samspill | 0 | 0 | 0 |
| Fit-konstant / FLAT | 0 | 0 | 0 |
| Alltid-LONG, samme faste enhet | 1 | 10781.441294 | 38.7555% |

Null handler og null P&L er ingen dokumentert selektivitet eller edge. Alltid-LONG er en historisk referanse for denne perioden, ikke en anbefaling. Den åpne LONG-posisjonen er medregnet med sluttreserve 10.059471 bps. Finansieringskostnad 6491.257284 bps viser at kostnadene påvirker sammenligningen vesentlig.

Kostnadene er den bundne current-terms-situasjonen, ikke dokumentert historisk brokerkostnad. Dette er en kontinuerlig Entry-argmax-forskningspolicy med fast enhet, ikke native Exit eller produksjonens risikostyring. Notional kan endre seg med gullprisen.

## Kontroll og avgrensning

Seks målrettede tester består, inkludert tidsavgrensning, gradienter uten mutasjon, positiv unik argmax og prising av åpen sluttposisjon. En separat capped audit har kontrollert alle 529545 målrader mot opprinnelig gross BID/ASK-mål og kostnadsformelen, alle 12 tidsavgrensninger, alle ledger-klokker og bevaring av originalcheckpoint. Maksimalt avvik ved fp32-avrunding er 0.000015258789 bps. En separat analytisk alltid-LONG-beregning avviker fra ledgeren med 0.000000106 bps.

TEST er forseglet. Ingen CONTROL-utfall eller native optimizersteg ble brukt.95-minuttersmålet var valgt på hele TRAIN til mai 2025; disse foldene er derfor gjenbrukt, betinget utviklingsevidens og aldri urørt OOS. Den tidligere negative native smoke-evidensen står uendret. Feilnivåer fra proben skal ikke sammenlignes direkte med native smoke fordi evalueringspopulasjonene er ulike.

## Bundne bevis

Alle kjøringer ligger under /home/andre2/GX1_RUNS/ENTRY_EDGE_TRAIN_20261010_001.

- DIAGNOSTIC_002/RESULT.json: SHA-256 85cdcf48c24d1d800c120a5d20e422b83fd6eafef63defed169d89b0225d8eae
- PROBE_001/RESULT.json: SHA-256 3b3600abfa52a5cb34c3fe1eb692279c4f0bb480a3d3647f3b1f4c8bd8e04b91
- FINAL_AUDIT_001/RESULT.json: SHA-256 e8ffad6379c05f7fe175c217601df4b031904800efc44907207b3496da291215
- COMPLETION_REVIEW_001.json: SHA-256 f16e7dcd6e4e90aeb3686d9e2a4e77cceb1a703d0c35578c333e83ed0488983c
- Kjøringskilde:diag6884dd1e;probe og sluttkontroll c606d882. Forhåndsregistreringer ligger i configs/research/ENTRY_EDGE_TRAIN_DIAGNOSTIC_20261010_002.json og ENTRY_EDGE_TRAIN_PROBE_20261010.json.
