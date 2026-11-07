# To framtidige tilstandstidspunkt — avgrenset TRAIN-kandidat 07.11.2026

**Status:** lesende fysisk TRAIN-kontroll fullført. To tidspunkt forkastes som beskrivelse av hele forløpet; ingen modellfit er innrømmet. Gjeldende arbeidsstatus står i `NEXT_RUN_POLICY.json` og `CURRENT_HANDOVER.md`.

## Hvorfor denne kontrollen trengs

Den valgte fasiten beskriver lokal tilstand ved ett endepunkt `t+Δ`, ikke utviklingen mellom `t` og endepunktet. En låst CPU-tabellmodell ga ingen forhåndsbundet M5-sidegevinst på sertifiserte rader. Den lesende M1-kontrollen viste korte **observerte singleton-fragmenter**, men de fleste slutter i AKTIV med ukjent side. Fragmentene beviser derfor ikke korte fysiske trender eller at endepunktmålet er feil. En serie med samme side ved to ender kan ha hatt en vending og retur imellom.

## Minste ekstra prognose

Ved hver M5-utstedelse `t` og for hver `s ∈ {M5,M15,H1,H4,D1}` beholdes eksisterende native ledetid `Δ_s = 5/15/60/240/1440` veggklokkeminutter. Kandidaten spør etter samme lokale fasit `S_s(v)` ved to eksakte UTC-tidspunkt:

- `v_mid = t + ceil(Δ_s/2)` minutter: henholdsvis `+3/+8/+30/+120/+720`.
- `v_end = t + Δ_s`: eksisterende, uendret endepunkt.

Det blir ti marginale tilstandsprognoser, ikke en 352-punkts modell og ikke en felles sannsynlighet for hele banen. Midtpunktet er en forhåndsbundet enkel tidsprobe; det er ingen påstand om en handelsvarighet. En ny prognose utstedt senere kan bruke nyere, da kjente data, men må ikke omskrive en eldre prognose.

Hvert punkt bruker den valgte lokale `W_s = 5/15/60/240/1440` observerte M1-barer og sist lukkede native TF-ATR ved **gyldighetstidspunktet** for fasit. Input til modellen ved `t` bruker bare data kjent ved `t`. Full lokal range under `0,5 × ATR14` gir `{FLATT}`; range fra og med `0,5 × ATR14` beviser AKTIV. Siden blir `{NED}` eller `{OPP}` bare ved eksisterende konservative lokale bevis fra og med `1,0 × ATR14`; ellers `{NED,OPP}`. Ved ufullstendig post-reset-vindu kan observert range fra og med `1,0 × ATR14` likevel bevise `{NED,OPP}` uten side; ellers gir ufullstendig støtte, fraværende kilde eller utilgjengelig ATR ingen supervisjon. Et umodent querypunkt før fold-/TRAIN-cutoff får bare IMMATURE, uten forklaring fra ennå framtidige kilderader. Dokumentert stengning skaper ingen pris eller tilstand. Ukjent side blir aldri FLATT. Ingen terskel velges fra modellskår.

Endepunktsmasken og M1-kilderaden skal være bitidentiske med den valgte serien på alle 118 706 × 5 eksisterende celler før noen midtpunktsstatistikk publiseres. En fold kan bare bruke et mål der det relevante querytidspunktet er modent før foldgrensen, og må bruke egen mask for hvert av de ti punktene. Eventuell loss og skår må for hvert av de ti punktene være på identiske kvalifiserte rader for modell, forhåndsvalgt TRAIN-prior og kausale enkle referanser. Ingen fit innrømmes av dette dokumentet.

## Fysisk kontroll før en fit

En lesende TRAIN-kontroll bruker hele observerte M1-tilstandstapen og publiserer bare aggregerte målinger. Et lite sett med kildeindekser kan bindes for etterfølgende lesende eksempelkontroll uten å publisere råprisbaner. For hver TF og måned rapporteres punktvis `{NED,FLATT,OPP}`, AKTIV med ukjent side, helt UKJENT, eksakt M1-dekning og umodne querypunkt separat fra manglende eksakt M1-rad. Øvrige årsaker til UKJENT er foreløpig samlet og må inspiseres særskilt før eventuell mållås. På intervallet `(t,t+Δ_s]` telles separat:

1. **Bekreftet oversett aktivitet:** begge utvalgte punkt er sertifisert FLATT, mens minst ett mellomliggende observert minutt er sertifisert AKTIV.
2. **Bekreftet oversett motside:** begge utvalgte punkt er samme sertifiserte side, mens minst ett mellomliggende observert minutt har sertifisert motsatt side.
3. **Sensur:** uavklart side, ukjent kildegap, dokumentert stengning eller data-/foldkant; disse gjør ikke «ingen hendelse» til et bevist resultat.

Disse er nedre grenser for oversette hendelser. M1-OHLC fastslår ikke intraminuttrekkefølge. En mellomliggende motside er ikke nødvendigvis en sammenhengende fysisk episode. To punkt kan heller ikke bevise fravær av vendinger mellom dem. Kontrollen skal vise eksempler med prisbane og mask før det avgjøres om to punkt er en nyttig, men begrenset prognose, eller om et rikere forløpsmål trengs. Ingen modellscore, VAL eller TEST brukes til målvalg.

Dagens endepunktmål beholdes som separat referanse. `training_enabled=false`; Entry, Exit, broker, TEST og handel er stengt.

## Målt utfall 07.11.2026

[Engangskontrollen](/home/andre2/GX1_RUNS/MARKET_STATE_TWO_CHECKPOINT_20261107_001/RESULT.json) SHA-256 `1848ade7716273ff416686e250d9b3b4e3e6aae3fbcaad5666a50e991d69af35` og [terminalen](/home/andre2/GX1_RUNS/MARKET_STATE_TWO_CHECKPOINT_20261107_001/TERMINAL.json) SHA-256 `3f402c471e7a9f45034f95d45228436101546a54d1c81048643ff840eb2fea00` bekrefter exit 0 etter 300,94 s. Uavhengig kvitteringskontroll avstemte 12/12 kildehash, 7/7 inputhash, måneds- og maskepartisjoner samt **0 bit-/kilderadavvik på alle 118 706 × 5 eksisterende endepunkt**. Det ble ikke kjørt modellfit, publisert ny fasit eller åpnet TEST.

Nevnerne under omfatter bare utstedelser der de to punktene er like og den aktuelle hendelsen kan observeres gjennom hele intervallet på M1-oppløsning. Tallene er overlappende, beskrivende TRAIN-intervaller, ikke uavhengige sannsynlighetsestimater.

| TF | AKTIV mellom to FLATT-punkt | Bekreftet motside mellom to like sidepunkt |
| --- | ---: | ---: |
| M5 | 2 237 / 3 882 (57,6 %) | 2 816 / 8 612 (32,7 %) |
| M15 | 3 132 / 6 167 (50,8 %) | 4 707 / 10 046 (46,9 %) |
| H1 | 2 867 / 5 165 (55,5 %) | 5 095 / 9 326 (54,6 %) |
| H4 | 724 / 1 041 (69,5 %) | 4 271 / 8 185 (52,2 %) |
| D1 | Ingen komplett 1 440-minutters M1-nevner | Ingen komplett 1 440-minutters M1-nevner |

D1s null komplette klokkeintervaller skyldes blant annet dokumenterte stengninger og er **ikke** null oversette bevegelser. Blant sensurerte D1-intervaller finnes minst 172 observerte tilfeller med aktivitet mellom to FLATT-punkt og 3 841 med bekreftet motside mellom like sidepunkt, men ingen sammenlignbar rate. M5-midtpunktet er AKTIV med ukjent side på 69 363/118 706 (58,4 %) utstedelser; et ekstra tidspunkt reparerer derfor ikke sideetikettenes observerbarhet.

Beslutning: de to punktene kan fortsatt være to gyldige punktprognoser, men de **skal ikke trenes eller rapporteres som en prognose for hele forløpet**. Det gamle endepunktet beholdes som referanse. Neste designarbeid er én delt, tidsstyrt marginal tilstandsprognose for `S_s(v)` på forhåndsvalgte framtidstidspunkt. Den eksisterende fem-TF-modellen har allerede ett felles dekoderhode; en tidskoordinat kan legges til i en ny, kildebundet modellkontrakt. Nåværende målbinding er låst til fem faste endepunkter og M1-tapen mellom dem er bare diagnostikk. Ny per-query-målbinding, maskert tap/skår, kausale referanser og eksplisitt treningskobling må derfor kvalifiseres før fit; ingen 352 uavhengige hoder innføres. Marginale prognoser alene gir ikke sannsynlighet for en felles vending/episode. Den fullstendige M1-tapen mellom de opprinnelige endepunktene er en ny lokal diagnostisk utvidelse med eksakt endepunktsparitet, ikke en tidligere publisert fasit på hvert minutt. Intraminuttrekkefølge og økonomisk verdi er ubevist.
