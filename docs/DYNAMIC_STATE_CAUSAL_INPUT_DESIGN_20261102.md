# Dynamisk nåtilstand som modellinput: dokumentert designkandidat

Status 02.11.2026: En avgrenset modell- og tapsvariant er implementert i kilde, men ikke kjørt eller kvalifisert. Den ene autoriserte korte treningen og første scoring er konsumert. Ingen ny modellkjøring, trening, scoring, TEST, Exit eller handel følger av dette dokumentet. Resultater og manglende ordinære terminalkvitteringer er beskrevet i [resultatrapporten](DYNAMIC_MTF_STATE_ONEPASS_RESULT_20261102.md).

## Oppgave og observasjon

Prognosen skal beskrive OPP, NED og FLATT og mulige tilstandsskifter på M5, M15, H1, H4 og D1. Opp- og nedbevegelser kan begge forekomme etter hverandre. Beslutningstidspunktet er et tidspunkt for prognosen, ikke en fast prisstrek. Entry og handel kommer senere. Den vedtatte målprototypen bruker 0,5 ganger kausal Wilder ATR14 per tidsramme som bevegelsesterskel og de siste 5/15/60/240/1440 observerte M1-barene som lokale vinduer for stillstand. Åpningstid og ukjent datatap håndteres forskjellig. Den gjennomførte modellrunden dekker 1–1440 kalenderminutter. En separat kildevariant kan nå bindes til et lengre minuttgitter, men er ikke kjørt eller validert. Et beregningsgitter er ikke en handelsregel eller maksimal holdetid.

I senere utviklingsdata er lagret M5-nåtilstand kjent i 8 987 av 17 288 rader. Hos 3 202 av de 8 987 er den lokale stirekkefølgen likevel uavklart. På 1 028 rader med kjent FLAT og observerbar neste tilstand var 691 fortsatt FLAT etter ett minutt, mens den trente modellen valgte FLAT i null tilfeller. Modellens gjennomsnittlige FLAT-sannsynlighet her var 0,1426347274. Referansen har eksplisitt tilgang til nåtilstanden; det hadde ikke modellen. Disse tallene dokumenterer en informasjonsforskjell og et mulig tiltak, ikke at tiltaket gir edge.

## Nåinpututtak implementert for historiske data

`query_historical_dynamic_mtf_current_state` i `gx1/contracts/entry_dynamic_mtf_state_targets_v1.py` gir et eget uttrekk av bundet parent-ID, utstedelsestid, kjent/ukjent nåtilstand og separate tilstands- og stimasker. Det returnerer ikke fremtidsetiketter, priser eller terskler. Funksjonen er uttrykkelig merket `historical_event_time_only_not_online_delivery`, kontrollerer parentidentitet, M1-modenhet og ATR-klokke, og avviser fremtidige kildetidspunkt. Den ordinære target-queryen gjenbruker samme oppslag og gir i tillegg den separate nå-stimasken; tidligere targetverdier er ikke endret.

Alle 34 syntetiske targettester bestod, inkludert tre nye kontroller for uavklart sti med kjent retning, manglende M1-rad, fremtidige pris-/ATR-endringer, feil parent-/klokkesamsvar og kopibeskyttelse. Et separat CPU-uttak gjenbrukte lagret kildebundet TRAIN- og utviklingsgrunnlag og verifiserte eksakt samsvar på alle 64 890 foreldre og fem tidsrammer mot de lagrede observasjonsmaskene og forrige klokkeaudit. Ingen modell, optimizer eller scoring ble kjørt. Dette er et reelt historisk datauttak, ikke en ONLINE-kvalifisering. Modellkoblingen i den nye kildevarianten er beskrevet nedenfor; faktisk modellforward, fysisk mottaksrekkefølge og restartparitet gjenstår. Den endrede kildefilen krever ny kildebinding og, dersom ONLINE-funksjonen endres, ny initialbaseline før eventuell ny kampanje.

Den vanlige commit-hooken krever også tre bredere modellkontrakttestfiler, og minst én av dem konstruerer modellen og kjører forward. De er uttrykkelig utelatt fordi nye modellkall er stengt; commit-hooken overstyres derfor for denne kildeendringen. Den separate guardrail-testen bestod 10/10. Dette er ikke en full ordinær pre-commit-PASS eller modellkvalifisering.

## Modellkobling og første-tilstandstap i kilde

En egen opt-in forskningsvariant kan nå motta nåtilstand per tidsramme som tre klasseindikatorer pluss separate masker for tilstand og sti. En lærbar projeksjon legger denne informasjonen til den genuine fellesrepresentasjonen før de eksisterende tilstands- og overgangshodene. Batch-koden henter bare disse nåfeltene via det separate historiske uttrekket, sjekker identiteten mot target og nekter kollisjon med øvrige inputs. Fremtidsetiketter holdes utenfor input. Den tidligere modellkontrakten og den brukte numeriske oppskriften beholder eksakt innhold og hash når varianten er avslått.

Varianten har et eksplisitt første-tilstandstap i tillegg til sekvens-NLL: `L = L_sekvens + w × L_første`, der hvert ledd normaliseres over egne observerte, modne celler. Ingen førsteetikett gir intet oppfunnet førsteledd. `w` må være en positiv, endelig flyttallsverdi og bindes i både modell-/view- og recipeidentitet. Testverdien 1,0 er utelukkende syntetisk; ingen vekt er valgt for trening og ingen ny kampanje er åpnet. Markov-forankringen nedenfor er fortsatt en separat designkandidat, ikke implementert i denne varianten.

36 fokuserte kontroller bestod under 4 GiB auditvakt, inkludert maske-/vektdomene, tapsalgebra, kilde-til-batch-ruting, historisk mål-/inputidentitet, gammel oppskrift og 10 guardrail-kontroller. En test fanget og rettet typeforveksling i hurtigbufferen før bestått sluttrunde. Ingen modell ble konstruert eller kjørt; verken gradienter, resume, reelt native minne/tid, bedre læring eller ONLINE-paritet er dermed bevist. Ordinær commit-hook har modelltester og ble uttrykkelig overstyrt under forbudet mot flere modellkall.

En TRAIN-only regnekontroll brukte lagrede sluttprediksjoner fra den tidligere modellen og den uavhengige saved-score-revisjonen. Sekvens-NLL var om lag **0,22665 per kvalifisert celle**, mens første-tilstands log-loss var **0,81028 per kvalifisert førstecelle**. På disse gamle prediksjonene ville `w=0,03` gi et ekstra tapsledd tilsvarende cirka **10,7 %** av sekvenstapet; `w=1,0` ville tilsvare cirka **357,5 %**. Dette er illustrative tapsverdier, ikke målte gradientandeler eller en anbefalt vekt. Den endrede modellen, ny initialisering, utvidet horisont og hver enkelt batch kan ha andre forhold. Ingen vekt er valgt. Lagret kontroll: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/FIRST_LOSS_SCALE_SAVED_TRAIN_001/RESULT.json` (SHA-256 `c503d383c9c6d92829f57941ede5b906c79f44606fbbfb0fcac7b9fc2c3f9d6d`).

## Kontrakt for tilgjengelig informasjon

1. Tilfør per tidsramme separat nåtilstand OPP/NED/FLATT eller UNKNOWN, en tilstandsobservasjonsmaske og en uavhengig maske for om den lokale stien er observerbar. UNKNOWN må ikke kodes som FLAT. En kjent slutt- eller nåtilstand beviser ikke kjent intraminutt-rekkefølge. Ved tvetydig OHLC kan bare tilstander med tilstrekkelig endepunktbevis brukes; uobserverte mellomliggende bevegelser skal ikke finnes på.
2. Bygg nåinformasjonen med egen kildebundet tilgang som bare har data frem til faktisk prognosetidspunkt. Dagens target-query inneholder både nåtilstand og fremtidige etiketter; sistnevnte skal aldri sendes til modellens forward. Fremtidige priser, terskler, ekstrempunkter, masker og etiketter skal ikke kunne påvirke input ved prognosetidspunktet. Behold alle genuine indikatorer, eksisterende tilgjengelighetsmasker, de åtte spesialistfamiliene og D1/makro.
3. Skill mellom kildens bartid/lukketid, faktisk mottaks- og ferdigberegningstid og tidspunktet prognosen utstedes. Historisk XAU M1-kildeskjema har `time` og OHLC/bid/ask/volum, men ingen egen navngitt mottaks- eller behandlingstid. På alle 8 987 senere rader med kjent M5-tilstand var siste M1-lukketid og deklarert ATR-tilgjengelighet lik den registrerte beslutningstiden. Det beviser ikke at verdiene faktisk rakk frem før beslutningen. Historisk event-time-replay skal merkes som dette; ingen fysisk mottakstid eller vilkårlig forsinkelse skal diktes opp. For ONLINE må tilstand og ATR være mottatt og ferdig beregnet før den faktiske utstedelsen, ellers er input utilgjengelig. Andre, ennå uundersøkte logger kan inneholde tidsbevis.
4. Viderefør den dynamiske ekstrem-/retningstilstanden og nødvendig lokalhistorikk over dokumenterte markedspauser og restarter, eller marker den som utilgjengelig inntil den kan gjenopprettes. Ukjente kildehull skal ikke behandles som kjente pauser eller interpoleres. Kalendergrunnlaget er hashbundet i resultatrapporten. Krev eksakt samsvar mellom sammenhengende kjøring og gjenopptakelse før ONLINE-bruk; vurder også utilgjengelig eller endret kalendergrunnlag.

Den avgrensede kildekontrollen fant ingen forekomst av `dynamic_mtf_state` i de 14 Python-filene under `gx1/execution` og `gx1/replay`. Den undersøkte eldre state-adapteren beskriver seg som ikke innrømmet for live-tail; Entry-adapteren sier at den gamle live-launchruten er avviklet. Dette er statisk kildeevidens innen disse katalogene, ikke bevis for at andre systemer eller logger ikke eksisterer. Det finnes derfor ikke dokumentert ONLINE-integrasjon av denne målprototypen i den kontrollerte banen.

## En mulig referanseforankring

For hver tidsramme kan modellen gi lærbare residualer rundt en TRAIN-bundet Markov-referanse. La `T` være overgangsmatrisen fra bare TRAIN, `pi` TRAIN-prioren og `e(s)` en one-hot-vektor for kjent nåtilstand. Sett startfordelingen for første fremtidige minutt til `e(s) @ T` når tilstanden er kjent, ellers `pi @ T`. Kandidaten er

```
initial_logits    = log(startfordeling) + lærbart_startresidual
transition_logits = log(T)              + lærbart_overgangsresidual
```

Softmax normaliserer de tre utfallene, og modellen kan fortsatt bruke alle eksisterende input til residualene. Null residual ved start skal gjenskape den forhåndsdefinerte Markov-referansen, med identiske masker og informasjon. TRAIN-statistikk må bindes til riktig trening/fold før senere kronologisk evaluering. De lagrede aktuelle TRAIN-tellingene er alle positive; andre folder med null i nødvendige telleceller må stoppe lukket til en særskilt begrunnet metode er valgt, ikke få en udokumentert pseudotelling.

En selvstendig NumPy-beregning på de faktiske lagrede TRAIN-tellingene dekket alle fem tidsrammer, tre kjente tilstander pluss UNKNOWN-fallback og 1 440 ledetider. Største absolutte forskjell mellom FP32-konstruksjonen og referansen i float64 var 0,00000895. Dette er bare numerisk designmulighet: faktisk native CPU/GPU- og modellintegrasjon er ikke kontrollert, og etterfølgende trening kan gjøre resultatet både bedre og dårligere enn referansen. En god referanse på langsomme tilstander er ikke i seg selv en handelsfordel.

## Tap og evaluering

Eksisterende sekvens-NLL gir et direkte første-tilstandsledd og betingede senere sekvensledd. I de lagrede sluttprognosene utgjorde førsteleddet 0,21/0,20/0,40/1,10/5,70 % av summert TRAIN-sekvens-NLL for henholdsvis M5/M15/H1/H4/D1, på rader med observert første minutt. Dette er verdibidrag, ikke målte gradientandeler og ikke bevis for årsak.

En kandidat er et eksplisitt, egnet sannsynlighetstap for første tilstand og forhåndsvalgte marginale ledetider sammen med sekvens-/overgangsleddet. Eventuell klasse- eller skiftevekt er en hypotese; eksakte vekter, ledetider og parameterisering er ikke valgt. Ikke optimaliser dem mot den allerede gjenbrukte senere utviklingsperioden. Uansett treningstap skal evalueringen rapportere uvektet log-loss, Brier, kalibrering, dekning, hver klasse og skifteretning, sammen med referanser som har samme tilgjengelige informasjon. Ukjente minutter gir ikke syntetiske skifter, og overlappende prognoser er ikke uavhengige markedshendelser. Stillstand, fortsettelse og begge skifteretninger må vurderes hver for seg.

## Dokumentert måltilgang utover ett døgn

En målgrunnlagskontroll uten modell brukte de lagrede 47 602 TRAIN- og 17 288 senere utviklingsforeldrene, med forhåndsvalgte endepunkter 1, 2, 3, 5, 7, 14 og 30 kalenderdager frem. Alle tre tilstandsklasser forekommer på hvert av de undersøkte endepunktene i begge datadeler og alle fem tidsrammer. Tabellen viser antall observerbare endepunkter som TRAIN / senere utvikling; foreldre og prognoser overlapper og er ikke uavhengige hendelser.

| Tidsramme | 2 døgn | 7 døgn | 30 døgn |
|---|---:|---:|---:|
| M5 | 15 185 / 5 081 | 25 337 / 8 194 | 13 405 / 3 457 |
| M15 | 23 746 / 8 308 | 39 773 / 13 500 | 21 055 / 5 460 |
| H1 | 26 664 / 9 625 | 44 684 / 15 482 | 23 662 / 6 330 |
| H4 | 26 738 / 9 785 | 44 657 / 15 687 | 23 741 / 6 414 |
| D1 | 24 081 / 9 765 | 40 380 / 15 618 | 21 582 / 6 382 |

Dette viser at det finnes et datagrunnlag for å undersøke lengre prognoser. En kjent tilstand på et fjernt endepunkt viser imidlertid ikke rekkefølgen eller alle reverseringene i mellomtiden. Manglende eller lukkede endepunkter forblir ukjente. Eksempelledetidene er verken en valgt prognosegrid eller en maksimal handelstid. Den faktisk trente modellen og dens loss er bundet til 1–1440 minutter. Den separate lengre kildevarianten nedenfor må kvalifiseres særskilt; ingen utvidet trenings- eller prediksjonsmåling finnes.

## Utvidbar modellhorisont i kilde

En separat, eksplisitt research-variant kan bindes til et valgt siste minutt utover 1 440. Den bruker alle mellomliggende heltallsminutter i målquery, modellkontrakt, loss og evaluering; en prognose på sju døgn har dermed 10 080 punkter, ikke bare ett fjernt endepunkt. Ukjente minutter forblir maskert, og evalueringskontrollen kobler ikke skifter over hull. TRAIN-målaudit, prior, modell og recipe må ha identisk grid og objektivhash; den gamle 1 440-minuttersauditen kan ikke kvalifisere en lengre variant. Den opprinnelige targetseriens forsøksgrid beskriver den gamle runden, ikke en maksgrense i selve queryfunksjonen.

85 fokuserte tester bestod under 4 GiB auditvakt, inkludert syntetisk sju-døgns tapsalgebra uten modell, kontroll av skifter rundt et ukjent minutt etter ett døgn, avvisning av feil grid og bevaring av tidligere objekt-/recipeidentiteter. Ingen modellforward, backward, optimizer, ekte scoring eller trening ble kjørt. Metadata- og FP32-koherensgrenser stopper åpenbart ubrukelige konfigurasjoner; de beviser ikke numerisk nøyaktighet på en faktisk lang modellkjøring.

Full lagring av FP32-sannsynligheter for 256 initiale, 47 602 TRAIN- og 17 288 senere foreldre på fem TF og tre klasser ved sju døgn krever minst **39 400 300 800 bytes**. Dette utelater blant annet filheadere, øvrige arrays, checkpoints og arbeidsplass. Det er en beregnet nedre grense, ikke plass- eller kjøretidskvalifisering. Lengre grid, faktisk ledig plass, minnebruk, tidsbudsjett, fersk TRAIN-audit og initialbaseline må avklares før en konkret plan; verken et grid eller en tapsvekt er nå valgt.

Ved en engangskontroll av PC-en var WSLs rapporterte ledige plass 619,8 GB, mens Windows-disken som inneholder Ubuntu-disken rapporterte 286,8 GB. Tallene gjelder bare kontrolltidspunktet og er ikke en reservasjon. En kildevakt for den utvidede varianten verifiserer nå WSL-volumet, den registrerte Windows-disken bak WSL og begge målene for ledig vertsplass. Den krever en eksplisitt, oppskriftsbundet reserve og kontrollerer både før modellstart og før hver lagringsfase. Dette er en kontroll på tidspunktet, ikke en plassreservasjon.

Lagringsvakten beregner eksakte logiske NPY-filstørrelser fra samme layout som produsenten, med filheadere og øvrige resultatarrayer. Reserven må minst dekke den bundne checkpointgrensen og én publiseringsgrense for JSON; andre filer, filsystem-/VHD-overhead og andre prosesser kan kreve mer. Ukjent distro, manglende eller viderelenket WSL-disk, endret vertsvolum eller utilstrekkelig plass stopper kontrollen. Vakten ble kontrollert mot små faktiske NPY-filer og deretter på PC-en under eksisterende lesevakt, uten modellkall eller nektede filåpninger. Det er ikke en gjennomført modell-/tids-/minnekvalifisering eller et løfte om at plassen forblir ledig. Ingen horisont eller reserve er valgt for en virkelig kjøring.

En tidligere privatkvalifisering hadde dessuten en fast formkontroll på 1 440 minutter. Kilden kontrollerer nå outputformen mot antall punkter i det bundne gridet; både gammel og utvidet form er testet med syntetiske metadata, uten modellkonstruksjon. Totalt 94 fokuserte kontroller bestod. Vanlig commit-hook inneholder modellkall og ble uttrykkelig overstyrt; guardrailtestene inngår separat.

## Effektiv TRAIN-telling og lukket auditutkast

En ny tellefunksjon gjenbruker målserien og teller hvor mange TRAIN-prognoser som treffer hvert faktisk observert minutt. Påfølgende kildepar vektes separat, og hull kobles ikke sammen. Dermed trengs ikke en materialisert matrise med alle foreldre ganger alle ledetider. Funksjonen avviser serier merket med TEST-bruk og aggregerer bare mål- og årsaksverdier strengt før TRAIN-grensen. Hele den allerede lagrede pre-TEST-serien blir hash-/strukturkontrollert ved lasting; senere etiketter inngår ikke i tellemetrikken eller utvalget.

På de eksisterende 47 602 TRAIN-foreldrene og den gamle 1 440-minuttersgriden gjenskapte metoden syv lagrede tellefelt og hele den tidligere usmoothede prioren eksakt. Syntetisk sju-døgns kontroll ga også eksakt samsvar med direkte oppslag, inkludert to manglende minutter. En funksjon kan lage et **ikke-admittert utkast** for en eksplisitt bundet fremtidig horisont ved å gjenbruke den opprinnelige uavhengige målregelrevisjonen. Utkastet blir avvist av treningsporten og krever separat kontroll og en kilde-/databundet plan. Ingen virkelig utvidet audit eller horisont er valgt eller kjørt. 88 fokuserte tester bestod uten modell, backward eller optimizer.

## Minimumskrav før eventuell bruk eller ny treningsplan

- Kontroller identitet, modenhet, kildespor og separate masker på alle faktiske parentrader; skill uttrykkelig historisk event-time fra dokumentert fysisk tilgjengelighet.
- Dokumenter at fremtidsendringer ikke endrer nåinput, at modell/baseline får samme observasjonsinformasjon, og at nullresidual gir samme referanse numerisk i faktisk native CPU/GPU-bane. Kontroller nulltellinger, både kjente tilstander og UNKNOWN.
- Bevis ONLINE-hendelsesrekkefølge og samsvar etter restart, inkludert dokumenterte pauser, ukjente hull og tvetydige OHLC-barer. Hvis dette ikke kan bevises, skal verdier og readiness gjenspeile usikkerheten.
- Bind kode, genuine inputs, target, loss, masker, tidsrammer, kilder og recipe sammen. En endret ONLINE-funksjon krever ny initialbaseline og egen mekanisk kvalifisering; gammel trenings-/testkvittering kan ikke kvalifisere den endrede funksjonen.
- Avtal og dokumenter på forhånd hvilke adekvate kronologiske data, referanser, ledetider og mål som brukes. Hold TEST forseglet og evaluer senere økonomi separat. Selve dokumentet åpner ingen kjøring, repetisjon, broker, paper eller handel.

## Revisjonsspor

- Skjemakontroll, bare Parquet-footer/-skjema, ingen datarader eller modell: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/ONLINE_CLOCK_SOURCE_SCHEMA_AUDIT_001/RESULT.json`, SHA-256 `cb662168d553d0e1a1109416bde5c802daa4f50544177505d1efea97150a3779`.
- NumPy-referanseforankring, ingen modell, optimizer, ny scoring eller TEST: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/BASELINE_ANCHOR_FEASIBILITY_001/RESULT.json`, SHA-256 `8fb74c6913f6018c0692274189c6054794e781ad8d2295a3b21a15c3f3b3344e`.
- Avgrenset kildekontroll av execution/replay, ingen kjøring: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/ONLINE_EXECUTION_SOURCE_AUDIT_001/RESULT.json`, SHA-256 `9ec6e0861b54d908e4222d226f3191ce992a813b563512c2723652b4fdb066c4`.
- Lagret historisk inpututtak på 64 890 foreldre: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/HISTORICAL_CURRENT_INPUT_ACCESSOR_001/RESULT.json`, SHA-256 `b4321b1a57ed5e12f8ae01cb01c1e0e99cd76f769ec3365978c9ec6746bf1e1f`.
- Testkvittering og eksplisitt utelatt modell-hook: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/HISTORICAL_CURRENT_INPUT_ACCESSOR_001/TEST_RECEIPT.json`, SHA-256 `b2ae8a296f0b0ec9941cb7e18c4cb4b6713b173bf851cd64e418d8db61ca5e24`.
- Modellkobling, tapsvariant og 36 kilde-/metadata-/tensor-/guardrailkontroller uten modell: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/CURRENT_MODEL_WIRING_SOURCE_001/SOURCE_REVIEW.json`, SHA-256 `0096ed929f2e25f8426991185c4ecec4058e59fa8cf61d6fc1a06c8b4030b1b7`.
- Lagret støtte for målendepunkter utover ett døgn, uten modell: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/SAVED_LONG_HORIZON_TARGET_SUPPORT_001/RESULT.json`, SHA-256 `b44043df9d6be4a35d666b32f68403d16fa5529c420a821cfb2d4cca26ead3f2`.
- Utvidbar tett minuttgrid, syntetisk loss-/evalueringskontroll og lagringsnedre grense: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/EXTENDED_DENSE_HORIZON_SOURCE_001/SOURCE_REVIEW.json`, SHA-256 `34e430bafc8b9006491bc3851886ae1874ab797c0d04e1e2969e682fec8bffab`.
- Engangskontroll av WSL- og Windows-vertslagring, ingen modell: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/WSL_HOST_STORAGE_OBSERVATION_001/RESULT.json`, SHA-256 `8784a98083ebebf241ddd34e04eeb6a6742593b07fa299f6c457648bddcdb2ec`.
- Dual lagringsvakt, syntetiske tester og PC-kontroll under lesevakt: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/EXTENDED_DUAL_STORAGE_GUARD_002/SOURCE_REVIEW.json`, SHA-256 `0067a138830c356aa4ddc60249a06afcc672ae6f737010336ee342d0022d30c3`.
- Lesevaktens kildeeier ved kontrollen: `gx1/scripts/run_entry_native_direction_research_window_v1.py`, SHA-256 `f83572e73594b0f06f459be956c7de86e0f2841f2781308b6dfcb5844ef8a6c5` (identisk med Git `1571c0ec`).
- Effektiv TRAIN-telling, lukket utkast og eksakt gjenbrukskontroll: `/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/TRAIN_COUNT_REUSE_CHECK_003/SOURCE_REVIEW.json`, SHA-256 `2149e88cd7281d36926f41d8ac792b6fb732c49e6f71001e9bdf88b4913cdb3b`.
