# Dynamisk nåtilstand som modellinput: dokumentert designkandidat

Status 02.11.2026: Forslag for kildearbeid, ikke en implementert eller kvalifisert modell. Den ene autoriserte korte treningen og første scoring er konsumert. Ingen ny modellkjøring, trening, scoring, TEST, Exit eller handel følger av dette dokumentet. Resultater og manglende ordinære terminalkvitteringer er beskrevet i [resultatrapporten](DYNAMIC_MTF_STATE_ONEPASS_RESULT_20261102.md).

## Oppgave og observasjon

Prognosen skal beskrive OPP, NED og FLATT og mulige tilstandsskifter på M5, M15, H1, H4 og D1. Opp- og nedbevegelser kan begge forekomme etter hverandre. Beslutningstidspunktet er et tidspunkt for prognosen, ikke en fast prisstrek. Entry og handel kommer senere. Den vedtatte målprototypen bruker 0,5 ganger kausal Wilder ATR14 per tidsramme som bevegelsesterskel og de siste 5/15/60/240/1440 observerte M1-barene som lokale vinduer for stillstand. Åpningstid og ukjent datatap håndteres forskjellig. Det historiske prognosegitteret dekker 1–1440 kalenderminutter: lengre swingprognoser er verken implementert eller validert, og gitteret er ikke en handelsregel eller maksimal holdetid.

I senere utviklingsdata er lagret M5-nåtilstand kjent i 8 987 av 17 288 rader. Hos 3 202 av de 8 987 er den lokale stirekkefølgen likevel uavklart. På 1 028 rader med kjent FLAT og observerbar neste tilstand var 691 fortsatt FLAT etter ett minutt, mens den trente modellen valgte FLAT i null tilfeller. Modellens gjennomsnittlige FLAT-sannsynlighet her var 0,1426347274. Referansen har eksplisitt tilgang til nåtilstanden; det hadde ikke modellen. Disse tallene dokumenterer en informasjonsforskjell og et mulig tiltak, ikke at tiltaket gir edge.

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
