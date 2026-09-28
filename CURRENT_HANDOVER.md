# Gjeldende status — 29.09.2026: native-forberedelse og full repo-gjennomgang

**Les først:** [GX1_RULES.md](GX1_RULES.md) (bindende regler), [AGENTS.md](AGENTS.md)
(arbeidsmåte), [GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md) (mål og vedtak) og
[VEIEN_VIDERE.md](VEIEN_VIDERE.md) (eksakt neste steg). `bash scripts/gx1_handover.sh --check`
overstyrer prosa.

## Nytt operatørvedtak: ferdigstill inputs, revider hele repoet, ingen trening

Brukeren har autorisert [native-forberedelse og full repo-gjennomgang](docs/NATIVE_PREPARATION_AND_REPO_REVIEW_20260927.md)
før eventuell trening. Dette overstyrer tidligere forbud mot videre datasetbygging.
Den konsoliderte native modellen med de nye inputene er ikke epoch-trent.
Ridge/HGB-porten under er avsluttet forskning, ikke et bevist tak for native læring.
SMC-rettelsen og prosjektvis låsing er kontrollert. Tidligere kilde-/testgjennomgang
er dokumentert; ny v37-inputbygging og gjenstående avklaringer står nedenfor.
Trening forblir deaktivert. Eksakt scope står i native-forberedelsens `PLAN.json`.

## Nå: lifecycle godtar samme låste priser som kanonisk tape

`CONTINUE_SUMMARY_MEMORY_20260928` lukket TRAIN-skriveren med 652 552 rader,
uten nytt minnedrap. Kjøringen stoppet 28.09 kl. 22:20 UTC / 29.09 kl. 00:20
norsk tid: lifecycle avviste M1-rader med BID lik ASK. Ingen ferdigmanifest
ble publisert; den lukkede TRAIN-parqueten er fortsatt ikke et godkjent datasett.

Kanonisk tape, Entry-utfall og optimal-stopping-eieren tillater allerede BID lik
ASK. Lifecycle-bygger, lifecycle-leser og offline replay bruker nå samme regel.
Kryssede priser, ugyldige tall og OHLC-brudd avvises fortsatt. Ingen priser,
features, mål, kostnader eller splitgrenser endres. 44 fokuserte tester består.
Faktisk TRAIN-vindu fra desember 2012: 2 732 rader og fire lifecycle-episoder
består tape-/bygger-/leserkontroll; alle priser er eksakt bevart. Kontrollen
brukte ikke TEST og måler ingen handelsverdi.

Ny engangsplan:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_LOCKED_QUOTES_20260929`.
Ny outputrot:
`/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928/DATASET_LOCKED_QUOTES_RECOVERY_20260929`.
Nyeste røde kjede og dens preflight er bundet som forelder; ferdige inputs
beholdes i den opprinnelige CHAIN-roten. Gjenbrukseieren kontrollerer eksakte
hashverdier og uendret upstream-kode. Bare de navngitte downstream-funksjonene
kan avvike; delte helpers/imports er AST-identiske. Ny preflight er obligatorisk.

Les nye PLAN/BINDING/WAITING/START/TERMINAL og prosesser før videre handling.
Tidligere forsøk og delvise outputs bevares. Ny POST_BUILD venter på denne
launcherens grønne kjede. Kilden fryses under venting/kjøring. Minnegrenser og
alle vakter er uendret. Automatisk oppfølging hver 30. minutt består.
Trening, optimizer, full VAL og TEST-utfall er fortsatt stengt. Post-rebuild,
lifecycle-bindinger og gjenstående kompleksitetsvurdering gjenstår.

## Historikk: rettet minnevekst i datasettskriving

`CONTINUE_RESOURCE_WAIT_20260928` passerte de tidligere blokkeringene, men ble
OOM-drept i sin 10 GiB cgroup kl. 20:51 UTC / 22:51 norsk tid. Siste loggførte
TRAIN-flush var 496 640 rader. Delvis parquet, Group-A-checkpoints og alle røde
kvitteringer bevares. Ingen split-manifest/ferdigkvittering godkjenner dette datasettet.

Skriveren beholdt én dictionary med 97 identitets-/labelkolonner per rad. Den
beholder nå kolonnebaserte batcher. På 100 000 syntetiske rader med faktisk feltsett
falt beholdt summary-lagring fra 712 800 984 til 77 625 088 bytes (89,1 %).
Verdier, datatyper og rekkefølge er eksakt like. Dette måler representasjonen,
ikke toppminnet i en full rebuild. 22 fokuserte tester består. Parquet-skriving,
features, labels, splitgrenser og minnetak er uendret.

Ny engangsplan:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_SUMMARY_MEMORY_20260928`.
Ny outputrot:
`/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928/DATASET_SUMMARY_RECOVERY_20260928`.
Eksisterende kjedeeier kan nå motta en separat, eksplisitt tidligere inputrot.
Den rehasher ferdige upstream-inputs, krever uendret produsentkode og tillater
bare endret downstream-datasettfunksjon/summary-hjelper. Delte helpers/imports
kontrolleres AST-identiske. Ny preflight kreves, og alle nye downstream-outputs
må være ferske. Gammel preflight er kun kildebevis; rødt blir aldri grønt ved kopiering.

Les PLAN/BINDING/WAITING/START/TERMINAL og prosesser i den nye runtime-roten.
Tidligere plan er konsumert. Ny POST_BUILD bindes til den nye launcheren og den
nye outputroten. Kilden fryses under kjøring. Automatisk oppfølging hver 30. minutt
består; trening, optimizer, full VAL og TEST-utfall er fortsatt stengt.

## Historikk for inputbyggingen nedenfor

## V37: ferdige featureflater; rettet preflight-leser og bundet videreføring

Kjøringen stoppet 28.09 kl. 13:27 UTC / 15:27 norsk tid etter ferdig M1/M5.
Alle 1 382 M1-chunks og featureflatene er ferdige: M1 5 570 522 rader, M5
1 152 859 rader, begge med 242 felt og PASS-manifest. Preflight har 29/29
beståtte kontroller og `READY_FOR_MODEL_NATIVE_SEQ513_REBUILD`.
Selve datasett-/lifecycle-rebuilden var ikke startet.

Den konkrete blokkeringen var kjedens filkontroll: publiseringseieren skriver
både JSON-hendelsen og obligatorisk `.json.order`-kvittering, men leseren krevde
én katalogoppføring. Leseren bruker nå eksisterende immutable-event-eier og
aksepterer bare én hendelse med gyldig kvittering. 14 kjedetester består,
inkludert ekte publisering og avvisning av endrede/manglende bevis, samt faktisk
kontroll av denne kjøringens preflight. Ingen feature-/modell-/dataprodusent er endret.

Første videreføring lå i
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_PREFLIGHT_ORDER_20260928`.
PLAN/BINDING/START/TERMINAL der eier ny kjøring. Eksplisitt preflight- og rød
forelder-hash kreves; kode under gx1 og øvrige scripts må være identisk med
forelderens commit. Alle gjenbrukte filer rehashes, eksakte perioder/stier
kontrolleres og senere outputs må fortsatt være ferske. Dette gjenbruker de
ferdige v37-inputene uten ny M1/M5-bygging. Gamle terminaler bevares.

Tidligere POST_BUILD stoppet korrekt på rød forelder og skal ikke gjenstartes.
Ny post-build-videreføring skal bindes til den nye launcherens PID/kilde og bare
kjøre eksisterende readiness etter grønn kjede. Automatisk oppfølging i samme
Codex-oppgave er opprettet hver 30. minutt (`gx1-v37-oppf-lging`): stabil drift
skal være stille; feil og ferdigstillelse følges opp innen samme autoriserte scope.
Det er ingen treningsautorisasjon. Under kjøring er kilden frosset.

## Ressursventing etter første videreføring — 28.09 kl. 19:48 UTC

`CONTINUE_PREFLIGHT_ORDER_20260928` besto den fysiske gjenbrukskontrollen,
men stoppet før model-source-identity på uendret krav om minst 20 GiB tilgjengelig
RAM. En separat EURUSD-produsent brukte omtrent 10 GiB; ingen av prosjektenes
vakter eller jobber endres. Datasett/lifecycle er fortsatt ikke bygget.

Ny engangsvidereføring ligger i
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928/CONTINUE_RESOURCE_WAIT_20260928`.
Launcheren avventer minnekravet lest fra eksisterende capped-run-eier, kontrollerer
hvert 15. minutt og starter deretter samme preflight-gjenbrukskommando én gang.
WAITING/RESOURCE_CHECKS/START/TERMINAL og prosess avgjør faktisk status.
Egen launcherlås avviser duplikater; kilde-/filbinding revalideres før start.
POST_BUILD bindes til denne launcheren. Også commit-kontrollen ble avvist av
minnevakten. En engangs runtime-kø binder den eksakte dokumentdiffen og venter
før normal commit/push, kilde-/filbinding og dispatch. `QUEUE_STATUS.json`
og `QUEUE_TERMINAL.json` viser dette fortrinnet. Ingen hook omgås.
Kilden og den bundne dokumentdiffen er frosset også under venting.
Tidligere røde terminaler og deres POST_BUILD bevares. Ingen ny tung jobb er
startet før START/prosess bekrefter det. Codex-oppfølging er aktiv hver 30. minutt;
den krever at Mac og Codex-appen kjører. Den eksterne launcheren venter på serveren.

## Opprinnelig v37-inputbygging — bevares som historikk

Forberedt kjøring: `/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_V37_20260928`.
`PLAN.json` avgrenser inputbyggingen; `BINDING.json` binder eksakt ren kilde
og inputhasher etter denne dokumentasjonscommiten. `START.json`, prosessene,
`progress.log` og `TERMINAL.json` avgjør om kjøringen faktisk er startet,
aktiv eller ferdig. Ikke start en kopi hvis planen allerede er konsumert.
Kilden fryses mens byggingen går; eventuell oppfølging skrives i runtime-mappen.

Faktiske registry-/squeeze-eiere har godkjent gjenbruk av frosne konstanter og
seks tidsrammers squeeze-parametre. Kalibrering er uendret 2009-06-01 til
2013-01-01 22:00 UTC, med indre fit-slutt 2012-04-11 22:00 UTC. Gamle feature-
matriser gjenbrukes ikke som v37-inputs. Nytt par, M5/M1, rangering og signal
bygges i `GX1_DATA/data/data/prebuilt/HISTORY2009W_NATIVE_V37_20260928`.
Historikkstart er målt 2010-06-14 22:00 UTC; TRAIN/VAL/TEST-grensene er uendret.
Eksisterende capped-run og prosjektlås gjelder, én tung jobb innen CURRENT.

Dette fullfører forutsetninger for en senere vurdert treningstest. Inputbygging
alene dokumenterer ikke beslutningsverdi eller edge. Post-rebuild/readiness,
lifecycle-bindinger og resterende kompleksitetsvurdering må fortsatt fullføres.
`training_enabled=false`; TEST bygges og forsegles uten utfallsevaluering.

## Kontrollert SMC-rettelse som den nye byggingen bruker

Den målte konflikten er rettet hos den delte SMC-eieren: posisjon og observert
intervallbredde er ett par. Positiv bredde beholder rå, uklippet posisjon; kjent
null bredde gir paret `(0, 0)`. Ukjent oppvarming beholder NaN i begge felt.
Lokalflaten får `smc_pivot_envelope_width_atr`; MTF bruker sitt eksisterende
`mtf_smc_range_width_atr`. Ingen pivotregel eller markedsrad endres.

Eksekverte eiere bekrefter signal v37 / SMC-primitiver v4: 242 signalfelt
(25 basis, 150 obligatoriske, 67 kandidater), fortsatt 71 kontinuerlige
kontekstfelt og 190 felt per høyere tidsramme. Ingen modellhoder er fjernet.
189 fokuserte tester og 92 integrasjonstester består. Faktisk M1-kontroll på
6 019 349 rader bekrefter bit-identiske posisjoner for alle 6 019 242 rader
med positiv bredde, 32 ærlige oppvarmingsrader og endelige koordinater etter
historikkgrensen. Alle sju berørte TRAIN-rader har bit-identisk lokal/MTF-
representasjon i sine eksakte pivotnabolag. Toppminnet var 3,12 GiB under
uendret 4 GiB-tak, uten cgroup-OOM. Dette er inputbevis, ikke læring eller edge.

De ferdige par-/M5-/rangerings-/signalartefaktene er **v36-bevis**, ikke ferdige
v37-inputs. De faktiske gamle signal- og M5-kontraktene avvises av v37-eierne.
Ingen manifest omskrives for å passere. De 1 382 ferdige Group-A-chunkene og
alle kvitteringer bevares. På dette historiske stoppunktet manglet M1-output;
v37-output er siden fullført som beskrevet øverst. Gjeldende videreføring
står øverst. Den gamle planen
skal ikke gjenstartes. Ingen rebuild eller trening ble startet i det tidligere
rettings-/oppryddingsarbeidet.

Bakgrunn: fullkjøringen fra `10c78d70` stoppet 02:59:28 UTC på sju TRAIN-rader
med fire like pivotpriser (2012 og 2019). Arrow-rettelsen var vellykket: RSS
9,52 → 6,85 GiB, alle Group-A-chunks fullført. Det første nye diagnoseforsøket
fikk SIGKILL under full MTF-materialisering; det er bevart som ufullført.
Den etterfølgende, avgrensede MTF-kontrollen besto med terminal exit 0.

## Parallelt prosjektarbeid etter brukerens vedtak 28.09

CURRENT har nå egen prosjektlås. Den gamle maskinfelles låsen og EURUSD-
prosjektet er urørt. Én tung jobb innen CURRENT og alle ressurs-/maskinvare-
vakter består. 294 vakttester og en faktisk kjøring mens den gamle låsen
var holdt bekrefter at separate prosjekter kan arbeide parallelt.

Detaljer, avgrensninger og kvitteringer: [repo-gjennomgang](docs/REPO_REVIEW_20260928.md).

## Fullført forskningsresultat — ikke gjeldende startinstruks

Codex har fullført tidlig kalibrering og én forhåndsbundet historisk sammenligning.
[Endelig resultat: NO-GO](docs/HISTORY2009W_EARLY_DECISION_RESULT_20260927.md).
Kalibrering slutter 2013-01-01 22:00 UTC; ti holdouts dekker juni 2015–juni 2025.
En separat råpriskontroll avdekket og rettet en eldre D1-foldgrensefeil; bare
modellsammenligningen ble beregnet på nytt, med samme artefakter og parametere.
54 tester og den separate kontrollen består. Bruk kun `walkforward_strict/`,
`DECISION_GATE_STRICT.json` og `VERIFICATION.json` under
`/home/andre2/GX1_RUNS/HISTORY2009W_EARLY_DECISION_20260927`.

HGB: +5,71 netto bps per beslutningsblokk (år likt vektet), +1,34 mot LONG,
men nedre grenser -5,12/-22,39 og bare 5/10 årsfordeler mot LONG. Ridge: -2,37.
Begge feilet porten. Ved dette historiske stoppunktet var videre bygging stengt;
brukerens senere vedtak ovenfor åpnet native-forberedelse, fortsatt uten trening.
Ikke gjenstart denne fullførte planen eller den avbrutte senkalibrerte C0.

Avsnittene under er historiske funn. Negativt resultat for målte oppsett beviser
ikke at all retning bare finnes på uker eller at et bestemt marked er ulærbart.

## Konsolidering (operatørvedtak 26.09)

Det fantes to spor: fra 19.09 ble det arbeidet i `/home/andre2/src/GX1_ENGINE` på en foreldet
08.09-base fordi rot-loaderen pekte dit. Nå er dette eneste kodebase; den arkiverte grenen er
tagget `archive/gx1-engine-audit-v9-20260926` og slått inn selektivt. Detaljer, hva som er tatt
og ikke tatt, og hvorfor: [docs/CONSOLIDATION_20260926.md](docs/CONSOLIDATION_20260926.md).
Rot-loaderen importerer nå denne kodebasens `CLAUDE.md`, som importerer `GX1_RULES.md` og
`AGENTS.md` — samme regler for Claude og Codex. Én agent om gangen.

## Hva vi vet

- **Tidligere måling av tidsskala (avgrenset evidens):** Trenden er ~1,6 % av bevegelsen
  per 95-minutters vindu; bekreftede M1-svingninger fortsetter med 49–51 %; retningsmålet hadde
  et hardt tak på 8 timer. På ukeshorisont tjente alltid-LONG etter all kost i 3 av 4 år, og
  ingenting (380 D1/H4-felt, trendregler) slo den — dette beskriver det undersøkte oppsettet og perioden 2021–26. Se
  [docs/DIRECTION_TIMESCALE_20260926.md](docs/DIRECTION_TIMESCALE_20260926.md).
- **Den direkte M1-hypotesen** (25.–26.09) bruker samme etiketter som knee-målet, og vent-målet
  velger side i ettertid (+13,7 bps skjevhet); den kjøres ikke videre.
- **Native lifecycle-v2-modellen:** Entry FLAT256/256 (alle side-Q negative etter kost), Exit
  slår umiddelbar lukking på gjenbrukt TRAIN, samlet læringsport ikke bestått
  ([docs/ENTRY_SELECTOR_CACHE_FIT_20260924.md](docs/ENTRY_SELECTOR_CACHE_FIT_20260924.md),
  [docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md](docs/ENTRY_EXIT_LINKAGE_AND_COST_20260919.md)).
- **Retningsforskningen 23.–24.09** (walk-forward, mønstre, kryss-aktiva, HGB, direkte
  klassifikasjon) er slått inn under `docs/`, med Codex' forbehold 24.09; ingen robust
  retning på 95 min–8 t ble funnet.
- **Ukesretning, forhåndsregistrert (26.09): NO-GO.** 0 av 48 celler slår alltid-LONG i noe år
  2023–26; modellene kollapser til «vær long» eller taper når de avviker. 2021–26 er ett
  oksemarked — lengre historikk med fallende markeder er den avgjørende forutsetningen;
  operatøren vedtok 26.09 å hente fra 2005; native M5 + M1 fra 2006-03-19 er hentet 27.09
  ([docs/HISTORY_2005_INTAKE_20260926.md](docs/HISTORY_2005_INTAKE_20260926.md))
  ([docs/WEEKLY_DIRECTION_RESULT_20260926.md](docs/WEEKLY_DIRECTION_RESULT_20260926.md)).

- **Gjennomgang 27.09:** featureflaten og grunnmuren er bygd for scalping; flere felt er epokeklokker
  på 2011–2025, og med dagens finansiering tjener alltid-LONG ~0 netto 2011–25 (drift 10,43 mot
  finansiering 10,35 bps/uke). Anbefalt: modellfri TSMOM-måling på D1 før seq513-rebuild
  ([docs/FEATURE_SURFACE_SWING_REVIEW_20260927.md](docs/FEATURE_SURFACE_SWING_REVIEW_20260927.md)).

- **Modellfrie grunnlinjer 27.09: NO-GO** (0/62): ingen enkel scalp- eller swingregel gir retningsgevinst
  etter kost på 2011–2025 ([docs/MODEL_FREE_BASELINES_RESULT_20260927.md](docs/MODEL_FREE_BASELINES_RESULT_20260927.md)).

- **Makrohendelser 27.09: NO-GO** (0/18): FOMC-, NFP- og KPI-tidspunkt gir ingen handelbar retning
  etter kost ([docs/MACRO_EVENT_BASELINES_RESULT_20260927.md](docs/MACRO_EVENT_BASELINES_RESULT_20260927.md)).

- **Intradag-mekanismer, bølge 1, 27.09: NO-GO** (0/61; operatørvedtak «kjør bølge 1 nå»): rundtall ($10/$50),
  de 36 oppsettene som speilede par over 2011–25, COMEX-momentum, LBMA-auksjonen og sesjoner/ORB på lokal klokke
  (sommertid). Ingen celle positiv netto ved policyens low-slippage (1 bps per utførelse); brutto ≈ −spread.
  Etterpåanalyse: fortsettelse etter brudd har +1–2,5 bps mid (beste $50-kryss, 1 t: +1,91, t 3,80, 11/13 år), mindre
  enn rundtur-spreaden på 2,3–2,7 bps ([docs/INTRADAY_MECHANISMS_RESULT_20260927.md](docs/INTRADAY_MECHANISMS_RESULT_20260927.md)).
  Walk-forward-eieren leser nå policyens navngitte slippage-scenarier (`load_slippage_scenarios`); primitiveieren
  har kolonnefilter (`build(..., keep_columns=)`). Neste: operatørvalg.

- **Rebuild v36 på 2009-tapene (operatørvedtak 27.09 «Ja»):** squeeze → C0 → par → squeeze(par) → seq513-kjeden,
  launcher `GX1_RUNS/HISTORY2009_REBUILD_20260927/`. Første forsøk stoppet i C0 på stillestående helgekvoter
  (2011); rettet med stengningskontrakt v2 og nytt parvedtak `OANDA_PAIR_PRETEST_2009_WEEKCLOSED_20260927`
  ([docs/HISTORY_2005_INTAKE_20260926.md](docs/HISTORY_2005_INTAKE_20260926.md)). Det daværende neste steget var v2-taper og tidlig historisk kontroll;
  disse er siden fullført. Dette er ikke en gjeldende startinstruks.

## Operativ tilstand

Gjeldende operativ status står øverst og i `NEXT_RUN_POLICY.json` /
`RUNNING_NATIVE_CALIBRATION.json`, med terminalkvitteringer fra samme navngitte run.
Gamle V9-/lifecycle-v2-checkpoints tilhører tidligere skjema og skal bevares som
historikk. De er ikke gyldige v37-modeller. TEST-resultater, optimizersteg,
full epoch/VAL og handel er stengt. Den nye v37-planens runtime-kvitteringer
avgjør nåværende byggestatus; gamle terminaler er historikk.
