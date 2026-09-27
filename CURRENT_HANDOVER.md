# Gjeldende status — 27.09.2026: native-forberedelse og full repo-gjennomgang

**Les først:** [GX1_RULES.md](GX1_RULES.md) (bindende regler), [AGENTS.md](AGENTS.md)
(arbeidsmåte), [GX1_ARBEIDSMAAL.md](GX1_ARBEIDSMAAL.md) (mål og vedtak) og
[VEIEN_VIDERE.md](VEIEN_VIDERE.md) (eksakt neste steg). `bash scripts/gx1_handover.sh --check`
overstyrer prosa.

## Nytt operatørvedtak: ferdigstill inputs, revider hele repoet, ingen trening

Brukeren har autorisert [native-forberedelse og full repo-gjennomgang](docs/NATIVE_PREPARATION_AND_REPO_REVIEW_20260927.md)
før eventuell trening. Dette overstyrer tidligere forbud mot videre datasetbygging.
Den konsoliderte native modellen med de nye inputene er ikke epoch-trent.
Ridge/HGB-porten under er avsluttet forskning, ikke et bevist tak for native læring.
Nå: ferdigstill den eksisterende byggekjeden med tidlig kalibrering; deretter
full kartlegging, nødvendige feilrettinger og dokumentert opprydding.
Trening forblir deaktivert. Eksakt scope står i native-forberedelsens `PLAN.json`.

Par, tidlig squeeze, M5-flate og M5-modellkilde er ferdige. Kildebeskrivelsen er
rettet og verifisert. Siste kontroll fant for tidlig oppvarmingsstart: den settes
til målt fullt definert kontekst 2010-06-14 22:00 UTC. Det gir de påkrevde 252
lukkede D1-barene før uendret TRAIN-start 2011-06-01. Markedsidentitet og komplett
MTF-preflight består; se forberedelsesrapporten for bevis. Ferdige outputs
gjenbrukes med eksplisitte manifest-hasher; runtime-terminaler avgjør nåstatus:
`/home/andre2/GX1_RUNS/HISTORY2009W_NATIVE_PREPARATION_20260927/CONTINUE_NATIVE_QUOTE_GEOMETRY_20260927`.
En serialiseringsfeil i arkitekturkontrollen er rettet (JSON-nøkkelrekkefølge);
32 tester og den faktiske gjenbruksvakten består.
M1-geometrifeilen er lokalisert til noen låste BID/ASK-quotes 2012-12-12.
Leserne følger nå samme ASK >= BID-kontrakt som native-kilden; råpriser og kost
er uendret. 189 tester og kontroll av faktisk M1/fill-flate består.
Feature-/modellslanking inngår i repo-revisjonen. 241 signal-felt omfatter også
71 kontinuerlige kontekstfelt; antall inputs alene er ikke bevis for overtilpasning.

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
Begge feiler porten; ingen native trening eller større datasettbygging.
Neste forsøk må begrunnes med endret informasjon eller utførelsesøkonomi.
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

- **Retningen ligger på uker–måneder, ikke på 95 minutter.** Trenden er ~1,6 % av bevegelsen
  per 95-minutters vindu; bekreftede M1-svingninger fortsetter med 49–51 %; retningsmålet hadde
  et hardt tak på 8 timer. På ukeshorisont tjente alltid-LONG etter all kost i 3 av 4 år, og
  ingenting (380 D1/H4-felt, trendregler) slo den — i 2021–26 er det lærbare driften. Se
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
  ([docs/HISTORY_2005_INTAKE_20260926.md](docs/HISTORY_2005_INTAKE_20260926.md)). Neste: hent v2-tapene, bygg
  kilder og lineage, kjør rebuilden på nytt; deretter forhåndsregistrert tak-måling (ridge/HGB, walk-forward
  2015–25 med tidlig kalibrering) før noen native trening.

## Tilstand

Aktivt steg må bekreftes i `GX1_RUNS/HISTORY2009W_EARLY_DECISION_20260927/progress.log`
og samme runs terminalkvitteringer. `training_enabled=false`; full epoch, full VAL, CONTROL, TEST, live og
papirhandel er stengt. TEST er forseglet. Kildekoden har nå den reparerte featureflaten v36
(signal 241, per-TF 190); datasettet må bygges på nytt. Eksisterende V9-/lifecycle-v2-artefakter
og sjekkpunkter hører til v34-flaten og evalueres bare på commit 7c9421a5 eller eldre.
Walk-forward-instrumentet støtter ukeshorisont (beslutningsklokke, ikke-overlappende statistikk,
kostpolicy, tidlig kalibrerte inputs). CURRENT har eget Python-miljø (`.venv`, identisk pakkesett,
avhengighetssjekken består), og `gx1_handover.sh` skriver kildeidentiteten som tunge ruter krever
(`source_identity_gate`).
