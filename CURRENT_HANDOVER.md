# CURRENT_HANDOVER — GX1

Arbeidssted: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`.
Les `GX1_RULES.md` helt. `NEXT_RUN_POLICY.json` er eneste arbeidsstatus.
Originale prosess-/terminalkvitteringer overstyrer prosa. Historikk finnes i Git.

## Gjeldende konklusjon 21.11.2026

Kvalifiseringsmålet `native_future_state_qualification_20261121_001` har bestått
sluttaudit. Target → ekte native output → loss → TRAIN-prior → score følger
samme nye kontrakt. Siste dokumentasjon/publikasjon bokføres i sluttkvitteringen.
Ingen modelltrening, ekstraepoke, serving, Entry/Exit eller handel er innrømmet.
Alle engangsporter er konsumert/stengt; ingen tung jobb er aktiv.

Operatøren valgte **high–low i neste helt framtidige TF-bar**, med alle
M5/M15/H1/H4/D1 beholdt. M5 er beslutningsklokke, ikke maksimal holdetid.
M30 er fortsatt egen senere utvidelse ved siden av M15. Alle8genuine familier,
61makrofelt og eksisterende kvalifiserte native inputs er bevart. Ingen M1.

Bevisrot: `/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_QUALIFICATION_20261121_001`.
Eksakte SHA og autoriteter står i policy; originaler/hjelpere/source/logg beholdes.

## Ny målkontrakt og observerbarhet

Velg første observerte TF-bar med nominell start>=issue. Tomme bøtter fylles
aldri. Pris-/range-/tilstandsfasit krever nominell slutt<foldgrense. Skala er
0,5×WilderATR14 fra siste lukkede TF ved issue, aldri målbarens framtidige ATR.

- Komplett range under skalaen er FLAT; positiv range ved/over skalaen er AKTIV.
- Eksakt komplett nullrange er FLAT også ved null/manglende ATR, uten epsilon.
  Ikke-null range uten kjent ATR forblir ukjent.
- Positiv observert delrange ved/over kjent skala beviser bare AKTIV. Liten
  ufullstendig range beviser ikke FLAT; delvis dekning sertifiserer aldri side.
- Ved komplett forventet vindu sertifiseres side bare hvis nøyaktig én vesentlig
  endpointbevegelse C−L eller H−C finnes ved samme skala. Begge/ingen er ukjent.
  Nullskala krever strengt positiv bevegelse. Ingen ekstremarekkefølge gjettes.

Nominell kompletthet beholdes som egen kontrollfakta. Forventet markedsklokke
unntar bare allerede manifestbundet native ukestenging. Andre daglige/
helligdags-/kildehull gjettes ikke bort. Endpoint ved komplett vindu er siste
forventede åpne M5-medlems close; ved ufullstendig vindu er metadata bare siste
observerte close, potensielt før nominell TF-slutt. Ingen pris fylles.
OANDAs complete=true betyr barens sluttid passert, ikke komplett kildedekning.

Den konservative sidefasiten er en eksplisitt endpoint-/range-kandidat, ikke
bevist universelt latent sluttregime. Rå open/close-fakta og gammel0,3ATR
NEUTRAL_NET er uendret og er egne referanser, aldri ommerket til FLAT.

## Genuin target-/prior-/scorekvalifisering

`STATE_AUDIT_RESULT001.json` / `STATE_AUDIT_TERMINAL001.json`: COMPLETE/exit0,
13,2498s under4GiB/512MiB/1tråd. Eksakt354550TRAIN-issueakse2021–2025 matcher
hovedfitten (`c0805f39...997ea`).1215513bevarte native kilderader dekodes fra
kvalifisert preTEST-arkiv; ingen CONTROL-issues scores, ingen TEST åpnes.
Én målprojeksjon/én ny TRAIN-priorfit,0normaliseringsfits/modeller/updates.

| TF | Kjent aktivitet, prognoseceller | Kjent FLAT | Kjent side | Unike komplette statebarer | Unike barer med sidebevis |
|---|---:|---:|---:|---:|---:|
| M5 | 354549 | 35847 | 228951 | 354549 | 228951 |
| M15 | 354482 | 51477 | 209906 | 118158 | 71520 |
| H1 | 354172 | 60391 | 193100 | 29504 | 16582 |
| H4 | 331078 | 30406 | 187026 | 6652 | 4140 |
| D1 | 338036 | 1379 | 44566 | 234 | 162 |

Gjentatte prognoseceller er ikke uavhengige utfall. D1s162unike sidebarer over
fem år er begrenset støtte. Ukentlig pausehåndtering gir234komplette D1-vinduer
mot0ved streng ubrutt24t-klokke; øvrige hull forblir ukjente. Positiv delrange
forklarer273726D1-aktivitetsceller uten full dekning, ikke nye side-/FLAT-labels.
Dette gir ikke ubetinget kalibrering eller nytte på alle perioder.

`FUTURE_STATE_TARGETS001.npz` er33,85MB med immutable metadata/arrayhasher;
`FUTURE_STATE_TRAIN_PRIOR001.json` og `FUTURE_STATE_PRIOR_SCORE001.json` er
lagret med atomisk no-replace og fil-/katalog-fsync. Ingen regenerering trengs.

## Modell, tap og reell inputkobling

`NativeFutureStateReadout` har separate ACTIVE- og UP|ACTIVE-parametre per TF:
DOWN=a(1−s), FLAT=1−a, UP=as. Tapet er kjent aktivitets-BCE pluss sertifisert
side-BCE, delt på kjente aktivitetsceller. Ukjent side gir null direkte
sidehodegradient med bevart aktivitetsgradient. Helt ukjent støtte gir skip.
Priorer/skårer bruker samme masker; null støtte er UNFITTED. Full treklasse-
skår gis bare ved kjent FLAT eller sertifisert side.

`NativeFutureStateModel` kobler samme native backbone til dette readoutet;
ingen gamle råbarhoder opprettes i den opt-in-ruten. Default råbarmodell er
bevart: AST etter fjerning av bare opt-in-tillegg matcher originalfactory,
seedet state/RNG-paritet består. Original B2-modellkilde er arkivert.

`MODEL_FORWARD_RESULT001.json` / `MODEL_FORWARD_TERMINAL001.json`:
COMPLETE/exit0,21,5068s under12GiB/512MiB/1tråd/64tasks/affinity0–7.
Én fersk CPU-modell,3kildefaste TRAIN-paneler/48rader,3forwards/3backbonepass.
Alle hidden er endelige float32[16,5,128]. Outputs er eksakt frossen ny prior;
tap/skår stemmer. Helmodell-state, norm/prior/inputbinding og RNG er uendret.
0checkpointloads/autograd/backward/optimizer/statistikkfits/updates.
27featurearrays og samme helTRAIN-normalisering gjenbrukes. Kvalifiserte
utviklingsbytes dekodes ved cacheload; ingen CONTROL-forward/skår eller TEST.

Null hodevekter gir prioroutputs uansett hidden. Dette beviser kobling/
identiteter, ikke feature-/familienytte, lært samarbeid eller kalibrering.
48rader er ikke en ny helkohort-initialbaseline eller treningskvalifisering.

## Tester, rettelser og neste steg

136fokuserte target/objective/readout-tilfeller består. Senere84utvalgte
modell-/hjelpertilfeller ble kvalifisert etter én konkret restorefeil:
77bestod først/7feilet;8berørte kontroller bestod etter minste rettelse.
Ikke-persistente hjelpebuffere skal ikke kreves i serialized state. Alle
persistente identiteter/inventar kontrolleres fortsatt før parameterkopiering.
Testomfangene overlapper; dette er ikke220uavhengige tester.

Preflight har nå JSON-only-lesing og eksakt autentisert kildeinventar før
hashing, slik at data/vekter ikke åpnes av en falsk sourcereferanse. H4/D1-fase,
ATR/nullgrenser, pauser/hull, sideukjent og checkpointidentitet er testet.
Audit001s feilaktige500000-kildegrense ble tidligere konsumert; ny002 bandt
produsentens1215513. Alle feillogger/kvitteringer beholdes, ingen relaunch.

**Før en senere fit:** fersk helkohort-initialbaseline for denne funksjonen,
avgrenset faktisk læring utover prior, målt aktivitets-/side-/TF-gradientbalanse,
kald optimizer/RNG/scheduler/cursor-replay og målt ressurs-/gjennomstrømningsprofil.
Frys så ett eksakt budsjett/epoke-/evalueringsopplegg. Ingen gammel godkjenning
eller konsumert plan åpner ny fit. Nyttige samarbeid/edge er fortsatt ubevist.

Den tidligere0,3ATR-hovedepoken (5540updates/~2t1min) og fysisk bitidentisk
recovery er bevart som egne tidligere bevis; nye modellen bruker ikke disse
lærte vektene. Tidligere gradientnorm30,75 avgjør ikke den nye oppgavens vekter.
Fullprosjektaudit og beståtte uendrede kontroller gjenbrukes. Alle antatte670GB
er ikke innholdsgodkjent; tidligere retention slettet35,7GB. Ingen ny sletting.
DATA/RUNS følger retention-eieren. Ingen live/paper/broker/ordre/spending/promotion.
CPUboost0 og eksisterende cgroups/hardwarevakter er uendret; ingen ny CPU-watt/
termisk måling eller absolutt optimalisering er påstått. Teknisk PASS er ikke profitt.
