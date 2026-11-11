# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_ENCODER_DELTA_ADMITTED.
Målet fra 11.11 er aktivt: isoler målte årsaker til svak retningslæring i
indikatorer, sammensetting og læring. [Årsaksstatus](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og 30–40 epoker er senere skalering, ikke et krav ved hver rettelse.
Alle trenings-/launchporter er false. Ingen jobb lever. Fire diagnostiske
planer er COMPLETE og konsumert. Én ny CPU-kontroll av to eksisterende
checkpoint er innrømmet, uten modellkonstruksjon eller trening.
GC er på pause; TEST, Entry/Exit, broker, handel og promotion er stengt.
Alle genuine indikatorer og M5/M15/H1/H4/D1-input beholdes.

## Gjort og målt

Bevar kvalifisert kandidat c171736d. Tidligere avgrenset readout-/sannsynlighets-
rettelse er fullført; robust retningsverdi og programmet som helhet er ikke ferdig.
[Opprinnelig resultatevidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).
INITIAL i den matchede studien er et importert warm-start, ikke en fersk
constructor. Aktuell MTF-varians økte under runden; generell kollaps er ubevist.

Den tidligere faste MTF-intervensjonen hadde lav M5-følsomhet: gjennomsnittlig
absolutt senere retningssannsynlighetsendring var S 0,01786 og F 0,01313
prosentpoeng. Brier-endringer hadde blandede fortegn; AUC ble ikke bedre.
Eksakt originalparitet og uavhengig regnskap består. Ingen modellendring følger.
Beholdt inputcensus og normaliseringskontroll viser numerisk dekning og
ikke-konstant støtte. Dette beviser ikke typisk spredning eller prediktiv verdi.

MARKET_STATE_TA_TEMPORAL_BOUNDARY_OBSERVATION_20261111_001 er COMPLETE:
PLAN 22fbc407 / RESULT b89942e5 / TERMINAL ba677e42.
Kilde d3dc09ee uendret. Controller 131,36 sekunder, exit 0.
Én modell og ett checkpoint; fire låste head-importer. 33 forwards, hvorav
32 instrumenterte; 1 280 temporal-familiekall og 32 kall per aksemodul.
512 rader, null backward/fit/optimizersteg, 65 922 130 publiserte NPZ-bytes.
Alle tidligere native- og fusionfelt er bitlike på begge panelene.
Modell/RNG/buffere/modi/parameteridentitet/hooks er uendret.
81 fokuserte tester bestod; etter siste avgrensede justering bestod 14 temporal-tester.
INDEPENDENT_BOUNDARY_REVIEW e5aafc79; INTERPRETATION_REVIEW 2092517a.

Sentrert RMS mellom beslutningsrader, TRAIN / senere:
- Temporal mean+last: 0,413 / 0,441.
- family_axis, som blander TF innen hver familie: 0,157 / 0,171.
- timeframe_axis, som blander familier innen hvert TF: 0,062 / 0,075.
- cross_tf input → output: 0,094 → 0,032 / 0,138 → 0,050.
- Endelig MTF-pool: 0,032 / 0,056.

Mean+last kansellerer ikke systematisk: positiv kovarians i 40/40 TRAIN- og
38/40 senere celler. Alle 40 temporal-ruter har ikke-null vindusvariasjon.
Parameterfri slutt-LayerNorm forklarer RMS rundt 1 og deler av første skalafall;
fallet mellom de to akse-outputene skjer ved omtrent samme RMS-skala.
Endelig pooling er ikke en generell ekstra variansinnsnevring. Negativ kovarians
mellom vektede TF-bidrag kan følge av normaliserte gatevekter rundt felles vektorer.
Varians, konsentrasjon og teknisk paritet er ikke prediktiv informasjon eller edge.
16 intakte B16-grupper per partisjon er avhengige; senere 2024 er gjenbrukt utvikling.

## Gjenstående og neste steg

Eksakt R1-kilde inkluderer alle åtte temporal- og tre akse-encodere i aktiv
parameterpartisjon og AdamW. Den gamle rapporten har bare samlet gradientnorm,
ikke gradient eller parameterdelta per encoder. Før en ny læringsdiagnose
brukes de to eksisterende warm-start/R1-checkpointene til en liten avgrenset
kontroll av per-modul endring og lagrede Adam-andremomenter fra tom initial
optimizer. Ingen modellkonstruksjon, featurelesing, forward, backward eller fit.
Bundet engangsplan: MARKET_STATE_TA_ENCODER_DELTA_20261111_001,
PLAN 2bdbd32a. Maks to CPU-dekodinger, 4 GiB, 300 sekunder.
Manglende Adam-state rapporteres som funn. Ingen gjentakelse av rute-/fusionmålingene.
Faktisk parameterendring vil heller ikke alene bevise nyttig retningslæring.
