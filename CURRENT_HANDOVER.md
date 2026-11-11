# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_MIXING_GRADIENT_ADMITTED.
Målet fra 11.11 er aktivt: isoler målte årsaker til svak retningslæring i
indikatorer, sammensetting og læring. [Årsaksstatus](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og 30–40 epoker er senere skalering, ikke et krav ved hver rettelse.
Alle trenings-/launchporter er false. Ingen jobb lever. Fem diagnostiske
planer er COMPLETE og konsumert. Én gradientobservasjon er innrømmet,
uten optimizersteg eller trening.
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

## Faktisk encoderlæring og neste steg

MARKET_STATE_TA_ENCODER_DELTA_20261111_001 er COMPLETE og konsumert.
PLAN 2bdbd32a / RESULT fe2c2df7. Kilde c729e0dd uendret.
4,95 sekunder på CPU; to eksisterende checkpoints, ingen modell eller optimizer
konstruert og ingen forward/backward/fit/updates. Begge modellhashene og de
opprinnelige checkpoint-seglene er kontrollert.

Alle åtte temporal- og tre akse-encodere har endrede vekter. Alle 228 undersøkte
parametertensorer har ikke-null Adam-andremomenter og stegteller 1849;
initial optimizer var tom. Ingen manglende Adam-state. Dette beviser historisk
gradientflyt til hver tensor og faktisk parameterendring, ikke nyttig læring
eller ikke-null gradient i hvert element ved hvert steg. Relativ L2-endring
per encodergruppe er 4,11–8,31 prosent av initialnormen.

Neste kontroll er innrømmet én gang:
MARKET_STATE_TA_MIXING_GRADIENT_OBSERVATION_20261111_001.
PLAN daee8105; metadata-preflight bd32b8c4. Samme c171-kandidat, eval-modus,
16 tidligere faste intakte B16 TRAIN-grupper, originale sampled-S/native-F-queryer.
17 forwards inkluderer én fersk no-grad-baseline, som må bli bitlik grad-forward.
Original loss-gradient til logits deles i S/M5-side, F/M5-side og resten.
Tre VJP-er pluss direkte totalgradient per graf gir 64 autograd-kall mot bare
36 parametertensorer i family_axis/timeframe_axis/cross_tf. Ingen optimizer,
parameteroppdatering, dropout eller senere featurelesing. Alle øvrige parametre
forblir frosne; modelltilstand og midlertidig trainability kontrolleres ved slutt.

83 rutetester og 11 gradienttester bestod. Etter eksakt JSON-bytebegrensning
bestod 4 berørte tester. Maks 12 GiB, 900 sekunder, 64 MiB beholdte gradienter,
1 MiB samlet CLAIM/RESULT/TERMINAL. Gradientvektorer publiseres ikke.
Sum av gradientbidrag må stemme med direkte totalgradient innen den bundne
FP32-grensen. Manglende derivat og eksakt null rapporteres atskilt.

Presisering: den tidligere 4096-raders tapskontrasten var INTERLEAVED minus
CONTROL, ikke INITIAL mot dagens R1. Den viste bedre samlet loss og dårligere
M5-sideledd, men beviser ikke historisk optimizerårsak. Den nye kontrollen
måler lokal gradientfordeling ved dagens korrigerte hoder. Den simulerer ikke
AdamW og begrunner ingen modellendring eller sletting av indikatorer alene.
