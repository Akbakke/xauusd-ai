# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_TEMPORAL_BOUNDARY_ADMITTED.
Mål11.11 er aktivt: isoler målte årsaker til svak retningslæring i indikatorer,
sammensetting og læring. [Årsaksstatus](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte runder er feilsøking, ikke kapasitetsdom. Full kvalifisert TRAIN-historikk
og30–40epoker er senere skalering, ikke et krav ved hver rettelse.
Alle trenings-/launchporter false; training_enabled_scope=null. Ingen jobb lever.
Tre tidligere diagnostiske planer er COMPLETE/konsumert. Én ny passiv
observasjon er innrømmet; trening er fortsatt stengt.
GC er på pause; TEST, Entry/Exit, broker, handel og promotion er stengt.
Alle genuine indikatorer og M5/M15/H1/H4/D1-input beholdes.

## Gjort og målt

Bevar kvalifisert kandidatc171736d. Tidligere avgrenset readout-/sannsynlighets-
rettelse er fullført; robust retningsverdi og programmet som helhet er ikke ferdig.
[Opprinnelig resultatevidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).
INITIAL i den matchede studien er et importert warm-start, ikke en fersk
constructor. Aktuell MTF-varians økte under runden; generell kollaps er ubevist.

MARKET_STATE_TA_MTF_RETAINED_INTERVENTION_20261111_001 er COMPLETE/konsumert:
PLAN85d140d3/RESULT3c682a01/TERMINAL27ab3bce. Kilded6d256d6 uendret.
Controller38,39sekunder, exit0.1modell/1checkpoint/4låsteheadimporter,
64joint/64S/64F,128query/128native-head/128M5-head,0encoder/fit/updates.
To FP64→FP32 TRAIN-gjennomsnitt ble låst før senerefangsten ble åpnet.
Baselinens joint-hidden, M5-representasjoner og alle native logits er bitlike;
modell/RNG/buffere/modi/parameteridentitet er uendret.67fokuserte tester består.

Gjennomsnittlig absolutt senereM5-retningssannsynlighetsendring over256rader
ved fast MTF-bidrag: S0,01786/F0,01313prosentpoeng. Dette er lav følsomhet
for akkurat denne intervensjonen, ikke unik informasjonsverdi.
M5Brier intervensjon minus original: TRAIN S−1,0385e−5/F+3,9634e−6;
senere S+1,0215e−6/F−1,5313e−6. Senere AUCendring S0/F−0,0007465.
Senere kjente mål:80S/104F. Grupper viser blandede fortegn; ingen modellendring
begrunnet. D1/F bedre Brier men svakere AUC er kun sekundær observasjon.
INDEPENDENT_NUMERIC_REVIEWd1bdf87d:1360countfelt/1968flyttallsfelt,
maksavvik4,44e−16, inkludert alle20pooledceller og alle16grupper i hver.
512rader i16intakte B16-grupper perpartisjon er ikke512uavhengige observasjoner.
Senere2024 er gjenbrukt utvikling, ikke urørt generalisering eller økonomi.

## Inputavklaring og neste steg

Beholdt rå census:0numerisk manglende MTF-verdier; konstante felt0/1/1/3/2
for M5/M15/H1/H4/D1. D1session_change_flag er0 på425konsumerte rader;
andre konstanter er presence/observed-flagg på1. Dette er ikke frakoblede familier.
TRAIN-normalisering har konstante felt0/1/3/3/2 på en annen populasjon.
Alle øvrige felt beholder transformspenn minst0,5604M5/0,8813andreTF.
Syv runtime-normaliseringsflater har eksakt0avvik i tidligere inputkontroll.
Dette viser dekning og ikke-konstant støtte, ikke typisk temporal spredning.
Makropublikasjon/vintage og prediktiv merverdi er ikke bevist av numerisk dekning.

Neste avgrensede arbeid er passiv måling ved temporal readout og siste TF-
pooling på de samme frosne radene. Skill variasjon innen vinduer fra variasjon
mellom beslutningsrader; krev eksakt outputparitet og gjenbruk inputkvitteringene.
Ingen ny ablasjon, predictorfit eller arkitekturendring nå. Målingen må
kjøres én gang gjennom den nye kildebundne engangsplanen.

Aktuell avgrensning: alle8familier×5TF, samme512rader; tidsvinduer16/64/96/96/252.
Bare parent-encoder-hooks, faktisk batchscopefalse beholdes; annenmodus feiler.
Kompakte mean/last- og RMS-målinger samt etterfølgende attentiongrenser,
maks64MiB. Inputtokens inkluderer posisjon/kontekstgater; varians er ikke
prediktiv verdi.33 forwards / 32 instrumenterte / 0 fit er innrømmet én gang.
PLAN 22fbc407, metadata-preflight 0953e70e. 81 fokuserte tester bestod;
etter siste avgrensede justering bestod 14 temporal-tester.
ANALYZE_BOUNDARIES f5d33178 er bundet før kjøring.
