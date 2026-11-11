# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_MATCHED_FUSION_DIAGNOSIS_COMPLETE.
Mål11.11 er aktivt: isoler målte årsaker til svak retningslæring i indikatorer,
sammensetting og læring. [Årsaksstatus](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte kjøringer er bevisst feilsøking, ikke en kapasitetsdom. Full kvalifisert
TRAIN-historikk og30–40epoker er senere skalering, ikke et krav ved hver rettelse.
Alle trenings-/launchporter false; training_enabled_scope=null. Ingen jobb lever.
Begge passive planer er COMPLETE/konsumert. Ingen ny kjøring er innrømmet.
GC er på pause; TEST, Entry/Exit, broker, handel og promotion er stengt.
Genuine M5/M15/H1/H4/D1-input beholdes.

## Gjort og målt

Tidligere M5 TRAIN-rettelse er fullført; det større edge-målet er aktivt.
Faste readouts og TRAIN-låst sannsynlighetsdemping bedret Brier/logtap svakt.
Senere AUC S=.523043378/F=.510747753 er ikke treffprosenter. Kandidatc171736d.
[Opprinnelig evidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).
Senere oktober–desember2024 er gjenbrukt utvikling, ikke urørt generalisering.

Passiv trent observasjon: MARKET_STATE_TA_FUSION_OBSERVATION_20261111_001,
PLANa6fb8f98/RESULT071c2976,129,1sekunder. INITIAL-sammenligning:
MARKET_STATE_TA_FUSION_INITIAL_OBSERVATION_20261111_001,PLAN178013d7,
MATCHED_NUMERIC_REVIEW80cab848,127,3sekunder. Begge worker/controller exit0.
Hver brukte512rader/33forwards/1checkpoint/0fit/backward/optimizer/updates.
Originale utdata, modelltilstand, RNG, modi og parametre var uendret.
Foreldre/tidsstempel/query/masker samsvarer eksakt mellom modelltilstandene.
16intakte B16-grupper perpartisjon er ikke512uavhengige observasjoner.
60fokuserte tester, metadata-preflight og faktisk CUDA-baselineparitet består.
Ingen modell-/treningskode er endret; observasjon og kvalifisering er lagt til.

Alle observerte ruter har positiv vekt. D1/session_change_flag har nøytral
konstant gate1.0. Rå joint-varians er83/83prosent kontekst i INITIAL og89/89
etter runden; dette er ikke beslutningspåvirkning etter joint-LayerNorm.
INITIAL er et importert warm-start, ikke fersk constructor: målte residualer
er ikke null. Tidligere forventning om null var feil; opprinnelige råmålinger
er bevart og separat MATCHED_INTERPRETATION_REVIEW korrigerer tolkningen.

Volatilitetsfamiliens lokale gjennomsnittsvekt faller fra97,3/97,5prosent til
5,9/9,3prosent. MTF-variansen øker8,14/19,90ganger; varians utenfor første
akse øker1,02/1,32ganger. Første aksens andel øker47,7/61,3 til93,4/97,4
prosent. Dette er særlig en voksende dominerende akse, ikke påvist generell
utsletting av variasjon. Sluttrepresentasjonens effektive variansdimensjoner
øker litt. Disse målene er ikke matriserang eller prediktiv merverdi.

## Ikke bevist og neste steg

Ingen unik årsak, tapt prediktiv informasjon eller arkitekturendring er begrunnet.
Familievis merverdi, tilstrekkelig trening, urørt generalisering og økonomi er åpne.
Bevar kandidaten. Neste avgrensede kontroll bruker eksisterende fangster:
frys lokal representasjon/kontekst, erstatt bare særskilt MTF-residual og
MTF-pool med faste TRAIN-panelgjennomsnitt, og gjenberegn uendrede decoderlag.
Krev først bitlik baseline; sammenlign eksisterende S/F-retningsloss og parede
B16-grupper.0nyebackbonekall/0predictorfit. Intervensjonen påvirker også
normaliseringen og må tolkes smalt. Ruten må kildekvalifiseres og bindes i en
ny engangsplan før kjøring; ingen slik ny plan er innrømmet nå.
