# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
Les GX1_RULES.md og kjør bash scripts/gx1_handover.sh --check.
NEXT_RUN_POLICY.json er eneste arbeidsstatus; terminal/prosess overstyrer prosa.

## Aktivt mål og grenser

TA_DIRECTION_EDGE_CAUSE_MATCHED_INITIAL_OBSERVATION_ADMITTED.
Mål11.11: isoler målte årsaker til svak retningslæring i indikatorer,
sammensetting og læringsforløp. [Årsaksstatus](docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json).
Korte kjøringer er bevisst feilsøking, ikke en kapasitetsdom. Full kvalifisert
TRAIN-historikk og30–40epoker er senere skalering, ikke et krav ved hver rettelse.
Alle trenings-/launchporter false; training_enabled_scope=null. Ingen jobb lever.
Ingen konsumert plan relanseres. GC er på pause; TEST, Entry/Exit, broker,
handel og promotion er stengt. Genuine M5/M15/H1/H4/D1-input beholdes.

## Gjort og målt

Den tidligere M5 TRAIN-rettelsen er fullført; det større edge-målet er aktivt.
Faste readouts på frosne representasjoner og TRAIN-låst sannsynlighetsdemping
bedret Brier/logtap svakt i fire matchede TRAIN/senere-celler. Senere AUC er
S=.523043378/F=.510747753, ikke treffprosenter. Kandidat c171736d2de79396d.
[Detaljer og opprinnelig evidens](docs/TA_M5_LOCKED_SHRINK_NATIVE_REVIEW.json).
Senere oktober–desember2024 er gjenbrukt utvikling, ikke urørt generalisering.

Passiv observasjon MARKET_STATE_TA_FUSION_OBSERVATION_20261111_001 er
COMPLETE/konsumert. PLANa6fb8f98, RESULT071c2976, NUMERIC_REVIEWdacc9eaa.
Kildea414436c uendret; worker/controller/guard exit0.129,1sekunder.
512deklarerte rader,33forwards,1checkpoint,0fit/backward/optimizer/updates.
Målingen beholdt eksakt samme outputs, modelltilstand, RNG, modi og parametre.
Alle observerte ruter har positiv vekt. D1/session_change_flag-gaten er
konstant1.0/nøytral; dette er ikke en frakobling.45fokuserte tester består.

Rå variasjon mellom rader i joint-input domineres89prosent av kontekstblokken.
Dette er ikke89prosent av beslutningspåvirkningen: joint-LayerNorm normaliserer
innen hver rad. MTF-poolens første akse bærer93,4/97,4prosent av variansen
på TRAIN/senere; konsentrasjonen finnes før residualprojeksjonen.
Alle tre residualprojeksjoner starter med null vekter/bias; små lærte bidrag
kan ikke alene bevise en arkitekturfeil. Uavhengig kilde-/regneregelkontroll
bekrefter tolkningen og eksakt residualrekonstruksjon.

## Ikke bevist og neste steg

Vi har ikke bevist tapt prediktiv informasjon, familievis merverdi, en unik
historisk optimizerårsak, endelig kapasitet, urørt generalisering eller profitt.
Bevar kandidat og originalkvitteringer. Kvalifiser en passiv sammenligning av
bevart INITIAL mot trent modell på identiske foreldre/batcher. Sammenlign
MTF-pool før nullprojeksjonen, alle fire joint-blokker og hidden. Ingen ny fit.
Én eksakt INITIAL-observasjon er innrømmet: PLAN178013d7 i
MARKET_STATE_TA_FUSION_INITIAL_OBSERVATION_20261111_001.60fokuserte tester
og metadata-preflight består.512rader/33forwards/0headimport/0updates.
16intakte B16-grupper perpartisjon er ikke512uavhengige observasjoner.
Kilde og policy fryses under kjøringen.
