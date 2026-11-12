# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_COMBINED_TF_DECODER_SOURCE_QUALIFIED. Edge-målet er fortsatt uoppnådd.
Den eneste nye kandidaten kombinerer eksisterende separate TF-aktivitetshoder med eksisterende
separate non-M5-sidehoder. Design og bevis: docs/TA_COMBINED_TF_DECODER_CANDIDATE.json.
Root /home/andre2/GX1_RUNS/MARKET_STATE_TA_COMBINED_DECODER_SOURCE_20261112_001.
SOURCE_QUALIFICATION.json og METADATA_QUALIFICATION.json binder eksakte filer og resultater.
Ingen ny native-/data-/checkpointjobb eller trening er admittert; ingen tung jobb kjører.

Nåværende delte non-M5-sidehoder var et bevisst valg for det isolerte aktivitetseksperimentet.
Tidligere side-only bedret sideleddet, men aktivitet ble verre. Activity-only reduserte
aktivitetsforverringen, men løste ikke retningen. Kombinasjonen er ikke målt på markedsdata.
Ingen utilsiktet bortkobling eller garantert gevinst er etablert.

Minste kodeendring: tillat begge eksisterende flagg under TA+dedicatedM5 i modell/feature-view,
og krev lik aktivitetsmodus ved den eksisterende sidemigreringen.16nye tensorer/1032parametre.
Forward-struktur, tapsaritmetikk, alle genuine features/8familier/5TF/D1-kontekst og normalisering
beholdes. Encoder/query er fortsatt delt; ikke påstå at all interferens er fjernet.
Alle gamle standard-, side-only- og activity-only-varianter beholdes.

337unike fokuserte syntetiske/metadata-kontroller kvalifisert: første suite328PASS/8FAIL;
feilene var foreldede faste kontrakthasher fra før span-inputprofilen. Pre-change11aa70d6-kode
reproduserer nåværende gamle kontrakter nøyaktig. Kun referansene ble oppdatert;8feilende og
1ekstra berørt kontroll ble kjørt på nytt,9PASS. Original feil-logg er bevart.
Kilde var uendret under kjøringene. Uavhengig kilde-/testgjennomgang fant ingen blokkeringer.
Dette er mekanikk, ikke faktisk startparitet, læring eller edge.

Eksakte eksisterende recipe/view/normaliserings-/konstruktørmetadata er gjenbrukt.
Gammelt view og gammel numerical recipe bygges identisk. Nytt source-view endrer kun bindingen
til oppdatert modellkilde og view-hash. Target-view legger bare til sideflagget og modellidentitet.
Input-/normaliserings-/konstruktøridentiteter er like; target-recipe endrer bare kontraktidentiteter.
SOURCE_FEATURE_VIEW.json, TARGET_FEATURE_VIEW.json og NUMERICAL_RECIPE.json er immutable.
Target S-kontrakt d70e8561, dual8ef0d2e6, loss0d5f9f50, numerical22babd9b.
0native arrays/checkpoint-payloads/modeller/forwards/updates i denne bølgen.

Neste: kvalifiser minste eksisterende benchmarktilpasning for én combined-kontroll mot bevart
COVERAGE. Gjenbruk source-checkpoint b1c5fba6 og uendret span-import til activity-only INITIAL
bef35daf, RNG67daad3d. Klon deretter bare16sidetensorer med eksisterende migrering.
Behold12288unike TRAIN-rader,768updates og faste4096TRAIN/LATER-paneler. Frisk faktisk INITIAL
og utgangs-/tapsparitet må bestå før første update; pass3 er primærmålingen. Alle5TF/S/F beholdes.
Kvalifiser faktisk checkpointkontrakt, startup, rapportering og eksisterende bevoktet kampanjerute
før egen nøyaktig engangsadmisjon. Ingen gammel plan gjenåpnes og ingen automatisk trening.

## Siste målte retningsevidens

Eksponeringskontrast og retained review er COMPLETE/CONSUMED:
/home/andre2/GX1_RUNS/MARKET_STATE_TA_COVERAGE_BENCHMARK_20261112_001.
RESULTd5ba8fa9, TERMINALe43be2ee, EXPOSURE_COMPARISON3e25678a;768updates/0skip.
COVERAGE bruker12288unike TRAIN-rader mot4096×3, med samme oppdateringsbudsjett og faste paneler.
Innhold/repetisjon/rekkefølge endres sammen; ikke isoler unik-radtall som eneste årsak.
Senere M5F AUC er0,528606 motFULL0,513672 ogINITIAL0,511290. Forbedringen motFULL er usikker;
motINITIAL er begge utforskende dag-/ukeintervaller positive. AUC er ikke winrate.
Alle5F-horisonter har bedre senere Brier/logloss ennFULL i begge punktvise95intervaller.
M15/H1/H4F er fortsatt dårligere enn frossen prior. Ingen robust/economisk edge er etablert.

Effektiv labelvekting er COMPLETE/CONSUMED, RESULT89c0a1f5 i
/home/andre2/GX1_RUNS/MARKET_STATE_TA_EFFECTIVE_SIDE_WEIGHT_20261112_001.
M5 gir lik kjent-radvekt; non-M5 har opptil16× forskjell via batchgjennomsnitt.
På COVERAGE er F/M15 pooledUP50,871%→effektiv52,169%; H154,334%→54,372%.
H4/D1 trekkes ned, altså ingen universell UP-drift fra vektingen. Ingen tapsrettelse er begrunnet.
Senere H1Pup var57,459% vedINITIAL og62,434% ettercoverage, mot50%kjent-side-UP-fasit.
M15 går57,820%→61,586% mot49,106%. Skjevheten fantes før og ble forsterket.
Tapsreferansen for konstante logits er ikke AdamW-banen eller en prognose for senere data.

Retained2×2-faktorisering ble kildekontrollert og avvist: TRAIN-captures lagrer ingen
representasjoner; LATER lagrer bare M5. Ingen non-M5-representasjon ble antatt eller invertert.
Tidligere decoder-/temporalstudier og komplementforsøk gjenbrukes; originalkvitteringer bevares.
Detaljert evidens og grenser finnes i docs/TA_DIRECTION_EDGE_CAUSE_REVIEW.json.

Global training/native/full epoch/fullVAL er false. TEST er forseglet; Entry/Exit, broker,
handel, spending og promotion er stengt. GC er på pause. Korte kontroller dømmer ikke
endelig kapasitet; full kvalifisert historikk og30–40epoker er senere skalering.
