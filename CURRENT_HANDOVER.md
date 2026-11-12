# CURRENT_HANDOVER — GX1

Arbeidssted: /home/andre2/src/GX1_CURRENT, gren work/gx1-current.
GX1_RULES.md er bindende. NEXT_RUN_POLICY.json eier status; prosess/terminal overstyrer prosa.

## Aktivt mål og neste steg

TA_COMBINED_TF_DECODER_BENCHMARK_SOURCE_QUALIFIED. Edge-målet er fortsatt uoppnådd.
Aktiv kilde: én combined-decoder-sammenligning mot fullført COVERAGE via den eksisterende
benchmarken. Root /home/andre2/GX1_RUNS/MARKET_STATE_TA_COMBINED_DECODER_BENCHMARK_20261112_001.
SOURCE_QUALIFICATION.json og METADATA_QUALIFICATION_001.json binder kode, tester og faktisk JSON.
Design: docs/TA_COMBINED_TF_DECODER_CANDIDATE.json. Ingen ny modell-/data-/checkpointjobb eller
trening er admittert. Alle source-test-/metadataporter er konsumert; native porter er false.

Kandidaten kombinerer eksisterende separate TF-aktivitetshoder og non-M5-sidehoder. Aktivitet-only
var et bevisst tidligere forsøk; ingen utilsiktet bortkobling eller garantert gevinst er etablert.
Tidligere source-/migreringsbevis (337unike kontroller) gjenbrukes fra
MARKET_STATE_TA_COMBINED_DECODER_SOURCE_20261112_001. Alle genuine features/8familier/5TF/D1,
normalisering, delt encoder/query og tapsaritmetikk beholdes.16nye tensorer/1032parametre.

Ny worker bygger activity-only-kilden fra eksakt source-view, gjenbruker råcheckpoint b1c5fba6 og
uendret span-import til INITIAL bef35daf. Deretter klones16sidetensorer inn i combined-target;
ny targetdigest måles og kan ikke kalles bef35daf. OriginalRNG67daad3d gjenopprettes etter begge
konstruksjoner. Én frisk optimizer. Nye INITIAL TRAIN/LATER må være like kontrollen før updates:
aksene/maskene/aktivitet/M5 er bitlike; bare non-M5-side har rtol1e-6/atol2e-6, tap atol5e-6.
TRAIN har ingen lagrede representasjoner og hevder ingen slik kontroll; LATER kontrollerer begge
lagrede M5-representasjoner. Senere coverage-view ombindes eksplisitt uten endrede fysiske inputs.
Behold12288unike TRAIN-rader/768updates, faste4096TRAIN/LATER-paneler og pass3 som primærmåling.
RNG kontrolleres etter hver pass; ingen pass1modell-/tapslikhet forventes mellom arkitekturene.

224unike fokuserte kontroller kvalifisert. Første suite218PASS/1FAIL; feilen var manglende
plan_sha256 i en syntetisk failure-fixture, ikke produksjonskode. Rettet kontroll og5tillegg
for eksisterende routing bestod (6PASS). Original feil-logg/XML er bevart. Uavhengig read-only
integrasjonsreview fant ingen blokkering. Felles målepanel måtte tillate begge boolflagg for å
bygge riktig tapskontrakt; gammel scope-adgang er uendret. Cap-script er uendret.
Faktisk JSON-preflight bestod med eksplisitte sperrer mot native filåpning, np.load, torch.load
og modellkonstruksjon: alle forsøk0.101kildeeiere og input-/normaliserings-/kontraktidentiteter
kontrollert. QUALIFIED_RECIPE_CANDIDATE_001.json binder testet worktree, ikke en launchplan.
Ingen faktisk ny INITIAL, markedsfit, senere læringsgevinst eller økonomisk edge er målt her.

Neste: bind fersk ren commit til eksisterende bevoktet kjørerute og en separat eksakt engangs-
admisjon for denne sammenligningen. Den faktiske INITIAL-kontrollen skal stoppe kjøringen før
første update ved avvik. Ikke gjenåpne tidligere planer. Korte kontroller dømmer ikke endelig
kapasitet; kvalifisert full historikk og30–40epoker er senere skalering.

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
