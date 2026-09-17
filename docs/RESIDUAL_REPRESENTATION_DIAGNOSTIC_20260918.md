# Residualkandidat — to native inferensmålinger

CPU-kontrollen bekreftet identiske inputs, radrekkefølge, targets og masker for
bevart TRAIN16 og ny sluttmåling. Gammel caches sluttprediksjoner tilhører den
kausale modellen; nye prediksjoner hentes fra residualkandidatens native resultat.

Eksisterende diagnose var låst til gammel review/verdict og krevde en separat
parity-kjøring. En avgrenset variant i samme native kjørevei måler nå bare
representasjoner i vanlig inferens: to forwards, null backward/optimizersteg.
Den krever den nye review/verdict og hashbundet CPU-inputkontroll, og kontrollerer
prediksjoner mot lagret fasit med uendret toleranse og identiske handlingsvalg.
Modell, treningsmatematikk, guards, features, targets og checkpoints er uendret.

25 fokuserte tester er kontrollert: 24 bestod første runde; den siste fikk en
rettelse i syntetisk inferenstensor-håndtering og bestod egen ny kjøring.
Bevis: BASE/RESIDUAL_REPRESENTATION_INPUT_AUDIT_20260917/{RESULT.json,TESTS.log,TEST_RECHECK.log}.

Gjeldende run-id: NATIVE_RESIDUAL_REPRESENTATION_20260918.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
PLAN.json binder ny sluttstate, original lærer, cache og native TRAIN-observasjon.
OPERATOR_HANDOVER/PREPARE.py og ACTIVATE_TEMPLATE.ps1 gjenbruker etablerte
native operatører. Forbered én gang etter ren commit/push via audit-vakt;
bind så faktisk campaign-hash og kildecommit før én aktivering. Kontroller
PREPARATION_RESULT, prosesser og terminal receipt; ikke relanser brukt plan.

Målepunkter: hoved-fuse før/etter, tre residualkorreksjoner, rå local/MTF/context,
joint LayerNorm, lineær mikser og Entry-hidden etter GELU. Sammenlign spredning
før/etter mot bevart initialmodell. Dette måler variasjon, ikke læring eller
forutsigbarhet i targets. Ingen Exit-/CONTROL-/VAL-/TEST-forward er åpnet.
Etter terminalt resultat: vurder rapporten, steng brukt scope og dokumenter
én neste evidensbasert handling. Ingen automatisk trening eller ny normalisering.
