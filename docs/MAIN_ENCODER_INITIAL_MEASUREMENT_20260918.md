# Fullført nullstegsmåling — historisk plan, ikke relanser

Native guard PASS; ny startbaseline og eksakt bevarte targets/state/RNG er
kontrollert i INITIAL_MEASUREMENT_AUDIT.json. Brukt scope stengt, task Disabled.
Se MAIN_ENCODER_FIXED256_20260918.md for den nye separat bundne prøven.

# Hovedencoder — én native TRAIN-startmåling

Run-id: NATIVE_MAIN_ENCODER_INITIAL_MEASUREMENT_20260918.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.

TRAIN-only/nullsteg kan nå åpnes ved eksakt samsvar mellom policy og recipe.
Læringsporten godtar TRAIN-only initialobservasjoner, men avviser foreldet
ONLINE-modellkilde selv med identisk vekthash. Blandede scopes og optimizersteg
> 0 er avvist ved initialmåling. 44 tester besto / 3 ugyldige kombinasjoner utelatt.
Ni handover-tester bekrefter at ny session velges fremfor forrige checkpoint.
Testlogger/XML: BASE/MAIN_ENCODER_NORMALIZATION_20260918/TRAIN_ONLY_TESTS.*
og INITIAL_HANDOVER_TESTS.*. Ingen modell-/target-/tapsendring i denne rettelsen.

PLAN.json under run-mappen binder én måling: TRAIN 256 Entry/256 Exit-ankre/
1024 samplede Exit-states, null optimizersteg, original frossen lærer, ingen
CONTROL/VAL/TEST eller økonomirollout. TRAIN16, FP32/TF32 av, ett native vindu,
eksisterende vakter. Ingen full epoch eller ny 256-prøve er åpnet.
Avledet initialisering: BASE/MAIN_ENCODER_INITIALIZATION_20260918/RESULT.json.
Original INITIAL_STATE.pt gjenbrukes byteeksakt; original RESULT er bevart.
Proveniens og constructor-audit er bundet. Ny ONLINE-funksjon krever nye
startprediksjoner; gamle prediksjoner er ikke ny baseline.

## Forbered og aktiver én gang

OPERATOR_HANDOVER/OPERATORS_MANIFEST.json under run-mappen binder operatørene.
BIND.py er allerede utført; ikke gjenta. Kjør PREPARE.py fra ren og pushet
GX1_CURRENT via bash scripts/gx1_capped_run.sh --class audit -- .venv/bin/python.
Den gjenbruker fungerende native nullstegsoppskrift og campaign-materialisering.
Kontroller PREPARATION_RESULT, bind faktisk campaign-hash/source_commit inn i
ACTIVATE_TEMPLATE.ps1, og aktiver én gang via den etablerte Windows-tasken.
Ingen gammel plan kan relanseres. Kilden fryses under kjøring. Observer prosess,
progress og terminal receipt; stabil kjøring kontrolleres omtrent hver time.

## Etter terminalt resultat

Deaktiver brukt Windows-task og bevar alle resultater. Tilpass AUDIT_INITIAL.py
fra BASE/NATIVE_PREFIX_INITIAL_MEASUREMENT_20260917 til ny kilde/runtime og bare
TRAIN. Kontroller receipt/guard PASS, byte-/tensorlik model/teacher/optimizer/
EMA/scheduler og bevart CPU/Python/NumPy-RNG, koordinater, masker og endelige verdier.
Sammenlign Entry-targets med CAUSAL_ENTRY_TRAIN_BASELINE_20260917, Exit-targets
med originale nullstegsobservasjoner. Registrer nye faktiske ONLINE-prediksjoner.
Steng brukt scope og oppdater handover. Dette er ingen læring. Først deretter
kan én separat256-prøve bindes. Krev tilstandsavhengig Entry OG Exit mot baselines,
begge sider/måneder; ingen automatisk utvidelse eller TEST.
