# Hovedencoder — én fast256 læringsprøve

Run-id: NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918.
BASE=/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912.
PLAN.json og OPERATOR_HANDOVER/OPERATORS_MANIFEST.json under run-mappen binder
prøve, baselines og operatører. Ingen modell-/treningskode er endret etter a1c4b443.

## Hvorfor prøven nå kan kjøres

Ny native nullstegsmåling er fullført med guard PASS. Audit under
BASE/NATIVE_MAIN_ENCODER_INITIAL_MEASUREMENT_20260918/INITIAL_MEASUREMENT_AUDIT.json
bekrefter identiske kausale Entry-targets, originale Exit-targets, koordinater,
vekter, optimizer, EMA, scheduler og CPU/Python/NumPy-RNG. Ny ONLINE-funksjon
ga nye startprediksjoner (største Entry-avvik 0,146888 Bps fra gammel initial).
Dette er en arkitekturforskjell før trening, ikke læring. Original initialisering
og checkpoints er bevart. Den brukte nullstegsplanen er stengt og skal ikke relanseres.

Hypotesen er én parameterfri final LayerNorm i hovedencoder, med opprinnelig
frossen lærer uten denne normaliseringen. Representasjonsdiagnosen lokaliserte
stor nesten felles hovedbane og svekket Entry-variasjon. Prøven skal teste om
rettelsen faktisk forbedrer læring; mer variasjon alene er ikke tilstrekkelig.

## Omfang og utførelse

Samme ferske initialvekter/RNG, samme4096 ordnede TRAIN-Entries, 256 optimizersteg,
frossen lærer og identiske mål/labels/kostnader/alle200 features og tidsrammer.
Ett native vindu, TRAIN16 og eksisterende vakter. Final ONLINE måles på TRAIN:
256 Entry, 256 Exit-ankre, 1024 samplede states. Ingen CONTROL/VAL/TEST, teacher
refresh, tapsvekt-/terskelsøk, full epoch, økonomirollout eller automatisk gjentakelse.

BIND.py er utført. Forbered PREPARE.py én gang fra ren, pushet GX1_CURRENT via
bash scripts/gx1_capped_run.sh --class audit -- .venv/bin/python.
Kontroller PREPARATION_RESULT og recipe/campaign-hashes. Bind faktisk kildecommit
og campaign-hash i ACTIVATE_TEMPLATE.ps1, og aktiver én gang via samme Windows-task.
Kilden fryses under kjøring. Se faktisk prosess/receipt før handling. En planlagt
fysisk Windows-omstart kan gi kort SSH-brudd; dette gir ingen grunn til relansering.
Kontroller stabil kjøring omtrent hver time; vaktene håndterer maskinvarekontroll.

## Vurdering etter terminalt resultat

Deaktiver brukt task og kontroller guard/receipt samt nøyaktig256 steg og uendret
lærer. Native finalmåler krever eksakt samme targets/masker/koordinater som ny
initialmåling. Gjenbruk lagrede outputs fra residual256 og kausal256; ikke kjør
historiske vekter gjennom ny arkitektur og kall det gamle modellprediksjoner.

Rapporter LONG/SHORT MSE, sentrert feil, korrelasjon, fellesverdi/LONG−SHORT-kontrast,
alle TRAIN-måneder, handlinger og valgt referanseverdi/regret. Sammenlign ny initial,
siste residual256 og relevante TRAIN-konstanter, inkludert FLAT=0. Krev forbedring
for både Entry og Exit (ankre og samplede states), med nyttige tilstandsavhengige
valg. Biasjustering, all-FLAT/all-HOLD eller Entry alene er ikke samlet PASS.

Dette er gjenbrukt TRAIN med fitted-overlapp, ikke generalisering. Referanse-
policyverdier er ikke realisert profitt. Steng scope og vurder utfallet før nytt
arbeid; et svakt resultat gir ikke ekstra trening eller nye brede søk.
