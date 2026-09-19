# Hovedencoder256 — mer Entry-signal, ingen bedre beslutninger

Paret CPU-analyse er fullført19. september2026. Samme256 TRAIN-Entries,
256 Exit-ankre og1024 samplede states, begge sider og alle ni TRAIN-måneder.
Ny native initialbaseline, kausal256, residual256 og TRAIN-konstanter er
sammenlignet på eksakt samme korrigerte targets/masker/cohort. Gamle outputs
og alle originalfiler er bevart. Ingen nye forwards, fits eller optimizersteg.
Saved-state tensorhash, lærer, optimizer/EMA256, cursor og receipts er verifisert.
Historiske aggregater er uavhengig gjenskapt.

| MSE, Bps² | Ny initial | Residual256 | Hovedencoder256 | TRAIN-konstant |
|---|---:|---:|---:|---:|
| Entry LONG | 633,953 | 613,846 | 610,234 | 608,803 |
| Entry SHORT | 685,293 | 606,286 | 604,074 | 603,814 |
| Exit-anker LONG | 629,634 | 626,062 | 626,710 | 625,037 |
| Exit-anker SHORT | 627,783 | 622,144 | 622,628 | 622,612 |
| Exit-samplet LONG | 864,789 | 865,095 | 864,918 | 864,710 |
| Exit-samplet SHORT | 859,502 | 858,597 | 858,564 | 859,420 |

Entry LONG−SHORT-korrelasjon steg0,094→0,242 mot ny initial. Når hver måneds
middel fjernes, steg samlet korrelasjon0,066→0,231; residual256 var0,115.
Korrelasjonen er positiv i alle ni måneder, men de små gjenbrukte utvalgene
beviser ikke robust generalisering. Sentrert Entry-feil slår residual på begge
sider samlet og i8/9 LONG-måneder og9/9 SHORT-måneder. Sammenlignet med ny
initial er samlet sentrert SHORT-feil svakt dårligere601,234 mot600,997.
Begge Entry-sider taper fortsatt mot konstanten på vanlig MSE.

Dette er en målbar, begrenset Entry-forbedring i TRAIN; mer enn bare endret bias.
Men Entry velger FLAT256/256, akkurat som residual256 og TRAIN-konstanten.
Beste LONG-minus-FLAT er−5,317Bps og beste SHORT-minus-FLAT er−5,866Bps.
LONG/SHORT-prediksjoner varierer bare0,186/0,276Bps i standardavvik rundt
−5,658/−6,275Bps. Variasjonen gir ingen endring av beslutningene.

Exit velger fortsatt HOLD for alle LONG og EXIT for alle SHORT, både ankre
og samplede states. Alle fire Exit-celler har dårligere sentrert MSE enn
residual256. Ingen handlings- eller referanseverdi/regret-forbedring mot residual.
På samplede states taper begge faste Exit-valg dessuten mot deres respektive
TRAIN-konstantvalg. Entry alene er ikke en bestått samlet læringsport.

Konklusjon: REJECT_EXPANSION_MAIN_ENCODER_PARTIAL_ENTRY_SIGNAL_NO_DECISION_IMPROVEMENT.
Ingen ny trening/full VAL/CONTROL/TEST er åpnet. Retningssignal i gjenbrukt TRAIN
betyr ikke lønnsom handel. Referanseverdi/regret er ikke realisert profitt.
Targetvarians inneholder støy; det er ikke riktig å skalere prediksjoner opp til
targetvariansen eller justere terskler for å tvinge fram handler.

Neste konkrete spørsmål er om tidligere representasjonskollaps fortsatt finnes
etter encoderrettelsen. Gjenbruk cached TRAIN16 og eksisterende native måler;
kontroller bindingsstøtte for NY initialfunksjon og dagens sluttreview før én
separat bundet initial/final representasjonsmåling. Ingen flere normaliseringer,
tapsendringer eller ekstra trening uten målt gjenværende årsak. First-batch-
loggen viser Entry-rutingsgradient21,237, men dette er ikke full oppdaterings-
attribusjon eller bevis mot konkurrerende gradienter i delte modellgrener.

Artefaktrot: /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912/NATIVE_MAIN_ENCODER_NORMALIZED_FIXED256_20260918.
PAIRED_TRAIN_REVIEW.json, DECISION_GAP_AUDIT.json, VERDICT.json og
COMPLETION_REVIEWED.json er nye bevis. REVIEW_OPERATOR.py og
DECISION_GAP_OPERATOR.py beskriver beregningen. Ikke gjenta ferdige analyser.
