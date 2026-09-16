# Exit-gradientklipping — avgrenset læringshypotese, 2026-09-16

## Målt blokkering

På den cachede første native batchen fra original95 er Entry- og auxiliary-
gradientene eksakt null i de private Exit-parametrene (`exit_*`,
`head_exit_action.*`). Likevel klippes Exit sammen med disse oppgavene.
Andre modellparametre har norm33,57565 mot Exit1,98419. Felles klippefaktor
blir0,0297316; egen Exit-gruppe med samme cap1 gir0,503983.

Med lagret Adam-historikk og samme femstegsmål er førsteordens endring i Exit-loss
−0,000258248 med gammel klipping og−0,008182480 med separat Exit-klipping,
omtrent31,7 ganger større lokal nedgang. Entry/auxiliary endres marginalt.
Adam-historikken motvirker noe av Exit-gradienten, men netto oppdatering er
allerede i riktig retning. En Adam-nullstilling er derfor ikke valgt.

Dette er CPU/eval-gradienter, ikke eksakt GPU/train-dropout-paritet, utførte
optimizersteg eller læringsbevis. Første probes ufullstendige cachevalg ble
avvist før gradientberegning; V2 bruker hashbundet samme-batch-cache. Ingen
checkpoint, Adam, lærer, data eller output-cache ble endret.

## Én rettelse

Gradientklipping deles i tre disjunkte grupper: private Exit-parametre,
øvrige modellparametre og task-logvarianser. Alle beholder cap1. Finite-kontroll
av samtlige grupper skjer før mutasjon. Modellen, tapene, femstegsmålene,
vektene, Adam-momentene, EMA, scheduler, RNG og sampler beholdes.

Ny policy: separate_exit_private_model_and_task_weights_v1. Kun original95-
overgangen til den bundne femstegskontrollen kan overføre den gamle
klippepolicyen med beholdt tilstand. Kvitteringen oppgir begge policyene og
bevarte momenter. Historiske kontrakter forblir lesbare.

213 avgrensede CPU-regresjoner bestod i seks testfiler: klippebudsjetter,
finite-before-mutation, komplett serialisert checkpoint-overføring, receipt,
fast32-stegsscope, resume og kernelprofilering. Ingen fullsuite gjentas.
En teststart med audit8G ble avvist før kjøring; faktisk test brukte uendret4G.

## Fast sammenligning før start

Én ny native kandidat: original95/global5777/offset1696 til global5809/offset1728,
32 normale TRAIN16-batcher. Ingen replay. Frossen lærer91, samme femstegsmål og
samme datasekvens som eksisterende reference96. Gammel reference og split kjøres
ikke igjen. Ny artefakt: NATIVE_EXIT_PRIVATE_CLIP_20260916_REFERENCE.
Native campaign, fysisk ny boot, vakter og FP32/TF32 av beholdes. training_enabled
forblir false; kun den eksplisitt bundne32-stegskandidaten er tillatt. Ingen VAL.

Gjenbruk lagret første16/64-kohort med femstegsmål og bred1024/4096-kohort.
Bare nye kandidatoutputs skal beregnes. Rapporter kandidat mot95 og eksisterende96:
Entry/Exit-MSE og MAE, null-/middelbaselines, sentrert prediksjon/target-korrelasjon,
konstant andel av prediksjonsendring, handlinger, retninger og måneder separat.
Bred Exit bruker gamle ettstegsmål som diagnose, ikke som femstegs-loss.

Rettelsen støttes som neste retning bare dersom den forbedrer Exit-tilpasning
og tilstandsavhengig endring mot96 uten klar Entry-forverring. Lavere loss som
bare flytter konstantnivå eller endrer allHOLD/allFLAT er utilstrekkelig.
32 steg gir ikke alene generell lærbarhet, senere VAL-kvalitet eller økonomisk
gevinst. Ved uklart eller negativt resultat: ingen automatisk utvidelse;
bevar resultat og lokaliser gjenstående mål-/verdiproblem. TEST er forseglet.

## Bevis

Under BASE/NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS:
- ADAM_DIRECTION95_20260916_V2/RESULT.json:
  ec3e825d670410f7f1810643fc0ab9bf35a8fa24803ee1f98c5cac271ddaffbb.
- EXIT_PRIVATE_CLIP_DIRECTION95_20260916/RESULT.json:
  83ffe6f978472d47234a23139ef7bebc4369420e5de013c0c258c03417995a9e.
  GRADIENTS.pt er beholdt for gjenbruk. NARROW_TESTS.log dokumenterer regresjoner.
Begge planlagte CPU-prober er ferdige. Ikke gjenta dem.
