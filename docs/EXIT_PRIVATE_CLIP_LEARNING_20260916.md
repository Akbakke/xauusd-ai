# Exit-gradientklipping — fullført og forkastet, 2026-09-16

## Gjeldende beslutning

**Ikke promoter separat Exit-klipping.** Standardkoden tilbakeføres eksakt til
klippe-/kontrakts-/testfilene fra c586520a. Eksperimentkilde34489f3a, original95,
referanse96, kandidat96, Adam, lærer, alle receipts og nye caches bevares.
Den dokumenterte gradientundertrykkingen var reell, men å fjerne den løste ikke
beslutningslæringen. Ingen flere treningssteg eller full VAL er åpnet.

Native kandidaten kjørte32 steg fra global5777 til5809 på fysisk boot447.
GuardPASS, trainer0, observer0; Windows-task er deaktivert. Topp59C core,
66C minne,181,75W og8226MiB. Treningen stoppet på planlagt grense.

| Samme frosne femstegsmål | Før95 | Referanse96 | Separat Exit-klipping |
|---|---:|---:|---:|
| Faktisk trent512, Exit-MSE |29,73131|29,70582|29,64614|
| Separat128 TRAIN, Exit-MSE |41,86735|42,01561|42,64748|

Modellen tilpasser seg altså de trente eksemplene: mot referansen er Exit-MSE
0,20% lavere. På separat128 blir den1,50% høyere.112 av disse128 lå utenfor de32
batchene; dette er fortsatt TRAIN, ikke holdout. Første16 trent alene blir også
verre,26,16781→26,40759; én førstebatch var ikke representativ for hele treningen.
Bred1024 ettstegsdiagnose blir verre9,08508→9,20005, men er ikke femstegs-loss.

Trente512 har gjennomsnittsmål LONG+0,18919 / SHORT−0,06561Bps. Separat128 har
LONG−0,49834 / SHORT+0,55380Bps. Kandidaten predikerer i snitt omtrent+0,257/−0,405
på trente512 og velger2048HOLD for LONG /2048EXIT for SHORT.75,50%/80,93% av
kvadrert prediksjonsendring fra95 er konstant sideforskyvning. Tilstandsavhengig
TRAIN-korrelasjon bedres til0,136/0,077, men separat128 blir−0,444/−0,182.
Dette avkrefter «ingen tilpasning overhodet», men viser at den svake tilpasningen
ikke gir overførbar beslutningskvalitet. Det beviser ikke én endelig kodeårsak.

Entry er fortsatt512/512FLAT på trente data og1024/1024FLAT bredt. Exit-klipping
endrer ikke den frosne, kostnadsdominerte Entry-supervisjonen. Lærerens horisont-
skjevhet bekreftes også på31 nyrekonstruerte batcher: LONG stopper etter ett
beregningssteg på1821/1984; SHORT bruker fem på1983/1984. Dette er targetpolicy,
ikke en handelsregel om holdetid. Alle200 features og tidsrammer er bevart.

CPU-evalueringen av kandidatoutputs tok178,5s. Den etterfølgende eksakte TRAIN32-
rekonstruksjonen tok1059,3s uten optimizer/backward/GPU. Den gjenbrukte første
batch og lagret resterende31 input-/target-/outputcacher. Femsteg128 og form-
analysen krevde ingen nye forwards. Ikke gjenta disse målingene.

Neste avgrensede diagnose: bruk lagrede trace/reward/Q-data til å skille
observert reward, læreravhengig stopp og bootstrap. Et eventuelt felles-horisont-
regnestykke er et kontrafaktisk analysegrunnlag, ikke targetendring eller
holdetidsgrense. Ingen nye klippe-, tapsvekt-, modell- eller optimizerforsøk før
én mekanisme er begrunnet. Ingen automatisk læreroppdatering.

Eksakte bindinger, resultater og beslutning: handover_snapshot/
EXIT_PRIVATE_CLIP_20260916/REVIEW.json. Opprinnelig hypotese og plan nedenfor er
historikk for denne fullførte kandidaten, aldri ny starttillatelse.

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
