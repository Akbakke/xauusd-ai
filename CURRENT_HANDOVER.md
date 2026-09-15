# GX1 — stoppet; native kontroll bestått, læring fortsatt utilstrekkelig, 2026-09-15

Eneste kilde er /home/andre2/src/GX1_CURRENT, branch work/gx1-current,
rettelsen er pushet i6ffd03ee13e28fd7ef84e65946ece1b038142166, med utgangspunkt
i stoppet kampanje9cd29a2f57a0880e33c6c0a61b279deb7c15b981.
NEXT_RUN_POLICY.json har training_enabled=false. En smal rettelse i optimizerens
gradientklipping er gjort og målrettet testet. Gammel policy er arkivert uendret.
Originale lagrede modellvekter/checkpoints er bevart. Ny native kontroll er
ferdig:16+16 steg, begge guard PASS og trainer/observer exit0. Checkpoint87 har
global5265/epoch_index1/offset1184. Windows-task er Disabled; ingen native prosess.
CPU-overgangen fra85 består bitnøyaktig på15tilstandskomponenter. Kun16+16
ekstra steg var tillatt (global5249/5265), med eksisterende maskinvarevakter.
Gjenopptakelsen over to fysiske booter består; alle726 Adamtilstander økte16 per
vindu, EMA25141→25157→25173. Ikke ny32-sammenhengende-mot16+16-likhetsmåling.
Se handover_snapshot/EXIT_CLIP_CONTROL_RESULT_20260915.json. Paret CPU-måling av
samme TRAIN-input før85/etter87 er ferdig på410.16s, uten optimizer/VAL/TEST.
Se RUNNING_NATIVE_CALIBRATION.json for plan og runtime. Optimizerkontrollens kilderevisjon
er6ffd03ee; to-stegsproben brukte2c593aa7 med identiske produksjonsbindingsfiler.
Gradientkontrollen er også ferdig på kilde194bb596, uten produksjonsendring.
Ingen aktiv jobb; dokumentasjon kan oppdateres.
TRAIN er 2025-06-01 inklusiv til 2026-06-01 eksklusiv:65 295rader/4081steg per
hel epoch. Hele juni2026 er utviklings-VAL. Ingen ny femårs-epoch. TEST er forseglet.

## Stoppet etter brukerens korrigering

Brukeren avviste videre epoch2-trening før læringsproblemet er målt og en konkret
justering er begrunnet. Ingen automatisk gjenopptakelse er tillatt på grunnlag av
forrige anbefaling. «Ingen kodefeil funnet» er ikke tilstrekkelig læringsbevis.

Windows-taskens fremtidige triggere ble deaktivert13:45:44UTC. Guard726 fikkTERM
13:46:26UTC og stoppet nativePID781. Prosessen er bekreftet borte. Siste checkpoint85
ble SHA-verifisert: epoch_index1,offset1152,global5233,slot0,
stateSHA9f9aaba84a7f3fd031666b9761d606af03b7c534409cf3049021b023805db974.
Dette er operatørstopp, ikke en normal native-vindusfullføring eller guardPASS.
Alle checkpoints/resultater er bevart. Den opprinnelige stoppkvitteringen finnes i
handover_snapshot/RUNNING_BEFORE_CLIP_CONTROL_20260915.json. Gjeldende runtime er
RUNNING_NATIVE_CALIBRATION.json; CPU-diagnostikk og tester er ferdige.

## Målt læringsproblem og minste rettelse

Fire forhåndsvalgte TRAIN-batcher à16 er målt med bevarte epoch1 ONLINE-/targetvekter,
native targets og isolert Exit-backward. 410.50sekunder, exit0, ingen optimizer/CUDA/
TEST; checkpoint-, modell- og policybevaring består. Ikke historisk RNG-gjenspilling.
På256 HOLD-celler per side er LONG255 positive prediksjoner mot120 positive targets;
SHORT20 mot128. HOLD-MSE er274.928/267.108, mot nullbaseline274.571/266.658.
Dette er et fast TRAIN-utvalg, ikke fullårsfit eller generalisering. ONLINE Entry
velger64FLAT, læreren58FLAT/3LONG/3SHORT. Selektivitet er ikke ferdig kalibrert.

Exit-gradienten når head/backbone. I batchen med størst feil bruker tapsvektens
gradient83.58% av kvadrert samlet norm. Felles klipping begrenser derfor også
modellens læring. Rettelsen klipper modell og task_log_variances separat, begge
med eksisterende cap1, etter samlet finite-kontroll. Økonomiske targets, tap,
Adam/EMA-tilstand og risiko er uendret. Ingen crash-targets fjernes eller klippes.
109 målrettede optimizer-/profiler-tester består, inkludert8 nye regresjoner.
Ytterligere134 overgangs-/session-/EMA-/campaign-tester består etter retting
av7 nye fixturetilfeller. Se handover_snapshot/OPTIMIZER_TRANSITION_TESTS_20260915.json.
Se handover_snapshot/EXIT_LEARNING_ADJUSTMENT_20260915.json for råbevis og hasher.
Dette beviser mekanisk rettelse, ikke at Exit-biasen er løst eller modellen profitabel.

Den smale overgangen fra checkpoint85 er verifisert på CPU og native GPU. Den bevarer
modell/target/Adam/EMA/RNG/scheduler/progress og binder ny klippepolicy eksplisitt.
NEXT_RUN_POLICY.json tillater bare16/32 ekstra steg for denne kontrollen; hele
epocher forblir blokkert.32 kontrollsteg og paret TRAIN-måling er ferdige; se nedenfor.
Gamle recipe/policy-bindinger skal ikke omskrives eller brukes til automatisk restart.

## Resultat av kontrollen — ikke grønt lys for hel epoch

Samme64 forhåndsvalgte TRAIN-rader,256 HOLD-celler per side, identisk frozen
teacher/input/dropout. HOLD-MSE LONG275.249→275.713, SHORT267.661→268.292;
begge er dårligere enn nullbaseline274.046/266.666. Fortegnssamsvar øker fra
49.22→54.30% og50.00→53.17%, men HOLD flyttes LONG236→89 og SHORT61→212.
Det er ikke dokumentert robust kalibrering. Før/etter32 steg har ingen gammel-
klipping-kontrollarm og isolerer ikke rettelsens årsakseffekt.

Entry velger64FLAT før og etter; verdilæreren58FLAT/3LONG/3SHORT. Den uavhengige
markedsprognosens MAE faller på alle fire eksisterende horisonter, men dette er64
TRAIN-rader før kostnader. På60min er68.75% retningssamsvar svakere enn utvalgets
alltid-opp-baseline76.56%. Ingen generalisering eller profitabilitet er bevist.
Se handover_snapshot/EXIT_CLIP_PAIRED_TRAIN_RESULT_20260915.json for råmåling.

Neste prioritet er det målte svake videreverdisignalet: de to native batchene
viser frozen videreverdi omtrent0.044Bps mot umiddelbar lukkeverdi omtrent−5.8Bps.
Dette forklarer kostnadsdominert Entry-lærer på disse batchene; det beviser ikke
at mulighetene er profitable eller at hyppigere læreroppdatering løser problemet.
Kildegjennomgang bekrefter successor=state_index+1 i
unified_exit_random_access_sampler_v1.py:233, én-stegs reward+frossen bootstrap i
unified_exit_fitted_q_v1.py:309 og Entry-koblingen i
unified_exit_random_access_training_v1.py:564. Læreren kopieres fra ONLINE først
etter komplett epoch/VAL i entry_v10_ctx_train_v3.py:14052–14058.32-stegskontrollen
endret derfor ingen frozen targets; forverret MSE beviser ikke en bedre lærer.

To-stegsundersøkelsen er nå FERDIG, CPU401.32s/peak8 462 888KiB/exit0. Den brukte
samme64 TRAIN-rader/256 HOLD-celler per side og samme frozen teacher. Ingen
optimizer, ONLINE-forward, GPU, VAL, TEST eller produksjonsendring. Opprinnelige
inputs og alle fire originale target-SHA-er ble reprodusert eksakt; kilde,
checkpoint, RNG, sampleplan, policy og modellvekter er bevart.

Resultat: LONG254 unik HOLD og2 unik EXIT ved første successor; SHORT20 HOLD
og236 EXIT. Ingen ties, reelle terminaler eller manglende neste tilstand i dette
utvalget. EXIT-grenen gir eksakt samme mål. Derfor kan flere steg med samme
frosne policy ikke tilføre videre reward i236/256SHORT-tilfellene. Ingen n-stegs-
treningsendring er begrunnet eller innført. Backup-lengde er ingen holdegrense.

Med identiske bevarte ONLINE87-prediksjoner mot et annet mål går LONG-MSE
275.713→191.065, men målspredningen faller16.549→13.828Bps og nullbaseline
274.046→191.353; MAE øker4.086→5.448. Dette er endrede targets, ikke læringsgevinst.
SHORT-MSE268.292→268.621. Ingen optimal n, generalisering eller profitt er bevist.
Se handover_snapshot/TWO_STEP_TARGET_PROBE_20260915.json. Råbatchene og
operatørskriptet ligger i kjøringens OPERATOR_OBSERVATIONS/TWO_STEP_TARGET_PROBE_CPU_20260915.

Nær-Entry-supervisjon mangler ikke:34 av256 sampled states erstate0,60 erstate1–15
og162 senere. State0 har egen bucket. Ingen ombygging av no-loss-ankeret nå.
Sekvensinput er også til stede: Entry96 historiske basebars og egne HTF-sekvenser;
Exit480 historiske M1-bars og kausale MTF-/trade-path-inputs. Entry bruker
transformerencodere; Exit sin nåværende sekvensrute bruker egne GRU-er og
familieattention. Dette er eksisterende arkitektur, ikke en ny endring. Sekvenshistorikk er
ikke det samme som hvilken framtidig økonomi treningsmålet overfører bakover.

Gradientkontrollen er nå FERDIG: V3, CPU 436.926 s, peak 12 697 500 KiB,
exit0 under eksisterende producer 20G/512M. Samme fire TRAIN-batcher, checkpoint87,
faste SHA-bundne targets og native loss-vekting. Ingen optimizer, GPU, VAL eller
TEST. Alle fire originale no_grad-ankre reproduseres eksakt; native CPU-gradientbane
har maksimalt 9.5367e-7 Bps prognoseavvik, ingen endrede Entry-/Exit-handlinger,
eksakt slutt-RNG og identiske rå forecast-/Exit-tap. Dette er ikke GPU-paritet.
Kilde, policy, checkpoints, modell, targets og sampleplan er bevart.

Faktisk Exit-/forecast-overlapp finnes på 138 parameterobjekter. Cosinus er
−0.01193, −0.02251, +0.04800 og +0.01727. Exit-norm på denne støtten er
0.001308/0.000700/0.054388/0.001666 mot forecast 0.823/0.340/0.385/2.716.
Samlet dot(forecast, non-Exit + komplett Exit) er positiv i alle fire batcher:
19.563314, 3.042481, 9.281150, 110.414097. Alle 324 Entry-private encoderparametere
og fire beskyttede rutingsparametere er uten Exit-gradient. Exit-tokenets
reinjeksjon når bare fire parametere i tokenprojeksjonen. Lokale gruppekonflikter
finnes, men disse målingene støtter ikke at Exit samlet ødelegger forecast-læringen.
Rå gradientgeometri er ikke bevis om faktisk Adam-steg, generalisering eller profitt.

Behold arkitektur, gradientgrenser, tapsvekter og eksisterende clipping. Denne
gradientdiagnosegrenen er avsluttet og skal ikke gjentas. Se
handover_snapshot/SHARED_TASK_GRADIENT_RESULT_20260915.json og
handover_snapshot/SHARED_TASK_GRADIENT_REVIEW_20260915.json. Råbatcher, eksakte
inputcacher, skript og logg er bevart under OPERATOR_OBSERVATIONS/
SHARED_TASK_GRADIENT_PROBE_CPU_20260915_V3. Inputcachene gjør nye relevante
avgrensede kontroller mulige uten å laste hele korpus på nytt.

V1 og V2 stoppet før gradientmåling ved krav om eksakt EntryQ-replay. V3
reproduserte eksplisitt originaloperatørens frosne parameterflagg i no_grad-ankeret
og gjenopprettet native flagg før gradientuttak. Alle fire ankrene bestod da.
Lavnivåårsaken til tidligere avvik er ikke isolert; ingen produksjonsfeil eller
korrigert GPU-paritet hevdes. Feillogger/skript er bevart, ikke overskrevet.

Neste arbeid er én avgrenset native læringskontroll rettet mot Exit-critic og
verdilæreren. Definer kontrollen fra checkpoint87 med eksisterende TRAIN-evidens:
vis først bedre tilpasning mot faste targets før en oppdatert lærer tas i bruk.
Dagens lærer oppdateres først etter hel epoch og full VAL; dette er en mulig
flaskehals for videreføring av verdi gjennom etterfølgere, ikke en bevist eneste
rotårsak. Hyppigere læreroppdatering må måles kontrollert og skal ikke innføres
blindt. Ingen ny full epoch, målfrekvensendring eller gjentakelse av ferdig
32-stegskontroll følger automatisk av denne rapporten. Bruk eksisterende native
campaign/profil/vakter; ingen alternativ treningsløype, brede regel-/modelltester
eller vilkårlige risiko-/holdetidsgrenser. Windows-task er Disabled. TEST er bevart.

Metodebakgrunn: policy-evaluering med fler-stegsretur og off-policy-korreksjon,
Munos mfl., https://arxiv.org/html/1606.02647v2 (seksjon1–2). Vår to-stegsprobe
fulgte en frossen greedy-policy og valgte aldri den beste realiserte exit-tiden
i etterkant; ingen generell konvergensgaranti for GX1 ble utledet fra artikkelen.

## Første ettårs-resultat og avgrenset ONLINE/EMA-sammenligning

Hele juniVAL er ferdig og klart negativt. Entry valgte4207LONG/1301SHORT/0FLAT.
Én-posisjonsreplay beholdt første handel til månedsslutt:0modell-exits,1åpen,
5507hoppet over. Kostnadsjustert cash pluss åpen verdi er−1235.8772Bps på fast
nominelt beløp, ikke sammensatt kontoavkastning. Uavhengige entrymuligheter har
snitt−495.3189Bps, ikke porteføljeresultat. Lukkede vinnere er ikke samlet profitt.
Kontrafaktisk lukkes alle5508SHORT vedstate0; LONG785EXIT og4723HOLD tilsplit-end.
Retningsprognosen treffer omtrent48–51% på5–60minutter i denne måneden.
Se handover_snapshot/EPOCH1_FULL_VAL_20260915.json. FullVAL71 315 567tilstandsvisninger/
291294forwards/42449.66beregningssekunder; samlet relativ speedup er ikke målt.

Bevarteepoch1-vekter ble sammenlignet på samme CPU-input: ONLINEvelgerFLAT påalle8
fasteEntries; EMA7LONG/1SHORT. Exit har samme sidehandlinger ibegge på10tilstander.
EMA beholder81.19% parametervekting fra start etter4081steg, men forklarer ikke
Exit-skjevheten alene. Ingen EMA-policy er endret. Åtte rader er ikke full ONLINE-VAL.

CPU-sanity forblir REVIEW_EMA_REFERENCE_MISMATCH: CPUbatch8 mot GPUEntrybatch16 har
max0.00012672Bps Q-avvik. Alle8EMA-handlinger matcher; minstemarginEMA0.20565Bps/
ONLINE4.48361Bps. Ingen konkret input-/normaliseringsfeil funnet. Det kvalitative
sammeCPU-funnet er tydelig, men numerisk paritetPASS og lønnsomhet hevdes ikke.
Toleransen er uendret; ingen gjentatt kjøring er nødvendig for dette funnet.
Se handover_snapshot/EPOCH1_ONLINE_EMA_COMPARE_20260915.json.

## Bevarte porter og arbeidsregler

Native porter på128c55f2 er bestått: faktisk overgang fra original315, GPU256-
paritet, målt absolutt fart og eksakt32mot16+16 resume på14tilstandskomponenter.
Dette er historiske bevis fra før optimizerrettelsen. Gjenbruk uendrede inferensbevis;
ny optimizer-/checkpointovergang må verifiseres før videre trening.
Gamle femårsresultater/original315og19 908steg er bevart; gammel juni analyseres
bare med første gamle epochs uforanderlige EMA. Gamle kildekopier er avhengigheter.

Ingen fast holde-/tapsgrense. Bevar alle200features,familier,tidsrammer,kostnader,
successor-semantikk og checkpoints. Én tung jobb samtidig; underagenter er autorisert
for avgrenset arbeid, men ingen sideoppgaver under venting. Stabil drift sjekkes
omtrent hver time. Stående autorisasjon gjelder. Ingen TEST/live/papir/spending.

Oppdater handover ved vesentlig endring og commit/push ferdig rettelse med bevis.
Resultatbevis og alle gamle kjøringer bevares. Eldre driftsdetaljer er bevart i
handover_snapshot/CURRENT_HANDOVER_BEFORE_EPOCH2_RESUME_20260915.md.
