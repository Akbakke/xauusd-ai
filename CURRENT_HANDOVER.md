# GX1 — stoppet; konkret læringsrettelse verifisert, 2026-09-15

Eneste kilde er /home/andre2/src/GX1_CURRENT, branch work/gx1-current,
utgangspunkt pushet commit9cd29a2f57a0880e33c6c0a61b279deb7c15b981.
NEXT_RUN_POLICY.json har training_enabled=false. En smal rettelse i optimizerens
gradientklipping er gjort og målrettet testet. Gammel policy er arkivert uendret.
Ingen lagrede modellvekter eller checkpoints er endret. Ingen aktiv trening.
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
Alle checkpoints/resultater er bevart. RUNNING_NATIVE_CALIBRATION.json binder
stoppkvittering og prosess-/checkpointbevis. CPU-diagnostikk og tester er ferdige.

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

En smal overgang fra checkpoint85 er under målrettet verifikasjon. Den bevarer
modell/target/Adam/EMA/RNG/scheduler/progress og binder ny klippepolicy eksplisitt.
NEXT_RUN_POLICY.json tillater bare16/32 ekstra steg for denne kontrollen; hele
epocher forblir blokkert. Neste er faktisk tilstandsovergang og avgrenset native
læring/resume før ny beslutning om hel epoch.
Gamle recipe/policy-bindinger skal ikke omskrives eller brukes til automatisk restart.

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
