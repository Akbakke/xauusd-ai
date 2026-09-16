## GPU-minnefeil avdekket og smalt rettet; ny kortkontroll kreves — 2026-09-16

Reference32 på0f0a3542/boot443 er terminal med CUDA OOM i første frozen-target
forward, før noen optimizersteg. Native checkpoint95/global5777/offset1696 er
lagret; faktisk modell/target/Adam/EMA/RNG/progress er bevart i alle15 felt.
Windows-task er Disabled, ingen native/guard/controller-prosess lever. Den gamle
ACTIVE_INVOCATION-filen er et restartefakt, ikke en levende kjøring. Ingen
terminal campaign-receipt ble produsert. Ikke start den feilede planen igjen.

336 samtidige lærer-rader krevde3.44GiB ekstra GRU-minne utover PyTorch-grensen
10.80GiB. Guard-topp10034MiB,48Ccore/50Cmem/155.82W; dette er ikke overoppheting.
Minste rettelse deler bare frossen trace-target-forward i den eksisterende
ettstegsgeometrien:64successors+16ankre=80rader.336 blir80+80+80+80+16.
Samme row-order, inputs, femstegsmål, anker, TRAIN16, VAL256, sampler, vekting og
én online-forward/backward. Ettstegsbanen og alle minne-/maskinvaregrenser beholdt.

48 målrettede tester består. Cachet faktisk CPU-paritet bruker samme lærer91:
Exit-handlinger og Entry-lærervalg er identiske, femstegs HOLD-targets eksakt like,
maxQ-avvik2.38419e-7Bps, maxEntry-targetavvik2.98023e-8Bps.19.39s totalt,
9.75s target-forwards, topp-RSS2296204KiB. Ingen ny datamaterialisering/GPU/
optimizer/backward/VAL/TEST. To måleskriptfeil før modellforward er bevart;
produksjonskode ble ikke endret mellom CPU-forsøkene. Ikke gjenta cachemålingen.

Neste: commit/push og ny sourcebundet reference32/split16+16 via eksisterende
native campaign og friske booter. Gjenbruk faktisk15-felts checkpointovergang;
overgangseierne er uendret. Ny GPU-minne/paritet/fart/resume og faktisk læring
er ennå ikke bevist. Ingen full epoch/VAL, læreroppdatering, EMA-reset eller replay.
Bevis:FROZEN_TRACE_GPU_MEMORY_REVIEW_20260916.json og
FROZEN_TRACE_MEMORY_CPU_PARITY_20260916.json i handover_snapshot.

## Native femstegs-binding testet; faktisk overgang og GPU-kontroll gjenstår — 2026-09-16

Eksisterende continuation-origin fra original95 kan nå eksplisitt binde
exit_backup_steps=5 i origin, recipe, NEXT_RUN_POLICY og session-kontrakt.
Manglende/ulike bindinger avvises. Gamle recipes beholder standard ett steg.
Ny scope tillater bare teknisk reference32 eller split16+16 fra5777 til5809
optimizersteg, uten full epoch/VAL, læreroppdatering, replay eller vektnullstilling.
training_enabled er fortsatt false. Den stoppede134-kjøringen skal aldri restartes.

262 målrettede tester består, inkludert bevart modell/target/Adam/EMA/RNG,
datarekkefølge/progress, serialisert resume og SHA-bundet EMA-historie; feil
originer, læringsendringer, økt steggrense og manglende bindinger avvises.
Dette er testbevis. Faktisk original95-overføring på CPU skal verifiseres før
guarded native32 mot16+16, GPU-minne/paritet/fart og gjenopptakelse måles.
Native-kilden skal fryses under kjøring; CPU-target-cachene gjenbrukes.
Se handover_snapshot/FROZEN_POLICY_TRACE_NATIVE_BINDING_20260916.json.
Ingen tung jobb eller native-kjøring aktiv; Windows-task er fortsatt deaktivert.

## Femstegs lærerberegning implementert, ikke aktivert i native trening — 2026-09-16

Valgfri backup_steps=5 er implementert i eksisterende state-view/adapter/fabrikk/
treningsberegning. Standard er fortsatt1; gammel native recipe aktiverer ikke
endringen. Én online-forward/én frossen forward/én backward, samme sampler,
vekter og Entry-anker. Ekstra fremtid brukes bare i targets. Lærerens unike
HOLD følger observert neste reward; EXIT/tie beholder lærerens verdi. Grensen
på fem beregningssteg og faktisk manglende ekstra successor bootstrappes,
uten ny terminal, maksimum holdetid eller etterpåklok maksgevinst. Ekte
terminaler og kalenderdiskontering bevares; ubekreftet markedsgap avvises.

48 målrettede tester for berørte læringsbaner består. Utvidet fabrikkfil har
10 beståtte og2 eksisterende feil: gamle tekstasserts forventer avviklede
fixed-step/separate-VAL-veier. Både tester og capped-runner er identiske med
HEAD før endringen; vakter er ikke endret. Dette er ikke en grønn fullsuite.

Én ekte TRAIN-batch16/64 sampled transitions ved offset1696 fra original95 er
kontrollert på CPU, frossen lærer91. Inputs, sampler, vekting og anker er eksakt
bevart. Lærerbatch80→336: identiske handlinger, maxQ-avvik2.38419e-7Bps;
første ettstegstarget avviker høyst4.76837e-7Bps, Entry-target2.98023e-8Bps.
69 av128 HOLD-targets får ekstra prissteg, maks endring25.486Bps. LONG følger
fem steg i5/64, SHORT64/64; resterende59LONG stopper ved lærerens første EXIT.
Dette er mekanisk målberegning på én batch, ikke læring, generalisering/profitt.

CPU-kontroll V2 tok380.27s, topp-RSS12872.16MiB, exit0; ingen optimizer/backward,
GPU/VAL/TEST. V1 feilet i rapportdelen etter forwards ved feil nivå i eksisterende
target-cache. Måleskriptet er rettet; produksjonskode uendret mellom forsøkene.
Begge operatorer/logger bevares. V2-inputs/targets er nå cachet før rapporten.
Ikke gjenta materialiseringen. Se FROZEN_POLICY_TRACE_REVIEW_20260916.json og
FROZEN_POLICY_TRACE_NATIVE_BATCH_20260916.json under handover_snapshot.

Neste nødvendige arbeid: eksplisitt bundet native backup-policy og smal
checkpointovergang, deretter GPU/minne/fart/resume og en avgrenset faktisk
læringskontroll. Ikke ny modell, tapsvekt, EMA-reset, regeljakt eller full epoch.
Native134/global8162 og Windows-task forblir stoppet/deaktivert; training_enabled
er false. Alle gamle checkpoints og ufullstendig VAL bevares. Ingen tung jobb aktiv.

## Bred faktisk TRAIN-kontroll ferdig — 2026-09-16

Én CPU-jobb på kilde4310254c er terminal/exit0 etter979.6728s, topp-RSS8744.6MiB.
1024 ulike TRAIN-innganger,64 forhåndsvalgte batch-offsets1696..4079 og4096 ekte
native epoch1-overganger dekker alle12måneder. Ingen optimizer/backward/GPU/VAL/
TEST. ONLINE95 og ONLINE134 sammenlignes i eval-modus mot identisk frossen lærer91.
Originaler, samplerrekkefølge og v4-økonomi beholdt. Alle inputs/targets/outputs
cachet; ikke materialiser eller forward denne kontrollen på nytt.

Exit HOLD-MSE LONG18.2180→18.2337 mot nullbaseline18.2128; SHORT18.0657→18.1232
mot18.1355. Ingen forbedring på målt kohort; ikke påstå statistisk signifikans
eller en bevist eneårsak. Entry-MSE12.5654→12.6256; begge1024FLAT mot lærerens
914FLAT/49LONG/61SHORT. Forecast forbedres ved25/60/120 nominelle minutter;
120min-MSE1968.89→1646.33, retningsandel63.57%→66.60%. Dette er TRAIN, ikke
holdt-utenfor prediksjon eller profitt.

Gjenbrukte cachedata bekrefter tidligere begrensede Entry-ankerdekomponering:
lærerens fremtidige verdi i snitt0.02456Bps, maks0.09434Bps.1021/1024 lærervalg
er identiske med umiddelbar første-M1-likvidasjon; bare3 FLAT blir nye handler.
Entry får dermed nesten ingen verdi fra senere hold. Koden bruker en-M1-backup
og normal målmodelloppdatering først etter full epoch/VAL. Dette er en konkret
hypotese om for treg tilbakeføring av fremtidig verdi, ikke et bevist tiltak.

Neste avgrensede læringshypotese er flertrinnsbackup under frossen lærerpolicy,
med fem observerte M1-successors knyttet til eksisterende M5-entrydatakadens.
Dette er ikke en holdetidsgrense: bootstrap må fortsette verdien utover grensen,
intermediære EXIT-valg følger lærerens kausale Q, aldri fasitbasert fremtidsmaks.
Bruk eksisterende eiere og tidligere to-stegsbevis; ikke en ny modell, EMA-reset,
tapsvekt-/gradientendring, terskeljakt eller fixed64-replay. Hypotesen er ennå
ikke implementert/aktivert og må måles som faktisk læring før noen ny full epoch.
Den eksisterende treningsblokkeringen og deaktivert Windows-task gjelder.

Bevis:BROAD_TRAIN95_134_RESULT_20260916.json,
BROAD_TRAIN95_134_TARGET_DECOMPOSITION_20260916.json og
BROAD_TRAIN_LEARNING_REVIEW_20260916.json under handover_snapshot.
Ingen tung jobb aktiv. Alle snapshots134/95/99/115 og ufullstendig VAL bevart.

## Verifisert fartstilpasning og aktuell læring — 2026-09-16

Native kjøring er fortsatt STOPPET. Den avsluttede fortsettelsesscope er fjernet
fra NEXT_RUN_POLICY.json; training_enabled er fortsatt false. Checkpoint134/global8162 bevart; ingen ny
VAL eller epoch er startet. Ufullstendig EMA-VAL hadde64,785,368 HOLD/nullEXIT.
Nåværende ONLINE134 er sammenlignet med egen EMA på samme forhåndsdefinerte
8 VAL-innganger/10 tilstander. ONLINE Entry8FLAT; EMA7FLAT/1SHORT. ONLINE Exit
19HOLD/1EXIT, EMA20HOLD. Dette er et lite fast utvalg, ikke full selektivitet,
retningskvalitet/profitt eller bevis for at EMA alene er årsaken til HOLD.
Begge bruker egne kausale Entry-representasjoner og samme faktiske markedsinputs.

Profilering av batch256/8CPU lokaliserte økonomiberegning som største av de tre
målte CPU-trinnene. Minste rettelse er tre hashkall i eksisterende VAL-adapter:
bruk den eksisterende JSON-hasheieren på JSON-økonomidata, uten rekursiv
array-projeksjon. Ingen endring i modell, læring, handling, økonomi eller vakter.
Paret måling med identiske cachetilstander:økonomitrinn1.32656x og målt samlet
CPU-materialisering/batchbygging/økonomi1.13905x. Alle faktiske tilstands-/økonomi-
hasher og verdier er eksakt lik frossen baseline; endret økonomikonvolutt avvises.
25 målrettede rollout/provider-tester består. Full native/GPU/totalfart etter
rettelsen er IKKE målt; ikke presenter CPU-tallet som full VAL-speedup.

ONLINE/EMA-målingen tok155.94s. Første profileringsdel ble avvist på for lav
måleskriptgrense før profilkjøring; bare operatoroppsettet ble rettet, og
modellforwards ble ikke gjentatt. CPU-profileringsjobben og paret rettelsesmåling
er begge terminale/exit0, henholdsvis165.37s og164.98s. Alle operatorer/logger
og originale checkpoints/resultater er bevart under kjøringens OPERATOR_OBSERVATIONS.

Neste læringsarbeid: bruk eksisterende TRAIN-target-/tapsbevis til å skille
svakt Exit-signal fra manglende generalisering. Ikke anta at mer fart, EMA-bytte
eller enda en epoch løser HOLD. Ingen bred regel-/modelljakt eller samme-kohort
fit-repetisjoner. Bevar brukerens risikomål uten fast grense; TEST forseglet.
Bevis:handover_snapshot/CHECKPOINT134_CPU_COMPARE_20260916/ og
handover_snapshot/REAL_TRAIN_VAL_STOP_20260916.json. Ingen tung jobb aktiv.

## STOPPET etter brukerbeskjed — 2026-09-16 08:13:52 UTC

Native VAL er stoppet og Windows-task er Disabled/Enabled=false/LastResult1.
Alle native-/guard-/capped-prosesser er avsluttet. Checkpoint134/global8162 og
originale resultater er bevart. OPERATOR_STOP_20260916 inneholder konsistent
TRAIN-pointer, continuation receipt og VAL-progress.32,392,684 tilstandsvisninger,
64,785,368 aktive sidebeslutninger, alle HOLD/nullEXIT på begge sider. Dette er
hypotetiske sidebaner, ikke unike/Entry-valgte handler eller samlet profitt.
Guard ryddet resterende workers etter operator-SIGTERM; dette er operatorstopp,
ikke en naturlig guardPASS eller maskinvarefeil. Tidligere ETA er ikke gjeldende.

Ny prioritet fra brukeren: maksimer faktisk fart og stopp unyttig VAL. Målt GPU
snitt4.35%/126W viser ledig kapasitet; flaskehals er ennå ikke lokalisert. Gjør én
avgrenset CPU-profil av eksisterende VAL-eiere og sammenlign aktuell ONLINE134
mot EMA på den allerede definerte kohorten. Bare målt blokkering gir kodeendring.
Ingen ny epoch/fullVAL/replay eller spekulativ EMA-/modellendring. TEST forseglet.
Kilde er fortsatt fdd70e5c; stoppet plan skal ikke startes automatisk.

## Brukerbestilt ETA/GPU-måling — 2026-09-16 08:05 UTC

FullVAL har31,642,401 av maksimalt84,049,614 tilstandsvisninger,37.65% dersom
HOLD fortsetter. Maksgrensen følger boundVAL-root84044106 successors+5508 entries.
Målt samlet beregningsfart1405.83views/s gir ca10.36 beregningstimer igjen;
omstarter/innlasting kommer i tillegg. Foreløpig ETA21–23 norsk tid16.sep,
avhengig av samme fart/HOLD; ingen garanti eller ferdig resultat.
20x1s GPU-prøver:snitt4.35% GPU(min1/max8),126.05W under300W-grense,1395MHz.
GPU er ikke mettet. Nøyaktig flaskehals er ikke lokalisert; ikke påstå maksimal
maskinvareutnyttelse. Gjeldende256/8CPU/3h-profil og vakter er i bruk.
Aktiv/frossen kjøring bevares, ingen spekulativ effekt-/batch-/kodeendring.
Bevis:handover_snapshot/REAL_TRAIN_VAL_ETA_GPU_20260916.json. Neste vanlig
kontroll rundt09:05UTC. HOLD-only-funnet består; ingen ny epoch automatisk.

## Nytt målt læringsproblem i pågående VAL — 2026-09-16 06:35:27 UTC

Brukerbestilt status viste52,052,574 registrerte aktive sidebeslutninger,
alleHOLD og0EXIT, fordelt likt påLONG/SHORT. Telleren er bekreftet i evaluatorens
accumulate_q_diagnostics:gjentatte beslutningspunkter på alle evaluerte mulige
prisbaner, ikke52m unike/valgte handler.26,026,287 tilstandsvisninger,18,378.32
beregningssekunder. FullVAL er IN_PROGRESS; Entry-selektivitet og samlet profitt
er ennå ikke vurdert. Dette er et konkret faresignal om gjeldende EMA-Exit.
Det er ikke bevis for at ONLINE har samme feil, eller for at EMA alene er årsaken.
Ingen ny epoch er tillatt. Aktiv kjøring/kilde bevares; ikke kode spekulativt.
Bevis:handover_snapshot/REAL_TRAIN_VAL_HOLD_OBSERVATION_20260916.json med binding.
Native751/guard696/capped645 var levende06:33:39UTC. Neste ordinære kontroll
rundt07:48UTC. Budsjettstopp etter fullVAL er fortsatt bindende.

## Full juni-VAL kjører — 2026-09-16 02:04:20 UTC

Native756/guard701/capped650 lever på boot440. Checkpoint134/global8162 er fortsatt
VAL-grunnlaget. ROLLOUT_PROGRESS har4743663 tilstandsvisninger/19008 forwards og
3364.97 beregningssekunder; IN_PROGRESS, ingen samlet resultatpåstand.
Innebygd nåværende batch256-paritet:identiske handlinger, maxQ-avvik1.49e-8Bps;
batch-inferens3.127x mot16. Cache og8CPU gir identiske handlinger/økonomisteg/hash.
Dette er innebygde profilkontroller, ikke samlet relativ speedup eller profitt.
Guard57C/66C/8814MiB. Neste ordinære kontroll rundt03:04UTC.

## TRAIN ferdig; full VAL starter — 2026-09-16 01:02:59 UTC

Alle2385 ekte resterende TRAIN-steg er lagret. Native checkpoint134 er
phase=validation/epoch_index1/global8162/offset0. Invokasjon1 endte RESUMABLE,
guardPASS, trainer0/observer0; pause=native_full_val_phase_boundary.
Guardmaks62C/70C/194.15W/8006MiB. Kampanjen gikk automatisk videre etter ny
fysisk boot00:51:31UTC. Invokasjon2 startet00:54:54UTC; native756/guard701/capped650
lever01:02:59UTC under innlasting for full juni utviklings-VAL. Ingen ny TRAIN-epoch.
Neste ordinære kontroll rundt02:03UTC. Ingen endring av frossen kilde.

## Siste driftskontroll — 2026-09-16 00:36:55 UTC

Native791/guard736/capped685 lever. Checkpoint127/global7825/epoch1offset3744
er lagret:2048 av2385 nye TRAIN-steg ferdige,337 gjenstår. Mellom to kontroller
ble1984 steg lagret på3671.38s, ca1.85s/steg. Dette er observert TRAIN-fart,
ikke full kampanjegjennomstrømning. Guard58C/68C/8006MiB. Neste kontroll rundt
01:01UTC ved forventet overgang til full VAL, deretter omtrent hver time.

# Aktiv native TRAIN→VAL — 2026-09-16

Kilde fdd70e5cf872e32f4f4102f659a47d55178a7f19 er committet, pushet og frosset.
CPU-overføringen av originalt ikke-replay95 består: alle15 tilstandskomponenter
bevart bortsett fra ny kjøreidentitet; original sampler gir2385 ekte resterende
TRAIN-batcher. Seks numeriske funksjoner er AST-identiske med opprinnelig FQI-kilde.
269 målrettede tester og vanlig commit-hook består. Hjelpeskriptets første
forsøk stoppet før state-load på ekstra metadatafelt; bare dette feltet ble rettet.
Bevis og begge operatørforsøk er bevart. CPU-kontrollen tok131.492124s.

Eksisterende Windows-task ble bundet til ny plan og startet én gang23:18:32UTC
etter fysisk boot439/23:17:59UTC15.sep. NativePID791/guard736/capped685 er observert
levende23:35:44UTC. Faktisk native95-overgang er bekreftet med state_preserved=true,
optimizer_procedure_changed=false og EMA25685. Checkpoint96/global5841/offset1760
er lagret:64 nye optimizersteg. Normal guard:core59C/memory66C/8006MiB.
Neste ordinære sjekk tidligst00:35:44UTC16.sep ved stabil drift.
Kjøreplanen tillater fullføring av eksisterende ettårs-epoch og hele juni-VAL,
med completed_val_epochs2 som stopp før neste epoch eller automatisk omstart.
Den begrenser ikke modellens holdetid eller tapsstørrelse. TRAIN16/VAL256/8CPU/3h.
Original95/99/115 bevares; repetisjonsresultatene brukes ikke som årsdekning.
Se RUNNING_NATIVE_CALIBRATION.json for plan-, recipe- og tilstandsbindinger.
Ikke endre frossen kilde eller start på nytt ved observasjonstimeout. TEST forseglet.

# Gjeldende arbeid — vanlig TRAIN til neste VAL, 2026-09-16

Lærbarhetskontrollen er avsluttet og bevart. Neste kjøring forberedes fra
originalt IKKE-replay checkpoint95/global5777/epoch1offset1696. Den bruker de
2385 gjenstående ekte TRAIN-batchene og full juni utviklings-VAL. Stopp etter
completed_val_epochs2/global8162 før neste epoch og automatisk omstart.

Konkret blokkering var at eksisterende origin-/budsjettbinding kun tillot den
avsluttede repetisjonskontrollen. Fem eksisterende eiere er justert for eksakt
95-overgang, eksisterende completed-VAL-budsjett og kampanjestopp ved VAL-grensen.
Modell, tap, optimizerberegning, sampler, risiko og data er uendret. Ingen ny
kjøring er startet; faktisk CPU-overføring av95 og planbinding gjenstår.

Målrettet verifikasjon:268 tester bestod første gang. Én eldre test antok at
original95 aldri kunne brukes til vanlig fortsettelse; den kontrollerer nå at
95 fortsatt avvises under gammel fixed64-policy, og bestod ved målrettet omkjøring.
269 ulike testtilfeller består. Tekniske bevis fra128c55f2 gjenbrukes med sine
opprinnelige kilde-/checkpointbindinger, ikke som ny måling på dagens kilde.
Neste resultat skal avgjøre faktisk Entry-/Exit-kvalitet og netto økonomi på VAL.
TEST forblir forseglet. Ingen ytterligere epoch uten resultatvurdering.

# Gjeldende status — Entry lærer selektive valg, 2026-09-16

Native fortsettelse99→115 er ferdig på frossen kilde b493e585.1024 nye
optimizersteg tok2679.503783s (44m39.5s) inkludert oppstart/observer, uten reboot.
Checkpoint115/global7057/epoch1offset2976 er diagnostisk. GuardPASS, trainer0,
observer0; Windows-task Disabled22:44:31UTC2026-09-15. Ingen aktiv tung jobb.
Planen er uttømt. Original95/99, lagret epoch_order og fast lærer91 er bevart.
Alle726 Adamtilstander økte1024; EMA25941→26965, offset19908.236tester og normal
commit-hook bestod før kjøringen. Modell, tap og gradientgrenser er uendret.

Paret CPU-måling99→115 er ferdig, session34866/exit0,27.498018s. Målte99outputs
ble gjenbrukt og deres samlede statistikk matchet nøyaktig; kun115 ble forwardet.
Samme64cachedinputs/targets/dropoutfrø; ingen korpus-/lærerberegning, optimizer,
backward, GPU, VAL eller TEST. En første evaluatorvariant hadde variabelkollisjon
før modellkonstruksjon; kun måleskriptet ble rettet. Feilforsøket er bevart.

Entry-Q-MSE17.321052→7.085304, ned59.09%. Valgene endret seg fra64FLAT til
61FLAT/2LONG/1SHORT. Alle tre handelsvalg stemmer med lærerens foretrukne side;
55av55 lærer-FLAT forblirFLAT. Seks av ni lærerforetrukne handler overses fortsatt.
Exit-MSE0.342115→0.055011. HOLD/EXIT-treff mot mål ca94.92% LONG og95.31% SHORT.
Forecast-retning ved5/25/60/120 nominelle minutter:78.125/81.25/84.375/89.0625%.
Dette er lokal læring etter samlet1280steg/320repetisjoner av64 TRAIN-innganger.
Ingen generalisering, sannsynlighetskalibrering eller lønnsomhet er dokumentert.

Avslutt nå fixed64-øvelsen; ikke jag perfekt treff på disse ni lærerpreferansene.
Neste arbeid er en avgrenset native fortsettelse på virkelige ettårs-TRAIN-rader
fra bevart IKKE-replay checkpoint95/global5777/epoch1offset1696, frem til neste
fulle juni utviklings-VAL. Gjenbruk modell/tap/profil og eksisterende tekniske
bevis.115 skal ikke promoteres til produksjon eller telles som årsdekning.
Avklar bare nødvendig resume-/sourcebinding i eksisterende native eier; ingen
nye modeller, terskler, tapsendringer eller ekstra samme-kohort-prober nå.
Ingen ny kjøring er ennå forberedt eller startet. Vurder faktisk policyøkonomi
etter neste VAL før enda en epoch. TEST forblir forseglet.

Bevis: handover_snapshot/ENTRY_REPLAY_CONTINUATION_REVIEW_20260916.json,
ENTRY_REPLAY_CONTINUATION_PAIRED_RESULT_20260916.json og native receipt.
Operatører/logger/resultater er bevart under NATIVE_ENTRY_REPLAY_CONT_B493E585.
RUNNING_NATIVE_CALIBRATION.json inneholder full binding og status.

# Gjeldende neste arbeid — fortsett samme læringskontroll fra99

Ny avgrenset CPU-observasjon av beholdt checkpoint98 er ferdig, session1529/exit0,
7.9742s. Samme64cachedTRAIN-inputs og samme diagnostiske dropout. Entry-MSE
95:19.576542 →98:17.918624 →99:17.321052; alle64 fortsattFLAT. Siste64 oppdateringer
reduserte feilen0.597572. Dette støtter videre avgrenset innlæring; ingen påvist
konvergens/platå. Ingen tidligere forwards/evalueringer ble gjentatt. Operatør og
logg er bevart under den avsluttede64-kontrollen; resultat i handover_snapshot.

Hypotesen om L1-median kontra forventningsverdi er ikke bekreftet som årsak.
Ikke endre forecast-tap, gradientgrenser eller modell nå. Fortsett nøyaktig samme
kombinerte læring fra99/global6033 til høyst7057:1024 ekstra optimizersteg på
samme64innganger,256 videre repetisjoner. Frossen lærer91, original epoch_order,
alle checkpoints og modell-/optimizer-/EMA-/RNG-tilstand ved overgang bevares.
Dette er ingen ny årsdekning; diagnostiske checkpoints er ikke produksjonsresume.

236 målrettede tester består. Bare tre eksisterende eiere for kjøregrense,
loader/resume og EMA-historikk er justert. AST-verifikasjon viser seks numeriske
forward/loss/backward/optimizer-eiere uendret. Ingen refaktorering eller ny modell.
Neste: commit/push, én faktiskCPU-overføring av99, deretter én native guarded
kjøring fra frisk fysisk boot. Ingen aktiv tung jobb ved denne commit. Gjenbruk
cachet99-resultat for paret99→115 etter kjøringen; hele epocher/VAL/TEST sperret.
Se RUNNING_NATIVE_CALIBRATION.json og NEXT_RUN_POLICY.json.

# Gjeldende status — lærbarhetskontroll ferdig, 2026-09-15

Frossen målekilde var GX1_CURRENT/work/gx1-current, commit6c9328d6.
Én native kontroll gjentok de samme64 faktiske TRAIN-inngangene64ganger med
uendret kombinert trening og fast lærer91. Checkpoint95/global5777 ble bevart;
privat checkpoint99/global6033/epoch1offset1952 er diagnostisk og skal ikke
fortsette produksjonstrening. Original epoch_order er uendret; ingen ny årsdekning.

Native kjøring tok1199.510192s inkludert oppstart/observer, uten fysisk reboot.
GuardPASS, trainer0, observer0, Windows-task Disabled21:17:36UTC; prosessene
764/709/658 er terminale. Planen er uttømt. Ingen tung jobb er aktiv.
216 målrettede tester og normal commit-hook bestod før kjøringen. Faktisk
CPU-overføring bevarte alle15 komponenter. Alle726 Adamtilstander økte256;
EMA25685→25941, historikkoffset19908 og lærer91 beholdt. Dette er ikke en ny
kontinuerlig-mot-delt-resume-verifikasjon; tidligere relevante bevis gjenbrukes.

Paret CPU-evaluering95→99 er ferdig (session57439/exit0,41.761691s,
peak2266684KiB). Fire opprinnelige95-referanser ble gjenskapt bitidentisk;
begge modeller brukte samme cached inputs/targets og diagnostiske dropoutfrø.
Ingen ny korpus-/lærerberegning, optimizer, backward, GPU, VAL eller TEST i evalueringen.

Målt på det gjentatte TRAIN-utvalget:
- Exit-MSE6.888519→0.342115: ned95.0335%. HOLD/EXIT stemmer med lærerens
  unike preferanse i88.6719% LONG- og90.2344% SHORT-tilfeller.
- Forecast-retning ved nominelle5/25/60/120min:56.25/60.94/54.69/54.69%
  →67.19/70.31/84.38/89.06%. Dette er L1-returanslag, ikke kalibrerte sannsynligheter.
- Entry-Q-MSE19.576542→17.321052: ned11.5214%, men fortsatt64FLAT mot
  lærer55FLAT/3LONG/6SHORT. Ingen av de ni lærerforetrukne handlene velges.

Dette viser lokal læring i Exit og markedsprognosene; Entry-handlingenes
økonomiske skille henger etter. Ingen generalisering eller lønnsomhet er bevist.
Entry-mikser/head endrer vekter, LR er9.890738e-5 og Entry-tapsvekten er
2.02215→1.95647, altså ikke slått av. Koden stopper fortsatt Entry-Q-gradienten
før markedsrepresentasjonen. Forecast lærer M5-close-retur over5–120min;
Entry-verdien inkluderer første M1-lukking pluss svak videreverdi fra lærer91.
Ulike fortegn mellom disse målene beviser ikke tidsfeil.30av256 observerte
Bellman-overganger er state0; ikke påstå at første tilstand aldri trenes.
Ingen lagret native tapskurve finnes i95/98/99; ikke påstå platå/konvergens.

Neste prioritet er å isolere Entry-representasjonens læring av økonomimålet,
med selvstendig markedssignal og fortsatt skjerming mot Exit-bootstrap.
Avgrens én presis target-/gradienthypotese før ny kodeendring; gjenbruk denne
kontrollen og eksisterende native vakter. Ikke innfør terskler/klassevekter,
bytt modell, gjenta fullførte prober eller start en hel epoch på dette grunnlaget.
Ingen ny produksjonsendring eller neste treningsplan er bestemt.

Bevis: handover_snapshot/ENTRY_LEARNABILITY_CONTROL_REVIEW_20260915.json,
ENTRY_LEARNABILITY_PAIRED_RESULT_20260915.json og tilhørende native receipt,
checkpoint-/forecastgjennomgang. Operatører og logger er bevart under
NATIVE_ENTRY_LEARNABILITY_6C9328D6. RUNNING_NATIVE_CALIBRATION.json er operativ status.

## Avsluttet FQI-kontroll og korrekt TRAIN-måling, 2026-09-15

Eneste kilde er /home/andre2/src/GX1_CURRENT, work/gx1-current. Frossen kilde for
fullførte målinger var f11c1dbeef12721bef58449dfac37fd735f4694f. Checkpoint95 har
global5777, epoch_index1/offset1696. Native256 steg tok1194.683976s inkludert
oppstart/observer, uten fysisk reboot. GuardPASS, trainer0, observer0. Windows-
task er Disabled; ingen aktiv tung jobb. Planen er brukt opp. Ingen ny lærer-
oppdatering eller hel epoch starter automatisk. TEST forblir forseglet.

169 målrettede regresjoner og vanlig commit-hook bestod. Faktisk CPU-overføring
kopierte target nøyaktig fra ONLINE91 og bevarte alle14 øvrige komponenter.
Native beholdt denne targeten gjennom256 steg. Alle726 Adamtilstander økte256;
EMA25429→25685, offset19908 beholdt. Ingen ny kontinuerlig-mot-delt-resume-påstand.
Gammel lærer og alle resultater/checkpoints er bevart. Kun target ble endret;
økonomiske formler, MSE, optimizer, modell og risiko er uendret. Dette er én
kontrollert læreroppdatering, ingen innført permanent «hver256»-regel.

## Resultat på faktiske native TRAIN-batcher

Korrigert paret CPU/no_grad-måling er ferdig, session3258/exit0,419.149976s,
peak9291828KiB innen eksisterende20GiB/512MiB-cap. Fire forhåndsvalgte faktiske
batcher fra kontrollen: epoch1offset1440/1525/1610/1695.64entries overalle12måneder,
4–7 per måned, og256 gyldige HOLD-celler per side. Lagret epoch_order og ordnet
materialisert sampleplan matcher nøyaktig. Samme nye lærer91, inputs, targets
og diagnostiske dropoutfrø før91/etter95. Ingen historisk GPU-dropoutreplay-påstand.
Alle source/checkpoints/inputs/targets/policy/RNG er bevart. Ingen nytrening,
backward, optimizer, VAL eller TEST ble brukt til målingen.

HOLD-MSE LONG14.09770→13.96113, nullbaseline14.10036; SHORT13.87834→13.59294,
nullbaseline13.51526. Samlet MSE falt1.5083%, men er bare0.2229% bedre enn null-
baseline. LONG-HOLD25→199/256; SHORT-HOLD256→205/256. Dette er forbedret Exit-fit
på utvalget, ikke bevist forventningsverdi, generalisering eller lønnsomhet.
Entry-Q-MSE19.43136→19.57654; studenten64FLAT begge, læreren55FLAT/3LONG/6SHORT.
Prognose-MAE bedres svakt på nominelle5/25/120min og forverres på60min; retning
bedres bare på5min. Prognoser er L1-returanslag før kostnader, ingen kalibrerte
sannsynligheter. De64 radene ble sett i den avsluttede treningskontrollen.
Ingen matchet kontroll med uendret lærer fra91 ble kjørt, så årsakseffekten av
selve læreroppdateringen er ikke isolert.

Se handover_snapshot/ACTUAL_WINDOW_PAIRED_TRAIN_RESULT_20260915.json og
ACTUAL_WINDOW_PAIRED_TRAIN_REVIEW_20260915.json. Råresultat-SHA:
aab5bacd4a5ba4ba4d885150073e290d43cdfa50d935705add22633a2f1decfa.
Originaler, fire input/targetcacher, batchrapporter, operator og logg ligger i
NATIVE_FQI_TARGET_REFRESH_F11C1DBE/OPERATOR_OBSERVATIONS/
ACTUAL_WINDOW_PAIRED_TRAIN_CPU_20260915_V2 under bundet prebuilt-rot.

Første operatorforsøk92416/exit1 feilet før første modellforward: det krevde
sample.epoch_index==1, mens full TRAIN-epoch1 inneholder interne sampler-chunker
25/26/27 for disse batchene. Første betingelse kortsluttet; ingen avvikende
rekkefølge var målt. Kun operatorens epoch-kontroll ble rettet. Full epoch og
chunk-indekser valideres separat; ordnet sample-digest er uendret. Feilet operator,
logg og COHORT er bevart i første output uten _V2. Ingen produksjonsrettelse.

## Målegrunnlag og neste arbeid

Native TRAIN er deterministisk stokket over hele året, ikke kalenderblokker.
Gjenskapt faktisk epoch_order matcher lagret hash84813100...fbfa65. Begge256-
kontrollene brukte4096entries overalle12måneder (henholdsvis314–367 og313–369
per måned). Ingen shuffle-rettelse er begrunnet.

De64 tidligere proberadene var fire sammenhengende75min entryblokker med eldre
sampler-transitions. Bare4 og5 av entry-IDene forekom i de to kontrollene. Den
små MSE-forverringen i gammel probe var reell på det utvalget, men må ikke omtales
som bevist forverring av hele modellen. Den nye målingen erstatter den gamle som
TRAIN-fit-grunnlag; begge bevares. Ingen av dem er en heldout-kvalitetsgate.
Tidligere target-preflight20.39s og etteranalyse24.63s skal ikke gjentas. Se
FQI_TARGET_REFRESH_CONTROL_RESULT_20260915.json, FQI_TARGET_PREFLIGHT_RESULT_20260915.json
og FQI_TARGET_LEARNING_RESULT_20260915.json i handover_snapshot.

Gjennomgangen av de ni lærer-valgte inngangene er ferdig på lagrede JSON-er.
FLAT ligger rundt0; å sette FLAT eksakt0 ville fortsatt gitt64FLAT. Alle online
LONG/SHORT-verdier er negative. På de ni lærer-valgte sidene er targets+0.481 til
+18.097Bps, etterprediksjon−6.651 til−4.912. Ni rader står allerede for87.69% av
EntryQ-kvadratfeilen; dette begrunner ikke automatisk klassevekting/oversampling.
Side-halvsummen ligger nær lærernivået (−5.708 vs−5.736), mens retningens
halvforskjell har std0.224 mot target5.406. Studenten lærer hovedsakelig det
felles negative nivået på dette utvalget, med lite betinget variasjon. Riktig
LONG-vs-SHORT-rangering på de ni er4/9 etter mot6/9 før. Dette skiller ikke alene
manglende nyttig informasjon fra utilstrekkelig læring eller svake targets.

Eksakt dekomponering fra allerede lagrede input/targettensorer er ferdig,
session36519/exit0,1.207s, CPU4GiB/512MiB. Ingen modell/forward/optimizer eller
nytt markedskorpus. For alle64 er Entry-target eksakt første lukkingsverdi pluss
max gyldig anchor-Q fra lærer91. Alle ni lærer-trader har positiv fysisk første
lukkingsverdi: gjennomsnitt6.431627Bps. Lærerens videre HOLD-bidrag er bare
0.028161Bps i snitt (0–0.045069), mot total6.459788. Fortsettelsesbidraget er
0.43594% av positiv målverdi i disse ni. Dette viser hvor signalet i prøven kommer
fra; det beviser ikke at disse observerte prisbevegelsene kunne forutsies ved
entry eller at læreren er en optimal profittfasit. Se
ACTUAL_WINDOW_ENTRY_GAP_REVIEW_20260915.json og
ACTUAL_WINDOW_ENTRY_ANCHOR_DECOMPOSITION_20260915.json i handover_snapshot.

Neste anbefalte arbeid er en avgrenset lærbarhetskontroll av eksisterende EntryQ
på de samme64 bundne native TRAIN-inputene, fast lærer, uendret MSE, og alle55FLAT-
rader beholdt. Formålet er å skille lokal tilpasningsevne fra svak betinget
informasjon/lærertarget. En eventuell optimizerkontroll må først gis en eksplisitt
avgrenset native plan med eksisterende vakter og bevarte checkpoints; ingen
alternativ treningsvei eller restart av uttømt plan. Ingen produksjonsrettelse,
klassevekter, handelsterskler, ny modell eller hel epoch er besluttet. Evne til å
tilpasse et lite TRAIN-utvalg er heller ingen generaliserings-/profittgate.
Gjenbruk alle fullførte målinger, særlig anchor-dekomponeringen; ikke gjenta dem.
Alle jobber er terminale. Sourcefrys er opphevet kun for ferdig arbeid/commit.
RUNNING_NATIVE_CALIBRATION.json er den operative statusen.

## Historikk: fixed-target-kontrollen87→91

Eneste kodebase er /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Den avsluttede kontrollens frosne kilde er da7ae15d35e4a898656a2c1e6f9519f84b3e9ed5.
Checkpoint91 har global5521, epoch_index1, next_batch_offset1440. Original87
er bevart. Ingen aktiv tung jobb; Windows-task er Disabled. NEXT_RUN_POLICY.json
har training_enabled=false. Den eneste tillatte256-stegskontrollen er brukt opp.
Ingen restart, hel epoch eller automatisk læreroppdatering er tillatt fra den planen.

Native256 steg tok1197.100656sekunder inkludert oppstart og observer, uten fysisk
reboot. Guard PASS, trainer0, observer0. Alle726 Adamtilstander økte256 steg;
EMA25173→25429. Frossen lærer, scheduler og datarekkefølge er bevart. Faktisk
CPU-overføring bevarte alle15 tilstandskomponenter. CPU- og native-kontraktene
avviker bare i run/output/recipe-identitet. Dette er faktisk checkpointoverføring,
ikke en ny sammenhengende-mot-delt-resume-likhetsmåling eller relativ speedup.
De143 målrettede regresjonene gjenbrukes; ingen ny fullsuite.

Paret CPU-evaluering er ferdig på38.254272sekunder, peak2279356KiB, exit0.
Samme64 TRAIN-rader og256 HOLD-celler per side, identiske targets/dropoutfrø;
alle fire checkpoint87-referanser reproduseres eksakt. Ingen autograd, optimizer,
GPU, VAL eller TEST. Endelig operatørskript/logg og en tidligere bindingsfeil
før modelleksekvering er bevart. Se:
- handover_snapshot/FIXED_TARGET_FIT_CONTROL_RESULT_20260915.json
- handover_snapshot/FIXED_TARGET_LEARNING_RESULT_20260915.json
- RUNNING_NATIVE_CALIBRATION.json

## Læringsresultat og neste konkrete justering

HOLD-MSE LONG275.713→274.338, nullbaseline274.046; SHORT268.292→266.439,
nullbaseline266.666. MAE forverres på begge sider. SHORT HOLD212→256 av256;
LONG HOLD89→30. Entry er fortsatt64FLAT mot lærer58FLAT/3LONG/3SHORT.
Dette viser endret tilpasning, ikke dokumentert selektivitet eller lønnsomhet.
Realiserte ettstegsfortegn er ingen feilfri fasit for forventningsverdi. Verken
MAE eller handlingssamsvar alene avgjør om en økonomisk policy er god.

Sekvensinput er allerede til stede. Den konkrete begrensningen i lærersløyfen
er én successor-backup mot frosne verdier, med ny ONLINE-lærer først etter full
epoch og VAL (entry_v10_ctx_train_v3.py, native epoch-overgang). Flere pass mot
samme lærer gjør ikke denne verdipropageringen raskere. Tidligere to-stegsprobe
og gradientkontroll er ferdige og skal ikke gjentas.

Neste anbefalte rettelse er én eksplisitt SHA-bundet FQI-læreroppdatering fra
bevart ONLINE91, frigjort fra full epoch/VAL, før videre avgrenset læringskontroll.
Dette er en kontrollert læringshypotese, ikke godkjenning av lærerens kvalitet.
Bevar gammel lærer og alle øvrige tilstander; kvitter bare tilsiktet target-kopi.
Verifiser smal overgang/resume før eventuell ny native kontroll. Ingen permanent
«hver256»-regel er besluttet. Ingen endring i MSE, økonomi, EMA, Adam, arkitektur,
risiko eller holdetid begrunnes av disse resultatene. Ikke klipp legitime store
utfall for å pynte på fit. Nye hele epocher forblir blokkert.

TRAIN er fortsatt ett år, alle200features/åtte familier/tidsrammer beholdes,
og TEST er forseglet. Nativeprofil TRAIN16/VAL256/8arbeidere/3timersVAL og vakter
beholdes. Historikken nedenfor gjelder avsluttede kontroller, ikke gjeldende
oppstartstillatelse. Nye oppstartsplaner må bindes til gjeldende kilde og policy.

## Historikk: GX1 — stoppet; native kontroll bestått, læring fortsatt utilstrekkelig, 2026-09-15

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
