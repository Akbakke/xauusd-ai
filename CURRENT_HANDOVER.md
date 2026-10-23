# CURRENT HANDOVER — XAUUSD M5 Entry

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current` på
`gx1-3090-lan`. GX1_RULES er bindende; NEXT_RUN_POLICY eier arbeidsstatus.
Prosesser og terminalkvitteringer overstyrer prosa. Én integrator, én capped
CURRENT-jobb og fryst kilde/policy gjennom kjøring. Originalmålet står i
[docs/ORIGINAL_PROJECT_GOAL.md](docs/ORIGINAL_PROJECT_GOAL.md).

## Mål og aktuelt neste steg

Bedre M5 Entry LONG/SHORT/FLAT med MBP-1 og lukket M15/H1/H4/D1:
retning, størrelse, liten MAE før første MFE, kostnad, kvalitet og dekning.
Solid featuregrunnmur og generell robusthet prioriteres. Ingen favorittmåned,
minuttjustering, HPO eller automatisk inversjon. Exit venter.

198 er COMPLETE/consumed. Fysisk snapshotpublisering er kvalifisert:
118249 TRAIN +17284 Q1CONTROL =135533 parentrader,3345 kolonner.
Egne observerte 1/5/15-masker: TRAIN118237/117799/116936 og
CONTROL17283/17219/17089. Alle parentrader beholdes; strictcutoffs01.01/01.04.2026.
På eksisterende512 modellrader er parent/klokke/M1pointer, terminalmasker,
observerte F32-sideverdier og F64-kontrast eksakt bevart.

198worker150,534s/ytre151,113s, peakRSS3256209408bytes, audit4GiB/512MiB/
éntråd/pids64/CPU0–7.15cache-NPY-filer ble ferskverifisert; tidligere fullmatrix-
liveness/squeeze/proveniens fra055/053 gjenbrukes. Dette er ikke ny
native-loader-ekvivalens. To originalOBS-lesinger,27,505GB rchar;
rchar er ikke samlet mmap/disk-I/O.239 inputstatidentiteter og kilde/policy var
uendret. HEAD8926dfe4/policyad8dc278 var fryst. Canonicalprosjektlås er fri.
Ingen referansefit, native forward eller optimizeroppdatering ble utført.

198TERMINALd098166b…/RESULT7edcdcfd…/METADATAf94356b3…/ytre7e793c16…,
ACTUAL_PHYSICAL_PEER_REVIEW og DECISION ligger under
`/home/andre2/GX1_RUNS/NATIVE_M5_SNAPSHOT_REFERENCE_LAYOUT_REPAIR_20261023_198`.
Seks kvalifiserte OUTPUT-filer skal gjenbrukes; originaldata skal ikke bygges om.
194s27PASS og196s29PASS gjenbrukes.195/197 er failed/consumed med uendrede
originalkvitteringer og partialoutputs; ingen relansering eller opprydding.
Feilene var testfixture-path, helperens NumPy-binding og tuple/JSON-listelikhet;
minste rettelser er kontrollert, siste feil fanges nå før store datalesinger.

199 er FAILED/consumed før fit,0,715s/0fits/0nativeF/U. En for streng C-only
NPZ-vakt avviste lovlig F-order i fire reference191-array (contrast og maske).
Header-only diagnose777ce698… og uendrede41inputidentiteter/kilde er bevart.
Ingen ny retningsmåling ble utført;198snapshotene er fortsatt kvalifisert.

200 har én egen eksakt admission etter minimal kildepeer uten blokkere.
`NATIVE_M5_FIXED_SNAPSHOT_DIRECTION_REFERENCE_REPAIR_20261023_200` inneholder
PLAN/SOURCE_REPAIR_REVIEW. Workeree7ceb0f… tillater sammenhengende C eller F
for NPZ; ingen transponering, verdi- eller aritmetikkendring. Snapshot-mmap
forblir C-order; alle dtype/shape/finite/digest/bit/mask/klokkevakter beholdes.
Samme3TRAIN-only ridge1/5/15,alpha1SUM-SSE/friintercept/055fryst,3TRAINmiddel/null.
Normal-equation-residual≤1e−8 ved positiv RHS, ellers eksakt nullrest, og finite
F64-tilstand/prognoser kreves; det beviser ikke velkondisjonering eller retning.
Audit4GiB/512MiB/éntråd/pids64/1800s+10TERM/8GiB rchar; faktisk fitkapasitet
og retningsresultat er umålt. Bare seks kvalifiserte198-filer, ingen originaldata,
nativeF/U, HPO eller nye toyfits.194/196 og199metode-/evalreview gjenbrukes.
Top training_enabled=true gjelder bare eksakt200fit; full_epoch/fullVAL/
native_launch=false. Kilde/policy fryses; etter terminal stenges admission og
hele resultatet tolkes før konkret native forbedring velges.

Samlet snapshot inneholder alle aktive kanoniske felt og femTF:3317koordinater
blir3345kolonner når seks nominelle SMC-tilstander erstattes av6×5one-hot og
session får4kategorier. Aliasene telles én gang. Positive referanseresultater
viser samlet snapshotinformasjon inklusive klokker/masker; negative avviser
ikke sekvensinformasjon eller nonlinear læring. Ingen familierangerings-,
featureutvalgs- eller nativearkitekturendring følger automatisk.

## Siste målte retning og hvorfor begge sider kan være negative

191 er COMPLETE/consumed: én TRAIN-passering,118249unike rader/7391oppdateringer,
B16/siste9/drop_lastfalse. Sluttmåling16+16ONLINEforwards på256TRAIN+256CONTROL,
full480. Canonical guardPASS/exit0 og planlagt RESUMABLE ved7391; ingen ferdig
trainerbundle eller ekstraepoke.191 startet fra187fresh, egen188initial og
189savedstate-audit etter190privat mekanikkvalifikasjon.184s eneste designendring
var separat MFE-gruppemiddel for1–15 og16–480. Nativeeiere er siden bevart.

192 er siste målte retningsanalyse,0nyeF/U/fit. Alle512handlinger er FLAT.
CONTROL nettkontrast-MSE i bps² på identiske rader/masker:

| Minutt | Egen188initial | Matchet177 | Ny191 |
|---|---:|---:|---:|
|1|57,459|61,678|58,108|
|5|228,358|227,369|225,698|
|15|772,101|774,629|774,889|

Fem av seks TRAIN/CONTROL-paneler er verre enn188; fire av seks bedre enn177.
Alle12par har dag-/ukeintervaller som krysser0. CONTROL fortegnstreff45,3/49,6/46,0%,
Pearson0,098/0,136/0,018 og bias−1,37/−1,33/−2,31bps. Predstd1,40/1,41/1,42
mot fasit7,50/15,09/27,73bps beviser ikke underfit. Ingen retningsgevinst er påvist.
Tidligpath-punktfeil er forbedret mot188, men ikke konsekvent mot177; dette er
anslagsfeil, ikke mottattMFE eller mindreMAE i handler.

186s historiske167-signaler:58TRAIN original−6,983bps/motsatt−5,115;
154Q1CONTROL original−6,091/motsatt−6,263. Ukeintervall for inversjonsforskjell
krysser0 i begge perioder. CONTROLpreMAE øker7,361→7,587 og MFE faller3,185→3,002.
Ingen generell snuregel. Bid/ask-spread ligger allerede i quoteavkastningen;
ytterligere4bps er to slippageledd à2bps,commission0. Quote-til-fill-truth er
ukjent. Scoped kildeavstemming fant ingen fortegns- eller dobbelavgiftsfeil.
Felles4bps kanselleres i retningskontrasten; de forklarer ikke svak retning alene.
PreMAE-straff er utilityproxy, ingen cashfee. Prisbanemål er ikke realisertP&L.

## Featuregrunnmur og grenser

Aktiv profil `compact_mbp_smc_ema_atr_clock_macro_entry_v1`:7lokale/5MTF-familier
etter19Oct-unntaket; åtte spesialister og9.D1/makro beholdes.175 kvalifiserte
695aktiveMTF-felt139×5 og aktiveGC-verdier/masker med0F32avvik. SMC,EMA/trend,
swing,støtte/motstand,volumprofil ogMBP-1 er bevart kausalt. Manglende data er
ukjente. MBP-1 er aggregertBBO/ordretall/OFI, ikke ordrekø eller skjultdybde.
180 målte lokal respons i alle32aktiveav48deklarerte familie/TF-laner og
alle8flytflater,16genuineTRAIN/1F/5VJP/0U og84640paritetsceller delta0.
Dette viser konsum, ikke optimal featurebruk eller kronologisk merverdi.

TRAIN[2024-05-01,2026-01-01), Q1CONTROL[2026-01-01,2026-04-01) er23gjenbrukte
utviklingsmåneder. April–juni venter fysisk binding; råkildemålet er26måneder.
WholeTRAIN055 og TRAIN-fittede konstanter tillater ikke internTRAIN-OOF.
199 skal vise alle måneder/kvartaler/ALL og alle1/5/15,2000UTCdag-/mandagsuke-
bootstrap med samme kalendervekter/egne masker; sparse<8blokker merkes.
Direkte191-sammenligning gjelder bare256+256lagrede rader,ingen nyeforwards.
Ingen av disse dataene er urørt generalisering.

M5 er beslutningstidspunkt;1–480min er prognosedomene, aldri fasttap eller
maksimal holdetid. Lang støtte sammenlignes med alltidLONG/kjøp-og-hold.
TEST fra juli er forseglet for tilpasning/utvalg; bare vedtatt månedlig forwardjournal
scorer uten refit. Ingen Exit/broker/live/paper/ordre/spending/promotion.
Originalkvitteringer/checkpoints bevares; DATA/RUNS-rydding krever retention-eier.
Historikk ligger i Git. Metoden er [docs/NATIVE_LEARNING.md](docs/NATIVE_LEARNING.md).
