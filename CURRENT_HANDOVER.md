# CURRENT HANDOVER — XAUUSD, retning først

Autoritet: `/home/andre2/src/GX1_CURRENT`, branch `work/gx1-current`,
`gx1-3090-lan`. Les GX1_RULES helt ved første arbeid; start overtakelse med
`bash scripts/gx1_handover.sh --check`. NEXT_RUN_POLICY er eneste arbeidsstatus.
Root eier integrasjon, policy og launch. Prosess/terminal overstyrer prosa.
Historikken finnes i Git og originale eksterne receipts; åpne bare ved konkret
avvik eller nødvendig ny gate. Gjenbruk beståtte kontroller og ferdige inputs.

## Mål og aktuell status

Bruker spisset24.10 til **retning først**: etterprøvbare OPP/NED-prognoser.
Trades, bevegelsesstørrelse/MFE/MAE, risiko og Exit utsettes. Målet er aktivt og
ikke oppnådd. TRAIN-fit, senere overføring, generalisering og økonomisk edge
krever forskjellige bevis. Teknisk PASS er ikke gode prognoser.

256 er COMPLETE/consumed og **forkastet som retningskandidat**. To uavhengige
actualmetadata/stat-peers er PASS. 258design og259kilde har tre peer-PASS.
259 er nå innrømmet for nøyaktig én CPU-kjøring: samme239R1, tre eksplisitte
felt og15 genuine fits. Root eier policy/PLAN/launch. Worker ogPLAN er fryst;
alle kilder og policy forblir uendret til terminal. Global training=true gjelder
bare denne CPU-testen. Native/full_epoch/fullVAL/CONTROL/TEST/Exit/trades er lukket.
Ingen gammel plan relanseres. Målt prognosegevinst for259 foreligger ikke ennå.

## Siste faktiske retningsmåling

256 tester én fast ExtraTrees med alle3345 snapshotfelt + tidligere ret_1 ved
**eksakt D−300s** + qualified_parent_present. Samme19parametre som252:
32trær/dybde8/minleaf256/sqrt/éntråd/seed20260911. Samme fem kronologiske
TRAIN-prefixer,245renprisfasiter og frosneR1/252prognoser. Ingen norm-/kontroll-
refit, klassevekter, HPO, terskelvalg eller kalibreringsfit.

| Horisont | TRAIN-fit, fem prefixer | Senere OOF | Lagret252 | AlltidOPP | Balansert OOF |
|---|---:|---:|---:|---:|---:|
|1min|55.89–56.84%|50.4370%|50.4972%|50.3099%|50.4703%|
|5min|54.50–57.30%|50.7732%|50.7493%|51.0086%|50.1549%|
|15min|55.81–60.32%|51.4609%|51.6819%|51.6899%|50.5289%|

Alle15TRAIN-paneler forbedrer logloss og Brier mot egen modneprefixprior.
Alle36pooled dag/uke95%-intervaller motprior/R1/252 krysser0. Fellesnegative
folds overalle6properloss-sammenligninger er1/0/1, underkravet4. Samlet proper-
pointdiff er verre enn252 påalle3h; bare5min bedrerprior ogbare1min bedrerR1.
Hardaccuracy feiler motalltidOPP5/15. Alle3followups ogsamletfollowup er false;
ingen kontroll blir automatisk vinner. Ingenp=0/1/.5 ellerloglossklipp i33paneler.

En konkret konsumgrense er nå målt: tidligere ret_1 fikk **7splitter i5av15
modeller**;10bruktefeltet aldri. Presence fikk0splitter. Kandidatpakken er
forkastet, men dette er ingen generell dom over temporal informasjon/native.
Inputbredde og tilfeldige featurevalg endres sammen; splitbruk beviser konsum,
ikke kausal prediktiv merverdi. TRAIN→senere-gapet består; støytilpasning,
ustabile relasjoner og representasjonsgrenser er ikke skilt fra hverandre.

## Hva som faktisk er kvalifisert

Globalt eksakt lag finnes for117799/118249foreldre.450ukjente består av1før
bundetdomene og449uteneksaktparent;213genuine rånuller beholdes medpresence.
Fravær er ingen kvalifisert historiskparent, ikke manglende markedsbar. Ingen
radfiltrering/row−1/nearest/backfill. Faktisk051første1minuttsemisjon betyr at
historiskparent vedD−5min er kjent innenD−4min;253binder dette til kilde/data.

Globalproof kom før0fits; femfoldproofs før0/3/6/9/12fits. Alle3345feltkart,
rekkefølge,210prefixnormaliseringer, kol1F32-hashes og15maturity-/fitmasker
matcher252. Firepayloads ble ferskSHA-verifisert. Beggekontrollers18fold/pooled
scores og14numeriske kontrollarrayer ble avstemt førfit. Alleforeldre beholdes.

Utføring:391,921sworker/393,560sytre, fryst e4664b19/policy, beggeexit0 og
sourceunchanged.7score/datagrupper +separat syntetiskforest varPASS. Genuine
15forestfits/480treefits/30populasjoner/960treepredict;synthetic1/32/2/64.
Alle480/32treeinnganger har F32/C/shared-memory-witness.413667encoderforeldre,
0norm-/R1-/252refit/nativeF/U/CONTROL/TEST/quote/originalOBS/forbidden.

Faktisk cap4GiB/512MiB/64pids/énnumerisktråd og prosjektlås er kvittert.
Prosess-RSSpeak2348969984B, rchar1843573020B, scratch1582171620B. RSS er ikke
cgrouppeak; endeligcgrouppeak ble ikke fanget.15egne modeller har verifisert
pickle-state/readonlystat uten ekstragenuineinferens;4897482Bmodeller og
17773490BOOF bevares utenforGit. Actualpeers avstemte33paneler,69kontraster/
276CI,15modell-/480treemetadata og alle kohorter uten payload-/modellkropper
eller numeriskreplay. Målekonsistens er PASS; forecastverdi er ikke kvalifisert.

255 er bevart som consumed prefit-feil: vår wrapper krevde siste beholdte rad
lik konfigurert kandidat-slutt21:55; faktiskbundet slutt er21:50. Ingen genuine
fit/encode skjedde i255.256 retter bare inkluderende intervallgrenser ognye
run/stage/feilnavn; alle3eksakte parent-/klokkehashes beholdes.255 er ikke relansert.

Eksakte terminal/resultat/actualpeer/decisionrefs står i policy256 og:
`/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREVIOUS_RETURN_BOUND_REPAIR_20261024_256`.
Originale receipts/checkpoints/forskningsmodeller bevares; ingen promotion.

## Neste konkrete spørsmål —259kildepreparasjon, ingenfitadmisjon

En liten direkte3kolonne-referanse kan måle additivt historiesignal med samme
faste239R1-eier: [nåværende ret_1, eksaktD−300s ret_1, qualified_parent_present].
Egne koeffisienter og logitbidrag gjør faktisk bruk målbar; dette lover ingen
prognosegevinst. Kilder239/245/252/256 er allerede bundet. Fem eksisterende245
F32[stop,66]-matriser har col65hash lik252/256col1 ogsamme210stats. Ingen3345-
reencoding/raw206/nye normaliseringer er nødvendig; nativefeatures beholdes.

Prospektivt8payloads144496754B,15kandidatfits/30prediksjonspopulasjoner,
F64trekolonnepopulasjon maksimalt2837976B. EksisterendeL2/C1/lbfgs/max500/
tol1e−4/seed20260911, ingenklassevekter ellerparameterjakt. Ressursanslag er
umålt og ingenadmisjon. Egen257RESULT binder eksakte refs/kildekonfigurasjon.
258 låser nå full input-/klokke-/kontroll-/score-/konsumbudsjettkontrakt:
15fits/30populasjoner, samme3referanser/36CI/4fellesnegativefolds ogalle15
mekaniskekonsumkrav.258designreview og259kildekvalifikasjon erPASS; én259CPU-kjøring er innrømmet.
PLAN/worker/SOURCE_PREPARATION_REVIEW er bundet i policy. Sju syntetiske grupper
og faktiske databevis/fit/ressurser/prognoser må bestå i selve kjøringen. Ingen prefixkontroll-
rekonstruksjon eller naivchi²-likelihoodtest på regularisert/tidsavhengig/gjenbrukt
utvikling. Samme-rad senere properloss ogdag/ukeusikkerhet er avgjørende.

## Datagrunnlag og varige grenser

245 bruker F64-midtpris vedD=bar_start+300s ogT=D+1/5/15min fraeksakteM1-open.
Full9-array198kostnads-/klokkeparitet er bit-eksakt; originale kostnader/targets
bevares.118249TRAINforeldre/88400OOF, binære modnerader88102/87944/87344.
Nullbevegelser/manglende endpoints/umodne rader beholdes i nevnerne. Fit:
observed & sign!=0 & D<cutoff & terminal<cutoff; score terminal<egetholdoutend.
OPP er betinget på observert ikke-null-bevegelse, ingen ubetingetFLAT-klasse.

Femcutoffs2024-10-01/2025-01-01/2025-04-01/2025-07-01/2025-10-01, enderneste
kvartal/siste2026-01-01. Prefix/holdout29849/17753,47602/17288,64890/17537,
82427/18072,100499/17750. Bootstrap2000/seed20260911/min8 er gjenbruktutvikling,
nominale eksplorative intervaller, ikke urørt generalisering eller økonomi.

Snapshotet har historiske sammendrag; native harordnet96M5 og16/64/96/96/252
TF-historikk, men annen fleroppgaveregresjon enn renprisretningen245. Aktiv3345
harret_1/EMA-/hendelsesaldre/CVD/profiler, ikke lokalret_20/roc100. Parkertchart
utelukker ikke17chart-navngitte aktive skalarefelt. NativeTransformer, åtte
spesialister/niendeD1makro, genuinefeatures/TF og055/057/111 beholdes.19.10-
kompaktprofilhar7aktivefamilier;chart er eksplisittparkert. Konsum er ikke verdi.

Ingen native/ONLINE-endring:223inputs/227fresh/229initial/230stateaudit/
231privatequalification/232epoke/233analyse bevares.232 gav ingen stabil retning;
alle12CONTROL-directionCI krysset0 ogalle512actions varFLAT. ONLINE-endring
krever fersk initialbaseline. Langstøtte er ingen Exit-/maksimalholderegel.

Fysiskgrunnlaget er23gjenbrukteutviklingsmåneder:TRAIN mai2024–desember2025,
CONTROL januar–mars2026.26måneder er målet; april–juni er ikke fysiskbundet.
Juni er gjenbruktutviklingsVAL. TEST frajuli er forseglet for alltilpasning/utvalg;
bare vedtattmånedligforwardjournal scorer utenrefit. IngenExit/live/paper/broker/
ordre/spending/promotion. Økonomi omfatter åpneposisjoner ogallevalgthandler.
Langhorisontreferanse er alltidLONG/kjøp-og-hold.

Én tungCURRENT-jobb via gx1_capped_run.sh med lås/hardwarevakter ogfryst kilde/
policy tilterminal. Stabilkjøring kontrolleres15–30min ellersjeldnere, ingen
minuttmodellpolling/fyllarbeid. IngenDATA/RUNS-rydding utenretention-eier.
Oppdater policy/handover i sammebølge; fokuserte kontroller/AST/stale-path/
gitdiff--check og ærlige grenser. Pushrett tilwork/gx1-current gjelder kode/docs/
interne refs/aggregater, aldri rådata/vekter/hemmeligheter ellerforcepush.
