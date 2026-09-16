# GX1 — gjeldende overlevering, 2026-09-16

Forsøket er nå **frosset før kronologisk evaluering**. Eksisterende verdilag
viser lærbarhet på gjenbrukt TRAIN, men generalisering og økonomisk verdi er
ikke bevist. Brukerens krav er en varig løsning som tåler ulike markeder;
videre tilpasning mot det samme kontrollutvalget er stoppet.

En ustabil Entry-tilpasning ble stabilisert med én regel beregnet kun fra
TRAIN512. Separat TRAIN128: Entry LONG-MSE 32,26→25,40, SHORT 80,44→27,25;
begge slår konstantbaseline. LONG bedres i 10/12 måneder, SHORT i 8/12;
mars-LONG er fortsatt klart verre. Frosne Entry/Exit-lag er kontrollert sammen
med eksakt cacheparitet, og Exit-forbedringen består. Ingen checkpoint er promotert.

Neste er en på forhånd bundet kontroll på 256 Entries fra juni2026, valgt med
eksisterende seed uten modellutfall. Juni er gjenbrukt utviklings-VAL, ikke
urørt holdout. Vekter, mål og utvalg er frosset. Ingen TEST, ny trening eller
full VAL er åpnet. Se docs/STABLE_READOUT_GENERALIZATION_20260916.md.

Evaluatordelen er nå kontrollert for det frosne utvalget: 12 nye og40 eksisterende
VAL-tester består. Originale rad-ID-er, pause/gjenopptak, åpne posisjoner/kostnader
og uendrede modellparametere utenom de frosne verdilagene er verifisert. Kandidaten
identifiseres som ONLINE, aldri som epoch-EMA. En delvis VAL kan ikke åpne full-VAL-porten.

Den skrivebeskyttede native-koblingen er nå kontrollert. 207 målrettede tester
består, inkludert eksakt reward-/klokkeparitet mot TRAIN, helgegap, sensur,
bevart bootstrap, uendret treningscursor og sperre mot nye optimizersteg.
Begge faktiske modeller bruker samme frosne Entry-lærer og samme opprinnelige
Exit-boundary-lærer. Exit-feil måles på state0 per valgt Entry; dette er ikke
feildekning av alle mulige holdetilstander. Native økonomi følger hele forløpet.

NEXT_RUN_POLICY åpner bare én eksisterende native evalueringsinvokasjon per
frosset variant, med256 forhåndsvalgte juni-Entries og null treningssteg.
Riktig kildebinding, fysisk omstart og alle eksisterende vakter kreves fortsatt.
Ingen senere VAL er kjørt ennå; full epoch/full5508 VAL og TEST forblir stengt.
Se handover_snapshot/NATIVE_FROZEN_READOUT_REVIEW_20260916.json.

Evalueringsutganger ligger under BASE/NATIVE_FROZEN_READOUT_20260916_BASELINE
og BASE/NATIVE_FROZEN_READOUT_20260916_CANDIDATE. PREPARATION_RESULT.json og
frozen_readout_val/OBSERVATION.json samt den bundne kampanjens receipts er
levende bevis; de finnes først etter respektive forberedelse/kjøring. Baseline
kjøres først. Candidate-planen forberedes deretter med den nye observerte booten,
slik at begge invokasjoner krever hver sin fysiske omstart. Begge bruker samme
rene kilde og frosne vekter. GitHub-publisering venter på ny eksplisitt godkjenning
etter automatisk avvisning; lokal evaluering er uavhengig av publiseringen.

## Start her

Aktiv arbeidsmappe: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
Dette er et Git-worktree av GX1_ENGINE; felles Git-katalog er
/home/andre2/src/GX1_ENGINE/.git. Data ligger fortsatt i /home/andre2/GX1_DATA.
GX1_ENGINE-mappen står på audit/v9-premiere-20260905 og er ikke aktiv kjørevei.
Mac /Users/andrekildalbakke/Desktop/GX1 XAUUSD er overleveringskopi.
Kjør ./handover.sh --check på Mac, eller bash scripts/gx1_handover.sh --check
på Linux. Les current_work og next_run i JSON; de eldre kilde-/checkpointfeltene
er merket completed_run_history. Scriptet er kun lesing og starter ingen jobb.
Les AGENTS.md, GX1_ARBEIDSMAAL.md, NEXT_RUN_POLICY.json og
[beslutningsgrunnlaget](docs/LEARNING_GATE_20260916.md).

Gamle «gjeldende»-avsnitt er samlet i handover_snapshot/
HANDOVER_HISTORY_BEFORE_LEARNING_GATE_20260916.md, uttrykkelig historikk.
Ingen eldre startplan, COMPLETED_RUN.json eller historisk kildekopi skal brukes
som alternativ vei. Statusnotater må kontrolleres mot levende prosess/receipt.

## Historikk: fullført referanse og split

Treningskilde fcd03e884c2b402979624df767a359c57562c2e1 var frosset gjennom hele
den fullførte kontrollen nedenfor. Ny rettelse endrer kun klippegrupper og
bundet overgang; denne kontrollens gamle kilde er referanse, ikke aktuell kode.
Finn gjeldende HEAD med git rev-parse HEAD og ny kampanjes bundne plan.

Reference32 bestod på fysisk boot444. Split16+16 bestod på boot445 og446;
alle tre receipts har guardPASS, trainer0 og observer0. Siste splitreceipt er
fra2026-09-16T12:13:56Z. Windows-task GX1NativeLearningCalibration er deaktivert
etter terminal kontroll. Ingen videre trening er planlagt av disse planene.

Faktisk tilstandssammenligning består: alle14 komponenter er eksakt like,
inkludert modell, frossen lærer, Adam, EMA, scheduler, RNG, epochrekkefølge og
progresjon. Bare checkpointnummer og sessionidentitet er unntatt som forventet.
Begge armer endte på global5809, epoch_index1, next_batch_offset1728, phase train,
complete=false. Reference er checkpoint96, split er97; begge starter fra
original95/global5777/offset1696. Original95 og stoppet134 er bevart.

Reference-løkken tok170.835s for32 steg/512 TRAIN-rader,5.33859s/steg.
Med checkpointlagring174.233s; hele den bevoktede invokasjonen832s, hvor655.178s
var før treningsløkken. Windows-omstart/controllerforberedelse kommer i tillegg.
Peak8224MiB,58Ccore/62Cmem/206.03W. Ingen relativ samlet speedup eller full-epochETA
hevdes. Ny full VAL256-paritet er ikke målt i denne TRAIN-kontrollen; tidligere
native GPU256-bevis står i policy med sine opprinnelige kildebindinger.
Førstebatchdiagnostikk før oppdatering er ikke etter32-læringsbevis.

## Eksakte tilstander og bevis

Artifactrot (kalt BASE nedenfor):
/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912

- Original95: BASE/NATIVE_FQI_TARGET_REFRESH_F11C1DBE/
  .gx1-candidate-training-session.CANDIDATE_BUNDLE/candidate_training_state_slot_0.pt
  SHA ca18cdf27f1ddf22741c893456d2040e1e53ce0cf1ac8e9532cb780973fd5dbe.
- Reference96: BASE/NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/
  .gx1-candidate-training-session.CANDIDATE_BUNDLE/candidate_training_state_slot_1.pt
  SHA77294b3f04afca6c0acb1e078fb68f392f6ba3ee0a1858a669a8c91b12622d0c.
- Split97: BASE/NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_SPLIT/
  .gx1-candidate-training-session.CANDIDATE_BUNDLE/candidate_training_state_slot_0.pt
  SHA94bbb8101036f25d0e27bf272b4b30a2eb1dfccff41a0e56466de0acfbd088ef.
- Resume: BASE/NATIVE_FROZEN_TRACE_RESUME_EQUIVALENCE_FCD03E88/RESULT.json,
  SHAcd52dda262aebd33a8197db25850fa54a3ac2c4dfe6c354fa856c8316ec61b18.
  OPERATOR.py og logger er bevart der. Første operatørforsøk manglet PYTHONPATH
  og stoppet før modellimport; korrigert kjøring bestod. Ingen trenerkode endret.
- Reference-fart/minne: BASE/NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/
  REFERENCE32_OBSERVATION_20260916.json,
  SHA09e539af550379a565e0d400bfb5181bbdc4f3d52a14033de0a6a3e7f9a09050.
- Receipts: /home/andre2/GX1_RUNS/
  NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE_BOOT443/receipts og
  NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_SPLIT_BOOT444/receipts.
  Runtime-navnet er forberedelsesboot, ikke nødvendigvis faktisk kjøreboot.
- Samlet oversikt og lokale kopier: handover_snapshot/
  NATIVE_FROZEN_TRACE_CONTROL_RESULT_20260916.json og
  NATIVE_FROZEN_TRACE_RESUME_EQUIVALENCE_FCD03E88.json.

## Læringsmåling95→96 ferdig — ingen utvidelse begrunnet

CPU-målingen gjenbrukte lagrede inputs/targets og før95-output:174.65s,
topp-RSS2 222 524KiB. Ingen ny materialisering, lærerberegning, backward,
GPU, VAL eller TEST. Før95-output på første batch ble reprodusert eksakt.

- Første trente batch,16 Entries/64 overganger, samme femstegsmål:
  Entry-MSE3.56205→3.40652 (4.37% ned); begge16FLAT, også læreren.
  Exit-MSE26.17057→26.16781 (0.01054% ned); MAE blir litt verre.
- Bred1024 TRAIN-kohort: Entry-MSE12.56542→12.66827 (0.82% opp), fortsatt
  1024FLAT. Læreren velger914FLAT/49LONG/61SHORT. Bare16 av disse1024 lå i
  siste32 oppdateringsbatcher; de andre1008 er fortsatt TRAIN, ikke holdout.
- Bred Exit bruker gamle faste ettstegsmål for sammenlignbar diagnose,
  ikke som påstått femstegs-loss. MSE9.07095→9.08508. LONG blir4096HOLD.
  94.70% av kvadrert LONG-prediksjonsendring skyldes en konstant forskyvning;
  korrelasjon med ettstegsmålet er nær null. Bedre situasjonsvurdering er ubevist.
- Bred forecast-MSE blir litt verre på alle fire horisonter.32 steg og én
  femstegsbatch avgjør ikke endelig lærbarhet, men åpner ikke større trening.

Bevis: handover_snapshot/FROZEN_TRACE_LEARNING95_96_REVIEW_20260916.json.
Fullt resultat og nye BATCH_00..63_AFTER96.json ligger under BASE/
NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/LEARNING95_96_20260916.
RESULT.json SHA47822d2d58f2fcfdab82c22065c3c393d8ef2f1073b409dccff57d8c052c071c.

Årsaksdiagnosen er nå ferdig; se neste avsnitt. Ikke gjenta32/1024-kontrollen.

## Årsaksdiagnose — målkjede og verdilesing

Se [full årsaksrapport](docs/VALUE_LEARNING_CAUSE_20260916.md). Én cacheanalyse
og en avsluttende hodeattribusjon ble kjørt på CPU med eksisterende audit-vakt;
ingen trening, nye targets, GPU, VAL eller TEST. Begge kilde-/checkpointbindinger
bestod. To evalueringer av én lagret batch reproduserte alle gamle outputs eksakt.

Entry får bare0,02456Bps videreverdi mot−5,85327Bps første likvidasjonsverdi;
1021/1024 lærerhandlinger er de samme uten videreverdien. Femstegsberegningen
endrer ikke denne Entry-supervisjonen. I den lagrede batchen stopper lærerpolicyen
LONG etter ett steg på59/64, mens SHORT bruker fem på64/64; alle successors finnes.
Dette er en målt sideavhengig målhorisont, ikke en holdetidsgrense.

På den samme lokale batchen kommer LONG-økningen0,15992Bps omtrent60% fra siste
verdihode og40% fra representasjonen. SHORT-fallet−0,03854Bps kommer fra
representasjonen. Felles biasendring er bare+0,00151Bps. Det er ikke dokumentert
at en biasrettelse løser problemet. Den brede konstante LONG-forskyvningen er
fortsatt94,70%; tilstandsavhengig læring er svak. Prognoser har TRAIN-signal,
men lærer andre mål og inngår ikke direkte i Entry-argmax.

Manglende tidlige Exit-eksempler ble avkreftet:497/4096 treningssamples er ved
tilstand0. Tidligere gradientbevis gir fortsatt ikke grunnlag for generell
backbone-/tapsvektendring. Den eksakte oppdelingen mellom upstream-moduler og
Adam-historikk er ikke målt; lærerens skjevhet er ikke alene bevis for retningen
på de32 ONLINE-oppdateringene. Dette var status ved cache-/hodeanalysen;
Adam-/klippebidrag er senere målt i rapporten om Exit-gradientklipping.

Planer, operatorer, logger og resultater ligger i
handover_snapshot/CACHED_VALUE_CAUSE_20260916/ og tilsvarende mappe under
BASE/NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS/.
RESULT SHAde50f50bf2d81817cbfa0a033aa9143c213a415b71d67b692972bf7ff2f266e2;
ATTRIBUTION_RESULT SHA98cbd1b223542f5a678265ba48f4910ed9596d1079f8937a5a5c5d12e0853559.
Cacheanalysen tok1,04s; hodeanalysen9,75s. Modell-/treningskode og checkpoints er bevart.

Neste: kandidat32 og faktisk512-måling er nå ferdige. Gjenbruk deres caches
for neste kontroll av observerte utfall ved Entry-ankeret; reward/stopp/bootstrap
er nå målt i TARGET_COMPONENT_CAUSE_20260916.md. Ingen nye
klippe-/tapsvekt-/modellforsøk, Adam-nullstilling eller større trening før én
målmekanisme er begrunnet. Se den oppdaterte klipperapporten.

Cachemappene under BASE/NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS/
FROZEN_POLICY_TRACE_NATIVE_BATCH_20260916_V2 og BROAD_TRAIN95_134_20260916 er
bevart. Videre analyse av nye output krever ikke nye forwards.

Bruk eksisterende eiere. CPU-modellfabrikken finnes i
run_unified_exit_random_access_val_v1; bruk av fabrikk til en CPU-måling er ikke
starttillatelse til separat VAL-runner. CPU-operator må ha
PYTHONPATH=/home/andre2/src/GX1_CURRENT og CUDA_VISIBLE_DEVICES='', kjøres via
gx1_capped_run.sh. Python er3.10; hashlib.file_digest finnes ikke. Eksisterende
file_sha256 og verify_candidate_checkpoint_resume_v1-eier skal gjenbrukes.

## Hva funnene betyr

TRAIN95→134 ga ikke bedre Entry/Exit-verdilæring i den brede målingen. Entry var
FLAT på1 024/1 024 i begge modeller. Noen lengre prognoser bedret seg på TRAIN,
uten at det beviser senere markedsfordel. Tidligere delvis VAL var bare HOLD i
de evaluerte hypotetiske sideforløpene. Lærerens verdiestimat er ikke fasit;
Entry-handlingen er fortsatt delvis avhengig av Exit-læreren.

Ny femstegstarget og minimal oppdeling av frosne target-forwards virker teknisk.
Det er ingen fast holdetidsgrense. Bare målbar læring og etterfølgende økonomisk
validering kan begrunne utvidelse. Full profitt omfatter alle valgte handler,
realisert cash, åpne verdier og kostnader. TEST forblir forseglet.

## Opprydding og bevaring

7 093 786 972 logiske bytes (7,09GB;6,61GiB) gamle midlertidige Entry-arrays er
slettet.7 093 800 960 allokerte bytes ble frigjort og kontrollert. Ingen levende
prosessreferanser fantes; eieren oppretter unike TemporaryDirectory-filer i w+
modus, ikke resume-input. Se handover_snapshot/DISK_CLEANUP_20260916.json.
Datasett, analyser/caches for neste måling, alle resultater/checkpoints og bundne
runtime-/kildeavhengigheter er bevart. En eldre9,2GiB Windows-backup inneholder
filer uten nåværende original; den er ikke dokumentert redundant og er beholdt.
Ingen WSL-diskkomprimering, reinstallasjon eller ekstra omstart er utført.

Overlevering, policytekst, statusskript, systemkart, Mac-speil og den eksisterende
timekontrollen skal være samordnet. Stabil langkjøring kontrolleres omtrent
hver time, én agent/én tung jobb. Ikke fyll ventetiden med arbeid.
