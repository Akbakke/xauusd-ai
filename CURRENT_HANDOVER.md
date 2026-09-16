# GX1 — gjeldende overlevering, 2026-09-16

**Krev målbar læring før mer omfattende trening.** Ingen ny full epoch eller full
VAL er aktivert. Læringsgevinst med den nye targetberegningen og lønnsomhet er
ikke dokumentert. Neste modellarbeid er en paret måling, ikke en ny oppstart.

## Start her

Eneste kodebase: /home/andre2/src/GX1_CURRENT, branch work/gx1-current.
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

## Bekreftet terminalstatus

Treningskilde fcd03e884c2b402979624df767a359c57562c2e1 var frosset gjennom hele
kontrollen. Etterfølgende handovercommit endrer dokumentasjon/lesestatus, ikke
modell-/treningsmatematikk. Finn gjeldende dokumentasjons-HEAD med git rev-parse HEAD.

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

## Neste konkrete modellarbeid

1. Gjenbruk CPU-cachen under BASE/NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/
   OPERATOR_OBSERVATIONS/FROZEN_POLICY_TRACE_NATIVE_BATCH_20260916_V2.
   BATCH_INPUTS.pt SHA5cff4b8ab5406f2cf68875d9e340ef617f1b877e22c56f5e54e67ae0af68e234;
   TARGETS.pt SHA253ceaf5ec91683adcb6c4f552cba4bfb53fc2aecf9a96da4b6fa9f558f8fed8.
   Sammenlign ONLINE95 mot reference96 på identiske inputs/femstegsmål og eval-
   innstillinger. Rapporter lokal Entry/Exit-læring mot baseline, ikke profitt.
2. Bred1024-kohort med cached inputs/targets/outputs finnes under samme
   OPERATOR_OBSERVATIONS/BROAD_TRAIN95_134_20260916. Gjenbruk før-resultater.
   Dette er64 faste native batcher fra offset1696..4079 gjennom tolv måneder.
   Skill eksponerte/ikke nettopp trente batcher og ettstegs-/femstegsmål.
3. Følg docs/LEARNING_GATE_20260916.md. Før et nytt avgrenset native læringsløp
   må måleopplegg og sluttpunkt bindes i eksisterende policy/campaign. Ikke utvid
   dagens32-scope, restart134, gjenta teknisk32, bruk replay eller start full VAL.
   Ingen ny lærer/EMA-reset eller bred regel-/modelljakt nå.

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
