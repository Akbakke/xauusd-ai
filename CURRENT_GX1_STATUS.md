<!-- GX1_DOCUMENT_CLASS: CANONICAL | short current status -->
# GX1 status nå

## Latest verified run and completion correction - 2026-09-11

V20 on BootId 363 completed three actual CUDA optimizer steps at batch 4 and
saved a checkpoint. The signed guard exited 0 at 17:03:57Z: peaks 51 C core,
52 C memory, 132.55 W and 2488 MiB VRAM. CPU preflight took 228 seconds;
the model stage took 13 seconds. Two measured post-warmup steps processed
8 Entry rows / 32 transitions in 5.663114863 seconds. This short measurement
includes smoke checkpoint overhead and is not qualified long-run throughput.

Campaign registration did not complete. Three exact completion defects were
verified: File.Replace is unsupported on the WSL UNC status path; Windows
Start-Process returns null ExitCode unless Handle is retained before exit;
and fixed-step progress used a no-LF digest while its campaign owner requires
LF. Original V20 progress, checkpoint, measurements and ACTIVE marker remain
unchanged. A source-identical terminal observer recheck returned 0, but the
canonical record attempt rejected the wrong progress digest. No successful
campaign receipt was created, and no progress was relabelled to force it.
The scheduler is Disabled with zero retries. Exact evidence:
LIFECYCLE_V2_LOCAL_PILOT_20260910/PRELAUNCH_EVIDENCE_V20_70884CBD_BOOT362/FAILED_REGISTRATION_READONLY.json.

The correction stores reporter status on local Windows NTFS, retains both
child process handles and rejects missing exit codes, and publishes fixed-step
progress with the receiving campaign owner's hash function. The production
publisher-to-campaign-record regression and Windows source tests passed
(15 tests); the actual Windows harness passed local atomic replacement and
exact child exit codes 0/7. Training calculations, sampler and features are
unchanged. Commit this narrow successor, rebind and continue the guarded
smoke path. Do not run another full suite or fabricate a V20 campaign PASS.


## Current operator goal and bootstrap correction - 2026-09-11

Train toward positive cost-adjusted net Bps with the complete feature contract
and learned cooperation across all eight families and timeframes. First run
the one-year TRAIN smoke (2025-06-01 through 2026-05-31), then full June VAL.
After required smoke/resume readiness, the operator authorizes up to 30 epochs
with early stopping. Use one heavy process, targeted checks only, and no
repeated full suites. TEST remains sealed. The selected random-access epoch
contains 16,384 Entry pairs / 65,536 transitions; it is not a full pass through
the 65,295 eligible one-year entries. Report that sampling definition honestly.

V19 on BootId 362 passed cold WSL, checkpoint-directory ownership and the
CPU-to-model guard transition. It stopped before any optimizer step at
UNIFIED_EXIT_RANDOM_ACCESS_BOOTSTRAP_KEYSET_INVALID. Its frozen checkpoint
contains the exact 36 static Exit keys retired by the established source-state
successor. All 758 retained tensors match current names/shapes/dtypes. The
scheduler is Disabled with zero retries; preserve the failed V19 ACTIVE marker.
Exact failure evidence lives in
LIFECYCLE_V2_LOCAL_PILOT_20260910/PRELAUNCH_EVIDENCE_V19_2D8C4DDB_BOOT361/FAILED_SMOKE_READONLY.json.

The bootstrap now shares the existing exact retirement-key owner with the old
migration tool. It rejects partial retirement, unknown keys and missing or
incompatible live tensors; preserves the full old source digest; strictly loads
all retained weights and retains new v2 initialization. No loose restore or
optimizer migration is introduced: this is the declared fresh v2 warm start.
Thirteen targeted tests passed. The actual frozen online and target checkpoint
both passed CPU bootstrap, exact retained-tensor checks, preserved v7 input
normalization and strict v2 restore. The proof is BOOTSTRAP_SUCCESSOR_CPU_READONLY.json
beside the failure snapshot. Commit clean source and rebind before guarded CUDA.

Efficiency audit: old no-fill FP32's measured 12.5% gain is not active in the
current deterministic_fp32 fixed-step executor. The selected sampler's 1,072.45
seconds measures CPU materialization/collation only, not GPU training or VAL.
No new GPU throughput or positive-Bps result exists yet. Fewer sampled rows and
unbounded Exit semantics change the learning problem; do not label the entire
old/new time difference as quality-preserving acceleration.


BootId 361, 2026-09-11. Kampanjen er deaktivert med null automatiske forsøk.
Ingen kandidatprosess kjører.

V18 passerte kald WSL-start og signert telemetri med ren commit
747111162705a8f7e8b10df4b0ede641cf4431ab. Første smokejobb stoppet før
optimizersteg på UNIFIED_EXIT_FIXED_STEP_CHECKPOINT_DIR_EXISTS: kampanjestarten
opprettet mappen som trenerens bootstrap krevde å opprette selv.
Feilloggen og den uavklarte ACTIVE-markøren er bevart; V18 skal ikke startes igjen.

Rettelsen gir treneren eierskap til den ferske checkpoint-mappen og legger
observatørens statusfiler i kampanjens kjøremappe. Trenerbanen signaliserer nå
til eksisterende guard etter CPU-kontroll av checkpoint og før modellstart.
En bekreftet sirkelavhengighet i neste port er også rettet: 4-stegs referanse
og 3+1-resume gjennomføres før det bevisbundne epochmanifestet opprettes.

32 berørte CPU-tester og reell Windows PowerShell-test passerer.
Brukeren prioriterte direkte fremdrift mot trening. Fjerde fullsuite ble stoppet
etter 683 tester og 14 deltester; dette er delvis verifikasjon, ikke full PASS.
De beståtte målrettede testene brukes for disse konkrete rettelsene. Ingen ny
fullsuite bare fordi endringen skal committes.

Målet er fortsatt én fersk lifecycle-v2 epoch med TRAIN 2025-06-01–2026-05-31
og full juni-2026 VAL gjennom de avtalte smoke- og resumeportene.
Ingen tvungen EXIT ved 512 eller maksimal tradelevetid. TEST er forseglet.
Én eier og én tung jobb om gangen. Eksakt feilbevis og neste steg står i
CURRENT_HANDOVER.md.
