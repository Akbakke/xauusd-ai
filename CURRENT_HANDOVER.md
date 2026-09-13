<!-- GX1_DOCUMENT_CLASS: CANONICAL | current operational takeover -->
# GX1 lifecycle-v2 takeover — 2026-09-13

The full one-year smoke and full June VAL are complete. First full five-year
TRAIN completed at 2026-09-13 04:35:21 UTC: 313,399 rows, 19,588 optimizer
steps, checkpoint 309. That exact trained state is preserved on the host and
in the verified Mac backup.

At 07:01 UTC the user explicitly ordered a throughput restart. The old
batch-16 VAL and Windows task were stopped. The new f2b597f8 source is pushed;
its guarded successor is prepared with Exit-VAL batch 128, TRAIN and Entry-VAL
batch 16. It restores model/target, optimizer, EMA, scheduler, order, RNG and
selection state from the completed TRAIN. June VAL starts with fresh
accumulators. This dated publication does not claim the new CUDA start or
speedup: the live observer and VAL_BATCH_THROUGHPUT_VERIFIED log provide that
subsequent evidence. No positive Bps, admitted model or live readiness is claimed.

## Takeover

Read GX1_ARBEIDSMAAL.md, then SYSTEM_MAP.md and docs/audit_20260912/.
On the training host, from /home/andre2/src/GX1_HANDOVER_20260913:

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
```

These are read-only source/binding/process/progress observations. They do not
load a model, rehash a full checkpoint/dataset or grant execution authority.
--source-only omits process/runtime reads; --verbose also prints this document.
A clone on another host requires the named artifacts restored before these
checks can pass. Missing evidence never authorizes a fresh training run.

The documentation branch is separate from the frozen execution source.
CURRENT_NATIVE_RUN.json selects the exact native campaign; the historical
PROJECT_STATE_xau_direction_launch.json and old handoffs do not describe it.
A missing native PID around a successful RESUMABLE/reboot boundary requires
checking the existing Windows task and boot, not launching another process.
Monitor about every 15 minutes and stay silent on ordinary healthy progress.

## Exact binding

- Frozen source: /home/andre2/src/GX1_VAL_THROUGHPUT_V34
- Commit: f2b597f8ff9a62a81dfb7106afd23b4ae74c894b
- Frozen branch: fix/native-val-throughput-20260913
- Data root (D): /home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912
- Runtime (R): /home/andre2/GX1_RUNS/UNIFIED_EXIT_FULL_TRAIN_VAL128_F2B597F8_BOOT398
- Plan: D/CAMPAIGN_NATIVE_VAL128_F2B597F8_BOOT398/CAMPAIGN_PLAN.json
- Plan file SHA: 6e990eed1a6d8a3d4013fb0689ddbec92c05caa62d2a4d9f0f1caa046a0bcc46
- Plan internal SHA: b995f4a3883ac7e099707b464649acc1ea36ad63d53e0a41cf463efef3f637b2
- Recipe: D/NATIVE_FULL_TRAIN_RECIPE_VAL128_V5.json
- Recipe file SHA: 1af1bec61ad9f794fc97964da19665f9de5f665dff6897e6488f09855f47f5fd
- Session (S): D/.gx1-candidate-training-session.FULL_TRAIN_VAL128_CANDIDATE_BUNDLE
- Expected successor session contract SHA: f7b0205ff969c863e81308db766dd323067d0dc3e2d0f615bd5728379dacda51
- Preserved TRAIN-origin state SHA: c5efc4aaea602bf725139fffe0081629bad607e770f0b5cad4f1ef1e9e294349
- Epoch-1 VAL progress: S/native_val/epoch_0001/ROLLOUT_PROGRESS.json
- Windows task: GX1RandomAccessCampaignV2
- Windows script checkout: C:\Users\Andre\GX1FullTrainNative_f77dd273 (historical name)
- Mac SSH alias: gx1-3090-lan; Windows SSH, then Ubuntu-22.04 as andre2.

D/R/S are prose abbreviations only; CURRENT_NATIVE_RUN.json holds full paths.
Alternating state slots are mutable: use the pointer, never the slot name as
resume authority. The current controller owns exact resume validation.

## Progress and automation

In the preserved predecessor, seven actual TRAIN resumes, the first main
TRAIN-to-VAL transition and the
first native VAL pause/reboot/resume are verified. Invocation 9 ended RESUMABLE
with guard PASS and both exit codes 0; invocation 10 continued on Boot 398.
At 06:31 UTC, VAL had 28,452 forwards, 454,248 state views and 4,840.6 active
seconds; completed_invocation_count=1. Final VAL/selection remains pending.
Read handover_snapshot/NATIVE_OBSERVATION.json and live status for newer
evidence. The prior smoke's finished VAL is distinct.

The plan's 600 windows are a resource ceiling, not 600 epochs. Native windows
are 5,400 seconds; outer guard 7,200; native VAL compute windows 4,200 and
resumable. TRAIN checkpoints every 64 optimizer steps; VAL every 64 forwards.
The startup task owns checkpoint/receipt/pause/reboot/resume. A real error
stops progression; it cannot repair software itself. Existing standing
authorization covers ordinary necessary work. Do not repeat approvals.

Current local limits: 300 W physical, 310 W actual-draw stop, 85 C core,
80 C memory junction, 12,288 MiB VRAM. The keeper lowers power to 200 W at
core >=80 C. Local guards own frequent monitoring. Recipe-owned CPU/workers
remain unchanged.

The predecessor was not fully utilized: GPU 31–41%, about 97–98 W actual,
1,454 MiB VRAM and 51 C core, with native process CPU about 110%. A bounded
10-second stack sample found the main thread predominantly in model calls;
this did not establish a need to force 20 CPU threads.

The successor changes only Exit-VAL batching to 128, leaving TRAIN/Entry-VAL
at 16. The first production batch compares the same rows with original
16-row forwards, requires close Q values and identical selected actions, and
logs measured inference time/speedup. It is a limited numerical/throughput
check, not a proof of identical results for every June state. Actual GPU use,
peak memory and end-to-end progress must be observed after the new start.
No performance gain is claimed in advance. FP32, all features, costs,
full June coverage, learned Exit and safety limits remain bound.

Four focused tests passed (exact training-state preservation, rejection of
changed TRAIN/model contracts and full-cohort batch-128 pause/resume). The
actual checkpoint-309 origin passed with 19,588 optimizer steps, 769 model
state entries, 726 optimizer state entries and no completed VAL snapshot.
The versioned Git hooks passed. No old training epoch or full smoke was rerun.
The old VAL is preserved at 39,204 forwards / 625,902 state views / 6,738.36
active seconds, state 113, cursor 3505. It is not mixed into the new batching
contract. Original source 2959cd09, session and logs remain available.

## Evaluation meaning

TRAIN covers 2021-06-01–2026-05-31: every Entry row once per epoch, with four
outcome-blind sampled Exit transitions per Entry, both sides and first-state
anchor. Full Entry coverage does not enumerate every possible Exit state.

June has 5,508 entries and 11,016 alternative long/short paths, not 11,016 live
orders. Entry chooses LONG/SHORT/FLAT by unique Q argmax. A forward is a batch
calculation; state_index advances position age across the cohort, not June's
calendar. Early OPEN counts do not prove month-long holding.

Acc means agreement with the frozen TRAIN-target Q action, not win rate or
calibrated confidence. An aggregate main TRAIN loss/accuracy has not been
reported in the compact log; do not invent values. Main VAL will provide the
actual coupled Entry/Exit net Bps and quality diagnostics.

June VAL follows every epoch. Maximum 30, patience 5; no blind 30-epoch wait.
A selected path still open at month end makes full-policy Bps unavailable.
Closed-only means cannot replace it. Inadmissible epochs consume patience.
Native COMPLETE may have no selected checkpoint and bundle_written=false.
Computational completion is not admission or achievement of positive Bps.

This cohort evaluation is not a capital/concurrency-constrained portfolio
backtest. TEST remains sealed; no paper/live, broker action or external spend.
No confidence cutoff or forced shorter Exit was added. MFE/MAE and current
executable PnL are causal Exit inputs; missing fields in compact progress do
not mean zero excursion.

## Completed smoke and history

Full one-year TRAIN (2025-06-01–2026-05-31): 65,295 entries, 4,081 optimizer
steps, 261,180 sampled Exit transitions, about 2 h 25 active including prep.
The earlier 16,384-entry technical sample was not a full year.

Full smoke June VAL V34: 5,508 entries, 11,016 side paths, 229,043 forwards,
3,526,097 state views, 26,270.708 active seconds (~7 h 18). Entry selected
SHORT everywhere. 5,500 closed selected paths averaged -0.5986875941 Bps;
8 selected paths were naturally censored. Full selected-policy Bps is
unavailable, not positive. No compute truncation or forced-512 exit.
Result: /home/andre2/GX1_RUNS/UNIFIED_EXIT_RANDOM_ACCESS_V21_ADCE4082_BOOT363/FULL_VAL_CPU_V34/rollout/VAL_RESULT.json
SHA: abed5ce37e9aca89c45812c93596ad6af1e1cd46484476215f75d20d3209e276

The older forced-state-512 five-year epoch is preserved, not a resume source.
The three pre-main zero-step failures (import guard, coordinator chunk argument,
unbounded CUDA dispatch) were fixed in small successor commits ending at
2959cd09. Failed initial sessions/logs remain preserved. No trained main
progress was discarded. Reuse completed data construction and passed checks.

## Known limits and next result review

The completed user-requested three-agent audit is in
docs/audit_20260912/SAMLET_GJENNOMGANG.md and its focused reports.
No proven current-run-invalidating defect was found. Important findings:

- Initial LONG financing omits the fill-to-first-decision minute:
  ~0.001026694045 Bps per selected LONG at the bound rate. SHORT is unaffected.
  Carry this into net-Bps assessment; do not silently alter frozen accounting.
- Epoch-end target coverage checks the last resume fragment; it could reject
  a valid later epoch tail. Epoch 1 passed and checkpoints precede the check.
- Tiny positive clamped entropy is not evidence of useful cooperation.
- Smoke Entry routing was almost collapsed (effective routes ~1.0006) and
  some feature gates saturated. Candidate recovery and input influence remain
  to be measured on its actual selected checkpoint.
- TRAIN has 3,136 unknown source gaps handled by censoring; VAL has zero after
  calendar correction. Preserve base normalization v7 and the distinct full
  TRAIN lifetime-summary fit; do not substitute newer-looking v8.
- Cost assumptions: commission 0, slippage 2 Bps/execution, long annual
  financing 5.4%, short financing 0, GSLO 0, risk penalty 0, annual capital
  hurdle 10%. This is a prospective scenario, not proven historical broker cost.
- June has been used in development. Feature wiring is not proof that every
  route contributes useful information. Same-bundle live parity and portfolio
  edge remain unproven.

Next: finish this candidate's June VAL, inspect Entry action counts,
exits/censoring, full-cohort net Bps, Q diagnostics and route/gate evidence.
Let existing checkpoint selection and early stopping continue. Change code
only for a named observed blocker, with the smallest fix and focused checks.

## Preservation

The docs/lifecycle-v2-takeover-20260913 branch descends from the successor
f2b597f8. Both the original 2959cd09 and successor f2b597f8 source branches
have been pushed to origin. The running checkout stays frozen; this separate
documentation checkout observes it. A local publication receipt records the
actual docs commit and verified remote ref after push.

A separate Mac checkpoint backup is recorded in
handover_snapshot/BACKUP_MANIFEST.json. Read its decision and verified hashes.
Git does not store the ~195 MB model state, raw prices/features or all GX1_DATA.
A Git push is not a full-machine backup; no full raw-data backup is claimed.
Restore exact source/recipe/contracts before considering a guarded resume.
Cleanup already reclaimed disk space; more cleanup/VHDX compaction is not
part of this takeover task.
