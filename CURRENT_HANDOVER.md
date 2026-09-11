<!-- GX1_DOCUMENT_CLASS: CANONICAL | current operational status -->
# GX1 current handover

Earlier probe evidence from `gx1_post_reboot_handover_snapshot_v1`, observed
`2026-09-11T10:54:04.2445771Z`. Snapshot file SHA-256:
`6c9228f1116a26b2d1037b63648a0f404bd9695425dbfaac68d567f18d9ed446`.

## Decision

**BLOCK_TRAINING_PENDING_VERIFIED_STARTUP_OWNERSHIP_SUCCESSOR.**
BootId 361 began at 2026-09-11T15:57:48.5000000Z. V18 bound clean source
747111162705a8f7e8b10df4b0ede641cf4431ab and passed Windows cold-start,
strict bridge configuration and signed telemetry checks. Its first smoke
invocation began at 15:59:23.577665Z and stopped at 16:03:15Z.

The exact trainer exception was UNIFIED_EXIT_FIXED_STEP_CHECKPOINT_DIR_EXISTS:
campaign begin created the same fresh directory the bootstrap trainer must own.
No optimizer step, resume pointer or terminal progress was produced. The
scheduler is Disabled with zero retries and no candidate process remains.
The unresolved V18 ACTIVE marker is preserved as failed evidence; never delete
it to retry this plan, and do not treat it as a currently running process.

The correction leaves fresh checkpoint-directory creation to the trainer and
places observer status under runtime_root. The fixed-step CUDA smoke now uses
the existing private-FIFO notification after CPU checkpoint verification and
before model construction, so the guard enforces its separate model budget.
The next producer also had a circular dependency: it required an epoch session
before the resume proof that authorizes that session. The existing 4-step
reference and 3+1 proof now form a resume-proof campaign; the subsequent
selected-training campaign contains only epoch windows and requires the
completed equivalence-bound epoch manifest.

Targeted CPU tests: 32 passed. The actual Windows hardening harness also passes,
including observer_checkpoint_isolation=PASS. The operator reprioritized direct
progress to one year / one epoch over repeated broad suites. The fourth full
suite was interrupted: 683 tests and 14 subtests passed in 176.24 seconds before
SIGINT; this is partial verification, not a full PASS. Use the passing targeted
checks for these concrete startup fixes. Do not repeat the full suite merely
because a commit is due. Source must be committed and rebound before launch.

V18 plan file SHA-256:
b45bd3c8cb7f01f2e0c6e03122d33e4be258589bc7cd90afdd980b39003dbe81.
Exact immutable failure snapshot:
LIFECYCLE_V2_LOCAL_PILOT_20260910/PRELAUNCH_EVIDENCE_V18_74711116_BOOT360/FAILED_SMOKE_READONLY.json.
Trainer log SHA-256:
f03ef6e951e632f717f26a5449ffc018293f058815cb617d3234e278209083d8.
Guard log SHA-256:
de28ab6bf59f25454e86352381b1c67f4266a6967f0a423ff31034eea2cc1e1b.
The logs are bound by that snapshot. Guard peaks were 51 C core, 52 C memory,
132.48 W draw and 490 MiB resident VRAM, with the physical limit at 160 W.

## Earlier context


- The old five-year epoch 1 is preserved as immutable historical evidence:
  313,399 TRAIN rows, 39,175 optimizer steps, 5,509 Entry VAL rows and
  5,632,000 Exit states. It projected 14.48 hours per old epoch. Entry and Exit
  were economically negative, so blind continuation stopped.
- Epoch 1 proved full feature plumbing and gradient contact, but not causal
  feature importance or trading edge. The Entry specialist gate collapsed
  heavily toward volatility and H4 while generic routes still carried all
  fields. Exit contributed about 92.2% of joint VAL loss and could backpropagate
  into the Entry token.
- The old checkpoint learned a forced terminal at Exit state 512 and is
  historical only. It is not a direct resume checkpoint for the repaired
  learning problem.
- Lifecycle v2 keeps 480 rows of local M1 history at the first Exit decision,
  uses at most 512 post-entry rows as a rolling detail tail, preserves lifetime
  summaries and exact t/t+1 successors, and has no maximum trade lifetime or
  capacity-forced EXIT. TRAIN uses bounded outcome-blind random-access samples;
  VAL rolls open trades until learned EXIT or reports compute truncation.
- The lifecycle-v2 mechanics, data/index/normalization/economics bindings,
  fixed-step training, resume proof, campaign receipts and full-VAL path were
  developed and CPU-tested. The post-reboot baseline was clean on
  `feature/unbounded-exit-lifecycle-v2-20260910` at
  `fb4f060d60d7017bf4188684cf5c0b5f05e110b4`; the successor source must be
  rebound to its exact clean commit by the collector before launch authority.
- Historical probe BootId 359 began at `2026-09-11T10:39:11.5000000Z`. The one-shot probe passed
  at `10:40:24.7603135Z`: cold Ubuntu `/bin/true` took 14,070 ms and returned 0;
  exact `wslpath` took 71 ms and returned `/mnt/c/ProgramData/GX1`.
- `WSLService`, `vmcompute` and `hns` were running. The probe task is Disabled,
  returned 0 and has zero automatic retries. The campaign and legacy WSL
  bootstrap are Disabled. `GX1LifecycleV2PilotResume` is absent.
- The corrected installer was exercised on the actual host and verified
  `RestartCount=0`, S4U owner `Andre`, exact V17 source/plan identity and the
  one-minute startup trigger. The task was disabled after the BootId 360
  bootstrap failure. Its result 1 is not a CUDA/training result.
- The observed 14.07-second healthy cold call explains the earlier 8–10 second
  campaign timeout failures. Use one 30-second timeout for the first cold WSL
  call, with zero retry, terminate, shutdown or reset. Later calls keep their
  normal short bound.
- No ACTIVE file, guard file, trainer process or CUDA process existed. The
  original snapshot's GPU fields were null. A later canonical signed
  preinstall query (recorded 14:34:18Z) verified 37 C core, 44 C memory,
  22.59 W draw, 160 W limit and 78 MiB used. Live guard checks still apply
  to every future invocation.
- The Codex heartbeat remains paused. The probe reboot is complete; do not
  request another reboot for this diagnosis.
- The simple staged probe can call only bounded Ubuntu-22.04 `/bin/true` and
  `wslpath`. It cannot call trainer, guard, CUDA, GPU tools, WSL recovery,
  shutdown or reboot. Its local staged source hashes are recorded below; the
  installed Windows bytes must be rehashed independently.

## Required post-reboot evidence

The probe boundary has passed. Run the read-only handover collector with one
exact file for every role: `data_authority`, `model_authority`,
`checkpoint_authority`, `campaign_plan`, `task_status`, `boot_status`,
`probe_status`, `guard_status`, `process_status`, and `gpu_safety_status`.
Snapshot files may state that an object is absent; absence must be explicit and
machine-readable.

The current snapshot can supply the boot/task/probe/guard/process safety roles,
and the later canonical signed query supplies the GPU measurement. Complete
data/model/checkpoint/plan authority must still be bound to the successor commit. A snapshot that explicitly records `null` stays unknown; it is not a
PASS by absence.


## Prior committed baseline verification

- The prior committed baseline suite passed: **4,666 tests and 14 subtests**,
  in 1,230.05 seconds, under the capped audit wrapper (4 GiB memory, 512 MiB
  swap, CPU 0-7, one numerical thread). Bash syntax, Python compilation and
  `git diff --check` also pass. No CUDA or candidate training job was launched.
- Exact results: `/tmp/gx1_lifecycle_v2_bridge_count_final_20260911.log` and
  `/tmp/gx1_lifecycle_v2_bridge_count_final_20260911.xml` on the WSL host.
- The earlier suites imported some test helpers from the original checkout.
  `tests/__init__.py` now binds helpers to this checkout, and `.venv` is a
  physical local environment with the existing dependencies. The original
  checkout and its environment were not changed.
- Current pretest launch recipes require all eight lifecycle-v2 artifact
  bindings. Immutable older recipes are still rejected; they were not patched
  or admitted. Staging/session fixtures now implement the epoch setter.
- The trainer accepts the four exact campaign identity environment fields
  already verified by the capped runner. Episode teacher targets exclude a
  final HOLD without an observed successor; the legal HOLD action remains
  available and no forced terminal is introduced. Native usefulness analysis
  preserves that distinction and validates the episode seal separately from
  its economics-readiness sidecar.
- The collector/controller targeted suite and actual Windows PowerShell
  hardening harness passed, including 200,000-byte stdout/stderr drain,
  bounded timeout and `one_shot_initial_state=PASS`.
- The actual PowerShell boot-identity parameter admits the 30,000 ms cold
  call. The real parameter-block regression accepts that value and rejects
  0 and 30,001 ms without executing host writes. The Windows harness passes
  with `boot_identity_parameter_binding=PASS`.
- The actual Windows harness also proves empty, missing-field and extra-field
  configuration comparisons under strict mode. The complete real
  `Confirm-Gx1SignedHostTelemetryReady` path passed on BootId 360 without
  candidate start or reboot.
- The previous handover task is archived and its duplicate test run was
  interrupted. Only this takeover may advance the work; one heavy job at a
  time. The interrupted duplicate log is not verification evidence.

The ten-role V18 bundle passed with source 747111162705a8f7e8b10df4b0ede641cf4431ab,
bundle SHA-256 46895a69b75023d1242ea4f879c8bdb1386e81933a5d7a748134fe6d78727de1.
It is historical capture evidence, not authority for the pending successor.

The ten-role V17 handover bundle was captured with clean source
`2de8ff8af31a9d61567de2c799923cf1b821735a` before this attempt. It remains
historical evidence, not authority for the corrected controller. Its bundle
SHA-256 is `5c56ac833bca77f469525fb50fb1c5fe478d196a97a8a89a61558d1ff6b09d59`.

## Next action

Commit the targeted-verified startup-ownership correction. Preserve V18 as failed, then use a new output/runtime root and the
actual prepared BootId 361. Only the canonical controller may request the
next physical reboot and sequential smoke invocations. After GPU selection,
use --phase resume-proof first; build the epoch session only after real 4 versus
3+1 equivalence passes, then materialize selected-training against that proof
plan. The TRAIN 2025-06-01 through 2026-05-31 and full June-2026 VAL objective,
TEST seal and hardware guards are unchanged.

1. Verify the committed successor is clean with `gx1_handover.sh --source-only`.
   Rebind the existing smoke recipes and campaign plan to that exact commit
   and the corrected controller/installer hashes.
2. Collect exact host/GPU status, install the source-bound controller and
   verify zero task retries. Do not launch until every existing guard passes.
3. Bind the exact data, model, checkpoint and current campaign-plan authorities.
4. Generate `gx1_handover_bundle_v1` with
   `scripts/collect_gx1_handover_readonly.py`, binding the expected clean
   branch and commit explicitly.
5. If any check is missing or inconsistent, keep the campaign disabled and fix
   that single gate before any launch decision.

## Local evidence hashes

These identify only the current Mac copies:

- Old epoch-1 final report: `39a2696f665caa1e484ef2537e8a8512f186904ab3b15647fbe20140e7b7d13a`
- Epoch-1 system audit: `935e458de064e05e91c3514267026b552743bb0a38f438ea6df27d364646c930`
- Boot 358 pre-run audit: `32b020b0575bddac87c75401a3345fa0e3fad1607262416ba0aad0c6540489d2`
- Staged simple probe: `7943ebfcf2cb68e2dcd269f00d99ba11a23931d5ff3b8739d759cb4f8d584c30`
- Staged simple-probe installer: `efe66c5e9de35afeb30d96612f2203d4b13c44d3757d748712e0a7d20002262d`
- Staged RestartCount-removal patch: `f3e016068ffff56c4131a388b019e596596b7a05b7d6baf7172ac14c1b9eaa08`
- Post-reboot snapshot: `6c9228f1116a26b2d1037b63648a0f404bd9695425dbfaac68d567f18d9ed446`
- Installed probe evidence: `14befaaaeb9e8a290afbc60af5fc2d79a65bdb3b6e5e79fdf9c928b04205fcaa`
