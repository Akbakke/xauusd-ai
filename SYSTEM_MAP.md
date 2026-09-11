<!-- GX1_DOCUMENT_CLASS: CANONICAL | stable lifecycle-v2 system map -->
# GX1 system map

This file is the stable map of the GX1 Entry Transformer V10 system. Runtime
facts belong in `CURRENT_HANDOVER.md`; immutable hashes belong in a generated
handover bundle.

## Authority order

1. A reviewed `gx1_handover_bundle_v1` generated from the exact host state.
2. `CURRENT_HANDOVER.md`, updated from the latest signed/read-only snapshot.
3. Clean lifecycle-v2 source commit and its immutable authority manifests.
4. Historical reports listed below. They describe what happened, but cannot
   authorize a new run.

TEST remains sealed. Nothing in this map grants training, reboot, task-enable,
CUDA, promotion, live trading or cloud-spend authority.

## System components

| Component | Canonical responsibility | Required authority/evidence |
|---|---|---|
| Source | Lifecycle-v2 code and contracts | clean repo path, branch and commit |
| Data | PRETEST TRAIN/VAL M1, MTF caches and Entry windows | data authority/manifest with exact file hashes and split limits |
| Model | Entry V10 plus unbounded Exit model | model/contract authority bound to source and feature order |
| Entry | Uses 480 local M1 rows at the first Exit state: 479 prior closed bars plus current post-fill bar | first-state witness and causal gather bindings |
| Exit lifecycle | HOLD/EXIT_NOW only for open trades; no 512-bar forced exit | lifecycle-v2/index/closure authorities |
| Exit memory | 1..512 post-entry rows as a rolling detail tail; lifetime may continue to the split end | random-access state-view and lifetime-summary contracts |
| Features | 238 local signals, 71 continuous context fields, one categorical context field, 176 fields per MTF lane, eight feature families | exact registries, normalization and M5/M15/H1/H4/D1 gather hashes |
| Economics | Side-correct executable PnL, cost policy, financing, closure-aware elapsed-time discount | frozen economics facts/policy and source receipts |
| Sampling | Outcome-blind bounded random-access transitions from the full holding-time tail | selected sampler receipt and schedule SHA |
| Training | Fixed-step TRAIN windows, target/online/optimizer/EMA/RNG resume | launch manifest, checkpoint pointer and equivalence receipts |
| Campaign | Fresh physical Windows boot before each heavy invocation; atomic progress and receipts | campaign plan, task/boot evidence and controller hash |
| Safety | Signed guard is the safety owner; sidecar status is observational | guard receipt/log plus process and GPU safety snapshots |
| Validation | Full 5,508 Entry cohort, both sides, learned Exit on open trades only | final TRAIN authority, full-VAL campaign and rollout cursor receipts |

## Current host boundary

The post-reboot snapshot binds BootId 359, boot time
`2026-09-11T10:39:11.5000000Z`, and a successful one-shot probe at
`2026-09-11T10:40:24.7603135Z`. Ubuntu `/bin/true` needed 14,070 ms on the first
cold call; the following exact `wslpath` call needed 71 ms. `WSLService`,
`vmcompute` and `hns` were running.

This identifies the immediate launch defect: the prior 8–10 second first-call
budget was shorter than a healthy observed 14.07-second cold start. The narrow
repair is one bounded 30-second allowance for the first cold WSL call, with no
retry, terminate, shutdown or reset. The campaign remains disabled until its
installed task has zero automatic retries and the source-bound controller is verified.

## Document disposition

`DOC_INDEX.md` is the complete Markdown inventory and disposition authority.
Canonical documents are the four current status/map/index files, `AGENTS.md`,
`GX1_RULES.md`, `README.md`, the data/telemetry/worktree/integrity contracts,
and the exact authority manifests. All other tracked Markdown is explicitly
marked `HISTORICAL`, `SUPERSEDED` or `PRIVATE / REMOVE` there. Historical and
superseded text cannot authorize training, reboot, task enable, CUDA, TEST,
promotion, live trading or cloud spend.
The post-reboot baseline was clean on branch
`feature/unbounded-exit-lifecycle-v2-20260910`, commit
`fb4f060d60d7017bf4188684cf5c0b5f05e110b4`; the successor source must be
resolved and bound explicitly by the read-only collector. Data, model, checkpoint and
campaign authority still require their exact manifest bindings before launch.
