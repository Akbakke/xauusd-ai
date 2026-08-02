# GX1 operating rules

`AGENTS.md` is the current operational constitution. `SYSTEM_MAP.md` is the
current architecture map. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` records
the current work state. If these disagree, fail closed and repair the
documentation and code together.

## Scope freeze

The only active work is offline shared-featurebase work for XAUUSD: Entry M5,
Exit M1, one shared eight-family causal feature owner, and offline
train/OOS/replay evidence. Live/paper/demo operation, broker work, daemons,
polling, live-tail admission, promotion, continual drift and online
adaptation are forbidden. Do not add routes or complexity outside this path.
Exact cache reuse and overlap/hash-proven tail append are required; invalid
evidence fails closed.

## Takeover quickstart

Start with the executable status, then the bounded reading order:

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
```

Read `GX1_RULES.md`, `AGENTS.md`, `SYSTEM_MAP.md`, the current handover and
the machine-readable launch state before touching code. The current offline
V8/V13 dataset and explicit CPU-safe smoke recipe are usable only for bounded
train/OOS/replay evidence; they do not change machine-readable `BLOCK` or
authorize a model. Do not start a second heavy job, direct-run the trainer,
select by `latest`/mtime, or touch live/paper/demo/broker/drift routes. Use the
handover's exact paths and resume stage, and treat any cap kill or missing
hash-bound artifact as terminal until a fresh gate proves otherwise.

## Non-negotiable rules

1. Trade and model XAUUSD only. Entry contracts must not depend on market data
   from another instrument or expose another traded output.
2. No fallback, guessed default, mutable `latest`, stale artifact, synthetic
   decision input or soft pass-through is allowed. This is absolute and covers
   evidence used to make a decision about the code, not only values inside it:

   a. Every decision-affecting number has exactly one of three legitimate
      origins: a named constant in a contract owner, a statistic fitted on
      real declared data, or an explicit CLI/recipe input. Anything else —
      including a value chosen because it "looked reasonable" — is forbidden.
      If you cannot name the origin in one sentence, it is a guessed default.
   b. Never invent a magnitude to make something work. When an existing value
      must change, adopt the convention the surrounding code already uses and
      say which one; do not introduce a new constant. Removing an exception is
      allowed, inventing a number is not.
   c. Synthetic, random, placeholder or toy-dimension data may never justify a
      conclusion about production behaviour, a code change or a claim to the
      user. A diagnostic on such data proves only that code runs. Conclusions
      require real declared bytes at real contract dimensions, or a proof from
      source and algebra that holds independent of data.
   d. State the evidence class for every claim: proven from source, measured
      on real data, measured on synthetic data, or unproven. Downgrade or
      withdraw a claim the moment its evidence class turns out weaker than
      stated, and record the withdrawal.
   e. A diagnostic instrument must only report what is valid where it runs. If
      a measurement cannot be taken at that point, omit the field rather than
      emitting a zero, an empty value or a placeholder that reads as a result.
   f. A threshold compared against a sampled statistic must be at least that
      statistic's sampling error at the sample size where the comparison is
      actually made, or the comparison must be taken on the complete declared
      population. A tolerance tighter than the noise of the quantity it judges
      does not measure the quantity; it trains on, or fails on, the noise.
      State the sample size and the resulting bound whenever a threshold is
      introduced or moved. Five separate defects of this exact shape have been
      found — deadness judged on 1,024 of 369,303 rows, and a 0.02 prior-match
      tolerance against a batch rate whose standard error is 0.0625.
   g. A measurement must be taken where the decision is made: on the rows the
      model trains or serves on, at the time the quantity is live, after any
      declared warmup, and through the same index mapping the model uses. When
      a gate reports a failure, prove the gate looked at the right rows before
      believing it about the system.
3. Entry direction comes only from the accepted model's calibrated
   `LONG/SHORT/FLAT` logits, and Exit comes only from the same bundle/shared
   encoder's calibrated `HOLD/EXIT_NOW` logits. No post-model trend, session,
   confidence, utility, threshold or close rule may veto, flip or manufacture
   either action. A separate Exit model is forbidden.
4. Keep every genuine feature family in the learned path. Removing a retired
   rule must never remove its underlying market evidence. The 378 registered
   causal-layer outputs are mandatory; only 101 additional specialist fields
   may be selected by deterministic TRAIN-only ranking.
5. The learned size head is mandatory evidence. Label-horizon sizing proof is
   diagnostic only; paper/live additionally requires candidate-bound replay
   of both heads in the same adopted bundle and fresh post-adoption broker
   runtime parity. Until both pass, emit no order. Fixed size is not a
   fallback.
6. Train equals serve: exact ordered fields, dimensions, normalization,
   timeframe construction, hashes and final-logit semantics must match.
7. Newest valid terminal evidence wins. A newer red event blocks every older
   green event. Missing or malformed evidence is red. A GREEN dataset admits
   only those exact bytes to the next evidence gate; it does not admit a model,
   direction, bundle or launch. Keep that distinction explicit in both
   Markdown and `PROJECT_STATE_xau_direction_launch.json`.
8. Every Entry rebuild needs one immutable dataset-build `--run-id`. Every
   train has a distinct output `--run-id` plus a launch-derived
   `dataset_run_id` that must match post-rebuild and all split manifests; it is
   not a caller override. IDs are lineage, not manual approval; evidence
   contracts alone admit execution. Never auto-promote an artifact.
9. Do not delete anything under `/home/andre2/GX1_DATA` or active run paths
   without an explicit verified cleanup decision. Preserve valid active
   collectors and unrelated dashboards, but never preserve or restart a
   retired canonical daemon/watchdog merely because a process or service name
   exists.
10. Remove disconnected repository code and stale docs once call-site scans,
    tests and evidence ownership show they are unnecessary.
11. Never expose secrets, force-push, hard-reset shared work or overwrite
    unrelated working-tree changes.
12. Finish every change with focused tests, syntax checks, stale-path scans,
    `git diff --check` and an honest statement of what remains unproved.
13. Source-wiring audits must prove import and executable use of the exact
    contract owner. Repeating the expected mode, dimension or field literal in
    a consumer is not ownership proof and must never be required as one.
14. Model-native training may receive decision-affecting environment values
    only from the canonical exact recipe owner. The immutable recipe must bind
    all 163 keys, split artifacts, prerequisite audits and executable source
    bytes; ambient values, wrapper defaults and hand-authored recipe evidence
    are forbidden.
15. Dataset-build and training-output identities are separate roles. Recipe,
    wrapper, trainer, bundle metadata/lock and handover must bind both; missing,
    collapsed or split-brain lineage fails closed.
16. Forward-outcome target domains are exact. Spread-aware MFE and path quality
    remain signed through validation and both train/validation losses; MAE
    remains a non-negative adverse magnitude. Clipping, taking absolute values
    or substituting parked zeros is a forbidden target rewrite.
17. Active head liveness checks must require the exact batch keys emitted by
    the canonical Dataset mapping. `y_direction` is converted once to class
    tensor `y`; adding aliases, defaults or duplicated targets to satisfy a
    head check is forbidden.
18. Input normalization is fitted once on the complete physical TRAIN
    population before sampling. Its exact ordered statistics, categorical
    domains, alias ownership and per-timeframe causal row hashes are immutable
    bundle/model state; VAL/TEST/live never refit.
19. Every 142+5 context field has exactly one specialist owner. Current-bar
    aliases are discovered from the actual ordered signal manifest, must be
    bit-identical and may not create a second normalization owner.
20. A usable bundle or immutable event may never appear under its final name
    before all bytes, hashes, strict-load checks and `fsync` complete. Publish
    only by atomic no-replace rename with exact inventory.
21. Environment text is not launch approval. Future ALLOW requires the newest
    immutable one-time approval bound to the complete launch-state hash and
    exact bundle commit, plus a runtime lease recheck before every new
    exposure. Missing broker trade identity never permits a counter-order.
22. Build on the existing script/contract owner for changes within its
    responsibility. Do not create a new version or parallel script for a minor
    edit, compatibility alias or workaround. New files require a genuinely
    new bounded authority that cannot live in the existing owner without
    mixing contracts, and must be wired through the existing public control
    surface.
23. Native canonical M1 and M5 share one OANDA-only immutable producer in the
    existing historical backfill owner. M1 uses fixed three-day chunks and M5
    fixed 15-day chunks, each capped at 4,320 theoretical slots; callers own
    neither cadence nor chunk size. The producer must retain exact response
    evidence, admit only literal complete MBA candles, stream hash-bound year
    partitions, prove source↔parquet identity and publish by atomic no-replace
    rename. Direct year-file mutation, alternate-provider repair, synthesis
    and empty success are forbidden. Production execution never implies that
    the separate native→canonical-v3/BASE28 bootstrap exists.
24. Advanced enough to trade, never more. Complexity must earn its place by
    removing a real failure mode, not by adding a mechanism that sounds
    thorough. Before writing code, prefer in this order: measure the existing
    system, change one recipe value, extend the existing owner, and only then
    add something new. When two designs fail closed equally well, keep the
    smaller one. Diagnose before building: a measurement that eliminates a
    hypothesis in ten minutes outranks a mechanism that might address it.
    Delete a diagnostic once its question is permanently settled unless it
    keeps earning its cost as observability.
25. Multi-timeframe is a causal multi-resolution pyramid, not five generic
    side inputs. All eight specialist families must have a timeframe-native
    M5/M15/H1/H4/D1 surface and an explicit learned family×timeframe route.
    The active V4 contract is exact: 111 fields per timeframe, 555
    feature×timeframe cells and 40 family×timeframe routes. `5 × 111 = 555`;
    a different count requires a versioned contract, never padding.
    Fine history is short and recent; progressively older history is represented
    at progressively coarser resolutions. Exact windows are required immutable
    recipe inputs with strictly increasing wall-clock coverage. Feature
    relevance is learned per feature×timeframe and conditioned on age/regime;
    fixed timeframe preferences, wrapper scale literals and hand-written
    confluence direction rules are forbidden. Gate values are diagnostics, not
    proof: retained routes need immutable raw/calibrated-margin ablation evidence.
26. Active cache schema v3 publishes only fully closed resample buckets and
    training proves the exact five declared decision windows at both ends of
    TRAIN/VAL/TEST. Serve-parity v11 requires independent local raw/final
    sensitivity for all 513 sequence and 513 snapshot routes, 142 continuous
    context routes and 555 MTF routes, plus valid counterfactual movement for
    five categorical routes. No launcher, dashboard, watchdog or service may
    advertise a retired daemon interface. A fresh advancing immutable
    live-tail admission is mandatory for each new Entry, but it is not a
    process-wide startup gate: an already admitted same-bundle runtime must
    still be able to manage open exposure through model-native Exit.
27. Takeover authority must describe the active boundary, not validate an old
    artifact merely because it still exists. With no admitted dataset,
    `current_smoke_launch_evidence` is null and handover may report historical
    runs only as diagnosis. Source implementation, real execution and
    admission are three separate states; never collapse them into “ready”.
28. A changed-path count is not a source identity. Takeover must bind HEAD,
    the complete tracked diff and every untracked file byte in a deterministic
    `worktree_fingerprint`. An unchanged document fingerprint cannot authorize
    continuation when that worktree identity changed; inspect the diff and
    rerun affected contracts first.

## Host-capacity hard stop

The current machine has a 43 GiB RAM envelope. Every heavy offline producer,
dataset build, audit, train, selective-edge run or replay must enter through
`scripts/gx1_capped_run.sh`. That runner is the only capacity authority and
enforces one heavy job, `MemoryMax/MemoryHigh <= 14G`, swap `<= 1G`, at least
16G host-available RAM before launch, two CPU cores and one numerical-library
thread. Any request above those limits, missing host state, lock contention or
missing cgroup is a hard failure. Never bypass, weaken, background or duplicate
a heavy job. Partial output after a cap, crash or reboot is invalid until its
immutable completion manifest and hashes pass.
