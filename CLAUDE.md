# GX1 operating rules

This file is the target of the `/home/andre2/CLAUDE.md` root loader, so it
governs sessions launched from the home directory as well as from this repo.
It was deleted by the documentation compaction in `d23e840b`, which left the
root loader importing a missing file. It is restored here as the single
process constitution; the rules below are the ones that compaction dropped and
that no other current document carries.

`GX1_RULES.md` is the binding project scope. `AGENTS.md` is the takeover
procedure. `SYSTEM_MAP.md` is the architecture map. `docs/DATA_CONTRACT.md` is
the data/lineage contract. `HANDOVER_XAU_DIRECTION_REPAIR_20260714.md` records
the work state, and `scripts/gx1_handover.sh` is the executable status owner
that outranks every document. If they disagree, fail closed and repair the
documentation and the code together.

The current status companion is
`docs/CURRENT_AUDIT_STATUS_20260828.md`: it makes the active audit/preflight
hold explicit, including inactive background services and the prohibition on
full training, TEST, demo and live work. Its 2026-08-30 continuation records
the exact technical checkpoint parity and VAL-journal repair; those are not
candidate, backtest or execution authority.

## Scope freeze

The only active work is offline XAUUSD work: Entry on native M5, Exit on native
M1, the same eight causal feature owners, and offline train/OOS/replay
evidence. Live, paper, demo, broker, daemon, collector, publisher, promotion,
drift and online-adaptation routes are forbidden and may not be re-opened by a
historical module that still exists on disk.

## Takeover quickstart

```bash
bash scripts/gx1_handover.sh --check
bash scripts/gx1_handover.sh
git status --short --untracked-files=all
git worktree list --porcelain
```

Do not start a second heavy job, direct-run the trainer, select an artifact by
`latest`/mtime/glob order, or touch a forbidden route. Treat any cap kill or
missing hash-bound artifact as terminal until a fresh gate proves otherwise.

## Non-negotiable rules

1. **XAUUSD only.** Entry contracts must not depend on another instrument's
   market data or expose another traded output.

2. **No fallback, guessed default, mutable `latest`, stale artifact, synthetic
   decision input or soft pass-through.** This is absolute, and it covers the
   evidence used to make a decision *about the code*, not only values inside
   it:

   a. Every decision-affecting number has exactly one of three legitimate
      origins: a named constant in a contract owner, a statistic fitted on real
      declared data, or an explicit CLI/recipe input. Anything else — including
      a value chosen because it "looked reasonable" — is forbidden. If you
      cannot name the origin in one sentence, it is a guessed default.
   b. Never invent a magnitude to make something work. When an existing value
      must change, adopt the convention the surrounding code already uses and
      say which one. Removing an exception is allowed; inventing a number is
      not.
   c. Synthetic, random, placeholder or toy-dimension data may never justify a
      conclusion about production behaviour, a code change, or a claim to the
      user. A diagnostic on such data proves only that the code runs.
      Conclusions require real declared bytes at real contract dimensions, or a
      proof from source and algebra that holds independent of data.
   d. State the evidence class for every claim: proven from source, measured on
      real data, measured on synthetic data, or unproven. Downgrade or withdraw
      a claim the moment its evidence class turns out weaker than stated, and
      record the withdrawal.
   e. A diagnostic instrument must report only what is valid where it runs. If
      a measurement cannot be taken at that point, omit the field rather than
      emitting a zero, an empty value or a placeholder that reads as a result.
   f. A threshold compared against a sampled statistic must be at least that
      statistic's sampling error at the sample size where the comparison is
      actually made, or the comparison must be taken on the complete declared
      population. A tolerance tighter than the noise of the quantity it judges
      does not measure the quantity; it trains on, or fails on, the noise.
      State the sample size and the resulting bound whenever a threshold is
      introduced or moved.
   g. A measurement must be taken where the decision is made: on the rows the
      model trains or serves on, at the time the quantity is live, after any
      declared warmup, and through the same index mapping the model uses. When
      a gate reports a failure, prove the gate looked at the right rows before
      believing it about the system.
   h. **Verify against the check's own code, never against your model of it.**
      Before launching anything expensive, replicate the gate's comparison field
      for field from its source — including path resolution, dtype and rounding.
      On 2026-08-20 a pre-flight that confirmed hash, generation id and window
      printed "ALL BINDINGS OK" and the chain then rejected the artifact on the
      two fields the pre-flight had not thought to compare; the gate also
      compares the recorded artifact *path*, resolved, so byte-identical copies
      in different locations are not interchangeable. A verification that checks
      the fields you assumed is the same defect class as the gate that measures
      the wrong rows — it returns a credible answer to a question nobody asked.

3. **One decision authority.** Entry direction comes only from the accepted
   model's unique argmax over `entry_action_q_bps` — per-action expected return
   in basis points over the valid actions, declared by
   `gx1/contracts/entry_fitted_q_v1.py` and
   `gx1/contracts/entry_model_native_training_objective_v1.py`. These are
   Q-values, **not** calibrated class probabilities, and nothing in the current
   path produces a calibrated `LONG/SHORT/FLAT` distribution; this rule said
   "calibrated logits" until 2026-08-19. Exit comes only from the same bundle
   and shared encoder's `unified_exit_action` Q-values over `HOLD/EXIT_NOW`. No
   post-model trend, session, confidence, utility, threshold or close rule may
   veto, flip or manufacture either action. A separate Exit model is forbidden.
   Exact ties and missing evidence fail closed.

4. **Keep every genuine feature family in the learned path.** Retiring a rule
   must never remove its underlying market evidence. The registered
   causal-layer outputs are mandatory; all code-owned remaining specialist
   fields are exposed to the learned model. **The authoritative
   composition is the owner tuples in
   `gx1/contracts/entry_model_native_signal_v1.py`, never a number restated
   here or in any other document** (rule 13: a repeated literal is not
   ownership, and every restated count in this repository has gone stale
   within days). The shape is: a frozen base block + the mandatory causal
   families + the complete code-owned candidate remainder, over the declared
   mandatory families. **Read the numbers by executing the owner, never from
   this file**:
   `MODEL_NATIVE_BASE_SIGNAL_DIM`, `MODEL_NATIVE_MANDATORY_SELECTED_FIELDS`,
   `MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS`, `MODEL_NATIVE_SIGNAL_DIM` and
   `MODEL_NATIVE_MANDATORY_FAMILY_FEATURES`. This rule previously restated
   "30 + 164 + 155 = 349 over 11 families" as a dated observation; on
   2026-08-15 the owner evaluated to 29 + 153 + 79 = 261 over 10 families, an
   88-field gap, and the restatement was removed rather than refreshed. A
   number written here is wrong within days by construction; that is the whole
   point of the rule, and this paragraph was the rule breaking itself.
   The direction of travel across v9–v19 was one thing repeated: every
   handwritten vote, scorebook and pre-fused composite was retired and replaced
   by the raw primitive or identified state it was built from — roughly 280
   fields in total, including all Fibonacci, all candle-pattern votes and the
   five regime composites. Removing a handwritten vote is allowed and required,
   but only while every field it consumed remains a model input, so the model
   learns the fusion instead of being handed it. The per-version detail is in
   git history and `docs/`; it is archaeology and does not belong in a file
   loaded into every session.

5. **The learned size head is mandatory evidence.** Label-horizon sizing proof
   is diagnostic only. Its target is trained only under the explicit tradable
   row mask against the exact selected-side path ECDF fitted on TRAIN; VAL/TEST
   apply that frozen ECDF. Fixed size is not a fallback, and sizing has no
   direction authority and may never create an order when direction is FLAT or
   invalid.

   The objective owner `gx1/contracts/entry_model_native_training_objective_v1.py`
   and the recipe owner `gx1/contracts/entry_model_native_train_recipe_v1.py`
   are the only authorities on what the trainer optimizes; **read their schema
   versions, keys and flags by executing them, never from this file** (rule 13).
   Proven from source 2026-08-19: the sole decision loss is masked raw-bps MSE
   on fitted-Q for both `entry_action_q` and `unified_exit_action`; task weights
   are learned (trainable homoscedastic log-variance, Kendall/Gal/Cipolla 2018),
   and the contract declares `fixed_relative_task_weights`,
   `handwritten_rank_losses`, `handwritten_composite_weights`,
   `handwritten_gate_regularization`, `fixed_target_normalization_scales` and
   `classification_or_probability_loss_authority` all False. No cross-entropy
   has decision authority; one masked `binary_cross_entropy_with_logits`
   survives on the `trendline_event` auxiliary head alone. This paragraph
   described "training-objective v6 and the 46-key recipe-v5 schema", plain
   unweighted CE on main/MTF/masked-side classification, and a pending "Wave C"
   of fixed magnitudes until 2026-08-19; none of that existed in source by then,
   and it was quoted back to the operator as authority in the same session it
   was disproven. **Not examined**: nobody has swept the trainer for every
   remaining static magnitude, so the contract's declaration is proven
   consistent, not proven complete.

   Six-clock TRAIN squeeze artifacts exist and are admissible as of 2026-08-18.
   The first set, fitted 2026-08-15, was rejected: its fit decoded globally
   (Viterbi/hard-EM) while serve decoded one step at a time, which made the
   high-volatility state absorbing on all six clocks -- M1 spent 33 of 352,193
   TRAIN rows in squeeze, emitted one release, and never returned. Fit and serve
   now share one causal forward filter, and re-estimating from the served
   sequence returns the published parameters bit-exactly. That episode is the
   standing example of rule 6 failing at the artifact level rather than in code:
   every gate was green and every contract held. Each native clock still
   requires its own exact immutable TRAIN artifact before rebuild or use;
   admitted parameters are not an admitted dataset. Exit is currently native
   closed M1; no tick-level dataset, OOS result or trading claim exists.

6. **Train equals serve.** Exact ordered fields, dimensions, normalization,
   timeframe construction, hashes and final-decision semantics must match.
   This is the requirement, not the current state. The serve-parity gate has
   still never executed — no `MODEL_NATIVE_SERVE_PARITY` event exists for a
   current bundle. The formerly divergent ATR, long-lookback context and
   float32-assembly paths are repaired in source, but source agreement is not
   empirical train==serve evidence. Treat train==serve as unproven until the
   same bound bundle emits and passes a real parity event.

7. **Newest valid terminal evidence wins.** A newer red event blocks every
   older green event. Missing or malformed evidence is red. A GREEN dataset
   admits only those exact bytes to the next evidence gate; it does not admit a
   model, a direction, a bundle or a launch. Keep that distinction explicit in
   both Markdown and `PROJECT_STATE_xau_direction_launch.json`.

8. **Run IDs are lineage, not approval.** Every dataset rebuild needs one
   immutable build `--run-id`. Every training run needs a distinct output
   `--run-id` plus a `dataset_run_id` that matches post-rebuild and all split
   manifests; it is not a caller override. Never auto-promote an artifact.

9. **Never delete under `/home/andre2/GX1_DATA` or an active run path** without
   an explicit verified cleanup decision, reachability proof and active-process
   check. This covers bytes *you* created seconds ago — a failed run's checkpoint
   directory is still evidence of what was attempted, and an `rm -rf` against it
   is the same forbidden act as deleting someone else's dataset (attempted and
   correctly blocked 2026-08-20). Orphaned scratch under `GX1_DATA` is left in
   place and reclaimed by the retention owner, never by hand. The reachability proof must cover manifest-recorded data-to-data
   references (successor parent pointers, lineage bindings), not only code
   references — a grep over the repository is not a reachability proof
   (learned 2026-08-11: retiring superseded tape generations broke the
   successor ancestor chain; repaired by the retention-attestation route in
   the provenance owner, see docs/DATA_CONTRACT.md "Retired ancestors").
   **That requirement is UNIMPLEMENTED and this rule claimed otherwise until
   2026-08-19.** What `authority_protected_paths` in
   `gx1/contracts/evidence_retention_v1.py` actually enforces, proven from
   source: it harvests absolute path strings out of three repo-root JSON files
   (`PROJECT_STATE_artifacts.json`, `PROJECT_STATE_xau_direction_launch.json`,
   `PROJECT_STATE_entry_iql_delete_incident.json`). It never opens a manifest
   under `GX1_DATA`, never follows a successor parent pointer and never
   resolves a lineage binding. Measured 2026-08-19 by executing that function:
   the protected set is **3 paths**, one of which no longer exists, guarding a
   34 GB tree. Until the owner covers data-to-data references, every
   reclamation owes a hand-built parent-pointer proof and may not lean on the
   protected set. Preserve valid active collectors and unrelated dashboards;
   never restart a retired daemon merely because its name still exists.

   a. **Reclamation is a standing duty, not an occasional errand** (operator
      directive 2026-08-13: *"husk kontinuerlig opprydding parallelt dette
      gjelder hele tiden — slett unødvendige GB"*). Every wave that supersedes
      a surface, a dataset generation or a bundle owes, in the same wave, a
      retention pass over what it just orphaned. Do not wait to be asked, do
      not wait for the disk to fill, and do not defer it to the end of a
      campaign — run it in parallel with the work that created the garbage.
      A superseded artifact that no longer has a named future use is
      unnecessary gigabytes, and its cost is measured in disk, in backup time
      and in the risk that a later session mistakes it for authority.
   b. The retention owner `gx1.scripts.cleanup_gx1_evidence_v1` is the only
      route: plan → approve → execute, hash-bound per-file inventory, no `rm`.
      An execution interrupted after its STAGED event is finished with
      `resume` (added 2026-08-13 after a foreground kill left 112 GB in
      quarantine that neither `execute` nor `recover` could complete);
      `recover` restores a staged transaction and cannot finish one.
      Retention runs re-hash every byte twice, so launch them in the
      background — a tool-timeout kill mid-delete is the exact state `resume`
      exists to repair.
   c. An artifact named as a baseline, comparison arm or authority in a
      current document is not garbage merely because its surface is retired.
      Reclaim it only after that document is changed and the operator has
      made the call explicitly. State the reclaimable size so the choice is
      informed.

10. **Remove disconnected repository code and stale docs** once call-site
    scans, tests and evidence ownership show they are unnecessary.

11. **Never expose secrets, force-push, hard-reset shared work, or overwrite
    unrelated working-tree changes.** Another agent's or the user's in-flight
    changes are to be built on, never reverted.

12. **Finish every change** with focused tests, syntax checks, stale-path
    scans, `git diff --check`, and an honest statement of what remains
    unproved.

13. **Source-wiring audits must prove import and executable use of the exact
    contract owner.** Repeating an expected mode, dimension or field literal in
    a consumer is not ownership proof and must never be accepted as one.

14. **Training may receive decision-affecting values only from the canonical
    exact recipe owner.** The immutable recipe binds its declared keys, split
    artifacts, prerequisite audits and executable source bytes. Ambient
    environment values, wrapper defaults and hand-authored recipe evidence are
    forbidden.

15. **Dataset-build and training-output identities are separate roles.**
    Recipe, wrapper, trainer, bundle metadata/lock and handover must bind both.
    Missing, collapsed or split-brain lineage fails closed.

16. **Forward-outcome target domains are exact.** Spread-aware MFE and path
    quality remain *signed* through validation and through both train and
    validation losses; MAE remains a non-negative adverse magnitude. Clipping,
    taking absolute values, or substituting parked zeros is a forbidden target
    rewrite.

17. **Active head-liveness checks must require the exact batch keys emitted by
    the canonical Dataset mapping.** `y_direction` is converted once to the
    class tensor `y`; adding aliases, defaults or duplicated targets to satisfy
    a head check is forbidden.

18. **Input normalization is fitted once on the complete physical TRAIN
    population, before sampling.** Its exact ordered statistics, categorical
    domains, alias ownership and per-timeframe causal row hashes are immutable
    bundle state. VAL, TEST and serve never refit.

19. **Every continuous and categorical context field has exactly one specialist
    owner.** Current-bar aliases are discovered from the actual ordered signal
    manifest, must be bit-identical, and may not create a second normalization
    owner.

20. **Atomic publication only.** A usable bundle or immutable event may never
    appear under its final name before all bytes, hashes, strict-load checks
    and `fsync` complete. Publish by atomic no-replace rename with an exact
    inventory.

21. **Build on the existing owner.** Do not create a new version or a parallel
    script for a minor edit, compatibility alias or workaround. A new file
    requires a genuinely new bounded authority that cannot live in the existing
    owner without mixing contracts, and it must be wired through the existing
    public control surface.

22. **Advanced enough to trade, never more.** Complexity must earn its place by
    removing a real failure mode, not by adding a mechanism that sounds
    thorough. Before writing code, prefer in this order: measure the existing
    system, change one recipe value, extend the existing owner, and only then
    add something new. When two designs fail closed equally well, keep the
    smaller one. Diagnose before building — a measurement that eliminates a
    hypothesis in ten minutes outranks a mechanism that might address it.

23. **Takeover authority describes the active boundary**, not an old artifact
    that merely still exists. Source implementation, real execution and
    admission are three separate states; never collapse them into "ready".

24. **A changed-path count is not a source identity.** Takeover binds HEAD, the
    complete tracked diff and untracked file bytes into a deterministic
    `worktree_fingerprint`. **"Untracked" means `git ls-files --others
    --exclude-standard`** (`scripts/gx1_handover.sh`), so everything the ignore
    rules exclude is invisible to that identity — this rule claimed "every
    untracked file byte" until 2026-08-19. Measured that day: `.git/info/exclude`
    — itself untracked and locally editable — carries `.claude/`, hiding
    `.claude/hooks/` (three guard scripts), `.claude/settings.reference.json`
    (three hook commands and `defaultMode: bypassPermissions`) and a 13 MB
    **registered git worktree** whose copies of `scripts/gx1_handover.sh` and
    `gx1/scripts/cleanup_gx1_evidence_v1.py` diverge from HEAD's and which still
    contains forbidden-route modules (paper runner, collector, live entry/exit).
    An unchanged document fingerprint cannot authorize continuation when that
    worktree identity changed; inspect the diff, read
    `git worktree list --porcelain`, and rerun the affected contracts first.

25. **Gate-green is not quality-green, and the operator must never have to
    dig for that distinction.** The gates prove *consistency* — train==serve,
    exact dimensions, finiteness, liveness, hash-bound lineage. They cannot
    prove *fidelity*: that a field computes the concept its name claims, that
    a constant has a defensible origin, that two owners agree on one clock,
    or that an indicator carries what a trader reading it would extract. Two
    internally consistent wrong answers pass every gate (learned 2026-08-13:
    a whole-surface review found three unrepaired noise-amplifier "slopes",
    two disagreeing session clocks, two daily clocks, 91 duplicated context
    fields and a 2-bit quantization block — none of them reachable by any
    gate, all of them found only because the operator insisted).

    a. Never answer a state question with "yes" on the strength of passing
       gates. Answer in three explicit classes: **measured** (a number, with
       the population and date), **proven consistent** (the contract holds —
       and say plainly that this does not prove quality), and **not
       examined** (nobody has looked). The third class is stated
       uninvited; silence about it is a false "yes".
    b. Run the deep review before claiming, not after being asked. A wave
       that adds or repairs a feature family owes a fidelity pass over that
       family, and any defect class found in one owner is swept across every
       other owner in the same commit wave.
    c. Record what was and was not verified in a durable document, so the
       next session inherits the map instead of the impression.

## Host-capacity hard stop

Every heavy offline producer, dataset build, audit, train, selective-edge run
or replay enters through `scripts/gx1_capped_run.sh`. That runner is the only
capacity authority: one heavy job at a time, `MemoryMax`/`MemoryHigh` at most
20G for the heavy dataset producers and the canonical trainer (raised from 10G
on 2026-08-09: real batch=640 candidate-training measurement, not a
misclassification workaround, showed a 640-row batch's pre-step host RSS
baseline alone is ~10.1G on the repaired V27 substrate, leaving no headroom
under the old ceiling even before a single training step; host has 31G total
in the WSL VM, so 20G leaves 11G for everything else), and 4G for
audits/tests, swap at most 512M, a
minimum of 20G host-available RAM before launch, CPU affinity 0-1 and one
numerical-library thread. Any request above those limits, missing host state,
lock contention or a missing cgroup is a hard failure. Never bypass, weaken,
background or duplicate a heavy job. Partial output after a cap kill, crash or
reboot is invalid until its immutable completion manifest and hashes pass.

Current hardware boundary, 2026-08-28: canonical batch-32 CUDA reaches about
8.95 GiB VRAM but two V46 smokes stopped safely at 71 C before a bundle. The
Windows-host driver rejected a physical lower power limit from WSL. The active
native guard is therefore 70 C/220 W/12 GiB: 220 W is a one-second stop, not a
throttle. The first batch-8 attempt was intentionally stopped when its
TRAIN-only 1,000-row subsample was found to leave 70,880 full VAL rows; it was
not a guard breach or crash. The repaired batch-8/32-row-per-split V46 smoke
then completed four CUDA optimizer steps and validation inside 65 C / 211.77 W
/ 8,751 MiB. Its active episode-native movement proof passed, but the bundle
loader incorrectly imposed a candidate-only Exit gate on smoke. Its next
repeat passed that repair but exposed a stale Regime-FiLM metadata requirement.
Commit `57d4ebcb` requires the retired component to be absent; `e0cf52ed` and
`64d648da` then align and statically check active-head metadata. The final
recipe-bound 32/32 smoke completed safely at 63 C / 212.37 W / 8,751 MiB and
published a diagnostic bundle. The exact evaluator then completed the immutable
70,880-row VAL artifact at 55 C / 156.03 W / 715 MiB after a vector-head
serialization repair. The final CPU smoke-bundle audit passes input, output and
lineage checks, but blocks on three specialist gates never top-ranking after
four optimizer steps. Before any further CUDA step, source tests and the
prediction-artifact schema/audit preflight must pass; only then may a bounded
learning-validation probe run behind the same guard. This is not candidate,
backtest or edge evidence. A remote machine, if
explicitly cost-approved, is still an offline bounded job with frozen
source/artifacts and an automatic stop; it is not permission to alter scope or
use broker credentials.

The current-source v10 rebind has now completed one guarded 60-step technical
segment safely (63 C / 195.53 W / 8,763 MiB) with all ten task paths live. It
remains a partial checkpoint with no bundle, VAL, TEST, edge or trading result.
