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

3. **One decision authority.** Entry direction comes only from the accepted
   model's calibrated `LONG/SHORT/FLAT` logits; Exit comes only from the same
   bundle and shared encoder's calibrated `HOLD/EXIT_NOW` logits. No post-model
   trend, session, confidence, utility, threshold or close rule may veto, flip
   or manufacture either action. A separate Exit model is forbidden. Exact ties
   and missing evidence fail closed.

4. **Keep every genuine feature family in the learned path.** Retiring a rule
   must never remove its underlying market evidence. The registered
   causal-layer outputs are mandatory; only the remaining specialist fields may
   be selected, by deterministic TRAIN-only ranking. The active composition is
   34 base + 425 mandatory causal + 133 TRAIN-ranked over 16 mandatory
   families (V29 event surface, 2026-08-11: level/trendline registries plus
   swing/momentum/regime event families joined the mandatory causal set; the
   authoritative counts derive from the owner tuples, per
   docs/V29_EVENT_SURFACE_DESIGN_20260811.md). Removing a hand-written vote is
   allowed and required — the
   `mtf_confluence` layer was removed on 2026-08-05 because it emitted derived
   confluence/abstain/direction-bias votes — but only while every field it
   consumed remains a model input, so the model learns the fusion instead of
   being handed it.

5. **The learned size head is mandatory evidence.** Label-horizon sizing proof
   is diagnostic only. Fixed size is not a fallback, and sizing may never
   create an order when direction is FLAT or invalid.

6. **Train equals serve.** Exact ordered fields, dimensions, normalization,
   timeframe construction, hashes and final-logit semantics must match.

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
   check. The reachability proof must cover manifest-recorded data-to-data
   references (successor parent pointers, lineage bindings), not only code
   references — a grep over the repository is not a reachability proof
   (learned 2026-08-11: retiring superseded tape generations broke the
   successor ancestor chain; repaired by the retention-attestation route in
   the provenance owner, see docs/DATA_CONTRACT.md "Retired ancestors").
   Preserve valid active collectors and unrelated dashboards; never
   restart a retired daemon merely because its name still exists.

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
    complete tracked diff and every untracked file byte into a deterministic
    `worktree_fingerprint`. An unchanged document fingerprint cannot authorize
    continuation when that worktree identity changed; inspect the diff and
    rerun the affected contracts first.

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
