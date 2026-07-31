# Development notes

Updated 2026-07-31.

Use `/home/andre2/src/GX1_ENGINE/.venv/bin/python`.

Use the canonical takeover order: `AGENTS.md`,
`PIPELINE_AUDIT_XAU_20260723.md` as historical audit context, `SYSTEM_MAP.md`,
the handover, then `PROJECT_STATE_xau_direction_launch.json`. Read only the
relevant code contracts/tests afterward. Preserve unrelated dirty-worktree
changes.

## Work pattern

- Search source with `rg`/`rg --files`; exclude `.git`, `.venv` and
  `/home/andre2/GX1_DATA`.
- Measure the existing owner before adding code.
- Extend the existing contract/producer/verifier/script for a minor change.
- Do not add a versioned copy, compatibility wrapper or one-off route when the
  responsibility already has an owner.
- Create a new file only for a genuinely new bounded authority and route it
  through `scripts/entry_next_edge_control.sh`.
- Source tests prove implementation contracts, not prediction edge.

## Active MTF contract

V4 is the only admissible Entry source/model/runtime contract. No current
model, direction or launch authority exists:

- 111 fields on each of M5/M15/H1/H4/D1;
- eight non-empty specialist families on every timeframe;
- 40 family×timeframe routes;
- 555 feature×timeframe gates;
- exact V4 cache/liveness/normalization/bundle/runtime contracts;
- recipe-owned progressively coarser historical windows.

V2/V3 loaders exist for immutable historical research artifacts only. Active
Entry code must import the generic loader and require V4 identity. Do not
restore the old `GX1_V10_MULTI_TF_V2_CACHE_DIR` environment name.

The frozen measured cache is
`.../v26_6yr_rebuild_20260725_seq513_model_native_v26/MULTI_TF_V4_CACHE_20260729`.
It is schema-v2 historical input proof, not a trainable recipe or edge result.
Active schema v3 requires closed trailing buckets and therefore needs a fresh
event-local rebuild.

## Value and decision rules

- Final calibrated three-class argmax is the sole direction authority.
- A missing decision is an error, never synthetic `FLAT`.
- The same bundle/shared encoder owns calibrated `HOLD/EXIT_NOW`; a missing
  Exit decision is an error, never synthetic `HOLD`.
- No separate Exit model, bridge, overlay or hand-written close policy.
- Per-timeframe feature relevance is learned; no fixed live TF weight.
- Engineered confluence fields are evidence, not direction rules.
- Gate values are liveness diagnostics; ablation proves decision influence.
- Serve-parity v11 separately proves sampled local sensitivity for 1,723
  numeric routes and counterfactual movement for five categorical routes.
- Q/V/A audits stay on the valid `Advantage = Q - V` manifold.
- Signed MFE/path-quality stays signed; MAE stays non-negative.
- No fill, fallback, alias, clip or default may absorb invalid state.
- Window, dropout, capacity and optimization values must be explicit recipe
  inputs before they can be swept.

## Verification

For a bounded change:

```bash
.venv/bin/python -m py_compile <changed-python-files>
.venv/bin/python -m pytest -q <focused-tests>
.venv/bin/python -m pytest --collect-only -q
bash -n <changed-shell-files>
git diff --check
bash scripts/gx1_handover.sh --check
```

Also scan active code for:

- retired V2/V3 Entry cache authority;
- obsolete environment names;
- fallback/default/pass-through wording;
- mutable artifact selection;
- wrong counts (`111`, `40`, `555`);
- stale serve-parity versions or collapsed seq/snapshot influence;
- retired daemon/`--loop`/watchdog/service assumptions;
- stale instrument references.

## Heavy jobs and external data

Do not run a dataset rebuild, trainer, large replay or live launcher merely as
a test. Real jobs require immutable prerequisites, explicit event identity,
capped RAM/swap and the host-wide heavy-job lock.

The next real job must publish fresh immutable native M1/M5 authority, rebuild
the schema-v3 V4 cache and create a fresh bound combined Entry/lifecycle
dataset/audit lineage, then run a declared TRAIN/VAL capacity sweep, smoke and
candidate. The existing dataset/model/trainer owners now contain the
same-bundle causal Exit lifecycle head and positive loss, but no trained
artifact proves them. TEST remains untouched until candidate evaluation.

No current dataset or bundle artifact is admitted. The rejected V18 bundle
and stale V19/V26 dataset/audit bytes were retired and deleted; use the
retained native/canonical source and frozen V4 input cache only as inputs to a
fresh schema-v3 lineage.

The existing native owner implements schema-v4 parent-CAS successors with
verified history reuse and one bounded overlap/tail refetch. Pair publication
emits the live-tail event before pointer activation; the admission owner
requires two consecutive pair events. No new Entry may be authorized until a
fresh real admission is launch-bound and equals the pair used for inference.
Do not turn freshness into a global startup condition: same-bundle Exit
recovery must remain available when publisher evidence is stale.

Native OANDA writes require an explicit vedtak and fresh immutable output root.
Destructive `GX1_DATA` work must use the evidence-retention cleanup owner with
exact targets, immutable plan, separate approval and terminal evidence.

## Handover lifecycle

`scripts/gx1_handover.sh --check` binds all authority documents and launch
state. With no admitted dataset, `current_smoke_launch_evidence` must be null;
the takeover path does not read a historical recipe. The compact handover
reports the dirty/clean source gate, exact resume stage and public route order.
It also emits a deterministic `worktree_fingerprint` over HEAD, the complete
tracked diff and every untracked file byte; changed-path count alone is never
source identity. A future current recipe must bind the then-admitted dataset
and current source bytes exactly.

Launch remains `BLOCK`.
