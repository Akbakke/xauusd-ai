# Development notes

Use `/home/andre2/src/GX1_ENGINE/.venv/bin/python`.

Before editing, read `AGENTS.md`, `SYSTEM_MAP.md` and the current handover.
Prefer the existing owner over a parallel script. Preserve unrelated changes
in the dirty worktree.

Use `rg`/`rg --files` for source searches. Do not recursively walk `.venv`,
`.git` or `/home/andre2/GX1_DATA`; they are not source and dominate disk I/O.

Minimum verification for a bounded change:

```bash
.venv/bin/python -m py_compile <changed-python-files>
.venv/bin/python -m pytest -q <focused-tests>
.venv/bin/python -m pytest --collect-only -q
git diff --check
```

Also scan for deleted filenames, retired contract modes, fallback wording,
mutable artifact selection and obsolete CLI arguments in active code.

Prefer extension over proliferation. If behavior is already owned by an
existing producer, verifier, contract or control command, change that owner
and its tests. Do not add `*_vN+1.py`, a compatibility wrapper or a parallel
one-off for a small change. Create a new file only for a genuinely new,
single-purpose authority boundary, record why it is new, and expose it through
the existing public control surface.

Contract-source verification should inspect parsed imports and executable use
from the canonical owner. A raw text search for duplicated mode/dimension/field
literals is not a valid wiring check and can reject correctly centralized
consumers.

Do not run a dataset rebuild, trainer, large replay or live launcher as a test.
Entry rebuild/training require immutable prerequisites. Rebuild shares one
dataset-build `--run-id`; training uses a new output `--run-id` and a separate
launch-derived `dataset_run_id` from the exact input evidence. These are
lineage rather than manual approval. Live launch and destructive data work keep
their separate authorization contracts.

The current model-native smoke path additionally requires the immutable
162-key recipe event through `model-native-smoke-train`; direct trainer calls
and ambient decision-setting overrides are invalid. `--dry-run` is the
non-writing contract test. `--execute` is a real capped training job and its
output must immediately enter `model-native-smoke-bundle-audit`.

The terminal V7 smoke recipe declared 25,000 stratified rows, eight epochs and
patience eight. It completed six epochs, then failed hard-red with no
checkpoint or bundle. It is immutable failure evidence and must not be reused
or silently treated as a default. V24 is also rejected for rebuild after the
post-V7 audit proved signed dip-MFE clipping.

Read `PIPELINE_AUDIT_XAU_20260723.md` before any Entry data/model edit. The two
P0 target/objective faults, replacement sampler, bidirectional/conditional
auxiliary evidence, recipe/M5/MTF byte boundary, complete physical-TRAIN
normalization, 142+5 family-owned context routing, all-head/group influence,
raw-bps units, positive TF scales, grad accumulation, strict bundle commit,
atomic bundle/event publication, active-Exit byte identity, immutable approval,
runtime lease and missing-trade-ID execution path are source-repaired and
regression-tested. The current-bar alias set is derived from actual ordered
signals; V24's count of 82 is not a code constant.

Do not infer dataset/model proof: V24/V7 predate the fixes. The audited source
boundary, including the canonical transactional candidate/promotion/launch
finalizer, is code-proven only. The finalizer must continue through the
existing `entry_next_edge_control.sh` surface; it requires a pre-existing
identity-bound vedtak, serializes canonical targets, proves the bundle,
operating point and active-Exit bytes, and rolls both targets back with durable
failure evidence on any partial error. It intentionally refuses the current
caller-supplied joint Exit replay/trace diagnostics. A launch-admissible proof
still needs one canonical producer that itself executes the exact active
XGB→V3→Exit-IQL/Strategy-F chain over full TEST from hash-bound state and emits
complete per-M1 evidence. No rebuild or trainer run is admissible until that
source P0 and canonical/live tape parity for the known December-2024 defect
have exact proof. No finalizer execution is admissible until fresh empirical
artifacts satisfy every prerequisite.

Checkpoint selection must consume exact epoch-wide `specialist_gate`,
`tf_gate` and `family_tf_cooperation_gate` health. Every token must retain mean
use above 0.01 and each gate must retain its direction-neutral entropy floor.
Training-batch proxies or delayed post-bundle checks cannot substitute for
this admission gate. Gate mean/entropy proves only use distribution; the
repaired admission contract must additionally prove class-margin influence by
specialist, timeframe, family×timeframe and all 26 fusion groups on supported
VAL slices.

Target-domain changes require producer, validator, train-loss and val-loss
review together. Every spread-aware MFE, including all six dip-MFE targets,
and path quality are signed forward outcomes and must not be clamped to zero;
MAE is a non-negative adverse magnitude.
Focused tests must cover both an admitted negative signed target and rejection
of negative MAE.

Head/target changes require an emitted-batch-key audit across both train and
validation. The Dataset maps immutable parquet `y_direction` once to class
tensor `y`; head checks consume `y` directly and may not add an alias or
fallback merely to satisfy liveness.

Every heavy GX1 job must use the capped runner, explicit RAM/swap limits and
the one host-wide heavy-job lock. Never start another heavy job merely to test
a wrapper. Destructive `GX1_DATA` work must use the sole evidence-retention
cleanup owner with exact leaf targets, immutable inventory, separate approval
and execution evidence.
