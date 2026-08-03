# GX1 current system map

Updated 2026-08-03. This map describes the active XAUUSD Entry/Exit architecture.
Historical implementation chronology belongs in `DECISION_LOG.md`.

## Authority boundary

```text
closed XAU market data
        |
        v
exact causal feature state
        |
        v
one immutable learned bundle + shared encoder
        |
        +-- calibrated Entry logits [LONG, SHORT, FLAT] -> unique exact argmax
        |
        +-- frozen Entry snapshot + exact closed-M1 path
        |       -> calibrated Exit logits [HOLD, EXIT_NOW] -> unique exact argmax
```

Nothing after either output may threshold, veto, flip, recover or replace the
model decision. Missing Entry evidence is an error, not `FLAT`; missing Exit
evidence is an error, not `HOLD`. An exact top tie is also missing decision
evidence and cannot inherit array order. External decision bridges, separate Entry or
Exit policies, manual EMA/regime/SMC/confluence rules, close overlays and stale
artifacts have no authority.

## Data flow

```text
immutable native OANDA XAU M1/M5
        |
        +-- exact complete-candle/source/hash manifests
        v
canonical model-agnostic market state
        |
        +-- M5 decision availability = bar start + 5 minutes
        +-- exact last-closed-bar HTF alignment
        +-- unique ordered UTC rows
        v
full causal M5 prefix + TRAIN-only rank reference
        |
        +-- 513 signal manifest
        +-- 142 continuous context
        +-- 5 categorical context
        +-- V4 M5/M15/H1/H4/D1 cache
        v
chronological TRAIN / VAL / TEST
```

The current V8/V13 dataset binds the exact immutable source, pair manifest and
schema-v3 V4 cache with complete trailing-resample closure. The cache's
full-input liveness decision is PASS. The older V26 schema-v2 cache remains
historical launch-checkpoint evidence only. There is no on-demand feature
fallback in admitted training or serving.

Source-contract regression checks have passed, but no current full-suite count
is used as model evidence. The current offline V8/V13 source line is immutable
and does not require fresh publication for the next bounded smoke.

The live source leg is implemented in source but has not executed an admitted
real chain:

```text
OANDA collector -> immutable native M1/M5 schema-v4 successors
                   (parent-manifest CAS, bounded exact overlap + new tail)
        -> immutable canonical pair successor + publication event
           (event PASS before pointer activation)
        -> second consecutive fresh successor/publication
        -> immutable short-lived admission
        -> static launch anchor for pair/event roots and producer
        -> newest admission == exact inference pair
        -> revalidation before inference, virtual open and real order
```

The existing native/pair owners and
`gx1/contracts/live_tail_publication_v1.py` own this route; the public control
surface exposes `model-native-live-tail-pair` and
`model-native-live-tail-admission`. The retired
`canonical_incremental --loop` mode is not a publisher. No real successor or
admission has been published, so new Entry remains blocked. Process presence
is never publication health evidence.

Live-tail freshness is deliberately outside the generic bundle loader and
Exit path. The static launch anchor remains historically verifiable after its
short validity expires; only creation of new exposure requires the newest
fresh admission. A publisher outage therefore cannot synthesize `HOLD`, block
an already-open trade's same-bundle Exit decision or turn an unavailable Entry
into model `FLAT`.

## Current-bar feature stack

`gx1/contracts/entry_model_native_signal_v1.py` owns the exact signal shape:

- contract mode `xau_seq513_model_native_direction_v4`;
- 34 genuine base fields;
- 479 specialist fields;
- 513 signals total;
- 142 continuous and five categorical context fields;
- sequence length 96.

The 479 specialist signals consist of all 378 mandatory outputs from twelve
causal layers, in exact registry order, followed by 101 deterministic
TRAIN-only ranked fields. Optional ranking cannot remove a target prerequisite
or mandatory family primitive.

The eight model-native families are:

1. structure/swing;
2. SMC/liquidity;
3. trend/EMA;
4. volatility/compression;
5. momentum/flow;
6. session/regime;
7. chart geometry;
8. price-action/candles.

Path quality, utility, timing, volatility/tail, top/bottom and learned sizing
are supervised evidence in the same model. They are not separate policies.

## V4 causal multi-resolution pyramid

V4 is the first MTF contract where all eight families exist independently at
all five timeframes.

| Family | M5 | M15 | H1 | H4 | D1 | Fields/TF |
|---|---:|---:|---:|---:|---:|---:|
| structure/swing | 5 | 5 | 5 | 5 | 5 | 5 |
| SMC/liquidity | 11 | 11 | 11 | 11 | 11 | 11 |
| trend/EMA | 10 | 10 | 10 | 10 | 10 | 10 |
| volatility/compression | 2 | 2 | 2 | 2 | 2 | 2 |
| momentum/flow | 4 | 4 | 4 | 4 | 4 | 4 |
| session/regime | 5 | 5 | 5 | 5 | 5 | 5 |
| chart geometry | 10 | 10 | 10 | 10 | 10 | 10 |
| price-action/candles | 64 | 64 | 64 | 64 | 64 | 64 |
| **Total** | **111** | **111** | **111** | **111** | **111** | **111** |

Exact model surfaces:

- five timeframe tensors × 111 fields = 555 feature×timeframe cells;
- five timeframes × eight families = 40 family×timeframe routes;
- five learned timeframe weights;
- eight specialist weights;
- exact ordered token names in bundle, parity and runtime evidence.

`5 × 111 = 555`. Any different number requires a versioned contract; padding
or an unowned field is invalid.

### Causality and age

Every HTF value comes from a bar closed at or before the M5 decision timestamp.
Unavailable leading history is a declared causal warmup, not zero-filled
evidence.

Time resolution becomes coarser with age:

```text
very recent       recent intraday       multi-day       multi-week/month
    M5        ->       M15/H1       ->      H4       ->        D1
```

Exact sequence lengths are immutable recipe inputs and must produce strictly
increasing wall-clock coverage. M5 is not retained for distant history merely
because it is available. Window choice is trained/validated offline; it is not
a live regime switch. There is no global-length compatibility fallback: the
Dataset requires all five ordered positive lengths and explicit closed-bar
mode.

Training also proves the requested length for each timeframe at the first and
last decision boundary of TRAIN, VAL and TEST. Cache preflight proves only
causal warmup and closed-bar reach; it never substitutes one 96-bar length for
all five timeframes. A final partial M15/H1/H4/D1 bucket is excluded.

## Learned cooperation path

```text
per-timeframe 111-field matrix
        |
        +-- per-TF TRAIN-only normalization
        +-- categorical embeddings
        +-- 555 feature×TF gates conditioned on learned state/regime
        v
eight family-specific temporal encoders per timeframe
        |
        +-- shared family semantics across resolutions
        +-- timeframe positional encoding
        +-- M5 remains independent of current-bar specialist state
        v
5 × 8 token grid
        |
        +-- attention over timeframes within each family
        +-- attention over families within each timeframe
        +-- 40 learned family×TF cooperation gates
        +-- five learned timeframe gates
        v
MTF representation
        |
        +-- eight current-bar specialist representations
        +-- 22 supervised Entry evidence heads
        v
26 exact evidence groups / 96 values
        |
        v
LayerNorm(96) -> Linear(96,128) -> GELU -> Linear(128,3)
        |
        v
immutable calibration -> argmax LONG/SHORT/FLAT
```

Feature gates are initialized direction-neutral. Relevance is learned
independently for each feature×timeframe and may change with regime, age and
other evidence. Fixed timeframe preferences and hand-written confluence
direction weights are forbidden.

Some 513 fields intentionally encode causal cross-family hypotheses, such as
trend near support or momentum aligned with structure. They are evidence
channels only. The independent raw V4 branches run beside them, so the model
can accept, condition, downweight or reject the composite. Formula coefficients
never become live direction authority.

## Normalization and identity

All continuous statistics are fitted on the complete physical TRAIN
population before sampling. Binary and categorical semantics remain exact.
VAL, TEST, replay and serve cannot refit.

The state binds:

- exact ordered field names and hashes;
- continuous location/scale/clip origin;
- categorical domains;
- selected causal source rows;
- V4 cache path, manifest and ten component hashes;
- V4 liveness contract;
- timeframe windows and architecture values;
- alias ownership;
- model/bundle/commit bytes.

A missing or different binding fails closed.

## Targets and heads

The direction path has 22 positively supervised Entry evidence heads. They
cover:

- public direction and MTF direction;
- tradability and LONG/SHORT hierarchy;
- side validity and bad path;
- MFE, MAE, path quality and survival;
- trendline rails and timeframe agreement;
- timing, tail and volatility;
- position size;
- counterfactual Q, expectile V and exact `Advantage = Q - V`.

Spread-aware MFE and path quality remain signed. MAE is a non-negative adverse
magnitude. Silent clip, absolute-value conversion or target substitution is
invalid.

Top/bottom evidence is target-aligned:

- learned LONG adverse-turn timing must align with realized `BOTTOM`;
- learned SHORT adverse-turn timing must align with realized `TOP`.

Q/V is internal evidence only. Q argmax is not a second direction selector.
The same shared encoder also feeds one positively trained unified Exit action
head ordered `HOLD/EXIT_NOW`. It is trained in the same smoke/candidate runs,
does not become a 27th direction-fusion group and may not be attached or
retrained after Entry evidence is measured.

## Training and admission

```text
fresh V4 source/dataset audit
        |
        v
TRAIN-only normalization/ranking/window/capacity selection
        |
        v
bounded unified smoke: Entry/Exit trainability + both action-class sets
        |
        v
same-bundle Entry+Exit candidate training
        |
        v
immutable VAL/calibration and untouched TEST
        |
        v
direction + abstention + top/bottom + path/utility + lifecycle/cost evidence
        |
        v
Entry/Exit train==serve and causal ablation
```

Smoke has no launch authority. A candidate must retain LONG/SHORT/FLAT,
HOLD/EXIT_NOW, every Entry evidence head, the unified Exit head and all
cooperation paths. It must beat declared baselines without moving an
acceptance threshold after seeing TEST. Exit cannot be retrained or replaced
after this same-bundle candidate is measured.

Attention and gate values are diagnostics. The current serve-parity v11
contract requires:

- all eight specialist ablations to move raw and calibrated class margins;
- all five timeframe ablations to move both surfaces;
- all 40 exact family×timeframe route ablations to move both surfaces;
- both context tensor ablations to move both surfaces;
- all 26 evidence-group replacements to move both surfaces;
- all 555 feature gates to be finite, ordered, non-saturated and
  context-responsive;
- sampled local raw/final class-margin gradients for 513 sequence routes, 513
  snapshot routes, 142 continuous-context routes and 555 MTF cells;
- raw/final margin movement for a valid next-category counterfactual on all
  five categorical context fields.

That is 1,723 numeric routes and five categorical routes across 1,215 named
fields. It proves local reachability at sampled TEST states. The route/group
replacements remain separate ablation evidence, and neither is a substitute
for immutable OOS direction edge.

## Runtime

Admitted serving would strict-load one committed bundle and the same ordered
state used in training. No current bundle or live-tail publisher is admitted.
When available, the journal carries:

- raw and calibrated direction logits/probabilities;
- final `LONG/SHORT/FLAT` decision;
- eight specialist gates;
- five timeframe gates;
- 40 family×timeframe gates;
- 555 feature×timeframe gates;
- all required heads and learned size evidence;
- source, state, bundle and evidence identities.

Downstream execution may refuse exposure for account, sizing or broker safety.
It may not change model direction.

## Exit, sizing and adaptation

Learned size is not capital authority until an accepted unified candidate is
replayed jointly with its own Exit head and fresh broker runtime parity.

The former separate Exit artifacts and source owners are deleted. The same
bundle/shared encoder must consume a hash-bound frozen Entry snapshot and
exact contiguous closed-M1 path state, then emit finite calibrated
`HOLD/EXIT_NOW` logits. Canonical full-TEST replay must bind the candidate
commit directly and iterate every non-FLAT decision from exact T+5 fills
through actual `EXIT_NOW`; it must not depend on an already-active registry.
The existing sizing owner now implements that canonical producer as
`model-native-sizing-produce-unified-joint-proof`. Caller-supplied replay rows
remain diagnostic-only and cannot authorize launch.

Adaptation is offline only:

```text
drift evidence -> offline challenger -> identical-path replay
    -> zero-order shadow -> explicit promotion -> optional rollback
```

There are no live gradients and replay never activates a bundle directly.

## Main owners

- signal identity:
  `gx1/contracts/entry_model_native_signal_v1.py`
- mandatory causal layers:
  `gx1/features/entry_model_native_feature_layers_v1.py`
- eight-family routing:
  `gx1/features/entry_specialist_feature_groups_v1.py`
- V4 timeframe features/cache:
  `gx1/features/htf_features.py`
- SMC/geometry V4 fields:
  `gx1/features/smc_v1.py`
- model:
  `gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py`
- trainer:
  `gx1/models/entry_v10/entry_v10_ctx_train_v3.py`
- full-input liveness:
  `gx1/contracts/entry_full_input_liveness_v1.py`
- serve/runtime influence:
  `gx1/contracts/model_native_serve_gate_v1.py`
- immutable training recipe:
  `gx1/contracts/entry_model_native_train_launch_v1.py`
- single operator control surface:
  `scripts/entry_next_edge_control.sh`
- handover:
  `scripts/gx1_handover.sh`

Extend these owners for minor changes. Do not create parallel versioned
scripts to bypass an existing contract.

## Current evidence boundary

Source/input V4 architecture is implemented. The current schema-v3 cache has
full-input liveness PASS and the V8/V13 dataset is the current offline
evidence. No smoke has published a valid terminal bundle. The latest exact
V16 attempt reached its first training step and was safely killed only inside
the 10 GiB/512 MiB cgroup; no trained model is admitted. Commit `45421e70`
preserves the complete architecture and batch objective while recomputing
Transformer activations per layer during backward. Its full-size batch proof
peaked at 3,732,668 KiB RSS with no swap, and immutable V17 is the current
recipe. V26 and V21C remain older-surface measurements only. The V19/V26
dataset/audit artifacts and rejected V18 bundle were retired and deleted.

Lifecycle materialization, the same-bundle Exit head, positive loss and
component-movement/export gates are implemented in source and covered by
focused regression tests. The fresh native-manifest-bound V8/V13 lifecycle
dataset now exists and passed its audits; no trained unified bundle exists.
The serving owner now also retains the frozen 128-value Entry representation,
builds the exact source-bound closed-M1 envelope, calls the same bundle,
commits one bar transactionally, and persists/idempotently journals the
decision in TradeState v8. The candidate-bound canonical replay producer also
exists in source, but it has no output because no admissible candidate exists.
Remaining offline Exit blockers are completed same-bundle training,
candidate-bound closed-M1 train==serve proof and execution of the implemented
full-TEST producer. Live-tail, paper, broker, promotion and launch are outside
the frozen active scope.
