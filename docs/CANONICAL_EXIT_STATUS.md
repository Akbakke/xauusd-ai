# Canonical Exit status

Updated 2026-07-31.

Status: **BLOCK**. `PROJECT_STATE_artifacts.json` has no active decision
artifact. The former separate Exit implementation, contracts, registry roles
and model artifacts are deleted. They cannot be padded, renamed, restored or
treated as an incumbent.

The only admissible design is one immutable bundle and shared encoder:

- Entry emits finite calibrated logits ordered `LONG/SHORT/FLAT`;
- Exit consumes the exact frozen Entry evidence plus a hash-bound, contiguous,
  closed-M1 post-entry path envelope;
- Exit emits finite calibrated logits ordered `HOLD/EXIT_NOW`;
- both decisions use exact argmax with no threshold, overlay, fallback or
  synthetic pass-through;
- dataset, trainer, bundle, replay, serving and journal bind the same ordered
  fields and bytes.

The immutable native OANDA M1/M5 producer and the snapshot-driven
native→canonical-v3/raw-BASE28 pair producer executed on 2026-07-24 under
vedtak `XAU_NATIVE_PAIR_BOOTSTRAP_20260724_V1`: the 2019-01-01→2026-07-24
roots are accepted and pair generation `077e5419…` is frozen source evidence.
It is not a current live-tail identity. The pair route binds native/source/code/
formula/timing lineage and publishes raw BASE28 with exactly 13 native M1
fields; it cannot copy an old prebuilt.

The source implementation now provides:

- one compact causal lifecycle episode per Entry snapshot and side;
- deterministic `HOLD/EXIT_NOW` samples from a 512-bar, 14-field M1 path;
- one same-bundle Exit head consuming the Entry model's shared representation;
- a positive Exit cross-entropy term in the canonical Entry trainer;
- fail-closed export/load evidence for both classes and movement of every Exit
  component;
- a canonical full-TEST producer in the existing sizing owner that loads the
  pre-activation candidate commit directly and runs the production adapter and
  TradeState path over exact closed-M1 rows.

These are source and regression-test results, not a trained artifact. The
remaining Exit P0 sequence is:

1. bind lifecycle input to a freshly published native M1 manifest whose raw
   OANDA responses re-prove literal `complete=true`;
2. rebuild the schema-v3 V4 cache and combined Entry/lifecycle dataset;
3. smoke- and candidate-train the integrated model and prove positive loss,
   both classes and component movement from the produced artifact;
4. prove the implemented exact closed-M1 inference/output path against the
   trained candidate with train==serve evidence;
5. execute the implemented candidate-bound full-TEST producer and runtime
   parity.

Until their immutable OOS and live-like gates pass, Exit cannot authorize
paper, demo or live operation.

The existing owners implement immutable native/canonical successors,
publication events and two-consecutive-event admission. No real admission
currently turns new collector data into launch-bound pair authority. That
upstream operational P0 must close before Exit can participate in shadow or
live admission.

The unified candidate prerequisite is the complete MTF V4 contract
(111 fields on each of five timeframes, all eight families, 40
family×timeframe routes and 555 feature×timeframe gates). V26/V21C predate
that surface; the rejected V18 bundle and stale V19/V26 dataset/audit bytes
are absent. Entry and Exit must therefore train together from the first V4
smoke through the immutable candidate. Exit may not wait for an accepted Entry
and then retrain, attach to or bootstrap from its older prediction
distribution.

## What remains empirically unproven or unadmitted

- no freshly published native-manifest-bound lifecycle dataset exists;
- no current model artifact has trained `head_exit_action`;
- no completed training run proves positive Exit loss or parameter movement;
- no trained candidate proves the implemented `decide_exit` adapter;
- no candidate-bound parity event proves the implemented runtime envelope
  equals the training tensor path;
- no candidate-bound full-TEST lifecycle replay artifact exists;
- no real live-tail publication/admission event is admitted.

The target, lifecycle materializer, loader, model head, loss and replay
producer now form one source path. They remain fail-closed because no immutable
current-regime data event or trained bundle has crossed those contracts.

Live-tail launch integration is now source-complete: launch stores a static
admission/pair/root/producer anchor, while only new Entry evaluates freshness
and requires the newest admission to equal the exact pair used for inference.
The gate is repeated before a virtual open and after broker reconciliation
before a real order. It is intentionally absent from generic bundle loading
and the Exit branch so stale publisher evidence cannot disable same-bundle
`HOLD/EXIT_NOW` management of open exposure.

## Next implementation sequence

1. Publish fresh immutable native M1/M5 sources and bind lifecycle M1 to the
   exact native/pair manifest rather than a free parquet path.
2. Rebuild V4 cache under schema v3 and bind a fresh combined dataset lineage.
3. Run integrated smoke and candidate training; require both Exit classes,
   positive validation loss and movement of every Exit component.
4. Prove the implemented serving envelope and same-bundle adapter against the
   candidate: exact tensors, logits, hashes and one-bar state transition.
5. Prove movement and ablation across every required family/timeframe route for
   both Entry and Exit outputs.
6. Run the canonical full-TEST producer against the immutable candidate commit,
   then prove train==serve and zero-order broker runtime parity.
7. Execute and admit two consecutive fresh live-tail successor publications,
   and only then consider paper/demo/live launch.
