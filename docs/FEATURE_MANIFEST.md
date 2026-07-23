# Entry feature manifest contract

There is no static hand-maintained list that may authorize a model. Each fresh
dataset split carries its own exact immutable feature manifest.

## Required identity

The accepted surface is 513 ordered signals:

- 34 genuine base price-state fields owned by
  `gx1/contracts/entry_model_native_signal_v1.py`;
- exactly 479 specialist fields in the manifest-declared order: all 378
  code-owned outputs from twelve registered causal feature layers first,
  followed by exactly 101 fields selected by deterministic TRAIN-only ranking;
- no duplicate base/selected names;
- none of the seven retired XGB bridge names;
- exact field-order SHA-256 stored and revalidated by every consumer.

The mandatory 378-field registry is owned by
`gx1/features/entry_model_native_feature_layers_v1.py`. It covers trend/EMA,
the 57-field foundation cross-family surface, SMC/liquidity, structure/swing,
momentum/flow, session/regime,
volatility/compression, chart geometry, price-action/candles,
support/resistance memory, MTF confluence and exact M5 EMA50/200 state/cross
evidence. All 479 specialist fields must
then satisfy the exact eight-encoder routing contract in
`gx1/features/entry_specialist_feature_groups_v1.py`.

## Admission rules

Every signal and each of the 142+5 context fields must be finite, live and
ordered identically across train, validation, test, bundle and serve. No field
may be silently dropped, zero-filled, renamed, appended, sorted or recovered
from another manifest.

The exact ordered surface is normalized only from the complete physical TRAIN
population before sampling. Binary/categorical semantics, robust continuous
statistics, alias ownership and every selected causal MTF source row are
hash-bound into metadata, lock and persistent model state. External scaler
paths and VAL/TEST refits are forbidden.

Feature selection is training-only. Test/OOS outcomes cannot select fields or
their order. The manifest binds source data, builder revision, split range,
feature audit, liveness audit and specialist audit by immutable hashes.

The TRAIN-rank NPZ is a feature-computation prerequisite, not a later dataset
side effect. The ranker must apply those exact ECDF/ATR bytes before it derives
regime/session candidates and must embed their path, hash, source and fit
window plus both the NPZ and sidecar hashes in ranking v7. Manifest v7 reopens
and validates both artifacts. Optional ranking uses the exact spread-aware
LONG-utility minus SHORT-utility target, combining final PnL, MFE, MAE and
path-quality evidence rather than H24 mid-close return. A ranking computed
from source-provided buckets cannot pass.

`gx1/scripts/materialize_entry_model_native_seq513_signal_manifest_v1.py` is
the sole current manifest producer. It cannot pass the ranking through as an
arbitrary field list: it prepends the exact mandatory 378 and takes only 101
eligible, causal, specialist-routable names from the validated ranking.

All current-bar fields used to condition structural auxiliary labels are
owned by `entry_structural_aux_label_signal_v1.py`. Every named requirement
must resolve inside the mandatory prefix. Target construction therefore
cannot depend on whether an optional field happened to win TRAIN ranking.

The four inputs used by the pretrain support/resistance channel-polarity proof
are separately owned by `entry_pretrain_polarity_signal_v1.py` and embedded in
the same signal identity. All four must be mandatory, including the signed
support-minus-resistance stack. Missing polarity stays RED, but it cannot stop
the audit from independently reporting target liveness and consistency.

Trend/session features are required model evidence. Their presence never
authorizes a downstream hand-written filter.
