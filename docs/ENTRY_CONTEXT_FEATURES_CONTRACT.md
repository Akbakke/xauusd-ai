# Entry context contract

The model-native Entry context is mandatory model evidence, never a pre- or
post-model direction gate.

## Exact surface

- 142 ordered continuous fields;
- 5 ordered categorical fields;
- exact order and hashes bound in each split manifest and bundle;
- identical construction and normalization in train, replay and serve;
- exactly one of eight specialist-family owners per field;
- family-specific continuous projections and separate categorical
  field/domain embeddings entering the owned specialist token before
  cross-attention.

The categorical surface includes session/regime, volatility/ATR, spread and
higher-timeframe trend state according to the exact manifest-owned names. The
continuous surface carries richer session, trend, volatility, structure,
liquidity, momentum and multi-timeframe state. Source code and immutable
manifest order are authoritative; this document does not duplicate the
142-name list.

These 147 current-bar context values also condition the V4 cooperation path:
555 feature×timeframe gates, 40 family×timeframe routes and five timeframe
gates. Context may change learned relevance; it may not switch to a manual
regime policy. The separate V4 market surface is exactly 111 fields on each of
M5/M15/H1/H4/D1 and is not counted inside the 142+5 context dimensions.

## No fallback

All 147 context values must be present, finite and schema-valid. Unknown
sessions, missing ATR/spread/trend state, invalid category IDs, alternate
124/6 dimensions, zeros inserted for absent columns and median/default fills
are contract failures. Live is not allowed a softer behavior than replay.

Every field must pass strict learnability on TRAIN and exact full-scan coverage
on validation and test. A chronological OOS split may legitimately remain in
one context/regime category, but that value must be inside the TRAIN vocabulary
and may not be filled, rewritten or silently ignored. Normalization statistics
are fit once on the complete physical TRAIN population before sampling,
bundle-bound and immutable. Current-bar signal/context aliases are derived
from the actual ordered signal names, must remain bit-identical and use the
context statistic as their single owner; V24's observed count of 82 is not a
hard-coded contract.

## Direction boundary

Session, trend, volatility and spread context condition the learned model
through embeddings/FiLM/fusion. They do not decide whether a session is
tradable, veto a side, apply a confidence threshold or rewrite final logits.
The final calibrated `LONG/SHORT/FLAT` argmax remains the only authority.

Serve-parity v11 requires every continuous context field to have nonzero
sampled local raw and final class-margin sensitivity. Every categorical field
must move both surfaces under a valid next-category counterfactual. Whole
context-tensor ablations remain a separate route-level proof.
