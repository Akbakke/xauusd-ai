# Entry context contract

The model-native Entry context is mandatory model evidence, never a pre- or
post-model direction gate.

## Exact surface

- 142 ordered continuous fields;
- 5 ordered categorical fields;
- exact order and hashes bound in each split manifest and bundle;
- identical construction and normalization in train, replay and serve.

The categorical surface includes session/regime, volatility/ATR, spread and
higher-timeframe trend state according to the exact manifest-owned names. The
continuous surface carries richer session, trend, volatility, structure,
liquidity, momentum and multi-timeframe state. Source code and immutable
manifest order are authoritative; this document does not duplicate the
142-name list.

## No fallback

All 147 context values must be present, finite and schema-valid. Unknown
sessions, missing ATR/spread/trend state, invalid category IDs, alternate
124/6 dimensions, zeros inserted for absent columns and median/default fills
are contract failures. Live is not allowed a softer behavior than replay.

Every field must pass full-input liveness on train, validation and test.
Normalization statistics are training-owned, bundle-bound and immutable.

## Direction boundary

Session, trend, volatility and spread context condition the learned model
through embeddings/FiLM/fusion. They do not decide whether a session is
tradable, veto a side, apply a confidence threshold or rewrite final logits.
The final calibrated `LONG/SHORT/FLAT` argmax remains the only authority.
