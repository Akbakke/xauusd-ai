# Risk of executing the wrong GX1 code

The dominant risk is not a syntax error. It is a plausible result produced by
the wrong artifact, feature order, contract mode, calibration, timeframe path
or downstream rule.

Current controls:

- one exact seq513 Entry schema and ordered hashes;
- explicit immutable inputs, no `latest` or glob selection;
- full 513+142+5 field liveness on every split;
- state, metadata, lock, objective, prediction and event hash binding;
- strict-load bundle audit and train==serve parity requirement;
- 20 active heads and one exact learned 23-group/75-value evidence fusion;
- calibrated final logits as the only direction authority;
- newest red/malformed terminal evidence blocks older green evidence;
- launch contract currently `BLOCK` with no accepted bundle;
- learned-calibrated sizing requires separate TEST utility/risk and
  train/replay/serve parity proof; otherwise no order is emitted;
- active Exit behavior kept separate from Entry cleanup.

Residual risks before a fresh accepted run include leakage, target mistakes,
silent feature duplication, rare-field collapse, poor class calibration,
selection overfit, insufficient pocket support, cost/slippage mismatch,
regime drift and live state divergence. These require empirical immutable OOS
and live-like evidence; unit tests cannot eliminate them.

If identity or evidence is uncertain, stop. Do not recover with a legacy model,
a hand-written rule or an assumed `FLAT` decision.
