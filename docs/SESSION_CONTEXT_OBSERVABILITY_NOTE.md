# Session context in model-native Entry

All market sessions are observed as learned context. The canonical UTC tags
are ASIA, EU, OVERLAP and US, with exact boundary semantics owned by the
feature code and immutable manifest.

Session identity, transitions, time within session and interactions with
trend, liquidity, volatility and momentum are evidence for the model. No
session is hard-coded as tradable or non-tradable after inference, and no
session allowlist may threshold the final direction logits.

Unknown or missing session state is a contract failure. It must not default to
ASIA or another category. Train, replay and serve must emit the same category
and continuous session fields for identical timestamps.
