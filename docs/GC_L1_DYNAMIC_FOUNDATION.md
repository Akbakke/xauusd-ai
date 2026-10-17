# GC L1 and dynamic profiles: preregistered research recipe, 17 October 2026

Scope: one free isolated OpenBB source audit (014), then one existing-data GC
feature build (015). No model, label access, native feature registration, Exit,
baseline, TEST, broker, paid endpoint or new Databento call. Original 004–013
outputs are immutable. These are input engineering gates, not economic evidence.

## L1 owner and family masks

Extend `gx1/features/gc_order_flow_v1.py` and its existing driver. Replay the
same 66 bound files once through the existing source parser. Use final-event
F_LAST snapshots, unadjusted outright prices and receive timestamps. The fixed
012 OFI/BBO candidates reject any bad BBO, gap flag, bad receive flag, snapshot,
reset, repeated T signature or file seam; they require a clean immediately prior
minute on the selected contract. Unknown aggressor alone never rejects L1.
Original trade masks and all their counters must reconcile exactly.

For previous best (B,A,Q,R) and current (b,a,q,r), OFI is
`1[b>=B]*q - 1[b<=B]*Q - 1[a<=A]*r + 1[a>=A]*R` in contracts.
Refill/depletion are positive/negative size changes only while the respective
best price is unchanged. These are L1 proxies, not identified new/cancelled
orders, hidden liquidity or L2 sweeps. Quote updates count changed final BBOs.

Integrate the last observed state on half-open receive-time intervals. BBO
imbalance is `(q-r)/(q+r)`; spread is `a-b` ticks; microprice offset from midpoint
is `(a-b)*(q-r)/(2*(q+r))` ticks. Also integrate both displayed sizes. BBO mask
additionally requires exactly 60,000,000,000 covered ns in every M1; no partial
integral is normalized as a full minute. OFI keeps its own event mask. M5 requires
all five actual coordinates and one selected contract. Mid response is the
last available midpoint before M5 close minus midpoint at its opening boundary.
No future markout enters features. Trade-record intensity keeps 004's unsigned
mask; quote count and intensity do not require signed trades. Invalid values
are missing. A held quote is an observed last state, not proof of feed delivery.

Replay the first candidate M5 per represented contract independently as a real
prefix ending at decision time; compare with the full stream. Numeric tolerance
1e-12 relative / 1e-10 absolute only permits floating summation order, not changed
masks, integer volumes or timestamps. Synthetic scalar formula, reflection,
batch-seam, duplicate-state invariance and future-mutation tests are separate.

## Real-trade dynamic profiles

Reuse hash-bound 005 ladders reconciled by 006, sum duplicate batch fragments,
and require exact per-M5 volume parity. Unsigned trades suffice; unknown side
still contributes actual volume. Build strict rolling 15/60-minute histograms
from contiguous admitted M5 bars on one contract. POC ties choose lower price.
A fixed 70% Value Area grows from POC by the greater immediately adjacent tick
volume, lower-price tie, including zero-volume intervening ticks; this is not
q25/q75. VWAP weights actual contract volume. No back adjustment or contract
mixing. This explicit algorithm is the recipe, not a claim all vendors agree.

Anchored-segment profiles reset at rejected bars, gaps, contract change or the
18:00 New York session anchor (DST-aware). Session-observed profiles accumulate
only admitted bars and publish accepted/expected minutes. They are incomplete
diagnostics unless the separate `session_complete_valid` mask proves every
minute since the anchor. The native-clock ladders omit some exchange minutes;
therefore a full GC session is NOT assumed. Neither current CME hours nor a
session anchor proves historical holiday hours. Do not use incomplete session
values in P0. Keep strict rolling/segment profiles separately usable.

GC-relative distances use GC close. Reaction fields use the previous admitted
level, never a future-confirmed pivot; they measure price relative to that
level, not absorption truth. Absolute GC prices are not XAUUSD execution levels.
Existing native ATR/RV/regimes and existing SMC remain their original owners.

## OpenBB source control

Use `openbb-core==2.0.1`, `openbb-fred==2.0.0`, `openbb-cftc==2.0.0` in an external
venv under run 014, never the native venv. Apache-2.0 package code is separate
from each data source's terms. Freeze transitive dependencies. Gold only,
futures-only legacy and disaggregated, 2024-05-01 through 2026-06-30, at most
200 reports per response, bounded child timeout. No request retries by GX1.
FRED requires an explicitly supplied existing FRED_API_KEY; never print it and
never silently fall back to a revised series for PIT training.

Existing native 61 = 4 legacy COT + 44 panel + 13 drivers. GLD fields are TONNES,
not ETF price. Broad USD exists outside these 61 in the prior H.10 study.
Managed Money is a distinct category; compare legacy OI/positions to existing
archives first. Past-52-report z uses only preceding reports (52-report warmup,
population standard deviation); every derived row remains review-only until
its actual publication and vintage are qualified.

OpenBB's FRED command exposes dates/transforms, not realtime_start/end. FRED
API defaults to today's vintage; ALFRED realtime periods are dates, not verified
intraday timestamps. The existing ALFRED audit result records failed retrievals;
its manifest language is not proof vintages exist. No PIT admission by lag alone.
CFTC provider returns report dates, not actual release receipts. Normal Friday
15:30 ET has exceptions (e.g. June 2026 schedule uses Monday June 22); a blanket
Tuesday+7-day statement is not evidence of every historical publication.

Potential additions to investigate, not new native fields: DFII5 term structure,
T5YIFR forward inflation, STLFSI4/NFCI stress, Managed Money. DGS2 is a yield with
term/risk premia, not a direct policy-rate-expectation probability. DFII10/DGS10/
DGS2 and broad USD are source checks or missing-data control, not new signals.

Official references:
- https://docs.openbb.co/odp/python/installation
- https://docs.openbb.co/odp/python/reference/cftc/cot
- https://docs.openbb.co/odp/python/reference/fred/economy/fred_series
- https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
- https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm
- https://www.cftc.gov/MarketReports/CommitmentsofTraders/DisaggregatedExplanatoryNotes/index.htm
- https://www.federalreserve.gov/releases/h10/About.HTM
- https://www.federalreserve.gov/releases/H10/Weights/
- https://arxiv.org/abs/1011.6402

## P0 admission

The next matched development test must bind the new F/L columns and their common
intersection before outcomes, original V owner, all eight factorial arms,
corrected v3 entry-notional economics, fixed risk/exposure, chronological folds,
actual-exit purge and one-day embargo. Native D1/h20 is a separate study.
Historical M5 gap/calendar classification, declared financing sensitivity,
exact rule/small-XGBoost recipe and an untouched confirmation period remain
explicit gates; feature construction alone does not authorize training.
