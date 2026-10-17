# First P0 development study — frozen before fitting

Authoritative recipe: configs/research/GC_P0_FIRST_DEVELOPMENT_20261017_021.json.
This is offline development on the existing 26 GC months. Native TEST, baseline,
Entry/Exit, production and live trading are unchanged. No raw data acquisition.

The new research control excludes exactly 61 D1 macro/positioning/panel inputs
from every arm and pauses the previously declared 27 pattern fields. CORE=211;
F=223, L=219, V=227, FL=231, FV=239, LV=235, FLV=247. F is the new twelve
book fields, L the eight strict rolling15 level/response fields, V the original
sixteen fields. Existing ATR-normalized SMC/MTF controls remain, so V is incremental
explicit volatility state, not removal of all volatility information. No duplicate
ATR implementation, GC delta surrogate or incomplete full-session profile is added.

Primary target: 30-minute future mid-log return after the 60-second execution
latency. Economic rationale: short intraday microstructure horizon, limited funding
burden, exact M1 path for all 37,181 input-qualified observations. 120/360-minute
outcomes diagnose the SAME primary forecast without extra fitting or selection.
No horizon imposes an exit or maximum holding time.

Five existing expanding chronological development folds start May/Aug/Nov 2025,
Feb/May 2026. TRAIN ends before the one-day embargo, actual label exits are purged.
One fixed LightGBM (existing 4.6.0, 64 rounds, depth4/15 leaves/min500, one thread)
per arm/fold: 40 fits, no HPO. The simple reference sums signs of core EMA trend,
OFI and rolling VWAP displacement for included layers; V conditions TRAIN-only RV
terciles. TRAIN state means predict market outcomes; unseen states fail closed.
The tree consumes every declared field. These scalar/MTF references are not a
Transformer sequence test or a replacement native architecture.

Strategy: identical input-qualified opportunities and fixed 1x initial-equity
notional, one position. Unique positive expected-net argmax chooses LONG/SHORT;
FLAT is zero. Expected cash uses +/- mid forecast less last closed XAU spread,
fees and projected horizon financing. Actual realized/marked cash uses exact
entry-side notional, executable XAU bid/ask and actual elapsed funding. Next
exact M1 open executes a change; missing quotes reject it. Positions remain
through missing feature windows, closures and fold changes until a model FLAT/flip.
No stop, time exit or hindsight-selected Exit is introduced. Final open positions
are marked and included. Full round-trip fee is reserved once at entry, including
terminal open exit; spread is already in bid/ask. Reserve treatment is conservative.

Eleven preregistered cost/funding scenarios keep the central signals and ledger
fixed. Central round trip 4bps, annual LONG569/SHORT0bps; no historical funding
truth is claimed. Report M1-observed equity drawdown, calendar-daily Sharpe with
365.25 scaling, daily/monthly net, execution costs/reserves, funding, notional
turnover, entries/closures/open marks and wall-clock exposure. Missing M1 quotes
hide intragap drawdown; bar timestamps are not verified first-tick fill times.
Paired weekly-block bootstrap reports raw95% and seven-contrast family99.2857%
intervals, 4000 replications, conditional on these fitted folds. Neither model fit
nor positive development economics establishes generalizable edge.

CFTC correction: the official 9 December 2025 revision accelerated catch-up to
29 December. The earlier November timetable is retained as historical evidence;
obsolete January catch-up dates are not current research availability. See
configs/research/CFTC_RESEARCH_RELEASE_HISTORY_V2_20261017.json and
https://www.cftc.gov/PressRoom/PressReleases/9147-25. Official schedule is intended
publication, not proof of payload delivery/vintage. The strict research selector
accepts only receipt-qualified release/vintage events, never reportdate+7 days.
No such complete historical receipt set exists here; macro remains excluded.

Measured results will be added only after the single-use terminal is complete.
