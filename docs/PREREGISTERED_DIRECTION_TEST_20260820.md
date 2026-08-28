# Pre-registered direction test — V34 substrate (aborted; rules retained)

Written 2026-08-20, BEFORE the dataset exists. Chain V34_20260820T145741Z was
intentionally aborted before it produced an admissible dataset. The grid, nulls
and VAL→TEST rule below remain frozen; a successor chain must bind fresh complete
bytes and may not inherit partial V34 output. Nothing below may be revised after
seeing a number; a revision is a new pre-registration with its own date, and both
stay on the record.

> Current-status note, 2026-08-28: the protocol remains frozen and has not
> been run on V46. V46 has no trained bundle because the local canonical CUDA
> smoke emitted no bundle after a smoke/candidate Exit-gate mismatch in the
> bundle loader, despite passing its active movement proof. Do not revise this
> protocol or treat the bounded smoke as a direction result; see the handover
> for commit `57d4ebcb`'s one-repeat
> plan.

## Why this document exists

Direction has been refuted four times: the June information-ceiling work, the
August walk-forward (held in 1 of 5 folds, -19.48 bps utility), a GBM this week
(0 of 5 folds beat the coin flip; OOS log-loss 0.9907 vs a constant prior's
0.9475), and a horizon sweep (no horizon cleared its own floor except H=3).

**All four asked the same question: average accuracy over all bars.** That
question is nearly guaranteed to answer "no" whether or not an edge exists,
because a model that abstains on 92% of bars and has real edge on the remaining
8% is indistinguishable, in that average, from a worthless one.

## The question

Not "can the model predict direction". Instead:

> Does there exist a subset of bars, selected by the model's own decision rule,
> on which realized mean bps per trade exceeds the coin-flip null by more than
> the sampling error of that subset — and does the excess survive on VAL?

This is a curve, not a scalar: mean bps as a function of coverage (fraction of
bars traded), swept from ~100% down to ~1%.

## Null and floor — both must be measured on the SAME bytes

1. **Coin-flip null**: random side on the same tradable mask, same costs.
   The often-quoted -1.87 bps / oracle +23.84 / skill +25.71 triple is currently
   marked *reported, not re-derived* -- no artifact on disk contains it and no
   coin-flip owner exists in source. It MUST be recomputed on fresh admitted
   successor-surface bytes as
   part of this test. If it cannot be, the test does not run.
2. **Autocorrelation-preserving floor**: circular-shift of the label series,
   >=200 draws. NOT iid permutation -- the iid floor is invalid for autocorrelated
   targets and was measured 2-8x too low. The decision compares against the
   circular-shift distribution's upper tail.

## Primary decision rule, committed now

At each coverage level c in {100, 50, 25, 10, 5, 2, 1}%:
  PASS(c) iff  mean_bps(c) - mean_bps_coinflip(c) > 2 * SE(c)
           AND mean_bps(c) exceeds the 95th percentile of the circular-shift null.

**Overall PASS** iff PASS(c) holds for at least one c <= 25% on VAL, AND the same
c passes on TEST when TEST is finally opened.

**Overall FAIL** iff no c passes on VAL. A FAIL is a real result and is recorded
as such; it is not a reason to re-cut the coverage grid.

## Gates that run BEFORE the primary test

- **Seed stability.** >=5 seeds, same recipe. If the seeds do not agree
  qualitatively (no-collapse vs FLAT-drift vs side-collapse), the run is void and
  no edge claim may be made from it. A single 3-seed run has already produced all
  three behaviours from one recipe.
- **VAL power.** VAL is 2025-06-01..2026-06-30, 13 months = ~6,200 independent label
  windows, 1 sigma ~0.64pp against an effect size of ~1.6pp.
  If the fitted horizon comes out much larger than 18 M5 bars, recompute and state
  the resulting resolution before reading any result.

## Stated in advance: what this test cannot do

- It cannot separate "no edge" from "edge exists but the relationship is not
  stationary". VAL spans the 2025-2026 volatility expansion (median M5 bar range
  ~2.5 -> ~5.1 USD) by design, so a FAIL is consistent with either. Distinguishing
  them needs walk-forward, which is deliberately not in this wave.
- It is a single split. This project's own history says single-split results do
  not survive walk-forward. A PASS here is a licence to run walk-forward, not an
  edge claim.
- Nothing here measures live behaviour. train==serve is currently UNPROVEN: the
  parity gate has never executed and there is no bundle for it to examine.

## Known substrate defects at test time, recorded so they cannot be discovered later as excuses

Not blocking, but on the record: mtf_level_bars_since_break is an exact duplicate
on six lanes; the level registry breaks with no confirmation band and fires on
~19% of bars on every clock from M1 to D1 (a property of SWING_LOOKBACK=3);
volatility.squeeze_active occupies ~87% of bars; 49 columns are bitwise identical
between the local surface and the per-TF M5 lane. If the test FAILS, these are not
retroactive explanations -- they were known and judged non-blocking today.

Status correction before any result: the replacement chain now runs a full
cross-surface input audit before dataset construction. It records the 49 M5 pairs
as route-excluded physical overlap for Entry (Entry's MTF route is M15/H1/H4/D1)
and fails on any undeclared duplicate actually active on either the Entry or Exit
decision route. This adds an input-integrity gate; it does not alter the frozen
coverage grid, null distributions, or primary decision rule above.
