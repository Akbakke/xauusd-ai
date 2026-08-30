# TRAIN window widening — derivation, measurements, and what stays unproved

> Runtime-state rule: this derivation owns no active candidate session. Obtain
> the verified recipe/source closure, contract and state identity from
> `bash scripts/gx1_handover.sh`, never from a checkpoint number below.

Date: 2026-08-19. Decision: widen the declared TRAIN window from
`2025-06-01 → 2026-05-31` (one year) to **`2021-06-01 → 2026-05-31`**, with
`--history-start` at the usable feature-surface start. VAL
(`2026-06-01 → 2026-06-30`) and TEST (`2026-07-01 → 2026-08-04T07:50`) are
unchanged.

> Current-status note, 2026-08-30: this is a historical derivation, not a
> command to rebuild or change V46. Resolve the exact active split from V46's
> hash-bound artifacts. The current next step is source/regression closure for
> the technical parity/journal/Exit-evidence binding repair, not a train-window decision. The
> current V46 TRAIN/VAL data remains immutable and TEST remains sealed. The
> active session is a partial checkpoint-640 candidate with fresh-process
> resume proof, not a reason to widen/rebuild the window.

Every number below is a dated observation of a *tape*, which is why it may be
written down at all (rule 13). Everything with a code owner is named, not
restated. Evidence classes are marked throughout (rule 2d).

## 1. The measured tape

**Measured 2026-08-19** under `scripts/gx1_capped_run.sh --class audit`, on the
pair-bound native M5 root `XAU_M5_NATIVE_2019_20260804_V4` and the published V31
M5 enriched surface `.../V31_CHAIN_20260819T013427Z/m5_enriched.parquet`.

| Quantity | Measured |
|---|---|
| Raw native M5 tape | `2019-01-01T23:00Z → 2026-08-04T07:50Z`, 537,861 rows |
| Usable post-warmup feature surface | `2019-11-07T21:55Z → 2026-08-04T07:50Z`, 477,229 rows |
| Closed D1 bars on the usable surface | 1,737 (`2019-11-07T22:00Z → 2026-08-02T22:00Z`) |
| First D1 open with a complete D1 receptive field inside that surface | `2020-10-29T22:00Z` |
| Closed D1 bars before `2021-01-01` / `2021-06-01` | 296 / 402 |

The `2019-11-07` usable start is confirmed. **The ceiling on `train_start` is
`2020-10-29`, i.e. 5.77 years of TRAIN, not 5.55 and not 6.6.** The 6.6-year
figure some documents implied counted the raw tape and ignored the receptive
field; the true limit is the first point at which the D1 lane of
`PRODUCTION_MTF_PER_TF_WINDOW_BARS` is complete *inside the usable surface*.

## 2. Why not before 2021 — spread regime

**Measured 2026-08-19** on the native M5 executable quotes (all 537,861 rows),
spread in bps of mid:

| year | rows | median | p90 | p99 | max |
|---|---|---|---|---|---|
| 2019 | 70,707 | 1.97 | 2.82 | 4.02 | 19.58 |
| 2020 | 71,033 | **2.86** | **12.46** | **32.27** | **50.79** |
| 2021 | 70,855 | 1.72 | 2.65 | 4.18 | 29.57 |
| 2022 | 70,881 | 1.46 | 1.79 | 2.59 | 30.25 |
| 2023 | 70,684 | 1.40 | 1.78 | 2.29 | 27.12 |
| 2024 | 71,173 | 1.61 | 1.90 | 2.70 | 24.84 |
| 2025 | 70,922 | 1.85 | 2.51 | 3.92 | 18.54 |
| 2026 | 41,606 | 1.64 | 2.41 | 4.57 | 22.45 |

2020 is a different cost regime by roughly 5× at p90 and 8× at p99; every other
year sits between 1.78 and 2.82 p90. Since direction supervision is
spread-aware executable PnL, training across that boundary would fit a cost
structure the system will never trade in. **Excluding 2020 is measured, not
stylistic.**

One correction to the figure this change was specified with: the "1.78–2.65 bps
in every other year" range omits 2019, whose p90 is 2.82.

The earliest *contiguous* anchor that excludes the 2020 regime is `2021-01-01`
(296 D1 bars of warmup, above the required receptive field). The adopted
`2021-06-01` is a round five years back from `train_end` and costs 106 D1 bars
of TRAIN relative to `2021-01-01`; it is the authorized value, and the cheaper
`2021-01-01` remains available if more TRAIN is ever wanted.

## 3. What the widening buys

**Measured 2026-08-19** on the usable surface:

| Window | closed D1 bars | M5 rows |
|---|---|---|
| Old TRAIN `2025-06-01 → 2026-05-31` | 258 | 70,668 |
| New TRAIN `2021-06-01 → 2026-05-31` | 1,290 | 354,570 |
| VAL `2026-06-01 → 2026-06-30` (unchanged) | 22 | 6,024 |
| TEST `2026-07-01 → 2026-08-04T07:50` (unchanged) | 23 | 6,671 |

**5.02× more TRAIN rows.** `--history-start` does not need to move for this
window: the usable surface already begins 402 D1 bars before `2021-06-01`. That
is a property of *this* anchor, not a general licence — see §4.

## 4. `--history-start` is now enforced, not assumed

**Implemented this wave.** `scripts/run_seq513_rebuild_chain_v1.sh`
(`model-source-identity`) previously checked only that at least 96 model-source
rows precede `--train-start` — the local M5 sequence warmup. The dominant warmup
is the D1 receptive field, and nothing checked it: a `--history-start` that
passed the 96-row test could still leave the first TRAIN rows with an incomplete
daily receptive field, silently.

The chain now additionally counts **closed D1 bars** in
`[--history-start, --train-start)` on the same axis the V4 cache uses, and fails
closed below the D1 entry of `PRODUCTION_MTF_PER_TF_WINDOW_BARS`
(`gx1/contracts/entry_exit_production_architecture_v1.py` — derived at runtime,
never restated). Counting bars rather than calendar days is the point: the row
clock skips weekends and market closures, so a day-based rule would
systematically overstate the available warmup.

No existing check was relaxed to make the new window fit.

## 5. The blocking defect this change addresses

**Reported to this wave, then re-measured here.** On the one-year window a set
of D1-derived fields is exactly constant over the fit population, and
`fit_surface_normalization` raises `[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]`, so
the trainer cannot start at all. The condition is exact and has no tolerance: a
field is unscaleable **iff every value in the fit population is identical** —
the raw IQR is zero *and* there is no positive absolute deviation from the
median. Two distinct values anywhere in the column are enough to scale it.

§7 measures this directly: **7 fields are exactly constant on the current
one-year window and 0 on the adopted five-year window.**

Separately, the `< 4 years → [ENTRY_INPUT_NORMALIZATION_BINARY_VALUE_INVALID]`
crash on `d1_ema_stack_aligned_v2` has been repaired in the normalization owner
independently of the window: the inferred-binary branch is gone and the field
takes the continuous branch, whose scale is the gap between the two observed
values and therefore strictly positive. **Do not rely on the window to fix
that**, and do not re-introduce a domain inferred from a sample.

## 6. Cost of the widening

**Proven from source.** The V29 registry constants are fitted at the head of the
M5 enriched lane from `[train_start, train_end]` and then applied to the *entire*
tape; the six-clock squeeze artifacts have the same shape. A `--train-end` or
`--train-start` change therefore does not truncate a window — it changes every
feature value at every timestamp. So the widened window invalidates and re-runs:
the six-clock squeeze fit (outside the chain, as a per-run stage 0), both
enriched feature lanes, the M5 model source, both feature-base surfaces, the
TRAIN feature ranker, and the dataset rebuild. Only `source-revision` and
`pair-authority` are window-independent.

**Measured** (7 V31 chain runs, 2026-08-18/19, event-root wall clock):
**88–104 minutes per chain**. Every one of those runs ended RED before
completing, and all ran on the *one-year* window, so this is a lower bound and a
weak one after a 5.02× TRAIN growth. The heaviest window-scaling stages are the
TRAIN feature ranker (full candidate matrix over every TRAIN row) and the
dataset rebuild's TRAIN split. Chains cannot be parallelised:
`scripts/gx1_capped_run.sh` holds a global one-job lock and pins CPU 0-1.

## 7. Measured: the seven constant D1 fields, and that the new window clears them

**Measured 2026-08-19.** Population: the D1 lane of the published V31 V4 MTF
cache (`.../V31_CHAIN_20260819T013427Z/MULTI_TF_V4_CACHE/D1_feats.npy`),
1,737 closed D1 bars, 177 ordered features taken from that cache's own
`manifest.json` (the one truth for those bytes — the repository owner has since
moved to a different width, which is exactly why the count is not restated as a
contract number here).

Exactly-constant columns, by window:

| TRAIN window | closed D1 bars | exactly-constant D1 features |
|---|---|---|
| `2025-06-01 → 2026-05-31` (current) | 258 | **7** |
| `2021-06-01 → 2026-05-31` (adopted) | 1,290 | **0** |
| `2021-01-01 → 2026-05-31` | 1,396 | **0** |

The seven, and why they were constant — measured value counts:

| field | current window (258 bars) | new window (1,290 bars) |
|---|---|---|
| `ema50_200_bull_state` | `1` × 258 | `0` × 230, `1` × 1,060 |
| `ema50_200_cross_up` | `0` × 258 | `0` × 1,286, `1` × 4 |
| `ema50_200_cross_down` | `0` × 258 | `0` × 1,286, `1` × 4 |
| `price_x_ema200_cross_up` | `0` × 258 | `0` × 1,267, `1` × 23 |
| `price_x_ema200_cross_down` | `0` × 258 | `0` × 1,267, `1` × 23 |
| `mtf_level_below_present` | `1` × 258 | `0` × 59, `1` × 1,231 |
| `mtf_level_below2_present` | `1` × 258 | `0` × 147, `1` × 1,143 |

The mechanism is now explicit rather than statistical: over the one-year window
the daily EMA50 sat above the EMA200 on **every** bar, so the state flag never
moved off `1` and both cross-event flags never left `0`; price never crossed the
daily EMA200 in either direction; and both level-presence flags were on
throughout. Five of the seven are therefore not "rare" features — they are
features whose entire information content is a regime change the window did not
contain.

**The adopted window clears all seven**, and two distinct values is all the
normalization needs: with the inferred-binary branch removed, a two-valued
column takes the continuous branch whose scale is the gap between the values and
is strictly positive. Thinnest margin: the two EMA50/200 cross fields carry
4 events in 1,290 bars. `2021-01-01` would not materially change that.

Two caveats, stated rather than assumed:

- `mtf_level_below_present` and `mtf_level_below2_present` are V29 level-registry
  outputs, and the bytes measured were produced with registry constants fitted
  on the **old** window. A rebuild at the new window refits those constants, so
  these two are proven non-constant *for these bytes*, not for the bytes the
  next build will emit. The five EMA fields are registry-independent and the
  result holds for them unconditionally.
- The normalization fit population is the Entry ∪ Exit unique physical row union
  over the TRAIN window, not the D1 bar grid. Each M5 row carries its closed D1
  bar's value, so a D1 column that is non-constant across the window's D1 bars is
  non-constant across those rows too. The implication runs in that direction
  only, which is why a constant result would be a warning and a non-constant
  result is a proof.

## 8. Not examined — stated uninvited (rule 25a)

- **Nothing has been built on the new window.** No dataset, no ranking, no
  normalization fit, no training run. Source implementation, real execution and
  admission are three separate states (rule 23).
- **The normalization fit itself has not been run on the new window.** §7
  measures constancy on the M5 enriched surface, which is a valid *disproof* of
  unscaleability but is not the fit. The only conclusive evidence is a fit that
  completes.
- **Complete-chain wall clock at the widened window is unknown**; §6 is a lower
  bound measured on RED partial runs at the old window.
- **VAL and TEST remain far too small to refute anything, and were deliberately
  left unchanged.** Measured: VAL is 22 closed D1 bars. The worst-case one-sigma
  bound on an accuracy estimate over `n` independent samples is
  `sqrt(0.25/n)`; at `n = 22` that is **10.7 pp**, and even counting
  non-overlapping label windows at the contract's maximum horizon
  (`ENTRY_DIRECTION_TARGET_POLICY_MAX_HORIZON_BARS`) gives `n ≥ 62` and
  **6.4 pp**. Both are far larger than the ~1.6 pp direction effect this project
  has previously chased. A result on this VAL can rule a large effect in or out;
  it cannot measure a small one, and must never be reported as if it could.
- **The per-row bps standard deviation is unmeasured**, so the bounds above are
  worst-case accuracy bounds only. The quantity actually decided on is mean bps
  per trade, and its standard error cannot be stated until someone measures σ on
  TRAIN. Nobody has.
- **Walk-forward is designed but deliberately not implemented.** The sequencing
  argument: if a single 5-year split shows nothing against the coin-flip null
  (−13.16 bps), walk-forward will not rescue it; if it does show something,
  walk-forward is exactly the test that then matters. The measured input for
  that later decision is in §6 — one fold costs one complete chain plus one
  training, strictly serial, with **no stage shareable between folds** because
  the registry and squeeze fits change every feature value on the whole tape.
