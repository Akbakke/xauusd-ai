# Transformer and TCN research protocol, 17 October 2026

The operator requested a new, bounded offline investigation. Transformer remains
GX1's main native candidate. Negative GC/M5 P0 remains NO-GO for that recipe;
it neither closes GX1 nor demonstrates that event ordering has no information.
Status and execution rights are exclusively in NEXT_RUN_POLICY.json.

## Native architecture and actual last training

`EntryV10CtxHybridTransformer` has a shared learned representation: local
feature-family sequence encoders, family-specific context/feature gates,
cross-family attention, Entry MTF M15/H1/H4/D1 encoders, family/timeframe attention and
context/token gates. Window readout is mean plus last closed token. Entry Q
uses the joint representation; unified M1 Exit adds shared feature encoders,
side/path state and recurrent episode/family/timeframe state. This is one
bundle, not a separate independent Exit model. The MTF M5 cache route is consumed
by Exit only; Entry uses its local M5 sequence. MTF widths are190; windows
M5/M15/H1/H4/D1 are16/64/96/96/252, with Entry local96 and Exit local480 bars.
The Exit detailed-path tail is512 bars and never a maximum trade duration.

The last completed native run was RECENT_365D_S20261016 on 15 October 2026,
09:48:48.814795–10:55:27.970012 UTC (11:48–12:55 Europe/Oslo): 4,071 optimizer
steps, 65,136 unique TRAIN rows, seed 20261016. Its source was
bd78c9f1ca2a7958d32f5c2b463deffc038f1ae4. It already trained independent
observed-market h20 Entry, not an Exit-derived target. h1/h5 were observations.
150 Exit state fields remained unchanged. Checkpoint/model-state/recipe and
receipt identities are verified by the new capped qualification, without
loading or editing native weights.

CONTROL4096 h20 contrast MSE changed 2,034,165.8973 to 1,889,225.7911 bps²;
rank IC 0.11735 to 0.27670. This is reused development CONTROL, not qualified
learning, a costed portfolio, an independent confirmation or profit.
RECENT_180D was armed and then disarmed for M5 work; it did not train.

Current native contract execution reports 315 signal, 132 continuous context,
one categorical context and local sequence length 96. Eight genuine families
plus D1 context remain. New GC OFI/BBO, dynamic volume levels and their time
sequences have not entered these native contracts or the last trained model.
No frozen native function is changed by this investigation. A future integration
would need a declared external research result, a new native input owner and
manifest/normalization/dataset/baseline/recipe binding through the existing gates.

## Separate D1/h20 target branch

Reuse the already built exact-M1-terminal research candidate. It preserves
entry-notional bid/ask economics and censors late terminal observations. It
still counts observed sessions: reconstructability is not calendar validity.
Historical daily/holiday regimes remain unqualified; the current NY template
cannot be projected backward as truth. Missing historical COT publication
receipts also affect the native macro inputs. Neither issue is fixed by a new
train/validation split. VAL tail reconstruction is blocked by an original raw
row group crossing sealed TEST. Native baseline/weights remain locked.

The candidate is explicitly `native_observed_sessions_exact_terminal_research_v1`,
not a qualified historical-calendar replacement. MFE/MAE remain separate path
diagnostics and are incomplete across unclassified gaps. v3 financing is a
source-bound projection, not verified historical broker cash terms. New native
training needs qualified semantics or an explicitly admitted limited research
binding, successor dataset/normalization/zero-step evidence and a native campaign.

## GC/M5 sequence hypothesis and fixed budget

Six **consecutive real closed M5 bins** (30 minutes of observed history) may
represent OFI/liquidity/price-response and volume-level interactions that a last
snapshot misses. This is a hypothesis, not an observed rescue of P0. Input order:
12 existing F fields, eight rolling15 L distances/reactions, 16 existing V fields,
one observed GC/XAU close-basis field. No indicators are reimplemented. Delta
with a known aggressor is reported as a separate stricter coverage count; absent
signed history is never invented or inserted in the primary model.

Every token must have qualified F/BBO and L masks, finite real values, exact
five-minute adjacency and the same selected dated GC contract. No rolling across
gaps, resets/invalid input prefixes or contract rolls; no forward-fill, padding
of missing market history or compression of irregular observations into sequence
positions. The TCN uses valid causal convolutions, without zero-padded history.
Source-clock/roll evidence is reused from the qualified sources, not re-audited.
These are sequences of MBP1-derived M5 metrics, not raw sub-minute event tensors.

The new recipe fixes length six, one seed, width32; Transformer has one layer,
four heads and FF64; TCN has kernel2, dilations1/2, two residual convolutions.
Both use GX1's existing mean-plus-last readout. Native encoders are unchanged.
The first chronological P0 fold is the sole proposed pilot: constant, two fixed
ridge fits (last snapshot versus six-bin sequence), saved P0 CORE/FLV LightGBM,
one Transformer and one TCN fit. Exactly two neural fits of128 steps, batch64,
no early stopping/retries/search. Existing native AdamW rate/decay/clip are reused.

All five original chronological folds receive coverage/purging counts. Only
dev_01 is proposed for training; the other folds are not automatically admitted.
Train labels close before evaluation, with the original24h embargo. Normalization
fits once on the unique physical TRAIN input rows, not repeated windows or VAL.
All model evaluation rows are identical; future label availability selects only
prediction metric rows, never actions. Saved LightGBM uses original211/247 fields,
while neural/linear models use37; this is a locked reference, not an equal-input
architecture attribution. Transformer versus TCN does use exactly the same inputs.

Prediction MSE/rank IC/calibration and paired week-block intervals are separate
from economic replay. Eight frozen contrasts include four versus FLV, three versus the snapshot
linear model and TCN versus Transformer. Prediction and economic week-block
intervals use Bonferroni adjustment within their separate eight-contrast families. Existing XAU bid/ask execution, one-position v3 ledger,
open-position marks, fee/funding scenarios, daily Sharpe/drawdown/turnover/exposure
are reused. Always-LONG and FLAT retain their comparator roles. Fixed-horizon
labels are not a native Exit or maximum hold rule. No GO for deployment can
follow from this reused development pilot. One fold cannot establish stability.

Qualification budget: capped audit4G, swap512M, one numerical thread, affinity0–7,
600s. Proposed pilot: same caps, <=1,800s total, <=600s/network. No GPU campaign.
Model construction and genuine-input forward checks are code/resource evidence,
not fitted models or predictive quality. Training remains explicitly closed.

## Sources and inference limits

The TCN implementation follows causal/dilated convolution principles, without
assuming they outperform attention on financial data:
[Bai, Kolter and Koltun](https://arxiv.org/abs/1803.01271).
The Transformer reuses the existing PyTorch encoder primitive and native readout:
[PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
Installed library versions, not mutable documentation, are frozen in the recipe.
No architecture recommendation is supported until matched trained results exist.

## Observed contract adapter failure and minimal repair

024 stopped after47.282s on contract alignment and is consumed. F and L IDs
match on all141,806 rows. The older base owner intentionally emits NULL contract
IDs on volume-invalid rows (`bind_m5_rows`); all52,972 apparent base mismatches
are those NULL rows, and zero are in the common F/L-qualified cohort. 026 checks
basis identity where the model could actually consume a row, retaining every
source mask. A nullable-ID test also proves a real mismatch within that cohort
still fails. Original failure and compact diagnosis are preserved. Diagnostic
NaN pair details were invalid repeated empty groups and were withdrawn; scalar
counts remain valid. No new horizon, model, field, target or threshold was chosen.

The calendar limitation was rechecked against primary pages in this wave:
[current OANDA XAU hours](https://www.oanda.com/bvi-en/cfds/hours-of-operation/)
state New York18:05–16:59; the separately published
[OANDA hours PDF, page4](https://www.oanda.com/assets/documents/471/Hours_of_operation_6.pdf)
uses GMT23:00–22:15 for gold Sunday–Thursday and22:00 Friday. These documents
supply no complete effective-date/holiday history for the TRAIN period. They
therefore cannot qualify a single historical calendar by themselves. This is
an evidence gap, not proof that every observed-session target is incorrect.

## Measured qualification result and remaining decisions

026 completed once, exit0,49.264s, source unchanged, TEST=false, zero fits and
optimizer steps. On141,806 native study windows,37,181 qualified single F/L/V/basis
windows yield12,482 strictly contiguous six-bin sequences, each37 fields. Every
one of26months has support. No roll/gap crossing or numeric imputation. Of
these,178 also have strict signed-aggressor history throughout; signed delta is
not silently substituted into the primary recipe. Basis spans−23.552 to289.616bps
(median45.108); this does not make futures and spot executable prices equivalent.

| Frozen chronological fold | Purged TRAIN | Matched evaluation rows |
| --- | ---: | ---: |
| dev_01, May–July2025 | 7,016 | 646 |
| dev_02, August–October2025 | 7,663 | 435 |
| dev_03, November2025–January2026 | 8,112 | 732 |
| dev_04, February–April2026 | 8,851 | 2,163 |
| dev_05, May–June2026 | 10,988 | 1,468 |

Only dev_01 is proposed for the bounded pilot. Its normalization uses13,391
unique physical TRAIN input rows. Frozen tree target/row parity is exact for
all646 rows. No model result or future horizon selected this cohort.

The small Transformer has9,985 parameters; TCN5,537. Each ran one finite CPU
forward on64 genuine normalized sequences with zero optimizer steps. Single
cold batches took2.955ms and7.879ms respectively. With n=1 these are integration
measurements, not a reliable efficiency ranking or training-throughput evidence.

Last native checkpoint/recipe hashes match the original review. The corrected
h20 file contains646,543 baseline-valid rows, of which642,214 have an exact
terminal quote;4,329 are separately censored. Every retained target is finite
and every censored target is NaN, including FLAT. This verifies the stored
research candidate, not historical calendar correctness or a newly trained model.

GO: strict GC sequence inputs for a separately admitted research pilot.
NO-GO: native h20 successor training until calendar/publication and successor
bindings are resolved. Transformer/TCN comparison, feature increment, trained
prediction metrics, strategy economics and architecture superiority are **not
measured**. P0's original NO-GO is preserved. Keep native Transformer as the
main candidate; no evidence supports replacing it now.

Proposed next executable scope is one capped CPU pilot,7,016/646 rows, two fixed
linear fits plus one128-step Transformer and one128-step TCN; at most30minutes,
4G/512M, one thread. The runner rejects pilot mode while the explicit
NEXT_RUN_POLICY pilot gate is false. Generic/native training remains false.
No data download, TEST read, production change or automatic extension is implied.

Evidence root: `/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SEQUENCE_QUALIFICATION_REPAIR_20261017_026`.
Completion review binds artifacts, original failure and96-case focused/14-case
repair XML controls (both zero failures/errors; overlapping cases are not added).
Existing20 P0 code tests also passed, and native pre-commit contract guards passed.
Syntax, stale-path scan and diff checks are separate code-consistency controls.

## Operator follow-up: all qualified market features and the 26-month study

The operator approved training/smoke after verifying the informative inputs.
Before any new model outcome, the scope was corrected: the old37-field recipe
omitted238 existing native market fields, including27 previously paused pattern
fields. New027 preserves all254 non-macro native signal fields across all eight
families, F12/L8/basis and native session category. All61 macro fields remain
excluded because historical publication is unqualified. No outcome selects which
market fields survive. Constant/sparse fields are reported, retained and not
claimed informative. Existing formula-ledger evidence is reused, with current
physical-row/domain/context-alias checks; not every native raw formula is rebuilt.

276 raw fields become283 model channels by one-hot encoding the exact native
SMC-state and session domains; missing observations are never filled. F columns
are independently checked against cached genuine M1 aggregation; L distances and
previous-level responses against stored levels/GC closes; basis against GC close
and actual XAU bid/ask mid. Cached producer prefix/volume/value-area proofs remain
in force. The added market/context fields require fresh normalization and a new
single-use qualification; no consumed qualification is rerun.

To cover the26months chronologically,028 is one fixed five-fold walk-forward
train/smoke study, with the first fold as integration smoke. Exactly four fits per
fold (two ridge, Transformer, TCN), no HPO/seed retries,128 steps per neural fit.
Batch86 = ceil(10,988 /128), derived from the largest previously qualified TRAIN
fold: every network sees every declared TRAIN row. Whole study remains capped
4G/512M/one numerical thread and30minutes. Native/GPU/h20/TEST remain closed.
Normalization is fitted once on each fold's unique physical TRAIN input rows.
Pooled economic replay carries positions across fold boundaries. Eight frozen
pooled contrasts are multiplicity-adjusted; per-fold metrics are descriptive.
This supersedes the unexecuted narrower first-fold proposal by the new explicit
user request, before model outcomes. It is not a rescue tuned on those outcomes.
